from __future__ import annotations

"""Model-preparation helpers for benchmark generation.

This module adds an optional *preparation* stage in front of benchmark-set
creation. The main use case is YOLO / Ultralytics models where a single exported
ONNX variant may fail as a full Hailo model although alternative export settings
(opset, simplify, end2end, embedded NMS) can succeed.

The preparation stage intentionally hides most complexity behind a compact mode:

``current``
    Use the currently selected ONNX as-is.

``screen_yolo_full_hailo``
    For Ultralytics detection models, first probe the current ONNX as a full
    Hailo build. If that fails, export a small built-in sequence of alternative
    variants and probe them one-by-one until one succeeds. The first successful
    full-Hailo variant becomes the selected model for the downstream benchmark
    workflow.

The helpers are written so GUI/Jobs/Profile-campaign code can call them without
knowing Ultralytics export details.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .evaluation_profiles import load_export_metadata_for_model
from ..dependency_bootstrap import (
    DEPENDENCY_GROUPS,
    DependencySpec,
    current_env_python_candidates,
    ensure_dependency_groups_for_python,
    missing_specs_for_python,
    ordered_python_candidates,
)

logger = logging.getLogger(__name__)

PreparationLog = Optional[Callable[[str], None]]



EXPORT_PREP_GROUP = ("export_screening",)


@dataclass(frozen=True)
class ExportVariantSpec:
    variant_id: str
    label: str
    opset: int = 18
    simplify: bool = True
    nms: bool = False
    dynamic_batch: bool = False
    batch_size: int = 1
    imgsz: int = 640
    end2end: Optional[bool] = None
    include_current_onnx: bool = False


@dataclass
class PreparationRuntimeOptions:
    hailo_backend: str = "auto"
    hw_arch: str = "hailo8"
    hef_fixup: bool = True
    hef_opt_level: int = 1
    calib_dir: Optional[str] = None
    calib_count: int = 64
    calib_batch_size: int = 8
    hef_force: bool = True
    hef_keep_artifacts: bool = True
    wsl_distro: Optional[str] = None
    wsl_venv: str = "auto"
    hef_timeout_s: int = 3600
    verify_mode: str = "ort"
    device: str = "cpu"


@dataclass
class PreparedVariantResult:
    variant_id: str
    label: str
    kind: str  # current_onnx | exported
    model_path: str
    export_json_path: Optional[str] = None
    categories_path: Optional[str] = None
    export_metadata: Dict[str, Any] = field(default_factory=dict)
    export_ok: bool = True
    export_error: Optional[str] = None
    full_hailo_ok: bool = False
    full_hailo_error: Optional[str] = None
    full_hailo_result_json: Optional[str] = None
    full_hailo_hef: Optional[str] = None
    full_hailo_endpoint_mode: Optional[str] = None
    full_hailo_end_node_names: List[str] = field(default_factory=list)
    raw_head_attempted: bool = False
    raw_head_ok: bool = False
    raw_head_error: Optional[str] = None
    raw_head_result_json: Optional[str] = None
    raw_head_hef: Optional[str] = None
    raw_head_elapsed_probe_s: Optional[float] = None
    raw_head_end_node_names: List[str] = field(default_factory=list)
    elapsed_export_s: Optional[float] = None
    elapsed_probe_s: Optional[float] = None
    tier2_attempted: bool = False
    tier2_ok: bool = False
    tier2_error: Optional[str] = None
    tier2_result_json: Optional[str] = None
    tier2_hef: Optional[str] = None
    tier2_elapsed_probe_s: Optional[float] = None
    tier2_end_node_names: List[str] = field(default_factory=list)


@dataclass
class ModelPreparationResult:
    requested_mode: str
    applied_mode: str
    source_model_path: str
    source_model_ref: Optional[str]
    screening_profile: Optional[str]
    screening_dir: Optional[str]
    success: bool
    skipped: bool
    selected_model_path: str
    selected_variant_id: Optional[str]
    selected_label: Optional[str]
    selected_export_json_path: Optional[str]
    selected_full_hailo_endpoint_mode: Optional[str] = None
    selected_full_hailo_end_node_names: List[str] = field(default_factory=list)
    selected_export_metadata: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    variants: List[PreparedVariantResult] = field(default_factory=list)
    tier2_success: bool = False
    tier2_variant_id: Optional[str] = None
    tier2_label: Optional[str] = None
    tier2_end_node_names: List[str] = field(default_factory=list)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "requested_mode": self.requested_mode,
            "applied_mode": self.applied_mode,
            "source_model_path": self.source_model_path,
            "source_model_ref": self.source_model_ref,
            "screening_profile": self.screening_profile,
            "screening_dir": self.screening_dir,
            "success": bool(self.success),
            "skipped": bool(self.skipped),
            "selected_model_path": self.selected_model_path,
            "selected_variant_id": self.selected_variant_id,
            "selected_label": self.selected_label,
            "selected_export_json_path": self.selected_export_json_path,
            "selected_full_hailo_endpoint_mode": self.selected_full_hailo_endpoint_mode,
            "selected_full_hailo_end_node_names": list(self.selected_full_hailo_end_node_names or []),
            "message": self.message,
            "variants": [asdict(v) for v in self.variants],
            "tier2_success": bool(self.tier2_success),
            "tier2_variant_id": self.tier2_variant_id,
            "tier2_label": self.tier2_label,
            "tier2_end_node_names": list(self.tier2_end_node_names or []),
        }


def normalize_model_preparation_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"", "current", "use_current", "use current onnx", "none", "off"}:
        return "current"
    if raw in {
        "screen_yolo_full_hailo",
        "screen yolo full hailo",
        "auto-screen yolo full-hailo",
        "screen_yolo",
        "yolo_full_hailo",
    }:
        return "screen_yolo_full_hailo"
    return "current"


def model_preparation_label(mode: str) -> str:
    norm = normalize_model_preparation_mode(mode)
    if norm == "screen_yolo_full_hailo":
        return "Auto-screen YOLO full-Hailo"
    return "Use current ONNX"


def preparation_result_is_selected_model(model_path: str | Path) -> bool:
    meta = load_export_metadata_for_model(model_path)
    prep = meta.get("preparation_screening") if isinstance(meta, Mapping) else None
    return bool(isinstance(prep, Mapping) and prep.get("selected"))


def load_preparation_full_hailo_end_nodes(model_path: str | Path) -> Dict[str, Any]:
    """Return full-model Hailo end-node metadata recorded by preparation.

    A YOLO detector can be selected as a Hailo *raw detection head* deployment:
    the Hailo graph ends at the six raw cv2/cv3 head Conv outputs while the
    decode/NMS tail remains outside the HEF. Downstream full-Hailo preflight and
    full-HEF generation must reuse those end nodes.
    """
    meta = load_export_metadata_for_model(model_path)
    prep = meta.get("preparation_screening") if isinstance(meta, Mapping) else None
    if not isinstance(prep, Mapping) or not bool(prep.get("selected")):
        return {"end_node_names": [], "strategy": None, "metadata": {}}
    strategy = str(
        prep.get("full_hailo_end_node_strategy")
        or prep.get("full_hailo_endpoint_mode")
        or prep.get("hailo_full_output_mode")
        or prep.get("selected_hailo_mode")
        or ""
    ).strip() or None
    raw_nodes = prep.get("full_hailo_end_node_names") or prep.get("hailo_full_end_node_names") or prep.get("end_node_names") or []
    if isinstance(raw_nodes, str):
        raw_nodes = [x.strip() for x in re.split(r"[,;\n]+", raw_nodes) if x.strip()]
    nodes: List[str] = []
    if isinstance(raw_nodes, (list, tuple)):
        for raw in raw_nodes:
            node = str(raw or "").strip()
            if node and node not in nodes:
                nodes.append(node)
    return {"end_node_names": nodes, "strategy": strategy, "metadata": dict(prep)}


def _resolve_preparation_artifact_path(value: Any, *, model_path: Path, prep_meta: Mapping[str, Any]) -> Optional[Path]:
    raw = str(value or "").strip()
    if not raw:
        return None
    candidates: List[Path] = []
    p = Path(raw).expanduser()
    candidates.append(p)
    if not p.is_absolute():
        candidates.append(model_path.parent / p)
        screening_dir = str(prep_meta.get("screening_dir") or "").strip()
        if screening_dir:
            candidates.append(Path(screening_dir).expanduser() / p)
    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except Exception:
            continue
    return p.resolve() if p.is_absolute() else None


def _load_preparation_hailo_result_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return dict(data or {}) if isinstance(data, Mapping) else {}
    except Exception:
        logger.debug("Failed to load preparation Hailo result JSON %s", path, exc_info=True)
    return {}


def _same_preparation_path(a: Any, b: Any) -> bool:
    """Best-effort path comparison that works across sidecar and summary files."""
    sa = str(a or "").strip()
    sb = str(b or "").strip()
    if not sa or not sb:
        return False
    try:
        return Path(sa).expanduser().resolve() == Path(sb).expanduser().resolve()
    except Exception:
        return os.path.normcase(os.path.abspath(os.path.expanduser(sa))) == os.path.normcase(os.path.abspath(os.path.expanduser(sb)))


def _unique_preparation_strings(values: Any) -> List[str]:
    if isinstance(values, str):
        raw_values = [x.strip() for x in re.split(r"[,;\n]+", values) if x.strip()]
    elif isinstance(values, (list, tuple)):
        raw_values = list(values)
    else:
        raw_values = []
    out: List[str] = []
    for raw in raw_values:
        val = str(raw or "").strip()
        if val and val not in out:
            out.append(val)
    return out


def _preparation_full_hailo_baseline_from_payload(prep: Mapping[str, Any], *, model_path: str | Path) -> Dict[str, Any]:
    model_p = Path(model_path).expanduser()
    if not isinstance(prep, Mapping) or not bool(prep.get("selected")):
        return {"selected": False, "ok": False, "reason": "no_preparation_selection"}

    hef_path = _resolve_preparation_artifact_path(
        prep.get("full_hailo_hef") or prep.get("raw_head_hailo_hef") or prep.get("hef_path"),
        model_path=model_p,
        prep_meta=prep,
    )
    result_json_path = _resolve_preparation_artifact_path(
        prep.get("full_hailo_result_json") or prep.get("raw_head_hailo_result_json") or prep.get("result_json_path"),
        model_path=model_p,
        prep_meta=prep,
    )
    result_json = _load_preparation_hailo_result_json(result_json_path)
    if hef_path is None and result_json.get("hef_path"):
        hef_path = _resolve_preparation_artifact_path(result_json.get("hef_path"), model_path=model_p, prep_meta=prep)

    parsed_har = _resolve_preparation_artifact_path(result_json.get("parsed_har_path"), model_path=model_p, prep_meta=prep)
    quant_har = _resolve_preparation_artifact_path(result_json.get("quant_har_path"), model_path=model_p, prep_meta=prep)
    fixed_onnx = _resolve_preparation_artifact_path(result_json.get("fixed_onnx_path"), model_path=model_p, prep_meta=prep)

    end_nodes = _unique_preparation_strings(
        prep.get("full_hailo_end_node_names")
        or prep.get("selected_full_hailo_end_node_names")
        or prep.get("hailo_full_end_node_names")
        or prep.get("end_node_names")
        or prep.get("raw_head_end_node_names")
    )
    endpoint_mode = str(
        prep.get("full_hailo_endpoint_mode")
        or prep.get("selected_full_hailo_endpoint_mode")
        or prep.get("full_hailo_end_node_strategy")
        or prep.get("hailo_full_output_mode")
        or prep.get("selected_hailo_mode")
        or ""
    ).strip() or ("custom_end_nodes" if end_nodes else "full")
    raw_head = endpoint_mode == "raw_detection_head" or str(prep.get("full_hailo_end_node_strategy") or "").strip() == "raw_detection_head"
    ok_flag = bool(prep.get("full_hailo_ok") or prep.get("raw_head_hailo_ok") or result_json.get("ok"))
    if result_json and result_json.get("ok") is False:
        ok_flag = False
    hef_exists = bool(hef_path is not None and hef_path.is_file())

    raw_contract_mode = "yolo26_one2one_raw_head" if any("one2one_cv" in str(x) for x in end_nodes) else "raw_detection_head"
    raw_contract_desc = (
        "YOLO26 one2one raw detection-head tensors; final transpose/concat/decode/postprocess remains outside the HEF."
        if raw_contract_mode == "yolo26_one2one_raw_head"
        else "YOLO raw cv2/cv3 detection-head tensors; decode/NMS remains outside the HEF."
    )

    return {
        "selected": True,
        "ok": bool(ok_flag and hef_exists),
        "reason": (None if bool(ok_flag and hef_exists) else ("missing_hef" if ok_flag else "preparation_not_ok")),
        "source": "model_preparation_screening",
        "model_path": str(model_p),
        "selected_model_path": prep.get("selected_model_path"),
        "source_model_path": prep.get("source_model_path"),
        "hef_path": (str(hef_path) if hef_path is not None else None),
        "result_json_path": (str(result_json_path) if result_json_path is not None else None),
        "parsed_har_path": (str(parsed_har) if parsed_har is not None else None),
        "quant_har_path": (str(quant_har) if quant_har is not None else None),
        "fixed_onnx_path": (str(fixed_onnx) if fixed_onnx is not None else None),
        "hw_arch": str(result_json.get("hw_arch") or prep.get("hw_arch") or "").strip() or None,
        "backend": str(result_json.get("backend") or prep.get("backend") or "").strip() or None,
        "net_name": str(result_json.get("net_name") or "").strip() or None,
        "elapsed_s": result_json.get("elapsed_s"),
        "calib_info": result_json.get("calib_info") if isinstance(result_json.get("calib_info"), Mapping) else None,
        "endpoint_mode": endpoint_mode,
        "end_node_names": list(end_nodes),
        "output_contract": {
            "mode": (raw_contract_mode if raw_head else ("custom_end_nodes" if end_nodes else "decoded_full_model")),
            "requires_external_postprocess": bool(raw_head),
            "description": (
                raw_contract_desc
                if raw_head else
                ("Custom full-model Hailo endpoint set." if end_nodes else "Decoded full-model output.")
            ),
            "end_node_names": list(end_nodes),
        },
        "true_full_model": bool(prep.get("full_hailo_true_full_model", not bool(end_nodes))),
        "variant_id": prep.get("variant_id"),
        "variant_label": prep.get("variant_label"),
        "screening_dir": prep.get("screening_dir"),
        "artifact_origin": prep.get("artifact_origin") or "sidecar",
        "metadata": dict(prep),
        "result_json": result_json,
    }


def load_preparation_full_hailo_baseline(model_path: str | Path) -> Dict[str, Any]:
    """Return a prepared full-Hailo baseline artifact for benchmark generation.

    The preparation screen may have already compiled a full-model Hailo HEF using
    custom end nodes, e.g. YOLO raw detection-head Conv endpoints. Benchmark-set
    generation should consume that artifact instead of trying the unsupported
    decoded YOLO tail again. The returned dictionary is deliberately plain JSON-
    friendly so service layers can copy it into suite manifests.
    """
    model_p = Path(model_path).expanduser()
    info = load_preparation_full_hailo_end_nodes(model_p)
    prep = info.get("metadata") if isinstance(info, Mapping) else None
    return _preparation_full_hailo_baseline_from_payload(prep if isinstance(prep, Mapping) else {}, model_path=model_p)


def _default_preparation_search_roots(source_model_path: str | Path) -> List[Path]:
    src = Path(source_model_path).expanduser()
    parent = src.parent
    roots = [
        parent / "BenchmarkSets" / "_prepared_models",
        parent / "_prepared_models",
        parent / "BenchmarkSets",
    ]
    out: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            out.append(root)
            seen.add(key)
    return out


def find_latest_preparation_full_hailo_baseline(source_model_path: str | Path, search_roots: Optional[Sequence[str | Path]] = None) -> Dict[str, Any]:
    """Find the newest reusable prepared full-Hailo HEF for this source ONNX.

    This covers the common workflow where model preparation selected a YOLO
    raw-detection-head variant, but the GUI still points benchmark generation at
    the original ONNX. Only graph-compatible prepared results are imported.
    """
    src = Path(source_model_path).expanduser()
    roots = list(search_roots or _default_preparation_search_roots(src))
    candidates: List[tuple[float, Dict[str, Any]]] = []
    seen: set[str] = set()
    for raw_root in roots:
        try:
            root = Path(raw_root).expanduser()
        except Exception:
            continue
        if not root.exists():
            continue
        for summary_path in root.rglob("screening_summary.json"):
            try:
                key = str(summary_path.resolve())
            except Exception:
                key = str(summary_path)
            if key in seen:
                continue
            seen.add(key)
            summary = _load_json_if_exists(summary_path)
            if not summary or not bool(summary.get("success")):
                continue
            summary_src = str(summary.get("source_model_path") or "").strip()
            if summary_src and not _same_preparation_path(summary_src, src):
                continue
            variants = summary.get("variants") if isinstance(summary.get("variants"), list) else []
            selected_id = str(summary.get("selected_variant_id") or "").strip()
            selected_variant: Dict[str, Any] = {}
            for raw in variants:
                if isinstance(raw, Mapping) and str(raw.get("variant_id") or "").strip() == selected_id:
                    selected_variant = dict(raw)
                    break
            if not selected_variant:
                for raw in variants:
                    if isinstance(raw, Mapping) and bool(raw.get("full_hailo_ok") or raw.get("raw_head_ok")):
                        selected_variant = dict(raw)
                        break
            if not selected_variant:
                continue
            endpoint_mode = str(
                summary.get("selected_full_hailo_endpoint_mode")
                or selected_variant.get("full_hailo_endpoint_mode")
                or ("raw_detection_head" if selected_variant.get("raw_head_ok") else "")
            ).strip()
            end_nodes = _unique_preparation_strings(
                summary.get("selected_full_hailo_end_node_names")
                or selected_variant.get("full_hailo_end_node_names")
                or selected_variant.get("raw_head_end_node_names")
            )
            is_raw_head = endpoint_mode == "raw_detection_head" or bool(selected_variant.get("raw_head_ok"))
            kind = str(selected_variant.get("kind") or "").strip()
            selected_model = str(summary.get("selected_model_path") or selected_variant.get("model_path") or "").strip()
            graph_compatible = bool(is_raw_head or kind == "current_onnx" or _same_preparation_path(selected_model, src))
            if not graph_compatible:
                continue
            prep = {
                "selected": True,
                "source_model_path": summary_src or str(src),
                "selected_model_path": selected_model or None,
                "screening_dir": str(summary_path.parent),
                "variant_id": selected_id or selected_variant.get("variant_id"),
                "variant_label": summary.get("selected_label") or selected_variant.get("label"),
                "full_hailo_ok": bool(selected_variant.get("full_hailo_ok") or selected_variant.get("raw_head_ok")),
                "raw_head_hailo_ok": bool(selected_variant.get("raw_head_ok")),
                "full_hailo_hef": selected_variant.get("full_hailo_hef") or selected_variant.get("raw_head_hef"),
                "full_hailo_result_json": selected_variant.get("full_hailo_result_json") or selected_variant.get("raw_head_result_json"),
                "full_hailo_endpoint_mode": endpoint_mode or ("raw_detection_head" if is_raw_head else None),
                "full_hailo_end_node_strategy": ("raw_detection_head" if is_raw_head else endpoint_mode or None),
                "full_hailo_end_node_names": list(end_nodes),
                "full_hailo_true_full_model": not bool(end_nodes),
                "artifact_origin": "screening_summary",
                "screening_summary_path": str(summary_path),
            }
            info = _preparation_full_hailo_baseline_from_payload(prep, model_path=summary_path)
            info["artifact_origin"] = "screening_summary"
            info["screening_summary_path"] = str(summary_path)
            info["graph_compatible_with_source"] = bool(graph_compatible)
            if not bool(info.get("ok")):
                continue
            try:
                mtime = float(summary_path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            candidates.append((mtime, info))
    if not candidates:
        return {"selected": False, "ok": False, "reason": "no_preparation_baseline_found"}
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(obj or {}), indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return dict(data or {}) if isinstance(data, Mapping) else {}
    except Exception:
        logger.debug("Failed to load JSON %s", path, exc_info=True)
    return {}


def _prepare_log(log: PreparationLog, line: str) -> None:
    if callable(log):
        try:
            log(str(line))
        except Exception:
            logger.debug("Preparation logger callback failed", exc_info=True)


def _norm_family_token(value: Any) -> str:
    s = str(value or "").strip().lower()
    for tok in ("yolo26", "yolo11", "yolo10", "yolo"):
        if tok in s:
            return tok
    return s


def _detect_yolo_family(export_meta: Mapping[str, Any], model_path: Path, entry: Optional[Mapping[str, Any]] = None) -> str:
    for raw in (
        (entry or {}).get("family") if isinstance(entry, Mapping) else None,
        export_meta.get("model_ref") if isinstance(export_meta, Mapping) else None,
        export_meta.get("model_name") if isinstance(export_meta, Mapping) else None,
        model_path.stem,
        model_path.name,
    ):
        tok = _norm_family_token(raw)
        if tok in {"yolo26", "yolo11", "yolo10", "yolo"}:
            return tok
    return ""


def _infer_ultralytics_model_ref(export_meta: Mapping[str, Any], model_path: Path, entry: Optional[Mapping[str, Any]] = None) -> str:
    raw = ""
    if isinstance(export_meta, Mapping):
        raw = str(export_meta.get("model_ref") or export_meta.get("model_name") or "").strip()
    if not raw and isinstance(entry, Mapping):
        raw = str(entry.get("model_ref") or entry.get("id") or "").strip()
    if raw:
        return raw
    stem = model_path.stem
    if stem.lower().startswith("yolo"):
        return f"{stem}.pt"
    return raw


def _builtin_screening_profile_name(family: str) -> str:
    if family == "yolo26":
        return "yolo26_full_hailo_screen_v1"
    if family == "yolo11":
        return "yolo11_full_hailo_screen_v1"
    if family == "yolo10":
        return "yolo10_full_hailo_screen_v1"
    return "yolo_full_hailo_screen_v1"


def _builtin_variant_specs(family: str) -> List[ExportVariantSpec]:
    if family == "yolo26":
        return [
            ExportVariantSpec("current", "Current ONNX", include_current_onnx=True),
            ExportVariantSpec("o18_e2e_simp", "Opset18 · simplify · native E2E", opset=18, simplify=True, nms=False, imgsz=640, end2end=None),
            ExportVariantSpec("o18_e2e_raw", "Opset18 · no simplify · native E2E", opset=18, simplify=False, nms=False, imgsz=640, end2end=None),
            ExportVariantSpec("o17_e2e_raw", "Opset17 · no simplify · native E2E", opset=17, simplify=False, nms=False, imgsz=640, end2end=None),
            ExportVariantSpec("o18_non_e2e_raw", "Opset18 · no simplify · end2end=false", opset=18, simplify=False, nms=False, imgsz=640, end2end=False),
            ExportVariantSpec("o18_non_e2e_nms", "Opset18 · no simplify · end2end=false · NMS", opset=18, simplify=False, nms=True, imgsz=640, end2end=False),
        ]
    return [
        ExportVariantSpec("current", "Current ONNX", include_current_onnx=True),
        ExportVariantSpec("o18_simp", "Opset18 · simplify", opset=18, simplify=True, nms=False, imgsz=640, end2end=None),
        ExportVariantSpec("o18_raw", "Opset18 · no simplify", opset=18, simplify=False, nms=False, imgsz=640, end2end=None),
        ExportVariantSpec("o17_raw", "Opset17 · no simplify", opset=17, simplify=False, nms=False, imgsz=640, end2end=None),
        ExportVariantSpec("o17_simp", "Opset17 · simplify", opset=17, simplify=True, nms=False, imgsz=640, end2end=None),
        ExportVariantSpec("o18_nms", "Opset18 · no simplify · NMS", opset=18, simplify=False, nms=True, imgsz=640, end2end=None),
    ]


def _family_supports_builtin_export_screening(family: str) -> bool:
    return str(family or '').strip().lower() in {"yolo11", "yolo26"}


def _resolve_export_model_ref(model_ref: str, family: str, model_path: Path) -> tuple[Optional[str], Optional[str]]:
    raw = str(model_ref or '').strip()
    if not raw:
        return None, 'no model_ref available for export screening'
    cand = Path(raw).expanduser()
    if cand.is_absolute() and cand.exists():
        return str(cand.resolve()), None
    local_adjacent = (model_path.parent / raw).expanduser()
    if local_adjacent.exists():
        return str(local_adjacent.resolve()), None
    if Path(raw).exists():
        return str(Path(raw).resolve()), None
    if _family_supports_builtin_export_screening(family):
        return raw, None
    return None, f'no local source checkpoint found for {raw!r}, and family {family!r} is not supported for built-in export screening'



def _clean_hailo_node_name_tokens(nodes: Sequence[Any]) -> List[str]:
    out: List[str] = []
    for raw in list(nodes or []):
        name = str(raw or "").strip()
        if not name:
            continue
        name = re.split(r"\s+Details were written|\s+Please verify|\s+Traceback", name, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        name = name.rstrip(".,;:]")
        if name and name not in out:
            out.append(name)
    return out


def _extract_parser_suggested_end_nodes(text: Any) -> List[str]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return []
    match = re.search(
        r"Please try to parse the model again,\s*using these end node names:\s*(.+)",
        raw_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []
    raw = str(match.group(1) or "")
    # Truncate helper/log text that may be appended after the suggested end nodes.
    raw = re.split(
        r"(?:\r?\n)+\s*(?:Details were written|\[prepare\]|__SPLITPOINT|Traceback)|\s+Details were written",
        raw,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    raw = raw.strip().strip('[](){}')
    nodes: List[str] = []
    for part in re.split(r"[,\n]+", raw):
        part_s = str(part or "").strip()
        if not part_s:
            continue
        part_s = re.split(r"\s+Details were written", part_s, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        token_match = re.search(r"(/[A-Za-z0-9_./:-]+)", part_s)
        if not token_match:
            continue
        node = str(token_match.group(1) or "").strip().rstrip(".,;")
        if node and node not in nodes:
            nodes.append(node)
    return _clean_hailo_node_name_tokens(nodes)



def _load_onnx_node_names(onnx_path: Path) -> List[str]:
    try:
        import onnx  # type: ignore
        model = onnx.load(str(onnx_path), load_external_data=False)
        out: List[str] = []
        for node in list(getattr(getattr(model, "graph", None), "node", []) or []):
            name = str(getattr(node, "name", "") or "").strip()
            if name:
                out.append(name)
        return out
    except Exception:
        logger.debug("Failed to inspect ONNX node names for %s", onnx_path, exc_info=True)
        return []


def _infer_yolo_raw_detection_head_end_nodes(onnx_path: Path, family: str = "") -> List[str]:
    """Infer Hailo-friendly raw detection-head Conv endpoints for YOLO exports.

    YOLO11/YOLO10-style Ultralytics exports usually expose six final raw head
    convolutions named ``cv2.*``/``cv3.*``.  YOLO26-style end-to-end exports use
    a related ``one2one_cv2.*``/``one2one_cv3.*`` head.  Hailo DFC can detect
    the latter as a YOLOv6-equivalent NMS structure and recommends terminating
    the accelerator graph at those one2one Conv nodes instead of at the final
    Transpose/Concat tail.

    This helper therefore returns a complete six-output raw-head endpoint set
    for either naming family.  The standard cv2/cv3 ordering is preserved for
    YOLO11 compatibility; the one2one set is returned in graph order because
    that matches the exported model topology and Hailo's own endpoint hints.
    """
    fam = str(family or "").strip().lower()
    if fam and fam not in {"yolo11", "yolo10", "yolo26", "yolo"}:
        return []
    node_names = _load_onnx_node_names(onnx_path)
    if not node_names:
        return []

    # prefix -> scale -> branch -> (final_idx, node_name, graph_index)
    GroupMap = Dict[str, Dict[int, Dict[int, tuple[int, str, int]]]]

    def _collect(*, one2one: bool) -> GroupMap:
        groups: GroupMap = {}
        head = "one2one_cv" if one2one else "cv"
        rx = re.compile(
            rf"^(?P<prefix>(?:.*?/)?model(?:/model)?\.\d+)/{head}(?P<branch>[23])\.(?P<scale>[0-2])/(?P<body>.+)/Conv$"
        )
        for graph_idx, raw_name in enumerate(node_names):
            name = str(raw_name or "")
            m = rx.match(name)
            if not m:
                continue
            try:
                branch = int(m.group("branch"))
                scale = int(m.group("scale"))
            except Exception:
                continue
            body = str(m.group("body") or "")
            suffix = [int(x) for x in re.findall(rf"{head}{branch}\.{scale}\.(\d+)(?:/|$)", body)]
            # Ultralytics normally uses .2 for the final prediction Conv. If a
            # future export changes that final index, use the highest observed
            # suffix for each branch/scale.
            score = max(suffix) if suffix else -1
            prefix = str(m.group("prefix") or "").strip()
            existing = groups.setdefault(prefix, {}).setdefault(scale, {}).get(branch)
            if (
                existing is None
                or score > existing[0]
                or (score == existing[0] and graph_idx > existing[2])
                or (score == existing[0] and graph_idx == existing[2] and len(name) > len(existing[1]))
            ):
                groups[prefix].setdefault(scale, {})[branch] = (score, str(name), int(graph_idx))
        return groups

    def _best_complete_group(groups: GroupMap) -> tuple[Optional[str], Optional[Dict[int, Dict[int, tuple[int, str, int]]]]]:
        best_prefix = None
        best_count = 0
        for prefix, by_scale in groups.items():
            count = sum(1 for scale in (0, 1, 2) for branch in (2, 3) if branch in by_scale.get(scale, {}))
            if count > best_count or (count == best_count and best_prefix is not None and prefix > best_prefix):
                best_prefix = prefix
                best_count = count
        if not best_prefix or best_count < 6:
            return None, None
        return best_prefix, groups.get(best_prefix)

    def _ordered_standard(by_scale: Dict[int, Dict[int, tuple[int, str, int]]]) -> List[str]:
        ordered: List[str] = []
        for scale in (0, 1, 2):
            for branch in (2, 3):
                item = by_scale.get(scale, {}).get(branch)
                if item is None:
                    return []
                ordered.append(item[1])
        return ordered

    def _ordered_graph(by_scale: Dict[int, Dict[int, tuple[int, str, int]]]) -> List[str]:
        items: List[tuple[int, str]] = []
        for scale in (0, 1, 2):
            for branch in (2, 3):
                item = by_scale.get(scale, {}).get(branch)
                if item is None:
                    return []
                items.append((int(item[2]), item[1]))
        return [name for _, name in sorted(items, key=lambda x: x[0])]

    standard_prefix, standard_group = _best_complete_group(_collect(one2one=False))
    one2one_prefix, one2one_group = _best_complete_group(_collect(one2one=True))

    # YOLO26 prefers the one2one end-to-end detection head.  For unknown YOLO
    # exports we also prefer one2one when it exists, because Hailo DFC reports it
    # as the post-processing-friendly endpoint set.  YOLO11/YOLO10 keep the
    # legacy cv2/cv3 raw-head set unless no standard set exists.
    if one2one_group is not None and fam not in {"yolo11", "yolo10"}:
        return _ordered_graph(one2one_group)
    if standard_group is not None:
        return _ordered_standard(standard_group)
    if one2one_group is not None:
        return _ordered_graph(one2one_group)
    return []


def infer_yolo_raw_detection_head_end_nodes(onnx_path: str | Path, family: str = "") -> List[str]:
    """Public helper used by benchmark generation for YOLO raw-head fallback."""
    return _infer_yolo_raw_detection_head_end_nodes(Path(onnx_path), family)


def _run_raw_detection_head_probe(
    *,
    base_variant_id: str,
    label: str,
    family: str,
    model_path: Path,
    runtime: PreparationRuntimeOptions,
    screening_dir: Path,
    log: PreparationLog = None,
) -> tuple[bool, List[str], Optional[str], Optional[str], Optional[str], Optional[float]]:
    nodes = _infer_yolo_raw_detection_head_end_nodes(model_path, family)
    if not nodes:
        _prepare_log(log, f"[prepare][raw-head] {base_variant_id}: no complete YOLO raw detection-head Conv endpoint set found")
        return False, [], None, None, None, None
    _prepare_log(log, f"[prepare][raw-head] {base_variant_id}: probing Hailo raw detection-head endpoints {nodes}")
    ok, err, result_json, hef_path, elapsed_probe, _probe_obj = _probe_full_hailo(
        model_path,
        f"{base_variant_id}__raw_head_endnodes",
        runtime=runtime,
        screening_dir=screening_dir,
        log=log,
        end_node_names=nodes,
    )
    if ok:
        _prepare_log(log, f"[prepare][raw-head] {base_variant_id}: raw detection-head endpoint probe succeeded")
    else:
        _prepare_log(log, f"[prepare][raw-head] {base_variant_id}: raw detection-head endpoint probe failed: {err}")
    return ok, nodes, err, result_json, hef_path, elapsed_probe


def _materialize_current_raw_head_variant(
    *,
    src_path: Path,
    exports_dir: Path,
    variant_id: str,
    export_metadata: Mapping[str, Any],
    log: PreparationLog = None,
) -> tuple[str, Optional[str], Optional[str]]:
    """Copy current ONNX into the preparation folder so endpoint metadata is local.

    The raw-head fallback changes the Hailo parse endpoint set. Keeping that
    metadata next to a prepared copy is cleaner than silently changing the
    user's original model sidecar.
    """
    exports_dir.mkdir(parents=True, exist_ok=True)
    dst = exports_dir / f"{src_path.stem}__{variant_id}.onnx"
    try:
        if str(dst.resolve()) != str(src_path.resolve()):
            shutil.copy2(src_path, dst)
    except Exception as exc:
        _prepare_log(log, f"[prepare] warning: could not copy current ONNX for raw-head selection: {type(exc).__name__}: {exc}")
        meta_path = src_path.with_suffix('.export.json')
        cat_path = src_path.with_suffix('.categories.json')
        return str(src_path), (str(meta_path) if meta_path.exists() else None), (str(cat_path) if cat_path.exists() else None)

    meta = dict(export_metadata or {})
    meta.setdefault('source', 'ultralytics')
    meta.setdefault('task_type', 'detection')
    meta.setdefault('model_name', src_path.stem)
    meta['prepared_copy_of'] = str(src_path)
    meta_path = dst.with_suffix('.export.json')
    try:
        _write_json(meta_path, meta)
    except Exception as exc:
        _prepare_log(log, f"[prepare] warning: could not write raw-head metadata sidecar: {type(exc).__name__}: {exc}")
        meta_path = None  # type: ignore[assignment]

    categories_path: Optional[str] = None
    cat_src = src_path.with_suffix('.categories.json')
    if cat_src.is_file():
        cat_dst = dst.with_suffix('.categories.json')
        try:
            shutil.copy2(cat_src, cat_dst)
            categories_path = str(cat_dst)
        except Exception:
            categories_path = str(cat_src)
    return str(dst), (str(meta_path) if meta_path is not None else None), categories_path


def _run_tier2_parser_endnode_probe(
    *,
    base_variant_id: str,
    label: str,
    model_path: Path,
    full_hailo_error: Optional[str],
    runtime: PreparationRuntimeOptions,
    screening_dir: Path,
    log: PreparationLog = None,
) -> tuple[bool, List[str], Optional[str], Optional[str], Optional[str], Optional[float]]:
    suggested = _extract_parser_suggested_end_nodes(full_hailo_error)
    if not suggested:
        return False, [], None, None, None, None
    _prepare_log(log, f"[prepare][tier2] {base_variant_id}: probing parser-suggested end nodes {suggested}")
    ok, err, result_json, hef_path, elapsed_probe, _probe_obj = _probe_full_hailo(
        model_path,
        f"{base_variant_id}__tier2_endnodes",
        runtime=runtime,
        screening_dir=screening_dir,
        log=log,
        end_node_names=suggested,
    )
    if ok:
        _prepare_log(log, f"[prepare][tier2] {base_variant_id}: parser-suggested end-node probe succeeded (diagnostic only)")
    else:
        _prepare_log(log, f"[prepare][tier2] {base_variant_id}: parser-suggested end-node probe failed: {err}")
    return ok, suggested, err, result_json, hef_path, elapsed_probe


def _export_script_path() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "export_vision_models_to_onnx.py"


def _bootstrap_export_python(log: PreparationLog = None) -> tuple[str, Optional[str]]:
    env_hint = str(os.environ.get("OSP_EXPORT_PYTHON") or "").strip()
    ordered = current_env_python_candidates()
    if env_hint:
        ordered = ordered_python_candidates(env_hint, *ordered)
    if not ordered:
        ordered = [str(sys.executable)]

    # Prefer an interpreter that already has the export-screening dependencies.
    for exe in ordered:
        missing = missing_specs_for_python(exe, DEPENDENCY_GROUPS['export_screening'])
        if not missing:
            if exe != sys.executable:
                _prepare_log(log, f"[prepare] using export Python with required deps: {exe}")
            return exe, None

    # Bootstrap the active interpreter first. This preserves the venv path instead
    # of collapsing it to the system-python symlink target.
    primary = ordered[0]
    missing_primary = missing_specs_for_python(primary, DEPENDENCY_GROUPS['export_screening'])
    if missing_primary:
        pkgs = []
        for spec in missing_primary:
            if spec.package not in pkgs:
                pkgs.append(spec.package)
        _prepare_log(log, f"[prepare] export environment missing packages in {primary}: {', '.join(pkgs)}")
        ok, remaining = ensure_dependency_groups_for_python(primary, EXPORT_PREP_GROUP, log=lambda line: _prepare_log(log, line))
        if ok:
            _prepare_log(log, f"[prepare] export dependencies ready in {primary}")
            return primary, None
        if remaining:
            remaining_pkgs = ', '.join(sorted({spec.package for spec in remaining}))
            _prepare_log(log, f"[prepare] bootstrap in {primary} incomplete; still missing: {remaining_pkgs}")

    # Fall back to any other interpreter that already satisfies the requirements.
    for exe in ordered[1:]:
        missing = missing_specs_for_python(exe, DEPENDENCY_GROUPS['export_screening'])
        if not missing:
            _prepare_log(log, f"[prepare] falling back to alternate export Python: {exe}")
            return exe, None

    return primary, "export dependency bootstrap failed; no interpreter with the required export packages is available"


def _summarize_export_failure(stderr: str, stdout: str, returncode: int) -> str:
    text = (stderr or "") + "\n" + (stdout or "")
    text = text.strip()
    for line in text.splitlines():
        line = line.strip()
        if "ModuleNotFoundError:" in line and "No module named" in line:
            dep = line.split("No module named", 1)[-1].strip().strip("\'\"")
            return f"export environment missing dependency: {dep}"
    if text:
        tail = text.splitlines()[-1].strip()
        if tail:
            return f"export failed with code {returncode}: {tail}"
    return f"export failed with code {returncode}"


def _run_export_script(
    model_ref: str,
    spec: ExportVariantSpec,
    output_path: Path,
    *,
    runtime: PreparationRuntimeOptions,
    log: PreparationLog = None,
) -> tuple[bool, Optional[str], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[float]]:
    script_path = _export_script_path()
    if not script_path.is_file():
        return False, f"Export script not found: {script_path}", None, None, None, None
    export_python, bootstrap_err = _bootstrap_export_python(log)
    if bootstrap_err:
        return False, bootstrap_err, None, None, None, 0.0
    cmd = [
        export_python,
        str(script_path),
        "--source",
        "ultralytics",
        "--model",
        str(model_ref),
        "--output",
        str(output_path),
        "--opset",
        str(int(spec.opset)),
        "--device",
        str(runtime.device or "cpu"),
        "--batch-size",
        str(int(spec.batch_size or 1)),
        "--verify",
        str(runtime.verify_mode or "ort"),
        "--imgsz",
        str(int(spec.imgsz or 640)),
    ]
    if spec.dynamic_batch:
        cmd.append("--dynamic-batch")
    if not spec.simplify:
        cmd.append("--no-simplify")
    if spec.nms:
        cmd.append("--nms")
    if spec.end2end is not None:
        cmd.extend(["--end2end", "true" if spec.end2end else "false"])
    _prepare_log(log, f"[prepare][export] {spec.label}: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.time() - t0
    if proc.stdout.strip():
        for line in proc.stdout.splitlines():
            if line.strip():
                _prepare_log(log, f"[prepare][export][stdout] {line}")
    if proc.stderr.strip():
        for line in proc.stderr.splitlines():
            if line.strip():
                _prepare_log(log, f"[prepare][export][stderr] {line}")
    if proc.returncode != 0:
        return False, _summarize_export_failure(proc.stderr, proc.stdout, proc.returncode), None, None, None, elapsed
    meta_path = output_path.with_suffix(".export.json")
    categories_path = output_path.with_suffix(".categories.json")
    meta = _load_json_if_exists(meta_path)
    if not output_path.is_file():
        return False, f"export did not produce ONNX file: {output_path}", meta, str(meta_path), str(categories_path) if categories_path.exists() else None, elapsed
    return True, None, meta, str(meta_path) if meta_path.exists() else None, str(categories_path) if categories_path.exists() else None, elapsed


def _probe_full_hailo(
    onnx_path: Path,
    variant_id: str,
    *,
    runtime: PreparationRuntimeOptions,
    screening_dir: Path,
    log: PreparationLog = None,
    end_node_names: Optional[List[str]] = None,
) -> tuple[bool, Optional[str], Optional[str], Optional[str], Optional[float], Optional[Any]]:
    outdir = screening_dir / "hailo_probe" / variant_id / "full"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    suffix = '' if not end_node_names else f" (end_nodes={end_node_names})"
    _prepare_log(log, f"[prepare][hailo] {variant_id}: probing full HEF build{suffix}")
    try:
        from ..hailo_backend import hailo_build_hef_auto
        result = hailo_build_hef_auto(
            onnx_path,
            backend=str(runtime.hailo_backend or "auto"),
            hw_arch=str(runtime.hw_arch or "hailo8"),
            net_name=f"{onnx_path.stem}_full",
            outdir=outdir,
            fixup=bool(runtime.hef_fixup),
            opt_level=int(runtime.hef_opt_level or 1),
            calib_dir=(str(runtime.calib_dir) if runtime.calib_dir else None),
            calib_count=int(runtime.calib_count or 64),
            calib_batch_size=int(runtime.calib_batch_size or 8),
            force=bool(runtime.hef_force),
            keep_artifacts=bool(runtime.hef_keep_artifacts),
            wsl_distro=runtime.wsl_distro,
            wsl_venv_activate=str(runtime.wsl_venv or "auto"),
            wsl_timeout_s=int(runtime.hef_timeout_s or 3600),
            on_log=(lambda level, msg: _prepare_log(log, f"[prepare][hailo][{level}] {msg}")),
            end_node_names=list(end_node_names or []) or None,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        return False, f"{type(exc).__name__}: {exc}", None, None, elapsed, None
    elapsed = time.time() - t0
    ok = bool(getattr(result, "ok", False))
    err = None if ok else str(getattr(result, "error", None) or "full Hailo probe failed")
    result_json = str(getattr(result, "result_json_path", None) or "") or None
    hef_path = str(getattr(result, "hef_path", None) or "") or None
    return ok, err, result_json, hef_path, elapsed, result


def _mark_export_metadata(meta_path: Optional[str], payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not meta_path:
        return {}
    p = Path(meta_path)
    meta = _load_json_if_exists(p)
    meta["preparation_screening"] = dict(payload or {})
    _write_json(p, meta)
    return meta


def prepare_model_for_benchmark(
    model_path: str | Path,
    *,
    export_metadata: Optional[Mapping[str, Any]] = None,
    requested_mode: str = "current",
    entry: Optional[Mapping[str, Any]] = None,
    output_root: str | Path,
    runtime: Optional[PreparationRuntimeOptions] = None,
    log: PreparationLog = None,
    cancel_event: Any = None,
) -> ModelPreparationResult:
    src_path = Path(model_path).expanduser().resolve()
    runtime = runtime or PreparationRuntimeOptions()
    mode = normalize_model_preparation_mode(requested_mode)
    meta = dict(export_metadata or load_export_metadata_for_model(src_path) or {})
    screening_dir: Optional[Path] = None

    if mode == "current":
        return ModelPreparationResult(
            requested_mode=requested_mode,
            applied_mode="current",
            source_model_path=str(src_path),
            source_model_ref=None,
            screening_profile=None,
            screening_dir=None,
            success=True,
            skipped=True,
            selected_model_path=str(src_path),
            selected_variant_id=None,
            selected_label=None,
            selected_export_json_path=str(src_path.with_suffix('.export.json')) if src_path.with_suffix('.export.json').exists() else None,
            selected_export_metadata=meta,
            message="Using current ONNX unchanged.",
            variants=[],
        )

    source = str(meta.get("source") or "").strip().lower()
    task_type = str(meta.get("task_type") or (entry or {}).get("task") or "").strip().lower()
    family = _detect_yolo_family(meta, src_path, entry)
    model_ref = _infer_ultralytics_model_ref(meta, src_path, entry)
    export_model_ref, export_model_ref_error = _resolve_export_model_ref(model_ref, family, src_path)

    # Heuristic fallback for plain YOLO ONNX files without sidecar metadata.
    if not source and family:
        source = "ultralytics"
        meta["source"] = source
    if not task_type and family:
        stem_l = src_path.stem.lower()
        task_type = "classification" if "-cls" in stem_l or stem_l.endswith("cls") else "detection"
        meta["task_type"] = task_type

    if source != "ultralytics" or task_type != "detection" or not family or not model_ref:
        message = "Preparation mode applies only to Ultralytics detection models. The current ONNX could not be identified as such, so preparation was skipped and the current ONNX remains selected."
        _prepare_log(log, f"[prepare] skipped: {message}")
        return ModelPreparationResult(
            requested_mode=requested_mode,
            applied_mode="current",
            source_model_path=str(src_path),
            source_model_ref=model_ref or None,
            screening_profile=None,
            screening_dir=None,
            success=True,
            skipped=True,
            selected_model_path=str(src_path),
            selected_variant_id=None,
            selected_label=None,
            selected_export_json_path=str(src_path.with_suffix('.export.json')) if src_path.with_suffix('.export.json').exists() else None,
            selected_export_metadata=meta,
            message=message,
            variants=[],
        )

    profile_name = _builtin_screening_profile_name(family)
    specs = _builtin_variant_specs(family)
    if not _family_supports_builtin_export_screening(family):
        specs = [spec for spec in specs if spec.include_current_onnx]
    ts = time.strftime("%Y%m%d_%H%M%S")
    screening_dir = Path(output_root).expanduser().resolve() / f"{src_path.stem}_prep_{ts}"
    exports_dir = screening_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    results: List[PreparedVariantResult] = []
    selected: Optional[PreparedVariantResult] = None

    _prepare_log(log, f"[prepare] mode={mode} family={family} model_ref={model_ref}")
    _prepare_log(log, f"[prepare] screening profile: {profile_name}")
    _prepare_log(log, f"[prepare] source model: {src_path}")
    if not _family_supports_builtin_export_screening(family):
        _prepare_log(log, f"[prepare] family {family!r} has no built-in export fallback set; only current ONNX and Tier-2 parser diagnostics will be attempted")

    for spec in specs:
        if cancel_event is not None and bool(getattr(cancel_event, 'is_set', lambda: False)()):
            raise RuntimeError('Preparation cancelled.')
        if spec.include_current_onnx:
            ok, err, result_json, hef_path, elapsed_probe, _probe_obj = _probe_full_hailo(src_path, spec.variant_id, runtime=runtime, screening_dir=screening_dir, log=log)
            rec = PreparedVariantResult(
                variant_id=spec.variant_id,
                label=spec.label,
                kind="current_onnx",
                model_path=str(src_path),
                export_json_path=str(src_path.with_suffix('.export.json')),
                categories_path=str(src_path.with_suffix('.categories.json')) if src_path.with_suffix('.categories.json').exists() else None,
                export_metadata=meta,
                export_ok=True,
                full_hailo_ok=ok,
                full_hailo_error=err,
                full_hailo_result_json=result_json,
                full_hailo_hef=hef_path,
                elapsed_probe_s=elapsed_probe,
            )
            results.append(rec)
            if ok:
                selected = rec
                _prepare_log(log, "[prepare] current ONNX already passes full-Hailo probe")
                break
            _prepare_log(log, f"[prepare] current ONNX failed full-Hailo probe: {err}")
            raw_ok, raw_nodes, raw_err, raw_json, raw_hef, raw_elapsed = _run_raw_detection_head_probe(
                base_variant_id=spec.variant_id,
                label=spec.label,
                family=family,
                model_path=src_path,
                runtime=runtime,
                screening_dir=screening_dir,
                log=log,
            )
            rec.raw_head_attempted = bool(raw_nodes)
            rec.raw_head_ok = bool(raw_ok)
            rec.raw_head_error = raw_err
            rec.raw_head_result_json = raw_json
            rec.raw_head_hef = raw_hef
            rec.raw_head_elapsed_probe_s = raw_elapsed
            rec.raw_head_end_node_names = list(raw_nodes or [])
            if raw_ok:
                selected_id = f"{spec.variant_id}_raw_heads"
                selected_path, selected_meta_path, selected_categories_path = _materialize_current_raw_head_variant(
                    src_path=src_path,
                    exports_dir=exports_dir,
                    variant_id=selected_id,
                    export_metadata=meta,
                    log=log,
                )
                rec.variant_id = selected_id
                rec.label = f"{spec.label} · Hailo raw detection-head endpoints"
                rec.model_path = selected_path
                rec.export_json_path = selected_meta_path
                rec.categories_path = selected_categories_path
                rec.full_hailo_ok = True
                rec.full_hailo_error = None
                rec.full_hailo_result_json = raw_json
                rec.full_hailo_hef = raw_hef
                rec.elapsed_probe_s = raw_elapsed
                rec.full_hailo_endpoint_mode = "raw_detection_head"
                rec.full_hailo_end_node_names = list(raw_nodes or [])
                selected = rec
                _prepare_log(log, f"[prepare] selected variant {selected_id}: {rec.label}")
                break
            allow_parser_tier2 = str(os.environ.get("OSP_YOLO_PARSER_SUGGESTED_TIER2") or "").strip().lower() in {"1", "true", "yes", "on"}
            if raw_nodes and not allow_parser_tier2:
                _prepare_log(log, f"[prepare][tier2] {spec.variant_id}: skipping parser-suggested decode-tail end nodes because raw-head endpoints were available; set OSP_YOLO_PARSER_SUGGESTED_TIER2=1 to force that diagnostic")
                continue
            t2_ok, t2_nodes, t2_err, t2_json, t2_hef, t2_elapsed = _run_tier2_parser_endnode_probe(
                base_variant_id=spec.variant_id,
                label=spec.label,
                model_path=src_path,
                full_hailo_error=err,
                runtime=runtime,
                screening_dir=screening_dir,
                log=log,
            )
            rec.tier2_attempted = bool(t2_nodes)
            rec.tier2_ok = bool(t2_ok)
            rec.tier2_error = t2_err
            rec.tier2_result_json = t2_json
            rec.tier2_hef = t2_hef
            rec.tier2_elapsed_probe_s = t2_elapsed
            rec.tier2_end_node_names = list(t2_nodes or [])
            continue

        if not export_model_ref:
            rec = PreparedVariantResult(
                variant_id=spec.variant_id,
                label=spec.label,
                kind="exported",
                model_path=str(exports_dir / f"{src_path.stem}__{spec.variant_id}.onnx"),
                export_ok=False,
                export_error=export_model_ref_error or "no export source checkpoint available",
            )
            _prepare_log(log, f"[prepare] export skipped for {spec.variant_id}: {rec.export_error}")
            results.append(rec)
            continue
        out_path = exports_dir / f"{src_path.stem}__{spec.variant_id}.onnx"
        exp_ok, exp_err, exp_meta, exp_json_path, exp_categories_path, elapsed_export = _run_export_script(export_model_ref, spec, out_path, runtime=runtime, log=log)
        rec = PreparedVariantResult(
            variant_id=spec.variant_id,
            label=spec.label,
            kind="exported",
            model_path=str(out_path),
            export_json_path=exp_json_path,
            categories_path=exp_categories_path,
            export_metadata=dict(exp_meta or {}),
            export_ok=bool(exp_ok),
            export_error=exp_err,
            elapsed_export_s=elapsed_export,
        )
        if exp_ok:
            ok, err, result_json, hef_path, elapsed_probe, _probe_obj = _probe_full_hailo(out_path, spec.variant_id, runtime=runtime, screening_dir=screening_dir, log=log)
            rec.full_hailo_ok = ok
            rec.full_hailo_error = err
            rec.full_hailo_result_json = result_json
            rec.full_hailo_hef = hef_path
            rec.elapsed_probe_s = elapsed_probe
            if ok:
                selected = rec
                results.append(rec)
                _prepare_log(log, f"[prepare] selected variant {spec.variant_id}: {spec.label}")
                break
            _prepare_log(log, f"[prepare] variant {spec.variant_id} failed full-Hailo probe: {err}")
            raw_ok, raw_nodes, raw_err, raw_json, raw_hef, raw_elapsed = _run_raw_detection_head_probe(
                base_variant_id=spec.variant_id,
                label=spec.label,
                family=family,
                model_path=out_path,
                runtime=runtime,
                screening_dir=screening_dir,
                log=log,
            )
            rec.raw_head_attempted = bool(raw_nodes)
            rec.raw_head_ok = bool(raw_ok)
            rec.raw_head_error = raw_err
            rec.raw_head_result_json = raw_json
            rec.raw_head_hef = raw_hef
            rec.raw_head_elapsed_probe_s = raw_elapsed
            rec.raw_head_end_node_names = list(raw_nodes or [])
            if raw_ok:
                selected_id = f"{spec.variant_id}_raw_heads"
                rec.variant_id = selected_id
                rec.label = f"{spec.label} · Hailo raw detection-head endpoints"
                rec.full_hailo_ok = True
                rec.full_hailo_error = None
                rec.full_hailo_result_json = raw_json
                rec.full_hailo_hef = raw_hef
                rec.elapsed_probe_s = raw_elapsed
                rec.full_hailo_endpoint_mode = "raw_detection_head"
                rec.full_hailo_end_node_names = list(raw_nodes or [])
                selected = rec
                results.append(rec)
                _prepare_log(log, f"[prepare] selected variant {selected_id}: {rec.label}")
                break
            allow_parser_tier2 = str(os.environ.get("OSP_YOLO_PARSER_SUGGESTED_TIER2") or "").strip().lower() in {"1", "true", "yes", "on"}
            if raw_nodes and not allow_parser_tier2:
                _prepare_log(log, f"[prepare][tier2] {spec.variant_id}: skipping parser-suggested decode-tail end nodes because raw-head endpoints were available; set OSP_YOLO_PARSER_SUGGESTED_TIER2=1 to force that diagnostic")
            else:
                t2_ok, t2_nodes, t2_err, t2_json, t2_hef, t2_elapsed = _run_tier2_parser_endnode_probe(
                    base_variant_id=spec.variant_id,
                    label=spec.label,
                    model_path=out_path,
                    full_hailo_error=err,
                    runtime=runtime,
                    screening_dir=screening_dir,
                    log=log,
                )
                rec.tier2_attempted = bool(t2_nodes)
                rec.tier2_ok = bool(t2_ok)
                rec.tier2_error = t2_err
                rec.tier2_result_json = t2_json
                rec.tier2_hef = t2_hef
                rec.tier2_elapsed_probe_s = t2_elapsed
                rec.tier2_end_node_names = list(t2_nodes or [])
        else:
            _prepare_log(log, f"[prepare] export failed for {spec.variant_id}: {exp_err}")
        results.append(rec)

    failure_message = "No export variant passed the full-Hailo probe."
    tier2_hits = [r for r in results if bool(r.tier2_ok)]
    raw_head_hits = [r for r in results if bool(r.raw_head_ok)]
    raw_head_best = raw_head_hits[0] if raw_head_hits else None
    tier2_best = (tier2_hits[0] if tier2_hits else raw_head_best)
    selected_endpoint_mode = str(getattr(selected, "full_hailo_endpoint_mode", "") or "") if selected else ""
    selected_endpoint_nodes = list(getattr(selected, "full_hailo_end_node_names", []) or []) if selected else []
    if selected is None:
        exported = [r for r in results if r.kind == "exported"]
        if raw_head_best is not None:
            failure_message = (
                f"No export variant passed the default full-Hailo probe. The YOLO raw detection-head endpoint probe was attempted for {raw_head_best.variant_id} "
                f"with end nodes {raw_head_best.raw_head_end_node_names}, but it did not compile: {raw_head_best.raw_head_error or 'unknown error'}"
            )
        elif tier2_best is not None:
            failure_message = (
                f"No export variant passed the full-Hailo probe. Tier-2 parser-suggested end-node probing succeeded for {tier2_best.variant_id} "
                f"with end nodes {tier2_best.tier2_end_node_names}, but it was not auto-selected."
            )
        elif exported and all(not r.export_ok for r in exported):
            dep_msgs = [str(r.export_error or "") for r in exported]
            uniq = []
            for m in dep_msgs:
                if m and m not in uniq:
                    uniq.append(m)
            if any("missing dependency" in m or "bootstrap failed" in m for m in dep_msgs):
                failure_message = "No export variant could be generated because the export environment is missing required Python packages. " + (uniq[0] if uniq else "")
            elif uniq:
                failure_message = "No export variant could be generated. " + uniq[0]
        elif results and results[0].kind == "current_onnx" and results[0].full_hailo_error and not exported:
            failure_message = "Current ONNX failed the full-Hailo probe and no fallback export variants were available."
    summary = ModelPreparationResult(
        requested_mode=requested_mode,
        applied_mode=mode,
        source_model_path=str(src_path),
        source_model_ref=model_ref,
        screening_profile=profile_name,
        screening_dir=str(screening_dir),
        success=selected is not None,
        skipped=False,
        selected_model_path=str(selected.model_path if selected else src_path),
        selected_variant_id=(selected.variant_id if selected else None),
        selected_label=(selected.label if selected else None),
        selected_export_json_path=(selected.export_json_path if selected else None),
        selected_full_hailo_endpoint_mode=(selected.full_hailo_endpoint_mode if selected else None),
        selected_full_hailo_end_node_names=list((selected.full_hailo_end_node_names if selected else []) or []),
        selected_export_metadata=dict(selected.export_metadata if selected else meta),
        message=((f"Selected variant {selected.variant_id}: {selected.label} · raw detection head endpoints" if getattr(selected, "full_hailo_endpoint_mode", None) == "raw_detection_head" else f"Selected variant {selected.variant_id}: {selected.label}") if selected else failure_message),
        variants=results,
        tier2_success=bool(tier2_best is not None),
        tier2_variant_id=(tier2_best.variant_id if tier2_best is not None else None),
        tier2_label=(tier2_best.label if tier2_best is not None else None),
        tier2_end_node_names=list(((tier2_best.tier2_end_node_names or tier2_best.raw_head_end_node_names) if tier2_best is not None else []) or []),
    )

    screening_summary_path = screening_dir / "screening_summary.json"
    _write_json(screening_summary_path, summary.to_metadata())
    if selected is not None:
        payload = {
            "mode": mode,
            "screening_profile": profile_name,
            "source_model_path": str(src_path),
            "source_model_ref": model_ref,
            "screening_dir": str(screening_dir),
            "selected": True,
            "variant_id": selected.variant_id,
            "variant_label": selected.label,
            "full_hailo_ok": bool(selected.full_hailo_ok or selected.raw_head_ok),
            "full_hailo_result_json": selected.full_hailo_result_json or selected.raw_head_result_json,
            "full_hailo_hef": selected.full_hailo_hef or selected.raw_head_hef,
            "raw_head_hailo_ok": bool(selected.raw_head_ok),
            "full_hailo_endpoint_mode": selected.full_hailo_endpoint_mode,
            "full_hailo_end_node_strategy": ("raw_detection_head" if selected.full_hailo_endpoint_mode == "raw_detection_head" else None),
            "full_hailo_end_node_names": list(selected.full_hailo_end_node_names or []),
            "full_hailo_true_full_model": not bool(selected.full_hailo_end_node_names),
        }
        selected_meta_path = selected.export_json_path or str(Path(selected.model_path).with_suffix('.export.json'))
        selected_meta = _mark_export_metadata(selected_meta_path, payload)
        if selected_meta:
            selected.export_json_path = selected_meta_path
            summary.selected_export_json_path = selected_meta_path
            summary.selected_export_metadata = selected_meta
            # copy categories/json if needed handled by export script already
        _write_json(screening_summary_path, summary.to_metadata())
    _prepare_log(log, f"[prepare] summary: {summary.message}")
    if summary.tier2_success:
        _prepare_log(log, f"[prepare] tier2 endpoint probe: variant={summary.tier2_variant_id} end_nodes={summary.tier2_end_node_names}")
    _prepare_log(log, f"[prepare] summary json: {screening_summary_path}")
    return summary
