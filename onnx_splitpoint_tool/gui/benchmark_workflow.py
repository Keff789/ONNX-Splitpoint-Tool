"""Benchmark benchmark-set workflow controller extracted from gui_app.py.

This module keeps the heavy benchmark-suite generation orchestration out of the
legacy Tk root class so gui_app.py can stay focused on widget state and thin
entrypoints.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import queue
import re
import threading
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from .. import __version__ as TOOL_VERSION
from ..benchmark_case_utils import build_benchmark_case_rejection
from ..config_values import parse_config_bool
from ..cache_verify_policy import (
    cache_miss_blocked_message,
    compiler_dispatch_forbidden,
)
from ..benchmark.generation_state import find_latest_resumable_set, read_json as read_generation_json
from ..benchmark.hailo_scoring import rerank_candidates_for_hailo
from ..benchmark.evaluation_profiles import load_export_metadata_for_model, resolve_evaluation_profile
from ..benchmark.classification_validation_presets import resolve_classification_validation_source
from ..benchmark.validation_assets import coco50_dataset_dir, coco200_dataset_dir
from ..benchmark.model_preparation import find_latest_preparation_full_hailo_baseline, load_preparation_full_hailo_baseline, load_preparation_full_hailo_end_nodes, normalize_model_preparation_mode, preparation_result_is_selected_model
from ..benchmark.resume_integrity import reconcile_generation_state
from ..benchmark.schema import stamp_benchmark_set_payload, write_json_atomic as write_benchmark_json_atomic
from ..benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationConfig,
    BenchmarkGenerationOrchestrationService,
    BenchmarkGenerationService,
    normalize_full_hef_policy,
    normalize_hailo_full_model_preflight_policy,
)
from .controller import write_benchmark_suite_script
from .widgets.benchmark_completion_dialog import show_benchmark_completion_dialog
from ..hailo.backend_mode import normalize_hailo_backend
from ..resources_utils import copy_resource_tree
from ..workdir import ensure_workdir

__version__ = TOOL_VERSION
logger = logging.getLogger(__name__)


def _infer_task_from_model_name_or_path(value: str) -> str:
    s = str(value or "").lower()
    if any(k in s for k in ("yolo", "coco", "detect", "ssd", "retina", "fasterrcnn")):
        return "detection"
    if any(k in s for k in ("resnet", "mobilenet", "regnet", "efficientnet", "convnext", "imagenet", "classifier", "classification")):
        return "classification"
    return "auto"


def _resolve_preset_images_dir(value: str) -> str:
    """Resolve a classification preset/path to an image directory when possible."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.lower() in {"coco", "coco_50", "coco50", "coco_50_data"}:
        try:
            p = coco50_dataset_dir()
            return str(p) if p.exists() else ""
        except Exception:
            return ""
    if raw.lower() in {"coco_200", "coco200", "coco_200_data"}:
        try:
            p = coco200_dataset_dir()
            return str(p) if p.exists() else ""
        except Exception:
            return ""
    try:
        p = resolve_classification_validation_source(raw)
        if p is None:
            return ""
        if p.is_file():
            root = p.parent
        else:
            root = p
        imgs = root / "images"
        return str(imgs if imgs.is_dir() else root)
    except Exception:
        return ""


def _resolve_validation_source_for_task(*, task: str, cls_preset: str, det_preset: str) -> str:
    task_l = str(task or "auto").lower()
    if task_l == "classification":
        return str(cls_preset or "imagenette_val_mini_200").strip()
    if task_l == "detection":
        det = str(det_preset or "coco_50").strip().lower()
        if det in {"coco", "coco_50", "coco50", "", "coco_50_data"}:
            return "coco_50"
        if det in {"coco_200", "coco200", "coco_200_data"}:
            return "coco_200"
        return det
    return ""


@dataclass
class ResolvedHailoBenchmarkHelpers:
    hailo_build_hef_fn: Optional[Any] = None
    hailo_parse_check_fn: Optional[Any] = None
    hailo_build_unavailable: Optional[str] = None
    hailo_part2_precheck_fn: Optional[Any] = None
    hailo_part2_precheck_error_fn: Optional[Any] = None
    hailo_part2_parser_precheck_fn: Optional[Any] = None
    hailo_part2_parser_precheck_error_fn: Optional[Any] = None
    hailo_part2_import_error: Optional[str] = None


def resolve_tool_core_version(*, importer=None) -> str:
    importer = importer or importlib.import_module
    try:
        mod = importer("onnx_splitpoint_tool.api")
        return str(getattr(mod, "__version__", "?"))
    except Exception:
        logger.debug("Failed to resolve tool core version from onnx_splitpoint_tool.api", exc_info=True)
        return "?"


def resolve_hailo_benchmark_helpers(*, need_build: bool, need_part2: bool, importer=None) -> ResolvedHailoBenchmarkHelpers:
    importer = importer or importlib.import_module
    resolved = ResolvedHailoBenchmarkHelpers()
    if not need_build and not need_part2:
        return resolved

    try:
        hailo_mod = importer("onnx_splitpoint_tool.hailo_backend")
    except Exception as exc:
        if need_build:
            resolved.hailo_build_unavailable = f"Hailo HEF build unavailable: {exc}"
        if need_part2:
            resolved.hailo_part2_import_error = f"{type(exc).__name__}: {exc}"
        return resolved

    if need_build:
        try:
            resolved.hailo_build_hef_fn = getattr(hailo_mod, "hailo_build_hef_auto")
        except Exception as exc:
            resolved.hailo_build_unavailable = f"Hailo HEF build unavailable: {exc}"
        try:
            resolved.hailo_parse_check_fn = getattr(hailo_mod, "hailo_parse_check_auto")
        except Exception:
            logger.debug("Failed to resolve hailo_parse_check_auto for benchmark workflow", exc_info=True)

    if need_part2:
        try:
            resolved.hailo_part2_precheck_error_fn = getattr(hailo_mod, "format_hailo_part2_activation_precheck_error")
            resolved.hailo_part2_parser_precheck_error_fn = getattr(hailo_mod, "format_hailo_part2_parser_blocker_error")
            resolved.hailo_part2_precheck_fn = getattr(hailo_mod, "hailo_part2_activation_precheck_from_manifest")
            resolved.hailo_part2_parser_precheck_fn = getattr(hailo_mod, "hailo_part2_parser_blocker_precheck_from_model")
        except Exception as exc:
            resolved.hailo_part2_import_error = f"{type(exc).__name__}: {exc}"

    return resolved




def _prepared_full_hailo_endpoint_override(model_path: str | Path) -> tuple[List[str], str]:
    """Read prepared full-Hailo endpoint override from model sidecar metadata."""
    try:
        info = load_preparation_full_hailo_end_nodes(model_path)
        nodes = [str(x).strip() for x in list(info.get('end_node_names') or []) if str(x).strip()]
        strategy = str(info.get('strategy') or '').strip()
        return nodes, strategy
    except Exception:
        logger.debug('Failed to resolve prepared full-Hailo endpoint override for %s', model_path, exc_info=True)
        return [], ''


def _prepared_full_hailo_baseline(model_path: str | Path) -> Dict[str, Any]:
    """Read a prepared full-Hailo HEF artifact from sidecar or latest screening summary."""
    try:
        sidecar = dict(load_preparation_full_hailo_baseline(model_path) or {})
        if bool(sidecar.get('ok')):
            return sidecar
        latest = dict(find_latest_preparation_full_hailo_baseline(model_path) or {})
        if bool(latest.get('ok')):
            return latest
        return sidecar if sidecar else latest
    except Exception:
        logger.debug('Failed to resolve prepared full-Hailo baseline for %s', model_path, exc_info=True)
        return {'selected': False, 'ok': False, 'reason': 'metadata_error'}

def _safe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None





def _manual_deepx_input_contract(model: Any) -> tuple[str, list[int]]:
    """Infer a simple NCHW input contract for manual DeepX full builds."""
    input_name = "images"
    shape = [1, 3, 640, 640]
    try:
        g = getattr(model, "graph", None)
        inputs = list(getattr(g, "input", []) or []) if g is not None else []
        if inputs:
            inp = inputs[0]
            input_name = str(getattr(inp, "name", None) or input_name)
            dims = []
            shp = inp.type.tensor_type.shape  # type: ignore[attr-defined]
            for d in shp.dim:
                v = int(getattr(d, "dim_value", 0) or 0)
                dims.append(v if v > 0 else None)
            if len(dims) == 4:
                # Default symbolic batch to one; keep known C/H/W.
                out = [int(dims[0] or 1), int(dims[1] or 3), int(dims[2] or 640), int(dims[3] or 640)]
                # If the model is NHWC, convert the config shape to NCHW because
                # the current DX-COM image loader emits NCHW after transpose.
                if out[1] not in (1, 3, 4) and out[-1] in (1, 3, 4):
                    out = [out[0], out[-1], out[1], out[2]]
                shape = out
    except Exception:
        pass
    return input_name, shape


def _manual_deepx_effective_calib_dir(
    out_dir: Path,
    validation_images: str,
    fallback_calib_dir: Optional[str],
    *,
    task_hint: str = "",
    model_path: str = "",
    input_shape: Optional[List[int]] = None,
) -> str:
    """Choose a calibration image directory for the manual DeepX full build.

    v52m: make this task-aware.  The manual Benchmark tab can build DeepX
    artifacts before the remote-suite refresh has copied validation assets into
    the suite.  For classification models this previously fell back to COCO-50,
    so ResNet/MobileNet/RegNet DXNNs were calibrated with detection images.  We
    now prefer tool-wide Imagenette/ImageNet mini datasets for classification and
    only use COCO as a last-resort fallback.
    """
    candidates: list[Path] = []

    task_s = str(task_hint or "").strip().lower()
    blob = " ".join([
        str(model_path or ""),
        str(validation_images or ""),
        " ".join(str(x) for x in (input_shape or [])),
    ]).lower()
    cls_markers = (
        "classification", "imagenet", "imagenette", "resnet", "mobilenet", "regnet",
        "efficientnet", "convnext", "squeezenet", "densenet", "vgg", "inception",
        "classifier", "cls",
    )
    det_markers = (
        "detection", "detect", "coco", "yolo", "ssd", "retinanet", "detr", "bbox",
    )
    # v60r: an explicit per-model task is authoritative.  Earlier code still
    # inspected stale validation paths after seeing task=detection; an old
    # ``imagenette_*`` path could therefore flip YOLO Part1 calibration back to
    # classification.  Heuristic markers are consulted only when the task is
    # genuinely unknown.
    if task_s == "classification":
        inferred_classification = True
        inferred_detection = False
    elif task_s == "detection":
        inferred_classification = False
        inferred_detection = True
    else:
        inferred_classification = any(m in blob for m in cls_markers)
        inferred_detection = any(m in blob for m in det_markers)
        if inferred_classification:
            inferred_detection = False

    def _global_validation_root() -> Path:
        return Path.home() / ".onnx_splitpoint_tool" / "validation_datasets"

    def _add_candidate(raw: object, *, also_parent: bool = True) -> None:
        raw_s = str(raw or "").strip()
        if not raw_s:
            return
        alias = raw_s.lower().replace("-", "_").replace(" ", "_")
        # Bare preset aliases used by the GUI/profile system.
        if alias in {"imagenette_val_mini_200", "imagenette200", "imagenette_200", "imagenet_val_mini_200", "imagenet200"}:
            raw_s = "resources/validation/classification/imagenette_val_mini_200/manifest.json"
        elif alias in {"imagenette_val_mini_500", "imagenette500", "imagenette_500", "imagenet_val_mini_500", "imagenet500"}:
            raw_s = "resources/validation/classification/imagenette_val_mini_500/manifest.json"
        elif alias in {"coco_50", "coco50", "coco_50_data"}:
            raw_s = "resources/validation/detection/coco_50_data"
        p = Path(raw_s).expanduser()
        if not p.is_absolute():
            p = out_dir / p
        candidates.append(p)
        if also_parent and p.name.lower() in {"manifest.json", "annotations.json"}:
            candidates.append(p.parent)
            candidates.append(p.parent / "images")

    def _add_global_classification_presets() -> None:
        names = ["imagenette_val_mini_200", "imagenette_val_mini_500", "imagenet_val_mini_200", "imagenet_val_mini_500"]
        roots: list[Path] = []
        # Prefer the tool's own helper functions when they exist, then fall back
        # to the stable default under ~/.onnx_splitpoint_tool.
        try:
            from ..benchmark.classification_validation_presets import classification_validation_default_root  # type: ignore
            roots.append(Path(classification_validation_default_root()).expanduser())
        except Exception:
            pass
        try:
            from ..benchmark.validation_assets import classification_dataset_root  # type: ignore
            roots.append(Path(classification_dataset_root()).expanduser())
        except Exception:
            pass
        roots.append(_global_validation_root() / "classification")
        seen_roots: set[str] = set()
        for root in roots:
            key = str(root)
            if key in seen_roots:
                continue
            seen_roots.add(key)
            for name in names:
                base = root / name
                candidates.extend([base / "manifest.json", base / "images", base])

    def _add_global_detection_presets() -> None:
        roots: list[Path] = []
        try:
            from ..benchmark.validation_assets import coco50_dataset_dir, coco200_dataset_dir  # type: ignore
            roots.append(Path(coco50_dataset_dir()).expanduser())
        except Exception:
            pass
        roots.extend([
            _global_validation_root() / "detection" / "coco_50_data",
            _global_validation_root() / "detection" / "coco_200_data",
            _global_validation_root() / "test_images",
        ])
        candidates.extend(roots)

    # Strong preference order.  For classification, treat validation_images as
    # the *semantic validation* source, not as the calibration source.  v52w
    # still allowed imagenette_val_mini_200 from validation_images to beat the
    # Tool-Config calibration preset imagenette_val_mini_500.  That made DeepX
    # Full reuse/build with the validation set despite the split config showing
    # separate calibration/validation presets.  Prefer explicit calibration
    # override and tool-wide 500-image calibration before validation_images.
    if inferred_classification:
        fb = str(fallback_calib_dir or "")
        if fb and "coco" not in fb.lower():
            _add_candidate(fb)
        _add_candidate("resources/validation/classification/imagenette_val_mini_500/manifest.json")
        _add_candidate("imagenette_val_mini_500")
        _add_global_classification_presets()
        # Only after calibration presets, consider the validation preset.
        _add_candidate(validation_images)
        _add_candidate("resources/validation/classification/imagenette_val_mini_200/manifest.json")
    else:
        _add_candidate(fallback_calib_dir or "")
        _add_candidate(validation_images)
        if inferred_detection:
            _add_candidate("resources/validation/detection/coco_50_data")
            _add_candidate("resources/validation/detection/coco_200_data")
            _add_global_detection_presets()
        else:
            # Unknown task: prefer generic image assets, but include both task
            # presets so the first existing image-rich directory wins.
            _add_candidate("resources/validation/classification/imagenette_val_mini_200/manifest.json")
            _add_global_classification_presets()
            _add_candidate("resources/validation/detection/coco_50_data")
            _add_candidate("resources/validation/detection/coco_200_data")
            _add_global_detection_presets()
    _add_candidate("resources/validation/test_images")
    _add_candidate("resources/validation")

    def _image_count(path: Path) -> int:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".PNG"}
        try:
            return sum(1 for x in path.iterdir() if x.is_file() and x.suffix in exts)
        except Exception:
            return 0

    def _resolve_to_image_dir(p: Path) -> Optional[Path]:
        try:
            if p.is_file() and p.name.lower() == "manifest.json":
                p = p.parent
            if not p.exists() or not p.is_dir():
                return None
            # Most generated classification manifests have an images/ sibling.
            img_sub = p / "images"
            if img_sub.is_dir() and (_image_count(img_sub) > 0 or any(c.is_dir() for c in img_sub.iterdir())):
                return img_sub
            if _image_count(p) > 0:
                return p
            children = sorted([c for c in p.iterdir() if c.is_dir()])
            if p.name.lower() == "images" and children:
                return p
            class_like = []
            for c in children:
                if _image_count(c) > 0:
                    class_like.append(c)
            # If this is a classification dataset root, return the root, not the
            # first class folder, so calibration can see all classes.
            if (p / "manifest.json").is_file() or len(class_like) >= 2:
                return p
            if len(class_like) == 1:
                return class_like[0]
        except Exception:
            return None
        return None

    def _is_coco_like(p: Path) -> bool:
        return "coco" in str(p).lower()

    seen: set[str] = set()
    # First pass: for classification, skip COCO-like candidates.
    for allow_coco in ([False, True] if inferred_classification else [True]):
        for p in candidates:
            key = str(p)
            if key in seen and allow_coco:
                continue
            if inferred_classification and not allow_coco and _is_coco_like(p):
                continue
            if allow_coco:
                seen.add(key)
            resolved = _resolve_to_image_dir(p)
            if resolved is not None:
                return str(resolved)
    return ""


def _materialize_manual_deepx_full_artifact(
    *,
    out_dir: Path,
    model_path: str,
    model: Any,
    bench_plan_runs: List[Dict[str, Any]],
    validation_images: str,
    validation_max_images: int,
    fallback_calib_dir: Optional[str],
    calibration_num: int = 0,
    classification_preprocessing: Optional[str] = None,
    task_hint: str = "",
    force_build: bool = False,
    log: Any = None,
    build_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build/reuse the DeepX full DXNN for the manual Benchmark tab.

    The Evaluation Workflow already has a formal DeepX binding.  This helper
    gives the manual Benchmark tab the same artifact-first behavior so users can
    quickly create a benchmark set, build the full DXNN, and then run it remotely.
    """
    force_build = parse_config_bool(force_build, field="deepx_build.force_build")
    from ..build_dispatch_policy import require_productive_force_off
    require_productive_force_off({"deepx_build": dict(build_config or {})})
    require_productive_force_off({"deepx_build": {"force_build": force_build}})
    selected = any(str(r.get("type") or "").strip().lower() in {"deepx", "deepx_m1", "dx_m1"} for r in list(bench_plan_runs or []))
    status_dir = out_dir / "deepx" / "deepx_m1" / "full"
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "deepx_artifact_status.json"
    payload: Dict[str, Any] = {
        "schema": "onnx-splitpoint/manual-deepx-full-artifact-status",
        "schema_version": 1,
        "selected": bool(selected),
        "backend": "deepx_m1",
        "variant": "full",
        "status": "not_selected",
    }
    if not selected:
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    try:
        from ..deepx.artifacts import (
            cache_dxnn_artifact,
            deepx_cache_key,
            deepx_cached_artifact_compatible,
            sha256_file,
            strict_artifact_sha256,
        )
        from ..deepx.compiler import compile_dxnn
        from ..deepx.config import image_model_dxcom_config, write_dxcom_config, resolve_profile_classification_preprocessing, materialize_imagenet_normalized_build_onnx
        from ..deepx.env_status import inspect_deepx_environment
    except Exception as exc:
        payload.update({"status": "deepx_modules_unavailable", "error": f"{type(exc).__name__}: {exc}"})
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    cache_verify_only = compiler_dispatch_forbidden()
    payload["artifact_policy"] = (
        "cache_verify_only" if cache_verify_only else "normal"
    )
    env = inspect_deepx_environment(probe_import=False, path_only=True, config=build_config)
    payload["environment_status"] = env
    if cache_verify_only and parse_config_bool(force_build, field="deepx_build.force_build"):
        payload.update({
            "status": "cache_miss_blocked",
            "build_status": "cache_miss_blocked",
            "message": cache_miss_blocked_message(
                "deepx_dx_com", "manual Full force_build requested"
            ),
        })
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    p_model = Path(str(model_path or "")).expanduser()
    if not p_model.is_file():
        payload.update({"status": "model_missing", "model_path": str(p_model)})
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    input_name, input_shape = _manual_deepx_input_contract(model)
    task_norm = str(task_hint or '').strip().lower()
    if task_norm not in {'classification', 'detection'}:
        task_norm = _infer_task_from_model_name_or_path(str(p_model))
    if task_norm not in {'classification', 'detection'}:
        payload.update({
            "status": "task_contract_missing",
            "message": "DeepX build requires an explicit classification or detection task.",
        })
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    classification_mode = resolve_profile_classification_preprocessing({**dict(build_config or {}), **({"classification_preprocessing": classification_preprocessing} if classification_preprocessing is not None else {})}) if task_norm == "classification" else "not_applicable"
    calib_dir = _manual_deepx_effective_calib_dir(
        out_dir,
        validation_images,
        fallback_calib_dir,
        task_hint=task_hint,
        model_path=str(p_model),
        input_shape=list(input_shape or []),
    )
    payload.update({"model_path": str(p_model), "input_name": input_name, "input_shape": input_shape, "task": task_norm, "calibration_dir": calib_dir, "calibration_task_hint": task_norm, "calibration_num": int(calibration_num or validation_max_images or 100), "classification_preprocessing": classification_mode, "full_cache_contract_mode": "manual_mean_std_adapter" if classification_mode == "imagenet_mean_std" else "legacy_implicit"})
    if callable(log):
        log(f"[deepx] model: {p_model}")
        log(f"[deepx] input contract: name={input_name} shape={input_shape}")
        log(f"[deepx] calibration task hint: {str(task_hint or 'auto')}")
        log(f"[deepx] calibration dir: {calib_dir or '<missing>'}")
        log(f"[deepx] calibration count: {int(calibration_num or validation_max_images or 100)}")
        vi = str(validation_images or '').strip()
        if vi:
            log(f"[deepx] validation/calibration source hint: {vi}")
    if not calib_dir:
        payload.update({"status": "calibration_dir_missing", "message": "DeepX full build needs calibration images. Set semantic validation images or Hailo/DeepX calibration dir."})
        status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    try:
        cfg = image_model_dxcom_config(
            task=task_norm,
            input_name=input_name,
            input_shape=input_shape,
            calibration_dir=calib_dir,
            calibration_num=max(1, int(calibration_num or validation_max_images or 100)),
            calibration_method="ema",
            classification_preprocessing=classification_mode if task_norm == "classification" else "current_scale_only",
        )
        cfg_path = write_dxcom_config(cfg, status_dir / "config_deepx.json")
        cache_root = Path(str(env.get("cache_dir") or "~/Models/BackendArtifacts/deepx")).expanduser()
        build_onnx = p_model
        adapter_receipt = {}
        if classification_mode == "imagenet_mean_std":
            build_onnx, adapter_receipt = materialize_imagenet_normalized_build_onnx(source_onnx=p_model, output_dir=status_dir / "build_onnx", input_name=input_name)
            adapter_path = status_dir / "build_onnx_adapter_receipt.json"
            adapter_path.write_text(json.dumps(adapter_receipt, indent=2) + "\n", encoding="utf-8")
            payload.update(build_onnx_adapter_receipt=str(adapter_path), build_onnx_sha256=strict_artifact_sha256(build_onnx), source_onnx_sha256=strict_artifact_sha256(p_model))
        key = deepx_cache_key(onnx_path=build_onnx, config_path=cfg_path, target="deepx_m1", variant="full_imagenet_mean_std" if classification_mode == "imagenet_mean_std" else "full")
        cached = cache_root / key / "model.dxnn"
        payload["cache_key"] = key
        payload["cache_root"] = str(cache_root)
        if callable(log):
            log(f"[deepx] dxcom config: {cfg_path}")
            log(f"[deepx] cache key: {key}")
            log(f"[deepx] cache root: {cache_root}")
        if cached.is_file() and not parse_config_bool(force_build, field="deepx_build.force_build"):
            dxnn = cached
            payload["build_status"] = "ready_reused"
            payload["cache_lookup"] = {
                "schema": "onnx-splitpoint/deepx-cache-lookup",
                "schema_version": 1,
                "outcome": "HIT",
                "role": "full",
                "model_id": str(p_model.stem),
                "identity": str(key),
                "reason": "model_present",
                "artifact": str(cached),
            }
            if callable(log):
                log(
                    f"[deepx-cache] HIT role=full model={p_model.stem} "
                    f"identity={key} reason=model_present artifact={dxnn}"
                )
                _cached_log = dxnn.parent / "deepx_build.log"
                if _cached_log.is_file():
                    log(f"[deepx] cached dxcom compiler log: {_cached_log}")
        elif cache_verify_only:
            miss_reason = (
                "force_build_requested" if parse_config_bool(force_build, field="deepx_build.force_build") else "not_found"
            )
            payload["cache_lookup"] = {
                "schema": "onnx-splitpoint/deepx-cache-lookup",
                "schema_version": 1,
                "outcome": "MISS",
                "role": "full",
                "model_id": str(p_model.stem),
                "identity": str(key),
                "reason": miss_reason,
                "artifact": str(cached) if cached.is_file() else "",
            }
            if callable(log):
                log(
                    f"[deepx-cache] MISS role=full model={p_model.stem} "
                    f"identity={key} reason={miss_reason}"
                )
            payload.update({
                "status": "cache_miss_blocked",
                "build_status": "cache_miss_blocked",
                "message": cache_miss_blocked_message(
                    "deepx_dx_com", "manual Full exact cache miss"
                ),
            })
            status_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            return payload
        else:
            miss_reason = (
                "force_build_requested" if parse_config_bool(force_build, field="deepx_build.force_build") else "not_found"
            )
            payload["cache_lookup"] = {
                "schema": "onnx-splitpoint/deepx-cache-lookup",
                "schema_version": 1,
                "outcome": "MISS",
                "role": "full",
                "model_id": str(p_model.stem),
                "identity": str(key),
                "reason": miss_reason,
                "artifact": str(cached) if cached.is_file() else "",
            }
            if callable(log):
                miss_line = (
                    f"[deepx-cache] MISS role=full model={p_model.stem} "
                    f"identity={key} reason={miss_reason}"
                )
                if payload["cache_lookup"]["artifact"]:
                    miss_line += (
                        f" artifact={payload['cache_lookup']['artifact']}"
                    )
                log(miss_line)
            env = inspect_deepx_environment(probe_import=True, config=build_config)
            payload["environment_status"] = env
            if not bool(env.get("compiler_ready")):
                message = "; ".join(str(hint) for hint in env.get("hints") or []) or "DeepX DX-COM compiler environment is not ready."
                payload.update({"status": "compiler_not_ready", "message": message})
                if callable(log):
                    log(f"[deepx] compiler_not_ready: {message}")
                status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return payload
            if callable(log):
                log("[deepx] building full DXNN via dxcom for manual benchmark set")
            res = compile_dxnn(
                onnx_path=build_onnx,
                config_path=cfg_path,
                output_dir=status_dir / "build",
                compiler_root=env.get("dx_all_suite_root"),
                compiler_venv=env.get("compiler_venv"),
                compiler_overlay=env.get("compiler_overlay"), build_config=build_config,
                opt_level=0,
                timeout_s=7200,
            )
            payload["build_result"] = getattr(res, "__dict__", {})
            # v52j: surface the DX-COM compiler output in the generation log so
            # users do not have to dig through deepx/deepx_m1/full/build.
            try:
                _lp = Path(str(getattr(res, "log_path", "") or ""))
                if callable(log) and _lp.is_file():
                    log(f"[deepx] dxcom compiler log: {_lp}")
                    _lines = _lp.read_text(encoding="utf-8", errors="replace").splitlines()
                    _interesting = [ln for ln in _lines if any(tok in ln.lower() for tok in ("error", "warning", "compile", "compiling", "final result", "added nodes", "skipped nodes", "npu", "cpu", "dxnn"))]
                    _tail = (_interesting[-30:] if _interesting else _lines[-30:])
                    for _ln in _tail:
                        log(f"[deepx][dxcom] {_ln[:500]}")
            except Exception:
                pass
            if not (res.ok and res.dxnn_path):
                payload.update({"status": str(res.status or "compile_failed"), "message": str(res.message or "DeepX compile failed")[-1200:]})
                status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                return payload
            dxnn = cache_dxnn_artifact(dxnn_path=res.dxnn_path, manifest={"build_result": getattr(res, "__dict__", {}), "cache_key": key}, cache_root=cache_root, cache_key=key)
            payload["build_status"] = "ready_built"

        dst = status_dir / "model.dxnn"
        shutil.copy2(dxnn, dst)
        if callable(log):
            try:
                log(f"[deepx] final dxnn cache/source: {dxnn} ({Path(dxnn).stat().st_size/1024/1024:.1f} MiB)")
            except Exception:
                log(f"[deepx] final dxnn cache/source: {dxnn}")
            try:
                log(f"[deepx] suite dxnn: {dst} ({Path(dst).stat().st_size/1024/1024:.1f} MiB)")
            except Exception:
                log(f"[deepx] suite dxnn: {dst}")
        preprocess_mode = 'resize' if task_norm == 'classification' else 'letterbox'
        letterbox_pad_value = 0 if preprocess_mode == 'resize' else 114
        if len(input_shape) >= 4 and int(input_shape[1]) in (1, 3, 4):
            runtime_h, runtime_w = int(input_shape[2]), int(input_shape[3])
        elif len(input_shape) >= 4:
            runtime_h, runtime_w = int(input_shape[1]), int(input_shape[2])
        else:
            runtime_h = runtime_w = 224 if task_norm == 'classification' else 640
        contract = {
            "backend": "deepx_m1",
            "variant": "full",
            "artifact_kind": "dxnn",
            "artifact": "deepx/deepx_m1/full/model.dxnn",
            "input": {
                "name": input_name,
                "shape": [runtime_h, runtime_w, 3],
                "dtype": "uint8",
                "layout": "HWC",
                "color_space": "RGB",
                "normalization": "embedded_dxcom_preprocessing",
                "classification_preprocessing": (
                    classification_mode
                ),
                "embedded_numeric_path": (
                    ("dxcom_div255_then_build_onnx_imagenet_mean_std" if classification_mode == "imagenet_mean_std" else "dxcom_div255_only")
                    if task_norm == "classification"
                    else "dxcom_div255_detection_default"
                ),
                "task": task_norm,
                "preprocess_mode_requested": "auto",
                "preprocess_mode": preprocess_mode,
                "preprocess_mode_effective": preprocess_mode,
                "letterbox_pad_value_requested": 114,
                "letterbox_pad_value_effective": letterbox_pad_value,
                "letterbox_pad_value": letterbox_pad_value,
                "contract_source": "task_bound_dxcom_image_loader_preprocessing",
            },
            "source_model_input": {
                "name": input_name,
                "shape": [int(x) for x in list(input_shape or [])],
                "layout": "NCHW",
            },
            "postprocessing": {"nms_on_host": True},
            "classification_preprocessing": (
                classification_mode
            ),
            "full_cache_contract_mode": "manual_mean_std_adapter" if classification_mode == "imagenet_mean_std" else "legacy_implicit",
            **({"build_onnx_adapter_receipt": adapter_receipt} if classification_mode == "imagenet_mean_std" else {}),
            **({"scientific_claim_exclusion_reason": "deepx_legacy_classification_preprocessing", "diagnostic_only": True} if classification_mode == "current_scale_only" else {}),
        }
        contract_path = status_dir / "output_contract.json"
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        if callable(log):
            log(f"[deepx] output contract: {contract_path}")
            log(f"[deepx] build status: {payload.get('build_status') or payload.get('status')}")
        payload.update({"status": "ok", "dxnn_path": str(dst), "output_contract": contract})
    except Exception as exc:
        payload.update({"status": "exception", "message": f"{type(exc).__name__}: {exc}"})

    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _deepx_onnx_io_contract(onnx_path: str | Path) -> tuple[str, list[int], list[str]]:
    """Return (first_input_name, first_input_shape, output_names) for a split ONNX."""
    p = Path(str(onnx_path)).expanduser()
    try:
        import onnx  # type: ignore
        m = onnx.load(str(p), load_external_data=False)
        init = {str(x.name) for x in getattr(m.graph, 'initializer', []) if getattr(x, 'name', None)}
        inputs = [i for i in list(getattr(m.graph, 'input', []) or []) if str(i.name) not in init]
        inp = inputs[0] if inputs else (list(getattr(m.graph, 'input', []) or [])[0])
        name = str(inp.name or 'images')
        dims: list[int] = []
        for d in inp.type.tensor_type.shape.dim:
            if getattr(d, 'dim_value', 0):
                dims.append(int(d.dim_value))
            else:
                dims.append(1 if len(dims) == 0 else 0)
        if len(dims) == 4:
            dims = [int(dims[0] or 1), int(dims[1] or 3), int(dims[2] or 224), int(dims[3] or 224)]
        outs = [str(o.name) for o in list(getattr(m.graph, 'output', []) or []) if str(o.name)]
        return name, dims or [1, 3, 224, 224], outs
    except Exception:
        return 'images', [1, 3, 224, 224], []


def _materialize_manual_deepx_part1_artifacts(
    *,
    out_dir: Path,
    bench_plan_runs: list[dict[str, Any]],
    validation_images: str,
    fallback_calib_dir: Optional[str],
    calibration_num: int = 0,
    task_hint: str = '',
    calibration_manifest: str = '',
    strict_calibration_source: bool = False,
    classification_preprocessing: Optional[str] = None,
    force_build: bool = False,
    selected_case_dirs: Optional[list[str]] = None,
    log: Any = None,
    build_config: Optional[Mapping[str, Any]] = None,
    profile_payload: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build/reuse DeepX Part1 DXNNs for DeepX->TensorRT split rows.

    This is the missing artifact path for the first executable DeepX split:
    ``part1.onnx -> part1.dxnn`` while ``part2.onnx`` remains an ORT/TensorRT
    model.  DeepX-as-Stage2 still goes through activation-proxy/tensor-loader
    experiments.
    """
    force_build = parse_config_bool(force_build, field="deepx_build.force_build")
    from ..build_dispatch_policy import require_productive_force_off
    require_productive_force_off(profile_payload or {})
    require_productive_force_off({"deepx_build": dict(build_config or {})})
    require_productive_force_off({"deepx_build": {"force_build": force_build}})

    def _is_deepx_stage1(run: Mapping[str, Any]) -> bool:
        # Only build per-case Part1 DXNNs for real DeepX->host split rows.
        # A full DeepX baseline also has stage1=deepx internally, but it does
        # not need bXXX/deepx/deepx_m1/part1/model.dxnn artifacts.
        if not isinstance(run, Mapping):
            return False
        rid = str(run.get('id') or '').strip().lower().replace('-', '_')
        if rid.startswith('deepx_m1_to') or rid.startswith('deepx_to') or rid.startswith('dx_m1_to'):
            return True
        typ = str(run.get('type') or '').strip().lower().replace('-', '_')
        if typ not in {'matrix', 'split', 'mixed_backend'}:
            return False
        st = run.get('stage1')
        st2 = run.get('stage2')
        if isinstance(st, Mapping):
            tok = str(st.get('type') or st.get('backend') or st.get('provider') or '').strip().lower().replace('-', '_')
        else:
            tok = str(st or '').strip().lower().replace('-', '_')
        if isinstance(st2, Mapping):
            tok2 = str(st2.get('type') or st2.get('backend') or st2.get('provider') or '').strip().lower().replace('-', '_')
        else:
            tok2 = str(st2 or '').strip().lower().replace('-', '_')
        return tok in {'deepx', 'deepx_m1', 'dx_m1', 'dxm1'} and tok2 not in {'deepx', 'deepx_m1', 'dx_m1', 'dxm1'}

    selected = any(_is_deepx_stage1(r) for r in list(bench_plan_runs or []) if isinstance(r, Mapping))
    model_label = (
        out_dir.parent.name
        if out_dir.name in {'benchmark_set', 'legacy_suite'}
        else out_dir.name
    )
    summary: dict[str, Any] = {
        'schema': 'onnx-splitpoint/manual-deepx-part1-artifacts-status',
        'schema_version': 1,
        'selected': bool(selected),
        'backend': 'deepx_m1',
        'variant': 'part1',
        'model_id': str(model_label),
        'cases': [],
        'status': 'not_selected',
    }
    suite_status_dir = out_dir / 'deepx' / 'deepx_m1' / 'part1'
    suite_status_dir.mkdir(parents=True, exist_ok=True)
    status_path = suite_status_dir / 'deepx_part1_artifact_status.json'
    if not selected:
        status_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        return summary

    if profile_payload is not None:
        from ..deepx.env_status import profile_compiler_configuration
        build_config = {**profile_compiler_configuration(profile_payload), **dict(build_config or {})}

    task_norm = str(task_hint or '').strip().lower()
    if task_norm not in {'classification', 'detection'}:
        summary.update({
            'status': 'task_contract_missing',
            'message': 'DeepX Part1 build requires an explicit classification or detection task.',
        })
        status_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        return summary

    try:
        from ..deepx.artifacts import (
            cache_dxnn_artifact,
            deepx_cache_key,
            deepx_cached_artifact_compatible,
            sha256_file,
            strict_artifact_sha256,
        )
        from ..deepx.compiler import compile_dxnn
        from ..deepx.config import (
            CLASSIFICATION_PREPROCESSING_CURRENT,
            CLASSIFICATION_PREPROCESSING_IMAGENET,
            canonical_classification_preprocessing,
            resolve_profile_classification_preprocessing,
            classification_profile_admission,
            deepx_classification_preprocessing_contract,
            image_model_dxcom_config,
            materialize_imagenet_normalized_build_onnx,
            write_dxcom_config,
        )
        from ..deepx.env_status import inspect_deepx_environment
    except Exception as exc:
        summary.update({'status': 'deepx_modules_unavailable', 'error': f'{type(exc).__name__}: {exc}'})
        status_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        return summary

    try:
        classification_mode = (
            resolve_profile_classification_preprocessing(
                {**dict(build_config or {}), **({"classification_preprocessing": classification_preprocessing} if classification_preprocessing is not None else {})}
            )
            if task_norm == 'classification'
            else 'not_applicable'
        )
    except ValueError as exc:
        summary.update({
            'status': 'classification_preprocessing_invalid',
            'classification_preprocessing': str(
                classification_preprocessing or ''
            ),
            'message': str(exc),
        })
        status_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        return summary
    summary['classification_preprocessing'] = classification_mode
    summary['scientific_claim_exclusion_reason'] = 'deepx_legacy_classification_preprocessing' if classification_mode == 'current_scale_only' else ''
    summary['diagnostic_only'] = classification_mode == 'current_scale_only'
    if profile_payload is not None:
        admission_profile = dict(profile_payload)
        admission_profile['deepx_build'] = {**dict(profile_payload.get('deepx_build') or {}), **dict(build_config or {}), 'classification_preprocessing': classification_mode if task_norm == 'classification' else 'imagenet_mean_std'}
        admission = classification_profile_admission(admission_profile, {'task': task_norm})
        summary['classification_admission'] = admission
        if not admission['allowed']:
            summary.update(status=admission['reason'], message=admission['required_setting'], compiler_dispatched=False)
            status_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
            return summary

    cache_verify_only = compiler_dispatch_forbidden()
    summary['artifact_policy'] = (
        'cache_verify_only' if cache_verify_only else 'normal'
    )
    env = inspect_deepx_environment(probe_import=False, path_only=True, config=build_config)
    summary['environment_status'] = env
    if cache_verify_only and parse_config_bool(force_build, field="deepx_build.force_build"):
        summary.update({
            'status': 'cache_miss_blocked',
            'message': cache_miss_blocked_message(
                'deepx_dx_com', 'manual Part1 force_build requested'
            ),
        })
        status_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        return summary
    cache_root = Path(str(env.get('cache_dir') or '~/Models/BackendArtifacts/deepx')).expanduser()
    # Cache identity must not depend on whether an earlier case in this batch
    # required an import/operations probe. Readiness observations may change.
    cache_environment = dict(env)
    case_manifests = sorted(out_dir.glob('b*/split_manifest.json'))
    if selected_case_dirs is not None:
        selected_folders = set(selected_case_dirs)
        case_manifests = [path for path in case_manifests if path.parent.name in selected_folders]
    if callable(log):
        log(f'[deepx] Part1 DXNN build selected for DeepX->TensorRT; cases={len(case_manifests)}')
    ok_count = 0
    fail_count = 0
    for manifest_path in case_manifests:
        case_dir = manifest_path.parent
        case_id = case_dir.name
        case_payload: dict[str, Any] = {'case_id': case_id, 'manifest': str(manifest_path)}
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            p1_rel = str(manifest.get('part1_model') or manifest.get('part1') or manifest.get('part1_path') or '').strip()
            if not p1_rel:
                raise FileNotFoundError('split manifest does not contain part1_model')
            p1 = Path(p1_rel)
            if not p1.is_absolute():
                p1 = case_dir / p1
            if not p1.is_file():
                raise FileNotFoundError(f'part1 ONNX missing: {p1}')
            input_name, input_shape, output_names = _deepx_onnx_io_contract(p1)
            calib_dir = _manual_deepx_effective_calib_dir(
                out_dir,
                "" if strict_calibration_source else validation_images,
                fallback_calib_dir,
                task_hint=task_hint,
                model_path=str(p1),
                input_shape=list(input_shape or []),
            )
            if strict_calibration_source and fallback_calib_dir:
                expected_root = Path(str(fallback_calib_dir)).expanduser().resolve()
                actual_root = Path(str(calib_dir or '')).expanduser().resolve() if calib_dir else None
                compatible = False
                if actual_root is not None:
                    try:
                        compatible = actual_root == expected_root or expected_root in actual_root.parents or actual_root in expected_root.parents
                    except Exception:
                        compatible = False
                if not compatible:
                    raise RuntimeError(
                        f"task-specific DeepX calibration source mismatch: task={task_hint or 'auto'} "
                        f"expected={expected_root} resolved={actual_root}"
                    )
            case_payload.update({'part1_onnx': str(p1), 'input_name': input_name, 'input_shape': input_shape, 'output_names': output_names, 'calibration_dir': calib_dir, 'calibration_num': int(calibration_num or 100)})
            if callable(log):
                log(f'[deepx] {case_id}: part1 ONNX={p1.name} input={input_name}{input_shape} outputs={len(output_names)}')
                log(f'[deepx] {case_id}: part1 calibration dir: {calib_dir or "<missing>"}')
            if not calib_dir:
                raise FileNotFoundError('no calibration image directory available for DeepX Part1')
            stage_dir = case_dir / 'deepx' / 'deepx_m1' / 'part1'
            stage_dir.mkdir(parents=True, exist_ok=True)
            source_onnx_sha256 = strict_artifact_sha256(p1)
            build_onnx = p1
            adapter_receipt: dict[str, Any] = {
                'schema': 'onnx-splitpoint/deepx-build-onnx-adapter',
                'schema_version': 1,
                'kind': 'identity',
                'source_onnx_sha256': source_onnx_sha256,
                'build_onnx_sha256': source_onnx_sha256,
            }
            adapter_receipt_path: Path | None = None
            if (
                task_norm == 'classification'
                and classification_mode
                == CLASSIFICATION_PREPROCESSING_IMAGENET
            ):
                build_onnx, adapter_receipt = (
                    materialize_imagenet_normalized_build_onnx(
                        source_onnx=p1,
                        output_dir=stage_dir / 'build_onnx',
                        input_name=input_name,
                    )
                )
                adapter_receipt_path = (
                    stage_dir / 'build_onnx_adapter_receipt.json'
                )
                adapter_receipt_path.write_text(
                    json.dumps(
                        adapter_receipt, indent=2, ensure_ascii=False,
                    ) + '\n',
                    encoding='utf-8',
                )
            build_onnx_sha256 = strict_artifact_sha256(build_onnx)
            preprocessing_contract = (
                deepx_classification_preprocessing_contract(
                    classification_mode
                )
                if task_norm == 'classification'
                else {
                    'schema': 'onnx-splitpoint/deepx-detection-preprocessing',
                    'schema_version': 1,
                    'mode': 'letterbox114_div255',
                }
            )
            cfg = image_model_dxcom_config(
                task=task_norm,
                input_name=input_name,
                input_shape=input_shape,
                calibration_dir=calib_dir,
                calibration_num=max(1, int(calibration_num or 100)),
                calibration_method='ema',
                classification_preprocessing=(
                    classification_mode
                    if task_norm == 'classification'
                    else CLASSIFICATION_PREPROCESSING_CURRENT
                ),
            )
            cfg_path = write_dxcom_config(cfg, stage_dir / 'config_deepx.json')
            manifest_path_obj = Path(str(calibration_manifest or '')).expanduser() if str(calibration_manifest or '').strip() else None
            manifest_identity = ''
            if manifest_path_obj is not None and manifest_path_obj.is_file():
                manifest_identity = 'sha256:' + sha256_file(manifest_path_obj)
            elif str(calibration_manifest or '').strip():
                manifest_identity = 'missing:' + str(manifest_path_obj)
            elif cache_verify_only:
                case_payload.update({
                    'ok': False,
                    'status': 'cache_miss_blocked',
                    'error': cache_miss_blocked_message(
                        'deepx_dx_com',
                        f'manual Part1 exact cache miss for {case_id}',
                    ),
                })
                fail_count += 1
                summary['cases'].append(case_payload)
                if callable(log):
                    log(
                        f'[deepx] {case_id}: Part1 cache miss blocked by '
                        'cache_verify_only',
                        level=logging.WARNING,
                    )
                continue
            else:
                # The directory is retained as a fallback identity for manual
                # suites, but EvaluationRuns normally provide a content-addressed
                # manifest.
                manifest_identity = 'directory:' + str(Path(calib_dir).expanduser().resolve())
            compiler_identity = str(
                cache_environment.get('compiler_version')
                or cache_environment.get('compiler_cli_version')
                or cache_environment.get('compiler_python_tag')
                or cache_environment.get('compiler_cli')
                or 'unknown'
            )
            cache_contract = {
                'schema': 'onnx-splitpoint/deepx-part1-cache-contract',
                'schema_version': 2,
                'task': task_norm,
                'calibration_manifest_identity': manifest_identity,
                'calibration_count': max(1, int(calibration_num or 100)),
                'calibration_method': 'ema',
                'calibration_dir': str(Path(calib_dir).expanduser().resolve()),
                'preprocessing_contract': ('dxcom_image_loader_v2:resize-pad0-rgb-hwc' if task_norm == 'classification' else 'dxcom_image_loader_v2:letterbox114-rgb-hwc'),
                'compiler_identity': compiler_identity,
                'input_name': str(input_name),
                'input_shape': [int(x) for x in list(input_shape or [])],
                'case_id': str(case_id),
            }
            if (
                task_norm == 'classification'
                and classification_mode
                == CLASSIFICATION_PREPROCESSING_IMAGENET
            ):
                # Only the corrected arm extends the existing Part1 cache
                # contract.  Detection and the historical scale-only arm keep
                # their established keys and do not incur unrelated rebuilds.
                cache_contract.update({
                    'classification_preprocessing': classification_mode,
                    'classification_preprocessing_contract': (
                        preprocessing_contract
                    ),
                    'source_onnx_sha256': source_onnx_sha256,
                    'build_onnx_sha256': build_onnx_sha256,
                })
            key = deepx_cache_key(
                onnx_path=build_onnx,
                config_path=cfg_path,
                target='deepx_m1',
                variant=f'part1_{case_id}',
                cache_contract=cache_contract,
            )
            cache_dir = cache_root / key
            cached = cache_dir / 'model.dxnn'
            cache_ok, cache_reason, _cache_manifest = deepx_cached_artifact_compatible(
                cache_dir=cache_dir,
                expected_contract=cache_contract,
                require_artifact_identity=True,
            )
            case_payload.update({
                'config_path': str(cfg_path),
                'cache_key': key,
                'cache_root': str(cache_root),
                'cache_contract': cache_contract,
                'cache_compatible': bool(cache_ok),
                'cache_compatibility_reason': str(cache_reason),
                'calibration_task': task_norm,
                'calibration_manifest': str(calibration_manifest or ''),
                'calibration_manifest_identity': manifest_identity,
                'classification_preprocessing': classification_mode,
                'source_onnx_sha256': source_onnx_sha256,
                'build_onnx_sha256': build_onnx_sha256,
                'build_onnx': str(build_onnx),
                'build_onnx_adapter_receipt': (
                    str(adapter_receipt_path)
                    if adapter_receipt_path is not None else ''
                ),
            })
            if cache_ok and not parse_config_bool(force_build, field="deepx_build.force_build"):
                dxnn = cached
                build_status = 'ready_reused'
                cache_lookup = {
                    'schema': 'onnx-splitpoint/deepx-cache-lookup',
                    'schema_version': 1,
                    'outcome': 'HIT',
                    'role': 'part1',
                    'model_id': str(model_label),
                    'case_id': str(case_id),
                    'identity': str(key),
                    'reason': str(cache_reason),
                    'artifact': str(cached),
                }
                case_payload['cache_lookup'] = cache_lookup
                if callable(log):
                    log(
                        f'[deepx-cache] HIT role=part1 model={model_label} '
                        f'case={case_id} identity={key} reason={cache_reason} '
                        f'artifact={dxnn}'
                    )
            else:
                miss_reason = (
                    'force_build_requested'
                    if parse_config_bool(force_build, field="deepx_build.force_build") else str(cache_reason or 'not_found')
                )
                cache_lookup = {
                    'schema': 'onnx-splitpoint/deepx-cache-lookup',
                    'schema_version': 1,
                    'outcome': 'MISS',
                    'role': 'part1',
                    'model_id': str(model_label),
                    'case_id': str(case_id),
                    'identity': str(key),
                    'reason': miss_reason,
                    'artifact': str(cached) if cached.is_file() else '',
                }
                case_payload['cache_lookup'] = cache_lookup
                if callable(log):
                    miss_line = (
                        f'[deepx-cache] MISS role=part1 model={model_label} '
                        f'case={case_id} identity={key} reason={miss_reason}'
                    )
                    if cache_lookup['artifact']:
                        miss_line += f" artifact={cache_lookup['artifact']}"
                    log(miss_line)
                if cached.is_file() and callable(log):
                    log(f'[deepx] {case_id}: cached Part1 DXNN rejected ({cache_reason}); rebuilding with task={task_norm} calibration={manifest_identity}')
                if not cache_verify_only:
                    env = inspect_deepx_environment(probe_import=True, config=build_config)
                    summary['environment_status'] = env
                if not cache_verify_only and not bool(env.get('compiler_ready')):
                    message = "; ".join(str(hint) for hint in env.get('hints') or []) or 'DeepX DX-COM compiler environment is not ready.'
                    case_payload['status'] = 'compiler_not_ready'
                    raise RuntimeError(f'compiler_not_ready: {message}')
                if callable(log):
                    log(f'[deepx] {case_id}: building Part1 DXNN via dxcom')
                res = compile_dxnn(
                    onnx_path=build_onnx,
                    config_path=cfg_path,
                    output_dir=stage_dir / 'build',
                    compiler_root=env.get('dx_all_suite_root'),
                    compiler_venv=env.get('compiler_venv'),
                    compiler_overlay=env.get('compiler_overlay'), build_config=build_config,
                    opt_level=0,
                    timeout_s=7200,
                )
                case_payload['build_result'] = getattr(res, '__dict__', {})
                try:
                    lp = Path(str(getattr(res, 'log_path', '') or ''))
                    if callable(log) and lp.is_file():
                        log(f'[deepx] {case_id}: part1 dxcom compiler log: {lp}')
                        lines = lp.read_text(encoding='utf-8', errors='replace').splitlines()
                        interesting = [ln for ln in lines if any(tok in ln.lower() for tok in ('error', 'warning', 'compile', 'compiling', 'final result', 'added nodes', 'skipped nodes', 'npu', 'cpu', 'dxnn'))]
                        for ln in (interesting[-16:] if interesting else lines[-16:]):
                            log(f'[deepx][part1][dxcom] {case_id}: {ln[:500]}')
                except Exception:
                    pass
                if not (res.ok and res.dxnn_path):
                    raise RuntimeError(str(res.message or res.status or 'DeepX Part1 compile failed')[-1200:])
                dxnn = cache_dxnn_artifact(
                    dxnn_path=res.dxnn_path,
                    manifest={
                        'schema': 'onnx-splitpoint/deepx-build-manifest',
                        'schema_version': 2,
                        'build_result': getattr(res, '__dict__', {}),
                        'cache_key': key,
                        'variant': 'part1',
                        'case_id': case_id,
                        'cache_contract': cache_contract,
                    },
                    cache_root=cache_root,
                    cache_key=key,
                )
                build_status = 'ready_built'
            dst = stage_dir / 'model.dxnn'
            shutil.copy2(dxnn, dst)
            dxnn_sha256 = strict_artifact_sha256(dst)
            dxnn_size_bytes = int(dst.stat().st_size)
            preprocess_mode = 'resize' if task_norm == 'classification' else 'letterbox'
            letterbox_pad_value = 0 if preprocess_mode == 'resize' else 114
            if len(input_shape) >= 4 and int(input_shape[1]) in (1, 3, 4):
                runtime_h, runtime_w = int(input_shape[2]), int(input_shape[3])
            elif len(input_shape) >= 4:
                runtime_h, runtime_w = int(input_shape[1]), int(input_shape[2])
            else:
                runtime_h = runtime_w = 224 if task_norm == 'classification' else 640
            contract = {
                'backend': 'deepx_m1',
                'variant': 'part1',
                'artifact_kind': 'dxnn',
                'artifact': f'{case_id}/deepx/deepx_m1/part1/model.dxnn',
                'input': {
                    'name': input_name,
                    'shape': [runtime_h, runtime_w, 3],
                    'dtype': 'uint8',
                    'layout': 'HWC',
                    'color_space': 'RGB',
                    'normalization': 'embedded_dxcom_preprocessing',
                    'classification_preprocessing': classification_mode,
                    'embedded_numeric_path': (
                        'dxcom_div255_then_build_onnx_imagenet_mean_std'
                        if task_norm == 'classification'
                        and classification_mode
                        == CLASSIFICATION_PREPROCESSING_IMAGENET
                        else 'dxcom_div255_only'
                        if task_norm == 'classification'
                        else 'dxcom_div255_detection_default'
                    ),
                    'task': task_norm,
                    'preprocess_mode_requested': 'auto',
                    'preprocess_mode': preprocess_mode,
                    'preprocess_mode_effective': preprocess_mode,
                    'letterbox_pad_value_requested': 114,
                    'letterbox_pad_value_effective': letterbox_pad_value,
                    'letterbox_pad_value': letterbox_pad_value,
                    'contract_source': 'task_bound_dxcom_image_loader_preprocessing',
                },
                'source_model_input': {
                    'name': input_name,
                    'shape': [int(x) for x in list(input_shape or [])],
                    'layout': 'NCHW',
                },
                'output_names': list(output_names),
                'outputs': [{'name': n} for n in output_names],
                'stage': 'part1',
                **({'scientific_claim_exclusion_reason': summary['scientific_claim_exclusion_reason'], 'diagnostic_only': True} if summary['diagnostic_only'] else {}),
                'case_id': case_id,
                'source_onnx_sha256': source_onnx_sha256,
                'build_onnx_sha256': build_onnx_sha256,
                'classification_preprocessing': classification_mode,
                'preprocessing_contract': preprocessing_contract,
                'build_onnx_adapter': {
                    key: value
                    for key, value in adapter_receipt.items()
                    if key != 'build_onnx_path'
                },
                'artifact_sha256': dxnn_sha256,
                'artifact_size_bytes': dxnn_size_bytes,
            }
            contract_path = stage_dir / 'output_contract.json'
            contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding='utf-8')
            output_contract_sha256 = strict_artifact_sha256(contract_path)
            status = {
                'ok': True,
                'status': 'ok',
                'build_status': build_status,
                'dxnn_path': os.path.relpath(dst, case_dir).replace('\\', '/'),
                'config_path': os.path.relpath(cfg_path, case_dir).replace('\\', '/'),
                'output_contract': os.path.relpath(contract_path, case_dir).replace('\\', '/'),
                'output_contract_sha256': output_contract_sha256,
                'source_onnx_sha256': source_onnx_sha256,
                'build_onnx_sha256': build_onnx_sha256,
                'build_onnx': (
                    os.path.relpath(build_onnx, case_dir).replace('\\', '/')
                ),
                'build_onnx_adapter_receipt': (
                    os.path.relpath(
                        adapter_receipt_path, case_dir,
                    ).replace('\\', '/')
                    if adapter_receipt_path is not None else ''
                ),
                'dxnn_sha256': dxnn_sha256,
                'dxnn_size_bytes': dxnn_size_bytes,
                'cache_key': key,
                'cache_schema_version': 2,
                'cache_contract': cache_contract,
                'calibration_task': task_norm,
                'calibration_manifest': str(calibration_manifest or ''),
                'calibration_manifest_identity': manifest_identity,
                'classification_preprocessing': classification_mode,
                'preprocessing_contract': preprocessing_contract,
                'output_names': list(output_names),
                'cache_lookup': dict(case_payload.get('cache_lookup') or {}),
            }
            (stage_dir / 'deepx_part1_artifact_status.json').write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
            manifest.setdefault('deepx', {})
            manifest['deepx']['part1_dxnn'] = dict(status)
            manifest['deepx_part1_dxnn'] = status['dxnn_path']
            manifest['deepx_part1_output_contract'] = status['output_contract']
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
            case_payload.update(status)
            ok_count += 1
            if callable(log):
                try:
                    log(f'[deepx] {case_id}: Part1 DXNN ready ({dst}, {dst.stat().st_size/1024/1024:.1f} MiB)')
                except Exception:
                    log(f'[deepx] {case_id}: Part1 DXNN ready ({dst})')
        except Exception as exc:
            fail_count += 1
            case_payload.update({'ok': False, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            if callable(log):
                log(f'[deepx] {case_id}: Part1 DXNN failed: {type(exc).__name__}: {exc}', level=logging.WARNING)
        summary['cases'].append(case_payload)
    all_cache_misses = bool(summary['cases']) and all(
        str(row.get('status') or '') == 'cache_miss_blocked'
        for row in summary['cases']
    )
    summary.update({
        'status': (
            'ok' if ok_count and not fail_count
            else 'partial' if ok_count
            else 'cache_miss_blocked' if cache_verify_only and all_cache_misses
            else 'failed'
        ),
        'ok_count': ok_count,
        'failed_count': fail_count,
    })
    status_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary


class BenchmarkWorkflowController:
    def __init__(self, app: Any):
        self.app = app

    def generate_benchmark_set(
        self,
        *,
        resume_dir: Optional[str] = None,
        offer_latest_resume: bool = True,
        output_parent_override: Optional[str] = None,
        completion_callback=None,
        show_result_dialogs: bool = True,
    ) -> Optional[str]:
        """Generate or resume a benchmark suite folder for the current model + top-k picks.

        The suite contains one subfolder per split candidate with:
          - part1 / part2 ONNX models
          - split_manifest.json
          - run_split_onnxruntime.py runner script

        At the top level it also contains:
          - benchmark_set.json (list of cases + predicted metrics)
          - benchmark_suite.py (runs all cases and collects results/plots)
        """
        app = self.app
        model_path = app.gui_state.current_model_path or app.model_path
        if app.analysis is None or model_path is None:
            messagebox.showinfo("Nothing to benchmark", "Load a model and run an analysis first.")
            return
        if bool(getattr(app, "_benchmark_generation_active", False)):
            messagebox.showinfo(
                "Benchmark set already running",
                "A benchmark-set generation is already running in the background.\n\n"
                "You can start one remote benchmark in parallel, but only one benchmark-set generation at a time.",
            )
            return
        candidate_pool: List[int] = list(app._benchmark_candidate_pool())
        if not candidate_pool:
            messagebox.showinfo(
                "No candidates",
                "No split candidates available. Try increasing Top-K and re-run Analyse.",
            )
            return

        initial_out = app.default_output_dir or os.path.dirname(model_path)
        try:
            if app.default_output_dir:
                initial_out = str(ensure_workdir(Path(app.default_output_dir)).benchmark_sets)
        except Exception:
            pass

        base = os.path.splitext(os.path.basename(model_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        resume_generation = False
        resume_state_hint: Optional[Dict[str, Any]] = None
        resume_report = None

        if resume_dir is None:
            if output_parent_override is not None:
                _op = Path(output_parent_override).expanduser()
                if not _op.is_absolute():
                    try:
                        _root = Path(getattr(app, "default_output_dir", ".") or ".").expanduser()
                        _root = ensure_workdir(_root).root
                        _op = _root / _op
                    except Exception:
                        _op = Path.cwd() / _op
                out_parent = str(_op)
                if not out_parent:
                    return None
            else:
                out_parent = filedialog.askdirectory(title="Select parent folder for benchmark set", initialdir=initial_out)
                if not out_parent:
                    return None
            out_dir = os.path.join(out_parent, f"{base}_benchmark_{ts}")
        else:
            out_dir = str(Path(resume_dir))
            out_parent = str(Path(out_dir).parent)
            resume_generation = True
            try:
                resume_state_hint = read_generation_json(Path(out_dir) / "generation_state.json", default={}) or {}
            except Exception:
                resume_state_hint = {}

        # Pull analysis objects once (used for strict-boundary filtering and TeX/plot export).
        a = app.analysis
        strict_boundary = bool(app.var_strict_boundary.get())
        generation_service = getattr(app, "_benchmark_generation_service", BenchmarkGenerationService())
        evaluation_profile_request = str(getattr(app, 'var_bench_evaluation_profile', tk.StringVar(value='')).get() or '').strip()
        evaluation_profile_resolution = None
        if evaluation_profile_request:
            try:
                evaluation_profile_resolution = resolve_evaluation_profile(evaluation_profile_request, model_path=model_path)
            except Exception:
                logger.exception('Failed to resolve benchmark evaluation profile %s', evaluation_profile_request)
                evaluation_profile_resolution = None

        try:
            prep_mode = normalize_model_preparation_mode(getattr(app, 'var_bench_model_preparation_mode', tk.StringVar(value='Use current ONNX')).get())
            export_meta = load_export_metadata_for_model(model_path)
            task_type = str(export_meta.get('task_type') or '').strip().lower()
            source = str(export_meta.get('source') or '').strip().lower()
            if prep_mode == 'screen_yolo_full_hailo' and source == 'ultralytics' and task_type == 'detection' and not preparation_result_is_selected_model(model_path):
                proceed = messagebox.askyesno(
                    'Model preparation recommended',
                    'This YOLO ONNX has not been screened for a full-Hailo-capable export variant yet.\n\n'
                    'For the manual workflow, use “Prepare current model…” first and then rerun Analyse on the selected prepared ONNX.\n\n'
                    'Do you want to continue anyway with the current ONNX?'
                )
                if not proceed:
                    return None
        except Exception:
            logger.debug('Failed to evaluate manual model-preparation guardrail', exc_info=True)

        # Read pruning params from the current GUI state (same source as Analyse).
        _params_for_split = app._read_params()
        prune_skip_block = bool(getattr(_params_for_split, "prune_skip_block", False))
        skip_min_span = int(getattr(_params_for_split, "skip_min_span", 0) or 0)
        if skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = int(getattr(_params_for_split, "skip_allow_last_n", 0) or 0)
        if skip_allow_last_n < 0:
            raise ValueError("Allow last N inside must be an integer ≥ 0.")

        # Convenience locals used during strict-boundary validation.
        model = a.get("model") if isinstance(a, dict) else None
        nodes = a.get("nodes") if isinstance(a, dict) else None
        order = a.get("order") if isinstance(a, dict) else None
        if model is None or nodes is None or order is None:
            messagebox.showerror(
                "Benchmark set failed",
                "Internal error: analysis data missing (model/nodes/order). Please re-run Analyse.",
            )
            return

        # How many candidates to export / inspect?
        #
        # ``ranked_candidates`` is the preferred shortlist (typically the currently
        # displayed Analyse picks). ``candidate_search_pool`` extends that shortlist
        # with broader ranked candidates so benchmark generation can backfill when
        # some splits are rejected later (for example due to Hailo Part2 activation
        # calibration depending on the original ``images`` input).

        ranked_candidates: List[int] = list(candidate_pool)

        def _strict_filter_boundaries(raw_bounds: List[int]) -> List[int]:
            try:
                return generation_service.strict_filter_boundaries(a, raw_bounds)
            except Exception:
                logger.debug("Strict boundary filtering via BenchmarkGenerationService failed", exc_info=True)
                return []

        # If the user enabled "Strict boundary" AFTER running Analyse (or the
        # analysis was done without strict-boundary metadata), re-check strictness
        # defensively for both the preferred shortlist and the broader backfill pool.
        if strict_boundary:
            ranked_candidates = _strict_filter_boundaries(ranked_candidates)
            if not ranked_candidates:
                messagebox.showinfo(
                    "No strict candidates",
                    "Strict boundary is enabled, but none of the available candidates satisfy the strict-boundary condition.\n\n"
                    "Tip: disable Strict boundary or re-run Analyse with Strict boundary unchecked.",
                )
                return

        candidate_search_pool: List[int] = list(app._benchmark_candidate_search_pool(ranked_candidates))
        if strict_boundary:
            candidate_search_pool = _strict_filter_boundaries(candidate_search_pool)

        if not candidate_search_pool:
            messagebox.showinfo(
                "No candidates",
                "No benchmark-export candidates are available after filtering.\n\n"
                "Tip: increase Top-K, disable Strict boundary, or re-run Analyse.",
            )
            return

        benchmark_selection_strategy = str(
            getattr(app, "var_bench_selection_strategy", tk.StringVar(value="rank_order")).get() or "rank_order"
        ).strip().lower().replace("-", "_")
        benchmark_compile_aware_ordering = bool(
            getattr(app, "var_bench_compile_aware_ordering", tk.BooleanVar(value=False)).get()
        )

        def _manual_stratified_order(preferred_bounds: List[int], pool_bounds: List[int], requested_count: int) -> tuple[List[int], List[Dict[str, Any]]]:
            """Spread manual BenchmarkSet picks over boundary windows.

            The Analyse table remains the scoring source; this function only chooses
            one high-ranked candidate per early/mid/late window before appending the
            normal ranked order as backfill.  It is deliberately boundary-index based
            so it works even when no richer graph metadata is available.
            """
            rank_order: List[int] = []
            seen_rank: set[int] = set()
            for raw_list in (preferred_bounds, pool_bounds):
                for raw_b in list(raw_list or []):
                    try:
                        bi = int(raw_b)
                    except Exception:
                        continue
                    if bi in seen_rank:
                        continue
                    seen_rank.add(bi)
                    rank_order.append(bi)
            if not rank_order:
                return list(pool_bounds or []), []
            pool_unique: List[int] = []
            seen_pool: set[int] = set()
            for raw_b in list(pool_bounds or rank_order):
                try:
                    bi = int(raw_b)
                except Exception:
                    continue
                if bi in seen_pool:
                    continue
                seen_pool.add(bi)
                pool_unique.append(bi)
            if len(pool_unique) <= max(1, int(requested_count or 1)):
                return list(rank_order), []
            rank_pos = {int(b): idx for idx, b in enumerate(rank_order)}
            by_boundary = sorted(pool_unique)
            lo = min(by_boundary)
            hi = max(by_boundary)
            n_windows = max(1, min(int(requested_count or 1), len(by_boundary)))
            span = max(1.0, float(hi - lo + 1))
            selected: List[int] = []
            audit: List[Dict[str, Any]] = []
            used: set[int] = set()
            for wi in range(n_windows):
                w_lo = lo + int((span * wi) / n_windows)
                w_hi = lo + int((span * (wi + 1)) / n_windows) - 1
                if wi == n_windows - 1:
                    w_hi = hi
                candidates = [b for b in by_boundary if b not in used and w_lo <= int(b) <= w_hi]
                if not candidates:
                    audit.append({"window": wi, "lo": w_lo, "hi": w_hi, "selected": None, "candidate_count": 0})
                    continue
                best = min(candidates, key=lambda b: (rank_pos.get(int(b), 10**9), int(b)))
                selected.append(best)
                used.add(best)
                audit.append({"window": wi, "lo": w_lo, "hi": w_hi, "selected": int(best), "candidate_count": len(candidates)})
            ordered = list(selected)
            for b in rank_order:
                if b not in used:
                    ordered.append(b)
                    used.add(b)
            for b in by_boundary:
                if b not in used:
                    ordered.append(b)
                    used.add(b)
            return ordered, audit

        profile_overrides = dict(evaluation_profile_resolution.overrides or {}) if (evaluation_profile_resolution is not None and evaluation_profile_resolution.matched) else {}
        require_single_part2_input = bool(
            getattr(
                app,
                "var_bench_require_single_part2_input",
                tk.BooleanVar(value=False),
            ).get()
        )
        if "require_single_part2_input" in profile_overrides:
            require_single_part2_input = bool(
                profile_overrides.get("require_single_part2_input")
            )

        part2_input_filter_audit: List[Dict[str, Any]] = []

        def _filter_single_part2_inputs(
            boundaries: List[int],
        ) -> List[int]:
            if not require_single_part2_input:
                return list(boundaries)
            try:
                from ..split_export_graph import (
                    part2_external_inputs_for_boundary,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Part-2 input filter is unavailable: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            eligible: List[int] = []
            for raw_boundary in list(boundaries or []):
                boundary = int(raw_boundary)
                try:
                    input_names = part2_external_inputs_for_boundary(
                        model, order, nodes, boundary
                    )
                    count = len(input_names)
                    error = ""
                except Exception as exc:
                    input_names = []
                    count = None
                    error = f"{type(exc).__name__}: {exc}"
                if count == 1:
                    eligible.append(boundary)
                    continue
                part2_input_filter_audit.append({
                    "boundary": boundary,
                    "reason": "part2_input_count_not_one",
                    "part2_input_count": count,
                    "part2_input_names": list(input_names),
                    "error": error,
                })
            return eligible

        if require_single_part2_input:
            ranked_candidates = _filter_single_part2_inputs(ranked_candidates)
            candidate_search_pool = _filter_single_part2_inputs(
                candidate_search_pool
            )
            if not candidate_search_pool:
                messagebox.showinfo(
                    "No single-input Part-2 candidates",
                    "The Part-2 input count = 1 filter removed every available "
                    "candidate.\n\nGeneric execution still supports multiple "
                    "inputs; disable the checkbox to include those cases.",
                )
                return
        try:
            profile_shortlist = int(profile_overrides.get('preferred_shortlist') or 0)
        except Exception:
            profile_shortlist = 0
        if profile_shortlist > 0:
            ranked_candidates = list(ranked_candidates[:max(1, profile_shortlist)])
        pool_limit = profile_overrides.get('candidate_search_pool')
        try:
            pool_limit = int(pool_limit) if pool_limit is not None else 0
        except Exception:
            pool_limit = 0
        if int(pool_limit or 0) > 0:
            candidate_search_pool = list(candidate_search_pool[:max(1, int(pool_limit))])
        try:
            profile_requested_cases = int(profile_overrides.get('requested_cases') or 0)
        except Exception:
            profile_requested_cases = 0

        # How many cases to generate?
        # Prefer the Benchmark tab entry (var_bench_topk). Fall back to a dialog only
        # if it's missing/invalid. ``k`` is the target number of accepted cases;
        # backfilling can continue deeper into the ranked search pool if needed.
        default_k = min(profile_requested_cases or 20, len(candidate_search_pool))
        k = None
        try:
            k = _safe_int((getattr(app, "var_bench_topk", tk.StringVar(value=str(default_k))).get() or "").strip())
        except Exception:
            k = None
        if k is None:
            k = simpledialog.askinteger(
                "Benchmark set",
                f"How many splits to generate for the benchmark set? (target accepted cases, max {len(candidate_search_pool)}, preferred shortlist {len(ranked_candidates)})",
                initialvalue=default_k,
                minvalue=1,
                maxvalue=len(candidate_search_pool),
            )
        if k is None:
            return
        k = int(k)
        if profile_requested_cases > 0:
            k = min(max(1, int(profile_requested_cases)), len(candidate_search_pool))
        if k < 1 or k > len(candidate_search_pool):
            messagebox.showerror(
                "Benchmark set",
                f"Requested cases must be between 1 and {len(candidate_search_pool)} (search pool size).",
            )
            return

        manual_selection_audit: List[Dict[str, Any]] = []
        if benchmark_selection_strategy in {"stratified", "stratified_windows", "windowed", "coverage_windows"}:
            try:
                ordered, manual_selection_audit = _manual_stratified_order(ranked_candidates, candidate_search_pool, int(k))
                if ordered:
                    candidate_search_pool = list(ordered)
                    ranked_candidates = list(ordered[:max(int(k), len(ranked_candidates))])
            except Exception:
                logger.debug("Manual stratified benchmark selection failed; preserving rank order", exc_info=True)

        # Offer to resume the newest incomplete benchmark set for this model.
        if not resume_generation and bool(offer_latest_resume):
            try:
                resumable_dir = find_latest_resumable_set(Path(out_parent), base)
            except Exception:
                resumable_dir = None
            if resumable_dir is not None:
                try:
                    resume_state_candidate = read_generation_json(resumable_dir / "generation_state.json", default={}) or {}
                except Exception:
                    resume_state_candidate = {}
                asked = messagebox.askyesno(
                    "Resume benchmark set?",
                    "Found an incomplete benchmark set for this model:\n\n"
                    f"{resumable_dir}\n\n"
                    "Resume it and reuse already generated artefacts?",
                )
                if asked:
                    out_dir = str(resumable_dir)
                    resume_generation = True
                    if isinstance(resume_state_candidate, dict):
                        resume_state_hint = dict(resume_state_candidate)

        if resume_generation:
            try:
                resume_report = reconcile_generation_state(
                    Path(out_dir),
                    resume_state_hint if isinstance(resume_state_hint, dict) else {},
                )
                resume_state_hint = dict(resume_report.repaired_state)
            except Exception as e:
                proceed = messagebox.askyesno(
                    "Resume benchmark set",
                    "The benchmark-set consistency check failed:\n\n"
                    f"{type(e).__name__}: {e}\n\n"
                    "Continue with the raw state anyway?",
                )
                if not proceed:
                    return
            else:
                if resume_report.changed or resume_report.warnings:
                    proceed = messagebox.askyesno(
                        "Resume consistency check",
                        "The selected benchmark set was checked against the files on disk.\n\n"
                        f"{resume_report.summary() or 'No changes required.'}\n\n"
                        "Continue with the repaired state?",
                    )
                    if not proceed:
                        return

            if isinstance(resume_state_hint, dict):
                try:
                    ranked_candidates = [int(x) for x in (resume_state_hint.get("ranked_candidates") or ranked_candidates)]
                except Exception:
                    pass
                try:
                    candidate_search_pool = [int(x) for x in (resume_state_hint.get("candidate_search_pool") or candidate_search_pool)]
                except Exception:
                    pass
                try:
                    k = int(resume_state_hint.get("requested_cases") or k)
                except Exception:
                    pass

        if require_single_part2_input:
            ranked_candidates = _filter_single_part2_inputs(ranked_candidates)
            candidate_search_pool = _filter_single_part2_inputs(
                candidate_search_pool
            )
            if not candidate_search_pool:
                messagebox.showerror(
                    "Benchmark set",
                    "The resumed candidate pool contains no case satisfying "
                    "Part-2 input count = 1.",
                )
                return
        k = max(1, min(int(k), len(candidate_search_pool)))
        os.makedirs(out_dir, exist_ok=True)

        # Export analysis artefacts (plots + TeX table) into the benchmark folder for paper usage.
        # Keep this anchored to the preferred shortlist so the paper-facing exports
        # stay aligned with the user-visible Analyse ranking, while the actual
        # benchmark generation may backfill from deeper-ranked candidates.
        try:
            app._export_benchmark_paper_assets(Path(out_dir), a, ranked_candidates[:k])
        except Exception as e:
            print(f"[warn] Failed to export paper assets into benchmark folder: {type(e).__name__}: {e}")

        # For a benchmark set we ALWAYS generate a runner skeleton (otherwise the suite isn't runnable).
        do_runner = True

        # Runner skeleton target (auto/cpu/cuda/tensorrt). Read it once here so the
        # background worker does not access Tk variables.
        runner_target = "auto"
        try:
            runner_target = str(app.var_runner_target.get() or "auto").strip().lower()
        except Exception:
            runner_target = "auto"
        if runner_target not in {"auto", "cpu", "cuda", "tensorrt"}:
            runner_target = "auto"

        # ---------------- Accelerators to benchmark (suite plan) ----------------
        # Read once here so the worker thread does not touch Tk variables.
        acc_cpu = bool(getattr(app, "var_bench_acc_cpu", tk.BooleanVar(value=True)).get())
        acc_cuda = bool(getattr(app, "var_bench_acc_cuda", tk.BooleanVar(value=False)).get())
        acc_trt = bool(getattr(app, "var_bench_acc_tensorrt", tk.BooleanVar(value=False)).get())
        acc_h8 = bool(getattr(app, "var_bench_acc_hailo8", tk.BooleanVar(value=False)).get())
        acc_h10 = bool(getattr(app, "var_bench_acc_hailo10", tk.BooleanVar(value=False)).get())
        acc_deepx = bool(getattr(app, "var_bench_acc_deepx_m1", tk.BooleanVar(value=False)).get())

        if not any([acc_cpu, acc_cuda, acc_trt, acc_h8, acc_h10, acc_deepx]):
            # Defensive default (otherwise the suite is pointless).
            acc_cpu = True

        # Resolve Hailo hw_arch values from Split&Export settings (single source of truth).
        hailo8_hw = (getattr(app, "var_hailo_hef_hailo8_hw_arch", tk.StringVar(value="hailo8")).get() or "hailo8").strip()
        hailo10_hw = (getattr(app, "var_hailo_hef_hailo10_hw_arch", tk.StringVar(value="hailo10h")).get() or "hailo10h").strip()

        # Per-run image scaling (passed through to the runner harness).
        plan_image_scale = (getattr(app, "var_bench_image_scale", tk.StringVar(value="auto")).get() or "auto").strip().lower()

        # v52w: semantic-validation dataset selection is centralized in Tool Config.
        # Ignore stale hidden Benchmark-tab variables from older sessions.
        plan_validation_images = ""
        plan_validation_max_images = 0
        plan_validation_reference_mode = "auto"
        plan_benchmark_task = "auto"
        plan_mini_coco_ap50 = False
        plan_mini_classification_eval = False

        # v52v: central Tool-Config settings for calibration/validation and activation-proxy producer.
        tool_cls_calib_preset = str(getattr(app, "var_tool_cls_calib_preset", tk.StringVar(value="imagenette_val_mini_500")).get() or "imagenette_val_mini_500").strip()
        tool_cls_val_preset = str(getattr(app, "var_tool_cls_validation_preset", tk.StringVar(value="imagenette_val_mini_200")).get() or "imagenette_val_mini_200").strip()
        tool_det_calib_preset = str(getattr(app, "var_tool_det_calib_preset", tk.StringVar(value="coco_200")).get() or "coco_200").strip()
        tool_det_val_preset = str(getattr(app, "var_tool_det_validation_preset", tk.StringVar(value="coco_50")).get() or "coco_50").strip()
        try:
            tool_cls_val_max = int(str(getattr(app, "var_tool_cls_validation_max", tk.StringVar(value="200")).get() or "200"))
        except Exception:
            tool_cls_val_max = 200
        try:
            tool_det_val_max = int(str(getattr(app, "var_tool_det_validation_max", tk.StringVar(value="50")).get() or "50"))
        except Exception:
            tool_det_val_max = 50
        try:
            tool_calib_count = int(str(getattr(app, "var_tool_calibration_count", tk.StringVar(value="200")).get() or "200"))
        except Exception:
            tool_calib_count = 100
        proxy_backend = str(getattr(app, "var_activation_proxy_backend", tk.StringVar(value="cuda_ort")).get() or "cuda_ort").strip().lower().replace("-", "_")
        proxy_store_samples = str(getattr(app, "var_activation_proxy_store_samples", tk.StringVar(value="0")).get() or "0").strip()
        proxy_strict = bool(getattr(app, "var_activation_proxy_strict", tk.BooleanVar(value=False)).get())
        enable_deepx_split_plan = bool(getattr(app, "var_enable_deepx_split_plan", tk.BooleanVar(value=True)).get())
        # v52z: DeepX→TensorRT is no longer a hidden/optional row in manual
        # benchmark-set generation. If both DX-M1 and TensorRT are selected, the
        # benchmark plan must contain deepx_m1_to_tensorrt and the generator must
        # materialize the corresponding Part1 DXNNs. The Tool-Config checkbox is
        # kept only as a legacy/default preference for cases where one of the
        # backends is not selected.
        if bool(acc_deepx) and bool(acc_trt):
            enable_deepx_split_plan = True
        enable_deepx_part2_build = bool(getattr(app, "var_enable_deepx_part2_experimental_build", tk.BooleanVar(value=False)).get())
        if proxy_backend not in {"ort_cpu", "cuda_ort", "tensorrt_ort", "remote_deepx_tensorrt", "remote_deepx_cuda"}:
            proxy_backend = "cuda_ort"
        os.environ["ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND"] = proxy_backend
        os.environ["SPLITPOINT_ACTIVATION_PROXY_BACKEND"] = proxy_backend
        if proxy_store_samples:
            os.environ["ONNX_SPLITPOINT_ACTIVATION_PROXY_STORE_SAMPLES"] = proxy_store_samples
        if proxy_strict:
            os.environ["ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT"] = "1"
            os.environ["SPLITPOINT_ACTIVATION_PROXY_STRICT"] = "1"
        else:
            os.environ.pop("ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT", None)
            os.environ.pop("SPLITPOINT_ACTIVATION_PROXY_STRICT", None)
        # DeepX→TensorRT Part1 split rows are a normal first-class target now.
        # Keep an explicit 0/1 so deeper generator code never falls back to
        # legacy environment defaults.
        os.environ["ONNX_SPLITPOINT_ENABLE_DEEPX_SPLIT_PLAN"] = "1" if enable_deepx_split_plan else "0"
        if enable_deepx_part2_build:
            os.environ["ONNX_SPLITPOINT_DEEPX_PART2_EXPERIMENTAL_BUILD"] = "1"
        else:
            os.environ.pop("ONNX_SPLITPOINT_DEEPX_PART2_EXPERIMENTAL_BUILD", None)

        inferred_task = _infer_task_from_model_name_or_path(str(model_path))
        effective_task_for_defaults = plan_benchmark_task if plan_benchmark_task in {"classification", "detection"} else inferred_task
        if effective_task_for_defaults == "classification":
            if not str(plan_validation_images or "").strip():
                plan_validation_images = _resolve_validation_source_for_task(task="classification", cls_preset=tool_cls_val_preset, det_preset=tool_det_val_preset)
            if int(plan_validation_max_images or 0) <= 0:
                plan_validation_max_images = int(tool_cls_val_max or 200)
            if str(plan_image_scale or "auto").lower() in {"", "auto", "norm"}:
                plan_image_scale = "imagenet"
            plan_mini_classification_eval = True
            plan_mini_coco_ap50 = False
        elif effective_task_for_defaults == "detection":
            if not str(plan_validation_images or "").strip():
                plan_validation_images = _resolve_validation_source_for_task(task="detection", cls_preset=tool_cls_val_preset, det_preset=tool_det_val_preset)
            if int(plan_validation_max_images or 0) <= 0:
                plan_validation_max_images = int(tool_det_val_max or 50)

        # Normalize the full-model HEF build order once on the GUI thread so the
        # worker and services only see backend tokens (start/end/skip).
        full_hef_policy = normalize_full_hef_policy(
            getattr(
                app,
                "var_hailo_full_hef_order",
                tk.StringVar(value="Build at end (recommended)"),
            ).get()
        )
        full_model_preflight_policy = normalize_hailo_full_model_preflight_policy(
            getattr(
                app,
                "var_hailo_full_model_preflight",
                tk.StringVar(value="Enabled (plan-aware)"),
            ).get()
        )
        plan_hailo_preset = str(getattr(app, "var_hailo_bench_preset", tk.StringVar(value="End-to-end compare")).get() or "")
        plan_hailo_custom_full = bool(getattr(app, "var_hailo_bench_custom_full", tk.BooleanVar(value=True)).get())
        plan_hailo_custom_composed = bool(getattr(app, "var_hailo_bench_custom_composed", tk.BooleanVar(value=True)).get())
        plan_hailo_custom_part1 = bool(getattr(app, "var_hailo_bench_custom_part1", tk.BooleanVar(value=False)).get())
        plan_hailo_custom_part2 = bool(getattr(app, "var_hailo_bench_custom_part2", tk.BooleanVar(value=False)).get())
        plan_matrix_trt_to_hailo = bool(getattr(app, "var_matrix_trt_to_hailo", tk.BooleanVar(value=False)).get())
        plan_matrix_hailo_to_trt = bool(getattr(app, "var_matrix_hailo_to_trt", tk.BooleanVar(value=False)).get())
        plan_matrix_deepx_to_trt = bool(getattr(app, "var_matrix_deepx_to_trt", tk.BooleanVar(value=True)).get())
        plan_matrix_trt_to_deepx = bool(getattr(app, "var_matrix_trt_to_deepx", tk.BooleanVar(value=False)).get())

        # v53c: benchmark-tab checkbox selections are explicit user intent.
        # Evaluation-profile imports may provide a useful preset (shortlist,
        # validation defaults, Hailo defaults), but they must not silently remove
        # accelerators/matrix directions that the user checked in the Benchmark
        # tab.  This was hiding DeepX rows when an older smoke profile only
        # mentioned Hailo/TensorRT.
        ui_acc_cpu = bool(acc_cpu)
        ui_acc_cuda = bool(acc_cuda)
        ui_acc_trt = bool(acc_trt)
        ui_acc_h8 = bool(acc_h8)
        ui_acc_h10 = bool(acc_h10)
        ui_acc_deepx = bool(acc_deepx)
        ui_matrix_trt_to_hailo = bool(plan_matrix_trt_to_hailo)
        ui_matrix_hailo_to_trt = bool(plan_matrix_hailo_to_trt)
        ui_matrix_deepx_to_trt = bool(plan_matrix_deepx_to_trt)
        ui_matrix_trt_to_deepx = bool(plan_matrix_trt_to_deepx)

        if profile_overrides:
            acc_cpu = bool(profile_overrides.get('acc_cpu', acc_cpu)) or ui_acc_cpu
            acc_cuda = bool(profile_overrides.get('acc_cuda', acc_cuda)) or ui_acc_cuda
            acc_trt = bool(profile_overrides.get('acc_trt', acc_trt)) or ui_acc_trt
            acc_h8 = bool(profile_overrides.get('acc_h8', acc_h8)) or ui_acc_h8
            acc_h10 = bool(profile_overrides.get('acc_h10', acc_h10)) or ui_acc_h10
            acc_deepx = bool(profile_overrides.get('acc_deepx', acc_deepx)) or ui_acc_deepx
            plan_image_scale = str(profile_overrides.get('image_scale', plan_image_scale) or plan_image_scale)
            plan_validation_images = str(profile_overrides.get('validation_images', plan_validation_images) or '')
            try:
                plan_validation_max_images = int(profile_overrides.get('validation_max_images', plan_validation_max_images) or 0)
            except Exception:
                pass
            plan_validation_reference_mode = str(profile_overrides.get('validation_reference_mode', plan_validation_reference_mode) or plan_validation_reference_mode)
            plan_benchmark_task = str(profile_overrides.get('benchmark_task', plan_benchmark_task) or plan_benchmark_task)
            plan_mini_coco_ap50 = bool(profile_overrides.get('mini_coco_ap50', plan_mini_coco_ap50))
            plan_mini_classification_eval = bool(profile_overrides.get('mini_classification_eval', plan_mini_classification_eval))
            plan_hailo_preset = str(profile_overrides.get('hailo_preset', plan_hailo_preset) or plan_hailo_preset)
            plan_hailo_custom_full = bool(profile_overrides.get('hailo_custom_full', plan_hailo_custom_full))
            plan_hailo_custom_composed = bool(profile_overrides.get('hailo_custom_composed', plan_hailo_custom_composed))
            plan_hailo_custom_part1 = bool(profile_overrides.get('hailo_custom_part1', plan_hailo_custom_part1))
            plan_hailo_custom_part2 = bool(profile_overrides.get('hailo_custom_part2', plan_hailo_custom_part2))
            plan_matrix_trt_to_hailo = bool(profile_overrides.get('matrix_trt_to_hailo', plan_matrix_trt_to_hailo)) or ui_matrix_trt_to_hailo
            plan_matrix_hailo_to_trt = bool(profile_overrides.get('matrix_hailo_to_trt', plan_matrix_hailo_to_trt)) or ui_matrix_hailo_to_trt
            plan_matrix_deepx_to_trt = bool(profile_overrides.get('matrix_deepx_to_trt', plan_matrix_deepx_to_trt)) or ui_matrix_deepx_to_trt
            plan_matrix_trt_to_deepx = bool(profile_overrides.get('matrix_trt_to_deepx', plan_matrix_trt_to_deepx)) or ui_matrix_trt_to_deepx
            if 'full_model_preflight_policy' in profile_overrides:
                full_model_preflight_policy = normalize_hailo_full_model_preflight_policy(profile_overrides.get('full_model_preflight_policy'))

        # Freeze the semantic task after profile/UI merging.  Hailo builders
        # receive this value explicitly; unknown models remain ``auto`` and are
        # rejected by the Hailo preprocessing preflight rather than being
        # classified from their input dimensions.
        plan_benchmark_task = str(plan_benchmark_task or "auto").strip().lower()
        effective_task_for_defaults = (
            plan_benchmark_task
            if plan_benchmark_task in {"classification", "detection"}
            else inferred_task
        )

        # Last guard: if the user selected TensorRT plus DX-M1 in the Benchmark tab,
        # keep the corresponding DeepX split directions visible even when a legacy
        # imported profile did not know about them.
        if bool(acc_trt) and bool(acc_deepx):
            if ui_matrix_deepx_to_trt:
                plan_matrix_deepx_to_trt = True
            if ui_matrix_trt_to_deepx:
                plan_matrix_trt_to_deepx = True

        # v53c: the visible Benchmark-tab checkboxes are the final authority
        # for run-target and split-matrix selection.  Evaluation-profile import is
        # useful as a preset, but it must not silently override manual toggles at
        # generation time.  This specifically fixes cases where the UI shows
        # TensorRT→DeepX checked, but an imported profile without that run removes
        # it from benchmark_plan.json.
        try:
            acc_cpu = bool(getattr(app, "var_bench_acc_cpu", tk.BooleanVar(value=bool(acc_cpu))).get())
            acc_cuda = bool(getattr(app, "var_bench_acc_cuda", tk.BooleanVar(value=bool(acc_cuda))).get())
            acc_trt = bool(getattr(app, "var_bench_acc_tensorrt", tk.BooleanVar(value=bool(acc_trt))).get())
            acc_h8 = bool(getattr(app, "var_bench_acc_hailo8", tk.BooleanVar(value=bool(acc_h8))).get())
            acc_h10 = bool(getattr(app, "var_bench_acc_hailo10", tk.BooleanVar(value=bool(acc_h10))).get())
            acc_deepx = bool(getattr(app, "var_bench_acc_deepx_m1", tk.BooleanVar(value=bool(acc_deepx))).get())
            plan_matrix_trt_to_hailo = bool(getattr(app, "var_matrix_trt_to_hailo", tk.BooleanVar(value=bool(plan_matrix_trt_to_hailo))).get())
            plan_matrix_hailo_to_trt = bool(getattr(app, "var_matrix_hailo_to_trt", tk.BooleanVar(value=bool(plan_matrix_hailo_to_trt))).get())
            plan_matrix_deepx_to_trt = bool(getattr(app, "var_matrix_deepx_to_trt", tk.BooleanVar(value=bool(plan_matrix_deepx_to_trt))).get())
            plan_matrix_trt_to_deepx = bool(getattr(app, "var_matrix_trt_to_deepx", tk.BooleanVar(value=bool(plan_matrix_trt_to_deepx))).get())
        except Exception:
            pass

        # v53c: after the final Benchmark-tab authority merge, make the
        # DeepX split environment flags match the actual run matrix.  Some
        # lower-level materializers still look at these env flags for legacy
        # compatibility, so setting them before profile/UI merging can suppress
        # DeepX Part1/Part2 artifact creation.
        if bool(acc_deepx) and bool(acc_trt) and (bool(plan_matrix_deepx_to_trt) or bool(plan_matrix_trt_to_deepx)):
            enable_deepx_split_plan = True
            os.environ["ONNX_SPLITPOINT_ENABLE_DEEPX_SPLIT_PLAN"] = "1"
        if bool(acc_deepx) and bool(acc_trt) and bool(plan_matrix_trt_to_deepx):
            enable_deepx_part2_build = True
            os.environ["ONNX_SPLITPOINT_DEEPX_PART2_EXPERIMENTAL_BUILD"] = "1"

        run_plan = generation_service.build_run_plan(
            acc_cpu=bool(acc_cpu),
            acc_cuda=bool(acc_cuda),
            acc_trt=bool(acc_trt),
            acc_h8=bool(acc_h8),
            acc_h10=bool(acc_h10),
            acc_deepx=bool(acc_deepx),
            hailo8_hw=str(hailo8_hw or ""),
            hailo10_hw=str(hailo10_hw or ""),
            image_scale=str(plan_image_scale or "auto"),
            validation_images=str(plan_validation_images or ""),
            validation_max_images=int(max(0, int(plan_validation_max_images or 0))),
            validation_reference_mode=str(plan_validation_reference_mode or "auto"),
            mini_coco_ap50=bool(plan_mini_coco_ap50),
            benchmark_task=str(effective_task_for_defaults or "auto"),
            mini_classification_eval=bool(plan_mini_classification_eval),
            hailo_preset=str(plan_hailo_preset or ""),
            hailo_custom_full=bool(plan_hailo_custom_full),
            hailo_custom_composed=bool(plan_hailo_custom_composed),
            hailo_custom_part1=bool(plan_hailo_custom_part1),
            hailo_custom_part2=bool(plan_hailo_custom_part2),
            matrix_trt_to_hailo=bool(plan_matrix_trt_to_hailo),
            matrix_hailo_to_trt=bool(plan_matrix_hailo_to_trt),
            matrix_deepx_to_trt=bool(plan_matrix_deepx_to_trt),
            matrix_trt_to_deepx=bool(plan_matrix_trt_to_deepx),
            full_hef_policy=str(full_hef_policy or "end"),
        )
        bench_plan_runs: List[Dict[str, Any]] = list(run_plan.bench_plan_runs)
        hef_targets: List[str] = list(run_plan.hef_targets)
        hailo_selected = bool(run_plan.hailo_selected)
        hef_full = bool(run_plan.hef_full)
        hef_part1 = bool(run_plan.hef_part1)
        hef_part2 = bool(run_plan.hef_part2)
        hailo_variants = list(run_plan.hailo_variants)

        hailo_compile_rank_meta: Dict[int, Dict[str, Any]] = {}
        hailo_outlook_rows = []
        hailo_outlook_summary = None
        try:
            analysis_candidate_rows_snapshot = [
                dict(row)
                for row in (getattr(app, "_candidate_rows", None) or getattr(app, "candidates", None) or [])
                if isinstance(row, dict)
            ]
        except Exception:
            analysis_candidate_rows_snapshot = []
        if isinstance(a, dict) and candidate_search_pool:
            analysis_for_plan = dict(a)
            if analysis_candidate_rows_snapshot:
                analysis_for_plan["_candidate_rows"] = list(analysis_candidate_rows_snapshot)
            try:
                _plan = generation_service.prepare_generation_plan(
                    analysis_for_plan,
                    ranked_candidates,
                    candidate_search_pool,
                    k,
                    strict_boundary=False,
                    hailo_selected=bool(hailo_selected),
                    outlook_top_n=12,
                )
                # v59t: Hailo compile/context outlook is diagnostic by default.
                # Reorder the user-visible Analyse / stratified order only when
                # the explicit Benchmark checkbox asks for compile-aware ordering.
                if bool(benchmark_compile_aware_ordering):
                    ranked_candidates = list(_plan.ranked_candidates)
                    candidate_search_pool = list(_plan.candidate_search_pool)
                    k = int(_plan.requested_cases)
                hailo_compile_rank_meta = dict(_plan.hailo_compile_rank_meta)
                hailo_outlook_rows = list(_plan.hailo_outlook_rows)
                hailo_outlook_summary = _plan.hailo_outlook_summary
            except Exception as _hailo_rank_exc:
                logger.debug('BenchmarkGenerationService.prepare_generation_plan failed: %s', _hailo_rank_exc)
                if bool(benchmark_compile_aware_ordering) and hailo_selected and isinstance(a, dict):
                    try:
                        _reranked_pool, hailo_compile_rank_meta = rerank_candidates_for_hailo(analysis_for_plan, candidate_search_pool)
                        if _reranked_pool:
                            _order = {int(b): idx for idx, b in enumerate(_reranked_pool)}
                            candidate_search_pool = list(_reranked_pool)
                            ranked_candidates = sorted([int(b) for b in ranked_candidates], key=lambda b: (_order.get(int(b), 10**9), int(b)))
                    except Exception as _fallback_exc:
                        logger.debug('Hailo compile-aware fallback reranking failed: %s', _fallback_exc)

        hef_opt_level = _safe_int((getattr(app, "var_hailo_hef_opt_level", tk.StringVar(value="1")).get() or "").strip()) or 1
        hef_calib_count = _safe_int((getattr(app, "var_hailo_hef_calib_count", tk.StringVar(value="64")).get() or "").strip()) or 64
        hef_calib_bs = _safe_int((getattr(app, "var_hailo_hef_calib_batch_size", tk.StringVar(value="8")).get() or "").strip()) or 8
        hef_calib_dir = (getattr(app, "var_hailo_hef_calib_dir", tk.StringVar(value="")).get() or "").strip() or None
        # v52v: if no explicit Hailo calib dir is configured, derive one from the central Tool-Config calibration preset.
        if not hef_calib_dir:
            if effective_task_for_defaults == "classification":
                hef_calib_dir = _resolve_preset_images_dir(tool_cls_calib_preset) or None
                if tool_calib_count > 0:
                    hef_calib_count = int(tool_calib_count)
            elif effective_task_for_defaults == "detection":
                hef_calib_dir = _resolve_preset_images_dir(tool_det_calib_preset) or None
                if tool_calib_count > 0:
                    hef_calib_count = int(tool_calib_count)
        hef_force = bool(getattr(app, "var_hailo_hef_force", tk.BooleanVar(value=False)).get())
        hef_keep = bool(getattr(app, "var_hailo_hef_keep_artifacts", tk.BooleanVar(value=False)).get())

        # Backend selection reuses the Hailo feasibility-check backend controls.
        hef_backend = normalize_hailo_backend(getattr(app, "var_hailo_backend", tk.StringVar(value="auto")).get())
        hef_wsl_distro = (getattr(app, "var_hailo_wsl_distro", tk.StringVar(value="")).get() or "").strip() or None
        hef_wsl_venv = (getattr(app, "var_hailo_wsl_venv", tk.StringVar(value="auto")).get() or "auto").strip() or "auto"
        hef_fixup = bool(getattr(app, "var_hailo_fixup", tk.BooleanVar(value=True)).get())

        do_ctx_full = bool(getattr(app, 'var_split_ctx_full', tk.BooleanVar(value=True)).get())
        do_ctx_cutflow = bool(getattr(app, 'var_split_ctx_cutflow', tk.BooleanVar(value=False)).get())
        ctx_hops = _safe_int(getattr(app, 'var_split_ctx_hops', tk.StringVar(value='2')).get()) or 2

        eps_txt = (app.var_split_eps.get() or "").strip()
        eps_default = 1e-4
        if eps_txt:
            try:
                eps_default = float(eps_txt)
            except Exception:
                eps_default = 1e-4

        # Read batch override once here (avoid reading Tk variables from worker thread).
        params = None
        try:
            params = app._read_params()
            batch_override = params.batch_override
        except Exception:
            batch_override = None

        # Determine a nice padding width for folder names. Prefer the full
        # search pool because benchmark generation may backfill beyond the currently
        # displayed shortlist.
        _pad_source = list(candidate_search_pool or ranked_candidates or app.current_picks or [0])
        pad = max(3, len(str(max(_pad_source)))) if _pad_source else 3

        # Dedicated per-run generation log (mirrors the live dialog output and HEF
        # sub-logs). This makes post-mortems possible even when the GUI log is long
        # or the host reboots mid-run.
        bench_log_path = os.path.join(out_dir, "benchmark_generation.log")

        # Snapshot GUI state before the background worker starts. Tk falls back to
        # ``app.tk`` lookups for missing attributes, so worker-thread access to a
        # missing ``app.candidates`` attribute turned into
        # ``AttributeError: '_tkinter.tkapp' object has no attribute 'candidates'``.
        # Keep the worker on plain Python data only.
        benchmark_gap = int(_safe_int(app.var_min_gap.get()) or 0)
        if profile_overrides:
            try:
                benchmark_gap = int(profile_overrides.get('min_gap', benchmark_gap) or 0)
            except Exception:
                pass
        llm_style_enabled = bool(app.var_llm_enable.get())
        value_bytes_map_snapshot = a.get("value_bytes") if isinstance(a, dict) else None
        analysis_topk_snapshot = int(getattr(params, 'topk', len(app.current_picks)))
        system_spec_payload = asdict(app._build_system_spec(params)) if params else None
        model_path_for_worker = str(model_path)
        full_model_src_for_worker = os.path.abspath(model_path_for_worker)
        hef_timeout_s = int(app._hailo_hef_timeout_seconds())
        hailo_publish_gui_diagnostics_cb = app._hailo_publish_gui_diagnostics
        analysis_predicted_metrics_for_boundary_fn = app._analysis_predicted_metrics_for_boundary
        hailo_parse_entry_for_boundary_fn = app._hailo_parse_entry_for_boundary
        hailo_parse_scalar_fields_fn = app._hailo_parse_scalar_fields
        hailo_part2_enable_suggested_endnode_fallback = (
            bool(getattr(app, 'var_bench_hailo_part2_suggested_fallback').get())
            if hasattr(app, 'var_bench_hailo_part2_suggested_fallback')
            else True
        )
        benchmark_objective_raw = (
            str(getattr(app, 'var_bench_objective', tk.StringVar(value='Use analysis objective')).get() or 'Use analysis objective').strip()
            if hasattr(app, 'var_bench_objective') else 'Use analysis objective'
        )
        if benchmark_objective_raw.lower().startswith('use analysis'):
            benchmark_objective = str(getattr(app, 'var_analysis_objective', tk.StringVar(value='Balanced')).get() or 'Balanced').strip()
        else:
            benchmark_objective = benchmark_objective_raw
        if not benchmark_objective:
            benchmark_objective = 'Balanced'
        orchestration_service = getattr(
            app,
            '_benchmark_generation_orchestration_service',
            BenchmarkGenerationOrchestrationService(generation_service, getattr(app, '_benchmark_generation_execution_service', None)),
        )

        # --- progress dialog + background worker ---
        job_id = f"generate-{base}-{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        cancel_event = threading.Event()
        app._jobs_register(
            job_id=job_id,
            kind="generate",
            type_label="Benchmark set",
            title=f"Generating benchmark set — {base}",
            name=str(Path(out_dir).name or base),
            output_dir=str(out_dir),
            log_path=str(bench_log_path),
            initial_status=f"Generating benchmark set with target of up to {k} accepted cases…",
            initial_lines=[
                f"Generating benchmark set with target of up to {k} accepted cases…",
                f"Output dir: {out_dir}",
                "Runs in the background. Close this window any time and reopen it from the Jobs tab.",
            ],
            progress_maximum=max(1, k),
            cancel_callback=lambda: cancel_event.set(),
            can_cancel=True,
            geometry="900x420",
        )

        q: "queue.Queue[tuple]" = queue.Queue()

        def _write_benchmark_suite_script(dst_dir: str, bench_json_name: str = "benchmark_set.json") -> str:
            """Write benchmark suite runner script from a template resource."""
            return write_benchmark_suite_script(dst_dir, bench_json_name=bench_json_name)

        def worker() -> None:
            nonlocal ranked_candidates, candidate_search_pool
            # Legacy regression note: discarded cases are still tracked for benchmark generation,
            # but now live inside BenchmarkGenerationRuntime / BenchmarkGenerationExecutionService.
            # discarded_cases = []
            runtime = None
            try:
                runtime = generation_service.start_generation_runtime(
                    out_dir=out_dir,
                    bench_log_path=bench_log_path,
                    requested_cases=int(k),
                    ranked_candidates=ranked_candidates,
                    candidate_search_pool=candidate_search_pool,
                    hef_full_policy=full_hef_policy,
                    model_name=base,
                    model_source=model_path_for_worker,
                    require_single_part2_input=bool(
                        require_single_part2_input
                    ),
                    resume_generation=bool(resume_generation),
                    resume_state_hint=resume_state_hint,
                )
                try:
                    runtime.log(f"[plan-ui] benchmark candidate selection: strategy={benchmark_selection_strategy or 'rank_order'} compile_aware_ordering={bool(benchmark_compile_aware_ordering)}")
                    if manual_selection_audit:
                        runtime.log(f"[plan-ui] stratified windows: {manual_selection_audit}")
                    if bool(plan_matrix_trt_to_hailo) and not bool(plan_matrix_hailo_to_trt):
                        runtime.log("[plan-ui][direction-warning] TensorRT→Hailo is selected and Hailo→TensorRT is not selected. Hailo Part2 compatibility filters apply; this is the reverse of the YOLOv7 paper Hailo-P1→TensorRT-P2 setup.")
                    if bool(plan_matrix_hailo_to_trt):
                        runtime.log("[plan-ui] Hailo→TensorRT selected: Hailo Part1 + TensorRT Part2 paper-style direction is enabled.")
                except Exception:
                    pass
                cases = runtime.cases
                errors = runtime.errors
                discarded_cases = runtime.discarded_cases
                suite_hailo_hefs = runtime.suite_hailo_hefs
                completed_boundaries = runtime.completed_boundaries
                accepted_boundaries = runtime.accepted_boundaries
                discarded_boundaries = runtime.discarded_boundaries
                made = int(len(cases))

                def log(line: str, *, level: int = logging.INFO) -> None:
                    runtime.log(line, queue_put=q.put, level=level)

                log("=== Generating benchmark set ===")
                log(f"out_dir: {out_dir}")
                log(f"model_path: {model_path_for_worker}")
                log(f"generation_log: {bench_log_path}")
                log(f"requested_cases (target accepted): {k}")
                log(f"preferred shortlist: {len(ranked_candidates)}")
                log(f"candidate search pool: {len(candidate_search_pool)}")
                log(
                    "Part-2 input count = 1 selection filter: "
                    f"{'on' if require_single_part2_input else 'off'}"
                )
                if part2_input_filter_audit:
                    log(
                        "Part-2 input filter exclusions: "
                        f"{part2_input_filter_audit}"
                    )
                log(f"[tool-config] task/defaults: inferred={inferred_task} effective={effective_task_for_defaults} image_scale={plan_image_scale}")
                log(f"[tool-config] validation source={plan_validation_images or 'auto'} max={plan_validation_max_images}")
                log(f"[tool-config] calibration dir={hef_calib_dir or 'auto'} count={hef_calib_count} batch={hef_calib_bs}")
                log(f"[tool-config] activation proxy backend={proxy_backend} store_samples={proxy_store_samples or '0'} strict={proxy_strict} deepx_split_plan={enable_deepx_split_plan} deepx_part2_experimental={enable_deepx_part2_build}")
                log(f"[plan-ui] accelerators: cpu={bool(acc_cpu)} cuda={bool(acc_cuda)} tensorrt={bool(acc_trt)} hailo8={bool(acc_h8)} hailo10={bool(acc_h10)} deepx_m1={bool(acc_deepx)}")
                log(f"[plan-ui] split matrix: trt_to_hailo={bool(plan_matrix_trt_to_hailo)} hailo_to_trt={bool(plan_matrix_hailo_to_trt)} deepx_to_trt={bool(plan_matrix_deepx_to_trt)} trt_to_deepx={bool(plan_matrix_trt_to_deepx)}")
                log(f"[plan-ui] generated runs: {', '.join([str(r.get('id') or '?') for r in bench_plan_runs]) if bench_plan_runs else '(none)'}")
                if resume_generation and runtime.completed_boundaries:
                    log(f"[resume] continuing existing benchmark set ({len(cases)} accepted, {len(discarded_cases)} discarded)")
                if resume_generation and resume_report is not None:
                    summary = (resume_report.summary() or "No resume consistency changes required.").splitlines()
                    for line in summary:
                        log(f"[resume] {line}" if line else "[resume]")

                full_model_src = full_model_src_for_worker
                full_model_dst = generation_service.copy_portable_full_model(runtime, full_model_src, log_cb=log)
                log(f"full model (suite copy): {full_model_dst}")

                # Resolve Hailo helpers once; the top-level orchestration now lives in
                # BenchmarkGenerationOrchestrationService instead of gui_app.py.
                hailo_build_hef_fn = None
                hailo_parse_check_fn = None
                hailo_build_unavailable: Optional[str] = None
                hailo_part2_precheck_fn = None
                hailo_part2_precheck_error_fn = None
                hailo_part2_parser_precheck_fn = None
                hailo_part2_parser_precheck_error_fn = None

                def _persist_generation_state(status: str = 'running', current_boundary: Optional[int] = None) -> None:
                    runtime.persist(status=status, current_boundary=current_boundary)

                resolved_hailo_helpers = resolve_hailo_benchmark_helpers(
                    need_build=bool(hef_targets and (hef_full or hef_part1 or hef_part2)),
                    need_part2=bool(hef_targets and hef_part2),
                )
                hailo_build_hef_fn = resolved_hailo_helpers.hailo_build_hef_fn
                hailo_parse_check_fn = resolved_hailo_helpers.hailo_parse_check_fn
                hailo_build_unavailable = resolved_hailo_helpers.hailo_build_unavailable
                hailo_part2_precheck_fn = resolved_hailo_helpers.hailo_part2_precheck_fn
                hailo_part2_precheck_error_fn = resolved_hailo_helpers.hailo_part2_precheck_error_fn
                hailo_part2_parser_precheck_fn = resolved_hailo_helpers.hailo_part2_parser_precheck_fn
                hailo_part2_parser_precheck_error_fn = resolved_hailo_helpers.hailo_part2_parser_precheck_error_fn

                if hailo_build_unavailable:
                    errors.append(hailo_build_unavailable)
                    log(hailo_build_unavailable)
                if resolved_hailo_helpers.hailo_part2_import_error:
                    log(f"hailo part2 precheck unavailable: {resolved_hailo_helpers.hailo_part2_import_error}")

                preferred_shortlist_original = list(ranked_candidates)
                benign_discard_reasons = {
                    "hailo_part2_prefilter",
                    "hailo_part2_precheck",
                    "hailo_part2_auto_filtered",
                    "hailo_part2_parser_prefilter",
                    "hailo_part2_parser_auto_filtered",
                    "hailo_part2_concat_sanity_prefilter",
                    "hailo_part2_concat_sanity_auto_filtered",
                    "hailo_failure_cluster_skip",
                }

                execution_cfg = BenchmarkGenerationExecutionConfig(
                    runtime=runtime,
                    target_cases=int(k),
                    gap=int(benchmark_gap),
                    ranked_candidates=list(ranked_candidates),
                    candidate_search_pool=list(candidate_search_pool),
                    out_dir=Path(out_dir),
                    base=base,
                    pad=int(pad),
                    strict_boundary=bool(strict_boundary),
                    model=model,
                    nodes=nodes,
                    order=order,
                    analysis_payload=a,
                    analysis_candidates=list(analysis_candidate_rows_snapshot),
                    require_single_part2_input=bool(
                        require_single_part2_input
                    ),
                    bench_plan_runs=list(bench_plan_runs),
                    benchmark_task=str(effective_task_for_defaults or ""),
                    runner_target=runner_target,
                    do_ctx_full=bool(do_ctx_full),
                    do_ctx_cutflow=bool(do_ctx_cutflow),
                    ctx_hops=int(ctx_hops),
                    llm_style=bool(llm_style_enabled),
                    value_bytes_map=value_bytes_map_snapshot,
                    full_model_src=str(full_model_src),
                    full_model_dst=str(full_model_dst),
                    tool_gui_version=__version__,
                    tool_core_version=resolve_tool_core_version(),
                    evaluation_profile_meta=(evaluation_profile_resolution.to_metadata() if evaluation_profile_resolution is not None else None),
                    hailo_compile_rank_meta=dict(hailo_compile_rank_meta or {}),
                    hef_targets=list(hef_targets),
                    hef_part1=bool(hef_part1),
                    hef_part2=bool(hef_part2),
                    hef_backend=str(hef_backend),
                    hef_fixup=bool(hef_fixup),
                    hef_opt_level=int(hef_opt_level),
                    hef_calib_dir=hef_calib_dir,
                    hef_calib_count=int(hef_calib_count),
                    hef_calib_bs=int(hef_calib_bs),
                    hef_force=bool(hef_force),
                    hef_keep=bool(hef_keep),
                    hef_wsl_distro=hef_wsl_distro,
                    hef_wsl_venv=str(hef_wsl_venv),
                    hef_timeout_s=int(hef_timeout_s),
                    hailo_build_hef_fn=hailo_build_hef_fn,
                    hailo_build_unavailable=hailo_build_unavailable,
                    hailo_part2_precheck_fn=hailo_part2_precheck_fn,
                    hailo_part2_precheck_error_fn=hailo_part2_precheck_error_fn,
                    hailo_part2_parser_precheck_fn=hailo_part2_parser_precheck_fn,
                    hailo_part2_parser_precheck_error_fn=hailo_part2_parser_precheck_error_fn,
                    hailo_part2_enable_suggested_endnode_fallback=bool(hailo_part2_enable_suggested_endnode_fallback),
                    should_cancel=lambda: bool(cancel_event.is_set()),
                )
                execution_callbacks = BenchmarkGenerationExecutionCallbacks(
                    log=log,
                    queue_put=q.put,
                    persist_state=lambda **kwargs: _persist_generation_state(**kwargs),
                    publish_hailo_diagnostics=lambda label, result, log_cb: hailo_publish_gui_diagnostics_cb(label, result, log_cb=log_cb),
                    predicted_metrics_for_boundary=lambda analysis_payload, boundary: analysis_predicted_metrics_for_boundary_fn(analysis_payload, boundary),
                    hailo_parse_entry_for_boundary=lambda analysis_payload, boundary: hailo_parse_entry_for_boundary_fn(analysis_payload, boundary),
                    hailo_parse_scalar_fields=lambda entry: hailo_parse_scalar_fields_fn(entry),
                )

                analysis_params_payload = {
                    'objective': (str(getattr(app, 'var_analysis_objective', tk.StringVar(value='Balanced')).get() or 'Balanced') if app is not None else 'Balanced'),
                    'ranking': str(getattr(params, 'ranking', 'score')),
                    'topk': int(analysis_topk_snapshot),
                    'min_gap': int(getattr(params, 'min_gap', 0)),
                    'exclude_trivial': bool(getattr(params, 'exclude_trivial', False)),
                    'only_single_tensor': bool(getattr(params, 'only_single_tensor', False)),
                    'require_single_part2_input': bool(
                        require_single_part2_input
                    ),
                    'part2_input_filter_audit': list(
                        part2_input_filter_audit
                    ),
                    'strict_boundary': bool(getattr(params, 'strict_boundary', False)),
                    'prune_skip_block': bool(getattr(params, 'prune_skip_block', False)),
                    'skip_min_span': int(getattr(params, 'skip_min_span', 0)),
                    'skip_allow_last_n': int(getattr(params, 'skip_allow_last_n', 0)),
                    'link_model': str(getattr(params, 'link_model', 'ideal')),
                    'bandwidth_value': getattr(params, 'bw_value', None),
                    'bandwidth_unit': str(getattr(params, 'bw_unit', 'MB/s')),
                    'gops_left': getattr(params, 'gops_left', None),
                    'gops_right': getattr(params, 'gops_right', None),
                    'link_overhead_ms': getattr(params, 'overhead_ms', 0.0),
                    'link_energy_pj_per_byte': getattr(params, 'link_energy_pj_per_byte', None),
                    'link_mtu_payload_bytes': getattr(params, 'link_mtu_payload_bytes', None),
                    'link_per_packet_overhead_ms': getattr(params, 'link_per_packet_overhead_ms', None),
                    'link_per_packet_overhead_bytes': getattr(params, 'link_per_packet_overhead_bytes', None),
                    'energy_pj_per_flop_left': getattr(params, 'energy_pj_per_flop_left', None),
                    'energy_pj_per_flop_right': getattr(params, 'energy_pj_per_flop_right', None),
                    'link_max_latency_ms': getattr(params, 'link_max_latency_ms', None),
                    'link_max_energy_mJ': getattr(params, 'link_max_energy_mJ', None),
                    'link_max_bytes': getattr(params, 'link_max_bytes', None),
                    'max_peak_act_left': getattr(params, 'max_peak_act_left', None),
                    'max_peak_act_left_unit': str(getattr(params, 'max_peak_act_left_unit', 'MiB')),
                    'max_peak_act_right': getattr(params, 'max_peak_act_right', None),
                    'max_peak_act_right_unit': str(getattr(params, 'max_peak_act_right_unit', 'MiB')),
                    'batch_override': batch_override,
                    'eps_default': float(eps_default),
                }
                resume_lines = []
                if resume_generation and resume_report is not None and (resume_report.changed or resume_report.warnings):
                    resume_lines = [line for line in (resume_report.summary() or '').splitlines() if line.strip()]

                prepared_full_hailo_baseline = _prepared_full_hailo_baseline(model_path)
                hailo_full_end_node_names, hailo_full_endpoint_mode = _prepared_full_hailo_endpoint_override(model_path)
                if not hailo_full_end_node_names and bool(prepared_full_hailo_baseline.get('ok')):
                    hailo_full_end_node_names = [str(x).strip() for x in list(prepared_full_hailo_baseline.get('end_node_names') or []) if str(x).strip()]
                    hailo_full_endpoint_mode = str(prepared_full_hailo_baseline.get('endpoint_mode') or hailo_full_endpoint_mode or '')
                if hailo_full_end_node_names:
                    log(
                        "suite: using prepared full-Hailo endpoint override "
                        f"mode={hailo_full_endpoint_mode or 'custom'} end_nodes={hailo_full_end_node_names}"
                    )
                if bool(prepared_full_hailo_baseline.get('selected')):
                    if bool(prepared_full_hailo_baseline.get('ok')):
                        log(
                            "suite: prepared full-Hailo HEF baseline available "
                            f"mode={prepared_full_hailo_baseline.get('endpoint_mode') or 'full'} "
                            f"hef={prepared_full_hailo_baseline.get('hef_path')}"
                        )
                    else:
                        log(
                            "suite: prepared full-Hailo metadata found but no usable HEF baseline is available "
                            f"({prepared_full_hailo_baseline.get('reason') or 'unknown'})"
                        )
                execution_cfg = replace(
                    execution_cfg,
                    hailo_full_end_node_names=list(hailo_full_end_node_names or []),
                    hailo_full_endpoint_mode=str(hailo_full_endpoint_mode or ''),
                    hailo_full_output_contract=(
                        dict(prepared_full_hailo_baseline.get('output_contract') or {})
                        if isinstance(prepared_full_hailo_baseline.get('output_contract'), dict)
                        else None
                    ),
                )

                orchestration_cfg = BenchmarkGenerationOrchestrationConfig(
                    runtime=runtime,
                    execution_cfg=execution_cfg,
                    execution_callbacks=execution_callbacks,
                    target_cases=int(k),
                    preferred_shortlist_original=preferred_shortlist_original,
                    ranked_candidates=list(ranked_candidates),
                    candidate_search_pool=list(candidate_search_pool),
                    out_dir=Path(out_dir),
                    base=base,
                    pad=int(pad),
                    full_model_src=full_model_src,
                    full_model_dst=str(full_model_dst),
                    analysis_payload=a,
                    analysis_params_payload=analysis_params_payload,
                    require_single_part2_input=bool(
                        require_single_part2_input
                    ),
                    system_spec_payload=system_spec_payload,
                    bench_log_path=str(bench_log_path),
                    bench_plan_runs=bench_plan_runs,
                    hef_targets=list(hef_targets),
                    hef_full=bool(hef_full),
                    hef_part1=bool(hef_part1),
                    hef_part2=bool(hef_part2),
                    hef_backend=str(hef_backend),
                    hef_fixup=bool(hef_fixup),
                    hef_opt_level=int(hef_opt_level),
                    hef_calib_dir=hef_calib_dir,
                    hef_calib_count=int(hef_calib_count),
                    hef_calib_bs=int(hef_calib_bs),
                    hef_force=bool(hef_force),
                    hef_keep=bool(hef_keep),
                    hef_wsl_distro=hef_wsl_distro,
                    hef_wsl_venv=str(hef_wsl_venv),
                    hef_timeout_s=int(hef_timeout_s),
                    full_hef_policy=str(full_hef_policy),
                    full_model_preflight_policy=str(full_model_preflight_policy or 'enabled'),
                    hailo_full_end_node_names=list(hailo_full_end_node_names or []),
                    hailo_full_endpoint_mode=str(hailo_full_endpoint_mode or ''),
                    hailo_full_output_contract=(
                        dict(prepared_full_hailo_baseline.get('output_contract') or {})
                        if isinstance(prepared_full_hailo_baseline.get('output_contract'), dict)
                        else None
                    ),
                    prepared_full_hailo_baseline=dict(prepared_full_hailo_baseline or {}),
                    hailo_build_hef_fn=hailo_build_hef_fn,
                    hailo_parse_check_fn=hailo_parse_check_fn,
                    hailo_build_unavailable=hailo_build_unavailable,
                    hailo_part2_precheck_fn=hailo_part2_precheck_fn,
                    hailo_part2_precheck_error_fn=hailo_part2_precheck_error_fn,
                    hailo_part2_parser_precheck_fn=hailo_part2_parser_precheck_fn,
                    hailo_part2_parser_precheck_error_fn=hailo_part2_parser_precheck_error_fn,
                    resume_generation=bool(resume_generation),
                    resume_report_summary_lines=resume_lines,
                    hailo_selected=bool(hailo_selected),
                    hailo_outlook_summary=hailo_outlook_summary,
                    benign_discard_reasons=sorted(benign_discard_reasons),
                    write_harness_script=_write_benchmark_suite_script,
                    copy_schema_tree=lambda: copy_resource_tree("resources", "schemas", dest=Path(out_dir) / "schemas"),
                    tool_gui_version=__version__,
                    tool_core_version=resolve_tool_core_version(),
                    evaluation_profile_meta=(evaluation_profile_resolution.to_metadata() if evaluation_profile_resolution is not None else None),
                    benchmark_objective=str(benchmark_objective),
                    should_cancel=lambda: bool(cancel_event.is_set()),
                )
                log("[progress] running benchmark-set orchestration: applying target policies, selecting final candidates, exporting splits, and building requested artifacts...")
                if ranked_candidates:
                    try:
                        log("[progress] preferred boundaries: " + ", ".join(f"b{int(x)}" for x in list(ranked_candidates)[:20]))
                    except Exception:
                        pass
                if candidate_search_pool:
                    try:
                        log("[progress] candidate pool size: " + str(len(candidate_search_pool)))
                    except Exception:
                        pass
                orchestration_result = orchestration_service.run(orchestration_cfg)
                ranked_candidates = list(orchestration_result.ranked_candidates)
                candidate_search_pool = list(orchestration_result.candidate_search_pool)
                final_kind = str(orchestration_result.final_status or 'warn').strip().lower()
                final_msg = str(orchestration_result.final_msg or '')
                final_summary = dict(orchestration_result.summary_data or {})

                # Manual Benchmark-tab DeepX full artifact materialization.  The
                # plan row is written by BenchmarkGenerationService; here we make
                # the referenced deepx/deepx_m1/full/model.dxnn real so the same
                # suite can be benchmarked remotely without going through the
                # formal Evaluation Workflow first.
                if bool(acc_deepx):
                    deepx_task = (
                        str(plan_benchmark_task).strip().lower()
                        if str(plan_benchmark_task).strip().lower() in {'classification', 'detection'}
                        else str(inferred_task).strip().lower()
                    )
                    deepx_status = _materialize_manual_deepx_full_artifact(
                        out_dir=Path(out_dir),
                        model_path=str(full_model_src),
                        model=model,
                        bench_plan_runs=list(bench_plan_runs),
                        validation_images=str(plan_validation_images or ''),
                        validation_max_images=int(plan_validation_max_images or 0),
                        fallback_calib_dir=hef_calib_dir,
                        calibration_num=int(hef_calib_count or 0),
                        task_hint=deepx_task,
                        force_build=parse_config_bool(hef_force, field="hailo_build.force_build"),
                        log=log,
                    )
                    final_summary['deepx_full_artifact_status'] = dict(deepx_status or {})
                    if str((deepx_status or {}).get('status') or '').lower() == 'ok':
                        log('[deepx] full DXNN ready for manual benchmark suite')
                    else:
                        msg_dx = '[deepx] full DXNN not ready: ' + str((deepx_status or {}).get('status') or 'unknown')
                        log(msg_dx, level=logging.WARNING)
                        if final_kind == 'ok':
                            final_kind = 'warn'
                        final_msg = (final_msg + '\n' + msg_dx).strip()

                    deepx_part1_status = _materialize_manual_deepx_part1_artifacts(
                        out_dir=Path(out_dir),
                        bench_plan_runs=list(bench_plan_runs),
                        validation_images=str(plan_validation_images or ''),
                        fallback_calib_dir=hef_calib_dir,
                        calibration_num=int(hef_calib_count or 0),
                        task_hint=deepx_task,
                        force_build=parse_config_bool(hef_force, field="hailo_build.force_build"),
                        log=log,
                    )
                    final_summary['deepx_part1_artifact_status'] = dict(deepx_part1_status or {})
                    if bool((deepx_part1_status or {}).get('selected')):
                        if str((deepx_part1_status or {}).get('status') or '').lower() in {'ok', 'partial'}:
                            log(f"[deepx] part1 DXNN artifacts ready: ok={(deepx_part1_status or {}).get('ok_count', 0)} failed={(deepx_part1_status or {}).get('failed_count', 0)}")
                        else:
                            msg_dx_p1 = '[deepx] part1 DXNN artifacts not ready: ' + str((deepx_part1_status or {}).get('status') or 'unknown')
                            log(msg_dx_p1, level=logging.WARNING)
                            if final_kind == 'ok':
                                final_kind = 'warn'
                            final_msg = (final_msg + '\n' + msg_dx_p1).strip()

                is_clean_success = final_kind == 'ok'
                raw_text = str(final_summary.get('raw_text') or '').strip()
                log(raw_text or final_msg, level=logging.INFO if is_clean_success else logging.WARNING)
                if is_clean_success:
                    q.put(("ok", final_msg, final_summary))
                elif final_kind == 'cancelled':
                    q.put(("cancelled", final_msg, final_summary))
                else:
                    q.put(("warn", final_msg, final_summary))
            except Exception as e:
                logging.exception("Benchmark set generation failed")
                try:
                    if runtime is not None:
                        _persist_generation_state(status="failed", current_boundary=None)
                except Exception:
                    pass
                try:
                    if runtime is not None and runtime.bench_log_fp is not None:
                        traceback.print_exc(file=runtime.bench_log_fp)
                        runtime.bench_log_fp.flush()
                except Exception:
                    pass
                q.put(("err", f"{type(e).__name__}: {e}\n\nGeneration log: {bench_log_path}"))
            finally:
                try:
                    if runtime is not None:
                        runtime.close()
                except Exception:
                    pass

        app._set_background_job_active("generate", True)
        gen_thread = threading.Thread(target=worker, daemon=True)
        app._benchmark_generation_thread = gen_thread
        try:
            gen_thread.start()
        except Exception as exc:
            app._set_background_job_active("generate", False)
            app._jobs_finish(job_id, status="error", message="Failed to start benchmark-set generation thread", output_dir=str(out_dir), log_path=str(bench_log_path))
            if callable(completion_callback):
                try:
                    completion_callback('err', str(out_dir), str(bench_log_path), f'{type(exc).__name__}: {exc}', {})
                except Exception:
                    logger.debug('Benchmark generation completion callback failed during startup', exc_info=True)
            raise

        def poll() -> None:
            final_status: Optional[str] = None
            final_msg: str = ""
            final_summary: Dict[str, Any] = {}

            # Drain the queue so log output stays responsive even when a lot of
            # lines arrive quickly (e.g. Hailo DFC compilation).
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break

                if not item:
                    continue

                status = str(item[0])

                if status == 'prog':
                    try:
                        made = int(item[1])
                        what = str(item[2]) if len(item) > 2 else ''
                        app._jobs_set_progress(
                            job_id,
                            value=float(made),
                            label=(f"{made}/{k}: {what}" if what else f"{made}/{k}"),
                            display=f"{made}/{k}",
                            progress_maximum=max(1, k),
                        )
                    except Exception:
                        pass
                    continue

                if status in ('log', 'hef'):
                    try:
                        if status == 'log':
                            line = str(item[1]) if len(item) > 1 else ''
                        else:
                            # ('hef', stream, line)
                            line = str(item[2]) if len(item) > 2 else ''
                        app._jobs_append_log(job_id, line)
                        if status == 'hef' and line:
                            current_value = 0.0
                            try:
                                current_value = float((getattr(app, '_background_jobs', {}) or {}).get(job_id).progress_value)
                            except Exception:
                                current_value = 0.0
                            app._jobs_set_progress(job_id, value=current_value, label=line[:220])
                    except Exception:
                        pass
                    continue

                if status in ('msg', 'note'):
                    try:
                        what = str(item[1]) if len(item) > 1 else ''
                        current_value = 0.0
                        try:
                            current_value = float((getattr(app, '_background_jobs', {}) or {}).get(job_id).progress_value)
                        except Exception:
                            current_value = 0.0
                        app._jobs_set_progress(job_id, value=current_value, label=what)
                    except Exception:
                        pass
                    continue

                if status in ('ok', 'warn', 'err', 'cancelled'):
                    final_status = status
                    final_msg = str(item[1]) if len(item) > 1 else ''
                    final_summary = dict(item[2] or {}) if len(item) > 2 and isinstance(item[2], dict) else {}
                    break

            try:
                app.update_idletasks()
            except Exception:
                pass

            if not final_status:
                app.after(80, poll)
                return

            app._set_background_job_active("generate", False)
            status_map = {
                'ok': 'success',
                'warn': 'warning',
                'cancelled': 'cancelled',
                'err': 'error',
            }
            app._jobs_finish(
                job_id,
                status=status_map.get(final_status, 'error'),
                message=final_msg,
                output_dir=str(out_dir),
                log_path=str(bench_log_path),
            )

            if final_status in {'ok', 'warn'}:
                try:
                    bench_json_path = Path(out_dir) / 'benchmark_set.json'
                    if bench_json_path.exists() and hasattr(app, 'var_remote_benchmark_set'):
                        app.var_remote_benchmark_set.set(str(bench_json_path))
                        try:
                            if hasattr(app, '_persist_settings'):
                                app._persist_settings()
                        except Exception:
                            pass
                        try:
                            refresh_preview = getattr(app, '_benchmark_refresh_accuracy_ui', None)
                            if callable(refresh_preview):
                                refresh_preview()
                        except Exception:
                            logger.debug('Failed to refresh validate-tab accuracy preview after benchmark generation', exc_info=True)
                except Exception:
                    logger.debug('Failed to auto-select generated benchmark_set.json after generation', exc_info=True)

            if callable(completion_callback):
                try:
                    completion_callback(final_status, str(out_dir), str(bench_log_path), final_msg, final_summary)
                except Exception:
                    logger.debug('Benchmark generation completion callback failed', exc_info=True)

            if show_result_dialogs:
                if final_status in {'ok', 'warn', 'cancelled'} and final_summary:
                    dialog_title = {
                        'ok': 'Benchmark set created',
                        'warn': 'Benchmark set created (with warnings)',
                        'cancelled': 'Benchmark set cancelled',
                    }.get(final_status, 'Benchmark set')
                    show_benchmark_completion_dialog(
                        app,
                        title=dialog_title,
                        summary_data=final_summary,
                        fallback_text=final_msg,
                    )
                elif final_status == 'ok':
                    messagebox.showinfo("Benchmark set created", final_msg)
                elif final_status == 'warn':
                    messagebox.showwarning("Benchmark set created (with warnings)", final_msg)
                elif final_status == 'cancelled':
                    messagebox.showwarning("Benchmark set cancelled", final_msg)
                else:
                    messagebox.showerror("Benchmark set failed", final_msg)

        poll()
        return job_id
