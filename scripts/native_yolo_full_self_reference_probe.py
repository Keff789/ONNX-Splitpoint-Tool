#!/usr/bin/env python3
"""Compare native YOLO output against a Full-ONNX self reference.

This probe deliberately avoids the external ``validation_report.json``.  It uses
exactly the native preprocessed runtime tensor from the boundary manifest, runs
the full ONNX model with ORT and compares it with the attested completed-task
artifact.  Native-Full physical raw heads are boundary evidence only; they are
never decoded as a substitute for an available completed-v2 endpoint.

Use this to isolate per-input native-versus-Full-ONNX semantic agreement from
dataset-wide quality evaluation.  It neither validates nor invalidates
dataset-wide AP/accuracy evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
    evaluate_detection_similarity,
)

try:
    import onnxruntime as ort  # type: ignore
    _ORT_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - helpers remain importable for contract-only diagnostics
    ort = None  # type: ignore
    _ORT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from native_producer_validate_visualize import (  # type: ignore
        _completed_v2_declared,
        _completed_v2_self_reference_detection,
        _decode_layout_candidates,
        _detection_contract_family,
        _expected_detection_contract,
        _match_detections,
        _match_detections_class_agnostic,
        _nms,
        _score_saturation,
    )
    from validate_output_dumps import load_dump, summarize  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"helper import failed: {type(exc).__name__}: {exc}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summ(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    valid = array[finite]
    if valid.size == 0:
        return {"shape": list(array.shape), "dtype": str(array.dtype), "finite": False}
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "finite": bool(finite.all()), "min": float(valid.min()),
        "max": float(valid.max()), "mean": float(valid.mean()),
        "std": float(valid.std()),
        "quantiles": np.quantile(valid, [0, .001, .01, .1, .5, .9, .99, .999, 1]).tolist(),
    }


def _shape_from_ort(shape: Iterable[Any]) -> list[int]:
    result: list[int] = []
    for dimension in shape:
        try:
            value = int(dimension)
            result.append(value if value > 0 else -1)
        except Exception:
            result.append(-1)
    return result


def _ort_dtype(inp: Any):
    value = str(getattr(inp, "type", "") or "").lower()
    if "uint8" in value:
        return np.uint8
    if "int8" in value:
        return np.int8
    if "float16" in value:
        return np.float16
    return np.float32


def _find_full_onnx(benchmark_set: Path, explicit: str = "") -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    models = benchmark_set / "models"
    if models.is_dir():
        candidates = [
            path for path in sorted(models.glob("*.onnx"))
            if "part" not in path.name.lower() and "bridge" not in path.name.lower()
        ] or sorted(models.glob("*.onnx"))
        if candidates:
            return candidates[0].resolve()
    candidates = [
        path for path in sorted(benchmark_set.glob("*.onnx"))
        if "part" not in path.name.lower() and "bridge" not in path.name.lower()
    ]
    if candidates:
        return candidates[0].resolve()
    candidates = [
        path for path in sorted(benchmark_set.glob("**/*.onnx"))
        if "part" not in path.name.lower() and "bridge" not in path.name.lower()
    ]
    return candidates[0].resolve() if candidates else None


def _input_dump_feed_from_manifest(
    manifest: dict[str, Any], input_shape: list[int], scale: str = "native",
) -> np.ndarray | None:
    def _resolve_artifact(value: Any) -> Path | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        manifest_path = Path(str(manifest.get("_manifest_path") or "")).expanduser()
        if manifest_path.is_file():
            rebased = manifest_path.parent / candidate.name if candidate.is_absolute() else manifest_path.parent / candidate
            if rebased.is_file():
                return rebased.resolve()
        return None

    selected_value = manifest.get("selected_input_dump") or manifest.get("selected_input_npy")
    if selected_value:
        selected_path = _resolve_artifact(selected_value)
        if selected_path is None:
            raise RuntimeError("exact_selected_input_dump_missing")
        expected_sha = str(manifest.get("selected_input_dump_sha256") or "").strip().lower()
        if expected_sha.startswith("sha256:"):
            expected_sha = expected_sha[7:]
        actual_sha = hashlib.sha256(selected_path.read_bytes()).hexdigest()
        if not expected_sha or actual_sha != expected_sha:
            raise RuntimeError("exact_selected_input_dump_sha256_mismatch")
        selected = np.load(selected_path, allow_pickle=False)
        expected_shape = [int(x) for x in list(manifest.get("selected_input_shape") or [])]
        expected_dtype = str(manifest.get("selected_input_dtype") or "")
        if expected_shape and list(selected.shape) != expected_shape:
            raise RuntimeError("exact_selected_input_dump_shape_mismatch")
        if expected_dtype and str(selected.dtype) != expected_dtype:
            raise RuntimeError("exact_selected_input_dump_dtype_mismatch")
        array = np.asarray(selected)
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        target_nchw = len(input_shape) == 4 and input_shape[1] in (1, 3, 4)
        target_nhwc = len(input_shape) == 4 and input_shape[-1] in (1, 3, 4)
        if array.ndim != 3:
            raise RuntimeError("exact_selected_input_layout_unsupported")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            chw = array
            hwc = np.transpose(array, (1, 2, 0))
        elif array.shape[-1] in (1, 3, 4):
            hwc = array
            chw = np.transpose(array, (2, 0, 1))
        else:
            raise RuntimeError("exact_selected_input_layout_unsupported")
        mode = str(scale or "native").strip().lower()
        if mode == "native":
            preprocess = manifest.get("preprocess") if isinstance(manifest.get("preprocess"), dict) else {}
            mode = str(preprocess.get("ort_model_scale") or "norm").lower()
        feed = chw[None] if target_nchw else hwc[None] if target_nhwc else None
        if feed is None:
            raise RuntimeError("full_onnx_input_layout_unsupported")
        feed = np.asarray(feed)
        if feed.dtype.kind in {"u", "i"} and mode not in {
            "raw", "resize_raw", "letterbox0_raw", "letterbox114_raw",
            "native_raw", "native_letterbox_raw", "yolo_raw", "yolo_letterbox_raw",
        }:
            feed = feed.astype(np.float32) / 255.0
        return np.ascontiguousarray(feed)

    if int(manifest.get("schema_version") or 0) >= 3 and str(manifest.get("backend") or "") == "deepx_to_trt":
        raise RuntimeError("exact_selected_input_evidence_missing")
    raw_path = manifest.get("input_dump") or manifest.get("preprocessed_input_file")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path).expanduser()
    manifest_path = Path(str(manifest.get("_manifest_path") or "")).expanduser()
    if not path.is_absolute():
        base = (
            manifest_path.parent
            if manifest_path.is_file()
            else Path(str(manifest.get("file") or "")).expanduser().parent
        )
        path = (base / path).resolve()
    elif not path.is_file() and manifest_path.is_file():
        # Legacy manifests record the producer's absolute Remote path.  After
        # the result bundle is extracted locally, the copied input dump lives
        # next to the copied manifest.  Rebase only this legacy byte dump by
        # basename; exact selected_input_dump evidence remains on the strict,
        # hash-verified path above.
        rebased = manifest_path.parent / path.name
        if rebased.is_file():
            path = rebased.resolve()
    if not path.is_file():
        return None
    shape = manifest.get("input_shape_hwc") or (manifest.get("preprocess") or {}).get("input_shape_hwc")
    if not shape:
        if len(input_shape) == 4 and input_shape[1] in (1, 3):
            shape = [int(input_shape[2]), int(input_shape[3]), int(input_shape[1])]
        elif len(input_shape) == 4 and input_shape[-1] in (1, 3):
            shape = [int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
    if not shape or len(shape) != 3:
        return None
    height, width, channels = [int(value) for value in shape]
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size != height * width * channels:
        return None
    array = raw.reshape((height, width, channels)).astype(np.float32)
    mode = str(scale or "native").strip().lower()
    if mode == "native":
        preprocess = manifest.get("preprocess") if isinstance(manifest.get("preprocess"), dict) else {}
        mode = str(preprocess.get("ort_model_scale") or "norm").lower()
    if mode not in {
        "raw", "resize_raw", "letterbox0_raw", "letterbox114_raw",
        "native_raw", "native_letterbox_raw", "yolo_raw", "yolo_letterbox_raw",
    }:
        array /= 255.0
    if len(input_shape) == 4 and input_shape[1] in (1, 3):
        return np.transpose(array, (2, 0, 1))[None].astype(np.float32)
    if len(input_shape) == 4 and input_shape[-1] in (1, 3):
        return array[None].astype(np.float32)
    return None


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    sx = float(np.std(x)); sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return None
    v = float(np.corrcoef(x, y)[0, 1])
    return v if math.isfinite(v) else None


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a); bb = np.asarray(b)
    out: dict[str, Any] = {
        "same_shape": list(aa.shape) == list(bb.shape),
        "shape_a": [int(x) for x in aa.shape],
        "shape_b": [int(x) for x in bb.shape],
        "same_size": int(aa.size) == int(bb.size),
    }
    if aa.size != bb.size or aa.size == 0:
        return out
    x = aa.astype(np.float64).reshape(-1)
    y = bb.astype(np.float64).reshape(-1)
    d = x - y
    out.update({
        "mean_abs": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "corr": _corr(x, y),
        "allclose_1e_4_1e_3": bool(np.allclose(x, y, rtol=1e-4, atol=1e-3)),
        "allclose_1e_3_1e_2": bool(np.allclose(x, y, rtol=1e-3, atol=1e-2)),
    })
    denom = float(np.mean(np.abs(y))) + 1e-9
    out["rel_mae"] = float(out["mean_abs"] / denom)
    return out


def _find_native_output_manifest(man_path: Path, explicit: str = "") -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_file():
            return p.resolve()
        raise FileNotFoundError(p)
    # Standard sibling layout:
    #   .../<precision>/native_fifo_boundary/native_fifo_boundary_manifest.json
    #   .../<precision>/native_fifo_outputs/native_fifo_output_manifest.json
    # Some C++ runners retain the historical plural filename.
    # Hailo10H E2E uses native_outputs/native_outputs_manifest.json.
    for filename in (
        "native_fifo_output_manifest.json",
        "native_fifo_outputs_manifest.json",
    ):
        p = man_path.parent.parent / "native_fifo_outputs" / filename
        if p.is_file():
            return p.resolve()
    p = man_path.parent.parent / "native_outputs" / "native_outputs_manifest.json"
    if p.is_file():
        return p.resolve()
    # Fallback: search nearby precision work dir.
    for name in ("native_fifo_output_manifest.json", "native_fifo_outputs_manifest.json", "runner_outputs_manifest.json", "native_outputs_manifest.json", "output_dump_manifest.json"):
        xs = sorted(man_path.parent.parent.rglob(name))
        if xs:
            return xs[0].resolve()
    return None


def _find_native_report(
    man_path: Path,
    native_manifest: Path,
    explicit: str = "",
) -> Path | None:
    """Find one nearby report which actually declares a completed-v2 endpoint."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()

    candidates = [
        native_manifest.parent / "runtime.json",
        native_manifest.parent.parent / "runtime.json",
        man_path.parent / "runtime.json",
        man_path.parent.parent / "runtime.json",
        native_manifest.parent / "native_report.json",
        native_manifest.parent.parent / "native_report.json",
    ]
    seen: set[Path] = set()
    verified: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            payload = _read_json(resolved)
        except Exception:
            continue
        if _completed_v2_declared(payload):
            verified.append(resolved)
    return verified[0] if len(verified) == 1 else None


def _run_full_onnx(full: Path, man: dict[str, Any], image_scale: str) -> tuple[list[str], list[np.ndarray], np.ndarray, dict[str, Any]]:
    if ort is None:
        raise RuntimeError(f"onnxruntime import failed: {_ORT_IMPORT_ERROR}")
    sess = ort.InferenceSession(str(full), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_shape = _shape_from_ort(inp.shape)
    feed = _input_dump_feed_from_manifest(man, input_shape, image_scale)
    if feed is None:
        raise RuntimeError("boundary manifest has no usable input_dump; rerun native with --native-boundary-debug")
    dtype = _ort_dtype(inp)
    outs = [np.asarray(x) for x in sess.run(None, {input_name: np.asarray(feed, dtype=dtype)})]
    names = [o.name for o in sess.get_outputs()]
    meta = {"input_name": input_name, "input_shape": input_shape, "input_type": str(getattr(inp, "type", "")), "feed_summary": _summ(feed)}
    return names, outs, feed, meta


def _model_size(man: dict[str, Any], feed: np.ndarray) -> tuple[int, int]:
    shp = man.get("input_shape_hwc") or (man.get("preprocess") or {}).get("input_shape_hwc") or []
    if isinstance(shp, list) and len(shp) >= 2:
        return int(shp[1]), int(shp[0])
    f = np.asarray(feed)
    if f.ndim == 4 and f.shape[1] in (1, 3):
        return int(f.shape[3]), int(f.shape[2])
    if f.ndim == 4:
        return int(f.shape[2]), int(f.shape[1])
    return 640, 640


def _candidates(tensors: dict[str, np.ndarray], img_w: int, img_h: int, conf: float) -> list[dict[str, Any]]:
    out = []
    for c in _decode_layout_candidates(tensors, img_w=img_w, img_h=img_h, conf=conf):
        dets = _nms(c.get("detections") or [])
        out.append({
            "mode": c.get("mode"),
            "kind": c.get("kind"),
            "detections": dets,
            "count": len(dets),
            "score_saturation": _score_saturation(dets),
            "debug": c.get("debug") or {},
        })
    return out



def _mode_sanity(mode: str, sat: dict[str, Any], count: int) -> tuple[int, int, int]:
    """Prefer plausible score/class layouts for self-reference.

    A wrong [class,score] interpretation can match Full and Native against each
    other because both are decoded wrongly in the same way.  For YOLO NMS arrays
    prefer layouts with score before class and low score saturation.
    """
    m = str(mode or '')
    satn = int((sat or {}).get('score_ge_0999') or 0)
    low_sat = int(satn <= max(1, int(0.10 * max(1, count))))
    score_class = int('_score_class' in m or m.endswith(':raw'))
    xyxy = int(':nms:xyxy_' in m)
    raw_penalty = int(m.endswith(':raw'))
    class_score_penalty = int('_class_score' in m)
    return (score_class, low_sat, xyxy - raw_penalty - class_score_penalty)


def _best_self_match(
    full_cands: list[dict[str, Any]],
    native_cands: list[dict[str, Any]],
    *,
    expected_family: str = "unknown",
    iou_threshold: float = 0.50,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    expected = str(expected_family or "unknown").strip().lower()
    if expected not in {"raw_head", "decoded_nms"}:
        expected = "unknown"
    for fc in full_cands:
        ref = fc.get("detections") or []
        if not ref:
            continue
        full_family = _detection_contract_family(str(fc.get("mode") or ""))
        for nc in native_cands:
            pred = nc.get("detections") or []
            native_family = _detection_contract_family(str(nc.get("mode") or ""))
            compatible = (
                native_family == full_family
                and native_family != "unknown"
                and (expected == "unknown" or native_family == expected)
            )
            if not compatible:
                rejected.append({
                    "full_mode": fc.get("mode"),
                    "native_mode": nc.get("mode"),
                    "full_contract_family": full_family,
                    "native_contract_family": native_family,
                    "expected_contract_family": expected,
                    "reason": "detection_contract_family_mismatch",
                })
                continue
            m = _match_detections(
                ref, pred, iou_thr=float(iou_threshold),
            )
            ca = _match_detections_class_agnostic(
                ref, pred, iou_thr=float(iou_threshold),
            )
            sat = nc.get("score_saturation") or {}
            same_mode = int(str(fc.get("mode")) == str(nc.get("mode")))
            sanity = _mode_sanity(str(nc.get("mode")), sat, len(pred))
            row = {
                "full_mode": fc.get("mode"),
                "native_mode": nc.get("mode"),
                "full_contract_family": full_family,
                "native_contract_family": native_family,
                "expected_contract_family": expected,
                "contract_family_match": True,
                "full_count": len(ref),
                "native_count": len(pred),
                "match": m,
                "class_agnostic_match": ca,
                "native_score_saturation": sat,
                "full_score_saturation": fc.get("score_saturation") or {},
                "selection_key": [
                    same_mode,
                    *sanity,
                    int(m.get("matched") or 0),
                    float(m.get("match_ratio") or 0.0),
                    float(m.get("mean_iou") or 0.0),
                    int(ca.get("matched") or 0),
                    float(ca.get("match_ratio") or 0.0),
                    -abs(len(pred) - len(ref)),
                    -int(sat.get("score_ge_0999") or 0),
                ],
            }
            rows.append(row)
    rows.sort(key=lambda r: tuple(r.get("selection_key") or []), reverse=True)
    if rows:
        return rows[0], rows, rejected
    empty = {
        "full_mode": "", "native_mode": "", "full_count": 0, "native_count": 0,
        "expected_contract_family": expected, "contract_family_match": False,
        "match": {"matched": 0, "match_ratio": 0.0},
        "class_agnostic_match": {"matched": 0, "match_ratio": 0.0},
    }
    return empty, [], rejected


def _tensor_compares(full_tensors: dict[str, np.ndarray], native_tensors: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for fn, fa in full_tensors.items():
        best_name = ""
        best_cmp: dict[str, Any] = {}
        # Prefer same name + same shape, otherwise any same-size tensor.
        names = list(native_tensors.keys())
        if fn in native_tensors:
            names = [fn] + [n for n in names if n != fn]
        for nn in names:
            if nn in used and len(native_tensors) > 1:
                continue
            na = native_tensors[nn]
            if np.asarray(na).size != np.asarray(fa).size:
                continue
            cmp = _compare(na, fa)
            key = (
                int(cmp.get("same_shape") is True),
                int(cmp.get("allclose_1e_3_1e_2") is True),
                float(cmp.get("corr") if cmp.get("corr") is not None else -999.0),
                -float(cmp.get("rel_mae") if cmp.get("rel_mae") is not None else 999.0),
            )
            bkey = (
                int(best_cmp.get("same_shape") is True),
                int(best_cmp.get("allclose_1e_3_1e_2") is True),
                float(best_cmp.get("corr") if best_cmp.get("corr") is not None else -999.0),
                -float(best_cmp.get("rel_mae") if best_cmp.get("rel_mae") is not None else 999.0),
            ) if best_cmp else (-1, -1, -999.0, -999.0)
            if key > bkey:
                best_name = nn; best_cmp = cmp
        if best_name:
            used.add(best_name)
        rows.append({"full_output": fn, "native_output": best_name, **best_cmp})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--boundary-manifest", required=True)
    ap.add_argument("--native-output-manifest", default="")
    ap.add_argument(
        "--native-report",
        default="",
        help=(
            "Runtime/native report containing the completed-v2 endpoint "
            "attestation and exact completed result artifact."
        ),
    )
    ap.add_argument("--full-onnx", default="")
    ap.add_argument("--image-scale", default="native")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument(
        "--quality-gate-json",
        default="",
        help=(
            "Versioned task-quality policy JSON/path. The Native detection "
            "similarity thresholds are taken from its native_contract block."
        ),
    )
    ap.add_argument("--out", default="")
    ns = ap.parse_args()
    policy = AccuracyGatePolicy.from_mapping(
        ns.quality_gate_json or None
    )
    confidence_threshold = (
        float(ns.conf)
        if ns.conf is not None
        else float(policy.native_self_reference_confidence_threshold)
    )

    bs = Path(ns.benchmark_set).expanduser().resolve()
    man_path = Path(ns.boundary_manifest).expanduser().resolve()
    man = _read_json(man_path)
    man["_manifest_path"] = str(man_path)
    full = _find_full_onnx(bs, ns.full_onnx)
    if full is None:
        raise FileNotFoundError(f"full ONNX not found under {bs}/models")
    native_manifest = _find_native_output_manifest(man_path, ns.native_output_manifest)
    if native_manifest is None:
        raise FileNotFoundError("native output manifest not found; pass --native-output-manifest")
    native_report = _find_native_report(
        man_path, native_manifest, ns.native_report,
    )
    endpoint_evidence = _read_json(native_report) if native_report else {}
    is_native_full = bool(
        man.get("schema") == "onnx-splitpoint/native-full-input-dump"
        or man_path.name == "native_full_input_manifest.json"
    )
    model_id = (
        bs.parent.name
        if bs.name == "benchmark_set" else full.stem
    )
    artifact_identity = {
        "model": model_id,
        "case": str(ns.case),
        "full_onnx_sha256": _sha256_file(full),
        "boundary_manifest_sha256": _sha256_file(man_path),
        "native_output_manifest_sha256": _sha256_file(native_manifest),
        "native_report_sha256": (
            _sha256_file(native_report) if native_report else ""
        ),
    }

    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent / "native_yolo_full_self_reference_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        full_names, full_outs, feed, full_meta = _run_full_onnx(full, man, ns.image_scale)
    except Exception as exc:
        evidence_error = f"{type(exc).__name__}: {exc}"
        payload = {
            "schema": "onnx-splitpoint/native-yolo-full-self-reference-probe",
            "schema_version": 6,
            "ok": False,
            "semantic_ok": False,
            "semantic_available": False,
            "diagnosis": "full_onnx_input_evidence_unavailable",
            "evidence_status": "unavailable_fail_closed",
            "evidence_error": evidence_error,
            "benchmark_set": str(bs),
            "case": ns.case,
            "full_onnx": str(full),
            "boundary_manifest": str(man_path),
            "native_output_manifest": str(native_manifest),
            "native_report": str(native_report) if native_report else "",
            **artifact_identity,
            "best": {},
            "match": {},
            "class_agnostic_match": {},
            "rows": [],
            "tensor_compares": [],
            "reference_detections": [],
            "native_detections": [],
            "note": "No semantic failure is inferred without an exact, hash-verified native input tensor.",
            "numerical_similarity_policy": policy.as_dict(),
            "numerical_similarity_policy_sha256": policy.sha256(),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        out.with_suffix(".md").write_text(
            "# Native YOLO full self-reference probe\n\n"
            "Diagnosis: `full_onnx_input_evidence_unavailable`\n\n"
            f"Evidence error: `{evidence_error}`\n\n"
            "The result is fail-closed `unavailable`; it is not a semantic failure.\n",
            encoding="utf-8",
        )
        print(json.dumps({
            "ok": False,
            "out": str(out),
            "diagnosis": "full_onnx_input_evidence_unavailable",
            "semantic_available": False,
        }, indent=2))
        return 5
    full_tensors = {str(full_names[i] if i < len(full_names) else f"output{i}"): np.asarray(o) for i, o in enumerate(full_outs)}
    native_tensors, native_meta = load_dump(str(native_manifest))

    img_w, img_h = _model_size(man, feed)
    completed_v2 = bool(_completed_v2_declared(endpoint_evidence))
    use_completed_v2 = bool(completed_v2 or is_native_full)
    completed: dict[str, Any] = {}
    full_cands: list[dict[str, Any]] = []
    native_cands: list[dict[str, Any]] = []
    rejected_contract_pairs: list[dict[str, Any]] = []
    tensor_rows = _tensor_compares(full_tensors, native_tensors)
    semantic_available = False
    if use_completed_v2:
        if completed_v2:
            completed = _completed_v2_self_reference_detection(
                full_tensors,
                native_tensors,
                endpoint_evidence,
                policy=policy,
                native_report=native_report,
            )
        else:
            completed = {
                "available": False,
                "completed_v2_verified": False,
                "reason": (
                    "native_full_completed_v2_runtime_report_missing_or_invalid"
                ),
            }
        if completed.get("available") is True:
            semantic_available = True
            best_full_dets = list(completed.get("reference_detections") or [])
            best_native_dets = list(completed.get("native_detections") or [])
            match = _match_detections(
                best_full_dets,
                best_native_dets,
                iou_thr=policy.native_self_reference_iou_threshold,
            )
            class_agnostic_match = _match_detections_class_agnostic(
                best_full_dets,
                best_native_dets,
                iou_thr=policy.native_self_reference_iou_threshold,
            )
            best = {
                "full_mode": str(completed.get("full_mode") or ""),
                "native_mode": str(completed.get("native_mode") or ""),
                "full_count": len(best_full_dets),
                "native_count": len(best_native_dets),
                "match": match,
                "class_agnostic_match": class_agnostic_match,
                "contract_family_match": True,
            }
            rows = [dict(best)]
            expected_family = "decoded_nms"
            expected_family_source = str(
                completed.get("expected_contract_source")
                or "verified_completed_task_comparison_endpoint_v2"
            )
        else:
            best_full_dets = []
            best_native_dets = []
            best = {}
            rows = []
            expected_family = "unknown"
            expected_family_source = "completed_task_comparison_endpoint_v2_invalid"
    else:
        full_cands = _candidates(
            full_tensors, img_w, img_h, confidence_threshold,
        )
        native_cands = _candidates(
            native_tensors, img_w, img_h, confidence_threshold,
        )
        expected_family, expected_family_source = _expected_detection_contract(
            native_manifest,
            native_report=native_report,
            boundary_manifest=man_path,
        )
        best, rows, rejected_contract_pairs = _best_self_match(
            full_cands,
            native_cands,
            expected_family=expected_family,
            iou_threshold=policy.native_self_reference_iou_threshold,
        )
        best_full_dets = next((c.get("detections") for c in full_cands if str(c.get("mode")) == str(best.get("full_mode"))), []) or []
        best_native_dets = next((c.get("detections") for c in native_cands if str(c.get("mode")) == str(best.get("native_mode"))), []) or []
        semantic_available = bool(rows)

    similarity = evaluate_detection_similarity(
        best.get("match") or {},
        best.get("class_agnostic_match") or {},
        policy,
    )
    mr = float((best.get("match") or {}).get("match_ratio") or 0.0)
    ca = float((best.get("class_agnostic_match") or {}).get("match_ratio") or 0.0)
    ref_count = int((best.get("match") or {}).get("ref_count") or best.get("full_count") or 0)
    matched = int((best.get("match") or {}).get("matched") or 0)
    if use_completed_v2 and not semantic_available:
        diagnosis = "completed_v2_evidence_unavailable"
    elif use_completed_v2 and similarity.get("numerical_similarity_pass") is True:
        diagnosis = "native_semantic_matches_full_self_reference"
    elif use_completed_v2:
        diagnosis = "native_semantic_differs_from_full_self_reference"
    elif not full_cands:
        diagnosis = "full_onnx_decode_unavailable"
    elif not native_cands:
        diagnosis = "native_output_decode_unavailable"
    elif not rows and rejected_contract_pairs:
        diagnosis = "detection_contract_family_mismatch"
    elif similarity.get("numerical_similarity_pass") is True:
        diagnosis = "native_semantic_matches_full_self_reference"
    elif ca >= 0.80:
        diagnosis = "native_boxes_match_full_but_class_score_contract_suspect"
    else:
        diagnosis = "native_semantic_differs_from_full_self_reference"

    payload = {
        "schema": "onnx-splitpoint/native-yolo-full-self-reference-probe",
        "schema_version": 6,
        "ok": bool(
            semantic_available
            and diagnosis == "native_semantic_matches_full_self_reference"
        ),
        "semantic_ok": bool(
            semantic_available
            and diagnosis == "native_semantic_matches_full_self_reference"
        ),
        "semantic_available": semantic_available,
        "diagnosis": diagnosis,
        "expected_contract_family": expected_family,
        "expected_contract_source": expected_family_source,
        "contract_family_match": bool(best.get("contract_family_match")),
        "rejected_contract_pairs": rejected_contract_pairs[:200],
        "benchmark_set": str(bs),
        "case": ns.case,
        "full_onnx": str(full),
        "boundary_manifest": str(man_path),
        "native_output_manifest": str(native_manifest),
        "native_report": str(native_report) if native_report else "",
        **artifact_identity,
        "native_output_manifest_meta": native_meta,
        "full_input": full_meta,
        "model_size": {"w": img_w, "h": img_h},
        "full_output_summaries": {k: summarize(v) for k, v in full_tensors.items()},
        "native_output_summaries": {k: summarize(v) for k, v in native_tensors.items()},
        "best": best,
        "match": best.get("match") or {},
        "class_agnostic_match": best.get("class_agnostic_match") or {},
        "decode_mode": best.get("native_mode") or "",
        "reference_detections": best_full_dets[:300],
        "native_detections": best_native_dets[:300],
        "rows": rows,
        "tensor_compares": tensor_rows,
        "full_candidates": [{k: v for k, v in c.items() if k != "detections"} for c in full_cands],
        "native_candidates": [{k: v for k, v in c.items() if k != "detections"} for c in native_cands],
        "completed_v2_evidence": {
            key: value for key, value in completed.items()
            if key not in {"reference_detections", "native_detections"}
        },
        "note": (
            "Native Full semantic evidence is the verified completed-v2 task "
            "artifact; physical outputs are retained only as boundary evidence."
            if use_completed_v2 else
            "External validation_report.json is intentionally not used by this legacy split probe."
        ),
        "numerical_similarity_policy": policy.as_dict(),
        "numerical_similarity_policy_sha256": policy.sha256(),
        **similarity,
    }

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = [
        "# Native YOLO full self-reference probe",
        "",
        f"Diagnosis: `{diagnosis}`",
        f"Expected contract family: `{expected_family}` ({expected_family_source})",
        f"Contract family match: `{bool(best.get('contract_family_match'))}`",
        f"Benchmark set: `{bs}`",
        f"Case: `{ns.case}`",
        f"Full ONNX: `{full}`",
        f"Boundary manifest: `{man_path}`",
        f"Native output manifest: `{native_manifest}`",
        "",
        "## Summary",
        "",
        f"Best full mode: `{best.get('full_mode','')}`",
        f"Best native mode: `{best.get('native_mode','')}`",
        f"Match: `{matched}/{ref_count}` ratio=`{mr}`",
        f"Class-agnostic ratio: `{ca}`",
        (
            "Policy: "
            f"`{similarity.get('numerical_similarity_policy_id')}`; "
            f"match ≥ `{similarity.get('numerical_similarity_threshold')}`; "
            "mean IoU ≥ "
            f"`{similarity.get('numerical_similarity_mean_iou_threshold')}`"
        ),
        "",
        "## Best self-reference candidates",
        "",
        "| rank | full mode | native mode | full pred | native pred | match | class-agnostic | native score>=0.999 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows[:30]):
        m = r.get("match") or {}; cg = r.get("class_agnostic_match") or {}; sat = r.get("native_score_saturation") or {}
        lines.append(
            f"| {i} | `{r.get('full_mode')}` | `{r.get('native_mode')}` | {r.get('full_count')} | {r.get('native_count')} | "
            f"{m.get('matched')}/{m.get('ref_count')} ({m.get('match_ratio')}) | "
            f"{cg.get('matched')}/{cg.get('ref_count')} ({cg.get('match_ratio')}) | {sat.get('score_ge_0999')} |"
        )
    lines += [
        "",
        "## Tensor compare: native output vs Full ONNX output",
        "",
        "| full output | native output | same shape | mean abs | max abs | rel MAE | corr | allclose 1e-3/1e-2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in tensor_rows:
        lines.append(
            f"| `{r.get('full_output')}` | `{r.get('native_output','')}` | {r.get('same_shape','')} | "
            f"{r.get('mean_abs','')} | {r.get('max_abs','')} | {r.get('rel_mae','')} | {r.get('corr','')} | {r.get('allclose_1e_3_1e_2','')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `native_semantic_matches_full_self_reference`: for the tested input, native and Full ONNX completed-task detections satisfy the configured self-reference similarity policy. This single-input semantic smoke does not evaluate or override dataset-wide AP/accuracy, which remains separate quality evidence.",
        "- `native_boxes_match_full_but_class_score_contract_suspect`: boxes match but class/score columns differ; inspect NMS output layout/class mapping.",
        "- `native_semantic_differs_from_full_self_reference`: native output still differs from Full ONNX even without the external reference; inspect boundary/HEF/TRT path.",
        "- `completed_v2_evidence_unavailable`: the completed-task attestation/result is missing or invalid; the probe fails closed and does not decode physical raw heads as a substitute.",
        "- `full_onnx_decode_unavailable`: the decoder itself cannot read the Full ONNX output; fix YOLO decoder before judging native output.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": bool(payload["ok"]),
        "out": str(out),
        "summary_md": str(md),
        "diagnosis": diagnosis,
        "best_match_ratio": mr,
        "best_class_agnostic_ratio": ca,
        "best_full_mode": best.get("full_mode", ""),
        "best_native_mode": best.get("native_mode", ""),
        "semantic_available": bool(payload["semantic_available"]),
    }, indent=2))
    return 0 if payload["ok"] and payload["semantic_available"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
