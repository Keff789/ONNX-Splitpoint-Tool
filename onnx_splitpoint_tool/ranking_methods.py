"""Pre-registered ranking baselines for scientific split-point evaluation.

The module provides the pre-registered ranking methods used by the scientific
reporting contract:

``cut_bytes_only``
    Rank by the amount of data crossing the boundary.
``weighted_score``
    The historic communication/compute-imbalance/crossing-tensor score.
``cycle_time_no_handover``
    Rank by the predicted pipeline bottleneck without a boundary handover term.
``cycle_time_with_handover``
    Rank by the predicted bottleneck plus a runner- and direction-specific
    handover model.
``onnx_real_boundary_hardware_aware``
    The former hardware-aware accelerator-fit workflow selector, retained as
    a comparison baseline after the Cut Bytes workflow-ranker freeze.

All methods are deterministic functions of the frozen candidate universe and a
versioned profile block.  Lower values are always better.  The module does not
fit coefficients from hold-out measurements; fitted handover models must be
supplied through the evaluation profile or through candidate fields that were
created before the benchmark run.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from .objective_scoring import accelerator_fit_metrics, predicted_handover_ms

RANKING_METHOD_SCHEMA = "onnx-splitpoint/ranking-method-predictions"
RANKING_METHOD_SCHEMA_VERSION = 2
RANKING_METHOD_IMPLEMENTATION = "v277-cut-bytes-only-workflow-freeze-1"
WORKFLOW_RANKING_METHOD = "cut_bytes_only"

METHOD_ORDER: tuple[str, ...] = (
    "cut_bytes_only",
    "weighted_score",
    "cycle_time_no_handover",
    "cycle_time_with_handover",
    "onnx_real_boundary_hardware_aware",
)

METHOD_LABELS: dict[str, str] = {
    "cut_bytes_only": "Cut bytes only",
    "weighted_score": "Weighted score",
    "cycle_time_no_handover": "Cycle time without handover",
    "cycle_time_with_handover": "Cycle time with runner/direction handover",
    "onnx_real_boundary_hardware_aware": "Hardware-aware ONNX-boundary baseline",
}

METHOD_UNITS: dict[str, str] = {
    "cut_bytes_only": "bytes",
    "weighted_score": "score",
    "cycle_time_no_handover": "ms",
    "cycle_time_with_handover": "ms",
    "onnx_real_boundary_hardware_aware": "score",
}


def _f(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def _i(value: Any) -> Optional[int]:
    number = _f(value)
    if number is None:
        return None
    try:
        return int(number)
    except Exception:
        return None


def _first(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _float_or_default(value: Any, default: float) -> float:
    parsed = _f(value)
    return float(default if parsed is None else parsed)


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text


def canonical_backend(value: Any) -> str:
    text = _slug(value)
    if text in {"trt", "tensor_rt", "onnxruntime_tensorrt", "ort_tensorrt", "tensorrt_ep"}:
        return "tensorrt"
    if "tensorrt" in text or text.endswith("_trt") or text.startswith("trt_"):
        return "tensorrt"
    if text in {"cuda", "cuda_ort", "ort_cuda", "onnxruntime_cuda"}:
        return "cuda"
    if text in {"cpu", "cpu_ort", "ort_cpu", "onnxruntime_cpu"}:
        return "cpu"
    if "hailo10h" in text:
        return "hailo10h"
    if "hailo10" in text:
        # The evaluation profiles historically used the family alias
        # ``hailo10`` while Native evidence uses the concrete Hailo-10H name.
        # They describe the same device in this tool; retaining two canonical
        # values split otherwise identical ranking/reporting strata.
        return "hailo10h"
    if "hailo8r" in text:
        return "hailo8r"
    if "hailo8l" in text:
        return "hailo8l"
    if "hailo8" in text or text == "hailo":
        return "hailo8"
    if "deepx" in text or text in {"dx", "dx_m1", "dxm1"}:
        return "deepx_m1"
    return text


def canonical_runner(value: Any) -> str:
    text = _slug(value)
    if "native" in text or "fifo" in text:
        return "native_fifo"
    return "generic"


def canonical_direction(value: Any = "", *, stage1: Any = "", stage2: Any = "") -> str:
    text = _slug(value)
    if "_to_" in text:
        left, right = text.split("_to_", 1)
        return f"{canonical_backend(left)}_to_{canonical_backend(right)}"
    left = canonical_backend(stage1)
    right = canonical_backend(stage2)
    return f"{left}_to_{right}" if left and right else text


def direction_parts(direction: str) -> tuple[str, str]:
    norm = canonical_direction(direction)
    if "_to_" in norm:
        return tuple(norm.split("_to_", 1))  # type: ignore[return-value]
    return "", ""


def candidate_case_id(candidate: Mapping[str, Any], index: int = 0) -> str:
    case_id = str(candidate.get("case_id") or candidate.get("case") or "").strip()
    if case_id:
        return case_id
    boundary = _i(_first(candidate, ("boundary", "split_index", "boundary_index")))
    return f"b{boundary:03d}" if boundary is not None else f"candidate_{index:04d}"


def candidate_cut_bytes(candidate: Mapping[str, Any]) -> Optional[float]:
    direct = _f(_first(candidate, ("cut_bytes", "cost_bytes", "crossing_bytes", "communication_bytes")))
    if direct is not None:
        return max(0.0, direct)
    mib = _f(_first(candidate, ("cut_mib_val", "cut_mib", "communication_mib")))
    if mib is not None:
        return max(0.0, mib * (1024.0 ** 2))
    mb = _f(_first(candidate, ("cut_mb_val", "cut_mb", "communication_mb")))
    if mb is not None:
        return max(0.0, mb * 1_000_000.0)
    return None


def candidate_cut_mib(candidate: Mapping[str, Any]) -> Optional[float]:
    direct = _f(_first(candidate, ("cut_mib_val", "cut_mib", "communication_mib")))
    if direct is not None:
        return max(0.0, direct)
    cut_bytes = candidate_cut_bytes(candidate)
    return (cut_bytes / (1024.0 ** 2)) if cut_bytes is not None else None


def _candidate_identity_tie_key(
    candidate: Mapping[str, Any], index: int = 0,
) -> tuple[int, str]:
    """Return the frozen deterministic tie order: boundary, then case ID."""
    boundary = _i(
        _first(candidate, ("boundary", "split_index", "boundary_index"))
    )
    return (
        boundary if boundary is not None else 10**9,
        candidate_case_id(candidate, index),
    )


def cut_bytes_only_sort_key(
    candidate: Mapping[str, Any], index: int = 0,
) -> tuple[int, float, int, str]:
    """Rank smaller boundary payloads first with the frozen identity tie-break.

    Missing/non-finite communication estimates sort after valid estimates.  No
    compute, topology, feasibility, or model-specific term participates in the
    key.
    """
    cut_bytes = candidate_cut_bytes(candidate)
    boundary, case_id = _candidate_identity_tie_key(candidate, index)
    return (
        0 if cut_bytes is not None else 1,
        float(cut_bytes) if cut_bytes is not None else math.inf,
        boundary,
        case_id,
    )


def candidate_imbalance(candidate: Mapping[str, Any]) -> Optional[float]:
    direct = _f(_first(candidate, ("imbalance_val", "imbalance_pred", "pred_imbalance", "imbalance")))
    if direct is not None:
        return max(0.0, direct)
    left = _f(_first(candidate, ("flops_left_abs", "flops_left", "left_flops")))
    right = _f(_first(candidate, ("flops_right_abs", "flops_right", "right_flops")))
    total = _f(candidate.get("total_flops"))
    if left is not None and right is None and total is not None:
        right = max(0.0, total - left)
    if left is not None and right is not None and left + right > 0:
        return abs(left - right) / (left + right)
    return None


def candidate_crossing_tensors(candidate: Mapping[str, Any]) -> Optional[int]:
    direct = _i(_first(candidate, ("n_cut_tensors", "crossing_tensor_count", "cut_tensor_count")))
    if direct is not None:
        return max(0, direct)
    tensors = candidate.get("crossing_tensors")
    if isinstance(tensors, (list, tuple)):
        return len(tensors)
    return None


def candidate_unknown_tensors(candidate: Mapping[str, Any]) -> int:
    direct = _i(_first(candidate, ("unknown_count", "unknown_crossing_tensors", "unknown_tensor_count")))
    if direct is not None:
        return max(0, direct)
    value = candidate.get("unknown_crossings")
    return len(value) if isinstance(value, (list, tuple)) else 0


def candidate_peak_right_mib(candidate: Mapping[str, Any]) -> float:
    return max(0.0, float(_f(_first(candidate, ("peak_right_mib_val", "peak_act_right_mib", "peak_right_mib"))) or 0.0))


def _stage_times_from_candidate(candidate: Mapping[str, Any], stage1: str, stage2: str, policy: Mapping[str, Any]) -> tuple[Optional[float], Optional[float], str]:
    # Prefer direction/backend-specific fields when the analysis already wrote
    # them.  This keeps fitted development models usable without duplicating
    # their logic in the reporter.
    s1 = canonical_backend(stage1)
    s2 = canonical_backend(stage2)
    stage1_fields = (
        f"predicted_{s1}_stage1_ms", f"predicted_stage1_{s1}_ms",
        "predicted_stage1_ms", "stage1_predicted_ms", "part1_predicted_ms",
    )
    stage2_fields = (
        f"predicted_{s2}_stage2_ms", f"predicted_stage2_{s2}_ms",
        "predicted_stage2_ms", "stage2_predicted_ms", "part2_predicted_ms",
    )
    t1 = _f(_first(candidate, stage1_fields))
    t2 = _f(_first(candidate, stage2_fields))
    if t1 is not None and t2 is not None:
        return max(0.0, t1), max(0.0, t2), "candidate_stage_times"

    left = _f(_first(candidate, ("flops_left_abs", "flops_left", "left_flops")))
    right = _f(_first(candidate, ("flops_right_abs", "flops_right", "right_flops")))
    total = _f(candidate.get("total_flops"))
    if left is not None and right is None and total is not None:
        right = max(0.0, total - left)

    # Development-fitted affine stage-time models take precedence over the
    # older zero-intercept throughput approximation.  They are still simple
    # and interpretable: t_ms = intercept_ms + per_gflop_ms * FLOPs_GFLOP.
    stage_models = policy.get("stage_time_models") if isinstance(policy.get("stage_time_models"), Mapping) else {}
    def _model_time(backend: str, flops: Optional[float]) -> Optional[float]:
        if flops is None or not isinstance(stage_models, Mapping):
            return None
        model = stage_models.get(backend)
        if not isinstance(model, Mapping) or str(model.get("status") or "fitted") not in {"fitted", "ready", "ok"}:
            return None
        mode = _slug(model.get("mode") or "affine_flops")
        if mode in {"affine_flops", "affine", "linear"}:
            intercept = float(_f(model.get("intercept_ms")) or 0.0)
            slope = _f(model.get("per_gflop_ms"))
            if slope is None:
                throughput_model = _f(model.get("throughput_gops"))
                slope = 1000.0 / throughput_model if throughput_model not in (None, 0) else None
            return max(0.0, intercept + float(slope) * (flops / 1e9)) if slope is not None else None
        if mode in {"throughput", "gops"}:
            throughput_model = _f(model.get("throughput_gops"))
            return flops / (throughput_model * 1e9) * 1000.0 if throughput_model not in (None, 0) else None
        return None

    model_t1 = _model_time(s1, left)
    model_t2 = _model_time(s2, right)
    if model_t1 is not None and model_t2 is not None:
        return model_t1, model_t2, "profile_fitted_stage_time_models"

    throughput = policy.get("backend_throughput_gops") if isinstance(policy.get("backend_throughput_gops"), Mapping) else {}
    p1 = _f(throughput.get(s1)) if isinstance(throughput, Mapping) else None
    p2 = _f(throughput.get(s2)) if isinstance(throughput, Mapping) else None
    if left is not None and right is not None and p1 not in (None, 0) and p2 not in (None, 0):
        return left / (p1 * 1e9) * 1000.0, right / (p2 * 1e9) * 1000.0, "profile_backend_throughput"

    bottleneck = _f(_first(candidate, ("predicted_bottleneck_ms", "cycle_time_no_handover_ms")))
    if bottleneck is not None:
        # A split between the two stages is not recoverable from a bottleneck
        # alone.  Returning it twice preserves the intended max() value.
        return bottleneck, bottleneck, "candidate_bottleneck"

    total_latency = _f(_first(candidate, ("predicted_total_latency_ms", "pred_latency_total_ms", "objective_latency_ms")))
    if total_latency is not None:
        left = _f(_first(candidate, ("flops_left_abs", "flops_left", "left_flops")))
        right = _f(_first(candidate, ("flops_right_abs", "flops_right", "right_flops")))
        total = _f(candidate.get("total_flops"))
        if left is not None and right is None and total is not None:
            right = max(0.0, total - left)
        if left is not None and right is not None and left + right > 0:
            t1 = total_latency * left / (left + right)
            t2 = total_latency * right / (left + right)
            return t1, t2, "latency_flop_fraction_fallback"
        return total_latency * 0.5, total_latency * 0.5, "latency_half_fallback"
    return None, None, "stage_time_unavailable"


def _model_candidates(direction: str) -> list[str]:
    norm = canonical_direction(direction)
    left, right = direction_parts(norm)
    keys = [norm]
    if left.startswith("hailo"):
        keys.append(f"hailo_to_{right}")
    if right.startswith("hailo"):
        keys.append(f"{left}_to_hailo")
    if left.startswith("deepx"):
        keys.append(f"deepx_to_{right}")
    if right.startswith("deepx"):
        keys.append(f"{left}_to_deepx")
    keys.append("default")
    return list(dict.fromkeys(keys))


def _handover_model(policy: Mapping[str, Any], runner: str, direction: str) -> Mapping[str, Any]:
    cycle_cfg = policy.get("cycle_time_with_handover") if isinstance(policy.get("cycle_time_with_handover"), Mapping) else {}
    models = cycle_cfg.get("handover_models") if isinstance(cycle_cfg.get("handover_models"), Mapping) else {}
    runner_block = models.get(canonical_runner(runner)) if isinstance(models, Mapping) else None
    if not isinstance(runner_block, Mapping):
        return {}
    for key in _model_candidates(direction):
        model = runner_block.get(key)
        if isinstance(model, Mapping):
            return dict(model)
    return {}


def _candidate_field_handover(candidate: Mapping[str, Any], runner: str) -> tuple[Optional[float], str]:
    runner = canonical_runner(runner)
    if runner == "native_fifo":
        fields = (
            "predicted_handover_ms_native_fifo", "native_fifo_predicted_handover_ms",
            "predicted_native_handoff_ms", "predicted_native_handover_ms",
        )
    else:
        fields = (
            "predicted_handover_ms_calibrated", "predicted_handover_ms",
            "predicted_handover_ms_uncalibrated", "predicted_transfer_latency_ms",
        )
    value = _f(_first(candidate, fields))
    return (max(0.0, value), f"candidate_field:{next((field for field in fields if candidate.get(field) not in (None, '')), '')}") if value is not None else (None, "candidate_field_unavailable")


def runner_direction_handover_ms(
    candidate: Mapping[str, Any],
    *,
    stage1: str,
    stage2: str,
    runner_regime: str,
    policy: Mapping[str, Any],
) -> tuple[Optional[float], str]:
    """Return a prospective handover prediction and an auditable source label."""
    runner = canonical_runner(runner_regime)
    direction = canonical_direction(stage1=stage1, stage2=stage2)
    model = _handover_model(policy, runner, direction)
    mode = _slug(model.get("mode") if model else "")

    # Generic runs can use the existing interpretable direction-specific model.
    if not mode and runner == "generic":
        mode = "objective_scoring_directional"
    # Native runs intentionally do not fall back to the Generic Runner model.
    # A native model must be pre-fitted/configured or written into the frozen
    # candidate rows before the hold-out is opened.
    if not mode and runner == "native_fifo":
        value, source = _candidate_field_handover(candidate, runner)
        return value, source if value is not None else "native_handover_model_unconfigured"

    if mode in {"objective_scoring", "objective_scoring_directional", "directional"}:
        value = predicted_handover_ms(
            cut_mib=candidate_cut_mib(candidate),
            n_cut_tensors=candidate_crossing_tensors(candidate),
            unknown_crossing_tensors=candidate_unknown_tensors(candidate),
            peak_act_right_mib=candidate_peak_right_mib(candidate),
            compile_risk_score=candidate.get("hailo_compile_risk_score"),
            single_context_probability=candidate.get("hailo_single_context_probability"),
            fallback_used=bool(candidate.get("hailo_part2_fallback_used")),
            parse_ok=candidate.get("hailo_parse_ok"),
            stage1=stage1,
            stage2=stage2,
            imbalance=candidate_imbalance(candidate),
            use_calibration=bool(model.get("use_calibration", True) if model else True),
        )
        return value, "objective_scoring_directional" if value is not None else "objective_scoring_unavailable"

    if mode in {"candidate", "candidate_field", "fields"}:
        fields = model.get("fields") if isinstance(model.get("fields"), Sequence) and not isinstance(model.get("fields"), (str, bytes)) else []
        value = _f(_first(candidate, tuple(str(x) for x in fields))) if fields else None
        if value is None:
            return _candidate_field_handover(candidate, runner)
        return max(0.0, value), "configured_candidate_field"

    if mode in {"constant", "fixed"}:
        value = _f(model.get("value_ms", model.get("intercept_ms")))
        return (max(0.0, value), "configured_constant") if value is not None else (None, "invalid_constant_model")

    cut_mib = float(candidate_cut_mib(candidate) or 0.0)
    n_cut = float(candidate_crossing_tensors(candidate) or 0)
    unknown = float(candidate_unknown_tensors(candidate))
    peak_right = float(candidate_peak_right_mib(candidate))
    imbalance = float(candidate_imbalance(candidate) or 0.0)
    base, _ = _candidate_field_handover(candidate, "generic")

    if mode in {"bandwidth", "fixed_plus_bandwidth", "transfer"}:
        intercept = float(_f(model.get("intercept_ms")) or 0.0)
        bw = _f(model.get("bandwidth_mib_per_ms"))
        if bw in (None, 0):
            bw_gib_s = _f(model.get("bandwidth_gib_per_s"))
            bw = (bw_gib_s * 1024.0 / 1000.0) if bw_gib_s not in (None, 0) else None
        if bw in (None, 0):
            return None, "invalid_bandwidth_model"
        value = intercept + cut_mib / bw
        value += float(_f(model.get("per_tensor_ms")) or 0.0) * max(0.0, n_cut - 1.0)
        value += float(_f(model.get("per_unknown_ms")) or 0.0) * unknown
        return max(0.0, value), "configured_bandwidth_model"

    if mode in {"affine", "affine_features", "linear_features"}:
        value = float(_f(model.get("intercept_ms")) or 0.0)
        value += _float_or_default(model.get("per_mib_ms", model.get("cut_mib_ms")), 0.0) * cut_mib
        value += _float_or_default(model.get("per_tensor_ms", model.get("crossing_tensor_ms")), 0.0) * max(0.0, n_cut - 1.0)
        value += _float_or_default(model.get("per_unknown_ms", model.get("unknown_tensor_ms")), 0.0) * unknown
        value += _float_or_default(model.get("per_peak_right_mib_ms", model.get("peak_right_mib_ms")), 0.0) * peak_right
        value += _float_or_default(model.get("per_imbalance_ms", model.get("imbalance_ms")), 0.0) * imbalance
        if base is not None:
            value += float(_f(model.get("candidate_handover_scale")) or 0.0) * base
        return max(0.0, value), "configured_affine_feature_model"

    return None, f"unsupported_handover_model:{mode or 'none'}"


def ranking_method_policy(raw: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    block = dict(raw or {})
    methods_raw = block.get("methods") or block.get("ranking_methods") or list(METHOD_ORDER)
    if isinstance(methods_raw, Mapping):
        methods = [str(k) for k, enabled in methods_raw.items() if bool(enabled)]
    elif isinstance(methods_raw, Sequence) and not isinstance(methods_raw, (str, bytes)):
        methods = [str(x) for x in methods_raw]
    else:
        methods = list(METHOD_ORDER)
    methods = [m for m in METHOD_ORDER if m in methods]
    if not methods:
        methods = list(METHOD_ORDER)

    weighted = block.get("weighted_score") if isinstance(block.get("weighted_score"), Mapping) else {}
    cycle_no = block.get("cycle_time_no_handover") if isinstance(block.get("cycle_time_no_handover"), Mapping) else {}
    cycle_with = block.get("cycle_time_with_handover") if isinstance(block.get("cycle_time_with_handover"), Mapping) else {}
    return {
        "schema": "onnx-splitpoint/ranking-method-policy",
        "schema_version": 1,
        "implementation": RANKING_METHOD_IMPLEMENTATION,
        "workflow_ranking_method": WORKFLOW_RANKING_METHOD,
        "methods": methods,
        "weighted_score": {
            "w_comm": _float_or_default(weighted.get("w_comm"), 1.0),
            "w_imb": _float_or_default(weighted.get("w_imb"), 3.0),
            "w_tensors": _float_or_default(weighted.get("w_tensors"), 0.2),
            "log_comm": bool(weighted.get("log_comm", True)),
        },
        "cycle_time_no_handover": {
            "backend_throughput_gops": dict(cycle_no.get("backend_throughput_gops") or {}) if isinstance(cycle_no.get("backend_throughput_gops"), Mapping) else {},
            "stage_time_models": dict(cycle_no.get("stage_time_models") or {}) if isinstance(cycle_no.get("stage_time_models"), Mapping) else {},
        },
        "cycle_time_with_handover": {
            "backend_throughput_gops": dict(cycle_with.get("backend_throughput_gops") or cycle_no.get("backend_throughput_gops") or {}) if isinstance(cycle_with.get("backend_throughput_gops") or cycle_no.get("backend_throughput_gops"), Mapping) else {},
            "stage_time_models": dict(cycle_with.get("stage_time_models") or cycle_no.get("stage_time_models") or {}) if isinstance(cycle_with.get("stage_time_models") or cycle_no.get("stage_time_models"), Mapping) else {},
            "handover_models": dict(cycle_with.get("handover_models") or block.get("handover_models") or {}) if isinstance(cycle_with.get("handover_models") or block.get("handover_models"), Mapping) else {},
        },
    }


def ranking_contexts_from_profile(profile: Mapping[str, Any]) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    for item in list(profile.get("run_profiles") or []):
        if not isinstance(item, Mapping):
            continue
        stage1 = canonical_backend(item.get("stage1"))
        stage2 = canonical_backend(item.get("stage2"))
        if not stage1 or not stage2 or stage1 == stage2:
            continue
        runner = canonical_runner(item.get("runner") or item.get("type") or item.get("id"))
        # A normal mixed_backend run is Generic Runner evidence even though its
        # identifier may contain a backend name.  Only explicit native/fifo
        # markers select the native regime.
        if str(item.get("type") or "").lower() == "mixed_backend" and "native" not in str(item.get("id") or "").lower():
            runner = "generic"
        contexts.append({"direction": canonical_direction(stage1=stage1, stage2=stage2), "stage1": stage1, "stage2": stage2, "runner_regime": runner})

    native = profile.get("native_producers") if isinstance(profile.get("native_producers"), Mapping) else {}
    if native and bool(native.get("enabled", True)):
        consumer = canonical_backend(native.get("consumer") or native.get("stage2") or "tensorrt")
        for backend in list(native.get("backends") or []):
            producer = canonical_backend(backend)
            if producer and consumer and producer != consumer:
                contexts.append({"direction": canonical_direction(stage1=producer, stage2=consumer), "stage1": producer, "stage2": consumer, "runner_regime": "native_fifo"})

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for context in contexts:
        unique[(context["direction"], context["runner_regime"])] = context
    return list(unique.values())


def _normalise(values: Sequence[Optional[float]]) -> list[Optional[float]]:
    valid = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not valid:
        return [None for _ in values]
    lo, hi = min(valid), max(valid)
    den = hi - lo
    if den <= 0:
        return [0.0 if v is not None else None for v in values]
    return [((float(v) - lo) / den) if v is not None else None for v in values]


def compute_ranking_predictions(
    candidates: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    policy: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Compute long-form, pre-benchmark predictions for all configured methods."""
    cfg = ranking_method_policy(policy)
    candidate_rows = [dict(c) for c in candidates if isinstance(c, Mapping) and c.get("strict_ok") is not False]
    if not candidate_rows:
        return []
    contexts_norm = [dict(c) for c in contexts if isinstance(c, Mapping)] or [{"direction": "", "stage1": "", "stage2": "", "runner_regime": "generic"}]

    cut_values = [candidate_cut_bytes(c) for c in candidate_rows]
    imbalances = [candidate_imbalance(c) for c in candidate_rows]
    tensor_values = [float(max(0, (candidate_crossing_tensors(c) or 1) - 1)) for c in candidate_rows]
    weighted_cfg = cfg["weighted_score"]
    comm_raw = [math.log10(1.0 + float(v)) if v is not None and weighted_cfg["log_comm"] else v for v in cut_values]
    comm_n = _normalise(comm_raw)
    imb_n = _normalise(imbalances)
    ten_n = _normalise(tensor_values)
    weighted_values: list[Optional[float]] = []
    for a, b, c in zip(comm_n, imb_n, ten_n):
        if a is None or b is None or c is None:
            weighted_values.append(None)
        else:
            weighted_values.append(weighted_cfg["w_comm"] * a + weighted_cfg["w_imb"] * b + weighted_cfg["w_tensors"] * c)

    output: list[dict[str, Any]] = []
    for context in contexts_norm:
        direction = canonical_direction(context.get("direction"), stage1=context.get("stage1"), stage2=context.get("stage2"))
        stage1, stage2 = direction_parts(direction)
        runner = canonical_runner(context.get("runner_regime"))
        method_values: dict[str, list[tuple[Optional[float], str, Optional[float], Optional[float]]]] = {m: [] for m in cfg["methods"]}
        for idx, candidate in enumerate(candidate_rows):
            no_policy = cfg.get("cycle_time_no_handover") or {}
            t1, t2, stage_source = _stage_times_from_candidate(candidate, stage1, stage2, no_policy)
            bottleneck = max(t1, t2) if t1 is not None and t2 is not None else None
            handover, handover_source = runner_direction_handover_ms(candidate, stage1=stage1, stage2=stage2, runner_regime=runner, policy=cfg)
            cycle = bottleneck + handover if bottleneck is not None and handover is not None else None
            if "cut_bytes_only" in method_values:
                method_values["cut_bytes_only"].append((cut_values[idx], "candidate_cut_bytes" if cut_values[idx] is not None else "cut_bytes_unavailable", bottleneck, handover))
            if "weighted_score" in method_values:
                method_values["weighted_score"].append((weighted_values[idx], "historic_weighted_score" if weighted_values[idx] is not None else "weighted_features_unavailable", bottleneck, handover))
            if "cycle_time_no_handover" in method_values:
                method_values["cycle_time_no_handover"].append((bottleneck, stage_source, bottleneck, 0.0 if bottleneck is not None else None))
            if "cycle_time_with_handover" in method_values:
                source = f"{stage_source}+{handover_source}"
                method_values["cycle_time_with_handover"].append((cycle, source, bottleneck, handover))
            if "onnx_real_boundary_hardware_aware" in method_values:
                fit = _f(accelerator_fit_metrics(dict(candidate), stage1=stage1, stage2=stage2).get("accelerator_fit_score"))
                method_values["onnx_real_boundary_hardware_aware"].append((fit, "objective_scoring.accelerator_fit_metrics" if fit is not None else "hardware_aware_features_unavailable", bottleneck, handover))

        for method_id in cfg["methods"]:
            values = method_values.get(method_id) or []
            sortable = sorted(
                ((float(value), idx) for idx, (value, _source, _b, _h) in enumerate(values) if value is not None and math.isfinite(float(value))),
                key=lambda pair: (
                    pair[0],
                    *_candidate_identity_tie_key(
                        candidate_rows[pair[1]], pair[1] + 1,
                    ),
                ),
            )
            ranks = {idx: rank for rank, (_value, idx) in enumerate(sortable, start=1)}
            for idx, candidate in enumerate(candidate_rows):
                value, source, bottleneck, handover = values[idx]
                output.append({
                    "schema": RANKING_METHOD_SCHEMA,
                    "schema_version": RANKING_METHOD_SCHEMA_VERSION,
                    "implementation": RANKING_METHOD_IMPLEMENTATION,
                    "case_id": candidate_case_id(candidate, idx + 1),
                    "boundary": _i(_first(candidate, ("boundary", "split_index", "boundary_index"))),
                    "direction": direction,
                    "stage1": stage1,
                    "stage2": stage2,
                    "runner_regime": runner,
                    "method_id": method_id,
                    "method_label": METHOD_LABELS[method_id],
                    "prediction_unit": METHOD_UNITS[method_id],
                    "predicted_value": value,
                    "predicted_rank": ranks.get(idx),
                    "prediction_available": value is not None,
                    "prediction_source": source,
                    "predicted_bottleneck_ms": bottleneck,
                    "predicted_handover_ms": handover,
                    "cut_bytes": cut_values[idx],
                    "cut_mib": candidate_cut_mib(candidate),
                    "imbalance": imbalances[idx],
                    "n_cut_tensors": candidate_crossing_tensors(candidate),
                    "unknown_crossing_tensors": candidate_unknown_tensors(candidate),
                    "strict_ok": candidate.get("strict_ok"),
                })
    return output


def method_macro_sort_key(row: Mapping[str, Any], primary_k: int = 5) -> tuple[Any, ...]:
    """Sort method macros by practical shortlist quality, then rank fidelity."""
    hit = _f(row.get(f"macro_hit_at_{primary_k}"))
    regret = _f(row.get(f"macro_regret_at_{primary_k}"))
    kendall = _f(row.get("macro_kendall_tau_b"))
    spearman = _f(row.get("macro_spearman_rho"))
    return (
        -(hit if hit is not None else -1.0),
        regret if regret is not None else math.inf,
        -(kendall if kendall is not None else -1.0),
        -(spearman if spearman is not None else -1.0),
        METHOD_ORDER.index(str(row.get("method_id"))) if str(row.get("method_id")) in METHOD_ORDER else 999,
    )


__all__ = [
    "METHOD_LABELS", "METHOD_ORDER", "METHOD_UNITS",
    "RANKING_METHOD_IMPLEMENTATION", "RANKING_METHOD_SCHEMA", "RANKING_METHOD_SCHEMA_VERSION",
    "WORKFLOW_RANKING_METHOD",
    "candidate_case_id", "candidate_cut_bytes", "candidate_cut_mib", "candidate_imbalance",
    "canonical_backend", "canonical_direction", "canonical_runner", "compute_ranking_predictions",
    "cut_bytes_only_sort_key", "direction_parts", "method_macro_sort_key",
    "ranking_contexts_from_profile", "ranking_method_policy",
    "runner_direction_handover_ms",
]
