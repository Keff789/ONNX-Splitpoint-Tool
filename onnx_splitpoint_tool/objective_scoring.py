"""Shared interpretable scoring helpers for GUI and benchmark audit.

This module keeps the throughput-oriented heuristic explicit and reusable.
Compared to earlier versions it now applies a small *calibration layer* derived
from the previously benchmarked YOLOv7 Hailo↔TensorRT run.  The calibration is
still interpretable: it adjusts the raw handover heuristic with direction-
specific affine terms that depend on the raw heuristic itself plus two simple
hinge features (cut-size excess and imbalance excess).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Base heuristic weights (interpretable, direction-aware)
# ---------------------------------------------------------------------------

_TH_CUT_W = 0.12
_TH_TENSOR_W = 0.010
_TH_UNKNOWN_W = 0.040
_TH_PEAK_RIGHT_W = 0.060
_TH_COMPILE_RISK_W = 0.250
_TH_SINGLE_CTX_W = 0.500
_TH_FALLBACK_W = 0.300
_TH_DIR_HAILO_TO_OTHER = 0.750
_TH_DIR_OTHER_TO_HAILO = 0.350
_TH_DIR_OTHER_TO_OTHER = 0.150

# ---------------------------------------------------------------------------
# Embedded calibration profile derived from the previously delivered YOLOv7
# benchmark data (best full vs. streaming comparison run).
# ---------------------------------------------------------------------------

THROUGHPUT_CALIBRATION_PROFILE_NAME = "yolov7_streaming_v1"
_THROUGHPUT_CALIBRATION_ENABLED = True


@dataclass(frozen=True)
class _DirectionCalibration:
    intercept: float
    raw: float
    cut_excess: float
    imb_excess: float


_H2T_CAL = _DirectionCalibration(
    intercept=2.1475457254111454,
    raw=-0.6536494472975787,
    cut_excess=0.2434774810758271,
    imb_excess=67.27850625718,
)

_T2H_CAL = _DirectionCalibration(
    intercept=-1.130669710470198,
    raw=0.5542079516015029,
    cut_excess=0.1403234543409089,
    imb_excess=0.01804664697194682,
)


def set_throughput_calibration_enabled(enabled: bool) -> None:
    global _THROUGHPUT_CALIBRATION_ENABLED
    _THROUGHPUT_CALIBRATION_ENABLED = bool(enabled)



def throughput_calibration_enabled() -> bool:
    return bool(_THROUGHPUT_CALIBRATION_ENABLED)



def active_throughput_calibration_profile(*, enabled: Optional[bool] = None) -> Optional[str]:
    use_enabled = throughput_calibration_enabled() if enabled is None else bool(enabled)
    return THROUGHPUT_CALIBRATION_PROFILE_NAME if use_enabled else None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None



def as_int(v: Any) -> Optional[int]:
    x = as_float(v)
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None



def feature_count(v: Any) -> Optional[int]:
    if isinstance(v, (list, tuple)):
        return len(v)
    return as_int(v)



def slug_backend(text: Any) -> str:
    s = str(text or "").strip().lower()
    out = []
    last_sep = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_")



def _direction_kind(stage1: str, stage2: str) -> str:
    s1 = slug_backend(stage1)
    s2 = slug_backend(stage2)
    if s1.startswith("hailo") and not s2.startswith("hailo"):
        return "h2x"
    if not s1.startswith("hailo") and s2.startswith("hailo"):
        return "x2h"
    return "other"



def direction_bias(stage1: str, stage2: str) -> float:
    kind = _direction_kind(stage1, stage2)
    if kind == "h2x":
        return _TH_DIR_HAILO_TO_OTHER
    if kind == "x2h":
        return _TH_DIR_OTHER_TO_HAILO
    return _TH_DIR_OTHER_TO_OTHER


# ---------------------------------------------------------------------------
# Decomposed metrics
# ---------------------------------------------------------------------------


def hailo_feasibility_risk(
    *,
    compile_risk_score: Any,
    single_context_probability: Any,
    fallback_used: bool = False,
    parse_ok: Any = None,
) -> Optional[float]:
    risk = float(as_float(compile_risk_score) or 0.0)
    single_prob = as_float(single_context_probability)
    parse_gate = 0.0
    if parse_ok is False:
        parse_gate = 1.0
    out = (
        _TH_COMPILE_RISK_W * max(0.0, risk - 1.0)
        + (0.0 if single_prob is None else _TH_SINGLE_CTX_W * max(0.0, 1.0 - float(single_prob)))
        + (_TH_FALLBACK_W if fallback_used else 0.0)
        + parse_gate
    )
    return out if math.isfinite(out) else None



def hailo_interface_penalty(
    *,
    cut_mib: Any,
    n_cut_tensors: Any,
    unknown_crossing_tensors: Any,
    peak_act_right_mib: Any,
    stage1: str = "",
    stage2: str = "",
) -> Optional[float]:
    cut = float(as_float(cut_mib) or 0.0)
    n_cut = float(feature_count(n_cut_tensors) or 0.0)
    unknown = float(feature_count(unknown_crossing_tensors) or 0.0)
    peak_right = float(as_float(peak_act_right_mib) or 0.0)
    out = (
        _TH_CUT_W * cut
        + _TH_TENSOR_W * n_cut
        + _TH_UNKNOWN_W * unknown
        + _TH_PEAK_RIGHT_W * peak_right
        + direction_bias(stage1, stage2)
    )
    return out if math.isfinite(out) else None



def predicted_handover_ms_raw(
    *,
    cut_mib: Any,
    n_cut_tensors: Any,
    unknown_crossing_tensors: Any,
    peak_act_right_mib: Any,
    compile_risk_score: Any,
    single_context_probability: Any,
    fallback_used: bool = False,
    parse_ok: Any = None,
    stage1: str = "",
    stage2: str = "",
) -> Optional[float]:
    feas = hailo_feasibility_risk(
        compile_risk_score=compile_risk_score,
        single_context_probability=single_context_probability,
        fallback_used=fallback_used,
        parse_ok=parse_ok,
    )
    iface = hailo_interface_penalty(
        cut_mib=cut_mib,
        n_cut_tensors=n_cut_tensors,
        unknown_crossing_tensors=unknown_crossing_tensors,
        peak_act_right_mib=peak_act_right_mib,
        stage1=stage1,
        stage2=stage2,
    )
    if feas is None and iface is None:
        return None
    return max(0.0, float((iface or 0.0) + (feas or 0.0)))



def calibrated_handover_ms(
    *,
    raw_handover_ms: Any,
    cut_mib: Any,
    imbalance: Any,
    stage1: str,
    stage2: str,
) -> Optional[float]:
    raw = as_float(raw_handover_ms)
    if raw is None:
        return None
    cut = float(as_float(cut_mib) or 0.0)
    imb = float(as_float(imbalance) or 0.0)
    cut_excess = max(0.0, cut - 6.0)
    imb_excess = max(0.0, imb - 0.45)
    kind = _direction_kind(stage1, stage2)
    if kind == "h2x":
        cal = _H2T_CAL
        value = cal.intercept + cal.raw * raw + cal.cut_excess * cut_excess + cal.imb_excess * imb_excess
    elif kind == "x2h":
        cal = _T2H_CAL
        value = cal.intercept + cal.raw * raw + cal.cut_excess * cut_excess + cal.imb_excess * imb_excess
    else:
        value = raw
    return max(0.0, float(value)) if math.isfinite(value) else None



def predicted_handover_ms(
    *,
    cut_mib: Any,
    n_cut_tensors: Any,
    unknown_crossing_tensors: Any,
    peak_act_right_mib: Any,
    compile_risk_score: Any,
    single_context_probability: Any,
    fallback_used: bool = False,
    parse_ok: Any = None,
    stage1: str = "",
    stage2: str = "",
    imbalance: Any = None,
    use_calibration: bool = True,
) -> Optional[float]:
    raw = predicted_handover_ms_raw(
        cut_mib=cut_mib,
        n_cut_tensors=n_cut_tensors,
        unknown_crossing_tensors=unknown_crossing_tensors,
        peak_act_right_mib=peak_act_right_mib,
        compile_risk_score=compile_risk_score,
        single_context_probability=single_context_probability,
        fallback_used=fallback_used,
        parse_ok=parse_ok,
        stage1=stage1,
        stage2=stage2,
    )
    # Only apply the benchmark-derived calibration when an imbalance estimate is
    # available.  This keeps generic/out-of-domain callers monotonic and close to
    # the raw interpretable heuristic, while benchmark-analysis paths that carry
    # richer metadata benefit from the calibrated profile.
    if (not bool(use_calibration)) or (not throughput_calibration_enabled()) or as_float(imbalance) is None:
        return raw
    return calibrated_handover_ms(
        raw_handover_ms=raw,
        cut_mib=cut_mib,
        imbalance=imbalance,
        stage1=stage1,
        stage2=stage2,
    )



def predicted_stream_cycle_ms(*, bottleneck_ms: Any, handover_ms: Any) -> Optional[float]:
    b = as_float(bottleneck_ms)
    h = as_float(handover_ms)
    if b is None or h is None:
        return None
    out = float(b) + float(h)
    return out if out > 0.0 and math.isfinite(out) else None



def predicted_stream_fps(*, bottleneck_ms: Any, handover_ms: Any) -> Optional[float]:
    c = predicted_stream_cycle_ms(bottleneck_ms=bottleneck_ms, handover_ms=handover_ms)
    if c is None or c <= 0.0:
        return None
    return 1000.0 / float(c)


# ---------------------------------------------------------------------------
# Objective utilities
# ---------------------------------------------------------------------------


def objective_label(objective: str) -> str:
    objective = str(objective or "Balanced").strip().lower()
    if objective.startswith("through"):
        return "Throughput"
    if objective.startswith("hailo"):
        return "Hailo feasibility"
    if objective.startswith("lat"):
        return "Latency"
    return "Balanced"



def candidate_objective_metrics(row: dict[str, Any], *, stage1: str, stage2: str, use_calibration: bool = True) -> dict[str, Any]:
    pred_lat_total = float(row.get("pred_latency_total_ms") or 0.0) if row.get("pred_latency_total_ms") not in (None, "") else None
    bottleneck_ms = None
    if pred_lat_total is not None and pred_lat_total > 0.0:
        fl_l = float(row.get("flops_left_abs") or 0.0)
        fl_r = float(row.get("flops_right_abs") or 0.0)
        if (fl_l + fl_r) > 0.0:
            share = max(fl_l, fl_r) / (fl_l + fl_r)
            bottleneck_ms = pred_lat_total * share
        else:
            bottleneck_ms = pred_lat_total * 0.5

    compile_risk = row.get("hailo_compile_risk_score")
    single_prob = row.get("hailo_single_context_probability")
    fallback_used = bool(row.get("hailo_part2_fallback_used"))
    parse_ok = row.get("hailo_parse_ok")
    imbalance = row.get("imbalance_val") or row.get("pred_imbalance") or row.get("imbalance_pred") or row.get("imbalance")

    feas = hailo_feasibility_risk(
        compile_risk_score=compile_risk,
        single_context_probability=single_prob,
        fallback_used=fallback_used,
        parse_ok=parse_ok,
    )
    iface = hailo_interface_penalty(
        cut_mib=row.get("cut_mb_val"),
        n_cut_tensors=row.get("n_cut_tensors"),
        unknown_crossing_tensors=row.get("unknown_count"),
        peak_act_right_mib=row.get("peak_right_mib_val"),
        stage1=stage1,
        stage2=stage2,
    )
    handover_raw = predicted_handover_ms_raw(
        cut_mib=row.get("cut_mb_val"),
        n_cut_tensors=row.get("n_cut_tensors"),
        unknown_crossing_tensors=row.get("unknown_count"),
        peak_act_right_mib=row.get("peak_right_mib_val"),
        compile_risk_score=compile_risk,
        single_context_probability=single_prob,
        fallback_used=fallback_used,
        parse_ok=parse_ok,
        stage1=stage1,
        stage2=stage2,
    )
    handover_uncal = predicted_handover_ms(
        cut_mib=row.get("cut_mb_val"),
        n_cut_tensors=row.get("n_cut_tensors"),
        unknown_crossing_tensors=row.get("unknown_count"),
        peak_act_right_mib=row.get("peak_right_mib_val"),
        compile_risk_score=compile_risk,
        single_context_probability=single_prob,
        fallback_used=fallback_used,
        parse_ok=parse_ok,
        stage1=stage1,
        stage2=stage2,
        imbalance=imbalance,
        use_calibration=False,
    )
    handover_cal = predicted_handover_ms(
        cut_mib=row.get("cut_mb_val"),
        n_cut_tensors=row.get("n_cut_tensors"),
        unknown_crossing_tensors=row.get("unknown_count"),
        peak_act_right_mib=row.get("peak_right_mib_val"),
        compile_risk_score=compile_risk,
        single_context_probability=single_prob,
        fallback_used=fallback_used,
        parse_ok=parse_ok,
        stage1=stage1,
        stage2=stage2,
        imbalance=imbalance,
        use_calibration=True,
    )
    handover = handover_cal if bool(use_calibration) else handover_uncal
    cycle_uncal = predicted_stream_cycle_ms(bottleneck_ms=bottleneck_ms, handover_ms=handover_uncal)
    pred_fps_uncal = predicted_stream_fps(bottleneck_ms=bottleneck_ms, handover_ms=handover_uncal)
    cycle_cal = predicted_stream_cycle_ms(bottleneck_ms=bottleneck_ms, handover_ms=handover_cal)
    pred_fps_cal = predicted_stream_fps(bottleneck_ms=bottleneck_ms, handover_ms=handover_cal)
    cycle = cycle_cal if bool(use_calibration) else cycle_uncal
    pred_fps = pred_fps_cal if bool(use_calibration) else pred_fps_uncal
    return {
        "hailo_feasibility_risk": feas,
        "hailo_interface_penalty": iface,
        "predicted_handover_ms_raw": handover_raw,
        "predicted_handover_ms_uncalibrated": handover_uncal,
        "predicted_handover_ms_calibrated": handover_cal,
        "predicted_handover_ms": handover,
        "predicted_stream_cycle_ms_uncalibrated": cycle_uncal,
        "predicted_stream_cycle_ms_calibrated": cycle_cal,
        "predicted_stream_cycle_ms": cycle,
        "predicted_stream_fps_uncalibrated": pred_fps_uncal,
        "predicted_stream_fps_calibrated": pred_fps_cal,
        "predicted_stream_fps": pred_fps,
        "predicted_bottleneck_ms": bottleneck_ms,
        "objective_balanced_score": row.get("score_pred"),
        "objective_latency_ms": pred_lat_total,
        "throughput_calibration_profile": (THROUGHPUT_CALIBRATION_PROFILE_NAME if bool(use_calibration) and as_float(imbalance) is not None else None),
        "throughput_calibration_enabled": bool(use_calibration),
    }



def candidate_objective_summary(row: dict[str, Any], *, objective: str, stage1: str, stage2: str, use_calibration: bool = True) -> dict[str, str]:
    metrics = candidate_objective_metrics(row, stage1=stage1, stage2=stage2, use_calibration=use_calibration)
    label = objective_label(objective)
    boundary = row.get("boundary")
    title = f"Objective: {label}"
    headline = f"Top candidate b{boundary}" if boundary is not None else "Top candidate"
    if label == "Throughput":
        fps = metrics.get("predicted_stream_fps")
        handover = metrics.get("predicted_handover_ms")
        detail = f"pred TH {float(fps):.2f} FPS" if fps is not None else "pred TH –"
        if handover is not None:
            detail += f" · HO {float(handover):.2f} ms"
        if row.get("cut_mb_val") is not None:
            detail += f" · cut {float(row.get('cut_mb_val')):.2f} MiB"
        detail += (f" · cal {THROUGHPUT_CALIBRATION_PROFILE_NAME}" if bool(use_calibration) else " · uncal")
        return {"title": title, "headline": headline, "detail": detail}
    if label == "Hailo feasibility":
        feas = metrics.get("hailo_feasibility_risk")
        iface = metrics.get("hailo_interface_penalty")
        single = row.get("hailo_single_context_probability")
        detail = f"risk {float(feas):.2f}" if feas is not None else "risk –"
        if iface is not None:
            detail += f" · iface {float(iface):.2f}"
        if single is not None:
            detail += f" · 1ctx {100.0 * float(single):.0f}%"
        return {"title": title, "headline": headline, "detail": detail}
    if label == "Latency":
        lat = metrics.get("objective_latency_ms")
        detail = f"pred latency {float(lat):.2f} ms" if lat is not None else "pred latency –"
        score = row.get("score_pred")
        if score is not None:
            detail += f" · score {float(score):.3f}"
        return {"title": title, "headline": headline, "detail": detail}
    score = row.get("score_pred")
    fps = metrics.get("predicted_stream_fps")
    detail = f"score {float(score):.3f}" if score is not None else "score –"
    if fps is not None:
        detail += f" · pred TH {float(fps):.2f} FPS"
    return {"title": title, "headline": headline, "detail": detail}



def objective_sort_key(row: dict[str, Any], *, objective: str, stage1: str, stage2: str, use_calibration: bool = True) -> tuple[Any, ...]:
    objective = str(objective or "Balanced").strip().lower()
    metrics = candidate_objective_metrics(row, stage1=stage1, stage2=stage2, use_calibration=use_calibration)
    if objective.startswith("through"):
        fps = metrics.get("predicted_stream_fps")
        return (0 if fps is not None else 1, -(float(fps) if fps is not None else 0.0), float(row.get("cut_mb_val", 0.0)), int(row.get("boundary", 10**9)))
    if objective.startswith("hailo"):
        feas = metrics.get("hailo_feasibility_risk")
        return (0 if feas is not None else 1, float(feas) if feas is not None else float("inf"), float(row.get("cut_mb_val", 0.0)), int(row.get("boundary", 10**9)))
    if objective.startswith("lat"):
        lat = metrics.get("objective_latency_ms")
        return (0 if lat is not None else 1, float(lat) if lat is not None else float("inf"), float(row.get("cut_mb_val", 0.0)), int(row.get("boundary", 10**9)))
    score = row.get("score_pred")
    source_rank = row.get("source_rank") or row.get("rank")
    return (0 if score is not None else 1, float(score) if score is not None else float("inf"), int(source_rank or 10**9), int(row.get("boundary", 10**9)))
