from __future__ import annotations

"""Result dashboard, thesis metrics and lightweight figures for EvaluationRuns.

v50 turns the formal workflow from a pure artifact bundle into a thesis-ready
results bundle.  This module intentionally reads the existing normalized/result
CSV/JSON contracts instead of parsing logs.  It keeps the dashboard honest:
component-only rows are visible, but only complete split rows are used for
Top-k/regret/speedup metrics.
"""

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .artifacts import now_iso, read_json, relpath, write_csv, write_json, write_text
from .evidence_status import project_native_evidence_status
from ..energy.comparison import resolve_energy_comparison


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", newline="", encoding="utf-8") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    except Exception:
        return []


def _float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        return None
    return None


def _bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "ok", "success", "pass", "passed"}:
        return True
    if s in {"0", "false", "no", "n", "fail", "failed", "error"}:
        return False
    return None


def _mean(values: Iterable[float]) -> Optional[float]:
    xs = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _median(values: Iterable[float]) -> Optional[float]:
    xs = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def _fmt(value: Any, ndigits: int = 6) -> Any:
    x = _float(value)
    if x is None:
        return ""
    return round(x, ndigits)


def _safe_min(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals = [_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _complete_split_latency(row: Mapping[str, Any]) -> Optional[float]:
    if str(row.get("variant") or "").lower() != "split":
        return None
    p1 = _float(row.get("part1_latency_ms"))
    p2 = _float(row.get("part2_latency_ms"))
    tr = _float(row.get("transfer_latency_ms"))
    total = _float(row.get("total_latency_ms"))
    status = str(row.get("component_measurement_status") or "").lower()
    measured = {str(x).lower() for x in list(row.get("measured_variants") or [])} if isinstance(row.get("measured_variants"), list) else set()
    skipped = {str(x).lower() for x in list(row.get("skipped_variants") or [])} if isinstance(row.get("skipped_variants"), list) else set()
    if total is not None:
        if status in {"composed", "split_parts"}:
            return total
        if "composed" in measured and "composed" not in skipped:
            return total
        if p1 is not None and p2 is not None:
            return total
        full = _float(row.get("full_latency_ms"))
        if full is None or abs(full - total) > 1e-9:
            return total
    if p1 is not None and p2 is not None:
        return float(p1) + float(p2) + float(tr or 0.0)
    return None




def _pipeline_cycle(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("pipeline_cycle_selected_ms", "pipeline_cycle_measured_ms", "pipeline_cycle_with_transfer_ms", "pipeline_cycle_no_transfer_ms", "throughput_cycle_est_ms"):
        v = _float(row.get(key))
        if v is not None:
            return v
    # Conservative fallback: no streaming/interleaving evidence means throughput
    # cycle cannot be better than the single-frame split latency.
    return _complete_split_latency(row)


def _pipeline_fps(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("pipeline_fps_selected", "pipeline_fps_measured", "throughput_fps_makespan", "pipeline_fps_with_transfer"):
        v = _float(row.get(key))
        if v is not None:
            return v
    c = _pipeline_cycle(row)
    if c not in (None, 0):
        return 1000.0 / float(c)
    return None


def _best_by_latency(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = [r for r in rows if _complete_split_latency(r) is not None]
    return min(valid, key=lambda r: float(_complete_split_latency(r) or 1e18), default={})


def _best_by_pipeline_cycle(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    valid = [r for r in rows if _pipeline_cycle(r) is not None]
    return min(valid, key=lambda r: float(_pipeline_cycle(r) or 1e18), default={})


def _throughput_speedup(full_latency_ms: Optional[float], cycle_ms: Optional[float]) -> Optional[float]:
    if full_latency_ms in (None, 0) or cycle_ms in (None, 0):
        return None
    try:
        return float(full_latency_ms) / float(cycle_ms)
    except Exception:
        return None


def _full_latency(row: Mapping[str, Any]) -> Optional[float]:
    # v51: only real normalized full rows participate in full-baseline
    # comparisons. Legacy split rows may still carry a full-reference timing,
    # but those companion facts are now materialized as separate variant=full
    # rows by workflow.results. This keeps mixed backends such as
    # hailo8_to_tensorrt out of best_full_backend.
    if str(row.get("variant") or "").lower() != "full":
        return None
    return _float(row.get("total_latency_ms")) or _float(row.get("full_latency_ms"))



def _full_baseline_backend(row: Mapping[str, Any]) -> str:
    if str(row.get("variant") or "").lower() != "full":
        return ""
    backend = str(row.get("full_backend") or row.get("full_provider") or row.get("backend") or "").strip().lower()
    backend = backend.replace("-", "_").replace(" ", "_")
    if "_to_" in backend or backend in {"", "auto", "unknown"}:
        return ""
    if backend in {"cpu", "ort_cpu"}:
        return "cpu_ort"
    if backend in {"cuda", "ort_cuda", "gpu"}:
        return "cuda_ort"
    if backend in {"trt", "tensor_rt"}:
        return "tensorrt"
    return backend



def _row_is_hailo_full(row: Mapping[str, Any]) -> bool:
    if str(row.get("variant") or "").lower() != "full":
        return False
    backend = str(row.get("backend") or row.get("full_backend") or row.get("full_provider") or "").lower()
    fullp = str(row.get("full_provider") or row.get("full_backend") or "").lower()
    return backend.startswith("hailo") or fullp.startswith("hailo")


def _row_is_hailo_complete_split(row: Mapping[str, Any]) -> bool:
    if str(row.get("variant") or "").lower() != "split":
        return False
    backend = str(row.get("backend") or "").lower()
    if "hailo" not in backend:
        return False
    status = str(row.get("component_measurement_status") or "").lower()
    return _complete_split_latency(row) is not None and status in {"composed", "split_parts", "full_raw_head_e2e"}


def _row_final_pass(row: Mapping[str, Any]) -> Optional[bool]:
    # v59ej: if the strict ranking gate is present, it owns final ranking pass.
    eligible = _bool(row.get("eligible_for_ranking"))
    if eligible is not None:
        return eligible
    # Legacy fallback for older reports without explicit gates.
    for key in ("final_pass_all", "final_pass", "validation_ok", "semantic_validation_ok", "semantic_validation_passed"):
        value = _bool(row.get(key))
        if value is not None:
            return value
    return None


def _row_eligible_for_ranking(row: Mapping[str, Any]) -> Optional[bool]:
    # v59ej: ranking must be gated by declared task accuracy, not by speed or
    # native contract-only self-reference.  New rows expose this directly.
    explicit = _bool(row.get("eligible_for_ranking"))
    if explicit is not None:
        return explicit
    # Legacy fallback: old reports did not have the separate gate fields.
    return _row_final_pass(row)


def _row_runtime_contract_only(row: Mapping[str, Any]) -> bool:
    for key in ("runtime_contract_only", "contract_only", "counts_as_split_benchmark"):
        value = _bool(row.get(key))
        if key == "counts_as_split_benchmark" and value is False:
            return True
        if key != "counts_as_split_benchmark" and value is True:
            return True
    status = str(row.get("stage2_accel_calibration_status") or row.get("deepx_stage2_contract_status") or "").lower()
    return status in {"requires_feature_tensor_calibration", "all_candidates_rejected", "contract_rejected", "native_unstable", "deepx_stage2_native_unstable", "unsupported"}



def _row_is_heterogeneous_accelerator_split(row: Mapping[str, Any]) -> bool:
    """Return True for cross-backend split pipelines.

    The benchmark rows are still being normalized across legacy and remote
    harnesses, so not every row carries explicit stage1/stage2 fields.  Use the
    backend/run label as the stable fallback and exclude same-backend pseudo
    splits such as plain TensorRT/CUDA/CPU split rows.
    """
    backend = str(row.get("backend") or row.get("run_id") or row.get("backend_label") or "").lower().replace("-", "_").replace(" ", "_")
    if not backend:
        return False
    if "_to_" in backend or "→" in backend or "->" in backend:
        return True
    stage1 = str(row.get("stage1_backend") or row.get("stage1") or "").strip().lower()
    stage2 = str(row.get("stage2_backend") or row.get("stage2") or "").strip().lower()
    return bool(stage1 and stage2 and stage1 != stage2)

def _throughput_kind(row: Mapping[str, Any]) -> str:
    value = str(row.get("throughput_kind") or "").strip()
    if value:
        return value
    variant = str(row.get("variant") or "").lower()
    if variant == "full":
        return "full_backend"
    if variant == "split" and _row_is_heterogeneous_accelerator_split(row):
        return "heterogeneous_pipeline"
    if variant == "split":
        return "same_backend_split_diagnostic"
    return "component_or_diagnostic"


def _pipeline_applicable(row: Mapping[str, Any]) -> bool:
    return _throughput_kind(row) == "heterogeneous_pipeline"


def _full_backend_fps(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("full_backend_throughput_fps", "throughput_primary_fps"):
        v = _float(row.get(key))
        if v is not None:
            return v
    lat = _full_latency(row)
    if lat not in (None, 0):
        return 1000.0 / float(lat)
    return None


def _heterogeneous_pipeline_fps(row: Mapping[str, Any]) -> Optional[float]:
    if _throughput_kind(row) != "heterogeneous_pipeline":
        return None
    return _float(row.get("heterogeneous_pipeline_fps")) or _pipeline_fps(row)


def _same_backend_diagnostic_fps(row: Mapping[str, Any]) -> Optional[float]:
    if _throughput_kind(row) != "same_backend_split_diagnostic":
        return None
    return _float(row.get("same_backend_composed_fps")) or _pipeline_fps(row)


def _row_energy_per_frame(row: Mapping[str, Any]) -> Optional[float]:
    # Historical alias kept for compatibility. Prefer the explicit helpers
    # below in new report tables so command-window and selected-FPS energy are
    # not confused.
    return _row_energy_command_window_j_per_frame(row) or _row_energy_selected_fps_j_per_frame(row)


def _row_energy_command_window_j_per_frame(row: Mapping[str, Any]) -> Optional[float]:
    resolved = resolve_energy_comparison(row)
    if str(row.get("host_normalization_role") or "none").strip().lower() == "tensorrt_full":
        return _float(resolved.get("comparison_energy_per_work_j"))
    for key in (
        "row_energy_streaming_j_per_frame",
        "energy_streaming_j_per_frame",
        "energy_per_pipeline_frame_j",
        "row_energy_latency_j_per_inference",
        "energy_per_inference_j",
    ):
        v = _float(row.get(key))
        if v is not None:
            return v
    return None


def _row_selected_reference_fps(row: Mapping[str, Any]) -> Optional[float]:
    kind = str(row.get("throughput_kind") or "").strip().lower()
    variant = str(row.get("variant") or row.get("energy_target_variant") or "").strip().lower()
    if kind == "full_backend" or variant == "full":
        for key in ("full_backend_throughput_fps", "throughput_primary_fps", "pipeline_fps_selected"):
            v = _float(row.get(key))
            if v is not None and v > 0:
                return v
    if kind == "heterogeneous_pipeline" or variant in {"split", "composed"}:
        for key in ("heterogeneous_pipeline_fps", "pipeline_fps_selected", "throughput_primary_fps"):
            v = _float(row.get(key))
            if v is not None and v > 0:
                return v
    for key in ("throughput_primary_fps", "pipeline_fps_selected", "full_backend_throughput_fps"):
        v = _float(row.get(key))
        if v is not None and v > 0:
            return v
    return None


def _row_energy_selected_fps_j_per_frame(row: Mapping[str, Any]) -> Optional[float]:
    # Prefer recomputing from the current row's own selected FPS and measured
    # streaming power. Stored *_from_selected_fps fields may originate from a
    # canonical target merge and can drift after later normalisation.
    resolved = resolve_energy_comparison(row)
    pwr = _float(resolved.get("comparison_average_power_w"))
    fps = _row_selected_reference_fps(row)
    if pwr is not None and fps is not None and fps > 0:
        return pwr / fps
    for key in (
        "energy_streaming_j_per_frame_from_selected_fps",
        "energy_j_per_frame_from_selected_fps",
    ):
        v = _float(row.get(key))
        if v is not None:
            return v
    return None


def _row_energy_frames_per_j(row: Mapping[str, Any]) -> Optional[float]:
    resolved = resolve_energy_comparison(row)
    pwr = _float(resolved.get("comparison_average_power_w"))
    fps = _row_selected_reference_fps(row)
    if pwr is not None and pwr > 0 and fps is not None and fps > 0:
        return fps / pwr
    for key in (
        "energy_streaming_fps_per_watt_from_selected_fps",
        "energy_work_units_per_j",
        "energy_streaming_frames_per_j",
    ):
        v = _float(row.get(key))
        if v is not None:
            return v
    return None



def _energy_status_token_v59k(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _energy_finite_positive_v59k(value: Any) -> Optional[float]:
    v = _float(value)
    if v is None:
        return None
    try:
        if v > 0:
            return float(v)
    except Exception:
        return None
    return None


def _row_has_usable_energy_metric_v59k(row: Mapping[str, Any]) -> bool:
    for key in (
        "row_energy_streaming_j_per_frame",
        "row_energy_latency_j_per_inference",
        "row_energy_j_per_frame",
        "row_energy_j_per_inference",
        "energy_streaming_j_per_frame",
        "energy_per_inference_j",
        "energy_streaming_avg_power_w",
        "energy_latency_avg_power_w",
        "energy_work_units_per_j",
        "energy_total_j",
        "avg_power_w",
    ):
        if _energy_finite_positive_v59k(row.get(key)) is not None:
            return True
    return False


def _row_energy_target_ok_v59k(row: Mapping[str, Any]) -> bool:
    status = _energy_status_token_v59k(row.get("energy_target_status"))
    if status in {"", "unknown"}:
        return _row_has_usable_energy_metric_v59k(row)
    if status not in {"ok", "success", "complete"}:
        return False
    return _row_has_usable_energy_metric_v59k(row)


def _energy_phase_ok_v59k(phase: Mapping[str, Any]) -> bool:
    if not isinstance(phase, Mapping) or not phase:
        return False
    if phase.get("ok") is False:
        return False
    status = _energy_status_token_v59k(phase.get("status"))
    if status in {"failed", "fail", "error", "partial", "skipped", "skip", "duration_probe_failed", "no_results"}:
        return False
    if phase.get("ok") is True or status in {"ok", "success", "complete"}:
        return True
    return any(_energy_finite_positive_v59k(phase.get(k)) is not None for k in (
        "avg_energy_total_j", "sum_energy_total_j", "energy_total_j",
        "avg_power_w", "weighted_avg_power_w", "avg_energy_per_inference_j",
        "avg_energy_per_work_unit_j", "avg_energy_per_pipeline_frame_j",
    ))


def _energy_aggregate_ok_v59k(agg: Mapping[str, Any]) -> bool:
    if not isinstance(agg, Mapping) or not agg:
        return False
    if agg.get("ok") is False:
        return False
    status = _energy_status_token_v59k(agg.get("status"))
    if status in {"failed", "fail", "error", "partial", "skipped", "skip", "duration_probe_failed", "no_results"}:
        return False
    phases = [p for p in (agg.get("phases") if isinstance(agg.get("phases"), list) else []) if isinstance(p, Mapping)]
    if phases:
        return any(_energy_phase_ok_v59k(p) for p in phases)
    return agg.get("ok") is True or status in {"ok", "success", "complete"}


def _energy_aggregate_has_metric_v59k(agg: Mapping[str, Any]) -> bool:
    if not isinstance(agg, Mapping) or not agg:
        return False
    for key in ("sum_energy_total_j", "energy_total_j", "target_energy_total_j", "avg_power_w", "avg_power_w_weighted"):
        if _energy_finite_positive_v59k(agg.get(key)) is not None:
            return True
    phases = [p for p in (agg.get("phases") if isinstance(agg.get("phases"), list) else []) if isinstance(p, Mapping)]
    for phase in phases:
        if not _energy_phase_ok_v59k(phase):
            continue
        for key in (
            "avg_energy_total_j", "sum_energy_total_j", "energy_total_j",
            "avg_power_w", "weighted_avg_power_w", "avg_energy_per_inference_j",
            "avg_energy_per_work_unit_j", "avg_energy_per_pipeline_frame_j",
        ):
            if _energy_finite_positive_v59k(phase.get(key)) is not None:
                return True
    return False


def _energy_aggregate_is_usable_v59k(agg: Mapping[str, Any]) -> bool:
    return _energy_aggregate_ok_v59k(agg) and _energy_aggregate_has_metric_v59k(agg)

def _row_has_urecs_energy(row: Mapping[str, Any]) -> bool:
    """True only if a normalized row has usable row-level u.RECS energy.

    v59k: target provenance alone does not count.  Partial/failed target
    aggregates are surfaced in reports but excluded from energy-complete counts.
    """
    source = str(row.get("energy_row_level_source") or "").strip().lower()
    coverage = str(row.get("energy_coverage_status") or "").strip().lower()
    if source in {"dispatch_only", "dispatch_only_no_row_energy", "target_aggregate_unusable"}:
        return False
    if coverage in {"dispatch_only_no_row_energy", "target_aggregate_partial", "target_aggregate_failed", "missing_row_level_energy", "missing_or_not_measured"}:
        return False

    explicit_row_metric = any(
        _energy_finite_positive_v59k(row.get(k)) is not None
        for k in (
            "row_energy_streaming_j_per_frame",
            "row_energy_latency_j_per_inference",
            "row_energy_j_per_frame",
            "row_energy_j_per_inference",
        )
    )
    if explicit_row_metric:
        return True

    target_hint = False
    for key in ("energy_target_id", "energy_target_case", "target_id", "target_case"):
        if row.get(key) not in (None, ""):
            target_hint = True
            break
    if not target_hint:
        for key in ("energy_aggregate_relpath", "energy_aggregate_path"):
            val = str(row.get(key) or "").replace("\\", "/")
            if val and "/targets/" in val:
                target_hint = True
                break
    if target_hint:
        return _row_energy_target_ok_v59k(row)

    if _bool(row.get("energy_row_scope")) is True:
        return _row_has_usable_energy_metric_v59k(row)
    return False

def _portable_energy_relpath(row: Mapping[str, Any], run_dir: Path) -> str:
    """Return a portable run-relative energy aggregate path if possible."""
    rel = str(row.get("energy_aggregate_relpath") or "").strip()
    path_s = str(row.get("energy_aggregate_path") or "").strip()
    if rel and rel not in {"energy_aggregate.json", "energy_summary.json"} and "/" in rel.replace("\\", "/"):
        return rel.replace("\\", "/")
    if path_s:
        p = Path(path_s)
        try:
            if p.is_absolute() and p.exists():
                return relpath(p, run_dir)
        except Exception:
            pass
        s = path_s.replace("\\", "/")
        idx = s.rfind("/models/")
        if idx >= 0:
            return s[idx + 1:]
        idx = s.rfind("/reports/")
        if idx >= 0:
            return s[idx + 1:]
    return rel or (Path(path_s).name if path_s else "")

def _copy_energy_context(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    """Copy measured energy fields without touching benchmark-owned timing/FPS fields."""
    keys = (
        "avg_power_w", "energy_measurement_scope", "energy_streaming_avg_power_w",
        "row_energy_streaming_j_per_frame", "energy_streaming_j_per_frame",
        "row_energy_latency_j_per_inference", "energy_per_inference_j",
        "energy_work_units_per_j", "energy_streaming_frames_per_j",
        "energy_streaming_j_per_frame_from_selected_fps",
        "energy_streaming_fps_per_watt_from_selected_fps",
        "energy_j_per_frame",
        "energy_enabled", "energy_source", "energy_row_scope", "energy_row_level_source",
        "energy_coverage_status", "energy_target_status", "energy_merge_source",
        "energy_target_case", "energy_target_variant", "energy_target_id", "energy_applies_to_all_cases",
        "energy_target_phase_count", "energy_target_window_count", "energy_target_valid_window_count", "energy_target_valid_window_ratio",
        "energy_aggregate_relpath", "energy_summary_relpath",
        "energy_aggregate_path", "energy_summary_path",
        "host_normalized_energy_est_j",
        "row_host_normalized_energy_latency_j_per_inference_est",
        "row_host_normalized_energy_streaming_j_per_frame_est",
        "host_normalized_streaming_avg_power_est_w",
        "host_normalized_average_power_est_w",
        "host_normalization_role", "host_normalization_source_run_id",
        "host_normalization_target_variant", "host_normalization_identity_verified",
        "accelerator_idle_correction_requested", "accelerator_idle_correction_applied",
        "accelerator_idle_correction_status", "accelerator_idle_correction_statuses",
        "accelerator_idle_w_applied", "accelerator_idle_calibration_verified",
        "accelerator_idle_calibration_status", "accelerator_idle_calibration_binding_path",
        "accelerator_idle_calibration_binding_sha256", "accelerator_idle_calibration_evidence",
        "accelerator_idle_calibrated_at", "energy_comparison_basis",
        "energy_comparison_status", "energy_comparison_claim_ready",
        "energy_efficiency_claim_eligible",
    )
    for k in keys:
        v = src.get(k)
        if v not in (None, ""):
            dst[k] = v
    for k, v in resolve_energy_comparison(src).items():
        if v not in (None, ""):
            dst[k] = v


def _energy_relpath_for_report(row: Mapping[str, Any], run_dir: Path, key: str = "energy_aggregate") -> str:
    """Return a portable run-relative path for energy artifacts when possible."""
    rel_key = f"{key}_relpath"
    path_key = f"{key}_path"
    raw_abs = str(row.get(path_key) or row.get(key) or "").strip()
    raw_rel = str(row.get(rel_key) or "").strip()
    if raw_abs:
        try:
            p = Path(raw_abs).expanduser()
            if p.is_absolute():
                return p.relative_to(run_dir).as_posix()
        except Exception:
            pass
    if raw_rel and raw_rel not in {"energy_aggregate.json", "energy_summary.json"}:
        return raw_rel
    return raw_rel

def _energy_phase_value(phase: Mapping[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        v = _float(phase.get(key))
        if v is not None:
            return v
    return None


def _energy_phase_summary(agg: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarise dispatch/target energy aggregates for report tables.

    Energy aggregate files evolved over several versions: target-level files carry
    a `phases` list, while dispatch-level files carry `target_results`.  This
    helper computes totals/window counts/weighted power from either shape so the
    CSVs do not show NaN for valid energy artefacts.
    """
    phases = agg.get("phases") if isinstance(agg.get("phases"), list) else []
    if not phases and isinstance(agg.get("target_results"), list):
        # Dispatch aggregate: keep direct target_result summaries when available.
        trs = [x for x in agg.get("target_results") or [] if isinstance(x, Mapping)]
        total = sum(float(_float(x.get("sum_energy_total_j")) or 0.0) for x in trs)
        powers = [_float(x.get("avg_power_w") or x.get("avg_power_w_weighted") or x.get("target_avg_power_w")) for x in trs]
        powers = [p for p in powers if p is not None]
        return {
            "total_energy_j": total if total else _float(agg.get("sum_energy_total_j") or agg.get("dispatch_energy_total_j")),
            "avg_power_w": _mean(powers),
            "window_count": agg.get("energy_window_count") or agg.get("dispatch_energy_window_count") or len(trs) or "",
            "valid_window_count": agg.get("valid_energy_window_count") or agg.get("dispatch_valid_energy_window_count") or len([x for x in trs if _bool(x.get("ok")) is not False]) or "",
        }
    total = 0.0
    valid_windows = 0
    windows = 0
    weighted_power_num = 0.0
    weighted_power_den = 0.0
    any_total = False
    failed_phase_count = 0
    for ph in phases:
        if not isinstance(ph, Mapping):
            continue
        if not _energy_phase_ok_v59k(ph):
            failed_phase_count += 1
            continue
        e = _energy_phase_value(ph, "sum_energy_total_j", "avg_energy_total_j", "energy_total_j", "target_energy_total_j")
        pwr = _energy_phase_value(ph, "avg_power_w_weighted", "avg_power_w", "target_avg_power_w")
        dur = _energy_phase_value(ph, "avg_active_duration_s", "avg_window_duration_s", "avg_workload_duration_s", "avg_collector_measurement_duration_s")
        wc = _float(ph.get("energy_window_count") or ph.get("run_count") or ph.get("valid_postprocessed_runs"))
        vwc = _float(ph.get("valid_energy_window_count") or ph.get("valid_postprocessed_runs") or ph.get("valid_runs") or wc)
        if e is not None:
            total += float(e); any_total = True
        if wc is not None:
            windows += int(wc)
        elif e is not None or pwr is not None:
            windows += 1
        if vwc is not None:
            valid_windows += int(vwc)
        elif e is not None or pwr is not None:
            valid_windows += 1
        if pwr is not None:
            weight = float(dur or vwc or wc or 1.0)
            weighted_power_num += float(pwr) * weight
            weighted_power_den += weight
    usable = _energy_aggregate_is_usable_v59k(agg)
    return {
        "total_energy_j": (total if any_total else (_float(agg.get("sum_energy_total_j") or agg.get("dispatch_energy_total_j") or agg.get("total_energy_j")) if usable else None)),
        "avg_power_w": (weighted_power_num / weighted_power_den if weighted_power_den > 0 else (_float(agg.get("avg_power_w_weighted") or agg.get("avg_power_w")) if usable else None)),
        "window_count": windows if usable else 0,
        "valid_window_count": valid_windows if usable else 0,
        "usable": bool(usable),
        "failed_phase_count": failed_phase_count,
        "target_status": str(agg.get("status") or ("ok" if agg.get("ok") is True else "unknown")),
    }


def _row_valid_complete_split(row: Mapping[str, Any]) -> bool:
    """Strict predicate for thesis/paper best-split selection."""
    if str(row.get("variant") or "").lower() != "split":
        return False
    if _complete_split_latency(row) is None:
        return False
    if _stage2_contract_rejected(row) or _row_runtime_contract_only(row):
        return False
    if _bool(row.get("runtime_ok")) is False:
        return False
    if _bool(row.get("semantic_validation_ok")) is False and _bool(row.get("validation_ok")) is not True:
        return False
    return _row_final_pass(row) is True


def _deepx_stage2_contract_status(row: Mapping[str, Any]) -> str:
    for key in ("deepx_stage2_contract_status", "stage2_contract_status"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _stage2_contract_rejected(row: Mapping[str, Any]) -> bool:
    status = _deepx_stage2_contract_status(row).lower()
    if status in {"all_candidates_rejected", "no_candidates", "rejected", "contract_rejected", "native_unstable", "deepx_stage2_native_unstable", "failed"}:
        return True
    err = str(row.get("error_class") or row.get("skip_reason") or row.get("stage2_accel_calibration_status") or "").lower()
    return "contract" in err and ("reject" in err or "failed" in err)


def _row_quality_label(row: Mapping[str, Any]) -> str:
    if _stage2_contract_rejected(row):
        return "contract_rejected"
    elig = _row_eligible_for_ranking(row)
    if elig is True:
        return "valid"
    if _bool(row.get("contract_consistent")) is True and elig is not True:
        return "contract_only_not_rank_eligible"
    fp = _row_final_pass(row)
    if fp is False or elig is False:
        return "invalid"
    status = str(row.get("component_measurement_status") or "").lower()
    return status or "unknown"

def _load_normalized_rows(run_dir: Path, model_id: str) -> List[Dict[str, Any]]:
    payload = read_json(run_dir / "models" / model_id / "benchmark_results" / "normalized_results.json", default={}) or {}
    if not isinstance(payload, Mapping):
        return []
    return [dict(x or {}) for x in list(payload.get("results") or []) if isinstance(x, Mapping)]


def _load_remote_preflight(run_dir: Path, model_id: str) -> Dict[str, Any]:
    mdir = run_dir / "models" / model_id / "benchmark_results" / "remote_diagnostics"
    for candidate in (mdir / "preflight.json", mdir / "lean_bundle" / "preflight.json"):
        payload = read_json(candidate, default={}) or {}
        if isinstance(payload, Mapping) and payload:
            return dict(payload)
    return {}


def _backend_display(backend: str) -> str:
    mapping = {
        "cpu_ort": "CPU ORT",
        "cuda_ort": "CUDA ORT",
        "tensorrt": "TensorRT",
        "hailo8": "Hailo-8",
        "hailo8_to_tensorrt": "Hailo-8 → TensorRT",
        "tensorrt_to_hailo8": "TensorRT → Hailo-8",
        "deepx_m1": "DeepX DX-M1",
        "deepx_m1_to_tensorrt": "DeepX DX-M1 → TensorRT",
        "tensorrt_to_deepx_m1": "TensorRT → DeepX DX-M1",
    }
    return mapping.get(str(backend or ""), str(backend or ""))


def _full_only_quality_model_health_ok_v27538(
    *,
    benchmark_contract: Mapping[str, Any] | None,
    benchmark_stage: Mapping[str, Any] | None,
    validation_contract: Mapping[str, Any] | None,
    hardware_contract: Mapping[str, Any] | None,
) -> bool:
    """Accept the sealed Quality-only N/A state without a performance claim.

    A Full-only Quality Canary deliberately has zero normalized performance
    rows.  Consequently ``validation_ok`` is N/A and ``hardware_verified`` is
    false even when both accelerator Quality endpoints completed exactly.  A
    dashboard must not turn that truthful N/A state into a technical failure,
    but row-local flags are insufficient authority.  Require the completed
    benchmark stage and all archived benchmark/validation/hardware contracts
    to agree; every incomplete or contradictory shape remains fail-closed.
    """

    benchmark = dict(benchmark_contract or {})
    stage = dict(benchmark_stage or {})
    validation = dict(validation_contract or {})
    hardware = dict(hardware_contract or {})

    def _exact_count_pair(
        payload: Mapping[str, Any],
        *,
        expected: int | None = None,
        required: bool = False,
    ) -> tuple[bool, int | None]:
        has_count = "quality_evidence_count" in payload
        has_expected = "expected_full_quality_count" in payload
        if not has_count and not has_expected:
            return (not required), expected
        if not has_count or not has_expected:
            return False, expected
        try:
            observed_value = int(payload.get("quality_evidence_count"))
            expected_value = int(payload.get("expected_full_quality_count"))
        except (TypeError, ValueError):
            return False, expected
        if expected_value <= 0 or observed_value != expected_value:
            return False, expected
        if expected is not None and expected_value != expected:
            return False, expected
        return True, expected_value

    benchmark_counts_ok, expected_count = _exact_count_pair(
        benchmark, required=True,
    )
    validation_counts_ok, _ = _exact_count_pair(
        validation, expected=expected_count,
    )
    hardware_counts_ok, _ = _exact_count_pair(
        hardware, expected=expected_count,
    )

    def _zero_count(payload: Mapping[str, Any], key: str) -> bool:
        try:
            return int(payload.get(key) or 0) == 0
        except (TypeError, ValueError):
            return False

    benchmark_rows = benchmark.get("results")
    if not (
        isinstance(benchmark_rows, Sequence)
        and not isinstance(benchmark_rows, (str, bytes, bytearray))
    ):
        return False

    return bool(
        str(stage.get("status") or "").strip().lower() == "ok"
        and str(stage.get("state") or "").strip().lower() == "completed"
        and stage.get("complete") is True
        and str(benchmark.get("status") or "").strip().lower()
        == "quality_evidence_only_complete"
        and benchmark.get("performance_matrix_applicable") is False
        and benchmark.get("quality_evidence_only_complete") is True
        and benchmark.get("matrix_complete") is True
        and not benchmark_rows
        and _zero_count(benchmark, "result_count")
        and _zero_count(benchmark, "normalized_result_count")
        and benchmark_counts_ok
        and str(validation.get("status") or "").strip().lower()
        == "not_applicable_quality_evidence_only"
        and validation.get("performance_matrix_applicable") is False
        and validation.get("quality_evidence_only_complete") is True
        and _zero_count(validation, "result_count")
        and _zero_count(validation, "measured_result_count")
        and _zero_count(validation, "invalid_result_count")
        and validation_counts_ok
        and str(hardware.get("status") or "").strip().lower()
        == "not_applicable_quality_evidence_only"
        and hardware.get("quality_evidence_verified") is True
        and hardware.get("hardware_verified") is False
        and _zero_count(hardware, "normalized_result_count")
        and _zero_count(hardware, "measured_hardware_result_count")
        and _zero_count(hardware, "runtime_ok_hardware_result_count")
        and hardware_counts_ok
    )


def _model_health(
    *,
    status_row: Mapping[str, Any],
    validation_row: Mapping[str, Any],
    hardware_row: Mapping[str, Any],
    model_reasons: Sequence[Mapping[str, Any]],
    benchmark_contract: Mapping[str, Any] | None = None,
    benchmark_stage: Mapping[str, Any] | None = None,
    validation_contract: Mapping[str, Any] | None = None,
    hardware_contract: Mapping[str, Any] | None = None,
) -> str:
    if any(bool(r.get("blocking")) for r in model_reasons):
        return "partial"
    val_ok = _bool(validation_row.get("validation_ok"))
    hw_ok = _bool(hardware_row.get("hardware_verified"))
    validation_status = str(validation_row.get("validation_status") or validation_row.get("status") or "").lower()
    invalid_count = int(float(validation_row.get("invalid_result_count") or 0)) if str(validation_row.get("invalid_result_count") or "0").replace(".", "", 1).isdigit() else 0
    quality_only_complete = _full_only_quality_model_health_ok_v27538(
        benchmark_contract=benchmark_contract,
        benchmark_stage=benchmark_stage,
        validation_contract=validation_contract,
        hardware_contract=hardware_contract,
    )
    if quality_only_complete:
        # N/A is satisfied, not fabricated PASS: the model card continues to
        # expose validation_ok=None and hardware_verified=False verbatim.
        val_ok = True
        hw_ok = True
    if validation_status == "valid_complete_splits_present":
        val_ok = True
    if val_ok is False:
        # v51j/v55g: a failed auxiliary/component row is a quality result, not
        # necessarily an incomplete workflow. Keep it visible as a warning
        # unless strict validation made it blocking.
        if validation_status == "validation_failed" and invalid_count > 0:
            return "warn"
        return "partial"
    if hw_ok is False:
        return "partial"

    # v51: Hailo runtime evidence is its own dashboard dimension.  A model that
    # benchmarked, validated and passed generic hardware smoke should not be
    # shown as warn solely because Hailo runtime rows are still missing.
    material_reasons = []
    for r in model_reasons:
        if bool(r.get("blocking")):
            continue
        kind = str(r.get("kind") or "").lower()
        if kind in {"hailo_runtime", "hailo_runtime_evidence", "hailo_runtime_missing"}:
            continue
        material_reasons.append(r)
    if material_reasons:
        return "warn"
    return "ok"


def _latex_escape(value: Any) -> str:
    s = str(value if value is not None else "")
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _paper_cell(value: Any) -> str:
    """Format report table values for paper/LaTeX output.

    CSV/JSON artifacts keep full precision. The LaTeX/Markdown thesis tables
    should be readable, so they use stable rounded cells and explicit blanks.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        try:
            if not math.isfinite(float(value)):
                return ""
            x = float(value)
            ax = abs(x)
            if ax == 0:
                return "0"
            if ax >= 100:
                return f"{x:.1f}"
            if ax >= 10:
                return f"{x:.2f}"
            if ax >= 1:
                return f"{x:.3f}"
            return f"{x:.4f}"
        except Exception:
            return str(value)
    s = str(value)
    try:
        x = float(s)
        if math.isfinite(x):
            return _paper_cell(x)
    except Exception:
        pass
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _write_latex_table(path: Path, *, caption: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> Path:
    cols = "l" * len(headers)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{tab:{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        r"\hline",
        " & ".join(_latex_escape(h) for h in headers) + ' \\\\',
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(_paper_cell(v)) for v in row) + ' \\\\')
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return write_text(path, "\n".join(lines))

def _try_write_figures(reports: Path, model_cards: Sequence[Mapping[str, Any]], thesis_rows: Sequence[Mapping[str, Any]], backend_rows: Sequence[Mapping[str, Any]], pred_rows: Sequence[Mapping[str, Any]]) -> List[Path]:
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        write_text(figs / "FIGURES_NOT_GENERATED.md", "Matplotlib was not available. CSV/JSON/LaTeX reports were still generated.\n")
        return []

    def _save(fig_path: Path) -> None:
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig(fig_path, dpi=150)
        plt.close()
        created.append(fig_path)

    # 1. Best complete split latency by model.
    xs = [str(m.get("model_id") or "") for m in model_cards]
    ys = [_float(m.get("best_complete_split_latency_ms")) for m in model_cards]
    if xs and any(y is not None for y in ys):
        plt.figure(figsize=(max(6, len(xs) * 1.4), 4))
        plt.bar(xs, [y if y is not None else 0 for y in ys])
        plt.ylabel("Latency [ms]")
        plt.title("Best measured complete split latency by model")
        plt.xticks(rotation=25, ha="right")
        _save(figs / "latency_by_model.png")

    # 1b. Best pipeline FPS by model. This is the key throughput view for
    # interleaved split execution: lower single-frame latency is not the same as
    # better steady-state throughput.
    pfs = [_float(m.get("best_pipeline_fps")) for m in model_cards]
    if xs and any(y is not None for y in pfs):
        plt.figure(figsize=(max(6, len(xs) * 1.4), 4))
        plt.bar(xs, [y if y is not None else 0 for y in pfs])
        plt.ylabel("Pipeline throughput [FPS]")
        plt.title("Best interleaved pipeline throughput by model")
        plt.xticks(rotation=25, ha="right")
        _save(figs / "pipeline_fps_by_model.png")

    # 1c. Latency-vs-cycle scatter: shows the distinction between single-sample
    # latency and steady-state pipeline cycle time.
    lx = []
    cy = []
    for row in backend_rows:
        l = _float(row.get("best_complete_split_latency_ms"))
        c = _float(row.get("best_pipeline_cycle_ms"))
        if l is not None and c is not None:
            lx.append(l); cy.append(c)
    if lx:
        plt.figure(figsize=(5, 5))
        plt.scatter(lx, cy)
        lo = min(lx + cy)
        hi = max(lx + cy)
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("Single-frame split latency [ms]")
        plt.ylabel("Pipeline cycle [ms]")
        plt.title("Split latency vs streaming cycle")
        _save(figs / "latency_vs_pipeline_cycle.png")

    # 2. Predicted vs measured for rows where both exist.
    pairs = []
    labels = []
    for row in pred_rows:
        p = _float(row.get("predicted_total_latency_ms"))
        m = _float(row.get("measured_total_latency_ms"))
        if p is not None and m is not None:
            pairs.append((p, m))
            labels.append(str(row.get("model_id") or ""))
    if pairs:
        plt.figure(figsize=(5, 5))
        px = [p for p, _ in pairs]
        my = [m for _, m in pairs]
        plt.scatter(px, my)
        lo = min(px + my)
        hi = max(px + my)
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("Predicted latency [ms]")
        plt.ylabel("Measured latency [ms]")
        plt.title("Predicted vs measured complete split latency")
        _save(figs / "predicted_vs_measured.png")

    # 3. Mean speedup by backend over CPU full.
    agg: Dict[str, List[float]] = {}
    for row in backend_rows:
        s = _float(row.get("speedup_vs_cpu_full"))
        b = str(row.get("backend") or "")
        if s is not None and b:
            agg.setdefault(b, []).append(s)
    if agg:
        labels_b = sorted(agg)
        vals = [_mean(agg[b]) or 0.0 for b in labels_b]
        plt.figure(figsize=(max(6, len(labels_b) * 1.2), 4))
        plt.bar([_backend_display(b) for b in labels_b], vals)
        plt.ylabel("Speedup vs CPU full")
        plt.title("Backend speedup summary")
        plt.xticks(rotation=25, ha="right")
        _save(figs / "speedup_by_backend.png")

    # 4. Median regret by model.
    reg_by_model: Dict[str, List[float]] = {}
    for row in pred_rows:
        r = _float(row.get("regret_pct"))
        mid = str(row.get("model_id") or "")
        if r is not None and mid:
            reg_by_model.setdefault(mid, []).append(r)
    if reg_by_model:
        labels_m = sorted(reg_by_model)
        vals = [_median(reg_by_model[m]) or 0.0 for m in labels_m]
        plt.figure(figsize=(max(6, len(labels_m) * 1.4), 4))
        plt.bar(labels_m, vals)
        plt.ylabel("Median regret [%]")
        plt.title("Regret by model")
        plt.xticks(rotation=25, ha="right")
        _save(figs / "regret_by_model.png")
    return created


def _try_write_claim_figures(reports: Path, claim_rows: Sequence[Mapping[str, Any]]) -> List[Path]:
    """Write paper-oriented claim figures.

    The key thesis view compares heterogeneous pipeline speedup against the
    TensorRT full baseline and energy/frame against the same baseline.  Points
    to the right of x=1 are throughput wins; points below y=1 use less measured
    command-window energy per frame than TensorRT full.
    """
    rows = [r for r in claim_rows if _float(r.get("pipeline_speedup_vs_tensorrt_full")) is not None and _float(r.get("energy_ratio_vs_tensorrt_command_window")) is not None]
    if not rows:
        return []
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    xs = [float(_float(r.get("pipeline_speedup_vs_tensorrt_full")) or 0.0) for r in rows]
    ys = [float(_float(r.get("energy_ratio_vs_tensorrt_command_window")) or 0.0) for r in rows]
    labels = [f"{r.get('model_id','')} {r.get('best_hetero_case','')}" for r in rows]
    plt.figure(figsize=(6.4, 4.6))
    plt.scatter(xs, ys)
    x_min = min([0.8] + xs); x_max = max([1.2] + xs)
    y_min = min([0.8] + ys); y_max = max([1.2] + ys)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    for x, y, label in zip(xs, ys, labels):
        try:
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        except Exception:
            pass
    pad_x = max(0.05, (x_max - x_min) * 0.12)
    pad_y = max(0.05, (y_max - y_min) * 0.12)
    plt.xlim(x_min - pad_x, x_max + pad_x)
    plt.ylim(max(0.0, y_min - pad_y), y_max + pad_y)
    plt.xlabel("Pipeline throughput speedup vs TensorRT full [×]")
    plt.ylabel("Measured energy/frame ratio vs TensorRT full [×]")
    plt.title("Heterogeneous split throughput-energy trade-off")
    try:
        plt.tight_layout()
    except Exception:
        pass
    out = figs / "speedup_vs_energy_ratio.png"
    plt.savefig(out, dpi=180)
    plt.close()
    created.append(out)
    return created



def _row_metric_fps_per_watt_from_jpf(j_per_frame: Any) -> Optional[float]:
    v = _float(j_per_frame)
    if v in (None, 0):
        return None
    try:
        return 1.0 / float(v)
    except Exception:
        return None


def _ratio_or_none(num: Any, denom: Any) -> Optional[float]:
    n = _float(num)
    d = _float(denom)
    if n is None or d in (None, 0):
        return None
    try:
        return float(n) / float(d)
    except Exception:
        return None


def _claim_analysis_row_label(row: Mapping[str, Any]) -> str:
    model = str(row.get("model_id") or "")
    backend = str(row.get("backend_label") or row.get("backend") or "")
    case = str(row.get("case_id") or "")
    role = str(row.get("row_group") or "")
    if role == "best_split":
        return f"{model}: {backend} {case}".strip()
    return f"{model}: {backend}".strip()


def _write_claim_analysis_figures(reports: Path, rows: Sequence[Mapping[str, Any]]) -> List[Path]:
    """Write paper-facing figures for the comprehensive claim table.

    These figures use the baseline-independent thesis view: every row is
    compared to the fastest full backend for throughput and to the most
    efficient full backend for FPS/W.  Legacy TensorRT-specific figures are kept
    separately for deep-dive analysis.
    """
    if not rows:
        return []
    figs = reports / "figures" / "claims"
    figs.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    def _save(fig_path: Path) -> None:
        try:
            plt.tight_layout()
        except Exception:
            pass
        plt.savefig(fig_path, dpi=180)
        plt.close()
        created.append(fig_path)

    def _bar(name: str, title: str, y_label: str, key: str, *, lower_better: bool = False) -> None:
        data = [(r, _float(r.get(key))) for r in rows]
        data = [(r, v) for r, v in data if v is not None]
        if not data:
            return
        # Keep overview readable: full rows first within model, then best split rows.
        labels = [_claim_analysis_row_label(r) for r, _ in data]
        vals = [float(v) for _, v in data]
        width = max(7.0, min(22.0, len(vals) * 0.85))
        plt.figure(figsize=(width, 4.8))
        plt.bar(list(range(len(vals))), vals)
        plt.ylabel(y_label)
        plt.title(title + (" (lower is better)" if lower_better else ""))
        plt.xticks(list(range(len(vals))), labels, rotation=40, ha="right", fontsize=8)
        _save(figs / name)

    _bar("claim_pipeline_latency_ms_bar.png", "Single-detection latency: full models and best splits", "Latency [ms]", "single_detection_latency_ms", lower_better=True)
    _bar("claim_pipeline_fps_bar.png", "Pipeline/streaming throughput: full models and best splits", "FPS", "pipeline_fps")
    _bar("claim_fps_per_watt_bar.png", "Energy efficiency: full models and best splits", "FPS/W", "fps_per_watt_selected_fps")

    split_rows = [r for r in rows if str(r.get("row_group") or "") == "best_split" and _float(r.get("throughput_speedup_vs_fastest_full")) is not None and _float(r.get("fps_per_watt_ratio_vs_most_efficient_full")) is not None]
    if split_rows:
        xs = [float(_float(r.get("throughput_speedup_vs_fastest_full")) or 0.0) for r in split_rows]
        ys = [float(_float(r.get("fps_per_watt_ratio_vs_most_efficient_full")) or 0.0) for r in split_rows]
        labels = [_claim_analysis_row_label(r) for r in split_rows]
        plt.figure(figsize=(7.0, 5.0))
        plt.scatter(xs, ys)
        plt.axvline(1.0, linestyle="--", linewidth=1)
        plt.axhline(1.0, linestyle="--", linewidth=1)
        for x, y, label in zip(xs, ys, labels):
            try:
                plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
            except Exception:
                pass
        x_min = min([0.8] + xs); x_max = max([1.2] + xs)
        y_min = min([0.8] + ys); y_max = max([1.2] + ys)
        pad_x = max(0.05, (x_max - x_min) * 0.15)
        pad_y = max(0.05, (y_max - y_min) * 0.15)
        plt.xlim(x_min - pad_x, x_max + pad_x)
        plt.ylim(max(0.0, y_min - pad_y), y_max + pad_y)
        plt.xlabel("Pipeline throughput vs fastest full backend [×]")
        plt.ylabel("FPS/W vs most efficient full backend [×]")
        plt.title("Best split trade-off vs best full baselines")
        _save(figs / "claim_speedup_efficiency_vs_best_full.png")

    # Model-level compact view: one best split per model compared to its best full baselines.
    best_split_by_model: Dict[str, Mapping[str, Any]] = {}
    for r in split_rows:
        mid = str(r.get("model_id") or "")
        if not mid:
            continue
        old = best_split_by_model.get(mid)
        if old is None or float(_float(r.get("pipeline_fps")) or -1.0) > float(_float(old.get("pipeline_fps")) or -1.0):
            best_split_by_model[mid] = r
    if best_split_by_model:
        labels = sorted(best_split_by_model)
        speedups = [float(_float(best_split_by_model[m].get("throughput_speedup_vs_fastest_full")) or 0.0) for m in labels]
        effs = [float(_float(best_split_by_model[m].get("fps_per_watt_ratio_vs_most_efficient_full")) or 0.0) for m in labels]
        x = list(range(len(labels)))
        width = max(6.5, len(labels) * 1.0)
        plt.figure(figsize=(width, 4.8))
        # Offset bars without specifying colors; matplotlib defaults are fine.
        plt.bar([i - 0.18 for i in x], speedups, width=0.36, label="Throughput vs fastest full")
        plt.bar([i + 0.18 for i in x], effs, width=0.36, label="FPS/W vs most efficient full")
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.ylabel("Ratio [×]")
        plt.title("Best split ratios against best full baselines")
        plt.xticks(x, labels, rotation=25, ha="right")
        try:
            plt.legend(fontsize=8)
        except Exception:
            pass
        _save(figs / "claim_best_split_ratios_by_model.png")
    return created


def _write_claim_analysis_reports(
    reports: Path,
    tables: Path,
    *,
    full_backend_rows: Sequence[Mapping[str, Any]],
    hetero_pipeline_rows: Sequence[Mapping[str, Any]],
    model_cards: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Write the comprehensive thesis claim table requested for documentation.

    It contains all full backends per model plus the best heterogeneous split per
    model/backend.  The primary comparisons are baseline independent:
    throughput uses the fastest full backend; energy efficiency uses the most
    efficient full backend by FPS/W.  TensorRT-specific legacy tables remain
    available for deep analysis.
    """
    out: Dict[str, Any] = {"rows": [], "full_rows": [], "best_split_rows": [], "figures": []}
    model_ids = sorted({str(m.get("model_id") or "") for m in model_cards} | {str(r.get("model_id") or "") for r in full_backend_rows} | {str(r.get("model_id") or "") for r in hetero_pipeline_rows})
    model_ids = [m for m in model_ids if m]
    rows: List[Dict[str, Any]] = []
    full_out_rows: List[Dict[str, Any]] = []
    split_out_rows: List[Dict[str, Any]] = []

    def _row_fps_per_watt_selected(row: Mapping[str, Any]) -> Optional[float]:
        direct = _float(row.get("energy_fps_per_watt") or row.get("fps_per_watt_from_selected_fps") or row.get("energy_streaming_fps_per_watt_from_selected_fps"))
        if direct is not None:
            return direct
        v = _row_metric_fps_per_watt_from_jpf(row.get("energy_j_per_frame_from_selected_fps"))
        if v is not None:
            return v
        fps = _float(row.get("pipeline_fps") or row.get("full_backend_throughput_fps"))
        pwr = _float(row.get("avg_power_w") or row.get("energy_streaming_avg_power_w"))
        return (fps / pwr) if (fps is not None and pwr not in (None, 0)) else None

    def _row_fps_per_watt_command(row: Mapping[str, Any]) -> Optional[float]:
        return _row_metric_fps_per_watt_from_jpf(row.get("energy_j_per_frame_command_window") or row.get("energy_j_per_frame"))

    def _backend_key(row: Mapping[str, Any]) -> str:
        return str(row.get("backend") or row.get("backend_label") or "").strip().lower()

    for mid in model_ids:
        fulls = [dict(r) for r in full_backend_rows if str(r.get("model_id") or "") == mid and _float(r.get("full_backend_throughput_fps")) is not None and _row_eligible_for_ranking(r) is True]
        splits_all = [dict(r) for r in hetero_pipeline_rows if str(r.get("model_id") or "") == mid and _float(r.get("pipeline_fps")) is not None and _row_eligible_for_ranking(r) is True and str(r.get("quality") or "").lower() != "invalid"]
        if not fulls and not splits_all:
            continue
        fastest_full = max(fulls, key=lambda r: float(_float(r.get("full_backend_throughput_fps")) or -1.0), default={})
        efficient_fulls = [r for r in fulls if _row_fps_per_watt_selected(r) is not None]
        most_efficient_full = max(efficient_fulls, key=lambda r: float(_row_fps_per_watt_selected(r) or -1.0), default={})
        fastest_full_fps = _float(fastest_full.get("full_backend_throughput_fps")) if fastest_full else None
        fastest_full_latency = _float(fastest_full.get("latency_ms")) if fastest_full else None
        efficient_full_fpw = _row_fps_per_watt_selected(most_efficient_full) if most_efficient_full else None
        efficient_full_jpf_sel = _float(most_efficient_full.get("energy_j_per_frame_from_selected_fps")) if most_efficient_full else None
        efficient_full_jpf_cmd = _float(most_efficient_full.get("energy_j_per_frame_command_window")) if most_efficient_full else None

        # Best split per heterogeneous backend, selected by streaming/pipeline FPS.
        best_split_by_backend: Dict[str, Dict[str, Any]] = {}
        for s in splits_all:
            key = _backend_key(s)
            old = best_split_by_backend.get(key)
            if old is None or float(_float(s.get("pipeline_fps")) or -1.0) > float(_float(old.get("pipeline_fps")) or -1.0):
                best_split_by_backend[key] = s
        best_split_overall_key = ""
        if best_split_by_backend:
            best_overall = max(best_split_by_backend.values(), key=lambda r: float(_float(r.get("pipeline_fps")) or -1.0))
            best_split_overall_key = _backend_key(best_overall) + "::" + str(best_overall.get("case_id") or "")

        def _make_row(source: Mapping[str, Any], *, row_group: str, is_best_split: bool = False) -> Dict[str, Any]:
            if row_group == "full_model":
                fps = _float(source.get("full_backend_throughput_fps"))
                latency = _float(source.get("latency_ms"))
                cycle = latency
                case_id = str(source.get("canonical_case_id") or source.get("case_id") or "")
            else:
                fps = _float(source.get("pipeline_fps"))
                latency = _float(source.get("split_latency_ms"))
                cycle = _float(source.get("pipeline_cycle_ms"))
                case_id = str(source.get("case_id") or "")
            j_cmd = _float(source.get("energy_j_per_frame_command_window") or source.get("energy_j_per_frame"))
            j_sel = _float(source.get("energy_j_per_frame_from_selected_fps"))
            fpw_sel = _row_fps_per_watt_selected(source)
            fpw_cmd = _row_fps_per_watt_command(source)
            speed_ratio = _ratio_or_none(fps, fastest_full_fps)
            eff_ratio = _ratio_or_none(fpw_sel, efficient_full_fpw)
            j_sel_ratio = _ratio_or_none(j_sel, efficient_full_jpf_sel)
            j_cmd_ratio = _ratio_or_none(j_cmd, efficient_full_jpf_cmd)
            fastest_key = _backend_key(fastest_full)
            efficient_key = _backend_key(most_efficient_full)
            backend_key = _backend_key(source)
            split_key = backend_key + "::" + case_id
            fastest = bool(row_group == "full_model" and fastest_key and backend_key == fastest_key)
            efficient = bool(row_group == "full_model" and efficient_key and backend_key == efficient_key)
            split_overall = bool(row_group == "best_split" and split_key == best_split_overall_key)
            if row_group == "full_model" and fastest and efficient:
                claim = "fastest_and_most_efficient_full_baseline"
            elif row_group == "full_model" and fastest:
                claim = "fastest_full_baseline"
            elif row_group == "full_model" and efficient:
                claim = "most_efficient_full_baseline"
            elif row_group == "full_model":
                claim = "full_baseline"
            elif _float(speed_ratio) is not None and float(_float(speed_ratio) or 0.0) >= 1.0 and _float(eff_ratio) is not None and float(_float(eff_ratio) or 0.0) >= 1.0:
                claim = "split_throughput_and_fps_per_watt_win_vs_best_full"
            elif _float(speed_ratio) is not None and float(_float(speed_ratio) or 0.0) >= 1.0:
                claim = "split_throughput_win_vs_fastest_full"
            elif _float(eff_ratio) is not None and float(_float(eff_ratio) or 0.0) >= 1.0:
                claim = "split_fps_per_watt_win_vs_most_efficient_full"
            else:
                claim = "valid_split_below_best_full"
            row = {
                "model_id": mid,
                "row_group": row_group,
                "claim_role": "best_split_overall" if split_overall else claim,
                "backend": source.get("backend", ""),
                "backend_label": source.get("backend_label", source.get("backend", "")),
                "case_id": case_id,
                "is_fastest_full_baseline": fastest,
                "is_most_efficient_full_baseline": efficient,
                "is_best_split_overall": split_overall,
                # User-facing names: latency is single detection/e2e latency;
                # cycle/FPS are steady-state streaming pipeline metrics.
                "single_detection_latency_ms": _fmt(latency),
                "pipeline_latency_ms": _fmt(latency),
                "pipeline_cycle_ms": _fmt(cycle),
                "pipeline_fps": _fmt(fps),
                "avg_power_w": _fmt(source.get("avg_power_w")),
                "fps_per_watt_selected_fps": _fmt(fpw_sel),
                "fps_per_watt_command_window": _fmt(fpw_cmd),
                "j_per_frame_selected_fps": _fmt(j_sel),
                "j_per_frame_command_window": _fmt(j_cmd),
                "fastest_full_backend": fastest_full.get("backend", "") if fastest_full else "",
                "fastest_full_backend_label": fastest_full.get("backend_label", fastest_full.get("backend", "")) if fastest_full else "",
                "fastest_full_latency_ms": _fmt(fastest_full_latency),
                "fastest_full_fps": _fmt(fastest_full_fps),
                "most_efficient_full_backend": most_efficient_full.get("backend", "") if most_efficient_full else "",
                "most_efficient_full_backend_label": most_efficient_full.get("backend_label", most_efficient_full.get("backend", "")) if most_efficient_full else "",
                "most_efficient_full_fps_per_watt": _fmt(efficient_full_fpw),
                "throughput_speedup_vs_fastest_full": _fmt(speed_ratio),
                "fps_per_watt_ratio_vs_most_efficient_full": _fmt(eff_ratio),
                "j_per_frame_selected_ratio_vs_most_efficient_full": _fmt(j_sel_ratio),
                "j_per_frame_command_ratio_vs_most_efficient_full": _fmt(j_cmd_ratio),
                "energy_coverage_status": source.get("energy_coverage_status", ""),
                "energy_target_status": source.get("energy_target_status", ""),
                "energy_measurement_scope": source.get("energy_measurement_scope", ""),
                "quality": source.get("quality", ""),
                "validation_claim_label": source.get("validation_claim_label", ""),
                "final_pass": source.get("final_pass", ""),
                "paper_note": "Speedup baseline is fastest full backend; FPS/W baseline is most energy-efficient full backend. TensorRT-specific legacy columns/tables are still exported separately.",
            }
            return row

        for f in sorted(fulls, key=lambda r: (_backend_key(r))):
            rr = _make_row(f, row_group="full_model")
            full_out_rows.append(rr)
            rows.append(rr)
        for _, s in sorted(best_split_by_backend.items(), key=lambda kv: kv[0]):
            rr = _make_row(s, row_group="best_split", is_best_split=True)
            split_out_rows.append(rr)
            rows.append(rr)

    rows.sort(key=lambda r: (str(r.get("model_id") or ""), 0 if str(r.get("row_group")) == "full_model" else 1, str(r.get("backend_label") or r.get("backend") or ""), str(r.get("case_id") or "")))
    columns = [
        "model_id", "row_group", "claim_role", "backend", "backend_label", "case_id",
        "is_fastest_full_baseline", "is_most_efficient_full_baseline", "is_best_split_overall",
        "single_detection_latency_ms", "pipeline_latency_ms", "pipeline_cycle_ms", "pipeline_fps",
        "avg_power_w", "fps_per_watt_selected_fps", "fps_per_watt_command_window",
        "j_per_frame_selected_fps", "j_per_frame_command_window",
        "fastest_full_backend", "fastest_full_backend_label", "fastest_full_latency_ms", "fastest_full_fps",
        "most_efficient_full_backend", "most_efficient_full_backend_label", "most_efficient_full_fps_per_watt",
        "throughput_speedup_vs_fastest_full", "fps_per_watt_ratio_vs_most_efficient_full",
        "j_per_frame_selected_ratio_vs_most_efficient_full", "j_per_frame_command_ratio_vs_most_efficient_full",
        "energy_coverage_status", "energy_target_status", "energy_measurement_scope",
        "quality", "validation_claim_label", "final_pass", "paper_note",
    ]
    p_comp_csv = write_csv(reports / "claim_table_comprehensive.csv", rows, columns)
    p_full_csv = write_csv(reports / "claim_table_full_models.csv", full_out_rows, columns)
    p_split_csv = write_csv(reports / "claim_table_best_splits.csv", split_out_rows, columns)
    p_comp_json = write_json(reports / "claim_table_comprehensive.json", {"schema": "onnx-splitpoint/thesis-claim-table", "schema_version": 1, "rows": rows})

    md = [
        "# Thesis claim table", "",
        "This table contains all full-model baselines and the best heterogeneous split per model/backend. Latency is the single-image or single-detection end-to-end time. Pipeline FPS is the steady-state streaming throughput for split pipelines and the measured/derived throughput for full backends.", "",
        "Throughput ratios use the fastest full backend as baseline. FPS/W ratios use the most energy-efficient full backend as baseline. This avoids hard-coding TensorRT as the only comparison while keeping the legacy TensorRT-specific deep-dive reports intact.", "",
        "| Model | Kind | Backend/case | Latency [ms] | Pipeline FPS | FPS/W | Speedup vs fastest full | FPS/W ratio vs best full | Claim | Validation |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        backend_case = str(r.get("backend_label") or r.get("backend") or "")
        if str(r.get("row_group") or "") == "best_split" and str(r.get("case_id") or ""):
            backend_case += f" / {r.get('case_id')}"
        md.append("| " + " | ".join([
            str(r.get("model_id") or ""),
            "Full" if str(r.get("row_group") or "") == "full_model" else "Best split",
            backend_case,
            _paper_cell(r.get("single_detection_latency_ms")),
            _paper_cell(r.get("pipeline_fps")),
            _paper_cell(r.get("fps_per_watt_selected_fps")),
            _paper_cell(r.get("throughput_speedup_vs_fastest_full")),
            _paper_cell(r.get("fps_per_watt_ratio_vs_most_efficient_full")),
            str(r.get("claim_role") or ""),
            str(r.get("validation_claim_label") or ""),
        ]) + " |")
    p_comp_md = write_text(reports / "claim_table_comprehensive.md", "\n".join(md) + "\n")
    p_comp_tex = _write_latex_table(
        tables / "claim_table_comprehensive.tex",
        caption="Full-model baselines and best heterogeneous split claims. Latency is single-detection latency; FPS/W uses selected streaming throughput divided by measured average power.",
        label="claim_table_comprehensive",
        headers=["Model", "Kind", "Backend/case", "Latency [ms]", "FPS", "FPS/W", "Speedup", "FPS/W ratio", "Claim"],
        rows=[[
            r.get("model_id", ""),
            "Full" if str(r.get("row_group") or "") == "full_model" else "Best split",
            (str(r.get("backend_label") or r.get("backend") or "") + ((" / " + str(r.get("case_id") or "")) if str(r.get("row_group") or "") == "best_split" else "")),
            r.get("single_detection_latency_ms", ""),
            r.get("pipeline_fps", ""),
            r.get("fps_per_watt_selected_fps", ""),
            r.get("throughput_speedup_vs_fastest_full", ""),
            r.get("fps_per_watt_ratio_vs_most_efficient_full", ""),
            r.get("claim_role", ""),
        ] for r in rows],
    )
    p_split_tex = _write_latex_table(
        tables / "claim_table_best_splits.tex",
        caption="Best heterogeneous split per model/backend compared against the fastest and most energy-efficient full-model baselines.",
        label="claim_table_best_splits",
        headers=["Model", "Split", "Case", "Latency [ms]", "Cycle [ms]", "FPS", "Speedup", "FPS/W ratio", "Validation"],
        rows=[[
            r.get("model_id", ""), r.get("backend_label", r.get("backend", "")), r.get("case_id", ""),
            r.get("single_detection_latency_ms", ""), r.get("pipeline_cycle_ms", ""), r.get("pipeline_fps", ""),
            r.get("throughput_speedup_vs_fastest_full", ""), r.get("fps_per_watt_ratio_vs_most_efficient_full", ""), r.get("validation_claim_label", ""),
        ] for r in split_out_rows],
    )
    figs = _write_claim_analysis_figures(reports, rows)
    out.update({
        "rows": rows,
        "full_rows": full_out_rows,
        "best_split_rows": split_out_rows,
        "comprehensive_csv": p_comp_csv,
        "comprehensive_json": p_comp_json,
        "comprehensive_md": p_comp_md,
        "comprehensive_tex": p_comp_tex,
        "full_models_csv": p_full_csv,
        "best_splits_csv": p_split_csv,
        "best_splits_tex": p_split_tex,
        "figures": figs,
    })
    return out


def _safe_filename(value: Any, default: str = "item") -> str:
    s = str(value or default).strip()
    if not s:
        s = default
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)[:160]



def _hash_int(value: str) -> int:
    try:
        return int(hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:8], 16)
    except Exception:
        return 0


# Imagenette is a 10-class subset of ImageNet.  The validation reports often
# carry ImageNet integer ids only, while the image filenames carry synset ids.
# These labels make the visual sanity report understandable without changing
# the numeric validation logic.
_IMAGENETTE_SYNSET_NAMES: Dict[str, str] = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}
_IMAGENETTE_ID_NAMES: Dict[int, str] = {
    0: "tench",
    217: "English springer",
    482: "cassette player",
    491: "chain saw",
    497: "church",
    566: "French horn",
    569: "garbage truck",
    571: "gas pump",
    574: "golf ball",
    701: "parachute",
}


def _imagenette_synset_from_name(value: Any) -> str:
    s = Path(str(value or "")).name
    if len(s) >= 9 and s.startswith("n") and s[1:9].isdigit():
        return s[:9]
    return ""


def _label_display(label_id: Any, *, image_name: str = "") -> str:
    if label_id is None or label_id == "":
        syn = _imagenette_synset_from_name(image_name)
        return _IMAGENETTE_SYNSET_NAMES.get(syn, "-")
    try:
        idx = int(label_id)
        return f"{idx} {_IMAGENETTE_ID_NAMES.get(idx, '')}".strip()
    except Exception:
        return str(label_id)


def _build_validation_image_index() -> Dict[str, Path]:
    roots = [Path.home() / ".onnx_splitpoint_tool" / "validation_datasets"]
    index: Dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        try:
            for p in root.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    index.setdefault(p.name, p)
                    try:
                        index.setdefault(str(p), p)
                        index.setdefault(str(p.relative_to(root)), p)
                    except Exception:
                        pass
        except Exception:
            continue
    return index


def _pick_dataset_visual_image(image_index: Mapping[str, Path], *, task: str, key: str) -> Optional[Path]:
    paths = sorted({Path(p) for p in image_index.values() if isinstance(p, Path) and p.is_file()})
    if not paths:
        return None
    task_l = str(task or "").lower()
    if task_l == "classification":
        preferred = [p for p in paths if "classification" in str(p) or "imagenette" in str(p)]
    elif task_l == "detection":
        preferred = [p for p in paths if "detection" in str(p) or "coco" in str(p)]
    else:
        preferred = []
    pool = preferred or paths
    return pool[_hash_int(key) % len(pool)] if pool else None


def _report_backend_from_dir(path: Path) -> str:
    name = path.parent.name
    if name.startswith("results_"):
        return name[len("results_"):]
    return name


def _infer_model_id_from_report_path(path: Path) -> str:
    try:
        parts = list(path.parts)
        if "models" in parts:
            idx = parts.index("models")
            if idx + 1 < len(parts):
                return str(parts[idx + 1])
    except Exception:
        pass
    return "model"


def _choose_variant(report: Mapping[str, Any]) -> str:
    primary = str(report.get("primary_variant") or "").lower()
    if primary in {"composed", "full", "part1", "part2"}:
        return primary
    variants = report.get("measured_variants") if isinstance(report.get("measured_variants"), list) else []
    if "composed" in variants:
        return "composed"
    if "full" in variants:
        return "full"
    return "composed"


def _report_task(report: Mapping[str, Any]) -> str:
    return str(report.get("benchmark_task_used") or report.get("benchmark_task_requested") or (report.get("viz") or {}).get("task") or "auto").lower()


def _variant_block(report: Mapping[str, Any], section: str, variant: str) -> Mapping[str, Any]:
    obj = report.get(section) if isinstance(report.get(section), Mapping) else {}
    variants = obj.get("variants") if isinstance(obj.get("variants"), Mapping) else {}
    block = variants.get(variant) if isinstance(variants.get(variant), Mapping) else {}
    if not block and variant != "composed":
        block = variants.get("composed") if isinstance(variants.get("composed"), Mapping) else {}
    if not block:
        block = variants.get("full") if isinstance(variants.get("full"), Mapping) else {}
    return block


def _variant_aggregate_metrics(report: Mapping[str, Any], *, task: str, variant: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if task == "classification":
        for section in ("mini_classification_eval", "validation_dataset"):
            block = _variant_block(report, section, variant)
            metrics = block.get("metrics") if isinstance(block.get("metrics"), Mapping) else block
            for k in ("top1_accuracy", "top5_accuracy", "top1_agreement", "top5_agreement", "images", "labeled_images", "status"):
                if k in metrics and k not in out:
                    out[k] = metrics.get(k)
    elif task == "detection":
        for section in ("mini_coco_ap50", "validation_dataset"):
            block = _variant_block(report, section, variant)
            metrics = block.get("metrics") if isinstance(block.get("metrics"), Mapping) else block
            for k in ("ap50", "ap50_proxy", "mean_iou", "mean_iou_matched", "matched", "n_ref", "n_out", "images", "status"):
                if k in metrics and k not in out:
                    out[k] = metrics.get(k)
    return out


def _image_exists_for_record(image_index: Mapping[str, Path], image_value: Any) -> bool:
    raw = str(image_value or "")
    if not raw:
        return False
    if raw in image_index or Path(raw).name in image_index:
        return True
    try:
        return Path(raw).expanduser().is_file()
    except Exception:
        return False


def _select_record_with_existing_image(records: Sequence[Any], image_index: Mapping[str, Path], *, key: str) -> Optional[Mapping[str, Any]]:
    dicts: List[Mapping[str, Any]] = []
    for r in records:
        if isinstance(r, Mapping):
            dicts.append(r)
        elif r:
            dicts.append({"image": str(r)})
    if not dicts:
        return None
    # Prefer a record whose image is present locally; visual verification is
    # meant to render an actual image, not just draw a metric card.
    present = [r for r in dicts if _image_exists_for_record(image_index, r.get("image"))]
    pool = present or dicts
    return pool[_hash_int(key) % len(pool)]


def _classification_records(report: Mapping[str, Any], variant: str) -> List[Any]:
    sources: List[Any] = []
    for section in ("mini_classification_eval", "validation_dataset"):
        block = _variant_block(report, section, variant)
        imgs = block.get("images") if isinstance(block.get("images"), list) else []
        if imgs:
            sources.extend(imgs)
    bd = report.get("backend_drift") if isinstance(report.get("backend_drift"), Mapping) else {}
    ds = bd.get("dataset") if isinstance(bd.get("dataset"), Mapping) else {}
    imgs = ds.get("images") if isinstance(ds.get("images"), list) else []
    if imgs:
        sources.extend(imgs)
    return sources


def _detection_records(report: Mapping[str, Any], variant: str) -> List[Any]:
    sources: List[Any] = []
    for section in ("mini_coco_ap50", "validation_dataset"):
        block = _variant_block(report, section, variant)
        imgs = block.get("images") if isinstance(block.get("images"), list) else []
        if imgs:
            sources.extend(imgs)
    return sources


def _select_visual_sample(report: Mapping[str, Any], report_path: Path, image_index: Optional[Mapping[str, Path]] = None) -> Dict[str, Any]:
    task = _report_task(report)
    variant = _choose_variant(report)
    model_id = _infer_model_id_from_report_path(report_path)
    backend = _report_backend_from_dir(report_path)
    case_id = str(report.get("case_id") or report_path.parent.parent.name)
    # Keep the same deterministic image for a model/case across backends when
    # possible.  This makes visual comparison easier, while still avoiding the
    # previous "always the first fish" behaviour across all cases.
    key = f"{model_id}|{case_id}|{variant}|visual-v2"
    image_index = image_index or {}

    if task == "classification":
        rec = _select_record_with_existing_image(_classification_records(report, variant), image_index, key=key)
        if rec is not None:
            return {"variant": variant, "task": task, "record": dict(rec), "image_name": str(rec.get("image") or ""), "aggregate_metrics": _variant_aggregate_metrics(report, task=task, variant=variant), "prediction_available": bool(isinstance(rec.get("metrics"), Mapping))}
    if task == "detection":
        viz = report.get("viz") if isinstance(report.get("viz"), Mapping) else {}
        block = viz.get(variant) if isinstance(viz.get(variant), Mapping) else viz.get("composed") if isinstance(viz.get("composed"), Mapping) else viz.get("full") if isinstance(viz.get("full"), Mapping) else {}
        j = block.get("json") if isinstance(block.get("json"), Mapping) else {}
        prov = j.get("provenance") if isinstance(j.get("provenance"), Mapping) else {}
        img = str(prov.get("image") or "")
        if img:
            return {"variant": variant, "task": task, "record": j, "image_name": img, "detections": j.get("detections") if isinstance(j.get("detections"), list) else [], "aggregate_metrics": _variant_aggregate_metrics(report, task=task, variant=variant), "prediction_available": True}
        rec = _select_record_with_existing_image(_detection_records(report, variant), image_index, key=key)
        if rec is not None:
            return {"variant": variant, "task": task, "record": dict(rec), "image_name": str(rec.get("image") or ""), "detections": rec.get("detections") if isinstance(rec.get("detections"), list) else [], "aggregate_metrics": _variant_aggregate_metrics(report, task=task, variant=variant), "prediction_available": bool(isinstance(rec.get("detections"), list) or isinstance(rec.get("metrics"), Mapping))}
    # Fall back to the lightweight viz top-k/detection synthetic sample.
    viz = report.get("viz") if isinstance(report.get("viz"), Mapping) else {}
    for cand in (variant, "composed", "full"):
        block = viz.get(cand) if isinstance(viz.get(cand), Mapping) else {}
        j = block.get("json") if isinstance(block.get("json"), Mapping) else {}
        if j:
            prov = j.get("provenance") if isinstance(j.get("provenance"), Mapping) else {}
            return {"variant": cand, "task": str(j.get("task") or task), "record": j, "image_name": str(prov.get("image") or ""), "aggregate_metrics": _variant_aggregate_metrics(report, task=task, variant=cand), "prediction_available": bool(j.get("topk") or j.get("detections"))}
    return {"variant": variant, "task": task, "record": {}, "image_name": "", "aggregate_metrics": _variant_aggregate_metrics(report, task=task, variant=variant), "prediction_available": False}


def _draw_visual_snapshot(*, report: Mapping[str, Any], report_path: Path, out_png: Path, image_index: Mapping[str, Path]) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    sample = _select_visual_sample(report, report_path, image_index=image_index)
    task = str(sample.get("task") or "auto")
    variant = str(sample.get("variant") or "")
    rec = sample.get("record") if isinstance(sample.get("record"), Mapping) else {}
    aggregate_metrics = sample.get("aggregate_metrics") if isinstance(sample.get("aggregate_metrics"), Mapping) else {}
    raw_image_value = str(sample.get("image_name") or "")
    image_name = Path(raw_image_value).name
    image_path = None
    if raw_image_value:
        image_path = image_index.get(raw_image_value) or image_index.get(image_name)
        if image_path is None:
            p = Path(raw_image_value).expanduser()
            if p.is_file():
                image_path = p
    prediction_available = bool(sample.get("prediction_available"))
    dataset_fallback_used = False
    # v59cr: do not render arbitrary fallback images for rows without an
    # actual per-image prediction. Such images looked like false predictions
    # and made visual reviews misleading.
    if not (image_path and Path(image_path).is_file()) and prediction_available:
        fallback = _pick_dataset_visual_image(image_index, task=task, key=f"{report_path}|{task}|fallback")
        if fallback is not None:
            image_path = fallback
            image_name = fallback.name
            dataset_fallback_used = True

    fig, ax = plt.subplots(figsize=(8, 5))
    if image_path and Path(image_path).is_file():
        try:
            img = plt.imread(str(image_path))
            ax.imshow(img)
            ax.set_axis_off()
        except Exception:
            ax.set_axis_off()
            ax.text(0.02, 0.95, f"Could not read image: {image_name}", va="top", transform=ax.transAxes)
    else:
        ax.set_axis_off()
        if not prediction_available:
            ax.text(0.02, 0.95, "NO PER-IMAGE PREDICTION STORED\nThis row is not visually claimable.\nUse dataset metrics or rerun with per-image dumps.", va="top", transform=ax.transAxes, fontsize=12, bbox={"facecolor":"mistyrose", "alpha":0.9, "pad":6})
        else:
            ax.text(0.02, 0.95, f"Validation image unavailable: {image_name or 'n/a'}", va="top", transform=ax.transAxes)

    run_cfg = report.get("run_cfg") if isinstance(report.get("run_cfg"), Mapping) else {}
    model_value = run_cfg.get("model_id")
    model_id = str(model_value) if model_value not in (None, "") else _infer_model_id_from_report_path(report_path)
    backend = _report_backend_from_dir(report_path)
    case_id = str(report.get("case_id") or report_path.parent.parent.name)
    status = "PASS" if _bool(report.get("final_pass")) is True else "FAIL" if _bool(report.get("final_pass")) is False else "UNKNOWN"
    title = f"{model_id} · {case_id} · {backend} · {variant} · {status}"
    ax.set_title(title, fontsize=10)

    lines: List[str] = []
    if task == "classification":
        viz = report.get("viz") if isinstance(report.get("viz"), Mapping) else {}
        block = viz.get(variant) if isinstance(viz.get(variant), Mapping) else {}
        topk = []
        j = block.get("json") if isinstance(block.get("json"), Mapping) else {}
        if isinstance(j.get("topk"), list):
            topk = j.get("topk")[:5]
        metrics = rec.get("metrics") if isinstance(rec.get("metrics"), Mapping) else {}
        gt = rec.get("gt") if isinstance(rec.get("gt"), Mapping) else {}
        lines.append(f"task=classification image={image_name or '-'}")
        if gt or rec.get("label_id") is not None:
            gt_id = gt.get("label_id", rec.get("label_id")) if isinstance(gt, Mapping) else rec.get("label_id")
            lines.append(f"GT={_label_display(gt_id, image_name=image_name)} top1_hit={gt.get('top1_hit') if isinstance(gt, Mapping) else '-'} top5_hit={gt.get('top5_hit') if isinstance(gt, Mapping) else '-'}")
        if metrics:
            lines.append(f"ref={_label_display(metrics.get('top1_ref'), image_name=image_name)} pred={_label_display(metrics.get('top1_out'), image_name=image_name)} top5_overlap={metrics.get('top5_overlap')}")
            cs = _float(metrics.get("cosine_similarity"))
            if cs is not None:
                lines.append(f"cosine={cs:.4f} max_abs={metrics.get('max_abs')}")
        if aggregate_metrics:
            t1 = _float(aggregate_metrics.get("top1_accuracy")); t5 = _float(aggregate_metrics.get("top5_accuracy"))
            a1 = _float(aggregate_metrics.get("top1_agreement")); a5 = _float(aggregate_metrics.get("top5_agreement"))
            parts = []
            if t1 is not None: parts.append(f"Top1={t1:.2f}")
            if t5 is not None: parts.append(f"Top5={t5:.2f}")
            if a1 is not None: parts.append(f"Agree1={a1:.2f}")
            if a5 is not None: parts.append(f"Agree5={a5:.2f}")
            if parts:
                lines.append("dataset: " + " ".join(parts))
        if topk:
            lines.append("top-k: " + ", ".join(f"{x.get('label', x.get('id'))}:{float(x.get('p',0)):.3f}" for x in topk if isinstance(x, Mapping)))
        if not prediction_available:
            lines.append("VISUAL FAIL: no per-image prediction stored; not claimable")
            lines.append("NOTE: aggregate metrics shown only")
    elif task == "detection":
        dets = sample.get("detections") if isinstance(sample.get("detections"), list) else []
        lines.append(f"task=detection image={image_name or '-'} detections={len(dets)}")
        for d in dets[:12]:
            if not isinstance(d, Mapping):
                continue
            x1 = _float(d.get("x1")); y1 = _float(d.get("y1")); x2 = _float(d.get("x2")); y2 = _float(d.get("y2"))
            label = str(d.get("label") or d.get("class_id") or "obj")
            score = _float(d.get("score")) or 0.0
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None and image_path:
                ax.add_patch(patches.Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1), fill=False, linewidth=1.4))
                ax.text(x1, max(0, y1 - 2), f"{label} {score:.2f}", fontsize=7, bbox={"facecolor": "white", "alpha": 0.6, "pad": 1})
        metrics = rec.get("metrics") if isinstance(rec.get("metrics"), Mapping) else {}
        if metrics:
            lines.append(f"matched={metrics.get('matched')} n_ref={metrics.get('n_ref')} n_out={metrics.get('n_out')} mean_iou={metrics.get('mean_iou_matched')}")
        if aggregate_metrics:
            vals = []
            for k in ("ap50", "ap50_proxy", "mean_iou", "mean_iou_matched"):
                x = _float(aggregate_metrics.get(k))
                if x is not None:
                    vals.append(f"{k}={x:.3f}")
            if vals:
                lines.append("dataset: " + " ".join(vals[:4]))
        if len(dets) == 0:
            if not prediction_available:
                lines.append("VISUAL FAIL: no per-image detections stored; not claimable")
            else:
                lines.append("NOTE: zero detections after postprocess")
    else:
        lines.append(f"task={task} image={image_name or '-'}")

    ax.text(0.01, 0.01, "\n".join(lines[:8]), fontsize=8, va="bottom", ha="left", transform=ax.transAxes, bbox={"facecolor": "white", "alpha": 0.78, "pad": 3})
    try:
        fig.tight_layout()
    except Exception:
        pass
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return {
        "image_name": image_name,
        "image_found": bool(image_path and Path(image_path).is_file()),
        "dataset_fallback_used": bool(dataset_fallback_used),
        "prediction_available": prediction_available,
        "visual_claimable": bool(prediction_available and status == "PASS"),
        "visual_status": ("visual_pass" if prediction_available and status == "PASS" else ("visual_fail_no_per_image_prediction" if not prediction_available else "visual_fail")),
        "task": task,
        "variant": variant,
        "backend": backend,
        "case_id": case_id,
        "status": status,
        "classification_top1": aggregate_metrics.get("top1_accuracy"),
        "classification_top5": aggregate_metrics.get("top5_accuracy"),
        "png": out_png.name,
    }


def _write_visual_verification_report(run_dir: Path, reports: Path) -> Dict[str, Any]:
    vis_dir = reports / "visual_verification"
    img_dir = vis_dir / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib  # noqa: F401
    except Exception:
        note = write_text(vis_dir / "VISUALS_NOT_GENERATED.md", "Matplotlib is not available. Visual verification snapshots were not generated.\n")
        return {"visual_verification_note": note, "visual_verification_count": 0}

    image_index = _build_validation_image_index()
    by_model: Dict[str, List[Path]] = {}
    report_paths: List[Path] = []
    for root in (run_dir / "models").glob("*/benchmark_results/remote_diagnostics/case_reports/results"):
        report_paths.extend(sorted(root.glob("*/results_*/validation_report.json")))
    for root in (run_dir / "models").glob("*/benchmark_results/remote_diagnostics/lean_bundle"):
        report_paths.extend(sorted(root.glob("*/results_*/validation_report.json")))
    for rp in report_paths:
        model = _infer_model_id_from_report_path(rp)
        by_model.setdefault(model, []).append(rp)

    rows: List[Dict[str, Any]] = []
    md = ["# Visual verification snapshots", "", "One deterministic validation image is rendered for each sampled model/case/backend report. The snapshots are intended for quick visual sanity checks; dataset metrics remain the authority for pass/fail decisions.", ""]
    max_per_model = 40
    for model, paths in sorted(by_model.items()):
        seen: set[Tuple[str, str, str]] = set()
        count = 0
        for rp in sorted(paths):
            try:
                data = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:
                continue
            case = str(data.get("case_id") or rp.parent.parent.name)
            backend = _report_backend_from_dir(rp)
            key = (model, case, backend)
            if key in seen:
                continue
            seen.add(key)
            if count >= max_per_model:
                break
            out_png = img_dir / f"{_safe_filename(model)}__{_safe_filename(case)}__{_safe_filename(backend)}.png"
            try:
                rec = _draw_visual_snapshot(report=data, report_path=rp, out_png=out_png, image_index=image_index)
            except Exception as exc:
                rec = {"png": "", "status": "ERROR", "error": str(exc), "backend": backend, "case_id": case, "task": _report_task(data), "variant": _choose_variant(data), "image_name": "", "image_found": False, "prediction_available": False}
            rec.update({"model_id": model, "report_path": relpath(rp, run_dir)})
            rows.append(rec)
            count += 1
    for r in rows:
        link = f"images/{r.get('png')}" if r.get("png") else ""
        md.append(f"## {r.get('model_id')} · {r.get('case_id')} · {r.get('backend')} · {r.get('variant')} · {r.get('status')}")
        md.append("")
        md.append(f"- task: `{r.get('task')}`")
        md.append(f"- source image: `{r.get('image_name') or '-'}` (found locally: `{r.get('image_found')}`, dataset fallback: `{r.get('dataset_fallback_used')}`, per-image prediction: `{r.get('prediction_available')}`)")
        md.append(f"- visual status: `{r.get('visual_status','')}` claimable=`{r.get('visual_claimable','')}`")
        if r.get("classification_top1") not in (None, "") or r.get("classification_top5") not in (None, ""):
            md.append(f"- classification aggregate: top1=`{r.get('classification_top1')}` top5=`{r.get('classification_top5')}`")
        md.append(f"- report: `{r.get('report_path')}`")
        if link:
            md.append(f"\n![{r.get('model_id')} {r.get('case_id')} {r.get('backend')}]({link})")
        if r.get("error"):
            md.append(f"\nGeneration error: `{r.get('error')}`")
        md.append("")
    p_csv = write_csv(vis_dir / "visual_verification.csv", rows, ["model_id", "case_id", "backend", "variant", "task", "status", "visual_status", "visual_claimable", "image_name", "image_found", "dataset_fallback_used", "prediction_available", "classification_top1", "classification_top5", "png", "report_path", "error"])
    p_md = write_text(vis_dir / "visual_verification.md", "\n".join(md))
    return {"visual_verification_md": p_md, "visual_verification_csv": p_csv, "visual_verification_count": len(rows)}



def _slug_for_artifact(value: Any) -> str:
    s = str(value or "").strip()
    if not s:
        return "item"
    out = []
    for ch in s:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_") or "item"




def _visual_validation_counts(visual_csv: Path) -> Dict[str, Dict[str, int]]:
    """Summarise visual verification PASS/FAIL rows by model.

    v58d: visual verification is generated from the same dataset/provenance
    used for semantic validation.  It must not silently disagree with
    validation_summary.csv; if visual rows show FAILs, central validation reports
    should expose them as visual_invalid rows.
    """
    counts: Dict[str, Dict[str, int]] = {}
    try:
        rows = _read_csv(visual_csv)
    except Exception:
        rows = []
    for row in rows:
        mid = str(row.get("model_id") or "").strip()
        if not mid:
            continue
        c = counts.setdefault(mid, {"visual_total": 0, "visual_valid": 0, "visual_invalid": 0, "visual_unvalidated": 0})
        c["visual_total"] += 1
        status = str(row.get("status") or "").strip().lower()
        if status in {"pass", "passed", "ok", "success", "true"}:
            c["visual_valid"] += 1
        elif status in {"fail", "failed", "error", "false"}:
            c["visual_invalid"] += 1
        else:
            c["visual_unvalidated"] += 1
    return counts


def _merge_visual_counts_into_validation_rows(val_rows: List[Dict[str, Any]], visual_counts: Mapping[str, Mapping[str, int]]) -> List[Dict[str, Any]]:
    """Attach visual-verification counts without double-counting validation rows.

    v58n: visual verification is a sampled sanity gallery, not an additional
    validation dataset.  Earlier code added PASS/FAIL visual rows to
    valid_result_count/invalid_result_count, which could produce impossible
    counters such as valid+invalid > result_count.  Keep central validation
    counts untouched and expose visual disagreements as separate conflict fields.
    """
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in val_rows:
        row = dict(raw or {})
        mid = str(row.get("model_id") or "").strip()
        if mid:
            seen.add(mid)
        vc = visual_counts.get(mid, {}) if mid else {}
        if vc:
            row["visual_verification_total"] = int(vc.get("visual_total") or 0)
            row["visual_verification_valid"] = int(vc.get("visual_valid") or 0)
            row["visual_verification_invalid"] = int(vc.get("visual_invalid") or 0)
            row["visual_verification_unvalidated"] = int(vc.get("visual_unvalidated") or 0)
            old_invalid = int(_float(row.get("invalid_result_count")) or 0)
            old_valid = int(_float(row.get("valid_result_count")) or 0)
            old_unval = int(_float(row.get("unvalidated_result_count")) or 0)
            vinv = int(vc.get("visual_invalid") or 0)
            # Reconciled view keeps validation counts unchanged.  Visual samples
            # can mark a conflict, but they are not extra result rows.
            row["invalid_result_count_with_visual"] = old_invalid
            row["valid_result_count_with_visual"] = old_valid
            row["unvalidated_result_count_with_visual"] = old_unval
            row["visual_conflict_count"] = vinv
            row["visual_validation_conflict_count"] = vinv
            row["visual_validation_ok"] = "False" if vinv > 0 else "True"
            if vinv > 0:
                row["validation_visual_consistency"] = "visual_failures_present"
                row["validation_ok_with_visual"] = "False"
                if str(row.get("validation_status") or "") in {"validated", "valid_complete_splits_present", "ok"}:
                    row["validation_status_with_visual"] = "valid_complete_splits_with_visual_failures"
            else:
                row.setdefault("validation_visual_consistency", "visual_rows_no_failures")
                _vw = row.get("validation_ok_with_visual")
                if _vw in (None, "") or str(_vw).strip().lower() in {"nan", "none", "null"}:
                    row["validation_ok_with_visual"] = row.get("validation_ok", "")
        out.append(row)
    for mid, vc in visual_counts.items():
        if mid in seen:
            continue
        out.append({
            "model_id": mid,
            "validation_status": "visual_verification_only",
            "validation_ok": "False" if int(vc.get("visual_invalid") or 0) else "",
            "validation_ok_with_visual": "False" if int(vc.get("visual_invalid") or 0) else "",
            "visual_validation_ok": "False" if int(vc.get("visual_invalid") or 0) else "True",
            "visual_verification_total": int(vc.get("visual_total") or 0),
            "visual_verification_valid": int(vc.get("visual_valid") or 0),
            "visual_verification_invalid": int(vc.get("visual_invalid") or 0),
            "visual_verification_unvalidated": int(vc.get("visual_unvalidated") or 0),
            "visual_conflict_count": int(vc.get("visual_invalid") or 0),
            "visual_validation_conflict_count": int(vc.get("visual_invalid") or 0),
        })
    return out


def _collect_visual_verification(run_dir: Path, reports: Path, *, max_per_model: int = 80) -> Dict[str, Any]:
    """Collect one-image visual sanity artifacts into reports/visual_verification.

    Per-case runners write classification_*.png / detections_*.png when they are
    given an image.  This collector copies the small review artifacts into a
    single gallery so the debug pack has quick human-verification evidence next
    to the CSV/JSON metrics.
    """
    out_root = reports / "visual_verification"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    model_root = run_dir / "models"
    if not model_root.is_dir():
        write_text(out_root / "VISUAL_VERIFICATION_NOT_GENERATED.md", "No model directories found.\n")
        return {"count": 0, "manifest": str(out_root / "manifest.json")}

    image_patterns = [
        "classification_composed.png", "detections_composed.png",
        "classification_full.png", "detections_full.png",
        "classification_part2.png", "detections_part2.png",
        "input_image.png",
    ]
    json_patterns = [
        "classification_composed.json", "detections_composed.json",
        "classification_full.json", "detections_full.json",
        "classification_part2.json", "detections_part2.json",
        "validation_report.json",
    ]

    for model_dir in sorted(p for p in model_root.iterdir() if p.is_dir()):
        model_id = model_dir.name
        copied_for_model = 0
        # Prefer compact lean bundles, then fuller case_reports, then any local benchmark_results tree.
        roots = [
            model_dir / "benchmark_results" / "remote_diagnostics" / "lean_bundle",
            model_dir / "benchmark_results" / "remote_diagnostics" / "case_reports" / "results",
            model_dir / "benchmark_results",
        ]
        seen_keys = set()
        for base in roots:
            if copied_for_model >= max_per_model:
                break
            if not base.is_dir():
                continue
            for result_dir in sorted(base.rglob("results_*")):
                if copied_for_model >= max_per_model:
                    break
                if not result_dir.is_dir():
                    continue
                rel = result_dir.relative_to(base)
                parts = rel.parts
                case_id = parts[-2] if len(parts) >= 2 and parts[-2].startswith("b") else "case"
                run_id = result_dir.name.replace("results_", "")
                key = (model_id, case_id, run_id)
                if key in seen_keys:
                    continue
                png_src = None
                for name in image_patterns:
                    cand = result_dir / name
                    if cand.is_file():
                        png_src = cand
                        break
                if png_src is None:
                    continue
                seen_keys.add(key)
                dst_dir = out_root / _slug_for_artifact(model_id)
                dst_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{_slug_for_artifact(model_id)}__{_slug_for_artifact(case_id)}__{_slug_for_artifact(run_id)}"
                dst_png = dst_dir / f"{stem}{png_src.suffix.lower()}"
                try:
                    shutil.copy2(png_src, dst_png)
                except Exception:
                    continue
                copied_jsons: Dict[str, str] = {}
                for name in json_patterns:
                    js = result_dir / name
                    if js.is_file():
                        dst_js = dst_dir / f"{stem}__{_slug_for_artifact(js.stem)}.json"
                        try:
                            shutil.copy2(js, dst_js)
                            copied_jsons[js.name] = relpath(dst_js, reports)
                        except Exception:
                            pass
                rows.append({
                    "model_id": model_id,
                    "case_id": case_id,
                    "run_id": run_id,
                    "image": relpath(dst_png, reports),
                    "source": relpath(png_src, run_dir),
                    "json_artifacts": copied_jsons,
                })
                copied_for_model += 1

    manifest = {
        "schema": "onnx-splitpoint/visual-verification-gallery",
        "schema_version": 1,
        "created_at": now_iso(),
        "count": len(rows),
        "rows": rows,
        "note": "One-image per case/run visual sanity artifacts copied from per-case runner outputs. Use metrics for final decisions; these images are fast human checks.",
    }
    write_json(out_root / "manifest.json", manifest)
    md = ["# Visual verification gallery", "", "These images are quick human sanity checks. They complement, not replace, dataset metrics.", ""]
    if rows:
        md.append("| Model | Case | Run | Image |")
        md.append("|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['model_id']} | {r['case_id']} | {r['run_id']} | ![]({r['image']}) |")
    else:
        md.append("No visual verification images were found. Ensure per-case runners receive a validation image and that Pillow is available in the runtime environment.")
    write_text(out_root / "visual_verification.md", "\n".join(md) + "\n")
    return {"count": len(rows), "manifest": str(out_root / "manifest.json"), "gallery_md": str(out_root / "visual_verification.md")}

def build_result_dashboard_reports(
    run_dir: Path,
    *,
    profile_id: str,
    tool_version: str,
    workflow_version: str,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    tables = reports / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    summary_rows = _read_csv(reports / "summary.csv")
    pred_rows = _read_csv(reports / "prediction_vs_benchmark.csv")
    hw_rows = _read_csv(reports / "hardware_smoke_summary.csv")
    val_rows = _read_csv(reports / "validation_summary.csv")
    hw_summary_rows = _read_csv(reports / "hardware_summary.csv")
    run_status = read_json(reports / "run_status_summary.json", default={}) or {}
    job_summary = read_json(run_dir / "jobs" / "job_summary.json", default={}) or {}
    native_evidence = read_json(
        reports / "native_evidence_status.json", default={},
    ) or {}
    if not isinstance(native_evidence, Mapping) or not native_evidence:
        embedded_evidence = run_status.get("native_evidence_status")
        native_evidence = (
            dict(embedded_evidence)
            if isinstance(embedded_evidence, Mapping) else {}
        )
    else:
        native_evidence = dict(native_evidence)
    native_evidence_summary = project_native_evidence_status(
        native_evidence
    )

    by_summary = {str(r.get("model_id") or ""): r for r in summary_rows}
    by_hw = {str(r.get("model_id") or ""): r for r in hw_rows}
    # v58d: generate/copy visual verification before model-card aggregation so
    # visual FAILs can be reflected in validation summaries instead of being a
    # disconnected gallery artifact.
    visual_artifacts = _write_visual_verification_report(run_dir, reports)
    visual_counts = _visual_validation_counts(Path(visual_artifacts.get("visual_verification_csv") or reports / "visual_verification" / "visual_verification.csv"))
    val_rows = _merge_visual_counts_into_validation_rows(val_rows, visual_counts)
    if val_rows:
        # Keep extra visual_* columns; write_csv with no explicit fields infers
        # a superset from all rows.
        write_csv(reports / "validation_summary.csv", val_rows)
    by_val = {str(r.get("model_id") or ""): r for r in val_rows}
    # Also reconcile the compact summary/model_summary CSVs so a reader does not
    # see validation_ok=True while visual verification contains FAIL rows.
    if visual_counts and summary_rows:
        for sr in summary_rows:
            mid = str(sr.get("model_id") or "")
            vc = visual_counts.get(mid, {})
            if not vc:
                continue
            sr["visual_verification_total"] = int(vc.get("visual_total") or 0)
            sr["visual_verification_invalid"] = int(vc.get("visual_invalid") or 0)
            sr["visual_validation_ok"] = "False" if int(vc.get("visual_invalid") or 0) > 0 else "True"
            if int(vc.get("visual_invalid") or 0) > 0:
                sr["validation_status"] = "valid_complete_splits_with_visual_failures"
                sr["validation_ok"] = "False"
        write_csv(reports / "summary.csv", summary_rows)
        write_csv(reports / "model_summary.csv", summary_rows)
    all_model_ids = sorted({*by_summary.keys(), *by_hw.keys(), *by_val.keys()} - {""})
    # v55i: if CSV aggregation is empty or stale, still show the model folders
    # in the dashboard.  Older runs produced the confusing state
    # models=0 while complete_split_models>0.
    if not all_model_ids:
        models_root = run_dir / "models"
        if models_root.is_dir():
            all_model_ids = sorted([p.name for p in models_root.iterdir() if p.is_dir()])

    blocking = [dict(r) for r in list(run_status.get("blocking_reasons") or []) if isinstance(r, Mapping)]
    non_blocking = [dict(r) for r in list(run_status.get("non_blocking_reasons") or []) if isinstance(r, Mapping)]
    reasons_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for reason in blocking + non_blocking:
        mid = str(reason.get("model_id") or "workflow")
        reasons_by_model.setdefault(mid, []).append(reason)

    model_cards: List[Dict[str, Any]] = []
    thesis_rows: List[Dict[str, Any]] = []
    backend_speedup_rows: List[Dict[str, Any]] = []
    all_normalized_rows: List[Dict[str, Any]] = []

    for model_id in all_model_ids:
        srow = by_summary.get(model_id, {})
        hrow = by_hw.get(model_id, {})
        vrow = by_val.get(model_id, {})
        model_dir = run_dir / "models" / model_id
        benchmark_contract = read_json(
            model_dir / "benchmark_results" / "normalized_results.json",
            default={},
        ) or {}
        rows = [
            dict(row) for row in list(
                benchmark_contract.get("results") or []
                if isinstance(benchmark_contract, Mapping) else []
            ) if isinstance(row, Mapping)
        ]
        benchmark_stage = read_json(
            model_dir / "stages" / "run_benchmarks" / "stage_result.json",
            default={},
        ) or {}
        validation_contract = read_json(
            model_dir / "validation" / "validation_summary.json",
            default={},
        ) or {}
        for _r in rows:
            _rr = dict(_r)
            _rr.setdefault("model_id", model_id)
            all_normalized_rows.append(_rr)
        pred_model_rows = [r for r in pred_rows if str(r.get("model_id") or "") == model_id]
        preflight = _load_remote_preflight(run_dir, model_id)
        pre_hailo = preflight.get("hailo") if isinstance(preflight.get("hailo"), Mapping) else {}
        hw_status = read_json(
            model_dir / "hardware" / "hardware_smoke_status.json",
            default={},
        ) or {}
        hw_hailo = hw_status.get("hailo") if isinstance(hw_status.get("hailo"), Mapping) else {}

        complete_split_rows: List[Dict[str, Any]] = []
        valid_complete_split_rows: List[Dict[str, Any]] = []
        contract_rejected_rows: List[Dict[str, Any]] = []
        component_only_rows: List[Dict[str, Any]] = []
        full_by_backend: Dict[str, float] = {}
        split_by_backend: Dict[str, float] = {}
        for row in rows:
            backend = str(row.get("backend") or "unknown")
            if str(row.get("variant") or "").lower() == "full":
                flat = _full_latency(row)
                full_backend = _full_baseline_backend(row)
                if flat is not None and full_backend:
                    old = full_by_backend.get(full_backend)
                    if old is None or flat < old:
                        full_by_backend[full_backend] = flat
            slat = _complete_split_latency(row)
            if str(row.get("variant") or "").lower() == "split":
                if slat is not None:
                    r = dict(row)
                    r["effective_split_latency_ms"] = slat
                    r["effective_pipeline_cycle_ms"] = _pipeline_cycle(r)
                    r["effective_pipeline_fps"] = _pipeline_fps(r)
                    r["quality_label"] = _row_quality_label(r)
                    complete_split_rows.append(r)
                    if _stage2_contract_rejected(r):
                        contract_rejected_rows.append(r)
                    if _row_valid_complete_split(r):
                        valid_complete_split_rows.append(r)
                        old = split_by_backend.get(backend)
                        if old is None or slat < old:
                            split_by_backend[backend] = slat
                elif str(row.get("component_measurement_status") or "").lower() in {"part1_only", "part2_only", "none"}:
                    rr = dict(row)
                    if _stage2_contract_rejected(rr):
                        contract_rejected_rows.append(rr)
                    component_only_rows.append(rr)

        # v55d: best split/speedups use valid complete split rows first.
        # If none are valid, show the fastest measured row as diagnostic only.
        split_selection_rows = valid_complete_split_rows or complete_split_rows
        best_split = _best_by_latency(split_selection_rows)
        hetero_split_selection_rows = [r for r in split_selection_rows if _row_is_heterogeneous_accelerator_split(r)]
        same_backend_split_selection_rows = [r for r in split_selection_rows if not _row_is_heterogeneous_accelerator_split(r)]
        best_hetero_latency_split = _best_by_latency(hetero_split_selection_rows)
        best_same_backend_diag_split = _best_by_latency(same_backend_split_selection_rows)
        # v55t: for the model-card "best pipeline split" prefer a genuine
        # cross-backend pipeline. Same-backend TensorRT/CUDA split rows remain
        # in the detailed throughput table, but they should not hide the
        # double-accelerator result the report is meant to highlight.
        best_pipeline_split = _best_by_pipeline_cycle(hetero_split_selection_rows or split_selection_rows)
        best_pipeline_any_split = _best_by_pipeline_cycle(split_selection_rows)
        best_split_latency = _float(best_split.get("effective_split_latency_ms")) if best_split else None
        best_hetero_latency = _float(best_hetero_latency_split.get("effective_split_latency_ms")) if best_hetero_latency_split else None
        best_same_backend_diag_latency = _float(best_same_backend_diag_split.get("effective_split_latency_ms")) if best_same_backend_diag_split else None
        best_pipeline_cycle = _pipeline_cycle(best_pipeline_split) if best_pipeline_split else None
        best_pipeline_fps = _pipeline_fps(best_pipeline_split) if best_pipeline_split else None
        best_pipeline_any_cycle = _pipeline_cycle(best_pipeline_any_split) if best_pipeline_any_split else None
        best_pipeline_any_fps = _pipeline_fps(best_pipeline_any_split) if best_pipeline_any_split else None
        best_full_backend, best_full_latency = ("", None)
        full_baseline_stats: Dict[str, Dict[str, Any]] = {}
        full_latency_samples: Dict[str, List[float]] = {}
        for rr in rows:
            fb = _full_baseline_backend(rr)
            fl = _full_latency(rr)
            if fb and fl is not None:
                full_latency_samples.setdefault(fb, []).append(float(fl))
        for fb, xs in full_latency_samples.items():
            if not xs:
                continue
            mn, mx = min(xs), max(xs)
            med = _median(xs)
            spread_pct = ((mx - mn) / mn * 100.0) if mn > 0 else None
            full_baseline_stats[fb] = {
                "backend": fb,
                "count": len(xs),
                "canonical_policy": "min_valid_full_latency",
                "canonical_latency_ms": mn,
                "median_latency_ms": med,
                "max_latency_ms": mx,
                "spread_pct": spread_pct,
                "ambiguous": bool(spread_pct is not None and spread_pct > 20.0 and len(xs) > 1),
            }
        if full_by_backend:
            best_full_backend, best_full_latency = min(full_by_backend.items(), key=lambda kv: kv[1])
        cpu_full = full_by_backend.get("cpu_ort")
        tensorrt_full = full_by_backend.get("tensorrt")
        tensorrt_full_median = _float(full_baseline_stats.get("tensorrt", {}).get("median_latency_ms")) if isinstance(full_baseline_stats.get("tensorrt"), Mapping) else None
        speedup_vs_cpu = (cpu_full / best_split_latency) if cpu_full is not None and best_split_latency not in (None, 0) else None
        speedup_vs_trt = (tensorrt_full / best_split_latency) if tensorrt_full is not None and best_split_latency not in (None, 0) else None
        pipeline_speedup_vs_cpu = _throughput_speedup(cpu_full, best_pipeline_cycle)
        pipeline_speedup_vs_trt = _throughput_speedup(tensorrt_full, best_pipeline_cycle)

        measured_pred_rows = [r for r in pred_model_rows if str(r.get("status") or "") == "measured"]
        regrets = [_float(r.get("regret_pct")) for r in measured_pred_rows]
        regrets = [r for r in regrets if r is not None]
        speedup_errors: List[float] = []
        for r in measured_pred_rows:
            pred = _float(r.get("predicted_total_latency_ms"))
            meas = _float(r.get("measured_total_latency_ms"))
            if pred is not None and meas not in (None, 0):
                speedup_errors.append(abs(pred - meas) / float(meas) * 100.0)
        pred_best = str(srow.get("predicted_best_case") or "")
        meas_best = str(srow.get("measured_best_case") or "")
        top1 = _bool(srow.get("top1_hit"))
        topk = _bool(srow.get("topk_hit"))
        rank_corr = _float(srow.get("rank_correlation_spearman"))
        accepted_outside_prediction = len([r for r in pred_model_rows if _bool(r.get("accepted_by_generator")) is True and "accepted_by_generator" == str(r.get("selection_scope") or "")])
        predicted_unmeasured = len([r for r in pred_model_rows if str(r.get("selection_scope") or "") == "prediction_selected" and str(r.get("status") or "") != "measured"])

        # Feasibility accuracy only when both prediction and compile/runtime labels are present.
        feasibility_pairs = []
        for r in pred_model_rows:
            pred_feas = _bool(r.get("predicted_hailo_feasible"))
            if pred_feas is None:
                continue
            measured_ok = _bool(r.get("compile_ok"))
            runtime_ok = _bool(r.get("runtime_ok"))
            if measured_ok is None and runtime_ok is None:
                continue
            actual = bool(measured_ok if measured_ok is not None else runtime_ok)
            feasibility_pairs.append(pred_feas == actual)
        feasibility_accuracy = (sum(1 for x in feasibility_pairs if x) / len(feasibility_pairs)) if feasibility_pairs else None

        health = _model_health(
            status_row=srow,
            validation_row=vrow,
            hardware_row=hrow,
            model_reasons=reasons_by_model.get(model_id, []),
            benchmark_contract=(
                benchmark_contract
                if isinstance(benchmark_contract, Mapping) else {}
            ),
            benchmark_stage=(
                benchmark_stage
                if isinstance(benchmark_stage, Mapping) else {}
            ),
            validation_contract=(
                validation_contract
                if isinstance(validation_contract, Mapping) else {}
            ),
            hardware_contract=(
                hw_status if isinstance(hw_status, Mapping) else {}
            ),
        )
        hailo_device_present = bool(pre_hailo.get("dev_nodes")) or _bool(hw_hailo.get("target_present")) is True
        hailo_service_running = bool(str(pre_hailo.get("ps_hailo") or "").strip())
        hailo_import_ok = _bool(pre_hailo.get("hailo_import_ok"))
        hailo_vdevice_ok = _bool(pre_hailo.get("vdevice_ok"))
        hailo_full_rows = [r for r in rows if _row_is_hailo_full(r) and _full_latency(r) is not None]
        hailo_split_rows = [r for r in rows if _row_is_hailo_complete_split(r)]
        best_hailo_full_latency = _safe_min(hailo_full_rows, "total_latency_ms") or _safe_min(hailo_full_rows, "full_latency_ms")
        best_hailo_split_latency = _safe_min(hailo_split_rows, "total_latency_ms")
        hailo_runtime_verified = _bool(hrow.get("hailo_runtime_verified"))
        hailo_full_runtime_verified = _bool(hrow.get("hailo_full_runtime_verified"))
        hailo_composed_runtime_verified = _bool(hrow.get("hailo_composed_runtime_verified"))
        if hailo_full_runtime_verified is None:
            hailo_full_runtime_verified = bool(hailo_full_rows)
        hailo_split_runtime_verified = _bool(hrow.get("hailo_split_runtime_verified", hrow.get("hailo_composed_runtime_verified")))
        if hailo_composed_runtime_verified is None:
            hailo_composed_runtime_verified = hailo_split_runtime_verified
        if hailo_split_runtime_verified is None:
            hailo_split_runtime_verified = bool(hailo_split_rows)
        hailo_runtime_verified = bool(hailo_runtime_verified or hailo_full_runtime_verified or hailo_split_runtime_verified)
        card = {
            "model_id": model_id,
            "health": health,
            "task": srow.get("task", vrow.get("task", "")),
            "accepted_case_count": int(_float(srow.get("accepted_case_count")) or 0),
            "measured_result_count": int(_float(srow.get("measured_result_count")) or 0),
            "complete_split_count": len(complete_split_rows),
            "valid_complete_split_count": len(valid_complete_split_rows),
            "contract_rejected_split_count": len(contract_rejected_rows),
            "component_only_split_count": len(component_only_rows),
            "split_measurement_status": srow.get("split_measurement_status", ""),
            "best_complete_split_case": best_split.get("case_id", "") if best_split else "",
            "best_complete_split_backend": best_split.get("backend", "") if best_split else "",
            "best_complete_split_quality": _row_quality_label(best_split) if best_split else "",
            "best_complete_split_contract_status": _deepx_stage2_contract_status(best_split) if best_split else "",
            "best_complete_split_latency_ms": _fmt(best_split_latency),
            "best_heterogeneous_split_case": best_hetero_latency_split.get("case_id", "") if best_hetero_latency_split else "",
            "best_heterogeneous_split_backend": best_hetero_latency_split.get("backend", "") if best_hetero_latency_split else "",
            "best_heterogeneous_split_latency_ms": _fmt(best_hetero_latency),
            "best_same_backend_diagnostic_case": best_same_backend_diag_split.get("case_id", "") if best_same_backend_diag_split else "",
            "best_same_backend_diagnostic_backend": best_same_backend_diag_split.get("backend", "") if best_same_backend_diag_split else "",
            "best_same_backend_diagnostic_latency_ms": _fmt(best_same_backend_diag_latency),
            "best_pipeline_split_case": best_pipeline_split.get("case_id", "") if best_pipeline_split else "",
            "best_pipeline_split_backend": best_pipeline_split.get("backend", "") if best_pipeline_split else "",
            "best_pipeline_split_quality": _row_quality_label(best_pipeline_split) if best_pipeline_split else "",
            "best_pipeline_cycle_ms": _fmt(best_pipeline_cycle),
            "best_pipeline_fps": _fmt(best_pipeline_fps),
            "best_pipeline_split_is_heterogeneous": bool(best_pipeline_split and _row_is_heterogeneous_accelerator_split(best_pipeline_split)),
            "best_pipeline_any_case": best_pipeline_any_split.get("case_id", "") if best_pipeline_any_split else "",
            "best_pipeline_any_backend": best_pipeline_any_split.get("backend", "") if best_pipeline_any_split else "",
            "best_pipeline_any_cycle_ms": _fmt(best_pipeline_any_cycle),
            "best_pipeline_any_fps": _fmt(best_pipeline_any_fps),
            "pipeline_speedup_vs_cpu_full": _fmt(pipeline_speedup_vs_cpu),
            "pipeline_speedup_vs_tensorrt_full": _fmt(pipeline_speedup_vs_trt),
            "split_latency_note": "Latency is single-frame time; pipeline cycle/FPS is steady-state interleaved throughput.",
            "best_full_backend": best_full_backend,
            "best_full_latency_ms": _fmt(best_full_latency),
            "cpu_full_latency_ms": _fmt(cpu_full),
            "tensorrt_full_latency_ms": _fmt(tensorrt_full),
            "canonical_full_baselines": full_baseline_stats,
            "tensorrt_full_baseline_ambiguous": bool(full_baseline_stats.get("tensorrt", {}).get("ambiguous")),
            "speedup_vs_cpu_full": _fmt(speedup_vs_cpu),
            "speedup_vs_tensorrt_full": _fmt(speedup_vs_trt),
            "validation_status": vrow.get("validation_status", srow.get("validation_status", "")),
            "validation_ok": (
                _bool(vrow.get("validation_ok_with_visual"))
                if _bool(vrow.get("validation_ok_with_visual")) is not None
                else (_bool(vrow.get("validation_ok")) if _bool(vrow.get("validation_ok")) is not None else _bool(srow.get("validation_ok")))
            ),
            "hardware_smoke_status": hrow.get("hardware_smoke_status", srow.get("hardware_smoke_status", "")),
            "hardware_verified": _bool(hrow.get("hardware_verified", srow.get("hardware_verified"))),
            "hailo_evidence": {
                "device_present": hailo_device_present,
                "service_running": hailo_service_running,
                "python_import_ok": hailo_import_ok,
                "vdevice_probe_ok": hailo_vdevice_ok,
                "hef_detected_count": int(_float(hrow.get("hailo_detected_hef_count", hw_hailo.get("detected_hef_count"))) or 0),
                "pending_case_hefs": int(_float(hrow.get("hailo_pending_case_hefs", hw_hailo.get("pending_case_hefs"))) or 0),
                "runtime_verified": hailo_runtime_verified,
                "full_runtime_verified": hailo_full_runtime_verified,
                "split_composed_runtime_verified": hailo_split_runtime_verified,
                "composed_runtime_verified": hailo_composed_runtime_verified,
                "full_runtime_result_count": int(_float(hrow.get("hailo_full_runtime_ok_result_count", hw_hailo.get("full_runtime_ok_result_count"))) or len(hailo_full_rows)),
                "split_runtime_result_count": int(_float(hrow.get("hailo_split_runtime_ok_result_count", hrow.get("hailo_composed_runtime_ok_result_count", hw_hailo.get("split_runtime_ok_result_count", hw_hailo.get("composed_runtime_ok_result_count"))))) or len(hailo_split_rows)),
                "best_full_latency_ms": _fmt(best_hailo_full_latency),
                "best_split_latency_ms": _fmt(best_hailo_split_latency),
                "component_only_result_count": int(_float(hw_status.get("hailo_component_only_result_count", hw_hailo.get("component_only_result_count"))) or 0),
            },
            "hailo_runtime_verified": hailo_runtime_verified,
            "hailo_full_runtime_verified": hailo_full_runtime_verified,
            "hailo_split_runtime_verified": hailo_split_runtime_verified,
            "best_hailo_full_latency_ms": _fmt(best_hailo_full_latency),
            "best_hailo_split_latency_ms": _fmt(best_hailo_split_latency),
            "remote_status": hrow.get("remote_status", ""),
            "warnings": [r for r in reasons_by_model.get(model_id, []) if not bool(r.get("blocking"))],
            "blocking_reasons": [r for r in reasons_by_model.get(model_id, []) if bool(r.get("blocking"))],
        }
        model_cards.append(card)

        thesis_rows.append({
            "model_id": model_id,
            "prediction_scope": "strict_predicted_topk_and_legacy_accepted",
            "predicted_best_case": pred_best,
            "measured_best_case": meas_best,
            "top1_hit": "" if top1 is None else bool(top1),
            "topk_hit": "" if topk is None else bool(topk),
            "measured_candidate_count": len(measured_pred_rows),
            "accepted_case_count": card["accepted_case_count"],
            "accepted_outside_prediction_count": accepted_outside_prediction,
            "predicted_unmeasured_count": predicted_unmeasured,
            "best_regret_pct": _fmt(min(regrets) if regrets else None),
            "mean_regret_pct": _fmt(_mean(regrets)),
            "rank_correlation_spearman": _fmt(rank_corr),
            "feasibility_accuracy": _fmt(feasibility_accuracy),
            "speedup_error_mean_pct": _fmt(_mean(speedup_errors)),
            "best_complete_split_latency_ms": card["best_complete_split_latency_ms"],
            "speedup_vs_cpu_full": card["speedup_vs_cpu_full"],
            "speedup_vs_tensorrt_full": card["speedup_vs_tensorrt_full"],
            "best_pipeline_cycle_ms": card["best_pipeline_cycle_ms"],
            "best_pipeline_fps": card["best_pipeline_fps"],
            "pipeline_speedup_vs_tensorrt_full": card["pipeline_speedup_vs_tensorrt_full"],
            "validation_ok": card["validation_ok"],
            "hailo_runtime_verified": card["hailo_evidence"]["runtime_verified"],
            "hailo_full_runtime_verified": card["hailo_evidence"]["full_runtime_verified"],
            "hailo_composed_runtime_verified": card["hailo_evidence"]["composed_runtime_verified"],
        })

        best_valid_split_by_backend: Dict[str, Dict[str, Any]] = {}
        best_pipeline_by_backend: Dict[str, Dict[str, Any]] = {}
        diagnostic_split_by_backend: Dict[str, Dict[str, Any]] = {}
        for r in valid_complete_split_rows:
            b = str(r.get("backend") or "")
            if not b:
                continue
            if _row_is_heterogeneous_accelerator_split(r):
                oldr = best_valid_split_by_backend.get(b)
                if oldr is None or float(r.get("effective_split_latency_ms") or 1e18) < float(oldr.get("effective_split_latency_ms") or 1e18):
                    best_valid_split_by_backend[b] = r
                pc = _pipeline_cycle(r)
                oldp = best_pipeline_by_backend.get(b)
                oldpc = _pipeline_cycle(oldp) if oldp else None
                if pc is not None and (oldp is None or oldpc is None or float(pc) < float(oldpc)):
                    best_pipeline_by_backend[b] = r
            else:
                oldd = diagnostic_split_by_backend.get(b)
                if oldd is None or float(r.get("effective_split_latency_ms") or 1e18) < float(oldd.get("effective_split_latency_ms") or 1e18):
                    diagnostic_split_by_backend[b] = r
        # v58n: backend_speedups is now a paper-facing heterogeneous-split
        # table. Same-backend split rows remain available in
        # same_backend_split_diagnostics.csv and should not appear as measured
        # split gains against TensorRT/CPU full.
        for backend in sorted(set(full_by_backend) | set(best_valid_split_by_backend)):
            full_lat = full_by_backend.get(backend)
            best_backend_row = best_valid_split_by_backend.get(backend, {})
            split_lat = _float(best_backend_row.get("effective_split_latency_ms")) if best_backend_row else None
            best_pipeline_backend_row = best_pipeline_by_backend.get(backend, best_backend_row)
            backend_speedup_rows.append({
                "model_id": model_id,
                "backend": backend,
                "backend_label": _backend_display(backend),
                "best_full_latency_ms": _fmt(full_lat),
                "best_complete_split_latency_ms": _fmt(split_lat),
                "best_pipeline_cycle_ms": _fmt(_pipeline_cycle(best_pipeline_backend_row) if best_pipeline_backend_row else None),
                "best_pipeline_fps": _fmt(_pipeline_fps(best_pipeline_backend_row) if best_pipeline_backend_row else None),
                "pipeline_speedup_vs_tensorrt_full": _fmt(_throughput_speedup(tensorrt_full, _pipeline_cycle(best_pipeline_backend_row) if best_pipeline_backend_row else None)),
                "pipeline_speedup_vs_tensorrt_full_median": _fmt(_throughput_speedup(tensorrt_full_median, _pipeline_cycle(best_pipeline_backend_row) if best_pipeline_backend_row else None)),
                "best_complete_split_case": best_backend_row.get("case_id", ""),
                "best_pipeline_split_case": best_pipeline_backend_row.get("case_id", "") if best_pipeline_backend_row else "",
                "best_complete_split_quality": _row_quality_label(best_backend_row) if best_backend_row else "",
                "best_pipeline_split_quality": _row_quality_label(best_pipeline_backend_row) if best_pipeline_backend_row else "",
                "stage2_contract_status": _deepx_stage2_contract_status(best_pipeline_backend_row or best_backend_row) if (best_pipeline_backend_row or best_backend_row) else "",
                "stage2_contract_probe_samples": (best_pipeline_backend_row or best_backend_row).get("deepx_stage2_contract_probe_samples", "") if (best_pipeline_backend_row or best_backend_row) else "",
                "stage2_contract_pass_ratio": (best_pipeline_backend_row or best_backend_row).get("deepx_stage2_contract_selected_pass_ratio", "") if (best_pipeline_backend_row or best_backend_row) else "",
                "stage2_calibration_source": (best_pipeline_backend_row or best_backend_row).get("stage2_calibration_source", "") if (best_pipeline_backend_row or best_backend_row) else "",
                "speedup_vs_cpu_full": _fmt((cpu_full / split_lat) if cpu_full is not None and split_lat not in (None, 0) else None),
                "speedup_vs_tensorrt_full": _fmt((tensorrt_full / split_lat) if tensorrt_full is not None and split_lat not in (None, 0) else None),
                "speedup_vs_tensorrt_full_median": _fmt((tensorrt_full_median / split_lat) if tensorrt_full_median is not None and split_lat not in (None, 0) else None),
                "throughput_kind": "heterogeneous_pipeline" if best_backend_row else "full_backend_only",
                "same_backend_diagnostic_case": diagnostic_split_by_backend.get(backend, {}).get("case_id", "") if backend in diagnostic_split_by_backend else "",
                "complete_split_measured": split_lat is not None and bool(best_backend_row),
            })

    source_run_status = str(run_status.get("status") or "unknown")
    reported_run_status = source_run_status
    if (
        native_evidence_summary.get("energy_zero_start_blocked") is True
        and source_run_status.strip().lower()
        in {"ok", "complete", "completed", "success", "passed"}
    ):
        # A stale parent-stage completion token cannot override objective
        # evidence that no planned Native measurement ever started.
        reported_run_status = "failed"

    overview = {
        "schema": "onnx-splitpoint/evaluation-result-dashboard",
        "schema_version": 1,
        "created_at": now_iso(),
        "run_id": run_dir.name,
        "profile_id": profile_id,
        "tool_version": tool_version,
        "workflow_version": workflow_version,
        "run_status": reported_run_status,
        "source_run_status": source_run_status,
        "model_count": len(model_cards),
        "model_health_counts": {status: sum(1 for m in model_cards if m.get("health") == status) for status in ["ok", "warn", "partial", "failed"]},
        "accepted_case_total": sum(int(m.get("accepted_case_count") or 0) for m in model_cards),
        "complete_split_total": sum(int(m.get("complete_split_count") or 0) for m in model_cards),
        "component_only_split_total": sum(int(m.get("component_only_split_count") or 0) for m in model_cards),
        "blocking_reason_count": int(run_status.get("blocking_reason_count") or 0),
        "non_blocking_reason_count": int(run_status.get("non_blocking_reason_count") or 0),
        "job_count": int(job_summary.get("job_count") or job_summary.get("total_jobs") or 0),
        "technical_status": native_evidence_summary["technical_status"],
        "technical_complete": native_evidence_summary[
            "technical_complete"
        ],
        "claim_decisions_complete": native_evidence_summary[
            "claim_decisions_complete"
        ],
        "scientific_status": native_evidence_summary[
            "scientific_status"
        ],
        "scientific_ready": native_evidence_summary["scientific_ready"],
        "energy_plan_denominator_count": native_evidence_summary[
            "energy_plan_denominator_count"
        ],
        "energy_matrix_denominator_count": native_evidence_summary[
            "energy_matrix_denominator_count"
        ],
        "energy_execution_status": native_evidence_summary[
            "energy_execution_status"
        ],
        "energy_measurement_started_count": native_evidence_summary[
            "energy_measurement_started_count"
        ],
        "energy_not_started_preflight_count": native_evidence_summary[
            "energy_not_started_preflight_count"
        ],
        "native_evidence_summary": native_evidence_summary,
    }

    canonical_full_rows: List[Dict[str, Any]] = []
    for card in model_cards:
        model_id = str(card.get("model_id") or "")
        baselines = card.get("canonical_full_baselines") if isinstance(card.get("canonical_full_baselines"), Mapping) else {}
        for backend, stats in baselines.items():
            if not isinstance(stats, Mapping):
                continue
            canonical_full_rows.append({
                "model_id": model_id,
                "backend": backend,
                "canonical_policy": stats.get("canonical_policy", "min_valid_full_latency"),
                "canonical_latency_ms": _fmt(stats.get("canonical_latency_ms")),
                "median_latency_ms": _fmt(stats.get("median_latency_ms")),
                "max_latency_ms": _fmt(stats.get("max_latency_ms")),
                "sample_count": stats.get("count", 0),
                "spread_pct": _fmt(stats.get("spread_pct")),
                "ambiguous": bool(stats.get("ambiguous")),
            })
    # v57m: produce small, semantically separated throughput tables.  A full
    # backend throughput value is not a heterogeneous split-pipeline result, and
    # same-backend composed rows are diagnostics rather than double-accelerator
    # evidence.
    full_backend_rows: List[Dict[str, Any]] = []
    hetero_pipeline_rows: List[Dict[str, Any]] = []
    same_backend_diag_rows: List[Dict[str, Any]] = []
    for row in all_normalized_rows:
        kind = _throughput_kind(row)
        model = str(row.get("model_id") or "")
        backend = str(row.get("backend") or "")
        case_id = str(row.get("case_id") or "")
        fp = _row_final_pass(row)
        quality = _row_quality_label(row)
        common = {
            "model_id": model,
            "backend": backend,
            "backend_label": _backend_display(backend),
            "case_id": case_id,
            "quality": quality,
            "final_pass": fp,
            "avg_power_w": _fmt(resolve_energy_comparison(row).get("comparison_average_power_w")),
            "raw_avg_power_w": _fmt(resolve_energy_comparison(row).get("raw_average_power_w")),
            "comparison_avg_power_w": _fmt(resolve_energy_comparison(row).get("comparison_average_power_w")),
            "energy_j_per_frame_command_window": _fmt(_row_energy_command_window_j_per_frame(row)),
            "energy_j_per_frame_from_selected_fps": _fmt(_row_energy_selected_fps_j_per_frame(row)),
            "energy_j_per_frame": _fmt(_row_energy_command_window_j_per_frame(row)),  # legacy/report compatibility
            "raw_energy_j_per_frame": _fmt(resolve_energy_comparison(row).get("raw_energy_per_work_j")),
            "host_normalized_energy_j_per_frame_est": _fmt(resolve_energy_comparison(row).get("host_normalized_energy_per_work_est_j")),
            "energy_comparison_basis": resolve_energy_comparison(row).get("energy_comparison_basis", ""),
            "energy_comparison_status": resolve_energy_comparison(row).get("energy_comparison_status", ""),
            "energy_comparison_claim_ready": resolve_energy_comparison(row).get("energy_comparison_claim_ready", False),
            "energy_measurement_scope": row.get("energy_measurement_scope", ""),
            "energy_row_level_source": row.get("energy_row_level_source", ""),
            "energy_coverage_status": row.get("energy_coverage_status", ""),
            "energy_target_status": row.get("energy_target_status", ""),
            "validation_claim_label": row.get("validation_claim_label", ""),
            "validation_claim_level": row.get("validation_claim_level", ""),
            "interface_contract_status": row.get("interface_contract_status", ""),
            "strict_boundary_numeric_status": row.get("strict_boundary_numeric_status", ""),
        }
        if kind == "full_backend" and str(row.get("variant") or "").lower() == "full" and _full_latency(row) is not None and fp is not False:
            full_backend_rows.append({
                **common,
                "throughput_kind": "full_backend",
                "latency_ms": _fmt(_full_latency(row)),
                "full_backend_throughput_fps": _fmt(_full_backend_fps(row)),
                "backend_tool_fps": _fmt(row.get("backend_tool_fps") or row.get("dxrt_tool_fps")),
            })
        elif kind == "heterogeneous_pipeline" and _row_valid_complete_split(row):
            hetero_pipeline_rows.append({
                **common,
                "throughput_kind": "heterogeneous_pipeline",
                "split_latency_ms": _fmt(_complete_split_latency(row)),
                "pipeline_cycle_ms": _fmt(_pipeline_cycle(row)),
                "pipeline_fps": _fmt(_heterogeneous_pipeline_fps(row)),
                "energy_fps_per_watt": _fmt(row.get("energy_streaming_fps_per_watt_from_selected_fps") or row.get("energy_streaming_frames_per_j")),
                "stage2_contract_status": _deepx_stage2_contract_status(row),
                "stage2_calibration_source": row.get("stage2_calibration_source", ""),
            })
        elif kind == "same_backend_split_diagnostic" and str(row.get("variant") or "").lower() == "split" and _complete_split_latency(row) is not None:
            same_backend_diag_rows.append({
                **common,
                "throughput_kind": "same_backend_split_diagnostic",
                "split_latency_ms": _fmt(_complete_split_latency(row)),
                "diagnostic_cycle_ms": _fmt(_pipeline_cycle(row)),
                "diagnostic_fps": _fmt(_same_backend_diagnostic_fps(row)),
            })

    # v58f: one canonical full-backend row per model/backend in paper-facing
    # throughput tables.  Companion full rows from multiple split cases are
    # useful internally but should not duplicate TensorRT/CPU/DeepX baselines.
    _fb_groups: Dict[tuple[str, str], list[Dict[str, Any]]] = {}
    for _r in full_backend_rows:
        _fb_groups.setdefault((str(_r.get("model_id") or ""), str(_r.get("backend") or "")), []).append(_r)
    _fb_out: List[Dict[str, Any]] = []
    for (_mid, _be), _rows in sorted(_fb_groups.items()):
        def _lat_key(_r: Mapping[str, Any]) -> float:
            _v = _float(_r.get("latency_ms"))
            return float(_v) if _v is not None else 1e18
        _sorted_rows = sorted(_rows, key=_lat_key)
        _best = dict(_sorted_rows[0])
        # v58n: canonical Full rows are selected by latency, but measured
        # u.RECS energy may live on another companion row (for example an
        # applies-to-all-cases canonical target).  Preserve the canonical
        # latency while copying energy context from the closest energy-bearing
        # row so full_backend_throughput.csv is not missing DeepX/TensorRT
        # energy while energy_summary.csv has it.
        _energy_rows = [r for r in _rows if _row_has_urecs_energy(r)]
        if _energy_rows and not _row_has_urecs_energy(_best):
            _same_case = [r for r in _energy_rows if str(r.get("case_id") or "") == str(_best.get("case_id") or "")]
            _copy_energy_context(_best, (_same_case or _energy_rows)[0])
        elif _energy_rows:
            # Fill any missing energy aliases from the first measured row.
            _copy_energy_context(_best, _energy_rows[0])
        # Keep the paper-facing row-level energy aliases in sync after copying
        # context from a companion energy row.
        if _float(_best.get("energy_j_per_frame_command_window")) is None:
            _ef = _row_energy_command_window_j_per_frame(_best)
            if _ef is not None:
                _best["energy_j_per_frame_command_window"] = _fmt(_ef)
                _best["energy_j_per_frame"] = _fmt(_ef)
        if _float(_best.get("energy_j_per_frame_from_selected_fps")) is None:
            _efs = _row_energy_selected_fps_j_per_frame(_best)
            if _efs is not None:
                _best["energy_j_per_frame_from_selected_fps"] = _fmt(_efs)
        if _float(_best.get("avg_power_w")) is None:
            _pw = _float(_best.get("energy_streaming_avg_power_w") or _best.get("energy_latency_avg_power_w") or _best.get("avg_power_w"))
            if _pw is not None:
                _best["avg_power_w"] = _fmt(_pw)
        _lats = [_float(_r.get("latency_ms")) for _r in _rows if _float(_r.get("latency_ms")) is not None]
        _best["canonical_case_id"] = _best.get("case_id", "")
        _best["sample_count"] = len(_rows)
        _best["canonical_full_policy"] = "min_valid_full_latency"
        if _lats:
            _best["min_latency_ms"] = _fmt(min(_lats))
            _best["max_latency_ms"] = _fmt(max(_lats))
            _best["median_latency_ms"] = _fmt(sorted(_lats)[len(_lats)//2])
            if min(_lats) > 0:
                _spread = 100.0 * (max(_lats) - min(_lats)) / min(_lats)
                _best["spread_pct"] = _fmt(_spread)
                _best["ambiguous"] = bool(_spread > 5.0)
        _fb_out.append(_best)
    full_backend_rows = sorted(_fb_out, key=lambda r: (_float(r.get("full_backend_throughput_fps")) or -1.0), reverse=True)
    hetero_pipeline_rows.sort(key=lambda r: (_float(r.get("pipeline_fps")) or -1.0), reverse=True)
    same_backend_diag_rows.sort(key=lambda r: (_float(r.get("diagnostic_fps")) or -1.0), reverse=True)

    p_full_throughput_csv = write_csv(reports / "full_backend_throughput.csv", full_backend_rows, [
        "model_id", "backend", "backend_label", "canonical_case_id", "throughput_kind",
        "latency_ms", "full_backend_throughput_fps", "backend_tool_fps",
        "sample_count", "min_latency_ms", "median_latency_ms", "max_latency_ms", "spread_pct", "ambiguous",
        "avg_power_w", "energy_j_per_frame_command_window", "energy_j_per_frame_from_selected_fps", "energy_j_per_frame", "energy_measurement_scope", "energy_row_level_source", "energy_coverage_status", "quality", "validation_claim_label", "final_pass",
    ])
    p_hetero_pipeline_csv = write_csv(reports / "heterogeneous_pipeline_throughput.csv", hetero_pipeline_rows, [
        "model_id", "backend", "backend_label", "case_id", "throughput_kind",
        "split_latency_ms", "pipeline_cycle_ms", "pipeline_fps",
        "avg_power_w", "energy_j_per_frame_command_window", "energy_j_per_frame_from_selected_fps", "energy_j_per_frame", "energy_fps_per_watt",
        "stage2_contract_status", "stage2_calibration_source", "interface_contract_status", "strict_boundary_numeric_status", "energy_measurement_scope", "energy_row_level_source", "energy_coverage_status", "quality", "validation_claim_label", "final_pass",
    ])
    p_same_backend_diag_csv = write_csv(reports / "same_backend_split_diagnostics.csv", same_backend_diag_rows, [
        "model_id", "backend", "backend_label", "case_id", "throughput_kind",
        "split_latency_ms", "diagnostic_cycle_ms", "diagnostic_fps",
        "avg_power_w", "energy_j_per_frame_command_window", "energy_j_per_frame_from_selected_fps", "energy_j_per_frame", "energy_measurement_scope", "energy_row_level_source", "energy_coverage_status", "quality", "validation_claim_label", "final_pass",
    ])

    # v59o: compact thesis/paper claim table.  This table is intentionally
    # case-level: it compares the best valid heterogeneous pipeline per model
    # against the TensorRT full baseline and keeps command-window energy and
    # selected-FPS energy separate.
    _full_by_model_backend: Dict[tuple[str, str], Mapping[str, Any]] = {
        (str(r.get("model_id") or ""), str(r.get("backend") or "").lower()): r
        for r in full_backend_rows
    }
    energy_claim_rows: List[Dict[str, Any]] = []
    for _mid in sorted({str(m.get("model_id") or "") for m in model_cards} | {str(r.get("model_id") or "") for r in hetero_pipeline_rows}):
        if not _mid:
            continue
        _cands = [r for r in hetero_pipeline_rows if str(r.get("model_id") or "") == _mid and _float(r.get("pipeline_fps")) is not None and _row_final_pass(r) is not False and str(r.get("quality") or "").lower() != "invalid"]
        if not _cands:
            continue
        _best_h = max(_cands, key=lambda r: float(_float(r.get("pipeline_fps")) or -1.0))
        _trt = _full_by_model_backend.get((_mid, "tensorrt")) or _full_by_model_backend.get((_mid, "ort_tensorrt"))
        if not _trt:
            # Some older result bundles use backend_label only.
            for _r in full_backend_rows:
                if str(_r.get("model_id") or "") == _mid and "tensorrt" in str(_r.get("backend") or _r.get("backend_label") or "").lower():
                    _trt = _r
                    break
        _cycle = _float(_best_h.get("pipeline_cycle_ms"))
        _split_lat = _float(_best_h.get("split_latency_ms"))
        _trt_lat = _float(_trt.get("latency_ms")) if _trt else None
        _trt_fps = _float(_trt.get("full_backend_throughput_fps")) if _trt else None
        _speed_cycle = _throughput_speedup(_trt_lat, _cycle)
        _speed_lat = _throughput_speedup(_trt_lat, _split_lat)
        _h_e_cmd = _float(_best_h.get("energy_j_per_frame_command_window"))
        _t_e_cmd = _float(_trt.get("energy_j_per_frame_command_window")) if _trt else None
        _h_e_sel = _float(_best_h.get("energy_j_per_frame_from_selected_fps"))
        _t_e_sel = _float(_trt.get("energy_j_per_frame_from_selected_fps")) if _trt else None
        _cmd_ratio = (_h_e_cmd / _t_e_cmd) if (_h_e_cmd is not None and _t_e_cmd not in (None, 0)) else None
        _sel_ratio = (_h_e_sel / _t_e_sel) if (_h_e_sel is not None and _t_e_sel not in (None, 0)) else None
        _row_case_ok = _row_final_pass(_best_h) is not False and "validation failed" not in str(_best_h.get("validation_claim_label") or "").lower()
        if _speed_cycle is not None and _speed_cycle >= 1.0 and _cmd_ratio is not None and _cmd_ratio <= 1.0 and _row_case_ok:
            _claim_status = "throughput_and_measured_energy_win"
        elif _speed_cycle is not None and _speed_cycle >= 1.0 and _row_case_ok:
            _claim_status = "throughput_win_energy_mixed"
        elif _cmd_ratio is not None and _cmd_ratio <= 1.0 and _row_case_ok:
            _claim_status = "energy_win_no_throughput_win"
        elif _row_case_ok:
            _claim_status = "valid_negative_or_control"
        else:
            _claim_status = "not_claim_ready"
        energy_claim_rows.append({
            "model_id": _mid,
            "best_hetero_backend": _best_h.get("backend", ""),
            "best_hetero_backend_label": _best_h.get("backend_label", _best_h.get("backend", "")),
            "best_hetero_case": _best_h.get("case_id", ""),
            "trt_full_latency_ms": _fmt(_trt_lat),
            "trt_full_fps": _fmt(_trt_fps),
            "hetero_split_latency_ms": _fmt(_split_lat),
            "hetero_pipeline_cycle_ms": _fmt(_cycle),
            "hetero_pipeline_fps": _fmt(_best_h.get("pipeline_fps")),
            "pipeline_speedup_vs_tensorrt_full": _fmt(_speed_cycle),
            "single_frame_latency_speedup_vs_tensorrt_full": _fmt(_speed_lat),
            "trt_energy_j_per_frame_command_window": _fmt(_t_e_cmd),
            "hetero_energy_j_per_frame_command_window": _fmt(_h_e_cmd),
            "energy_ratio_vs_tensorrt_command_window": _fmt(_cmd_ratio),
            "trt_energy_j_per_frame_selected_fps": _fmt(_t_e_sel),
            "hetero_energy_j_per_frame_selected_fps": _fmt(_h_e_sel),
            "energy_ratio_vs_tensorrt_selected_fps": _fmt(_sel_ratio),
            "energy_coverage_status": _best_h.get("energy_coverage_status", ""),
            "energy_target_status": _best_h.get("energy_target_status", ""),
            "validation_claim_label": _best_h.get("validation_claim_label", ""),
            "case_final_pass": _row_final_pass(_best_h),
            "claim_status": _claim_status,
            "paper_note": "Use command-window energy for measured replay-window claims; use selected-FPS energy only when discussing steady-state pipeline modelling.",
        })
    p_energy_claim_csv = write_csv(reports / "energy_claim_summary.csv", energy_claim_rows, [
        "model_id", "best_hetero_backend", "best_hetero_backend_label", "best_hetero_case",
        "trt_full_latency_ms", "trt_full_fps", "hetero_split_latency_ms", "hetero_pipeline_cycle_ms", "hetero_pipeline_fps",
        "pipeline_speedup_vs_tensorrt_full", "single_frame_latency_speedup_vs_tensorrt_full",
        "trt_energy_j_per_frame_command_window", "hetero_energy_j_per_frame_command_window", "energy_ratio_vs_tensorrt_command_window",
        "trt_energy_j_per_frame_selected_fps", "hetero_energy_j_per_frame_selected_fps", "energy_ratio_vs_tensorrt_selected_fps",
        "energy_coverage_status", "energy_target_status", "validation_claim_label", "case_final_pass", "claim_status", "paper_note",
    ])
    _claim_md = [
        "# Energy and throughput claim summary", "",
        "This table compares the best valid heterogeneous pipeline per model against the TensorRT full baseline. Command-window energy is the direct u.RECS replay-window value. Selected-FPS energy is a derived steady-state estimate and should be labelled separately.", "",
        "| Model | Hetero case | Pipeline speedup vs TRT | Energy ratio vs TRT (window) | Energy ratio vs TRT (selected FPS) | Claim status | Validation |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for _r in energy_claim_rows:
        _claim_md.append("| " + " | ".join([
            str(_r.get("model_id") or ""),
            f"{_r.get('best_hetero_backend_label','')} / {_r.get('best_hetero_case','')}",
            _paper_cell(_r.get("pipeline_speedup_vs_tensorrt_full")),
            _paper_cell(_r.get("energy_ratio_vs_tensorrt_command_window")),
            _paper_cell(_r.get("energy_ratio_vs_tensorrt_selected_fps")),
            str(_r.get("claim_status") or ""),
            str(_r.get("validation_claim_label") or ""),
        ]) + " |")
    p_energy_claim_md = write_text(reports / "energy_claim_summary.md", "\n".join(_claim_md) + "\n")
    p_energy_claim_tex = _write_latex_table(
        tables / "energy_claim_summary.tex",
        caption="Best heterogeneous split-pipeline claim summary against the TensorRT full baseline. Command-window and selected-FPS energy are reported separately.",
        label="energy_claim_summary",
        headers=["Model", "Backend/case", "TRT latency", "Cycle", "FPS", "Speedup vs TRT", "J/frame ratio", "Claim"],
        rows=[[
            r.get("model_id", ""),
            f"{r.get('best_hetero_backend_label','')} / {r.get('best_hetero_case','')}",
            r.get("trt_full_latency_ms", ""),
            r.get("hetero_pipeline_cycle_ms", ""),
            r.get("hetero_pipeline_fps", ""),
            r.get("pipeline_speedup_vs_tensorrt_full", ""),
            r.get("energy_ratio_vs_tensorrt_command_window", ""),
            r.get("claim_status", ""),
        ] for r in energy_claim_rows],
    )
    p_thesis_best_tex = _write_latex_table(
        tables / "thesis_best_summary.tex",
        caption="Thesis-ready best split summary. Heterogeneous speedup is computed from the TensorRT full latency divided by the split pipeline cycle.",
        label="thesis_best_summary",
        headers=["Model", "Best hetero", "Case", "Latency", "Cycle", "FPS", "Speedup", "Validation"],
        rows=[[
            r.get("model_id", ""), r.get("best_hetero_backend_label", ""), r.get("best_hetero_case", ""),
            r.get("hetero_split_latency_ms", ""), r.get("hetero_pipeline_cycle_ms", ""), r.get("hetero_pipeline_fps", ""),
            r.get("pipeline_speedup_vs_tensorrt_full", ""), r.get("validation_claim_label", ""),
        ] for r in energy_claim_rows],
    )
    # v59p: comprehensive documentation/claim table.  Keep the legacy
    # TensorRT-oriented energy_claim_summary, but add the dissertation-facing
    # table that contains all full models and the best split(s) per model.
    claim_analysis_reports = _write_claim_analysis_reports(
        reports,
        tables,
        full_backend_rows=full_backend_rows,
        hetero_pipeline_rows=hetero_pipeline_rows,
        model_cards=model_cards,
    )

    # v58b: compact paper/debug energy tables.  These are intentionally row-level
    # tables, not dispatch totals; dispatch energy is only a diagnostic.
    energy_summary_rows: List[Dict[str, Any]] = []
    energy_pipeline_rows: List[Dict[str, Any]] = []
    energy_dispatch_rows: List[Dict[str, Any]] = []
    energy_target_rows: List[Dict[str, Any]] = []
    for row in all_normalized_rows:
        if _row_has_urecs_energy(row):
            er = {
                "model_id": str(row.get("model_id") or ""),
                "backend": str(row.get("backend") or ""),
                "case_id": str(row.get("case_id") or ""),
                "variant": str(row.get("variant") or ""),
                "throughput_kind": _throughput_kind(row),
                "final_pass": _row_final_pass(row),
                "semantic_validation_ok": row.get("semantic_validation_ok", ""),
                "semantic_e2e_pass": row.get("semantic_e2e_pass", ""),
                "interface_contract_pass": row.get("interface_contract_pass", ""),
                "strict_boundary_numeric_pass": row.get("strict_boundary_numeric_pass", ""),
                "validation_claim_label": row.get("validation_claim_label", ""),
                "energy_measurement_scope": row.get("energy_measurement_scope", ""),
                "energy_row_level_source": row.get("energy_row_level_source", ""),
                "energy_coverage_status": row.get("energy_coverage_status", ""),
                "energy_target_status": row.get("energy_target_status", ""),
                "energy_target_id": row.get("energy_target_id", ""),
                "energy_target_ok": row.get("energy_target_ok", ""),
                "energy_target_error_summary": row.get("energy_target_error_summary", ""),
                "energy_target_window_count": row.get("energy_target_window_count", row.get("energy_window_count", "")),
                "energy_target_valid_window_count": row.get("energy_target_valid_window_count", row.get("energy_valid_window_count", "")),
                "energy_target_valid_window_ratio": _fmt(row.get("energy_target_valid_window_ratio")),
                "energy_merge_source": row.get("energy_merge_source", ""),
                "latency_ms": _fmt(_full_latency(row) or _complete_split_latency(row)),
                "throughput_primary_fps": _fmt(row.get("throughput_primary_fps") or row.get("full_backend_throughput_fps") or row.get("heterogeneous_pipeline_fps") or row.get("pipeline_fps_selected")),
                "pipeline_cycle_ms": _fmt(_pipeline_cycle(row)),
                "pipeline_fps_selected": _fmt(row.get("pipeline_fps_selected")),
                "energy_latency_j_per_inference": _fmt(row.get("row_energy_latency_j_per_inference") or row.get("energy_per_inference_j")),
                "energy_j_per_frame_command_window": _fmt(_row_energy_command_window_j_per_frame(row)),
                "energy_streaming_j_per_frame": _fmt(row.get("row_energy_streaming_j_per_frame") or row.get("energy_streaming_j_per_frame")),
                "energy_j_per_frame_from_selected_fps": _fmt(_row_energy_selected_fps_j_per_frame(row)),
                "energy_streaming_j_per_frame_from_selected_fps": _fmt(row.get("energy_streaming_j_per_frame_from_selected_fps")),
                "energy_streaming_avg_power_w": _fmt(row.get("energy_streaming_avg_power_w") or row.get("avg_power_w")),
                "energy_frames_per_j": _fmt(_row_energy_frames_per_j(row)),
                "fps_per_watt_from_selected_fps": _fmt(row.get("energy_streaming_fps_per_watt_from_selected_fps")),
                "energy_aggregate_relpath": _energy_relpath_for_report(row, run_dir, "energy_aggregate"),
                "artifact_summary_only": False,
            }
            energy_summary_rows.append(er)
            if er["throughput_kind"] == "heterogeneous_pipeline":
                energy_pipeline_rows.append(er)

    # v58f: Evaluation dispatch-level energy artifacts are diagnostics, not row
    # metrics. Keep them in a separate table so `energy_summary.csv` remains a
    # row-level table and does not advertise dispatch totals as J/inference.
    for agg_path in sorted((run_dir / "models").glob("*/benchmark_results/energy/**/energy_aggregate.json")):
        if any(part in {"latency", "streaming", "probe"} or part.startswith("run_") for part in agg_path.parts):
            continue
        try:
            model_id = agg_path.relative_to(run_dir / "models").parts[0]
        except Exception:
            model_id = ""
        try:
            agg = read_json(agg_path, default={}) or {}
        except Exception:
            agg = {}
        if not isinstance(agg, Mapping):
            continue
        # v58s: dispatch_summary is run-level only.  Row/target-level
        # aggregates are still useful for debugging, but live in a separate
        # energy_target_summary.csv table.
        is_target_agg = False
        try:
            is_target_agg = (
                "targets" in agg_path.parts
                or agg.get("energy_target_case") is not None
                or agg.get("target_case") is not None
                or agg.get("energy_target_variant") is not None
                or agg.get("target_variant") is not None
                or agg.get("target_id") is not None
            )
        except Exception:
            is_target_agg = False
        backend = str(agg.get("run_id") or agg_path.parents[1].name).strip()
        if is_target_agg:
            try:
                rel_t = str(agg_path.relative_to(run_dir))
            except Exception:
                rel_t = str(agg_path)
            phase_count = len(list(agg.get("phases") or [])) if isinstance(agg.get("phases"), list) else ""
            es = _energy_phase_summary(agg)
            _target_errors = []
            for _ph in (agg.get("phases") if isinstance(agg.get("phases"), list) else []):
                if isinstance(_ph, Mapping) and not _energy_phase_ok_v59k(_ph):
                    _target_errors.append(f"{_ph.get('phase') or _ph.get('name') or 'phase'}:{_ph.get('error') or _ph.get('status') or 'failed'}")
            energy_target_rows.append({
                "model_id": model_id,
                "backend": backend,
                "run_id": backend,
                "case_id": str(agg.get("energy_target_case") or agg.get("target_case") or ""),
                "variant": str(agg.get("energy_target_variant") or agg.get("target_variant") or ""),
                "target_status": str(agg.get("status") or ("ok" if agg.get("ok") is True else "unknown")),
                "target_ok": bool(es.get("usable")),
                "target_error_summary": "; ".join(_target_errors)[:500],
                "phase_count": phase_count,
                "energy_measurement_scope": str(agg.get("energy_measurement_scope") or "target_command_energy"),
                "target_energy_total_j": _fmt(es.get("total_energy_j")),
                "target_avg_power_w": _fmt(es.get("avg_power_w")),
                "target_energy_window_count": es.get("window_count", ""),
                "target_valid_energy_window_count": es.get("valid_window_count", ""),
                "energy_aggregate_relpath": rel_t,
                "artifact_summary_only": True,
            })
            continue
        try:
            rel = str(agg_path.relative_to(run_dir))
        except Exception:
            rel = str(agg_path)
        es = _energy_phase_summary(agg)
        total_j = es.get("total_energy_j")
        valid_wc = _float(es.get("valid_window_count"))
        energy_dispatch_rows.append({
            "model_id": model_id,
            "backend": backend,
            "run_id": backend,
            "energy_measurement_scope": str(agg.get("energy_measurement_scope") or "dispatch_command_energy"),
            "dispatch_energy_total_j": _fmt(total_j),
            "dispatch_avg_power_w": _fmt(es.get("avg_power_w")),
            "dispatch_energy_window_count": es.get("window_count", ""),
            "dispatch_valid_energy_window_count": es.get("valid_window_count", ""),
            "dispatch_avg_energy_j_per_window": _fmt((float(total_j) / float(valid_wc)) if total_j is not None and valid_wc not in (None, 0) else agg.get("dispatch_avg_energy_j_per_window") or agg.get("avg_energy_j_per_window")),
            "energy_aggregate_relpath": rel,
            "artifact_summary_only": True,
        })

    p_energy_summary_csv = write_csv(reports / "energy_summary.csv", energy_summary_rows, [
        "model_id", "backend", "case_id", "variant", "throughput_kind", "final_pass", "semantic_validation_ok", "semantic_e2e_pass",
        "interface_contract_pass", "strict_boundary_numeric_pass", "validation_claim_label",
        "latency_ms", "throughput_primary_fps", "pipeline_cycle_ms", "pipeline_fps_selected",
        "energy_streaming_avg_power_w", "energy_latency_j_per_inference", "energy_j_per_frame_command_window", "energy_streaming_j_per_frame",
        "energy_j_per_frame_from_selected_fps", "energy_streaming_j_per_frame_from_selected_fps", "energy_frames_per_j", "fps_per_watt_from_selected_fps",
        "energy_measurement_scope", "energy_row_level_source", "energy_coverage_status", "energy_target_status", "energy_target_ok", "energy_target_id", "energy_target_error_summary",
        "energy_target_window_count", "energy_target_valid_window_count", "energy_target_valid_window_ratio", "energy_merge_source",
        "energy_aggregate_relpath", "artifact_summary_only",
    ])
    p_energy_pipeline_csv = write_csv(reports / "energy_pipeline.csv", energy_pipeline_rows, [
        "model_id", "backend", "case_id", "variant", "throughput_kind", "final_pass", "validation_claim_label",
        "pipeline_cycle_ms", "pipeline_fps_selected", "energy_streaming_avg_power_w",
        "energy_j_per_frame_command_window", "energy_streaming_j_per_frame", "energy_j_per_frame_from_selected_fps", "energy_streaming_j_per_frame_from_selected_fps",
        "energy_frames_per_j", "fps_per_watt_from_selected_fps", "energy_measurement_scope", "energy_row_level_source", "energy_coverage_status",
        "energy_target_status", "energy_target_id", "energy_target_valid_window_count", "energy_merge_source",
    ])
    p_energy_dispatch_csv = write_csv(reports / "energy_dispatch_summary.csv", energy_dispatch_rows, [
        "model_id", "backend", "run_id", "energy_measurement_scope",
        "dispatch_energy_total_j", "dispatch_avg_power_w", "dispatch_energy_window_count",
        "dispatch_valid_energy_window_count", "dispatch_avg_energy_j_per_window",
        "energy_aggregate_relpath", "artifact_summary_only",
    ])
    p_energy_target_csv = write_csv(reports / "energy_target_summary.csv", energy_target_rows, [
        "model_id", "backend", "run_id", "case_id", "variant", "target_status", "target_ok", "target_error_summary", "phase_count", "energy_measurement_scope",
        "target_energy_total_j", "target_avg_power_w", "target_energy_window_count",
        "target_valid_energy_window_count", "energy_aggregate_relpath", "artifact_summary_only",
    ])
    profile_energy = {}
    try:
        import yaml  # optional dependency already used by workflow profile loading
        _profile_payload = yaml.safe_load((run_dir / "profile.yaml").read_text(encoding="utf-8")) or {}
        profile_energy = (_profile_payload.get("energy") if isinstance(_profile_payload, Mapping) else {}) or {}
    except Exception:
        profile_energy = {}
    def _truthy_v59l(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "strict"}

    def _int_or_none_v59l(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except Exception:
            return None

    _profile_final_all_energy = _truthy_v59l(profile_energy.get("final_all_split_energy") or profile_energy.get("all_split_energy") or profile_energy.get("require_complete_split_energy"))
    _profile_energy_enabled = _truthy_v59l(profile_energy.get("enabled")) or _profile_final_all_energy
    _profile_target_policy = "all" if _profile_final_all_energy else (str(profile_energy.get("target_policy") or "").strip().lower() or "default")
    _profile_max_targets = 0 if _profile_final_all_energy else _int_or_none_v59l(profile_energy.get("max_targets_per_run_id"))
    _profile_strict = _profile_final_all_energy or _truthy_v59l(profile_energy.get("strict"))
    _profile_skip_backends_raw = profile_energy.get("skip_backends")
    if isinstance(_profile_skip_backends_raw, str):
        _profile_skip_backends = [x.strip() for x in _profile_skip_backends_raw.split(",") if x.strip()]
    elif isinstance(_profile_skip_backends_raw, Sequence) and not isinstance(_profile_skip_backends_raw, (str, bytes, bytearray)):
        _profile_skip_backends = [str(x).strip() for x in _profile_skip_backends_raw if str(x).strip()]
    else:
        _profile_skip_backends = []

    def _profile_final_skip_cpu_ort_v59n() -> bool:
        if not _profile_final_all_energy:
            return False
        # v59n default: final all-split energy means all thesis-relevant
        # accelerator/GPU rows.  CPU ORT same-backend diagnostics are very slow
        # and optional unless the profile explicitly asks to include them.
        for _k in ("include_cpu_ort_in_final_energy", "measure_cpu_ort_in_final_energy", "final_energy_include_cpu_ort"):
            if _truthy_v59l(profile_energy.get(_k)):
                return False
        for _k in ("final_energy_skip_cpu_ort", "skip_cpu_ort_in_final_energy", "exclude_cpu_ort_in_final_energy"):
            if _k in profile_energy:
                return _truthy_v59l(profile_energy.get(_k))
        return True

    _cpu_ort_skip_ids_v59n = {"ort_cpu", "cpu_ort", "cpu"}
    _final_cpu_ort_skipped = _profile_final_skip_cpu_ort_v59n()
    if _final_cpu_ort_skipped:
        for _rid in sorted(_cpu_ort_skip_ids_v59n):
            if _rid not in _profile_skip_backends:
                _profile_skip_backends.append(_rid)
    else:
        # Advanced include_cpu_ort_in_final_energy=true removes stale CPU skip
        # entries from profile files so CPU rows remain part of completeness.
        _profile_skip_backends = [x for x in _profile_skip_backends if str(x).strip().lower().replace("-", "_") not in _cpu_ort_skip_ids_v59n]

    def _row_backend_id_v59n(row: Mapping[str, Any]) -> str:
        return str(row.get("backend") or row.get("run_id") or row.get("backend_id") or row.get("provider") or "").strip().lower().replace("-", "_")

    def _row_excluded_from_energy_completeness_v59n(row: Mapping[str, Any]) -> bool:
        rid = _row_backend_id_v59n(row)
        if not rid:
            return False
        skips = {str(x).strip().lower().replace("-", "_") for x in _profile_skip_backends if str(x).strip()}
        if rid in skips:
            return True
        if rid == "cpu_ort" and ("ort_cpu" in skips or "cpu" in skips):
            return True
        if rid == "ort_cpu" and ("cpu_ort" in skips or "cpu" in skips):
            return True
        return False

    _expected_split_energy_rows_all = [
        r for r in all_normalized_rows
        if str(r.get("variant") or "").lower() == "split"
        and _complete_split_latency(r) is not None
        and _row_final_pass(r) is not False
    ]
    _expected_split_energy_rows = [r for r in _expected_split_energy_rows_all if not _row_excluded_from_energy_completeness_v59n(r)]
    _excluded_split_energy_rows = [r for r in _expected_split_energy_rows_all if _row_excluded_from_energy_completeness_v59n(r)]
    _expected_split_energy_present = [r for r in _expected_split_energy_rows if _row_has_urecs_energy(r)]
    _missing_split_energy_rows = [r for r in _expected_split_energy_rows if not _row_has_urecs_energy(r)]

    _energy_configuration_warnings: List[str] = []
    if _profile_energy_enabled:
        if _profile_target_policy != "all":
            _energy_configuration_warnings.append(f"target_policy is {_profile_target_policy!r}; use 'all' for final all-split energy.")
        if _profile_max_targets not in (None, 0):
            _energy_configuration_warnings.append(f"max_targets_per_run_id={_profile_max_targets}; use 0 for no target cap.")
        _non_cpu_skips = [x for x in _profile_skip_backends if str(x).strip().lower().replace("-", "_") not in _cpu_ort_skip_ids_v59n]
        if _non_cpu_skips:
            _energy_configuration_warnings.append("skip_backends contains non-CPU entries: " + ",".join(_non_cpu_skips))
        if not _profile_strict:
            _energy_configuration_warnings.append("strict=false; incomplete split energy will be reported but will not fail the workflow.")
    _all_split_energy_config_ok = bool(_profile_energy_enabled and not _energy_configuration_warnings)
    _row_level_energy_complete = bool(_expected_split_energy_rows) and not _missing_split_energy_rows
    _strict_energy_gate_would_fail = bool(_profile_energy_enabled and _profile_strict and _expected_split_energy_rows and _missing_split_energy_rows)
    if not _profile_energy_enabled:
        _energy_status = "disabled"
    elif _missing_split_energy_rows:
        _energy_status = "row_level_energy_partial"
    elif energy_summary_rows:
        _energy_status = "row_level_energy_present"
    elif energy_dispatch_rows:
        _energy_status = "dispatch_energy_only"
    else:
        _energy_status = "enabled_but_no_energy_rows"
    p_energy_status_json = write_json(reports / "energy_status.json", {
        "schema": "onnx-splitpoint/energy-report-status",
        "schema_version": 4,
        "energy_enabled_in_profile": _profile_energy_enabled,
        "final_all_split_energy": _profile_final_all_energy,
        "final_energy_skip_cpu_ort": _final_cpu_ort_skipped,
        "energy_completeness_excluded_backend_ids": sorted(_cpu_ort_skip_ids_v59n) if _final_cpu_ort_skipped else [],
        "energy_summary_row_count": len(energy_summary_rows),
        "energy_pipeline_row_count": len(energy_pipeline_rows),
        "energy_dispatch_row_count": len(energy_dispatch_rows),
        "energy_target_row_count": len(energy_target_rows),
        "expected_complete_split_energy_row_count": len(_expected_split_energy_rows),
        "complete_split_rows_excluded_from_energy_count": len(_excluded_split_energy_rows),
        "complete_split_rows_with_row_level_energy_count": len(_expected_split_energy_present),
        "missing_complete_split_row_level_energy_count": len(_missing_split_energy_rows),
        "row_level_energy_complete_for_complete_splits": _row_level_energy_complete,
        "all_split_energy_config_ok": _all_split_energy_config_ok,
        "strict_energy_gate_would_fail": _strict_energy_gate_would_fail,
        "energy_configuration_warnings": _energy_configuration_warnings,
        "missing_complete_split_energy_examples": [
            {"model_id": r.get("model_id", ""), "backend": r.get("backend", ""), "case_id": r.get("case_id", ""), "variant": r.get("variant", "")}
            for r in _missing_split_energy_rows[:30]
        ],
        "excluded_complete_split_energy_examples": [
            {"model_id": r.get("model_id", ""), "backend": r.get("backend", ""), "case_id": r.get("case_id", ""), "variant": r.get("variant", ""), "reason": "cpu_ort_energy_excluded_by_final_preset"}
            for r in _excluded_split_energy_rows[:30]
        ],
        "status": _energy_status,
        "note": ("Energy measurement is disabled in the evaluation profile; empty energy CSVs are expected." if not _profile_energy_enabled else "Energy reports are generated from row-level u.RECS fields only; dispatch rows are diagnostics and are not copied into per-row J/frame metrics."),
        "profile_energy": profile_energy,
        "native_evidence_status": native_evidence,
        "native_evidence_summary": native_evidence_summary,
    })

    # v59m: derived next-run recommendation report.  This does not change
    # measured results; it turns the dashboard into a planning aid for short
    # overnight runs by combining prediction hit/miss, validation, speedup and
    # row-level energy completeness.
    _missing_energy_keys = {
        (str(r.get("model_id") or ""), str(r.get("backend") or ""), str(r.get("case_id") or ""), str(r.get("variant") or ""))
        for r in _missing_split_energy_rows
    }
    _thesis_by_model = {str(r.get("model_id") or ""): r for r in thesis_rows}
    next_run_recommendation_rows: List[Dict[str, Any]] = []
    prediction_calibration_rows: List[Dict[str, Any]] = []
    for m in model_cards:
        mid = str(m.get("model_id") or "")
        task = str(m.get("task") or "")
        tr = _thesis_by_model.get(mid, {})
        model_hetero = [r for r in hetero_pipeline_rows if str(r.get("model_id") or "") == mid]
        def _pfps(r: Mapping[str, Any]) -> float:
            v = _float(r.get("pipeline_fps"))
            return float(v) if v is not None else -1.0
        best_h = max(model_hetero, key=_pfps) if model_hetero else {}
        best_case = str(best_h.get("case_id") or m.get("best_pipeline_split_case") or m.get("best_complete_split_case") or "")
        best_backend = str(best_h.get("backend") or m.get("best_pipeline_split_backend") or m.get("best_complete_split_backend") or "")
        speed = _float(best_h.get("pipeline_speedup_vs_tensorrt_full") or m.get("pipeline_speedup_vs_tensorrt_full"))
        speed = float(speed) if speed is not None else None
        energy_status = str(best_h.get("energy_coverage_status") or "")
        energy_ok = energy_status in {"target_aggregate_merged", "row_level_energy", "benchmark_row_scope"}
        best_key = (mid, best_backend, best_case, "split")
        best_energy_missing = bool(best_key in _missing_energy_keys or (best_h and not energy_ok))
        # v59o: plan around the best case-level split, not only the coarse
        # model-level validation flag.  A model can contain invalid optional
        # rows (e.g. Hailo full YOLO diagnostics) while its best heterogeneous
        # split is valid and claim-worthy.
        best_row_final = _row_final_pass(best_h) if best_h else None
        best_row_claim = str(best_h.get("validation_claim_label") or "").lower() if best_h else ""
        best_row_quality = str(best_h.get("quality") or "").strip().lower() if best_h else ""
        best_row_val_ok = bool(best_h) and best_row_final is not False and best_row_quality != "invalid" and "validation failed" not in best_row_claim
        val_ok = bool(best_row_val_ok)
        model_validation_ok = _truthy_v59l(m.get("validation_ok"))
        top1 = _truthy_v59l(tr.get("top1_hit")) if tr else False
        topk = _truthy_v59l(tr.get("topk_hit")) if tr else False
        rank_corr = _float(tr.get("rank_correlation_spearman")) if tr else None
        accepted_outside = int(_float(tr.get("accepted_outside_prediction_count")) or 0) if tr else 0
        pred_unmeasured = int(_float(tr.get("predicted_unmeasured_count")) or 0) if tr else 0
        reasons: List[str] = []
        priority = 0
        action = "review"
        if speed is not None and speed >= 1.0 and val_ok:
            priority += 80
            action = "rerun_best_case_with_final_energy"
            reasons.append(f"heterogeneous pipeline beats TensorRT full ({speed:.2f}x)")
        elif speed is not None and speed >= 0.9 and val_ok:
            priority += 55
            action = "rerun_as_near_miss_control"
            reasons.append(f"near TensorRT full ({speed:.2f}x); useful sensitivity/control")
        elif val_ok:
            priority += 25
            action = "keep_as_negative_control"
            if speed is not None:
                reasons.append(f"valid but below TensorRT full ({speed:.2f}x)")
        else:
            priority += 10
            action = "fix_validation_before_claim"
            reasons.append("validation/visual status is not fully clean")
        if best_energy_missing:
            priority += 20
            reasons.append("best split lacks complete row-level energy")
        if task.lower().startswith("detection"):
            reasons.append("use strict task metrics for final detection claim")
        if not top1:
            if topk:
                reasons.append("prediction top-k contains useful case but top-1 missed")
            else:
                reasons.append("prediction ranking missed measured best; use spread/backfill candidates")
        next_run_recommendation_rows.append({
            "model_id": mid,
            "task": task,
            "priority_score": priority,
            "recommended_action": action,
            "recommended_case": best_case,
            "recommended_backend": best_backend,
            "pipeline_speedup_vs_tensorrt_full": _fmt(speed),
            "validation_ok": val_ok,
            "model_validation_ok": model_validation_ok,
            "best_case_validation_ok": best_row_val_ok,
            "best_energy_coverage_status": energy_status,
            "best_row_level_energy_ok": bool(not best_energy_missing),
            "predicted_best_case": tr.get("predicted_best_case", "") if tr else "",
            "measured_best_case": tr.get("measured_best_case", "") if tr else "",
            "top1_hit": tr.get("top1_hit", "") if tr else "",
            "topk_hit": tr.get("topk_hit", "") if tr else "",
            "rank_correlation_spearman": tr.get("rank_correlation_spearman", "") if tr else "",
            "rationale": "; ".join(reasons),
        })
        if top1 and (rank_corr is None or rank_corr >= 0.5):
            reliability = "good"
        elif topk or (rank_corr is not None and rank_corr >= 0.3):
            reliability = "usable"
        else:
            reliability = "weak"
        calibration_note = []
        if accepted_outside:
            calibration_note.append(f"{accepted_outside} accepted case(s) came from generator/backfill outside prediction top-k")
        if pred_unmeasured:
            calibration_note.append(f"{pred_unmeasured} predicted case(s) were not benchmarked")
        if rank_corr is not None and rank_corr < 0.3:
            calibration_note.append("low rank correlation; do not rely on top-1 prediction alone")
        prediction_calibration_rows.append({
            "model_id": mid,
            "task": task,
            "prediction_reliability": reliability,
            "predicted_best_case": tr.get("predicted_best_case", "") if tr else "",
            "measured_best_case": tr.get("measured_best_case", "") if tr else "",
            "top1_hit": tr.get("top1_hit", "") if tr else "",
            "topk_hit": tr.get("topk_hit", "") if tr else "",
            "rank_correlation_spearman": tr.get("rank_correlation_spearman", "") if tr else "",
            "accepted_outside_prediction_count": accepted_outside,
            "predicted_unmeasured_count": pred_unmeasured,
            "recommendation": "benchmark measured-best plus a spread of late/mid candidates" if reliability == "weak" else ("keep top-k plus measured-best backfill" if reliability == "usable" else "top-k prediction is acceptable"),
            "note": "; ".join(calibration_note),
        })
    next_run_recommendation_rows.sort(key=lambda r: int(r.get("priority_score") or 0), reverse=True)
    p_next_run_csv = write_csv(reports / "next_run_recommendations.csv", next_run_recommendation_rows, [
        "model_id", "task", "priority_score", "recommended_action", "recommended_case", "recommended_backend",
        "pipeline_speedup_vs_tensorrt_full", "validation_ok", "model_validation_ok", "best_case_validation_ok", "best_energy_coverage_status", "best_row_level_energy_ok",
        "predicted_best_case", "measured_best_case", "top1_hit", "topk_hit", "rank_correlation_spearman", "rationale",
    ])
    p_prediction_calibration_csv = write_csv(reports / "prediction_calibration.csv", prediction_calibration_rows, [
        "model_id", "task", "prediction_reliability", "predicted_best_case", "measured_best_case", "top1_hit", "topk_hit", "rank_correlation_spearman",
        "accepted_outside_prediction_count", "predicted_unmeasured_count", "recommendation", "note",
    ])
    p_next_run_json = write_json(reports / "next_run_recommendations.json", {
        "schema": "onnx-splitpoint/next-run-recommendations",
        "schema_version": 1,
        "created_at": now_iso(),
        "rows": next_run_recommendation_rows,
        "prediction_calibration": prediction_calibration_rows,
    })
    _rec_lines = ["# Next-run recommendations", "", "| Model | Priority | Action | Case / backend | Speedup vs TRT | Energy OK | Rationale |", "|---|---:|---|---|---:|---:|---|"]
    for r in next_run_recommendation_rows[:25]:
        _rec_lines.append("| " + " | ".join([
            str(r.get("model_id") or ""),
            str(r.get("priority_score") or ""),
            str(r.get("recommended_action") or ""),
            f"{r.get('recommended_case','')} / {r.get('recommended_backend','')}",
            str(r.get("pipeline_speedup_vs_tensorrt_full") or ""),
            str(r.get("best_row_level_energy_ok") or ""),
            str(r.get("rationale") or ""),
        ]) + " |")
    p_next_run_md = write_text(reports / "next_run_recommendations.md", "\n".join(_rec_lines) + "\n")

    p_full_throughput_tex = _write_latex_table(
        tables / "full_backend_throughput.tex",
        caption="Full-backend throughput baselines. These rows describe one complete backend, not a split pipeline.",
        label="full_backend_throughput",
        headers=["Model", "Backend", "Latency [ms]", "Full FPS", "Energy/frame [J]"],
        rows=[[r.get("model_id", ""), r.get("backend_label", r.get("backend", "")), r.get("latency_ms", ""), r.get("full_backend_throughput_fps", ""), r.get("energy_j_per_frame_command_window", r.get("energy_j_per_frame", ""))] for r in full_backend_rows[:60]],
    )
    p_hetero_pipeline_tex = _write_latex_table(
        tables / "heterogeneous_pipeline_throughput.tex",
        caption="Heterogeneous split-pipeline throughput. These rows correspond to cross-backend Stage~1/Stage~2 executions.",
        label="heterogeneous_pipeline_throughput",
        headers=["Model", "Backend", "Case", "Latency [ms]", "Cycle [ms]", "FPS", "Energy/frame [J]"],
        rows=[[r.get("model_id", ""), r.get("backend_label", r.get("backend", "")), r.get("case_id", ""), r.get("split_latency_ms", ""), r.get("pipeline_cycle_ms", ""), r.get("pipeline_fps", ""), r.get("energy_j_per_frame_command_window", r.get("energy_j_per_frame", ""))] for r in hetero_pipeline_rows[:80]],
    )

    p_canonical_full = write_csv(reports / "canonical_full_baselines.csv", canonical_full_rows, [
        "model_id", "backend", "canonical_policy", "canonical_latency_ms",
        "median_latency_ms", "max_latency_ms", "sample_count", "spread_pct", "ambiguous"
    ])

    # v57: make benchmark input semantics a first-class report artifact.
    # Hailo and ORT/TensorRT timing loops repeat the same prepared feed by
    # default; semantic validation and visual verification use dataset images
    # separately.  DeepX run_model rows are backend microbenchmarks and should
    # not be read as cycling Imagenette/COCO images.
    benchmark_input_policy_rows: List[Dict[str, Any]] = []
    seen_input_rows = set()
    for model_id in all_model_ids:
        for rr in _load_normalized_rows(run_dir, model_id):
            pol = rr.get("benchmark_input_policy") if isinstance(rr.get("benchmark_input_policy"), Mapping) else {}
            mode = str(rr.get("benchmark_input_mode") or pol.get("mode") or "").strip()
            loop_count = rr.get("benchmark_loop_count", pol.get("loop_count"))
            unique_images = rr.get("benchmark_unique_dataset_images", pol.get("unique_dataset_images"))
            if not (mode or loop_count not in (None, "") or unique_images not in (None, "")):
                continue
            row = {
                "model_id": model_id,
                "case_id": rr.get("case_id", ""),
                "backend": rr.get("backend", rr.get("run_id", "")),
                "variant": rr.get("variant", ""),
                "benchmark_input_mode": mode,
                "benchmark_loop_count": loop_count,
                "benchmark_unique_dataset_images": unique_images,
                "benchmark_warmup_count": pol.get("warmup_count", rr.get("warmup_count", "")),
                "input_source": pol.get("input_source", ""),
                "image_input_name": pol.get("image_input_name", ""),
                "image_scale": pol.get("image_scale", ""),
                "uses_validation_dataset_images": pol.get("uses_validation_dataset_images", ""),
                "semantic_validation_uses_dataset_images": pol.get("semantic_validation_uses_dataset_images", ""),
                "same_image_schedule_as_ort_trt": pol.get("same_image_schedule_as_ort_trt", ""),
                "dxrt_tool_fps": rr.get("dxrt_tool_fps", ""),
                "dxrt_tool_fps_semantics": rr.get("dxrt_tool_fps_semantics", ""),
                "note": pol.get("note", ""),
            }
            key = tuple(str(row.get(k, "")) for k in ("model_id", "case_id", "backend", "variant", "benchmark_input_mode"))
            if key in seen_input_rows:
                continue
            seen_input_rows.add(key)
            benchmark_input_policy_rows.append(row)
    p_input_policy_csv = write_csv(reports / "benchmark_input_policy.csv", benchmark_input_policy_rows, [
        "model_id", "case_id", "backend", "variant", "benchmark_input_mode",
        "benchmark_loop_count", "benchmark_unique_dataset_images", "benchmark_warmup_count",
        "input_source", "image_input_name", "image_scale", "uses_validation_dataset_images",
        "semantic_validation_uses_dataset_images", "same_image_schedule_as_ort_trt",
        "dxrt_tool_fps", "dxrt_tool_fps_semantics", "note"
    ])

    dashboard = {
        **overview,
        "overview": dict(overview),
        "models": model_cards,
        "thesis_metrics": thesis_rows,
        "backend_speedups": backend_speedup_rows,
        "next_run_recommendations": next_run_recommendation_rows,
        "prediction_calibration": prediction_calibration_rows,
        "benchmark_input_policy": benchmark_input_policy_rows,
        "hardware_summary": hw_summary_rows,
        "run_status_summary": run_status,
        "native_evidence_status": native_evidence,
        "native_evidence_summary": native_evidence_summary,
    }

    p_dashboard_json = write_json(reports / "result_dashboard.json", dashboard)
    dashboard_csv_fields = [
        "model_id", "health", "task", "accepted_case_count", "measured_result_count",
        "complete_split_count", "valid_complete_split_count", "contract_rejected_split_count",
        "component_only_split_count", "best_complete_split_case",
        "best_complete_split_backend", "best_complete_split_quality", "best_complete_split_contract_status",
        "best_complete_split_latency_ms", "best_heterogeneous_split_case", "best_heterogeneous_split_backend", "best_heterogeneous_split_latency_ms",
        "best_same_backend_diagnostic_case", "best_same_backend_diagnostic_backend", "best_same_backend_diagnostic_latency_ms",
        "best_pipeline_split_case", "best_pipeline_split_backend",
        "best_pipeline_cycle_ms", "best_pipeline_fps", "pipeline_speedup_vs_cpu_full", "pipeline_speedup_vs_tensorrt_full", "best_full_backend",
        "best_full_latency_ms", "cpu_full_latency_ms", "tensorrt_full_latency_ms",
        "speedup_vs_cpu_full", "speedup_vs_tensorrt_full", "validation_status",
        "validation_ok", "hardware_smoke_status", "hardware_verified",
        "hailo_runtime_verified", "hailo_full_runtime_verified", "hailo_split_runtime_verified",
        "best_hailo_full_latency_ms", "best_hailo_split_latency_ms", "remote_status",
    ]
    p_dashboard_csv = write_csv(reports / "result_dashboard.csv", model_cards, dashboard_csv_fields)
    thesis_fields = [
        "model_id", "prediction_scope", "predicted_best_case", "measured_best_case", "top1_hit", "topk_hit",
        "measured_candidate_count", "accepted_case_count", "accepted_outside_prediction_count", "predicted_unmeasured_count",
        "best_regret_pct", "mean_regret_pct", "rank_correlation_spearman", "feasibility_accuracy", "speedup_error_mean_pct",
        "best_complete_split_latency_ms", "speedup_vs_cpu_full", "speedup_vs_tensorrt_full", "best_pipeline_cycle_ms", "best_pipeline_fps", "pipeline_speedup_vs_tensorrt_full", "validation_ok", "hailo_runtime_verified", "hailo_full_runtime_verified", "hailo_composed_runtime_verified",
    ]
    p_thesis_csv = write_csv(reports / "thesis_metrics.csv", thesis_rows, thesis_fields)
    p_thesis_json = write_json(reports / "thesis_metrics.json", {"schema": "onnx-splitpoint/thesis-metrics", "schema_version": 1, "created_at": now_iso(), "rows": thesis_rows})
    p_backend_csv = write_csv(reports / "backend_speedups.csv", backend_speedup_rows, ["model_id", "backend", "backend_label", "best_full_latency_ms", "best_complete_split_latency_ms", "best_pipeline_cycle_ms", "best_pipeline_fps", "pipeline_speedup_vs_tensorrt_full", "best_complete_split_case", "best_pipeline_split_case", "best_complete_split_quality", "best_pipeline_split_quality", "stage2_contract_status", "stage2_contract_probe_samples", "stage2_contract_pass_ratio", "stage2_calibration_source", "speedup_vs_cpu_full", "speedup_vs_tensorrt_full", "speedup_vs_tensorrt_full_median", "pipeline_speedup_vs_tensorrt_full_median", "complete_split_measured", "throughput_kind", "same_backend_diagnostic_case"])

    def _evidence_text_v272(value: Any) -> str:
        if value is None:
            return "unavailable"
        return str(value)

    _energy_success = _evidence_text_v272(
        native_evidence_summary["energy_measurement_success_count"]
    )
    _energy_plan_denominator = _evidence_text_v272(
        native_evidence_summary["energy_plan_denominator_count"]
    )
    _energy_matrix_denominator = _evidence_text_v272(
        native_evidence_summary["energy_matrix_denominator_count"]
    )
    _energy_started = _evidence_text_v272(
        native_evidence_summary["energy_measurement_started_count"]
    )
    _energy_not_started = _evidence_text_v272(
        native_evidence_summary["energy_not_started_preflight_count"]
    )
    md_lines = [
        f"# Result Dashboard — {run_dir.name}",
        "",
        f"Profile: `{profile_id}`",
        f"Tool: `{tool_version}`",
        f"Workflow: `{workflow_version}`",
        f"Run status: **{overview['run_status']}**",
        "",
        "## Overview",
        "",
        f"- Models: {overview['model_count']}",
        f"- Accepted benchmark cases: {overview['accepted_case_total']}",
        f"- Complete split measurements: {overview['complete_split_total']}",
        f"- Component-only split rows: {overview['component_only_split_total']}",
        f"- Blocking reasons: {overview['blocking_reason_count']}",
        f"- Non-blocking warnings: {overview['non_blocking_reason_count']}",
        "",
        "## Scientific evidence status",
        "",
        (
            "- Native evidence available: "
            f"**{native_evidence_summary['available']}**"
        ),
        (
            "- Scientific ready: "
            f"**{_evidence_text_v272(native_evidence_summary['scientific_ready'])}** "
            f"(status: `{native_evidence_summary['scientific_status']}`)"
        ),
        (
            "- Technical completeness: "
            f"**{_evidence_text_v272(native_evidence_summary['technical_complete'])}** "
            f"(status: `{native_evidence_summary['technical_status']}`)"
        ),
        (
            "- Claim decisions complete: "
            f"**{_evidence_text_v272(native_evidence_summary['claim_decisions_complete'])}**"
        ),
        (
            "- Native energy execution: "
            f"**{native_evidence_summary['energy_execution_status']}**"
        ),
        (
            "- Native energy attempts: "
            f"**{_energy_started} started, "
            f"{_energy_not_started} not started**"
        ),
        f"- Accounting: {native_evidence_summary.get('energy_accounting_summary') or 'unavailable for legacy evidence'}",
        (
            "- Energy plan completion: "
            f"**{_energy_success}/{_energy_plan_denominator}** "
            "(successful measurements / planned measurements)"
        ),
        (
            "- Energy matrix coverage: "
            f"**{_energy_success}/{_energy_matrix_denominator}** "
            "(successful measurements / expected matrix rows)"
        ),
        (
            "- Energy plan exclusions: "
            f"**{_evidence_text_v272(native_evidence_summary['energy_plan_excluded_count'])}**"
        ),
        (
            "- Energy claim-eligible measurements: "
            f"**{_evidence_text_v272(native_evidence_summary['energy_claim_eligible_count'])}**"
        ),
        (
            "- Final all-split energy complete: "
            f"**{_evidence_text_v272(native_evidence_summary['final_all_split_energy_complete'])}**"
        ),
        "",
        "## Benchmark input policy",
        "",
        "Timing and energy loops use backend benchmark inputs, not the full validation dataset.",
        "Hailo and ORT/TensorRT runner rows repeat one prepared feed for `--runs=N`; Imagenette/COCO are used separately for semantic validation and visual verification.",
        "DeepX `run_model` rows are DXRT backend microbenchmarks and keep their internal throughput as diagnostics, not as application-level pipeline FPS.",
        "",
        f"- Input-policy rows: {len(benchmark_input_policy_rows)}",
        "- See `reports/benchmark_input_policy.csv` for per-row provenance.",
        "",
        "## Model cards",
        "",
        "| Model | Health | Accepted | Valid / measured splits | Contract rejects | Best hetero split | Hetero latency [ms] | Same-backend diagnostic | Diagnostic latency [ms] | Best pipeline split | Cycle [ms] | Pipeline FPS | Pipeline speedup vs TensorRT | Validation | Hailo evidence |",
        "|---|---:|---:|---:|---:|---|---:|---|---:|---|---:|---:|---:|---|---|",
    ]
    for m in model_cards:
        hailo = m.get("hailo_evidence") if isinstance(m.get("hailo_evidence"), Mapping) else {}
        hailo_text = (
            f"HEF={hailo.get('hef_detected_count', 0)}, dev={hailo.get('device_present')}, "
            f"svc={hailo.get('service_running')}, import={hailo.get('python_import_ok')}, "
            f"full={hailo.get('full_runtime_verified')}, split={hailo.get('split_composed_runtime_verified')}"
        )
        md_lines.append(
            "| " + " | ".join([
                str(m.get("model_id") or ""),
                str(m.get("health") or ""),
                str(m.get("accepted_case_count") or 0),
                f"{m.get('valid_complete_split_count', 0)} / {m.get('complete_split_count', 0)}",
                str(m.get('contract_rejected_split_count') or 0),
                f"{m.get('best_heterogeneous_split_case','')} / {_backend_display(str(m.get('best_heterogeneous_split_backend','')))}",
                str(m.get("best_heterogeneous_split_latency_ms") or ""),
                f"{m.get('best_same_backend_diagnostic_case','')} / {_backend_display(str(m.get('best_same_backend_diagnostic_backend','')))}",
                str(m.get("best_same_backend_diagnostic_latency_ms") or ""),
                f"{m.get('best_pipeline_split_case','')} / {_backend_display(str(m.get('best_pipeline_split_backend','')))}",
                str(m.get("best_pipeline_cycle_ms") or ""),
                str(m.get("best_pipeline_fps") or ""),
                str(m.get("pipeline_speedup_vs_tensorrt_full") or ""),
                (f"{m.get('validation_status','')} ({m.get('validation_ok')})" if str(m.get('validation_ok') or '') not in {'', 'None'} else str(m.get('validation_status',''))),
                hailo_text,
            ]) + " |"
        )
    if canonical_full_rows:
        amb = [r for r in canonical_full_rows if str(r.get("ambiguous")).lower() == "true"]
        md_lines.extend([
            "",
            "## Canonical full baselines",
            "",
            "Full-baseline comparisons use a canonical per-backend policy (`min_valid_full_latency`). If multiple full baselines differ strongly, the row is marked ambiguous and should be reviewed before making speedup claims.",
            "",
            f"See `canonical_full_baselines.csv` for {len(canonical_full_rows)} baseline rows. Ambiguous rows: {len(amb)}.",
        ])

    md_lines.extend([
        "",
        "## Throughput taxonomy",
        "",
        "The dashboard uses a deliberately small set of decision metrics. `Full backend throughput` describes a complete model on one backend. `Heterogeneous pipeline throughput` describes a true two-stage cross-backend split. `Same-backend split diagnostics` are kept for debugging split overhead but are not treated as double-accelerator evidence.",
        "",
        "| Kind | Primary metric | Energy metric | Used for claims |",
        "|---|---|---|---|",
        "| Full backend | `full_backend_throughput_fps` | `energy_j_per_frame` | Baseline throughput/energy |",
        "| Heterogeneous pipeline | `pipeline_fps` / `pipeline_cycle_ms` | `energy_fps_per_watt`, `energy_j_per_frame` | Split throughput and FPS/W claims |",
        "| Same-backend split diagnostic | `diagnostic_fps` | optional | Debugging only |",
    ])

    md_lines.extend([
        "",
        "## Heterogeneous pipeline throughput leaders",
        "",
        "These rows focus on double-accelerator / cross-backend pipelines. Same-backend TensorRT/CUDA split rows are useful diagnostics but are listed separately below.",
        "",
        "| Model | Backend | Case | Split latency [ms] | Pipeline cycle [ms] | Pipeline FPS | Speedup vs TensorRT full | Quality |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    hetero_pipe_leaders = sorted([r for r in backend_speedup_rows if _float(r.get("best_pipeline_cycle_ms")) is not None and _row_is_heterogeneous_accelerator_split(r)], key=lambda r: (str(r.get("model_id") or ""), float(_float(r.get("best_pipeline_cycle_ms")) or 1e18)))
    for r in hetero_pipe_leaders[:60]:
        md_lines.append("| " + " | ".join([
            str(r.get("model_id") or ""),
            str(r.get("backend_label") or r.get("backend") or ""),
            str(r.get("best_complete_split_case") or ""),
            str(r.get("best_complete_split_latency_ms") or ""),
            str(r.get("best_pipeline_cycle_ms") or ""),
            str(r.get("best_pipeline_fps") or ""),
            str(r.get("pipeline_speedup_vs_tensorrt_full") or ""),
            str(r.get("best_complete_split_quality") or ""),
        ]) + " |")
    md_lines.extend([
        "",
        "## Pipeline throughput leaders (all split rows)",
        "",
        "These rows rank steady-state interleaved throughput. `Split latency` is the single-frame sequential latency, while `Cycle`/`FPS` estimate or measure the pipeline bottleneck under streaming.",
        "",
        "| Model | Backend | Case | Split latency [ms] | Pipeline cycle [ms] | Pipeline FPS | Speedup vs TensorRT full | Quality |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ])
    pipe_leaders = sorted([r for r in backend_speedup_rows if _float(r.get("best_pipeline_cycle_ms")) is not None], key=lambda r: (str(r.get("model_id") or ""), float(_float(r.get("best_pipeline_cycle_ms")) or 1e18)))
    for r in pipe_leaders[:60]:
        md_lines.append("| " + " | ".join([
            str(r.get("model_id") or ""),
            str(r.get("backend_label") or r.get("backend") or ""),
            str(r.get("best_complete_split_case") or ""),
            str(r.get("best_complete_split_latency_ms") or ""),
            str(r.get("best_pipeline_cycle_ms") or ""),
            str(r.get("best_pipeline_fps") or ""),
            str(r.get("pipeline_speedup_vs_tensorrt_full") or ""),
            str(r.get("best_complete_split_quality") or ""),
        ]) + " |")
    md_lines.extend([
        "",
        "## Thesis metrics",
        "",
        "| Model | Top-1 hit | Top-k hit | Regret mean [%] | Rank corr. | Speedup error mean [%] | Materialization gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in thesis_rows:
        md_lines.append("| " + " | ".join([
            str(r.get("model_id") or ""),
            str(r.get("top1_hit") or ""),
            str(r.get("topk_hit") or ""),
            str(r.get("mean_regret_pct") or ""),
            str(r.get("rank_correlation_spearman") or ""),
            str(r.get("speedup_error_mean_pct") or ""),
            f"accepted-outside={r.get('accepted_outside_prediction_count', '')}; predicted-unmeasured={r.get('predicted_unmeasured_count', '')}",
        ]) + " |")
    md_lines.extend([
        "",
        "## Next-run recommendations", "",
        "See `next_run_recommendations.csv` and `prediction_calibration.csv` for the machine-readable planning tables. The recommendation combines validation, TensorRT-speedup, prediction reliability and row-level energy completeness.", "",
        "| Model | Priority | Action | Case / backend | Speedup vs TRT | Energy OK |",
        "|---|---:|---|---|---:|---:|",
    ])
    for r in next_run_recommendation_rows[:20]:
        md_lines.append("| " + " | ".join([
            str(r.get("model_id") or ""),
            str(r.get("priority_score") or ""),
            str(r.get("recommended_action") or ""),
            f"{r.get('recommended_case','')} / {r.get('recommended_backend','')}",
            str(r.get("pipeline_speedup_vs_tensorrt_full") or ""),
            str(r.get("best_row_level_energy_ok") or ""),
        ]) + " |")
    md_lines.extend([
        "",
        "## Stage-2 contract summary",
        "",
        "| Model | Backend | Best case | Contract status | Probe samples | Pass ratio | Calibration source |",
        "|---|---|---|---|---:|---:|---|",
    ])
    for r in backend_speedup_rows:
        if str(r.get("stage2_contract_status") or "").strip() or "deepx" in str(r.get("backend") or "").lower():
            md_lines.append("| " + " | ".join([
                str(r.get("model_id") or ""),
                str(r.get("backend_label") or r.get("backend") or ""),
                str(r.get("best_complete_split_case") or ""),
                str(r.get("stage2_contract_status") or ""),
                str(r.get("stage2_contract_probe_samples") or ""),
                str(r.get("stage2_contract_pass_ratio") or ""),
                str(r.get("stage2_calibration_source") or ""),
            ]) + " |")
    md_lines.extend([
        "",
        "## Notes",
        "",
        "- Complete split metrics use only rows with composed latency or part1+part2(+transfer).",
        "- Full-baseline latency is not used as split latency.",
        "- Hailo evidence is separated into device/service/import/HEF/runtime so reports do not imply that a HEF was executed unless a runtime row proves it.",
        "- Accepted cases can differ from prediction Top-k because the BenchmarkSet generator may promote/backfill feasible policy cases; those gaps are kept visible in `prediction_vs_benchmark.csv` and `thesis_metrics.csv`.",
        "",
    ])
    p_dashboard_md = write_text(reports / "result_dashboard.md", "\n".join(md_lines))

    p_model_tex = _write_latex_table(
        tables / "model_selection.tex",
        caption="Evaluation model selection and measured coverage.",
        label="model_selection",
        headers=["Model", "Task", "Accepted", "Complete splits", "Validation", "Hardware"],
        rows=[[m.get("model_id", ""), m.get("task", ""), m.get("accepted_case_count", ""), m.get("complete_split_count", ""), m.get("validation_status", ""), m.get("hardware_smoke_status", "")] for m in model_cards],
    )
    p_pred_tex = _write_latex_table(
        tables / "prediction_accuracy.tex",
        caption="Prediction-vs-benchmark accuracy metrics.",
        label="prediction_accuracy",
        headers=["Model", "Top-1", "Top-k", "Mean regret [\\%]", "Rank corr.", "Accepted cases"],
        rows=[[r.get("model_id", ""), r.get("top1_hit", ""), r.get("topk_hit", ""), r.get("mean_regret_pct", ""), r.get("rank_correlation_spearman", ""), r.get("accepted_case_count", "")] for r in thesis_rows],
    )
    p_speed_tex = _write_latex_table(
        tables / "backend_speedups.tex",
        caption="Backend split latency and pipeline-throughput summary.",
        label="backend_speedups",
        headers=["Model", "Backend", "Split latency [ms]", "Pipeline cycle [ms]", "Pipeline FPS", "Speedup vs TRT cycle", "Measured"],
        rows=[[r.get("model_id", ""), r.get("backend_label", r.get("backend", "")), r.get("best_complete_split_latency_ms", ""), r.get("best_pipeline_cycle_ms", ""), r.get("best_pipeline_fps", ""), r.get("pipeline_speedup_vs_tensorrt_full", ""), r.get("complete_split_measured", "")] for r in backend_speedup_rows if r.get("complete_split_measured")][:40],
    )

    # v55r: paper-ready pipeline throughput tables. These are the primary
    # throughput metrics for the interleaved split architecture.
    # Keep the legacy file name for downstream notebooks, but make the primary
    # pipeline table heterogeneous-only. Same-backend diagnostics are exported
    # separately as `same_backend_split_diagnostics.csv`.
    pipeline_rows = [dict(r) for r in backend_speedup_rows if _float(r.get("best_pipeline_cycle_ms")) is not None and _row_is_heterogeneous_accelerator_split(r)]
    p_pipeline_csv = write_csv(reports / "pipeline_throughput.csv", pipeline_rows, [
        "model_id", "backend", "backend_label", "best_pipeline_split_case",
        "best_complete_split_latency_ms", "best_pipeline_cycle_ms", "best_pipeline_fps",
        "pipeline_speedup_vs_tensorrt_full", "best_complete_split_quality",
        "stage2_contract_status", "stage2_calibration_source",
    ])
    p_pipeline_tex = _write_latex_table(
        tables / "pipeline_throughput.tex",
        caption="Steady-state pipeline throughput for interleaved split execution.",
        label="pipeline_throughput",
        headers=["Model", "Backend", "Case", "Latency [ms]", "Cycle [ms]", "FPS", "Speedup vs TRT"],
        rows=[[r.get("model_id", ""), r.get("backend_label", r.get("backend", "")), r.get("best_pipeline_split_case", r.get("best_complete_split_case", "")), r.get("best_complete_split_latency_ms", ""), r.get("best_pipeline_cycle_ms", ""), r.get("best_pipeline_fps", ""), r.get("pipeline_speedup_vs_tensorrt_full", "")] for r in pipeline_rows[:60]],
    )

    # visual_artifacts was generated before model-card aggregation so its FAIL/PASS
    # counts can be merged into validation_summary.csv.
    figure_paths = _try_write_figures(reports, model_cards, thesis_rows, backend_speedup_rows, pred_rows)
    figure_paths += _try_write_claim_figures(reports, energy_claim_rows)
    artifacts = {
        "result_dashboard_json": p_dashboard_json,
        "result_dashboard_md": p_dashboard_md,
        "result_dashboard_csv": p_dashboard_csv,
        "thesis_metrics_csv": p_thesis_csv,
        "thesis_metrics_json": p_thesis_json,
        "backend_speedups_csv": p_backend_csv,
        "pipeline_throughput_csv": p_pipeline_csv,
        "pipeline_throughput_tex": p_pipeline_tex,
        "full_backend_throughput_csv": p_full_throughput_csv,
        "full_backend_throughput_tex": p_full_throughput_tex,
        "heterogeneous_pipeline_throughput_csv": p_hetero_pipeline_csv,
        "heterogeneous_pipeline_throughput_tex": p_hetero_pipeline_tex,
        "same_backend_split_diagnostics_csv": p_same_backend_diag_csv,
        "energy_summary_csv": p_energy_summary_csv,
        "energy_pipeline_csv": p_energy_pipeline_csv,
        "energy_dispatch_summary_csv": p_energy_dispatch_csv,
        "energy_target_summary_csv": p_energy_target_csv,
        "energy_status_json": p_energy_status_json,
        "energy_claim_summary_csv": p_energy_claim_csv,
        "energy_claim_summary_md": p_energy_claim_md,
        "energy_claim_summary_tex": p_energy_claim_tex,
        "thesis_best_summary_tex": p_thesis_best_tex,
        "claim_table_comprehensive_csv": claim_analysis_reports.get("comprehensive_csv"),
        "claim_table_comprehensive_json": claim_analysis_reports.get("comprehensive_json"),
        "claim_table_comprehensive_md": claim_analysis_reports.get("comprehensive_md"),
        "claim_table_comprehensive_tex": claim_analysis_reports.get("comprehensive_tex"),
        "claim_table_full_models_csv": claim_analysis_reports.get("full_models_csv"),
        "claim_table_best_splits_csv": claim_analysis_reports.get("best_splits_csv"),
        "claim_table_best_splits_tex": claim_analysis_reports.get("best_splits_tex"),
        "next_run_recommendations_csv": p_next_run_csv,
        "next_run_recommendations_json": p_next_run_json,
        "next_run_recommendations_md": p_next_run_md,
        "prediction_calibration_csv": p_prediction_calibration_csv,
        "model_selection_tex": p_model_tex,
        "prediction_accuracy_tex": p_pred_tex,
        "backend_speedups_tex": p_speed_tex,
    }
    for key, value in visual_artifacts.items():
        if isinstance(value, Path):
            artifacts[key] = value
    for idx, fig in enumerate(figure_paths):
        artifacts[f"figure_{idx}_{fig.stem}"] = fig
    for idx, fig in enumerate(list(claim_analysis_reports.get("figures") or [])):
        if isinstance(fig, Path):
            artifacts[f"claim_figure_{idx}_{fig.stem}"] = fig

    return {
        "artifacts": artifacts,
        "overview": overview,
        "native_evidence_status": native_evidence,
        "native_evidence_summary": native_evidence_summary,
        "model_count": len(model_cards),
        "thesis_metric_rows": len(thesis_rows),
        "backend_speedup_rows": len(backend_speedup_rows),
        "full_backend_throughput_rows": len(full_backend_rows),
        "heterogeneous_pipeline_rows": len(hetero_pipeline_rows),
        "same_backend_diagnostic_rows": len(same_backend_diag_rows),
        "energy_claim_rows": len(energy_claim_rows),
        "claim_table_rows": len(claim_analysis_reports.get("rows") or []),
        "claim_table_full_rows": len(claim_analysis_reports.get("full_rows") or []),
        "claim_table_best_split_rows": len(claim_analysis_reports.get("best_split_rows") or []),
        "next_run_recommendation_rows": len(next_run_recommendation_rows),
        "prediction_calibration_rows": len(prediction_calibration_rows),
        "figure_count": len(figure_paths) + len(claim_analysis_reports.get("figures") or []),
        "claim_table_figure_count": len(claim_analysis_reports.get("figures") or []),
        "visual_verification_count": int(visual_artifacts.get("visual_verification_count") or 0),
        "canonical_full_baselines_csv": p_canonical_full,
        "benchmark_input_policy_csv": p_input_policy_csv,
    }
