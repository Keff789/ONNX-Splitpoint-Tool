from __future__ import annotations

"""Native Split/Full energy ingestion and setup-local comparison reporting."""

from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json
from .accuracy_reporting import assessment_fields
import math
import re
import statistics
from statistics import NormalDist

from .energy.config import ENERGY_PRIMARY_METHOD, ENERGY_SHADOW_METHOD
from .energy.comparison import resolve_energy_comparison
from .preprocessing_contract import preprocessing_contract_sha256
from .native_energy_quality_admission import (
    verify_sealed_energy_quality_admission,
)
from .workflow.evidence_status import project_native_evidence_status
from .workflow.status_reporting import energy_axis_description


# Frozen claim-level pairing rule.  The Full workload converts the common
# target duration to a fixed work-unit count, while Split runs duration-based.
# The measured active windows may therefore differ slightly, but energy-ratio
# claims are admitted only when their absolute difference is at most 5 % of
# the explicitly recorded common target duration.
NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE = 0.05
NATIVE_ENERGY_ACTIVE_DURATION_COMPARISON_POLICY = (
    "abs(split_active_duration_s-baseline_active_duration_s)"
    "/common_target_duration_s<=relative_tolerance"
)
NATIVE_ENERGY_ACTIVE_DURATION_TOLERANCE_SOURCE = (
    "native_energy_reporting.NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE"
)


def _num(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _final_energy_quality_result_fields(
    *,
    plan_qualified: bool,
    admission_verified: bool,
    measurement_started: bool,
    raw_energy_collected: bool,
) -> dict[str, Any]:
    """Clamp labels to both physical acquisition and verified admission."""

    if not measurement_started:
        return {
            "energy_quality_qualified": False,
            "energy_quality_status": "energy_not_collected",
            "native_energy_after_technical_error": (
                "not_collected_measurement_not_started"
            ),
        }
    if not raw_energy_collected:
        return {
            "energy_quality_qualified": False,
            "energy_quality_status": (
                "collection_started_raw_energy_unavailable"
            ),
            "native_energy_after_technical_error": (
                "collection_started_raw_energy_unavailable"
            ),
        }

    qualified = bool(plan_qualified and admission_verified)
    return {
        "energy_quality_qualified": qualified,
        "energy_quality_status": (
            "quality_qualified"
            if qualified else "raw_energy_quality_not_qualified"
        ),
        "native_energy_after_technical_error": (
            "not_applicable_quality_qualified"
            if qualified else "collect_raw_quality_unqualified"
        ),
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


_ENERGY_KEYS = {
    "avg_power_w",
    "mean_power_w",
    "energy_total_j",
    "avg_energy_total_j",
    "avg_energy_per_inference_j",
    "avg_energy_per_work_unit_j",
    "avg_energy_dynamic_j",
    "avg_host_normalized_energy_est_j",
    "avg_host_normalized_energy_per_work_unit_est_j",
    "avg_host_normalized_work_units_per_j_est",
    "avg_host_normalized_average_power_est_w",
    "host_normalized_energy_per_work_unit_est_j",
    "host_normalized_average_power_est_w",
    "host_normalized_energy_per_work_unit_est_j_sample_stddev",
    "host_normalized_energy_per_work_unit_est_j_ci_low",
    "host_normalized_energy_per_work_unit_est_j_ci_high",
    "host_normalization_role",
    "host_normalization_source_run_id",
    "host_normalization_target_variant",
    "host_normalization_identity_verified",
    "accelerator_idle_correction_requested",
    "accelerator_idle_correction_applied",
    "accelerator_idle_correction_status",
    "accelerator_idle_correction_statuses",
    "accelerator_idle_w_applied",
    "accelerator_idle_calibration_verified",
    "accelerator_idle_calibration_status",
    "accelerator_idle_calibration_binding_path",
    "accelerator_idle_calibration_binding_sha256",
    "accelerator_idle_calibration_evidence",
    "accelerator_idle_calibrated_at",
    "energy_j",
    "energy_per_inference_j",
    "joules_per_inference",
    "energy_per_work_unit_j",
    "energy_work_units_used",
    "energy_configured_work_units",
    "inference_count",
    "work_units",
    "measurement_duration_s",
    "active_duration_s",
    "energy_measured_workload_duration_s",
    "energy_work_units_source",
    "energy_work_unit_sources",
    "runtime_completed_work_unit_run_count",
    "run_count",
    "valid_postprocessed_runs",
    "avg_active_duration_s",
    "avg_energy_work_units_used",
    "energy_efficiency_claim_eligible",
    "energy_efficiency_source",
    "energy_physical_scope",
    "energy_physical_scope_canonical",
    "energy_window_requested",
    "energy_window_effective",
    "energy_window_effective_values",
    "energy_window_alignment_status",
    "energy_calibration_manifest",
    "energy_calibration_sha256",
    "energy_calibration_verification",
    "full_system_scope_calibration_status",
    "full_system_current_scale_verification",
    "full_system_current_scale_configured",
    "full_system_current_scale_applicable",
    "full_system_current_scale_verified",
    "full_system_current_scale_verification_status",
    "full_system_current_scale_factor_configured",
    "full_system_current_scale_calibration_evidence",
    "full_system_current_scale_calibration_sha256_expected",
    "full_system_current_scale_calibration_sha256_actual",
    "full_system_current_scale_verification_errors",
    "full_system_current_scale_applied",
    "full_system_current_scale_applied_run_count",
    "full_system_current_scale_factor_applied",
    "energy_primary_metric",
    "energy_calibrated_input_unsubtracted",
    "energy_raw_primary",
    "final_energy_gate_status",
    "final_energy_gate_failure_count",
    "collector_rc",
    "workload_command_rc",
    "workload_timing",
    "pipeline_fps_selected",
    "pipeline_fps_per_watt",
    "pipeline_fps_per_watt_from_selected_fps",
    "postprocess_status",
    "source_key",
    "scientific_primary_method_frozen",
    "scientific_primary_method",
    "scientific_primary_energy_status",
    "scientific_primary_valid_run_count",
    "scientific_primary_claim_eligible",
    "scientific_primary_energy_total_j",
    "scientific_primary_energy_per_work_unit_j",
    "scientific_primary_active_duration_s",
    "scientific_primary_avg_power_w",
    "scientific_shadow_method",
    "scientific_shadow_role",
    "scientific_shadow_energy_status",
    "scientific_shadow_valid_run_count",
    "scientific_shadow_claim_eligible",
    "scientific_shadow_affects_primary_claim",
    "scientific_shadow_affects_primary_result",
    "scientific_shadow_affects_final_gate",
    "shadow_role",
    "chapter4_legacy_shadow_status",
    "chapter4_legacy_shadow_energy_j",
    "chapter4_legacy_shadow_energy_total_j",
    "chapter4_legacy_shadow_energy_per_work_unit_j",
    "chapter4_legacy_shadow_active_duration_s",
    "chapter4_legacy_shadow_avg_power_w",
    "candidate_method",
    "candidate_role",
    "candidate_eligible_for_auto_switch",
    "candidate_v263_shadow_energy_total_j",
    "candidate_v263_shadow_energy_per_work_unit_j",
    "confidence_level",
    "effective_run_count",
    "requested_run_count",
    "raw_postprocessed_run_count",
    "legacy_window_comparison_requested",
    "legacy_window_comparison_attempted_runs",
    "legacy_window_comparison_successful_runs",
    "scientific_primary_energy_statistics",
    "chapter4_legacy_shadow_statistics",
    "candidate_v263_shadow_statistics",
    "window_method_comparison_statistics",
    "legacy_minus_command_window_energy_percent_mean",
    "legacy_minus_command_window_duration_percent_mean",
    "diagnostic_only",
    "claim_exclusion_reason",
    "scientific_claim_exclusion_reasons",
}


_MARKER_METHOD_ALIASES = {
    ENERGY_PRIMARY_METHOD,
    "candidate_v263",
    "collector_sample_marker_crop",
}


def _canonical_energy_method(value: Any) -> str:
    """Canonicalise current aliases without rewriting historical role data."""
    method = str(value or "").strip()
    if method in _MARKER_METHOD_ALIASES:
        return ENERGY_PRIMARY_METHOD
    if method in {ENERGY_SHADOW_METHOD, "chapter4_baseline"}:
        return ENERGY_SHADOW_METHOD
    return method


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "ok", "pass", "passed", "claim_ok", "eligible"}


def _calibrated_unsubtracted_input_primary(value: Mapping[str, Any]) -> bool:
    """Accept the explicit v2 semantic and the legacy compatibility label."""
    metric = str(value.get("energy_primary_metric") or "").strip().lower()
    if metric == "calibrated_input_energy_unsubtracted":
        return value.get("energy_calibrated_input_unsubtracted") is True
    return metric == "raw_input_energy" and value.get("energy_raw_primary") is True


def _first(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "", [], {})), "")


def _canonical_scope(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"fs", "full_system", "fullsystem", "system_input"}:
        return "full_system"
    if text in {"mb", "mainboard", "motherboard"}:
        return "MB"
    return text or "unknown"


def _canonical_window(value: Any) -> str:
    if isinstance(value, list):
        values = {str(item or "").strip().lower() for item in value if str(item or "").strip()}
        if len(values) == 1:
            value = next(iter(values))
        elif values:
            return "mixed"
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"command", "command_energy", "command_window"}:
        return "command_window"
    if text in {"trim", "trimmed", "trimmed_activity_window", "default_trimmed_detection"}:
        return "trimmed_activity_window"
    if text in {"complete", "complete_window", "complete_measurement_window"}:
        return "complete_measurement_window"
    return text or "unknown"


def _infer_task(model: str, explicit: Any) -> tuple[str, str]:
    if str(explicit or "").strip():
        return str(explicit).strip().lower(), "declared"
    text = str(model or "").lower()
    if any(token in text for token in ("yolo", "ssd", "retina", "detect")):
        return "detection", "model_id_inference"
    if any(token in text for token in ("resnet", "regnet", "imagenet", "mobilenet", "efficientnet")):
        return "classification", "model_id_inference"
    return "unknown", "unavailable"


def _canonical_direction(value: Any, backend: str = "", comparison_backend: str = "") -> str:
    text = str(_first(value, comparison_backend, backend) or "").strip().lower().replace("-", "_")
    aliases = {
        "hailo8": "hailo8_to_trt",
        "h8": "hailo8_to_trt",
        "native_full_hailo8": "hailo8_to_trt",
        "hailo10": "hailo10h_to_trt",
        "hailo10h": "hailo10h_to_trt",
        "native_full_hailo10": "hailo10h_to_trt",
        "native_full_hailo10h": "hailo10h_to_trt",
        "deepx": "deepx_to_trt",
        "native_full_deepx": "deepx_to_trt",
    }
    return aliases.get(text, text or "unknown")


def _canonical_runner(value: Any, backend: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text in {"native", "native_fifo", "fifo"}:
        return "native_fifo"
    if text in {"native_full", "native_full_baseline", "full"}:
        return "native_full"
    return "native_full" if backend.startswith("native_full_") else "native_fifo"


def _compact_failure_text(value: Any, *, limit: int = 500) -> str:
    """Return one bounded, single-line diagnostic suitable for report tables."""
    if isinstance(value, (list, tuple, set)):
        text = ";".join(str(item or "").strip() for item in value if str(item or "").strip())
    else:
        text = str(value or "").strip()
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def _measurement_failure_reason(
    item: Mapping[str, Any],
    run_info: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    execution_ok: bool,
) -> str:
    if execution_ok:
        return ""
    candidates = (
        item.get("error"),
        item.get("skipped"),
        run_info.get("energy_aggregate_embed_error"),
        run_info.get("energy_aggregate_validation_errors"),
        run_info.get("error"),
        item.get("failure_category"),
        run_info.get("stderr_tail"),
        run_info.get("stdout_tail"),
    )
    for candidate in candidates:
        text = _compact_failure_text(candidate)
        if text:
            return text
    postprocess = str(payload.get("postprocess_status") or "").strip()
    final_gate = str(payload.get("final_energy_gate_status") or "").strip()
    aggregate = str(
        (payload.get("_native_energy_aggregate") or {}).get("status")
        if isinstance(payload.get("_native_energy_aggregate"), Mapping)
        else ""
    ).strip()
    diagnostic = ";".join(
        value
        for value in (
            f"postprocess_status={postprocess}" if postprocess else "",
            f"final_energy_gate_status={final_gate}" if final_gate else "",
            f"energy_aggregate_status={aggregate}" if aggregate else "",
        )
        if value
    )
    return diagnostic or "measurement_execution_failed"


def _calibration_fields(payload: Mapping[str, Any], plan: Mapping[str, Any]) -> tuple[str, str, bool, str]:
    # The collector uses a configured, verified and applied FS input scale in
    # preference to the older optional method-calibration manifest. Preserve
    # that contract when importing its receipt; do not demand both formats.
    scope = _canonical_scope(_first(
        payload.get("energy_physical_scope_canonical"),
        payload.get("energy_physical_scope"),
    ))
    if scope == "full_system" and payload.get("full_system_current_scale_configured") is True:
        manifest = str(payload.get("full_system_current_scale_calibration_evidence") or "")
        expected = _strict_sha256_token(payload.get("full_system_current_scale_calibration_sha256_expected"))
        digest = _strict_sha256_token(payload.get("full_system_current_scale_calibration_sha256_actual"))
        factor = _num(payload.get("full_system_current_scale_factor_configured"))
        applied_factor = _num(payload.get("full_system_current_scale_factor_applied"))
        errors = payload.get("full_system_current_scale_verification_errors")
        receipt = payload.get("full_system_current_scale_verification")
        receipt_matches = not isinstance(receipt, Mapping) or all(
            payload.get(key) == value for key, value in receipt.items()
            if key.startswith("full_system_current_scale_") and key in payload
        )
        verified = bool(
            payload.get("full_system_current_scale_applicable") is True
            and payload.get("full_system_current_scale_verified") is True
            and payload.get("full_system_current_scale_verification_status") == "verified"
            and payload.get("full_system_scope_calibration_status") == "pass"
            and manifest and expected and digest == expected
            and factor is not None and math.isfinite(factor) and factor > 0
            and isinstance(errors, list) and not errors and receipt_matches
        )
        if not verified:
            return manifest, digest, False, "full_system_current_scale_verification_failed"
        applied_count = _num(payload.get("full_system_current_scale_applied_run_count"))
        valid_count = _num(payload.get("valid_postprocessed_runs"))
        applied = bool(
            payload.get("full_system_current_scale_applied") is True
            and applied_factor == factor
            and (applied_count is None or (
                valid_count is not None and valid_count > 0 and applied_count == valid_count
            ))
        )
        if not applied:
            return manifest, digest, False, "full_system_current_scale_not_applied"
        return manifest, digest, True, "verified_full_system_current_scale"
    verification = payload.get("energy_calibration_verification")
    verification = verification if isinstance(verification, Mapping) else {}
    manifest = str(_first(verification.get("path"), payload.get("energy_calibration_manifest"), plan.get("energy_calibration_manifest")) or "")
    digest = str(_first(verification.get("actual_sha256"), payload.get("energy_calibration_sha256"), plan.get("energy_calibration_sha256")) or "").lower()
    verified = verification.get("verified") is True
    scope_status = str(
        payload.get("full_system_scope_calibration_status") or ""
    ).strip()
    status = (
        scope_status
        if scope_status in {"not_full_system", "not_required"}
        else str(verification.get("status") or scope_status or "missing")
    )
    return manifest, digest, verified, status


def _normalized_calibration_axes(
    measured_scope: str,
    *,
    verified: bool,
    status: str,
) -> tuple[bool, str, str]:
    """Keep MB primary-input provenance separate from FS calibration."""

    normalized = str(status or "missing")
    if measured_scope == "full_system":
        required = True
        if not verified and normalized in {
            "", "missing", "not_full_system", "not_required",
            "not_applicable_mb",
        }:
            normalized = "missing_required_full_system_calibration"
        external_status = normalized
    elif measured_scope == "MB":
        required = False
        if not verified and normalized in {
            "", "missing", "not_full_system", "not_required",
        }:
            normalized = "not_applicable_mb"
        external_status = "not_required_non_full_system"
    else:
        # An unknown physical scope must never be silently re-labelled MB.
        # Requiring calibration fail-closed keeps an unresolved scope out of
        # scientific-primary availability until its provenance is explicit.
        required = True
        if not verified:
            normalized = "scope_unknown_calibration_unresolved"
        external_status = normalized
    return required, normalized, external_status


def _exact_runtime_count(payload: Mapping[str, Any]) -> bool:
    source = str(payload.get("energy_work_units_source") or "")
    sources = payload.get("energy_work_unit_sources")
    if isinstance(sources, list) and sources:
        source_ok = all(str(item) == "runtime_completed_work_units" for item in sources)
    else:
        source_ok = source == "runtime_completed_work_units"
    completed = _num(payload.get("runtime_completed_work_unit_run_count"))
    repeats = _num(payload.get("valid_postprocessed_runs") or payload.get("run_count"))
    if completed is not None and repeats is not None:
        return source_ok and completed > 0 and completed == repeats
    return source_ok


def _walk_scalars(value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in _ENERGY_KEYS and item not in (None, ""):
                out[str(key)] = item
            _walk_scalars(item, out)
    elif isinstance(value, list):
        for item in value:
            _walk_scalars(item, out)


def _project_authoritative_aggregate(
    aggregate: Mapping[str, Any], out: dict[str, Any],
) -> None:
    """Project collector aggregate fields without descending into repeats.

    Bounded stdout is necessarily best-effort and may contain only a nested
    final repeat.  Once an exact aggregate has been embedded, however, its
    top-level values are the scientific contract.  Recursing through
    ``aggregate["runs"]`` would allow the last repeat to overwrite aggregate
    counts and statuses (for example, reporting 3/3 when the aggregate says
    2/3).  Preserve top-level statistics mappings as mappings and only use the
    top-level A/B policy as a compatibility source for its requested/effective
    counts when those counts are not themselves present at the top level.
    """
    for key, item in aggregate.items():
        name = str(key)
        if name in _ENERGY_KEYS and item not in (None, ""):
            out[name] = item
    ab_policy = aggregate.get("energy_window_method_ab")
    if isinstance(ab_policy, Mapping):
        for name in ("requested_run_count", "effective_run_count"):
            if name not in aggregate and ab_policy.get(name) not in (None, ""):
                out[name] = ab_policy.get(name)
    requested = _num(
        aggregate.get("requested_valid_repeat_count")
        or aggregate.get("run_count")
        or aggregate.get("effective_run_count")
    )
    valid = _num(aggregate.get("valid_postprocessed_runs"))
    repeat_contract_declared = (
        "repeat_contract_complete" in aggregate
    )
    repeat_count_incomplete = bool(
        requested is not None
        and requested > 0
        and valid is not None
        and valid < requested
    )
    repeat_contract_failed = bool(
        (
            repeat_contract_declared
            and aggregate.get("repeat_contract_complete") is not True
        )
        or repeat_count_incomplete
    )
    if repeat_contract_failed:
        out["final_energy_gate_status"] = "fail"
        if valid is not None and valid > 0:
            out["postprocess_status"] = "incomplete_valid_repeats"
        elif str(aggregate.get("postprocess_status") or "") == "ok":
            out["postprocess_status"] = (
                "acquisition_integrity_failed_no_valid_repeats"
            )


def _parse_json_fragments(text: str) -> list[Any]:
    """Decode any complete JSON objects embedded in a truncated stdout tail."""
    decoder = json.JSONDecoder()
    values: list[Any] = []
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        values.append(value)
    return values


def _regex_scalars(text: str, out: dict[str, Any]) -> None:
    # The energy CLI output can be tail-truncated before the opening brace.  The
    # scalar fields are still recoverable because JSON keys and numeric values
    # remain intact in the tail.
    for key in _ENERGY_KEYS:
        pattern = re.compile(r'"' + re.escape(key) + r'"\s*:\s*("(?:[^"\\]|\\.)*"|true|false|null|[-+0-9.eE]+)')
        matches = pattern.findall(text)
        if not matches:
            continue
        raw = matches[-1]
        try:
            out[key] = json.loads(raw)
        except Exception:
            out[key] = raw.strip('"')


def _energy_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    _walk_scalars(item, out)
    run = item.get("run") if isinstance(item.get("run"), Mapping) else {}
    for field in ("stdout", "stdout_tail", "stderr", "stderr_tail"):
        text = str(run.get(field) or "")
        if not text:
            continue
        for value in _parse_json_fragments(text):
            _walk_scalars(value, out)
        _regex_scalars(text, out)
    aggregate = _energy_aggregate(item)
    if aggregate:
        # The compact aggregate is authoritative. Apply it after truncated
        # stdout fragments so means/CIs cannot be overwritten by the last
        # individual repeat seen in the bounded log tail.
        _project_authoritative_aggregate(aggregate, out)
        out["_native_energy_aggregate"] = aggregate
    return out


def _aggregate_score(value: Mapping[str, Any]) -> int:
    return sum(
        1 for key in (
            "valid_postprocessed_runs", "scientific_primary_energy_statistics",
            "chapter4_legacy_shadow_statistics", "candidate_v263_shadow_statistics",
            "window_method_comparison_statistics",
            "avg_energy_total_j", "run_count",
        ) if key in value
    )


def _energy_aggregate(item: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the compact collector aggregate without relying on log order."""
    run = item.get("run") if isinstance(item.get("run"), Mapping) else {}
    embedded = run.get("energy_aggregate") if isinstance(run.get("energy_aggregate"), Mapping) else {}
    if embedded and _aggregate_score(embedded) >= 2:
        return dict(embedded)

    aggregate_path = str(run.get("energy_aggregate_path") or "").strip()
    if not aggregate_path:
        command = run.get("cmd")
        if isinstance(command, list):
            try:
                index = [str(part) for part in command].index("--out")
                aggregate_path = str(Path(str(command[index + 1])).expanduser() / "energy_aggregate.json")
            except Exception:
                aggregate_path = ""
    if aggregate_path:
        value = _load(Path(aggregate_path).expanduser())
        if value and _aggregate_score(value) >= 2:
            return value

    candidates: list[dict[str, Any]] = []
    for field in ("stdout", "stdout_tail", "stderr", "stderr_tail"):
        for value in _parse_json_fragments(str(run.get(field) or "")):
            if isinstance(value, Mapping) and _aggregate_score(value) >= 2:
                candidates.append(dict(value))
    return max(candidates, key=_aggregate_score) if candidates else {}


def _energy_repeat_records(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return complete per-repeat mappings still visible in bounded logs."""
    run = item.get("run") if isinstance(item.get("run"), Mapping) else {}
    best: dict[int, dict[str, Any]] = {}
    for field in ("stdout", "stdout_tail", "stderr", "stderr_tail"):
        for value in _parse_json_fragments(str(run.get(field) or "")):
            if not isinstance(value, Mapping) or value.get("run_index") is None:
                continue
            if not (
                isinstance(value.get("window_method_comparison"), Mapping)
                or value.get("postprocess_status") not in (None, "")
            ):
                continue
            try:
                index = int(value.get("run_index"))
            except Exception:
                continue
            row = dict(value)
            current = best.get(index)
            if current is None or len(row) > len(current):
                best[index] = row
    return [best[index] for index in sorted(best)]


def _repeat_statistics(values: Sequence[float], confidence_level: float = 0.95) -> dict[str, Any]:
    vals = [float(value) for value in values if _num(value) is not None]
    n = len(vals)
    result: dict[str, Any] = {
        "n": n,
        "confidence_level": float(confidence_level),
        "mean": statistics.fmean(vals) if vals else None,
        "sample_stddev": None,
        "standard_error": None,
        "ci_low": None,
        "ci_high": None,
        "method": "student_t_two_sided",
    }
    if n < 2:
        result["status"] = "single_valid_repeat" if n == 1 else "no_valid_repeats"
        return result
    stddev = statistics.stdev(vals)
    standard_error = stddev / math.sqrt(n)
    # Match the collector's dependency-free Student-t approximation closely
    # enough for an offline fallback when only repeat records remain.
    z = NormalDist().inv_cdf(0.5 + min(0.999, max(0.50, confidence_level)) / 2.0)
    df = float(max(1, n - 1))
    critical = z + (z**3 + z) / (4.0 * df) + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
    half_width = critical * standard_error
    result.update({
        "sample_stddev": stddev,
        "standard_error": standard_error,
        "ci_low": result["mean"] - half_width,
        "ci_high": result["mean"] + half_width,
        "ci_half_width": half_width,
        "critical_value": critical,
        "degrees_of_freedom": n - 1,
        "status": "ok",
    })
    return result


def _stats_block(value: Any, metric: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    block = value.get(metric)
    return dict(block) if isinstance(block, Mapping) else {}


def _fallback_ab_statistics(records: Sequence[Mapping[str, Any]], confidence_level: float) -> dict[str, Any]:
    def values(metric: str, field: str) -> list[float]:
        out: list[float] = []
        for record in records:
            comparison = record.get("window_method_comparison")
            comparison = comparison if isinstance(comparison, Mapping) else {}
            if str(comparison.get("status") or "") != "ok" or comparison.get("same_raw_trace_verified") is not True:
                continue
            delta = comparison.get("legacy_minus_command_window")
            delta = delta if isinstance(delta, Mapping) else {}
            metric_block = delta.get(metric)
            metric_block = metric_block if isinstance(metric_block, Mapping) else {}
            number = _num(metric_block.get(field))
            if number is not None:
                out.append(number)
        return out
    return {
        "diagnostic_only": True,
        "difference_direction": "legacy_minus_command_window",
        "energy_j": {"relative_percent": _repeat_statistics(values("energy_j", "relative_percent"), confidence_level)},
        "duration_s": {"relative_percent": _repeat_statistics(values("duration_s", "relative_percent"), confidence_level)},
        "average_power_w": {"relative_percent": _repeat_statistics(values("average_power_w", "relative_percent"), confidence_level)},
        "source": "bounded_repeat_record_fallback",
    }


def _execution_mode(backend: str, row: Mapping[str, Any]) -> str:
    explicit = str(row.get("execution_mode") or "").strip()
    if explicit:
        return explicit
    return "native_full_baseline" if backend.startswith("native_full_") else "native_split"


def _normalized_sha256(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _strict_sha256_token(value: Any) -> str:
    """Canonicalise a bare or singly-prefixed SHA-256 token only."""
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[len("sha256:"):]
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


def _native_validation_identity(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(row.get("backend") or "").strip().lower(),
        str(row.get("model") or row.get("model_id") or "").strip(),
        str(row.get("case") or row.get("case_id") or "full").strip(),
        str(row.get("precision") or "").strip().lower(),
        str(row.get("setup_id") or "").strip(),
        str(row.get("comparison_backend") or "").strip().lower(),
    )


def _native_validation_map(run: Path) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    path = (
        run / "reports" / "native_validation"
        / "native_producer_validation_summary.json"
    )
    payload = _load(path)
    out: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for value in list(payload.get("rows") or []):
        if isinstance(value, Mapping):
            row = dict(value)
            out.setdefault(_native_validation_identity(row), []).append(row)
    return out


def _validation_evidence_for_plan(
    plan: Mapping[str, Any],
    validations: Mapping[tuple[str, ...], Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], str]:
    matches = list(validations.get(_native_validation_identity(plan)) or [])
    if len(matches) == 1:
        return dict(matches[0]), "exact_unique"
    if len(matches) > 1:
        return {}, "ambiguous"
    return {}, "missing"


def _identity_value(
    plan: Mapping[str, Any], validation: Mapping[str, Any], *keys: str,
) -> tuple[str, bool]:
    """Resolve one exact identity field and report explicit conflicts."""
    plan_raw = [plan.get(key) for key in keys if str(plan.get(key) or "").strip()]
    validation_raw = [
        validation.get(key) for key in keys
        if str(validation.get(key) or "").strip()
    ]
    plan_values = {_strict_sha256_token(value) for value in plan_raw}
    validation_values = {_strict_sha256_token(value) for value in validation_raw}
    if (
        (plan_raw and "" in plan_values)
        or (validation_raw and "" in validation_values)
        or len(plan_values) > 1 or len(validation_values) > 1
    ):
        return "", True
    if plan_values and validation_values and plan_values != validation_values:
        return "", True
    values = plan_values or validation_values
    return (next(iter(values)) if values else ""), False


def _project_bound_preprocessing_pad(
    plan: dict[str, Any], validation: Mapping[str, Any], join_status: str,
) -> None:
    """Fill an absent reporting field only from the exact bound input contract."""
    if plan.get("prepared_feed_letterbox_pad_value") not in (None, ""):
        return
    if join_status != "exact_unique":
        return
    contract = validation.get("preprocessing_contract")
    if not isinstance(contract, Mapping):
        return
    contract_sha = preprocessing_contract_sha256(contract)
    for field in ("preprocessing_contract_sha256", "source_request_sha256", "model_sha256"):
        left = _strict_sha256_token(plan.get(field))
        if not left or left != _strict_sha256_token(validation.get(field)):
            return
    if contract_sha != plan["preprocessing_contract_sha256"]:
        return
    image_sha = _strict_sha256_token(plan.get("prepared_feed_source_image_sha256"))
    if not image_sha or image_sha != _strict_sha256_token(validation.get("input_image_sha256")):
        return
    if (contract.get("task") != plan.get("prepared_feed_task")
            or contract.get("preprocess_mode") != plan.get("prepared_feed_preprocess_mode")):
        return
    pad = contract.get("letterbox_pad_value", contract.get("pad_value"))
    if type(pad) not in (int, float) or not math.isfinite(pad) or not 0 <= pad <= 255:
        return
    if contract.get("pad_value", pad) != pad:
        return
    plan["prepared_feed_letterbox_pad_value"] = str(int(pad)) if int(pad) == pad else str(pad)
    plan["prepared_feed_identity_source"] = "exact_bound_validation_preprocessing_contract"


def _endpoint_id_from_validation(
    plan: Mapping[str, Any], validation: Mapping[str, Any], task: str,
) -> str:
    task = str(task or "").strip().lower()
    task_values = {
        str(source.get("task") or "").strip().lower()
        for source in (plan, validation)
        if str(source.get("task") or "").strip()
    }
    if task not in {"classification", "detection"} or (
        task_values and task_values != {task}
    ):
        return ""
    endpoint_hash, conflict = _identity_value(
        plan, validation, "endpoint_contract_hash",
    )
    stages = {
        str(source.get(field) or "").strip().lower()
        for source in (plan, validation)
        for field in ("stage", "contract_family")
        if str(source.get(field) or "").strip()
    }
    if len(stages) != 1:
        return ""
    stage = next(iter(stages))
    allowed_stages = {
        "classification": {"classification_logits", "classification_probabilities"},
        "detection": {"raw_head", "decoded_pre_nms", "decoded_nms"},
    }
    complete_values = [
        source.get("endpoint_contract_complete")
        for source in (plan, validation)
        if "endpoint_contract_complete" in source
    ]
    if (
        conflict or not endpoint_hash or stage not in allowed_stages[task]
        or not complete_values or any(value is not True for value in complete_values)
    ):
        return ""
    expected = f"{task}:{stage}:{endpoint_hash}"
    explicit_values = {
        str(source.get("output_endpoint_id") or "").strip().lower()
        for source in (plan, validation)
        if str(source.get("output_endpoint_id") or "").strip()
    }
    if explicit_values and explicit_values != {expected}:
        return ""
    return expected


def _endpoint_identity_from_explicit_fields(
    source: Mapping[str, Any],
    task: str,
    *,
    endpoint_field: str,
    hash_field: str,
    stage_fields: Sequence[str],
    complete_field: str,
    comparison: bool = False,
) -> tuple[str, str, str]:
    """Return one internally consistent endpoint identity or empty values."""
    task = str(task or "").strip().lower()
    endpoint_hash = _strict_sha256_token(source.get(hash_field))
    stages = {
        str(source.get(field) or "").strip().lower()
        for field in stage_fields
        if str(source.get(field) or "").strip()
    }
    endpoint_id = str(source.get(endpoint_field) or "").strip().lower()
    allowed_stages = {
        "classification": {
            "classification_logits", "classification_probabilities",
        },
        "detection": {"raw_head", "decoded_pre_nms", "decoded_nms"},
    }
    if (
        task not in allowed_stages
        or len(stages) != 1
        or source.get(complete_field) is not True
        or not endpoint_hash
    ):
        return "", "", ""
    stage = next(iter(stages))
    if stage not in allowed_stages[task]:
        return "", "", ""
    expected = (
        f"{task}:{stage}:comparison:{endpoint_hash}"
        if comparison else f"{task}:{stage}:{endpoint_hash}"
    )
    if endpoint_id != expected:
        return "", "", ""
    return expected, endpoint_hash, stage


def _physical_energy_endpoint(
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    task: str,
) -> dict[str, Any]:
    """Preserve the measured physical endpoint without using it for V2 pairing."""
    dual_identity_declared = any(
        key in plan
        for key in (
            "physical_output_endpoint_id",
            "physical_endpoint_contract_hash",
            "physical_endpoint_stage",
            "physical_endpoint_contract_complete",
        )
    )
    if dual_identity_declared:
        plan_id, plan_hash, plan_stage = (
            _endpoint_identity_from_explicit_fields(
                plan,
                task,
                endpoint_field="physical_output_endpoint_id",
                hash_field="physical_endpoint_contract_hash",
                stage_fields=("physical_endpoint_stage",),
                complete_field="physical_endpoint_contract_complete",
            )
        )
    else:
        legacy_id = _endpoint_id_from_validation(
            plan, validation, task,
        )
        legacy_hash = _strict_sha256_token(
            _first(
                plan.get("endpoint_contract_hash"),
                validation.get("endpoint_contract_hash"),
            )
        )
        legacy_stage = (
            legacy_id.split(":", 2)[1]
            if legacy_id.count(":") == 2 else ""
        )
        return {
            "physical_output_endpoint_id": legacy_id,
            "physical_endpoint_contract_hash": (
                legacy_hash if legacy_id else ""
            ),
            "physical_endpoint_stage": (
                legacy_stage if legacy_id else ""
            ),
            "physical_endpoint_contract_complete": bool(legacy_id),
            "physical_output_endpoint_match": bool(legacy_id),
            "physical_endpoint_identity_status": (
                "verified_legacy_plan_validation_resolution"
                if legacy_id else "unavailable"
            ),
        }

    validation_id = _endpoint_id_from_validation({}, validation, task)
    validation_hash = _strict_sha256_token(
        validation.get("endpoint_contract_hash")
    )
    validation_stage = (
        validation_id.split(":", 2)[1]
        if validation_id.count(":") == 2 else ""
    )
    validation_declared = bool(validation)
    physical_match = bool(
        plan_id
        and (
            not validation_declared
            or (
                validation_id == plan_id
                and validation_hash == plan_hash
                and validation_stage == plan_stage
            )
        )
    )
    return {
        "physical_output_endpoint_id": plan_id or validation_id,
        "physical_endpoint_contract_hash": plan_hash or validation_hash,
        "physical_endpoint_stage": plan_stage or validation_stage,
        "physical_endpoint_contract_complete": bool(
            (plan_id or validation_id) and physical_match
        ),
        "physical_output_endpoint_match": physical_match,
        "physical_endpoint_identity_status": (
            "verified_plan_validation_match"
            if validation_declared and physical_match
            else "verified_plan_only"
            if physical_match
            else "plan_validation_mismatch"
            if plan_id and validation_id
            else "unavailable"
        ),
    }


def _completed_detection_energy_endpoint(
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    *,
    physical_match: bool,
    fresh_completions: Sequence[Mapping[str, Any]] = (),
    energy_repeats: Sequence[Mapping[str, Any]] = (),
) -> tuple[str, str, str, str]:
    """Resolve V2 only when plan and Validation independently attest it."""
    if (
        plan.get("completion_pairing_eligible") is not True
        or plan.get("endpoint_contract_complete") is not True
        or plan.get("output_endpoint_match") is not True
        or not physical_match
        or str(plan.get("completion_pairing_status") or "").strip()
        not in {"strict_completed_detection_endpoint_verified", "strict_fast_postflight_completion_verified"}
    ):
        return "", "", "", "plan_completion_not_strictly_eligible"

    plan_id, plan_hash, plan_stage = (
        _endpoint_identity_from_explicit_fields(
            plan,
            "detection",
            endpoint_field="comparison_output_endpoint_id",
            hash_field="comparison_endpoint_contract_hash",
            stage_fields=("comparison_endpoint_stage",),
            complete_field="completion_pairing_eligible",
            comparison=True,
        )
    )
    if not plan_id:
        return "", "", "", "plan_comparison_endpoint_invalid"
    if (
        str(plan.get("output_endpoint_id") or "").strip().lower() != plan_id
        or _strict_sha256_token(plan.get("endpoint_contract_hash"))
        != plan_hash
        or str(plan.get("endpoint_stage") or "").strip().lower()
        != plan_stage
    ):
        return "", "", "", "plan_canonical_endpoint_alias_mismatch"

    if plan.get("completion_pairing_status") == "strict_fast_postflight_completion_verified":
        from .native_detection_postprocess import FrozenPostprocessError, verify_detection_completion_execution_contract
        from .native_three_stage import NativeThreeStageError, verify_fast_completion_attestation
        try:
            execution = verify_detection_completion_execution_contract(validation.get("completion_execution_contract"))
            verify_fast_completion_attestation(validation.get("completed_task_endpoint_attestation"), execution_contract=execution)
            source = execution["source_endpoint"]
            comparison = execution["comparison_endpoint_contract"]
            if (source.get("endpoint_contract_hash") != plan.get("physical_endpoint_contract_hash")
                or source.get("output_endpoint_id") != plan.get("physical_output_endpoint_id")
                or comparison.get("endpoint_contract_hash") != plan_hash
                or comparison.get("output_endpoint_id") != plan_id):
                raise ValueError("fast_completion_source_mismatch")
            repeats = {r.get("logical_repeat_index", r.get("run_index")): r for r in energy_repeats}
            if not repeats or len(repeats) != len(energy_repeats) or len(fresh_completions) != len(repeats):
                raise ValueError("fast_completion_repeats_missing")
            seen = set()
            for proof in fresh_completions:
                index = proof.get("logical_repeat_index", proof.get("run_index"))
                att = verify_fast_completion_attestation(proof.get("completion_execution_attestation"), execution_contract=execution)
                timing = proof.get("workload_timing") or {}
                if (index in seen or index not in repeats
                    or proof.get("status") != "fresh_energy_completion_nonce_count_and_window_verified"
                    or not proof.get("preflight_nonce") or not _strict_sha256_token(proof.get("stdout_sha256"))
                    or proof.get("comparison_output_endpoint_id") != plan_id
                    or proof.get("completed_work_units") != repeats[index].get("energy_work_units_used")
                    or att.get("completed_work_units") != proof.get("completed_work_units")
                    or timing.get("status") != "ok" or timing.get("rc") != 0
                    or not isinstance(timing.get("start_ns"), int) or not isinstance(timing.get("end_ns"), int)
                    or timing["end_ns"] <= timing["start_ns"]):
                    raise ValueError("fast_completion_repeat_binding_mismatch")
                seen.add(index)
        except (FrozenPostprocessError, NativeThreeStageError, ValueError, TypeError, KeyError):
            return "", "", "", "validation_fast_completion_evidence_invalid"
        return plan_id, plan_hash, plan_stage, "strict_plan_validation_fresh_energy_completion_match"

    validation_hash = _strict_sha256_token(
        validation.get(
            "completed_task_comparison_endpoint_contract_hash"
        )
    )
    validation_id = str(
        validation.get(
            "completed_task_comparison_output_endpoint_id"
        ) or ""
    ).strip().lower()
    validation_stages = {
        str(validation.get(field) or "").strip().lower()
        for field in (
            "completed_task_stage", "completed_task_contract_family",
        )
        if str(validation.get(field) or "").strip()
    }
    completion_mode = str(
        validation.get("completed_task_completion_mode") or ""
    ).strip()
    attestation = validation.get("completed_task_endpoint_attestation")
    attestation = (
        dict(attestation) if isinstance(attestation, Mapping) else {}
    )
    nested_hash = _strict_sha256_token(
        attestation.get(
            "completed_task_comparison_endpoint_contract_hash"
        )
    )
    nested_id = str(
        attestation.get(
            "completed_task_comparison_output_endpoint_id"
        ) or ""
    ).strip().lower()
    nested_mode = str(
        attestation.get("completed_task_completion_mode") or ""
    ).strip()
    top_status = str(
        validation.get("completed_task_endpoint_attestation_status") or ""
    ).strip().lower()
    supported_modes = {
        "frozen_host_tail",
        "integrated_accelerator_plus_frozen_normalization",
        "detection_completion_execution_v1",
    }
    if completion_mode == "detection_completion_execution_v1":
        from .native_detection_postprocess import (
            FrozenPostprocessError,
            verify_detection_completion_execution_contract,
            verify_detection_completion_execution_attestation,
        )
        try:
            execution = verify_detection_completion_execution_contract(
                validation.get("completion_execution_contract")
            )
            verified = verify_detection_completion_execution_attestation(
                attestation, execution_contract=execution,
                expected_observation_relation="same_hotloop_sentinel",
            )
            source = execution.get("source_endpoint") or {}
            if (
                verified.get("exact_result_claim_bound") is not True
                or source.get("endpoint_contract_hash") != plan.get("physical_endpoint_contract_hash")
                or source.get("output_endpoint_id") != plan.get("physical_output_endpoint_id")
            ):
                return "", "", "", "validation_completion_source_mismatch"
        except (FrozenPostprocessError, TypeError, ValueError):
            return "", "", "", "validation_completion_attestation_invalid"
    comparison_contract = validation.get(
        "completed_task_comparison_endpoint_contract"
    )
    comparison_contract = (
        dict(comparison_contract)
        if isinstance(comparison_contract, Mapping) else {}
    )
    nested_contract = attestation.get(
        "completed_task_comparison_endpoint_contract"
    )
    nested_contract = (
        dict(nested_contract)
        if isinstance(nested_contract, Mapping) else {}
    )
    if (
        validation.get("completed_task_endpoint_attested") is not True
        or top_status != "passed"
        or attestation.get("attested") is not True
        or str(attestation.get("status") or "").strip().lower()
        != "passed"
        or str(attestation.get("stage") or "").strip().lower()
        != "decoded_nms"
        or str(attestation.get("endpoint") or "").strip().lower()
        != "decoded_nms"
        or completion_mode not in supported_modes
        or nested_mode != completion_mode
        or validation_stages != {"decoded_nms"}
        or validation_id != plan_id
        or validation_hash != plan_hash
        or nested_id != plan_id
        or nested_hash != plan_hash
        or not comparison_contract
        or not nested_contract
        or comparison_contract != nested_contract
        or _strict_sha256_token(
            comparison_contract.get("endpoint_contract_hash")
        ) != plan_hash
        or str(
            comparison_contract.get("output_endpoint_id") or ""
        ).strip().lower() != plan_id
    ):
        return "", "", "", "validation_completed_endpoint_mismatch"
    validation_comparison_alias = str(
        validation.get("comparison_output_endpoint_id") or ""
    ).strip().lower()
    if validation_comparison_alias and validation_comparison_alias != plan_id:
        return "", "", "", "validation_comparison_endpoint_alias_mismatch"
    return (
        plan_id,
        plan_hash,
        plan_stage,
        "strict_plan_validation_completed_endpoint_match",
    )


def _resolved_energy_endpoint_identity(
    plan: Mapping[str, Any],
    validation: Mapping[str, Any],
    task: str,
    *, fresh_completions: Sequence[Mapping[str, Any]] = (),
    energy_repeats: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Resolve physical provenance and the endpoint admitted for pairing."""
    physical = _physical_energy_endpoint(plan, validation, task)
    result = {
        **physical,
        "comparison_output_endpoint_id": "",
        "comparison_endpoint_contract_hash": "",
        "comparison_endpoint_stage": "",
        "completion_pairing_eligible": False,
        "completion_pairing_status": str(
            plan.get("completion_pairing_status") or ""
        ),
        "completion_pairing_validation_status": "not_applicable",
        "output_endpoint_id": "",
        "endpoint_contract_hash": "",
        "endpoint_stage": "",
        "endpoint_contract_complete": False,
        "output_endpoint_match": False,
    }
    task = str(task or "").strip().lower()
    if task == "classification":
        endpoint_id = str(
            physical.get("physical_output_endpoint_id") or ""
        )
        endpoint_hash = _strict_sha256_token(
            physical.get("physical_endpoint_contract_hash")
        )
        endpoint_stage = str(
            physical.get("physical_endpoint_stage") or ""
        )
        eligible = bool(
            endpoint_id
            and endpoint_hash
            and endpoint_stage
            and physical.get("physical_endpoint_contract_complete") is True
        )
        result.update({
            "comparison_output_endpoint_id": endpoint_id if eligible else "",
            "comparison_endpoint_contract_hash": (
                endpoint_hash if eligible else ""
            ),
            "comparison_endpoint_stage": endpoint_stage if eligible else "",
            "completion_pairing_eligible": eligible,
            "completion_pairing_validation_status": (
                "classification_physical_endpoint_preserved"
                if eligible else "classification_physical_endpoint_unavailable"
            ),
            "output_endpoint_id": endpoint_id if eligible else "",
            "endpoint_contract_hash": endpoint_hash if eligible else "",
            "endpoint_stage": endpoint_stage if eligible else "",
            "endpoint_contract_complete": eligible,
            "output_endpoint_match": eligible,
        })
        return result
    if task != "detection":
        result["completion_pairing_validation_status"] = "task_unsupported"
        return result

    endpoint_id, endpoint_hash, endpoint_stage, status = (
        _completed_detection_energy_endpoint(
            plan,
            validation,
            physical_match=bool(
                physical.get("physical_output_endpoint_match")
            ),
            fresh_completions=fresh_completions,
            energy_repeats=energy_repeats,
        )
    )
    eligible = bool(endpoint_id and endpoint_hash and endpoint_stage)
    result.update({
        "comparison_output_endpoint_id": endpoint_id,
        "comparison_endpoint_contract_hash": endpoint_hash,
        "comparison_endpoint_stage": endpoint_stage,
        "completion_pairing_eligible": eligible,
        "completion_pairing_validation_status": status,
        "output_endpoint_id": endpoint_id,
        "endpoint_contract_hash": endpoint_hash,
        "endpoint_stage": endpoint_stage,
        "endpoint_contract_complete": eligible,
        "output_endpoint_match": eligible,
    })
    return result


def _canonical_row_endpoint(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return a row endpoint only when ID, task, stage and SHA agree."""
    task = str(row.get("task") or "").strip().lower()
    dual_identity_declared = any(
        key in row
        for key in (
            "completion_pairing_eligible",
            "comparison_output_endpoint_id",
            "comparison_endpoint_contract_hash",
            "comparison_endpoint_stage",
            "physical_output_endpoint_id",
        )
    )
    if task == "detection" and dual_identity_declared:
        endpoint_id, endpoint_hash, _stage = (
            _endpoint_identity_from_explicit_fields(
                row,
                task,
                endpoint_field="comparison_output_endpoint_id",
                hash_field="comparison_endpoint_contract_hash",
                stage_fields=("comparison_endpoint_stage",),
                complete_field="completion_pairing_eligible",
                comparison=True,
            )
        )
        if (
            not endpoint_id
            or row.get("endpoint_contract_complete") is not True
            or row.get("output_endpoint_match") is not True
            or str(row.get("output_endpoint_id") or "").strip().lower()
            != endpoint_id
            or _strict_sha256_token(row.get("endpoint_contract_hash"))
            != endpoint_hash
        ):
            return "", ""
        return endpoint_id, endpoint_hash
    endpoint_id = _endpoint_id_from_validation({}, row, task)
    endpoint_hash = _strict_sha256_token(row.get("endpoint_contract_hash"))
    if not endpoint_id or not endpoint_hash or not endpoint_id.endswith(endpoint_hash):
        return "", ""
    return endpoint_id, endpoint_hash


def collect_native_energy(run_dir: str | Path) -> list[dict[str, Any]]:
    run = Path(run_dir)
    src = run / "reports" / "native_energy_measurements" / "native_producer_energy_results.json"
    data = _load(src)
    validations = _native_validation_map(run)
    out: list[dict[str, Any]] = []
    for item in list(data.get("rows") or []):
        if not isinstance(item, Mapping):
            continue
        plan = dict(item.get("row") or {})
        validation, validation_join_status = _validation_evidence_for_plan(
            plan, validations,
        )
        _project_bound_preprocessing_pad(plan, validation, validation_join_status)
        identity_conflicts: list[str] = []

        def identity(field: str, *aliases: str) -> str:
            value, conflict = _identity_value(
                plan, validation, field, *aliases,
            )
            if conflict:
                identity_conflicts.append(field)
            return value

        run_info = dict(item.get("run") or {})
        payload = _energy_payload(item)
        aggregate = payload.get("_native_energy_aggregate") if isinstance(payload.get("_native_energy_aggregate"), Mapping) else {}
        repeat_records = _energy_repeat_records(item)
        backend = str(plan.get("backend") or "")
        marker_energy_per_work = _num(_first(
            payload.get("avg_energy_per_work_unit_j"),
            payload.get("energy_per_work_unit_j"),
            payload.get("avg_energy_per_inference_j"),
            payload.get("energy_per_inference_j"),
            payload.get("joules_per_inference"),
        ))
        marker_total_energy = _num(_first(
            payload.get("avg_energy_total_j"),
            payload.get("energy_total_j"),
            payload.get("energy_j"),
        ))
        work_units = _num(
            _first(payload.get("avg_energy_work_units_used"), payload.get("energy_work_units_used"), payload.get("work_units"), payload.get("inference_count"))
        )
        marker_average_power = _num(_first(payload.get("avg_power_w"), payload.get("mean_power_w")))
        marker_active_duration = _num(
            _first(payload.get("avg_active_duration_s"), payload.get("active_duration_s"), payload.get("energy_measured_workload_duration_s"), payload.get("measurement_duration_s"))
        )
        primary_frozen = payload.get("scientific_primary_method_frozen") is True
        primary_method = str(payload.get("scientific_primary_method") or "").strip()
        canonical_primary_method = _canonical_energy_method(primary_method)
        # This exact identifier denotes the frozen role used in 2.63--2.66.
        # It must remain a Chapter-4 primary when archived runs are read; only
        # v2.67 artefacts explicitly naming command_marker_window use the new
        # role assignment.
        historical_chapter4_primary = bool(
            primary_frozen and primary_method == "chapter4_baseline"
        )
        primary_status = str(payload.get("scientific_primary_energy_status") or "").strip()
        primary_available = bool(
            primary_frozen
            and (
                historical_chapter4_primary
                or canonical_primary_method == ENERGY_PRIMARY_METHOD
            )
            and primary_status == "available"
        )
        if primary_frozen:
            # A frozen aggregate is authoritative in both contract generations.
            # Missing primary scalars fail closed instead of falling back to the
            # diagnostic method.
            total_energy = _num(payload.get("scientific_primary_energy_total_j"))
            energy_per_work = _num(payload.get("scientific_primary_energy_per_work_unit_j"))
            average_power = _num(payload.get("scientific_primary_avg_power_w"))
            active_duration = _num(payload.get("scientific_primary_active_duration_s"))
        else:
            total_energy = marker_total_energy
            energy_per_work = marker_energy_per_work
            average_power = marker_average_power
            active_duration = marker_active_duration
        target_duration = _num(_first(plan.get("target_duration_s"), plan.get("duration_s")))
        if average_power is None and total_energy is not None and active_duration not in (None, 0):
            average_power = total_energy / active_duration
        screening_energy_per_work = _num(payload.get("energy_per_work_unit_screening_estimate_j"))
        if screening_energy_per_work is None and total_energy is not None and work_units not in (None, 0):
            screening_energy_per_work = total_energy / work_units

        confidence_level = _num(payload.get("confidence_level")) or 0.95
        primary_statistics = payload.get("scientific_primary_energy_statistics")
        primary_statistics = dict(primary_statistics) if isinstance(primary_statistics, Mapping) else {}
        candidate_statistics = payload.get("candidate_v263_shadow_statistics")
        candidate_statistics = dict(candidate_statistics) if isinstance(candidate_statistics, Mapping) else {}
        chapter4_shadow_statistics = payload.get("chapter4_legacy_shadow_statistics")
        chapter4_shadow_statistics = (
            dict(chapter4_shadow_statistics)
            if isinstance(chapter4_shadow_statistics, Mapping) else {}
        )
        legacy_candidate_total_energy = _num(_first(
            payload.get("candidate_v263_shadow_energy_total_j"),
            marker_total_energy if historical_chapter4_primary else None,
        ))
        legacy_candidate_energy_per_work = _num(_first(
            payload.get("candidate_v263_shadow_energy_per_work_unit_j"),
            marker_energy_per_work if historical_chapter4_primary else None,
        ))
        chapter4_shadow_total_energy = _num(_first(
            payload.get("chapter4_legacy_shadow_energy_total_j"),
            payload.get("chapter4_legacy_shadow_energy_j"),
        ))
        chapter4_shadow_energy_per_work = _num(
            payload.get("chapter4_legacy_shadow_energy_per_work_unit_j")
        )
        if historical_chapter4_primary:
            shadow_method = str(
                payload.get("scientific_shadow_method")
                or payload.get("candidate_method")
                or "candidate_v263"
            ).strip()
            shadow_status = "available" if legacy_candidate_total_energy is not None else "unavailable"
            shadow_total_energy = legacy_candidate_total_energy
            shadow_energy_per_work = legacy_candidate_energy_per_work
            shadow_statistics = candidate_statistics
        else:
            shadow_method = str(
                payload.get("scientific_shadow_method") or ENERGY_SHADOW_METHOD
            ).strip()
            shadow_status = str(
                payload.get("scientific_shadow_energy_status")
                or payload.get("chapter4_legacy_shadow_status")
                or ("available" if chapter4_shadow_total_energy is not None else "unavailable")
            ).strip()
            shadow_total_energy = chapter4_shadow_total_energy
            shadow_energy_per_work = chapter4_shadow_energy_per_work
            shadow_statistics = chapter4_shadow_statistics
        ab_statistics = payload.get("window_method_comparison_statistics")
        ab_statistics = dict(ab_statistics) if isinstance(ab_statistics, Mapping) else {}
        if not ab_statistics and repeat_records:
            ab_statistics = _fallback_ab_statistics(repeat_records, confidence_level)

        primary_energy_stats = _stats_block(primary_statistics, "energy_j")
        primary_per_work_stats = _stats_block(primary_statistics, "energy_per_work_unit_j")
        primary_power_stats = _stats_block(primary_statistics, "avg_power_w")
        shadow_energy_stats = _stats_block(shadow_statistics, "energy_j")
        shadow_per_work_stats = _stats_block(shadow_statistics, "energy_per_work_unit_j")
        ab_energy_stats = _stats_block(
            ab_statistics.get("energy_j") if isinstance(ab_statistics.get("energy_j"), Mapping) else {},
            "relative_percent",
        )
        ab_duration_stats = _stats_block(
            ab_statistics.get("duration_s") if isinstance(ab_statistics.get("duration_s"), Mapping) else {},
            "relative_percent",
        )
        ab_power_stats = _stats_block(
            ab_statistics.get("average_power_w") if isinstance(ab_statistics.get("average_power_w"), Mapping) else {},
            "relative_percent",
        )
        requested_repeats = int(
            _num(payload.get("run_count"))
            or _num(payload.get("effective_run_count"))
            or _num(payload.get("requested_run_count"))
            or 0
        )
        if requested_repeats <= 0:
            for record in repeat_records:
                ab_policy = record.get("energy_window_method_ab")
                if not isinstance(ab_policy, Mapping):
                    continue
                requested_repeats = int(
                    _num(ab_policy.get("effective_run_count"))
                    or _num(ab_policy.get("smoke_repeats_required"))
                    or 0
                )
                if requested_repeats:
                    break
        visible_valid_repeats = sum(
            1 for record in repeat_records
            if str(record.get("postprocess_status") or "") == "ok"
            and str(record.get("final_energy_gate_status") or "pass") == "pass"
        )
        aggregate_valid_repeats = _num(payload.get("valid_postprocessed_runs"))
        statistics_valid_repeats = _num(primary_energy_stats.get("n"))
        authoritative_valid_counts = [
            int(value)
            for value in (
                aggregate_valid_repeats,
                statistics_valid_repeats,
            )
            if value is not None and value >= 0
        ]
        valid_repeats = (
            min(authoritative_valid_counts)
            if authoritative_valid_counts
            else visible_valid_repeats
        )
        observed_repeat_count = int(
            _num(payload.get("raw_postprocessed_run_count"))
            or len(repeat_records)
            or valid_repeats
        )
        if requested_repeats <= 0 and observed_repeat_count > 0:
            requested_repeats = observed_repeat_count
        repeat_status = (
            "complete" if requested_repeats > 0 and valid_repeats >= requested_repeats
            else "incomplete_valid_repeats" if valid_repeats > 0
            else "aggregate_unavailable" if not aggregate
            else "no_valid_repeats"
        )
        ab_requested = bool(
            payload.get("legacy_window_comparison_requested") is True
            or primary_frozen
            or ab_statistics
        )
        aggregate_ab_valid_repeats = _num(
            payload.get("legacy_window_comparison_successful_runs")
        )
        statistics_ab_valid_repeats = _num(ab_energy_stats.get("n"))
        authoritative_ab_valid_counts = [
            int(value)
            for value in (
                aggregate_ab_valid_repeats,
                statistics_ab_valid_repeats,
            )
            if value is not None and value >= 0
        ]
        ab_valid_repeats = (
            min(authoritative_ab_valid_counts)
            if authoritative_ab_valid_counts else 0
        )
        ab_attempted_repeats = int(
            _num(payload.get("legacy_window_comparison_attempted_runs"))
            or len(repeat_records)
            or 0
        )
        energy_ab_status = (
            "not_requested" if not ab_requested
            else "complete" if requested_repeats > 0 and ab_valid_repeats >= requested_repeats
            else "incomplete_valid_repeats" if ab_valid_repeats > 0
            else "no_valid_ab_repeats"
        )

        task, task_source = _infer_task(
            str(plan.get("model") or ""),
            _first(validation.get("task"), plan.get("task")),
        )
        comparison_backend = str(plan.get("comparison_backend") or "")
        direction = _canonical_direction(plan.get("direction"), backend, comparison_backend)
        requested_scope = _canonical_scope(_first(plan.get("energy_scope"), plan.get("scope")))
        # The measurement payload is authoritative.  A plan requesting FS must
        # never rename an MB observation when the collector reported MB.
        measured_scope = _canonical_scope(_first(payload.get("energy_physical_scope_canonical"), payload.get("energy_physical_scope")))
        requested_window = _canonical_window(_first(payload.get("energy_window_requested"), plan.get("energy_window"), plan.get("window")))
        effective_window = _canonical_window(_first(payload.get("energy_window_effective_values"), payload.get("energy_window_effective")))
        calibration_manifest, calibration_sha, calibration_verified, calibration_status = _calibration_fields(payload, plan)
        (
            calibration_required,
            calibration_status,
            external_calibration_status,
        ) = _normalized_calibration_axes(
            measured_scope,
            verified=calibration_verified,
            status=calibration_status,
        )
        external_calibration_verified = calibration_verified
        if measured_scope == "full_system" and payload.get("full_system_current_scale_configured") is True:
            # The optional external method manifest remains a separate fact:
            # using a valid FS scale must not invent that absent provenance.
            external = payload.get("energy_calibration_verification")
            external = external if isinstance(external, Mapping) else {}
            external_calibration_verified = external.get("verified") is True
            external_calibration_status = str(external.get("status") or "missing")
        execution_ok = bool(item.get("ok")) and int(run_info.get("rc") or 0) == 0
        measurement_claim = payload.get("energy_efficiency_claim_eligible") is True
        if validation_join_status == "exact_unique":
            semantic_claim = bool(
                _truth(validation.get("claim_ok"))
                and validation.get("semantic_ok") is True
                and _truth(validation.get("contract_consistent"))
                and (
                    task != "classification"
                    or validation.get("top1_match") is True
                )
            )
            contract_consistent = _truth(validation.get("contract_consistent"))
        else:
            semantic_claim = bool(
                plan.get("semantic_claim_ok") is True
                or _truth(plan.get("semantic_claim_ok"))
                or plan.get("claim_ok") is True
                or _truth(plan.get("claim_ok"))
            )
            contract_consistent = bool(
                plan.get("contract_consistent") is True
                or _truth(plan.get("contract_consistent"))
            )
        quality_gate_declared = bool(
            validation_join_status != "missing"
            or any(key in plan for key in (
                "central_quality_evidence_verified",
                "precision_quality_verified",
                "precision_quality_binding_verified",
                "task_quality_observation_valid",
                "accuracy_gate_pass",
                "quality_claim_result_verified",
                "energy_quality_admission",
            ))
        )
        quality_admission_declared = (
            "energy_quality_admission" in plan
        )
        admission = (
            dict(plan.get("energy_quality_admission") or {})
            if isinstance(
                plan.get("energy_quality_admission"), Mapping,
            )
            else {}
        )
        quality_admission_status = (
            "legacy_energy_quality_admission_not_declared"
        )
        quality_admission_verified = False
        if quality_admission_declared:
            try:
                _, quality_admission_status = (
                    verify_sealed_energy_quality_admission(
                        plan,
                        required=True,
                    )
                )
            except ValueError as exc:
                quality_admission_status = str(exc)
            else:
                quality_admission_verified = True
        if quality_admission_verified:
            central_quality_verified = bool(
                admission["central_quality_evidence_verified"]
            )
            precision_quality_binding_verified = bool(
                admission["precision_quality_binding_verified"]
            )
            task_quality_observation_valid = bool(
                admission["task_quality_observation_valid"]
            )
            accuracy_gate_pass = bool(
                admission["accuracy_gate_pass"]
            )
            quality_claim_result_verified = bool(
                admission["quality_claim_result_verified"]
            )
            quality_claim_result_value = (
                quality_claim_result_verified
            )
        elif quality_admission_declared:
            central_quality_verified = False
            precision_quality_binding_verified = False
            task_quality_observation_valid = False
            accuracy_gate_pass = False
            quality_claim_result_verified = False
            quality_claim_result_value = False
        else:
            central_quality_verified = bool(
                _first(
                    validation.get(
                        "central_quality_evidence_verified"
                    ),
                    plan.get(
                        "central_quality_evidence_verified"
                    ),
                ) is True
            )
            binding_value = _first(
                validation.get(
                    "precision_quality_binding_verified"
                ),
                plan.get("precision_quality_binding_verified"),
            )
            precision_quality_binding_verified = bool(
                central_quality_verified and binding_value is True
            )
            task_observation_value = _first(
                validation.get("task_quality_observation_valid"),
                plan.get("task_quality_observation_valid"),
            )
            accuracy_gate_value = _first(
                validation.get("accuracy_gate_pass"),
                plan.get("accuracy_gate_pass"),
            )
            task_quality_observation_valid = bool(
                precision_quality_binding_verified
                and task_observation_value is True
                and isinstance(accuracy_gate_value, bool)
            )
            accuracy_gate_pass = bool(
                task_quality_observation_valid
                and accuracy_gate_value is True
            )
            quality_claim_result_value = _first(
                validation.get(
                    "quality_claim_result_verified"
                ),
                plan.get("quality_claim_result_verified"),
            )
            quality_claim_result_verified = bool(
                quality_claim_result_value is True
            )
        quality_gate_verified = bool(
            precision_quality_binding_verified
            and task_quality_observation_valid
            and accuracy_gate_pass
        )
        endpoint_contract_declared = bool(
            validation_join_status != "missing"
            or any(key in plan for key in (
                "output_endpoint_id", "endpoint_contract_hash",
                "endpoint_contract_complete", "output_endpoint_match",
                "physical_output_endpoint_id",
                "physical_endpoint_contract_hash",
                "comparison_output_endpoint_id",
                "comparison_endpoint_contract_hash",
                "completion_pairing_eligible",
            ))
        )
        endpoint_identity = _resolved_energy_endpoint_identity(
            plan, validation, task,
            fresh_completions=run_info.get("fresh_energy_completion_evidence") or [],
            energy_repeats=aggregate.get("runs") or [],
        )
        endpoint_contract_complete = bool(
            endpoint_identity.get("endpoint_contract_complete") is True
        )
        output_endpoint_id = str(
            endpoint_identity.get("output_endpoint_id") or ""
        )
        output_endpoint_stage = (
            str(endpoint_identity.get("endpoint_stage") or "")
        )
        source_request_hash = identity(
            "source_request_sha256", "quality_source_sha256",
        )
        model_hash = identity(
            "model_sha256", "source_model_sha256", "source_onnx_sha256",
            "model_hash",
        )
        validation_dataset_hash = identity(
            "validation_dataset_sha256", "validation_dataset_manifest_sha256",
            "dataset_manifest_sha256", "dataset_sha256",
        )
        validation_image_ids_hash = identity(
            "validation_dataset_image_ids_sha256", "validation_image_ids_sha256",
            "dataset_image_ids_sha256", "image_ids_sha256",
        )
        validation_ground_truth_hash = identity(
            "validation_dataset_ground_truth_sha256", "validation_ground_truth_sha256",
            "dataset_ground_truth_sha256", "ground_truth_sha256",
        )
        accuracy_gate_policy_hash = identity(
            "accuracy_gate_policy_sha256",
        )
        task_quality_policy_hash = identity(
            "task_quality_policy_sha256", "policy_sha256",
        )
        runtime_quality_policy_hash = identity(
            "runtime_quality_gate_policy_sha256",
        )
        # Pipeline-freeze hashes and producer-quality hashes are different
        # scientific namespaces.  A DeepX producer implementation is expected
        # to have different decoder/preprocess hashes from the generic Split
        # runner, so never treat their coexistence as an identity conflict.
        pipeline_contract_hash = identity(
            "pipeline_contract_sha256", "pipeline_contract_hash",
        )
        pipeline_preprocessing_hash = identity(
            "pipeline_preprocessing_sha256", "pipeline_preprocessing_hash",
        )
        pipeline_decoder_hash = identity(
            "pipeline_decoder_sha256", "pipeline_decoder_hash",
        )
        pipeline_nms_hash = identity(
            "pipeline_nms_sha256", "pipeline_nms_hash",
        )
        quality_contract_hash = identity("quality_contract_sha256")
        quality_preprocessing_hash = identity("preprocessing_contract_sha256")
        quality_decoder_hash = identity("decoder_contract_sha256")
        quality_nms_hash = identity("nms_contract_sha256")
        legacy_contract_hash = _normalized_sha256(plan.get("contract_hash"))
        legacy_preprocessing_hash = _normalized_sha256(plan.get("preprocessing_hash"))
        legacy_decoder_hash = _normalized_sha256(plan.get("decoder_hash"))
        legacy_nms_hash = _normalized_sha256(plan.get("nms_hash"))
        contract_hash = pipeline_contract_hash or quality_contract_hash or legacy_contract_hash
        preprocessing_hash = (
            pipeline_preprocessing_hash or quality_preprocessing_hash
            or legacy_preprocessing_hash
        )
        decoder_hash = pipeline_decoder_hash or quality_decoder_hash or legacy_decoder_hash
        nms_hash = pipeline_nms_hash or quality_nms_hash or legacy_nms_hash
        endpoint_contract_hash = _strict_sha256_token(
            endpoint_identity.get("endpoint_contract_hash")
        )
        required_quality_hashes = [
            source_request_hash, model_hash, validation_dataset_hash,
            validation_image_ids_hash, validation_ground_truth_hash,
            accuracy_gate_policy_hash, task_quality_policy_hash,
            runtime_quality_policy_hash,
            quality_contract_hash, quality_preprocessing_hash,
        ]
        if task == "detection":
            required_quality_hashes.extend((quality_decoder_hash, quality_nms_hash))
        quality_provenance_complete = bool(
            all(required_quality_hashes)
            and len({
                accuracy_gate_policy_hash, task_quality_policy_hash,
                runtime_quality_policy_hash,
            }) == 1
        )
        if quality_admission_verified:
            quality_provenance_complete = bool(
                admission["quality_provenance_complete"]
            )
        elif quality_admission_declared:
            quality_provenance_complete = False
        quality_verified = bool(
            quality_gate_verified
            and quality_provenance_complete
            and not identity_conflicts
        )
        output_endpoint_match = bool(
            endpoint_identity.get("output_endpoint_match") is True
        )
        exact_count = _exact_runtime_count(payload)
        postprocess_ok = str(payload.get("postprocess_status") or "") == "ok"
        collector_reported_final_gate_status = str(
            payload.get("final_energy_gate_status") or ""
        )
        final_gate_ok = bool(
            collector_reported_final_gate_status == "pass"
            and repeat_status == "complete"
        )
        final_energy_gate_status = (
            "pass" if final_gate_ok else "fail"
        )
        command_window_ok = effective_window == "command_window"
        scope_known = measured_scope not in {"", "unknown"}
        calibration_ok = measured_scope != "full_system" or calibration_verified
        raw_primary_ok = _calibrated_unsubtracted_input_primary(payload)
        # ``0`` is the valid effective pad for classification/resize.  Do not
        # collapse it through truthiness when checking the frozen feed
        # identity; only an absent or empty value is missing.
        prepared_feed_identity_complete = all(
            plan.get(field) is not None and str(plan.get(field)).strip() != ""
            for field in (
                "prepared_feed_task", "prepared_feed_preprocess_mode",
                "prepared_feed_letterbox_pad_value",
                "prepared_feed_source_image_sha256",
            )
        )
        scientific_primary_ok = bool(
            not primary_frozen
            or (
                primary_available
                and total_energy is not None
                and energy_per_work is not None
            )
        )
        screening_policy_nonclaimable = bool(
            plan.get("screening_only") is True
            or plan.get("screening_energy") is True
            or plan.get("diagnostic_only") is True
            or payload.get("diagnostic_only") is True
        )
        claim_reasons: list[str] = []
        for condition, reason in (
            (execution_ok, "measurement_execution_failed"),
            (
                semantic_claim or screening_policy_nonclaimable,
                "semantic_claim_not_admitted",
            ),
            (contract_consistent, "contract_not_consistent"),
            (
                not quality_gate_declared or quality_provenance_complete,
                "native_quality_provenance_missing_or_conflicting",
            ),
            (
                not endpoint_contract_declared
                or (endpoint_contract_complete and output_endpoint_match),
                "native_output_endpoint_not_verified",
            ),
            (not identity_conflicts, "native_identity_evidence_conflict"),
            (
                measurement_claim or screening_policy_nonclaimable,
                "collector_claim_gate_not_passed",
            ),
            (final_gate_ok, "final_energy_gate_not_passed"),
            (postprocess_ok, "postprocess_not_ok"),
            (exact_count, "exact_runtime_work_units_missing"),
            (command_window_ok, "effective_command_window_not_verified"),
            (scope_known, "physical_scope_unknown"),
            (calibration_ok, "full_system_calibration_not_verified"),
            (raw_primary_ok, "raw_input_energy_not_primary"),
            (prepared_feed_identity_complete, "prepared_feed_identity_missing"),
            (scientific_primary_ok, "scientific_primary_unavailable_or_unverified"),
        ):
            if not condition:
                claim_reasons.append(reason)
        if screening_policy_nonclaimable:
            claim_reasons.append(
                "screening_energy_policy_nonclaimable"
            )
        if (
            quality_admission_declared
            and not quality_admission_verified
        ):
            claim_reasons.append(quality_admission_status)
        if quality_gate_declared:
            if not precision_quality_binding_verified:
                claim_reasons.append(
                    "native_precision_quality_binding_not_verified"
                )
            elif not task_quality_observation_valid:
                claim_reasons.append(
                    "native_task_quality_observation_invalid"
                )
            elif not accuracy_gate_pass:
                accuracy_status = str(
                    validation.get("quality_gate_status")
                    or validation.get("task_quality_status")
                    or validation.get("status")
                    or ""
                ).strip().lower()
                claim_reasons.append(
                    "native_accuracy_gate_inconclusive"
                    if "inconclusive" in accuracy_status
                    else "native_accuracy_gate_not_passed"
                )
            elif (
                quality_claim_result_value is not None
                and not quality_claim_result_verified
            ):
                claim_reasons.append(
                    "native_quality_claim_result_not_verified"
                )
        claim_reasons = list(dict.fromkeys(claim_reasons))
        claim_eligible = not claim_reasons
        measurement_failure_reason = _measurement_failure_reason(
            item,
            run_info,
            payload,
            execution_ok=execution_ok,
        )
        try:
            measurement_run_rc = (
                int(run_info.get("rc"))
                if run_info.get("rc") not in (None, "")
                else None
            )
        except Exception:
            measurement_run_rc = None
        is_full = backend.startswith("native_full_")
        legacy_comparison_precision = str(
            _first(
                plan.get("comparison_precision"),
                plan.get("legacy_comparison_precision"),
                plan.get("precision") if is_full else "",
            )
            or ""
        )
        explicit_runtime_precision = str(
            _first(
                validation.get("full_runtime_precision"),
                validation.get("execution_precision"),
                validation.get("runtime_precision_identity"),
                plan.get("full_runtime_precision"), plan.get("execution_precision"),
            )
            or ""
        ) if is_full else str(_first(
            validation.get("execution_precision"),
            validation.get("runtime_precision_identity"),
            plan.get("execution_precision"),
            plan.get("split_boundary_precision"),
            plan.get("precision"),
        ) or "")
        start_observation = (
            run_info.get("measurement_start_observation")
            if isinstance(
                run_info.get("measurement_start_observation"), Mapping,
            ) else {}
        )
        raw_energy_collected = bool(
            total_energy is not None
            or energy_per_work is not None
            or marker_total_energy is not None
            or marker_energy_per_work is not None
            or (
                primary_available
                and any(
                    value is not None
                    for value in (average_power, active_duration)
                )
            )
        )
        measurement_started = bool(
            start_observation.get("measurement_started") is True
            or raw_energy_collected
        )
        final_energy_quality_fields = _final_energy_quality_result_fields(
            plan_qualified=plan.get("energy_quality_qualified") is True,
            admission_verified=quality_admission_verified,
            measurement_started=measurement_started,
            raw_energy_collected=raw_energy_collected,
        )
        out.append({
            "backend": backend,
            "model": str(plan.get("model") or ""),
            "task": task,
            "task_source": task_source,
            "evaluation_role": str(plan.get("evaluation_role") or "unknown").strip().lower() or "unknown",
            "runner_regime": _canonical_runner(plan.get("runner_regime"), backend),
            "direction": direction,
            "case": str(plan.get("case") or ("full" if backend.startswith("native_full_") else "")),
            # For historical Native Full rows, ``precision`` is the Split
            # comparison stratum, not Full execution precision.  Preserve it
            # for compatibility while exposing its role explicitly.
            "precision": str(plan.get("precision") or ""),
            "precision_role": str(
                plan.get("precision_role")
                or ("legacy_comparison_precision" if is_full else "split_boundary_precision")
            ),
            "execution_precision": explicit_runtime_precision,
            "split_boundary_precision": str(
                plan.get("split_boundary_precision")
                or (plan.get("precision") if not is_full else "")
                or ""
            ),
            "full_runtime_precision": explicit_runtime_precision if is_full else "",
            "comparison_precision": legacy_comparison_precision,
            "legacy_comparison_precision": legacy_comparison_precision,
            "runtime_precision_status": str(
                (
                    "verified_explicit"
                    if explicit_runtime_precision
                    and validation_join_status == "exact_unique"
                    else plan.get("runtime_precision_status")
                    or ("verified_explicit" if explicit_runtime_precision else "unavailable")
                )
            ),
            "setup_id": str(plan.get("setup_id") or ""),
            "comparison_backend": comparison_backend,
            "execution_mode": _execution_mode(backend, plan),
            "semantic_gate": str(plan.get("semantic_gate") or ""),
            "semantic_claim_ok": semantic_claim,
            "contract_consistent": contract_consistent,
            "native_validation_join_status": validation_join_status,
            "native_identity_evidence_conflicts": identity_conflicts,
            "central_quality_evidence_verified": (
                central_quality_verified
            ),
            "precision_quality_verified": (
                precision_quality_binding_verified
            ),
            "precision_quality_binding_verified": (
                precision_quality_binding_verified
            ),
            "task_quality_observation_valid": (
                task_quality_observation_valid
            ),
            "accuracy_gate_pass": accuracy_gate_pass,
            **assessment_fields(
                validation.get("accuracy_assessment") if validation_join_status == "exact_unique" else admission.get("accuracy_assessment")),
            "quality_claim_result_verified": (
                quality_claim_result_verified
            ),
            "energy_quality_admission_verified": (
                quality_admission_verified
            ),
            "energy_quality_admission_status": (
                quality_admission_status
            ),
            "measurement_started": measurement_started,
            "raw_energy_collected": raw_energy_collected,
            **final_energy_quality_fields,
            "quality_verified": quality_verified,
            "quality_provenance_complete": quality_provenance_complete,
            "quality_gate_status": str(_first(
                validation.get("quality_gate_status"),
                validation.get("gate_status"), plan.get("quality_gate_status"),
            ) or ""),
            "contract_hash": contract_hash,
            "preprocessing_hash": preprocessing_hash,
            "decoder_hash": decoder_hash,
            "nms_hash": nms_hash,
            "pipeline_contract_sha256": pipeline_contract_hash,
            "pipeline_preprocessing_sha256": pipeline_preprocessing_hash,
            "pipeline_decoder_sha256": pipeline_decoder_hash,
            "pipeline_nms_sha256": pipeline_nms_hash,
            "quality_contract_sha256": quality_contract_hash,
            "preprocessing_contract_sha256": quality_preprocessing_hash,
            "decoder_contract_sha256": quality_decoder_hash,
            "nms_contract_sha256": quality_nms_hash,
            "output_endpoint_id": output_endpoint_id,
            "stage": output_endpoint_stage,
            "contract_family": output_endpoint_stage,
            "endpoint_contract_hash": endpoint_contract_hash,
            "endpoint_contract_complete": endpoint_contract_complete,
            "output_endpoint_match": output_endpoint_match,
            "physical_output_endpoint_id": str(
                endpoint_identity.get("physical_output_endpoint_id") or ""
            ),
            "physical_endpoint_contract_hash": _strict_sha256_token(
                endpoint_identity.get("physical_endpoint_contract_hash")
            ),
            "physical_endpoint_stage": str(
                endpoint_identity.get("physical_endpoint_stage") or ""
            ),
            "physical_endpoint_contract_complete": bool(
                endpoint_identity.get(
                    "physical_endpoint_contract_complete"
                ) is True
            ),
            "physical_output_endpoint_match": bool(
                endpoint_identity.get("physical_output_endpoint_match")
                is True
            ),
            "physical_endpoint_identity_status": str(
                endpoint_identity.get(
                    "physical_endpoint_identity_status"
                ) or ""
            ),
            "comparison_output_endpoint_id": str(
                endpoint_identity.get("comparison_output_endpoint_id") or ""
            ),
            "comparison_endpoint_contract_hash": _strict_sha256_token(
                endpoint_identity.get(
                    "comparison_endpoint_contract_hash"
                )
            ),
            "comparison_endpoint_stage": str(
                endpoint_identity.get("comparison_endpoint_stage") or ""
            ),
            "completion_pairing_eligible": bool(
                endpoint_identity.get("completion_pairing_eligible") is True
            ),
            "completion_pairing_status": str(
                endpoint_identity.get("completion_pairing_status") or ""
            ),
            "completion_pairing_validation_status": str(
                endpoint_identity.get(
                    "completion_pairing_validation_status"
                ) or ""
            ),
            "source_request_sha256": source_request_hash,
            "model_sha256": model_hash,
            "validation_dataset_sha256": validation_dataset_hash,
            "validation_dataset_image_ids_sha256": validation_image_ids_hash,
            "validation_dataset_ground_truth_sha256": validation_ground_truth_hash,
            "accuracy_gate_policy_sha256": accuracy_gate_policy_hash,
            "task_quality_policy_sha256": task_quality_policy_hash,
            "runtime_quality_gate_policy_sha256": runtime_quality_policy_hash,
            "validation_input_or_image_sha256": str(
                plan.get("validation_input_or_image_sha256")
                or plan.get("validation_input_sha256")
                or plan.get("validation_image_sha256")
                or plan.get("input_image_sha256")
                or ""
            ),
            "prepared_feed_task": str(plan.get("prepared_feed_task") or ""),
            "prepared_feed_preprocess_mode": str(
                plan.get("prepared_feed_preprocess_mode") or ""
            ),
            "prepared_feed_letterbox_pad_value": str(
                plan.get("prepared_feed_letterbox_pad_value")
                if plan.get("prepared_feed_letterbox_pad_value") is not None
                else ""
            ),
            "prepared_feed_source_image_sha256": str(
                plan.get("prepared_feed_source_image_sha256") or ""
            ),
            "prepared_feed_identity_source": str(
                plan.get("prepared_feed_identity_source") or ""
            ),
            "fps": _num(plan.get("fps") or payload.get("pipeline_fps_selected")),
            "duration_s": target_duration,
            "target_duration_s": target_duration,
            "active_duration_s": active_duration,
            "active_duration_relative_tolerance": NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE,
            "active_duration_comparison_policy": NATIVE_ENERGY_ACTIVE_DURATION_COMPARISON_POLICY,
            "active_duration_tolerance_source": NATIVE_ENERGY_ACTIVE_DURATION_TOLERANCE_SOURCE,
            "average_power_w": average_power,
            "energy_total_j": total_energy,
            "energy_per_work_j": energy_per_work,
            "energy_per_inference_j": energy_per_work,
            "scientific_primary_method_frozen": primary_frozen,
            "scientific_primary_method": primary_method or ("candidate_v263" if not primary_frozen else "unknown"),
            "scientific_primary_energy_status": primary_status,
            "scientific_primary_claim_eligible": payload.get("scientific_primary_claim_eligible") is True,
            "scientific_primary_contract_generation": (
                "v2.63-v2.66_chapter4_primary"
                if historical_chapter4_primary else "v2.67_command_marker_primary"
                if primary_frozen and canonical_primary_method == ENERGY_PRIMARY_METHOD
                else "legacy_unfrozen"
            ),
            "scientific_shadow_method": shadow_method,
            "scientific_shadow_role": str(
                payload.get("scientific_shadow_role")
                or payload.get("shadow_role")
                or "same_trace_sensitivity_only"
            ),
            "scientific_shadow_energy_status": shadow_status,
            "scientific_shadow_valid_run_count": int(
                _num(payload.get("scientific_shadow_valid_run_count"))
                or _num(shadow_energy_stats.get("n"))
                or 0
            ),
            "scientific_shadow_claim_eligible": False,
            "scientific_shadow_affects_primary_claim": False,
            "scientific_shadow_affects_primary_result": False,
            "scientific_shadow_affects_final_gate": False,
            "shadow_energy_total_j": shadow_total_energy,
            "shadow_energy_per_work_j": shadow_energy_per_work,
            "shadow_energy_total_j_mean": (
                _num(shadow_energy_stats.get("mean"))
                if shadow_energy_stats else shadow_total_energy
            ),
            "shadow_energy_total_j_ci_low": _num(shadow_energy_stats.get("ci_low")),
            "shadow_energy_total_j_ci_high": _num(shadow_energy_stats.get("ci_high")),
            "shadow_energy_per_work_j_mean": (
                _num(shadow_per_work_stats.get("mean"))
                if shadow_per_work_stats else shadow_energy_per_work
            ),
            "shadow_energy_per_work_j_ci_low": _num(shadow_per_work_stats.get("ci_low")),
            "shadow_energy_per_work_j_ci_high": _num(shadow_per_work_stats.get("ci_high")),
            "chapter4_legacy_shadow_energy_total_j": (
                chapter4_shadow_total_energy if not historical_chapter4_primary else None
            ),
            "chapter4_legacy_shadow_energy_per_work_j": (
                chapter4_shadow_energy_per_work if not historical_chapter4_primary else None
            ),
            "candidate_method": str(
                payload.get("candidate_method")
                or ("candidate_v263" if historical_chapter4_primary else "")
            ),
            "candidate_role": str(
                payload.get("candidate_role")
                or ("shadow_only" if historical_chapter4_primary else "")
            ),
            "candidate_eligible_for_auto_switch": payload.get("candidate_eligible_for_auto_switch") is True,
            "candidate_v263_shadow_energy_total_j": legacy_candidate_total_energy,
            "candidate_v263_shadow_energy_per_work_j": legacy_candidate_energy_per_work,
            "energy_per_work_screening_estimate_j": screening_energy_per_work,
            "energy_dynamic_j": _num(_first(payload.get("avg_energy_dynamic_j"), payload.get("energy_dynamic_j"))),
            "host_normalized_energy_est_j": _num(_first(payload.get("avg_host_normalized_energy_est_j"), payload.get("host_normalized_energy_est_j"))),
            "energy_normalization_repeats": [
                {key: record.get(key) for key in (
                    "run_index", "energy_total_j", "active_duration_s", "avg_power_w",
                    "energy_work_units_used", "energy_per_work_unit_j",
                    "host_normalized_energy_est_j", "host_normalized_energy_per_work_unit_est_j",
                    "host_normalized_average_power_est_w", "accelerator_idle_w_applied",
                    "postprocess_status", "accelerator_idle_correction_applied",
                )}
                for record in aggregate.get("runs", []) if isinstance(record, Mapping)
                and record.get("postprocess_status") == "ok"
            ],
            "host_normalized_energy_per_work_est_j": _num(_first(payload.get("avg_host_normalized_energy_per_work_unit_est_j"), payload.get("host_normalized_energy_per_work_unit_est_j"))),
            "host_normalized_energy_per_work_est_j_sample_stddev": _num(payload.get("host_normalized_energy_per_work_unit_est_j_sample_stddev")),
            "host_normalized_energy_per_work_est_j_ci_low": _num(payload.get("host_normalized_energy_per_work_unit_est_j_ci_low")),
            "host_normalized_energy_per_work_est_j_ci_high": _num(payload.get("host_normalized_energy_per_work_unit_est_j_ci_high")),
            "host_normalized_average_power_est_w": _num(_first(payload.get("avg_host_normalized_average_power_est_w"), payload.get("host_normalized_average_power_est_w"))),
            "host_normalization_role": str(payload.get("host_normalization_role") or "none"),
            "host_normalization_contract_present": "host_normalization_role" in payload,
            "host_normalization_source_run_id": str(payload.get("host_normalization_source_run_id") or ""),
            "host_normalization_target_variant": str(payload.get("host_normalization_target_variant") or ""),
            "host_normalization_identity_verified": payload.get("host_normalization_identity_verified") is True,
            "accelerator_idle_correction_requested": payload.get("accelerator_idle_correction_requested") is True,
            "accelerator_idle_correction_applied": payload.get("accelerator_idle_correction_applied") is True,
            "accelerator_idle_correction_statuses": payload.get("accelerator_idle_correction_statuses"),
            "accelerator_idle_w_applied": _num(payload.get("accelerator_idle_w_applied")),
            "accelerator_idle_calibration_verified": payload.get("accelerator_idle_calibration_verified") is True,
            "accelerator_idle_calibration_status": str(payload.get("accelerator_idle_calibration_status") or ""),
            "accelerator_idle_calibration_binding_path": str(payload.get("accelerator_idle_calibration_binding_path") or ""),
            "accelerator_idle_calibration_binding_sha256": str(payload.get("accelerator_idle_calibration_binding_sha256") or ""),
            "accelerator_idle_calibration_evidence": str(payload.get("accelerator_idle_calibration_evidence") or ""),
            "accelerator_idle_calibrated_at": str(payload.get("accelerator_idle_calibrated_at") or ""),
            "work_units": work_units,
            "work_units_source": str(payload.get("energy_work_units_source") or plan.get("work_units_source") or ""),
            "runtime_work_units_exact": exact_count,
            "energy_scope_requested": requested_scope,
            "energy_scope": measured_scope,
            "energy_physical_scope": measured_scope,
            "energy_window_requested": requested_window,
            "energy_window": effective_window,
            "energy_window_effective": effective_window,
            "energy_calibration_manifest": calibration_manifest,
            "energy_calibration_sha256": calibration_sha,
            "energy_calibration_verified": calibration_verified,
            "energy_calibration_status": calibration_status,
            "energy_calibration_required": calibration_required,
            "external_calibration_verified": external_calibration_verified,
            "external_calibration_status": (
                external_calibration_status
            ),
            "energy_primary_metric": str(payload.get("energy_primary_metric") or "unverified_postprocessor_energy"),
            "energy_calibrated_input_unsubtracted": payload.get("energy_calibrated_input_unsubtracted") is True,
            "primary_energy_input_verified": (
                payload.get(
                    "energy_calibrated_input_unsubtracted"
                ) is True
            ),
            "energy_raw_primary": payload.get("energy_raw_primary") is True,
            "energy_repeat_n": valid_repeats,
            "energy_repeat_valid_n": valid_repeats,
            "energy_repeat_requested_n": requested_repeats,
            "energy_repeat_observed_n": observed_repeat_count,
            "energy_repeat_status": repeat_status,
            "screening_comparable": bool(
                plan.get("screening_comparable") is True
                if "screening_comparable" in plan
                else claim_eligible
            ),
            "claim_comparable": bool(
                plan.get("claim_comparable") is True
                if "claim_comparable" in plan
                else claim_eligible
            ),
            "energy_confidence_level": confidence_level,
            "energy_total_j_mean": _num(primary_energy_stats.get("mean")) if primary_energy_stats else total_energy,
            "energy_total_j_sample_stddev": _num(primary_energy_stats.get("sample_stddev")),
            "energy_total_j_ci_low": _num(primary_energy_stats.get("ci_low")),
            "energy_total_j_ci_high": _num(primary_energy_stats.get("ci_high")),
            "energy_per_work_j_mean": _num(primary_per_work_stats.get("mean")) if primary_per_work_stats else energy_per_work,
            "energy_per_work_j_sample_stddev": _num(primary_per_work_stats.get("sample_stddev")),
            "energy_per_work_j_ci_low": _num(primary_per_work_stats.get("ci_low")),
            "energy_per_work_j_ci_high": _num(primary_per_work_stats.get("ci_high")),
            "average_power_w_mean": _num(primary_power_stats.get("mean")) if primary_power_stats else average_power,
            "average_power_w_ci_low": _num(primary_power_stats.get("ci_low")),
            "average_power_w_ci_high": _num(primary_power_stats.get("ci_high")),
            "candidate_energy_total_j_mean": (
                _num(shadow_energy_stats.get("mean"))
                if historical_chapter4_primary and shadow_energy_stats
                else legacy_candidate_total_energy
            ),
            "candidate_energy_total_j_ci_low": (
                _num(shadow_energy_stats.get("ci_low"))
                if historical_chapter4_primary else None
            ),
            "candidate_energy_total_j_ci_high": (
                _num(shadow_energy_stats.get("ci_high"))
                if historical_chapter4_primary else None
            ),
            "candidate_energy_per_work_j_mean": (
                _num(shadow_per_work_stats.get("mean"))
                if historical_chapter4_primary and shadow_per_work_stats
                else legacy_candidate_energy_per_work
            ),
            "candidate_energy_per_work_j_ci_low": (
                _num(shadow_per_work_stats.get("ci_low"))
                if historical_chapter4_primary else None
            ),
            "candidate_energy_per_work_j_ci_high": (
                _num(shadow_per_work_stats.get("ci_high"))
                if historical_chapter4_primary else None
            ),
            "energy_ab_requested": ab_requested,
            "energy_ab_attempted_n": ab_attempted_repeats,
            "energy_ab_valid_n": ab_valid_repeats,
            "energy_ab_status": energy_ab_status,
            "ab_difference_direction": str(ab_statistics.get("difference_direction") or "legacy_minus_command_window"),
            "ab_energy_relative_percent_mean": _num(ab_energy_stats.get("mean")),
            "ab_energy_relative_percent_ci_low": _num(ab_energy_stats.get("ci_low")),
            "ab_energy_relative_percent_ci_high": _num(ab_energy_stats.get("ci_high")),
            "ab_duration_relative_percent_mean": _num(ab_duration_stats.get("mean")),
            "ab_duration_relative_percent_ci_low": _num(ab_duration_stats.get("ci_low")),
            "ab_duration_relative_percent_ci_high": _num(ab_duration_stats.get("ci_high")),
            "ab_power_relative_percent_mean": _num(ab_power_stats.get("mean")),
            "ab_power_relative_percent_ci_low": _num(ab_power_stats.get("ci_low")),
            "ab_power_relative_percent_ci_high": _num(ab_power_stats.get("ci_high")),
            "scientific_primary_energy_statistics": primary_statistics,
            "scientific_shadow_energy_statistics": shadow_statistics,
            "chapter4_legacy_shadow_statistics": chapter4_shadow_statistics,
            "candidate_v263_shadow_statistics": candidate_statistics,
            "window_method_comparison_statistics": ab_statistics,
            "energy_aggregate_status": str(aggregate.get("status") or "unavailable"),
            "energy_aggregate_embedded": bool(run_info.get("energy_aggregate_embedded") or aggregate),
            "energy_aggregate_bound": bool(
                run_info.get("energy_aggregate_bound") is True
                or run_info.get("energy_aggregate_verified") is True
                or (
                    run_info.get("energy_aggregate_embedded") is True
                    and aggregate
                )
            ),
            "energy_aggregate_complete": bool(
                run_info.get("energy_aggregate_complete") is True
                or run_info.get("energy_aggregate_verified") is True
            ),
            "energy_aggregate_verified": bool(
                run_info.get("energy_aggregate_verified") is True
            ),
            "energy_aggregate_import_status": str(
                run_info.get("energy_aggregate_import_status") or ""
            ),
            "energy_aggregate_source_path": str(run_info.get("energy_aggregate_path") or ""),
            "pipeline_fps_per_watt": _num(payload.get("pipeline_fps_per_watt") or payload.get("pipeline_fps_per_watt_from_selected_fps")),
            "postprocess_status": str(payload.get("postprocess_status") or ""),
            "collector_reported_final_energy_gate_status": (
                collector_reported_final_gate_status
            ),
            "final_energy_gate_status": final_energy_gate_status,
            "ok": execution_ok,
            "measurement_status": (
                "available" if execution_ok else "measurement_failed"
            ),
            "measurement_failure_reason": measurement_failure_reason,
            "measurement_failure_category": str(
                item.get("failure_category")
                or ("" if execution_ok else "measurement_execution_failed")
            ),
            "measurement_run_rc": measurement_run_rc,
            "source": str(src),
            "observation_tier": "claim_eligible" if claim_eligible else "development_or_screening",
            "diagnostic_only": bool(
                plan.get("diagnostic_only") is True
                or payload.get("diagnostic_only") is True
            ),
            "claim_eligible": claim_eligible,
            "claim_ok": claim_eligible,
            "claim_exclusion_reasons": claim_reasons,
            "claim_exclusion_reason": ";".join(claim_reasons),
        })
        out[-1].update(resolve_energy_comparison(out[-1]))
        if (
            str(out[-1].get("backend") or "").strip().lower()
            == "native_full_tensorrt"
            and out[-1].get("energy_comparison_claim_ready") is not True
        ):
            reason = str(
                out[-1].get("energy_comparison_status")
                or "required_tensorrt_full_normalization_unavailable"
            )
            reasons = list(out[-1].get("claim_exclusion_reasons") or [])
            if reason != "host_normalized_verified" and reason not in reasons:
                reasons.append(reason)
            out[-1]["claim_exclusion_reasons"] = reasons
            out[-1]["claim_exclusion_reason"] = ";".join(reasons)
            out[-1]["claim_eligible"] = False
            out[-1]["claim_ok"] = False
            out[-1]["observation_tier"] = "development_or_screening"
    return out


def scientific_energy_rows(
    run_dir: str | Path | None = None,
    *,
    observations: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return every canonical Native energy attempt for the Scientific Report.

    ``observations`` permits a hardware-free replay from the already exported
    ``native_energy_observations.json``.  The normal workflow continues to
    collect the observations from ``run_dir``.
    """
    if observations is None:
        if run_dir is None:
            raise ValueError("run_dir or observations is required")
        source_observations = collect_native_energy(run_dir)
    else:
        source_observations = [
            item for item in observations if isinstance(item, Mapping)
        ]
    rows: list[dict[str, Any]] = []
    for item in source_observations:
        comparison = resolve_energy_comparison(item)
        measurement_ok = item.get("ok") is True
        measurement_status = (
            "available" if measurement_ok else "measurement_failed"
        )
        measurement_failure_reason = str(
            item.get("measurement_failure_reason") or ""
        ).strip()
        if not measurement_ok and not measurement_failure_reason:
            measurement_failure_reason = ";".join(
                value
                for value in (
                    (
                        f"postprocess_status={item.get('postprocess_status')}"
                        if item.get("postprocess_status")
                        else ""
                    ),
                    (
                        "final_energy_gate_status="
                        f"{item.get('final_energy_gate_status')}"
                        if item.get("final_energy_gate_status")
                        else ""
                    ),
                    (
                        f"energy_aggregate_status={item.get('energy_aggregate_status')}"
                        if item.get("energy_aggregate_status")
                        else ""
                    ),
                )
                if value
            ) or "measurement_execution_failed"
        rows.append({
            "row_role": "native_energy_measurement",
            "row_status": measurement_status,
            "model_id": item.get("model"),
            "task": item.get("task"),
            "case_id": item.get("case"),
            "backend": item.get("backend"),
            "variant": item.get("execution_mode"),
            "setup_id": item.get("setup_id"),
            "evaluation_role": item.get("evaluation_role"),
            "runner_regime": item.get("runner_regime"),
            "direction": item.get("direction"),
            "contract_hash": item.get("contract_hash"),
            "preprocessing_hash": item.get("preprocessing_hash"),
            "decoder_hash": item.get("decoder_hash"),
            "nms_hash": item.get("nms_hash"),
            "pipeline_contract_sha256": item.get("pipeline_contract_sha256"),
            "pipeline_preprocessing_sha256": item.get("pipeline_preprocessing_sha256"),
            "pipeline_decoder_sha256": item.get("pipeline_decoder_sha256"),
            "pipeline_nms_sha256": item.get("pipeline_nms_sha256"),
            "quality_contract_sha256": item.get("quality_contract_sha256"),
            "preprocessing_contract_sha256": item.get("preprocessing_contract_sha256"),
            "decoder_contract_sha256": item.get("decoder_contract_sha256"),
            "nms_contract_sha256": item.get("nms_contract_sha256"),
            "source_request_sha256": item.get("source_request_sha256"),
            "model_sha256": item.get("model_sha256"),
            "validation_dataset_sha256": item.get("validation_dataset_sha256"),
            "validation_dataset_image_ids_sha256": item.get("validation_dataset_image_ids_sha256"),
            "validation_dataset_ground_truth_sha256": item.get("validation_dataset_ground_truth_sha256"),
            "accuracy_gate_policy_sha256": item.get("accuracy_gate_policy_sha256"),
            "task_quality_policy_sha256": item.get("task_quality_policy_sha256"),
            "runtime_quality_gate_policy_sha256": item.get("runtime_quality_gate_policy_sha256"),
            "validation_input_or_image_sha256": item.get("validation_input_or_image_sha256"),
            "energy_quality_qualified": item.get(
                "energy_quality_qualified"
            ),
            "energy_quality_status": item.get(
                "energy_quality_status"
            ),
            "native_energy_after_technical_error": item.get(
                "native_energy_after_technical_error"
            ),
            "prepared_feed_task": item.get("prepared_feed_task"),
            "prepared_feed_preprocess_mode": item.get("prepared_feed_preprocess_mode"),
            "prepared_feed_letterbox_pad_value": item.get("prepared_feed_letterbox_pad_value"),
            "prepared_feed_source_image_sha256": item.get("prepared_feed_source_image_sha256"),
            "prepared_feed_identity_source": item.get("prepared_feed_identity_source"),
            "comparison_backend": item.get("comparison_backend"),
            "execution_mode": item.get("execution_mode"),
            "precision": item.get("precision"),
            "precision_role": item.get("precision_role"),
            "execution_precision": item.get("execution_precision"),
            "split_boundary_precision": item.get("split_boundary_precision"),
            "full_runtime_precision": item.get("full_runtime_precision"),
            "comparison_precision": item.get("comparison_precision"),
            "legacy_comparison_precision": item.get("legacy_comparison_precision"),
            "runtime_precision_status": item.get("runtime_precision_status"),
            "output_endpoint_id": item.get("output_endpoint_id"),
            "stage": item.get("stage"),
            "contract_family": item.get("contract_family"),
            "endpoint_contract_hash": item.get("endpoint_contract_hash"),
            "endpoint_contract_complete": item.get("endpoint_contract_complete"),
            "output_endpoint_match": item.get("output_endpoint_match"),
            "physical_output_endpoint_id": item.get(
                "physical_output_endpoint_id"
            ),
            "physical_endpoint_contract_hash": item.get(
                "physical_endpoint_contract_hash"
            ),
            "physical_endpoint_stage": item.get(
                "physical_endpoint_stage"
            ),
            "physical_endpoint_contract_complete": item.get(
                "physical_endpoint_contract_complete"
            ),
            "physical_output_endpoint_match": item.get(
                "physical_output_endpoint_match"
            ),
            "physical_endpoint_identity_status": item.get(
                "physical_endpoint_identity_status"
            ),
            "comparison_output_endpoint_id": item.get(
                "comparison_output_endpoint_id"
            ),
            "comparison_endpoint_contract_hash": item.get(
                "comparison_endpoint_contract_hash"
            ),
            "comparison_endpoint_stage": item.get(
                "comparison_endpoint_stage"
            ),
            "completion_pairing_eligible": item.get(
                "completion_pairing_eligible"
            ),
            "completion_pairing_status": item.get(
                "completion_pairing_status"
            ),
            "completion_pairing_validation_status": item.get(
                "completion_pairing_validation_status"
            ),
            "central_quality_evidence_verified": item.get("central_quality_evidence_verified"),
            "precision_quality_verified": item.get("precision_quality_verified"),
            "precision_quality_binding_verified": item.get(
                "precision_quality_binding_verified"
            ),
            "task_quality_observation_valid": item.get(
                "task_quality_observation_valid"
            ),
            "accuracy_gate_pass": item.get("accuracy_gate_pass"),
            **assessment_fields(item.get("accuracy_assessment")),
            "quality_claim_result_verified": item.get(
                "quality_claim_result_verified"
            ),
            "quality_verified": item.get("quality_verified"),
            "quality_provenance_complete": item.get("quality_provenance_complete"),
            "quality_gate_status": item.get("quality_gate_status"),
            "native_validation_join_status": item.get("native_validation_join_status"),
            "throughput_fps": item.get("fps"),
            "target_duration_s": item.get("target_duration_s"),
            "active_duration_s": item.get("active_duration_s"),
            "active_duration_relative_tolerance": item.get("active_duration_relative_tolerance"),
            "active_duration_comparison_policy": item.get("active_duration_comparison_policy"),
            "active_duration_tolerance_source": item.get("active_duration_tolerance_source"),
            "average_power_w": comparison.get("comparison_average_power_w"),
            "raw_average_power_w": comparison.get("raw_average_power_w"),
            "host_normalized_average_power_est_w": comparison.get("host_normalized_average_power_est_w"),
            "energy_per_work_j": comparison.get("comparison_energy_per_work_j"),
            "raw_energy_per_work_j": comparison.get("raw_energy_per_work_j"),
            "host_normalized_energy_per_work_est_j": comparison.get("host_normalized_energy_per_work_est_j"),
            "energy_total_j": comparison.get("comparison_energy_total_j"),
            "raw_energy_total_j": comparison.get("raw_energy_total_j"),
            "energy_comparison_basis": comparison.get("energy_comparison_basis"),
            "energy_comparison_status": comparison.get("energy_comparison_status"),
            "energy_comparison_claim_ready": comparison.get("energy_comparison_claim_ready"),
            "scientific_primary_method": item.get("scientific_primary_method"),
            "scientific_primary_method_frozen": item.get("scientific_primary_method_frozen"),
            "scientific_primary_contract_generation": item.get("scientific_primary_contract_generation"),
            "scientific_shadow_method": item.get("scientific_shadow_method"),
            "scientific_shadow_role": item.get("scientific_shadow_role"),
            "scientific_shadow_energy_status": item.get("scientific_shadow_energy_status"),
            "scientific_shadow_valid_run_count": item.get("scientific_shadow_valid_run_count"),
            "scientific_shadow_affects_primary_claim": False,
            "scientific_shadow_affects_primary_result": False,
            "scientific_shadow_affects_final_gate": False,
            "shadow_energy_per_work_j": item.get("shadow_energy_per_work_j"),
            "shadow_energy_total_j": item.get("shadow_energy_total_j"),
            "energy_work_units": item.get("work_units"),
            "energy_work_units_source": item.get("work_units_source"),
            "energy_scope": item.get("energy_scope"),
            "energy_window": item.get("energy_window"),
            "energy_window_effective": item.get("energy_window_effective"),
            "energy_calibration_sha256": item.get("energy_calibration_sha256"),
            "energy_calibration_verified": item.get("energy_calibration_verified"),
            "energy_calibration_manifest": item.get(
                "energy_calibration_manifest"
            ),
            "energy_calibration_status": item.get(
                "energy_calibration_status"
            ),
            "energy_calibration_required": item.get(
                "energy_calibration_required"
            ),
            "external_calibration_verified": item.get(
                "external_calibration_verified"
            ),
            "external_calibration_status": item.get(
                "external_calibration_status"
            ),
            "energy_primary_metric": item.get("energy_primary_metric"),
            "energy_calibrated_input_unsubtracted": item.get(
                "energy_calibrated_input_unsubtracted"
            ),
            "primary_energy_input_verified": item.get(
                "primary_energy_input_verified"
            ),
            "energy_raw_primary": item.get("energy_raw_primary"),
            "scientific_primary_energy_status": item.get(
                "scientific_primary_energy_status"
            ),
            "scientific_primary_claim_eligible": item.get(
                "scientific_primary_claim_eligible"
            ),
            "energy_dynamic_j": item.get("energy_dynamic_j"),
            "host_normalized_energy_est_j": item.get("host_normalized_energy_est_j"),
            "energy_repeat_n": item.get("energy_repeat_n"),
            "energy_repeat_valid_n": item.get("energy_repeat_valid_n"),
            "energy_repeat_requested_n": item.get("energy_repeat_requested_n"),
            "energy_repeat_observed_n": item.get(
                "energy_repeat_observed_n"
            ),
            "energy_repeat_status": item.get("energy_repeat_status"),
            "screening_comparable": item.get(
                "screening_comparable"
            ) is True,
            "claim_comparable": item.get(
                "claim_comparable"
            ) is True,
            "energy_confidence_level": item.get("energy_confidence_level"),
            "energy_per_work_sample_stddev_j": comparison.get("comparison_energy_per_work_sample_stddev_j"),
            "energy_per_work_ci_low_j": comparison.get("comparison_energy_per_work_ci_low_j"),
            "energy_per_work_ci_high_j": comparison.get("comparison_energy_per_work_ci_high_j"),
            "average_power_ci_low_w": item.get("average_power_w_ci_low"),
            "average_power_ci_high_w": item.get("average_power_w_ci_high"),
            "energy_ab_status": item.get("energy_ab_status"),
            "energy_ab_valid_n": item.get("energy_ab_valid_n"),
            "ab_energy_relative_percent_mean": item.get("ab_energy_relative_percent_mean"),
            "ab_energy_relative_percent_ci_low": item.get("ab_energy_relative_percent_ci_low"),
            "ab_energy_relative_percent_ci_high": item.get("ab_energy_relative_percent_ci_high"),
            "measurement_ok": measurement_ok,
            "measurement_status": measurement_status,
            "measurement_failure_reason": measurement_failure_reason,
            "measurement_failure_category": item.get("measurement_failure_category"),
            "diagnostic_only": item.get("diagnostic_only") is True,
            "declared_claim_eligible": item.get("claim_eligible") is True,
            "measurement_run_rc": item.get("measurement_run_rc"),
            "postprocess_status": item.get("postprocess_status"),
            "collector_reported_final_energy_gate_status": item.get(
                "collector_reported_final_energy_gate_status"
            ),
            "final_energy_gate_status": item.get("final_energy_gate_status"),
            "energy_aggregate_status": item.get("energy_aggregate_status"),
            "performance_eligible": False,
            "energy_eligible": bool(
                measurement_ok and item.get("claim_eligible")
            ),
            "ranking_eligible": False,
            "eligibility_status": (
                "claim_eligible"
                if measurement_ok and item.get("claim_eligible")
                else "screening_only"
                if measurement_ok
                else "measurement_failed"
            ),
            "exclusion_reason": (
                ""
                if measurement_ok and item.get("claim_eligible")
                else item.get("claim_exclusion_reason")
                or measurement_failure_reason
                or "measurement_execution_failed"
            ),
            "scientific_claim_exclusion_reasons": list(
                item.get("claim_exclusion_reasons") or []
            ),
            "source_path": item.get("source"),
            "source_kind": "native_energy_measurement",
            "native_semantic_gate": item.get("semantic_gate"),
        })
    return rows


def build_native_energy_ab_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return one explicit repeat/A-B aggregate per measured Native row."""
    fields = (
        "model", "task", "backend", "case", "execution_mode", "setup_id",
        "comparison_backend", "direction", "energy_repeat_valid_n",
        "source_request_sha256", "model_sha256", "validation_dataset_sha256",
        "validation_dataset_image_ids_sha256",
        "validation_dataset_ground_truth_sha256",
        "accuracy_gate_policy_sha256",
        "task_quality_policy_sha256", "runtime_quality_gate_policy_sha256",
        "quality_contract_sha256", "preprocessing_contract_sha256",
        "decoder_contract_sha256", "nms_contract_sha256",
        "quality_provenance_complete",
        "energy_repeat_requested_n", "energy_repeat_observed_n",
        "energy_repeat_status", "energy_confidence_level",
        "energy_total_j_mean", "energy_total_j_sample_stddev",
        "energy_total_j_ci_low", "energy_total_j_ci_high",
        "energy_per_work_j_mean", "energy_per_work_j_sample_stddev",
        "energy_per_work_j_ci_low", "energy_per_work_j_ci_high",
        "average_power_w_mean", "average_power_w_ci_low", "average_power_w_ci_high",
        "candidate_energy_total_j_mean", "candidate_energy_total_j_ci_low",
        "candidate_energy_total_j_ci_high", "candidate_energy_per_work_j_mean",
        "candidate_energy_per_work_j_ci_low", "candidate_energy_per_work_j_ci_high",
        "energy_ab_requested", "energy_ab_attempted_n", "energy_ab_valid_n",
        "energy_ab_status", "ab_difference_direction",
        "ab_energy_relative_percent_mean", "ab_energy_relative_percent_ci_low",
        "ab_energy_relative_percent_ci_high", "ab_duration_relative_percent_mean",
        "ab_duration_relative_percent_ci_low", "ab_duration_relative_percent_ci_high",
        "ab_power_relative_percent_mean", "ab_power_relative_percent_ci_low",
        "ab_power_relative_percent_ci_high", "scientific_primary_method",
        "scientific_primary_method_frozen", "scientific_primary_contract_generation",
        "scientific_shadow_method", "scientific_shadow_role",
        "scientific_shadow_energy_status", "scientific_shadow_valid_run_count",
        "scientific_shadow_affects_primary_claim",
        "scientific_shadow_affects_primary_result",
        "scientific_shadow_affects_final_gate",
        "shadow_energy_total_j", "shadow_energy_per_work_j",
        "shadow_energy_total_j_mean", "shadow_energy_total_j_ci_low",
        "shadow_energy_total_j_ci_high", "shadow_energy_per_work_j_mean",
        "shadow_energy_per_work_j_ci_low", "shadow_energy_per_work_j_ci_high",
        "chapter4_legacy_shadow_energy_total_j",
        "chapter4_legacy_shadow_energy_per_work_j",
        "candidate_method", "candidate_role", "energy_aggregate_status",
        "energy_aggregate_embedded", "energy_aggregate_source_path",
        "measurement_status", "measurement_failure_reason",
        "measurement_failure_category", "energy_calibration_manifest",
        "energy_calibration_sha256", "energy_calibration_status",
        "energy_calibration_required", "energy_calibration_verified",
        "external_calibration_status", "external_calibration_verified",
        "energy_calibrated_input_unsubtracted",
        "primary_energy_input_verified",
        "scientific_primary_energy_status",
        "scientific_primary_claim_eligible",
        "claim_eligible", "claim_exclusion_reason",
    )
    out: list[dict[str, Any]] = []
    for source in rows:
        row = {field: source.get(field) for field in fields}
        # Preserve the complete statistics blocks in JSON. CSV serialisation is
        # intentionally a compact textual cell rather than one row per repeat.
        row["scientific_primary_energy_statistics"] = source.get("scientific_primary_energy_statistics") or {}
        row["scientific_shadow_energy_statistics"] = source.get("scientific_shadow_energy_statistics") or {}
        row["chapter4_legacy_shadow_statistics"] = source.get("chapter4_legacy_shadow_statistics") or {}
        row["candidate_v263_shadow_statistics"] = source.get("candidate_v263_shadow_statistics") or {}
        row["window_method_comparison_statistics"] = source.get("window_method_comparison_statistics") or {}
        out.append(row)
    return out


def build_native_energy_pairs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Full baselines have no Split boundary.  Match them by the setup-local
    # experiment identity and preserve their runtime precision separately.
    # Multiple candidates at this identity remain visible and fail closed.
    full: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row.get("execution_mode") or "") == "native_full_baseline" and bool(row.get("ok")):
            key = (
                str(row.get("model") or ""),
                str(row.get("setup_id") or ""),
                str(row.get("backend") or ""),
                str(row.get("direction") or ""),
            )
            full.setdefault(key, []).append(row)
    producer_vendor = {
        "hailo8_to_trt": "native_full_hailo8",
        "hailo10h_to_trt": "native_full_hailo10h",
        "deepx_to_trt": "native_full_deepx",
    }
    pairs: list[dict[str, Any]] = []
    for split in rows:
        if str(split.get("execution_mode") or "") != "native_split" or not bool(split.get("ok")):
            continue
        split_backend = str(split.get("backend") or "")
        baselines = [
            ("vendor_full", producer_vendor.get(split_backend)),
            ("tensorrt_full", "native_full_tensorrt"),
        ]
        for kind, baseline_backend in baselines:
            if not baseline_backend:
                continue
            baseline_candidates = list(full.get((
                str(split.get("model") or ""),
                str(split.get("setup_id") or ""),
                baseline_backend,
                str(split.get("direction") or ""),
            )) or [])
            identity_fields = (
                "task", "model_sha256", "validation_input_or_image_sha256",
                "validation_dataset_sha256",
                "validation_dataset_image_ids_sha256",
                "validation_dataset_ground_truth_sha256",
                "accuracy_gate_policy_sha256",
                "task_quality_policy_sha256",
                "runtime_quality_gate_policy_sha256",
                "prepared_feed_task", "prepared_feed_preprocess_mode",
                "prepared_feed_letterbox_pad_value",
                "prepared_feed_source_image_sha256",
            )
            endpoint_evidence_declared = any(
                bool(
                    str(row.get("output_endpoint_id") or "").strip()
                    or str(row.get("endpoint_contract_hash") or "").strip()
                    or str(
                        row.get("physical_output_endpoint_id") or ""
                    ).strip()
                    or str(
                        row.get("comparison_output_endpoint_id") or ""
                    ).strip()
                    or row.get("endpoint_contract_complete") is True
                    or (
                        str(row.get("task") or "").strip().lower()
                        == "detection"
                        and "completion_pairing_eligible" in row
                    )
                    or str(row.get("native_validation_join_status") or "")
                    not in {"", "missing"}
                )
                for row in ([split] + baseline_candidates)
            )
            identity_matches = [
                row for row in baseline_candidates
                if all(str(split.get(field) or "") and str(split.get(field) or "") == str(row.get(field) or "") for field in identity_fields)
            ]
            if len(identity_matches) == 1:
                baseline = identity_matches[0]
                baseline_ambiguous = False
            elif not identity_matches and len(baseline_candidates) == 1:
                baseline = baseline_candidates[0]
                baseline_ambiguous = False
            else:
                baseline = None
                baseline_ambiguous = len(baseline_candidates) > 1 or len(identity_matches) > 1
            task = str(split.get("task") or "").strip().lower()
            split_comparison = resolve_energy_comparison(split)
            baseline_comparison = resolve_energy_comparison(baseline or {})
            split_energy = _num(split_comparison.get("comparison_energy_per_work_j"))
            baseline_energy = _num(baseline_comparison.get("comparison_energy_per_work_j")) if baseline else None
            baseline_normalization_required = bool(
                kind == "tensorrt_full" and baseline is not None
            )
            baseline_normalization_ready = bool(
                not baseline_normalization_required
                or (
                    baseline_comparison.get("energy_comparison_basis")
                    == "host_normalized_accelerator_idle_subtracted"
                    and baseline_comparison.get("energy_comparison_status") == "host_normalized_verified"
                )
            )
            split_precision = str(
                split.get("split_boundary_precision") or split.get("precision") or ""
            )
            baseline_runtime_precision = str(
                baseline.get("full_runtime_precision") or baseline.get("execution_precision") or ""
            ) if baseline else ""
            baseline_comparison_precision = str(
                baseline.get("comparison_precision")
                or baseline.get("legacy_comparison_precision")
                or baseline.get("precision")
                or ""
            ) if baseline else ""
            split_target_duration = _num(_first(
                split.get("target_duration_s"), split.get("duration_s"),
            ))
            baseline_target_duration = _num(_first(
                baseline.get("target_duration_s"), baseline.get("duration_s"),
            )) if baseline else None
            split_active_duration = _num(split.get("active_duration_s"))
            baseline_active_duration = _num(
                baseline.get("active_duration_s")
            ) if baseline else None
            split_target_valid = bool(
                split_target_duration is not None
                and math.isfinite(split_target_duration)
                and split_target_duration > 0
            )
            baseline_target_valid = bool(
                baseline_target_duration is not None
                and math.isfinite(baseline_target_duration)
                and baseline_target_duration > 0
            )
            target_duration_match = bool(
                split_target_valid
                and baseline_target_valid
                and math.isclose(
                    float(split_target_duration), float(baseline_target_duration),
                    rel_tol=1e-9, abs_tol=1e-9,
                )
            )
            split_active_valid = bool(
                split_active_duration is not None
                and math.isfinite(split_active_duration)
                and split_active_duration > 0
            )
            baseline_active_valid = bool(
                baseline_active_duration is not None
                and math.isfinite(baseline_active_duration)
                and baseline_active_duration > 0
            )
            active_duration_absolute_delta = None
            active_duration_relative_delta = None
            active_duration_within_tolerance: bool | None = None
            if (
                target_duration_match
                and split_active_valid
                and baseline_active_valid
                and split_target_duration is not None
                and split_active_duration is not None
                and baseline_active_duration is not None
            ):
                active_duration_absolute_delta = abs(
                    split_active_duration - baseline_active_duration
                )
                active_duration_relative_delta = (
                    active_duration_absolute_delta / split_target_duration
                )
                active_duration_within_tolerance = bool(
                    active_duration_relative_delta
                    <= NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE
                )
            reasons: list[str] = []
            if baseline is None:
                reasons.append("baseline_ambiguous" if baseline_ambiguous else "baseline_missing")
            else:
                required_equal = (
                    ("setup_id", "setup_mismatch_or_missing"),
                    ("task", "task_mismatch_or_missing"),
                    ("direction", "direction_mismatch_or_missing"),
                    ("model_sha256", "model_hash_mismatch_or_missing"),
                    ("validation_dataset_sha256", "validation_dataset_hash_mismatch_or_missing"),
                    ("validation_dataset_image_ids_sha256", "validation_image_ids_hash_mismatch_or_missing"),
                    ("validation_dataset_ground_truth_sha256", "validation_ground_truth_hash_mismatch_or_missing"),
                    ("accuracy_gate_policy_sha256", "accuracy_gate_policy_hash_mismatch_or_missing"),
                    ("task_quality_policy_sha256", "task_quality_policy_hash_mismatch_or_missing"),
                    ("runtime_quality_gate_policy_sha256", "runtime_quality_policy_hash_mismatch_or_missing"),
                    ("validation_input_or_image_sha256", "validation_input_or_image_hash_mismatch_or_missing"),
                    ("prepared_feed_task", "prepared_feed_task_mismatch_or_missing"),
                    ("prepared_feed_preprocess_mode", "prepared_feed_preprocess_mode_mismatch_or_missing"),
                    ("prepared_feed_letterbox_pad_value", "prepared_feed_pad_value_mismatch_or_missing"),
                    ("prepared_feed_source_image_sha256", "prepared_feed_source_image_hash_mismatch_or_missing"),
                    ("energy_scope", "physical_scope_mismatch_or_missing"),
                    ("energy_window_effective", "energy_window_mismatch_or_missing"),
                )
                hash_identity_fields = {
                    "model_sha256", "validation_dataset_sha256",
                    "validation_dataset_image_ids_sha256",
                    "validation_dataset_ground_truth_sha256",
                    "accuracy_gate_policy_sha256",
                    "task_quality_policy_sha256",
                    "runtime_quality_gate_policy_sha256",
                    "validation_input_or_image_sha256",
                    "prepared_feed_source_image_sha256",
                }
                for field, reason in required_equal:
                    left = str(split.get(field) or "")
                    right = str(baseline.get(field) or "")
                    if field in hash_identity_fields:
                        left = _strict_sha256_token(left)
                        right = _strict_sha256_token(right)
                    if not left or left in {"unknown", "mixed"} or left != right:
                        reasons.append(reason)
                required_per_row_quality_hashes = [
                    "source_request_sha256", "model_sha256",
                    "validation_dataset_sha256",
                    "validation_dataset_image_ids_sha256",
                    "validation_dataset_ground_truth_sha256",
                    "accuracy_gate_policy_sha256",
                    "task_quality_policy_sha256",
                    "runtime_quality_gate_policy_sha256",
                    "quality_contract_sha256", "preprocessing_contract_sha256",
                ]
                if task == "detection":
                    required_per_row_quality_hashes.extend((
                        "decoder_contract_sha256", "nms_contract_sha256",
                    ))
                for side, evidence_row in (("split", split), ("baseline", baseline)):
                    for field in required_per_row_quality_hashes:
                        if not _strict_sha256_token(evidence_row.get(field)):
                            reasons.append(f"{side}_{field}_missing_or_invalid")
                    accuracy_policy = _strict_sha256_token(
                        evidence_row.get("accuracy_gate_policy_sha256")
                    )
                    task_policy = _strict_sha256_token(
                        evidence_row.get("task_quality_policy_sha256")
                    )
                    runtime_policy = _strict_sha256_token(
                        evidence_row.get("runtime_quality_gate_policy_sha256")
                    )
                    if (
                        accuracy_policy and task_policy and runtime_policy
                        and len({accuracy_policy, task_policy, runtime_policy}) != 1
                    ):
                        reasons.append(f"{side}_quality_policy_hash_conflict")
                # A frozen pipeline contract, when present, is common pair
                # evidence and must match.  Producer-specific Central-Quality
                # contracts are checked per row and deliberately are not
                # required to have the same implementation hash.
                for field, reason in (
                    ("pipeline_contract_sha256", "pipeline_contract_mismatch_or_missing"),
                    ("pipeline_preprocessing_sha256", "pipeline_preprocessing_mismatch_or_missing"),
                    ("pipeline_decoder_sha256", "pipeline_decoder_mismatch_or_missing"),
                    ("pipeline_nms_sha256", "pipeline_nms_mismatch_or_missing"),
                ):
                    left = str(split.get(field) or "")
                    right = str(baseline.get(field) or "")
                    if left or right:
                        if not left or left != right:
                            reasons.append(reason)
                # Legacy imported rows predate the explicit pipeline/quality
                # namespaces.  Retain their strict all-contract comparison,
                # but use it only when neither row carries namespaced evidence.
                # New rows may legitimately use different producer-quality
                # implementation hashes and are compared through their common
                # frozen pipeline identity plus per-row quality gates instead.
                namespaced_fields = (
                    "pipeline_contract_sha256", "pipeline_preprocessing_sha256",
                    "pipeline_decoder_sha256", "pipeline_nms_sha256",
                    "quality_contract_sha256", "preprocessing_contract_sha256",
                    "decoder_contract_sha256", "nms_contract_sha256",
                )
                if not any(
                    str(row.get(field) or "").strip()
                    for row in (split, baseline) for field in namespaced_fields
                ):
                    legacy_fields = [
                        ("contract_hash", "contract_hash_mismatch_or_missing"),
                        ("preprocessing_hash", "preprocessing_hash_mismatch_or_missing"),
                    ]
                    if task == "detection":
                        legacy_fields.extend([
                            ("decoder_hash", "decoder_hash_mismatch_or_missing"),
                            ("nms_hash", "nms_hash_mismatch_or_missing"),
                        ])
                    for field, reason in legacy_fields:
                        left = str(split.get(field) or "")
                        right = str(baseline.get(field) or "")
                        if not left or left != right:
                            reasons.append(reason)
                if endpoint_evidence_declared:
                    split_endpoint, split_endpoint_hash = _canonical_row_endpoint(split)
                    baseline_endpoint, baseline_endpoint_hash = (
                        _canonical_row_endpoint(baseline)
                    )
                    if not split_endpoint or split_endpoint != baseline_endpoint:
                        reasons.append("output_endpoint_mismatch_or_missing")
                    if (
                        not split_endpoint_hash or not baseline_endpoint_hash
                        or split_endpoint_hash != baseline_endpoint_hash
                    ):
                        reasons.append("endpoint_contract_hash_mismatch_or_missing")
                    for side, row in (("split", split), ("baseline", baseline)):
                        if row.get("endpoint_contract_complete") is not True:
                            reasons.append(f"{side}_endpoint_contract_incomplete")
                        if row.get("output_endpoint_match") is not True:
                            reasons.append(f"{side}_output_endpoint_not_verified")
                if not split_precision:
                    reasons.append("split_boundary_precision_missing")
                for side, row in (("split", split), ("baseline", baseline)):
                    if str(row.get("task_source") or "") != "declared":
                        reasons.append(f"{side}_task_not_explicitly_declared")
                    if row.get("claim_eligible") is not True:
                        reasons.append(f"{side}_claim_gate_failed")
                    if row.get("semantic_claim_ok") is not True:
                        reasons.append(f"{side}_semantic_claim_failed")
                    if row.get("contract_consistent") is not True:
                        reasons.append(f"{side}_contract_inconsistent")
                    if row.get("runtime_work_units_exact") is not True:
                        reasons.append(f"{side}_runtime_work_units_not_exact")
                    if not _calibrated_unsubtracted_input_primary(row):
                        reasons.append(f"{side}_calibrated_unsubtracted_input_energy_not_primary")
                    if str(row.get("energy_window_effective") or "") != "command_window":
                        reasons.append(f"{side}_command_window_not_verified")
                if str(split.get("energy_scope") or "") == "full_system":
                    if split.get("energy_calibration_verified") is not True:
                        reasons.append("split_full_system_calibration_not_verified")
                    if baseline.get("energy_calibration_verified") is not True:
                        reasons.append("baseline_full_system_calibration_not_verified")
                    left_cal = str(split.get("energy_calibration_sha256") or "")
                    right_cal = str(baseline.get("energy_calibration_sha256") or "")
                    if not left_cal or left_cal != right_cal:
                        reasons.append("calibration_hash_mismatch_or_missing")
                if split_energy is None or baseline_energy is None:
                    reasons.append("comparison_energy_per_work_missing")
                if not baseline_normalization_ready:
                    reasons.append("baseline_required_host_normalization_unavailable")
                if not split_target_valid:
                    reasons.append("split_target_duration_missing_or_invalid")
                if not baseline_target_valid:
                    reasons.append("baseline_target_duration_missing_or_invalid")
                if split_target_valid and baseline_target_valid and not target_duration_match:
                    reasons.append("target_duration_mismatch")
                if not split_active_valid:
                    reasons.append("split_active_duration_missing_or_invalid")
                if not baseline_active_valid:
                    reasons.append("baseline_active_duration_missing_or_invalid")
                if active_duration_within_tolerance is False:
                    reasons.append("active_duration_relative_tolerance_exceeded")
            reasons = list(dict.fromkeys(reasons))
            comparable = baseline is not None and not reasons
            screening_reasons: list[str] = []
            if baseline is None:
                screening_reasons.append(
                    "baseline_ambiguous"
                    if baseline_ambiguous else "baseline_missing"
                )
            else:
                if split.get("screening_comparable") is not True:
                    screening_reasons.append(
                        "split_screening_not_comparable"
                    )
                if baseline.get("screening_comparable") is not True:
                    screening_reasons.append(
                        "baseline_screening_not_comparable"
                    )
                for field, reason in (
                    ("setup_id", "setup_mismatch_or_missing"),
                    ("task", "task_mismatch_or_missing"),
                    ("direction", "direction_mismatch_or_missing"),
                    ("energy_scope", "physical_scope_mismatch_or_missing"),
                    (
                        "energy_window_effective",
                        "energy_window_mismatch_or_missing",
                    ),
                ):
                    left = str(split.get(field) or "")
                    right = str(baseline.get(field) or "")
                    if not left or left != right:
                        screening_reasons.append(reason)
                if split_energy is None or baseline_energy is None:
                    screening_reasons.append(
                        "comparison_energy_per_work_missing"
                    )
                if not baseline_normalization_ready:
                    screening_reasons.append(
                        "baseline_required_host_normalization_unavailable"
                    )
                if (
                    not split_target_valid
                    or not baseline_target_valid
                    or not target_duration_match
                ):
                    screening_reasons.append(
                        "target_duration_mismatch_or_missing"
                    )
                if (
                    not split_active_valid
                    or not baseline_active_valid
                    or active_duration_within_tolerance is False
                ):
                    screening_reasons.append(
                        "active_duration_not_comparable"
                    )
            screening_reasons = list(dict.fromkeys(screening_reasons))
            screening_comparable = bool(
                baseline is not None and not screening_reasons
            )
            pairs.append({
                "model": split.get("model"),
                "task": split.get("task"),
                "evaluation_role": split.get("evaluation_role"),
                "direction": split.get("direction"),
                # Compatibility alias plus unambiguous scientific labels.
                "precision": split_precision,
                "split_precision": split_precision,
                "split_boundary_precision": split_precision,
                "baseline_comparison_precision": baseline_comparison_precision,
                "baseline_legacy_comparison_precision": baseline_comparison_precision,
                "baseline_runtime_precision": baseline_runtime_precision,
                "comparison_precision_match_required": False,
                "full_runtime_precision_match_required": False,
                "case": split.get("case"),
                "setup_id": split.get("setup_id"),
                "split_backend": split_backend,
                "baseline_kind": kind,
                "baseline_backend": baseline_backend,
                "baseline_candidate_count": len(baseline_candidates),
                "baseline_candidate_comparison_precisions": sorted({
                    str(
                        row.get("comparison_precision")
                        or row.get("legacy_comparison_precision")
                        or row.get("precision")
                        or ""
                    )
                    for row in baseline_candidates
                }),
                "baseline_candidate_runtime_precisions": sorted({
                    str(row.get("full_runtime_precision") or row.get("execution_precision") or "")
                    for row in baseline_candidates
                    if str(row.get("full_runtime_precision") or row.get("execution_precision") or "")
                }),
                "comparable": comparable,
                "screening_comparable": screening_comparable,
                "claim_comparable": comparable,
                "comparison_status": (
                    "claim_comparable"
                    if comparable
                    else "screening_comparable"
                    if screening_comparable
                    else "ineligible_or_contract_mismatch"
                ),
                "screening_comparison_reasons": screening_reasons,
                "screening_comparison_reason": ";".join(
                    screening_reasons
                ),
                "comparison_reasons": reasons,
                "comparison_reason": ";".join(reasons),
                "contract_hash": split.get("contract_hash"),
                "preprocessing_hash": split.get("preprocessing_hash"),
                "decoder_hash": split.get("decoder_hash"),
                "nms_hash": split.get("nms_hash"),
                "pipeline_contract_sha256": split.get("pipeline_contract_sha256"),
                "pipeline_preprocessing_sha256": split.get("pipeline_preprocessing_sha256"),
                "pipeline_decoder_sha256": split.get("pipeline_decoder_sha256"),
                "pipeline_nms_sha256": split.get("pipeline_nms_sha256"),
                "quality_contract_sha256": split.get("quality_contract_sha256"),
                "preprocessing_contract_sha256": split.get("preprocessing_contract_sha256"),
                "decoder_contract_sha256": split.get("decoder_contract_sha256"),
                "nms_contract_sha256": split.get("nms_contract_sha256"),
                "source_request_sha256": split.get("source_request_sha256"),
                "split_source_request_sha256": split.get("source_request_sha256"),
                "baseline_source_request_sha256": (
                    baseline.get("source_request_sha256") if baseline else None
                ),
                "output_endpoint_id": split.get("output_endpoint_id"),
                "split_output_endpoint_id": split.get("output_endpoint_id"),
                "baseline_output_endpoint_id": (
                    baseline.get("output_endpoint_id") if baseline else None
                ),
                "comparison_output_endpoint_id": split.get(
                    "comparison_output_endpoint_id"
                ),
                "split_comparison_output_endpoint_id": split.get(
                    "comparison_output_endpoint_id"
                ),
                "baseline_comparison_output_endpoint_id": (
                    baseline.get("comparison_output_endpoint_id")
                    if baseline else None
                ),
                "physical_output_endpoint_id": split.get(
                    "physical_output_endpoint_id"
                ),
                "split_physical_output_endpoint_id": split.get(
                    "physical_output_endpoint_id"
                ),
                "baseline_physical_output_endpoint_id": (
                    baseline.get("physical_output_endpoint_id")
                    if baseline else None
                ),
                "endpoint_contract_hash": split.get("endpoint_contract_hash"),
                "comparison_endpoint_contract_hash": split.get(
                    "comparison_endpoint_contract_hash"
                ),
                "split_comparison_endpoint_contract_hash": split.get(
                    "comparison_endpoint_contract_hash"
                ),
                "baseline_comparison_endpoint_contract_hash": (
                    baseline.get("comparison_endpoint_contract_hash")
                    if baseline else None
                ),
                "physical_endpoint_contract_hash": split.get(
                    "physical_endpoint_contract_hash"
                ),
                "split_physical_endpoint_contract_hash": split.get(
                    "physical_endpoint_contract_hash"
                ),
                "baseline_physical_endpoint_contract_hash": (
                    baseline.get("physical_endpoint_contract_hash")
                    if baseline else None
                ),
                "split_completion_pairing_eligible": split.get(
                    "completion_pairing_eligible"
                ),
                "baseline_completion_pairing_eligible": (
                    baseline.get("completion_pairing_eligible")
                    if baseline else None
                ),
                "endpoint_contract_complete": split.get("endpoint_contract_complete"),
                "model_sha256": split.get("model_sha256"),
                "validation_dataset_sha256": split.get("validation_dataset_sha256"),
                "validation_dataset_image_ids_sha256": split.get("validation_dataset_image_ids_sha256"),
                "validation_dataset_ground_truth_sha256": split.get("validation_dataset_ground_truth_sha256"),
                "accuracy_gate_policy_sha256": split.get("accuracy_gate_policy_sha256"),
                "task_quality_policy_sha256": split.get("task_quality_policy_sha256"),
                "runtime_quality_gate_policy_sha256": split.get("runtime_quality_gate_policy_sha256"),
                "prepared_feed_task": split.get("prepared_feed_task"),
                "prepared_feed_preprocess_mode": split.get("prepared_feed_preprocess_mode"),
                "prepared_feed_letterbox_pad_value": split.get("prepared_feed_letterbox_pad_value"),
                "prepared_feed_source_image_sha256": split.get("prepared_feed_source_image_sha256"),
                "energy_scope": split.get("energy_scope"),
                "energy_window_effective": split.get("energy_window_effective"),
                "energy_calibration_sha256": split.get("energy_calibration_sha256"),
                "target_duration_s": (
                    split_target_duration if target_duration_match else None
                ),
                "split_target_duration_s": split_target_duration,
                "baseline_target_duration_s": baseline_target_duration,
                "target_duration_match": target_duration_match,
                "split_active_duration_s": split_active_duration,
                "baseline_active_duration_s": baseline_active_duration,
                "active_duration_absolute_delta_s": active_duration_absolute_delta,
                "active_duration_relative_delta": active_duration_relative_delta,
                "active_duration_relative_tolerance": NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE,
                "active_duration_within_tolerance": active_duration_within_tolerance,
                "active_duration_comparison_policy": NATIVE_ENERGY_ACTIVE_DURATION_COMPARISON_POLICY,
                "active_duration_tolerance_source": NATIVE_ENERGY_ACTIVE_DURATION_TOLERANCE_SOURCE,
                "split_energy_per_work_j": split_energy,
                "baseline_energy_per_work_j": baseline_energy,
                "split_raw_energy_per_work_j": split_comparison.get("raw_energy_per_work_j"),
                "baseline_raw_energy_per_work_j": baseline_comparison.get("raw_energy_per_work_j") if baseline else None,
                "baseline_host_normalized_energy_per_work_est_j": baseline_comparison.get("host_normalized_energy_per_work_est_j") if baseline else None,
                "split_energy_comparison_basis": split_comparison.get("energy_comparison_basis"),
                "baseline_energy_comparison_basis": baseline_comparison.get("energy_comparison_basis") if baseline else None,
                "baseline_energy_comparison_status": baseline_comparison.get("energy_comparison_status") if baseline else "baseline_missing",
                "energy_delta_j": split_energy - baseline_energy if comparable and split_energy is not None and baseline_energy is not None else None,
                "energy_ratio": split_energy / baseline_energy if comparable and split_energy is not None and baseline_energy not in (None, 0) else None,
                "energy_saving_fraction": 1.0 - split_energy / baseline_energy if comparable and split_energy is not None and baseline_energy not in (None, 0) else None,
                "split_average_power_w": split.get("average_power_w"),
                "baseline_average_power_w": baseline.get("average_power_w") if baseline else None,
            })
    return pairs


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def write_native_energy_reports(run_dir: str | Path, report_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    rows = collect_native_energy(run)
    pairs = build_native_energy_pairs(rows)
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    native_evidence = _load(
        run / "reports" / "native_evidence_status.json"
    )
    native_evidence_summary = project_native_evidence_status(
        native_evidence
    )
    _write_csv(out / "screening_energy_observations.csv", rows)
    _write_csv(out / "native_energy_observations.csv", rows)
    _write_csv(out / "native_energy_pair_comparison.csv", pairs)
    (out / "screening_energy_observations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "native_energy_observations.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "native_energy_pair_comparison.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    status_payload = {
        "schema": "onnx-splitpoint/native-energy-report-status",
        "schema_version": 1,
        "row_count": len(rows),
        "pair_count": len(pairs),
        "native_evidence_status": native_evidence,
        "native_evidence_summary": native_evidence_summary,
    }
    status_path = out / "native_energy_report_status.json"
    status_path.write_text(
        json.dumps(status_payload, indent=2), encoding="utf-8",
    )

    def status_text(value: Any) -> str:
        return "unavailable" if value is None else str(value)

    success = status_text(
        native_evidence_summary["energy_measurement_success_count"]
    )
    plan_denominator = status_text(
        native_evidence_summary["energy_plan_denominator_count"]
    )
    matrix_denominator = status_text(
        native_evidence_summary["energy_matrix_denominator_count"]
    )
    measurement_started = status_text(
        native_evidence_summary["energy_measurement_started_count"]
    )
    not_started_preflight = status_text(
        native_evidence_summary["energy_not_started_preflight_count"]
    )
    md = [
        "# Native energy pair comparison",
        "",
        "## Scientific evidence status",
        "",
        (
            "- Scientific ready: "
            f"**{status_text(native_evidence_summary['scientific_ready'])}** "
            f"(status: `{native_evidence_summary['scientific_status']}`)"
        ),
        (
            "- Technical completeness: "
            f"**{status_text(native_evidence_summary['technical_complete'])}** "
            f"(status: `{native_evidence_summary['technical_status']}`)"
        ),
        (
            "- Claim decisions complete: "
            f"**{status_text(native_evidence_summary['claim_decisions_complete'])}**"
        ),
        (
            "- Native energy execution: "
            f"**{energy_axis_description(native_evidence)}**"
        ),
        (
            "- Native energy attempts: "
            f"**{measurement_started} started, "
            f"{not_started_preflight} not started**"
        ),
        f"- Accounting: {native_evidence_summary.get('energy_accounting_summary') or 'unavailable for legacy evidence'}",
        (
            "- Energy plan completion: "
            f"**{success}/{plan_denominator}** "
            "(successful measurements / planned measurements)"
        ),
        (
            "- Energy matrix coverage: "
            f"**{success}/{matrix_denominator}** "
            "(successful measurements / expected matrix rows)"
        ),
        (
            "- Energy plan exclusions: "
            f"**{status_text(native_evidence_summary['energy_plan_excluded_count'])}**"
        ),
        (
            "- Energy claim-eligible measurements: "
            f"**{status_text(native_evidence_summary['energy_claim_eligible_count'])}**"
        ),
        "",
        "## Pair observations",
        "",
        f"Observations: **{len(rows)}**",
        f"Pairs: **{len(pairs)}**",
        "",
    ]
    for pair in pairs:
        md.append(
            f"- {pair.get('model')} / {pair.get('split_backend')} / {pair.get('case')} vs "
            f"{pair.get('baseline_backend')} on {pair.get('setup_id')}: {pair.get('comparison_status')}"
        )
    (out / "native_energy_pair_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {
        "row_count": len(rows),
        "pair_count": len(pairs),
        "rows": rows,
        "pairs": pairs,
        "native_evidence_status": native_evidence,
        "native_evidence_summary": native_evidence_summary,
        "status_json": status_path,
    }


__all__ = [
    "NATIVE_ENERGY_ACTIVE_DURATION_RELATIVE_TOLERANCE",
    "NATIVE_ENERGY_ACTIVE_DURATION_COMPARISON_POLICY",
    "collect_native_energy",
    "scientific_energy_rows",
    "build_native_energy_pairs",
    "write_native_energy_reports",
]
