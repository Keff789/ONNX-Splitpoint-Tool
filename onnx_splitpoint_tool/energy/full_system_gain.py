"""Verified per-board Full-System input gain calibration.

The normal u.RECS method manifest remains responsible for the calibrated
transfer function/non-linearity.  This module adds one setup-bound scalar for
the remaining absolute DC gain error (for example the tolerance of the 20 mOhm
input shunt).  The scalar is used only for FS/FULL_SYSTEM measurements and only
when its immutable calibration evidence is present and SHA-256 verified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


FULL_SYSTEM_CURRENT_SCALE_SCHEMA = (
    "onnx-splitpoint/full-system-input-scale-calibration"
)
FULL_SYSTEM_CURRENT_SCALE_SCHEMA_VERSION = 2
FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION = "9V_20V_IN_after_R16_to_GND"
FULL_SYSTEM_CURRENT_SCALE_MODEL = (
    "least_squares_through_origin_with_adjacent_idle_baselines"
)
FULL_SYSTEM_CURRENT_SCALE_TARGET_CURRENTS_A = (0.5, 1.0)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _setup_value(setup: Any, key: str, default: Any = None) -> Any:
    if isinstance(setup, Mapping):
        return setup.get(key, default)
    return getattr(setup, key, default)


def _canonical_scope(value: Any) -> str:
    return str(value or "").strip().upper()


def _same_number(left: Any, right: Any, *, rel_tol: float = 1e-9) -> bool:
    a = _finite_number(left)
    b = _finite_number(right)
    return bool(
        a is not None
        and b is not None
        and math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-12)
    )


def _recompute_factor(points: list[Mapping[str, Any]]) -> float | None:
    measured: list[float] = []
    reference: list[float] = []
    for point in points:
        delta = _finite_number(point.get("measured_increment_w"))
        ref = _finite_number(point.get("reference_power_w"))
        if delta is None or ref is None or delta <= 0.0 or ref <= 0.0:
            return None
        measured.append(delta)
        reference.append(ref)
    denominator = sum(value * value for value in measured)
    if denominator <= 0.0:
        return None
    return sum(m * r for m, r in zip(measured, reference)) / denominator


def verify_full_system_current_scale_calibration(
    setup: Any,
    *,
    physical_scope: Any = "FS",
) -> dict[str, Any]:
    """Verify the exact evidence backing a configured Full-System scale.

    Absence is a valid identity configuration.  Once any scale field is
    configured, all fields and evidence must verify; consumers can therefore
    fail closed instead of silently applying an unbound number.
    """

    scope = _canonical_scope(physical_scope)
    applicable = scope in {"FS", "FULL_SYSTEM"}
    factor_raw = _setup_value(setup, "full_system_current_scale_factor")
    calibrated_at = str(
        _setup_value(setup, "full_system_current_scale_calibrated_at", "") or ""
    ).strip()
    evidence_raw = str(
        _setup_value(setup, "full_system_current_scale_calibration_evidence", "")
        or ""
    ).strip()
    expected_sha = str(
        _setup_value(setup, "full_system_current_scale_calibration_sha256", "")
        or ""
    ).strip().lower()
    configured = bool(
        factor_raw is not None or calibrated_at or evidence_raw or expected_sha
    )
    result: dict[str, Any] = {
        "full_system_current_scale_configured": configured,
        "full_system_current_scale_applicable": applicable,
        "full_system_current_scale_verified": False,
        "full_system_current_scale_verification_status": (
            "not_configured_identity" if not configured else "verification_failed"
        ),
        "full_system_current_scale_factor_configured": None,
        "full_system_current_scale_calibrated_at": calibrated_at,
        "full_system_current_scale_calibration_evidence": evidence_raw,
        "full_system_current_scale_calibration_sha256_expected": expected_sha,
        "full_system_current_scale_calibration_sha256_actual": "",
        "full_system_current_scale_verification_errors": [],
    }
    if not configured:
        result["full_system_current_scale_verified"] = True
        return result

    errors: list[str] = result["full_system_current_scale_verification_errors"]
    factor = _finite_number(factor_raw)
    if factor is None or not 0.5 <= factor <= 1.5:
        errors.append("factor_missing_or_out_of_range")
    else:
        result["full_system_current_scale_factor_configured"] = factor
    if not calibrated_at:
        errors.append("calibrated_at_missing")
    if not evidence_raw:
        errors.append("evidence_path_missing")
    if not _SHA256_RE.fullmatch(expected_sha):
        errors.append("evidence_sha256_missing_or_invalid")

    evidence_path: Path | None = None
    if evidence_raw:
        candidate = Path(evidence_raw).expanduser()
        if not candidate.is_absolute():
            errors.append("evidence_path_not_absolute")
        else:
            try:
                evidence_path = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                errors.append("evidence_path_missing")
            else:
                # The writer records a resolved absolute path.  Requiring that
                # exact canonical spelling avoids later symlink/``..`` retargeting
                # of an otherwise matching SHA-bound evidence reference.
                if candidate != evidence_path:
                    errors.append("evidence_path_not_canonical")
                if candidate.is_symlink() or not evidence_path.is_file():
                    errors.append("evidence_path_not_regular_file")
    payload: Mapping[str, Any] | None = None
    if evidence_path is not None and _SHA256_RE.fullmatch(expected_sha):
        try:
            actual_sha = sha256_file(evidence_path)
        except OSError:
            errors.append("evidence_sha256_unreadable")
        else:
            result["full_system_current_scale_calibration_sha256_actual"] = actual_sha
            if actual_sha != expected_sha:
                errors.append("evidence_sha256_mismatch")
        try:
            loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append("evidence_json_invalid")
        else:
            if not isinstance(loaded, Mapping):
                errors.append("evidence_root_not_mapping")
            else:
                payload = loaded

    if payload is not None:
        if payload.get("schema") != FULL_SYSTEM_CURRENT_SCALE_SCHEMA:
            errors.append("evidence_schema_mismatch")
        if payload.get("schema_version") != FULL_SYSTEM_CURRENT_SCALE_SCHEMA_VERSION:
            errors.append("evidence_schema_version_mismatch")
        if payload.get("status") != "ok":
            errors.append("evidence_status_not_ok")
        if payload.get("save_requested") is not True:
            errors.append("evidence_not_authorized_for_save")
        if payload.get("measurement_scope") != "full_system_input":
            errors.append("evidence_measurement_scope_mismatch")
        if _canonical_scope(payload.get("physical_scope")) not in {
            "FS",
            "FULL_SYSTEM",
        }:
            errors.append("evidence_scope_mismatch")
        if payload.get("load_connection") != FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION:
            errors.append("evidence_load_connection_mismatch")
        fit = payload.get("fit")
        if not isinstance(fit, Mapping):
            errors.append("evidence_fit_missing")
            fit = {}
        if fit.get("model") != FULL_SYSTEM_CURRENT_SCALE_MODEL:
            errors.append("evidence_fit_model_mismatch")
        quality = payload.get("quality_gate")
        if not isinstance(quality, Mapping) or quality.get("pass") is not True:
            errors.append("evidence_quality_gate_not_passed")
        elif list(quality.get("reasons") or []):
            errors.append("evidence_quality_gate_reasons_not_empty")
        if str(payload.get("finished_at") or "").strip() != calibrated_at:
            errors.append("evidence_calibrated_at_mismatch")
        restoration = payload.get("restoration")
        restoration_verified = False
        if isinstance(restoration, Mapping):
            new_restoration_fields = {
                "initial_state_restored",
                "jetson_initial_state_restored",
                "m2_untouched",
                "initial_jetson_ssh_ready",
                "final_jetson_ssh_ready",
                "initial_m2_present",
                "final_m2_present",
            }
            has_new_restoration_contract = any(
                key in restoration for key in new_restoration_fields
            )
            # Backwards compatibility for calibration evidence written by
            # v2.79.16/17: those workflows always forced the platform back to
            # Jetson ON / M.2 present and recorded only these two booleans.  A
            # partially present new contract must not fall back to this weaker
            # legacy projection.
            legacy_restoration_verified = bool(
                not has_new_restoration_contract
                and restoration.get("jetson_ssh_ready") is True
                and restoration.get("m2_present") is True
            )

            # v2.79.18 restores the state observed at entry and never toggles
            # M.2.  In particular, an explicitly admitted already-off Jetson
            # must finish off; requiring SSH-ready=True would invalidate an
            # otherwise successful calibration.  All new fields are inside
            # the existing SHA-bound evidence, so this is an additive schema
            # v2 contract rather than a new provenance layer.
            initial_jetson = restoration.get("initial_jetson_ssh_ready")
            final_jetson = restoration.get("final_jetson_ssh_ready")
            initial_m2 = restoration.get("initial_m2_present")
            final_m2 = restoration.get("final_m2_present")
            jetson_states_match = bool(
                type(initial_jetson) is bool
                and type(final_jetson) is bool
                and initial_jetson is final_jetson
            )
            m2_states_admissible = bool(
                (initial_m2 is None or type(initial_m2) is bool)
                and (final_m2 is None or type(final_m2) is bool)
                and (
                    initial_m2 is None
                    or final_m2 is None
                    or initial_m2 is final_m2
                )
            )
            initial_state_restoration_verified = bool(
                has_new_restoration_contract
                and new_restoration_fields.issubset(restoration.keys())
                and restoration.get("initial_state_restored") is True
                and restoration.get("jetson_initial_state_restored") is True
                and restoration.get("m2_untouched") is True
                and jetson_states_match
                and m2_states_admissible
                and restoration.get("jetson_ssh_ready") is final_jetson
                and restoration.get("m2_present") is (final_m2 is True)
            )
            restoration_verified = bool(
                legacy_restoration_verified
                or initial_state_restoration_verified
            )
        if not restoration_verified:
            errors.append("evidence_platform_restoration_not_verified")

        expected_setup_id = str(_setup_value(setup, "setup_id", "") or "")
        evidence_setup_id = str(payload.get("setup_id") or "")
        if expected_setup_id and evidence_setup_id != expected_setup_id:
            errors.append("evidence_setup_id_mismatch")
        expected_accelerator = str(
            _setup_value(setup, "accelerator", "") or ""
        ).strip()
        if expected_accelerator and str(payload.get("accelerator") or "").strip() != expected_accelerator:
            errors.append("evidence_accelerator_mismatch")
        expected_address = str(
            _setup_value(setup, "urecs_address", "") or ""
        ).strip()
        binding = payload.get("setup_binding")
        if not isinstance(binding, Mapping):
            errors.append("evidence_setup_binding_missing")
            binding = {}
        if expected_address and str(binding.get("urecs_address") or "").strip() != expected_address:
            errors.append("evidence_urecs_address_mismatch")
        expected_port = _setup_value(setup, "data_port", None)
        if expected_port is not None and binding.get("data_port") != expected_port:
            errors.append("evidence_data_port_mismatch")

        fit_factor = fit.get("scale_factor")
        if factor is not None and not _same_number(fit_factor, factor):
            errors.append("evidence_factor_mismatch")
        points_raw = fit.get("point_results")
        points = (
            [point for point in points_raw if isinstance(point, Mapping)]
            if isinstance(points_raw, list)
            else []
        )
        if len(points) != 2:
            errors.append("evidence_point_count_mismatch")
        else:
            try:
                targets = tuple(
                    sorted(float(point.get("target_current_a")) for point in points)
                )
            except (TypeError, ValueError):
                targets = ()
            if targets != FULL_SYSTEM_CURRENT_SCALE_TARGET_CURRENTS_A:
                errors.append("evidence_target_currents_mismatch")
            root_targets = payload.get("target_currents_a")
            try:
                canonical_root_targets = tuple(
                    sorted(float(value) for value in list(root_targets or []))
                )
            except (TypeError, ValueError):
                canonical_root_targets = ()
            if canonical_root_targets != FULL_SYSTEM_CURRENT_SCALE_TARGET_CURRENTS_A:
                errors.append("evidence_root_target_currents_mismatch")
            for point in points:
                current = _finite_number(point.get("reference_current_a"))
                voltage = _finite_number(point.get("reference_voltage_v"))
                reference = _finite_number(point.get("reference_power_w"))
                if (
                    current is None
                    or voltage is None
                    or reference is None
                    or current <= 0.0
                    or voltage <= 0.0
                    or not math.isclose(
                        current * voltage, reference, rel_tol=1e-9, abs_tol=1e-9
                    )
                ):
                    errors.append("evidence_reference_power_mismatch")
                    break
            recomputed = _recompute_factor(points)
            if recomputed is None or not _same_number(recomputed, fit_factor):
                errors.append("evidence_fit_recomputation_mismatch")

    result["full_system_current_scale_verified"] = not errors
    result["full_system_current_scale_verification_status"] = (
        "verified" if not errors else "verification_failed"
    )
    return result


def apply_verified_full_system_current_scale(
    summary: Mapping[str, Any],
    *,
    physical_scope: Any,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a verified scale to all Full-System power/energy quantities."""

    out = dict(summary)
    scope = _canonical_scope(physical_scope)
    applicable = scope in {"FS", "FULL_SYSTEM"}
    configured = verification.get("full_system_current_scale_configured") is True
    verified = verification.get("full_system_current_scale_verified") is True
    factor = _finite_number(
        verification.get("full_system_current_scale_factor_configured")
    )
    applied = bool(applicable and configured and verified and factor is not None)
    if applied:
        for key in (
            "energy_total_j",
            "avg_power_w",
            "max_frame_energy_j",
            "idle_frame_energy_j",
        ):
            numeric = _finite_number(out.get(key))
            if numeric is None:
                continue
            out[f"full_system_input_unscaled_{key}"] = numeric
            out[key] = numeric * factor
    out.update(dict(verification))
    out.update(
        {
            "full_system_current_scale_factor_applied": factor if applied else None,
            "full_system_current_scale_applied": applied,
            "full_system_current_scale_status": (
                "applied"
                if applied
                else "not_applicable_non_full_system"
                if not applicable
                else "not_configured_identity"
                if not configured
                else "verified_identity"
                if verified and factor is not None and math.isclose(factor, 1.0)
                else "verification_failed"
            ),
        }
    )
    return out
