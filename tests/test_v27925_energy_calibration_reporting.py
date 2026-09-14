"""FS scale receipts survive Native import without inventing method evidence."""
from copy import deepcopy
import json

import pytest

from onnx_splitpoint_tool.native_energy_reporting import (
    _calibration_fields, _energy_payload, collect_native_energy,
)


def _aggregate():
    verification = {
        "full_system_current_scale_configured": True,
        "full_system_current_scale_applicable": True,
        "full_system_current_scale_verified": True,
        "full_system_current_scale_verification_status": "verified",
        "full_system_current_scale_factor_configured": 0.9136343897390969,
        "full_system_current_scale_calibration_evidence": "/calibration/fs_scale.json",
        "full_system_current_scale_calibration_sha256_expected": "a" * 64,
        "full_system_current_scale_calibration_sha256_actual": "a" * 64,
        "full_system_current_scale_verification_errors": [],
    }
    return {
        **verification,
        "full_system_current_scale_verification": deepcopy(verification),
        "full_system_current_scale_applied": True,
        "full_system_current_scale_applied_run_count": 3,
        "full_system_current_scale_factor_applied": 0.9136343897390969,
        "full_system_scope_calibration_status": "pass",
        "energy_calibration_verification": {"verified": False, "status": "missing"},
        "energy_physical_scope": "FS",
        "run_count": 3,
        "valid_postprocessed_runs": 3,
        "avg_energy_total_j": 600.0,
        "avg_power_w": 20.0,
        "avg_active_duration_s": 30.0,
        "avg_energy_per_work_unit_j": 0.02,
        "avg_energy_work_units_used": 30000,
        "final_energy_gate_status": "pass",
        "postprocess_status": "ok",
    }


def test_verified_applied_fs_receipt_survives_embedded_import():
    aggregate = _aggregate()
    # A later last-repeat diagnostic must not overwrite authoritative counts.
    stale = {"full_system_current_scale_applied_run_count": 1}
    item = {"run": {"energy_aggregate": aggregate, "stdout_tail": json.dumps(stale)}}
    payload = _energy_payload(item)
    path, digest, verified, status = _calibration_fields(payload, {})
    assert verified is True
    assert path == "/calibration/fs_scale.json"
    assert digest == "a" * 64
    assert status == "verified_full_system_current_scale"


@pytest.mark.parametrize("field,value", [
    ("full_system_current_scale_verified", False),
    ("full_system_current_scale_applicable", False),
    ("full_system_current_scale_verification_status", "failed"),
    ("full_system_scope_calibration_status", "full_system_current_scale_verification_failed"),
    ("full_system_current_scale_calibration_sha256_actual", "b" * 64),
    ("full_system_current_scale_calibration_sha256_expected", "not-a-hash"),
    ("full_system_current_scale_calibration_evidence", ""),
    ("full_system_current_scale_verification_errors", ["setup_mismatch"]),
    ("full_system_current_scale_factor_configured", float("nan")),
    ("full_system_current_scale_applied", False),
    ("full_system_current_scale_factor_applied", 1.0),
    ("full_system_current_scale_applied_run_count", 2),
])
def test_unverified_or_unapplied_scale_is_not_rescued_by_legacy_manifest(field, value):
    aggregate = _aggregate()
    aggregate[field] = value
    aggregate["energy_calibration_verification"] = {"verified": True, "status": "verified"}
    assert _calibration_fields(aggregate, {})[2] is False


def test_conflicting_embedded_verification_receipt_is_rejected():
    aggregate = _aggregate()
    aggregate["full_system_current_scale_verification"]["full_system_current_scale_calibration_sha256_actual"] = "b" * 64
    assert _calibration_fields(aggregate, {})[2] is False


def test_legacy_verified_manifest_and_unconfigured_scale_keep_existing_behavior():
    aggregate = _aggregate()
    aggregate["full_system_current_scale_configured"] = False
    aggregate["energy_calibration_verification"] = {
        "verified": True, "status": "verified", "path": "/legacy.json", "actual_sha256": "b" * 64,
    }
    assert _calibration_fields(aggregate, {}) == ("/legacy.json", "b" * 64, True, "verified")


def test_reporting_keeps_quality_and_optional_external_provenance_separate(tmp_path):
    path = tmp_path / "reports/native_energy_measurements/native_producer_energy_results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rows": [{
        "ok": True,
        "row": {"backend": "hailo8_to_trt", "model": "mobilenet_v3_large", "case": "b027", "energy_scope": "full_system"},
        "run": {"rc": 0, "energy_aggregate": _aggregate()},
    }]}))
    [row] = collect_native_energy(tmp_path)
    assert row["energy_calibration_verified"] is True
    assert row["energy_calibration_status"] == "verified_full_system_current_scale"
    assert row["external_calibration_verified"] is False
    assert row["external_calibration_status"] == "missing"
    assert row["claim_eligible"] is False
    assert "full_system_calibration_not_verified" not in row["claim_exclusion_reasons"]
    assert row["claim_exclusion_reasons"]
    assert row["energy_total_j"] == 600.0
