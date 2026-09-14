from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy.comparison import (
    resolve_energy_comparison,
    verify_accelerator_idle_calibration_binding,
)
from onnx_splitpoint_tool.energy.config import EnergySetup


STARTED_AT = "2026-09-03T12:00:00+00:00"
FINISHED_AT = "2026-09-03T12:02:00+00:00"


def _write_simple_evidence(
    tmp_path: Path,
    **updates: object,
) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = {
        "schema": "onnx-splitpoint/m2-idle-power-calibration",
        "schema_version": 2,
        "setup_id": "orin_nx_hailo8_01",
        "accelerator": "hailo8",
        "urecs_address": "192.0.2.10",
        "data_port": 3000,
        "physical_scope": "FS",
        "started_at": STARTED_AT,
        "finished_at": FINISHED_AT,
        "status": "ok",
        "saved": True,
        "restored_m2_on": True,
        "accelerator_idle_power_w": 2.0,
        "m2_off": {"avg_power_w": 8.0},
        "m2_on": {"avg_power_w": 10.0},
    }
    payload.update(updates)
    path = tmp_path / "m2_idle_power_calibration.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path.resolve(), payload


def _setup(path: Path, *, value_w: float = 2.0) -> EnergySetup:
    return EnergySetup(
        setup_id="orin_nx_hailo8_01",
        accelerator="hailo8",
        urecs_address="192.0.2.10",
        data_port=3000,
        accelerator_idle_w=value_w,
        accelerator_idle_calibrated_at=FINISHED_AT,
        accelerator_idle_calibration_evidence=str(path),
    )


def _simple_verified_trt_row(path: Path) -> dict[str, object]:
    return {
        "host_normalization_role": "tensorrt_full",
        "host_normalization_source_run_id": "native_full_tensorrt",
        "host_normalization_target_variant": "full",
        "host_normalization_identity_verified": True,
        "accelerator_idle_correction_requested": True,
        "accelerator_idle_correction_applied": True,
        "accelerator_idle_correction_statuses": ["applied"],
        "accelerator_idle_w_applied": 2.0,
        "accelerator_idle_calibration_verified": True,
        "accelerator_idle_calibration_status": "verified",
        "accelerator_idle_calibration_evidence": str(path),
        "accelerator_idle_calibrated_at": FINISHED_AT,
        "energy_efficiency_claim_eligible": True,
        "energy_total_j": 100.0,
        "host_normalized_energy_est_j": 80.0,
        "energy_per_work_j": 0.1,
        "host_normalized_energy_per_work_est_j": 0.08,
        "average_power_w": 10.0,
        "host_normalized_average_power_est_w": 8.0,
        "active_duration_s": 10.0,
        "work_units": 1000,
    }


def test_simple_evidence_verifies_without_any_binding_or_sha(tmp_path: Path) -> None:
    path, _payload = _write_simple_evidence(tmp_path)

    result = verify_accelerator_idle_calibration_binding(_setup(path))

    assert result["accelerator_idle_calibration_verified"] is True
    assert result["accelerator_idle_calibration_status"] == "verified"
    assert result["accelerator_idle_calibration_mode"] == "simple_json"
    assert result["accelerator_idle_calibration_evidence"] == str(path)
    assert "accelerator_idle_calibration_binding_sha256" not in result


def test_simple_evidence_accepts_coherent_readable_top_level_aliases(
    tmp_path: Path,
) -> None:
    path, _payload = _write_simple_evidence(
        tmp_path,
        idle_power_without_m2_w=8.0,
        idle_power_with_m2_w=10.0,
    )

    result = verify_accelerator_idle_calibration_binding(_setup(path))

    assert result["accelerator_idle_calibration_verified"] is True


@pytest.mark.parametrize(
    ("updates", "setup_value", "setup_id", "calibrated_at", "reason"),
    (
        (
            {"physical_scope": "MB"},
            2.0,
            "orin_nx_hailo8_01",
            FINISHED_AT,
            "physical_scope_not_full_system",
        ),
        ({"status": "failed"}, 2.0, "orin_nx_hailo8_01", FINISHED_AT, "status_not_ok"),
        ({"saved": False}, 2.0, "orin_nx_hailo8_01", FINISHED_AT, "registry_save_not_confirmed"),
        (
            {"restored_m2_on": False},
            2.0,
            "orin_nx_hailo8_01",
            FINISHED_AT,
            "m2_restore_not_verified",
        ),
        (
            {"accelerator_idle_power_w": 3.0},
            2.0,
            "orin_nx_hailo8_01",
            FINISHED_AT,
            "accelerator_idle_power_delta_mismatch",
        ),
        ({}, 3.0, "orin_nx_hailo8_01", FINISHED_AT, "registry_accelerator_idle_power_mismatch"),
        ({"setup_id": "other"}, 2.0, "orin_nx_hailo8_01", FINISHED_AT, "setup_id_mismatch"),
        ({"accelerator": "hailo10h"}, 2.0, "orin_nx_hailo8_01", FINISHED_AT, "accelerator_mismatch"),
        ({"urecs_address": "192.0.2.99"}, 2.0, "orin_nx_hailo8_01", FINISHED_AT, "urecs_address_mismatch"),
        ({"data_port": 4000}, 2.0, "orin_nx_hailo8_01", FINISHED_AT, "data_port_mismatch"),
        ({}, 2.0, "orin_nx_hailo8_01", STARTED_AT, "registry_calibrated_at_mismatch"),
        (
            {"idle_power_without_m2_w": 9.0},
            2.0,
            "orin_nx_hailo8_01",
            FINISHED_AT,
            "m2_off_top_level_alias_mismatch",
        ),
    ),
)
def test_simple_evidence_rejects_inconsistent_contract(
    tmp_path: Path,
    updates: dict[str, object],
    setup_value: float,
    setup_id: str,
    calibrated_at: str,
    reason: str,
) -> None:
    path, _payload = _write_simple_evidence(tmp_path, **updates)
    setup = _setup(path, value_w=setup_value)
    setup.setup_id = setup_id
    setup.accelerator_idle_calibrated_at = calibrated_at

    result = verify_accelerator_idle_calibration_binding(setup)

    assert result["accelerator_idle_calibration_verified"] is False
    assert result["accelerator_idle_calibration_status"] == (
        "unavailable_simple_evidence_validation_failed"
    )
    assert reason in result["accelerator_idle_calibration_failure_reasons"]


@pytest.mark.parametrize(
    "value",
    (None, True, "2.0", -1.0, 0.0, float("nan"), float("inf")),
)
def test_simple_evidence_rejects_nonpositive_or_nonliteral_registry_value(
    tmp_path: Path, value: object,
) -> None:
    path, _payload = _write_simple_evidence(tmp_path)
    setup = _setup(path)
    setup.accelerator_idle_w = value  # type: ignore[assignment]

    result = verify_accelerator_idle_calibration_binding(setup)

    assert result["accelerator_idle_calibration_verified"] is False
    assert "registry_accelerator_idle_power_invalid" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_simple_evidence_path_must_be_direct_absolute_canonical_json(
    tmp_path: Path,
) -> None:
    path, _payload = _write_simple_evidence(tmp_path)
    setup = _setup(path)
    setup.accelerator_idle_calibration_evidence = path.name

    result = verify_accelerator_idle_calibration_binding(setup)

    assert result["accelerator_idle_calibration_verified"] is False
    assert result["accelerator_idle_calibration_status"] == (
        "unavailable_simple_evidence_path_invalid"
    )


def test_comparison_accepts_simple_calibration_reference_without_sha(
    tmp_path: Path,
) -> None:
    path, _payload = _write_simple_evidence(tmp_path)

    result = resolve_energy_comparison(_simple_verified_trt_row(path))

    assert result["energy_comparison_status"] == "host_normalized_verified"
    assert result["energy_comparison_claim_ready"] is True
    assert result["energy_comparison_basis"] == (
        "host_normalized_accelerator_idle_subtracted"
    )
    assert result["comparison_energy_total_j"] == pytest.approx(80.0)
    assert result["comparison_average_power_w"] == pytest.approx(8.0)


def test_comparison_still_accepts_historical_binding_sha_reference() -> None:
    row = _simple_verified_trt_row(Path("/not/used/by/resolver.json"))
    row.pop("accelerator_idle_calibration_evidence")
    row.pop("accelerator_idle_calibrated_at")
    row["accelerator_idle_calibration_binding_sha256"] = "a" * 64

    result = resolve_energy_comparison(row)

    assert result["energy_comparison_status"] == "host_normalized_verified"
    assert result["energy_comparison_claim_ready"] is True


def test_comparison_rejects_verified_flag_without_simple_or_legacy_reference(
    tmp_path: Path,
) -> None:
    path, _payload = _write_simple_evidence(tmp_path)
    row = _simple_verified_trt_row(path)
    row.pop("accelerator_idle_calibration_evidence")
    row.pop("accelerator_idle_calibrated_at")

    result = resolve_energy_comparison(row)

    assert result["energy_comparison_status"] == (
        "required_tensorrt_full_calibration_unverified"
    )
    assert result["energy_comparison_claim_ready"] is False
