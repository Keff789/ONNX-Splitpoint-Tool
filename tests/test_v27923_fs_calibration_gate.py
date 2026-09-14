"""Collector regressions without a u.RECS device or remote workload."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy import collector
from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup
from onnx_splitpoint_tool.energy.full_system_gain import (
    FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION,
    FULL_SYSTEM_CURRENT_SCALE_MODEL,
    FULL_SYSTEM_CURRENT_SCALE_SCHEMA,
)


def _setup(tmp_path: Path, *, calibrated: bool = True) -> EnergySetup:
    fields = dict(setup_id="test_fs", enabled=True, urecs_address="192.0.2.10")
    if calibrated:
        payload = {
            "schema": FULL_SYSTEM_CURRENT_SCALE_SCHEMA,
            "schema_version": 2,
            "status": "ok",
            "save_requested": True,
            "setup_id": "test_fs",
            "physical_scope": "FS",
            "measurement_scope": "full_system_input",
            "load_connection": FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION,
            "setup_binding": {"urecs_address": "192.0.2.10", "data_port": 3000},
            "finished_at": "2026-09-05T13:00:00+00:00",
            "target_currents_a": [0.5, 1.0],
            "quality_gate": {"pass": True, "reasons": []},
            "restoration": {"jetson_ssh_ready": True, "m2_present": True},
            "fit": {
                "model": FULL_SYSTEM_CURRENT_SCALE_MODEL,
                "scale_factor": 1.02,
                "point_results": [
                    {"target_current_a": current, "reference_current_a": current,
                     "reference_voltage_v": 19.0, "reference_power_w": current * 19.0,
                     "measured_increment_w": current * 19.0 / 1.02}
                    for current in (0.5, 1.0)
                ],
            },
        }
        evidence = tmp_path / "scale.json"
        evidence.write_text(json.dumps(payload))
        fields.update(
            full_system_current_scale_factor=1.02,
            full_system_current_scale_calibrated_at=payload["finished_at"],
            full_system_current_scale_calibration_evidence=str(evidence.resolve()),
            full_system_current_scale_calibration_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        )
    return EnergySetup(**fields)


def _measure(tmp_path, monkeypatch, setup, *, binding_ok=True, exact_count=True,
             timing_ok=True, legacy_verified=False):
    # Emulate only external acquisition/postprocessing. Calibration verification,
    # scale application, timing parsing, work counters and final gates are real.
    monkeypatch.setattr(collector, "check_energy_tools", lambda _defaults: {
        "collector_found": True, "power_calculations_found": True,
    })

    def capture(_command, **kwargs):
        run = kwargs["stdout_path"].parent
        (run / "collector_storage" / "trace.parquet").write_bytes(b"x" * 4096)
        if timing_ok:
            (run / "workload_timing.txt").write_text(
                "start_ns=1000000000\nend_ns=4000000000\nrc=0\n"
            )
        (run / "workload_stdout.log").write_text(
            "__SPLITPOINT_WORK_UNITS__=100\n"
            "__SPLITPOINT_WORK_UNITS_SOURCE__=completed_frames\n"
            f"__SPLITPOINT_WORK_UNITS_EXACT__={int(exact_count)}\n"
        )
        return {"rc": 0}

    def postprocess(_command, **kwargs):
        run = kwargs["stdout_path"].parent
        (run / "processed" / "results.yaml").write_text(
            "firmware_results:\n  energy: 30.0\n  duration: 3.0\n"
        )
        return {"rc": 0}

    monkeypatch.setattr(collector, "_run_one", capture)
    monkeypatch.setattr(collector, "_run_powercalc_limited", postprocess)
    monkeypatch.setattr(collector, "_prepare_command_window_request_v2", lambda *a, **k: {
        "available": True, "eligible_for_postprocessor": True,
        "request_path": str(tmp_path / "request.json"),
    })
    monkeypatch.setattr(collector, "_command_window_binding", lambda *a, **k: {
        "verified": binding_ok,
        "binding_method": collector._COMMAND_WINDOW_BINDING_METHOD_V2,
        "status": "verified" if binding_ok else "trace_does_not_cover_window",
        "raw_input_energy_verified": binding_ok,
        "calibrated_input_energy_unsubtracted_verified": binding_ok,
    })
    if legacy_verified:
        monkeypatch.setattr(collector, "_verify_calibration_manifest", lambda *a, **k: {
            "verified": True, "status": "verified",
        })
    return collector.run_fast_firmware_measurement(
        "unused-fake-workload", tmp_path / "measurement", setup=setup,
        defaults=EnergyDefaults(pre_duration_s=0, post_duration_s=0),
        duration_s=1.0, run_count=1, exact_run_count=True,
        compare_legacy_window=False, physical_scope="FS", window_label="command",
        require_runtime_work_units=True, require_command_window_alignment=True,
    )


def test_verified_applied_fs_scale_passes_without_old_manifest(tmp_path, monkeypatch):
    result = _measure(tmp_path, monkeypatch, _setup(tmp_path))
    row = result["runs"][0]
    assert result["ok"] is True
    assert result["full_system_scope_calibration_status"] == "pass"
    assert row["full_system_scope_calibration_status"] == "pass"
    assert row["energy_calibration_verification"]["verified"] is False
    assert row["full_system_current_scale_applied"] is True
    assert row["energy_total_j"] == pytest.approx(30.6)
    assert row["energy_per_work_unit_j"] == pytest.approx(0.306)
    assert row["energy_configured_workload_duration_s"] == 1.0
    assert row["workload_duration_s"] == 1.0  # historical configured alias
    assert row["energy_measured_workload_duration_s"] == 3.0
    assert row["energy_measured_workload_duration_source"] == "captured_workload_command"
    assert row["energy_measured_work_units_per_s"] == pytest.approx(100 / 3)
    assert result["avg_energy_configured_workload_duration_s"] == 1.0
    assert result["avg_energy_measured_workload_duration_s"] == 3.0


def test_unconfigured_identity_is_not_a_calibration(tmp_path, monkeypatch):
    result = _measure(tmp_path, monkeypatch, _setup(tmp_path, calibrated=False))
    row = result["runs"][0]
    assert row["full_system_current_scale_verified"] is True
    assert row["full_system_current_scale_configured"] is False
    assert result["ok"] is False
    assert result["full_system_scope_calibration_status"] == "missing"
    assert "full_system_calibration_not_locally_verified" in row["final_energy_gate_reasons"]


def test_verified_legacy_calibration_still_passes(tmp_path, monkeypatch):
    result = _measure(tmp_path, monkeypatch, _setup(tmp_path, calibrated=False), legacy_verified=True)
    assert result["ok"] is True
    assert result["full_system_scope_calibration_status"] == "pass"


@pytest.mark.parametrize("fault", ["setup", "sha"])
def test_wrong_setup_or_sha_blocks_before_capture(tmp_path, monkeypatch, fault):
    from dataclasses import replace
    setup = _setup(tmp_path)
    setup = replace(setup, **(
        {"setup_id": "wrong_setup"} if fault == "setup" else
        {"full_system_current_scale_calibration_sha256": "0" * 64}
    ))
    result = _measure(tmp_path, monkeypatch, setup)
    assert result["status"] == "full_system_current_scale_claim_blocked"
    assert result["collector_started"] is False
    assert not (tmp_path / "measurement" / "run_000").exists()


@pytest.mark.parametrize("fault", ["trace", "counter", "timing"])
def test_fs_calibration_does_not_override_other_gates(tmp_path, monkeypatch, fault):
    result = _measure(
        tmp_path, monkeypatch, _setup(tmp_path), binding_ok=fault != "trace",
        exact_count=fault != "counter", timing_ok=fault != "timing",
    )
    row = result["runs"][0]
    assert row["full_system_scope_calibration_status"] == "pass"
    assert row["final_energy_gate_status"] == "fail"
    assert result["ok"] is False
    assert "energy_per_work_unit_j" not in row
    if fault == "timing":
        assert "energy_measured_workload_duration_s" not in row
        assert "energy_measured_work_units_per_s" not in row


@pytest.mark.parametrize("field", ["full_system_current_scale_configured",
    "full_system_current_scale_applicable", "full_system_current_scale_verified", "applied"])
def test_fs_scale_requires_configuration_applicability_verification_and_application(field):
    scale = dict(full_system_current_scale_configured=True,
                 full_system_current_scale_applicable=True,
                 full_system_current_scale_verified=True)
    scale[field] = False
    assert collector._full_system_scope_calibration_status(
        "FS", {"verified": False, "status": "missing"}, scale,
        scale_applied=field != "applied",
    ) != "pass"
