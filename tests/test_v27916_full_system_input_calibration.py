from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import pytest

from onnx_splitpoint_tool import platform_power as pp
from onnx_splitpoint_tool.energy.collector import (
    HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
    apply_configured_energy_baselines,
    apply_full_system_current_scale,
    run_fast_firmware_measurement,
)
from onnx_splitpoint_tool.energy.config import EnergyDefaults, energy_setup_from_raw
from onnx_splitpoint_tool.energy.full_system_gain import (
    FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION,
    FULL_SYSTEM_CURRENT_SCALE_MODEL,
    FULL_SYSTEM_CURRENT_SCALE_SCHEMA,
    FULL_SYSTEM_CURRENT_SCALE_SCHEMA_VERSION,
    sha256_file,
    verify_full_system_current_scale_calibration,
)


def _status(*, jetson: bool | None, m2: bool | None) -> pp.PlatformStatus:
    return pp.PlatformStatus(
        setup_id="setup",
        checked_at="now",
        urecs_address="192.0.2.10",
        urecs_host="192.0.2.10",
        urecs_port=3000,
        urecs_configured=True,
        urecs_reachable=True,
        urecs_detail="reachable",
        jetson_host="nx@example",
        jetson_configured=True,
        jetson_ssh_ready=jetson,
        jetson_detail="ready" if jetson else "not ready",
        accelerator="hailo8",
        m2_present=m2,
        m2_detail="present" if m2 else "absent",
    )


def _registry() -> dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "energy_defaults": {"physical_scope": "FS", "window_label": "command"},
        "hardware_setups": [
            {
                "id": "setup",
                "accelerator": "hailo8",
                "host": {"address": "example", "user": "nx", "port": 22},
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    "data_port": 3000,
                    "idle_baseline_w": 2.0,
                    "accelerator_idle_w": 1.0,
                    "accelerator_idle_calibrated_at": "old",
                    "accelerator_idle_calibration_evidence": "/tmp/old.json",
                    "accelerator_idle_calibration_binding_path": "/tmp/old-binding.json",
                    "accelerator_idle_calibration_binding_sha256": "1" * 64,
                },
                "power_control": {},
            }
        ],
    }


def _write_valid_scale_evidence(
    tmp_path: Path,
    *,
    factor: float = 1.02,
    setup_id: str = "setup",
    accelerator: str = "hailo8",
    address: str = "192.0.2.10",
    data_port: int = 3000,
    restoration: Mapping[str, Any] | None = None,
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    finished_at = "2026-09-03T08:00:00+00:00"
    points: list[dict[str, Any]] = []
    for target in (0.5, 1.0):
        reference_voltage = 19.0
        reference_power = target * reference_voltage
        measured_increment = reference_power / factor
        points.append(
            {
                "target_current_a": target,
                "reference_current_a": target,
                "reference_voltage_v": reference_voltage,
                "reference_power_w": reference_power,
                "measured_increment_w": measured_increment,
                "point_scale_factor": factor,
                "fit_residual_pct": 0.0,
            }
        )
    payload = {
        "schema": FULL_SYSTEM_CURRENT_SCALE_SCHEMA,
        "schema_version": FULL_SYSTEM_CURRENT_SCALE_SCHEMA_VERSION,
        "status": "ok",
        "save_requested": True,
        "setup_id": setup_id,
        "accelerator": accelerator,
        "physical_scope": "FS",
        "measurement_scope": "full_system_input",
        "load_connection": FULL_SYSTEM_CURRENT_SCALE_LOAD_CONNECTION,
        "setup_binding": {"urecs_address": address, "data_port": data_port},
        "started_at": "2026-09-03T07:55:00+00:00",
        "finished_at": finished_at,
        "target_currents_a": [0.5, 1.0],
        "fit": {
            "model": FULL_SYSTEM_CURRENT_SCALE_MODEL,
            "scale_factor": factor,
            "point_results": points,
            "point_factors": [factor, factor],
            "point_spread_pct": 0.0,
            "max_fit_residual_pct": 0.0,
            "idle_powers_w": [5.0, 5.0, 5.0],
            "idle_drift_w": 0.0,
        },
        "quality_gate": {"pass": True, "reasons": [], "limits": {}},
        "restoration": dict(
            restoration
            or {
                "jetson_ssh_ready": True,
                "m2_present": True,
                "final_status": {},
            }
        ),
    }
    path = (tmp_path / "full_system_input_scale_calibration.json").resolve()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256_file(path)
    setup = {
        "setup_id": setup_id,
        "accelerator": accelerator,
        "urecs_address": address,
        "data_port": data_port,
        "full_system_current_scale_factor": factor,
        "full_system_current_scale_calibrated_at": finished_at,
        "full_system_current_scale_calibration_evidence": str(path),
        "full_system_current_scale_calibration_sha256": digest,
    }
    return path, digest, setup, payload


def test_verified_fs_scale_accepts_initially_off_restoration_and_rejects_drift(
    tmp_path: Path,
) -> None:
    restoration = {
        # Legacy projections remain false for an intentionally off Jetson.
        "jetson_ssh_ready": False,
        "m2_present": False,
        "initial_jetson_ssh_ready": False,
        "final_jetson_ssh_ready": False,
        "initial_m2_present": None,
        "final_m2_present": None,
        "initial_state_restored": True,
        "jetson_initial_state_restored": True,
        "m2_untouched": True,
        "final_status": {},
    }
    path, _digest, setup, payload = _write_valid_scale_evidence(
        tmp_path, restoration=restoration
    )
    verification = verify_full_system_current_scale_calibration(
        setup, physical_scope="FS"
    )
    assert verification["full_system_current_scale_verified"] is True

    payload["restoration"]["final_jetson_ssh_ready"] = True
    # New-contract evidence must never fall back to these legacy projections.
    payload["restoration"]["jetson_ssh_ready"] = True
    payload["restoration"]["m2_present"] = True
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    setup["full_system_current_scale_calibration_sha256"] = sha256_file(path)
    rejected = verify_full_system_current_scale_calibration(
        setup, physical_scope="FS"
    )
    assert rejected["full_system_current_scale_verified"] is False
    assert "evidence_platform_restoration_not_verified" in rejected[
        "full_system_current_scale_verification_errors"
    ]


def test_verified_fs_scale_allows_unobservable_final_m2_without_legacy_fallback(
    tmp_path: Path,
) -> None:
    restoration = {
        "jetson_ssh_ready": True,
        "m2_present": False,
        "initial_jetson_ssh_ready": True,
        "final_jetson_ssh_ready": True,
        "initial_m2_present": True,
        "final_m2_present": None,
        "initial_state_restored": True,
        "jetson_initial_state_restored": True,
        "m2_untouched": True,
        "final_status": {},
    }
    _path, _digest, setup, _payload = _write_valid_scale_evidence(
        tmp_path, restoration=restoration
    )
    verification = verify_full_system_current_scale_calibration(
        setup, physical_scope="FS"
    )
    assert verification["full_system_current_scale_verified"] is True


def test_verified_fs_scale_applies_only_to_full_system_and_detects_tamper(
    tmp_path: Path,
) -> None:
    path, digest, setup, _payload = _write_valid_scale_evidence(tmp_path)
    verification = verify_full_system_current_scale_calibration(
        setup, physical_scope="FS"
    )
    assert verification["full_system_current_scale_verified"] is True
    assert verification["full_system_current_scale_verification_status"] == "verified"

    summary = {
        "energy_total_j": 20.0,
        "active_duration_s": 2.0,
        "avg_power_w": 10.0,
        "max_frame_energy_j": 1.0,
    }
    fs = apply_full_system_current_scale(
        summary,
        physical_scope="FS",
        scale_factor=1.02,
        calibrated_at=setup["full_system_current_scale_calibrated_at"],
        calibration_evidence=path,
        calibration_sha256=digest,
        setup_id="setup",
        accelerator="hailo8",
        urecs_address="192.0.2.10",
        data_port=3000,
    )
    assert fs["energy_total_j"] == pytest.approx(20.4)
    assert fs["avg_power_w"] == pytest.approx(10.2)
    assert fs["max_frame_energy_j"] == pytest.approx(1.02)
    assert fs["full_system_input_unscaled_energy_total_j"] == 20.0
    assert fs["full_system_current_scale_factor_applied"] == 1.02
    assert fs["full_system_current_scale_status"] == "applied"

    mb = apply_full_system_current_scale(
        summary,
        physical_scope="MB",
        scale_factor=1.02,
        calibrated_at=setup["full_system_current_scale_calibrated_at"],
        calibration_evidence=path,
        calibration_sha256=digest,
        setup_id="setup",
        accelerator="hailo8",
        urecs_address="192.0.2.10",
        data_port=3000,
    )
    assert mb["energy_total_j"] == 20.0
    assert mb["full_system_current_scale_applied"] is False
    assert mb["full_system_current_scale_status"] == "not_applicable_non_full_system"

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    rejected = verify_full_system_current_scale_calibration(setup, physical_scope="FS")
    assert rejected["full_system_current_scale_verified"] is False
    assert "evidence_sha256_mismatch" in rejected[
        "full_system_current_scale_verification_errors"
    ]


def test_baselines_remain_in_post_scale_domain(tmp_path: Path) -> None:
    _path, _digest, setup, _payload = _write_valid_scale_evidence(
        tmp_path, factor=1.1
    )
    verification = verify_full_system_current_scale_calibration(
        setup, physical_scope="FS"
    )
    out = apply_configured_energy_baselines(
        {"energy_total_j": 20.0, "active_duration_s": 2.0, "avg_power_w": 10.0},
        idle_baseline_w=2.0,
        accelerator_idle_w=1.0,
        host_normalization_role=HOST_NORMALIZATION_ROLE_TENSORRT_FULL,
        accelerator_idle_calibration={
            "accelerator_idle_calibration_verified": True,
            "accelerator_idle_calibration_status": "verified",
        },
        host_normalization_source_run_id="tensorrt_full",
        host_normalization_target_variant="full",
        physical_scope="FS",
        full_system_current_scale_verification=verification,
    )
    assert out["energy_total_j"] == pytest.approx(22.0)
    # Baselines are freshly measured after the FS scale is saved and are not
    # multiplied a second time: 22 J - 2 W * 2 s.
    assert out["energy_dynamic_j"] == pytest.approx(18.0)
    assert out["accelerator_idle_w_applied"] == pytest.approx(1.0)
    assert out["host_normalized_energy_est_j"] == pytest.approx(20.0)
    assert out["energy_baseline_domain"] == "post_full_system_scale"


def test_energy_setup_requires_complete_evidence_binding() -> None:
    valid = energy_setup_from_raw(
        {
            "id": "setup",
            "host": {"address": "example", "user": "nx", "port": 22},
            "energy": {
                "urecs_address": "192.0.2.10",
                "full_system_current_scale_factor": 1.003,
                "full_system_current_scale_calibrated_at": "time",
                "full_system_current_scale_calibration_evidence": "/tmp/evidence.json",
                "full_system_current_scale_calibration_sha256": "0" * 64,
            },
        }
    )
    assert valid.full_system_current_scale_factor == pytest.approx(1.003)
    assert valid.registry_contract_valid is True

    invalid = energy_setup_from_raw(
        {
            "id": "setup",
            "host": {"address": "example", "user": "nx", "port": 22},
            "energy": {
                "urecs_address": "192.0.2.10",
                "full_system_current_scale_factor": "1.003",
            },
        }
    )
    assert invalid.full_system_current_scale_factor is None
    assert invalid.registry_contract_valid is False
    assert "setup_full_system_current_scale_factor_missing_or_invalid" in (
        invalid.registry_contract_errors
    )
    assert "setup_full_system_current_scale_sha256_missing_or_invalid" in (
        invalid.registry_contract_errors
    )


def test_two_point_fit_uses_adjacent_idle_windows() -> None:
    factor = 1.02
    idle_before = {"avg_power_w": 5.0, "capture_center_monotonic_s": 0.0}
    idle_between = {"avg_power_w": 5.2, "capture_center_monotonic_s": 50.0}
    idle_after = {"avg_power_w": 5.4, "capture_center_monotonic_s": 100.0}
    loaded = [
        {
            "target_current_a": 0.5,
            "reference_current_a": 0.5,
            "reference_voltage_v": 19.0,
            "avg_power_w": 5.1 + 9.5 / factor,
            "capture_center_monotonic_s": 25.0,
        },
        {
            "target_current_a": 1.0,
            "reference_current_a": 1.0,
            "reference_voltage_v": 19.0,
            "avg_power_w": 5.3 + 19.0 / factor,
            "capture_center_monotonic_s": 75.0,
        },
    ]
    analysis = pp._full_system_calibration_analysis(
        idle_before,
        loaded,
        idle_after,
        idle_between=idle_between,
    )
    assert analysis["scale_factor"] == pytest.approx(factor)
    assert analysis["point_spread_pct"] == pytest.approx(0.0)
    assert analysis["max_fit_residual_pct"] == pytest.approx(0.0)
    assert analysis["quality_gate"]["pass"] is True
    points = analysis["point_results"]
    assert points[0]["idle_baseline_power_w"] == pytest.approx(5.1)
    assert points[1]["idle_baseline_power_w"] == pytest.approx(5.3)
    assert all(
        point["idle_baseline_method"] == "mean_of_adjacent_idle_windows"
        for point in points
    )


def test_two_point_fit_uses_actual_currents_and_keeps_nominal_targets() -> None:
    factor = 1.02
    actual_currents = (0.4876, 0.9932)
    idle_before = {"avg_power_w": 5.0, "capture_center_monotonic_s": 0.0}
    idle_between = {"avg_power_w": 5.2, "capture_center_monotonic_s": 50.0}
    idle_after = {"avg_power_w": 5.4, "capture_center_monotonic_s": 100.0}
    loaded = [
        {
            "target_current_a": 0.5,
            "reference_current_a": actual_currents[0],
            "reference_voltage_v": 19.0,
            "avg_power_w": 5.1 + actual_currents[0] * 19.0 / factor,
            "capture_center_monotonic_s": 25.0,
        },
        {
            "target_current_a": 1.0,
            "reference_current_a": actual_currents[1],
            "reference_voltage_v": 19.0,
            "avg_power_w": 5.3 + actual_currents[1] * 19.0 / factor,
            "capture_center_monotonic_s": 75.0,
        },
    ]

    analysis = pp._full_system_calibration_analysis(
        idle_before,
        loaded,
        idle_after,
        idle_between=idle_between,
    )

    assert analysis["scale_factor"] == pytest.approx(factor)
    assert analysis["quality_gate"]["pass"] is True
    assert [point["target_current_a"] for point in analysis["point_results"]] == [
        0.5,
        1.0,
    ]
    assert [
        point["reference_current_a"] for point in analysis["point_results"]
    ] == pytest.approx(actual_currents)
    assert [
        point["reference_power_w"] for point in analysis["point_results"]
    ] == pytest.approx(
        [actual_currents[0] * 19.0, actual_currents[1] * 19.0]
    )


def test_registry_merge_binds_hash_and_invalidates_old_idle_values() -> None:
    registry = _registry()
    invalidated = pp._merge_full_system_current_scale_fields(
        registry,
        "setup",
        1.0123,
        calibration_record={
            "finished_at": "2026-09-03T08:00:00+00:00",
            "evidence_path": "/tmp/full-system.json",
            "evidence_sha256": "a" * 64,
        },
    )
    assert invalidated is True
    energy = registry["hardware_setups"][0]["energy"]
    assert energy["full_system_current_scale_factor"] == pytest.approx(1.0123)
    assert energy["full_system_current_scale_calibration_sha256"] == "a" * 64
    assert energy["idle_baseline_w"] is None
    assert energy["accelerator_idle_w"] is None
    assert energy["accelerator_idle_calibration_binding_path"] == ""


def test_guided_backend_runs_middle_idle_restores_then_saves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = _registry()
    setup = registry["hardware_setups"][0]
    cfg = {
        "energy": setup["energy"],
        "power_control": dict(pp.DEFAULT_POWER_CONTROL),
    }
    order: list[str] = []
    state = {"jetson": True, "m2": True}

    source_registry = registry
    monkeypatch.setattr(
        pp,
        "resolve_setup",
        lambda setup_id, registry_path=None, registry=None: (
            source_registry if registry is None else registry,
            setup,
            cfg,
        ),
    )
    monkeypatch.setattr(pp, "load_hardware_registry", lambda _path=None: registry)
    udp = {
        "address": "192.0.2.10",
        "port": 3000,
        "udp_terminator": "lf",
        "jetson_command": "jetson",
        "m2_command": "m.2",
        "resolved_endpoints": (("AF_INET", "192.0.2.10"),),
    }
    monkeypatch.setattr(pp, "_validate_platform_udp_configuration", lambda *_a, **_k: udp)
    monkeypatch.setattr(pp, "_admit_pinned_udp_preflight", lambda _s, _c, value: value)

    @contextmanager
    def no_lock(*_args, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", no_lock)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: _status(jetson=state["jetson"], m2=state["m2"]),
    )

    def set_m2(_setup_id: str, desired: bool, **_kwargs: Any) -> dict[str, Any]:
        state["m2"] = bool(desired)
        state["jetson"] = True
        return {"ok": True, "changed": True}

    def set_jetson(_setup_id: str, desired: bool, **_kwargs: Any) -> dict[str, Any]:
        state["jetson"] = bool(desired)
        return {"ok": True, "changed": True}

    monkeypatch.setattr(pp, "_set_m2_state_locked", set_m2)
    monkeypatch.setattr(pp, "_set_jetson_state_locked", set_jetson)

    def restore(*_args: Any, **_kwargs: Any) -> tuple[pp.PlatformStatus, list[dict[str, Any]]]:
        order.append("restore")
        state.update(jetson=True, m2=True)
        return _status(jetson=True, m2=True), []

    monkeypatch.setattr(pp, "_restore_full_system_calibration_start_state", restore)
    target_factor = 1.02
    actual_currents = {
        "load_0.5A": 0.4978,
        "load_1A": 0.9963,
    }
    powers = {
        "idle_before": 5.0,
        "load_0.5A": (
            5.0 + actual_currents["load_0.5A"] * 19.0 / target_factor
        ),
        "idle_between": 5.0,
        "load_1A": (
            5.0 + actual_currents["load_1A"] * 19.0 / target_factor
        ),
        "idle_after": 5.0,
    }

    def measure(*_args: Any, state_label: str, **_kwargs: Any) -> tuple[float, dict[str, Any]]:
        value = powers[state_label]
        return value, {"ok": True, "avg_power_w": value}

    monkeypatch.setattr(pp, "measure_idle_power", measure)

    def save(_setup_id: str, factor: float, **kwargs: Any) -> bool:
        order.append("save")
        assert factor == pytest.approx(target_factor)
        record = dict(kwargs["calibration_record"])
        evidence_path = Path(record["evidence_path"])
        assert evidence_path.is_file()
        assert sha256_file(evidence_path) == record["evidence_sha256"]
        return True

    monkeypatch.setattr(pp, "_save_full_system_current_scale", save)
    steps: list[str] = []

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        steps.append(str(request.get("step_id") or ""))
        if request["kind"] == "load_step":
            return {
                "confirmed": True,
                "current_a": actual_currents[str(request["step_id"])],
                "voltage_v": 19.0,
            }
        if request["kind"] == "review":
            assert request["quality_passed"] is True
            return {"save": True}
        return {"confirmed": True}

    result = pp.calibrate_full_system_input_scale(
        "setup",
        output_dir=tmp_path / "calibration",
        stabilize_s=0.0,
        measure_s=5.0,
        load_settle_s=0.0,
        operator_prompt=prompt,
    )
    assert result.scale_factor == pytest.approx(target_factor)
    assert result.saved is True
    assert result.quality_passed is True
    assert result.restored_jetson_ready is True
    assert result.restored_m2_on is True
    assert result.invalidated_idle_baselines is True
    assert order == ["restore", "save"]
    assert steps == [
        "preflight",
        "idle_before",
        "load_0.5A",
        "idle_between",
        "load_1A",
        "idle_after",
        "review",
    ]
    evidence_path = Path(result.evidence_path)
    assert evidence_path.is_file()
    assert sha256_file(evidence_path) == result.evidence_sha256
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["fit"]["model"] == FULL_SYSTEM_CURRENT_SCALE_MODEL
    assert evidence["quality_gate"]["pass"] is True
    assert "idle_between" in evidence["captures"]
    assert evidence["target_currents_a"] == [0.5, 1.0]
    assert evidence["captures"]["load_0.5A"]["target_current_a"] == 0.5
    assert evidence["captures"]["load_1A"]["target_current_a"] == 1.0
    assert evidence["captures"]["load_0.5A"][
        "reference_current_a"
    ] == pytest.approx(0.4978)
    assert evidence["captures"]["load_1A"][
        "reference_current_a"
    ] == pytest.approx(0.9963)
    assert [
        point["reference_current_a"]
        for point in evidence["fit"]["point_results"]
    ] == pytest.approx([0.4978, 0.9963])
    assert [
        point["reference_power_w"]
        for point in evidence["fit"]["point_results"]
    ] == pytest.approx([0.4978 * 19.0, 0.9963 * 19.0])


def test_collection_blocks_tampered_configured_fs_scale_before_start(
    tmp_path: Path,
) -> None:
    path, digest, _setup, _payload = _write_valid_scale_evidence(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "tamper\n", encoding="utf-8")
    setup = energy_setup_from_raw(
        {
            "id": "setup",
            "accelerator": "hailo8",
            "host": {"address": "example", "user": "nx", "port": 22},
            "energy": {
                "enabled": True,
                "urecs_address": "192.0.2.10",
                "data_port": 3000,
                "full_system_current_scale_factor": 1.02,
                "full_system_current_scale_calibrated_at": "2026-09-03T08:00:00+00:00",
                "full_system_current_scale_calibration_evidence": str(path),
                "full_system_current_scale_calibration_sha256": digest,
            },
        }
    )
    result = run_fast_firmware_measurement(
        "true",
        tmp_path / "blocked-run",
        setup=setup,
        defaults=EnergyDefaults(),
        physical_scope="FS",
        run_count=1,
    )
    assert result["ok"] is False
    assert result["status"] == "full_system_current_scale_claim_blocked"
    assert result["collector_started"] is False
    errors = result["full_system_current_scale_verification"][
        "full_system_current_scale_verification_errors"
    ]
    assert "evidence_sha256_mismatch" in errors


def test_gui_contains_guided_per_setup_calibration_and_quality_gate() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "onnx_splitpoint_tool/gui/panels/panel_hardware.py"
    ).read_text(encoding="utf-8")
    assert "class _FullSystemInputCalibrationDialog" in source
    assert 'text="Calibrate full-system input"' in source
    assert "middle-idle" in source
    assert "quality_passed" in source
    assert "Save blocked: measurement technically invalid" in source
    assert "evidence_sha256" in source
    assert "9V_20V_IN" in source
    assert 'text="Sollpunkt [A]"' in source
    assert 'text="Tatsächlich angezeigter Strom [A]"' in source
    assert "target_current_var" in source
    assert "default_current_a') or 0.0):.4f" in source
    assert "simpledialog" not in source


def test_guided_prompt_requires_exact_confirmation() -> None:
    with pytest.raises(pp.PlatformCalibrationCancelled):
        pp._prompt_operator(
            lambda _request: {},
            {"step_id": "mandatory", "kind": "confirm"},
        )
    with pytest.raises(
        pp.PlatformCalibrationCancelled,
        match="load_0.5A",
    ):
        pp._prompt_operator(
            lambda _request: {"cancelled": True},
            {"step_id": "load_0.5A", "kind": "load_step"},
        )
    assert pp._prompt_operator(
        lambda _request: {"confirmed": True},
        {"step_id": "mandatory", "kind": "confirm"},
    )["confirmed"] is True


def test_load_step_cancel_after_platform_mutation_restores_and_records_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = _registry()
    setup = registry["hardware_setups"][0]
    cfg = {
        "energy": setup["energy"],
        "power_control": dict(pp.DEFAULT_POWER_CONTROL),
    }
    state = {"jetson": True, "m2": True}
    transitions: list[str] = []
    restored: list[bool] = []
    source_registry = registry

    monkeypatch.setattr(
        pp,
        "resolve_setup",
        lambda setup_id, registry_path=None, registry=None: (
            source_registry if registry is None else registry,
            setup,
            cfg,
        ),
    )
    monkeypatch.setattr(pp, "load_hardware_registry", lambda _path=None: registry)
    udp = {
        "address": "192.0.2.10",
        "port": 3000,
        "udp_terminator": "lf",
        "jetson_command": "jetson",
        "m2_command": "m.2",
        "resolved_endpoints": (("AF_INET", "192.0.2.10"),),
    }
    monkeypatch.setattr(
        pp, "_validate_platform_udp_configuration", lambda *_a, **_k: udp
    )
    monkeypatch.setattr(
        pp, "_admit_pinned_udp_preflight", lambda _s, _c, value: value
    )

    @contextmanager
    def no_lock(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", no_lock)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: _status(
            jetson=state["jetson"], m2=state["m2"]
        ),
    )

    def set_m2(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("full-system input calibration must not toggle M.2")

    def set_jetson(
        _setup_id: str, desired: bool, **_kwargs: Any
    ) -> dict[str, Any]:
        transitions.append("jetson_on" if desired else "jetson_off")
        state["jetson"] = bool(desired)
        return {"ok": True, "changed": True}

    monkeypatch.setattr(pp, "_set_m2_state_locked", set_m2)
    monkeypatch.setattr(pp, "_set_jetson_state_locked", set_jetson)

    def restore(
        *_args: Any, **_kwargs: Any
    ) -> tuple[pp.PlatformStatus, list[dict[str, Any]]]:
        restored.append(True)
        state.update(jetson=True, m2=True)
        return _status(jetson=True, m2=True), [{"transition": "restored"}]

    monkeypatch.setattr(
        pp, "_restore_full_system_calibration_start_state", restore
    )

    def measure(
        *_args: Any, state_label: str, **_kwargs: Any
    ) -> tuple[float, dict[str, Any]]:
        assert state_label == "idle_before"
        return 5.0, {"ok": True, "avg_power_w": 5.0}

    monkeypatch.setattr(pp, "measure_idle_power", measure)
    monkeypatch.setattr(
        pp,
        "_save_full_system_current_scale",
        lambda *_a, **_k: pytest.fail("cancelled calibration must not save"),
    )
    steps: list[str] = []

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        step_id = str(request.get("step_id") or "")
        steps.append(step_id)
        if step_id == "load_0.5A":
            return {"cancelled": True}
        return {"confirmed": True}

    output_dir = tmp_path / "cancel-after-mutation"
    with pytest.raises(
        pp.PlatformCalibrationCancelled, match="load_0.5A"
    ) as exc_info:
        pp.calibrate_full_system_input_scale(
            "setup",
            output_dir=output_dir,
            stabilize_s=0.0,
            measure_s=5.0,
            load_settle_s=0.0,
            operator_prompt=prompt,
        )

    assert transitions == ["jetson_off"]
    assert steps == [
        "preflight", "idle_before", "load_0.5A", "recovery_zero_load",
    ]
    assert restored == [True]
    assert state == {"jetson": True, "m2": True}
    operational_path = (
        output_dir / "full_system_input_scale_calibration_operational.json"
    )
    assert Path(exc_info.value.evidence_path) == operational_path
    operational = json.loads(operational_path.read_text(encoding="utf-8"))
    assert operational["status"] == "cancelled"
    assert operational["error_type"] == "PlatformCalibrationCancelled"
    assert list(operational["captures"]) == ["idle_before"]
    assert [event["transition"] for event in operational["events"]] == [
        "jetson_off",
    ]
    assert operational["recovery"]["events"] == [{"transition": "restored"}]
    assert operational["restored_jetson_ready"] is True
    assert operational["restored_m2_on"] is True
    assert operational["initial_state_restored"] is True
    assert operational["m2_untouched"] is True
    assert "recovery_skipped_reason" not in operational
    assert not (
        output_dir / "full_system_input_scale_calibration.json"
    ).exists()


def test_ambiguous_first_toggle_failure_still_enters_recovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_registry = _registry()
    setup = source_registry["hardware_setups"][0]
    cfg = {
        "energy": setup["energy"],
        "power_control": dict(pp.DEFAULT_POWER_CONTROL),
    }
    restored: list[bool] = []

    monkeypatch.setattr(
        pp,
        "resolve_setup",
        lambda setup_id, registry_path=None, registry=None: (
            source_registry if registry is None else registry,
            setup,
            cfg,
        ),
    )
    monkeypatch.setattr(
        pp, "load_hardware_registry", lambda _path=None: source_registry
    )
    udp = {
        "address": "192.0.2.10",
        "port": 3000,
        "udp_terminator": "lf",
        "jetson_command": "jetson",
        "m2_command": "m.2",
        "resolved_endpoints": (("AF_INET", "192.0.2.10"),),
    }
    monkeypatch.setattr(
        pp, "_validate_platform_udp_configuration", lambda *_a, **_k: udp
    )
    monkeypatch.setattr(
        pp, "_admit_pinned_udp_preflight", lambda _s, _c, value: value
    )

    @contextmanager
    def no_lock(*_args: Any, **_kwargs: Any):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", no_lock)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: _status(jetson=True, m2=True),
    )

    def ambiguous_toggle(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise pp.PlatformStateError("post-command verification failed")

    monkeypatch.setattr(pp, "_set_jetson_state_locked", ambiguous_toggle)
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_a, **_k: pytest.fail(
            "full-system input calibration must not toggle M.2"
        ),
    )

    def restore(*_args: Any, **_kwargs: Any) -> tuple[pp.PlatformStatus, list[dict[str, Any]]]:
        restored.append(True)
        return _status(jetson=True, m2=True), []

    monkeypatch.setattr(
        pp, "_restore_full_system_calibration_start_state", restore
    )

    def prompt(request: Mapping[str, Any]) -> dict[str, Any]:
        return {"confirmed": True}

    with pytest.raises(pp.PlatformStateError, match="post-command verification failed"):
        pp.calibrate_full_system_input_scale(
            "setup",
            output_dir=tmp_path / "ambiguous-toggle",
            stabilize_s=0.0,
            measure_s=5.0,
            load_settle_s=0.0,
            operator_prompt=prompt,
        )
    assert restored == [True]
    operational = json.loads(
        (
            tmp_path
            / "ambiguous-toggle"
            / "full_system_input_scale_calibration_operational.json"
        ).read_text(encoding="utf-8")
    )
    assert operational["status"] == "failed"
    assert "recovery_skipped_reason" not in operational
    assert operational["restored_jetson_ready"] is True
    assert operational["restored_m2_on"] is True
    assert operational["initial_state_restored"] is True
    assert operational["m2_untouched"] is True
