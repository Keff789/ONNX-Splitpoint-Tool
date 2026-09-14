from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool import platform_power as pp
from onnx_splitpoint_tool.energy import method_manifest
from onnx_splitpoint_tool.energy.comparison import (
    verify_accelerator_idle_calibration_binding,
)
from onnx_splitpoint_tool.energy.config import (
    energy_setup_from_registry,
    load_hardware_registry,
)
from onnx_splitpoint_tool.gui.panels.panel_hardware import (
    _fs_energy_method_badge_state,
    _run_gui_m2_idle_calibration,
)


SETUP_ID = "orin_nx_hailo8_01"


def _legacy_registry(path: Path, *, idle_w: float = 0.75) -> Path:
    """Write the exact kind of pre-v2.79.13 registry seen on Smartmirror2."""

    path.write_text(
        yaml.safe_dump(
            {
                "schema": "onnx-splitpoint/hardware-setups",
                "schema_version": 2,
                # Deliberately no energy_defaults.physical_scope or
                # energy_defaults.window_label.  M.2 idle calibration must not
                # depend on that energy-claim configuration.
                "hardware_setups": [
                    {
                        "id": SETUP_ID,
                        "accelerator": "hailo8",
                        "host": {
                            "address": "192.0.2.20",
                            "user": "nx",
                            "port": 22,
                        },
                        "energy": {
                            "enabled": True,
                            "urecs_address": "192.0.2.10",
                            "accelerator_idle_w": idle_w,
                            "accelerator_idle_calibration_binding_path": (
                                "/old/sealed-binding.json"
                            ),
                            "accelerator_idle_calibration_binding_sha256": (
                                "a" * 64
                            ),
                        },
                        "power_control": {
                            "enabled": True,
                            "m2_command": "m.2",
                        },
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _status(*, m2_present: bool, ssh_ready: bool = True) -> pp.PlatformStatus:
    return pp.PlatformStatus(
        setup_id=SETUP_ID,
        checked_at="2026-09-03T06:00:00+00:00",
        urecs_address="192.0.2.10",
        urecs_host="192.0.2.10",
        urecs_port=3000,
        urecs_configured=True,
        urecs_reachable=True,
        urecs_detail="reachable",
        jetson_host="nx@192.0.2.20:22",
        jetson_configured=True,
        jetson_ssh_ready=ssh_ready,
        jetson_detail="ready" if ssh_ready else "unreachable",
        accelerator="hailo8",
        m2_present=m2_present,
        m2_detail="detected" if m2_present else "not detected",
    )


@contextmanager
def _unlocked(*_args, **_kwargs):
    yield


def _pin_test_udp_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pp,
        "_resolve_udp_endpoints",
        lambda host, port: [
            (
                pp.socket.AF_INET,
                pp.socket.SOCK_DGRAM,
                17,
                "",
                (host, port),
            )
        ],
    )


def _forbid_energy_method_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("M.2 idle calibration touched the energy-method path")

    # Some revisions retain the old helper for compatibility and some remove
    # it completely.  In both cases, installing this tripwire proves that the
    # active calibration path does not invoke it.
    monkeypatch.setattr(
        pp, "_resolve_verified_energy_method", forbidden, raising=False
    )
    monkeypatch.setattr(
        method_manifest,
        "verify_configured_energy_method",
        forbidden,
    )
    monkeypatch.setattr(
        method_manifest,
        "prepare_configured_energy_method",
        forbidden,
    )


def test_gui_helper_is_direct_and_has_no_name_or_prepare_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    callback = calls.append
    registry_path = tmp_path / "hardware_setups.yaml"

    _forbid_energy_method_path(monkeypatch)
    monkeypatch.setattr(
        pp,
        "calibrate_m2_accelerator_idle_power",
        lambda setup_id, **kwargs: calls.append((setup_id, kwargs))
        or "calibrated",
    )

    result = _run_gui_m2_idle_calibration(
        SETUP_ID,
        registry_path=registry_path,
        callback=callback,
    )

    assert result == "calibrated"
    assert calls == [
        (
            SETUP_ID,
            {"registry_path": registry_path, "callback": callback},
        )
    ]
    assert set(inspect.signature(_run_gui_m2_idle_calibration).parameters) == {
        "setup_id",
        "registry_path",
        "callback",
    }


def test_gui_badge_reports_fixed_full_system_scope_without_verification() -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("badge attempted an energy-method verification")

    badge = _fs_energy_method_badge_state(
        SETUP_ID,
        {
            "calibration_manifest": "",
            "calibration_sha256": "",
        },
        registry_path="/does/not/matter.yaml",
        verifier=forbidden,
    )

    assert badge["status"] == "full_system"
    assert badge["level"] == "ok"
    assert badge["text"] == "Energy measurement: full system"
    assert "complete system" in badge["detail"]
    assert badge["path"] == ""


def test_measure_idle_power_uses_direct_full_system_diagnostic_capture(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    registry = {
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "accelerator": "hailo8",
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    # Stale provenance must not leak into the isolated capture.
                    "calibration_manifest": "/old/method.json",
                    "calibration_sha256": "b" * 64,
                },
            }
        ]
    }

    def runner(_command, output_dir, **kwargs):
        captured["output_dir"] = Path(output_dir)
        captured.update(kwargs)
        return {"ok": True, "status": "ok", "avg_power_w": 12.5}

    power, result = pp.measure_idle_power(
        SETUP_ID,
        registry=registry,
        setup={"id": SETUP_ID, "accelerator": "hailo8"},
        cfg={"energy": {"enabled": True, "urecs_address": "192.0.2.10"}},
        state_label="m2_off",
        duration_s=5,
        output_dir=tmp_path / "m2_off",
        measurement_runner=runner,
    )

    assert power == pytest.approx(12.5)
    assert result["avg_power_w"] == pytest.approx(12.5)
    assert captured["physical_scope"] == "FS"
    assert captured["window_label"] == "command"
    runtime_defaults = captured["defaults"]
    assert runtime_defaults.physical_scope == "FS"
    assert runtime_defaults.window_label == "command"
    assert captured["diagnostic_only"] is True
    assert captured["claim_exclusion_reason"] == (
        "m2_accelerator_idle_power_calibration"
    )
    assert captured["run_count"] == 1
    assert captured["exact_run_count"] is True
    assert "calibration_manifest" not in captured
    assert "calibration_sha256" not in captured
    runtime_setup = captured["setup"]
    assert runtime_setup.calibration_manifest == ""
    assert runtime_setup.calibration_sha256 == ""
    assert runtime_setup.expected_channel_bindings == ()
    assert runtime_setup.expected_channel_bindings_valid is False


def test_legacy_registry_runs_off_on_and_saves_one_simple_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _legacy_registry(tmp_path / "hardware_setups.yaml")
    raw_before = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert "energy_defaults" not in raw_before
    assert "calibration_manifest" not in raw_before["hardware_setups"][0][
        "energy"
    ]
    assert "calibration_sha256" not in raw_before["hardware_setups"][0][
        "energy"
    ]

    _pin_test_udp_endpoint(monkeypatch)
    _forbid_energy_method_path(monkeypatch)
    monkeypatch.setattr(pp, "platform_operation_lock", _unlocked)
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)

    current = {"m2": True}
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(m2_present=current["m2"]),
    )
    transitions: list[bool] = []

    def transition(_setup_id, desired, **_kwargs):
        transitions.append(bool(desired))
        current["m2"] = bool(desired)
        return {
            "ok": True,
            "desired_accelerator_present": bool(desired),
            "after": _status(m2_present=bool(desired)).to_dict(),
        }

    monkeypatch.setattr(pp, "_set_m2_state_locked", transition)
    captures: list[dict[str, object]] = []
    powers = {"m2_off": 11.2, "m2_on": 12.8}

    def measurement_runner(_command, output_dir, **kwargs):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state = str(kwargs["run_id"]).removeprefix("m2_idle_calibration_")
        raw_path = output / "raw_measurement.csv"
        raw_path.write_text("time_s,power_w\n0,12.0\n", encoding="utf-8")
        captures.append(dict(kwargs))
        return {
            "ok": True,
            "status": "ok",
            "avg_power_w": powers[state],
            "raw_measurement": str(raw_path),
            "defaults": kwargs["defaults"].to_dict(),
        }

    output_root = tmp_path / "calibration"
    result = pp.calibrate_m2_accelerator_idle_power(
        SETUP_ID,
        registry_path=registry_path,
        stabilize_s=0,
        measure_s=5,
        output_dir=output_root,
        measurement_runner=measurement_runner,
    )

    assert transitions == [False, True]
    assert [row["run_id"] for row in captures] == [
        "m2_idle_calibration_m2_off",
        "m2_idle_calibration_m2_on",
    ]
    assert all(row["physical_scope"] == "FS" for row in captures)
    assert all(row["diagnostic_only"] is True for row in captures)
    assert result.saved is True
    assert result.restored_m2_on is True
    assert result.idle_power_without_m2_w == pytest.approx(11.2)
    assert result.idle_power_with_m2_w == pytest.approx(12.8)
    assert result.accelerator_idle_power_w == pytest.approx(1.6)

    evidence_path = output_root / "m2_idle_power_calibration.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == 2
    assert evidence["status"] == "ok"
    assert evidence["saved"] is True
    assert evidence["restored_m2_on"] is True
    assert evidence["physical_scope"] == "FS"
    assert evidence["measurement_scope"] == "full_system"
    assert evidence["configuration_projection"]["energy_defaults"][
        "physical_scope"
    ] == "FS"
    assert evidence["m2_off"]["measurement"]["defaults"][
        "physical_scope"
    ] == "FS"
    assert evidence["m2_on"]["measurement"]["defaults"][
        "physical_scope"
    ] == "FS"
    assert evidence["idle_power_without_m2_w"] == pytest.approx(11.2)
    assert evidence["idle_power_with_m2_w"] == pytest.approx(12.8)
    assert evidence["accelerator_idle_power_w"] == pytest.approx(1.6)
    assert evidence["m2_off"]["pre_measurement_status"]["m2_present"] is False
    assert evidence["m2_off"]["post_measurement_status"]["m2_present"] is False
    assert evidence["m2_on"]["pre_measurement_status"]["m2_present"] is True
    assert evidence["m2_on"]["post_measurement_status"]["m2_present"] is True
    assert not list(output_root.glob("*binding*.json"))
    assert not list(output_root.glob("*sealed*.json"))

    stored = load_hardware_registry(registry_path)
    setup = next(
        row for row in stored["hardware_setups"] if row["id"] == SETUP_ID
    )
    energy = setup["energy"]
    assert energy["accelerator_idle_w"] == pytest.approx(1.6)
    assert energy["accelerator_idle_calibrated_at"] == evidence["finished_at"]
    assert energy["accelerator_idle_calibration_evidence"] == str(
        evidence_path.resolve()
    )
    assert "accelerator_idle_calibration_binding_path" not in energy
    assert "accelerator_idle_calibration_binding_sha256" not in energy
    downstream = verify_accelerator_idle_calibration_binding(
        energy_setup_from_registry(stored, SETUP_ID)
    )
    assert downstream["accelerator_idle_calibration_verified"] is True
    assert downstream["accelerator_idle_calibration_mode"] == "simple_json"


def test_capture_failure_recovers_m2_and_does_not_save_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _legacy_registry(tmp_path / "hardware_setups.yaml")
    _pin_test_udp_endpoint(monkeypatch)
    _forbid_energy_method_path(monkeypatch)
    monkeypatch.setattr(pp, "platform_operation_lock", _unlocked)
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)

    current = {"m2": True}
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(m2_present=current["m2"]),
    )
    transitions: list[bool] = []

    def transition(_setup_id, desired, **_kwargs):
        transitions.append(bool(desired))
        current["m2"] = bool(desired)
        return {"ok": True, "desired_accelerator_present": bool(desired)}

    monkeypatch.setattr(pp, "_set_m2_state_locked", transition)

    def failed_capture(*_args, **_kwargs):
        raise RuntimeError("simulated u.RECS capture failure")

    output_root = tmp_path / "failed-calibration"
    with pytest.raises(RuntimeError, match="simulated u.RECS capture failure"):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            stabilize_s=0,
            measure_s=5,
            output_dir=output_root,
            measurement_runner=failed_capture,
        )

    # First transition is the requested off state; the second is bounded
    # recovery after a fresh probe observed the accelerator still absent.
    assert transitions == [False, True]
    assert current["m2"] is True
    evidence = json.loads(
        (output_root / "m2_idle_power_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "failed"
    assert evidence["saved"] is False
    assert evidence["restored_m2_on"] is True
    assert evidence["recovery"]["desired_accelerator_present"] is True

    # No calibration-owned field changes are committed after a failed capture.
    stored_raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    energy = stored_raw["hardware_setups"][0]["energy"]
    assert energy["accelerator_idle_w"] == pytest.approx(0.75)
    assert energy["accelerator_idle_calibration_binding_path"] == (
        "/old/sealed-binding.json"
    )
    assert energy["accelerator_idle_calibration_binding_sha256"] == "a" * 64
    assert "accelerator_idle_calibrated_at" not in energy
    assert "accelerator_idle_calibration_evidence" not in energy


def test_failed_initial_state_does_not_toggle_or_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _legacy_registry(tmp_path / "hardware_setups.yaml")
    _pin_test_udp_endpoint(monkeypatch)
    _forbid_energy_method_path(monkeypatch)
    monkeypatch.setattr(pp, "platform_operation_lock", _unlocked)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(m2_present=False),
    )
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_args, **_kwargs: pytest.fail("preflight failure toggled hardware"),
    )

    output_root = tmp_path / "preflight-failed"
    with pytest.raises(pp.PlatformStateError, match="M.2-present state"):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            stabilize_s=0,
            measure_s=5,
            output_dir=output_root,
            measurement_runner=lambda *_args, **_kwargs: pytest.fail(
                "preflight failure measured power"
            ),
        )

    evidence = json.loads(
        (output_root / "m2_idle_power_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["events"] == []
    assert evidence["saved"] is False
    assert evidence["recovery_skipped_reason"] == "preflight_not_completed"
    stored_raw = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert stored_raw["hardware_setups"][0]["energy"][
        "accelerator_idle_w"
    ] == pytest.approx(0.75)
