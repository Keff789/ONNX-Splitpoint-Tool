from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
import subprocess

import pytest
import yaml

from onnx_splitpoint_tool import platform_power as pp
from onnx_splitpoint_tool.energy.config import load_hardware_registry
from onnx_splitpoint_tool import platform_power_cli
from onnx_splitpoint_tool.workflow.run_control import (
    EvaluationRunLock,
    WorkflowRunLockedError,
)


def _registry(path: Path, *, idle: float = 0.75) -> Path:
    method = _verified_energy_method(path.parent)
    payload = {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 1,
        "energy_defaults": {
            "data_port": 3000,
            "channel": 0,
            "sample_rate": 2000,
            "physical_scope": "FS",
            "window_label": "command",
        },
        "hardware_setups": [
            {
                "id": "orin_nx_hailo8_01",
                "label": "Test Jetson + Hailo-8",
                "accelerator": "hailo8",
                "host": {
                    "address": "192.0.2.20",
                    "user": "nx",
                    "port": 22,
                    "base_dir": "~/splitpoint_runs",
                },
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    "idle_baseline_w": 12.0,
                    "accelerator_idle_w": idle,
                    "calibration_manifest": method["path"],
                    "calibration_sha256": method["sha256"],
                },
                "power_control": {
                    "udp_port": 3000,
                    "require_ping_before_toggle": True,
                },
            }
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _status(*, m2: bool | None = True, ssh: bool | None = True) -> pp.PlatformStatus:
    return pp.PlatformStatus(
        setup_id="orin_nx_hailo8_01",
        checked_at="2026-09-02T00:00:00+00:00",
        urecs_address="192.0.2.10",
        urecs_host="192.0.2.10",
        urecs_port=3000,
        urecs_configured=True,
        urecs_reachable=True,
        urecs_detail="reachable",
        jetson_host="nx@192.0.2.20:22",
        jetson_configured=True,
        jetson_ssh_ready=ssh,
        jetson_detail="ready" if ssh else "down",
        accelerator="hailo8",
        m2_present=m2,
        m2_detail="detected" if m2 else "not detected",
    )


def _verified_energy_method(tmp_path: Path) -> dict[str, object]:
    path = tmp_path / "verified_energy_method.json"
    if not path.exists():
        path.write_text(
            json.dumps(
                {
                    "schema": "onnx-splitpoint/energy-calibration-manifest",
                    "schema_version": 2,
                    "evidence_mode": "inherited_validated_method",
                    "channel_bindings": [
                        {
                            "setup_id": "orin_nx_hailo8_01",
                            "urecs_address": "192.0.2.10",
                            "data_port": 3000,
                            "channel": 0,
                            "sample_rate_hz": 2000,
                            "scope": "FS",
                            "measurement_point": "complete_system_input",
                        }
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    digest = pp._sha256_file(path)
    expected_bindings = [
        {
            "setup_id": "orin_nx_hailo8_01",
            "urecs_address": "192.0.2.10",
            "data_port": 3000,
            "channel": 0,
            "sample_rate_hz": 2000,
            "scope": "FS",
            "measurement_point": "complete_system_input",
        }
    ]
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "declared_sha256": digest,
        "verification_status": "inherited_validated_method_verified",
        "runtime_binding_id": "orin_nx_hailo8_01",
        "verification": {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": "orin_nx_hailo8_01",
        },
        "expected_channel_bindings": expected_bindings,
        "verified": True,
    }


def _parquet_fixture_bytes(label: str) -> bytes:
    metadata = ("fixture-metadata:" + label).encode()
    return (
        b"PAR1"
        + ("fixture-body:" + label).encode()
        + metadata
        + len(metadata).to_bytes(4, "little")
        + b"PAR1"
    )


def _write_valid_command_window_chain(
    run_dir: Path,
    raw: Path,
    result_path: Path,
    *,
    phase: str,
    power_w: float,
) -> Path:
    """Use the production writer with a complete internally bound v2 chain."""
    from onnx_splitpoint_tool.energy.collector import (
        _write_command_window_binding_v2,
    )

    command = run_dir / "energy_command.sh"
    command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    timing = run_dir / "workload_timing.txt"
    timing.write_text(
        "start_ns=1000000000\nend_ns=2000000000\nrc=0\n",
        encoding="utf-8",
    )
    marker = run_dir / "collector_storage" / "command_window_markers.json"
    marker.write_text(
        json.dumps(
            {
                "schema": "urecs-data-collector/command-window-markers",
                "schema_version": 2,
                "run_id": f"m2_idle_calibration_{phase}",
                "window_id": "repeat:000",
                "command": {
                    "text": str(command.resolve()),
                    "argv": [str(command.resolve())],
                    "process_id": 123,
                    "process_rc": 0,
                    "process_error": None,
                    "start_realtime_ns": 1_000_000_000,
                    "end_realtime_ns": 2_000_000_000,
                    "start_monotonic_ns": 10_000,
                    "end_monotonic_ns": 20_000,
                    "start_clock_read_uncertainty_ns": 10,
                    "end_clock_read_uncertainty_ns": 10,
                },
                "stream": {
                    "source": "fast_firmware",
                    "trace_path": str(raw.resolve()),
                    "sample_rate_hz": 2000,
                    "first_sample_index": 0,
                    "last_sample_index": 1999,
                    "total_samples": 2500,
                    "boundary_uncertainty_samples": 1,
                    "dropped_samples": 0,
                    "trace_covers_window": True,
                },
                "valid_for_final_energy": True,
                "validation_errors": [],
            }
        ),
        encoding="utf-8",
    )
    request = run_dir / "command_window_request.json"
    request_payload = {
        "schema": "onnx-splitpoint/command-window-request",
        "schema_version": 2,
        "binding_method": "collector_sample_marker_crop",
        "run_id": f"m2_idle_calibration_{phase}",
        "window_id": "repeat:000",
        "source": "fast_firmware",
        "marker_path": str(marker.resolve()),
        "marker_sha256": pp._sha256_file(marker),
        "trace_path": str(raw.resolve()),
        "trace_sha256": pp._sha256_file(raw),
        "command_path": str(command.resolve()),
        "command_sha256": pp._sha256_file(command),
        "workload_command_path": str(command.resolve()),
        "workload_command_sha256": pp._sha256_file(command),
        "timing_path": str(timing.resolve()),
        "timing_sha256": pp._sha256_file(timing),
        "first_sample_index": 0,
        "last_sample_index": 1999,
        "sample_count": 2000,
        "interval_count": 1999,
        "sample_rate_hz": 2000,
        "duration_s": 1999 / 2000,
        "process_rc": 0,
        "process_id": 123,
        "process_error": None,
        "start_clock_read_uncertainty_ns": 10,
        "end_clock_read_uncertainty_ns": 10,
        "command_start_realtime_ns": 1_000_000_000,
        "command_end_realtime_ns": 2_000_000_000,
        "command_start_monotonic_ns": 10_000,
        "command_end_monotonic_ns": 20_000,
        "dropped_samples": 0,
        "boundary_uncertainty_samples": 1,
        "maximum_boundary_uncertainty_samples": 64,
        "trace_covers_window": True,
        "valid_for_final_energy": True,
        "total_samples": 2500,
        "index_semantics": "zero_based_inclusive_rows",
        "energy_semantics": "calibrated_input_energy_unsubtracted",
        "energy_field": "firmware_results.energy",
        "created_at_unix_ns": 1,
    }
    request.write_text(json.dumps(request_payload), encoding="utf-8")
    post = run_dir / "postprocessor_window_result.json"
    post.write_text(
        json.dumps(
            {
                "schema": "power-calculations/command-window-result",
                "schema_version": 2,
                "status": "ok",
                "run_id": f"m2_idle_calibration_{phase}",
                "window_id": "repeat:000",
                "source": "fast_firmware",
                "request_path": str(request.resolve()),
                "marker_path": str(marker.resolve()),
                "trace_path": str(raw.resolve()),
                "command_path": str(command.resolve()),
                "timing_path": str(timing.resolve()),
                "results_path": str(result_path.resolve()),
                "request_sha256": pp._sha256_file(request),
                "marker_sha256": pp._sha256_file(marker),
                "trace_sha256": pp._sha256_file(raw),
                "command_sha256": pp._sha256_file(command),
                "timing_sha256": pp._sha256_file(timing),
                "first_sample_index": 0,
                "last_sample_index": 1999,
                "sample_count": 2000,
                "interval_count": 1999,
                "sample_rate_hz": 2000,
                "duration_s": 1999 / 2000,
                "process_rc": 0,
                "command_process_id": 123,
                "command_start_realtime_ns": 1_000_000_000,
                "command_end_realtime_ns": 2_000_000_000,
                "command_start_monotonic_ns": 10_000,
                "command_end_monotonic_ns": 20_000,
                "command_start_clock_read_uncertainty_ns": 10,
                "command_end_clock_read_uncertainty_ns": 10,
                "drop_count": 0,
                "boundary_uncertainty_samples": 1,
                "maximum_boundary_uncertainty_samples": 64,
                "trace_covers_window": True,
                "energy_semantics": "calibrated_input_energy_unsubtracted",
                "energy_field": "firmware_results.energy",
                "energy_j": power_w * (1999 / 2000),
            }
        ),
        encoding="utf-8",
    )
    binding = _write_command_window_binding_v2(
        run_dir,
        request_path=request,
        postprocessor_result_path=post,
        result_path=result_path,
    )
    assert binding is not None
    return binding


def _measurement_result(
    output_dir: Path,
    method: dict[str, object],
    power_w: float,
    *,
    materialize_capture: bool = False,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, object] = {
        "ok": True,
        "status": "ok",
        "avg_power_w": power_w,
        "energy_calibration_manifest": method["path"],
        "energy_calibration_sha256": method["sha256"],
        "energy_calibration_verification": {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": "orin_nx_hailo8_01",
        },
    }
    if materialize_capture:
        phase = output_dir.name
        run_dir = output_dir / "run_000"
        storage = run_dir / "collector_storage"
        storage.mkdir(parents=True, exist_ok=True)
        raw = storage / "raw.parquet"
        raw.write_bytes(_parquet_fixture_bytes(phase))
        raw_sha = pp._sha256_file(raw)
        result_yaml = run_dir / "processed" / "results.yaml"
        result_yaml.parent.mkdir(parents=True, exist_ok=True)
        result_yaml.write_text(
            json.dumps(
                {
                    "firmware_results": {
                        "energy": power_w * (1999 / 2000),
                        "duration": 1999 / 2000,
                        "start_stop_idx": [0, 1999],
                        "max_frame_energy": 0.0,
                        "idle_frame_energy": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        result_sha = pp._sha256_file(result_yaml)
        command_binding = _write_valid_command_window_chain(
            run_dir,
            raw,
            result_yaml,
            phase=phase,
            power_w=power_w,
        )
        run = {
            "run_index": 0,
            "status": "collector_finished",
            "collector_rc": 0,
            "workload_command_rc": 0,
            "postprocess_status": "ok",
            "final_energy_gate_status": "pass",
            "avg_power_w": power_w,
            "energy_calibration_sha256": method["sha256"],
            "energy_calibration_verification": {
                "verified": True,
                "status": "inherited_validated_method_verified",
                "runtime_binding_id": "orin_nx_hailo8_01",
            },
            "raw_input_energy_provenance_status": (
                "verified_calibrated_input_energy_unsubtracted"
            ),
            "raw_input_energy_provenance": {
                "trace_path": str(raw.resolve()),
                "trace_sha256": raw_sha,
                "result_path": str(result_yaml.resolve()),
                "result_sha256": result_sha,
                "binding_status": "verified",
            },
            "command_window_binding_manifest": str(command_binding.resolve()),
            "command_window_trace_binding": {
                "status": "verified",
                "manifest_path": str(command_binding.resolve()),
            },
        }
        (run_dir / "energy_summary.json").write_text(
            json.dumps(run) + "\n", encoding="utf-8"
        )
        aggregate = {
            **result,
            "setup_id": "orin_nx_hailo8_01",
            "run_id": f"m2_idle_calibration_{phase}",
            "runs": [run],
        }
        for name in ("energy_aggregate.json", "energy_summary.json"):
            (output_dir / name).write_text(
                json.dumps(aggregate, default=str) + "\n", encoding="utf-8"
            )
        result["runs"] = [run]
    return result


def test_parse_urecs_address_and_power_defaults() -> None:
    assert pp.parse_urecs_address("192.0.2.1", 3000) == ("192.0.2.1", 3000)
    assert pp.parse_urecs_address("192.0.2.1:3100", 3000) == ("192.0.2.1", 3100)
    assert pp.parse_urecs_address("[fd00::1]:3200", 3000) == ("fd00::1", 3200)
    assert pp.parse_urecs_address("fd00::1", 3000) == ("fd00::1", 3000)
    with pytest.raises(
        pp.PlatformConfigurationError, match="calibration_measure_s"
    ):
        pp.merged_power_control(
            {"calibration_measure_s": 2, "udp_terminator": "none"}
        )
    merged = pp.merged_power_control(
        {"calibration_measure_s": 5, "udp_terminator": "none"}
    )
    assert merged["calibration_measure_s"] == 5.0
    assert merged["udp_port"] == 3000
    assert merged["jetson_command"] == "jetson"
    assert merged["m2_command"] == "m.2"
    assert merged["m2_post_boot_settle_s"] == 5.0
    assert merged["m2_verify_observations"] == 2


def test_send_udp_command_uses_exact_configured_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[tuple[bytes, tuple[str, int]]] = []

    class FakeSocket:
        def __init__(self, *_args):
            self.timeout = None

        def settimeout(self, timeout):
            self.timeout = timeout

        def sendto(self, payload, sockaddr):
            sent.append((payload, sockaddr))
            return len(payload)

        def close(self):
            return None

    monkeypatch.setattr(
        pp.socket,
        "getaddrinfo",
        lambda host, port, type: [(pp.socket.AF_INET, pp.socket.SOCK_DGRAM, 0, "", (host, port))],
    )
    result = pp.send_udp_command(
        "192.0.2.10",
        "m2",
        port=3000,
        terminator="lf",
        socket_factory=FakeSocket,
    )
    assert result["ok"] is True
    assert result["delivery_evidence"] == "local_udp_send_only_unacknowledged"
    assert result["controller_acknowledged"] is False
    assert result["physical_state_confirmed"] is False
    assert sent == [(b"m2\n", ("192.0.2.10", 3000))]

    sent.clear()
    pp.send_udp_command(
        "192.0.2.10:3001",
        "jetson",
        terminator="none",
        socket_factory=FakeSocket,
    )
    assert sent == [(b"jetson", ("192.0.2.10", 3001))]


def test_registry_migration_adds_power_defaults_without_overwriting_user_values(tmp_path: Path) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    data = load_hardware_registry(path)
    assert data["schema_version"] == 2
    setup = next(row for row in data["hardware_setups"] if row["id"] == "orin_nx_hailo8_01")
    assert setup["power_control"]["udp_port"] == 3000
    assert setup["power_control"]["jetson_command"] == "jetson"
    assert setup["power_control"]["calibration_stabilize_s"] == 30.0
    assert setup["energy"]["accelerator_idle_w"] == 0.75


def test_status_probe_combines_ping_authenticated_ssh_and_accelerator_probe(tmp_path: Path) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")

    class FakeTransport:
        def __init__(self, host):
            self.host = host

        def test_connection(self, timeout_s=5):
            return True, f"authenticated {self.host.user_host}"

        def run_read_only(self, command, timeout_s=12):
            assert "__SPLITPOINT_M2_PRESENT__" in command
            return 0, (
                "__SPLITPOINT_M2_KNOWN__=1\n"
                "__SPLITPOINT_M2_PRESENT__=1\n"
                "__SPLITPOINT_M2_REASON__=/dev/hailo0\n"
            )

    def ping_runner(command, **_kwargs):
        assert command[-1] == "192.0.2.10"
        return subprocess.CompletedProcess(command, 0, stdout="1 received", stderr="")

    status = pp.probe_platform_status(
        "orin_nx_hailo8_01",
        registry_path=path,
        transport_factory=FakeTransport,
        ping_runner=ping_runner,
    )
    assert status.urecs_reachable is True
    assert status.jetson_ssh_ready is True
    assert status.m2_present is True
    assert status.m2_detail == "/dev/hailo0"


def test_calibration_saves_delta_only_after_both_verified_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml", idle=0.75)
    method = _verified_energy_method(tmp_path)
    monkeypatch.setattr(
        pp, "_resolve_verified_energy_method", lambda *_args, **_kwargs: method
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.campaign.verify_energy_calibration_manifest",
        lambda *_args, **_kwargs: {"ok": True},
    )
    configured_calls: list[object] = []

    def verify_configured(_setup_id, **kwargs):
        configured_calls.append(kwargs.get("registry"))
        return {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "path": method["path"],
            "sha256": method["sha256"],
            "configured_method_admission": {"ok": True},
            "source_integrity_verification": {
                "ok": True,
                "status": "verified",
            },
        }

    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_configured_energy_method",
        verify_configured,
    )
    monkeypatch.setattr(pp, "assert_no_active_workflow", lambda **_kwargs: None)

    @contextmanager
    def unlocked(_setup_id, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    current = {"m2": True}
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(m2=current["m2"]),
    )
    transitions: list[bool] = []

    def transition(_setup_id, desired, **_kwargs):
        transitions.append(bool(desired))
        current["m2"] = bool(desired)
        return {
            "ok": True,
            "changed": True,
            "desired_accelerator_present": bool(desired),
            "after": _status(m2=bool(desired)).to_dict(),
        }

    monkeypatch.setattr(pp, "_set_m2_state_locked", transition)
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)
    powers = iter((11.25, 12.85))

    def measurement_runner(_command, output_dir, **_kwargs):
        return _measurement_result(
            Path(output_dir), method, next(powers), materialize_capture=True
        )

    result = pp.calibrate_m2_accelerator_idle_power(
        "orin_nx_hailo8_01",
        registry_path=path,
        stabilize_s=0,
        measure_s=5,
        output_dir=tmp_path / "calibration",
        measurement_runner=measurement_runner,
    )
    assert transitions == [False, True]
    assert result.saved is True
    assert result.accelerator_idle_power_w == pytest.approx(1.60)
    assert configured_calls and all(
        isinstance(value, dict) for value in configured_calls
    )
    stored = load_hardware_registry(path)
    setup = next(row for row in stored["hardware_setups"] if row["id"] == "orin_nx_hailo8_01")
    assert setup["energy"]["accelerator_idle_w"] == pytest.approx(1.60)
    evidence = json.loads(
        (tmp_path / "calibration" / "m2_idle_power_calibration.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "ok"
    assert evidence["saved"] is True
    assert evidence["restored_m2_on"] is True
    assert evidence["m2_off"]["pre_measurement_status"]["m2_present"] is False
    assert evidence["m2_off"]["post_measurement_status"]["m2_present"] is False
    assert evidence["m2_on"]["pre_measurement_status"]["m2_present"] is True
    assert evidence["m2_on"]["post_measurement_status"]["m2_present"] is True
    assert evidence["calibration_binding_verification"][
        "accelerator_idle_calibration_verified"
    ] is True
    assert evidence["calibration_binding_verification"][
        "accelerator_idle_calibration_status"
    ] == "verified"


def test_calibration_failure_preserves_previous_registry_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml", idle=0.75)
    method = _verified_energy_method(tmp_path)
    monkeypatch.setattr(
        pp, "_resolve_verified_energy_method", lambda *_args, **_kwargs: method
    )
    monkeypatch.setattr(pp, "assert_no_active_workflow", lambda **_kwargs: None)

    @contextmanager
    def unlocked(_setup_id, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    current = {"m2": True}
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(m2=current["m2"]),
    )

    def transition(_setup_id, desired, **_kwargs):
        current["m2"] = bool(desired)
        return {"ok": True, "desired_present": bool(desired)}

    monkeypatch.setattr(pp, "_set_m2_state_locked", transition)
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)
    calls = 0

    def measurement_runner(_command, output_dir, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _measurement_result(Path(output_dir), method, 11.25)
        raise RuntimeError("second capture failed")

    with pytest.raises(RuntimeError, match="second capture failed"):
        pp.calibrate_m2_accelerator_idle_power(
            "orin_nx_hailo8_01",
            registry_path=path,
            stabilize_s=0,
            measure_s=5,
            output_dir=tmp_path / "failed-calibration",
            measurement_runner=measurement_runner,
        )
    stored = load_hardware_registry(path)
    setup = next(row for row in stored["hardware_setups"] if row["id"] == "orin_nx_hailo8_01")
    assert setup["energy"]["accelerator_idle_w"] == pytest.approx(0.75)
    evidence = json.loads(
        (tmp_path / "failed-calibration" / "m2_idle_power_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    assert evidence["status"] == "failed"
    assert evidence["saved"] is False


def test_power_control_enabled_is_authoritative(tmp_path: Path) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["hardware_setups"][0]["power_control"]["enabled"] = False
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(pp.PlatformConfigurationError, match="disabled"):
        pp.toggle_jetson("orin_nx_hailo8_01", registry_path=path)


def test_unsupported_accelerator_presence_is_unknown() -> None:
    class FakeTransport:
        def __init__(self, _host):
            pass

        def run_read_only(self, command, timeout_s=12):
            assert "__SPLITPOINT_M2_KNOWN__" in command
            return 0, (
                "__SPLITPOINT_M2_KNOWN__=0\n"
                "__SPLITPOINT_M2_PRESENT__=0\n"
                "__SPLITPOINT_M2_REASON__=unsupported\n"
            )

    present, detail = pp.probe_accelerator_presence(
        {
            "id": "unknown_01",
            "accelerator": "unknown_accelerator",
            "host": {"address": "192.0.2.20", "user": "nx", "port": 22},
        },
        transport_factory=FakeTransport,
    )
    assert present is None
    assert detail == "unsupported"


def test_jetson_off_verification_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")

    @contextmanager
    def unlocked(_setup_id, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    statuses = iter((_status(), _status()))
    monkeypatch.setattr(
        pp, "probe_platform_status", lambda *_args, **_kwargs: next(statuses)
    )
    monkeypatch.setattr(
        pp, "_stop_jetson_and_cut_rail", lambda *_args, **_kwargs: []
    )

    with pytest.raises(pp.PlatformStateError, match="target was not verified"):
        pp.set_jetson_state(
            "orin_nx_hailo8_01",
            False,
            expected_ssh_ready=True,
            registry_path=path,
        )


def test_cli_returns_nonzero_for_jetson_verification_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*_args, **_kwargs):
        raise pp.PlatformStateError("Jetson remained SSH-ready")

    monkeypatch.setattr(platform_power_cli, "toggle_jetson", fail)
    assert platform_power_cli.main(["toggle-jetson", "orin_nx_hailo8_01"]) == 2
    assert "PlatformStateError" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("expected_ssh_ready", "fresh_ssh_ready", "desired_up"),
    [
        (False, True, True),
        (True, False, False),
    ],
    ids=("stale_down_became_ready", "stale_ready_became_down"),
)
def test_explicit_jetson_target_aborts_without_mutation_on_stale_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    expected_ssh_ready: bool,
    fresh_ssh_ready: bool,
    desired_up: bool,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")

    @contextmanager
    def unlocked(_setup_id, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(ssh=fresh_ssh_ready),
    )
    mutations: list[str] = []
    monkeypatch.setattr(
        pp,
        "_start_jetson",
        lambda *_args, **_kwargs: mutations.append("start"),
    )
    monkeypatch.setattr(
        pp,
        "_stop_jetson_and_cut_rail",
        lambda *_args, **_kwargs: mutations.append("stop"),
    )

    with pytest.raises(pp.PlatformStateError, match="ABORT_NO_MUTATION"):
        pp.set_jetson_state(
            "orin_nx_hailo8_01",
            desired_up,
            expected_ssh_ready=expected_ssh_ready,
            registry_path=path,
        )

    assert mutations == []


def test_cli_set_jetson_uses_explicit_target(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, bool]] = []

    def set_target(setup_id, desired_up, **_kwargs):
        calls.append((setup_id, desired_up))
        return {"ok": True, "desired_ssh_ready": desired_up}

    monkeypatch.setattr(platform_power_cli, "set_jetson_state", set_target)
    assert (
        platform_power_cli.main(
            ["set-jetson", "orin_nx_hailo8_01", "off"]
        )
        == 0
    )
    assert calls == [("orin_nx_hailo8_01", False)]


def test_m2_post_boot_verification_rejects_delayed_enumeration_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml")
    registry, setup, cfg = pp.resolve_setup(
        "orin_nx_hailo8_01", registry_path=path
    )
    cfg["power_control"].update(
        {
            "m2_post_boot_settle_s": 0.0,
            "m2_verify_observations": 2,
            "m2_verify_interval_s": 0.0,
        }
    )
    # The first post-boot absence could merely be driver-enumeration lag.  A
    # second observation sees the device and must invalidate the off target.
    statuses = iter(
        (
            _status(m2=True),
            _status(m2=False),
            _status(m2=True),
        )
    )
    monkeypatch.setattr(
        pp, "probe_platform_status", lambda *_args, **_kwargs: next(statuses)
    )
    monkeypatch.setattr(
        pp, "_stop_jetson_and_cut_rail", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        pp,
        "_send_configured_toggle",
        lambda *_args, **_kwargs: {
            "ok": True,
            "delivery_evidence": "local_udp_send_only_unacknowledged",
        },
    )
    monkeypatch.setattr(pp, "_start_jetson", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)

    with pytest.raises(pp.PlatformStateError, match="observation 2/2"):
        pp._set_m2_state_locked(
            "orin_nx_hailo8_01",
            False,
            registry=registry,
            setup=setup,
            cfg=cfg,
            callback=None,
            transport_factory=pp.SSHTransport,
        )


def test_calibration_state_change_during_capture_fails_without_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml", idle=0.75)
    method = _verified_energy_method(tmp_path)
    monkeypatch.setattr(
        pp, "_resolve_verified_energy_method", lambda *_args, **_kwargs: method
    )

    @contextmanager
    def unlocked(_setup_id, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    current = {"m2": True}
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(m2=current["m2"]),
    )
    transitions: list[bool] = []

    def transition(_setup_id, desired, **_kwargs):
        transitions.append(bool(desired))
        current["m2"] = bool(desired)
        return {"ok": True, "desired_present": bool(desired)}

    monkeypatch.setattr(pp, "_set_m2_state_locked", transition)

    def measurement_runner(_command, output_dir, **_kwargs):
        current["m2"] = True
        return _measurement_result(Path(output_dir), method, 11.25)

    output = tmp_path / "state-changed-during-capture"
    with pytest.raises(
        pp.PlatformStateError, match="m2_off/post_measurement"
    ):
        pp.calibrate_m2_accelerator_idle_power(
            "orin_nx_hailo8_01",
            registry_path=path,
            stabilize_s=0,
            measure_s=5,
            output_dir=output,
            measurement_runner=measurement_runner,
        )

    assert transitions == [False]
    stored = load_hardware_registry(path)
    setup = next(
        row
        for row in stored["hardware_setups"]
        if row["id"] == "orin_nx_hailo8_01"
    )
    assert setup["energy"]["accelerator_idle_w"] == pytest.approx(0.75)
    evidence = json.loads(
        (output / "m2_idle_power_calibration.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "failed"
    assert evidence["saved"] is False


def test_failed_calibration_preflight_never_runs_recovery_powercycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml", idle=0.75)
    method = _verified_energy_method(tmp_path)
    monkeypatch.setattr(
        pp, "_resolve_verified_energy_method", lambda *_args, **_kwargs: method
    )

    @contextmanager
    def unlocked(_setup_id, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    monkeypatch.setattr(
        pp, "probe_platform_status", lambda *_args, **_kwargs: _status(m2=False)
    )
    transitions: list[bool] = []
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda _setup_id, desired, **_kwargs: transitions.append(bool(desired)),
    )
    output = tmp_path / "preflight-failed"

    with pytest.raises(pp.PlatformStateError, match="verified M.2-present"):
        pp.calibrate_m2_accelerator_idle_power(
            "orin_nx_hailo8_01",
            registry_path=path,
            stabilize_s=0,
            measure_s=5,
            output_dir=output,
        )

    assert transitions == []
    evidence = json.loads(
        (output / "m2_idle_power_calibration.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "failed"
    assert evidence["recovery_skipped_reason"] == "preflight_not_completed"
    assert "recovery" not in evidence


@pytest.mark.parametrize(
    ("failure", "expected_type"),
    [
        (KeyboardInterrupt(), "KeyboardInterrupt"),
        (SystemExit(73), "SystemExit"),
    ],
    ids=("keyboard_interrupt", "system_exit"),
)
def test_calibration_interrupt_recovers_and_checkpoints_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_type: str,
) -> None:
    path = _registry(tmp_path / "hardware_setups.yaml", idle=0.75)
    method = _verified_energy_method(tmp_path)
    monkeypatch.setattr(
        pp, "_resolve_verified_energy_method", lambda *_args, **_kwargs: method
    )

    @contextmanager
    def unlocked(_setup_id, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    current = {"m2": True}
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: _status(m2=current["m2"]),
    )
    transitions: list[bool] = []

    def transition(_setup_id, desired, **_kwargs):
        transitions.append(bool(desired))
        current["m2"] = bool(desired)
        return {"ok": True, "desired_present": bool(desired)}

    monkeypatch.setattr(pp, "_set_m2_state_locked", transition)
    monkeypatch.setattr(pp.time, "sleep", lambda _seconds: None)

    def interrupted_measurement(*_args, **_kwargs):
        raise failure

    output = tmp_path / f"interrupt-{expected_type}"
    with pytest.raises(type(failure)):
        pp.calibrate_m2_accelerator_idle_power(
            "orin_nx_hailo8_01",
            registry_path=path,
            stabilize_s=0,
            measure_s=5,
            output_dir=output,
            measurement_runner=interrupted_measurement,
        )

    assert transitions == [False, True]
    evidence = json.loads(
        (output / "m2_idle_power_calibration.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "failed"
    assert evidence["error_type"] == expected_type
    assert evidence["interrupted"] is True
    assert evidence["recovery"]["ok"] is True
    assert evidence["restored_m2_on"] is True


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX flock")
def test_evaluation_run_and_platform_power_use_one_global_interlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    out_root = tmp_path / "runs"
    first = EvaluationRunLock(
        out_root=out_root,
        run_dir=out_root / "EvaluationRun_active",
        owner={"session_id": "test-evaluation"},
    )
    with first:
        with pytest.raises(pp.PlatformBusyError, match="EvaluationRun is active"):
            with pp.platform_operation_lock("orin_nx_hailo8_01"):
                pytest.fail("exclusive platform lock must not be admitted")

    with pp.platform_operation_lock("orin_nx_hailo8_01"):
        second = EvaluationRunLock(
            out_root=out_root,
            run_dir=out_root / "EvaluationRun_blocked",
            owner={"session_id": "test-platform-power"},
        )
        with pytest.raises(WorkflowRunLockedError, match="platform-power"):
            second.acquire()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX flock")
def test_legacy_workflow_guard_is_checked_inside_platform_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    monkeypatch.setenv("HOME", str(tmp_path))
    lock_root = tmp_path / ".onnx_splitpoint_tool" / "locks"
    lock_root.mkdir(parents=True)
    legacy = (lock_root / "legacy_campaign.lock").open("a+b")
    try:
        fcntl.flock(legacy.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(pp.PlatformBusyError, match="workflow lock is active"):
            with pp.platform_operation_lock("orin_nx_hailo8_01"):
                pytest.fail("held legacy workflow lock must block power operation")
    finally:
        fcntl.flock(legacy.fileno(), fcntl.LOCK_UN)
        legacy.close()
