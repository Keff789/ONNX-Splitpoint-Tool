from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.energy import collector
from onnx_splitpoint_tool.energy.comparison import (
    CALIBRATION_BINDING_SCHEMA,
    CALIBRATION_BINDING_SCHEMA_VERSION,
    verify_accelerator_idle_calibration_binding,
)
from onnx_splitpoint_tool.energy.config import (
    DuplicateEnergySetupIdError,
    energy_defaults_from_registry,
    energy_setup_from_raw,
    energy_setup_from_registry,
    load_hardware_registry,
)


SETUP_ID = "orin_nx_hailo8_01"
URECS_ADDRESS = "192.168.0.197"
DATA_PORT = 3000
RUNTIME_BINDING_ID = SETUP_ID
JETSON_HOST = {
    "address": "192.168.0.104",
    "user": "nx",
    "port": 22,
    "ssh_extra_args": "",
}


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))
    return path.resolve()


def _parquet_fixture_bytes(label: str) -> bytes:
    metadata = ("fixture-metadata:" + label).encode()
    return (
        b"PAR1"
        + ("fixture-body:" + label).encode()
        + metadata
        + len(metadata).to_bytes(4, "little")
        + b"PAR1"
    )


def _identity(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _payload_sha(payload: dict) -> str:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "binding_payload_sha256"
    }
    rendered = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


def _measurement_payload(
    power_w: float, method_sha: str, phase: str
) -> dict:
    return {
        "ok": True,
        "status": "ok",
        "setup_id": SETUP_ID,
        "run_id": f"m2_idle_calibration_{phase}",
        "avg_power_w": power_w,
        "energy_calibration_sha256": method_sha,
        "energy_calibration_verification": {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": RUNTIME_BINDING_ID,
        },
    }


def _run_payload(
    power_w: float,
    method_sha: str,
    *,
    raw: dict[str, str],
    result: dict[str, str],
    command_binding: dict[str, str],
) -> dict:
    return {
        "status": "collector_finished",
        "collector_rc": 0,
        "workload_command_rc": 0,
        "postprocess_status": "ok",
        "final_energy_gate_status": "pass",
        "avg_power_w": power_w,
        "energy_calibration_sha256": method_sha,
        "energy_calibration_verification": {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": RUNTIME_BINDING_ID,
        },
        "raw_input_energy_provenance_status": (
            "verified_calibrated_input_energy_unsubtracted"
        ),
        "raw_input_energy_provenance": {
            "binding_status": "verified",
            "trace_path": raw["path"],
            "trace_sha256": raw["sha256"],
            "result_path": result["path"],
            "result_sha256": result["sha256"],
        },
        "command_window_binding_manifest": command_binding["path"],
        "command_window_trace_binding": {
            "status": "verified",
            "manifest_path": command_binding["path"],
        },
    }


def _observation(m2_present: bool) -> dict[str, object]:
    return {
        "setup_id": SETUP_ID,
        "urecs_address": URECS_ADDRESS,
        "jetson_host": dict(JETSON_HOST),
        "jetson_ssh_ready": True,
        "m2_present": m2_present,
    }


def _command_binding_payload(
    raw: dict[str, str],
    result: dict[str, str],
    phase_root: Path,
    *,
    phase: str,
    power_w: float,
) -> dict[str, object]:
    """Materialize the complete internally cross-bound production v2 chain."""
    run_root = phase_root / "run_000"
    command_path = run_root / "energy_command.sh"
    command_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    timing_path = run_root / "workload_timing.txt"
    timing_path.write_text(
        "start_ns=1000000000\nend_ns=2000000000\nrc=0\n",
        encoding="utf-8",
    )
    marker_path = _write_json(
        run_root / "collector_storage" / "command_window_markers.json",
        {
            "schema": "urecs-data-collector/command-window-markers",
            "schema_version": 2,
            "run_id": f"m2_idle_calibration_{phase}",
            "window_id": "repeat:000",
            "command": {
                "text": str(command_path.resolve()),
                "argv": [str(command_path.resolve())],
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
                "trace_path": raw["path"],
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
        },
    )
    request_path = _write_json(
        run_root / "command_window_request.json",
        {
            "schema": "onnx-splitpoint/command-window-request",
            "schema_version": 2,
            "binding_method": "collector_sample_marker_crop",
            "run_id": f"m2_idle_calibration_{phase}",
            "window_id": "repeat:000",
            "source": "fast_firmware",
            "marker_path": str(marker_path),
            "marker_sha256": _identity(marker_path)["sha256"],
            "trace_path": raw["path"],
            "trace_sha256": raw["sha256"],
            "command_path": str(command_path.resolve()),
            "command_sha256": _identity(command_path)["sha256"],
            "workload_command_path": str(command_path.resolve()),
            "workload_command_sha256": _identity(command_path)["sha256"],
            "timing_path": str(timing_path.resolve()),
            "timing_sha256": _identity(timing_path)["sha256"],
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
        },
    )
    post_path = _write_json(
        run_root / "postprocessor_window_result.json",
        {
            "schema": "power-calculations/command-window-result",
            "schema_version": 2,
            "status": "ok",
            "run_id": f"m2_idle_calibration_{phase}",
            "window_id": "repeat:000",
            "source": "fast_firmware",
            "request_path": str(request_path),
            "marker_path": str(marker_path),
            "trace_path": raw["path"],
            "command_path": str(command_path.resolve()),
            "timing_path": str(timing_path.resolve()),
            "results_path": result["path"],
            "request_sha256": _identity(request_path)["sha256"],
            "marker_sha256": _identity(marker_path)["sha256"],
            "trace_sha256": raw["sha256"],
            "command_sha256": _identity(command_path)["sha256"],
            "timing_sha256": _identity(timing_path)["sha256"],
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
        },
    )
    payload: dict[str, object] = {
        "schema": "onnx-splitpoint/command-window-binding",
        "schema_version": 2,
        "binding_method": "collector_sample_marker_crop",
        "run_id": f"m2_idle_calibration_{phase}",
        "window_id": "repeat:000",
        "source": "fast_firmware",
        "request_path": str(request_path),
        "request_sha256": _identity(request_path)["sha256"],
        "marker_path": str(marker_path),
        "marker_sha256": _identity(marker_path)["sha256"],
        "trace_path": raw["path"],
        "trace_sha256": raw["sha256"],
        "command_path": str(command_path.resolve()),
        "command_sha256": _identity(command_path)["sha256"],
        "workload_command_path": str(command_path.resolve()),
        "workload_command_sha256": _identity(command_path)["sha256"],
        "timing_path": str(timing_path.resolve()),
        "timing_sha256": _identity(timing_path)["sha256"],
        "postprocessor_result_path": str(post_path),
        "postprocessor_result_sha256": _identity(post_path)["sha256"],
        "result_path": result["path"],
        "result_sha256": result["sha256"],
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
        "dropped_samples": 0,
        "boundary_uncertainty_samples": 1,
        "maximum_boundary_uncertainty_samples": 64,
        "trace_covers_window": True,
        "energy_semantics": "calibrated_input_energy_unsubtracted",
        "energy_field": "firmware_results.energy",
        "energy_j": power_w * (1999 / 2000),
        "created_at_unix_ns": 1,
    }
    return payload


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, SimpleNamespace]:
    monkeypatch.setattr(
        "onnx_splitpoint_tool.campaign.verify_energy_calibration_manifest",
        lambda *_args, **_kwargs: {"ok": True},
    )
    method = _write_json(
        tmp_path / "method" / "energy_calibration_manifest.json",
        {
            "schema": "onnx-splitpoint/energy-calibration-manifest",
            "schema_version": 2,
            "evidence_mode": "inherited_validated_method",
            "method": {"sample_rate_hz": 2000},
            "channel_bindings": [
                {
                    "setup_id": SETUP_ID,
                    "urecs_address": URECS_ADDRESS,
                    "data_port": DATA_PORT,
                    "channel": 0,
                    "sample_rate_hz": 2000,
                    "scope": "FS",
                }
            ],
        },
    )
    method_id = _identity(method)
    calibration_root = tmp_path / "calibration"
    captures: dict[str, dict] = {}
    for phase, power in (("m2_off", 10.0), ("m2_on", 12.25)):
        phase_root = calibration_root / phase
        aggregate = _write_json(
            phase_root / "energy_aggregate.json",
            _measurement_payload(power, method_id["sha256"], phase),
        )
        report = _write_json(
            phase_root / "energy_summary.json",
            _measurement_payload(power, method_id["sha256"], phase),
        )
        raw = phase_root / "run_000" / "collector_storage" / "raw.parquet"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(_parquet_fixture_bytes(phase))
        raw_id = _identity(raw)
        result = phase_root / "run_000" / "processed" / "results.yaml"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            json.dumps(
                {
                    "firmware_results": {
                        "energy": power * (1999 / 2000),
                        "duration": 1999 / 2000,
                        "start_stop_idx": [0, 1999],
                        "max_frame_energy": 0.0,
                        "idle_frame_energy": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        result_id = _identity(result)
        command_binding = _write_json(
            phase_root / "run_000" / "command_window_binding.json",
            _command_binding_payload(
                raw_id,
                result_id,
                phase_root,
                phase=phase,
                power_w=power,
            ),
        )
        command_binding_id = _identity(command_binding)
        run = _write_json(
            phase_root / "run_000" / "energy_summary.json",
            _run_payload(
                power,
                method_id["sha256"],
                raw=raw_id,
                result=result_id,
                command_binding=command_binding_id,
            ),
        )
        captures[phase] = {
            "avg_power_w": power,
            "aggregate": _identity(aggregate),
            "report": _identity(report),
            "raw": raw_id,
            "run": _identity(run),
            "command_window_binding": command_binding_id,
            "postprocessor_result": result_id,
        }
    method_binding = {
        **method_id,
        "verification_status": "inherited_validated_method_verified",
        "runtime_binding_id": RUNTIME_BINDING_ID,
    }
    state_observations = {
        "initial": _observation(True),
        "m2_off": {
            "pre_measurement": _observation(False),
            "post_measurement": _observation(False),
        },
        "m2_on": {
            "pre_measurement": _observation(True),
            "post_measurement": _observation(True),
        },
    }
    transitions = {
        "m2_off": {
            "ok": True,
            "changed": True,
            "desired_accelerator_present": False,
            "after": _observation(False),
        },
        "m2_on": {
            "ok": True,
            "changed": True,
            "desired_accelerator_present": True,
            "after": _observation(True),
        },
    }
    evidence = _write_json(
        calibration_root / "accelerator_idle_calibration_evidence.json",
        {
            "schema": "onnx-splitpoint/accelerator-idle-calibration-evidence",
            "schema_version": 1,
            "setup_id": SETUP_ID,
            "accelerator": "hailo8",
            "urecs_address": URECS_ADDRESS,
            "data_port": DATA_PORT,
            "jetson_host": dict(JETSON_HOST),
            "started_at": "2026-09-02T12:00:00+00:00",
            "finished_at": "2026-09-02T12:03:00+00:00",
            "status": "ok",
            "restored_m2_on": True,
            "idle_power_without_m2_w": 10.0,
            "idle_power_with_m2_w": 12.25,
            "accelerator_idle_power_w": 2.25,
            "energy_calibration_manifest": method_binding,
            "captures": captures,
            "state_observations": state_observations,
            "transitions": transitions,
        },
    )
    binding = {
        "schema": CALIBRATION_BINDING_SCHEMA,
        "schema_version": CALIBRATION_BINDING_SCHEMA_VERSION,
        "setup_id": SETUP_ID,
        "accelerator": "hailo8",
        "urecs_address": URECS_ADDRESS,
        "data_port": DATA_PORT,
        "jetson_host": dict(JETSON_HOST),
        "started_at": "2026-09-02T12:00:00+00:00",
        "finished_at": "2026-09-02T12:03:00+00:00",
        "idle_power_without_m2_w": 10.0,
        "idle_power_with_m2_w": 12.25,
        "accelerator_idle_power_w": 2.25,
        "restored_m2_on": True,
        "status": "ok",
        "energy_calibration_manifest": method_binding,
        "calibration_evidence": _identity(evidence),
        "captures": captures,
        "state_observations": state_observations,
        "transitions": transitions,
    }
    binding["binding_payload_sha256"] = _payload_sha(binding)
    binding_path = _write_json(
        calibration_root / "accelerator_idle_calibration_binding.json",
        binding,
    )
    setup = SimpleNamespace(
        setup_id=SETUP_ID,
        accelerator="hailo8",
        urecs_address=URECS_ADDRESS,
        data_port=DATA_PORT,
        jetson_address=JETSON_HOST["address"],
        jetson_user=JETSON_HOST["user"],
        jetson_port=JETSON_HOST["port"],
        jetson_ssh_extra_args=JETSON_HOST["ssh_extra_args"],
        calibration_manifest=method_id["path"],
        calibration_sha256=method_id["sha256"],
        accelerator_idle_w=2.25,
        accelerator_idle_calibration_binding_path=str(binding_path),
        accelerator_idle_calibration_binding_sha256=hashlib.sha256(
            binding_path.read_bytes()
        ).hexdigest(),
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "path": method_id["path"],
            "sha256": method_id["sha256"],
            "configured_method_admission": {"ok": True},
            "source_integrity_verification": {
                "ok": True,
                "status": "verified",
            },
        },
    )
    return binding, setup


def _reseal_binding(binding: dict, setup: SimpleNamespace) -> None:
    binding["binding_payload_sha256"] = _payload_sha(binding)
    path = Path(setup.accelerator_idle_calibration_binding_path)
    path.write_bytes(_json_bytes(binding))
    setup.accelerator_idle_calibration_binding_sha256 = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _reseal_evidence_and_binding(binding: dict, setup: SimpleNamespace) -> None:
    evidence_path = Path(binding["calibration_evidence"]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["captures"] = binding["captures"]
    evidence["state_observations"] = binding.get("state_observations")
    evidence["transitions"] = binding.get("transitions")
    evidence_path.write_bytes(_json_bytes(evidence))
    binding["calibration_evidence"] = _identity(evidence_path)
    _reseal_binding(binding, setup)


def _reseal_complete_method_chain(
    binding: dict, setup: SimpleNamespace
) -> dict[str, str]:
    """Re-hash every method-bearing fixture artefact after method mutation."""
    method_identity = _identity(Path(setup.calibration_manifest))
    setup.calibration_sha256 = method_identity["sha256"]
    method_binding = {
        **method_identity,
        "verification_status": "inherited_validated_method_verified",
        "runtime_binding_id": RUNTIME_BINDING_ID,
    }
    binding["energy_calibration_manifest"] = method_binding
    for phase in ("m2_off", "m2_on"):
        capture = binding["captures"][phase]
        for role in ("aggregate", "report"):
            artifact_path = Path(capture[role]["path"])
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            payload["energy_calibration_sha256"] = method_identity["sha256"]
            artifact_path.write_bytes(_json_bytes(payload))
            capture[role] = _identity(artifact_path)
        run_path = Path(capture["run"]["path"])
        run = json.loads(run_path.read_text(encoding="utf-8"))
        run["energy_calibration_sha256"] = method_identity["sha256"]
        run_path.write_bytes(_json_bytes(run))
        capture["run"] = _identity(run_path)
    evidence_path = Path(binding["calibration_evidence"]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["energy_calibration_manifest"] = method_binding
    evidence["captures"] = binding["captures"]
    evidence_path.write_bytes(_json_bytes(evidence))
    binding["calibration_evidence"] = _identity(evidence_path)
    _reseal_binding(binding, setup)
    return method_identity


def _refresh_phase_chain(
    binding: dict,
    setup: SimpleNamespace,
    phase: str,
) -> None:
    """Refresh the outer seals after an inner command-chain mutation."""

    capture = binding["captures"][phase]
    command_path = Path(capture["command_window_binding"]["path"])
    capture["command_window_binding"] = _identity(command_path)
    run_path = Path(capture["run"]["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["command_window_binding_manifest"] = str(command_path.resolve())
    run["command_window_trace_binding"] = {
        "status": "verified",
        "manifest_path": str(command_path.resolve()),
    }
    provenance = run["raw_input_energy_provenance"]
    provenance.update(
        {
            "trace_path": capture["raw"]["path"],
            "trace_sha256": capture["raw"]["sha256"],
            "result_path": capture["postprocessor_result"]["path"],
            "result_sha256": capture["postprocessor_result"]["sha256"],
        }
    )
    run_path.write_bytes(_json_bytes(run))
    capture["run"] = _identity(run_path)
    _reseal_evidence_and_binding(binding, setup)


def test_v2_binding_verifies_complete_local_evidence_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is True
    assert result["accelerator_idle_calibration_status"] == "verified"
    assert result["accelerator_idle_calibration_runtime_binding_id"] == SETUP_ID


def test_v2_binding_uses_explicit_custom_registry_for_current_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    custom_registry = {"custom_registry": True}
    calls: list[object] = []

    def verify_method(_setup_id: str, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs.get("registry"))
        return {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "path": setup.calibration_manifest,
            "sha256": setup.calibration_sha256,
            "configured_method_admission": {"ok": True},
            "source_integrity_verification": {
                "ok": True,
                "status": "verified",
            },
        }

    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_configured_energy_method",
        verify_method,
    )
    result = verify_accelerator_idle_calibration_binding(
        setup, registry=custom_registry
    )
    assert result["accelerator_idle_calibration_verified"] is True
    assert calls == [custom_registry]


def _custom_registry_for_calibrated_setup(setup: SimpleNamespace) -> dict:
    return {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "energy_defaults": {
            "enabled": True,
            "collector_binary": "urecs-data-collector",
            "power_calculations_binary": "power_calculations",
            "mode": "fast_firmware",
            "data_port": DATA_PORT,
            "channel": 0,
            "sample_rate": 2000,
            "physical_scope": "FS",
            "window_label": "command",
        },
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "accelerator": "hailo8",
                "host": dict(JETSON_HOST),
                "energy": {
                    "enabled": True,
                    "urecs_address": URECS_ADDRESS,
                    "accelerator_idle_w": setup.accelerator_idle_w,
                    "accelerator_idle_calibration_binding_path": (
                        setup.accelerator_idle_calibration_binding_path
                    ),
                    "accelerator_idle_calibration_binding_sha256": (
                        setup.accelerator_idle_calibration_binding_sha256
                    ),
                    "calibration_manifest": setup.calibration_manifest,
                    "calibration_sha256": setup.calibration_sha256,
                },
            }
        ],
    }


def test_tensorrt_full_uses_exact_custom_registry_before_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, calibrated = _fixture(tmp_path, monkeypatch)
    custom_path = tmp_path / "custom" / "hardware_setups.yaml"
    custom_path.parent.mkdir(parents=True)
    custom_path.write_text(
        yaml.safe_dump(
            _custom_registry_for_calibrated_setup(calibrated),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    default_path = tmp_path / "default" / "hardware_setups.yaml"
    default_path.parent.mkdir(parents=True)
    default_path.write_text(
        yaml.safe_dump(
            {
                "schema": "onnx-splitpoint/hardware-setups",
                "schema_version": 2,
                "hardware_setups": [],
            }
        ),
        encoding="utf-8",
    )
    registry = load_hardware_registry(custom_path)
    setup = energy_setup_from_registry(
        registry,
        SETUP_ID,
        registry_path=custom_path,
    )
    defaults = energy_defaults_from_registry(registry)
    loaded_paths: list[Path] = []
    real_load = collector.load_hardware_registry

    def load_exact(path: str | Path | None = None) -> dict:
        loaded_paths.append(Path(path or default_path).resolve())
        return real_load(path or default_path)

    tool_checks: list[bool] = []
    monkeypatch.setattr(collector, "load_hardware_registry", load_exact)
    monkeypatch.setattr(collector, "default_registry_path", lambda: default_path)
    monkeypatch.setattr(
        collector,
        "_verify_calibration_manifest",
        lambda *_args, **_kwargs: {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "evidence_mode": "inherited_validated_method",
            "actual_sha256": setup.calibration_sha256,
            "runtime_fail_closed_required": True,
        },
    )
    monkeypatch.setattr(
        collector,
        "check_energy_tools",
        lambda _defaults: (
            tool_checks.append(True)
            or {"collector_found": False, "power_calculations_found": False}
        ),
    )
    result = collector.run_fast_firmware_measurement(
        "printf workload-must-not-run",
        tmp_path / "claim",
        setup=setup,
        defaults=defaults,
        duration_s=1.0,
        run_count=1,
        setup_id=SETUP_ID,
        run_id="native_full_tensorrt",
        physical_scope="FS",
        window_label="command",
        host_normalization_role="tensorrt_full",
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert result["error"] == "urecs-data-collector not found"
    assert tool_checks == [True]
    assert loaded_paths == [custom_path.resolve()]
    assert default_path.resolve() not in loaded_paths


def test_tensorrt_full_changed_custom_registry_starts_no_tools_or_workload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, calibrated = _fixture(tmp_path, monkeypatch)
    custom_path = tmp_path / "custom" / "hardware_setups.yaml"
    custom_path.parent.mkdir(parents=True)
    original = _custom_registry_for_calibrated_setup(calibrated)
    custom_path.write_text(
        yaml.safe_dump(original, sort_keys=False), encoding="utf-8"
    )
    setup = energy_setup_from_registry(
        load_hardware_registry(custom_path),
        SETUP_ID,
        registry_path=custom_path,
    )
    changed = dict(original)
    changed["operator_note"] = "changed-after-admission"
    custom_path.write_text(
        yaml.safe_dump(changed, sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        collector,
        "check_energy_tools",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tools must not be inspected")
        ),
    )
    monkeypatch.setattr(
        collector,
        "run_duration_probe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("workload must not start")
        ),
    )
    result = collector.run_fast_firmware_measurement(
        "printf workload-must-not-run",
        tmp_path / "blocked",
        setup=setup,
        defaults=energy_defaults_from_registry(original),
        duration_s=1.0,
        run_count=1,
        setup_id=SETUP_ID,
        run_id="native_full_tensorrt",
        physical_scope="FS",
        window_label="command",
        host_normalization_role="tensorrt_full",
        host_normalization_source_run_id="native_full_tensorrt",
        host_normalization_target_variant="full",
    )
    assert result["status"] == "accelerator_idle_calibration_claim_blocked"
    assert result["collector_started"] is False
    assert result["workload_started"] is False
    assert result["transport_started"] is False
    assert "hardware_registry_snapshot_sha256_mismatch" in result[
        "hardware_registry_verification"
    ]["errors"]


def test_v2_binding_rejects_resealed_extra_top_level_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["unexpected_claim_field"] = "attacker-controlled"
    _reseal_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "binding_v2_fields_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_resealed_extra_capture_or_identity_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["captures"]["unexpected_phase"] = {}
    binding["captures"]["m2_off"]["raw"]["size"] = 123
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert result["accelerator_idle_calibration_verified"] is False
    assert "captures_fields_mismatch" in reasons
    assert "m2_off_raw_identity_fields_mismatch" in reasons


def test_v2_binding_rejects_resealed_float_evidence_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    evidence_path = Path(binding["calibration_evidence"]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["schema_version"] = 1.0
    evidence_path.write_bytes(_json_bytes(evidence))
    binding["calibration_evidence"] = _identity(evidence_path)
    _reseal_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "calibration_evidence_schema_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


@pytest.mark.parametrize("mutation", ["float_schema", "extra_field"])
def test_v2_binding_rejects_resealed_nonexact_command_window_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    identity = binding["captures"]["m2_off"]["command_window_binding"]
    command_path = Path(identity["path"])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    if mutation == "float_schema":
        command["schema_version"] = 2.0
    else:
        command["unexpected_claim_field"] = True
    command_path.write_bytes(_json_bytes(command))
    binding["captures"]["m2_off"]["command_window_binding"] = _identity(
        command_path
    )
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    expected = (
        "m2_off_command_window_binding_contract_mismatch"
        if mutation == "float_schema"
        else "m2_off_command_window_binding_fields_mismatch"
    )
    assert expected in reasons


@pytest.mark.parametrize(
    "artifact",
    [
        "request", "marker", "trace", "command", "workload_command",
        "timing", "postprocessor_result", "result",
    ],
)
@pytest.mark.parametrize("operation", ["delete", "tamper"])
def test_v2_binding_rejects_missing_or_tampered_command_supporting_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    operation: str,
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    binding = json.loads(
        Path(setup.accelerator_idle_calibration_binding_path).read_text(
            encoding="utf-8"
        )
    )
    command_identity = binding["captures"]["m2_off"][
        "command_window_binding"
    ]
    command = json.loads(
        Path(command_identity["path"]).read_text(encoding="utf-8")
    )
    supporting_path = Path(command[f"{artifact}_path"])
    if operation == "delete":
        supporting_path.unlink()
    else:
        supporting_path.write_bytes(supporting_path.read_bytes() + b"tampered")
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    error = "file_missing" if operation == "delete" else "sha256_mismatch"
    assert f"m2_off_command_window_{artifact}_{error}" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


@pytest.mark.parametrize(
    "artifact",
    [
        "request", "marker", "trace", "command", "workload_command",
        "timing", "postprocessor_result", "result",
    ],
)
def test_v2_binding_rejects_fully_resealed_inner_command_chain_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    capture = binding["captures"]["m2_off"]
    command_path = Path(capture["command_window_binding"]["path"])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    artifact_path = Path(command[f"{artifact}_path"])

    if artifact in {"request", "marker", "postprocessor_result"}:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["run_id"] = "resealed-but-internally-inconsistent"
        artifact_path.write_bytes(_json_bytes(payload))
    elif artifact == "result":
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        payload["firmware_results"]["energy"] += 1.0
        artifact_path.write_bytes(_json_bytes(payload))
    elif artifact == "trace":
        original = artifact_path.read_bytes()
        artifact_path.write_bytes(original[:-4] + b"tampered" + original[-4:])
    elif artifact == "timing":
        artifact_path.write_text(
            "start_ns=1000000000\nend_ns=1999999999\nrc=0\n",
            encoding="utf-8",
        )
    else:
        artifact_path.write_bytes(artifact_path.read_bytes() + b"# tampered\n")

    new_identity = _identity(artifact_path)
    command[f"{artifact}_sha256"] = new_identity["sha256"]
    # The fixture intentionally binds command and workload_command to one
    # script.  Keep both outer identities coherent so the replay, rather than
    # a superficial SHA check, proves the inner request is stale.
    if artifact in {"command", "workload_command"}:
        command["command_sha256"] = new_identity["sha256"]
        command["workload_command_sha256"] = new_identity["sha256"]
    if artifact == "trace":
        capture["raw"] = new_identity
    if artifact == "result":
        capture["postprocessor_result"] = new_identity
    command_path.write_bytes(_json_bytes(command))
    _refresh_phase_chain(binding, setup, "m2_off")

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert any(
        reason.startswith("m2_off_command_window_replay_failed:")
        for reason in result["accelerator_idle_calibration_failure_reasons"]
    )


@pytest.mark.parametrize(
    ("artifact", "expected_reason"),
    [
        ("request", "m2_off_command_window_request_fields_mismatch"),
        ("marker", "m2_off_command_window_marker_nested_fields_mismatch"),
        (
            "postprocessor_result",
            "m2_off_command_window_postprocessor_fields_mismatch",
        ),
    ],
)
def test_v2_binding_rejects_fully_resealed_supporting_shape_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
    expected_reason: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    capture = binding["captures"]["m2_off"]
    command_path = Path(capture["command_window_binding"]["path"])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    artifact_path = Path(command[f"{artifact}_path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact == "marker":
        payload["stream"]["contradictory_trace_path"] = "/attacker/trace"
    else:
        payload["unexpected_claim_field"] = "attacker-controlled"
    artifact_path.write_bytes(_json_bytes(payload))
    command[f"{artifact}_sha256"] = _identity(artifact_path)["sha256"]
    command_path.write_bytes(_json_bytes(command))
    _refresh_phase_chain(binding, setup, "m2_off")

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert expected_reason in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


@pytest.mark.parametrize(
    ("started_at", "finished_at", "expected_reason"),
    [
        (
            "bogus",
            "also-bogus",
            "started_at_not_canonical_utc_timestamp",
        ),
        (
            "2026-09-03T12:00:00+00:00",
            "2026-09-02T12:00:00+00:00",
            "calibration_timestamp_chronology_invalid",
        ),
    ],
)
def test_v2_binding_rejects_invalid_or_reversed_calibration_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    started_at: str,
    finished_at: str,
    expected_reason: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["started_at"] = started_at
    binding["finished_at"] = finished_at
    evidence_path = Path(binding["calibration_evidence"]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["started_at"] = started_at
    evidence["finished_at"] = finished_at
    evidence_path.write_bytes(_json_bytes(evidence))
    binding["calibration_evidence"] = _identity(evidence_path)
    _reseal_binding(binding, setup)

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert expected_reason in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_fake_raw_parquet_after_full_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    capture = binding["captures"]["m2_off"]
    raw_path = Path(capture["raw"]["path"])
    raw_path.write_bytes(b"not a parquet stream")
    capture["raw"] = _identity(raw_path)
    command_path = Path(capture["command_window_binding"]["path"])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    command["trace_sha256"] = capture["raw"]["sha256"]
    command_path.write_bytes(_json_bytes(command))
    _refresh_phase_chain(binding, setup, "m2_off")

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "m2_off_raw_not_nonempty_parquet" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_resealed_energy_to_power_inconsistency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    capture = binding["captures"]["m2_off"]
    command_path = Path(capture["command_window_binding"]["path"])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    result_path = Path(command["result_path"])
    selected = json.loads(result_path.read_text(encoding="utf-8"))
    selected["firmware_results"]["energy"] = 99_999.0
    result_path.write_bytes(_json_bytes(selected))
    result_identity = _identity(result_path)
    post_path = Path(command["postprocessor_result_path"])
    post = json.loads(post_path.read_text(encoding="utf-8"))
    post["energy_j"] = 99_999.0
    post_path.write_bytes(_json_bytes(post))
    command["energy_j"] = 99_999.0
    command["result_sha256"] = result_identity["sha256"]
    command["postprocessor_result_sha256"] = _identity(post_path)["sha256"]
    command_path.write_bytes(_json_bytes(command))
    capture["postprocessor_result"] = result_identity
    _refresh_phase_chain(binding, setup, "m2_off")

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "m2_off_command_window_energy_power_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_aggregate_schema_status_and_role_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    capture = binding["captures"]["m2_off"]
    aggregate_path = Path(capture["aggregate"]["path"])
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["schema_version"] = False
    aggregate["status"] = ["failed"]
    aggregate_path.write_bytes(_json_bytes(aggregate))
    capture["aggregate"] = _identity(aggregate_path)
    capture["aggregate"], capture["report"] = (
        capture["report"], capture["aggregate"]
    )
    _reseal_evidence_and_binding(binding, setup)

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "m2_off_aggregate_role_path_mismatch" in reasons
    assert "m2_off_report_role_path_mismatch" in reasons


def test_v2_binding_rejects_resealed_schema_less_aggregate_spoofing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    capture = binding["captures"]["m2_off"]
    for role in ("aggregate", "report"):
        path = Path(capture[role]["path"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = False
        payload["status"] = ["ok"]
        path.write_bytes(_json_bytes(payload))
        capture[role] = _identity(path)
    _reseal_evidence_and_binding(binding, setup)

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "m2_off_aggregate_status_mismatch" in reasons
    assert "m2_off_aggregate_unexpected_schema_fields" in reasons


def test_v2_binding_rejects_resealed_complete_phase_bundle_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["captures"]["m2_off"], binding["captures"]["m2_on"] = (
        binding["captures"]["m2_on"],
        binding["captures"]["m2_off"],
    )
    _reseal_evidence_and_binding(binding, setup)

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "m2_off_aggregate_role_path_mismatch" in reasons
    assert "m2_on_aggregate_role_path_mismatch" in reasons


def test_v2_binding_rejects_delta_below_fresh_registry_minimum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["idle_power_with_m2_w"] = 10.0
    binding["accelerator_idle_power_w"] = 0.0
    setup.accelerator_idle_w = 0.0
    evidence_path = Path(binding["calibration_evidence"]["path"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["idle_power_with_m2_w"] = 10.0
    evidence["accelerator_idle_power_w"] = 0.0
    evidence_path.write_bytes(_json_bytes(evidence))
    binding["calibration_evidence"] = _identity(evidence_path)
    _reseal_binding(binding, setup)
    registry = {
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "power_control": {
                    "require_positive_calibration_delta": True,
                    "minimum_calibration_delta_w": 0.02,
                },
            }
        ]
    }

    result = verify_accelerator_idle_calibration_binding(
        setup, registry=registry
    )
    assert result["accelerator_idle_calibration_verified"] is False
    assert "accelerator_idle_power_below_configured_minimum" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


@pytest.mark.parametrize("field", ["address", "user"])
def test_v2_binding_rejects_resealed_padded_bound_jetson_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["jetson_host"][field] += " "
    _reseal_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert f"jetson_host_{field}_not_canonical" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


@pytest.mark.parametrize("field", ["jetson_address", "jetson_user"])
def test_v2_binding_rejects_padded_current_jetson_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setattr(setup, field, getattr(setup, field) + " ")
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert f"current_{field}_not_canonical" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("setup_id", f" {SETUP_ID}", "current_setup_id_not_canonical"),
        ("setup_id", 123, "current_setup_id_not_canonical"),
        (
            "urecs_address",
            f"{URECS_ADDRESS} ",
            "current_urecs_address_not_canonical",
        ),
        ("urecs_address", 123, "current_urecs_address_not_canonical"),
    ],
)
def test_v2_binding_rejects_noncanonical_current_setup_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    reason: str,
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setattr(setup, field, value)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert reason in result["accelerator_idle_calibration_failure_reasons"]


@pytest.mark.parametrize("target", ["bound", "current"])
def test_v2_binding_rejects_lossy_accelerator_normalisation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    if target == "bound":
        binding["accelerator"] = "Hailo-8"
        _reseal_binding(binding, setup)
        reason = "accelerator_identity_not_canonical"
    else:
        setup.accelerator = "Hailo-8"
        reason = "current_accelerator_identity_not_canonical"
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert reason in result["accelerator_idle_calibration_failure_reasons"]


@pytest.mark.parametrize("location", ["binding", "evidence", "capture"])
def test_v2_binding_rejects_numeric_string_power_scalars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    if location == "binding":
        binding["idle_power_without_m2_w"] = "10.0"
        _reseal_binding(binding, setup)
    elif location == "evidence":
        evidence_path = Path(binding["calibration_evidence"]["path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["idle_power_without_m2_w"] = "10.0"
        evidence_path.write_bytes(_json_bytes(evidence))
        binding["calibration_evidence"] = _identity(evidence_path)
        _reseal_binding(binding, setup)
    else:
        binding["captures"]["m2_off"]["avg_power_w"] = "10.0"
        _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("binding", "started_at", 1),
        ("binding", "finished_at", 2),
        ("evidence", "started_at", 1),
        ("evidence", "finished_at", 2),
        ("evidence", "data_port", 3000.0),
    ],
)
def test_v2_binding_rejects_resealed_nonliteral_evidence_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    value: object,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    if target == "binding":
        binding[field] = value
        _reseal_binding(binding, setup)
    else:
        evidence_path = Path(binding["calibration_evidence"]["path"])
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence[field] = value
        evidence_path.write_bytes(_json_bytes(evidence))
        binding["calibration_evidence"] = _identity(evidence_path)
        _reseal_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("collector_rc", 0.0),
        ("collector_rc", False),
        ("workload_command_rc", 0.0),
    ],
)
def test_v2_binding_rejects_resealed_noninteger_run_return_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    run_identity = binding["captures"]["m2_off"]["run"]
    run_path = Path(run_identity["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run[field] = value
    run_path.write_bytes(_json_bytes(run))
    binding["captures"]["m2_off"]["run"] = _identity(run_path)
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False


@pytest.mark.parametrize("value", ["2.25", True, False])
def test_v2_binding_rejects_coerced_current_idle_power_scalar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setup.accelerator_idle_w = value
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert result["accelerator_idle_calibration_status"] == (
        "unavailable_invalid_accelerator_idle_w"
    )


def test_v2_binding_rejects_fully_rehashed_nonexact_current_method(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    method_path = Path(setup.calibration_manifest)
    method = json.loads(method_path.read_text(encoding="utf-8"))
    method["method"]["sample_rate_hz"] = 2000.0
    method_path.write_bytes(_json_bytes(method))
    method_identity = _reseal_complete_method_chain(binding, setup)

    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "verified": False,
            "status": "configured_energy_method_admission_failed",
            "path": method_identity["path"],
            "sha256": method_identity["sha256"],
            "configured_method_admission": {"ok": False},
            "source_integrity_verification": {
                "ok": True,
                "status": "verified",
            },
        },
    )
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "current_configured_energy_method_not_strictly_verified" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("sample_rate_hz", 2000.0, "command_window_sample_rate_mismatch"),
        ("process_rc", False, "command_window_process_rc_mismatch"),
        ("trace_covers_window", 1, "command_window_trace_coverage_unverified"),
        ("dropped_samples", False, "command_window_dropped_samples_not_zero"),
        ("energy_semantics", "raw_input_energy", "command_window_energy_semantics_mismatch"),
        ("source", "FastFirmware", "command_window_source_mismatch"),
        ("energy_field", "energy", "command_window_energy_field_mismatch"),
        ("energy_j", "1.0", "command_window_energy_not_exact_positive_number"),
        ("sample_count", 2.0, "command_window_sample_interval_mismatch"),
        ("first_sample_index", False, "command_window_sample_interval_mismatch"),
        ("duration_s", "30.0", "command_window_duration_not_exact_positive_number"),
        ("created_at_unix_ns", 1.0, "command_window_created_at_not_exact_positive_int"),
    ],
)
def test_v2_binding_rejects_resealed_invalid_command_window_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    reason: str,
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    identity = binding["captures"]["m2_off"]["command_window_binding"]
    command_path = Path(identity["path"])
    command = json.loads(command_path.read_text(encoding="utf-8"))
    command[field] = value
    command_path.write_bytes(_json_bytes(command))
    binding["captures"]["m2_off"]["command_window_binding"] = _identity(
        command_path
    )
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert f"m2_off_{reason}" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_current_energy_data_port_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setup.data_port = 3999
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "current_data_port_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_energy_setup_projects_accelerator_and_jetson_identity() -> None:
    setup = energy_setup_from_raw(
        {
            "id": SETUP_ID,
            "accelerator": "hailo8",
            "host": dict(JETSON_HOST),
            "energy": {"urecs_address": URECS_ADDRESS},
        }
    )
    assert setup.accelerator == "hailo8"
    assert setup.jetson_address == JETSON_HOST["address"]
    assert setup.jetson_user == JETSON_HOST["user"]
    assert setup.jetson_port == JETSON_HOST["port"]
    assert setup.jetson_ssh_extra_args == ""


def test_remote_only_legacy_host_projects_and_pairs_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, calibrated = _fixture(tmp_path, monkeypatch)
    setup = energy_setup_from_raw(
        {
            "id": SETUP_ID,
            "accelerator": "hailo8",
            "remote": {
                "host": JETSON_HOST["address"],
                "user": JETSON_HOST["user"],
                "port": JETSON_HOST["port"],
            },
            "energy": {
                "urecs_address": URECS_ADDRESS,
                "accelerator_idle_w": calibrated.accelerator_idle_w,
                "accelerator_idle_calibration_binding_path": (
                    calibrated.accelerator_idle_calibration_binding_path
                ),
                "accelerator_idle_calibration_binding_sha256": (
                    calibrated.accelerator_idle_calibration_binding_sha256
                ),
                "calibration_manifest": calibrated.calibration_manifest,
                "calibration_sha256": calibrated.calibration_sha256,
            },
        }
    )
    result = verify_accelerator_idle_calibration_binding(setup)
    assert setup.jetson_address == JETSON_HOST["address"]
    assert setup.jetson_user == JETSON_HOST["user"]
    assert setup.jetson_port == JETSON_HOST["port"]
    assert setup.jetson_ssh_extra_args == ""
    assert result["accelerator_idle_calibration_verified"] is True


@pytest.mark.parametrize(
    "identity_path",
    [
        ("energy_calibration_manifest",),
        ("calibration_evidence",),
        ("captures", "m2_off", "aggregate"),
        ("captures", "m2_on", "report"),
        ("captures", "m2_off", "raw"),
        ("captures", "m2_on", "run"),
        ("captures", "m2_off", "command_window_binding"),
        ("captures", "m2_on", "postprocessor_result"),
    ],
)
def test_v2_binding_rejects_every_tampered_referenced_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_path: tuple[str, ...],
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    identity: object = binding
    for key in identity_path:
        identity = identity[key]  # type: ignore[index]
    path = Path(identity["path"])  # type: ignore[index]
    path.write_bytes(path.read_bytes() + b"tampered")
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert result["accelerator_idle_calibration_status"] == (
        "unavailable_binding_v2_validation_failed"
    )
    assert any(
        "sha256_mismatch" in reason
        for reason in result["accelerator_idle_calibration_failure_reasons"]
    )


def test_v2_binding_rejects_unsealed_payload_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["finished_at"] = "later"
    path = Path(setup.accelerator_idle_calibration_binding_path)
    path.write_bytes(_json_bytes(binding))
    setup.accelerator_idle_calibration_binding_sha256 = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    result = verify_accelerator_idle_calibration_binding(setup)
    assert "binding_payload_sha256_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_scalar_or_setup_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["accelerator_idle_power_w"] = 3.0
    _reseal_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert "accelerator_idle_power_delta_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]
    assert "registry_accelerator_idle_power_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]

    binding, setup = _fixture(tmp_path / "second", monkeypatch)
    binding["setup_id"] = "orin_nx_hailo10_01"
    _reseal_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert "setup_id_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_cross_capture_raw_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["captures"]["m2_off"]["raw"], binding["captures"]["m2_on"]["raw"] = (
        binding["captures"]["m2_on"]["raw"],
        binding["captures"]["m2_off"]["raw"],
    )
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "m2_off_run_raw_identity_mismatch" in reasons
    assert "m2_on_run_raw_identity_mismatch" in reasons


def test_v2_binding_rejects_resealed_run_provenance_downgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    run_identity = binding["captures"]["m2_off"]["run"]
    run_path = Path(run_identity["path"])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["energy_calibration_verification"]["verified"] = False
    run["raw_input_energy_provenance"]["binding_status"] = "unverified"
    run_path.write_bytes(_json_bytes(run))
    binding["captures"]["m2_off"]["run"] = _identity(run_path)
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "m2_off_run_method_verification_mismatch" in reasons
    assert "m2_off_run_raw_provenance_unverified" in reasons


def test_v2_binding_rejects_missing_physical_state_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    del binding["state_observations"]["m2_off"]["post_measurement"]
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "state_m2_off_observation_fields_mismatch" in reasons
    assert "state_m2_off_post_measurement_missing" in reasons


def test_v2_binding_rejects_tampered_physical_m2_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    binding["state_observations"]["m2_off"]["post_measurement"][
        "m2_present"
    ] = True
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    assert "state_m2_off_post_measurement_m2_state_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_transition_target_cross_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding, setup = _fixture(tmp_path, monkeypatch)
    transition = binding["transitions"]["m2_off"]
    transition["desired_accelerator_present"] = True
    transition["after"] = _observation(True)
    _reseal_evidence_and_binding(binding, setup)
    result = verify_accelerator_idle_calibration_binding(setup)
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "transition_m2_off_target_mismatch" in reasons
    assert "transition_m2_off_after_m2_state_mismatch" in reasons
    assert "transition_m2_off_after_pre_state_mismatch" in reasons


def test_v2_binding_rejects_distinct_current_energy_method_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    second_method = _write_json(
        tmp_path / "method-v2" / "energy_calibration_manifest.json",
        {
            "schema": "onnx-splitpoint/energy-calibration-manifest",
            "schema_version": 2,
            "evidence_mode": "inherited_validated_method",
            "channel_bindings": [
                {
                    "setup_id": SETUP_ID,
                    "urecs_address": URECS_ADDRESS,
                    "channel": 0,
                    "sample_rate_hz": 2000,
                    "scope": "FS",
                }
            ],
            "method_revision": "M2-distinct-current-method",
        },
    )
    second_identity = _identity(second_method)
    setup.calibration_manifest = second_identity["path"]
    setup.calibration_sha256 = second_identity["sha256"]

    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    reasons = result["accelerator_idle_calibration_failure_reasons"]
    assert "current_energy_method_path_mismatch" in reasons
    assert "current_energy_method_sha256_mismatch" in reasons


def test_v2_binding_requires_current_energy_method_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setup.calibration_manifest = ""
    setup.calibration_sha256 = ""
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "current_energy_method_path_missing" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_accepts_prefixed_current_method_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setup.calibration_sha256 = "sha256:" + setup.calibration_sha256
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is True
    assert result["accelerator_idle_calibration_status"] == "verified"


def test_v2_binding_rejects_current_accelerator_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setup.accelerator = "deepx_m1"
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "current_accelerator_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_current_jetson_host_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setup.jetson_address = "192.168.0.102"
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "current_jetson_address_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_v2_binding_rejects_current_jetson_ssh_extra_args_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _binding, setup = _fixture(tmp_path, monkeypatch)
    setup.jetson_ssh_extra_args = "-o HostName=other.example"
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert "current_jetson_ssh_extra_args_mismatch" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_energy_setup_normalises_effective_ssh_extra_args() -> None:
    setup = energy_setup_from_raw(
        {
            "id": SETUP_ID,
            "accelerator": "hailo8",
            "remote": {
                "host": JETSON_HOST["address"],
                "user": JETSON_HOST["user"],
                "port": JETSON_HOST["port"],
                "ssh_extra_args": (
                    "  -o   HostName=calibrated.example  "
                    "-o 'ProxyCommand=ssh jump nc %h %p' "
                ),
            },
        }
    )
    assert setup.jetson_ssh_extra_args == (
        "-o HostName=calibrated.example -o "
        "'ProxyCommand=ssh jump nc %h %p'"
    )


@pytest.mark.parametrize(
    ("host", "expected_reason"),
    [
        (
            {"address": "", "user": "nx", "port": 22},
            "current_jetson_address_not_string_or_blank",
        ),
        (
            {"address": JETSON_HOST["address"], "user": "", "port": 22},
            "current_jetson_user_not_string_or_blank",
        ),
        (
            {"address": JETSON_HOST["address"], "user": "nx", "port": 0},
            "current_jetson_port_out_of_range",
        ),
        (
            {"address": JETSON_HOST["address"], "user": "nx", "port": "22"},
            "current_jetson_port_not_integer",
        ),
        (
            {
                "address": JETSON_HOST["address"],
                "user": "nx",
                "port": 22,
                "ssh_extra_args": ["-o", "HostName=other"],
            },
            "current_jetson_ssh_extra_args_not_string",
        ),
    ],
)
def test_malformed_current_jetson_identity_cannot_verify_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: dict[str, object],
    expected_reason: str,
) -> None:
    _binding, calibrated = _fixture(tmp_path, monkeypatch)
    setup = energy_setup_from_raw(
        {
            "id": SETUP_ID,
            "accelerator": "hailo8",
            "host": host,
            "energy": {
                "urecs_address": URECS_ADDRESS,
                "accelerator_idle_w": calibrated.accelerator_idle_w,
                "accelerator_idle_calibration_binding_path": (
                    calibrated.accelerator_idle_calibration_binding_path
                ),
                "accelerator_idle_calibration_binding_sha256": (
                    calibrated.accelerator_idle_calibration_binding_sha256
                ),
                "calibration_manifest": calibrated.calibration_manifest,
                "calibration_sha256": calibrated.calibration_sha256,
            },
        }
    )
    result = verify_accelerator_idle_calibration_binding(setup)
    assert setup.jetson_identity_valid is False
    assert result["accelerator_idle_calibration_verified"] is False
    assert "current_jetson_identity_invalid" in result[
        "accelerator_idle_calibration_failure_reasons"
    ]
    assert expected_reason in result[
        "accelerator_idle_calibration_failure_reasons"
    ]


def test_energy_setup_registry_rejects_duplicate_selected_id() -> None:
    row = {
        "id": SETUP_ID,
        "host": dict(JETSON_HOST),
        "energy": {"urecs_address": URECS_ADDRESS},
    }
    with pytest.raises(DuplicateEnergySetupIdError, match="duplicate hardware"):
        energy_setup_from_registry(
            {"hardware_setups": [row, dict(row)]},
            SETUP_ID,
        )


def test_energy_setup_registry_projects_exact_global_data_port() -> None:
    setup = energy_setup_from_registry(
        {
            "energy_defaults": {"data_port": 3999},
            "hardware_setups": [
                {
                    "id": SETUP_ID,
                    "accelerator": "hailo8",
                    "host": dict(JETSON_HOST),
                    "energy": {"urecs_address": URECS_ADDRESS},
                }
            ],
        },
        SETUP_ID,
    )
    assert setup.data_port == 3999
    assert setup.data_port_valid is True


def test_v1_binding_is_explicitly_diagnostic_only(
    tmp_path: Path,
) -> None:
    path = _write_json(
        tmp_path / "accelerator_idle_calibration_binding.json",
        {
            "schema": CALIBRATION_BINDING_SCHEMA,
            "schema_version": 1,
            "setup_id": SETUP_ID,
            "status": "ok",
            "restored_m2_on": True,
            "accelerator_idle_power_w": 2.25,
        },
    )
    setup = SimpleNamespace(
        setup_id=SETUP_ID,
        accelerator_idle_w=2.25,
        accelerator_idle_calibration_binding_path=str(path),
        accelerator_idle_calibration_binding_sha256=hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
    )
    result = verify_accelerator_idle_calibration_binding(setup)
    assert result["accelerator_idle_calibration_verified"] is False
    assert result["accelerator_idle_calibration_status"] == (
        "unavailable_binding_legacy_v1_untrusted"
    )
