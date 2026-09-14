from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool import platform_power as pp


SETUP_ID = "orin_nx_hailo8_01"


def _registry(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema": "onnx-splitpoint/hardware-setups",
                "schema_version": 2,
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
                            "accelerator_idle_w": 0.75,
                        },
                        "power_control": {"enabled": True},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _method(tmp_path: Path) -> dict[str, object]:
    manifest = tmp_path / "method.json"
    manifest.write_text('{"test":"verified"}\n', encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    expected_bindings = [
        {
            "setup_id": SETUP_ID,
            "urecs_address": "192.0.2.10",
            "data_port": 3000,
            "channel": 0,
            "sample_rate_hz": 2000,
            "scope": "FS",
            "measurement_point": "complete_system_input",
        }
    ]
    return {
        "path": str(manifest.resolve()),
        "sha256": digest,
        "declared_sha256": digest,
        "verification_status": "inherited_validated_method_verified",
        "runtime_binding_id": SETUP_ID,
        "verification": {
            "verified": True,
            "status": "inherited_validated_method_verified",
            "runtime_binding_id": SETUP_ID,
        },
        "expected_channel_bindings": expected_bindings,
        "verified": True,
    }


def test_missing_energy_method_fails_before_any_platform_probe_or_toggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path / "hardware_setups.yaml")

    @contextmanager
    def unlocked(*_args, **_kwargs):
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("platform probe ran before method gate")
        ),
    )
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hardware toggle ran before method gate")
        ),
    )
    output = tmp_path / "blocked"
    with pytest.raises(
        pp.PlatformConfigurationError,
        match="energy calibration method is not configured",
    ):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            output_dir=output,
            stabilize_s=0,
            measure_s=5,
        )
    evidence = json.loads(
        (output / "m2_idle_power_calibration.json").read_text(encoding="utf-8")
    )
    assert evidence["status"] == "failed"
    assert evidence["events"] == []
    assert evidence["recovery_skipped_reason"] == "preflight_not_completed"


def test_source_integrity_mismatch_fails_before_any_ssh_or_udp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path / "hardware_setups.yaml")
    method_path = tmp_path / "method.json"
    method_path.write_text("{}\n", encoding="utf-8")
    method_sha = hashlib.sha256(method_path.read_bytes()).hexdigest()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["hardware_setups"][0]["energy"].update(
        {
            "calibration_manifest": str(method_path.resolve()),
            "calibration_sha256": method_sha,
        }
    )
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    source_verification = {
        "ok": False,
        "status": "source_integrity_binding_failed",
        "errors": ["current_installed_source_integrity_failed"],
    }
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_configured_energy_method",
        lambda *_args, **_kwargs: {
            "configured": True,
            "verified": False,
            "status": "configured_energy_method_admission_failed",
            "path": str(method_path.resolve()),
            "declared_sha256": method_sha,
            "sha256": method_sha,
            "verification": {
                "configured_method_admission": {
                    "source_integrity_binding_verification": (
                        source_verification
                    )
                }
            },
            "source_integrity_verification": source_verification,
        },
    )

    @contextmanager
    def unlocked(*_args, **_kwargs):
        yield

    calls: list[str] = []
    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: calls.append("ssh_probe"),
    )
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_args, **_kwargs: calls.append("m2_toggle"),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_args, **_kwargs: calls.append("udp"),
    )

    output = tmp_path / "source-integrity-blocked"
    with pytest.raises(
        pp.PlatformConfigurationError,
        match="Installed release source integrity is not bound and verified",
    ):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            output_dir=output,
            stabilize_s=0,
            measure_s=5,
        )

    assert calls == []
    evidence = json.loads(
        (output / "m2_idle_power_calibration.json").read_text(encoding="utf-8")
    )
    assert evidence["events"] == []
    assert evidence["recovery_skipped_reason"] == "preflight_not_completed"


@pytest.mark.parametrize("field", ["configured", "verified"])
@pytest.mark.parametrize("truthy_non_bool", ["false", 1, {"value": True}])
def test_truthy_nonboolean_method_gate_never_authorizes_platform_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    truthy_non_bool: object,
) -> None:
    registry_path = _registry(tmp_path / "hardware_setups.yaml")
    method_path = tmp_path / "method.json"
    method_path.write_text("{}\n", encoding="utf-8")
    method_sha = hashlib.sha256(method_path.read_bytes()).hexdigest()
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["hardware_setups"][0]["energy"].update(
        {
            "calibration_manifest": str(method_path.resolve()),
            "calibration_sha256": method_sha,
        }
    )
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    method_result: dict[str, object] = {
        "configured": True,
        "verified": True,
        "status": "inherited_validated_method_verified",
        "path": str(method_path.resolve()),
        "sha256": method_sha,
        "declared_sha256": method_sha,
        "runtime_binding_id": "binding-present",
        "source_integrity_verification": {
            "ok": True,
            "status": "verified",
            "errors": [],
        },
    }
    method_result[field] = truthy_non_bool
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_configured_energy_method",
        lambda *_args, **_kwargs: method_result,
    )

    @contextmanager
    def unlocked(*_args, **_kwargs):
        yield

    calls: list[str] = []
    monkeypatch.setattr(pp, "platform_operation_lock", unlocked)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_a, **_k: calls.append("probe"),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_a, **_k: calls.append("udp"),
    )
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_a, **_k: calls.append("m2"),
    )

    with pytest.raises(pp.PlatformConfigurationError):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            output_dir=tmp_path / f"blocked-{field}",
            stabilize_s=0,
            measure_s=5,
        )

    assert calls == []


def test_configuration_change_while_waiting_for_lock_fails_before_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path / "hardware_setups.yaml")

    @contextmanager
    def change_while_waiting(*_args, **_kwargs):
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        payload["hardware_setups"][0]["host"]["address"] = "203.0.113.77"
        registry_path.write_text(
            yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
        )
        yield

    monkeypatch.setattr(pp, "platform_operation_lock", change_while_waiting)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hardware probe ran after configuration changed")
        ),
    )
    output = tmp_path / "lock-race"
    with pytest.raises(
        pp.PlatformStateError,
        match="configuration_changed_while_waiting_for_platform_lock",
    ):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            output_dir=output,
            stabilize_s=0,
            measure_s=5,
        )
    evidence = json.loads(
        (output / "m2_idle_power_calibration.json").read_text(encoding="utf-8")
    )
    assert evidence["events"] == []
    assert evidence["recovery_skipped_reason"] == "preflight_not_completed"


@pytest.mark.parametrize("source", ["host", "remote"])
def test_ssh_extra_args_change_while_waiting_for_lock_fails_before_hardware(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    registry_path = _registry(tmp_path / "hardware_setups.yaml")
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    setup = payload["hardware_setups"][0]
    setup.setdefault(source, {})["ssh_extra_args"] = "-o BatchMode=yes"
    registry_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )

    @contextmanager
    def change_while_waiting(*_args, **_kwargs):
        current = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        current["hardware_setups"][0][source]["ssh_extra_args"] = (
            "-o BatchMode=yes -o ConnectTimeout=9"
        )
        registry_path.write_text(
            yaml.safe_dump(current, sort_keys=False), encoding="utf-8"
        )
        yield

    calls: list[str] = []
    monkeypatch.setattr(pp, "platform_operation_lock", change_while_waiting)
    monkeypatch.setattr(
        pp,
        "probe_platform_status",
        lambda *_args, **_kwargs: calls.append("probe"),
    )
    monkeypatch.setattr(
        pp,
        "_set_m2_state_locked",
        lambda *_args, **_kwargs: calls.append("shutdown"),
    )
    monkeypatch.setattr(
        pp,
        "send_udp_command",
        lambda *_args, **_kwargs: calls.append("udp"),
    )

    with pytest.raises(
        pp.PlatformStateError,
        match="configuration_changed_while_waiting_for_platform_lock",
    ):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            output_dir=tmp_path / f"lock-race-{source}",
            stabilize_s=0,
            measure_s=5,
        )
    assert calls == []


def test_preexisting_calibration_output_collision_is_rejected(
    tmp_path: Path,
) -> None:
    registry_path = _registry(tmp_path / "hardware_setups.yaml")
    output = tmp_path / "collision"
    (output / "m2_off").mkdir(parents=True)
    (output / "m2_off" / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(
        pp.PlatformConfigurationError,
        match="output root is not empty",
    ):
        pp.calibrate_m2_accelerator_idle_power(
            SETUP_ID,
            registry_path=registry_path,
            output_dir=output,
            stabilize_s=0,
            measure_s=5,
        )
    assert (output / "m2_off" / "sentinel").read_text(encoding="utf-8") == "keep"
    assert not (output / "m2_idle_power_calibration.json").exists()


def test_idle_measurement_passes_method_and_surfaces_exact_gate_reason(
    tmp_path: Path,
) -> None:
    method = _method(tmp_path)
    captured: dict[str, object] = {}

    def runner(_command, _output_dir, **kwargs):
        captured.update(kwargs)
        return {
            "ok": False,
            "status": "final_energy_gate_failed",
            "avg_power_w": None,
            "full_system_scope_calibration_status": "missing",
            "final_energy_gate_failures": [
                {
                    "run_index": 0,
                    "reasons": [
                        "full_system_calibration_not_locally_verified"
                    ],
                }
            ],
        }

    output = tmp_path / "measurement"
    registry = {
        "energy_defaults": {
            "data_port": 3000,
            "channel": 0,
            "sample_rate": 2000,
            "physical_scope": "FS",
            "window_label": "command",
        },
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "accelerator": "hailo8",
                "energy": {
                    "enabled": True,
                    "urecs_address": "192.0.2.10",
                    "calibration_manifest": method["path"],
                    "calibration_sha256": method["sha256"],
                },
            }
        ],
    }
    with pytest.raises(pp.PlatformPowerError) as failure:
        pp.measure_idle_power(
            SETUP_ID,
            registry=registry,
            setup={"id": SETUP_ID, "accelerator": "hailo8"},
            cfg={"energy": {"urecs_address": "192.0.2.10"}},
            state_label="m2_off",
            duration_s=5,
            output_dir=output,
            measurement_runner=runner,
            energy_method=method,
        )
    message = str(failure.value)
    assert "full_system_calibration_not_locally_verified" in message
    assert "calibration_status=missing" in message
    assert f"evidence_dir={output.resolve()}" in message
    assert captured["calibration_manifest"] == method["path"]
    assert captured["calibration_sha256"] == method["sha256"]
    assert captured["physical_scope"] == "FS"
    captured_setup = captured["setup"]
    assert captured_setup.expected_channel_bindings_valid is True
    assert list(captured_setup.expected_channel_bindings) == method[
        "expected_channel_bindings"
    ]


def test_capture_identity_binds_aggregate_report_raw_and_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "m2_off"
    run_dir = root / "run_000"
    storage = run_dir / "collector_storage"
    storage.mkdir(parents=True)
    aggregate = root / "energy_aggregate.json"
    report = root / "energy_summary.json"
    run_report = run_dir / "energy_summary.json"
    raw = storage / "raw.parquet"
    aggregate.write_text('{"ok":true}\n', encoding="utf-8")
    report.write_text('{"ok":true}\n', encoding="utf-8")
    run_report.write_text('{"status":"collector_finished"}\n', encoding="utf-8")
    raw.write_bytes(b"PAR1-real-raw-trace")
    raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    capture = pp._capture_evidence_identity(
        "m2_off",
        output_dir=root,
        avg_power_w=13.25,
        measurement={
            "runs": [
                {
                    "run_index": 0,
                    "raw_input_energy_provenance_status": (
                        "verified_calibrated_input_energy_unsubtracted"
                    ),
                    "raw_input_energy_provenance": {
                        "trace_path": str(raw.resolve()),
                        "trace_sha256": raw_sha,
                        "binding_status": "verified",
                    },
                }
            ]
        },
    )
    assert capture["avg_power_w"] == pytest.approx(13.25)
    assert capture["aggregate"]["path"] == str(aggregate.resolve())
    assert capture["report"]["path"] == str(report.resolve())
    assert capture["run"]["path"] == str(run_report.resolve())
    assert capture["raw"] == {"path": str(raw.resolve()), "sha256": raw_sha}


def test_measurement_failure_detail_flattens_run_gate_reasons() -> None:
    detail = pp._measurement_failure_detail(
        {
            "status": "final_energy_gate_failed",
            "runs": [
                {
                    "final_energy_gate_reasons": [
                        "raw_input_energy_provenance_unverified",
                        "postprocess_failed_or_energy_missing",
                    ]
                }
            ],
        }
    )
    assert "raw_input_energy_provenance_unverified" in detail
    assert "postprocess_failed_or_energy_missing" in detail


@pytest.mark.parametrize("truthy_lookalike", ["false", "yes", 1])
def test_measurement_power_requires_literal_true_ok(
    truthy_lookalike: object,
) -> None:
    with pytest.raises(pp.PlatformPowerError, match="u.RECS measurement failed"):
        pp._measurement_power(
            {
                "ok": truthy_lookalike,
                "status": "collector_finished",
                "avg_power_w": 13.25,
            }
        )
