from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.energy import collector
from onnx_splitpoint_tool.campaign import create_energy_calibration_manifest
from onnx_splitpoint_tool.energy.config import (
    EnergyDefaults,
    EnergySetup,
    energy_defaults_from_registry,
    energy_setup_from_raw,
    energy_setup_from_registry,
)
from onnx_splitpoint_tool.energy.method_manifest import (
    EXACT_IMPLEMENTATION_REUSE_POLICY,
    _canonical_implementation_artifacts,
)
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.source_integrity import create_source_integrity_binding
from onnx_splitpoint_tool.workflow.artifacts import sha256_json


SETUP_ID = "orin_nx_hailo8_01"
SECOND_SETUP_ID = "orin_nx_hailo10_01"


def _source_integrity_report() -> dict[str, Any]:
    return {
        "ok": True,
        "status": "verified",
        "package_version": VERSION,
        "build_id": BUILD_ID,
        "manifest_path": "/opt/onnx-splitpoint-tool/SOURCE_MANIFEST.json",
        "manifest_sha256": "a" * 64,
        "sha256sums_path": "/opt/onnx-splitpoint-tool/SHA256SUMS.txt",
        "sha256sums_sha256": "b" * 64,
    }


@pytest.fixture(autouse=True)
def _verified_installed_source_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_installed_source_integrity",
        lambda: dict(_source_integrity_report()),
    )


def _verified_inherited_method(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "verified": True,
        "status": "inherited_validated_method_verified",
        "evidence_mode": "inherited_validated_method",
        "runtime_fail_closed_required": True,
        "actual_sha256": "a" * 64,
    }


def _run_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    setup: EnergySetup,
    defaults: EnergyDefaults,
    call_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started: list[str] = []

    def _forbidden(name: str):
        def _call(*_args: Any, **_kwargs: Any) -> Any:
            started.append(name)
            raise AssertionError(f"{name} must not start")

        return _call

    monkeypatch.setattr(
        collector, "_verify_calibration_manifest", _verified_inherited_method
    )
    monkeypatch.setattr(collector, "check_energy_tools", _forbidden("tools"))
    monkeypatch.setattr(collector, "run_duration_probe", _forbidden("workload"))
    monkeypatch.setattr(collector, "_run_one", _forbidden("collector_or_transport"))

    out_dir = tmp_path / "blocked"
    call = {
        "setup": setup,
        "defaults": defaults,
        "duration_s": 1.0,
        "run_count": 1,
        "setup_id": SETUP_ID,
        "physical_scope": "FS",
        "calibration_manifest": "/synthetic/inherited-method.json",
        "calibration_sha256": "a" * 64,
    }
    call.update(call_overrides or {})
    result = collector.run_fast_firmware_measurement(
        "echo must-not-run",
        out_dir,
        **call,
    )

    assert result["status"] == "energy_input_registry_contract_blocked"
    assert result["collector_started"] is False
    assert result["workload_started"] is False
    assert result["transport_started"] is False
    assert started == []
    assert not (out_dir / "probe").exists()
    assert list(out_dir.glob("run_*")) == []
    assert json.loads((out_dir / "energy_summary.json").read_text()) == result
    return result


def _valid_setup(**overrides: Any) -> EnergySetup:
    manifest = "/synthetic/inherited-method.json"
    values: dict[str, Any] = {
        "setup_id": SETUP_ID,
        "enabled": True,
        "urecs_address": "192.168.0.197",
        "data_port": 3000,
        "data_port_valid": True,
        "calibration_manifest": manifest,
        "calibration_sha256": "a" * 64,
        "expected_channel_bindings": (
            {
                "setup_id": SETUP_ID,
                "urecs_address": "192.168.0.197",
                "data_port": 3000,
                "channel": 0,
                "sample_rate_hz": 2000,
                "scope": "FS",
                "measurement_point": "complete_system_input",
            },
        ),
        "expected_channel_bindings_valid": True,
    }
    values.update(overrides)
    return EnergySetup(**values)


@pytest.mark.parametrize("value", [0, 1, -0.25, 2.5])
def test_raw_accelerator_idle_power_accepts_only_finite_numeric_values(
    value: int | float,
) -> None:
    setup = energy_setup_from_raw(
        {
            "id": SETUP_ID,
            "energy": {
                "enabled": True,
                "urecs_address": "192.168.0.197",
                "accelerator_idle_w": value,
            },
        }
    )
    assert setup.accelerator_idle_w == pytest.approx(float(value))
    assert "setup_accelerator_idle_w_missing_or_invalid" not in (
        setup.registry_contract_errors
    )


@pytest.mark.parametrize(
    "value", ["1.25", "", True, False, float("nan"), float("inf"), -float("inf")]
)
def test_raw_accelerator_idle_power_rejects_coercion_and_nonfinite_values(
    value: object,
) -> None:
    setup = energy_setup_from_raw(
        {
            "id": SETUP_ID,
            "energy": {
                "enabled": True,
                "urecs_address": "192.168.0.197",
                "accelerator_idle_w": value,
            },
        }
    )
    assert setup.accelerator_idle_w is None
    assert setup.registry_contract_valid is False
    assert "setup_accelerator_idle_w_missing_or_invalid" in (
        setup.registry_contract_errors
    )


def _inherited_manifest(
    tmp_path: Path,
    bindings: list[dict[str, Any]],
) -> tuple[Path, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    collector_binary = bin_dir / "urecs-data-collector"
    power_binary = bin_dir / "power_calculations"
    for binary in (collector_binary, power_binary):
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
    artifact_defaults = EnergyDefaults(
        collector_binary=str(collector_binary.resolve()),
        power_calculations_binary=str(power_binary.resolve()),
    )
    implementation_artifacts = _canonical_implementation_artifacts(
        artifact_defaults
    )
    spec = {
        "evidence_mode": "inherited_validated_method",
        "channel_id": "urecs_fs_input_channel_0",
        "scope": "FS",
        "measurement_point": "complete_system_input",
        "sample_rate_hz": 2000,
        "locked": True,
        "method": {
            "collector": "urecs-data-collector",
            "collector_mode": "fast_firmware",
            "postprocessor": "power_calculations",
            "sample_rate_hz": 2000,
            "data_port": 3000,
            "output_semantics": "calibrated_input_energy_unsubtracted",
            "implementation_policy": EXACT_IMPLEMENTATION_REUSE_POLICY,
        },
        "validation_reference": {
            "reference_id": "wachsmuth2026masterarbeit",
            "author": "Wachsmuth, Joris",
            "title": (
                "Entwicklung und Validierung eines Messsystems zur "
                "energetischen Bewertung eingebetteter KI-Beschleuniger"
            ),
            "institution": "Bielefeld University",
            "work_type": "Master's thesis",
            "internal_identifier": "M99",
            "year": 2026,
        },
        "reuse_attestation": {
            "validated_method_accepted": True,
            "exact_implementation_reused": True,
            "new_calibration_required": False,
            "attested_by": "Kevin Mika",
            "attested_at": "2026-09-02T00:00:00+00:00",
        },
        "source_release_integrity": create_source_integrity_binding(
            _source_integrity_report()
        ),
        "channel_bindings": bindings,
        "artifacts": [
            {
                "id": artifact_id,
                "kind": "measurement_implementation",
                "path": str(path),
            }
            for artifact_id, path in implementation_artifacts
        ],
    }
    spec_path = tmp_path / "energy_method_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    manifest = create_energy_calibration_manifest(
        spec=spec_path,
        output=tmp_path / "inherited_energy_method.json",
    )
    # Historical manifest construction did not persist the UDP port.  v2.79.12
    # binds it explicitly in both the method and every setup row.
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["method"]["data_port"] = 3000
    for row in payload["channel_bindings"]:
        row["data_port"] = 3000
    payload["channel_binding_set_sha256"] = sha256_json(
        payload["channel_bindings"]
    )
    payload["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _binding(
    setup_id: str,
    address: str,
) -> dict[str, Any]:
    return {
        "setup_id": setup_id,
        "urecs_address": address,
        "data_port": 3000,
        "channel": 0,
        "sample_rate_hz": 2000,
        "scope": "FS",
        "measurement_point": "complete_system_input",
    }


def _method_bound_registry(
    manifest: Path,
    digest: str,
    *,
    selected_energy: dict[str, Any] | None = None,
    second_energy: dict[str, Any] | None = None,
    defaults_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = {
        "data_port": 3000,
        "channel": 0,
        "sample_rate": 2000,
        "collector_binary": str(
            (manifest.parent / "bin" / "urecs-data-collector").resolve()
        ) if (manifest.parent / "bin" / "urecs-data-collector").is_file()
        else "urecs-data-collector",
        "power_calculations_binary": str(
            (manifest.parent / "bin" / "power_calculations").resolve()
        ) if (manifest.parent / "bin" / "power_calculations").is_file()
        else "power_calculations",
        "mode": "fast_firmware",
        "physical_scope": "FS",
        "window_label": "command",
    }
    defaults.update(defaults_override or {})

    def energy(address: str, extra: dict[str, Any] | None) -> dict[str, Any]:
        row = {
            "enabled": True,
            "urecs_address": address,
            "calibration_manifest": str(manifest.resolve()),
            "calibration_sha256": digest,
            "unrelated_custom_key": {"preserved": True},
        }
        row.update(extra or {})
        return row

    return {
        "energy_defaults": defaults,
        "hardware_setups": [
            {
                "id": SETUP_ID,
                "accelerator": "hailo8",
                "energy": energy("192.168.0.197", selected_energy),
            },
            {
                "id": SECOND_SETUP_ID,
                "accelerator": "hailo10",
                "energy": energy("192.168.0.176", second_energy),
            },
        ],
    }


@pytest.mark.parametrize(
    ("data_port", "data_port_valid", "reason"),
    [
        ("3000", True, "setup_data_port_missing_or_invalid"),
        (True, True, "setup_data_port_missing_or_invalid"),
        (0, True, "setup_data_port_missing_or_invalid"),
        (65536, True, "setup_data_port_missing_or_invalid"),
        (3000, False, "setup_data_port_not_validated"),
        (3001, True, "setup_data_port_defaults_mismatch"),
    ],
)
def test_inherited_collection_rejects_setup_data_port_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_port: Any,
    data_port_valid: bool,
    reason: str,
) -> None:
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(
            data_port=data_port,
            data_port_valid=data_port_valid,
        ),
        defaults=EnergyDefaults(
            data_port=3000,
            channel=0,
            sample_rate=2000,
            physical_scope="FS",
            window_label="command",
        ),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("data_port", "3000", "energy_defaults_data_port_missing_or_invalid"),
        ("data_port", True, "energy_defaults_data_port_missing_or_invalid"),
        ("data_port", 0, "energy_defaults_data_port_missing_or_invalid"),
        ("data_port", 65536, "energy_defaults_data_port_missing_or_invalid"),
        ("channel", "0", "energy_defaults_channel_missing_or_invalid"),
        ("channel", True, "energy_defaults_channel_missing_or_invalid"),
        ("channel", -1, "energy_defaults_channel_missing_or_invalid"),
        ("sample_rate", "2000", "energy_defaults_sample_rate_missing_or_invalid"),
        ("sample_rate", True, "energy_defaults_sample_rate_missing_or_invalid"),
        ("sample_rate", 0, "energy_defaults_sample_rate_missing_or_invalid"),
    ],
)
def test_inherited_collection_rejects_malformed_defaults_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    defaults = EnergyDefaults(data_port=3000, channel=0, sample_rate=2000)
    setattr(defaults, field, value)
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(),
        defaults=defaults,
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("address", "reason"),
    [
        (197, "setup_urecs_address_not_string"),
        ("   ", "setup_urecs_address_blank"),
        (
            "192.168.0.197 ",
            "setup_urecs_address_contains_whitespace_or_control",
        ),
        (
            "192.168.0.197\n",
            "setup_urecs_address_contains_whitespace_or_control",
        ),
    ],
)
def test_inherited_collection_rejects_malformed_urecs_address_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    address: Any,
    reason: str,
) -> None:
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(urecs_address=address),
        defaults=EnergyDefaults(
            data_port=3000,
            channel=0,
            sample_rate=2000,
            physical_scope="FS",
            window_label="command",
        ),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize("enabled", [False, "true", 1, 0, None])
def test_inherited_collection_requires_effective_setup_enabled_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled: Any,
) -> None:
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(enabled=enabled),
        defaults=EnergyDefaults(),
    )
    assert "setup_energy_not_enabled" in result["reasons"]


@pytest.mark.parametrize(
    "value",
    [False, 0, "", "FS ", " fs", "fs", "full-system", "MB"],
)
def test_inherited_collection_preserves_and_rejects_nonexact_scope_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
) -> None:
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(),
        defaults=EnergyDefaults(),
        call_overrides={"physical_scope": value},
    )
    assert "effective_physical_scope_not_exact_full_system" in result["reasons"]


@pytest.mark.parametrize(
    "value",
    [False, 0, "", "command ", " command", "COMMAND", "command_window"],
)
def test_inherited_collection_preserves_and_rejects_nonexact_window_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
) -> None:
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(),
        defaults=EnergyDefaults(),
        call_overrides={"window_label": value},
    )
    assert "effective_window_label_not_exact_command" in result["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("physical_scope", False, "energy_defaults_physical_scope_missing_or_invalid"),
        ("physical_scope", "MB", "energy_defaults_physical_scope_missing_or_invalid"),
        ("physical_scope", " fs", "energy_defaults_physical_scope_missing_or_invalid"),
        ("physical_scope", "full-system", "energy_defaults_physical_scope_missing_or_invalid"),
        ("window_label", 0, "energy_defaults_window_label_missing_or_invalid"),
        ("window_label", "trimmed_activity_window", "energy_defaults_window_label_missing_or_invalid"),
        ("window_label", "complete_measurement_window", "energy_defaults_window_label_missing_or_invalid"),
        ("window_label", " command", "energy_defaults_window_label_missing_or_invalid"),
        ("window_label", "command_window", "energy_defaults_window_label_missing_or_invalid"),
    ],
)
def test_inherited_collection_rejects_malformed_direct_default_scope_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    defaults = EnergyDefaults()
    setattr(defaults, field, value)
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(),
        defaults=defaults,
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("registry", "reason"),
    [
        (
            {
                "energy_defaults": {"data_port": 3000, "channel": 0, "sample_rate": 2000},
                "hardware_setups": [{"id": SETUP_ID, "energy": "invalid"}],
            },
            "setup_energy_not_mapping",
        ),
        (
            {
                "energy_defaults": "invalid",
                "hardware_setups": [
                    {
                        "id": SETUP_ID,
                        "energy": {
                            "enabled": True,
                            "urecs_address": "192.168.0.197",
                        },
                    }
                ],
            },
            "energy_defaults_not_mapping",
        ),
        (
            {
                "energy_defaults": {"data_port": "3000", "channel": 0, "sample_rate": 2000},
                "hardware_setups": [
                    {
                        "id": SETUP_ID,
                        "energy": {
                            "enabled": True,
                            "urecs_address": "192.168.0.197",
                        },
                    }
                ],
            },
            "energy_defaults_data_port_missing_or_invalid",
        ),
    ],
)
def test_raw_registry_malformed_contract_survives_parsing_and_blocks_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registry: dict[str, Any],
    reason: str,
) -> None:
    setup = energy_setup_from_registry(registry, SETUP_ID)
    defaults = energy_defaults_from_registry(registry)
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=setup,
        defaults=defaults,
    )
    assert reason in result["reasons"]


def test_valid_direct_dataclasses_pass_registry_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        collector, "_verify_calibration_manifest", _verified_inherited_method
    )

    def _tools(_defaults: EnergyDefaults) -> dict[str, Any]:
        calls.append("tools")
        return {"collector_found": False}

    monkeypatch.setattr(collector, "check_energy_tools", _tools)
    result = collector.run_fast_firmware_measurement(
        "echo not-reached-without-tools",
        tmp_path / "valid",
        setup=_valid_setup(),
        defaults=EnergyDefaults(
            data_port=3000,
            channel=0,
            sample_rate=2000,
            physical_scope="FS",
            window_label="command",
        ),
        duration_s=1.0,
        setup_id=SETUP_ID,
        physical_scope="FS",
        calibration_manifest="/synthetic/inherited-method.json",
        calibration_sha256="a" * 64,
    )
    assert calls == ["tools"]
    assert result["error"] == "urecs-data-collector not found"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("collector_binary", False, "energy_defaults_collector_binary_missing_or_invalid"),
        ("collector_binary", 0, "energy_defaults_collector_binary_missing_or_invalid"),
        ("collector_binary", "", "energy_defaults_collector_binary_missing_or_invalid"),
        ("collector_binary", " urecs-data-collector", "energy_defaults_collector_binary_missing_or_invalid"),
        ("collector_binary", "urecs-data-collector ", "energy_defaults_collector_binary_missing_or_invalid"),
        ("power_calculations_binary", False, "energy_defaults_power_calculations_binary_missing_or_invalid"),
        ("power_calculations_binary", 0, "energy_defaults_power_calculations_binary_missing_or_invalid"),
        ("power_calculations_binary", "", "energy_defaults_power_calculations_binary_missing_or_invalid"),
        ("power_calculations_binary", " power_calculations", "energy_defaults_power_calculations_binary_missing_or_invalid"),
        ("power_calculations_binary", "power_calculations ", "energy_defaults_power_calculations_binary_missing_or_invalid"),
        ("mode", False, "energy_defaults_mode_missing_or_invalid"),
        ("mode", 0, "energy_defaults_mode_missing_or_invalid"),
        ("mode", "", "energy_defaults_mode_missing_or_invalid"),
        ("mode", "FAST_FIRMWARE", "energy_defaults_mode_not_fast_firmware"),
        ("mode", "fast-firmware", "energy_defaults_mode_not_fast_firmware"),
        ("mode", "fast_firmware ", "energy_defaults_mode_not_fast_firmware"),
    ],
)
def test_inherited_collection_rejects_malformed_tools_and_mode_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    defaults = EnergyDefaults()
    setattr(defaults, field, value)
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(),
        defaults=defaults,
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (3001, "setup_data_port_defaults_mismatch"),
        ("3000", "setup_data_port_missing_or_invalid"),
        (True, "setup_data_port_missing_or_invalid"),
        (0, "setup_data_port_missing_or_invalid"),
        (65536, "setup_data_port_missing_or_invalid"),
    ],
)
def test_raw_setup_data_port_is_not_overwritten_by_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: Any,
    reason: str,
) -> None:
    manifest = Path("/synthetic/inherited-method.json")
    registry = _method_bound_registry(
        manifest,
        "a" * 64,
        selected_energy={"data_port": value},
    )
    setup = energy_setup_from_registry(registry, SETUP_ID)
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=setup,
        defaults=energy_defaults_from_registry(registry),
    )
    assert reason in result["reasons"]
    if value == 3001:
        assert setup.data_port == 3001


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("data_port", 3001, f"expected_channel_bindings_data_port_defaults_mismatch:{SECOND_SETUP_ID}"),
        ("channel", 1, f"expected_channel_bindings_channel_defaults_mismatch:{SECOND_SETUP_ID}"),
        ("sample_rate", 1000, f"expected_channel_bindings_sample_rate_defaults_mismatch:{SECOND_SETUP_ID}"),
        ("sample_rate_hz", 1000, f"expected_channel_bindings_sample_rate_hz_defaults_mismatch:{SECOND_SETUP_ID}"),
    ],
)
def test_nonselected_manifest_bound_setup_override_blocks_selected_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
        second_energy={field: value},
    )
    setup = energy_setup_from_registry(registry, SETUP_ID)
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=setup,
        defaults=energy_defaults_from_registry(registry),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("collector_binary", False, "energy_defaults_collector_binary_missing_or_invalid"),
        ("collector_binary", " urecs-data-collector", "energy_defaults_collector_binary_missing_or_invalid"),
        ("power_calculations_binary", 0, "energy_defaults_power_calculations_binary_missing_or_invalid"),
        ("power_calculations_binary", "power_calculations ", "energy_defaults_power_calculations_binary_missing_or_invalid"),
        ("mode", False, "energy_defaults_mode_missing_or_invalid"),
        ("mode", "wrong", "energy_defaults_mode_not_fast_firmware"),
    ],
)
def test_raw_defaults_tool_and_mode_errors_survive_fallback_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
        defaults_override={field: value},
    )
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("defaults_override", "reason"),
    [
        ({"enabled": "true"}, "energy_defaults_enabled_not_bool"),
        ({"physical_scope": False}, "energy_defaults_physical_scope_missing_or_invalid"),
        ({"physical_scope": " fs"}, "energy_defaults_physical_scope_missing_or_invalid"),
        ({"measurement_physical_scope": 0}, "energy_defaults_measurement_physical_scope_missing_or_invalid"),
        (
            {"physical_scope": "FS", "measurement_physical_scope": "MB"},
            "energy_defaults_physical_scope_alias_mismatch",
        ),
        ({"window_label": ""}, "energy_defaults_window_label_missing_or_invalid"),
        ({"measurement_window": False}, "energy_defaults_measurement_window_missing_or_invalid"),
        (
            {"window_label": "command", "measurement_window": "trimmed_activity_window"},
            "energy_defaults_window_label_alias_mismatch",
        ),
    ],
)
def test_raw_default_enable_scope_window_errors_survive_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defaults_override: dict[str, Any],
    reason: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
        defaults_override=defaults_override,
    )
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("enabled", "true", "setup_energy_enabled_not_bool"),
        ("physical_scope", "MB", "setup_physical_scope_override_not_allowed"),
        ("physical_scope", "FS", "setup_physical_scope_override_not_allowed"),
        ("measurement_physical_scope", "FS ", "setup_measurement_physical_scope_override_not_allowed"),
        ("window_label", "command_window", "setup_window_label_override_not_allowed"),
        ("window_label", "command", "setup_window_label_override_not_allowed"),
        ("measurement_window", False, "setup_measurement_window_override_not_allowed"),
        ("collector_binary", "other", "setup_collector_binary_override_not_allowed"),
        ("collector_binary", "urecs-data-collector", "setup_collector_binary_override_not_allowed"),
        ("power_calculations_binary", "other", "setup_power_calculations_binary_override_not_allowed"),
        ("mode", "legacy", "setup_mode_override_not_allowed"),
    ],
)
def test_selected_setup_ignored_claim_override_is_preserved_and_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
        selected_energy={field: value},
    )
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("enabled", "true", "setup_energy_enabled_not_bool"),
        ("physical_scope", "MB", "setup_physical_scope_override_not_allowed"),
        ("physical_scope", "FS", "setup_physical_scope_override_not_allowed"),
        ("window_label", "command_window", "setup_window_label_override_not_allowed"),
        ("window_label", "command", "setup_window_label_override_not_allowed"),
        ("collector_binary", "other", "setup_collector_binary_override_not_allowed"),
        ("collector_binary", "urecs-data-collector", "setup_collector_binary_override_not_allowed"),
        ("power_calculations_binary", "other", "setup_power_calculations_binary_override_not_allowed"),
        ("mode", "legacy", "setup_mode_override_not_allowed"),
    ],
)
def test_nonselected_method_bound_ignored_claim_override_blocks_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
        second_energy={field: value},
    )
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
    )
    assert (
        f"expected_channel_bindings_{reason}:{SECOND_SETUP_ID}"
        in result["reasons"]
    )


@pytest.mark.parametrize("enabled_state", [False, "missing"])
def test_nonselected_method_bound_setup_must_be_literal_enabled_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    enabled_state: Any,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
        second_energy={"enabled": enabled_state},
    )
    if enabled_state == "missing":
        registry["hardware_setups"][1]["energy"].pop("enabled")
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
    )
    assert (
        f"expected_channel_bindings_setup_energy_not_enabled:{SECOND_SETUP_ID}"
        in result["reasons"]
    )


@pytest.mark.parametrize(
    ("energy_override", "reason"),
    [
        (
            {"address": "192.168.0.176"},
            "setup_urecs_address_alias_mismatch",
        ),
        (
            {"address": "bad address"},
            "setup_energy_address_contains_whitespace_or_control",
        ),
        (
            {"urecs_address": "bad address", "address": "192.168.0.197"},
            "setup_urecs_address_contains_whitespace_or_control",
        ),
    ],
)
def test_dual_urecs_address_keys_mismatch_or_malformed_block_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    energy_override: dict[str, Any],
    reason: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
        selected_energy=energy_override,
    )
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize("address_shape", ["alias_only", "matching_dual"])
def test_valid_urecs_address_alias_shapes_reach_tool_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    address_shape: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
    )
    selected = registry["hardware_setups"][0]["energy"]
    if address_shape == "alias_only":
        selected["address"] = selected.pop("urecs_address")
    else:
        selected["address"] = selected["urecs_address"]
    setup = energy_setup_from_registry(registry, SETUP_ID)
    calls: list[str] = []
    monkeypatch.setattr(
        collector, "_verify_calibration_manifest", _verified_inherited_method
    )

    def missing_tools(_defaults: EnergyDefaults) -> dict[str, Any]:
        calls.append("tools")
        return {"collector_found": False}

    monkeypatch.setattr(collector, "check_energy_tools", missing_tools)
    result = collector.run_fast_firmware_measurement(
        "echo not-reached-without-tools",
        tmp_path / address_shape,
        setup=setup,
        defaults=energy_defaults_from_registry(registry),
        duration_s=1,
        setup_id=SETUP_ID,
    )
    assert setup.urecs_address == "192.168.0.197"
    assert setup.expected_channel_bindings_valid is True
    assert calls == ["tools"]
    assert result["error"] == "urecs-data-collector not found"


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("setup_id", 1, "expected_channel_binding_setup_id_invalid:0"),
        ("setup_id", "orin nx", "expected_channel_binding_setup_id_invalid:0"),
        ("setup_id", "orin_nx_hailo8_01 ", "expected_channel_binding_setup_id_invalid:0"),
        ("scope", "fs", "expected_channel_binding_scope_invalid:0"),
        ("scope", " FS", "expected_channel_binding_scope_invalid:0"),
        ("scope", False, "expected_channel_binding_scope_invalid:0"),
        (
            "measurement_point",
            "complete_system_input ",
            "expected_channel_binding_measurement_point_invalid:0",
        ),
        (
            "measurement_point",
            False,
            "expected_channel_binding_measurement_point_invalid:0",
        ),
    ],
)
def test_direct_expected_binding_text_fields_are_not_coerced_before_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    reason: str,
) -> None:
    row = dict(_valid_setup().expected_channel_bindings[0])
    row[field] = value
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=_valid_setup(expected_channel_bindings=(row,)),
        defaults=EnergyDefaults(physical_scope="FS", window_label="command"),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("raw_id", "reason"),
    [
        (1, "expected_channel_bindings_hardware_setup_id_not_string:1"),
        ("", "expected_channel_bindings_hardware_setup_id_blank:1"),
        (
            "orin nx h10",
            "expected_channel_bindings_hardware_setup_id_contains_whitespace_or_control:1",
        ),
        (
            "orin_nx_hailo10_01 ",
            "expected_channel_bindings_hardware_setup_id_contains_whitespace_or_control:1",
        ),
    ],
)
def test_nonselected_raw_hardware_setup_id_must_be_exact_before_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_id: Any,
    reason: str,
) -> None:
    registry = _method_bound_registry(
        Path("/synthetic/inherited-method.json"),
        "a" * 64,
    )
    registry["hardware_setups"][1]["id"] = raw_id
    result = _run_blocked(
        tmp_path,
        monkeypatch,
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
    )
    assert reason in result["reasons"]


@pytest.mark.parametrize(
    ("host", "remote", "reason"),
    [
        (
            {"address": "192.0.2.20 ", "user": "nx", "port": 22},
            None,
            "jetson_host_address_not_canonical",
        ),
        (
            {"address": "192.0.2.20", "user": "n x", "port": 22},
            None,
            "jetson_host_user_not_canonical",
        ),
        (
            {"address": "192.0.2.20", "user": "nx", "port": 22},
            {"host": "192.0.2.21", "user": "nx", "port": 22},
            "jetson_host_remote_address_mismatch",
        ),
        (
            {"address": "192.0.2.20", "user": "nx", "port": 22},
            {"host": "192.0.2.20", "user": "other", "port": 22},
            "jetson_host_remote_user_mismatch",
        ),
        (
            {"address": "192.0.2.20", "user": "nx", "port": 22},
            {"host": "192.0.2.20", "user": "nx", "port": 2222},
            "jetson_host_remote_port_mismatch",
        ),
        (
            {
                "address": "192.0.2.20",
                "user": "nx",
                "port": 22,
                "ssh_extra_args": "-o BatchMode=yes",
            },
            {
                "host": "192.0.2.20",
                "user": "nx",
                "port": 22,
                "ssh_extra_args": "-o BatchMode=no",
            },
            "jetson_host_remote_ssh_extra_args_mismatch",
        ),
    ],
)
def test_jetson_identity_parser_preserves_raw_conflicts_and_noncanonical_text(
    host: Any,
    remote: Any,
    reason: str,
) -> None:
    raw = {
        "id": SETUP_ID,
        "host": host,
        "energy": {"enabled": True, "urecs_address": "192.168.0.197"},
    }
    if remote is not None:
        raw["remote"] = remote
    setup = energy_setup_from_raw(raw)
    assert setup.jetson_identity_valid is False
    assert reason in setup.jetson_identity_errors


@pytest.mark.parametrize(
    ("setup_id", "manifest", "digest", "reason"),
    [
        ("other_setup", "/synthetic/inherited-method.json", "a" * 64, "effective_setup_id_setup_mismatch"),
        ("", "/synthetic/inherited-method.json", "a" * 64, "effective_setup_id_missing_or_noncanonical"),
        (SETUP_ID, "/synthetic/other-method.json", "a" * 64, "call_calibration_manifest_setup_mismatch"),
        (SETUP_ID, "/synthetic/inherited-method.json", "b" * 64, "call_calibration_sha256_setup_mismatch"),
        (SETUP_ID, "/synthetic/inherited-method.json", "", "call_calibration_manifest_sha256_pair_incomplete"),
        (SETUP_ID, "", "a" * 64, "call_calibration_manifest_sha256_pair_incomplete"),
    ],
)
def test_call_binding_override_or_incomplete_pair_is_rejected_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    setup_id: str,
    manifest: str,
    digest: str,
    reason: str,
) -> None:
    started: list[str] = []

    def forbidden(name: str):
        def call(*_args: Any, **_kwargs: Any) -> Any:
            started.append(name)
            raise AssertionError(f"{name} must not start")
        return call

    monkeypatch.setattr(
        collector, "_verify_calibration_manifest", _verified_inherited_method
    )
    monkeypatch.setattr(collector, "check_energy_tools", forbidden("tools"))
    monkeypatch.setattr(collector, "run_duration_probe", forbidden("workload"))
    monkeypatch.setattr(collector, "_run_one", forbidden("collector"))
    result = collector.run_fast_firmware_measurement(
        "echo must-not-run",
        tmp_path / "call-binding-blocked",
        setup=_valid_setup(),
        defaults=EnergyDefaults(),
        duration_s=1,
        setup_id=setup_id,
        physical_scope="FS",
        calibration_manifest=manifest,
        calibration_sha256=digest,
    )
    assert result["status"] == "energy_input_registry_contract_blocked"
    assert reason in result["reasons"]
    assert result["collector_started"] is False
    assert result["workload_started"] is False
    assert result["transport_started"] is False
    assert started == []


def test_cli_empty_method_options_inherit_exact_setup_owned_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = [
        _binding(SETUP_ID, "192.168.0.197"),
        _binding(SECOND_SETUP_ID, "192.168.0.176"),
    ]
    manifest, digest = _inherited_manifest(tmp_path, bindings)
    registry = _method_bound_registry(manifest, digest)
    setup = energy_setup_from_registry(registry, SETUP_ID)
    defaults = energy_defaults_from_registry(registry)
    calls: list[str] = []
    verified_call: dict[str, Any] = {}
    original_verify = collector._verify_calibration_manifest

    def capture_verify(manifest_arg: Any, sha_arg: Any, **kwargs: Any):
        verified_call.update(
            {"manifest": manifest_arg, "sha256": sha_arg, **kwargs}
        )
        return original_verify(manifest_arg, sha_arg, **kwargs)

    def missing_tools(_defaults: EnergyDefaults) -> dict[str, Any]:
        calls.append("tools")
        return {"collector_found": False}

    monkeypatch.setattr(collector, "check_energy_tools", missing_tools)
    monkeypatch.setattr(
        collector, "_verify_calibration_manifest", capture_verify
    )
    result = collector.run_fast_firmware_measurement(
        "echo not-reached-without-tools",
        tmp_path / "valid-auto-inherit",
        setup=setup,
        defaults=defaults,
        duration_s=1,
        setup_id=SETUP_ID,
        calibration_manifest="",
        calibration_sha256="",
    )
    assert calls == ["tools"]
    assert result["error"] == "urecs-data-collector not found"
    assert verified_call["manifest"] == str(manifest.resolve())
    assert verified_call["sha256"] == digest
    assert verified_call["setup_id"] == SETUP_ID
    assert verified_call["physical_scope"] == "FS"
    assert verified_call["expected_channel_bindings"] == (
        tuple(setup.expected_channel_bindings)
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "missing",
        "rebound",
        "port",
        "setup_id_type",
        "setup_id_whitespace",
        "scope_lower",
        "scope_padded",
        "measurement_point_padded",
        "measurement_point_type",
    ],
)
def test_fully_resigned_manifest_binding_set_mutation_never_starts_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    registry_bindings = [
        _binding(SETUP_ID, "192.168.0.197"),
        _binding(SECOND_SETUP_ID, "192.168.0.176"),
    ]
    manifest, _digest = _inherited_manifest(tmp_path, registry_bindings)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if mutation == "extra":
        payload["channel_bindings"].append(
            _binding("orin_nx_deepx_m1_01", "192.168.0.185")
        )
    elif mutation == "missing":
        payload["channel_bindings"] = payload["channel_bindings"][:1]
    elif mutation == "rebound":
        payload["channel_bindings"][1]["urecs_address"] = "192.168.0.250"
    elif mutation == "port":
        payload["channel_bindings"][1]["data_port"] = 3001
    elif mutation == "setup_id_type":
        payload["channel_bindings"][1]["setup_id"] = 1
    elif mutation == "setup_id_whitespace":
        payload["channel_bindings"][1]["setup_id"] = "orin nx hailo10"
    elif mutation == "scope_lower":
        payload["channel_bindings"][1]["scope"] = "fs"
    elif mutation == "scope_padded":
        payload["channel_bindings"][1]["scope"] = " FS"
    elif mutation == "measurement_point_padded":
        payload["channel_bindings"][1]["measurement_point"] = (
            "complete_system_input "
        )
    else:
        payload["channel_bindings"][1]["measurement_point"] = False
    payload["channel_binding_set_sha256"] = sha256_json(
        payload["channel_bindings"]
    )
    payload["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    registry = _method_bound_registry(manifest, digest)
    setup = energy_setup_from_registry(registry, SETUP_ID)
    defaults = energy_defaults_from_registry(registry)
    started: list[str] = []

    def forbidden(name: str):
        def call(*_args: Any, **_kwargs: Any) -> Any:
            started.append(name)
            raise AssertionError(f"{name} must not start")
        return call

    monkeypatch.setattr(collector, "check_energy_tools", forbidden("tools"))
    monkeypatch.setattr(collector, "run_duration_probe", forbidden("workload"))
    monkeypatch.setattr(collector, "_run_one", forbidden("collector"))
    result = collector.run_fast_firmware_measurement(
        "echo must-not-run",
        tmp_path / f"resigned-{mutation}",
        setup=setup,
        defaults=defaults,
        duration_s=1,
        setup_id=SETUP_ID,
        physical_scope="FS",
    )
    assert result["status"] == "energy_input_provenance_runtime_binding_blocked"
    assert result["collector_started"] is False
    assert result["workload_started"] is False
    assert result["transport_started"] is False
    assert started == []
    verification = result["energy_calibration_verification"]
    assert verification["status"] == "inherited_method_manifest_invalid"
    assert "runtime_expected_channel_binding_set_mismatch" in verification[
        "runtime_binding_errors"
    ]


def test_fully_resigned_manifest_with_appended_string_binding_never_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = [
        _binding(SETUP_ID, "192.168.0.197"),
        _binding(SECOND_SETUP_ID, "192.168.0.176"),
    ]
    manifest, _digest = _inherited_manifest(tmp_path, bindings)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["channel_bindings"].append("malicious-filtered-row")
    payload["channel_binding_set_sha256"] = sha256_json(
        payload["channel_bindings"]
    )
    payload["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    registry = _method_bound_registry(manifest, digest)
    started: list[str] = []

    def forbidden(name: str):
        def call(*_args: Any, **_kwargs: Any) -> Any:
            started.append(name)
            raise AssertionError(f"{name} must not start")
        return call

    monkeypatch.setattr(collector, "check_energy_tools", forbidden("tools"))
    monkeypatch.setattr(collector, "run_duration_probe", forbidden("workload"))
    monkeypatch.setattr(collector, "_run_one", forbidden("collector"))
    result = collector.run_fast_firmware_measurement(
        "echo must-not-run",
        tmp_path / "resigned-appended-string",
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
        duration_s=1,
        setup_id=SETUP_ID,
        physical_scope="FS",
    )
    assert result["status"] == "energy_input_provenance_runtime_binding_blocked"
    assert started == []
    assert result["collector_started"] is False
    assert result["workload_started"] is False
    assert result["transport_started"] is False
    verification = result["energy_calibration_verification"]
    assert verification["status"] == "inherited_method_manifest_invalid"
    assert verification["runtime_binding_errors"] == [
        "runtime_channel_binding_rows_not_exact"
    ]


def test_fully_resigned_legacy_underbound_method_never_starts_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = [
        _binding(SETUP_ID, "192.168.0.197"),
        _binding(SECOND_SETUP_ID, "192.168.0.176"),
    ]
    manifest, _digest = _inherited_manifest(tmp_path, bindings)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["method"].pop("implementation_policy", None)
    payload["artifacts"] = payload["artifacts"][:1]
    payload["artifact_set_sha256"] = sha256_json(payload["artifacts"])
    payload["channel_binding_set_sha256"] = sha256_json(
        payload["channel_bindings"]
    )
    payload["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    registry = _method_bound_registry(manifest, digest)
    started: list[str] = []

    def forbidden(name: str):
        def call(*_args: Any, **_kwargs: Any) -> Any:
            started.append(name)
            raise AssertionError(f"{name} must not start")
        return call

    monkeypatch.setattr(collector, "check_energy_tools", forbidden("tools"))
    monkeypatch.setattr(collector, "run_duration_probe", forbidden("workload"))
    monkeypatch.setattr(collector, "_run_one", forbidden("collector"))
    result = collector.run_fast_firmware_measurement(
        "echo must-not-run",
        tmp_path / "resigned-legacy-underbound",
        setup=energy_setup_from_registry(registry, SETUP_ID),
        defaults=energy_defaults_from_registry(registry),
        duration_s=1,
        setup_id=SETUP_ID,
        physical_scope="FS",
    )
    assert result["status"] == "energy_input_provenance_runtime_binding_blocked"
    assert started == []
    assert result["collector_started"] is False
    assert result["workload_started"] is False
    assert result["transport_started"] is False
    verification = result["energy_calibration_verification"]
    assert verification["status"] == "inherited_method_manifest_invalid"
    semantic = verification["inherited_content_verification"]
    assert semantic["expected_method_identity_ok"] is False
    assert semantic["expected_implementation_artifacts_ok"] is False
