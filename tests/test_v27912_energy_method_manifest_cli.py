from __future__ import annotations

import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.campaign import verify_energy_calibration_manifest
from onnx_splitpoint_tool.energy import config as energy_config
from onnx_splitpoint_tool.energy.config import load_hardware_registry
from onnx_splitpoint_tool.energy.method_manifest import (
    CANONICAL_IMPLEMENTATION_ARTIFACT_IDS,
    EnergyMethodManifestError,
    InvalidEnergyMethodRegistryContractError,
    STANDARD_PLATFORM_SETUP_IDS,
    prepare_configured_energy_method,
    verify_configured_energy_method,
)
from onnx_splitpoint_tool.platform_power_cli import _parser, main
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.workflow.artifacts import sha256_json
from scripts import energy_measurement_cli as measurement_cli


URECS_BY_SETUP = {
    "orin_nx_hailo8_01": "192.168.0.197",
    "orin_nx_hailo10_01": "192.168.0.176",
    "orin_nx_deepx_m1_01": "192.168.0.185",
}


def _source_integrity_report() -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/installed-source-integrity-verification",
        "schema_version": 1,
        "ok": True,
        "status": "verified",
        "root": "/opt/onnx-splitpoint-tool",
        "package_version": VERSION,
        "build_id": BUILD_ID,
        "manifest_path": "/opt/onnx-splitpoint-tool/SOURCE_MANIFEST.json",
        "manifest_sha256": "a" * 64,
        "sha256sums_path": "/opt/onnx-splitpoint-tool/SHA256SUMS.txt",
        "sha256sums_sha256": "b" * 64,
        "checks": {"all": True},
        "errors": [],
    }


@pytest.fixture(autouse=True)
def _verified_installed_source_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _source_integrity_report()
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_installed_source_integrity",
        lambda: copy.deepcopy(report),
    )


def _registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("urecs-data-collector", "power_calculations"):
        executable = bin_dir / name
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    path = tmp_path / "hardware_setups.yaml"
    payload = {
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 2,
        "energy_defaults": {
            "collector_binary": "urecs-data-collector",
            "power_calculations_binary": "power_calculations",
            "mode": "fast_firmware",
            "channel": 0,
            "sample_rate": 2000,
            "physical_scope": "FS",
            "window_label": "command",
        },
        "hardware_setups": [
            {
                "id": setup_id,
                "label": setup_id,
                "accelerator": "test",
                "energy": {
                    "enabled": True,
                    "urecs_address": URECS_BY_SETUP[setup_id],
                    "accelerator_idle_w": 1.25,
                },
                "unrelated": {"preserved": True},
            }
            for setup_id in STANDARD_PLATFORM_SETUP_IDS
        ],
        "hardware_groups": {
            "all_accelerators": list(STANDARD_PLATFORM_SETUP_IDS)
        },
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return path


def _resign_manifest_and_registry(
    *,
    manifest: Path,
    registry_path: Path,
    payload: dict[str, object],
) -> str:
    artifacts = payload.get("artifacts")
    assert isinstance(artifacts, list)
    payload["artifact_set_sha256"] = sha256_json(artifacts)
    bindings = payload.get("channel_bindings")
    assert isinstance(bindings, list)
    payload["channel_binding_set_sha256"] = sha256_json(bindings)
    payload["manifest_payload_sha256"] = sha256_json(
        {
            key: value
            for key, value in payload.items()
            if key != "manifest_payload_sha256"
        }
    )
    manifest.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for setup in registry["hardware_setups"]:
        setup["energy"]["calibration_sha256"] = digest
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    return digest


def test_prepare_energy_method_requires_explicit_attestation() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["prepare-energy-method", "--attested-by", "Kevin Mika"]
        )
    with pytest.raises(SystemExit):
        _parser().parse_args(
            ["prepare-energy-method", "--accept-validated-method-reuse"]
        )


def test_prepare_source_integrity_failure_changes_neither_output_nor_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    original_registry = registry_path.read_bytes()
    output_dir = tmp_path / "refused-source-integrity"
    failed = _source_integrity_report()
    failed.update(
        {
            "ok": False,
            "status": "source_integrity_failed",
            "errors": ["release_files_changed"],
        }
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest.verify_installed_source_integrity",
        lambda: copy.deepcopy(failed),
    )

    with pytest.raises(
        EnergyMethodManifestError,
        match="installed release source integrity preflight failed",
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
    assert registry_path.read_bytes() == original_registry


def test_prepare_energy_method_verifies_all_bindings_and_updates_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    output_dir = tmp_path / "method"
    result = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=output_dir,
    )

    assert result["ok"] is True
    assert result["hardware_action_performed"] is False
    assert result["setup_ids"] == list(STANDARD_PLATFORM_SETUP_IDS)
    assert len(result["manifest_sha256"]) == 64
    assert result["semantic_verification"]["ok"] is True
    manifest = Path(result["manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["reuse_attestation"]["attested_by"] == "Kevin Mika"
    assert payload["reuse_attestation"]["validated_method_accepted"] is True
    assert payload["reuse_attestation"]["new_calibration_required"] is False
    assert payload["source_release_integrity"]["manifest_sha256"] == "a" * 64
    assert payload["source_release_integrity"]["verification_report"]["ok"] is True
    assert [row["setup_id"] for row in payload["channel_bindings"]] == list(
        STANDARD_PLATFORM_SETUP_IDS
    )

    registry = load_hardware_registry(registry_path)
    rows = {
        row["id"]: row for row in registry["hardware_setups"]
    }
    for setup_id in STANDARD_PLATFORM_SETUP_IDS:
        energy = rows[setup_id]["energy"]
        assert energy["calibration_manifest"] == str(manifest.resolve())
        assert energy["calibration_sha256"] == result["manifest_sha256"]
        assert energy["accelerator_idle_w"] == 1.25
        assert rows[setup_id]["unrelated"] == {"preserved": True}
        verification = verify_configured_energy_method(
            setup_id, registry=registry
        )
        assert verification["configured"] is True
        assert verification["verified"] is True
        assert verification["status"] == "inherited_validated_method_verified"
        assert verification["sha256"] == result["manifest_sha256"]
        assert verification["runtime_binding_errors"] == []


def test_configured_method_does_not_coerce_truthy_verified_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method-truthy-verifier",
    )

    monkeypatch.setattr(
        "onnx_splitpoint_tool.energy.method_manifest._verify_reference",
        lambda **_kwargs: {
            "verified": "yes",
            "status": "malformed_truthy_verifier_result",
            "path": prepared["manifest"],
            "actual_sha256": prepared["manifest_sha256"],
            "runtime_binding_errors": [],
            "configured_method_admission_errors": [],
        },
    )
    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "malformed_truthy_verifier_result"


@pytest.mark.parametrize("legacy_scope", ["missing", "MB"])
def test_prepare_migrates_exact_v27911_defaults_only_at_final_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_scope: str,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    defaults = registry["energy_defaults"]
    defaults.pop("window_label")
    if legacy_scope == "missing":
        defaults.pop("physical_scope")
    else:
        defaults["physical_scope"] = "MB"
    first_energy = registry["hardware_setups"][0]["energy"]
    first_energy.update(
        {
            "accelerator_idle_calibrated_at": "2026-09-02T12:00:00Z",
            "accelerator_idle_calibration_evidence": "/preserved/evidence.json",
            "accelerator_idle_calibration_binding_path": "/preserved/binding.json",
            "accelerator_idle_calibration_binding_sha256": "c" * 64,
        }
    )
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    result = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / f"method_{legacy_scope}",
    )

    migration = result["energy_defaults_migration"]
    assert migration["applied"] is True
    assert migration["effective_physical_scope"] == "FS"
    assert migration["effective_window_label"] == "command"
    stored = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert stored["energy_defaults"]["physical_scope"] == "FS"
    assert stored["energy_defaults"]["window_label"] == "command"
    stored_first = stored["hardware_setups"][0]
    assert stored_first["unrelated"] == {"preserved": True}
    assert stored_first["energy"]["accelerator_idle_w"] == 1.25
    assert (
        stored_first["energy"]["accelerator_idle_calibration_binding_sha256"]
        == "c" * 64
    )


@pytest.mark.parametrize(
    "mutation",
    ["padded_scope", "conflicting_scope_alias", "invalid_window"],
)
def test_prepare_refuses_nonhistorical_defaults_without_registry_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    defaults = registry["energy_defaults"]
    if mutation == "padded_scope":
        defaults["physical_scope"] = " MB"
    elif mutation == "conflicting_scope_alias":
        defaults["physical_scope"] = "MB"
        defaults["measurement_physical_scope"] = "FS"
    elif mutation == "invalid_window":
        defaults["window_label"] = " command"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    original = registry_path.read_bytes()

    with pytest.raises(EnergyMethodManifestError):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=tmp_path / f"refused_{mutation}",
        )

    assert registry_path.read_bytes() == original
    assert not (tmp_path / f"refused_{mutation}").exists()


def test_prepare_legacy_defaults_registry_race_has_no_partial_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["energy_defaults"]["physical_scope"] = "MB"
    registry["energy_defaults"].pop("window_label")
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    @contextmanager
    def racing_registry_lock(_path: Path):
        raced = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        raced["energy_defaults"]["data_port"] = 3999
        registry_path.write_text(
            yaml.safe_dump(raced, sort_keys=False), encoding="utf-8"
        )
        yield

    monkeypatch.setattr(
        energy_config, "_hardware_registry_write_lock", racing_registry_lock
    )
    with pytest.raises(
        EnergyMethodManifestError,
        match="hardware energy configuration changed while the manifest was prepared",
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=tmp_path / "raced_legacy_method",
        )

    persisted = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert persisted["energy_defaults"]["data_port"] == 3999
    assert persisted["energy_defaults"]["physical_scope"] == "MB"
    assert "window_label" not in persisted["energy_defaults"]
    for setup in persisted["hardware_setups"]:
        assert not setup["energy"].get("calibration_manifest")
        assert not setup["energy"].get("calibration_sha256")


def test_measure_cli_claim_uses_one_registry_snapshot_not_stale_standalone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup = energy_config.EnergySetup(
        setup_id="setup",
        enabled=True,
        urecs_address="192.0.2.10",
        data_port=3000,
        data_port_valid=True,
        calibration_manifest="/configured/method.json",
        calibration_sha256="a" * 64,
        expected_channel_bindings=(
            {
                "setup_id": "setup",
                "urecs_address": "192.0.2.10",
                "data_port": 3000,
                "channel": 0,
                "sample_rate_hz": 2000,
                "scope": "FS",
                "measurement_point": "complete_system_input",
            },
        ),
        expected_channel_bindings_valid=True,
    )
    registry = {"snapshot": "fresh"}
    fresh_defaults = energy_config.EnergyDefaults(
        physical_scope="FS", window_label="command"
    )
    monkeypatch.setattr(
        measurement_cli, "load_hardware_registry", lambda: registry
    )
    monkeypatch.setattr(
        measurement_cli,
        "energy_setup_from_registry",
        lambda observed, setup_id, **_kwargs: setup
        if observed is registry and setup_id == "setup"
        else (_ for _ in ()).throw(AssertionError("wrong registry snapshot")),
    )
    monkeypatch.setattr(
        measurement_cli,
        "energy_defaults_from_registry",
        lambda observed: fresh_defaults
        if observed is registry
        else (_ for _ in ()).throw(AssertionError("wrong registry snapshot")),
    )
    monkeypatch.setattr(
        measurement_cli,
        "load_energy_defaults",
        lambda: (_ for _ in ()).throw(
            AssertionError("stale standalone MB defaults must not be read")
        ),
    )
    monkeypatch.setattr(
        measurement_cli,
        "get_setup_energy",
        lambda _setup_id: (_ for _ in ()).throw(
            AssertionError("setup must not be re-read")
        ),
    )
    reached: dict[str, object] = {}
    monkeypatch.setattr(
        measurement_cli,
        "run_fast_firmware_measurement",
        lambda *args, **kwargs: reached.update(kwargs) or {"ok": True},
    )
    args = SimpleNamespace(
        setup_id="setup", out=str(tmp_path / "out"), workdir="", run_id="claim",
        command="true", command_file="", cwd=None, duration=1.0, runs=1,
        exact_run_count=False, timeout=10.0, inference_count=1, pipeline_fps=1.0,
        physical_scope="FS", window_label="command",
        require_runtime_work_units=False, require_command_window_alignment=False,
        compare_legacy_window=False, calibration_manifest="", calibration_sha256="",
        preflight_command="", preflight_command_file="", preflight_timeout_s=10.0,
        preflight_attestation_max_age_s=60.0, preflight_runtime_attestation_path="",
        preflight_expected_command_contract_sha256="", invalid_repeat_max_retries=0,
        diagnostic_only=False, claim_exclusion_reason="", window_method_ab_json="",
        host_normalization_source_run_id="", host_normalization_target_variant="",
    )

    assert measurement_cli.cmd_measure(args) == 0
    assert reached["setup"] is setup
    assert reached["defaults"] is fresh_defaults


def test_prepare_energy_method_cli_performs_no_platform_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("platform control must not run during preparation")

    monkeypatch.setattr(
        "onnx_splitpoint_tool.platform_power.send_udp_command", forbidden
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.platform_power.set_jetson_state", forbidden
    )
    rc = main(
        [
            "--registry",
            str(registry_path),
            "prepare-energy-method",
            "--attested-by",
            "Kevin Mika",
            "--accept-validated-method-reuse",
            "--output-dir",
            str(tmp_path / "cli_method"),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    result = json.loads(captured.out)
    assert result["ok"] is True
    assert result["hardware_action_performed"] is False


def test_configured_method_detects_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    result = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    manifest = Path(result["manifest"])
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )
    assert verification["configured"] is True
    assert verification["verified"] is False
    assert verification["status"] == "sha256_mismatch"


def test_configured_method_rejects_same_binary_names_after_path_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    result = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    assert result["ok"] is True

    replacement_dir = tmp_path / "replacement_bin"
    replacement_dir.mkdir()
    for name in ("urecs-data-collector", "power_calculations"):
        executable = replacement_dir / name
        executable.write_text(
            "#!/bin/sh\necho unrelated-replacement\n", encoding="utf-8"
        )
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(replacement_dir))

    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )
    assert verification["configured"] is True
    assert verification["verified"] is False
    assert verification["status"] == "inherited_method_runtime_binding_mismatch"
    reasons = verification["runtime_binding_errors"]
    assert "runtime_collector_binary_artifact_path_mismatch" in reasons
    assert "runtime_collector_binary_artifact_sha256_mismatch" in reasons
    assert "runtime_postprocessor_binary_artifact_path_mismatch" in reasons
    assert "runtime_postprocessor_binary_artifact_sha256_mismatch" in reasons
    binary = verification["verification"]["runtime_binary_verification"]
    assert binary["collector"]["verified"] is False
    assert binary["postprocessor"]["verified"] is False


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            "missing_policy",
            "configured_method_exact_implementation_policy_missing",
        ),
        (
            "missing_source",
            "configured_method_artifact_tool_source_metrics_missing",
        ),
        (
            "duplicate_source",
            "configured_method_artifact_tool_source_metrics_ambiguous",
        ),
        (
            "substituted_source",
            "configured_method_artifact_tool_source_metrics_identity_mismatch",
        ),
        (
            "extra_artifact",
            "configured_method_artifact_set_not_exact",
        ),
    ],
)
def test_configured_method_admission_rejects_resigned_stripped_or_rebound_v2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
) -> None:
    """A self-consistently rehashed, but weakened v2 manifest is not enough."""
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    manifest = Path(prepared["manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    metrics_rows = [
        row for row in artifacts if row["id"] == "tool_source_metrics"
    ]
    assert len(metrics_rows) == 1

    if mutation == "missing_policy":
        payload["method"].pop("implementation_policy")
    elif mutation == "missing_source":
        payload["artifacts"] = [
            row
            for row in artifacts
            if row["id"] != "tool_source_metrics"
        ]
    elif mutation == "duplicate_source":
        payload["artifacts"].append(dict(metrics_rows[0]))
    elif mutation == "substituted_source":
        decoy = tmp_path / "metrics.py"
        decoy.write_text("# internally consistent but not tool code\n", encoding="utf-8")
        metrics_rows[0]["path"] = str(decoy)
        metrics_rows[0]["sha256"] = hashlib.sha256(decoy.read_bytes()).hexdigest()
        metrics_rows[0]["size_bytes"] = decoy.stat().st_size
    elif mutation == "extra_artifact":
        decoy = tmp_path / "extra.py"
        decoy.write_text("# unapproved ninth artifact\n", encoding="utf-8")
        payload["artifacts"].append(
            {
                "id": "unapproved_ninth_implementation_artifact",
                "kind": "measurement_implementation",
                "path": str(decoy),
                "requested_executable": "",
                "resolution": "path",
                "sha256": "sha256:"
                + hashlib.sha256(decoy.read_bytes()).hexdigest(),
                "size_bytes": decoy.stat().st_size,
            }
        )
    else:  # pragma: no cover - the parametrization is closed above.
        raise AssertionError(mutation)

    _resign_manifest_and_registry(
        manifest=manifest,
        registry_path=registry_path,
        payload=payload,
    )
    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )

    assert verification["configured"] is True
    assert verification["verified"] is False
    assert verification["status"] in {
        "configured_energy_method_admission_failed",
        "inherited_method_manifest_invalid",
    }
    assert expected_error in verification["configured_method_admission_errors"]
    admission = verification["verification"]["configured_method_admission"]
    assert admission["ok"] is False
    assert admission["required_artifact_ids"] == list(
        CANONICAL_IMPLEMENTATION_ARTIFACT_IDS
    )
    # Both the shared expected-binding verifier and the configured-method
    # exact-implementation verifier are allowed to reject first.


def test_resigned_method_cannot_rebind_authoritative_source_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    manifest = Path(prepared["manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    binding = payload["source_release_integrity"]
    binding["manifest_sha256"] = "c" * 64
    binding["verification_report"]["manifest_sha256"] = "c" * 64
    _resign_manifest_and_registry(
        manifest=manifest,
        registry_path=registry_path,
        payload=payload,
    )

    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )

    assert verification["configured"] is True
    assert verification["verified"] is False
    assert (
        "configured_method_source_integrity_binding_invalid"
        in verification["configured_method_admission_errors"]
    )
    source = verification["source_integrity_verification"]
    assert source["ok"] is False
    assert "current_source_manifest_sha256_mismatch" in source["errors"]


def test_resigned_method_rejects_float_source_binding_schema_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    manifest = Path(prepared["manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_release_integrity"]["schema_version"] = 1.0
    _resign_manifest_and_registry(
        manifest=manifest,
        registry_path=registry_path,
        payload=payload,
    )
    resigned_payload = json.loads(manifest.read_text(encoding="utf-8"))
    semantic = verify_energy_calibration_manifest(
        resigned_payload,
        require_final=True,
        expected_channel_bindings=resigned_payload["channel_bindings"],
        require_source_integrity_binding=True,
    )

    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )

    assert semantic["ok"] is False
    assert semantic["source_integrity_binding_shape_ok"] is False
    assert verification["verified"] is False
    assert (
        "configured_method_source_integrity_binding_invalid"
        in verification["configured_method_admission_errors"]
    )
    source = verification["source_integrity_verification"]
    assert source["ok"] is False
    assert "source_integrity_binding_version_invalid" in source["errors"]


@pytest.mark.parametrize(
    "mutation",
    [
        "schema_float",
        "schema_bool",
        "evidence_mode_padded",
        "policy_padded",
        "artifact_id_padded",
        "artifact_kind_padded",
        "artifact_path_padded",
        "artifact_extra_field",
        "channel_id_padded",
        "measurement_point_padded",
        "top_sample_rate_float",
        "top_sample_rate_bool",
        "method_sample_rate_string",
        "binding_setup_id_padded",
        "binding_address_padded",
        "binding_data_port_float",
        "binding_channel_bool",
        "binding_sample_rate_float",
        "binding_scope_padded",
        "binding_measurement_point_padded",
        "extra_top_level",
        "bindings_alias",
        "target_bindings_alias",
    ],
)
def test_resigned_manifest_identity_literals_are_never_coerced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    manifest = Path(prepared["manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    artifact = next(
        row
        for row in payload["artifacts"]
        if row["id"] == "tool_source_metrics"
    )
    binding = payload["channel_bindings"][0]
    if mutation == "schema_float":
        payload["schema_version"] = 2.0
    elif mutation == "schema_bool":
        payload["schema_version"] = True
    elif mutation == "evidence_mode_padded":
        payload["evidence_mode"] += " "
    elif mutation == "policy_padded":
        payload["method"]["implementation_policy"] += " "
    elif mutation == "artifact_id_padded":
        artifact["id"] += " "
    elif mutation == "artifact_kind_padded":
        artifact["kind"] += " "
    elif mutation == "artifact_path_padded":
        artifact["path"] += " "
    elif mutation == "artifact_extra_field":
        artifact["attacker_claim_alias"] = "ignored-before-v2.79.12"
    elif mutation == "channel_id_padded":
        payload["channel_id"] += " "
    elif mutation == "measurement_point_padded":
        payload["measurement_point"] += " "
    elif mutation == "top_sample_rate_float":
        payload["sample_rate_hz"] = 2000.0
    elif mutation == "top_sample_rate_bool":
        payload["sample_rate_hz"] = True
    elif mutation == "method_sample_rate_string":
        payload["method"]["sample_rate_hz"] = "2000"
    elif mutation == "binding_setup_id_padded":
        binding["setup_id"] += " "
    elif mutation == "binding_address_padded":
        binding["urecs_address"] += " "
    elif mutation == "binding_data_port_float":
        binding["data_port"] = 3000.0
    elif mutation == "binding_channel_bool":
        binding["channel"] = False
    elif mutation == "binding_sample_rate_float":
        binding["sample_rate_hz"] = 2000.0
    elif mutation == "binding_scope_padded":
        binding["scope"] += " "
    elif mutation == "binding_measurement_point_padded":
        binding["measurement_point"] += " "
    elif mutation == "extra_top_level":
        payload["attacker_claim_alias"] = []
    elif mutation == "bindings_alias":
        payload["bindings"] = payload["channel_bindings"]
    elif mutation == "target_bindings_alias":
        payload["target_bindings"] = payload["channel_bindings"]
    else:  # pragma: no cover
        raise AssertionError(mutation)

    _resign_manifest_and_registry(
        manifest=manifest,
        registry_path=registry_path,
        payload=payload,
    )
    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )
    assert verification["configured"] is True
    assert verification["verified"] is False
    assert verification["status"] in {
        "inherited_method_manifest_invalid",
        "configured_energy_method_admission_failed",
    }
    assert verification["configured_method_admission_errors"]


def test_duplicate_target_setup_ids_fail_closed_for_verify_and_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "valid_method",
    )
    assert prepared["ok"] is True

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    setup_id = STANDARD_PLATFORM_SETUP_IDS[0]
    original = next(
        row for row in registry["hardware_setups"] if row["id"] == setup_id
    )
    original["remote"] = {"host": "192.168.0.104", "user": "nx"}
    duplicate = json.loads(json.dumps(original))
    duplicate["remote"]["host"] = "192.168.0.250"
    duplicate["energy"]["urecs_address"] = "192.168.0.251"
    registry["hardware_setups"].append(duplicate)
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    verification = verify_configured_energy_method(
        setup_id, registry_path=registry_path
    )
    assert verification["configured"] is False
    assert verification["verified"] is False
    assert verification["status"] == "duplicate_hardware_setup_id"
    assert verification["duplicate_setup_ids"] == [setup_id]
    assert verification["selection_errors"] == [
        f"duplicate hardware setup id(s): {setup_id}"
    ]

    refused_output = tmp_path / "duplicate_method"
    with pytest.raises(
        EnergyMethodManifestError,
        match=rf"duplicate hardware setup id\(s\): {setup_id}",
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=refused_output,
        )
    assert not refused_output.exists()


def test_configured_method_rejects_runtime_data_port_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    manifest_payload = json.loads(
        Path(prepared["manifest"]).read_text(encoding="utf-8")
    )
    assert manifest_payload["method"]["data_port"] == 3000
    assert {
        row["data_port"] for row in manifest_payload["channel_bindings"]
    } == {3000}

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["energy_defaults"]["data_port"] = 3999
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "inherited_method_manifest_invalid"
    assert verification["verification"][
        "expected_channel_bindings_exact"
    ] is False
    assert "configured_method_data_port_mismatch" in verification[
        "configured_method_admission_errors"
    ]
    assert "configured_method_binding_data_port_mismatch" in verification[
        "configured_method_admission_errors"
    ]


def test_prepare_energy_method_detects_data_port_registry_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)

    @contextmanager
    def racing_registry_lock(_path: Path):
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        registry["energy_defaults"]["data_port"] = 3999
        registry_path.write_text(
            yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
        )
        yield

    monkeypatch.setattr(
        energy_config,
        "_hardware_registry_write_lock",
        racing_registry_lock,
    )
    with pytest.raises(
        EnergyMethodManifestError,
        match="hardware energy configuration changed while the manifest was prepared",
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=tmp_path / "raced_method",
        )

    persisted = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    assert persisted["energy_defaults"]["data_port"] == 3999
    for setup in persisted["hardware_setups"]:
        assert not setup["energy"].get("calibration_manifest")
        assert not setup["energy"].get("calibration_sha256")


@pytest.mark.parametrize("bad_port", ["3000", True, 0, 65536])
def test_raw_data_port_never_falls_back_for_verify_or_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_port: object,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "valid_method",
    )
    assert prepared["ok"] is True

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["energy_defaults"]["data_port"] = bad_port
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    setup_id = STANDARD_PLATFORM_SETUP_IDS[0]
    verification = verify_configured_energy_method(
        setup_id, registry_path=registry_path
    )
    assert verification["configured"] is True
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_defaults_data_port"
    assert "energy_defaults.data_port" in verification[
        "configuration_errors"
    ][0]

    refused = tmp_path / "invalid_port_method"
    with pytest.raises(
        EnergyMethodManifestError,
        match=r"energy_defaults\.data_port must be an integer UDP port",
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=refused,
        )
    assert not refused.exists()


def test_prepare_rejects_padded_setup_id_without_output_or_registry_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["hardware_setups"][0]["id"] = (
        " " + registry["hardware_setups"][0]["id"] + " "
    )
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    before = registry_path.read_bytes()
    output = tmp_path / "must-not-exist"

    with pytest.raises(
        InvalidEnergyMethodRegistryContractError,
        match="hardware setup id must be a nonempty canonical string",
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=output,
        )

    assert registry_path.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize(
    ("attested_by", "accepted"),
    [
        ("Kevin Mika", "false"),
        ("Kevin Mika", 1),
        (123, True),
        (" Kevin Mika", True),
        ("Kevin Mika ", True),
        ("Kevin\nMika", True),
    ],
)
def test_prepare_rejects_nonliteral_reuse_attestation_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attested_by: object,
    accepted: object,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    before = registry_path.read_bytes()
    output = tmp_path / "must-not-exist-attestation"

    with pytest.raises(EnergyMethodManifestError):
        prepare_configured_energy_method(
            attested_by=attested_by,  # type: ignore[arg-type]
            accepted_validated_method_reuse=accepted,  # type: ignore[arg-type]
            registry_path=registry_path,
            output_dir=output,
        )

    assert registry_path.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "expected_fragment"),
    [
        ("defaults_nonmapping", "energy_defaults must be a mapping"),
        ("channel_string", "energy_defaults.channel"),
        ("sample_rate_string", "energy_defaults.sample_rate"),
        ("setup_energy_nonmapping", "energy must be a mapping"),
        ("urecs_list", "u.RECS address"),
        ("urecs_whitespace", "without whitespace/control characters"),
    ],
)
def test_raw_method_registry_contract_never_uses_permissive_coercion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_fragment: str,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "valid_method",
    )
    assert prepared["ok"] is True

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    setup_id = STANDARD_PLATFORM_SETUP_IDS[0]
    setup = next(
        row for row in registry["hardware_setups"] if row["id"] == setup_id
    )
    if mutation == "defaults_nonmapping":
        registry["energy_defaults"] = ["not", "a", "mapping"]
    elif mutation == "channel_string":
        registry["energy_defaults"]["channel"] = "bad"
    elif mutation == "sample_rate_string":
        registry["energy_defaults"]["sample_rate"] = "bad"
    elif mutation == "setup_energy_nonmapping":
        setup["energy"] = ["not", "a", "mapping"]
    elif mutation == "urecs_list":
        setup["energy"]["urecs_address"] = ["192.168.0.197"]
    elif mutation == "urecs_whitespace":
        setup["energy"]["urecs_address"] = " 192.168.0.197"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    verification = verify_configured_energy_method(
        setup_id, registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_method_registry_contract"
    assert any(
        expected_fragment in error
        for error in verification["configuration_errors"]
    )

    refused = tmp_path / "invalid_registry_method"
    with pytest.raises(EnergyMethodManifestError, match=expected_fragment):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=refused,
        )
    assert not refused.exists()


@pytest.mark.parametrize(
    "mutation",
    ["extra_binding", "missing_binding", "rebound_nonselected_binding"],
)
def test_configured_method_requires_exact_shared_registry_binding_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    manifest = Path(prepared["manifest"])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    bindings = payload["channel_bindings"]
    selected_id = STANDARD_PLATFORM_SETUP_IDS[0]
    nonselected_id = STANDARD_PLATFORM_SETUP_IDS[1]

    if mutation == "extra_binding":
        bindings.append(
            {
                "setup_id": "unconfigured_extra_setup",
                "urecs_address": "192.168.0.250",
                "data_port": 3000,
                "channel": 0,
                "sample_rate_hz": 2000,
                "scope": "FS",
                "measurement_point": "complete_system_input",
            }
        )
    elif mutation == "missing_binding":
        payload["channel_bindings"] = [
            row for row in bindings if row["setup_id"] != nonselected_id
        ]
    elif mutation == "rebound_nonselected_binding":
        rebound = next(
            row for row in bindings if row["setup_id"] == nonselected_id
        )
        rebound["urecs_address"] = "192.168.0.250"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    _resign_manifest_and_registry(
        manifest=manifest,
        registry_path=registry_path,
        payload=payload,
    )
    verification = verify_configured_energy_method(
        selected_id, registry_path=registry_path
    )
    assert verification["configured"] is True
    assert verification["verified"] is False
    inner = verification["verification"]
    assert inner["inherited_content_verified"] is False
    assert inner["inherited_content_verification"][
        "expected_channel_bindings_ok"
    ] is False
    assert "configured_method_channel_binding_set_mismatch" in verification[
        "configured_method_admission_errors"
    ]


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("data_port", 3001),
        ("data_port", "3000"),
        ("data_port", True),
        ("data_port", 0),
        ("data_port", 65536),
        ("channel", 1),
        ("channel", "0"),
        ("sample_rate", 1000),
        ("sample_rate_hz", "2000"),
        ("physical_scope", "MB"),
        ("window_label", "command_window"),
        ("collector_binary", "urecs-data-collector"),
        ("power_calculations_binary", "power_calculations"),
        ("mode", "fast_firmware"),
        ("measurement_physical_scope", "FS"),
        ("measurement_window", "command"),
    ],
)
def test_per_setup_measurement_identity_override_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad_value: object,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    setup_id = STANDARD_PLATFORM_SETUP_IDS[0]
    setup = next(
        row for row in registry["hardware_setups"] if row["id"] == setup_id
    )
    setup["energy"][key] = bad_value
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    verification = verify_configured_energy_method(
        setup_id, registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_method_registry_contract"
    assert any(
        f"energy.{key}" in error
        for error in verification["configuration_errors"]
    )
    refused = tmp_path / "invalid_setup_override_method"
    with pytest.raises(EnergyMethodManifestError, match=rf"energy\.{key}"):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=refused,
        )
    assert not refused.exists()


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("collector_binary", False),
        ("power_calculations_binary", 0),
        ("mode", False),
        ("mode", "fast-firmware"),
        ("enabled", "true"),
    ],
)
def test_explicit_runtime_implementation_defaults_are_not_coerced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad_value: object,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["energy_defaults"][key] = bad_value
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    setup_id = STANDARD_PLATFORM_SETUP_IDS[0]
    verification = verify_configured_energy_method(
        setup_id, registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_method_registry_contract"
    assert any(
        f"energy_defaults.{key}" in error
        for error in verification["configuration_errors"]
    )
    refused = tmp_path / "invalid_runtime_default_method"
    with pytest.raises(
        EnergyMethodManifestError, match=rf"energy_defaults\.{key}"
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=refused,
        )
    assert not refused.exists()


@pytest.mark.parametrize("bad_enabled", [False, "true", 1, None])
def test_configured_method_requires_literal_enabled_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_enabled: object,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepared = prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    setup = registry["hardware_setups"][0]
    if bad_enabled is None:
        setup["energy"].pop("enabled")
    else:
        setup["energy"]["enabled"] = bad_enabled
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    verification = verify_configured_energy_method(
        setup["id"], registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_method_registry_contract"
    assert any(
        "energy.enabled" in error
        for error in verification["configuration_errors"]
    )
    assert Path(prepared["manifest"]).is_file()
    with pytest.raises(EnergyMethodManifestError, match=r"energy\.enabled"):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=tmp_path / "refused",
        )


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("physical_scope", "full_system"),
        ("physical_scope", False),
        ("window_label", "command_window"),
        ("window_label", " command"),
        ("window_label", False),
    ],
)
def test_method_defaults_require_exact_full_system_command_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    bad_value: object,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["energy_defaults"][key] = bad_value
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    setup_id = STANDARD_PLATFORM_SETUP_IDS[0]

    verification = verify_configured_energy_method(
        setup_id, registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_method_registry_contract"
    assert any(
        f"energy_defaults.{key}" in error
        for error in verification["configuration_errors"]
    )
    with pytest.raises(
        EnergyMethodManifestError, match=rf"energy_defaults\.{key}"
    ):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=tmp_path / "refused",
        )


@pytest.mark.parametrize(
    ("legacy_address", "expected_fragment"),
    [
        ("192.168.0.250", "must be equal"),
        (" 192.168.0.197", "canonical string"),
        (False, "canonical string"),
    ],
)
def test_dual_urecs_address_aliases_must_be_valid_and_equal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_address: object,
    expected_fragment: str,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    setup_id = STANDARD_PLATFORM_SETUP_IDS[0]
    setup = next(
        row for row in registry["hardware_setups"] if row["id"] == setup_id
    )
    setup["energy"]["address"] = legacy_address
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    verification = verify_configured_energy_method(
        setup_id, registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_method_registry_contract"
    assert any(
        expected_fragment in error
        for error in verification["configuration_errors"]
    )
    with pytest.raises(EnergyMethodManifestError, match=expected_fragment):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=tmp_path / "refused",
        )


@pytest.mark.parametrize(
    ("primary", "alias", "bad_alias"),
    [
        ("physical_scope", "measurement_physical_scope", "FULL_SYSTEM"),
        ("physical_scope", "measurement_physical_scope", False),
        ("window_label", "measurement_window", "command_window"),
        ("window_label", "measurement_window", False),
    ],
)
def test_method_default_aliases_cannot_conflict_or_coerce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    primary: str,
    alias: str,
    bad_alias: object,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["energy_defaults"][alias] = bad_alias
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "invalid_energy_method_registry_contract"
    assert any(
        f"energy_defaults.{alias}" in error
        or f"{primary} aliases" in error
        for error in verification["configuration_errors"]
    )
    with pytest.raises(EnergyMethodManifestError):
        prepare_configured_energy_method(
            attested_by="Kevin Mika",
            accepted_validated_method_reuse=True,
            registry_path=registry_path,
            output_dir=tmp_path / "refused",
        )


def test_nonselected_bound_setup_method_override_blocks_selected_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = _registry(tmp_path, monkeypatch)
    prepare_configured_energy_method(
        attested_by="Kevin Mika",
        accepted_validated_method_reuse=True,
        registry_path=registry_path,
        output_dir=tmp_path / "method",
    )
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    registry["hardware_setups"][1]["energy"]["window_label"] = "trimmed"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    verification = verify_configured_energy_method(
        STANDARD_PLATFORM_SETUP_IDS[0], registry_path=registry_path
    )
    assert verification["verified"] is False
    assert verification["status"] == "configured_energy_method_binding_contract_invalid"
    assert any(
        "energy.window_label" in error
        for error in verification["configuration_errors"]
    )
