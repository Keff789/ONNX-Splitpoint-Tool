from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.energy.config import (
    EffectiveEnergyState,
    resolve_effective_energy_config,
    resolve_energy_ab_config,
    transition_energy_state,
    write_effective_energy_manifest,
)
from onnx_splitpoint_tool.protocol_freeze import (
    CONFIRMATORY_HOLDOUT_ROLE,
    create_protocol_freeze,
    normalize_evaluation_role,
    verify_configured_protocol_freeze,
    verify_protocol_freeze,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import _profile_model_entries


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _protocol_profile(root: Path, *, amendment: str = "initial") -> Path:
    manifests = {}
    for kind in ("candidate", "dag", "prediction", "policy", "energy"):
        path = _write(root / f"{kind}.json", {"kind": kind, "version": 1})
        manifests[kind] = path.name
    profile = {
        "name": "confirmatory-test",
        "campaign": {
            "id": "confirmatory-test",
            "mode": "final",
            "require_protocol_freeze": True,
            "protocol_freeze": {
                "artifact": "protocol_freeze.json",
                "version": "1.0",
                "amendment": amendment,
                "release_identity": {"release_id": "release-2.63", "tool_version": "2.63"},
                "evaluation_identity": {"evaluation_id": "eval-001"},
                "manifests": manifests,
            },
        },
        "model_suite": {
            "primary": [
                {
                    "id": "development-model",
                    "model_sha256": "sha256:" + "1" * 64,
                    "family_id": "development-family",
                    "evaluation_role": "development",
                    "generalization_scope": "development",
                    "candidate_universe": {"mode": "declared_shortlist"},
                },
                {
                    "id": "holdout-model",
                    "model_sha256": "sha256:" + "2" * 64,
                    "family_id": "holdout-family",
                    "evaluation_role": "confirmatory_holdout",
                    "generalization_scope": "model_family_holdout",
                    "candidate_universe": {"mode": "deterministic_audit", "audit_size": 20},
                },
            ],
            "reserve": [],
        },
    }
    return _write(root / "profile.json", profile)


def test_protocol_role_uses_confirmatory_name_and_accepts_legacy_alias() -> None:
    assert normalize_evaluation_role("confirmatory_holdout") == CONFIRMATORY_HOLDOUT_ROLE
    assert normalize_evaluation_role("holdout") == CONFIRMATORY_HOLDOUT_ROLE
    assert normalize_evaluation_role("development") == "development"


def test_scientific_reporting_normalizes_legacy_holdout_profile_rows() -> None:
    entries = _profile_model_entries({
        "model_suite": {
            "primary": [{"id": "legacy", "evaluation_role": "holdout"}],
            "reserve": [],
        }
    })
    assert entries["legacy"]["evaluation_role"] == "confirmatory_holdout"


def test_protocol_freeze_detects_silent_manifest_change(tmp_path: Path) -> None:
    profile = _protocol_profile(tmp_path)
    freeze = create_protocol_freeze(profile=profile, output=tmp_path / "protocol_freeze.json", signer="test")

    first = verify_protocol_freeze(freeze, profile=profile)
    assert first["ok"] is True
    assert first["protocol_version"] == "1.0"
    assert first["amendment"] == "initial"
    configured = verify_configured_protocol_freeze(
        json.loads(profile.read_text(encoding="utf-8")), profile_path=profile
    )
    assert configured["ok"] is True
    assert configured["status"] == "verified"

    _write(tmp_path / "dag.json", {"kind": "dag", "version": 2})
    changed = verify_protocol_freeze(freeze, profile=profile)
    assert changed["ok"] is False
    assert any(row["field"].endswith("manifests.dag.sha256") for row in changed["mismatches"])

    # Restoring the artefact but silently changing the protocol version is also
    # a mismatch; an intentional change needs a new amendment/freeze.
    _write(tmp_path / "dag.json", {"kind": "dag", "version": 1})
    profile_payload = json.loads(profile.read_text(encoding="utf-8"))
    profile_payload["campaign"]["protocol_freeze"]["version"] = "1.1"
    _write(profile, profile_payload)
    version_changed = verify_protocol_freeze(freeze, profile=profile)
    assert version_changed["ok"] is False
    assert any(row["field"] == "protocol_version" for row in version_changed["mismatches"])


def test_protocol_amendment_requires_reason_and_superseded_identity(tmp_path: Path) -> None:
    profile = _protocol_profile(tmp_path, amendment="1")
    with pytest.raises(ValueError, match="amendment_reason"):
        create_protocol_freeze(profile=profile, output=tmp_path / "protocol_freeze.json")


def test_protocol_freeze_is_optional_for_unmodified_v262_profiles() -> None:
    profile = {"campaign": {"mode": "development"}}
    result = verify_configured_protocol_freeze(profile)
    assert result["ok"] is True
    assert result["configured"] is False
    assert result["status"] == "not_configured_legacy_compatible"


def test_legacy_native_energy_is_one_effective_requested_configuration() -> None:
    profile = {
        # v2.62 used this false value to mean generic energy was disabled even
        # though the native measurement path was explicitly requested.
        "energy": {
            "enabled": False,
            "requested_native_energy": True,
            "measurement_path": "native_only",
        },
        "native_producers": {
            "energy": {
                "enabled": True,
                "window_method_validation_probe": {
                    "enabled": True,
                    "repeats": 3,
                    "include_raw_parquet": True,
                    "strict": True,
                },
            }
        },
    }
    result = resolve_effective_energy_config(profile)
    assert result["lifecycle"]["status"] == "configured"
    assert result["lifecycle"]["requested"] is True
    assert result["lifecycle"]["configured"] is True
    assert result["measurement_path"] == "native_only"
    assert result["window_method_ab"]["primary_method"] == "command_marker_window"
    assert result["window_method_ab"]["shadow_method"] == "chapter4_legacy_window"
    assert result["window_method_ab"]["baseline_method"] == "chapter4_baseline"
    assert result["window_method_ab"]["candidate_method"] == "candidate_v263"
    assert result["window_method_ab"]["smoke_repeats"] == 3


def test_explicit_false_request_cannot_silently_run_enabled_native_energy() -> None:
    profile = {
        "energy": {"requested": False, "measurement_path": "native_only"},
        "native_producers": {"energy": {"enabled": True}},
    }
    result = resolve_effective_energy_config(profile)
    assert result["lifecycle"]["status"] == "failed"
    assert "explicit_energy_request_false_conflicts_with_enabled_execution_path" in result["configuration_errors"]


def test_energy_ab_is_same_capture_shadow_only_and_never_auto_switches() -> None:
    good = resolve_energy_ab_config({})
    assert good["valid"] is True
    assert good["same_raw_capture"] is True
    assert good["mode"] == "shadow"
    assert good["auto_switch"] is False
    assert good["requires_picoscope"] is False
    assert good["scientific_primary_method"] == "command_marker_window"
    assert good["scientific_shadow_method"] == "chapter4_legacy_window"

    invalid = resolve_energy_ab_config({"auto_switch": True, "requires_picoscope": True})
    assert invalid["valid"] is False
    assert "automatic_method_switch_forbidden" in invalid["validation_errors"]
    assert "picoscope_not_required_for_postprocessing_ab" in invalid["validation_errors"]


def test_energy_lifecycle_and_manifest_are_explicit(tmp_path: Path) -> None:
    state = EffectiveEnergyState(status="requested", requested=True, configured=False)
    state = transition_energy_state(state, "configured")
    state = transition_energy_state(state, "scheduled")
    state = transition_energy_state(state, "running")
    state = transition_energy_state(state, "completed")
    assert state.completed is True
    assert state.running is False
    with pytest.raises(ValueError, match="invalid energy state transition"):
        transition_energy_state(state, "running")

    profile = {
        "energy": {
            "requested": True,
            "enabled": True,
            "measurement_path": "native_only",
            "window_method_ab": {
                "enabled": True,
                "baseline_method": "chapter4_baseline",
                "candidate_method": "candidate_v263",
                "same_raw_capture": True,
                "mode": "shadow",
                "auto_switch": False,
                "smoke_repeats": 3,
                "requires_picoscope": False,
            },
        },
        "native_producers": {"energy": {"enabled": True}},
    }
    path = write_effective_energy_manifest(tmp_path / "energy_protocol.json", profile)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    lifecycle = manifest["effective_energy"]["lifecycle"]
    assert lifecycle["status"] == "configured"
    assert lifecycle["requested"] is True
    assert lifecycle["configured"] is True
    assert manifest["manifest_payload_sha256"].startswith("sha256:")
