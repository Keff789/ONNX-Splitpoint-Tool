from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.workflow.profile_options import (
    load_runtime_profile_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _profile(version: str) -> Path:
    return ROOT / f"profiles/yolo11l_v{version}_r8b_full_b067_gate.yaml"


def _recovery_policy(version: str) -> dict:
    return {
        "schema": "onnx-splitpoint/yolo11-r8b-recovery-policy",
        "schema_version": 1,
        "source_release": "2.79.6",
        "import_mode": "read_only_hash_bound",
        "imported_terminal_paths": ["hailo8_b067_composed"],
        "diagnostic_only_imports": ["hailo10h_full_generic"],
        "fresh_terminal_paths": [
            "hailo8_full",
            "hailo10h_full",
            "deepx_full",
            "hailo10h_b067_composed",
            "deepx_b067_composed",
        ],
        "recovery_manifest": f"v{version}_recovery_manifest.json",
        "forbid_generic_full_success": True,
        "require_native_full_backends": [
            "native_full_hailo8",
            "native_full_hailo10h",
            "native_full_deepx",
        ],
    }


@pytest.mark.parametrize("version", ["2797", "2798", "27910"])
def test_shipped_yolo11_gate_profile_passes_the_strict_profile_loader(
    version: str,
) -> None:
    profile = _profile(version)
    key = f"v{version}_yolo11_recovery"

    loaded = load_evaluation_profile(
        str(profile), base_dir=profile.parent, validate=True,
    )

    assert loaded is not None
    assert loaded.profile_id == f"yolo11l_v{version}_r8b_full_b067_gate"
    assert loaded.raw_profile[key] == _recovery_policy(version)


@pytest.mark.parametrize("version", ["2797", "2798", "27910"])
def test_shipped_yolo11_gate_profile_passes_the_real_workflow_loader(
    version: str,
) -> None:
    profile = _profile(version).resolve()
    key = f"v{version}_yolo11_recovery"

    resolved, start_snapshot = load_runtime_profile_snapshot(str(profile))

    expected = _recovery_policy(version)
    assert resolved[key] == expected
    assert start_snapshot["profile_id"] == (
        f"yolo11l_v{version}_r8b_full_b067_gate"
    )
    assert start_snapshot["source_profile"][key] == expected
    assert start_snapshot["resolved_profile"][key] == expected
    assert start_snapshot["consistency"] == {
        "status": "ok",
        "mismatches": [],
    }


def _v27910_payload() -> dict:
    payload = yaml.safe_load(_profile("27910").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_policy_property",
        "wrong_policy_schema",
        "wrong_import_mode",
        "missing_fresh_terminal_path",
        "duplicate_native_full_backend",
        "allow_generic_full_success",
        "wrong_versioned_manifest",
        "both_versioned_recovery_keys",
    ],
)
def test_yolo11_recovery_policy_schema_is_fail_closed(mutation: str) -> None:
    payload = _v27910_payload()
    policy = payload["v27910_yolo11_recovery"]

    if mutation == "unknown_policy_property":
        policy["unreviewed_import"] = True
    elif mutation == "wrong_policy_schema":
        policy["schema"] = "onnx-splitpoint/yolo11-r8b-recovery-policy-typo"
    elif mutation == "wrong_import_mode":
        policy["import_mode"] = "copy_mutable"
    elif mutation == "missing_fresh_terminal_path":
        policy["fresh_terminal_paths"].pop()
    elif mutation == "duplicate_native_full_backend":
        policy["require_native_full_backends"][-1] = "native_full_hailo8"
    elif mutation == "allow_generic_full_success":
        policy["forbid_generic_full_success"] = False
    elif mutation == "wrong_versioned_manifest":
        policy["recovery_manifest"] = "v2797_recovery_manifest.json"
    elif mutation == "both_versioned_recovery_keys":
        historical = copy.deepcopy(policy)
        historical["recovery_manifest"] = "v2797_recovery_manifest.json"
        payload["v2797_yolo11_recovery"] = historical
    else:  # pragma: no cover - the parametrization above is exhaustive.
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="Invalid evaluation profile"):
        validate_evaluation_profile_payload(
            payload,
            source=f"v27910-yolo11-recovery-negative:{mutation}",
        )
