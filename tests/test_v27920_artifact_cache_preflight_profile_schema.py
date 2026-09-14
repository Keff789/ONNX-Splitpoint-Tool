from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    resolve_artifact_cache_preflight_policy,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "onnx_splitpoint_tool/resources/evaluation_profiles"
    / "native_resnet_yolo26s_hailo8_smoke_v1.yaml"
)


def _profile() -> dict:
    value = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_root_cache_preflight_policy_validates_and_resolves() -> None:
    profile = _profile()
    profile["artifact_cache_preflight"] = {
        "enabled": True,
        "strict": True,
        "default_expectation": "warm",
        "expected_warm": [
            "regnet_x_1_6gf:trt_full:full",
            {"model_id": "yolo11l", "role": "hailo8_hef", "case_id": "b062"},
        ],
        "expected_cold": [
            {"model": "yolo26s", "role": "trt_p2", "item_id": "b024"},
        ],
    }

    validated = validate_evaluation_profile_payload(
        profile, source="v2.79.20 root cache-preflight policy"
    )
    policy = resolve_artifact_cache_preflight_policy(validated)

    assert validated == profile
    assert policy["enabled"] is True
    assert policy["default_expectation"] == "warm"
    assert policy["block_on_unexpected_cold_builds"] is True
    assert policy["declarations"] == [
        {
            "model_id": "yolo26s",
            "role": "trt_p2",
            "item_id": "b024",
            "expectation": "cold",
        },
        {
            "model_id": "regnet_x_1_6gf",
            "role": "trt_full",
            "item_id": "full",
            "expectation": "warm",
        },
        {
            "model_id": "yolo11l",
            "role": "hailo8_hef",
            "item_id": "b062",
            "expectation": "warm",
        },
    ]


def test_nested_workflow_cache_preflight_policy_is_schema_valid() -> None:
    profile = _profile()
    profile["workflow"]["artifact_cache_preflight"] = {
        "enabled": False,
        "expectation": "cold",
        "block_on_unexpected_cold_builds": False,
        "expected_warm": [{"role": "tensorrt_part2", "item_id": "b038"}],
    }

    validated = validate_evaluation_profile_payload(
        profile, source="v2.79.20 nested cache-preflight policy"
    )
    policy = resolve_artifact_cache_preflight_policy(validated)

    assert policy["enabled"] is False
    assert policy["default_expectation"] == "cold"
    assert policy["block_on_unexpected_cold_builds"] is False
    assert policy["declarations"] == [
        {
            "model_id": "*",
            "role": "trt_p2",
            "item_id": "b038",
            "expectation": "warm",
        }
    ]


@pytest.mark.parametrize(
    "policy",
    [
        {"enabled": "yes"},
        {"default_expectation": "maybe"},
        {"strict": False, "surprise": True},
        {"expected_warm": ["yolo26s:not_a_backend:b024"]},
        {"expected_cold": [{"role": "not_a_backend"}]},
        {"expected_cold": [{}]},
        {"expected_warm": [{"model": "x", "model_id": "y"}]},
        {"default_expectation": "warm", "expectation": "cold"},
    ],
)
def test_cache_preflight_policy_rejects_invalid_or_ambiguous_values(
    policy: dict,
) -> None:
    profile = _profile()
    profile["artifact_cache_preflight"] = policy

    with pytest.raises(ValueError, match="Invalid evaluation profile"):
        validate_evaluation_profile_payload(
            profile, source="invalid v2.79.20 cache-preflight policy"
        )


def test_run_mode_materialises_diagnostic_default() -> None:
    profile = _profile()
    profile["execution_preset"] = {
        "id": "smoke",
        "follow_tool_config": True,
    }

    resolved, _audit = apply_run_mode(
        profile,
        config=default_run_modes_config(),
    )

    assert resolved["workflow"]["artifact_cache_preflight"] == {
        "enabled": True,
        "default_expectation": "unspecified",
        "block_on_unexpected_cold_builds": False,
    }
    assert validate_evaluation_profile_payload(resolved) == resolved


def test_run_mode_preserves_root_and_nested_explicit_policy() -> None:
    base = _profile()
    base["execution_preset"] = {
        "id": "standard",
        "follow_tool_config": True,
    }

    root_profile = copy.deepcopy(base)
    root_profile["artifact_cache_preflight"] = {
        "require_warm_cache": True,
        "expected_cold": ["yolo26s:trt_p2:b024"],
    }
    root_resolved, _ = apply_run_mode(
        root_profile, config=default_run_modes_config()
    )
    assert root_resolved["artifact_cache_preflight"] == (
        root_profile["artifact_cache_preflight"]
    )

    nested_profile = copy.deepcopy(base)
    nested_profile["workflow"]["artifact_cache_preflight"] = {
        "enabled": True,
        "strict": True,
        "default_expectation": "warm",
    }
    nested_resolved, _ = apply_run_mode(
        nested_profile, config=default_run_modes_config()
    )
    assert nested_resolved["workflow"]["artifact_cache_preflight"] == (
        nested_profile["workflow"]["artifact_cache_preflight"]
    )
    validate_evaluation_profile_payload(root_resolved)
    validate_evaluation_profile_payload(nested_resolved)
