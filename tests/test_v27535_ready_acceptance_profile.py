from __future__ import annotations

from pathlib import Path

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.workflow.start_snapshot import (
    materialize_runtime_profile,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "resnet50_v27535_small_acceptance.yaml"


def test_packaged_resnet50_acceptance_profile_has_exact_small_scope() -> None:
    source = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))

    selection = source["selection_policy"]
    assert selection["max_accepted_cases_per_model"] == 1
    assert selection["preferred_shortlist"] == 5
    assert selection["min_gap"] == 0
    assert selection["audit_size"] == 4
    assert selection["minimum_valid_audit_candidates"] == 3
    assert selection["audit_seed"] == 20260710
    assert selection["selection_strategy"] == "score_independent_audit"
    assert selection["require_single_part2_input"] is False

    assert [row["id"] for row in source["model_suite"]["primary"]] == [
        "resnet50"
    ]
    assert [row["id"] for row in source["run_profiles"]] == [
        "ort_tensorrt",
        "hailo8",
        "hailo8_to_trt",
        "hailo10",
        "hailo10_to_tensorrt",
        "deepx_m1_full",
        "deepx_m1_to_tensorrt",
    ]
    assert source["execution_preset"] == {
        "id": "standard",
        "follow_tool_config": True,
        "overrides": {
            "native_enabled": True,
            "energy_enabled": False,
        },
    }


def test_packaged_resnet50_acceptance_profile_resolves_energy_and_probe_off(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "missing_user_run_modes.yaml"),
    )
    loaded = load_evaluation_profile(PROFILE)

    assert loaded is not None
    profile = loaded.raw_profile
    assert profile["selection_policy"]["audit_size"] == 4
    assert profile["selection_policy"]["minimum_valid_audit_candidates"] == 3
    assert profile["selection_policy"]["min_gap"] == 0
    assert profile["native_producers"]["enabled"] is True
    assert profile["energy"]["enabled"] is False
    assert profile["energy"]["generic_enabled"] is False
    assert profile["energy"]["requested_native_energy"] is False
    assert profile["native_producers"]["energy"]["enabled"] is False

    plan = build_effective_execution_plan(profile)
    assert plan["models"] == ["resnet50"]
    assert plan["score_independent_audit_counts"] == {"resnet50": 4}
    assert plan["score_independent_audit_minimum_valid_counts"] == {
        "resnet50": 3
    }
    assert plan["execution_union_candidate_counts_min_by_model"] == {
        "resnet50": 4
    }
    assert plan["execution_union_candidate_counts_upper_bound_by_model"] == {
        "resnet50": 5
    }
    assert plan["native_split_backends"] == ["hailo8", "hailo10h", "deepx"]
    assert plan["native_full_baselines"] is True
    assert plan["native_frames"] == 1000
    assert plan["native_warmup"] == 100
    assert plan["native_performance_repetitions"] == 3
    assert plan["expected_generic_result_rows_min_by_model"] == {
        "resnet50": 20
    }
    assert plan["expected_generic_result_rows_by_model"] == {"resnet50": 24}

    runtime_profile, runtime_bindings = materialize_runtime_profile(
        profile,
        profile_path=str(PROFILE),
    )
    probe = runtime_profile["native_producers"]["energy"][
        "window_method_validation_probe"
    ]
    assert probe["enabled"] is False
    assert probe["activation_reason"] == (
        "not_activated_without_native_energy_or_explicit_probe_request"
    )
    assert runtime_bindings["window_method_validation_probe"] == probe
