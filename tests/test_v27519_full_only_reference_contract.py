from __future__ import annotations

from collections import Counter
from copy import deepcopy

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.native_full_quality import (
    resolve_native_full_plan,
    resolve_native_split_plan,
)
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _benchmark_runs_from_profile,
)
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _native_expected_full_rows_v61b,
    _native_stage_backend_plan_v27519,
)


MODELS = ("resnet50", "yolo26s", "yolov7_paper")
PRODUCERS = ("hailo8", "hailo10h", "deepx")
SETUP_IDS = {
    "hailo8": "orin_nx_hailo8_01",
    "hailo10h": "orin_nx_hailo10_01",
    "deepx": "orin_nx_deepx_m1_01",
}
FULL_BACKENDS_BY_PRODUCER = {
    "hailo8": ("hailo8", "tensorrt"),
    "hailo10h": ("hailo10h", "tensorrt"),
    "deepx": ("deepx", "tensorrt"),
}


def _full_only_run_profiles() -> list[dict[str, object]]:
    """Mirror the four logical Full selections from the v2.75.19 fixture."""

    return [
        {
            "id": "ort_tensorrt",
            "type": "same_backend_reference",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
            "required": True,
        },
        {
            "id": "hailo8",
            "type": "same_backend_reference",
            "full": "hailo8",
            "stage1": "hailo8",
            "stage2": "hailo8",
            "required": False,
        },
        {
            "id": "hailo10",
            "type": "same_backend_reference",
            "full": "hailo10",
            "stage1": "hailo10",
            "stage2": "hailo10",
            "required": False,
        },
        {
            "id": "deepx_m1_full",
            "type": "same_backend_reference",
            "full": "deepx_m1",
            "stage1": "deepx_m1",
            "stage2": "deepx_m1",
            "required": False,
        },
    ]


def _three_mixed_run_profiles() -> list[dict[str, object]]:
    """Use the exact forward split IDs emitted by the profile editor."""

    return [
        {
            "id": "hailo8_to_trt",
            "type": "mixed_backend",
            "stage1": "hailo8",
            "stage2": "tensorrt",
            "required": False,
        },
        {
            "id": "hailo10_to_tensorrt",
            "type": "mixed_backend",
            "stage1": "hailo10",
            "stage2": "tensorrt",
            "required": False,
        },
        {
            "id": "deepx_m1_to_tensorrt",
            "type": "mixed_backend",
            "stage1": "deepx_m1",
            "stage2": "tensorrt",
            "required": False,
        },
    ]


def _profile(run_profiles: list[dict[str, object]]) -> dict[str, object]:
    return {
        "name": "v27519_full_only_reference_contract",
        "selection_policy": {
            "max_accepted_cases_per_model": 1,
            "preferred_shortlist": 10,
            "require_single_part2_input": False,
        },
        "model_suite": {
            "primary": [
                {"id": "resnet50", "task": "classification"},
                {"id": "yolo26s", "task": "detection"},
                {"id": "yolov7_paper", "task": "detection"},
            ]
        },
        "run_profiles": deepcopy(run_profiles),
        "native_producers": {
            "enabled": True,
            # This is adapter capability inventory, not an implicit split request.
            "backends": list(PRODUCERS),
            "energy": {"enabled": False, "mode": "plan"},
        },
        "energy": {
            "enabled": False,
            "requested_native_energy": False,
        },
        "execution_preset": {
            "id": "smoke",
            "follow_tool_config": True,
            "overrides": {
                "native_enabled": True,
                "energy_enabled": False,
            },
            "snapshot": {
                "defaults": {
                    "native_enabled": False,
                    "energy_enabled": False,
                }
            },
        },
    }


def test_split_plan_uses_logical_profiles_as_authoritative_selection() -> None:
    full_only = resolve_native_split_plan(_profile(_full_only_run_profiles()))
    assert full_only.enabled is False
    assert full_only.selected_split_backends == ()
    assert full_only.source == "evaluation_profile.run_profiles"

    mixed = resolve_native_split_plan(
        _profile(_full_only_run_profiles() + _three_mixed_run_profiles())
    )
    assert mixed.enabled is True
    assert mixed.selected_split_backends == PRODUCERS
    assert mixed.source == "evaluation_profile.run_profiles"


def test_disabled_logical_rows_cannot_activate_native_split_or_full() -> None:
    profile = _profile([
        {
            "id": "ort_tensorrt",
            "type": "same_backend_reference",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
        },
        {
            "id": "hailo8",
            "type": "same_backend_reference",
            "full": "hailo8",
            "stage1": "hailo8",
            "stage2": "hailo8",
            "enabled": False,
        },
        {
            "id": "hailo8_to_trt",
            "type": "mixed_backend",
            "stage1": "hailo8",
            "stage2": "tensorrt",
            "enabled": False,
        },
    ])

    split_plan = resolve_native_split_plan(profile)
    full_plan = resolve_native_full_plan(profile)

    assert split_plan.enabled is False
    assert split_plan.selected_split_backends == ()
    assert full_plan.enabled is False
    assert full_plan.active_producers == ()
    assert full_plan.backends_by_producer == {}


def test_full_only_plan_expands_to_eighteen_setup_local_native_rows() -> None:
    full_plan = resolve_native_full_plan(_profile(_full_only_run_profiles()))

    assert full_plan.enabled is True
    assert full_plan.active_producers == PRODUCERS
    assert full_plan.backends_by_producer == FULL_BACKENDS_BY_PRODUCER

    rows = _native_expected_full_rows_v61b(
        MODELS,
        full_plan.backends_by_producer,
        SETUP_IDS,
    )

    assert len(rows) == 18
    assert Counter(row["backend"] for row in rows) == Counter(
        {
            "native_full_hailo8": 3,
            "native_full_hailo10h": 3,
            "native_full_deepx": 3,
            "native_full_tensorrt": 9,
        }
    )
    assert {
        (row["backend_key"], row["backend"], row["setup_id"])
        for row in rows
    } == {
        ("hailo8", "native_full_hailo8", SETUP_IDS["hailo8"]),
        ("hailo8", "native_full_tensorrt", SETUP_IDS["hailo8"]),
        ("hailo10h", "native_full_hailo10h", SETUP_IDS["hailo10h"]),
        ("hailo10h", "native_full_tensorrt", SETUP_IDS["hailo10h"]),
        ("deepx", "native_full_deepx", SETUP_IDS["deepx"]),
        ("deepx", "native_full_tensorrt", SETUP_IDS["deepx"]),
    }
    assert {row["model"] for row in rows} == set(MODELS)
    assert all(row["case"] == "full" for row in rows)
    assert all(row["execution_mode"] == "native_full_baseline" for row in rows)


def test_effective_plan_requires_single_part2_input_only_for_selected_splits() -> None:
    full_only = build_effective_execution_plan(
        _profile(_full_only_run_profiles())
    )
    assert full_only["native_split_backends"] == []
    assert full_only["native_split_requires_single_part2_input"] is False
    assert full_only["effective_require_single_part2_input"] is False
    assert full_only["native_multi_input_policy"] == "not_applicable"

    mixed = build_effective_execution_plan(
        _profile(_full_only_run_profiles() + _three_mixed_run_profiles())
    )
    assert mixed["native_split_backends"] == list(PRODUCERS)
    assert mixed["native_split_requires_single_part2_input"] is True
    assert mixed["effective_require_single_part2_input"] is False
    assert (
        mixed["native_multi_input_policy"]
        == "supported_subset_of_selected_generic_cases"
    )


def test_run_mode_materializes_empty_split_selection_and_three_by_two_full_map() -> None:
    resolved, _audit = apply_run_mode(
        _profile(_full_only_run_profiles()),
        mode_id="smoke",
        config=default_run_modes_config(),
    )

    native = resolved["native_producers"]
    expected_map = {
        producer: list(backends)
        for producer, backends in FULL_BACKENDS_BY_PRODUCER.items()
    }
    assert native["backends"] == list(PRODUCERS)
    assert native["split_backends"] == []
    assert native["split_selection_source"] == "evaluation_profile.run_profiles"
    assert native["full_baselines"]["enabled"] is True
    assert native["full_baselines"]["backends_by_producer"] == expected_map
    assert len(native["full_baselines"]["backends_by_producer"]) == 3
    assert all(
        len(backends) == 2
        for backends in native["full_baselines"]["backends_by_producer"].values()
    )
    effective = resolved["execution_preset"]["effective"]
    assert effective["native_split_backends"] == []
    assert effective["native_full_backends_by_producer"] == expected_map

    stage_plan = _native_stage_backend_plan_v27519(resolved, native)
    assert stage_plan["split_backends"] == []
    assert stage_plan["execution_backends"] == list(PRODUCERS)
    assert stage_plan["full_enabled"] is True
    assert stage_plan["full_backends_by_producer"] == expected_map


def test_ort_tensorrt_target_projects_to_cuda_ort_baseline() -> None:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.profile_payload = {"run_profiles": [_full_only_run_profiles()[0]]}
    runner.options = WorkflowOptions(profile="", out="")

    targets = runner._targets()
    assert targets == ["tensorrt"]
    assert runner._baseline_backends(targets) == ["cuda_ort"]


def test_central_management_cpu_reference_is_not_added_in_cache_verify() -> None:
    central_profile = {
        "run_profiles": [_full_only_run_profiles()[0]],
        "quality_gate": {
            "statistics": {"execution_location": "central_management"}
        },
    }

    runs = _benchmark_runs_from_profile(central_profile, ["tensorrt"])
    cpu_rows = [row for row in runs if row.get("id") == "ort_cpu"]
    assert [row["id"] for row in runs] == ["ort_tensorrt", "ort_cpu"]
    assert len(cpu_rows) == 1
    assert cpu_rows[0]["semantic_reference_only"] is True
    assert cpu_rows[0]["canonical_cpu_reference"] is True
    assert cpu_rows[0]["automatic_reference"] is True
    assert cpu_rows[0]["execution_location"] == "central_management"
    assert cpu_rows[0]["performance_eligible"] is False
    assert cpu_rows[0]["energy_eligible"] is False
    assert cpu_rows[0]["ranking_eligible"] is False
    assert cpu_rows[0]["pareto_eligible"] is False

    cache_verify_profile = deepcopy(central_profile)
    cache_verify_profile["execution_guard"] = {"mode": "cache_verify_only"}
    guarded_runs = _benchmark_runs_from_profile(
        cache_verify_profile,
        ["tensorrt"],
    )
    assert [row["id"] for row in guarded_runs] == ["ort_tensorrt"]
    assert not any(row.get("semantic_reference_only") for row in guarded_runs)
