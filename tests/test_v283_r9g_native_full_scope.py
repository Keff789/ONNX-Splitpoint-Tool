"""A narrowed GUI selection must replace the previous Native Full matrix."""
from copy import deepcopy

import pytest

from onnx_splitpoint_tool.native_full_quality import resolve_native_full_plan
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow.runner import (
    _native_expected_full_rows_v61b,
    _native_stage_backend_plan_v27519,
)


PRODUCERS = ("hailo8", "hailo10h", "deepx")


def _profile():
    return {
        "native_producers": {"enabled": True, "backends": list(PRODUCERS)},
        "run_profiles": [
            {"id": backend, "full": backend,
             "stage1": backend, "stage2": backend}
            for backend in (*PRODUCERS, "tensorrt")
        ] + [
            {"id": producer + "_to_trt", "stage1": producer, "stage2": "tensorrt"}
            for producer in PRODUCERS
        ],
    }


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("narrowing", ["remove", "disable"])
def test_current_selection_replaces_materialized_full_map(producer, narrowing):
    config = default_run_modes_config()
    profile, _ = apply_run_mode(_profile(), mode_id="standard", config=config)
    assert set(profile["native_producers"]["full_baselines"]["backends_by_producer"]) == set(PRODUCERS)
    wanted = {producer, "tensorrt", producer + "_to_trt"}
    if narrowing == "remove":
        profile["run_profiles"] = [r for r in profile["run_profiles"] if r["id"] in wanted]
    else:
        for row in profile["run_profiles"]:
            row["enabled"] = row["id"] in wanted
    # Leave the old materialized Full map and adapter inventory in place,
    # just as when the normal GUI narrows a previously resolved profile.
    before = deepcopy(profile)
    plan = resolve_native_full_plan(profile)
    assert profile == before
    assert plan.active_producers == (producer,)
    assert plan.backends_by_producer == {producer: (producer, "tensorrt")}
    for _ in range(2):
        profile, _ = apply_run_mode(profile, config=config)
        native = profile["native_producers"]
        stage = _native_stage_backend_plan_v27519(profile, native)
        assert native["backends"] == list(PRODUCERS)
        assert stage["split_backends"] == [producer]
        assert stage["execution_backends"] == [producer]
        assert stage["full_backends_by_producer"] == {producer: [producer, "tensorrt"]}
        rows = _native_expected_full_rows_v61b(
            ["model_a", "model_b"], stage["full_backends_by_producer"],
            {p: "setup_" + p for p in PRODUCERS},
        )
        assert len(rows) == 4
        assert {r["setup_id"] for r in rows} == {"setup_" + producer}


def test_legacy_full_map_remains_fallback_without_recognized_producer():
    profile = {
        "native_producers": {
            "enabled": True,
            "full_baselines": {"backends_by_producer": {"hailo10h": ["tensorrt"]}},
        },
        "run_profiles": [{"id": "custom_gpu_full", "full": "tensorrt"}],
    }
    plan = resolve_native_full_plan(profile)
    assert plan.enabled
    assert plan.active_producers == ("hailo10h",)
    assert plan.backends_by_producer == {"hailo10h": ("tensorrt",)}
