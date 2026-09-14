from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.run_modes import (
    native_full_backends_from_run_profiles,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import (
    _benchmark_runs_from_profile,
)
from onnx_splitpoint_tool.workflow.hardware_matrix import (
    run_profile_accelerators,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
import onnx_splitpoint_tool.workflow.runner as runner_module


def _runner(profile: dict[str, object]) -> EvaluationWorkflowRunner:
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.profile_payload = profile
    runner._profile_with_cli_hardware_overrides = lambda: profile
    return runner


def test_logical_matrix_outranks_stale_physical_targets() -> None:
    runner = _runner({
        "targets": ["hailo8"],
        "run_profiles": [{
            "id": "ort_tensorrt",
            "full": "tensorrt",
            "stage1": "tensorrt",
            "stage2": "tensorrt",
        }],
    })

    assert runner._targets() == ["tensorrt"]
    assert runner._baseline_backends(runner._targets()) == ["cuda_ort"]


def test_disabled_and_unknown_explicit_rows_never_manufacture_targets() -> None:
    runner = _runner({
        "run_profiles": [
            {
                "id": "hailo8_to_trt", "enabled": "false",
                "stage1": "hailo8", "stage2": "tensorrt",
            },
            {
                "id": "ort_tensorrt", "enabled": True,
                "full": "tensorrt", "stage1": "tensorrt",
                "stage2": "tensorrt",
            },
        ],
    })
    assert runner._targets() == ["tensorrt"]

    for rows in (
        [{"id": "hailo8", "enabled": False}],
        [{"id": "future_backend", "enabled": True}],
    ):
        explicit = _runner({"run_profiles": rows})
        assert explicit._targets() == []
        assert explicit._baseline_backends(explicit._targets()) == []


def test_legacy_target_and_hardware_fallbacks_require_no_logical_matrix(
    monkeypatch,
) -> None:
    legacy = _runner({"run_profiles": [], "targets": ["hailo8"]})
    assert legacy._targets() == ["hailo8"]
    assert legacy._baseline_backends(legacy._targets()) == ["hailo8"]

    fallback = _runner({})
    monkeypatch.setattr(
        runner_module,
        "normalize_hardware_targets",
        lambda _profile: [{"accelerator": "deepx_m1"}],
    )
    assert fallback._targets() == ["deepx_m1"]

    monkeypatch.setattr(
        runner_module, "normalize_hardware_targets", lambda _profile: [],
    )
    assert fallback._targets() == ["cpu_ort", "cuda_ort", "hailo8"]


def test_disabled_rows_are_filtered_by_all_logical_plan_consumers() -> None:
    rows = [
        {
            "id": "hailo8", "enabled": "false", "full": "hailo8",
            "stage1": "hailo8", "stage2": "hailo8",
        },
        {
            "id": "ort_tensorrt", "full": "tensorrt",
            "stage1": "tensorrt", "stage2": "tensorrt",
        },
    ]
    assert native_full_backends_from_run_profiles(rows) == ["tensorrt"]
    assert run_profile_accelerators(rows) == []
    assert [
        row["id"] for row in _benchmark_runs_from_profile(
            {"run_profiles": rows}, [],
        )
    ] == ["ort_tensorrt"]

    schema = json.loads(
        Path(
            "onnx_splitpoint_tool/resources/schemas/"
            "evaluation_profile.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["$defs"]["runProfile"]["properties"]["enabled"] == {
        "type": "boolean",
    }
