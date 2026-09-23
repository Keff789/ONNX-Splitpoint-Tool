from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.workflow.runner import (
    WORKFLOW_VERSION,
    EvaluationWorkflowRunner,
    _native_case_selection_decision,
    _native_model_preflight_rows,
    _native_model_selection_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v27917_identity_is_unique_and_consistent() -> None:
    assert package.__version__ == package.__release__ == VERSION == "2.79.17"
    assert (
        package.__build_id__
        == BUILD_ID
        == WORKFLOW_VERSION
        == "v2.79.17-native-energy-row-isolation-closure"
    )
    assert "v27917_release_identity_closure" in package.__build_features__


def test_missing_case_map_entry_blocks_only_that_model() -> None:
    runnable, effective, excluded, global_errors = (
        _native_case_selection_decision(
            ["resnet50", "yolov7_ultralytics"],
            case_policy="case_map_only",
            case_map={"resnet50": ["b001"]},
            available_case_map={
                "resnet50": ["b001"],
                "yolov7_ultralytics": ["b001"],
            },
        )
    )
    assert runnable == ["resnet50"]
    assert effective == {"resnet50": ["b001"]}
    assert global_errors == []
    assert excluded["yolov7_ultralytics"]["failure_reason"] == (
        "case_map_missing_requested_models:yolov7_ultralytics"
    )


@pytest.mark.parametrize(
    ("case_map", "expected_reason"),
    [
        ({"resnet50": ["b001"], "yolo26s": ["b002", "b002"]},
         "case_map_cases_duplicate:yolo26s"),
        ({"resnet50": ["b001"], "yolo26s": ["b999"]},
         "case_map_cases_missing:yolo26s:b999"),
    ],
)
def test_invalid_explicit_cases_are_model_local(
    case_map: dict[str, list[str]], expected_reason: str,
) -> None:
    runnable, effective, excluded, global_errors = (
        _native_case_selection_decision(
            ["resnet50", "yolo26s"],
            case_policy="case_map_only",
            case_map=case_map,
            available_case_map={
                "resnet50": ["b001"],
                "yolo26s": ["b002"],
            },
        )
    )
    assert runnable == ["resnet50"]
    assert effective == {"resnet50": ["b001"]}
    assert global_errors == []
    assert excluded["yolo26s"]["failure_reason"] == expected_reason


def test_discovered_cases_empty_is_model_local() -> None:
    runnable, effective, excluded, global_errors = (
        _native_case_selection_decision(
            ["resnet50", "yolo26s"],
            case_policy="all_accepted",
            case_map={},
            available_case_map={"resnet50": ["b001"], "yolo26s": []},
        )
    )
    assert runnable == ["resnet50"]
    assert effective == {}
    assert global_errors == []
    assert excluded["yolo26s"]["failure_reason"] == (
        "discovered_cases_empty:yolo26s"
    )


def test_unknown_case_map_model_remains_global_configuration_error() -> None:
    runnable, effective, excluded, global_errors = (
        _native_case_selection_decision(
            ["resnet50"],
            case_policy="case_map_only",
            case_map={"resnet50": ["b001"], "not_requested": ["b002"]},
            available_case_map={"resnet50": ["b001"]},
        )
    )
    assert runnable == ["resnet50"]
    assert effective == {"resnet50": ["b001"]}
    assert excluded == {}
    assert global_errors == ["case_map_unknown_models:not_requested"]


def test_all_model_local_failures_become_a_global_stop_only_when_none_run() -> None:
    exclusions = {
        "resnet50": {
            "failure_reason": "case_map_cases_empty:resnet50",
        },
        "yolo26s": {
            "failure_reason": "discovered_cases_empty:yolo26s",
        },
    }
    runnable, blocking = _native_model_selection_decision([], exclusions)
    assert runnable == []
    assert blocking == [
        "case_map_cases_empty:resnet50",
        "discovered_cases_empty:yolo26s",
    ]


def test_preflight_rows_project_the_actual_model_local_reason_to_energy() -> None:
    rows = _native_model_preflight_rows(
        ["resnet50"],
        {
            "yolo26s": {
                "category": "model_case_selection",
                "failure_reason": "case_map_cases_missing:yolo26s:b999",
            },
        },
        {"resnet50": {"valid": True}},
        energy_requested=True,
    )
    blocked = rows[0]
    assert blocked["category"] == "model_case_selection"
    assert blocked["failure_reason"] == "case_map_cases_missing:yolo26s:b999"
    assert blocked["energy_not_started_reason"] == (
        "Energy requested, not started: Native preflight blocked by "
        "case_map_cases_missing:yolo26s:b999"
    )
    assert rows[1]["model"] == "resnet50"
    assert rows[1]["status"] == "ready"


def test_stage_integrates_model_local_case_exclusions_before_remote_launch() -> None:
    source = inspect.getsource(EvaluationWorkflowRunner._stage_run_native_producers_steps)
    assert "_native_case_selection_decision(" in source
    assert "excluded_models[model] = exclusion" in source
    assert "models = case_runnable_models" in source
    assert '"configured_models": configured_models' in source


def test_profile_editor_explains_raw_unqualified_energy_semantics() -> None:
    source = (ROOT / "onnx_splitpoint_tool/gui/profile_editor.py").read_text(
        encoding="utf-8"
    )
    assert "Physisch erfasste Rohenergie bleibt" in source
    assert (
        "bei Quality-Problemen bleibt Rohenergie als nicht qualifiziert erhalten"
        in source
    )
    assert "nur nach erfolgreichem Contract-/Task-Gate" not in source


def test_current_release_aliases_point_to_v27917() -> None:
    assert "from .v27917_smoke import" in (
        ROOT / "onnx_splitpoint_tool/v279_smoke.py"
    ).read_text(encoding="utf-8")
    assert "run_v27917_small_acceptance.sh" in (
        ROOT / "scripts/run_v279_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    assert "run_v27917_small_acceptance.sh" in (
        ROOT / "scripts/run_local_acceptance.sh"
    ).read_text(encoding="utf-8")
