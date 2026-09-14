from __future__ import annotations

from onnx_splitpoint_tool.ranking_methods import (
    RANKING_METHOD_IMPLEMENTATION,
    WORKFLOW_RANKING_METHOD,
    compute_ranking_predictions,
    cut_bytes_only_sort_key,
    ranking_method_policy,
)
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _runner(profile: dict) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out="."))
    runner.profile_payload = profile
    return runner


def test_workflow_ranker_is_frozen_to_cut_bytes_for_all_targets() -> None:
    profiles = (
        {
            "selection_policy": {
                "selection_strategy": "stratified_windows",
                "objective": "Hailo feasibility",
            },
            "run_profiles": [
                {"id": "hailo8_to_trt", "stage1": "hailo8", "stage2": "tensorrt"}
            ],
        },
        {
            "selection_policy": {"selection_strategy": "stratified_windows"},
            "run_profiles": [
                {"id": "deepx_to_trt", "stage1": "deepx_m1", "stage2": "tensorrt"}
            ],
        },
    )

    assert WORKFLOW_RANKING_METHOD == "cut_bytes_only"
    assert RANKING_METHOD_IMPLEMENTATION == "v277-cut-bytes-only-workflow-freeze-1"
    default_policy = ranking_method_policy({})
    assert default_policy["implementation"] == RANKING_METHOD_IMPLEMENTATION
    assert default_policy["workflow_ranking_method"] == "cut_bytes_only"
    assert {
        _runner(profile)._selection_objective_for_targets()
        for profile in profiles
    } == {"cut_bytes_only"}


def test_cut_bytes_rank_is_ascending_then_boundary_then_case_id() -> None:
    candidates = [
        {"case_id": "case_z", "boundary": 3, "cut_bytes": 100, "strict_ok": True},
        {"case_id": "later_boundary", "boundary": 9, "cut_bytes": 100, "strict_ok": True},
        {"case_id": "smallest_payload", "boundary": 20, "cut_bytes": 50, "strict_ok": True},
        {"case_id": "case_a", "boundary": 3, "cut_bytes": 100, "strict_ok": True},
    ]
    expected = ["smallest_payload", "case_a", "case_z", "later_boundary"]

    assert [
        row["case_id"] for row in sorted(candidates, key=cut_bytes_only_sort_key)
    ] == expected
    assert [
        row["case_id"]
        for row in sorted(
            candidates,
            key=EvaluationWorkflowRunner._workflow_candidate_rank_key,
        )
    ] == expected

    rows = compute_ranking_predictions(candidates, [], {"methods": ["cut_bytes_only"]})
    assert [
        row["case_id"]
        for row in sorted(rows, key=lambda row: int(row["predicted_rank"]))
    ] == expected


def test_stratified_policy_values_are_not_changed_by_ranker_freeze() -> None:
    runner = _runner({
        "selection_policy": {
            "selection_strategy": "stratified_windows",
            "max_accepted_cases_per_model": 7,
            "preferred_shortlist": 23,
            "min_gap": 4,
            "candidate_search_pool": 41,
            "audit_size": 20,
            "audit_seed": 20260710,
        }
    })

    assert runner._selection_strategy() == "stratified_windows"
    assert runner._selection_numbers() == (7, 23, 4)
    assert runner._candidate_search_pool_size(23) == 41
    assert runner._selection_policy_payload()["audit_seed"] == 20260710


def test_stratified_windows_use_cut_bytes_not_legacy_source_rank() -> None:
    runner = _runner({
        "selection_policy": {
            "selection_strategy": "stratified_windows",
            "max_accepted_cases_per_model": 2,
            "preferred_shortlist": 4,
            "min_gap": 0,
        }
    })
    candidates = [
        {"case_id": "b002", "boundary": 2, "cut_bytes": 1000, "source_rank": 1},
        {"case_id": "b004", "boundary": 4, "cut_bytes": 100, "source_rank": 99},
        {"case_id": "b007", "boundary": 7, "cut_bytes": 300, "source_rank": 2},
        {"case_id": "b009", "boundary": 9, "cut_bytes": 200, "source_rank": 98},
    ]

    selected, _excluded, windows = runner._select_stratified_split_candidates(
        candidates,
        requested=2,
        min_gap=0,
        node_count=10,
    )

    assert [row["case_id"] for row in selected] == ["b004", "b009"]
    assert [row["selected_boundary"] for row in windows] == [4, 9]
