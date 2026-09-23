from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _native_expected_full_rows_v61b,
    _native_selection_contract_runs_v270e,
)


def _candidate(case_id: str, *, rank: int, part2_input_count: int) -> dict:
    boundary = int(case_id[1:])
    return {
        "case_id": case_id,
        "boundary": boundary,
        "split_index": boundary,
        "rank": rank,
        "source_rank": rank,
        # Since v2.77 the frozen production ranker is cut-bytes-only; rank is
        # retained as evidence.  Give this legacy fixture the metric it means
        # to order by so the test exercises backfill, not missing-metric
        # boundary-ID fallback.
        "cut_bytes": rank,
        "part2_input_count": part2_input_count,
        "part2_input_names": [
            f"{case_id}_input_{index}"
            for index in range(part2_input_count)
        ],
    }


def _runner(
    root: Path,
    *,
    forced_cases: Mapping[str, Sequence[str]],
) -> EvaluationWorkflowRunner:
    run_dir = root / "native-multi-input-policy"
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(root), run_id=run_dir.name)
    )
    runner.profile_id = "native_multi_input_policy"
    runner.run_id = run_dir.name
    runner.run_dir = run_dir
    runner.run_dir.mkdir(parents=True, exist_ok=True)
    runner.profile_payload = {
        "selection_policy": {
            "max_accepted_cases_per_model": 3,
            "preferred_shortlist": 5,
            "selection_strategy": "stratified_windows",
            "min_gap": 1,
            "candidate_search_pool": "auto",
            "require_single_part2_input": False,
            "forced_cases": {
                model: list(cases)
                for model, cases in forced_cases.items()
            },
        },
        "native_producers": {
            "enabled": True,
            "backends": ["hailo8", "hailo10h", "deepx"],
        },
    }
    return runner


def _select(
    runner: EvaluationWorkflowRunner,
    model_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[dict, dict, dict, str, str]:
    analysis_dir = runner.run_dir / "models" / model_id / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "prediction.json").write_text(
        json.dumps(
            {
                "schema": "onnx-splitpoint/split-prediction",
                "schema_version": 1,
                "artifact_id": f"prediction_{model_id}",
                "model_id": model_id,
                "node_count": 300,
                "policy_excluded_candidates": [],
                "candidates": [dict(candidate) for candidate in candidates],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifacts, metrics, message, status = (
        runner._stage_select_split_candidates(
            model_id,
            {
                "id": model_id,
                "family": model_id,
                "task": (
                    "classification"
                    if model_id == "resnet50"
                    else "detection"
                ),
                "evaluation_role": "development",
            },
        )
    )
    plan = json.loads(
        artifacts["final_candidate_plan_json"].read_text(encoding="utf-8")
    )
    selection_input = json.loads(
        artifacts["selection_input_json"].read_text(encoding="utf-8")
    )
    return plan, selection_input, dict(metrics), message, status


def _case_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("case_id") or row.get("case") or "") for row in rows]


def test_native_keeps_forced_multitensor_cases_in_requested_order(tmp_path: Path) -> None:
    selected = ["b116", "b044", "b216"]
    runner = _runner(tmp_path, forced_cases={"yolov7_paper": selected})
    candidates = [_candidate(case, rank=i + 1, part2_input_count=1 if case == "b044" else 3)
                  for i, case in enumerate(selected)]
    plan, selection_input, metrics, _, status = _select(runner, "yolov7_paper", candidates)
    assert status == "ok"
    assert _case_ids(plan["selected_candidates"]) == selected
    for payload in (plan, selection_input, metrics):
        assert payload["requested_require_single_part2_input"] is False
        assert payload["native_split_requires_single_part2_input"] is True
        assert payload["effective_require_single_part2_input"] is False
    assert plan["native_capability_backfills"] == []
    assert plan["native_capability_excluded_candidates"] == []


def test_native_selection_is_deterministic_when_prediction_order_changes(tmp_path: Path) -> None:
    candidates = [_candidate("b116", rank=2, part2_input_count=3),
                  _candidate("b044", rank=1, part2_input_count=1),
                  _candidate("b216", rank=3, part2_input_count=3)]
    runner = _runner(tmp_path, forced_cases={"yolov7_paper": ["b116", "b044", "b216"]})
    first = _select(runner, "yolov7_paper", candidates)[0]
    second = _select(runner, "yolov7_paper", list(reversed(candidates)))[0]
    assert first["selected_candidates"] == second["selected_candidates"]
    assert first["native_capability_backfills"] == second["native_capability_backfills"] == []


def test_native_does_not_fill_a_generic_shortfall(tmp_path: Path) -> None:
    runner = _runner(tmp_path, forced_cases={"yolov7_paper": ["b116", "b044", "b216"]})
    plan, _, metrics, message, status = _select(runner, "yolov7_paper",
        [_candidate("b116", rank=2, part2_input_count=3), _candidate("b044", rank=1, part2_input_count=1)])
    assert status == "partial"
    assert _case_ids(plan["selected_candidates"]) == ["b116", "b044"]
    assert metrics["selection_shortfall"] == 1 and "shortfall=1" in message
    assert plan["native_capability_backfills"] == []
