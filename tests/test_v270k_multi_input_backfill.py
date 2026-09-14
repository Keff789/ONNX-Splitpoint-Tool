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


def test_native_requested_off_is_effectively_single_input_with_overnight_backfill(
    tmp_path: Path,
) -> None:
    forced_cases = {
        "resnet50": ["b039", "b052", "b082"],
        "yolo26s": ["b038", "b142", "b036"],
        "yolov7_paper": ["b044", "b116", "b216"],
    }
    candidates = {
        "resnet50": [
            _candidate("b039", rank=1, part2_input_count=1),
            _candidate("b052", rank=2, part2_input_count=1),
            _candidate("b082", rank=3, part2_input_count=1),
        ],
        "yolo26s": [
            _candidate("b038", rank=1, part2_input_count=1),
            _candidate("b035", rank=2, part2_input_count=1),
            _candidate("b036", rank=3, part2_input_count=1),
            _candidate("b142", rank=39, part2_input_count=3),
        ],
        "yolov7_paper": [
            _candidate("b044", rank=1, part2_input_count=1),
            _candidate("b066", rank=2, part2_input_count=1),
            _candidate("b064", rank=3, part2_input_count=1),
            _candidate("b116", rank=10, part2_input_count=3),
            _candidate("b216", rank=11, part2_input_count=3),
        ],
    }
    runner = _runner(tmp_path, forced_cases=forced_cases)

    plans: dict[str, dict] = {}
    effective_case_map: dict[str, list[str]] = {}
    for model_id, model_candidates in candidates.items():
        plan, selection_input, metrics, _message, status = _select(
            runner, model_id, model_candidates
        )
        plans[model_id] = plan
        effective_case_map[model_id] = _case_ids(
            plan["selected_candidates"]
        )

        assert status == "ok"
        assert plan["requested_cases"] == 3
        assert metrics["selection_shortfall"] == 0
        assert metrics["deployment_shortlist_count"] == 3
        for payload in (plan, selection_input, metrics):
            assert payload["requested_require_single_part2_input"] is False
            assert payload["native_split_requires_single_part2_input"] is True
            assert payload["effective_require_single_part2_input"] is True
        assert (
            plan["native_multi_input_policy"]
            == "reject_and_backfill_from_frozen_prediction"
        )

    assert runner.profile_payload["selection_policy"][
        "require_single_part2_input"
    ] is False
    assert effective_case_map == {
        "resnet50": ["b039", "b052", "b082"],
        "yolo26s": ["b038", "b036", "b035"],
        "yolov7_paper": ["b044", "b066", "b064"],
    }

    capability_exclusions = {
        (
            model_id,
            str(row["case_id"]),
            int(row["observed_part2_input_count"]),
            str(row["exclude_reason"]),
            str(row["exclude_source"]),
        )
        for model_id, plan in plans.items()
        for row in plan["native_capability_excluded_candidates"]
    }
    assert capability_exclusions == {
        (
            "yolo26s",
            "b142",
            3,
            "part2_input_count_not_one",
            "native_split_capability",
        ),
        (
            "yolov7_paper",
            "b116",
            3,
            "part2_input_count_not_one",
            "native_split_capability",
        ),
        (
            "yolov7_paper",
            "b216",
            3,
            "part2_input_count_not_one",
            "native_split_capability",
        ),
    }
    backfill_pairs = {
        (
            model_id,
            str(row["case_id"]),
            str(row["native_backfill_replaces_case"]),
            str(row["native_backfill_scope"]),
        )
        for model_id, plan in plans.items()
        for row in plan["native_capability_backfills"]
    }
    assert backfill_pairs == {
        (
            "yolo26s",
            "b035",
            "b142",
            "same_stratified_window_or_global_rank_fallback",
        ),
        (
            "yolov7_paper",
            "b066",
            "b116",
            "same_stratified_window_or_global_rank_fallback",
        ),
        (
            "yolov7_paper",
            "b064",
            "b216",
            "same_stratified_window_or_global_rank_fallback",
        ),
    }

    split_row_count = 0
    for backend in ("hailo8", "hailo10h", "deepx"):
        contracts = _native_selection_contract_runs_v270e(
            backend, effective_case_map
        )
        split_row_count += sum(
            len(cases)
            for contract in contracts
            for cases in contract["case_map"].values()
        )
    full_rows = _native_expected_full_rows_v61b(
        list(effective_case_map),
        {
            "hailo8": ["hailo8", "tensorrt"],
            "hailo10h": ["hailo10h", "tensorrt"],
            "deepx": ["deepx", "tensorrt"],
        },
        {
            "hailo8": "hailo8-setup",
            "hailo10h": "hailo10h-setup",
            "deepx": "deepx-setup",
        },
    )
    assert split_row_count == 27
    assert len(full_rows) == 18
    assert split_row_count + len(full_rows) == 45


def test_native_backfill_is_deterministic_when_prediction_order_changes(
    tmp_path: Path,
) -> None:
    forced_cases = {
        "yolov7_paper": ["b044", "b116", "b216"],
    }
    candidates = [
        _candidate("b044", rank=1, part2_input_count=1),
        _candidate("b066", rank=2, part2_input_count=1),
        _candidate("b064", rank=3, part2_input_count=1),
        _candidate("b116", rank=10, part2_input_count=3),
        _candidate("b216", rank=11, part2_input_count=3),
    ]

    observed: list[tuple[list[str], list[tuple[str, str]], str]] = []
    for index, prediction_rows in enumerate((candidates, list(reversed(candidates)))):
        runner = _runner(
            tmp_path / f"order-{index}",
            forced_cases=forced_cases,
        )
        plan, _selection_input, metrics, _message, status = _select(
            runner, "yolov7_paper", prediction_rows
        )
        assert status == "ok"
        assert metrics["selection_shortfall"] == 0
        observed.append(
            (
                _case_ids(plan["selected_candidates"]),
                [
                    (
                        str(row["case_id"]),
                        str(row["native_backfill_replaces_case"]),
                    )
                    for row in plan["native_capability_backfills"]
                ],
                str(plan["artifact_id"]),
            )
        )

    assert observed[0] == observed[1]
    assert observed[0][0] == ["b044", "b066", "b064"]
    assert observed[0][1] == [
        ("b066", "b116"),
        ("b064", "b216"),
    ]


def test_native_backfill_shortfall_keeps_configured_target_visible(
    tmp_path: Path,
) -> None:
    runner = _runner(
        tmp_path,
        forced_cases={
            "yolov7_paper": ["b044", "b116", "b216"],
        },
    )
    plan, selection_input, metrics, message, status = _select(
        runner,
        "yolov7_paper",
        [
            _candidate("b044", rank=1, part2_input_count=1),
            _candidate("b066", rank=2, part2_input_count=1),
            _candidate("b116", rank=10, part2_input_count=3),
            _candidate("b216", rank=11, part2_input_count=3),
        ],
    )

    assert status == "partial"
    assert plan["requested_cases"] == 3
    assert selection_input["requested_cases"] == 3
    assert metrics["deployment_shortlist_count"] == 2
    assert metrics["selection_shortfall"] == 1
    assert _case_ids(plan["selected_candidates"]) == ["b044", "b066"]
    assert _case_ids(plan["native_capability_excluded_candidates"]) == [
        "b116",
        "b216",
    ]
    assert [
        (
            row["case_id"],
            row["native_backfill_replaces_case"],
        )
        for row in plan["native_capability_backfills"]
    ] == [("b066", "b116")]
    assert "2/3 requested deployment cases" in message
    assert "shortfall=1" in message
