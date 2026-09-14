from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow import scientific_reporting
from onnx_splitpoint_tool.campaign import create_candidate_universe_manifest
from onnx_splitpoint_tool.native_performance_reporting import (
    collect_native_performance_matrix,
)
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _load_prediction_freeze,
    _merge_native_ranking_rows,
    _native_ranking_audit_summary,
    _ranking_audit_request,
    _ranking_method_comparison,
)
from onnx_splitpoint_tool.workflow.artifacts import sha256_file, write_csv
from onnx_splitpoint_tool.workflow.start_snapshot import (
    StartSnapshotConsistencyError,
    build_profile_start_snapshot,
    profile_selection_view,
)


def _candidate(index: int, *, score: float | None = None) -> dict:
    row = {
        "case_id": f"b{index:03d}",
        "boundary": index,
        "split_index": index,
        "rank": 101 - index,
        "source_rank": 101 - index,
        "strict_ok": True,
        "part2_input_count": 1,
        "cut_bytes": index * 4096,
        "imbalance_val": abs(50 - index) / 50.0,
        "n_cut_tensors": 1 + index % 3,
    }
    if score is not None:
        row["accelerator_fit_score"] = score
    return row


def _runner(root: Path) -> EvaluationWorkflowRunner:
    runner = EvaluationWorkflowRunner(
        WorkflowOptions(profile="", out=str(root), run_id="ranking-audit")
    )
    runner.run_id = "ranking-audit"
    runner.run_dir = root / runner.run_id
    runner.run_dir.mkdir(parents=True)
    runner.profile_payload = {
        "selection_policy": {
            "max_accepted_cases_per_model": 4,
            "preferred_shortlist": 5,
            "min_gap": 0,
            "selection_strategy": "score_independent_audit",
            "score_independent_audit_enabled": True,
            "audit_size": 20,
            "minimum_valid_audit_candidates": 10,
            "audit_seed": 20260710,
        },
        "native_producers": {"enabled": True, "backends": ["hailo8"]},
        "ranking_validation": {"enabled": True},
    }
    return runner


def _write_inputs(runner: EvaluationWorkflowRunner, candidates: list[dict]) -> None:
    analysis = runner.run_dir / "models" / "resnet50" / "analysis"
    analysis.mkdir(parents=True)
    prediction = analysis / "prediction.json"
    prediction.write_text(json.dumps({
        "model_id": "resnet50", "node_count": 100, "candidates": candidates,
        "policy_excluded_candidates": [],
    }), encoding="utf-8")
    create_candidate_universe_manifest(
        model_id="resnet50", candidates=candidates,
        mode="deterministic_audit", output_dir=analysis,
        audit_size=20, minimum_valid_candidates=10, seed=20260710,
    )


def _selected(root: Path, candidates: list[dict]) -> dict:
    runner = _runner(root)
    _write_inputs(runner, candidates)
    artifacts, metrics, _message, status = runner._stage_select_split_candidates(
        "resnet50", {"id": "resnet50", "task": "classification", "evaluation_role": "development"}
    )
    assert status == "ok"
    plan = json.loads(artifacts["final_candidate_plan_json"].read_text(encoding="utf-8"))
    assert metrics["audit_candidate_count"] == 20
    assert metrics["minimum_valid_audit_candidates"] == 10
    assert metrics["deployment_shortlist_count"] == 4
    return plan


def test_development_audit_is_score_independent_and_unioned_with_shortlist(tmp_path: Path) -> None:
    baseline = [_candidate(index, score=float(index)) for index in range(1, 101)]
    reversed_scores = [_candidate(index, score=float(101 - index)) for index in range(1, 101)]
    first = _selected(tmp_path / "first", baseline)
    second = _selected(tmp_path / "second", reversed_scores)

    first_audit = [row["case_id"] for row in first["audit_candidates"]]
    second_audit = [row["case_id"] for row in second["audit_candidates"]]
    assert first_audit == second_audit
    assert len(first_audit) == len(set(first_audit)) == 20
    assert first["score_independent"] is True
    assert first["selection_uses_predictions"] is False
    assert first["selection_uses_measurements"] is False
    assert all("audit" in row["candidate_execution_roles"] for row in first["audit_candidates"])
    assert len(first["selected_candidates"]) <= 24


def test_forced_audit_anchor_executes_first_without_changing_audit(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(index, score=float(index))
        for index in range(1, 101)
    ]
    baseline = _selected(tmp_path / "baseline", candidates)
    baseline_audit = [
        row["case_id"] for row in baseline["audit_candidates"]
    ]
    assert "b067" in baseline_audit

    runner = _runner(tmp_path / "forced")
    runner.profile_payload["selection_policy"][
        "max_accepted_cases_per_model"
    ] = 1
    runner.profile_payload["selection_policy"]["forced_cases"] = {
        "resnet50": ["b067"],
    }
    _write_inputs(runner, candidates)
    artifacts, metrics, _message, status = (
        runner._stage_select_split_candidates(
            "resnet50",
            {
                "id": "resnet50",
                "task": "classification",
                "evaluation_role": "development",
            },
        )
    )

    assert status == "ok"
    plan = json.loads(
        artifacts["final_candidate_plan_json"].read_text(encoding="utf-8")
    )
    selection_input = json.loads(
        artifacts["selection_input_json"].read_text(encoding="utf-8")
    )
    audit_plan = json.loads(
        artifacts["audit_plan_json"].read_text(encoding="utf-8")
    )
    assert [row["case_id"] for row in plan["audit_candidates"]] == (
        baseline_audit
    )
    assert [row["case_id"] for row in plan["deployment_shortlist"]] == [
        "b067"
    ]
    assert plan["selected_candidates"][0]["case_id"] == "b067"
    assert set(
        plan["selected_candidates"][0]["candidate_execution_roles"]
    ) == {"audit", "deployment_shortlist"}
    assert len(plan["selected_candidates"]) == len(baseline_audit)
    assert metrics["audit_candidate_count"] == len(baseline_audit)
    for evidence in (plan, selection_input, audit_plan):
        assert evidence["execution_union_order_policy"] == (
            "forced_deployment_then_audit"
        )


def test_forced_audit_anchor_missing_from_frozen_prediction_fails_closed(
    tmp_path: Path,
) -> None:
    candidates = [
        _candidate(index, score=float(index))
        for index in range(1, 101)
    ]
    runner = _runner(tmp_path)
    runner.profile_payload["selection_policy"]["forced_cases"] = {
        "resnet50": ["b999"],
    }
    _write_inputs(runner, candidates)

    with pytest.raises(
        ValueError,
        match="absent from the capability-eligible frozen prediction",
    ):
        runner._stage_select_split_candidates(
            "resnet50",
            {
                "id": "resnet50",
                "task": "classification",
                "evaluation_role": "development",
            },
        )


def test_candidate_universe_policy_enables_development_audit(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    policy = runner._candidate_universe_policy_for_model({"evaluation_role": "development"})
    assert policy == {
        "mode": "deterministic_audit",
        "audit_size": 20,
        "minimum_valid_candidates": 10,
        "seed": 20260710,
        "development_audit": True,
    }


def _ranking_contract() -> tuple[dict, dict, dict]:
    audit_ids = ["b001", "b002"]
    predictions = {
        "resnet50": {
            "_ranking_method_predictions": [
                {
                    "method_id": "cut_bytes_only",
                    "case_id": case_id,
                    "direction": "hailo8_to_tensorrt",
                    "runner_regime": "native_fifo",
                    "prediction_available": True,
                    "predicted_value": float(index),
                    "predicted_rank": index,
                    "prediction_unit": "bytes",
                }
                for index, case_id in enumerate([*audit_ids, "b999"], start=1)
            ],
            "_prediction_freeze": {
                "valid": True,
                "ranking_predictions_valid": True,
                "status": "prospective_frozen",
                "candidate_universe_scope": "predeclared_audit_universe",
                "candidate_universe_selected_case_ids": audit_ids,
                "candidate_universe_minimum_valid_candidates": 2,
                "candidate_universe_valid": True,
                "candidate_universe_complete": True,
            },
        }
    }
    profile = {
        "model_suite": {
            "primary": [
                {
                    "id": "resnet50",
                    "evaluation_role": "development",
                }
            ]
        }
    }
    policy = {
        "methods": ["cut_bytes_only"],
        "k_values": [1],
        "elite_q_values": [1],
        "primary_k": 1,
        "minimum_candidates_for_correlation": 2,
        "require_complete_candidate_universe": True,
        "require_frozen_predictions": True,
        "near_optimal_relative_epsilon": 0.01,
        "method_policy": {},
    }
    return predictions, profile, policy


def _native_ranking_row(
    case_id: str,
    cycle_ms: float,
    *,
    setup_id: str = "orin-hailo8",
    precision: str = "hailo_hef_sha256:abc",
    endpoint: str = "classification_logits",
) -> dict:
    return {
        "model_id": "resnet50",
        "evaluation_role": "development",
        "case_id": case_id,
        "variant": "native_split",
        "direction": "hailo8_to_tensorrt",
        "runner_regime": "native_fifo",
        "setup_id": setup_id,
        "runtime_precision_identity": precision,
        "comparison_backend": "tensorrt",
        "comparison_output_endpoint_id": endpoint,
        "completed_task_comparison_endpoint_contract_hash": "e" * 64,
        "e2e_scope": "full_task_pipeline",
        "comparison_endpoint_stratum": "completed_task",
        "measurement_concurrency": "pipeline_steady_state",
        "completed_task_stage": "classification_logits",
        "completed_task_completion_mode": "runtime_output",
        "frozen_host_postprocess_contract_sha256": "h" * 64,
        "host_postprocessing_available": True,
        "host_postprocess_required": False,
        "native_measured_cycle_ms": cycle_ms,
        "ranking_eligible": True,
        "runtime_executable": True,
    }


def _producer_shaped_native_ranking_row(
    case_id: str,
    cycle_ms: float,
    *,
    endpoint: str = "classification:classification_logits:shared",
    endpoint_hash: str = "e" * 64,
) -> dict:
    row = _native_ranking_row(case_id, cycle_ms, endpoint=endpoint)
    row.update({
        # This is the shape emitted by the real producer in v2.75.31: the
        # explicit completed-task comparison stratum is blank, while the
        # legacy observation identity embeds the case id.
        "comparison_endpoint_stratum": "",
        "output_endpoint_comparison_stratum": [
            "resnet50", case_id, "native_split",
        ],
        "completed_task_comparison_output_endpoint_id": endpoint,
        "completed_task_comparison_endpoint_contract_hash": endpoint_hash,
        "host_postprocessing_available": False,
        "host_postprocess_required": False,
    })
    return row


def _audit_contract_for_cases(
    case_ids: list[str],
    *,
    runner: str = "native_fifo",
    minimum: int = 3,
) -> tuple[dict, dict, dict]:
    predictions, profile, policy = _ranking_contract()
    predictions["resnet50"]["_ranking_method_predictions"] = [
        {
            "method_id": "cut_bytes_only",
            "case_id": case_id,
            "direction": "hailo8_to_tensorrt",
            "runner_regime": runner,
            "prediction_available": True,
            "predicted_value": float(index),
            "predicted_rank": index,
            "prediction_unit": "bytes",
        }
        for index, case_id in enumerate(case_ids, start=1)
    ]
    freeze = predictions["resnet50"]["_prediction_freeze"]
    freeze["candidate_universe_selected_case_ids"] = list(case_ids)
    freeze["candidate_universe_minimum_valid_candidates"] = minimum
    return predictions, profile, policy


def test_development_audit_excludes_deployment_shortlist_rows() -> None:
    predictions, profile, policy = _ranking_contract()
    rows = [
        _native_ranking_row("b001", 10.0),
        _native_ranking_row("b002", 20.0),
        _native_ranking_row("b999", 0.01),
    ]

    details, _macro, _summary = _ranking_method_comparison(
        rows, predictions, profile, policy
    )

    assert len(details) == 1
    result = details[0]
    assert result["audit_scope_enforced"] is True
    assert result["unscoped_measured_candidate_count"] == 3
    assert result["measured_candidate_count"] == 2
    assert result["deployment_only_measurement_count_excluded"] == 1
    assert result["expected_candidate_count"] == 2
    assert result["candidate_universe_complete"] is True
    assert result["best_measured_case"] == "b001"
    assert result["status"] == "development_evidence_only"


def test_real_producer_case_local_endpoint_identity_groups_four_candidates() -> None:
    case_ids = ["b024", "b052", "b067", "b095"]
    predictions, profile, policy = _audit_contract_for_cases(case_ids)
    rows = [
        _producer_shaped_native_ranking_row(case_id, cycle_ms)
        for case_id, cycle_ms in zip(case_ids, (10.0, 8.0, 6.0, 7.0))
    ]

    details, _macro, _summary = _ranking_method_comparison(
        rows, predictions, profile, policy
    )

    assert len(details) == 1
    assert details[0]["measured_candidate_count"] == 4
    assert details[0]["valid_candidate_count"] == 4
    assert details[0]["paired_candidate_count"] == 4
    assert details[0]["comparison_endpoint_stratum"] == ""
    assert details[0]["status"] == "development_evidence_only"


def test_case_local_identity_does_not_collapse_incompatible_completed_tasks() -> None:
    case_ids = ["b024", "b052", "b067", "b095"]
    predictions, profile, policy = _audit_contract_for_cases(case_ids)
    rows = [
        _producer_shaped_native_ranking_row("b024", 10.0),
        _producer_shaped_native_ranking_row("b052", 8.0),
        _producer_shaped_native_ranking_row(
            "b067",
            6.0,
            endpoint="detection:decoded_nms:other",
            endpoint_hash="d" * 64,
        ),
        _producer_shaped_native_ranking_row("b095", 7.0),
    ]
    rows[-1].update({
        "host_postprocess_required": True,
        "host_postprocessing_available": True,
        "frozen_host_postprocess_contract_sha256": "h" * 64,
    })

    details, _macro, _summary = _ranking_method_comparison(
        rows, predictions, profile, policy
    )

    assert len(details) == 3
    assert sorted(row["measured_candidate_count"] for row in details) == [1, 1, 2]
    assert len({row["comparison_output_endpoint_id"] for row in details}) == 2
    assert len({row["host_postprocess_required"] for row in details}) == 2


def _generic_screening_ranking_row(
    case_id: str,
    cycle_ms: float,
    *,
    decision: str = "pass",
    extra_exclusion: str = "",
) -> dict:
    exclusions = ["screening_only"]
    if extra_exclusion:
        exclusions.append(extra_exclusion)
    passed = decision == "pass"
    return {
        "model_id": "resnet50",
        "evaluation_role": "development",
        "case_id": case_id,
        "variant": "split",
        "direction": "hailo8_to_tensorrt",
        "runner_regime": "generic",
        "setup_id": "orin-hailo8",
        "runtime_precision_identity": "fp16",
        "comparison_backend": "tensorrt",
        "comparison_output_endpoint_id": "classification_logits",
        "completed_task_comparison_endpoint_contract_hash": "e" * 64,
        "e2e_scope": "full_task_pipeline",
        "comparison_endpoint_stratum": "completed_task",
        "measurement_concurrency": "sequential",
        "completed_task_stage": "classification_logits",
        "completed_task_completion_mode": "runtime_output",
        "host_postprocessing_available": False,
        "host_postprocess_required": False,
        "cycle_ms": cycle_ms,
        "buildable": True,
        "runtime_executable": True,
        "contract_consistent": True,
        "accuracy_gate_pass": passed,
        "accuracy_gate_policy_match": True,
        "task_quality_decision": decision,
        "task_quality_status": decision,
        "ranking_eligible": False,
        "ranking_exclusion_reason": "screening_only",
        "exclusion_reason": "screening_only",
        "accuracy_gate_reason": "screening_only",
        "gate_status": "screening_only",
        "performance_claim_eligible": False,
        "performance_claim_exclusion_reasons": exclusions,
    }


def test_generic_standard_screening_pass_is_development_audit_eligible_only() -> None:
    case_ids = ["b001", "b002", "b003", "b004", "b005"]
    predictions, profile, policy = _audit_contract_for_cases(
        case_ids,
        runner="generic",
        minimum=2,
    )
    profile["execution_preset"] = {"id": "standard"}
    profile["selection_policy"] = {
        "selection_strategy": "score_independent_audit",
        "score_independent_audit_enabled": True,
    }
    rows = [
        _generic_screening_ranking_row("b001", 10.0),
        _generic_screening_ranking_row("b002", 8.0),
        _generic_screening_ranking_row("b003", 6.0, decision="fail"),
        _generic_screening_ranking_row(
            "b004", 7.0, decision="inconclusive"
        ),
        _generic_screening_ranking_row(
            "b005", 5.0, extra_exclusion="unbound_provenance"
        ),
    ]

    details, _macro, _summary = _ranking_method_comparison(
        rows, predictions, profile, policy
    )

    assert len(details) == 1
    result = details[0]
    assert result["valid_candidate_count"] == 2
    assert result["development_screening_exception_candidate_count"] == 2
    assert result["quality_vetoed_candidate_count"] == 2
    assert result["best_measured_case"] == "b002"
    assert result["status"] == "development_evidence_only"
    # The reporting-only exception must never upgrade final claim fields.
    assert all(row["performance_claim_eligible"] is False for row in rows)
    assert all(row["ranking_eligible"] is False for row in rows)

    holdout_profile = copy.deepcopy(profile)
    holdout_profile["model_suite"]["primary"][0][
        "evaluation_role"
    ] = "confirmatory_holdout"
    holdout_details, _macro, _summary = _ranking_method_comparison(
        rows, predictions, holdout_profile, policy
    )
    assert holdout_details[0]["valid_candidate_count"] == 0
    assert holdout_details[0]["status"] == "insufficient_valid_audit_candidates"


def test_generic_screening_exception_requires_exact_audit_scope_and_mode() -> None:
    case_ids = ["b001", "b002"]
    predictions, profile, policy = _audit_contract_for_cases(
        case_ids,
        runner="generic",
        minimum=2,
    )
    profile["execution_preset"] = {"id": "standard"}
    profile["selection_policy"] = {
        "selection_strategy": "score_independent_audit",
        "score_independent_audit_enabled": True,
    }
    rows = [
        _generic_screening_ranking_row("b001", 10.0),
        _generic_screening_ranking_row("b002", 8.0),
    ]

    variants: list[tuple[str, dict, dict]] = []

    missing_mode = copy.deepcopy(profile)
    missing_mode.pop("execution_preset")
    variants.append(("missing resolved run-mode id", missing_mode, predictions))

    final_mode = copy.deepcopy(profile)
    final_mode["execution_preset"] = {"id": "final"}
    variants.append(("non-standard run mode", final_mode, predictions))

    stratified = copy.deepcopy(profile)
    stratified["selection_policy"]["selection_strategy"] = (
        "stratified_windows"
    )
    stratified["selection_policy"]["score_independent_audit_enabled"] = False
    variants.append(("non-audit selection", stratified, predictions))

    contradictory = copy.deepcopy(profile)
    contradictory["selection_policy"][
        "score_independent_audit_enabled"
    ] = False
    variants.append(("explicitly disabled audit", contradictory, predictions))

    unscoped_predictions = copy.deepcopy(predictions)
    unscoped_predictions["resnet50"]["_prediction_freeze"][
        "candidate_universe_scope"
    ] = "complete_feasible_universe"
    variants.append(("non-audit candidate scope", profile, unscoped_predictions))

    for label, variant_profile, variant_predictions in variants:
        details, _macro, _summary = _ranking_method_comparison(
            rows,
            variant_predictions,
            variant_profile,
            policy,
        )
        assert len(details) == 1, label
        assert details[0]["valid_candidate_count"] == 0, label
        assert (
            details[0][
                "development_screening_exception_candidate_count"
            ]
            == 0
        ), label


def test_native_ranking_uses_complete_measurement_stratum() -> None:
    predictions, profile, policy = _ranking_contract()
    rows = [
        _native_ranking_row("b001", 10.0, setup_id="setup-a"),
        _native_ranking_row("b002", 20.0, setup_id="setup-a"),
        _native_ranking_row(
            "b001",
            40.0,
            setup_id="setup-b",
            precision="fp16",
            endpoint="decoded_nms",
        ),
        _native_ranking_row(
            "b002",
            5.0,
            setup_id="setup-b",
            precision="fp16",
            endpoint="decoded_nms",
        ),
    ]

    details, _macro, _summary = _ranking_method_comparison(
        rows, predictions, profile, policy
    )

    assert len(details) == 2
    by_setup = {row["setup_id"]: row for row in details}
    assert set(by_setup) == {"setup-a", "setup-b"}
    assert by_setup["setup-a"]["best_measured_case"] == "b001"
    assert by_setup["setup-b"]["best_measured_case"] == "b002"
    assert by_setup["setup-a"]["runtime_precision_identity"] == (
        "hailo_hef_sha256:abc"
    )
    assert by_setup["setup-b"]["runtime_precision_identity"] == "fp16"
    assert by_setup["setup-a"]["comparison_output_endpoint_id"] == (
        "classification_logits"
    )
    assert by_setup["setup-b"]["comparison_output_endpoint_id"] == (
        "decoded_nms"
    )


def test_native_audit_top_level_is_not_available_for_non_audit_ranking() -> None:
    summary = _native_ranking_audit_summary(
        [
            {
                "runner_regime": "native_fifo",
                "audit_scope_enforced": False,
                "candidate_universe_scope": "",
                "status": "development_evidence_only",
                "spearman_rho": 0.9,
                "hit_at_1": True,
            }
        ],
        {"requested": False, "source": "none"},
    )

    assert summary["status"] == "not_requested"
    assert summary["audit_requested"] is False
    assert summary["audit_scope"] == "not_requested"
    assert summary["usable_metric_row_count"] == 0


def test_native_audit_top_level_reports_quality_veto_shortfall() -> None:
    summary = _native_ranking_audit_summary(
        [
            {
                "runner_regime": "native_fifo",
                "audit_scope_enforced": True,
                "candidate_universe_scope": "predeclared_audit_universe",
                "status": "insufficient_valid_audit_candidates",
                "measured_candidate_count": 4,
                "valid_candidate_count": 0,
                "quality_vetoed_candidate_count": 4,
                "minimum_valid_candidates_required": 3,
                "spearman_rho": None,
                "hit_at_1": None,
                "regret_at_1": None,
            }
        ],
        {"requested": True, "source": "selection_plan+prediction_freeze"},
    )

    assert summary["status"] == (
        "insufficient_valid_candidates_after_quality_veto"
    )
    assert summary["audit_requested"] is True
    assert summary["audit_scope"] == (
        "predeclared_score_independent_candidates"
    )
    assert summary["usable_metric_row_count"] == 0
    assert summary["quality_veto_shortfall_group_method_row_count"] == 1


def test_native_audit_top_level_requires_a_usable_metric_for_available() -> None:
    common = {
        "runner_regime": "native_fifo",
        "audit_scope_enforced": True,
        "candidate_universe_scope": "predeclared_audit_universe",
        "status": "development_evidence_only",
        "measured_candidate_count": 4,
        "valid_candidate_count": 4,
        "minimum_valid_candidates_required": 3,
    }
    unavailable = _native_ranking_audit_summary(
        [common], {"requested": True, "source": "prediction_freeze"}
    )
    available = _native_ranking_audit_summary(
        [{**common, "hit_at_1": True}],
        {"requested": True, "source": "prediction_freeze"},
    )
    invalid = _native_ranking_audit_summary(
        [{**common, "status": "predictions_not_frozen", "hit_at_1": True}],
        {"requested": True, "source": "selection_plan"},
    )

    assert unavailable["status"] == "requested_but_no_usable_metrics"
    assert available["status"] == "available"
    assert available["usable_metric_row_count"] == 1
    assert invalid["status"] == "requested_but_no_usable_metrics"


def test_ranking_audit_request_uses_plan_aliases_and_prediction_freeze() -> None:
    for strategy in (
        "score_independent_audit",
        "score-independent-audit",
        "ranking_audit",
        "deterministic_audit",
    ):
        request = _ranking_audit_request(
            {"selection_policy": {"selection_strategy": strategy}}, {}
        )
        assert request["requested"] is True
        assert request["source"] == "selection_plan"

    request = _ranking_audit_request(
        {"selection_policy": {"selection_strategy": "stratified_windows"}},
        {
            "resnet50": {
                "_prediction_freeze": {
                    "candidate_universe_scope": (
                        "predeclared_audit_universe"
                    )
                }
            }
        },
    )
    assert request["requested"] is True
    assert request["source"] == "prediction_freeze"
    assert request["frozen_audit_model_ids"] == ["resnet50"]


def test_partial_native_enrichment_never_collapses_conflicting_strata() -> None:
    endpoint_a = _native_ranking_row(
        "b001", 10.0, endpoint="classification_logits"
    )
    endpoint_b = _native_ranking_row(
        "b001", 10.0, endpoint="decoded_nms"
    )
    partial_matrix_row = {
        key: value
        for key, value in endpoint_a.items()
        if key != "comparison_output_endpoint_id"
    }

    merged = _merge_native_ranking_rows(
        [endpoint_a, endpoint_b], [partial_matrix_row]
    )

    assert len(merged) == 2
    assert {
        row["comparison_output_endpoint_id"] for row in merged
    } == {"classification_logits", "decoded_nms"}


def test_prediction_freeze_manifest_completeness_overrides_universe_fallback(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "models" / "resnet50"
    analysis = model_dir / "analysis"
    analysis.mkdir(parents=True)
    candidates = [_candidate(index) for index in range(1, 4)]
    prediction_path = analysis / "prediction.json"
    prediction_path.write_text(
        json.dumps({"model_id": "resnet50", "candidates": candidates}),
        encoding="utf-8",
    )
    prediction_csv = write_csv(analysis / "predictions_frozen.csv", candidates)
    ranking_csv = write_csv(
        analysis / "ranking_predictions_frozen.csv",
        [
            {
                "method_id": "cut_bytes_only",
                "case_id": row["case_id"],
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "native_fifo",
                "prediction_available": True,
                "predicted_value": row["cut_bytes"],
            }
            for row in candidates
        ],
    )
    universe_path, universe_csv, universe = create_candidate_universe_manifest(
        model_id="resnet50",
        candidates=candidates,
        mode="deterministic_audit",
        output_dir=analysis,
        audit_size=3,
        minimum_valid_candidates=2,
        source_prediction_sha256=str(sha256_file(prediction_path) or ""),
    )
    assert universe["declared_complete"] is True
    _write_json(
        analysis / "prediction_freeze_manifest.json",
        {
            "prospective": True,
            "prediction_sha256": sha256_file(prediction_path),
            "prediction_csv": prediction_csv.name,
            "prediction_csv_sha256": sha256_file(prediction_csv),
            "ranking_prediction_csv": ranking_csv.name,
            "ranking_prediction_csv_sha256": sha256_file(ranking_csv),
            "candidate_universe_manifest": universe_path.name,
            "candidate_universe_sha256": universe["universe_sha256"],
            "candidate_universe_csv": universe_csv.name,
            "candidate_universe_csv_sha256": sha256_file(universe_csv),
            # Frozen execution result is authoritative over the manifest's
            # compatibility-level declared_complete fallback.
            "candidate_universe_complete": False,
        },
    )

    freeze = _load_prediction_freeze(model_dir, prediction_path)

    assert freeze["valid"] is True
    assert freeze["candidate_universe_valid"] is True
    assert freeze["candidate_universe_complete"] is False


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_native_quality_veto_is_row_local_and_fail_closed(tmp_path: Path) -> None:
    common = {
        "model": "resnet50",
        "backend": "hailo8_to_trt",
        "execution_mode": "native_split",
        "status": "ok",
        "ok": True,
        "setup_id": "orin-hailo8",
        "comparison_backend": "tensorrt",
        "runtime_precision_identity": "hailo_hef_sha256:abc",
        "precision": "uint8",
        "quality_central_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "task_quality_pass": True,
        "quality_claim_result_verified": False,
        "quality_accuracy_gate_pass": True,
        "quality_eligible_for_ranking": False,
        "quality_gate_status": "screening_only",
        "quality_ranking_exclusion_reason": "screening_only",
    }
    passing = {**common, "case": "b001", "fps_median": 100.0}
    vetoed = {
        **common,
        "case": "b002",
        "fps_median": 80.0,
        "task_quality_pass": False,
        "quality_accuracy_gate_pass": False,
        "quality_gate_status": "fail",
        "quality_ranking_exclusion_reason": "accuracy_gate_failed",
    }
    _write_json(
        tmp_path / "reports" / "native_producer_combined_summary.json",
        {"rows": [passing, vetoed]},
    )

    observations = collect_native_performance_matrix(tmp_path)["observations"]
    by_case = {row["case_id"]: row for row in observations}

    assert by_case["b001"]["ranking_eligible"] is True
    assert by_case["b001"]["quality_claim_result_verified"] is False
    assert by_case["b001"]["quality_eligible_for_ranking"] is False
    assert by_case["b001"]["quality_ranking_exclusion_reason"] == (
        "screening_only"
    )
    assert by_case["b001"]["native_measured_cycle_ms"] == 10.0
    assert by_case["b001"]["runtime_precision_identity"] == (
        "hailo_hef_sha256:abc"
    )
    assert by_case["b002"]["ranking_eligible"] is False
    assert by_case["b002"]["quality_gate_status"] == "fail"
    assert by_case["b002"]["quality_ranking_exclusion_reason"] == (
        "accuracy_gate_failed"
    )
    assert "quality_accuracy_gate_not_passed" in by_case["b002"][
        "ranking_exclusion_reasons"
    ]


def test_explicit_inconclusive_quality_status_vetoes_positive_booleans(
    tmp_path: Path,
) -> None:
    row = {
        "model": "resnet50",
        "backend": "hailo8_to_trt",
        "execution_mode": "native_split",
        "status": "ok",
        "ok": True,
        "case": "b001",
        "setup_id": "orin-hailo8",
        "comparison_backend": "tensorrt",
        "runtime_precision_identity": "hailo_hef_sha256:abc",
        "fps_median": 100.0,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "quality_central_evidence_verified": True,
        "task_quality_pass": True,
        "quality_accuracy_gate_pass": True,
        "quality_gate_status": "inconclusive",
    }
    _write_json(
        tmp_path / "reports" / "native_producer_combined_summary.json",
        {"rows": [row]},
    )

    observation = collect_native_performance_matrix(tmp_path)["observations"][0]

    assert observation["ranking_eligible"] is False
    assert (
        "quality_gate_status_not_admissible:inconclusive"
        in observation["ranking_exclusion_reasons"]
    )


def test_evalrun_feeds_native_observations_into_ranking_before_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, profile, _policy = _ranking_contract()
    profile["name"] = "native-ranking-integration"
    profile["ranking_validation"] = {
        "enabled": True,
        "methods": ["cut_bytes_only"],
        "k_values": [1],
        "elite_q_values": [1],
        "primary_k": 1,
        "minimum_candidates_for_correlation": 2,
        "require_complete_candidate_universe": True,
        "require_frozen_predictions": True,
    }
    (tmp_path / "profile.yaml").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    native_rows = [
        _native_ranking_row("b001", 10.0),
        _native_ranking_row("b002", 20.0),
    ]
    # Simulate a future/legacy normalized Native projection with the same
    # stable identity but without the newer endpoint/host-tail stratum fields.
    already_normalized_native_rows = [
        {
            key: row[key]
            for key in (
                "model_id",
                "evaluation_role",
                "case_id",
                "variant",
                "direction",
                "runner_regime",
                "setup_id",
                "runtime_precision_identity",
                "comparison_backend",
                "native_measured_cycle_ms",
                "ranking_eligible",
                "runtime_executable",
            )
        }
        for row in native_rows
    ]
    captured: dict = {}

    monkeypatch.setattr(
        scientific_reporting,
        "_load_evalrun_rows",
        lambda *_args, **_kwargs: (
            already_normalized_native_rows,
            predictions,
            [],
        ),
    )
    monkeypatch.setattr(
        scientific_reporting,
        "collect_native_performance_matrix",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "observation_count": 2,
            "expected_row_count": 2,
            "observations": native_rows,
        },
    )
    monkeypatch.setattr(
        scientific_reporting,
        "compute_cross_runner_report",
        lambda *_args, **_kwargs: {
            "status": "unavailable",
            "pairs": [],
            "groups": [],
            "macro": {},
        },
    )
    monkeypatch.setattr(
        scientific_reporting, "collect_native_energy", lambda *_args: []
    )
    monkeypatch.setattr(
        scientific_reporting, "scientific_energy_rows", lambda *_args: []
    )
    monkeypatch.setattr(
        scientific_reporting,
        "_write_reports",
        lambda _root, payload, **_kwargs: captured.update(
            {"payload": payload}
        )
        or {},
    )
    monkeypatch.setattr(
        scientific_reporting,
        "_augment_v60z_quality_evidence",
        lambda *_args, **_kwargs: {},
    )
    from onnx_splitpoint_tool.workflow import run_discovery

    monkeypatch.setattr(
        run_discovery,
        "build_measurement_set_contract",
        lambda *_args, **_kwargs: {},
    )

    result = scientific_reporting.build_scientific_reports(
        tmp_path, cleanup_legacy=False
    )

    ranking_rows = captured["payload"]["ranking_method_comparison"]
    assert result["ranking_method_comparison_rows"] == 1
    assert len(ranking_rows) == 1
    assert ranking_rows[0]["runner_regime"] == "native_fifo"
    assert ranking_rows[0]["valid_candidate_count"] == 2
    assert ranking_rows[0]["comparison_output_endpoint_id"] == (
        "classification_logits"
    )
    assert ranking_rows[0]["status"] == "development_evidence_only"


def _snapshot_profile() -> dict:
    return {
        "name": "audit-selection",
        "model_suite": {"primary": []},
        "run_profiles": [],
        "selection_policy": {
            "selection_strategy": "score_independent_audit",
            "score_independent_audit_enabled": True,
            "audit_candidate_universe": "deterministic_audit",
            "audit_size": 20,
            "minimum_valid_audit_candidates": 10,
            "audit_seed": 20260710,
        },
        "native_producers": {"enabled": False, "energy": {"enabled": False}},
        "energy": {"requested_native_energy": False},
    }


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("score_independent_audit_enabled", False),
        ("audit_candidate_universe", "all_feasible"),
        ("audit_size", 21),
        ("minimum_valid_audit_candidates", 11),
        ("audit_seed", 17),
    ],
)
def test_audit_selection_fields_are_fingerprinted_and_drift_checked(
    field: str,
    changed: object,
) -> None:
    source = _snapshot_profile()
    assert profile_selection_view(source)["selection_policy"][field] == (
        source["selection_policy"][field]
    )
    baseline = build_profile_start_snapshot(
        profile_request="audit-selection",
        source_profile=source,
        resolved_profile=source,
        profile_id="audit-selection",
        profile_path="profile.yaml",
        profile_source="test",
    )
    changed_profile = copy.deepcopy(source)
    changed_profile["selection_policy"][field] = changed
    changed_snapshot = build_profile_start_snapshot(
        profile_request="audit-selection",
        source_profile=changed_profile,
        resolved_profile=changed_profile,
        profile_id="audit-selection",
        profile_path="profile.yaml",
        profile_source="test",
    )
    assert changed_snapshot["selection_fingerprint"] != baseline[
        "selection_fingerprint"
    ]

    with pytest.raises(StartSnapshotConsistencyError, match="selection_policy"):
        build_profile_start_snapshot(
            profile_request="audit-selection",
            source_profile=source,
            resolved_profile=changed_profile,
            profile_id="audit-selection",
            profile_path="profile.yaml",
            profile_source="test",
        )
