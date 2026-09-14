from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.endpoint_lifecycle import (
    build_evalrun_endpoint_lifecycle,
    build_endpoint_lifecycle_ledger,
    classify_runtime_failure_measurements,
    compare_historical_invariants,
    historical_runtime_case_failures,
)


def _endpoint(case_id: str) -> dict[str, object]:
    return {
        "model_id": "resnet50",
        "case_id": case_id,
        "setup_id": "orin_nx_hailo8_01",
        "backend": "hailo8",
        "direction": "part1_hailo_part2_ort",
        "precision": "fp16_int8",
    }


def test_lifecycle_ledger_preserves_measured_failed_and_open_endpoints() -> None:
    planned = [_endpoint("b010"), _endpoint("b020"), _endpoint("b030")]
    result = {
        **_endpoint("b010"),
        "runtime_executable": True,
        "measurement_ok": True,
        "latency_ms": 3.25,
    }
    rejected = {
        **_endpoint("b020"),
        "generation_status": "rejected_by_benchmark_generator",
        "reject_reason": "unsupported_op: NonMaxSuppression",
    }

    payload = build_endpoint_lifecycle_ledger(
        planned,
        materialization_evidence=[result, rejected],
        measurement_evidence=[result],
    )

    by_case = {row["case_id"]: row for row in payload["rows"]}
    assert by_case["b010"]["lifecycle_status"] == "measured"
    assert by_case["b010"]["terminal_reason"] == "measurement_available"
    assert by_case["b010"]["materialized"] is True
    assert by_case["b010"]["measured"] is True
    assert by_case["b010"]["terminal"] is True
    assert by_case["b010"]["terminal_failure"] is False

    assert by_case["b020"]["materialized"] is False
    assert by_case["b020"]["measured"] is False
    assert by_case["b020"]["terminal"] is True
    assert by_case["b020"]["terminal_failure"] is True
    assert by_case["b020"]["lifecycle_status"] == "terminal_without_measurement"
    assert by_case["b020"]["terminal_reason"] == "unsupported_op: NonMaxSuppression"

    assert by_case["b030"]["materialized"] is False
    assert by_case["b030"]["measured"] is False
    assert by_case["b030"]["terminal"] is False
    assert by_case["b030"]["lifecycle_status"] == "planned_not_materialized"
    assert by_case["b030"]["terminal_reason"] == "not_terminal_missing_materialization_evidence"
    assert payload["summary"] == {
        "planned": 3,
        "materialized": 1,
        "measurement_payload_present": 1,
        "payload_missing": 2,
        "measured": 1,
        "valid_measurement": 1,
        "valid_measurement_count": 1,
        "invalid_measurement_terminal": 0,
        "invalid_measurement_count": 0,
        "missing_measurement_payload_count": 2,
        "without_valid_measurement_count": 2,
        "nonmeasured": 2,
        "terminal": 2,
        "nonterminal": 1,
    }


def test_missing_or_flag_only_result_never_becomes_a_measurement() -> None:
    plan = _endpoint("b040")
    flag_only = {**plan, "measured": True, "measurement_ok": True}
    payload = build_endpoint_lifecycle_ledger(
        [plan], measurement_evidence=[flag_only]
    )
    row = payload["rows"][0]
    assert row["measured"] is False
    assert row["terminal"] is False
    assert row["valid_measurement_evidence_count"] == 0


def test_terminal_flag_without_concrete_reason_remains_fail_closed() -> None:
    plan = _endpoint("b050")
    payload = build_endpoint_lifecycle_ledger(
        [plan], terminal_evidence=[{**plan, "terminal": True}]
    )
    row = payload["rows"][0]
    assert row["terminal"] is False
    assert row["lifecycle_status"] == "planned_not_materialized"
    assert row["terminal_reason"] == "not_terminal_missing_materialization_evidence"

    generic_failure = build_endpoint_lifecycle_ledger(
        [plan], terminal_evidence=[{**plan, "status": "failed"}]
    )["rows"][0]
    assert generic_failure["terminal"] is False


def test_specific_terminal_status_is_a_concrete_failure_class() -> None:
    plan = _endpoint("b051")
    row = build_endpoint_lifecycle_ledger(
        [plan], terminal_evidence=[{**plan, "status": "compile_failed"}]
    )["rows"][0]
    assert row["terminal"] is True
    assert row["terminal_reason"] == "status:compile_failed"


def test_numeric_payload_with_terminal_failure_is_visible_but_invalid() -> None:
    plan = _endpoint("b052")
    payload_row = {
        **plan,
        "latency_ms": 4.0,
        "runtime_executable": False,
        "measurement_valid": False,
        "terminal": True,
        "terminal_failure": True,
        "terminal_reason": "case_runner_nonzero_rc:-11",
        "terminal_reason_scope": "case_runtime_process",
        "runner_returncode": -11,
    }
    row = build_endpoint_lifecycle_ledger(
        [plan],
        measurement_evidence=[payload_row],
        terminal_evidence=[payload_row],
    )["rows"][0]
    assert row["measurement_payload_present"] is True
    assert row["measured"] is True
    assert row["valid_measurement"] is False
    assert row["technical_measurement_valid"] is False
    assert row["terminal_failure"] is True
    assert row["lifecycle_status"] == "measured_with_terminal_failure"
    assert row["terminal_reason"] == "case_runner_nonzero_rc:-11"


def test_bound_remote_stdout_reclassifies_only_exact_active_case(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    results = run / "models" / "yolov7_paper" / "benchmark_results"
    results.mkdir(parents=True)
    stdout = results / "remote_benchmark_stdout_orin_nx_hailo8_01.txt"
    stdout.write_text(
        "\n".join([
            "free text rc=-11 must not match",
            "[hailo8_to_trt] [1/2] Running b013 (stage1=hailo8)",
            "[warn] case failed: b999 (rc=-11)",
            "[warn] case failed: b013 (rc=-11)",
            "[hailo8] [2/2] Running b013",
            "[warn] case failed: b013 rc=-11",
        ]) + "\n",
        encoding="utf-8",
    )
    status = results / "remote_benchmark_status_orin_nx_hailo8_01.json"
    status.write_text(json.dumps({
        "model_id": "yolov7_paper",
        "hardware_target_id": "orin_nx_hailo8_01",
        "stdout_path": str(stdout.relative_to(run)),
    }), encoding="utf-8")

    failures, diagnostics = historical_runtime_case_failures(run)
    assert [row["endpoint_id"] for row in failures] == [
        "generic:yolov7_paper:hailo8_to_trt:b013"
    ]
    assert failures[0]["runner_returncode"] == -11
    assert any(
        row["reason"] == "remote_case_failure_without_matching_active_case"
        for row in diagnostics
    )
    rows = [
        {
            "model_id": "yolov7_paper",
            "case_id": "b013",
            "run_id": "hailo8_to_trt",
            "runtime_executable": True,
            "latency_ms": 2.0,
            "performance_claim_exclusion_reasons": "[]",
        },
        {
            "model_id": "yolov7_paper",
            "case_id": "full",
            "run_id": "hailo8",
            "runtime_executable": True,
            "latency_ms": 1.0,
        },
    ]
    classified, matched, _diagnostics = (
        classify_runtime_failure_measurements(run, rows)
    )
    assert len(matched) == 1
    assert classified[0]["row_status"] == "terminal_process_failure"
    assert classified[0]["measurement_valid"] is False
    assert classified[0]["ranking_eligible"] is False
    assert classified[0]["performance_claim_exclusion_reasons"] == [
        "terminal_process_failure"
    ]
    assert classified[1].get("terminal_failure") is not True


def test_unplanned_evidence_is_diagnostic_not_a_ledger_row() -> None:
    plan = _endpoint("b060")
    extra = {**_endpoint("b999"), "latency_ms": 1.0}
    payload = build_endpoint_lifecycle_ledger(
        [plan], measurement_evidence=[extra]
    )
    assert len(payload["rows"]) == 1
    assert payload["diagnostics"]["unmatched_evidence_count"] == 1


def test_duplicate_planned_endpoint_is_rejected() -> None:
    plan = _endpoint("b070")
    with pytest.raises(ValueError, match="Duplicate planned endpoint identity"):
        build_endpoint_lifecycle_ledger([plan, dict(plan)])


def test_v27549_historical_ranking_audit_invariants() -> None:
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "v276"
        / "v27549_ranking_audit_invariants.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = {
        "planned": 264,
        "measurement_payload_present": 208,
        "payload_missing": 56,
        "valid_measurement": 206,
        "invalid_measurement_count": 2,
        "audit_coverage": {
            "resnet50": {"observed": 19, "planned": 20},
            "yolo26s": {"observed": 8, "planned": 20},
            "yolov7_paper": {"observed": 19, "planned": 20},
        },
    }
    comparison = compare_historical_invariants(fixture["observed"], expected)
    assert comparison == {"ok": True, "mismatches": []}
    # The fixture is a report-level factual invariant, not fake endpoint data.
    assert "rows" not in fixture


def test_evalrun_adapter_builds_explicit_generic_lifecycle_read_only(
    tmp_path: Path,
) -> None:
    run = tmp_path / "evalrun"
    (run / "models" / "resnet50" / "analysis").mkdir(parents=True)
    (run / "models" / "resnet50" / "benchmark_set").mkdir(parents=True)
    effective_plan = {
        "models": ["resnet50"],
        "logical_run_profiles": [
            "ort_tensorrt",
            "hailo8",
            "hailo8_to_tensorrt",
            "unknown_experimental_recipe",
        ],
        "effective_generic_run_ids": [
            "ort_tensorrt",
            "hailo8",
            "hailo8_to_tensorrt",
            "unknown_experimental_recipe",
        ],
        # 2 candidate reference rows + reference Full + two Hailo split rows
        # + Hailo Full = six explicitly planned endpoints.
        "expected_generic_result_rows_total": 6,
    }
    candidate_plan = {
        "model_id": "resnet50",
        "selected_candidates": [
            {"case_id": "b010", "boundary": 10},
            {"case_id": "b020", "boundary": 20},
        ],
    }
    generation_decisions = {
        "model_id": "resnet50",
        "accepted_cases": [{"case_id": "b010", "boundary": 10}],
        "rejected_cases": [{
            "case_id": "b020",
            "boundary": 20,
            "status": "rejected",
            "reason": "unsupported_op: GatherElements",
        }],
    }
    (run / "effective_execution_plan.json").write_text(
        json.dumps(effective_plan), encoding="utf-8"
    )
    (run / "models" / "resnet50" / "analysis" / "final_candidate_plan.json").write_text(
        json.dumps(candidate_plan), encoding="utf-8"
    )
    (run / "models" / "resnet50" / "benchmark_set" / "generation_decisions.json").write_text(
        json.dumps(generation_decisions), encoding="utf-8"
    )

    before = {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*") if path.is_file()
    }
    normalized_rows = [
        {
            "model_id": "resnet50",
            "case_id": "full",
            "run_id": "ort_tensorrt",
            "runtime_executable": True,
            "latency_ms": 1.0,
        },
        {
            "model_id": "resnet50",
            "case_id": "b010",
            "run_id": "ort_tensorrt",
            "runtime_executable": True,
            "latency_ms": 2.0,
        },
        {
            "model_id": "resnet50",
            "boundary": 10,
            # Historical result alias must map to the explicit plan recipe.
            "run_id": "hailo8_to_trt",
            "runtime_executable": True,
            "throughput_fps": 100.0,
        },
    ]
    payload = build_evalrun_endpoint_lifecycle(run, normalized_rows)
    after = {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*") if path.is_file()
    }

    assert after == before
    assert payload["adapter"]["read_only"] is True
    assert payload["adapter"]["mapped_run_profiles"] == [
        "ort_tensorrt", "hailo8", "hailo8_to_trt"
    ]
    assert payload["adapter"]["unknown_run_profiles"] == [
        "unknown_experimental_recipe"
    ]
    assert payload["adapter"]["expected_count_matches"] is True
    assert payload["adapter"]["status"] == "partial"
    assert payload["adapter"]["evidence_completeness_status"] == "partial"
    assert "adapter_partial" in payload["adapter"][
        "evidence_completeness_reasons"
    ]
    assert "planned_endpoints_nonterminal" in payload["adapter"][
        "evidence_completeness_reasons"
    ]
    assert payload["adapter"]["measurement_completeness_status"] == (
        "incomplete_with_open_endpoints"
    )
    assert {
        key: payload["summary"][key]
        for key in (
            "planned", "materialized", "measurement_payload_present",
            "measured", "valid_measurement", "nonmeasured", "terminal",
            "nonterminal",
        )
    } == {
        "planned": 6,
        "materialized": 3,
        "measurement_payload_present": 3,
        "measured": 3,
        "valid_measurement": 3,
        "nonmeasured": 3,
        "terminal": 5,
        "nonterminal": 1,
    }
    by_id = {row["endpoint_id"]: row for row in payload["rows"]}
    assert by_id["generic:resnet50:ort_tensorrt:b020"]["terminal_reason"] == (
        "legacy_case_rejected:unsupported_op: GatherElements"
    )
    assert by_id["generic:resnet50:ort_tensorrt:b020"][
        "terminal_reason_scope"
    ] == "case_generation"
    assert by_id["generic:resnet50:hailo8_to_trt:b020"]["terminal"] is True
    assert by_id["generic:resnet50:hailo8:full"]["lifecycle_status"] == (
        "planned_not_materialized"
    )
    assert by_id["generic:resnet50:hailo8:full"]["terminal"] is False


def test_evalrun_adapter_scopes_backend_terminal_state_to_dependent_backend(
    tmp_path: Path,
) -> None:
    run = tmp_path / "evalrun"
    analysis = run / "models" / "resnet50" / "analysis"
    benchmark_set = run / "models" / "resnet50" / "benchmark_set"
    analysis.mkdir(parents=True)
    benchmark_set.mkdir(parents=True)
    (run / "effective_execution_plan.json").write_text(json.dumps({
        "models": ["resnet50"],
        "effective_generic_run_ids": ["ort_tensorrt", "hailo8_to_trt"],
        "expected_generic_result_rows_total": 3,
    }), encoding="utf-8")
    (analysis / "final_candidate_plan.json").write_text(json.dumps({
        "selected_candidates": [{"case_id": "b010"}],
        "audit_candidates": [{"case_id": "b010"}],
    }), encoding="utf-8")
    (benchmark_set / "generation_decisions.json").write_text(json.dumps({
        "accepted_cases": [{
            "case_id": "b010",
            "hailo_backend_terminal_states": [{
                "hw_arch": "hailo8",
                "variant": "part1",
                "status": "terminal_failed",
                "terminal": True,
                "available": False,
                "reason": "unsupported_op: GatherElements",
            }],
        }],
        "rejected_cases": [],
    }), encoding="utf-8")

    payload = build_evalrun_endpoint_lifecycle(run, [])
    by_id = {row["endpoint_id"]: row for row in payload["rows"]}
    hailo = by_id["generic:resnet50:hailo8_to_trt:b010"]
    ort = by_id["generic:resnet50:ort_tensorrt:b010"]
    assert hailo["terminal"] is True
    assert hailo["materialized"] is False
    assert hailo["terminal_reason"] == (
        "backend_terminal_state:unsupported_op: GatherElements"
    )
    assert hailo["terminal_reason_scope"] == "backend_artifact"
    assert hailo["causal_backend"] == "hailo8"
    assert ort["terminal"] is False
    assert ort["materialized"] is True
    assert ort["lifecycle_status"] == "materialized_not_terminal"
