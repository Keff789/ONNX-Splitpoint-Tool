from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.reporting_quality_decomposition import (
    _backend, build_quality_decomposition,
)
from onnx_splitpoint_tool.workflow.endpoint_lifecycle import build_evalrun_endpoint_lifecycle
from onnx_splitpoint_tool.workflow.evidence_state_model import (
    project_evidence_state, summarize_run_scope,
)


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("backend,expected", [
    ("hailo10_to_tensorrt", "hailo10h"),
    ("hailo10_to_trt", "hailo10h"),
    ("hailo10h", "hailo10h"),
    ("hailo8_to_tensorrt", "hailo8"),
    ("deepx_to_tensorrt", "deepx"),
    ("ort_tensorrt", "tensorrt"),
    ("tensorrt_to_tensorrt", "tensorrt"),
])
def test_composed_backend_uses_producer_family(backend, expected):
    assert _backend({"backend": backend}) == expected


def test_quality_export_keeps_metrics_decision_and_input_request():
    rows = [{
        "model_id": "mobilenet_v3_large", "case_id": "b119",
        "backend": "hailo10_to_tensorrt", "variant": "composed",
        "source_setup_id": "orin_nx_hailo10_01", "task": "classification",
        "source_request": "original/task_quality_input.json",
        "source_request_sha256": "a" * 64, "reference_identity": "original-reference",
        "decision": "accuracy_loss", "technical_status": "completed",
        "primary": {"metric": "top1", "candidate": 0.5844, "reference": 0.737,
                    "delta": -0.1526, "ci_low": -0.16, "ci_high": -0.14,
                    "n": 5000, "bootstrap_repetitions": 1000},
    }]
    before = deepcopy(rows)
    exported, _ = build_quality_decomposition(rows)
    control = deepcopy(rows)
    control[0]["backend"] = "hailo10h"
    expected, _ = build_quality_decomposition(control)
    assert exported == expected
    assert exported[0]["backend"] == "hailo10h"
    assert exported[0]["quality_decision"] == "accuracy_loss"
    assert rows == before


@pytest.mark.parametrize("decision", ["reference_close", "accuracy_loss", "not_estimable"])
def test_completed_accuracy_decision_is_preserved_without_new_claim(decision):
    row = {"status": "completed", "variant": "split", "claim_eligible": True,
           "task_quality_gates_by_variant": {"composed": {
               "technical_status": "completed", "decision": decision,
               "source_request_sha256": "a" * 64}}}
    before = deepcopy(row)
    state = project_evidence_state(row, scope={"quality_applicability": "applicable"})
    assert state["quality_completion"] == "completed"
    assert state["quality_decision"] == decision
    assert state["claim_eligible"] is False
    assert row == before


def test_accuracy_label_alone_does_not_prove_completed_statistics():
    state = project_evidence_state({
        "status": "completed", "variant": "full",
        "task_quality_gate": {"decision": "not_estimable", "technical_status": "failed"},
    })
    assert state["quality_completion"] == "missing"
    assert state["claim_eligible"] is False


@pytest.mark.parametrize("decision", ["reference_close", "accuracy_loss", "not_estimable"])
def test_scientific_header_recognizes_completed_accuracy_without_claim_release(decision):
    from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_decision_axes
    axes = _scientific_decision_axes(
        {"rows": []}, technical_execution_status="ok",
        central_quality_reporting={"technical_status": "ok", "quality_decision": decision},
    )
    assert axes["quality_evaluation_status"] == "evaluated"
    assert axes["quality_decision"] == decision
    assert axes["claim_readiness_status"] == "blocked"
    assert axes["scientific_pass"] is False
    assert axes["claim_eligible_row_count"] == 0
    assert axes["claim_readiness_reasons"] == [f"aggregate_quality_{decision}", "no_claim_eligible_rows"]


@pytest.mark.parametrize("technical", ["failed", "partial", "unavailable"])
def test_scientific_header_keeps_incomplete_quality_negative(technical):
    from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_decision_axes
    axes = _scientific_decision_axes(
        {"rows": []}, technical_execution_status="ok",
        central_quality_reporting={"technical_status": technical, "quality_decision": "reference_close"},
    )
    assert axes["quality_evaluation_status"] == technical
    assert axes["claim_readiness_status"] == "blocked"
    assert axes["scientific_pass"] is False


def test_scientific_header_does_not_invent_a_missing_quality_decision():
    from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_decision_axes
    axes = _scientific_decision_axes(
        {"rows": []}, technical_execution_status="ok",
        central_quality_reporting={"technical_status": "ok", "quality_decision": "not_evaluated"},
    )
    assert axes["quality_evaluation_status"] == "not_evaluated"
    assert axes["scientific_pass"] is False


@pytest.fixture
def missing_scope(tmp_path):
    model = tmp_path / "models/yolo26m"
    entry = dict(model_id="yolo26m", case_id="b274", variant="split",
                 run_id="hailo10_to_tensorrt", backend="hailo10_to_tensorrt",
                 expected_setup_id="orin_nx_hailo10_01", requested=True,
                 terminal_outcome_required=True, success_required=False,
                 quality_applicability="applicable", logical_identity_sha256="a" * 64)
    evidence = dict(ok=False, mapping_failed=True, error="Compilation failed: No successful assignments: fixture")
    case = dict(folder="b274", hailo_compile={"hailo10": {"part1": evidence}})
    suite = dict(model_name="yolo26m", cases=[case])
    scope = model / "benchmark_set/required_run_scope.json"
    original = model / "benchmark_set/legacy_suite/benchmark_set.json"
    write(scope, dict(model_id="yolo26m", identities=[entry]))
    write(original, suite)
    return tmp_path, model, entry, scope, suite, original


def test_original_negative_is_terminal_without_fabricating_measurement(missing_scope):
    run, _model, _entry, _scope, _suite, original = missing_scope
    before = original.read_bytes()
    summary = summarize_run_scope(run)
    assert summary["matrix_required"] == 1
    assert summary["matrix_present"] == 0
    assert summary["quality_applicable"] == 0
    assert summary["quality_missing"] == 0
    assert summary["quality_not_applicable"] == 1
    state = summary["states"][0]
    assert state["build_runtime_state"] == "build_failed"
    assert state["raw_representation"] == "missing"
    assert state["claim_eligible"] is False
    assert state["upstream_evidence_path"] == str(original)
    assert original.read_bytes() == before


@pytest.mark.parametrize("mutation", [
    "foreign_model", "foreign_case", "foreign_backend", "foreign_run",
    "not_requested", "success_required", "duplicate_case", "timeout",
    "cancelled", "unknown_failure",
])
def test_unproven_negative_remains_missing(missing_scope, mutation):
    run, _model, entry, scope, suite, original = missing_scope
    evidence = suite["cases"][0]["hailo_compile"]["hailo10"]["part1"]
    if mutation == "foreign_model": suite["model_name"] = "different-model"
    elif mutation == "foreign_case": suite["cases"][0]["folder"] = "b001"
    elif mutation == "foreign_backend": entry["backend"] = "hailo8_to_trt"
    elif mutation == "foreign_run": entry["run_id"] = "hailo8_to_trt"
    elif mutation == "not_requested": entry["requested"] = False
    elif mutation == "success_required": entry["success_required"] = True
    elif mutation == "duplicate_case": suite["cases"].append(deepcopy(suite["cases"][0]))
    elif mutation == "timeout": evidence["timed_out"] = True
    elif mutation == "cancelled": evidence["cancelled"] = True
    elif mutation == "unknown_failure": evidence.update(mapping_failed=False, error="unexplained failure")
    write(scope, dict(model_id="yolo26m", identities=[entry]))
    write(original, suite)
    summary = summarize_run_scope(run)
    assert summary["quality_missing"] == 1
    assert summary["quality_not_applicable"] == 0
    assert summary["states"][0]["claim_eligible"] is False


def test_historical_part1_policy_is_separate_from_compile_failure(missing_scope):
    run, model, _entry, _scope, suite, original = missing_scope
    suite["cases"][0]["hailo_compile"] = {}
    write(original, suite)
    manifest = model / "benchmark_set/legacy_suite/b274/split_manifest.json"
    policy = dict(folder="b274", hw_arch="hailo10", stage="part1", status="rejected",
                  reason="hailo_yolo26_static_part1_guard")
    write(manifest, {"hailo": {"backend_exclusions": [policy]}})
    summary = summarize_run_scope(run)
    assert summary["quality_not_applicable"] == 1
    assert summary["states"][0]["build_runtime_state"] == "policy_excluded"
    assert summary["states"][0]["terminal_evidence"] == policy
    assert summary["states"][0]["upstream_evidence_path"] == str(manifest)
    # A Part2 policy or another case cannot exclude the requested completed task.
    for field, value in [("stage", "part2"), ("folder", "b001"), ("hw_arch", "hailo8")]:
        write(manifest, {"hailo": {"backend_exclusions": [{**policy, field: value}]}})
        assert summarize_run_scope(run)["quality_missing"] == 1


def test_measured_quality_has_priority_over_historical_negative(missing_scope):
    run, model, entry, _scope, _suite, _original = missing_scope
    row = {**entry, "setup_id": entry["expected_setup_id"], "status": "completed",
           "source_request_sha256": "b" * 64,
           "task_quality_gates_by_variant": {"composed": {
               "technical_status": "completed", "decision": "accuracy_loss"}}}
    write(model / "benchmark_results/normalized_results.json", {"results": [row]})
    summary = summarize_run_scope(run)
    assert summary["matrix_present"] == 1
    assert summary["quality_completed"] == 1
    assert summary["quality_not_applicable"] == 0
    assert summary["states"][0]["quality_decision"] == "accuracy_loss"
    assert "terminal_evidence" not in summary["states"][0]


@pytest.mark.parametrize("run_id", ["hailo10_to_trt", "hailo10_to_tensorrt"])
def test_hailo10_original_terminal_state_does_not_leak_to_other_backends(tmp_path, run_id):
    model = tmp_path / "models/yolo26m"
    write(tmp_path / "effective_execution_plan.json", {
        "models": ["yolo26m"],
        "effective_generic_run_ids": ["ort_tensorrt", "hailo8_to_trt", run_id],
        "expected_generic_result_rows_total": 4})
    write(model / "analysis/final_candidate_plan.json", {
        "selected_candidates": [{"case_id": "b274"}],
        "audit_candidates": [{"case_id": "b274"}]})
    write(model / "benchmark_set/generation_decisions.json", {
        "accepted_cases": [{"case_id": "b274", "hailo_backend_terminal_states": [{
            "hw_arch": "hailo10", "variant": "part1", "status": "terminal_failed",
            "terminal": True, "available": False, "reason": "exact original compile failure"}]}],
        "rejected_cases": []})
    ledger = build_evalrun_endpoint_lifecycle(tmp_path, [])
    rows = {row["endpoint_id"].split(":")[2]: row for row in ledger["rows"] if row["case_id"] == "b274"}
    assert rows["hailo10_to_tensorrt"]["lifecycle_status"] == "terminal_without_measurement"
    assert rows["hailo10_to_tensorrt"]["terminal_reason"] == "backend_terminal_state:exact original compile failure"
    assert rows["hailo8_to_trt"]["terminal"] is False
    assert rows["ort_tensorrt"]["terminal"] is False
