"""Regression of the D-run's contradictory summary and hidden Full error."""
import json
from pathlib import Path
import sys
import types

import pytest

from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, _native_concise_summary_v60w
from onnx_splitpoint_tool.workflow.status_reporting import (
    blocking_reasons_for_display, deepx_full_primary_failure, energy_axis_description,
)
from onnx_splitpoint_tool.workflow.validation_binding import _decide_validation_status


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runner(tmp_path, *, strict=False):
    runner = EvaluationWorkflowRunner(WorkflowOptions(
        profile="unused", out=str(tmp_path), validation_mode="strict" if strict else "summary_only",
    ))
    runner.run_dir = tmp_path
    runner.profile_payload = {}
    runner.stage_results = [
        {"model_id": "yolo11l", "stage": "run_benchmarks", "status": "partial", "notes": ["Full runtime failed"]},
        {"model_id": "yolo11l", "stage": "validate_outputs", "status": "warn", "notes": ["Invalid measured rows"]},
    ]
    _write(tmp_path / "models/yolo11l/benchmark_results/normalized_results.json", {"result_count": 3})
    return runner


def test_summary_only_validation_warning_does_not_duplicate_runtime_blocker(tmp_path):
    runner = _runner(tmp_path)
    status, result = runner._derive_final_status()
    assert status == "partial"
    assert [r["stage"] for r in result["blocking_reasons"]] == ["run_benchmarks"]
    assert [r["stage"] for r in result["non_blocking_reasons"]] == ["validate_outputs"]


@pytest.mark.parametrize("field,value", [("status", "failed"), ("status", "partial"), ("error_detail", "Worker crashed"), ("error_class", "RuntimeError")])
def test_validation_runtime_errors_are_still_blocking(tmp_path, field, value):
    runner = _runner(tmp_path)
    runner.stage_results[1][field] = value
    _, result = runner._derive_final_status()
    assert any(r["stage"] == "validate_outputs" for r in result["blocking_reasons"])


def test_strict_validation_warning_remains_blocking(tmp_path):
    runner = _runner(tmp_path, strict=True)
    _, result = runner._derive_final_status()
    assert any(r["stage"] == "validate_outputs" for r in result["blocking_reasons"])


def test_empty_blocking_list_does_not_promote_legacy_warning_alias():
    warning = {"blocking": False, "reason": "Quality inconclusive"}
    assert blocking_reasons_for_display({"blocking_reasons": [], "partial_reasons": [warning]}) == []
    assert blocking_reasons_for_display({"partial_reasons": [{"reason": "old blocker"}]}) == [{"reason": "old blocker"}]


def test_plan_complete_is_not_displayed_as_matrix_complete():
    evidence = {
        "schema": "onnx-splitpoint/native-evidence-status", "schema_version": 5,
        "energy": {"requested": True, "status": "complete", "plan_included_count": 2,
                   "plan_excluded_count": 1, "matrix_expected_count": 3, "measurement_success_count": 2},
    }
    result = energy_axis_description(evidence)
    assert "plan 2/2 successful (complete)" in result
    assert "matrix 2/3 measured (incomplete)" in result
    assert "excluded 1" in result
    assert "unavailable" in energy_axis_description({})
    assert energy_axis_description({"energy": {"requested": False}}) == "not_applicable"


def _full_failure(tmp_path, **overrides):
    record = {"model_id": "yolo11l", "backend": "deepx_m1", "variant": "full", "runtime_ok": False,
              "deepx_prepared_feed_benchmark": {"error": "FrozenPostprocessError: decoded_pre_nms_values_invalid"}}
    record.update(overrides)
    _write(tmp_path / "models/yolo11l/benchmark_results/benchmark_results_deepx_m1_full_auto.json", [record])


def test_native_concise_surfaces_original_error_and_preserves_aggregate(tmp_path):
    _full_failure(tmp_path)
    reports = tmp_path / "reports"
    _write(reports / "native_producer_summary.json", {"rows": [{
        "backend": "native_full_deepx", "model": "yolo11l", "case": "full", "ok": False,
        "status": "partial_repetitions", "failure_reason": "native_full_repetition_set_incomplete",
    }]})
    _, rows = _native_concise_summary_v60w(reports)
    assert rows[0]["runtime_status"] == "partial_repetitions"
    assert rows[0]["failure_reason"] == "FrozenPostprocessError: decoded_pre_nms_values_invalid"
    assert rows[0]["aggregate_failure_reason"] == "native_full_repetition_set_incomplete"
    assert rows[0]["primary_failure_source"].endswith("benchmark_results_deepx_m1_full_auto.json")


def test_preparation_failure_keeps_zero_repetitions_and_direct_primary_error(tmp_path):
    reports = tmp_path / "reports"
    _write(reports / "native_producer_summary.json", {"rows": [{
        "backend": "native_full_deepx", "model": "yolo11l", "case": "full", "ok": False,
        "status": "blocked_before_repetitions", "failure_reason": "deepx_shared_prepared_input_manifest_missing",
        "original_full_error": "FrozenPostprocessError: decoded_pre_nms_values_invalid",
        "original_full_failure_context_file": "results/deepx_m1_full/original_full_failure.json",
        "preparation_count_attempted": 1, "repetition_count_requested": 3,
        "repetition_count_attempted": 0, "repetition_count_valid": 0,
    }]})
    _, rows = _native_concise_summary_v60w(reports)
    assert rows[0]["runtime_status"] == "blocked_before_repetitions"
    assert rows[0]["failure_reason"] == "FrozenPostprocessError: decoded_pre_nms_values_invalid"
    assert rows[0]["aggregate_failure_reason"] == "deepx_shared_prepared_input_manifest_missing"
    assert rows[0]["repetition_count_attempted"] == 0
    assert rows[0]["repetition_count_valid"] == 0
    assert rows[0]["repetition_count_requested"] == 3
    assert rows[0]["preparation_count_attempted"] == 1


@pytest.mark.parametrize("overrides", [{"backend": "tensorrt"}, {"variant": "split"}, {"runtime_ok": True}])
def test_primary_error_lookup_does_not_attach_unrelated_or_successful_row(tmp_path, overrides):
    _full_failure(tmp_path, **overrides)
    assert deepx_full_primary_failure(tmp_path, "yolo11l") == {}
    assert deepx_full_primary_failure(tmp_path, "../yolo11l") == {}


def test_task_quality_pass_never_overrides_failed_physical_validation():
    row = {"validation_ok": False, "runtime_ok": True, "task_quality_status": "pass",
           "runtime_contract_decision": "fail", "structural_contract_status": "fail"}
    status, ok, reason = _decide_validation_status(row, "detection", {"mode": "summary_only"})
    assert status == "invalid" and ok is False
    assert "task_quality=pass" in reason
    assert row["runtime_contract_decision"] == "fail"


@pytest.fixture
def fix2_suite(tmp_path, monkeypatch):
    """Load the generated suite; hardware is never needed for matrix writing."""
    template = Path(__file__).resolve().parents[1] / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    module = types.ModuleType("v27932_fix2_status_suite")
    module.__file__ = str(tmp_path / "benchmark_suite.py")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = template.read_text(encoding="utf-8").replace("__BENCH_JSON__", "benchmark_set.json")
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _fix2_recorded_full_row():
    # Minimal, literal projection of the uploaded Smartmirror H32-2 result:
    # deepx_full_workflow_v27932_20260909T084426Z_vfp5w253.zip,
    # results/suite_diagnostics/benchmark_results_deepx_m1_full_auto.json.
    # This is report-replay evidence, not another hardware measurement.
    return {
        "run_id": "deepx_m1_full", "case_id": "full", "primary_variant": "full",
        "runtime_ok": True, "latency_ok": True,
        "variant_status": {"full": "ok", "part1": "missing", "part2": "missing", "composed": "missing"},
        "semantic_validation_passed": False, "final_pass": False, "final_pass_all": False,
        "task_quality_gate_status": "unavailable", "task_quality_gate_decision": "unavailable",
    }


def _fix2_run():
    return {"id": "deepx_m1_full", "type": "deepx", "provider": "deepx_m1",
            "stage1": {"backend": "deepx_m1", "type": "deepx"},
            "stage2": {"backend": "deepx_m1", "type": "deepx"}, "variants": ["full"]}


def _fix2_matrix(tmp_path):
    return json.loads((tmp_path / "benchmark_suite_status_matrix.json").read_text())


@pytest.mark.parametrize("preset", ["auto", "detection", "classification"])
def test_fix2_normal_main_matrix_uses_the_result_it_wrote(fix2_suite, tmp_path, monkeypatch, preset):
    row = _fix2_recorded_full_row()
    run = _fix2_run()
    run["tag"] = "display_label_is_not_the_written_result_id"
    _write(tmp_path / "benchmark_set.json", {"model_id": "yolo11l", "cases": [{"case_id": "b003", "folder": "b003"}]})
    _write(tmp_path / "benchmark_plan.json", {"runs": [run]})
    monkeypatch.setattr(fix2_suite, "_deepx_verified_suite_run_identity", lambda *_: {})
    monkeypatch.setattr(fix2_suite, "_run_deepx_full_run", lambda *_, **__: [dict(row)])
    monkeypatch.setattr(fix2_suite, "_write_v60_scientific_report", lambda *_, **__: None)
    monkeypatch.setattr(sys, "argv", ["benchmark_suite.py", "--preset", preset, "--no-plot"])
    assert fix2_suite.main() == 0
    result_path = tmp_path / f"benchmark_results_deepx_m1_full_{preset}.json"
    written = json.loads(result_path.read_text())[0]
    assert written["runtime_ok"] is True
    assert written["final_pass_all"] is False
    assert written["task_quality_gate_status"] == "unavailable"
    matrix = _fix2_matrix(tmp_path)
    assert matrix[0]["full"] == "ok"
    assert matrix[0]["tag"] == f"deepx_m1_full_{preset}"
    assert all(matrix[0][part] == "missing" for part in ("part1", "part2", "composed"))


@pytest.mark.parametrize("has_variant_status", [True, False])
def test_fix2_recorded_success_replays_all_matrix_formats_without_changing_quality(fix2_suite, tmp_path, has_variant_status):
    import csv
    row = _fix2_recorded_full_row()
    if not has_variant_status:
        row.pop("variant_status")
    path = tmp_path / "benchmark_results_deepx_m1_full_auto.json"
    _write(path, [row])
    before = path.read_bytes()
    fix2_suite._write_status_matrix(tmp_path, [_fix2_run()], preset="auto")
    assert path.read_bytes() == before
    assert _fix2_matrix(tmp_path)[0]["full"] == "ok"
    with (tmp_path / "benchmark_suite_status_matrix.csv").open(newline="") as stream:
        assert next(csv.DictReader(stream))["full"] == "ok"
    assert "| ✅ | — | — | — |" in (tmp_path / "benchmark_suite_status_matrix.md").read_text()
    saved = json.loads(path.read_text())[0]
    assert saved["final_pass"] is saved["final_pass_all"] is saved["semantic_validation_passed"] is False
    assert saved["task_quality_gate_decision"] == "unavailable"


def test_fix2_actual_missing_dxnn_failure_is_error_in_current_preset(fix2_suite, tmp_path):
    rows = fix2_suite._run_deepx_full_run(tmp_path, _fix2_run(), types.SimpleNamespace(), expected_endpoint_identity={})
    assert rows[0]["runtime_ok"] is False and rows[0]["error_class"] == "missing_artifact"
    _write(tmp_path / "benchmark_results_deepx_m1_full_auto.json", rows)
    fix2_suite._write_status_matrix(tmp_path, [_fix2_run()], preset="auto")
    assert _fix2_matrix(tmp_path)[0]["full"] == "error"


@pytest.mark.parametrize("stale_tag", [None, "deepx_m1_full_detection", "deepx_m1_full", "another_run_auto"])
def test_fix2_absent_current_result_never_uses_stale_other_preset_or_run(fix2_suite, tmp_path, stale_tag):
    stale = tmp_path / f"benchmark_results_{stale_tag}.json"
    if stale_tag is not None:
        _write(stale, [_fix2_recorded_full_row()])
    fix2_suite._write_status_matrix(tmp_path, [_fix2_run()], preset="auto")
    assert _fix2_matrix(tmp_path)[0]["full"] == "missing"
    if stale_tag is not None:
        assert json.loads(stale.read_text())[0] == _fix2_recorded_full_row()


def test_fix2_legacy_explicit_tag_still_works_without_current_preset(fix2_suite, tmp_path):
    run = dict(_fix2_run(), tag="legacy_export")
    _write(tmp_path / "benchmark_results_legacy_export.json", [_fix2_recorded_full_row()])
    fix2_suite._write_status_matrix(tmp_path, [run])
    assert _fix2_matrix(tmp_path)[0]["full"] == "ok"
    assert _fix2_matrix(tmp_path)[0]["tag"] == "legacy_export"


@pytest.mark.parametrize("skip_reason", ["unsupported_provider", "case_filter_empty"])
def test_fix2_normal_main_skipped_run_cannot_reuse_same_preset_success(fix2_suite, tmp_path, monkeypatch, skip_reason):
    skipped = {"id": "ort_skipped", "type": "onnxruntime", "provider": "cpu"}
    if skip_reason == "unsupported_provider":
        skipped["provider"] = "unsupported_provider"
    else:
        skipped["case_id"] = "b999"
    _write(tmp_path / "benchmark_set.json", {"model_id": "yolo11l", "cases": [{"case_id": "b003", "folder": "b003"}]})
    _write(tmp_path / "benchmark_plan.json", {"runs": [_fix2_run(), skipped]})
    stale = tmp_path / "benchmark_results_ort_skipped_auto.json"
    _write(stale, [{"run_id": "ort_skipped", "primary_variant": "full", "runtime_ok": True,
                    "variant_status": {"full": "ok"}}])
    before = stale.read_bytes()
    monkeypatch.setattr(fix2_suite, "_deepx_verified_suite_run_identity", lambda *_: {})
    monkeypatch.setattr(fix2_suite, "_run_deepx_full_run", lambda *_, **__: [_fix2_recorded_full_row()])
    monkeypatch.setattr(fix2_suite, "_write_v60_scientific_report", lambda *_, **__: None)
    monkeypatch.setattr(sys, "argv", ["benchmark_suite.py", "--preset", "auto", "--no-plot"])
    assert fix2_suite.main() == 0
    matrix = {row["run"]: row for row in _fix2_matrix(tmp_path)}
    assert matrix["deepx_m1_full"]["full"] == "ok"
    assert matrix["ort_skipped"]["full"] == "missing"
    assert stale.read_bytes() == before
    if skip_reason == "case_filter_empty":
        status = json.loads((tmp_path / "benchmark_suite_status.json").read_text())
        assert any(run.get("reason") == "case_filter_empty" for run in status["failed_runs"])


@pytest.mark.parametrize("current_state", ["missing_tag", "empty_rows", "negative_rows"])
def test_fix2_explicit_current_rows_override_same_preset_stale_file(fix2_suite, tmp_path, current_state):
    stale = tmp_path / "benchmark_results_deepx_m1_full_auto.json"
    _write(stale, [_fix2_recorded_full_row()])
    before = stale.read_bytes()
    rows_by_tag = {}
    expected = "missing"
    if current_state == "empty_rows":
        rows_by_tag["deepx_m1_full_auto"] = []
    elif current_state == "negative_rows":
        rows_by_tag["deepx_m1_full_auto"] = fix2_suite._run_deepx_full_run(
            tmp_path, _fix2_run(), types.SimpleNamespace(), expected_endpoint_identity={})
        expected = "error"
    fix2_suite._write_status_matrix(tmp_path, [_fix2_run()], preset="auto", rows_by_tag=rows_by_tag)
    assert _fix2_matrix(tmp_path)[0]["full"] == expected
    assert stale.read_bytes() == before
