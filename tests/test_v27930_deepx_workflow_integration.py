"""Normal Native Full driver, preflight, merge, aggregation and report chain."""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path
import pytest
from scripts import native_full_baseline_eval_runner as runner
from scripts import native_producer_final_report as report
from scripts import native_producer_energy_plan as energy
from onnx_splitpoint_tool.workflow.runner import _native_concise_summary_v60w
from onnx_splitpoint_tool.workflow.status_reporting import energy_axis_description
from test_v27930_native_full_semantic_merge import prepare_runner_case, measured_and_semantic
from test_v27930_deepx_semantic_pre_nms import write_json
ROOT = Path(__file__).resolve().parents[1]

def run_driver(case, monkeypatch):
    monkeypatch.setattr(runner, "_select_engine_python", lambda *_: (sys.executable, {}))
    monkeypatch.setattr(runner, "_site_packages_for_python", lambda *_: [])
    monkeypatch.setattr(runner.sys, "argv", ["native_full_baseline_eval_runner.py", "--root", str(case.root.parent.parent),
        "--models", case.model, "--backends", "deepx", "--setup-id", case.setup_id, "--comparison-backend", "deepx",
        "--comparison-precision", "uint8_cast_fp16", "--repetitions", "3", "--frames", "3", "--warmup", "0",
        "--dump-outputs", "--image-map", json.dumps(case.ns.image_map_data)])
    code = runner.main()
    path = case.root.parent.parent / "analysis_tables/native_full_baseline_eval.json"
    return code, json.loads(path.read_text())["rows"][0]

@pytest.mark.parametrize("mutation", ["missing_contract", "missing_result", "false", "wrong_stage", "bad_signature"])
def test_f01_missing_semantic_evidence_stops_before_any_performance(tmp_path, monkeypatch, mutation):
    case = prepare_runner_case(tmp_path, monkeypatch)
    def mutate(payload):
        field = {"missing_contract": "frozen_host_postprocess_contract", "missing_result": "frozen_host_postprocess_result", "false": "host_postprocess_frozen", "wrong_stage": "stage", "bad_signature": "tensor_signature"}[mutation]
        payload[field] = False if mutation == "false" else "raw_head" if mutation == "wrong_stage" else {}
    case.semantic_mutation = mutate
    code, row = run_driver(case, monkeypatch)
    assert code != 0
    assert row["status"] == "blocked_before_repetitions", row
    assert row["preparation_count_attempted"] == 1
    assert row["repetition_count_requested"] == 3
    assert row["repetition_count_attempted"] == row["repetition_count_valid"] == 0
    assert row["repetition_records"] == [] and row["runtime_success"] is False
    assert row["fps_makespan"] is None and "returncode" not in row
    assert "semantic_frozen_postprocess" in row["failure_reason"] or "semantic_endpoint_stage_mismatch" in row["failure_reason"]
    assert case.processes == ["semantic"]
    projected = report._rows_from_native_full(case.root.parent.parent)
    assert projected[0]["repetition_count_attempted"] == 0
    assert report._aggregate_repetitions(projected)[0]["repetition_count_attempted"] == 0


def test_f02_entire_normal_driver_three_simulated_instances_final_report(tmp_path, monkeypatch):
    case = prepare_runner_case(tmp_path, monkeypatch)
    code, row = run_driver(case, monkeypatch)
    assert code == 0 and row["ok"] is True, row
    assert row["repetition_count_valid"] == 3
    assert row["repetition_independence_verified"] is True
    assert len(set(row["repetition_runtime_instance_ids"])) == 3
    assert all(r["runtime_success"] is True for r in row["repetition_records"])
    assert case.processes == ["semantic", "performance", "performance", "performance"]
    projected = report._rows_from_native_full(case.root.parent.parent)
    aggregate = report._aggregate_repetitions(projected)
    assert len(aggregate) == 1 and aggregate[0]["ok"] is True, aggregate
    assert aggregate[0]["postprocess_completed_frames"] == 3
    assert aggregate[0]["stage"] == "decoded_pre_nms"
    assert aggregate[0]["completed_task_endpoint_attestation"]["completed_frames"] == 3
    assert energy._runtime_successful_for_energy(aggregate[0]) is True


def test_f03_recorded_failure_survives_real_final_report_and_concise(tmp_path):
    archived = json.loads((ROOT / "tests/fixtures/v27930/merge_v27929_native_full_failure.json").read_text())["row"]
    before = copy.deepcopy(archived)
    write_json(tmp_path / "analysis_tables/native_full_baseline_eval.json", {"rows": [archived]})
    rows = report._aggregate_repetitions(report._rows_from_native_full(tmp_path))
    row = rows[0]
    assert row["ok"] is False and row["runtime_success"] is True
    assert row["primary_repetition_failure_reason"] == "semantic_performance_frozen_postprocess_contract_mismatch"
    assert row["repetition_count_valid"] == 0
    write_json(tmp_path / "reports/native_producer_combined_summary.json", {"rows": rows})
    _, concise = _native_concise_summary_v60w(tmp_path / "reports")
    assert concise[0]["primary_failure_reason"] == row["primary_repetition_failure_reason"]
    assert concise[0]["primary_failure_source"] == "primary_repetition_failure"
    assert concise[0]["runtime_success"] is True
    assert concise[0]["runtime_repetition_count_successful"] == 3
    assert concise[0]["repetition_count_valid"] == 0
    assert concise[0]["aggregate_failure_reason"].startswith("native_repetition_")
    assert archived == before
    assert energy._verify_full_command_contract(row.get("full_command_contract"), expected_identity={})[0] is None

@pytest.mark.parametrize("case", ["current_repetition", "current_semantic", "wrong_setup", "original_only"])
def test_f03_error_priority_never_borrows_other_setup(tmp_path, case):
    row = {"model": "yolo11l", "backend": "native_full_deepx", "case": "full", "setup_id": "current", "comparison_backend": "deepx",
        "ok": False, "failure_reason": "native_repetition_set_incomplete", "status": "partial_repetitions"}
    write_json(tmp_path / "models/yolo11l/benchmark_results/benchmark_results_deepx_m1_full_auto.json", [{
        "model_id": "yolo11l", "backend": "deepx_m1", "variant": "full", "setup_id": "old", "runtime_ok": False,
        "deepx_prepared_feed_benchmark": {"error": "wrong_setup_old_error"}}])
    if case == "current_repetition": row.update(primary_repetition_failure_reason="current_repeat", semantic_dump_failure_reason="current_semantic", original_full_error="old_generic")
    elif case == "current_semantic": row.update(semantic_dump_failure_reason="current_semantic", original_full_error="old_generic")
    elif case == "original_only": row.update(original_full_error="old_generic")
    write_json(tmp_path / "reports/native_producer_summary.json", {"rows": [row]})
    _, concise = _native_concise_summary_v60w(tmp_path / "reports")
    expected = {"current_repetition": "current_repeat", "current_semantic": "current_semantic", "wrong_setup": "native_repetition_set_incomplete", "original_only": "old_generic"}[case]
    assert concise[0]["failure_reason"] == expected


def test_f04_energy_runtime_admission_is_not_runtime_success_alone(tmp_path, monkeypatch):
    case = prepare_runner_case(tmp_path, monkeypatch)
    code, row = run_driver(case, monkeypatch)
    assert code == 0
    assert energy._runtime_successful_for_energy(row) is True
    verified, status = energy._verify_full_command_contract(row.get("full_command_contract"), expected_identity={"model": case.model, "remote_root": str(case.root.parent.parent), "remote_tool_dir": str(ROOT)})
    assert verified is not None, (status, row.get("full_command_contract"))
    bad_case = prepare_runner_case(tmp_path / "bad", monkeypatch)
    bad_case.semantic_mutation = lambda p: p.update(frozen_host_postprocess_contract={})
    bad_code, failed = run_driver(bad_case, monkeypatch)
    assert bad_code != 0
    assert energy._runtime_successful_for_energy(failed) is False
    assert energy._verify_full_command_contract(failed.get("full_command_contract"), expected_identity={})[0] is None
    for fixture_case, performance_row, expected in ((case, row, 1), (bad_case, failed, 0)):
        summary = fixture_case.root.parent.parent / "native_producer_summary.json"
        write_json(summary, {"rows": [performance_row]})
        output = fixture_case.root.parent.parent / "energy_plan"
        monkeypatch.setattr(energy.sys, "argv", ["native_producer_energy_plan.py", "--summary", str(summary),
            "--out-dir", str(output), "--remote-root", str(fixture_case.root.parent.parent),
            "--remote-tool-dir", str(ROOT), "--deepx-ssh", "synthetic-offline-target",
            "--duration-s", "1", "--allow-unpaired", "--measure-all-runtime-successful"])
        assert energy.main() == 0
        planned = json.loads((output / "native_producer_energy_plan.json").read_text())
        assert len(planned["rows"]) == expected, planned.get("excluded_rows")
    text = energy_axis_description({"schema": "onnx-splitpoint/native-evidence-status", "schema_version": 5,
        "energy": {"requested": True, "status": "complete", "plan_included_count": 2, "plan_excluded_count": 1,
                   "matrix_expected_count": 3, "measurement_success_count": 2}})
    assert "plan 2/2" in text and "matrix 2/3" in text and "incomplete" in text
