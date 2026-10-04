"""Local completion replay; process leaves are controlled, never hardware."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.native_job_identity import (
    native_subset_projection, missing_native_rows, merge_native_result_rows,
    native_job_prerequisite, native_capability_exclusion,
)
from onnx_splitpoint_tool.workflow import runner as workflow
from scripts import native_producer_final_report as reporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def job(case="b001", **updates):
    return dict(backend="deepx_to_trt", backend_key="deepx", model="resnet50", case=case,
                comparison_backend="deepx", precision="float32_layout_fp16", setup_id="test_deepx",
                execution_mode="native_split", **updates)


def test_actual_inputs_terminal_and_original_results_survive_second_resume(tmp_path):
    manifest = tmp_path / "b003/split_manifest.json"
    write(manifest, {"cut_tensors": ["boundary"], "part2_external_inputs": ["boundary", "images"]})
    expected = [job("b001"), job("b002"), job("b003")]
    proof = {"model": "resnet50", "case": "b003", "split_manifest": str(manifest),
             "part2_input_names": ["boundary", "images"], "part2_input_count": 2,
             "native_supported": False, "reason": "part2_input_count_not_one"}
    projection = native_subset_projection(expected, [proof])
    original = [{**job(), "ok": True, "tool_version": "2.91.0", "repetition_records": [{"id": "original"}]}]
    frozen = copy.deepcopy(original)
    pending = missing_native_rows(expected, original, projection)
    assert [row["case"] for row in pending] == ["b002"]
    assert native_job_prerequisite(job("b003"), suite_dir=tmp_path, model_dir=tmp_path,
                                   quality_error="quality_missing")["status"] == "not_supported"
    new = [{**pending[0], "ok": True, "tool_version": "2.91.1"}]
    merged = merge_native_result_rows(original, new, projection)
    assert original == frozen and merged[0] == original[0]
    assert missing_native_rows(expected, merged, projection) == []
    matrix = workflow._native_expected_matrix_status_v60y(expected, merged, [])
    assert matrix["not_supported_expected_row_count"] == 1
    assert matrix["successful_expected_row_count"] == 2
    assert matrix["failed_expected_row_count"] == 0
    assert matrix["technical_execution_complete"] is True
    assert matrix["execution_success_complete"] is False
    assert reporter._evidence_summary(merged, matrix)["matrix_counts_consistent"] is True
    altered = copy.deepcopy(projection[0]); altered["native_capability_exclusion"]["part2_input_names"] = ["fake", "inputs"]
    assert not native_capability_exclusion(altered)
    with pytest.raises(ValueError, match="would_be_replaced"):
        merge_native_result_rows(original, [{**original[0], "tool_version": "2.91.1"}])


def test_normal_native_stage_only_dispatches_missing_leaf_and_reentry_skips_it(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
    options = workflow.WorkflowOptions(profile="", out=str(tmp_path))
    options.resume_missing = True
    runner = workflow.EvaluationWorkflowRunner(options)
    runner.run_id = "controlled_native_completion"
    runner.run_dir = tmp_path / runner.run_id
    runner._supplement_original_root = tmp_path / "original"
    runner._remote_process_registry.configure_journal(
        scope=RemoteProcessLeaseScope(runner.run_id, runner.session_id), journal_dir=tmp_path / "leases")
    runner.manifest = {"models": {"resnet50": {}}}
    runner.profile_payload = {"execution_preset": {"id": "smoke"},
        "run_profiles": [{"id": "deepx_to_trt", "type": "split", "stage1": "deepx", "stage2": "tensorrt"}]}
    cfg = {"enabled": True, "models": ["resnet50"], "backends": ["deepx"], "split_backends": ["deepx"],
        "precision": "float32_layout_fp16", "case_policy": "case_map_only",
        "case_map": {"resnet50": ["b001", "b002", "b003"]},
        "remotes": {"deepx": {"ssh": "nx@controlled-test", "setup_id": "test_deepx"}},
        "copy_benchmarksets": False, "cleanup_remote_native_root": False,
        "validation": {"enabled": False}, "energy": {"enabled": False},
        "full_baselines": {"enabled": True, "backends_by_producer": {"deepx": ["deepx", "tensorrt"]}}}
    suite = runner.run_dir / "models/resnet50/benchmark_set"
    write(suite / "benchmark_set.json", {"task": "classification", "cases": [{"id": f"b{i:03d}"} for i in (1, 2, 3)]})
    for case in ("b001", "b002", "b003"):
        write(suite / case / "split_manifest.json", {"part2_external_inputs": ["boundary", "images"] if case == "b003" else ["boundary"]})
    write(runner.run_dir / "quality_management/central_quality_summary.json", {"results": []})
    previous = [{**job(), "ok": True, "tool_version": "2.91.0", "fps_makespan": 10},
        {**job("full"), "backend": "native_full_deepx", "execution_mode": "native_full_baseline", "ok": True, "tool_version": "2.91.0"}]
    previous_path = runner._supplement_original_root / "reports/native_producer_summary.json"
    write(previous_path, {"rows": previous})
    before = previous_path.read_bytes()
    labels, commands = [], []
    energy_fail = False
    def stream(command, *args, **kwargs):
        label = kwargs.get("label", "")
        labels.append(label); commands.append((label, command))
        if label == "final_report":
            stage = json.loads((runner.run_dir / "reports/native_producer_stage.json").read_text())
            rows = merge_native_result_rows(json.loads(previous_path.read_text())["rows"], [], stage["native_not_supported_rows"])
            write(runner.run_dir / "reports/native_producer_combined_summary.json", {"rows": rows})
        if label == "energy:measure":
            out = Path(command[command.index("--out-dir") + 1])
            if energy_fail:
                write(out / "resume_failure.json", {
                    "status": "resume_preparation_or_cohort_preflight_failed",
                    "started_measurement_count": 0, "measurement_wrapper_started_count": 0,
                    "collector_started_repeat_count": 0, "workload_started_repeat_count": 0,
                    "resume_preparation": {"cohort_preflight": {
                        "preflight_attempted_count": 222, "preflight_passed_count": 142,
                        "preflight_failed_count": 80, "rows": []}}})
                return subprocess.CompletedProcess(command, 2, "preflight aborted", "")
            write(out / "plan/native_producer_energy_plan.json", {"rows": [{"model": "resnet50"}], "preflight_status": "passed"})
            write(out / "native_producer_energy_results.json", {
                "rows": [{"ok": True}], "ok": True, "complete": True, "status": "completed",
                "started_measurement_count": 0, "supplement_reused_row_count": 4})
        return subprocess.CompletedProcess(command, 0, "", "")
    def binding(*args, **kwargs):
        selected = kwargs["selections"][0]
        assert selected["case_id"] == "b002", "reused and unsupported rows must not request prerequisites"
        key = "|".join(selected[k] for k in ("model_id", "case_id", "backend"))
        return {"bindings_by_model_case_backend": {key: {"preselection": selected}}}
    monkeypatch.setattr(workflow, "run_streaming", stream)
    monkeypatch.setattr(workflow, "normalize_hardware_targets", lambda *_: [])
    monkeypatch.setattr(workflow, "benchmark_set_postcondition_v60v", lambda path: {"valid": True, "selected_suite_dir": str(path)})
    monkeypatch.setattr(workflow, "_sync_remote_script_v60i", lambda *a, **k: [])
    monkeypatch.setattr(workflow, "native_remote_package_closure", lambda: [])
    monkeypatch.setattr(runner, "_finish_native_direct_remote_lease", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_native_producer_config", lambda: cfg)
    monkeypatch.setattr("onnx_splitpoint_tool.trt_quality_chain.load_split_binding_set_from_central_quality_summary", binding)
    monkeypatch.setattr("onnx_splitpoint_tool.trt_quality_chain.load_producer_set_from_central_quality_summary",
                        lambda *a, **k: {"producers_by_model": {"resnet50": {"model_id": "resnet50"}}})
    monkeypatch.setattr(workflow, "_merge_row_local_quality_contracts", lambda values, **kwargs: values[0])
    monkeypatch.setattr("onnx_splitpoint_tool.deepx.config.classification_profile_admission", lambda *_: {"allowed": True, "claim_eligible": True})
    runner._stage_run_native_producers()
    stage = json.loads((runner.run_dir / "reports/native_producer_stage.json").read_text())
    assert stage["selective_completion"]["reused_row_count"] == 2
    assert stage["selective_completion"]["not_supported_row_count"] == 1
    assert stage["selective_completion"]["pending_row_count"] == 2
    dispatch = [(label, cmd) for label, cmd in commands if label.startswith("split:")]
    assert len(dispatch) == 1
    assert 'b002' in str(dispatch) and 'b001' not in str(dispatch) and 'b003' not in str(dispatch)
    full = [(label, cmd) for label, cmd in commands if label.startswith("full:")]
    assert [label for label, _ in full] == ["full:deepx:tensorrt"]
    assert "--out-dir" in str(full) and "--models resnet50" in str(full)
    assert previous_path.read_bytes() == before
    assert all(str(runner.run_dir / "native_producers/deepx") not in str(cmd)
               for label, cmd in commands if label.startswith("collect:"))
    # A later attempt sees completed new rows in its preserved prior summary.
    write(previous_path, {"rows": [*previous, {**job("b002"), "ok": True, "tool_version": "2.91.1"},
        {**job("full"), "backend": "native_full_tensorrt", "execution_mode": "native_full_baseline", "ok": True}]})
    labels.clear(); commands.clear()
    runner._stage_run_native_producers()
    assert not any(label.startswith(("split:", "full:")) for label in labels)
    stage = json.loads((runner.run_dir / "reports/native_producer_stage.json").read_text())
    assert stage["selective_completion"]["pending_row_count"] == 0

    # A completed strict Energy cohort is also reusable without a new capture.
    # Exercise the same direct stage, controlling only the subprocess boundary.
    registry = tmp_path / "hardware.json"
    write(registry, {})
    monkeypatch.setattr(workflow, "normalize_hardware_targets", lambda *_: [{
        "id": "test_deepx", "enabled": True, "setup_source": str(registry)}])
    monkeypatch.setattr(runner, "_native_energy_registry_file", lambda: str(registry))
    runner.profile_payload["execution_preset"] = {"id": "standard"}
    cfg["energy"] = {"enabled": True, "mode": "measure", "strict": True}
    write(runner._supplement_original_root / "reports/native_energy_measurements/native_producer_energy_results.json", {"rows": []})
    labels.clear(); commands.clear()
    runner._stage_run_native_producers()
    assert not any(label.startswith(("split:", "full:")) for label in labels)
    assert labels.count("energy:measure") == 1
    energy_cmd = next(cmd for label, cmd in commands if label == "energy:measure")
    assert "--previous-results" in energy_cmd
    stage = json.loads((runner.run_dir / "reports/native_producer_stage.json").read_text())
    assert stage["native_energy"]["strict_requested"] is True
    assert stage["native_energy"]["status"] == "ok"
    assert stage["native_energy"]["strict_failure"] is False
    assert stage["native_energy"]["started_measurement_count"] == 0
    assert stage["native_energy"]["reused_measurement_count"] == 4
    # The next attempt's preflight abort must report its own failure, even
    # while an earlier successful results projection remains at that location.
    energy_fail = True
    historical = Path(stage["native_energy"]["results_json"])
    historical_bytes = historical.read_bytes()
    labels.clear(); commands.clear()
    runner._stage_run_native_producers()
    stage = json.loads((runner.run_dir / "reports/native_producer_stage.json").read_text())
    assert not any(label.startswith(("split:", "full:")) for label in labels)
    assert stage["native_energy"]["strict_failure"] is True
    assert stage["native_energy"]["started_measurement_count"] == 0
    assert stage["native_energy"]["reused_measurement_count"] == 0
    assert '142/222 PASS, 80 FAIL' in stage["energy_not_started_reason"]
    assert 'resume_failure.json' in stage["energy_not_started_reason"]
    assert stage["energy_not_started_categories"] == ['native_runtime_preflight']
    assert historical.read_bytes() == historical_bytes


def test_normal_reporter_merges_previous_and_new_with_terminal_projection(tmp_path, monkeypatch):
    manifest = tmp_path / "b003/split_manifest.json"
    write(manifest, {"part2_external_inputs": ["boundary", "images"]})
    proof = {"model": "resnet50", "case": "b003", "split_manifest": str(manifest),
             "part2_input_names": ["boundary", "images"], "part2_input_count": 2,
             "reason": "part2_input_count_not_one"}
    projection = native_subset_projection([job("b003")], [proof])
    previous = tmp_path / "previous.json"
    old_row = {**job(), "ok": True, "tool_version": "2.91.0", "repetition_records": [{"repetition_id": "old"}]}
    write(previous, {"rows": [old_row]})
    out = tmp_path / "reports"
    write(out / "native_producer_stage.json", {"native_not_supported_rows": projection})
    current = {**job("b002"), "ok": True, "tool_version": "2.91.1"}
    root = tmp_path / "attempt"; root.mkdir()
    monkeypatch.setattr(reporter, "_rows_from_native_fifo_roots", lambda *_: [current])
    monkeypatch.setattr(reporter, "_aggregate_repetitions", lambda rows: rows)
    monkeypatch.setattr(sys, "argv", ["reporter", "--root", str(root), "--out-dir", str(out),
                                      "--previous-summary", str(previous)])
    assert reporter.main() == 0
    rows = json.loads((out / "native_producer_combined_summary.json").read_text())["rows"]
    by_case = {row["case"]: row for row in rows}
    assert set(by_case) == {"b001", "b002", "b003"}
    assert by_case["b001"]["tool_version"] == "2.91.0"
    assert by_case["b001"]["repetition_records"] == old_row["repetition_records"]
    assert by_case["b003"]["status"] == "not_supported"
    assert json.loads(previous.read_text())["rows"] == [old_row]
