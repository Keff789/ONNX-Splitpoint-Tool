"""Offline Full context handoff; SDK/process fixtures are never hardware evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import statistics
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from onnx_splitpoint_tool.runners.request_latency import RequestLatency
from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
from onnx_splitpoint_tool.native_rate_endpoints import report_rate_fields
from scripts import native_producer_energy_plan as planner
from scripts import native_producer_final_report as reporter
from test_energy_final_contract_regressions import (
    _local_full_hotloop_contract, _planner_runtime_options, _seal_full_contract,
)
from test_v2955_full_semantic_artifact_roles import actual_full_case

TOOL = str(Path(__file__).resolve().parents[1])


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def context_record(root, setup="setup", tool=TOOL):
    return {"schema": "onnx-splitpoint/native-final-report-remote-execution-context",
            "schema_version": 1, "setup_id": setup,
            "remote_root": str(root), "remote_tool_dir": tool}


def contract_point(tmp_path):
    """Small sealed test contracts for command context boundaries only."""
    output = tmp_path / "acceptance"
    reports, rows, runs = [], [], []
    for repeat in (1, 2, 3):
        root = tmp_path / f"rep{repeat:02d}_inputs"
        root.mkdir(parents=True)
        contract = _local_full_hotloop_contract(root, "deepx")
        contract["runtime_options"] = {**_planner_runtime_options(), "frames": 1000,
            "warmup": 100, "deepx_runtime_mode": "A4", "deepx_max_inflight": 4}
        contract["energy_workload"].update(deepx_runtime_mode="A4", deepx_max_inflight=4)
        _seal_full_contract(contract)
        row = {key: contract[key] for key in ("backend", "model", "case", "setup_id", "comparison_backend")}
        latency = RequestLatency(1000, task_complete=True, start_anchor="test admission", end_anchor="test completion")
        for frame in range(1000):
            latency.start(frame, frame * 2_000_000)
            latency.complete(frame, frame * 2_000_000 + 1_000_000)
        row.update(full_command_contract=contract, full_command_contract_sha256=contract["contract_sha256"],
            repetition_count_valid=1, ok=True, outer_makespan_verified=True, completed_frames=1000,
            warmup_count=100, postprocess_completion_verified=True, fps_makespan=1000.,
            request_latency=latency.report(), repetition_records=[{"runtime_instance_id": f"test-process-{repeat}"}])
        report = write(output / f"rep{repeat:02d}/download/rep{repeat:02d}_outputs/analysis_tables/native_full_baseline_eval.json",
                       {"ok": True, "rows": [row]})
        rows.append(row)
        reports.append(report)
        runs.append({"repeat": repeat, "native_full": True, "env": {},
            "argv": [sys.executable, "worker.py", "--root", str(root), "--deepx-runtime-mode", "A4"],
            "full_benchmark_set": contract["benchmark_set"],
            "remote_output_root": f"/fresh/rep{repeat:02d}_outputs",
            "result_json": f"/fresh/rep{repeat:02d}_outputs/analysis_tables/native_full_baseline_eval.json"})
    identity = {key: rows[0][key] for key in ("backend", "model", "case", "setup_id")}
    validation = write(tmp_path / "quality.json", {"rows": []})
    prepared = {"identity": identity, "output": str(output), "deepx_runtime_mode": "A4",
        "source_primary_run": str(tmp_path / "authority"), "source_validation": str(validation),
        "remote_tool_dir": TOOL, "runs": runs, "source_contract": rows[0]["full_command_contract"],
        "target": {"remote": {"user": "nx", "host": "192.0.2.1"}}, "phase_matrix": {}}
    return SimpleNamespace(prepared=prepared, rows=rows, reports=reports, output=output)


def command_contexts(command):
    return reporter._parse_remote_execution_context_allowlist([
        command[index + 1] for index, value in enumerate(command)
        if value == "--remote-execution-context-json"])


def test_normal_import_passes_three_external_worker_roots_to_real_contract_reader(tmp_path, monkeypatch):
    point = contract_point(tmp_path)
    captured = []

    def report_process(command, context, **kwargs):
        captured.append(command)
        contexts = command_contexts(command)
        expected = {row["full_command_contract"]["root"] for row in point.rows}
        assert {record["remote_root"] for record in contexts} == expected
        assert all(record["remote_tool_dir"] == TOOL for record in contexts)
        for row in point.rows:
            digest, errors = reporter._verified_performance_command_contract(row, remote_execution_contexts=contexts)
            assert errors == []
            assert digest == row["full_command_contract"]["contract_sha256"]
        summary = copy.deepcopy(point.rows[0])
        summary["repetition_count_valid"] = 3
        write(point.output / "reports/native_producer_combined_summary.json", {"rows": [summary]})
        return SimpleNamespace(returncode=0, stdout="offline real contract-reader handoff", stderr="")

    monkeypatch.setattr(native, "_run_local", report_process)
    summary = native._normal_performance_import(point.prepared, point.reports, {"python": sys.executable})
    assert summary.is_file() and len(captured) == 1
    assert native.verify_performance_import(point.prepared, point.reports)["command_contract_sha256"] == point.rows[0]["full_command_contract_sha256"]


@pytest.mark.parametrize("mutation", ["foreign_root", "other_repeat_tensor", "tool_root", "setup"])
def test_three_root_allowlist_does_not_authorize_crossroot_role_or_identity(tmp_path, mutation):
    point = contract_point(tmp_path)
    contexts = reporter._parse_remote_execution_context_allowlist([
        context_record(row["full_command_contract"]["root"]) for row in point.rows])
    row = copy.deepcopy(point.rows[1])
    contract = row["full_command_contract"]
    if mutation == "foreign_root":
        contract["root"] = str(tmp_path / "foreign_inputs")
    elif mutation == "other_repeat_tensor":
        contract["artifacts"]["runtime_input_tensor"]["path"] = point.rows[0]["full_command_contract"]["artifacts"]["runtime_input_tensor"]["path"]
    elif mutation == "tool_root":
        contexts[1]["remote_tool_dir"] = str(tmp_path / "foreign_tool")
    else:
        row["setup_id"] = "foreign_setup"
    _seal_full_contract(contract)
    row["full_command_contract_sha256"] = contract["contract_sha256"]
    digest, errors = reporter._verified_performance_command_contract(row, remote_execution_contexts=contexts)
    assert digest == "" and errors


class CommandCaptured(Exception):
    def __init__(self, command):
        self.command = command


def energy_command(point, monkeypatch, summary_row):
    summary_row = copy.deepcopy(summary_row)
    summary_row["repetition_count_valid"] = 3
    summary = write(point.output / "reports/native_producer_combined_summary.json", {"rows": [summary_row]})
    write(point.output / "PREPARED.json", point.prepared)
    write(point.output / "performance_import.json", {"source_reports": [{"path": str(path), "sha256": native._sha(path)} for path in point.reports]})
    monkeypatch.setattr(native, "verify_performance_import", lambda *args: {
        "command_contract_sha256": summary_row["full_command_contract"]["contract_sha256"]})
    monkeypatch.setattr(native, "_native_phase_debt", lambda *args: {})
    monkeypatch.setattr(native, "_budget_counts", lambda *args: {})
    def capture(command, *args, **kwargs):
        raise CommandCaptured(command)
    monkeypatch.setattr(native, "_run_local", capture)
    context = {"python": sys.executable, "run_dir": str(point.output.parent), "budget_file": str(point.output / "budget.json"),
               "hardware_registry": str(point.output / "hardware.yaml"), "profile": {}}
    native._energy_capture(point.prepared, summary, context)


def test_energy_capture_uses_selected_second_repeat_context_not_global_or_first_root(tmp_path, monkeypatch):
    point = contract_point(tmp_path)
    selected = point.rows[1]
    with pytest.raises(CommandCaptured) as exc:
        energy_command(point, monkeypatch, selected)
    command = exc.value.command
    root = command[command.index("--remote-root") + 1]
    tool = command[command.index("--remote-tool-dir") + 1]
    assert root == selected["full_command_contract"]["root"]
    assert root != point.rows[0]["full_command_contract"]["root"]
    # Exercise the actual energy planner's context and strict role validator.
    actual = planner._full_row_remote_context(selected, SimpleNamespace(remote_root=root, remote_tool_dir=tool))
    expected = {**point.prepared["identity"], "remote_root": actual[0], "remote_tool_dir": actual[1]}
    contract, status = planner._verify_full_command_contract(selected["full_command_contract"], expected_identity=expected)
    assert contract is not None, status
    assert command[command.index("--runs") + 1] == "3"
    assert command[command.index("--duration-s") + 1] == "60"


def test_energy_capture_rejects_contract_from_another_accepted_cohort(tmp_path, monkeypatch):
    point = contract_point(tmp_path / "own")
    foreign = contract_point(tmp_path / "other")
    with pytest.raises(ValueError, match="root|context|scope|prepared|accepted"):
        energy_command(point, monkeypatch, foreign.rows[0])


def test_three_real_full_writer_processes_retain_seals_and_form_one_valid_cohort(tmp_path, monkeypatch):
    """Actual semantic/template/main writers; only physical boundaries simulated."""
    rows, original_seals, cases = [], [], []
    for repeat in (1, 2, 3):
        case = actual_full_case(tmp_path / f"rep{repeat:02d}_inputs", monkeypatch,
                                frames=1000, warmup=100, numerical_only=False)
        report_root = Path(case.ns.out_dir)
        rows.extend(reporter._rows_from_native_full(report_root))
        original_seals.append(case.command_contract["contract_sha256"])
        cases.append(case)
    assert len(set(original_seals)) == 3 and len(rows) == 3
    contexts = reporter._parse_remote_execution_context_allowlist([context_record(case.command_contract["root"], setup=case.setup_id) for case in cases])
    aggregate = reporter._aggregate_repetitions(rows, remote_execution_contexts=contexts)
    assert len(aggregate) == 1
    row = aggregate[0]
    assert row["ok"] is True
    assert row["repetition_count_valid"] == 3
    assert row["repetition_claim_identity_drift"] is False
    assert row["repetition_operating_point_verified"] is True
    assert [source["full_command_contract_sha256"] for source in rows] == original_seals
    assert len(row["source_reports"]) == 3

    # Continue through the existing normal import command and real reporter
    # main, replacing only its local subprocess boundary with an in-process
    # invocation. Quality is intentionally unavailable in this test fixture.
    output = tmp_path / "normal_import"
    reports = [write(output / f"rep{index:02d}/download/rep{index:02d}_outputs/analysis_tables/native_full_baseline_eval.json", case.raw)
               for index, case in enumerate(cases, 1)]
    identity = {key: cases[0].command_contract[key] for key in ("backend", "model", "case", "setup_id")}
    prepared = {"identity": identity, "output": str(output), "deepx_runtime_mode": "A4",
        "source_primary_run": str(tmp_path / "original_authority"),
        "source_validation": str(write(tmp_path / "unavailable_quality.json", {"rows": []})),
        "remote_tool_dir": TOOL, "source_contract": cases[0].command_contract,
        "runs": [{"repeat": index, "native_full": True, "env": {},
            "argv": [sys.executable, "worker.py", "--root", case.command_contract["root"], "--deepx-runtime-mode", "A4"],
            "full_benchmark_set": case.command_contract["benchmark_set"],
            "remote_output_root": f"/fresh/rep{index:02d}_outputs",
            "result_json": f"/fresh/rep{index:02d}_outputs/analysis_tables/native_full_baseline_eval.json"}
            for index, case in enumerate(cases, 1)]}
    write(output / "PREPARED.json", prepared)
    def real_reporter_process(command, context, **kwargs):
        module_index = command.index("-m") + 1
        with monkeypatch.context() as main_patch:
            main_patch.setattr(sys, "argv", [command[module_index], *command[module_index + 1:]])
            rc = reporter.main()
        return SimpleNamespace(returncode=rc, stdout="offline normal reporter", stderr="")
    monkeypatch.setattr(native, "_run_local", real_reporter_process)
    summary = native._normal_performance_import(prepared, reports, {"python": sys.executable})
    imported = json.loads(summary.read_text())["rows"][0]
    assert imported["repetition_count_valid"] == 3 and imported["ok"] is True
    assert imported["repetition_claim_identity_drift"] is False
    assert imported["quality_evidence_verified"] is False
    original_fps = [case.row["fps_makespan"] for case in cases]
    assert imported["fps_median"] == pytest.approx(statistics.median(original_fps))
    assert imported["completed_task_work_unit_counts"] == [1000, 1000, 1000]
    assert len(imported["completed_task_measurement_times_s"]) == 3
    assert all(value > 0 for value in imported["completed_task_measurement_times_s"])
    matrix = collect_native_performance_matrix(output)
    assert len(matrix["observations"]) == 1
    observed = matrix["observations"][0]
    assert observed["fps_median"] == pytest.approx(statistics.median(original_fps))
    assert observed["completed_task_work_unit_counts"] == [1000, 1000, 1000]
    assert observed["completed_task_measurement_times_s"] == imported["completed_task_measurement_times_s"]
    assert observed["quality_claim_result_verified"] is not True
    assert native.verify_performance_import(prepared, reports)["command_contract_sha256"] in original_seals
    replay_identity = {**identity, "remote_root": imported["full_command_contract"]["root"], "remote_tool_dir": TOOL}
    contract, status = planner._verify_full_command_contract(imported["full_command_contract"], expected_identity=replay_identity)
    assert contract is not None, status

    plan_dir = output / "offline_energy_plan"
    with monkeypatch.context() as plan_patch:
        plan_patch.setattr(sys, "argv", [planner.__file__, "--summary", str(summary),
            "--validation-summary", prepared["source_validation"], "--out-dir", str(plan_dir),
            "--deepx-ssh", "nx@192.0.2.1", "--remote-root", contract["root"], "--remote-tool-dir", TOOL,
            "--runs", "3", "--duration-s", "60", "--frames", "0", "--warmup", "0", "--allow-unpaired",
            "--physical-scope", "FS", "--require-runtime-work-units", "--require-command-window-alignment"])
        assert planner.main() == 0
    plan = json.loads((plan_dir / "native_producer_energy_plan.json").read_text())
    assert len(plan["rows"]) == 1
    selected = plan["rows"][0]
    assert selected["successful_command_contract_sha256"] == contract["contract_sha256"]
    assert selected["deepx_runtime_mode"] == "A4" and selected["deepx_max_inflight"] == 4
    assert plan["energy_runs_per_row"] == 3

    # Reopening a valid normal cohort must never turn its summary proof tag
    # into authority for changed raw observations, UUIDs or prepared roots.
    for key, value in (("runtime_instance_id", "another-runtime"),
                       ("completed_frames", 999), ("makespan_ms", 0.001)):
        changed = copy.deepcopy(imported)
        changed["repetition_records"][1][key] = value
        assert report_rate_fields(changed, output)["completed_task_fps"] is None
    prepared_bytes = (output / "PREPARED.json").read_bytes()
    changed_prepared = copy.deepcopy(prepared)
    changed_prepared["runs"][1]["argv"][3] = str(tmp_path / "foreign_root")
    write(output / "PREPARED.json", changed_prepared)
    assert report_rate_fields(imported, output)["completed_task_fps"] is None
    (output / "PREPARED.json").write_bytes(prepared_bytes)
    receipt = output / "performance_import.json"
    receipt_bytes = receipt.read_bytes()
    receipt.write_text("{}")
    assert report_rate_fields(imported, output)["completed_task_fps"] is None
    receipt.write_bytes(receipt_bytes)
    original_raw = reports[1].read_bytes()
    reports[1].write_bytes(original_raw + b"\n")
    assert report_rate_fields(imported, output)["completed_task_fps"] is None
    reports[1].write_bytes(original_raw)
    assert report_rate_fields(imported, output)["completed_task_work_unit_counts"] == [1000, 1000, 1000]
