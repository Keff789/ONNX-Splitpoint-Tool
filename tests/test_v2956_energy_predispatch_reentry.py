"""Preserve immutable zero-capture failures while selecting a normal attempt."""
import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.resources.remote_scripts import run_native_producer_energy_from_summary as runner
from onnx_splitpoint_tool.workflow import native_energy_reentry as repair
from onnx_splitpoint_tool.workflow.checkpoints import AtomicRowJournal, write_stage_checkpoint
from test_v2951_native_refresh import completed_cohort


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(tmp_path, kind="plan"):
    run = tmp_path / "normal-parent"
    output = run / "native/m__deepx__full"
    view = output / "energy_view"
    measurement = view / "reports/native_energy_measurements"
    identity = {"backend": "native_full_deepx" if kind == "plan" else "deepx_to_trt",
                "model": "m", "case": "full" if kind == "plan" else "b027", "setup_id": "deepx"}
    row = {**identity, "comparison_backend": "deepx", "precision": "" if kind == "plan" else "fp16", "duration_s": 60,
           "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
           "successful_command_contract_sha256": "a" * 64,
           "measurement_requested_repeats": 3, "measurement_effective_repeats": 3}
    plan = {"schema": "energy-plan", "schema_version": 1, "rows": [row],
            "physical_scope": "FS", "window_label": "command",
            "require_runtime_work_units": True, "require_command_window_alignment": True}
    report = {"plan_payload": plan, "rows": [], "started_measurement_count": 0,
              "collector_started_repeat_count": 0, "workload_started_repeat_count": 0}
    plan_hash, identities, hashes = runner._managed_journal_contract([row], plan)
    journal = AtomicRowJournal.create(measurement / "checkpoints/native_energy",
        plan_hash=plan_hash, rows=[row], identities=identities, row_contract_hashes=hashes)
    if kind == "plan":
        report.update(status="blocked_energy_plan_cohort_invalid", cohort_validation_errors=[{
            "error": "ValueError: energy_quality_admission_precision_drift"}])
        stage_state = "failed"
    else:
        empty_output = measurement / "measurements/empty-attempt"
        empty_output.mkdir(parents=True)
        result = {"row": row, "ok": False, "run": {"rc": 1,
            "stdout": "ValueError: final cohort requires non-diagnostic exact command-window task measurement",
            "measurement_start_observation": {"measurement_started": False,
                "collector_started_repeat_count": 0, "workload_started_repeat_count": 0}}}
        journal.transition(0, "running", execution={"output_dir": str(empty_output), "measurement_wrapper_started": True})
        journal.transition(0, "failed", result=result)
        report["rows"] = [result]
        stage_state = "running"
    source = measurement / ("native_producer_energy_results.json" if kind == "plan"
                            else "native_producer_energy_results.partial.json")
    write(source, report)
    write_stage_checkpoint(measurement / "stages/native_energy/stage_result.json", stage="native_energy",
        state=stage_state, complete=stage_state == "failed", input_hash="b" * 64, run_root=measurement)
    budget = run / "energy/energy_task_budget.json"
    row_id = json.dumps([identity[key] for key in ("backend", "model", "case", "setup_id")], separators=(",", ":"))
    write(budget, {"campaign": True, "max_transport_failures": 20,
        "sources": {"source": {"stop_reason": ""}}, "tasks": {row_id: {
            "source_id": "source", "stop_reason": "", "chains": [],
            "limits": {"max_chains": 9, "max_retries": 2, "max_transport_failures": 20}}}})
    prepared = {"identity": identity, "deepx_runtime_mode": "A4", "output": str(output),
                "target_binding": {"urecs_address": "source"}}
    context = {"resume": True, "finite_config": {"phase": "complete"},
               "budget_file": str(budget), "run_dir": str(run)}
    return prepared, context, view, source, row_id


def files(root):
    return {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("kind", ["plan", "claim"])
def test_known_zero_capture_failure_gets_fresh_normal_view_without_rewriting_history(tmp_path, kind):
    prepared, context, view, source, _ = fixture(tmp_path, kind)
    original = files(tmp_path)
    selected, previous = repair.select_energy_view(prepared, context, view, "a" * 64)
    assert selected.parent == view / "reports/native_energy_measurements/resume_attempts"
    assert selected.name.startswith("resume_") and not selected.exists()
    assert previous == source and files(tmp_path) == original
    # A later same-run Resume follows the real selected output and predecessor.
    measurement = selected / "reports/native_energy_measurements"
    write_stage_checkpoint(measurement / "stages/native_energy/stage_result.json", stage="native_energy",
        state="completed", complete=True, input_hash="c" * 64, run_root=measurement)
    write(Path(prepared["output"]) / "energy_dispatch.json", {
        "command": ["normal-runner", "--out-dir", str(measurement)], "pre_capture_predecessor": str(source)})
    assert repair.select_energy_view(prepared, context, view, "a" * 64) == (selected, source)


@pytest.mark.parametrize("mutation", ["chain", "source_stop", "task_stop", "physical_output", "observation", "contract"])
def test_predispatch_repair_cannot_bypass_physical_history_or_unknown_evidence(tmp_path, mutation):
    prepared, context, view, source, row_id = fixture(tmp_path, "claim")
    if mutation in {"chain", "source_stop", "task_stop"}:
        budget = repair._read(context["budget_file"])
        if mutation == "chain":
            budget["tasks"][row_id]["chains"] = [{"collector_started": True}]
        elif mutation == "source_stop":
            budget["sources"]["source"]["stop_reason"] = "campaign_source_completion_unresolved"
        else:
            budget["tasks"][row_id]["stop_reason"] = "task_chain_limit"
        write(Path(context["budget_file"]), budget)
    elif mutation == "physical_output":
        (source.parent / "measurements/empty-attempt/collector_stdout.log").write_text("physical evidence")
    elif mutation == "observation":
        journal_path = source.parent / "checkpoints/native_energy"
        report = repair._read(source)
        row = report["plan_payload"]["rows"][0]
        plan_hash, identities, hashes = runner._managed_journal_contract([row], report["plan_payload"])
        # A correctly sealed authoritative checkpoint can still contradict the summary.
        journal = AtomicRowJournal.open_for_resume(journal_path, plan_hash=plan_hash,
            identities=identities, row_contract_hashes=hashes, refresh_manifest=False)
        checkpoint = journal.rows[0]
        checkpoint["result"]["run"]["measurement_start_observation"]["collector_started_repeat_count"] = 1
        from onnx_splitpoint_tool.workflow.checkpoints import canonical_json_sha256
        checkpoint.pop("checkpoint_sha256")
        checkpoint["checkpoint_sha256"] = canonical_json_sha256(checkpoint)
        write(journal_path / "rows" / journal._row_filename(0, identities[0]), checkpoint)
    before = files(tmp_path)
    with pytest.raises(ValueError):
        repair.select_energy_view(prepared, context, view, "f" * 64 if mutation == "contract" else "a" * 64)
    assert files(tmp_path) == before


@pytest.mark.parametrize("change", ["unrelated_error", "captured", "not_resume", "hailo", "s1"])
def test_other_failures_and_backends_keep_existing_resume_path(tmp_path, change):
    prepared, context, view, source, _ = fixture(tmp_path, "claim")
    if change in {"unrelated_error", "captured"}:
        previous = repair._read(source)
        if change == "captured":
            previous["collector_started_repeat_count"] = 1
        else:
            previous["status"] = "unrelated"
            # This is reflected in the authoritative failed row, not just report prose.
            row_path = next((source.parent / "checkpoints/native_energy/rows").glob("*.json"))
            checkpoint = repair._read(row_path)
            checkpoint["result"]["run"]["stdout"] = "unrelated SDK failure"
            from onnx_splitpoint_tool.workflow.checkpoints import canonical_json_sha256
            checkpoint.pop("checkpoint_sha256")
            checkpoint["checkpoint_sha256"] = canonical_json_sha256(checkpoint)
            write(row_path, checkpoint)
        write(source, previous)
    elif change == "not_resume":
        context["resume"] = False
    elif change == "hailo":
        prepared["identity"]["backend"] = "hailo8_to_trt"
    else:
        prepared["deepx_runtime_mode"] = "S1"
    before = files(tmp_path)
    assert repair.select_energy_view(prepared, context, view, "a" * 64) == (view, None)
    assert files(tmp_path) == before


def test_normal_caller_uses_fresh_view_and_archives_original_dispatch(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
    prepared, context, view, source, _ = fixture(tmp_path, "claim")
    output = Path(prepared["output"])
    validation = tmp_path / "validation.json"
    write(validation, {})
    prepared.update(source_validation=str(validation), phase_matrix={}, remote_tool_dir="/normal-tool",
        target={"remote": {"user": "nx", "host": "example.invalid"}}, runs=[{"env": {}}])
    write(output / "PREPARED.json", prepared)
    summary = output / "reports/native_producer_combined_summary.json"
    write(summary, {"rows": [{**prepared["identity"], "native_command_contract": {"contract_sha256": "a" * 64}}]})
    original_dispatch = {"command": ["normal-runner", "--out-dir", str(source.parent)], "returncode": 1}
    write(output / "energy_dispatch.json", original_dispatch)
    original_bytes = (output / "energy_dispatch.json").read_bytes()
    context.update(python="unused-python", hardware_registry="unused-registry", profile={})
    monkeypatch.setattr(native, "verify_performance_import", lambda *args: {"command_contract_sha256": "a" * 64})
    monkeypatch.setattr(native, "_native_phase_debt", lambda *args: {})
    monkeypatch.setattr(native, "_budget_counts", lambda *args: {})
    monkeypatch.setattr(native, "_authorize_energy_cohort", lambda *args: {"cohort_id": Path(context["run_dir"]).name})
    captured = []

    class Capture(Exception):
        pass

    def command(argv, *args, **kwargs):
        captured.append(argv)
        raise Capture

    monkeypatch.setattr(native, "_run_local", command)
    with pytest.raises(Capture):
        native._energy_capture(prepared, summary, context)
    argv = captured[0]
    measurement = Path(argv[argv.index("--out-dir") + 1])
    selected = measurement.parent.parent
    assert selected.parent == source.parent / "resume_attempts"
    assert "--resume-checkpoint" not in argv and "--previous-results" not in argv
    assert argv[argv.index("--campaign-budget-file") + 1] == context["budget_file"]
    assert argv[argv.index("--campaign-cohort") + 1] == Path(context["run_dir"]).name
    assert str(selected) in argv[argv.index("--completed-row-validator") + 1]
    assert (selected / "resume_history/energy_dispatch.json").read_bytes() == original_bytes
    pending = native._read(output / "energy_dispatch.json")
    assert pending["returncode"] is None and pending["pending"] is True
    assert repair.select_energy_view(prepared, context, view, "a" * 64) == (selected, source)


def test_nested_success_flows_through_cohort_and_same_run_normal_readers(completed_cohort, monkeypatch):
    from onnx_splitpoint_tool.workflow import native_refresh as refresh
    native = refresh.native
    cell, prepared, reference, output, energy = completed_cohort
    original_view = output / "energy_view"
    predecessor = Path(energy["source"])
    old_bytes = predecessor.read_bytes()
    selected = original_view / "reports/native_energy_measurements/resume_attempts/resume_portable"
    measurement = selected / "reports/native_energy_measurements"
    contract = measurement / "command_contract.json"
    write(contract, native._read(Path(energy["command_contract_source"])))
    payload = json.loads(old_bytes)
    payload["rows"][0]["row"].update(command_contract_file=str(contract), command_contract_file_sha256=native._sha(contract))
    source = measurement / "native_producer_energy_results.json"
    write(source, payload)
    energy.update(source=str(source), command_contract_source=str(contract))
    write(output / "energy_dispatch.json", {"command": ["normal-runner", "--out-dir", str(measurement)],
        "pre_capture_predecessor": str(predecessor), "start_accounting_before": {"collector_starts": 3, "valid_repeats": 3}})
    views = []

    def collect(view, **kwargs):
        views.append(Path(view))
        return [energy] if Path(view) == selected else []

    monkeypatch.setattr(native, "collect_native_energy", collect)
    proof = refresh._cohort_proof(cell, prepared, reference)
    assert proof["energy_result"]["path"] == str(source)
    write(output / "PERFORMANCE_RESUME.json", {"dispatch_count_before": 3})
    monkeypatch.setattr(native, "_budget_counts", lambda *args: {"collector_starts": 3, "valid_repeats": 3})
    result = {"performance_summary": str(output / "reports/native_producer_combined_summary.json"),
              "new_capture_count": 0, "new_performance_processes": 0,
              "retained_valid_energy_repeats": 3, "fresh_hardware_executed": False}
    assert native.verify_same_run_resume_result(result, {"resume": True, "prepared_native": prepared,
        "run_dir": str(output.parent.parent)})
    assert views == [selected, selected] and predecessor.read_bytes() == old_bytes
