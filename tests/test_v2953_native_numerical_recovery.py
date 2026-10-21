"""Generic Native numerical recovery uses only temporary locks and fake owners."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.remote import process_lease, resource_recovery as recovery
from onnx_splitpoint_tool.native_deepx_async import NativeExecutionLifecycle
from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical, run_control
from test_v2952_split_reference_completion import actual_worker_report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reindex(case):
    """Construct a different coherent fixture before exercising the validator."""
    write(case.run / "reports/acceptance_prepared/cell-native.json", json.loads(case.prepared.read_text()))
    write(case.run / "reports/acceptance_results/cell-native.json", json.loads(case.result.read_text()))
    path = case.run / "artifact_index.json"
    index = json.loads(path.read_text())
    for record in index["artifacts"]:
        source = case.run / record["path"]
        record.update(sha256="sha256:" + sha(source), size_bytes=source.stat().st_size)
    write(path, index)


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: tmp_path / "locks/interlock")
    writer = tmp_path / "writer"
    writer.mkdir()
    code, raw, _ = actual_worker_report(writer, monkeypatch)
    assert code == 0
    # The independent completion evidence must work with the old missing field.
    del raw["deepx_numerical_preflight"]["serial_reference_calls_completed"]
    contract = raw["native_command_contract"]
    identity = {key: contract[key] for key in ("model", "setup_id", "case", "backend")}
    run = tmp_path / "ordinary_runs" / "an_unrelated_native_attempt_name"
    base = run / "native" / "__".join(identity[key] for key in ("model", "setup_id", "case"))
    directory = base / "numerical"
    session, pid = "a" * 32, 99999991
    prepared = {"schema": "onnx-splitpoint/hardware-acceptance-native-plan/v1", "identity": identity,
        "output": str(base), "deepx_runtime_mode": "A4", "source_contract": contract,
        "phase_matrix": {"path": "/original/FINITE_MATRIX.json", "sha256": "b" * 64},
        "target": {"remote": {"host": "192.0.2.102", "user": "fixture"}},
        "target_binding": {"ssh": {"host": "192.0.2.102", "user": "fixture", "port": 22}},
        "remote_tool_dir": "/tool"}
    argv = ["/venv/bin/python", "/tool/scripts/native_deepx_trt_e2e_from_benchmarkset.py",
        "--benchmark-set", "/private/rep01/benchmark_set", "--case", identity["case"], "--image", "/original/timing.jpg",
        "--out-dir", "/private/rep01", "--expected-runner-sha256", raw["runner_sha256"],
        "--frames", "1000", "--warmup", "100", "--repetitions", "1", "--duration-s", "0", "--deepx-runtime-mode", "A4"]
    prepared["runs"] = [{"argv": argv, "env": {"ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "0", "PYTHONDONTWRITEBYTECODE": "1"},
        "subprocess_only_env": True, "remote_output_root": "/private/rep01", "result_json": "/private/rep01/deepx_native_fifo_e2e_results.json"}]
    for name in ("source_plan", "source_raw_report", "source_validation"):
        path = tmp_path / (name + ".json")
        write(path, {"original": name})
        prepared[name], prepared[name + "_sha256"] = str(path), sha(path)
    prepared_path = base / "PREPARED.json"
    write(prepared_path, prepared)
    command = numerical._numerical_item(prepared)
    images = [command["numerical_benchmark_set"] + f"/resources/numerical/{i:02d}.png" for i in range(32)]
    write(directory / "image_bundle/images.json", {"images": images})
    numerical._set_option(command["argv"], "--image", images[0])
    numerical._set_option(command["argv"], "--deepx-numerical-images", command["numerical_benchmark_set"] + "/resources/numerical/images.json")
    raw_path = directory / "download/outputs/deepx_native_fifo_e2e_results.json"
    write(raw_path, raw)
    write(directory / "COMMAND.json", command)
    write(directory / "DISPATCH.json", {"argv": command["argv"], "returncode": 0})
    diagnostic = base / "diagnostics/ssh_original.json"
    write(diagnostic, {"command": "cd -- /tool && " + shlex.join(command["argv"]), "returncode": 0,
        "cleanup_proven": True, "target_id": identity["setup_id"]})
    result_path = base / "NUMERICAL_RESULT.json"
    write(result_path, {"schema": numerical.SCHEMA, "identity": identity, "status": "failed", "attempt_state": "terminal",
        "fresh_hardware_executed": True, "new_capture_count": 0, "new_performance_processes": 0,
        "phase_matrix": prepared["phase_matrix"], "source_contract_sha256": contract["contract_sha256"],
        "worker_report": {"path": str(raw_path), "sha256": sha(raw_path)},
        "owned_worker_closure": {"proven": True, "diagnostic": {"path": str(diagnostic), "sha256": sha(diagnostic)}}})
    write(base / "RUNTIME_DEPLOYMENT.json", {"status": "verified", "steps": [{
        "name": "verify_remote_script_native_deepx_trt_e2e_from_benchmarkset", "rc": 0, "sha256_ok": True,
        "expected_sha256": raw["runner_sha256"], "remote_sha256": raw["runner_sha256"]}]})
    write(run / "run_manifest.json", {"run_id": run.name, "run_dir": str(run), "current_session_id": session, "status": "partial"})
    write(run / "jobs/workflow_control.json", {"run_id": run.name, "session_id": session, "state": "finished"})
    write(run / "reports/artifact_index_closure.json", {"run_id": run.name, "session_id": session, "status": "pass", "workflow_status": "partial"})
    journal = run / "jobs/remote_process_leases" / session
    write(journal / "controller.resources.json", {"run_id": run.name, "run_dir": str(run), "session_id": session,
        "parent_pid": pid, "parent_start_ticks": "1234", "setups": {identity["setup_id"]: ["dut:192.0.2.102"]}})
    request = {"run_id": run.name, "session_id": session, "operation_id": "transfer-a", "token": "b" * 32,
        "setup_id": identity["setup_id"], "purpose": "transfer", "phase": "FINALIZING", "process_started": False}
    write(journal / "transfer-a.resource-request.json", request)
    write(journal / "transfer-a.resource-reply.json", {**request, "state": "RELEASED"})
    owner = {"run_id": run.name, "session_id": session, "pid": pid}
    run_lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    run_lock.acquire(); run_lock.release()
    lock = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={**owner, "purpose": "hardware_acceptance_native_numerical"})
    lock.acquire(); lock.commit_quarantine_fence({"reason": recovery._PRE_INFERENCE_REASON, "evidence": str(directory)}); lock.release()
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"schema_version": 1, "hardware_setups": [{"id": identity["setup_id"],
        "host": {"address": "192.0.2.102", "user": "fixture", "port": 22}}]}))
    observations = []
    def observe(*args, **kwargs):
        observations.append(kwargs)
        competitor = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={})
        with pytest.raises(run_control.WorkflowRunLockedError):
            competitor.acquire_for_recovery()
        return {"ok": True, "read_only": True}
    monkeypatch.setattr(recovery, "observe_resource_owners", observe)
    write(run / "reports/acceptance_prepared/cell-native.json", prepared)
    write(run / "reports/acceptance_results/cell-native.json", json.loads(result_path.read_text()))
    # Match the normal terminal index: Native leaves are joined through indexed
    # prepared/result projections; they are not themselves directly indexed.
    records = [{"path": str(path.relative_to(run)), "sha256": "sha256:" + sha(path), "size_bytes": path.stat().st_size}
        for path in run.rglob("*") if path.is_file() and path.relative_to(run).parts[0] != "native"]
    write(run / "artifact_index.json", {"run_id": run.name, "terminal_closure": {"status": "pass", "session_id": session}, "artifacts": records})
    return SimpleNamespace(run=run, base=base, directory=directory, prepared=prepared_path, raw=raw_path,
        result=result_path, diagnostic=diagnostic, journal=journal, lock=lock, observations=observations,
        kwargs={"run_dir": run, "prepared_path": prepared_path, "setup_id": identity["setup_id"],
            "operator": "fixture-operator", "evidence_output": tmp_path / "new-report/recovery/result.json", "registry_path": registry})


def test_generic_completed_numerical_recovery_keeps_failed_result_missing_counter_and_fence_history(case):
    old = {path: path.read_bytes() for path in case.run.rglob("*") if path.is_file()}
    fence = case.lock.quarantine_path.read_bytes()
    assert recovery.recover_native_numerical_resource(**case.kwargs)["status"] == "released"
    receipt = json.loads(case.kwargs["evidence_output"].read_text())
    assert receipt["original_quarantine_bytes"].encode() == fence
    assert receipt["raw_counters_rewritten"] is receipt["old_attempt_validated"] is False
    assert receipt["measurements_started"] == 0 and case.observations
    assert all(path.read_bytes() == data for path, data in old.items())
    assert "serial_reference_calls_completed" not in json.loads(case.raw.read_text())["deepx_numerical_preflight"]
    assert json.loads(case.result.read_text())["status"] == "failed"
    assert recovery.recover_native_numerical_resource(**case.kwargs)["status"] == "already_released"
    assert len(case.observations) == 1


def test_generic_explicit_pre_runtime_failure_has_separate_safe_release_ground(case):
    original = json.loads(case.raw.read_text())
    raw = {"ok": False, "error": "pre-runtime software guard rejected", "case": "b027", "deepx_runtime_mode": "A4",
        "runner_sha256": original["runner_sha256"], "deepx_execution_lifecycle": NativeExecutionLifecycle().report()}
    write(case.raw, raw)
    result = json.loads(case.result.read_text()); result["worker_report"]["sha256"] = sha(case.raw); write(case.result, result)
    dispatch = json.loads((case.directory / "DISPATCH.json").read_text()); dispatch["returncode"] = 4; write(case.directory / "DISPATCH.json", dispatch)
    diagnostic = json.loads(case.diagnostic.read_text()); diagnostic["returncode"] = 4; write(case.diagnostic, diagnostic)
    result["owned_worker_closure"]["diagnostic"]["sha256"] = sha(case.diagnostic); write(case.result, result)
    reindex(case)
    assert recovery.recover_native_numerical_resource(**case.kwargs)["status"] == "released"
    receipt = json.loads(case.kwargs["evidence_output"].read_text())
    assert receipt["device_completion_evidence"]["pre_runtime_proven"] is True
    assert json.loads(case.raw.read_text()) == raw


@pytest.mark.parametrize("fault", ["raw_sha", "argv", "env", "deployment", "cleanup", "run_active", "foreign_owner", "foreign_fence", "open_lease", "capture_stop", "run_quarantined", "active_owner", "changed_snapshot", "new_resource_request"])
def test_generic_native_recovery_rejects_unbound_busy_or_changed_evidence(case, monkeypatch, fault):
    path, key, value = None, None, None
    if fault == "raw_sha": path, key, value = case.raw, "ok", False
    elif fault == "argv": path, key, value = case.directory / "DISPATCH.json", "argv", []
    elif fault == "env": path, key, value = case.directory / "COMMAND.json", "env", {"ONNX_SPLITPOINT_NATIVE_TRT_BUILD": "1"}
    elif fault == "deployment": path, key, value = case.base / "RUNTIME_DEPLOYMENT.json", "status", "failed"
    elif fault == "cleanup": path, key, value = case.diagnostic, "cleanup_proven", False
    elif fault == "run_active": path, key, value = case.run / "jobs/workflow_control.json", "state", "running"
    elif fault == "foreign_owner": path, key, value = case.lock.lock_path, "run_id", "another-run"
    elif fault == "foreign_fence": path, key, value = case.lock.quarantine_path, "owner", {"reason": "unknown SDK"}
    elif fault == "capture_stop": path, key, value = case.journal / "transfer-a.resource-request.json", "purpose", "capture"
    elif fault == "open_lease": write(case.journal / "still.remote-lease.json", {"state": "unknown"})
    elif fault == "run_quarantined":
        lock = run_control.EvaluationRunLock(out_root=case.run.parent, run_dir=case.run, owner={})
        write(lock.quarantine_path, {"reason": "independent unresolved run cleanup"})
    elif fault == "active_owner": monkeypatch.setattr(recovery, "observe_resource_owners", lambda *a, **k: {"ok": False})
    elif fault == "changed_snapshot":
        def change(*a, **k):
            write(case.directory / "COMMAND.json", {"argv": []}); return {"ok": True}
        monkeypatch.setattr(recovery, "observe_resource_owners", change)
    else:
        def new_request(*a, **k):
            write(case.journal / "late.resource-request.json", {"phase": "STARTING"}); return {"ok": True}
        monkeypatch.setattr(recovery, "observe_resource_owners", new_request)
    if path:
        document = json.loads(path.read_text()); document[key] = value; write(path, document)
    fence = case.lock.quarantine_path.read_bytes()
    with pytest.raises((run_control.WorkflowRunControlError, ValueError, OSError)):
        recovery.recover_native_numerical_resource(**case.kwargs)
    assert case.lock.quarantine_path.read_bytes() == fence


@pytest.mark.parametrize("field,value", [("runtime_calls_completed", 256), ("runtime_calls_completed", True), ("completion", "unknown")])
def test_generic_native_recovery_never_releases_unknown_or_mismatched_sdk_completion(case, field, value):
    raw = json.loads(case.raw.read_text()); raw["deepx_execution_lifecycle"][field] = value; write(case.raw, raw)
    result = json.loads(case.result.read_text()); result["worker_report"]["sha256"] = sha(case.raw); write(case.result, result)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="unsafe or unproven"):
        recovery.recover_native_numerical_resource(**case.kwargs)
    assert case.lock.quarantine_path.exists() and not case.observations


@pytest.mark.parametrize("raw", [{}, {"deepx_runtime_observation": {"pending": 0, "unreleased": 0}},
                                  {"deepx_final_drain_complete": True}])
def test_generic_recovery_rejects_absent_lifecycle_even_with_zero_pending_or_drain(case, raw):
    write(case.raw, raw)
    result = json.loads(case.result.read_text()); result["worker_report"]["sha256"] = sha(case.raw); write(case.result, result)
    reindex(case)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="unsafe or unproven"):
        recovery.recover_native_numerical_resource(**case.kwargs)
    assert case.lock.quarantine_path.exists() and not case.observations


def test_generic_native_recovery_final_receipt_failure_restores_fence_and_owner(case, monkeypatch):
    fence, owner = case.lock.quarantine_path.read_bytes(), case.lock.lock_path.read_bytes()
    normal = recovery._write
    def fail(path, value):
        if Path(path) == case.kwargs["evidence_output"] and value.get("status") == "released":
            raise OSError("fixture disk full")
        return normal(path, value)
    monkeypatch.setattr(recovery, "_write", fail)
    with pytest.raises(OSError, match="disk full"):
        recovery.recover_native_numerical_resource(**case.kwargs)
    assert case.lock.quarantine_path.read_bytes() == fence
    assert json.loads(case.lock.lock_path.read_text()) == json.loads(owner)


def test_rewriting_raw_and_result_binding_does_not_bypass_terminal_index(case):
    raw = json.loads(case.raw.read_text()); raw["unrelated_edit"] = "changed historical raw"; write(case.raw, raw)
    result = json.loads(case.result.read_text()); result["worker_report"]["sha256"] = sha(case.raw); write(case.result, result)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="indexed prepared/result projection differs"):
        recovery.recover_native_numerical_resource(**case.kwargs)
    assert case.lock.quarantine_path.exists() and not case.observations


def test_generic_native_recovery_cli_uses_bound_operation(case, capsys):
    args = ["recover-native-numerical"]
    for flag, name in (("run-dir", "run_dir"), ("prepared", "prepared_path"), ("setup-id", "setup_id"),
                       ("operator", "operator"), ("evidence-output", "evidence_output"), ("registry", "registry_path")):
        args += ["--" + flag, str(case.kwargs[name])]
    assert process_lease.process_lease_cli_main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "released"


def test_existing_recovery_receipt_never_releases_a_new_foreign_fence(case):
    recovery.recover_native_numerical_resource(**case.kwargs)
    later = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={"run_id": "later-job"})
    later.acquire(); later.commit_quarantine_fence({"reason": "unknown SDK completion"}); later.release()
    fence = later.quarantine_path.read_bytes()
    with pytest.raises(run_control.WorkflowRunControlError):
        recovery.recover_native_numerical_resource(**case.kwargs)
    assert later.quarantine_path.read_bytes() == fence


def test_held_physical_resource_lock_blocks_generic_recovery(case):
    blocker = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={})
    blocker.acquire_for_recovery()
    try:
        with pytest.raises(run_control.WorkflowRunLockedError):
            recovery.recover_native_numerical_resource(**case.kwargs)
    finally:
        blocker.release(write_released_state=False)
    assert case.lock.quarantine_path.exists() and not case.observations
