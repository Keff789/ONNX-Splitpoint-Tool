"""Legacy preparation recovery uses temporary existing locks and no real DUT."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import yaml

from onnx_splitpoint_tool.remote import process_lease, resource_recovery as recovery
from onnx_splitpoint_tool.workflow import run_control


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: tmp_path / "locks/interlock")
    root = tmp_path / "old-report"
    run = root / "hardware_runs/mobilenet_v3_large__setup" / recovery._PRE_INFERENCE_RUN
    setup, session, pid = "fixture-deepx", "a" * 32, 99999991
    identity = {"model": "mobilenet_v3_large", "case": "b027", "backend": "deepx_to_trt", "setup_id": setup}
    base = run / ("native/mobilenet_v3_large__" + setup + "__b027")
    directory = base / "numerical"
    source = ("def main():\n"
              " report={'runner_sha256': _assert_expected_sha256('runner', Path(__file__).resolve(), ns.expected_runner_sha256)}\n"
              " from dx_engine import InferenceEngine\n"
              " engine=InferenceEngine(str(dxnn))\n"
              " trt=NativeTRT(engine_path)\n")
    worker_sha = hashlib.sha256(source.encode()).hexdigest()
    archive = root / "release/original.zip"
    archive.parent.mkdir(parents=True)
    member = "ONNX-Splitpoint-Tool_v2.95.0/scripts/native_deepx_trt_e2e_from_benchmarkset.py"
    dependencies = {"scripts/native_hailo10_trt_e2e_from_benchmarkset.py": "class NativeTRT: pass\n",
                    "onnx_splitpoint_tool/native_deepx_async.py": "class NativeDeepXAsync: pass\n"}
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, source)
        for relative, data in dependencies.items():
            bundle.writestr("ONNX-Splitpoint-Tool_v2.95.0/" + relative, data)
    monkeypatch.setattr(recovery, "_PRE_INFERENCE_WORKER", worker_sha)
    monkeypatch.setattr(recovery, "_PRE_INFERENCE_ARCHIVE", digest(archive))
    proof = root / "notes/source-proof.json"
    write(proof, {"archive": str(archive), "archive_sha256": digest(archive), "worker_member": member,
                  "worker_sha256": worker_sha, "observed_failure_worker_sha256": worker_sha,
                  "source_excerpt": [{"line": 99999, "text": "untrusted explanatory positions are ignored"}]})
    target = {"remote": {"host": "192.0.2.102", "user": "fixture"}}
    write(base / "PREPARED.json", {"identity": identity, "target": target})
    argv = ["/venv/bin/python", "/tool/scripts/native_deepx_trt_e2e_from_benchmarkset.py",
            "--expected-runner-sha256", "1" * 64, "--deepx-runtime-mode", "A4",
            "--deepx-numerical-only", "--frames", "64", "--warmup", "0", "--repetitions", "1", "--duration-s", "0"]
    raw = directory / "download/outputs/deepx_native_fifo_e2e_results.json"
    write(raw, {"ok": False, "energy_workload_only": False,
                "error": "RuntimeError: replay artifact SHA-256 mismatch for runner: expected=" + "1" * 64
                + " actual=" + worker_sha + " path=" + argv[1]})
    monkeypatch.setattr(recovery, "_PRE_INFERENCE_RAW", digest(raw))
    result = base / "NUMERICAL_RESULT.json"
    write(result, {"identity": identity, "status": "failed", "new_capture_count": 0, "new_performance_processes": 0,
                   "worker_report": {"path": str(raw), "sha256": digest(raw)}, "serial_reference_count": None})
    write(directory / "COMMAND.json", {"argv": argv})
    write(directory / "DISPATCH.json", {"argv": argv, "returncode": 4})
    steps = [{
        "name": "verify_remote_script_native_deepx_trt_e2e_from_benchmarkset", "rc": 0,
        "sha256_ok": True, "expected_sha256": worker_sha, "remote_sha256": worker_sha}]
    for relative, data in dependencies.items():
        name = ("verify_remote_script_native_hailo10_trt_e2e_from_benchmarkset" if relative.startswith("scripts/")
                else "verify_remote_asset_onnx_splitpoint_tool_native_deepx_async.py")
        sha = hashlib.sha256(data.encode()).hexdigest()
        steps.append({"name": name, "rc": 0, "sha256_ok": True, "expected_sha256": sha, "remote_sha256": sha})
    write(base / "RUNTIME_DEPLOYMENT.json", {"status": "verified", "steps": steps})
    write(run / "run_manifest.json", {"run_id": run.name, "run_dir": str(run), "status": "partial", "current_session_id": session})
    write(run / "jobs/workflow_control.json", {"run_id": run.name, "session_id": session, "state": "finished"})
    write(run / "reports/artifact_index_closure.json", {"run_id": run.name, "session_id": session, "status": "pass"})
    journal = run / "jobs/remote_process_leases" / session
    write(journal / "controller.resources.json", {"run_id": run.name, "run_dir": str(run), "session_id": session,
        "parent_pid": pid, "parent_start_ticks": "1234", "setups": {setup: ["dut:192.0.2.102"]}})
    request = {"run_id": run.name, "session_id": session, "operation_id": "transfer-a", "token": "b" * 32,
               "setup_id": setup, "purpose": "transfer", "phase": "FINALIZING", "process_started": False}
    write(journal / "transfer-a.resource-request.json", request)
    write(journal / "transfer-a.resource-reply.json", {**request, "state": "RELEASED"})
    owner = {"run_id": run.name, "session_id": session, "pid": pid}
    run_lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    run_lock.acquire()
    run_lock.release()
    lock = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={**owner, "purpose": "hardware_acceptance_native_numerical"})
    lock.acquire()
    lock.commit_quarantine_fence({"reason": recovery._PRE_INFERENCE_REASON, "evidence": str(directory)})
    lock.release()
    write(root / "evidence/PHYSICAL_DUT_QUARANTINE_ORIGINAL.json", json.loads(lock.quarantine_path.read_text()))
    write(root / "evidence/PHYSICAL_DUT_QUARANTINE_BINDING.json", {"original_path": str(lock.quarantine_path),
        "sha256": digest(lock.quarantine_path), "resource": "dut:192.0.2.102"})
    registry = tmp_path / "registry.yaml"
    registry.write_text(yaml.safe_dump({"schema_version": 1, "hardware_setups": [{"id": setup,
        "host": {"address": "192.0.2.102", "user": "fixture", "port": 22}}]}))
    observed = []
    def observe(_setup, **kwargs):
        observed.append(kwargs)
        competing = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={})
        with pytest.raises(run_control.WorkflowRunLockedError):
            competing.acquire_for_recovery()
        return {"ok": True, "controller": {"recorded_owners": [{"state": "absent"}]}, "remote": {"read_only": True}}
    monkeypatch.setattr(recovery, "observe_resource_owners", observe)
    return SimpleNamespace(root=root, run=run, base=base, directory=directory, proof=proof, raw=raw, result=result,
        journal=journal, lock=lock, archive=archive, dependencies=dependencies, observed=observed,
        kwargs={"run_dir": run, "setup_id": setup, "source_proof": proof, "operator": "fixture-operator",
                "evidence_output": tmp_path / "new-report/recovery/result.json", "registry_path": registry})


def test_exact_case_releases_under_lock_preserves_originals_and_repeats(case):
    original = {path: path.read_bytes() for path in case.root.rglob("*") if path.is_file()}
    fence = case.lock.quarantine_path.read_bytes()
    assert recovery.recover_pre_inference_resource(**case.kwargs)["status"] == "released"
    assert case.observed and not case.lock.quarantine_path.exists()
    receipt = json.loads(case.kwargs["evidence_output"].read_text())
    assert receipt["original_quarantine_bytes"].encode() == fence
    assert receipt["pre_inference_proof"]["raw_counters_rewritten"] is False
    assert receipt["old_attempt_validated"] is False and receipt["measurements_started"] == 0
    assert all(path.read_bytes() == data for path, data in original.items())
    assert json.loads(case.result.read_text())["serial_reference_count"] is None
    assert recovery.recover_pre_inference_resource(**case.kwargs)["status"] == "already_released"
    assert len(case.observed) == 1


@pytest.mark.parametrize("fault", ["proof_missing", "archive_changed", "deployment_changed", "import_dependency_changed", "foreign_owner", "foreign_fence", "active_owner", "open_job", "snapshot_changed", "capture", "different_dut", "run_active", "run_quarantined"])
def test_invalid_or_busy_original_never_releases(case, monkeypatch, fault):
    if fault == "proof_missing":
        case.proof.unlink()
    elif fault == "archive_changed":
        case.archive.write_bytes(b"different archive")
    elif fault == "deployment_changed":
        write(case.base / "RUNTIME_DEPLOYMENT.json", {"status": "failed", "steps": []})
    elif fault == "import_dependency_changed":
        path = case.base / "RUNTIME_DEPLOYMENT.json"; value = json.loads(path.read_text()); value["steps"][1]["remote_sha256"] = "0" * 64; write(path, value)
    elif fault == "foreign_owner":
        value = json.loads(case.lock.lock_path.read_text()); value["pid"] += 1; write(case.lock.lock_path, value)
    elif fault == "foreign_fence":
        value = json.loads(case.lock.quarantine_path.read_text()); value["owner"]["reason"] = "other SDK stop"; write(case.lock.quarantine_path, value)
    elif fault == "active_owner":
        monkeypatch.setattr(recovery, "observe_resource_owners", lambda *a, **k: {"ok": False})
    elif fault == "open_job":
        write(case.journal / "pending.remote-lease.json", {"state": "unknown"})
    elif fault == "snapshot_changed":
        def changed(*args, **kwargs):
            write(case.directory / "COMMAND.json", {"argv": []}); return {"ok": True}
        monkeypatch.setattr(recovery, "observe_resource_owners", changed)
    elif fault == "capture":
        path = case.journal / "transfer-a.resource-request.json"; value = json.loads(path.read_text()); value["purpose"] = "capture"; write(path, value)
    elif fault == "different_dut":
        value = yaml.safe_load(case.kwargs["registry_path"].read_text()); value["hardware_setups"][0]["host"]["address"] = "192.0.2.103"; case.kwargs["registry_path"].write_text(yaml.safe_dump(value))
    elif fault == "run_active":
        path = case.run / "jobs/workflow_control.json"; value = json.loads(path.read_text()); value["state"] = "running"; write(path, value)
    elif fault == "run_quarantined":
        lock = run_control.EvaluationRunLock(out_root=case.run.parent, run_dir=case.run, owner={})
        write(lock.quarantine_path, {"reason": "independent unresolved local cleanup"})
    original = case.lock.quarantine_path.read_bytes()
    with pytest.raises((run_control.WorkflowRunControlError, OSError)):
        recovery.recover_pre_inference_resource(**case.kwargs)
    assert case.lock.quarantine_path.read_bytes() == original


@pytest.mark.parametrize("raw", [
    {"ok": False, "energy_workload_only": False, "steps": []},
    {"ok": False, "energy_workload_only": False, "error": "TimeoutError after submit", "accepted": 1},
    {"ok": False, "energy_workload_only": False, "error": "SSH disconnected", "completed": None},
    {"ok": True, "deepx_final_drain_complete": True, "accepted": 64, "completed": 64},
])
def test_missing_counts_timeout_disconnect_or_drain_are_not_this_release_ground(case, monkeypatch, raw):
    write(case.raw, raw)
    monkeypatch.setattr(recovery, "_PRE_INFERENCE_RAW", digest(case.raw))
    result = json.loads(case.result.read_text()); result["worker_report"]["sha256"] = digest(case.raw); write(case.result, result)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="exact pre-inference runner guard"):
        recovery.recover_pre_inference_resource(**case.kwargs)
    assert case.lock.quarantine_path.exists() and not case.observed


def test_final_receipt_write_failure_restores_exact_fence_and_owner(case, monkeypatch):
    fence, owner = case.lock.quarantine_path.read_bytes(), case.lock.lock_path.read_bytes()
    normal = recovery._write
    def fail(path, value):
        if Path(path) == case.kwargs["evidence_output"] and value.get("status") == "released":
            raise OSError("fixture disk full")
        return normal(path, value)
    monkeypatch.setattr(recovery, "_write", fail)
    with pytest.raises(OSError, match="disk full"):
        recovery.recover_pre_inference_resource(**case.kwargs)
    assert case.lock.quarantine_path.read_bytes() == fence
    assert json.loads(case.lock.lock_path.read_text()) == json.loads(owner)


def test_failure_after_unlink_restores_exact_original_fence(case, monkeypatch):
    original = case.lock.quarantine_path.read_bytes()
    normal, failed = Path.unlink, False
    def unlink(path, *args, **kwargs):
        nonlocal failed
        result = normal(path, *args, **kwargs)
        if path == case.lock.quarantine_path and not failed:
            failed = True
            raise OSError("fixture directory sync failure after unlink")
        return result
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="after unlink"):
        recovery.recover_pre_inference_resource(**case.kwargs)
    assert case.lock.quarantine_path.read_bytes() == original


def test_source_order_is_checked_independently_of_explanatory_claims(case, monkeypatch):
    proof = json.loads(case.proof.read_text())
    source = ("def main():\n engine=InferenceEngine('model')\n"
              " evidence=_assert_expected_sha256('runner', Path(__file__).resolve(), ns.expected_runner_sha256)\n")
    worker_sha = hashlib.sha256(source.encode()).hexdigest()
    with zipfile.ZipFile(case.archive, "w") as bundle:
        bundle.writestr(proof["worker_member"], source)
        for relative, data in case.dependencies.items():
            bundle.writestr("ONNX-Splitpoint-Tool_v2.95.0/" + relative, data)
    proof.update(archive_sha256=digest(case.archive), worker_sha256=worker_sha,
                 observed_failure_worker_sha256=worker_sha, current_worker_matches_archive=True)
    write(case.proof, proof)
    monkeypatch.setattr(recovery, "_PRE_INFERENCE_WORKER", worker_sha)
    monkeypatch.setattr(recovery, "_PRE_INFERENCE_ARCHIVE", digest(case.archive))
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="precede all SDK/TRT"):
        recovery.recover_pre_inference_resource(**case.kwargs)
    assert case.lock.quarantine_path.exists() and not case.observed


def test_held_dut_lock_prevents_recovery(case):
    blocker = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={})
    blocker.acquire_for_recovery()
    try:
        with pytest.raises(run_control.WorkflowRunLockedError):
            recovery.recover_pre_inference_resource(**case.kwargs)
    finally:
        blocker.release(write_released_state=False)
    assert case.lock.quarantine_path.exists() and not case.observed


def test_old_success_cannot_release_a_foreign_later_fence(case):
    recovery.recover_pre_inference_resource(**case.kwargs)
    later = run_control.EvaluationRunLock.for_resource("dut:192.0.2.102", owner={"run_id": "later"})
    later.acquire(); later.commit_quarantine_fence({"reason": "unknown SDK completion"}); later.release()
    before = later.quarantine_path.read_bytes()
    with pytest.raises(run_control.WorkflowRunControlError):
        recovery.recover_pre_inference_resource(**case.kwargs)
    assert later.quarantine_path.read_bytes() == before


def test_supported_cli_uses_exact_recovery_operation(case, capsys):
    kwargs = case.kwargs
    args = ["recover-pre-inference"]
    for name in ("run_dir", "setup_id", "source_proof", "operator", "evidence_output"):
        args.extend(["--" + name.replace("_", "-"), str(kwargs[name])])
    args.extend(["--registry", str(kwargs["registry_path"])])
    assert process_lease.process_lease_cli_main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "released"
