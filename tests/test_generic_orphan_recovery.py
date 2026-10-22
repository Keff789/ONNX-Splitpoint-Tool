"""Bounded Generic orphan recovery: temporary locks and controlled processes.

No test connects to hardware. Original process identities are captured from
real local children, rather than supplied through a purported cleanup flag.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.remote import resource_recovery
from onnx_splitpoint_tool.workflow import run_control


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _descriptor(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw)}


def _identity(pid: int) -> dict:
    raw = Path(f"/proc/{pid}/stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    return {"pid": pid, "ppid": int(fields[1]), "pgrp": int(fields[2]),
            "session_id": int(fields[3]), "process_start_ticks": int(fields[19]),
            "argv": [value.decode() for value in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if value]}


@pytest.fixture
def original_process_group(tmp_path):
    """A logger owns one foreground controller; both exit on a bounded input."""
    script = tmp_path / "run_evaluation_workflow.py"
    script.write_text("import sys\nsys.stdin.buffer.read(1)\n")
    profile = tmp_path / "profile.json"
    _write(profile, {"name": "fixture-original-quality", "hardware": {
        "selected_setups": ["fixture-deepx"], "setups_file": str(tmp_path / "registry.yaml")},
        "energy": {"enabled": False}, "native_producers": {"enabled": False},
        "run_profiles": [{"enabled": True, "quality_source_only": True,
            "hardware_setup_id": "fixture-deepx", "case_ids": ["b003"]}]})
    action = tmp_path / "action.json"
    workflow_argv = [sys.executable, "-B", str(script), "--profile", str(profile),
                     "--out", str(tmp_path / "runs"), "--profile-driven", "--json"]
    target = {"setup_id": "fixture-deepx", "host": "192.0.2.102", "user": "fixture"}
    scope = {"Native": False, "energy": False, "FPS": False,
             "model_id": "fixture", "case_ids": ["b003"]}
    _write(action, {"argv": workflow_argv, "profile": _descriptor(profile),
                    "target": target, "scope": scope})
    logger_script = tmp_path / "run_product_action.py"
    source = (
        "import json,pathlib,subprocess,sys\n"
        "a=json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "p=subprocess.Popen(a['argv'],stdin=subprocess.PIPE)\n"
        "print(json.dumps({'controller_pid':p.pid}),flush=True)\n"
        "sys.stdin.buffer.read(1)\n"
        "p.stdin.write(b'x');p.stdin.flush();p.wait(timeout=5)\n"
    )
    logger_script.write_text(source)
    logger = subprocess.Popen([sys.executable, "-B", str(logger_script), str(action)], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert logger.stdout is not None
    controller = json.loads(logger.stdout.readline())["controller_pid"]
    identities = {"logger": _identity(logger.pid), "controller": _identity(controller)}

    def finish():
        if logger.poll() is None:
            assert logger.stdin is not None
            logger.stdin.write("x")
            logger.stdin.flush()
        logger.wait(timeout=5)

    try:
        yield SimpleNamespace(logger=logger, controller_pid=controller,
                              identities=identities, finish=finish, action=action,
                              profile=profile, target=target, scope=scope)
    finally:
        finish()


@pytest.fixture
def group_case(tmp_path, monkeypatch, original_process_group, request):
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path",
                        lambda: tmp_path / "locks/workflow_platform_interlock.lock")
    run = tmp_path / "runs/original-generic-quality"
    session = "a" * 32
    controller = original_process_group.identities["controller"]
    owner = {"run_id": run.name, "session_id": session,
             "pid": controller["pid"], "ppid": controller["ppid"],
             "pgid": controller["pgrp"], "hostname": socket.gethostname()}
    options = {"only_model": "fixture", "energy_enabled": False, "native_producer_enabled": False}
    plan = {"hardware_targets": ["fixture-deepx"]}
    contract = run_control.build_resume_contract(
        profile_payload=json.loads(original_process_group.profile.read_text()),
        effective_execution_plan=plan, options=options)
    owner.update(profile_sha256=contract["profile_sha256"], plan_sha256=contract["plan_sha256"])
    run_lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    run_lock.acquire()
    run_lock.release(write_released_state=False)
    resource = "dut:192.0.2.102"
    physical_owner = {k: v for k, v in owner.items() if k != "session_id"}
    physical = run_control.EvaluationRunLock.for_resource(
        resource, owner={**physical_owner, "purpose": "generic"})
    physical.acquire()
    physical.commit_quarantine_fence({"reason": "previous_writer_terminated_uncleanly",
                                      "previous_owner": copy.deepcopy(physical.owner)})
    physical.release(write_released_state=False)
    manifest_path = run / "run_manifest.json"
    manifest = {"schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
                "run_id": run.name, "run_dir": str(run), "current_session_id": session,
                "status": "running", "technical_status": "running",
                "quality_decision": "not_evaluated",
                "execution_sessions": [{"session_id": session, "status": "running"}],
                "models": {"fixture": {"stages": {"capture": {"state": "running"}}}}}
    _write(manifest_path, manifest)
    result = run / "models/fixture/results/b003/composed_candidate.json"
    result.parent.mkdir(parents=True)
    result.write_bytes(b'{"record_count":2,"records":[{"image_id":"a"},{"image_id":"b"}]}\n')
    if getattr(request, "param", None) != "active":
        original_process_group.finish()
    after = copy.deepcopy(manifest)
    after.update(status="cancelled", technical_status="cancelled")
    after["execution_sessions"][0]["status"] = "cancelled"
    receipt = {"schema": "onnx-splitpoint/generic-orphan-recovery", "schema_version": 1,
               "recovery_id": "fixture-recovery", "run_id": run.name,
               "run_dir": str(run), "session_id": session,
               "operator": "fixture-operator", "recorded_process_identities": original_process_group.identities}
    (tmp_path / "recovery").mkdir()
    return SimpleNamespace(run=run, session=session, run_lock=run_lock,
        physical_locks={resource: physical}, resource=resource,
        manifest_path=manifest_path, manifest_before=manifest_path.read_bytes(),
        manifest_after=after, receipt_path=tmp_path / "recovery/receipt.json",
        receipt_data=receipt, result=result, result_bytes=result.read_bytes(),
        processes=original_process_group)


@contextmanager
def _held_group(case):
    locks = [case.run_lock, *case.physical_locks.values()]
    acquired = []
    try:
        for lock in locks:
            lock.acquire_for_recovery()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release(write_released_state=False)


def _commit(case):
    with _held_group(case):
        return run_control.commit_generic_orphan_recovery_group(
            run_lock=case.run_lock, physical_locks=case.physical_locks,
            receipt_path=case.receipt_path, receipt_data=case.receipt_data,
            manifest_path=case.manifest_path, manifest_before=case.manifest_before,
            manifest_after=case.manifest_after)


def _normal_lock(case, *, physical=False):
    if physical:
        return run_control.EvaluationRunLock.for_resource(case.resource, owner={"run_id": "next-run"})
    return run_control.EvaluationRunLock(out_root=case.run.parent, run_dir=case.run,
                                       owner={"run_id": case.run.name})


def test_group_commit_closes_interruption_preserves_results_and_missing_session(group_case):
    case = group_case
    original_physical = json.loads(next(iter(case.physical_locks.values())).lock_path.read_text())
    assert "session_id" not in original_physical
    _commit(case)
    manifest = json.loads(case.manifest_path.read_text())
    assert manifest["status"] == "cancelled"
    assert manifest["execution_sessions"][0]["status"] == "cancelled"
    assert manifest["models"] == json.loads(case.manifest_before)["models"]
    assert manifest["quality_decision"] == "not_evaluated"
    assert case.result.read_bytes() == case.result_bytes
    receipt = json.loads(case.receipt_path.read_text())
    assert receipt["schema"] == "onnx-splitpoint/generic-orphan-recovery"
    assert receipt.get("state", receipt.get("status")) == "committed"
    assert "session_id" not in json.loads(next(iter(case.physical_locks.values())).lock_path.read_text())
    for physical in (False, True):
        lock = _normal_lock(case, physical=physical)
        lock.acquire()
        lock.release()


def test_identical_group_receipt_is_idempotent_without_rewriting_originals(group_case):
    case = group_case
    _commit(case)
    paths = [case.receipt_path, case.manifest_path, case.run_lock.lock_path,
             *(lock.lock_path for lock in case.physical_locks.values()), case.result]
    before = {path: path.read_bytes() for path in paths}
    _commit(case)
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("field", ["run_id", "session_id", "recovery_id"])
def test_existing_receipt_cannot_be_reused_for_a_different_recovery(group_case, field):
    case = group_case
    _commit(case)
    before = {path: path.read_bytes() for path in
              (case.receipt_path, case.manifest_path, case.run_lock.lock_path)}
    case.receipt_data[field] = "foreign-identity"
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("fault", ["second_owner", "second_sidecar"])
def test_group_partial_commit_rolls_back_and_keeps_normal_admission_closed(group_case, monkeypatch, fault):
    case = group_case
    locks = [case.run_lock, *case.physical_locks.values()]
    owners_before = {lock.lock_path: lock.lock_path.read_bytes() for lock in locks}
    original_write = run_control.EvaluationRunLock.write_recovery_owner
    original_unlink = os.unlink
    touched = []

    def fail_write(lock, owner):
        if fault == "second_owner" and owner.get("state") == "released":
            touched.append(lock.lock_path)
            if len(touched) == 2:
                raise OSError("controlled second released-owner write failure")
        return original_write(lock, owner)

    def fail_unlink(path, *args, **kwargs):
        if fault == "second_sidecar" and os.fspath(path) in {str(lock.quarantine_path) for lock in locks}:
            touched.append(Path(path))
            if len(touched) == 2:
                raise OSError("controlled second quarantine unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(run_control.EvaluationRunLock, "write_recovery_owner", fail_write)
    monkeypatch.setattr(os, "unlink", fail_unlink)
    with pytest.raises(OSError, match="controlled second"):
        _commit(case)
    assert len(touched) >= 2
    assert case.manifest_path.read_bytes() == case.manifest_before
    assert case.result.read_bytes() == case.result_bytes
    assert all(path.read_bytes() == before for path, before in owners_before.items())
    for physical in (False, True):
        lock = _normal_lock(case, physical=physical)
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            lock.acquire()


@pytest.mark.parametrize("fault", ["committing", "blocked", "corrupt", "foreign"])
def test_receipt_guard_fails_closed_without_a_quarantine_sidecar(group_case, fault):
    case = group_case
    _commit(case)
    for lock in [case.run_lock, *case.physical_locks.values()]:
        assert not lock.quarantine_path.exists()
    receipt = json.loads(case.receipt_path.read_text())
    if fault == "corrupt":
        case.receipt_path.write_text("{incomplete receipt")
    else:
        if fault == "foreign":
            receipt["session_id"] = "b" * 32
        else:
            receipt["state"] = fault
            if "status" in receipt:
                receipt["status"] = fault
        _write(case.receipt_path, receipt)
    for physical in (False, True):
        lock = _normal_lock(case, physical=physical)
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            lock.acquire()
    assert case.result.read_bytes() == case.result_bytes


def test_crash_after_first_fence_removal_cannot_admit_either_group_member(group_case, monkeypatch):
    case = group_case
    paths = {lock.quarantine_path for lock in [case.run_lock, *case.physical_locks.values()]}
    original_unlink = Path.unlink

    def crash_after_unlink(path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if path in paths:
            os._exit(77)
        return result

    # Only the temporary child dies. The ordinary acquire path in this
    # surviving parent must honor a durable incomplete commit by itself.
    child = os.fork()
    if child == 0:
        try:
            monkeypatch.setattr(Path, "unlink", crash_after_unlink)
            _commit(case)
        except BaseException:
            os._exit(78)
        os._exit(79)
    waited, status = os.waitpid(child, 0)
    assert waited == child and os.waitstatus_to_exitcode(status) == 77
    receipt = json.loads(case.receipt_path.read_text())
    assert receipt["state"] == "committing"
    assert any(not path.exists() for path in paths)
    for physical in (False, True):
        lock = _normal_lock(case, physical=physical)
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            lock.acquire()
    assert case.result.read_bytes() == case.result_bytes


def test_failed_fence_restoration_keeps_receipt_marker_and_admission_closed(group_case, monkeypatch):
    case = group_case
    locks = [case.run_lock, *case.physical_locks.values()]
    original_unlink = Path.unlink
    original_write = run_control._generic_orphan_atomic_write
    removed = []

    def fail_second_unlink(path, *args, **kwargs):
        if path in {lock.quarantine_path for lock in locks}:
            removed.append(path)
            if len(removed) == 2:
                raise OSError("controlled second unlink failure")
        return original_unlink(path, *args, **kwargs)

    def fail_restore(path, data):
        if (path == case.run_lock.quarantine_path
                and b"generic_orphan_recovery_blocked" in data):
            raise OSError("controlled persistent fence restoration failure")
        return original_write(path, data)

    monkeypatch.setattr(Path, "unlink", fail_second_unlink)
    monkeypatch.setattr(run_control, "_generic_orphan_atomic_write", fail_restore)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="rollback"):
        _commit(case)
    assert not case.run_lock.quarantine_path.exists()
    owner = json.loads(case.run_lock.lock_path.read_text())
    assert owner["generic_orphan_recovery"]["receipt_path"] == str(case.receipt_path)
    assert json.loads(case.receipt_path.read_text())["state"] == "blocked"
    assert case.manifest_path.read_bytes() == case.manifest_before
    for physical in (False, True):
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            _normal_lock(case, physical=physical).acquire()
    assert case.result.read_bytes() == case.result_bytes


def test_truncated_released_owner_still_fails_closed_without_sidecar(group_case):
    case = group_case
    _commit(case)
    assert not case.run_lock.quarantine_path.exists()
    case.run_lock.lock_path.write_bytes(b'{"generic_orphan_recovery":')
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _normal_lock(case).acquire()


def _refresh_evidence_descriptors(case):
    evidence = json.loads(case.evidence_path.read_text())
    for name in ("initial_lineage", "handoff", "interruption"):
        evidence[name] = _descriptor(Path(evidence[name]["path"]))
    _write(case.evidence_path, evidence)
    auth = json.loads(case.authorization_path.read_text())
    auth["evidence"] = _descriptor(case.evidence_path)
    _write(case.authorization_path, auth)


@pytest.fixture
def orphan_case(group_case, monkeypatch):
    """Authentic fixture launch records plus a remote observation double."""
    from onnx_splitpoint_tool.remote import resource_recovery_observation

    case = group_case
    root = case.run.parent.parent
    processes = case.processes
    workflow, logger = (processes.identities[name] for name in ("controller", "logger"))
    started = root / "action.started.json"
    action = json.loads(processes.action.read_text())
    _write(started, {"action_spec": str(processes.action), "argv": action["argv"]})
    lineage_path = root / "action.initial-owner.json"
    lineage = {"schema": "generic420-normal-quality-group-initial-local-lineage-v1",
        "initial_lineage_status": "bound", "observation_error": None,
        "action": _descriptor(processes.action), "profile": _descriptor(processes.profile),
        "started": _descriptor(started), "workflow_owner": workflow, "logger_owner": logger,
        "target": processes.target, "scope": processes.scope}
    _write(lineage_path, lineage)
    registry_path = root / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({"schema_version": 1, "hardware_setups": [{
        "id": "fixture-deepx", "host": {"address": "192.0.2.102", "user": "fixture", "port": 22}}]}))
    profile = json.loads(processes.profile.read_text())
    options = {"only_model": "fixture", "energy_enabled": False, "native_producer_enabled": False}
    effective_plan = {"hardware_targets": ["fixture-deepx"]}
    contract = run_control.build_resume_contract(profile_payload=profile,
        effective_execution_plan=effective_plan, options=options)
    (case.run / "profile.yaml").write_text(yaml.safe_dump(profile))
    (case.run / "profile_source.yaml").write_text(yaml.safe_dump(profile))
    _write(case.run / "effective_execution_plan.json", effective_plan)
    manifest = json.loads(case.manifest_path.read_text())
    manifest.update(options=options, resume_contract=contract, profile_path=str(processes.profile),
        hardware_targets=[{"setup_id": "fixture-deepx", "host": "192.0.2.102"}])
    manifest["execution_sessions"][0].update(
        **{key: contract[key] for key in ("profile_sha256", "plan_sha256", "resume_contract_sha256")},
        start_origin={"pid": workflow["pid"]})
    _write(case.manifest_path, manifest)
    _write(case.run / "jobs/workflow_control.json", {"schema": "onnx-splitpoint/workflow-control", "run_id": case.run.name,
        "session_id": case.session, "state": "attached", "cancel_requested": False})
    _write(case.run / "models/fixture/benchmark_results/remote_benchmark_dispatch_fixture-deepx.json", {
        "schema": "onnx-splitpoint/remote-benchmark-dispatch", "hardware_target_id": "fixture-deepx",
        "model_id": "fixture", "host": {"host": "192.0.2.102", "user": "fixture", "port": 22},
        "args": {"quality_evidence_eval_id": case.run.name, "quality_evidence_setup_id": "fixture-deepx"}})
    (case.run / "jobs/remote_process_leases" / case.session).mkdir(parents=True)
    handoff_path = root / "active_actions.json"
    histories = [{**row, "start_ticks": row["process_start_ticks"]}
                 for row in (workflow, logger)]
    original = {"normal_run_id": case.run.name, "normal_run_dir": str(case.run),
        "session_id": case.session, "initial_lineage": _descriptor(lineage_path),
        **{name: lineage[name] for name in ("action", "profile", "started")},
        "workflow_pid": workflow["pid"], "workflow_process_start_ticks": workflow["process_start_ticks"],
        "logger_pid": logger["pid"], "logger_process_start_ticks": logger["process_start_ticks"],
        "target": processes.target, "selected_scope": {"model_id": "fixture", "case_ids": ["b003"]},
        "resume_contract": contract, "current_local_processes": histories, "lease_records": []}
    _write(handoff_path, {"actions": [original]})
    interruption_path = root / "LIFECYCLE_INTERRUPTION.json"
    _write(interruption_path, {"schema": "generic420-continuation-v3-lifecycle-interruption-v1",
        "source_handoff": str(handoff_path), "workflows": [{"normal_run_dir": str(case.run),
            "session_id": case.session, "setup_id": "fixture-deepx", "physical_resource": case.resource,
            "original_action_path": str(processes.action), "phase": {},
            "last_genuine_local_owner_snapshot": {"pid": workflow["pid"], "ppid": workflow["ppid"],
                "pgrp": workflow["pgrp"], "start_ticks": workflow["process_start_ticks"]}}],
        "local_orphan_scope_latest": {}})
    evidence_path = root / "recovery/input.json"
    identity = {"run_id": case.run.name, "run_dir": str(case.run),
                "session_id": case.session, "setup_id": "fixture-deepx"}
    _write(evidence_path, {"schema": "onnx-splitpoint/generic-orphan-recovery-input", "schema_version": 1,
        **identity, "initial_lineage": _descriptor(lineage_path), "handoff": _descriptor(handoff_path),
        "interruption": _descriptor(interruption_path)})
    authorization_path = root / "recovery/authorization.json"
    _write(authorization_path, {"schema": "onnx-splitpoint/generic-orphan-recovery-authorization",
        "schema_version": 1, **identity, "authorized": True, "action": "recover_generic_orphan_run",
        "operator": "fixture-operator", "source": "explicit test authorization",
        "authorized_at": datetime.now(timezone.utc).isoformat(), "evidence": _descriptor(evidence_path)})
    remote = {"ok": True, "read_only": True, "status": "no_known_competing_work_seen",
        "recorded_owners": [], "processes": [], "live_product_leases": [], "visibility_errors": [],
        "cleanup_proven": False, "exclusive_start_permission": False}

    class Transport:
        def __init__(self, host, *args, **kwargs):
            self.host = host

        def run_read_only(self, *args, **kwargs):
            return 0, "RESOURCE_OWNERS=" + json.dumps(remote)

    monkeypatch.setattr(resource_recovery_observation, "SSHTransport", Transport)
    case.evidence_path, case.authorization_path = evidence_path, authorization_path
    case.lineage_path, case.handoff_path = lineage_path, handoff_path
    case.interruption_path, case.registry_path = interruption_path, registry_path
    case.remote_observation = remote
    case.kwargs = dict(run_dir=case.run, session_id=case.session, setup_id="fixture-deepx",
        evidence_path=evidence_path, authorization_path=authorization_path,
        evidence_output=case.receipt_path, registry_path=registry_path)
    case.preserved = {path: path.read_bytes() for path in (processes.action, processes.profile,
        started, lineage_path, handoff_path, interruption_path, registry_path, case.result)}
    case.original_manifest = case.manifest_path.read_bytes()
    return case


def test_dead_original_group_recovery_has_new_receipt_and_preserves_scientific_bytes(orphan_case):
    case = orphan_case
    result = resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert result["status"] == "committed"
    assert all(path.read_bytes() == before for path, before in case.preserved.items())
    manifest = json.loads(case.manifest_path.read_text())
    assert manifest["status"] == "cancelled"
    assert manifest["execution_sessions"][0]["status"] == "cancelled"
    assert manifest["quality_decision"] == "not_evaluated"
    assert manifest["models"] == json.loads(case.original_manifest)["models"]
    receipt = json.loads(case.receipt_path.read_text())
    assert receipt["state"] == "committed"
    assert receipt["session_id"] == case.session
    assert receipt["recovery_id"]
    assert "session_id" not in json.loads(next(iter(case.physical_locks.values())).lock_path.read_text())


def test_exact_recovery_receipt_replay_does_not_repeat_cleanup_or_rewrite_bytes(orphan_case, monkeypatch):
    case = orphan_case
    resource_recovery.recover_generic_orphan_run(**case.kwargs)
    paths = [case.receipt_path, case.manifest_path, case.run_lock.lock_path,
             *(lock.lock_path for lock in case.physical_locks.values()), *case.preserved]
    before = {path: path.read_bytes() for path in paths}

    def no_new_observation(*args, **kwargs):
        pytest.fail("exact committed receipt replay must not repeat the original cleanup")

    monkeypatch.setattr(resource_recovery, "_generic_orphan_local_observation", no_new_observation)
    result = resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert result["status"] == "already_committed"
    assert all(path.read_bytes() == value for path, value in before.items())


def test_missing_authorization_evidence_descriptor_cannot_recover(orphan_case):
    case = orphan_case
    authorization = json.loads(case.authorization_path.read_text())
    authorization.pop("evidence")
    _write(case.authorization_path, authorization)
    owners = {lock.lock_path: lock.lock_path.read_bytes() for lock in
              [case.run_lock, *case.physical_locks.values()]}
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert all(path.read_bytes() == before for path, before in owners.items())
    assert not case.receipt_path.exists()


def test_committed_receipt_rejects_changed_resolved_profile(orphan_case):
    case = orphan_case
    resource_recovery.recover_generic_orphan_run(**case.kwargs)
    protected = [case.receipt_path, case.manifest_path, case.run_lock.lock_path,
                 *(lock.lock_path for lock in case.physical_locks.values()), case.result]
    before = {path: path.read_bytes() for path in protected}
    profile_path = case.run / "profile.yaml"
    profile = yaml.safe_load(profile_path.read_text())
    profile["name"] = "changed-after-recovery"
    profile_path.write_text(yaml.safe_dump(profile))
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("group_case", ["active"], indirect=True)
def test_live_original_controller_cannot_recover(orphan_case):
    case = orphan_case
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.result.read_bytes() == case.result_bytes
    assert not case.receipt_path.exists()


@pytest.mark.parametrize("fault", ["request_session", "input_run", "physical_owner", "authorization"])
def test_foreign_scope_or_missing_authorization_cannot_recover(orphan_case, fault):
    case = orphan_case
    if fault == "request_session":
        case.kwargs["session_id"] = "b" * 32
    elif fault == "input_run":
        value = json.loads(case.evidence_path.read_text())
        value["run_id"] = "foreign-run"
        _write(case.evidence_path, value)
        _refresh_evidence_descriptors(case)
    elif fault == "physical_owner":
        path = next(iter(case.physical_locks.values())).lock_path
        value = json.loads(path.read_text())
        value["run_id"] = "foreign-run"
        _write(path, value)
    else:
        value = json.loads(case.authorization_path.read_text())
        value["authorized"] = False
        _write(case.authorization_path, value)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert not case.receipt_path.exists()


def test_live_remote_lease_does_not_release_original_group(orphan_case):
    case = orphan_case
    case.remote_observation.update(ok=False, status="busy",
        live_product_leases=[{"pid": 123, "start_time_ticks": "456"}])
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert not case.receipt_path.exists()


def test_detached_descendant_with_original_run_argument_blocks_recovery(orphan_case):
    case = orphan_case
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)", str(case.run)],
                             stdin=subprocess.PIPE, start_new_session=True)
    try:
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            resource_recovery.recover_generic_orphan_run(**case.kwargs)
        assert case.manifest_path.read_bytes() == case.original_manifest
        assert not case.receipt_path.exists()
    finally:
        assert child.stdin is not None
        child.stdin.write(b"x")
        child.stdin.flush()
        child.wait(timeout=5)
