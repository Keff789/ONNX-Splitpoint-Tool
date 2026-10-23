"""Bounded final-cleanup recovery with real local predecessor identities.

All locks and archival inputs belong to temporary test runs. Remote observation
and unrelated shared-host work are isolated; no test connects to a DUT.
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone
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

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.remote import resource_recovery, resource_recovery_observation
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot


STARTED = "2026-10-22T19:20:49+00:00"
FINISHED = "2026-10-22T22:48:50+00:00"
RESULT_AT = "2026-10-22T22:49:00+00:00"


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _descriptor(path: Path) -> dict:
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _identity(pid: int) -> dict:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
            "session_id": int(fields[3]), "start_ticks": fields[19]}


def _refresh(case) -> None:
    """Authorize changed fixture bytes, letting semantic validators see them."""
    evidence = json.loads(case.evidence_path.read_text())
    evidence["original_files"] = [_descriptor(Path(row["path"])) for row in evidence["original_files"]]
    for name in ("action", "started", "result", "controller_observation", "state_binding", "registry",
                 "remote_boot_observation"):
        if name in evidence:
            evidence[name] = _descriptor(Path(evidence[name]["path"]))
    evidence["released_resources"] = [_descriptor(Path(row["path"]))
                                      for row in evidence["released_resources"]]
    _write(case.evidence_path, evidence)
    authorization = json.loads(case.authorization_path.read_text())
    authorization["evidence"] = _descriptor(case.evidence_path)
    _write(case.authorization_path, authorization)


@pytest.fixture
def final_recovery_case(tmp_path, monkeypatch, request):
    interlock = tmp_path / "locks/workflow_platform_interlock.lock"
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: interlock)
    run = tmp_path / "runs/original-generic-final-cleanup"
    run.mkdir(parents=True)
    session, setup_id = "c" * 32, "fixture-h8"
    identity = {"run_id": run.name, "run_dir": str(run), "session_id": session, "setup_id": setup_id}
    target = {"id": setup_id, "remote": {"host": "192.0.2.104", "user": "fixture", "port": 22}}
    profile_path = tmp_path / "original-profile.json"
    profile = {"name": "original-generic-quality", "hardware": {"selected_setups": [setup_id]},
        "model_suite": {"primary": [{"id": "fixture", "enabled": True}]},
        "energy": {"enabled": False}, "native_producers": {"enabled": False},
        "run_profiles": [{"id": "fixture-h8-quality", "enabled": True, "quality_source_only": True,
                          "hardware_setup_id": setup_id, "case_ids": ["b003"]}]}
    _write(profile_path, profile)
    script = tmp_path / "run_evaluation_workflow.py"
    script.write_text("import sys\nsys.stdin.buffer.read(1)\n")
    argv = [sys.executable, "-B", str(script), "--profile", str(profile_path), "--out", str(run.parent),
            "--profile-driven", "--json"]
    controller = subprocess.Popen(argv, stdin=subprocess.PIPE, start_new_session=True)
    original_identity = _identity(controller.pid)

    def finish():
        if controller.poll() is None:
            assert controller.stdin is not None
            controller.stdin.write(b"x")
            controller.stdin.flush()
        controller.wait(timeout=5)

    request.addfinalizer(finish)
    if getattr(request, "param", None) != "active":
        finish()
    options = {"only_model": "fixture", "energy_enabled": False, "native_producer_enabled": False}
    plan = build_effective_execution_plan(profile)
    contract = run_control.build_resume_contract(profile_payload=profile,
        effective_execution_plan=plan, options=options)
    snapshot = build_profile_start_snapshot(profile_request=str(profile_path), source_profile=profile,
        resolved_profile=profile, profile_id="fixture", profile_path=str(profile_path), profile_source="fixture",
        runtime_bindings={"hardware_targets": [target]})
    owner = {"run_id": run.name, "session_id": session, "pid": original_identity["pid"],
             "ppid": original_identity["ppid"], "pgid": original_identity["pgid"],
             "hostname": socket.gethostname(),
             **{key: contract[key] for key in ("profile_sha256", "plan_sha256")}}
    marker = {"schema": "onnx-splitpoint/p01-cleanup-quarantine", "schema_version": 1,
        "run_id": run.name, "session_id": session, "created_at": FINISHED,
        "phase": "final_cleanup", "local_process_quiescence_proven": True,
        "remote_process_quiescence_proven": False,
        "details": {"reason": "final_process_quiescence_unresolved", "errors": [],
                    "management_service_shutdown": {"run_id": run.name, "session_id": session,
                        "owner_pid": controller.pid, "finished": True, "admission_closed": True, "errors": [],
                        "phase": "completed", "shutdown_thread": {"alive": False},
                        "management_reference": {"threads": [], "futures": {}},
                        "central_quality_coordinator": {"threads": [], "futures": {}},
                        "central_quality_service": {"finished": True, "admission_closed": True,
                            "dispatcher_alive": False, "manager_alive": False, "queue_size": 0,
                            "workers": [], "inflight_keys": [], "scratch_cleanup_errors": []}},
                    "remote_reports": [{"operation_id": "ssh-stream-original"}],
                    "remote_remaining": 1, "late_remote_remaining": 1}}
    run_lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    run_lock.acquire()
    run_lock.commit_quarantine_fence({**marker, "reason": marker["details"]["reason"]})
    run_lock.release(write_released_state=False)
    resource = "dut:192.0.2.104"
    physical = run_control.EvaluationRunLock.for_resource(resource, owner={
        **{key: value for key, value in owner.items() if key != "session_id"}, "purpose": "generic"})
    physical.acquire()
    physical.commit_quarantine_fence({"reason": "generic_remote_cleanup_requires_reconciliation"})
    physical.release(write_released_state=False)
    manifest_path = run / "run_manifest.json"
    manifest = {"schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": run.name, "run_dir": str(run), "current_session_id": session,
        "status": "partial", "technical_status": "partial", "finished_at": FINISHED,
        "quality_decision": "FAIL", "options": options, "resume_contract": contract,
        "profile_start_snapshot": snapshot, "hardware_targets": [target],
        "execution_sessions": [{"session_id": session, "status": "partial", "finished_at": FINISHED,
            "start_origin": {"pid": controller.pid}, **{key: contract[key] for key in
                ("profile_sha256", "plan_sha256", "resume_contract_sha256")}}],
        "models": {"fixture": {"stages": {"quality": {"status": "completed", "quality": "FAIL"}}}}}
    _write(manifest_path, manifest)
    marker_path = run / "jobs/p01_unresolved_cleanup_quarantine.json"
    _write(marker_path, marker)
    _write(run / "jobs/workflow_control.json", {"schema": "onnx-splitpoint/workflow-control",
        "run_id": run.name, "session_id": session, "state": "finished"})
    for name in ("profile.yaml", "profile_source.yaml"):
        (run / name).write_text(yaml.safe_dump(profile))
    _write(run / "effective_execution_plan.json", plan)
    _write(run / "profile_start_snapshot.json", snapshot)
    dispatch_path = run / "models/fixture/benchmark_results/remote_benchmark_dispatch_fixture-h8.json"
    _write(dispatch_path, {"schema": "onnx-splitpoint/remote-benchmark-dispatch", "model_id": "fixture",
        "hardware_target_id": setup_id, "host": target["remote"], "created_at": STARTED,
        "args": {"quality_evidence_eval_id": run.name, "quality_evidence_setup_id": setup_id}})
    scope = RemoteProcessLeaseScope(run.name, session)
    journal_path = run / "jobs/remote_process_leases" / session / "ssh-stream-original.remote-lease.json"
    _write(journal_path, {"schema": "onnx-splitpoint/remote-process-lease-journal", "schema_version": 1,
        "run_sha256": scope.run_sha256, "session_sha256": scope.session_sha256,
        "operation_id": "ssh-stream-original", "token": "original-guardian-token", "remote_lease_root": "",
        "control_argv_prefix": ["ssh", "-p", "22", "fixture@192.0.2.104"]})
    action_path, started_path = tmp_path / "action.json", tmp_path / "action.started.json"
    result_path, observed_path = tmp_path / "action.result.json", tmp_path / "controller-observation.json"
    _write(action_path, {"argv": argv, "profile": _descriptor(profile_path),
        "scope": {"model_id": "fixture", "Native": False, "energy": False, "FPS": False},
        "target": {"setups": [{"setup_id": setup_id, "host": "192.0.2.104", "user": "fixture", "port": 22}]}})
    _write(started_path, {"action_spec": str(action_path), "argv": argv, "started_at": STARTED})
    _write(result_path, {"action_spec": str(action_path), "argv": argv, "started_at": STARTED,
                         "finished_at": RESULT_AT, "returncode": 1})
    _write(observed_path, {**identity, "controller_identity_matches": True,
        "expected_controller_start_ticks": original_identity["start_ticks"], "controller": original_identity})
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({"schema_version": 1, "hardware_setups": [{
        "id": setup_id, "host": {"address": "192.0.2.104", "user": "fixture", "port": 22}}]}))
    binding_path = tmp_path / "state-binding.json"
    state_paths = [registry_path, interlock, run_lock.lock_path, run_lock.quarantine_path,
                   physical.lock_path, physical.quarantine_path]
    _write(binding_path, {"uid": os.getuid(), "expanded_home": str(Path.home()),
        "registry_path": str(registry_path), "platform_interlock_path": str(interlock),
        "physical_resource": resource, "normal_run_lock_path": str(run_lock.lock_path),
        "normal_h8_resource_lock_path": str(physical.lock_path), "checkout_relative_resource_state": False,
        "paths": [{"path": str(path), "device": path.stat().st_dev, "inode": path.stat().st_ino,
                   "is_symlink": False} for path in state_paths]})
    evidence_path, authorization_path = tmp_path / "recovery/input.json", tmp_path / "recovery/authorization.json"
    original_files = [run_lock.lock_path, run_lock.quarantine_path, physical.lock_path, physical.quarantine_path,
        manifest_path, marker_path, run / "jobs/workflow_control.json", run / "profile.yaml",
        run / "profile_source.yaml", run / "effective_execution_plan.json", run / "profile_start_snapshot.json",
        dispatch_path, journal_path]
    _write(evidence_path, {"schema": "onnx-splitpoint/generic-final-cleanup-recovery-input", "schema_version": 1,
        **identity, "original_files": [_descriptor(path) for path in original_files], "released_resources": [],
        **{name: _descriptor(path) for name, path in (("action", action_path), ("started", started_path),
            ("result", result_path), ("controller_observation", observed_path),
            ("state_binding", binding_path), ("registry", registry_path))}})
    checkout = Path(resource_recovery.__file__).resolve().parents[2]
    _write(authorization_path, {"schema": "onnx-splitpoint/generic-final-cleanup-recovery-authorization",
        "schema_version": 1, **identity, "action": "recover_generic_final_cleanup", "authorized": True,
        "operator": "fixture-operator", "source": "explicit fixture authorization",
        "authorized_at": datetime.now(timezone.utc).isoformat(), "source_checkout": str(checkout),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip(),
        "evidence": _descriptor(evidence_path)})
    remote = {"read_only": True, "ownership": {"ok": True, "read_only": True,
        "status": "no_known_competing_work_seen", "processes": [], "recorded_owners": [],
        "live_product_leases": [], "visibility_errors": []},
        "operations": [{"operation_id": "ssh-stream-original", "lease_present": False, "guardian_drained": True}]}
    calls = []

    def observe(setup, *, operations, diagnostics_dir):
        calls.append((setup, copy.deepcopy(operations), diagnostics_dir))
        return copy.deepcopy(remote)

    monkeypatch.setattr(resource_recovery_observation, "observe_generic_final_cleanup_remote", observe)
    raw_observe = resource_recovery_observation._observe_processes

    def isolated_raw_observation(owners):
        # This shared testhost has real unrelated workflows. Preserve the real
        # predecessor and visibility probe; isolate only unrelated raw work.
        value = raw_observe(owners)
        value["processes"] = [row for row in value["processes"] if row["pid"] == controller.pid]
        value["live_product_leases"] = []
        busy = bool(value["processes"] or any(row["state"] not in {"absent", "pid_reused", "zombie"}
                                              for row in value["recorded_owners"]))
        value.update(ok=not busy and not value["visibility_errors"],
                     status="busy" if busy else "no_known_competing_work_seen")
        return value

    monkeypatch.setattr(resource_recovery_observation, "_observe_processes", isolated_raw_observation)
    scientific_path = run / "models/fixture/quality.json"
    _write(scientific_path, {"quality": "FAIL", "records": [{"image_id": "a", "score": 0.812}]})
    receipt_path = tmp_path / "recovery/receipt.json"
    return SimpleNamespace(run=run, identity=identity, run_lock=run_lock, physical=physical, resource=resource,
        controller=controller, controller_identity=original_identity, finish=finish,
        evidence_path=evidence_path, authorization_path=authorization_path, registry_path=registry_path,
        manifest_path=manifest_path, marker_path=marker_path, result_path=result_path, observed_path=observed_path,
        journal_path=journal_path, receipt_path=receipt_path, scientific_path=scientific_path,
        original_files=original_files, remote=remote, remote_calls=calls,
        kwargs=dict(run_dir=run, session_id=session, setup_id=setup_id, evidence_path=evidence_path,
                    authorization_path=authorization_path, registry_path=registry_path, evidence_output=receipt_path))


def _assert_blocked(case, *, match=None):
    paths = [*case.original_files, case.scientific_path]
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match=match):
        resource_recovery.recover_generic_final_cleanup(**case.kwargs)
    assert not case.receipt_path.exists()
    assert all(path.read_bytes() == raw for path, raw in before.items())
    for lock in (case.run_lock, case.physical):
        assert lock.quarantine_path.exists()
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            lock.acquire()


def _add_original_released_setup(case):
    """Extend the archived run; never acquire today's other-DUT lock."""
    action_path = Path(json.loads(case.evidence_path.read_text())["action"]["path"])
    action = json.loads(action_path.read_text())
    profile_path = Path(action["profile"]["path"])
    profile = json.loads(profile_path.read_text())
    other_setup, other_resource = "fixture-deepx", "dut:fixture-deepx-canonical"
    other_target = {"id": other_setup, "physical_host_id": "fixture-deepx-canonical",
                    "remote": {"host": "192.0.2.102", "user": "fixture", "port": 22}}
    profile["hardware"]["selected_setups"].append(other_setup)
    profile["run_profiles"].append({**profile["run_profiles"][0],
        "id": "fixture-deepx-quality", "hardware_setup_id": other_setup})
    _write(profile_path, profile)
    for name in ("profile.yaml", "profile_source.yaml"):
        (case.run / name).write_text(yaml.safe_dump(profile))
    manifest = json.loads(case.manifest_path.read_text())
    targets = [*manifest["hardware_targets"], other_target]
    plan = build_effective_execution_plan(profile)
    contract = run_control.build_resume_contract(profile_payload=profile,
        effective_execution_plan=plan, options=manifest["options"])
    archived = json.loads((case.run / "profile_start_snapshot.json").read_text())
    snapshot = build_profile_start_snapshot(
        **{key: archived[key] for key in ("profile_request", "profile_id", "profile_path", "profile_source")},
        source_profile=profile, resolved_profile=profile, runtime_bindings={"hardware_targets": targets})
    _write(case.run / "effective_execution_plan.json", plan)
    _write(case.run / "profile_start_snapshot.json", snapshot)
    manifest.update(resume_contract=contract, hardware_targets=targets, profile_start_snapshot=snapshot)
    manifest["execution_sessions"][0].update({key: contract[key] for key in
        ("profile_sha256", "plan_sha256", "resume_contract_sha256")})
    _write(case.manifest_path, manifest)
    for lock in (case.run_lock, case.physical):
        owner = json.loads(lock.lock_path.read_text())
        owner.update({key: contract[key] for key in ("profile_sha256", "plan_sha256")})
        _write(lock.lock_path, owner)
    action["profile"] = _descriptor(profile_path)
    action["target"]["setups"].append({"setup_id": other_setup, **other_target["remote"]})
    _write(action_path, action)
    registry = yaml.safe_load(case.registry_path.read_text())
    registry["hardware_setups"].append({"id": other_setup, "physical_host_id": "fixture-deepx-canonical",
        "host": {"address": "192.0.2.102", "user": "fixture", "port": 22}})
    case.registry_path.write_text(yaml.safe_dump(registry))
    other_lock = run_control.EvaluationRunLock.for_resource(other_resource, owner={})
    assert not other_lock.lock_path.exists()
    released_owner = {**json.loads(case.physical.lock_path.read_text()), "physical_resource": other_resource,
        "run_dir": str(other_lock.run_dir), "state": "released", "acquired_at": STARTED,
        "released_at": "2026-10-22T22:48:49+00:00"}
    release_path = case.evidence_path.parent / "original-deepx-release.json"
    _write(release_path, {"external_session_binding": case.identity["session_id"],
        "run_id": case.run.name, "lock_path": str(other_lock.lock_path),
        "quarantine_exists": False, "quarantine": None, "owner": released_owner})
    evidence = json.loads(case.evidence_path.read_text())
    evidence["released_resources"] = [_descriptor(release_path)]
    _write(case.evidence_path, evidence)
    _refresh(case)
    case.other_resource, case.other_lock_path, case.release_path = other_resource, other_lock.lock_path, release_path
    return case


@pytest.fixture
def multi_final_recovery_case(final_recovery_case):
    return _add_original_released_setup(final_recovery_case)


def test_multi_setup_final_recovery_uses_archived_release_and_holds_only_remaining_group(
        multi_final_recovery_case, monkeypatch):
    case = multi_final_recovery_case
    released_before = case.release_path.read_bytes()
    manifest_before = json.loads(case.manifest_path.read_text())
    acquire = run_control.EvaluationRunLock.acquire_for_recovery
    acquired = []

    def track(lock):
        acquired.append(lock.lock_path)
        return acquire(lock)

    monkeypatch.setattr(run_control.EvaluationRunLock, "acquire_for_recovery", track)
    outcome = resource_recovery.recover_generic_final_cleanup(**case.kwargs)
    assert outcome["status"] == "committed" and outcome["resources"] == [case.resource]
    assert acquired == [case.run_lock.lock_path, case.physical.lock_path]
    assert not case.other_lock_path.exists()
    assert case.release_path.read_bytes() == released_before
    receipt = json.loads(case.receipt_path.read_text())
    assert _descriptor(case.release_path) in receipt["source_evidence"]
    assert {row["physical_resource"] for row in receipt["group"]} == {None, case.resource}
    manifest = json.loads(case.manifest_path.read_text())
    manifest.pop("generic_orphan_recovery")
    assert manifest == manifest_before


@pytest.mark.parametrize("fault", ["missing", "foreign_session"])
def test_multi_setup_final_recovery_requires_bound_original_other_dut_release(multi_final_recovery_case, fault):
    case = multi_final_recovery_case
    if fault == "missing":
        evidence = json.loads(case.evidence_path.read_text())
        evidence["released_resources"] = []
        _write(case.evidence_path, evidence)
    else:
        release = json.loads(case.release_path.read_text())
        release["external_session_binding"] = "d" * 32
        _write(case.release_path, release)
    _refresh(case)
    release_before = case.release_path.read_bytes()
    _assert_blocked(case, match="other-DUT release|multi-setup release inventory")
    assert not case.other_lock_path.exists()
    assert case.release_path.read_bytes() == release_before
    assert not case.remote_calls


@pytest.mark.parametrize("fault", [None, "wrong_registry", "missing_registry_target"])
def test_final_recovery_rechecks_independent_controller_canonical_registry_scope(
        multi_final_recovery_case, monkeypatch, fault):
    from onnx_splitpoint_tool.workflow import generic_completed_task

    case = multi_final_recovery_case
    authority = {"run_id": "current-independent-deepx", "session_id": "d" * 32,
        "resources": ["dut:192.0.2.102", case.other_resource],
        "ssh_targets": [{"user_host": "fixture@192.0.2.102", "port": "22"}]}
    calls = []

    def scoped(targets, observed):
        calls.append((targets, observed))
        return {"ok": True, "selected_resources": [case.resource], "normal_quality_owners": [authority]}

    monkeypatch.setattr(generic_completed_task, "_scoped_controller_owners", scoped)
    if fault:
        if fault == "wrong_registry":
            registry = yaml.safe_load(case.registry_path.read_text())
            next(row for row in registry["hardware_setups"] if row["id"] == "fixture-deepx")[
                "physical_host_id"] = "foreign-canonical-chassis"
            case.registry_path.write_text(yaml.safe_dump(registry))
            _refresh(case)
            reason = "independent controller physical registry scope differs"
        else:
            authority["ssh_targets"][0]["user_host"] = "fixture@192.0.2.105"
            reason = "independent controller registry target is unbound"
        _assert_blocked(case, match=reason)
        assert not case.remote_calls
        assert len(calls) == 1
    else:
        assert resource_recovery.recover_generic_final_cleanup(**case.kwargs)["status"] == "committed"
        assert len(calls) == 2
        receipt = json.loads(case.receipt_path.read_text())
        for name in ("local_observation_before", "local_observation_after"):
            assert receipt[name]["recorded_scope"]["scoped_controller_owners"]["normal_quality_owners"] == [authority]
    assert not case.other_lock_path.exists()


def test_final_binding_rejects_foreign_held_platform_interlock_inode(final_recovery_case):
    case = final_recovery_case
    evidence = json.loads(case.evidence_path.read_text())
    binding = json.loads(Path(evidence["state_binding"]["path"]).read_text())
    case.run_lock.acquire_for_recovery()
    try:
        case.physical.acquire_for_recovery()
        try:
            original = case.run_lock._platform_interlock_fh
            with (case.evidence_path.parent / "foreign-platform.lock").open("w+b") as foreign:
                case.run_lock._platform_interlock_fh = foreign
                try:
                    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError,
                                       match="held platform interlock"):
                        resource_recovery._generic_final_state_binding(binding,
                            registry_file=case.registry_path, run_lock=case.run_lock,
                            physical=case.physical, resource=case.resource)
                finally:
                    case.run_lock._platform_interlock_fh = original
        finally:
            case.physical.release(write_released_state=False)
    finally:
        case.run_lock.release(write_released_state=False)


def test_final_recovery_preserves_partial_history_and_exact_original_evidence(final_recovery_case):
    case = final_recovery_case
    before = {path: path.read_bytes() for path in case.original_files}
    scientific = case.scientific_path.read_bytes()
    outcome = resource_recovery.recover_generic_final_cleanup(**case.kwargs)
    assert outcome["status"] == "committed"
    assert outcome["measurements_started"] == 0
    assert outcome["resources"] == [case.resource]
    manifest = json.loads(case.manifest_path.read_text())
    reference = manifest.pop("generic_orphan_recovery")
    assert manifest == json.loads(before[case.manifest_path])
    assert reference["receipt_path"] == str(case.receipt_path)
    assert manifest["status"] == manifest["execution_sessions"][0]["status"] == "partial"
    receipt = json.loads(case.receipt_path.read_text())
    assert receipt["state"] == "committed"
    assert receipt["release_ground"] == "verified_generic_final_cleanup"
    assert receipt["remote_terminal_proof"] == {"guardian_drained": True}
    assert receipt["original_completion_proven"] is False
    assert base64.b64decode(receipt["original_final_cleanup_base64"]) == before[case.marker_path]
    assert len(receipt["group"]) == 2
    for row in receipt["group"]:
        for name, path in (("owner", row["lock_path"]), ("fence", row["quarantine_path"])):
            assert base64.b64decode(row[f"original_{name}_base64"]) == before[Path(path)]
    assert base64.b64decode(receipt["terminal_journal_entries"][0]["original_base64"]) == before[case.journal_path]
    assert not case.journal_path.exists()
    assert case.scientific_path.read_bytes() == scientific
    mutable = {case.manifest_path, case.journal_path, case.run_lock.lock_path, case.run_lock.quarantine_path,
               case.physical.lock_path, case.physical.quarantine_path}
    assert all(path.read_bytes() == raw for path, raw in before.items() if path not in mutable)
    assert "session_id" not in json.loads(case.physical.lock_path.read_text())
    assert len(case.remote_calls) == 1
    assert case.remote_calls[0][1][0]["operation_id"] == "ssh-stream-original"
    for lock in (case.run_lock, case.physical):
        assert json.loads(lock.lock_path.read_text())["state"] == "released"
        assert not lock.quarantine_path.exists()
        lock.acquire()
        lock.release()


def test_final_recovery_replay_is_byte_preserving_and_skips_new_observation(final_recovery_case):
    case = final_recovery_case
    resource_recovery.recover_generic_final_cleanup(**case.kwargs)
    paths = [case.receipt_path, case.manifest_path, case.run_lock.lock_path, case.physical.lock_path,
             case.marker_path, case.scientific_path]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert resource_recovery.recover_generic_final_cleanup(**case.kwargs)["status"] == "already_committed"
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == raw for path, raw in before.items())
    assert len(case.remote_calls) == 1


@pytest.mark.parametrize("fault", ["request_session", "input_run", "registry", "run_fence", "physical_fence",
                                  "profile", "plan", "snapshot", "controller", "journal_target", "authorization"])
def test_final_recovery_rejects_foreign_or_changed_bound_sources(final_recovery_case, fault):
    case = final_recovery_case
    if fault == "request_session":
        case.kwargs["session_id"] = "d" * 32
    elif fault == "input_run":
        value = json.loads(case.evidence_path.read_text())
        value["run_id"] = "foreign-run"
        _write(case.evidence_path, value)
        _refresh(case)
    elif fault == "registry":
        registry = yaml.safe_load(case.registry_path.read_text())
        registry["hardware_setups"][0]["host"]["address"] = "192.0.2.105"
        case.registry_path.write_text(yaml.safe_dump(registry))
        _refresh(case)
    elif fault in {"run_fence", "physical_fence"}:
        path = case.run_lock.quarantine_path if fault == "run_fence" else case.physical.quarantine_path
        value = json.loads(path.read_text())
        value["owner"]["session_id"] = "d" * 32
        _write(path, value)
        _refresh(case)
    elif fault == "profile":
        path = case.run / "profile.yaml"
        value = yaml.safe_load(path.read_text())
        value["description"] = "changed-after-original-start"
        path.write_text(yaml.safe_dump(value))
        _refresh(case)
    elif fault == "plan":
        path = case.run / "effective_execution_plan.json"
        value = json.loads(path.read_text())
        value["models"] = ["foreign-model"]
        _write(path, value)
        _refresh(case)
    elif fault == "snapshot":
        path = case.run / "profile_start_snapshot.json"
        value = json.loads(path.read_text())
        value["snapshot_sha256"] = "sha256:" + "0" * 64
        _write(path, value)
        _refresh(case)
    elif fault == "controller":
        value = json.loads(case.observed_path.read_text())
        value["controller"]["start_ticks"] = str(int(value["controller"]["start_ticks"]) + 1)
        _write(case.observed_path, value)
        _refresh(case)
    elif fault == "journal_target":
        value = json.loads(case.journal_path.read_text())
        value["control_argv_prefix"][-1] = "fixture@192.0.2.102"
        _write(case.journal_path, value)
        _refresh(case)
    else:
        value = json.loads(case.authorization_path.read_text())
        value["authorized"] = False
        _write(case.authorization_path, value)
    _assert_blocked(case)


@pytest.mark.parametrize("fault", ["workers", "futures", "shutdown_thread", "queue"])
def test_final_recovery_requires_genuine_finished_management_shutdown(final_recovery_case, fault):
    case = final_recovery_case
    marker = json.loads(case.marker_path.read_text())
    service = marker["details"]["management_service_shutdown"]
    if fault == "workers":
        service["central_quality_service"]["workers"] = [{"pid": 123, "alive": True}]
    elif fault == "futures":
        service["management_reference"]["futures"] = {"fixture": "running"}
    elif fault == "shutdown_thread":
        service["shutdown_thread"]["alive"] = True
    else:
        service["central_quality_service"]["queue_size"] = 1
    _write(case.marker_path, marker)
    _refresh(case)
    _assert_blocked(case, match="management workers/futures shutdown")
    assert not case.remote_calls


def test_final_recovery_rechecks_original_bytes_after_remote_observation(final_recovery_case, monkeypatch):
    case = final_recovery_case
    before = {path: path.read_bytes() for path in [case.manifest_path, case.journal_path, case.scientific_path,
        case.run_lock.lock_path, case.run_lock.quarantine_path, case.physical.lock_path, case.physical.quarantine_path]}

    def changed_during_observation(*args, **kwargs):
        path = case.run / "profile.yaml"
        path.write_bytes(path.read_bytes() + b"\n")
        return copy.deepcopy(case.remote)

    monkeypatch.setattr(resource_recovery_observation, "observe_generic_final_cleanup_remote", changed_during_observation)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="changed during observation"):
        resource_recovery.recover_generic_final_cleanup(**case.kwargs)
    assert not case.receipt_path.exists()
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_final_recovery_replay_rejects_changed_original_source_bytes(final_recovery_case):
    case = final_recovery_case
    resource_recovery.recover_generic_final_cleanup(**case.kwargs)
    paths = [case.receipt_path, case.manifest_path, case.run_lock.lock_path, case.physical.lock_path]
    before = {path: path.read_bytes() for path in paths}
    profile = case.run / "profile.yaml"
    profile.write_bytes(profile.read_bytes() + b"\n")
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="original profile changed"):
        resource_recovery.recover_generic_final_cleanup(**case.kwargs)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert len(case.remote_calls) == 1


@pytest.mark.parametrize("fault", ["no_drain_or_boot", "lease_present", "busy", "visibility", "wrong_operation"])
def test_final_recovery_requires_exact_fresh_remote_ownership_and_terminal_proof(final_recovery_case, fault):
    case = final_recovery_case
    if fault in {"no_drain_or_boot", "lease_present"}:
        case.remote["operations"][0]["guardian_drained"] = False
        case.remote["operations"][0]["lease_present"] = fault == "lease_present"
    elif fault == "busy":
        case.remote["ownership"].update(ok=False, status="busy", processes=[{"pid": 123}])
    elif fault == "visibility":
        case.remote["ownership"]["visibility_errors"] = ["procfs_visibility_restricted"]
    else:
        case.remote["operations"][0]["operation_id"] = "foreign-operation"
    _assert_blocked(case)


@pytest.mark.parametrize("final_recovery_case", ["active"], indirect=True)
def test_live_original_controller_blocks_final_recovery(final_recovery_case):
    _assert_blocked(final_recovery_case, match="active, same-DUT or unknown")
    assert not final_recovery_case.remote_calls


def test_detached_descendant_bound_to_original_run_blocks_final_recovery(final_recovery_case):
    case = final_recovery_case
    child = subprocess.Popen([sys.executable, "-B", "-c", "import sys; sys.stdin.buffer.read(1)", str(case.run)],
                             stdin=subprocess.PIPE, start_new_session=True)
    try:
        _assert_blocked(case, match="active, same-DUT or unknown")
        assert not case.remote_calls
    finally:
        assert child.stdin is not None
        child.stdin.write(b"x")
        child.stdin.flush()
        child.wait(timeout=5)


@pytest.mark.parametrize("status", ["same_dut", "unknown"])
def test_unresolved_local_admission_cannot_be_overridden_by_remote_drain(final_recovery_case, monkeypatch, status):
    monkeypatch.setattr(resource_recovery, "_generic_orphan_local_observation",
                        lambda *args, **kwargs: {"ok": False, "recorded_scope": {"status": status}})
    _assert_blocked(final_recovery_case, match="active, same-DUT or unknown")
    assert not final_recovery_case.remote_calls


def _clock(after: float) -> dict:
    boot = int(after) + 2
    return {"read_only": True, "boot_id": "bee0d1a6-269c-4604-ab90-65c92e1ee391",
        "kernel_btime_unix_s": boot, "proc_uptime_s": 100.5,
        "clock_boottime_start_s": 100.0, "clock_boottime_end_s": 101.0,
        "remote_start_unix_ns": int((boot + 100.25) * 1e9),
        "remote_end_unix_ns": int((boot + 101.25) * 1e9),
        "local_start_unix_ns": int((boot + 100.1) * 1e9),
        "local_end_unix_ns": int((boot + 101.4) * 1e9)}


def _bind_boot(case) -> None:
    clock = _clock(datetime.fromisoformat(RESULT_AT).timestamp())
    case.remote.update(clock)
    case.remote["operations"][0]["guardian_drained"] = False
    path = case.evidence_path.parent / "boot-observation.json"
    _write(path, {"schema": "generic420-h8-kernel-boot-time-readonly-result-v1", **case.identity,
        "user_host": "fixture@192.0.2.104", "returncode": 0, "read_only": True,
        "bound_original_sources_unchanged": True,
        "bound_original_sources": [_descriptor(case.marker_path), _descriptor(case.result_path)],
        "observation": {key: value for key, value in clock.items() if not key.startswith("local_")},
        **{key: clock[key] for key in ("local_start_unix_ns", "local_end_unix_ns")}})
    evidence = json.loads(case.evidence_path.read_text())
    evidence["remote_boot_observation"] = _descriptor(path)
    _write(case.evidence_path, evidence)
    _refresh(case)


def test_absent_old_lease_can_close_only_with_bound_matching_post_cleanup_boot(final_recovery_case):
    case = final_recovery_case
    _bind_boot(case)
    assert resource_recovery.recover_generic_final_cleanup(**case.kwargs)["status"] == "committed"
    proof = json.loads(case.receipt_path.read_text())["remote_terminal_proof"]
    assert proof["old_kernel_scope_ended"] is True
    assert proof["historical_guardian_drain_proven"] is False


@pytest.mark.parametrize("fault", ["fresh_boot_changed", "old_sources_unbound", "old_boot_wrong_session"])
def test_absent_lease_boot_proof_cannot_be_rebound_to_foreign_history(final_recovery_case, fault):
    case = final_recovery_case
    _bind_boot(case)
    evidence = json.loads(case.evidence_path.read_text())
    path = Path(evidence["remote_boot_observation"]["path"])
    old = json.loads(path.read_text())
    if fault == "fresh_boot_changed":
        case.remote["boot_id"] = "e7c09869-722d-487c-9497-c582e9071f8f"
    elif fault == "old_sources_unbound":
        old["bound_original_sources"] = []
    else:
        old["session_id"] = "d" * 32
    _write(path, old)
    _refresh(case)
    _assert_blocked(case)


def test_kernel_boot_proof_requires_consistent_clocks_and_post_marker_boundary():
    clock = _clock(1000)
    assert resource_recovery._generic_final_clock_proof(clock, after=1000)["kernel_btime_unix_s"] == 1002


@pytest.mark.parametrize("fault", ["utc_outside_call", "clock_skew", "uptime", "btime", "before_marker", "nan"])
def test_kernel_boot_proof_rejects_inconsistent_or_nonterminal_clock_evidence(fault):
    clock, after = _clock(1000), 1000
    if fault == "utc_outside_call":
        clock["local_start_unix_ns"] = clock["remote_start_unix_ns"] + 1
    elif fault == "clock_skew":
        clock["remote_end_unix_ns"] += 2_000_000_000
        clock["local_end_unix_ns"] += 2_000_000_000
    elif fault == "uptime":
        clock["proc_uptime_s"] = 120
    elif fault == "btime":
        clock["kernel_btime_unix_s"] -= 10
    elif fault == "before_marker":
        after = clock["kernel_btime_unix_s"] - 1
    else:
        clock["proc_uptime_s"] = float("nan")
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        resource_recovery._generic_final_clock_proof(clock, after=after)


def test_local_wrapper_accepts_only_proven_independent_dut_authority(final_recovery_case, monkeypatch):
    from onnx_splitpoint_tool.workflow import generic_completed_task

    case = final_recovery_case
    authority = {"ok": True, "selected_resources": [case.resource], "normal_quality_owners": [
        {"resources": ["dut:192.0.2.102"], "ssh_targets": [{"user_host": "fixture@192.0.2.102", "port": "22"}]}]}
    calls = []

    def scoped(targets, observed):
        calls.append((targets, observed))
        return authority

    monkeypatch.setattr(generic_completed_task, "_scoped_controller_owners", scoped)
    kwargs = dict(run_dir=case.run, profile_path=str(case.run / "profile.yaml"),
        process_groups={case.controller_identity["pgid"]}, ssh_host="192.0.2.104",
        known_operations=["ssh-stream-original"], controller_targets={"fixture-h8": {
            "id": "fixture-h8", "remote": {"host": "192.0.2.104", "user": "fixture"}}})
    owners = [{"pid": case.controller_identity["pid"], "start_time_ticks": case.controller_identity["start_ticks"]}]
    assert resource_recovery._generic_orphan_local_observation(owners, **kwargs)["ok"] is True
    assert calls[0][1]["recorded_owners"] == []
    authority["normal_quality_owners"][0]["resources"] = [case.resource]
    assert resource_recovery._generic_orphan_local_observation(owners, **kwargs)["ok"] is False
