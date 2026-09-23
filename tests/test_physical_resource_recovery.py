"""Explicit four-resource recovery, using temporary real kernel locks only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import threading

import pytest
import yaml

from onnx_splitpoint_tool.remote import process_lease, resource_recovery
from onnx_splitpoint_tool.workflow import run_control


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def recovery_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    interlock_path = tmp_path / "locks" / "workflow_platform_interlock.lock"
    monkeypatch.setattr(
        run_control, "platform_workflow_interlock_path",
        lambda: interlock_path,
    )
    monkeypatch.setattr(
        resource_recovery, "observe_resource_owners",
        lambda *args, **kwargs: {"ok": True, "controller": {}, "remote": {}},
    )
    run_dir = tmp_path / "runs" / "failed-capture"
    session_id, operation_id = "session-a", "capture-" + "a" * 32
    setup_id = "fixture-h10"
    resources = [
        "controller:capture", "controller:nic",
        "dut:192.0.2.145", "source:192.0.2.176",
    ]
    journal = run_dir / "jobs" / "remote_process_leases" / session_id
    attempt = run_dir / "reports" / "energy" / "run_000"
    request = {
        "run_id": run_dir.name, "session_id": session_id,
        "operation_id": operation_id, "token": "b" * 32,
        "setup_id": setup_id, "attempt_dir": str(attempt),
        "owner_pid": 99999991, "owner_start_ticks": "101",
        "collector_pid": 99999992, "collector_start_ticks": "102",
        "purpose": "capture", "phase": "FINALIZING", "sequence": 3,
        "process_started": True, "continuation": None,
    }
    request_path = journal / f"{operation_id}.resource-request.json"
    reply_path = journal / f"{operation_id}.resource-reply.json"
    _write(request_path, request)
    reply = {k: request[k] for k in ("run_id", "session_id", "operation_id", "token")}
    reply.update(state="STOP", reason="campaign_source_completion_unresolved")
    _write(reply_path, reply)
    config_path = journal / "controller.resources.json"
    _write(config_path, {
        "run_id": run_dir.name, "session_id": session_id,
        "run_dir": str(run_dir), "parent_pid": 99999993,
        "parent_start_ticks": "103",
        "setups": {setup_id: resources + ["controller:cpu", "controller:io"]},
    })
    _write(attempt / "collector_stdout.log.cleanup.json", {
        "collector_pid": request["collector_pid"],
        "collector_start_ticks": request["collector_start_ticks"],
        "process_started": True, "owned_tree_quiescent": True,
    })
    (attempt / "collector_stdout.log").write_text(
        "Fast Firmware GO sent monotonic_ns=1 local=192.0.2.1:4000 requested_device_us=86000000\n"
        "Fast Firmware source closed monotonic_ns=2 local=192.0.2.1:4000 verified=false reason=Err(\"silence timeout\")\n",
        encoding="utf-8",
    )
    _write(run_dir / "run_manifest.json", {
        "schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": run_dir.name, "run_dir": str(run_dir), "status": "failed",
        "current_session_id": session_id,
        "execution_sessions": [{"session_id": session_id, "status": "failed"}],
    })
    _write(run_dir / "energy_task_budget.json", {
        "tasks": {"original": {"chains": [{"collector_started": True, "finished": False}]}},
        "sources": {"192.0.2.176": {"stop_reason": "", "transport_failures": 1}},
    })
    registry_path = tmp_path / "hardware.yaml"
    registry_path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "energy_defaults": {"data_port": 3000, "channel": 0, "sample_rate": 2000},
        "hardware_setups": [{
            "id": setup_id, "label": "Temporary fixture",
            "host": {"address": "192.0.2.145", "user": "fixture", "port": 22,
                     "base_dir": "/fixture"},
            "energy": {"enabled": True, "urecs_address": "192.0.2.176"},
        }],
    }), encoding="utf-8")
    locks = {}
    for resource in resources:
        lock = run_control.EvaluationRunLock.for_resource(resource, owner={
            "run_id": run_dir.name, "session_id": session_id,
            "operation_id": operation_id, "owner_pid": request["owner_pid"],
            "pid": 99999993,
        })
        lock.acquire()
        lock.commit_quarantine_fence({
            "reason": "campaign_source_completion_unresolved",
            "resource_request": request, "source_completion_unproven": True,
        })
        lock.release()
        locks[resource] = lock
    operator = {
        "source": "source:192.0.2.176", "operator": "fixture-operator",
        "action": "Fixture: documented source recovery completed",
        "performed_at": datetime.now(timezone.utc).isoformat(),
        "supply_effect": "Fixture: source only; DUT supply unchanged",
        "ready_observation": "Fixture: source returned to documented ready state",
        "action_performed": True,
    }
    kwargs = dict(run_dir=run_dir, session_id=session_id,
                  operation_id=operation_id, setup_id=setup_id,
                  operator_evidence=operator, registry_path=registry_path)
    preserved = {
        p: p.read_bytes() for p in [
            run_dir / "run_manifest.json", run_dir / "energy_task_budget.json",
            request_path, attempt / "collector_stdout.log",
            attempt / "collector_stdout.log.cleanup.json", registry_path,
        ]
    }
    return SimpleNamespace(
        kwargs=kwargs, run_dir=run_dir, journal=journal, resources=resources,
        locks=locks, request=request, request_path=request_path,
        reply=reply, reply_path=reply_path, config_path=config_path,
        preserved=preserved, operator=operator, interlock_path=interlock_path,
    )


def _fences(case):
    return {resource: lock.quarantine_path.read_bytes() for resource, lock in case.locks.items()}


def _assert_preserved(case):
    assert all(path.read_bytes() == data for path, data in case.preserved.items())
    reply = json.loads(case.reply_path.read_text())
    assert all(reply[key] == value for key, value in case.reply.items())


def test_recovery_releases_exact_group_and_normal_admission_preserves_old_failure(recovery_case):
    case = recovery_case
    unrelated = run_control.EvaluationRunLock.for_resource(
        "source:192.0.2.199", owner={"run_id": "unrelated-run"},
    )
    unrelated.acquire()
    unrelated.commit_quarantine_fence({"reason": "unrelated_source_stop"})
    unrelated.release()
    unrelated_before = unrelated.quarantine_path.read_bytes()
    result = resource_recovery.recover_capture_resources(**case.kwargs)
    assert result["status"] == "released"
    assert set(result["resources"]) == set(case.resources)
    reply = json.loads(case.reply_path.read_text())
    assert all(reply[key] == value for key, value in case.reply.items())
    assert reply["recovery"]
    assert case.operator["operator"] in json.dumps(reply["recovery"])
    assert unrelated.quarantine_path.read_bytes() == unrelated_before
    for resource in case.resources:
        assert not case.locks[resource].quarantine_path.exists()
        normal = run_control.EvaluationRunLock.for_resource(resource, owner={"run_id": "next-run"})
        normal.acquire()
        normal.release()
    _assert_preserved(case)


def test_repeat_recovery_is_idempotent_without_new_mutation(recovery_case):
    case = recovery_case
    resource_recovery.recover_capture_resources(**case.kwargs)
    before = {case.reply_path: case.reply_path.read_bytes()}
    before.update({lock.lock_path: lock.lock_path.read_bytes() for lock in case.locks.values()})
    result = resource_recovery.recover_capture_resources(**case.kwargs)
    assert result["status"] == "already_released"
    assert all(path.read_bytes() == data for path, data in before.items())
    _assert_preserved(case)


def test_old_recovery_cannot_clear_a_later_capture_quarantine(recovery_case):
    case = recovery_case
    resource_recovery.recover_capture_resources(**case.kwargs)
    resource = "source:192.0.2.176"
    later = run_control.EvaluationRunLock.for_resource(resource, owner={
        "run_id": "later-run", "session_id": "later-session", "operation_id": "later-capture",
    })
    later.acquire()
    later.commit_quarantine_fence({"reason": "campaign_source_completion_unresolved"})
    later.release()
    original = later.quarantine_path.read_bytes()
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert later.quarantine_path.read_bytes() == original
    _assert_preserved(case)


def test_held_resource_blocks_whole_recovery_group(recovery_case):
    case = recovery_case
    original = _fences(case)
    with case.locks["dut:192.0.2.145"].lock_path.open("r+b") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(Exception):
            resource_recovery.recover_capture_resources(**case.kwargs)
        assert _fences(case) == original
    _assert_preserved(case)


@pytest.mark.parametrize("reason", ["live_controller_owner", "live_remote_owner", "remote_visibility_unknown"])
def test_live_or_unproven_owners_keep_all_fences(recovery_case, monkeypatch, reason):
    case = recovery_case
    original = _fences(case)
    monkeypatch.setattr(resource_recovery, "observe_resource_owners", lambda *a, **k: {
        "ok": False, "reason": reason,
        "controller": {"status": "busy" if reason == "live_controller_owner" else "clear"},
        "remote": {"status": "unknown" if reason == "remote_visibility_unknown" else "busy"},
    })
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == original
    _assert_preserved(case)


@pytest.mark.parametrize("change", [
    "cleanup_missing", "cleanup_identity", "cleanup_unproven",
    "parent_running", "remote_descriptor",
])
def test_unresolved_old_execution_cannot_release(recovery_case, change):
    case = recovery_case
    if change == "remote_descriptor":
        _write(case.journal / "unresolved.remote-lease.json", {"unproven": True})
    elif change == "parent_running":
        path = case.run_dir / "run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["status"] = "running"
        _write(path, manifest)
        case.preserved[path] = path.read_bytes()
    else:
        path = Path(case.request["attempt_dir"]) / "collector_stdout.log.cleanup.json"
        if change == "cleanup_missing":
            path.unlink()
            del case.preserved[path]
        else:
            cleanup = json.loads(path.read_text())
            if change == "cleanup_identity":
                cleanup["collector_start_ticks"] = "999"
            else:
                cleanup["owned_tree_quiescent"] = False
            _write(path, cleanup)
            case.preserved[path] = path.read_bytes()
    original = _fences(case)
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == original
    _assert_preserved(case)


def test_changed_registry_during_observation_blocks_without_clearing_fences(recovery_case, monkeypatch):
    case = recovery_case
    original = _fences(case)
    registry = case.kwargs["registry_path"]
    changed = registry.read_bytes() + b"\n# concurrent registry update\n"

    def observe(*args, **kwargs):
        registry.write_bytes(changed)
        return {"ok": True, "controller": {}, "remote": {}}

    monkeypatch.setattr(resource_recovery, "observe_resource_owners", observe)
    with pytest.raises(Exception, match="registry or ownership changed"):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == original
    assert registry.read_bytes() == changed
    case.preserved[registry] = changed
    _assert_preserved(case)


@pytest.mark.parametrize("change", ["operation", "resource_set", "foreign_fence", "foreign_lock_owner", "foreign_hostname"])
def test_wrong_operation_resource_or_foreign_fence_is_not_released(recovery_case, change):
    case = recovery_case
    kwargs = copy.deepcopy(case.kwargs)
    if change == "operation":
        kwargs["operation_id"] = "capture-" + "c" * 32
    elif change == "resource_set":
        config = json.loads(case.config_path.read_text())
        config["setups"][kwargs["setup_id"]].append("source:192.0.2.199")
        _write(case.config_path, config)
    elif change == "foreign_fence":
        path = case.locks["controller:nic"].quarantine_path
        fence = json.loads(path.read_text())
        fence["owner"]["resource_request"]["operation_id"] = "capture-" + "c" * 32
        _write(path, fence)
    else:
        path = case.locks["controller:nic"].lock_path
        owner = json.loads(path.read_text())
        owner["hostname" if change == "foreign_hostname" else "operation_id"] = "foreign-owner"
        _write(path, owner)
    original = _fences(case)
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**kwargs)
    assert _fences(case) == original
    _assert_preserved(case)


@pytest.mark.parametrize("field,value", [
    ("action_performed", False), ("operator", ""), ("action", ""),
    ("ready_observation", ""), ("supply_effect", ""),
    ("source", "source:192.0.2.199"), ("performed_at", "2099-01-01T00:00:00+00:00"),
    ("performed_at", "2026-09-23T12:00:00"),
])
def test_missing_or_invalid_operator_evidence_cannot_release(recovery_case, field, value):
    case = recovery_case
    original = _fences(case)
    case.operator[field] = value
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == original
    _assert_preserved(case)


def test_concurrent_recovery_cannot_enter_same_group(recovery_case, monkeypatch):
    case = recovery_case
    entered, finish = threading.Event(), threading.Event()

    def observe(*args, **kwargs):
        entered.set()
        assert finish.wait(5), "test did not release the controlled observer"
        return {"ok": True, "controller": {}, "remote": {}}

    monkeypatch.setattr(resource_recovery, "observe_resource_owners", observe)
    with ThreadPoolExecutor(max_workers=1) as workers:
        first = workers.submit(resource_recovery.recover_capture_resources, **case.kwargs)
        try:
            assert entered.wait(5), "recovery did not reach observer with held locks"
            with pytest.raises(Exception):
                resource_recovery.recover_capture_resources(**case.kwargs)
        finally:
            finish.set()
        assert first.result(timeout=5)["status"] == "released"
    _assert_preserved(case)


def test_second_sidecar_failure_restores_all_original_fences(recovery_case, monkeypatch):
    case = recovery_case
    originals = {r: json.loads(lock.quarantine_path.read_text())["owner"] for r, lock in case.locks.items()}
    paths = {str(lock.quarantine_path) for lock in case.locks.values()}
    original_unlink = os.unlink
    removals = []

    def fail_second(path, *args, **kwargs):
        if os.fspath(path) in paths:
            removals.append(os.fspath(path))
            if len(removals) == 2:
                raise OSError("controlled second-sidecar unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_second)
    with pytest.raises(Exception, match="controlled second-sidecar"):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert len(removals) >= 2
    for resource, lock in case.locks.items():
        assert json.loads(lock.quarantine_path.read_text())["owner"] == originals[resource]
        normal = run_control.EvaluationRunLock.for_resource(resource, owner={"run_id": "next-run"})
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            normal.acquire()
    _assert_preserved(case)
    # The same exact operation can finish after the transient I/O failure;
    # its original STOP and failed measurement remain unchanged.
    result = resource_recovery.recover_capture_resources(**case.kwargs)
    assert result["status"] == "released"
    assert all(not lock.quarantine_path.exists() for lock in case.locks.values())
    _assert_preserved(case)


def test_second_released_owner_write_failure_keeps_normal_capture_blocked(recovery_case, monkeypatch):
    case = recovery_case
    originals = {
        resource: json.loads(lock.quarantine_path.read_text())
        for resource, lock in case.locks.items()
    }
    original_write = run_control.EvaluationRunLock.write_recovery_owner
    released_writes = []

    def fail_second_released(self, owner):
        if owner.get("state") == "released":
            released_writes.append(self.lock_path)
            if len(released_writes) == 2:
                raise OSError("controlled second released-owner write failure")
        return original_write(self, owner)

    monkeypatch.setattr(run_control.EvaluationRunLock, "write_recovery_owner", fail_second_released)
    with pytest.raises(OSError, match="controlled second released-owner"):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert len(released_writes) == 2
    assert json.loads(case.reply_path.read_text())["recovery"]["status"] == "blocked"
    for resource, lock in case.locks.items():
        assert json.loads(lock.quarantine_path.read_text()) == originals[resource]
        normal = run_control.EvaluationRunLock.for_resource(resource, owner={"run_id": "next-run"})
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            normal.acquire()
    _assert_preserved(case)


@pytest.mark.parametrize("stage", ["reply", "sidecar"])
def test_strict_directory_sync_failure_restores_complete_group(recovery_case, monkeypatch, stage):
    case = recovery_case
    originals = {
        resource: json.loads(lock.quarantine_path.read_text())
        for resource, lock in case.locks.items()
    }
    failing_directory = case.journal if stage == "reply" else next(iter(case.locks.values())).lock_path.parent
    original_sync = process_lease.RemoteProcessLeaseJournal._fsync_directory
    failed = []

    def fail_once(directory, *, strict=False):
        if Path(directory) == failing_directory and strict and not failed:
            failed.append(Path(directory))
            raise OSError("controlled strict directory fsync failure")
        return original_sync(directory, strict=strict)

    monkeypatch.setattr(
        process_lease.RemoteProcessLeaseJournal, "_fsync_directory", staticmethod(fail_once),
    )
    with pytest.raises(OSError, match="controlled strict directory fsync"):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert failed == [failing_directory]
    for resource, lock in case.locks.items():
        assert json.loads(lock.quarantine_path.read_text()) == originals[resource]
        normal = run_control.EvaluationRunLock.for_resource(resource, owner={"run_id": "next-run"})
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            normal.acquire()
    _assert_preserved(case)


def test_local_crash_during_last_owner_truncate_retains_all_four_fences(recovery_case):
    case = recovery_case
    original_fences = _fences(case)
    script = """
import json
import os
from pathlib import Path
import sys
from onnx_splitpoint_tool.remote import resource_recovery
from onnx_splitpoint_tool.workflow import run_control

payload = json.loads(sys.argv[1])
run_control.platform_workflow_interlock_path = lambda: Path(payload['interlock_path'])
resource_recovery.observe_resource_owners = lambda *a, **k: {'ok': True, 'controller': {}, 'remote': {}}
original_write = run_control.EvaluationRunLock.write_recovery_owner
released_writes = []
def crash_during_last_owner_write(self, owner):
    if owner.get('state') == 'released':
        released_writes.append(str(self.lock_path))
        if len(released_writes) == 4:
            self._fh.seek(0)
            self._fh.truncate(0)
            self._fh.flush()
            os.fsync(self._fh.fileno())
            os._exit(87)
    return original_write(self, owner)
run_control.EvaluationRunLock.write_recovery_owner = crash_during_last_owner_write
resource_recovery.recover_capture_resources(**payload['kwargs'])
raise AssertionError('controlled last-owner crash was not reached')
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script, json.dumps({
            "interlock_path": str(case.interlock_path), "kwargs": case.kwargs,
        }, default=str)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert completed.returncode == 87, completed.stdout + completed.stderr
    assert _fences(case) == original_fences
    assert sum(lock.lock_path.read_bytes() == b"" for lock in case.locks.values()) == 1
    for resource in case.resources:
        normal = run_control.EvaluationRunLock.for_resource(resource, owner={"run_id": "next-run"})
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            normal.acquire()
    _assert_preserved(case)


def _recovery_cli_argv(case):
    argv = ["recover-capture"]
    for key in ("run_dir", "session_id", "operation_id", "setup_id"):
        argv.extend(["--" + key.replace("_", "-"), str(case.kwargs[key])])
    argv.extend(["--registry", str(case.kwargs["registry_path"])])
    for key, value in case.operator.items():
        if key != "action_performed":
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    return argv


def test_cli_recover_capture_dispatches_exact_confirmed_product_call(recovery_case, monkeypatch, capsys):
    case = recovery_case
    calls = []
    expected_result = {"status": "released", "resources": case.resources}

    def recover(**kwargs):
        calls.append(kwargs)
        return expected_result

    monkeypatch.setattr(resource_recovery, "recover_capture_resources", recover)
    assert process_lease.process_lease_cli_main(
        _recovery_cli_argv(case) + ["--confirm-action-performed"]
    ) == 0
    assert calls == [{
        **case.kwargs,
        "run_dir": str(case.run_dir),
        "registry_path": str(case.kwargs["registry_path"]),
    }]
    assert json.loads(capsys.readouterr().out) == expected_result
    _assert_preserved(case)


def test_cli_recover_capture_requires_explicit_action_confirmation(recovery_case, monkeypatch, capsys):
    case = recovery_case
    calls = []
    monkeypatch.setattr(resource_recovery, "recover_capture_resources", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit) as exc:
        process_lease.process_lease_cli_main(_recovery_cli_argv(case))
    assert exc.value.code == 2
    assert "--confirm-action-performed" in capsys.readouterr().err
    assert calls == []
    _assert_preserved(case)
