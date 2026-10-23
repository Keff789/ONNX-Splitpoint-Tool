"""Atomic final-cleanup closure using temporary run and physical lock files.

The caller owns scientific and process-evidence validation. These tests exercise
only the existing durable group transition; no test contacts hardware.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import run_control


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False) + "\n")


@pytest.fixture
def final_case(tmp_path, monkeypatch):
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path",
                        lambda: tmp_path / "locks/workflow_platform_interlock.lock")
    run = tmp_path / "runs/original-final-cleanup"
    session = "c" * 32
    owner = {"run_id": run.name, "session_id": session,
             "pid": 12345, "process_start_ticks": 98765,
             "profile_sha256": "original-profile", "plan_sha256": "original-plan"}
    run_lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    run_lock.acquire()
    run_lock.commit_quarantine_fence({
        "run_id": run.name, "session_id": session, "phase": "final_cleanup",
        "reason": "final_process_quiescence_unresolved",
        "local_process_quiescence_proven": True,
        "remote_process_quiescence_proven": False})
    run_lock.release(write_released_state=False)
    resource = "dut:192.0.2.104"
    physical = run_control.EvaluationRunLock.for_resource(resource, owner={
        **{key: value for key, value in owner.items() if key != "session_id"},
        "purpose": "generic"})
    physical.acquire()
    physical.commit_quarantine_fence({"reason": "generic_remote_cleanup_requires_reconciliation"})
    physical.release(write_released_state=False)
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": run.name, "run_dir": str(run), "current_session_id": session,
        "status": "partial", "technical_status": "partial",
        "finished_at": "2026-10-23T00:48:50+02:00",
        "updated_at": "2026-10-23T00:48:50+02:00", "quality_decision": "FAIL",
        "execution_sessions": [
            {"session_id": "b" * 32, "status": "completed", "finished_at": "old-finish"},
            {"session_id": session, "status": "partial", "finished_at": "original-finish",
             "last_updated_at": "original-update", "scientific_summary": {"quality": "FAIL"}}],
        "models": {"fixture": {"stages": {"quality": {"status": "completed", "value": 0.812}}}},
        "unresolved_cleanup": {"original_reason": "remote operation unresolved"}}
    manifest_path = run / "run_manifest.json"
    _write(manifest_path, manifest)
    journal_path = (run / "jobs/remote_process_leases" / session
                    / "ssh-stream-original.remote-lease.json")
    _write(journal_path, {"operation_id": "ssh-stream-original", "session_id": session,
                          "historical": "original unresolved H8 descriptor"})
    quarantine_path = run / "jobs/p01_unresolved_cleanup_quarantine.json"
    _write(quarantine_path, {"phase": "final_cleanup", "session_id": session,
                             "details": {"reason": "final_process_quiescence_unresolved"}})
    result = run / "models/fixture/quality.json"
    _write(result, {"quality": "FAIL", "records": [{"image": "a", "score": 0.812}]})
    receipt_path = tmp_path / "recovery/receipt.json"
    receipt_data = {
        "schema": "onnx-splitpoint/generic-orphan-recovery", "schema_version": 1,
        "release_ground": "verified_generic_final_cleanup", "recovery_id": "today-final-cleanup",
        "run_id": run.name, "session_id": session, "run_dir": str(run),
        "operator": "fixture-operator"}
    locks = [run_lock, physical]
    return SimpleNamespace(
        run=run, session=session, run_lock=run_lock, physical_locks={resource: physical},
        resource=resource, locks=locks, manifest_path=manifest_path,
        manifest_before=manifest_path.read_bytes(), manifest_after=copy.deepcopy(manifest),
        journal_path=journal_path, journal_entries={journal_path: journal_path.read_bytes()},
        receipt_path=receipt_path, receipt_data=receipt_data,
        preserved={path: path.read_bytes() for path in (quarantine_path, result)},
        original_group={path: path.read_bytes() for lock in locks
                        for path in (lock.lock_path, lock.quarantine_path)})


@contextmanager
def _held_group(case):
    held = []
    try:
        for lock in [case.run_lock, *case.physical_locks.values()]:
            lock.acquire_for_recovery()
            held.append(lock)
        yield
    finally:
        for lock in reversed(held):
            lock.release(write_released_state=False)


def _commit(case, **overrides):
    values = dict(run_lock=case.run_lock, physical_locks=case.physical_locks,
                  receipt_path=case.receipt_path, receipt_data=case.receipt_data,
                  manifest_path=case.manifest_path, manifest_before=case.manifest_before,
                  manifest_after=case.manifest_after, terminal_journal_entries=case.journal_entries)
    values.update(overrides)
    with _held_group(case):
        return run_control.commit_generic_orphan_recovery_group(**values)


def _normal_lock(case, *, physical=False):
    if physical:
        return run_control.EvaluationRunLock.for_resource(case.resource, owner={"run_id": "next-run"})
    return run_control.EvaluationRunLock(out_root=case.run.parent, run_dir=case.run,
                                       owner={"run_id": case.run.name})


def _assert_originals(case):
    assert case.manifest_path.read_bytes() == case.manifest_before
    assert all(path.read_bytes() == raw for path, raw in case.original_group.items())
    assert all(path.read_bytes() == raw for path, raw in case.journal_entries.items())
    assert all(path.read_bytes() == raw for path, raw in case.preserved.items())


def test_final_cleanup_preserves_complete_partial_history_and_archives_original_bytes(final_case):
    case = final_case
    receipt = _commit(case)
    manifest = json.loads(case.manifest_path.read_bytes())
    marker = manifest.pop("generic_orphan_recovery")
    assert manifest == json.loads(case.manifest_before)
    assert marker == {"recovery_id": case.receipt_data["recovery_id"], "session_id": case.session,
                      "receipt_path": str(case.receipt_path)}
    assert receipt["state"] == "committed"
    assert receipt["schema"] == "onnx-splitpoint/generic-orphan-recovery"
    assert base64.b64decode(receipt["manifest_before_base64"]) == case.manifest_before
    assert receipt["manifest_before_sha256"] == hashlib.sha256(case.manifest_before).hexdigest()
    assert len(receipt["group"]) == 2
    for row in receipt["group"]:
        for name, path in (("owner", Path(row["lock_path"])),
                           ("fence", Path(row["quarantine_path"]))):
            raw = case.original_group[path]
            assert base64.b64decode(row[f"original_{name}_base64"]) == raw
            assert row[f"original_{name}_sha256"] == hashlib.sha256(raw).hexdigest()
        assert not Path(row["quarantine_path"]).exists()
    assert "session_id" not in json.loads(next(iter(case.physical_locks.values())).lock_path.read_bytes())
    assert receipt["terminal_journal_entries"] == [{
        "path": str(case.journal_path),
        "original_sha256": hashlib.sha256(case.journal_entries[case.journal_path]).hexdigest(),
        "original_base64": base64.b64encode(case.journal_entries[case.journal_path]).decode("ascii")}]
    assert not case.journal_path.exists()
    assert all(path.read_bytes() == raw for path, raw in case.preserved.items())


def test_final_cleanup_receipt_is_accepted_by_existing_normal_owner_guard(final_case):
    case = final_case
    _commit(case)
    for physical in (False, True):
        lock = _normal_lock(case, physical=physical)
        lock.acquire()
        lock.release()


def test_identical_final_cleanup_is_idempotent_without_rewriting_history(final_case):
    case = final_case
    first = _commit(case)
    paths = [case.receipt_path, case.manifest_path, *[lock.lock_path for lock in case.locks],
             *case.preserved]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert _commit(case) == first
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == snapshot
               for path, snapshot in before.items())
    assert not case.journal_path.exists()


@pytest.mark.parametrize("fault", ["different_original_bytes", "journal_returned", "new_journal"])
def test_final_cleanup_replay_rejects_changed_terminal_journal_group(final_case, fault):
    case = final_case
    _commit(case)
    paths = [case.receipt_path, case.manifest_path, *[lock.lock_path for lock in case.locks]]
    before = {path: path.read_bytes() for path in paths}
    if fault == "different_original_bytes":
        case.journal_entries[case.journal_path] += b" "
    elif fault == "journal_returned":
        case.journal_path.write_bytes(case.journal_entries[case.journal_path])
    else:
        (case.journal_path.parent / "foreign.remote-lease.json").write_bytes(b"{}")
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize("ground", [None, "verified_cleanup", "final_cleanup", "collector_cleanup"])
def test_unsupported_release_ground_cannot_mutate_original_group(final_case, ground):
    case = final_case
    case.receipt_data["release_ground"] = ground
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="release ground"):
        _commit(case)
    _assert_originals(case)
    assert not case.receipt_path.exists()


def test_partial_manifest_and_terminal_journals_do_not_enter_default_orphan_transition(final_case):
    case = final_case
    case.receipt_data.pop("release_ground")
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case, terminal_journal_entries=None)
    _assert_originals(case)


@pytest.mark.parametrize("fault", ["root_status", "technical_status", "root_finished", "updated_at",
                                  "quality", "stages", "session_status", "session_finished",
                                  "session_scientific", "session_marker", "foreign_session",
                                  "numeric_type"])
def test_final_cleanup_cannot_rewrite_any_existing_manifest_field(final_case, fault):
    case = final_case
    after = case.manifest_after
    if fault == "root_status":
        after["status"] = "cancelled"
    elif fault == "technical_status":
        after["technical_status"] = "completed"
    elif fault == "root_finished":
        after["finished_at"] = "today"
    elif fault == "updated_at":
        after["updated_at"] = "today"
    elif fault == "quality":
        after["quality_decision"] = "PASS"
    elif fault == "stages":
        after["models"]["fixture"]["stages"]["quality"]["value"] = 1.0
    elif fault == "session_status":
        after["execution_sessions"][-1]["status"] = "cancelled"
    elif fault == "session_finished":
        after["execution_sessions"][-1]["finished_at"] = "today"
    elif fault == "session_scientific":
        after["execution_sessions"][-1]["scientific_summary"]["quality"] = "PASS"
    elif fault == "session_marker":
        after["execution_sessions"][-1]["generic_orphan_recovery"] = {"recovery_id": "today"}
    elif fault == "numeric_type":
        after["schema_version"] = True
    else:
        after["execution_sessions"][0]["finished_at"] = "today"
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="complete original partial"):
        _commit(case)
    _assert_originals(case)


@pytest.mark.parametrize("fault", ["not_partial", "not_last", "duplicate_session", "foreign_current"])
def test_final_cleanup_requires_unique_current_last_partial_session(final_case, fault):
    case = final_case
    manifest = json.loads(case.manifest_before)
    if fault == "not_partial":
        manifest["execution_sessions"][-1]["status"] = "running"
    elif fault == "not_last":
        manifest["execution_sessions"].reverse()
    elif fault == "duplicate_session":
        manifest["execution_sessions"].insert(0, copy.deepcopy(manifest["execution_sessions"][-1]))
    else:
        manifest["current_session_id"] = "foreign-session"
    _write(case.manifest_path, manifest)
    case.manifest_before = case.manifest_path.read_bytes()
    case.manifest_after = copy.deepcopy(manifest)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    _assert_originals(case)


@pytest.mark.parametrize("fault", ["reason", "phase", "local", "remote", "session", "run",
                                  "outer_run", "missing", "physical_reason", "physical_session"])
def test_final_cleanup_requires_exact_original_run_and_physical_fences(final_case, fault):
    case = final_case
    lock = next(iter(case.physical_locks.values())) if fault.startswith("physical_") else case.run_lock
    path = lock.quarantine_path
    fence = json.loads(path.read_bytes())
    if fault == "missing":
        path.unlink()
    else:
        key, value = {
            "reason": ("reason", "source_collector_owned"),
            "phase": ("phase", "resume_prior_remote_recovery"),
            "local": ("local_process_quiescence_proven", False),
            "remote": ("remote_process_quiescence_proven", True),
            "session": ("session_id", "foreign-session"),
            "run": ("run_id", "foreign-run"),
            "outer_run": ("run_id", "foreign-run"),
            "physical_reason": ("reason", "previous_writer_terminated_uncleanly"),
            "physical_session": ("session_id", "foreign-session")}[fault]
        target = fence if fault == "outer_run" else fence["owner"]
        target[key] = value
        _write(path, fence)
    protected = {candidate: candidate.read_bytes() for candidate in case.original_group
                 if candidate.exists()}
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    assert all(candidate.read_bytes() == raw for candidate, raw in protected.items())
    assert case.manifest_path.read_bytes() == case.manifest_before
    assert case.journal_path.read_bytes() == case.journal_entries[case.journal_path]
    assert not case.receipt_path.exists()


def test_final_cleanup_cannot_release_a_second_physical_resource(final_case):
    case = final_case
    resource = "dut:192.0.2.102"
    lock = run_control.EvaluationRunLock.for_resource(resource, owner={"run_id": "foreign-active-run"})
    lock.acquire()
    try:
        before = lock.lock_path.read_bytes()
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="only the selected"):
            _commit(case, physical_locks={**case.physical_locks, resource: lock})
        assert lock.lock_path.read_bytes() == before
        _assert_originals(case)
    finally:
        lock.release()


@pytest.mark.parametrize("fault", ["foreign_session", "outside_journal", "changed_bytes",
                                  "omitted_descriptor", "extra_descriptor"])
def test_final_cleanup_terminal_journal_scope_and_original_bytes_are_exact(final_case, fault):
    case = final_case
    entries = dict(case.journal_entries)
    if fault in {"foreign_session", "outside_journal"}:
        path = (case.journal_path.parent.parent / "foreign-session/other.remote-lease.json"
                if fault == "foreign_session" else case.run / "other.remote-lease.json")
        _write(path, {})
        entries = {path: path.read_bytes()}
    elif fault == "changed_bytes":
        entries[case.journal_path] += b" "
    elif fault == "omitted_descriptor":
        entries = {}
    else:
        _write(case.journal_path.parent / "foreign.remote-lease.json", {})
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case, terminal_journal_entries=entries)
    _assert_originals(case)
    assert not case.receipt_path.exists()


@pytest.mark.parametrize("fault", ["second_owner", "manifest", "journal_unlink", "journal_fsync",
                                  "second_sidecar", "committed_receipt"])
def test_failed_final_cleanup_commit_restores_manifest_owners_fences_and_journals(final_case, monkeypatch, fault):
    case = final_case
    original_owner = run_control.EvaluationRunLock.write_recovery_owner
    original_write = run_control._generic_orphan_atomic_write
    original_unlink = Path.unlink
    original_sync = run_control._generic_orphan_sync_directory
    failed = False
    physical = next(iter(case.physical_locks.values()))

    def fail_owner(lock, owner):
        nonlocal failed
        if fault == "second_owner" and lock is physical and owner.get("state") == "released" and not failed:
            failed = True
            raise OSError("controlled second owner failure")
        return original_owner(lock, owner)

    def fail_write(path, data):
        nonlocal failed
        if not failed and ((fault == "manifest" and path == case.manifest_path)
                           or (fault == "committed_receipt" and path == case.receipt_path
                               and json.loads(data).get("state") == "committed")):
            failed = True
            raise OSError("controlled durable write failure")
        return original_write(path, data)

    def fail_unlink(path, *args, **kwargs):
        nonlocal failed
        if not failed and ((fault == "journal_unlink" and path == case.journal_path)
                           or (fault == "second_sidecar" and path == physical.quarantine_path)):
            failed = True
            raise OSError("controlled unlink failure")
        return original_unlink(path, *args, **kwargs)

    def fail_sync(path):
        nonlocal failed
        if fault == "journal_fsync" and path == case.journal_path.parent and not failed:
            failed = True
            raise OSError("controlled terminal journal directory fsync failure")
        return original_sync(path)

    monkeypatch.setattr(run_control.EvaluationRunLock, "write_recovery_owner", fail_owner)
    monkeypatch.setattr(run_control, "_generic_orphan_atomic_write", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(run_control, "_generic_orphan_sync_directory", fail_sync)
    with pytest.raises(OSError, match="controlled"):
        _commit(case)
    assert failed
    _assert_originals(case)
    assert json.loads(case.receipt_path.read_bytes())["state"] == "blocked"
    for selected in (False, True):
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            _normal_lock(case, physical=selected).acquire()


def test_failed_terminal_journal_restoration_keeps_original_fences_and_blocked_receipt(final_case, monkeypatch):
    case = final_case
    original_write = run_control._generic_orphan_atomic_write

    def fail_commit_and_journal_restore(path, data):
        if path == case.receipt_path and json.loads(data).get("state") == "committed":
            raise OSError("controlled final receipt failure")
        if path == case.journal_path:
            raise OSError("controlled terminal journal restoration failure")
        return original_write(path, data)

    monkeypatch.setattr(run_control, "_generic_orphan_atomic_write", fail_commit_and_journal_restore)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="rollback_errors=.*journal"):
        _commit(case)
    assert case.manifest_path.read_bytes() == case.manifest_before
    assert all(path.read_bytes() == raw for path, raw in case.original_group.items())
    assert not case.journal_path.exists()
    receipt = json.loads(case.receipt_path.read_bytes())
    assert receipt["state"] == "blocked"
    assert base64.b64decode(receipt["terminal_journal_entries"][0]["original_base64"]) == case.journal_entries[case.journal_path]
    for physical in (False, True):
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            _normal_lock(case, physical=physical).acquire()


@pytest.mark.parametrize("state", ["committing", "blocked", "corrupt"])
def test_final_cleanup_incomplete_receipt_keeps_normal_admission_closed(final_case, state):
    case = final_case
    _commit(case)
    if state == "corrupt":
        case.receipt_path.write_bytes(b"{incomplete")
    else:
        receipt = json.loads(case.receipt_path.read_bytes())
        receipt["state"] = state
        _write(case.receipt_path, receipt)
    for physical in (False, True):
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            _normal_lock(case, physical=physical).acquire()


@pytest.mark.skipif(os.name != "posix", reason="bounded child crash requires fork")
def test_crash_after_terminal_journal_removal_keeps_both_group_members_closed(final_case, monkeypatch):
    case = final_case
    original_sync = run_control._generic_orphan_sync_directory

    def crash_after_journal_sync(path):
        original_sync(path)
        if path == case.journal_path.parent:
            os._exit(77)

    child = os.fork()
    if child == 0:
        try:
            monkeypatch.setattr(run_control, "_generic_orphan_sync_directory", crash_after_journal_sync)
            _commit(case)
        except BaseException:
            os._exit(78)
        os._exit(79)
    waited, status = os.waitpid(child, 0)
    assert waited == child and os.waitstatus_to_exitcode(status) == 77
    assert not case.journal_path.exists()
    receipt = json.loads(case.receipt_path.read_bytes())
    assert receipt["state"] == "committing"
    assert base64.b64decode(receipt["terminal_journal_entries"][0]["original_base64"]) == case.journal_entries[case.journal_path]
    for physical in (False, True):
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            _normal_lock(case, physical=physical).acquire()
