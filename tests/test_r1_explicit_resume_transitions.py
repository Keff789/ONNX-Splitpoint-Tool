"""Bounded Resume entry tests; no accelerator or collector is contacted."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock

import pytest

import test_p01_runner_gui_guards as guards
from onnx_splitpoint_tool.workflow.run_control import (
    EvaluationRunLock, WorkflowRunCleanupQuarantineError,
    WorkflowRunLockedError, WorkflowRunTargetError, recoverable_cleanup_evidence,
)
from onnx_splitpoint_tool.workflow.run_discovery import inspect_evaluation_run
from onnx_splitpoint_tool.workflow import runner as runner_module


def _runner(root, *, resume=False):
    runner = guards.RunnerOwnershipGuardTests._runner(root)
    runner.options.resume = resume
    # This fixture has no model/start-snapshot artifacts. Real option/profile
    # contract admission and all lock/recovery entrypoints remain active.
    runner._validate_resume_profile_snapshot = lambda: None
    return runner


def _write_run(runner):
    guards.RunnerOwnershipGuardTests._write_current_resume_fixture(runner, runner.run_dir)
    path = runner.run_dir / "run_manifest.json"
    data = json.loads(path.read_text())
    data.update(status="running", options=runner.options.to_dict(),
                execution_sessions=[{"session_id": runner.session_id, "status": "running"}])
    path.write_text(json.dumps(data))
    (runner.run_dir / "finished_work.json").write_text('{"quality":"FAIL","attempt_count":2}\n')


def _write_remote_only_quarantine(runner):
    directory = runner.run_dir / "jobs" / "remote_process_leases" / runner.session_id
    directory.mkdir(parents=True, exist_ok=True)
    runner._write_cleanup_quarantine(
        phase="resume_prior_remote_recovery", local_proven=True, remote_proven=False,
        details={"reason": "prior_remote_lease_cleanup_unresolved", "invalid_entries": [],
                 "journals": [{"session_id": runner.session_id, "journal_dir": str(directory),
                               "resolved": False, "remaining": 1}]},
    )
    runner._commit_lock_cleanup_fence(
        phase="resume_prior_remote_recovery", local_proven=True, remote_proven=False,
        reason="prior_remote_lease_cleanup_unresolved",
    )


def _crash_controller(root, proof):
    root = Path(root).resolve()
    runner = _runner(root)
    def work():
        _write_run(runner)
        if proof:
            # Existing cleanup transition after local registry quiescence;
            # remote reachability is the only unresolved ownership question.
            runner._process_registry.assert_quiescent()
            _write_remote_only_quarantine(runner)
        (root / "ready").write_text("ready")
        while True:
            time.sleep(.05)
    runner._run_locked = work
    runner.run()


class _RemoteState:
    resolved = True
    calls = []

    def configure_journal(self, *, scope, journal_dir):
        self.scope = scope
        self.directory = journal_dir

    def cancel_all(self, *, grace_s):
        self.calls.append(self.directory)
        return [{"ok": self.resolved}]

    def active_count(self):
        return 0 if self.resolved else 1


def _start_controller(tmp_path, proof):
    child = subprocess.Popen(
        [sys.executable, "-B", __file__, "--controller", str(tmp_path), str(int(proof))],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 25
    while not (tmp_path / "ready").is_file():
        if child.poll() is not None:
            stdout, stderr = child.communicate()
            raise AssertionError((child.returncode, stdout, stderr))
        if time.monotonic() >= deadline:
            child.kill()
            stdout, stderr = child.communicate()
            raise AssertionError(("controller startup timeout", stdout, stderr))
        time.sleep(.03)
    return child


def test_explicit_recovery_of_owned_controller_crash_is_under_exclusive_lock(tmp_path):
    child = _start_controller(tmp_path, True)
    try:
        with pytest.raises(WorkflowRunLockedError):
            _runner(tmp_path, resume=True).run()
        child.kill()  # only this fixture's Popen, never a productive process
        child.communicate(timeout=5)
        run_dir = tmp_path / "fixture_run"
        finished = (run_dir / "finished_work.json").read_bytes()
        assert not inspect_evaluation_run(run_dir).resumable
        assert inspect_evaluation_run(run_dir, allow_cleanup_recovery=True).resumable
        runner = _runner(tmp_path, resume=True)
        _RemoteState.calls = []
        _RemoteState.resolved = True
        def resumed():
            assert runner._run_lock is not None
            with pytest.raises(WorkflowRunLockedError):
                EvaluationRunLock(out_root=tmp_path, run_dir=run_dir, owner={}).acquire()
            return SimpleNamespace(status="partial")
        runner._run_locked = resumed
        with mock.patch.object(runner_module, "RemoteProcessLeaseRegistry", _RemoteState):
            assert runner.run().status == "partial"
        assert runner.run_id == "fixture_run"
        assert _RemoteState.calls
        assert (run_dir / "finished_work.json").read_bytes() == finished
        assert not runner._cleanup_quarantine_path.exists()
        sessions = json.loads((run_dir / "run_manifest.json").read_text())["execution_sessions"]
        assert sessions[-1]["cleanup_recovery"]["remote_process_quiescence_proven"] is True
        assert sessions[-1]["status"] == "cancelled"
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


@pytest.mark.parametrize("proof", [False, True])
def test_hard_crash_without_proof_or_with_unreachable_remote_remains_blocked(tmp_path, proof):
    child = _start_controller(tmp_path, proof)
    try:
        child.kill()
        child.communicate(timeout=5)
        runner = _runner(tmp_path, resume=True)
        runner._run_locked = lambda: pytest.fail("unresolved writer executed work")
        _RemoteState.resolved = False
        with mock.patch.object(runner_module, "RemoteProcessLeaseRegistry", _RemoteState):
            with pytest.raises((WorkflowRunTargetError, WorkflowRunCleanupQuarantineError)) as blocked:
                runner.run()
        if not proof:
            assert "local-process cleanup proof" in str(blocked.value)
        else:
            # The same persisted cleanup can be retried after the remote state
            # is resolved, without clearing any marker by hand.
            _RemoteState.resolved = True
            continued = _runner(tmp_path, resume=True)
            continued._run_locked = lambda: SimpleNamespace(status="partial")
            with mock.patch.object(runner_module, "RemoteProcessLeaseRegistry", _RemoteState):
                assert continued.run().status == "partial"
    finally:
        _RemoteState.resolved = True
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


def test_regular_cancel_resumes_same_selected_run_and_preserves_finished_work(tmp_path):
    runner = _runner(tmp_path)
    def cancel():
        _write_run(runner)
        runner.request_cancel("fixture_user_cancel")
        path = runner.run_dir / "run_manifest.json"
        data = json.loads(path.read_text())
        data["status"] = "cancelled"
        data["execution_sessions"][-1]["status"] = "cancelled"
        path.write_text(json.dumps(data))
        return SimpleNamespace(status="cancelled")
    runner._run_locked = cancel
    assert runner.run().status == "cancelled"
    finished = (runner.run_dir / "finished_work.json").read_bytes()
    continued = _runner(tmp_path, resume=True)
    continued._run_locked = lambda: SimpleNamespace(status="partial")
    assert continued.run().status == "partial"
    assert continued.run_id == runner.run_id
    assert (runner.run_dir / "finished_work.json").read_bytes() == finished


def test_gui_resume_uses_selected_archived_options_not_live_profile(tmp_path):
    runner = _runner(tmp_path)
    runner.run_id = "fixture_run"
    runner.run_dir = tmp_path / runner.run_id
    _write_run(runner)
    runner.options.benchmark_runs = 17
    path = runner.run_dir / "run_manifest.json"
    data = json.loads(path.read_text())
    data["options"] = runner.options.to_dict()
    path.write_text(json.dumps(data))
    app_module = guards._import_gui_app_headless()
    app = SimpleNamespace(var_eval_workflow_last_run_dir=SimpleNamespace(get=lambda: str(runner.run_dir)))
    options = app_module.SplitPointAnalyserGUI._eval_workflow_snapshot_options(app, resume_override=True)
    assert options.resume is True
    assert options.run_id == "fixture_run"
    assert options.benchmark_runs == 17
    assert options.profile == runner.options.profile
    assert options.out == str(tmp_path)


@pytest.mark.parametrize("change", ["source_lock", "local_unproven", "other_session", "contract_drift"])
def test_resume_cleanup_does_not_clear_unsupported_fences_or_changed_request(tmp_path, change):
    child = _start_controller(tmp_path, True)
    try:
        child.kill()
        child.communicate(timeout=5)
        run_dir = tmp_path / "fixture_run"
        path = run_dir / "jobs" / "p01_unresolved_cleanup_quarantine.json"
        marker = json.loads(path.read_text())
        if change == "source_lock":
            marker["phase"] = "collector_source_stop"
        elif change == "local_unproven":
            marker["local_process_quiescence_proven"] = False
        elif change == "other_session":
            marker["session_id"] = "unrelated"
        path.write_text(json.dumps(marker))
        runner = _runner(tmp_path, resume=True)
        if change == "contract_drift":
            runner.options.benchmark_runs += 1
        runner._run_locked = lambda: pytest.fail("blocked request executed work")
        _RemoteState.calls = []
        with mock.patch.object(runner_module, "RemoteProcessLeaseRegistry", _RemoteState):
            with pytest.raises((WorkflowRunTargetError, WorkflowRunCleanupQuarantineError)):
                runner.run()
        assert path.read_bytes() == json.dumps(marker).encode()
        assert _RemoteState.calls == []
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=5)


if __name__ == "__main__" and sys.argv[1] == "--controller":
    _crash_controller(sys.argv[2], bool(int(sys.argv[3])))
