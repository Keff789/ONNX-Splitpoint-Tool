from __future__ import annotations

import importlib
import json
from pathlib import Path
import queue
import threading
import time
from types import MethodType, SimpleNamespace

import pytest

from _v27930_terminal_lifecycle_fixture import (
    TerminalLifecycleRunner, closure, options_for, snapshot,
)
from onnx_splitpoint_tool.workflow import runner as runner_module
from onnx_splitpoint_tool.workflow.run_control import WorkflowRunCleanupQuarantineError


@pytest.fixture
def make_runner(tmp_path, monkeypatch):
    # Provenance hashing has separate tests.  The runner/lock/cleanup/job/stream
    # path under examination remains the production implementation.
    monkeypatch.setattr(runner_module, "package_build_snapshot", lambda: {"build_id": "synthetic-only"})
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.artifacts.package_build_snapshot",
                        lambda: {"build_id": "synthetic-only"})
    monkeypatch.setenv("ONNX_SPLITPOINT_CACHE_DIR", str(tmp_path / "cache"))

    def build(status="failed", **kwargs):
        runner = TerminalLifecycleRunner(options_for(tmp_path), **kwargs)
        runner.measurement_status = status
        return runner
    return build


@pytest.mark.parametrize("measurement_status", ["ok", "failed"])
def test_ux01_actual_run_finishes_after_verifiers_handlers_and_unlock(make_runner, monkeypatch, measurement_status):
    events = []
    logs = []
    progress = []
    runner = make_runner(measurement_status, progress=lambda *row: progress.append(row))
    def log(line):
        logs.append(line)
        if "measurement_phase_complete" in line:
            events.append("measurements")
        if "[workflow] finished " in line:
            assert runner._run_lock is None
            assert runner._previous_signal_handlers == {}
            assert runner._run_control_write_enabled is False
            assert runner._cancellation_watcher is None or not runner._cancellation_watcher.is_alive()
            events.append("finished")
    runner._external_log = log
    cleanup = runner._finalize_owned_processes
    def traced_cleanup():
        assert not any("[workflow] finished " in row for row in logs)
        events.append("cleanup")
        return cleanup()
    monkeypatch.setattr(runner, "_finalize_owned_processes", traced_cleanup)
    verify = runner._verify_terminal_artifact_index
    def traced_verify(**kwargs):
        assert runner._run_lock is not None
        outcome = verify(**kwargs)
        events.append("verify")
        return outcome
    monkeypatch.setattr(runner, "_verify_terminal_artifact_index", traced_verify)
    restore = runner._restore_signal_handlers
    def traced_restore():
        events.append("restore_handlers")
        return restore()
    monkeypatch.setattr(runner, "_restore_signal_handlers", traced_restore)
    release = runner_module.EvaluationRunLock.release
    def traced_release(lock):
        release(lock)
        events.append("unlock")
    monkeypatch.setattr(runner_module.EvaluationRunLock, "release", traced_release)
    result = runner.run()
    assert events == ["measurements", "cleanup", "verify", "verify", "restore_handlers", "unlock", "finished"]
    assert result.status == measurement_status
    assert result.ok is (measurement_status == "ok")
    assert result.completed is (measurement_status != "failed")
    assert closure(runner)["status"] == "pass"
    assert closure(runner)["workflow_status"] == measurement_status
    assert "finalization_status=pass" in logs[-1]
    assert sum("[workflow] finished " in row for row in logs) == 1
    assert all(done < total for done, total, label in progress if not label.startswith("finished:"))
    assert {"finalize_processes", "finalize_artifacts"} <= {row[2].split(" |", 1)[0] for row in progress}


def test_ux02_unknown_inventory_and_chunk_progress_throttled(make_runner, tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow.artifacts import sha256_file_uncached
    logs, progress = [], []
    runner = make_runner(log=logs.append, progress=lambda *row: progress.append(row))
    runner._terminal_sealing = True
    clock = [10.0]
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock[0])
    for n in range(100):
        runner._terminal_progress(phase="inventory", done=n, current_path="reports/" + "x" * 500)
    clock[0] += 1.0
    runner._terminal_progress(phase="inventory", done=100)
    assert len(logs) == 2
    assert "Dateien sammeln: bisher 100" in logs[-1]
    assert not any("%" in line or "ETA" in line for line in logs)
    assert max(map(len, logs)) < 450
    payload = tmp_path / "large.bin"
    payload.write_bytes(bytes(range(256)) * (8 * 1024))
    seen = [0]
    def on_chunk(size):
        seen[0] += size
        clock[0] += 0.26
        runner._terminal_progress(phase="verify_pending_index", done=0, total=1,
                                  bytes_done=seen[0], bytes_total=payload.stat().st_size,
                                  current_path="models/large.bin")
    sha256_file_uncached(payload, 64 * 1024, on_chunk=on_chunk)
    assert seen[0] == payload.stat().st_size
    assert 3 < len(logs) < 15
    assert "Nutzbytes:" in logs[-1] and "models/large.bin" in logs[-1]
    assert progress and all(done < total for done, total, _ in progress)


def test_ux03_late_log_progress_cancel_and_detach_leave_all_sealed_bytes(make_runner):
    logs = []
    runner = make_runner(log=logs.append)
    runner.run()
    frozen = snapshot(runner.run_dir)
    runner._emit_log("[workflow] late observer heartbeat")
    runner._terminal_progress(phase="late_observer", done=1, total=1)
    assert runner.request_cancel("late_gui") is False
    runner.request_detach("late_dialog_close")
    assert snapshot(runner.run_dir) == frozen
    assert "late observer heartbeat" in runner.parent_log_path.read_text()
    assert "[workflow] finished " not in runner.run_log_path.read_text()
    assert "finalize_artifacts start" in runner.run_log_path.read_text()
    assert not runner._cancel_event.is_set()


@pytest.mark.parametrize("registry_name", ["_process_registry", "_remote_process_registry"])
def test_ux04_cleanup_ownership_failure_skips_closure_preserves_quarantine(make_runner, monkeypatch, registry_name):
    logs = []
    runner = make_runner(log=logs.append)
    registry = getattr(runner, registry_name)
    method = "terminate_all" if registry_name == "_process_registry" else "cancel_all"
    def unresolved(**kwargs):
        raise OSError("synthetic ownership unavailable")
    monkeypatch.setattr(registry, method, unresolved)
    with pytest.raises(WorkflowRunCleanupQuarantineError) as error:
        runner.run()
    assert "quarantin" in str(error.value)
    assert runner._run_lock is None
    assert not (runner.run_dir / "reports/artifact_index_closure.json").exists()
    assert any("synthetic ownership unavailable" in body.decode(errors="replace")
               for name, body in snapshot(runner.run_dir).items() if "quarantine" in name)
    assert not any("[workflow] finished " in line for line in logs)
    assert "finalization_failed phase=finalize_processes" in logs[-1]
    frozen = snapshot(runner.run_dir)
    runner._emit_log("[workflow] observer after failed cleanup")
    assert snapshot(runner.run_dir) == frozen


def test_ux05_closure_exception_revokes_pass_without_rewriting_sealed_history(make_runner, monkeypatch):
    logs = []
    runner = make_runner("ok", log=logs.append)
    verify = runner._verify_terminal_artifact_index
    calls = []
    frozen = {}
    def failing_pass(**kwargs):
        calls.append(1)
        if len(calls) == 2:
            frozen.update({name: body for name, body in snapshot(runner.run_dir).items()
                           if name == "run_manifest.json" or name == "evaluation_workflow.log" or name.startswith("jobs/")})
            raise OSError("synthetic sha256 read failure")
        return verify(**kwargs)
    monkeypatch.setattr(runner, "_verify_terminal_artifact_index", failing_pass)
    with pytest.raises(RuntimeError, match="terminal_commit_verify_failed"):
        runner.run()
    assert closure(runner)["status"] == "fail"
    assert closure(runner)["workflow_status"] == "ok"
    assert all((runner.run_dir / name).read_bytes() == body for name, body in frozen.items())
    assert not any("[workflow] finished " in line for line in logs)
    assert "finalization_failed measurement_status=ok" in logs[-1]


class _QueuedRoot:
    def __init__(self):
        self.callbacks = queue.Queue()
        self.closed = False
    def after(self, delay, fn):
        if self.closed:
            raise RuntimeError("observer closed")
        self.callbacks.put(fn)
    def drain(self):
        while True:
            try:
                fn = self.callbacks.get_nowait()
            except queue.Empty:
                return
            fn()


class _Monitor:
    alive = True
    def __init__(self):
        self.finished = False
        self.lines = []
        self.cancel_state = "normal"
        self.btn_cancel = SimpleNamespace(configure=lambda **kw: setattr(self, "cancel_state", kw.get("state")))
    def append(self, line):
        self.lines.append(line)
    def set_absolute_progress(self, value, label):
        self.value, self.label = value, label
    def set_status(self, label):
        self.label = label
    def finish(self, **kwargs):
        self.finished = True


def _gui_fixture(tmp_path, monkeypatch, *, request, fail_closure=False, measurement_status="failed"):
    import matplotlib
    monkeypatch.setattr(matplotlib, "use", lambda *args, **kwargs: None)
    gui = importlib.import_module("onnx_splitpoint_tool.gui.app")
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(gui.messagebox, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(runner_module, "package_build_snapshot", lambda: {"build_id": "synthetic-only"})
    # environment_snapshot resolves this function in artifacts, independently
    # of the runner's imported name. Keep the intended provenance-only stub
    # complete: a growing shared hash cache must not consume the race deadline.
    monkeypatch.setattr("onnx_splitpoint_tool.workflow.artifacts.package_build_snapshot",
                        lambda: {"build_id": "synthetic-only"})
    root = _QueuedRoot()
    state = SimpleNamespace(value="", set=lambda value: setattr(state, "value", value))
    opts = options_for(tmp_path)
    app = SimpleNamespace(
        root=root, _background_jobs={}, _background_job_order=[], _gui_closing=False,
        _JOB_STATUS_LABELS=gui.SplitPointAnalyserGUI._JOB_STATUS_LABELS,
        var_eval_workflow_status=state,
        _jobs_refresh_views=lambda: None,
        _eval_workflow_snapshot_options=lambda **kw: opts,
        _eval_workflow_command_preview=lambda opts: "synthetic management fixture",
        _eval_workflow_text_set=lambda text: None,
        _eval_workflow_text_append=lambda text: None,
        _eval_workflow_render_result=lambda payload: None,
    )
    for name in ("_jobs_register", "_jobs_append_log", "_jobs_set_progress", "_jobs_finish", "_jobs_request_cancel",
                 "_jobs_status_label", "_jobs_workflow_status_to_gui", "_jobs_handle_workflow_job_event"):
        setattr(app, name, MethodType(getattr(gui.SplitPointAnalyserGUI, name), app))
    app._jobs_parse_iso_datetime = MethodType(gui.SplitPointAnalyserGUI._jobs_parse_iso_datetime, app)
    app._jobs_open_monitor = lambda job_id: setattr(app._background_jobs[job_id], "monitor", _Monitor())
    sealed, proceed = threading.Event(), threading.Event()
    made = []
    def create(options, **callbacks):
        runner = TerminalLifecycleRunner(options, **callbacks)
        runner.measurement_status = measurement_status
        made.append(runner)
        finalizer = runner._finalize_artifact_index
        def delayed_finalizer(**kwargs):
            sealed.set()
            assert proceed.wait(10), "test did not release terminal gate"
            return finalizer(**kwargs)
        runner._finalize_artifact_index = delayed_finalizer
        if fail_closure:
            verify = runner._verify_terminal_artifact_index
            calls = []
            def broken(**kwargs):
                calls.append(1)
                if len(calls) == 2:
                    raise OSError("synthetic final GUI integrity failure")
                return verify(**kwargs)
            runner._verify_terminal_artifact_index = broken
        return runner
    monkeypatch.setattr(gui, "EvaluationWorkflowRunner", create)
    job_id = gui.SplitPointAnalyserGUI._queue_evaluation_workflow(app)
    request.addfinalizer(lambda: (proceed.set(), app._background_jobs[job_id].worker_thread.join(10)))
    assert sealed.wait(10), "real runner did not reach terminal gate"
    return app, job_id, made[0], proceed


@pytest.mark.parametrize("fail_closure", [False, True])
def test_ux06_ux07_real_gui_callbacks_racing_cancel_and_late_events(tmp_path, monkeypatch, request, fail_closure):
    app, job_id, runner, proceed = _gui_fixture(tmp_path, monkeypatch, request=request, fail_closure=fail_closure)
    record = app._background_jobs[job_id]
    # A click queued before the seal is delivered before the queued progress
    # disable.  The real runner must reject it without a GUI cancellation flag.
    assert record.can_cancel is True
    app._jobs_request_cancel(job_id)
    assert record.status == "running" and not record.can_cancel
    assert not runner._cancel_event.is_set()
    app.root.drain()
    assert not record.monitor.finished
    assert record.progress_value < record.progress_maximum
    assert "finalize_artifacts" in record.status_text
    root_rows = [r for r in app._background_jobs.values() if r.type_label == "EvaluationWorkflowJob"]
    assert len(root_rows) == 1 and root_rows[0].status == "running"
    before = snapshot(runner.run_dir)
    runner.request_detach("closed_progress_window")
    assert snapshot(runner.run_dir) == before
    # Hundreds of chunk observers still enqueue only a bounded pair of latest
    # GUI updates.  No Tk widget is touched from the hash callback.
    for i in range(200):
        runner._terminal_progress(phase="verify_pending_index", done=i, total=200,
                                  bytes_done=i, bytes_total=200, force=True)
    assert app.root.callbacks.qsize() <= 2
    proceed.set()
    record.worker_thread.join(10)
    assert not record.worker_thread.is_alive()
    frozen = snapshot(runner.run_dir)
    app.root.drain()
    assert record.status == "error"  # measured failed or a true closure error
    assert record.monitor.finished
    assert record.status != "cancelled"
    assert root_rows[0].status == "error"
    assert snapshot(runner.run_dir) == frozen
    assert closure(runner)["status"] == ("fail" if fail_closure else "pass")
    if fail_closure:
        assert "terminal_commit_verify_failed" in record.last_message
        assert "Messphase beendet: failed" in record.last_message
    # A root event queued before return but serviced after it must not mark
    # the already completed live root as running again.
    app._jobs_handle_workflow_job_event({
        "job_id": runner.jobs.root_job_id, "job_type": "EvaluationWorkflowJob",
        "event": "failed", "status": "failed", "run_dir": str(runner.run_dir),
        "message": "measurement_phase_complete status=failed; finalization pending",
    }, workflow_scope=job_id)
    assert root_rows[0].status == "error"
    app._jobs_append_log(job_id, "late GUI-only heartbeat")
    app._jobs_finish(job_id, status="error", log_path=str(runner.run_log_path), message="late GUI-only finish")
    assert snapshot(runner.run_dir) == frozen


@pytest.mark.parametrize("fail_closure", [False, True])
def test_ux06_closed_observer_does_not_change_integrity_result(tmp_path, monkeypatch, request, fail_closure):
    app, job_id, runner, proceed = _gui_fixture(tmp_path, monkeypatch, request=request,
                                               measurement_status="ok", fail_closure=fail_closure)
    record = app._background_jobs[job_id]
    record.monitor.alive = False
    app._gui_closing = True
    app.root.closed = True
    proceed.set()
    record.worker_thread.join(10)
    assert not record.worker_thread.is_alive()
    assert closure(runner)["status"] == ("fail" if fail_closure else "pass")
    assert closure(runner)["workflow_status"] == "ok"
    parent_log = runner.parent_log_path.read_text()
    assert ("[workflow] finished status=ok" in parent_log) is (not fail_closure)
    if fail_closure:
        assert "finalization_failed measurement_status=ok" in parent_log


def test_ux07_preseal_cancel_preserves_cooperative_contract(make_runner):
    runner = make_runner()
    assert runner.request_cancel("before_start") is True
    assert runner._cancel_event.is_set()
    assert runner._stop_requested is True
    assert runner._control_state == "cancel_requested"
