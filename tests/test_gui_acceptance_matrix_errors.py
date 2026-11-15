"""Keep original matrix worker failures visible after deferred Tk dispatch.

The ordinary worker runs in full, with a controlled setup-configuration error
before any service or hardware boundary. Tk callbacks wait until it exits.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("show_result_dialogs", [True, False])
def test_matrix_worker_error_survives_deferred_tk_callbacks(
    monkeypatch, tmp_path, show_result_dialogs
):
    from onnx_splitpoint_tool.gui import app as gui

    cause = "owned setup configuration is unavailable"
    pending = []
    registrations = []
    active = []
    finished = []
    dialogs = []
    events = []

    def after(delay, callback):
        assert delay == 0
        pending.append(callback)

    def unavailable_payload(setup_id):
        assert setup_id == "owned_offline_setup"
        events.append("configuration_failed")
        raise RuntimeError(cause)

    def forbidden(*args, **kwargs):
        pytest.fail("Configuration failure must precede service/host/hardware work")

    class InlineThread:
        def __init__(self, *, target, name, daemon):
            self.target = target

        def start(self):
            events.append("worker_started")
            self.target()
            events.append("worker_exited")

    monkeypatch.setattr(gui.threading, "Thread", InlineThread)
    monkeypatch.setattr(gui, "RemoteBenchmarkService", forbidden)
    monkeypatch.setattr(
        gui.messagebox, "showerror", lambda *args: dialogs.append(args)
    )
    # Keep worker configuration deterministic without reading user preferences.
    monkeypatch.setenv("ONNX_SPLITPOINT_PARALLEL_REMOTE_SETUPS", "0")
    monkeypatch.setenv("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS", "1")
    monkeypatch.setenv("ONNX_SPLITPOINT_POWER_CALC_WORKERS", "1")

    app = SimpleNamespace(
        root=SimpleNamespace(after=after),
        _remote_result_group_name=lambda run_id: "owned_offline_group",
        _jobs_register=lambda **kwargs: registrations.append(kwargs),
        _jobs_append_log=lambda *args: None,
        _set_background_job_active=lambda *args: active.append(args),
        _hardware_setup_remote_payload=unavailable_payload,
        _hardware_setup_host_config=forbidden,
        _jobs_finish=lambda *args, **kwargs: finished.append((args, kwargs)),
    )

    job_id = gui.SplitPointAnalyserGUI._start_remote_benchmark_matrix_job(
        app,
        bench_path=tmp_path / "owned suite" / "benchmark_set.json",
        local_working_dir=tmp_path / "owned results",
        suite_name="Owned offline suite",
        run_id="owned_test_run",
        dispatches=[
            {"setup_id": "owned_offline_setup", "run_id": "ort_cpu_full", "status": "ready"}
        ],
        base_args=SimpleNamespace(),
        show_result_dialogs=show_result_dialogs,
    )

    assert job_id == "remote-matrix-owned_test_run"
    assert registrations[0]["job_id"] == job_id
    assert events == ["worker_started", "configuration_failed", "worker_exited"]
    assert active == [("remote_run", True), ("remote_run", False)]
    assert not finished and not dialogs, "Tk callbacks must remain queued until worker exit"

    # Exception targets have been cleared by Python here. The original broken
    # callbacks raise NameError when this queue is dispatched.
    while pending:
        pending.pop(0)()

    assert finished == [((job_id,), {"status": "error", "message": f"RuntimeError: {cause}"})]
    if show_result_dialogs:
        assert dialogs == [("Remote benchmark", f"Hardware matrix run failed:\n\nRuntimeError: {cause}")]
    else:
        assert dialogs == []
    assert not (tmp_path / "owned results").exists()
