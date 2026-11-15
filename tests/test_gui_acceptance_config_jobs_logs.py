"""Focused GUI regressions; external hardware calls are replaced explicitly."""

from __future__ import annotations

import copy
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace

import pytest


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _button(widget, text):
    return next(w for w in _walk(widget) if isinstance(w, ttk.Button) and w.cget("text") == text)


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    yield window
    window.destroy()


def _logs_panel(root, monkeypatch, path):
    from onnx_splitpoint_tool.gui.panels import panel_logs

    monkeypatch.setattr(panel_logs, "_discover_logs", lambda: [path])
    monkeypatch.setattr(panel_logs, "resolve_active_log_path", lambda: path)
    monkeypatch.setattr(panel_logs.messagebox, "showinfo", lambda *a, **kw: None)
    panel = panel_logs.build_panel(root)
    panel.pack(fill="both", expand=True)
    root.update_idletasks()
    return panel_logs, panel


def test_copy_last_error_keeps_traceback_root_cause(root, monkeypatch, tmp_path):
    log = tmp_path / "own log.log"
    log.write_text(
        "2026-11-15 12:00:00 ERROR worker failed\n"
        "Traceback (most recent call last):\n"
        '  File "worker.py", line 4, in run\n'
        "    subprocess.run(command, timeout=2)\n"
        "subprocess.TimeoutExpired: Command 'own job' timed out after 2 seconds\n"
        "2026-11-15 12:00:01 INFO GUI remains responsive\n",
        encoding="utf-8",
    )
    _, panel = _logs_panel(root, monkeypatch, log)
    _button(panel, "Copy last error").invoke()
    copied = root.clipboard_get()
    assert "subprocess.TimeoutExpired" in copied
    assert "timed out after 2 seconds" in copied
    assert 'File "worker.py"' in copied
    assert "GUI remains responsive" not in copied


def test_log_export_preserves_complete_bytes_and_cancel(root, monkeypatch, tmp_path):
    log = tmp_path / "own log.log"
    raw = b"first\n\x00\xfflast cause\n"
    log.write_bytes(raw)
    module, panel = _logs_panel(root, monkeypatch, log)
    destination = tmp_path / "folder with spaces" / "exported log.log"
    destination.parent.mkdir()
    monkeypatch.setattr(module.filedialog, "asksaveasfilename", lambda **kw: str(destination))
    _button(panel, "Export…").invoke()
    assert destination.read_bytes() == raw
    monkeypatch.setattr(module.filedialog, "asksaveasfilename", lambda **kw: "")
    _button(panel, "Export…").invoke()
    assert log.read_bytes() == raw
    assert destination.read_bytes() == raw


@pytest.fixture
def remote_card(root, monkeypatch, tmp_path):
    from onnx_splitpoint_tool.gui.panels import panel_hardware
    from onnx_splitpoint_tool.energy import config

    monkeypatch.setenv("ONNX_SPLITPOINT_RUN_MODES_FILE", str(tmp_path / "run modes.yaml"))
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setattr(panel_hardware, "_schedule_initial_platform_power_refresh", lambda *a: None)
    monkeypatch.setattr(config, "default_registry_path", lambda: tmp_path / "hardware.yaml")
    registry = {"hardware_setups": [{"id": "orin_nx_hailo8_01", "host": {"address": "old.invalid"}}]}
    calls = []
    app = SimpleNamespace(
        root=root, after=root.after, accel_specs={},
        _hardware_registry_load=lambda: copy.deepcopy(registry),
        _hardware_registry_save=lambda data: calls.append(("save", data)),
        _hardware_setup_remote_payload=lambda sid: {"host": "old.invalid", "port": 22},
        _test_hardware_setup_async=lambda sid, **kw: calls.append(("ssh", sid)),
    )
    panel = panel_hardware.build_panel(root, app=app)
    panel.pack(fill="both", expand=True)
    host_variable = app.var_hwsetup_orin_nx_hailo8_01_host
    entry = next(w for w in _walk(panel) if isinstance(w, ttk.Entry) and w.cget("textvariable") == str(host_variable))
    card = entry.master.master
    return panel_hardware, app, card, calls


@pytest.mark.parametrize("action", ["Test", "Test energy"])
def test_remote_test_stops_when_setup_save_fails(remote_card, monkeypatch, action):
    module, app, card, calls = remote_card
    app.var_hwsetup_orin_nx_hailo8_01_port.set("not a port")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: calls.append(("energy", a)) or SimpleNamespace(returncode=1, stdout="offline", stderr=""))
    monkeypatch.setattr(module.messagebox, "showwarning", lambda *a, **kw: None)
    monkeypatch.setattr(module.messagebox, "showerror", lambda *a, **kw: None)
    _button(card, action).invoke()
    assert app.var_hwsetup_orin_nx_hailo8_01_status.get().startswith("save failed:")
    assert calls == [], "Failed config save must not test stale remote/energy configuration"


@pytest.mark.parametrize("port", ["0", "-1", "65536"])
def test_remote_setup_save_rejects_invalid_tcp_port(remote_card, port):
    _, app, card, calls = remote_card
    app.var_hwsetup_orin_nx_hailo8_01_port.set(port)
    _button(card, "Save").invoke()
    assert app.var_hwsetup_orin_nx_hailo8_01_status.get().startswith("save failed:")
    assert calls == []


def test_jobs_cancel_repeated_request_is_idempotent():
    from onnx_splitpoint_tool.gui.app import BackgroundJobRecord, SplitPointAnalyserGUI

    calls = []
    record = BackgroundJobRecord(
        job_id="own-job", kind="generate", type_label="Local test",
        title="Own job", name="own job", can_cancel=True,
        cancel_callback=lambda: calls.append("cancel") or True,
    )
    owner = SimpleNamespace(_background_jobs={record.job_id: record},
                            _jobs_refresh_views=lambda: None,
                            _jobs_status_label=lambda status: SplitPointAnalyserGUI._JOB_STATUS_LABELS.get(status, status))
    SplitPointAnalyserGUI._jobs_request_cancel(owner, record.job_id)
    SplitPointAnalyserGUI._jobs_request_cancel(owner, record.job_id)
    assert calls == ["cancel"]
    assert record.status == "cancelling"
    assert not record.can_cancel


@pytest.mark.parametrize("hook", ["tools", "setup"])
def test_deferred_energy_error_preserves_cause(root, monkeypatch, tmp_path, hook):
    from onnx_splitpoint_tool.gui import app as app_module
    from onnx_splitpoint_tool.energy import collector, config

    received = []
    callback_errors = []
    root.report_callback_exception = lambda *args: callback_errors.append(args)
    monkeypatch.setattr(app_module, "show_diagnostic_dialog", lambda *a, **kw: received.append(kw))
    monkeypatch.setattr(config, "load_hardware_registry", lambda *a, **kw: {})
    monkeypatch.setattr(config, "energy_defaults_from_registry", lambda *a, **kw: object())
    monkeypatch.setattr(config, "energy_setup_from_registry", lambda *a, **kw: SimpleNamespace(urecs_address="192.0.2.10"))
    monkeypatch.setattr(config, "energy_measurements_root", lambda *a, **kw: tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("controlled original collector cause")

    monkeypatch.setattr(collector, "check_energy_tools", fail)
    monkeypatch.setattr(collector, "test_fast_firmware_sleep", fail)
    owner = SimpleNamespace(root=root, _hardware_setups_path=lambda: tmp_path / "hardware.yaml",
                            _hardware_setup_id_from_display=lambda sid: sid,
                            default_output_dir=str(tmp_path))

    def poll():
        if received or callback_errors:
            root.quit()
        else:
            root.after(10, poll)

    if hook == "tools":
        app_module.SplitPointAnalyserGUI._energy_test_tools_async(owner)
    else:
        app_module.SplitPointAnalyserGUI._energy_test_setup_async(owner, "own-offline-setup")
    root.after(10, poll)
    deadline = root.after(3000, root.quit)
    root.mainloop()
    root.after_cancel(deadline)
    assert callback_errors == []
    assert len(received) == 1
    assert "controlled original collector cause" in received[0]["summary"]
