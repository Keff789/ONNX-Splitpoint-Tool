"""Regressions found while operating the normal Benchmark tab.

These use real Tk widgets. Interactive message boxes are observed separately;
no compiler, remote execution or model-preparation probe is permitted.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
import ast
import gc
from pathlib import Path
from types import SimpleNamespace

import pytest


def _descendants(parent):
    for widget in parent.winfo_children():
        yield widget
        yield from _descendants(widget)


def _enter_benchmark_tab(app):
    app.main_notebook.select(app.panel_frames["validate"])
    app.update()
    # Let the normal deferred suite discovery finish while Tk's real event
    # loop is running, before this short-lived test root is torn down.
    app.after(700, app.quit)
    app.mainloop()


def test_manual_preparation_remains_reachable_and_current_mode_does_not_probe(
    tmp_path, monkeypatch
):
    from onnx_splitpoint_tool.gui import app as gui
    from onnx_splitpoint_tool.settings import SettingsStore

    store = SettingsStore(home=tmp_path / "settings")
    store.save({"output_dir": str(tmp_path / "work"), "tk_vars": {}})
    monkeypatch.setattr(gui, "SettingsStore", lambda: store)
    monkeypatch.setenv("ONNX_SPLITPOINT_TOOL_ROOT", str(tmp_path))

    def forbidden(*args, **kwargs):
        raise AssertionError("Use current ONNX must never launch a preparation probe")

    monkeypatch.setattr(gui, "prepare_model_for_benchmark", forbidden)
    monkeypatch.setattr(gui, "snapshot_preparation_runtime", forbidden)
    messages = []
    try:
        app = gui.SplitPointAnalyserGUI()
    except tk.TclError as exc:
        pytest.skip(f"real Tk display unavailable: {exc}")
    try:
        # The legacy constructor installs its standard diagnostic messagebox
        # adapters; observe messages after those adapters have been installed.
        monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: messages.append(a))
        monkeypatch.setattr(gui.messagebox, "showwarning", lambda *a, **k: messages.append(a))
        _enter_benchmark_tab(app)
        controls = list(_descendants(app.panel_frames["validate"]))
        toggle = next(
            w for w in controls
            if isinstance(w, ttk.Button)
            and "Model preparation / Hailo diagnostics" in str(w.cget("text"))
        )
        # The section must survive legacy profile-control hiding. This check
        # fails on the original row-10..15 filter before the repair.
        assert toggle.winfo_viewable(), "Manual preparation section was hidden"
        toggle.invoke()
        app.update()
        prepare = next(
            w for w in _descendants(app.panel_frames["validate"])
            if isinstance(w, ttk.Button) and w.cget("text") == "Prepare current model…"
        )
        assert prepare.winfo_viewable()
        prepare.invoke()
        assert messages[-1] == ("Model preparation", "Please load a model first.")

        app.gui_state.current_model_path = str(tmp_path / "classification.onnx")
        app.model_path = app.gui_state.current_model_path
        app.var_bench_model_preparation_mode.set("Use current ONNX")
        before = set(app._background_jobs)
        prepare.invoke()
        assert "Nothing to prepare" in messages[-1][1]
        assert set(app._background_jobs) == before
        assert app.gui_state.current_model_path.endswith("classification.onnx")
    finally:
        app.destroy()
        del app
        gc.collect()


@pytest.mark.parametrize(
    ("dialog_name", "completion_name", "service_name"),
    [
        ("_open_imagenet_preset_import_dialog", "_finish_import", "build_imagenet_validation_preset"),
        ("_open_prepare_validation_sets_dialog", "_finish", "prepare_all_validation_assets"),
    ],
)
def test_deferred_dataset_worker_error_retains_original_exception(
    tmp_path, dialog_name, completion_name, service_name
):
    """Execute the actual nested worker, then its queued Tk callback later.

    The preparation service is a controlled offline failure; this verifies the
    asynchronous error handoff without downloading data or touching presets.
    """
    from onnx_splitpoint_tool.gui.panels import panel_validate

    tree = ast.parse(Path(panel_validate.__file__).read_text())
    dialog = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == dialog_name
    )
    worker = next(
        node for node in ast.walk(dialog)
        if isinstance(node, ast.FunctionDef) and node.name == "_worker"
    )
    callbacks, finished = [], []
    error = OSError("controlled missing validation input")

    def fail_service(**kwargs):
        raise error

    namespace = {
        service_name: fail_service,
        completion_name: lambda result, exc: finished.append((result, exc)),
        "dlg": SimpleNamespace(after=lambda delay, cb: callbacks.append(cb)),
        "preset": "imagenet_val_mini_200",
        "val_dir": tmp_path,
        "gt_file": tmp_path / "missing.txt",
        "copy_mode_var": SimpleNamespace(get=lambda: "copy"),
        "overwrite_var": SimpleNamespace(get=lambda: False),
        "selected_coco": False,
        "selected_i200": False,
        "selected_i500": False,
        "selected_test": True,
    }
    module = ast.fix_missing_locations(ast.Module(body=[worker], type_ignores=[]))
    exec(compile(module, str(panel_validate.__file__), "exec"), namespace)
    namespace["_worker"]()
    assert len(callbacks) == 1
    assert not finished
    callbacks.pop()()
    assert finished == [(None, error)]


def test_preparation_duplicate_click_is_guarded_and_cancel_allows_next_job(
    tmp_path, monkeypatch
):
    from onnx_splitpoint_tool.gui import app as gui
    from onnx_splitpoint_tool.settings import SettingsStore

    store = SettingsStore(home=tmp_path / "settings")
    store.save({"output_dir": str(tmp_path / "work"), "tk_vars": {}})
    monkeypatch.setattr(gui, "SettingsStore", lambda: store)
    monkeypatch.setenv("ONNX_SPLITPOINT_TOOL_ROOT", str(tmp_path))
    try:
        app = gui.SplitPointAnalyserGUI()
    except tk.TclError as exc:
        pytest.skip(f"real Tk display unavailable: {exc}")
    try:
        messages, workers = [], []
        monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: messages.append(a))
        monkeypatch.setattr(gui.messagebox, "showwarning", lambda *a, **k: messages.append(a))
        monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: messages.append(a))
        # Defer only this preparation worker. Its real job registration,
        # duplicate guard, cancellation and finalization remain in use.
        real_thread = gui.threading.Thread

        class ControlledThread(real_thread):
            def start(self):
                target = self._target
                if target is not None and "_queue_prepare_current_model" in target.__qualname__:
                    workers.append(target)
                else:
                    super().start()

        def controlled_preparation(*args, **kwargs):
            assert kwargs["cancel_event"].is_set()
            raise RuntimeError("controlled preparation cancelled before any probe")

        _enter_benchmark_tab(app)
        monkeypatch.setattr(gui.threading, "Thread", ControlledThread)
        monkeypatch.setattr(gui, "snapshot_preparation_runtime", lambda app: {})
        monkeypatch.setattr(gui, "prepare_model_for_benchmark", controlled_preparation)
        app.gui_state.current_model_path = str(tmp_path / "model.onnx")
        app.model_path = app.gui_state.current_model_path
        app.var_bench_model_preparation_mode.set("Auto-screen YOLO full-Hailo")
        prepare = next(
            w for w in _descendants(app.panel_frames["validate"])
            if isinstance(w, ttk.Button) and w.cget("text") == "Prepare current model…"
        )
        prepare.invoke()
        first = app._jobs_latest_record("prepare_model")
        assert first is not None and first.status == "running"
        prepare.invoke()
        assert len(workers) == 1
        assert app._jobs_latest_record("prepare_model") is first
        assert "already running" in messages[-1][1].lower()
        first.monitor.btn_cancel.invoke()
        assert first.status == "cancelling"
        workers[0]()
        app.update()
        assert first.status == "cancelled"
        assert "controlled preparation cancelled" in messages[-1][1]
        prepare.invoke()
        assert len(workers) == 2
        assert app._jobs_latest_record("prepare_model").status == "running"
    finally:
        workers.clear()
        app.destroy()
        del app
        gc.collect()


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("2", 2), ("bad", 2)])
def test_benchmark_context_hops_snapshot_keeps_zero(raw, expected):
    from onnx_splitpoint_tool.gui import benchmark_workflow as workflow

    tree = ast.parse(Path(workflow.__file__).read_text())
    generation = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_benchmark_set"
    )
    index = next(
        i for i, node in enumerate(generation.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ctx_hops" for target in node.targets)
    )
    nodes = [generation.body[index]]
    if isinstance(generation.body[index + 1], ast.If):
        nodes.append(generation.body[index + 1])
    # Execute the actual GUI snapshot statements used by the later execution
    # config. The lightweight Tk variable substitute avoids needing a display
    # for this parser regression, without replacing the value conversion.
    namespace = {
        "_safe_int": workflow._safe_int,
        "app": SimpleNamespace(var_split_ctx_hops=SimpleNamespace(get=lambda: raw)),
        "tk": SimpleNamespace(StringVar=lambda **kwargs: SimpleNamespace(get=lambda: kwargs["value"])),
    }
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(workflow.__file__), "exec"), namespace)
    assert namespace["ctx_hops"] == expected
