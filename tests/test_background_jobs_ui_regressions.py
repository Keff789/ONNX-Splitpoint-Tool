from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_APP = ROOT / "onnx_splitpoint_tool" / "gui" / "app.py"
LEGACY_GUI = ROOT / "onnx_splitpoint_tool" / "gui_app.py"
SERVICES = ROOT / "onnx_splitpoint_tool" / "benchmark" / "services.py"
PANEL_JOBS = ROOT / "onnx_splitpoint_tool" / "gui" / "panels" / "panel_jobs.py"
TEXT_PROGRESS = ROOT / "onnx_splitpoint_tool" / "gui" / "widgets" / "text_progress_dialog.py"
GUI_CLASS_NAME = "SplitPointAnalyserGUI"
TEXT_PROGRESS_CLASS_NAME = "TextProgressDialog"



def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")



def _find_method(path: Path, method_name: str, *, class_name: str = GUI_CLASS_NAME) -> ast.FunctionDef:
    src = _read(path)
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"method {method_name!r} not found in {path}")



def _method_source(path: Path, method_name: str, *, class_name: str = GUI_CLASS_NAME) -> str:
    src = _read(path)
    node = _find_method(path, method_name, class_name=class_name)
    segment = ast.get_source_segment(src, node)
    if not segment:
        raise AssertionError(f"could not recover source for {method_name!r}")
    return segment



def test_notebook_shell_exposes_jobs_tab_and_bottom_status_bar() -> None:
    src = _read(NOTEBOOK_APP)
    assert '("jobs", "Jobs")' in src
    init_src = _method_source(NOTEBOOK_APP, "_init_central_notebook")
    assert '"jobs": panel_jobs.build_panel(self.main_notebook, app=self)' in init_src
    assert 'self.jobs_status_bar = ttk.Frame(self)' in init_src
    assert 'self.job_bar_generate = tk.Label(' in init_src
    assert 'self.job_bar_remote = tk.Label(' in init_src
    assert 'self._jobs_refresh_views()' in init_src



def test_jobs_panel_exposes_monitor_log_output_cancel_and_dismiss_actions() -> None:
    src = _read(PANEL_JOBS)
    assert 'text="Open monitor"' in src
    assert 'text="Open log"' in src
    assert 'text="Open output"' in src
    assert 'text="Cancel"' in src
    assert 'text="Dismiss"' in src
    assert 'Tip: double-click a job row to reopen its progress window.' in src
    assert 'app.jobs_tree = tree' in src
    assert 'app.btn_jobs_cancel = btn_cancel' in src



def test_text_progress_close_no_longer_triggers_cancel_callback() -> None:
    src = _method_source(TEXT_PROGRESS, "_close", class_name=TEXT_PROGRESS_CLASS_NAME)
    assert "self._on_cancel" not in src
    assert "self._alive = False" in src
    assert "self._on_close" in src



def test_jobs_helpers_track_monitor_reopen_paths_and_selection_actions() -> None:
    register_src = _method_source(NOTEBOOK_APP, "_jobs_register")
    assert "show_monitor: bool = True" in register_src
    assert "self._jobs_open_monitor(record.job_id)" in register_src

    open_monitor_src = _method_source(NOTEBOOK_APP, "_jobs_open_monitor")
    assert "existing.window.deiconify()" in open_monitor_src
    assert "TextProgressDialog(" in open_monitor_src
    assert "on_close=(lambda _job_id=record.job_id: self._jobs_on_monitor_closed(_job_id))" in open_monitor_src

    dismiss_src = _method_source(NOTEBOOK_APP, "_jobs_dismiss_selected")
    assert "Close the monitor window instead." in dismiss_src



def test_generate_job_wires_soft_cancel_into_execution_and_orchestration_services() -> None:
    src = _method_source(LEGACY_GUI, "_generate_benchmark_set")
    assert "cancel_event = threading.Event()" in src
    assert "cancel_callback=lambda: cancel_event.set()" in src
    assert "should_cancel=lambda: bool(cancel_event.is_set())" in src
    assert 'q.put(("cancelled", final_msg))' in src



def test_benchmark_generation_services_have_explicit_cancel_support() -> None:
    src = _read(SERVICES)
    assert "class BenchmarkGenerationCancelled(RuntimeError):" in src
    assert "should_cancel: Optional[Callable[[], bool]] = None" in src
    assert "raise BenchmarkGenerationCancelled(" in src
    assert "final_status = 'cancelled' if cancellation_reason" in src
