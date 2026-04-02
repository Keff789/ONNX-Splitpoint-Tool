from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_APP = ROOT / "onnx_splitpoint_tool" / "gui_app.py"
NOTEBOOK_APP = ROOT / "onnx_splitpoint_tool" / "gui" / "app.py"
PANEL_VALIDATE = ROOT / "onnx_splitpoint_tool" / "gui" / "panels" / "panel_validate.py"
GUI_CLASS_NAME = "SplitPointAnalyserGUI"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_method(path: Path, method_name: str) -> ast.FunctionDef:
    src = _read(path)
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.ClassDef) and node.name == GUI_CLASS_NAME:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child
    raise AssertionError(f"method {method_name!r} not found in {path}")


def _method_source(path: Path, method_name: str) -> str:
    src = _read(path)
    node = _find_method(path, method_name)
    segment = ast.get_source_segment(src, node)
    if not segment:
        raise AssertionError(f"could not recover source for {method_name!r}")
    return segment


def test_legacy_gui_initializes_parallel_background_job_flags() -> None:
    src = _method_source(GUI_APP, "__init__")
    assert "self._benchmark_generation_active: bool = False" in src
    assert "self._remote_benchmark_active: bool = False" in src
    assert "self._benchmark_generation_thread: Optional[threading.Thread] = None" in src
    assert "self._remote_benchmark_thread: Optional[threading.Thread] = None" in src



def test_background_job_helper_refreshes_ui_state_for_generate_and_remote_jobs() -> None:
    src = _method_source(GUI_APP, "_set_background_job_active")
    assert 'if kind == "generate":' in src
    assert 'elif kind in {"remote", "remote_run", "run"}:' in src
    assert "self._benchmark_generation_active = value" in src
    assert "self._remote_benchmark_active = value" in src
    assert "self._set_ui_state(self._infer_ui_state())" in src



def test_action_buttons_consult_parallel_background_job_flags() -> None:
    src = _method_source(GUI_APP, "_update_action_buttons")
    assert 'generation_active = bool(getattr(self, "_benchmark_generation_active", False))' in src
    assert 'remote_run_active = bool(getattr(self, "_remote_benchmark_active", False))' in src
    assert '_set_enabled("btn_benchmark", has_benchmark_candidates and not generation_active)' in src
    assert '_set_enabled("btn_resume_benchmark", has_analysis and not generation_active)' in src
    assert '_set_enabled("btn_remote_benchmark", not remote_run_active)' in src



def test_generate_benchmark_method_is_modeless_and_tracks_single_generation_job() -> None:
    src = _method_source(GUI_APP, "_generate_benchmark_set")
    assert "Benchmark set already running" in src
    assert 'self._set_background_job_active("generate", True)' in src
    assert 'self._set_background_job_active("generate", False)' in src
    assert "Runs in the background." in src
    assert "self._jobs_register(" in src
    assert "cancel_event = threading.Event()" in src
    assert "status in ('ok', 'warn', 'err', 'cancelled')" in src
    assert "self._jobs_finish(" in src
    assert "dlg.grab_set()" not in src
    assert 'self.configure(cursor="watch")' not in src
    assert "runtime = None" in src
    assert "if runtime is not None" in src



def test_remote_benchmark_method_tracks_single_run_job_and_finalizes_state() -> None:
    src = _method_source(NOTEBOOK_APP, "_remote_run_benchmark")
    assert "Remote benchmark already running" in src
    assert 'self._set_background_job_active("remote_run", True)' in src
    assert "self._jobs_register(" in src
    assert 'kind="remote_run"' in src
    assert "cancel_callback=lambda: cancel_event.set()" in src
    assert "self._finalize_remote_benchmark_job(job_id, _kind, _out)" in src

    finalize_src = _method_source(NOTEBOOK_APP, "_finalize_remote_benchmark_job")
    assert 'self._set_background_job_active("remote_run", False)' in finalize_src
    assert "self._jobs_finish(" in finalize_src
    assert "self._handle_remote_benchmark_result(final_kind, out)" in finalize_src
    result_src = _method_source(NOTEBOOK_APP, "_handle_remote_benchmark_result")
    assert 'elif final_kind == "cancelled":' in result_src



def test_validation_panel_exposes_resume_and_remote_run_buttons_for_state_updates() -> None:
    src = _read(PANEL_VALIDATE)
    assert "app.btn_resume_benchmark = btn_resume" in src
    assert "app.btn_remote_benchmark = btn_run" in src
