from __future__ import annotations

import ast
from pathlib import Path


GUI_APP = Path("onnx_splitpoint_tool/gui_app.py")
CONTROLLER = Path("onnx_splitpoint_tool/gui/benchmark_workflow.py")
GUI_CLASS_NAME = "SplitPointAnalyserGUI"


def _method_source(path: Path, method_name: str, *, class_name: str = GUI_CLASS_NAME) -> str:
    src = path.read_text(encoding="utf-8")
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    segment = ast.get_source_segment(src, child)
                    if not segment:
                        raise AssertionError(f"could not recover source for {method_name!r}")
                    return segment
    raise AssertionError(f"method {method_name!r} not found in {path}")


def test_legacy_gui_imports_extracted_benchmark_workflow_controller() -> None:
    src = GUI_APP.read_text(encoding="utf-8")
    assert "from .gui.benchmark_workflow import BenchmarkWorkflowController" in src
    assert "def _get_benchmark_workflow_controller(self) -> BenchmarkWorkflowController:" in src



def test_generate_benchmark_set_is_now_a_thin_delegate() -> None:
    src = _method_source(GUI_APP, "_generate_benchmark_set")
    assert "Thin compatibility wrapper around the extracted benchmark workflow controller." in src
    assert "return self._get_benchmark_workflow_controller().generate_benchmark_set(" in src



def test_extracted_controller_contains_benchmark_generation_implementation() -> None:
    src = CONTROLLER.read_text(encoding="utf-8")
    assert "class BenchmarkWorkflowController:" in src
    assert "def generate_benchmark_set(" in src
    assert "app = self.app" in src
    assert "app._jobs_register(" in src
    assert 'app._set_background_job_active("generate", True)' in src


def test_extracted_controller_does_not_use_invalid_nested_gui_import() -> None:
    src = CONTROLLER.read_text(encoding="utf-8")
    assert "from .gui.controller import write_benchmark_suite_script" not in src
    assert "return write_benchmark_suite_script(dst_dir, bench_json_name=bench_json_name)" in src


def test_extracted_controller_reads_full_hef_order_from_app_state() -> None:
    src = CONTROLLER.read_text(encoding="utf-8")
    assert "getattr(\n                app,\n                \"var_hailo_full_hef_order\"" in src


def test_extracted_controller_uses_app_owned_orchestration_service_cache() -> None:
    src = CONTROLLER.read_text(encoding="utf-8")
    assert "orchestration_service = getattr(\n            app,\n            '_benchmark_generation_orchestration_service'" in src


def test_extracted_controller_routes_refactor_sensitive_imports_through_resolver_helpers() -> None:
    src = CONTROLLER.read_text(encoding="utf-8")
    assert "from .hailo_backend import" not in src
    assert "onnx_splitpoint_tool.gui.hailo_backend" not in src
    assert "resolve_hailo_benchmark_helpers(" in src
    assert "resolve_tool_core_version()" in src
