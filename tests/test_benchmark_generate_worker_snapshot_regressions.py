from __future__ import annotations

import ast
from pathlib import Path


GUI_APP_PATH = Path("onnx_splitpoint_tool/gui_app.py")
GUI_CLASS_NAME = "SplitPointAnalyserGUI"


def _find_gui_class(src: str) -> ast.ClassDef:
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.ClassDef) and node.name == GUI_CLASS_NAME:
            return node
    raise AssertionError(f"class {GUI_CLASS_NAME!r} not found")


def _find_gui_method(src: str, name: str) -> ast.FunctionDef:
    cls = _find_gui_class(src)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found on {GUI_CLASS_NAME!r}")


def _find_inner_function(src: str, outer_name: str, inner_name: str) -> ast.FunctionDef:
    outer = _find_gui_method(src, outer_name)
    for node in outer.body:
        if isinstance(node, ast.FunctionDef) and node.name == inner_name:
            return node
    raise AssertionError(f"inner function {inner_name!r} not found inside {outer_name!r}")


def _self_attribute_assignments(func: ast.FunctionDef) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for maybe_target in node.targets:
                if isinstance(maybe_target, ast.Attribute) and isinstance(maybe_target.value, ast.Name) and maybe_target.value.id == "self":
                    names.add(maybe_target.attr)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                names.add(target.attr)
    return names


def test_generate_benchmark_worker_avoids_direct_self_access_from_background_thread() -> None:
    src = GUI_APP_PATH.read_text(encoding="utf-8")
    worker = _find_inner_function(src, "_generate_benchmark_set", "worker")
    self_attrs = {
        node.attr
        for node in ast.walk(worker)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    assert not self_attrs


def test_candidate_row_metadata_alias_is_initialized_and_maintained() -> None:
    src = GUI_APP_PATH.read_text(encoding="utf-8")
    init_assigns = _self_attribute_assignments(_find_gui_method(src, "__init__"))
    update_assigns = _self_attribute_assignments(_find_gui_method(src, "_update_table"))
    clear_assigns = _self_attribute_assignments(_find_gui_method(src, "_clear_results"))

    assert "candidates" in init_assigns
    assert "candidates" in update_assigns
    assert "candidates" in clear_assigns
