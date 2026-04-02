from __future__ import annotations

import ast
from pathlib import Path


SRC_PATH = Path("onnx_splitpoint_tool/gui_app.py")



def _method_source(class_name: str, method_name: str) -> str:
    src = SRC_PATH.read_text(encoding="utf-8")
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return ast.get_source_segment(src, item) or ""
    raise AssertionError(f"method {class_name}.{method_name} not found")



def test_selection_helpers_use_named_boundary_column_after_hailo_column_addition() -> None:
    src = _method_source("SplitPointAnalyserGUI", "_selected_boundary_index")
    assert "_tree_item_value_by_column(item, \"boundary\")" in src
    assert "vals[2]" not in src



def test_plot_click_selection_uses_named_boundary_column_after_hailo_column_addition() -> None:
    src = _method_source("SplitPointAnalyserGUI", "_on_plot_click_select_candidate")
    assert "_tree_item_value_by_column(item, \"boundary\")" in src
    assert "vals[2]" not in src



def test_tree_value_helper_exists_for_future_column_order_changes() -> None:
    src = _method_source("SplitPointAnalyserGUI", "_tree_item_value_by_column")
    assert "column_name" in src
    assert "columns.index(column_name)" in src
