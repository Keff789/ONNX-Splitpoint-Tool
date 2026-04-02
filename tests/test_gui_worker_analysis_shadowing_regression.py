import ast
from pathlib import Path


def _function_local_stores(src: str, outer_name: str, inner_name: str) -> set[str]:
    mod = ast.parse(src)
    outer = None
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == outer_name:
            outer = node
            break
    assert outer is not None, f"outer function {outer_name!r} not found"
    inner = None
    for node in outer.body:
        if isinstance(node, ast.FunctionDef) and node.name == inner_name:
            inner = node
            break
    assert inner is not None, f"inner function {inner_name!r} not found inside {outer_name!r}"
    stores: set[str] = set()
    for node in ast.walk(inner):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores.add(node.id)
    return stores


def test_generate_benchmark_worker_does_not_shadow_analysis_variable() -> None:
    src = Path("onnx_splitpoint_tool/gui_app.py").read_text(encoding="utf-8")
    stores = _function_local_stores(src, "_generate_benchmark_set", "worker")
    assert "a" not in stores



def test_split_selected_worker_does_not_shadow_analysis_variable() -> None:
    src = Path("onnx_splitpoint_tool/gui_app.py").read_text(encoding="utf-8")
    stores = _function_local_stores(src, "_split_selected_boundary", "worker")
    assert "a" not in stores


def _function_nonlocals(src: str, outer_name: str, inner_name: str) -> set[str]:
    mod = ast.parse(src)
    outer = None
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == outer_name:
            outer = node
            break
    assert outer is not None, f"outer function {outer_name!r} not found"
    inner = None
    for node in outer.body:
        if isinstance(node, ast.FunctionDef) and node.name == inner_name:
            inner = node
            break
    assert inner is not None, f"inner function {inner_name!r} not found inside {outer_name!r}"
    names: set[str] = set()
    for node in inner.body:
        if isinstance(node, ast.Nonlocal):
            names.update(node.names)
    return names


def test_generate_benchmark_worker_declares_nonlocal_shortlist_state() -> None:
    src = Path("onnx_splitpoint_tool/gui_app.py").read_text(encoding="utf-8")
    names = _function_nonlocals(src, "_generate_benchmark_set", "worker")
    assert "ranked_candidates" in names
    assert "candidate_search_pool" in names
