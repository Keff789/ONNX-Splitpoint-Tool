from __future__ import annotations

import ast
from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService, normalize_full_hef_policy


GUI_APP_PATH = Path("onnx_splitpoint_tool/gui_app.py")


def _function_local_stores(src: str, outer_name: str) -> set[str]:
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == outer_name:
            stores: set[str] = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    stores.add(child.id)
            return stores
    raise AssertionError(f"function {outer_name!r} not found")


def test_normalize_full_hef_policy_accepts_gui_labels_and_backend_tokens() -> None:
    assert normalize_full_hef_policy("Build at end (recommended)") == "end"
    assert normalize_full_hef_policy("Build at start") == "start"
    assert normalize_full_hef_policy("Skip full-model HEF") == "skip"
    assert normalize_full_hef_policy("end") == "end"
    assert normalize_full_hef_policy("start") == "start"
    assert normalize_full_hef_policy("skip") == "skip"
    assert normalize_full_hef_policy("something unexpected") == "end"


def test_generate_benchmark_set_defines_full_hef_policy_before_use() -> None:
    src = GUI_APP_PATH.read_text(encoding="utf-8")
    stores = _function_local_stores(src, "_generate_benchmark_set")
    assert "full_hef_policy" in stores


def test_start_generation_runtime_normalizes_gui_label_policy(tmp_path: Path) -> None:
    service = BenchmarkGenerationService()
    runtime = service.start_generation_runtime(
        out_dir=tmp_path / "suite",
        bench_log_path=tmp_path / "suite" / "benchmark_generation.log",
        requested_cases=1,
        ranked_candidates=[80],
        candidate_search_pool=[80],
        hef_full_policy="Build at start",
        model_name="demo",
        model_source="/tmp/demo.onnx",
    )
    try:
        assert runtime.hef_full_policy == "start"
        assert runtime.generation_state.get("hef_full_policy") == "start"
    finally:
        runtime.close()
