from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path("onnx_splitpoint_tool")


def test_gui_app_source_contains_backfill_search_pool_and_generation_log() -> None:
    src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    assert 'def _benchmark_candidate_search_pool' in src
    assert 'candidate_bounds_all' in src
    assert 'candidate_search_pool: List[int] = list(self._benchmark_candidate_search_pool(ranked_candidates))' in src
    assert 'picks_iter = list(candidate_search_pool)' in src
    assert 'requested_cases (target accepted): {k}' in src
    assert 'benchmark_generation.log' in src
    assert "'candidate_search_pool': int(len(candidate_search_pool))" in src
    assert 'Generation log: {bench_log_path}' in src



def test_gui_logging_source_restores_info_level_and_cwd_mirror() -> None:
    src = (ROOT / "gui_app.py").read_text(encoding="utf-8")
    assert 'if root.level == logging.NOTSET or int(root.level) > int(logging.INFO):' in src
    assert 'SPLITPOINT_WRITE_CWD_LOG", "1"' in src
    assert 'os.environ["ONNX_SPLITPOINT_CWD_LOG_PATH"] = str(cwd_log_path)' in src
    assert 'logging.log(level, "[benchmark] %s", line)' in src
    assert '_bench_log(msg)' in src



def test_gui_app_python_compiles_after_backfill_and_logging_changes() -> None:
    py_compile.compile(str(ROOT / 'gui_app.py'), doraise=True)
