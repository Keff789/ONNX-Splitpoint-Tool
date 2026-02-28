"""GUI controller utilities (no Tk widget code).

This module is intended to hold non-UI application logic that is invoked by the GUI.
Keep this free of tkinter/ttk imports so it can be unit-tested headlessly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from shutil import copytree, rmtree
from typing import Optional

log = logging.getLogger(__name__)


def _templates_dir() -> Path:
    # controller.py lives in onnx_splitpoint_tool/gui/
    # resources/ lives in onnx_splitpoint_tool/resources/
    return Path(__file__).resolve().parents[1] / "resources" / "templates"


def load_template_text(filename: str, *, encoding: str = "utf-8") -> str:
    """Load a text template from onnx_splitpoint_tool/resources/templates/.

    Raises FileNotFoundError if missing.
    """
    path = _templates_dir() / filename
    return path.read_text(encoding=encoding)


def write_benchmark_suite_script(dst_dir: str | Path, *, bench_json_name: str = "benchmark_set.json") -> str:
    """Create a benchmark suite runner script in *dst_dir*.

    Returns the written script path as string.
    """
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    script_path = dst_dir / "benchmark_suite.py"
    template = load_template_text("benchmark_suite.py.txt")
    script = template.replace("__BENCH_JSON__", bench_json_name)

    # Only rewrite the script if the content actually changed. This keeps
    # bundle caching effective (mtime stays stable on no-op updates).
    if script_path.exists():
        try:
            if script_path.read_text(encoding="utf-8") == script:
                return str(script_path)
        except Exception:
            pass

    script_path.write_text(script, encoding="utf-8")
    try:
        # Make executable on POSIX; harmless on Windows.
        os.chmod(script_path, 0o755)
    except Exception:
        pass

    log.info("Wrote benchmark suite script: %s", script_path)

    # Phase-1: vendor the lightweight runner library into the suite root.
    # This enables remote execution without requiring an installed
    # onnx_splitpoint_tool package on the target.
    try:
        _copy_runner_lib(dst_dir)
    except Exception as e:
        # Do not hard-fail suite generation if vendoring fails.
        log.warning("Could not copy runner lib into suite: %s: %s", type(e).__name__, e)

    return str(script_path)


def _copy_runner_lib(suite_dir: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "runners"
    if not src.exists():
        return

    dst = suite_dir / "splitpoint_runners"
    if dst.exists():
        rmtree(dst)

    copytree(src, dst, ignore=_ignore_pycache)


def _ignore_pycache(_dir: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__"}
    ignored.update({n for n in names if n.endswith(".pyc")})
    return ignored
