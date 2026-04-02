"""GUI controller utilities (no Tk widget code).

This module is intended to hold non-UI application logic that is invoked by the GUI.
Keep this free of tkinter/ttk imports so it can be unit-tested headlessly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ..resources_utils import copy_resource_tree, read_text

log = logging.getLogger(__name__)


def _templates_dir() -> Path:
    # Kept for backwards-compatible callers that still inspect the path.
    from ..resources_utils import persistent_resource_path

    return persistent_resource_path("resources", "templates")


def load_template_text(filename: str, *, encoding: str = "utf-8") -> str:
    """Load a text template from onnx_splitpoint_tool/resources/templates/.

    Raises FileNotFoundError if missing.
    """
    return read_text("resources", "templates", filename, encoding=encoding)


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
    script_changed = True
    if script_path.exists():
        try:
            if script_path.read_text(encoding="utf-8") == script:
                script_changed = False
        except Exception:
            pass

    if script_changed:
        script_path.write_text(script, encoding="utf-8")
        try:
            # Make executable on POSIX; harmless on Windows.
            os.chmod(script_path, 0o755)
        except Exception:
            pass
        log.info("Wrote benchmark suite script: %s", script_path)

    # Phase-1: vendor the lightweight runner library into the suite root.
    # This must run even when benchmark_suite.py itself did not change,
    # because benchmark bundles for existing suites rely on the vendored
    # splitpoint_runners package being refreshed independently.
    try:
        _copy_runner_lib(dst_dir)
    except Exception as e:
        # Do not hard-fail suite generation if vendoring fails.
        log.warning("Could not copy runner lib into suite: %s: %s", type(e).__name__, e)

    return str(script_path)


def _copy_runner_lib(suite_dir: Path) -> None:
    dst = suite_dir / "splitpoint_runners"
    copy_resource_tree("runners", dest=dst)


def _ignore_pycache(_dir: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__"}
    ignored.update({n for n in names if n.endswith(".pyc")})
    return ignored
