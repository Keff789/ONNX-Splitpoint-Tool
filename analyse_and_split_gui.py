#!/usr/bin/env python3
"""Backwards-compatible GUI entry point.

The GUI implementation lives in ``onnx_splitpoint_tool.gui_app``.

When the tool is updated in-place, an existing ``.venv`` can lag behind newly
introduced Python dependencies. This launcher performs a small bootstrap for the
minimal GUI dependency group used by startup. Heavy export/screening packages are installed on demand by Model preparation.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _load_main():
    try:
        from onnx_splitpoint_tool.gui_app import main as _main
        return _main
    except ModuleNotFoundError:
        from onnx_splitpoint_tool.dependency_bootstrap import ensure_dependency_groups_for_python, current_env_python_candidates

        print('[bootstrap] Missing Python package(s) detected for GUI startup.', flush=True)
        print('[bootstrap] Installing shared runtime dependency groups into the active environment ...', flush=True)
        candidates = current_env_python_candidates(cwd=Path(__file__).resolve().parent)
        python_exe = candidates[0] if candidates else sys.executable
        ok, remaining = ensure_dependency_groups_for_python(python_exe, ['gui_core'], log=print)
        if not ok:
            pkgs = ', '.join(sorted({spec.package for spec in remaining})) if remaining else 'unknown'
            raise RuntimeError(f'GUI dependency bootstrap failed; unresolved packages: {pkgs}')
        from onnx_splitpoint_tool.gui_app import main as _main
        return _main


main = _load_main()


if __name__ == '__main__':
    main()
