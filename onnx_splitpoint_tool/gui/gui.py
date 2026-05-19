"""Compatibility shim for the old GUI module path.

Historically parts of the benchmark workflow tried to import
``onnx_splitpoint_tool.gui.gui``.  The maintained GUI entry points are
``onnx_splitpoint_tool.gui.app`` and ``onnx_splitpoint_tool.gui_app``.  Re-export
from ``gui.app`` first and fall back to the top-level GUI module to keep old
runs/resume paths working.
"""
from __future__ import annotations

try:  # pragma: no cover - import compatibility only
    from .app import *  # noqa: F401,F403
except Exception:  # pragma: no cover
    from ..gui_app import *  # noqa: F401,F403
