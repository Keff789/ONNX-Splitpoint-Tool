"""Modular GUI package for ONNX Splitpoint Tool.

This module intentionally avoids eager imports to prevent circular dependencies
with the legacy ``onnx_splitpoint_tool.gui_app`` module.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SplitPointAnalyserGUI", "main"]


def __getattr__(name: str) -> Any:
    if name in {"SplitPointAnalyserGUI", "main"}:
        from .app import SplitPointAnalyserGUI, main

        return {"SplitPointAnalyserGUI": SplitPointAnalyserGUI, "main": main}[name]
    raise AttributeError(name)
