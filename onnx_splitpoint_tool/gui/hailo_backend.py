"""Compatibility shim for older benchmark-generation imports.

Some legacy YOLO26 benchmark-generation runs imported
``onnx_splitpoint_tool.gui.hailo_backend`` although the Hailo backend lives at
``onnx_splitpoint_tool.hailo_backend``.  Keep this re-export so resumed or older
jobs do not fail with ``ModuleNotFoundError``.
"""
from __future__ import annotations

from ..hailo_backend import *  # noqa: F401,F403
