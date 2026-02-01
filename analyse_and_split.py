#!/usr/bin/env python3
"""Backwards-compatible entry point.

The project was refactored into the `onnx_splitpoint_tool` package.
This file remains as a small wrapper for older scripts and for convenience.
"""

from onnx_splitpoint_tool.api import *  # re-export for compatibility
from onnx_splitpoint_tool.cli import main as _main


if __name__ == "__main__":
    raise SystemExit(_main())
