#!/usr/bin/env python3
"""Compact source-tree wrapper for backend semantic smoke audits."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.backend_semantic_smoke import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
