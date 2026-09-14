#!/usr/bin/env python3
"""Verify existing adapter, quality, canary, and self-reference evidence."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.existing_evidence_verifier import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
