"""Path helpers for persistent tool state.

We intentionally keep these helpers tiny and dependency-free because they are
used by both the GUI and subprocess backends.
"""

from __future__ import annotations

from pathlib import Path


def splitpoint_home() -> Path:
    """Root directory for all persistent tool state."""

    return Path.home() / ".onnx_splitpoint_tool"


def splitpoint_logs_dir() -> Path:
    """Directory for log files."""

    return splitpoint_home() / "logs"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
