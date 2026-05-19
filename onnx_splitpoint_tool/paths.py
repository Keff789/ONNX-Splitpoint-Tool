"""Path helpers for persistent tool state and project-local logs.

The tool keeps *state* (settings, managed Hailo venvs, caches) under the
user home directory, but stores human-facing logs under the repository/tool
root so they are easy to find during local development and debugging.
"""

from __future__ import annotations

import os
from pathlib import Path


def splitpoint_home() -> Path:
    """Root directory for persistent user state."""

    return Path.home() / ".onnx_splitpoint_tool"


def splitpoint_project_root() -> Path:
    """Best-effort repository/tool root.

    Allows an explicit override via ``ONNX_SPLITPOINT_TOOL_ROOT`` for unusual
    deployment layouts.
    """

    env = str(os.environ.get("ONNX_SPLITPOINT_TOOL_ROOT", "") or "").strip()
    if env:
        try:
            return Path(env).expanduser().resolve()
        except Exception:
            return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent


def splitpoint_logs_dir() -> Path:
    """Directory for project-local log files.

    Falls back to the user state directory only if the project root is not
    writable.
    """

    preferred = splitpoint_project_root() / "logs"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except Exception:
        fallback = splitpoint_home() / "logs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def splitpoint_gui_logs_dir() -> Path:
    return splitpoint_logs_dir() / "gui"


def splitpoint_wsl_debug_dir() -> Path:
    return splitpoint_logs_dir() / "wsl_debug"


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p
