"""Persistent settings for ONNX Splitpoint Tool.

The GUI is intentionally stateful (paths, presets, last selections). This package
stores that state in a single versioned JSON file under the user's home folder.

Design goals:
  * Atomic writes (no corrupted settings on crash)
  * Resilient loads (backup and fall back to defaults)
  * No secrets (SSH passwords etc.)
"""

from .store import SettingsStore

__all__ = ["SettingsStore"]
