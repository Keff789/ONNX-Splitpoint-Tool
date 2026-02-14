"""Panel module scaffold for incremental GUI extraction."""

from __future__ import annotations

from tkinter import ttk


def build_panel(parent, app=None) -> ttk.Frame:
    """Create panel frame.

    The full UI implementation still lives in the legacy GUI class and is moved
    incrementally.
    """
    frame = ttk.Frame(parent)
    return frame
