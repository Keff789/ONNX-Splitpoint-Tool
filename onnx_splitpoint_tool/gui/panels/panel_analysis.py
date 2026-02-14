"""Analyse tab container."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    legacy_mount = ttk.Frame(frame)
    legacy_mount.grid(row=0, column=0, sticky="nsew")
    legacy_mount.columnconfigure(0, weight=1)
    legacy_mount.rowconfigure(2, weight=1)

    frame.legacy_mount = legacy_mount  # type: ignore[attr-defined]
    return frame
