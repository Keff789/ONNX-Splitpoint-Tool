"""Split & Export tab placeholder for incremental migration."""

from __future__ import annotations

from tkinter import ttk


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    ttk.Label(
        frame,
        text="Split- und Export-Optionen werden schrittweise aus gui_app.py hierhin verschoben.",
    ).pack(anchor="w", padx=12, pady=12)
    return frame
