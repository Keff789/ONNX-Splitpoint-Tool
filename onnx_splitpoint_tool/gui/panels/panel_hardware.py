"""Hardware tab placeholder."""

from __future__ import annotations

from tkinter import ttk


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    ttk.Label(
        frame,
        text="Hardware-spezifische Einstellungen werden inkrementell in dieses Panel migriert.",
    ).pack(anchor="w", padx=12, pady=12)
    return frame
