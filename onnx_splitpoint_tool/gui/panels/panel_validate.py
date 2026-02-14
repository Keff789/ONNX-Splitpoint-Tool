"""Validation/benchmark tab placeholder."""

from __future__ import annotations

from tkinter import ttk


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    ttk.Label(
        frame,
        text="Validierungs- und Benchmark-Bedienelemente folgen inkrementell in diesem Panel.",
    ).pack(anchor="w", padx=12, pady=12)
    return frame
