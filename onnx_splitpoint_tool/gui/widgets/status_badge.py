"""Small colored status badge."""

from __future__ import annotations

import tkinter as tk


class StatusBadge(tk.Label):
    COLORS = {
        "ok": "#2e7d32",
        "warn": "#ef6c00",
        "error": "#c62828",
        "idle": "#616161",
    }

    def __init__(self, master: tk.Misc, text: str = "Idle", level: str = "idle"):
        super().__init__(master, padx=8, pady=2, fg="white")
        self.set(text=text, level=level)

    def set(self, *, text: str, level: str) -> None:
        self.config(text=text, bg=self.COLORS.get(level, self.COLORS["idle"]))
