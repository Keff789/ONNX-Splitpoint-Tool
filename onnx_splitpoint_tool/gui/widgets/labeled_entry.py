"""Reusable label + entry composite widget."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class LabeledEntry(ttk.Frame):
    def __init__(self, master: tk.Misc, label: str, *, textvariable: tk.StringVar | None = None, width: int = 12):
        super().__init__(master)
        self.variable = textvariable or tk.StringVar()
        self.label = ttk.Label(self, text=label)
        self.label.grid(row=0, column=0, sticky="w")
        self.entry = ttk.Entry(self, textvariable=self.variable, width=width)
        self.entry.grid(row=0, column=1, sticky="w", padx=(4, 0))

    def get(self) -> str:
        return self.variable.get()
