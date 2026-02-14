"""Progress bar wrapper used by long-running analysis tasks."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ProgressBarTank(ttk.Frame):
    def __init__(self, master: tk.Misc):
        super().__init__(master)
        self.variable = tk.DoubleVar(value=0.0)
        self.bar = ttk.Progressbar(self, orient="horizontal", mode="determinate", variable=self.variable)
        self.bar.pack(fill="x", expand=True)

    def set_progress(self, value: float) -> None:
        self.variable.set(max(0.0, min(100.0, float(value))))
