"""Simple collapsible ttk section."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class CollapsibleSection(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str, *, expanded: bool = True):
        super().__init__(master)
        self._expanded = expanded
        self._btn = ttk.Button(self, text=self._title(title), command=self.toggle)
        self._btn.pack(fill="x")
        self.body = ttk.Frame(self)
        if expanded:
            self.body.pack(fill="both", expand=True, pady=(4, 0))

    def _title(self, title: str) -> str:
        return f"{'▼' if self._expanded else '▶'} {title}"

    def toggle(self) -> None:
        self._expanded = not self._expanded
        text = self._btn.cget("text")
        raw = text[2:] if len(text) > 2 else text
        self._btn.config(text=self._title(raw))
        if self._expanded:
            self.body.pack(fill="both", expand=True, pady=(4, 0))
        else:
            self.body.pack_forget()
