"""Reusable collapsible ttk section widget."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional


class CollapsibleSection(ttk.Frame):
    """A reusable section with a clickable header and collapsible content body."""

    def __init__(
        self,
        master: tk.Misc,
        title: str,
        *,
        expanded: bool = True,
        on_toggle: Optional[Callable[[bool], None]] = None,
    ):
        super().__init__(master)
        self._title = title
        self._expanded = bool(expanded)
        self._on_toggle = on_toggle

        self.columnconfigure(0, weight=1)

        self.header = ttk.Frame(self)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.columnconfigure(0, weight=1)

        self.toggle_button = ttk.Button(self.header, text=self._header_text(), command=self.toggle)
        self.toggle_button.grid(row=0, column=0, sticky="ew")

        self.body = ttk.Frame(self)
        if self._expanded:
            self.body.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

    def _header_text(self) -> str:
        return f"{'▼' if self._expanded else '▶'} {self._title}"

    @property
    def expanded(self) -> bool:
        return self._expanded

    def set_title(self, title: str) -> None:
        self._title = title
        self.toggle_button.configure(text=self._header_text())

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if self._expanded == expanded:
            return
        self._expanded = expanded
        self.toggle_button.configure(text=self._header_text())
        if self._expanded:
            self.body.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        else:
            self.body.grid_remove()
        if self._on_toggle:
            self._on_toggle(self._expanded)

    def toggle(self) -> None:
        self.set_expanded(not self._expanded)
