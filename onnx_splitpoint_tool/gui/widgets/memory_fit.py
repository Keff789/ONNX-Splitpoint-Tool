"""Reusable memory-fit visualization widget for candidate inspector."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Mapping


class MemoryFitWidget(ttk.Frame):
    """Show left/right RAM usage against hardware limits."""

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.columnconfigure(1, weight=1)
        self._style = ttk.Style(self)

        self._bars: Dict[str, ttk.Progressbar] = {}
        self._name_vars: Dict[str, tk.StringVar] = {}
        self._detail_vars: Dict[str, tk.StringVar] = {}

        for row, side in enumerate(("left", "right")):
            title = side.capitalize()
            self._name_vars[side] = tk.StringVar(value=f"{title} device")
            self._detail_vars[side] = tk.StringVar(value="—")

            ttk.Label(self, textvariable=self._name_vars[side], width=34).grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
                pady=(2, 2),
            )

            bar = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100.0)
            bar.grid(row=row, column=1, sticky="ew", pady=(2, 2))
            self._bars[side] = bar

            ttk.Label(self, textvariable=self._detail_vars[side], width=26).grid(
                row=row,
                column=2,
                sticky="e",
                padx=(8, 0),
                pady=(2, 2),
            )

        self.update_from_estimate(None)

    def _set_bar_color(self, side: str, ratio_pct: float) -> None:
        if ratio_pct >= 100.0:
            color = "#d32f2f"
        elif ratio_pct >= 85.0:
            color = "#f57c00"
        else:
            color = "#2e7d32"

        style_name = f"MemoryFit.{side}.{id(self)}.Horizontal.TProgressbar"
        self._style.configure(style_name, troughcolor="#e9ecef", background=color)
        self._bars[side].configure(style=style_name)

    @staticmethod
    def _to_float(data: Mapping[str, Any], *keys: str) -> float:
        for key in keys:
            try:
                value = data.get(key)
                if value is not None:
                    return float(value)
            except Exception:
                continue
        return 0.0

    def _update_side(self, side: str, data: Mapping[str, Any] | None) -> None:
        data = data or {}
        title = side.capitalize()

        name = str(data.get("name") or data.get("device") or f"{title} device")
        limit_mb = self._to_float(data, "ram_limit_mb", "limit_mb")
        total_mb = self._to_float(data, "total_mb")

        self._name_vars[side].set(name)

        if limit_mb > 0:
            ratio_pct = (total_mb / limit_mb) * 100.0
            self._bars[side].configure(maximum=limit_mb, value=max(0.0, min(total_mb, limit_mb)))
            self._detail_vars[side].set(
                f"{total_mb / 1024.0:.2f} / {limit_mb / 1024.0:.2f} GB ({ratio_pct:.1f}%)"
            )
        else:
            ratio_pct = 0.0
            self._bars[side].configure(maximum=100.0, value=0.0)
            if total_mb > 0:
                self._detail_vars[side].set(f"{total_mb / 1024.0:.2f} GB used / limit —")
            else:
                self._detail_vars[side].set("—")

        self._set_bar_color(side, ratio_pct)

    def update_from_estimate(self, estimate_dict: Mapping[str, Any] | None) -> None:
        estimate = estimate_dict if isinstance(estimate_dict, Mapping) else {}
        self._update_side("left", estimate.get("left") if isinstance(estimate.get("left"), Mapping) else None)
        self._update_side("right", estimate.get("right") if isinstance(estimate.get("right"), Mapping) else None)
