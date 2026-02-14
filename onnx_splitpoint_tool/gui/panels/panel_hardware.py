"""Hardware panel for accelerator and RAM fit overview."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..widgets.progress_bar_tank import ProgressBarTank
from ..widgets.status_badge import StatusBadge


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)

    top = ttk.LabelFrame(frame, text="Accelerators")
    top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

    accel_names = []
    iface_names = []
    if app is not None:
        specs = getattr(app, "accel_specs", {}) or {}
        accel_names = [str(x.get("name")) for x in (specs.get("accelerators") or [])]
        iface_names = [str(x.get("name")) for x in (specs.get("interfaces") or [])]

    left_var = getattr(app, "var_memf_left_accel", tk.StringVar(value=accel_names[0] if accel_names else ""))
    right_var = getattr(app, "var_memf_right_accel", tk.StringVar(value=accel_names[1] if len(accel_names) > 1 else (accel_names[0] if accel_names else "")))
    iface_var = getattr(app, "var_memf_interface", tk.StringVar(value=iface_names[0] if iface_names else ""))

    ttk.Label(top, text="Left Accelerator:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
    ttk.Combobox(top, textvariable=left_var, values=accel_names, state="readonly", width=30).grid(row=0, column=1, sticky="w", pady=8)
    ttk.Label(top, text="Right Accelerator:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    ttk.Combobox(top, textvariable=right_var, values=accel_names, state="readonly", width=30).grid(row=1, column=1, sticky="w", pady=(0, 8))
    ttk.Label(top, text="Interface:").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    ttk.Combobox(top, textvariable=iface_var, values=iface_names, state="readonly", width=30).grid(row=2, column=1, sticky="w", pady=(0, 8))

    ram = ttk.LabelFrame(frame, text="RAM Fit")
    ram.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
    ram.columnconfigure(1, weight=1)

    left_tank = ProgressBarTank(ram)
    right_tank = ProgressBarTank(ram)
    left_badge = StatusBadge(ram, text="Left: Idle", level="idle")
    right_badge = StatusBadge(ram, text="Right: Idle", level="idle")

    ttk.Label(ram, text="Left RAM:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(8, 4))
    left_tank.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=(8, 4))
    left_badge.grid(row=0, column=2, sticky="e", padx=(0, 8), pady=(8, 4))

    ttk.Label(ram, text="Right RAM:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    right_tank.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
    right_badge.grid(row=1, column=2, sticky="e", padx=(0, 8), pady=(0, 8))

    def _badge_for_ratio(ratio: float, side: str):
        if ratio < 80.0:
            return f"{side}: Fit", "ok"
        if ratio < 95.0:
            return f"{side}: Near", "warn"
        return f"{side}: OOM", "error"

    def _update(*_args) -> None:
        mem = getattr(app, "memory_by_boundary", {}) if app is not None else {}
        b = -1
        cand = getattr(app, "selected_candidate", None) if app is not None else None
        if cand is not None:
            b = int(getattr(cand, "boundary_id", -1))
        item = mem.get(b, {}) if isinstance(mem, dict) else {}
        for side, tank, badge in (
            ("left", left_tank, left_badge),
            ("right", right_tank, right_badge),
        ):
            side_data = item.get(side, {}) if isinstance(item, dict) else {}
            used = float(side_data.get("total_mb", 0.0) or 0.0)
            limit = float(side_data.get("limit_mb", 0.0) or 0.0)
            if limit <= 0 and app is not None:
                acc = getattr(app, "_accel_by_name", lambda _n: {})(left_var.get() if side == "left" else right_var.get())
                limit = float(acc.get("ram_limit_mb", 0.0) or 0.0)
            ratio = (used / limit * 100.0) if limit > 0 else 0.0
            tank.set_progress(ratio)
            text, level = _badge_for_ratio(ratio, side.capitalize())
            badge.set(text=text, level=level)

    for v in (left_var, right_var, iface_var):
        v.trace_add("write", _update)
    _update()

    frame.ram_widgets = {"left_tank": left_tank, "right_tank": right_tank, "left_badge": left_badge, "right_badge": right_badge}  # type: ignore[attr-defined]
    return frame
