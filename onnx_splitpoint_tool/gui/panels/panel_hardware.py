"""Hardware panel for accelerator and RAM fit overview."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..widgets.progress_bar_tank import ProgressBarTank
from ..widgets.status_badge import StatusBadge


def _str_var(app, name: str, default: str = "") -> tk.StringVar:
    if app is None:
        return tk.StringVar(value=default)
    return getattr(app, name, tk.StringVar(value=default))


def _bool_var(app, name: str, default: bool = False) -> tk.BooleanVar:
    if app is None:
        return tk.BooleanVar(value=default)
    return getattr(app, name, tk.BooleanVar(value=default))


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)

    accel_names = []
    iface_names = []
    iface_by_name = {}
    if app is not None:
        specs = getattr(app, "accel_specs", {}) or {}
        accelerators = specs.get("accelerators") or []
        interfaces = specs.get("interfaces") or []
        accel_names = [str(x.get("name")) for x in accelerators]
        iface_names = [str(x.get("name")) for x in interfaces]
        iface_by_name = {str(x.get("name")): x for x in interfaces}

    top = ttk.LabelFrame(frame, text="Accelerators")
    top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

    left_var = getattr(app, "var_memf_left_accel", tk.StringVar(value=accel_names[0] if accel_names else ""))
    right_var = getattr(
        app,
        "var_memf_right_accel",
        tk.StringVar(value=accel_names[1] if len(accel_names) > 1 else (accel_names[0] if accel_names else "")),
    )
    iface_var = getattr(app, "var_memf_interface", tk.StringVar(value=iface_names[0] if iface_names else ""))

    ttk.Label(top, text="Left Accelerator:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
    ttk.Combobox(top, textvariable=left_var, values=accel_names, state="readonly", width=30).grid(
        row=0, column=1, sticky="w", pady=8
    )
    ttk.Label(top, text="Right Accelerator:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    ttk.Combobox(top, textvariable=right_var, values=accel_names, state="readonly", width=30).grid(
        row=1, column=1, sticky="w", pady=(0, 8)
    )
    ttk.Label(top, text="Link/Interface:").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    ttk.Combobox(top, textvariable=iface_var, values=iface_names, state="readonly", width=30).grid(
        row=2, column=1, sticky="w", pady=(0, 8)
    )

    link = ttk.LabelFrame(frame, text="Communication link")
    link.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

    bw_var = _str_var(app, "var_bw", "")
    overhead_var = _str_var(app, "var_overhead", "0")

    ttk.Label(link, text="Link bandwidth (MB/s):").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
    ttk.Entry(link, textvariable=bw_var, width=12).grid(row=0, column=1, sticky="w", pady=8)

    ttk.Label(link, text="Overhead (ms):").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=8)
    ttk.Entry(link, textvariable=overhead_var, width=10).grid(row=0, column=3, sticky="w", pady=8)

    var_link_adv = _bool_var(app, "var_link_adv", False)
    ttk.Checkbutton(link, text="Detail (Advanced)", variable=var_link_adv).grid(
        row=0, column=4, sticky="w", padx=(12, 8), pady=8
    )

    adv = ttk.Frame(link)
    adv.grid(row=1, column=0, columnspan=5, sticky="ew", padx=8, pady=(0, 8))

    link_model_var = _str_var(app, "var_link_model", "ideal")
    mtu_var = _str_var(app, "var_link_mtu", "")
    pkt_ovh_ms_var = _str_var(app, "var_link_pkt_ovh_ms", "")

    ttk.Label(adv, text="Link model:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(adv, textvariable=link_model_var, values=["ideal", "packetized"], state="readonly", width=11).grid(
        row=0, column=1, sticky="w", padx=(4, 12)
    )
    ttk.Label(adv, text="MTU payload (B):").grid(row=0, column=2, sticky="w")
    ttk.Entry(adv, textvariable=mtu_var, width=10).grid(row=0, column=3, sticky="w", padx=(4, 12))
    ttk.Label(adv, text="pkt overhead (ms):").grid(row=0, column=4, sticky="w")
    ttk.Entry(adv, textvariable=pkt_ovh_ms_var, width=10).grid(row=0, column=5, sticky="w", padx=(4, 0))

    def _toggle_link_adv(*_args) -> None:
        if bool(var_link_adv.get()):
            adv.grid()
        else:
            adv.grid_remove()

    def _sync_link_from_interface(*_args) -> None:
        iface = iface_by_name.get(str(iface_var.get()))
        if not iface:
            return
        bw = iface.get("bandwidth_mb_s")
        ovh = iface.get("latency_overhead_ms")
        if bw is not None:
            bw_var.set(str(bw))
        if ovh is not None:
            overhead_var.set(str(ovh))

    var_link_adv.trace_add("write", _toggle_link_adv)
    iface_var.trace_add("write", _sync_link_from_interface)
    _toggle_link_adv()
    _sync_link_from_interface()

    ram = ttk.LabelFrame(frame, text="RAM Fit")
    ram.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
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

    frame.ram_widgets = {
        "left_tank": left_tank,
        "right_tank": right_tank,
        "left_badge": left_badge,
        "right_badge": right_badge,
    }  # type: ignore[attr-defined]
    return frame
