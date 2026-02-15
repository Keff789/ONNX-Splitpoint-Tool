"""Hardware panel for accelerator and communication-link configuration."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


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

    return frame
