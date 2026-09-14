"""Jobs tab for background benchmark generation and remote runs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk



def build_panel(parent, app=None) -> ttk.Frame:
    outer = ttk.Frame(parent)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(1, weight=1)

    desc = ttk.Label(
        outer,
        text=(
            "Monitor background jobs here. Progress windows are optional: you can close them, keep working, "
            "and reopen them later from this tab."
        ),
        wraplength=980,
        justify="left",
    )
    desc.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

    table_frame = ttk.Frame(outer)
    table_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
    table_frame.columnconfigure(0, weight=1)
    table_frame.rowconfigure(0, weight=1)

    columns = ("type", "name", "status", "progress", "elapsed", "start", "last")
    tree = ttk.Treeview(table_frame, columns=columns, show="tree headings", selectmode="browse", height=12)
    tree.grid(row=0, column=0, sticky="nsew")
    yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
    yscroll.grid(row=0, column=1, sticky="ns")
    xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
    xscroll.grid(row=1, column=0, sticky="ew")
    tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

    headings = {
        "type": "Typ",
        "name": "Name / Zielordner",
        "status": "Status",
        "progress": "Fortschritt",
        "elapsed": "Dauer",
        "start": "Startzeit",
        "last": "Letzte Meldung",
    }
    widths = {
        "type": 130,
        "name": 360,
        "status": 140,
        "progress": 130,
        "elapsed": 90,
        "start": 155,
        "last": 320,
    }
    stretch = {"name": True, "last": True}
    tree.heading("#0", text="Job tree")
    tree.column("#0", width=270, stretch=True, anchor="w")
    for col in columns:
        tree.heading(col, text=headings[col])
        tree.column(col, width=widths[col], stretch=bool(stretch.get(col, False)), anchor="w")

    btn_row = ttk.Frame(outer)
    btn_row.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    btn_monitor = ttk.Button(btn_row, text="Open monitor", command=getattr(app, "_jobs_open_monitor_selected", None))
    btn_monitor.pack(side=tk.LEFT)
    btn_log = ttk.Button(btn_row, text="Open log", command=getattr(app, "_jobs_open_log_selected", None))
    btn_log.pack(side=tk.LEFT, padx=(8, 0))
    btn_output = ttk.Button(btn_row, text="Open output", command=getattr(app, "_jobs_open_output_selected", None))
    btn_output.pack(side=tk.LEFT, padx=(8, 0))
    btn_expand = ttk.Button(btn_row, text="Expand tree", command=getattr(app, "_jobs_expand_all", None))
    btn_expand.pack(side=tk.LEFT, padx=(16, 0))
    btn_collapse = ttk.Button(btn_row, text="Collapse tree", command=getattr(app, "_jobs_collapse_all", None))
    btn_collapse.pack(side=tk.LEFT, padx=(8, 0))
    btn_cancel = ttk.Button(btn_row, text="Cancel", command=getattr(app, "_jobs_cancel_selected", None))
    btn_cancel.pack(side=tk.LEFT, padx=(24, 0))
    btn_dismiss = ttk.Button(btn_row, text="Dismiss", command=getattr(app, "_jobs_dismiss_selected", None))
    btn_dismiss.pack(side=tk.LEFT, padx=(8, 0))

    hint = ttk.Label(
        outer,
        text="Tip: Evaluation Workflow rows are shown as a parent/subjob tree. Dauer zeigt die tatsächliche Laufzeit ab echtem Start; queued rows bleiben leer. Double-click a row to reopen its monitor/log context.",
    )
    hint.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 12))

    if app is not None:
        app.jobs_tree = tree
        app.btn_jobs_open_monitor = btn_monitor
        app.btn_jobs_open_log = btn_log
        app.btn_jobs_open_output = btn_output
        app.btn_jobs_expand = btn_expand
        app.btn_jobs_collapse = btn_collapse
        app.btn_jobs_cancel = btn_cancel
        app.btn_jobs_dismiss = btn_dismiss

        try:
            tree.bind("<<TreeviewSelect>>", lambda _event: app._jobs_update_panel_buttons())
            tree.bind("<Double-1>", lambda _event: app._jobs_open_monitor_selected())
            # v58aa: defer initial job-tree population by timer rather than
            # after_idle. after_idle can be starved by other scheduled startup
            # events and made the GUI look unresponsive.
            if hasattr(app, "after"):
                app.after(1200, app._jobs_refresh_views)
        except Exception:
            pass

    return outer
