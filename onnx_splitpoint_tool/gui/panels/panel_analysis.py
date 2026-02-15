"""Analyse panel with structured controls and persistent results area."""

from __future__ import annotations

import os
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Iterable

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ..analysis_params import ANALYSIS_PARAM_SPECS
from ..widgets.collapsible_section import CollapsibleSection

logger = logging.getLogger(__name__)


GLOBAL_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "CV Default": {
        "analysis": {
            "topk": "10",
            "min_gap": "2",
            "min_compute_pct": "1",
            "unknown_tensor_proxy_mb": "2.0",
            "exclude_trivial": True,
            "only_single_tensor": False,
            "strict_boundary": True,
            "rank": "score",
        },
        "llm": {"enable": False, "preset": "Standard", "mode": "decode", "prefill": "512", "decode": "2048", "use_ort_symbolic": True},
    },
    "LLM Standard": {
        "analysis": {"topk": "12", "min_gap": "2", "min_compute_pct": "1", "unknown_tensor_proxy_mb": "3.0", "exclude_trivial": True, "only_single_tensor": False, "strict_boundary": True, "rank": "score"},
        "llm": {"enable": True, "preset": "Standard", "mode": "decode", "prefill": "512", "decode": "2048", "use_ort_symbolic": True},
    },
    "LLM Latency Critical": {
        "analysis": {"topk": "15", "min_gap": "1", "min_compute_pct": "1", "unknown_tensor_proxy_mb": "3.0", "exclude_trivial": True, "only_single_tensor": True, "strict_boundary": True, "rank": "latency"},
        "llm": {"enable": True, "preset": "Latency", "mode": "decode", "prefill": "256", "decode": "1024", "use_ort_symbolic": True},
    },
    "LLM Throughput/RAG": {
        "analysis": {"topk": "20", "min_gap": "2", "min_compute_pct": "1", "unknown_tensor_proxy_mb": "4.0", "exclude_trivial": True, "only_single_tensor": False, "strict_boundary": True, "rank": "score"},
        "llm": {"enable": True, "preset": "Long Context", "mode": "prefill", "prefill": "2048", "decode": "4096", "use_ort_symbolic": True},
    },
}



def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(2, weight=1)

    top_model_bar = ttk.Frame(frame)
    top_model_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
    top_model_bar.columnconfigure(1, weight=1)

    model_info = ttk.Frame(top_model_bar)
    model_info.grid(row=0, column=1, sticky="ew", padx=8)
    model_info.columnconfigure(0, weight=1)

    model_name_var = tk.StringVar(value="(no model loaded)")
    ttk.Label(model_info, textvariable=model_name_var).grid(row=0, column=0, sticky="w")

    model_type_var = tk.StringVar(value="ONNX")
    type_badge = tk.Label(model_info, textvariable=model_type_var, bg="#37474f", fg="white", padx=6, pady=1)
    type_badge.grid(row=0, column=1, sticky="w", padx=(8, 6))

    external_var = tk.StringVar(value="External data: unknown")
    ttk.Label(model_info, textvariable=external_var).grid(row=0, column=2, sticky="w", padx=(6, 0))

    preset_bar = ttk.Frame(frame)
    preset_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
    preset_bar.columnconfigure(2, weight=1)

    ttk.Label(preset_bar, text="Global preset:").grid(row=0, column=0, sticky="w")
    preset_var = tk.StringVar(value="CV Default")
    preset_cb = ttk.Combobox(preset_bar, textvariable=preset_var, values=list(GLOBAL_PRESETS.keys()), state="readonly", width=26)
    preset_cb.grid(row=0, column=1, sticky="w", padx=(6, 10))

    modified_var = tk.StringVar(value="")
    ttk.Label(preset_bar, textvariable=modified_var, foreground="#b26a00").grid(row=0, column=2, sticky="w")

    output_btn = ttk.Button(top_model_bar, text="Output folder…")
    output_btn.grid(row=0, column=2, sticky="e")

    open_btn = ttk.Button(top_model_bar, text="Open Model…")
    open_btn.grid(row=0, column=0, sticky="w")

    main = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
    main.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

    settings_host = ttk.Frame(main)
    settings_host.columnconfigure(0, weight=1)
    settings_host.rowconfigure(0, weight=1)
    main.add(settings_host, weight=1)

    center_host = ttk.Frame(main)
    center_host.columnconfigure(0, weight=1)
    center_host.rowconfigure(0, weight=1)
    main.add(center_host, weight=3)

    inspector_host = ttk.LabelFrame(main, text="Candidate Inspector")
    inspector_host.columnconfigure(0, weight=1)
    inspector_host.rowconfigure(0, weight=1)
    main.add(inspector_host, weight=2)

    canvas = tk.Canvas(settings_host, highlightthickness=0)
    yscroll = ttk.Scrollbar(settings_host, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    yscroll.grid(row=0, column=1, sticky="ns")

    settings_stack = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=settings_stack, anchor="nw")

    def _sync_scroll(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfigure(window_id, width=canvas.winfo_width())

    settings_stack.bind("<Configure>", _sync_scroll)
    canvas.bind("<Configure>", _sync_scroll)

    sec_candidate = CollapsibleSection(settings_stack, "Candidate Generation", expanded=True)
    sec_candidate.pack(fill="x", pady=(0, 6))
    sec_scoring = CollapsibleSection(settings_stack, "Scoring", expanded=True)
    sec_scoring.pack(fill="x", pady=(0, 6))
    sec_shape = CollapsibleSection(settings_stack, "Shape & Unknown Handling", expanded=False)
    sec_shape.pack(fill="x", pady=(0, 6))
    sec_llm = CollapsibleSection(settings_stack, "LLM Presets", expanded=False)
    sec_llm.pack(fill="x", pady=(0, 6))

    frame.top_model_bar = top_model_bar  # type: ignore[attr-defined]
    frame.preset_bar = preset_bar  # type: ignore[attr-defined]
    frame.output_btn = output_btn  # type: ignore[attr-defined]
    frame.open_btn = open_btn  # type: ignore[attr-defined]
    frame.model_name_var = model_name_var  # type: ignore[attr-defined]
    frame.model_type_var = model_type_var  # type: ignore[attr-defined]
    frame.external_var = external_var  # type: ignore[attr-defined]
    frame.preset_var = preset_var  # type: ignore[attr-defined]
    frame.modified_var = modified_var  # type: ignore[attr-defined]
    frame.preset_cb = preset_cb  # type: ignore[attr-defined]
    frame.settings_sections = {
        "candidate": sec_candidate,
        "scoring": sec_scoring,
        "shape": sec_shape,
        "llm": sec_llm,
    }  # type: ignore[attr-defined]
    frame.results_host = center_host  # type: ignore[attr-defined]
    frame.inspector_host = inspector_host  # type: ignore[attr-defined]

    if app is not None:
        _wire_panel_logic(frame, app)

    return frame


def build_ui(frame: ttk.Frame, app: Any) -> None:
    """Build analysis UI directly with the correct parent widgets."""
    sections = frame.settings_sections
    section_hosts: Dict[str, ttk.Frame] = {}
    for sec_key, sec in sections.items():
        host = ttk.Frame(sec.body)
        host.pack(fill="x", padx=6, pady=(0, 6))
        section_hosts[sec_key] = host

    row_by_section: Dict[str, int] = {k: 0 for k in section_hosts}
    for spec in ANALYSIS_PARAM_SPECS:
        var = getattr(app, spec.var_name, None) if spec.var_name else None
        if var is None:
            continue
        host = section_hosts.get(spec.section)
        if host is None:
            continue

        row = row_by_section[spec.section]
        row_by_section[spec.section] = row + 1

        if spec.deprecated:
            ttk.Label(host, text=f"{spec.label}:", foreground="#666666").grid(row=row, column=0, sticky="w", pady=2)
            ttk.Label(host, text=f"Deprecated ({spec.validation})", foreground="#8a6d3b").grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
            continue

        if spec.param_type == "bool":
            ttk.Checkbutton(host, text=spec.label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
            continue

        ttk.Label(host, text=f"{spec.label}:").grid(row=row, column=0, sticky="w", pady=2)
        if spec.param_type == "choice" and spec.options:
            w = ttk.Combobox(host, textvariable=var, values=list(spec.options), width=18, state="readonly")
        else:
            w = ttk.Entry(host, textvariable=var, width=10)
        w.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)

    action_wrap = ttk.Frame(section_hosts["candidate"])
    action_wrap.grid(row=row_by_section["candidate"] + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))
    btn_analyse = ttk.Button(action_wrap, text="Analyse", command=app._on_analyse, width=12)
    btn_analyse.pack(side=tk.LEFT)
    btn_split = ttk.Button(action_wrap, text="Split selected…", command=app._split_selected_boundary)
    btn_split.pack(side=tk.LEFT, padx=(8, 0))
    btn_benchmark = ttk.Button(action_wrap, text="Benchmark set…", command=app._generate_benchmark_set)
    btn_benchmark.pack(side=tk.LEFT, padx=(8, 0))
    btn_export_tex = ttk.Button(action_wrap, text="Export TeX table…", command=app._export_tex_table)
    btn_export_tex.pack(side=tk.LEFT, padx=(8, 0))

    # Rebind action references so state-machine logic controls the new buttons.
    app.btn_analyse = btn_analyse
    app.btn_split = btn_split
    app.btn_benchmark = btn_benchmark
    app.btn_export_tex = btn_export_tex
    if hasattr(app, "_set_ui_state") and hasattr(app, "_infer_ui_state"):
        app._set_ui_state(app._infer_ui_state())

    _build_center_results(frame.results_host, app)
    _build_candidate_inspector(frame.inspector_host, app)



def _build_center_results(parent: ttk.Frame, app: Any) -> None:
    """Build center pane with candidate table (top) and plots (bottom)."""
    mid = ttk.PanedWindow(parent, orient=tk.VERTICAL)
    mid.grid(row=0, column=0, sticky="nsew")

    table_frame = ttk.LabelFrame(mid, text="Suggested Boundaries")
    mid.add(table_frame, weight=1)
    table_frame.columnconfigure(0, weight=1)
    table_frame.rowconfigure(1, weight=1)

    filter_row = ttk.Frame(table_frame)
    filter_row.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
    for ci, w in enumerate((0, 0, 0, 0, 1)):
        filter_row.columnconfigure(ci, weight=w)

    app.var_cand_search = tk.StringVar(value="")
    app.var_cand_search_regex = tk.BooleanVar(value=False)
    app.var_cand_hide_dirty = tk.BooleanVar(value=False)
    app.var_cand_group_semantic = tk.BooleanVar(value=False)
    app.var_cand_sort = tk.StringVar(value="Rank ↑")
    app.var_cand_advanced = tk.BooleanVar(value=False)

    ttk.Label(filter_row, text="Search:").grid(row=0, column=0, sticky="w", padx=(0, 4))
    app.ent_cand_search = ttk.Entry(filter_row, textvariable=app.var_cand_search, width=28)
    app.ent_cand_search.grid(row=0, column=1, sticky="w", padx=(0, 6))
    app.chk_cand_regex = ttk.Checkbutton(filter_row, text="Regex", variable=app.var_cand_search_regex, command=app._refresh_candidates_table)
    app.chk_cand_regex.grid(row=0, column=2, sticky="w", padx=(0, 8))
    app.chk_cand_dirty = ttk.Checkbutton(filter_row, text="Hide dirty splits", variable=app.var_cand_hide_dirty, command=app._refresh_candidates_table)
    app.chk_cand_dirty.grid(row=0, column=3, sticky="w", padx=(0, 8))
    app.chk_cand_group = ttk.Checkbutton(filter_row, text="Group by semantic transition", variable=app.var_cand_group_semantic, command=app._refresh_candidates_table)
    app.chk_cand_group.grid(row=0, column=4, sticky="w", padx=(0, 8))

    ttk.Label(filter_row, text="Sort:").grid(row=0, column=5, sticky="e", padx=(6, 4))
    app.cb_cand_sort = ttk.Combobox(filter_row, textvariable=app.var_cand_sort, state="readonly", width=14,
                                    values=["Rank ↑", "Boundary ↑", "Boundary ↓", "Cut MB ↑", "Cut MB ↓", "Clean (best)"])
    app.cb_cand_sort.grid(row=0, column=6, sticky="e", padx=(0, 8))

    app.chk_cand_advanced = ttk.Checkbutton(filter_row, text="Detail (Advanced)", variable=app.var_cand_advanced, command=app._refresh_candidates_table)
    app.chk_cand_advanced.grid(row=0, column=7, sticky="e")
    app.ent_cand_search.bind("<KeyRelease>", app._refresh_candidates_table, add=True)
    app.cb_cand_sort.bind("<<ComboboxSelected>>", app._refresh_candidates_table, add=True)

    cols = ["rank", "clean", "boundary", "semantic", "cut_mb", "num_tensors", "gflops_left", "gflops_right", "left_op", "right_op", "peak_left_mib", "peak_right_mib", "peak_max_mib", "fits_left", "fits_right", "ram_left_gb", "ram_right_gb"]
    table_inner = ttk.Frame(table_frame)
    table_inner.grid(row=1, column=0, sticky="nsew")
    table_inner.columnconfigure(0, weight=1)
    table_inner.rowconfigure(0, weight=1)

    app.tree = ttk.Treeview(table_inner, columns=cols, show="headings")
    for key, text in (
        ("rank", "#"), ("clean", "Clean"), ("boundary", "Boundary"), ("semantic", "Semantic"),
        ("left_op", "Left op"), ("right_op", "Right op"), ("cut_mb", "Cut (MB)"), ("num_tensors", "#Tensors"),
        ("gflops_left", "Compute Left (GFLOPs)"), ("gflops_right", "Compute Right (GFLOPs)"),
        ("peak_left_mib", "Peak L (MiB)"), ("peak_right_mib", "Peak R (MiB)"), ("peak_max_mib", "Peak max (MiB)"),
        ("fits_left", "Fits L"), ("fits_right", "Fits R"), ("ram_left_gb", "RAM L (GB)"), ("ram_right_gb", "RAM R (GB)"),
    ):
        app.tree.heading(key, text=text)
    app.tree.column("rank", width=40, anchor=tk.E)
    app.tree.column("clean", width=60, anchor=tk.CENTER)
    app.tree.column("boundary", width=80, anchor=tk.E)
    app.tree.column("semantic", width=190)
    app.tree.column("left_op", width=150)
    app.tree.column("right_op", width=150)
    app.tree.column("cut_mb", width=90, anchor=tk.E)
    app.tree.column("num_tensors", width=80, anchor=tk.E)
    app.tree.column("gflops_left", width=135, anchor=tk.E)
    app.tree.column("gflops_right", width=135, anchor=tk.E)
    app.tree.column("peak_left_mib", width=110, anchor=tk.E)
    app.tree.column("peak_right_mib", width=110, anchor=tk.E)
    app.tree.column("peak_max_mib", width=110, anchor=tk.E)
    app.tree.column("fits_left", width=60, anchor=tk.CENTER)
    app.tree.column("fits_right", width=60, anchor=tk.CENTER)
    app.tree.column("ram_left_gb", width=95, anchor=tk.E)
    app.tree.column("ram_right_gb", width=95, anchor=tk.E)
    app.tree.grid(row=0, column=0, sticky="nsew")

    vsb = ttk.Scrollbar(table_inner, orient="vertical", command=app.tree.yview)
    app.tree.configure(yscroll=vsb.set)
    vsb.grid(row=0, column=1, sticky="ns")
    app.tree.tag_configure("pick", background="#eef6ff")
    app.tree.tag_configure("dirty", background="#fff2f2")
    app._configure_candidate_columns()
    app.tree.bind("<<TreeviewSelect>>", app._on_tree_selection_changed, add=True)
    app.tree.bind("<Motion>", app._on_tree_motion_clean_tooltip, add=True)
    app.tree.bind("<Leave>", app._hide_tree_clean_tooltip, add=True)

    plot_frame = ttk.LabelFrame(mid, text="Plots")
    mid.add(plot_frame, weight=3)
    plot_frame.columnconfigure(0, weight=1)
    plot_frame.rowconfigure(0, weight=1)
    plot_frame.rowconfigure(1, weight=0)
    plot_frame.rowconfigure(2, weight=0)

    app.fig = Figure(figsize=(10, 6), constrained_layout=True)
    app.ax_comm = app.fig.add_subplot(2, 2, 1)
    app.ax_comp = app.fig.add_subplot(2, 2, 2)
    app.ax_pareto = app.fig.add_subplot(2, 2, 3)
    app.ax_lat = app.fig.add_subplot(2, 2, 4)

    app.canvas = FigureCanvasTkAgg(app.fig, master=plot_frame)
    app.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
    app.canvas.mpl_connect("button_press_event", app._on_plot_click_select_candidate)

    toolbar_frame = ttk.Frame(plot_frame)
    toolbar_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 0))
    app.toolbar = NavigationToolbar2Tk(app.canvas, toolbar_frame)
    app.toolbar.update()
    try:
        app.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
    except Exception:
        logger.exception("Failed to pack matplotlib toolbar")

    export_bar = ttk.Frame(plot_frame)
    export_bar.grid(row=2, column=0, sticky="ew", padx=6, pady=(2, 6))
    app.btn_export_svg = ttk.Button(export_bar, text="Export SVG (overview)", command=lambda: app._export_overview("svg"))
    app.btn_export_pdf = ttk.Button(export_bar, text="Export PDF (overview)", command=lambda: app._export_overview("pdf"))
    app.btn_export_svg_s = ttk.Button(export_bar, text="Export SVGs (single)", command=lambda: app._export_single("svg"))
    app.btn_export_pdf_s = ttk.Button(export_bar, text="Export PDFs (single)", command=lambda: app._export_single("pdf"))
    app.btn_export_svg.pack(side=tk.LEFT, padx=(0, 6))
    app.btn_export_pdf.pack(side=tk.LEFT, padx=(0, 6))
    app.btn_export_svg_s.pack(side=tk.LEFT, padx=(0, 6))
    app.btn_export_pdf_s.pack(side=tk.LEFT)



def _build_candidate_inspector(parent: ttk.Frame, app: Any) -> None:
    """Build the right-side candidate inspector directly in the analysis panel."""
    summary = ttk.LabelFrame(parent, text="Summary")
    summary.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
    summary.columnconfigure(1, weight=1)

    vars_map = {
        "boundary": tk.StringVar(value="–"),
        "semantic": tk.StringVar(value="–"),
        "compute": tk.StringVar(value="–"),
        "cut": tk.StringVar(value="–"),
        "counts": tk.StringVar(value="–"),
        "llm": tk.StringVar(value="–"),
        "proxy": tk.StringVar(value="Proxy: –"),
    }
    rows = [
        ("Boundary", "boundary"),
        ("Semantic transition", "semantic"),
        ("Compute L/R", "compute"),
        ("Cut MB", "cut"),
        ("Tensor counts", "counts"),
    ]
    for r, (lbl, key) in enumerate(rows):
        ttk.Label(summary, text=f"{lbl}:").grid(row=r, column=0, sticky="nw", padx=(6, 8), pady=2)
        ttk.Label(summary, textvariable=vars_map[key], wraplength=360, justify="left").grid(row=r, column=1, sticky="ew", padx=(0, 6), pady=2)

    llm_box = ttk.LabelFrame(parent, text="LLM Comm Breakdown")
    llm_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
    ttk.Label(llm_box, textvariable=vars_map["llm"], justify="left", wraplength=380).pack(anchor="w", padx=6, pady=(6, 2))
    ttk.Label(llm_box, textvariable=vars_map["proxy"], foreground="#6b4f00", wraplength=380).pack(anchor="w", padx=6, pady=(0, 6))

    notebook = ttk.Notebook(parent)
    notebook.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 6))
    lists: Dict[str, Any] = {}
    for name in ("Activations", "Meta", "Constants"):
        tab = ttk.Frame(notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        notebook.add(tab, text=name)
        holder = tab
        tv = ttk.Treeview(holder, columns=("name", "size"), show="headings", height=9)
        tv.heading("name", text="Tensor")
        tv.heading("size", text="MB")
        tv.column("name", width=280, anchor=tk.W)
        tv.column("size", width=80, anchor=tk.E)
        holder.rowconfigure(0, weight=1)
        tv.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(holder, orient="vertical", command=tv.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tv.configure(yscrollcommand=scroll.set)
        lists[name.lower()] = tv

    actions = ttk.Frame(parent)
    actions.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
    ttk.Button(actions, text="Split selected…", command=getattr(app, "_split_selected_boundary", None)).pack(side=tk.LEFT)
    ttk.Button(actions, text="Export context…", command=getattr(app, "_split_selected_boundary", None)).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(actions, text="Benchmark this split…", command=getattr(app, "_generate_benchmark_set", None)).pack(side=tk.LEFT, padx=(8, 0))

    def _classify_tensor(name: str, initializers: set[str]) -> str:
        n = (name or "").lower()
        if name in initializers:
            return "constants"
        if any(k in n for k in ("mask", "pos", "position", "rope", "cache", "past", "present", "token", "ids", "shape", "len")):
            return "meta"
        return "activations"

    def _update(candidate=None):
        cand = candidate if candidate is not None else getattr(app, "selected_candidate", None)
        for tv in lists.values():
            tv.delete(*tv.get_children())
        if cand is None:
            for k in vars_map:
                vars_map[k].set("–" if k != "proxy" else "Proxy: –")
            return

        b = int(getattr(cand, "boundary_id", -1))
        row = next((r for r in getattr(app, "_candidate_rows", []) if int(r.get("boundary", -1)) == b), {})
        analysis = getattr(app, "analysis", {}) if isinstance(getattr(app, "analysis", {}), dict) else {}
        costs = analysis.get("costs_bytes") or []
        unknown_counts = analysis.get("unknown_crossing_counts") or []
        proxy_mb = float(analysis.get("unknown_tensor_proxy_mb", 0.0) or 0.0)
        proxy_kb = float(analysis.get("unknown_tensor_proxy_kb_int", 0.0) or 0.0)
        value_bytes = analysis.get("value_bytes") or {}
        inits = set(analysis.get("initializer_names") or [])

        cut_tensors = list(getattr(cand, "cut_tensors", []) or row.get("cut_tensors") or [])
        unknown_n = int(unknown_counts[b]) if b < len(unknown_counts) else int(row.get("unknown_count", 0) or 0)
        cut_mb = (float(costs[b]) / 1e6) if b < len(costs) else float(row.get("cut_mb_val", 0.0) or 0.0)

        vars_map["boundary"].set(str(b))
        vars_map["semantic"].set(str(getattr(cand, "semantic_label", "") or row.get("semantic", "–")))
        vars_map["compute"].set(f"{row.get('gflops_left', '–')} / {row.get('gflops_right', '–')} GFLOPs")
        vars_map["cut"].set(f"{cut_mb:.3f} MB")
        vars_map["counts"].set(f"total={len(cut_tensors)}, unknown={unknown_n}")

        sums = {"activations": 0.0, "meta": 0.0, "constants": 0.0, "unknown": float(unknown_n) * proxy_mb}
        for t in cut_tensors:
            grp = _classify_tensor(str(t), inits)
            size_mb = float(value_bytes.get(t, 0.0) or 0.0) / 1e6
            sums[grp] += size_mb
            lists[grp].insert("", "end", values=(t, f"{size_mb:.3f}" if size_mb > 0 else "?"))

        vars_map["llm"].set(
            f"Hidden/Act: {sums['activations']:.3f} MB\n"
            f"Meta: {sums['meta']:.3f} MB\n"
            f"Unknown (proxy): {sums['unknown']:.3f} MB"
        )
        vars_map["proxy"].set(f"Proxy-Hinweis: float={proxy_mb:g} MB/Tensor, int/bool={proxy_kb:g} KB/Tensor")

    _update(None)
    if hasattr(app, "events"):
        app.events.on_candidate_selected(_update)


def _wire_panel_logic(frame: ttk.Frame, app: Any) -> None:
    preset_cb = frame.preset_cb
    build_ui(frame, app)

    def _refresh_model_bar(_model_info: Any = None) -> None:
        path = str(getattr(app, "model_path", None) or getattr(getattr(app, "gui_state", None), "current_model_path", "") or "")
        frame.model_name_var.set(os.path.basename(path) if path else "(no model loaded)")
        mtype = str(getattr(getattr(app, "gui_state", None), "model_type", "onnx") or "onnx").upper()
        frame.model_type_var.set(mtype)

        has_external = False
        if path:
            guess = f"{path}.data"
            has_external = os.path.exists(guess)
        frame.external_var.set(f"External data: {'yes' if has_external else 'no/unknown'}")

    def _refresh_modified_marker() -> None:
        preset_name = frame.preset_var.get()
        if hasattr(app, "_sync_gui_state_from_vars"):
            app._sync_gui_state_from_vars()
        modified: Iterable[str] = []
        if hasattr(app, "_analysis_modified_fields"):
            modified = app._analysis_modified_fields(preset_name, GLOBAL_PRESETS)
        modified = list(modified)
        frame.modified_var.set("" if not modified else f"modified: {', '.join(modified[:4])}{' …' if len(modified) > 4 else ''}")

    def _on_preset_selected(_event=None) -> None:
        if hasattr(app, "_apply_analysis_global_preset"):
            app._apply_analysis_global_preset(frame.preset_var.get(), GLOBAL_PRESETS)
        _refresh_modified_marker()

    preset_cb.bind("<<ComboboxSelected>>", _on_preset_selected, add=True)

    if hasattr(app, "_on_open"):
        def _open_model() -> None:
            # Call the legacy open-model flow and always refresh the panel badges
            # afterwards so model status updates are visible immediately.
            app._on_open()
            _refresh_model_bar()

        frame.open_btn.configure(command=_open_model)

        out_cmd = getattr(app, "_on_pick_output_folder", None)
        if callable(out_cmd):
            frame.output_btn.configure(command=out_cmd)

    if hasattr(app, "btn_toggle_settings"):
        try:
            app.btn_toggle_settings.pack_forget()
        except Exception:
            logger.exception("Failed to hide legacy settings toggle with pack_forget")
        try:
            app.btn_toggle_settings.grid_forget()
        except Exception:
            logger.exception("Failed to hide legacy settings toggle with grid_forget")

    if hasattr(app, "lbl_model"):
        app.lbl_model.pack_forget()

    if hasattr(app, "_emit_settings_changed"):
        vars_to_watch = [spec.var_name for spec in ANALYSIS_PARAM_SPECS if spec.var_name]
        for name in vars_to_watch:
            obj = getattr(app, name, None)
            if obj is not None and hasattr(obj, "trace_add"):
                obj.trace_add("write", lambda *_: _refresh_modified_marker())

    _refresh_model_bar()
    _refresh_modified_marker()

    if hasattr(app, "events") and hasattr(app.events, "on_model_loaded"):
        app.events.on_model_loaded(_refresh_model_bar)


def render_analysis(frame: ttk.Frame, app: Any, analysis_result: Any) -> None:
    """Single source of truth for analysis-result UI updates in the analysis panel."""
    payload = getattr(analysis_result, "plot_data", None)
    if not isinstance(payload, dict):
        payload = analysis_result if isinstance(analysis_result, dict) else {}

    result_dict: Dict[str, Any] = {}
    if isinstance(analysis_result, dict):
        result_dict = analysis_result
    elif hasattr(analysis_result, "__dict__"):
        try:
            result_dict = dict(getattr(analysis_result, "__dict__", {}) or {})
        except Exception:
            result_dict = {}

    candidates = result_dict.get("candidates", getattr(analysis_result, "candidates", []))
    if not isinstance(candidates, list):
        candidates = []
    logger.info("UI render: candidates=%d keys=%s", len(candidates), sorted(result_dict.keys()))

    analysis = payload.get("analysis")
    picks = payload.get("picks")
    params = payload.get("params")
    if not isinstance(analysis, dict) or not isinstance(picks, list) or params is None:
        logger.warning("render_analysis skipped: incomplete payload")
        return

    app._update_diagnostics(analysis)
    app._update_table(analysis, picks, params)
    table_rows = len(app.tree.get_children("")) if hasattr(app, "tree") else 0
    logger.info("UI table rows after populate=%d", table_rows)
    app._update_plots(analysis, picks, params)

    # Ensure inspector and plot highlights are initialized from a concrete row.
    try:
        children = list(app.tree.get_children("")) if hasattr(app, "tree") else []
        if children and not app.tree.selection():
            first = children[0]
            app.tree.selection_set(first)
            app.tree.focus(first)
            app.tree.see(first)
            app._on_tree_selection_changed()
    except Exception:
        logger.exception("Failed to select initial candidate row after render_analysis")

    app._set_ui_state(app._infer_ui_state())
    app._refresh_memory_forecast()


def hide_legacy_widgets(root_children: list[Any], app: Any) -> None:
    """Hide legacy root-level widgets instead of re-parenting them."""
    params_frame = getattr(app, "params_frame", None)
    mid_pane = getattr(app, "mid_pane", None)

    for widget in (params_frame, mid_pane):
        if widget is None:
            continue
        try:
            widget.pack_forget()
        except Exception:
            logger.exception("Failed to hide legacy widget: %r", widget)

    for widget in root_children:
        if widget in {params_frame, mid_pane, getattr(app, "btn_open", None), getattr(app, "lbl_model", None), getattr(app, "btn_toggle_settings", None)}:
            continue
        try:
            widget.pack_forget()
        except Exception:
            logger.exception("Failed to hide root widget during analysis panel mount: %r", widget)

    logger.info("Analysis panel initialized without legacy re-parenting")
