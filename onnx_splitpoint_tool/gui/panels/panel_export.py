"""Split/export panel widgets."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
import shutil
from tkinter import ttk


def _human_size(num: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)

    checks = ttk.LabelFrame(frame, text="Export-Checklist")
    checks.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

    vars_map = {
        "models": tk.BooleanVar(value=True),
        "runner": tk.BooleanVar(value=True),
        "context": tk.BooleanVar(value=True),
        "reports": tk.BooleanVar(value=True),
    }
    if app is not None:
        vars_map["runner"] = getattr(app, "var_split_runner", vars_map["runner"])

    ttk.Checkbutton(checks, text="Models", variable=vars_map["models"]).pack(side=tk.LEFT, padx=(8, 10), pady=8)
    ttk.Checkbutton(checks, text="Runner", variable=vars_map["runner"]).pack(side=tk.LEFT, padx=(0, 10), pady=8)
    ttk.Checkbutton(checks, text="Context", variable=vars_map["context"]).pack(side=tk.LEFT, padx=(0, 10), pady=8)
    ttk.Checkbutton(checks, text="Reports", variable=vars_map["reports"]).pack(side=tk.LEFT, padx=(0, 10), pady=8)

    context_group = ttk.LabelFrame(frame, text="Context export")
    context_group.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

    var_ctx_full = getattr(app, "var_split_ctx_full", tk.BooleanVar(value=True)) if app is not None else tk.BooleanVar(value=True)
    var_ctx_cutflow = getattr(app, "var_split_ctx_cutflow", tk.BooleanVar(value=True)) if app is not None else tk.BooleanVar(value=True)
    var_ctx_hops = getattr(app, "var_split_ctx_hops", tk.StringVar(value="2")) if app is not None else tk.StringVar(value="2")

    ttk.Checkbutton(context_group, text="Full", variable=var_ctx_full).pack(side=tk.LEFT, padx=(8, 10), pady=8)
    ttk.Checkbutton(context_group, text="Cut-flow", variable=var_ctx_cutflow).pack(side=tk.LEFT, padx=(0, 10), pady=8)
    ttk.Label(context_group, text="hops:").pack(side=tk.LEFT, padx=(4, 4))
    ttk.Combobox(
        context_group,
        textvariable=var_ctx_hops,
        values=["0", "1", "2", "3"],
        width=3,
        state="readonly",
    ).pack(side=tk.LEFT, pady=8)

    split_output_group = ttk.LabelFrame(frame, text="Split output")
    split_output_group.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
    var_split_folder = getattr(app, "var_split_folder", tk.BooleanVar(value=True)) if app is not None else tk.BooleanVar(value=True)
    ttk.Checkbutton(split_output_group, text="Export as folder", variable=var_split_folder).pack(side=tk.LEFT, padx=8, pady=8)

    graphviz_status = tk.StringVar(value="")
    lbl_graphviz = ttk.Label(frame, textvariable=graphviz_status, foreground="#b36b00")
    lbl_graphviz.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 8))

    def _update_graphviz_status(*_args) -> None:
        need_graphviz = bool(var_ctx_full.get() or var_ctx_cutflow.get())
        has_graphviz = shutil.which("dot") is not None
        if need_graphviz and not has_graphviz:
            graphviz_status.set("Warning: Graphviz 'dot' not found. DOT files can be written, but PDF/SVG rendering is unavailable.")
        elif need_graphviz:
            graphviz_status.set("Graphviz status: 'dot' found. PDF/SVG context rendering is available.")
        else:
            graphviz_status.set("")

    for v in (var_ctx_full, var_ctx_cutflow):
        v.trace_add("write", _update_graphviz_status)
    _update_graphviz_status()

    preview = ttk.LabelFrame(frame, text="Will create:")
    preview.grid(row=4, column=0, sticky="nsew", padx=12, pady=(0, 12))
    preview.columnconfigure(0, weight=1)
    preview.rowconfigure(0, weight=1)

    text = tk.Text(preview, height=14, wrap="word")
    text.grid(row=0, column=0, sticky="nsew")
    text.configure(state="disabled")

    def _render_preview(*_args) -> None:
        out_dir = Path.cwd() / "split_export"
        est = 0.0
        lines = [f"{out_dir}/"]
        if vars_map["models"].get():
            lines += ["  - part1.onnx", "  - part2.onnx"]
            est += 120 * 1024 * 1024
        if vars_map["runner"].get():
            lines += ["  - run_split_onnxruntime.py"]
            est += 16 * 1024
        if vars_map["context"].get():
            lines += ["  - split_context_bXXX.dot/.svg/.pdf", "  - llm_compact_context_bXXX.json/.pdf"]
            est += 2 * 1024 * 1024
        if vars_map["reports"].get():
            lines += ["  - split_manifest.json", "  - validation_report.json"]
            est += 128 * 1024
        lines += ["", f"Estimated size: ~{_human_size(est)}"]

        text.configure(state="normal")
        text.delete("1.0", tk.END)
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    for v in vars_map.values():
        v.trace_add("write", _render_preview)
    _render_preview()

    frame.export_checklist_vars = vars_map  # type: ignore[attr-defined]
    return frame
