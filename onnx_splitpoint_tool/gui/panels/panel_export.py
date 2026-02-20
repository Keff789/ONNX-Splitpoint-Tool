"""Split/export panel widgets."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Optional
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

    def _selected_boundary() -> Optional[int]:
        """Best-effort: read current boundary selection from the main table."""
        if app is None:
            return None
        fn = getattr(app, "_selected_boundary_index", None)
        if not callable(fn):
            return None
        try:
            return fn()
        except Exception:
            return None

    def _render_preview(*_args) -> None:
        # app.model_path can be None during early startup (before a model is opened).
        mp = getattr(app, "model_path", None) if app is not None else None
        if mp:
            model_path = Path(mp)
            base = model_path.stem or "model"
        else:
            model_path = Path.cwd()
            base = "model"

        b = _selected_boundary()
        b_tag = f"b{b}" if isinstance(b, int) else "bXXX"

        out_parent = getattr(app, "default_output_dir", None) if app is not None else None
        out_parent = Path(out_parent) if out_parent else model_path.parent
        out_dir = out_parent / f"{base}_split_{b_tag}" if bool(var_split_folder.get()) else out_parent

        has_graphviz = shutil.which("dot") is not None
        ctx_formats = ".dot/.svg/.pdf" if has_graphviz else ".dot"

        est = 0.0
        lines = [f"{out_dir}/"]
        if vars_map["models"].get():
            lines += [f"  - {base}_part1_{b_tag}.onnx", f"  - {base}_part2_{b_tag}.onnx"]
            # rough estimate: assume similar size to original model
            try:
                est += float(model_path.stat().st_size)
            except Exception:
                est += 120 * 1024 * 1024
        if vars_map["runner"].get():
            lines += ["  - run_split_onnxruntime.py", "  - run_split_onnxruntime.bat", "  - run_split_onnxruntime.sh"]
            est += 48 * 1024

        if vars_map["context"].get():
            if bool(var_ctx_full.get()):
                lines += [f"  - split_context_{b_tag}{ctx_formats}"]
            if bool(var_ctx_cutflow.get()):
                lines += [f"  - split_context_{b_tag}_cutflow{ctx_formats}"]
                # Not part of filenames, but affects what is included.
                lines += [f"    (cut-flow hops={var_ctx_hops.get()})"]
            est += 2 * 1024 * 1024

        if vars_map["reports"].get():
            lines += ["  - split_manifest.json"]
            do_validate = bool(getattr(app, "var_split_validate", tk.BooleanVar(value=False)).get()) if app is not None else False
            if do_validate:
                lines += ["  - validation_report.json"]
            est += 128 * 1024

        lines += ["", f"Estimated size: ~{_human_size(est)}"]

        text.configure(state="normal")
        text.delete("1.0", tk.END)
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")

    for v in (
        *vars_map.values(),
        var_ctx_full,
        var_ctx_cutflow,
        var_ctx_hops,
        var_split_folder,
    ):
        v.trace_add("write", _render_preview)
    _render_preview()

    # Keep preview in sync with boundary selection changes.
    # This avoids requiring the user to toggle any checkbox to refresh.
    if app is not None and hasattr(app, "events"):
        try:
            app.events.on_candidate_selected(lambda _b: _render_preview())
        except Exception:
            # Never let preview wiring break the UI.
            pass

    frame.export_checklist_vars = vars_map  # type: ignore[attr-defined]
    return frame
