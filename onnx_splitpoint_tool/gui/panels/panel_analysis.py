"""Analyse panel with structured controls and persistent results area."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import TclError
from tkinter import ttk
from typing import Any, Dict, Iterable

from ..widgets.collapsible_section import CollapsibleSection
from . import panel_candidates


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

    main = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
    main.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))

    settings_host = ttk.Frame(main)
    settings_host.columnconfigure(0, weight=1)
    settings_host.rowconfigure(0, weight=1)
    main.add(settings_host, weight=0)

    results_host = ttk.Frame(main)
    results_host.columnconfigure(0, weight=1)
    results_host.rowconfigure(0, weight=1)
    main.add(results_host, weight=1)

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
    frame.results_host = results_host  # type: ignore[attr-defined]

    if app is not None:
        _wire_panel_logic(frame, app)

    return frame


def _wire_panel_logic(frame: ttk.Frame, app: Any) -> None:
    preset_cb = frame.preset_cb

    def _same_parent(widget: Any, parent: Any) -> bool:
        winfo_parent = getattr(widget, "winfo_parent", None)
        winfo_pathname = getattr(widget, "winfo_pathname", None)
        winfo_id = getattr(parent, "winfo_id", None)
        if not (callable(winfo_parent) and callable(winfo_pathname) and callable(winfo_id)):
            return False
        try:
            return bool(winfo_parent() == winfo_pathname(winfo_id()))
        except Exception:
            return False

    def _widget_text(widget: Any) -> str:
        """Best-effort text lookup for ttk/tk widgets.

        Some widgets (e.g. Frame, PanedWindow) do not support `-text` and raise
        ``TclError`` on ``cget('text')``. We must guard this during generic child
        iteration to keep GUI startup robust on all platforms.
        """

        cget = getattr(widget, "cget", None)
        if not callable(cget):
            return ""
        try:
            v = cget("text")
        except (TclError, Exception):
            return ""
        return str(v or "")

    def _refresh_model_bar() -> None:
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
        for child in frame.top_model_bar.winfo_children():
            if _widget_text(child) == "Output folder…":
                continue
        open_btn = getattr(app, "btn_open", None)
        if open_btn is not None and _same_parent(open_btn, frame.top_model_bar):
            open_btn.pack_forget()
            open_btn.pack(in_=frame.top_model_bar, side=tk.LEFT)
        else:
            ttk.Button(frame.top_model_bar, text="Open Model…", command=app._on_open).pack(side=tk.LEFT)
        out_cmd = getattr(app, "_split_selected_boundary", None)
        if callable(out_cmd):
            frame.output_btn.configure(command=out_cmd)

    if hasattr(app, "btn_toggle_settings"):
        if _same_parent(app.btn_toggle_settings, frame.top_model_bar):
            app.btn_toggle_settings.pack_forget()
            app.btn_toggle_settings.pack(in_=frame.top_model_bar, side=tk.RIGHT)
        else:
            ttk.Button(frame.top_model_bar, text="Hide settings", command=app._toggle_settings).pack(side=tk.RIGHT)

    if hasattr(app, "lbl_model"):
        app.lbl_model.pack_forget()

    if hasattr(app, "_emit_settings_changed"):
        vars_to_watch = [
            "var_topk", "var_min_gap", "var_min_compute", "var_unknown_mb", "var_rank",
            "var_exclude_trivial", "var_only_one", "var_strict_boundary", "var_llm_enable",
            "var_llm_preset", "var_llm_mode", "var_llm_prefill", "var_llm_decode", "var_llm_use_ort_symbolic",
        ]
        for name in vars_to_watch:
            obj = getattr(app, name, None)
            if obj is not None and hasattr(obj, "trace_add"):
                obj.trace_add("write", lambda *_: _refresh_modified_marker())

    _refresh_model_bar()
    _refresh_modified_marker()


def mount_legacy_widgets(frame: ttk.Frame, root_children: list[Any], app: Any) -> None:
    params_frame = getattr(app, "params_frame", None)
    mid_pane = getattr(app, "mid_pane", None)

    if params_frame is not None:
        params_frame.pack_forget()
        params_frame.pack(in_=frame.settings_sections["candidate"].body, fill="x")

        children = params_frame.grid_slaves()
        by_row = {int(w.grid_info().get("row", -1)): w for w in children}
        if 0 in by_row:
            by_row[0].grid_configure(in_=frame.settings_sections["candidate"].body, row=0, column=0, sticky="ew", padx=0, pady=0)
        if 1 in by_row:
            by_row[1].grid_configure(in_=frame.settings_sections["scoring"].body, row=0, column=0, sticky="ew", padx=0, pady=0)
        if 2 in by_row:
            by_row[2].grid_configure(in_=frame.settings_sections["llm"].body, row=0, column=0, sticky="ew", padx=0, pady=0)
        if 6 in by_row:
            by_row[6].grid_configure(in_=frame.settings_sections["shape"].body, row=0, column=0, sticky="ew", padx=0, pady=0)

    if mid_pane is not None:
        mid_pane.pack_forget()
        try:
            mid_pane.configure(orient=tk.VERTICAL)
        except Exception:
            pass
        split = panel_candidates.mount_split_view(frame.results_host, app)
        mid_pane.pack(in_=split.left_host, fill="both", expand=True)

    for widget in root_children:
        if widget in {params_frame, mid_pane, getattr(app, "btn_open", None), getattr(app, "lbl_model", None), getattr(app, "btn_toggle_settings", None)}:
            continue
        try:
            widget.pack_forget()
        except Exception:
            pass
