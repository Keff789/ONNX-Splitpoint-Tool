"""Analyse panel with structured controls and persistent results area."""

from __future__ import annotations

import os
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Iterable

from ..analysis_params import ANALYSIS_PARAM_SPECS
from ..widgets.collapsible_section import CollapsibleSection
from . import panel_candidates

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
    frame.results_host = results_host  # type: ignore[attr-defined]

    if app is not None:
        _wire_panel_logic(frame, app)

    return frame


def _build_controls_from_schema(frame: ttk.Frame, app: Any) -> None:
    """Build analysis controls directly in the new accordion from central schema."""
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



def _wire_panel_logic(frame: ttk.Frame, app: Any) -> None:
    preset_cb = frame.preset_cb
    _build_controls_from_schema(frame, app)

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


def mount_legacy_widgets(frame: ttk.Frame, root_children: list[Any], app: Any) -> None:
    params_frame = getattr(app, "params_frame", None)
    mid_pane = getattr(app, "mid_pane", None)

    if params_frame is not None:
        # Keep the legacy container hidden and mount its key content blocks into
        # the new section bodies. This keeps the new left-side navigation usable
        # (settings visible + Analyse button available) while the migration is in
        # progress.
        try:
            params_frame.pack_forget()
        except Exception:
            logger.exception("Failed to hide legacy params_frame before remounting controls")

        for name, target in (
            ("diag_frame", "shape"),
            ("memf_frame", "shape"),
            ("adv_container", "llm"),
        ):
            widget = getattr(app, name, None)
            if widget is None:
                continue
            target_body = frame.settings_sections[target].body
            widget_parent_path = ""
            target_parent_path = ""
            try:
                widget_parent_path = str(widget.nametowidget(widget.winfo_parent()))
                target_parent_path = str(target_body)
            except Exception:
                logger.exception("Failed to resolve Tk parent path while mounting '%s'", name)
            # NOTE:
            #   Legacy widgets were created as children of `params_frame`.
            #   `pack(in_=...)` does not reparent widgets, it only changes geometry
            #   management. If parent != target container, Tk rejects the mount
            #   (`can't pack ... inside ...`), leaving accordion sections empty.
            if widget_parent_path and target_parent_path and widget_parent_path != target_parent_path:
                logger.warning(
                    "Skipping legacy mount for '%s': pack(in_=...) would fail because widget parent (%s) != target (%s)",
                    name,
                    widget_parent_path,
                    target_parent_path,
                )
                continue
            try:
                widget.grid_forget()
            except Exception:
                logger.exception("Failed to clear legacy geometry manager (grid) for widget '%s'", name)
            try:
                widget.pack_forget()
            except Exception:
                logger.exception("Failed to clear legacy geometry manager (pack) for widget '%s'", name)
            try:
                widget.pack(in_=frame.settings_sections[target].body, fill="x", pady=(0, 6))
            except Exception:
                logger.exception("Failed to mount legacy widget '%s' into settings section '%s'", name, target)

        try:
            frame.settings_sections["shape"].set_expanded(True)
            frame.settings_sections["llm"].set_expanded(True)
        except Exception:
            logger.exception("Failed to expand migrated settings sections after mounting")

    if mid_pane is not None:
        mid_pane.pack_forget()
        try:
            mid_pane.configure(orient=tk.VERTICAL)
        except Exception:
            logger.exception("Failed to set mid_pane orientation to vertical during mount")
        split = panel_candidates.mount_split_view(frame.results_host, app)
        mid_pane.pack(in_=split.left_host, fill="both", expand=True)

    for widget in root_children:
        if widget in {params_frame, mid_pane, getattr(app, "btn_open", None), getattr(app, "lbl_model", None), getattr(app, "btn_toggle_settings", None)}:
            continue
        try:
            widget.pack_forget()
        except Exception:
            logger.exception("Failed to hide root widget during analysis panel mount: %r", widget)

    logger.info("Analysis panel mounting completed")
