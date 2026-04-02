"""Validation/benchmark tab widgets.

This tab owns the *benchmark set* generator UI (moved from the Analysis tab).

Note
----
Hailo HEF generation settings live in the **Split & Export** tab. The benchmark
set generator reuses those settings when building a suite.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ...workdir import ensure_workdir
from ..widgets.tooltip import attach_tooltip
from ..widgets.status_badge import StatusBadge
from ..widgets.collapsible_section import CollapsibleSection
from ...benchmark.services import BenchmarkGenerationService


logger = logging.getLogger(__name__)


def _bool_var(app, name: str, default: bool) -> tk.BooleanVar:
    if app is None:
        return tk.BooleanVar(value=default)
    existing = getattr(app, name, None)
    if existing is not None:
        return existing
    created = tk.BooleanVar(value=default)
    setattr(app, name, created)
    return created


def _str_var(app, name: str, default: str) -> tk.StringVar:
    if app is None:
        return tk.StringVar(value=default)
    existing = getattr(app, name, None)
    if existing is not None:
        return existing
    created = tk.StringVar(value=default)
    setattr(app, name, created)
    return created


def build_panel(parent, app=None) -> ttk.Frame:
    outer = ttk.Frame(parent)
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=1)

    canvas = tk.Canvas(outer, highlightthickness=0)
    vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    canvas.grid(row=0, column=0, sticky="nsew")
    vscroll.grid(row=0, column=1, sticky="ns")

    frame = ttk.Frame(canvas)
    frame.columnconfigure(0, weight=1)
    window_id = canvas.create_window((0, 0), window=frame, anchor="nw")

    def _sync_scroll(_event=None):
        try:
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window_id, width=canvas.winfo_width())
        except Exception:
            pass

    def _on_mousewheel(event):
        try:
            delta = getattr(event, "delta", 0)
            if delta:
                canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            else:
                num = getattr(event, "num", None)
                if num == 4:
                    canvas.yview_scroll(-1, "units")
                elif num == 5:
                    canvas.yview_scroll(1, "units")
        except Exception:
            pass

    frame.bind("<Configure>", _sync_scroll)
    canvas.bind("<Configure>", _sync_scroll)
    for widget in (outer, canvas, frame):
        try:
            widget.bind_all("<MouseWheel>", _on_mousewheel, add="+")
            widget.bind_all("<Button-4>", _on_mousewheel, add="+")
            widget.bind_all("<Button-5>", _on_mousewheel, add="+")
        except Exception:
            pass

    sec_set = CollapsibleSection(frame, "Benchmark set", expanded=True)
    sec_set.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
    sec_set.body.columnconfigure(0, weight=1)

    sec_plan = CollapsibleSection(frame, "Run plan", expanded=True)
    sec_plan.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
    sec_plan.body.columnconfigure(0, weight=1)

    sec_exec = CollapsibleSection(frame, "Execution", expanded=True)
    sec_exec.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))
    sec_exec.body.columnconfigure(0, weight=1)

    # ----------------------------- Benchmark set -----------------------------

    bench_group = ttk.LabelFrame(sec_set.body, text="Benchmark set")
    bench_group.grid(row=0, column=0, sticky="ew")
    bench_group.columnconfigure(6, weight=1)

    var_bench_topk = _str_var(app, "var_bench_topk", "20")
    var_bench_objective = _str_var(app, "var_bench_objective", "Use analysis objective")

    btn_bench = ttk.Button(bench_group, text="Generate benchmark set…", command=getattr(app, "_generate_benchmark_set", None))
    btn_bench.grid(row=0, column=0, padx=(8, 10), pady=8, sticky="w")
    attach_tooltip(
        btn_bench,
        "Create a folder with multiple split cases (part1/part2 ONNX + manifests + runner scripts).\n"
        "Optional: also build Hailo HEFs for selected Hailo targets.",
    )

    btn_resume = ttk.Button(bench_group, text="Resume benchmark set…", command=getattr(app, "_resume_benchmark_set", None))
    btn_resume.grid(row=0, column=1, padx=(0, 10), pady=8, sticky="w")
    attach_tooltip(
        btn_resume,
        "Resume an existing, incomplete benchmark set.\n"
        "The tool runs a consistency check first and reuses intact case artefacts on disk.",
    )

    # Rebind legacy state-machine reference (the Analysis tab no longer owns this button).
    if app is not None:
        app.btn_benchmark = btn_bench
        app.btn_resume_benchmark = btn_resume
        try:
            if hasattr(app, "after_idle") and hasattr(app, "_set_ui_state") and hasattr(app, "_infer_ui_state"):
                app.after_idle(lambda: app._set_ui_state(app._infer_ui_state()))
        except Exception:
            logger.exception("Failed to refresh Benchmark button state after notebook rebind")

    ttk.Label(bench_group, text="Use top N picks:").grid(row=0, column=2, sticky="w")
    ent_topk = ttk.Entry(bench_group, textvariable=var_bench_topk, width=6)
    ent_topk.grid(row=0, column=3, sticky="w", padx=(4, 12))
    attach_tooltip(ent_topk, "How many of the currently ranked picks should be exported into the benchmark set.")

    ttk.Label(bench_group, text="Benchmark objective:").grid(row=0, column=4, sticky="w")
    cb_bench_objective = ttk.Combobox(
        bench_group,
        textvariable=var_bench_objective,
        state="readonly",
        width=18,
        values=["Use analysis objective", "Balanced", "Throughput", "Hailo feasibility", "Latency"],
    )
    cb_bench_objective.grid(row=0, column=5, sticky="w", padx=(4, 8))
    objective_badge = StatusBadge(bench_group, text='Balanced', level='idle')
    objective_badge.grid(row=0, column=6, sticky='w', padx=(0, 8))
    def _update_bench_objective_badge(*_args):
        raw = str(var_bench_objective.get() or 'Use analysis objective').strip()
        if raw == 'Use analysis objective' and app is not None and hasattr(app, 'var_analysis_objective'):
            try:
                raw = str(app.var_analysis_objective.get() or 'Balanced')
            except Exception:
                raw = 'Balanced'
        slug = raw.lower()
        level = 'idle'
        if slug.startswith('through'):
            level = 'ok'
        elif slug.startswith('hailo'):
            level = 'warn'
        elif slug.startswith('lat'):
            level = 'error'
        objective_badge.set(text=raw, level=level)
    try:
        var_bench_objective.trace_add('write', _update_bench_objective_badge)
    except Exception:
        pass
    _update_bench_objective_badge()
    attach_tooltip(
        cb_bench_objective,
        "Controls which objective label is attached to the generated benchmark set and downstream remote runs.\n"
        "Use analysis objective mirrors the currently selected analysis ranking objective.",
    )

    info = ttk.Label(
        bench_group,
        text="Hailo compile settings are configured in the 'Split & Export' tab.",
    )
    info.grid(row=1, column=0, columnspan=8, sticky="w", padx=(8, 8), pady=(0, 8))

    var_hailo_outlook_summary = _str_var(app, "var_bench_hailo_outlook_summary", "No analysis loaded yet.")
    var_hailo_outlook_detail = _str_var(app, "var_bench_hailo_outlook_detail", "")

    hailo_outlook_group = ttk.LabelFrame(bench_group, text="Hailo compile/context outlook")
    hailo_outlook_group.grid(row=2, column=0, columnspan=8, sticky="ew", padx=(8, 8), pady=(0, 8))
    hailo_outlook_group.columnconfigure(0, weight=1)

    lbl_hailo_summary = ttk.Label(hailo_outlook_group, textvariable=var_hailo_outlook_summary, font=("TkDefaultFont", 10, "bold"))
    lbl_hailo_summary.grid(row=0, column=0, sticky="w", padx=(8, 8), pady=(8, 2))
    lbl_hailo_detail = ttk.Label(hailo_outlook_group, textvariable=var_hailo_outlook_detail, wraplength=960, justify="left")
    lbl_hailo_detail.grid(row=1, column=0, sticky="w", padx=(8, 8), pady=(0, 6))

    var_hailo_part2_suggested_fallback = _bool_var(app, "var_bench_hailo_part2_suggested_fallback", True)
    chk_hailo_part2_suggested_fallback = ttk.Checkbutton(
        hailo_outlook_group,
        text="Use suggested end-node fallback for Hailo Part2",
        variable=var_hailo_part2_suggested_fallback,
    )
    chk_hailo_part2_suggested_fallback.grid(row=2, column=0, sticky="w", padx=(8, 8), pady=(0, 6))
    attach_tooltip(
        chk_hailo_part2_suggested_fallback,
        "When a Hailo Part2 parser precheck detects a blocked tail (for example a late postprocess head),\n"
        "the tool may try parser-suggested intermediate end nodes such as Transpose_2 / Transpose_3 and export a truncated Part2.\n"
        "Disable this to require original Part2 outputs only.",
    )

    outlook_table = ttk.Frame(hailo_outlook_group)
    outlook_table.grid(row=3, column=0, sticky="ew", padx=(8, 8), pady=(0, 8))
    outlook_table.columnconfigure(0, weight=1)

    outlook_cols = ("rank", "boundary", "risk", "single_prob", "cut", "peak", "score", "recommendation")
    tv_hailo_outlook = ttk.Treeview(outlook_table, columns=outlook_cols, show="headings", height=6)
    tv_hailo_outlook.grid(row=0, column=0, sticky="ew")
    sb_hailo_outlook = ttk.Scrollbar(outlook_table, orient="vertical", command=tv_hailo_outlook.yview)
    sb_hailo_outlook.grid(row=0, column=1, sticky="ns")
    tv_hailo_outlook.configure(yscrollcommand=sb_hailo_outlook.set)
    headings = {
        "rank": "#",
        "boundary": "Boundary",
        "risk": "Compile risk",
        "single_prob": "1-context %",
        "cut": "Cut MiB",
        "peak": "Peak right MiB",
        "score": "Base score",
        "recommendation": "Recommendation",
    }
    widths = {
        "rank": 44,
        "boundary": 78,
        "risk": 104,
        "single_prob": 96,
        "cut": 88,
        "peak": 108,
        "score": 88,
        "recommendation": 260,
    }
    stretches = {"recommendation": True}
    for col in outlook_cols:
        tv_hailo_outlook.heading(col, text=headings[col])
        tv_hailo_outlook.column(col, width=widths[col], stretch=bool(stretches.get(col, False)))
    attach_tooltip(
        tv_hailo_outlook,
        "Predicted Hailo compile/context outlook for the currently selected top-N candidates.\n"
        "Lower compile risk and higher 1-context probability are better.\n"
        "This drives the benchmark-set Hailo-aware candidate ordering.",
    )

    # ---------------------- Accelerators to benchmark ----------------------

    acc_group = ttk.LabelFrame(sec_plan.body, text="Accelerators to benchmark")
    acc_group.grid(row=0, column=0, sticky="ew")
    acc_group.columnconfigure(10, weight=1)

    var_acc_cpu = _bool_var(app, "var_bench_acc_cpu", True)
    var_acc_cuda = _bool_var(app, "var_bench_acc_cuda", False)
    var_acc_trt = _bool_var(app, "var_bench_acc_tensorrt", False)
    var_acc_h8 = _bool_var(app, "var_bench_acc_hailo8", False)
    var_acc_h10 = _bool_var(app, "var_bench_acc_hailo10", False)
    var_image_scale = _str_var(app, "var_bench_image_scale", "auto")

    ttk.Label(acc_group, text="ONNXRuntime:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
    chk_cpu = ttk.Checkbutton(acc_group, text="ORT CPU", variable=var_acc_cpu)
    chk_cpu.grid(row=0, column=1, sticky="w", padx=(0, 8), pady=8)
    chk_cuda = ttk.Checkbutton(acc_group, text="ORT CUDA", variable=var_acc_cuda)
    chk_cuda.grid(row=0, column=2, sticky="w", padx=(0, 8), pady=8)
    chk_trt = ttk.Checkbutton(acc_group, text="TensorRT", variable=var_acc_trt)
    chk_trt.grid(row=0, column=3, sticky="w", padx=(0, 8), pady=8)

    ttk.Label(acc_group, text="Hailo:").grid(row=0, column=5, sticky="w", padx=(18, 6), pady=8)
    chk_h8 = ttk.Checkbutton(acc_group, text="Hailo-8", variable=var_acc_h8)
    chk_h8.grid(row=0, column=6, sticky="w", padx=(0, 8), pady=8)
    chk_h10 = ttk.Checkbutton(acc_group, text="Hailo-10", variable=var_acc_h10)
    chk_h10.grid(row=0, column=7, sticky="w", padx=(0, 8), pady=8)

    attach_tooltip(chk_cpu, "Benchmark split cases with ONNXRuntime on CPU.")
    attach_tooltip(
        chk_cuda,
        "Benchmark split cases with ONNXRuntime CUDA EP (requires a CUDA-capable GPU and onnxruntime-gpu).",
    )
    attach_tooltip(
        chk_trt,
        "Benchmark split cases with TensorRT EP via ONNXRuntime (requires TensorRT + ORT TRT EP).",
    )
    attach_tooltip(
        chk_h8,
        "Include Hailo-8 in the benchmark plan. HEFs will be built using the settings in 'Split & Export'.",
    )
    attach_tooltip(
        chk_h10,
        "Include Hailo-10 in the benchmark plan. HEFs will be built using the settings in 'Split & Export'.",
    )

    # Input scaling (passed through to run_split_onnxruntime.py)
    ttk.Label(acc_group, text="Input image scale:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    cb_scale = ttk.Combobox(
        acc_group,
        textvariable=var_image_scale,
        values=["auto", "norm", "raw", "imagenet", "clip"],
        width=12,
        state="readonly",
    )
    cb_scale.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=(0, 8))
    attach_tooltip(
        cb_scale,
        "Input scaling for the image preprocessing harness (passed to the runner as --image-scale).\n"
        "auto = probe (YOLO), otherwise use a fixed scaling mode.",
    )

    var_validation_images = _str_var(app, "var_bench_validation_images", "")
    var_validation_max_images = _str_var(app, "var_bench_validation_max_images", "50")

    ttk.Label(acc_group, text="Semantic validation images:").grid(row=1, column=3, sticky="w", padx=(18, 6), pady=(0, 2))
    ent_val_images = ttk.Entry(acc_group, textvariable=var_validation_images, width=38)
    ent_val_images.grid(row=1, column=4, columnspan=3, sticky="ew", padx=(0, 6), pady=(0, 2))

    def _browse_validation_images_file():
        p_sel = filedialog.askopenfilename(
            title="Select semantic validation image set",
            filetypes=[
                ("Images / lists", "*.png *.jpg *.jpeg *.bmp *.webp *.txt *.json *.jsonl"),
                ("All", "*"),
            ],
        )
        if p_sel:
            var_validation_images.set(str(p_sel))

    def _browse_validation_images_dir():
        p_sel = filedialog.askdirectory(title="Select semantic validation image folder")
        if p_sel:
            var_validation_images.set(str(p_sel))

    btn_val_file = ttk.Button(acc_group, text="Datei…", command=_browse_validation_images_file)
    btn_val_file.grid(row=1, column=7, sticky="w", padx=(0, 6), pady=(0, 2))
    btn_val_dir = ttk.Button(acc_group, text="Ordner…", command=_browse_validation_images_dir)
    btn_val_dir.grid(row=1, column=8, sticky="w", padx=(0, 6), pady=(0, 2))
    ttk.Label(acc_group, text="Max:").grid(row=1, column=9, sticky="e", padx=(8, 4), pady=(0, 2))
    ent_val_max = ttk.Entry(acc_group, textvariable=var_validation_max_images, width=5)
    ent_val_max.grid(row=1, column=10, sticky="w", padx=(0, 8), pady=(0, 2))

    lbl_val_default = ttk.Label(
        acc_group,
        text="Default if empty: built-in COCO-50 (50 images, embedded once per suite)",
    )
    lbl_val_default.grid(row=2, column=4, columnspan=7, sticky="w", padx=(0, 6), pady=(0, 8))
    attach_tooltip(
        ent_val_images,
        "Optional image folder / image / text / JSON list for dataset-based semantic validation.\n"
        "Used only for proxy detection validation (Boxes / Scores / Klassen nach Decoding/NMS).\n"
        "If left empty, generated YOLO benchmark suites automatically embed the built-in COCO-50 set once at suite level.\n"
        "0 images = disabled. Recommended for tool-validation: ~50 images.",
    )
    attach_tooltip(ent_val_max, "How many images from the semantic validation set should be used per case/run. Default: 50 from the built-in COCO-50 set when no custom path is given. Use 0 only if you explicitly want to disable dataset validation.")
    attach_tooltip(lbl_val_default, "If no file or folder is selected, the built-in COCO-50 validation set is used automatically.")

    # ------------------------ Hailo benchmark modes ------------------------

    # A benchmark plan may include multiple Hailo variants. We expose a simple
    # preset-based selector here and optionally allow custom combinations.
    var_hailo_bench_preset = _str_var(app, "var_hailo_bench_preset", "End-to-end compare")
    var_hailo_custom_full = _bool_var(app, "var_hailo_bench_custom_full", True)
    var_hailo_custom_composed = _bool_var(app, "var_hailo_bench_custom_composed", True)
    var_hailo_custom_part1 = _bool_var(app, "var_hailo_bench_custom_part1", False)
    var_hailo_custom_part2 = _bool_var(app, "var_hailo_bench_custom_part2", False)
    var_hailo_bench_info = _str_var(app, "var_hailo_bench_info", "")
    var_hailo_full_hef_order = _str_var(app, "var_hailo_full_hef_order", "Build at end (recommended)")

    ttk.Label(acc_group, text="Hailo benchmark mode:").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    cb_hailo_mode = ttk.Combobox(
        acc_group,
        textvariable=var_hailo_bench_preset,
        values=["End-to-end compare", "Split diagnostics", "Everything", "Custom"],
        width=20,
        state="readonly",
    )
    cb_hailo_mode.grid(row=2, column=1, sticky="w", padx=(0, 8), pady=(0, 8))
    attach_tooltip(
        cb_hailo_mode,
        "Choose which Hailo variants should be benchmarked.\n\n"
        "End-to-end compare: full + composed\n"
        "Split diagnostics: composed + part1 + part2\n"
        "Everything: full + composed + part1 + part2\n"
        "Custom: choose variants manually.",
    )

    custom_frame = ttk.Frame(acc_group)
    custom_frame.grid(row=3, column=1, columnspan=9, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Checkbutton(custom_frame, text="full", variable=var_hailo_custom_full).pack(side=tk.LEFT)
    ttk.Checkbutton(custom_frame, text="composed", variable=var_hailo_custom_composed).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Checkbutton(custom_frame, text="part1", variable=var_hailo_custom_part1).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Checkbutton(custom_frame, text="part2", variable=var_hailo_custom_part2).pack(side=tk.LEFT, padx=(8, 0))
    attach_tooltip(
        custom_frame,
        "Custom Hailo benchmark variants.\n\n"
        "full = run the full-model HEF\n"
        "part1/part2 = measure stage latencies separately\n"
        "composed = run part1 -> part2 as a pipeline",
    )

    info_hailo = ttk.Label(acc_group, textvariable=var_hailo_bench_info)
    info_hailo.grid(row=4, column=0, columnspan=11, sticky="w", padx=(8, 8), pady=(0, 6))

    ttk.Label(acc_group, text="Full-model HEF build:").grid(row=5, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    cb_full_hef = ttk.Combobox(
        acc_group,
        textvariable=var_hailo_full_hef_order,
        values=["Build at end (recommended)", "Build at start", "Skip full-model HEF"],
        width=24,
        state="readonly",
    )
    cb_full_hef.grid(row=5, column=1, sticky="w", padx=(0, 8), pady=(0, 8), columnspan=2)
    attach_tooltip(
        cb_full_hef,
        "Control when suite-level full-model Hailo HEFs are built.\n\n"
        "Build at end (recommended): compile part1/part2 cases first and full-model HEFs last.\n"
        "Build at start: keep the legacy behaviour.\n"
        "Skip full-model HEF: do not plan/run full-model Hailo benchmarks for this suite.",
    )

    diag_tools = ttk.Frame(acc_group)
    diag_tools.grid(row=6, column=0, columnspan=11, sticky="w", padx=(8, 8), pady=(0, 8))
    btn_last_diag = ttk.Button(
        diag_tools,
        text="Last Hailo diagnostics…",
        command=getattr(app, "_hailo_gui_show_last_diagnostics", None),
    )
    btn_last_diag.pack(side=tk.LEFT)
    attach_tooltip(
        btn_last_diag,
        "Show the most recent Hailo HEF build diagnostics collected in this GUI session.\n"
        "Useful after benchmark-set generation to inspect partition search, SNR and timeout/failure postmortems.",
    )
    btn_open_diag = ttk.Button(
        diag_tools,
        text="Open result JSON…",
        command=getattr(app, "_hailo_gui_open_result_json", None),
    )
    btn_open_diag.pack(side=tk.LEFT, padx=(8, 0))
    attach_tooltip(
        btn_open_diag,
        "Open a hailo_hef_build_result.json from disk and render the extracted diagnostics in a scrollable popup.",
    )

    _in_update = {"flag": False}

    def _variants_from_ui() -> list[str]:
        p = (var_hailo_bench_preset.get() or "").strip().lower()
        if p.startswith("end"):
            return ["full", "composed"]
        if p.startswith("split"):
            return ["composed", "part1", "part2"]
        if p.startswith("every"):
            return ["full", "composed", "part1", "part2"]

        # Custom
        out: list[str] = []
        if bool(var_hailo_custom_full.get()):
            out.append("full")
        if bool(var_hailo_custom_composed.get()):
            out.append("composed")
        if bool(var_hailo_custom_part1.get()):
            out.append("part1")
        if bool(var_hailo_custom_part2.get()):
            out.append("part2")

        # Avoid empty selections (fall back to the default end-to-end view).
        if not out:
            out = ["full", "composed"]
        return out

    def _required_hefs(variants: list[str]) -> list[str]:
        v = set(variants)
        req: list[str] = []
        if "full" in v:
            req.append("full")
        if "part1" in v or "composed" in v:
            req.append("part1")
        if "part2" in v or "composed" in v:
            req.append("part2")
        return req

    def _update_hailo_mode_ui(*_args):
        if _in_update["flag"]:
            return
        _in_update["flag"] = True
        try:
            p = (var_hailo_bench_preset.get() or "").strip().lower()
            is_custom = p.startswith("custom")
            if is_custom:
                try:
                    custom_frame.grid()
                except Exception:
                    pass
            else:
                try:
                    custom_frame.grid_remove()
                except Exception:
                    pass

            variants = _variants_from_ui()
            req = _required_hefs(variants)
            hailo_selected = bool(var_acc_h8.get() or var_acc_h10.get())
            if not hailo_selected:
                var_hailo_bench_info.set("Hailo not selected for benchmark (enable Hailo-8/Hailo-10 above).")
            else:
                full_mode = (var_hailo_full_hef_order.get() or "").strip()
                if str(full_mode).strip().lower().startswith("skip"):
                    full_note = "full HEF skipped"
                elif str(full_mode).strip().lower().startswith("build at start"):
                    full_note = "full HEF first"
                else:
                    full_note = "full HEF last"
                var_hailo_bench_info.set(
                    f"Will benchmark (Hailo): {', '.join(variants)} · Requires HEFs: {', '.join(req)} · {full_note}"
                )
        finally:
            _in_update["flag"] = False

    # Keep the info line and custom visibility in sync.
    for v in (
        var_hailo_bench_preset,
        var_hailo_custom_full,
        var_hailo_custom_composed,
        var_hailo_custom_part1,
        var_hailo_custom_part2,
        var_acc_h8,
        var_acc_h10,
        var_hailo_full_hef_order,
    ):
        try:
            v.trace_add("write", _update_hailo_mode_ui)
        except Exception:
            pass
    _update_hailo_mode_ui()

    def _benchmark_hailo_compile_outlook_rows() -> list[int]:
        analysis = getattr(app, "analysis", None) if app is not None else None
        if not isinstance(analysis, dict):
            return []
        try:
            topk_raw = str(var_bench_topk.get() or "20").strip()
            topk = max(1, int(topk_raw)) if topk_raw else 20
        except Exception:
            topk = 20
        rows = []
        try:
            if app is not None and hasattr(app, "_benchmark_candidate_pool"):
                rows = [int(b) for b in list(app._benchmark_candidate_pool())]
        except Exception:
            rows = []
        if not rows:
            try:
                if app is not None and hasattr(app, "_benchmark_candidate_search_pool"):
                    rows = [int(b) for b in list(app._benchmark_candidate_search_pool())]
            except Exception:
                rows = []
        if not rows and isinstance(analysis.get("candidate_bounds_selected"), list):
            try:
                rows = [int(b) for b in list(analysis.get("candidate_bounds_selected") or [])]
            except Exception:
                rows = []
        return rows[:max(1, topk)]

    def _refresh_hailo_compile_outlook(*_args):
        analysis = getattr(app, "analysis", None) if app is not None else None
        for iid in tv_hailo_outlook.get_children(""):
            tv_hailo_outlook.delete(iid)
        if not isinstance(analysis, dict):
            var_hailo_outlook_summary.set("No analysis loaded yet.")
            var_hailo_outlook_detail.set("Run Analyse first to estimate which splits are more Hailo-friendly to compile and more likely to stay single-context.")
            return
        boundaries = _benchmark_hailo_compile_outlook_rows()
        if not boundaries:
            var_hailo_outlook_summary.set("No benchmark candidates available.")
            var_hailo_outlook_detail.set("Increase Top-K or re-run Analyse to populate benchmark candidates.")
            return
        service = getattr(app, "_benchmark_generation_service", BenchmarkGenerationService()) if app is not None else BenchmarkGenerationService()
        try:
            rows, summary, _meta = service.build_hailo_outlook(analysis, boundaries, top_n=min(12, len(boundaries)))
        except Exception as exc:
            logger.exception("Failed to build Hailo compile outlook")
            var_hailo_outlook_summary.set("Hailo compile outlook unavailable.")
            var_hailo_outlook_detail.set(f"Prediction failed: {type(exc).__name__}: {exc}")
            return

        hailo_selected = bool(var_acc_h8.get() or var_acc_h10.get())
        if summary is None:
            var_hailo_outlook_summary.set("No Hailo outlook data available.")
            var_hailo_outlook_detail.set("The current analysis did not expose enough metadata for Hailo-aware compile scoring.")
            return
        likely = int(summary.likely_single_context_count)
        total = max(1, int(summary.candidate_count))
        top_boundary = (f"b{int(summary.top_boundary)}" if summary.top_boundary is not None else "n/a")
        avg_risk = (f"{summary.avg_risk_score:.2f}" if summary.avg_risk_score is not None else "n/a")
        var_hailo_outlook_summary.set(
            f"Top Hailo compile outlook: {top_boundary} · likely 1-context {likely}/{total} · avg risk {avg_risk}"
        )
        if hailo_selected:
            var_hailo_outlook_detail.set(
                f"Benchmark-set generation will use compile-aware ordering for Hailo runs. "
                f"Risk bands low/medium/high: {summary.low_risk_count}/{summary.medium_risk_count}/{summary.high_risk_count}."
            )
        else:
            var_hailo_outlook_detail.set(
                f"Enable Hailo-8 or Hailo-10 above to activate compile-aware ordering during benchmark-set generation. "
                f"Risk bands low/medium/high: {summary.low_risk_count}/{summary.medium_risk_count}/{summary.high_risk_count}."
            )
        for idx, row in enumerate(rows, start=1):
            score_txt = "" if row.base_score is None else f"{row.base_score:.3f}"
            cut_txt = "" if row.cut_mib is None else f"{row.cut_mib:.2f}"
            peak_txt = "" if row.peak_act_right_mib is None else f"{row.peak_act_right_mib:.2f}"
            tv_hailo_outlook.insert(
                "",
                "end",
                values=(
                    idx,
                    f"b{int(row.boundary)}",
                    f"{row.compile_risk_score:.2f} ({row.risk_band})",
                    f"{100.0 * row.single_context_probability:.0f}%",
                    cut_txt,
                    peak_txt,
                    score_txt,
                    row.recommendation,
                ),
            )

    if app is not None:
        app._benchmark_refresh_hailo_compile_outlook = _refresh_hailo_compile_outlook
        app._benchmark_validate_sections = {"set": sec_set, "plan": sec_plan, "execution": sec_exec}
        app._benchmark_hailo_outlook_tree = tv_hailo_outlook
        app._benchmark_hailo_outlook_summary_var = var_hailo_outlook_summary
        app._benchmark_hailo_outlook_detail_var = var_hailo_outlook_detail

    for _v in (var_bench_topk,):
        try:
            _v.trace_add("write", _refresh_hailo_compile_outlook)
        except Exception:
            pass

    # ------------------------ Split pipeline matrix ------------------------

    # Matrix runs benchmark split pipelines where stage1 and stage2 can use different backends
    # (e.g. TensorRT for part1 and Hailo for part2).
    var_matrix_preset = _str_var(app, "var_matrix_preset", "None")
    var_matrix_trt_to_hailo = _bool_var(app, "var_matrix_trt_to_hailo", False)
    var_matrix_hailo_to_trt = _bool_var(app, "var_matrix_hailo_to_trt", False)
    var_matrix_info = _str_var(app, "var_matrix_info", "")

    ttk.Label(acc_group, text="Split matrix:").grid(row=7, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    cb_matrix = ttk.Combobox(
        acc_group,
        textvariable=var_matrix_preset,
        values=["None", "TRT ↔ Hailo (split)", "Custom"],
        width=20,
        state="readonly",
    )
    cb_matrix.grid(row=7, column=1, sticky="w", padx=(0, 8), pady=(0, 8))
    attach_tooltip(
        cb_matrix,
        "Generate additional matrix runs where the split stages use different backends.\n\n"
        "TRT ↔ Hailo (split) adds:\n"
        "  - TensorRT(part1) → Hailo(part2)\n"
        "  - Hailo(part1) → TensorRT(part2)\n\n"
        "Matrix runs benchmark: part1, part2, composed (no full).",
    )

    matrix_custom = ttk.Frame(acc_group)
    matrix_custom.grid(row=8, column=1, columnspan=9, sticky="w", padx=(0, 8), pady=(0, 8))
    chk_trt_to_hailo = ttk.Checkbutton(matrix_custom, text="TensorRT → Hailo", variable=var_matrix_trt_to_hailo)
    chk_trt_to_hailo.pack(side=tk.LEFT)
    chk_hailo_to_trt = ttk.Checkbutton(matrix_custom, text="Hailo → TensorRT", variable=var_matrix_hailo_to_trt)
    chk_hailo_to_trt.pack(side=tk.LEFT, padx=(12, 0))

    attach_tooltip(chk_trt_to_hailo, "Matrix run: stage1=TensorRT, stage2=Hailo (per selected Hailo target).")
    attach_tooltip(chk_hailo_to_trt, "Matrix run: stage1=Hailo, stage2=TensorRT (per selected Hailo target).")

    lbl_matrix = ttk.Label(acc_group, textvariable=var_matrix_info)
    lbl_matrix.grid(row=9, column=0, columnspan=11, sticky="w", padx=(8, 8), pady=(0, 8))

    _in_matrix = {"flag": False}

    def _update_matrix_ui(*_args):
        if _in_matrix["flag"]:
            return
        _in_matrix["flag"] = True
        try:
            trt_on = bool(var_acc_trt.get())
            hailo_on = bool(var_acc_h8.get() or var_acc_h10.get())
            enabled = trt_on and hailo_on

            p = (var_matrix_preset.get() or "").strip().lower()
            is_custom = p.startswith("custom")

            if not enabled:
                # Disable matrix UI when prerequisites are missing.
                try:
                    cb_matrix.configure(state="disabled")
                except Exception:
                    pass
                try:
                    chk_trt_to_hailo.configure(state="disabled")
                    chk_hailo_to_trt.configure(state="disabled")
                except Exception:
                    pass
                try:
                    matrix_custom.grid_remove()
                except Exception:
                    pass
                var_matrix_info.set("Matrix runs disabled (enable TensorRT + at least one Hailo target above).")
                return

            try:
                cb_matrix.configure(state="readonly")
            except Exception:
                pass

            if p.startswith("none"):
                try:
                    matrix_custom.grid_remove()
                except Exception:
                    pass
                var_matrix_trt_to_hailo.set(False)
                var_matrix_hailo_to_trt.set(False)
            elif p.startswith("trt"):
                try:
                    matrix_custom.grid_remove()
                except Exception:
                    pass
                var_matrix_trt_to_hailo.set(True)
                var_matrix_hailo_to_trt.set(True)
            else:
                try:
                    matrix_custom.grid()
                except Exception:
                    pass

            st = "normal" if is_custom else "disabled"
            try:
                chk_trt_to_hailo.configure(state=st)
                chk_hailo_to_trt.configure(state=st)
            except Exception:
                pass

            runs = []
            if bool(var_matrix_trt_to_hailo.get()):
                runs.append("TensorRT→Hailo")
            if bool(var_matrix_hailo_to_trt.get()):
                runs.append("Hailo→TensorRT")
            if runs:
                var_matrix_info.set(
                    f"Matrix runs: {', '.join(runs)} · Will benchmark: part1, part2, composed (per selected Hailo target)"
                )
            else:
                var_matrix_info.set("Matrix runs: none")
        finally:
            _in_matrix["flag"] = False

    for v in (
        var_matrix_preset,
        var_matrix_trt_to_hailo,
        var_matrix_hailo_to_trt,
        var_acc_trt,
        var_acc_h8,
        var_acc_h10,
        var_hailo_full_hef_order,
    ):
        try:
            v.trace_add("write", _update_matrix_ui)
        except Exception:
            pass
    _update_matrix_ui()

    # ------------------------ Accuracy / baseline ------------------------

    accy_group = ttk.LabelFrame(sec_plan.body, text="Accuracy / baseline")
    accy_group.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    accy_group.columnconfigure(1, weight=1)

    ttk.Label(accy_group, text="Baseline:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(8, 2))
    badge_baseline = StatusBadge(accy_group, text="Full @ ORT CPU", level="idle")
    badge_baseline.grid(row=0, column=1, sticky="w", padx=(0, 8), pady=(8, 2))
    attach_tooltip(
        badge_baseline,
        "Reference output for accuracy checks.\n"
        "A full-model inference is always executed on ONNXRuntime CPU.\n"
        "All measured variants/providers are compared against this baseline.\n"
        "Baseline outputs are cached per case and reused across runs when inputs match.\n"
        "(cache file: baseline_full_cpu_outputs.npz)\n"
        "The baseline is not part of the benchmark timing results.",
    )

    var_accy_compare_info = _str_var(app, "var_accuracy_compare_info", "")
    var_accy_hef_info = _str_var(app, "var_accuracy_hef_info", "")

    lbl_compare = ttk.Label(accy_group, textvariable=var_accy_compare_info)
    lbl_compare.grid(row=1, column=0, columnspan=2, sticky="w", padx=(8, 8), pady=(0, 2))

    lbl_hef = ttk.Label(accy_group, textvariable=var_accy_hef_info)
    lbl_hef.grid(row=2, column=0, columnspan=2, sticky="w", padx=(8, 8), pady=(0, 8))
    attach_tooltip(
        lbl_hef,
        "Derived automatically from your selected benchmark variants (Phase 6).\n"
        "Full -> full HEF, part1 -> part1 HEF, part2 -> part2 HEF, composed -> part1+part2 HEFs.",
    )

    # Planned validations table (computed from current Benchmark tab selections).
    table_frame = ttk.Frame(accy_group)
    table_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=(8, 8), pady=(0, 8))
    table_frame.columnconfigure(0, weight=1)

    _cols = ("run", "stage1", "stage2", "variants", "requires")
    tv_plan = ttk.Treeview(table_frame, columns=_cols, show="headings", height=6)
    tv_plan.grid(row=0, column=0, sticky="ew")

    vsb_plan = ttk.Scrollbar(table_frame, orient="vertical", command=tv_plan.yview)
    vsb_plan.grid(row=0, column=1, sticky="ns")
    tv_plan.configure(yscrollcommand=vsb_plan.set)

    tv_plan.heading("run", text="Run")
    tv_plan.heading("stage1", text="Stage1")
    tv_plan.heading("stage2", text="Stage2")
    tv_plan.heading("variants", text="Variants compared")
    tv_plan.heading("requires", text="Requires")

    tv_plan.column("run", width=160, stretch=False)
    tv_plan.column("stage1", width=110, stretch=False)
    tv_plan.column("stage2", width=110, stretch=False)
    tv_plan.column("variants", width=240, stretch=True)
    tv_plan.column("requires", width=160, stretch=False)

    attach_tooltip(
        tv_plan,
        "Planned accuracy comparisons (all compared against CPU full baseline).\n"
        "This table is computed from the current Benchmark tab settings.",
    )

    def _get_hailo_hw(which: str) -> str:
        """Resolve selected hw_arch strings from Split&Export settings."""
        varname = f"var_hailo_hef_{which}_hw_arch"
        fallback = "hailo8" if which == "hailo8" else "hailo10h"
        try:
            v = getattr(app, varname, None)
            if v is not None:
                s = str(v.get() or "").strip()
                if s:
                    if s == "hailo10":
                        s = "hailo10h"
                    return s
        except Exception:
            pass
        return fallback

    def _hef_req_for(stage1: str, stage2: str, variants: list[str]) -> list[str]:
        vset = {str(v).strip().lower() for v in variants if str(v).strip()}
        req: list[str] = []
        is_h1 = str(stage1).strip().lower().startswith("hailo")
        is_h2 = str(stage2).strip().lower().startswith("hailo")

        if "full" in vset and (is_h1 or is_h2):
            req.append("full")
        if is_h1 and ("part1" in vset or "composed" in vset):
            req.append("part1")
        if is_h2 and ("part2" in vset or "composed" in vset):
            req.append("part2")

        order = ["full", "part1", "part2"]
        return [x for x in order if x in req]

    _in_accy = {"flag": False}

    def _update_accuracy_ui(*_args):
        if _in_accy["flag"]:
            return
        _in_accy["flag"] = True
        try:
            planned: list[tuple[str, str, str, list[str], str]] = []

            ort_variants = ["full", "part1", "part2", "composed"]
            if bool(var_acc_cpu.get()):
                planned.append(("ORT CPU", "cpu", "cpu", ort_variants, "onnx"))
            if bool(var_acc_cuda.get()):
                planned.append(("ORT CUDA", "cuda", "cuda", ort_variants, "onnx"))
            if bool(var_acc_trt.get()):
                planned.append(("TensorRT", "tensorrt", "tensorrt", ort_variants, "onnx"))

            hailo_targets: list[str] = []
            if bool(var_acc_h8.get()):
                hailo_targets.append(_get_hailo_hw("hailo8"))
            if bool(var_acc_h10.get()):
                hailo_targets.append(_get_hailo_hw("hailo10"))

            hailo_variants = _variants_from_ui()
            for hw in hailo_targets:
                s1 = hw
                s2 = hw
                req = _hef_req_for(s1, s2, hailo_variants)
                req_txt = f"hef: {', '.join(req)}" if req else "hef: none"
                planned.append((f"Hailo ({hw})", s1, s2, list(hailo_variants), req_txt))

            matrix_variants = ["part1", "part2", "composed"]
            if bool(var_matrix_trt_to_hailo.get()):
                for hw in hailo_targets:
                    s1 = "tensorrt"
                    s2 = hw
                    req = _hef_req_for(s1, s2, matrix_variants)
                    req_txt = f"hef: {', '.join(req)}" if req else "hef: none"
                    planned.append((f"TensorRT→{hw}", s1, s2, list(matrix_variants), req_txt))

            if bool(var_matrix_hailo_to_trt.get()):
                for hw in hailo_targets:
                    s1 = hw
                    s2 = "tensorrt"
                    req = _hef_req_for(s1, s2, matrix_variants)
                    req_txt = f"hef: {', '.join(req)}" if req else "hef: none"
                    planned.append((f"{hw}→TensorRT", s1, s2, list(matrix_variants), req_txt))

            # Info lines
            union_variants: set[str] = set()
            hef_needed: set[str] = set()
            for _name, s1, s2, vv, _reqtxt in planned:
                for v in vv:
                    union_variants.add(str(v).strip().lower())
                for h in _hef_req_for(s1, s2, vv):
                    hef_needed.add(h)

            order_cmp = ["full", "composed", "part1", "part2"]
            cmp_list = [v for v in order_cmp if v in union_variants]
            if not cmp_list:
                var_accy_compare_info.set("Will compare (against CPU full): none (no accelerators selected)")
            else:
                var_accy_compare_info.set(f"Will compare (against CPU full): {', '.join(cmp_list)}")

            order_hef = ["full", "part1", "part2"]
            hef_list = [h for h in order_hef if h in hef_needed]
            if hef_list:
                var_accy_hef_info.set(f"Hailo HEFs needed (auto): {', '.join(hef_list)}")
            else:
                var_accy_hef_info.set("Hailo HEFs needed (auto): none")

            # Table
            for item in tv_plan.get_children():
                tv_plan.delete(item)
            for name, s1, s2, vv, reqtxt in planned:
                tv_plan.insert("", "end", values=(name, s1, s2, ", ".join(vv), reqtxt))
        finally:
            _in_accy["flag"] = False

    for _v in (
        var_acc_cpu,
        var_acc_cuda,
        var_acc_trt,
        var_acc_h8,
        var_acc_h10,
        var_hailo_bench_preset,
        var_hailo_custom_full,
        var_hailo_custom_composed,
        var_hailo_custom_part1,
        var_hailo_custom_part2,
        var_matrix_preset,
        var_matrix_trt_to_hailo,
        var_matrix_hailo_to_trt,
    ):
        try:
            _v.trace_add("write", _update_accuracy_ui)
        except Exception:
            pass

    # Also update when hw_arch selection changes in the Split&Export tab (if present).
    for _nm in ("var_hailo_hef_hailo8_hw_arch", "var_hailo_hef_hailo10_hw_arch"):
        try:
            _hv = getattr(app, _nm, None)
            if _hv is not None:
                _hv.trace_add("write", _update_accuracy_ui)
        except Exception:
            pass

    _update_accuracy_ui()
    for _maybe_var in (var_acc_h8, var_acc_h10, getattr(app, "var_strict_boundary", None) if app is not None else None):
        try:
            if _maybe_var is not None:
                _maybe_var.trace_add("write", _refresh_hailo_compile_outlook)
        except Exception:
            pass
    _refresh_hailo_compile_outlook()

# ----------------------------- Runner (ORT) ------------------------------

    runner_group = ttk.LabelFrame(sec_exec.body, text="Runner")
    runner_group.grid(row=0, column=0, sticky="ew")

    var_split_runner = _bool_var(app, "var_split_runner", True)
    var_runner_target = _str_var(app, "var_runner_target", "auto")

    chk_runner = ttk.Checkbutton(runner_group, text="Generate runner skeleton", variable=var_split_runner)
    chk_runner.pack(side=tk.LEFT, padx=(8, 10), pady=8)
    ttk.Label(runner_group, text="Runner target:").pack(side=tk.LEFT, padx=(0, 4))
    cb_runner = ttk.Combobox(
        runner_group,
        textvariable=var_runner_target,
        values=["auto", "cpu", "cuda", "tensorrt"],
        width=10,
        state="readonly",
    )
    cb_runner.pack(side=tk.LEFT, pady=8)

    attach_tooltip(
        chk_runner,
        "Create helper scripts next to the exported split (run_split_onnxruntime.*) to run inference quickly.",
    )
    attach_tooltip(
        cb_runner,
        "Select which runner backend to generate.\n"
        "auto: choose a sensible default based on the environment.",
    )

    # Ensure UI state reflects current model/load state (legacy state machine).
    if app is not None and hasattr(app, "_set_ui_state") and hasattr(app, "_infer_ui_state"):
        try:
            app._set_ui_state(app._infer_ui_state())
        except Exception:
            pass

    # ------------------------- Remote Benchmark (SSH) -------------------------

    remote_group = ttk.LabelFrame(sec_exec.body, text="Remote benchmark (SSH)")
    remote_group.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    remote_group.columnconfigure(0, weight=0)
    remote_group.columnconfigure(1, weight=1)
    remote_group.columnconfigure(2, weight=0)
    remote_group.columnconfigure(3, weight=0)

    sec_adv = CollapsibleSection(sec_exec.body, "Advanced", expanded=False)
    sec_adv.grid(row=2, column=0, sticky="ew", pady=(8, 0))
    sec_adv.body.columnconfigure(0, weight=1)
    advanced_group = ttk.LabelFrame(sec_adv.body, text="Advanced benchmark options")
    advanced_group.grid(row=0, column=0, sticky="ew")
    advanced_group.columnconfigure(1, weight=1)

    if app is not None:
        # Suite path (with discovery inside <working_dir>/BenchmarkSets)
        ttk.Label(remote_group, text="benchmark_set.json:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(8, 4))

        def _suite_initialdir() -> str:
            root = getattr(app, "default_output_dir", None)
            if not root:
                return os.getcwd()
            try:
                return str(ensure_workdir(Path(root)).benchmark_sets)
            except Exception:
                return str(root)

        def _list_suites() -> list[str]:
            root = getattr(app, "default_output_dir", None)
            if not root:
                return []
            try:
                wd = ensure_workdir(Path(root))
                hits = sorted(wd.benchmark_sets.rglob("benchmark_set.json"))
                return [str(p) for p in hits]
            except Exception:
                logger.exception("Failed to discover benchmark suites")
                return []

        cb_suite = ttk.Combobox(
            remote_group,
            textvariable=getattr(app, "var_remote_benchmark_set", None),
            values=_list_suites(),
            width=50,
        )
        cb_suite.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=(8, 4))

        def _refresh_suite_values():
            cb_suite["values"] = _list_suites()

        # Re-scan right before the dropdown opens.
        try:
            cb_suite.configure(postcommand=_refresh_suite_values)
        except Exception:
            pass

        def _browse_suite():
            p = filedialog.askopenfilename(
                title="Select benchmark_set.json",
                filetypes=[("benchmark_set.json", "benchmark_set.json"), ("JSON", "*.json"), ("All", "*")],
                initialdir=_suite_initialdir(),
            )
            if p:
                try:
                    app.var_remote_benchmark_set.set(p)
                except Exception:
                    pass
                if hasattr(app, "_persist_settings"):
                    try:
                        app._persist_settings()
                    except Exception:
                        pass

        btn_browse = ttk.Button(remote_group, text="Browse…", command=_browse_suite)
        btn_browse.grid(row=0, column=2, sticky="w", padx=(0, 8), pady=(8, 4))
        attach_tooltip(btn_browse, "Pick a generated benchmark suite (benchmark_set.json).")

        attach_tooltip(
            cb_suite,
            "Choose an existing suite from the Working Dir (BenchmarkSets) or paste a path to any benchmark_set.json.",
        )

        # Host
        ttk.Label(remote_group, text="Remote host:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=4)
        host_disp = tk.StringVar(value="")
        cb_host = ttk.Combobox(remote_group, textvariable=host_disp, state="readonly", width=26)
        cb_host.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=4)

        def _refresh_hosts():
            vals = []
            try:
                vals = app._remote_hosts_values_for_combo()
            except Exception:
                vals = []
            cb_host["values"] = vals
            # Restore display from selected id
            try:
                sel_id = app.var_remote_host_id.get()
            except Exception:
                sel_id = ""
            if sel_id:
                for v in vals:
                    if v.startswith(sel_id + " ") or v.startswith(sel_id + "—") or v.startswith(sel_id + " —"):
                        host_disp.set(v)
                        break

        def _on_host_selected(event=None):
            try:
                # Let app parse and store only the id
                app._remote_on_host_combo_selected(event)
            except Exception:
                pass

        cb_host.bind("<<ComboboxSelected>>", _on_host_selected)

        def _open_hosts():
            try:
                logger.info("[remote] Opening hosts dialog")
                app._remote_open_hosts_dialog(refresh_callback=_refresh_hosts)
            except Exception as e:
                logger.exception("Failed to open hosts dialog")
                try:
                    messagebox.showerror("Remote hosts", f"Failed to open hosts dialog:\n{e}")
                except Exception:
                    pass

        btn_hosts = ttk.Button(remote_group, text="Hosts…", command=_open_hosts)
        btn_hosts.grid(row=1, column=2, sticky="w", padx=(0, 6), pady=4)
        attach_tooltip(btn_hosts, "Add/edit remote SSH hosts (no passwords stored).")

        btn_test = ttk.Button(remote_group, text="Test", command=getattr(app, "_remote_test_connection", None))
        btn_test.grid(row=1, column=3, sticky="w", padx=(0, 8), pady=4)
        attach_tooltip(btn_test, "Run a quick 'ssh echo OK' to verify connectivity.")

        # Transfer mode
        ttk.Label(advanced_group, text="Transfer:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=(8,4))
        cb_transfer = ttk.Combobox(
            advanced_group,
            textvariable=app.var_remote_transfer_mode,
            values=["bundle", "direct"],
            width=10,
            state="readonly",
        )
        cb_transfer.grid(row=0, column=1, sticky="w", padx=(0, 6), pady=(8,4))
        attach_tooltip(
            cb_transfer,
            "bundle = pack suite as tar.gz (fast, cacheable)\n"
            "direct = scp -r full suite directory (useful for debugging; may be slower).",
        )

        chk_reuse = ttk.Checkbutton(
            advanced_group,
            text="Reuse cached bundle",
            variable=app.var_remote_reuse_bundle,
        )
        chk_reuse.grid(row=0, column=2, sticky="w", padx=(8, 8), pady=(8,4))
        attach_tooltip(chk_reuse, "If enabled, re-pack only when suite files changed.")

        suite_tools = ttk.Frame(advanced_group)
        suite_tools.grid(row=0, column=3, sticky="w", padx=(0, 8), pady=(8,4))

        btn_rebuild_bundle = ttk.Button(
            suite_tools,
            text="Rebuild bundle",
            command=getattr(app, "_remote_rebuild_suite_bundle", None),
        )
        btn_rebuild_bundle.pack(side=tk.LEFT)
        attach_tooltip(
            btn_rebuild_bundle,
            "Delete the cached suite_bundle.tar.gz for the selected suite.\n"
            "Use this when runner/template changes were made but the Generate button is unavailable.\n"
            "The next remote benchmark run will rebuild the bundle.",
        )

        btn_refresh_suite = ttk.Button(
            suite_tools,
            text="Refresh suite harness",
            command=getattr(app, "_refresh_selected_suite_harness", None),
        )
        btn_refresh_suite.pack(side=tk.LEFT, padx=(8, 0))
        attach_tooltip(
            btn_refresh_suite,
            "Refresh benchmark_suite.py, vendored splitpoint_runners and per-case ORT runner wrappers\n"
            "inside the selected benchmark suite without rebuilding the full benchmark set.",
        )

        # Provider / run params
        ttk.Label(remote_group, text="Provider (override):").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=4)
        cb_provider = ttk.Combobox(
            remote_group,
            textvariable=getattr(app, "var_remote_provider", None),
            values=["auto", "cpu", "cuda", "tensorrt", "openvino"],
            width=10,
            state="readonly",
        )
        cb_provider.grid(row=2, column=1, sticky="w", padx=(0, 6), pady=4)
        attach_tooltip(cb_provider, "auto = run benchmark_plan.json (all runs).\nOther values override to a single provider.")

        # Compact run controls (Repeats/Warmup/Runs) + a live "Total runs" preview.
        run_params = ttk.Frame(remote_group)
        run_params.grid(row=2, column=2, columnspan=2, sticky="ne", padx=(16, 8), pady=4)

        ttk.Label(run_params, text="Repeats:").grid(row=0, column=0, sticky="e", padx=(0, 4))
        ent_rep = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_repeats", None), width=6)
        ent_rep.grid(row=0, column=1, sticky="w", padx=(0, 10))

        ttk.Label(run_params, text="Warmup:").grid(row=0, column=2, sticky="e", padx=(0, 4))
        ent_warm = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_warmup", None), width=6)
        ent_warm.grid(row=0, column=3, sticky="w", padx=(0, 10))

        ttk.Label(run_params, text="Runs:").grid(row=0, column=4, sticky="e", padx=(0, 4))
        ent_runs = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_iters", None), width=6)
        ent_runs.grid(row=0, column=5, sticky="w", padx=(0, 10))

        ttk.Label(run_params, text="Outer timeout (s):").grid(row=1, column=0, sticky="e", padx=(0, 4), pady=(6, 0))
        ent_timeout = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_timeout", None), width=8)
        ent_timeout.grid(row=1, column=1, sticky="w", padx=(0, 4), pady=(6, 0))
        ttk.Label(run_params, text="0 = off").grid(row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))

        var_tp_preset = _str_var(app, "var_remote_streaming_preset", "default")
        ttk.Label(run_params, text="Streaming preset:").grid(row=2, column=0, sticky="e", padx=(0, 4), pady=(6, 0))
        cb_tp_preset = ttk.Combobox(
            run_params,
            textvariable=var_tp_preset,
            values=["disabled", "latency", "default", "throughput", "aggressive", "custom"],
            width=12,
            state="readonly",
        )
        cb_tp_preset.grid(row=2, column=1, sticky="w", padx=(0, 10), pady=(6, 0), columnspan=2)

        ttk.Label(run_params, text="Streaming frames:").grid(row=3, column=0, sticky="e", padx=(0, 4), pady=(6, 0))
        ent_tp_frames = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_throughput_frames", None), width=6)
        ent_tp_frames.grid(row=3, column=1, sticky="w", padx=(0, 10), pady=(6, 0))

        ttk.Label(run_params, text="Warmup frames:").grid(row=3, column=2, sticky="e", padx=(0, 4), pady=(6, 0))
        ent_tp_warm = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_throughput_warmup_frames", None), width=6)
        ent_tp_warm.grid(row=3, column=3, sticky="w", padx=(0, 10), pady=(6, 0))

        ttk.Label(run_params, text="Queue depth:").grid(row=3, column=4, sticky="e", padx=(0, 4), pady=(6, 0))
        ent_tp_q = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_throughput_queue_depth", None), width=6)
        ent_tp_q.grid(row=3, column=5, sticky="w", padx=(0, 10), pady=(6, 0))

        total_runs_var = tk.StringVar(value="")
        lbl_total = ttk.Label(run_params, textvariable=total_runs_var, justify="left")
        lbl_total.grid(row=0, column=6, rowspan=4, sticky="w")

        _tp_sync = {"flag": False}
        _tp_service = getattr(app, "_benchmark_generation_service", BenchmarkGenerationService()) if app is not None else BenchmarkGenerationService()

        def _apply_streaming_preset(*_args):
            if _tp_sync["flag"]:
                return
            preset = str(var_tp_preset.get() or "default").strip().lower()
            if preset == "custom":
                _update_total_runs()
                return
            spec = _tp_service.resolve_streaming_preset(preset)
            _tp_sync["flag"] = True
            try:
                getattr(app, "var_remote_throughput_frames", None).set(str(int(spec.get("frames", 0))))
                getattr(app, "var_remote_throughput_warmup_frames", None).set(str(int(spec.get("warmup", 0))))
                getattr(app, "var_remote_throughput_queue_depth", None).set(str(int(spec.get("queue_depth", 1))))
            finally:
                _tp_sync["flag"] = False
            _update_total_runs()

        def _detect_streaming_preset(*_args):
            if _tp_sync["flag"]:
                return
            try:
                frames = int(str(getattr(app, "var_remote_throughput_frames", None).get()).strip())
                warm = int(str(getattr(app, "var_remote_throughput_warmup_frames", None).get()).strip())
                qd = int(str(getattr(app, "var_remote_throughput_queue_depth", None).get()).strip())
            except Exception:
                return
            detected = _tp_service.detect_streaming_preset(frames, warm, qd)
            _tp_sync["flag"] = True
            try:
                var_tp_preset.set(detected)
            finally:
                _tp_sync["flag"] = False

        def _safe_int(v) -> int:
            try:
                return int(str(v).strip())
            except Exception:
                return 0

        def _update_total_runs(*_):
            r = max(0, _safe_int(getattr(app, "var_remote_repeats", tk.StringVar(value="1")).get()))
            it = max(0, _safe_int(getattr(app, "var_remote_iters", tk.StringVar(value="1")).get()))
            warm = max(0, _safe_int(getattr(app, "var_remote_warmup", tk.StringVar(value="0")).get()))
            timeout_raw = str(getattr(app, "var_remote_timeout", tk.StringVar(value="7200")).get()).strip()
            tp_frames = max(0, _safe_int(getattr(app, "var_remote_throughput_frames", tk.StringVar(value="24")).get()))
            tp_warm = max(0, _safe_int(getattr(app, "var_remote_throughput_warmup_frames", tk.StringVar(value="6")).get()))
            tp_q = max(1, _safe_int(getattr(app, "var_remote_throughput_queue_depth", tk.StringVar(value="2")).get()))
            measured_total = r * it
            total_invocations = warm + measured_total
            timeout_desc = "off" if timeout_raw in {"", "0"} else f"{_safe_int(timeout_raw)}s"
            total_runs_var.set(
                f"Measured: {measured_total} · Total/run: {total_invocations}\nOuter timeout: {timeout_desc}\nStreaming: {tp_frames}f + {tp_warm} warm · q={tp_q}"
            )

        # Update preview live when the user edits repeats/warmup/runs.
        try:
            getattr(app, "var_remote_repeats", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_warmup", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_iters", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_timeout", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_throughput_frames", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_throughput_warmup_frames", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_throughput_queue_depth", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_throughput_frames", None).trace_add("write", _detect_streaming_preset)  # type: ignore
            getattr(app, "var_remote_throughput_warmup_frames", None).trace_add("write", _detect_streaming_preset)  # type: ignore
            getattr(app, "var_remote_throughput_queue_depth", None).trace_add("write", _detect_streaming_preset)  # type: ignore
            var_tp_preset.trace_add("write", _apply_streaming_preset)
        except Exception:
            # Older Tk versions may not have trace_add – ignore.
            pass
        _detect_streaming_preset()
        _update_total_runs()

        attach_tooltip(ent_rep, "Repeats multiplies the measured 'Runs'.")
        attach_tooltip(ent_warm, "Warmup runs are executed once per benchmark run and are not measured.")
        attach_tooltip(ent_runs, "Measured runs per benchmark case (will be multiplied by Repeats).")
        attach_tooltip(ent_timeout, "Outer timeout for the remote benchmark command in seconds. Use 0 to disable the outer timeout.")
        attach_tooltip(cb_tp_preset, "Praktische Presets für den zusätzlichen Streaming-/Interleaving-Lauf. disabled=aus, latency=kurzer Lauf mit geringer Queue, throughput/default/aggressive erhöhen Frames und Overlap. Eigene Änderungen setzen das Preset auf custom.")
        attach_tooltip(ent_tp_frames, "Measured streaming/interleaving frames for heterogene Stage1→Stage2-Pipelines. 0 deaktiviert den zusätzlichen Throughput-Messlauf.")
        attach_tooltip(ent_tp_warm, "Warmup-Frames vor der gemessenen Streaming-/Interleaving-Auswertung.")
        attach_tooltip(ent_tp_q, "Queue-Tiefe für den Streaming-/Interleaving-Lauf. Höher erlaubt mehr Overlap, kann aber Host-Speicher kosten.")
        attach_tooltip(lbl_total, "Measured = Repeats × Runs. Total/run = Warmup + measured runs. Streaming zeigt den zusätzlichen heterogenen Throughput-Lauf (Frames/Warmup/Queue-Tiefe).")

        ttk.Label(advanced_group, text="Remote venv:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(4, 0))
        ent_venv = ttk.Entry(advanced_group, textvariable=getattr(app, "var_remote_venv", None), width=55)
        ent_venv.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=(4, 0))
        attach_tooltip(
            ent_venv,
            "Optional: activate a venv / environment on the remote host before running the suite.\n"
            "If the activated env is missing core packages (onnx/onnxruntime), the tool falls back to the remote default python3 and appends the env site-packages to PYTHONPATH.\n"
            "Examples:\n"
            "  ~/hailo_py/bin/activate\n"
            "  source /opt/hailo/setup_env.sh",
        )

        ttk.Label(advanced_group, text="Extra args:").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(4, 8))
        ttk.Entry(advanced_group, textvariable=getattr(app, "var_remote_add_args", None), width=55).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=(4, 8)
        )

        btn_run = ttk.Button(remote_group, text="Run remote benchmark", command=getattr(app, "_remote_run_benchmark", None))
        btn_run.grid(row=3, column=0, columnspan=4, sticky="w", padx=(8, 8), pady=(4, 10))
        attach_tooltip(
            btn_run,
            "Bundle → scp upload → ssh run benchmark_suite.py → download results.\n"
            "Results are stored under: <OutputDir>/Results/<suite>/<repeat>/<run_id>/",
        )
        app.btn_remote_benchmark = btn_run

        _refresh_hosts()
    outer.content_frame = frame  # type: ignore[attr-defined]
    return outer
