"""Validation/benchmark tab widgets.

This tab owns the *benchmark set* generator UI (moved from the Analysis tab).

Note
----
Hailo HEF generation settings live in the **Split & Export** tab. The benchmark
set generator reuses those settings when building a suite.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

from ...workdir import ensure_workdir
from ..widgets.tooltip import attach_tooltip
from ..widgets.status_badge import StatusBadge
from ..widgets.collapsible_section import CollapsibleSection
from ...benchmark.services import BenchmarkGenerationService
from ...benchmark.classification_validation_presets import (
    list_available_presets,
    build_imagenet_validation_preset,
    classification_validation_default_root,
    resolve_classification_validation_source,
)
from ...benchmark.validation_assets import (
    validation_assets_status,
)
from ...benchmark.evaluation_profiles import (
    evaluation_profile_default_root,
    format_profile_comparison_text,
    list_available_evaluation_profiles,
    load_evaluation_profile,
    profile_brief_text,
    resolve_evaluation_profile,
    resolve_evaluation_profile_source,
    save_evaluation_profile_yaml,
    validate_evaluation_profile_payload,
)
from ...benchmark.model_preparation import (
    model_preparation_label,
    normalize_model_preparation_mode,
    preparation_result_is_selected_model,
)
from ...benchmark.validation_assets import (
    prepare_all_validation_assets,
    prepare_coco50,
    prepare_imagenette_mini_dataset,
    status_coco50,
    validation_assets_summary,
)


logger = logging.getLogger(__name__)


def _stage_label_from_run_stage(stage: object, fallback: str = "") -> str:
    if isinstance(stage, Mapping):
        stage_type = str(stage.get("type") or "").strip().lower()
        if stage_type == "onnxruntime":
            provider = str(stage.get("provider") or fallback or "onnxruntime").strip().lower()
            if provider in {"cuda", "gpu"}:
                return "cuda"
            if provider in {"tensorrt", "trt"}:
                return "tensorrt"
            if provider in {"cpu"}:
                return "cpu"
            return provider or "onnxruntime"
        if stage_type == "hailo":
            hw_arch = str(stage.get("hw_arch") or stage.get("arch") or stage.get("id") or fallback or "hailo").strip()
            return hw_arch or "hailo"
        if stage_type:
            return stage_type
    s = str(fallback or "").strip().lower()
    return s


def _pretty_plan_run_name(run: Mapping[str, Any]) -> str:
    run_id = str(run.get("id") or run.get("name") or "").strip()
    if run_id == "ort_cpu":
        return "ORT CPU"
    if run_id == "ort_cuda":
        return "ORT CUDA"
    if run_id == "ort_tensorrt":
        return "TensorRT"

    stage1 = _stage_label_from_run_stage(run.get("stage1"), fallback=str(run.get("provider") or run.get("hw_arch") or ""))
    stage2 = _stage_label_from_run_stage(run.get("stage2"), fallback=str(run.get("provider") or run.get("hw_arch") or ""))
    run_type = str(run.get("type") or "").strip().lower()
    if run_type == "hailo" and stage1 and stage2 and stage1 == stage2:
        return f"Hailo ({stage1})"
    if stage1 and stage2 and stage1 != stage2:
        left = "TensorRT" if stage1 == "tensorrt" else stage1
        right = "TensorRT" if stage2 == "tensorrt" else stage2
        return f"{left}→{right}"
    return run_id or "Run"


def _variants_for_plan_run(run: Mapping[str, Any]) -> list[str]:
    variants = [str(x).strip().lower() for x in list(run.get("variants") or []) if str(x).strip()]
    if variants:
        return variants
    run_type = str(run.get("type") or "").strip().lower()
    if run_type == "matrix":
        return ["part1", "part2", "composed"]
    if run_type == "hailo":
        return ["full", "composed"]
    return ["full", "part1", "part2", "composed"]


def _hef_req_for_labels(stage1: str, stage2: str, variants: Sequence[str]) -> list[str]:
    vset = {str(v).strip().lower() for v in variants if str(v).strip()}
    req: list[str] = []
    is_h1 = str(stage1).strip().lower().startswith("hailo")
    is_h2 = str(stage2).strip().lower().startswith("hailo")
    if "full" in vset and (is_h1 or is_h2):
        req.append("full")
    if is_h1 and ({"part1", "composed"} & vset):
        req.append("part1")
    if is_h2 and ({"part2", "composed"} & vset):
        req.append("part2")
    order = ["full", "part1", "part2"]
    return [x for x in order if x in req]


def _reference_label_for_plan_run(run: Mapping[str, Any]) -> str:
    mode = str(run.get("validation_reference_mode") or "auto").strip().lower()
    if mode and mode != "auto":
        return mode
    stage1 = _stage_label_from_run_stage(run.get("stage1"), fallback=str(run.get("provider") or run.get("hw_arch") or ""))
    stage2 = _stage_label_from_run_stage(run.get("stage2"), fallback=str(run.get("provider") or run.get("hw_arch") or ""))
    run_type = str(run.get("type") or "").strip().lower()
    if run_type == "matrix":
        return "cpu_full"
    if stage1 and stage2 and stage1 == stage2:
        return "same_backend_full"
    return "cpu_full"


def _load_actual_suite_plan_preview(bench_json_path: Path) -> Optional[dict[str, Any]]:
    p = Path(bench_json_path).expanduser()
    if p.is_dir():
        p = p / "benchmark_set.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read benchmark_set preview from %s", p)
        return None

    if not isinstance(payload, dict):
        return None
    plan = payload.get("plan") if isinstance(payload.get("plan"), Mapping) else {}
    runs = [dict(x) for x in list(plan.get("runs") or []) if isinstance(x, Mapping)]
    if not runs:
        return None

    hailo = payload.get("hailo") if isinstance(payload.get("hailo"), Mapping) else {}
    build = dict(hailo.get("build") or {}) if isinstance(hailo.get("build"), Mapping) else {}
    preflight = hailo.get("full_model_preflight") if isinstance(hailo.get("full_model_preflight"), Mapping) else {}
    dropped_runs = [str(x) for x in list(preflight.get("dropped_runs") or []) if str(x).strip()]
    adjusted_runs = [str(x) for x in list(preflight.get("adjusted_runs") or []) if str(x).strip()]
    blocked_by_hw_raw = preflight.get("blocked_by_hw") if isinstance(preflight.get("blocked_by_hw"), Mapping) else {}
    blocked_by_hw = {str(k): [str(x) for x in list(v or []) if str(x).strip()] for k, v in dict(blocked_by_hw_raw).items()}

    rows: list[tuple[str, str, str, str, list[str], str]] = []
    all_variants: set[str] = set()
    hef_needed: set[str] = set()
    hef_blocked: set[str] = set()

    for run in runs:
        variants = _variants_for_plan_run(run)
        stage1 = _stage_label_from_run_stage(run.get("stage1"), fallback=str(run.get("provider") or run.get("hw_arch") or ""))
        stage2 = _stage_label_from_run_stage(run.get("stage2"), fallback=str(run.get("provider") or run.get("hw_arch") or ""))
        reference = _reference_label_for_plan_run(run)
        req = _hef_req_for_labels(stage1, stage2, variants)
        for variant in variants:
            all_variants.add(str(variant).strip().lower())
        for kind in req:
            if kind in build and build.get(kind) is False:
                hef_blocked.add(kind)
            else:
                hef_needed.add(kind)
        req_parts: list[str] = []
        actual_req = [kind for kind in req if kind not in hef_blocked]
        blocked_req = [kind for kind in req if kind in hef_blocked]
        if actual_req:
            req_parts.append(f"hef: {', '.join(actual_req)}")
        if blocked_req:
            req_parts.append(f"blocked: {', '.join(blocked_req)}")
        if not req_parts:
            req_parts.append("onnx")
        rows.append((
            _pretty_plan_run_name(run),
            stage1 or "-",
            stage2 or "-",
            reference,
            list(variants),
            " · ".join(req_parts),
        ))

    order_cmp = ["full", "composed", "part1", "part2"]
    cmp_list = [v for v in order_cmp if v in all_variants]
    compare_bits = [f"Preview source: actual selected suite ({len(rows)} run{'s' if len(rows) != 1 else ''})"]
    if dropped_runs:
        compare_bits.append(f"dropped by preflight: {', '.join(dropped_runs[:6])}{' …' if len(dropped_runs) > 6 else ''}")
    elif adjusted_runs:
        compare_bits.append(f"adjusted by preflight: {len(adjusted_runs)}")
    if cmp_list:
        compare_bits.append(f"variants: {', '.join(cmp_list)}")
    compare_info = " · ".join(compare_bits)

    hef_bits: list[str] = []
    order_hef = ["full", "part1", "part2"]
    actual_hef = [h for h in order_hef if h in hef_needed]
    blocked_hef = [h for h in order_hef if h in hef_blocked or (h in build and build.get(h) is False)]
    if actual_hef:
        hef_bits.append(f"Suite Hailo HEFs (actual): {', '.join(actual_hef)}")
    else:
        hef_bits.append("Suite Hailo HEFs (actual): none")
    if blocked_hef:
        hef_bits.append(f"blocked by preflight: {', '.join(blocked_hef)}")
    if blocked_by_hw:
        hw_notes = []
        for hw, kinds in sorted(blocked_by_hw.items()):
            if kinds:
                hw_notes.append(f"{hw}→{', '.join(kinds)}")
        if hw_notes:
            hef_bits.append(f"blocked_by_hw: {'; '.join(hw_notes)}")
    eval_profile = payload.get("evaluation_profile") if isinstance(payload.get("evaluation_profile"), Mapping) else {}
    profile_info = ''
    if eval_profile:
        prof_id = str(eval_profile.get('profile_id') or eval_profile.get('requested') or '').strip()
        matched = str(eval_profile.get('matched_model_id') or '').strip()
        if prof_id:
            profile_info = f"Evaluation profile: {prof_id}" + (f" → {matched}" if matched else '')
    source_info = f"Using actual suite plan from {p.name} (post-preflight)"
    if profile_info:
        source_info += f" · {profile_info}"
    return {
        "rows": rows,
        "compare_info": compare_info,
        "hef_info": " · ".join(hef_bits),
        "source_info": source_info,
        "benchmark_set_path": str(p),
        "dropped_runs": list(dropped_runs),
        "adjusted_runs": list(adjusted_runs),
        "blocked_by_hw": dict(blocked_by_hw),
        "evaluation_profile": dict(eval_profile),
    }


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
    acc_group.columnconfigure(12, weight=1)

    var_acc_cpu = _bool_var(app, "var_bench_acc_cpu", True)
    var_acc_cuda = _bool_var(app, "var_bench_acc_cuda", False)
    var_acc_trt = _bool_var(app, "var_bench_acc_tensorrt", False)
    var_acc_h8 = _bool_var(app, "var_bench_acc_hailo8", False)
    var_acc_h10 = _bool_var(app, "var_bench_acc_hailo10", False)
    var_image_scale = _str_var(app, "var_bench_image_scale", "auto")
    var_validation_reference_mode = _str_var(app, "var_bench_validation_reference_mode", "auto")
    var_bench_task = _str_var(app, "var_bench_task", "auto")
    var_mini_coco_ap50 = _bool_var(app, "var_bench_mini_coco_ap50", False)
    var_mini_classification_eval = _bool_var(app, "var_bench_mini_classification_eval", False)

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
    ttk.Label(acc_group, text="Validation ref:").grid(row=0, column=8, sticky="e", padx=(18, 6), pady=8)
    cb_val_ref = ttk.Combobox(
        acc_group,
        textvariable=var_validation_reference_mode,
        values=["auto", "same_backend_full", "cpu_full", "annotations"],
        width=18,
        state="readonly",
    )
    cb_val_ref.grid(row=0, column=9, sticky="w", padx=(0, 8), pady=8)
    ttk.Label(acc_group, text="Task:").grid(row=0, column=10, sticky="e", padx=(8, 6), pady=8)
    cb_task = ttk.Combobox(
        acc_group,
        textvariable=var_bench_task,
        values=["auto", "detection", "classification"],
        width=16,
        state="readonly",
    )
    cb_task.grid(row=0, column=11, sticky="w", padx=(0, 8), pady=8)

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
    attach_tooltip(
        cb_val_ref,
        "Reference policy for detection validation. auto prefers same_backend_full for homogeneous runs and falls back to cpu_full for mixed runs. annotations keeps GT sidecars for dataset-only absolute checks.",
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

    def _classification_preset_root(preset: str) -> Optional[Path]:
        try:
            resolved = resolve_classification_validation_source(str(preset or ""))
            if resolved is None:
                return None
            rp = Path(resolved)
            if rp.is_file():
                return rp.parent.resolve()
            return rp.resolve()
        except Exception:
            logger.debug("Could not resolve classification preset root", exc_info=True)
            return None

    def _classification_preset_images_dir(preset: str) -> Optional[Path]:
        root = _classification_preset_root(preset)
        if root is None:
            return None
        images = root / "images"
        return images if images.is_dir() else root

    def _apply_classification_preset_defaults(preset: str, *, update_hailo_calib: bool = False) -> None:
        preset = str(preset or "").strip()
        if not preset:
            return
        var_bench_task.set("classification")
        var_validation_images.set(preset)
        if preset == "imagenet_val_mini_500":
            var_validation_max_images.set("500")
        else:
            var_validation_max_images.set("200")
        var_mini_classification_eval.set(True)
        try:
            cur_scale = str(var_image_scale.get() or "").strip().lower()
            if cur_scale in {"", "auto", "norm"}:
                var_image_scale.set("imagenet")
        except Exception:
            pass
        if update_hailo_calib:
            img_root = _classification_preset_images_dir(preset)
            if img_root is not None:
                calib_dir_var = _str_var(app, "var_hailo_hef_calib_dir", "")
                calib_count_var = _str_var(app, "var_hailo_hef_calib_count", "64")
                calib_bs_var = _str_var(app, "var_hailo_hef_calib_batch_size", "8")
                calib_dir_var.set(str(img_root))
                try:
                    if int(str(calib_count_var.get() or "0")) <= 0:
                        calib_count_var.set("64")
                except Exception:
                    calib_count_var.set("64")
                try:
                    if int(str(calib_bs_var.get() or "0")) <= 0:
                        calib_bs_var.set("8")
                except Exception:
                    calib_bs_var.set("8")

    btn_val_file = ttk.Button(acc_group, text="Datei…", command=_browse_validation_images_file)
    btn_val_file.grid(row=1, column=7, sticky="w", padx=(0, 6), pady=(0, 2))
    btn_val_dir = ttk.Button(acc_group, text="Ordner…", command=_browse_validation_images_dir)
    btn_val_dir.grid(row=1, column=8, sticky="w", padx=(0, 6), pady=(0, 2))
    ttk.Label(acc_group, text="Max:").grid(row=1, column=9, sticky="e", padx=(8, 4), pady=(0, 2))
    ent_val_max = ttk.Entry(acc_group, textvariable=var_validation_max_images, width=5)
    ent_val_max.grid(row=1, column=10, sticky="w", padx=(0, 8), pady=(0, 2))

    chk_mini_coco = ttk.Checkbutton(acc_group, text="Mini-COCO AP50", variable=var_mini_coco_ap50)
    chk_mini_coco.grid(row=2, column=8, columnspan=2, sticky="w", padx=(0, 8), pady=(0, 8))
    chk_mini_cls = ttk.Checkbutton(acc_group, text="Mini-Classification", variable=var_mini_classification_eval)
    chk_mini_cls.grid(row=2, column=10, columnspan=2, sticky="w", padx=(0, 8), pady=(0, 8))

    ttk.Label(acc_group, text="Classification preset:").grid(row=2, column=3, sticky="e", padx=(18, 6), pady=(0, 8))
    preset_values = [""] + list_available_presets()
    var_validation_preset = _str_var(app, "var_bench_validation_preset", "")
    cb_val_preset = ttk.Combobox(
        acc_group,
        textvariable=var_validation_preset,
        values=preset_values,
        width=24,
        state="readonly",
    )
    cb_val_preset.grid(row=2, column=4, columnspan=2, sticky="w", padx=(0, 6), pady=(0, 8))

    def _refresh_classification_preset_status() -> None:
        try:
            vals = [""] + list_available_presets()
            cb_val_preset.configure(values=vals)
        except Exception:
            pass
        preset = str(var_validation_preset.get() or var_validation_images.get() or "").strip()
        if preset in list_available_presets():
            root = _classification_preset_root(preset)
            if root is not None:
                lbl_cls_preset_status.configure(text=f"Preset ready: {preset} → {root}", foreground="#2b6b2b")
            else:
                lbl_cls_preset_status.configure(text=f"Preset not imported locally yet: {preset}", foreground="#9a6500")
        else:
            lbl_cls_preset_status.configure(text="ImageNet/Imagenette mini preset optional for classification", foreground="#666")

    def _on_validation_preset_selected(_event=None):
        preset = str(var_validation_preset.get() or "").strip()
        if not preset:
            _refresh_classification_preset_status()
            return
        _apply_classification_preset_defaults(preset, update_hailo_calib=False)
        _refresh_classification_preset_status()

    def _use_preset_for_hailo_calib():
        preset = str(var_validation_preset.get() or var_validation_images.get() or "").strip()
        if preset not in list_available_presets():
            messagebox.showinfo("Classification preset", "Bitte zuerst ein Classification-Preset auswählen oder importieren.")
            return
        root = _classification_preset_images_dir(preset)
        if root is None:
            messagebox.showwarning("Classification preset", f"Preset {preset} ist lokal noch nicht verfügbar. Bitte zuerst importieren.")
            return
        _apply_classification_preset_defaults(preset, update_hailo_calib=True)
        _refresh_classification_preset_status()
        messagebox.showinfo(
            "Hailo calibration",
            f"Hailo calibration dir wurde auf das Classification-Preset gesetzt:\n{root}\n\nDas vermeidet COCO-Bilder als Kalibrierquelle für Classification-Modelle.",
        )

    def _open_imagenet_preset_import_dialog():
        dlg = tk.Toplevel(acc_group)
        dlg.title("Import ImageNet mini preset")
        dlg.transient(acc_group.winfo_toplevel())
        dlg.grab_set()
        dlg.columnconfigure(1, weight=1)
        preset_var = tk.StringVar(value=str(var_validation_preset.get() or "imagenet_val_mini_200"))
        val_dir_var = tk.StringVar(value="")
        gt_file_var = tk.StringVar(value="")
        copy_mode_var = tk.StringVar(value="copy")
        overwrite_var = tk.BooleanVar(value=True)
        set_calib_var = tk.BooleanVar(value=True)
        status_var = tk.StringVar(value=f"Output: {classification_validation_default_root()}")

        ttk.Label(dlg, text="Preset:").grid(row=0, column=0, sticky="e", padx=(12, 6), pady=(12, 4))
        ttk.Combobox(dlg, textvariable=preset_var, values=list_available_presets(), state="readonly", width=28).grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(12, 4))

        ttk.Label(dlg, text="ImageNet val folder:").grid(row=1, column=0, sticky="e", padx=(12, 6), pady=4)
        ttk.Entry(dlg, textvariable=val_dir_var, width=64).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(dlg, text="Browse…", command=lambda: (lambda p: val_dir_var.set(p) if p else None)(filedialog.askdirectory(title="Select ILSVRC2012_img_val folder"))).grid(row=1, column=2, sticky="w", padx=(0, 12), pady=4)

        ttk.Label(dlg, text="Ground truth txt:").grid(row=2, column=0, sticky="e", padx=(12, 6), pady=4)
        ttk.Entry(dlg, textvariable=gt_file_var, width=64).grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(dlg, text="Browse…", command=lambda: (lambda p: gt_file_var.set(p) if p else None)(filedialog.askopenfilename(title="Select ILSVRC2012_validation_ground_truth.txt", filetypes=[("Text", "*.txt"), ("All", "*")]))).grid(row=2, column=2, sticky="w", padx=(0, 12), pady=4)

        opts = ttk.Frame(dlg)
        opts.grid(row=3, column=1, columnspan=2, sticky="w", padx=(0, 12), pady=4)
        ttk.Label(opts, text="Copy mode:").pack(side=tk.LEFT)
        ttk.Combobox(opts, textvariable=copy_mode_var, values=["copy", "symlink", "manifest-only"], state="readonly", width=14).pack(side=tk.LEFT, padx=(6, 12))
        ttk.Checkbutton(opts, text="Overwrite existing", variable=overwrite_var).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(opts, text="Use as Hailo calib", variable=set_calib_var).pack(side=tk.LEFT)

        ttk.Label(dlg, textvariable=status_var, foreground="#555", wraplength=760).grid(row=4, column=0, columnspan=3, sticky="ew", padx=12, pady=(6, 4))

        btns = ttk.Frame(dlg)
        btns.grid(row=5, column=0, columnspan=3, sticky="e", padx=12, pady=(8, 12))

        def _do_import():
            preset = str(preset_var.get() or "").strip()
            val_dir = Path(str(val_dir_var.get() or "")).expanduser()
            gt_file = Path(str(gt_file_var.get() or "")).expanduser()
            if not preset:
                messagebox.showwarning("Import ImageNet mini", "Bitte ein Preset auswählen.", parent=dlg)
                return
            if not val_dir.is_dir():
                messagebox.showwarning("Import ImageNet mini", "Bitte den ImageNet-val-Ordner auswählen.", parent=dlg)
                return
            if not gt_file.is_file():
                messagebox.showwarning("Import ImageNet mini", "Bitte die Ground-Truth-Textdatei auswählen.", parent=dlg)
                return
            for child in btns.winfo_children():
                try:
                    child.configure(state="disabled")
                except Exception:
                    pass
            status_var.set("Import läuft…")

            def _worker():
                try:
                    out = build_imagenet_validation_preset(
                        preset_name=preset,
                        imagenet_val_dir=val_dir,
                        ground_truth_file=gt_file,
                        copy_mode=str(copy_mode_var.get() or "copy"),
                        overwrite=bool(overwrite_var.get()),
                    )
                except Exception as exc:
                    dlg.after(0, lambda: _finish_import(None, exc))
                    return
                dlg.after(0, lambda: _finish_import(out, None))

            def _finish_import(out_path, exc):
                for child in btns.winfo_children():
                    try:
                        child.configure(state="normal")
                    except Exception:
                        pass
                if exc is not None:
                    status_var.set(f"Import failed: {type(exc).__name__}: {exc}")
                    messagebox.showerror("Import ImageNet mini", f"Import failed:\n{type(exc).__name__}: {exc}", parent=dlg)
                    return
                var_validation_preset.set(preset)
                _apply_classification_preset_defaults(preset, update_hailo_calib=bool(set_calib_var.get()))
                _refresh_classification_preset_status()
                status_var.set(f"Import fertig: {out_path}")
                messagebox.showinfo("Import ImageNet mini", f"Preset importiert:\n{out_path}", parent=dlg)
                try:
                    dlg.destroy()
                except Exception:
                    pass

            threading.Thread(target=_worker, daemon=True).start()

        ttk.Button(btns, text="Import", command=_do_import).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.RIGHT)
        try:
            dlg.geometry("900x260")
            dlg.update_idletasks()
        except Exception:
            pass

    def _open_prepare_validation_sets_dialog():
        dlg = tk.Toplevel(acc_group)
        dlg.title("Prepare validation sets")
        dlg.transient(acc_group.winfo_toplevel())
        dlg.grab_set()
        dlg.columnconfigure(0, weight=1)

        status = status_coco50()
        summary = validation_assets_summary()
        imagenette200_ready = bool(summary.get("imagenette200_ready"))
        imagenette500_ready = bool(summary.get("imagenette500_ready"))
        imagenette200_images = int(summary.get("imagenette200_images") or 0)
        imagenette500_images = int(summary.get("imagenette500_images") or 0)

        coco_var = tk.BooleanVar(value=not bool(status.ready))
        img200_var = tk.BooleanVar(value=not imagenette200_ready)
        img500_var = tk.BooleanVar(value=False)
        test_var = tk.BooleanVar(value=True)
        overwrite_var = tk.BooleanVar(value=False)

        header = (
            f"COCO-50 detection: {'ready' if status.ready else 'not ready'} · {status.note}\n"
            f"Location: {status.path}\n"
            "COCO images are downloaded from the public COCO val2017 image host; HTTP fallback is used if HTTPS certificates fail.\n\n"
            f"Imagenette mini-200 classification: {'ready' if imagenette200_ready else 'not ready'} · {imagenette200_images}/200 images\n"
            f"Location: {summary.get('imagenette200_dir', '')}\n"
            f"Imagenette mini-500 classification: {'ready' if imagenette500_ready else 'not ready'} · {imagenette500_images}/500 images\n"
            f"Location: {summary.get('imagenette500_dir', '')}\n"
            "Classification download uses fast.ai Imagenette2-320 mapped to ImageNet-1k class IDs. It is a public fallback, not the original ImageNet validation set.\n"
        )

        text = scrolledtext.ScrolledText(dlg, width=110, height=16, wrap=tk.WORD)
        text.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=12, pady=(12, 6))
        text.insert(tk.END, header + "\n")
        text.configure(state="disabled")
        dlg.rowconfigure(0, weight=1)

        opts = ttk.LabelFrame(dlg, text="Assets to prepare")
        opts.grid(row=1, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 6))
        opts.columnconfigure(1, weight=1)
        ttk.Checkbutton(opts, text="COCO-50 detection", variable=coco_var).grid(row=0, column=0, sticky="w", padx=(8, 12), pady=4)
        ttk.Checkbutton(opts, text="Imagenette mini-200 classification", variable=img200_var).grid(row=0, column=1, sticky="w", padx=(8, 12), pady=4)
        ttk.Checkbutton(opts, text="Imagenette mini-500 classification", variable=img500_var).grid(row=1, column=1, sticky="w", padx=(8, 12), pady=4)
        ttk.Checkbutton(opts, text="Runner test images", variable=test_var).grid(row=1, column=0, sticky="w", padx=(8, 12), pady=4)
        ttk.Checkbutton(opts, text="Overwrite existing files", variable=overwrite_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=(8, 12), pady=(4, 8))

        btns = ttk.Frame(dlg)
        btns.grid(row=2, column=0, columnspan=3, sticky="e", padx=12, pady=(6, 12))

        def _append(line: str):
            try:
                text.configure(state="normal")
                text.insert(tk.END, str(line).rstrip() + "\n")
                text.see(tk.END)
                text.configure(state="disabled")
            except Exception:
                pass

        def _start():
            selected_coco = bool(coco_var.get())
            selected_i200 = bool(img200_var.get())
            selected_i500 = bool(img500_var.get())
            selected_test = bool(test_var.get())
            if not any([selected_coco, selected_i200, selected_i500, selected_test]):
                messagebox.showwarning("Prepare validation sets", "Please select at least one asset set.", parent=dlg)
                return
            for child in btns.winfo_children():
                try:
                    child.configure(state="disabled")
                except Exception:
                    pass
            _append("Starting validation asset preparation…")
            if selected_i200 or selected_i500:
                _append("Note: Imagenette2-320 download is about 326 MB; only the selected mini subset is stored in the preset folder.")

            def _worker():
                try:
                    def _progress(msg: str):
                        dlg.after(0, lambda m=msg: _append(m))
                    result = prepare_all_validation_assets(
                        include_coco50=selected_coco,
                        include_imagenette200=selected_i200,
                        include_imagenette500=selected_i500,
                        include_test_images=selected_test,
                        overwrite=bool(overwrite_var.get()),
                        log=_progress,
                    )
                except Exception as exc:
                    dlg.after(0, lambda: _finish(None, exc))
                    return
                dlg.after(0, lambda: _finish(result, None))

            def _finish(result, exc):
                for child in btns.winfo_children():
                    try:
                        child.configure(state="normal")
                    except Exception:
                        pass
                if exc is not None:
                    _append(f"FAILED: {type(exc).__name__}: {exc}")
                    messagebox.showerror("Prepare validation sets", f"Preparation failed:\n{type(exc).__name__}: {exc}", parent=dlg)
                    return
                st = validation_assets_summary()
                _append("Done.")
                _append(f"COCO-50: {st.get('coco50_images', 0)}/50 images, {st.get('coco50_annotations', 0)}/50 annotation files")
                _append(f"Imagenette mini-200: {st.get('imagenette200_images', 0)}/200 images")
                _append(f"Imagenette mini-500: {st.get('imagenette500_images', 0)}/500 images")
                try:
                    _refresh_classification_preset_status()
                except Exception:
                    pass
                messagebox.showinfo("Prepare validation sets", "Validation assets are ready or updated.", parent=dlg)

            threading.Thread(target=_worker, daemon=True).start()

        ttk.Button(btns, text="Prepare selected", command=_start).pack(side=tk.RIGHT, padx=(8, 0))
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side=tk.RIGHT)
        try:
            dlg.geometry("980x470")
            dlg.update_idletasks()
        except Exception:
            pass

    cb_val_preset.bind("<<ComboboxSelected>>", _on_validation_preset_selected)
    btn_import_cls = ttk.Button(acc_group, text="Import…", command=_open_imagenet_preset_import_dialog)
    btn_import_cls.grid(row=2, column=6, sticky="w", padx=(0, 6), pady=(0, 8))
    btn_prepare_val = ttk.Button(acc_group, text="Prepare validation sets…", command=_open_prepare_validation_sets_dialog)
    btn_prepare_val.grid(row=2, column=7, sticky="w", padx=(0, 6), pady=(0, 8))
    btn_cls_calib = ttk.Button(acc_group, text="Use for Hailo calib", command=_use_preset_for_hailo_calib)
    btn_cls_calib.grid(row=3, column=7, sticky="w", padx=(0, 6), pady=(0, 4))
    lbl_cls_preset_status = ttk.Label(acc_group, text="ImageNet/Imagenette mini preset optional for classification", foreground="#666")
    lbl_cls_preset_status.grid(row=3, column=3, columnspan=4, sticky="w", padx=(18, 6), pady=(0, 4))
    attach_tooltip(btn_import_cls, "Importiert imagenet_val_mini_200/500 aus einem lokalen ImageNet-Validation-Ordner plus Ground-Truth-Datei. Die Bilder werden lokal unter ~/.onnx_splitpoint_tool/validation_datasets/classification abgelegt.")
    attach_tooltip(btn_prepare_val, "Bereitet herunterladbare Validation-Sets außerhalb der Tool-ZIP vor. COCO-50 Detection und downloadbares Imagenette-mini für Classification werden außerhalb der Tool-ZIP vorbereitet.")
    attach_tooltip(btn_cls_calib, "Setzt das Hailo-HEF-Kalibrierverzeichnis auf das ausgewählte Classification-Preset. Sinnvoll für ResNet/MobileNet/EfficientNet, damit nicht versehentlich COCO-Kalibrierbilder verwendet werden.")
    _refresh_classification_preset_status()

    lbl_val_default = ttk.Label(
        acc_group,
        text="Detection default if empty: prepared COCO-50 · Classification: local ImageNet mini preset or labeled dataset",
    )
    lbl_val_default.grid(row=7, column=4, columnspan=8, sticky="w", padx=(0, 6), pady=(0, 8))
    attach_tooltip(
        ent_val_images,
        "Optional image folder / image / text / JSON list for dataset-based semantic validation.\n"
        "Detection: empty means the locally prepared COCO-50 set.\n"
        "Classification: enter a labeled dataset path or a preset alias such as imagenet_val_mini_200.\n"
        "0 images = disabled. Recommended defaults: 50 for COCO-50, 200 for ImageNet mini.",
    )
    attach_tooltip(ent_val_max, "How many images from the semantic validation set should be used per case/run. Detection default: 50 from prepared COCO-50. Classification preset defaults: 200 or 500 depending on the selected ImageNet mini preset. Use 0 only if you explicitly want to disable dataset validation.")
    attach_tooltip(cb_val_preset, "Built-in classification preset aliases. They resolve to a locally imported dataset under ~/.onnx_splitpoint_tool/validation_datasets/classification and are copied into the suite for remote runs.")
    attach_tooltip(chk_mini_coco, "Optional report-only Mini-COCO AP50 on the semantic validation set. Reuses decoded detections + annotations, adds no extra inference passes, and never gates final_pass.")
    attach_tooltip(chk_mini_cls, "Optional report-only Mini-Classification Top-1/Top-5 report on the semantic validation set. Requires a labeled classification dataset or a local ImageNet-mini preset and never gates final_pass.")
    attach_tooltip(lbl_val_default, "Detection: a locally prepared COCO-50 set is used automatically when the field is empty. Use ‘Prepare validation sets…’ once after installing the clean release. Classification: if a local ImageNet-mini or downloadable Imagenette-mini preset is available, it becomes the default; otherwise provide a labeled dataset path.")

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
    var_hailo_full_model_preflight = _str_var(app, "var_hailo_full_model_preflight", "Enabled (plan-aware)")

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

    ttk.Label(acc_group, text="Full-model parser preflight:").grid(row=5, column=3, sticky="w", padx=(8, 6), pady=(0, 8))
    cb_full_preflight = ttk.Combobox(
        acc_group,
        textvariable=var_hailo_full_model_preflight,
        values=["Enabled (plan-aware)", "Disabled (always try full HEF)"],
        width=28,
        state="readonly",
    )
    cb_full_preflight.grid(row=5, column=4, sticky="w", padx=(0, 8), pady=(0, 8), columnspan=3)
    attach_tooltip(
        cb_full_preflight,
        "Control the suite-level parser preflight for the unsplit full Hailo model.\n\n"
        "Enabled (plan-aware): run the fast parser preflight first and allow it to adjust the benchmark plan.\n"
        "Disabled (always try full HEF): skip the parser preflight and still attempt the actual full-model HEF build later.\n"
        "Useful when you want final-evaluation comparability and prefer a real DFC build attempt over an early parser block.",
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

    def _effective_hailo_variants_for_validation() -> list[str]:
        variants = _variants_from_ui()
        ref_mode = (var_validation_reference_mode.get() or "auto").strip().lower()
        if ref_mode in ("auto", "same_backend_full") and "full" not in variants:
            return ["full", *variants]
        return variants

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

            variants = _effective_hailo_variants_for_validation()
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
                ref_mode = (var_validation_reference_mode.get() or "auto").strip().lower()
                ref_note = "same_backend_full" if ref_mode in ("auto", "same_backend_full") else ref_mode
                preflight_mode = (var_hailo_full_model_preflight.get() or "Enabled (plan-aware)").strip().lower()
                preflight_note = "preflight off" if preflight_mode.startswith("disabled") else "preflight on"
                var_hailo_bench_info.set(
                    f"Will benchmark (Hailo): {', '.join(variants)} · Reference: {ref_note} · Requires HEFs: {', '.join(req)} · {full_note} · {preflight_note}"
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
        var_hailo_full_model_preflight,
        var_validation_reference_mode,
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
        var_validation_reference_mode,
        var_acc_trt,
        var_acc_h8,
        var_acc_h10,
        var_hailo_full_hef_order,
        var_validation_reference_mode,
    ):
        try:
            v.trace_add("write", _update_matrix_ui)
        except Exception:
            pass
    _update_matrix_ui()

    # ------------------------ Evaluation profile ------------------------

    var_eval_profile = _str_var(app, "var_bench_evaluation_profile", "")
    var_eval_profile_info = _str_var(app, "var_bench_evaluation_profile_info", "No evaluation profile selected.")
    var_profile_models_root = _str_var(app, "var_bench_profile_models_root", "")
    var_profile_include_reserve = _bool_var(app, "var_bench_profile_include_reserve", False)
    var_profile_auto_remote = _bool_var(app, "var_bench_profile_auto_remote", False)
    var_profile_auto_analysis = _bool_var(app, "var_bench_profile_auto_analysis", False)
    var_profile_queue_info = _str_var(app, "var_bench_profile_queue_info", "Campaign queue inactive.")
    var_model_preparation_mode = _str_var(app, "var_bench_model_preparation_mode", "Use current ONNX")
    var_model_preparation_info = _str_var(app, "var_bench_model_preparation_info", "Uses the current ONNX unchanged.")

    ttk.Label(acc_group, text="Evaluation profile:").grid(row=10, column=0, sticky="w", padx=(8, 6), pady=(0, 6))
    cb_eval_profile = ttk.Combobox(
        acc_group,
        textvariable=var_eval_profile,
        values=[""] + list_available_evaluation_profiles(),
        width=28,
        state="readonly",
    )
    cb_eval_profile.grid(row=10, column=1, columnspan=2, sticky="w", padx=(0, 6), pady=(0, 6))

    def _current_model_path_for_profile() -> Optional[Path]:
        try:
            mp = (getattr(getattr(app, 'gui_state', None), 'current_model_path', None) or getattr(app, 'model_path', None) or '') if app is not None else ''
        except Exception:
            mp = ''
        mp = str(mp or '').strip()
        return Path(mp) if mp else None

    def _resolve_profile_path(req: str) -> Optional[Path]:
        req = str(req or '').strip()
        if not req:
            return None
        return resolve_evaluation_profile_source(req)

    def _refresh_eval_profile_info(*_args):
        req = str(var_eval_profile.get() or '').strip()
        if not req:
            var_eval_profile_info.set('No evaluation profile selected.')
            return
        try:
            resolved = resolve_evaluation_profile(req, model_path=_current_model_path_for_profile())
        except Exception as exc:
            logger.exception('Failed to resolve evaluation profile %s', req)
            var_eval_profile_info.set(f'Profile resolve failed: {type(exc).__name__}: {exc}')
            return
        if resolved is None:
            var_eval_profile_info.set(f'Profile not found: {req}')
            return
        var_eval_profile_info.set(profile_brief_text(resolved))

    def _apply_eval_profile_to_ui():
        req = str(var_eval_profile.get() or '').strip()
        if not req:
            _refresh_eval_profile_info()
            return
        try:
            resolved = resolve_evaluation_profile(req, model_path=_current_model_path_for_profile())
        except Exception as exc:
            messagebox.showerror('Evaluation profile', f'Profile resolve failed:\n{type(exc).__name__}: {exc}')
            return
        if resolved is None:
            messagebox.showwarning('Evaluation profile', f'Profile not found: {req}')
            _refresh_eval_profile_info()
            return
        if not resolved.matched:
            _refresh_eval_profile_info()
            messagebox.showinfo('Evaluation profile', 'Das aktuelle Modell gehört nicht zur ausgewählten Profil-Suite. Die Auswahl wird gespeichert, aber es wurden keine GUI-Werte überschrieben.')
            return
        ov = dict(resolved.overrides or {})
        try:
            var_bench_task.set(str(ov.get('benchmark_task') or var_bench_task.get() or 'auto'))
        except Exception:
            pass
        try:
            if str(ov.get('image_scale') or '').strip():
                var_image_scale.set(str(ov.get('image_scale')))
        except Exception:
            pass
        try:
            if 'validation_images' in ov:
                var_validation_images.set(str(ov.get('validation_images') or ''))
                if str(ov.get('validation_images') or '').strip() in list_available_presets():
                    var_validation_preset.set(str(ov.get('validation_images') or '').strip())
            if int(ov.get('validation_max_images') or 0) > 0:
                var_validation_max_images.set(str(int(ov.get('validation_max_images'))))
        except Exception:
            pass
        try:
            if str(ov.get('validation_reference_mode') or '').strip():
                var_validation_reference_mode.set(str(ov.get('validation_reference_mode')))
        except Exception:
            pass
        try:
            var_mini_coco_ap50.set(bool(ov.get('mini_coco_ap50')))
        except Exception:
            pass
        try:
            var_mini_classification_eval.set(bool(ov.get('mini_classification_eval')))
        except Exception:
            pass
        for vname, key in ((var_acc_cpu, 'acc_cpu'), (var_acc_cuda, 'acc_cuda'), (var_acc_trt, 'acc_trt'), (var_acc_h8, 'acc_h8'), (var_acc_h10, 'acc_h10')):
            try:
                if key in ov:
                    vname.set(bool(ov.get(key)))
            except Exception:
                pass
        try:
            if str(ov.get('hailo_preset') or '').strip():
                var_hailo_bench_preset.set(str(ov.get('hailo_preset')))
        except Exception:
            pass
        for vname, key in ((var_hailo_custom_full, 'hailo_custom_full'), (var_hailo_custom_composed, 'hailo_custom_composed'), (var_hailo_custom_part1, 'hailo_custom_part1'), (var_hailo_custom_part2, 'hailo_custom_part2')):
            try:
                if key in ov:
                    vname.set(bool(ov.get(key)))
            except Exception:
                pass
        try:
            if str(ov.get('full_model_preflight_policy') or '').strip():
                pol = str(ov.get('full_model_preflight_policy') or '').strip().lower()
                var_hailo_full_model_preflight.set('Disabled (always try full HEF)' if pol == 'skip' else 'Enabled (plan-aware)')
        except Exception:
            pass
        try:
            prep_mode = str(ov.get('model_preparation_mode') or '').strip()
            if prep_mode:
                var_model_preparation_mode.set(model_preparation_label(prep_mode))
        except Exception:
            pass
        try:
            trt_h = bool(ov.get('matrix_trt_to_hailo'))
            h_trt = bool(ov.get('matrix_hailo_to_trt'))
            var_matrix_trt_to_hailo.set(trt_h)
            var_matrix_hailo_to_trt.set(h_trt)
            if trt_h and h_trt:
                var_matrix_preset.set('TRT ↔ Hailo (split)')
            elif trt_h or h_trt:
                var_matrix_preset.set('Custom')
            else:
                var_matrix_preset.set('None')
        except Exception:
            pass
        try:
            if int(ov.get('requested_cases') or 0) > 0:
                _str_var(app, 'var_bench_topk', '').set(str(int(ov.get('requested_cases'))))
        except Exception:
            pass
        try:
            if int(ov.get('min_gap') or 0) >= 0 and hasattr(app, 'var_min_gap'):
                app.var_min_gap.set(str(int(ov.get('min_gap'))))
        except Exception:
            pass
        _refresh_classification_preset_status()
        _refresh_eval_profile_info()

    def _browse_eval_profile_file():
        p_sel = filedialog.askopenfilename(title='Select evaluation profile YAML', filetypes=[('YAML', '*.yaml *.yml'), ('All', '*')])
        if p_sel:
            var_eval_profile.set(str(p_sel))
            _refresh_eval_profile_info()

    def _validate_eval_profile_source(show_popup: bool = True) -> bool:
        req = str(var_eval_profile.get() or '').strip()
        if not req:
            if show_popup:
                messagebox.showwarning('Evaluation profile', 'No profile selected.')
            return False
        src = _resolve_profile_path(req)
        if src is None:
            if show_popup:
                messagebox.showerror('Evaluation profile', f'Profile not found: {req}')
            return False
        try:
            loaded = load_evaluation_profile(src, validate=True)
            if loaded is None or isinstance(loaded, tuple):
                raise FileNotFoundError(f'Profile not found: {src}')
        except Exception as exc:
            if show_popup:
                messagebox.showerror('Evaluation profile', f'Validation failed:\n\n{type(exc).__name__}: {exc}')
            return False
        if show_popup:
            raw = dict(loaded.raw_profile or {})
            name = str(raw.get('name') or loaded.profile_id).strip() or loaded.profile_id
            purpose = str(raw.get('purpose') or '').strip()
            msg = f'Profile schema validation passed.\n\nName: {name}\nSource: {loaded.profile_path}\nKind: {loaded.source}'
            if purpose:
                msg += f'\nPurpose: {purpose}'
            messagebox.showinfo('Evaluation profile', msg)
        return True

    def _open_eval_profile_editor():
        req = str(var_eval_profile.get() or '').strip()
        src = _resolve_profile_path(req)
        initial_text = ''
        initial_path: Optional[Path] = None
        if src is not None and src.exists():
            initial_path = src
            try:
                initial_text = src.read_text(encoding='utf-8')
            except Exception as exc:
                messagebox.showerror('Profile editor', f'Failed to read profile:\n\n{exc}')
                return
        elif req:
            initial_text = (
                f"name: {req}\n"
                "purpose: ''\n"
                "selection_policy: {}\n"
                "model_suite:\n"
                "  primary: []\n"
                "  reserve: []\n"
                "run_profiles: []\n"
                "validation: {}\n"
                "reporting: {}\n"
            )
        else:
            initial_text = (
                "name: new_profile_v1\n"
                "purpose: ''\n"
                "selection_policy: {}\n"
                "model_suite:\n"
                "  primary: []\n"
                "  reserve: []\n"
                "run_profiles: []\n"
                "validation: {}\n"
                "reporting: {}\n"
            )

        top = tk.Toplevel(outer)
        top.title('Evaluation profile editor')
        top.geometry('980x720')
        top.columnconfigure(0, weight=1)
        top.rowconfigure(1, weight=1)

        info_var = tk.StringVar(value=str(initial_path or 'Unsaved profile buffer'))
        ttk.Label(top, textvariable=info_var).grid(row=0, column=0, sticky='ew', padx=8, pady=(8, 4))
        txt = scrolledtext.ScrolledText(top, wrap='none', undo=True)
        txt.grid(row=1, column=0, sticky='nsew', padx=8, pady=(0, 8))
        txt.insert('1.0', initial_text)

        btns = ttk.Frame(top)
        btns.grid(row=2, column=0, sticky='ew', padx=8, pady=(0, 8))

        def _editor_parse_payload() -> Dict[str, Any]:
            text_buf = txt.get('1.0', 'end-1c')
            try:
                import yaml as _yaml
                payload = _yaml.safe_load(text_buf)
            except Exception as exc:
                raise ValueError(f'YAML parse failed: {exc}') from exc
            if payload is None:
                payload = {}
            if not isinstance(payload, Mapping):
                raise ValueError('Profile YAML root must be a mapping/object.')
            return dict(payload)

        def _editor_validate() -> bool:
            try:
                payload = _editor_parse_payload()
                validate_evaluation_profile_payload(payload, source=(str(initial_path) if initial_path is not None else 'unsaved_profile.yaml'))
            except Exception as exc:
                messagebox.showerror('Profile editor', f'Validation failed:\n\n{type(exc).__name__}: {exc}')
                return False
            messagebox.showinfo('Profile editor', 'Profile schema validation passed.')
            return True

        def _editor_open() -> None:
            nonlocal initial_path
            p_sel = filedialog.askopenfilename(title='Open evaluation profile YAML', filetypes=[('YAML', '*.yaml *.yml'), ('All', '*')])
            if not p_sel:
                return
            path = Path(p_sel)
            try:
                buf = path.read_text(encoding='utf-8')
            except Exception as exc:
                messagebox.showerror('Profile editor', f'Failed to open profile:\n\n{exc}')
                return
            txt.delete('1.0', 'end')
            txt.insert('1.0', buf)
            initial_path = path
            info_var.set(str(path))

        def _editor_write(target: Path) -> bool:
            nonlocal initial_path
            try:
                payload = _editor_parse_payload()
                out = save_evaluation_profile_yaml(target, payload, validate=True)
            except Exception as exc:
                messagebox.showerror('Profile editor', f'Save failed:\n\n{type(exc).__name__}: {exc}')
                return False
            initial_path = out
            info_var.set(str(out))
            var_eval_profile.set(str(out))
            _refresh_eval_profile_info()
            try:
                cb_eval_profile.configure(values=[''] + list_available_evaluation_profiles())
            except Exception:
                pass
            return True

        def _editor_save() -> None:
            if initial_path is None:
                _editor_save_as()
                return
            _editor_write(Path(initial_path))

        def _editor_save_as() -> None:
            p_sel = filedialog.asksaveasfilename(title='Save evaluation profile YAML', defaultextension='.yaml', filetypes=[('YAML', '*.yaml *.yml'), ('All', '*')])
            if not p_sel:
                return
            _editor_write(Path(p_sel))

        def _editor_use() -> None:
            if initial_path is None or not Path(initial_path).exists():
                if not messagebox.askyesno('Profile editor', 'The profile is not saved yet. Save it now?'):
                    return
                _editor_save_as()
                if initial_path is None or not Path(initial_path).exists():
                    return
            var_eval_profile.set(str(initial_path))
            _refresh_eval_profile_info()
            top.destroy()

        ttk.Button(btns, text='Open…', command=_editor_open).pack(side='left')
        ttk.Button(btns, text='Validate', command=_editor_validate).pack(side='left', padx=(6, 0))
        ttk.Button(btns, text='Save', command=_editor_save).pack(side='left', padx=(6, 0))
        ttk.Button(btns, text='Save As…', command=_editor_save_as).pack(side='left', padx=(6, 0))
        ttk.Button(btns, text='Use', command=_editor_use).pack(side='left', padx=(12, 0))
        ttk.Button(btns, text='Close', command=top.destroy).pack(side='right')

    def _browse_profile_models_root():
        initial = str(var_profile_models_root.get() or (app.default_output_dir if app is not None else '') or os.getcwd())
        picked = filedialog.askdirectory(title='Select model root for profile campaign', initialdir=initial)
        if picked:
            var_profile_models_root.set(str(picked))

    def _refresh_profile_queue_info(*_args):
        root = str(var_profile_models_root.get() or '').strip()
        reserve = bool(var_profile_include_reserve.get())
        auto_remote = bool(var_profile_auto_remote.get())
        auto_analysis = bool(var_profile_auto_analysis.get())
        prep_mode = normalize_model_preparation_mode(var_model_preparation_mode.get())
        prep_label = model_preparation_label(prep_mode)
        if not root:
            var_profile_queue_info.set('Campaign queue inactive. Select a model root to enable batch processing for the profile.')
            return
        desc = f'Model root: {root} · reserve={reserve} · remote={auto_remote} · analysis={auto_analysis} · prep={prep_label}'
        if auto_analysis and not auto_remote:
            desc += ' · note: auto analysis runs after remote benchmarks only'
        var_profile_queue_info.set(desc)

    def _refresh_model_preparation_info(*_args):
        prep_mode = normalize_model_preparation_mode(var_model_preparation_mode.get())
        current_model = _current_model_path_for_profile()
        if prep_mode == 'current':
            var_model_preparation_info.set('Uses the current ONNX unchanged. Good default for classifiers and already screened models.')
            return
        if current_model is None:
            var_model_preparation_info.set('Auto-screen YOLO full-Hailo is enabled. Load a model and use “Prepare current model…” or run a profile campaign.')
            return
        if preparation_result_is_selected_model(current_model):
            var_model_preparation_info.set('Current ONNX already comes from a preparation-screening result. Re-run Analyse on this prepared model before generating the benchmark set.')
            return
        var_model_preparation_info.set('Manual use: click “Prepare current model…” before Analyse. Profile campaigns run preparation automatically as the first queue stage.')

    ttk.Button(acc_group, text='Profile YAML…', command=_browse_eval_profile_file).grid(row=10, column=3, sticky='w', padx=(0, 6), pady=(0, 6))
    ttk.Button(acc_group, text='Apply', command=_apply_eval_profile_to_ui).grid(row=10, column=4, sticky='w', padx=(0, 6), pady=(0, 6))
    ttk.Button(acc_group, text='Edit…', command=_open_eval_profile_editor).grid(row=10, column=5, sticky='w', padx=(0, 6), pady=(0, 6))
    ttk.Button(acc_group, text='Validate', command=_validate_eval_profile_source).grid(row=10, column=6, sticky='w', padx=(0, 6), pady=(0, 6))

    lbl_eval_profile = ttk.Label(acc_group, textvariable=var_eval_profile_info, foreground='#555')
    lbl_eval_profile.grid(row=11, column=0, columnspan=11, sticky='w', padx=(8, 8), pady=(0, 4))

    ttk.Label(acc_group, text='Profile model root:').grid(row=12, column=0, sticky='w', padx=(8, 6), pady=(0, 6))
    ent_profile_root = ttk.Entry(acc_group, textvariable=var_profile_models_root, width=56)
    ent_profile_root.grid(row=12, column=1, columnspan=3, sticky='ew', padx=(0, 6), pady=(0, 6))
    ttk.Button(acc_group, text='Browse…', command=_browse_profile_models_root).grid(row=12, column=4, sticky='w', padx=(0, 6), pady=(0, 6))
    ttk.Button(acc_group, text='Queue profile…', command=getattr(app, '_queue_profile_campaign', None)).grid(row=12, column=5, sticky='w', padx=(0, 6), pady=(0, 6))

    chk_profile_reserve = ttk.Checkbutton(acc_group, text='Include reserve', variable=var_profile_include_reserve)
    chk_profile_reserve.grid(row=13, column=1, sticky='w', padx=(0, 8), pady=(0, 4))
    chk_profile_remote = ttk.Checkbutton(acc_group, text='Auto remote run', variable=var_profile_auto_remote)
    chk_profile_remote.grid(row=13, column=2, sticky='w', padx=(0, 8), pady=(0, 4))
    chk_profile_analysis = ttk.Checkbutton(acc_group, text='Auto export analysis', variable=var_profile_auto_analysis)
    chk_profile_analysis.grid(row=13, column=3, sticky='w', padx=(0, 8), pady=(0, 4))

    lbl_profile_queue = ttk.Label(acc_group, textvariable=var_profile_queue_info, foreground='#555')
    lbl_profile_queue.grid(row=14, column=0, columnspan=11, sticky='w', padx=(8, 8), pady=(0, 8))

    prep_section = CollapsibleSection(acc_group, 'Model preparation', expanded=False)
    prep_section.grid(row=15, column=0, columnspan=11, sticky='ew', padx=(8, 8), pady=(0, 8))
    try:
        prep_section.body.columnconfigure(1, weight=1)
    except Exception:
        pass
    ttk.Label(prep_section.body, text='Preparation mode:').grid(row=0, column=0, sticky='w', padx=(0, 6), pady=(6, 4))
    cb_prep_mode = ttk.Combobox(
        prep_section.body,
        textvariable=var_model_preparation_mode,
        values=['Use current ONNX', 'Auto-screen YOLO full-Hailo'],
        width=28,
        state='readonly',
    )
    cb_prep_mode.grid(row=0, column=1, sticky='w', padx=(0, 6), pady=(6, 4))
    ttk.Button(prep_section.body, text='Prepare current model…', command=getattr(app, '_queue_prepare_current_model', None)).grid(row=0, column=2, sticky='w', padx=(0, 6), pady=(6, 4))
    lbl_prep_info = ttk.Label(prep_section.body, textvariable=var_model_preparation_info, foreground='#555')
    lbl_prep_info.grid(row=1, column=0, columnspan=3, sticky='w', padx=(0, 6), pady=(0, 6))

    attach_tooltip(cb_eval_profile, 'Versionierte Evaluationsprofile setzen eine feste Run-Matrix und task-spezifische Defaults für die finale Kampagne. Die eigentlichen Cases werden weiter normal generiert; das Profil überschreibt nur die Plan-/Validierungsdefaults.')
    attach_tooltip(lbl_eval_profile, 'Zeigt, ob das aktuelle Modell von der ausgewählten Profil-Suite abgedeckt wird und welche Run-Matrix/Defaults daraus abgeleitet werden.')
    attach_tooltip(ent_profile_root, 'Root folder containing the exported ONNX models referenced by the selected profile. The profile campaign scans this folder recursively and matches models by export metadata / file name.')
    attach_tooltip(lbl_profile_queue, 'The profile campaign uses the existing Jobs infrastructure as a sequential queue: optional model preparation → analysis → benchmark-set generation → optional remote execution → optional automatic benchmark-analysis export.')
    attach_tooltip(cb_prep_mode, 'Keeps the everyday UI simple. “Use current ONNX” does nothing extra. “Auto-screen YOLO full-Hailo” is the compact advanced mode: it tries the current ONNX first, then a tiny built-in set of Ultralytics export variants until one passes the full-Hailo probe.')
    attach_tooltip(lbl_prep_info, 'Manual workflow: use “Prepare current model…” first, then rerun Analyse on the selected prepared ONNX. Profile campaigns perform this stage automatically before analysis.')
    try:
        var_eval_profile.trace_add('write', _refresh_eval_profile_info)
    except Exception:
        pass
    for _v in (var_profile_models_root, var_profile_include_reserve, var_profile_auto_remote, var_profile_auto_analysis, var_model_preparation_mode):
        try:
            _v.trace_add('write', _refresh_profile_queue_info)
        except Exception:
            pass
    try:
        var_model_preparation_mode.trace_add('write', _refresh_model_preparation_info)
    except Exception:
        pass
    _refresh_eval_profile_info()
    _refresh_profile_queue_info()
    _refresh_model_preparation_info()
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
    var_accy_plan_source_info = _str_var(app, "var_accuracy_plan_source_info", "")

    lbl_compare = ttk.Label(accy_group, textvariable=var_accy_compare_info)
    lbl_compare.grid(row=1, column=0, columnspan=2, sticky="w", padx=(8, 8), pady=(0, 2))

    lbl_hef = ttk.Label(accy_group, textvariable=var_accy_hef_info)
    lbl_hef.grid(row=2, column=0, columnspan=2, sticky="w", padx=(8, 8), pady=(0, 2))

    lbl_plan_source = ttk.Label(accy_group, textvariable=var_accy_plan_source_info)
    lbl_plan_source.grid(row=3, column=0, columnspan=2, sticky="w", padx=(8, 8), pady=(0, 8))
    attach_tooltip(
        lbl_hef,
        "Derived automatically from your selected benchmark variants (Phase 6).\n"
        "Full -> full HEF, part1 -> part1 HEF, part2 -> part2 HEF, composed -> part1+part2 HEFs.",
    )

    # Planned validations table (computed from current Benchmark tab selections).
    table_frame = ttk.Frame(accy_group)
    table_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=(8, 8), pady=(0, 8))
    table_frame.columnconfigure(0, weight=1)

    _cols = ("run", "stage1", "stage2", "reference", "variants", "requires")
    tv_plan = ttk.Treeview(table_frame, columns=_cols, show="headings", height=6)
    tv_plan.grid(row=0, column=0, sticky="ew")

    vsb_plan = ttk.Scrollbar(table_frame, orient="vertical", command=tv_plan.yview)
    vsb_plan.grid(row=0, column=1, sticky="ns")
    tv_plan.configure(yscrollcommand=vsb_plan.set)

    tv_plan.heading("run", text="Run")
    tv_plan.heading("stage1", text="Stage1")
    tv_plan.heading("stage2", text="Stage2")
    tv_plan.heading("reference", text="Reference")
    tv_plan.heading("variants", text="Variants compared")
    tv_plan.heading("requires", text="Requires")

    tv_plan.column("run", width=160, stretch=False)
    tv_plan.column("stage1", width=110, stretch=False)
    tv_plan.column("stage2", width=110, stretch=False)
    tv_plan.column("reference", width=150, stretch=False)
    tv_plan.column("variants", width=240, stretch=True)
    tv_plan.column("requires", width=160, stretch=False)

    attach_tooltip(
        tv_plan,
        "Planned accuracy comparisons with the selected validation reference policy.\n"
        "Homogeneous runs prefer same_backend_full; mixed runs fall back to cpu_full unless you explicitly choose another mode.",
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
            planned: list[tuple[str, str, str, str, list[str], str]] = []
            suite_preview: Optional[dict[str, Any]] = None
            suite_path = ""
            try:
                suite_path = str(getattr(app, "var_remote_benchmark_set", None).get() or "").strip() if app is not None else ""
            except Exception:
                suite_path = ""
            if suite_path:
                try:
                    suite_preview = _load_actual_suite_plan_preview(Path(suite_path))
                except Exception:
                    logger.exception("Failed to load actual suite plan preview from %s", suite_path)
                    suite_preview = None

            if suite_preview:
                planned = [
                    (
                        str(name),
                        str(s1),
                        str(s2),
                        str(ref_mode),
                        list(vv),
                        str(reqtxt),
                    )
                    for name, s1, s2, ref_mode, vv, reqtxt in list(suite_preview.get("rows") or [])
                ]
                var_accy_compare_info.set(str(suite_preview.get("compare_info") or ""))
                var_accy_hef_info.set(str(suite_preview.get("hef_info") or ""))
                var_accy_plan_source_info.set(str(suite_preview.get("source_info") or ""))
            else:
                ort_variants = ["full", "part1", "part2", "composed"]
                if bool(var_acc_cpu.get()):
                    planned.append(("ORT CPU", "cpu", "cpu", "same_backend_full", ort_variants, "onnx"))
                if bool(var_acc_cuda.get()):
                    planned.append(("ORT CUDA", "cuda", "cuda", "same_backend_full", ort_variants, "onnx"))
                if bool(var_acc_trt.get()):
                    planned.append(("TensorRT", "tensorrt", "tensorrt", "same_backend_full", ort_variants, "onnx"))

                hailo_targets: list[str] = []
                if bool(var_acc_h8.get()):
                    hailo_targets.append(_get_hailo_hw("hailo8"))
                if bool(var_acc_h10.get()):
                    hailo_targets.append(_get_hailo_hw("hailo10"))

                hailo_variants = _effective_hailo_variants_for_validation()
                for hw in hailo_targets:
                    s1 = hw
                    s2 = hw
                    req = _hef_req_for(s1, s2, hailo_variants)
                    req_txt = f"hef: {', '.join(req)}" if req else "hef: none"
                    planned.append((f"Hailo ({hw})", s1, s2, "same_backend_full", list(hailo_variants), req_txt))

                matrix_variants = ["full", "part1", "part2", "composed"]
                if bool(var_matrix_trt_to_hailo.get()):
                    for hw in hailo_targets:
                        s1 = "tensorrt"
                        s2 = hw
                        req = _hef_req_for(s1, s2, matrix_variants)
                        req_txt = f"hef: {', '.join(req)}" if req else "hef: none"
                        planned.append((f"TensorRT→{hw}", s1, s2, "cpu_full", list(matrix_variants), req_txt))

                if bool(var_matrix_hailo_to_trt.get()):
                    for hw in hailo_targets:
                        s1 = hw
                        s2 = "tensorrt"
                        req = _hef_req_for(s1, s2, matrix_variants)
                        req_txt = f"hef: {', '.join(req)}" if req else "hef: none"
                        planned.append((f"{hw}→TensorRT", s1, s2, "cpu_full", list(matrix_variants), req_txt))

                union_variants: set[str] = set()
                hef_needed: set[str] = set()
                for _name, s1, s2, _ref, vv, _reqtxt in planned:
                    for v in vv:
                        union_variants.add(str(v).strip().lower())
                    for h in _hef_req_for(s1, s2, vv):
                        hef_needed.add(h)

                order_cmp = ["full", "composed", "part1", "part2"]
                cmp_list = [v for v in order_cmp if v in union_variants]
                ref_mode_ui = (var_validation_reference_mode.get() or "auto").strip().lower()
                if not cmp_list:
                    var_accy_compare_info.set("Will compare: none (no accelerators selected)")
                elif ref_mode_ui in ("auto", "same_backend_full"):
                    var_accy_compare_info.set(
                        f"Will compare: homogeneous runs vs same_backend_full; mixed runs fallback to cpu_full ({', '.join(cmp_list)})"
                    )
                elif ref_mode_ui == "cpu_full":
                    var_accy_compare_info.set(f"Will compare (against CPU full): {', '.join(cmp_list)}")
                else:
                    var_accy_compare_info.set(
                        f"Semantic GT mode: dataset validation uses annotations when available; split-fidelity stays model-based ({', '.join(cmp_list)})"
                    )

                order_hef = ["full", "part1", "part2"]
                hef_list = [h for h in order_hef if h in hef_needed]
                if hef_list:
                    var_accy_hef_info.set(f"Hailo HEFs needed (auto): {', '.join(hef_list)}")
                else:
                    var_accy_hef_info.set("Hailo HEFs needed (auto): none")
                var_accy_plan_source_info.set("Preview source: current GUI selection (pre-generation)")

            for item in tv_plan.get_children():
                tv_plan.delete(item)
            for name, s1, s2, ref_mode, vv, reqtxt in planned:
                tv_plan.insert("", "end", values=(name, s1, s2, ref_mode, ", ".join(vv), reqtxt))
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
        var_validation_reference_mode,
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

    try:
        _suite_var = getattr(app, "var_remote_benchmark_set", None) if app is not None else None
        if _suite_var is not None:
            _suite_var.trace_add("write", _update_accuracy_ui)
    except Exception:
        pass
    if app is not None:
        app._benchmark_refresh_accuracy_ui = _update_accuracy_ui

    _update_accuracy_ui()
    for _maybe_var in (var_acc_h8, var_acc_h10, getattr(app, "var_strict_boundary", None) if app is not None else None):
        try:
            if _maybe_var is not None:
                _maybe_var.trace_add("write", _refresh_hailo_compile_outlook)
        except Exception:
            pass

    def _schedule_hailo_compile_outlook_refresh(delay_ms: int = 250):
        if app is None or not hasattr(app, "after"):
            _refresh_hailo_compile_outlook()
            return
        try:
            app.after(delay_ms, _refresh_hailo_compile_outlook)
            logger.info("Benchmark Hailo compile outlook refresh scheduled after %dms", int(delay_ms))
        except Exception:
            logger.debug("Failed to schedule Hailo compile outlook refresh; falling back to immediate refresh", exc_info=True)
            _refresh_hailo_compile_outlook()

    _schedule_hailo_compile_outlook_refresh()

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
            t0 = time.perf_counter()
            try:
                wd = ensure_workdir(Path(root))
                suite_root = wd.benchmark_sets
                hits: list[str] = []
                if not suite_root.exists():
                    return []
                for dirpath, dirnames, filenames in os.walk(suite_root):
                    dirnames[:] = [
                        d for d in dirnames
                        if d not in {"__pycache__", ".git", ".hg", ".svn"} and not str(d).startswith('.')
                    ]
                    if "benchmark_set.json" in filenames:
                        hits.append(str(Path(dirpath) / "benchmark_set.json"))
                        # benchmark_set.json marks the suite root; do not descend into large case trees.
                        dirnames[:] = []
                hits = sorted(dict.fromkeys(hits))
                logger.info("Discovered %d benchmark suites in %.3fs under %s", len(hits), time.perf_counter() - t0, suite_root)
                return hits
            except Exception:
                logger.exception("Failed to discover benchmark suites")
                return []

        try:
            _selected_suite = str(getattr(app, "var_remote_benchmark_set", None).get() or "").strip()
        except Exception:
            _selected_suite = ""
        _initial_suite_values = [_selected_suite] if _selected_suite else []

        cb_suite = ttk.Combobox(
            remote_group,
            textvariable=getattr(app, "var_remote_benchmark_set", None),
            values=_initial_suite_values,
            width=50,
        )
        cb_suite.grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=(8, 4))

        suite_scan_status_var = tk.StringVar(value=("Suite discovery pending…" if getattr(app, "default_output_dir", None) else "Set Working Dir to enable suite discovery."))
        lbl_suite_scan = ttk.Label(remote_group, textvariable=suite_scan_status_var)
        lbl_suite_scan.grid(row=4, column=1, columnspan=3, sticky="w", padx=(0, 6), pady=(0, 4))

        _suite_scan = {"thread": None, "request_id": 0, "values": list(_initial_suite_values), "scanned": False}

        def _apply_suite_values(values: list[str], *, elapsed_s: float | None = None):
            merged: list[str] = []
            current = ""
            try:
                current = str(getattr(app, "var_remote_benchmark_set", None).get() or "").strip()
            except Exception:
                current = ""
            if current:
                merged.append(current)
            for item in values:
                s = str(item).strip()
                if s and s not in merged:
                    merged.append(s)
            cb_suite["values"] = merged
            _suite_scan["values"] = list(merged)
            _suite_scan["scanned"] = True
            if elapsed_s is None:
                suite_scan_status_var.set(f"Suite list ready ({len(merged)}).")
            else:
                suite_scan_status_var.set(f"Suite list ready ({len(merged)}) in {elapsed_s:.2f}s.")

        def _finish_suite_scan(request_id: int, values: list[str], elapsed_s: float):
            if int(_suite_scan.get("request_id", 0)) != int(request_id):
                return
            _suite_scan["thread"] = None
            _apply_suite_values(values, elapsed_s=elapsed_s)

        def _start_suite_scan(*, force: bool = False):
            root = getattr(app, "default_output_dir", None)
            if not root:
                suite_scan_status_var.set("Set Working Dir to enable suite discovery.")
                return
            thread = _suite_scan.get("thread")
            if isinstance(thread, threading.Thread) and thread.is_alive():
                if force:
                    suite_scan_status_var.set("Suite discovery already running…")
                return
            cached = list(_suite_scan.get("values") or [])
            scanned = bool(_suite_scan.get("scanned"))
            if cached and scanned and not force:
                _apply_suite_values(cached)
                return
            _suite_scan["request_id"] = int(_suite_scan.get("request_id", 0)) + 1
            request_id = int(_suite_scan["request_id"])
            suite_scan_status_var.set("Scanning BenchmarkSets in background…")

            def _worker():
                t0 = time.perf_counter()
                values = _list_suites()
                elapsed_s = time.perf_counter() - t0
                try:
                    app.after(0, lambda: _finish_suite_scan(request_id, values, elapsed_s))
                except Exception:
                    logger.debug("Failed to post suite scan result back to Tk thread", exc_info=True)

            thread = threading.Thread(target=_worker, name="osp-suite-scan", daemon=True)
            _suite_scan["thread"] = thread
            thread.start()
            logger.info("Benchmark suite discovery started in background")

        def _refresh_suite_values():
            _start_suite_scan(force=True)

        # Re-scan right before the dropdown opens, but keep startup non-blocking.
        try:
            cb_suite.configure(postcommand=_refresh_suite_values)
        except Exception:
            pass
        try:
            cb_suite.bind("<Button-1>", lambda _event: _start_suite_scan(force=False), add="+")
        except Exception:
            pass
        try:
            app.after(300, _start_suite_scan)
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
            "Choose an existing suite from the Working Dir (BenchmarkSets) or paste a path to any benchmark_set.json.\n"
            "Discovery now runs in the background and no longer blocks GUI startup.",
        )
        attach_tooltip(lbl_suite_scan, "Background status for benchmark suite discovery inside the Working Dir.")

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
        btn_run.grid(row=5, column=0, columnspan=4, sticky="w", padx=(8, 8), pady=(4, 10))
        attach_tooltip(
            btn_run,
            "Bundle → scp upload → ssh run benchmark_suite.py → download results.\n"
            "Results are stored under: <OutputDir>/Results/<suite>/<repeat>/<run_id>/",
        )
        app.btn_remote_benchmark = btn_run

        _refresh_hosts()
    outer.content_frame = frame  # type: ignore[attr-defined]
    return outer
