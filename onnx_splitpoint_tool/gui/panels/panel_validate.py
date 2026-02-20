"""Validation/benchmark tab widgets.

This tab owns the *benchmark set* generator UI (moved from the Analysis tab).

Note
----
Hailo HEF generation settings live in the **Split & Export** tab. The benchmark
set generator reuses those settings when building a suite.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..widgets.tooltip import attach_tooltip


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
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)

    # ----------------------------- Benchmark set -----------------------------

    bench_group = ttk.LabelFrame(frame, text="Benchmark set")
    bench_group.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
    bench_group.columnconfigure(6, weight=1)

    var_bench_topk = _str_var(app, "var_bench_topk", "20")

    btn_bench = ttk.Button(bench_group, text="Generate benchmark set…", command=getattr(app, "_generate_benchmark_set", None))
    btn_bench.grid(row=0, column=0, padx=(8, 10), pady=8, sticky="w")
    attach_tooltip(
        btn_bench,
        "Create a folder with multiple split cases (part1/part2 ONNX + manifests + runner scripts).\n"
        "Optional: also build Hailo HEFs for selected Hailo targets.",
    )

    # Rebind legacy state-machine reference (the Analysis tab no longer owns this button).
    if app is not None:
        app.btn_benchmark = btn_bench

    ttk.Label(bench_group, text="Use top N picks:").grid(row=0, column=1, sticky="w")
    ent_topk = ttk.Entry(bench_group, textvariable=var_bench_topk, width=6)
    ent_topk.grid(row=0, column=2, sticky="w", padx=(4, 12))
    attach_tooltip(ent_topk, "How many of the currently ranked picks should be exported into the benchmark set.")

    info = ttk.Label(
        bench_group,
        text="Hailo HEF generation is configured in the 'Split & Export' tab.",
    )
    info.grid(row=1, column=0, columnspan=7, sticky="w", padx=(8, 8), pady=(0, 8))

    # ---------------------- Accelerators to benchmark ----------------------

    acc_group = ttk.LabelFrame(frame, text="Accelerators to benchmark")
    acc_group.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
    acc_group.columnconfigure(10, weight=1)

    var_acc_cpu = _bool_var(app, "var_bench_acc_cpu", True)
    var_acc_cuda = _bool_var(app, "var_bench_acc_cuda", False)
    var_acc_trt = _bool_var(app, "var_bench_acc_tensorrt", False)
    var_acc_h8 = _bool_var(app, "var_bench_acc_hailo8", False)
    var_acc_h10 = _bool_var(app, "var_bench_acc_hailo10", False)

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

    # ----------------------------- Runner (ORT) ------------------------------

    runner_group = ttk.LabelFrame(frame, text="Runner")
    runner_group.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

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
    return frame
