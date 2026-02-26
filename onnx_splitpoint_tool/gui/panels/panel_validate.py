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

    # ------------------------- Remote Benchmark (SSH) -------------------------

    remote_group = ttk.LabelFrame(frame, text="Remote benchmark (SSH)")
    remote_group.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
    remote_group.columnconfigure(1, weight=1)

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
            width=55,
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
        cb_host = ttk.Combobox(remote_group, textvariable=host_disp, state="readonly", width=28)
        cb_host.grid(row=1, column=1, sticky="w", padx=(0, 6), pady=4)

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
        ttk.Label(remote_group, text="Transfer:").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=4)
        cb_transfer = ttk.Combobox(
            remote_group,
            textvariable=app.var_remote_transfer_mode,
            values=["bundle", "direct"],
            width=10,
            state="readonly",
        )
        cb_transfer.grid(row=2, column=1, sticky="w", padx=(0, 6), pady=4)
        attach_tooltip(
            cb_transfer,
            "bundle = pack suite as tar.gz (fast, cacheable)\n"
            "direct = scp -r full suite directory (useful for debugging; may be slower).",
        )

        chk_reuse = ttk.Checkbutton(
            remote_group,
            text="Reuse cached bundle",
            variable=app.var_remote_reuse_bundle,
        )
        chk_reuse.grid(row=2, column=2, columnspan=2, sticky="w", padx=(8, 8), pady=4)
        attach_tooltip(chk_reuse, "If enabled, re-pack only when suite files changed.")

        # Provider / run params
        ttk.Label(remote_group, text="Provider (override):").grid(row=3, column=0, sticky="w", padx=(8, 6), pady=4)
        cb_provider = ttk.Combobox(
            remote_group,
            textvariable=getattr(app, "var_remote_provider", None),
            values=["auto", "cpu", "cuda", "tensorrt", "openvino"],
            width=10,
            state="readonly",
        )
        cb_provider.grid(row=3, column=1, sticky="w", padx=(0, 6), pady=4)
        attach_tooltip(cb_provider, "auto = run benchmark_plan.json (all runs).\nOther values override to a single provider.")

        # Compact run controls (Repeats/Warmup/Runs) + a live "Total runs" preview.
        run_params = ttk.Frame(remote_group)
        run_params.grid(row=3, column=2, columnspan=2, sticky="e", padx=(8, 8), pady=4)

        ttk.Label(run_params, text="Repeats:").grid(row=0, column=0, sticky="e", padx=(0, 4))
        ent_rep = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_repeats", None), width=6)
        ent_rep.grid(row=0, column=1, sticky="w", padx=(0, 10))

        ttk.Label(run_params, text="Warmup:").grid(row=0, column=2, sticky="e", padx=(0, 4))
        ent_warm = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_warmup", None), width=6)
        ent_warm.grid(row=0, column=3, sticky="w", padx=(0, 10))

        ttk.Label(run_params, text="Runs:").grid(row=0, column=4, sticky="e", padx=(0, 4))
        ent_runs = ttk.Entry(run_params, textvariable=getattr(app, "var_remote_iters", None), width=6)
        ent_runs.grid(row=0, column=5, sticky="w", padx=(0, 10))

        total_runs_var = tk.StringVar(value="")
        lbl_total = ttk.Label(run_params, textvariable=total_runs_var)
        lbl_total.grid(row=0, column=6, sticky="w")

        def _safe_int(v) -> int:
            try:
                return int(str(v).strip())
            except Exception:
                return 0

        def _update_total_runs(*_):
            r = _safe_int(getattr(app, "var_remote_repeats", tk.StringVar(value="1")).get())
            it = _safe_int(getattr(app, "var_remote_iters", tk.StringVar(value="1")).get())
            total = max(0, r) * max(0, it)
            # Keep it short (this sits inline next to the entries)
            total_runs_var.set(f"Total: {total} (= {r}×{it})")

        # Update preview live when the user edits repeats/runs.
        try:
            getattr(app, "var_remote_repeats", None).trace_add("write", _update_total_runs)  # type: ignore
            getattr(app, "var_remote_iters", None).trace_add("write", _update_total_runs)  # type: ignore
        except Exception:
            # Older Tk versions may not have trace_add – ignore.
            pass
        _update_total_runs()

        attach_tooltip(ent_rep, "Repeats multiplies the measured 'Runs' (Total = Repeats × Runs).")
        attach_tooltip(ent_warm, "Warmup runs are not measured.")
        attach_tooltip(ent_runs, "Measured runs per benchmark case (will be multiplied by Repeats).")

        ttk.Label(remote_group, text="Extra args:").grid(row=4, column=0, sticky="w", padx=(8, 6), pady=(4, 8))
        ttk.Entry(remote_group, textvariable=getattr(app, "var_remote_add_args", None), width=55).grid(
            row=4, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=(4, 8)
        )

        btn_run = ttk.Button(remote_group, text="Run remote benchmark", command=getattr(app, "_remote_run_benchmark", None))
        btn_run.grid(row=5, column=0, columnspan=4, sticky="w", padx=(8, 8), pady=(0, 10))
        attach_tooltip(
            btn_run,
            "Bundle → scp upload → ssh run benchmark_suite.py → download results.\n"
            "Results are stored under: <OutputDir>/Results/<suite>/<repeat>/<run_id>/",
        )

        _refresh_hosts()
    return frame
