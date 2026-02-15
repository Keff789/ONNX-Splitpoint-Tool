"""Validation/benchmark tab widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def _bool_var(app, name: str, default: bool) -> tk.BooleanVar:
    if app is None:
        return tk.BooleanVar(value=default)
    return getattr(app, name, tk.BooleanVar(value=default))


def _str_var(app, name: str, default: str) -> tk.StringVar:
    if app is None:
        return tk.StringVar(value=default)
    return getattr(app, name, tk.StringVar(value=default))


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)

    validation_group = ttk.LabelFrame(frame, text="ORT validation")
    validation_group.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

    var_split_validate = _bool_var(app, "var_split_validate", False)
    var_split_eps = _str_var(app, "var_split_eps", "1e-4")

    ttk.Checkbutton(
        validation_group,
        text="Validate split outputs (ORT)",
        variable=var_split_validate,
    ).pack(side=tk.LEFT, padx=(8, 10), pady=8)
    ttk.Label(validation_group, text="eps:").pack(side=tk.LEFT, padx=(0, 4))
    ttk.Entry(validation_group, textvariable=var_split_eps, width=10).pack(side=tk.LEFT, pady=8)

    runner_group = ttk.LabelFrame(frame, text="Runner")
    runner_group.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

    var_split_runner = _bool_var(app, "var_split_runner", True)
    var_runner_target = _str_var(app, "var_runner_target", "auto")

    ttk.Checkbutton(runner_group, text="Generate runner skeleton", variable=var_split_runner).pack(
        side=tk.LEFT, padx=(8, 10), pady=8
    )
    ttk.Label(runner_group, text="Runner target:").pack(side=tk.LEFT, padx=(0, 4))
    ttk.Combobox(
        runner_group,
        textvariable=var_runner_target,
        values=["auto", "cpu", "cuda", "tensorrt"],
        width=10,
        state="readonly",
    ).pack(side=tk.LEFT, pady=8)

    benchmark_group = ttk.LabelFrame(frame, text="Benchmark settings")
    benchmark_group.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    var_topk = _str_var(app, "var_topk", "10")
    var_min_gap = _str_var(app, "var_min_gap", "2")
    var_min_compute = _str_var(app, "var_min_compute", "0")

    ttk.Label(benchmark_group, text="Top-k:").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=8)
    ttk.Spinbox(benchmark_group, from_=1, to=1000, textvariable=var_topk, width=8).grid(
        row=0, column=1, sticky="w", padx=(0, 12), pady=8
    )
    ttk.Label(benchmark_group, text="Min gap:").grid(row=0, column=2, sticky="w", padx=(0, 4), pady=8)
    ttk.Spinbox(benchmark_group, from_=0, to=10000, textvariable=var_min_gap, width=8).grid(
        row=0, column=3, sticky="w", padx=(0, 12), pady=8
    )
    ttk.Label(benchmark_group, text="Min compute:").grid(row=0, column=4, sticky="w", padx=(0, 4), pady=8)
    ttk.Entry(benchmark_group, textvariable=var_min_compute, width=10).grid(
        row=0, column=5, sticky="w", padx=(0, 8), pady=8
    )

    return frame
