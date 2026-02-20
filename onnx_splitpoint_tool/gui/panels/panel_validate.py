"""Validation/benchmark tab widgets."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..widgets.tooltip import attach_tooltip


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

    runner_group = ttk.LabelFrame(frame, text="Runner")
    runner_group.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

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
    return frame
