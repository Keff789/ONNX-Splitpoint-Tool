#!/usr/bin/env python3
"""analyse_and_split_gui.py

Interactive GUI for analysing ONNX models and suggesting single-split boundaries.

Highlights
----------
- Enumerates all graph boundaries (topological order).
- Estimates activation bytes crossing each boundary (Comm(b)).
- Estimates compute on both sides via per-operator FLOP formulas.
- Ranks boundaries by: cut-bytes, a weighted score, or a simple latency model.
- Visualises Comm / compute / Pareto (Comm vs imbalance) / latency.
- Exports publication-ready plots as PDF/SVG (overview or single plots).
- NEW: exports the top-k split table directly as LaTeX (booktabs).

Requirements
------------
- Python 3.8+
- onnx, numpy
- matplotlib

Run
---
python analyse_and_split_gui.py
"""

from __future__ import annotations

import math
import json
import csv
import statistics
import os
import re
import threading
import queue
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

# Matplotlib backend must be selected BEFORE importing pyplot/backends
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import onnx
from onnx import shape_inference

from . import api as asc

__version__ = "0.10.39"


# ------------------------------- Tooltips -------------------------------

class ToolTip:
    """Simple hover tooltip for Tk/ttk widgets."""

    def __init__(self, widget: tk.Widget, text: str, *, wraplength: int = 360):
        self.widget = widget
        self.text = text
        self.wraplength = wraplength
        self._tip: Optional[tk.Toplevel] = None

        widget.bind("<Enter>", self._on_enter, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<ButtonPress>", self._on_leave, add=True)

    def _on_enter(self, _evt=None):
        if self._tip is not None or not self.text:
            return

        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8

        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            wraplength=self.wraplength,
        )
        label.pack(ipadx=6, ipady=3)
        self._tip = tw

    def _on_leave(self, _evt=None):
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


# ----------------------------- Helper logic -----------------------------

@dataclass
class Params:
    topk: int
    min_gap: int
    min_compute_pct: float

    batch_override: Optional[int]
    assume_bpe: Optional[int]

    exclude_trivial: bool
    only_single_tensor: bool
    strict_boundary: bool

    # Skip-/Block-aware candidate pruning
    prune_skip_block: bool
    skip_min_span: int
    skip_allow_last_n: int

    ranking: str  # cut | score | latency
    log_comm: bool
    w_comm: float
    w_imb: float
    w_tensors: float
    show_pareto_front: bool

    # Link / system model (plugin-like, see core LinkModelSpec)
    link_model: str
    bw_value: Optional[float]
    bw_unit: str
    gops_left: Optional[float]
    gops_right: Optional[float]
    overhead_ms: float

    link_energy_pj_per_byte: Optional[float]
    link_mtu_payload_bytes: Optional[int]
    link_per_packet_overhead_ms: Optional[float]
    link_per_packet_overhead_bytes: Optional[int]

    # Optional compute-energy model
    energy_pj_per_flop_left: Optional[float]
    energy_pj_per_flop_right: Optional[float]

    # Optional link constraints (per inference)
    link_max_latency_ms: Optional[float]
    link_max_energy_mJ: Optional[float]
    link_max_bytes: Optional[int]

    # Optional activation-memory constraints (peak during execution)
    max_peak_act_left: Optional[float]
    max_peak_act_left_unit: str
    max_peak_act_right: Optional[float]
    max_peak_act_right_unit: str

    show_top_tensors: int
def _safe_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(s: str) -> Optional[int]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _latex_escape(s: str) -> str:
    """Escape a string for use in LaTeX text contexts (caption etc.)."""
    # minimal escaping for typical filenames
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_")
    s = s.replace("%", "\\%")
    s = s.replace("&", "\\&")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("$", "\\$")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def _label_sanitize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "model"


def pareto_front(points: List[Tuple[float, float]]) -> List[int]:
    """Return indices of non-dominated points (minimisation in both dims)."""
    if not points:
        return []
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    front: List[int] = []
    best_y = float("inf")
    for i in order:
        _, y = points[i]
        if y < best_y:
            front.append(i)
            best_y = y
    return front


# ------------------------------ Main GUI --------------------------------


class SplitPointAnalyserGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(f"ONNX Split-Point Analyser v{__version__} (core v{getattr(asc, '__version__', '?')})")
        self.geometry("1250x860")

        self.model_path: Optional[str] = None
        self.analysis: Optional[Dict] = None
        self.current_picks: List[int] = []

        self._build_ui()

    # -------------------------- UI construction --------------------------

    def _build_ui(self):
        # --- Top bar: open model ---
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        self.btn_open = ttk.Button(top, text="Open Model…", command=self._on_open)
        self.btn_open.pack(side=tk.LEFT)

        self.lbl_model = ttk.Label(top, text="(no model loaded)")
        self.lbl_model.pack(side=tk.LEFT, padx=10)

        # --- Parameters ---
        self.params_frame = ttk.LabelFrame(self, text="Analysis Parameters")
        self.params_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        # General parameters row
        general = ttk.Frame(self.params_frame)
        general.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.params_frame.columnconfigure(0, weight=1)

        col = 0
        self.var_topk = tk.StringVar(value="10")
        ttk.Label(general, text="Top-k:").grid(row=0, column=col, sticky="w")
        col += 1
        self.sp_topk = ttk.Spinbox(general, from_=1, to=1000, textvariable=self.var_topk, width=6)
        self.sp_topk.grid(row=0, column=col, sticky="w", padx=(4, 14))
        col += 1

        self.var_min_gap = tk.StringVar(value="2")
        ttk.Label(general, text="Min gap:").grid(row=0, column=col, sticky="w")
        col += 1
        self.sp_min_gap = ttk.Spinbox(general, from_=0, to=1000, textvariable=self.var_min_gap, width=6)
        self.sp_min_gap.grid(row=0, column=col, sticky="w", padx=(4, 14))
        col += 1

        self.var_min_compute = tk.StringVar(value="1")
        ttk.Label(general, text="Min compute each side (%):").grid(row=0, column=col, sticky="w")
        col += 1
        self.ent_min_compute = ttk.Entry(general, textvariable=self.var_min_compute, width=6)
        self.ent_min_compute.grid(row=0, column=col, sticky="w", padx=(4, 14))
        col += 1

        self.var_batch = tk.StringVar(value="")
        ttk.Label(general, text="Batch override:").grid(row=0, column=col, sticky="w")
        col += 1
        self.ent_batch = ttk.Entry(general, textvariable=self.var_batch, width=8)
        self.ent_batch.grid(row=0, column=col, sticky="w", padx=(4, 14))
        col += 1

        self.var_bpe = tk.StringVar(value="")
        ttk.Label(general, text="Assume act bytes/elt:").grid(row=0, column=col, sticky="w")
        col += 1
        self.ent_bpe = ttk.Entry(general, textvariable=self.var_bpe, width=6)
        self.ent_bpe.grid(row=0, column=col, sticky="w", padx=(4, 14))
        col += 1

        # second row
        self.var_exclude_trivial = tk.BooleanVar(value=True)
        self.chk_exclude_trivial = ttk.Checkbutton(general, text="Exclude trivial ops", variable=self.var_exclude_trivial)
        self.chk_exclude_trivial.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.var_only_one = tk.BooleanVar(value=False)
        self.chk_only_one = ttk.Checkbutton(general, text="Only one crossing tensor", variable=self.var_only_one)
        self.chk_only_one.grid(row=1, column=2, columnspan=3, sticky="w", pady=(4, 0))

        # Strict boundary (feasible split): Part2 must not depend on original model inputs.
        self.var_strict_boundary = tk.BooleanVar(value=True)
        self.chk_strict = ttk.Checkbutton(general, text="Strict boundary", variable=self.var_strict_boundary)
        self.chk_strict.grid(row=1, column=5, columnspan=2, sticky="w", pady=(4, 0), padx=(0, 10))

        self.var_show_top_tensors = tk.StringVar(value="3")
        ttk.Label(general, text="Show top tensors per boundary:").grid(row=1, column=7, sticky="e", pady=(4, 0))
        self.sp_show_top_tensors = ttk.Spinbox(general, from_=0, to=50, textvariable=self.var_show_top_tensors, width=6)
        self.sp_show_top_tensors.grid(row=1, column=8, sticky="w", padx=(4, 14), pady=(4, 0))

        # third row: Skip-/Block-aware pruning
        self.var_prune_skip_block = tk.BooleanVar(value=True)
        self.chk_prune_skip_block = ttk.Checkbutton(general, text="Skip/Block pruning", variable=self.var_prune_skip_block)
        self.chk_prune_skip_block.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ToolTip(self.chk_prune_skip_block, "Heuristic pruning: avoid split candidates *inside* residual/skip blocks (e.g., long skip into Add/Concat).")

        ttk.Label(general, text="Min skip span (ops):").grid(row=2, column=2, sticky="w", pady=(4, 0))
        self.var_skip_min_span = tk.StringVar(value="8")
        self.ent_skip_min_span = ttk.Entry(general, textvariable=self.var_skip_min_span, width=6)
        self.ent_skip_min_span.grid(row=2, column=3, sticky="w", padx=(4, 14), pady=(4, 0))

        ttk.Label(general, text="Allow last N inside:").grid(row=2, column=4, sticky="w", pady=(4, 0))
        self.var_skip_allow_last_n = tk.StringVar(value="0")
        self.ent_skip_allow_last_n = ttk.Entry(general, textvariable=self.var_skip_allow_last_n, width=6)
        self.ent_skip_allow_last_n.grid(row=2, column=5, sticky="w", padx=(4, 14), pady=(4, 0))

        # Ranking frame
        self.rank_frame = ttk.LabelFrame(self.params_frame, text="Ranking")
        self.rank_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.rank_frame.columnconfigure(0, weight=1)

        r = ttk.Frame(self.rank_frame)
        r.pack(fill=tk.X, padx=8, pady=6)

        self.var_rank = tk.StringVar(value="score")
        ttk.Label(r, text="Ranking:").grid(row=0, column=0, sticky="w")
        self.cb_rank = ttk.Combobox(r, textvariable=self.var_rank, values=["cut", "score", "latency"], width=10, state="readonly")
        self.cb_rank.grid(row=0, column=1, sticky="w", padx=(4, 10))
        self.cb_rank.bind("<<ComboboxSelected>>", lambda _e: self._on_rank_changed(), add=True)

        self.var_log_comm = tk.BooleanVar(value=True)
        self.chk_log_comm = ttk.Checkbutton(r, text="log10(1+comm)", variable=self.var_log_comm)
        self.chk_log_comm.grid(row=0, column=2, sticky="w", padx=(0, 10))

        self.var_w_comm = tk.StringVar(value="1.0")
        self.var_w_imb = tk.StringVar(value="3.0")
        self.var_w_tensors = tk.StringVar(value="0.2")

        ttk.Label(r, text="w_comm").grid(row=0, column=3, sticky="e")
        self.ent_w_comm = ttk.Entry(r, textvariable=self.var_w_comm, width=6)
        self.ent_w_comm.grid(row=0, column=4, sticky="w", padx=(2, 6))

        ttk.Label(r, text="w_imb").grid(row=0, column=5, sticky="e")
        self.ent_w_imb = ttk.Entry(r, textvariable=self.var_w_imb, width=6)
        self.ent_w_imb.grid(row=0, column=6, sticky="w", padx=(2, 6))

        ttk.Label(r, text="w_tensors").grid(row=0, column=7, sticky="e")
        self.ent_w_tensors = ttk.Entry(r, textvariable=self.var_w_tensors, width=6)
        self.ent_w_tensors.grid(row=0, column=8, sticky="w", padx=(2, 10))

        self.var_show_pareto = tk.BooleanVar(value=True)
        self.chk_show_pareto = ttk.Checkbutton(r, text="Show Pareto front", variable=self.var_show_pareto)
        self.chk_show_pareto.grid(row=0, column=9, sticky="w")

        # Latency model (collapsible)
        # This block is optional and can take a lot of vertical space. We keep a compact
        # toggle row visible and show/hide the full settings panel.
        self.var_lat_expanded = tk.BooleanVar(value=False)

        self.lat_container = ttk.Frame(self.params_frame)
        self.lat_container.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.lat_container.columnconfigure(0, weight=1)

        lat_toggle = ttk.Frame(self.lat_container)
        lat_toggle.grid(row=0, column=0, sticky="ew")
        lat_toggle.columnconfigure(0, weight=1)

        self.btn_lat_toggle = ttk.Button(
            lat_toggle,
            text="▶ Latency model (optional)",
            command=self._toggle_latency_frame,
        )
        self.btn_lat_toggle.grid(row=0, column=0, sticky="w")

        self.lat_frame = ttk.LabelFrame(self.lat_container, text="Latency model")
        self.lat_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        l = ttk.Frame(self.lat_frame)
        l.pack(fill=tk.X, padx=8, pady=6)

        self.var_bw = tk.StringVar(value="")
        ttk.Label(l, text="Link bandwidth:").grid(row=0, column=0, sticky="w")
        self.ent_bw = ttk.Entry(l, textvariable=self.var_bw, width=10)
        self.ent_bw.grid(row=0, column=1, sticky="w", padx=(4, 6))

        self.var_bw_unit = tk.StringVar(value="MB/s")
        self.cb_bw_unit = ttk.Combobox(l, textvariable=self.var_bw_unit, values=sorted(asc.BANDWIDTH_MULT.keys()), width=8, state="readonly")
        self.cb_bw_unit.grid(row=0, column=2, sticky="w", padx=(0, 12))

        self.var_gops_l = tk.StringVar(value="")
        ttk.Label(l, text="GOPS left:").grid(row=0, column=3, sticky="w")
        self.ent_gops_l = ttk.Entry(l, textvariable=self.var_gops_l, width=10)
        self.ent_gops_l.grid(row=0, column=4, sticky="w", padx=(4, 12))

        self.var_gops_r = tk.StringVar(value="")
        ttk.Label(l, text="GOPS right:").grid(row=0, column=5, sticky="w")
        self.ent_gops_r = ttk.Entry(l, textvariable=self.var_gops_r, width=10)
        self.ent_gops_r.grid(row=0, column=6, sticky="w", padx=(4, 12))

        self.var_overhead = tk.StringVar(value="0")
        ttk.Label(l, text="Overhead (ms):").grid(row=0, column=7, sticky="w")
        self.ent_overhead = ttk.Entry(l, textvariable=self.var_overhead, width=8)
        self.ent_overhead.grid(row=0, column=8, sticky="w")

        # Row 1: link model + energy
        self.var_link_model = tk.StringVar(value="ideal")
        ttk.Label(l, text="Link model:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.cb_link_model = ttk.Combobox(l, textvariable=self.var_link_model, values=["ideal", "packetized"], width=10, state="readonly")
        self.cb_link_model.grid(row=1, column=1, sticky="w", padx=(4, 12), pady=(6, 0))
        ToolTip(self.cb_link_model, "ideal: overhead + bytes/bandwidth\npacketized: adds per-packet overhead (MTU, headers).")

        self.var_link_energy = tk.StringVar(value="")
        ttk.Label(l, text="E_link (pJ/B):").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.ent_link_energy = ttk.Entry(l, textvariable=self.var_link_energy, width=10)
        self.ent_link_energy.grid(row=1, column=3, sticky="w", padx=(4, 12), pady=(6, 0))

        self.var_link_mtu = tk.StringVar(value="")
        ttk.Label(l, text="MTU payload (B):").grid(row=1, column=4, sticky="w", pady=(6, 0))
        self.ent_link_mtu = ttk.Entry(l, textvariable=self.var_link_mtu, width=10)
        self.ent_link_mtu.grid(row=1, column=5, sticky="w", padx=(4, 12), pady=(6, 0))

        self.var_link_pkt_ovh_ms = tk.StringVar(value="")
        ttk.Label(l, text="pkt ovh (ms):").grid(row=1, column=6, sticky="w", pady=(6, 0))
        self.ent_link_pkt_ovh_ms = ttk.Entry(l, textvariable=self.var_link_pkt_ovh_ms, width=8)
        self.ent_link_pkt_ovh_ms.grid(row=1, column=7, sticky="w", padx=(4, 12), pady=(6, 0))

        self.var_link_pkt_ovh_bytes = tk.StringVar(value="")
        ttk.Label(l, text="pkt ovh (B):").grid(row=1, column=8, sticky="w", pady=(6, 0))
        self.ent_link_pkt_ovh_bytes = ttk.Entry(l, textvariable=self.var_link_pkt_ovh_bytes, width=8)
        self.ent_link_pkt_ovh_bytes.grid(row=1, column=9, sticky="w", padx=(4, 0), pady=(6, 0))

        # Row 2: link constraints + compute energy
        self.var_link_max_ms = tk.StringVar(value="")
        ttk.Label(l, text="Link max ms:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.ent_link_max_ms = ttk.Entry(l, textvariable=self.var_link_max_ms, width=10)
        self.ent_link_max_ms.grid(row=2, column=1, sticky="w", padx=(4, 12), pady=(6, 0))

        self.var_link_max_mJ = tk.StringVar(value="")
        ttk.Label(l, text="Link max mJ:").grid(row=2, column=2, sticky="w", pady=(6, 0))
        self.ent_link_max_mJ = ttk.Entry(l, textvariable=self.var_link_max_mJ, width=10)
        self.ent_link_max_mJ.grid(row=2, column=3, sticky="w", padx=(4, 12), pady=(6, 0))

        self.var_link_max_bytes = tk.StringVar(value="")
        ttk.Label(l, text="Link max bytes:").grid(row=2, column=4, sticky="w", pady=(6, 0))
        self.ent_link_max_bytes = ttk.Entry(l, textvariable=self.var_link_max_bytes, width=10)
        self.ent_link_max_bytes.grid(row=2, column=5, sticky="w", padx=(4, 12), pady=(6, 0))

        self.var_energy_left = tk.StringVar(value="")
        ttk.Label(l, text="E_left (pJ/F):").grid(row=2, column=6, sticky="w", pady=(6, 0))
        self.ent_energy_left = ttk.Entry(l, textvariable=self.var_energy_left, width=10)
        self.ent_energy_left.grid(row=2, column=7, sticky="w", padx=(4, 12), pady=(6, 0))

        self.var_energy_right = tk.StringVar(value="")
        ttk.Label(l, text="E_right (pJ/F):").grid(row=2, column=8, sticky="w", pady=(6, 0))
        self.ent_energy_right = ttk.Entry(l, textvariable=self.var_energy_right, width=10)
        self.ent_energy_right.grid(row=2, column=9, sticky="w", padx=(4, 0), pady=(6, 0))

        # Row 3: activation-memory constraints (peak)
        self.var_mem_left = tk.StringVar(value="")
        ttk.Label(l, text="Max act mem left:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.ent_mem_left = ttk.Entry(l, textvariable=self.var_mem_left, width=10)
        self.ent_mem_left.grid(row=3, column=1, sticky="w", padx=(4, 6), pady=(6, 0))

        self.var_mem_left_unit = tk.StringVar(value="MiB")
        self.cb_mem_left_unit = ttk.Combobox(l, textvariable=self.var_mem_left_unit, values=sorted(asc.UNIT_MULT.keys()), width=8, state="readonly")
        self.cb_mem_left_unit.grid(row=3, column=2, sticky="w", padx=(0, 12), pady=(6, 0))

        self.var_mem_right = tk.StringVar(value="")
        ttk.Label(l, text="Max act mem right:").grid(row=3, column=3, sticky="w", pady=(6, 0))
        self.ent_mem_right = ttk.Entry(l, textvariable=self.var_mem_right, width=10)
        self.ent_mem_right.grid(row=3, column=4, sticky="w", padx=(4, 6), pady=(6, 0))

        self.var_mem_right_unit = tk.StringVar(value="MiB")
        self.cb_mem_right_unit = ttk.Combobox(l, textvariable=self.var_mem_right_unit, values=sorted(asc.UNIT_MULT.keys()), width=8, state="readonly")
        self.cb_mem_right_unit.grid(row=3, column=5, sticky="w", padx=(0, 0), pady=(6, 0))

        ToolTip(self.ent_mem_left, "Optional: constrain peak activation memory of part1 (approx, from value spans).")
        ToolTip(self.ent_mem_right, "Optional: constrain peak activation memory of part2 (approx, from value spans).")

        # Start collapsed by default, unless latency ranking is selected.
        self._set_latency_expanded((self.var_rank.get() or "").strip().lower() == "latency")

        # Diagnostics frame
        self.diag_frame = ttk.LabelFrame(self.params_frame, text="Diagnostics")
        self.diag_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))

        d = ttk.Frame(self.diag_frame)
        d.pack(fill=tk.X, padx=8, pady=6)

        self.var_shape_coverage = tk.StringVar(value="(run analysis)")
        self.var_unknown_crossing = tk.StringVar(value="(run analysis)")
        self.var_diag_note = tk.StringVar(value="")

        ttk.Label(d, text="Shape coverage (known/produced):").grid(row=0, column=0, sticky="w")
        self.lbl_cov = ttk.Label(d, textvariable=self.var_shape_coverage)
        self.lbl_cov.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(d, text="Max unknown crossing tensors:").grid(row=0, column=2, sticky="w")
        self.lbl_unk = ttk.Label(d, textvariable=self.var_unknown_crossing)
        self.lbl_unk.grid(row=0, column=3, sticky="w", padx=(6, 18))

        self.lbl_note = ttk.Label(d, textvariable=self.var_diag_note, foreground="#b00020")
        self.lbl_note.grid(row=0, column=4, sticky="w")

        # Nordstern (relevance analysis for unknown tensor sizes)
        self.btn_nordstern = ttk.Button(d, text="Nordstern…", command=self._show_nordstern)
        self.btn_nordstern.grid(row=0, column=5, sticky="e", padx=(10, 0))
        ToolTip(self.btn_nordstern, "Show a relevance analysis for tensors with unknown activation sizes.\n"
                                  "This helps identify which missing shape information could impact split decisions most.")

        # ---------------- Actions (analyse + export/split) ----------------
        # Keep this compact (small Analyse button) and always visible.
        action_bar = ttk.Frame(self.params_frame)
        action_bar.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        action_bar.columnconfigure(0, weight=1)

        # Row 0: main actions
        main_actions = ttk.Frame(action_bar)
        main_actions.grid(row=0, column=0, sticky="w")

        self.btn_analyse = ttk.Button(main_actions, text="Analyse", command=self._on_analyse, width=12)
        self.btn_analyse.pack(side=tk.LEFT)

        self.btn_export_tex = ttk.Button(main_actions, text="Export TeX table…", command=self._export_tex_table)
        self.btn_export_tex.pack(side=tk.LEFT, padx=(10, 0))

        self.btn_split = ttk.Button(main_actions, text="Split selected…", command=self._split_selected_boundary)
        self.btn_split.pack(side=tk.LEFT, padx=(10, 0))

        self.btn_benchmark = ttk.Button(main_actions, text="Benchmark set…", command=self._generate_benchmark_set)
        self.btn_benchmark.pack(side=tk.LEFT, padx=(10, 0))
        ToolTip(
            self.btn_benchmark,
            "Export the current top-k split candidates into a benchmark suite folder.\n"
            "It will generate split sub-models + runner scripts for each split and a master script to run/collect them.",
        )

        # Row 1: split options (compact)
        split_opts = ttk.Frame(action_bar)
        split_opts.grid(row=1, column=0, sticky="w", pady=(4, 0))

        self.var_split_validate = tk.BooleanVar(value=False)
        self.chk_split_validate = ttk.Checkbutton(split_opts, text="Validate (ORT)", variable=self.var_split_validate)
        self.chk_split_validate.pack(side=tk.LEFT)

        ttk.Label(split_opts, text="eps").pack(side=tk.LEFT, padx=(8, 2))
        self.var_split_eps = tk.StringVar(value="1e-4")
        self.ent_split_eps = ttk.Entry(split_opts, textvariable=self.var_split_eps, width=8)
        self.ent_split_eps.pack(side=tk.LEFT)

        self.var_split_runner = tk.BooleanVar(value=True)
        self.chk_split_runner = ttk.Checkbutton(split_opts, text="Runner skeleton", variable=self.var_split_runner)
        self.chk_split_runner.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(split_opts, text="Runner target:").pack(side=tk.LEFT, padx=(10, 2))
        self.var_runner_target = tk.StringVar(value="auto")
        self.cb_runner_target = ttk.Combobox(split_opts, textvariable=self.var_runner_target, values=["auto", "cpu", "cuda", "tensorrt"], width=10, state="readonly")
        self.cb_runner_target.pack(side=tk.LEFT)
        ToolTip(self.cb_runner_target, "Default execution provider for the generated runner (override via --provider).")

        self.var_split_folder = tk.BooleanVar(value=True)
        self.chk_split_folder = ttk.Checkbutton(split_opts, text="Export split as folder", variable=self.var_split_folder)
        self.chk_split_folder.pack(side=tk.LEFT, padx=(10, 0))

        # Split-context diagrams (paper/debug).
        self.var_split_ctx_full = tk.BooleanVar(value=True)
        self.chk_split_ctx_full = ttk.Checkbutton(split_opts, text="Context (full)", variable=self.var_split_ctx_full)
        self.chk_split_ctx_full.pack(side=tk.LEFT, padx=(10, 0))
        ToolTip(self.chk_split_ctx_full, "Export a detailed GraphViz context diagram around the selected split boundary.")

        self.var_split_ctx_cutflow = tk.BooleanVar(value=True)
        self.chk_split_ctx_cutflow = ttk.Checkbutton(split_opts, text="Context (cut-flow)", variable=self.var_split_ctx_cutflow)
        self.chk_split_ctx_cutflow.pack(side=tk.LEFT, padx=(10, 0))
        ToolTip(self.chk_split_ctx_cutflow, "Export a compact diagram showing only the causal cut-flow (better suited for papers).")

        # Context hops: how much surrounding graph to include in the context diagrams.
        ttk.Label(split_opts, text="hops:").pack(side=tk.LEFT, padx=(6, 2))
        self.var_split_ctx_hops = tk.StringVar(value="2")
        self.cb_split_ctx_hops = ttk.Combobox(
            split_opts,
            textvariable=self.var_split_ctx_hops,
            values=["0", "1", "2", "3"],
            width=3,
            state="readonly",
        )
        self.cb_split_ctx_hops.pack(side=tk.LEFT)
        ToolTip(
            self.cb_split_ctx_hops,
            "Context expansion depth for exported GraphViz split-context diagrams.\n"
            "0 = only immediate producers/consumers of the cut tensors.\n"
            "Higher values include more surrounding context (may get large).",
        )


        # Disable actions until an analysis was run + a boundary row is selected
        self.btn_export_tex.state(["disabled"])
        self.btn_split.state(["disabled"])
        self.btn_benchmark.state(["disabled"])

        # --- Results split: table + plots ---
        mid = ttk.PanedWindow(self, orient=tk.VERTICAL)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # ------------------------------ Table ------------------------------
        table_frame = ttk.LabelFrame(mid, text="Suggested Boundaries")
        mid.add(table_frame, weight=1)

        # Use grid so the bottom of the frame is never "eaten" by the Treeview.
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        cols = [
            "rank",
            "boundary",
            "left_op",
            "right_op",
            "cut_mb",
            "num_tensors",
            "gflops_left",
            "gflops_right",
            "peak_left_mib",
            "peak_right_mib",
            "peak_max_mib",
        ]

        table_inner = ttk.Frame(table_frame)
        table_inner.grid(row=0, column=0, sticky="nsew")
        table_inner.columnconfigure(0, weight=1)
        table_inner.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_inner, columns=cols, show="headings")
        self.tree.heading("rank", text="#")
        self.tree.heading("boundary", text="Boundary")
        self.tree.heading("left_op", text="Left op")
        self.tree.heading("right_op", text="Right op")
        self.tree.heading("cut_mb", text="Cut (MB)")
        self.tree.heading("num_tensors", text="#Tensors")
        self.tree.heading("gflops_left", text="Compute Left (GFLOPs)")
        self.tree.heading("gflops_right", text="Compute Right (GFLOPs)")
        self.tree.heading("peak_left_mib", text="Peak L (MiB)")
        self.tree.heading("peak_right_mib", text="Peak R (MiB)")
        self.tree.heading("peak_max_mib", text="Peak max (MiB)")

        self.tree.column("rank", width=40, anchor=tk.E)
        self.tree.column("boundary", width=80, anchor=tk.E)
        self.tree.column("left_op", width=150)
        self.tree.column("right_op", width=150)
        self.tree.column("cut_mb", width=90, anchor=tk.E)
        self.tree.column("num_tensors", width=80, anchor=tk.E)
        self.tree.column("gflops_left", width=135, anchor=tk.E)
        self.tree.column("gflops_right", width=135, anchor=tk.E)
        self.tree.column("peak_left_mib", width=110, anchor=tk.E)
        self.tree.column("peak_right_mib", width=110, anchor=tk.E)
        self.tree.column("peak_max_mib", width=110, anchor=tk.E)

        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(table_inner, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns")

        self.tree.tag_configure("pick", background="#eef6ff")

        # Enable split only when a boundary row (not a child tensor row) is selected
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._update_action_buttons(), add=True)

        # ------------------------------ Plots ------------------------------
        plot_frame = ttk.LabelFrame(mid, text="Plots")
        mid.add(plot_frame, weight=2)

        # Use grid so canvas does not hide the toolbar/export controls.
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.rowconfigure(1, weight=0)
        plot_frame.rowconfigure(2, weight=0)

        self.fig = Figure(figsize=(10, 6), constrained_layout=True)
        self.ax_comm = self.fig.add_subplot(2, 2, 1)
        self.ax_comp = self.fig.add_subplot(2, 2, 2)
        self.ax_pareto = self.fig.add_subplot(2, 2, 3)
        self.ax_lat = self.fig.add_subplot(2, 2, 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Matplotlib navigation toolbar (zoom/pan/save/etc.)
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 0))
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        try:
            # Some Matplotlib versions auto-pack; calling pack() is harmless and makes it robust.
            self.toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        except Exception:
            pass

        # Export buttons near plots
        export_bar = ttk.Frame(plot_frame)
        export_bar.grid(row=2, column=0, sticky="ew", padx=6, pady=(2, 6))

        self.btn_export_svg = ttk.Button(export_bar, text="Export SVG (overview)", command=lambda: self._export_overview("svg"))
        self.btn_export_pdf = ttk.Button(export_bar, text="Export PDF (overview)", command=lambda: self._export_overview("pdf"))
        self.btn_export_svg_s = ttk.Button(export_bar, text="Export SVGs (single)", command=lambda: self._export_single("svg"))
        self.btn_export_pdf_s = ttk.Button(export_bar, text="Export PDFs (single)", command=lambda: self._export_single("pdf"))

        self.btn_export_svg.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_export_pdf.pack(side=tk.LEFT, padx=(0, 18))
        self.btn_export_svg_s.pack(side=tk.LEFT, padx=(0, 6))
        self.btn_export_pdf_s.pack(side=tk.LEFT)


        # Tooltips
        self._add_tooltips()

    def _add_tooltips(self):
        ToolTip(self.btn_open, "Open an ONNX model (*.onnx).")

        ToolTip(self.sp_topk, "Number of split points to return (after filtering + min-gap suppression).")
        ToolTip(self.sp_min_gap, "Minimum boundary-index distance between returned split points.")
        ToolTip(self.ent_min_compute, "Require at least this percent of total compute on BOTH sides.\nAvoids degenerate 0/100% splits.")
        ToolTip(self.ent_batch, "Override batch dimension for shape/FLOP estimation (useful for symbolic batch).")
        ToolTip(self.ent_bpe, "Override bytes-per-element for activations (e.g., 1 INT8, 2 FP16, 4 FP32).\nIf empty, ONNX element type is used.")

        ToolTip(self.chk_exclude_trivial, "Remove boundaries adjacent to trivial ops (reshape/transpose/etc.).")
        ToolTip(self.chk_only_one, "Keep only boundaries with exactly one crossing activation tensor.")
        ToolTip(self.chk_strict, "Only keep boundaries where Part2 has no dependencies on original model inputs (besides weights). This guarantees a self-contained split.")
        ToolTip(self.sp_show_top_tensors, "Show the k largest crossing tensors per suggested boundary as child rows (0 disables).")

        ToolTip(
            self.ent_skip_min_span,
            "Skip-/block-aware pruning: minimum skip span in ops.\n"
            "If a tensor is forwarded over >= N ops into an Add/Concat-like merge, this is treated as a skip/residual block.\n"
            "Larger values are less aggressive (fewer candidates pruned).",
        )
        ToolTip(
            self.ent_skip_allow_last_n,
            "Skip-/block-aware pruning: allow the last N ops inside a detected skip/residual block.\n"
            "Use this to still permit splits close to the merge op.",
        )

        ToolTip(self.cb_rank, "Ranking mode:\n- cut: minimise communication bytes\n- score: weighted trade-off (comm + imbalance + tensor penalty)\n- latency: minimise predicted latency (requires bandwidth + GOPS L/R)")
        ToolTip(self.chk_log_comm, "Use log10(1+comm) inside the score to reduce domination by very large activations.")
        ToolTip(self.ent_w_comm, "Weight for communication term in the score.")
        ToolTip(self.ent_w_imb, "Weight for compute imbalance term in the score.")
        ToolTip(self.ent_w_tensors, "Weight for crossing-tensor penalty in the score.")
        ToolTip(self.chk_show_pareto, "Overlay the Pareto front (comm vs imbalance) in the Pareto plot.")

        ToolTip(
            self.btn_lat_toggle,
            "Show/hide optional latency/link settings.\n"
            "This is only needed for 'latency' ranking or the latency plot.\n"
            "Tip: the panel auto-expands when you switch Ranking to 'latency'.",
        )

        ToolTip(self.ent_bw, "Link bandwidth for latency model.")
        ToolTip(self.cb_bw_unit, "Bandwidth units (bytes/s and bits/s variants).")
        ToolTip(self.ent_gops_l, "Compute throughput of the left device in GOPS (10^9 ops/s).")
        ToolTip(self.ent_gops_r, "Compute throughput of the right device in GOPS (10^9 ops/s).")
        ToolTip(self.ent_overhead, "Constant overhead added to latency model (ms).")

        ToolTip(self.ent_link_energy, "Optional: link energy per transferred byte in pJ/B (used for energy constraints + export).")
        ToolTip(self.ent_link_mtu, "Packetized link model: MTU payload bytes per packet (data payload, not including headers).")
        ToolTip(self.ent_link_pkt_ovh_ms, "Packetized link model: additional per-packet latency overhead in ms (e.g., scheduling/airtime gaps).")
        ToolTip(self.ent_link_pkt_ovh_bytes, "Packetized link model: per-packet header overhead in bytes (protocol headers, framing, etc.).")

        ToolTip(self.ent_link_max_ms, "Constraint: maximum allowed link latency (ms) per inference. Candidates exceeding are filtered.")
        ToolTip(self.ent_link_max_mJ, "Constraint: maximum allowed link energy (mJ) per inference. Requires E_link (pJ/B).")
        ToolTip(self.ent_link_max_bytes, "Constraint: maximum allowed transferred bytes per inference.")

        ToolTip(self.ent_energy_left, "Optional: compute energy per flop for left device (pJ/F). Used for energy export/constraints.")
        ToolTip(self.ent_energy_right, "Optional: compute energy per flop for right device (pJ/F). Used for energy export/constraints.")

        ToolTip(self.cb_mem_left_unit, "Units for 'Max act mem left' constraint.")
        ToolTip(self.cb_mem_right_unit, "Units for 'Max act mem right' constraint.")

        ToolTip(self.lbl_cov, "Share of produced activation tensors whose size could be inferred.\nIf low, Comm(b) can be underestimated.")
        ToolTip(self.lbl_unk, "Max number of crossing tensors with unknown size on any boundary.\nIf >0, Comm(b) is a lower bound on those boundaries.")

        ToolTip(self.btn_analyse, "Run the analysis with the current settings.")

        ToolTip(self.btn_export_tex, "Export the current top-k table as LaTeX (booktabs).\nUseful for directly pasting into the paper.")
        ToolTip(
            self.btn_split,
            "Split the loaded ONNX model at the selected boundary and export two sub-models (part1/part2).\n"
            "The boundary tensors are chosen automatically as all activations crossing the boundary.",
        )
        ToolTip(
            self.chk_split_validate,
            "Validate the split using onnxruntime on CPU with random inputs.\n"
            "Checks that full(x) \u2248 part2(part1(x)). Requires: pip install onnxruntime",
        )
        ToolTip(
            self.ent_split_eps,
            "Validation tolerance eps for max absolute output difference.\n"
            "Leave empty to just report the diff without PASS/FAIL.",
        )
        ToolTip(
            self.chk_split_runner,
            "Generate a small onnxruntime runner script next to the exported models.\n"
            "Useful as a starting point for benchmarking/integration.",
        )
        ToolTip(
            self.chk_split_folder,
            "Export split into a dedicated folder (recommended).\n"
            "This keeps models, runner, plots, and metadata together.",
        )

        ToolTip(self.btn_export_svg, "Export the 2x2 plot overview as a single SVG.")
        ToolTip(self.btn_export_pdf, "Export the 2x2 plot overview as a single PDF.")
        ToolTip(self.btn_export_svg_s, "Export each plot as its own SVG file.")
        ToolTip(self.btn_export_pdf_s, "Export each plot as its own PDF file.")

    # ------------------------- Latency panel helpers ------------------------

    def _on_rank_changed(self):
        """Auto-expand latency settings when the user selects latency ranking."""
        if (self.var_rank.get() or "").strip().lower() == "latency":
            self._set_latency_expanded(True)

    def _toggle_latency_frame(self):
        self._set_latency_expanded(not bool(self.var_lat_expanded.get()))

    def _set_latency_expanded(self, expanded: bool):
        expanded = bool(expanded)
        self.var_lat_expanded.set(expanded)

        if expanded:
            # Restore the full panel.
            try:
                self.lat_frame.grid()
            except Exception:
                self.lat_frame.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            self.btn_lat_toggle.configure(text="▼ Latency model (optional)")
        else:
            # Hide the full panel and keep only the toggle row.
            self.lat_frame.grid_remove()
            self.btn_lat_toggle.configure(text="▶ Latency model (optional)")

    # ----------------------------- Event handlers -----------------------------

    def _on_open(self):
        path = filedialog.askopenfilename(
            title="Open ONNX model",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
        if not path:
            return
        self.model_path = path
        self.lbl_model.configure(text=os.path.basename(path))
        self._clear_results()

    def _on_analyse(self):
        if not self.model_path:
            messagebox.showwarning("No model", "Please open an ONNX model first.")
            return

        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("Invalid parameters", str(e))
            return

        try:
            self.analysis = self._analyse_model(self.model_path, params)
            self.current_picks = self._select_picks(self.analysis, params)
            # Nordstern: relevance analysis for unknown activation sizes
            self._compute_nordstern(self.analysis, self.current_picks, params)
            self._update_diagnostics(self.analysis)
            self._update_table(self.analysis, self.current_picks, params)
            self._update_plots(self.analysis, self.current_picks, params)
            self._update_action_buttons()
        except Exception as e:
            messagebox.showerror("Analysis failed", f"{type(e).__name__}: {e}")
            raise

    # ----------------------------- Parameters -----------------------------

    def _read_params(self) -> Params:
        topk = _safe_int(self.var_topk.get())
        if topk is None or topk <= 0:
            raise ValueError("Top-k must be a positive integer.")

        min_gap = _safe_int(self.var_min_gap.get())
        if min_gap is None or min_gap < 0:
            raise ValueError("Min gap must be an integer ≥ 0.")

        min_comp = _safe_float(self.var_min_compute.get())
        if min_comp is None:
            min_comp = 0.0
        if min_comp < 0:
            raise ValueError("Min compute each side (%) must be ≥ 0.")

        batch = _safe_int(self.var_batch.get())
        bpe = _safe_int(self.var_bpe.get())

        exclude_trivial = bool(self.var_exclude_trivial.get())
        only_one = bool(self.var_only_one.get())
        strict_boundary = bool(self.var_strict_boundary.get())

        prune_skip_block = bool(self.var_prune_skip_block.get())
        skip_min_span = _safe_int(self.var_skip_min_span.get())
        if skip_min_span is None or skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = _safe_int(self.var_skip_allow_last_n.get())
        if skip_allow_last_n is None:
            skip_allow_last_n = 0
        if skip_allow_last_n < 0:
            raise ValueError("Allow last N inside must be an integer ≥ 0.")

        ranking = (self.var_rank.get() or "cut").strip().lower()
        if ranking not in {"cut", "score", "latency"}:
            raise ValueError("Ranking must be one of: cut, score, latency")

        log_comm = bool(self.var_log_comm.get())

        w_comm = _safe_float(self.var_w_comm.get())
        w_imb = _safe_float(self.var_w_imb.get())
        w_tensors = _safe_float(self.var_w_tensors.get())
        if w_comm is None or w_imb is None or w_tensors is None:
            raise ValueError("Weights w_comm, w_imb, w_tensors must be numeric.")

        show_pareto_front = bool(self.var_show_pareto.get())

        bw_value = _safe_float(self.var_bw.get())
        bw_unit = (self.var_bw_unit.get() or "MB/s").strip()
        if bw_unit not in asc.BANDWIDTH_MULT:
            raise ValueError("Unknown bandwidth unit.")

        gops_left = _safe_float(self.var_gops_l.get())
        gops_right = _safe_float(self.var_gops_r.get())

        overhead_ms = _safe_float(self.var_overhead.get())
        if overhead_ms is None:
            overhead_ms = 0.0

        link_model = (self.var_link_model.get() or "ideal").strip().lower()
        if link_model not in {"ideal", "packetized"}:
            raise ValueError("Link model must be one of: ideal, packetized")

        link_energy_pj_per_byte = _safe_float(self.var_link_energy.get())
        link_mtu_payload_bytes = _safe_int(self.var_link_mtu.get())
        link_per_packet_overhead_ms = _safe_float(self.var_link_pkt_ovh_ms.get())
        link_per_packet_overhead_bytes = _safe_int(self.var_link_pkt_ovh_bytes.get())

        link_max_latency_ms = _safe_float(self.var_link_max_ms.get())
        link_max_energy_mJ = _safe_float(self.var_link_max_mJ.get())
        link_max_bytes = _safe_int(self.var_link_max_bytes.get())

        energy_pj_per_flop_left = _safe_float(self.var_energy_left.get())
        energy_pj_per_flop_right = _safe_float(self.var_energy_right.get())


        # Activation-memory constraints (optional, peak during execution; approx from value spans)
        max_peak_act_left = _safe_float(self.var_mem_left.get())
        max_peak_act_left_unit = (self.var_mem_left_unit.get() or "MiB").strip()
        if max_peak_act_left_unit not in asc.UNIT_MULT:
            raise ValueError("Unknown memory unit for left peak activation memory.")

        max_peak_act_right = _safe_float(self.var_mem_right.get())
        max_peak_act_right_unit = (self.var_mem_right_unit.get() or "MiB").strip()
        if max_peak_act_right_unit not in asc.UNIT_MULT:
            raise ValueError("Unknown memory unit for right peak activation memory.")

        show_top_tensors = _safe_int(self.var_show_top_tensors.get())
        if show_top_tensors is None or show_top_tensors < 0:
            raise ValueError("Show top tensors must be an integer ≥ 0.")

        return Params(
            topk=int(topk),
            min_gap=int(min_gap),
            min_compute_pct=float(min_comp),
            batch_override=batch,
            assume_bpe=bpe,
            exclude_trivial=exclude_trivial,
            only_single_tensor=only_one,
            strict_boundary=strict_boundary,
            prune_skip_block=bool(prune_skip_block),
            skip_min_span=int(skip_min_span),
            skip_allow_last_n=int(skip_allow_last_n),
            ranking=ranking,
            log_comm=log_comm,
            w_comm=float(w_comm),
            w_imb=float(w_imb),
            w_tensors=float(w_tensors),
            show_pareto_front=show_pareto_front,
            link_model=str(link_model),
            bw_value=bw_value,
            bw_unit=bw_unit,
            gops_left=gops_left,
            gops_right=gops_right,
            overhead_ms=float(overhead_ms),
            link_energy_pj_per_byte=link_energy_pj_per_byte,
            link_mtu_payload_bytes=link_mtu_payload_bytes,
            link_per_packet_overhead_ms=link_per_packet_overhead_ms,
            link_per_packet_overhead_bytes=link_per_packet_overhead_bytes,
            energy_pj_per_flop_left=energy_pj_per_flop_left,
            energy_pj_per_flop_right=energy_pj_per_flop_right,
            link_max_latency_ms=link_max_latency_ms,
            link_max_energy_mJ=link_max_energy_mJ,
            link_max_bytes=link_max_bytes,
            max_peak_act_left=max_peak_act_left,
            max_peak_act_left_unit=str(max_peak_act_left_unit),
            max_peak_act_right=max_peak_act_right,
            max_peak_act_right_unit=str(max_peak_act_right_unit),
            show_top_tensors=int(show_top_tensors),
        )

    def _build_system_spec(self, p: Params) -> asc.SystemSpec:
        """Construct a SystemSpec from GUI parameters.

        Separation rationale:
        - Workload metrics (Comm/ FLOPs) come from the ONNX graph analysis.
        - System metrics (compute GOPS, link model, constraints, energy) are user-specified.
        """

        def _mem_to_bytes(val: Optional[float], unit: str) -> Optional[int]:
            if val is None:
                return None
            try:
                mul = float(asc.UNIT_MULT.get(str(unit), 0.0))
            except Exception:
                mul = 0.0
            if mul <= 0:
                return None
            try:
                return int(max(0.0, float(val)) * mul)
            except Exception:
                return None

        mem_constraints = asc.MemoryConstraints(
            max_peak_act_left_bytes=_mem_to_bytes(getattr(p, 'max_peak_act_left', None), str(getattr(p, 'max_peak_act_left_unit', 'MiB'))),
            max_peak_act_right_bytes=_mem_to_bytes(getattr(p, 'max_peak_act_right', None), str(getattr(p, 'max_peak_act_right_unit', 'MiB'))),
        )

        link_constraints = asc.LinkConstraints(
            max_latency_ms=p.link_max_latency_ms,
            max_energy_mJ=p.link_max_energy_mJ,
            max_bytes=p.link_max_bytes,
        )

        link = asc.LinkModelSpec(
            type=str(p.link_model or 'ideal'),
            name='gui',
            bandwidth_value=p.bw_value,
            bandwidth_unit=str(p.bw_unit or 'MB/s'),
            overhead_ms=float(p.overhead_ms or 0.0),
            energy_pj_per_byte=p.link_energy_pj_per_byte,
            mtu_payload_bytes=p.link_mtu_payload_bytes,
            per_packet_overhead_ms=float(p.link_per_packet_overhead_ms or 0.0),
            per_packet_overhead_bytes=int(p.link_per_packet_overhead_bytes or 0),
            constraints=link_constraints,
        )

        sys = asc.SystemSpec(
            left=asc.ComputeSpec(gops=p.gops_left, energy_pj_per_flop=p.energy_pj_per_flop_left),
            right=asc.ComputeSpec(gops=p.gops_right, energy_pj_per_flop=p.energy_pj_per_flop_right),
            link=link,
            memory=mem_constraints,
            overhead_ms=0.0,
        )
        return sys

    # ----------------------------- Core analysis -----------------------------

    def _analyse_model(self, model_path: str, p: Params) -> Dict:
        model = onnx.load(model_path)
        model = shape_inference.infer_shapes(model)

        vimap = asc.value_info_map(model)
        asc.backfill_quant_shapes(model, vimap, batch=p.batch_override)

        nodes, producer_of, consumers_of = asc.build_producers_consumers(model)
        order = asc.topo_sort(nodes, producer_of)

        value_bytes = asc.compute_tensor_bytes_per_value(vimap, p.batch_override, p.assume_bpe)
        costs_bytes, val_span = asc.boundary_costs(order, producer_of, consumers_of, value_bytes)

        # Peak activation memory per boundary (approx)
        # Derived from value spans (producer -> last consumer) via Comm(b) live-set bytes.
        peak_l, peak_r, peak_max = asc.peak_activation_memory_per_boundary(costs_bytes)


        # Crossing tensor counts: known sizes vs all (unknown sizes enabled via value_bytes_all)
        counts_known = asc.boundary_tensor_counts(order, producer_of, consumers_of, value_bytes)
        value_bytes_all = {k: 1 for k in producer_of.keys()}
        counts_all = asc.boundary_tensor_counts(order, producer_of, consumers_of, value_bytes_all)
        unknown_counts = [max(0, int(a) - int(k)) for a, k in zip(counts_all, counts_known)]

        # Spans for ALL crossing values (including unknown sizes). Useful for Nordstern.
        _, val_span_all = asc.boundary_costs(order, producer_of, consumers_of, value_bytes_all)

        # Coverage of produced tensors that have an inferred size
        produced = list(producer_of.keys())
        known_produced = sum(1 for v in produced if value_bytes.get(v, 0) > 0)
        coverage = (float(known_produced) / float(len(produced))) if produced else 1.0

        # FLOPs per node
        node_flops_list = asc.per_node_flops(
            model,
            vimap,
            p.batch_override,
            bn_cost_per_elt=4,
            act_cost_per_elt=1,
            resize_cost_per_elt=1,
        )
        flops_by_node = {idx: fl for (idx, _, __, fl) in node_flops_list}
        total_flops = float(sum(flops_by_node.values()))
        flops_left_prefix = asc.compute_boundary_flops_prefix(order, flops_by_node)

        # Imbalance per boundary
        imb = []
        for b in range(len(costs_bytes)):
            fl_l = float(flops_left_prefix[b])
            fl_r = float(total_flops - flops_left_prefix[b])
            imb.append(abs(fl_l - fl_r) / total_flops if total_flops > 0 else 0.0)

        # Strict boundary feasibility (analysis-time)
        # ------------------------------------------
        # If enabled, we compute *exactly* whether a boundary is strict-feasible according to
        # the same rule used by split_model_on_cut_tensors(strict_boundary=True):
        #   Part2 may only require the cut tensors as external inputs (initializers are embedded).
        #
        # Rationale: A cheap topological heuristic can be wrong in graphs where some auxiliary
        # branches (e.g. Shape/Resize helpers) depend on original inputs and are scheduled early in
        # the topo order, but still end up inside Part2. Therefore we directly compute the external
        # inputs of the Part2 backward slice.
        strict_ok = [True] * (len(costs_bytes))
        strict_extras = {}
        if p.strict_boundary:
            for b in range(len(costs_bytes)):
                cut_tensors = asc.cut_tensors_for_boundary(order, nodes, b)
                extras = asc.strict_boundary_extras(model, cut_tensors)
                ok = (len(extras) == 0)
                strict_ok[b] = ok
                if not ok:
                    strict_extras[int(b)] = list(extras)

        return {
            "model": model,
            "nodes": nodes,
            "producer_of": producer_of,
            "consumers_of": consumers_of,
            "order": order,
            "vimap": vimap,
            "value_bytes": value_bytes,
            "costs_bytes": costs_bytes,
            "peak_act_mem_left_bytes": peak_l,
            "peak_act_mem_right_bytes": peak_r,
            "peak_act_mem_max_bytes": peak_max,
            "val_span": val_span,
            "val_span_all": val_span_all,
            "crossing_counts_known": counts_known,
            "crossing_counts_all": counts_all,
            "unknown_crossing_counts": unknown_counts,
            "shape_coverage": coverage,
            "known_produced": known_produced,
            "total_produced": len(produced),
            "max_unknown_crossing": max(unknown_counts) if unknown_counts else 0,
            "node_flops_list": node_flops_list,
            "flops_by_node": flops_by_node,
            "total_flops": total_flops,
            "flops_left_prefix": flops_left_prefix,
            "imbalance": imb,
            "strict_ok": strict_ok,
            "strict_last_ext_input_consumer_pos": None,
        }

    # ----------------------------- Candidate selection -----------------------------

    def _select_picks(self, a: Dict, p: Params) -> List[int]:
        nodes = a["nodes"]
        order = a["order"]
        costs = a["costs_bytes"]
        counts_all = a["crossing_counts_all"]
        flops_left_prefix = a["flops_left_prefix"]
        total_flops = float(a["total_flops"])
        peak_left_b = a.get("peak_act_mem_left_bytes") or []
        peak_right_b = a.get("peak_act_mem_right_bytes") or []
        peak_max_b = a.get("peak_act_mem_max_bytes") or []

        M = len(costs)
        candidates = list(range(M))

        # --- Candidate filters ---
        if p.exclude_trivial:
            TRIVIAL = {
                "Relu",
                "Reshape",
                "BatchNormalization",
                "Transpose",
                "Squeeze",
                "Unsqueeze",
                "Flatten",
                "Identity",
            }

            def _drop(b: int) -> bool:
                return nodes[order[b]].op_type in TRIVIAL or nodes[order[b + 1]].op_type in TRIVIAL

            candidates = [b for b in candidates if not _drop(b)]

        if p.only_single_tensor:
            candidates = [b for b in candidates if int(counts_all[b]) == 1]

        if p.strict_boundary:
            strict_ok = a.get("strict_ok")
            if strict_ok:
                candidates = [b for b in candidates if bool(strict_ok[b])]

        if p.min_compute_pct > 0 and total_flops > 0:
            thr = (p.min_compute_pct / 100.0) * total_flops
            candidates = [
                b
                for b in candidates
                if float(flops_left_prefix[b]) >= thr and float(total_flops - flops_left_prefix[b]) >= thr
            ]

        # --- Skip-/Block-aware candidate pruning ---
        if p.prune_skip_block:
            try:
                blocks = asc.detect_skip_blocks(
                    order,
                    nodes,
                    a.get("producer_of") or {},
                    min_skip_len=int(p.skip_min_span),
                )
                kept, forbid = asc.prune_candidates_skip_block(
                    candidates,
                    blocks,
                    allow_last_n_inside=int(p.skip_allow_last_n),
                )
                a["skip_blocks"] = blocks
                a["skip_block_forbidden"] = forbid
                a["candidate_prune_skip_block"] = {
                    "enabled": True,
                    "before": int(len(candidates)),
                    "after": int(len(kept)),
                    "forbidden": int(len(forbid)),
                }
                candidates = kept
            except Exception as e:
                a["candidate_prune_skip_block_error"] = str(e)
        else:
            a["skip_blocks"] = []
            a["skip_block_forbidden"] = []
            a["candidate_prune_skip_block"] = {"enabled": False, "before": int(len(candidates)), "after": int(len(candidates))}

        # --- System model (link plugin + constraints) ---
        sys = self._build_system_spec(p)
        try:
            a["system_spec"] = asdict(sys)
        except Exception:
            a["system_spec"] = None

        # Precompute predicted metrics for export/plotting (None if parameters are missing)
        pred_lat_total: List[Optional[float]] = [None] * M
        pred_lat_link: List[Optional[float]] = [None] * M
        pred_energy_total: List[Optional[float]] = [None] * M
        pred_energy_link: List[Optional[float]] = [None] * M

        for b in range(M):
            m = sys.estimate_boundary(
                comm_bytes=float(costs[b]),
                flops_left=float(flops_left_prefix[b]),
                flops_total=float(total_flops),
            )
            pred_lat_total[b] = m.get("latency_total_ms")
            pred_lat_link[b] = m.get("latency_link_ms")
            pred_energy_total[b] = m.get("energy_total_mJ")
            pred_energy_link[b] = m.get("energy_link_mJ")

        a["pred_latency_total_ms"] = pred_lat_total
        a["pred_latency_link_ms"] = pred_lat_link
        a["pred_energy_total_mJ"] = pred_energy_total
        a["pred_energy_link_mJ"] = pred_energy_link

        # Enforce link constraints (if configured)
        before_lc = int(len(candidates))
        candidates = [b for b in candidates if sys.link.is_feasible(float(costs[b]))]
        a["candidate_prune_link_constraints"] = {
            "before": before_lc,
            "after": int(len(candidates)),
            "pruned": int(before_lc - len(candidates)),
        }

        # Enforce activation-memory constraints (if configured)
        peak_left = a.get("peak_act_mem_left_bytes") or []
        peak_right = a.get("peak_act_mem_right_bytes") or []
        before_mem = int(len(candidates))
        if peak_left and peak_right:
            candidates = [
                b for b in candidates
                if sys.is_memory_feasible(
                    peak_left_bytes=float(peak_left[b]) if b < len(peak_left) else float('inf'),
                    peak_right_bytes=float(peak_right[b]) if b < len(peak_right) else float('inf'),
                )
            ]
        a["candidate_prune_memory_constraints"] = {
            "before": before_mem,
            "after": int(len(candidates)),
            "pruned": int(before_mem - len(candidates)),
            "max_peak_act_left_bytes": getattr(getattr(sys, 'memory', None), 'max_peak_act_left_bytes', None),
            "max_peak_act_right_bytes": getattr(getattr(sys, 'memory', None), 'max_peak_act_right_bytes', None),
        }

        # --- Ranking ---
        scores = None
        latency_ms = None

        if p.ranking in {"score", "latency"}:
            scores = asc.compute_scores_for_candidates(
                candidates,
                costs,
                counts_all,
                flops_left_prefix,
                total_flops,
                w_comm=p.w_comm,
                w_imb=p.w_imb,
                w_tensors=p.w_tensors,
                linear_comm=not p.log_comm,
            )

        if p.ranking == "cut":
            candidates.sort(key=lambda b: float(costs[b]))

        elif p.ranking == "score":
            candidates.sort(key=lambda b: float(scores.get(b, 0.0) if scores is not None else 0.0))

        elif p.ranking == "latency":
            # Prefer the plugin-based system model; fall back gracefully if incomplete.
            latency_ms = {b: pred_lat_total[b] for b in candidates if pred_lat_total[b] is not None}
            if latency_ms:
                candidates.sort(key=lambda b: float(pred_lat_total[b] if pred_lat_total[b] is not None else float("inf")))
            else:
                candidates.sort(key=lambda b: float(scores.get(b, 0.0) if scores is not None else costs[b]))

        # Non-maximum suppression by boundary index
        picks: List[int] = []
        for b in candidates:
            if all(abs(b - s) > p.min_gap for s in picks):
                picks.append(b)
            if len(picks) >= p.topk:
                break

        # Store for plots/export
        a["candidate_bounds"] = list(candidates)
        a["scores"] = scores
        a["latency_ms_dict"] = latency_ms

        return picks

    # ----------------------------- Diagnostics UI -----------------------------

    def _update_diagnostics(self, a: Dict) -> None:
        cov = float(a.get("shape_coverage", 1.0))
        kp = int(a.get("known_produced", 0))
        tp = int(a.get("total_produced", 0))
        max_unk = int(a.get("max_unknown_crossing", 0))

        self.var_shape_coverage.set(f"{kp}/{tp} ({100.0*cov:.1f}%)")
        self.var_unknown_crossing.set(str(max_unk))

        if cov < 0.999 or max_unk > 0:
            self.var_diag_note.set("Comm(b) may be underestimated (lower bound)")
        else:
            self.var_diag_note.set("")

    # ----------------------------- Nordstern -----------------------------

    def _compute_nordstern(self, a: Dict, picks: List[int], p: Params) -> None:
        """Compute a pragmatic 'Nordstern' relevance analysis for unknown activation sizes.

        Motivation
        ----------
        If ONNX shape inference cannot determine the size of some intermediate activations,
        the tool can only provide a *lower bound* for Comm(b) because these tensors are
        effectively treated as having size 0.

        This helper:
        - identifies *crossing* tensors with unknown sizes,
        - attaches their producer/consumer span,
        - ranks them by how much they can affect the *currently selected* top-k split candidates.

        The analysis is deliberately lightweight (no symbolic algebra): each unknown tensor is
        treated as an independent variable with coefficient 1 in Comm(b).
        """

        try:
            val_span_all: Dict[str, Tuple[int, int]] = a.get("val_span_all", {})
            value_bytes: Dict[str, int] = a.get("value_bytes", {})
            vimap: Dict[str, onnx.ValueInfoProto] = a.get("vimap", {})
            nodes = a.get("nodes", [])
            order = a.get("order", [])

            unknown_vals = [
                v for v in val_span_all.keys() if int(value_bytes.get(v, 0) or 0) <= 0
            ]

            # Heuristic "unit" size for a single unknown tensor (used only for rough impact numbers).
            known_sizes = [int(b) for b in value_bytes.values() if int(b) > 0]
            if known_sizes:
                try:
                    assume_unknown_bytes = int(statistics.median(known_sizes))
                except Exception:
                    assume_unknown_bytes = int(sum(known_sizes) / max(1, len(known_sizes)))
            else:
                assume_unknown_bytes = 1024 * 1024

            top1 = int(picks[0]) if picks else None

            rows = []
            for v in sorted(unknown_vals):
                span = val_span_all.get(v)
                if not span:
                    continue
                p_pos, last = int(span[0]), int(span[1])
                span_len = max(0, last - p_pos)

                hits = sum(1 for b in picks if p_pos <= int(b) < last)
                affects_top1 = bool(top1 is not None and p_pos <= top1 < last)

                # Metadata
                vi = vimap.get(v)
                dtype = "?"
                shape_s = "?"
                if vi is not None:
                    try:
                        dtype = onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
                    except Exception:
                        dtype = "?"
                    try:
                        sh = asc.shape_from_vi(vi)
                        if sh is None:
                            shape_s = "?"
                        elif len(sh) == 0:
                            # Scalar or unknown rank. We keep '?' because it is ambiguous in ONNX.
                            shape_s = "?"
                        else:
                            shape_s = str(sh)
                    except Exception:
                        shape_s = "?"

                prod_desc = "?"
                last_desc = "?"
                try:
                    if 0 <= p_pos < len(order):
                        n = nodes[int(order[p_pos])]
                        prod_desc = f"{n.op_type}"
                    if 0 <= last < len(order):
                        n = nodes[int(order[last])]
                        last_desc = f"{n.op_type}"
                except Exception:
                    pass

                # Rough impact proxy: how many boundaries would include this tensor if it were large
                impact_mib = (float(span_len) * float(assume_unknown_bytes)) / (1024.0 * 1024.0)

                rows.append(
                    {
                        "name": v,
                        "dtype": dtype,
                        "shape": shape_s,
                        "producer_pos": p_pos,
                        "last_consumer_pos": last,
                        "span_boundaries": span_len,
                        "hits_in_topk": int(hits),
                        "affects_top1": affects_top1,
                        "producer": prod_desc,
                        "last_consumer": last_desc,
                        "impact_mib_proxy": impact_mib,
                    }
                )

            # Sort by: influences current decision first, then by span length
            rows.sort(
                key=lambda r: (
                    -int(r.get("hits_in_topk", 0)),
                    -int(r.get("span_boundaries", 0)),
                    str(r.get("name", "")),
                )
            )

            a["nordstern"] = {
                "unknown_count": int(len(rows)),
                "assume_unknown_bytes": int(assume_unknown_bytes),
                "top1_boundary": top1,
                "rows": rows,
            }
        except Exception as e:
            # Never fail analysis because of Nordstern.
            a["nordstern"] = {
                "unknown_count": 0,
                "assume_unknown_bytes": 0,
                "top1_boundary": None,
                "rows": [],
                "error": f"{type(e).__name__}: {e}",
            }

    def _show_nordstern(self) -> None:
        if self.analysis is None:
            messagebox.showinfo("Nordstern", "Run an analysis first.")
            return

        ns = self.analysis.get("nordstern") or {}
        rows = list(ns.get("rows") or [])
        if not rows:
            # Still show a small dialog (useful feedback)
            msg = "No unknown activation sizes detected for crossing tensors."
            err = ns.get("error")
            if err:
                msg += f"\n\n(note) Nordstern error: {err}"
            messagebox.showinfo("Nordstern", msg)
            return

        assume_b = int(ns.get("assume_unknown_bytes") or 0)
        assume_mib = float(assume_b) / (1024.0 * 1024.0) if assume_b > 0 else 0.0
        top1 = ns.get("top1_boundary")

        win = tk.Toplevel(self)
        win.title("Nordstern — Unknown-size relevance")
        win.geometry("980x520")

        header = ttk.Frame(win)
        header.pack(fill=tk.X, padx=10, pady=(10, 6))

        txt = (
            f"Unknown crossing tensors: {len(rows)}\n"
            f"Proxy size per unknown tensor: {assume_mib:.3f} MiB (median of known activations)\n"
            f"Sorted by: hits in current Top-k (then span length)."
        )
        if top1 is not None:
            txt += f"\nCurrent top-1 boundary: {top1}"

        ttk.Label(header, text=txt, justify=tk.LEFT).pack(side=tk.LEFT)

        body = ttk.Frame(win)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        cols = (
            "name",
            "dtype",
            "shape",
            "producer",
            "last_consumer",
            "span_boundaries",
            "hits_in_topk",
            "affects_top1",
            "impact_mib_proxy",
        )
        tree = ttk.Treeview(body, columns=cols, show="headings", height=16)
        vsb = ttk.Scrollbar(body, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.heading("name", text="Tensor")
        tree.heading("dtype", text="DType")
        tree.heading("shape", text="Shape")
        tree.heading("producer", text="Producer op")
        tree.heading("last_consumer", text="Last consumer op")
        tree.heading("span_boundaries", text="Span (#boundaries)")
        tree.heading("hits_in_topk", text="Hits in Top-k")
        tree.heading("affects_top1", text="Affects Top-1")
        tree.heading("impact_mib_proxy", text="Impact proxy (MiB)")

        tree.column("name", width=260, anchor="w")
        tree.column("dtype", width=80, anchor="center")
        tree.column("shape", width=180, anchor="w")
        tree.column("producer", width=110, anchor="center")
        tree.column("last_consumer", width=120, anchor="center")
        tree.column("span_boundaries", width=120, anchor="e")
        tree.column("hits_in_topk", width=90, anchor="e")
        tree.column("affects_top1", width=90, anchor="center")
        tree.column("impact_mib_proxy", width=110, anchor="e")

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        for r in rows:
            tree.insert(
                "",
                "end",
                values=(
                    r.get("name", ""),
                    r.get("dtype", "?"),
                    r.get("shape", "?"),
                    r.get("producer", "?"),
                    r.get("last_consumer", "?"),
                    int(r.get("span_boundaries", 0)),
                    int(r.get("hits_in_topk", 0)),
                    "yes" if r.get("affects_top1") else "no",
                    f"{float(r.get('impact_mib_proxy', 0.0)):.2f}",
                ),
            )

    # ----------------------------- Table UI -----------------------------

    def _update_table(self, a: Dict, picks: List[int], p: Params):
        self.tree.delete(*self.tree.get_children())

        nodes = a["nodes"]
        order = a["order"]
        costs = a["costs_bytes"]
        counts_all = a["crossing_counts_all"]
        unknown = a["unknown_crossing_counts"]
        flops_left_prefix = a["flops_left_prefix"]
        total_flops = float(a["total_flops"])
        peak_left_b = a.get("peak_act_mem_left_bytes") or []
        peak_right_b = a.get("peak_act_mem_right_bytes") or []
        peak_max_b = a.get("peak_act_mem_max_bytes") or []

        def _mb(x_bytes: float) -> float:
            return float(x_bytes) / 1e6

        for r, b in enumerate(picks, 1):
            lidx, ridx = order[b], order[b + 1]

            cut_mb = _mb(costs[b])
            num_tensors = int(counts_all[b]) if b < len(counts_all) else 0

            fl_l = float(flops_left_prefix[b])
            fl_r = float(total_flops - flops_left_prefix[b])

            gfl_l = fl_l / 1e9
            gfl_r = fl_r / 1e9

            parent = self.tree.insert(
                "",
                "end",
                values=(
                    r,
                    b,
                    nodes[lidx].op_type,
                    nodes[ridx].op_type,
                    f"{cut_mb:.3f}",
                    num_tensors,
                    f"{gfl_l:.3f}",
                    f"{gfl_r:.3f}",
                    f"{(float(peak_left_b[b]) / (1024.0**2)):.2f}" if b < len(peak_left_b) else "",
                    f"{(float(peak_right_b[b]) / (1024.0**2)):.2f}" if b < len(peak_right_b) else "",
                    f"{(float(peak_max_b[b]) / (1024.0**2)):.2f}" if b < len(peak_max_b) else "",
                ),
                tags=("pick",),
            )

            # Unknown-size hint as first child row (if relevant)
            if b < len(unknown) and int(unknown[b]) > 0:
                self.tree.insert(
                    parent,
                    "end",
                    values=("", "↳ unknown sizes", "", "", "", f"+{int(unknown[b])}", "", "", "", "", ""),
                )

            if p.show_top_tensors > 0:
                crossing = asc.collect_crossing_values_for_boundary(b, a["val_span"], a["value_bytes"])
                for name, sz in crossing[: p.show_top_tensors]:
                    self.tree.insert(
                        parent,
                        "end",
                        values=(
                            "",
                            f"↳ {name}",
                            "",
                            "",
                            f"{_mb(sz):.3f}",
                            "",
                            "",
                            "",
                            "",
                            "",
                            "",
                        ),
                    )

    # ----------------------------- Plotting -----------------------------

    def _update_plots(self, a: Dict, picks: List[int], p: Params):
        for ax in (self.ax_comm, self.ax_comp, self.ax_pareto, self.ax_lat):
            ax.clear()

        costs = a["costs_bytes"]
        flops_left_prefix = a["flops_left_prefix"]
        total_flops = float(a["total_flops"])
        imbalance = a["imbalance"]

        M = len(costs)
        xs = list(range(M))

        comm_mb = [float(cb) / 1e6 for cb in costs]
        fl_l_g = [float(f) / 1e9 for f in flops_left_prefix]

        # Peak activation memory arrays (MiB) for paper plots
        peak_left_b = list(a.get("peak_act_mem_left_bytes") or [])[:M]
        peak_right_b = list(a.get("peak_act_mem_right_bytes") or [])[:M]
        peak_max_b = list(a.get("peak_act_mem_max_bytes") or [])[:M]
        peak_left_mib = [float(x) / (1024.0**2) for x in peak_left_b] if peak_left_b else []
        peak_right_mib = [float(x) / (1024.0**2) for x in peak_right_b] if peak_right_b else []
        peak_max_mib = [float(x) / (1024.0**2) for x in peak_max_b] if peak_max_b else []
        fl_r_g = [(float(total_flops - flops_left_prefix[i]) / 1e9) for i in range(M)]

        pick_set = set(picks)

        # (1) Communication bytes per boundary
        bars = self.ax_comm.bar(xs, comm_mb)
        self.ax_comm.set_title("Activation bytes crossing each boundary")
        self.ax_comm.set_xlabel("Boundary index")
        self.ax_comm.set_ylabel("Crossing size (MB)")

        for i, b in enumerate(bars):
            if i in pick_set:
                b.set_edgecolor("black")
                b.set_linewidth(1.5)
                b.set_hatch("//")

        # (2) Cumulative compute
        self.ax_comp.plot(xs, fl_l_g, label="Compute left (GFLOPs)")
        self.ax_comp.plot(xs, fl_r_g, label="Compute right (GFLOPs)")
        self.ax_comp.set_title("Cumulative compute around boundaries")
        self.ax_comp.set_xlabel("Boundary index")
        self.ax_comp.set_ylabel("GFLOPs")
        self.ax_comp.legend(loc="best")

        for b in picks:
            self.ax_comp.axvline(b, linestyle="--", linewidth=1)

        # (3) Pareto plot: comm vs imbalance
        cand_bounds = a.get("candidate_bounds") or list(range(M))
        pts = [(float(costs[b]) / 1e6, float(imbalance[b])) for b in cand_bounds]
        self.ax_pareto.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], s=12)
        self.ax_pareto.set_title("Pareto: communication vs imbalance")
        self.ax_pareto.set_xlabel("Cut (MB)")
        self.ax_pareto.set_ylabel("Imbalance |L-R|/Total")

        if p.show_pareto_front and pts:
            fi = pareto_front(pts)
            fi = sorted(fi, key=lambda i: pts[i][0])
            self.ax_pareto.plot([pts[i][0] for i in fi], [pts[i][1] for i in fi], linewidth=2)

        if picks:
            px = [float(costs[b]) / 1e6 for b in picks]
            py = [float(imbalance[b]) for b in picks]
            self.ax_pareto.scatter(px, py, s=40, marker="o", edgecolors="black")

        # (4) Latency model plot
        bw_bps = asc.bandwidth_to_bytes_per_s(p.bw_value, p.bw_unit)
        if bw_bps is not None and p.gops_left is not None and p.gops_right is not None and total_flops > 0:
            gl = float(p.gops_left)
            gr = float(p.gops_right)

            lat_ms = [
                1000.0
                * (
                    float(flops_left_prefix[b]) / (gl * 1e9)
                    + float(costs[b]) / float(bw_bps)
                    + float(total_flops - flops_left_prefix[b]) / (gr * 1e9)
                )
                + float(p.overhead_ms)
                for b in range(M)
            ]
            comm_ms = [1000.0 * (float(costs[b]) / float(bw_bps)) for b in range(M)]

            self.ax_lat.plot(xs, lat_ms, label="Total latency (ms)")
            self.ax_lat.plot(xs, comm_ms, label="Comm-only (ms)")
            self.ax_lat.set_title("Latency model vs boundary")
            self.ax_lat.set_xlabel("Boundary index")
            self.ax_lat.set_ylabel("ms")
            self.ax_lat.legend(loc="best")

            for b in picks:
                self.ax_lat.axvline(b, linestyle="--", linewidth=1)
        else:
            self.ax_lat.set_title("Latency model (set bandwidth + GOPS L/R)")
            self.ax_lat.set_xlabel("Boundary index")
            self.ax_lat.set_ylabel("ms")
            self.ax_lat.text(
                0.05,
                0.6,
                "Provide:\n- link bandwidth\n- GOPS left / right\n(optional overhead)",
                transform=self.ax_lat.transAxes,
            )

        self.canvas.draw_idle()

    # ----------------------------- Split models -----------------------------


    def _is_boundary_row(self, item: str) -> bool:
        """Return True if the item is a *top-level* boundary row (not a child tensor row)."""
        if not item:
            return False
        if self.tree.parent(item):
            return False
        vals = self.tree.item(item, "values")
        if len(vals) < 2:
            return False
        try:
            int(vals[0])  # rank
            int(vals[1])  # boundary index
            return True
        except Exception:
            return False

    def _selected_boundary_index(self) -> Optional[int]:
        """Return the selected boundary index, but ONLY if a boundary row is selected."""
        sel = self.tree.selection()
        if not sel:
            return None
        item = sel[0]
        if not self._is_boundary_row(item):
            return None
        vals = self.tree.item(item, "values")
        try:
            return int(vals[1])
        except Exception:
            return None

    def _update_action_buttons(self) -> None:
        """Enable/disable buttons based on current state + selection."""
        if self.analysis is None:
            try:
                self.btn_export_tex.state(["disabled"])
                self.btn_split.state(["disabled"])
                self.btn_nordstern.state(["disabled"])
                self.btn_benchmark.state(["disabled"])
            except Exception:
                pass
            return

        # Export table only if we have picks
        if self.current_picks:
            self.btn_export_tex.state(["!disabled"])
        else:
            self.btn_export_tex.state(["disabled"])

        # Benchmark-set generation also requires picks
        if self.current_picks:
            self.btn_benchmark.state(["!disabled"])
        else:
            self.btn_benchmark.state(["disabled"])

        # Split only if a boundary row is selected
        if self._selected_boundary_index() is None:
            self.btn_split.state(["disabled"])
        else:
            self.btn_split.state(["!disabled"])

        # Nordstern only if unknown sizes exist
        try:
            ns = (self.analysis or {}).get("nordstern") or {}
            if int(ns.get("unknown_count") or 0) > 0:
                self.btn_nordstern.state(["!disabled"])
            else:
                self.btn_nordstern.state(["disabled"])
        except Exception:
            pass


    def _split_selected_boundary(self) -> None:
        if self.analysis is None or self.model_path is None:
            messagebox.showinfo("Nothing to split", "Load a model and run an analysis first.")
            return

        b = self._selected_boundary_index()
        if b is None:
            messagebox.showinfo(
                "Select a boundary", "Select a *boundary* row in the table (not a ↳ tensor row)."
            )
            return

        a = self.analysis
        order = a["order"]
        nodes = a["nodes"]
        model = a["model"]

        try:
            cut_tensors = asc.cut_tensors_for_boundary(order, nodes, int(b))
        except Exception as e:
            messagebox.showerror("Cut tensor error", f"Failed to determine cut tensors: {e}")
            return

        if not cut_tensors:
            messagebox.showinfo("No cut", "Selected boundary produced no cut tensors (unexpected).")
            return

        # If there is exactly one activation crossing, allow optional rename for stricter cut.
        p1_cut_names = None
        p2_cut_names = None
        if len(cut_tensors) == 1:
            default_name = cut_tensors[0]
            new_name = simpledialog.askstring(
                "Strict cut name",
                "Optional: rename the cut activation tensor for the split models.\n\n"
                "- Leave empty to keep the original name.\n"
                "- Helpful when you want to enforce a single, explicit boundary tensor.",
                initialvalue=default_name,
            )
            if new_name is not None:
                new_name = new_name.strip()
                if new_name and new_name != default_name:
                    # Keep original cut tensor name (must exist in the full graph),
                    # but rename the boundary tensor inside the split sub-models.
                    p1_cut_names = [new_name]
                    p2_cut_names = [new_name]

        out_parent = filedialog.askdirectory(title="Select output folder")
        if not out_parent:
            return

        base = os.path.splitext(os.path.basename(self.model_path))[0]
        export_as_folder = bool(self.var_split_folder.get())
        if export_as_folder:
            out_dir = os.path.join(out_parent, f"{base}_split_b{b}")
        else:
            out_dir = out_parent
        os.makedirs(out_dir, exist_ok=True)

        p1_path = os.path.join(out_dir, f"{base}_part1_b{b}.onnx")
        p2_path = os.path.join(out_dir, f"{base}_part2_b{b}.onnx")
        manifest_path = os.path.join(out_dir, "split_manifest.json")

        strict_boundary = bool(self.var_strict_boundary.get())

        prune_skip_block = bool(self.var_prune_skip_block.get())
        skip_min_span = _safe_int(self.var_skip_min_span.get())
        if skip_min_span is None or skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = _safe_int(self.var_skip_allow_last_n.get())
        if skip_allow_last_n is None:
            skip_allow_last_n = 0
        if skip_allow_last_n < 0:
            raise ValueError("Allow last N inside must be an integer ≥ 0.")
        do_validate = bool(self.var_split_validate.get())
        do_runner = bool(self.var_split_runner.get())
        runner_target = str(self.var_runner_target.get() or "auto")

        do_ctx_full = bool(getattr(self, 'var_split_ctx_full', tk.BooleanVar(value=True)).get())
        do_ctx_cutflow = bool(getattr(self, 'var_split_ctx_cutflow', tk.BooleanVar(value=False)).get())

        ctx_hops = _safe_int(getattr(self, 'var_split_ctx_hops', tk.StringVar(value='2')).get()) or 2


        eps_txt = (self.var_split_eps.get() or "").strip()
        eps = None
        if eps_txt:
            try:
                eps = float(eps_txt)
            except Exception:
                messagebox.showerror(
                    "Invalid eps",
                    f"Could not parse eps='{eps_txt}'. Use e.g. 1e-4 or leave empty.",
                )
                return

        # Read batch override once here (avoid reading Tk variables from worker thread).
        params = None
        try:
            params = self._read_params()
            batch_override = params.batch_override
        except Exception:
            batch_override = None

        # ---- progress dialog + background worker ----
        dlg = tk.Toplevel(self)
        dlg.title("Splitting…")
        dlg.transient(self)
        try:
            dlg.grab_set()
        except Exception:
            pass
        dlg.resizable(False, False)

        ttk.Label(dlg, text=f"Splitting boundary {b} and exporting artifacts…").pack(
            padx=16, pady=(16, 8)
        )
        pb = ttk.Progressbar(dlg, mode="indeterminate", length=320)
        pb.pack(padx=16, pady=(0, 16))
        pb.start(10)

        try:
            self.configure(cursor="watch")
            self.update_idletasks()
        except Exception:
            pass

        q: "queue.Queue[tuple[str, str]]" = queue.Queue()

        def worker() -> None:
            try:
                msg = []

                # Split
                p1, p2, split_manifest = asc.split_model_on_cut_tensors(
                    model,
                    cut_tensors=cut_tensors,
                    strict_boundary=strict_boundary,
                    p1_cut_names=p1_cut_names,
                    p2_cut_names=p2_cut_names,
                )
                asc.save_model(p1, p1_path)
                asc.save_model(p2, p2_path)

                msg.append(f"Boundary: {b}")
                msg.append(f"Cut tensors: {cut_tensors}")
                msg.append(f"Strict boundary: {strict_boundary}")
                msg.append(f"Wrote: {p1_path}")
                msg.append(f"Wrote: {p2_path}")

                # Manifest (must include cut-name mapping for the runner)
                manifest_out = {
                    "tool": {
                        "gui": __version__,
                        "core": getattr(asc, "__version__", "?"),
                    },
                    "boundary": int(b),
                    "cut_tensors": cut_tensors,
                    "strict_boundary": strict_boundary,
                    "split_context_hops": int(ctx_hops),
                    "full_model": os.path.abspath(self.model_path),
                    "part1": os.path.abspath(p1_path),
                    "part2": os.path.abspath(p2_path),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                if isinstance(split_manifest, dict):
                    manifest_out.update(split_manifest)

                # Split-context diagrams around the boundary (GraphViz if available; otherwise fallback).
                if do_ctx_full:
                    try:
                        ctx = asc.export_boundary_graphviz_context(
                            model,
                            order,
                            int(b),
                            cut_tensors,
                            out_dir,
                            basename=f"split_context_b{b}",
                            render=True,
                            hops=int(ctx_hops),
                        )
                        manifest_out["split_context"] = ctx
                        if isinstance(ctx, dict):
                            for k in ("dot", "svg", "pdf", "png"):
                                if ctx.get(k):
                                    msg.append(f"Wrote: {os.path.join(out_dir, ctx[k])}")
                    except Exception as e:
                        msg.append(f"Split context export (full) failed: {e}")

                if do_ctx_cutflow:
                    try:
                        ctx_cf = asc.export_boundary_graphviz_context(
                            model,
                            order,
                            int(b),
                            cut_tensors,
                            out_dir,
                            basename=f"split_context_b{b}_cutflow",
                            render=True,
                            hops=int(ctx_hops),
                            cut_flow_only=True,
                            include_internal_consumers=False,
                            include_external_inputs=False,
                        )
                        manifest_out["split_context_cutflow"] = ctx_cf
                        if isinstance(ctx_cf, dict):
                            for k in ("dot", "svg", "pdf", "png"):
                                if ctx_cf.get(k):
                                    msg.append(f"Wrote: {os.path.join(out_dir, ctx_cf[k])}")
                    except Exception as e:
                        msg.append(f"Split context export (cut-flow) failed: {e}")


                # Runner skeleton
                if do_runner:
                    try:
                        runner_path = asc.write_runner_skeleton_onnxruntime(
                            out_dir,
                            manifest_filename=os.path.basename(manifest_path),
                            target=runner_target,
                        )
                        manifest_out["runner"] = os.path.basename(runner_path)
                        msg.append(f"Runner skeleton ({runner_target}): {runner_path}")
                    except Exception as e:
                        msg.append(f"Runner skeleton failed: {e}")

                # Optional ORT validation (quick numeric check, no timing report)
                if do_validate:
                    try:
                        res = asc.validate_split_onnxruntime(
                            full_model_path=os.path.abspath(self.model_path),
                            part1_path=os.path.abspath(p1_path),
                            part2_path=os.path.abspath(p2_path),
                            manifest=manifest_out,
                            batch_override=batch_override,
                            eps=eps,
                        )
                        ok = bool(res.get("ok", False)) if isinstance(res, dict) else False
                        val_path = os.path.join(out_dir, "validation_core.json")
                        try:
                            with open(val_path, "w", encoding="utf-8") as f:
                                json.dump(res, f, indent=2)
                            msg.append(f"Wrote: {val_path}")
                        except Exception:
                            pass
                        msg.append(f"ORT validation: {'PASS' if ok else 'FAIL'}")
                    except Exception as e:
                        msg.append(f"ORT validation failed: {e}")

                # Write manifest last (so it contains runner/context/validation fields)
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_out, f, indent=2)
                msg.append(f"Wrote: {manifest_path}")

                q.put(("ok", "\n".join(msg)))
            except Exception as e:
                q.put(("err", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()

        def poll() -> None:
            try:
                status, payload = q.get_nowait()
            except queue.Empty:
                self.after(100, poll)
                return

            try:
                pb.stop()
            except Exception:
                pass
            try:
                dlg.destroy()
            except Exception:
                pass

            try:
                self.configure(cursor="")
            except Exception:
                pass

            if status == "ok":
                messagebox.showinfo("Split complete", payload)
            else:
                messagebox.showerror("Split failed", payload)

        self.after(100, poll)

    # ----------------------------- Export: plots -----------------------------

    def _generate_benchmark_set(self) -> None:
        """Generate a benchmark suite folder for the current model + top-k picks.

        The suite contains one subfolder per split candidate with:
          - part1 / part2 ONNX models
          - split_manifest.json
          - run_split_onnxruntime.py runner script

        At the top level it also contains:
          - benchmark_set.json (list of cases + predicted metrics)
          - benchmark_suite.py (runs all cases and collects results/plots)
        """
        if self.analysis is None or self.model_path is None:
            messagebox.showinfo("Nothing to benchmark", "Load a model and run an analysis first.")
            return
        if not self.current_picks:
            messagebox.showinfo(
                "No candidates",
                "No split candidates available. Try increasing Top-K and re-run Analyse.",
            )
            return

        out_parent = filedialog.askdirectory(title="Select parent folder for benchmark set")
        if not out_parent:
            return

        base = os.path.splitext(os.path.basename(self.model_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_parent, f"{base}_benchmark_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        # Pull analysis objects once (used for strict-boundary filtering and TeX/plot export).
        a = self.analysis
        strict_boundary = bool(self.var_strict_boundary.get())

        prune_skip_block = bool(self.var_prune_skip_block.get())
        skip_min_span = _safe_int(self.var_skip_min_span.get())
        if skip_min_span is None or skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = _safe_int(self.var_skip_allow_last_n.get())
        if skip_allow_last_n is None:
            skip_allow_last_n = 0
        if skip_allow_last_n < 0:
            raise ValueError("Allow last N inside must be an integer ≥ 0.")

        # Convenience locals used during strict-boundary validation.
        model = a.get("model") if isinstance(a, dict) else None
        nodes = a.get("nodes") if isinstance(a, dict) else None
        order = a.get("order") if isinstance(a, dict) else None
        if model is None or nodes is None or order is None:
            messagebox.showerror(
                "Benchmark set failed",
                "Internal error: analysis data missing (model/nodes/order). Please re-run Analyse.",
            )
            return

        # How many candidates to export?
        # Note: We use the *ranked candidate list* as source of truth so we can fall back to additional
        # candidates if some splits fail (e.g., strict-boundary violations, missing shapes).

        ranked_candidates: List[int] = list((a.get("candidate_bounds") or self.current_picks) if isinstance(a, dict) else self.current_picks)

        # If the user enabled "Strict boundary" AFTER running Analyse (or the analysis was done without
        # strict-boundary metadata), we defensively re-check strictness here and filter candidates.
        if strict_boundary:
            strict_ok = a.get("strict_ok") if isinstance(a, dict) else None
            if isinstance(strict_ok, list) and len(strict_ok) > 0:
                ranked_candidates = [b for b in ranked_candidates if 0 <= b < len(strict_ok) and bool(strict_ok[b])]

            # Always enforce strictness via a direct graph check. This avoids stale metadata (e.g. when the
            # analysis was performed with "Strict boundary" disabled, which sets strict_ok=[True,...]).
            filtered: List[int] = []
            for b in ranked_candidates:
                try:
                    cut_tensors = asc.cut_tensors_for_boundary(order, nodes, b)
                    extras = asc.strict_boundary_extras(model, cut_tensors)
                except Exception:
                    # If we cannot validate strictness, treat it as NOT strict.
                    continue
                if len(extras) == 0:
                    filtered.append(b)
            ranked_candidates = filtered

            if not ranked_candidates:
                messagebox.showinfo(
                    "No strict candidates",
                    "Strict boundary is enabled, but none of the available candidates satisfy the strict-boundary condition.\n\n"
                    "Tip: disable Strict boundary or re-run Analyse with Strict boundary unchecked.",
                )
                return

        default_k = min(10, len(ranked_candidates))
        k = simpledialog.askinteger(
            "Benchmark set",
            f"How many splits to generate for the benchmark set? (max {len(ranked_candidates)})",
            initialvalue=default_k,
            minvalue=1,
            maxvalue=len(ranked_candidates),
        )

        if k is None:
            return
        k = int(k)

        # Export analysis artefacts (plots + TeX table) into the benchmark folder for paper usage.
        try:
            self._export_benchmark_paper_assets(Path(out_dir), a, ranked_candidates[:k])
        except Exception as e:
            print(f"[warn] Failed to export paper assets into benchmark folder: {type(e).__name__}: {e}")

        # For a benchmark set we ALWAYS generate a runner skeleton (otherwise the suite isn't runnable).
        do_runner = True

        # Runner skeleton target (auto/cpu/cuda/tensorrt). Read it once here so the
        # background worker does not access Tk variables.
        runner_target = "auto"
        try:
            runner_target = str(self.var_runner_target.get() or "auto").strip().lower()
        except Exception:
            runner_target = "auto"
        if runner_target not in {"auto", "cpu", "cuda", "tensorrt"}:
            runner_target = "auto"

        do_ctx_full = bool(getattr(self, 'var_split_ctx_full', tk.BooleanVar(value=True)).get())
        do_ctx_cutflow = bool(getattr(self, 'var_split_ctx_cutflow', tk.BooleanVar(value=False)).get())
        ctx_hops = _safe_int(getattr(self, 'var_split_ctx_hops', tk.StringVar(value='2')).get()) or 2

        eps_txt = (self.var_split_eps.get() or "").strip()
        eps_default = 1e-4
        if eps_txt:
            try:
                eps_default = float(eps_txt)
            except Exception:
                eps_default = 1e-4

        # Read batch override once here (avoid reading Tk variables from worker thread).
        params = None
        try:
            params = self._read_params()
            batch_override = params.batch_override
        except Exception:
            batch_override = None

        # Determine a nice padding width for folder names.
        pad = max(3, len(str(max(self.current_picks)))) if self.current_picks else 3

        # --- progress dialog + background worker ---
        dlg = tk.Toplevel(self)
        dlg.title("Generating benchmark set…")
        dlg.transient(self)
        try:
            dlg.grab_set()
        except Exception:
            pass
        dlg.resizable(False, False)

        lbl = ttk.Label(dlg, text=f"Generating benchmark set with up to {k} cases…")
        lbl.pack(padx=16, pady=(16, 8))

        pb = ttk.Progressbar(dlg, mode="determinate", length=360, maximum=max(1, k))
        pb.pack(padx=16, pady=(0, 4))
        pb["value"] = 0

        lbl2 = ttk.Label(dlg, text="")
        lbl2.pack(padx=16, pady=(0, 16))

        try:
            self.configure(cursor="watch")
            self.update_idletasks()
        except Exception:
            pass

        q: "queue.Queue[tuple]" = queue.Queue()

        def _write_benchmark_suite_script(dst_dir: str, bench_json_name: str = "benchmark_set.json") -> str:
            """Write a tiny benchmark harness that runs all exported splits and aggregates results."""
            script_path = os.path.join(dst_dir, "benchmark_suite.py")
            script = r'''#!/usr/bin/env python3
"""
Benchmark harness for an ONNX split benchmark set.

- Runs each case folder (bXXX/) via its generated runner (run_split_onnxruntime.py)
- Collects validation reports (timing + eps-pass + optional agreement KPIs)
- Writes:
  * benchmark_results_<tag>.json  (canonical)
  * benchmark_results_<tag>.csv   (optional convenience)
  * benchmark_summary_<tag>.md    (1-page summary)
  * benchmark_table_<tag>.tex     (paper-ready table)
  * paper_figures_<tag>/...       (paper-ready plots as PDF + SVG)

Notes:
- For ARCS scope (single machine/provider), the default objective is end-to-end latency
  measured by the composed graph (two sessions: part1 -> part2).
- For future work (multi-device pipelining), the bottleneck objective max(part1, part2)
  can be selected via --objective=max_parts.
"""

from __future__ import annotations

import argparse
import json
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - xm) ** 2 for a in x))
    deny = math.sqrt(sum((b - ym) ** 2 for b in y))
    den = denx * deny
    return (num / den) if den > 0 else float("nan")


def _spearman(x: List[float], y: List[float]) -> float:
    # Spearman correlation via Pearson on ranks (average ranks for ties).
    def _ranks(vals: List[float]) -> List[float]:
        n = len(vals)
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        # ranks are 1..n
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            # average rank for ties
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _ranks([float(v) for v in x])
    ry = _ranks([float(v) for v in y])
    return _pearson(rx, ry)


def _kendall_tau(x: List[float], y: List[float]) -> float:
    # simple O(n^2) kendall tau-a (good enough for small K)
    n = len(x)
    if n != len(y) or n < 2:
        return float("nan")
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            s = dx * dy
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    denom = conc + disc
    return (conc - disc) / denom if denom > 0 else float("nan")


def _linreg(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    # y = a*x + b, returns (a,b,R^2)
    import numpy as np  # type: ignore

    if len(x) != len(y) or len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float)
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    Yhat = a * X + b
    ss_res = float(((Y - Yhat) ** 2).sum())
    ss_tot = float(((Y - float(Y.mean())) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(a), float(b), float(r2)


def _fmt(x: Any, nd: int = 3) -> str:
    try:
        if x is None:
            return "-"
        if isinstance(x, bool):
            return "True" if x else "False"
        xf = float(x)
        if math.isnan(xf):
            return "nan"
        return f"{xf:.{nd}f}"
    except Exception:
        return str(x)


def _objective_value(row: Dict[str, Any], objective: str) -> Optional[float]:
    full = row.get("full_mean_ms")
    p1 = row.get("part1_mean_ms")
    p2 = row.get("part2_mean_ms")
    comp = row.get("composed_mean_ms")
    if objective == "full":
        return float(full) if full is not None else None
    if objective == "composed":
        return float(comp) if comp is not None else None
    if objective == "sum_parts":
        if p1 is None or p2 is None:
            return None
        return float(p1) + float(p2)
    if objective == "max_parts":
        if p1 is None or p2 is None:
            return None
        return max(float(p1), float(p2))
    raise ValueError(f"unknown objective: {objective}")


def _write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv

    if not rows:
        return
    keys: List[str] = []
    # stable-ish key order
    preferred = [
        "boundary",
        "cut_mib",
        "n_cut_tensors",
        "flops_left",
        "flops_right",
        "imbalance_pred",
        "score_pred",
        "eps_pass",
        "full_mean_ms",
        "full_std_ms",
        "part1_mean_ms",
        "part1_std_ms",
        "part2_mean_ms",
        "part2_std_ms",
        "composed_mean_ms",
        "composed_std_ms",
        "sum_parts_ms",
        "overhead_ms",
        "speedup_full_over_composed",
    ]
    for k in preferred:
        if k in rows[0]:
            keys.append(k)
    for k in rows[0].keys():
        if k not in keys:
            keys.append(k)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_table_tex(path: Path, rows: List[Dict[str, Any]], tag: str, objective: str, topk: int = 10) -> None:
    def _tex_escape(text: str) -> str:
        # minimal escaping for captions/tt
        return (
            text.replace("\\", "\\textbackslash{}")
                .replace("_", "\\_")
                .replace("%", "\\%")
                .replace("&", "\\&")
                .replace("#", "\\#")
                .replace("$", "\\$")
                .replace("{", "\\{")
                .replace("}", "\\}")
        )

    label_tag = "bench-" + tag.replace("_", "-")
    # top-k by objective (ascending)
    def keyfn(r: Dict[str, Any]) -> float:
        v = _objective_value(r, objective)
        return float("inf") if v is None else float(v)

    rows_sorted = sorted(rows, key=keyfn)[: max(1, min(topk, len(rows)))]

    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{rrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Boundary & Cut (MiB) & \#T & $F_L$ (GFLOPs) & $F_R$ (GFLOPs) & $t_{\mathrm{full}}$ (ms) & $t_{\mathrm{comp}}$ (ms) & Pass \\")
    lines.append(r"\midrule")
    for r in rows_sorted:
        b = r.get("boundary", "-")
        cut = _fmt(r.get("cut_mib"), 3)
        nt = r.get("cut_tensors", r.get("n_cut_tensors", "-"))
        fl = _fmt(float(r.get("flops_left", 0.0)) / 1e9 if r.get("flops_left") is not None else None, 2)
        fr = _fmt(float(r.get("flops_right", 0.0)) / 1e9 if r.get("flops_right") is not None else None, 2)
        tf = _fmt(r.get("full_mean_ms"), 2)
        tc = _fmt(r.get("composed_mean_ms"), 2)
        ps = r.get("ok", r.get("eps_pass"))
        ps_s = "True" if ps is True else ("False" if ps is False else "-")
        lines.append(f"{b} & {cut} & {nt} & {fl} & {fr} & {tf} & {tc} & {ps_s} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    tag_tt = _tex_escape(tag)
    obj_tt = _tex_escape(objective)
    lines.append(r"\\caption{Top split candidates for \\texttt{" + tag_tt + r"} (sorted by objective: \\texttt{" + obj_tt + r"}).}")
    lines.append(r"\label{tab:benchmark_top_" + label_tag + r"}")
    lines.append(r"\end{table}")
    lines.append("")  # trailing newline

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_paper_figures(out_dir: Path, rows: List[Dict[str, Any]], tag: str, objective: str, topk: int = 10) -> List[Path]:
    if not HAVE_PLOT or not rows:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare arrays
    flL = [float(r.get("flops_left", float("nan"))) / 1e9 for r in rows]
    flR = [float(r.get("flops_right", float("nan"))) / 1e9 for r in rows]
    t1 = [float(r.get("part1_mean_ms", float("nan"))) for r in rows]
    t2 = [float(r.get("part2_mean_ms", float("nan"))) for r in rows]
    score = [float(r.get("score", r.get("score_pred", float("nan")))) for r in rows]
    cut = [float(r.get("cut_mib", float("nan"))) for r in rows]
    comp = [float(r.get("composed_mean_ms", float("nan"))) for r in rows]
    full = [float(r.get("full_mean_ms", float("nan"))) for r in rows]
    sum_parts = [float(r.get("sum_parts_ms", float("nan"))) for r in rows]
    overhead = [float(r.get("overhead_ms", float("nan"))) for r in rows]

    obj = [float(_objective_value(r, objective) or float("nan")) for r in rows]

    files: List[Path] = []

    # Fig 1: FLOPs vs time (left)
    a, b, r2 = _linreg(flL, t1)
    r = _pearson(flL, t1)
    plt.figure(figsize=(5.2, 4.0))
    plt.scatter(flL, t1)
    xline = [min(flL), max(flL)]
    yline = [a * x + b for x in xline]
    plt.plot(xline, yline)
    plt.xlabel("FLOPs left (GFLOPs)")
    plt.ylabel("Measured time part1 (ms)")
    plt.title(f"Compute model plausibility (left)  r={_fmt(r,2)}, R$^2$={_fmt(r2,2)}")
    p_pdf = out_dir / f"fig_flops_vs_time_left_{tag}.pdf"
    p_svg = out_dir / f"fig_flops_vs_time_left_{tag}.svg"
    plt.tight_layout()
    plt.savefig(p_pdf)
    plt.savefig(p_svg)
    plt.close()
    files += [p_pdf, p_svg]

    # Fig 2: FLOPs vs time (right)
    a, b, r2 = _linreg(flR, t2)
    r = _pearson(flR, t2)
    plt.figure(figsize=(5.2, 4.0))
    plt.scatter(flR, t2)
    xline = [min(flR), max(flR)]
    yline = [a * x + b for x in xline]
    plt.plot(xline, yline)
    plt.xlabel("FLOPs right (GFLOPs)")
    plt.ylabel("Measured time part2 (ms)")
    plt.title(f"Compute model plausibility (right)  r={_fmt(r,2)}, R$^2$={_fmt(r2,2)}")
    p_pdf = out_dir / f"fig_flops_vs_time_right_{tag}.pdf"
    p_svg = out_dir / f"fig_flops_vs_time_right_{tag}.svg"
    plt.tight_layout()
    plt.savefig(p_pdf)
    plt.savefig(p_svg)
    plt.close()
    files += [p_pdf, p_svg]

    # Fig 3: score vs objective time (ranking quality proxy)
    # highlight top-k predicted and top-k measured (objective)
    import numpy as np  # type: ignore

    score_arr = np.asarray(score, dtype=float)
    obj_arr = np.asarray(obj, dtype=float)
    valid = np.isfinite(score_arr) & np.isfinite(obj_arr)
    score_v = score_arr[valid]
    obj_v = obj_arr[valid]

    if len(score_v) >= 2:
        # indices in original list of valid points
        valid_idx = np.where(valid)[0]
        k = max(1, min(topk, len(score_v)))
        pred_order = np.argsort(score_v)  # smaller score => better
        meas_order = np.argsort(obj_v)    # smaller time => better
        pred_top = set(valid_idx[pred_order[:k]].tolist())
        meas_top = set(valid_idx[meas_order[:k]].tolist())

        colors = []
        for i in range(len(rows)):
            if i in pred_top and i in meas_top:
                colors.append("green")  # agree in top-k
            elif i in pred_top:
                colors.append("red")    # predicted top-k only
            elif i in meas_top:
                colors.append("orange") # measured top-k only
            else:
                colors.append(None)

        plt.figure(figsize=(5.6, 4.2))
        for i, r0 in enumerate(rows):
            if not (math.isfinite(score[i]) and math.isfinite(obj[i])):
                continue
            if colors[i] is None:
                plt.scatter(score[i], obj[i], alpha=0.6)
            else:
                plt.scatter(score[i], obj[i], edgecolors="black", linewidths=0.5, s=60, c=colors[i])
        sp = _spearman([float(s) for s in score_v], [float(t) for t in obj_v])
        kt = _kendall_tau([float(s) for s in score_v], [float(t) for t in obj_v])
        plt.xlabel("Predicted score (higher is better)")
        plt.ylabel(f"Measured objective time: {objective} (ms)")
        plt.title(f"Ranking agreement: Spearman={_fmt(sp,2)}, Kendall={_fmt(kt,2)}")
        p_pdf = out_dir / f"fig_score_vs_{objective}_{tag}.pdf"
        p_svg = out_dir / f"fig_score_vs_{objective}_{tag}.svg"
        plt.tight_layout()
        plt.savefig(p_pdf)
        plt.savefig(p_svg)
        plt.close()
        files += [p_pdf, p_svg]

    # Fig 4: noise summary (CV%)
    def cv(mean: float, std: float) -> float:
        return (std / mean) * 100.0 if mean and mean > 0 else float("nan")

    cv_full = [cv(float(r.get("full_mean_ms", float("nan"))), float(r.get("full_std_ms", float("nan")))) for r in rows]
    cv_p1 = [cv(float(r.get("part1_mean_ms", float("nan"))), float(r.get("part1_std_ms", float("nan")))) for r in rows]
    cv_p2 = [cv(float(r.get("part2_mean_ms", float("nan"))), float(r.get("part2_std_ms", float("nan")))) for r in rows]
    cv_comp = [cv(float(r.get("composed_mean_ms", float("nan"))), float(r.get("composed_std_ms", float("nan")))) for r in rows]

    def _mean_std(vals: List[float]) -> Tuple[float, float]:
        vv = [v for v in vals if math.isfinite(v)]
        if not vv:
            return float("nan"), float("nan")
        m = sum(vv) / len(vv)
        s = math.sqrt(sum((v - m) ** 2 for v in vv) / max(1, len(vv) - 1))
        return m, s

    names = ["full", "part1", "part2", "composed"]
    means = []
    stds = []
    for arr in [cv_full, cv_p1, cv_p2, cv_comp]:
        m, s = _mean_std(arr)
        means.append(m)
        stds.append(s)

    plt.figure(figsize=(5.4, 3.6))
    plt.bar(names, means, yerr=stds)
    plt.ylabel("CV (std/mean) [%]")
    plt.title("Measurement noise (across benchmark cases)")
    p_pdf = out_dir / f"fig_noise_cv_{tag}.pdf"
    p_svg = out_dir / f"fig_noise_cv_{tag}.svg"
    plt.tight_layout()
    plt.savefig(p_pdf)
    plt.savefig(p_svg)
    plt.close()
    files += [p_pdf, p_svg]

    # Fig 5 (optional): overhead vs cut size
    if any(math.isfinite(v) for v in overhead) and any(math.isfinite(v) for v in cut):
        plt.figure(figsize=(5.4, 3.8))
        plt.scatter(cut, overhead)
        plt.axhline(0.0)
        plt.xlabel("Cut size (MiB)")
        plt.ylabel("Overhead: composed - (part1+part2) [ms]")
        plt.title("Split overhead vs cut size")
        p_pdf = out_dir / f"fig_overhead_vs_cut_{tag}.pdf"
        p_svg = out_dir / f"fig_overhead_vs_cut_{tag}.svg"
        plt.tight_layout()
        plt.savefig(p_pdf)
        plt.savefig(p_svg)
        plt.close()
        files += [p_pdf, p_svg]

    return files


def _write_summary_md(path: Path, rows: List[Dict[str, Any]], tag: str, objective: str, topk: int = 10) -> None:
    if not rows:
        path.write_text("# Benchmark summary\n\nNo results.\n", encoding="utf-8")
        return

    # derived objective arrays
    obj = [(_objective_value(r, objective) or float("nan")) for r in rows]
    score = [float(r.get("score", r.get("score_pred", float("nan")))) for r in rows]

    # compute model plausibility
    flL = [float(r["flops_left"]) / 1e9 for r in rows]
    flR = [float(r["flops_right"]) / 1e9 for r in rows]
    t1 = [float(r["part1_mean_ms"]) for r in rows]
    t2 = [float(r["part2_mean_ms"]) for r in rows]
    rL = _pearson(flL, t1)
    rR = _pearson(flR, t2)
    aL, bL, r2L = _linreg(flL, t1)
    aR, bR, r2R = _linreg(flR, t2)

    # ranking agreement (score vs objective time)
    import numpy as np  # type: ignore

    score_arr = np.asarray(score, dtype=float)
    obj_arr = np.asarray(obj, dtype=float)
    valid = np.isfinite(score_arr) & np.isfinite(obj_arr)
    score_v = score_arr[valid].tolist()
    obj_v = obj_arr[valid].tolist()
    sp = _spearman([float(s) for s in score_v], [-float(t) for t in obj_v]) if len(score_v) >= 2 else float("nan")
    kt = _kendall_tau([float(s) for s in score_v], [-float(t) for t in obj_v]) if len(score_v) >= 2 else float("nan")

    # top-k overlap (pred score vs measured objective)
    k = max(1, min(topk, len(score_v)))
    pred_order = sorted(range(len(score_v)), key=lambda i: score_v[i], reverse=True)
    meas_order = sorted(range(len(obj_v)), key=lambda i: obj_v[i])
    overlap = len(set(pred_order[:k]).intersection(set(meas_order[:k])))
    overlap_ratio = overlap / k

    # noise summary (CV)
    def cv(mean: float, std: float) -> float:
        return (std / mean) * 100.0 if mean and mean > 0 else float("nan")

    cv_comp = [cv(float(r.get("composed_mean_ms", float("nan"))), float(r.get("composed_std_ms", float("nan")))) for r in rows]
    cv_comp_v = [v for v in cv_comp if math.isfinite(v)]
    cv_comp_med = sorted(cv_comp_v)[len(cv_comp_v) // 2] if cv_comp_v else float("nan")

    # best splits by objective
    def keyfn(r: Dict[str, Any]) -> float:
        v = _objective_value(r, objective)
        return float("inf") if v is None else float(v)

    best = sorted(rows, key=keyfn)[: min(10, len(rows))]

    lines: List[str] = []
    lines.append(f"# Benchmark summary: {tag}\n")
    lines.append(f"- Objective: **{objective}** (lower is better)\n")
    lines.append("## Compute-model plausibility (FLOPs vs measured time)\n")
    lines.append(f"- Left:  Pearson r = {_fmt(rL,2)},  R^2 = {_fmt(r2L,2)},  fit: t(ms) ≈ {_fmt(aL,3)}·GFLOPs + {_fmt(bL,3)}\n")
    lines.append(f"- Right: Pearson r = {_fmt(rR,2)},  R^2 = {_fmt(r2R,2)},  fit: t(ms) ≈ {_fmt(aR,3)}·GFLOPs + {_fmt(bR,3)}\n")
    lines.append("## Ranking agreement (predicted score vs measured objective time)\n")
    lines.append(f"- Spearman ρ = {_fmt(sp,2)}\n")
    lines.append(f"- Kendall τ = {_fmt(kt,2)}\n")
    lines.append(f"- Top-{k} overlap = {overlap}/{k} = {_fmt(overlap_ratio,2)}\n")
    lines.append("## Measurement noise\n")
    lines.append(f"- Median CV(composed) = {_fmt(cv_comp_med,2)} %\n")
    lines.append("## Best splits by objective\n")
    lines.append("| boundary | cut (MiB) | #T | t_full (ms) | t_comp (ms) | sum_parts (ms) | overhead (ms) | pass |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
    for r in best:
        lines.append(
            f"| {r.get('boundary','-')} | {_fmt(r.get('cut_mib'),3)} | {r.get('n_cut_tensors','-')} | "
            f"{_fmt(r.get('full_mean_ms'),2)} | {_fmt(r.get('composed_mean_ms'),2)} | "
            f"{_fmt(r.get('sum_parts_ms'),2)} | {_fmt(r.get('overhead_ms'),2)} | "
            f"{'✅' if r.get('eps_pass') else '❌'} |\n"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def _run_case(case_dir: Path, provider: str, image: str, preset: str, warmup: int, runs: int, timeout_s: Optional[int], case_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    # Each case contains run_split_onnxruntime.py
    runner = case_dir / "run_split_onnxruntime.py"
    if not runner.exists():
        print(f"[warn] missing runner: {runner}")
        return None

    cmd = [sys.executable, str(runner), "--provider", provider, "--preset", preset, "--warmup", str(warmup), "--runs", str(runs), "--out-dir", f"results_{provider}"]
    if image and str(image).lower() != "default":
        cmd += ["--image", image]
    try:
        res = subprocess.run(cmd, cwd=str(case_dir), timeout=timeout_s)
        if res.returncode != 0:
            print(f"[warn] case failed: {case_dir.name} (rc={res.returncode})")
            return None
    except subprocess.TimeoutExpired:
        print(f"[warn] timeout: {case_dir.name} (>{timeout_s}s)")
        return None

    # report location depends on runner (it writes into results_<provider> by default)
    report = case_dir / f"results_{provider}" / "validation_report.json"
    if not report.exists():
        # some runners may use "results_auto"
        report_auto = case_dir / "results_auto" / "validation_report.json"
        report = report_auto if report_auto.exists() else report
    if not report.exists():
        print(f"[warn] missing report: {report}")
        return None

    r = _read_json(report)

    # Flatten predicted fields from newer runner versions (runner may store them under "predicted")
    pred = r.get("predicted")
    if isinstance(pred, dict):
        for k, v in pred.items():
            if k not in r:
                r[k] = v
        # common aliases used by this suite
        if "n_cut_tensors" not in r:
            if "crossing_tensors_all" in pred:
                r["n_cut_tensors"] = pred["crossing_tensors_all"]
            elif "crossing_tensors_known" in pred:
                r["n_cut_tensors"] = pred["crossing_tensors_known"]
            elif isinstance(r.get("cut_tensors"), list):
                r["n_cut_tensors"] = len(r["cut_tensors"])
        if "score" in pred and "score_pred" not in r:
            r["score_pred"] = pred["score"]
        if "imbalance" in pred and "imbalance_pred" not in r:
            r["imbalance_pred"] = pred["imbalance"]
        if "total_flops" in pred and "flops_total" not in r:
            r["flops_total"] = pred["total_flops"]
    # derive total flops if possible
    if "flops_total" not in r and ("flops_left" in r) and ("flops_right" in r):
        try:
            r["flops_total"] = float(r["flops_left"]) + float(r["flops_right"])
        except Exception:
            pass

    # enrich with static case meta if present
    manifest = case_dir / "split_manifest.json"
    if manifest.exists():
        m = _read_json(manifest)
        for k in ["boundary", "cut_mib", "n_cut_tensors", "flops_left", "flops_right", "imbalance_pred", "score_pred"]:
            if k in m and k not in r:
                r[k] = m[k]
        # keep all manifest fields (useful for later)
        for k, v in m.items():
            r.setdefault(k, v)


    # Merge benchmark_set.json metadata (predicted fields) if provided
    if isinstance(case_meta, dict):
        # merge predicted dict
        mp = case_meta.get("predicted")
        if isinstance(mp, dict):
            if not isinstance(r.get("predicted"), dict):
                r["predicted"] = mp
            else:
                for kk, vv in mp.items():
                    r["predicted"].setdefault(kk, vv)
        # surface a few helpful meta fields
        for kk in ["boundary_index", "boundary_id", "boundary_name"]:
            if kk in case_meta:
                r.setdefault(kk, case_meta.get(kk))

    # Ensure predicted fields are also available at top-level (compat)
    pred2 = r.get("predicted")
    if isinstance(pred2, dict):
        for k, v in pred2.items():
            r.setdefault(k, v)


    # Compatibility aliases for older summary/table code
    # - our predictor uses a *score* (higher is better); some scripts use score_pred (lower is better)
    # - we keep score_pred as an alias to score for reporting
    if "score_pred" not in r and "score" in r:
        r["score_pred"] = r.get("score")
    if "imbalance_pred" not in r and "imbalance" in r:
        r["imbalance_pred"] = r.get("imbalance")
    if "n_cut_tensors" not in r:
        if "cut_tensors_toggle" in r:
            r["n_cut_tensors"] = r.get("cut_tensors_toggle")
        elif "cut_tensors" in r and isinstance(r.get("cut_tensors"), (int, float)):
            r["n_cut_tensors"] = int(r.get("cut_tensors"))
        elif isinstance(r.get("cut_tensors"), (list, tuple)):
            r["n_cut_tensors"] = len(r.get("cut_tensors"))
        else:
            # fall back to predicted crossing tensors list if available
            pred_tmp = r.get("predicted")
            if isinstance(pred_tmp, dict):
                if isinstance(pred_tmp.get("crossing_tensors_all"), (list, tuple)):
                    r["n_cut_tensors"] = len(pred_tmp.get("crossing_tensors_all"))
                elif isinstance(pred_tmp.get("crossing_tensors_known"), (list, tuple)):
                    r["n_cut_tensors"] = len(pred_tmp.get("crossing_tensors_known"))
    # pass flag alias
    if "eps_pass" not in r and "ok" in r:
        r["eps_pass"] = bool(r.get("ok"))

    # Ensure timing fields are also available at top-level (compat)
    tms = r.get("timing_ms") or r.get("timing")
    if isinstance(tms, dict):
        for _name in ("full", "part1", "part2", "composed"):
            blk = tms.get(_name)
            if isinstance(blk, dict):
                mean = blk.get("mean", blk.get("mean_ms"))
                std = blk.get("std", blk.get("std_ms"))
                if mean is not None:
                    r.setdefault(f"{_name}_mean_ms", mean)
                if std is not None:
                    r.setdefault(f"{_name}_std_ms", std)

    # derived fields
    try:
        p1 = float(r.get("part1_mean_ms", float("nan")))
        p2 = float(r.get("part2_mean_ms", float("nan")))
        comp = float(r.get("composed_mean_ms", float("nan")))
        if math.isfinite(p1) and math.isfinite(p2):
            r["sum_parts_ms"] = p1 + p2
            if math.isfinite(comp):
                r["overhead_ms"] = comp - (p1 + p2)
    except Exception:
        pass
    try:
        full = float(r.get("full_mean_ms", float("nan")))
        comp = float(r.get("composed_mean_ms", float("nan")))
        if math.isfinite(full) and math.isfinite(comp) and comp > 0:
            r["speedup_full_over_composed"] = full / comp
    except Exception:
        pass

    return r


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="auto", choices=["auto", "cpu", "cuda", "tensorrt"], help="Execution provider to use.")
    parser.add_argument("--image", default="default", help="Image path or 'default'.")
    parser.add_argument("--preset", default="auto", choices=["auto", "classification", "detection"], help="Output agreement preset.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=None, help="Per-case timeout in seconds.")
    parser.add_argument("--objective", default="composed", choices=["composed", "sum_parts", "max_parts", "full"], help="Objective for ranking/summaries.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k for highlights/tables.")
    parser.add_argument("--no-csv", action="store_true", help="Do not export CSV.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    bench = _read_json(root / "benchmark_set.json")
    cases: List[Dict[str, Any]] = bench.get("cases", [])
    if not cases:
        print("[warn] benchmark_set.json has no cases")
        return 1

    provider = args.provider
    tag = f"{provider}_{args.preset}"

    rows: List[Dict[str, Any]] = []
    for i, c in enumerate(cases):
        b = c.get("boundary")
        case_dir_key = c.get("case_dir") or c.get("folder")
        if case_dir_key is None:
            case_dir_key = f"b{int(b):03d}" if b is not None else "case"
        rel = Path(str(case_dir_key))
        case_dir = (root / rel).resolve()
        print(f"\n[{i+1}/{len(cases)}] Running {case_dir.name} (provider={provider})")
        r = _run_case(case_dir, provider=provider, image=args.image, preset=args.preset, warmup=args.warmup, runs=args.runs, timeout_s=args.timeout)
        if r is not None:
            rows.append(r)

    if not rows:
        print("[warn] No results found to collect (did you run the suite?)")
        return 2

    results_json = root / f"benchmark_results_{tag}.json"
    _write_json(results_json, rows)
    print(f"Wrote {results_json.name}")

    if not args.no_csv:
        results_csv = root / f"benchmark_results_{tag}.csv"
        _write_results_csv(results_csv, rows)
        print(f"Wrote {results_csv.name}")

    # Summary (1 page)
    summary_md = root / f"benchmark_summary_{tag}.md"
    _write_summary_md(summary_md, rows, tag=tag, objective=args.objective, topk=args.topk)
    print(f"Wrote {summary_md.name}")

    # Paper-ready LaTeX table
    table_tex = root / f"benchmark_table_{tag}.tex"
    _write_table_tex(table_tex, rows, tag=tag, objective=args.objective, topk=args.topk)
    print(f"Wrote {table_tex.name}")

    # Paper-ready figures
    fig_dir = root / f"paper_figures_{tag}"
    fig_files = _save_paper_figures(fig_dir, rows, tag=tag, objective=args.objective, topk=args.topk)
    if fig_files:
        print(f"Wrote {len(fig_files)} figure files to {fig_dir.name}/")
    else:
        if not HAVE_PLOT:
            print("[info] matplotlib not available; skipping figure export")
        else:
            print("[info] no figures exported (no rows?)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
            script = script.replace('__BENCH_JSON__', bench_json_name)
            with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(script)
            try:
                os.chmod(script_path, 0o755)
            except Exception:
                pass
            return script_path

        def worker() -> None:
            try:
                cases = []
                errors = []
                made = 0

                # Try candidates in ranked order and keep adding cases until we have k successful splits.
                # We respect the GUI "Min gap" setting to avoid exporting near-duplicate boundaries.
                gap = _safe_int(self.var_min_gap.get()) or 0
                chosen: List[int] = []

                picks_iter = list(ranked_candidates)

                for bi, b0 in enumerate(picks_iter):
                    if made >= k:
                        break
                    b = int(b0)

                    if gap > 0 and any(abs(b - bb) <= gap for bb in chosen):
                        continue

                    # Update the dialog label while keeping the progress value at the
                    # number of successfully generated cases so far.
                    q.put(("prog", made, f"Splitting b{b} ({made+1}/{k})..."))
                    folder = f"b{b:0{pad}d}"
                    case_dir = os.path.join(out_dir, folder)
                    os.makedirs(case_dir, exist_ok=True)

                    try:
                        cut_tensors = asc.cut_tensors_for_boundary(order, nodes, b)
                    except Exception as e:
                        errors.append(f"b{b}: cut tensor error: {e}")
                        q.put(("prog", made, f"b{b} (skip)"))
                        continue

                    if not cut_tensors:
                        errors.append(f"b{b}: no cut tensors")
                        q.put(("prog", made, f"b{b} (skip)"))
                        continue

                    p1_path = os.path.join(case_dir, f"{base}_part1_b{b}.onnx")
                    p2_path = os.path.join(case_dir, f"{base}_part2_b{b}.onnx")
                    manifest_path = os.path.join(case_dir, "split_manifest.json")

                    # Split models
                    try:
                        p1, p2, split_manifest = asc.split_model_on_cut_tensors(
                            model,
                            cut_tensors=cut_tensors,
                            strict_boundary=strict_boundary,
                        )
                        asc.save_model(p1, p1_path)
                        asc.save_model(p2, p2_path)
                    except Exception as e:
                        errors.append(f"b{b}: split failed: {type(e).__name__}: {e}")
                        q.put(("prog", made, f"b{b} (split failed)"))
                        continue

                    # Predicted metrics (from current analysis)
                    pred = {}
                    try:
                        costs = a.get('costs_bytes') or []
                        mul = float(asc.UNIT_MULT.get('MiB', 1024.0 ** 2))
                        cut_bytes = int(costs[b]) if b < len(costs) else None
                        pred['cut_bytes'] = cut_bytes
                        pred['cut_mib'] = (float(cut_bytes) / mul) if (cut_bytes is not None and mul > 0) else None

                        pred['crossing_tensors_known'] = int((a.get('crossing_counts_known') or [])[b]) if b < len(a.get('crossing_counts_known') or []) else None
                        pred['crossing_tensors_all'] = int((a.get('crossing_counts_all') or [])[b]) if b < len(a.get('crossing_counts_all') or []) else None
                        pred['unknown_crossing_tensors'] = int((a.get('unknown_crossing_counts') or [])[b]) if b < len(a.get('unknown_crossing_counts') or []) else None

                        flp = a.get('flops_left_prefix') or []
                        total_fl = float(a.get('total_flops') or 0.0)
                        fl_left = float(flp[b]) if b < len(flp) else None
                        pred['flops_left'] = fl_left
                        pred['flops_right'] = (total_fl - fl_left) if (fl_left is not None) else None
                        pred['total_flops'] = total_fl
                        pred['imbalance'] = float((a.get('imbalance') or [])[b]) if b < len(a.get('imbalance') or []) else None

                        # Scores/latency are computed only for the selected ranking modes; we store them if present.
                        scores = a.get('scores') or {}
                        if isinstance(scores, dict) and b in scores:
                            pred['score'] = float(scores[b])
                        lat = a.get('latency_ms_dict') or {}
                        if isinstance(lat, dict) and b in lat:
                            pred['latency_ms'] = float(lat[b])

                        # System-model predictions (available regardless of ranking)
                        lt = a.get('pred_latency_total_ms') or []
                        if b < len(lt) and lt[b] is not None:
                            pred['latency_total_ms'] = float(lt[b])
                        ll = a.get('pred_latency_link_ms') or []
                        if b < len(ll) and ll[b] is not None:
                            pred['link_latency_ms'] = float(ll[b])
                        el = a.get('pred_energy_link_mJ') or []
                        if b < len(el) and el[b] is not None:
                            pred['link_energy_mJ'] = float(el[b])
                        et = a.get('pred_energy_total_mJ') or []
                        if b < len(et) and et[b] is not None:
                            pred['energy_total_mJ'] = float(et[b])


                        # Peak activation memory (approx, bytes + MiB)
                        pL = a.get('peak_act_mem_left_bytes') or []
                        pR = a.get('peak_act_mem_right_bytes') or []
                        pM = a.get('peak_act_mem_max_bytes') or []
                        if b < len(pL):
                            pred['peak_act_left_bytes'] = int(pL[b])
                            pred['peak_act_left_mib'] = float(pL[b]) / mul if mul > 0 else None
                        if b < len(pR):
                            pred['peak_act_right_bytes'] = int(pR[b])
                            pred['peak_act_right_mib'] = float(pR[b]) / mul if mul > 0 else None
                        if b < len(pM):
                            pred['peak_act_max_bytes'] = int(pM[b])
                            pred['peak_act_max_mib'] = float(pM[b]) / mul if mul > 0 else None

                        strict_ok = a.get('strict_ok') or []
                        pred['strict_ok'] = bool(strict_ok[b]) if b < len(strict_ok) else None
                    except Exception:
                        pass

                    # Per-case manifest
                    manifest_out = {
                        'tool': {'gui': __version__, 'core': getattr(asc, '__version__', '?')},
                        'boundary': int(b),
                        'boundary_index': int(b),
                        'cut_tensors': list(cut_tensors),
                        'strict_boundary': bool(strict_boundary),
                        'predicted': pred,
                        'full_model': os.path.abspath(self.model_path),
                        'part1': os.path.abspath(p1_path),
                        'part2': os.path.abspath(p2_path),
                        'created_at': datetime.now().isoformat(timespec='seconds'),
                    }
                    if isinstance(split_manifest, dict):
                        manifest_out.update(split_manifest)

                    # Context diagrams
                    if do_ctx_full:
                        try:
                            ctx = asc.export_boundary_graphviz_context(
                                model,
                                order,
                                b,
                                cut_tensors,
                                case_dir,
                                basename=f"split_context_b{b}",
                                render=True,
                                hops=int(ctx_hops),
                            )
                            manifest_out['split_context'] = ctx
                        except Exception as e:
                            manifest_out['split_context_error'] = str(e)

                    if do_ctx_cutflow:
                        try:
                            ctx_cf = asc.export_boundary_graphviz_context(
                                model,
                                order,
                                b,
                                cut_tensors,
                                case_dir,
                                basename=f"split_context_b{b}_cutflow",
                                render=True,
                                hops=int(ctx_hops),
                                cut_flow_only=True,
                                include_internal_consumers=False,
                                include_external_inputs=False,
                            )
                            manifest_out['split_context_cutflow'] = ctx_cf
                        except Exception as e:
                            manifest_out['split_context_cutflow_error'] = str(e)

                    # Runner skeleton (required)
                    try:
                        runner_path = asc.write_runner_skeleton_onnxruntime(
                            case_dir,
                            manifest_filename=os.path.basename(manifest_path),
                            target=runner_target,
                        )
                        manifest_out['runner'] = os.path.basename(runner_path)
                    except Exception as e:
                        errors.append(f"b{b}: runner skeleton failed: {e}")

                    with open(manifest_path, 'w', encoding='utf-8') as f:
                        json.dump(manifest_out, f, indent=2)

                    cases.append({
                        'boundary': int(b),
                        'case_dir': folder,
                        'folder': folder,
                        # Stored relative to the per-case folder; the suite resolves it under case_dir.
                        'manifest': os.path.basename(manifest_path),
                        'predicted': pred,
                    })

                    chosen.append(int(b))
                    made += 1
                    q.put(("prog", made, f"b{b}"))

                # Write benchmark_set.json
                bench = {
                    'tool': {'gui': __version__, 'core': getattr(asc, '__version__', '?')},
                    'model': os.path.abspath(self.model_path),
                    'model_name': base,
                    'created_at': datetime.now().isoformat(timespec='seconds'),
                    'analysis_params': {
                        'ranking': str(getattr(params, 'ranking', 'score')),
                        'topk': int(getattr(params, 'topk', len(self.current_picks))),
                        'min_gap': int(getattr(params, 'min_gap', 0)),
                        'exclude_trivial': bool(getattr(params, 'exclude_trivial', False)),
                        'only_single_tensor': bool(getattr(params, 'only_single_tensor', False)),
                        'strict_boundary': bool(getattr(params, 'strict_boundary', False)),

                        # Skip-/Block pruning
                        'prune_skip_block': bool(getattr(params, 'prune_skip_block', False)),
                        'skip_min_span': int(getattr(params, 'skip_min_span', 0)),
                        'skip_allow_last_n': int(getattr(params, 'skip_allow_last_n', 0)),

                        # System model (compute + link)
                        'link_model': str(getattr(params, 'link_model', 'ideal')),
                        'bandwidth_value': getattr(params, 'bw_value', None),
                        'bandwidth_unit': str(getattr(params, 'bw_unit', 'MB/s')),
                        'gops_left': getattr(params, 'gops_left', None),
                        'gops_right': getattr(params, 'gops_right', None),
                        'link_overhead_ms': getattr(params, 'overhead_ms', 0.0),
                        'link_energy_pj_per_byte': getattr(params, 'link_energy_pj_per_byte', None),
                        'link_mtu_payload_bytes': getattr(params, 'link_mtu_payload_bytes', None),
                        'link_per_packet_overhead_ms': getattr(params, 'link_per_packet_overhead_ms', None),
                        'link_per_packet_overhead_bytes': getattr(params, 'link_per_packet_overhead_bytes', None),
                        'energy_pj_per_flop_left': getattr(params, 'energy_pj_per_flop_left', None),
                        'energy_pj_per_flop_right': getattr(params, 'energy_pj_per_flop_right', None),

                        # Link constraints
                        'link_max_latency_ms': getattr(params, 'link_max_latency_ms', None),
                        'link_max_energy_mJ': getattr(params, 'link_max_energy_mJ', None),
                        'link_max_bytes': getattr(params, 'link_max_bytes', None),

                        # Activation-memory constraints (peak, approx)
                        'max_peak_act_left': getattr(params, 'max_peak_act_left', None),
                        'max_peak_act_left_unit': str(getattr(params, 'max_peak_act_left_unit', 'MiB')),
                        'max_peak_act_right': getattr(params, 'max_peak_act_right', None),
                        'max_peak_act_right_unit': str(getattr(params, 'max_peak_act_right_unit', 'MiB')),

                        'batch_override': batch_override,
                        'eps_default': float(eps_default),
                    },
                    'system_spec': asdict(self._build_system_spec(params)) if params else None,
                    'cases': cases,
                    'errors': errors,
                }
                bench_path = os.path.join(out_dir, 'benchmark_set.json')
                with open(bench_path, 'w', encoding='utf-8') as f:
                    json.dump(bench, f, indent=2)

                # Write harness script
                harness_path = _write_benchmark_suite_script(out_dir, 'benchmark_set.json')

                # Small README
                readme = os.path.join(out_dir, 'README_BENCHMARK.txt')
                with open(readme, 'w', encoding='utf-8') as f:
                    f.write(
                        "Benchmark suite generated by the ONNX Split-Point Analyser.\n\n"
                        f"Model: {os.path.abspath(self.model_path)}\n"
                        f"Cases: {len(cases)} (requested: {k})\n\n"
                        "Next steps:\n"
                        "  1) (optional) install deps: pip install onnx onnxruntime numpy pillow matplotlib\n"
                        "  2) run: python benchmark_suite.py --provider cpu\n"
                        "  3) or : python benchmark_suite.py --provider cuda\n"
                        "  4) to also generate human-readable outputs: add --image default --preset auto\n\n"
                        "Outputs:\n"
                        "  - benchmark_results_<provider>.csv / .json\n"
                        "  - benchmark_plots_<provider>.pdf (if matplotlib installed)\n"
                    )

                msg = []
                msg.append(f"Generated benchmark set: {out_dir}")
                msg.append(f"Cases: {len(cases)} (requested {k})")
                msg.append(f"Harness: {harness_path}")
                if errors:
                    msg.append("\nWarnings:")
                    msg.extend([f"  - {e}" for e in errors[:10]])
                    if len(errors) > 10:
                        msg.append(f"  ... and {len(errors)-10} more")
                q.put(("ok", "\n".join(msg)))
            except Exception as e:
                q.put(("err", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()

        def poll() -> None:
            try:
                item = q.get_nowait()
            except queue.Empty:
                self.after(100, poll)
                return

            if not item:
                self.after(100, poll)
                return

            status = str(item[0])
            if status == 'prog':
                try:
                    made = int(item[1])
                    what = str(item[2]) if len(item) > 2 else ''
                    pb['value'] = made
                    lbl2.configure(text=f"{made}/{k}: {what}")
                    self.update_idletasks()
                except Exception:
                    pass
                self.after(50, poll)
                return

            # non-progress informational message (do not close the dialog)
            if status in ('msg', 'note'):
                try:
                    what = str(item[1]) if len(item) > 1 else ''
                    lbl2.configure(text=what)
                    self.update_idletasks()
                except Exception:
                    pass
                self.after(50, poll)
                return

            # final
            try:
                dlg.destroy()
            except Exception:
                pass
            try:
                self.configure(cursor="")
            except Exception:
                pass

            if status == 'ok':
                messagebox.showinfo("Benchmark set created", str(item[1]))
            else:
                messagebox.showerror("Benchmark set failed", str(item[1]))

        poll()

    def _export_overview(self, fmt: str):
        if fmt not in {"svg", "pdf"}:
            return
        if self.analysis is None:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return

        ext = "." + fmt
        path = filedialog.asksaveasfilename(
            title=f"Export overview as {fmt.upper()}",
            defaultextension=ext,
            filetypes=[(fmt.upper(), f"*{ext}"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            self.fig.savefig(path, format=fmt)
            messagebox.showinfo("Export complete", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"{type(e).__name__}: {e}")

    def _export_single(self, fmt: str):
        """Export the 4 single plots (Comm, Compute, Pareto, Latency) into a chosen folder."""
        if fmt not in {"svg", "pdf"}:
            return
        if self.analysis is None:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return

        out_dir = filedialog.askdirectory(title=f"Choose folder for {fmt.upper()} exports")
        if not out_dir:
            return

        try:
            self._export_paper_assets(
                Path(out_dir),
                self.analysis,
                self.current_picks or [],
                formats=(fmt,),
                include_overview=False,
                include_table=False,
                include_json=False,
            )
            messagebox.showinfo("Export complete", f"Saved {fmt.upper()} plots to: {out_dir}")
        except Exception as e:
            messagebox.showerror("Export failed", f"{type(e).__name__}: {e}")

    def _export_paper_assets(
        self,
        out_dir: Path,
        a: Dict[str, Any],
        picks: List[int],
        *,
        formats=("pdf", "svg"),
        include_overview: bool = True,
        include_table: bool = True,
        include_json: bool = True,
    ) -> None:
        """Write paper-ready assets (plots, tables, and metadata) into a folder.

        This helper is used for:
        - benchmark set creation (always exports both PDF+SVG + TeX table + metadata)
        - optional manual exports
        """

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pull arrays from analysis dict.
        costs = list(a.get("costs_bytes") or [])
        flops_left_prefix = list(a.get("flops_left_prefix") or [])
        imbalance = list(a.get("imbalance") or [])
        total_flops = float(a.get("total_flops") or 0.0)

        M = min(len(costs), len(flops_left_prefix), len(imbalance))
        if M <= 0:
            return

        costs = costs[:M]
        flops_left_prefix = flops_left_prefix[:M]
        imbalance = imbalance[:M]

        xs = list(range(M))
        comm_mb = [float(cb) / 1e6 for cb in costs]
        fl_l_g = [float(f) / 1e9 for f in flops_left_prefix]
        fl_r_g = [float(total_flops - flops_left_prefix[i]) / 1e9 for i in range(M)] if total_flops > 0 else [0.0] * M

        # Peak activation memory arrays (MiB)
        peak_left_b = list(a.get("peak_act_mem_left_bytes") or [])[:M]
        peak_right_b = list(a.get("peak_act_mem_right_bytes") or [])[:M]
        peak_max_b = list(a.get("peak_act_mem_max_bytes") or [])[:M]
        peak_left_mib = [float(x) / (1024.0**2) for x in peak_left_b] if peak_left_b else []
        peak_right_mib = [float(x) / (1024.0**2) for x in peak_right_b] if peak_right_b else []
        peak_max_mib = [float(x) / (1024.0**2) for x in peak_max_b] if peak_max_b else []

        # Normalise picks.
        pp: List[int] = []
        for b in picks or []:
            try:
                bi = int(b)
            except Exception:
                continue
            if 0 <= bi < M:
                pp.append(bi)
        picks = pp
        pick_set = set(picks)

        # Read GUI params (system model).
        try:
            p = self._read_params()
        except Exception:
            p = None

        def plot_comm(ax):
            bars = ax.bar(xs, comm_mb)
            ax.set_title("Activation bytes crossing each boundary")
            ax.set_xlabel("Boundary index")
            ax.set_ylabel("Crossing size (MB)")
            for i, bar in enumerate(bars):
                if i in pick_set:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.5)
                    bar.set_hatch("//")

        def plot_comp(ax):
            ax.plot(xs, fl_l_g, label="Compute left (GFLOPs)")
            ax.plot(xs, fl_r_g, label="Compute right (GFLOPs)")
            ax.set_title("Cumulative compute around boundaries")
            ax.set_xlabel("Boundary index")
            ax.set_ylabel("GFLOPs")
            ax.legend(loc="best")
            for b in picks:
                ax.axvline(b, linestyle="--", linewidth=1)

        def plot_pareto(ax):
            cand_bounds = a.get("candidate_bounds") or list(range(M))
            cb: List[int] = []
            for b in cand_bounds:
                try:
                    bi = int(b)
                except Exception:
                    continue
                if 0 <= bi < M:
                    cb.append(bi)
            cb = sorted(set(cb))

            pts = [(float(costs[b]) / 1e6, float(imbalance[b])) for b in cb] if cb else []
            ax.scatter([pt[0] for pt in pts], [pt[1] for pt in pts], s=12)
            ax.set_title("Pareto: communication vs imbalance")
            ax.set_xlabel("Cut (MB)")
            ax.set_ylabel("Imbalance |L-R|/Total")

            if p is not None and bool(getattr(p, "show_pareto_front", False)) and pts:
                fi = pareto_front(pts)
                fi = sorted(fi, key=lambda i: pts[i][0])
                ax.plot([pts[i][0] for i in fi], [pts[i][1] for i in fi], linewidth=2)

            if picks:
                ax.scatter(
                    [float(costs[b]) / 1e6 for b in picks],
                    [float(imbalance[b]) for b in picks],
                    s=40,
                    marker="o",
                    edgecolors="black",
                )

        def _latency_placeholder(ax):
            ax.set_title("Latency model")
            ax.set_xlabel("Boundary index")
            ax.set_ylabel("ms")
            ax.text(0.05, 0.6, "Provide bandwidth + GOPS L/R\n(optionally link energy)", transform=ax.transAxes)

        def plot_latency(ax):
            if p is None or total_flops <= 0:
                _latency_placeholder(ax)
                return

            try:
                sys = self._build_system_spec(p)
            except Exception:
                sys = None

            if sys is None:
                _latency_placeholder(ax)
                return

            lat_total = []
            lat_link = []
            for b in range(M):
                m = sys.estimate_boundary(
                    comm_bytes=float(costs[b]),
                    flops_left=float(flops_left_prefix[b]),
                    flops_total=float(total_flops),
                )
                lat_total.append(m.get("latency_total_ms"))
                lat_link.append(m.get("latency_link_ms"))

            if all(v is None for v in lat_total):
                _latency_placeholder(ax)
                return

            lt = [float(v) if v is not None else float("nan") for v in lat_total]
            lk = [float(v) if v is not None else float("nan") for v in lat_link]

            ax.plot(xs, lt, label="Total latency (ms)")
            ax.plot(xs, lk, label="Link-only (ms)")
            ax.set_title("Latency model vs boundary")
            ax.set_xlabel("Boundary index")
            ax.set_ylabel("ms")
            ax.legend(loc="best")
            for b in picks:
                ax.axvline(b, linestyle="--", linewidth=1)

        def plot_peak_mem(ax):
            if not peak_left_mib and not peak_right_mib:
                ax.set_title('Peak activation memory')
                ax.set_xlabel('Boundary index')
                ax.set_ylabel('MiB')
                ax.text(0.05, 0.6, 'No memory data available', transform=ax.transAxes)
                return

            ax.plot(xs, peak_left_mib, label='Peak act mem left (MiB)')
            ax.plot(xs, peak_right_mib, label='Peak act mem right (MiB)')
            if peak_max_mib:
                ax.plot(xs, peak_max_mib, label='Peak act mem max (MiB)')

            # Optional constraint lines (from current GUI params)
            if p is not None:
                try:
                    sys2 = self._build_system_spec(p)
                except Exception:
                    sys2 = None
                if sys2 is not None:
                    ml = getattr(getattr(sys2, 'memory', None), 'max_peak_act_left_bytes', None)
                    mr = getattr(getattr(sys2, 'memory', None), 'max_peak_act_right_bytes', None)
                    if ml is not None:
                        ax.axhline(float(ml) / (1024.0**2), linestyle='--', linewidth=1, label='Limit left')
                    if mr is not None:
                        ax.axhline(float(mr) / (1024.0**2), linestyle='--', linewidth=1, label='Limit right')

            ax.set_title('Peak activation memory vs boundary (approx)')
            ax.set_xlabel('Boundary index')
            ax.set_ylabel('MiB')
            ax.legend(loc='best')
            for b in picks:
                ax.axvline(b, linestyle='--', linewidth=1)

        def save_one(fname_stem: str, plot_fn, fmt: str, *, figsize=(6, 4)) -> None:
            fig = Figure(figsize=figsize, constrained_layout=True)
            ax = fig.add_subplot(1, 1, 1)
            plot_fn(ax)
            out_path = out_dir / f"{fname_stem}.{fmt}"
            fig.savefig(str(out_path), format=fmt, bbox_inches="tight")

        def save_overview(fmt: str) -> None:
            fig = Figure(figsize=(10, 6), constrained_layout=True)
            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)
            ax3 = fig.add_subplot(2, 2, 3)
            ax4 = fig.add_subplot(2, 2, 4)
            plot_comm(ax1)
            plot_comp(ax2)
            plot_pareto(ax3)
            plot_latency(ax4)
            fig.savefig(str(out_dir / f"analysis_plots_overview.{fmt}"), format=fmt, bbox_inches="tight")

        for fmt in formats:
            if fmt not in {"pdf", "svg"}:
                continue
            if include_overview:
                try:
                    save_overview(fmt)
                except Exception as e:
                    print(f"[warn] Failed to export overview {fmt.upper()}: {e}")

            try:
                save_one("analysis_activation_bytes", plot_comm, fmt)
                save_one("analysis_cumulative_compute", plot_comp, fmt)
                save_one("analysis_pareto_comm_imbalance", plot_pareto, fmt)
                save_one("analysis_latency_model", plot_latency, fmt)
                save_one("analysis_peak_activation_memory", plot_peak_mem, fmt)
            except Exception as e:
                print(f"[warn] Failed to export single plots ({fmt.upper()}): {e}")

        if include_table:
            try:
                tex = self._make_tex_table(a, picks)
                (out_dir / "split_candidates.tex").write_text(tex, encoding="utf-8")
            except Exception as e:
                print(f"[warn] Failed to export TeX candidate table: {e}")

        if include_json:
            # System / workload separation + Pareto export (dissertation-friendly)
            try:
                if p is not None:
                    sys = self._build_system_spec(p)
                    (out_dir / "system_config.json").write_text(json.dumps(asdict(sys), indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[warn] Failed to export system_config.json: {e}")

            try:
                workload = {
                    "tool": {"gui": __version__, "core": getattr(asc, "__version__", "?")},
                    "model": os.path.abspath(self.model_path) if self.model_path else None,
                    "model_name": os.path.splitext(os.path.basename(self.model_path or "model"))[0],
                    "n_boundaries": int(M),
                    "total_flops": float(total_flops),
                    "comm_bytes": [int(x) for x in costs],
                    "imbalance": [float(x) for x in imbalance],
                    "flops_left_prefix": [float(x) for x in flops_left_prefix],
                    "crossing_tensors_all": [int(x) for x in (a.get("crossing_counts_all") or [])[:M]],
                    "crossing_tensors_known": [int(x) for x in (a.get("crossing_counts_known") or [])[:M]],
                    "unknown_crossing_tensors": [int(x) for x in (a.get("unknown_crossing_counts") or [])[:M]],
                    "peak_act_mem_left_bytes": [int(x) for x in (a.get("peak_act_mem_left_bytes") or [])[:M]],
                    "peak_act_mem_right_bytes": [int(x) for x in (a.get("peak_act_mem_right_bytes") or [])[:M]],
                    "peak_act_mem_max_bytes": [int(x) for x in (a.get("peak_act_mem_max_bytes") or [])[:M]],
                }
                (out_dir / "workload_profile.json").write_text(json.dumps(workload, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[warn] Failed to export workload_profile.json: {e}")

            try:
                pruning = {
                    "skip_blocks": a.get("skip_blocks"),
                    "skip_block_forbidden": a.get("skip_block_forbidden"),
                    "candidate_prune_skip_block": a.get("candidate_prune_skip_block"),
                    "candidate_prune_link_constraints": a.get("candidate_prune_link_constraints"),
                    "candidate_prune_memory_constraints": a.get("candidate_prune_memory_constraints"),
                }
                (out_dir / "candidate_pruning.json").write_text(json.dumps(pruning, indent=2), encoding="utf-8")
            except Exception as e:
                print(f"[warn] Failed to export candidate_pruning.json: {e}")

            try:
                cand_bounds = a.get("candidate_bounds") or list(range(M))
                cand_list: List[int] = []
                for b in cand_bounds:
                    try:
                        bi = int(b)
                    except Exception:
                        continue
                    if 0 <= bi < M:
                        cand_list.append(bi)
                cand_list = sorted(set(cand_list))

                pts = [(float(costs[b]) / 1e6, float(imbalance[b])) for b in cand_list]
                front_idx = pareto_front(pts) if pts else []
                pareto_set = {cand_list[i] for i in front_idx}

                sys = None
                if p is not None:
                    try:
                        sys = self._build_system_spec(p)
                    except Exception:
                        sys = None

                out_csv = out_dir / "pareto_export.csv"
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "boundary",
                        "is_candidate",
                        "is_pick",
                        "is_pareto_comm_imb",
                        "comm_bytes",
                        "comm_MiB",
                        "crossing_tensors_all",
                        "imbalance",
                        "flops_left",
                        "flops_right",
                        "latency_total_ms",
                        "latency_link_ms",
                        "energy_total_mJ",
                        "energy_link_mJ",
                        "peak_act_left_bytes",
                        "peak_act_left_MiB",
                        "peak_act_right_bytes",
                        "peak_act_right_MiB",
                        "peak_act_max_bytes",
                        "peak_act_max_MiB",
                        "mem_feasible",
                        "link_feasible",
                    ])
                    cand_set = set(cand_list)
                    for b in range(M):
                        is_cand = b in cand_set
                        is_pick = b in pick_set
                        is_pareto = b in pareto_set

                        lat_total = lat_link = e_total = e_link = None
                        link_feasible = None
                        if sys is not None:
                            m = sys.estimate_boundary(
                                comm_bytes=float(costs[b]),
                                flops_left=float(flops_left_prefix[b]),
                                flops_total=float(total_flops),
                            )
                            lat_total = m.get("latency_total_ms")
                            lat_link = m.get("latency_link_ms")
                            e_total = m.get("energy_total_mJ")
                            e_link = m.get("energy_link_mJ")
                            try:
                                link_feasible = bool(sys.link.is_feasible(float(costs[b])))
                            except Exception:
                                link_feasible = None

                        cross_all = (a.get("crossing_counts_all") or [])
                        cross_all_v = "" if b >= len(cross_all) else int(cross_all[b])

                        w.writerow([
                            int(b),
                            int(1 if is_cand else 0),
                            int(1 if is_pick else 0),
                            int(1 if is_pareto else 0),
                            int(costs[b]),
                            float(costs[b]) / (1024.0**2),
                            cross_all_v,
                            float(imbalance[b]),
                            float(flops_left_prefix[b]),
                            float(total_flops - flops_left_prefix[b]),
                            "" if lat_total is None else float(lat_total),
                            "" if lat_link is None else float(lat_link),
                            "" if e_total is None else float(e_total),
                            "" if e_link is None else float(e_link),
                            int((a.get("peak_act_mem_left_bytes") or [])[b]) if b < len(a.get("peak_act_mem_left_bytes") or []) else "",
                            float((a.get("peak_act_mem_left_bytes") or [])[b]) / (1024.0**2) if b < len(a.get("peak_act_mem_left_bytes") or []) else "",
                            int((a.get("peak_act_mem_right_bytes") or [])[b]) if b < len(a.get("peak_act_mem_right_bytes") or []) else "",
                            float((a.get("peak_act_mem_right_bytes") or [])[b]) / (1024.0**2) if b < len(a.get("peak_act_mem_right_bytes") or []) else "",
                            int((a.get("peak_act_mem_max_bytes") or [])[b]) if b < len(a.get("peak_act_mem_max_bytes") or []) else "",
                            float((a.get("peak_act_mem_max_bytes") or [])[b]) / (1024.0**2) if b < len(a.get("peak_act_mem_max_bytes") or []) else "",
                            "" if sys is None else int(1 if sys.is_memory_feasible(peak_left_bytes=float((a.get("peak_act_mem_left_bytes") or [])[b]) if b < len(a.get("peak_act_mem_left_bytes") or []) else float('inf'), peak_right_bytes=float((a.get("peak_act_mem_right_bytes") or [])[b]) if b < len(a.get("peak_act_mem_right_bytes") or []) else float('inf')) else 0),
                            "" if link_feasible is None else int(1 if link_feasible else 0),
                        ])
            except Exception as e:
                print(f"[warn] Failed to export pareto_export.csv: {e}")

    def _export_benchmark_paper_assets(self, out_dir: Path, a: Dict[str, Any], picks: List[int]) -> None:
        """Export paper-ready assets into a benchmark-set folder (PDF+SVG, TeX, and metadata)."""
        self._export_paper_assets(
            Path(out_dir),
            a,
            picks,
            formats=("pdf", "svg"),
            include_overview=True,
            include_table=True,
            include_json=True,
        )

    def _export_tex_table(self):
        if self.analysis is None or not self.current_picks:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return

        default_name = "split_candidates.tex"
        if self.model_path:
            base = os.path.splitext(os.path.basename(self.model_path))[0]
            default_name = f"split_candidates_{_label_sanitize(base)}.tex"

        path = filedialog.asksaveasfilename(
            title="Export LaTeX table",
            defaultextension=".tex",
            initialfile=default_name,
            filetypes=[("LaTeX", "*.tex"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            tex = self._make_tex_table(self.analysis, self.current_picks)
            with open(path, "w", encoding="utf-8") as f:
                f.write(tex)
            messagebox.showinfo("Export complete", f"Saved: {path}")
        except Exception as e:
            messagebox.showerror("Export failed", f"{type(e).__name__}: {e}")

    def _make_tex_table(self, a: Dict, picks: List[int]) -> str:
        model_name = os.path.splitext(os.path.basename(self.model_path or "model"))[0]
        model_name_tex = _latex_escape(model_name)
        label = f"tab:split_candidates_{_label_sanitize(model_name)}"

        costs = a["costs_bytes"]
        counts_all = a["crossing_counts_all"]
        unknown = a["unknown_crossing_counts"]
        flops_left_prefix = a["flops_left_prefix"]
        total_flops = float(a["total_flops"])

        def comm_mib(b: int) -> float:
            return float(costs[b]) / (1024.0**2)

        def gflops_left(b: int) -> float:
            return float(flops_left_prefix[b]) / 1e9

        def gflops_right(b: int) -> float:
            return float(total_flops - flops_left_prefix[b]) / 1e9

        any_unknown = any(int(unknown[b]) > 0 for b in picks if b < len(unknown))

        lines: List[str] = []
        lines.append("% Generated by analyse_and_split_gui.py")
        lines.append("% Requires: \\usepackage{booktabs}")
        lines.append("\\begin{table}[t]")
        lines.append("  \\centering")
        lines.append("  \\small")
        lines.append("  \\setlength{\\tabcolsep}{4pt}")
        lines.append("  \\begin{tabular}{@{}r r r r r@{}}")
        lines.append("    \\toprule")
        # NOTE: we need a literal LaTeX line break "\\" at the end of the header/rows.
        # In Python string literals that means writing "\\\\".
        lines.append("    Boundary & Comm (MiB) & \\#Tensors & $F_L$ (GFLOP) & $F_R$ (GFLOP) \\\\")
        lines.append("    \\midrule")

        for b in picks:
            lines.append(
                f"    {int(b)} & {comm_mib(b):.3f} & {int(counts_all[b])} & {gflops_left(b):.3f} & {gflops_right(b):.3f} \\\\")

        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        caption = f"Example split candidates for the {model_name_tex} graph (top-$k$, illustrative)."
        lines.append(f"  \\caption{{{caption}}}")
        lines.append(f"  \\label{{{label}}}")
        lines.append("\\end{table}")

        if any_unknown:
            lines.append("")
            lines.append("% Note: Some crossing tensors had unknown sizes; communication may be a lower bound.")

        return "\n".join(lines) + "\n"

    # ----------------------------- Utilities -----------------------------

    def _clear_results(self):
        self.analysis = None
        self.current_picks = []
        try:
            self.btn_export_tex.state(["disabled"])
            self.btn_split.state(["disabled"])
        except Exception:
            pass
        self.tree.delete(*self.tree.get_children())
        for ax in (self.ax_comm, self.ax_comp, self.ax_pareto, self.ax_lat):
            ax.clear()
        self.canvas.draw_idle()
        self.var_shape_coverage.set("(run analysis)")
        self.var_unknown_crossing.set("(run analysis)")
        self.var_diag_note.set("")


def main():
    # Debug: show which files are being executed/imported (helps when multiple versions exist)
    try:
        print(f"[GUI] {os.path.abspath(__file__)} (v{__version__})")
        print(f"[CORE] {os.path.abspath(getattr(asc, '__file__', ''))} (v{getattr(asc, '__version__', '?')})")
    except Exception:
        pass
    app = SplitPointAnalyserGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
