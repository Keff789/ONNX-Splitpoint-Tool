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
import hashlib
import os
import shutil
import re
import tempfile
import logging
import sys
import threading
import subprocess
import shlex
import queue
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .workdir import ensure_workdir

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
from .accelerator_specs import load_accelerator_specs
from .gui.events import GuiEvents
from .gui.panels import panel_candidates as cand_panel
from .gui.analysis_params import iter_specs
from .gui.state import AppUiState, AnalysisResult, GuiState, SelectedCandidate
from .core_params import Params, gui_state_to_params_dict
from .memory_utils import estimate_ram_bytes, kv_cache_bytes_per_layer, kv_for_boundary, layer_split_index_for_boundary, precompute_initializer_spans, weights_for_all_boundaries

from . import __version__ as TOOL_VERSION

__version__ = TOOL_VERSION

# Module logger (used by worker threads as well).
logger = logging.getLogger(__name__)


# ---------------------------- Semantic clustering helpers ----------------------------

# Many transformer/LLM ONNX exports use paths like:
#   /model/layers.20/attn/...   or   /model/blocks.3/...
# and KV-cache values like:
#   present.20.key, present.20.value
#
# We use these patterns to provide a "semantic layers" clustering mode that
# selects the best boundary per layer-transition (rather than per fixed op window).
_SEM_LAYER_RE = re.compile(r"(?:^|/)(layers|blocks)[\._](\d+)(?:/|$)")
_SEM_PRESENT_RE = re.compile(r"(?:^|/)?present\.(\d+)\.")


def _semantic_group_for_node(node: onnx.NodeProto) -> Optional[str]:
    """Best-effort semantic group label for a node.

    Returns values like:
      - "layers.20" / "blocks.3" when a layer index is detectable
      - "embed", "lm_head", "final_norm" for common non-layer regions
      - None if no reasonable semantic tag can be inferred

    This is intentionally heuristic and designed to be:
      - stable across minor graph changes
      - good enough to cluster candidates for LLMs
      - safe to ignore for CNNs (falls back to uniform clustering)
    """

    # Prefer node.name, but also look at value names (inputs/outputs) because
    # many exporters embed the semantic path there instead.
    cand: List[str] = []
    if getattr(node, "name", ""):
        cand.append(str(node.name))
    cand.extend([str(x) for x in getattr(node, "output", []) if x])
    cand.extend([str(x) for x in getattr(node, "input", []) if x])

    for s in cand:
        m = _SEM_LAYER_RE.search(s)
        if m:
            try:
                return f"{m.group(1)}.{int(m.group(2))}"
            except Exception:
                return f"{m.group(1)}.{m.group(2)}"

    # KV-cache naming (present.<layer>.key/value)
    for s in cand:
        m = _SEM_PRESENT_RE.search(s)
        if m:
            try:
                return f"layers.{int(m.group(1))}"
            except Exception:
                return f"layers.{m.group(1)}"

    # Coarse heuristics for common top/bottom transformer regions.
    joined = " ".join(cand).lower()
    if "embed" in joined or "embedding" in joined:
        return "embed"
    if "lm_head" in joined or "logits" in joined:
        return "lm_head"
    if "final" in joined and "norm" in joined:
        return "final_norm"

    return None




def _semantic_labels_for_boundaries(nodes: List[onnx.NodeProto], order: List[int]) -> List[str]:
    """Return a semantic label for each boundary index.

    The label is a best-effort, human-readable tag that is stable across small
    graph changes, intended primarily for transformer/LLM graphs.

    Examples:
      - "layers.19->layers.20" for a clean inter-layer boundary
      - "layers.20 (in-layer)" for a cut inside a layer/block

    For CNN-like graphs where no semantic tags can be inferred, labels will be
    empty strings.
    """

    n_pos = int(len(order))
    if n_pos <= 1:
        return []

    pos_group: List[Optional[str]] = []
    pos_is_const: List[bool] = []
    for node_idx in order:
        n = nodes[int(node_idx)]
        pos_group.append(_semantic_group_for_node(n))
        pos_is_const.append(str(getattr(n, 'op_type', '')) == 'Constant')

    # Nearest non-constant semantic tag to the left
    prev_group: List[Optional[str]] = [None] * n_pos
    last: Optional[str] = None
    for i in range(n_pos):
        g = pos_group[i]
        if (not pos_is_const[i]) and g:
            last = g
        prev_group[i] = last

    # Nearest non-constant semantic tag to the right
    next_group: List[Optional[str]] = [None] * n_pos
    last = None
    for i in range(n_pos - 1, -1, -1):
        g = pos_group[i]
        if (not pos_is_const[i]) and g:
            last = g
        next_group[i] = last

    M = n_pos - 1
    labels: List[str] = [''] * M
    for b in range(M):
        lg = prev_group[b] if 0 <= b < n_pos else None
        rg = next_group[b + 1] if 0 <= (b + 1) < n_pos else None
        if lg and rg:
            if lg == rg:
                labels[b] = f"{lg} (in-layer)"
            else:
                labels[b] = f"{lg}->{rg}"
        elif lg:
            labels[b] = str(lg)
        elif rg:
            labels[b] = str(rg)
        else:
            labels[b] = ''
    return labels


def _setup_gui_logging() -> Optional[str]:
    """Configure logging to a persistent file.

    The GUI is often started without a visible console. A log file helps debug
    crashes or backend issues.
    """

    try:
        log_dir = Path.home() / ".onnx_splitpoint_tool"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "gui.log"

        # Don't clobber an existing logging configuration (e.g. when embedded).
        root = logging.getLogger()
        if not root.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[
                    logging.FileHandler(str(log_path), mode="a", encoding="utf-8"),
                    logging.StreamHandler(sys.stdout),
                ],
            )

        # Convenience: also write a `./gui.log` in the current working directory
        # (best-effort). This helps when users expect a local log file next to the
        # repo or the launcher script.
        try:
            cwd_log_path = Path.cwd() / "gui.log"
            if str(cwd_log_path) != str(log_path):
                fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
                cwd_fh = logging.FileHandler(str(cwd_log_path), mode="a", encoding="utf-8")
                cwd_fh.setFormatter(fmt)
                logging.getLogger().addHandler(cwd_fh)
        except Exception:
            pass

        # Hook unhandled exceptions so we get a traceback in the log file.
        def _excepthook(exc_type, exc, tb):
            logging.error("Unhandled exception", exc_info=(exc_type, exc, tb))
            try:
                sys.__excepthook__(exc_type, exc, tb)
            except Exception:
                pass

        sys.excepthook = _excepthook

        if hasattr(threading, "excepthook"):
            def _thread_excepthook(args):
                logging.error("Unhandled thread exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
            threading.excepthook = _thread_excepthook  # type: ignore[attr-defined]

        logging.info("GUI started (v%s)", __version__)
        return str(log_path)
    except Exception:
        return None


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

        self.gui_state = GuiState()
        self.events = GuiEvents()
        self.model_path: Optional[str] = None
        self.analysis: Optional[Dict] = None
        self.current_picks: List[int] = []
        self.analysis_result: Optional[AnalysisResult] = None
        self.selected_candidate: Optional[SelectedCandidate] = None
        self._last_params: Optional[Params] = None
        self.ui_state: AppUiState = AppUiState.NO_MODEL

        # Hailo feasibility-check result cache (persists across runs).
        # Keyed by (backend|hw_arch|fixup|sha1(model_bytes)).
        self._hailo_cache_path: Path = self._default_hailo_cache_path()
        self._hailo_cache: Dict[str, Dict[str, Any]] = {}
        self._hailo_cache_dirty: bool = False
        self._load_hailo_cache()

        self.accel_specs = load_accelerator_specs()
        self.memory_by_boundary: Dict[int, Dict[str, Any]] = {}
        self._candidate_rows: List[Dict[str, Any]] = []
        self._cand_by_iid: Dict[str, Dict[str, Any]] = {}
        self._tree_clean_tooltips: Dict[str, str] = {}
        self._clean_tooltip_tip: Optional[tk.Toplevel] = None
        self._clean_tooltip_row: Optional[str] = None

        self._register_event_handlers()

        # Remembered default folder for split/export dialogs.
        self.default_output_dir: Optional[str] = None

        self._build_ui()
        self._set_ui_state(AppUiState.NO_MODEL)

    def _infer_ui_state(self) -> AppUiState:
        if self.gui_state.current_model_path is None:
            return AppUiState.NO_MODEL
        if "disabled" in self.btn_analyse.state():
            return AppUiState.ANALYSING
        if self.analysis is None:
            return AppUiState.MODEL_LOADED
        if self._selected_boundary_index() is not None:
            return AppUiState.SPLIT_READY
        return AppUiState.ANALYSED

    def _set_ui_state(self, state: AppUiState) -> None:
        self.ui_state = state
        self._update_action_buttons()

    def _register_event_handlers(self) -> None:
        self.events.on_model_loaded(self._handle_model_loaded)
        self.events.on_analysis_done(self._handle_analysis_done)
        self.events.on_candidate_selected(self._handle_candidate_selected)
        self.events.on_settings_changed(self._handle_settings_changed)

    def _handle_model_loaded(self, _model_info: Dict[str, Any]) -> None:
        self._clear_results()
        self._set_ui_state(AppUiState.MODEL_LOADED)

    def _handle_analysis_done(self, analysis_result: AnalysisResult) -> None:
        payload = analysis_result.plot_data if isinstance(analysis_result.plot_data, dict) else {}
        analysis = payload.get("analysis")
        picks = payload.get("picks")
        params = payload.get("params")
        if not isinstance(analysis, dict) or not isinstance(picks, list) or not isinstance(params, Params):
            return
        self._update_diagnostics(analysis)
        self._update_table(analysis, picks, params)
        self._update_plots(analysis, picks, params)
        self._set_ui_state(self._infer_ui_state())
        self._refresh_memory_forecast()

    def _handle_candidate_selected(self, _candidate: Optional[SelectedCandidate]) -> None:
        self._set_ui_state(self._infer_ui_state())
        self._refresh_memory_forecast()
        self._highlight_selected_boundary_in_plots()

    def _handle_settings_changed(self) -> None:
        self._set_ui_state(self._infer_ui_state())
        self._refresh_memory_forecast()

    def _emit_settings_changed(self, *_args: Any) -> None:
        self._sync_gui_state_from_vars()
        self.events.emit_settings_changed()

    # -------------------------- UI construction --------------------------

    def _toggle_settings(self) -> None:
        """Show/hide the settings block below the top model bar."""
        visible = bool(getattr(self, "var_settings_visible", tk.BooleanVar(value=True)).get())
        # current state is visible -> hide
        if visible:
            try:
                self.params_frame.pack_forget()
            except Exception:
                pass
            try:
                self.btn_toggle_settings.configure(text="Show settings")
            except Exception:
                pass
            self.var_settings_visible.set(False)
            return

        # currently hidden -> show again just below top bar
        try:
            self.params_frame.pack(fill=tk.X, padx=10, pady=(0, 8), after=self.winfo_children()[0])
        except Exception:
            try:
                self.params_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
            except Exception:
                pass
        try:
            self.btn_toggle_settings.configure(text="Hide settings")
        except Exception:
            pass
        self.var_settings_visible.set(True)

    def _on_pick_output_folder(self) -> None:
        """Choose and remember a default *working directory* for exports."""
        initial = self.default_output_dir
        if not initial:
            model_path = self.model_path if isinstance(self.model_path, str) else ""
            if model_path:
                initial = os.path.dirname(model_path)
        if not initial:
            initial = os.getcwd()

        picked = filedialog.askdirectory(title="Select working directory", initialdir=initial)
        if not picked:
            return

        self.default_output_dir = picked
        try:
            self._set_status(f"Working directory: {picked}")
        except Exception:
            pass

    def _build_ui(self):
        # --- Top bar: open model ---
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        self.btn_open = ttk.Button(top, text="Open Model…", command=self._on_open)
        self.btn_open.pack(side=tk.LEFT)

        self.lbl_model = ttk.Label(top, text="(no model loaded)")
        self.lbl_model.pack(side=tk.LEFT, padx=10)

        # Allow quickly hiding the settings to maximize plot area.
        self.var_settings_visible = tk.BooleanVar(value=True)
        self.btn_toggle_settings = ttk.Button(top, text="Hide settings", command=self._toggle_settings)
        self.btn_toggle_settings.pack(side=tk.RIGHT)

        # --- Parameters ---
        self.params_frame = ttk.LabelFrame(self, text="Analysis Parameters")
        self.params_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        # General parameters row
        general = ttk.Frame(self.params_frame)
        general.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        self.general_frame = general
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
        self.var_unknown_mb = tk.StringVar(value="2.0")

        # second row
        self.var_exclude_trivial = tk.BooleanVar(value=True)
        self.chk_exclude_trivial = ttk.Checkbutton(general, text="Exclude trivial ops", variable=self.var_exclude_trivial)
        self.chk_exclude_trivial.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.var_only_one = tk.BooleanVar(value=False)
        self.chk_only_one = ttk.Checkbutton(general, text="Only one crossing tensor", variable=self.var_only_one)
        self.chk_only_one.grid(row=1, column=2, columnspan=3, sticky="w", pady=(4, 0))

        # Strict boundary (feasible split): Part2 must not depend on additional *intermediate*
        # activations besides the selected cut tensors. Original model inputs are allowed
        # (common in transformer exports where e.g. masks are used throughout the network).
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

        # Candidate clustering: keep only the best-scoring candidate per region/bin.
        self.var_cluster_best_region = tk.BooleanVar(value=True)
        self.chk_cluster_best_region = ttk.Checkbutton(general, text="Best per region", variable=self.var_cluster_best_region)
        self.chk_cluster_best_region.grid(row=2, column=6, sticky="w", pady=(4, 0), padx=(0, 10))
        ToolTip(self.chk_cluster_best_region, "Reduce duplicate candidates: keep only the best-scoring split per region of boundary indices.")

        ttk.Label(general, text="Region (ops):").grid(row=2, column=7, sticky="e", pady=(4, 0))
        self.var_cluster_region_ops = tk.StringVar(value="auto")
        self.ent_cluster_region_ops = ttk.Entry(general, textvariable=self.var_cluster_region_ops, width=6)
        self.ent_cluster_region_ops.grid(row=2, column=8, sticky="w", padx=(4, 14), pady=(4, 0))

        ttk.Label(general, text="Mode:").grid(row=2, column=9, sticky="e", pady=(4, 0))
        self.var_cluster_mode = tk.StringVar(value="Auto")
        self.cb_cluster_mode = ttk.Combobox(
            general,
            textvariable=self.var_cluster_mode,
            values=["Auto", "Uniform", "Semantic (LLM)"],
            width=14,
            state="readonly",
        )
        self.cb_cluster_mode.grid(row=2, column=10, sticky="w", padx=(4, 0), pady=(4, 0))
        ToolTip(self.cb_cluster_mode, "Clustering mode for candidates. Auto: Semantic (LLM) if LLM presets are enabled, otherwise Uniform.")

        self.shape_unknown_frame = ttk.LabelFrame(self.params_frame, text="Shape & Unknown Handling")
        self.shape_unknown_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.shape_unknown_frame.columnconfigure(0, weight=1)
        su = ttk.Frame(self.shape_unknown_frame)
        su.pack(fill=tk.X, padx=8, pady=6)

        ttk.Label(su, text="Assume act bytes/elt:").grid(row=0, column=0, sticky="w")
        self.ent_bpe = ttk.Entry(su, textvariable=self.var_bpe, width=6)
        self.ent_bpe.grid(row=0, column=1, sticky="w", padx=(4, 14))

        self.lbl_unknown_mb = ttk.Label(su, text="Unknown MB/tensor:")
        self.lbl_unknown_mb.grid(row=0, column=2, sticky="w")
        self.ent_unknown_mb = ttk.Entry(su, textvariable=self.var_unknown_mb, width=6)
        self.ent_unknown_mb.grid(row=0, column=3, sticky="w", padx=(4, 14))

        # Ranking frame
        self.rank_frame = ttk.LabelFrame(self.params_frame, text="Ranking")
        self.rank_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.rank_frame.columnconfigure(0, weight=1)

        r = ttk.Frame(self.rank_frame)
        r.pack(fill=tk.X, padx=8, pady=6)

        self.var_rank = tk.StringVar(value="score")
        ttk.Label(r, text="Ranking:").grid(row=0, column=0, sticky="w")
        self.cb_rank = ttk.Combobox(r, textvariable=self.var_rank, values=["cut", "score", "latency"], width=10, state="readonly")
        self.cb_rank.grid(row=0, column=1, sticky="w", padx=(4, 10))
        self.cb_rank.bind("<<ComboboxSelected>>", lambda _e: self._on_rank_changed(), add=True)

        self.var_score_advanced = tk.BooleanVar(value=False)
        self.chk_score_advanced = ttk.Checkbutton(
            r,
            text="Detail (Advanced)",
            variable=self.var_score_advanced,
            command=self._toggle_scoring_advanced,
        )
        self.chk_score_advanced.grid(row=0, column=2, sticky="w", padx=(0, 10))

        self.score_adv_frame = ttk.Frame(self.rank_frame)
        self.score_adv_frame.pack(fill=tk.X, padx=8, pady=(0, 6))

        self.var_log_comm = tk.BooleanVar(value=True)
        self.chk_log_comm = ttk.Checkbutton(self.score_adv_frame, text="log10(1+comm)", variable=self.var_log_comm)
        self.chk_log_comm.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.var_w_comm = tk.StringVar(value="1.0")
        self.var_w_imb = tk.StringVar(value="3.0")
        self.var_w_tensors = tk.StringVar(value="0.2")

        ttk.Label(self.score_adv_frame, text="w_comm").grid(row=0, column=1, sticky="e")
        self.ent_w_comm = ttk.Entry(self.score_adv_frame, textvariable=self.var_w_comm, width=6)
        self.ent_w_comm.grid(row=0, column=2, sticky="w", padx=(2, 6))

        ttk.Label(self.score_adv_frame, text="w_imb").grid(row=0, column=3, sticky="e")
        self.ent_w_imb = ttk.Entry(self.score_adv_frame, textvariable=self.var_w_imb, width=6)
        self.ent_w_imb.grid(row=0, column=4, sticky="w", padx=(2, 6))

        ttk.Label(self.score_adv_frame, text="w_tensors").grid(row=0, column=5, sticky="e")
        self.ent_w_tensors = ttk.Entry(self.score_adv_frame, textvariable=self.var_w_tensors, width=6)
        self.ent_w_tensors.grid(row=0, column=6, sticky="w", padx=(2, 10))

        self.var_show_pareto = tk.BooleanVar(value=True)
        self.chk_show_pareto = ttk.Checkbutton(self.score_adv_frame, text="Show Pareto front", variable=self.var_show_pareto)
        self.chk_show_pareto.grid(row=0, column=7, sticky="w")

        self._toggle_scoring_advanced()

        # LLM shape presets (optional)
        # Useful for decoder-style models with KV-cache (e.g., Gemma/Llama) where
        # symbolic dimensions like batch_size / sequence_length / past_sequence_length
        # need to be concretized for reliable shape inference and FLOPs/bytes estimates.
        self.var_llm_enable = tk.BooleanVar(value=False)
        self.var_llm_preset = tk.StringVar(value="Standard")
        self.var_llm_mode = tk.StringVar(value="decode")   # 'decode' or 'prefill'
        self.var_llm_prefill = tk.StringVar(value="512")   # prompt length (tokens)
        self.var_llm_decode = tk.StringVar(value="2048")   # KV cache (past) length (tokens)
        self.var_llm_use_ort_symbolic = tk.BooleanVar(value=True)


        # Advanced options (collapsible + tabbed)
        self.var_adv_expanded = tk.BooleanVar(value=False)
        self.adv_container = ttk.Frame(self.params_frame)
        self.adv_container.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.adv_container.columnconfigure(0, weight=1)

        adv_toggle = ttk.Frame(self.adv_container)
        adv_toggle.grid(row=0, column=0, sticky="ew")
        adv_toggle.columnconfigure(0, weight=1)

        self.btn_adv_toggle = ttk.Button(adv_toggle, text="▶ Additional options (LLM / Latency / Hailo / Memory)")
        self.btn_adv_toggle.grid(row=0, column=0, sticky="w")

        self.adv_body = ttk.Frame(self.adv_container)
        self.adv_body.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.adv_body.columnconfigure(0, weight=1)

        self.adv_tabs = ttk.Notebook(self.adv_body)
        self.adv_tabs.grid(row=0, column=0, sticky="ew")

        tab_llm = ttk.Frame(self.adv_tabs)
        tab_lat = ttk.Frame(self.adv_tabs)
        tab_hailo = ttk.Frame(self.adv_tabs)
        tab_mem = ttk.Frame(self.adv_tabs)
        self.adv_tabs.add(tab_llm, text="LLM")
        self.adv_tabs.add(tab_lat, text="Latency")
        self.adv_tabs.add(tab_hailo, text="Hailo")
        self.adv_tabs.add(tab_mem, text="Memory")

        def _toggle_adv() -> None:
            expanded = not bool(self.var_adv_expanded.get())
            self.var_adv_expanded.set(expanded)
            if expanded:
                self.btn_adv_toggle.configure(text="▼ Additional options (LLM / Latency / Hailo / Memory)")
                self.adv_body.grid()
            else:
                self.btn_adv_toggle.configure(text="▶ Additional options (LLM / Latency / Hailo / Memory)")
                self.adv_body.grid_remove()
            self._update_dtype_proxy_visibility()

        self.btn_adv_toggle.configure(command=_toggle_adv)
        self.adv_body.grid_remove()
        self._update_dtype_proxy_visibility()

        # LLM shape presets tab
        llm_frame = ttk.LabelFrame(tab_llm, text="LLM shape presets")
        llm_frame.pack(fill=tk.X, padx=4, pady=4)
        for c in range(0, 8):
            llm_frame.columnconfigure(c, weight=0)
        llm_frame.columnconfigure(7, weight=1)

        # Preset selection + lengths
        preset_values = [
            "Standard",
            "Latency Critical (Chat)",
            "Throughput/RAG",
            "Custom",
        ]
        preset_map = {
            "Latency Critical (Chat)": (128, 512),
            "Standard": (512, 2048),
            "Throughput/RAG": (2048, 128),
        }

        def _apply_llm_preset(event=None) -> None:
            p = self.var_llm_preset.get().strip()
            if p in preset_map:
                prefill, dec = preset_map[p]
                self.var_llm_prefill.set(str(prefill))
                self.var_llm_decode.set(str(dec))

        ttk.Checkbutton(llm_frame, text="Enable LLM preset", variable=self.var_llm_enable).grid(
            row=0, column=0, sticky="w", padx=(6, 8), pady=(4, 2)
        )

        ttk.Label(llm_frame, text="Preset:").grid(row=0, column=1, sticky="e", padx=(0, 4))
        llm_preset_cb = ttk.Combobox(llm_frame, textvariable=self.var_llm_preset, values=preset_values, state="readonly", width=22)
        llm_preset_cb.grid(row=0, column=2, sticky="w", padx=(0, 10))
        llm_preset_cb.bind("<<ComboboxSelected>>", _apply_llm_preset)

        ttk.Label(llm_frame, text="Prefill (tokens):").grid(row=0, column=3, sticky="e", padx=(0, 4))
        ttk.Entry(llm_frame, textvariable=self.var_llm_prefill, width=8).grid(row=0, column=4, sticky="w", padx=(0, 10))

        ttk.Label(llm_frame, text="Decode past (tokens):").grid(row=0, column=5, sticky="e", padx=(0, 4))
        ttk.Entry(llm_frame, textvariable=self.var_llm_decode, width=8).grid(row=0, column=6, sticky="w", padx=(0, 10))

        ttk.Label(llm_frame, text="Apply as:").grid(row=1, column=1, sticky="e", padx=(0, 4), pady=(0, 4))
        ttk.Radiobutton(llm_frame, text="Decode", variable=self.var_llm_mode, value="decode").grid(
            row=1, column=2, sticky="w", padx=(0, 10), pady=(0, 4)
        )
        ttk.Radiobutton(llm_frame, text="Prefill", variable=self.var_llm_mode, value="prefill").grid(
            row=1, column=3, sticky="w", padx=(0, 10), pady=(0, 4)
        )

        ttk.Label(
            llm_frame,
            text="Tip: Decode uses seq_len=1 and past=Decode past. Prefill uses seq_len=Prefill and past=0.",
            foreground="#555555",
        ).grid(row=1, column=4, columnspan=4, sticky="w", padx=(0, 6), pady=(0, 4))

        ttk.Checkbutton(
            llm_frame,
            text="Use ORT symbolic shape inference (better shape coverage)",
            variable=self.var_llm_use_ort_symbolic,
        ).grid(row=2, column=1, columnspan=7, sticky="w", padx=(0, 6), pady=(0, 2))

        _apply_llm_preset()

        # Latency tab
        self.lat_frame = ttk.LabelFrame(tab_lat, text="Latency model")
        self.lat_frame.pack(fill=tk.X, padx=4, pady=4)

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

        self.var_mem_left = tk.StringVar(value="")
        ttk.Label(l, text="Peak act L:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.ent_mem_left = ttk.Entry(l, textvariable=self.var_mem_left, width=10)
        self.ent_mem_left.grid(row=3, column=1, sticky="w", padx=(4, 4), pady=(6, 0))

        self.var_mem_left_unit = tk.StringVar(value="MiB")
        self.cb_mem_left_unit = ttk.Combobox(l, textvariable=self.var_mem_left_unit, values=sorted(asc.UNIT_MULT.keys()), width=7, state="readonly")
        self.cb_mem_left_unit.grid(row=3, column=2, sticky="w", padx=(0, 12), pady=(6, 0))

        self.var_mem_right = tk.StringVar(value="")
        ttk.Label(l, text="Peak act R:").grid(row=3, column=3, sticky="w", pady=(6, 0))
        self.ent_mem_right = ttk.Entry(l, textvariable=self.var_mem_right, width=10)
        self.ent_mem_right.grid(row=3, column=4, sticky="w", padx=(4, 4), pady=(6, 0))

        self.var_mem_right_unit = tk.StringVar(value="MiB")
        self.cb_mem_right_unit = ttk.Combobox(l, textvariable=self.var_mem_right_unit, values=sorted(asc.UNIT_MULT.keys()), width=7, state="readonly")
        self.cb_mem_right_unit.grid(row=3, column=5, sticky="w", padx=(0, 12), pady=(6, 0))

        # Hailo tab
        self.hailo_frame = ttk.LabelFrame(tab_hailo, text="Hailo feasibility check")
        self.hailo_frame.pack(fill=tk.X, padx=4, pady=4)

        hf = ttk.Frame(self.hailo_frame)
        hf.pack(fill=tk.X, padx=8, pady=6)

        self.var_hailo_check = tk.BooleanVar(value=False)
        self.chk_hailo_check = ttk.Checkbutton(
            hf,
            text="Enable parse-check in ranking",
            variable=self.var_hailo_check,
        )
        self.chk_hailo_check.grid(row=0, column=0, sticky="w")
        ToolTip(
            self.chk_hailo_check,
            "If enabled, the tool will export Part1/Part2 for top candidates and run a Hailo translate (parse-only).\n"
            "Requires Hailo DFC/SDK either in this Python environment (Linux), or via WSL2 (Windows backend mode).",
        )

        self.var_hailo_hw_arch = tk.StringVar(value="hailo8")
        ttk.Label(hf, text="HW arch:").grid(row=0, column=1, sticky="e", padx=(14, 2))
        self.cb_hailo_hw_arch = ttk.Combobox(
            hf,
            textvariable=self.var_hailo_hw_arch,
            # Editable so future arch strings (hailo10*, etc.) do not require
            # a GUI patch.
            values=["hailo8", "hailo8l", "hailo8r", "hailo10", "hailo10h"],
            width=10,
            state="normal",
        )
        self.cb_hailo_hw_arch.grid(row=0, column=2, sticky="w", padx=(0, 12))

        self.var_hailo_max_checks = tk.StringVar(value="25")
        ttk.Label(hf, text="Max checks:").grid(row=0, column=3, sticky="e", padx=(0, 2))
        self.ent_hailo_max_checks = ttk.Entry(hf, textvariable=self.var_hailo_max_checks, width=8)
        self.ent_hailo_max_checks.grid(row=0, column=4, sticky="w", padx=(0, 12))

        self.var_hailo_fixup = tk.BooleanVar(value=True)
        self.chk_hailo_fixup = ttk.Checkbutton(hf, text="ONNX fixup", variable=self.var_hailo_fixup)
        self.chk_hailo_fixup.grid(row=0, column=5, sticky="w")

        self.var_hailo_keep = tk.BooleanVar(value=False)
        self.chk_hailo_keep = ttk.Checkbutton(hf, text="Keep artifacts", variable=self.var_hailo_keep)
        self.chk_hailo_keep.grid(row=0, column=6, sticky="w", padx=(10, 0))

        self.var_hailo_target = tk.StringVar(value="either")
        ttk.Label(hf, text="Target:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.cb_hailo_target = ttk.Combobox(
            hf,
            textvariable=self.var_hailo_target,
            values=["either", "part2", "part1"],
            width=10,
            state="readonly",
        )
        self.cb_hailo_target.grid(row=1, column=1, sticky="w", pady=(6, 0))

        self.var_hailo_backend = tk.StringVar(value="auto")
        ttk.Label(hf, text="Backend:").grid(row=1, column=2, sticky="e", padx=(18, 2), pady=(6, 0))
        self.cb_hailo_backend = ttk.Combobox(
            hf,
            textvariable=self.var_hailo_backend,
            values=["auto", "local", "wsl", "venv"],
            width=10,
            state="readonly",
        )
        self.cb_hailo_backend.grid(row=1, column=3, sticky="w", pady=(6, 0))

        self.var_hailo_wsl_distro = tk.StringVar(value="")
        ttk.Label(hf, text="WSL distro:").grid(row=1, column=4, sticky="e", padx=(14, 2), pady=(6, 0))
        self.ent_hailo_wsl_distro = ttk.Entry(hf, textvariable=self.var_hailo_wsl_distro, width=18)
        self.ent_hailo_wsl_distro.grid(row=1, column=5, sticky="w", pady=(6, 0))

        # Use managed DFC profiles by default (Hailo-8 vs Hailo-10). Users can
        # still override this with an explicit `source <venv>/bin/activate` path.
        self.var_hailo_wsl_venv = tk.StringVar(value="auto")
        ttk.Label(hf, text="Venv override:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.ent_hailo_wsl_venv = ttk.Entry(hf, textvariable=self.var_hailo_wsl_venv, width=56)
        self.ent_hailo_wsl_venv.grid(row=2, column=1, columnspan=5, sticky="w", pady=(6, 0))

        btns = ttk.Frame(hf)
        btns.grid(row=2, column=6, sticky="w", padx=(12, 0), pady=(6, 0))
        self.btn_hailo_test = ttk.Button(btns, text="Test backend", command=self._hailo_test_backend)
        self.btn_hailo_test.pack(side=tk.TOP, fill=tk.X)
        self.btn_hailo_clear_cache = ttk.Button(btns, text="Clear cache", command=self._hailo_clear_cache)
        self.btn_hailo_clear_cache.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # Memory tab
        self.var_memf_left_accel = tk.StringVar(value="")
        self.var_memf_right_accel = tk.StringVar(value="")
        self.var_memf_interface = tk.StringVar(value="")
        self.var_memf_include_kv = tk.BooleanVar(value=True)
        self.var_memf_include_comm = tk.BooleanVar(value=True)
        self.var_memf_policy = tk.StringVar(value="max_peak_or_comm")
        self.var_memf_filter_fit = tk.BooleanVar(value=False)
        self.var_memf_left_text = tk.StringVar(value="Left: n/a")
        self.var_memf_right_text = tk.StringVar(value="Right: n/a")

        accel_names = [str(x.get("name")) for x in (self.accel_specs.get("accelerators") or [])]
        iface_names = [str(x.get("name")) for x in (self.accel_specs.get("interfaces") or [])]
        if accel_names:
            self.var_memf_left_accel.set(accel_names[0])
            self.var_memf_right_accel.set(accel_names[min(1, len(accel_names)-1)])
        if iface_names:
            self.var_memf_interface.set(iface_names[0])

        memf = ttk.LabelFrame(tab_mem, text="Memory forecast")
        memf.pack(fill=tk.X, padx=4, pady=4)
        row0 = ttk.Frame(memf)
        row0.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(row0, text="Left accelerator:").pack(side=tk.LEFT)
        ttk.Combobox(row0, textvariable=self.var_memf_left_accel, values=accel_names, width=28, state="readonly").pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(row0, text="Right accelerator:").pack(side=tk.LEFT)
        ttk.Combobox(row0, textvariable=self.var_memf_right_accel, values=accel_names, width=28, state="readonly").pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(row0, text="Interface:").pack(side=tk.LEFT)
        ttk.Combobox(row0, textvariable=self.var_memf_interface, values=iface_names, width=22, state="readonly").pack(side=tk.LEFT, padx=(4, 10))

        row1 = ttk.Frame(memf)
        row1.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Checkbutton(row1, text="include KV cache", variable=self.var_memf_include_kv).pack(side=tk.LEFT)
        ttk.Checkbutton(row1, text="include comm buffers", variable=self.var_memf_include_comm).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Label(row1, text="Policy:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Combobox(row1, textvariable=self.var_memf_policy, values=["max_peak_or_comm", "sum_peak_and_comm"], width=20, state="readonly").pack(side=tk.LEFT)
        ttk.Checkbutton(row1, text="show only fitting candidates", variable=self.var_memf_filter_fit, command=self._refresh_memory_forecast).pack(side=tk.LEFT, padx=(12, 0))

        try:
            ttk.Style(self).configure("MemGreen.Horizontal.TProgressbar", troughcolor="#eeeeee", background="#1c9c4a")
            ttk.Style(self).configure("MemRed.Horizontal.TProgressbar", troughcolor="#eeeeee", background="#c0392b")
        except Exception:
            pass
        self.pb_mem_left = ttk.Progressbar(memf, orient="horizontal", mode="determinate", maximum=100, style="MemGreen.Horizontal.TProgressbar")
        self.pb_mem_left.pack(fill=tk.X, padx=8, pady=(0, 2))
        ttk.Label(memf, textvariable=self.var_memf_left_text).pack(anchor="w", padx=8)
        self.pb_mem_right = ttk.Progressbar(memf, orient="horizontal", mode="determinate", maximum=100, style="MemGreen.Horizontal.TProgressbar")
        self.pb_mem_right.pack(fill=tk.X, padx=8, pady=(2, 2))
        ttk.Label(memf, textvariable=self.var_memf_right_text).pack(anchor="w", padx=8, pady=(0, 6))

        for v in (self.var_memf_left_accel, self.var_memf_right_accel, self.var_memf_interface, self.var_memf_policy, self.var_memf_include_comm, self.var_memf_include_kv):
            v.trace_add("write", lambda *_: self._refresh_memory_forecast())


        # Diagnostics frame
        self.diag_frame = ttk.LabelFrame(self.params_frame, text="Diagnostics")
        self.diag_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))

        d = ttk.Frame(self.diag_frame)
        d.pack(fill=tk.X, padx=8, pady=6)

        self.var_shape_coverage = tk.StringVar(value="(run analysis)")
        self.var_unknown_crossing = tk.StringVar(value="(run analysis)")
        self.var_diag_note = tk.StringVar(value="")
        # Extra info line for LLM presets (effective shapes, mode, etc.).
        self.var_llm_info = tk.StringVar(value="")

        ttk.Label(d, text="Shape coverage (known/produced):").grid(row=0, column=0, sticky="w")
        self.lbl_cov = ttk.Label(d, textvariable=self.var_shape_coverage)
        self.lbl_cov.grid(row=0, column=1, sticky="w", padx=(6, 18))

        ttk.Label(d, text="Max unknown crossing tensors:").grid(row=0, column=2, sticky="w")
        self.lbl_unk = ttk.Label(d, textvariable=self.var_unknown_crossing)
        self.lbl_unk.grid(row=0, column=3, sticky="w", padx=(6, 18))

        self.lbl_note = ttk.Label(d, textvariable=self.var_diag_note, foreground="#b00020")
        self.lbl_note.grid(row=0, column=4, sticky="w")

        # LLM info line (shown only when the LLM preset is enabled / used).
        self.lbl_llm_info = ttk.Label(d, textvariable=self.var_llm_info, foreground="#444444")
        self.lbl_llm_info.grid(row=1, column=0, columnspan=6, sticky="w", pady=(4, 0))

        # Nordstern (relevance analysis for unknown tensor sizes)
        self.btn_nordstern = ttk.Button(d, text="Nordstern…", command=self._show_nordstern)
        self.btn_nordstern.grid(row=0, column=5, sticky="e", padx=(10, 0))
        ToolTip(self.btn_nordstern, "Show a relevance analysis for tensors with unknown activation sizes.\n"
                                  "This helps identify which missing shape information could impact split decisions most.")

        # ---------------- Actions (analyse + export/split) ----------------
        # Keep this compact (small Analyse button) and always visible.
        action_bar = ttk.Frame(self.params_frame)
        action_bar.grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.action_bar = action_bar
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
        self.mid_pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        mid = self.mid_pane
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # ------------------------------ Table ------------------------------
        table_frame = ttk.LabelFrame(mid, text="Suggested Boundaries")
        mid.add(table_frame, weight=1)

        # Use grid so the bottom of the frame is never "eaten" by the Treeview.
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(1, weight=1)

        filter_row = ttk.Frame(table_frame)
        filter_row.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
        for ci, w in enumerate((0, 0, 0, 0, 1)):
            filter_row.columnconfigure(ci, weight=w)

        self.var_cand_search = tk.StringVar(value="")
        self.var_cand_search_regex = tk.BooleanVar(value=False)
        self.var_cand_hide_dirty = tk.BooleanVar(value=False)
        self.var_cand_group_semantic = tk.BooleanVar(value=False)
        self.var_cand_sort = tk.StringVar(value="Rank ↑")
        self.var_cand_advanced = tk.BooleanVar(value=False)

        ttk.Label(filter_row, text="Search:").grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.ent_cand_search = ttk.Entry(filter_row, textvariable=self.var_cand_search, width=28)
        self.ent_cand_search.grid(row=0, column=1, sticky="w", padx=(0, 6))
        self.chk_cand_regex = ttk.Checkbutton(filter_row, text="Regex", variable=self.var_cand_search_regex, command=self._refresh_candidates_table)
        self.chk_cand_regex.grid(row=0, column=2, sticky="w", padx=(0, 8))
        self.chk_cand_dirty = ttk.Checkbutton(filter_row, text="Hide dirty splits", variable=self.var_cand_hide_dirty, command=self._refresh_candidates_table)
        self.chk_cand_dirty.grid(row=0, column=3, sticky="w", padx=(0, 8))
        self.chk_cand_group = ttk.Checkbutton(filter_row, text="Group by semantic transition", variable=self.var_cand_group_semantic, command=self._refresh_candidates_table)
        self.chk_cand_group.grid(row=0, column=4, sticky="w", padx=(0, 8))

        ttk.Label(filter_row, text="Sort:").grid(row=0, column=5, sticky="e", padx=(6, 4))
        self.cb_cand_sort = ttk.Combobox(filter_row, textvariable=self.var_cand_sort, state="readonly", width=14,
                                         values=["Rank ↑", "Boundary ↑", "Boundary ↓", "Cut MB ↑", "Cut MB ↓", "Clean (best)"])
        self.cb_cand_sort.grid(row=0, column=6, sticky="e", padx=(0, 8))

        self.chk_cand_advanced = ttk.Checkbutton(filter_row, text="Detail (Advanced)", variable=self.var_cand_advanced, command=self._refresh_candidates_table)
        self.chk_cand_advanced.grid(row=0, column=7, sticky="e")

        self.ent_cand_search.bind("<KeyRelease>", self._refresh_candidates_table, add="+")
        self.cb_cand_sort.bind("<<ComboboxSelected>>", self._refresh_candidates_table, add="+")

        cols = [
            "rank",
            "clean",
            "boundary",
            "semantic",
            "cut_mb",
            "num_tensors",
            "gflops_left",
            "gflops_right",
            "left_op",
            "right_op",
            "peak_left_mib",
            "peak_right_mib",
            "peak_max_mib",
            "fits_left",
            "fits_right",
            "ram_left_gb",
            "ram_right_gb",
        ]

        table_inner = ttk.Frame(table_frame)
        table_inner.grid(row=1, column=0, sticky="nsew")
        table_inner.columnconfigure(0, weight=1)
        table_inner.rowconfigure(0, weight=1)

        tree_style = ttk.Style(self)
        tree_style.map(
            "Candidate.Treeview",
            background=[("selected", "#1f6feb")],
            foreground=[("selected", "#ffffff")],
        )

        self.tree = ttk.Treeview(table_inner, columns=cols, show="headings", style="Candidate.Treeview")
        self.tree.heading("rank", text="#")
        self.tree.heading("clean", text="Clean")
        self.tree.heading("boundary", text="Boundary")
        self.tree.heading("semantic", text="Semantic")
        self.tree.heading("left_op", text="Left op")
        self.tree.heading("right_op", text="Right op")
        self.tree.heading("cut_mb", text="Cut (MB)")
        self.tree.heading("num_tensors", text="#Tensors")
        self.tree.heading("gflops_left", text="Compute Left (GFLOPs)")
        self.tree.heading("gflops_right", text="Compute Right (GFLOPs)")
        self.tree.heading("peak_left_mib", text="Peak L (MiB)")
        self.tree.heading("peak_right_mib", text="Peak R (MiB)")
        self.tree.heading("peak_max_mib", text="Peak max (MiB)")
        self.tree.heading("fits_left", text="Fits L")
        self.tree.heading("fits_right", text="Fits R")
        self.tree.heading("ram_left_gb", text="RAM L (GB)")
        self.tree.heading("ram_right_gb", text="RAM R (GB)")

        self.tree.column("rank", width=40, anchor=tk.E)
        self.tree.column("clean", width=60, anchor=tk.CENTER)
        self.tree.column("boundary", width=80, anchor=tk.E)
        self.tree.column("semantic", width=190)
        self.tree.column("left_op", width=150)
        self.tree.column("right_op", width=150)
        self.tree.column("cut_mb", width=90, anchor=tk.E)
        self.tree.column("num_tensors", width=80, anchor=tk.E)
        self.tree.column("gflops_left", width=135, anchor=tk.E)
        self.tree.column("gflops_right", width=135, anchor=tk.E)
        self.tree.column("peak_left_mib", width=110, anchor=tk.E)
        self.tree.column("peak_right_mib", width=110, anchor=tk.E)
        self.tree.column("peak_max_mib", width=110, anchor=tk.E)
        self.tree.column("fits_left", width=60, anchor=tk.CENTER)
        self.tree.column("fits_right", width=60, anchor=tk.CENTER)
        self.tree.column("ram_left_gb", width=95, anchor=tk.E)
        self.tree.column("ram_right_gb", width=95, anchor=tk.E)

        self.tree.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(table_inner, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        vsb.grid(row=0, column=1, sticky="ns")

        self.tree.tag_configure("pick", background="#eef6ff")
        self.tree.tag_configure("dirty", background="#fff2f2")

        self._configure_candidate_columns()

        # Enable split only when a boundary row (not a child tensor row) is selected
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_selection_changed, add="+")
        self.tree.bind("<Button-1>", self._on_tree_button_1, add="+")
        self.tree.bind("<Motion>", self._on_tree_motion_clean_tooltip, add="+")
        self.tree.bind("<Leave>", self._hide_tree_clean_tooltip, add="+")
        # ------------------------------ Plots ------------------------------
        plot_frame = ttk.LabelFrame(mid, text="Plots")
        mid.add(plot_frame, weight=3)

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
        self.canvas.mpl_connect("button_press_event", self._on_plot_click_select_candidate)

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

        ToolTip(self.ent_cluster_region_ops, "Candidate clustering region width in ops (or 'auto').")

        ToolTip(self.cb_rank, "Ranking mode:\n- cut: minimise communication bytes\n- score: weighted trade-off (comm + imbalance + tensor penalty)\n- latency: minimise predicted latency (requires bandwidth + GOPS L/R)")
        ToolTip(self.chk_score_advanced, "Show/hide advanced score-ranking controls (weights, log comm, Pareto overlay).")
        ToolTip(self.chk_log_comm, "Use log10(1+comm) inside the score to reduce domination by very large activations.")
        ToolTip(self.ent_w_comm, "Weight for communication term in the score.")
        ToolTip(self.ent_w_imb, "Weight for compute imbalance term in the score.")
        ToolTip(self.ent_w_tensors, "Weight for crossing-tensor penalty in the score.")
        ToolTip(self.chk_show_pareto, "Overlay the Pareto front (comm vs imbalance) in the Pareto plot.")

        ToolTip(
            self.btn_adv_toggle,
            "Show/hide additional option tabs (LLM, Latency, Hailo, Memory).\n"
            "Use the Latency tab for latency ranking and the Memory tab for RAM fit forecast.",
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

        settings_vars = [
            self.var_topk, self.var_min_gap, self.var_min_compute, self.var_batch, self.var_bpe,
            self.var_unknown_mb, self.var_exclude_trivial, self.var_only_one, self.var_strict_boundary,
            self.var_rank, self.var_log_comm, self.var_w_comm, self.var_w_imb, self.var_w_tensors, self.var_show_pareto,
            self.var_memf_left_accel, self.var_memf_right_accel, self.var_memf_interface,
            self.var_memf_include_kv, self.var_memf_include_comm, self.var_memf_policy,
            self.var_split_validate, self.var_split_runner, self.var_split_folder,
            self.var_split_ctx_full, self.var_split_ctx_cutflow, self.var_split_ctx_hops,
            self.var_llm_enable, self.var_llm_preset, self.var_llm_mode, self.var_llm_prefill,
            self.var_llm_decode, self.var_llm_use_ort_symbolic,
        ]
        for var in settings_vars:
            var.trace_add("write", self._emit_settings_changed)

    # ------------------------- Advanced panel helpers ------------------------

    def _toggle_scoring_advanced(self) -> None:
        if bool(getattr(self, "var_score_advanced", tk.BooleanVar(value=False)).get()):
            self.score_adv_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
        else:
            self.score_adv_frame.pack_forget()

    def _update_dtype_proxy_visibility(self, coverage: Optional[float] = None) -> None:
        """Show dtype/unknown proxy controls only in advanced mode or at low coverage."""
        low_coverage = False
        try:
            low_coverage = (coverage is not None) and float(coverage) < 0.90
        except Exception:
            low_coverage = False
        show_proxy = bool(getattr(self, "var_adv_expanded", tk.BooleanVar(value=False)).get()) or low_coverage
        if show_proxy:
            self.lbl_unknown_mb.grid()
            self.ent_unknown_mb.grid()
        else:
            self.lbl_unknown_mb.grid_remove()
            self.ent_unknown_mb.grid_remove()

    def _on_rank_changed(self):
        """Auto-open advanced options and jump to Latency tab when needed."""
        if (self.var_rank.get() or "").strip().lower() == "latency":
            if not bool(self.var_adv_expanded.get()):
                self.var_adv_expanded.set(True)
                self.btn_adv_toggle.configure(text="▼ Additional options (LLM / Latency / Hailo / Memory)")
                self.adv_body.grid()
            try:
                self.adv_tabs.select(1)
            except Exception:
                pass

    # ------------------------- Hailo: status probes ------------------------

    def _hailo_schedule_status_refresh(self, delay_ms: int = 400) -> None:
        """Debounced refresh for the Hailo backend status badges."""

        try:
            job = getattr(self, "_hailo_status_refresh_job", None)
            if job is not None:
                try:
                    self.after_cancel(job)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            self._hailo_status_refresh_job = self.after(delay_ms, self._hailo_refresh_status)
        except Exception:
            # If scheduling fails (early startup), fall back to immediate.
            try:
                self._hailo_refresh_status()
            except Exception:
                pass

    def _hailo_refresh_status(self) -> None:
        """Probe Hailo DFC availability for Hailo-8 and Hailo-10 and update UI."""

        # Avoid overlapping probes.
        if bool(getattr(self, "_hailo_status_probe_running", False)):
            return
        self._hailo_status_probe_running = True

        backend = (getattr(self, "var_hailo_backend", tk.StringVar(value="auto")).get() or "auto").strip()
        wsl_distro = (getattr(self, "var_hailo_wsl_distro", tk.StringVar(value="")).get() or "").strip()
        wsl_venv = (getattr(self, "var_hailo_wsl_venv", tk.StringVar(value="auto")).get() or "auto").strip() or "auto"

        # Best-effort normalization (fix common typo "Ubuntu_22.04" -> "Ubuntu-22.04").
        try:
            from .hailo_backend import normalize_wsl_distro_name

            norm = normalize_wsl_distro_name(wsl_distro)
            if norm != wsl_distro:
                wsl_distro = norm
                try:
                    getattr(self, "var_hailo_wsl_distro").set(norm)
                except Exception:
                    pass
        except Exception:
            pass

        def _worker() -> None:
            try:
                from .hailo_backend import hailo_probe_auto

                res_h8 = hailo_probe_auto(
                    backend=backend,
                    hw_arch="hailo8",
                    wsl_distro=wsl_distro,
                    wsl_venv_activate=wsl_venv,
                    # First import/probe can be slow on cold-start WSL (starting the VM,
                    # importing large deps, JIT init, etc.). 20s was too aggressive and
                    # produced false negatives.
                    timeout_s=90,
                )
                res_h10 = hailo_probe_auto(
                    backend=backend,
                    hw_arch="hailo10",
                    wsl_distro=wsl_distro,
                    wsl_venv_activate=wsl_venv,
                    timeout_s=90,
                )
            except Exception as e:
                res_h8 = None
                res_h10 = None
                err = f"{type(e).__name__}: {e}"

                def _apply_err() -> None:
                    self._hailo_status_probe_running = False
                    for attr in ("hailo_badge_h8", "hailo_badge_h10"):
                        b = getattr(self, attr, None)
                        if b is not None:
                            try:
                                b.set(text="Hailo: error", level="error")
                            except Exception:
                                pass
                    v = getattr(self, "var_hailo_status_details", None)
                    if v is not None:
                        try:
                            v.set(err)
                        except Exception:
                            pass

                    # On errors we still allow provisioning attempts.
                    btn_prov = getattr(self, "_hailo_btn_provision", None)
                    if btn_prov is not None and not bool(getattr(self, "_hailo_provision_running", False)):
                        try:
                            btn_prov.configure(state="normal")
                        except Exception:
                            pass

                try:
                    self.after(0, _apply_err)
                except Exception:
                    _apply_err()
                return

            def _apply() -> None:
                self._hailo_status_probe_running = False

                # Store last results so the user can inspect details by clicking the badges.
                try:
                    self._hailo_last_probe_h8 = res_h8
                    self._hailo_last_probe_h10 = res_h10
                except Exception:
                    pass

                def _badge_text(prefix: str, ok: bool, backend_name: str) -> str:
                    mark = "✓" if ok else "✗"
                    # Keep it compact; backend info is useful when 'auto' picks WSL.
                    return f"{prefix} {mark} ({backend_name})"

                # Update badges
                b8 = getattr(self, "hailo_badge_h8", None)
                b10 = getattr(self, "hailo_badge_h10", None)
                if b8 is not None and res_h8 is not None:
                    try:
                        b8.set(text=_badge_text("Hailo-8", bool(res_h8.ok), str(res_h8.backend)), level=("ok" if res_h8.ok else "error"))
                    except Exception:
                        pass
                if b10 is not None and res_h10 is not None:
                    try:
                        b10.set(text=_badge_text("Hailo-10", bool(res_h10.ok), str(res_h10.backend)), level=("ok" if res_h10.ok else "error"))
                    except Exception:
                        pass

                # Details line (optional)
                details_var = getattr(self, "var_hailo_status_details", None)
                if details_var is not None:
                    try:
                        msg_parts = []
                        if res_h8 is not None and (not res_h8.ok):
                            msg_parts.append(f"H8: {res_h8.reason}")
                        if res_h10 is not None and (not res_h10.ok):
                            msg_parts.append(f"H10: {res_h10.reason}")
                        details_var.set((" | ".join(msg_parts) + "  (click badge for details)") if msg_parts else "")
                    except Exception:
                        pass

                # Enable the Provision button only when something is not OK.
                btn_prov = getattr(self, "_hailo_btn_provision", None)
                if btn_prov is not None and not bool(getattr(self, "_hailo_provision_running", False)):
                    try:
                        all_ok = bool(res_h8 is not None and res_h8.ok) and bool(res_h10 is not None and res_h10.ok)
                        btn_prov.configure(state=("disabled" if all_ok else "normal"))
                    except Exception:
                        pass

            try:
                self.after(0, _apply)
            except Exception:
                _apply()

        threading.Thread(target=_worker, daemon=True).start()

    def _hailo_show_probe_details(self, which: str) -> None:
        """Show a detail dialog for the last Hailo DFC probe (Hailo-8 / Hailo-10)."""

        w = (which or "").strip().lower()
        if w in ("hailo8", "h8"):
            res = getattr(self, "_hailo_last_probe_h8", None)
            title = "Hailo DFC status (Hailo-8)"
        else:
            res = getattr(self, "_hailo_last_probe_h10", None)
            title = "Hailo DFC status (Hailo-10)"

        if res is None:
            messagebox.showinfo(title, "No probe result available yet. Click 'Refresh status' first.")
            return

        lines = []
        try:
            lines.append(f"OK: {bool(getattr(res, 'ok', False))}")
            lines.append(f"Backend: {getattr(res, 'backend', '')}")
            r = str(getattr(res, "reason", "") or "").strip()
            if r:
                lines.append(f"Reason: {r}")

            det = getattr(res, "details", None) or {}
            if isinstance(det, dict):
                for k in ("profile_id", "wsl_distro", "wsl_venv_activate", "venv_activate", "venv_python", "returncode"):
                    if k in det and det[k] is not None:
                        lines.append(f"{k}: {det[k]}")

                tail = str(det.get("output_tail") or "").strip()
                if tail:
                    lines.append("")
                    lines.append("Output (tail):")
                    lines.append(tail)

            # Small actionable hint for common failure.
            if "protobuf" in r.lower() and "mismatch" in r.lower():
                lines.append("")
                lines.append("Hint: Your managed DFC venv likely drifted (protobuf upgraded).")
                lines.append("Fix: delete the venv and re-run ./scripts/provision_hailo_dfcs_wsl.sh --all")

            # Early glibc blocker (common on Ubuntu 20.04).
            if "glibc" in r.lower() and ("too old" in r.lower() or "distro too old" in r.lower()):
                lines.append("")
                lines.append("Hint: Your selected Linux/WSL distro ships an older glibc than the Hailo DFC wheel requires.")
                lines.append("This typically shows up on Ubuntu 20.04 (glibc 2.31) while the DFC wheels require glibc >= 2.34.")
                lines.append("")
                lines.append("Fix (Windows/WSL):")
                lines.append("- Install a newer distro: wsl --install -d Ubuntu-22.04")
                lines.append("- In the GUI, set 'WSL distro' to Ubuntu-22.04")
                lines.append("- Click 'Provision DFC' (this creates venv_hailo8 / venv_hailo10 in that distro)")
                lines.append("- Click 'Refresh status'")
                lines.append("")
                lines.append("Note: each WSL distro has its own home directory and managed venvs, so provisioning must run inside the target distro.")
        except Exception:
            lines = ["Failed to format probe result."]

        messagebox.showinfo(title, "\n".join(lines))

    # ------------------------- Layout helpers ------------------------

    def _hailo_test_backend(self) -> None:
        """Quick sanity-check that the selected Hailo backend is reachable."""
        backend = (self.var_hailo_backend.get() or "auto").strip()
        wsl_distro = (self.var_hailo_wsl_distro.get() or "").strip()
        wsl_venv = (self.var_hailo_wsl_venv.get() or "").strip() or "auto"
        hw_arch = (self.var_hailo_hw_arch.get() or "hailo8").strip()

        try:
            from .hailo_backend import hailo_probe_auto

            res = hailo_probe_auto(
                backend=backend,
                hw_arch=hw_arch,
                wsl_distro=wsl_distro,
                wsl_venv_activate=wsl_venv,
                timeout_s=30.0,
            )
        except Exception as e:
            messagebox.showerror("Hailo backend test failed", str(e))
            return

        if res.ok:
            details = ""
            if res.details:
                # Compact key=value list
                parts = []
                for k, v in res.details.items():
                    if v is None:
                        continue
                    s = str(v)
                    if len(s) > 200:
                        s = s[:200] + "..."
                    parts.append(f"{k}={s}")
                if parts:
                    details = "\n\n" + "\n".join(parts)

            messagebox.showinfo(
                "Hailo backend",
                f"OK ({res.backend}).{details}",
            )
        else:
            messagebox.showerror(
                "Hailo backend",
                f"Not ready ({res.backend}).\n\n{res.reason}",
            )

    def _hailo_clear_cache(self) -> None:
        """Clear the persistent Hailo parse-check cache on disk."""
        self._hailo_cache = {}
        self._hailo_cache_dirty = False
        try:
            if self._hailo_cache_path.exists():
                self._hailo_cache_path.unlink()
        except Exception:
            pass
        messagebox.showinfo("Hailo cache", "Cleared Hailo parse-check cache.")

    def _hailo_provision_dfcs(self) -> None:
        """Provision/repair managed Hailo DFC environments (Hailo-8/Hailo-10).

        Runs the provisioning helper in the appropriate backend:
        - Windows: via WSL
        - Linux: directly

        Output is streamed into gui.log (Logs tab).
        """

        if getattr(self, "_hailo_provision_running", False):
            messagebox.showinfo("Hailo DFC provisioning", "Provisioning is already running. See Logs tab.")
            return

        self._hailo_provision_running = True

        btn = getattr(self, "_hailo_btn_provision", None)
        try:
            if btn is not None:
                btn.configure(state="disabled")
        except Exception:
            pass

        backend = (self.var_hailo_backend.get() or "auto").strip().lower()
        wsl_distro = (self.var_hailo_wsl_distro.get() or "").strip()
        wsl_venv = (self.var_hailo_wsl_venv.get() or "auto").strip() or "auto"

        # Best-effort normalization (fix common typo "Ubuntu_22.04" -> "Ubuntu-22.04").
        try:
            from .hailo_backend import normalize_wsl_distro_name

            norm = normalize_wsl_distro_name(wsl_distro)
            if norm != wsl_distro:
                wsl_distro = norm
                try:
                    self.var_hailo_wsl_distro.set(norm)
                except Exception:
                    pass
        except Exception:
            pass

        # Resolve repo root (contains scripts/).
        repo_root = Path(__file__).resolve().parents[1]

        def _worker() -> None:
            ok = False
            summary = ""
            prov_rc: Optional[int] = None
            prov_err: Optional[str] = None
            res_h8 = None
            res_h10 = None
            missing_wheels: List[str] = []
            try:
                # Quick preflight: are the DFC wheels present in resources/?
                try:
                    base = repo_root / "onnx_splitpoint_tool" / "resources" / "hailo"
                    for sub in ("hailo8", "hailo10"):
                        d = base / sub
                        if not d.exists() or not list(d.glob("*.whl")):
                            missing_wheels.append(sub)
                except Exception:
                    pass

                if sys.platform == "win32":
                    from .hailo_backend import hailo_wsl_available, windows_path_to_wsl, _wsl_exe

                    if not hailo_wsl_available():
                        raise RuntimeError("WSL backend not available (wsl.exe not found).")

                    repo_wsl = windows_path_to_wsl(str(repo_root))
                    bash = (
                        "set -e; "
                        f"cd {shlex.quote(repo_wsl)}; "
                        "./scripts/provision_hailo_dfcs_wsl.sh --all --force-reinstall"
                    )

                    cmd: List[str] = [_wsl_exe()]
                    if wsl_distro:
                        cmd += ["-d", wsl_distro]
                    cmd += ["--", "bash", "-lc", bash]
                else:
                    # Linux: call the same helper script directly.
                    script = repo_root / "scripts" / "provision_hailo_dfcs_wsl.sh"
                    if not script.exists():
                        raise RuntimeError(f"Provision script missing: {script}")
                    cmd = ["bash", str(script), "--all", "--force-reinstall"]

                logger.info("[hailo][provision] starting provisioning (backend=%s)", backend)
                logger.info("[hailo][provision] cmd=%s", cmd)
                if missing_wheels:
                    logger.warning("[hailo][provision] missing DFC wheels for: %s", ", ".join(missing_wheels))

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    logger.info("[hailo][provision] %s", line)
                rc = proc.wait()

                prov_rc = int(rc)
                ok = (rc == 0)
            except Exception as e:
                ok = False
                prov_err = f"{type(e).__name__}: {e}"
                summary = f"Provisioning failed: {prov_err}"
                logger.exception("[hailo][provision] failed")

            # Build a compact post-provision status summary (Hailo-8 / Hailo-10)
            # so users immediately know what is usable.
            # Post-provision sanity probe (best effort).
            res_h8 = None
            res_h10 = None
            try:
                from .hailo_backend import hailo_probe_auto

                res_h8 = hailo_probe_auto(
                    backend=backend,
                    hw_arch="hailo8",
                    wsl_distro=wsl_distro,
                    wsl_venv_activate=wsl_venv,
                    timeout_s=90,
                )
                res_h10 = hailo_probe_auto(
                    backend=backend,
                    hw_arch="hailo10",
                    wsl_distro=wsl_distro,
                    wsl_venv_activate=wsl_venv,
                    timeout_s=90,
                )
            except Exception:
                logger.exception("[hailo][provision] post-provision probe failed")

            # Format summary dialog.
            try:
                lines: List[str] = []
                if prov_err:
                    lines.append(f"Provisioning: FAILED ({prov_err})")
                elif prov_rc is None:
                    lines.append("Provisioning: FAILED (unknown error)")
                else:
                    lines.append(f"Provisioning: {'OK' if prov_rc == 0 else 'FAILED'} (exit code {prov_rc})")

                def _fmt_res(label: str, r) -> None:
                    if r is None:
                        lines.append(f"{label}: (no result)")
                        return
                    if bool(getattr(r, "ok", False)):
                        lines.append(f"{label}: OK ({getattr(r, 'backend', '')})")
                    else:
                        reason = str(getattr(r, "reason", "") or "").strip()
                        backend_name = str(getattr(r, "backend", "") or "")
                        if reason:
                            lines.append(f"{label}: FAIL ({backend_name}) - {reason}")
                        else:
                            lines.append(f"{label}: FAIL ({backend_name})")

                _fmt_res("Hailo-8", res_h8)
                _fmt_res("Hailo-10", res_h10)

                # Extra hints for common hard blockers.
                hints: List[str] = []
                for r, name in ((res_h8, "Hailo-8"), (res_h10, "Hailo-10")):
                    if r is None or bool(getattr(r, "ok", False)):
                        continue
                    reason = str(getattr(r, "reason", "") or "")
                    if "glibc" in reason.lower() or "GLIBC_" in reason:
                        hints.append(
                            f"{name}: Your Linux/WSL distro is too old for this DFC wheel (needs glibc >= 2.34). "
                            "Use Ubuntu 22.04/24.04 for DFC provisioning and set the GUI 'WSL distro' field accordingly."
                        )
                    if "pkg_resources" in reason:
                        hints.append(
                            f"{name}: 'pkg_resources' is missing. Fix by pinning setuptools<82 (Provision DFC does this automatically)."
                        )
                    if "hailo_sdk_client" in reason and ("not importable" in reason or "not installed" in reason):
                        hints.append(
                            f"{name}: DFC not installed in the managed venv. Make sure the DFC wheel is present in onnx_splitpoint_tool/resources/hailo/<profile>/ and re-run 'Provision DFC'."
                        )

                if hints:
                    lines.append("")
                    lines.append("Hints:")
                    for h in hints:
                        lines.append(f"- {h}")

                if missing_wheels:
                    lines.append("")
                    lines.append("Missing wheels:")
                    for sub in missing_wheels:
                        lines.append(
                            f"- {sub}: place the matching hailo_dataflow_compiler-*.whl into onnx_splitpoint_tool/resources/hailo/{sub}/"
                        )

                summary = "\n".join(lines)
            except Exception:
                # Fall back to a basic summary if formatting fails.
                if not summary:
                    summary = "Provisioning completed." if ok else "Provisioning failed. Check Logs tab."

            def _finish() -> None:
                self._hailo_provision_running = False
                try:
                    if btn is not None:
                        btn.configure(state="normal")
                except Exception:
                    pass

                # Decide dialog icon based on overall state.
                try:
                    all_ok = bool(res_h8 is not None and getattr(res_h8, "ok", False)) and bool(res_h10 is not None and getattr(res_h10, "ok", False))
                except Exception:
                    all_ok = False

                if ok and all_ok:
                    messagebox.showinfo("Hailo DFC provisioning", summary)
                elif ok:
                    messagebox.showwarning("Hailo DFC provisioning", summary)
                else:
                    messagebox.showerror("Hailo DFC provisioning", summary)

                # Refresh badges after provisioning attempt.
                try:
                    self._hailo_refresh_status()
                except Exception:
                    pass

            try:
                self.after(0, _finish)
            except Exception:
                _finish()

        threading.Thread(target=_worker, daemon=True).start()

    # ----------------------------- Event handlers -----------------------------

    def _set_selected_candidate_from_boundary(self, boundary: Optional[int]) -> None:
        """Set cached SelectedCandidate from boundary and emit selection event."""
        if boundary is None:
            self.selected_candidate = None
            self.events.emit_candidate_selected(None)
            return

        sem = ""
        if isinstance(self.analysis, dict):
            labels = self.analysis.get("semantic_labels_by_boundary") or []
            if int(boundary) < len(labels):
                sem = str(labels[int(boundary)] or "")

        cut_tensors: List[str] = []
        stats: Dict[str, Any] = {}
        if isinstance(self.analysis, dict):
            try:
                cut_tensors = list(
                    asc.cut_tensors_for_boundary(self.analysis["order"], self.analysis["nodes"], int(boundary))
                )
            except Exception:
                cut_tensors = []
            if self.analysis_result and self.analysis_result.memory_estimate:
                stats = dict(self.analysis_result.memory_estimate.get(int(boundary), {}))

        self.selected_candidate = SelectedCandidate(
            boundary_id=int(boundary),
            semantic_label=sem,
            cut_tensors=cut_tensors,
            stats=stats,
        )
        self.events.emit_candidate_selected(self.selected_candidate)
    def _apply_selected_row_table_tag(self, boundary: Optional[int]) -> None:
        """Highlight the currently selected boundary row in the candidates table.

        We use a tag-based highlight (see panel_analysis.py tag_configure) instead of
        relying on the native Treeview selection highlight, because per-row tag
        backgrounds (e.g. 'pick'/'dirty') can hide the selection background on
        some Tk themes/platforms.
        """
        tree = getattr(self, 'tree', None)
        if tree is None or boundary is None:
            return

        iid = f"b{int(boundary)}"
        try:
            if not tree.exists(iid):
                return
        except Exception:
            # If the widget is not a Treeview or already destroyed.
            return

        prev = getattr(self, '_selected_row_iid', None)
        if prev and prev != iid:
            try:
                if tree.exists(prev):
                    prev_tags = [t for t in tree.item(prev, 'tags') if t != 'selected_row']
                    tree.item(prev, tags=tuple(prev_tags))
            except Exception:
                pass

        try:
            cur_tags = list(tree.item(iid, 'tags'))
            if 'selected_row' not in cur_tags:
                cur_tags.append('selected_row')
                tree.item(iid, tags=tuple(cur_tags))
        except Exception:
            return

        self._selected_row_iid = iid


    def _on_tree_selection_changed(self, event=None) -> None:
        sel = self.tree.selection() if hasattr(self, "tree") else ()
        logger.debug("Tree selection event: sel=%s", sel)
        boundary = self._selected_boundary_index()
        iid = sel[0] if sel else ""
        tags = self.tree.item(iid, "tags") if iid else ()
        logger.info("Tree selection changed: iid=%s, tags=%s, boundary=%s", iid, tags, boundary)
        self._set_selected_candidate_from_boundary(boundary)
        # Keep the selected row visually marked in the table.
        self._apply_selected_row_table_tag(boundary)

    def _on_tree_button_1(self, evt=None):
        """Handle candidate-table left-clicks; only intercept clean-column clicks."""
        if evt is None or not hasattr(self, "tree"):
            return None
        row_id = self.tree.identify_row(evt.y)
        col_id = self.tree.identify_column(evt.x)
        logger.debug("Tree click: row=%s col=%s", row_id, col_id)
        if col_id == "#2" and row_id and row_id in self._cand_by_iid:
            logger.debug("Intercepted clean-column click for row=%s", row_id)
            return "break"
        return None

    def _on_plot_click_select_candidate(self, event) -> None:
        """Map clicks in boundary-index plots back to tree/candidate selection."""
        if event is None or event.xdata is None:
            return
        if event.inaxes not in {getattr(self, "ax_comm", None), getattr(self, "ax_comp", None), getattr(self, "ax_lat", None)}:
            return
        if not self._candidate_rows:
            return
        try:
            boundary = int(round(float(event.xdata)))
        except Exception:
            return
        candidates = [int(r.get("boundary", -1)) for r in self._candidate_rows]
        if not candidates:
            return
        boundary = min(candidates, key=lambda b: abs(int(b) - boundary))
        for item in self.tree.get_children(""):
            vals = self.tree.item(item, "values")
            if len(vals) < 3:
                continue
            try:
                if int(vals[2]) == int(boundary):
                    self.tree.selection_set(item)
                    self.tree.focus(item)
                    self.tree.see(item)
                    self._on_tree_selection_changed()
                    return
            except Exception:
                continue

    def _highlight_selected_boundary_in_plots(self) -> None:
        """Draw selection marker in plots so table↔plot↔inspector stay in sync."""
        b = self._selected_boundary_index()
        for ax in (getattr(self, "ax_comm", None), getattr(self, "ax_comp", None), getattr(self, "ax_pareto", None), getattr(self, "ax_lat", None)):
            if ax is None:
                continue
            for line in list(ax.lines):
                if getattr(line, "_split_selected_marker", False):
                    try:
                        line.remove()
                    except Exception:
                        pass
        if b is None:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass
            return
        for ax in (self.ax_comm, self.ax_comp, self.ax_lat):
            marker = ax.axvline(float(b), color="#d32f2f", linestyle="-", linewidth=1.2, alpha=0.9)
            setattr(marker, "_split_selected_marker", True)

        # Pareto selection cross
        axp = getattr(self, "ax_pareto", None)
        if axp is not None:
            try:
                analysis_result = getattr(self, "analysis_result", None)
                pd = getattr(analysis_result, "plot_data", {}) if analysis_result else {}
                costs = pd.get("costs_bytes") or []
                imb = pd.get("imbalance") or []
                if isinstance(costs, (list, tuple)) and isinstance(imb, (list, tuple)) and b < len(costs) and b < len(imb):
                    x = float(costs[b]) / 1e6
                    y = float(imb[b])
                    ln = axp.plot(
                        [x], [y],
                        marker="x",
                        color="#d32f2f",
                        markersize=9,
                        markeredgewidth=2,
                        linestyle="None",
                        zorder=5,
                    )[0]
                    setattr(ln, "_split_selected_marker", True)
            except Exception:
                pass
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _sync_gui_state_from_vars(self) -> None:
        analysis_params: Dict[str, Any] = {}
        llm_params: Dict[str, Any] = {}
        for spec in iter_specs():
            if getattr(spec, "deprecated", False):
                continue
            if not spec.var_name:
                continue
            var = getattr(self, spec.var_name, None)
            if var is None:
                continue
            value = var.get()
            if spec.param_type == "bool":
                value = bool(value)
            target = llm_params if spec.scope == "llm" else analysis_params
            target[spec.key] = value
        self.gui_state.analysis_params = analysis_params
        self.gui_state.llm_params = llm_params
        self.gui_state.hardware_selection = {
            "left_accel": self.var_memf_left_accel.get(),
            "right_accel": self.var_memf_right_accel.get(),
            "interface": self.var_memf_interface.get(),
            "policy": self.var_memf_policy.get(),
            "include_kv": bool(self.var_memf_include_kv.get()),
            "include_comm": bool(self.var_memf_include_comm.get()),
        }
        self.gui_state.export_flags = {
            "split_validate": bool(self.var_split_validate.get()),
            "split_runner": bool(self.var_split_runner.get()),
            "split_folder": bool(self.var_split_folder.get()),
            "ctx_full": bool(self.var_split_ctx_full.get()),
            "ctx_cutflow": bool(self.var_split_ctx_cutflow.get()),
            "ctx_hops": self.var_split_ctx_hops.get(),
        }

    def _sync_vars_from_gui_state(self) -> None:
        """Apply state dictionaries back to bound tkinter variables."""
        ap = dict(getattr(self.gui_state, "analysis_params", {}) or {})
        llm = dict(getattr(self.gui_state, "llm_params", {}) or {})

        for spec in iter_specs():
            if getattr(spec, "deprecated", False):
                continue
            if not spec.var_name:
                continue
            source = llm if spec.scope == "llm" else ap
            if spec.key not in source:
                continue
            var = getattr(self, spec.var_name, None)
            if var is None:
                continue
            caster = bool if spec.param_type == "bool" else str
            var.set(caster(source.get(spec.key)))

    def _apply_analysis_global_preset(self, preset_name: str, presets: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        """Apply global analysis/LLM preset through GuiState dictionaries."""
        self._sync_gui_state_from_vars()
        preset = dict((presets or {}).get(preset_name, {}) or {})
        self.gui_state.analysis_params.update(dict(preset.get("analysis", {}) or {}))
        self.gui_state.llm_params.update(dict(preset.get("llm", {}) or {}))
        self._sync_vars_from_gui_state()
        self.events.emit_settings_changed()

    def _analysis_modified_fields(self, preset_name: str, presets: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> List[str]:
        """Return fields diverging from the selected global preset."""
        preset = dict((presets or {}).get(preset_name, {}) or {})
        if not preset:
            return []
        self._sync_gui_state_from_vars()
        modified: List[str] = []
        for scope_name, source in (("analysis", self.gui_state.analysis_params), ("llm", self.gui_state.llm_params)):
            target = dict(preset.get(scope_name, {}) or {})
            for key, expected in target.items():
                if str(source.get(key)) != str(expected):
                    modified.append(f"{scope_name}.{key}")
        return modified

    def _on_open(self):
        path = filedialog.askopenfilename(
            title="Open ONNX model",
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
        )
        if not path:
            return
        self.model_path = path
        self.gui_state.current_model_path = path
        self.gui_state.model_type = "onnx"
        self.lbl_model.configure(text=os.path.basename(path))
        self.events.emit_model_loaded({"path": path, "model_type": "onnx"})

    def _on_analyse(self):
        if not self.gui_state.current_model_path:
            messagebox.showwarning("No model", "Please open an ONNX model first.")
            return

        try:
            params = self._read_params()
        except ValueError as e:
            messagebox.showerror("Invalid parameters", str(e))
            return

        # Run analysis in a background thread (Hailo parse-checks can take a while).
        self.btn_analyse.state(["disabled"])
        self._set_ui_state(AppUiState.ANALYSING)

        dlg = tk.Toplevel(self)
        dlg.title("Analysing model")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        msg_var = tk.StringVar(value="Starting...")
        ttk.Label(dlg, textvariable=msg_var).pack(padx=14, pady=(14, 8))
        pb = ttk.Progressbar(dlg, mode="indeterminate", length=360)
        pb.pack(padx=14, pady=(0, 14))
        pb.start(10)

        # Avoid leaving a half-finished UI state if the user closes the dialog.
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)

        q: "queue.Queue[tuple]" = queue.Queue()

        def _progress(msg: str) -> None:
            q.put(("progress", msg))

        def _worker() -> None:
            try:
                analysis = self._analyse_model(self.gui_state.current_model_path, params, progress_cb=_progress)
                _progress("Selecting candidates...")
                picks = self._select_picks(analysis, params, progress_cb=_progress)
                _progress("Computing diagnostics...")
                self._compute_nordstern(analysis, picks, params)
                q.put(("ok", analysis, picks))
            except Exception:
                logging.exception("Analysis failed")
                import traceback

                q.put(("err", traceback.format_exc()))

        threading.Thread(target=_worker, daemon=True).start()

        def _poll() -> None:
            try:
                while True:
                    item = q.get_nowait()
                    kind = item[0]
                    if kind == "progress":
                        msg_var.set(str(item[1]))
                    elif kind == "ok":
                        self.analysis = item[1]
                        self.current_picks = item[2]
                        self.analysis_result = AnalysisResult(
                            candidates=list(self.current_picks),
                            plot_data={"analysis": self.analysis, "picks": list(self.current_picks), "params": params},
                        )
                        self._last_params = params
                        self.events.emit_analysis_done(self.analysis_result)
                        # Persist Hailo cache updates.
                        self._save_hailo_cache()
                        pb.stop()
                        dlg.destroy()
                        self.btn_analyse.state(["!disabled"])
                        self._set_ui_state(AppUiState.ANALYSED)
                        return
                    elif kind == "err":
                        err_text = str(item[1])
                        pb.stop()
                        dlg.destroy()
                        self.btn_analyse.state(["!disabled"])
                        self._set_ui_state(self._infer_ui_state())
                        messagebox.showerror("Analysis failed", err_text)
                        return
            except queue.Empty:
                pass
            self.after(100, _poll)

        _poll()

    # ----------------------------- Parameters -----------------------------

    def _read_params(self) -> Params:
        self._sync_gui_state_from_vars()
        ap = self.gui_state.analysis_params
        llm = self.gui_state.llm_params
        hw = self.gui_state.hardware_selection
        mapped = gui_state_to_params_dict(ap, llm)

        topk = _safe_int(str(mapped.get("topk", "")))
        if topk is None or topk <= 0:
            raise ValueError("Top-k must be a positive integer.")

        min_gap = _safe_int(str(mapped.get("min_gap", "")))
        if min_gap is None or min_gap < 0:
            raise ValueError("Min gap must be an integer ≥ 0.")

        min_comp = _safe_float(str(mapped.get("min_compute_pct", "")))
        if min_comp is None:
            min_comp = 0.0
        if min_comp < 0:
            raise ValueError("Min compute each side (%) must be ≥ 0.")

        batch = _safe_int(str(mapped.get("batch_override", "")))
        bpe = _safe_int(str(mapped.get("assume_bpe", "")))

        unknown_mb = _safe_float(str(mapped.get("unknown_tensor_proxy_mb", "")))
        if unknown_mb is None:
            unknown_mb = 0.0
        if unknown_mb < 0:
            unknown_mb = 0.0

        exclude_trivial = bool(mapped.get("exclude_trivial", False))
        only_one = bool(mapped.get("only_single_tensor", False))
        strict_boundary = bool(mapped.get("strict_boundary", False))

        # Read pruning params directly from the mapped GUI state.
        # IMPORTANT: do NOT call _read_params() recursively here (would cause
        # infinite recursion / RecursionError).
        prune_skip_block = bool(mapped.get("prune_skip_block", False))
        skip_min_span = _safe_int(str(mapped.get("skip_min_span", ""))) or 0
        if skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = _safe_int(str(mapped.get("skip_allow_last_n", ""))) or 0
        if skip_allow_last_n < 0:
            raise ValueError("Allow last N inside must be an integer ≥ 0.")

        ranking = (str(mapped.get("ranking", "cut")) or "cut").strip().lower()
        if ranking not in {"cut", "score", "latency"}:
            raise ValueError("Ranking must be one of: cut, score, latency")

        log_comm = bool(mapped.get("log_comm", True))

        w_comm = _safe_float(str(mapped.get("w_comm", "")))
        w_imb = _safe_float(str(mapped.get("w_imb", "")))
        w_tensors = _safe_float(str(mapped.get("w_tensors", "")))
        if w_comm is None or w_imb is None or w_tensors is None:
            raise ValueError("Weights w_comm, w_imb, w_tensors must be numeric.")

        show_pareto_front = bool(mapped.get("show_pareto_front", True))

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

        # Hailo feasibility check (optional)
        hailo_check = bool(self.var_hailo_check.get())
        hailo_hw_arch = (self.var_hailo_hw_arch.get() or "hailo8").strip()
        if hailo_hw_arch not in {"hailo8", "hailo8l", "hailo8r", "hailo10", "hailo10h"}:
            raise ValueError("Hailo HW arch must be one of: hailo8, hailo8l, hailo8r, hailo10, hailo10h")

        hailo_max_checks = _safe_int(self.var_hailo_max_checks.get())
        if hailo_max_checks is None:
            hailo_max_checks = 25
        if hailo_max_checks <= 0:
            raise ValueError("Hailo max checks must be a positive integer.")

        hailo_fixup = bool(self.var_hailo_fixup.get())
        hailo_keep_artifacts = bool(self.var_hailo_keep.get())

        # Default is 'part2' for backwards compatibility if GUI doesn't expose the option.
        hailo_target = (self.var_hailo_target.get() if hasattr(self, 'var_hailo_target') else 'part2')
        hailo_target = (hailo_target or 'part2').strip().lower()
        if hailo_target not in {'part1', 'part2', 'either'}:
            raise ValueError("Hailo target must be one of: part1, part2, either")

        hailo_backend = (self.var_hailo_backend.get() if hasattr(self, 'var_hailo_backend') else 'auto')
        hailo_backend = (hailo_backend or 'auto').strip().lower()
        if hailo_backend not in {'auto', 'local', 'wsl'}:
            raise ValueError("Hailo backend must be one of: auto, local, wsl")

        hailo_wsl_distro = (self.var_hailo_wsl_distro.get() if hasattr(self, 'var_hailo_wsl_distro') else '')
        hailo_wsl_distro = (hailo_wsl_distro or '').strip()
        if not hailo_wsl_distro:
            hailo_wsl_distro = None

        hailo_wsl_venv_activate = (self.var_hailo_wsl_venv.get() if hasattr(self, 'var_hailo_wsl_venv') else '')
        hailo_wsl_venv_activate = (hailo_wsl_venv_activate or '').strip() or "~/hailo_dfc_venv/bin/activate"

        # Keep this reasonably small to avoid 'stuck' GUI sessions if the backend hangs.
        hailo_wsl_timeout_s = 180

        show_top_tensors = _safe_int(str(mapped.get("show_top_tensors", "")))
        if show_top_tensors is None or show_top_tensors < 0:
            raise ValueError("Show top tensors must be an integer ≥ 0.")

        cluster_region_raw = str(mapped.get("cluster_region_ops", "auto") or "auto").strip().lower()
        if not cluster_region_raw or cluster_region_raw == 'auto':
            cluster_region_ops = None
        else:
            cluster_region_ops = _safe_int(cluster_region_raw)
            if cluster_region_ops is None or cluster_region_ops < 0:
                raise ValueError("Region (ops) must be 'auto' or an integer ≥ 0.")

        cluster_mode = str(mapped.get("cluster_mode", "auto") or "auto")
        cluster_mode = (cluster_mode or 'auto').strip().lower()
        if cluster_mode.startswith('auto'):
            cluster_mode = 'auto'
        elif 'semantic' in cluster_mode:
            cluster_mode = 'semantic'
        elif 'uniform' in cluster_mode:
            cluster_mode = 'uniform'
        else:
            cluster_mode = 'auto'

        return Params(
            topk=int(topk),
            min_gap=int(min_gap),
            min_compute_pct=float(min_comp),
            batch_override=batch,
            llm_enable=bool(mapped.get("llm_enable", False)),
            llm_preset=str(mapped.get("llm_preset", "Standard")),
            llm_mode=str(mapped.get("llm_mode", "decode")),
            llm_prefill_len=int(_safe_int(str(mapped.get("llm_prefill_len", "0"))) or 0),
            llm_decode_past_len=int(_safe_int(str(mapped.get("llm_decode_past_len", "0"))) or 0),
            llm_use_ort_symbolic=bool(mapped.get("llm_use_ort_symbolic", True)),
            assume_bpe=bpe,
            unknown_tensor_proxy_mb=float(unknown_mb),
            cluster_best_per_region=bool(mapped.get("cluster_best_per_region", True)),
            cluster_mode=str(cluster_mode),
            cluster_region_ops=cluster_region_ops,

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
            hailo_check=hailo_check,
            hailo_hw_arch=str(hailo_hw_arch),
            hailo_max_checks=int(hailo_max_checks),
            hailo_fixup=hailo_fixup,
            hailo_keep_artifacts=hailo_keep_artifacts,
            hailo_target=str(hailo_target),
            hailo_backend=str(hailo_backend),
            hailo_wsl_distro=hailo_wsl_distro,
            hailo_wsl_venv_activate=str(hailo_wsl_venv_activate),
            hailo_wsl_timeout_s=int(hailo_wsl_timeout_s),
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

    def _analyse_model(self, model_path: str, p: Params, progress_cb: Optional[Callable[[str], None]] = None) -> Dict:
        if progress_cb:
            progress_cb("Loading ONNX model...")
        # IMPORTANT: For large models exported with external tensor data (e.g. LLMs),
        # loading external data into memory can easily consume many GB and crash the GUI.
        # For analysis we only need graph structure + tensor shapes, not the raw weights.
        try:
            model = onnx.load(model_path, load_external_data=False)
        except TypeError:
            # Some older onnx versions do not expose the flag on `onnx.load(...)`.
            # Fall back to parsing the protobuf directly (this does not pull in external weights).
            with open(model_path, "rb") as f:
                model = onnx.ModelProto.FromString(f.read())
        except Exception:
            # If anything goes wrong, re-raise with context.
            raise

        if progress_cb:
            progress_cb("Running ONNX shape inference...")

        # ------------------------------------------------------------
        # LLM shape presets (optional)
        #
        # Two levers:
        #   1) Pre-seed concrete shapes on common I/O tensors (input_ids, attention_mask,
        #      past/present KV caches, logits). This helps both onnx.shape_inference and
        #      ORT's symbolic shape inference.
        #   2) Post-inference: apply dim_param->dim_value overrides across the entire
        #      graph (covers value_info entries that shape inference adds).
        # ------------------------------------------------------------

        # Defaults (used later for comm/shape hints). Populated when LLM preset is enabled.
        llm_enabled = bool(p.llm_enable)
        llm_batch = 1
        llm_seq_len = 1
        llm_past_len = 0
        llm_total_len = 1
        llm_mode = "decode"

        if p.llm_enable:
            try:
                prefill_len = int(p.llm_prefill_len)
                decode_past_len = int(p.llm_decode_past_len)
                mode = p.llm_mode  # 'prefill' or 'decode'

                if mode == "decode":
                    seq_len = 1
                    past_len = decode_past_len
                else:
                    seq_len = prefill_len
                    past_len = 0
                total_len = seq_len + past_len

                # Batch override can be empty in the GUI (None). For LLM shape
                # overrides we always need a concrete batch.
                try:
                    batch_llm = int(p.batch_override) if p.batch_override is not None else 1
                except Exception:
                    batch_llm = 1
                batch_llm = max(1, batch_llm)

                # Persist for later comm estimation.
                llm_batch = batch_llm
                llm_seq_len = seq_len
                llm_past_len = past_len
                llm_total_len = total_len
                llm_mode = str(mode or "decode")

                try:
                    asc.apply_llm_io_shape_overrides(
                        model,
                        batch=batch_llm,
                        seq_len=seq_len,
                        past_len=past_len,
                        total_len=total_len,
                    )
                    logger.info(
                        "[llm] Pre-seeded I/O shapes: mode=%s batch=%d seq_len=%d past_len=%d total_len=%d",
                        mode,
                        batch_llm,
                        seq_len,
                        past_len,
                        total_len,
                    )
                except Exception as e:
                    logger.warning("[llm] Failed to pre-seed I/O shapes (continuing): %s", e)

            except Exception as e:
                llm_enabled = False
                logger.exception("[llm] Failed to compute LLM preset lengths: %s", e)

        model = asc.infer_shapes_safe(model, use_ort_symbolic=bool(llm_enabled and p.llm_use_ort_symbolic))

        llm_dim_overrides: Dict[str, int] = {}
        # Post-inference: apply dim_param overrides to all value_info.
        if llm_enabled:
            try:
                prefill_len = int(p.llm_prefill_len)
                decode_past_len = int(p.llm_decode_past_len)
                mode = llm_mode

                mapping = asc.make_llm_symbolic_dim_overrides(
                    model,
                    batch=llm_batch,
                    prefill_len=prefill_len,
                    decode_past_len=decode_past_len,
                    mode=mode,
                )

                if mapping:
                    llm_dim_overrides = dict(mapping)
                    # Apply post-infer to avoid ORT clearing edits
                    changes2 = asc.apply_dim_param_overrides(model, mapping, only_inputs=False)
                    if changes2:
                        logger.info(
                            "[llm] Applied symbolic dim overrides after shape inference (%d dims updated)",
                            len(changes2),
                        )
                else:
                    logger.warning(
                        "[llm] LLM preset enabled, but no dim_param names could be mapped. Comm estimation may be conservative."
                    )
            except Exception as e:
                logger.exception("[llm] Failed to apply dim overrides after shape inference: %s", e)

        if progress_cb:
            progress_cb("Collecting value infos...")
        vimap = asc.value_info_map(model)
        asc.backfill_quant_shapes(model, vimap, batch=p.batch_override)

        if progress_cb:
            progress_cb("Building graph metadata...")
        nodes, producer_of, consumers_of = asc.build_producers_consumers(model)
        order = asc.topo_sort(nodes, producer_of)

        # Semantic (LLM) labels per boundary (best-effort, used for UI readability).
        semantic_labels_by_boundary = _semantic_labels_for_boundaries(nodes, order)

        if progress_cb:
            progress_cb("Estimating activation bytes and boundary costs...")

        # Many LLM ONNX exports include a lot of tiny Constant-node outputs (INT64
        # scalars, shape helpers, etc.). These are safe to duplicate into both
        # split parts and should not be counted as runtime communication.
        const_values = {
            v
            for v, pi in producer_of.items()
            if 0 <= int(pi) < len(nodes) and getattr(nodes[int(pi)], "op_type", "") == "Constant"
        }

        llm_hints = None
        if llm_enabled:
            llm_hints = {
                "batch": int(llm_batch),
                "seq_len": int(llm_seq_len),
                "past_len": int(llm_past_len),
                "total_len": int(llm_total_len),
            }

        # Best-effort: infer the model hidden size from *weight* tensor shapes.
        # This is useful when shape inference does not annotate intermediate activations
        # but the 2D weight matrices still expose the hidden dimension.
        hidden_size_hint = None
        if llm_enabled:
            try:
                from collections import Counter

                dims = []
                for init in getattr(model.graph, "initializer", []):
                    try:
                        dlist = list(getattr(init, "dims", []))
                    except Exception:
                        continue
                    if len(dlist) != 2:
                        continue
                    for d in dlist:
                        try:
                            di = int(d)
                        except Exception:
                            continue
                        # Hidden sizes are typically in the low-thousands; filter out vocab (huge)
                        # and tiny scalars.
                        if 256 <= di <= 16384:
                            dims.append(di)
                if dims:
                    dim, cnt = Counter(dims).most_common(1)[0]
                    # Require that the dimension shows up often, otherwise it could be an
                    # intermediate size for a single layer.
                    if cnt >= 16:
                        hidden_size_hint = int(dim)
            except Exception:
                hidden_size_hint = None

        batch_for_sizes = p.batch_override
        if llm_enabled and (batch_for_sizes is None or batch_for_sizes <= 0):
            batch_for_sizes = int(llm_batch)

        value_bytes_raw = asc.compute_tensor_bytes_per_value(
            vimap,
            batch_for_sizes,
            p.assume_bpe,
            llm_hints=llm_hints,
            hidden_size_hint=hidden_size_hint,
        )
        # Filter constants from comm + tensor-count metrics.
        value_bytes = {k: (0 if k in const_values else int(b)) for k, b in value_bytes_raw.items()}

        costs_bytes, val_span = asc.boundary_costs(order, producer_of, consumers_of, value_bytes)

        # Peak activation memory per boundary (approx)
        # Derived from value spans (producer -> last consumer) via Comm(b) live-set bytes.
        peak_l, peak_r, peak_max = asc.peak_activation_memory_per_boundary(costs_bytes)


        # Crossing tensor counts: known sizes vs all (unknown sizes enabled via value_bytes_all)
        counts_known = asc.boundary_tensor_counts(order, producer_of, consumers_of, value_bytes)
        value_bytes_all = {k: (0 if k in const_values else 1) for k in producer_of.keys()}
        counts_all = asc.boundary_tensor_counts(order, producer_of, consumers_of, value_bytes_all)
        value_bytes_const = {k: (1 if k in const_values else 0) for k in producer_of.keys()}
        counts_const = asc.boundary_tensor_counts(order, producer_of, consumers_of, value_bytes_const)
        unknown_counts = [max(0, int(a) - int(k)) for a, k in zip(counts_all, counts_known)]

        # Optional: pessimistically account for tensors whose shape cannot be inferred.
        # Without this, the communication cost becomes a *lower bound* and the optimiser
        # may incorrectly prefer "dirty" cuts that look cheap only because shape
        # inference failed.
        costs_bytes_lb = list(costs_bytes)

        # DType-aware proxy bytes for unknown-size tensors (conservative but less misleading):
        #   - FLOAT16/FLOAT32/... : unknown_tensor_proxy_mb (default 2 MiB)
        #   - INT/BOOL-like       : 64 KiB (hard default; these are usually masks/ids)
        unknown_float_proxy_mb = float(getattr(p, "unknown_tensor_proxy_mb", 0.0) or 0.0)
        unknown_int_proxy_kb = 64.0
        if (unknown_float_proxy_mb > 0 or unknown_int_proxy_kb > 0) and unknown_counts:
            try:
                M = max(0, len(order) - 1)
                delta = [0.0] * (M + 1)
                pos_of = {n: i for i, n in enumerate(order)}

                float_proxy_b = float(unknown_float_proxy_mb) * 1024.0 * 1024.0
                int_proxy_b = float(unknown_int_proxy_kb) * 1024.0

                int_like = {
                    onnx.TensorProto.BOOL,
                    onnx.TensorProto.INT8,
                    onnx.TensorProto.UINT8,
                    onnx.TensorProto.INT16,
                    onnx.TensorProto.UINT16,
                    onnx.TensorProto.INT32,
                    onnx.TensorProto.UINT32,
                    onnx.TensorProto.INT64,
                    onnx.TensorProto.UINT64,
                }

                for v, prod in producer_of.items():
                    if v in const_values:
                        continue
                    if value_bytes.get(v, 0) > 0:
                        continue  # size known
                    cons = consumers_of.get(v, [])
                    if not cons:
                        continue
                    p_pos = pos_of.get(prod, None)
                    if p_pos is None:
                        continue
                    cons_pos = [pos_of[c] for c in cons if c in pos_of]
                    if not cons_pos:
                        continue
                    last = max(cons_pos)
                    if last <= p_pos:
                        continue

                    et = asc.elemtype_from_vi(vimap.get(v))
                    pb = int_proxy_b if et in int_like else float_proxy_b
                    if pb <= 0:
                        continue

                    if p_pos < len(delta):
                        delta[p_pos] += pb
                    if last < len(delta):
                        delta[last] -= pb

                running = 0.0
                proxy_bytes_by_boundary: List[float] = []
                for b in range(M):
                    running += delta[b]
                    proxy_bytes_by_boundary.append(running)

                if proxy_bytes_by_boundary:
                    costs_bytes = [
                        int(cb) + int(pb) for cb, pb in zip(costs_bytes, proxy_bytes_by_boundary)
                    ]
            except Exception as e:
                logger.warning("[proxy] Failed to compute dtype-aware unknown-size proxy: %s", e)

        peak_l, peak_r, peak_max = asc.peak_activation_memory_per_boundary(costs_bytes)


# Spans for ALL crossing values (including unknown sizes). Useful for Nordstern.
        _, val_span_all = asc.boundary_costs(order, producer_of, consumers_of, value_bytes_all)

        # Coverage of produced tensors that have an inferred size
        produced = list(producer_of.keys())
        produced_act = [v for v in produced if v not in const_values]
        known_produced = sum(1 for v in produced_act if value_bytes.get(v, 0) > 0)
        coverage = (float(known_produced) / float(len(produced_act))) if produced_act else 1.0

        # Memory components per boundary
        init_spans = precompute_initializer_spans(model, nodes, order)
        weights_left_b, weights_right_b = weights_for_all_boundaries(init_spans, len(costs_bytes))

        kv_per_layer = kv_cache_bytes_per_layer(model, vimap, llm_hints if llm_enabled else None) if llm_enabled else {}
        kv_left_b: List[int] = []
        kv_right_b: List[int] = []
        for bb in range(len(costs_bytes)):
            split_idx = layer_split_index_for_boundary(nodes, order, bb)
            kl, kr = kv_for_boundary(kv_per_layer, split_idx)
            kv_left_b.append(int(kl))
            kv_right_b.append(int(kr))

        # FLOPs per node
        if progress_cb:
            progress_cb("Estimating per-node FLOPs...")
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
            if progress_cb:
                progress_cb("Checking strict boundary feasibility...")
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
            "semantic_labels_by_boundary": semantic_labels_by_boundary,
            "vimap": vimap,
            "value_bytes": value_bytes,
            "costs_bytes": costs_bytes,
            "costs_bytes_lb": costs_bytes_lb,
            "peak_act_mem_left_bytes": peak_l,
            "peak_act_mem_right_bytes": peak_r,
            "peak_act_mem_max_bytes": peak_max,
            "weights_left_bytes": weights_left_b,
            "weights_right_bytes": weights_right_b,
            "kv_left_bytes": kv_left_b,
            "kv_right_bytes": kv_right_b,
            "kv_per_layer": kv_per_layer,
            "val_span": val_span,
            "val_span_all": val_span_all,
            "crossing_counts_known": counts_known,
            "crossing_counts_all": counts_all,
            "crossing_counts_const": counts_const,
            "const_value_count": len(const_values),
            "unknown_crossing_counts": unknown_counts,
            "unknown_tensor_proxy_mb": float(p.unknown_tensor_proxy_mb) if p.unknown_tensor_proxy_mb else 0.0,
            "unknown_tensor_proxy_kb_int": float(unknown_int_proxy_kb),
            "llm_dim_overrides": llm_dim_overrides,
            "llm_hints": llm_hints,
            "llm_mode": llm_mode,
            "hidden_size_hint": hidden_size_hint,
            "shape_coverage": coverage,
            "known_produced": known_produced,
            "total_produced": len(produced_act),
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

    def _select_picks(self, a: Dict, p: Params, progress_cb: Optional[Callable[[str], None]] = None) -> List[int]:
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

        
        raw_candidates = list(candidates)

        # Candidate clustering (diversity): keep only the best-scoring candidate per region/bin.
        #
        # For CNN-like graphs we use uniform op windows (legacy behaviour).
        # For transformer/LLM graphs we can cluster semantically by "layer/block transitions",
        # which removes the "dirty splits" inside attention/MLP blocks.
        if getattr(p, "cluster_best_per_region", False) and candidates:
            cluster_mode = str(getattr(p, "cluster_mode", "auto") or "auto").strip().lower()
            if (not cluster_mode) or cluster_mode.startswith("auto"):
                cluster_mode = "semantic" if bool(getattr(p, "llm_enable", False)) else "uniform"
            elif "sem" in cluster_mode:
                cluster_mode = "semantic"
            elif "uni" in cluster_mode:
                cluster_mode = "uniform"
            else:
                cluster_mode = "uniform"

            # --- Semantic clustering (LLMs): best split per layer transition ---
            if cluster_mode == "semantic":
                n_pos = int(len(order))
                # Map each topo position to a semantic group label (or None).
                pos_group = []
                pos_is_const = []
                for node_idx in order:
                    n = nodes[int(node_idx)]
                    pos_group.append(_semantic_group_for_node(n))
                    pos_is_const.append(str(getattr(n, "op_type", "")) == "Constant")

                # Nearest "real" group to the left/right (skip constants and unknown groups).
                prev_group: List[Optional[str]] = [None] * n_pos
                last: Optional[str] = None
                for i in range(n_pos):
                    g = pos_group[i]
                    if (not pos_is_const[i]) and g:
                        last = g
                    prev_group[i] = last
                next_group: List[Optional[str]] = [None] * n_pos
                last = None
                for i in range(n_pos - 1, -1, -1):
                    g = pos_group[i]
                    if (not pos_is_const[i]) and g:
                        last = g
                    next_group[i] = last

                sem_key_by_b: List[Optional[str]] = [None] * M
                for b in range(M):
                    lg = prev_group[b] if 0 <= b < n_pos else None
                    rg = next_group[b + 1] if 0 <= (b + 1) < n_pos else None
                    if lg and rg and lg != rg:
                        sem_key_by_b[b] = f"{lg}->{rg}"

                used_keys = set()
                clustered = []
                for b in candidates:
                    key = sem_key_by_b[int(b)] if 0 <= int(b) < len(sem_key_by_b) else None
                    if not key:
                        continue
                    if key in used_keys:
                        continue
                    used_keys.add(key)
                    clustered.append(b)

                if clustered:
                    logger.info(
                        "[cluster] semantic layers: %d -> %d candidates (%d unique transitions)",
                        len(candidates),
                        len(clustered),
                        len(used_keys),
                    )
                    candidates = clustered
                else:
                    # If we couldn't extract any semantic transitions (non-transformer graph,
                    # or model exported without layer names), fall back to uniform regions.
                    logger.info(
                        "[cluster] semantic layers: no transitions detected; falling back to uniform regions",
                    )
                    cluster_mode = "uniform"

            # --- Uniform clustering: best split per fixed op window ---
            if cluster_mode == "uniform":
                region_ops_cfg = getattr(p, "cluster_region_ops", None)
                region_ops = int(region_ops_cfg) if region_ops_cfg is not None else 0
                if region_ops <= 0:
                    # Auto: choose a region/bin size proportional to the model's graph size.
                    #
                    # We intentionally make this *independent* of Top-K to keep the clustering
                    # stable when the user changes the display depth. For large graphs (LLMs)
                    # this typically yields ~2% of the topo-op count per region (≈ 50 regions).
                    n_ops = max(1, int(len(order)))
                    region_ops = int(round(n_ops / 50.0))  # ~2% of model size
                    # Clamp to avoid degenerate behavior on tiny/huge graphs.
                    region_ops = max(5, min(250, region_ops))
                    logger.info(
                        "[cluster] auto region_ops=%d (n_ops=%d, ~%.2f%%)",
                        region_ops,
                        n_ops,
                        100.0 * float(region_ops) / float(n_ops),
                    )
                used_regions = set()
                clustered = []
                for b in candidates:
                    rid = b // region_ops
                    if rid in used_regions:
                        continue
                    used_regions.add(rid)
                    clustered.append(b)
                if len(clustered) != len(candidates):
                    logger.info(
                        "[cluster] best-per-region: %d -> %d candidates (region_ops=%d)",
                        len(candidates),
                        len(clustered),
                        region_ops,
                    )
                candidates = clustered

# Non-maximum suppression by boundary index
        # Optionally inject a Hailo feasibility check into the pick selection.
        picks: List[int] = []

        hailo_enabled = bool(getattr(p, "hailo_check", False))
        hailo_results: Dict[int, Dict[str, Any]] = {}
        hailo_summary: Dict[str, Any] = {
            "enabled": hailo_enabled,
            "backend": str(getattr(p, "hailo_backend", "auto")),
            "hw_arch": getattr(p, "hailo_hw_arch", None),
            "max_checks": getattr(p, "hailo_max_checks", None),
            "fixup": bool(getattr(p, "hailo_fixup", True)),
            "keep_artifacts": bool(getattr(p, "hailo_keep_artifacts", False)),
            "target": str(getattr(p, "hailo_target", "part2")),
            "wsl_distro": getattr(p, "hailo_wsl_distro", None),
            "wsl_venv_activate": getattr(p, "hailo_wsl_venv_activate", None),
            "wsl_timeout_s": getattr(p, "hailo_wsl_timeout_s", None),
        }

        hailo_work_root: Optional[Path] = None
        if hailo_enabled:
            from .hailo_backend import hailo_sdk_available, hailo_wsl_available

            if progress_cb:
                progress_cb("Hailo check enabled: verifying environment…")

            backend = str(getattr(p, "hailo_backend", "auto") or "auto").strip().lower()
            if backend not in {"auto", "local", "wsl"}:
                backend = "auto"

            have_local = bool(hailo_sdk_available())
            have_wsl = bool(hailo_wsl_available())

            if backend == "local" and not have_local:
                raise RuntimeError(
                    "Hailo check backend is set to 'local', but hailo_sdk_client is not importable in this Python env. "
                    "Either install the Hailo DFC/SDK into this environment, or switch the backend to 'wsl'."
                )
            if backend == "wsl" and not have_wsl:
                raise RuntimeError(
                    "Hailo check backend is set to 'wsl', but WSL is not available (wsl.exe not found). "
                    "Install WSL2 and a Linux distro, or switch backend to 'local'."
                )
            if backend == "auto" and not (have_local or have_wsl):
                raise RuntimeError(
                    "Hailo check is enabled, but no backend is available. "
                    "Install hailo_sdk_client in this Python environment, or configure the WSL backend on Windows."
                )

            if bool(getattr(p, "hailo_keep_artifacts", False)) and getattr(self, "model_path", None):
                md = Path(str(self.model_path)).resolve().parent
                hailo_work_root = md / f"hailo_check_{Path(str(self.model_path)).stem}"
                hailo_work_root.mkdir(parents=True, exist_ok=True)
            else:
                hailo_work_root = Path(tempfile.mkdtemp(prefix="hailo_check_"))

            hailo_summary["workdir"] = str(hailo_work_root)

        checks_budget = int(getattr(p, "hailo_max_checks", 0) or 0) if hailo_enabled else 0
        checks_done = 0
        checks_passed = 0
        checks_failed = 0

        for b in candidates:
            if len(picks) >= p.topk:
                break

            if not all(abs(b - s) > p.min_gap for s in picks):
                continue

            if hailo_enabled:
                if checks_done >= checks_budget:
                    hailo_summary["budget_exhausted"] = True
                    break

                checks_done += 1
                if progress_cb:
                    progress_cb(f"Hailo parse-check {checks_done}/{checks_budget}: boundary b={b}")

                ok, info = self._hailo_parse_check_boundary(a, int(b), p, hailo_work_root)
                hailo_results[int(b)] = info
                if not ok:
                    checks_failed += 1
                    continue
                checks_passed += 1

            picks.append(int(b))

        # Attach Hailo info to analysis (useful for exports / reproducibility)
        if hailo_enabled:
            hailo_summary.update(
                {
                    "checked": int(checks_done),
                    "passed": int(checks_passed),
                    "failed": int(checks_failed),
                    "picks": int(len(picks)),
                }
            )
            a["hailo_check"] = hailo_summary
            a["hailo_check_results"] = hailo_results

            # If we used a temporary workdir, clean it up.
            if hailo_work_root is not None and not bool(getattr(p, "hailo_keep_artifacts", False)):
                try:
                    shutil.rmtree(hailo_work_root, ignore_errors=True)
                    a["hailo_check_temp_cleaned"] = True
                except Exception:
                    a["hailo_check_temp_cleaned"] = False

        # Store for plots/export
        a["candidate_bounds_all"] = list(raw_candidates)
        a["candidate_bounds"] = list(candidates)
        a["scores"] = scores
        a["latency_ms_dict"] = latency_ms

        return picks

    def _hailo_parse_check_boundary(self, a: Dict, boundary: int, p: Params, work_root: Optional[Path]) -> Tuple[bool, Dict[str, Any]]:
        """Run a Hailo translate (parse-only) feasibility check for a boundary.

        Depending on `p.hailo_target`, this checks:
          - Part2 only (suffix)
          - Part1 only (prefix)
          - either (Part1 OR Part2)

        The intention is a *hard feasibility filter* during pick selection.
        """

        from .hailo_backend import hailo_parse_check_auto

        model: Optional[onnx.ModelProto] = a.get("model")
        nodes = a.get("nodes") or []
        order = a.get("order") or []

        if model is None:
            return False, {"ok": False, "error": "No model loaded in analysis."}

        b = int(boundary)
        policy = str(getattr(p, "hailo_target", "part2") or "part2").strip().lower()
        if policy not in {"part1", "part2", "either"}:
            policy = "part2"

        cut_tensors = asc.cut_tensors_for_boundary(order, nodes, b)
        if not cut_tensors:
            return False, {"ok": False, "error": "No cut tensors for boundary.", "b": b, "policy": policy}

        # Materialize artifacts on disk for the Hailo translator.
        if work_root is None:
            work_root = Path(tempfile.mkdtemp(prefix="hailo_check_"))

        case_dir = Path(work_root) / f"b{b:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        g = model.graph
        init_names = {i.name for i in g.initializer}
        orig_inputs = [vi.name for vi in g.input if vi.name not in init_names]
        orig_outputs = [o.name for o in g.output]

        info: Dict[str, Any] = {
            "ok": False,
            "b": b,
            "policy": policy,
            "cut_tensors": list(cut_tensors),
        }

        def _run_one(which: str, submodel: onnx.ModelProto, onnx_path: Path) -> Dict[str, Any]:
            keep = bool(getattr(p, "hailo_keep_artifacts", False))
            backend_req = str(getattr(p, "hailo_backend", "auto"))
            hw_arch = str(getattr(p, "hailo_hw_arch", "hailo8"))
            fixup = bool(getattr(p, "hailo_fixup", True))

            # Persistent cache (best effort): only used when we don't need
            # artifacts on disk. This can dramatically speed up repeated runs
            # and avoids repeated WSL/DFC startup overhead.
            cache_key: Optional[str] = None
            if not keep:
                try:
                    backend_eff = backend_req
                    if backend_req == "auto":
                        from .hailo_backend import hailo_sdk_available

                        backend_eff = "local" if hailo_sdk_available() else "wsl"
                    sha1 = self._sha1_bytes(submodel.SerializeToString())
                    cache_key = f"{backend_eff}|{hw_arch}|fixup={int(fixup)}|{which}|sha1={sha1}"
                    cached = self._hailo_cache_get(cache_key)
                    if cached is not None:
                        out = dict(cached)
                        out["cached"] = True
                        out["cache_key"] = cache_key
                        return out
                except Exception:
                    cache_key = None

            try:
                asc.save_model(submodel, str(onnx_path), external_data=False)
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Failed to save {which} ONNX: {type(e).__name__}: {e}",
                }

            res = hailo_parse_check_auto(
                onnx_path,
                backend=backend_req,
                hw_arch=hw_arch,
                net_name=f"{which}_b{b}",
                outdir=onnx_path.parent if keep else None,
                fixup=fixup,
                save_har=keep,
                wsl_distro=getattr(p, "hailo_wsl_distro", None),
                wsl_venv_activate=str(getattr(p, "hailo_wsl_venv_activate", "~/hailo_dfc_venv/bin/activate")),
                wsl_timeout_s=int(getattr(p, "hailo_wsl_timeout_s", 180)),
            )

            out: Dict[str, Any] = {
                "ok": bool(res.ok),
                "elapsed_s": float(res.elapsed_s),
                "hw_arch": str(res.hw_arch),
                "net_name": str(res.net_name),
                "backend": str(res.backend) if getattr(res, "backend", None) else None,
                "error": res.error,
                "fixup_report": res.fixup_report,
            }

            if bool(getattr(p, "hailo_keep_artifacts", False)):
                out["work_dir"] = str(onnx_path.parent)
                out["onnx_path"] = str(onnx_path)
                out["har_path"] = res.har_path
                out["fixed_onnx_path"] = res.fixed_onnx_path

            if cache_key is not None and not keep:
                try:
                    cache_payload = dict(out)
                    cache_payload["ts"] = float(time.time())
                    cache_payload.pop("cached", None)
                    self._hailo_cache_put(cache_key, cache_payload)
                    self._hailo_cache_dirty = True
                except Exception:
                    pass

            return out

        # Decide which parts to check.
        check_part1 = policy in {"part1", "either"}
        check_part2 = policy in {"part2", "either"}

        part1_res: Optional[Dict[str, Any]] = None
        part2_res: Optional[Dict[str, Any]] = None

        # By default, check Part2 first for backwards compatibility, then Part1 if needed.
        # This keeps runtime similar to the older "Part2-only" behavior in the common case.
        check_order = ["part2", "part1"]

        def _try_part2() -> Dict[str, Any]:
            # If Hailo is the *consumer* side (Part2 on Hailo), non-strict boundaries become
            # multi-input on Hailo and are much harder to integrate. Keep this as a hard fail.
            extras = asc.strict_boundary_extras(model, cut_tensors)
            if extras:
                return {
                    "ok": False,
                    "error": f"Non-strict boundary: Part2 requires extra inputs: {extras}",
                    "strict_extras": list(extras),
                }

            try:
                p2_model, ext_inputs = asc.build_submodel(
                    model,
                    outputs=orig_outputs,
                    stop_names=set(cut_tensors) | set(orig_inputs),
                    model_name=f"part2_b{b}",
                    force_inputs=list(cut_tensors),
                )
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Failed to build Part2 submodel: {type(e).__name__}: {e}",
                }

            ext_only = [x for x in ext_inputs if x not in cut_tensors]
            if ext_only:
                return {
                    "ok": False,
                    "error": f"Part2 still has external inputs beyond cut tensors: {ext_only}",
                    "external_inputs": list(ext_inputs),
                }

            p2_dir = case_dir / "part2"
            p2_dir.mkdir(parents=True, exist_ok=True)
            return _run_one("part2", p2_model, p2_dir / "part2.onnx")

        def _try_part1() -> Dict[str, Any]:
            try:
                p1_model, ext_inputs = asc.build_submodel(
                    model,
                    outputs=list(cut_tensors),
                    stop_names=set(orig_inputs),
                    model_name=f"part1_b{b}",
                )
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"Failed to build Part1 submodel: {type(e).__name__}: {e}",
                }

            ext_only = [x for x in ext_inputs if x not in orig_inputs]
            if ext_only:
                return {
                    "ok": False,
                    "error": f"Unexpected external inputs for Part1: {ext_only}",
                    "external_inputs": list(ext_inputs),
                }

            p1_dir = case_dir / "part1"
            p1_dir.mkdir(parents=True, exist_ok=True)
            return _run_one("part1", p1_model, p1_dir / "part1.onnx")

        for which in check_order:
            if which == "part2" and check_part2 and part2_res is None:
                part2_res = _try_part2()
                info["part2"] = part2_res
                if policy != "either":
                    # policy is part2-only; no need to check others
                    break
                if bool(part2_res.get("ok")):
                    # In 'either' mode, accept early to keep runtime small.
                    break

            if which == "part1" and check_part1 and part1_res is None:
                part1_res = _try_part1()
                info["part1"] = part1_res
                if policy != "either":
                    break
                if bool(part1_res.get("ok")):
                    break

        ok_part1 = bool(part1_res and part1_res.get("ok"))
        ok_part2 = bool(part2_res and part2_res.get("ok"))

        ok = False
        accepted_by = None
        if policy == "part1":
            ok = ok_part1
            accepted_by = "part1" if ok else None
        elif policy == "part2":
            ok = ok_part2
            accepted_by = "part2" if ok else None
        else:
            ok = ok_part1 or ok_part2
            # Prefer Part2 for backwards compatibility, otherwise Part1.
            if ok_part2:
                accepted_by = "part2"
            elif ok_part1:
                accepted_by = "part1"

        info["ok"] = bool(ok)
        info["accepted_by"] = accepted_by
        info["ok_part1"] = ok_part1
        info["ok_part2"] = ok_part2

        return bool(ok), info

    # ----------------------------- Diagnostics UI -----------------------------

    def _update_diagnostics(self, a: Dict) -> None:
        if self.analysis_result is None:
            self.analysis_result = AnalysisResult()
        cov = float(a.get("shape_coverage", 1.0))
        self._update_dtype_proxy_visibility(cov)
        kp = int(a.get("known_produced", 0))
        tp = int(a.get("total_produced", 0))
        max_unk = int(a.get("max_unknown_crossing", 0))

        self.var_shape_coverage.set(f"{kp}/{tp} ({100.0*cov:.1f}%)")
        self.var_unknown_crossing.set(str(max_unk))

        if cov < 0.999 or max_unk > 0:
            proxy_mb = float(a.get("unknown_tensor_proxy_mb", 0.0) or 0.0)
            proxy_kb_int = float(a.get("unknown_tensor_proxy_kb_int", 0.0) or 0.0)
            if max_unk > 0 and proxy_mb > 0:
                if proxy_kb_int > 0:
                    self.var_diag_note.set(
                        f"Comm(b) includes unknown proxy (float {proxy_mb:g} MB, int/bool {proxy_kb_int:g} KB)"
                    )
                else:
                    self.var_diag_note.set(f"Comm(b) includes unknown proxy ({proxy_mb:g} MB/tensor)")
            else:
                self.var_diag_note.set("Comm(b) may be underestimated (lower bound)")
        else:
            self.var_diag_note.set("")
        self.var_memf_left_text.set("Left: n/a")
        self.var_memf_right_text.set("Right: n/a")
        self._set_ui_state(self._infer_ui_state())

        # LLM preset info (helps interpret comm/shape numbers).
        llm_hints = a.get("llm_hints")
        if isinstance(llm_hints, dict) and llm_hints:
            self.var_memf_include_kv.set(True)
            llm_mode = str(a.get("llm_mode") or "")
            b = llm_hints.get("batch")
            s = llm_hints.get("seq_len")
            past = llm_hints.get("past_len")
            total = llm_hints.get("total_len")
            hidden_hint = a.get("hidden_size_hint")
            mode_note = " (per-token)" if llm_mode == "decode" else ""
            msg = f"LLM effective shapes: mode={llm_mode}{mode_note} batch={b} seq_len={s} past_len={past} total_len={total}"
            if hidden_hint:
                msg += f" (hidden≈{hidden_hint})"
            self.var_llm_info.set(msg)
        else:
            self.var_llm_info.set("")
            self.var_memf_include_kv.set(False)

        self.analysis_result.diagnostics = {
            "shape_coverage": self.var_shape_coverage.get(),
            "unknown_crossing": self.var_unknown_crossing.get(),
            "diag_note": self.var_diag_note.get(),
            "llm_info": self.var_llm_info.get(),
        }

    def _accel_by_name(self, name: str) -> Dict[str, Any]:
        for a in (self.accel_specs.get("accelerators") or []):
            if str(a.get("name")) == str(name):
                return a
        return {}

    def _refresh_memory_forecast(self) -> None:
        a = self.analysis or {}
        if not self.memory_by_boundary:
            self.var_memf_left_text.set("Left: n/a")
            self.var_memf_right_text.set("Right: n/a")
            self.pb_mem_left.configure(value=0)
            self.pb_mem_right.configure(value=0)
            return

        b = self._selected_boundary_index()
        if b is None and self.current_picks:
            b = int(self.current_picks[0])
        if b is None:
            return

        left_acc = self._accel_by_name(self.var_memf_left_accel.get())
        right_acc = self._accel_by_name(self.var_memf_right_accel.get())
        m = self.memory_by_boundary.get(int(b)) or {}
        if not left_acc or not right_acc or not m:
            return

        limit_l = float(left_acc.get("ram_limit_mb") or 0.0)
        limit_r = float(right_acc.get("ram_limit_mb") or 0.0)
        tot_l_mb = float(m.get("left", {}).get("total_mb") or 0.0)
        tot_r_mb = float(m.get("right", {}).get("total_mb") or 0.0)
        util_l = (100.0 * tot_l_mb / limit_l) if limit_l > 0 else 0.0
        util_r = (100.0 * tot_r_mb / limit_r) if limit_r > 0 else 0.0

        self.pb_mem_left.configure(value=max(0.0, min(100.0, util_l)), style=("MemRed.Horizontal.TProgressbar" if util_l > 100.0 else "MemGreen.Horizontal.TProgressbar"))
        self.pb_mem_right.configure(value=max(0.0, min(100.0, util_r)), style=("MemRed.Horizontal.TProgressbar" if util_r > 100.0 else "MemGreen.Horizontal.TProgressbar"))
        self.var_memf_left_text.set(f"Left: {tot_l_mb/1024.0:.2f} / {limit_l/1024.0:.2f} GB ({util_l:.1f}%)")
        self.var_memf_right_text.set(f"Right: {tot_r_mb/1024.0:.2f} / {limit_r/1024.0:.2f} GB ({util_r:.1f}%)")

        if hasattr(self, "_last_params") and self._last_params is not None:
            picks = list(self.analysis_result.candidates if self.analysis_result else self.current_picks)
            if bool(self.var_memf_filter_fit.get()):
                picks = [x for x in picks if (self.memory_by_boundary.get(int(x), {}).get("left", {}).get("fits") and self.memory_by_boundary.get(int(x), {}).get("right", {}).get("fits"))]
            self._update_table(a, picks, self._last_params)


    def _get_memory_stats_for_boundary(self, boundary, left_accel_name=None, right_accel_name=None):
        """Compute RAM usage for a given split boundary.

        Returns a dict shaped for `gui.widgets.memory_fit.MemoryFitWidget`:
            {"left": {...}, "right": {...}}

        This is used by the refactored GUI (Analyse tab -> Candidate Inspector).
        """
        try:
            b = int(boundary)
        except Exception:
            return {"left": {}, "right": {}}

        # Resolve accelerator names (caller -> HW tab -> legacy memf vars)
        if left_accel_name is None:
            left_accel_name = left_accel or ""
            if not left_accel_name:
                for attr in ("var_hw_left_accel", "var_memf_left_accel"):
                    try:
                        v = getattr(self, attr, None)
                        if v is not None:
                            left_accel_name = v.get()
                            break
                    except Exception:
                        pass
        if right_accel_name is None:
            right_accel_name = right_accel or ""
            if not right_accel_name:
                for attr in ("var_hw_right_accel", "var_memf_right_accel"):
                    try:
                        v = getattr(self, attr, None)
                        if v is not None:
                            right_accel_name = v.get()
                            break
                    except Exception:
                        pass

        left_accel_name = (left_accel_name or "").strip()
        right_accel_name = (right_accel_name or "").strip()

        left_accel = self._accel_by_name(left_accel_name) if left_accel_name else {}
        right_accel = self._accel_by_name(right_accel_name) if right_accel_name else {}

        def _to_float(x, default=0.0):
            try:
                if x is None:
                    return default
                return float(x)
            except Exception:
                return default

        # Device limits / overheads (in MiB in the UI spec)
        limit_l_mb = _to_float(left_accel.get("ram_limit_mb", 0.0), 0.0)
        limit_r_mb = _to_float(right_accel.get("ram_limit_mb", 0.0), 0.0)
        overhead_l_mb = _to_float(left_accel.get("runtime_overhead_mb", 0.0), 0.0)
        overhead_r_mb = _to_float(right_accel.get("runtime_overhead_mb", 0.0), 0.0)

        # Memory-fit options
        pol = "max_peak_or_comm"
        include_kv = False
        include_comm = True
        try:
            pol = self.var_memf_policy.get()
        except Exception:
            pass
        try:
            include_kv = bool(self.var_memf_include_kv.get())
        except Exception:
            pass
        try:
            include_comm = bool(self.var_memf_include_comm.get())
        except Exception:
            pass

        plot_data = None
        try:
            if getattr(self, "analysis_result", None) is not None:
                plot_data = getattr(self.analysis_result, "plot_data", None)
        except Exception:
            plot_data = None
        if plot_data is None:
            plot_data = getattr(self, "_last_plot_data", None)

        # plot_data is a wrapper in the refactored GUI; for Memory Fit we need the raw
        # analysis dict that contains per-boundary arrays (weights/peak-mem/comm bytes).
        data = plot_data
        if isinstance(plot_data, dict):
            _a = plot_data.get("analysis")
            if isinstance(_a, dict):
                data = _a

        MiB = 1024.0 * 1024.0

        def _at(arr, idx, default=0.0):
            try:
                return float(arr[idx])
            except Exception:
                return default

        # Defaults
        weights_l_b = weights_r_b = 0.0
        peak_l_b = peak_r_b = 0.0
        kv_l_b = kv_r_b = 0.0
        comm_b = 0.0
        total_l_mb = total_r_mb = 0.0

        # ------------------------------------------------------------------
        # Prefer cached per-boundary estimates (created during table render).
        # They already contain device-aware totals in MB.
        # ------------------------------------------------------------------
        est = None
        try:
            if (
                getattr(self, "analysis_result", None) is not None
                and isinstance(getattr(self.analysis_result, "memory_estimate", None), dict)
            ):
                est = self.analysis_result.memory_estimate.get(b)
                if est is None:
                    est = self.analysis_result.memory_estimate.get(str(b))
            if est is None and isinstance(getattr(self, "memory_by_boundary", None), dict):
                est = self.memory_by_boundary.get(b)
                if est is None:
                    est = self.memory_by_boundary.get(str(b))
        except Exception:
            est = None

        if isinstance(est, dict):
            try:
                le = est.get("left", {})
                re = est.get("right", {})
                if isinstance(le, dict) and isinstance(re, dict):
                    total_l_mb = _to_float(le.get("total_mb", 0.0), 0.0)
                    total_r_mb = _to_float(re.get("total_mb", 0.0), 0.0)

                    # Optional breakdowns (stored in MB). Convert to bytes for the
                    # below dict that expects *_b values.
                    weights_l_b = _to_float(le.get("weights_mb", 0.0), 0.0) * MiB
                    weights_r_b = _to_float(re.get("weights_mb", 0.0), 0.0) * MiB
                    peak_l_b = _to_float(le.get("peak_activations_mb", le.get("peak_act_mb", 0.0)), 0.0) * MiB
                    peak_r_b = _to_float(re.get("peak_activations_mb", re.get("peak_act_mb", 0.0)), 0.0) * MiB
                    kv_l_b = _to_float(le.get("kv_mb", le.get("kv_cache_mb", 0.0)), 0.0) * MiB
                    kv_r_b = _to_float(re.get("kv_mb", re.get("kv_cache_mb", 0.0)), 0.0) * MiB
                    comm_b = _to_float(le.get("comm_mb", re.get("comm_mb", 0.0)), 0.0) * MiB
            except Exception:
                pass

        # ------------------------------------------------------------------
        # If we don't have cached totals (or they are 0), compute them from the
        # raw analysis arrays in plot_data['analysis'].
        # ------------------------------------------------------------------
        has_arrays = isinstance(data, dict) and any(
            k in data
            for k in (
                "weights_left_bytes",
                "weights_right_bytes",
                "peak_act_mem_left_bytes",
                "peak_act_mem_right_bytes",
            )
        )

        if (total_l_mb <= 0.0 and total_r_mb <= 0.0) and has_arrays:
            weights_l_b = _at(data.get("weights_left_bytes", []), b, 0.0)
            weights_r_b = _at(data.get("weights_right_bytes", []), b, 0.0)
            peak_l_b = _at(data.get("peak_act_mem_left_bytes", []), b, 0.0)
            peak_r_b = _at(data.get("peak_act_mem_right_bytes", []), b, 0.0)

            if include_kv:
                kv_l_b = _at(data.get("kv_left_bytes", []), b, 0.0)
                kv_r_b = _at(data.get("kv_right_bytes", []), b, 0.0)

            if include_comm:
                costs = None
                if isinstance(data.get("crossing_cost_bytes_by_boundary"), (list, tuple)):
                    costs = data.get("crossing_cost_bytes_by_boundary")
                elif isinstance(plot_data, dict):
                    costs = plot_data.get("costs_bytes", [])
                comm_b = _at(costs or [], b, 0.0)
            else:
                comm_b = 0.0

            overhead_l_b = overhead_l_mb * MiB
            overhead_r_b = overhead_r_mb * MiB

            try:
                mem_l_b = estimate_ram_bytes(weights_l_b, peak_l_b, kv_l_b, overhead_l_b, comm_b, policy=pol)
            except Exception:
                mem_l_b = weights_l_b + peak_l_b + kv_l_b + overhead_l_b + comm_b

            try:
                mem_r_b = estimate_ram_bytes(weights_r_b, peak_r_b, kv_r_b, overhead_r_b, comm_b, policy=pol)
            except Exception:
                mem_r_b = weights_r_b + peak_r_b + kv_r_b + overhead_r_b + comm_b

            total_l_mb = mem_l_b / MiB
            total_r_mb = mem_r_b / MiB

        left = {
            "name": left_accel_name or "Left",
            "ram_limit_mb": limit_l_mb,
            "total_mb": total_l_mb,
            "weights_mb": weights_l_b / MiB if weights_l_b else 0.0,
            "peak_act_mb": peak_l_b / MiB if peak_l_b else 0.0,
            "kv_cache_mb": kv_l_b / MiB if kv_l_b else 0.0,
            "comm_mb": comm_b / MiB if comm_b else 0.0,
            "overhead_mb": overhead_l_mb,
        }
        right = {
            "name": right_accel_name or "Right",
            "ram_limit_mb": limit_r_mb,
            "total_mb": total_r_mb,
            "weights_mb": weights_r_b / MiB if weights_r_b else 0.0,
            "peak_act_mb": peak_r_b / MiB if peak_r_b else 0.0,
            "kv_cache_mb": kv_r_b / MiB if kv_r_b else 0.0,
            "comm_mb": comm_b / MiB if comm_b else 0.0,
            "overhead_mb": overhead_r_mb,
        }

        if limit_l_mb > 0:
            left["fits"] = bool(total_l_mb <= limit_l_mb)
        if limit_r_mb > 0:
            right["fits"] = bool(total_r_mb <= limit_r_mb)

        return {"left": left, "right": right}


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
            # Prefer the user-provided proxy (same knob used for Comm(b) ranking), fall back
            # to median known activation size if the proxy is 0.
            if float(getattr(p, "unknown_tensor_proxy_mb", 0.0) or 0.0) > 0:
                assume_unknown_bytes = int(float(p.unknown_tensor_proxy_mb) * 1e6)
            else:
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

    def _configure_candidate_columns(self) -> None:
        summary_cols = ["rank", "clean", "boundary", "semantic", "cut_mb", "num_tensors", "gflops_left", "gflops_right"]
        advanced = bool(self.var_cand_advanced.get()) if hasattr(self, "var_cand_advanced") else False
        display = list(self.tree["columns"]) if advanced else summary_cols
        self.tree.configure(displaycolumns=display)

    def _refresh_candidates_table(self, _evt=None) -> None:
        if not isinstance(self.analysis, dict) or self._last_params is None:
            return
        self._update_table(self.analysis, self.current_picks, self._last_params)

    def _candidate_passes_search(self, row: Dict[str, Any], search: str, use_regex: bool) -> bool:
        if not search:
            return True
        hay = " | ".join([str(row.get("semantic", "")), str(row.get("left_op", "")), str(row.get("right_op", "")), str(row.get("boundary", "")), " ".join(row.get("cut_tensors", []))])
        if use_regex:
            try:
                return re.search(search, hay, flags=re.IGNORECASE) is not None
            except re.error:
                return True
        return search.lower() in hay.lower()

    def _candidate_clean_rank(self, symbol: str) -> int:
        return {"✅": 0, "⚠️": 1, "❌": 2}.get(str(symbol), 3)

    def _filtered_sorted_candidate_rows(self, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        search = str(self.var_cand_search.get() or "").strip() if hasattr(self, "var_cand_search") else ""
        use_regex = bool(self.var_cand_search_regex.get()) if hasattr(self, "var_cand_search_regex") else False
        hide_dirty = bool(self.var_cand_hide_dirty.get()) if hasattr(self, "var_cand_hide_dirty") else False
        group_sem = bool(self.var_cand_group_semantic.get()) if hasattr(self, "var_cand_group_semantic") else False
        sort_mode = str(self.var_cand_sort.get() or "Rank ↑") if hasattr(self, "var_cand_sort") else "Rank ↑"

        out = [r for r in rows if self._candidate_passes_search(r, search, use_regex)]
        if hide_dirty:
            out = [r for r in out if str(r.get("clean_symbol", "")) == "✅"]

        if group_sem:
            by_sem: Dict[str, Dict[str, Any]] = {}
            for r in out:
                sem = str(r.get("semantic", "") or "<none>")
                prev = by_sem.get(sem)
                if prev is None or float(r.get("cut_mb_val", 0.0)) < float(prev.get("cut_mb_val", 0.0)):
                    by_sem[sem] = r
            out = list(by_sem.values())

        if sort_mode == "Boundary ↑":
            out.sort(key=lambda r: int(r.get("boundary", -1)))
        elif sort_mode == "Boundary ↓":
            out.sort(key=lambda r: int(r.get("boundary", -1)), reverse=True)
        elif sort_mode == "Cut MB ↑":
            out.sort(key=lambda r: float(r.get("cut_mb_val", 0.0)))
        elif sort_mode == "Cut MB ↓":
            out.sort(key=lambda r: float(r.get("cut_mb_val", 0.0)), reverse=True)
        elif sort_mode == "Clean (best)":
            out.sort(key=lambda r: (self._candidate_clean_rank(str(r.get("clean_symbol", ""))), float(r.get("cut_mb_val", 0.0))))
        else:
            out.sort(key=lambda r: int(r.get("rank", 10**9)))
        return out

    def _on_tree_motion_clean_tooltip(self, evt=None) -> None:
        if evt is None:
            return
        row_id = self.tree.identify_row(evt.y)
        col_id = self.tree.identify_column(evt.x)
        if not row_id or col_id != "#2":
            self._hide_tree_clean_tooltip()
            return
        txt = self._tree_clean_tooltips.get(row_id, "")
        if not txt:
            self._hide_tree_clean_tooltip()
            return
        if self._clean_tooltip_tip is not None and self._clean_tooltip_row == row_id:
            return
        self._hide_tree_clean_tooltip()
        tw = tk.Toplevel(self.tree)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{self.tree.winfo_rootx()+evt.x+12}+{self.tree.winfo_rooty()+evt.y+12}")
        tk.Label(tw, text=txt, justify=tk.LEFT, background="#ffffe0", relief=tk.SOLID, borderwidth=1, wraplength=420).pack(ipadx=6, ipady=3)
        self._clean_tooltip_tip = tw
        self._clean_tooltip_row = row_id

    def _hide_tree_clean_tooltip(self, _evt=None) -> None:
        if self._clean_tooltip_tip is not None:
            try:
                self._clean_tooltip_tip.destroy()
            except Exception:
                pass
            self._clean_tooltip_tip = None
            self._clean_tooltip_row = None

    def _update_table(self, a: Dict, picks: List[int], p: Params):
        if self.analysis_result is None:
            self.analysis_result = AnalysisResult()
        self.analysis_result.candidates = list(picks)
        self._last_params = p
        self.tree.delete(*self.tree.get_children())
        self.tree.state(("!disabled",))
        self._hide_tree_clean_tooltip()
        self._cand_by_iid = {}
        self._tree_clean_tooltips = {}

        nodes = a["nodes"]
        order = a["order"]
        costs = a["costs_bytes"]
        counts_all = a["crossing_counts_all"]
        unknown = a["unknown_crossing_counts"]
        flops_left_prefix = a["flops_left_prefix"]
        sem_labels = a.get("semantic_labels_by_boundary") or []
        total_flops = float(a["total_flops"])
        peak_left_b = a.get("peak_act_mem_left_bytes") or []
        peak_right_b = a.get("peak_act_mem_right_bytes") or []
        peak_max_b = a.get("peak_act_mem_max_bytes") or []
        weights_left_b = a.get("weights_left_bytes") or []
        weights_right_b = a.get("weights_right_bytes") or []
        kv_left_b = a.get("kv_left_bytes") or []
        kv_right_b = a.get("kv_right_bytes") or []
        vimap = a.get("vimap") or {}

        left_acc = self._accel_by_name(self.var_memf_left_accel.get())
        right_acc = self._accel_by_name(self.var_memf_right_accel.get())
        pol = str(self.var_memf_policy.get() or "max_peak_or_comm")
        include_kv = bool(self.var_memf_include_kv.get())
        include_comm = bool(self.var_memf_include_comm.get())

        self.memory_by_boundary = {}
        candidate_rows: List[Dict[str, Any]] = []

        def _mb(x_bytes: float) -> float:
            return float(x_bytes) / 1e6

        for r, b in enumerate(picks, 1):
            lidx, ridx = order[b], order[b + 1]
            cut_tensors: List[str] = []
            try:
                cut_tensors = list(asc.cut_tensors_for_boundary(order, nodes, int(b)))
            except Exception:
                cut_tensors = []

            cut_mb = _mb(costs[b])
            num_tensors = int(counts_all[b]) if b < len(counts_all) else 0
            fl_l = float(flops_left_prefix[b])
            fl_r = float(total_flops - flops_left_prefix[b])
            gfl_l = fl_l / 1e9
            gfl_r = fl_r / 1e9

            wl = int(weights_left_b[b]) if b < len(weights_left_b) else 0
            wr = int(weights_right_b[b]) if b < len(weights_right_b) else 0
            kl = int(kv_left_b[b]) if (include_kv and b < len(kv_left_b)) else 0
            kr = int(kv_right_b[b]) if (include_kv and b < len(kv_right_b)) else 0
            comm = int(costs[b]) if include_comm else 0

            ovh_l = int(float(left_acc.get("runtime_overhead_mb") or 0.0) * 1024.0 * 1024.0)
            ovh_r = int(float(right_acc.get("runtime_overhead_mb") or 0.0) * 1024.0 * 1024.0)
            total_l = estimate_ram_bytes(wl, int(peak_left_b[b]) if b < len(peak_left_b) else 0, kl, ovh_l, comm, policy=pol)
            total_r = estimate_ram_bytes(wr, int(peak_right_b[b]) if b < len(peak_right_b) else 0, kr, ovh_r, comm, policy=pol)
            lim_l = int(float(left_acc.get("ram_limit_mb") or 0.0) * 1024.0 * 1024.0)
            lim_r = int(float(right_acc.get("ram_limit_mb") or 0.0) * 1024.0 * 1024.0)
            fits_l = (total_l <= lim_l) if lim_l > 0 else False
            fits_r = (total_r <= lim_r) if lim_r > 0 else False

            self.memory_by_boundary[int(b)] = {
                "left": {
                    "weights_mb": wl / (1024.0**2), "kv_mb": kl / (1024.0**2),
                    "peak_activations_mb": (int(peak_left_b[b]) if b < len(peak_left_b) else 0) / (1024.0**2),
                    "comm_mb": comm / (1024.0**2), "runtime_overhead_mb": ovh_l / (1024.0**2),
                    "total_mb": total_l / (1024.0**2), "fits": bool(fits_l),
                },
                "right": {
                    "weights_mb": wr / (1024.0**2), "kv_mb": kr / (1024.0**2),
                    "peak_activations_mb": (int(peak_right_b[b]) if b < len(peak_right_b) else 0) / (1024.0**2),
                    "comm_mb": comm / (1024.0**2), "runtime_overhead_mb": ovh_r / (1024.0**2),
                    "total_mb": total_r / (1024.0**2), "fits": bool(fits_r),
                },
            }

            semantic = (sem_labels[b] if b < len(sem_labels) else "")
            left_op = nodes[lidx].op_type
            right_op = nodes[ridx].op_type
            clean = cand_panel.compute_candidate_clean_status(
                boundary=int(b),
                semantic_label=str(semantic),
                left_op=str(left_op),
                right_op=str(right_op),
                cut_tensor_names=cut_tensors,
                vimap=vimap,
            )
            tooltip = "Clean split" if not clean.reasons else "\n".join(clean.reasons)

            candidate_rows.append({
                "rank": int(r),
                "boundary": int(b),
                "semantic": str(semantic),
                "left_op": str(left_op),
                "right_op": str(right_op),
                "cut_mb": f"{cut_mb:.3f}",
                "cut_mb_val": float(cut_mb),
                "num_tensors": int(num_tensors),
                "gflops_left": f"{gfl_l:.3f}",
                "gflops_right": f"{gfl_r:.3f}",
                "peak_left_mib": f"{(float(peak_left_b[b]) / (1024.0**2)):.2f}" if b < len(peak_left_b) else "",
                "peak_right_mib": f"{(float(peak_right_b[b]) / (1024.0**2)):.2f}" if b < len(peak_right_b) else "",
                "peak_max_mib": f"{(float(peak_max_b[b]) / (1024.0**2)):.2f}" if b < len(peak_max_b) else "",
                "fits_left": "✅" if fits_l else "❌",
                "fits_right": "✅" if fits_r else "❌",
                "ram_left_gb": f"{total_l/(1024.0**3):.2f}",
                "ram_right_gb": f"{total_r/(1024.0**3):.2f}",
                "clean_symbol": clean.symbol,
                "clean_tooltip": tooltip,
                "clean_flags": sorted(list(clean.flags)),
                "cut_tensors": cut_tensors,
                "unknown_count": int(unknown[b]) if b < len(unknown) else 0,
            })

        self._candidate_rows = list(candidate_rows)
        rows_for_view = self._filtered_sorted_candidate_rows(candidate_rows)
        self.analysis_result.candidates = [int(r["boundary"]) for r in rows_for_view]

        for row in rows_for_view:
            iid = f"b{int(row['boundary'])}"
            parent = self.tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    row["rank"],
                    row["clean_symbol"],
                    row["boundary"],
                    row["semantic"],
                    row["cut_mb"],
                    row["num_tensors"],
                    row["gflops_left"],
                    row["gflops_right"],
                    row["left_op"],
                    row["right_op"],
                    row["peak_left_mib"],
                    row["peak_right_mib"],
                    row["peak_max_mib"],
                    row["fits_left"],
                    row["fits_right"],
                    row["ram_left_gb"],
                    row["ram_right_gb"],
                ),
                tags=(("pick", "dirty") if row["clean_symbol"] != "✅" else ("pick",)),
            )
            self._cand_by_iid[iid] = row
            self._tree_clean_tooltips[parent] = str(row.get("clean_tooltip", ""))

            if int(row.get("unknown_count", 0)) > 0:
                self.tree.insert(
                    parent,
                    "end",
                    values=("", "", "", "↳ unknown sizes", "", f"+{int(row['unknown_count'])}", "", "", "", "", "", "", "", "", "", "", ""),
                )

            if p.show_top_tensors > 0:
                b = int(row["boundary"])
                crossing = asc.collect_crossing_values_for_boundary(b, a["val_span"], a["value_bytes"])
                for name, sz in crossing[: p.show_top_tensors]:
                    self.tree.insert(
                        parent,
                        "end",
                        values=("", "", "", f"↳ {name}", f"{_mb(sz):.3f}", "", "", "", "", "", "", "", "", "", "", "", ""),
                    )

        self._configure_candidate_columns()

        if self.analysis_result is None:
            self.analysis_result = AnalysisResult()
        self.analysis_result.memory_estimate = dict(self.memory_by_boundary)

        # Optional debug dump for Memory Fit: write the full per-boundary table so we can
        # inspect/plot it outside the GUI without guessing.
        if os.environ.get("SPLITPOINT_DEBUG_MEMFIT", "").strip().lower() in {"1", "true", "yes"}:
            try:
                dbg_dir = Path.home() / ".onnx_splitpoint_tool"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                dbg_csv = dbg_dir / "memfit_debug.csv"
                with dbg_csv.open("w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(
                        [
                            "boundary",
                            "left_total_mb",
                            "right_total_mb",
                            "left_weights_mb",
                            "right_weights_mb",
                            "left_peak_activations_mb",
                            "right_peak_activations_mb",
                            "left_kv_mb",
                            "right_kv_mb",
                            "left_overhead_mb",
                            "right_overhead_mb",
                            "comm_mb",
                            "left_fits",
                            "right_fits",
                        ]
                    )
                    for bb in sorted(self.memory_by_boundary.keys()):
                        est = self.memory_by_boundary.get(bb) or {}
                        l = est.get("left") or {}
                        r = est.get("right") or {}
                        w.writerow(
                            [
                                bb,
                                l.get("total_mb", 0.0),
                                r.get("total_mb", 0.0),
                                l.get("weights_mb", 0.0),
                                r.get("weights_mb", 0.0),
                                l.get("peak_activations_mb", 0.0),
                                r.get("peak_activations_mb", 0.0),
                                l.get("kv_mb", 0.0),
                                r.get("kv_mb", 0.0),
                                l.get("overhead_mb", 0.0),
                                r.get("overhead_mb", 0.0),
                                l.get("comm_mb", 0.0),
                                l.get("fits", False),
                                r.get("fits", False),
                            ]
                        )
                logger.info("MemFit debug CSV written: %s", dbg_csv)
            except Exception:
                logger.exception("Failed to write MemFit debug CSV")
        logger.debug("Candidate table populated: rows=%d mapped=%d", len(rows_for_view), len(self._cand_by_iid))
    # ----------------------------- Plotting -----------------------------

    def _update_plots(self, a: Dict, picks: List[int], p: Params):
        if self.analysis_result is None:
            self.analysis_result = AnalysisResult()
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

        # Candidate split markers (dashed) – keep consistent with the compute/latency plots.
        if picks:
            for b in picks:
                self.ax_comm.axvline(
                    b,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    zorder=3,
                )
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
        self.ax_lat.set_title("Latency model vs boundary")
        self.ax_lat.set_xlabel("Boundary index")
        self.ax_lat.set_ylabel("ms")

        # Recompute latency using the full SystemSpec (supports packetized links)
        bw_bps = asc.bandwidth_to_bytes_per_s(p.bw_value, p.bw_unit)
        has_bw = bw_bps is not None and bw_bps > 0
        has_gops = (
            p.gops_left is not None
            and p.gops_right is not None
            and p.gops_left > 0
            and p.gops_right > 0
            and total_flops > 0
        )

        if has_bw:
            sys = self._build_system_spec(p)
            comm_ms = []
            total_ms = []
            for cb, fl in zip(costs, flops_left_prefix):
                est = sys.estimate_boundary(
                    comm_bytes=float(cb), flops_left=float(fl), flops_total=float(total_flops)
                )
                lat_link = est.get("latency_link_ms")
                lat_tot = est.get("latency_total_ms") if has_gops else None
                comm_ms.append(float(lat_link) if lat_link is not None else float("nan"))
                total_ms.append(float(lat_tot) if lat_tot is not None else float("nan"))

            if has_gops:
                self.ax_lat.plot(xs, total_ms, label="Total latency (ms)")
            self.ax_lat.plot(xs, comm_ms, label="Comm-only (ms)")

            # Candidate split markers (dashed) – match the compute plot.
            if picks:
                for b in picks:
                    self.ax_lat.axvline(b, linestyle="--", linewidth=1.0, alpha=0.8)
            self.ax_lat.legend()

        else:
            self.ax_lat.text(
                0.02,
                0.95,
                "Provide: link bandwidth + GOPS left/right\n(optional overhead)",
                transform=self.ax_lat.transAxes,
                va="top",
            )
        self._highlight_selected_boundary_in_plots()
        self.canvas.draw_idle()

        # IMPORTANT: do NOT clobber the original plot_data dict coming from the
        # analysis thread. It contains additional arrays (e.g. weights / peak mem)
        # that are required by other UI features like Memory Fit.
        pd = self.analysis_result.plot_data
        if not isinstance(pd, dict):
            pd = {}
        pd.update(
            {
                "costs_bytes": list(costs),
                "flops_left_prefix": list(flops_left_prefix),
                "total_flops": float(total_flops),
                "imbalance": list(imbalance),
                "selected_candidates": list(picks),
            }
        )
        self.analysis_result.plot_data = pd

    # ----------------------------- Split models -----------------------------



    def _schedule_latency_recompute(self, reason: str = "") -> None:
        """Debounced latency plot recompute (triggered by HW/link changes)."""
        if getattr(self, "analysis_result", None) is None:
            return
        try:
            pd = getattr(self.analysis_result, "plot_data", None)
            if not isinstance(pd, dict):
                return
        except Exception:
            return

        # Debounce: HW widgets can update several vars in quick succession.
        after_id = getattr(self, "_latency_recompute_after_id", None)
        if after_id is not None:
            try:
                self.after_cancel(after_id)
            except Exception:
                pass

        self._latency_recompute_after_id = self.after(150, self._update_latency_plot_only)

    def _update_latency_plot_only(self) -> None:
        """Recompute just the latency subplot using current HW settings."""
        if getattr(self, "analysis_result", None) is None:
            return

        try:
            p = self._read_params()
        except Exception:
            return

        pd = getattr(self.analysis_result, "plot_data", None)
        if not isinstance(pd, dict):
            return

        # Backward compatible: plot_data may be either the raw analysis dict
        # or a wrapper that contains it under the "analysis" key.
        analysis = pd.get("analysis") if isinstance(pd.get("analysis"), dict) else pd
        if not isinstance(analysis, dict):
            return

        costs = analysis.get("costs_bytes") or []
        flops_left_prefix = analysis.get("flops_left_prefix") or []
        total_flops = float(analysis.get("total_flops") or 0.0)
        M = min(len(costs), len(flops_left_prefix))
        if M <= 0:
            return

        costs = costs[:M]
        flops_left_prefix = flops_left_prefix[:M]
        picks = pd.get("selected_candidates") or []

        # Redraw only ax_lat
        self.ax_lat.clear()
        self.ax_lat.set_title("Latency model vs boundary")
        self.ax_lat.set_xlabel("Boundary index")
        self.ax_lat.set_ylabel("ms")

        bw_bps = asc.bandwidth_to_bytes_per_s(p.bw_value, p.bw_unit)
        has_bw = bw_bps is not None and bw_bps > 0
        has_gops = (
            p.gops_left is not None
            and p.gops_right is not None
            and p.gops_left > 0
            and p.gops_right > 0
            and total_flops > 0
        )

        if has_bw:
            sys = self._build_system_spec(p)
            xs = list(range(M))
            comm_ms = []
            total_ms = []
            for cb, fl in zip(costs, flops_left_prefix):
                est = sys.estimate_boundary(
                    comm_bytes=float(cb), flops_left=float(fl), flops_total=float(total_flops)
                )
                lat_link = est.get("latency_link_ms")
                lat_tot = est.get("latency_total_ms") if has_gops else None
                comm_ms.append(float(lat_link) if lat_link is not None else float("nan"))
                total_ms.append(float(lat_tot) if lat_tot is not None else float("nan"))

            if has_gops:
                self.ax_lat.plot(xs, total_ms, label="Total latency (ms)")
            self.ax_lat.plot(xs, comm_ms, label="Comm-only (ms)")
            self.ax_lat.legend()

            # Keep the same pick markers (dashed vlines)
            for pick in picks:
                try:
                    b = int(pick)
                except Exception:
                    continue
                if 0 <= b < M:
                    self.ax_lat.axvline(b, color="C0", linestyle="--", linewidth=1.0, alpha=0.8)

        else:
            self.ax_lat.text(
                0.02,
                0.95,
                "Provide: link bandwidth + GOPS left/right\n(optional overhead)",
                transform=self.ax_lat.transAxes,
                va="top",
            )

        # Restore selection marker (red vline / pareto cross)
        self._highlight_selected_boundary_in_plots()
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def _is_boundary_row(self, item: str) -> bool:
        """Return True if the item is a *top-level* boundary row (not a child tensor row)."""
        if not item:
            return False
        if item not in self._cand_by_iid:
            # Covers child rows and optional semantic-group header rows.
            return False
        if self.tree.parent(item):
            return False
        vals = self.tree.item(item, "values")
        if len(vals) < 3:
            return False
        try:
            int(vals[0])  # rank
            int(vals[2])  # boundary index
            return True
        except Exception:
            return False

    def _selected_boundary_index(self) -> Optional[int]:
        """Return selected boundary with tree selection priority and robust child-row handling."""
        sel = self.tree.selection()
        if sel:
            item = sel[0]

            # Child rows (e.g., tensor rows) should resolve to their boundary parent row.
            if not self._is_boundary_row(item):
                parent = self.tree.parent(item)
                if parent:
                    item = parent

            if self._is_boundary_row(item):
                row = self._cand_by_iid.get(item)
                if row is not None:
                    try:
                        return int(row.get("boundary", -1))
                    except Exception:
                        return None
                vals = self.tree.item(item, "values")
                if len(vals) >= 3:
                    try:
                        return int(vals[2])
                    except Exception:
                        return None
            return None

        # Fallback only when tree selection is empty.
        if self.selected_candidate is not None:
            try:
                return int(self.selected_candidate.boundary_id)
            except Exception:
                return None
        return None

    def _update_action_buttons(self) -> None:
        """Enable/disable buttons based on explicit app UI state."""
        state = self.ui_state
        try:
            # Analyse is only blocked while a run is active.
            if state == AppUiState.ANALYSING:
                self.btn_analyse.state(["disabled"])
            else:
                self.btn_analyse.state(["!disabled"])

            has_analysis = state in {AppUiState.ANALYSED, AppUiState.SPLIT_READY}
            has_selection = state == AppUiState.SPLIT_READY

            self.btn_export_tex.state(["!disabled"] if has_analysis else ["disabled"])
            self.btn_split.state(["!disabled"] if has_selection else ["disabled"])
            # Acceptance: no benchmark without split-ready selection.
            self.btn_benchmark.state(["!disabled"] if has_selection else ["disabled"])

            ns = (self.analysis or {}).get("nordstern") or {}
            has_unknown = int(ns.get("unknown_count") or 0) > 0
            self.btn_nordstern.state(["!disabled"] if has_analysis and has_unknown else ["disabled"])
        except Exception:
            pass


    def _split_selected_boundary(self) -> None:
        model_path = self.gui_state.current_model_path or self.model_path
        if self.analysis is None or model_path is None:
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

        initial_out = self.default_output_dir or os.path.dirname(model_path)
        try:
            if self.default_output_dir:
                initial_out = str(ensure_workdir(Path(self.default_output_dir)).split_networks)
        except Exception:
            pass
        out_parent = filedialog.askdirectory(title="Select working directory", initialdir=initial_out)
        if not out_parent:
            return
        # Keep the *working dir root* stable if the user picked a subfolder like
        # <root>/SplitNetworks.
        try:
            base = os.path.basename(out_parent.rstrip(os.sep))
            if base in ("SplitNetworks", "BenchmarkSets", "Results"):
                self.default_output_dir = os.path.dirname(out_parent)
            else:
                self.default_output_dir = out_parent
        except Exception:
            self.default_output_dir = out_parent

        base = os.path.splitext(os.path.basename(model_path))[0]
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

        # Read pruning params from the current GUI state (same source as Analyse).
        _params_for_split = self._read_params()
        prune_skip_block = bool(getattr(_params_for_split, "prune_skip_block", False))
        skip_min_span = int(getattr(_params_for_split, "skip_min_span", 0) or 0)
        if skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = int(getattr(_params_for_split, "skip_allow_last_n", 0) or 0)
        if skip_allow_last_n < 0:
            raise ValueError("Allow last N inside must be an integer ≥ 0.")
        do_validate = bool(self.var_split_validate.get())
        do_runner = bool(self.var_split_runner.get())
        runner_target = str(self.var_runner_target.get() or "auto")

        # Context export toggles.
        #
        # Practical default: when exporting a split as a folder, always emit context artifacts.
        # This avoids the "where are the PDFs?" confusion when users forget that the
        # context checkboxes exist or are unchecked.
        do_ctx_full_user = bool(getattr(self, 'var_split_ctx_full', tk.BooleanVar(value=True)).get())
        do_ctx_cutflow_user = bool(getattr(self, 'var_split_ctx_cutflow', tk.BooleanVar(value=True)).get())
        if (not do_ctx_full_user) and (not do_ctx_cutflow_user):
            # Safety: ensure at least one context artifact for the split folder
            do_ctx_cutflow_user = True
        do_ctx_full = do_ctx_full_user or export_as_folder
        do_ctx_cutflow = do_ctx_cutflow_user or export_as_folder

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

        # -------- Hailo HEF generation (optional) --------
        hef_part1 = bool(getattr(self, "var_hailo_hef_part1", tk.BooleanVar(value=False)).get())
        hef_part2 = bool(getattr(self, "var_hailo_hef_part2", tk.BooleanVar(value=False)).get())

        hef_targets: List[str] = []
        if bool(getattr(self, "var_hailo_hef_hailo8_enable", tk.BooleanVar(value=True)).get()):
            hw = (getattr(self, "var_hailo_hef_hailo8_hw_arch", tk.StringVar(value="hailo8")).get() or "hailo8").strip()
            if hw:
                hef_targets.append(hw)
        if bool(getattr(self, "var_hailo_hef_hailo10_enable", tk.BooleanVar(value=False)).get()):
            hw = (getattr(self, "var_hailo_hef_hailo10_hw_arch", tk.StringVar(value="hailo10h")).get() or "hailo10h").strip()
            if hw:
                hef_targets.append(hw)

        hef_opt_level = _safe_int((getattr(self, "var_hailo_hef_opt_level", tk.StringVar(value="1")).get() or "").strip()) or 1
        hef_calib_count = _safe_int((getattr(self, "var_hailo_hef_calib_count", tk.StringVar(value="64")).get() or "").strip()) or 64
        hef_calib_bs = _safe_int((getattr(self, "var_hailo_hef_calib_batch_size", tk.StringVar(value="8")).get() or "").strip()) or 8
        hef_calib_dir = (getattr(self, "var_hailo_hef_calib_dir", tk.StringVar(value="")).get() or "").strip() or None
        hef_force = bool(getattr(self, "var_hailo_hef_force", tk.BooleanVar(value=False)).get())
        hef_keep = bool(getattr(self, "var_hailo_hef_keep_artifacts", tk.BooleanVar(value=False)).get())

        # Backend selection reuses the Hailo feasibility-check backend controls.
        hef_backend = (getattr(self, "var_hailo_backend", tk.StringVar(value="auto")).get() or "auto").strip()
        hef_wsl_distro = (getattr(self, "var_hailo_wsl_distro", tk.StringVar(value="")).get() or "").strip() or None
        hef_wsl_venv = (getattr(self, "var_hailo_wsl_venv", tk.StringVar(value="auto")).get() or "auto").strip() or "auto"
        hef_fixup = bool(getattr(self, "var_hailo_fixup", tk.BooleanVar(value=True)).get())

        # ---- progress dialog + background worker ----
        dlg = tk.Toplevel(self)
        dlg.title("Split and Export")
        dlg.transient(self)
        try:
            dlg.grab_set()
        except Exception:
            pass
        dlg.resizable(True, True)
        dlg.minsize(760, 420)

        status_var = tk.StringVar(value=f"Splitting boundary {b} and exporting artifacts…")
        ttk.Label(dlg, textvariable=status_var).pack(padx=16, pady=(16, 6), anchor="w")

        pb = ttk.Progressbar(dlg, mode="indeterminate")
        pb.pack(padx=16, pady=(0, 10), fill="x")
        pb.start(10)

        log_frame = ttk.Frame(dlg)
        log_frame.pack(padx=16, pady=(0, 16), fill="both", expand=True)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        log_txt = tk.Text(log_frame, height=14, wrap="none", state="disabled")
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical", command=log_txt.yview)
        log_txt.configure(yscrollcommand=log_vsb.set)
        log_txt.grid(row=0, column=0, sticky="nsew")
        log_vsb.grid(row=0, column=1, sticky="ns")

        try:
            self.configure(cursor="watch")
            self.update_idletasks()
        except Exception:
            pass

        q: "queue.Queue[tuple]" = queue.Queue()

        def worker() -> None:
            try:
                msg = []

                q.put(("stage", f"Splitting boundary {b} and exporting artifacts…"))

                # Split
                p1, p2, split_manifest = asc.split_model_on_cut_tensors(
                    model,
                    cut_tensors=cut_tensors,
                    strict_boundary=strict_boundary,
                    p1_cut_names=p1_cut_names,
                    p2_cut_names=p2_cut_names,
                )

                # If the model uses external data (weights in separate *.data files),
                # materialize those files into the export folder (prefer hardlink / symlink
                # to avoid copying multi-GB weights).
                try:
                    actions = {}
                    actions.update(
                        asc.ensure_external_data_files(
                            p1,
                            source_model_path=self.model_path,
                            dest_dir=out_dir,
                            mode="auto",
                        )
                    )
                    actions.update(
                        asc.ensure_external_data_files(
                            p2,
                            source_model_path=self.model_path,
                            dest_dir=out_dir,
                            mode="auto",
                        )
                    )
                    if actions:
                        if any(str(v).startswith("absolute(") for v in actions.values()):
                            msg.append(
                                "External data: could not link/copy weights into export folder; "
                                "wrote absolute paths into ONNX (non-portable)."
                            )
                        else:
                            msg.append(f"External data: materialized {len(actions)} file(s) in export folder")
                except FileNotFoundError as e:
                    # The export would be unusable. Turn into a clear message.
                    raise RuntimeError(
                        "This model references external weight data, but the referenced *.data file was not found. "
                        "Keep the external data file next to the original ONNX (matching the `location` inside the model)."
                    ) from e
                asc.save_model(p1, p1_path)
                asc.save_model(p2, p2_path)

                msg.append(f"Boundary: {b}")
                msg.append(f"Cut tensors: {cut_tensors}")
                msg.append(f"Strict boundary: {strict_boundary}")
                msg.append(f"Wrote: {p1_path}")
                msg.append(f"Wrote: {p2_path}")

                # Make the export folder self-contained for portability.
                # If the copy fails, we keep the original absolute path so the export remains usable locally.
                full_model_src = os.path.abspath(self.model_path)
                full_model_local = os.path.join(out_dir, os.path.basename(full_model_src))
                full_model_field = full_model_src
                try:
                    if os.path.abspath(full_model_local) != full_model_src:
                        shutil.copy2(full_model_src, full_model_local)
                        msg.append(f"Wrote: {full_model_local}")
                    if os.path.exists(full_model_local):
                        full_model_field = os.path.basename(full_model_local)
                except Exception as e:
                    msg.append(f"[warn] Could not copy full model into export folder: {e}")

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
                    "full_model": str(full_model_field).replace('\\', '/'),
                    "full_model_source": str(full_model_src).replace('\\', '/'),
                    "part1": os.path.basename(p1_path).replace('\\', '/'),
                    "part1_model": os.path.basename(p1_path).replace('\\', '/'),
                    "part2": os.path.basename(p2_path).replace('\\', '/'),
                    "part2_model": os.path.basename(p2_path).replace('\\', '/'),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                if isinstance(split_manifest, dict):
                    manifest_out.update(split_manifest)
                # Normalized schema fields (v1). Keep legacy flat keys for backwards compatibility.
                try:
                    manifest_out.setdefault("schema", "onnx-splitpoint/split-manifest")
                    manifest_out.setdefault("schema_version", 1)

                    manifest_out["models"] = {
                        "full": {
                            "path": str(manifest_out.get("full_model") or "").replace('\\', '/'),
                            "source": str(manifest_out.get("full_model_source") or "").replace('\\', '/'),
                        },
                        "part1": {
                            "path": str(manifest_out.get("part1_model") or manifest_out.get("part1") or "").replace('\\', '/')
                        },
                        "part2": {
                            "path": str(manifest_out.get("part2_model") or manifest_out.get("part2") or "").replace('\\', '/')
                        },
                    }

                    cut_full = manifest_out.get("cut_tensors_full") or manifest_out.get("cut_tensors") or []
                    cut_p1 = manifest_out.get("part1_cut_names") or []
                    cut_p2 = manifest_out.get("part2_cut_names") or []
                    manifest_out["cut"] = {
                        "full_names": list(cut_full) if isinstance(cut_full, list) else [],
                        "part1_names": list(cut_p1) if isinstance(cut_p1, list) else [],
                        "part2_names": list(cut_p2) if isinstance(cut_p2, list) else [],
                    }

                    manifest_out["io"] = {
                        "orig_inputs": list(manifest_out.get("orig_inputs") or []),
                        "orig_outputs": list(manifest_out.get("orig_outputs") or []),
                        "part1_external_inputs": list(manifest_out.get("part1_external_inputs") or []),
                        "part2_external_inputs": list(manifest_out.get("part2_external_inputs") or []),
                    }

                    links = []
                    if isinstance(cut_p1, list) and isinstance(cut_p2, list) and len(cut_p1) == len(cut_p2):
                        for a, bname in zip(cut_p1, cut_p2):
                            if a and bname:
                                links.append({"from": str(a), "to": str(bname)})
                    manifest_out["pipeline"] = {
                        "stage1": "part1",
                        "stage2": "part2",
                        "links": links,
                    }
                except Exception:
                    pass


                mem = (self.memory_by_boundary or {}).get(int(b))
                left_acc = self._accel_by_name(self.var_memf_left_accel.get())
                right_acc = self._accel_by_name(self.var_memf_right_accel.get())
                mode = str(self.var_llm_mode.get() or "decode")
                seq_len = int(self.var_llm_prefill.get() or 0) if mode == "prefill" else 1
                past_len = 0 if mode == "prefill" else int(self.var_llm_decode.get() or 0)
                if mem:
                    manifest_out["memory_forecast"] = {
                        "mode": mode,
                        "batch": int(params.batch_override or 1) if params is not None else 1,
                        "seq_len": int(seq_len),
                        "past_len": int(past_len),
                        "left_accelerator": left_acc.get("id"),
                        "right_accelerator": right_acc.get("id"),
                        "interface": self.var_memf_interface.get(),
                        "policy": str(self.var_memf_policy.get()),
                        "left": mem.get("left", {}),
                        "right": mem.get("right", {}),
                        "left_spec": left_acc,
                        "right_spec": right_acc,
                    }

                # Split-context diagrams around the boundary (GraphViz if available; otherwise fallback).
                # Selected row metadata lives in the current analysis model.
                # We avoid relying on Treeview payload here because the worker runs asynchronously.
                semantic_label = None
                try:
                    sem = self.analysis.get("semantic_labels_by_boundary") if isinstance(self.analysis, dict) else None
                    if isinstance(sem, list) and 0 <= int(b) < len(sem):
                        semantic_label = sem[int(b)]
                except Exception:
                    semantic_label = None

                # Use a compact, "LLM-friendly" context rendering only for LLM runs.
                # (Non-LLM models like YOLO/ResNet should keep the full context graph rendering.)
                #
                # NOTE: This worker runs asynchronously and should not depend on GUI variables
                # that may not exist in this scope. We infer "LLM style" from the selected
                # boundary metadata (semantic labels) and/or whether LLM presets are enabled.
                llm_style = bool(getattr(params, "llm_enable", False))
                value_bytes_map = self.analysis.get("value_bytes") if isinstance(self.analysis, dict) else None

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
                            strict_boundary=bool(self.var_strict_boundary.get()),
                            include_external_inputs=(not llm_style),
                            semantic_label=semantic_label,
                            value_bytes_map=value_bytes_map,
                            force_matplotlib_fallback=llm_style,
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
                            strict_boundary=bool(self.var_strict_boundary.get()),
                            cut_flow_only=True,
                            include_internal_consumers=False,
                            include_external_inputs=False,
                            semantic_label=semantic_label,
                            value_bytes_map=value_bytes_map,
                            force_matplotlib_fallback=llm_style,
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

                # Optional: Build Hailo HEFs (part1/part2) for selected targets.
                if hef_targets and (hef_part1 or hef_part2):
                    q.put(("stage", "Building Hailo HEFs…"))
                    try:
                        from .hailo_backend import hailo_build_hef_auto
                    except Exception as e:
                        msg.append(f"Hailo HEF build unavailable: {e}")
                    else:
                        manifest_out.setdefault("hailo", {})
                        manifest_out["hailo"].setdefault("hefs", {})
                        manifest_out["hailo"]["config"] = {
                            "backend": str(hef_backend),
                            "wsl_distro": hef_wsl_distro,
                            "wsl_venv": str(hef_wsl_venv),
                            "opt_level": int(hef_opt_level),
                            "calib_count": int(hef_calib_count),
                            "calib_batch_size": int(hef_calib_bs),
                            "calib_dir": (str(hef_calib_dir).replace('\\', '/') if hef_calib_dir else None),
                            "fixup": bool(hef_fixup),
                            "force": bool(hef_force),
                            "keep_artifacts": bool(hef_keep),
                        }

                        def _hef_on_log(stream_name: str, line: str) -> None:
                            # Stream build output to the progress dialog.
                            try:
                                q.put(("hef", stream_name, line))
                            except Exception:
                                pass

                        for hw_arch in hef_targets:
                            hw_arch = str(hw_arch).strip()
                            if not hw_arch:
                                continue
                            tgt_out = {}
                            if hef_part1:
                                q.put(("stage", f"Building HEF ({hw_arch}) part1…"))
                                out_p1 = os.path.join(out_dir, "hailo", hw_arch, "part1")
                                os.makedirs(out_p1, exist_ok=True)
                                r1 = hailo_build_hef_auto(
                                    p1_path,
                                    backend=hef_backend,
                                    hw_arch=hw_arch,
                                    net_name=f"{base}_part1_b{b}",
                                    outdir=out_p1,
                                    fixup=hef_fixup,
                                    opt_level=int(hef_opt_level),
                                    calib_dir=hef_calib_dir,
                                    calib_count=int(hef_calib_count),
                                    calib_batch_size=int(hef_calib_bs),
                                    force=hef_force,
                                    keep_artifacts=hef_keep,
                                    wsl_distro=hef_wsl_distro,
                                    wsl_venv_activate=hef_wsl_venv,
                                    wsl_timeout_s=3600,
                                    on_log=_hef_on_log,
                                )
                                if r1.ok:
                                    rel = os.path.relpath(r1.hef_path or os.path.join(out_p1, "compiled.hef"), out_dir)
                                    tgt_out["part1"] = rel.replace('\\', '/')
                                    msg.append(f"Hailo HEF ({hw_arch}) part1: {tgt_out['part1']}")
                                else:
                                    tgt_out["part1_error"] = r1.error
                                    msg.append(f"Hailo HEF ({hw_arch}) part1 failed: {r1.error}")

                            if hef_part2:
                                q.put(("stage", f"Building HEF ({hw_arch}) part2…"))
                                out_p2 = os.path.join(out_dir, "hailo", hw_arch, "part2")
                                os.makedirs(out_p2, exist_ok=True)
                                r2 = hailo_build_hef_auto(
                                    p2_path,
                                    backend=hef_backend,
                                    hw_arch=hw_arch,
                                    net_name=f"{base}_part2_b{b}",
                                    outdir=out_p2,
                                    fixup=hef_fixup,
                                    opt_level=int(hef_opt_level),
                                    calib_dir=hef_calib_dir,
                                    calib_count=int(hef_calib_count),
                                    calib_batch_size=int(hef_calib_bs),
                                    force=hef_force,
                                    keep_artifacts=hef_keep,
                                    wsl_distro=hef_wsl_distro,
                                    wsl_venv_activate=hef_wsl_venv,
                                    wsl_timeout_s=3600,
                                    on_log=_hef_on_log,
                                )
                                if r2.ok:
                                    rel = os.path.relpath(r2.hef_path or os.path.join(out_p2, "compiled.hef"), out_dir)
                                    tgt_out["part2"] = rel.replace('\\', '/')
                                    msg.append(f"Hailo HEF ({hw_arch}) part2: {tgt_out['part2']}")
                                else:
                                    tgt_out["part2_error"] = r2.error
                                    msg.append(f"Hailo HEF ({hw_arch}) part2 failed: {r2.error}")

                            if tgt_out:
                                manifest_out["hailo"]["hefs"][hw_arch] = tgt_out

                # Write manifest last (so it contains runner/context/validation fields)
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest_out, f, indent=2)
                msg.append(f"Wrote: {manifest_path}")

                # Copy schema docs (plan/manifest schema files) into the split folder for portability.
                try:
                    src_schemas = Path(__file__).resolve().parent / "resources" / "schemas"
                    dst_schemas = Path(out_dir) / "schemas"
                    if src_schemas.exists():
                        if dst_schemas.exists():
                            shutil.rmtree(dst_schemas)
                        shutil.copytree(src_schemas, dst_schemas)
                        msg.append(f"Wrote: {str(dst_schemas)}")
                except Exception:
                    pass


                q.put(("done", "ok", "\n".join(msg)))
            except Exception as e:
                logging.exception("Split selected boundary failed")
                q.put(("done", "err", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()

        prog_re = re.compile(r"^(?P<stage>[A-Za-z][A-Za-z0-9 _\-]+):\s*(?P<pct>\d{1,3})%\|")
        last_prog: Dict[str, Any] = {"stage": None, "pct": None}

        def _append_log(line: str) -> None:
            if not line:
                return
            try:
                log_txt.configure(state="normal")
                log_txt.insert("end", line + "\n")
                # Trim to keep the UI responsive.
                try:
                    n_lines = int(log_txt.index("end-1c").split(".")[0])
                    if n_lines > 1200:
                        log_txt.delete("1.0", "200.0")
                except Exception:
                    pass
                log_txt.see("end")
                log_txt.configure(state="disabled")
            except Exception:
                pass

        def _is_interesting(line: str) -> bool:
            # Default to showing (almost) everything so users can see that work is ongoing.
            s = (line or "").strip()
            if not s:
                return False
            if s.startswith("__SPLITPOINT"):
                return False
            return True

        done_status: Optional[str] = None
        done_payload: str = ""

        def poll() -> None:
            nonlocal done_status, done_payload
            processed = 0
            try:
                while processed < 250:
                    item = q.get_nowait()
                    processed += 1

                    kind = item[0] if item else None
                    if kind == "done":
                        done_status = str(item[1]) if len(item) > 1 else "err"
                        done_payload = str(item[2]) if len(item) > 2 else ""
                        break
                    elif kind == "stage":
                        if len(item) > 1:
                            status_var.set(str(item[1]))
                            _append_log(f"[stage] {item[1]}")
                    elif kind == "log":
                        if len(item) > 1:
                            _append_log(str(item[1]))
                    elif kind == "hef":
                        # ('hef', stream, line)
                        line = str(item[2]) if len(item) > 2 else ""
                        s = line.strip()

                        # Update progress bar from tqdm-like lines.
                        m = prog_re.match(s)
                        if m:
                            stage = (m.group("stage") or "").strip()
                            try:
                                pct = int(m.group("pct"))
                            except Exception:
                                pct = None
                            if pct is not None:
                                # Switch from indeterminate to determinate once we have real progress.
                                try:
                                    if str(pb["mode"]) != "determinate":
                                        pb.stop()
                                        pb.configure(mode="determinate", maximum=100)
                                except Exception:
                                    pass
                                try:
                                    pb["value"] = max(0, min(100, pct))
                                except Exception:
                                    pass

                                # Avoid excessive UI churn.
                                if last_prog.get("stage") != stage or last_prog.get("pct") != pct:
                                    last_prog["stage"] = stage
                                    last_prog["pct"] = pct
                                    status_var.set(f"HEF build: {stage} {pct}%")

                        # Only log lines that are likely useful.
                        if _is_interesting(s):
                            _append_log(s)
                    else:
                        # Unknown event type: best-effort display.
                        try:
                            _append_log(str(item))
                        except Exception:
                            pass
            except queue.Empty:
                pass

            if done_status is None:
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

            if done_status == "ok":
                messagebox.showinfo("Split complete", done_payload)
            else:
                messagebox.showerror("Split failed", done_payload)

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
        model_path = self.gui_state.current_model_path or self.model_path
        if self.analysis is None or model_path is None:
            messagebox.showinfo("Nothing to benchmark", "Load a model and run an analysis first.")
            return
        if not self.current_picks:
            messagebox.showinfo(
                "No candidates",
                "No split candidates available. Try increasing Top-K and re-run Analyse.",
            )
            return
        if self._selected_boundary_index() is None:
            messagebox.showinfo(
                "Select a boundary",
                "Select a boundary row first; benchmark export is bound to split-ready state.",
            )
            return

        initial_out = self.default_output_dir or os.path.dirname(model_path)
        try:
            if self.default_output_dir:
                initial_out = str(ensure_workdir(Path(self.default_output_dir)).benchmark_sets)
        except Exception:
            pass
        out_parent = filedialog.askdirectory(title="Select parent folder for benchmark set", initialdir=initial_out)
        if not out_parent:
            return

        base = os.path.splitext(os.path.basename(model_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(out_parent, f"{base}_benchmark_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        # Pull analysis objects once (used for strict-boundary filtering and TeX/plot export).
        a = self.analysis
        strict_boundary = bool(self.var_strict_boundary.get())

        # Read pruning params from the current GUI state (same source as Analyse).
        _params_for_split = self._read_params()
        prune_skip_block = bool(getattr(_params_for_split, "prune_skip_block", False))
        skip_min_span = int(getattr(_params_for_split, "skip_min_span", 0) or 0)
        if skip_min_span < 0:
            raise ValueError("Min skip span must be an integer ≥ 0.")
        skip_allow_last_n = int(getattr(_params_for_split, "skip_allow_last_n", 0) or 0)
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

        # How many cases to generate?
        # Prefer the Benchmark tab entry (var_bench_topk). Fall back to a dialog only
        # if it's missing/invalid.
        default_k = min(20, len(ranked_candidates))
        k = None
        try:
            k = _safe_int((getattr(self, "var_bench_topk", tk.StringVar(value=str(default_k))).get() or "").strip())
        except Exception:
            k = None
        if k is None:
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
        if k < 1 or k > len(ranked_candidates):
            messagebox.showerror("Benchmark set", f"Top-K must be between 1 and {len(ranked_candidates)}.")
            return

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

        # ---------------- Accelerators to benchmark (suite plan) ----------------
        # Read once here so the worker thread does not touch Tk variables.
        acc_cpu = bool(getattr(self, "var_bench_acc_cpu", tk.BooleanVar(value=True)).get())
        acc_cuda = bool(getattr(self, "var_bench_acc_cuda", tk.BooleanVar(value=False)).get())
        acc_trt = bool(getattr(self, "var_bench_acc_tensorrt", tk.BooleanVar(value=False)).get())
        acc_h8 = bool(getattr(self, "var_bench_acc_hailo8", tk.BooleanVar(value=False)).get())
        acc_h10 = bool(getattr(self, "var_bench_acc_hailo10", tk.BooleanVar(value=False)).get())

        if not any([acc_cpu, acc_cuda, acc_trt, acc_h8, acc_h10]):
            # Defensive default (otherwise the suite is pointless).
            acc_cpu = True

        # Resolve Hailo hw_arch values from Split&Export settings (single source of truth).
        hailo8_hw = (getattr(self, "var_hailo_hef_hailo8_hw_arch", tk.StringVar(value="hailo8")).get() or "hailo8").strip()
        hailo10_hw = (getattr(self, "var_hailo_hef_hailo10_hw_arch", tk.StringVar(value="hailo10h")).get() or "hailo10h").strip()

        # Per-run image scaling (passed through to the runner harness).
        plan_image_scale = (getattr(self, "var_bench_image_scale", tk.StringVar(value="auto")).get() or "auto").strip().lower()
        if plan_image_scale not in {"auto", "norm", "raw", "imagenet", "clip"}:
            plan_image_scale = "auto"

        bench_plan_runs: List[Dict[str, Any]] = []
        if acc_cpu:
            bench_plan_runs.append({"id": "ort_cpu", "type": "onnxruntime", "provider": "cpu", "image_scale": plan_image_scale, "stage1": {"type": "onnxruntime", "provider": "cpu"}, "stage2": {"type": "onnxruntime", "provider": "cpu"}})
        if acc_cuda:
            bench_plan_runs.append({"id": "ort_cuda", "type": "onnxruntime", "provider": "cuda", "image_scale": plan_image_scale, "stage1": {"type": "onnxruntime", "provider": "cuda"}, "stage2": {"type": "onnxruntime", "provider": "cuda"}})
        if acc_trt:
            bench_plan_runs.append({"id": "ort_tensorrt", "type": "onnxruntime", "provider": "tensorrt", "image_scale": plan_image_scale, "stage1": {"type": "onnxruntime", "provider": "tensorrt"}, "stage2": {"type": "onnxruntime", "provider": "tensorrt"}})
        # Hailo runs: Use the selected hw_arch as the run id, so results are self-describing
        # (e.g. "hailo8l", "hailo10h").
        if acc_h8 and hailo8_hw:
            bench_plan_runs.append({"id": hailo8_hw, "type": "hailo", "hw_arch": hailo8_hw, "image_scale": plan_image_scale, "stage1": {"type": "hailo", "hw_arch": hailo8_hw}, "stage2": {"type": "hailo", "hw_arch": hailo8_hw}})
        if acc_h10 and hailo10_hw:
            bench_plan_runs.append({"id": hailo10_hw, "type": "hailo", "hw_arch": hailo10_hw, "image_scale": plan_image_scale, "stage1": {"type": "hailo", "hw_arch": hailo10_hw}, "stage2": {"type": "hailo", "hw_arch": hailo10_hw}})

        # ---------------- Hailo HEF build settings (reused by suite) ----------------
        hailo_selected = bool(acc_h8 or acc_h10)

        # For benchmark sets, we build HEFs if (and only if) Hailo is selected as an accelerator.
        # Which parts to build is derived from the benchmark plan (currently: both parts).
        hef_targets: List[str] = []
        if hailo_selected:
            if acc_h8 and hailo8_hw:
                hef_targets.append(hailo8_hw)
            if acc_h10 and hailo10_hw:
                hef_targets.append(hailo10_hw)

        # Build both parts for now (single-device benchmark plan). Matrix support can narrow this later.
        hef_part1 = bool(hailo_selected)
        hef_part2 = bool(hailo_selected)

        hef_opt_level = _safe_int((getattr(self, "var_hailo_hef_opt_level", tk.StringVar(value="1")).get() or "").strip()) or 1
        hef_calib_count = _safe_int((getattr(self, "var_hailo_hef_calib_count", tk.StringVar(value="64")).get() or "").strip()) or 64
        hef_calib_bs = _safe_int((getattr(self, "var_hailo_hef_calib_batch_size", tk.StringVar(value="8")).get() or "").strip()) or 8
        hef_calib_dir = (getattr(self, "var_hailo_hef_calib_dir", tk.StringVar(value="")).get() or "").strip() or None
        hef_force = bool(getattr(self, "var_hailo_hef_force", tk.BooleanVar(value=False)).get())
        hef_keep = bool(getattr(self, "var_hailo_hef_keep_artifacts", tk.BooleanVar(value=False)).get())

        # Backend selection reuses the Hailo feasibility-check backend controls.
        hef_backend = (getattr(self, "var_hailo_backend", tk.StringVar(value="auto")).get() or "auto").strip()
        hef_wsl_distro = (getattr(self, "var_hailo_wsl_distro", tk.StringVar(value="")).get() or "").strip() or None
        hef_wsl_venv = (getattr(self, "var_hailo_wsl_venv", tk.StringVar(value="auto")).get() or "auto").strip() or "auto"
        hef_fixup = bool(getattr(self, "var_hailo_fixup", tk.BooleanVar(value=True)).get())

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
        dlg.resizable(True, True)
        dlg.geometry("900x420")
        dlg.minsize(700, 260)

        lbl = ttk.Label(dlg, text=f"Generating benchmark set with up to {k} cases…")
        lbl.pack(padx=16, pady=(16, 8))

        pb = ttk.Progressbar(dlg, mode="determinate", maximum=max(1, k))
        pb.pack(fill="x", padx=16, pady=(0, 4))
        pb["value"] = 0

        lbl2 = ttk.Label(dlg, text="")
        lbl2.pack(fill="x", padx=16, pady=(0, 8))

        # Live log output (same idea as the Split & Export progress window)
        log_frame = ttk.Frame(dlg)
        log_frame.pack(fill="both", expand=True, padx=16, pady=(0, 8))

        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")

        log_txt = tk.Text(
            log_frame,
            height=12,
            wrap="none",
            font=("TkFixedFont", 9),
            yscrollcommand=log_scroll.set,
        )
        log_txt.pack(side="left", fill="both", expand=True)
        log_scroll.config(command=log_txt.yview)
        log_txt.configure(state="disabled")

        # Buttons: copy is useful for bug reports; close is enabled after completion.
        btn_row = ttk.Frame(dlg)
        btn_row.pack(fill="x", padx=16, pady=(0, 16))

        running_flag = {"running": True}

        def _copy_log() -> None:
            try:
                text = log_txt.get("1.0", "end-1c")
                dlg.clipboard_clear()
                dlg.clipboard_append(text)
                dlg.update_idletasks()
                _append_log("[ui] Copied log to clipboard")
            except Exception:
                pass

        def _close_dialog() -> None:
            # Prevent closing while the worker is still updating the UI.
            if running_flag.get("running"):
                return
            try:
                dlg.destroy()
            except Exception:
                pass

        btn_copy = ttk.Button(btn_row, text="Copy log", command=_copy_log)
        btn_copy.pack(side="left")

        btn_close = ttk.Button(btn_row, text="Close", command=_close_dialog)
        btn_close.pack(side="right")
        btn_close.state(["disabled"])

        dlg.protocol("WM_DELETE_WINDOW", _close_dialog)

        def _append_log(line: str) -> None:
            line = (line or "").rstrip("\n")
            if not line:
                return
            log_txt.configure(state="normal")
            log_txt.insert("end", line + "\n")
            log_txt.see("end")
            log_txt.configure(state="disabled")

        try:
            self.configure(cursor="watch")
            self.update_idletasks()
        except Exception:
            pass

        q: "queue.Queue[tuple]" = queue.Queue()

        def _write_benchmark_suite_script(dst_dir: str, bench_json_name: str = "benchmark_set.json") -> str:
            """Write benchmark suite runner script from a template resource."""
            from .gui.controller import write_benchmark_suite_script
            return write_benchmark_suite_script(dst_dir, bench_json_name=bench_json_name)

        def worker() -> None:
            try:
                cases = []
                errors = []
                made = 0

                def log(line: str) -> None:
                    # Worker thread -> UI thread
                    q.put(("log", line))

                log("=== Generating benchmark set ===")
                log(f"out_dir: {out_dir}")
                log(f"model_path: {self.model_path}")
                log(f"max_cases (top-k): {k}")

                # Make the benchmark folder self-contained for portability.
                # We copy the full model once into <benchmark_root>/models/ and reference it via
                # relative paths from each case directory.
                full_model_src = os.path.abspath(self.model_path)
                models_dir = os.path.join(out_dir, "models")
                os.makedirs(models_dir, exist_ok=True)
                full_model_dst = os.path.join(models_dir, os.path.basename(full_model_src))
                try:
                    if os.path.abspath(full_model_dst) != full_model_src:
                        shutil.copy2(full_model_src, full_model_dst)
                except Exception as e:
                    errors.append(f"full model copy failed: {type(e).__name__}: {e}")
                    full_model_dst = full_model_src
                log(f"full model (suite copy): {full_model_dst}")

                # Try candidates in ranked order and keep adding cases until we have k successful splits.
                # We respect the GUI "Min gap" setting to avoid exporting near-duplicate boundaries.
                gap = _safe_int(self.var_min_gap.get()) or 0
                chosen: List[int] = []

                picks_iter = list(ranked_candidates)

                log(f"min_gap: {gap}")
                log(f"ranked candidates considered: {len(picks_iter)}")

                for bi, b0 in enumerate(picks_iter):
                    if made >= k:
                        break
                    b = int(b0)

                    if gap > 0 and any(abs(b - bb) <= gap for bb in chosen):
                        log(f"b{b}: skip (min_gap)")
                        continue

                    # Update the dialog label while keeping the progress value at the
                    # number of successfully generated cases so far.
                    q.put(("prog", made, f"Splitting b{b} ({made+1}/{k})..."))
                    log(f"--- [{made+1}/{k}] Split boundary b{b} ---")
                    folder = f"b{b:0{pad}d}"
                    case_dir = os.path.join(out_dir, folder)
                    os.makedirs(case_dir, exist_ok=True)

                    try:
                        cut_tensors = asc.cut_tensors_for_boundary(order, nodes, b)
                    except Exception as e:
                        errors.append(f"b{b}: cut tensor error: {e}")
                        log(f"b{b}: cut tensor error: {e}")
                        q.put(("prog", made, f"b{b} (skip)"))
                        continue

                    if not cut_tensors:
                        errors.append(f"b{b}: no cut tensors")
                        log(f"b{b}: no cut tensors")
                        q.put(("prog", made, f"b{b} (skip)"))
                        continue

                    log(f"b{b}: cut tensors: {len(cut_tensors)}")

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
                        log(f"b{b}: split failed: {type(e).__name__}: {e}")
                        q.put(("prog", made, f"b{b} (split failed)"))
                        continue

                    log(f"b{b}: wrote {os.path.basename(p1_path)}")
                    log(f"b{b}: wrote {os.path.basename(p2_path)}")

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
                        # Portable paths (store forward slashes for cross-platform use).
                        'full_model': (
                            Path(os.path.relpath(full_model_dst, start=case_dir)).as_posix()
                            if os.path.exists(full_model_dst)
                            else str(full_model_src).replace('\\', '/')
                        ),
                        'full_model_source': str(full_model_src).replace('\\', '/'),
                        'part1': os.path.basename(p1_path).replace('\\', '/'),
                        'part2': os.path.basename(p2_path).replace('\\', '/'),
                        'created_at': datetime.now().isoformat(timespec='seconds'),
                    }
                    if isinstance(split_manifest, dict):
                        manifest_out.update(split_manifest)
                    # Normalized schema fields (v1). Keep legacy flat keys for backwards compatibility.
                    try:
                        manifest_out.setdefault("schema", "onnx-splitpoint/split-manifest")
                        manifest_out.setdefault("schema_version", 1)

                        manifest_out["models"] = {
                            "full": {
                                "path": str(manifest_out.get("full_model") or "").replace('\\', '/'),
                                "source": str(manifest_out.get("full_model_source") or "").replace('\\', '/'),
                            },
                            "part1": {
                                "path": str(manifest_out.get("part1_model") or manifest_out.get("part1") or "").replace('\\', '/')
                            },
                            "part2": {
                                "path": str(manifest_out.get("part2_model") or manifest_out.get("part2") or "").replace('\\', '/')
                            },
                        }

                        cut_full = manifest_out.get("cut_tensors_full") or manifest_out.get("cut_tensors") or []
                        cut_p1 = manifest_out.get("part1_cut_names") or []
                        cut_p2 = manifest_out.get("part2_cut_names") or []
                        manifest_out["cut"] = {
                            "full_names": list(cut_full) if isinstance(cut_full, list) else [],
                            "part1_names": list(cut_p1) if isinstance(cut_p1, list) else [],
                            "part2_names": list(cut_p2) if isinstance(cut_p2, list) else [],
                        }

                        manifest_out["io"] = {
                            "orig_inputs": list(manifest_out.get("orig_inputs") or []),
                            "orig_outputs": list(manifest_out.get("orig_outputs") or []),
                            "part1_external_inputs": list(manifest_out.get("part1_external_inputs") or []),
                            "part2_external_inputs": list(manifest_out.get("part2_external_inputs") or []),
                        }

                        links = []
                        if isinstance(cut_p1, list) and isinstance(cut_p2, list) and len(cut_p1) == len(cut_p2):
                            for a, bname in zip(cut_p1, cut_p2):
                                if a and bname:
                                    links.append({"from": str(a), "to": str(bname)})
                        manifest_out["pipeline"] = {
                            "stage1": "part1",
                            "stage2": "part2",
                            "links": links,
                        }
                    except Exception:
                        pass


                    # Context diagrams
                    semantic_label = None
                    try:
                        for c in (self.candidates or []):
                            if int(c.get("boundary", -1)) == int(b) and c.get("semantic"):
                                semantic_label = c.get("semantic")
                                break
                    except Exception:
                        semantic_label = None

                    llm_style = bool(self.var_llm_enable.get())
                    value_bytes_map = self.analysis.get("value_bytes") if isinstance(self.analysis, dict) else None

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
                                strict_boundary=bool(self.var_strict_boundary.get()),
                                include_external_inputs=(not llm_style),
                                semantic_label=semantic_label,
                                value_bytes_map=value_bytes_map,
                                force_matplotlib_fallback=llm_style,
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
                                strict_boundary=bool(self.var_strict_boundary.get()),
                                cut_flow_only=True,
                                include_internal_consumers=False,
                                include_external_inputs=False,
                                semantic_label=semantic_label,
                                value_bytes_map=value_bytes_map,
                                force_matplotlib_fallback=llm_style,
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

                    # Optional: Build Hailo HEFs (part1/part2) for selected targets.
                    if hef_targets and (hef_part1 or hef_part2):
                        log(
                            f"b{b}: Hailo HEF generation requested (backend={hef_backend}, targets={hef_targets}, part1={hef_part1}, part2={hef_part2})"
                        )
                        try:
                            from .hailo_backend import hailo_build_hef_auto
                        except Exception as e:
                            manifest_out['hailo_error'] = f"Hailo HEF build unavailable: {e}"
                            log(f"b{b}: Hailo HEF build unavailable: {e}")
                        else:
                            manifest_out.setdefault("hailo", {})
                            manifest_out["hailo"].setdefault("hefs", {})
                            manifest_out["hailo"]["config"] = {
                                "backend": str(hef_backend),
                                "wsl_distro": hef_wsl_distro,
                                "wsl_venv": str(hef_wsl_venv),
                                "opt_level": int(hef_opt_level),
                                "calib_count": int(hef_calib_count),
                                "calib_batch_size": int(hef_calib_bs),
                                "calib_dir": (str(hef_calib_dir).replace('\\', '/') if hef_calib_dir else None),
                                "fixup": bool(hef_fixup),
                                "force": bool(hef_force),
                                "keep_artifacts": bool(hef_keep),
                            }

                            for hw_arch in hef_targets:
                                hw_arch = str(hw_arch).strip()
                                if not hw_arch:
                                    continue

                                log(f"b{b}: build HEF for hw_arch={hw_arch}")

                                def _on_hef_log(stream: str, line: str, _b: int = b, _hw: str = hw_arch) -> None:
                                    # Forward Hailo build logs into the progress dialog.
                                    # Use parentheses to avoid clashing with the runner's [n/m] progress regex.
                                    q.put(("hef", stream, f"(b{_b} {_hw}) {line}"))

                                tgt_out = {}

                                if hef_part1:
                                    out_p1 = os.path.join(case_dir, "hailo", hw_arch, "part1")
                                    os.makedirs(out_p1, exist_ok=True)
                                    r1 = hailo_build_hef_auto(
                                        p1_path,
                                        backend=hef_backend,
                                        hw_arch=hw_arch,
                                        net_name=f"{base}_part1_b{b}",
                                        outdir=out_p1,
                                        fixup=hef_fixup,
                                        opt_level=int(hef_opt_level),
                                        calib_dir=hef_calib_dir,
                                        calib_count=int(hef_calib_count),
                                        calib_batch_size=int(hef_calib_bs),
                                        force=hef_force,
                                        keep_artifacts=hef_keep,
                                        wsl_distro=hef_wsl_distro,
                                        wsl_venv_activate=hef_wsl_venv,
                                        wsl_timeout_s=3600,
                                        on_log=_on_hef_log,
                                    )
                                    if r1.ok:
                                        rel = os.path.relpath(r1.hef_path or os.path.join(out_p1, "compiled.hef"), case_dir)
                                        tgt_out["part1"] = rel.replace('\\', '/')
                                        log(f"b{b}: HEF(part1,{hw_arch}) OK")
                                    else:
                                        tgt_out["part1_error"] = r1.error
                                        log(f"b{b}: HEF(part1,{hw_arch}) FAILED: {r1.error}")

                                if hef_part2:
                                    out_p2 = os.path.join(case_dir, "hailo", hw_arch, "part2")
                                    os.makedirs(out_p2, exist_ok=True)
                                    r2 = hailo_build_hef_auto(
                                        p2_path,
                                        backend=hef_backend,
                                        hw_arch=hw_arch,
                                        net_name=f"{base}_part2_b{b}",
                                        outdir=out_p2,
                                        fixup=hef_fixup,
                                        opt_level=int(hef_opt_level),
                                        calib_dir=hef_calib_dir,
                                        calib_count=int(hef_calib_count),
                                        calib_batch_size=int(hef_calib_bs),
                                        force=hef_force,
                                        keep_artifacts=hef_keep,
                                        wsl_distro=hef_wsl_distro,
                                        wsl_venv_activate=hef_wsl_venv,
                                        wsl_timeout_s=3600,
                                        on_log=_on_hef_log,
                                    )
                                    if r2.ok:
                                        rel = os.path.relpath(r2.hef_path or os.path.join(out_p2, "compiled.hef"), case_dir)
                                        tgt_out["part2"] = rel.replace('\\', '/')
                                        log(f"b{b}: HEF(part2,{hw_arch}) OK")
                                    else:
                                        tgt_out["part2_error"] = r2.error
                                        log(f"b{b}: HEF(part2,{hw_arch}) FAILED: {r2.error}")

                                if tgt_out:
                                    manifest_out["hailo"]["hefs"][hw_arch] = tgt_out

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
                    'schema': 'onnx-splitpoint/benchmark-set',
                    'schema_version': 1,
                    'tool': {'gui': __version__, 'core': getattr(asc, '__version__', '?')},
                    # Keep both a portable path (relative within the benchmark folder)
                    # and the original source path (useful for provenance).
                    'model': (
                        Path(os.path.relpath(full_model_dst, start=out_dir)).as_posix()
                        if os.path.exists(full_model_dst)
                        else str(full_model_src).replace('\\', '/')
                    ),
                    'model_source': str(full_model_src).replace('\\', '/'),
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

                # Suite runner plan (what to benchmark). This is a lightweight config so
                # the benchmark harness can be generic and re-usable across models.
                bench_plan = {
                    'schema': 'onnx-splitpoint/benchmark-plan',
                    'schema_version': 1,
                    'created_at': datetime.now().isoformat(timespec='seconds'),
                    'runs': list(bench_plan_runs),
                    # Future extension: allow different accelerators for stage1/stage2.
                    # Keep it empty for now, but the schema is reserved.
                    'matrix': [],
                }
                bench['plan'] = bench_plan

                # Also write a standalone plan file for external tools.
                plan_path = os.path.join(out_dir, 'benchmark_plan.json')
                try:
                    with open(plan_path, 'w', encoding='utf-8') as f:
                        json.dump(bench_plan, f, indent=2)
                except Exception:
                    pass

                # Record requested Hailo artifact generation at suite-level.
                if hef_targets and (hef_part1 or hef_part2):
                    bench['hailo'] = {
                        'targets': [str(x) for x in hef_targets],
                        'build': {'part1': bool(hef_part1), 'part2': bool(hef_part2)},
                        'config': {
                            'backend': str(hef_backend),
                            'wsl_distro': hef_wsl_distro,
                            'wsl_venv': str(hef_wsl_venv),
                            'opt_level': int(hef_opt_level),
                            'calib_count': int(hef_calib_count),
                            'calib_batch_size': int(hef_calib_bs),
                            'calib_dir': (str(hef_calib_dir).replace('\\', '/') if hef_calib_dir else None),
                            'fixup': bool(hef_fixup),
                            'force': bool(hef_force),
                            'keep_artifacts': bool(hef_keep),
                        },
                    }
                bench_path = os.path.join(out_dir, 'benchmark_set.json')
                with open(bench_path, 'w', encoding='utf-8') as f:
                    json.dump(bench, f, indent=2)

                # Write harness script
                harness_path = _write_benchmark_suite_script(out_dir, 'benchmark_set.json')

                # Copy schema docs for reproducibility (plan/manifest schema files).
                try:
                    src_schemas = Path(__file__).resolve().parent / "resources" / "schemas"
                    dst_schemas = Path(out_dir) / "schemas"
                    if src_schemas.exists():
                        if dst_schemas.exists():
                            shutil.rmtree(dst_schemas)
                        shutil.copytree(src_schemas, dst_schemas)
                except Exception:
                    pass


                # Small README
                readme = os.path.join(out_dir, 'README_BENCHMARK.txt')
                with open(readme, 'w', encoding='utf-8') as f:
                    txt = (
                        "Benchmark suite generated by the ONNX Split-Point Analyser.\n\n"
                        f"Model (portable): {bench.get('model')}\n"
                        f"Model (source):   {bench.get('model_source')}\n"
                        f"Cases: {len(cases)} (requested: {k})\n\n"
                        "Benchmark plan:\n"
                        "  - See benchmark_plan.json (and benchmark_set.json -> plan).\n\n"
                        "Next steps:\n"
                        "  1) (optional) install deps: pip install onnx onnxruntime numpy pillow matplotlib\n"
                        "  2) run ALL runs from the plan: python benchmark_suite.py\n"
                        "  3) run a single ORT provider: python benchmark_suite.py --provider cpu\n"
                        "     (or: --provider cuda / --provider tensorrt)\n"
                        "  4) to also generate human-readable outputs: add --image default --preset auto\n\n"
                        "Outputs:\n"
                        "  - benchmark_results_<tag>.csv / .json\n"
                        "    (tag is typically: <run-id>_<preset>, e.g. ort_cpu_auto)\n"
                        "\n"
                        "Paper-ready analysis exports (created during benchmark set generation):\n"
                        "  - analysis_plots/   (PDF + SVG)\n"
                        "  - analysis_tables/  (.tex, .csv, .json)\n"
                    )
                    if bench.get('hailo'):
                        txt += (
                            "\nHailo artifacts:\n"
                            "  - HEFs (if enabled) are under each case folder: case_*/hailo/<hw_arch>/(part1|part2)/compiled.hef\n"
                            "  - Suite-level config is recorded in benchmark_set.json under 'hailo'.\n"
                        )
                    f.write(txt)

                msg = []
                msg.append(f"Generated benchmark set: {out_dir}")
                msg.append(f"Cases: {len(cases)} (requested {k})")
                msg.append(f"Harness: {harness_path}")
                try:
                    run_ids = [str(r.get('id') or r.get('name') or '').strip() for r in bench_plan_runs if isinstance(r, dict)]
                    run_ids = [x for x in run_ids if x]
                    if run_ids:
                        msg.append(f"Plan runs: {', '.join(run_ids)}")
                except Exception:
                    pass
                if errors:
                    msg.append("\nWarnings:")
                    msg.extend([f"  - {e}" for e in errors[:10]])
                    if len(errors) > 10:
                        msg.append(f"  ... and {len(errors)-10} more")
                q.put(("ok", "\n".join(msg)))
            except Exception as e:
                logging.exception("Benchmark set generation failed")
                q.put(("err", f"{type(e).__name__}: {e}"))

        threading.Thread(target=worker, daemon=True).start()

        def poll() -> None:
            final_status: Optional[str] = None
            final_msg: str = ""

            # Drain the queue so log output stays responsive even when a lot of
            # lines arrive quickly (e.g. Hailo DFC compilation).
            while True:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break

                if not item:
                    continue

                status = str(item[0])

                if status == 'prog':
                    try:
                        made = int(item[1])
                        what = str(item[2]) if len(item) > 2 else ''
                        pb['value'] = made
                        lbl2.configure(text=f"{made}/{k}: {what}")
                    except Exception:
                        pass
                    continue

                if status in ('log', 'hef'):
                    try:
                        if status == 'log':
                            line = str(item[1]) if len(item) > 1 else ''
                        else:
                            # ('hef', stream, line)
                            line = str(item[2]) if len(item) > 2 else ''
                        _append_log(line)
                        if status == 'hef' and line:
                            lbl2.configure(text=line[:220])
                    except Exception:
                        pass
                    continue

                if status in ('msg', 'note'):
                    try:
                        what = str(item[1]) if len(item) > 1 else ''
                        lbl2.configure(text=what)
                    except Exception:
                        pass
                    continue

                if status in ('ok', 'err'):
                    final_status = status
                    final_msg = str(item[1]) if len(item) > 1 else ''
                    break

            try:
                self.update_idletasks()
            except Exception:
                pass

            if not final_status:
                self.after(80, poll)
                return

            # final: keep the dialog open so logs can be inspected/copied
            running_flag["running"] = False

            try:
                dlg.grab_release()
            except Exception:
                pass

            try:
                btn_close.state(["!disabled"])
            except Exception:
                try:
                    btn_close.configure(state="normal")
                except Exception:
                    pass

            try:
                dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
            except Exception:
                pass

            try:
                self.configure(cursor="")
            except Exception:
                pass

            if final_status == 'ok':
                messagebox.showinfo("Benchmark set created", final_msg)
            else:
                messagebox.showerror("Benchmark set failed", final_msg)

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
        plots_subdir: Optional[str] = None,
        tables_subdir: Optional[str] = None,
    ) -> None:
        """Write paper-ready assets (plots, tables, and metadata) into a folder.

        This helper is used for:
        - benchmark set creation (always exports both PDF+SVG + TeX table + metadata)
        - optional manual exports
        """

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = out_dir
        tables_dir = out_dir
        if plots_subdir:
            plots_dir = out_dir / str(plots_subdir)
            plots_dir.mkdir(parents=True, exist_ok=True)
        if tables_subdir:
            tables_dir = out_dir / str(tables_subdir)
            tables_dir.mkdir(parents=True, exist_ok=True)

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
            out_path = plots_dir / f"{fname_stem}.{fmt}"
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
            fig.savefig(str(plots_dir / f"analysis_plots_overview.{fmt}"), format=fmt, bbox_inches="tight")

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
                (tables_dir / "split_candidates.tex").write_text(tex, encoding="utf-8")

                # Also provide a CSV version of the same top-k table.
                # This is handy for plotting/aggregation without LaTeX parsing.
                sem_labels = a.get("semantic_labels_by_boundary") or []
                costs_bytes = a.get("costs_bytes") or []
                counts_all = a.get("crossing_counts_all") or []
                unknown_counts = a.get("unknown_crossing_counts") or []
                flops_left_prefix = a.get("flops_left_prefix") or []
                total_flops = float(a.get("total_flops") or 0.0)

                out_csv = tables_dir / "split_candidates.csv"
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "boundary",
                        "semantic",
                        "comm_bytes",
                        "comm_MiB",
                        "crossing_tensors_all",
                        "unknown_crossing_tensors",
                        "flops_left_GFlop",
                        "flops_right_GFlop",
                    ])
                    for b in picks:
                        bi = int(b)
                        sem = str(sem_labels[bi]) if bi < len(sem_labels) else ""
                        comm_b = int(costs_bytes[bi]) if bi < len(costs_bytes) else 0
                        comm_mib = float(comm_b) / (1024.0**2)
                        nt = int(counts_all[bi]) if bi < len(counts_all) else 0
                        nu = int(unknown_counts[bi]) if bi < len(unknown_counts) else 0
                        fl_l = float(flops_left_prefix[bi]) / 1e9 if bi < len(flops_left_prefix) else 0.0
                        fl_r = float(total_flops - float(flops_left_prefix[bi])) / 1e9 if bi < len(flops_left_prefix) else 0.0
                        w.writerow([bi, sem, comm_b, f"{comm_mib:.6f}", nt, nu, f"{fl_l:.6f}", f"{fl_r:.6f}"])
            except Exception as e:
                print(f"[warn] Failed to export TeX candidate table: {e}")

        if include_json:
            # System / workload separation + Pareto export (dissertation-friendly)
            try:
                if p is not None:
                    sys = self._build_system_spec(p)
                    (tables_dir / "system_config.json").write_text(json.dumps(asdict(sys), indent=2), encoding="utf-8")
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
                (tables_dir / "workload_profile.json").write_text(json.dumps(workload, indent=2), encoding="utf-8")
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
                (tables_dir / "candidate_pruning.json").write_text(json.dumps(pruning, indent=2), encoding="utf-8")
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

                out_csv = tables_dir / "pareto_export.csv"
                with open(out_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "boundary",
                        "semantic",
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
                    sem_labels = a.get("semantic_labels_by_boundary") or []
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
                            str(sem_labels[b]) if b < len(sem_labels) else "",
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
            plots_subdir="analysis_plots",
            tables_subdir="analysis_tables",
        )

    def _export_tex_table(self):
        if self.analysis is None or not self.current_picks:
            messagebox.showinfo("Nothing to export", "Run an analysis first.")
            return

        default_name = "split_candidates.tex"
        if self.model_path:
            base = os.path.splitext(os.path.basename(model_path))[0]
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
        sem_labels = a.get("semantic_labels_by_boundary") or []

        def comm_mib(b: int) -> float:
            return float(costs[b]) / (1024.0**2)

        def gflops_left(b: int) -> float:
            return float(flops_left_prefix[b]) / 1e9

        def gflops_right(b: int) -> float:
            return float(total_flops - flops_left_prefix[b]) / 1e9

        any_unknown = any(int(unknown[b]) > 0 for b in picks if b < len(unknown))

        def semantic_label(b: int) -> str:
            try:
                s = sem_labels[int(b)] if int(b) < len(sem_labels) else ""
            except Exception:
                s = ""
            return _latex_escape(str(s))

        lines: List[str] = []
        lines.append("% Generated by analyse_and_split_gui.py")
        lines.append("% Requires: \\usepackage{booktabs}")
        lines.append("\\begin{table}[t]")
        lines.append("  \\centering")
        lines.append("  \\small")
        lines.append("  \\setlength{\\tabcolsep}{4pt}")
        lines.append("  \\begin{tabular}{@{}r l r r r r@{}}")
        lines.append("    \\toprule")
        # NOTE: we need a literal LaTeX line break "\\" at the end of the header/rows.
        # In Python string literals that means writing "\\\\".
        lines.append("    Boundary & Semantic & Comm (MiB) & \\#Tensors & $F_L$ (GFLOP) & $F_R$ (GFLOP) \\\\")
        lines.append("    \\midrule")

        for b in picks:
            lines.append(
                f"    {int(b)} & {semantic_label(b)} & {comm_mib(b):.3f} & {int(counts_all[b])} & {gflops_left(b):.3f} & {gflops_right(b):.3f} \\\\")

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


    # ------------------------ Memory Fit (refactor bridge) ------------------------

    def _compute_memory_fit_estimate_from_selection(self, boundary: int):
        # Return a MemoryFitWidget-compatible estimate dict for the given boundary.
        #
        # Used by the refactored Analysis panel (Candidate Inspector) to display RAM
        # usage against the currently selected accelerators.
        #
        # Structure:
        #   {"left": {"name": str, "ram_limit_mb": float, "total_mb": float, "fits": bool},
        #    "right": {...}}
        #
        # Units are MiB (MB = 1024^2 bytes). We recompute runtime overhead based on
        # the *current* device selection, so changing accelerators updates the
        # estimate without re-running analysis.
        try:
            b = int(boundary)
        except Exception:
            return None

        mem_by_b = getattr(self, "memory_by_boundary", None)
        if not isinstance(mem_by_b, dict):
            return None

        mem = mem_by_b.get(b)
        if not isinstance(mem, dict):
            return None

        left_base = mem.get("left") or {}
        right_base = mem.get("right") or {}

        def _base_sum(d: dict) -> float:
            try:
                return float(d.get("weights_mb") or 0.0) + float(d.get("peak_act_mb") or 0.0) + float(d.get("const_mb") or 0.0)
            except Exception:
                return 0.0

        # Current accelerator selection (from Hardware tab)
        try:
            left_name = str(self.var_memf_left_accel.get())
        except Exception:
            left_name = ""
        try:
            right_name = str(self.var_memf_right_accel.get())
        except Exception:
            right_name = ""

        try:
            left_spec = self._get_accel_spec_by_name(left_name)
        except Exception:
            left_spec = None
        try:
            right_spec = self._get_accel_spec_by_name(right_name)
        except Exception:
            right_spec = None

        left_limit_mb = float((left_spec or {}).get("ram_limit_mb") or 0.0)
        right_limit_mb = float((right_spec or {}).get("ram_limit_mb") or 0.0)
        left_over_mb = float((left_spec or {}).get("runtime_overhead_mb") or 0.0)
        right_over_mb = float((right_spec or {}).get("runtime_overhead_mb") or 0.0)

        left_total_mb = _base_sum(left_base) + left_over_mb
        right_total_mb = _base_sum(right_base) + right_over_mb

        return {
            "left": {
                "name": left_name,
                "ram_limit_mb": left_limit_mb,
                "total_mb": left_total_mb,
                "fits": bool(left_limit_mb > 0 and left_total_mb <= left_limit_mb),
            },
            "right": {
                "name": right_name,
                "ram_limit_mb": right_limit_mb,
                "total_mb": right_total_mb,
                "fits": bool(right_limit_mb > 0 and right_total_mb <= right_limit_mb),
            },
        }

    # ----------------------------- Utilities -----------------------------

    def _default_hailo_cache_path(self) -> Path:
        """Return the default persistent cache path for Hailo parse-check results."""
        base = Path.home() / ".onnx_splitpoint_tool"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; fall back to a temp dir.
            base = Path(tempfile.gettempdir()) / "onnx_splitpoint_tool"
            base.mkdir(parents=True, exist_ok=True)
        return base / "hailo_parse_cache.json"

    def _load_hailo_cache(self) -> None:
        path = self._hailo_cache_path
        self._hailo_cache = {}
        try:
            if path.is_file():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("entries"), dict):
                    self._hailo_cache = data["entries"]  # type: ignore[assignment]
        except Exception:
            # Corrupt cache -> ignore.
            self._hailo_cache = {}

    def _save_hailo_cache(self) -> None:
        if not self._hailo_cache_dirty:
            return
        try:
            payload = {
                "version": 1,
                "entries": self._hailo_cache,
            }
            tmp = self._hailo_cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(self._hailo_cache_path)
            self._hailo_cache_dirty = False
        except Exception:
            # Best-effort; ignore.
            pass

    @staticmethod
    def _sha1_bytes(data: bytes) -> str:
        return hashlib.sha1(data).hexdigest()

    def _hailo_cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            v = self._hailo_cache.get(key)
            if isinstance(v, dict):
                return v
        except Exception:
            pass
        return None

    def _hailo_cache_put(self, key: str, value: Dict[str, Any]) -> None:
        self._hailo_cache[key] = value
        self._hailo_cache_dirty = True

    def _clear_results(self):
        self.analysis = None
        self.current_picks = []
        self.analysis_result = None
        self.selected_candidate = None
        self._last_params = None
        self.memory_by_boundary = {}
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
        self.var_memf_left_text.set("Left: n/a")
        self.var_memf_right_text.set("Right: n/a")
        self._set_ui_state(self._infer_ui_state())


def main():
    """Compatibility entrypoint delegating to ``onnx_splitpoint_tool.gui.app``."""
    from .gui.app import main as _new_main

    _new_main()


if __name__ == "__main__":
    main()
