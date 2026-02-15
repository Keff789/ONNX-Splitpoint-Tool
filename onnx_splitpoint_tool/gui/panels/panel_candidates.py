"""Candidate table helpers and split-view candidate inspector."""

from __future__ import annotations

import re
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import ttk
from typing import List, Mapping, Optional, Sequence, Set

import onnx

from ..widgets.collapsible_section import CollapsibleSection

CUT_INSIDE_MLP = "CUT_INSIDE_MLP"
CUT_INSIDE_ATTN = "CUT_INSIDE_ATTN"
CUT_CONTAINS_PRESENT = "CUT_CONTAINS_PRESENT"
CUT_TOO_MANY_FLOATS = "CUT_TOO_MANY_FLOATS"


@dataclass
class CandidateCleanStatus:
    flags: Set[str] = field(default_factory=set)
    reasons: List[str] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        if not self.flags:
            return "✅"
        severe = {CUT_INSIDE_MLP, CUT_INSIDE_ATTN, CUT_CONTAINS_PRESENT}
        if any(f in severe for f in self.flags):
            return "❌"
        return "⚠️"


def _is_float_dtype(vimap: Mapping[str, onnx.ValueInfoProto], tensor_name: str) -> bool:
    vi = vimap.get(tensor_name)
    if vi is None:
        return False
    try:
        elem_type = int(vi.type.tensor_type.elem_type)
    except Exception:
        return False
    return elem_type in {
        int(onnx.TensorProto.FLOAT),
        int(onnx.TensorProto.FLOAT16),
        int(onnx.TensorProto.DOUBLE),
        int(onnx.TensorProto.BFLOAT16),
        int(onnx.TensorProto.FLOAT8E4M3FN),
        int(onnx.TensorProto.FLOAT8E4M3FNUZ),
        int(onnx.TensorProto.FLOAT8E5M2),
        int(onnx.TensorProto.FLOAT8E5M2FNUZ),
    }


def compute_candidate_clean_status(
    *,
    boundary: int,
    semantic_label: str,
    left_op: str,
    right_op: str,
    cut_tensor_names: Sequence[str],
    vimap: Optional[Mapping[str, onnx.ValueInfoProto]] = None,
    max_float_tensors: int = 6,
) -> CandidateCleanStatus:
    """Compute clean/dirty flags for one candidate boundary."""
    flags: Set[str] = set()
    reasons: List[str] = []

    joined = " ".join(
        [semantic_label or "", left_op or "", right_op or "", *[str(t) for t in cut_tensor_names]]
    ).lower()

    if re.search(r"\b(mlp|ffn|feed[\s_\-]?forward|dense_h_to_4h|dense_4h_to_h)\b", joined):
        flags.add(CUT_INSIDE_MLP)
        reasons.append("Split liegt vermutlich innerhalb eines MLP/FFN-Blocks.")

    if re.search(r"\b(attn|attention|qkv|query|key|value|self_attn)\b", joined):
        flags.add(CUT_INSIDE_ATTN)
        reasons.append("Split liegt vermutlich innerhalb eines Attention-Blocks.")

    if any(re.search(r"(present|past[_\.]?key[_\.]?values?|kv[_\.]?cache)", str(t), re.IGNORECASE) for t in cut_tensor_names):
        flags.add(CUT_CONTAINS_PRESENT)
        reasons.append("Cut enthält present/past(KV)-Tensors.")

    if vimap:
        float_count = sum(1 for t in cut_tensor_names if _is_float_dtype(vimap, t))
        if float_count > int(max_float_tensors):
            flags.add(CUT_TOO_MANY_FLOATS)
            reasons.append(f"Viele Float-Aktivierungen im Cut ({float_count}>{int(max_float_tensors)}).")

    return CandidateCleanStatus(flags=flags, reasons=reasons)


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    return frame


def mount_split_view(parent, app) -> ttk.Frame:
    """Create a split-view: left candidate area, right inspector panel."""
    host = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
    host.pack(fill="both", expand=True)

    left = ttk.Frame(host)
    left.columnconfigure(0, weight=1)
    left.rowconfigure(0, weight=1)
    host.add(left, weight=3)

    right = ttk.LabelFrame(host, text="Candidate Inspector")
    right.columnconfigure(0, weight=1)
    right.rowconfigure(2, weight=1)
    host.add(right, weight=2)

    _build_inspector(right, app)
    host.left_host = left  # type: ignore[attr-defined]
    host.inspector_host = right  # type: ignore[attr-defined]
    return host


def _build_inspector(parent: ttk.Frame, app) -> None:
    summary = ttk.LabelFrame(parent, text="Summary")
    summary.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
    summary.columnconfigure(1, weight=1)

    vars_map = {
        "boundary": tk.StringVar(value="–"),
        "semantic": tk.StringVar(value="–"),
        "compute": tk.StringVar(value="–"),
        "cut": tk.StringVar(value="–"),
        "counts": tk.StringVar(value="–"),
        "llm": tk.StringVar(value="–"),
        "proxy": tk.StringVar(value="Proxy: –"),
    }
    rows = [
        ("Boundary", "boundary"),
        ("Semantic transition", "semantic"),
        ("Compute L/R", "compute"),
        ("Cut MB", "cut"),
        ("Tensor counts", "counts"),
    ]
    for r, (lbl, key) in enumerate(rows):
        ttk.Label(summary, text=f"{lbl}:").grid(row=r, column=0, sticky="nw", padx=(6, 8), pady=2)
        ttk.Label(summary, textvariable=vars_map[key], wraplength=360, justify="left").grid(row=r, column=1, sticky="ew", padx=(0, 6), pady=2)

    llm_box = ttk.LabelFrame(parent, text="LLM Comm Breakdown")
    llm_box.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
    ttk.Label(llm_box, textvariable=vars_map["llm"], justify="left", wraplength=380).pack(anchor="w", padx=6, pady=(6, 2))
    ttk.Label(llm_box, textvariable=vars_map["proxy"], foreground="#6b4f00", wraplength=380).pack(anchor="w", padx=6, pady=(0, 6))

    notebook = ttk.Notebook(parent)
    notebook.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 6))
    lists = {}
    for name in ("Activations", "Meta", "Constants"):
        tab = ttk.Frame(notebook)
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)
        notebook.add(tab, text=name)
        if name == "Constants":
            sec = CollapsibleSection(tab, "Constants", expanded=False)
            sec.grid(row=0, column=0, sticky="nsew")
            holder = sec.body
            holder.columnconfigure(0, weight=1)
        else:
            holder = tab
        tv = ttk.Treeview(holder, columns=("name", "size"), show="headings", height=9)
        tv.heading("name", text="Tensor")
        tv.heading("size", text="MB")
        tv.column("name", width=280, anchor=tk.W)
        tv.column("size", width=80, anchor=tk.E)
        holder.rowconfigure(0, weight=1)
        tv.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(holder, orient="vertical", command=tv.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        tv.configure(yscrollcommand=scroll.set)
        lists[name.lower()] = tv

    actions = ttk.Frame(parent)
    actions.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
    ttk.Button(actions, text="Split selected…", command=getattr(app, "_split_selected_boundary", None)).pack(side=tk.LEFT)
    ttk.Button(actions, text="Export context…", command=getattr(app, "_split_selected_boundary", None)).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(actions, text="Benchmark this split…", command=getattr(app, "_generate_benchmark_set", None)).pack(side=tk.LEFT, padx=(8, 0))

    def _classify_tensor(name: str, initializers: Set[str]) -> str:
        n = (name or "").lower()
        if name in initializers:
            return "constants"
        if any(k in n for k in ("mask", "pos", "position", "rope", "cache", "past", "present", "token", "ids", "shape", "len")):
            return "meta"
        return "activations"

    def _update(candidate=None):
        cand = candidate if candidate is not None else getattr(app, "selected_candidate", None)
        for tv in lists.values():
            tv.delete(*tv.get_children())
        if cand is None:
            for k in vars_map:
                vars_map[k].set("–" if k != "proxy" else "Proxy: –")
            return

        b = int(getattr(cand, "boundary_id", -1))
        row = next((r for r in getattr(app, "_candidate_rows", []) if int(r.get("boundary", -1)) == b), {})
        analysis = getattr(app, "analysis", {}) if isinstance(getattr(app, "analysis", {}), dict) else {}
        costs = analysis.get("costs_bytes") or []
        unknown_counts = analysis.get("unknown_crossing_counts") or []
        proxy_mb = float(analysis.get("unknown_tensor_proxy_mb", 0.0) or 0.0)
        proxy_kb = float(analysis.get("unknown_tensor_proxy_kb_int", 0.0) or 0.0)
        value_bytes = analysis.get("value_bytes") or {}
        inits = set(analysis.get("initializer_names") or [])

        cut_tensors = list(getattr(cand, "cut_tensors", []) or row.get("cut_tensors") or [])
        unknown_n = int(unknown_counts[b]) if b < len(unknown_counts) else int(row.get("unknown_count", 0) or 0)
        cut_mb = (float(costs[b]) / 1e6) if b < len(costs) else float(row.get("cut_mb_val", 0.0) or 0.0)

        vars_map["boundary"].set(str(b))
        vars_map["semantic"].set(str(getattr(cand, "semantic_label", "") or row.get("semantic", "–")))
        vars_map["compute"].set(f"{row.get('gflops_left', '–')} / {row.get('gflops_right', '–')} GFLOPs")
        vars_map["cut"].set(f"{cut_mb:.3f} MB")
        vars_map["counts"].set(f"total={len(cut_tensors)}, unknown={unknown_n}")

        sums = {"activations": 0.0, "meta": 0.0, "constants": 0.0, "unknown": float(unknown_n) * proxy_mb}
        for t in cut_tensors:
            grp = _classify_tensor(str(t), inits)
            size_mb = float(value_bytes.get(t, 0.0) or 0.0) / 1e6
            sums[grp] += size_mb
            lists[grp].insert("", "end", values=(t, f"{size_mb:.3f}" if size_mb > 0 else "?"))

        vars_map["llm"].set(
            f"Hidden/Act: {sums['activations']:.3f} MB\n"
            f"Meta: {sums['meta']:.3f} MB\n"
            f"Unknown (proxy): {sums['unknown']:.3f} MB"
        )
        vars_map["proxy"].set(f"Proxy-Hinweis: float={proxy_mb:g} MB/Tensor, int/bool={proxy_kb:g} KB/Tensor")

    _update(None)
    if hasattr(app, "events"):
        app.events.on_candidate_selected(_update)


def build_candidate_inspector(parent: ttk.Frame, app) -> None:
    """Build only the right-side candidate inspector into an existing parent."""
    _build_inspector(parent, app)


__all__ = [
    "CUT_INSIDE_MLP",
    "CUT_INSIDE_ATTN",
    "CUT_CONTAINS_PRESENT",
    "CUT_TOO_MANY_FLOATS",
    "CandidateCleanStatus",
    "compute_candidate_clean_status",
    "build_panel",
    "mount_split_view",
    "build_candidate_inspector",
]
