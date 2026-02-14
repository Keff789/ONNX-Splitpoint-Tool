"""Context helpers and compact report export for LLM validation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_META_TOKENS = (
    "mask",
    "position",
    "pos",
    "rope",
    "cache",
    "past",
    "present",
    "token",
    "ids",
    "shape",
    "len",
)


def run_case(case_dir: Path, provider: str, image: str, preset: str, warmup: int, runs: int, timeout_s: Optional[int], case_meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Compatibility runner hook.

    Uses the legacy benchmark runner only when available to avoid import cycles.
    """
    try:
        from ...gui_app import _run_case  # type: ignore
    except Exception:
        return None
    return _run_case(case_dir, provider, image, preset, warmup, runs, timeout_s, case_meta)


def _dtype_from_vi(vi: Any) -> str:
    try:
        import onnx

        et = int(vi.type.tensor_type.elem_type)
        return str(onnx.TensorProto.DataType.Name(et))
    except Exception:
        return "?"


def _shape_from_vi(vi: Any) -> List[Any]:
    out: List[Any] = []
    try:
        dims = vi.type.tensor_type.shape.dim
    except Exception:
        return out
    for d in dims:
        if getattr(d, "dim_value", 0):
            out.append(int(d.dim_value))
        elif getattr(d, "dim_param", ""):
            out.append(str(d.dim_param))
        else:
            out.append("?")
    return out


def _estimate_bytes(shape: Sequence[Any], dtype: str) -> Optional[int]:
    if not shape or any(not isinstance(x, int) or x <= 0 for x in shape):
        return None
    bpe = {
        "FLOAT": 4,
        "FLOAT16": 2,
        "BFLOAT16": 2,
        "DOUBLE": 8,
        "INT8": 1,
        "UINT8": 1,
        "INT16": 2,
        "UINT16": 2,
        "INT32": 4,
        "UINT32": 4,
        "INT64": 8,
        "UINT64": 8,
        "BOOL": 1,
    }.get(dtype, 4)
    n = 1
    for x in shape:
        n *= int(x)
    return int(n * bpe)


def _classify_tensor(name: str, initializers: Iterable[str]) -> str:
    if name in set(initializers):
        return "Meta"
    low = (name or "").lower()
    if any(tok in low for tok in _META_TOKENS):
        return "Meta"
    return "Activation"


def _flow_summary(analysis: Mapping[str, Any], boundary: int, window: int = 2) -> Dict[str, Any]:
    nodes = list(analysis.get("nodes") or [])
    order = list(analysis.get("order") or [])
    if not nodes or not order or boundary < 0 or boundary >= len(order) - 1:
        return {"left": [], "right": []}

    def _node_repr(n: Any) -> Dict[str, Any]:
        return {
            "name": str(getattr(n, "name", "") or "<anon>"),
            "op_type": str(getattr(n, "op_type", "")),
            "outputs": [str(x) for x in list(getattr(n, "output", []) or [])[:3]],
        }

    left_ids = order[max(0, boundary - window + 1) : boundary + 1]
    right_ids = order[boundary + 1 : min(len(order), boundary + 1 + window)]
    return {
        "left": [_node_repr(nodes[i]) for i in left_ids],
        "right": [_node_repr(nodes[i]) for i in right_ids],
    }


def export_compact_context_report(
    output_dir: Path,
    *,
    analysis: Mapping[str, Any],
    candidate: Mapping[str, Any],
    include_flow_summary: bool = True,
) -> Dict[str, Path]:
    """Export a compact LLM context report as JSON + PDF.

    The report is intentionally compact and avoids embedding huge DOT graphs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = int(candidate.get("boundary", -1))
    sem_labels = list(analysis.get("semantic_labels_by_boundary") or [])
    semantic = str(candidate.get("semantic") or (sem_labels[boundary] if 0 <= boundary < len(sem_labels) else ""))
    llm_hints = dict(analysis.get("llm_hints") or {})

    vimap = analysis.get("vimap") or {}
    value_bytes = analysis.get("value_bytes") or {}
    cut_tensors = list(candidate.get("cut_tensors") or [])
    init_names = set(analysis.get("initializer_names") or [])

    tensor_rows: List[Dict[str, Any]] = []
    for name in cut_tensors:
        vi = vimap.get(name)
        dtype = _dtype_from_vi(vi) if vi is not None else "?"
        shape = _shape_from_vi(vi) if vi is not None else []
        est = int(value_bytes.get(name, 0) or 0) or _estimate_bytes(shape, dtype)
        tensor_rows.append(
            {
                "name": str(name),
                "kind": _classify_tensor(str(name), init_names),
                "dtype": dtype,
                "shape": shape,
                "bytes_estimate": int(est) if est else None,
            }
        )

    header = {
        "boundary": boundary,
        "semantic": semantic,
        "mode": str(analysis.get("llm_mode") or "n/a"),
        "shapes": {
            "batch": llm_hints.get("batch"),
            "seq_len": llm_hints.get("seq_len"),
            "past_len": llm_hints.get("past_len"),
            "total_len": llm_hints.get("total_len"),
        },
    }

    clean_flags = {
        "symbol": str(candidate.get("clean_symbol") or "?"),
        "flags": list(candidate.get("clean_flags") or []),
        "reasons": [x.strip() for x in str(candidate.get("clean_tooltip") or "").splitlines() if x.strip() and x.strip() != "Clean split"],
    }

    payload: Dict[str, Any] = {
        "header": header,
        "cut_tensors": tensor_rows,
        "clean": clean_flags,
    }
    if include_flow_summary:
        payload["flow_summary"] = _flow_summary(analysis, boundary)

    json_path = output_dir / f"llm_compact_context_b{boundary}.json"
    pdf_path = output_dir / f"llm_compact_context_b{boundary}.pdf"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        y = 0.98
        ax.text(0.02, y, "LLM Compact Context Report", fontsize=14, fontweight="bold", transform=ax.transAxes)
        y -= 0.05
        for k, v in header.items():
            ax.text(0.02, y, f"{k}: {v}", fontsize=9.5, transform=ax.transAxes)
            y -= 0.03
        y -= 0.01
        ax.text(0.02, y, "Clean Flags", fontsize=11, fontweight="bold", transform=ax.transAxes)
        y -= 0.03
        ax.text(0.02, y, f"symbol={clean_flags['symbol']} flags={clean_flags['flags']}", fontsize=9, transform=ax.transAxes)
        y -= 0.03
        if clean_flags["reasons"]:
            for r in clean_flags["reasons"][:5]:
                ax.text(0.04, y, f"- {r}", fontsize=8.5, transform=ax.transAxes)
                y -= 0.025
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(11.69, 8.27))
        ax2.axis("off")
        cols = ["Tensor", "Activation/Meta", "dtype", "shape", "bytes est."]
        rows = [
            [
                r["name"],
                r["kind"],
                r["dtype"],
                str(r["shape"]),
                "?" if r["bytes_estimate"] is None else str(r["bytes_estimate"]),
            ]
            for r in tensor_rows
        ]
        if not rows:
            rows = [["<none>", "-", "-", "-", "-"]]
        tbl = ax2.table(cellText=rows[:30], colLabels=cols, loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.0, 1.2)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    return {"json": json_path, "pdf": pdf_path}


__all__ = ["run_case", "export_compact_context_report"]
