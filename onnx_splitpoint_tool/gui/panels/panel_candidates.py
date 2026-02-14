"""Candidate table helpers and (incremental) panel scaffold."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from tkinter import ttk
from typing import List, Mapping, Optional, Sequence, Set

import onnx

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


__all__ = [
    "CUT_INSIDE_MLP",
    "CUT_INSIDE_ATTN",
    "CUT_CONTAINS_PRESENT",
    "CUT_TOO_MANY_FLOATS",
    "CandidateCleanStatus",
    "compute_candidate_clean_status",
    "build_panel",
]
