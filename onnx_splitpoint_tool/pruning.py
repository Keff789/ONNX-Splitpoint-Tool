"""Candidate pruning heuristics.

This module is used by the GUI to reduce the number of candidate split boundaries.

The main feature is "skip/block pruning" which tries to avoid split boundaries that
sit *inside* a residual/skip block. For transformer models, long-range edges from
*global* tensors (attention masks, RoPE caches, KV-cache plumbing) can be
mis-detected as skip connections. We therefore add a conservative filter that
classifies tensors that are derived purely from external inputs/constants through
trivial elementwise/shape ops as "external flow" and ignores them for skip-block
construction.

Additionally, transformer layers often contain multiple residual merges that
"touch" each other. For graphs that include a layer index in node/tensor names
(e.g. "/model/layers.20/..."), we can safely merge touching residual blocks *within
that same layer* to yield a single "layer block". This tends to remove dirty split
candidates that cut between attention and MLP inside one layer.
"""

from __future__ import annotations

from typing import List, Dict, Set, Optional

import re

import onnx


# Ops that are considered "trivial" for the purpose of determining whether a
# tensor ultimately originates from graph inputs / constants.
#
# IMPORTANT: do NOT include ops like Gather/MatMul/Gemm etc. Gather in
# particular is used for token embeddings and should be treated as
# "non-trivial" so that embedding outputs are considered activations.
_TRIVIAL_FLOW_OPS: Set[str] = {
    # Shape / view ops
    "Cast",
    "Reshape",
    "Transpose",
    "Squeeze",
    "Unsqueeze",
    "Expand",
    "Slice",
    "Concat",
    "Split",
    "Tile",
    "Identity",
    "Flatten",
    "Pad",
    "Resize",
    "Shape",
    "Size",
    "NonZero",
    "ConstantOfShape",
    # Elementwise / boolean ops commonly used for masks
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Neg",
    "Abs",
    "Floor",
    "Ceil",
    "Round",
    "Clip",
    "Max",
    "Min",
    "Where",
    "Equal",
    "Not",
    "And",
    "Or",
    "Xor",
    "Less",
    "Greater",
    "LessOrEqual",
    "GreaterOrEqual",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceMean",
    "ReduceProd",
}


# Heuristic name patterns that are almost always "external plumbing" in LLM graphs.
# If a tensor name matches, we treat it as external flow and do not use it to
# build skip blocks.
_EXTERNAL_NAME_HINTS = (
    "attn_mask",
    "attention_mask",
    "position_ids",
    "pos_ids",
    "cos_cache",
    "sin_cache",
    "rotary",
    "rope",
    "past_key_values",
    "present.",
)


_LAYER_PATTERNS = (
    re.compile(r"layers\.(\d+)", re.IGNORECASE),
    re.compile(r"/layers/(\d+)", re.IGNORECASE),
    re.compile(r"layers_(\d+)", re.IGNORECASE),
)


def _extract_layer_id_from_node(node: onnx.NodeProto) -> Optional[int]:
    """Try to extract a transformer layer index from a node.

    Many exported LLM ONNX graphs use names like "/model/layers.20/...".
    """

    def _scan(s: str) -> Optional[int]:
        if not s:
            return None
        for pat in _LAYER_PATTERNS:
            m = pat.search(s)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    # Prefer node.name, then outputs, then inputs.
    lid = _scan(getattr(node, "name", "") or "")
    if lid is not None:
        return lid
    for s in getattr(node, "output", []) or []:
        lid = _scan(s)
        if lid is not None:
            return lid
    for s in getattr(node, "input", []) or []:
        lid = _scan(s)
        if lid is not None:
            return lid
    return None


def _looks_like_external_tensor_name(name: str) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(h in low for h in _EXTERNAL_NAME_HINTS)


def _is_external_flow_tensor(
    tensor_name: str,
    *,
    producer_of: Dict[str, int],
    nodes: List[onnx.NodeProto],
    _cache: Dict[str, bool],
    _visiting: Set[str],
) -> bool:
    """Return True if a tensor is derived only from external inputs/constants.

    This is a conservative filter used in skip-block detection.

    We treat a tensor as "external flow" if:
      * It originates from a graph input / initializer (no producer), OR
      * It originates from a Constant op, OR
      * It is produced by a chain of "trivial" ops whose inputs are all external.

    If the chain hits a non-trivial op (e.g., MatMul, Gemm, MultiHeadAttention,
    Gather), we treat the tensor as an activation (i.e., NOT external flow).
    """

    if _looks_like_external_tensor_name(tensor_name):
        return True

    if tensor_name in _cache:
        return _cache[tensor_name]

    if tensor_name in _visiting:
        # Cycle guard: be conservative.
        return True

    _visiting.add(tensor_name)

    p = producer_of.get(tensor_name)
    if p is None:
        _cache[tensor_name] = True
        _visiting.remove(tensor_name)
        return True

    if p < 0 or p >= len(nodes):
        _cache[tensor_name] = True
        _visiting.remove(tensor_name)
        return True

    node = nodes[p]
    op = getattr(node, "op_type", "") or ""

    if op == "Constant" or op == "ConstantOfShape":
        _cache[tensor_name] = True
        _visiting.remove(tensor_name)
        return True

    if op not in _TRIVIAL_FLOW_OPS:
        _cache[tensor_name] = False
        _visiting.remove(tensor_name)
        return False

    # For trivial ops, only external if *all* inputs are external.
    for inp in getattr(node, "input", []) or []:
        if not inp:
            continue
        if not _is_external_flow_tensor(
            inp,
            producer_of=producer_of,
            nodes=nodes,
            _cache=_cache,
            _visiting=_visiting,
        ):
            _cache[tensor_name] = False
            _visiting.remove(tensor_name)
            return False

    _cache[tensor_name] = True
    _visiting.remove(tensor_name)
    return True


def detect_skip_blocks(
    order: List[int],
    nodes: List[onnx.NodeProto],
    producer_of: Dict[str, int],
    *,
    merge_ops=("Add", "Concat", "Sum", "Mul"),
    min_skip_len: int = 8,
    merge_touching_same_layer: bool = True,
    ignore_external_flows: bool = True,
) -> List[Dict[str, object]]:
    """Detect skip/residual blocks.

    Returns a list of dicts:
      {"start_pos": int, "end_pos": int, "members": [...], "layer_id": Optional[int]}

    Positions are in the *analysis order* (topological order array index), not
    raw node indices.

    Notes:
      * We consider merge ops in `merge_ops` AND any op_type that starts with
        "Skip" (e.g., SkipLayerNormalization / SkipSimplifiedLayerNormalization)
        as potential residual merges.
      * For each merge op, we look for inputs produced far earlier than the merge.
      * For transformer graphs, we can merge touching residual blocks within the
        same layer (based on layer id extracted from names).
    """

    merge_ops_set = set(merge_ops)

    # Map node_idx -> position in analysis order.
    pos_of: Dict[int, int] = {node_idx: i for i, node_idx in enumerate(order)}

    ext_cache: Dict[str, bool] = {}

    raw_blocks: List[Dict[str, object]] = []

    for i, node_idx in enumerate(order):
        if node_idx < 0 or node_idx >= len(nodes):
            continue

        node = nodes[node_idx]
        op_type = getattr(node, "op_type", "") or ""
        is_merge = op_type in merge_ops_set or op_type.startswith("Skip")
        if not is_merge:
            continue

        layer_id = _extract_layer_id_from_node(node)

        # Identify long-range inputs.
        for inp in getattr(node, "input", []) or []:
            if not inp:
                continue
            p_node_idx = producer_of.get(inp)
            if p_node_idx is None:
                continue
            p_pos = pos_of.get(p_node_idx)
            if p_pos is None:
                continue

            span = i - p_pos
            if span < min_skip_len:
                continue

            if ignore_external_flows:
                if _is_external_flow_tensor(
                    inp,
                    producer_of=producer_of,
                    nodes=nodes,
                    _cache=ext_cache,
                    _visiting=set(),
                ):
                    continue

            raw_blocks.append(
                {
                    "start_pos": int(p_pos),
                    "end_pos": int(i),
                    "members": [(int(p_pos), int(i), inp, op_type)],
                    "layer_id": layer_id,
                }
            )

    if not raw_blocks:
        return []

    # Sort and merge overlapping / (optionally) touching blocks.
    raw_blocks.sort(key=lambda b: (b["start_pos"], b["end_pos"]))

    merged: List[Dict[str, object]] = []

    def _layer_compatible(a: Optional[int], b: Optional[int]) -> bool:
        # If we can't determine a layer id for either block, be conservative:
        # do NOT treat them as belonging to the same layer.
        if a is None or b is None:
            return False
        return a == b

    for blk in raw_blocks:
        s = int(blk["start_pos"])
        e = int(blk["end_pos"])
        lid = blk.get("layer_id")

        if not merged:
            merged.append(dict(blk))
            continue

        last = merged[-1]
        last_s = int(last["start_pos"])
        last_e = int(last["end_pos"])
        last_lid = last.get("layer_id")

        overlap = s < last_e
        touching = s == last_e

        can_merge_touch = merge_touching_same_layer and touching and _layer_compatible(last_lid, lid)
        can_merge_overlap = overlap and (
            # If both have layer id and match -> safe.
            (lid is not None and last_lid is not None and lid == last_lid)
            # If one is missing, avoid merging overlap (prevents huge blocks in messy graphs).
        )

        if can_merge_overlap or can_merge_touch:
            # Merge into last.
            last["end_pos"] = max(last_e, e)
            last_members = list(last.get("members", []))
            last_members.extend(blk.get("members", []))
            last["members"] = last_members
            # Keep layer_id.
            continue

        # No merge.
        merged.append(dict(blk))

    # Normalize members (optional).
    for b in merged:
        b["start_pos"] = int(b["start_pos"])
        b["end_pos"] = int(b["end_pos"])

    return merged


def prune_candidates_skip_block(
    candidates: List[int],
    skip_blocks: List[Dict[str, object]],
    *,
    allow_last_n_inside: int = 0,
) -> List[int]:
    """Prune candidate boundary indices that fall inside skip/residual blocks.

    We allow splits:
      * at the start of a block (boundary index == start_pos)
      * after the end of a block (boundary index == end_pos + 1)

    Everything strictly inside is removed, except optionally the last N boundaries
    immediately before end_pos ("allow_last_n_inside").
    """

    if not candidates or not skip_blocks:
        return candidates

    forbidden: Set[int] = set()

    for blk in skip_blocks:
        s = int(blk["start_pos"])
        e = int(blk["end_pos"])

        inner_start = s + 1
        # Forbid boundaries up to and including the merge/end node position (end_pos).
        # This prevents cuts that land *right before* the merge op.
        inner_end = max(inner_start, (e + 1) - max(0, int(allow_last_n_inside)))

        for b in range(inner_start, inner_end):
            forbidden.add(b)

    return [b for b in candidates if b not in forbidden]
