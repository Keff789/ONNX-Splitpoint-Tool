from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import TensorProto

from .onnx_utils import dtype_nbytes, elemtype_from_vi, numel_from_shape, shape_from_vi

_PRESENT_RE = re.compile(r"^present\.(\d+)\.(key|value)$")
_SEM_LAYER_RE = re.compile(r"(?:^|/)(?:layers|blocks)[\._](\d+)(?:/|$)")


def initializer_nbytes(tensor_proto: TensorProto) -> int:
    n = 1
    for d in tensor_proto.dims:
        n *= int(d) if int(d) > 0 else 1
    return int(n) * int(dtype_nbytes(int(tensor_proto.data_type), default=4))


def _constant_node_tensor_nbytes(node: onnx.NodeProto) -> int:
    """Best-effort size (bytes) of a Constant node output tensor.

    We mainly care about Constant nodes that hold large weight tensors (common for some
    exporters). If we can't reliably infer the tensor, we return 0 and simply don't
    count it as persistent weight memory.
    """
    try:
        # Typical ONNX Constant uses an attribute named "value" with a TensorProto.
        for a in getattr(node, "attribute", []) or []:
            if getattr(a, "name", None) == "value" and getattr(a, "type", None) == onnx.AttributeProto.TENSOR:
                t = getattr(a, "t", None)
                if t is not None:
                    return int(initializer_nbytes(t))
        # Some Constant nodes use sparse_value.
        for a in getattr(node, "attribute", []) or []:
            if getattr(a, "name", None) == "sparse_value" and getattr(a, "type", None) == onnx.AttributeProto.SPARSE_TENSOR:
                st = getattr(a, "sparse_tensor", None)
                if st is not None and getattr(st, "values", None) is not None:
                    return int(initializer_nbytes(st.values))
    except Exception:
        return 0
    return 0


def precompute_initializer_spans(model: onnx.ModelProto, nodes: List[onnx.NodeProto], order: List[int]) -> Dict[str, Tuple[int, int, int]]:
    """Compute consumer-span ranges for persistent tensors.

    Historically we only accounted for graph initializers. Some exporters, however,
    encode large weight tensors as Constant nodes instead of initializers. Those
    constants should still count towards RAM usage and are typically replicated into
    the split subgraphs (similar to initializers).

    We therefore treat BOTH:
      - model.graph.initializer tensors
      - outputs of Constant nodes (when we can infer their byte size)

    as persistent tensors and compute (min_consumer_pos, max_consumer_pos, nbytes).
    """
    pos_of = {int(n): i for i, n in enumerate(order)}

    # Build consumer positions for every value name by scanning node inputs once.
    consumers: Dict[str, List[int]] = {}
    for idx, node in enumerate(nodes):
        p = pos_of.get(idx)
        if p is None:
            continue
        for inp in getattr(node, "input", []) or []:
            if inp:
                consumers.setdefault(str(inp), []).append(int(p))

    spans: Dict[str, Tuple[int, int, int]] = {}

    # 1) Standard ONNX initializers
    for init in model.graph.initializer:
        name = str(init.name)
        c = consumers.get(name) or []
        if not c:
            continue
        spans[name] = (min(c), max(c), initializer_nbytes(init))

    # 2) Constant node outputs (weights encoded as nodes)
    for node in nodes:
        if str(getattr(node, "op_type", "")) != "Constant":
            continue
        nbytes = int(_constant_node_tensor_nbytes(node))
        if nbytes <= 0:
            continue
        outs = [str(o) for o in getattr(node, "output", []) or [] if o]
        if not outs:
            continue
        # Constant nodes are expected to have a single output; to avoid overcounting,
        # assign the tensor size to the first output only.
        out0 = outs[0]
        c = consumers.get(out0) or []
        if not c:
            continue
        spans[out0] = (min(c), max(c), nbytes)

    return spans


def weights_for_boundary(spans: Dict[str, Tuple[int, int, int]], boundary: int) -> Tuple[int, int]:
    left = 0
    right = 0
    for mn, mx, nbytes in spans.values():
        if mx <= boundary:
            left += nbytes
        elif mn > boundary:
            right += nbytes
        else:
            left += nbytes
            right += nbytes
    return int(left), int(right)


def weights_for_all_boundaries(spans: Dict[str, Tuple[int, int, int]], m_boundaries: int) -> Tuple[List[int], List[int]]:
    l: List[int] = []
    r: List[int] = []
    for b in range(max(0, m_boundaries)):
        wl, wr = weights_for_boundary(spans, b)
        l.append(wl)
        r.append(wr)
    return l, r


def kv_cache_bytes_per_layer(model: onnx.ModelProto, vimap: Dict[str, onnx.ValueInfoProto], llm_hints: Optional[Dict[str, int]] = None) -> Dict[int, int]:
    out: Dict[int, int] = {}
    layer_key: Dict[int, int] = {}
    layer_val: Dict[int, int] = {}
    for o in model.graph.output:
        m = _PRESENT_RE.match(str(o.name or ""))
        if not m:
            continue
        idx = int(m.group(1))
        kind = m.group(2)
        vi = vimap.get(o.name)
        shp = shape_from_vi(vi)
        dt = elemtype_from_vi(vi)
        if shp is None:
            # conservative fallback for unknown shape
            total_len = int((llm_hints or {}).get("total_len") or 1)
            shp = [1, total_len, 1024]
            dt = dt or TensorProto.FLOAT16
        nbytes = int(numel_from_shape(shp)) * int(dtype_nbytes(int(dt or TensorProto.FLOAT16), default=2))
        if kind == "key":
            layer_key[idx] = nbytes
        else:
            layer_val[idx] = nbytes
    for i in sorted(set(layer_key.keys()) | set(layer_val.keys())):
        out[i] = int(layer_key.get(i, 0) + layer_val.get(i, 0))
    return out


def layer_split_index_for_boundary(nodes: List[onnx.NodeProto], order: List[int], boundary: int) -> Optional[int]:
    max_left = None
    for p in range(0, min(len(order), boundary + 1)):
        n = nodes[order[p]]
        cand = [str(getattr(n, "name", ""))] + [str(x) for x in n.input] + [str(x) for x in n.output]
        for s in cand:
            m = _SEM_LAYER_RE.search(s)
            if m:
                i = int(m.group(1))
                if max_left is None or i > max_left:
                    max_left = i
    return max_left


def kv_for_boundary(kv_per_layer: Dict[int, int], split_layer_idx: Optional[int]) -> Tuple[int, int]:
    if not kv_per_layer:
        return 0, 0
    if split_layer_idx is None:
        return 0, int(sum(kv_per_layer.values()))
    l = sum(v for i, v in kv_per_layer.items() if i <= split_layer_idx)
    r = sum(v for i, v in kv_per_layer.items() if i > split_layer_idx)
    return int(l), int(r)


def estimate_ram_bytes(weights_bytes: int, peak_activations_bytes: int, kv_cache_bytes: int, runtime_overhead_bytes: int, comm_bytes: int, policy: str = "max_peak_or_comm") -> int:
    if policy == "sum_peak_and_comm":
        dyn = int(max(0, peak_activations_bytes)) + int(max(0, comm_bytes))
    else:
        dyn = int(max(max(0, peak_activations_bytes), max(0, comm_bytes)))
    return int(max(0, weights_bytes) + max(0, kv_cache_bytes) + max(0, runtime_overhead_bytes) + dyn)
