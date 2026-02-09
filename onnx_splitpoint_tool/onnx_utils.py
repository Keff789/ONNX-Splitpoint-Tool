"""ONNX graph parsing helpers.

Includes:
- dtype/shape helpers
- value_info map extraction
- producer/consumer maps
- topological sort
- quantized-shape backfilling
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import onnx
from onnx import TensorProto, helper, shape_inference


def dtype_nbytes(elem_type: int, default: int = 4) -> int:
    """Return bytes per element for ONNX TensorProto elem_type."""
    T = TensorProto
    mapping = {
        T.FLOAT: 4,
        T.FLOAT16: 2,
        T.BFLOAT16: 2,
        T.DOUBLE: 8,
        T.UINT8: 1,
        T.INT8: 1,
        T.UINT16: 2,
        T.INT16: 2,
        T.UINT32: 4,
        T.INT32: 4,
        T.UINT64: 8,
        T.INT64: 8,
        T.BOOL: 1,
    }
    return int(mapping.get(int(elem_type), default))


# ---------------------------- ValueInfo helpers ----------------------------

def shape_from_vi(vi) -> Optional[List[Optional[int]]]:
    if vi is None or not vi.type.HasField("tensor_type"):
        return None
    shp: List[Optional[int]] = []
    for d in vi.type.tensor_type.shape.dim:
        shp.append(int(d.dim_value) if d.HasField("dim_value") else None)
    return shp


def elemtype_from_vi(vi) -> Optional[int]:
    if vi is None or not vi.type.HasField("tensor_type"):
        return None
    return int(vi.type.tensor_type.elem_type)


def numel_from_shape(shape: Optional[List[Optional[int]]], batch_override: Optional[int] = None) -> int:
    if not shape:
        return 0
    shp = list(shape)
    if batch_override is not None and shp:
        if shp[0] is None or int(shp[0] or 0) == 0:
            shp[0] = int(batch_override)
    n = 1
    for d in shp:
        n *= int(d) if (d is not None and int(d) > 0) else 1
    return int(n)


def value_info_map(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    """Map tensor name -> ValueInfo.

    Notes
    -----
    ONNX models do not always populate `graph.value_info` for every produced tensor.
    In practice (especially for transformer exports), a significant number of tensors
    are produced by `Constant` nodes and never get a ValueInfo from shape inference.
    
    For our use-cases (activation byte estimates, unknown-shape diagnostics), it is
    still very helpful to at least know dtype + static dims for these constants.
    The constant payloads are usually tiny (shape/control tensors), so this improves
    coverage without impacting memory.
    """

    vis = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    m: Dict[str, onnx.ValueInfoProto] = {vi.name: vi for vi in vis}

    # Backfill missing ValueInfo for Constant node outputs.
    try:
        from onnx import AttributeProto
    except Exception:  # pragma: no cover
        AttributeProto = None  # type: ignore

    if AttributeProto is not None:
        for n in model.graph.node:
            if n.op_type != "Constant" or not n.output:
                continue
            out = n.output[0]
            if out in m:
                continue

            # Prefer the tensor-valued "value" attribute when present.
            t = None
            for a in n.attribute:
                if a.name == "value" and a.type == AttributeProto.TENSOR:
                    t = a.t
                    break

            if t is not None:
                try:
                    # TensorProto.dims can be empty for scalars.
                    m[out] = helper.make_tensor_value_info(out, t.data_type, list(t.dims))
                except Exception:
                    # If something goes wrong, just leave it unknown.
                    pass

    return m


# ---------------------------- Graph utilities ----------------------------

def build_producers_consumers(model: onnx.ModelProto):
    nodes = list(model.graph.node)
    producer_of: Dict[str, int] = {}
    consumers_of: Dict[str, List[int]] = defaultdict(list)

    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                producer_of[out] = idx

    for idx, node in enumerate(nodes):
        for inp in node.input:
            if inp in producer_of:
                consumers_of[inp].append(idx)

    return nodes, producer_of, consumers_of


def topo_sort(nodes: List[onnx.NodeProto], producer_of: Dict[str, int]) -> List[int]:
    """Kahn topological sort over ONNX node indices."""
    preds: List[set] = [set() for _ in nodes]
    succs: List[set] = [set() for _ in nodes]

    for j, node in enumerate(nodes):
        for inp in node.input:
            if inp in producer_of:
                p = producer_of[inp]
                preds[j].add(p)
                succs[p].add(j)

    indeg = [len(p) for p in preds]
    q = deque([i for i, d in enumerate(indeg) if d == 0])
    order: List[int] = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in list(succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # If graph has cycles/unconnected remnants, append remaining nodes
    if len(order) != len(nodes):
        remaining = [i for i in range(len(nodes)) if i not in set(order)]
        order.extend(remaining)

    return order


# ---------------------------- Quant shape backfilling ----------------------------

def _get_attr(node: onnx.NodeProto, name: str, default=None):
    for a in node.attribute:
        if a.name != name:
            continue
        if a.type == onnx.AttributeProto.INTS:
            return list(a.ints)
        if a.type == onnx.AttributeProto.INT:
            return int(a.i)
        if a.type == onnx.AttributeProto.FLOAT:
            return float(a.f)
        if a.type == onnx.AttributeProto.FLOATS:
            return list(a.floats)
        if a.type == onnx.AttributeProto.STRING:
            try:
                return a.s.decode("utf-8")
            except Exception:
                return str(a.s)
    return default


def _conv_output_hw(h_in: int, w_in: int, k: List[int], s: List[int], p: List[int], d: List[int]) -> Tuple[int, int]:
    kh, kw = (k + [1, 1])[:2]
    sh, sw = (s + [1, 1])[:2]
    dh, dw = (d + [1, 1])[:2]

    if len(p) == 2:
        pt, pl, pb, pr = p[0], p[1], p[0], p[1]
    elif len(p) == 4:
        pt, pl, pb, pr = p
    else:
        pt = pl = pb = pr = 0

    def out_dim(Lin: int, k: int, s: int, p0: int, p1: int, dil: int) -> int:
        return (Lin + p0 + p1 - dil * (k - 1) - 1) // s + 1

    hout = out_dim(int(h_in), int(kh), int(sh), int(pt), int(pb), int(dh))
    wout = out_dim(int(w_in), int(kw), int(sw), int(pl), int(pr), int(dw))
    return max(1, int(hout)), max(1, int(wout))


def _matmul_out_shape(sa: Optional[List[Optional[int]]], sb: Optional[List[Optional[int]]]) -> List[int]:
    """Broadcast-like matmul output shape (approx.)."""

    def norm2(s):
        if not s:
            return [1, 1]
        s2 = [d or 1 for d in s]
        if len(s2) == 1:
            return [1, s2[0]]
        return s2

    A = norm2(sa)
    B = norm2(sb)

    m, k1 = A[-2], A[-1]
    k2, n = B[-2], B[-1]
    _k = min(k1, k2)

    lead_a = A[:-2]
    lead_b = B[:-2]
    L = max(len(lead_a), len(lead_b))
    lead_a = [1] * (L - len(lead_a)) + lead_a
    lead_b = [1] * (L - len(lead_b)) + lead_b
    lead = [max(a, b) for a, b in zip(lead_a, lead_b)]

    return [int(x) for x in (lead + [m, n])]


def backfill_quant_shapes(
    model: onnx.ModelProto,
    vimap: Dict[str, onnx.ValueInfoProto],
    batch_override: Optional[int] = None,
    batch: Optional[int] = None,
) -> None:
    """Backfill ValueInfo entries for common quant operators.

    Accepts both `batch_override` and `batch` for compatibility.
    """
    if batch_override is None and batch is not None:
        batch_override = batch

    # Only need initializer *shapes* for quant backfilling.
    # Avoid materializing large weights into memory (external data).
    init_map = {init.name: list(init.dims) for init in model.graph.initializer}

    def ensure_vi(name: str, shape: List[int], elem_type: int):
        if name in vimap:
            return
        vi = helper.make_tensor_value_info(name, elem_type, list(shape))
        vimap[name] = vi
        model.graph.value_info.extend([vi])

    def shp(name: str) -> Optional[List[Optional[int]]]:
        return shape_from_vi(vimap.get(name))

    for node in model.graph.node:
        op = node.op_type

        if op == "QLinearConv":
            # Spec: inputs: x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, B(optional)
            if len(node.input) < 4 or not node.output:
                continue
            x, w, y = node.input[0], node.input[3], node.output[0]
            if y in vimap:
                continue
            xs = shp(x)
            W_dims = init_map.get(w)
            if not xs or len(xs) != 4 or not W_dims or len(W_dims) != 4:
                continue
            N, C, H, Wd = [batch_override if (i == 0 and batch_override) else int(xs[i] or 1) for i in range(4)]
            Cout, _, kH, kW = [int(d or 1) for d in W_dims]
            Ho, Wo = _conv_output_hw(
                H,
                Wd,
                [int(kH), int(kW)],
                _get_attr(node, "strides", [1, 1]) or [1, 1],
                _get_attr(node, "pads", [0, 0, 0, 0]) or [0, 0, 0, 0],
                _get_attr(node, "dilations", [1, 1]) or [1, 1],
            )
            ensure_vi(y, [int(N), int(Cout), int(Ho), int(Wo)], TensorProto.UINT8)

        elif op == "QLinearAdd":
            # Typical: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
            if len(node.input) < 4 or not node.output:
                continue
            a, b, y = node.input[0], node.input[3], node.output[0]
            if y in vimap:
                continue
            sa, sb = shp(a), shp(b)
            if not sa and not sb:
                continue
            A = [int(d or 1) for d in (sa or [])]
            B = [int(d or 1) for d in (sb or [])]
            L = max(len(A), len(B))
            A = [1] * (L - len(A)) + A
            B = [1] * (L - len(B)) + B
            out = [max(x, y2) for x, y2 in zip(A, B)]
            ensure_vi(y, out, TensorProto.UINT8)

        elif op == "QLinearGlobalAveragePool":
            if not node.input or not node.output:
                continue
            x, y = node.input[0], node.output[0]
            if y in vimap:
                continue
            xs = shp(x)
            if not xs or len(xs) != 4:
                continue
            N, C, _, _ = [batch_override if (i == 0 and batch_override) else int(xs[i] or 1) for i in range(4)]
            ensure_vi(y, [int(N), int(C), 1, 1], TensorProto.UINT8)

        elif op == "QLinearMatMul":
            # Spec: a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp
            if len(node.input) < 4 or not node.output:
                continue
            a, b, y = node.input[0], node.input[3], node.output[0]
            if y in vimap:
                continue
            sa, sb = shp(a), shp(b)
            if not sa and not sb:
                continue
            out = _matmul_out_shape(sa, sb)
            ensure_vi(y, out, TensorProto.UINT8)

        elif op == "QuantizeLinear":
            if not node.input or not node.output:
                continue
            x, y = node.input[0], node.output[0]
            if y in vimap:
                continue
            sa = shp(x)
            if sa:
                ensure_vi(y, [int(d or 1) for d in sa], TensorProto.UINT8)

        elif op == "DequantizeLinear":
            if not node.input or not node.output:
                continue
            x, y = node.input[0], node.output[0]
            if y in vimap:
                continue
            sa = shp(x)
            if sa:
                ensure_vi(y, [int(d or 1) for d in sa], TensorProto.FLOAT)
