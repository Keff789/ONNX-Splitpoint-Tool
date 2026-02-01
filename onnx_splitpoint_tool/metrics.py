"""Metrics and ranking for split boundaries.

Includes:
- activation byte estimation
- boundary communication costs
- approximate peak activation memory
- FLOP estimation per node and per boundary
- weighted score ranking
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import onnx

from .onnx_utils import (
    dtype_nbytes,
    elemtype_from_vi,
    numel_from_shape,
    shape_from_vi,
)


def compute_tensor_bytes_per_value(
    vimap: Dict[str, onnx.ValueInfoProto],
    batch_override: Optional[int] = None,
    assume_activation_bytes: Optional[int] = None,
    batch: Optional[int] = None,
) -> Dict[str, int]:
    """Return mapping value_name -> estimated bytes.

    Accepts both `batch_override` and `batch` for compatibility.
    """
    if batch_override is None and batch is not None:
        batch_override = batch

    out: Dict[str, int] = {}
    for name, vi in vimap.items():
        shp = shape_from_vi(vi)
        n = numel_from_shape(shp, batch_override)
        if n <= 0:
            out[name] = 0
            continue
        if assume_activation_bytes is not None:
            bpe = int(assume_activation_bytes)
        else:
            et = elemtype_from_vi(vi) or 0
            bpe = dtype_nbytes(et)
        out[name] = int(n) * int(bpe)
    return out


def boundary_costs(
    order: List[int],
    producer_of: Dict[str, int],
    consumers_of: Dict[str, List[int]],
    value_bytes: Dict[str, int],
) -> Tuple[List[int], Dict[str, Tuple[int, int]]]:
    """Compute Comm(b) for all boundaries and return (costs, val_span).

    val_span maps value_name -> (producer_position, last_consumer_position).
    Only values with a positive byte size are included.
    """
    N = len(order)
    M = max(0, N - 1)
    if M == 0:
        return [], {}

    pos_of = {node_idx: pos for pos, node_idx in enumerate(order)}

    deltas = [0] * (M + 1)
    val_span: Dict[str, Tuple[int, int]] = {}

    for val, p_node in producer_of.items():
        b = int(value_bytes.get(val, 0) or 0)
        if b <= 0:
            continue
        p_pos = int(pos_of[p_node])
        cons_pos = [int(pos_of[c]) for c in consumers_of.get(val, []) if int(pos_of[c]) > p_pos]
        if not cons_pos:
            continue
        last = int(max(cons_pos))
        # contributes to boundaries p_pos <= b < last
        deltas[p_pos] += b
        deltas[min(last, M)] -= b
        val_span[val] = (p_pos, last)

    costs: List[int] = []
    run = 0
    for i in range(M):
        run += deltas[i]
        costs.append(int(run))

    return costs, val_span

def peak_activation_memory_per_boundary(costs_bytes: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """Derive *approximate* peak activation memory for left/right partitions per boundary.

    We interpret the per-boundary live activation size L[b] as:
      L[b] = sum(bytes(v)) for all tensors v that are alive across boundary b

    In this tool, L[b] is identical to Comm(b) (crossing activation bytes) because
    both are derived from the same producer->last-consumer value spans.

    For a boundary b:
      - part1 executes nodes [0..b]  and ends at boundary b
      - part2 executes nodes [b+1..] and starts at boundary b

    Therefore, an upper-bound approximation for the peak activation memory needed is:
      peak_left[b]  = max_{i <= b} L[i]
      peak_right[b] = max_{i >= b} L[i]
      peak_max[b]   = max(peak_left[b], peak_right[b])

    Notes
    -----
    - This is a coarse estimate: it does not model buffer reuse, in-place ops,
      streaming of cut tensors, recomputation, or weight/parameter memory.
    - Unknown activation sizes are treated as 0 bytes (lower bound), consistent
      with Comm(b) handling in the rest of the tool.
    """
    if not costs_bytes:
        return [], [], []

    L = [int(max(0, int(x))) for x in costs_bytes]
    M = len(L)

    peak_left: List[int] = [0] * M
    run = 0
    for i in range(M):
        run = max(run, int(L[i]))
        peak_left[i] = int(run)

    peak_right: List[int] = [0] * M
    run = 0
    for i in range(M - 1, -1, -1):
        run = max(run, int(L[i]))
        peak_right[i] = int(run)

    peak_max = [int(max(peak_left[i], peak_right[i])) for i in range(M)]
    return peak_left, peak_right, peak_max



def boundary_tensor_counts(
    order: List[int],
    producer_of: Dict[str, int],
    consumers_of: Dict[str, List[int]],
    value_bytes: Dict[str, int],
) -> List[int]:
    """Count number of crossing tensors per boundary.

    Only values with value_bytes[val] > 0 are counted.
    """
    N = len(order)
    M = max(0, N - 1)
    if M == 0:
        return []

    pos_of = {node_idx: pos for pos, node_idx in enumerate(order)}
    deltas = [0] * (M + 1)

    for val, p_node in producer_of.items():
        if int(value_bytes.get(val, 0) or 0) <= 0:
            continue
        p_pos = int(pos_of[p_node])
        cons_pos = [int(pos_of[c]) for c in consumers_of.get(val, []) if int(pos_of[c]) > p_pos]
        if not cons_pos:
            continue
        last = int(max(cons_pos))
        deltas[p_pos] += 1
        deltas[min(last, M)] -= 1

    counts: List[int] = []
    run = 0
    for i in range(M):
        run += deltas[i]
        counts.append(int(run))

    return counts


def collect_crossing_values_for_boundary(
    b: int, val_span: Dict[str, Tuple[int, int]], value_bytes: Dict[str, int]
) -> List[Tuple[str, int]]:
    """Return a sorted list of (value_name, bytes) crossing boundary b."""
    items = []
    for val, (p, last) in val_span.items():
        if p <= b < last:
            items.append((val, int(value_bytes.get(val, 0) or 0)))
    items.sort(key=lambda x: x[1], reverse=True)
    return items


# ---------------------------- FLOPs per node ----------------------------

def _prod(xs: Iterable[int]) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return int(n)


def _matmul_dims(sa: Optional[List[Optional[int]]], sb: Optional[List[Optional[int]]]) -> Tuple[int, int, int, int]:
    if not sa or not sb or len(sa) < 2 or len(sb) < 2:
        return (0, 0, 0, 0)
    A = [int(d or 1) for d in sa]
    B = [int(d or 1) for d in sb]
    M, Ka = A[-2], A[-1]
    Kb, N = B[-2], B[-1]
    K = min(Ka, Kb)
    Ba = _prod(A[:-2])
    Bb = _prod(B[:-2])
    batch = max(Ba, Bb)
    return (int(batch), int(M), int(K), int(N))


def flops_per_output_element(node: onnx.NodeProto, vimap: Dict[str, onnx.ValueInfoProto], batch_override: Optional[int] = None) -> int:
    if not node.output:
        return 0
    out_shape = shape_from_vi(vimap.get(node.output[0]))
    return int(numel_from_shape(out_shape, batch_override))


def flops_conv(node: onnx.NodeProto, vimap, init_map, batch_override: Optional[int] = None) -> int:
    if len(node.input) < 2 or not node.output:
        return 0
    W = init_map.get(node.input[1])
    if W is None or getattr(W, "ndim", 0) != 4:
        return 0
    C_out, C_in_per_group, kH, kW = W.shape
    out_shape = shape_from_vi(vimap.get(node.output[0]))
    if not out_shape or len(out_shape) != 4:
        return 0
    N, _, H, Wd = [int(d or 1) for d in out_shape]
    if batch_override is not None:
        N = int(batch_override)
    return int(2 * N * H * Wd * int(C_out) * int(C_in_per_group) * int(kH) * int(kW))


def flops_qlinearconv(node: onnx.NodeProto, vimap, init_map, batch_override: Optional[int] = None) -> int:
    # weight is input[3]
    if len(node.input) < 4 or not node.output:
        return 0
    W = init_map.get(node.input[3])
    if W is None or getattr(W, "ndim", 0) != 4:
        return 0
    C_out, C_in_per_group, kH, kW = W.shape
    out_shape = shape_from_vi(vimap.get(node.output[0]))
    if not out_shape or len(out_shape) != 4:
        return 0
    N, _, H, Wd = [int(d or 1) for d in out_shape]
    if batch_override is not None:
        N = int(batch_override)
    return int(2 * N * H * Wd * int(C_out) * int(C_in_per_group) * int(kH) * int(kW))


def flops_convtranspose(node: onnx.NodeProto, vimap, init_map, batch_override: Optional[int] = None) -> int:
    if len(node.input) < 2 or not node.output:
        return 0
    W = init_map.get(node.input[1])
    if W is None or getattr(W, "ndim", 0) != 4:
        return 0
    # ConvTranspose weight layout: [C_in, C_out/group, kH, kW]
    C_in, C_out_per_group, kH, kW = W.shape
    out_shape = shape_from_vi(vimap.get(node.output[0]))
    if not out_shape or len(out_shape) != 4:
        return 0
    N, _, H, Wd = [int(d or 1) for d in out_shape]
    if batch_override is not None:
        N = int(batch_override)
    return int(2 * N * H * Wd * int(C_in) * int(C_out_per_group) * int(kH) * int(kW))


def flops_matmul(node: onnx.NodeProto, vimap) -> int:
    if len(node.input) < 2:
        return 0
    sa = shape_from_vi(vimap.get(node.input[0]))
    sb = shape_from_vi(vimap.get(node.input[1]))
    B, M, K, N = _matmul_dims(sa, sb)
    if B == 0:
        return 0
    return int(2 * B * M * N * K)


def flops_qlinearmatmul(node: onnx.NodeProto, vimap) -> int:
    # spec: a at input[0], b at input[3]
    if len(node.input) < 4:
        return 0
    sa = shape_from_vi(vimap.get(node.input[0]))
    sb = shape_from_vi(vimap.get(node.input[3]))
    B, M, K, N = _matmul_dims(sa, sb)
    if B == 0:
        return 0
    return int(2 * B * M * N * K)


def flops_gemm(node: onnx.NodeProto, vimap) -> int:
    if len(node.input) < 2:
        return 0
    sa = shape_from_vi(vimap.get(node.input[0]))
    sb = shape_from_vi(vimap.get(node.input[1]))
    B, M, K, N = _matmul_dims(sa, sb)
    if B == 0:
        return 0
    return int(2 * B * M * N * K)


def flops_batchnorm(node: onnx.NodeProto, vimap, batch_override: Optional[int], ops_per_elt: int = 4) -> int:
    return int(ops_per_elt) * flops_per_output_element(node, vimap, batch_override)


def flops_activation(node: onnx.NodeProto, vimap, batch_override: Optional[int], ops_per_elt: int = 1) -> int:
    return int(ops_per_elt) * flops_per_output_element(node, vimap, batch_override)


def flops_resize(node: onnx.NodeProto, vimap, batch_override: Optional[int], default_ops: int = 1) -> int:
    mode = _get_attr(node, "mode", None)
    ops = int(default_ops)
    if isinstance(mode, str):
        m = mode.lower()
        if m in ("nearest",):
            ops = 1
        elif m in ("linear", "bilinear", "trilinear"):
            ops = 8
        elif m in ("cubic", "bicubic"):
            ops = 16
    return int(ops) * flops_per_output_element(node, vimap, batch_override)


def per_node_flops(
    model: onnx.ModelProto,
    vimap: Dict[str, onnx.ValueInfoProto],
    batch_override: Optional[int] = None,
    batch: Optional[int] = None,
    bn_cost_per_elt: int = 4,
    act_cost_per_elt: int = 1,
    resize_cost_per_elt: int = 1,
) -> List[Tuple[int, str, str, int]]:
    """Return list of (node_index, op_type, name, flops)."""
    if batch_override is None and batch is not None:
        batch_override = batch

    init_map = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}

    recs: List[Tuple[int, str, str, int]] = []
    for idx, node in enumerate(model.graph.node):
        op = node.op_type
        name = node.name or f"{op}_{idx}"
        fl = 0

        if op == "Conv":
            fl = flops_conv(node, vimap, init_map, batch_override)
            has_bias = len(node.input) > 2 and bool(node.input[2])
            if has_bias:
                fl += flops_per_output_element(node, vimap, batch_override)

        elif op == "QLinearConv":
            fl = flops_qlinearconv(node, vimap, init_map, batch_override)
            has_bias = len(node.input) > 8 and bool(node.input[8])
            if has_bias:
                fl += flops_per_output_element(node, vimap, batch_override)

        elif op == "ConvTranspose":
            fl = flops_convtranspose(node, vimap, init_map, batch_override)
            has_bias = len(node.input) > 2 and bool(node.input[2])
            if has_bias:
                fl += flops_per_output_element(node, vimap, batch_override)

        elif op == "MatMul":
            fl = flops_matmul(node, vimap)

        elif op == "QLinearMatMul":
            fl = flops_qlinearmatmul(node, vimap)

        elif op == "Gemm":
            fl = flops_gemm(node, vimap)
            has_bias = len(node.input) > 2 and bool(node.input[2])
            if has_bias:
                fl += flops_per_output_element(node, vimap, batch_override)

        elif op in ("BatchNormalization",):
            fl = flops_batchnorm(node, vimap, batch_override, ops_per_elt=bn_cost_per_elt)

        elif op in ("Relu", "LeakyRelu", "Sigmoid", "Tanh", "Softsign", "Softplus", "Gelu", "Clip"):
            fl = flops_activation(node, vimap, batch_override, ops_per_elt=act_cost_per_elt)

        elif op in ("Resize", "Upsample", "Interpolate"):
            fl = flops_resize(node, vimap, batch_override, default_ops=resize_cost_per_elt)

        elif op in ("Add", "Sub", "Mul", "Div"):
            fl = flops_per_output_element(node, vimap, batch_override)

        else:
            fl = 0

        recs.append((idx, op, name, int(fl)))

    return recs


def compute_boundary_flops_prefix(order: List[int], flops_by_node: Dict[int, int]) -> List[int]:
    """Return prefix FLOPs for each boundary index b (sum of nodes positions <= b)."""
    run = 0
    prefix: List[int] = []
    for node_idx in order:
        run += int(flops_by_node.get(node_idx, 0))
        prefix.append(int(run))
    # boundaries are between nodes -> N-1 boundaries
    return prefix[:-1]


# ---------------------------- Score ranking ----------------------------

def compute_scores_for_candidates(
    candidates: List[int],
    costs_bytes: List[int],
    crossing_counts: List[int],
    flops_left_prefix: List[int],
    total_flops: float,
    *,
    w_comm: float = 1.0,
    w_imb: float = 3.0,
    w_tensors: float = 0.2,
    linear_comm: bool = False,
) -> Dict[int, float]:
    """Compute a weighted score for each candidate boundary.

    Lower score is better.
    """

    if not candidates:
        return {}

    comm_raw: List[float] = []
    imb_raw: List[float] = []
    ten_raw: List[float] = []

    for b in candidates:
        cb = float(costs_bytes[b])
        if not linear_comm:
            cb = math.log10(1.0 + cb)
        comm_raw.append(cb)

        fl_l = float(flops_left_prefix[b])
        fl_r = float(total_flops) - fl_l
        imb = abs(fl_l - fl_r) / float(total_flops) if float(total_flops) > 0 else 0.0
        imb_raw.append(float(imb))

        t = max(0.0, float(int(crossing_counts[b]) - 1))
        ten_raw.append(float(t))

    def _norm(xs: List[float]) -> List[float]:
        if not xs:
            return []
        mn = min(xs)
        mx = max(xs)
        den = (mx - mn) if (mx > mn) else 1.0
        return [(x - mn) / den for x in xs]

    comm_n = _norm(comm_raw)
    imb_n = _norm(imb_raw)
    ten_n = _norm(ten_raw)

    scores: Dict[int, float] = {}
    for i, b in enumerate(candidates):
        scores[b] = float(w_comm) * comm_n[i] + float(w_imb) * imb_n[i] + float(w_tensors) * ten_n[i]

    return scores
