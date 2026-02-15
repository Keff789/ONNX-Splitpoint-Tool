"""Core-level lightweight analysis helpers used by smoke tests and scripts."""

from __future__ import annotations

from typing import Any, Dict, Optional

import onnx

from .metrics import (
    boundary_costs,
    compute_boundary_flops_prefix,
    compute_tensor_bytes_per_value,
    peak_activation_memory_per_boundary,
    per_node_flops,
)
from .onnx_utils import build_producers_consumers, topo_sort, value_info_map
from .split_export import infer_shapes_safe


def analyze_model(
    model_path: str,
    *,
    batch_override: Optional[int] = None,
    assume_bpe: Optional[int] = None,
    min_gap: int = 0,
) -> Dict[str, Any]:
    """Run a lightweight end-to-end analysis for one ONNX model.

    Returns communication costs, peak activation arrays and candidate boundaries.
    """
    model = onnx.load(model_path, load_external_data=False)
    model = infer_shapes_safe(model)

    nodes, producer_of, consumers_of = build_producers_consumers(model)
    order = topo_sort(nodes, producer_of)
    vimap = value_info_map(model)
    value_bytes = compute_tensor_bytes_per_value(vimap, batch_override, assume_bpe)

    costs_bytes, val_span = boundary_costs(order, producer_of, consumers_of, value_bytes)
    peak_l, peak_r, peak_max = peak_activation_memory_per_boundary(costs_bytes)

    node_flops_list = per_node_flops(
        model,
        vimap,
        batch_override,
        bn_cost_per_elt=4,
        act_cost_per_elt=1,
        resize_cost_per_elt=1,
    )
    flops_by_node = {idx: fl for (idx, _, __, fl) in node_flops_list}
    total_flops = float(sum(flops_by_node.values()))
    flops_left_prefix = compute_boundary_flops_prefix(order, flops_by_node)
    imbalance = [
        abs(float(flops_left_prefix[b]) - float(total_flops - flops_left_prefix[b])) / total_flops if total_flops > 0 else 0.0
        for b in range(len(costs_bytes))
    ]

    start = max(0, int(min_gap))
    end = max(start, len(costs_bytes) - max(0, int(min_gap)))
    candidates = list(range(start, end))

    return {
        "model": model,
        "nodes": nodes,
        "order": order,
        "value_bytes": value_bytes,
        "val_span": val_span,
        "costs_bytes": costs_bytes,
        "peak_act_mem_left_bytes": peak_l,
        "peak_act_mem_right_bytes": peak_r,
        "peak_act_mem_max_bytes": peak_max,
        "node_flops_list": node_flops_list,
        "flops_left_prefix": flops_left_prefix,
        "total_flops": total_flops,
        "imbalance": imbalance,
        "candidate_bounds": candidates,
        "candidates": candidates,
    }
