"""Public API surface (compatibility layer).

This module re-exports the most commonly used functions/classes so the GUI and
external scripts can simply import a single module.
"""

from __future__ import annotations

from . import __version__

# Units
from .units import BANDWIDTH_MULT, FLOP_UNITS, UNIT_MULT, bandwidth_to_bytes_per_s

# System model
from .system_model import ComputeSpec, LinkConstraints, LinkModelSpec, MemoryConstraints, SystemSpec

# Pruning
from .pruning import detect_skip_blocks, prune_candidates_skip_block

# ONNX parsing utilities
from .onnx_utils import (
    backfill_quant_shapes,
    build_producers_consumers,
    dtype_nbytes,
    elemtype_from_vi,
    numel_from_shape,
    shape_from_vi,
    topo_sort,
    value_info_map,
)

# Metrics / ranking
from .metrics import (
    boundary_costs,
    boundary_tensor_counts,
    collect_crossing_values_for_boundary,
    compute_boundary_flops_prefix,
    compute_scores_for_candidates,
    compute_tensor_bytes_per_value,
    peak_activation_memory_per_boundary,
    per_node_flops,
)

# Splitting + export helpers
from .split_export import (
    build_submodel,
    compute_strict_boundary_ok,
    cut_tensors_for_boundary,
    ensure_external_data_files,
    export_boundary_graphviz_context,
    get_value_info_or_fallback,
    infer_shapes_safe,
    make_fallback_value_info,
    make_random_inputs,
    model_external_data_locations,
    rename_value_in_model,
    save_model,
    split_model_on_cut_tensors,
    strict_boundary_extras,
    validate_split_onnxruntime,
    write_netron_launcher,
    write_runner_skeleton,
    write_runner_skeleton_onnxruntime,
)

__all__ = [
    "__version__",
    # units
    "UNIT_MULT",
    "BANDWIDTH_MULT",
    "FLOP_UNITS",
    "bandwidth_to_bytes_per_s",
    # system model
    "ComputeSpec",
    "LinkConstraints",
    "LinkModelSpec",
    "MemoryConstraints",
    "SystemSpec",
    # pruning
    "detect_skip_blocks",
    "prune_candidates_skip_block",
    # onnx utils
    "dtype_nbytes",
    "shape_from_vi",
    "numel_from_shape",
    "elemtype_from_vi",
    "value_info_map",
    "build_producers_consumers",
    "topo_sort",
    "backfill_quant_shapes",
    # metrics
    "compute_tensor_bytes_per_value",
    "boundary_costs",
    "peak_activation_memory_per_boundary",
    "boundary_tensor_counts",
    "collect_crossing_values_for_boundary",
    "per_node_flops",
    "compute_boundary_flops_prefix",
    "compute_scores_for_candidates",
    # splitting/export
    "infer_shapes_safe",
    "make_fallback_value_info",
    "get_value_info_or_fallback",
    "build_submodel",
    "strict_boundary_extras",
    "compute_strict_boundary_ok",
    "rename_value_in_model",
    "cut_tensors_for_boundary",
    "model_external_data_locations",
    "ensure_external_data_files",
    "export_boundary_graphviz_context",
    "split_model_on_cut_tensors",
    "save_model",
    "make_random_inputs",
    "validate_split_onnxruntime",
    "write_runner_skeleton_onnxruntime",
    "write_runner_skeleton",
    "write_netron_launcher",
]
