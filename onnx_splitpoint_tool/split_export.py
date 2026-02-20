"""ONNX model splitting and export utilities.

This module used to be a single large file. It has been split into smaller modules:

- split_export_graph.py: ONNX graph manipulation and splitting primitives
- split_export_validation.py: optional ONNXRuntime validation helpers
- split_export_runners.py: helper script generators (runner skeletons, netron launcher, etc.)

For backwards compatibility, this file re-exports the public API.
"""

from __future__ import annotations

# Re-export public API (and keep legacy attribute access working).
from .split_export_graph import *  # noqa: F401,F403
from .split_export_validation import *  # noqa: F401,F403
from .split_export_runners import *  # noqa: F401,F403

__all__ = [
    # Graph primitives
    "make_fallback_value_info",
    "get_value_info_or_fallback",
    "infer_shapes_safe",
    "build_submodel",
    "strict_boundary_extras",
    "compute_strict_boundary_ok",
    "rename_value_in_model",
    "cut_tensors_for_boundary",
    "export_boundary_graphviz_context",
    "split_model_on_cut_tensors",
    "save_model",
    "model_external_data_locations",
    "ensure_external_data_files",
    # Validation
    "make_random_inputs",
    "validate_split_onnxruntime",
    # Helper scripts
    "write_runner_skeleton_onnxruntime",
    "write_runner_skeleton",
    "write_netron_launcher",
    "write_benchmark_set_json",
    "write_benchmark_set_runner",
    "write_dummy_test_image",
    "main",
]
