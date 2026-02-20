"""ONNX split/export: graph primitives.

This module contains ONNX graph manipulation and model-splitting utilities.
It was extracted from split_export.py to keep modules small and maintainable.
"""

from __future__ import annotations

import copy
import json
import math
import logging
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, shape_inference

from .onnx_utils import (
    backfill_custom_op_passthrough_shapes,
    build_producers_consumers,
    elemtype_from_vi,
    shape_from_vi,
    topo_sort,
    value_info_map,
)

LOGGER = logging.getLogger("onnx_splitpoint_tool.split_export")
from .metrics import compute_tensor_bytes_per_value


def _infer_shapes_ort_symbolic_safe(model: onnx.ModelProto) -> onnx.ModelProto:
    """Best-effort symbolic shape inference via onnxruntime (if available).

    ORT's symbolic shape inference can resolve many dynamic shapes that
    `onnx.shape_inference` can't (especially transformer graphs). We keep this
    optional and *never* fail hard if ORT isn't present or inference errors.
    """
    try:
        # onnxruntime is an optional dependency for this tool.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference  # type: ignore
    except Exception:
        return model

    # The API signature has changed a few times across ORT versions. We use a
    # small set of robust fallbacks.
    try:
        # 1) Classmethod/staticmethod style
        if hasattr(SymbolicShapeInference, "infer_shapes"):
            fn = getattr(SymbolicShapeInference, "infer_shapes")
            if callable(fn):
                try:
                    return fn(
                        model,
                        int_max=2**31 - 1,
                        auto_merge=True,
                        guess_output_rank=True,
                        verbose=0,
                    )
                except TypeError:
                    # Try minimal signature.
                    try:
                        return fn(model)
                    except TypeError:
                        pass

        # 2) Instance method style
        try:
            try:
                inf = SymbolicShapeInference(
                    int_max=2**31 - 1,
                    auto_merge=True,
                    guess_output_rank=True,
                    verbose=0,
                )
            except TypeError:
                inf = SymbolicShapeInference()  # type: ignore

            if hasattr(inf, "infer_shapes"):
                fn2 = getattr(inf, "infer_shapes")
                if callable(fn2):
                    try:
                        return fn2(model)
                    except TypeError:
                        # Some variants expose the same kwargs on the method.
                        return fn2(
                            model,
                            int_max=2**31 - 1,
                            auto_merge=True,
                            guess_output_rank=True,
                            verbose=0,
                        )
        except Exception:
            pass
    except Exception:
        return model

    return model


def infer_shapes_safe(
    model: onnx.ModelProto,
    *,
    use_ort_symbolic: bool = False,
    custom_op_passthrough: bool = True,
) -> onnx.ModelProto:
    """Try to run ONNX shape inference.

    Shape inference is useful for comm estimation + nicer I/O ValueInfo, but
    splitting should still work even if inference fails.

    When `use_ort_symbolic` is True, we also attempt ORT's symbolic shape
    inference as a second pass.

    When `custom_op_passthrough` is True, we apply a pragmatic passthrough
    rule for custom/fused ops (output shape == input shape) to avoid inference
    chains breaking on unknown nodes.
    """
    out = model

    # 1) ONNX shape inference
    try:
        out = shape_inference.infer_shapes(out, strict_mode=False)
    except TypeError:
        # Older onnx may not accept strict_mode
        try:
            out = shape_inference.infer_shapes(out)
        except Exception as e:
            LOGGER.debug("onnx.shape_inference failed (continuing): %s", e)
    except Exception as e:
        LOGGER.debug("onnx.shape_inference failed (continuing): %s", e)

    # 2) ORT symbolic shape inference (optional)
    if use_ort_symbolic:
        out = _infer_shapes_ort_symbolic_safe(out)

    # 3) Passthrough for custom/fused ops (optional)
    if custom_op_passthrough:
        try:
            n_upd = backfill_custom_op_passthrough_shapes(out, max_iters=2)
            if n_upd:
                LOGGER.info("Custom-op passthrough: filled %d missing ValueInfo shapes", n_upd)
                # Rerun inference to propagate beyond newly-typed tensors
                try:
                    out = shape_inference.infer_shapes(out, strict_mode=False)
                except TypeError:
                    out = shape_inference.infer_shapes(out)
                except Exception as e:
                    LOGGER.debug("onnx.shape_inference (2nd pass) failed (continuing): %s", e)

                if use_ort_symbolic:
                    out = _infer_shapes_ort_symbolic_safe(out)
        except Exception as e:
            LOGGER.debug("custom-op passthrough failed (continuing): %s", e)

    return out


def make_fallback_value_info(name: str):
    """Create a minimal ValueInfoProto with float tensor and unknown shape."""
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)


def get_value_info_or_fallback(vimap: Dict[str, object], name: str):
    if name in vimap:
        return copy.deepcopy(vimap[name])
    return make_fallback_value_info(name)


def _producer_map_from_graph_nodes(nodes: List[onnx.NodeProto]) -> Dict[str, int]:
    produced_by: Dict[str, int] = {}
    for idx, node in enumerate(nodes):
        for out in node.output:
            if out:
                produced_by[out] = idx
    return produced_by


def _collect_required_node_indices(
    nodes: List[onnx.NodeProto],
    *,
    output_names: List[str],
    stop_names: set,
    initializer_names: set,
) -> set:
    """Backward slice: nodes needed to compute output_names from stop_names."""
    produced_by = _producer_map_from_graph_nodes(nodes)

    required: set = set()
    visited_values: set = set()
    stack: List[str] = list(output_names)
    while stack:
        v = stack.pop()
        if not v or v in visited_values:
            continue
        visited_values.add(v)

        if v in stop_names:
            continue
        if v in initializer_names:
            continue

        if v not in produced_by:
            # No producer: external input (graph input) or optional missing.
            continue

        node_idx = produced_by[v]
        if node_idx not in required:
            required.add(node_idx)
            node = nodes[node_idx]
            for inp in node.input:
                if inp:
                    stack.append(inp)
    return required


def _compute_external_inputs(nodes: List[onnx.NodeProto], initializer_names: set) -> List[str]:
    produced = set()
    used = set()
    for n in nodes:
        for o in n.output:
            if o:
                produced.add(o)
        for i in n.input:
            if i:
                used.add(i)
    return sorted((used - produced) - initializer_names)


def build_submodel(
    full_model: onnx.ModelProto,
    *,
    outputs: List[str],
    stop_names: set,
    model_name: str,
    force_inputs: Optional[List[str]] = None,
) -> Tuple[onnx.ModelProto, List[str]]:
    """Create an ONNX submodel via backward slicing.

    Returns: (model, external_input_names)
    """
    full_model = infer_shapes_safe(full_model)
    g = full_model.graph
    nodes_full = list(g.node)

    initializer_names = {init.name for init in g.initializer}
    vimap = value_info_map(full_model)

    required_idx = _collect_required_node_indices(
        nodes_full,
        output_names=list(outputs),
        stop_names=set(stop_names),
        initializer_names=initializer_names,
    )
    # Ensure topological node order in the exported subgraph.
    # Some exporters already provide a topologically sorted node list, but not all do.
    nodes_for_sort, producer_of, _ = build_producers_consumers(full_model)
    order = topo_sort(nodes_for_sort, producer_of)
    nodes = [copy.deepcopy(nodes_full[i]) for i in order if i in required_idx]

    external_inputs = _compute_external_inputs(nodes, initializer_names)
    if force_inputs is not None:
        forced = list(force_inputs)
        for x in external_inputs:
            if x not in forced:
                forced.append(x)
        external_inputs = forced

    # Collect required initializers
    needed_init_names = set()
    for n in nodes:
        for inp in n.input:
            if inp in initializer_names:
                needed_init_names.add(inp)
    initializers = [copy.deepcopy(init) for init in g.initializer if init.name in needed_init_names]

    inputs_vi = [get_value_info_or_fallback(vimap, name) for name in external_inputs]
    outputs_vi = [get_value_info_or_fallback(vimap, name) for name in outputs]

    new_graph = helper.make_graph(
        nodes=nodes,
        name=model_name,
        inputs=inputs_vi,
        outputs=outputs_vi,
        initializer=initializers,
    )

    # Keep some value_info for better tooling / debuggability
    keep_names = set(external_inputs) | set(outputs)
    for n in nodes:
        keep_names.update([x for x in n.input if x])
        keep_names.update([x for x in n.output if x])
    for vi in g.value_info:
        if vi.name in keep_names:
            new_graph.value_info.append(copy.deepcopy(vi))

    new_model = helper.make_model(
        new_graph,
        producer_name="kmd-onnx-split",
        opset_imports=list(full_model.opset_import),
    )
    new_model.ir_version = full_model.ir_version
    new_model.producer_version = full_model.producer_version
    new_model.domain = full_model.domain
    new_model.model_version = full_model.model_version
    new_model.doc_string = full_model.doc_string
    for mp in full_model.metadata_props:
        new_model.metadata_props.append(copy.deepcopy(mp))

    # NOTE: For external-data models (large weights stored in a separate *.data file),
    # `onnx.checker.check_model(ModelProto)` may fail because it cannot resolve
    # relative paths for external tensors without a filesystem base path.
    # In that case we run a lighter check (full_check=False) if available, or we
    # allow the build to proceed and validate later when saving with linked data.
    uses_external_data = any(
        (init.data_location == TensorProto.EXTERNAL) or bool(init.external_data)
        for init in new_model.graph.initializer
    )

    try:
        if uses_external_data:
            try:
                onnx.checker.check_model(new_model, full_check=False)  # type: ignore[arg-type]
            except TypeError:
                # Older ONNX versions may not support the `full_check` keyword.
                onnx.checker.check_model(new_model)
        else:
            onnx.checker.check_model(new_model)
    except Exception as e:
        msg = str(e)
        if uses_external_data and (
            "Data of TensorProto" in msg
            or "should be stored" in msg
            or "doesn't exist" in msg
            or "not accessible" in msg
            or "external" in msg.lower()
        ):
            LOGGER.warning(
                "ONNX checker could not verify external tensor data for submodel '%s' (%s). "
                "Proceeding; external data will be validated when saving/loading.",
                model_name,
                msg,
            )
        else:
            raise RuntimeError(f"ONNX checker failed for submodel '{model_name}': {e}") from e

    return new_model, external_inputs


def strict_boundary_extras(full_model: onnx.ModelProto, cut_tensors: Iterable[str]) -> List[str]:
    """Return extra *non-input* tensors required by part2 besides the cut tensors.

    Historically the tool treated a boundary as *strict* only if part2 required **only** the
    cut tensors (and initializers) as inputs.

    For many models (especially transformers), part2 still legitimately depends on one or more
    original model inputs (e.g. `attention_mask`, `position_ids`, ...). These are not *produced*
    by part1 and therefore are not part of the inter-device communication budget. They are also
    typically available to both parts.

    We therefore define "strict" as:

    - part2 requires no **additional intermediate activations** besides the cut tensors;
    - original graph inputs are allowed.

    The returned list contains only additional tensors that are neither cut tensors nor original
    model inputs.
    """

    g = full_model.graph
    nodes_full = list(g.node)

    orig_outputs = [o.name for o in g.output]
    orig_inputs = [i.name for i in g.input]
    init_names = {init.name for init in g.initializer}

    cut_set = set(cut_tensors)
    stop = cut_set | set(orig_inputs)

    # Backward slice: which nodes are required to produce the original outputs,
    # *assuming* cut tensors (and original inputs) are externally provided?
    needed_idxs = _collect_required_node_indices(
        nodes_full,
        output_names=orig_outputs,
        stop_names=stop,
        initializer_names=init_names,
    )
    needed_nodes = [nodes_full[i] for i in needed_idxs]

    external_inputs = _compute_external_inputs(needed_nodes, initializer_names=init_names)

    # Allow original graph inputs (they are not produced by part1).
    orig_input_set = set(orig_inputs)
    extras = sorted([x for x in external_inputs if x not in cut_set and x not in orig_input_set])
    return extras


def compute_strict_boundary_ok(
    full_model: onnx.ModelProto,
    order: List[int],
    nodes: List[onnx.NodeProto],
) -> Tuple[List[bool], List[List[str]]]:
    """Compute whether each boundary is *strict* and (if not) which extra inputs appear.

    A boundary is considered *strict* if the right subgraph (part2) depends only on the cut
    activation tensors (plus initializers and original graph inputs). In other words, it must
    not require additional *intermediate* activations from the left side.

    Returns:
        strict_ok: list[bool] of length (len(order) - 1)
        strict_extras: list[list[str]] of same length; names of extra required inputs
    """

    n_bounds = max(0, len(order) - 1)
    strict_ok: List[bool] = [True] * n_bounds
    strict_extras: List[List[str]] = [[] for _ in range(n_bounds)]

    for b in range(n_bounds):
        try:
            cut = cut_tensors_for_boundary(order, nodes, b)
            extras = strict_boundary_extras(full_model, cut)
            strict_extras[b] = extras
            strict_ok[b] = (len(extras) == 0)
        except Exception:
            # Be conservative: if we cannot determine strictness, mark as non-strict.
            strict_ok[b] = False
            strict_extras[b] = ["<unknown>"]

    return strict_ok, strict_extras


def rename_value_in_model(model: onnx.ModelProto, old: str, new: str) -> None:
    """Rename a value everywhere it can appear (inputs/outputs/value_info/node inputs/outputs)."""
    if old == new:
        return
    g = model.graph
    for n in g.node:
        for i in range(len(n.input)):
            if n.input[i] == old:
                n.input[i] = new
        for i in range(len(n.output)):
            if n.output[i] == old:
                n.output[i] = new

    for coll in (g.input, g.output, g.value_info):
        for vi in coll:
            if vi.name == old:
                vi.name = new
    for init in g.initializer:
        if init.name == old:
            init.name = new


def cut_tensors_for_boundary(order: List[int], nodes: List[onnx.NodeProto], b: int) -> List[str]:
    """Return all tensors crossing a topo boundary b.

    Boundary b splits between order[b] (left) and order[b+1] (right).
    Crossing tensors are outputs of left nodes consumed by right nodes.

    This is independent of whether shape inference could estimate their sizes.
    """
    if b < 0 or b >= len(order) - 1:
        raise ValueError("Boundary index out of range")

    left = [nodes[i] for i in order[: b + 1]]
    right = [nodes[i] for i in order[b + 1 :]]

    right_inputs = set()
    for n in right:
        for inp in n.input:
            if inp:
                right_inputs.add(inp)

    cut: List[str] = []
    seen = set()
    for n in left:
        for out in n.output:
            if out and out in right_inputs and out not in seen:
                cut.append(out)
                seen.add(out)
    return cut


# --------------------------- Export: graph visualisation ---------------------------


def export_boundary_graphviz_context(
    model: onnx.ModelProto,
    order: List[int],
    boundary_index: int,
    cut_tensors: List[str],
    out_dir: str,
    *,
    hops: int = 2,
    basename: str = "split_context",
    render: bool = True,
    max_edge_label: int = 48,
    include_internal_consumers: bool = True,
    cut_flow_only: bool = False,
    include_external_inputs: bool = True,
    # GUI/API-compat extras (safe to ignore for classic models)
    strict_boundary: bool = False,
    semantic_label: Optional[str] = None,
    value_bytes_map: Optional[Dict[str, int]] = None,
    force_matplotlib_fallback: bool = False,
    matplotlib_max_nodes: int = 80,
) -> Dict[str, Optional[str]]:
    """Export a small GraphViz diagram around a split boundary.

    This produces a *context subgraph* around the cut tensors (producer + consumers),
    expanded by `hops` edges upstream/downstream. It is meant as a lightweight
    alternative to Netron screenshots for papers/debugging.

    Files written into `out_dir`:
      - `<basename>.dot` always
      - `<basename>.pdf`, `<basename>.svg`, `<basename>.png` if GraphViz `dot` is available

    Visual semantics (to reduce ambiguity)
    -------------------------------------
    The DOT renderer draws **cut tensors as dedicated ellipse nodes** placed between the
    left and right subgraphs. This helps clarify two common confusions:

    1) A *single* cut tensor can feed *multiple* consumers on the right side (fan-out). That
       still counts as **one crossing tensor**, but would appear as multiple edges.
    2) A tensor can have consumers on both sides; only the edges into the opposite side are
       actual *boundary crossings*.

    Coloring:
      - left-side compute nodes: light blue fill, blue border
      - right-side compute nodes: light yellow fill, orange border
      - the two boundary nodes: light red fill, red border (thicker)
      - cut tensor nodes: light red ellipse with red border
      - edges into the right side (boundary crossings): thick red
      - edges to same-side consumers (optional): thin dashed gray

    Modes
    -----
    If `cut_flow_only=True`, the exported diagram removes auxiliary context branches and
    focuses on the *causal cut-flow* (ancestors of cut producers on the left, and
    descendants of cut consumers on the right). This is typically more paper-friendly.

    Returns
    -------
    Dict with keys: dot, pdf, svg, png (values are basenames or None).
    """
    # NOTE: 'strict_boundary' affects the actual split operation, not the diagram export.
    # We accept it here for GUI/API compatibility.
    _ = strict_boundary


    os.makedirs(out_dir, exist_ok=True)

    nodes, producer_of, consumers_of = build_producers_consumers(model)
    if not order or boundary_index < 0 or boundary_index >= len(order) - 1:
        raise ValueError(f"Invalid boundary_index={boundary_index} for order length {len(order)}")

    # De-duplicate cut tensors (preserve order)
    cut_list: List[str] = []
    seen_cut = set()
    for t in cut_tensors:
        if t and t not in seen_cut:
            cut_list.append(str(t))
            seen_cut.add(str(t))
    cut_set = set(cut_list)
    num_cut_all = len([t for t in cut_list if t])

    left_set = set(order[: boundary_index + 1])
    right_set = set(order[boundary_index + 1 :])

    # Seed set: producers/consumers of cut tensors
    seed = set()
    for t in cut_list:
        p = producer_of.get(t)
        if p is not None:
            seed.add(int(p))
        for c in consumers_of.get(t, []):
            seed.add(int(c))

    # Expand by hops (BFS)
    selected = set(seed)
    frontier = set(seed)
    for _ in range(max(0, int(hops))):
        nxt = set()
        for ni in frontier:
            n = nodes[ni]
            # upstream
            for inp in n.input:
                p = producer_of.get(inp)
                if p is not None:
                    nxt.add(int(p))
            # downstream
            for out in n.output:
                for c in consumers_of.get(out, []):
                    nxt.add(int(c))
        nxt -= selected
        selected |= nxt
        frontier = nxt

    # Always include the two boundary nodes themselves
    b_left = int(order[boundary_index])
    b_right = int(order[boundary_index + 1])
    selected.add(b_left)
    selected.add(b_right)

    # Best-effort enrichment: shape/dtype/size for tensors (cut + external placeholders)
    vimap = None
    vb: Optional[Dict[str, int]] = None
    try:
        vimap = value_info_map(infer_shapes_safe(model))
        vb = compute_tensor_bytes_per_value(vimap, batch_override=None, assume_activation_bytes=None)
    except Exception:
        vimap = None
        vb = None

    def _shape_str(sh: Optional[List[Optional[int]]]) -> str:
        if not sh:
            return "?"
        return "[" + ",".join(str(int(d)) if (d is not None and int(d) > 0) else "?" for d in sh) + "]"

    def _dtype_str(et: Optional[int]) -> str:
        if not et:
            return "?"
        try:
            return onnx.TensorProto.DataType.Name(int(et))
        except Exception:
            return str(int(et))

    def _tensor_meta(name: str) -> Dict[str, str]:
        if vimap is None or vb is None:
            return {}
        vi = vimap.get(name)
        return {
            "shape": _shape_str(shape_from_vi(vi)),
            "dtype": _dtype_str(elemtype_from_vi(vi)),
            "mib": f"{(vb.get(name, 0) or 0) / (1024 * 1024):.3f}",
        }

    # Precompute meta for cut tensors (used for ellipse labels)
    cut_meta: Dict[str, Dict[str, str]] = {t: _tensor_meta(t) for t in cut_list}

    # Identify "main" cut-flow nodes vs auxiliary context.
    #
    # Motivation: in graphs with concatenations / skip connections, consumers of the cut
    # tensor often have *additional inputs* produced by other right-side branches.
    # Those branches are useful context, but can be visually confusing. We classify
    # nodes that are not on the causal chain (ancestors of cut producers on the left,
    # descendants of cut consumers on the right) as "aux".
    cut_producers = {producer_of.get(t) for t in cut_list if producer_of.get(t) is not None}
    cut_producers = {int(x) for x in cut_producers if x is not None and int(x) in selected}

    cut_consumers_right = set()
    for t in cut_list:
        for c in consumers_of.get(t, []):
            ci = int(c)
            if ci in selected and ci in right_set:
                cut_consumers_right.add(ci)

    # Left-flow: ancestors of cut producers (restricted to selected + left_set)
    left_flow = set([b_left])
    q_up = deque([n for n in cut_producers if n in left_set])
    while q_up:
        ni = int(q_up.popleft())
        if ni in left_flow:
            continue
        left_flow.add(ni)
        for inp in nodes[ni].input:
            p = producer_of.get(inp)
            if p is None:
                continue
            pj = int(p)
            if pj in selected and pj in left_set and pj not in left_flow:
                q_up.append(pj)

    # Right-flow: descendants of cut consumers (restricted to selected + right_set)
    right_flow = set([b_right])
    q_dn = deque(list(cut_consumers_right) + ([b_right] if b_right in right_set else []))
    while q_dn:
        ni = int(q_dn.popleft())
        if ni in right_flow:
            continue
        if ni not in right_set:
            continue
        right_flow.add(ni)
        for out in nodes[ni].output:
            if not out:
                continue
            for c in consumers_of.get(out, []):
                ci = int(c)
                if ci in selected and ci in right_set and ci not in right_flow:
                    q_dn.append(ci)

    def _is_aux_node(ni: int) -> bool:
        if ni == b_left or ni == b_right:
            return False
        if ni in left_set:
            return ni not in left_flow
        if ni in right_set:
            return ni not in right_flow
        return True

    # Optional: prune to the causal cut-flow only (paper-friendly view).
    if cut_flow_only:
        selected = set(left_flow) | set(right_flow) | {b_left, b_right}

    initializer_names = {i.name for i in model.graph.initializer}
    graph_input_names = {vi.name for vi in model.graph.input}

    # Filter cut tensors: skip initializers and outputs of Constant nodes (visual noise)
    const_tensor_names: Set[str] = set()
    for t, prod_ix in producer_of.items():
        try:
            if prod_ix is not None and 0 <= int(prod_ix) < len(nodes) and nodes[int(prod_ix)].op_type == "Constant":
                const_tensor_names.add(t)
        except Exception:
            continue
    cut_list = [t for t in cut_list if t and t not in initializer_names and t not in const_tensor_names]
    num_cut_omitted = max(0, num_cut_all - len(cut_list))

    # Drop Constant ops from the context graph to keep it readable
    selected = {i for i in selected if nodes[i].op_type != "Constant"}


    # Helper: escape labels for DOT
    def esc(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\"')

    def short_edge_label(s: str) -> str:
        s = str(s)

        # Prefer the tail of long hierarchical ONNX names (".../layers.20/.../output_0")
        # to keep diagrams readable. This is purely a visualization aid.
        if "/" in s:
            parts = [p for p in s.split("/") if p]
            if len(parts) >= 4:
                s = "/".join(parts[-4:])
            elif len(parts) >= 2:
                s = "/".join(parts[-2:])
            else:
                s = parts[-1] if parts else s

        if max_edge_label and len(s) > int(max_edge_label):
            return s[: max(0, int(max_edge_label) - 3)] + "..."
        return s

    dot_path = os.path.join(out_dir, f"{basename}.dot")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    svg_path = os.path.join(out_dir, f"{basename}.svg")
    # Colors (GraphViz understands hex colors)
    col_left_fill = "#d9ecff"
    col_right_fill = "#fff3d9"
    col_boundary_fill = "#ffd6d6"

    col_left_border = "#2563eb"
    col_right_border = "#f59e0b"
    col_boundary_border = "#e11d48"
    col_internal_edge = "#777777"

    # Cluster background (very light to keep node colors readable)
    col_cluster_left_bg = "#f4faff"
    col_cluster_right_bg = "#fffaf0"

    def _node_stmt(ni: int) -> str:
        n = nodes[ni]
        label = f"{ni}: {n.op_type}"
        if n.name:
            label += f"\n{short_edge_label(n.name)}"
        fill = "white"
        border = "black"
        pen = 1.2
        style = "filled"

        # Base coloring encodes side membership.
        if ni == b_left or ni == b_right:
            fill = col_boundary_fill
            border = col_boundary_border
            pen = 2.2
            style = "filled"
        elif ni in left_set:
            fill = col_left_fill
            border = col_left_border
            pen = 1.4
        elif ni in right_set:
            fill = col_right_fill
            border = col_right_border
            pen = 1.4

        # Auxiliary nodes are context-only (e.g. side-input producers feeding into a concat).
        # They are drawn with a dashed outline and a neutral border to avoid the impression
        # that they are part of the cut-flow.
        if _is_aux_node(int(ni)) and not (ni == b_left or ni == b_right):
            border = "#94a3b8"
            style = "filled,dashed"
        return f'    n{ni} [label="{esc(label)}", style="{style}", fillcolor="{fill}", color="{border}", penwidth={pen}];'

    lines: List[str] = []
    lines.append('digraph G {')
    lines.append('  rankdir=LR;')
    lines.append('  graph [fontname="Helvetica", fontsize=10];')
    lines.append('  node  [shape=box, fontname="Helvetica", fontsize=10];')
    lines.append('  edge  [fontname="Helvetica", fontsize=9];')
    lines.append('  labelloc="t";')
    _legend = "red ellipses = cut tensors"
    if include_external_inputs:
        _legend += ", dashed ellipses = external inputs"
    if not cut_flow_only:
        _legend += ", dashed boxes = auxiliary context"
    _mode = "cut-flow" if cut_flow_only else "context"
    _sem = f"  |  {semantic_label}" if semantic_label else ""
    _sem = _sem.replace('\"', "'")
    lines.append(
        f'  label="Split boundary {boundary_index}{_sem}  |  #cut_tensors={len(cut_list)} (+{num_cut_omitted} const/init omitted)  |  {_mode}\\n({_legend})";'
    )
    lines.append('')

    # Left/right clusters (helps show which side a node belongs to)
    lines.append('  subgraph cluster_left {')
    lines.append('    label="Left";')
    lines.append(f'    color="{col_left_border}";')
    lines.append(f'    style="rounded,filled";')
    lines.append(f'    fillcolor="{col_cluster_left_bg}";')
    for ni in sorted(selected):
        if ni in left_set:
            lines.append(_node_stmt(int(ni)))
    lines.append('  }')
    lines.append('')

    lines.append('  subgraph cluster_right {')
    lines.append('    label="Right";')
    lines.append(f'    color="{col_right_border}";')
    lines.append(f'    style="rounded,filled";')
    lines.append(f'    fillcolor="{col_cluster_right_bg}";')
    for ni in sorted(selected):
        if ni in right_set:
            lines.append(_node_stmt(int(ni)))
    lines.append('  }')
    lines.append('')

    # Cut tensor nodes (ellipse, between clusters)
    cut_node_ids: Dict[str, str] = {}
    for i, t in enumerate(cut_list):
        vid = f"v{i}"
        cut_node_ids[t] = vid
        meta = cut_meta.get(t, {})
        t_short = short_edge_label(t)
        if meta:
            label = f"{t_short}\n{meta.get('shape','?')} {meta.get('dtype','?')}\n{meta.get('mib','?')} MiB"
        else:
            label = t_short
        lines.append(
            f'  {vid} [shape=ellipse, label="{esc(label)}", style="filled,bold", fillcolor="{col_boundary_fill}", color="{col_boundary_border}", penwidth=2.0];'
        )
    if cut_list:
        lines.append('')

    if include_external_inputs:
        # External-input placeholders
        # --------------------------
        # If a selected node consumes an activation tensor whose producer lies *outside*
        # the selected context window (because `hops` is small), GraphViz would otherwise
        # render the node as a "source" with no incoming edges. These placeholders make
        # such dependencies explicit and answer the common question: "where does this node
        # get its data from?".
        ext_node_ids: Dict[str, str] = {}
        ext_nodes: List[str] = []
        ext_edges: List[str] = []
        ext_ctr = 0
        for dst in sorted(selected):
            dn = nodes[int(dst)]
            for inp in dn.input:
                if not inp:
                    continue
                if inp in initializer_names:
                    # weights/biases are intentionally not drawn
                    continue
                if inp in cut_set:
                    # cut tensors are routed through dedicated ellipse nodes
                    continue

                p = producer_of.get(inp)
                if p is not None and int(p) in selected:
                    continue

                if inp not in ext_node_ids:
                    eid = f"e{ext_ctr}"
                    ext_ctr += 1
                    ext_node_ids[inp] = eid

                    meta = _tensor_meta(inp)
                    origin = "(input)" if inp in graph_input_names else "(external)"
                    if meta:
                        lbl = f"{inp}\n{origin}\n{meta.get('shape','?')} {meta.get('dtype','?')}\n{meta.get('mib','?')} MiB"
                    else:
                        lbl = f"{inp}\n{origin}"
                    ext_nodes.append(
                        f'  {eid} [shape=ellipse, label="{esc(lbl)}", style="dashed", color="#94a3b8", fontcolor="#334155"];'
                    )
                ext_edges.append(
                    f'  {ext_node_ids[inp]} -> n{int(dst)} [color="#94a3b8", style="dashed", penwidth=1.0];'
                )

        if ext_nodes:
            lines.append('  // External inputs into the context window (producers not shown)')
            lines.extend(ext_nodes)
            lines.extend(ext_edges)
            lines.append('')

    # Edges for cut tensors: route through the tensor-node.
    # We only paint *actual boundary crossings* in red.
    for t in cut_list:
        vid = cut_node_ids[t]
        p = producer_of.get(t)
        if p is None:
            # Graph input / constant without a node producer (rare for our cuts, but possible)
            src_placeholder = f"src_{vid}"
            lines.append(f'  {src_placeholder} [shape=plaintext, label="(input)"];')
            lines.append(f'  {src_placeholder} -> {vid} [color="{col_boundary_border}", penwidth=2.0];')
        else:
            p = int(p)
            if p in selected:
                lines.append(f'  n{p} -> {vid} [color="{col_boundary_border}", penwidth=2.0];')

        for c in consumers_of.get(t, []):
            ci = int(c)
            if ci not in selected:
                continue
            if ci in right_set:
                # crossing edge
                lines.append(f'  {vid} -> n{ci} [color="{col_boundary_border}", penwidth=2.0];')
            else:
                # same-side consumer; optional
                if include_internal_consumers:
                    lines.append(
                        f'  {vid} -> n{ci} [color="{col_internal_edge}", style="dashed", penwidth=1.0];'
                    )

    if cut_list:
        lines.append('')

    # Other edges (only among selected nodes), excluding cut tensors to avoid duplicates.
    for src in sorted(selected):
        n = nodes[int(src)]
        for tname in n.output:
            if not tname:
                continue
            if tname in cut_set:
                continue
            for dst in consumers_of.get(tname, []):
                dst_i = int(dst)
                if dst_i not in selected:
                    continue

                # If something crosses the boundary but is not in cut_set, flag it (helps debugging).
                crosses = (int(src) in left_set) and (dst_i in right_set)
                color = col_boundary_border if crosses else "black"
                pen = 2.0 if crosses else 1.0
                style = "dotted" if crosses else "solid"
                lbl = esc(short_edge_label(tname))
                lines.append(
                    f'  n{int(src)} -> n{dst_i} [label="{lbl}", color="{color}", penwidth={pen}, style="{style}"];'
                )

    lines.append('}')
    Path(dot_path).write_text("\n".join(lines), encoding="utf-8")

    out: Dict[str, Optional[str]] = {
        "dot": os.path.basename(dot_path),
        "pdf": None,
        "svg": None,
        "png": None,
    }

    # GraphViz renderer (preferred)
    if render:
        dot_exe = shutil.which("dot")
        if render and dot_exe and (not force_matplotlib_fallback):
            # PDF
            try:
                r = subprocess.run(
                    [dot_exe, "-Tpdf", dot_path, "-o", pdf_path],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if r.returncode != 0:
                    LOGGER.warning(
                        "Graphviz 'dot' failed to render PDF (rc=%s). stderr: %s",
                        r.returncode,
                        (r.stderr.decode("utf-8", errors="ignore")[:600] if r.stderr else ""),
                    )
                if os.path.exists(pdf_path):
                    out["pdf"] = os.path.basename(pdf_path)
                else:
                    # Keep key absent so fallback can run.
                    out.pop("pdf", None)
            except Exception:
                out.pop("pdf", None)
                LOGGER.exception("Graphviz PDF render raised unexpectedly")

            # SVG
            try:
                r = subprocess.run(
                    [dot_exe, "-Tsvg", dot_path, "-o", svg_path],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if r.returncode != 0:
                    LOGGER.warning(
                        "Graphviz 'dot' failed to render SVG (rc=%s). stderr: %s",
                        r.returncode,
                        (r.stderr.decode("utf-8", errors="ignore")[:600] if r.stderr else ""),
                    )
                if os.path.exists(svg_path):
                    out["svg"] = os.path.basename(svg_path)
                else:
                    out.pop("svg", None)
            except Exception:
                out.pop("svg", None)
                LOGGER.exception("Graphviz SVG render raised unexpectedly")


    # Fallback renderer (matplotlib) if GraphViz is missing or rendering failed.
    # This yields a compact, paper-friendly diagram even without the `dot` executable.
    # Important: PNG rendering is optional/disabled by default. Do NOT treat a missing PNG as a render failure.
    # Need flags should reflect the actual files on disk, not just dict keys.
    need_pdf = not os.path.exists(pdf_path)
    need_svg = not os.path.exists(svg_path)
    if render and (need_pdf or need_svg):
        try:
            import matplotlib.pyplot as plt
            import textwrap as _tw

            def _mb(x_bytes: int) -> float:
                return float(x_bytes) / (1024.0 * 1024.0)

            def _short(s: str, max_len: int = 72) -> str:
                s = s or ""
                if len(s) <= max_len:
                    return s
                return s[: max(0, max_len - 3)] + "..."

            # Keep the fallback readable: shorten/wrap long labels, cap the cut tensor list.
            def _node_summary(ni: int) -> str:
                if ni < 0 or ni >= len(nodes):
                    return "(unknown)"
                n = nodes[ni]
                name = short_edge_label(n.name) if getattr(n, "name", "") else ""
                return f"{ni}: {n.op_type}{(' | ' + name) if name else ''}"

            def _wrap_keep_newlines(s: str, width: int) -> str:
                """Wrap each line individually while preserving explicit newlines."""
                out_lines = []
                for ln in (s or "").splitlines() or [""]:
                    ln = ln.rstrip("\r")
                    if not ln:
                        out_lines.append("")
                        continue
                    out_lines.append(_tw.fill(ln, width=width, break_long_words=True, break_on_hyphens=False))
                return "\n".join(out_lines)

            # Keep boundary summaries compact and wrapped so side boxes don't sprawl
            # into the center column (which caused visual overlap in dense diagrams).
            left_text = "LEFT\n" + _wrap_keep_newlines(
                _short(_node_summary(int(b_left)).replace("\n", " "), 90), width=34
            )
            right_text = "RIGHT\n" + _wrap_keep_newlines(
                _short(_node_summary(int(b_right)).replace("\n", " "), 90), width=34
            )

            total_cut = 0
            for t in cut_list:
                try:
                    total_cut += int(size_map.get(t, 0))
                except Exception:
                    continue

            # Show each cut tensor as its own box (much more readable than a
            # single giant text blob). Cap the list so the diagram stays usable.
            max_show = 22
            shown = cut_list[:max_show]
            omitted = max(0, len(cut_list) - len(shown))

            import re as _re

            def _shorten_path(name: str, keep_tail: int = 4) -> str:
                name = (name or "").strip()
                name = _re.sub(r"^/model/", "", name)
                parts = [p for p in name.split("/") if p]
                if len(parts) > keep_tail:
                    parts = parts[-keep_tail:]
                # Abbreviate layers.N -> LN
                parts = [_re.sub(r"^layers\.(\d+)$", r"L\1", p) for p in parts]
                return "\n".join(parts) if parts else name

            def _pretty_cut_item(raw: str, idx: int) -> str:
                # The cut list often comes as multi-line strings:
                #   <tensor_name>\n[shape] DTYPE
                # Our previous whitespace split broke on the shape line.
                s = ("" if raw is None else str(raw)).strip()
                lines = [ln for ln in s.splitlines() if ln.strip()]
                if not lines:
                    name_line, meta_lines = "(unknown)", []
                else:
                    name_line = lines[0].strip()
                    meta_lines = [ln.strip() for ln in lines[1:]]
                    # If we did not actually have a newline-based split, also support
                    # the legacy single-line format: "name meta...".
                    if len(lines) == 1 and " " in name_line:
                        first, rest = name_line.split(" ", 1)
                        name_line, meta_lines = first.strip(), [rest.strip()]

                name_short = _shorten_path(name_line, keep_tail=4)
                name_short = _wrap_keep_newlines(name_short, width=22)
                meta = "\n".join(meta_lines)
                meta = _wrap_keep_newlines(meta, width=34) if meta else ""

                label = name_short
                if meta:
                    label = f"{label}\n{meta}"
                if size_line:
                    label = f"{label}\n{size_line}"

                # Apply numbering and indent wrapped lines.
                prefix = f"{idx:>2}. "
                lab_lines = label.splitlines() or [""]
                lab_lines[0] = prefix + lab_lines[0]
                pad = " " * len(prefix)
                for i in range(1, len(lab_lines)):
                    lab_lines[i] = pad + lab_lines[i]
                return "\n".join(lab_lines)

            cut_nodes: List[str] = [_pretty_cut_item(t, i + 1) for i, t in enumerate(shown)]
            if omitted:
                cut_nodes.append(f"... (+{omitted} more)")
            if not cut_nodes:
                cut_nodes = ["(none)"]

            # Figure size adapts to the number of cut tensors (and the side boxes).
            n_cut = len(cut_nodes)
            fig_h = max(3.2, 0.28 * n_cut + 2.0)
            fig_w = 14.0

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.set_axis_off()
            fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.06)

            box_kw = dict(
                boxstyle="round,pad=0.45",
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
            )

            ax.text(
                0.03,
                0.5,
                left_text,
                va="center",
                ha="left",
                fontsize=9,
                family="monospace",
                bbox=box_kw,
                transform=ax.transAxes,
            )
            ax.text(
                0.50,
                0.875,
                "CUT TENSORS",
                va="center",
                ha="center",
                fontsize=10,
                family="monospace",
                transform=ax.transAxes,
            )
            ax.text(
                0.97,
                0.5,
                right_text,
                va="center",
                ha="right",
                fontsize=9,
                family="monospace",
                bbox=box_kw,
                transform=ax.transAxes,
            )

            # Lay out cut tensor nodes vertically between the two boundary boxes.
            if len(cut_nodes) == 1:
                ys = [0.5]
            else:
                top_y, bot_y = 0.73, 0.15
                step = (top_y - bot_y) / (len(cut_nodes) - 1)
                ys = [top_y - i * step for i in range(len(cut_nodes))]

            cut_box_kw = dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
            )

            # Anchor points for arrows (axes fraction coordinates).
            left_anchor = (0.24, 0.5)
            right_anchor = (0.76, 0.5)
            for y, label in zip(ys, cut_nodes):
                ax.text(
                    0.50,
                    y,
                    label,
                    va="center",
                    ha="center",
                    fontsize=8,
                    family="monospace",
                    bbox=cut_box_kw,
                    transform=ax.transAxes,
                )

                # Left -> cut
                ax.annotate(
                    "",
                    xy=(0.43, y),
                    xytext=left_anchor,
                    arrowprops=dict(arrowstyle="->", lw=0.9),
                    xycoords=ax.transAxes,
                )
                # Cut -> right
                ax.annotate(
                    "",
                    xy=right_anchor,
                    xytext=(0.57, y),
                    arrowprops=dict(arrowstyle="->", lw=0.9),
                    xycoords=ax.transAxes,
                )

            fig.suptitle(
                f"Split boundary {boundary_index}{(' | ' + semantic_label) if semantic_label else ''} | cut={_mb(total_cut):.3f} MiB | "
                f"tensors={len(cut_list)} (+{num_cut_omitted} const/init omitted)",
                fontsize=11,
                y=0.985,
            )

            # Save only the missing formats; do not overwrite graphviz output if present.
            if need_pdf:
                try:
                    fig.savefig(pdf_path, bbox_inches="tight")
                    if os.path.exists(pdf_path):
                        out["pdf"] = os.path.basename(pdf_path)
                    else:
                        LOGGER.warning("Matplotlib savefig did not create PDF: %s", pdf_path)
                except Exception:
                    LOGGER.exception("Matplotlib failed to save PDF: %s", pdf_path)
            if need_svg:
                try:
                    fig.savefig(svg_path, bbox_inches="tight")
                    if os.path.exists(svg_path):
                        out["svg"] = os.path.basename(svg_path)
                    else:
                        LOGGER.warning("Matplotlib savefig did not create SVG: %s", svg_path)
                except Exception:
                    LOGGER.exception("Matplotlib failed to save SVG: %s", svg_path)

            plt.close(fig)
        except Exception:
            LOGGER.exception("Matplotlib fallback context rendering failed")
    return out


def split_model_on_cut_tensors(
    full_model: onnx.ModelProto,
    cut_tensors: List[str],
    *,
    # Preferred names
    p1_cut_names: Optional[List[str]] = None,
    p2_cut_names: Optional[List[str]] = None,
    # Backwards-compatible aliases (older GUI versions)
    rename_cut_tensor_part1: Optional[List[str]] = None,
    rename_cut_tensor_part2: Optional[List[str]] = None,
    strict_boundary: bool = False,
) -> Tuple[onnx.ModelProto, onnx.ModelProto, Dict[str, object]]:
    """Split full_model into (part1, part2) given explicit cut tensors.

    part1: original_inputs -> cut_tensors
    part2: cut_tensors (+ any other external inputs) -> original_outputs

    Returns: (p1_model, p2_model, manifest_dict)
    """
    cut_tensors = [c for c in cut_tensors if c]
    # Accept old kwarg aliases
    if p1_cut_names is None and rename_cut_tensor_part1 is not None:
        p1_cut_names = rename_cut_tensor_part1
    if p2_cut_names is None and rename_cut_tensor_part2 is not None:
        p2_cut_names = rename_cut_tensor_part2

    if not cut_tensors:
        raise ValueError("Cut tensor list is empty; cannot split")
    if len(set(cut_tensors)) != len(cut_tensors):
        raise ValueError("Duplicate cut tensor names")

    model = infer_shapes_safe(full_model)
    g = model.graph
    init_names = {i.name for i in g.initializer}
    orig_inputs = [vi.name for vi in g.input if vi.name not in init_names]
    orig_outputs = [vi.name for vi in g.output]

    p1_model, p1_external_inputs = build_submodel(
        model,
        outputs=list(cut_tensors),
        stop_names=set(orig_inputs),
        model_name="part1",
        force_inputs=list(orig_inputs),
    )

    stop = set(cut_tensors) | set(orig_inputs)
    p2_model, p2_external_inputs = build_submodel(
        model,
        outputs=list(orig_outputs),
        stop_names=stop,
        model_name="part2",
    )

    if strict_boundary:
        # Strict boundary here means: part2 can be run with *only* the cut tensors
        # plus any original model inputs (e.g. masks, ids) that are available anyway.
        # We do NOT allow additional intermediate activations from the left part.
        cut_set = set(cut_tensors)
        orig_in_set = set(orig_inputs)
        extras = sorted([x for x in p2_external_inputs if x not in cut_set and x not in orig_in_set])
        if extras:
            raise RuntimeError(
                "Strict boundary check failed. Part2 still requires additional activations besides cut tensors/original inputs: "
                + ", ".join(extras)
            )

    # Defaults: keep names
    p1_cut_names_eff = list(cut_tensors)
    p2_cut_names_eff = list(cut_tensors)
    if p1_cut_names is not None:
        if len(p1_cut_names) != len(cut_tensors):
            raise ValueError("p1_cut_names must match number of cut tensors")
        if len(set(p1_cut_names)) != len(p1_cut_names):
            raise ValueError("Duplicate p1_cut_names")
        p1_cut_names_eff = list(p1_cut_names)
    if p2_cut_names is not None:
        if len(p2_cut_names) != len(cut_tensors):
            raise ValueError("p2_cut_names must match number of cut tensors")
        if len(set(p2_cut_names)) != len(p2_cut_names):
            raise ValueError("Duplicate p2_cut_names")
        p2_cut_names_eff = list(p2_cut_names)

    for old, new in zip(cut_tensors, p1_cut_names_eff):
        rename_value_in_model(p1_model, old, new)
    for old, new in zip(cut_tensors, p2_cut_names_eff):
        rename_value_in_model(p2_model, old, new)

    manifest = {
        "part1_cut_names": p1_cut_names_eff,
        "part2_cut_names": p2_cut_names_eff,
        "cut_tensors_full": list(cut_tensors),
        "orig_inputs": list(orig_inputs),
        "orig_outputs": list(orig_outputs),
        "part1_external_inputs": list(p1_external_inputs),
        "part2_external_inputs": list(p2_external_inputs),
    }
    return p1_model, p2_model, manifest


def save_model(model: onnx.ModelProto, path: str, *, external_data: bool = False) -> None:
    """Save model to disk (optionally as external data)."""
    if external_data:
        onnx.save_model(
            model,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(path) + ".data",
            size_threshold=1024,
        )
    else:
        onnx.save(model, path)


def model_external_data_locations(model: onnx.ModelProto) -> List[str]:
    """Return a sorted list of external-data `location` strings used by initializers.

    Many large models (e.g. LLMs) store weights in an external *.data file. In that case
    initializers have `data_location=EXTERNAL` and the actual payload is referenced via
    `TensorProto.external_data` entries.
    """
    locs: Set[str] = set()
    for init in model.graph.initializer:
        if int(getattr(init, "data_location", 0)) != int(TensorProto.EXTERNAL):
            continue
        loc: Optional[str] = None
        for kv in init.external_data:
            if kv.key == "location":
                loc = kv.value
                break
        if loc:
            locs.add(loc)
    return sorted(locs)


def _rewrite_external_data_locations_inplace(
    model: onnx.ModelProto, *, location_map: Dict[str, str]
) -> None:
    """Rewrite external-data locations in-place.

    `location_map` maps old `location` values to new ones.
    """
    if not location_map:
        return
    for init in model.graph.initializer:
        if int(getattr(init, "data_location", 0)) != int(TensorProto.EXTERNAL):
            continue
        for kv in init.external_data:
            if kv.key == "location" and kv.value in location_map:
                kv.value = location_map[kv.value]


def ensure_external_data_files(
    model: onnx.ModelProto,
    *,
    source_model_path: str,
    dest_dir: Optional[str] = None,
    # Backwards-compat alias used by older GUI code.
    dest_base: Optional[str] = None,
    mode: str = "auto",
    copy_threshold_mb: int = 256,
) -> Dict[str, str]:
    """Ensure external-data files referenced by `model` are accessible from `dest_dir`.

    Strategy:
    - Prefer hardlinks (fast, no extra disk usage).
    - Fall back to symlinks.
    - Fall back to copying only for smaller data files (< copy_threshold_mb).
    - As a last resort, rewrite the model to use absolute paths (portable only on the same machine).

    Returns a dict {location -> action} where action is one of:
      - "hardlink"
      - "symlink"
      - "copied"
      - "absolute"

    Notes
    -----
    This function does *not* load any external weights into RAM.
    """
    # dest_base was the name used in a previous iteration of the GUI.
    # Prefer dest_dir when both are provided.
    if dest_dir is None:
        dest_dir = dest_base
    if dest_dir is None:
        raise TypeError("ensure_external_data_files() missing required argument: 'dest_dir'")

    actions: Dict[str, str] = {}
    locs = model_external_data_locations(model)
    if not locs:
        return actions

    src_base = os.path.dirname(os.path.abspath(source_model_path))
    dst_base = os.path.abspath(dest_dir)

    # Helper to normalize the ONNX external-data `location` string into a platform path.
    def _loc_to_relpath(loc: str) -> str:
        loc_norm = loc.replace("\\", "/")
        return os.path.join(*[p for p in loc_norm.split("/") if p not in ("", ".")])

    # First attempt: materialize all external files into dest_dir (hardlink/symlink/copy).
    for loc in locs:
        rel = _loc_to_relpath(loc)
        src_path = loc if os.path.isabs(loc) else os.path.join(src_base, rel)
        dst_path = os.path.join(dst_base, rel)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        if os.path.exists(dst_path):
            actions[loc] = "exists"
            continue

        if not os.path.exists(src_path):
            # Try a simple fallback: if there is exactly one *.data next to the model,
            # assume that's the intended file.
            cand = [
                os.path.join(src_base, f)
                for f in os.listdir(src_base)
                if f.lower().endswith(".data")
            ]
            if len(cand) == 1 and os.path.exists(cand[0]):
                src_path = cand[0]
            else:
                raise FileNotFoundError(
                    f"External data file referenced by ONNX not found: location='{loc}' -> '{src_path}'. "
                    "Make sure the *.data file is present next to the ONNX with the expected name."
                )

        # Mode can force a specific mechanism.
        tried: List[str] = []

        def _try_hardlink() -> bool:
            tried.append("hardlink")
            try:
                os.link(src_path, dst_path)
                actions[loc] = "hardlink"
                return True
            except Exception:
                return False

        def _try_symlink() -> bool:
            tried.append("symlink")
            try:
                os.symlink(src_path, dst_path)
                actions[loc] = "symlink"
                return True
            except Exception:
                return False

        def _try_copy() -> bool:
            tried.append("copy")
            try:
                shutil.copy2(src_path, dst_path)
                actions[loc] = "copied"
                return True
            except Exception:
                return False

        if mode in ("auto", "hardlink") and _try_hardlink():
            continue
        if mode in ("auto", "symlink") and _try_symlink():
            continue
        if mode in ("copy", "auto"):
            try:
                sz_mb = int(os.path.getsize(src_path) / (1024 * 1024))
            except Exception:
                sz_mb = copy_threshold_mb + 1
            if mode == "copy" or sz_mb <= int(copy_threshold_mb):
                if _try_copy():
                    continue

        # Last resort: rewrite locations to absolute paths so the model can find the original file.
        abs_src = os.path.abspath(src_path)
        _rewrite_external_data_locations_inplace(model, location_map={loc: abs_src})
        actions[loc] = f"absolute({abs_src})"
        LOGGER.warning(
            "Could not materialize external-data file for location '%s' in '%s' (tried: %s). "
            "Falling back to absolute path reference: %s",
            loc,
            dst_base,
            ",".join(tried),
            abs_src,
        )

    return actions


# ---------------------------- Optional: splitting validation & runners ----------------------------

