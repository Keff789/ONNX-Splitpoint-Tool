"""ONNX model splitting and export utilities.

This module contains:
- submodel construction (backward slicing)
- boundary cut tensor extraction
- split export (part1/part2) and manifests
- optional GraphViz context diagrams
- optional validation + runner skeleton generation
- simple YOLO-style visualization utilities
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


def infer_shapes_safe(model: onnx.ModelProto, *, use_ort_symbolic: bool = False) -> onnx.ModelProto:
    """Try to run ONNX shape inference.

    Shape inference is useful for comm estimation + nicer I/O ValueInfo, but
    splitting should still work even if inference fails.

    When `use_ort_symbolic` is True, we also attempt ORT's symbolic shape
    inference as a second pass.
    """
    out = model
    try:
        out = shape_inference.infer_shapes(out)
    except Exception as e:
        LOGGER.debug("onnx.shape_inference failed (continuing): %s", e)

    if use_ort_symbolic:
        out2 = _infer_shapes_ort_symbolic_safe(out)
        out = out2

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

    # Helper: escape labels for DOT
    def esc(s: str) -> str:
        return str(s).replace("\\", "\\\\").replace('"', '\"')

    def short_edge_label(s: str) -> str:
        s = str(s)
        if max_edge_label and len(s) > int(max_edge_label):
            return s[: max(0, int(max_edge_label) - 3)] + "..."
        return s

    dot_path = os.path.join(out_dir, f"{basename}.dot")
    pdf_path = os.path.join(out_dir, f"{basename}.pdf")
    svg_path = os.path.join(out_dir, f"{basename}.svg")
    png_path = os.path.join(out_dir, f"{basename}.png")

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
            label += f"\n{n.name}"
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
    lines.append(
        f'  label="Split boundary {boundary_index}  |  #cut_tensors={len(cut_list)}  |  {_mode}\n({_legend})";'
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
        if meta:
            label = f"{t}\n{meta.get('shape','?')} {meta.get('dtype','?')}\n{meta.get('mib','?')} MiB"
        else:
            label = t
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
        if dot_exe:
            try:
                subprocess.run([dot_exe, "-Tpdf", dot_path, "-o", pdf_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out["pdf"] = os.path.basename(pdf_path)
            except Exception:
                pass
            try:
                subprocess.run([dot_exe, "-Tsvg", dot_path, "-o", svg_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out["svg"] = os.path.basename(svg_path)
            except Exception:
                pass
            try:
                subprocess.run([dot_exe, "-Tpng", dot_path, "-o", png_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                out["png"] = os.path.basename(png_path)
            except Exception:
                pass

    # Fallback renderer (matplotlib) if GraphViz is missing or rendering failed.
    # This yields a compact, paper-friendly diagram even without the `dot` executable.
    if render and (out.get("pdf") is None or out.get("svg") is None or out.get("png") is None):
        try:
            import matplotlib.pyplot as plt  # type: ignore
            from matplotlib.patches import FancyBboxPatch  # type: ignore

            # Try to attach shape + dtype + sizes for cut tensors (best-effort)
            sizes: Dict[str, int] = {}
            shapes: Dict[str, Optional[List[Optional[int]]]] = {}
            dtypes: Dict[str, Optional[int]] = {}
            try:
                vimap = value_info_map(infer_shapes_safe(model))
                vb = compute_tensor_bytes_per_value(vimap, batch_override=None, assume_activation_bytes=None)
                for t in cut_list:
                    sizes[t] = int(vb.get(t, 0) or 0)
                    vi = vimap.get(t)
                    shapes[t] = shape_from_vi(vi)
                    dtypes[t] = elemtype_from_vi(vi)
            except Exception:
                pass

            def _mb(x: int) -> str:
                if not x:
                    return "?"
                return f"{x / (1024 * 1024):.3f} MiB"

            def _shape_str(sh: Optional[List[Optional[int]]]) -> str:
                if not sh:
                    return "?"
                return "[" + ",".join(str(int(d)) if (d is not None and int(d) > 0) else "?" for d in sh) + "]"

            def _dtype_str(et: Optional[int]) -> str:
                if not et:
                    return "?"
                try:
                    # protobuf enum helper
                    return onnx.TensorProto.DataType.Name(int(et))
                except Exception:
                    return str(int(et))

            left_node = nodes[b_left]
            right_node = nodes[b_right]
            left_lbl = f"Left boundary\n{b_left}: {left_node.op_type}\n{left_node.name or ''}".strip()
            right_lbl = f"Right boundary\n{b_right}: {right_node.op_type}\n{right_node.name or ''}".strip()

            cut_lines = []
            for t in cut_list:
                s = sizes.get(t, 0)
                shs = _shape_str(shapes.get(t))
                dts = _dtype_str(dtypes.get(t))
                if s:
                    cut_lines.append(f"{t}\n{shs} {dts}  ({_mb(s)})")
                else:
                    cut_lines.append(f"{t}\n{shs} {dts}")
            if not cut_lines:
                cut_lines = ["(no cut tensors)"]

            total_cut = sum(int(sizes.get(t, 0) or 0) for t in cut_list)

            fig = plt.figure(figsize=(10.0, 2.6))
            ax = fig.add_subplot(111)
            ax.set_axis_off()

            # Boxes
            box_kw = dict(boxstyle="round,pad=0.4", linewidth=1.2)
            b1 = FancyBboxPatch((0.05, 0.25), 0.32, 0.5, **box_kw)
            b2 = FancyBboxPatch((0.63, 0.25), 0.32, 0.5, **box_kw)
            ax.add_patch(b1)
            ax.add_patch(b2)

            ax.text(0.21, 0.50, left_lbl, ha="center", va="center", fontsize=10)
            ax.text(0.79, 0.50, right_lbl, ha="center", va="center", fontsize=10)

            # Arrow + cut tensor list
            ax.annotate("", xy=(0.63, 0.50), xytext=(0.37, 0.50), arrowprops=dict(arrowstyle="->", lw=2.0))
            ax.text(0.50, 0.50, "\n".join(cut_lines), ha="center", va="center", fontsize=9)

            ax.text(
                0.5,
                0.92,
                f"Split boundary {boundary_index}  |  cut={_mb(total_cut)}  (#tensors={len(cut_list)})",
                ha="center",
                va="center",
                fontsize=11,
            )

            try:
                fig.savefig(pdf_path, bbox_inches="tight")
                out["pdf"] = os.path.basename(pdf_path)
            except Exception:
                pass
            try:
                fig.savefig(svg_path, bbox_inches="tight")
                out["svg"] = os.path.basename(svg_path)
            except Exception:
                pass
            try:
                fig.savefig(png_path, bbox_inches="tight", dpi=200)
                out["png"] = os.path.basename(png_path)
            except Exception:
                pass
            plt.close(fig)
        except Exception:
            pass

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

def _try_import_onnxruntime() -> bool:
    """Return True if onnxruntime can be imported."""
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def _np_dtype_from_onnx(elem_type: int) -> np.dtype:
    """Map ONNX TensorProto elem_type -> numpy dtype."""
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.BFLOAT16: np.float16,
        TensorProto.DOUBLE: np.float64,
        TensorProto.UINT8: np.uint8,
        TensorProto.INT8: np.int8,
        TensorProto.UINT16: np.uint16,
        TensorProto.INT16: np.int16,
        TensorProto.UINT32: np.uint32,
        TensorProto.INT32: np.int32,
        TensorProto.UINT64: np.uint64,
        TensorProto.INT64: np.int64,
        TensorProto.BOOL: np.bool_,
    }
    if int(elem_type) not in mapping:
        raise ValueError(f"Unsupported / unknown ONNX elem_type={elem_type}")
    return mapping[int(elem_type)]


def make_random_inputs(
    model: onnx.ModelProto,
    *,
    batch_override: Optional[int] = None,
    seed: int = 0,
    default_dim: int = 1,
) -> Dict[str, np.ndarray]:
    """Generate random input tensors for a model.

    This is intended for onnxruntime validation of split graphs.

    Rules:
    - Initializers are not treated as inputs.
    - Unknown dims become:
        * batch_override if it's the first dimension and batch_override is set
        * otherwise default_dim
    """
    rng = np.random.default_rng(int(seed))

    model = infer_shapes_safe(model)
    g = model.graph
    init_names = {i.name for i in g.initializer}

    feeds: Dict[str, np.ndarray] = {}
    for vi in g.input:
        if vi.name in init_names:
            continue
        if not vi.type.HasField("tensor_type"):
            continue

        tt = vi.type.tensor_type
        dtype = _np_dtype_from_onnx(tt.elem_type)
        shape = shape_from_vi(vi) or []
        dims: List[int] = []
        for i, d in enumerate(shape):
            if d is None or int(d) <= 0:
                if i == 0 and batch_override is not None:
                    dims.append(int(batch_override))
                else:
                    dims.append(int(default_dim))
            else:
                dims.append(int(d))
        if not dims:
            dims = [int(default_dim)]

        if np.issubdtype(dtype, np.floating):
            arr = rng.standard_normal(dims).astype(dtype)
        elif dtype == np.bool_:
            arr = (rng.random(dims) > 0.5).astype(dtype)
        else:
            # int types
            if dtype == np.int8:
                lo, hi = -5, 6
            elif dtype == np.uint8:
                lo, hi = 0, 11
            else:
                lo, hi = 0, 11
            arr = rng.integers(lo, hi, size=dims, dtype=dtype)
        feeds[vi.name] = arr
    return feeds


def validate_split_onnxruntime(
    *,
    full_model_path: str,
    part1_path: str,
    part2_path: str,
    manifest: Dict[str, object],
    batch_override: Optional[int] = None,
    seed: int = 0,
    eps: Optional[float] = 1e-4,
) -> Dict[str, object]:
    """Validate that: full(x) ~= part2(part1(x)) using onnxruntime.

    Returns a dict with:
      - ok (bool)
      - max_abs / mean_abs (aggregated)
      - per_output list

    If onnxruntime is not installed, raises RuntimeError.
    """
    if not _try_import_onnxruntime():
        raise RuntimeError("onnxruntime is not installed. Install with: pip install onnxruntime")

    import onnxruntime as ort

    # Load ONNX to generate random inputs (ORT alone has weaker dtype info)
    full_model = onnx.load(full_model_path)
    feeds_full = make_random_inputs(full_model, batch_override=batch_override, seed=seed)

    def _sess(path: str):
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    sess_full = _sess(full_model_path)
    full_out_vals = sess_full.run(None, feeds_full)
    full_out_names = [o.name for o in sess_full.get_outputs()]
    full_out = dict(zip(full_out_names, full_out_vals))

    sess_p1 = _sess(part1_path)
    p1_in_names = [i.name for i in sess_p1.get_inputs()]
    feeds_p1 = {k: feeds_full[k] for k in p1_in_names if k in feeds_full}
    missing_p1 = [k for k in p1_in_names if k not in feeds_p1]
    if missing_p1:
        raise RuntimeError(f"Validation error: missing inputs for part1: {missing_p1}")

    p1_out_vals = sess_p1.run(None, feeds_p1)
    p1_out_names = [o.name for o in sess_p1.get_outputs()]
    p1_out = dict(zip(p1_out_names, p1_out_vals))

    # Prepare part2 feeds
    sess_p2 = _sess(part2_path)
    p2_in_names = [i.name for i in sess_p2.get_inputs()]
    feeds_p2: Dict[str, np.ndarray] = {}

    p1_cut_names = list(manifest.get("part1_cut_names", []))
    p2_cut_names = list(manifest.get("part2_cut_names", []))
    if len(p1_cut_names) != len(p2_cut_names):
        raise RuntimeError("Validation error: manifest cut-name mismatch (part1 vs part2)")

    for p1n, p2n in zip(p1_cut_names, p2_cut_names):
        if p1n not in p1_out:
            raise RuntimeError(f"Validation error: part1 did not produce expected cut output '{p1n}'")
        feeds_p2[p2n] = p1_out[p1n]

    for name in p2_in_names:
        if name in feeds_p2:
            continue
        if name in feeds_full:
            feeds_p2[name] = feeds_full[name]
        else:
            raise RuntimeError(
                f"Validation error: cannot satisfy part2 input '{name}'. "
                "This boundary may not be a strict cut or requires additional external inputs."
            )

    p2_out_vals = sess_p2.run(None, feeds_p2)
    p2_out_names = [o.name for o in sess_p2.get_outputs()]
    p2_out = dict(zip(p2_out_names, p2_out_vals))

    # Compare outputs by name (preferred) and fall back to positional
    common = [n for n in full_out_names if n in p2_out]
    if not common and len(full_out_vals) == len(p2_out_vals):
        common = list(full_out_names)

    per_out = []
    max_abs_all = 0.0
    mean_abs_sum = 0.0
    n_out = 0

    for name in common:
        a = full_out.get(name)
        b = p2_out.get(name)
        if a is None or b is None:
            continue
        if tuple(a.shape) != tuple(b.shape):
            raise RuntimeError(f"Output shape mismatch for '{name}': full={a.shape} split={b.shape}")

        # Compute diffs in float64 for stability
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs = float(diff.max(initial=0.0))
        mean_abs = float(diff.mean())
        per_out.append({"name": name, "max_abs": max_abs, "mean_abs": mean_abs, "shape": list(a.shape)})
        max_abs_all = max(max_abs_all, max_abs)
        mean_abs_sum += mean_abs
        n_out += 1

    mean_abs_all = mean_abs_sum / max(n_out, 1)
    ok = True
    if eps is not None and max_abs_all > float(eps):
        ok = False

    return {
        "ok": bool(ok),
        "max_abs": float(max_abs_all),
        "mean_abs": float(mean_abs_all),
        "eps": None if eps is None else float(eps),
        "seed": int(seed),
        "batch_override": None if batch_override is None else int(batch_override),
        "per_output": per_out,
    }



# ---------------------------- Runner resources ----------------------------

_TEST_IMAGE_PNG_B64 = """\
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
AAAASUVORK5CYII=
"""

def write_runner_skeleton_onnxruntime(out_dir: str, *, manifest_filename: str = "split_manifest.json", target: str = "auto") -> str:
    """Generate a Python runner skeleton (onnxruntime) next to the exported split models.

    The runner is meant as a starting point for benchmarking / integration and performs:
      - full model run: full(x)
      - split run: part2(part1(x))
      - timing with warmup + measured runs (defaults: warmup=5, runs=10)
      - output comparison + optional report plots

    The script is self-contained and can optionally use an image as input to "see something"
    (useful for CV models like ResNet/MobileNet/YOLO). If no image is provided, random inputs are used.

    It also drops a small default test image into the export folder: test_image.png
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "run_split_onnxruntime.py")

    # Runner script (plain string + placeholder replacement to avoid f-string brace issues)
        # Runner script (plain string + placeholder replacement to avoid f-string brace issues)
    script = r'''
#!/usr/bin/env python3
# Auto-generated by onnx_splitpoint_tool
# Runner: ORT benchmark for full / part1 / part2 / composed (+ optional CV viz)
#
# Features:
# - provider selection: CPU / CUDA / TensorRT (with cache + fast-build preset)
# - inputs: random generation, optional --inputs-npz, optional image auto-feed (test_image.png)
# - output diff (max_abs/mean_abs) with --eps
# - optional report plots (validation_report.png/.pdf) if matplotlib is available
# - optional YOLO visualization (detections_*.png/.json) if PIL is available

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort

# Optional deps (runner must still work without them)
try:
    from PIL import Image, ImageDraw  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore


DEFAULT_MANIFEST = "__MANIFEST_FILENAME__"
DEFAULT_PROVIDER = "__DEFAULT_PROVIDER__"


# ----------------------------
# Small utilities
# ----------------------------
def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _available_providers() -> List[str]:
    try:
        return list(ort.get_available_providers())
    except Exception:
        return []


def _as_ort_opt(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def _pick_providers(requested: str, available: List[str]) -> List[str]:
    avail = set(available)
    req = (requested or "auto").lower().strip()

    def keep(lst: List[str]) -> List[str]:
        out = [p for p in lst if p in avail]
        if "CPUExecutionProvider" not in out:
            out.append("CPUExecutionProvider")
        return out

    if req in ("auto", "default"):
        if "TensorrtExecutionProvider" in avail:
            return keep(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        if "CUDAExecutionProvider" in avail:
            return keep(["CUDAExecutionProvider", "CPUExecutionProvider"])
        return ["CPUExecutionProvider"]

    if req in ("tensorrt", "trt"):
        return keep(["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    if req in ("cuda", "gpu"):
        return keep(["CUDAExecutionProvider", "CPUExecutionProvider"])
    if req in ("cpu",):
        return ["CPUExecutionProvider"]

    raise SystemExit(f"Unknown --provider '{requested}'. Expected: auto|tensorrt|cuda|cpu")


@dataclass
class SessionBuildInfo:
    name: str
    model_path: str
    build_seconds: float
    providers_requested: List[str]
    providers_in_use: List[str]


class _Spinner(threading.Thread):
    def __init__(self, prefix: str, every_sec: float = 1.0):
        super().__init__(daemon=True)
        self.prefix = prefix
        self.every_sec = every_sec
        self._stop = threading.Event()
        self.t0 = time.time()
        self._frames = ["|", "/", "-", "\\"]

    def stop(self):
        self._stop.set()

    def run(self):
        i = 0
        while not self._stop.is_set():
            elapsed = int(time.time() - self.t0)
            frame = self._frames[i % len(self._frames)]
            msg = f"\\r{self.prefix} {frame}  elapsed={elapsed}s"
            try:
                sys.stdout.write(msg)
                sys.stdout.flush()
            except Exception:
                pass
            i += 1
            time.sleep(self.every_sec)
        try:
            sys.stdout.write("\\r" + " " * (len(self.prefix) + 60) + "\\r")
            sys.stdout.flush()
        except Exception:
            pass


def _create_session(
    name: str,
    model_path: Path,
    providers: List[str],
    provider_options: Optional[List[dict]],
    sess_options: ort.SessionOptions,
) -> Tuple[ort.InferenceSession, SessionBuildInfo]:
    spin = _Spinner(prefix=f"[init] building '{name}' (TensorRT build can take minutes)")
    t0 = time.time()
    spin.start()
    try:
        sess = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options,
        )
    finally:
        spin.stop()
        spin.join(timeout=0.2)
    dt = time.time() - t0
    info = SessionBuildInfo(
        name=name,
        model_path=str(model_path),
        build_seconds=float(dt),
        providers_requested=list(providers),
        providers_in_use=list(sess.get_providers()),
    )
    print(f"[init] session '{name}' ready in {dt:.1f}s | providers in use: {info.providers_in_use}")
    return sess, info


# ----------------------------
# ONNX IO helpers
# ----------------------------
def _get_initializer_names(model: onnx.ModelProto) -> set:
    names = set()
    for t in model.graph.initializer:
        if t.name:
            names.add(t.name)
    for t in getattr(model.graph, "sparse_initializer", []):
        if t.name:
            names.add(t.name)
    return names


def _get_non_initializer_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    init_names = _get_initializer_names(model)
    return [vi for vi in model.graph.input if vi.name not in init_names]


def _np_dtype_from_onnx(elem_type: int) -> np.dtype:
    # Compatible across ONNX versions (onnx.mapping moved/changed).
    try:
        from onnx import helper as onnx_helper
        return np.dtype(onnx_helper.tensor_dtype_to_np_dtype(elem_type))
    except Exception:
        try:
            tp = onnx.TensorProto
            m = {
                tp.FLOAT: np.float32,
                tp.FLOAT16: np.float16,
                tp.DOUBLE: np.float64,
                tp.INT64: np.int64,
                tp.INT32: np.int32,
                tp.INT16: np.int16,
                tp.INT8: np.int8,
                tp.UINT64: np.uint64,
                tp.UINT32: np.uint32,
                tp.UINT16: np.uint16,
                tp.UINT8: np.uint8,
                tp.BOOL: np.bool_,
            }
            return np.dtype(m.get(int(elem_type), np.float32))
        except Exception:
            return np.float32


def _shape_from_value_info(vi: onnx.ValueInfoProto) -> List[Optional[int]]:
    shape = []
    try:
        dims = vi.type.tensor_type.shape.dim
        for d in dims:
            if d.HasField("dim_value"):
                shape.append(int(d.dim_value))
            else:
                shape.append(None)
    except Exception:
        return []
    return shape


def _parse_shape_override(s: Optional[str]) -> Dict[str, Tuple[int, ...]]:
    out: Dict[str, Tuple[int, ...]] = {}
    if not s:
        return out
    # "name=1x128 other=1x3x640x640"
    for chunk in s.replace(",", " ").split():
        if "=" not in chunk:
            continue
        name, val = chunk.split("=", 1)
        name = name.strip()
        val = val.strip().lower().replace("x", " ")
        dims = [int(x) for x in val.split() if x.strip().isdigit()]
        if name and dims:
            out[name] = tuple(dims)
    return out


def _make_random_inputs(
    model: onnx.ModelProto,
    batch: Optional[int],
    seed: int,
    shape_overrides: Dict[str, Tuple[int, ...]],
    only_names: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    out: Dict[str, np.ndarray] = {}
    for vi in _get_non_initializer_inputs(model):
        name = vi.name
        if only_names is not None and name not in only_names:
            continue
        dtype = _np_dtype_from_onnx(vi.type.tensor_type.elem_type)
        shp = _shape_from_value_info(vi)
        if name in shape_overrides:
            shp = list(shape_overrides[name])
        # Replace unknown dims with defaults
        fixed = []
        for i, d in enumerate(shp):
            if d is None or d == 0:
                if i == 0:
                    fixed.append(int(batch) if batch else 1)
                else:
                    fixed.append(1)
            else:
                fixed.append(int(d))
        if not fixed:
            fixed = [int(batch) if batch else 1]
        if dtype == np.bool_:
            out[name] = (rng.random(fixed) > 0.5)
        elif np.issubdtype(dtype, np.integer):
            out[name] = rng.integers(low=0, high=2, size=fixed, dtype=dtype)
        else:
            out[name] = rng.standard_normal(size=fixed).astype(dtype)
    return out


def _save_npz(path: str, arrays: Dict[str, np.ndarray], meta: dict) -> None:
    payload = {k: v for k, v in arrays.items()}
    payload["__meta__"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def _load_inputs_npz(path: str) -> Tuple[Dict[str, np.ndarray], Optional[dict]]:
    d = np.load(path, allow_pickle=False)
    out: Dict[str, np.ndarray] = {}
    meta = None
    for k in d.files:
        if k == "__meta__":
            try:
                meta = json.loads(bytes(d[k].tolist()).decode("utf-8"))
            except Exception:
                meta = None
            continue
        out[k] = d[k]
    return out, meta


# ----------------------------
# Image input helper (optional)
# ----------------------------
def _is_probably_image_input(shape: List[Optional[int]]) -> bool:
    # Expect NCHW: [N,3,H,W]
    if len(shape) != 4:
        return False
    c = shape[1]
    return (c == 3) or (c is None)


def _load_image_as_nchw(
    img_path: Path,
    *,
    target_hw: Tuple[int, int],
    dtype: np.dtype,
    scale: str,
) -> Optional[np.ndarray]:
    if Image is None:
        print("[warn] PIL not available -> cannot load image input; using random inputs.")
        return None
    try:
        img = Image.open(str(img_path)).convert("RGB")
    except Exception as e:
        print(f"[warn] failed to load image '{img_path}': {type(e).__name__}: {e}")
        return None

    w, h = target_hw[1], target_hw[0]
    try:
        img_rs = img.resize((w, h))
    except Exception:
        img_rs = img

    arr = np.array(img_rs, dtype=np.float32)  # HWC, 0..255
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, axis=0)  # NCHW

    if scale == "raw":
        # 0..255
        pass
    elif scale == "norm":
        # 0..1
        arr = arr / 255.0
    elif scale == "imagenet":
        # 0..1 -> (x-mean)/std
        arr = arr / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        arr = (arr - mean) / std
    elif scale == "clip":
        # 0..1 -> (x-mean)/std (CLIP)
        arr = arr / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 3, 1, 1)
        arr = (arr - mean) / std
    else:
        # caller handles "auto"
        pass

    if np.issubdtype(dtype, np.floating):
        return arr.astype(dtype, copy=False)
    if np.issubdtype(dtype, np.integer):
        return np.clip(arr, 0, np.iinfo(dtype).max).astype(dtype)
    return arr.astype(dtype, copy=False)


# ----------------------------
# Interface dump
# ----------------------------
def _dump_interface_npz(
    mode: str,
    out_prefix: Optional[str],
    out_dir: Path,
    feeds_left: Dict[str, np.ndarray],
    feeds_right: Dict[str, np.ndarray],
    meta_base: dict,
) -> None:
    def nbytes(d: Dict[str, np.ndarray]) -> int:
        return int(sum(int(v.nbytes) for v in d.values()))

    left_b = nbytes(feeds_left)
    right_b = nbytes(feeds_right)

    def write_one(path: Path, arrays: Dict[str, np.ndarray], extra: dict):
        meta = dict(meta_base)
        meta.update(extra)
        _save_npz(str(path), arrays, meta)

    if out_prefix:
        p = Path(out_prefix)
        if not p.is_absolute():
            p = out_dir / p
        base = p
    else:
        base = out_dir / "interface"

    if mode == "either":
        p_left = base.with_name(base.name + "_left.npz")
        p_right = base.with_name(base.name + "_right.npz")
        write_one(p_left, feeds_left, {"which": "left", "total_nbytes": left_b})
        write_one(p_right, feeds_right, {"which": "right", "total_nbytes": right_b})
        print(f"[dump-interface] wrote {p_left} ({left_b/1024/1024:.3f} MiB)")
        print(f"[dump-interface] wrote {p_right} ({right_b/1024/1024:.3f} MiB)")
        return

    if mode == "min":
        mode = "left" if left_b <= right_b else "right"

    if mode == "left":
        p = base.with_suffix(".npz")
        write_one(p, feeds_left, {"which": "left", "total_nbytes": left_b})
        print(f"[dump-interface] wrote {p} ({left_b/1024/1024:.3f} MiB)")
        return

    if mode == "right":
        p = base.with_suffix(".npz")
        write_one(p, feeds_right, {"which": "right", "total_nbytes": right_b})
        print(f"[dump-interface] wrote {p} ({right_b/1024/1024:.3f} MiB)")
        return

    raise SystemExit(f"Invalid --dump-interface mode '{mode}'")


# ----------------------------
# Benchmark + report
# ----------------------------
def _bench(tag: str, fn, warmup: int, runs: int) -> Tuple[float, float, List[float]]:
    print(f"[{tag}] warmup: {warmup} runs (not measured)")
    for i in range(warmup):
        fn()
    print(f"[{tag}] measured: {runs} runs")
    times = []
    for i in range(runs):
        t0 = time.time()
        fn()
        dt = (time.time() - t0) * 1000.0
        times.append(dt)
        print(f"  run {i+1}/{runs}: {dt:.3f} ms")
    return float(np.mean(times)), float(np.std(times)), times


def _try_write_report_plots(out_dir: Path, report: dict) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[viz] matplotlib not available -> skipping validation_report.png/.pdf")
        return

    t = report.get("timing_ms", {})
    labels = ["full", "part1", "part2", "composed"]
    means = [t.get("full_mean", 0.0), t.get("part1_mean", 0.0), t.get("part2_mean", 0.0), t.get("composed_mean", 0.0)]
    stds = [t.get("full_std", 0.0), t.get("part1_std", 0.0), t.get("part2_std", 0.0), t.get("composed_std", 0.0)]

    try:
        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 1, 1)
        xs = np.arange(len(labels))
        ax.bar(xs, means, yerr=stds, capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Latency (ms)")
        passed = report.get("output_diff", {}).get("passed", True)
        ax.set_title(f"Split benchmark (PASS={passed})")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
        fig.tight_layout()

        p_png = out_dir / "validation_report.png"
        p_pdf = out_dir / "validation_report.pdf"
        fig.savefig(str(p_png), dpi=150)
        fig.savefig(str(p_pdf))
        plt.close(fig)
        print(f"Wrote {p_png} and .pdf")
        report.setdefault("artifacts", {})
        report["artifacts"]["validation_report_png"] = str(p_png.name)
        report["artifacts"]["validation_report_pdf"] = str(p_pdf.name)
    except Exception as e:
        print(f"[viz] failed to write report plots: {type(e).__name__}: {e}")


# ----------------------------
# YOLO decode + draw (optional)
# ----------------------------
_COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _is_yolo_multiscale(outputs: List[np.ndarray]) -> bool:
    if len(outputs) != 3:
        return False
    for o in outputs:
        if not isinstance(o, np.ndarray):
            return False
        if o.ndim != 5:
            return False
        if o.shape[1] != 3:
            return False
        if o.shape[-1] < 6:
            return False
    return True


# Default YOLOv5/YOLOv7 anchors for 640 input (works well for many exported yolov7 models)
_YOLO_ANCHORS = [
    [(12, 16), (19, 36), (40, 28)],
    [(36, 75), (76, 55), (72, 146)],
    [(142, 110), (192, 243), (459, 401)],
]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float, max_det: int) -> List[int]:
    # boxes: Nx4 (x1,y1,x2,y2)
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def _decode_yolo(
    outs: List[np.ndarray],
    img_hw: Tuple[int, int],
    conf_thr: float,
    iou_thr: float,
    max_det: int,
) -> List[dict]:
    # outs: list of 3 arrays, each (B,3,H,W,5+nc)
    H_img, W_img = img_hw
    # Sort by grid size desc to match anchor groups
    outs_sorted = sorted(outs, key=lambda a: int(a.shape[2]) * int(a.shape[3]), reverse=True)

    all_boxes = []
    all_scores = []
    all_cls = []

    for si, p in enumerate(outs_sorted):
        p = p.astype(np.float32, copy=False)
        b, na, gh, gw, ch = p.shape
        nc = ch - 5
        if nc <= 0:
            continue

        # stride inferred from input size / grid size
        stride_w = W_img / float(gw)
        stride_h = H_img / float(gh)
        stride = float((stride_w + stride_h) * 0.5)

        anchors = np.array(_YOLO_ANCHORS[min(si, len(_YOLO_ANCHORS)-1)], dtype=np.float32).reshape(1, na, 1, 1, 2)

        yv, xv = np.meshgrid(np.arange(gh, dtype=np.float32), np.arange(gw, dtype=np.float32), indexing="ij")
        grid = np.stack((xv, yv), axis=-1).reshape(1, 1, gh, gw, 2)

        xy = _sigmoid(p[..., 0:2]) * 2.0 - 0.5
        wh = (_sigmoid(p[..., 2:4]) * 2.0) ** 2
        obj = _sigmoid(p[..., 4:5])
        cls_scores = _sigmoid(p[..., 5:])

        cls_id = np.argmax(cls_scores, axis=-1).astype(np.int32)
        cls_max = np.max(cls_scores, axis=-1, keepdims=True)
        conf = (obj * cls_max).squeeze(-1)

        # Filter
        mask = conf > conf_thr
        if not np.any(mask):
            continue

        xy = (xy + grid) * stride
        wh = wh * anchors

        # boxes in xywh -> xyxy
        x = xy[..., 0]
        y = xy[..., 1]
        w = wh[..., 0]
        h = wh[..., 1]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0

        sel_x1 = x1[mask]
        sel_y1 = y1[mask]
        sel_x2 = x2[mask]
        sel_y2 = y2[mask]
        sel_conf = conf[mask]
        sel_cls = cls_id[mask]

        boxes = np.stack([sel_x1, sel_y1, sel_x2, sel_y2], axis=1)
        all_boxes.append(boxes)
        all_scores.append(sel_conf)
        all_cls.append(sel_cls)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_ids = np.concatenate(all_cls, axis=0)

    # NMS (class-agnostic for simplicity)
    keep = _nms(boxes, scores, iou_thr=iou_thr, max_det=max_det)
    dets = []
    for i in keep:
        c = int(cls_ids[i])
        dets.append(
            {
                "x1": float(boxes[i, 0]),
                "y1": float(boxes[i, 1]),
                "x2": float(boxes[i, 2]),
                "y2": float(boxes[i, 3]),
                "score": float(scores[i]),
                "class_id": c,
                "class_name": _COCO80[c] if 0 <= c < len(_COCO80) else str(c),
            }
        )
    return dets


def _yolo_plausibility(dets: List[dict]) -> float:
    # Prefer moderate number of detections and reasonable confidence
    if not dets:
        return -1e9
    n = len(dets)
    avg = float(np.mean([d["score"] for d in dets])) if n else 0.0
    penalty = 0.0
    if n > 200:
        penalty += (n - 200) * 5.0
    if n < 1:
        penalty += 100.0
    # Center around ~10 detections
    penalty += abs(n - 10) * 1.0
    return avg * 100.0 - penalty



def _detr_plausibility(dets: List[dict]) -> float:
    """Heuristic for choosing the right image normalization for DETR/YOLOS-like outputs."""
    if not dets:
        return -1e9
    scores = [float(d.get("score", 0.0)) for d in dets]
    scores.sort(reverse=True)
    topk = scores[:10]
    top_mean = float(np.mean(topk)) if topk else 0.0
    # Penalize too many detections (usually indicates wrong preprocessing)
    return top_mean - 0.01 * float(len(dets))


def _draw_dets(img_path: Path, dets: List[dict], out_path: Path, *, img_hw: Tuple[int, int]) -> None:
    if Image is None or ImageDraw is None:
        return
    try:
        img = Image.open(str(img_path)).convert("RGB")
        img = img.resize((img_hw[1], img_hw[0]))
        draw = ImageDraw.Draw(img)
        for d in dets:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            txt = f'{d["class_name"]} {d["score"]:.2f}'
            draw.text((x1 + 2, y1 + 2), txt, fill=(255, 0, 0))
        img.save(str(out_path))
    except Exception as e:
        print(f"[viz] failed to draw detections: {type(e).__name__}: {e}")


def _maybe_yolo_viz(
    out_dir: Path,
    img_path: Optional[Path],
    img_hw: Optional[Tuple[int, int]],
    full_out: List[np.ndarray],
    comp_out: List[np.ndarray],
    conf_thr: float,
    iou_thr: float,
    max_det: int,
) -> dict:
    info = {"enabled": False}
    if img_path is None or img_hw is None:
        print("[viz] no image input -> skipping YOLO visualization")
        return info
    if not _is_yolo_multiscale(full_out) or not _is_yolo_multiscale(comp_out):
        return info

    print("[viz] YOLO output format: multiscale head (B,3,H,W,...)")
    dets_full = _decode_yolo(full_out, img_hw=img_hw, conf_thr=conf_thr, iou_thr=iou_thr, max_det=max_det)
    dets_comp = _decode_yolo(comp_out, img_hw=img_hw, conf_thr=conf_thr, iou_thr=iou_thr, max_det=max_det)

    p_json_full = out_dir / "detections_full.json"
    p_json_comp = out_dir / "detections_composed.json"
    _write_json(p_json_full, {"image": str(img_path.name), "detections": dets_full})
    _write_json(p_json_comp, {"image": str(img_path.name), "detections": dets_comp})

    p_img_full = out_dir / "detections_full.png"
    p_img_comp = out_dir / "detections_composed.png"
    _draw_dets(img_path, dets_full, p_img_full, img_hw=img_hw)
    _draw_dets(img_path, dets_comp, p_img_comp, img_hw=img_hw)

    if p_img_full.exists():
        print(f"Wrote {p_img_full}")
    if p_img_comp.exists():
        print(f"Wrote {p_img_comp}")
    print(f"Wrote {p_json_full.name} and {p_json_comp.name}")

    info = {
        "enabled": True,
        "image": str(img_path.name),
        "img_hw": list(img_hw),
        "full": {"count": len(dets_full), "json": p_json_full.name, "png": p_img_full.name},
        "composed": {"count": len(dets_comp), "json": p_json_comp.name, "png": p_img_comp.name},
        "params": {"conf_thr": conf_thr, "iou_thr": iou_thr, "max_det": max_det},
    }
    return info




# ----------------------------
# DETR/YOLOS-style postprocess (logits + pred_boxes)
# ----------------------------
def _is_detr_like(output_names: List[str], outputs: List[np.ndarray]) -> bool:
    # Heuristic: outputs contain logits (..,C) and boxes (..,4)
    if len(outputs) < 2:
        return False
    names = [str(n) for n in output_names]
    # Common HF/DETR names
    if ("logits" in names and "pred_boxes" in names):
        return True
    # Shape-based fallback
    shapes = [getattr(a, "shape", ()) for a in outputs]
    has_boxes = any(len(s) >= 2 and s[-1] == 4 for s in shapes)
    has_logits = any(len(s) >= 2 and s[-1] >= 10 for s in shapes)  # num classes usually >= 10
    return bool(has_boxes and has_logits)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / (s + 1e-12)


def _load_labels(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
        lines = [ln for ln in lines if ln]
        return lines if lines else None
    except Exception:
        return None


def _decode_detr(
    output_names: List[str],
    outputs: List[np.ndarray],
    img_hw: Tuple[int, int],
    conf_thr: float,
    iou_thr: float,
    max_det: int,
    labels: Optional[List[str]] = None,
) -> List[dict]:
    """
    Decode end-to-end detector outputs.

    Supported layouts:
      A) logits (B,Q,C)/(Q,C) + boxes (B,Q,4)/(Q,4)
         - If C == (#labels + 1) -> DETR-style softmax with background class (NMS optional)
         - Else -> sigmoid / already-probabilities without background (RT-DETR / YOLOv10/YOLO26 end2end)
                This path is NMS-free by design (we keep top-k only).

      B) dets (B,Q,6)/(Q,6) where last dim is [x1,y1,x2,y2,score,cls]
         (common Ultralytics end-to-end ONNX export for YOLOv10/YOLO26).

    Boxes may be cxcywh or xyxy, normalized or absolute.
    Returned detections are in xyxy pixel coordinates (relative to the input image).
    """
    H, W = img_hw

    name2 = {n: outputs[i] for i, n in enumerate(output_names)}

    # --- Layout B: (Q,6) / (B,Q,6) ---
    det6 = None
    for n, a in name2.items():
        if isinstance(a, np.ndarray) and a.ndim in (2, 3) and a.shape[-1] == 6:
            det6 = a
            break
    if det6 is not None:
        a = det6[0] if det6.ndim == 3 else det6  # (Q,6)
        if a.size == 0:
            return []
        a = a.astype(np.float32, copy=False)
        boxes = a[:, 0:4]
        scores = a[:, 4].astype(np.float32, copy=False)
        cls = a[:, 5].astype(np.int64, copy=False)

        # filter + top-k
        keep = np.where(scores >= float(conf_thr))[0]
        if keep.size == 0:
            return []
        scores = scores[keep]
        cls = cls[keep]
        boxes = boxes[keep]

        if scores.size > max_det:
            order = np.argsort(scores)[::-1][:max_det]
            scores = scores[order]
            cls = cls[order]
            boxes = boxes[order]

        b_max = float(np.nanmax(boxes)) if boxes.size else 0.0
        norm = b_max <= 1.5

        # Assume xyxy for this layout.
        if norm:
            x1 = boxes[:, 0] * W
            y1 = boxes[:, 1] * H
            x2 = boxes[:, 2] * W
            y2 = boxes[:, 3] * H
        else:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        x1 = np.clip(x1, 0.0, W - 1.0)
        y1 = np.clip(y1, 0.0, H - 1.0)
        x2 = np.clip(x2, 0.0, W - 1.0)
        y2 = np.clip(y2, 0.0, H - 1.0)
        # ensure ordering
        x1_, x2_ = np.minimum(x1, x2), np.maximum(x1, x2)
        y1_, y2_ = np.minimum(y1, y2), np.maximum(y1, y2)
        x1, x2, y1, y2 = x1_, x2_, y1_, y2_

        dets = []
        for i in range(scores.shape[0]):
            dets.append({
                "x1": float(x1[i]),
                "y1": float(y1[i]),
                "x2": float(x2[i]),
                "y2": float(y2[i]),
                "score": float(scores[i]),
                "class_id": int(cls[i]),
                "class_name": labels[int(cls[i])] if labels and 0 <= int(cls[i]) < len(labels) else f"cls{int(cls[i])}",
            })
        return dets

    # --- Layout A: logits + boxes ---
    logits = None
    boxes = None
    for n, a in name2.items():
        if not isinstance(a, np.ndarray):
            continue
        if a.ndim in (2, 3) and a.shape[-1] == 4:
            boxes = a
        elif a.ndim in (2, 3) and a.shape[-1] >= 10:
            logits = a

    if logits is None or boxes is None:
        return []

    l = logits[0] if logits.ndim == 3 else logits  # (Q,C)
    b = boxes[0] if boxes.ndim == 3 else boxes      # (Q,4)

    if l.ndim != 2 or b.ndim != 2 or b.shape[1] != 4:
        return []

    Q, C = l.shape

    # Decide whether there is an explicit background class (DETR style).
    has_background = False
    if labels is not None and C == (len(labels) + 1):
        has_background = True
    # Common DETR exports (COCO) sometimes have 91+1 or 80+1 classes.
    if labels is None and C in (81, 92):
        has_background = True

    # Compute class probabilities.
    if has_background:
        probs = _softmax(l.astype(np.float32, copy=False), axis=-1)
        probs = probs[:, :-1]  # drop background
    else:
        lf = l.astype(np.float32, copy=False)
        lmin = float(np.nanmin(lf)) if lf.size else 0.0
        lmax = float(np.nanmax(lf)) if lf.size else 0.0
        if 0.0 <= lmin and lmax <= 1.0:
            probs = lf  # already probabilities
        else:
            probs = _sigmoid(lf)

    # Best class per query (NMS-free models rely on low absolute scores for "no object" queries).
    cls = np.argmax(probs, axis=1).astype(np.int64)
    scores = probs[np.arange(Q), cls]

    keep = np.where(scores >= float(conf_thr))[0]
    if keep.size == 0:
        return []

    scores = scores[keep]
    cls = cls[keep]
    b = b[keep].astype(np.float32, copy=False)

    # Keep top-k by score (always).
    if scores.size > max_det:
        order = np.argsort(scores)[::-1][:max_det]
        scores = scores[order]
        cls = cls[order]
        b = b[order]

    # Determine whether boxes are normalized and whether they are xyxy or cxcywh.
    b_max = float(np.nanmax(b)) if b.size else 0.0
    norm = b_max <= 1.5

    # If columns look like x2>=x1 and y2>=y1 for almost all boxes, treat as xyxy.
    frac_xyxy = float(np.mean((b[:, 2] >= b[:, 0]) & (b[:, 3] >= b[:, 1]))) if b.size else 0.0
    box_mode = "xyxy" if frac_xyxy > 0.95 else "cxcywh"

    if norm:
        if box_mode == "xyxy":
            x1 = b[:, 0] * W
            y1 = b[:, 1] * H
            x2 = b[:, 2] * W
            y2 = b[:, 3] * H
        else:
            cx = b[:, 0] * W
            cy = b[:, 1] * H
            bw = b[:, 2] * W
            bh = b[:, 3] * H
            x1 = cx - 0.5 * bw
            y1 = cy - 0.5 * bh
            x2 = cx + 0.5 * bw
            y2 = cy + 0.5 * bh
    else:
        if box_mode == "xyxy":
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        else:
            cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            x1 = cx - 0.5 * bw
            y1 = cy - 0.5 * bh
            x2 = cx + 0.5 * bw
            y2 = cy + 0.5 * bh

    x1 = np.clip(x1, 0.0, W - 1.0)
    y1 = np.clip(y1, 0.0, H - 1.0)
    x2 = np.clip(x2, 0.0, W - 1.0)
    y2 = np.clip(y2, 0.0, H - 1.0)
    # ensure ordering
    x1_, x2_ = np.minimum(x1, x2), np.maximum(x1, x2)
    y1_, y2_ = np.minimum(y1, y2), np.maximum(y1, y2)
    x1, x2, y1, y2 = x1_, x2_, y1_, y2_

    # Apply NMS only for explicit-background DETR style.
    if has_background and scores.size > 0 and float(iou_thr) < 0.999:
        keep_idx = _nms(np.stack([x1, y1, x2, y2], axis=1), scores, iou_thr=float(iou_thr), max_det=max_det)
        x1 = x1[keep_idx]
        y1 = y1[keep_idx]
        x2 = x2[keep_idx]
        y2 = y2[keep_idx]
        scores = scores[keep_idx]
        cls = cls[keep_idx]

    dets = []
    for i in range(scores.shape[0]):
        dets.append({
            "x1": float(x1[i]),
            "y1": float(y1[i]),
            "x2": float(x2[i]),
            "y2": float(y2[i]),
            "score": float(scores[i]),
            "class_id": int(cls[i]),
            "class_name": labels[int(cls[i])] if labels and 0 <= int(cls[i]) < len(labels) else f"cls{int(cls[i])}",
        })
    return dets

def _maybe_detr_viz(
    out_dir: Path,
    img_path: Optional[Path],
    img_hw: Optional[Tuple[int, int]],
    output_names: List[str],
    full_out: List[np.ndarray],
    comp_out: List[np.ndarray],
    conf_thr: float,
    iou_thr: float,
    max_det: int,
    labels: Optional[List[str]],
) -> dict:
    info = {"enabled": False}
    if img_path is None or img_hw is None:
        print("[viz] no image input -> skipping DETR visualization")
        return info
    if not _is_detr_like(output_names, full_out) or not _is_detr_like(output_names, comp_out):
        return info

    print("[viz] DETR/YOLOS output format: logits + pred_boxes")
    dets_full = _decode_detr(output_names, full_out, img_hw=img_hw, conf_thr=conf_thr, iou_thr=iou_thr, max_det=max_det, labels=labels)
    dets_comp = _decode_detr(output_names, comp_out, img_hw=img_hw, conf_thr=conf_thr, iou_thr=iou_thr, max_det=max_det, labels=labels)

    p_json_full = out_dir / "detections_full.json"
    p_json_comp = out_dir / "detections_composed.json"
    _write_json(p_json_full, {"image": str(img_path.name), "detections": dets_full})
    _write_json(p_json_comp, {"image": str(img_path.name), "detections": dets_comp})

    p_img_full = out_dir / "detections_full.png"
    p_img_comp = out_dir / "detections_composed.png"
    _draw_dets(img_path, dets_full, p_img_full, img_hw=img_hw)
    _draw_dets(img_path, dets_comp, p_img_comp, img_hw=img_hw)

    if p_img_full.exists():
        print(f"Wrote {p_img_full}")
    if p_img_comp.exists():
        print(f"Wrote {p_img_comp}")
    print(f"Wrote {p_json_full.name} and {p_json_comp.name}")

    info = {
        "enabled": True,
        "image": str(img_path.name),
        "img_hw": list(img_hw),
        "full": {"count": len(dets_full), "json": p_json_full.name, "png": p_img_full.name},
        "composed": {"count": len(dets_comp), "json": p_json_comp.name, "png": p_img_comp.name},
        "params": {"conf_thr": conf_thr, "iou_thr": iou_thr, "max_det": max_det},
    }
    return info


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Run ORT benchmark for a split ONNX model set.")
    ap.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Split manifest JSON (default from export).")
    ap.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=["auto", "tensorrt", "cuda", "cpu"], help="Execution provider preference.")
    ap.add_argument("--out-dir", type=str, default="results", help="Output dir for reports (default: results/)")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup runs per benchmark")
    ap.add_argument("--runs", type=int, default=10, help="Measured runs per benchmark")
    ap.add_argument("--eps", type=float, default=1e-4, help="Max-abs threshold for output diff PASS/FAIL")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for random input generation")
    ap.add_argument("--batch", type=int, default=None, help="Batch dim for random input generation (if dim0 unknown)")
    ap.add_argument("--shape-override", type=str, default=None, help="Override input shapes for random generation, shorthand: name=1x128;other=1x3x640x640")
    ap.add_argument("--inputs-npz", type=str, default=None, help="Load model inputs from an .npz file (keys are input names). Missing inputs are generated.")
    ap.add_argument("--save-inputs-npz", type=str, default=None, help="Save the final full-model input feed dict to an .npz file (includes __meta__).")

    # Image helper (for CV models)
    ap.add_argument("--image", type=str, default="test_image.png", help="Optional image file for CV inputs (default: test_image.png)")
    ap.add_argument("--image-scale", type=str, default="auto", choices=["auto", "norm", "raw", "imagenet", "clip"], help="Image preprocessing for float inputs: raw=0..255, norm=img/255, imagenet=(img/255-mean)/std, clip=CLIP norm, auto=probe (YOLO/DETR)")

    # Visualization
    ap.add_argument("--viz", type=str, default="auto", choices=["auto", "none", "yolo", "detr"], help="Visualization: auto (YOLO/DETR if detected), yolo, detr, none")
    ap.add_argument("--no-report-plots", action="store_true", help="Disable validation_report.png/.pdf generation")
    ap.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--yolo-iou", type=float, default=0.45, help="YOLO NMS IoU threshold")
    ap.add_argument("--yolo-max-det", type=int, default=200, help="YOLO max detections after NMS")
    ap.add_argument("--detr-conf", type=float, default=0.25, help="DETR/YOLOS confidence threshold")
    ap.add_argument("--detr-iou", type=float, default=0.50, help="DETR/YOLOS NMS IoU threshold")
    ap.add_argument("--detr-max-det", type=int, default=200, help="DETR/YOLOS max detections after NMS")
    ap.add_argument("--labels", type=str, default=None, help="Optional label file (one class name per line) for visualization")

    # Interface dump
    ap.add_argument("--dump-interface", type=str, default=None, choices=["right", "left", "min", "either"], help="Dump interface NPZ(s): right=part2 feed, left=part1 feed, either=both, min=smaller.")
    ap.add_argument("--dump-interface-out", type=str, default=None, help="Output path/prefix for interface NPZ(s). Default: <out-dir>/interface_*.npz")

    # Session options
    ap.add_argument("--build-only", action="store_true", help="Only build sessions/engines, run 1 inference each, exit.")
    ap.add_argument("--ort-log-severity", type=int, default=2, help="ORT log severity (0=verbose,1=info,2=warning,3=error,4=fatal).")

    # TRT options
    ap.add_argument("--trt-cache-dir", type=str, default="trt_cache", help="TensorRT engine cache directory")
    ap.add_argument("--trt-cache", dest="trt_cache", action="store_true", default=True, help="Enable engine cache")
    ap.add_argument("--no-trt-cache", dest="trt_cache", action="store_false", help="Disable engine cache")
    ap.add_argument("--trt-fp16", action="store_true", help="Enable TRT FP16")
    ap.add_argument("--trt-dump-subgraphs", action="store_true", help="Enable TRT subgraph dump (ORT TensorRT EP)")
    ap.add_argument("--trt-fast-build", action="store_true", help="Faster TRT build preset (opt level 2)")

    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(manifest_path)

    # Manifest schema compatibility
    full_rel = manifest.get("full_model") or manifest.get("full") or manifest.get("model")
    p1_rel = manifest.get("part1_model") or manifest.get("part1") or manifest.get("part1_path")
    p2_rel = manifest.get("part2_model") or manifest.get("part2") or manifest.get("part2_path")
    if not full_rel or not p1_rel or not p2_rel:
        raise SystemExit(
            "Manifest schema not recognized. Expected keys like "
            "'full_model' and ('part1'/'part1_model') and ('part2'/'part2_model'). "
            f"Found keys: {sorted(manifest.keys())}"
        )

    full_path = Path(full_rel)
    p1_path = Path(p1_rel)
    p2_path = Path(p2_rel)
    if not full_path.is_absolute():
        full_path = base_dir / full_path
    if not p1_path.is_absolute():
        p1_path = base_dir / p1_path
    if not p2_path.is_absolute():
        p2_path = base_dir / p2_path

    print(f"[info] manifest: {manifest_path}")
    print(f"[info] full : {full_path.name}")
    print(f"[info] part1: {p1_path.name}")
    print(f"[info] part2: {p2_path.name}")

    # Load only the ONNX skeleton (avoid pulling huge external weights into RAM)
    full_model = onnx.load(str(full_path), load_external_data=False)

    shape_overrides = _parse_shape_override(args.shape_override)

    feeds_full: Dict[str, np.ndarray] = {}
    loaded_inputs_meta = None
    if args.inputs_npz:
        feeds_full, loaded_inputs_meta = _load_inputs_npz(args.inputs_npz)
        print(f"[inputs] loaded {len(feeds_full)} tensor(s) from {args.inputs_npz}")

    required_inputs = [vi.name for vi in _get_non_initializer_inputs(full_model)]
    missing = [n for n in required_inputs if n not in feeds_full]

    # Optional: use image for missing CV input(s)
    img_path = None
    img_hw = None
    img_candidates = None  # type: ignore
    if missing and not args.inputs_npz and args.image:
        p_img = Path(args.image)
        if not p_img.is_absolute():
            p_img = base_dir / p_img
        if p_img.exists():
            # Find first image-like input among missing
            for vi in _get_non_initializer_inputs(full_model):
                if vi.name not in missing:
                    continue
                shp = _shape_from_value_info(vi)
                if vi.name in shape_overrides:
                    shp = list(shape_overrides[vi.name])
                if not _is_probably_image_input(shp):
                    continue
                dtype = _np_dtype_from_onnx(vi.type.tensor_type.elem_type)
                # Determine H,W
                H = int(shp[2]) if shp[2] else 640
                W = int(shp[3]) if shp[3] else 640
                img_hw = (H, W)
                img_path = p_img
                if args.image_scale == "auto":
                    img_candidates = {
                        "raw": _load_image_as_nchw(p_img, target_hw=img_hw, dtype=dtype, scale="raw"),
                        "norm": _load_image_as_nchw(p_img, target_hw=img_hw, dtype=dtype, scale="norm"),
                        "imagenet": _load_image_as_nchw(p_img, target_hw=img_hw, dtype=dtype, scale="imagenet"),
                        "clip": _load_image_as_nchw(p_img, target_hw=img_hw, dtype=dtype, scale="clip"),
                        "name": vi.name,
                    }
                    # default to norm
                    if img_candidates.get("norm") is not None:
                        feeds_full[vi.name] = img_candidates["norm"]
                        missing.remove(vi.name)
                else:
                    arr = _load_image_as_nchw(p_img, target_hw=img_hw, dtype=dtype, scale=args.image_scale)
                    if arr is not None:
                        feeds_full[vi.name] = arr
                        missing.remove(vi.name)
                if vi.name in feeds_full:
                    print(f"[inputs] using image '{p_img.name}' for input '{vi.name}' shape={list(feeds_full[vi.name].shape)} scale={args.image_scale}")
                    break

    if missing:
        feeds_full.update(
            _make_random_inputs(
                full_model,
                batch=args.batch,
                seed=args.seed,
                shape_overrides=shape_overrides,
                only_names=missing,
            )
        )
        if args.inputs_npz:
            print(f"[inputs] generated {len(missing)} missing input(s): {missing[:8]}{'...' if len(missing)>8 else ''}")

    if args.save_inputs_npz:
        meta = {
            "format": "model_inputs_npz_v1",
            "created": _now_ts(),
            "created_time_unix": time.time(),
            "manifest": str(manifest_path.name),
            "shape_override": args.shape_override,
            "loaded_meta": loaded_inputs_meta,
            "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "nbytes": int(v.nbytes)} for k, v in feeds_full.items()},
        }
        out_npz = Path(args.save_inputs_npz)
        if not out_npz.is_absolute():
            out_npz = out_dir / out_npz
        _save_npz(str(out_npz), feeds_full, meta)
        print(f"[inputs] wrote {out_npz} (tensors={len(feeds_full)})")

    avail = _available_providers()
    print(f"ORT available providers: {avail}")
    providers = _pick_providers(args.provider, avail)
    print(f"Using providers: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = int(args.ort_log_severity)

    provider_options: Optional[List[dict]] = None
    if "TensorrtExecutionProvider" in providers:
        cache_dir = Path(args.trt_cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = base_dir / cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[tensorrt] engine cache dir: {cache_dir}")

        trt_opts = {
            "trt_engine_cache_enable": _as_ort_opt(bool(args.trt_cache)),
            "trt_engine_cache_path": str(cache_dir),
            "trt_timing_cache_enable": _as_ort_opt(bool(args.trt_cache)),
            "trt_timing_cache_path": str(cache_dir / "timing.cache"),
            "trt_fp16_enable": _as_ort_opt(bool(args.trt_fp16)),
            "trt_dump_subgraphs": _as_ort_opt(bool(args.trt_dump_subgraphs)),
        }
        if args.trt_fast_build:
            trt_opts["trt_builder_optimization_level"] = _as_ort_opt(2)
            trt_opts["trt_build_heuristics_enable"] = _as_ort_opt(True)

        provider_options = []
        for p in providers:
            provider_options.append(trt_opts if p == "TensorrtExecutionProvider" else {})

    sess_full, info_full = _create_session("full", full_path, providers, provider_options, sess_options)
    sess_p1, info_p1 = _create_session("part1", p1_path, providers, provider_options, sess_options)
    sess_p2, info_p2 = _create_session("part2", p2_path, providers, provider_options, sess_options)

    # If image_scale=auto, try to pick a sane preprocessing preset.
    if img_candidates is not None and args.image_scale == "auto":
        out_names = [o.name for o in sess_full.get_outputs()]

        def _run_full_with_scale(scale: str):
            feeds = dict(feeds_full)
            feeds[img_candidates["name"]] = img_candidates[scale]
            return sess_full.run(None, feeds)

        # Probe using "norm" once to identify output family (YOLO vs DETR-like vs unknown)
        out_norm = _run_full_with_scale("norm")

        if _is_yolo_multiscale(out_norm):
            print("[auto] Probing YOLO input scaling (norm vs raw)...")
            out_raw = _run_full_with_scale("raw")
            n_norm, p_norm = _yolo_plausibility(out_norm)
            n_raw, p_raw = _yolo_plausibility(out_raw)
            print(f"  image_scale=norm: {n_norm} dets, plausibility={p_norm:.1f}")
            print(f"  image_scale=raw : {n_raw} dets, plausibility={p_raw:.1f}")
            selected = "norm" if p_norm >= p_raw else "raw"
            print(f"[auto] Selected image_scale={selected}.")
            feeds_full[img_candidates["name"]] = img_candidates[selected]

        elif _is_detr_like(out_norm, out_names):
            print("[auto] Probing DETR/YOLOS input normalization (imagenet/norm/raw/clip)...")
            cand_order = ["imagenet", "norm", "raw", "clip"]
            best_scale = None
            best_plaus = -1e9
            for sc in cand_order:
                if sc not in img_candidates:
                    continue
                out_sc = out_norm if sc == "norm" else _run_full_with_scale(sc)
                dets = _decode_detr(
                    out_sc,
                    out_names,
                    img_hw=img_hw,
                    conf_thr=args.viz_conf,
                    iou_thr=args.viz_iou,
                    max_det=args.viz_max_det,
                )
                plaus = _detr_plausibility(dets)
                print(f"  image_scale={sc:8s}: {len(dets)} dets, plausibility={plaus:.3f}")
                if plaus > best_plaus:
                    best_plaus = plaus
                    best_scale = sc
            if best_scale is None:
                best_scale = "norm"
            print(f"[auto] Selected image_scale={best_scale}.")
            feeds_full[img_candidates["name"]] = img_candidates.get(best_scale, img_candidates["norm"])
        else:
            print("[auto] image_scale=auto -> defaulting to norm (unknown outputs).")
            feeds_full[img_candidates["name"]] = img_candidates["norm"]

    # Build feeds for part1/part2
    p1_inputs = [i.name for i in sess_p1.get_inputs()]
    feeds_p1: Dict[str, np.ndarray] = {}
    missing_p1: List[str] = []
    for name in p1_inputs:
        if name in feeds_full:
            feeds_p1[name] = feeds_full[name]
        else:
            missing_p1.append(name)
    if missing_p1:
        raise SystemExit(f"Missing inputs for part1: {missing_p1}")

    p1_out0 = sess_p1.run(None, feeds_p1)
    p1_map0: Dict[str, np.ndarray] = {o.name: a for o, a in zip(sess_p1.get_outputs(), p1_out0)}

    p2_inputs = [i.name for i in sess_p2.get_inputs()]
    missing_p2 = [n for n in p2_inputs if (n not in p1_map0 and n not in feeds_full)]
    if missing_p2:
        raise SystemExit(f"Missing inputs for part2 (neither cut tensor nor in full inputs): {missing_p2}")

    def build_feeds_p2(p1_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {}
        for n in p2_inputs:
            d[n] = p1_map[n] if n in p1_map else feeds_full[n]
        return d

    feeds_p2_0 = build_feeds_p2(p1_map0)

    if args.dump_interface:
        meta_base = {
            "format": "split_interface_npz_v1",
            "created": _now_ts(),
            "created_time_unix": time.time(),
            "manifest": str(manifest_path.name),
            "mode": args.dump_interface,
            "provider": args.provider,
            "providers_in_use": {"full": info_full.providers_in_use, "part1": info_p1.providers_in_use, "part2": info_p2.providers_in_use},
            "shape_override": args.shape_override,
            "inputs_npz": args.inputs_npz,
            "note": "NPZ keys are exact ORT input names. '__meta__' contains JSON metadata.",
            "left_inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "nbytes": int(v.nbytes)} for k, v in feeds_p1.items()},
            "right_inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "nbytes": int(v.nbytes)} for k, v in feeds_p2_0.items()},
        }
        _dump_interface_npz(args.dump_interface, args.dump_interface_out, out_dir, feeds_p1, feeds_p2_0, meta_base)

    if args.build_only:
        print("[build-only] running 1 inference(s) to finalize engine builds/caches...")
        _ = sess_full.run(None, feeds_full)
        _ = sess_p1.run(None, feeds_p1)
        _ = sess_p2.run(None, feeds_p2_0)
        report = {
            "created": _now_ts(),
            "manifest": str(manifest_path),
            "sessions": {"full": info_full.__dict__, "part1": info_p1.__dict__, "part2": info_p2.__dict__},
            "providers_available": avail,
            "providers_requested": providers,
            "note": "Build-only mode: no benchmarks/diffs.",
        }
        out_report = out_dir / "build_report.json"
        _write_json(out_report, report)
        print(f"[build-only] wrote {out_report}")
        print("[build-only] done.")
        return 0

    def run_full():
        return sess_full.run(None, feeds_full)

    def run_p1():
        return sess_p1.run(None, feeds_p1)

    def run_p2():
        return sess_p2.run(None, feeds_p2_0)

    def run_composed():
        out_p1 = sess_p1.run(None, feeds_p1)
        p1_map = {o.name: a for o, a in zip(sess_p1.get_outputs(), out_p1)}
        return sess_p2.run(None, build_feeds_p2(p1_map))

    full_mean, full_std, _ = _bench("full", run_full, args.warmup, args.runs)
    p1_mean, p1_std, _ = _bench("part1", run_p1, args.warmup, args.runs)
    p2_mean, p2_std, _ = _bench("part2", run_p2, args.warmup, args.runs)
    comp_mean, comp_std, _ = _bench("composed", run_composed, args.warmup, args.runs)

    full_out = run_full()
    comp_out = run_composed()

    n = min(len(full_out), len(comp_out))
    per_out = []
    max_abs_global = 0.0
    mean_abs_global = 0.0
    if n > 0:
        for i in range(n):
            a = full_out[i]
            b = comp_out[i]
            da = a.astype(np.float32, copy=False) if isinstance(a, np.ndarray) else np.array(a, dtype=np.float32)
            db = b.astype(np.float32, copy=False) if isinstance(b, np.ndarray) else np.array(b, dtype=np.float32)
            diff = np.abs(da - db)
            max_abs = float(np.max(diff)) if diff.size else 0.0
            mean_abs = float(np.mean(diff)) if diff.size else 0.0
            per_out.append({"index": i, "shape": list(getattr(a, "shape", [])), "max_abs": max_abs, "mean_abs": mean_abs})
            max_abs_global = max(max_abs_global, max_abs)
            mean_abs_global += mean_abs
        mean_abs_global = mean_abs_global / float(n)

    passed = (max_abs_global <= float(args.eps)) if n > 0 else True

    print("==== Timing summary (ms) ====")
    print(f"full     : {full_mean:.3f} ± {full_std:.3f}")
    print(f"part1    : {p1_mean:.3f} ± {p1_std:.3f}")
    print(f"part2    : {p2_mean:.3f} ± {p2_std:.3f}")
    print(f"composed : {comp_mean:.3f} ± {comp_std:.3f}")
    print("(note) part1+part2 (means) = {:.3f} ms".format(p1_mean + p2_mean))
    print("(note) composed - full       = {:.3f} ms".format(comp_mean - full_mean))
    print("==== Output diff ====")
    print(f"Compared outputs: {n}")
    print(f"max_abs : {max_abs_global}")
    print(f"mean_abs: {mean_abs_global}")
    print(f"PASS({args.eps}): {passed}")

    report = {
        "created": _now_ts(),
        "created_time_unix": time.time(),
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "args": vars(args),
        "providers_available": avail,
        "providers_requested": providers,
        "sessions": {"full": info_full.__dict__, "part1": info_p1.__dict__, "part2": info_p2.__dict__},
        "timing_ms": {
            "full_mean": full_mean,
            "full_std": full_std,
            "part1_mean": p1_mean,
            "part1_std": p1_std,
            "part2_mean": p2_mean,
            "part2_std": p2_std,
            "composed_mean": comp_mean,
            "composed_std": comp_std,
        },
        "output_diff": {"eps": float(args.eps), "passed": bool(passed), "max_abs": float(max_abs_global), "mean_abs": float(mean_abs_global), "per_output": per_out},
    }

    # Optional plots
    if not args.no_report_plots:
        _try_write_report_plots(out_dir, report)

    # Optional visualization
    viz_mode = (args.viz or "auto").lower()
    if viz_mode != "none":
        # Determine output names for heuristic detection
        out_names = [o.name for o in sess_full.get_outputs()]

        if viz_mode == "yolo" or (viz_mode == "auto" and _is_yolo_multiscale(full_out)):
            yolo_info = _maybe_yolo_viz(
                out_dir,
                img_path=img_path,
                img_hw=img_hw,
                full_out=full_out,
                comp_out=comp_out,
                conf_thr=float(args.yolo_conf),
                iou_thr=float(args.yolo_iou),
                max_det=int(args.yolo_max_det),
            )
            if yolo_info.get("enabled"):
                report.setdefault("visualization", {})
                report["visualization"]["yolo"] = yolo_info

        if viz_mode == "detr" or (viz_mode == "auto" and _is_detr_like(out_names, full_out)):
            labels = _load_labels(getattr(args, "labels", None))
            detr_info = _maybe_detr_viz(
                out_dir,
                img_path=img_path,
                img_hw=img_hw,
                output_names=out_names,
                full_out=full_out,
                comp_out=comp_out,
                conf_thr=float(getattr(args, "detr_conf", 0.25)),
                iou_thr=float(getattr(args, "detr_iou", 0.50)),
                max_det=int(getattr(args, "detr_max_det", 200)),
                labels=labels,
            )
            if detr_info.get("enabled"):
                report.setdefault("visualization", {})
                report["visualization"]["detr"] = detr_info

    out_report = out_dir / "validation_report.json"
    _write_json(out_report, report)
    print(f"Wrote {out_report}")

    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
'''
    script = script.replace("__MANIFEST_FILENAME__", str(manifest_filename))
    target = (target or "auto").lower()
    if target not in {"auto","cpu","cuda","tensorrt"}:
        target = "auto"
    script = script.replace("__DEFAULT_PROVIDER__", str(target))

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(script)
    try:
        os.chmod(out_path, 0o755)
    except Exception:
        pass

    # Also drop a default test image for convenience (so you can run the runner immediately).
    # If the tool package provides a `test_image.png` next to this file, prefer that.
    # Otherwise fall back to the embedded tiny placeholder.
    try:
        img_path = os.path.join(out_dir, "test_image.png")
        if not os.path.exists(img_path):
            import shutil
            pkg_dir = os.path.dirname(__file__)
            src = os.path.join(pkg_dir, "test_image.png")
            if os.path.exists(src):
                shutil.copyfile(src, img_path)
            else:
                import base64 as _b64
                with open(img_path, "wb") as f:
                    f.write(_b64.b64decode(_TEST_IMAGE_PNG_B64))
    except Exception:
        pass

    # Convenience wrappers (double-click friendly on Windows)
    try:
        script_name = os.path.basename(out_path)
        bat_path = os.path.join(out_dir, "run_split_onnxruntime.bat")
        bat = (
            "@echo off\n"
            "setlocal\n"
            "cd /d %~dp0\n"
            f"python \"{script_name}\" --manifest \"{manifest_filename}\" %*\n"
            "pause\n"
        )
        with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write(bat)
    except Exception:
        pass

    try:
        script_name = os.path.basename(out_path)
        sh_path = os.path.join(out_dir, "run_split_onnxruntime.sh")
        sh = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "cd \"$(dirname \"$0\")\"\n"
            f"python3 \"{script_name}\" --manifest \"{manifest_filename}\" \"$@\"\n"
        )
        with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(sh)
        try:
            os.chmod(sh_path, 0o755)
        except Exception:
            pass
    except Exception:
        pass

    return out_path


# Backwards-compatible alias (older GUI versions may call write_runner_skeleton)

def write_runner_skeleton(out_dir: str, *, manifest_filename: str = "split_manifest.json", target: str = "auto") -> str:
    return write_runner_skeleton_onnxruntime(out_dir, manifest_filename=manifest_filename, target=target)


def write_netron_launcher(out_dir: str, *, manifest_filename: str = "split_manifest.json") -> dict:
    """Create helper scripts to open the (split) models in Netron.

    This writes three files into out_dir:
      - open_netron_split.py
      - open_netron_split.bat
      - open_netron_split.sh

    The scripts read the split manifest to locate full/part1/part2 ONNX models and start
    Netron for each present file.

    Notes:
      - Netron is optional and must be installed separately: `pip install netron`.
      - Netron does not provide a stable programmatic PDF export API. Use your browser's
        print-to-PDF instead.
    """

    os.makedirs(out_dir, exist_ok=True)

    py_name = "open_netron_split.py"
    bat_name = "open_netron_split.bat"
    sh_name = "open_netron_split.sh"

    py_path = os.path.join(out_dir, py_name)
    bat_path = os.path.join(out_dir, bat_name)
    sh_path = os.path.join(out_dir, sh_name)

    py = r"""#!/usr/bin/env python3
import argparse
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path


def _load_manifest(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _start_netron(netron, model_path: Path, *, browse: bool, port: int):
    # Netron's Python API differs slightly across versions. Try common call patterns.
    try:
        return netron.start(str(model_path), browse=browse, port=port)
    except TypeError:
        try:
            return netron.start(str(model_path), browse=browse)
        except TypeError:
            return netron.start(str(model_path))


def main() -> int:
    ap = argparse.ArgumentParser(description='Open split models in Netron')
    ap.add_argument('--manifest', type=str, default='split_manifest.json')
    ap.add_argument('--what', type=str, default='all', choices=['all', 'full', 'part1', 'part2'])
    ap.add_argument('--no-browse', action='store_true', help='Do not auto-open the browser')
    ap.add_argument('--port', type=int, default=0, help='Port for Netron (0=auto). Use different ports if starting multiple servers.')
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return 2

    m = _load_manifest(manifest_path)
    base = manifest_path.parent

    # Paths in the manifest are stored relative to the split folder.
    candidates = {
        'full': base / m.get('full_model', ''),
        'part1': base / m.get('part1', ''),
        'part2': base / m.get('part2', ''),
    }

    files = []
    if args.what in ('all', 'full') and candidates['full'].exists():
        files.append(('full', candidates['full']))
    if args.what in ('all', 'part1') and candidates['part1'].exists():
        files.append(('part1', candidates['part1']))
    if args.what in ('all', 'part2') and candidates['part2'].exists():
        files.append(('part2', candidates['part2']))

    if not files:
        print('No model files found to open (check manifest paths).')
        return 3

    try:
        import netron  # type: ignore
    except Exception as e:
        print('Netron is not installed or failed to import.')
        print('Install with:  pip install netron')
        print(f'Import error: {type(e).__name__}: {e}')
        return 4

    print('Starting Netron...')
    for tag, p in files:
        try:
            url = _start_netron(netron, p, browse=(not args.no_browse), port=args.port)
            print(f'  {tag}: {p} -> {url}')
        except Exception as e:
            print(f'  {tag}: failed to start Netron for {p} ({type(e).__name__}: {e})')

    print("\nIf the browser did not open automatically, copy one of the URLs above.")
    print('Press Ctrl+C to stop Netron.')
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('Stopping.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
"""

    bat = """@echo off
setlocal
cd /d %~dp0
python open_netron_split.py --manifest {manifest}
pause
""".format(manifest=manifest_filename)

    sh = """#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 open_netron_split.py --manifest {manifest}
""".format(manifest=manifest_filename)

    with open(py_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(py)
    try:
        os.chmod(py_path, 0o755)
    except Exception:
        pass

    with open(bat_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(bat)

    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except Exception:
        pass

    return {
        'netron_py': py_name,
        'netron_bat': bat_name,
        'netron_sh': sh_name,
    }
