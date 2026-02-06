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
from .metrics import compute_tensor_bytes_per_value


def infer_shapes_safe(model: onnx.ModelProto) -> onnx.ModelProto:
    """Try to run ONNX shape inference.

    Shape inference is useful for nicer I/O ValueInfo, but splitting should work
    even if inference fails.
    """
    try:
        return shape_inference.infer_shapes(model)
    except Exception:
        return model


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

    try:
        onnx.checker.check_model(new_model)
    except Exception as e:
        raise RuntimeError(f"ONNX checker failed for submodel '{model_name}': {e}")

    return new_model, external_inputs


def strict_boundary_extras(full_model: onnx.ModelProto, cut_tensors: Iterable[str]) -> List[str]:
    """Return a list of *additional* external inputs required by part2 besides the cut tensors.

    This performs the same dependency analysis as the actual split (part2 creation) but returns
    only the extra inputs, without materializing or saving the submodel. If the returned list is
    empty, the boundary is *strict* in the sense that part2 depends only on the cut tensors
    (plus initializers/constants bundled into the model).
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
    extras = sorted([x for x in external_inputs if x not in cut_set])
    return extras


def compute_strict_boundary_ok(
    full_model: onnx.ModelProto,
    order: List[int],
    nodes: List[onnx.NodeProto],
) -> Tuple[List[bool], List[List[str]]]:
    """Compute whether each boundary is *strict* and (if not) which extra inputs appear.

    A boundary is considered *strict* if the right subgraph (part2) depends only on the cut
    activation tensors (plus initializers), i.e., it does not require any additional original
    graph inputs.

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
        extras = [x for x in p2_external_inputs if x not in cut_tensors]
        if extras:
            raise RuntimeError(
                "Strict boundary check failed. Part2 still requires external inputs besides cut tensors: "
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
iVBORw0KGgoAAAANSUhEUgAAAUAAAADwCAIAAAD+Tyo8AAATaklEQVR4nO3deXwUVbYH8Hurqpd0
ZyMsCQHZ1UEUEQi7DCAjuPvUzxue76MPwRVGB9kFFARBJLKMjAwozkR9MuPT8TmoPFFERBSQRUQd
RBFFBCIQIFun091V9f4oUlMkTZbu6q461b/vX7cq3feeOn2Pp5JQkbM5KgMAmiQuKlbHAAAxkriI
DgxAlcQlFDAAVejAAIShgAEIQwEDEIbvgQEIQwcGIAwFDEAYbqEBCEMHBiAMBQxAGAoYgDAUMABh
koCnkQDIQgcGIAwFDEAYChiAMBQwAGFxFfCJobkmhgL1aPnhL1aHAHaEDkwDPiaICgVMAz4miMqc
As7bdjT+SaCu4n752gAFDFFJXDBhZ5gyCdQDGYaozOnA6A+JhgxDVChgGpBhiAq30DQgwxAVOjAN
yDBEhQ5MAzIMUZnzOCGeSUw0ZLiuHy/q1PgXd/j2YOIisRA6MA3I8A9dOsfz9rrV3vHA9/FMaBMo
YBpSM8MHO3VJ3OTG/yJ0OnggcQslFAqYhpTK8PcdLmzMyzr/+J1Zc+r/pWjSnHaAAqYhRTJ8oN1F
5/tSl5++rXOuCTmp+/aoa2l1Hm0tm8KvkWhwdoa/a3Px+b504ZH9NUOTM6DPXHd1vbYNq9sUOjAN
Ts3wt61/FfX8Rce+qRkm/ML1teoGo9W2IRjbkTg3o4DNmATq4bwM78/rWvfkxcX7aoYWXK++eq3Y
tMI2xGYj6MA0OCzD37S6pNaZXx3/J2PMkrqtSwumVpD787rWBGkjKGAaHJPhfS261TrT9eTXjDGb
lK6RFpgxYK2kawK2BRQwDc7I8D9zLjUeXnLqK8aYDUvXSAvSGPm+Ft1qIrcevgemgXqGv252mfGw
2+kvGWM2L10jLWD9KrR6rrkKK6ED00A6w19ldTceXlq6l1DpGl1autd4LV83u+zS0r0WxsPQgamg
m+EvMy/Xx5eVfcEYI1q9Gu0S9Iv6Kqt7zUVZAx2YBooZ3pvew3jYvWIP6dI16l6xR786rZi7V+yx
JBJJ4GY8TmjGJFAPchne4+9pPOxRuZsxYpdQvx6Vu43XuDe9R4/K3ckPA7fQNNDK8Oe+Xvr4isAu
xphjeq+Rdmn6xe7x96y52ORBAdNAKMO703rr455VOx1ZukY9q3bql/y5r1fPqp3JXB0FTAOVDO/y
FujjXsEdjq9eTa/gDv3Cd6f17hXckbSlUcA0kMjwTk8ffdy7+rMUqV5N7+rP9Mvf5S3oXf1ZctZF
AdNg/wzvcPfVxwWh7SlVvZqC0HY9CTs9fQpC25OwKAqYBptn+DNXP33cJ7wtBatX0ye8TU/FDnff
PuFtiV4RBUwDlQz3jWxN2erV9I1s3S7118ZJ+NRQwDTYOcPbxAHaoJ/8aYpXr6af/KmWk+1S/37y
pwldS+JmZNyUSaAets3wVnGgPrZtkBbaJg7oL3+SuPnRgWmwZ4Y/FQbp4wHKFrRf3QBli56creLA
AcqWBC2EDkyDzTM8UPkY1VvLQOXjT4QrtXHiPj4UMA02zPAWYbA2GKRsRvVGNUjZrGXpE+HKQcrm
RCyBAqbBbhn+WPi1PrZbbPa0RRh8pfKR6dOigGmwbYYHK5vQfusxWNm0WRiijRPxIUqCGQ95mTIJ
1MNWGd4kDNMGQ5SNDntIMBGGKBu1jG0WhgxRNpo7OTowDfbJ8IfCVfrYPlFRsUkYNlT5wMQJUcA0
2DDDw5QNuHlupGHKho3CcG1s7keJAqbBJhn+QPiNPrZJSORsFIZfpbxv1mwoYBrsluHhyntov00y
XHlvg3C1Njbx05S4akYBmzEJ1MMOGX5fHKGP7RAPXRuEq38jrzdlKnRgGmyV4avld9F+Y3C1/O57
4khtbNYHig5Mg60ybKtgiDIrh+jANFie4Xela7XByMg6tN+YjYys0zK5XrpmZGRd/BOiA9Ngnwzb
JxLqTMkkCpgGazO8zn29TSJxkv9zXXdt6O04J0EB02CTDF9X/RazRyR0XVf91jueG7Rx/B8rCpgG
m2TYJmE4Bgo4VViY4bfSbrJDGI70tvfGG6r+Ec8MKGAa7JDhGwNv4v7ZFDcG3lzru1kbx/nJSoJi
xuOEZkwC9bBDhu0Qg/PEmVV0YBrskGE7xOA88XZgFDAJVmX4jczbtMEtZa/j/tlEt5S9ruX2fzNu
vaXs9ZjnkbhiRgGbMQnUw/IMWx6Ag8WTW3RgGizPsOUBOFg8uUUHpsHyDFsegIPF14FRwBRYnmHL
A3AwFLDzWZ5hywNwMHwP7HyWZPhvebdrg1HFa/AjaNONKl6jZfjV3P8YVbwmtknQgWmwNsP4fBMt
5gyjgGlAATtbHAUsm1HAZkwC9bA2w/h8Ey3mDKMD04AO7Gy4hXY4FLCzoYAdDgXsbLEXsCCb8Tih
GZNAPazNMD7fRIs5w+jANKADOxtuoR0OBexs+DWSw+HXSM4Wx6+RUMAUoICdDb8HdjjcQjsbbqEd
Dh3Y2XAL7XAoYGeLOcP8/i0rYl515aBxMb8XmiSejyke+kdsVQAOZkpuzenAkGiWf0yWB+Bg8eQW
BUyD5R+T5QE4mGUFPP795TG/F5rG6vpBAScOOjAkHPZJ4qCAIeGwTxInntzyh9YuMy8ScKBnbpyg
DbBVTGRWVs15nBBSAbZKIsSZVdxCQ2NhqyRCnFmVeASfCjTK0tsmTfzb01ZH4QRLRk3Wx3EWIDow
NGDSK4WL/3OKNsZuMdekVwrj/AUhChiaALvFXPHnEwUMTfD0nVOn/OUpq6OgrfCuafoYBQzJMHX1
wkV3T9fG2DBmmbp6Yfz/wA4/xIKmwYYxiymZ5NOfXRD/LJAKFo6foQ2wZ2Jmeg5xCw1Nhj0TP7Ny
iFtoaLInH5o5Y8kTVkdBz4KJs/SxWXWHDgyNNbNw3vwpj2pjbJt4zCycZ9bzoShgiMX8KY/OWjjX
6igoeWL6Y/rYxKJDAUMTPDr/8XkzZ2tj7JzYPDr/cRP/PAO+B4YYzZs5+7HH51gdBQ1zZ8/Rx6g4
AAAA+vjcabMafhXAuR576l+/RsIWqkeiE4UfYkEs5k2e+ejT87UxtlBjzJs8MxF/WpQ/8fAM0yeF
FDFr6dl/D4hdFFUS8oOfQoMJZi1dMP/BR6yOwl5mLn9SHyeuyviCcdMTNDWkghkrFupj7CVd0tKC
74EhLk/eN+2RVWcf8cdequvJ+6Yl9P+qwReOnZq42SFFTH9hkTbAdmLJzQY6MJhp+guLnho9xeoo
rDStqFAfJ6G4+KI7Jjf8KoCGTH35X39xNmU3VfKTwAtvn5SEZSAVTFmzWB+n4L6y5PL507+dmJyV
IBVMfnWJPk6prWXVhfPFtz6ctMUgFUz6+1J9nCK7y8JL5ktunpDM9SAVTHxzmT52/Aaz9mL5shse
SvKSkAomvPWM8dCR28wO18j/cO2DyV8VUsHv1y03Hjpsp9nk6vgzI35nycKQIh5a/0d97JjNZp+L
4suHj7dweUgFD2541nhIesvZ7Vr4H4eOszYCSAW/+3CF8ZDorrPhVfBnBz9gdQyQKsZv/pPxkNDe
s23kfMXA+62OAVLIuE9W1jpj8x1o84D5n/rfZ3UMkHIe2Lqq1hkb7kMSQfKVfe61OgZIRfd/9lzd
kzbZjXaOrRa+qtc9VscAqeu+Xc9HPW/JtrRVMI3En7vibqtjgFR37+erz/elJOxPa1ePE3+++1ir
YwA46569L5zvS6Zv1GSulTh8dbcxVscAcI67v/5zY17WpK2biDntgL/Q9S6rYwCIbuy+vyRnIbpV
wP988WirYwBo2Jj9ReZO6Iydz4u63Gl1DACxGH3gpca/2Kn7nL/Y6Q6rYwCAGElcwZ+VBaAKBQxA
GP6wOwBh6MAAhKGAAQiTuIoCBqAKHRiAMBQwAGEoYADC8D0wAGHowACEoQMDEIYCBiBMEhTF6hgA
IEbowACEoYABCEMBAxCGAgYgDAUMQBgKGIAwiTMUMABV6MAAhKEDAxCGDgxAGDowAGEoYADCUMAA
hKGAAQiTBIbHCQGoQgcGIAwFDEAYChiAMBQwAGES5yhgAKrQgQEIQwcGIAwFDEAYChiAMBQwAGEo
YADCGi7gtXLJ/8gnXIyHmTpKbHm92JwxNiC45zLBv8p9ofaaIcEvNnkvP9/5rUrZG/LJQlcnxthB
NTgn/GOR+2KBccbYN0pgeeRohKkiY7Nd7XO5u24A2iQH1eBupfw2sSVjrChSPFrK0776oXzmr/Jx
xtgepbKH4GeM/VZsdZWYXc8V6dECUNdAAW+Vy/4hn1zp7pLBxXJV/n3o+1aCq6+Q4eZcZuputbyX
kM4YY5xp80Q9P0DMeFU+/rla3lNIXxw6PNXVVuSMMZUxNjdyaJm7cy53fSCfWRY5stDdIUoQnHGu
duaezoJHe1eR/Mtdrlzti8OkrGFSFmPs18G9z3surHlPvf9VqokWgLoGCvgl+fjD7vxMQWBMzeTC
BHf+ynBxP086Y+wBV97K8LHV3i7aK/V5op6f6GozO/TTna6W+YK7u+jTC+yUGglzmXNpiJTZXJA4
VwdXfXmL1HyvXMkZm+tp34a79UkGV325Oe2yleHigKqMDx1Y4elcK1rtZSVqeE7ocECVfVyc476A
MWY8bM5dxmgBSJMEXt/jhD8owa6iV3/k8BLRe7A6qL2lr+R7LqLuUsoKxHTGmD5P1POdRFd3MW1x
6MiraRcZV3zInTc2+N0gMfM6KbtATGdMCatKN8E70Z33TuT0ktDPS70djJMLXBnnbrUmcmKltyOr
8yCk9rIloSPXSFnXS83ejpxeGj6iMmY8XOBpZ5wQgDSJCw30Ii6oXB8zhTOmvYUL6jhP7rPVv/Rx
+VnNyXrOVzJZ5LyKy9mCoE9+kzt7qCtjY6SsMHR0mJT5gCeXMX6VO5Mz9Wp31pLQMX0tffJag1qh
MsZ2ypVz09pypo5wZ/2hopgxZjysNSEAaQ3cQncWPd8ogR6iTzvcJ1d1ET3aWzhXCySfGGI75Apm
uCmNev5zOVDB5Me8+Qurjzzja6+98rQaOaSEeoi+f3NnD3Gl31JxYJy3lcCZxFXOGGeqm3N9LX3y
WgOjmpMqr5lBu1c3HtaaEIC0BjrwXd7mS4PFK/zt0rlYrsrLqovHe1sZm9h4b6tngr+wOr3ReF5m
amHlsUJ/27aC+7XwqU1y2VBXBmOMq2xy5eFX0jvmCa5SOdJacHFBlVX1Y7l8iCvj/dCZPpI/agdW
maoKqlAnWu2rBS7/hkjpde6sDaHSApefMWY8RAcGJ2mgAw90+Y8r4TEVP7o5D6vq7d6cfi7tR1Bn
W1lvV5oryMNMMXa/WudfCZb0d/kvEF2MqdN8ufdW/NTf5UvjQg4X5vhaTwoc9jBBZGyeP59z1c35
hnBZUfXJDC5qZ9oL7tXVJ+7xttAn7yX5Hqw8tCK93bnBnv3q5LTcxwJHXwudSuPCPF++ypjx8NwJ
AWjjXzfvanUM5+h3av+2nIutjgKAhoZ/iJV8NgwJwJ74vly0OwCq8G+hAQiz4y00ADSSxEUUMABV
DXTg9wKVL5aXMsZ2Bat6edMYY3dkZI70pdfzllWlZ+7LytYPex7+YfcFHc0JFgDO1UABj0j3jUj3
McauOPTjmtata07X95ZVZafvb5ZlPIO7dIAEacL3wNorSxXl8RMlJ2Q5rKqPtMi53ON5qbTstfJy
xtjUnJxdwWBAUUYXH3sxP6/WG3v8cGiE3789WHVvdvbOquDu6uCdmZljsrO+C4VmnigpU5R/z0gf
k511UpZnnDhZKittXdKmQGBXh/Z1V0xAHgBIanIBLzxR8l/NMnp4PUcjkXuOHn+nXf7y02c+6tCm
OCKvOFW6OK9FUWnZS21zjV1ae2O1qt6enT5Byrryh5/fuKD1ZCn71sPHxuZkvlxWNrVF9oVu98if
jozNyXzy+KnrM/w3Zfjfqwisq6jkglp3xUQkAoAiSRAb+2Cd9sqPA1WHImHtTJWqqKIyNN076fiJ
O7IzlrbJ0R7xqzWndigwdrnfJTLm4vxyv0tgrEpRBVGZkZu1tizwYVWgQlEEUdleVbUov5nAleGZ
XvEXJohK3RVFs64egLgmd+AIYy+3b+nhXGFsR6BaEtQlbXK2B6pfKClfW175dH4Oq/NNr3bo4lwS
VMaYh3PR+OzhzyevyfCNzkl/+XQ5F9SQ9gAjZ0xVVZVxQa27oonXD0Bao3+NxJn2ygK/e31F4KZs
30flwaKSim6+nDGHSv7asUV3f87A/ce4qKpMVUXDo0I1bzzfYG9VaEW7nBJZCamMi2pvn/v9yqrr
s9LWlwZUxrio1lqxXwYeQgA4q8kdeHabrGmHz/z3qUqJs0UXZGe6+PBM703fH1cYm5CbyQW1T7pn
7KGTRR2b13rj+QZ3tvDffPD4JV5XlsjDXJndJmvCT6eKSsp7+d0+kXNBrbUifqYNoOM/97TXz4Qe
PnTm3lb+rmmuLwLhuT+X/v0i9FuA87Ldv8Qak+ubdbjUK/CQoi7okGW38ABshR/tm9fwqwDAlmzX
gQGg8VDAAIShgAEIQwEDEIYCBiAMBQxAGAoYgDAUMABhTXicEADsBh0YgDAUMABhKGAAwlDAAIRJ
XEIBA1CFDgxAGAoYgDDcQgMQhg4MQBgKGIAwFDAAYfgeGIAwdGAAwiQBBQxA1v8DLwZwb+dSBq8A
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
    script = r'''#!/usr/bin/env python3
"""run_split_onnxruntime.py

Auto-generated runner skeleton by the ONNX Split-Point Analyser.

It validates and benchmarks:
  full(x)
  part2(part1(x))

Timing protocol (defaults):
  - warmup: 5 runs (not measured)
  - measured: 10 runs

It writes:
  - validation_report.json (always)
  - validation_report.png / .pdf (if matplotlib is installed)

Usage:
  pip install onnx onnxruntime numpy
  # optional for image input + plots:
  pip install pillow matplotlib

  python run_split_onnxruntime.py --manifest __MANIFEST_FILENAME__
  python run_split_onnxruntime.py --manifest __MANIFEST_FILENAME__ --image test_image.png --preset auto

Notes:
  - Inputs are generated from ONNX input shapes/dtypes (unknown dims become 1 or --batch).
  - If --image is given and the selected input looks like an image tensor, the image is used for that input.
  - Use --provider (auto|cpu|cuda|tensorrt) to select the onnxruntime Execution Provider.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto


def pick_providers(provider: str) -> List[str]:
    """Pick ORT execution providers.

    Notes:
      * We intentionally do **NOT** prefer TensorRT in 'auto'. Many ORT GPU wheels
        list TensorRT as available, but loading it fails unless TensorRT DLLs are
        installed (nvinfer_*.dll, etc.). For workstation benchmarking we want a
        reliable default.
      * In 'auto', we prefer CUDA if present, otherwise fall back to CPU.
      * Users can explicitly request TensorRT via --provider tensorrt.
    """
    avail = ort.get_available_providers()
    provider = (provider or '').strip().lower()

    if provider in ("auto", ""):
        if "CUDAExecutionProvider" in avail:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "DmlExecutionProvider" in avail:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if provider in ("cuda", "gpu"):
        return [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in avail]

    if provider in ("tensorrt", "trt"):
        pref = []
        if "TensorrtExecutionProvider" in avail:
            pref.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider" in avail:
            pref.append("CUDAExecutionProvider")
        pref.append("CPUExecutionProvider")
        return pref

    if provider in ("cpu",):
        return ["CPUExecutionProvider"]

    # Explicit provider string
    cap = provider[0].upper() + provider[1:]
    cand = cap + "ExecutionProvider"
    if cand in avail:
        return [cand]

    # Fallback
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def _np_dtype_from_onnx(elem_type: int) -> np.dtype:
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


def _get_non_initializer_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    g = model.graph
    init_names = {i.name for i in g.initializer}
    return [vi for vi in g.input if vi.name and vi.name not in init_names]


def _dims_from_vi(vi: onnx.ValueInfoProto, batch: Optional[int]) -> List[int]:
    tt = vi.type.tensor_type
    dims: List[int] = []
    for i, d in enumerate(tt.shape.dim):
        if d.dim_value and d.dim_value > 0:
            dims.append(int(d.dim_value))
        else:
            dims.append(int(batch) if (i == 0 and batch is not None) else 1)
    return dims or [1]


def _make_random_inputs(model: onnx.ModelProto, batch: Optional[int], seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    feeds: Dict[str, np.ndarray] = {}
    for vi in _get_non_initializer_inputs(model):
        tt = vi.type.tensor_type
        dtype = _np_dtype_from_onnx(tt.elem_type)
        dims = _dims_from_vi(vi, batch)
        if np.issubdtype(dtype, np.floating):
            arr = rng.standard_normal(dims).astype(dtype)
        elif dtype == np.bool_:
            arr = (rng.random(dims) > 0.5).astype(dtype)
        else:
            lo, hi = (-5, 6) if dtype == np.int8 else (0, 11)
            arr = rng.integers(lo, hi, size=dims, dtype=dtype)
        feeds[vi.name] = arr
    return feeds


def _try_import_pil() -> bool:
    try:
        from PIL import Image  # noqa: F401
        return True
    except Exception:
        return False


def _infer_layout_hw(shape: List[int]) -> Tuple[str, int, int, int]:
    """Infer layout and H/W/C from a 4D (or 3D) image-like shape.
    Returns (layout, H, W, C). layout is 'NCHW' or 'NHWC'.
    """
    if len(shape) == 4:
        _, d1, d2, d3 = shape
        if d1 in (1, 3, 4):  # NCHW
            return "NCHW", int(d2), int(d3), int(d1)
        if d3 in (1, 3, 4):  # NHWC
            return "NHWC", int(d1), int(d2), int(d3)
        return "NCHW", int(d2), int(d3), int(d1)
    if len(shape) == 3:
        d0, d1, d2 = shape
        if d0 in (1, 3, 4):  # CHW
            return "NCHW", int(d1), int(d2), int(d0)
        if d2 in (1, 3, 4):  # HWC
            return "NHWC", int(d0), int(d1), int(d2)
        return "NCHW", int(d1), int(d2), int(d0)
    return "NCHW", 224, 224, 3


def _letterbox(img, new_w: int, new_h: int, color=(114, 114, 114)):
    """YOLO-style letterbox resize."""
    from PIL import Image

    w, h = img.size
    scale = min(new_w / w, new_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (new_w, new_h), color)
    pad_w = (new_w - nw) // 2
    pad_h = (new_h - nh) // 2
    canvas.paste(img_resized, (pad_w, pad_h))
    return canvas


def _image_to_tensor(
    image_path: str,
    *,
    target_shape: List[int],
    elem_type: int,
    preset: str,
    yolo_scale: str = "norm",
    batch: Optional[int],
    return_meta: bool = False,
):
    """Load an image and convert it to a tensor compatible with an image-like ONNX input.

    If return_meta=True, returns (tensor, meta_dict). meta_dict is mainly useful for YOLO
    to map decoded boxes back from letterboxed input space to the original image.
    """
    if not _try_import_pil():
        raise RuntimeError("Image support requires Pillow: pip install pillow")
    from PIL import Image

    shape = list(target_shape)
    if batch is not None and len(shape) >= 1:
        shape[0] = int(batch)

    layout, H, W, _ = _infer_layout_hw(shape)

    # Heuristic for unknown H/W (symbolic dims)
    if H <= 1 or W <= 1:
        if (preset or "auto").lower() == "yolo":
            H = W = 640
        else:
            H = W = 224

    img = Image.open(image_path).convert("RGB")

    p = (preset or "auto").lower()
    if p == "auto":
        # If the model expects integer inputs (uint8/int8), the safest default is raw pixels.
        # Many quantized ONNX models perform any normalization inside the graph.
        dtype_auto = _np_dtype_from_onnx(int(elem_type))
        if np.issubdtype(dtype_auto, np.integer):
            p = "raw"
        else:
            p = "yolo" if max(H, W) >= 512 else "imagenet"

    # Guardrail: float-normalization presets don't make sense for integer model inputs unless
    # you also know the quantization parameters (scale/zero-point). If a user accidentally
    # selects such a preset (or uses auto on a quantized model), fall back to raw to avoid
    # producing near-all-zero / saturated inputs.
    dtype_chk = _np_dtype_from_onnx(int(elem_type))
    if np.issubdtype(dtype_chk, np.integer) and p not in ("raw", "yolo"):
        print(f"[warn] preset={p} but model input is {dtype_chk}. Falling back to preset=raw.")
        p = "raw"


    value_range = "0_1"  # assume arr in [0,1] unless noted otherwise

    meta = {
        "preset": p,
        "orig_size": [int(img.size[0]), int(img.size[1])],
        "input_size": [int(W), int(H)],
        "layout": layout,
    }

    if p == "yolo":
        # Store letterbox parameters so we can map boxes back to the original image.
        orig_w, orig_h = img.size
        scale = min(W / orig_w, H / orig_h)
        nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
        pad_w = (W - nw) // 2
        pad_h = (H - nh) // 2
        meta.update(
            {
                "scale": float(scale),
                "pad": [int(pad_w), int(pad_h)],
                "new_unpadded": [int(nw), int(nh)],
            }
        )

        img = _letterbox(img, W, H)
        arr = np.asarray(img).astype(np.float32)

        ys = (yolo_scale or "norm").lower()
        meta["yolo_scale"] = ys
        if ys in ("norm", "normalized", "0_1", "1/255", "div255", "true"):
            arr = arr / 255.0
            value_range = "0_1"
        elif ys in ("raw", "0_255", "255"):
            # keep 0..255
            value_range = "0_255"
        else:
            raise ValueError(f"Unknown yolo_scale: {yolo_scale}")
    elif p == "imagenet":
        img = img.resize((W, H), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
    elif p == "raw":
        img = img.resize((W, H), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if layout == "NCHW":
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        arr = np.expand_dims(arr, 0)  # N
    else:
        arr = np.expand_dims(arr, 0)  # N, H, W, C

    if batch is not None and int(batch) > 1:
        arr = np.repeat(arr, int(batch), axis=0)

    dtype = _np_dtype_from_onnx(int(elem_type))
    is_01 = (value_range == "0_1")
    if np.issubdtype(dtype, np.floating):
        arr = arr.astype(dtype)
    elif dtype == np.uint8:
        tmp = (arr * 255.0) if is_01 else arr
        arr = np.clip(tmp, 0, 255).astype(np.uint8)
    elif dtype == np.int8:
        tmp = ((arr * 255.0) if is_01 else arr).astype(np.int16) - 128
        arr = np.clip(tmp, -128, 127).astype(np.int8)
    else:
        arr = arr.astype(dtype)

    if return_meta:
        return arr, meta
    return arr



def _bench(name: str, fn, warmup: int, runs: int):
    """Benchmark helper with visible progress."""
    warmup = int(warmup)
    runs = int(runs)

    # NOTE: keep this as a *single-line* string literal in the generated runner.
    # A previous version accidentally inserted a real newline inside the quotes,
    # producing a SyntaxError (unterminated string literal).
    print(f"\n[{name}] warmup: {warmup} runs (not measured)")
    for i in range(warmup):
        fn()
        if warmup <= 10:
            print(f"  warmup {i+1}/{warmup}", flush=True)
        else:
            if (i + 1) % 5 == 0:
                print(f"  warmup {i+1}/{warmup}", flush=True)

    times: List[float] = []
    last_out = None
    print(f"[{name}] measured: {runs} runs")
    for i in range(runs):
        t0 = time.perf_counter()
        last_out = fn()
        t1 = time.perf_counter()
        ms = 1000.0 * (t1 - t0)
        times.append(float(ms))
        print(f"  run {i+1}/{runs}: {ms:.3f} ms", flush=True)

    mean = float(np.mean(times)) if times else 0.0
    std = float(np.std(times)) if times else 0.0
    return mean, std, times, last_out


def _compare_outputs(
    full_names: List[str],
    full_out: List[np.ndarray],
    p2_names: List[str],
    p2_out: List[np.ndarray],
) -> Dict[str, object]:
    full_map = dict(zip(full_names, full_out))
    p2_map = dict(zip(p2_names, p2_out))

    common = [n for n in full_names if n in p2_map]
    if not common and len(full_out) == len(p2_out):
        common = list(full_names)

    per = []
    max_abs_all = 0.0
    mean_abs_sum = 0.0
    n_out = 0
    for name in common:
        a = full_map.get(name)
        b = p2_map.get(name)
        if a is None or b is None:
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        max_abs = float(diff.max(initial=0.0))
        mean_abs = float(diff.mean())
        per.append({"name": name, "max_abs": max_abs, "mean_abs": mean_abs, "shape": list(a.shape)})
        max_abs_all = max(max_abs_all, max_abs)
        mean_abs_sum += mean_abs
        n_out += 1

    return {
        "max_abs": float(max_abs_all),
        "mean_abs": float(mean_abs_sum / max(n_out, 1)),
        "per_output": per,
        "n_outputs_compared": int(n_out),
        "common_outputs": common,
    }



# ---------------------------- Optional: YOLO visualization ----------------------------

COCO80_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush",
]

# Default YOLOv7 COCO anchors (pixels) for 640 input: P3/8, P4/16, P5/32
YOLOV7_ANCHORS = np.array(
    [
        [[12, 16], [19, 36], [40, 28]],
        [[36, 75], [76, 55], [72, 146]],
        [[142, 110], [192, 243], [459, 401]],
    ],
    dtype=np.float32,
)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_yolo_tensor(arr: np.ndarray) -> Optional[np.ndarray]:
    """Try to interpret arr as YOLO tensor and return shape (A=3, H, W, C)."""
    if arr is None:
        return None
    if arr.ndim == 5:
        a0 = arr[0]
        # (3, H, W, C)
        if a0.ndim == 4 and a0.shape[0] == 3:
            return a0
        # (H, W, 3, C)
        if a0.ndim == 4 and a0.shape[2] == 3:
            return np.transpose(a0, (2, 0, 1, 3))
    if arr.ndim == 4:
        # (B, 3*(5+nc), H, W)
        a0 = arr[0]
        if a0.ndim == 3:
            c, h, w = a0.shape
            if c % 3 == 0:
                c_per = c // 3
                return a0.reshape(3, c_per, h, w).transpose(0, 2, 3, 1)
    return None


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU between one box (4,) and many boxes (N,4) in xyxy."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-9
    return inter / union


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> List[int]:
    """Pure numpy NMS. Returns indices to keep."""
    if boxes.size == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep: List[int] = []
    while idxs.size > 0 and len(keep) < int(max_det):
        i = int(idxs[0])
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _iou_xyxy(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= float(iou_thres)]
    return keep


def _plausibility_score_dets(
    dets: Optional[List[dict]],
    img_meta: Optional[dict],
    *,
    max_det: int,
) -> float:
    """Heuristic score for how *plausible* a detection set is.

    We use this for two things:
      (1) choosing between different YOLO output decoders (some exports output
          already-decoded boxes, others expose raw head outputs),
      (2) probing whether a YOLO model expects 0..1 input ("norm") or 0..255
          input ("raw").

    The goal is not to be perfect, but to avoid obvious failure modes such as:
      - saturating confidences close to 1.0 for *many* boxes,
      - hundreds of tiny boxes with nearly-identical scores.
    """
    if dets is None or len(dets) == 0:
        return -1e9

    n = int(len(dets))
    topk = int(min(50, n))
    scores = np.asarray([float(d.get("score", 0.0)) for d in dets[:topk]], dtype=np.float32)
    mean_s = float(np.mean(scores)) if scores.size else 0.0
    std_s = float(np.std(scores)) if scores.size else 0.0
    sat = float(np.mean(scores > 0.999)) if scores.size else 0.0

    # Boxes: prefer that the top detections are not *all* extremely tiny.
    boxes_list = []
    for d in dets[:topk]:
        b = d.get("box_xyxy_in") or d.get("box_xyxy_orig")
        if b is None or len(b) != 4:
            continue
        try:
            boxes_list.append([float(x) for x in b])
        except Exception:
            pass

    area_term = 0.0
    if boxes_list:
        boxes = np.asarray(boxes_list, dtype=np.float32)
        wh = np.maximum(0.0, boxes[:, 2:4] - boxes[:, 0:2])
        areas = wh[:, 0] * wh[:, 1]

        in_w, in_h = (img_meta.get("input_size") if img_meta else (None, None)) or (None, None)
        if in_w is None or in_h is None:
            in_w, in_h = 640, 640
        denom = float(in_w) * float(in_h) + 1e-9
        areas = areas / denom
        med_area = float(np.median(areas)) if areas.size else 0.0
        # Map median area to [0,1] on log scale (1e-6 .. 1)
        area_term = float(np.clip((np.log10(med_area + 1e-12) + 6.0) / 6.0, 0.0, 1.0))

    # Prefer a moderate number of detections (not empty, not maxed out).
    count_term = float(np.exp(-((float(n) - 20.0) / 40.0) ** 2))
    maxdet_pen = 1.0 if n >= int(max_det) else 0.0

    # Combine terms.
    score = 100.0 * mean_s + 30.0 * count_term + 60.0 * area_term - 120.0 * sat - 20.0 * maxdet_pen
    if mean_s > 0.99 and std_s < 1e-3 and n >= int(max_det):
        score -= 50.0
    return float(score)


def _decode_yolo_v7_multiscale(
    names: List[str],
    outs: List[np.ndarray],
    *,
    img_meta: Optional[dict],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> Optional[List[dict]]:
    """Decode YOLOv7-style outputs (3 scales) and return list of detections dicts.

    Expected per-scale output shapes include (B,3,H,W,5+nc). This decoder assumes the
    common YOLOv5/v7 head parameterization:
      xy = (sigmoid(xy)*2 - 0.5 + grid) * stride
      wh = (sigmoid(wh)*2)**2 * anchor
      score = sigmoid(obj) * sigmoid(cls)

    NOTE: anchors are assumed to be the default YOLOv7 COCO anchors.
    """
    # pick candidate outputs
    cands = []
    for n, a in zip(names, outs):
        yt = _as_yolo_tensor(a)
        if yt is None:
            continue
        if yt.shape[0] != 3 or yt.shape[-1] < 6:
            continue
        H, W = int(yt.shape[1]), int(yt.shape[2])
        cands.append((H * W, H, W, n, yt))

    if len(cands) < 3:
        return None

    # choose 3 highest-resolution scales (largest H*W)
    cands.sort(key=lambda x: x[0], reverse=True)
    cands = cands[:3]

    # order: large grid -> small grid
    cands.sort(key=lambda x: x[0], reverse=True)

    # input size (letterboxed)
    in_w, in_h = 640, 640
    if img_meta and "input_size" in img_meta:
        in_w, in_h = int(img_meta["input_size"][0]), int(img_meta["input_size"][1])

    all_boxes = []
    all_scores = []
    all_cls = []

    for si, (_, H, W, oname, yt) in enumerate(cands):
        # infer stride from input size and grid
        stride_w = float(in_w) / float(W)
        stride_h = float(in_h) / float(H)
        stride = 0.5 * (stride_w + stride_h)

        anchors = YOLOV7_ANCHORS[min(si, YOLOV7_ANCHORS.shape[0]-1)].reshape(3, 1, 1, 2)

        # grid (H, W, 2) -> broadcast to (3, H, W, 2)
        gy, gx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
        grid = np.stack((gx, gy), axis=-1)  # (H,W,2)
        grid = grid.reshape(1, H, W, 2)

        # yt: (3,H,W,C)
        xy = (_sigmoid(yt[..., 0:2]) * 2.0 - 0.5 + grid) * stride
        wh = (_sigmoid(yt[..., 2:4]) * 2.0) ** 2 * anchors

        obj = _sigmoid(yt[..., 4:5])  # (3,H,W,1)
        cls = _sigmoid(yt[..., 5:])   # (3,H,W,nc)
        scores = obj * cls            # (3,H,W,nc)

        best_cls = scores.argmax(axis=-1)      # (3,H,W)
        best_score = scores.max(axis=-1)       # (3,H,W)

        mask = best_score > float(conf_thres)
        if not np.any(mask):
            continue

        xy_m = xy[mask]  # (N,2)
        wh_m = wh[mask]
        x1y1 = xy_m - wh_m / 2.0
        x2y2 = xy_m + wh_m / 2.0
        boxes = np.concatenate([x1y1, x2y2], axis=1)  # (N,4)
        all_boxes.append(boxes)
        all_scores.append(best_score[mask].astype(np.float32))
        all_cls.append(best_cls[mask].astype(np.int64))

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    cls_ids = np.concatenate(all_cls, axis=0)

    # NMS (class-aware by default)
    dets: List[dict] = []
    for c in np.unique(cls_ids):
        idx = np.where(cls_ids == c)[0]
        keep = _nms_xyxy(boxes[idx], scores[idx], float(iou_thres), int(max_det))
        for k in keep:
            i = int(idx[k])
            dets.append(
                {
                    "cls_id": int(cls_ids[i]),
                    "score": float(scores[i]),
                    "box_xyxy_in": [float(x) for x in boxes[i].tolist()],
                }
            )

    dets.sort(key=lambda d: d["score"], reverse=True)
    dets = dets[: int(max_det)]

    # map boxes back to original image (undo letterbox), if meta available
    if img_meta and img_meta.get("preset") == "yolo" and "scale" in img_meta and "pad" in img_meta:
        scale = float(img_meta.get("scale", 1.0))
        pad_w, pad_h = img_meta.get("pad", [0, 0])
        pad_w = float(pad_w)
        pad_h = float(pad_h)
        orig_w, orig_h = img_meta.get("orig_size", [in_w, in_h])
        orig_w = float(orig_w)
        orig_h = float(orig_h)

        for d in dets:
            x1, y1, x2, y2 = d["box_xyxy_in"]
            x1 = (x1 - pad_w) / scale
            x2 = (x2 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            y2 = (y2 - pad_h) / scale
            # clip
            x1 = max(0.0, min(orig_w, x1))
            x2 = max(0.0, min(orig_w, x2))
            y1 = max(0.0, min(orig_h, y1))
            y2 = max(0.0, min(orig_h, y2))
            d["box_xyxy_orig"] = [x1, y1, x2, y2]
    else:
        for d in dets:
            d["box_xyxy_orig"] = d["box_xyxy_in"]

    # attach label
    for d in dets:
        cid = int(d["cls_id"])
        d["label"] = COCO80_NAMES[cid] if 0 <= cid < len(COCO80_NAMES) else f"cls{cid}"
    return dets





def _decode_dets_separate(
    output_names: List[str],
    outputs: List[np.ndarray],
    img_meta: Optional[dict],
    *,
    conf_thres: float,
    max_det: int,
) -> Optional[List[Dict[str, object]]]:
    """Fallback for models that output detections as separate tensors.

    Common pattern (often when NMS is inside the graph):
      boxes   : (B,N,4) or (N,4)
      scores  : (B,N) or (N,)
      classes : (B,N) or (N,)
      num_dets: (B,) or (1,) (optional)

    Returns list with keys compatible to YOLO drawing: cls_id, score, box_xyxy_orig.
    """

    # Collect candidates
    boxes_cand = []
    scores_cand = []
    classes_cand = []
    num_cand = []

    for name, arr in zip(output_names, outputs):
        if not isinstance(arr, np.ndarray):
            continue
        a = np.asarray(arr)
        if a.size == 0:
            continue

        # num detections (scalar or length-1)
        if a.ndim == 0 or (a.ndim == 1 and a.size == 1):
            num_cand.append((name, a))
            continue

        # boxes
        if (a.ndim == 3 and a.shape[0] >= 1 and a.shape[-1] == 4):
            boxes_cand.append((name, a[0]))
            continue
        if (a.ndim == 2 and a.shape[-1] == 4):
            boxes_cand.append((name, a))
            continue

        # 1D scores/classes
        if a.ndim == 2 and a.shape[0] >= 1 and a.shape[1] >= 1:
            # (B,N)
            scores_cand.append((name, a[0]))
            classes_cand.append((name, a[0]))
            continue
        if a.ndim == 1 and a.shape[0] >= 1:
            scores_cand.append((name, a))
            classes_cand.append((name, a))
            continue

    if not boxes_cand:
        return None

    # Choose the largest boxes tensor and try to match scores/classes by length
    boxes_name, boxes = max(boxes_cand, key=lambda x: x[1].shape[0])
    boxes = np.asarray(boxes, dtype=np.float32)
    N = int(boxes.shape[0])

    # Pick score tensor that best matches N (prefer exact)
    score = None
    for n, s in scores_cand:
        s = np.asarray(s).reshape(-1)
        if s.shape[0] == N:
            score = (n, s)
            break
    if score is None and scores_cand:
        score = min(scores_cand, key=lambda x: abs(x[1].reshape(-1).shape[0] - N))

    cls = None
    for n, c in classes_cand:
        c = np.asarray(c).reshape(-1)
        if c.shape[0] == N:
            cls = (n, c)
            break
    if cls is None and classes_cand:
        cls = min(classes_cand, key=lambda x: abs(x[1].reshape(-1).shape[0] - N))

    if score is None or cls is None:
        return None

    scores = np.asarray(score[1], dtype=np.float32).reshape(-1)
    classes = np.asarray(cls[1], dtype=np.float32).reshape(-1)

    # Optional num_dets
    num_keep = None
    for _, nd in num_cand:
        try:
            v = int(np.asarray(nd).reshape(-1)[0])
            if 0 < v <= N:
                num_keep = v
                break
        except Exception:
            pass

    if num_keep is not None:
        boxes = boxes[:num_keep]
        scores = scores[:num_keep]
        classes = classes[:num_keep]
        N = num_keep

    # Filter by conf threshold
    keep = scores > float(conf_thres)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]

    if boxes.shape[0] == 0:
        return []

    # Map coords (normalized/pixel) and undo letterbox to original.
    in_w, in_h = (img_meta.get("input_size") if img_meta else (None, None)) or (None, None)
    if in_w is None or in_h is None:
        in_w = float(np.max(boxes[:, [0, 2]]))
        in_h = float(np.max(boxes[:, [1, 3]]))

    if float(np.max(boxes)) <= 2.0:
        boxes[:, [0, 2]] *= float(in_w)
        boxes[:, [1, 3]] *= float(in_h)

    boxes[:, 0] = np.clip(boxes[:, 0], 0, float(in_w))
    boxes[:, 2] = np.clip(boxes[:, 2], 0, float(in_w))
    boxes[:, 1] = np.clip(boxes[:, 1], 0, float(in_h))
    boxes[:, 3] = np.clip(boxes[:, 3], 0, float(in_h))

    scale = float((img_meta or {}).get("scale", 1.0))
    pad = (img_meta or {}).get("pad", [0, 0])
    pad_w, pad_h = float(pad[0]), float(pad[1])
    orig = (img_meta or {}).get("orig_size", [float(in_w), float(in_h)])
    orig_w, orig_h = float(orig[0]), float(orig[1])

    dets = []
    idxs = np.argsort(scores)[::-1][: int(max_det)]
    for i in idxs:
        x1, y1, x2, y2 = boxes[i]
        x1 = (x1 - pad_w) / scale
        x2 = (x2 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        y2 = (y2 - pad_h) / scale
        x1 = float(np.clip(x1, 0, orig_w))
        x2 = float(np.clip(x2, 0, orig_w))
        y1 = float(np.clip(y1, 0, orig_h))
        y2 = float(np.clip(y2, 0, orig_h))
        dets.append({
            "cls_id": int(round(float(classes[i]))),
            "score": float(scores[i]),
            "box_xyxy_orig": [x1, y1, x2, y2],
        })

    return dets


def _decode_yolo_v7_flat(
    names: List[str],
    outs: List[np.ndarray],
    *,
    img_meta: Optional[dict],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> Optional[List[dict]]:
    """Decode flattened YOLO outputs of shape (B,N,5+nc) or (N,5+nc).

    Unfortunately, there is no single convention across model zoos.
    Common cases:
      A) *Decoded* export: x,y,w,h are already in input pixel space (often xywh).
      B) *Head* export: x,y,w,h are still raw head values and require grid/stride/
         anchor decoding (but the three scales are already concatenated into one
         big (B,N,...) tensor).

    We try both interpretations and pick the one that yields the more plausible
    detections (based on simple heuristics).
    """
    cand = None
    for _, a in zip(names, outs):
        arr = np.asarray(a)
        if arr.ndim == 2 and arr.shape[1] >= 6:
            cand = arr[None, ...]  # add batch
            break
        if arr.ndim == 3 and arr.shape[-1] >= 6:
            cand = arr
            break
    if cand is None:
        return None

    pred = np.asarray(cand[0])  # visualise first batch only
    if pred.ndim != 2 or pred.shape[1] < 6:
        return None

    # input size (letterboxed)
    in_w, in_h = 640, 640
    if img_meta and "input_size" in img_meta:
        in_w, in_h = int(img_meta["input_size"][0]), int(img_meta["input_size"][1])

    def _nms_class_aware(box_xyxy: np.ndarray, scores: np.ndarray, cls_id: np.ndarray) -> List[dict]:
        dets_local: List[dict] = []
        for c in np.unique(cls_id):
            idx = np.where(cls_id == c)[0]
            keep = _nms_xyxy(box_xyxy[idx], scores[idx], float(iou_thres), int(max_det))
            for k in keep:
                i = int(idx[k])
                dets_local.append(
                    {
                        "cls_id": int(cls_id[i]),
                        "score": float(scores[i]),
                        "box_xyxy_in": [float(x) for x in box_xyxy[i].tolist()],
                    }
                )
        dets_local.sort(key=lambda d: d["score"], reverse=True)
        return dets_local[: int(max_det)]

    def _decode_assume_decoded() -> List[dict]:
        box = pred[:, 0:4].astype(np.float32)
        obj = pred[:, 4].astype(np.float32)
        cls_raw = pred[:, 5:].astype(np.float32)

        # Apply sigmoid if values look like logits
        if np.nanmax(obj) > 1.5 or np.nanmin(obj) < -0.1:
            obj2 = _sigmoid(obj)
        else:
            obj2 = obj
        if cls_raw.size:
            if np.nanmax(cls_raw) > 1.5 or np.nanmin(cls_raw) < -0.1:
                cls2 = _sigmoid(cls_raw)
            else:
                cls2 = cls_raw
        else:
            cls2 = cls_raw

        if cls2.size == 0:
            cls_id = np.zeros((box.shape[0],), dtype=np.int64)
            best_score = obj2.astype(np.float32)
        else:
            scores_all = obj2[:, None] * cls2
            cls_id = scores_all.argmax(axis=-1).astype(np.int64)
            best_score = scores_all.max(axis=-1).astype(np.float32)

        mask = best_score > float(conf_thres)
        if not np.any(mask):
            return []
        box = box[mask]
        best_score = best_score[mask]
        cls_id = cls_id[mask]

        # Heuristic: decide between xyxy vs xywh
        is_xyxy = (np.mean(box[:, 2] >= box[:, 0]) > 0.99) and (np.mean(box[:, 3] >= box[:, 1]) > 0.99)
        if is_xyxy:
            box_xyxy = box
        else:
            x, y, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
            x1 = x - w / 2.0
            y1 = y - h / 2.0
            x2 = x + w / 2.0
            y2 = y + h / 2.0
            box_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        # If coordinates look normalised (rough heuristic), scale to input size
        if np.nanmax(box_xyxy) <= 2.0:
            box_xyxy[:, [0, 2]] *= float(in_w)
            box_xyxy[:, [1, 3]] *= float(in_h)

        # Clip to input space
        box_xyxy[:, 0] = np.clip(box_xyxy[:, 0], 0.0, float(in_w))
        box_xyxy[:, 2] = np.clip(box_xyxy[:, 2], 0.0, float(in_w))
        box_xyxy[:, 1] = np.clip(box_xyxy[:, 1], 0.0, float(in_h))
        box_xyxy[:, 3] = np.clip(box_xyxy[:, 3], 0.0, float(in_h))

        return _nms_class_aware(box_xyxy, best_score, cls_id)

    def _decode_assume_head() -> Optional[List[dict]]:
        # Expect concatenation of 3 YOLO scales (stride 8/16/32).
        N, C = int(pred.shape[0]), int(pred.shape[1])
        nc = C - 5
        if nc <= 0:
            return None

        # Compute expected per-scale sizes.
        desc = []  # (stride, gh, gw, n)
        for s in (8, 16, 32):
            if int(in_w) % int(s) != 0 or int(in_h) % int(s) != 0:
                continue
            gw = int(in_w) // int(s)
            gh = int(in_h) // int(s)
            desc.append((int(s), int(gh), int(gw), int(3 * gh * gw)))
        if not desc:
            return None
        if int(sum(d[3] for d in desc)) != int(N):
            return None

        anchor_map = {8: 0, 16: 1, 32: 2}

        best = None  # (score, dets)
        for order in (desc, list(reversed(desc))):
            for reshape_mode in ("hw3c", "3hwc"):
                off = 0
                boxes_all: List[np.ndarray] = []
                scores_all: List[np.ndarray] = []
                cls_all: List[np.ndarray] = []
                ok = True

                for (s, gh, gw, nseg) in order:
                    seg = pred[off : off + nseg]
                    off += nseg
                    try:
                        if reshape_mode == "hw3c":
                            yt = seg.reshape(gh, gw, 3, C).transpose(2, 0, 1, 3)
                        else:
                            yt = seg.reshape(3, gh, gw, C)
                    except Exception:
                        ok = False
                        break

                    raw = yt.astype(np.float32)

                    # YOLOv5/v7 head decoding.
                    xy = _sigmoid(raw[..., 0:2]) * 2.0 - 0.5
                    wh = (_sigmoid(raw[..., 2:4]) * 2.0) ** 2

                    # grid
                    gx, gy = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
                    grid = np.stack([gx, gy], axis=-1)[None, ...]  # (1,gh,gw,2)

                    xy = (xy + grid) * float(s)

                    ai = anchor_map.get(int(s), None)
                    if ai is None:
                        ok = False
                        break
                    anchors = YOLOV7_ANCHORS[ai].astype(np.float32).reshape(3, 1, 1, 2)
                    wh = wh * anchors

                    x1y1 = xy - wh / 2.0
                    x2y2 = xy + wh / 2.0
                    box_xyxy = np.concatenate([x1y1, x2y2], axis=-1).astype(np.float32)  # (3,gh,gw,4)

                    obj = _sigmoid(raw[..., 4:5])
                    cls = _sigmoid(raw[..., 5:])
                    sc = (obj * cls).reshape(-1, nc)
                    bs = sc.max(axis=-1).astype(np.float32)
                    cid = sc.argmax(axis=-1).astype(np.int64)

                    boxes_all.append(box_xyxy.reshape(-1, 4))
                    scores_all.append(bs)
                    cls_all.append(cid)

                if not ok:
                    continue

                boxes = np.concatenate(boxes_all, axis=0)
                scores = np.concatenate(scores_all, axis=0)
                cls_id = np.concatenate(cls_all, axis=0)

                # Filter + clip
                m = scores > float(conf_thres)
                if not np.any(m):
                    dets_local = []
                else:
                    boxes = boxes[m]
                    scores = scores[m]
                    cls_id = cls_id[m]
                    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(in_w))
                    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(in_w))
                    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(in_h))
                    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(in_h))
                    dets_local = _nms_class_aware(boxes, scores, cls_id)

                s = _plausibility_score_dets(dets_local, img_meta, max_det=int(max_det))
                if best is None or s > best[0]:
                    best = (float(s), dets_local)

        return None if best is None else best[1]

    dets_decoded = _decode_assume_decoded()
    dets_head = _decode_assume_head()

    s_dec = _plausibility_score_dets(dets_decoded, img_meta, max_det=int(max_det))
    s_head = _plausibility_score_dets(dets_head, img_meta, max_det=int(max_det)) if dets_head is not None else -1e9

    use_head = bool(dets_head is not None and s_head > s_dec)
    dets = dets_head if use_head else dets_decoded
    fmt = "flat-head" if use_head else "flat-decoded"
    print(f"[viz] YOLO flat decode selected: {fmt} (score head={s_head:.1f}, decoded={s_dec:.1f})")

    # Map boxes back to original image (undo letterbox), if meta available.
    if img_meta and img_meta.get("preset") == "yolo" and "scale" in img_meta and "pad" in img_meta:
        scale = float(img_meta.get("scale", 1.0))
        pad_w, pad_h = img_meta.get("pad", [0, 0])
        pad_w = float(pad_w)
        pad_h = float(pad_h)
        orig_w, orig_h = img_meta.get("orig_size", [in_w, in_h])
        orig_w = float(orig_w)
        orig_h = float(orig_h)

        for d in dets:
            x1, y1, x2, y2 = d["box_xyxy_in"]
            x1 = (x1 - pad_w) / scale
            x2 = (x2 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            y2 = (y2 - pad_h) / scale
            x1 = max(0.0, min(orig_w, x1))
            x2 = max(0.0, min(orig_w, x2))
            y1 = max(0.0, min(orig_h, y1))
            y2 = max(0.0, min(orig_h, y2))
            d["box_xyxy_orig"] = [x1, y1, x2, y2]
    else:
        for d in dets:
            d["box_xyxy_orig"] = d["box_xyxy_in"]

    for d in dets:
        cid = int(d["cls_id"])
        d["label"] = COCO80_NAMES[cid] if 0 <= cid < len(COCO80_NAMES) else f"cls{cid}"
    return dets


def _decode_yolo_auto(
    output_names: List[str],
    outputs: List[np.ndarray],
    *,
    img_meta: Optional[dict],
    conf_thres: float,
    iou_thres: float,
    max_det: int,
) -> Optional[List[dict]]:
    """Try multiple YOLO output conventions (multiscale head, then flat tensor)."""
    dets = _decode_yolo_v7_multiscale(
        output_names,
        outputs,
        img_meta=img_meta,
        conf_thres=float(conf_thres),
        iou_thres=float(iou_thres),
        max_det=int(max_det),
    )
    if dets is not None:
        print("[viz] YOLO output format: multiscale head (B,3,H,W,...) ")
        return dets
    print("[viz] YOLO output format: flat tensor (B,N,5+nc)")
    return _decode_yolo_v7_flat(
        output_names,
        outputs,
        img_meta=img_meta,
        conf_thres=float(conf_thres),
        iou_thres=float(iou_thres),
        max_det=int(max_det),
    )
def _draw_detections(image_path: str, dets: List[dict], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for d in dets:
        x1, y1, x2, y2 = d.get("box_xyxy_orig", d.get("box_xyxy_in", [0, 0, 0, 0]))
        score = float(d.get("score", 0.0))
        label = str(d.get("label", "obj"))
        text = f"{label} {score:.2f}"

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        # label
        tx, ty = x1 + 2, max(0.0, y1 - 12)
        if font is not None:
            draw.text((tx, ty), text, fill=(255, 0, 0), font=font)
        else:
            draw.text((tx, ty), text, fill=(255, 0, 0))

    img.save(out_path)
    print(f"Wrote {out_path}")




def _with_suffix(name: str, suffix: str) -> str:
    """Append suffix before file extension (if any)."""
    p = Path(name)
    if p.suffix:
        return p.with_name(p.stem + suffix + p.suffix).name
    return name + suffix


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)


def _load_labels(labels_file: Optional[str], out_dir: Path, expected_min: int = 10) -> Optional[List[str]]:
    candidates: List[Path] = []
    if labels_file:
        candidates.append(Path(labels_file))
    candidates.append(out_dir / "imagenet_labels.txt")
    candidates.append(Path(__file__).resolve().parent / "imagenet_labels.txt")

    for p in candidates:
        try:
            if p.exists():
                lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if len(lines) >= expected_min:
                    return lines
        except Exception:
            continue
    return None


def _pick_classification_vector(output_names: List[str], outputs: List[np.ndarray]) -> Optional[Tuple[str, np.ndarray]]:
    """Heuristic: pick the largest 1D/2D output as classification vector."""
    best: Optional[Tuple[str, np.ndarray]] = None
    best_c = -1
    for name, arr in zip(output_names, outputs):
        if not isinstance(arr, np.ndarray):
            continue
        vec: Optional[np.ndarray] = None
        if arr.ndim == 2 and arr.shape[1] >= 10:
            vec = arr[0]
        elif arr.ndim == 1 and arr.shape[0] >= 10:
            vec = arr
        if vec is None:
            continue
        c = int(vec.shape[0])
        if c > best_c:
            best_c = c
            best = (name, np.asarray(vec))
    return best


def _decode_classification_topk(
    output_names: List[str],
    outputs: List[np.ndarray],
    *,
    topk: int,
    out_dir: Path,
    labels_file: Optional[str],
) -> Optional[Dict[str, object]]:
    picked = _pick_classification_vector(output_names, outputs)
    if picked is None:
        return None
    name, vec = picked
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)

    # Decide whether vec looks like probabilities.
    s = float(np.sum(vec))
    is_prob = (vec.min() >= 0.0) and (vec.max() <= 1.0) and (abs(s - 1.0) < 0.05)
    probs = vec if is_prob else _softmax_1d(vec)

    k = max(1, int(topk))
    idxs = np.argsort(probs)[::-1][:k]

    labels = _load_labels(labels_file, out_dir, expected_min=max(10, int(probs.shape[0]) // 2))

    top: List[Dict[str, object]] = []
    for i in idxs:
        ii = int(i)
        label = labels[ii] if (labels is not None and 0 <= ii < len(labels)) else str(ii)
        top.append({"idx": ii, "label": label, "prob": float(probs[ii])})

    return {
        "type": "classification",
        "output": name,
        "num_classes": int(probs.shape[0]),
        "topk": top,
        "prob_vector_sum": float(np.sum(probs)),
        "used_softmax": (not is_prob),
    }


def _draw_classification_overlay(image_path: str, topk: List[Dict[str, object]], out_path: str, title: str = "") -> None:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return

    draw = ImageDraw.Draw(img)
    lines: List[str] = []
    if title:
        lines.append(title)
    for i, e in enumerate(topk):
        label = str(e.get("label", e.get("idx", "?")))
        prob = float(e.get("prob", 0.0))
        lines.append(f"{i+1}: {label} ({prob:.3f})")

    if not lines:
        img.save(out_path)
        return

    x0, y0 = 10, 10
    # crude text box sizing without font metrics
    max_len = max(len(l) for l in lines)
    box_w = 10 + 7 * max_len + 10
    box_h = 10 + 16 * len(lines) + 10

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(255, 255, 255))
    for i, l in enumerate(lines):
        draw.text((x0 + 10, y0 + 10 + i * 16), l, fill=(0, 0, 0))

    img.save(out_path)


def _plot_classification_topk(topk: List[Dict[str, object]], out_dir: Path, tag: str) -> Optional[Dict[str, str]]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    labels = [str(e.get("label", e.get("idx", "?"))) for e in topk]
    probs = [float(e.get("prob", 0.0)) for e in topk]

    fig = plt.figure(figsize=(6, 3))
    plt.bar(range(len(probs)), probs)
    plt.xticks(range(len(probs)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("probability")
    plt.title(f"Top-{len(probs)} predictions ({tag})")
    fig.tight_layout()

    png = out_dir / f"predictions_{tag}.png"
    pdf = out_dir / f"predictions_{tag}.pdf"
    fig.savefig(str(png), dpi=200)
    fig.savefig(str(pdf))
    plt.close(fig)

    return {"png": png.name, "pdf": pdf.name}


def _decode_dets_6col(
    output_names: List[str],
    outputs: List[np.ndarray],
    img_meta: Optional[dict],
    *,
    conf_thres: float,
    max_det: int,
) -> Optional[List[Dict[str, object]]]:
    """Fallback for models that already output Nx6 or Nx7 detections.

    Tries to parse an array shaped (B,N,6/7) or (N,6/7) as:
      - [x1,y1,x2,y2,score,cls] (6)
      - [batch,x1,y1,x2,y2,score,cls] (7)

    Returns list with keys compatible to YOLO drawing: cls_id, score, box_xyxy_orig.
    """

    cand: Optional[np.ndarray] = None
    for _, arr in zip(output_names, outputs):
        if not isinstance(arr, np.ndarray):
            continue
        a = arr
        if a.ndim == 3 and a.shape[0] >= 1 and a.shape[-1] in (6, 7):
            a = a[0]
        if a.ndim == 2 and a.shape[-1] in (6, 7) and a.shape[0] >= 1:
            if cand is None or a.shape[0] > cand.shape[0]:
                cand = a

    if cand is None:
        return None

    a = np.asarray(cand, dtype=np.float32)
    if a.shape[1] == 7:
        # Heuristic: first column looks like batch indices?
        bcol = a[:, 0]
        if np.all(np.isfinite(bcol)) and np.max(bcol) <= 32 and np.min(bcol) >= 0 and np.all(np.abs(bcol - np.round(bcol)) < 1e-3):
            a = a[:, 1:]
        else:
            # unknown 7-col layout
            return None

    # Now a is Nx6
    xyxy = a[:, 0:4]
    scores = a[:, 4]
    cls = a[:, 5]

    keep = scores > float(conf_thres)
    xyxy = xyxy[keep]
    scores = scores[keep]
    cls = cls[keep]

    if xyxy.shape[0] == 0:
        return []

    # Map coords (normalized/pixel) and undo letterbox to original.
    in_w, in_h = (img_meta.get("input_size") if img_meta else (None, None)) or (None, None)
    if in_w is None or in_h is None:
        # fall back: infer from max coord
        in_w = float(np.max(xyxy[:, [0, 2]]))
        in_h = float(np.max(xyxy[:, [1, 3]]))

    if float(np.max(xyxy)) <= 2.0:
        xyxy[:, [0, 2]] *= float(in_w)
        xyxy[:, [1, 3]] *= float(in_h)

    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, float(in_w))
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, float(in_w))
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, float(in_h))
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, float(in_h))

    scale = float((img_meta or {}).get("scale", 1.0))
    pad = (img_meta or {}).get("pad", [0, 0])
    pad_w, pad_h = float(pad[0]), float(pad[1])
    orig = (img_meta or {}).get("orig_size", [float(in_w), float(in_h)])
    orig_w, orig_h = float(orig[0]), float(orig[1])

    dets: List[Dict[str, object]] = []
    # take top max_det by score
    idxs = np.argsort(scores)[::-1][: int(max_det)]
    for i in idxs:
        x1, y1, x2, y2 = xyxy[i]
        x1 = (x1 - pad_w) / scale
        x2 = (x2 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        y2 = (y2 - pad_h) / scale
        x1 = float(np.clip(x1, 0, orig_w))
        x2 = float(np.clip(x2, 0, orig_w))
        y1 = float(np.clip(y1, 0, orig_h))
        y2 = float(np.clip(y2, 0, orig_h))
        dets.append({
            "cls_id": int(round(float(cls[i]))),
            "score": float(scores[i]),
            "box_xyxy_orig": [x1, y1, x2, y2],
        })

    return dets


def _maybe_visualize_outputs(
    *,
    preset: str,
    image_path: Optional[str],
    image_used: bool,
    img_meta: Optional[dict],
    output_names: List[str],
    outputs: List[np.ndarray],
    out_dir: Path,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    detections_out_base: str,
    detections_json_base: str,
    class_topk: int,
    labels_file: Optional[str],
    tag: str,
) -> Optional[Dict[str, object]]:
    """Try to create human-readable outputs (detection overlays / top-k classifications)."""

    if not image_used or image_path is None:
        return None

    # Prefer the actually-used preset if available.
    used_preset = ((img_meta or {}).get("preset") or preset or "auto").lower()

    # 1) Try YOLO-like decoding (multiscale, flat, or Nx6/Nx7 post-NMS).
    # Only attempt YOLO decoding when preset == 'yolo' (avoids false positives on non-detection models).
    dets = None
    if used_preset == "yolo":
        dets = _decode_yolo_auto(
            output_names,
            outputs,
            img_meta=img_meta,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )

        if dets is None:
            dets = _decode_dets_6col(
                output_names,
                outputs,
                img_meta,
                conf_thres=float(conf_thres),
                max_det=int(max_det),
            )
        if dets is None:
            dets = _decode_dets_separate(
                output_names,
                outputs,
                img_meta,
                conf_thres=float(conf_thres),
                max_det=int(max_det),
            )

        if dets is not None:
            # Use PIL-based drawing (no extra deps). Write JSON in a format the agreement KPI can read.
            out_img = out_dir / _with_suffix(detections_out_base, f"_{tag}")
            out_json = out_dir / _with_suffix(detections_json_base, f"_{tag}")

            try:
                _draw_detections(image_path, dets, out_img)
            except Exception as e:
                print(f"[viz] WARNING: drawing detections failed ({type(e).__name__}): {e}")

            try:
                out_json.write_text(
                    json.dumps(
                        {
                            "preset": used_preset,
                            "image": image_path,
                            "img_meta": img_meta,
                            "conf_thres": float(conf_thres),
                            "iou_thres": float(iou_thres),
                            "max_det": int(max_det),
                            "detections": dets,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception:
                pass

            print(f"Wrote {out_json.name} and {out_img.name}")
            return {
                "type": "detection",
                "tag": tag,
                "image": out_img.name,
                "json": out_json.name,
                "count": int(len(dets) if dets is not None else 0),
            }

    # 2) Try ImageNet-like classification.
    cls = _decode_classification_topk(
        output_names,
        outputs,
        topk=class_topk,
        out_dir=out_dir,
        labels_file=labels_file,
    )
    if cls is not None:
        out_img = out_dir / f"classification_{tag}.png"
        out_json = out_dir / f"classification_{tag}.json"
        _draw_classification_overlay(
            image_path,
            cls.get("topk", []),
            str(out_img),
            title=f"{tag}: top-{int(class_topk)}",
        )
        try:
            out_json.write_text(
                json.dumps({"preset": used_preset, "image": image_path, "img_meta": img_meta, **cls}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        plot = _plot_classification_topk(cls.get("topk", []), out_dir, tag=tag)
        if plot is not None:
            print(f"Wrote {plot.get('png')} and {plot.get('pdf')}")

        print(f"Wrote {out_json.name} and {out_img.name}")
        return {
            "type": "classification",
            "tag": tag,
            "image": out_img.name,
            "json": out_json.name,
            "plot": plot,
            "output": cls.get("output"),
            "num_classes": cls.get("num_classes"),
            "used_softmax": cls.get("used_softmax"),
        }

    return None


def _maybe_write_plots(summary: Dict[str, object], out_dir: Path):
    """Write a compact PNG/PDF report.

    The report is meant to give quick visual feedback instead of only console logs:
      - timing summary (full/part1/part2/composed)
      - output diff summary (top outputs), or a clear "all-zero" message

    If matplotlib is not installed, we silently skip plot generation.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not installed -> skipping validation_report.png/pdf")
        return

    # ---- Extract timing ----
    timing = summary.get("timing_ms", {})
    if not isinstance(timing, dict):
        timing = {}

    def _get_mean_std(key: str) -> Tuple[float, float]:
        d = timing.get(key, {})
        if not isinstance(d, dict):
            return (float("nan"), 0.0)
        return (float(d.get("mean", float("nan"))), float(d.get("std", 0.0)))

    full_m, full_s = _get_mean_std("full")
    p1_m, p1_s = _get_mean_std("part1")
    p2_m, p2_s = _get_mean_std("part2")
    comp_m, comp_s = _get_mean_std("composed")

    derived = summary.get("derived", {})
    if not isinstance(derived, dict):
        derived = {}
    p1p2_m = float(derived.get("part1_plus_part2_mean", (p1_m + p2_m)))
    overhead_split = float(derived.get("composed_minus_part1_part2_mean", (comp_m - p1p2_m)))
    overhead_full = float(derived.get("composed_minus_full_mean", (comp_m - full_m)))

    ok = bool(summary.get("ok", False))
    eps = float(summary.get("eps", 0.0))

    # ---- Extract output diffs ----
    cmp = summary.get("output_compare", {})
    if not isinstance(cmp, dict):
        cmp = {}
    per_output = cmp.get("per_output", [])
    if not isinstance(per_output, list):
        per_output = []

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    # (A) Timing plot
    ax_t = fig.add_subplot(gs[0, :])
    labels = ["full", "part1", "part2", "part1+part2", "composed"]
    means = [full_m, p1_m, p2_m, p1p2_m, comp_m]
    stds = [full_s, p1_s, p2_s, 0.0, comp_s]
    ax_t.bar(range(len(labels)), means, yerr=stds, capsize=4)
    ax_t.set_xticks(range(len(labels)))
    ax_t.set_xticklabels(labels)
    ax_t.set_ylabel("ms")
    ax_t.set_title("Timing summary (mean ± std)")

    txt_box = (
        f"composed - (part1+part2) = {overhead_split:.3f} ms\n"
        f"composed - full          = {overhead_full:.3f} ms\n"
        f"PASS(eps={eps:g}) = {ok}"
    )
    ax_t.text(0.01, 0.98, txt_box, transform=ax_t.transAxes, va="top")

    # (B) Output diff plots (or a clear all-zero message)
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    per = list(per_output)
    if not per:
        ax1.axis("off")
        ax2.axis("off")
        ax1.text(0.02, 0.9, "No comparable outputs found.", transform=ax1.transAxes, va="top")
    else:
        per_sorted = sorted(per, key=lambda x: float(x.get("max_abs", 0.0)), reverse=True)
        top = per_sorted[: min(20, len(per_sorted))]
        out_labels = [str(x.get("name", "?")) for x in top]
        maxv = [float(x.get("max_abs", 0.0)) for x in top]
        meanv = [float(x.get("mean_abs", 0.0)) for x in top]

        if max(maxv + [0.0]) == 0.0 and max(meanv + [0.0]) == 0.0:
            ax1.axis("off")
            ax2.axis("off")
            msg = (
                "All compared outputs are bit-identical:\n"
                "max|diff| = 0 for all outputs."
            )
            ax1.text(0.02, 0.98, msg, transform=ax1.transAxes, va="top")
            # Show a compact output list (helps sanity-check which outputs were compared)
            lines = []
            for x in top:
                nm = str(x.get("name", "?"))
                shp = x.get("shape", [])
                lines.append(f"{nm}: {shp}")
            ax1.text(
                0.02,
                0.70,
                "Outputs:\n" + "\n".join(lines),
                transform=ax1.transAxes,
                va="top",
                family="monospace",
                fontsize=9,
            )
        else:
            ax1.bar(range(len(out_labels)), maxv)
            ax1.set_title("Per-output max |diff| (top)")
            ax1.set_xticks(range(len(out_labels)))
            ax1.set_xticklabels(out_labels, rotation=60, ha="right", fontsize=8)
            ax1.set_ylabel("max abs")
            ax1.set_ylim(bottom=0.0)

            ax2.bar(range(len(out_labels)), meanv)
            ax2.set_title("Per-output mean |diff| (top)")
            ax2.set_xticks(range(len(out_labels)))
            ax2.set_xticklabels(out_labels, rotation=60, ha="right", fontsize=8)
            ax2.set_ylabel("mean abs")
            ax2.set_ylim(bottom=0.0)

    fig.suptitle(f"Split validation report: {'PASS' if ok else 'FAIL'}", y=0.99)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_report.png", dpi=200)
    fig.savefig(out_dir / "validation_report.pdf")
    plt.close(fig)
    print(f"Wrote {out_dir / 'validation_report.png'} and .pdf")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="__MANIFEST_FILENAME__")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--warmup", type=int, default=5, help="Warmup runs (not measured).")
    ap.add_argument("--runs", type=int, default=10, help="Measured runs.")
    ap.add_argument("--eps", type=float, default=1e-4, help="Max-abs tolerance for PASS/FAIL.")

    ap.add_argument(
        "--provider",
        type=str,
        default="__DEFAULT_PROVIDER__",
        choices=["auto", "cpu", "cuda", "tensorrt"],
        help="Preferred onnxruntime execution provider (falls back if unavailable).",
    )

    ap.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Directory to store reports and visualizations (relative to split folder unless absolute).",
    )
    ap.add_argument(
        "--pause",
        action="store_true",
        help="Wait for Enter at end (useful when double-clicking).",
    )

    ap.add_argument("--image", type=str, default=None, help="Optional image file to use as input.")
    ap.add_argument(
        "--preset",
        type=str,
        default="auto",
        choices=["auto", "imagenet", "yolo", "raw"],
        help="Image preprocessing preset.",
    )
    ap.add_argument(
        "--yolo-scale",
        type=str,
        default="auto",
        choices=["auto", "norm", "raw"],
        help=("YOLO input scaling: norm=divide by 255, raw=0..255. "
              "auto probes both (requires --image and a YOLO-like output)."),
    )
    ap.add_argument(
        "--image-input",
        type=str,
        default=None,
        help="Name of the ONNX input to feed the image into (default: first non-initializer input).",
    )

    # Optional: if the model looks like YOLOv7, decode and draw detections for visual sanity-checking.
    ap.add_argument("--conf-thres", type=float, default=0.25, help="YOLO confidence threshold (visualization).")
    ap.add_argument("--iou-thres", type=float, default=0.45, help="YOLO IoU threshold for NMS (visualization).")
    ap.add_argument("--max-det", type=int, default=200, help="Max detections to draw (visualization).")
    ap.add_argument("--detections-out", type=str, default="detections.png", help="Output image with drawn detections.")
    ap.add_argument("--detections-json", type=str, default="detections.json", help="Output detections as JSON.")

    # Optional: for ImageNet-like classification models, print/plot top-k predictions.
    ap.add_argument("--class-topk", type=int, default=5, help="Classification top-k (visualization).")
    ap.add_argument("--labels-file", type=str, default=None, help="Optional labels file (one label per line) for classification visualization.")

    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    with manifest_path.open("r", encoding="utf-8") as f:
        m = json.load(f)

    # Resolve to an absolute directory to avoid subtle path-joining quirks
    # when the manifest path is relative (e.g. "split_manifest.json").
    base_dir = manifest_path.parent.resolve()

    def _looks_like_windows_abs(s: str) -> bool:
        s = (s or "").strip()
        return len(s) >= 3 and s[1] == ":" and (s[2] == "\\" or s[2] == "/")

    def _resolve_path(p: str) -> Path:
        """Resolve a manifest path.

        The benchmark/split folders are often created on Windows and then copied to
        Linux devices (Jetson, Raspberry Pi). In that case the manifest may still contain
        Windows absolute paths. We try to recover gracefully by falling back to the basename
        inside the current folder (or its parent).
        """
        if p is None:
            return base_dir

        ps = os.path.expandvars(os.path.expanduser(str(p)))
        ps_norm = ps.replace("\\", "/")

        # Windows absolute path stored in manifest, but now running on a different OS.
        if _looks_like_windows_abs(ps_norm):
            if os.path.exists(ps):
                return Path(ps)
            bn = os.path.basename(ps_norm)
            cand = base_dir / bn
            if cand.exists():
                return cand
            cand2 = base_dir.parent / bn
            if cand2.exists():
                return cand2
            return Path(ps_norm)

        pp = Path(ps_norm)
        return pp if pp.is_absolute() else (base_dir / pp)

    full_path = _resolve_path(m["full_model"])
    p1_path = _resolve_path(m["part1"])
    p2_path = _resolve_path(m["part2"])

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_model = onnx.load(str(full_path))
    has_nms = any((n.op_type == "NonMaxSuppression") for n in full_model.graph.node)
    print(f"[info] full graph contains NonMaxSuppression: {has_nms}")
    feeds_full = _make_random_inputs(full_model, batch=args.batch, seed=args.seed)
    img_meta = None
    image_used = False
    image_input_name = None
    image_input_shape = None
    image_input_elem_type = None

    img_path = args.image
    if img_path is None or str(img_path).strip().lower() in ("default", "auto", ""):
        default_img = base_dir / "test_image.png"
        if default_img.exists():
            img_path = str(default_img)
            print(f"Using default test image: {img_path}")
        else:
            img_path = None

    if img_path is not None:
        inputs = _get_non_initializer_inputs(full_model)
        if inputs:
            image_in = None
            if args.image_input:
                for vi in inputs:
                    if vi.name == args.image_input:
                        image_in = vi
                        break
                if image_in is None:
                    print(f"WARNING: --image-input '{args.image_input}' not found; using first input.")
            if image_in is None:
                image_in = inputs[0]

            try:
                dims = _dims_from_vi(image_in, args.batch)
                elem_type = int(image_in.type.tensor_type.elem_type)
                ys = args.yolo_scale
                if (ys or "").lower() == "auto":
                    ys = "norm"
                tensor, img_meta = _image_to_tensor(
                    img_path,
                    target_shape=dims,
                    elem_type=elem_type,
                    preset=args.preset,
                    yolo_scale=ys,
                    batch=args.batch,
                    return_meta=True,
                )
                feeds_full[image_in.name] = tensor
                image_input_name = image_in.name
                image_input_shape = dims
                image_input_elem_type = elem_type
                image_used = True
                print(
                    f"Image fed into input: {image_in.name} "
                    f"shape={feeds_full[image_in.name].shape} dtype={feeds_full[image_in.name].dtype}"
                )
            except Exception as e:
                print(f"WARNING: Could not use image input ({type(e).__name__}): {e}")
        else:
            print("WARNING: No non-initializer inputs found to feed an image into.")

    providers = pick_providers(args.provider)
    print(f"ORT available providers: {ort.get_available_providers()}")
    print(f"Using providers: {providers}")

    sess_full = ort.InferenceSession(str(full_path), providers=providers)
    sess_p1 = ort.InferenceSession(str(p1_path), providers=providers)
    sess_p2 = ort.InferenceSession(str(p2_path), providers=providers)



    # Auto-select YOLO input scaling if requested and the used preset resolved to YOLO.
    if image_used and (args.yolo_scale or "").lower() == "auto" and (img_meta or {}).get("preset") == "yolo":
        try:
            out_names_full = [o.name for o in sess_full.get_outputs()]
            best = None  # (score, yolo_scale, tensor, meta)

            print("[auto] Probing YOLO input scaling (norm vs raw)...")
            for ys in ["norm", "raw"]:
                try:
                    tens, meta_c = _image_to_tensor(
                        img_path,
                        target_shape=image_input_shape,
                        elem_type=image_input_elem_type,
                        preset="yolo",
                        yolo_scale=ys,
                        batch=args.batch,
                        return_meta=True,
                    )
                    fd = dict(feeds_full)
                    fd[image_input_name] = tens
                    outs = sess_full.run(None, fd)
                    dets = _decode_yolo_auto(
                        out_names_full,
                        outs,
                        img_meta=meta_c,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        max_det=args.max_det,
                    )
                    if dets is None:
                        dets = _decode_dets_6col(out_names_full, outs, meta_c, conf_thres=args.conf_thres, max_det=args.max_det)
                    if dets is None:
                        dets = _decode_dets_separate(out_names_full, outs, meta_c, conf_thres=args.conf_thres, max_det=args.max_det)
                    s = _plausibility_score_dets(dets, meta_c, max_det=int(args.max_det))
                    print(f"  yolo_scale={ys}: {0 if not dets else len(dets)} dets, plausibility={s:.1f}")
                    if dets and (best is None or s > best[0]):
                        best = (float(s), ys, tens, meta_c)
                except Exception as e:
                    print(f"  yolo_scale={ys}: failed ({type(e).__name__}: {e})")

            if best is not None:
                _, ys, tens, meta_c = best
                feeds_full[image_input_name] = tens
                img_meta = meta_c
                print(f"[auto] Selected yolo_scale={ys}.")
        except Exception as e:
            print(f"[auto] YOLO scale probing failed: {type(e).__name__}: {e}")

    p1_in = {i.name for i in sess_p1.get_inputs()}
    feeds_p1 = {k: v for k, v in feeds_full.items() if k in p1_in}

    def build_feeds_p2(p1_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        feeds_p2: Dict[str, np.ndarray] = {}
        for p1n, p2n in zip(m.get("part1_cut_names", []), m.get("part2_cut_names", [])):
            feeds_p2[p2n] = p1_map[p1n]
        for inp in sess_p2.get_inputs():
            if inp.name in feeds_p2:
                continue
            if inp.name in feeds_full:
                feeds_p2[inp.name] = feeds_full[inp.name]
            else:
                raise RuntimeError(f"Missing part2 input '{inp.name}' (split not self-contained?)")
        return feeds_p2

    def run_full():
        return sess_full.run(None, feeds_full)

    def run_p1() -> Dict[str, np.ndarray]:
        out = sess_p1.run(None, feeds_p1)
        names = [o.name for o in sess_p1.get_outputs()]
        return dict(zip(names, out))

    p1_map0 = run_p1()
    feeds_p2_0 = build_feeds_p2(p1_map0)

    def run_p2_only():
        return sess_p2.run(None, feeds_p2_0)

    def run_composed():
        p1_map = run_p1()
        feeds_p2 = build_feeds_p2(p1_map)
        return sess_p2.run(None, feeds_p2)

    full_mean, full_std, _, full_out = _bench("full", run_full, args.warmup, args.runs)
    p1_mean, p1_std, _, _ = _bench("part1", lambda: list(run_p1().values()), args.warmup, args.runs)
    p2_mean, p2_std, _, _ = _bench("part2", run_p2_only, args.warmup, args.runs)
    comp_mean, comp_std, _, comp_out = _bench("composed", run_composed, args.warmup, args.runs)

    full_names = [o.name for o in sess_full.get_outputs()]
    p2_names = [o.name for o in sess_p2.get_outputs()]
    cmp = _compare_outputs(full_names, full_out, p2_names, comp_out)

    max_abs = float(cmp["max_abs"])
    ok = max_abs <= float(args.eps)

    summary = {
        "timing_ms": {
            "full": {"mean": full_mean, "std": full_std},
            "part1": {"mean": p1_mean, "std": p1_std},
            "part2": {"mean": p2_mean, "std": p2_std},
            "composed": {"mean": comp_mean, "std": comp_std},
        },
        "derived": {
            "part1_plus_part2_mean": float(p1_mean + p2_mean),
            "composed_minus_full_mean": float(comp_mean - full_mean),
            "composed_minus_part1_part2_mean": float(comp_mean - (p1_mean + p2_mean)),
        },
        "runs": {"warmup": int(args.warmup), "measured": int(args.runs)},
        "output_compare": cmp,
        "eps": float(args.eps),
        "ok": bool(ok),
    }

    print("\n==== Timing summary (ms) ====")
    print(f"full     : {full_mean:.3f} ± {full_std:.3f}")
    print(f"part1    : {p1_mean:.3f} ± {p1_std:.3f}")
    print(f"part2    : {p2_mean:.3f} ± {p2_std:.3f}")
    print(f"composed : {comp_mean:.3f} ± {comp_std:.3f}")
    print(f"(note) part1+part2 (means) = {(p1_mean+p2_mean):.3f} ms")
    print(f"(note) composed - (part1+part2) = {(comp_mean-(p1_mean+p2_mean)):.3f} ms")
    print(f"(note) composed - full       = {(comp_mean-full_mean):.3f} ms")
    print("\n==== Output diff ====")
    print(f"Compared outputs: {cmp.get('n_outputs_compared', 0)}")
    print(f"max_abs : {max_abs:.6g}")
    print(f"mean_abs: {float(cmp.get('mean_abs', 0.0)):.6g}")
    print(f"PASS({args.eps}): {ok}")

    report_path = out_dir / "validation_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")

    _maybe_write_plots(summary, out_dir)

    viz_all: Dict[str, dict] = {}

    viz_full = _maybe_visualize_outputs(
        preset=args.preset,
        image_path=img_path,
        image_used=image_used,
        img_meta=img_meta,
        output_names=full_names,
        outputs=full_out,
        out_dir=out_dir,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        detections_out_base=args.detections_out,
        detections_json_base=args.detections_json,
        class_topk=args.class_topk,
        labels_file=args.labels_file,
        tag="full",
    )
    if viz_full is not None:
        viz_all["full"] = viz_full

    viz_comp = _maybe_visualize_outputs(
        preset=args.preset,
        image_path=img_path,
        image_used=image_used,
        img_meta=img_meta,
        output_names=full_names,
        outputs=comp_out,
        out_dir=out_dir,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        detections_out_base=args.detections_out,
        detections_json_base=args.detections_json,
        class_topk=args.class_topk,
        labels_file=args.labels_file,
        tag="composed",
    )
    if viz_comp is not None:
        viz_all["composed"] = viz_comp


    # Agreement KPIs (semantic agreement full vs composed)
    # These are meant as an *additional* sanity check beyond max-abs eps:
    # - Classification: top-1 / top-5 agreement
    # - Detection: class-aware IoU matching + precision/recall/F1
    try:
        agree = None
        if viz_full is not None and viz_comp is not None and viz_full.get('type') == viz_comp.get('type'):
            t = str(viz_full.get('type') or '')
            jf = out_dir / str(viz_full.get('json') or '')
            jc = out_dir / str(viz_comp.get('json') or '')
            if jf.exists() and jc.exists():
                def _readj(p: Path):
                    try:
                        return json.loads(p.read_text(encoding='utf-8'))
                    except Exception:
                        return None

                def _mean(xs: List[float]) -> float:
                    return float(sum(xs) / len(xs)) if xs else float('nan')

                def _iou_xyxy(a: List[float], b: List[float]) -> float:
                    x1 = max(float(a[0]), float(b[0]))
                    y1 = max(float(a[1]), float(b[1]))
                    x2 = min(float(a[2]), float(b[2]))
                    y2 = min(float(a[3]), float(b[3]))
                    iw = max(0.0, x2 - x1)
                    ih = max(0.0, y2 - y1)
                    inter = iw * ih
                    if inter <= 0.0:
                        return 0.0
                    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
                    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
                    union = area_a + area_b - inter
                    return float(inter / union) if union > 0.0 else 0.0

                djf = _readj(jf)
                djc = _readj(jc)
                if isinstance(djf, dict) and isinstance(djc, dict):
                    if t == 'classification':
                        topf = djf.get('topk') or []
                        topc = djc.get('topk') or []

                        def _idxs(tl, k: int) -> List[int]:
                            out: List[int] = []
                            for e in (tl or [])[: int(k)]:
                                try:
                                    out.append(int(e.get('idx')))
                                except Exception:
                                    pass
                            return out

                        i1f = _idxs(topf, 1)
                        i1c = _idxs(topc, 1)
                        top1_equal = bool(i1f and i1c and i1f[0] == i1c[0])

                        s_f = set(_idxs(topf, 5))
                        s_c = set(_idxs(topc, 5))
                        inter = len(s_f & s_c)
                        union = len(s_f | s_c)

                        agree = {
                            'type': 'classification',
                            'top1_equal': (top1_equal if (i1f and i1c) else None),
                            'top1_idx_full': (i1f[0] if i1f else None),
                            'top1_idx_comp': (i1c[0] if i1c else None),
                            'top5_overlap': (int(inter) if union > 0 else None),
                            'top5_jaccard': (float(inter / union) if union > 0 else None),
                        }

                    elif t == 'detection':
                        full_d = djf.get('detections') or []
                        comp_d = djc.get('detections') or []

                        # Class-aware greedy matching by IoU.
                        # NOTE: This is *not* mAP and not against GT; it only checks if split preserves detections.
                        thr_match = 0.50

                        full_sorted = sorted(full_d, key=lambda d: float(d.get('score', 0.0)), reverse=True)
                        comp_sorted = sorted(comp_d, key=lambda d: float(d.get('score', 0.0)), reverse=True)

                        used = set()
                        ious: List[float] = []
                        for fd in full_sorted:
                            try:
                                fcls = int(fd.get('cls_id', -1))
                                fbox = fd.get('box_xyxy_orig')
                                if not isinstance(fbox, (list, tuple)) or len(fbox) != 4:
                                    continue
                                fbox_f = [float(x) for x in fbox]
                            except Exception:
                                continue

                            best_j = None
                            best_iou = 0.0
                            for j, cd in enumerate(comp_sorted):
                                if j in used:
                                    continue
                                try:
                                    if int(cd.get('cls_id', -2)) != fcls:
                                        continue
                                    cbox = cd.get('box_xyxy_orig')
                                    if not isinstance(cbox, (list, tuple)) or len(cbox) != 4:
                                        continue
                                    cbox_f = [float(x) for x in cbox]
                                except Exception:
                                    continue
                                iou = _iou_xyxy(fbox_f, cbox_f)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_j = j

                            if best_j is not None and best_iou >= thr_match:
                                used.add(best_j)
                                ious.append(float(best_iou))

                        n_full = int(len(full_sorted))
                        n_comp = int(len(comp_sorted))
                        n_match = int(len(ious))

                        prec = (float(n_match) / float(n_comp)) if n_comp > 0 else (1.0 if n_full == 0 else 0.0)
                        rec = (float(n_match) / float(n_full)) if n_full > 0 else (1.0 if n_comp == 0 else 0.0)
                        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
                        mean_iou = float(_mean(ious)) if ious else (1.0 if (n_full == 0 and n_comp == 0) else 0.0)

                        agree = {
                            'type': 'detection',
                            'n_full': n_full,
                            'n_comp': n_comp,
                            'n_matched': n_match,
                            'precision': float(prec),
                            'recall': float(rec),
                            'f1': float(f1),
                            'mean_iou': float(mean_iou),
                            'match_iou_thr': float(thr_match),
                        }

        if agree is not None:
            summary['agreement'] = agree
            report_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(f"Updated {report_path} with agreement KPIs.")
    except Exception as e:
        print(f"[agreement] failed: {e}")

    # Optional: create side-by-side comparisons (full vs composed)
    try:
        if viz_full is not None and viz_comp is not None and viz_full.get('type') == viz_comp.get('type'):
            vis_cmp = {}
            if viz_full.get('type') == 'detection':
                a = viz_full.get('image')
                b = viz_comp.get('image')
                if a and b and os.path.exists(str(out_dir / a)) and os.path.exists(str(out_dir / b)):
                    from PIL import Image, ImageDraw
                    im1 = Image.open(str(out_dir / a)).convert('RGB')
                    im2 = Image.open(str(out_dir / b)).convert('RGB')
                    W = im1.width + im2.width
                    H = max(im1.height, im2.height)
                    canvas = Image.new('RGB', (W, H), (255,255,255))
                    canvas.paste(im1, (0,0))
                    canvas.paste(im2, (im1.width,0))
                    # small labels
                    dr = ImageDraw.Draw(canvas)
                    dr.text((5,5), 'full', fill=(0,0,0))
                    dr.text((im1.width+5,5), 'composed', fill=(0,0,0))
                    outp = str(out_dir / 'detections_compare.png')
                    canvas.save(outp)
                    vis_cmp['detections_compare'] = Path(outp).name
            elif viz_full.get('type') == 'classification':
                a = viz_full.get('image')
                b = viz_comp.get('image')
                if a and b and os.path.exists(str(out_dir / a)) and os.path.exists(str(out_dir / b)):
                    from PIL import Image, ImageDraw
                    im1 = Image.open(str(out_dir / a)).convert('RGB')
                    im2 = Image.open(str(out_dir / b)).convert('RGB')
                    W = im1.width + im2.width
                    H = max(im1.height, im2.height)
                    canvas = Image.new('RGB', (W, H), (255,255,255))
                    canvas.paste(im1, (0,0))
                    canvas.paste(im2, (im1.width,0))
                    dr = ImageDraw.Draw(canvas)
                    dr.text((5,5), 'full', fill=(0,0,0))
                    dr.text((im1.width+5,5), 'composed', fill=(0,0,0))
                    outp = str(out_dir / 'classification_compare.png')
                    canvas.save(outp)
                    vis_cmp['classification_compare'] = Path(outp).name
            if vis_cmp:
                summary['visualization_compare'] = vis_cmp
    except Exception as e:
        print(f"[viz] compare creation failed: {e}")

    if viz_all:
        summary["visualization"] = viz_all
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Updated {report_path} with visualization info.")

    if args.pause:
        try:
            input("Press Enter to exit...")
        except Exception:
            pass

    return 0


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
