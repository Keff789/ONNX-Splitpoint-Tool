from __future__ import annotations

"""Static sanity checks for exported Hailo Part2 ONNX graphs.

These checks are intentionally conservative. They do *not* try to prove that a
Part2 graph is Hailo-compatible; they only catch high-confidence structural
problems early so the benchmark generator can skip obviously bad splits before
spending minutes in the Hailo compiler.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        iv = int(value)
        if iv <= 0:
            return None
        return iv
    except Exception:
        return None


def _normalize_shape(shape: Sequence[Any]) -> Optional[Tuple[Optional[int], ...]]:
    dims: List[Optional[int]] = []
    try:
        values = list(shape or [])
    except Exception:
        return None
    if not values:
        return None
    for dim in values:
        dims.append(_safe_int(dim))
    return tuple(dims)


@dataclass(frozen=True)
class Part2ConcatMismatch:
    node_name: str
    axis: int
    input_names: Tuple[str, ...]
    input_shapes: Tuple[Tuple[Optional[int], ...], ...]
    rank_mismatch: bool
    mismatched_dims: Tuple[int, ...]


_HEAD_PREFIXES: Tuple[str, ...] = (
    "/model.22/",
    "/model.23/",
)


def _value_info_shape_map(model: Any) -> Dict[str, Tuple[Optional[int], ...]]:
    out: Dict[str, Tuple[Optional[int], ...]] = {}
    try:
        graph = model.graph
    except Exception:
        return out

    def _extract(value_info: Any) -> Optional[Tuple[Optional[int], ...]]:
        try:
            tensor_type = value_info.type.tensor_type
            shape = tensor_type.shape.dim
        except Exception:
            return None
        dims: List[Optional[int]] = []
        for dim in list(shape or []):
            val: Optional[int] = None
            try:
                if dim.HasField("dim_value"):
                    val = _safe_int(dim.dim_value)
                elif dim.HasField("dim_param"):
                    val = None
            except Exception:
                val = None
            dims.append(val)
        return tuple(dims) if dims else None

    for coll_name in ("input", "output", "value_info"):
        try:
            coll = list(getattr(graph, coll_name) or [])
        except Exception:
            coll = []
        for value_info in coll:
            try:
                name = str(value_info.name or "").strip()
            except Exception:
                name = ""
            if not name or name in out:
                continue
            shp = _extract(value_info)
            if shp is not None:
                out[name] = shp
    return out


def _infer_shapes_if_possible(model: Any) -> Any:
    try:
        import onnx  # type: ignore
    except Exception:
        return model
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        return model


def _node_is_head_relevant(node_name: str, inputs: Sequence[str], outputs: Sequence[str]) -> bool:
    node_name = str(node_name or "").strip()
    names = [node_name, *[str(x or "").strip() for x in list(inputs or [])], *[str(x or "").strip() for x in list(outputs or [])]]
    return any(name.startswith(_HEAD_PREFIXES) for name in names if name)


def detect_part2_concat_mismatches(model: Any, *, head_only: bool = True) -> List[Part2ConcatMismatch]:
    inferred = _infer_shapes_if_possible(model)
    shape_map = _value_info_shape_map(inferred)
    try:
        nodes = list(inferred.graph.node or [])
    except Exception:
        return []

    mismatches: List[Part2ConcatMismatch] = []
    for node in nodes:
        try:
            if str(node.op_type or "") != "Concat":
                continue
        except Exception:
            continue
        node_name = str(getattr(node, "name", "") or "").strip() or "<unnamed_concat>"
        input_names = [str(x or "").strip() for x in list(getattr(node, "input", []) or []) if str(x or "").strip()]
        output_names = [str(x or "").strip() for x in list(getattr(node, "output", []) or []) if str(x or "").strip()]
        if head_only and not _node_is_head_relevant(node_name, input_names, output_names):
            continue

        axis = 0
        try:
            for attr in list(getattr(node, "attribute", []) or []):
                if str(getattr(attr, "name", "") or "") == "axis":
                    axis = int(getattr(attr, "i", 0))
                    break
        except Exception:
            axis = 0

        shapes: List[Tuple[Optional[int], ...]] = []
        for name in input_names:
            shp = shape_map.get(name)
            if shp is not None:
                shapes.append(shp)
        if len(shapes) < 2:
            continue

        ranks = {len(shp) for shp in shapes}
        rank_mismatch = len(ranks) > 1
        if rank_mismatch:
            mismatches.append(
                Part2ConcatMismatch(
                    node_name=node_name,
                    axis=int(axis),
                    input_names=tuple(input_names),
                    input_shapes=tuple(shapes),
                    rank_mismatch=True,
                    mismatched_dims=tuple(),
                )
            )
            continue

        rank = len(shapes[0])
        if rank <= 0:
            continue
        axis_norm = axis if axis >= 0 else rank + axis
        if axis_norm < 0 or axis_norm >= rank:
            mismatches.append(
                Part2ConcatMismatch(
                    node_name=node_name,
                    axis=int(axis),
                    input_names=tuple(input_names),
                    input_shapes=tuple(shapes),
                    rank_mismatch=False,
                    mismatched_dims=tuple(range(rank)),
                )
            )
            continue

        mismatched_dims: List[int] = []
        for dim_idx in range(rank):
            if dim_idx == axis_norm:
                continue
            known = {int(v) for shp in shapes for v in [shp[dim_idx]] if v is not None}
            if len(known) >= 2:
                mismatched_dims.append(dim_idx)
        if mismatched_dims:
            mismatches.append(
                Part2ConcatMismatch(
                    node_name=node_name,
                    axis=int(axis_norm),
                    input_names=tuple(input_names),
                    input_shapes=tuple(shapes),
                    rank_mismatch=False,
                    mismatched_dims=tuple(mismatched_dims),
                )
            )
    return mismatches


def hailo_part2_concat_sanity_from_model(model: Any, *, split_manifest: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    try:
        mismatches = detect_part2_concat_mismatches(model, head_only=True)
    except Exception as exc:
        return {
            "inspect_ok": False,
            "compatible": None,
            "reason": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        }

    if not mismatches:
        return {
            "inspect_ok": True,
            "compatible": True,
            "reason": None,
            "mismatch_count": 0,
        }

    first = mismatches[0]
    info: Dict[str, Any] = {
        "inspect_ok": True,
        "compatible": False,
        "reason": "concat_shape_mismatch",
        "mismatch_count": int(len(mismatches)),
        "node_name": str(first.node_name),
        "axis": int(first.axis),
        "rank_mismatch": bool(first.rank_mismatch),
        "mismatched_dims": list(first.mismatched_dims),
        "input_names": list(first.input_names),
        "input_shapes": [list(shp) for shp in first.input_shapes],
        "mismatches": [
            {
                "node_name": str(m.node_name),
                "axis": int(m.axis),
                "rank_mismatch": bool(m.rank_mismatch),
                "mismatched_dims": list(m.mismatched_dims),
                "input_names": list(m.input_names),
                "input_shapes": [list(shp) for shp in m.input_shapes],
            }
            for m in mismatches[:8]
        ],
    }
    if isinstance(split_manifest, Mapping):
        try:
            outputs = list(split_manifest.get("part2_outputs_effective") or split_manifest.get("part2_outputs") or [])
        except Exception:
            outputs = []
        if outputs:
            info["part2_outputs"] = [str(x) for x in outputs if str(x).strip()]
    return info


def format_hailo_part2_concat_sanity_error(info: Mapping[str, Any]) -> str:
    node_name = str(info.get("node_name") or "<unknown_concat>")
    axis = info.get("axis")
    input_shapes = list(info.get("input_shapes") or [])
    mismatched_dims = list(info.get("mismatched_dims") or [])
    rank_mismatch = bool(info.get("rank_mismatch"))
    msg = f"Hailo Part2 concat sanity failed at {node_name}"
    if axis not in (None, ""):
        msg += f" (axis={axis})"
    if rank_mismatch:
        msg += ": input ranks differ"
    elif mismatched_dims:
        msg += f": non-concat dims differ at {mismatched_dims}"
    else:
        msg += ": incompatible concat input shapes"
    if input_shapes:
        msg += f" | input_shapes={input_shapes}"
    outputs = list(info.get("part2_outputs") or [])
    if outputs:
        msg += f" | part2_outputs={outputs}"
    return msg
