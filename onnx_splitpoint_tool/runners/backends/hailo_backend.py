from __future__ import annotations

from ...hailo_attempt_receipts import start_hailo_attempt

import base64
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from .base import PreparedHandle
from .hailo_utils import get_dfc_manager
from .._types import BackendCaps, BackendRunOut, RunCfg
from ...cache_verify_policy import (
    cache_miss_blocked_message,
    compiler_dispatch_forbidden,
)
from ...hailo_timeout_policy import parse_hailo_timeout_seconds

_RESULT_MARKER = "__SPLITPOINT_HAILO_RESULT__"


def _hailo_layer_slot(name: str) -> Optional[int]:
    lname = str(name or '').strip().lower()
    if not lname:
        return None
    import re
    m = re.search(r'(?:^|[/:._-])(?:input|output)_layer[_-]?(\d+)(?:$|[/:._-])', lname)
    if not m:
        return None
    try:
        idx = int(m.group(1)) - 1
    except Exception:
        return None
    return idx if idx >= 0 else None


def _hailo_stream_sort_key(name: str, default_index: int = 0) -> tuple[int, int, int, str]:
    slot = _hailo_layer_slot(name)
    if slot is not None:
        return (0, int(slot), int(default_index), str(name))
    return (1, int(default_index), 0, str(name))


def _hailo_stream_names_ordered(infos: list[Any]) -> list[str]:
    indexed: list[tuple[int, str]] = []
    for idx, info in enumerate(infos):
        indexed.append((int(idx), str(getattr(info, 'name', ''))))
    indexed.sort(key=lambda item: _hailo_stream_sort_key(item[1], item[0]))
    return [name for _, name in indexed]


def _onnx_value_info_shape(value_info: Any) -> Optional[tuple[int, ...]]:
    try:
        tensor_type = getattr(getattr(value_info, "type", None), "tensor_type", None)
        shape = getattr(tensor_type, "shape", None)
        dims = getattr(shape, "dim", None)
        if dims is None:
            return None
        out: list[int] = []
        for dim in dims:
            dim_value = getattr(dim, "dim_value", None)
            if isinstance(dim_value, int) and dim_value > 0:
                out.append(int(dim_value))
            else:
                return None
        return tuple(out)
    except Exception:
        return None


def _load_onnx_io_names_and_shapes(model_path: Optional[Path]) -> tuple[list[str], list[str], dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    if model_path is None:
        return [], [], {}, {}
    try:
        import onnx  # type: ignore

        model = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return [], [], {}, {}

    initializer_names = {str(getattr(t, "name", "") or "") for t in getattr(model.graph, "initializer", [])}

    input_names: list[str] = []
    output_names: list[str] = []
    input_shapes: dict[str, tuple[int, ...]] = {}
    output_shapes: dict[str, tuple[int, ...]] = {}

    for value_info in getattr(model.graph, "input", []):
        name = str(getattr(value_info, "name", "") or "")
        if not name or name in initializer_names:
            continue
        input_names.append(name)
        shape = _onnx_value_info_shape(value_info)
        if shape is not None:
            input_shapes[name] = shape

    for value_info in getattr(model.graph, "output", []):
        name = str(getattr(value_info, "name", "") or "")
        if not name:
            continue
        output_names.append(name)
        shape = _onnx_value_info_shape(value_info)
        if shape is not None:
            output_shapes[name] = shape

    return input_names, output_names, input_shapes, output_shapes


def _shape_numel_ignoring_batch(shape: Optional[tuple[int, ...]]) -> Optional[int]:
    if not shape:
        return None
    dims = [int(x) for x in shape if isinstance(x, int) and int(x) > 0]
    if len(dims) > 1 and dims[0] == 1:
        dims = dims[1:]
    if not dims:
        return None
    total = 1
    for dim in dims:
        total *= int(dim)
    return int(total)


def _canonical_slot_name_order(
    onnx_names: list[str],
    preferred_names: Optional[list[str]] = None,
) -> list[str]:
    """Return canonical slot order.

    For split part2 Hailo HEFs generated from part2 ONNX, generic input_layerN
    follows the exported part2 ONNX input order. Callers should therefore avoid
    overriding that order with manifest cut-name permutations unless they know
    the HEF was built with that exact slot contract.
    """
    onnx_list = [str(x) for x in (onnx_names or []) if str(x)]
    preferred_list = [str(x) for x in (preferred_names or []) if str(x)]
    if not onnx_list:
        out: list[str] = []
        seen: set[str] = set()
        for name in preferred_list:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out
    if not preferred_list:
        return list(onnx_list)
    out: list[str] = []
    seen: set[str] = set()
    for name in preferred_list:
        if name not in onnx_list or name in seen:
            continue
        seen.add(name)
        out.append(name)
    for name in onnx_list:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _build_hailo_io_name_map(
    hailo_names: list[str],
    onnx_names: list[str],
    *,
    hailo_shapes: Optional[dict[str, tuple[int, ...]]] = None,
    onnx_shapes: Optional[dict[str, tuple[int, ...]]] = None,
    slot_names: Optional[list[str]] = None,
) -> dict[str, str]:
    """Best-effort alias map from HEF stream names to source-ONNX IO names.

    Why:
      Hailo often rewrites split-model IO names to generic stream names such as
      ``input_layer1`` / ``output_layer1``. The split benchmark logic, however,
      reasons about the original ONNX boundary tensor names. Re-exposing HEF
      streams under the source-ONNX IO names keeps stage1 -> stage2 wiring
      deterministic and lets the existing manifest-aware mapping logic do exact
      matches again.
    """
    mapping: dict[str, str] = {}
    used_onnx: set[str] = set()

    hailo_list = [str(x) for x in hailo_names or []]
    onnx_list = _canonical_slot_name_order([str(x) for x in onnx_names or []], slot_names)

    # 1) Exact-name matches first.
    for name in hailo_list:
        if name in onnx_list and name not in used_onnx:
            mapping[name] = name
            used_onnx.add(name)

    # 2) Unique element-count matches (ignoring a leading batch dim).
    remaining_hailo = [name for name in hailo_list if name not in mapping]
    remaining_onnx = [name for name in onnx_list if name not in used_onnx]
    if hailo_shapes and onnx_shapes and remaining_hailo and remaining_onnx:
        hailo_by_numel: dict[int, list[str]] = {}
        onnx_by_numel: dict[int, list[str]] = {}
        for name in remaining_hailo:
            numel = _shape_numel_ignoring_batch(hailo_shapes.get(name))
            if numel is not None:
                hailo_by_numel.setdefault(int(numel), []).append(name)
        for name in remaining_onnx:
            numel = _shape_numel_ignoring_batch(onnx_shapes.get(name))
            if numel is not None:
                onnx_by_numel.setdefault(int(numel), []).append(name)
        for numel, hailo_group in hailo_by_numel.items():
            onnx_group = onnx_by_numel.get(int(numel)) or []
            if len(hailo_group) == 1 and len(onnx_group) == 1:
                h_name = hailo_group[0]
                o_name = onnx_group[0]
                if h_name not in mapping and o_name not in used_onnx:
                    mapping[h_name] = o_name
                    used_onnx.add(o_name)

    # 3) Generic ``input_layerN`` / ``output_layerN`` slot names map by slot.
    remaining_hailo = [name for name in hailo_list if name not in mapping]
    remaining_onnx = [name for name in onnx_list if name not in used_onnx]
    for name in remaining_hailo:
        slot = _hailo_layer_slot(name)
        if slot is None or slot < 0 or slot >= len(onnx_list):
            continue
        target = onnx_list[int(slot)]
        if target not in remaining_onnx or target in used_onnx:
            continue
        mapping[name] = target
        used_onnx.add(target)

    # 4) Conservative positional fallback when counts still line up.
    remaining_hailo = [name for name in hailo_list if name not in mapping]
    remaining_onnx = [name for name in onnx_list if name not in used_onnx]
    if remaining_hailo and len(remaining_hailo) == len(remaining_onnx):
        for h_name, o_name in zip(remaining_hailo, remaining_onnx):
            if o_name in used_onnx:
                continue
            mapping[h_name] = o_name
            used_onnx.add(o_name)

    return mapping


_HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "output": (1, 3, 80, 80, 85),
    "clone_1": (1, 3, 40, 40, 85),
    "clone_2": (1, 3, 20, 20, 85),
}


def _validated_attested_source_output_shapes(
    value: Any,
) -> dict[str, tuple[int, ...]]:
    """Validate the deliberately narrow Hailo-8 YOLOv7 output exception.

    The normal Hailo mapper remains best-effort for legacy callers.  This
    separate option is only issued after the runner has verified the
    hash-sealed Source-ONNX attestation, so accepting any other name or shape
    here would turn that attestation into a generic positional reshape escape.
    """

    if value in (None, {}):
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError("attested_source_output_shapes must be an object")
    normalized: dict[str, tuple[int, ...]] = {}
    for raw_name, raw_shape in value.items():
        name = str(raw_name or "").strip()
        if (
            not name
            or name in normalized
            or not isinstance(raw_shape, (list, tuple))
            or not raw_shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in raw_shape
            )
        ):
            raise RuntimeError("attested_source_output_shapes is malformed")
        normalized[name] = tuple(int(dim) for dim in raw_shape)
    if (
        list(normalized) != list(_HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES)
        or normalized != _HAILO8_YOLOV7_ATTESTED_OUTPUT_SHAPES
    ):
        raise RuntimeError(
            "attested_source_output_shapes is not the sealed Hailo-8 YOLOv7 contract"
        )
    return normalized


def _attested_output_shape_compatible(
    runtime_shape: tuple[int, ...], canonical_shape: tuple[int, ...],
) -> bool:
    """Return whether one physical Hailo shape has one permitted layout."""

    physical = tuple(int(dim) for dim in runtime_shape)
    canonical = tuple(int(dim) for dim in canonical_shape)
    if len(canonical) != 5 or canonical[0] != 1:
        return False
    _batch, anchors, height, width, channels = canonical
    packed_channels = int(anchors) * int(channels)
    return physical in {
        canonical,
        (height, width, packed_channels),
        (1, height, width, packed_channels),
    }


def _build_attested_hailo_output_map(
    hailo_names: list[str],
    canonical_shapes: Mapping[str, tuple[int, ...]],
    hailo_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, str]:
    """Map attested raw heads by exact shape semantics, never by position."""

    canonical = _validated_attested_source_output_shapes(canonical_shapes)
    physical_names = [str(name) for name in hailo_names]
    if len(physical_names) != len(canonical) or len(set(physical_names)) != len(physical_names):
        raise RuntimeError("attested Hailo output count/name mismatch")
    if set(hailo_shapes) != set(physical_names):
        raise RuntimeError("attested Hailo output shape metadata mismatch")

    mapping: dict[str, str] = {}
    used: set[str] = set()
    for physical_name in physical_names:
        shape = tuple(int(dim) for dim in hailo_shapes[physical_name])
        candidates = [
            canonical_name
            for canonical_name, canonical_shape in canonical.items()
            if canonical_name not in used
            and _attested_output_shape_compatible(shape, canonical_shape)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                "attested Hailo output mapping is missing or ambiguous"
            )
        mapping[physical_name] = candidates[0]
        used.add(candidates[0])
    if used != set(canonical):
        raise RuntimeError("attested Hailo output mapping is incomplete")
    return mapping


def _adapt_attested_hailo_output_tensor(
    value: np.ndarray, target_shape: tuple[int, ...],
) -> np.ndarray:
    """Apply only layouts explicitly admitted by the sealed raw-head contract."""

    array = np.asarray(value)
    target = tuple(int(dim) for dim in target_shape)
    if not _attested_output_shape_compatible(tuple(array.shape), target):
        raise RuntimeError(
            f"attested Hailo output shape mismatch: {tuple(array.shape)} != {target}"
        )
    if tuple(array.shape) == target:
        return np.ascontiguousarray(array)
    _batch, anchors, height, width, channels = target
    if tuple(array.shape) == (1, height, width, anchors * channels):
        array = array[0]
    # The compatibility check above leaves exactly packed HWC at this point.
    return np.ascontiguousarray(
        array.reshape(height, width, anchors, channels)
        .transpose(2, 0, 1, 3)[None, ...]
    )


def _reconcile_attested_source_output_metadata(
    onnx_names: list[str],
    onnx_shapes: Mapping[str, tuple[int, ...]],
    attested_shapes: Mapping[str, tuple[int, ...]],
) -> tuple[list[str], dict[str, tuple[int, ...]]]:
    """Use sealed metadata only when optional ONNX metadata is absent/exact."""

    attested = _validated_attested_source_output_shapes(attested_shapes)
    if not attested:
        return list(onnx_names), dict(onnx_shapes)
    names = [str(name) for name in onnx_names]
    if names:
        if names != list(attested):
            raise RuntimeError("loaded ONNX output names conflict with attestation")
        if set(onnx_shapes) != set(attested) or any(
            tuple(onnx_shapes[name]) != attested[name] for name in attested
        ):
            raise RuntimeError("loaded ONNX output shapes conflict with attestation")
    elif onnx_shapes:
        raise RuntimeError("loaded ONNX output metadata is internally incomplete")
    return list(attested), dict(attested)


def _order_attested_output_mapping(
    outputs: Mapping[str, np.ndarray],
    attested_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    """Emit the canonical Source-ONNX order after strict name/shape mapping."""

    attested = _validated_attested_source_output_shapes(attested_shapes)
    if not attested:
        return dict(outputs)
    if set(outputs) != set(attested):
        raise RuntimeError("attested Hailo canonical output set is incomplete")
    return {name: outputs[name] for name in attested}


def _ensure_c_contiguous_cached(cache: dict[str, np.ndarray], key: str, arr: np.ndarray) -> np.ndarray:
    """Return a writable, C-contiguous tensor for the Hailo runtime.

    ``InferVStreams.infer`` may acquire a writable view of its input.  Arrays
    from immutable buffers can already be C-contiguous while still having
    ``WRITEABLE=False``; those must be staged just like strided arrays.
    """
    a = np.asarray(arr)
    if a.flags.c_contiguous and a.flags.writeable and a.flags.owndata:
        return a

    buf = cache.get(key)
    if (
        buf is None
        or buf.shape != a.shape
        or buf.dtype != a.dtype
        or not buf.flags.c_contiguous
        or not buf.flags.writeable
    ):
        buf = np.empty(a.shape, dtype=a.dtype, order="C")
        cache[key] = buf
    np.copyto(buf, a)
    return buf


def _owned_writable_c_buffer(arr: np.ndarray) -> np.ndarray:
    """Materialize an independent buffer accepted by Hailo VStreams.

    Pillow-backed arrays and batch/transpose views can be C-contiguous and even
    writable while still not owning their storage.  HailoRT may request a
    writable native view from the object passed to ``InferVStreams.infer``.
    Make the ownership boundary explicit immediately before that call.
    """
    out = np.array(np.asarray(arr), copy=True, order="C", subok=False)
    if not out.flags.c_contiguous or not out.flags.writeable or not out.flags.owndata:
        raise RuntimeError("failed to stage an owned writable C-contiguous Hailo input")
    return out


def _ensure_frames_dim(x: np.ndarray) -> np.ndarray:
    """Ensure a leading frames dimension for Hailo InferVStreams."""
    arr = np.asarray(x)
    if arr.ndim == 3:
        return arr[None, ...]
    return arr


HAILO_LAYOUT_CONTRACT_REVISION = "nwc_ncw_exact_v282"


def _adapt_tensor(x: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Best-effort tensor adaptation between canonical ONNX and Hailo runtime layouts.

    This covers the common NCHW <-> NHWC / CHW <-> HWC cases and also the
    packed YOLO-head layouts that Hailo frequently exposes as ``H x W x (na*ch)``
    while the ONNX-side canonical layout is ``1 x na x H x W x ch``.
    """
    arr = np.asarray(x)
    tgt = tuple(int(v) for v in target_shape)

    if tuple(arr.shape) == tgt:
        return arr

    # Remove / add a leading batch or frames dimension.
    if arr.ndim >= 1 and arr.shape[0] == 1 and tuple(arr.shape[1:]) == tgt:
        return arr[0]
    if arr.ndim + 1 == len(tgt) and len(tgt) >= 1 and tgt[0] == 1 and tuple(arr.shape) == tgt[1:]:
        return arr[None, ...]

    # Rank-3 NWC <-> NCW with the singleton batch squeezed by HailoRT.
    # For example, output [1, 8400, 84] reaches this adapter as [8400, 84]
    # while the source ONNX declares [1, 84, 8400].  Equal element count
    # does not permit a reshape here: the two non-batch axes must be swapped.
    # Admit only that exact reversed-axis shape contract.  Same-shape / plain
    # singleton cases above retain their order (including equal axis sizes).
    if (
        arr.ndim == 2 and len(tgt) == 3 and tgt[0] == 1
        and tuple(arr.shape) == (tgt[2], tgt[1])
    ):
        return np.transpose(arr, (1, 0))[None, ...]
    if (
        arr.ndim == 3 and arr.shape[0] == 1 and len(tgt) == 2
        and tuple(arr.shape[1:]) == (tgt[1], tgt[0])
    ):
        return np.transpose(arr[0], (1, 0))
    if (
        arr.ndim == 3 and len(tgt) == 3 and arr.shape[0] == tgt[0] == 1
        and tuple(arr.shape[1:]) == (tgt[2], tgt[1])
    ):
        return np.transpose(arr, (0, 2, 1))

    # Packed YOLO head: HWC/NHWC -> canonical 1xA xH xW xC.
    if arr.ndim == 3 and len(tgt) == 5 and tgt[0] == 1:
        gh, gw, flat = (int(v) for v in arr.shape)
        _, na, tgt_h, tgt_w, ch = tgt
        if gh == tgt_h and gw == tgt_w and flat == int(na) * int(ch):
            return arr.reshape(gh, gw, int(na), int(ch)).transpose(2, 0, 1, 3)[None, ...]
    if arr.ndim == 4 and len(tgt) == 5 and tgt[0] == 1 and arr.shape[0] == 1:
        _, gh, gw, flat = (int(v) for v in arr.shape)
        _, na, tgt_h, tgt_w, ch = tgt
        if gh == tgt_h and gw == tgt_w and flat == int(na) * int(ch):
            return arr.reshape(1, gh, gw, int(na), int(ch)).transpose(0, 3, 1, 2, 4)
    if arr.ndim == 4 and len(tgt) == 3:
        # [na, H, W, ch] -> [H, W, na*ch]
        na, gh, gw, ch = (int(v) for v in arr.shape)
        if tgt == (gh, gw, na * ch):
            return arr.transpose(1, 2, 0, 3).reshape(tgt)
    if arr.ndim == 5 and len(tgt) == 3 and arr.shape[0] == 1:
        # [1, na, H, W, ch] -> [H, W, na*ch]
        _, na, gh, gw, ch = (int(v) for v in arr.shape)
        if tgt == (gh, gw, na * ch):
            return arr.transpose(0, 2, 3, 1, 4).reshape(tgt)
    if arr.ndim == 5 and len(tgt) == 4 and arr.shape[0] == 1 and tgt[0] == 1:
        # [1, na, H, W, ch] -> [1, H, W, na*ch]
        _, na, gh, gw, ch = (int(v) for v in arr.shape)
        if tgt == (1, gh, gw, na * ch):
            return arr.transpose(0, 2, 3, 1, 4).reshape(tgt)
    if arr.ndim == 3 and len(tgt) == 4 and tgt[0] == 1:
        # [H, W, na*ch] -> [1, H, W, na*ch]
        if tuple(arr.shape) == tuple(tgt[1:]):
            return arr[None, ...]

    # Generic CHW <-> HWC / NCHW <-> NHWC permutations.
    if arr.ndim == 3 and len(tgt) == 3:
        if tuple(arr.shape) == (tgt[2], tgt[0], tgt[1]):
            return np.transpose(arr, (1, 2, 0))
        if tuple(arr.shape) == (tgt[1], tgt[2], tgt[0]):
            return np.transpose(arr, (2, 0, 1))

    if arr.ndim == 4 and len(tgt) == 4:
        if tuple(arr.shape) == (tgt[0], tgt[3], tgt[1], tgt[2]):
            return np.transpose(arr, (0, 2, 3, 1))
        if tuple(arr.shape) == (tgt[0], tgt[2], tgt[3], tgt[1]):
            return np.transpose(arr, (0, 3, 1, 2))

    # 1xCHW -> HWC
    if arr.ndim == 4 and arr.shape[0] == 1 and len(tgt) == 3 and tuple(arr.shape[1:]) == (tgt[2], tgt[0], tgt[1]):
        return np.transpose(arr[0], (1, 2, 0))
    # HWC -> 1xCHW
    if arr.ndim == 3 and len(tgt) == 4 and tgt[0] == 1 and tuple(arr.shape) == (tgt[2], tgt[3], tgt[1]):
        return np.transpose(arr, (2, 0, 1))[None, ...]

    # As a last resort, only reshape when the element count matches.
    try:
        if int(np.prod(arr.shape)) == int(np.prod(tgt)):
            return np.reshape(arr, tgt)
    except Exception:
        pass

    raise ValueError(f"Cannot adapt tensor from shape {tuple(arr.shape)} to target shape {tgt}")


def _try_adapt_tensor(x: np.ndarray, target_shape: Optional[tuple[int, ...]]) -> tuple[bool, np.ndarray]:
    arr = np.asarray(x)
    if target_shape is None:
        return True, arr
    try:
        return True, _adapt_tensor(arr, target_shape)
    except Exception:
        return False, arr


def _import_hailo_module() -> Any:
    """Import Hailo Python bindings.

    Hailo packages have used different top-level module names over time.
    Prefer `hailo_platform` when available, and fall back to `hailort`.
    """
    try:
        import hailo_platform as hpf  # type: ignore
        return hpf
    except Exception as exc_platform:
        try:
            import hailort as hpf  # type: ignore
            return hpf
        except Exception as exc_hailort:
            raise RuntimeError(
                "Hailo runtime is unavailable: cannot import 'hailo_platform' or 'hailort'. "
                "Install HailoRT Python bindings in the runtime environment."
            ) from exc_hailort


def _format_type_by_name(hpf: Any, name: str):
    fmt = getattr(hpf, "FormatType", None)
    if fmt is None:
        return None
    return getattr(fmt, str(name).upper(), None)


def _np_dtype_from_hailo_format_name(name: str) -> Any:
    value = str(name or "").strip().lower()
    if "uint16" in value:
        return np.uint16
    if "uint8" in value:
        return np.uint8
    if "int16" in value:
        return np.int16
    if "int8" in value:
        return np.int8
    return np.float32


def _hailo_info_format_type_name(info: Any, fallback: str = "FLOAT32") -> str:
    try:
        fmt = getattr(info, "format", None)
        typ = getattr(fmt, "type", None)
        if typ is not None:
            return str(typ).split(".")[-1].upper()
    except Exception:
        pass
    return str(fallback).upper()


def _hailo_native_vstream_format_type_name(
    hef: Any,
    info: Any,
    *,
    direction: str,
    fallback: str = "",
) -> str:
    """Resolve the device-native integer type behind one Hailo vstream.

    InferModel ``set_format_type`` controls a host-side transform.  Reading only
    the vstream's currently exposed format can therefore select FLOAT32 even
    though the HEF stream itself is UINT8/UINT16.  Resolve the underlying stream
    when the installed HailoRT exposes that metadata, and use the vstream format
    only as a compatibility fallback.
    """

    supported = {"UINT8", "UINT16"}
    vstream_name = str(getattr(info, "name", "") or "")
    stream_info_method = (
        "get_input_stream_infos"
        if str(direction).lower() == "input"
        else "get_output_stream_infos"
    )
    low_level_available = all(
        callable(getattr(hef, method, None))
        for method in (
            "get_network_group_names",
            stream_info_method,
            "get_stream_names_from_vstream_name",
        )
    )
    # When the installed HailoRT exposes the low-level HEF contract, an
    # ambiguous/missing mapping must not silently fall back to a possibly
    # host-transformed vstream format.  That would only *look* native.
    if low_level_available:
        try:
            groups = [
                str(value) for value in list(hef.get_network_group_names())
            ]
            if len(groups) != 1 or not vstream_name:
                return str(fallback).upper()
            group = groups[0]
            low_level_infos = list(getattr(hef, stream_info_method)(group))
            by_name = {
                str(getattr(stream_info, "name", "") or ""): stream_info
                for stream_info in low_level_infos
            }
            stream_names = [
                str(value)
                for value in list(
                    hef.get_stream_names_from_vstream_name(vstream_name, group)
                )
            ]
            native_types = {
                _hailo_info_format_type_name(by_name[name], "")
                for name in stream_names
                if name in by_name
            }
            native_types.intersection_update(supported)
            if len(stream_names) == 1 and len(native_types) == 1:
                return next(iter(native_types))
        except Exception:
            return str(fallback).upper()
        return str(fallback).upper()

    # Compatibility path for older HailoRT versions that expose only vstream
    # metadata.  Current Hailo-10 releases take the strict branch above.
    exposed = _hailo_info_format_type_name(info, fallback)
    return exposed if exposed in supported else str(fallback).upper()


def _hailort_transform_utils(hpf: Any) -> Any:
    """Return HailoRT's exact host quantization helper or fail closed."""

    transform = getattr(hpf, "HailoRTTransformUtils", None)
    if transform is not None:
        return transform
    try:
        from hailo_platform.pyhailort.pyhailort import (  # type: ignore
            HailoRTTransformUtils,
        )
    except Exception as exc:
        raise RuntimeError(
            "HailoRTTransformUtils is required for HEF-native host I/O"
        ) from exc
    return HailoRTTransformUtils


def _single_hailo_quant_info(
    stream: Any,
    fallback_info: Any,
    *,
    label: str,
) -> Any:
    """Resolve the one exact QuantInfo attached to a non-NMS stream."""

    candidates: list[Any] = []
    try:
        raw = getattr(stream, "quant_infos", None)
        if raw is not None:
            candidates = list(raw)
    except Exception:
        candidates = []
    if not candidates:
        try:
            quant = getattr(fallback_info, "quant_info", None)
            if quant is not None:
                candidates = [quant]
        except Exception:
            candidates = []
    if len(candidates) != 1:
        raise RuntimeError(
            f"{label} requires exactly one HEF QuantInfo; got {len(candidates)}"
        )
    quant = candidates[0]
    scale = getattr(quant, "qp_scale", getattr(quant, "scale", None))
    zero_point = getattr(
        quant,
        "qp_zp",
        getattr(quant, "zero_point", getattr(quant, "zp", None)),
    )
    try:
        scale_value = float(scale)
        zero_point_value = float(zero_point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{label} HEF QuantInfo is invalid") from exc
    if (
        not math.isfinite(scale_value)
        or scale_value <= 0.0
        or not math.isfinite(zero_point_value)
    ):
        raise RuntimeError(f"{label} HEF QuantInfo is invalid")
    return quant


def _maybe_call(obj: Any, name: str, *args: Any) -> Any:
    fn = getattr(obj, name, None)
    if fn is None:
        raise AttributeError(name)
    return fn(*args)


def _option_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


@dataclass
class _HailoSession:
    hef_path: Path
    quantized_inputs: bool
    quantized_outputs: bool
    persistent_activation: bool = False
    onnx_model_path: Optional[Path] = None
    canonical_input_slot_names: Optional[list[str]] = None
    canonical_output_slot_names: Optional[list[str]] = None
    attested_source_output_shapes: Optional[dict[str, Any]] = None
    _hpf: Any = None
    _vdevice: Any = None
    _network_group: Any = None
    _network_group_params: Any = None
    _pipe: Any = None
    input_names: list[str] = None  # type: ignore[assignment]
    output_names: list[str] = None  # type: ignore[assignment]
    input_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]
    output_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]
    runtime_input_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]
    runtime_output_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]

    _activation_lock = threading.RLock()

    def __post_init__(self) -> None:
        self.input_names = []
        self.output_names = []
        if self.canonical_input_slot_names is None:
            self.canonical_input_slot_names = []
        if self.canonical_output_slot_names is None:
            self.canonical_output_slot_names = []
        self.attested_source_output_shapes = _validated_attested_source_output_shapes(
            self.attested_source_output_shapes
        )
        self.input_shapes = {}
        self.output_shapes = {}
        self.runtime_input_shapes = {}
        self.runtime_output_shapes = {}
        self._active_handle = None
        self._active_entered = False
        self._input_contig_cache: dict[str, np.ndarray] = {}
        self._hef_input_names: list[str] = []
        self._hef_output_names: list[str] = []
        self._hef_input_shapes: dict[str, tuple[int, ...]] = {}
        self._hef_output_shapes: dict[str, tuple[int, ...]] = {}
        self._input_name_hef_to_canonical: dict[str, str] = {}
        self._input_name_canonical_to_hef: dict[str, str] = {}
        self._output_name_hef_to_canonical: dict[str, str] = {}
        self._open()

    def _format_type(self, *, want_uint8: bool):
        fmt = getattr(self._hpf, "FormatType", None)
        if fmt is None:
            return None
        if want_uint8:
            return getattr(fmt, "UINT8", getattr(fmt, "AUTO", None))
        return getattr(fmt, "FLOAT32", getattr(fmt, "AUTO", None))

    def _open(self) -> None:
        hpf = _import_hailo_module()

        if not self.hef_path.exists() or self.hef_path.stat().st_size <= 0:
            raise RuntimeError(f"Invalid HEF file: {self.hef_path}")

        self._hpf = hpf
        hef = hpf.HEF(str(self.hef_path))
        self._vdevice = hpf.VDevice()
        if hasattr(self._vdevice, "__enter__"):
            self._vdevice.__enter__()

        cfg = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_groups = self._vdevice.configure(hef, cfg)
        if not network_groups:
            raise RuntimeError("No network groups returned by Hailo VDevice.configure")
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()

        in_infos = list(hef.get_input_vstream_infos())
        out_infos = list(hef.get_output_vstream_infos())
        self._hef_input_names = _hailo_stream_names_ordered(in_infos)
        self._hef_output_names = _hailo_stream_names_ordered(out_infos)
        self._hef_input_shapes = {str(x.name): tuple(x.shape) for x in in_infos}
        self._hef_output_shapes = {str(x.name): tuple(x.shape) for x in out_infos}

        onnx_input_names, onnx_output_names, onnx_input_shapes, onnx_output_shapes = _load_onnx_io_names_and_shapes(self.onnx_model_path)
        onnx_output_names, onnx_output_shapes = _reconcile_attested_source_output_metadata(
            onnx_output_names,
            onnx_output_shapes,
            self.attested_source_output_shapes,
        )
        input_aliases = _build_hailo_io_name_map(
            self._hef_input_names,
            onnx_input_names,
            hailo_shapes=self._hef_input_shapes,
            onnx_shapes=onnx_input_shapes,
            slot_names=self.canonical_input_slot_names,
        )
        if self.attested_source_output_shapes:
            output_aliases = _build_attested_hailo_output_map(
                self._hef_output_names,
                self.attested_source_output_shapes,
                self._hef_output_shapes,
            )
        else:
            output_aliases = _build_hailo_io_name_map(
                self._hef_output_names,
                onnx_output_names,
                hailo_shapes=self._hef_output_shapes,
                onnx_shapes=onnx_output_shapes,
                slot_names=self.canonical_output_slot_names,
            )

        self._input_name_hef_to_canonical = {name: str(input_aliases.get(name, name)) for name in self._hef_input_names}
        self._input_name_canonical_to_hef = {}
        for name in self._hef_input_names:
            canonical = self._input_name_hef_to_canonical.get(name, name)
            self._input_name_canonical_to_hef.setdefault(str(canonical), str(name))

        self._output_name_hef_to_canonical = {name: str(output_aliases.get(name, name)) for name in self._hef_output_names}

        self.input_names = [self._input_name_hef_to_canonical.get(name, name) for name in self._hef_input_names]
        self.output_names = [self._output_name_hef_to_canonical.get(name, name) for name in self._hef_output_names]
        if self.attested_source_output_shapes:
            self.output_names = list(self.attested_source_output_shapes)
        self.runtime_input_shapes = {
            self._input_name_hef_to_canonical.get(name, name): tuple(self._hef_input_shapes.get(name, ()))
            for name in self._hef_input_names
        }
        self.runtime_output_shapes = {
            self._output_name_hef_to_canonical.get(name, name): tuple(self._hef_output_shapes.get(name, ()))
            for name in self._hef_output_names
        }
        self.input_shapes = {
            self._input_name_hef_to_canonical.get(name, name): tuple(
                onnx_input_shapes.get(self._input_name_hef_to_canonical.get(name, name), self._hef_input_shapes.get(name, ()))
            )
            for name in self._hef_input_names
        }
        self.output_shapes = {
            self._output_name_hef_to_canonical.get(name, name): tuple(
                onnx_output_shapes.get(self._output_name_hef_to_canonical.get(name, name), self._hef_output_shapes.get(name, ()))
            )
            for name in self._hef_output_names
        }

        in_params = hpf.InputVStreamParams.make(
            self._network_group,
            quantized=bool(self.quantized_inputs),
            format_type=self._format_type(want_uint8=bool(self.quantized_inputs)) or hpf.FormatType.AUTO,
        )
        out_params = hpf.OutputVStreamParams.make(
            self._network_group,
            quantized=bool(self.quantized_outputs),
            format_type=self._format_type(want_uint8=bool(self.quantized_outputs)) or hpf.FormatType.AUTO,
        )
        self._pipe = hpf.InferVStreams(self._network_group, in_params, out_params)
        if hasattr(self._pipe, "__enter__"):
            self._pipe.__enter__()
        if self.persistent_activation:
            self._activate_once()

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._pipe is None or self._network_group is None:
            raise RuntimeError("Hailo session is closed")

        infer_inputs: Dict[str, np.ndarray] = {}
        for canonical_name in self.input_names:
            hef_name = self._input_name_canonical_to_hef.get(str(canonical_name), str(canonical_name))
            if canonical_name in inputs:
                arr = np.asarray(inputs[canonical_name])
            elif hef_name in inputs:
                arr = np.asarray(inputs[hef_name])
            else:
                raise KeyError(f"Missing required Hailo input '{canonical_name}'")

            runtime_shape = self.runtime_input_shapes.get(str(canonical_name))
            ok, adapted = _try_adapt_tensor(arr, runtime_shape)
            arr = adapted if ok else arr
            arr = _ensure_frames_dim(arr)
            # VStreams is stricter than NumPy's WRITEABLE flag alone.  Pass an
            # independent buffer at the actual native-call boundary so a
            # foreign base object or a reused view can never trigger
            # ``ValueError: array is not writeable`` inside pyhailort.
            infer_inputs[hef_name] = _owned_writable_c_buffer(arr)

        if self.persistent_activation and self._active_handle is not None:
            out_raw = self._pipe.infer(infer_inputs)
        else:
            with _HailoSession._activation_lock:
                activation = self._network_group.activate(self._network_group_params)
                entered = False
                try:
                    if hasattr(activation, "__enter__"):
                        activation.__enter__()
                        entered = True
                    out_raw = self._pipe.infer(infer_inputs)
                finally:
                    try:
                        if entered and hasattr(activation, "__exit__"):
                            activation.__exit__(None, None, None)
                        elif hasattr(activation, "release"):
                            activation.release()  # type: ignore[attr-defined]
                        elif hasattr(activation, "close"):
                            activation.close()  # type: ignore[attr-defined]
                    except Exception:
                        pass

        raw_outputs: dict[str, np.ndarray] = {}
        for key, value in out_raw.items():
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.shape[0] == 1:
                try:
                    arr = np.squeeze(arr, axis=0)
                except Exception:
                    pass
            raw_outputs[str(key)] = arr

        ordered_raw_names = [name for name in self._hef_output_names if name in raw_outputs]
        ordered_raw_names.extend([name for name in raw_outputs.keys() if name not in ordered_raw_names])

        out: dict[str, np.ndarray] = {}
        for raw_name in ordered_raw_names:
            canonical_name = self._output_name_hef_to_canonical.get(str(raw_name), str(raw_name))
            arr = raw_outputs[raw_name]
            canonical_shape = self.output_shapes.get(str(canonical_name))
            attested_shapes = getattr(
                self, "attested_source_output_shapes", None,
            ) or {}
            if str(canonical_name) in attested_shapes:
                out[str(canonical_name)] = _adapt_attested_hailo_output_tensor(
                    arr, attested_shapes[str(canonical_name)]
                )
            else:
                ok, adapted = _try_adapt_tensor(arr, canonical_shape)
                out[str(canonical_name)] = adapted if ok else arr
        return _order_attested_output_mapping(
            out, getattr(self, "attested_source_output_shapes", None) or {},
        )

    def _canonical_outputs_from_slot_buffers(
        self, raw_buffers: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Expose async slot outputs under the same contract as ``infer``."""
        outputs: dict[str, np.ndarray] = {}
        for raw_name in self._hef_output_names:
            arr = np.asarray(raw_buffers[str(raw_name)])
            if arr.ndim > 0 and arr.shape[0] == 1:
                try:
                    arr = np.squeeze(arr, axis=0)
                except Exception:
                    pass
            canonical_name = self._output_name_hef_to_canonical.get(
                str(raw_name), str(raw_name),
            )
            canonical_shape = self.output_shapes.get(str(canonical_name))
            attested_shapes = getattr(
                self, "attested_source_output_shapes", None,
            ) or {}
            if str(canonical_name) in attested_shapes:
                outputs[str(canonical_name)] = _adapt_attested_hailo_output_tensor(
                    arr, attested_shapes[str(canonical_name)]
                )
            else:
                ok, adapted = _try_adapt_tensor(arr, canonical_shape)
                outputs[str(canonical_name)] = adapted if ok else arr
        return _order_attested_output_mapping(
            outputs, getattr(self, "attested_source_output_shapes", None) or {},
        )

    def _activate_once(self) -> None:
        if not self.persistent_activation or self._network_group is None or self._network_group_params is None:
            return
        if self._active_handle is not None:
            return
        activation = self._network_group.activate(self._network_group_params)
        entered = False
        if hasattr(activation, "__enter__"):
            activation.__enter__()
            entered = True
        self._active_handle = activation
        self._active_entered = bool(entered)

    def _release_persistent_activation(self) -> None:
        activation = self._active_handle
        if activation is None:
            return
        try:
            if self._active_entered and hasattr(activation, "__exit__"):
                activation.__exit__(None, None, None)
            elif hasattr(activation, "release"):
                activation.release()  # type: ignore[attr-defined]
            elif hasattr(activation, "close"):
                activation.close()  # type: ignore[attr-defined]
        finally:
            self._active_handle = None
            self._active_entered = False

    def close(self) -> None:
        try:
            self._release_persistent_activation()
        except Exception:
            pass
        try:
            if self._pipe is not None and hasattr(self._pipe, "__exit__"):
                self._pipe.__exit__(None, None, None)
        finally:
            self._pipe = None
        try:
            if self._vdevice is not None and hasattr(self._vdevice, "__exit__"):
                self._vdevice.__exit__(None, None, None)
        finally:
            self._vdevice = None


@dataclass
class _HailoInferModelSession:
    """HailoRT 5.x InferModel session used by Hailo-10/Hailo-15 devices.

    Hailo-10H returns HAILO_NOT_IMPLEMENTED for the legacy
    VDevice.configure/InferVStreams path. The newer API mirrors
    `hailortcli run2`: create_infer_model -> configure -> create_bindings ->
    run_async.
    """

    hef_path: Path
    quantized_inputs: bool
    quantized_outputs: bool
    persistent_activation: bool = False
    onnx_model_path: Optional[Path] = None
    canonical_input_slot_names: Optional[list[str]] = None
    canonical_output_slot_names: Optional[list[str]] = None
    attested_source_output_shapes: Optional[dict[str, Any]] = None
    batch_size: int = 1
    timeout_ms: int = 10000
    scheduler_group_id: str = "SHARED"
    scheduler_priority: int = 0
    hotloop: bool = True
    copy_outputs: bool = True
    _hpf: Any = None
    _hef: Any = None
    _vdevice: Any = None
    _infer_model: Any = None
    _config_ctx: Any = None
    _configured_model: Any = None
    _last_job: Any = None
    input_names: list[str] = None  # type: ignore[assignment]
    output_names: list[str] = None  # type: ignore[assignment]
    input_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]
    output_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]
    runtime_input_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]
    runtime_output_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_names = []
        self.output_names = []
        if self.canonical_input_slot_names is None:
            self.canonical_input_slot_names = []
        if self.canonical_output_slot_names is None:
            self.canonical_output_slot_names = []
        self.attested_source_output_shapes = _validated_attested_source_output_shapes(
            self.attested_source_output_shapes
        )
        self.input_shapes = {}
        self.output_shapes = {}
        self.runtime_input_shapes = {}
        self.runtime_output_shapes = {}
        self._input_contig_cache: dict[str, np.ndarray] = {}
        self._hef_input_names: list[str] = []
        self._hef_output_names: list[str] = []
        self._hef_input_shapes: dict[str, tuple[int, ...]] = {}
        self._hef_output_shapes: dict[str, tuple[int, ...]] = {}
        self._input_name_hef_to_canonical: dict[str, str] = {}
        self._input_name_canonical_to_hef: dict[str, str] = {}
        self._output_name_hef_to_canonical: dict[str, str] = {}
        self._input_format_names: dict[str, str] = {}
        self._input_dtypes: dict[str, Any] = {}
        self._input_quant_infos: dict[str, Any] = {}
        self._output_format_names: dict[str, str] = {}
        self._output_dtypes: dict[str, Any] = {}
        self._output_quant_infos: dict[str, Any] = {}
        self._hot_binding: Any = None
        self._hot_input_buffers: dict[str, np.ndarray] = {}
        self._hot_output_buffers: dict[str, np.ndarray] = {}
        self._open()

    def _open(self) -> None:
        hpf = _import_hailo_module()
        if not self.hef_path.exists() or self.hef_path.stat().st_size <= 0:
            raise RuntimeError(f"Invalid HEF file: {self.hef_path}")

        self._hpf = hpf
        self._hef = hpf.HEF(str(self.hef_path))
        in_infos = list(self._hef.get_input_vstream_infos())
        out_infos = list(self._hef.get_output_vstream_infos())
        self._hef_input_names = _hailo_stream_names_ordered(in_infos)
        self._hef_output_names = _hailo_stream_names_ordered(out_infos)
        self._hef_input_shapes = {str(x.name): tuple(x.shape) for x in in_infos}
        self._hef_output_shapes = {str(x.name): tuple(x.shape) for x in out_infos}

        onnx_input_names, onnx_output_names, onnx_input_shapes, onnx_output_shapes = _load_onnx_io_names_and_shapes(self.onnx_model_path)
        onnx_output_names, onnx_output_shapes = _reconcile_attested_source_output_metadata(
            onnx_output_names,
            onnx_output_shapes,
            self.attested_source_output_shapes,
        )
        input_aliases = _build_hailo_io_name_map(
            self._hef_input_names,
            onnx_input_names,
            hailo_shapes=self._hef_input_shapes,
            onnx_shapes=onnx_input_shapes,
            slot_names=self.canonical_input_slot_names,
        )
        if self.attested_source_output_shapes:
            output_aliases = _build_attested_hailo_output_map(
                self._hef_output_names,
                self.attested_source_output_shapes,
                self._hef_output_shapes,
            )
        else:
            output_aliases = _build_hailo_io_name_map(
                self._hef_output_names,
                onnx_output_names,
                hailo_shapes=self._hef_output_shapes,
                onnx_shapes=onnx_output_shapes,
                slot_names=self.canonical_output_slot_names,
            )

        self._input_name_hef_to_canonical = {name: str(input_aliases.get(name, name)) for name in self._hef_input_names}
        self._input_name_canonical_to_hef = {}
        for name in self._hef_input_names:
            canonical = self._input_name_hef_to_canonical.get(name, name)
            self._input_name_canonical_to_hef.setdefault(str(canonical), str(name))
        self._output_name_hef_to_canonical = {name: str(output_aliases.get(name, name)) for name in self._hef_output_names}

        self.input_names = [self._input_name_hef_to_canonical.get(name, name) for name in self._hef_input_names]
        self.output_names = [self._output_name_hef_to_canonical.get(name, name) for name in self._hef_output_names]
        if self.attested_source_output_shapes:
            self.output_names = list(self.attested_source_output_shapes)
        self.runtime_input_shapes = {
            self._input_name_hef_to_canonical.get(name, name): tuple(self._hef_input_shapes.get(name, ()))
            for name in self._hef_input_names
        }
        self.runtime_output_shapes = {
            self._output_name_hef_to_canonical.get(name, name): tuple(self._hef_output_shapes.get(name, ()))
            for name in self._hef_output_names
        }
        self.input_shapes = {
            self._input_name_hef_to_canonical.get(name, name): tuple(
                onnx_input_shapes.get(self._input_name_hef_to_canonical.get(name, name), self._hef_input_shapes.get(name, ()))
            )
            for name in self._hef_input_names
        }
        self.output_shapes = {
            self._output_name_hef_to_canonical.get(name, name): tuple(
                onnx_output_shapes.get(self._output_name_hef_to_canonical.get(name, name), self._hef_output_shapes.get(name, ()))
            )
            for name in self._hef_output_names
        }

        params = None
        create_params = getattr(hpf.VDevice, "create_params", None)
        if callable(create_params):
            params = create_params()
            scheduling = getattr(hpf, "HailoSchedulingAlgorithm", None)
            round_robin = getattr(scheduling, "ROUND_ROBIN", None) if scheduling is not None else None
            if round_robin is not None and hasattr(params, "scheduling_algorithm"):
                params.scheduling_algorithm = round_robin
            if self.scheduler_group_id and hasattr(params, "group_id"):
                params.group_id = str(self.scheduler_group_id)
        self._vdevice = hpf.VDevice(params) if params is not None else hpf.VDevice()

        self._infer_model = self._vdevice.create_infer_model(str(self.hef_path))
        if hasattr(self._infer_model, "set_batch_size"):
            self._infer_model.set_batch_size(max(1, int(self.batch_size)))

        self._set_input_formats(in_infos)
        self._set_output_formats(out_infos)

        self._config_ctx = self._infer_model.configure()
        self._configured_model = self._config_ctx.__enter__() if hasattr(self._config_ctx, "__enter__") else self._config_ctx
        if hasattr(self._configured_model, "set_scheduler_priority"):
            self._configured_model.set_scheduler_priority(int(self.scheduler_priority))
        if self.hotloop:
            self._init_hotloop_binding()

    def _infer_input(self, name: Optional[str] = None) -> Any:
        if name is not None:
            try:
                return self._infer_model.input(str(name))
            except Exception:
                pass
        return self._infer_model.input()

    def _infer_output(self, name: Optional[str] = None) -> Any:
        if name is not None:
            try:
                return self._infer_model.output(str(name))
            except Exception:
                pass
        return self._infer_model.output()

    def _binding_input(self, binding: Any, name: Optional[str] = None) -> Any:
        if name is not None:
            try:
                return binding.input(str(name))
            except Exception:
                pass
        return binding.input()

    def _binding_output(self, binding: Any, name: Optional[str] = None) -> Any:
        if name is not None:
            try:
                return binding.output(str(name))
            except Exception:
                pass
        return binding.output()

    def _set_input_formats(self, in_infos: list[Any]) -> None:
        info_by_name = {str(getattr(info, "name", "")): info for info in in_infos}
        for name in self._hef_input_names or [None]:  # type: ignore[list-item]
            if self.quantized_inputs:
                info = info_by_name.get(str(name))
                fmt_name = _hailo_native_vstream_format_type_name(
                    self._hef, info, direction="input",
                )
                if fmt_name != "UINT8":
                    raise RuntimeError(
                        "Hailo-10 native Part1 input requires a HEF-native "
                        f"UINT8 stream; {name!r} resolved to {fmt_name or 'UNKNOWN'}"
                    )
            else:
                fmt_name = "FLOAT32"
            fmt = _format_type_by_name(self._hpf, fmt_name)
            if fmt is None:
                raise RuntimeError(
                    f"HailoRT FormatType lacks required input type {fmt_name}"
                )
            try:
                self._infer_input(name).set_format_type(fmt)
            except Exception:
                if len(self._hef_input_names) <= 1:
                    raise
            self._input_format_names[str(name)] = fmt_name
            self._input_dtypes[str(name)] = _np_dtype_from_hailo_format_name(
                fmt_name
            )
            if self.quantized_inputs:
                self._input_quant_infos[str(name)] = _single_hailo_quant_info(
                    self._infer_input(name), info_by_name.get(str(name)),
                    label=f"Hailo input {name!r}",
                )

    def _set_output_formats(self, out_infos: list[Any]) -> None:
        info_by_name = {str(getattr(info, "name", "")): info for info in out_infos}
        for name in self._hef_output_names:
            info = info_by_name.get(name)
            if self.quantized_outputs:
                fmt_name = _hailo_native_vstream_format_type_name(
                    self._hef, info, direction="output",
                )
                if fmt_name != "UINT8":
                    raise RuntimeError(
                        "Hailo-10 native Part1 output must resolve to UINT8; "
                        f"{name!r} resolved to {fmt_name or 'UNKNOWN'}"
                    )
            else:
                fmt_name = "FLOAT32"
            fmt = _format_type_by_name(self._hpf, fmt_name)
            if fmt is None:
                raise RuntimeError(
                    f"HailoRT FormatType lacks required output type {fmt_name}"
                )
            try:
                self._infer_output(name).set_format_type(fmt)
            except Exception:
                if len(self._hef_output_names) <= 1:
                    raise
            self._output_format_names[name] = fmt_name
            self._output_dtypes[name] = _np_dtype_from_hailo_format_name(fmt_name)
            if self.quantized_outputs:
                self._output_quant_infos[name] = _single_hailo_quant_info(
                    self._infer_output(name), info,
                    label=f"Hailo output {name!r}",
                )

    def quantize_input(self, name: str, values: np.ndarray) -> np.ndarray:
        """Quantize one canonical float input with its exact HEF QuantInfo.

        Callers invoke this while preparing the fixed feed, before any measured
        InferModel submission.  Pre-quantized UINT8 feeds (for example a sealed
        energy replay) are accepted unchanged after the exact format check.
        """

        canonical_name = str(name)
        hef_name = self._input_name_canonical_to_hef.get(
            canonical_name, canonical_name,
        )
        expected = np.dtype(self._input_dtypes.get(hef_name, np.uint8))
        if expected != np.dtype(np.uint8):
            raise RuntimeError(
                f"Hailo input {canonical_name!r} is not HEF-native UINT8"
            )
        source = np.asarray(values)
        if np.dtype(source.dtype) == expected:
            return np.ascontiguousarray(source)
        source_f32 = np.ascontiguousarray(source, dtype=np.float32)
        destination = np.empty(source_f32.shape, dtype=expected)
        quant = self._input_quant_infos.get(hef_name)
        if quant is None:
            raise RuntimeError(
                f"Hailo input {canonical_name!r} has no exact HEF QuantInfo"
            )
        _hailort_transform_utils(self._hpf).quantize_input_buffer(
            source_f32, destination, int(source_f32.size), quant,
        )
        return destination

    def dequantize_output(self, name: str, values: np.ndarray) -> np.ndarray:
        """Dequantize one native output using its own exact HEF QuantInfo."""

        canonical_name = str(name)
        hef_name = next(
            (
                raw_name for raw_name, mapped in
                self._output_name_hef_to_canonical.items()
                if str(mapped) == canonical_name
            ),
            canonical_name,
        )
        source = np.ascontiguousarray(values)
        native_dtype = np.dtype(
            self._output_dtypes.get(hef_name, source.dtype)
        )
        if np.dtype(source.dtype) != native_dtype:
            raise TypeError(
                f"Hailo output {canonical_name!r} expected {native_dtype}, "
                f"got {source.dtype}"
            )
        quant = self._output_quant_infos.get(hef_name)
        if quant is None:
            raise RuntimeError(
                f"Hailo output {canonical_name!r} has no exact HEF QuantInfo"
            )
        destination = np.empty(source.shape, dtype=np.float32)
        _hailort_transform_utils(self._hpf).dequantize_output_buffer(
            source, destination, int(source.size), quant,
        )
        return destination

    def _output_buffer_shape(self, name: str) -> tuple[int, ...]:
        try:
            shape = getattr(self._infer_output(name), "shape", None)
            if shape:
                return tuple(int(x) for x in shape)
        except Exception:
            pass
        return tuple(self._hef_output_shapes.get(str(name), ()))

    def _alloc_output_buffers(self) -> dict[str, np.ndarray]:
        return {
            name: np.empty(self._output_buffer_shape(name), dtype=self._output_dtypes.get(name, np.float32))
            for name in self._hef_output_names
        }

    def _input_buffer_dtype(self) -> Any:
        if not self.quantized_inputs:
            return np.float32
        native_dtypes = {
            np.dtype(value) for value in self._input_dtypes.values()
        }
        if native_dtypes != {np.dtype(np.uint8)}:
            raise RuntimeError(
                "Hailo-10 Part1 input format is not the required native UINT8"
            )
        return np.uint8

    def _make_binding(self, infer_inputs: dict[str, np.ndarray]) -> Any:
        output_buffers = self._alloc_output_buffers()
        binding = self._configured_model.create_bindings(output_buffers=output_buffers)
        for hef_name, arr in infer_inputs.items():
            self._binding_input(binding, hef_name).set_buffer(np.asarray(arr))
        return binding

    def _init_hotloop_binding(self) -> None:
        self._hot_output_buffers = self._alloc_output_buffers()
        self._hot_binding = self._configured_model.create_bindings(output_buffers=self._hot_output_buffers)
        self._hot_input_buffers = {}
        dtype = self._input_buffer_dtype()
        for hef_name in self._hef_input_names:
            shape = tuple(self._hef_input_shapes.get(str(hef_name), ())) or (1,)
            buf = np.empty(shape, dtype=dtype)
            self._hot_input_buffers[str(hef_name)] = buf
            self._binding_input(self._hot_binding, str(hef_name)).set_buffer(buf)

    def _hotloop_binding_for_inputs(self, infer_inputs: dict[str, np.ndarray]) -> Any:
        if self._hot_binding is None:
            self._init_hotloop_binding()
        for hef_name, arr in infer_inputs.items():
            src = np.asarray(arr)
            buf = self._hot_input_buffers.get(str(hef_name))
            if buf is None or tuple(buf.shape) != tuple(src.shape) or buf.dtype != src.dtype:
                buf = np.empty(src.shape, dtype=src.dtype)
                self._hot_input_buffers[str(hef_name)] = buf
                self._binding_input(self._hot_binding, str(hef_name)).set_buffer(buf)
            np.copyto(buf, src, casting="unsafe")
        return self._hot_binding

    def _prepare_infer_inputs(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        infer_inputs: Dict[str, np.ndarray] = {}
        for canonical_name in self.input_names:
            hef_name = self._input_name_canonical_to_hef.get(str(canonical_name), str(canonical_name))
            if canonical_name in inputs:
                arr = np.asarray(inputs[canonical_name])
            elif hef_name in inputs:
                arr = np.asarray(inputs[hef_name])
            else:
                raise KeyError(f"Missing required Hailo input '{canonical_name}'")

            runtime_shape = self.runtime_input_shapes.get(str(canonical_name))
            ok, adapted = _try_adapt_tensor(arr, runtime_shape)
            arr = adapted if ok else arr
            expected_dtype = np.dtype(
                self._input_dtypes.get(
                    str(hef_name),
                    np.uint8 if self.quantized_inputs else np.float32,
                )
            )
            if np.dtype(arr.dtype) != expected_dtype:
                raise TypeError(
                    "Hailo InferModel input was not prepared in its selected "
                    f"host format: {canonical_name!r} expected {expected_dtype}, "
                    f"got {arr.dtype}. Quantization must happen before the "
                    "measured InferModel hotloop."
                )
            infer_inputs[hef_name] = _ensure_c_contiguous_cached(self._input_contig_cache, str(canonical_name), arr)
        return infer_inputs

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._configured_model is None:
            raise RuntimeError("Hailo InferModel session is closed")

        infer_inputs = self._prepare_infer_inputs(inputs)
        binding = self._hotloop_binding_for_inputs(infer_inputs) if self.hotloop else self._make_binding(infer_inputs)
        callback_state: dict[str, Any] = {}
        done = threading.Event()

        def _callback(completion_info: Any) -> None:
            exc = getattr(completion_info, "exception", None)
            if exc:
                callback_state["exception"] = exc
            done.set()

        if hasattr(self._configured_model, "wait_for_async_ready"):
            self._configured_model.wait_for_async_ready(timeout_ms=int(self.timeout_ms))
        self._last_job = self._configured_model.run_async([binding], _callback)
        if hasattr(self._last_job, "wait"):
            self._last_job.wait(int(self.timeout_ms))
        done.wait(max(0.001, float(self.timeout_ms) / 1000.0))
        if callback_state.get("exception"):
            raise RuntimeError(f"Hailo InferModel inference failed: {callback_state['exception']}")

        raw_outputs: dict[str, np.ndarray] = {}
        for name in self._hef_output_names:
            arr = np.asarray(self._binding_output(binding, name).get_buffer())
            if self.copy_outputs:
                arr = np.array(arr, copy=True)
            raw_outputs[str(name)] = arr
        return self._canonical_outputs_from_slot_buffers(raw_outputs)

    def _canonical_outputs_from_slot_buffers(
        self, raw_buffers: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Expose async slot outputs under the same contract as ``infer``."""
        out: dict[str, np.ndarray] = {}
        for raw_name in self._hef_output_names:
            arr = np.asarray(raw_buffers[str(raw_name)])
            if self.copy_outputs:
                arr = np.array(arr, copy=True)
            if arr.ndim > 0 and arr.shape[0] == 1:
                try:
                    arr = np.squeeze(arr, axis=0)
                except Exception:
                    pass
            canonical_name = self._output_name_hef_to_canonical.get(str(raw_name), str(raw_name))
            canonical_shape = self.output_shapes.get(str(canonical_name))
            attested_shapes = getattr(
                self, "attested_source_output_shapes", None,
            ) or {}
            if str(canonical_name) in attested_shapes:
                out[str(canonical_name)] = _adapt_attested_hailo_output_tensor(
                    arr, attested_shapes[str(canonical_name)]
                )
            else:
                ok, adapted = _try_adapt_tensor(arr, canonical_shape)
                out[str(canonical_name)] = adapted if ok else arr
        return _order_attested_output_mapping(
            out, getattr(self, "attested_source_output_shapes", None) or {},
        )

    def _create_reusable_binding_slot(self) -> dict[str, Any]:
        output_buffers = self._alloc_output_buffers()
        binding = self._configured_model.create_bindings(output_buffers=output_buffers)
        input_buffers: dict[str, np.ndarray] = {}
        dtype = self._input_buffer_dtype()
        for hef_name in self._hef_input_names:
            shape = tuple(self._hef_input_shapes.get(str(hef_name), ())) or (1,)
            buf = np.empty(shape, dtype=dtype)
            input_buffers[str(hef_name)] = buf
            self._binding_input(binding, str(hef_name)).set_buffer(buf)
        done = threading.Event()
        done.set()
        return {
            "binding": binding,
            "input_buffers": input_buffers,
            "output_buffers": output_buffers,
            "job": None,
            "done": done,
            "exception": None,
            "prefilled": False,
            "submitted_count": 0,
            "completed_count": 0,
            "postprocess_completed_count": 0,
            "counter_lock": threading.Lock(),
        }

    def _fill_reusable_slot_inputs(self, slot: dict[str, Any], infer_inputs: dict[str, np.ndarray]) -> None:
        input_buffers = slot["input_buffers"]
        binding = slot["binding"]
        for hef_name, arr in infer_inputs.items():
            src = np.asarray(arr)
            buf = input_buffers.get(str(hef_name))
            if buf is None or tuple(buf.shape) != tuple(src.shape) or buf.dtype != src.dtype:
                buf = np.empty(src.shape, dtype=src.dtype)
                input_buffers[str(hef_name)] = buf
                self._binding_input(binding, str(hef_name)).set_buffer(buf)
            np.copyto(buf, src, casting="unsafe")
        slot["prefilled"] = True

    def _wait_reusable_slot(self, slot: dict[str, Any]) -> None:
        job = slot.get("job")
        had_job = job is not None
        if job is not None and hasattr(job, "wait"):
            job.wait(int(self.timeout_ms))
        done = slot.get("done")
        if done is not None and hasattr(done, "wait"):
            callback_seen = bool(done.wait(max(0.001, float(self.timeout_ms) / 1000.0)))
            if had_job and not callback_seen:
                slot["job"] = None
                raise TimeoutError("Hailo InferModel async callback was not observed before timeout")
        exc = slot.get("exception")
        slot["job"] = None
        if exc:
            slot["exception"] = None
            raise RuntimeError(f"Hailo InferModel async inference failed: {exc}")

    def _submit_reusable_slot(
        self, slot: dict[str, Any], infer_inputs: dict[str, np.ndarray], *,
        copy_inputs: bool,
        postprocess_callback: Optional[Callable[[dict[str, np.ndarray]], Any]] = None,
    ) -> None:
        self._wait_reusable_slot(slot)
        if copy_inputs or not bool(slot.get("prefilled")):
            self._fill_reusable_slot_inputs(slot, infer_inputs)
        slot["exception"] = None
        slot["done"].clear()

        def _callback(completion_info: Any, *, _slot: dict[str, Any] = slot) -> None:
            exc = getattr(completion_info, "exception", None)
            if exc:
                _slot["exception"] = exc
            else:
                try:
                    if postprocess_callback is not None:
                        # The slot may not be reused until this callback has
                        # completed.  A Full-task work unit is therefore counted
                        # only after its exact raw outputs have passed the frozen
                        # host decoder and NMS inside the measured interval.
                        postprocess_callback(
                            self._canonical_outputs_from_slot_buffers(
                                dict(_slot["output_buffers"]),
                            )
                        )
                    with _slot["counter_lock"]:
                        _slot["completed_count"] = int(_slot.get("completed_count") or 0) + 1
                        if postprocess_callback is not None:
                            _slot["postprocess_completed_count"] = int(
                                _slot.get("postprocess_completed_count") or 0
                            ) + 1
                except Exception as callback_exc:
                    _slot["exception"] = callback_exc
            _slot["done"].set()

        if hasattr(self._configured_model, "wait_for_async_ready"):
            self._configured_model.wait_for_async_ready(timeout_ms=int(self.timeout_ms))
        slot["job"] = self._configured_model.run_async([slot["binding"]], _callback)
        with slot["counter_lock"]:
            slot["submitted_count"] = int(slot.get("submitted_count") or 0) + 1

    def benchmark_throughput(
        self,
        inputs: dict[str, np.ndarray],
        *,
        frames: int,
        inflight: int = 8,
        warmup_frames: int = 0,
        copy_inputs: bool = True,
        duration_s: float = 0.0,
        postprocess_callback: Optional[Callable[[dict[str, np.ndarray]], Any]] = None,
    ) -> dict[str, Any]:
        if self._configured_model is None:
            raise RuntimeError("Hailo InferModel session is closed")
        frames = max(0, int(frames))
        duration_s = max(0.0, float(duration_s or 0.0))
        warmup_frames = max(0, int(warmup_frames))
        inflight = max(1, int(inflight))
        infer_inputs = self._prepare_infer_inputs(inputs)
        slots = [self._create_reusable_binding_slot() for _ in range(inflight)]

        def _completed_count() -> int:
            total = 0
            for slot in slots:
                with slot["counter_lock"]:
                    total += int(slot.get("completed_count") or 0)
            return int(total)

        def _postprocess_completed_count() -> int:
            total = 0
            for slot in slots:
                with slot["counter_lock"]:
                    total += int(slot.get("postprocess_completed_count") or 0)
            return int(total)

        def _run_frame_count(count: int) -> int:
            before = _completed_count()
            postprocess_before = _postprocess_completed_count()
            used_slot_indices: set[int] = set()
            for idx in range(int(count)):
                slot_index = idx % inflight
                used_slot_indices.add(slot_index)
                self._submit_reusable_slot(
                    slots[slot_index], infer_inputs,
                    copy_inputs=bool(copy_inputs),
                    postprocess_callback=postprocess_callback,
                )
            # Drain only slots that received work in this interval.  Waiting on
            # untouched slots used to make the completion contract depend on
            # their initial Event state instead of observed callbacks.
            for slot_index in sorted(used_slot_indices):
                self._wait_reusable_slot(slots[slot_index])
            completed = _completed_count() - before
            if completed != int(count):
                raise RuntimeError(
                    f"Hailo InferModel completion count mismatch: requested={int(count)} observed={completed}"
                )
            postprocess_completed = _postprocess_completed_count() - postprocess_before
            if postprocess_callback is not None and postprocess_completed != int(count):
                raise RuntimeError(
                    "Hailo InferModel frozen postprocess completion count mismatch: "
                    f"requested={int(count)} observed={postprocess_completed}"
                )
            return completed

        if warmup_frames:
            warmup_completed = _run_frame_count(warmup_frames)
        else:
            warmup_completed = 0

        # ``frames`` is a minimum work budget, never a duration estimate.  For
        # energy replays the loop additionally remains active for the requested
        # wall-clock duration.  Additional chunks reuse the same configured
        # model, bindings and slots and every chunk is fully drained before the
        # next one begins.
        t0 = time.perf_counter()
        completed_frames = 0
        measured_postprocess_before = _postprocess_completed_count()
        while completed_frames < frames or time.perf_counter() - t0 < duration_s:
            remaining_frames = max(0, frames - completed_frames)
            elapsed = max(0.0, time.perf_counter() - t0)
            if remaining_frames > 0:
                chunk = remaining_frames
            else:
                observed_fps = completed_frames / elapsed if completed_frames and elapsed > 0 else 0.0
                remaining_s = max(0.0, duration_s - elapsed)
                chunk = max(inflight, int(math.ceil(observed_fps * min(1.0, remaining_s))))
            # Bound duration-extension chunks so cancellation/diagnostics are
            # not hidden behind another very long submission batch.
            chunk = max(1, min(int(chunk), 4096))
            completed_frames += _run_frame_count(chunk)
        t1 = time.perf_counter()
        elapsed_s = max(0.0, float(t1 - t0))
        postprocess_completed_frames = (
            _postprocess_completed_count() - measured_postprocess_before
        )
        fps = (float(completed_frames) / elapsed_s) if elapsed_s > 0.0 and completed_frames > 0 else 0.0
        return {
            "frames": int(completed_frames),
            "requested_frames": int(frames),
            "minimum_requested_frames": int(frames),
            "completed_frames": int(completed_frames),
            "completed_work_units": int(completed_frames),
            "completed_work_units_source": (
                "hailo_infermodel_frozen_postprocess_success_callback_counter"
                if postprocess_callback is not None
                else "hailo_infermodel_success_callback_counter"
            ),
            "completed_work_units_status": "exact_runtime_counter",
            "postprocess_included": postprocess_callback is not None,
            "postprocess_completed_frames": int(postprocess_completed_frames),
            "postprocess_completion_status": (
                "exact_runtime_counter"
                if postprocess_callback is not None
                and postprocess_completed_frames == completed_frames
                else "not_requested" if postprocess_callback is None
                else "count_mismatch"
            ),
            "warmup_frames": int(warmup_frames),
            "warmup_completed_frames": int(warmup_completed),
            "inflight": int(inflight),
            "copy_inputs": bool(copy_inputs),
            "elapsed_s": elapsed_s,
            "requested_duration_s": duration_s,
            "minimum_duration_satisfied": bool(duration_s <= 0.0 or elapsed_s >= duration_s),
            "measurement_control": "minimum_frames_and_duration" if duration_s > 0.0 else "exact_frames",
            "fps": fps,
            "completion_interval_mean_ms": (1000.0 / fps) if fps > 0.0 else 0.0,
            "completion_interval_semantics": "reciprocal_steady_state_observed_completion_throughput",
            "latency_measured": False,
        }

    def describe_io(self) -> dict[str, Any]:
        """Expose the actual InferModel host formats selected from the HEF."""
        input_formats = sorted(set(self._input_format_names.values()))
        output_formats = sorted(set(self._output_format_names.values()))
        def _quant_payload(value: Any) -> dict[str, float]:
            return {
                "scale": float(getattr(
                    value, "qp_scale", getattr(value, "scale", 0.0),
                )),
                "zero_point": float(getattr(
                    value, "qp_zp",
                    getattr(value, "zero_point", getattr(value, "zp", 0.0)),
                )),
            }
        return {
            "schema_version": 3,
            "backend": "hailo",
            "runtime_api": "infer_model",
            "artifact": str(self.hef_path),
            "quantized_inputs": bool(self.quantized_inputs),
            "quantized_outputs": bool(self.quantized_outputs),
            "runtime_input_format": (
                input_formats[0].lower()
                if len(input_formats) == 1 else "mixed"
            ),
            "runtime_output_format": (
                output_formats[0].lower()
                if len(output_formats) == 1 else "mixed"
            ),
            "runtime_input_formats": dict(self._input_format_names),
            "runtime_output_formats": dict(self._output_format_names),
            "runtime_input_quantization": {
                name: _quant_payload(value)
                for name, value in self._input_quant_infos.items()
            },
            "runtime_output_quantization": {
                name: _quant_payload(value)
                for name, value in self._output_quant_infos.items()
            },
            "outputs_dequantized": not bool(self.quantized_outputs),
            "hotloop": bool(self.hotloop),
            "copy_outputs": bool(self.copy_outputs),
            "hef_inputs": list(self._hef_input_names),
            "canonical_inputs": list(self.input_names),
            "hef_outputs": list(self._hef_output_names),
            "canonical_outputs": list(self.output_names),
        }

    def close(self) -> None:
        try:
            if self._last_job is not None and hasattr(self._last_job, "wait"):
                self._last_job.wait(int(self.timeout_ms))
        except Exception:
            pass
        try:
            if self._config_ctx is not None and hasattr(self._config_ctx, "__exit__"):
                self._config_ctx.__exit__(None, None, None)
        finally:
            self._configured_model = None
            self._config_ctx = None
        try:
            if self._vdevice is not None and hasattr(self._vdevice, "__exit__"):
                self._vdevice.__exit__(None, None, None)
            elif self._vdevice is not None and hasattr(self._vdevice, "close"):
                self._vdevice.close()
        finally:
            self._vdevice = None


@dataclass
class _HailoPrepared:
    hef_path: Path
    session: Optional[Any]
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, tuple[int, ...]]
    output_shapes: dict[str, tuple[int, ...]]
    runtime_input_shapes: dict[str, tuple[int, ...]]
    runtime_output_shapes: dict[str, tuple[int, ...]]
    quantized_inputs: bool
    quantized_outputs: bool



def _timeout_seconds(value: Any, default: int) -> int:
    """Parse a hard timeout while allowing explicit off/unlimited values."""
    return parse_hailo_timeout_seconds(
        value,
        default=default,
        label="Hailo hard timeout",
    )


class HailoBackend:
    name = "hailo"
    capabilities = BackendCaps(
        needs_compiler=True,
        supports_two_stage=True,
        supports_cache_dir=True,
        supports_fp16=False,
    )

    def __init__(self, *, strict: bool = True, **cfg: Any) -> None:
        self.cfg = dict(cfg)
        self.default_compile_backend = str(self.cfg.get("compile_backend", "auto"))
        if strict:
            self._check_available()

    def _check_available(self) -> None:
        backend = str(self.cfg.get("compile_backend", "auto") or "auto").strip().lower()
        if backend in {"wsl"}:
            if shutil.which("wsl.exe") is None:
                raise RuntimeError("Hailo backend compile_backend='wsl' requires wsl.exe in PATH.")
            return

        if backend in {"local", "auto", "venv"}:
            dev_ok = (
                any(Path(p).exists() for p in ("/dev/hailo0", "/dev/hailo1", "/dev/h1x-0"))
                or any(Path("/dev").glob("h1x-*"))
                or shutil.which("hailortcli") is not None
            )
            if not dev_ok and backend == "local":
                raise RuntimeError("Hailo runtime device not detected (/dev/hailo0 or /dev/h1x-*). Install HailoRT driver.")
            try:
                __import__("hailo_platform")
            except Exception:
                try:
                    __import__("hailort")
                except Exception:
                    if backend == "local":
                        raise RuntimeError("Cannot import hailo_platform/hailort. Install HailoRT python bindings.")

    def prepare(self, run_cfg: RunCfg, artifacts_dir: Path) -> PreparedHandle:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        options = dict(self.cfg)
        options.update(run_cfg.options or {})

        model_path = Path(run_cfg.model_path).expanduser()
        if not model_path.is_absolute():
            model_path = model_path.resolve()

        hw_arch = str(options.get("hw_arch", "hailo8"))
        net_name = str(options.get("net_name") or model_path.stem)
        opt_level = int(options.get("opt_level", 1))
        calib_dir = options.get("calib_dir")
        calib_count = int(options.get("calib_count", 64))
        calib_batch_size = int(options.get("calib_batch_size", 8))
        fixup = bool(options.get("fixup", True))
        force_rebuild = bool(options.get("force_rebuild", False))
        keep_artifacts = bool(options.get("keep_artifacts", False))
        compile_backend = str(options.get("compile_backend", self.default_compile_backend) or "auto").lower().strip()
        timeout_s = _timeout_seconds(
            options.get("hard_timeout_s", options.get("timeout_s", 1800)),
            1800,
        )
        wsl_timeout_s = _timeout_seconds(
            options.get("wsl_hard_timeout_s", options.get("wsl_timeout_s", timeout_s)),
            timeout_s,
        )
        wsl_distro = options.get("wsl_distro")
        wsl_venv_activate = str(options.get("wsl_venv_activate", "auto"))

        direct_hef_raw = (
            options.get("hef_path")
            or options.get("precompiled_hef_path")
            or options.get("compiled_hef")
        )
        direct_hef = direct_hef_raw is not None or model_path.suffix.lower() == ".hef"
        onnx_model_path: Optional[Path] = None

        if direct_hef:
            hef_path = Path(str(direct_hef_raw or model_path)).expanduser()
            if not hef_path.is_absolute():
                hef_path = hef_path.resolve()

            onnx_raw = options.get("onnx_model_path") or options.get("onnx_path")
            if onnx_raw:
                onnx_model_path = Path(str(onnx_raw)).expanduser()
                if not onnx_model_path.is_absolute():
                    onnx_model_path = onnx_model_path.resolve()
                if not onnx_model_path.exists():
                    raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
            elif model_path.suffix.lower() == ".onnx":
                if not model_path.exists():
                    raise FileNotFoundError(f"ONNX model not found: {model_path}")
                onnx_model_path = model_path
        else:
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
            onnx_model_path = model_path

            hef_path = artifacts_dir / "compiled.hef"
            sha1 = hashlib.sha1(model_path.read_bytes()).hexdigest()
            cache_root = Path.home() / ".onnx_splitpoint_tool" / "hailo" / "hef_cache"
            cache_dir = cache_root / hw_arch / f"sha1_{sha1}" / f"opt{opt_level}_cal{calib_count}_bs{calib_batch_size}_fix{int(fixup)}"
            cached_hef = cache_dir / "compiled.hef"

            if force_rebuild:
                if compiler_dispatch_forbidden():
                    raise RuntimeError(
                        cache_miss_blocked_message(
                            "hailo_dfc", "force_rebuild requested"
                        )
                    )
                if hef_path.exists():
                    hef_path.unlink()

            if not hef_path.exists() and cached_hef.exists() and cached_hef.stat().st_size > 0:
                hef_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_hef, hef_path)

            if not hef_path.exists() or hef_path.stat().st_size <= 0:
                if compiler_dispatch_forbidden():
                    raise RuntimeError(
                        cache_miss_blocked_message(
                            "hailo_dfc", "native runner HEF cache miss"
                        )
                    )
                endpoint = str(
                    options.get("hailo_full_endpoint_mode")
                    or options.get("endpoint_mode")
                    or options.get("measurement_endpoint")
                    or ("full" if str(options.get("variant") or "").lower() == "full" else "part1")
                )
                attempt_kind = str(
                    options.get("hailo_attempt_kind")
                    or ("raw_head_fallback" if bool(options.get("raw_head_fallback")) else "decoded_full" if endpoint in {"decoded_nms", "decoded_pre_nms", "full"} else "part1")
                )
                end_nodes_raw = options.get("end_nodes") or options.get("hailo_end_nodes") or []
                end_nodes = (
                    [str(value) for value in end_nodes_raw]
                    if isinstance(end_nodes_raw, (list, tuple))
                    else [str(end_nodes_raw)] if str(end_nodes_raw or "").strip()
                    else []
                )
                attempt = start_hailo_attempt(
                    artifacts_dir,
                    attempt_kind=attempt_kind,
                    endpoint=endpoint,
                    source_onnx=model_path,
                    compiler_onnx=Path(str(options.get("compiler_onnx_path") or model_path)),
                    end_nodes=end_nodes,
                    timeout_policy={
                        "hard_timeout_s": int(timeout_s),
                        "wsl_hard_timeout_s": int(wsl_timeout_s),
                        "hard_timeout_enabled": bool(timeout_s > 0),
                        "heartbeat_enabled": True,
                        "manual_abort_enabled": True,
                    },
                    metadata={
                        "hw_arch": hw_arch,
                        "net_name": net_name,
                        "opt_level": opt_level,
                        "calib_count": calib_count,
                        "calib_batch_size": calib_batch_size,
                        "compile_backend": compile_backend,
                    },
                )
                attempt.heartbeat(compiler_phase="dispatch", detail=compile_backend)
                try:
                    result = self._compile_hef(
                        model_path=model_path,
                        hef_path=hef_path,
                        hw_arch=hw_arch,
                        net_name=net_name,
                        opt_level=opt_level,
                        calib_dir=Path(calib_dir).expanduser().resolve() if calib_dir else None,
                        calib_count=calib_count,
                        calib_batch_size=calib_batch_size,
                        fixup=fixup,
                        keep_artifacts=keep_artifacts,
                        compile_backend=compile_backend,
                        wsl_distro=wsl_distro,
                        wsl_venv_activate=wsl_venv_activate,
                        timeout_s=timeout_s,
                        wsl_timeout_s=wsl_timeout_s,
                    )
                except BaseException as exc:
                    attempt.finish(
                        error=exc,
                        compiler_phase="compile_dispatch",
                        returned=False,
                    )
                    raise
                if not result.get("ok"):
                    attempt.finish(
                        value=result,
                        compiler_phase=str(result.get("compiler_phase") or "compile_terminal"),
                        stdout_tail=str(result.get("subprocess_stdout_tail") or ""),
                        stderr_tail=str(result.get("subprocess_stderr_tail") or result.get("error") or ""),
                        returned=True,
                    )
                    raise RuntimeError(result.get("error") or "Hailo compilation failed")

                if not hef_path.exists() or hef_path.stat().st_size <= 0:
                    missing_result = dict(result)
                    missing_result.update({
                        "ok": False,
                        "status": "failed",
                        "semantic_status": "failed",
                        "error": f"Hailo compilation did not create valid HEF: {hef_path}",
                    })
                    attempt.finish(
                        value=missing_result,
                        compiler_phase=str(result.get("compiler_phase") or "compile_terminal"),
                        stdout_tail=str(result.get("subprocess_stdout_tail") or ""),
                        stderr_tail=str(missing_result["error"]),
                        returned=True,
                    )
                    raise RuntimeError(missing_result["error"])

                attempt.finish(
                    value=result,
                    compiler_phase=str(result.get("compiler_phase") or "compile_terminal"),
                    stdout_tail=str(result.get("subprocess_stdout_tail") or ""),
                    stderr_tail=str(result.get("subprocess_stderr_tail") or ""),
                    returned=True,
                )

                cache_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(hef_path, cached_hef)

        if not hef_path.exists() or hef_path.stat().st_size <= 0:
            raise RuntimeError(f"HEF missing or empty: {hef_path}")

        quantized_inputs = bool(options.get("quantized_inputs", False))
        quantized_outputs = bool(options.get("quantized_outputs", False))
        persistent_activation = bool(options.get("persistent_activation", False))
        runtime_api = str(options.get("runtime_api", "auto") or "auto").strip().lower()
        hw_arch_l = str(hw_arch or "").strip().lower()
        if runtime_api not in {"auto", "vstreams", "infer_model", "infermodel", "async"}:
            raise RuntimeError(f"Unsupported Hailo runtime_api '{runtime_api}'")
        if runtime_api == "vstreams" and (
            "hailo10" in hw_arch_l or "hailo15" in hw_arch_l
        ):
            raise RuntimeError(
                "Hailo-10/15 requires runtime_api='infer_model' so the exact "
                "HEF UINT8 QuantInfo contract is applied"
            )
        use_infer_model = runtime_api in {"infer_model", "infermodel", "async"} or (
            runtime_api == "auto" and ("hailo10" in hw_arch_l or "hailo15" in hw_arch_l)
        )

        session_kwargs = dict(
            hef_path=hef_path,
            quantized_inputs=quantized_inputs,
            quantized_outputs=quantized_outputs,
            persistent_activation=persistent_activation,
            onnx_model_path=onnx_model_path,
            canonical_input_slot_names=[str(x) for x in (options.get("canonical_input_slot_names") or []) if str(x)],
            canonical_output_slot_names=[str(x) for x in (options.get("canonical_output_slot_names") or []) if str(x)],
            attested_source_output_shapes=options.get(
                "attested_source_output_shapes"
            ),
        )
        if use_infer_model:
            try:
                session = _HailoInferModelSession(
                    **session_kwargs,
                    batch_size=int(options.get("batch_size", 1)),
                    timeout_ms=int(options.get("hailo_timeout_ms", options.get("timeout_ms", 10000))),
                    scheduler_group_id=str(options.get("scheduler_group_id", "SHARED") or "SHARED"),
                    scheduler_priority=int(options.get("scheduler_priority", 0)),
                    hotloop=_option_bool(options.get("hotloop", options.get("reuse_bindings", True)), True),
                    copy_outputs=_option_bool(options.get("copy_outputs", True), True),
                )
            except Exception as exc:
                raise RuntimeError(f"Hailo InferModel runtime failed to initialize: {type(exc).__name__}: {exc}") from exc
        else:
            session = _HailoSession(**session_kwargs)

        prepared = _HailoPrepared(
            hef_path=hef_path,
            session=session,
            input_names=list(session.input_names),
            output_names=list(session.output_names),
            input_shapes=dict(session.input_shapes),
            output_shapes=dict(session.output_shapes),
            runtime_input_shapes=dict(session.runtime_input_shapes),
            runtime_output_shapes=dict(session.runtime_output_shapes),
            quantized_inputs=quantized_inputs,
            quantized_outputs=quantized_outputs,
        )

        return PreparedHandle(
            input_names=list(prepared.input_names),
            output_names=list(prepared.output_names),
            handle=prepared,
        )

    def _compile_hef(
        self,
        *,
        model_path: Path,
        hef_path: Path,
        hw_arch: str,
        net_name: str,
        opt_level: int,
        calib_dir: Optional[Path],
        calib_count: int,
        calib_batch_size: int,
        fixup: bool,
        keep_artifacts: bool,
        compile_backend: str,
        wsl_distro: Optional[str],
        wsl_venv_activate: str,
        timeout_s: int,
        wsl_timeout_s: int,
    ) -> dict[str, Any]:
        mode = compile_backend
        if mode == "auto":
            mode = "local"
            try:
                __import__("hailo_sdk_client")
            except Exception:
                mode = "wsl" if shutil.which("wsl.exe") else "venv"

        if mode == "local":
            return self._compile_local(
                model_path=model_path,
                hef_path=hef_path,
                hw_arch=hw_arch,
                net_name=net_name,
                opt_level=opt_level,
                calib_dir=calib_dir,
                calib_count=calib_count,
                calib_batch_size=calib_batch_size,
                fixup=fixup,
                keep_artifacts=keep_artifacts,
            )
        if mode == "venv":
            return self._compile_subprocess(
                python_bin=sys.executable,
                model_path=model_path,
                hef_path=hef_path,
                hw_arch=hw_arch,
                net_name=net_name,
                opt_level=opt_level,
                calib_dir=calib_dir,
                calib_count=calib_count,
                calib_batch_size=calib_batch_size,
                fixup=fixup,
                keep_artifacts=keep_artifacts,
                timeout_s=timeout_s,
            )
        if mode == "wsl":
            return self._compile_wsl(
                model_path=model_path,
                hef_path=hef_path,
                hw_arch=hw_arch,
                net_name=net_name,
                opt_level=opt_level,
                calib_dir=calib_dir,
                calib_count=calib_count,
                calib_batch_size=calib_batch_size,
                fixup=fixup,
                keep_artifacts=keep_artifacts,
                timeout_s=wsl_timeout_s,
                wsl_distro=wsl_distro,
                wsl_venv_activate=wsl_venv_activate,
            )

        raise RuntimeError(f"Unsupported compile_backend '{compile_backend}'")

    def _compile_local(self, **kwargs: Any) -> dict[str, Any]:
        return _compile_hailo_hef_core(**kwargs)

    def _compile_subprocess(self, *, python_bin: str, timeout_s: int, **kwargs: Any) -> dict[str, Any]:
        payload = base64.b64encode(json.dumps(_serialize_kwargs(kwargs), sort_keys=True).encode("utf-8")).decode("ascii")
        code = (
            "import base64,json,sys;"
            "from pathlib import Path;"
            "from onnx_splitpoint_tool.runners.backends.hailo_backend import _compile_hailo_hef_core;"
            "d=json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'));"
            "d['model_path']=Path(d['model_path']);d['hef_path']=Path(d['hef_path']);"
            "d['calib_dir']=Path(d['calib_dir']) if d.get('calib_dir') else None;"
            "r=_compile_hailo_hef_core(**d);"
            f"print('{_RESULT_MARKER}'+json.dumps(r,sort_keys=True))"
        )
        proc = subprocess.run(
            [python_bin, "-c", code, payload],
            capture_output=True,
            text=True,
            timeout=(None if int(timeout_s) <= 0 else float(timeout_s)),
        )
        mix = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = _find_marker_json(mix)
        if parsed is None:
            raise RuntimeError(f"VENV compile failed (rc={proc.returncode}): no structured result marker found")
        parsed = dict(parsed)
        parsed.update({
            "subprocess_returncode": int(proc.returncode),
            "subprocess_stdout_tail": str(proc.stdout or "")[-16000:],
            "subprocess_stderr_tail": str(proc.stderr or "")[-16000:],
        })
        return parsed

    def _compile_wsl(
        self,
        *,
        model_path: Path,
        hef_path: Path,
        hw_arch: str,
        net_name: str,
        opt_level: int,
        calib_dir: Optional[Path],
        calib_count: int,
        calib_batch_size: int,
        fixup: bool,
        keep_artifacts: bool,
        timeout_s: int,
        wsl_distro: Optional[str],
        wsl_venv_activate: str,
    ) -> dict[str, Any]:
        wsl = shutil.which("wsl.exe")
        if not wsl:
            raise RuntimeError("compile_backend='wsl' requested but wsl.exe not found")

        resolved = get_dfc_manager().resolve_wsl_runtime(
            hw_arch=hw_arch,
            wsl_distro=wsl_distro,
            wsl_venv_activate=wsl_venv_activate,
        )
        activate = resolved.wsl_venv_activate
        if not activate:
            raise RuntimeError("No WSL venv activation path resolved for selected hw_arch.")

        payload = {
            "model_path": str(model_path),
            "hef_path": str(hef_path),
            "hw_arch": hw_arch,
            "net_name": net_name,
            "opt_level": int(opt_level),
            "calib_dir": str(calib_dir) if calib_dir else None,
            "calib_count": int(calib_count),
            "calib_batch_size": int(calib_batch_size),
            "fixup": bool(fixup),
            "keep_artifacts": bool(keep_artifacts),
        }
        b64 = base64.b64encode(json.dumps(payload, sort_keys=True).encode("utf-8")).decode("ascii")
        code = (
            "import base64,json,sys;"
            "from pathlib import Path;"
            "from onnx_splitpoint_tool.runners.backends.hailo_backend import _compile_hailo_hef_core;"
            "d=json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'));"
            "d['model_path']=Path(d['model_path']);d['hef_path']=Path(d['hef_path']);"
            "d['calib_dir']=Path(d['calib_dir']) if d.get('calib_dir') else None;"
            "r=_compile_hailo_hef_core(**d);"
            f"print('{_RESULT_MARKER}'+json.dumps(r,sort_keys=True))"
        )
        bash_cmd = (
            "set -e; "
            f"source {shlex.quote(activate)}; "
            f"python -c {shlex.quote(code)} {shlex.quote(b64)}"
        )
        cmd = [wsl]
        if resolved.wsl_distro:
            cmd += ["-d", str(resolved.wsl_distro)]
        cmd += ["--", "bash", "-lc", bash_cmd]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=(None if int(timeout_s) <= 0 else float(timeout_s)),
        )
        mix = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = _find_marker_json(mix)
        if parsed is None:
            tail = mix[-4000:] if mix else "<no output>"
            raise RuntimeError(f"WSL compile failed (rc={proc.returncode}): no result marker. Output:\n{tail}")
        parsed = dict(parsed)
        parsed.update({
            "subprocess_returncode": int(proc.returncode),
            "subprocess_stdout_tail": str(proc.stdout or "")[-16000:],
            "subprocess_stderr_tail": str(proc.stderr or "")[-16000:],
        })
        return parsed

    def run(self, prepared: PreparedHandle, inputs: dict) -> BackendRunOut:
        prep: _HailoPrepared = prepared.handle
        if prep.session is None:
            raise RuntimeError("Hailo prepared session already cleaned up")

        missing = [name for name in prep.input_names if name not in inputs]
        if missing:
            raise KeyError(f"Missing required inputs for Hailo backend: {missing}")

        t0 = time.perf_counter()
        outputs = prep.session.infer({k: np.asarray(v) for k, v in inputs.items()})
        t1 = time.perf_counter()
        return BackendRunOut(outputs=outputs, metrics={"infer_ms": (t1 - t0) * 1000.0})

    def cleanup(self, prepared: PreparedHandle) -> None:
        prep: _HailoPrepared = prepared.handle
        try:
            if prep.session is not None:
                prep.session.close()
        finally:
            prep.session = None


def _serialize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _find_marker_json(text: str) -> Optional[dict[str, Any]]:
    marker = _RESULT_MARKER
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line.startswith(marker):
            continue
        payload = line[len(marker):].strip()
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _compile_hailo_hef_core(
    *,
    model_path: Path,
    hef_path: Path,
    hw_arch: str,
    net_name: str,
    opt_level: int,
    calib_dir: Optional[Path],
    calib_count: int,
    calib_batch_size: int,
    fixup: bool,
    keep_artifacts: bool,
) -> dict[str, Any]:
    del calib_dir, fixup
    if compiler_dispatch_forbidden():
        return {
            "ok": False,
            "status": "cache_miss_blocked",
            "error": cache_miss_blocked_message("hailo_dfc"),
        }
    try:
        from hailo_sdk_client import ClientRunner  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "error": (
                "Cannot import hailo_sdk_client for HEF compilation. "
                "Install Hailo DFC SDK in the selected compile environment. "
                f"Details: {type(exc).__name__}: {exc}"
            ),
        }

    hef_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runner = ClientRunner(hw_arch=str(hw_arch))
        runner.translate_onnx_model(
            model=str(model_path),
            net_name=str(net_name),
            net_input_shapes=None,
            disable_rt_metadata_extraction=True,
        )
        bs = max(1, min(int(calib_batch_size), int(calib_count)))
        script = (
            f"model_optimization_flavor(optimization_level={int(opt_level)}, batch_size={int(bs)})\n"
            f"model_optimization_config(calibration, batch_size={int(bs)}, calibset_size={int(calib_count)})\n"
        )
        runner.load_model_script(script)

        hn = runner.get_hn_dict() or {}
        layers = hn.get("layers") or {}
        input_names = [name for name, meta in layers.items() if isinstance(meta, dict) and meta.get("type") == "input_layer"]
        if not input_names:
            raise RuntimeError("No input layers found in HN metadata")

        rng = np.random.default_rng(0)
        calib_inputs: dict[str, np.ndarray] = {}
        for name in input_names:
            shape = _hn_shape(layers.get(name) or {})
            if not shape:
                shape = [1]
            calib_inputs[name] = np.ascontiguousarray(rng.random((int(calib_count), *shape), dtype=np.float32))

        runner.optimize(calib_inputs)
        if keep_artifacts:
            try:
                runner.save_har(str(hef_path.parent / "quantized.har"))
            except Exception:
                pass

        hef_bytes = runner.compile()
        hef_path.write_bytes(hef_bytes)
        return {"ok": True, "hef_path": str(hef_path)}
    except Exception as exc:
        return {"ok": False, "error": f"HEF compile failed: {type(exc).__name__}: {exc}"}


def _hn_shape(meta: dict[str, Any]) -> list[int]:
    for key in ("input_shape", "shape", "output_shapes", "input_shapes"):
        value = meta.get(key)
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        if not isinstance(value, list):
            continue
        dims = [int(x) for x in value if isinstance(x, int) and x > 0]
        if dims and dims[0] == 1 and len(dims) > 1:
            dims = dims[1:]
        if dims:
            return dims
    return []
