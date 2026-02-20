"""ONNX split/export: validation helpers (ONNXRuntime).

This module contains optional ONNXRuntime-based validation utilities.
It is kept separate so graph splitting code can be used without onnxruntime installed.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import onnx
from onnx import TensorProto

from .onnx_utils import shape_from_vi, value_info_map
from .split_export_graph import infer_shapes_safe


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
