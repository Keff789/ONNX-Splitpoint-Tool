#!/usr/bin/env python3
"""Generic validation helper for native FIFO TensorRT output dumps.

This intentionally does not implement task-specific postprocessing.  It validates
that native FIFO output dumps are complete, finite, and optionally compares them
against a reference .npz with matching output names.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Any

import numpy as np

_DTYPE = {
    "float32": np.float32, "FLOAT": np.float32, "fp32": np.float32,
    "float16": np.float16, "HALF": np.float16, "fp16": np.float16,
    "uint8": np.uint8, "UINT8": np.uint8,
    "int8": np.int8, "INT8": np.int8,
    "int32": np.int32, "INT32": np.int32,
    "int64": np.int64, "INT64": np.int64,
}

def load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))

def _arr_from_entry(root: Path, e: dict[str, Any]) -> np.ndarray:
    dtype = _DTYPE.get(str(e.get("dtype") or ""), np.float32)
    shape = tuple(int(x) for x in (e.get("shape") or []))
    f = root / str(e.get("file") or "")
    arr = np.fromfile(f, dtype=dtype)
    if shape and int(np.prod(shape)) == arr.size:
        arr = arr.reshape(shape)
    return arr

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="native_fifo_outputs_manifest.json")
    ap.add_argument("--reference-npz", default="", help="Optional npz with tensors named like native outputs")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--out", default="")
    ns = ap.parse_args()
    manifest = Path(ns.manifest).expanduser().resolve()
    root = manifest.parent
    payload = load_json(manifest)
    ref = np.load(ns.reference_npz) if ns.reference_npz else None
    rows = []
    ok = True
    for e in payload.get("outputs", []):
        arr = _arr_from_entry(root, e)
        finite = bool(np.isfinite(arr.astype(np.float32, copy=False)).all()) if arr.size else True
        row = {
            "name": e.get("name"), "file": e.get("file"), "dtype": e.get("dtype"),
            "shape": list(arr.shape), "size": int(arr.size), "finite": finite,
            "min": float(np.nanmin(arr.astype(np.float32))) if arr.size else None,
            "max": float(np.nanmax(arr.astype(np.float32))) if arr.size else None,
            "mean": float(np.nanmean(arr.astype(np.float32))) if arr.size else None,
        }
        if not finite:
            ok = False
        if ref is not None:
            name = str(e.get("name") or "")
            key = name if name in ref.files else None
            if key is None:
                # fallback by sanitized filename stem
                stem = Path(str(e.get("file") or "")).stem
                key = stem if stem in ref.files else None
            if key is None:
                row["reference_status"] = "missing"
                ok = False
            else:
                rarr = np.asarray(ref[key])
                if rarr.shape != arr.shape:
                    row["reference_status"] = "shape_mismatch"
                    row["reference_shape"] = list(rarr.shape)
                    ok = False
                else:
                    diff = np.abs(arr.astype(np.float32) - rarr.astype(np.float32))
                    row["reference_status"] = "ok"
                    row["max_abs"] = float(diff.max()) if diff.size else 0.0
                    row["mean_abs"] = float(diff.mean()) if diff.size else 0.0
                    row["allclose"] = bool(np.allclose(arr, rarr, rtol=ns.rtol, atol=ns.atol))
                    if not row["allclose"]:
                        ok = False
        rows.append(row)
    out = {"schema": "onnx-splitpoint/native-fifo-output-validation", "schema_version": 1, "ok": ok, "manifest": str(manifest), "reference_npz": str(ns.reference_npz or ""), "outputs": rows}
    out_path = Path(ns.out).expanduser().resolve() if ns.out else root / "native_fifo_output_validation.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"ok": ok, "out": str(out_path), "output_count": len(rows)}, indent=2))
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
