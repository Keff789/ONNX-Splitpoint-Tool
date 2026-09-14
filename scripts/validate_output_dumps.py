#!/usr/bin/env python3
"""Validate or compare runner output dumps using one shared dump contract.

Supported inputs:
- native_fifo_outputs_manifest.json (native C++ FIFO runner)
- output_dump_manifest.json (produced by npz_to_output_dump_manifest.py)
- .npz directly (baseline/generic runner output cache)

This is deliberately task-agnostic. It is the common tensor-level layer below
classification/detection validators. Task-specific validators can consume the
same normalized tensor map later.
"""
from __future__ import annotations
import argparse, json, re
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


def _safe_name(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", x or "tensor").strip("_") or "tensor"


def _load_manifest(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = path.parent
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, np.ndarray] = {}
    for e in payload.get("outputs", []):
        name = str(e.get("name") or Path(str(e.get("file") or "tensor")).stem)
        dtype = _DTYPE.get(str(e.get("dtype") or ""), np.float32)
        shape = tuple(int(x) for x in (e.get("shape") or []))
        recorded = Path(str(e.get("file") or "")).expanduser()
        f = recorded if recorded.is_absolute() else root / recorded
        if not f.is_file() and recorded.name:
            # Copied remote manifests retain their producer-host absolute
            # paths while rsync materialises each tensor beside the manifest.
            # Rebase only to that exact sibling basename; never search a
            # broader tree or silently select between multiple candidates.
            sibling = root / recorded.name
            if sibling.is_file():
                f = sibling
        if not f.is_file():
            raise FileNotFoundError(
                f"output tensor is missing for {name!r}: "
                f"recorded={recorded} resolved={f}"
            )
        arr = np.fromfile(f, dtype=dtype)
        if shape and int(np.prod(shape)) == arr.size:
            arr = arr.reshape(shape)
        out[name] = arr
    return out, payload


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    d = np.load(path, allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"schema": "npz", "source_npz": str(path)}
    for k in d.files:
        if k == "__meta__":
            try:
                meta["embedded_meta"] = json.loads(bytes(d[k].tolist()).decode("utf-8"))
            except Exception:
                pass
            continue
        out[k] = np.asarray(d[k])
    return out, meta


def load_dump(path: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if p.suffix.lower() == ".npz":
        return _load_npz(p)
    return _load_manifest(p)


def summarize(arr: np.ndarray) -> dict[str, Any]:
    f = arr.astype(np.float32, copy=False) if arr.size else arr
    return {
        "dtype": str(arr.dtype),
        "shape": [int(x) for x in arr.shape],
        "size": int(arr.size),
        "bytes": int(arr.nbytes),
        "finite": bool(np.isfinite(f).all()) if arr.size else True,
        "min": float(np.nanmin(f)) if arr.size else None,
        "max": float(np.nanmax(f)) if arr.size else None,
        "mean": float(np.nanmean(f)) if arr.size else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="Candidate dump: manifest JSON or .npz")
    ap.add_argument("--reference", default="", help="Optional reference dump: manifest JSON or .npz")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cand, cand_meta = load_dump(args.candidate)
    ref, ref_meta = load_dump(args.reference) if args.reference else ({}, {})
    rows = []
    ok = True
    for name, arr in cand.items():
        row = {"name": name, **summarize(arr)}
        if not row["finite"]:
            ok = False
        if ref:
            key = name if name in ref else (_safe_name(name) if _safe_name(name) in ref else None)
            if key is None:
                row["reference_status"] = "missing"
                ok = False
            else:
                r = ref[key]
                if r.shape != arr.shape:
                    row["reference_status"] = "shape_mismatch"
                    row["reference_shape"] = [int(x) for x in r.shape]
                    ok = False
                else:
                    diff = np.abs(arr.astype(np.float32) - r.astype(np.float32))
                    row["reference_status"] = "ok"
                    row["max_abs"] = float(diff.max()) if diff.size else 0.0
                    row["mean_abs"] = float(diff.mean()) if diff.size else 0.0
                    row["allclose"] = bool(np.allclose(arr, r, rtol=args.rtol, atol=args.atol))
                    if not row["allclose"]:
                        ok = False
        rows.append(row)
    if ref:
        missing_candidate = [name for name in ref.keys() if name not in cand]
    else:
        missing_candidate = []
    if missing_candidate:
        ok = False
    report = {
        "schema": "onnx-splitpoint/output-dump-validation",
        "schema_version": 1,
        "ok": ok,
        "candidate": str(Path(args.candidate).expanduser().resolve()),
        "reference": str(Path(args.reference).expanduser().resolve()) if args.reference else "",
        "candidate_meta": cand_meta,
        "reference_meta": ref_meta,
        "output_count": len(rows),
        "missing_candidate_outputs": missing_candidate,
        "outputs": rows,
    }
    out = Path(args.out).expanduser().resolve() if args.out else Path(args.candidate).expanduser().resolve().parent / "output_dump_validation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": ok, "out": str(out), "output_count": len(rows)}, indent=2))
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
