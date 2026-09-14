#!/usr/bin/env python3
"""Convert a runner .npz output dump/baseline into the common output-dump manifest.

This is the bridge between the generated Python runner and native FIFO runner:
- generic runner can already write .npz caches such as baseline_full_cpu_outputs.npz
- native FIFO writes binary tensors + manifest
This tool converts NPZ to the same manifest/bin layout so one validator can handle both.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np


def safe_name(x: str) -> str:
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x or "tensor").strip("_")
    return x or "tensor"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="generic_runner_npz")
    ap.add_argument("--variant", default="full")
    args = ap.parse_args()
    npz = Path(args.npz).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npz, allow_pickle=False)
    outputs = []
    for name in data.files:
        if name == "__meta__":
            continue
        arr = np.asarray(data[name])
        fname = safe_name(name) + ".bin"
        arr.astype(arr.dtype, copy=False).tofile(out_dir / fname)
        outputs.append({
            "name": name,
            "file": fname,
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "bytes": int(arr.nbytes),
        })
    manifest = {
        "schema": "onnx-splitpoint/output-dump",
        "schema_version": 1,
        "producer": args.source,
        "variant": args.variant,
        "source_npz": str(npz),
        "outputs": outputs,
    }
    out = out_dir / "output_dump_manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": True, "manifest": str(out), "output_count": len(outputs)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
