#!/usr/bin/env python3
"""Build and benchmark a native TensorRT engine for an ONNX model or split Part2.

This is a Phase-2 smoke tool. It uses trtexec rather than ORT TensorRT EP, so it can
be run on the Jetson/NX target without changing the full evaluation runner yet.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import onnx
    from onnx import TensorProto
except Exception as exc:  # pragma: no cover
    print(f"[native-trt] ERROR: failed to import onnx: {exc!r}", file=sys.stderr)
    sys.exit(2)


def _find_trtexec(explicit: Optional[str] = None) -> Optional[str]:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += ["trtexec", "/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec", "/usr/local/bin/trtexec"]
    for c in candidates:
        p = shutil.which(c) if os.sep not in c else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


def _dim_to_int(dim: Any, default_batch: int = 1) -> int:
    if getattr(dim, "dim_value", 0):
        return int(dim.dim_value)
    # Keep batch at 1 and use 1 for truly dynamic dims in smoke mode.
    return int(default_batch)


def _onnx_inputs(path: Path) -> List[Dict[str, Any]]:
    m = onnx.load(str(path), load_external_data=False)
    init_names = {x.name for x in m.graph.initializer}
    out = []
    for vi in m.graph.input:
        if vi.name in init_names:
            continue
        t = vi.type.tensor_type
        shape = [_dim_to_int(d) for d in t.shape.dim]
        out.append({"name": vi.name, "shape": shape, "dtype": int(t.elem_type)})
    return out


def _find_case_onnx(suite: Path, case: str, part: str) -> Path:
    if part == "full":
        hits = sorted((suite / "models").glob("*.onnx")) if (suite / "models").exists() else []
        if not hits:
            hits = sorted(suite.glob("models/*.onnx"))
        if not hits:
            raise FileNotFoundError(f"no full ONNX found under {suite}/models")
        return hits[0]
    c = case if case.startswith("b") else f"b{int(case):03d}"
    cdir = suite / c
    if not cdir.exists():
        cdir = suite / f"b{int(case):d}"
    pats = ["*_part2_*.onnx"] if part == "part2" else ["*_part1_*.onnx"]
    for pat in pats:
        hits = sorted(cdir.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no {part} ONNX found in {cdir}")


def _parse_trtexec_output(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    patterns = {
        "throughput_qps": r"Throughput:\s*([0-9.]+)\s*qps",
        "host_latency_mean_ms": r"Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "gpu_compute_mean_ms": r"GPU Compute Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "h2d_mean_ms": r"H2D Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "d2h_mean_ms": r"D2H Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
    }
    for k, rx in patterns.items():
        m = re.search(rx, text, flags=re.IGNORECASE)
        if m:
            metrics[k] = float(m.group(1))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx", type=Path, help="Explicit ONNX path")
    src.add_argument("--suite", type=Path, help="Benchmark-set root")
    ap.add_argument("--case", default="b066", help="Case id when --suite is used")
    ap.add_argument("--part", choices=["full", "part1", "part2"], default="part2", help="Which ONNX to use from benchmark set")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--trtexec", default=None, help="Path to trtexec")
    ap.add_argument("--out", type=Path, default=None, help="Output directory")
    ap.add_argument("--engine", type=Path, default=None, help="Output engine path")
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--avg-runs", type=int, default=100)
    ap.add_argument("--workspace-mib", type=int, default=4096)
    ap.add_argument("--use-cuda-graph", action="store_true")
    ap.add_argument("--no-data-transfers", action="store_true", help="Pass --noDataTransfers to trtexec to isolate GPU compute")
    ap.add_argument("--shapes", default=None, help="Explicit trtexec shapes string, e.g. input:1x3x640x640,cat:1x512x80x80")
    args = ap.parse_args()

    trtexec = _find_trtexec(args.trtexec)
    if not trtexec:
        print("[native-trt] ERROR: trtexec not found", file=sys.stderr)
        return 2

    if args.onnx:
        onnx_path = args.onnx.expanduser().resolve()
        outdir = (args.out or (onnx_path.parent / "native_trt_smoke")).expanduser().resolve()
    else:
        suite = args.suite.expanduser().resolve()
        onnx_path = _find_case_onnx(suite, args.case, args.part)
        case_dir = onnx_path.parent if args.part != "full" else suite
        outdir = (args.out or (case_dir / "native_trt_smoke" / args.part)).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    engine = (args.engine.expanduser().resolve() if args.engine else outdir / (onnx_path.stem + f".{args.precision}.engine"))
    timing_cache = outdir / "trt_timing.cache"

    inputs = _onnx_inputs(onnx_path)
    if args.shapes:
        shapes_arg = args.shapes
    else:
        shapes_arg = ",".join(f"{x['name']}:{'x'.join(str(d) for d in x['shape'])}" for x in inputs)

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine}",
        f"--timingCacheFile={timing_cache}",
        f"--workspace={args.workspace_mib}",
        f"--duration={args.duration}",
        f"--warmUp={args.warmup}",
        f"--avgRuns={args.avg_runs}",
    ]
    if shapes_arg:
        cmd.append(f"--shapes={shapes_arg}")
    if args.precision == "fp16":
        cmd.append("--fp16")
    if args.use_cuda_graph:
        cmd.append("--useCudaGraph")
    if args.no_data_transfers:
        cmd.append("--noDataTransfers")

    print("[native-trt] command:")
    print(" ".join(str(x) for x in cmd))
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    text = proc.stdout
    (outdir / "trtexec_output.txt").write_text(text, encoding="utf-8", errors="replace")
    metrics = _parse_trtexec_output(text)
    summary = {
        "onnx": str(onnx_path),
        "engine": str(engine),
        "precision": args.precision,
        "returncode": proc.returncode,
        "trtexec": trtexec,
        "inputs": inputs,
        "shapes": shapes_arg,
        "metrics": metrics,
        "command": cmd,
    }
    (outdir / "native_trt_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[native-trt] wrote {outdir / 'native_trt_summary.json'}")
    if metrics:
        print("[native-trt] metrics:", json.dumps(metrics, sort_keys=True))
    else:
        print("[native-trt] WARNING: no metrics parsed; inspect trtexec_output.txt")
    return int(proc.returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
