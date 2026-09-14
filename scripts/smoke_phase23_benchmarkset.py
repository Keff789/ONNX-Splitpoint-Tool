#!/usr/bin/env python3
"""Run Phase-2/3 smoke diagnostics on an existing benchmark set.

Phase 2: Native TensorRT Part2 benchmark with trtexec.
Phase 3: Boundary contract/transfer-size inspection for float32/fp16/int8 modes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True, type=Path)
    ap.add_argument("--cases", nargs="+", default=["b066"], help="Case ids, e.g. b044 b066")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--also-full", action="store_true")
    ap.add_argument("--no-data-transfers", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    failures = 0
    for case in args.cases:
        print(f"\n=== Boundary contract: {case} ===")
        rc = subprocess.call([sys.executable, str(root / "inspect_boundary_contract.py"), "--suite", str(args.suite), "--case", case])
        failures += int(rc != 0)
        print(f"\n=== Native TensorRT Part2: {case} ===")
        cmd = [
            sys.executable, str(root / "smoke_native_trt_engine.py"),
            "--suite", str(args.suite),
            "--case", case,
            "--part", "part2",
            "--precision", args.precision,
            "--duration", str(args.duration),
            "--warmup", str(args.warmup),
        ]
        if args.no_data_transfers:
            cmd.append("--no-data-transfers")
        rc = subprocess.call(cmd)
        failures += int(rc != 0)
    if args.also_full:
        print("\n=== Native TensorRT Full model ===")
        rc = subprocess.call([
            sys.executable, str(root / "smoke_native_trt_engine.py"),
            "--suite", str(args.suite),
            "--case", args.cases[0],
            "--part", "full",
            "--precision", args.precision,
            "--duration", str(args.duration),
            "--warmup", str(args.warmup),
        ])
        failures += int(rc != 0)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
