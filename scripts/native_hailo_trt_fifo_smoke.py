#!/usr/bin/env python3
"""Build and run the native HailoRT→TensorRT FIFO fastpath.

This script is intentionally artifact-driven. It does not regenerate models; it
uses an existing benchmark set with a Hailo Part1 HEF and a native TensorRT Part2
engine. The first target is the YOLOv7 b066 paper-fingerprint split, but the
runner is generic for one-input/one-output Hailo boundaries.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _native_src() -> Path:
    return _tool_root() / "onnx_splitpoint_tool" / "native_fastpath" / "hailo_trt_fifo_fastpath.cpp"


def _case_dir(bs: Path, case: str) -> Path:
    p = bs / case
    if not p.exists():
        raise FileNotFoundError(f"case directory not found: {p}")
    return p


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_hef(bs: Path, case: str, hw_arch: str) -> Path:
    c = _case_dir(bs, case)
    pats = [
        c / "hailo" / hw_arch / "part1" / "compiled.hef",
        c / "hailo" / hw_arch / "part1" / f"{case}_part1.hef",
    ]
    p = _first_existing(pats)
    if p:
        return p
    hits = sorted((c / "hailo").glob(f"**/{hw_arch}/part1/*.hef")) if (c / "hailo").exists() else []
    hits += sorted(c.glob(f"**/{hw_arch}/part1/*.hef"))
    hits += sorted(c.glob("**/part1/*.hef"))
    hits = [p for p in hits if ".hailo-generations" not in p.parts and p.is_file()]
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No Hailo part1 HEF found for case={case} hw_arch={hw_arch} under {c}")


def find_engine(bs: Path, case: str, precision: str) -> Path:
    roots = [
        bs / "native_trt" / case / "part2" / precision,
        _case_dir(bs, case) / "native_trt" / "part2" / precision,
    ]
    names = [f"part2_{precision}.engine", "part2.engine", "model.engine"]
    for r in roots:
        for n in names:
            p = r / n
            if p.exists():
                return p
        hits = sorted(r.glob("*.engine")) if r.exists() else []
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No native TRT part2 engine found for case={case} precision={precision}. Build it first with scripts/native_trt_from_benchmarkset.py")


def _run(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None, check: bool = True) -> subprocess.CompletedProcess:
    print("[native-fifo] $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=check)


def _try_cmd(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    except Exception:
        return False


def build_binary(build_dir: Path, force: bool = False) -> Path:
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "hailo_trt_fifo_fastpath"
    src = _native_src()
    if exe.exists() and not force:
        return exe
    if not src.exists():
        raise FileNotFoundError(src)

    # Prefer direct g++ because it is more transparent on Jetsons where CMake may
    # not have a FindHailoRT module.
    cmd = [
        os.environ.get("CXX", "g++"),
        "-O3", "-std=c++17",
        str(src),
        "-o", str(exe),
        "-I/usr/local/cuda/include",
        "-I/usr/include/aarch64-linux-gnu",
        "-L/usr/local/cuda/lib64",
        "-L/usr/lib/aarch64-linux-gnu",
        "-lhailort", "-lnvinfer", "-lnvinfer_plugin", "-lcudart", "-pthread",
    ]
    # Some HailoRT installations publish pkg-config metadata. Append it if
    # available to handle non-standard include/lib locations.
    if _try_cmd(["pkg-config", "--exists", "hailort"]):
        try:
            cflags = subprocess.check_output(["pkg-config", "--cflags", "hailort"], text=True).strip().split()
            libs = subprocess.check_output(["pkg-config", "--libs", "hailort"], text=True).strip().split()
            cmd = cmd[:3] + cflags + cmd[3:] + libs
        except Exception:
            pass
    _run(cmd)
    return exe


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-set", required=True, type=Path)
    ap.add_argument("--case", default="b066")
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--precision", default="uint8_cast_fp16", help="Native TRT part2 precision/tag, e.g. uint8_cast_fp16 or fp16")
    ap.add_argument("--hef", type=Path, default=None)
    ap.add_argument("--engine", type=Path, default=None)
    ap.add_argument("--build-dir", type=Path, default=None)
    ap.add_argument("--force-build", action="store_true")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--queue-depth", type=int, default=2)
    ap.add_argument("--device-id", default="")
    ap.add_argument("--no-copy-outputs", action="store_true")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bs = args.benchmark_set.resolve()
    case = args.case
    hef = args.hef.resolve() if args.hef else find_hef(bs, case, args.hw_arch).resolve()
    engine = args.engine.resolve() if args.engine else find_engine(bs, case, args.precision).resolve()
    build_dir = args.build_dir or (bs / "native_fifo" / "build")
    out_json = args.out_json or (bs / case / "native_fifo" / f"hailo_trt_fifo_{args.hw_arch}_{args.precision}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "benchmark_set": str(bs),
        "case": case,
        "hw_arch": args.hw_arch,
        "hef": str(hef),
        "engine": str(engine),
        "out_json": str(out_json),
    }, indent=2), flush=True)

    if args.dry_run:
        return 0

    exe = build_binary(build_dir.resolve(), force=args.force_build)
    cmd = [
        str(exe),
        "--hef", str(hef),
        "--engine", str(engine),
        "--out-json", str(out_json),
        "--frames", str(max(1, args.frames)),
        "--warmup", str(max(0, args.warmup)),
        "--queue-depth", str(max(1, args.queue_depth)),
    ]
    if args.device_id:
        cmd += ["--device-id", args.device_id]
    if args.no_copy_outputs:
        cmd += ["--no-copy-outputs"]
    rc = _run(cmd, cwd=bs, check=False).returncode
    if out_json.exists():
        try:
            print(out_json.read_text())
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
