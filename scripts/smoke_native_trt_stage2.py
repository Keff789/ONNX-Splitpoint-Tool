#!/usr/bin/env python3
"""Build/benchmark native TensorRT engines for a benchmark set full model and/or split Part2.

This is a Phase-2 smoke harness.  It deliberately uses trtexec first because it is
available on Jetson/TensorRT installations and gives a clean native TensorRT baseline
without changing the generic Python/ORT runner yet.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the repository checkout without installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnx_splitpoint_tool.native_trt_utils import (  # noqa: E402
    boundary_contract_summary,
    build_trtexec_build_cmd,
    build_trtexec_run_cmd,
    find_onnx_files,
    find_trtexec,
    load_io_info,
    parse_shape_overrides,
    parse_trtexec_output,
    run_cmd,
    shape_arg_from_io,
    write_json,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Native TensorRT smoke test for full/Part2 ONNX in a split benchmark set.")
    ap.add_argument("--benchmark-set", required=True, type=Path, help="Benchmark set root, e.g. .../yolov7_paper_benchmark_20260625_160757")
    ap.add_argument("--case", default="", help="Split case id, e.g. b066 or 66. Required for --which part2/both.")
    ap.add_argument("--which", choices=["full", "part2", "both"], default="both")
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--shape", action="append", default=[], help="Input shape override name:1x3x640x640. Can be repeated.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--trtexec", default="")
    ap.add_argument("--workspace-mib", type=int, default=4096)
    ap.add_argument("--duration-s", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--warmup-ms", type=int, default=500)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--no-build", action="store_true", help="Do not rebuild engine if it already exists.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--extra-build-arg", action="append", default=[])
    ap.add_argument("--extra-run-arg", action="append", default=[])
    args = ap.parse_args()

    bench = args.benchmark_set.expanduser().resolve()
    if not bench.exists():
        raise FileNotFoundError(bench)
    trtexec = find_trtexec(args.trtexec)
    shapes_override = parse_shape_overrides(args.shape)
    out_dir = (args.out_dir or (bench / "native_trt_smoke" / (args.case or "full"))).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_onnx_files(bench, args.case or None)
    targets = []
    if args.which in {"full", "both"}:
        if not files.get("full"):
            raise FileNotFoundError("Could not locate full ONNX below benchmark set")
        targets.append(("full", files["full"]))
    if args.which in {"part2", "both"}:
        if not args.case:
            raise ValueError("--case is required for Part2")
        if not files.get("part2"):
            raise FileNotFoundError(f"Could not locate Part2 ONNX for case {args.case}")
        targets.append(("part2", files["part2"]))

    report = {
        "benchmark_set": str(bench),
        "case": args.case or None,
        "precision": args.precision,
        "trtexec": trtexec,
        "targets": [],
    }

    if args.case and files.get("part1") and files.get("part2"):
        try:
            contract = boundary_contract_summary(files["part1"], files["part2"], files.get("full"))
            write_json(out_dir / "boundary_contract_probe.json", contract)
            report["boundary_contract_probe"] = str(out_dir / "boundary_contract_probe.json")
        except Exception as exc:
            report["boundary_contract_error"] = f"{type(exc).__name__}: {exc}"

    for label, onnx_path in targets:
        assert onnx_path is not None
        io_info = load_io_info(onnx_path)
        shapes = shape_arg_from_io(io_info, shapes_override)
        engine = out_dir / f"{label}_{args.precision}.engine"
        timing_cache = out_dir / "native_trt_timing.cache"
        build_cmd = build_trtexec_build_cmd(
            trtexec,
            onnx_path,
            engine,
            shapes,
            precision=args.precision,
            workspace_mib=args.workspace_mib,
            timing_cache=timing_cache,
            extra=args.extra_build_arg,
        )
        run_cmdline = build_trtexec_run_cmd(
            trtexec,
            engine,
            duration_s=args.duration_s,
            warmup_ms=args.warmup_ms,
            iterations=args.iterations,
            extra=args.extra_run_arg,
        )
        rec = {
            "label": label,
            "onnx": str(onnx_path),
            "engine": str(engine),
            "io": io_info,
            "shapes_arg": shapes,
            "build_cmd": build_cmd,
            "run_cmd": run_cmdline,
        }
        if args.dry_run:
            rec["status"] = "dry_run"
        else:
            if not args.no_build or not engine.exists():
                print(f"[native-trt] building {label}: {' '.join(build_cmd)}", flush=True)
                build_res = run_cmd(build_cmd, timeout=args.timeout_s)
                rec["build"] = build_res
                (out_dir / f"{label}_build.log").write_text(build_res.get("stdout", ""), encoding="utf-8", errors="replace")
                if build_res["returncode"] != 0:
                    rec["status"] = "build_failed"
                    report["targets"].append(rec)
                    continue
            print(f"[native-trt] benchmarking {label}: {' '.join(run_cmdline)}", flush=True)
            run_res = run_cmd(run_cmdline, timeout=args.timeout_s)
            rec["run"] = run_res
            rec["metrics"] = parse_trtexec_output(run_res.get("stdout", ""))
            (out_dir / f"{label}_run.log").write_text(run_res.get("stdout", ""), encoding="utf-8", errors="replace")
            rec["status"] = "ok" if run_res["returncode"] == 0 else "run_failed"
        report["targets"].append(rec)

    write_json(out_dir / "native_trt_smoke_report.json", report)
    print(json.dumps({
        "report": str(out_dir / "native_trt_smoke_report.json"),
        "summary": [
            {"label": r.get("label"), "status": r.get("status"), **(r.get("metrics") or {})}
            for r in report["targets"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
