#!/usr/bin/env python3
"""Inspect split boundary tensor sizes and INT8-contract potential for a benchmark set case."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnx_splitpoint_tool.native_trt_utils import boundary_contract_summary, find_onnx_files, write_json  # noqa: E402


def maybe_parse_hef(hef: Path) -> dict:
    cli = shutil.which("hailortcli")
    if not cli or not hef.exists():
        return {"available": False, "reason": "hailortcli_or_hef_missing"}
    proc = subprocess.run([cli, "parse-hef", str(hef)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"available": True, "returncode": proc.returncode, "stdout": proc.stdout}


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect Boundary Contract for a split case.")
    ap.add_argument("--benchmark-set", required=True, type=Path)
    ap.add_argument("--case", required=True)
    ap.add_argument("--hef", type=Path, default=None, help="Optional Hailo Part1 HEF to parse.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bench = args.benchmark_set.expanduser().resolve()
    files = find_onnx_files(bench, args.case)
    if not files.get("part1") or not files.get("part2"):
        raise FileNotFoundError(f"part1/part2 ONNX not found for case {args.case}")
    report = boundary_contract_summary(files["part1"], files["part2"], files.get("full"))
    if args.hef:
        report["hef_parse"] = maybe_parse_hef(args.hef.expanduser().resolve())
    out = (args.out or (bench / "native_trt_smoke" / args.case / "boundary_contract_probe.json")).expanduser().resolve()
    write_json(out, report)
    md = out.with_suffix(".md")
    lines = [f"# Boundary contract probe: {args.case}\n", "\n"]
    lines.append(f"Part1: `{report['part1_onnx']}`\n\n")
    lines.append(f"Part2: `{report['part2_onnx']}`\n\n")
    lines.append("| Tensor | Shape | fp32 MiB | fp16 MiB | int8 MiB |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for row in report.get("boundary_size_modes") or []:
        lines.append(
            f"| `{row.get('name')}` | `{row.get('shape')}` | "
            f"{row.get('mib_float32', 0):.3f} | {row.get('mib_float16', 0):.3f} | {row.get('mib_int8', 0):.3f} |\n"
        )
    lines.append("\n")
    if report.get("quantized_int8_candidate"):
        lines.append("**INT8 candidate:** yes, but only safe with an explicit producer/consumer quantization contract.\n")
    else:
        lines.append("**INT8 candidate:** not a simple single-feature-input case.\n")
    md.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(out), "markdown": str(md), "feature_input_count": report.get("feature_input_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
