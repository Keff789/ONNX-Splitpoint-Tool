#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""WSL-side helper for Hailo parse-only checks.

Why this exists
---------------
The Hailo Dataflow Compiler (DFC) is only available as a Linux Python wheel.
Many users run the GUI on Windows, but keep the DFC in a WSL2 virtualenv.

The Windows GUI can invoke this script via `wsl.exe ... python3 wsl_hailo_check.py`.
The script prints a single machine-readable result line with a marker prefix.

Important
---------
The Hailo SDK/DFC can emit log lines to stdout/stderr. Therefore we do not try
to keep stdout "clean". The caller should search for the marker line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


RESULT_MARKER = "__SPLITPOINT_HAILO_RESULT__"


def _repo_root() -> Path:
    # <repo>/onnx_splitpoint_tool/wsl_hailo_check.py -> parents[1] == <repo>
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--onnx", required=True, help="Path to ONNX model (WSL path)")
    ap.add_argument("--hw-arch", default="hailo8", choices=["hailo8", "hailo8l", "hailo8r"])
    ap.add_argument("--net-name", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--fixup", default="1", choices=["0", "1"], help="Apply ONNX fixups")
    ap.add_argument("--add-conv-defaults", default="1", choices=["0", "1"], help="Add Conv defaults during fixup")
    ap.add_argument("--save-har", default="0", choices=["0", "1"], help="Save parsed.har to outdir")
    ap.add_argument(
        "--disable-rt-metadata-extraction",
        default="1",
        choices=["0", "1"],
        help="Pass disable_rt_metadata_extraction to DFC translate",
    )
    args = ap.parse_args()

    # Ensure we can import the tool package even if it isn't installed in the venv.
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from onnx_splitpoint_tool.hailo_backend import hailo_parse_check  # local import

    t0 = time.time()
    onnx_path = Path(args.onnx)
    outdir = Path(args.outdir) if args.outdir else None
    net_name = args.net_name or onnx_path.stem

    res = hailo_parse_check(
        onnx_path,
        hw_arch=str(args.hw_arch),
        net_name=str(net_name),
        outdir=str(outdir) if outdir is not None else None,
        fixup=(args.fixup == "1"),
        add_conv_defaults=(args.add_conv_defaults == "1"),
        save_har=(args.save_har == "1"),
        disable_rt_metadata_extraction=(args.disable_rt_metadata_extraction == "1"),
    )

    payload = asdict(res)
    payload["elapsed_s"] = float(payload.get("elapsed_s") or (time.time() - t0))
    payload["backend"] = "wsl"
    # Single marker line - caller extracts this.
    print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False))

    return 0 if res.ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
