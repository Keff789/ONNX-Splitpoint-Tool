#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""WSL-side helper for building a HEF (translate + optimize + compile).

This script is invoked from the Windows GUI via `wsl.exe ... python3 ...`.
It prints a single machine-readable result line with a marker prefix.

We keep this as a small wrapper so that:
- the GUI doesn't need direct access to `hailo_sdk_client`
- all heavy logic lives in `onnx_splitpoint_tool.hailo_backend`
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
    # <repo>/onnx_splitpoint_tool/wsl_hailo_build_hef.py -> parents[1] == <repo>
    return Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--onnx", required=True, help="Path to ONNX model (WSL path)")
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--net-name", default=None)
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--fixup", default="1", choices=["0", "1"], help="Apply ONNX fixups")
    ap.add_argument("--add-conv-defaults", default="1", choices=["0", "1"], help="Add Conv defaults during fixup")
    ap.add_argument(
        "--disable-rt-metadata-extraction",
        default="1",
        choices=["0", "1"],
        help="Pass disable_rt_metadata_extraction to DFC translate",
    )

    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--calib-count", type=int, default=64)
    ap.add_argument("--calib-batch-size", type=int, default=8)
    ap.add_argument("--force", default="0", choices=["0", "1"], help="Overwrite compiled.hef if it exists")
    ap.add_argument("--keep-artifacts", default="0", choices=["0", "1"], help="Save parsed/quantized HAR")

    args = ap.parse_args()

    # Ensure we can import the tool package even if it isn't installed in the venv.
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from onnx_splitpoint_tool.hailo_backend import hailo_build_hef  # local import

    t0 = time.time()
    onnx_path = Path(args.onnx)
    outdir = Path(args.outdir) if args.outdir else None
    net_name = args.net_name or onnx_path.stem

    res = hailo_build_hef(
        onnx_path,
        hw_arch=str(args.hw_arch),
        net_name=str(net_name),
        outdir=str(outdir) if outdir is not None else None,
        fixup=(args.fixup == "1"),
        add_conv_defaults=(args.add_conv_defaults == "1"),
        disable_rt_metadata_extraction=(args.disable_rt_metadata_extraction == "1"),
        opt_level=int(args.opt_level),
        calib_dir=str(args.calib_dir) if args.calib_dir else None,
        calib_count=int(args.calib_count),
        calib_batch_size=int(args.calib_batch_size),
        force=(args.force == "1"),
        keep_artifacts=(args.keep_artifacts == "1"),
    )

    payload = asdict(res)
    payload["elapsed_s"] = float(payload.get("elapsed_s") or (time.time() - t0))
    payload["backend"] = "wsl"
    print(RESULT_MARKER + json.dumps(payload, ensure_ascii=False))

    return 0 if res.ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
