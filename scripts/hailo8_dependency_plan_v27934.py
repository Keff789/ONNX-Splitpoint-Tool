#!/usr/bin/env python3
"""Read-only Hailo8 dependency plan, or separately reviewed offline staging.

No downloads and no pip operations. Run from the management Tool venv. Planning
requires the explicitly selected Hailo8 vendor venv's bin/python. Package names
are selected individually after reviewing current TensorFlow metadata. This
command never runs on ordinary GUI start, build preflight, cache HIT or install.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_splitpoint_tool.hailo_dependency_plan import (
    build_plan, collect_inventory, stage_reviewed_plan,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Read current metadata and write reviewable JSON only")
    plan.add_argument("--python", required=True, dest="selected_python")
    plan.add_argument("--target", required=True, help="New sibling directory named hailo8_cuda_...")
    plan.add_argument("--package", action="append", default=[], help="Explicit current TF CUDA dependency name (repeatable)")
    plan.add_argument("--wheel", action="append", default=[], help="Already available exact local NVIDIA wheel (repeatable)")
    plan.add_argument("--output", required=True)
    stage = commands.add_parser("stage-reviewed", help="Explicit separate action: extract reviewed local wheels into private target")
    stage.add_argument("--plan", required=True)
    stage.add_argument("--reviewed-plan-sha256", required=True)
    stage.add_argument("--confirm-private-hailo8-addition", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = build_plan(collect_inventory(args.selected_python), args.target,
                                packages=args.package, wheels=args.wheel)
            output = Path(args.output).expanduser().absolute()
            data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
            # A review output is never silently overwritten or conflated with an old plan.
            with output.open("xb") as stream:
                stream.write(data)
            print("HAILO8_DEPENDENCY_PLAN=" + str(output))
            print("PLAN_SHA256=" + hashlib.sha256(data).hexdigest())
            print("PLAN_STATUS=" + result["status"])
            print("HAILO8_GPU_READINESS=NOT_PROVEN")
            return 2 if result["status"] == "BLOCKED" else 0
        plan_path = Path(args.plan).expanduser().absolute()
        manifest = stage_reviewed_plan(json.loads(plan_path.read_text()),
                                      expected_plan_sha256=args.reviewed_plan_sha256,
                                      plan_file=plan_path)
        print("HAILO8_PRIVATE_OVERLAY_MANIFEST=" + str(manifest))
        print("HAILO8_GPU_READINESS=NOT_PROVEN")
        print("ROLLBACK=Deselect manifest; remove only its private parent after all child jobs exit.")
        return 0
    except Exception as exc:
        print("HAILO8_DEPENDENCY_STATUS=BLOCKED", file=sys.stderr)
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
