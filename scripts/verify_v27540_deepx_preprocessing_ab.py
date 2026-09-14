#!/usr/bin/env python3
"""Verify and compare the two v2.75.40 DeepX preprocessing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.deepx.preprocessing_ab import compare_run_directories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-a", required=True, help="current_scale_only EvaluationRun")
    parser.add_argument("--arm-b", required=True, help="imagenet_mean_std EvaluationRun")
    parser.add_argument("--out", required=True, help="Comparison JSON")
    parser.add_argument("--expected-records", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    args = parser.parse_args()
    payload = compare_run_directories(
        arm_a_dir=args.arm_a,
        arm_b_dir=args.arm_b,
        expected_records=int(args.expected_records),
        bootstrap_repetitions=int(args.bootstrap_repetitions),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "paired": payload["paired"],
        "standard_plus_ready": payload["standard_plus_ready"],
        "next_step": payload["next_step"],
        "output": str(output.resolve()),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

