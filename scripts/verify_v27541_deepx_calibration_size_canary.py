#!/usr/bin/env python3
"""Verify the v2.75.41 DeepX ResNet50 B500/B1000 canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.deepx.calibration_size_canary import (
    compare_run_directories,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed verification of the ResNet50 DeepX imagenet_mean_std "
            "calibration-size canary (B500 versus v2.75.41 B1000)."
        ),
    )
    parser.add_argument("--baseline-b500", required=True, help="B500 EvaluationRun")
    parser.add_argument("--candidate-b1000", required=True, help="B1000 EvaluationRun")
    parser.add_argument(
        "--baseline-calibration-manifest",
        help=(
            "Frozen B500 manifest. Required when the run/profile reference no "
            "longer resolves to bytes matching the B500 receipt."
        ),
    )
    parser.add_argument(
        "--candidate-calibration-manifest",
        help=(
            "Frozen B1000 manifest. Required when the run/profile reference no "
            "longer resolves to bytes matching the B1000 receipt."
        ),
    )
    parser.add_argument("--out", required=True, help="Output comparison JSON")
    args = parser.parse_args()

    payload = compare_run_directories(
        baseline_dir=args.baseline_b500,
        candidate_dir=args.candidate_b1000,
        baseline_calibration_manifest=args.baseline_calibration_manifest,
        candidate_calibration_manifest=args.candidate_calibration_manifest,
    )
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({
        "status": payload["status"],
        "b500_is_subset_of_b1000": payload["calibration_cohort"][
            "b500_is_subset_of_b1000"
        ],
        "b1000_vs_trt_decision": payload[
            "authoritative_b1000_vs_setup_local_trt_guardrail"
        ]["decision"],
        "b1000_quality_pass": payload["b1000_quality_pass"],
        "standard_plus_ready": payload["standard_plus_ready"],
        "next_step": payload["next_step"],
        "output": str(output.resolve()),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
