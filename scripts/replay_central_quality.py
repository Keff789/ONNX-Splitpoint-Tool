#!/usr/bin/env python3
"""Replay central quality gates from prediction JSONs without hardware."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.quality_replay import (  # noqa: E402
    OfflineQualityReplayError,
    REPLAY_CSV_NAME,
    REPLAY_OUTPUT_NAME,
    replay_evaluation_run,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate existing central-quality request/candidate/reference "
            "JSONs with the current AP50:95/AP50/AP75 algorithm. No inference "
            "runtime or accelerator is invoked."
        )
    )
    parser.add_argument("--eval-run-dir", required=True, help="Original EvaluationRun directory")
    parser.add_argument(
        "--out-dir",
        help=(
            "Separate replay output directory (default: "
            "<run.parent>/<run.name>_offline_quality_replay_v27522)"
        ),
    )
    parser.add_argument("--workers", type=int, default=4, help="Management CPU bootstrap workers")
    parser.add_argument(
        "--full-only",
        action="store_true",
        help=(
            "Replay exactly the four canonical Hailo-8/TRT@H8/Hailo-10H/TRT@H10H "
            "full identities; fail if any identity is missing or duplicated"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if int(args.workers) < 1:
        print("ERROR: --workers must be at least 1", file=sys.stderr)
        return 2
    try:
        output = replay_evaluation_run(
            args.eval_run_dir,
            out_dir=args.out_dir,
            workers=args.workers,
            full_only=args.full_only,
        )
    except OfflineQualityReplayError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    target = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (
            Path(args.eval_run_dir).expanduser().resolve().parent
            / (
                Path(args.eval_run_dir).expanduser().resolve().name
                + "_offline_quality_replay_v27522"
            )
        )
    )
    print(
        f"OFFLINE_REPLAY=PASS requests={output['request_count']} "
        f"technical_status={output['technical_status']} "
        f"all_quality_decision={output['quality_decision']} "
        f"all_decisions={output['decision_counts']} "
        f"canonical_full_only_quality_decision="
        f"{output['canonical_full_only_quality_decision']} "
        f"canonical_full_only_decisions="
        f"{output['canonical_full_only_decision_counts']} "
        "hardware_executed=false"
    )
    print(f"REPLAY_JSON={target / REPLAY_OUTPUT_NAME}")
    print(f"REPLAY_CSV={target / REPLAY_CSV_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
