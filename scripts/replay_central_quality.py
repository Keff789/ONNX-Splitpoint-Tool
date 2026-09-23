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
    parser.add_argument("--workers", type=int, help="Bootstrap workers (profile setting, otherwise 4)")
    parser.add_argument("--profile", help="Normal evaluation profile for execution settings; saved request budgets remain unchanged")
    parser.add_argument("--engine", choices=("legacy", "optimized_coco_v1"))
    parser.add_argument("--block-repetitions", type=int, help="Execution block size, never the scientific repetition budget")
    parser.add_argument("--checkpoint-blocks", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prepared-cache-limit-mib", type=int)
    parser.add_argument("--capture-draws", action="store_true", help="Retain diagnostic draws without changing saved budgets")
    parser.add_argument("--row-index", action="append", type=int,
                        help="Repeat for zero-based indices of central_quality_summary.json results")
    parser.add_argument("--producer-reference-status", action="append", default=[], metavar="INDEX=PATH",
                        help="Explicit admission of a cancelled historical statistics row from completed producer records and its model-bound reference status")
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
    try:
        producer_statuses = {}
        for entry in args.producer_reference_status:
            index, separator, path = entry.partition("=")
            if not separator or not index.isdecimal() or not path.strip() or int(index) in producer_statuses:
                raise OfflineQualityReplayError("--producer-reference-status requires unique INDEX=PATH entries")
            producer_statuses[int(index)] = path
        statistics = {key: value for key, value in {
            "engine": args.engine, "block_repetitions": args.block_repetitions,
            "checkpoint_blocks": args.checkpoint_blocks,
            "prepared_cache_limit_mib": args.prepared_cache_limit_mib,
        }.items() if value is not None}
        if args.capture_draws:
            statistics["capture_draws"] = True
        output = replay_evaluation_run(
            args.eval_run_dir,
            out_dir=args.out_dir,
            workers=args.workers,
            full_only=args.full_only,
            row_indices=args.row_index,
            profile=args.profile,
            statistics=statistics,
            producer_reference_statuses=producer_statuses,
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
    print(f"SELECTION={output['selection_scope']} rows={output['selected_source_row_indices']}")
    for kind, path in output["scientific_report_paths"].items():
        print(f"SCIENTIFIC_{kind.upper()}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
