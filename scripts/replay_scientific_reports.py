#!/usr/bin/env python3
from __future__ import annotations

"""Rebuild scientific reports from an EvaluationRun without modifying it."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onnx_splitpoint_tool import __build_id__, __version__  # noqa: E402
from onnx_splitpoint_tool.workflow.scientific_replay import (  # noqa: E402
    replay_scientific_reports,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reproject scientific reports from an existing EvaluationRun. "
            "The source run stays unchanged and the destination must be new."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Existing EvaluationRun used only as report input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "New report directory outside the source run. Defaults to a "
            "sibling named <run>_reports_reprojected_v276."
        ),
    )
    parser.add_argument(
        "--native-run-dir",
        type=Path,
        default=None,
        help=(
            "Optional separate EvaluationRun containing the exactly bound "
            "Native producer and validation evidence. It is read-only."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_dir = args.run_dir.expanduser()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            run_dir.parent
            / f"{run_dir.name}_reports_reprojected_v276"
        )
    try:
        result = replay_scientific_reports(
            run_dir,
            output_dir,
            native_run_dir=args.native_run_dir,
            tool_version=__version__,
            workflow_version=__build_id__,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "status": result.get("status"),
                "execution_mode": result.get("execution_mode"),
                "source_run_dir": result.get("source_run_dir"),
                "source_mutated": result.get("source_mutated"),
                "native_source_run_dir": result.get(
                    "native_source_run_dir"
                ),
                "native_source_is_separate": result.get(
                    "native_source_is_separate"
                ),
                "output_dir": result.get("output_dir"),
                "report_generation_status": result.get(
                    "report_generation_status"
                ),
                "evidence_completeness_status": result.get(
                    "evidence_completeness_status"
                ),
                "evidence_completeness_reasons": result.get(
                    "evidence_completeness_reasons"
                ),
                "measurement_completeness_status": result.get(
                    "measurement_completeness_status"
                ),
                "scientific_row_count": result.get(
                    "scientific_row_count"
                ),
                "ranking_method_comparison_rows": result.get(
                    "ranking_method_comparison_rows"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
