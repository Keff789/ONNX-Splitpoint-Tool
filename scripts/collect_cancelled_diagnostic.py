#!/usr/bin/env python3
"""Collect a verified, control-plane-first archive from an interrupted run.

This command is intentionally collection-only.  Request cancellation through
the normal Evaluation Workflow UI/CLI first; after a crash or power outage it
is also safe to run directly against the interrupted EvaluationRun directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.workflow.debug_pack import (  # noqa: E402
    DEFAULT_MAX_SMALL_FILE_BYTES,
    DEFAULT_TAIL_BYTES,
    create_evaluation_debug_pack,
)
from onnx_splitpoint_tool.workflow.debug_pack_policy import (  # noqa: E402
    is_cancelled_diagnostic_priority,
)
from onnx_splitpoint_tool.workflow.run_discovery import (  # noqa: E402
    inspect_evaluation_run,
)
from onnx_splitpoint_tool.workflow.zip_utils import (  # noqa: E402
    iter_safe_pack_files,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--eval-run-dir")
    source.add_argument("--run-dir", dest="eval_run_dir")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--max-small-file-bytes",
        type=int,
        default=DEFAULT_MAX_SMALL_FILE_BYTES,
    )
    parser.add_argument("--tail-bytes", type=int, default=DEFAULT_TAIL_BYTES)
    return parser


def _priority_members(run_dir: Path) -> list[str]:
    members: list[str] = []
    for path in iter_safe_pack_files(run_dir, run_dir):
        relative = path.relative_to(run_dir).as_posix()
        if is_cancelled_diagnostic_priority(relative):
            members.append(relative)
    return sorted(set(members))


def collect_cancelled_diagnostic(
    run_dir: Path,
    output: Path,
    *,
    max_small_file_bytes: int = DEFAULT_MAX_SMALL_FILE_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
) -> dict[str, object]:
    """Create the archive and verify all priority members and ZIP metadata."""

    priority = _priority_members(run_dir)
    result = create_evaluation_debug_pack(
        run_dir,
        output,
        max_small_file_bytes=max(0, int(max_small_file_bytes)),
        tail_bytes=max(1, int(tail_bytes)),
        source_selection_policy="cancelled_diagnostic_explicit_run",
    )
    with zipfile.ZipFile(output, "r") as archive:
        names = set(archive.namelist())
        missing = sorted(set(priority) - names)
        invalid_timestamps = sorted(
            info.filename
            for info in archive.infolist()
            if not 1980 <= int(info.date_time[0]) <= 2107
        )
        bad_member = archive.testzip()
    if missing:
        raise RuntimeError(
            "cancelled_diagnostic_priority_members_missing:"
            + ",".join(missing)
        )
    if invalid_timestamps:
        raise RuntimeError(
            "cancelled_diagnostic_zip_timestamp_out_of_range:"
            + ",".join(invalid_timestamps)
        )
    if bad_member:
        raise RuntimeError(f"cancelled_diagnostic_zip_crc_failed:{bad_member}")
    return {
        **dict(result),
        "collector": "prioritized_cancelled_diagnostic",
        "source_run_dir": str(run_dir),
        "priority_source_members": priority,
        "priority_member_count": len(priority),
        "priority_complete": True,
        "zip_timestamp_contract": "clamped_1980_2107",
    }


def main() -> int:
    args = _parser().parse_args()
    raw_run = Path(args.eval_run_dir).expanduser()
    if raw_run.is_symlink():
        raise SystemExit(f"EvaluationRun source must not be a symlink: {raw_run}")
    run_dir = raw_run.resolve(strict=True)
    inspection = inspect_evaluation_run(run_dir)
    if not inspection.identified:
        raise SystemExit(
            f"No identifiable EvaluationRun at {run_dir}: "
            f"{','.join(inspection.reason_codes)}"
        )
    output = (
        Path(args.out).expanduser()
        if args.out
        else Path.home() / "Downloads"
        / f"{run_dir.name}_cancelled_diagnostic.zip"
    ).resolve(strict=False)
    result = collect_cancelled_diagnostic(
        run_dir,
        output,
        max_small_file_bytes=args.max_small_file_bytes,
        tail_bytes=args.tail_bytes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
