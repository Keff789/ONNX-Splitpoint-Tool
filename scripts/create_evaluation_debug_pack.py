#!/usr/bin/env python3
"""Create the canonical compact Debug Pack for an EvaluationRun."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.workflow.debug_pack import (  # noqa: E402
    CENTRAL_DESCRIPTOR_MAX_FILE_BYTES,
    CENTRAL_DESCRIPTOR_MAX_REQUESTS,
    CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES,
    DEFAULT_MAX_SMALL_FILE_BYTES,
    DEFAULT_TAIL_BYTES,
    INCLUDE,
    PROVENANCE_EVIDENCE_NAMES,
    REPLAY_CORE,
    create_evaluation_debug_pack,
    should_include_debug_file,
)
from onnx_splitpoint_tool.workflow.run_discovery import (  # noqa: E402
    inspect_evaluation_run,
)

# Compatibility names for callers that imported the former standalone CLI.
CENTRAL_REPLAY_MAX_REQUESTS = CENTRAL_DESCRIPTOR_MAX_REQUESTS
CENTRAL_REPLAY_MAX_FILE_BYTES = CENTRAL_DESCRIPTOR_MAX_FILE_BYTES
CENTRAL_REPLAY_MAX_TOTAL_BYTES = CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES


def _should_include(
    run_dir: Path,
    path: Path,
    max_bytes: int,
    *,
    probe_include_raw: bool,
    central_replay_input: bool = False,
) -> tuple[bool, str]:
    """Compatibility wrapper around the installed package policy."""

    return should_include_debug_file(
        Path(run_dir),
        Path(path),
        max_bytes,
        probe_include_raw=probe_include_raw,
        central_request_descriptor=central_replay_input,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--max-small-file-bytes",
        type=int,
        default=DEFAULT_MAX_SMALL_FILE_BYTES,
    )
    parser.add_argument("--tail-bytes", type=int, default=DEFAULT_TAIL_BYTES)
    return parser


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
    result = create_evaluation_debug_pack(
        run_dir,
        Path(args.out).expanduser() if args.out else None,
        max_small_file_bytes=max(0, int(args.max_small_file_bytes)),
        tail_bytes=max(1, int(args.tail_bytes)),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
