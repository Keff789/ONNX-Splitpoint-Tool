#!/usr/bin/env python3
"""Create canonical Debug and/or Analysis packs for an EvaluationRun."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.filesystem_admission import require_write_target  # noqa: E402
from onnx_splitpoint_tool.workflow.debug_pack import (  # noqa: E402
    INCLUDE,
    INCLUDE_ROOT_FILES,
    REPLAY_CORE,
    create_evaluation_debug_pack,
)
from onnx_splitpoint_tool.workflow.run_discovery import (  # noqa: E402
    inspect_evaluation_run,
)

try:
    from onnx_splitpoint_tool import __version__ as TOOL_VERSION  # noqa: E402
except Exception:  # pragma: no cover
    TOOL_VERSION = "unknown"


def create_debug_pack(run_dir: Path, out_zip: Path) -> dict:
    """Compatibility adapter for the canonical installed-package builder."""

    return create_evaluation_debug_pack(run_dir, out_zip)


def create_analysis_pack(run_dir: Path, out_zip: Path) -> dict:
    from onnx_splitpoint_tool.workflow.analysis_pack import create_analysis_pack as _create

    return _create(
        run_dir,
        out_zip,
        tool_version=TOOL_VERSION,
        materialize_missing_report=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--kind", choices=["debug", "analysis", "both"], default="both")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Default: ONNX_SPLITPOINT_EXPORT_DIR or ~/Downloads",
    )
    args = parser.parse_args()

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
    if args.kind in {"analysis", "both"} and not inspection.analysis_ready:
        raise SystemExit(
            f"EvaluationRun is not eligible for an analysis pack: {run_dir}: "
            f"{','.join(inspection.reason_codes)}"
        )

    if args.out_dir:
        output_dir = Path(args.out_dir).expanduser().resolve(strict=False)
    else:
        configured = str(os.environ.get("ONNX_SPLITPOINT_EXPORT_DIR", "") or "").strip()
        output_dir = (
            Path(configured).expanduser().resolve(strict=False)
            if configured else (Path.home() / "Downloads").resolve(strict=False)
        )
    require_write_target(
        output_dir,
        operation="EvaluationRun pack export",
        minimum_free_bytes=16 * 1024 * 1024,
        minimum_free_inodes=16,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "ok": True,
        "run_dir": str(run_dir),
        "artifacts": {},
    }
    artifacts = result["artifacts"]
    assert isinstance(artifacts, dict)
    if args.kind in {"debug", "both"}:
        artifacts["debug"] = create_debug_pack(
            run_dir, output_dir / f"{run_dir.name}_debug_pack.zip"
        )
    if args.kind in {"analysis", "both"}:
        artifacts["analysis"] = create_analysis_pack(
            run_dir, output_dir / f"{run_dir.name}_analysis_pack.zip"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
