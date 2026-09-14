#!/usr/bin/env python3
"""Recover an exact, read-only build-evidence index from a B5 run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.build_evidence import (  # noqa: E402
    BuildEvidenceError,
    create_build_evidence_index,
    validate_external_output_path,
    verify_build_evidence_index,
)
from onnx_splitpoint_tool.runtime_evidence import (  # noqa: E402
    runtime_evidence_index,
    write_runtime_evidence_index,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover exact deterministic compiler outcomes without changing "
            "the source EvaluationRun. Runtime evidence remains a separate ledger."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create", help="Create an exclusive build-evidence index outside the run."
    )
    create.add_argument("--run-dir", required=True)
    create.add_argument("--workflow-log")
    create.add_argument("--out", required=True)
    create.add_argument(
        "--runtime-out",
        help=(
            "Optionally create a separate empty runtime ledger. The build "
            "recovery never promotes compile evidence to runtime evidence."
        ),
    )
    verify = subparsers.add_parser(
        "verify", help="Verify an index and rehash all positive artifacts."
    )
    verify.add_argument("--index", required=True)
    verify.add_argument("--run-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            # Admit every requested output before the first write. In
            # particular, a refused runtime ledger must not leave a build
            # index behind or mutate the historical source run.
            build_output = validate_external_output_path(
                source_run=args.run_dir,
                output=args.out,
                label="output",
            )
            runtime_output_path = None
            if args.runtime_out:
                runtime_output_path = validate_external_output_path(
                    source_run=args.run_dir,
                    output=args.runtime_out,
                    label="runtime_output",
                    distinct_from=build_output,
                )
            payload = create_build_evidence_index(
                run_dir=args.run_dir,
                workflow_log=args.workflow_log,
                output=build_output,
            )
            runtime_output = None
            if runtime_output_path is not None:
                runtime_payload = runtime_evidence_index(
                    source_label=Path(args.run_dir).name
                )
                write_runtime_evidence_index(
                    runtime_output_path, runtime_payload
                )
                runtime_output = str(runtime_output_path)
            result = {
                "schema": "onnx-splitpoint/build-evidence-recovery-result/v1",
                "schema_version": 1,
                "ok": True,
                "status": (
                    "PARTIAL"
                    if payload["unresolved_observation_count"]
                    else "PASS"
                ),
                "source_run_mutated": False,
                "build_index": str(build_output),
                "runtime_ledger": runtime_output,
                "record_count": payload["record_count"],
                "reusable_record_count": payload["reusable_record_count"],
                "unresolved_observation_count": payload[
                    "unresolved_observation_count"
                ],
                "state_counts": payload["state_counts"],
                "runtime_evidence_included": False,
            }
        else:
            verification = verify_build_evidence_index(
                args.index,
                artifact_root=args.run_dir,
            )
            result = {
                **verification,
                "status": "PASS" if verification["ok"] else "CONFLICT",
            }
    except BuildEvidenceError as exc:
        result = {
            "schema": "onnx-splitpoint/build-evidence-recovery-result/v1",
            "schema_version": 1,
            "ok": False,
            "status": "REFUSED",
            "reason_code": exc.code,
            "reason_detail": exc.detail,
            "source_run_mutated": False,
            "runtime_evidence_included": False,
        }
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
