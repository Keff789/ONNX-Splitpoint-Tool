#!/usr/bin/env python3
"""Replay the Scientific Report row contract without hardware execution.

The replay consumes an existing ``scientific_report.json`` and its exported
``native_energy_observations.json``.  It discards previously projected energy
rows, retains only the performance population, reprojects every Native energy
attempt with the current code and recomputes the role-scoped summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.native_energy_reporting import (  # noqa: E402
    scientific_energy_rows,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import (  # noqa: E402
    NATIVE_ENERGY_ROW_ROLE,
    PERFORMANCE_ROW_ROLE,
    _refresh_scientific_summary,
    _row_status,
)

REPLAY_SCHEMA = "onnx-splitpoint/scientific-report-contract-replay"
REPLAY_SCHEMA_VERSION = 1


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return None
    text = str(value).strip().lower()
    if text in {
        "1",
        "true",
        "yes",
        "ok",
        "pass",
        "passed",
        "valid",
        "claim_ok",
        "eligible",
    }:
        return True
    if text in {
        "0",
        "false",
        "no",
        "fail",
        "failed",
        "invalid",
        "error",
    }:
        return False
    return None


def _native_observation_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        source = payload
    elif isinstance(payload, Mapping):
        source = (
            payload.get("observations")
            or payload.get("rows")
            or payload.get("native_energy_observations")
            or []
        )
    else:
        source = []
    return [dict(row) for row in source if isinstance(row, Mapping)]


def _performance_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in list(payload.get("rows") or []):
        if not isinstance(source, Mapping):
            continue
        role = str(source.get("row_role") or "").strip()
        source_kind = str(source.get("source_kind") or "").strip()
        if role == NATIVE_ENERGY_ROW_ROLE or source_kind == NATIVE_ENERGY_ROW_ROLE:
            continue
        row = dict(source)
        row["row_role"] = PERFORMANCE_ROW_ROLE
        row["row_status"] = _row_status(row)
        rows.append(row)
    return rows


def _structure_state(row: Mapping[str, Any]) -> tuple[bool, bool | None]:
    for key in (
        "structural_contract_pass",
        "claim_structural_gate_pass",
        "contract_consistent",
        "interface_contract_pass",
    ):
        if key in row:
            return True, _as_bool(row.get(key))
    return False, None


def _row_identity(
    row: Mapping[str, Any],
    *,
    population: str,
    index: int,
) -> dict[str, Any]:
    return {
        "population": population,
        "index": index,
        "model": row.get("model_id") or row.get("model"),
        "backend": row.get("backend"),
        "case": row.get("case_id") or row.get("case"),
    }


def _claim_contract_audit(
    report: Mapping[str, Any],
    performance_rows: Sequence[Mapping[str, Any]],
    native_observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    native_matrix = (
        report.get("native_performance_matrix")
        if isinstance(report.get("native_performance_matrix"), Mapping)
        else {}
    )
    populations: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("performance_observation", performance_rows),
        (
            "native_performance_matrix",
            [
                row
                for row in list(native_matrix.get("observations") or [])
                if isinstance(row, Mapping)
            ],
        ),
        ("native_energy_observation", native_observations),
    ]
    violations: list[dict[str, Any]] = []
    checked_count = 0
    structural_failure_count = 0
    for population, rows in populations:
        for index, row in enumerate(rows):
            has_structure, structure = _structure_state(row)
            if not has_structure:
                continue
            checked_count += 1
            if structure is True:
                continue
            structural_failure_count += 1
            identity = _row_identity(
                row,
                population=population,
                index=index,
            )
            if _as_bool(row.get("claim_ok")) is True:
                violations.append(
                    {
                        **identity,
                        "reason": "claim_ok_true_with_structural_contract_not_passed",
                    }
                )
            if str(row.get("status") or "").strip().lower() == "claim_ok":
                violations.append(
                    {
                        **identity,
                        "reason": "claim_ok_status_with_structural_contract_not_passed",
                    }
                )
            source_claim = _as_bool(row.get("claim_ok_source"))
            structurally_clamped = _as_bool(
                row.get("claim_ok_structural_clamped")
            )
            if source_claim is True and structurally_clamped is not True:
                violations.append(
                    {
                        **identity,
                        "reason": "provisional_claim_not_structurally_clamped",
                    }
                )
            for key in (
                "performance_eligible",
                "ranking_eligible",
                "energy_eligible",
                "claim_eligible",
            ):
                if _as_bool(row.get(key)) is True:
                    violations.append(
                        {
                            **identity,
                            "reason": (
                                f"{key}_true_with_structural_contract_not_passed"
                            ),
                        }
                    )
    return {
        "checked_row_count": checked_count,
        "structural_failure_or_unavailable_count": structural_failure_count,
        "violation_count": len(violations),
        "violations": violations,
    }


def replay_contract(
    report: Mapping[str, Any],
    native_observations: Sequence[Mapping[str, Any]],
    *,
    expected_performance: int,
    expected_energy_attempts: int,
    expected_energy_successes: int,
    expected_energy_failures: int,
) -> dict[str, Any]:
    performance_rows = _performance_rows(report)
    energy_rows = scientific_energy_rows(
        observations=native_observations,
    )
    reconstructed_rows = performance_rows + energy_rows
    summary = _refresh_scientific_summary(
        report.get("summary")
        if isinstance(report.get("summary"), Mapping)
        else {},
        reconstructed_rows,
    )
    failed_energy_rows = [
        row
        for row in energy_rows
        if row.get("row_status") == "measurement_failed"
    ]
    task_quality_contamination = [
        row
        for row in energy_rows
        if any(
            row.get(key) not in (None, "", [], {})
            for key in (
                "task_quality_status",
                "task_quality_decision",
                "task_quality_metric",
                "task_quality_delta",
            )
        )
    ]
    failed_with_numeric_result = [
        row
        for row in failed_energy_rows
        if any(
            row.get(key) not in (None, "")
            for key in (
                "average_power_w",
                "energy_per_work_j",
                "energy_total_j",
            )
        )
    ]
    claim_audit = _claim_contract_audit(
        report,
        performance_rows,
        native_observations,
    )

    checks: list[dict[str, Any]] = []

    def check(
        check_id: str,
        *,
        actual: Any,
        expected: Any,
        detail: str,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "ok": actual == expected,
                "actual": actual,
                "expected": expected,
                "detail": detail,
            }
        )

    expected_total = expected_performance + expected_energy_attempts
    check(
        "performance_row_count",
        actual=len(performance_rows),
        expected=expected_performance,
        detail="Only performance rows may feed task-quality reporting.",
    )
    check(
        "native_energy_attempt_count",
        actual=len(energy_rows),
        expected=expected_energy_attempts,
        detail="Every exported Native energy attempt must be reprojected.",
    )
    check(
        "native_energy_success_count",
        actual=summary.get("native_energy_success_count"),
        expected=expected_energy_successes,
        detail="Successful attempts retain their measured energy fields.",
    )
    check(
        "native_energy_failed_count",
        actual=summary.get("native_energy_failed_count"),
        expected=expected_energy_failures,
        detail="Failed attempts remain visible as measurement_failed rows.",
    )
    check(
        "reconstructed_row_count",
        actual=len(reconstructed_rows),
        expected=expected_total,
        detail="The replay must rebuild one disjoint row population.",
    )
    check(
        "row_role_counts",
        actual=summary.get("row_role_counts"),
        expected={
            PERFORMANCE_ROW_ROLE: expected_performance,
            NATIVE_ENERGY_ROW_ROLE: expected_energy_attempts,
        },
        detail="Performance and Native energy rows must remain role-separated.",
    )
    check(
        "task_quality_row_count",
        actual=summary.get("task_quality_row_count"),
        expected=expected_performance,
        detail="Energy attempts must never enter the task-quality denominator.",
    )
    check(
        "task_quality_status_denominator",
        actual=sum(
            int(value)
            for value in dict(
                summary.get("quality_status_counts") or {}
            ).values()
        ),
        expected=expected_performance,
        detail="Task-quality status counters must cover performance only.",
    )
    check(
        "task_quality_energy_contamination",
        actual=len(task_quality_contamination),
        expected=0,
        detail="Projected energy attempts must not expose task-quality rows.",
    )
    check(
        "failed_energy_ineligible",
        actual=sum(
            1
            for row in failed_energy_rows
            if _as_bool(row.get("energy_eligible")) is not True
        ),
        expected=expected_energy_failures,
        detail="Every failed energy attempt must be fail-closed.",
    )
    check(
        "failed_energy_without_numeric_result",
        actual=len(failed_with_numeric_result),
        expected=0,
        detail="A failed measurement must not invent an energy result.",
    )
    check(
        "structural_claim_violation_count",
        actual=claim_audit["violation_count"],
        expected=0,
        detail="claim_ok and eligibility must remain false when structure fails.",
    )

    replay_ok = all(item["ok"] for item in checks)
    reconstructed_report = dict(report)
    reconstructed_report["rows"] = reconstructed_rows
    reconstructed_report["summary"] = summary
    reconstructed_report["scientific_report_contract_replay"] = {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "status": "passed" if replay_ok else "failed",
    }
    return {
        "schema": REPLAY_SCHEMA,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "ok": replay_ok,
        "status": "passed" if replay_ok else "failed",
        "checks": checks,
        "summary": summary,
        "claim_ok_contract": claim_audit,
        "failed_energy_attempts": [
            {
                "model_id": row.get("model_id"),
                "backend": row.get("backend"),
                "case_id": row.get("case_id"),
                "row_status": row.get("row_status"),
                "energy_eligible": row.get("energy_eligible"),
                "measurement_failure_reason": row.get(
                    "measurement_failure_reason"
                ),
            }
            for row in failed_energy_rows
        ],
        "reconstructed_report": reconstructed_report,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scientific-report",
        type=Path,
        required=True,
        help="Existing scientific_report.json from the 2.70l pack.",
    )
    parser.add_argument(
        "--native-energy-observations",
        type=Path,
        required=True,
        help="Existing native_energy_observations.json from the same pack.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination for the replay result JSON.",
    )
    parser.add_argument("--expect-performance", type=int, default=48)
    parser.add_argument("--expect-energy-attempts", type=int, default=24)
    parser.add_argument("--expect-energy-successes", type=int, default=22)
    parser.add_argument("--expect-energy-failures", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report_payload = _load_json(args.scientific_report)
        if not isinstance(report_payload, Mapping):
            raise ValueError("scientific report must be a JSON object")
        native_payload = _load_json(args.native_energy_observations)
        native_observations = _native_observation_rows(native_payload)
        result = replay_contract(
            report_payload,
            native_observations,
            expected_performance=args.expect_performance,
            expected_energy_attempts=args.expect_energy_attempts,
            expected_energy_successes=args.expect_energy_successes,
            expected_energy_failures=args.expect_energy_failures,
        )
        result["inputs"] = {
            "scientific_report": str(args.scientific_report.resolve()),
            "scientific_report_sha256": _sha256(args.scientific_report),
            "native_energy_observations": str(
                args.native_energy_observations.resolve()
            ),
            "native_energy_observations_sha256": _sha256(
                args.native_energy_observations
            ),
        }
    except Exception as exc:
        result = {
            "schema": REPLAY_SCHEMA,
            "schema_version": REPLAY_SCHEMA_VERSION,
            "ok": False,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_json(args.out, result)
    print(
        json.dumps(
            {
                "ok": result.get("ok"),
                "status": result.get("status"),
                "out": str(args.out),
                "failed_checks": [
                    item.get("id")
                    for item in list(result.get("checks") or [])
                    if not item.get("ok")
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
