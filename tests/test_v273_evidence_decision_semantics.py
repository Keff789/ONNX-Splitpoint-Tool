from __future__ import annotations

import pytest

from onnx_splitpoint_tool.workflow.evidence_status import (
    blocking_status,
    derive_native_evidence_status,
)


def _matrix(count: int) -> dict[str, int]:
    return {
        "expected_row_count": count,
        "present_expected_row_count": count,
        "successful_expected_row_count": count,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
    }


def _row(decision: str, *, claim_ok: bool = False) -> dict[str, object]:
    return {
        "semantic_available": True,
        "semantic_ok": True,
        "claim_ok": claim_ok,
        "task_quality_status": decision,
        "accuracy_gate_decision": decision,
        "task_quality_gate": {
            "decision": decision,
            "status": decision,
        },
    }


def _derive(rows: list[dict[str, object]], mode: str = "standard") -> dict:
    return derive_native_evidence_status(
        run_mode=mode,
        expected_matrix=_matrix(len(rows)),
        validation_payload={
            "technical_error_count": 0,
            "technical_chain_complete": True,
            "rows": rows,
        },
        validation_requested=True,
        energy_requested=False,
    )


def test_fail_and_inconclusive_are_complete_but_not_positive_claims() -> None:
    evidence = _derive([_row("fail"), _row("inconclusive")])

    assert evidence["task_quality"] == {
        "status": "complete",
        "requested": True,
        "complete": True,
        "row_count": 2,
        "decision_count": 2,
        "pass_count": 0,
        "fail_count": 1,
        "inconclusive_count": 1,
        "reference_count": 0,
        "unavailable_count": 0,
        "alias_conflict_count": 0,
    }
    assert evidence["claim_decisions_complete"] is True
    assert evidence["claim"]["eligible_count"] == 0
    assert evidence["positive_performance_claim_available"] is False
    assert evidence["evidence_complete"] is True
    assert evidence["technical_status"] == "complete"
    assert evidence["scientific_ready"] is True
    assert blocking_status(evidence, run_mode="standard") == ""


@pytest.mark.parametrize(
    ("mode", "expected_block"),
    [("standard", "partial"), ("final", "failed")],
)
def test_unavailable_is_missing_evidence_not_a_negative_decision(
    mode: str,
    expected_block: str,
) -> None:
    evidence = _derive(
        [_row("pass", claim_ok=True), _row("unavailable")],
        mode,
    )

    assert evidence["task_quality"]["pass_count"] == 1
    assert evidence["task_quality"]["unavailable_count"] == 1
    assert evidence["task_quality"]["decision_count"] == 1
    assert evidence["claim_decisions_complete"] is False
    assert evidence["evidence_complete"] is False
    assert evidence["technical_quality_failure"] is False
    assert evidence["technical_status"] == "incomplete"
    assert blocking_status(evidence, run_mode=mode) == expected_block


def test_conflicting_quality_aliases_fail_closed_as_unavailable() -> None:
    row = _row("pass", claim_ok=True)
    row["accuracy_gate_decision"] = "fail"

    evidence = _derive([row])

    assert evidence["task_quality"]["alias_conflict_count"] == 1
    assert evidence["task_quality"]["unavailable_count"] == 1
    assert evidence["claim_decisions_complete"] is False
    assert blocking_status(evidence, run_mode="standard") == "partial"


def test_legacy_rows_without_quality_fields_remain_archive_compatible() -> None:
    evidence = _derive([
        {
            "semantic_available": True,
            "semantic_ok": False,
            "claim_ok": False,
        },
    ])

    assert evidence["task_quality"]["fail_count"] == 1
    assert evidence["claim_decisions_complete"] is True
    assert evidence["evidence_complete"] is True
    assert blocking_status(evidence, run_mode="standard") == ""


def test_fully_measured_rejected_rows_are_thesis_ready_without_claims() -> None:
    rows = [_row("fail"), _row("inconclusive")]
    energy_rows = [
        {
            "backend": "deepx_to_trt",
            "model": "resnet50",
            "case": f"b{index:03d}",
            "setup_id": "deepx",
            "comparison_backend": "deepx",
            "precision": "fp16",
        }
        for index in range(2)
    ]
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(2),
        validation_payload={
            "technical_error_count": 0,
            "technical_chain_complete": True,
            "rows": rows,
        },
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": energy_rows,
            "excluded_rows": [],
        },
        energy_results_payload={
            "rows": [
                {
                    "row": row,
                    "measurement_ok": True,
                    "energy_claim_eligible": False,
                }
                for row in energy_rows
            ],
        },
    )

    assert evidence["energy"]["matrix_measurements_complete"] is True
    assert evidence["positive_performance_claim_available"] is False
    assert evidence["positive_energy_claim_available"] is False
    assert evidence["evidence_complete"] is True
    assert evidence["scientific_ready"] is True
    assert blocking_status(evidence, run_mode="standard") == ""
