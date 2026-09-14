from __future__ import annotations

from onnx_splitpoint_tool.workflow.evidence_status import (
    blocking_status,
    derive_native_evidence_status,
)


def _matrix() -> dict[str, int]:
    return {
        "expected_row_count": 2,
        "present_expected_row_count": 2,
        "successful_expected_row_count": 2,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
    }


def _validation() -> dict:
    return {
        "technical_error_count": 0,
        "technical_chain_complete": True,
        "rows": [
            {
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": True,
            },
            {
                "semantic_available": False,
                "semantic_ok": None,
                "claim_ok": False,
                "semantic_status": "portable_result_hash_mismatch",
            },
        ],
    }


def test_standard_semantic_unavailable_is_partial_not_technical() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(),
        validation_payload=_validation(),
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["semantics"]["status"] == "incomplete"
    assert evidence["technical_quality_failure"] is False
    assert evidence["evidence_complete"] is False
    assert evidence["scientific_ready"] is False
    assert blocking_status(evidence, run_mode="standard") == "partial"


def test_final_semantic_unavailable_remains_fail_closed() -> None:
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix=_matrix(),
        validation_payload=_validation(),
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["technical_quality_failure"] is False
    assert evidence["scientific_ready"] is False
    assert blocking_status(evidence, run_mode="final") == "partial"  # v2.82: local gap; scientific readiness remains closed.


def test_declared_validation_chain_error_is_still_technical() -> None:
    validation = _validation()
    validation["technical_error_count"] = 1
    validation["technical_chain_complete"] = False
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(),
        validation_payload=validation,
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["technical_quality_failure"] is True
    assert blocking_status(evidence, run_mode="standard") == "partial"  # v2.82: local gap; scientific readiness remains closed.
