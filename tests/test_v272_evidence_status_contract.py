from __future__ import annotations

from pathlib import Path
from typing import Any

from onnx_splitpoint_tool.workflow.evidence_status import (
    SCHEMA_VERSION,
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


def _validation(
    count: int,
    *,
    claim_eligible_count: int | None = None,
) -> dict[str, Any]:
    eligible = (
        count if claim_eligible_count is None else claim_eligible_count
    )
    return {
        "technical_error_count": 0,
        "technical_chain_complete": True,
        "rows": [
            {
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": index < eligible,
            }
            for index in range(count)
        ],
    }


def _energy_row(index: int) -> dict[str, Any]:
    return {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": f"b{index:03d}",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "precision": "fp16",
    }


def test_v272_golden_overnight_energy_axes_are_not_conflated() -> None:
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix=_matrix(54),
        validation_payload=_validation(54, claim_eligible_count=35),
        validation_requested=True,
        energy_requested=True,
        energy_status="ok",
        energy_plan_payload={
            "legacy_partial_plan_fixture": True,
            "legacy_partial_plan_reason": (
                "historical_v2_72_energy_preflight_scope"
            ),
            "rows": [_energy_row(index) for index in range(18)],
            "excluded_rows": [
                {
                    **_energy_row(index),
                    "reason": (
                        "legacy_energy_command_preflight_not_verified"
                    ),
                    "technical_exclusion": True,
                    "legacy_exclusion": True,
                }
                for index in range(18, 54)
            ],
        },
        energy_results_payload={
            "rows": [
                {
                    "row": _energy_row(index),
                    "ok": True,
                    "energy_claim_eligible": False,
                }
                for index in range(18)
            ],
        },
    )

    assert evidence["schema_version"] == 5 == SCHEMA_VERSION
    assert evidence["technical_status"] == "incomplete"
    assert evidence["claim_decisions_complete"] is True
    assert evidence["scientific_status"] == "not_ready"
    assert evidence["energy_matrix_expected_count"] == 54
    assert evidence["energy_plan_included_count"] == 18
    assert evidence["energy_plan_excluded_count"] == 36
    assert evidence["energy_measurement_success_count"] == 18
    assert evidence["energy_measurement_failed_count"] == 0
    assert evidence["energy_claim_eligible_count"] == 0
    assert evidence["energy_planned_completion_fraction"] == 1.0
    assert evidence["energy_matrix_coverage_fraction"] == 0.3333
    assert evidence["final_all_split_energy_required"] is True
    assert evidence["final_all_split_energy_complete"] is False
    assert evidence["energy"]["planned_measurements_complete"] is True
    assert evidence["deprecated_aliases"]["energy_complete"]["value"] is True
    assert evidence["deprecated_aliases"]["claim_ready"]["value"] is True
    assert evidence["scientific_ready"] is False
    assert blocking_status(evidence, run_mode="final") == "partial"  # v2.82: local gap; scientific readiness remains closed.


def test_v272_explicit_empty_plan_is_never_complete_even_with_ok_status() -> None:
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix=_matrix(2),
        validation_payload=_validation(2),
        validation_requested=True,
        energy_requested=True,
        energy_status="ok",
        energy_strict=False,
        energy_plan_payload={},
        energy_results_payload={},
    )

    assert evidence["energy"]["coverage_contract_active"] is True
    assert evidence["energy"]["coverage_source"] == (
        "v4_identity_plan_results_counts"
    )
    assert evidence["energy"]["status"] == "empty_plan"
    assert evidence["energy"]["planned_measurements_complete"] is False
    assert evidence["energy_plan_included_count"] == 0
    assert evidence["energy_measurement_success_count"] == 0
    assert evidence["energy_planned_completion_fraction"] == 0.0
    assert evidence["energy_matrix_coverage_fraction"] == 0.0
    assert evidence["final_all_split_energy_complete"] is False
    assert evidence["technical_status"] == "incomplete"
    assert evidence["scientific_ready"] is False
    assert blocking_status(evidence, run_mode="final") == "partial"  # v2.82: local gap; scientific readiness remains closed.


def test_v272_full_matrix_with_eligible_energy_can_be_scientific_ready() -> None:
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix=_matrix(2),
        validation_payload=_validation(2),
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [_energy_row(0), _energy_row(1)],
            "excluded_rows": [],
        },
        energy_results_payload={
            "rows": [
                {
                    "row": _energy_row(0),
                    "measurement_ok": True,
                    "energy_claim_eligible": True,
                },
                {
                    "row": _energy_row(1),
                    "measurement_ok": True,
                    "energy_claim_eligible": True,
                },
            ],
        },
    )

    assert evidence["energy_planned_completion_fraction"] == 1.0
    assert evidence["energy_matrix_coverage_fraction"] == 1.0
    assert evidence["energy_claim_eligible_count"] == 2
    assert evidence["final_all_split_energy_complete"] is True
    assert evidence["technical_status"] == "complete"
    assert evidence["scientific_status"] == "ready"
    assert evidence["scientific_ready"] is True
    assert blocking_status(evidence, run_mode="final") == ""


def test_v272_explicit_counts_override_payload_derived_counts() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(4),
        validation_payload=_validation(4),
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={"rows": []},
        energy_results_payload={"rows": []},
        energy_counts={
            "energy_plan_included_count": 3,
            "energy_plan_excluded_count": 1,
            "energy_measurement_success_count": 2,
            "energy_measurement_failed_count": 1,
            "energy_claim_eligible_count": 1,
        },
    )

    assert evidence["energy_plan_included_count"] == 3
    assert evidence["energy_plan_excluded_count"] == 1
    assert evidence["energy_measurement_success_count"] == 2
    assert evidence["energy_measurement_failed_count"] == 1
    assert evidence["energy_claim_eligible_count"] == 1
    assert evidence["energy_planned_completion_fraction"] == 0.6667
    assert evidence["energy_plan_completion_fraction"] == 0.6667
    assert evidence["energy_planned_matrix_coverage_fraction"] == 0.75
    assert evidence["energy_successful_matrix_coverage_fraction"] == 0.5
    assert evidence["energy_matrix_coverage_fraction"] == 0.5
    assert evidence["energy_coverage_contract_valid"] is False
    assert evidence["energy"]["status"] == "measurement_failed"
    assert evidence["energy"]["planned_measurements_complete"] is False


def test_v272_legacy_status_token_api_remains_available_and_deprecated() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(1),
        validation_payload=_validation(1),
        validation_requested=True,
        energy_requested=True,
        energy_status="ok",
    )

    assert evidence["energy"]["coverage_contract_active"] is False
    assert evidence["energy"]["coverage_source"] == "legacy_status_token"
    assert "complete" not in evidence["energy"]
    assert "energy_complete" not in evidence
    assert "claim_ready" not in evidence
    assert evidence["deprecated_aliases"]["energy_complete"]["value"] is True
    assert evidence["deprecated_aliases"]["claim_ready"]["deprecated"] is True


def test_v2721_counts_only_coverage_denominator_mismatch_is_invalid() -> None:
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix=_matrix(4),
        validation_payload=_validation(4),
        validation_requested=True,
        energy_requested=True,
        energy_counts={
            "energy_plan_included_count": 3,
            "energy_plan_excluded_count": 0,
            "energy_measurement_success_count": 3,
            "energy_measurement_failed_count": 0,
        },
    )

    assert evidence["energy_plan_ledger_valid"] is False
    assert evidence["energy_result_ledger_valid"] is True
    assert evidence["energy_coverage_contract_valid"] is False
    assert evidence["energy"]["planned_measurements_complete"] is True
    assert evidence["final_all_split_energy_complete"] is False


def test_v2721_explicit_empty_ledgers_cannot_override_positive_counts() -> None:
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix=_matrix(2),
        validation_payload=_validation(2),
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [],
            "excluded_rows": [],
        },
        energy_results_payload={"rows": []},
        energy_counts={
            "energy_plan_included_count": 2,
            "energy_plan_excluded_count": 0,
            "energy_measurement_success_count": 2,
            "energy_measurement_failed_count": 0,
        },
    )

    assert evidence["energy_plan_ledger_valid"] is False
    assert evidence["energy_result_ledger_valid"] is False
    assert evidence["energy_coverage_contract_valid"] is False
    assert evidence["final_all_split_energy_complete"] is False


def test_v272_claim_decision_axis_is_objective_in_diagnostic_mode() -> None:
    validation = _validation(1)
    evidence = derive_native_evidence_status(
        run_mode="smoke",
        expected_matrix=_matrix(2),
        validation_payload=validation,
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["claim"]["required"] is False
    assert evidence["claim"]["decision_complete"] is True
    assert evidence["claim"]["decisions_complete"] is False
    assert evidence["claim_decisions_complete"] is False


def test_v272_runner_threads_archived_energy_payloads_into_both_paths() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "onnx_splitpoint_tool"
        / "workflow"
        / "runner.py"
    ).read_text(encoding="utf-8")

    assert "def _native_energy_payload_v272(" in source
    assert source.count(
        "energy_plan_payload=archived_energy_plan_payload"
    ) == 2
    assert source.count(
        "energy_results_payload=archived_energy_results_payload"
    ) == 2
    assert source.count(
        'native_energy_state,\n                "plan_json",'
    ) == 1
    assert source.count(
        'energy_state,\n            "plan_json",'
    ) == 1
    assert source.count(
        "final_all_split_energy_required=("
    ) == 2


def test_v272_final_profile_requires_all_split_energy() -> None:
    profile = (
        Path(__file__).resolve().parents[1]
        / "onnx_splitpoint_tool"
        / "resources"
        / "evaluation_profiles"
        / "thesis_final_campaign_v1.yaml"
    ).read_text(encoding="utf-8")

    assert "final_all_split_energy: true" in profile


def test_v272_negative_quality_is_downstream_of_plan_membership() -> None:
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix=_matrix(2),
        validation_payload=_validation(2, claim_eligible_count=1),
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [_energy_row(0), _energy_row(1)],
            "excluded_rows": [],
        },
        energy_results_payload={
            "rows": [
                {
                    "row": _energy_row(0),
                    "measurement_ok": True,
                    "energy_claim_eligible": True,
                },
                {
                    "row": _energy_row(1),
                    "measurement_ok": True,
                    "energy_claim_eligible": False,
                },
            ],
        },
    )

    assert evidence["energy_plan_included_count"] == 2
    assert evidence["energy_plan_excluded_count"] == 0
    assert evidence["energy_measurement_success_count"] == 2
    assert evidence["task_quality"]["pass_count"] == 1
    assert evidence["task_quality"]["fail_count"] == 1
    assert evidence["energy_claim_eligible_count"] == 1
    assert evidence["final_all_split_energy_complete"] is True
    assert evidence["positive_energy_claim_available"] is True
    assert evidence["scientific_ready"] is True
