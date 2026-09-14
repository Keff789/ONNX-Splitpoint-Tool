from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.workflow.evidence_status import (
    blocking_status,
    derive_native_evidence_status,
    project_native_evidence_status,
)
from onnx_splitpoint_tool.workflow import scientific_reporting


def _matrix(count: int = 2) -> dict[str, int]:
    return {
        "expected_row_count": count,
        "present_expected_row_count": count,
        "successful_expected_row_count": count,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
    }


def test_disabled_energy_placeholder_artifacts_are_neutral_na() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(),
        validation_payload=None,
        validation_requested=False,
        energy_requested=False,
        energy_status="disabled",
        energy_strict=True,
        # The coordinator archives these placeholders even when Energy is off.
        energy_plan_payload={"rows": [], "excluded_rows": []},
        energy_results_payload={"rows": []},
        energy_counts={
            "energy_plan_included_count": 0,
            "energy_plan_excluded_count": 0,
            "energy_measurement_success_count": 0,
            "energy_measurement_failed_count": 0,
        },
    )

    energy = evidence["energy"]
    assert energy["status"] == "not_applicable"
    assert energy["requested"] is False
    assert energy["strict"] is False
    assert energy["coverage_contract_active"] is False
    assert energy["coverage_contract_status"] == "not_applicable"
    assert energy["coverage_source"] == "not_applicable"
    assert energy["coverage_contract_valid"] is None
    assert energy["plan_ledger_valid"] is None
    assert energy["result_ledger_valid"] is None
    assert energy["expected_plan_identity_match"] is None
    assert energy["matrix_expected_count"] is None
    assert energy["planned_measurements_complete"] is True
    assert energy["matrix_measurements_complete"] is True

    assert evidence["energy_coverage_contract_active"] is False
    assert evidence["energy_coverage_contract_status"] == "not_applicable"
    assert evidence["energy_coverage_contract_valid"] is None
    assert evidence["energy_plan_ledger_valid"] is None
    assert evidence["energy_result_ledger_valid"] is None
    assert evidence["energy_expected_plan_identity_match"] is None
    assert evidence["energy_matrix_expected_count"] is None
    assert evidence["energy_plan_completion_fraction"] is None
    assert evidence["positive_energy_claim_available"] is None
    assert evidence["final_all_split_energy_required"] is False
    assert evidence["final_all_split_energy_complete"] is None
    assert evidence["technical_quality_failure"] is False
    assert evidence["evidence_complete"] is True
    assert blocking_status(evidence, run_mode="standard") == ""


def test_projection_repairs_archived_disabled_energy_false_ledgers() -> None:
    projection = project_native_evidence_status({
        "schema": "onnx-splitpoint/native-evidence-status",
        "schema_version": 5,
        "energy": {
            "status": "not_applicable",
            "requested": False,
            "coverage_contract_active": True,
            "coverage_contract_valid": False,
            "plan_ledger_valid": False,
            "result_ledger_valid": True,
            "expected_plan_identity_match": False,
            "matrix_expected_count": 21,
            "plan_included_count": 0,
            "measurement_success_count": 0,
            "matrix_measurements_complete": True,
            "final_all_split_required": False,
            "final_all_split_complete": False,
        },
        "energy_matrix_expected_count": 21,
        "energy_plan_included_count": 0,
        "energy_measurement_success_count": 0,
        "energy_plan_completion_fraction": 0.0,
        "energy_planned_matrix_coverage_fraction": 0.0,
        "energy_successful_matrix_coverage_fraction": 0.0,
        "energy_coverage_contract_valid": False,
        "energy_plan_ledger_valid": False,
        "energy_result_ledger_valid": True,
        "energy_expected_plan_identity_match": False,
        "final_all_split_energy_required": False,
        "final_all_split_energy_complete": False,
    })

    assert projection["energy_status"] == "not_applicable"
    assert projection["energy_requested"] is False
    assert projection["energy_coverage_contract_active"] is False
    assert (
        projection["energy_coverage_contract_status"]
        == "not_applicable"
    )
    assert projection["energy_coverage_contract_valid"] is None
    assert projection["energy_plan_ledger_valid"] is None
    assert projection["energy_result_ledger_valid"] is None
    assert projection["energy_expected_plan_identity_match"] is None
    assert projection["energy_matrix_expected_count"] is None
    assert projection["energy_plan_included_count"] is None
    assert projection["energy_measurement_success_count"] is None
    assert projection["energy_plan_completion_fraction"] is None
    assert projection["final_all_split_energy_complete"] is None
    assert projection["energy_zero_start_blocked"] is False


def test_requested_strict_empty_energy_contract_still_fails_closed() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(),
        validation_payload=None,
        validation_requested=False,
        energy_requested=True,
        energy_strict=True,
        energy_plan_payload={"rows": [], "excluded_rows": []},
        energy_results_payload={"rows": []},
    )

    assert evidence["energy"]["coverage_contract_active"] is True
    assert evidence["energy"]["coverage_contract_status"] == "invalid"
    assert evidence["energy_coverage_contract_valid"] is False
    assert evidence["energy_plan_ledger_valid"] is False
    assert evidence["energy"]["status"] == "empty_plan"
    assert evidence["technical_quality_failure"] is True
    assert blocking_status(evidence, run_mode="standard") == "partial"  # v2.82: local gap; scientific readiness remains closed.


def test_scientific_summary_preserves_disabled_energy_na(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (tmp_path / "profile.yaml").write_text(
        "name: energy-na\nmodel_suite:\n  primary: []\n  reserve: []\n",
        encoding="utf-8",
    )
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(1),
        validation_payload=None,
        validation_requested=False,
        energy_requested=False,
        energy_plan_payload={},
        energy_results_payload={},
    )
    (reports / "native_evidence_status.json").write_text(
        json.dumps(evidence), encoding="utf-8",
    )
    (reports / "run_status_summary.json").write_text(
        json.dumps({"technical_status": "ok"}), encoding="utf-8",
    )

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        scientific_reporting,
        "_write_reports",
        lambda _root, payload, **_kwargs: captured.update(
            {"payload": payload}
        )
        or {},
    )
    monkeypatch.setattr(
        scientific_reporting,
        "_augment_v60z_quality_evidence",
        lambda *_args, **_kwargs: {},
    )

    scientific_reporting.build_scientific_reports(
        tmp_path, cleanup_legacy=False,
    )
    summary = captured["payload"]["summary"]  # type: ignore[index]
    assert summary["energy_execution_status"] == "not_applicable"
    assert summary["energy_coverage_contract_active"] is False
    assert summary["energy_coverage_contract_status"] == "not_applicable"
    assert summary["energy_coverage_contract_valid"] is None
    assert summary["energy_plan_ledger_valid"] is None
    assert summary["energy_result_ledger_valid"] is None
    assert summary["energy_matrix_expected_count"] is None
    assert summary["energy_plan_included_count"] is None
    assert summary["energy_measurement_success_count"] is None
    assert summary["energy_plan_completion_fraction"] is None
    assert summary["final_all_split_energy_complete"] is None
