from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onnx_splitpoint_tool import native_energy_reporting
from onnx_splitpoint_tool.workflow import dashboard
from onnx_splitpoint_tool.workflow.evidence_status import (
    derive_native_evidence_status,
    project_native_evidence_status,
)
from onnx_splitpoint_tool.workflow.results import (
    build_normalized_results_payload,
)


def _golden_overnight_evidence() -> dict[str, Any]:
    expected_count = 54
    identities = [
        {
            "backend": "hailo8_to_trt",
            "model": f"resnet50_{index:02d}",
            "case": f"b{index:03d}",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "precision": "fp16",
        }
        for index in range(expected_count)
    ]
    return derive_native_evidence_status(
        run_mode="final",
        expected_matrix={
            "expected_row_count": expected_count,
            "present_expected_row_count": expected_count,
            "successful_expected_row_count": expected_count,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
        },
        validation_payload={
            "technical_error_count": 0,
            "technical_chain_complete": True,
            "rows": [
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": index < 35,
                }
                for index in range(expected_count)
            ],
        },
        validation_requested=True,
        energy_requested=True,
        energy_status="ok",
        energy_plan_payload={
            # Preserve the historical 18/54 projection without encoding the
            # old claim-gated planner rule.  These 36 rows stand for a legacy
            # technical-preflight scope that was never admitted to measure.
            "legacy_partial_plan_fixture": True,
            "legacy_partial_plan_reason": (
                "historical_v2_72_energy_preflight_scope"
            ),
            "rows": identities[:18],
            "excluded_rows": [
                {
                    **identities[index],
                    "reason": (
                        "legacy_energy_command_preflight_not_verified"
                    ),
                    "technical_exclusion": True,
                    "legacy_exclusion": True,
                }
                for index in range(18, expected_count)
            ],
        },
        energy_results_payload={
            "rows": [
                {
                    "row": identities[index],
                    "measurement_ok": True,
                    "energy_claim_eligible": False,
                }
                for index in range(18)
            ],
        },
    )


def _write_evidence(run_dir: Path) -> dict[str, Any]:
    evidence = _golden_overnight_evidence()
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "native_evidence_status.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8",
    )
    return evidence


def test_v272_projection_keeps_objective_axes_and_both_denominators() -> None:
    evidence = _golden_overnight_evidence()
    projection = project_native_evidence_status(evidence)

    assert projection["available"] is True
    assert projection["technical_status"] == "incomplete"
    assert projection["technical_complete"] is False
    assert projection["claim_decisions_complete"] is True
    assert projection["task_quality_decision_count"] == 54
    assert projection["task_quality_pass_count"] == 35
    assert projection["task_quality_fail_count"] == 19
    assert projection["task_quality_inconclusive_count"] == 0
    assert projection["task_quality_unavailable_count"] == 0
    assert projection["scientific_status"] == "not_ready"
    assert projection["scientific_ready"] is False
    assert projection["energy_measurement_success_count"] == 18
    assert projection["energy_plan_denominator_count"] == 18
    assert projection["energy_matrix_denominator_count"] == 54
    assert projection["energy_plan_completion_fraction"] == 1.0
    assert projection["energy_planned_matrix_coverage_fraction"] == 0.3333
    assert (
        projection["energy_successful_matrix_coverage_fraction"]
        == 0.3333
    )
    assert projection["energy_coverage_contract_valid"] is True
    assert projection["energy_matrix_measurements_complete"] is False
    assert projection["energy_plan_completion"] == {
        "numerator_count": 18,
        "denominator_count": 18,
        "excluded_count": 36,
        "fraction": 1.0,
        "complete": True,
    }
    assert projection["energy_matrix_coverage"] == {
        "numerator_count": 18,
        "denominator_count": 54,
        "fraction": 0.3333,
        "complete": False,
    }


def test_v2721_projection_keeps_three_distinct_energy_fractions() -> None:
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix={
            "expected_row_count": 4,
            "present_expected_row_count": 4,
            "successful_expected_row_count": 4,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
        },
        validation_payload={
            "technical_error_count": 0,
            "technical_chain_complete": True,
            "rows": [
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                }
                for _ in range(4)
            ],
        },
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "legacy_partial_plan_fixture": True,
            "legacy_partial_plan_reason": (
                "historical_v2_72_energy_preflight_scope"
            ),
            "rows": [{}, {}, {}],
            "excluded_rows": [{
                "reason": "legacy_energy_command_preflight_not_verified",
                "technical_exclusion": True,
                "legacy_exclusion": True,
            }],
        },
        energy_results_payload={
            "rows": [
                {"measurement_ok": True},
                {"measurement_ok": True},
                {"measurement_ok": False},
            ],
        },
    )

    projection = project_native_evidence_status(evidence)
    assert projection["energy_plan_completion_fraction"] == 0.6667
    assert projection["energy_planned_completion_fraction"] == 0.6667
    assert projection["energy_planned_matrix_coverage_fraction"] == 0.75
    assert (
        projection["energy_successful_matrix_coverage_fraction"]
        == 0.5
    )
    assert projection["energy_matrix_coverage_fraction"] == 0.5
    assert projection["energy_planned_matrix_coverage"] == {
        "numerator_count": 3,
        "denominator_count": 4,
        "fraction": 0.75,
        "complete": False,
    }
    assert projection["energy_successful_matrix_coverage"] == {
        "numerator_count": 2,
        "denominator_count": 4,
        "fraction": 0.5,
        "complete": False,
    }


def test_v2721_projection_recovers_canonical_axes_from_legacy_aliases() -> None:
    projection = project_native_evidence_status({
        "energy_matrix_expected_count": 4,
        "energy_plan_included_count": 3,
        "energy_measurement_success_count": 2,
        "energy_planned_completion_fraction": 0.6667,
        "energy_matrix_coverage_fraction": 0.5,
    })

    assert projection["energy_plan_completion_fraction"] == 0.6667
    assert projection["energy_planned_matrix_coverage_fraction"] == 0.75
    assert (
        projection["energy_successful_matrix_coverage_fraction"]
        == 0.5
    )


def test_v2721_projection_prefers_canonical_fractions_over_aliases() -> None:
    projection = project_native_evidence_status({
        "energy_plan_completion_fraction": 0.5,
        "energy_planned_completion_fraction": 0.9,
        "energy_planned_matrix_coverage_fraction": 0.75,
        "energy_successful_matrix_coverage_fraction": 0.25,
        "energy_matrix_coverage_fraction": 1.0,
    })

    assert projection["energy_plan_completion_fraction"] == 0.5
    assert projection["energy_planned_completion_fraction"] == 0.5
    assert projection["energy_planned_matrix_coverage_fraction"] == 0.75
    assert (
        projection["energy_successful_matrix_coverage_fraction"]
        == 0.25
    )
    assert projection["energy_matrix_coverage_fraction"] == 0.25


def test_v272_projection_never_uses_legacy_requirement_aware_claim_axis() -> None:
    projection = project_native_evidence_status({
        "claim": {
            "decision_complete": True,
        },
    })

    assert projection["available"] is True
    assert projection["claim_decisions_complete"] is None


def test_v272_missing_evidence_is_unavailable_not_false_or_zero() -> None:
    projection = project_native_evidence_status(None)

    assert projection["available"] is False
    assert projection["technical_status"] == "unavailable"
    assert projection["technical_complete"] is None
    assert projection["claim_decisions_complete"] is None
    assert projection["scientific_status"] == "unavailable"
    assert projection["scientific_ready"] is None
    assert projection["energy_measurement_success_count"] is None
    assert projection["energy_plan_denominator_count"] is None
    assert projection["energy_matrix_denominator_count"] is None


def test_v272_results_dashboard_and_energy_report_share_projection(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "EvaluationRun_reporting"
    evidence = _write_evidence(run_dir)
    expected = project_native_evidence_status(evidence)

    normalized = build_normalized_results_payload(
        model_id="resnet50",
        results=[],
        sources=[],
        run_root=run_dir,
    )
    assert normalized["native_evidence_status"] == evidence
    assert normalized["native_evidence_summary"] == expected

    monkeypatch.setattr(
        dashboard,
        "_write_visual_verification_report",
        lambda _run, reports: {
            "visual_verification_csv": (
                reports / "visual_verification" / "visual_verification.csv"
            ),
            "visual_verification_count": 0,
        },
    )
    monkeypatch.setattr(
        dashboard, "_try_write_figures", lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        dashboard, "_try_write_claim_figures",
        lambda *_args, **_kwargs: [],
    )
    dashboard_result = dashboard.build_result_dashboard_reports(
        run_dir,
        profile_id="thesis_final_campaign_v1",
        tool_version="2.72.0",
        workflow_version="2.72.0",
    )
    dashboard_payload = json.loads(
        (run_dir / "reports" / "result_dashboard.json").read_text(
            encoding="utf-8"
        )
    )
    energy_status = json.loads(
        (run_dir / "reports" / "energy_status.json").read_text(
            encoding="utf-8"
        )
    )
    dashboard_markdown = (
        run_dir / "reports" / "result_dashboard.md"
    ).read_text(encoding="utf-8")

    assert dashboard_result["native_evidence_summary"] == expected
    assert dashboard_result["overview"]["native_evidence_summary"] == expected
    assert dashboard_payload["native_evidence_summary"] == expected
    assert dashboard_payload["overview"]["native_evidence_summary"] == expected
    assert energy_status["native_evidence_summary"] == expected
    assert "Energy plan completion: **18/18**" in dashboard_markdown
    assert "Energy matrix coverage: **18/54**" in dashboard_markdown
    assert "Scientific ready: **False**" in dashboard_markdown

    measurement_root = (
        run_dir / "reports" / "native_energy_measurements"
    )
    measurement_root.mkdir(parents=True, exist_ok=True)
    (measurement_root / "native_producer_energy_results.json").write_text(
        json.dumps({"rows": []}), encoding="utf-8",
    )
    scientific_dir = run_dir / "reports" / "scientific"
    energy_result = native_energy_reporting.write_native_energy_reports(
        run_dir, scientific_dir,
    )
    energy_report_status = json.loads(
        (
            scientific_dir / "native_energy_report_status.json"
        ).read_text(encoding="utf-8")
    )
    energy_markdown = (
        scientific_dir / "native_energy_pair_comparison.md"
    ).read_text(encoding="utf-8")

    assert energy_result["native_evidence_summary"] == expected
    assert energy_report_status["native_evidence_summary"] == expected
    assert "Energy plan completion: **18/18**" in energy_markdown
    assert "Energy matrix coverage: **18/54**" in energy_markdown
    assert "Scientific ready: **False**" in energy_markdown
