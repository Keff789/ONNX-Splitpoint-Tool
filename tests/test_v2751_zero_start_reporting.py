from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from onnx_splitpoint_tool import native_energy_reporting
from onnx_splitpoint_tool.workflow import dashboard
from onnx_splitpoint_tool.workflow.evidence_status import (
    project_native_evidence_status,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _write_reports,
)


def _zero_start_evidence() -> dict[str, Any]:
    """Return the contradictory v2.75 zero-start shape seen in reports."""
    return {
        "schema": "onnx-splitpoint/native-evidence-status",
        "schema_version": 5,
        "run_mode": "smoke",
        "runtime": {"status": "incomplete", "complete": False},
        "semantics": {"status": "incomplete", "complete": False},
        "claim": {"status": "incomplete", "decisions_complete": False},
        "energy": {
            # These stale completion aliases must lose to the objective
            # start/not-started ledger below.
            "status": "complete",
            "requested": True,
            "plan_included_count": 0,
            "matrix_expected_count": 27,
            "measurement_success_count": 0,
            "measurement_failed_count": 27,
            "measurement_started_count": 0,
            "not_started_preflight_count": 27,
            "terminal_result_count": 27,
            "planned_measurements_complete": True,
            "matrix_measurements_complete": True,
        },
        "technical_status": "complete",
        "evidence_complete": True,
        "claim_decisions_complete": False,
        "scientific_status": "ready",
        "scientific_ready": True,
        "positive_energy_claim_available": True,
        "energy_matrix_expected_count": 27,
        "energy_plan_included_count": 0,
        "energy_measurement_success_count": 0,
        "energy_measurement_failed_count": 27,
        "energy_measurement_started_count": 0,
        "energy_not_started_preflight_count": 27,
        "energy_terminal_result_count": 27,
        "energy_plan_completion_fraction": 1.0,
        "energy_successful_matrix_coverage_fraction": 1.0,
        "final_all_split_energy_required": True,
        "final_all_split_energy_complete": True,
    }


def _write_native_evidence(run_dir: Path) -> dict[str, Any]:
    evidence = _zero_start_evidence()
    reports = run_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "native_evidence_status.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8",
    )
    return evidence


def test_zero_start_projection_recovers_plan_and_fails_closed() -> None:
    projection = project_native_evidence_status(_zero_start_evidence())

    assert projection["schema_version"] == 3
    assert projection["energy_zero_start_blocked"] is True
    assert projection["energy_execution_status"] == (
        "blocked_before_measurement"
    )
    assert projection["energy_measurement_attempted_count"] == 0
    assert projection["energy_measurement_started_count"] == 0
    assert projection["energy_not_started_preflight_count"] == 27
    assert projection["energy_terminal_result_count"] == 27
    assert projection["energy_plan_denominator_count"] == 27
    assert projection["energy_matrix_denominator_count"] == 27
    assert projection["energy_measurement_success_count"] == 0
    assert projection["energy_measurement_failed_count"] == 0
    assert projection["energy_plan_completion_fraction"] == 0.0
    assert projection["energy_matrix_coverage_fraction"] == 0.0
    assert projection["energy_plan_completion"]["complete"] is False
    assert projection["energy_matrix_measurements_complete"] is False
    assert projection["final_all_split_energy_complete"] is False
    assert projection["positive_energy_claim_available"] is False
    assert projection["technical_status"] == "failed"
    assert projection["technical_complete"] is False
    assert projection["scientific_status"] == "not_ready"
    assert projection["scientific_ready"] is False


def test_scientific_report_renders_zero_of_plan_not_zero_of_zero(
    tmp_path: Path,
) -> None:
    report_root = tmp_path / "scientific"
    _write_reports(
        report_root,
        {
            "created_at": "2026-08-03T00:00:00+02:00",
            "profile_id": "zero_start_smoke",
            "rows": [],
            "summary": {},
            "native_evidence_status": _zero_start_evidence(),
        },
    )

    markdown = (report_root / "scientific_report.md").read_text(
        encoding="utf-8",
    )
    payload = json.loads(
        (report_root / "scientific_report.json").read_text(
            encoding="utf-8",
        )
    )
    projection = payload["native_evidence_summary"]

    assert "Native energy evidence: **blocked_before_measurement**" in markdown
    assert "Native energy attempts: **0 started, 27 not started**" in markdown
    assert "Energy plan completion: **0/27**" in markdown
    assert "Energy matrix coverage: **0/27**" in markdown
    assert "Technical status: **failed**" in markdown
    assert "Scientific status: **not_ready**" in markdown
    assert "**0/0**" not in markdown
    assert projection["energy_plan_denominator_count"] == 27
    assert projection["energy_zero_start_blocked"] is True


def test_dashboard_and_energy_report_share_zero_start_projection(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    run_dir = tmp_path / "EvaluationRun_zero_start"
    _write_native_evidence(run_dir)
    (run_dir / "reports" / "run_status_summary.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8",
    )

    monkeypatch.setattr(
        dashboard,
        "_write_visual_verification_report",
        lambda _run, reports: {
            "visual_verification_csv": (
                reports
                / "visual_verification"
                / "visual_verification.csv"
            ),
            "visual_verification_count": 0,
        },
    )
    monkeypatch.setattr(
        dashboard, "_try_write_figures", lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        dashboard,
        "_try_write_claim_figures",
        lambda *_args, **_kwargs: [],
    )

    dashboard_result = dashboard.build_result_dashboard_reports(
        run_dir,
        profile_id="zero_start_smoke",
        tool_version="2.75.1",
        workflow_version="2.75.1",
    )
    dashboard_markdown = (
        run_dir / "reports" / "result_dashboard.md"
    ).read_text(encoding="utf-8")
    dashboard_projection = dashboard_result["native_evidence_summary"]

    assert dashboard_result["overview"]["source_run_status"] == "completed"
    assert dashboard_result["overview"]["run_status"] == "failed"
    assert dashboard_projection["energy_plan_denominator_count"] == 27
    assert "Native energy execution: **blocked_before_measurement**" in (
        dashboard_markdown
    )
    assert "Native energy attempts: **0 started, 27 not started**" in (
        dashboard_markdown
    )
    assert "Energy plan completion: **0/27**" in dashboard_markdown
    assert "Energy matrix coverage: **0/27**" in dashboard_markdown
    assert "**0/0**" not in dashboard_markdown

    energy_result = native_energy_reporting.write_native_energy_reports(
        run_dir, run_dir / "reports" / "scientific",
    )
    energy_markdown = (
        run_dir
        / "reports"
        / "scientific"
        / "native_energy_pair_comparison.md"
    ).read_text(encoding="utf-8")

    assert energy_result["native_evidence_summary"] == dashboard_projection
    assert "Native energy execution: **blocked_before_measurement**" in (
        energy_markdown
    )
    assert "Native energy attempts: **0 started, 27 not started**" in (
        energy_markdown
    )
    assert "Energy plan completion: **0/27**" in energy_markdown
    assert "Energy matrix coverage: **0/27**" in energy_markdown
    assert "**0/0**" not in energy_markdown
