from __future__ import annotations

import csv
import json
from pathlib import Path

import onnx_splitpoint_tool.native_energy_reporting as native_energy_reporting
import onnx_splitpoint_tool.workflow.scientific_reporting as scientific_reporting
from onnx_splitpoint_tool.native_energy_reporting import scientific_energy_rows
from onnx_splitpoint_tool.native_energy_reporting import (
    _measurement_failure_reason,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _refresh_scientific_summary,
    _write_reports,
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_measurement_failure_reason_preserves_runner_diagnostic() -> None:
    assert (
        _measurement_failure_reason(
            {},
            {
                "rc": 1,
                "stderr_tail": (
                    "Traceback: ModuleNotFoundError: "
                    "onnx_splitpoint_tool"
                ),
            },
            {},
            execution_ok=False,
        )
        == "Traceback: ModuleNotFoundError: onnx_splitpoint_tool"
    )
    assert (
        _measurement_failure_reason(
            {},
            {"rc": 0, "stderr_tail": "irrelevant"},
            {},
            execution_ok=True,
        )
        == ""
    )


def test_scientific_energy_rows_preserve_success_and_failed_attempts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    collected = [
        {
            "ok": True,
            "model": "resnet50",
            "task": "classification",
            "backend": "native_full_tensorrt",
            "case": "full",
            "execution_mode": "native_full_baseline",
            "average_power_w": 11.0,
            "energy_per_work_j": 0.11,
            "claim_eligible": False,
            "claim_exclusion_reason": "smoke_policy",
        },
        {
            "ok": False,
            "model": "yolov7_paper",
            "task": "detection",
            "backend": "native_full_deepx",
            "case": "full",
            "execution_mode": "native_full_baseline",
            "claim_eligible": False,
            "measurement_failure_reason": "ModuleNotFoundError: onnx_splitpoint_tool",
            "measurement_failure_category": "runner_failed",
            "measurement_run_rc": 1,
        },
    ]
    monkeypatch.setattr(
        native_energy_reporting,
        "collect_native_energy",
        lambda _run_dir: collected,
    )

    rows = scientific_energy_rows(tmp_path)

    assert len(rows) == 2
    assert {row["row_role"] for row in rows} == {
        "native_energy_measurement"
    }
    assert [row["row_status"] for row in rows] == [
        "available",
        "measurement_failed",
    ]
    failed = rows[1]
    assert failed["measurement_ok"] is False
    assert failed["energy_eligible"] is False
    assert failed["eligibility_status"] == "measurement_failed"
    assert failed["average_power_w"] is None
    assert failed["energy_per_work_j"] is None
    assert (
        failed["measurement_failure_reason"]
        == "ModuleNotFoundError: onnx_splitpoint_tool"
    )
    assert failed["exclusion_reason"] == failed["measurement_failure_reason"]


def test_summary_counts_48_performance_and_all_24_energy_attempts() -> None:
    performance_rows = [
        {
            "row_role": "performance_observation",
            "row_status": "available",
            "model_id": f"model_{index % 3}",
            "runtime_executable": True,
            "throughput_fps": 100.0 + index,
            "performance_eligible": False,
            "ranking_eligible": False,
            "energy_eligible": False,
            "task_quality_decision": (
                "pass"
                if index < 42
                else "fail"
                if index < 45
                else "inconclusive"
            ),
            "eligibility_status": "screening_only",
        }
        for index in range(48)
    ]
    successful_energy_rows = [
        {
            "row_role": "native_energy_measurement",
            "row_status": "available",
            "model_id": f"model_{index % 3}",
            "measurement_ok": True,
            "average_power_w": 10.0,
            "energy_per_work_j": 0.1,
            "energy_eligible": False,
            "eligibility_status": "screening_only",
        }
        for index in range(22)
    ]
    failed_energy_rows = [
        {
            "row_role": "native_energy_measurement",
            "row_status": "measurement_failed",
            "model_id": f"model_{index}",
            "measurement_ok": False,
            "energy_eligible": False,
            "eligibility_status": "measurement_failed",
            "measurement_failure_reason": "measurement failed",
        }
        for index in range(2)
    ]

    summary = _refresh_scientific_summary(
        {"native_performance_observation_count": 45},
        performance_rows + successful_energy_rows + failed_energy_rows,
    )

    assert summary["row_count"] == 72
    assert summary["row_role_counts"] == {
        "performance_observation": 48,
        "native_energy_measurement": 24,
    }
    assert summary["row_status_counts"] == {
        "available": 70,
        "measurement_failed": 2,
    }
    assert summary["performance_observation_count"] == 48
    assert summary["task_quality_row_count"] == 48
    assert summary["eligibility_row_count"] == 48
    assert summary["native_energy_attempt_count"] == 24
    assert summary["native_energy_success_count"] == 22
    assert summary["native_energy_failed_count"] == 2
    assert summary["quality_status_counts"] == {
        "pass": 42,
        "fail": 3,
        "inconclusive": 3,
    }
    assert sum(summary["eligibility_status_counts"].values()) == 48
    assert summary["quality_failed_count"] == 3
    assert summary["quality_inconclusive_count"] == 3
    # Native performance rows already participate in ``performance_rows``;
    # the matrix counter is context, not a second observation population.
    assert summary["screening_performance_observation_count"] == 48
    assert summary["screening_energy_observation_count"] == 22
    assert summary["screening_energy_attempt_count"] == 24


def test_report_exports_keep_failures_visible_and_task_quality_role_scoped(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        scientific_reporting,
        "_make_figures",
        lambda *_args, **_kwargs: [],
    )
    performance_rows = [
        {
            "row_role": "performance_observation",
            "row_status": "available",
            "model_id": "resnet50",
            "backend": f"backend_{index}",
            "setup_id": f"setup_{index}",
            "case_id": f"case_{index}",
            "runtime_executable": True,
            "throughput_fps": 100.0 + index,
            "performance_eligible": False,
            "ranking_eligible": False,
            "energy_eligible": False,
            "task_quality_status": "pass",
            "task_quality_decision": "pass",
            "task_quality_metric": "top1_accuracy",
            "eligibility_status": "screening_only",
        }
        for index in range(2)
    ]
    successful_energy = {
        "row_role": "native_energy_measurement",
        "row_status": "available",
        "model_id": "resnet50",
        "backend": "native_full_tensorrt",
        "setup_id": "setup_h8",
        "case_id": "full",
        "measurement_ok": True,
        "average_power_w": 12.0,
        "energy_per_work_j": 0.12,
        "energy_eligible": False,
        "eligibility_status": "screening_only",
        "exclusion_reason": "smoke_policy",
    }
    failed_energy = {
        "row_role": "native_energy_measurement",
        "row_status": "measurement_failed",
        "model_id": "yolov7_paper",
        "backend": "native_full_deepx",
        "setup_id": "setup_deepx",
        "case_id": "full",
        "measurement_ok": False,
        "average_power_w": None,
        "energy_per_work_j": None,
        "energy_eligible": False,
        "eligibility_status": "measurement_failed",
        "measurement_failure_reason": "Module import failed",
        "exclusion_reason": "Module import failed",
    }
    native_observations = [
        {
            "model": "resnet50",
            "backend": "native_full_tensorrt",
            "case": "full",
            "measurement_status": "available",
            "energy_per_work_j": 0.12,
            "claim_eligible": False,
        },
        {
            "model": "yolov7_paper",
            "backend": "native_full_deepx",
            "case": "full",
            "measurement_status": "measurement_failed",
            "measurement_failure_reason": "Module import failed",
            "claim_eligible": False,
        },
    ]

    report_root = tmp_path / "scientific"
    _write_reports(
        report_root,
        {
            "created_at": "2026-07-26T00:00:00+02:00",
            "profile_id": "focused_test",
            "rows": performance_rows + [successful_energy, failed_energy],
            "summary": {},
            "native_energy_observations": native_observations,
            "native_evidence_status": {
                "runtime": {"status": "complete"},
                "semantics": {"status": "complete_fail"},
                "claim": {"status": "complete_not_eligible"},
                "energy": {"status": "complete"},
                "technical_status": "complete",
                "claim_decisions_complete": True,
                "energy_matrix_expected_count": 54,
                "energy_plan_included_count": 18,
                "energy_measurement_success_count": 18,
                "energy_claim_eligible_count": 0,
                "final_all_split_energy_complete": False,
                "scientific_status": "not_ready",
                "scientific_ready": False,
            },
        },
    )

    report = json.loads(
        (report_root / "scientific_report.json").read_text(encoding="utf-8")
    )
    summary = report["summary"]
    assert summary["row_count"] == 4
    assert summary["performance_observation_count"] == 2
    assert summary["task_quality_row_count"] == 2
    assert summary["native_energy_attempt_count"] == 2
    assert summary["native_energy_success_count"] == 1
    assert summary["native_energy_failed_count"] == 1
    assert len(_csv_rows(report_root / "row_eligibility.csv")) == 4
    assert len(_csv_rows(report_root / "task_quality.csv")) == 2
    assert len(
        _csv_rows(report_root / "screening_performance_observations.csv")
    ) == 2
    assert (
        (report_root / "claim_eligible_performance.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("model_id,task,case_id,backend")
    )
    assert (
        (report_root / "claim_eligible_energy.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
        .startswith("model_id,task,case_id,backend")
    )
    exclusion_summary = json.loads(
        (report_root / "claim_exclusion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert exclusion_summary["performance_claim_eligible_count"] == 0
    assert exclusion_summary["energy_claim_eligible_count"] == 0
    assert exclusion_summary["performance_excluded_count"] == 2
    assert exclusion_summary["energy_excluded_count"] == 2
    assert exclusion_summary["schema_version"] == 2
    assert exclusion_summary["group_dimensions"] == [
        "model_id", "backend", "setup_id", "reason",
    ]
    assert len(exclusion_summary["exclusion_details"]) == 4
    assert {
        (
            row["model_id"],
            row["backend"],
            row["setup_id"],
            row["reason"],
        )
        for row in exclusion_summary["exclusion_details"]
    } == {
        (
            "resnet50", "backend_0", "setup_0",
            "not_performance_eligible",
        ),
        (
            "resnet50", "backend_1", "setup_1",
            "not_performance_eligible",
        ),
        (
            "resnet50", "native_full_tensorrt", "setup_h8",
            "smoke_policy",
        ),
        (
            "yolov7_paper", "native_full_deepx", "setup_deepx",
            "Module import failed",
        ),
    }
    assert {
        row["dimension"]
        for row in exclusion_summary["dimension_exclusion_counts"]
    } == {"model", "backend", "setup", "reason"}
    assert (
        exclusion_summary["grouped_exclusion_counts"]
        == sorted(
            exclusion_summary["grouped_exclusion_counts"],
            key=lambda row: (
                0 if row["claim_kind"] == "performance" else 1,
                row["model_id"],
                row["backend"],
                row["setup_id"],
                row["reason"],
            ),
        )
    )
    report_breakdown = report["claim_exclusion_summary"]
    assert report_breakdown == exclusion_summary

    screening_energy_rows = _csv_rows(
        report_root / "screening_energy_observations.csv"
    )
    assert len(screening_energy_rows) == 2
    failed_csv = next(
        row
        for row in screening_energy_rows
        if row["row_status"] == "measurement_failed"
    )
    assert failed_csv["energy_per_work_j"] == ""
    assert failed_csv["measurement_failure_reason"] == "Module import failed"

    markdown = (report_root / "scientific_report.md").read_text(
        encoding="utf-8"
    )
    assert "Native energy attempts: **2** (**1** successful, **1** failed)" in markdown
    assert "measurement_failed" in markdown
    assert "Module import failed" in markdown
    assert "Energy plan completion: **18/18**" in markdown
    assert "Energy matrix coverage: **18/54**" in markdown
    assert "Energy claim eligible: **0**" in markdown
    assert "Scientific ready: **False**" in markdown

    screening_tex = (
        report_root
        / "thesis_tables"
        / "screening_energy_observations.tex"
    ).read_text(encoding="utf-8")
    native_tex = (
        report_root / "thesis_tables" / "native_energy_observations.tex"
    ).read_text(encoding="utf-8")
    assert "measurement\\_failed" in screening_tex
    assert "Module import failed" in screening_tex
    assert "measurement\\_failed" in native_tex
    assert "Module import failed" in native_tex


def test_claim_csvs_honor_explicit_v272_claim_vetoes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        scientific_reporting,
        "_make_figures",
        lambda *_args, **_kwargs: [],
    )
    performance = {
        "row_role": "performance_observation",
        "row_status": "available",
        "model_id": "yolov7_paper",
        "backend": "deepx_to_trt",
        "case_id": "b063",
        "runtime_executable": True,
        "throughput_fps": 100.0,
        "performance_eligible": True,
        "performance_claim_eligible": False,
        "performance_claim_exclusion_reasons": [
            "independent_replay_not_exact_claim_bound"
        ],
        "task_quality_status": "pass",
        "task_quality_decision": "pass",
        "eligibility_status": "screening_only",
    }
    energy = {
        "row_role": "native_energy_measurement",
        "row_status": "available",
        "model_id": "yolov7_paper",
        "backend": "deepx_to_trt",
        "case_id": "b063",
        "measurement_ok": True,
        "energy_per_work_j": 0.1,
        "energy_eligible": True,
        "energy_claim_eligible": False,
        "scientific_claim_exclusion_reasons": [
            "accuracy_gate_failed"
        ],
        "eligibility_status": "screening_only",
    }
    report_root = tmp_path / "claim-veto"

    _write_reports(
        report_root,
        {
            "rows": [performance, energy],
            "summary": {},
            "native_energy_observations": [],
        },
    )

    assert _csv_rows(
        report_root / "claim_eligible_performance.csv"
    ) == []
    assert _csv_rows(report_root / "claim_eligible_energy.csv") == []
    assert len(
        _csv_rows(report_root / "screening_performance_observations.csv")
    ) == 1
    assert len(
        _csv_rows(report_root / "screening_energy_observations.csv")
    ) == 1
    exclusions = json.loads(
        (report_root / "claim_exclusion_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert exclusions["performance_claim_eligible_count"] == 0
    assert exclusions["energy_claim_eligible_count"] == 0
