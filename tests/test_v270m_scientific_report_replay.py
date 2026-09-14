from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "replay_scientific_report_contract.py"
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _fixture_payloads() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    performance_rows = [
        {
            "model_id": f"model_{index % 3}",
            "backend": f"generic_{index % 4}",
            "case_id": f"b{index:03d}",
            "runtime_executable": True,
            "contract_consistent": index >= 3,
            "throughput_fps": 100.0 + index,
            "performance_eligible": False,
            "ranking_eligible": False,
            "energy_eligible": False,
            "task_quality_status": (
                "pass" if index < 42 else "fail"
            ),
            "task_quality_decision": (
                "pass" if index < 42 else "fail"
            ),
            "eligibility_status": (
                "screening_only"
                if index >= 3
                else "contract_fail_or_unavailable"
            ),
        }
        for index in range(48)
    ]
    # This reproduces the 2.70l state: 22 successful energy rows were already
    # appended to the report, while the two failed attempts were absent.
    old_projected_energy_rows = [
        {
            "source_kind": "native_energy_measurement",
            "model_id": f"model_{index % 3}",
            "backend": f"native_{index % 4}",
            "case_id": "full",
            "average_power_w": 10.0,
            "energy_per_work_j": 0.1,
            "energy_eligible": False,
            "eligibility_status": "screening_only",
        }
        for index in range(22)
    ]
    native_observations = [
        {
            "ok": True,
            "model": f"model_{index % 3}",
            "task": "classification",
            "backend": f"native_{index % 4}",
            "case": "full",
            "execution_mode": "native_full_baseline",
            "contract_consistent": True,
            "claim_ok": False,
            "claim_eligible": False,
            "average_power_w": 10.0,
            "energy_per_work_j": 0.1,
            "energy_total_j": 10.0,
            "claim_exclusion_reason": "smoke_policy",
        }
        for index in range(22)
    ]
    native_observations.extend(
        [
            {
                "ok": False,
                "model": "yolov7_paper",
                "task": "detection",
                "backend": backend,
                "case": "full",
                "execution_mode": "native_full_baseline",
                "contract_consistent": False,
                "claim_ok": False,
                "claim_eligible": False,
                "postprocess_status": "collector_failed",
                "final_energy_gate_status": "fail",
                "energy_aggregate_status": (
                    "acquisition_integrity_failed_no_valid_runs"
                ),
            }
            for backend in (
                "native_full_deepx",
                "native_full_hailo10h",
            )
        ]
    )
    report = {
        "schema": "onnx-splitpoint/scientific-report",
        "schema_version": 3,
        "rows": performance_rows + old_projected_energy_rows,
        "summary": {
            "row_count": 70,
            "native_performance_observation_count": 45,
        },
        "native_performance_matrix": {
            "observations": [
                {
                    "model_id": "yolov7_paper",
                    "backend": "hailo8_to_trt",
                    "case_id": "b044",
                    "structural_contract_pass": False,
                    "claim_structural_gate_pass": False,
                    "claim_ok": False,
                    "claim_ok_source": True,
                    "claim_ok_structural_clamped": True,
                    "performance_eligible": False,
                    "ranking_eligible": False,
                    "energy_eligible": False,
                }
            ]
        },
    }
    return report, native_observations


def _run(
    tmp_path: Path,
    report: dict[str, Any],
    observations: list[dict[str, Any]],
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    report_path = tmp_path / "scientific_report.json"
    observations_path = tmp_path / "native_energy_observations.json"
    out_path = tmp_path / "replay.json"
    _write_json(report_path, report)
    _write_json(observations_path, observations)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--scientific-report",
            str(report_path),
            "--native-energy-observations",
            str(observations_path),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(out_path.read_text(encoding="utf-8"))


def test_replay_reconstructs_48_plus_24_and_preserves_two_failures(
    tmp_path: Path,
) -> None:
    report, observations = _fixture_payloads()

    completed, replay = _run(tmp_path, report, observations)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert replay["ok"] is True
    assert replay["status"] == "passed"
    summary = replay["summary"]
    assert summary["row_count"] == 72
    assert summary["row_role_counts"] == {
        "performance_observation": 48,
        "native_energy_measurement": 24,
    }
    assert summary["row_status_counts"] == {
        "available": 70,
        "measurement_failed": 2,
    }
    assert summary["task_quality_row_count"] == 48
    assert sum(summary["quality_status_counts"].values()) == 48
    assert summary["native_energy_success_count"] == 22
    assert summary["native_energy_failed_count"] == 2
    assert len(replay["failed_energy_attempts"]) == 2
    assert all(
        row["energy_eligible"] is False
        and row["row_status"] == "measurement_failed"
        and "collector_failed" in row["measurement_failure_reason"]
        for row in replay["failed_energy_attempts"]
    )
    reconstructed_rows = replay["reconstructed_report"]["rows"]
    assert len(reconstructed_rows) == 72
    energy_rows = [
        row
        for row in reconstructed_rows
        if row["row_role"] == "native_energy_measurement"
    ]
    assert len(energy_rows) == 24
    assert not any("task_quality_status" in row for row in energy_rows)
    assert replay["claim_ok_contract"]["violation_count"] == 0


def test_replay_fails_closed_on_claim_ok_with_structural_failure(
    tmp_path: Path,
) -> None:
    report, observations = _fixture_payloads()
    matrix_row = report["native_performance_matrix"]["observations"][0]
    matrix_row["claim_ok"] = True

    completed, replay = _run(tmp_path, report, observations)

    assert completed.returncode != 0
    assert replay["ok"] is False
    failed_checks = {
        item["id"] for item in replay["checks"] if item["ok"] is False
    }
    assert "structural_claim_violation_count" in failed_checks
    reasons = {
        row["reason"]
        for row in replay["claim_ok_contract"]["violations"]
    }
    assert "claim_ok_true_with_structural_contract_not_passed" in reasons


def test_replay_fails_when_an_energy_attempt_is_missing(
    tmp_path: Path,
) -> None:
    report, observations = _fixture_payloads()
    observations.pop()

    completed, replay = _run(tmp_path, report, observations)

    assert completed.returncode != 0
    failed_checks = {
        item["id"] for item in replay["checks"] if item["ok"] is False
    }
    assert "native_energy_attempt_count" in failed_checks
    assert "native_energy_failed_count" in failed_checks
    assert "reconstructed_row_count" in failed_checks
