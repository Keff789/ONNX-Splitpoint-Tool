from __future__ import annotations

import csv
import json
from pathlib import Path

from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
from onnx_splitpoint_tool.workflow.scientific_reporting import _write_reports


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_explicit_zero_present_count_survives_stale_native_summary(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    _write(
        reports / "native_expected_matrix.json",
        {
            "expected_row_count": 4,
            "present_expected_row_count": 0,
            "successful_expected_row_count": 0,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 4,
            "present_expected_rows": [],
            "missing_expected_rows": [{}, {}, {}, {}],
        },
    )
    # A resume directory may retain an older summary.  The new fail-closed
    # preflight matrix remains authoritative for the configured denominator.
    _write(
        reports / "native_producer_summary.json",
        {
            "rows": [
                {
                    "model": "resnet50",
                    "backend": "hailo8_to_trt",
                    "case": "b001",
                    "precision": "fp16",
                    "status": "ok",
                    "ok": True,
                    "fps_makespan": 100.0,
                }
            ]
        },
    )

    matrix = collect_native_performance_matrix(tmp_path)

    assert len(matrix["observations"]) == 1
    assert matrix["expected_row_count"] == 4
    assert matrix["present_expected_row_count"] == 0
    assert matrix["missing_expected_row_count"] == 4
    assert matrix["successful_expected_row_count"] == 0
    assert matrix["matrix_complete"] is False


def test_full_and_split_median_latency_ci_and_repetitions_reach_scientific_outputs(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    identities = [
        ("native_full_tensorrt", "full", "native_full_baseline", 8.0, 7.6, 8.4),
        ("hailo8_to_trt", "b038", "native_split", 5.0, 4.7, 5.3),
    ]
    concise_rows = []
    combined_rows = []
    expected_rows = []
    for backend, case, mode, median, low, high in identities:
        identity = {
            "model": "yolo26s",
            "backend": backend,
            "case": case,
            "precision": "fp16",
            "setup_id": "jetson-1",
            "comparison_backend": "hailo8",
        }
        # The concise compatibility value is intentionally different: the
        # scientific observation must use the combined report's median.
        concise_rows.append({
            **identity,
            "execution_mode": mode,
            "runtime_status": "ok",
            "fps": 999.0,
            "semantic_status": "claim_ok",
            "semantic_ok": True,
            "contract_consistent": True,
        })
        combined_rows.append({
            **identity,
            "execution_mode": mode,
            "status": "ok",
            "ok": True,
            "fps_makespan": 100.0,
            "fps_median": 100.0,
            "fps_ci95_low": 95.0,
            "fps_ci95_high": 105.0,
            "task": "detection",
            "repetition_records": [
                {"ok": True, "task": "detection", "measurement_endpoint": "completed_task",
                 "measurement_boundary": "workers_ready_to_last_completed_task_frame",
                 "postprocess_completion_verified": True, "completed_task_endpoint_attested": True,
                 "completed_task_endpoint_contract_hash": "c"*64,
                 "completed_work_units": 100, "makespan_ms": 100000 / fps,
                 "fps_makespan": fps, "repetition_id": f"fixture-{index}"}
                for index, fps in enumerate((95.0, 100.0, 105.0))
            ],
            "latency_mean_ms": median,
            "latency_median_ms": median,
            "latency_ci95_low_ms": low,
            "latency_ci95_high_ms": high,
            "latency_semantics": "outer_end_to_end_makespan",
            "repetition_count_requested": 3,
            "repetition_count_attempted": 3,
            "repetition_count_valid": 3,
            "repetition_status": "complete",
            "repetition_aggregation": "median_with_deterministic_percentile_bootstrap_ci95",
            "repetition_runtime_scope": "fresh_runtime_per_repetition",
            "repetition_independence_verified": True,
        })
        expected_rows.append(identity)

    _write(reports / "native_stage_concise_summary.json", {"rows": concise_rows})
    _write(reports / "native_producer_combined_summary.json", {"rows": combined_rows})
    _write(reports / "native_expected_matrix.json", {
        "expected_row_count": 2,
        "present_expected_row_count": 2,
        "missing_expected_row_count": 0,
        "present_expected_rows": expected_rows,
    })

    matrix = collect_native_performance_matrix(tmp_path)
    assert matrix["matrix_complete"] is True
    assert len(matrix["observations"]) == 2
    for row, (_, _, _, median, low, high) in zip(
        sorted(matrix["observations"], key=lambda value: value["execution_mode"]),
        sorted(identities, key=lambda value: value[2]),
    ):
        assert row["native_measured_throughput_fps"] == 100.0
        assert row["latency_ms"] == median
        assert row["latency_median_ms"] == median
        assert row["latency_ci95_low_ms"] == low
        assert row["latency_ci95_high_ms"] == high
        assert row["repetition_count_requested"] == 3
        assert row["repetition_count_attempted"] == 3
        assert row["repetition_count_valid"] == 3
        assert row["performance_statistic"] == "median_of_independent_repetitions"
        assert row["best_of_used"] is False

    report_root = reports / "scientific"
    _write_reports(report_root, {
        "created_at": "2026-07-20T00:00:00+02:00",
        "profile_id": "test",
        "rows": [],
        "summary": {},
        "native_performance_matrix": matrix,
    })
    exported = json.loads((report_root / "native_performance_observations.json").read_text(encoding="utf-8"))
    assert {row["latency_median_ms"] for row in exported} == {5.0, 8.0}
    with (report_root / "native_performance_observations.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert {row["latency_ci95_low_ms"] for row in csv_rows} == {"4.7", "7.6"}
    assert {row["repetition_count_valid"] for row in csv_rows} == {"3"}
    assert all(row["best_of_used"] == "False" for row in csv_rows)

    matrix_md = (report_root / "native_performance_matrix.md").read_text(encoding="utf-8")
    scientific_md = (report_root / "scientific_report.md").read_text(encoding="utf-8")
    table_tex = (report_root / "thesis_tables" / "native_performance_observations.tex").read_text(encoding="utf-8")
    assert "Legacy latency median [ms]" in matrix_md
    assert all(row["request_latency_mean_ms"] is None for row in exported)
    assert "never as a best-of value" in scientific_md
    assert "median and 95-percent confidence interval" in table_tex


def test_legacy_best_of_aggregation_is_retained_only_as_ineligible_diagnostic(
    tmp_path: Path,
) -> None:
    row = {
        "model": "resnet50",
        "backend": "native_full_tensorrt",
        "case": "full",
        "precision": "fp16",
        "setup_id": "jetson-1",
        "execution_mode": "native_full_baseline",
        "status": "ok",
        "ok": True,
        "fps_makespan": 120.0,
        "latency_median_ms": 8.3,
        "repetition_count_valid": 3,
        "repetition_status": "complete",
        "repetition_aggregation": "best_of_three",
        "performance_claim_eligible": True,
    }
    _write(tmp_path / "reports" / "native_producer_combined_summary.json", {"rows": [row]})

    observation = collect_native_performance_matrix(tmp_path)["observations"][0]
    assert observation["performance_statistic"] == "unsupported_best_of"
    assert observation["best_of_used"] is True
    assert observation["performance_contract_eligible"] is False
    assert observation["performance_eligible"] is False
    assert "best_of_aggregation_not_allowed" in observation["performance_claim_exclusion_reasons"]
