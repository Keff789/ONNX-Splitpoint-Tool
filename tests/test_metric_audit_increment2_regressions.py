import csv
import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.analysis import load_benchmark_analysis
from onnx_splitpoint_tool.benchmark.interleaving_analysis import (
    build_metric_audit_fps_figure,
    build_metric_audit_rank_figure,
    compute_interleaving_analysis,
    metric_audit_rows,
    metric_audit_summary,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_suite(base: Path) -> Path:
    suite = base / "suite"
    suite.mkdir(parents=True)
    bench = {
        "summary": {"requested_cases": 2, "generated_cases": 2},
        "cases": [
            {
                "boundary": 10,
                "predicted": {
                    "score": 0.4,
                    "cut_mib": 2.0,
                    "latency_total_ms": 8.0,
                    "n_cut_tensors": 12,
                    "unknown_crossing_tensors": 1,
                    "peak_act_right_mib": 4.0,
                    "imbalance": 0.10,
                    "hailo_compile_risk_score": 1.4,
                    "hailo_single_context_probability": 0.90,
                },
            },
            {
                "boundary": 20,
                "predicted": {
                    "score": 0.9,
                    "cut_mib": 6.0,
                    "latency_total_ms": 12.0,
                    "n_cut_tensors": 32,
                    "unknown_crossing_tensors": 4,
                    "peak_act_right_mib": 10.0,
                    "imbalance": 0.45,
                    "hailo_compile_risk_score": 2.6,
                    "hailo_single_context_probability": 0.35,
                },
            },
        ],
    }
    (suite / "benchmark_set.json").write_text(json.dumps(bench), encoding="utf-8")
    _write_csv(
        suite / "benchmark_results_hailo8_to_tensorrt.csv",
        [
            {
                "boundary": 10,
                "final_pass_all": True,
                "score_pred": 0.4,
                "cut_mib": 2.0,
                "stage1_provider": "hailo8",
                "stage2_provider": "tensorrt",
                "part1_mean_ms": 15.0,
                "part2_mean_ms": 20.0,
                "composed_mean_ms": 37.0,
                "overhead_ms": 2.0,
                "throughput_stage1_mean_ms": 15.0,
                "throughput_stage2_mean_ms": 20.0,
                "throughput_fps_cycle_est": 45.0,
                "throughput_cycle_est_ms": 22.0,
                "throughput_latency_mean_ms": 40.0,
            },
            {
                "boundary": 20,
                "final_pass_all": True,
                "score_pred": 0.9,
                "cut_mib": 6.0,
                "stage1_provider": "hailo8",
                "stage2_provider": "tensorrt",
                "part1_mean_ms": 18.0,
                "part2_mean_ms": 19.0,
                "composed_mean_ms": 39.0,
                "overhead_ms": 8.0,
                "throughput_stage1_mean_ms": 18.0,
                "throughput_stage2_mean_ms": 19.0,
                "throughput_fps_cycle_est": 36.0,
                "throughput_cycle_est_ms": 27.8,
                "throughput_latency_mean_ms": 44.0,
            },
        ],
    )
    _write_csv(
        suite / "benchmark_results_hailo8.csv",
        [
            {"boundary": 10, "full_mean_ms": 25.0, "composed_mean_ms": 30.0, "final_pass_all": True, "score_pred": 0.4, "latency_total_ms": 8.0, "cut_mib": 2.0},
            {"boundary": 20, "full_mean_ms": 25.0, "composed_mean_ms": 31.0, "final_pass_all": True, "score_pred": 0.9, "latency_total_ms": 12.0, "cut_mib": 6.0},
        ],
    )
    _write_csv(
        suite / "benchmark_results_ort_tensorrt.csv",
        [
            {"boundary": 10, "full_mean_ms": 34.0, "composed_mean_ms": 36.0, "final_pass_all": True, "score_pred": 0.4, "latency_total_ms": 8.0, "cut_mib": 2.0},
            {"boundary": 20, "full_mean_ms": 34.0, "composed_mean_ms": 38.0, "final_pass_all": True, "score_pred": 0.9, "latency_total_ms": 12.0, "cut_mib": 6.0},
        ],
    )
    return suite


def test_metric_audit_rows_and_summary(tmp_path: Path) -> None:
    suite = _make_suite(tmp_path)
    report = load_benchmark_analysis(suite, cache_base=tmp_path / "cache")
    inter = compute_interleaving_analysis(report)

    rows = metric_audit_rows(report, inter)
    assert [r["boundary"] for r in rows] == [10, 20]
    assert rows[0]["predicted_rank_old"] == 1
    assert rows[0]["predicted_stream_fps"] is not None
    assert rows[0]["hailo_feasibility_risk"] is not None
    assert rows[0]["hailo_interface_penalty"] is not None
    assert rows[1]["predicted_handover_ms"] is not None
    assert rows[1]["rank_error_new_abs"] is not None

    summary = metric_audit_summary(report, inter)
    assert summary["count"] == 2
    assert summary["best_boundary"] == 10
    assert summary["old_top1_hit"] is True
    assert summary["new_top3_hit"] is True

    assert build_metric_audit_rank_figure(report, inter) is not None
    assert build_metric_audit_fps_figure(report, inter) is not None
