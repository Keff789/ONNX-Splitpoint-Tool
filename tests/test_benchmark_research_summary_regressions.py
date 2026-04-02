import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.analysis import load_benchmark_analysis
from onnx_splitpoint_tool.benchmark.interleaving_analysis import (
    build_interleaving_residual_overhead_figure,
    compute_interleaving_analysis,
    research_best_full_vs_split_rows,
    research_prediction_audit_rows,
    research_stage_breakdown_rows,
    research_summary_cards,
)
from onnx_splitpoint_tool.gui.panels import panel_benchmark_analysis


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
        "summary": {
            "requested_cases": 2,
            "generated_cases": 2,
            "preferred_shortlist_cases": 3,
            "candidate_search_pool": 6,
        },
        "cases": [
            {"boundary": 10, "case_dir": "b010", "manifest": "split_manifest.json", "predicted": {"score": 0.4, "cut_mib": 2.0}},
            {"boundary": 20, "case_dir": "b020", "manifest": "split_manifest.json", "predicted": {"score": 0.9, "cut_mib": 4.5}},
        ],
    }
    (suite / "benchmark_set.json").write_text(json.dumps(bench), encoding="utf-8")
    _write_csv(
        suite / "benchmark_results_hailo8.csv",
        [
            {"boundary": 10, "full_mean_ms": 25.0, "composed_mean_ms": 30.0, "final_pass_all": True, "score_pred": 0.4, "latency_total_ms": 6.0, "cut_mib": 2.0},
            {"boundary": 20, "full_mean_ms": 25.0, "composed_mean_ms": 31.0, "final_pass_all": True, "score_pred": 0.9, "latency_total_ms": 8.0, "cut_mib": 4.5},
        ],
    )
    _write_csv(
        suite / "benchmark_results_ort_tensorrt.csv",
        [
            {"boundary": 10, "full_mean_ms": 34.0, "composed_mean_ms": 36.0, "final_pass_all": True, "score_pred": 0.4, "latency_total_ms": 7.0, "cut_mib": 2.0},
            {"boundary": 20, "full_mean_ms": 34.0, "composed_mean_ms": 38.0, "final_pass_all": True, "score_pred": 0.9, "latency_total_ms": 11.0, "cut_mib": 4.5},
        ],
    )
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
                "cut_mib": 4.5,
                "stage1_provider": "hailo8",
                "stage2_provider": "tensorrt",
                "part1_mean_ms": 18.0,
                "part2_mean_ms": 19.0,
                "composed_mean_ms": 39.0,
                "overhead_ms": 1.0,
                "throughput_stage1_mean_ms": 18.0,
                "throughput_stage2_mean_ms": 19.0,
                "throughput_fps_cycle_est": 49.0,
                "throughput_cycle_est_ms": 20.5,
                "throughput_latency_mean_ms": 44.0,
            },
        ],
    )
    return suite


def test_research_summary_helpers_and_residual_plot(tmp_path: Path) -> None:
    suite = _make_suite(tmp_path)
    report = load_benchmark_analysis(suite, cache_base=tmp_path / "cache")
    inter = compute_interleaving_analysis(report)

    best_rows = research_best_full_vs_split_rows(report, inter)
    assert len(best_rows) == 2
    assert best_rows[0]["role"] == "best_full"
    assert best_rows[1]["role"] == "best_split"
    assert best_rows[1]["boundary"] == 20

    audit_rows = research_prediction_audit_rows(report, inter)
    assert [row["boundary"] for row in audit_rows] == [20, 10]
    assert audit_rows[0]["actual_rank"] == 1
    assert audit_rows[0]["predicted_rank"] == 2

    stage_rows = research_stage_breakdown_rows(inter)
    assert stage_rows[0]["boundary"] == 20
    assert stage_rows[0]["residual_overhead_ms"] is not None

    cards = research_summary_cards(report, inter)
    assert cards["best_full_label"] == "hailo8"
    assert cards["best_split_predicted_rank"] == 2
    assert cards["top3_hit"] is True

    fig = build_interleaving_residual_overhead_figure(inter)
    assert fig is not None


def test_benchmark_analysis_panel_exposes_research_summary_tab() -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Tk unavailable: {exc}")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk not available in environment: {exc}")

    root.withdraw()
    app = type("App", (), {"root": root, "default_output_dir": "."})()
    frame = panel_benchmark_analysis.build_panel(root, app=app)
    frame.pack(fill="both", expand=True)
    # The tab state should be present and a load should be possible later.
    assert hasattr(frame, "benchmark_analysis_state")
    root.destroy()
