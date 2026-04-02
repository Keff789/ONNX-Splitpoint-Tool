import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.analysis import (
    comparison_candidate_rows,
    comparison_hailo_rows,
    comparison_provider_rows,
    export_benchmark_comparison,
    load_benchmark_analysis_comparison,
)
from onnx_splitpoint_tool.gui.panels import panel_benchmark_analysis


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)



def _make_suite(base: Path, *, suffix: str, hailo_ctx_b20: int, hailo_ms_b20: float, trt_best_ms_b10: float) -> Path:
    suite = base / suffix
    suite.mkdir(parents=True)
    bench = {
        "summary": {
            "requested_cases": 2,
            "generated_cases": 2,
            "preferred_shortlist_cases": 2,
            "candidate_search_pool": 4,
        },
        "cases": [
            {
                "boundary": 10,
                "case_dir": "b010",
                "manifest": "split_manifest.json",
                "predicted": {"score": 0.9, "cut_mib": 2.0},
                "hailo_compile": {
                    "hailo8": {
                        "part1": {"context_count": 1, "context_mode": "single_context_used"},
                        "part2": {
                            "context_count": 1,
                            "context_mode": "single_context_used",
                            "partition_time_s": 1.0,
                            "allocation_time_s": 2.0,
                            "compilation_time_s": 3.0,
                            "calib_source": "activation_from_part1",
                        },
                    }
                },
            },
            {
                "boundary": 20,
                "case_dir": "b020",
                "manifest": "split_manifest.json",
                "predicted": {"score": 0.6, "cut_mib": 5.0},
                "hailo_compile": {
                    "hailo8": {
                        "part1": {"context_count": 1, "context_mode": "single_context_used"},
                        "part2": {
                            "context_count": hailo_ctx_b20,
                            "context_mode": "single_context_used" if hailo_ctx_b20 == 1 else "single_context_failed_to_multi",
                            "partition_time_s": 5.0,
                            "allocation_time_s": 6.0,
                            "compilation_time_s": 7.0,
                        },
                    }
                },
            },
        ],
    }
    (suite / "benchmark_set.json").write_text(json.dumps(bench), encoding="utf-8")
    _write_csv(
        suite / "benchmark_results_hailo8.csv",
        [
            {
                "boundary": 10,
                "full_mean_ms": 25.0,
                "composed_mean_ms": 26.0,
                "final_pass_all": True,
                "score_pred": 0.9,
                "latency_total_ms": 6.0,
                "cut_mib": 2.0,
            },
            {
                "boundary": 20,
                "full_mean_ms": 25.0,
                "composed_mean_ms": hailo_ms_b20,
                "final_pass_all": True,
                "score_pred": 0.6,
                "latency_total_ms": 9.0,
                "cut_mib": 5.0,
            },
        ],
    )
    _write_csv(
        suite / "benchmark_results_ort_tensorrt.csv",
        [
            {
                "boundary": 10,
                "full_mean_ms": 34.0,
                "composed_mean_ms": trt_best_ms_b10,
                "final_pass_all": True,
                "score_pred": 0.9,
                "latency_total_ms": 7.0,
                "cut_mib": 2.0,
            },
            {
                "boundary": 20,
                "full_mean_ms": 34.0,
                "composed_mean_ms": 38.0,
                "final_pass_all": True,
                "score_pred": 0.6,
                "latency_total_ms": 11.0,
                "cut_mib": 5.0,
            },
        ],
    )
    return suite



def test_benchmark_analysis_comparison_export_and_summary(tmp_path: Path) -> None:
    suite_a = _make_suite(tmp_path, suffix="suite_a", hailo_ctx_b20=2, hailo_ms_b20=29.0, trt_best_ms_b10=36.0)
    suite_b = _make_suite(tmp_path, suffix="suite_b", hailo_ctx_b20=1, hailo_ms_b20=27.0, trt_best_ms_b10=35.0)

    comparison = load_benchmark_analysis_comparison(suite_a, suite_b, cache_base=tmp_path / "cache")
    assert "Benchmark-Vergleich" in comparison.summary_markdown
    assert "part2 Single-Context" in comparison.summary_markdown

    provider_rows = comparison_provider_rows(comparison)
    assert any(row["provider"] == "hailo8" for row in provider_rows)
    assert any(row["provider"] == "ort_tensorrt" for row in provider_rows)

    candidate_rows = comparison_candidate_rows(comparison)
    assert any(row["boundary"] == 10 for row in candidate_rows)

    hailo_rows = comparison_hailo_rows(comparison)
    row_b20 = next(row for row in hailo_rows if row["boundary"] == 20 and row["hw_arch"] == "hailo8")
    assert row_b20["left_part2_context_count"] == 2
    assert row_b20["right_part2_context_count"] == 1
    assert bool(row_b20["single_context_changed"]) is True

    out_dir = tmp_path / "export"
    paths = export_benchmark_comparison(comparison, out_dir)
    assert (out_dir / "benchmark_comparison_summary.md").exists()
    assert (out_dir / "benchmark_comparison_provider_summary.csv").exists()
    assert (out_dir / "benchmark_comparison_hailo_context_summary.csv").exists()
    assert (out_dir / "benchmark_comparison_provider_latency.pdf").exists()
    assert (out_dir / "benchmark_comparison_hailo_context_delta.svg").exists()
    assert "provider_csv" in paths



def test_benchmark_analysis_panel_builds_compare_controls() -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Tk unavailable: {exc}")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk not available in environment: {exc}")

    root.withdraw()
    app = type("App", (), {"root": root, "default_output_dir": str(root.tk.exprstring('$::env(HOME)')) if False else "."})()
    frame = panel_benchmark_analysis.build_panel(root, app=app)
    frame.pack(fill="both", expand=True)
    assert hasattr(frame, "var_benchmark_analysis_source")
    assert hasattr(frame, "var_benchmark_analysis_compare_source")
    assert frame.benchmark_analysis_state["comparison"] is None
    root.destroy()
