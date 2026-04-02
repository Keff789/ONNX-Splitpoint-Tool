import csv
import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.analysis import (
    export_benchmark_analysis,
    hailo_context_rows,
    hailo_outlook_rows,
    hailo_outlook_summary,
    hailo_part2_fallback_rows,
    hailo_part2_fallback_summary,
    load_benchmark_analysis,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_benchmark_analysis_exports_hailo_context_tables_and_pdf_svg(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir(parents=True)
    bench = {
        "summary": {
            "requested_cases": 2,
            "generated_cases": 2,
            "preferred_shortlist_cases": 3,
            "candidate_search_pool": 8,
        },
        "cases": [
            {
                "boundary": 10,
                "case_dir": "b010",
                "manifest": "split_manifest.json",
                "predicted": {"score": 0.9, "cut_mib": 2.0},
                "hailo_part2_output_strategy": "hailo_parser_suggested_end_nodes",
                "hailo_part2_effective_outputs": ["t2", "t3"],
                "hailo_compile": {
                    "hailo8": {
                        "part1": {"context_count": 1, "context_mode": "single_context_used", "partition_time_s": 1.2},
                        "part2": {
                            "context_count": 1,
                            "context_mode": "single_context_used",
                            "partition_time_s": 2.3,
                            "allocation_time_s": 3.4,
                            "compilation_time_s": 4.5,
                            "elapsed_s": 10.0,
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
                            "context_count": 2,
                            "context_mode": "single_context_failed_to_multi",
                            "partition_time_s": 8.0,
                            "allocation_time_s": 9.0,
                            "compilation_time_s": 10.0,
                        },
                    }
                },
            },
        ],
    }
    (suite / "benchmark_set.json").write_text(json.dumps(bench), encoding="utf-8")

    rows_hailo8 = [
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
            "composed_mean_ms": 29.0,
            "final_pass_all": True,
            "score_pred": 0.6,
            "latency_total_ms": 9.0,
            "cut_mib": 5.0,
        },
    ]
    rows_trt = [
        {
            "boundary": 10,
            "full_mean_ms": 34.0,
            "composed_mean_ms": 36.0,
            "final_pass_all": True,
            "score_pred": 0.9,
            "latency_total_ms": 7.0,
            "cut_mib": 2.0,
        },
        {
            "boundary": 20,
            "full_mean_ms": 34.0,
            "composed_mean_ms": 35.0,
            "final_pass_all": True,
            "score_pred": 0.6,
            "latency_total_ms": 5.5,
            "cut_mib": 5.0,
        },
    ]
    _write_csv(suite / "benchmark_results_hailo8.csv", rows_hailo8)
    _write_csv(suite / "benchmark_results_ort_tensorrt.csv", rows_trt)

    report = load_benchmark_analysis(suite, cache_base=tmp_path / "cache")
    hailo_rows = hailo_context_rows(report)
    assert len(hailo_rows) == 2
    assert any(bool(row["part2_single_context"]) for row in hailo_rows)
    outlook_rows = hailo_outlook_rows(report)
    outlook_summary = hailo_outlook_summary(report)
    fallback_rows = hailo_part2_fallback_rows(report)
    fallback_summary = hailo_part2_fallback_summary(report)
    assert outlook_rows
    assert outlook_summary["candidate_count"] == 2
    assert fallback_rows
    assert fallback_rows[0]["marker"] == "↩"
    assert fallback_summary["fallback_count"] == 1
    assert "## Hailo Context Fit" in report.summary_markdown
    assert "## Hailo Part2 Suggested-End Fallback" in report.summary_markdown
    assert "activation_from_part1" in report.summary_markdown or any(row["part2_calib_source"] == "activation_from_part1" for row in hailo_rows)

    out_dir = tmp_path / "export"
    paths = export_benchmark_analysis(report, out_dir)
    assert (out_dir / "benchmark_analysis_hailo_context_fit.csv").exists()
    assert (out_dir / "benchmark_analysis_hailo_outlook.csv").exists()
    assert (out_dir / "benchmark_analysis_hailo_part2_fallback.csv").exists()
    assert (out_dir / "benchmark_analysis_hailo_context_fit.png").exists()
    assert (out_dir / "benchmark_analysis_hailo_context_fit.pdf").exists()
    assert (out_dir / "benchmark_analysis_hailo_context_fit.svg").exists()
    assert (out_dir / "benchmark_analysis_hailo_part2_fallback.png").exists()
    assert (out_dir / "benchmark_analysis_captions.md").exists()
    assert "hailo_context_csv" in paths
    assert "hailo_fallback_csv" in paths
