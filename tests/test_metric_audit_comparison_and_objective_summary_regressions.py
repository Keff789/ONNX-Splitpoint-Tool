from __future__ import annotations

import csv
import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.analysis import load_benchmark_analysis
from onnx_splitpoint_tool.benchmark.interleaving_analysis import (
    build_comparison_metric_audit_rank_error_figure,
    compute_interleaving_analysis,
    metric_audit_comparison_rows,
    metric_audit_comparison_summary,
)
from onnx_splitpoint_tool.objective_scoring import candidate_objective_summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_suite(base: Path, *, fps_shift: float) -> Path:
    suite = base / 'suite'
    suite.mkdir(parents=True)
    bench = {
        'summary': {'requested_cases': 2, 'generated_cases': 2},
        'cases': [
            {'boundary': 10, 'predicted': {'score': 0.4, 'cut_mib': 2.0, 'latency_total_ms': 8.0, 'n_cut_tensors': 12, 'unknown_crossing_tensors': 1, 'peak_act_right_mib': 4.0, 'imbalance': 0.10, 'hailo_compile_risk_score': 1.4, 'hailo_single_context_probability': 0.90}},
            {'boundary': 20, 'predicted': {'score': 0.9, 'cut_mib': 6.0, 'latency_total_ms': 12.0, 'n_cut_tensors': 32, 'unknown_crossing_tensors': 4, 'peak_act_right_mib': 10.0, 'imbalance': 0.45, 'hailo_compile_risk_score': 2.6, 'hailo_single_context_probability': 0.35}},
        ],
    }
    (suite / 'benchmark_set.json').write_text(json.dumps(bench), encoding='utf-8')
    _write_csv(
        suite / 'benchmark_results_hailo8_to_tensorrt.csv',
        [
            {'boundary': 10, 'final_pass_all': True, 'score_pred': 0.4, 'cut_mib': 2.0, 'stage1_provider': 'hailo8', 'stage2_provider': 'tensorrt', 'part1_mean_ms': 15.0, 'part2_mean_ms': 20.0, 'composed_mean_ms': 37.0, 'overhead_ms': 2.0, 'throughput_stage1_mean_ms': 15.0, 'throughput_stage2_mean_ms': 20.0, 'throughput_fps_cycle_est': 45.0 + fps_shift, 'throughput_cycle_est_ms': 22.0, 'throughput_latency_mean_ms': 40.0},
            {'boundary': 20, 'final_pass_all': True, 'score_pred': 0.9, 'cut_mib': 6.0, 'stage1_provider': 'hailo8', 'stage2_provider': 'tensorrt', 'part1_mean_ms': 18.0, 'part2_mean_ms': 19.0, 'composed_mean_ms': 39.0, 'overhead_ms': 8.0, 'throughput_stage1_mean_ms': 18.0, 'throughput_stage2_mean_ms': 19.0, 'throughput_fps_cycle_est': 36.0 + fps_shift, 'throughput_cycle_est_ms': 27.8, 'throughput_latency_mean_ms': 44.0},
        ],
    )
    return suite


def test_metric_audit_comparison_rows_and_summary(tmp_path: Path) -> None:
    left_suite = _make_suite(tmp_path / 'left', fps_shift=0.0)
    right_suite = _make_suite(tmp_path / 'right', fps_shift=5.0)
    left_report = load_benchmark_analysis(left_suite, cache_base=tmp_path / 'cache_left')
    right_report = load_benchmark_analysis(right_suite, cache_base=tmp_path / 'cache_right')
    left_inter = compute_interleaving_analysis(left_report)
    right_inter = compute_interleaving_analysis(right_report)

    rows = metric_audit_comparison_rows(left_report, left_inter, right_report, right_inter)
    assert [r['boundary'] for r in rows] == [10, 20]
    assert rows[0]['fps_delta_abs'] is not None
    assert rows[0]['new_rank_error_abs_delta'] is not None

    summary = metric_audit_comparison_summary(left_report, left_inter, right_report, right_inter)
    assert summary['common_case_count'] == 2
    assert summary['new_spearman_rank_delta'] is not None
    assert summary['new_mean_abs_rank_error_delta'] is not None

    assert build_comparison_metric_audit_rank_error_figure(left_report, left_inter, right_report, right_inter) is not None


def test_candidate_objective_summary_prefers_throughput_text() -> None:
    row = {
        'boundary': 118,
        'score_pred': 0.42,
        'cut_mb_val': 3.5,
        'n_cut_tensors': 16,
        'unknown_count': 1,
        'peak_right_mib_val': 5.0,
        'hailo_compile_risk_score': 1.4,
        'hailo_single_context_probability': 0.88,
        'pred_latency_total_ms': 9.0,
        'flops_left_abs': 20.0,
        'flops_right_abs': 10.0,
    }
    summary = candidate_objective_summary(row, objective='Throughput', stage1='hailo8', stage2='tensorrt')
    assert 'Throughput' in summary['title']
    assert 'b118' in summary['headline']
    assert 'pred TH' in summary['detail']
