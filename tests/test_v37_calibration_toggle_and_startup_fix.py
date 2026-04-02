from __future__ import annotations

import csv
import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.analysis import load_benchmark_analysis
from onnx_splitpoint_tool.benchmark.interleaving_analysis import (
    compute_interleaving_analysis,
    metric_audit_rows,
    metric_audit_summary,
)
from onnx_splitpoint_tool.objective_scoring import candidate_objective_metrics


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_suite(base: Path) -> Path:
    suite = base / 'suite'
    suite.mkdir(parents=True)
    bench = {
        'summary': {'requested_cases': 2, 'generated_cases': 2, 'objective': 'throughput'},
        'cases': [
            {'boundary': 10, 'predicted': {'score': 0.4, 'cut_mib': 2.0, 'latency_total_ms': 8.0, 'n_cut_tensors': 12, 'unknown_crossing_tensors': 1, 'peak_act_right_mib': 4.0, 'imbalance': 0.10, 'hailo_compile_risk_score': 1.4, 'hailo_single_context_probability': 0.90}},
            {'boundary': 20, 'predicted': {'score': 0.9, 'cut_mib': 6.0, 'latency_total_ms': 12.0, 'n_cut_tensors': 32, 'unknown_crossing_tensors': 4, 'peak_act_right_mib': 10.0, 'imbalance': 0.55, 'hailo_compile_risk_score': 2.6, 'hailo_single_context_probability': 0.35}},
        ],
    }
    (suite / 'benchmark_set.json').write_text(json.dumps(bench), encoding='utf-8')
    _write_csv(
        suite / 'benchmark_results_hailo8_to_tensorrt.csv',
        [
            {'boundary': 10, 'final_pass_all': True, 'score_pred': 0.4, 'cut_mib': 2.0, 'stage1_provider': 'hailo8', 'stage2_provider': 'tensorrt', 'part1_mean_ms': 15.0, 'part2_mean_ms': 20.0, 'composed_mean_ms': 37.0, 'overhead_ms': 2.0, 'throughput_stage1_mean_ms': 15.0, 'throughput_stage2_mean_ms': 20.0, 'throughput_fps_cycle_est': 45.0, 'throughput_cycle_est_ms': 22.0, 'throughput_latency_mean_ms': 40.0},
            {'boundary': 20, 'final_pass_all': True, 'score_pred': 0.9, 'cut_mib': 6.0, 'stage1_provider': 'hailo8', 'stage2_provider': 'tensorrt', 'part1_mean_ms': 18.0, 'part2_mean_ms': 19.0, 'composed_mean_ms': 39.0, 'overhead_ms': 8.0, 'throughput_stage1_mean_ms': 18.0, 'throughput_stage2_mean_ms': 19.0, 'throughput_fps_cycle_est': 36.0, 'throughput_cycle_est_ms': 27.8, 'throughput_latency_mean_ms': 44.0},
        ],
    )
    return suite


def test_analysis_panel_uses_static_labelframe_text() -> None:
    src = Path('onnx_splitpoint_tool/gui/panels/panel_analysis.py').read_text(encoding='utf-8')
    assert "Objective summary" in src
    assert 'ttk.LabelFrame(table_frame, textvariable=app.var_cand_objective_title)' not in src


def test_candidate_metrics_expose_calibrated_and_uncalibrated_values() -> None:
    row = {
        'cut_mb_val': 6.0,
        'n_cut_tensors': 32,
        'unknown_count': 4,
        'peak_right_mib_val': 10.0,
        'hailo_compile_risk_score': 2.6,
        'hailo_single_context_probability': 0.35,
        'pred_latency_total_ms': 12.0,
        'flops_left_abs': 30.0,
        'flops_right_abs': 10.0,
        'imbalance': 0.55,
    }
    metrics_cal = candidate_objective_metrics(row, stage1='hailo8', stage2='tensorrt', use_calibration=True)
    metrics_raw = candidate_objective_metrics(row, stage1='hailo8', stage2='tensorrt', use_calibration=False)
    assert metrics_cal['predicted_stream_fps_calibrated'] is not None
    assert metrics_raw['predicted_stream_fps_uncalibrated'] is not None
    assert metrics_cal['predicted_stream_fps'] == metrics_cal['predicted_stream_fps_calibrated']
    assert metrics_raw['predicted_stream_fps'] == metrics_raw['predicted_stream_fps_uncalibrated']


def test_metric_audit_toggle_exposes_uncal_and_cal_summary(tmp_path: Path) -> None:
    suite = _make_suite(tmp_path)
    report = load_benchmark_analysis(suite, cache_base=tmp_path / 'cache')
    inter = compute_interleaving_analysis(report)

    rows_raw = metric_audit_rows(report, inter, use_calibration=False)
    rows_cal = metric_audit_rows(report, inter, use_calibration=True)
    assert rows_raw and rows_cal
    assert 'predicted_rank_uncal' in rows_raw[0]
    assert 'predicted_rank_cal' in rows_cal[0]

    summary_raw = metric_audit_summary(report, inter, use_calibration=False)
    summary_cal = metric_audit_summary(report, inter, use_calibration=True)
    assert 'uncal_spearman_rank' in summary_raw
    assert 'cal_spearman_rank' in summary_cal
    assert summary_raw['calibration_enabled'] is False
    assert summary_cal['calibration_enabled'] is True
