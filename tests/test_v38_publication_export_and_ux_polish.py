from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.benchmark.analysis import load_benchmark_analysis, load_benchmark_analysis_comparison
from onnx_splitpoint_tool.benchmark.interleaving_analysis import (
    compute_interleaving_analysis,
    export_publication_analysis,
    export_publication_comparison,
)
from onnx_splitpoint_tool.gui.panels import panel_benchmark_analysis


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_suite(base: Path, *, score_shift: float = 0.0, fps_shift: float = 0.0) -> Path:
    suite = base / 'suite'
    suite.mkdir(parents=True)
    bench = {
        'summary': {'requested_cases': 2, 'generated_cases': 2, 'objective': 'throughput'},
        'cases': [
            {'boundary': 10, 'predicted': {'score': 0.4 + score_shift, 'cut_mib': 2.0, 'latency_total_ms': 8.0, 'n_cut_tensors': 12, 'unknown_crossing_tensors': 1, 'peak_act_right_mib': 4.0, 'imbalance': 0.10, 'hailo_compile_risk_score': 1.4, 'hailo_single_context_probability': 0.90}},
            {'boundary': 20, 'predicted': {'score': 0.9 + score_shift, 'cut_mib': 6.0, 'latency_total_ms': 12.0, 'n_cut_tensors': 32, 'unknown_crossing_tensors': 4, 'peak_act_right_mib': 10.0, 'imbalance': 0.55, 'hailo_compile_risk_score': 2.6, 'hailo_single_context_probability': 0.35}},
        ],
    }
    (suite / 'benchmark_set.json').write_text(json.dumps(bench), encoding='utf-8')
    _write_csv(
        suite / 'benchmark_results_hailo8.csv',
        [
            {'boundary': 10, 'full_mean_ms': 25.0, 'composed_mean_ms': 30.0, 'final_pass_all': True, 'score_pred': 0.4 + score_shift, 'latency_total_ms': 8.0, 'cut_mib': 2.0},
            {'boundary': 20, 'full_mean_ms': 25.0, 'composed_mean_ms': 31.0, 'final_pass_all': True, 'score_pred': 0.9 + score_shift, 'latency_total_ms': 12.0, 'cut_mib': 6.0},
        ],
    )
    _write_csv(
        suite / 'benchmark_results_hailo8_to_tensorrt.csv',
        [
            {'boundary': 10, 'final_pass_all': True, 'score_pred': 0.4 + score_shift, 'cut_mib': 2.0, 'stage1_provider': 'hailo8', 'stage2_provider': 'tensorrt', 'part1_mean_ms': 15.0, 'part2_mean_ms': 20.0, 'composed_mean_ms': 37.0, 'overhead_ms': 2.0, 'throughput_stage1_mean_ms': 15.0, 'throughput_stage2_mean_ms': 20.0, 'throughput_fps_cycle_est': 45.0 + fps_shift, 'throughput_cycle_est_ms': 22.0, 'throughput_latency_mean_ms': 40.0},
            {'boundary': 20, 'final_pass_all': True, 'score_pred': 0.9 + score_shift, 'cut_mib': 6.0, 'stage1_provider': 'hailo8', 'stage2_provider': 'tensorrt', 'part1_mean_ms': 18.0, 'part2_mean_ms': 19.0, 'composed_mean_ms': 39.0, 'overhead_ms': 8.0, 'throughput_stage1_mean_ms': 18.0, 'throughput_stage2_mean_ms': 19.0, 'throughput_fps_cycle_est': 36.0 + fps_shift, 'throughput_cycle_est_ms': 27.8, 'throughput_latency_mean_ms': 44.0},
        ],
    )
    return suite


def test_publication_exports_single_and_comparison(tmp_path: Path) -> None:
    suite_a = _make_suite(tmp_path / 'a', score_shift=0.0, fps_shift=0.0)
    suite_b = _make_suite(tmp_path / 'b', score_shift=-0.1, fps_shift=4.0)

    report_a = load_benchmark_analysis(suite_a, cache_base=tmp_path / 'cache_a')
    inter_a = compute_interleaving_analysis(report_a)
    out_a = export_publication_analysis(report_a, inter_a, tmp_path / 'pub_a', use_calibration=True)
    assert (tmp_path / 'pub_a' / 'publication_best_full_vs_split.tex').exists()
    assert (tmp_path / 'pub_a' / 'publication_metric_audit_cal.csv').exists()
    assert out_a['publication_summary_md'].exists()

    cmp = load_benchmark_analysis_comparison(suite_a, suite_b, cache_base=tmp_path / 'cache_cmp')
    inter_b = compute_interleaving_analysis(cmp.right)
    out_cmp = export_publication_comparison(cmp.left, inter_a, cmp.right, inter_b, tmp_path / 'pub_cmp', use_calibration=True)
    assert (tmp_path / 'pub_cmp' / 'publication_comparison_research_summary.tex').exists()
    assert (tmp_path / 'pub_cmp' / 'publication_comparison_metric_audit.csv').exists()
    assert out_cmp['publication_comparison_summary_md'].exists()


def test_benchmark_analysis_panel_has_publication_export_and_research_compare_tab() -> None:
    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover
        pytest.skip(f'Tk unavailable: {exc}')

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tk not available in environment: {exc}')

    root.withdraw()
    app = type('App', (), {'root': root, 'default_output_dir': '.'})()
    frame = panel_benchmark_analysis.build_panel(root, app=app)
    frame.pack(fill='both', expand=True)
    src = Path('onnx_splitpoint_tool/gui/panels/panel_benchmark_analysis.py').read_text(encoding='utf-8')
    assert 'Publication export…' in src
    assert 'Research-Vergleich' in src
    root.destroy()


def test_panel_analysis_has_single_calibration_checkbox_block() -> None:
    src = Path('onnx_splitpoint_tool/gui/panels/panel_analysis.py').read_text(encoding='utf-8')
    assert src.count('Use TH calibration') == 1
    assert '_analysis_calibration_badge' in src
