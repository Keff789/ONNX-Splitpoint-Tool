"""Real archived report exports, thread safety, units and current release."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from test_v290_r9l_scope import A7
from onnx_splitpoint_tool.workflow import scientific_reporting as reports
from onnx_splitpoint_tool.workflow import dashboard
from onnx_splitpoint_tool.native_energy_reporting import _repeat_statistics
from onnx_splitpoint_tool.release_identity import VERSION, BUILD_ID

SOURCE = A7 / 'reports/scientific/scientific_report.json'


def payload():
    return json.loads(SOURCE.read_text())


def test_export_worker_uses_no_gui_manager_or_global_backend_switch(tmp_path, monkeypatch):
    import matplotlib
    import matplotlib.pyplot as plt
    backend = matplotlib.get_backend()
    def forbidden(*args, **kwargs): raise AssertionError('GUI pyplot/global backend touched')
    for name in ('subplots', 'figure', 'savefig', 'close'): monkeypatch.setattr(plt, name, forbidden)
    monkeypatch.setattr(matplotlib, 'use', forbidden)
    original = payload(); before = deepcopy(original)
    with warnings.catch_warnings(record=True) as caught, ThreadPoolExecutor(1) as pool:
        pool.submit(reports._write_reports, tmp_path / 'scientific', original).result()
        pool.submit(dashboard._try_write_figures, tmp_path / 'dashboard', [dict(model_id='a', best_complete_split_latency_ms=5, best_pipeline_fps=200), dict(model_id='missing')], [], [], []).result()
        pool.submit(dashboard._try_write_claim_figures, tmp_path / 'dashboard', [dict(model_id='a', pipeline_speedup_vs_tensorrt_full=1.1, energy_ratio_vs_tensorrt_command_window=.9)]).result()
        pool.submit(dashboard._write_claim_analysis_figures, tmp_path / 'dashboard', [dict(model_id='a', row_group='best_split', pipeline_fps=200, single_detection_latency_ms=5, fps_per_watt_selected_fps=20, throughput_speedup_vs_fastest_full=1.1, fps_per_watt_ratio_vs_most_efficient_full=1.2)]).result()
    assert matplotlib.get_backend() == backend
    assert not any('GUI outside' in str(w.message) for w in caught)
    assert original['rows'] == before['rows']
    saved = json.loads((tmp_path / 'scientific/native_performance_observations.json').read_text())
    assert saved == before['native_performance_matrix']['observations']
    energy = json.loads((tmp_path / 'scientific/native_energy_observations.json').read_text())
    assert energy == before['native_energy_observations']
    assert (tmp_path / 'scientific/figures/screening_energy_observations.png').is_file()


def test_real_tk_canvas_survives_reporting_worker(tmp_path):
    import tkinter as tk
    import matplotlib
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    root = tk.Tk(); root.withdraw()
    try:
        figure = Figure(); figure.subplots().plot([0, 1], [0, 1])
        canvas = FigureCanvasTkAgg(figure, master=root); canvas.draw()
        backend = matplotlib.get_backend()
        with warnings.catch_warnings(record=True) as caught, ThreadPoolExecutor(1) as pool:
            pool.submit(reports._make_figures, payload()['rows'], [], [], tmp_path).result()
        canvas.draw(); root.update_idletasks()
        assert matplotlib.get_backend() == backend
        assert not any('GUI outside' in str(w.message) for w in caught)
    finally:
        root.destroy()


def test_measured_request_latency_and_energy_denominators_remain_distinct():
    data = payload()
    for row in data['native_performance_matrix']['observations']:
        evidence = row['request_latency']
        pairs = [pair for rep in evidence['repetitions'] for pair in rep['request_latency']['pairs']]
        measured = np.array([(pair[2]-pair[1])/1e6 for pair in pairs])
        assert row['request_latency_mean_ms'] == pytest.approx(measured.mean())
        assert row['request_latency_p50_ms'] == pytest.approx(np.percentile(measured, 50))
        assert row['request_latency_p95_ms'] == pytest.approx(np.percentile(measured, 95))
    for row in data['native_energy_observations']:
        assert row['energy_scope'] == 'full_system'
        repeats = row['energy_normalization_repeats']
        assert row['energy_repeat_n'] == len(repeats) == 3
        assert row['energy_per_work_j'] == pytest.approx(np.mean([r['energy_total_j']/r['energy_work_units_used'] for r in repeats]))
        assert row['energy_per_work_j_sample_stddev'] == pytest.approx(np.std([r['energy_per_work_unit_j'] for r in repeats], ddof=1))
    single = _repeat_statistics([1.25], .95)
    assert single['sample_stddev'] is None
    assert single['ci_low'] is None and single['ci_high'] is None


def test_table_missing_values_and_boolean_false_are_visible(tmp_path):
    table = tmp_path / 'table.tex'
    reports._write_tex_table(table, [dict(name='missing', value=None, claim=False, reason='not measured')],
        [('name', 'Model', 'text'), ('value', 'Energy [J/image]', 'number'), ('claim', 'Claim', 'text'), ('reason', 'Reason', 'text')],
        'Raw FS input energy.', 'tab:missing', preview=True)
    assert 'N/A' in table.read_text() and 'False' in table.read_text() and 'not measured' in table.read_text()
    assert (tmp_path / 'table.png').is_file()
    assert (tmp_path / 'table.pdf').is_file()


def test_dashboard_missing_metric_is_not_zero(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    bars=[]; original=Axes.bar
    def capture(self,x,height,*args,**kwargs):
        bars.append(list(height)); return original(self,x,height,*args,**kwargs)
    monkeypatch.setattr(Axes,'bar',capture)
    dashboard._try_write_figures(tmp_path, [dict(model_id='measured', best_pipeline_fps=12), dict(model_id='missing')], [], [], [])
    assert bars and bars[0][0] == 12 and np.isnan(bars[0][1])


def test_current_release_and_gui_identity():
    import ast
    import tomllib
    from onnx_splitpoint_tool import __version__, __build_id__
    root=Path(__file__).resolve().parents[1]
    assert VERSION == __version__ == '2.91.0'
    assert BUILD_ID == __build_id__ == 'v2.91.0'
    assert tomllib.loads((root/'pyproject.toml').read_text())['project']['version'] == VERSION
    lock=tomllib.loads((root/'uv.lock').read_text())
    assert next(p for p in lock['package'] if p['name']=='onnx-splitpoint-tool')['version']==VERSION
    # Importing the full GUI selects Tk; inspect its central identity binding
    # here and exercise a real Tk canvas separately, without starting workflows.
    tree=ast.parse((root/'onnx_splitpoint_tool/gui/app.py').read_text())
    assert any(isinstance(n,ast.ImportFrom) and any(a.name=='__release__' and a.asname=='TOOL_VERSION' for a in n.names) for n in tree.body)


def test_archived_eval_technical_negatives_export_na_and_reasons(tmp_path):
    from test_v283_r9k_raw_full_contract import RUN
    path = RUN / 'reports/scientific/native_performance_observations.json'
    source = path.read_bytes(); rows = json.loads(source)
    failed = [r for r in rows if r.get('runtime_executable') is False]
    assert len(failed) == 6 and {r['model_id'] for r in failed} == {'yolov7_paper'}
    table = tmp_path / 'negative.tex'
    reports._write_tex_table(table, failed,
        [('backend','Backend','text'), ('native_measured_throughput_fps','Completed FPS','number'),
         ('request_latency_mean_ms','Request mean [ms]','number'), ('request_latency_unavailable_reason','Reason','text')],
        'Original EVAL1 technical failures remain unavailable.', 'tab:negative', preview=True)
    for row in failed:
        assert row['native_measured_throughput_fps'] is None and row['request_latency_mean_ms'] is None
        assert row['request_latency_unavailable_reason'] in table.read_text().replace('\\_', '_')
    assert 'N/A' in table.read_text()
    assert path.read_bytes() == source


def test_table_preview_allocates_space_for_real_header_glyphs():
    from onnx_splitpoint_tool.reporting_figures import table_preview
    figure = table_preview(['Model / backend','Mean J/image','95% CI high','n','Basis'],
        [['regnet_x_1_6gf / native_full_tensorrt','0.134','0.147','3','TRT normalized']],
        caption='Stored FS energy; no changed data.')
    figure.canvas.draw(); renderer=figure.canvas.get_renderer()
    table=next(iter(figure.axes[0].tables))
    for cell in table.get_celld().values():
        assert cell.get_text().get_window_extent(renderer).width < cell.get_window_extent(renderer).width
