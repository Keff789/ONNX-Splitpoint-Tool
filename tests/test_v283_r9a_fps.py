import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_rate_endpoints import rate_endpoint_fields, format_rate_endpoints
from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
from scripts import native_producer_final_report as report


@pytest.fixture(params=[('yolov7_paper', 53.41058834258677, 104.31164),
                       ('yolo11l', 43.520034581536116, 90.35041)])
def original(request):
    model, task, p2 = request.param
    raw = json.loads((Path(__file__).parent/'fixtures/v283_r9a'/f'{model}_rates.json').read_text())
    return model, raw, task, p2


def test_original_raw_reader_matrix_gui_export_endpoint_replay(original, tmp_path):
    model, raw, task, p2 = original
    path = tmp_path/'raw.json'; path.write_text(json.dumps(raw))
    tables = tmp_path/'analysis_tables'; tables.mkdir()
    (tables/'native_fifo_eval_runner_r9a.json').write_text(json.dumps({'rows': [
        {'model': model, 'case': 'b001', 'report': str(path), 'ok': True}]}))
    row, = report._rows_from_native_fifo_runner(tmp_path, include_direct_fallback=False)
    assert row['fps_median'] == pytest.approx(task)
    assert row['performance_endpoint'] == 'completed_task'
    reports = tmp_path/'reports'; reports.mkdir()
    (reports/'native_producer_combined_summary.json').write_text(json.dumps({'rows': [row]}))
    normalized, = collect_native_performance_matrix(tmp_path)['observations']
    assert normalized['native_measured_throughput_fps'] == pytest.approx(task)
    assert normalized['p2_output_fps'] == pytest.approx(p2)
    for prefix, endpoint in [('completed_task', 'completed_task'), ('p2_output', 'raw_model_outputs')]:
        source = raw['endpoint_results'][endpoint]
        assert normalized[prefix+'_fps_ci95_low'] == source['fps_ci95_low']
        assert normalized[prefix+'_fps_ci95_high'] == source['fps_ci95_high']
        assert normalized[prefix+'_work_unit_counts'] == [1000]*3
        assert len(normalized[prefix+'_measurement_times_s']) == 3
    from onnx_splitpoint_tool.workflow.runner import _native_concise_summary_v60w
    from onnx_splitpoint_tool.workflow.scientific_reporting import _write_reports
    _, concise = _native_concise_summary_v60w(reports)
    assert concise[0]['fps'] == pytest.approx(task)
    assert concise[0]['p2_output_fps'] == pytest.approx(p2)
    _write_reports(reports/'scientific', {'created_at':'2026-09-16', 'profile_id':'r9a', 'rows':[], 'summary':{},
        'native_performance_matrix': collect_native_performance_matrix(tmp_path)})
    exported = json.loads((reports/'scientific/native_performance_observations.json').read_text())
    assert exported[0]['completed_task_fps'] == pytest.approx(task)
    assert 'P2 output FPS' in (reports/'scientific/native_performance_matrix.md').read_text()
    assert normalized['latency_ms'] is None
    text = format_rate_endpoints(normalized)
    assert f'{task:.3f} FPS' in text and f'{p2:.3f} FPS' in text
    assert 'Einbildlatenz: unavailable' in text


@pytest.mark.parametrize('mutation,reason', [
    ('missing', 'endpoint_evidence_missing'),
    ('count', 'completed_count_or_measurement_time_missing'),
    ('time', 'completed_count_or_measurement_time_missing'),
    ('mixed', 'aggregate_repetition_endpoint_contract_mismatch'),
    ('series', 'aggregate_repetition_series_mismatch'),
    ('interval', 'interval_repetition_series_mismatch'),
    ('identity', 'repetition_identity_missing_or_duplicate'),
])
def test_missing_or_mixed_completion_never_falls_back_to_p2(original, mutation, reason):
    _, source, _, p2 = original
    raw = copy.deepcopy(source); completed = raw['endpoint_results']['completed_task']
    if mutation == 'missing': del raw['endpoint_results']['completed_task']
    elif mutation == 'count':
        completed['repetition_records'][0].pop('completed_work_units', None)
        completed['repetition_records'][0].pop('completed_frames', None)
    elif mutation == 'time': completed['repetition_records'][0].pop('makespan_ms')
    elif mutation == 'mixed': completed['repetition_records'][0] = raw['endpoint_results']['raw_model_outputs']['repetition_records'][0]
    elif mutation == 'series': completed['fps_median'] = p2
    elif mutation == 'interval': completed['fps_ci95_high'] = p2
    elif mutation == 'identity': completed['repetition_records'][1]['repetition_id'] = completed['repetition_records'][0]['repetition_id']
    fields = rate_endpoint_fields(raw)
    assert fields['completed_task_fps'] is None
    assert fields['fps_ci95_low'] is None
    assert fields['completed_task_fps_unavailable_reason'] == reason
    assert fields['p2_output_fps'] == p2


def test_estimated_cycle_and_unbound_legacy_fps_are_not_completed_tasks():
    result = rate_endpoint_fields({'fps_makespan': 100, 'paper_equivalent_fps': 200})
    assert result['completed_task_fps'] is None
    assert result['p2_output_fps'] is None


def test_classification_completion_does_not_require_detection_postprocess():
    raw = {'task': 'classification', 'measurement_boundary': 'workers_ready_to_last_completed_trt_frame',
           'completed_frames': 100, 'makespan_ms': 2000, 'fps_makespan': 50,
           'postprocess_completion_verified': False}
    assert rate_endpoint_fields(raw)['completed_task_fps'] == 50


@pytest.mark.parametrize('field,value', [
    ('completed_task_endpoint_contract_hash','f'*64), ('task','classification'),
    ('completed_work_units_status','unavailable'), ('postprocess_completed_frames',1),
    ('completed_work_units',999),
])
def test_conflicting_parent_contract_or_completion_counts(original, field, value):
    _,raw,_,p2=original
    for row in raw['endpoint_results']['completed_task']['repetition_records']:
        row[field]=value
    projected=rate_endpoint_fields(raw)
    assert projected['completed_task_fps'] is None
    assert projected['completed_task_fps_unavailable_reason']
    assert projected['p2_output_fps'] == p2


def test_full_summary_uses_own_three_repetitions_when_report_is_last_single(tmp_path):
    from onnx_splitpoint_tool.native_rate_endpoints import report_rate_fields
    records=[{'task':'classification','measurement_boundary':'first_task_start_to_last_task_completion',
              'measurement_endpoint':'completed_task','completed_frames':100,'measured_duration_s':100/fps,
              'fps_makespan':fps,'repetition_id':str(index)} for index,fps in enumerate([48.,50.,52.])]
    raw=tmp_path/'last.json';raw.write_text(json.dumps(dict(records[-1],case='full',model='model')))
    summary={'model':'model','case':'full','report':str(raw),'task':'classification',
             'repetition_count_valid':3,'repetition_records':records,'fps_median':50.,'fps_ci95_low':48.,'fps_ci95_high':52.}
    fields=report_rate_fields(summary)
    assert fields['completed_task_fps'] == 50
    assert fields['completed_task_work_unit_counts'] == [100]*3
    assert fields['rate_endpoint_source']=='summary_repetition_evidence'
    summary['model']='other'
    fields=report_rate_fields(summary)
    assert fields['completed_task_fps'] is None
    assert fields['completed_task_rate']['reason']==fields['completed_task_fps_unavailable_reason']=='raw_report_summary_identity_conflict'


@pytest.mark.parametrize('semantics',['reciprocal_steady_state_throughput','not_measured_async_or_streaming_throughput'])
def test_legacy_reciprocal_is_never_displayed_as_image_latency(semantics):
    assert 'Einbildlatenz: unavailable' in format_rate_endpoints({'latency_median_ms':10.,'latency_semantics':semantics})
