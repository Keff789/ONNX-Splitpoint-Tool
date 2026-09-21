"""Stored B2 completion must survive the real Scientific/CSV consumer chain."""
from copy import deepcopy
import csv
import json

import pytest

from test_v283_r9j_continuation import b2_raw_full, read
from onnx_splitpoint_tool.validation.accuracy_gates import apply_accuracy_gate_to_row
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_files
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _performance_cohort_projection,
    _scientific_row,
    _write_reports,
)


@pytest.fixture
def b2_completed_rows():
    rows = []
    for model in ('yolo11l', 'yolo26s'):
        for backend in ('hailo8', 'hailo10'):
            path, _ = b2_raw_full(model, backend)
            normalized, _ = normalize_benchmark_files(model_id=model, source_paths=[path])
            row, = normalized
            original = next(r for r in read(path.parent / 'normalized_results.json')['results']
                            if r['backend'] == backend)
            row['task_quality_gate'] = deepcopy(original['task_quality_gate'])
            apply_accuracy_gate_to_row(row, row['task_quality_policy'])
            assert row['technical_status'] == 'ok'
            assert row['task_quality_status'] == 'accuracy_loss'
            rows.append(row)
    return rows


def test_b2_completed_endpoint_survives_scientific_observations_and_csv(tmp_path, b2_completed_rows):
    scientific = [_scientific_row(row) for row in b2_completed_rows]
    report = tmp_path / 'scientific'
    _write_reports(report, {'run_id': 'offline_b2', 'rows': scientific})
    exported = json.loads((report / 'performance_observations.json').read_text())
    with (report / 'performance_observations.csv').open(newline='') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(scientific) == len(exported) == len(csv_rows) == 4
    for original, projected, stored, csv_row in zip(b2_completed_rows, scientific, exported, csv_rows):
        assert projected.get('measurement_endpoint') == 'completed_detection'
        assert stored.get('measurement_endpoint') == csv_row.get('measurement_endpoint') == 'completed_detection'
        assert projected.get('postprocess_completion_verified') is True
        assert stored.get('postprocess_completed_frames') == 5
        assert csv_row.get('postprocess_completed_frames') == '5'
        for field in ('postprocess_included', 'decoder_id', 'host_postprocessing_evidence_status',
                      'host_postprocessing_evidence_source', 'raw_stage_mean_ms',
                      'host_tail_mean_ms', 'completed_task_mean_ms'):
            assert projected.get(field) == stored.get(field) == original[field]
        assert stored['accuracy_class'] == 'accuracy_loss'
        assert stored['runtime_executable'] is True
        assert stored['performance_claim_eligible'] == original.get('performance_claim_eligible')


@pytest.mark.parametrize('damage', ['missing', 'negative', 'raw', 'terminal'])
def test_scientific_projection_does_not_invent_completion_or_clear_negatives(b2_completed_rows, damage):
    row = deepcopy(b2_completed_rows[0])
    fields = ('measurement_endpoint', 'postprocess_included', 'postprocess_completion_verified',
              'postprocess_completed_frames', 'completed_task_mean_ms')
    if damage == 'missing':
        for field in fields:
            row.pop(field, None)
    elif damage == 'negative':
        row.update(postprocess_included=False, postprocess_completion_verified=False,
                   postprocess_completed_frames=0)
    elif damage == 'raw':
        row['measurement_endpoint'] = 'p2_output'
    else:
        row.update(runtime_ok=False, runner_returncode=-11, terminal_failure=True,
                   measurement_valid=False)
        apply_accuracy_gate_to_row(row, row['task_quality_policy'])
    scientific = _scientific_row(row)
    for field in fields:
        assert scientific.get(field) == row.get(field)
    cohorts, _ = _performance_cohort_projection([scientific])
    assert scientific['performance_claim_eligible'] == row.get('performance_claim_eligible')
    if damage == 'terminal':
        assert cohorts[0]['claim_cohort_eligible'] is False
        assert cohorts[0]['technical_cohort_eligible'] is False
        assert scientific['runner_returncode'] == -11


@pytest.mark.parametrize('failure', ['timeout', 'negative_exit'])
def test_completed_report_cannot_hide_failed_execution_on_import(tmp_path, failure):
    _, raw = b2_raw_full('yolo11l', 'hailo8')
    if failure == 'timeout':
        raw.update(runtime_ok=False, error_class='timeout')
    else:
        raw['runner_returncode'] = -11
    source = tmp_path / 'benchmark_results_hailo8_auto.json'
    source.write_text(json.dumps([raw]))
    rows, _ = normalize_benchmark_files(model_id='yolo11l', source_paths=[source])
    row, = rows
    assert row['runtime_executable'] is False
    scientific = _scientific_row(row)
    report = tmp_path / 'scientific'
    _write_reports(report, {'run_id': 'offline_negative', 'rows': [scientific]})
    assert json.loads((report / 'performance_observations.json').read_text()) == []
    diagnostic, = json.loads((report / 'performance_cohorts.json').read_text())
    assert diagnostic['measurement_endpoint'] == 'completed_detection'
    assert diagnostic['technical_cohort_eligible'] is False
    assert diagnostic['claim_cohort_eligible'] is False
