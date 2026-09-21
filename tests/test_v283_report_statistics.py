"""Regression of report transitions and finite-sample energy uncertainty."""
import copy
import json
import math

import pytest

from onnx_splitpoint_tool.energy.collector import _repeat_statistics, _student_t_critical
from scripts import native_producer_final_report as report
from scripts.native_producer_validate_visualize import (
    _copy_completed_v2_projection, _completed_v2_self_reference_detection,
    _technical_quality_error,
)
from scripts.validate_output_dumps import load_dump
from test_v282_hailo8_sentinel_evidence import actual_fifo_run


@pytest.mark.parametrize('confidence,df,expected', [
    (.95, 1, 12.706204736174694), (.95, 2, 4.302652729749462),
    (.95, 3, 3.182446305284263), (.95, 10, 2.2281388519649385),
    (.95, 100, 1.9839715184496334), (.90, 1, 6.313751514800932),
    (.90, 2, 2.919985580355516), (.99, 5, 4.032142983555228),
    (.80, 30, 1.310415025391396),
])
def test_two_sided_student_t(confidence, df, expected):
    assert _student_t_critical(confidence, df) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize('confidence', [0, 1, -1, float('nan'), float('inf')])
def test_invalid_confidence_is_not_silently_clamped(confidence):
    with pytest.raises(ValueError):
        _repeat_statistics([1., 2.], confidence)


def test_repeat_statistics_retains_valid_counts_and_small_sample_limits():
    assert _repeat_statistics([], .95)['ci_low'] is None
    assert _repeat_statistics([3.], .95)['status'] == 'single_valid_repeat'
    result = _repeat_statistics([9., 10., 11., float('inf'), float('nan')], .95)
    assert result['n'] == 3 and result['mean'] == 10. and result['sample_stddev'] == 1.
    assert result['ci_half_width'] == pytest.approx(4.302652729749462 / math.sqrt(3), rel=1e-8)
    constant = _repeat_statistics([4., 4.], .95)
    assert constant['ci_low'] == constant['ci_high'] == 4.


def _project(tmp_path, payload):
    raw = tmp_path / 'raw.json'
    raw.write_text(json.dumps(payload))
    tables = tmp_path / 'analysis_tables'
    tables.mkdir(exist_ok=True)
    (tables / 'native_fifo_eval_runner_test.json').write_text(json.dumps({
        'rows': [{'model': 'yolov7_paper', 'case': 'b009', 'report': str(raw), 'ok': True}],
    }))
    row, = report._rows_from_native_fifo_runner(tmp_path, include_direct_fallback=False)
    normalized = dict(row)
    _copy_completed_v2_projection(normalized, row)
    return normalized


@pytest.mark.parametrize('mutation', ['', 'runtime', 'index', 'missing_id', 'mixed', 'image', 'artifact'])
def test_actual_fifo_report_projection_to_real_validator(actual_fifo_run, tmp_path, mutation):
    payload, _, _ = actual_fifo_run
    payload = copy.deepcopy(payload)
    if mutation == 'runtime':
        payload['semantic_evidence_runtime_instance_id'] = 'wrong'
    elif mutation == 'index':
        payload['semantic_evidence_repetition_index'] = 1
    elif mutation == 'missing_id':
        payload.pop('semantic_evidence_repetition_id')
    elif mutation == 'mixed':
        payload['semantic_evidence_repetition_id'] = payload['repetition_records'][0]['semantic_evidence_repetition_id']
    row = _project(tmp_path, payload)
    tensors, _ = load_dump(payload['native_fifo_output_manifest'])
    meta = json.loads(open(payload['native_fifo_output_manifest']).read())
    if mutation == 'image':
        meta['input_image_sha256'] = 'a' * 64
    elif mutation == 'artifact':
        meta['completion_attestation_sha256'] = 'a' * 64
    result = _completed_v2_self_reference_detection(tensors, tensors, row, dump_metadata=meta)
    assert result['available'] is (not mutation), result


def test_selection_mismatch_counts_as_technical_error():
    assert _technical_quality_error({'self_reference_reason': 'NativeThreeStageError:fast_oracle_dump_repetition_selection_mismatch'})


def test_projection_never_completes_a_partial_authoritative_tuple():
    keys = ('semantic_evidence_repetition_index', 'semantic_evidence_repetition_id', 'semantic_evidence_runtime_instance_id')
    complete = dict(zip(keys, (3, 'old', 'old-runtime')))
    authoritative = {keys[0]: 3, keys[2]: 'new-runtime'}
    projected = report._claim_contract_fields(complete, authoritative)
    assert projected.get(keys[1]) is None
    assert projected.get(keys[2]) == 'new-runtime'
    assert projected['completed_v2_projection_conflicts']
