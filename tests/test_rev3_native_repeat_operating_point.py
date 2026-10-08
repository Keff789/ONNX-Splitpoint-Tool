"""Separate process output locations are not Native operating-point changes."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts import native_producer_final_report as final
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256, seal_native_command_contract
from test_v269f_native_split_final_energy_integrity import _quality_first_row


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repeat(base, root, index, mutate=None):
    row = copy.deepcopy(base)
    command = row['native_command_contract']
    folder = root / f'rep{index:02}'
    for role in ('output', 'boundary'):
        item = command['artifacts'][f'semantic_{role}_manifest']
        old = Path(item['path'])
        manifest = json.loads(old.read_text())
        dst = folder / 'semantic' / old.name
        for payload in manifest['payload_artifacts']:
            src = Path(payload['path'])
            target = dst.parent / src.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(src.read_bytes())
            payload['path'] = str(target)
        manifest['payload_artifacts_sha256'] = canonical_json_sha256(manifest['payload_artifacts'])
        _write(dst, manifest)
        item.update(path=str(dst), sha256=_sha(dst), size_bytes=dst.stat().st_size)
        row[f'native_fifo_{role}_manifest'] = str(dst)
    if mutate:
        mutate(command)
    command.pop('contract_sha256', None)
    command = seal_native_command_contract(command)
    row['native_command_contract'] = command
    row['native_command_contract_sha256'] = command['contract_sha256']
    row['workload_contract_sha256'] = command['contract_sha256']
    att = row['native_split_quality_consumer_attestation']
    att.pop('attestation_sha256', None)
    att['command_contract_sha256'] = command['contract_sha256']
    for role in ('output', 'boundary'):
        att[f'semantic_{role}_manifest_sha256'] = command['artifacts'][f'semantic_{role}_manifest']['sha256']
    att['attestation_sha256'] = canonical_json_sha256(att)
    row['report'] = str(folder / 'result.json')
    _write(Path(row['report']), row)
    return row


def test_verified_repetition_locations_preserve_original_seals(tmp_path):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    rows = [_repeat(base, tmp_path, i) for i in (1, 2, 3)]
    original = [r['native_command_contract_sha256'] for r in rows]
    assert len(set(original)) == 3
    proofs = [final._verified_repetition_operating_point(r) for r in rows]
    assert all(proofs)
    assert len({r['operating_point_sha256'] for r in proofs}) == 1
    assert [r['native_command_contract_sha256'] for r in rows] == original


@pytest.mark.parametrize('mutate', [
    lambda c: c.update(runner_sha256='f' * 64),
    lambda c: c.update(input_image_sha256='e' * 64),
    lambda c: c['runtime_options'].update(queue_depth=7),
    lambda c: c['artifacts'].update(unrecognized_source={'path':'/fixed/other','sha256':'e'*64}),
])
def test_source_input_scheduling_and_unlisted_fields_never_coalesce(tmp_path, mutate):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    a = _repeat(base, tmp_path, 1)
    b = _repeat(base, tmp_path, 2, mutate)
    first = final._verified_repetition_operating_point(a)
    second = final._verified_repetition_operating_point(b)
    assert first
    assert not second or first['operating_point_sha256'] != second['operating_point_sha256']


def test_payload_tampering_fails_closed(tmp_path):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    row = _repeat(base, tmp_path, 1)
    assert final._verified_repetition_operating_point(row)
    manifest = json.loads(Path(row['native_fifo_output_manifest']).read_text())
    Path(manifest['payload_artifacts'][0]['path']).write_bytes(b'changed')
    assert final._verified_repetition_operating_point(row) == {}


def test_duplicate_command_hash_assertion_cannot_hide_drift(tmp_path):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    row = _repeat(base, tmp_path, 1)
    row['native_command_contract_sha256'] = 'f' * 64
    assert final._verified_repetition_operating_point(row) == {}


def test_normal_rate_projection_consumes_three_actual_count_time_records(tmp_path):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    records = []
    for index, fps in enumerate((20.0, 25.0, 30.0), 1):
        row = _repeat(base, tmp_path, index)
        row.update(ok=True, status='ok', task='detection',
                   measurement_boundary='workers_ready_to_last_completed_task_frame',
                   completed_work_units=1000, completed_frames=1000,
                   makespan_ms=1000000 / fps, fps_makespan=fps,
                   postprocess_completed_frames=1000, postprocess_completion_verified=True,
                   completed_task_endpoint_attested=True,
                   completed_task_endpoint_contract_hash='d' * 64,
                   repetition_id=f'repeat-{index}', runtime_instance_id=f'process-{index}')
        _write(Path(row['report']), row)
        records.append(row)
    aggregate = copy.deepcopy(records[0])
    aggregate.update(repetition_records=records, repetition_count_requested=3,
                     repetition_count_attempted=3, repetition_count_valid=3,
                     repetition_status='complete', repetition_aggregation='median',
                     repetition_runtime_scope='fresh_process_per_repetition',
                     repetition_independence_verified=True, fps_repetition_samples=[20., 25., 30.],
                     fps_makespan=25., fps_median=25., fps_ci95_low=20., fps_ci95_high=30.)
    fields = final._final_rate_fields(aggregate)
    assert fields['completed_task_fps'] == 25.
    assert fields['completed_task_rate']['work_unit_counts'] == [1000, 1000, 1000]
    assert fields['rate_endpoint_source'] == 'verified_separate_process_repetition_evidence'
    records[1]['makespan_ms'] = 123.
    assert final._final_rate_fields(aggregate)['completed_task_fps'] is None


def test_quality_mismatch_preserves_fresh_completion_evidence(tmp_path):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    base.update(task='detection', stage='raw_head', contract_family='raw_head',
                endpoint_contract_complete=True, endpoint_contract_hash='a' * 64,
                completed_task_endpoint_contract_hash='b' * 64,
                completed_task_completion_mode='fresh_mode', postprocess_completed_frames=1000)
    quality = copy.deepcopy(base)
    quality.update(completed_task_endpoint_contract_hash='c' * 64,
                   completed_task_completion_mode='historical_mode', postprocess_completed_frames=100)
    result, _ = final._attach_quality_evidence([base], {
        'schema':'onnx-splitpoint/native-producer-validation-summary',
        'schema_version':7, 'status':'complete', 'row_count':1, 'rows':[quality],
    }, quality_summary_status='loaded')
    row = result[0]
    assert row['quality_match_count'] == 1
    assert row['completed_task_endpoint_contract_hash'] == 'b' * 64
    assert row['completed_task_completion_mode'] == 'fresh_mode'
    assert row['postprocess_completed_frames'] == 1000
    assert row['quality_physical_evidence_conflict'] is True
    assert row['quality_evidence_verified'] is False
