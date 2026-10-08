"""Separate process output locations are not Native operating-point changes."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from scripts import native_producer_final_report as final
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256, seal_native_command_contract
from test_v269f_native_split_final_energy_integrity import _quality_first_row
from test_yolo_stored_reference_provenance import sample as _bn6_sample
from test_v272_hailo8_completed_detection_hotloop import _sealed_hailo8_command_contract


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


def _bn6_repeat(base, root, index, *, original_wh=(640, 480), source_sample=None):
    row = _repeat(base, root, index)
    folder = root / f'rep{index:02}'
    source_path, tensors, execution = source_sample or _bn6_sample(folder / 'benchmark_set', original_wh=original_wh)
    manifest_path = Path(row['native_fifo_output_manifest'])
    manifest = json.loads(manifest_path.read_text())
    manifest.update(json.loads(source_path.read_text()))
    payload = Path(manifest['payload_artifacts'][0]['path'])
    tensor = tensors['output0']
    payload.write_bytes(tensor.tobytes())
    manifest['payload_artifacts'][0].update(sha256=_sha(payload), size_bytes=payload.stat().st_size)
    manifest['payload_artifacts_sha256'] = canonical_json_sha256(manifest['payload_artifacts'])
    manifest['outputs'] = [{'name': 'output0', 'file': payload.name, 'shape': list(tensor.shape),
                            'dtype': 'float32', 'sha256': _sha(payload)}]
    manifest['completion_sentinel_identity'] = {'execution_contract_sha256': execution['contract_sha256'],
                                               'repetition_id': str(index), 'runtime_instance_id': f'process-{index}'}
    _write(manifest_path, manifest)
    command = row['native_command_contract']
    mixed = _sealed_hailo8_command_contract('hailo8_python_vstreams_fifo')
    for field in ('task', 'producer_impl', 'mixed_runtime_site_policy', 'process_local_extra_sites'):
        command['runtime_options'][field] = mixed['runtime_options'][field]
    command['interpreter_identity'].update(runtime_mode=mixed['interpreter_identity']['runtime_mode'],
        process_local_extra_sites=mixed['interpreter_identity']['process_local_extra_sites'],
        resolved_executable=command['python_executable'])
    command['mixed_runtime_contract'] = mixed['mixed_runtime_contract']
    command['mixed_runtime_contract'].update(python_executable=command['python_executable'],
                                           resolved_python_executable=command['python_executable'])
    command['artifacts']['native_trt_consumer_source'] = mixed['artifacts']['native_trt_consumer_source']
    command['artifacts']['native_executable'] = copy.deepcopy(command['artifacts']['python_executable'])
    command['artifacts']['semantic_output_manifest'].update(sha256=_sha(manifest_path),
                                                           size_bytes=manifest_path.stat().st_size)
    command['runtime_options'].update(completion_execution_contract=execution,
                                      completion_execution_contract_sha256=execution['contract_sha256'])
    command.pop('contract_sha256')
    command = seal_native_command_contract(command)
    attestation = row['native_split_quality_consumer_attestation']
    attestation.pop('attestation_sha256')
    attestation.update(command_contract_sha256=command['contract_sha256'],
                       semantic_output_manifest_sha256=_sha(manifest_path))
    attestation['attestation_sha256'] = canonical_json_sha256(attestation)
    row.update(native_command_contract=command, native_command_contract_sha256=command['contract_sha256'],
               workload_contract_sha256=command['contract_sha256'],
               completed_task_endpoint_contract_hash=execution['completed_endpoint_contract']['endpoint_contract_hash'],
               completed_task_output_endpoint_id=execution['completed_endpoint_contract']['output_endpoint_id'],
               frozen_host_postprocess_contract_sha256=execution['processor_contract']['contract_sha256'])
    _write(Path(row['report']), row)
    return row


def test_bn6_declaration_locations_rebuild_only_the_report_comparison(tmp_path):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    rows = [_bn6_repeat(base, tmp_path, index) for index in (1, 2, 3)]
    protected = {Path(row['report']): Path(row['report']).read_bytes() for row in rows}
    proofs = [final._verified_repetition_operating_point(row) for row in rows]
    assert all(proofs)
    assert all(proof['completion_contract_provenance_normalized'] for proof in proofs)
    assert len({proof['operating_point_sha256'] for proof in proofs}) == 1
    assert len({row['native_command_contract_sha256'] for row in rows}) == 3
    grouped = final._aggregate_repetitions(rows)
    assert len(grouped) == 1 and grouped[0]['repetition_operating_point_verified'] is True
    assert all(path.read_bytes() == data for path, data in protected.items())


@pytest.mark.parametrize('damage', ['declaration_bytes', 'processor', 'execution', 'attestation',
                                    'sentinel', 'geometry', 'row_endpoint'])
def test_bn6_original_declaration_primitives_and_dependent_seals_fail_closed(tmp_path, damage):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    first = _bn6_repeat(base, tmp_path, 1)
    second = _bn6_repeat(base, tmp_path, 2, original_wh=(640, 400) if damage == 'geometry' else (640, 480))
    expected = final._verified_repetition_operating_point(first)
    assert expected
    if damage == 'declaration_bytes':
        declaration = tmp_path / 'rep02/benchmark_set/output_contracts.json'
        declaration.write_text(declaration.read_text() + ' ')
    elif damage == 'row_endpoint':
        second['completed_task_endpoint_contract_hash'] = 'f' * 64
    elif damage in {'processor', 'execution'}:
        execution = second['native_command_contract']['runtime_options']['completion_execution_contract']
        (execution['processor_contract'] if damage == 'processor' else execution)['contract_sha256'] = 'f' * 64
        command = second['native_command_contract']; command.pop('contract_sha256')
        command = seal_native_command_contract(command)
        second.update(native_command_contract=command, native_command_contract_sha256=command['contract_sha256'],
                      workload_contract_sha256=command['contract_sha256'])
        attestation = second['native_split_quality_consumer_attestation']; attestation.pop('attestation_sha256')
        attestation['command_contract_sha256'] = command['contract_sha256']
        attestation['attestation_sha256'] = canonical_json_sha256(attestation)
        _write(Path(second['report']), second)
    elif damage in {'attestation', 'sentinel'}:
        manifest = Path(second['native_fifo_output_manifest'])
        payload = json.loads(manifest.read_text())
        if damage == 'attestation': payload['output_endpoint_attestation']['attested'] = False
        else: payload['completion_sentinel_identity']['execution_contract_sha256'] = 'f' * 64
        _write(manifest, payload)
        command = second['native_command_contract']; command.pop('contract_sha256')
        command['artifacts']['semantic_output_manifest'].update(sha256=_sha(manifest), size_bytes=manifest.stat().st_size)
        command = seal_native_command_contract(command)
        second.update(native_command_contract=command, native_command_contract_sha256=command['contract_sha256'],
                      workload_contract_sha256=command['contract_sha256'])
        attestation = second['native_split_quality_consumer_attestation']; attestation.pop('attestation_sha256')
        attestation.update(command_contract_sha256=command['contract_sha256'], semantic_output_manifest_sha256=_sha(manifest))
        attestation['attestation_sha256'] = canonical_json_sha256(attestation)
        _write(Path(second['report']), second)
    actual = final._verified_repetition_operating_point(second)
    assert not actual or actual['operating_point_sha256'] != expected['operating_point_sha256']


def _bn6_rate_aggregate(tmp_path, *, shared_declaration=False):
    base, _, _ = _quality_first_row(tmp_path / 'base')
    records = []
    shared_source = _bn6_sample(tmp_path / 'shared/benchmark_set') if shared_declaration else None
    for index, fps in enumerate((20., 25., 30.), 1):
        row = _bn6_repeat(base, tmp_path, index, source_sample=shared_source)
        row.update(ok=True, status='ok', runtime_success=True, task='detection',
                   measurement_boundary='workers_ready_to_last_completed_task_frame',
                   completed_work_units=1000, completed_frames=1000,
                   makespan_ms=1000000 / fps, fps_makespan=fps,
                   postprocess_completed_frames=1000, postprocess_completion_verified=True,
                   completed_task_endpoint_attested=True,
                   completion_execution_contract_sha256=row['native_command_contract']['runtime_options']['completion_execution_contract_sha256'],
                   repetition_id=f'repeat-{index}', runtime_instance_id=f'process-{index}',
                   repetition_count_requested=1, repetition_count_attempted=1)
        _write(Path(row['report']), row)
        records.append(row)
    aggregate = copy.deepcopy(records[0])
    aggregate.update(repetition_records=records, repetition_count_requested=3,
                     repetition_count_attempted=3, repetition_count_valid=3,
                     repetition_status='complete', repetition_aggregation='median',
                     repetition_runtime_scope='fresh_process_per_repetition',
                     repetition_independence_verified=True, fps_repetition_samples=[20., 25., 30.],
                     fps_makespan=25., fps_median=25., fps_ci95_low=20., fps_ci95_high=30.)
    return aggregate


def test_bn6_rate_projection_preserves_three_original_contracts_and_count_time_series(tmp_path):
    aggregate = _bn6_rate_aggregate(tmp_path)
    records_before = copy.deepcopy(aggregate['repetition_records'])
    files_before = {Path(r['report']): Path(r['report']).read_bytes() for r in records_before}
    for key in ('completed_task_endpoint_contract_hash', 'completion_execution_contract_sha256'):
        assert len({r[key] for r in records_before}) == 3
    fields = final._final_rate_fields(aggregate)
    assert fields['completed_task_fps_median'] == 25.
    assert fields['completed_task_rate']['samples'] == [20., 25., 30.]
    assert fields['completed_task_rate']['work_unit_counts'] == [1000, 1000, 1000]
    assert fields['completed_task_rate']['measurement_times_s'] == [50., 40., 1000 / 30.]
    assert fields['completed_task_rate']['repetition_ids'] == ['repeat-1', 'repeat-2', 'repeat-3']
    assert fields['fps_ci95_method'] == 'deterministic_percentile_bootstrap_median_4000'
    aggregate.update(fields)
    assert final._final_rate_fields(aggregate)['completed_task_rate'] == fields['completed_task_rate']
    assert aggregate['repetition_records'] == records_before
    assert all(path.read_bytes() == content for path, content in files_before.items())


@pytest.mark.parametrize('field,value', [
    ('fps_makespan', 24.), ('makespan_ms', 1.), ('completed_frames', 999),
    ('completed_task_endpoint_contract_hash', 'f' * 64),
    ('completion_execution_contract_sha256', 'f' * 64),
    ('measurement_boundary', 'workers_ready_to_last_completed_trt_frame'),
    ('postprocess_completed_frames', 999), ('postprocess_completion_verified', False),
    ('completed_task_endpoint_attested', False), ('repetition_id', 'repeat-1'),
    ('task', 'classification'),
])
def test_bn6_rate_projection_rejects_changed_original_single_rate(tmp_path, field, value):
    aggregate = _bn6_rate_aggregate(tmp_path)
    aggregate['repetition_records'][1][field] = value
    assert final._final_rate_fields(aggregate)['completed_task_fps'] is None


def test_bn6_shared_declaration_retains_original_verbatim_operating_point_path(tmp_path):
    aggregate = _bn6_rate_aggregate(tmp_path, shared_declaration=True)
    original = copy.deepcopy(aggregate['repetition_records'])
    proofs = [final._verified_repetition_operating_point(r, require_observation=True) for r in original]
    assert all(proofs) and len({p['operating_point_sha256'] for p in proofs}) == 1
    assert all(p['completion_contract_provenance_normalized'] is False for p in proofs)
    assert len({r['completion_execution_contract_sha256'] for r in original}) == 1
    fields = final._final_rate_fields(aggregate)
    assert fields['completed_task_fps_median'] == 25.
    assert fields['fps_repetition_samples'] == [20., 25., 30.]
    assert aggregate['repetition_records'] == original
    aggregate.update(fields)
    assert final._final_rate_fields(aggregate)['completed_task_rate'] == fields['completed_task_rate']
