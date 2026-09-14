"""Synthetic complete import contracts; original-data replay is reported separately."""
from __future__ import annotations
import copy
import hashlib
import json
import shlex
from pathlib import Path
import pytest
from scripts import run_native_producer_energy_from_summary as energy
from scripts import replay_selected_energy_attempts_v282 as replay
from test_v267_energy_aggregate_import import _aggregate, _command, _write_aggregate
from test_v27931_hailo_manifest_energy import _fresh


def _case(tmp_path: Path, *, repeat=0, attempt=1):
    row, fresh, base, directory = _fresh(tmp_path)
    root = directory.parent
    runs = []
    for index in range(3):
        selected_attempt = attempt if index == repeat else 0
        run = copy.deepcopy(base)
        history = []
        for number in range(selected_attempt + 1):
            target = (root / f'run_{index:03d}' if number == 0 else root /
                      'repeat_retry_attempts' / f'repeat_{index:03d}' / f'attempt_{number:02d}' / 'run_000')
            target.mkdir(parents=True, exist_ok=True)
            raw = ('__SPLITPOINT_ENERGY_COMPLETION__=' + json.dumps(fresh) + '\n').encode()
            if number != selected_attempt:
                raw = b'original failed acquisition\n'
            (target / 'workload_stdout.log').write_bytes(raw)
            history.append({'attempt_index': number, 'selected': number == selected_attempt,
                            'run_directory': str(target), 'retry_reasons': [] if number == selected_attempt else ['marker_missing'],
                            'accepted_for_logical_repeat': number == selected_attempt,
                            'final_energy_gate_status': 'pass' if number == selected_attempt else 'fail'})
        run.update(run_index=index, logical_repeat_index=index, selected_repeat_attempt_index=selected_attempt,
                   repeat_attempt_count=len(history), repeat_retry_attempted=selected_attempt > 0,
                   repeat_retry_recovered=selected_attempt > 0, repeat_attempt_history=history,
                   storage_dir=str(target / 'collector_storage'))
        run['collector_and_workload_log_diagnostics'] = {'workload_stdout': {
            'available': True, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
        runs.append(run)
    aggregate = _aggregate(root, setup_id=row['setup_id'])
    aggregate.update(runs=runs, preflight_requested=True,
                     preflight_expected_command_contract_sha256=row['native_command_contract_sha256'],
                     preflight_verified_run_count=3)
    path, started = _write_aggregate(root, aggregate)
    command = _command(root, setup_id=row['setup_id'])
    kwargs = dict(expected_runs=3, expected_effective_runs=3, expected_run_id='native-row-a',
                  expected_setup_id=row['setup_id'], expected_native_row=row,
                  expected_command_contract_sha256=row['native_command_contract_sha256'],
                  execution_started_ns=started, aggregate_absent_before_execution=True)
    return root, aggregate, row, command, kwargs


def _attach(case):
    root, aggregate, row, command, kwargs = case
    _write_aggregate(root, aggregate)
    return energy._attach_energy_aggregate({'rc': 0}, command, root, **kwargs)


@pytest.mark.parametrize('repeat', [0, 1, 2])
@pytest.mark.parametrize('attempt', [0, 1, 2])
def test_complete_import_uses_selected_attempt_without_changing_logical_n(tmp_path, repeat, attempt):
    case = _case(tmp_path, repeat=repeat, attempt=attempt)
    result = _attach(case)
    assert result['energy_aggregate_verified'] is True, result
    assert result['energy_aggregate']['scientific_primary_energy_statistics']['energy_j']['n'] == 3
    evidence = result['fresh_energy_completion_evidence'][repeat]
    assert evidence['logical_repeat_index'] == repeat
    assert evidence['selected_repeat_attempt_index'] == attempt
    assert evidence['selected_repeat_run_directory'].endswith('run_000' if attempt else f'run_{repeat:03d}')
    assert len(result['fresh_energy_completion_evidence']) == 3
    assert result['energy_aggregate']['runs'] == case[1]['runs']


@pytest.mark.parametrize('mutation', ['no_selected', 'two_selected', 'duplicate', 'counter', 'unordered',
    'bool', 'negative', 'string', 'missing_history', 'missing_index', 'not_recovered', 'accepted_false',
    'cancelled', 'older_selected', 'wrong_repeat', 'foreign_job', 'traversal', 'storage_conflict',
    'missing_stdout', 'changed_stdout', 'nonce', 'count', 'warmup', 'window', 'duplicate_marker', 'command', 'endpoint'])
def test_invalid_selection_and_existing_completion_contracts_remain_negative(tmp_path, mutation):
    case = _case(tmp_path, repeat=2)
    root, aggregate, row, command, kwargs = case
    run = aggregate['runs'][2]
    history = run['repeat_attempt_history']
    selected_dir = Path(history[1]['run_directory'])
    if mutation == 'no_selected': history[1]['selected'] = False
    elif mutation == 'two_selected': history[0]['selected'] = True
    elif mutation == 'duplicate': history[1]['attempt_index'] = 0
    elif mutation == 'counter': run['repeat_attempt_count'] = 3
    elif mutation == 'unordered': history.reverse()
    elif mutation == 'bool': run['selected_repeat_attempt_index'] = True
    elif mutation == 'negative': run['selected_repeat_attempt_index'] = -1
    elif mutation == 'string': run['logical_repeat_index'] = '2'
    elif mutation == 'missing_history': run.pop('repeat_attempt_history')
    elif mutation == 'missing_index': run.pop('selected_repeat_attempt_index')
    elif mutation == 'not_recovered': run['repeat_retry_recovered'] = False
    elif mutation == 'accepted_false': history[1]['accepted_for_logical_repeat'] = False
    elif mutation == 'cancelled': run['cancelled'] = True
    elif mutation == 'older_selected':
        history[0]['selected'] = True; history[0]['accepted_for_logical_repeat'] = True
        history[1]['selected'] = False; history[1]['accepted_for_logical_repeat'] = False
        run['selected_repeat_attempt_index'] = 0
    elif mutation == 'wrong_repeat': history[1]['run_directory'] = str(selected_dir).replace('repeat_002', 'repeat_001')
    elif mutation == 'foreign_job': history[1]['run_directory'] = str(tmp_path / 'other/run_000')
    elif mutation == 'traversal': history[1]['run_directory'] = str(root / '../other/run_000')
    elif mutation == 'storage_conflict': run['storage_dir'] = str(root / 'run_002/collector_storage')
    elif mutation == 'missing_stdout': (selected_dir / 'workload_stdout.log').unlink()
    elif mutation == 'changed_stdout': (selected_dir / 'workload_stdout.log').write_text('different')
    else:
        path = selected_dir / 'workload_stdout.log'
        fresh = json.loads(path.read_text().split('=', 1)[1])
        if mutation == 'nonce': fresh['energy_preflight_nonce'] = 'wrong'
        elif mutation == 'count': run['runtime_work_unit_evidence']['count'] = 2
        elif mutation == 'warmup': fresh['warmup'] = 1
        elif mutation == 'window': run['workload_timing']['end_ns'] = 1
        elif mutation == 'command': fresh['source_contract_sha256'] = 'f' * 64
        elif mutation == 'endpoint': row['completed_task_comparison_output_endpoint_id'] = 'different'
        raw = ('__SPLITPOINT_ENERGY_COMPLETION__=' + json.dumps(fresh) + '\n').encode()
        if mutation == 'duplicate_marker': raw *= 2
        path.write_bytes(raw)
        run['collector_and_workload_log_diagnostics']['workload_stdout'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    result = _attach(case)
    assert result['energy_aggregate_verified'] is False, mutation
    assert result['energy_aggregate_completion_errors']


@pytest.mark.parametrize('outside', [False, True])
def test_symlink_to_other_attempt_or_job_is_rejected(tmp_path, outside):
    case = _case(tmp_path)
    root, aggregate, *_ = case
    directory = Path(aggregate['runs'][0]['repeat_attempt_history'][1]['run_directory'])
    target = tmp_path / 'foreign' if outside else root / 'run_000'
    target.mkdir(exist_ok=True)
    (directory / 'workload_stdout.log').unlink()
    directory.rmdir()
    directory.symlink_to(target, target_is_directory=True)
    assert _attach(case)['energy_aggregate_verified'] is False


def test_legacy_without_any_retry_hint_and_partial_provenance(tmp_path):
    case = _case(tmp_path, attempt=0)
    for run in case[1]['runs']:
        for key in list(run):
            if key.startswith('repeat_') or key in ('selected_repeat_attempt_index', 'logical_repeat_index'):
                run.pop(key)
    assert _attach(case)['energy_aggregate_verified'] is True
    case[1]['runs'][0]['repeat_retry_attempted'] = False
    assert _attach(case)['energy_aggregate_verified'] is False


def _checkpoint(case):
    root, aggregate, row, command, kwargs = case
    execution = {key: value for key, value in kwargs.items() if key != 'expected_native_row'}
    execution.update(runtime_row=row, output_dir=str(root), command=command, argv=shlex.split(command))
    return {'state': 'failed', 'execution': execution, 'result': {'row': row, 'run': {
        'rc': 0, 'cmd': shlex.split(command),
        'energy_aggregate_sha256': hashlib.sha256((root / 'energy_aggregate.json').read_bytes()).hexdigest(),
        'energy_aggregate_mtime_ns': (root / 'energy_aggregate.json').stat().st_mtime_ns,
        'energy_aggregate_verified': False,
        'energy_aggregate_import_status': 'bound_incomplete',
        'energy_aggregate_completion_errors': ['old_stdout_hash_mismatch']}}}


def test_recovery_and_reimport_use_same_full_import_and_original_process_provenance(tmp_path):
    case = _case(tmp_path, repeat=1)
    checkpoint = _checkpoint(case)
    recovered = energy._recover_managed_running_result(checkpoint)
    assert recovered['ok'] is True
    derived = energy._reimport_energy_checkpoint(checkpoint)
    assert derived['run']['fresh_energy_completion_evidence'] == recovered['run']['fresh_energy_completion_evidence']
    assert derived['original_import_decision']['energy_aggregate_verified'] is False
    assert derived['claim_eligible'] is False
    checkpoint.pop('result')
    assert energy._recover_managed_running_result(checkpoint) is None
    with pytest.raises(ValueError, match='historical_process_result_missing'):
        energy._reimport_energy_checkpoint(checkpoint)


@pytest.mark.parametrize('field,value', [('execution_started_ns', None), ('aggregate_absent_before_execution', None)])
def test_reimport_never_invents_original_binding_facts(tmp_path, field, value):
    case = _case(tmp_path)
    checkpoint = _checkpoint(case)
    checkpoint['execution'][field] = value
    assert energy._reimport_energy_checkpoint(checkpoint)['ok'] is False


def test_read_only_cli_is_idempotent_and_refuses_inplace_or_conflicting_output(tmp_path, monkeypatch):
    case = _case(tmp_path / 'original')
    checkpoint = _checkpoint(case)
    source = tmp_path / 'original/checkpoint.json'
    source.write_text(json.dumps(checkpoint))
    original = source.read_bytes()
    output = tmp_path / 'derived.json'
    # Fail on any external execution, even if a later result would hide it.
    monkeypatch.setattr(energy.subprocess, 'Popen', lambda *a, **k: pytest.fail('hardware/subprocess forbidden'))
    args = ['--run-root', str(tmp_path / 'original'), '--checkpoint', str(source), '--output', str(output)]
    assert replay.main(args) == 0
    content = output.read_bytes()
    assert replay.main(args) == 0
    assert output.read_bytes() == content and source.read_bytes() == original
    output.write_text('{}')
    with pytest.raises(FileExistsError): replay.main(args)
    with pytest.raises(SystemExit): replay.main(args[:-1] + [str(source.parent / 'derived.json')])


def test_explicit_archived_root_projection_preserves_original_bytes(tmp_path):
    case = _case(tmp_path / 'original')
    checkpoint = _checkpoint(case)
    root = case[0]
    destination = tmp_path / 'extracted'
    root.rename(destination)
    before = (destination / 'energy_aggregate.json').read_bytes()
    derived = energy._reimport_energy_checkpoint(checkpoint, measurement_path_mapping=(root, destination))
    assert derived['ok'] is True, derived
    assert (destination / 'energy_aggregate.json').read_bytes() == before
    assert derived['run']['energy_archive_path_projection']['original_bytes_modified'] is False
    assert derived['run']['fresh_energy_completion_evidence'][0]['selected_repeat_run_directory'].startswith(str(destination))


def test_remote_mirror_is_identical():
    root = Path(__file__).resolve().parents[1]
    for name in ('run_native_producer_energy_from_summary.py', 'replay_selected_energy_attempts_v282.py'):
        assert (root / 'scripts' / name).read_bytes() == (root / 'onnx_splitpoint_tool/resources/remote_scripts' / name).read_bytes()


@pytest.mark.parametrize('model', ['yolo26m', 'yolo26s'])
def test_original_sept13_retry_through_complete_checkpoint_import(model, tmp_path):
    fixture = Path(__file__).parent / 'fixtures/v282_energy_retry_original' / model
    checkpoint_path = fixture / 'checkpoint.json'
    source = checkpoint_path.read_bytes()
    checkpoint = json.loads(source)
    execution = checkpoint['execution']
    root = fixture / 'measurement'
    aggregate_before = (root / 'energy_aggregate.json').read_bytes()
    aggregate = json.loads(aggregate_before)
    # The precise previous bug is reproducible against unchanged original bytes.
    wrong, reason = energy._verify_fresh_fast_energy_completion(
        aggregate['runs'][0], execution['runtime_row'], run_dir=root / 'run_000')
    assert wrong is None and reason == 'energy_completion_captured_stdout_hash_mismatch'
    mapping = (execution['output_dir'], root.resolve())
    derived = energy._reimport_energy_checkpoint(checkpoint, measurement_path_mapping=mapping)
    assert derived['ok'] is True, derived
    assert derived['run']['energy_aggregate_sha256'] == checkpoint['result']['run']['energy_aggregate_sha256']
    assert derived['original_import_decision']['energy_aggregate_verified'] is False
    evidence = derived['run']['fresh_energy_completion_evidence']
    assert [(r['logical_repeat_index'], r['selected_repeat_attempt_index']) for r in evidence] == [(0, 1), (1, 0), (2, 0)]
    assert derived['run']['energy_aggregate']['scientific_primary_energy_statistics']['energy_j']['n'] == 3
    assert derived['run']['historical_process_result_source'] == 'result.run'
    assert source == checkpoint_path.read_bytes() and aggregate_before == (root / 'energy_aggregate.json').read_bytes()
    output = tmp_path / (model + '_reimport.json')
    args = ['--run-root', str(fixture.resolve()), '--checkpoint', str(checkpoint_path.resolve()),
            '--original-measurement-root', execution['output_dir'],
            '--extracted-measurement-root', str(root.resolve()), '--output', str(output)]
    assert replay.main(args) == 0
    assert replay.main(args) == 0


def test_original_checkpoint_corruption_is_not_reinterpreted_as_provenance():
    fixture = Path(__file__).parent / 'fixtures/v282_energy_retry_original/yolo26m'
    checkpoint = json.loads((fixture / 'checkpoint.json').read_text())
    checkpoint['execution']['aggregate_absent_before_execution'] = False
    with pytest.raises(ValueError, match='historical_checkpoint_digest_mismatch'):
        energy._reimport_energy_checkpoint(checkpoint)


@pytest.mark.parametrize('field,path_key', [
    ('workload_timing', 'path'), ('runtime_work_unit_evidence', 'evidence_path'),
    ('preflight_evidence', 'attestation_path'),
])
def test_embedded_completion_evidence_cannot_point_to_discarded_attempt(tmp_path, field, path_key):
    case = _case(tmp_path)
    case[1]['runs'][0].setdefault(field, {})[path_key] = str(case[0] / 'run_000/foreign')
    assert _attach(case)['energy_aggregate_verified'] is False


def test_reimport_rejects_aggregate_changed_after_original_bound_import(tmp_path):
    case = _case(tmp_path)
    checkpoint = _checkpoint(case)
    checkpoint['result']['run']['energy_aggregate_sha256'] = 'f' * 64
    result = energy._reimport_energy_checkpoint(checkpoint)
    assert result['ok'] is False
    assert 'historical_bound_aggregate_hash_mismatch' in result['run']['energy_aggregate_validation_errors']


@pytest.mark.parametrize('value', [None, True, '3', -1])
def test_reimport_does_not_reconstruct_missing_expected_count_from_command(tmp_path, value):
    checkpoint = _checkpoint(_case(tmp_path))
    checkpoint['execution']['expected_runs'] = value
    with pytest.raises(ValueError, match='historical_expected_repeat_contract_missing'):
        energy._reimport_energy_checkpoint(checkpoint)


def test_full_projection_joins_original_identity_and_preserves_untouched_rows():
    fixtures = Path(__file__).parent / 'fixtures/v282_energy_retry_original'
    rows, records = [], []
    for index, model in enumerate(('yolo26m', 'yolo26s')):
        fixture = fixtures / model
        checkpoint = json.loads((fixture / 'checkpoint.json').read_text())
        derived = energy._reimport_energy_checkpoint(checkpoint, measurement_path_mapping=(
            checkpoint['execution']['output_dir'], (fixture / 'measurement').resolve()))
        rows.append(checkpoint['result'])
        records.append({'source_row_index': index, **derived})
    rows.append({'row': {'model': 'unchanged'}, 'run': {'energy_aggregate_verified': True}, 'ok': True})
    before = copy.deepcopy(rows)
    projection = replay._project_existing_energy_report({'rows': rows}, records)
    assert projection['original_verified_import_count'] == 1
    assert projection['projected_verified_import_count'] == 3
    assert projection['unchanged_row_count'] == 1
    assert projection['rows'][2] == rows[2] and rows == before
    wrong = copy.deepcopy(records)
    wrong[0]['row']['case'] = 'b999'
    with pytest.raises(ValueError, match='identity conflict'):
        replay._project_existing_energy_report({'rows': rows}, wrong)
    wrong = copy.deepcopy(records)
    wrong[0]['run']['energy_aggregate_sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='digest conflict'):
        replay._project_existing_energy_report({'rows': rows}, wrong)


def test_selected_stdout_changed_during_read_is_rejected(tmp_path, monkeypatch):
    case = _case(tmp_path)
    selected = Path(case[1]['runs'][0]['repeat_attempt_history'][1]['run_directory']) / 'workload_stdout.log'
    real_read = Path.read_bytes
    def changed_read(path):
        raw = real_read(path)
        if path == selected:
            path.write_bytes(raw + b'changed during import\n')
        return raw
    monkeypatch.setattr(Path, 'read_bytes', changed_read)
    result = _attach(case)
    assert result['energy_aggregate_verified'] is False
    assert any('stdout_changed_during_import' in value for value in result['energy_aggregate_completion_errors'])


def test_archive_normalized_mtime_uses_bound_historical_mtime_without_touching_source(tmp_path):
    import os
    case = _case(tmp_path / 'original')
    checkpoint = _checkpoint(case)
    original_root = case[0]
    extracted = tmp_path / 'extracted'
    original_root.rename(extracted)
    aggregate = extracted / 'energy_aggregate.json'
    raw = aggregate.read_bytes()
    os.utime(aggregate, (315532800, 315532800))  # deterministic ZIP member date 1980
    derived = energy._reimport_energy_checkpoint(checkpoint, measurement_path_mapping=(original_root, extracted))
    assert derived['ok'] is True, derived
    assert aggregate.read_bytes() == raw and aggregate.stat().st_mtime_ns == 315532800000000000
    projection = derived['run']['energy_archive_path_projection']
    assert projection['original_aggregate_mtime_ns'] == checkpoint['result']['run']['energy_aggregate_mtime_ns']
    assert projection['extracted_aggregate_mtime_ns'] == 315532800000000000
    checkpoint['result']['run']['energy_aggregate_mtime_ns'] = 1
    assert energy._reimport_energy_checkpoint(checkpoint, measurement_path_mapping=(original_root, extracted))['ok'] is False
    checkpoint['result']['run'].pop('energy_aggregate_mtime_ns')
    assert energy._reimport_energy_checkpoint(checkpoint, measurement_path_mapping=(original_root, extracted))['ok'] is False
