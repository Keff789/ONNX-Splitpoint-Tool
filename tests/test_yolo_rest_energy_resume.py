"""Six stored energy rows and exact recovery; no device/collector dispatch."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import shutil

import pytest

from test_yolo_augmentation_rest_blocks import actual_plan, phase_fixture, AUG
from test_physical_resource_recovery import recovery_case
from test_yolo_rest_capture_recovery import rest_case
from onnx_splitpoint_tool.workflow import yolo_augmentation_rest as rest
from onnx_splitpoint_tool.remote import resource_recovery, process_lease
from onnx_splitpoint_tool.native_energy_reporting import collect_native_energy

PREVIOUS = AUG / 'attempts/restabschluss_20261004_065600'


@pytest.fixture
def continuation(actual_plan):
    return rest.energy_continuation(actual_plan, PREVIOUS)


def test_actual_eleven_pairs_six_energy_rows_are_retained(continuation):
    assert continuation['reused_energy_rows'] == 6
    assert continuation['remaining_energy_rows'] == 5
    inventory = continuation['inventory']
    assert inventory['generic_cases'] == inventory['native_cases'] == 11
    assert inventory['generic_repetitions'] == inventory['native_repetitions'] == 33
    assert inventory['quality_results'] == 12 and inventory['block1_generic'] == []
    rows = collect_native_energy(continuation['previous_block'], include_partial=True)
    assert len(rows) == 7
    assert sum(row['ok'] is True for row in rows) == 6
    assert sum(row['energy_repeat_valid_n'] or 0 for row in rows if row['ok'] is True) == 18
    # The unchanged old publication has six complete rows. The current
    # read-only campaign projection also verifies three later H10 repeats;
    # it must preserve their old row's failure and claim status.
    partial = [row for row in rows if row['ok'] is not True]
    assert sum(row['energy_repeat_valid_n'] or 0 for row in partial) == 3
    assert all(row['claim_eligible'] is not True for row in partial)


@pytest.mark.parametrize('fault', ['nonterminal', 'foreign_plan', 'aggregate_sha'])
def test_previous_attempt_or_aggregate_drift_is_rejected(actual_plan, monkeypatch, fault):
    read = rest.io.read_json
    def altered(path):
        data = read(path)
        if Path(path) == PREVIOUS / 'terminal-status.json' and fault == 'nonterminal':
            data['status'] = 'running'
        if Path(path) == PREVIOUS / 'production-start.json' and fault == 'foreign_plan':
            data['plan'] = '/other/plan.json'
        if Path(path).name == 'native_producer_energy_results.partial.json' and fault == 'aggregate_sha':
            data['rows'][0]['run']['energy_aggregate_sha256'] = '0' * 64
        return data
    monkeypatch.setattr(rest.io, 'read_json', altered)
    with pytest.raises((ValueError, RuntimeError)):
        rest.energy_continuation(actual_plan, PREVIOUS)


def test_existing_energy_cli_gets_previous_results_and_original_budget(actual_plan, continuation, tmp_path):
    run = Path(actual_plan['augmentation_run'])
    cfg = rest.io.read_json(run / 'reports/native_producer_stage_config.json')
    cmd = rest.energy_command(run, cfg, run / 'reports/native_producer_combined_summary.json',
        run / 'reports/native_validation/native_producer_validation_summary.json', tmp_path / 'block1',
        previous_results=Path(continuation['previous_results']))
    assert cmd[cmd.index('--previous-results') + 1] == continuation['previous_results']
    assert cmd[cmd.index('--campaign-budget-file') + 1] == str(run / 'energy_task_budget.json')
    assert cmd[cmd.index('--runs') + 1] == '3'
    assert '--resume-existing' not in cmd and '--resume-checkpoint' not in cmd


@pytest.mark.parametrize('stage', ['absent', 'early_partial', 'imported_six'])
def test_energy_delivery_keeps_six_once_even_before_import(continuation, tmp_path, stage):
    work = tmp_path / 'work'; block = work / 'block1'
    payload = rest.io.read_json(Path(continuation['previous_results']))
    if stage != 'absent':
        if stage == 'early_partial': payload['rows'] = []
        rest.io.write_json(block / 'reports/native_energy_measurements/native_producer_energy_results.partial.json', payload)
        source = Path(continuation['previous_block']) / 'reports/native_validation/native_producer_validation_summary.json'
        rest._copy_validation(source, block)
    roots = rest._energy_delivery_roots(work, continuation)
    assert roots == ([block] if stage == 'imported_six' else [Path(continuation['previous_block'])])
    rows = collect_native_energy(roots[0], include_partial=True)
    assert sum(row['ok'] is True for row in rows) == 6
    assert sum(row['energy_repeat_valid_n'] or 0 for row in rows if row['ok'] is True) == 18
    # The unchanged old publication has six complete rows. The current
    # read-only campaign projection also verifies three later H10 repeats;
    # it must preserve their old row's failure and claim status.
    partial = [row for row in rows if row['ok'] is not True]
    assert sum(row['energy_repeat_valid_n'] or 0 for row in partial) == 3
    assert all(row['claim_eligible'] is not True for row in partial)


def test_new_measurement_cannot_hide_missing_preserved_aggregate(continuation, tmp_path):
    payload = rest.io.read_json(Path(continuation['previous_results']))
    payload['rows'] = payload['rows'][:1]
    rest.io.write_json(tmp_path / 'block1/reports/native_energy_measurements/native_producer_energy_results.partial.json', payload)
    with pytest.raises(ValueError, match='every verified previous aggregate'):
        rest._energy_delivery_roots(tmp_path, continuation)


def test_resume_dispatches_energy_only_and_preserves_preflight_failure(phase_fixture, monkeypatch):
    output, phases, delivery, _ = phase_fixture
    previous = output.parent / 'old'; block = output.parent / 'old_block'
    descriptor = dict(previous_attempt=str(previous), previous_results=str(block / 'partial.json'),
        previous_block=str(block), inventory={'block1_energy': [('model', 'case', 'setup')]})
    monkeypatch.setattr(rest, 'energy_continuation', lambda *args: descriptor)
    monkeypatch.setattr(rest, '_energy_delivery_roots', lambda *args: [block])
    calls = []
    def command(*args, **kwargs):
        assert kwargs == {'previous_results': Path(descriptor['previous_results'])}
        return ['energy']
    monkeypatch.setattr(rest, 'energy_command', command)
    @contextmanager
    def owned(*args, **kwargs):
        assert kwargs['previous_attempt'] == str(previous)
        def execute(cmd, label, *args):
            calls.append(label)
            raise RuntimeError('campaign_source_interrupted_chain_unresolved')
        yield execute, {}, None, None
    monkeypatch.setattr(rest, 'owned_execution', owned)
    monkeypatch.setattr(rest, 'singlecase_command', lambda *args: pytest.fail('no Native before completed energy'))
    assert rest.finish_rest(Path('plan.json'), output, include_b066=True, previous_energy_attempt=previous) == 1
    assert calls == ['block1-remaining-energy'] and phases == []
    assert 'campaign_source_interrupted_chain_unresolved' in delivery['error']
    assert not (output / 'cleanup-status.json').exists()
    stop = rest.io.read_json(output / 'technical-stop.json')
    assert stop['primary_error'] == stop['error'] and stop['cleanup_error'] == ''


def test_admission_failure_before_yield_does_not_require_cleanup_file(phase_fixture, monkeypatch):
    output, phases, delivery, _ = phase_fixture
    @contextmanager
    def owned(*args, **kwargs):
        raise RuntimeError('physical recovery not released')
        yield
    monkeypatch.setattr(rest, 'owned_execution', owned)
    assert rest.finish_rest(Path('plan.json'), output) == 1
    assert phases == [] and 'physical recovery not released' in delivery['error']
    assert not (output / 'cleanup-status.json').exists()


def test_projection_conflict_does_not_replace_primary_failure(phase_fixture, monkeypatch):
    output, phases, delivery, reject = phase_fixture
    reject.append('block1-eleven-energy')
    def conflict(*args): raise ValueError('preserved aggregate differs')
    monkeypatch.setattr(rest, '_energy_delivery_roots', conflict)
    assert rest.finish_rest(Path('plan.json'), output) == 1
    assert delivery['status'] == 'energy_source_conflict'
    assert 'controlled phase failure' in delivery['error'] and 'preserved aggregate differs' in delivery['error']
    proof = rest.io.read_json(output / 'energy-source-conflict.json')
    assert 'controlled phase failure' in proof['primary_error']


def test_original_child_error_and_separate_cleanup_error_are_both_preserved(actual_plan, tmp_path, monkeypatch):
    from onnx_splitpoint_tool import native_progress
    run = tmp_path / 'run'; output = tmp_path / 'new_session'; output.mkdir()
    rest.io.write_json(run / 'jobs/workflow_control.json', {'run_id': run.name, 'session_id': 'old'})
    cfg = rest.io.read_json(Path(actual_plan['augmentation_run']) / 'reports/native_producer_stage_config.json')
    monkeypatch.setattr(rest.io, 'bound_targets', lambda *args: {})
    monkeypatch.setattr(rest.io, 'observe_owners', lambda *args: None)
    monkeypatch.setattr(native_progress, 'run_streaming', lambda *args, **kwargs:
        SimpleNamespace(returncode=1, stdout='Traceback: campaign_source_completion_unresolved', stderr=''))
    original = process_lease.ControllerResourceBroker.close
    def close(broker):
        original(broker)
        raise RuntimeError('independent broker cleanup failure')
    monkeypatch.setattr(process_lease.ControllerResourceBroker, 'close', close)
    with pytest.raises(RuntimeError, match='campaign_source_completion_unresolved'):
        with rest.owned_execution({'augmentation_run': str(run), 'rows': []}, output, cfg) as (execute, *_):
            execute(['controlled-child'], 'energy')
    cleanup = rest.io.read_json(output / 'cleanup-status.json')
    assert 'campaign_source_completion_unresolved' in cleanup['primary_error']
    assert cleanup['cleanup_error'] == 'rest cleanup is not proven'
    assert any('independent broker cleanup failure' in value for value in cleanup['broker_errors'])
    assert 'campaign_source_completion_unresolved' in rest.io.read_json(output / 'primary-failure.json')['error']


@pytest.fixture
def run_fence_case(rest_case, tmp_path):
    case = rest_case
    owner = dict(run_id=case.run_dir.name, session_id=case.rest_attempt.name,
                 purpose='yolo_rest_completion', pid=99999993)
    fence = dict(schema='onnx-splitpoint/evaluation-run-lock-quarantine', run_id=case.run_dir.name,
        run_dir=str(case.run_dir), owner=dict(reason='rest_cleanup_unproven', pending=0, errors=['source stopped']))
    return case, dict(augmentation_run=str(case.run_dir)), tmp_path / 'new_attempt', SimpleNamespace(previous_owner=owner), fence


def test_run_fence_accepts_only_normal_exact_released_source(run_fence_case):
    case, plan, output, lock, fence = run_fence_case
    before = case.reply_path.read_bytes()
    resource_recovery.recover_capture_resources(**case.kwargs)
    assert rest._recover_rest_run_fence(plan, case.rest_attempt, output, lock, fence) is True
    proof = rest.io.read_json(output / 'reports/rest-runlock-recovery.json')
    assert proof['original_run_fence'] == fence
    assert proof['physical_recovery_reply']['state'] == 'STOP'
    assert before != case.reply_path.read_bytes()
    for path, data in case.preserved.items():
        # The fixture relocates only the journal and deliberately changes the old manifest to partial.
        if path.is_file() and path.name != 'run_manifest.json': assert path.read_bytes() == data


@pytest.mark.parametrize('fault', ['missing_release', 'foreign', 'active', 'missing_cleanup', 'foreign_cleanup', 'physical_fence'])
def test_run_fence_remains_blocked_for_incomplete_recovery(run_fence_case, monkeypatch, fault):
    case, plan, output, lock, fence = run_fence_case
    if fault != 'missing_release': resource_recovery.recover_capture_resources(**case.kwargs)
    if fault == 'foreign':
        payload = rest.io.read_json(case.reply_path); payload['recovery']['session_id'] = 'foreign'
        rest.io.write_json(case.reply_path, payload)
    elif fault == 'active':
        from onnx_splitpoint_tool import process_control
        monkeypatch.setattr(process_control, '_proc_start_time', lambda pid: '103')
    elif fault == 'missing_cleanup': (case.rest_attempt / 'cleanup-status.json').unlink()
    elif fault == 'foreign_cleanup':
        path = case.rest_attempt / 'cleanup-status.json'; payload = rest.io.read_json(path)
        payload['foreign_processes_stopped'] = True; rest.io.write_json(path, payload)
    elif fault == 'physical_fence':
        rest.io.write_json(next(iter(case.locks.values())).quarantine_path, {'state': 'quarantined'})
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        rest._recover_rest_run_fence(plan, case.rest_attempt, output, lock, fence)
    assert not (output / 'reports/rest-runlock-recovery.json').exists()
