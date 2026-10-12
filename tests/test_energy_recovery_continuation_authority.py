"""Scientific continuation after a distinct, already released physical recovery."""
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os

import pytest
import yaml

from test_physical_resource_recovery import recovery_case, _write
from test_bounded_energy_capture_recovery import bounded_case, interrupted_native_case
from onnx_splitpoint_tool.energy import task_budget
from onnx_splitpoint_tool.energy.recovery_accounting import reconcile_stopped_capture
from onnx_splitpoint_tool.process_control import _proc_start_time
from onnx_splitpoint_tool.remote import resource_recovery


@pytest.fixture
def continuation_case(interrupted_native_case, tmp_path):
    c = interrupted_native_case
    budget = json.loads(c.budget.read_text())
    original_budget = c.budget
    c.budget = c.run_dir.parent / 'energy_task_budget.json'
    config_path = c.run_dir / 'energy_parent_plan.json'
    config = json.loads(config_path.read_text())
    config['campaign_budget_file'] = str(c.budget)
    _write(config_path, config)
    _write(c.run_dir / 'PREPARED.json', {'ok': True,
        'plan_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest()})
    start_path = c.run_dir / 'production-start.json'
    start = json.loads(start_path.read_text())
    start['argv'] = [str(c.budget) if x == str(original_budget) else x for x in start['argv']]
    _write(start_path, start)
    source = '192.0.2.176'
    c.source = source
    second = copy.deepcopy(budget['tasks'][c.row]['chains'][0])
    second.update(logical_repeat='repeat:2', run_directory=str(c.capture.parent / 'valid2'))
    budget['tasks'][c.row]['chains'].append(second)
    authority = tmp_path / 'original_authority.txt'
    authority.write_text('Existing bounded campaign authority, preserved.\n')
    budget['sources'][source]['continuous_admission'] = {
        'continuous_campaign': True, 'authorization_file': str(authority),
        'authorization_sha256': hashlib.sha256(authority.read_bytes()).hexdigest(),
        'row_ids': [c.row], 'campaign_row': c.row,
        'chain': copy.deepcopy(budget['tasks'][c.row]['chains'][0]), 'previous_grants': [],
        'original_source_stop_reason': '', 'original_task_stops': {c.row: ''},
        'source_stop_reason': '', 'task_stop_reasons': {}, 'limit_changes': {},
        'max_transport_failures': 20}
    _write(c.budget, budget)
    reconcile_stopped_capture(c.budget, parent_attempt=c.run_dir,
                             operation_id=c.request['operation_id'], run_dir=c.run_dir)
    assert resource_recovery.recover_capture_resources(**c.kwargs)['status'] == 'released'
    c.reply = json.loads(c.reply_path.read_text())
    profile = tmp_path / 'scientific_continuation.yaml'
    profile.write_text(yaml.safe_dump({
        'run_dir': str(c.run_dir), 'historical_operation': c.request['operation_id'],
        'source': source, 'setup_id': c.request['setup_id'], 'same_original_budget': str(c.budget),
        'unchanged_limits': budget['tasks'][c.row]['limits'], 'source_completion_remains_unproven': True,
        'missing_slots': ['repeat:1'], 'duration_s': 60, 'cohort_repeats': 3}))
    c.authorization = {**c.kwargs['retry_authorization'],
        'authorized_at': datetime.now(timezone.utc).isoformat(),
        'reason': 'Explicit bounded scientific continuation after the earlier physical release',
        'new_workflow_profile': str(profile)}
    c.parent = {'run_id': c.run_dir.name, 'session_id': 'scientific-continuation',
                'pid': os.getpid(), 'start_ticks': _proc_start_time(os.getpid())}
    return c


def install(c, authorization=True):
    return task_budget.install_authorized_source_reentry(c.budget, capture_request=c.request,
        recovery_reply=c.reply, new_parent=c.parent,
        continuation_authorization=c.authorization if authorization else None)


def test_normal_reconcile_and_distinct_continuation_preserve_originals(continuation_case):
    c = continuation_case
    before = json.loads(c.budget.read_text())
    originals = {p: p.read_bytes() for p in c.run_dir.rglob('*') if p.is_file()}
    old = before['sources'][c.source]['continuous_admission']
    assert before['sources'][c.source]['stop_reason'] == ''
    assert old['source_stop_reason'] == 'campaign_source_interrupted_chain_unresolved'
    grant = install(c)
    after = json.loads(c.budget.read_text())
    assert after['tasks'] == before['tasks']
    assert after['sources'][c.source]['stop_reason'] == ''
    assert after['sources'][c.source]['transport_failures'] == before['sources'][c.source]['transport_failures']
    assert grant['previous_grants'] == [old]
    assert grant['retry_authorization'] == c.reply['recovery']['retry_authorization']
    assert grant['continuation_authorization'] == c.authorization
    assert grant['source_stop_reason'] == '' and grant['task_stop_reasons'] == {}
    assert grant['chain']['valid'] is False and grant['chain']['source_completion_verified'] is False
    assert all(p.read_bytes() == data for p, data in originals.items())
    assert install(c) == grant
    assert task_budget._source_reentry(after, c.source, {}) == grant


@pytest.mark.parametrize('fault', ['raw_stop', 'missing_accounting', 'changed_accounting',
    'current_stop', 'foreign_task_stop', 'extra_unresolved', 'no_continuous',
    'stale', 'future', 'foreign_source', 'wrong_parent', 'wrong_operation',
    'changed_limits', 'wrong_budget', 'diagnostic_profile', 'no_new_authorization'])
def test_new_continuation_rejects_drift_without_changing_budget(continuation_case, fault):
    c = continuation_case
    root = json.loads(c.budget.read_text())
    source = root['sources'][c.source]
    key = c.request['session_id'] + '/' + c.request['operation_id']
    if fault == 'raw_stop': source['stop_reason'] = 'foreign_STOP'
    elif fault == 'missing_accounting': root['failure_reconciliations'].pop(key)
    elif fault == 'changed_accounting': root['failure_reconciliations'][key]['booked_chain']['valid'] = True
    elif fault == 'current_stop': source['continuous_admission']['source_stop_reason'] = 'foreign_STOP'
    elif fault == 'foreign_task_stop': source['continuous_admission']['task_stop_reasons']['foreign'] = 'STOP'
    elif fault == 'extra_unresolved':
        chain = copy.deepcopy(root['tasks'][c.row]['chains'][1]); chain['run_directory'] += '-other'
        root['tasks'][c.row]['chains'].append(chain)
    elif fault == 'no_continuous': source.pop('continuous_admission')
    elif fault in {'stale', 'future'}:
        delta = timedelta(days=-1 if fault == 'stale' else 1)
        c.authorization['authorized_at'] = (datetime.now(timezone.utc) + delta).isoformat()
    elif fault == 'foreign_source': c.authorization['source'] = 'source:192.0.2.199'
    elif fault in {'wrong_parent', 'wrong_operation', 'changed_limits', 'wrong_budget', 'diagnostic_profile'}:
        from pathlib import Path
        path = Path(c.authorization['new_workflow_profile']); profile = yaml.safe_load(path.read_text())
        if fault == 'wrong_parent': profile['run_dir'] += '-other'
        elif fault == 'wrong_operation': profile['historical_operation'] = 'capture-other'
        elif fault == 'changed_limits': profile['unchanged_limits']['max_chains'] += 1
        elif fault == 'wrong_budget': profile['same_original_budget'] += '-other'
        else: profile = {'hardware_acceptance': True, 'repeats': 1}
        path.write_text(yaml.safe_dump(profile))
    _write(c.budget, root)
    before = c.budget.read_bytes()
    with pytest.raises(ValueError): install(c, authorization=fault != 'no_new_authorization')
    assert c.budget.read_bytes() == before


def test_bound_continuation_profile_cannot_change_after_handoff(continuation_case):
    from pathlib import Path
    c = continuation_case; install(c)
    path = Path(c.authorization['new_workflow_profile']); path.write_text(path.read_text() + '# changed\n')
    with pytest.raises(ValueError, match='continuation profile changed'):
        task_budget._source_reentry(json.loads(c.budget.read_text()), c.source, {})


def test_profile_change_during_handoff_cannot_be_sealed(continuation_case, monkeypatch):
    from pathlib import Path
    c = continuation_case
    path = Path(c.authorization['new_workflow_profile'])
    original_read = Path.read_bytes
    changed = False
    def read_bytes(p):
        nonlocal changed
        data = original_read(p)
        if p == path and not changed:
            changed = True
            path.write_text('hardware_acceptance: true\n')
        return data
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    before = c.budget.read_bytes()
    with pytest.raises(ValueError, match='profile changed during handoff'): install(c)
    assert c.budget.read_bytes() == before


def test_later_normal_recovery_keeps_prior_continuation_only_in_history(continuation_case):
    from pathlib import Path
    c = continuation_case
    first = install(c)
    root = json.loads(c.budget.read_text())
    chain = copy.deepcopy(first['chain'])
    chain['run_directory'] += '-later-failure'
    chain['transport_reasons'] = ['receive_silence_timeout']
    root['tasks'][c.row]['chains'].append(chain)
    source = root['sources'][c.source]
    source['transport_failures'] += 1
    reason = 'campaign_source_completion_unresolved'
    source['continuous_admission']['source_stop_reason'] = reason
    source['continuous_admission']['task_stop_reasons'] = {c.row: reason}
    previous = copy.deepcopy(source['continuous_admission'])
    _write(c.budget, root)
    request = {**c.request, 'operation_id': 'capture-' + 'f' * 32,
               'attempt_dir': chain['run_directory'], 'phase': 'FINALIZING', 'process_started': True}
    profile = Path(c.authorization['new_workflow_profile']).with_name('later_normal.yaml')
    payload = yaml.safe_load(Path(c.authorization['new_workflow_profile']).read_text())
    payload['historical_operation'] = request['operation_id']
    profile.write_text(yaml.safe_dump(payload))
    authorization = {**c.authorization, 'authorized_at': datetime.now(timezone.utc).isoformat(),
                     'new_workflow_profile': str(profile)}
    identity = {k: request[k] for k in ('run_id', 'session_id', 'operation_id')}
    recovery = {**c.reply['recovery'], **identity, 'retry_authorization': authorization,
                'released_at': datetime.now(timezone.utc).isoformat()}
    recovery.pop('interrupted_capture')
    reply = {**c.reply, **identity, 'reason': reason, 'recovery': recovery}
    grant = task_budget.install_authorized_source_reentry(c.budget, capture_request=request,
        recovery_reply=reply, new_parent={**c.parent, 'session_id': 'later-normal-recovery'})
    assert 'continuation_authorization' not in grant and 'continuation_profile_sha256' not in grant
    assert grant['previous_grants'] == [previous]
    assert grant['previous_grants'][0]['continuation_authorization'] == c.authorization
    after = json.loads(c.budget.read_text())
    assert after['tasks'] == root['tasks']
    assert task_budget._source_reentry(after, c.source, {}) == grant


def test_consumed_handoff_cannot_replace_its_continuation_authority(continuation_case):
    c = continuation_case; install(c)
    before = c.budget.read_bytes()
    c.authorization['reason'] = 'different authorization'
    with pytest.raises(ValueError, match='later STOP requires'): install(c)
    assert c.budget.read_bytes() == before


@pytest.mark.parametrize('replay_status', ['already_released', 'released'])
def test_recover_routes_old_physical_and_new_scientific_authorities_separately(
        continuation_case, monkeypatch, tmp_path, replay_status):
    from types import SimpleNamespace
    from onnx_splitpoint_tool.workflow import energy_resume, run_control
    c = continuation_case
    authorization_path = tmp_path / 'continuation_authorization.json'
    _write(authorization_path, c.authorization)
    class InspectLock:
        def __init__(self, **kwargs): self.quarantine_path = c.run_dir / 'cleanup-status.json'
        def acquire_for_recovery(self): pass
        def release(self, **kwargs): assert kwargs == {'write_released_state': False}
    monkeypatch.setattr(run_control, 'EvaluationRunLock', InspectLock)
    monkeypatch.setattr(energy_resume, 'budget_path', lambda run: c.budget)
    monkeypatch.setattr(energy_resume, 'predecessor', lambda *args: (c.journal, c.request, c.reply, {}))
    physical = []
    def replay(**kwargs):
        physical.append(kwargs['retry_authorization'])
        # Physical replay was covered with real temporary locks above; this
        # orchestration check substitutes only that already verified leaf.
        return {'status': replay_status, 'resources': c.reply['recovery']['resources']}
    monkeypatch.setattr(resource_recovery, 'recover_capture_resources', replay)
    handoffs = []
    monkeypatch.setattr(energy_resume, 'recover_run_fence', lambda *a, **k: handoffs.append(k))
    before = c.reply_path.read_bytes(), c.budget.read_bytes()
    args = SimpleNamespace(run_dir=str(c.run_dir), failed_attempt=str(c.run_dir),
                           authorization=str(authorization_path))
    if replay_status == 'already_released':
        energy_resume.recover(args)
        assert handoffs == [{'continuation_authorization': c.authorization}]
    else:
        with pytest.raises(ValueError, match='unchanged already released'): energy_resume.recover(args)
        assert handoffs == []
    assert physical == [c.reply['recovery']['retry_authorization']]
    assert before == (c.reply_path.read_bytes(), c.budget.read_bytes())
