"""Local budget transitions only; released recovery evidence is a test fixture.

The real broker metadata writer binds the current test parent. No collector,
source recovery, remote process or measurement is started by these tests.
"""
import copy
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy.task_budget import (
    EnergyTaskBudget, campaign_budget_stop_reason, install_authorized_source_reentry,
)
from onnx_splitpoint_tool.process_control import _proc_start_time
from onnx_splitpoint_tool.remote.process_lease import (
    ControllerResourceBroker, RemoteProcessLeaseJournal, RemoteProcessLeaseScope,
)

SOURCE = '192.0.2.176'
ROW = json.dumps(['hailo10h_to_trt', 'yolov7_paper', 'b044', 'fixture-h10'], separators=(',', ':'))
LIMITS = {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20}
LIFECYCLE = ('Fast Firmware GO sent monotonic_ns=1 local=127.0.0.1:1234 requested_device_us=1000000\n'
             'Fast Firmware protocol end verified monotonic_ns=1000000001 local=127.0.0.1:1234 elapsed_us=1000000\n'
             'Fast Firmware source closed monotonic_ns=1000000002 local=127.0.0.1:1234 verified=true reason=Ok(())\n')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path):
    run = tmp_path / 'runs' / 'campaign'
    run.mkdir(parents=True)
    previous = run / 'old' / 'run_001'
    previous.mkdir(parents=True)
    request = dict(run_id=run.name, session_id='previous', operation_id='capture-' + 'a'*32,
                   token='b'*32, setup_id='fixture-h10', attempt_dir=str(previous),
                   owner_pid=99999991, owner_start_ticks='101', collector_pid=99999992,
                   collector_start_ticks='102', phase='FINALIZING', process_started=True, purpose='capture')
    chains = [dict(owner_pid=99999991, owner_start_ticks='101', logical_repeat='repeat:0',
                   run_directory=str(run / 'old' / 'run_000'), preflight_requested=True,
                   collector_started=True, finished=True, valid=False,
                   source_completion_verified=True, transport_reasons=['marker_dropped_samples_nonzero']),
              dict(owner_pid=99999991, owner_start_ticks='101', logical_repeat='repeat:1',
                   run_directory=str(previous), preflight_requested=True, collector_started=True, finished=False)]
    root = dict(campaign=True, max_transport_failures=20,
                sources={SOURCE: dict(stop_reason='', transport_failures=3)},
                tasks={ROW: dict(limits=LIMITS.copy(), source_id=SOURCE,
                                 stop_reason='task_execution_interrupted', chains=chains),
                       'previous-completed': dict(limits=LIMITS.copy(), source_id=SOURCE, stop_reason='',
                           chains=[dict(chains[0], run_directory=str(run / 'previous' / str(i))) for i in range(2)])})
    path = run / 'energy_task_budget.json'
    write(path, root)
    resources = ['controller:capture', 'controller:nic', 'dut:192.0.2.145', 'source:' + SOURCE]
    now = datetime.now(timezone.utc)
    profile = tmp_path / 'profile.yaml'; profile.write_text('enabled: true\n')
    reply = {k: request[k] for k in ('run_id', 'session_id', 'operation_id', 'token')}
    reply.update(state='STOP', reason='campaign_source_completion_unresolved', recovery={
        **{k: request[k] for k in ('run_id', 'session_id', 'operation_id')},
        'status': 'released', 'old_attempt_validated': False, 'measurements_started': 0,
        'released_at': now.isoformat(), 'resources': resources, 'ownership_observation': {'ok': True},
        'original_resources': {r: {'quarantine': {'quarantined_at': (now-timedelta(minutes=1)).isoformat()}} for r in resources},
        'retry_authorization': dict(source='source:'+SOURCE, operator='software-fixture',
            authorized_at=now.isoformat(), reason='offline unit fixture, no physical release claimed',
            new_workflow_profile=str(profile), authorize_new_attempt=True, scope='one_new_workflow',
            source_completion_unproven=True)})
    parent = dict(run_id=run.name, session_id='authorized-new', pid=os.getpid(), start_ticks=_proc_start_time(os.getpid()))
    journal = RemoteProcessLeaseJournal(scope=RemoteProcessLeaseScope(run.name, parent['session_id']),
                                       directory=str(tmp_path / 'journal'))
    ControllerResourceBroker(journal, None, run_dir=run, setups={'fixture-h10': resources})
    env = {**journal.environment(), 'ONNX_SPLITPOINT_CONTROLLER_RESOURCES': 'required'}
    return SimpleNamespace(path=path, root=root, run=run, request=request, reply=reply, parent=parent,
                           env=env, row=ROW, source=SOURCE, journal=journal)


def install(case):
    return install_authorized_source_reentry(case.path, capture_request=case.request,
        recovery_reply=case.reply, new_parent=case.parent)


@contextmanager
def budget(case, row=None, limits=None, cancel_event=None):
    with case.path.open('r+') as handle:
        yield EnergyTaskBudget(handle, limits or LIMITS, cancel_event, campaign_row=row or case.row,
                               source_id=case.source, environment=case.env)


def historical_unchanged(case):
    value = json.loads(case.path.read_text())
    assert value['tasks'][case.row]['chains'][:2] == case.root['tasks'][case.row]['chains']
    for key in ('stop_reason', 'limits', 'source_id'):
        assert value['tasks'][case.row][key] == case.root['tasks'][case.row][key]
    assert value['sources'][case.source]['stop_reason'] == case.root['sources'][case.source]['stop_reason']
    assert value['max_transport_failures'] == 20
    return value


def test_existing_guard_is_reproduced_without_authorization(case):
    with budget(case) as current:
        assert current.is_set()
        assert not current.allowed('repeat:1')
        assert current.counts()['begun_chains'] == 2
        assert current.counts()['valid_logical_repeats'] == 0


def test_one_parent_can_use_remaining_budget_without_rewriting_history(case):
    before = copy.deepcopy(case.root)
    grant = install(case)
    assert install(case) == grant
    current = historical_unchanged(case)
    current['sources'][SOURCE].pop('authorized_reentry')
    assert current == before
    with budget(case) as current:
        assert not current.is_set() and current.allowed('repeat:1')
        assert current.counts()['begun_chains'] == 2
        assert current.counts()['valid_logical_repeats'] == 0
        assert current.source['transport_failures'] == 3
        chain = current.reserve('repeat:1', case.run/'new'/'run_001', True)
        assert chain and current.counts()['begun_chains'] == 3
        assert current.report({'ok': True}, case.run/'report')['ok'] is True
    historical_unchanged(case)


def test_fresh_unknown_source_stops_this_session_and_diagnostic_rows(case):
    install(case)
    with budget(case) as current:
        chain = current.reserve('repeat:1', case.run/'new'/'run_001', True)
        current.collector_started(chain)
        current.finish(chain, {'ok': False}, case.run/'new'/'run_001')
        assert current.is_set()
        result = current.report({'ok': False}, case.run/'report')
        assert result['error'] == 'campaign_source_completion_unresolved'
        assert result['task_budget']['stop_reason'] == 'task_execution_interrupted'
    with budget(case, row='diagnostic', limits={'max_chains': 1, 'max_retries': 0, 'max_transport_failures': 20}) as current:
        assert not current.allowed('repeat:0')
    value = historical_unchanged(case)
    assert campaign_budget_stop_reason(value, case.row, case.env) == 'campaign_source_completion_unresolved'
    # The exact same installation cannot clear a subsequent source failure.
    install(case)
    with budget(case) as current:
        assert current.is_set()


def test_completed_invalid_capture_keeps_normal_retry_and_source_count(case):
    install(case)
    run = case.run/'new'/'run_001'; run.mkdir(parents=True)
    (run/'collector_stdout.log').write_text(LIFECYCLE)
    with budget(case) as current:
        chain = current.reserve('repeat:1', run, True)
        current.collector_started(chain)
        current.finish(chain, {'ok': False, 'status': 'collector_finished', 'collector_started': True}, run)
        assert not current.is_set() and current.allowed('repeat:1')
        assert current.data['chains'][-1]['source_completion_verified'] is True
        assert current.data['chains'][-1]['valid'] is False
        assert current.counts()['begun_chains'] == 3
    historical_unchanged(case)


@pytest.mark.parametrize('failure', ['task_cancelled', 'task_execution_interrupted', 'task_chain_limit'])
def test_new_stop_is_effective_even_when_identical_to_historical_stop(case, failure):
    install(case)
    with budget(case) as current:
        current.stop(failure)
        assert current.is_set() and not current.allowed('repeat:2')
        assert current.report({'ok': True}, case.run/'report')['error'] == failure
    value = historical_unchanged(case)
    assert campaign_budget_stop_reason(value, case.row, case.env) == failure


@pytest.mark.parametrize('kind', ['foreign_session', 'foreign_pid', 'changed_chain', 'changed_limits', 'changed_old_stop'])
def test_consumer_binding_mutation_is_rejected_without_checkpoint_write(case, kind):
    install(case)
    if kind == 'foreign_session': case.env['ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID'] = 'different'
    elif kind == 'foreign_pid':
        p = case.journal.directory/'controller.resources.json'; v=json.loads(p.read_text()); v['parent_pid']=99999994; write(p,v)
    else:
        value = json.loads(case.path.read_text())
        if kind == 'changed_chain': value['tasks'][ROW]['chains'][1]['finished'] = True
        elif kind == 'changed_limits': value['tasks'][ROW]['limits']['max_chains'] = 10
        else: value['tasks'][ROW]['stop_reason'] = ''
        write(case.path, value)
    before = case.path.read_bytes()
    with pytest.raises(ValueError, match='authorized energy source reentry'):
        with budget(case): pass
    assert case.path.read_bytes() == before


@pytest.mark.parametrize('kind', ['token', 'old_validated', 'owner', 'source', 'row', 'additional_unknown', 'quota', 'parent', 'reason', 'authorization'])
def test_install_rejects_unbound_or_exhausted_capture(case, kind):
    value = json.loads(case.path.read_text())
    if kind == 'token': case.reply['token'] = 'c'*32
    elif kind == 'old_validated': case.reply['recovery']['old_attempt_validated'] = True
    elif kind == 'owner': case.request['owner_pid'] = 99999995
    elif kind == 'source': case.reply['recovery']['resources'][-1] = 'source:OTHER'
    elif kind == 'row': case.request['setup_id'] = 'different'
    elif kind == 'additional_unknown': value['tasks'][ROW]['chains'].append(dict(value['tasks'][ROW]['chains'][1],run_directory='/another'))
    elif kind == 'quota': value['sources'][SOURCE]['transport_failures'] = 20
    elif kind == 'parent': case.parent['session_id'] = 'previous'
    elif kind == 'reason': value['tasks'][ROW]['stop_reason'] = 'task_chain_limit'
    elif kind == 'authorization': case.reply['recovery']['retry_authorization']['authorize_new_attempt'] = False
    write(case.path, value)
    before = case.path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        install(case)
    assert case.path.read_bytes() == before


def test_authorization_cannot_be_rebound_to_another_parent(case):
    install(case)
    case.parent['session_id'] = 'next'
    before = case.path.read_bytes()
    with pytest.raises(ValueError, match='already consumed'): install(case)
    assert case.path.read_bytes() == before


def test_old_attempt_counts_continue_to_enforce_retry_limit(case):
    install(case)
    with budget(case) as current:
        for i in range(2):
            run = case.run/'new'/str(i); run.mkdir(parents=True)
            (run/'collector_stdout.log').write_text(LIFECYCLE)
            chain=current.reserve('repeat:1',run,True); assert chain
            current.collector_started(chain)
            current.finish(chain, {'ok': False}, run)
        assert not current.allowed('repeat:1')
        assert current.counts()['begun_chains'] == 4
        assert current.report({'ok': False}, case.run/'report')['error'] == 'task_logical_retry_limit'
    historical_unchanged(case)


def test_another_dead_unfinished_chain_is_not_covered(case):
    install(case)
    value=json.loads(case.path.read_text())
    value['tasks'][ROW]['chains'].append(dict(value['tasks'][ROW]['chains'][1],run_directory='/new-unknown'))
    write(case.path,value)
    with budget(case) as current:
        assert current.is_set()
        assert current.report({'ok': False}, case.run/'report')['error'] == 'campaign_source_interrupted_chain_unresolved'
    historical_unchanged(case)


def test_reporter_uses_current_refusal_and_keeps_old_capture_negative(case, monkeypatch):
    from scripts.run_native_producer_energy_from_summary import _campaign_budget_block_projection
    import shlex
    install(case)
    for key,value in case.env.items(): monkeypatch.setenv(key,value)
    command=shlex.join(['energy_measurement_cli.py','--campaign-budget-file',str(case.path),'--campaign-row-id',ROW])
    assert _campaign_budget_block_projection(command,case.run/'report') == {}
    with budget(case) as current: current.stop('task_cancelled')
    result=_campaign_budget_block_projection(command,case.run/'report')
    assert result['energy_not_started_reason'] == 'task_cancelled'
    assert result['execution_status'] == 'NOT_RUN'
    historical_unchanged(case)


def test_normal_local_software_diagnostic_finish_keeps_original_costs(case, tmp_path, monkeypatch):
    from test_v283_campaign_energy_budget import campaign_rig
    install(case)
    rig_root = tmp_path/'local-software-processes'; rig_root.mkdir()
    rig = campaign_rig(rig_root, monkeypatch, ['success'])
    # Existing local process fixtures emit normal marker/sample/preflight data;
    # no real collector binary or physical source is involved.
    diagnostic_limits = {'max_chains': 1, 'max_retries': 0, 'max_transport_failures': 20}
    run = rig_root/'diagnostic'/'run_000'
    with budget(case, row='diagnostic', limits=diagnostic_limits) as current:
        identity = current.reserve('repeat:0', run, True)
        result = rig.measure('diagnostic', task_budget_file=None, task_max_chains=None,
            task_max_transport_failures=None, run_count=1, exact_run_count=True,
            invalid_repeat_max_retries=0)
        assert result['ok'] and result['repeat_contract_complete']
        current.collector_started(identity)
        current.finish(identity, result['runs'][0], run)
        assert current.counts()['valid_logical_repeats'] == 1
        assert current.source['transport_failures'] == 3
        assert not current.is_set()
    assert rig.counts() == {'preflight': 1, 'collector': 1, 'workload': 1}
    with budget(case) as current:
        assert current.allowed('repeat:0') and current.allowed('repeat:1')
        assert current.counts()['begun_chains'] == 2
        assert current.counts()['valid_logical_repeats'] == 0
        assert [c['logical_repeat'] for c in current.data['chains']] == ['repeat:0','repeat:1']
    historical_unchanged(case)
