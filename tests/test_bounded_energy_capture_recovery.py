"""Recover real temporary locks using the external energy parent's old layout."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from test_physical_resource_recovery import recovery_case, _write, _fences, _retry_evidence
from onnx_splitpoint_tool.remote import resource_recovery


@pytest.fixture
def bounded_case(recovery_case):
    case = recovery_case
    journal = case.run_dir / 'remote-process-leases'
    shutil.move(str(case.journal), journal)
    case.journal = journal
    case.reply_path = journal / case.reply_path.name
    case.request_path = journal / case.request_path.name
    case.config_path = journal / case.config_path.name
    stamp = datetime.now(timezone.utc).isoformat()
    plan = case.run_dir / 'energy_parent_plan.json'
    _write(plan, {'schema':'yolov7-full-fix/bounded-energy-parent-v1','run_dir':str(case.run_dir)})
    _write(case.run_dir/'PREPARED.json', {'ok':True,'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()})
    _write(case.run_dir/'production-start.json', {'started_at':stamp,'run_id':case.run_dir.name,
        'session_id':case.kwargs['session_id'],'parent_pid':99999993})
    _write(case.run_dir/'production-result.json', {'finished_at':stamp,'returncode':130})
    _write(case.run_dir/'cleanup-status.json', {'at':stamp,'owned_remote_active':0,
        'owned_local_active':0,'foreign_processes_stopped':False,'cleanup_proven':False})
    case.kwargs.pop('operator_evidence')
    case.kwargs.update(bounded_energy_attempt_dir=case.run_dir,retry_authorization=_retry_evidence(case))
    # The external parent never had a workflow manifest. Do not invent one.
    (case.run_dir/'run_manifest.json').unlink()
    return case


def test_bounded_parent_recovery_preserves_terminal_evidence_and_budget(bounded_case):
    case = bounded_case
    paths = [case.run_dir/name for name in ('production-start.json','production-result.json',
        'cleanup-status.json','energy_task_budget.json','energy_parent_plan.json','PREPARED.json')]
    before = {p:p.read_bytes() for p in paths}
    assert resource_recovery.recover_capture_resources(**case.kwargs)['status']=='released'
    assert all(p.read_bytes()==data for p,data in before.items())
    assert json.loads(case.reply_path.read_text())['state']=='STOP'
    assert all(not lock.quarantine_path.exists() for lock in case.locks.values())
    assert resource_recovery.recover_capture_resources(**case.kwargs)['status']=='already_released'


@pytest.mark.parametrize('fault',['missing_result','live_cleanup','foreign_cleanup','session','pid','plan','time','symlink'])
def test_bounded_parent_refuses_unbound_recovery(bounded_case,fault):
    case = bounded_case
    if fault=='missing_result': (case.run_dir/'production-result.json').unlink()
    elif fault=='plan':
        _write(case.run_dir/'energy_parent_plan.json',{'run_dir':'/foreign'})
    elif fault=='symlink':
        p=case.run_dir/'production-result.json'; q=case.run_dir/'result-copy.json'
        p.rename(q);p.symlink_to(q)
    else:
        name='cleanup-status.json' if fault in ('live_cleanup','foreign_cleanup') else 'production-start.json'
        p=case.run_dir/name; data=json.loads(p.read_text())
        if fault=='live_cleanup':data['owned_remote_active']=1
        elif fault=='foreign_cleanup':data['foreign_processes_stopped']=True
        elif fault=='session':data['session_id']='foreign'
        elif fault=='pid':data['parent_pid']=99999994
        elif fault=='time':data['started_at']='2099-01-01T00:00:00+00:00'
        _write(p,data)
    before=_fences(case)
    with pytest.raises(Exception):resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case)==before


@pytest.fixture
def interrupted_native_case(bounded_case):
    from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
    c = bounded_case
    capture = (c.run_dir / 'measurement/measurements/native-task' / ('plan_' + 'c' * 32)
               / ('attempt_' + 'd' * 32) / 'repeat_retry_attempts/repeat_001/attempt_02/run_000')
    capture.parent.mkdir(parents=True)
    shutil.move(c.request['attempt_dir'], capture)
    (capture / 'collector_stdout.log.cleanup.json').unlink()
    c.request.update(attempt_dir=str(capture), phase='ACQUIRING', sequence=2)
    c.request.pop('process_started')
    _write(c.request_path, c.request)
    c.reply['reason'] = 'controller_resource_owner_lost'
    _write(c.reply_path, c.reply)
    for lock in c.locks.values():
        fence = json.loads(lock.quarantine_path.read_text())
        fence['owner'].update(reason=c.reply['reason'], resource_request=c.request)
        _write(lock.quarantine_path, fence)
    identity = ['deepx_to_trt', 'resnet50', 'b119', c.kwargs['setup_id']]
    row = json.dumps(identity, separators=(',', ':'))
    fields = dict(zip(('backend', 'model', 'case', 'setup_id'), identity))
    contract = dict(fields)
    digest = canonical_json_sha256(contract)
    contract['contract_sha256'] = digest
    command_path = c.run_dir / 'plancheck/plan/task.sh'
    command_path.parent.mkdir(parents=True)
    command_path.write_text('fixture original native command\n')
    contract_path = command_path.with_suffix('.command_contract.json')
    _write(contract_path, contract)
    summary_path = c.run_dir / 'inputs/native_summary.json'
    _write(summary_path, {'rows': [{**fields, 'native_command_contract_sha256': digest}]})
    _write(c.run_dir / 'plancheck/plan/native_producer_energy_plan.json', {'rows': [{**fields,
        'successful_command_contract_sha256': digest, 'command_contract_file': str(contract_path),
        'command_file': str(command_path),
        'command_contract_file_sha256': hashlib.sha256(contract_path.read_bytes()).hexdigest()}]})
    attestation = capture / 'preflight/preflight_attestation.json'
    _write(attestation, {'ok': True, 'nonce': 'original-native-nonce', 'command_contract_sha256': digest})
    _write(capture / 'preflight/preflight_evidence.json', {'ok': True, 'rc': 0,
        'expected_command_contract_sha256': digest, 'nonce': 'original-native-nonce',
        'attestation_file_sha256': hashlib.sha256(attestation.read_bytes()).hexdigest(),
        'workload_template_sha256': hashlib.sha256(command_path.read_text().strip().encode()).hexdigest()})
    _write(capture / 'energy_summary.json', {'collector_started': True,
        'collector_pid': c.request['collector_pid'], 'status': 'collector_running',
        'collector_cmd': ['urecs-data-collector', '-c=' + str(capture / 'energy_command.sh'),
                          '--address=192.0.2.176']})
    (capture / 'collector_stderr.log').write_text('receive socket error: locally cancelled\n')
    (capture / 'collector_stdout.log').write_text(
        'Fast Firmware GO sent monotonic_ns=1 local=192.0.2.1:4000 requested_device_us=86000000\n'
        'Fast Firmware first packet monotonic_ns=2 local=192.0.2.1:4000\n'
        'Running command: ' + str(capture / 'energy_command.sh') + '\n'
        'Fast Firmware source closed monotonic_ns=3 local=192.0.2.1:4000 verified=false '
        'reason=Err("collector cancelled locally; device completion unverified")\nWriter Flushed\n')
    (capture / 'workload_timing.txt').write_text('start_ns=1\n')
    plan_path = c.run_dir / 'energy_parent_plan.json'
    plan = json.loads(plan_path.read_text())
    budget = c.run_dir / 'energy_task_budget.json'
    plan.update(campaign_budget_file=str(budget), summary=str(summary_path),
        summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), capture_identities=[identity],
        targets={c.kwargs['setup_id']: {'energy': {'urecs_address': '192.0.2.176'}}})
    _write(plan_path, plan)
    _write(c.run_dir / 'PREPARED.json', {'ok': True,
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest()})
    started_path = c.run_dir / 'production-start.json'
    started = json.loads(started_path.read_text())
    started['argv'] = ['fixture-python', 'run_native_producer_energy_from_summary.py',
        '--summary', str(summary_path), '--out-dir', str(c.run_dir / 'measurement'),
        '--campaign-budget-file', str(budget)]
    _write(started_path, started)
    chain = {k: c.request[k] for k in ('owner_pid', 'owner_start_ticks')}
    chain.update(logical_repeat='repeat:1', run_directory=str(capture), preflight_requested=True,
                 collector_started=True, finished=False)
    valid = {**chain, 'logical_repeat': 'repeat:0', 'run_directory': str(capture.parent / 'valid'),
             'finished': True, 'valid': True, 'source_completion_verified': True, 'transport_reasons': []}
    _write(budget, {'campaign': True, 'max_transport_failures': 20,
        'tasks': {row: {'source_id': '192.0.2.176', 'stop_reason': '',
                       'limits': {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20},
                       'chains': [valid, chain]}},
        'sources': {'192.0.2.176': {'stop_reason': '', 'transport_failures': 0}}})
    c.capture, c.budget, c.row = capture, budget, row
    return c


def test_native_interrupted_recovery_uses_fresh_observation_preserves_old_cleanup(interrupted_native_case, monkeypatch):
    c = interrupted_native_case
    originals = {p: p.read_bytes() for p in c.run_dir.rglob('*') if p.is_file()}
    observations = []
    def observe(setup, **kwargs):
        observations.append(kwargs['local_owners'])
        return {'ok': True, 'controller': {}, 'remote': {}}
    monkeypatch.setattr(resource_recovery, 'observe_resource_owners', observe)
    assert resource_recovery.recover_capture_resources(**c.kwargs)['status'] == 'released'
    assert len(observations) == 1 and len(observations[0]) == 3
    assert all(p.read_bytes() == data for p, data in originals.items() if p != c.reply_path)
    assert not (c.capture / 'collector_stdout.log.cleanup.json').exists()
    reply = json.loads(c.reply_path.read_text())
    assert reply['state'] == 'STOP' and reply['reason'] == 'controller_resource_owner_lost'
    assert reply['recovery']['old_attempt_validated'] is False
    assert reply['recovery']['interrupted_capture']['source_completion_verified'] is False
    assert reply['recovery']['measurements_started'] == 0
    assert json.loads((c.run_dir / 'cleanup-status.json').read_text())['cleanup_proven'] is False
    assert resource_recovery.recover_capture_resources(**c.kwargs)['status'] == 'already_released'


def test_native_interruption_reconciles_negative_chain_once_without_releasing(interrupted_native_case):
    from onnx_splitpoint_tool.energy.recovery_accounting import reconcile_stopped_capture
    c = interrupted_native_case
    originals = {p: p.read_bytes() for p in c.run_dir.rglob('*') if p.is_file() and p != c.budget}
    before = json.loads(c.budget.read_text())
    fences = _fences(c)
    def reconcile():
        return reconcile_stopped_capture(c.budget, parent_attempt=c.run_dir,
            operation_id=c.request['operation_id'], run_dir=c.run_dir)
    receipt = reconcile()
    assert reconcile() == receipt
    assert _fences(c) == fences
    assert all(p.read_bytes() == data for p, data in originals.items())
    after = json.loads(c.budget.read_text())
    assert after['tasks'][c.row]['limits'] == before['tasks'][c.row]['limits']
    assert after['tasks'][c.row]['chains'][0] == before['tasks'][c.row]['chains'][0]
    chain = after['tasks'][c.row]['chains'][-1]
    assert chain['finished'] is True and chain['valid'] is False
    assert chain['source_completion_verified'] is False and chain['workload_started'] is True
    assert after['sources']['192.0.2.176']['transport_failures'] == 1
    assert receipt['original_chain'] == before['tasks'][c.row]['chains'][-1]
    assert resource_recovery.recover_capture_resources(**c.kwargs)['status'] == 'released'
    assert reconcile() == receipt


def test_native_recovery_replay_rejects_changed_original_log(interrupted_native_case):
    c = interrupted_native_case
    resource_recovery.recover_capture_resources(**c.kwargs)
    before = {lock.lock_path: lock.lock_path.read_bytes() for lock in c.locks.values()}
    before[c.reply_path] = c.reply_path.read_bytes()
    path = c.capture / 'collector_stdout.log'
    path.write_text(path.read_text() + 'later changed original log\n')
    with pytest.raises(Exception, match='recorded interrupted evidence changed'):
        resource_recovery.recover_capture_resources(**c.kwargs)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert all(not lock.quarantine_path.exists() for lock in c.locks.values())


def test_resumed_native_interruption_binds_separate_measurement_tree(interrupted_native_case):
    from onnx_splitpoint_tool.energy.recovery_accounting import reconcile_stopped_capture
    c = interrupted_native_case
    attempt_id = 'e' * 32
    attempt = c.run_dir / 'resume_attempts' / attempt_id
    attempt.mkdir(parents=True)
    for name in ('production-start.json', 'production-result.json', 'cleanup-status.json', 'remote-process-leases'):
        shutil.move(str(c.run_dir / name), attempt / name)
    old_capture = c.capture
    output = c.run_dir / 'measurement/resume_attempts' / attempt_id
    output.mkdir(parents=True)
    shutil.move(str(c.run_dir / 'measurement/measurements'), output / 'measurements')
    capture = output / old_capture.relative_to(c.run_dir / 'measurement')
    c.request.update(attempt_dir=str(capture))
    c.journal = attempt / 'remote-process-leases'
    c.request_path = c.journal / c.request_path.name
    c.reply_path = c.journal / c.reply_path.name
    c.config_path = c.journal / c.config_path.name
    _write(c.request_path, c.request)
    for lock in c.locks.values():
        fence = json.loads(lock.quarantine_path.read_text())
        fence['owner']['resource_request'] = c.request
        _write(lock.quarantine_path, fence)
    start_path = attempt / 'production-start.json'
    start = json.loads(start_path.read_text())
    start['argv'][start['argv'].index('--out-dir') + 1] = str(output)
    _write(start_path, start)
    summary_path = capture / 'energy_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['collector_cmd'] = [arg.replace(str(old_capture), str(capture)) for arg in summary['collector_cmd']]
    _write(summary_path, summary)
    stdout = capture / 'collector_stdout.log'
    stdout.write_text(stdout.read_text().replace(str(old_capture), str(capture)))
    budget = json.loads(c.budget.read_text())
    budget['tasks'][c.row]['chains'][-1]['run_directory'] = str(capture)
    _write(c.budget, budget)
    c.kwargs['bounded_energy_attempt_dir'] = attempt
    original = {p: p.read_bytes() for p in attempt.rglob('*') if p.is_file()}
    fences = _fences(c)
    receipt = reconcile_stopped_capture(c.budget, parent_attempt=attempt,
        operation_id=c.request['operation_id'], run_dir=c.run_dir)
    assert receipt['interrupted_capture']['measurement_root'] == str(output / 'measurements')
    assert receipt['booked_chain']['source_completion_verified'] is False
    assert all(p.read_bytes() == data for p, data in original.items()) and _fences(c) == fences
    assert resource_recovery.recover_capture_resources(**c.kwargs)['status'] == 'released'
    assert json.loads((attempt / 'cleanup-status.json').read_text())['cleanup_proven'] is False


@pytest.mark.parametrize('fault', ['foreign_chain', 'valid_chain', 'foreign_out', 'duplicate_out',
    'summary_drift', 'contract_identity', 'attestation', 'missing_flush', 'missing_first_packet',
    'source_verified', 'wrong_collector', 'active_owner', 'remote_descriptor'])
def test_native_interrupted_recovery_rejects_unproven_or_foreign_evidence(interrupted_native_case, monkeypatch, fault):
    c = interrupted_native_case
    if fault in {'foreign_chain', 'valid_chain'}:
        value = json.loads(c.budget.read_text()); chain = value['tasks'][c.row]['chains'][-1]
        chain.update({'owner_pid': 99999990} if fault == 'foreign_chain' else {'valid': True})
        _write(c.budget, value)
    elif fault in {'foreign_out', 'duplicate_out'}:
        path = c.run_dir / 'production-start.json'; value = json.loads(path.read_text())
        if fault == 'foreign_out': value['argv'][value['argv'].index('--out-dir') + 1] = '/foreign'
        else: value['argv'] += ['--out-dir=/foreign']
        _write(path, value)
    elif fault == 'summary_drift':
        (c.run_dir / 'inputs/native_summary.json').write_text('{}')
    elif fault == 'contract_identity':
        path = c.run_dir / 'plancheck/plan/task.command_contract.json'
        value = json.loads(path.read_text()); value['case'] = 'foreign'; _write(path, value)
    elif fault == 'attestation':
        path = c.capture / 'preflight/preflight_attestation.json'
        value = json.loads(path.read_text()); value['nonce'] = 'foreign'; _write(path, value)
    elif fault in {'missing_flush', 'missing_first_packet', 'source_verified'}:
        path = c.capture / 'collector_stdout.log'; value = path.read_text()
        if fault == 'missing_flush': value = value.replace('Writer Flushed', '')
        elif fault == 'missing_first_packet': value = value.replace('Fast Firmware first packet', 'missing')
        else: value = value.replace('verified=false', 'verified=true')
        path.write_text(value)
    elif fault == 'wrong_collector':
        path = c.capture / 'energy_summary.json'; value = json.loads(path.read_text())
        value['collector_pid'] = 99999990; _write(path, value)
    elif fault == 'active_owner':
        monkeypatch.setattr(resource_recovery, 'observe_resource_owners', lambda *a, **k: {'ok': False})
    else: _write(c.journal / 'unknown.remote-lease.json', {'foreign': True})
    originals = {p: p.read_bytes() for p in c.run_dir.rglob('*') if p.is_file()}
    fences = _fences(c)
    with pytest.raises(Exception): resource_recovery.recover_capture_resources(**c.kwargs)
    assert _fences(c) == fences
    assert all(p.read_bytes() == data for p, data in originals.items())


def test_cli_can_recover_bounded_native_without_resuming_campaign(interrupted_native_case, capsys):
    from onnx_splitpoint_tool.remote.process_lease import process_lease_cli_main
    c = interrupted_native_case; auth = c.kwargs['retry_authorization']
    argv = ['recover-capture', '--run-dir', str(c.run_dir), '--session-id', c.kwargs['session_id'],
        '--operation-id', c.kwargs['operation_id'], '--setup-id', c.kwargs['setup_id'],
        '--registry', str(c.kwargs['registry_path']), '--bounded-energy-attempt-dir', str(c.run_dir),
        '--source', auth['source'], '--operator', auth['operator'], '--authorized-at', auth['authorized_at'],
        '--authorization-reason', auth['reason'], '--new-workflow-profile', auth['new_workflow_profile'],
        '--authorize-new-attempt']
    assert process_lease_cli_main(argv) == 0
    assert json.loads(capsys.readouterr().out)['measurements_started'] == 0
