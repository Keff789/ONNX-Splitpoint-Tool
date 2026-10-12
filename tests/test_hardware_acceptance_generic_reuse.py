"""Reuse only fresh same-phase Generic processes; never reset capture credits."""
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shlex

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_generic as adapter
from onnx_splitpoint_tool.workflow import generic_completed_task as completed
from test_generic_completion_reporting import case, result, write_result
from test_hardware_acceptance_generic import _owned_run
from tests.test_generic_energy_generated_processes import generated_case


def binding(path):
    return {'path': str(path), 'sha256': completed.file_sha256(path)}


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    phase = tmp_path / 'phase'
    parent = phase / 'hardware_runs/resnet50__test_setup'
    origin, current = parent / 'origin', parent / 'energy_retry'
    origin.mkdir(parents=True)
    current.mkdir()
    primary = tmp_path / 'primary'
    primary.mkdir()
    target = {'id': 'test_setup', 'remote': {'host': '192.0.2.1', 'user': 'fixture', 'port': 22},
              'energy': {'urecs_address': '192.0.2.2'}}
    completed.write_json(primary / 'run_manifest.json', {'hardware_targets': [target]})
    row = case('b074', 'test_setup')
    row.update(row['identity'])
    row.update(original_run=str(primary), case_output_relative='test_setup/resnet50/b074',
               warmup_count=100, task_count=1000, repetitions=3,
               build_allowed=False, fallback_allowed=False)
    source = tmp_path / 'historical_plan.json'
    completed.write_json(source, {'rows': [row], 'original_run': str(primary)})
    registry = tmp_path / 'registry.json'
    registry.write_text('{}')
    monkeypatch.setattr(adapter, 'source_and_target', lambda *a: ({}, target, ['dut:fixture'], {}, {}))
    monkeypatch.setattr(completed, 'original_generic_environment', lambda r: {
        'python': '/usr/bin/python3', 'environment': {}, 'source': 'normal.log', 'original_line': 'normal'})
    def runtime(row, remote, execution_id):
        config = {k: copy.deepcopy(row[k]) for k in ('identity', 'input_binding', 'completion_contract',
            'measurement_boundary', 'expected_runtime_artifacts', 'functional_reference', 'task', 'original_run')}
        config.update(execution_id=execution_id, repetitions=3, warmup_count=100, task_count=1000,
            queue_depth=2, original_run_cfg={'native_split_quality_binding': remote + '/configs/binding.json',
                                           'streaming_impl': 'generic', 'throughput_queue_depth': 2})
        return config, []
    monkeypatch.setattr(completed, 'runtime_configuration', runtime)
    def bundle(destination):
        (destination / 'runner').mkdir(parents=True)
        (destination / 'runner/run_split_onnxruntime.py').write_text('# generated normal runner\n')
        (destination / 'splitpoint_runners').mkdir()
        (destination / 'splitpoint_runners/generic_energy.py').write_text('# normal energy verifier\n')
        path = destination.with_suffix('.tar.gz')
        path.write_bytes(b'normal generated runtime bundle')
        return path
    monkeypatch.setattr(completed, 'prepare_runner_bundle', bundle)
    cell = {'model_id': 'resnet50', 'setup_id': 'test_setup', 'case_id': 'b074',
            'runtime_variants': ['generic_completed'], 'generic_case_key': row['case_key'],
            'generic_source_plan': binding(source)}
    matrix = phase / 'FINITE_MATRIX.json'
    now = datetime.now(timezone.utc)
    completed.write_json(matrix, {'recorded_before_hardware_start': (now - timedelta(seconds=10)).isoformat(),
                                  'cells': [cell]})
    profile = {'hardware_acceptance': {'enabled': True, 'schema': 'onnx-splitpoint/hardware-acceptance/v1',
        'matrix_evidence': str(matrix), 'cells': [cell]},
        'native_producers': {'warmup': 100, 'frames': 1000, 'repetitions': 3,
            'energy': {'duration_s': 60, 'physical_scope': 'FS', 'window_label': 'command',
                       'task_budget': {'enabled': True, 'max_retries': 2, 'max_transport_failures': 20}}}}
    context = {'run_dir': str(origin), 'source_primary_run': str(primary), 'hardware_registry_path': str(registry),
               'controller_root': str(tmp_path), 'profile': profile, 'execute_allowed': True, 'log': lambda *a: None}
    old_prepared = adapter.prepare_generic(cell, context)
    old_output = Path(old_prepared['output_root'])
    plan_path = Path(old_prepared['execution_plan'])
    plan = completed.read_json(plan_path)
    old_row = plan['rows'][0]
    payload = result(old_row['runtime_config'])
    payload.update(execution_id=plan['execution_id'], process_repetition_policy='separate_processes')
    source_map = {s['path']: s['sha256'] for s in old_row['runtime_config']['runtime_source_files']}
    for index, rep in enumerate(payload['repetitions'], 1):
        rep.update(execution_id=plan['execution_id'], runtime_instance_id=f'actual-runtime-{index}',
                   process_start_ticks=10000 + index, boot_id='same-boot',
                   execution_identity={'python_executable': '/usr/bin/python3', 'environment': {}},
                   runtime_source_binding={'before': source_map, 'after': source_map, 'unchanged': True})
    raw_path = Path(old_row['result_path'])
    write_result(raw_path, payload)
    attempts = []
    for index in (1, 2, 3):
        cfg = str(Path(old_row['runtime_config']['original_run_cfg']['native_split_quality_binding']).parent / 'generic_completion_config.json')
        command = 'cd -- ' + shlex.quote(old_row['remote_root']) + ' && ' + shlex.join([
            old_row['remote_python_executable'], '-B', '-u', old_row['remote_root'] + '/runner/run_split_onnxruntime.py',
            '--generic-completion-config', cfg, '--generic-completion-repetition', str(index),
            '--out-dir', old_row['remote_root'] + '/results/' + old_row['case_output_relative']])
        attempts.append({'repetition_index': index, 'command': command, 'returncode': 0,
                         'download': {'result_collected': True}})
    dispatch = raw_path.parent / 'dispatch_attempts.json'
    completed.write_json(dispatch, {'attempts': attempts})
    from onnx_splitpoint_tool.generic_completion_reporting import write_completion_reports
    write_completion_reports(plan, old_output / 'results', old_output / 'reports')
    for directory in (origin, current):
        completed.write_json(directory / 'profile.yaml', profile)
        completed.write_json(directory / 'run_manifest.json', {'schema': 'onnx-splitpoint/evaluation-run-manifest',
            'run_id': directory.name, 'status': 'partial' if directory == origin else 'running',
            'created_at': now.isoformat(), 'hardware_acceptance': {
                'schema': 'onnx-splitpoint/hardware-acceptance/v1', 'source_campaign_resumed': False}})
    closure = origin / 'reports/artifact_index_closure.json'
    completed.write_json(closure, {'schema': 'onnx-splitpoint/artifact-index-terminal-closure',
        'status': 'pass', 'run_id': origin.name, 'workflow_status': 'partial'})
    context['run_dir'] = str(current)
    row_key = adapter._campaign_key(row['identity'])
    chain = {'logical_repeat': 'repeat:0', 'run_directory': str(old_output / 'measurement/capture_1/run_000'),
             'owner_pid': 12345, 'owner_start_ticks': '100', 'collector_started': True,
             'finished': True, 'valid': False, 'source_completion_verified': False,
             'transport_reasons': ['transport_log_error']}
    budget = origin / 'energy/energy_task_budget.json'
    completed.write_json(budget, {'campaign': True, 'max_transport_failures': 20,
        'sources': {'192.0.2.2': {'stop_reason': 'campaign_source_completion_unresolved'}},
        'tasks': {row_key: {'limits': {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20},
            'source_id': '192.0.2.2', 'chains': [chain], 'stop_reason': 'campaign_source_completion_unresolved'}}})
    request_path = origin / 'jobs/remote_process_leases/session/capture-op.resource-request.json'
    request = {'run_id': origin.name, 'session_id': 'old-session', 'operation_id': 'capture-op',
        'token': 'a' * 32, 'purpose': 'capture', 'phase': 'FINALIZING', 'process_started': True,
        'setup_id': 'test_setup', 'attempt_dir': chain['run_directory'],
        'owner_pid': chain['owner_pid'], 'owner_start_ticks': chain['owner_start_ticks']}
    completed.write_json(request_path, request)
    reply_path = request_path.with_name('capture-op.resource-reply.json')
    from onnx_splitpoint_tool.workflow.execution_binding import physical_dut_key
    recovery = {k: request[k] for k in ('run_id', 'session_id', 'operation_id')}
    recovery.update(status='released', measurements_started=0, old_attempt_validated=False,
        ownership_observation={'ok': True}, resources=[physical_dut_key(target), 'source:192.0.2.2',
                                                      'controller:capture', 'controller:nic'])
    reply = {k: request[k] for k in ('run_id', 'session_id', 'operation_id', 'token')}
    reply.update(state='STOP', reason='campaign_source_completion_unresolved', recovery=recovery)
    completed.write_json(reply_path, reply)
    reference = {name: binding(path) for name, path in {
        'prepared': old_output / 'PREPARED.json', 'execution_plan': plan_path, 'performance_result': raw_path,
        'summary': old_output / 'reports/generic_completion_summary.json', 'dispatch_attempts': dispatch,
        'runner_bundle': Path(old_prepared['runner_bundle']), 'run_manifest': origin / 'run_manifest.json',
        'closure': closure, 'profile': origin / 'profile.yaml'}.items()}
    reference['repeats'] = [{k: rep[k] for k in ('repetition_index', 'runtime_instance_id', 'pid',
                                               'process_start_ticks', 'boot_id')} for rep in payload['repetitions']]
    reference['reuse_scope'] = {'prior_budget': binding(budget), 'prior_capture_chains': [
        {k: chain[k] for k in ('logical_repeat', 'run_directory', 'owner_pid', 'owner_start_ticks')}],
        'source_recovery': {'request': binding(request_path), 'reply': binding(reply_path)}}
    cell['fresh_performance_reference'] = reference
    prepared = adapter.prepare_generic(cell, context)
    return prepared, context, cell, reference, old_output, chain


def test_reference_revalidates_actual_processes_and_retains_raw_bytes(cohort):
    prepared, context, _, reference, output, _ = cohort
    original = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
    retained = adapter.collect_generic_performance_case(prepared, context, reference)
    assert retained['projection']['valid_repetition_count'] == 3
    assert retained['proof']['new_performance_processes'] == 0
    assert retained['proof']['retained_fresh_performance_processes'] == 3
    assert prepared['performance_processes'] == 0
    assert not Path(prepared['output_root'], 'results/test_setup/resnet50/b074/generic_completion_results.json').exists()
    assert all(p.read_bytes() == payload for p, payload in original.items())


@pytest.mark.parametrize('change', ['raw_hash', 'output_sample', 'runtime', 'dispatch', 'source_closure',
                                  'current_code', 'unclosed', 'old_phase', 'profile', 'prior_budget',
                                  'chain_identity', 'missing_recovery', 'invalid_recovery'])
def test_invalid_or_historical_reference_cannot_authorize_reuse(cohort, change):
    prepared, context, _, reference, output, _ = cohort
    if change == 'raw_hash':
        Path(reference['performance_result']['path']).write_text('{}')
    elif change == 'output_sample':
        next((output / 'results').rglob('output.npy')).write_bytes(b'changed')
    elif change in {'runtime', 'source_closure'}:
        path = Path(reference['performance_result']['path'])
        raw = completed.read_json(path)
        if change == 'runtime':
            raw['repetitions'][1]['process_start_ticks'] = raw['repetitions'][0]['process_start_ticks']
            raw['repetitions'][1]['pid'] = raw['repetitions'][0]['pid']
        else:
            raw['repetitions'][0]['runtime_source_binding']['unchanged'] = False
        completed.write_json(path, raw)
        reference['performance_result'] = binding(path)
    elif change == 'dispatch':
        path = Path(reference['dispatch_attempts']['path'])
        data = completed.read_json(path)
        data['attempts'][0]['command'] += ' --build'
        completed.write_json(path, data)
        reference['dispatch_attempts'] = binding(path)
    elif change == 'current_code':
        Path(prepared['output_root'], 'runner-bundle/runner/run_split_onnxruntime.py').write_text('changed')
    elif change in {'unclosed', 'old_phase'}:
        path = Path(reference['run_manifest']['path'])
        data = completed.read_json(path)
        if change == 'unclosed':
            data['status'] = 'running'
        else:
            data['created_at'] = '2000-01-01T00:00:00+00:00'
        completed.write_json(path, data)
        reference['run_manifest'] = binding(path)
    elif change == 'profile':
        context['profile']['native_producers']['warmup'] = 0
    elif change == 'prior_budget':
        Path(reference['reuse_scope']['prior_budget']['path']).write_text('{}')
    elif change == 'chain_identity':
        reference['reuse_scope']['prior_capture_chains'][0]['owner_pid'] += 1
    elif change == 'missing_recovery':
        reference['reuse_scope'].pop('source_recovery')
    elif change == 'invalid_recovery':
        recovery = reference['reuse_scope']['source_recovery']
        path = Path(recovery['reply']['path'])
        data = completed.read_json(path)
        data['recovery']['old_attempt_validated'] = True
        completed.write_json(path, data)
        recovery['reply'] = binding(path)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.collect_generic_performance_case(prepared, context, reference)


def _energy_plan(prepared, context):
    row = adapter.collect_generic_performance_case(prepared, context, prepared['fresh_performance_reference'])['case']
    budget = adapter._budget(prepared, row, context)
    return {'identity': row['identity'], 'campaign_budget_file': str(budget),
        'campaign_row_id': adapter._campaign_key(row['identity']), 'campaign_source_id': '192.0.2.2',
        'campaign_limits': {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20}}


def test_old_failed_start_spends_slot_credit_without_rewriting_old_budget(cohort):
    prepared, context, _, reference, _, _ = cohort
    old_budget = Path(reference['reuse_scope']['prior_budget']['path'])
    original = old_budget.read_bytes()
    plan = _energy_plan(prepared, context)
    admitted = adapter._phase_energy_admission(prepared, context, plan, {'repetition_index': 1})
    assert admitted['prior_and_current_chains'] == {'repeat:0': 1, 'repeat:1': 0, 'repeat:2': 0}
    assert admitted['remaining_single_starts_for_slot'] == 2
    assert admitted['remaining_single_starts_for_case'] == 8
    assert admitted['normal_single_internal_retries'] == 0
    assert admitted['profile_max_retries'] == 2
    current = Path(plan['campaign_budget_file'])
    budget = completed.read_json(current)
    budget['tasks'][plan['campaign_row_id']]['chains'] = [
        {'logical_repeat': 'repeat:0', 'finished': True, 'valid': False, 'source_completion_verified': True,
         'run_directory': str(Path(context['run_dir']) / f'measurement/attempt_{index}/run_000')}
        for index in (1, 2)]
    completed.write_json(current, budget)
    with pytest.raises(RuntimeError, match='cumulative phase chain/retry limit'):
        adapter._phase_energy_admission(prepared, context, plan, {'repetition_index': 1})
    assert old_budget.read_bytes() == original


def test_prior_accepted_energy_slot_rejects_performance_only_continuation(cohort):
    prepared, context, _, reference, _, _ = cohort
    path = Path(reference['reuse_scope']['prior_budget']['path'])
    budget = completed.read_json(path)
    chain = next(iter(budget['tasks'].values()))['chains'][0]
    chain.update(valid=True, source_completion_verified=True)
    completed.write_json(path, budget)
    reference['reuse_scope']['prior_budget'] = binding(path)
    with pytest.raises(ValueError, match='prior accepted energy slot'):
        adapter.collect_generic_performance_case(prepared, context, reference)
    plan = {'campaign_row_id': adapter._campaign_key(prepared['selected_identity']),
            'campaign_source_id': '192.0.2.2',
            'campaign_limits': {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20}}
    prepared['fresh_performance_reference']['reuse_scope']['prior_budget'] = binding(path)
    with pytest.raises(RuntimeError, match='accepted energy slot'):
        adapter._phase_energy_admission(prepared, context, plan, {'repetition_index': 1})


@pytest.mark.parametrize('state', ['accepted', 'unresolved', 'all_credit_spent'])
def test_current_phase_admission_preserves_valid_slots_and_source_and_case_limits(cohort, state):
    prepared, context, _, _, _, _ = cohort
    plan = _energy_plan(prepared, context)
    path = Path(plan['campaign_budget_file'])
    budget = completed.read_json(path)
    chains = [{'logical_repeat': 'repeat:0', 'finished': True, 'valid': state == 'accepted',
               'source_completion_verified': state != 'unresolved',
               'run_directory': str(Path(context['run_dir']) / 'measurement/attempt_1/run_000')}]
    if state == 'all_credit_spent':
        chains = [{'logical_repeat': f'repeat:{slot}', 'finished': True, 'valid': False,
                   'source_completion_verified': True,
                   'run_directory': str(Path(context['run_dir']) / f'measurement/attempt_{slot}_{index}/run_000')}
                  for slot, count in ((0, 2), (1, 3), (2, 3)) for index in range(count)]
    budget['tasks'][plan['campaign_row_id']]['chains'] = chains
    completed.write_json(path, budget)
    message = {'accepted': 'accepted energy slot', 'unresolved': 'source completion is unresolved',
               'all_credit_spent': 'cumulative phase chain/retry limit'}[state]
    with pytest.raises(RuntimeError, match=message):
        adapter._phase_energy_admission(prepared, context, plan, {'repetition_index': 1})


def test_normal_energy_callback_refusal_precedes_capture_receipt_and_process_start(
        generated_case, tmp_path, monkeypatch):
    """The phase guard runs at the actual ordinary Single admission boundary."""
    from onnx_splitpoint_tool import native_progress
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseJournal
    from onnx_splitpoint_tool.runners.generic_energy import canonical_hash
    plan, _, case = copy.deepcopy(generated_case)
    for call in plan['calls']:
        argv = call['argv']
        argv[argv.index('--out') + 1] = str(tmp_path / f"capture_{call['repetition_index']}")
        call['command_sha256'] = canonical_hash(argv)
    path = tmp_path / 'generic_energy_plan.json'
    completed.write_json(path, plan)
    monkeypatch.setenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES', 'required')
    monkeypatch.setattr(RemoteProcessLeaseJournal, 'from_environment', classmethod(lambda cls, **k: object()))
    monkeypatch.setattr(completed, 'check_energy_campaign', lambda *a, **k: {})
    staged = []
    monkeypatch.setattr(completed, 'stage_energy_case', lambda *a: staged.append(True))
    monkeypatch.setattr(native_progress, 'run_streaming', lambda *a, **k: pytest.fail('No start after refusal'))
    calls = []
    def refuse(normal_plan, call):
        calls.append(call['repetition_index'])
        assert normal_plan == plan
        assert not (tmp_path / 'capture_1_started.json').exists()
        raise RuntimeError('cumulative phase credit exhausted')
    with pytest.raises(RuntimeError, match='phase credit exhausted'):
        completed.run_energy_case(path, case, None, before_capture=refuse)
    assert calls == [1] and staged == [True]
    assert not (tmp_path / 'capture_1_started.json').exists()
    assert not (tmp_path / 'capture_1').exists()
    assert completed.read_json(tmp_path / 'controller_cleanup.json')['owned_local_remaining'] == 0


@pytest.mark.parametrize('energy_fails', [False, True])
def test_energy_only_continuation_has_zero_new_performance_dispatches(cohort, monkeypatch, energy_fails):
    prepared, context, _, _, old_output, _ = cohort
    value, context = _owned_run((prepared, context, None, None), monkeypatch)
    original = {p: p.read_bytes() for p in old_output.rglob('*') if p.is_file()}
    monkeypatch.setattr(completed, 'run_case', lambda *a, **k: pytest.fail('Retained performance must not run again'))
    monkeypatch.setattr(completed, 'stage_case', lambda *a, **k: 'normal-existing-config')
    def prepare_energy(row, **kwargs):
        assert Path(row['result_path']).is_relative_to(old_output)
        plan = {'identity': row['identity'], 'campaign_budget_file': str(kwargs['campaign_budget_file']),
                'campaign_row_id': adapter._campaign_key(row['identity']), 'campaign_source_id': '192.0.2.2',
                'campaign_limits': {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20}}
        completed.write_json(kwargs['output'] / 'generic_energy_plan.json', plan)
        return plan
    def run_energy(path, row, transport, **kwargs):
        if energy_fails:
            raise RuntimeError('controlled unstarted energy failure')
        plan = completed.read_json(path)
        budget = Path(plan['campaign_budget_file'])
        for index in (1, 2, 3):
            kwargs['before_capture'](plan, {'repetition_index': index})
            completed.write_json(path.parent / f'capture_{index}_started.json', {'cli_attempt': True})
            data = completed.read_json(budget)
            data['tasks'][plan['campaign_row_id']]['chains'].append({'logical_repeat': f'repeat:{index - 1}',
                'collector_started': True, 'finished': True, 'valid': True, 'source_completion_verified': True,
                'run_directory': str(path.parent / f'capture_{index}/run_000')})
            completed.write_json(budget, data)
        energy = {'status': 'completed', 'normal_report_verified': True, 'valid_repetitions': 3,
                  'energy_efficiency_claim_eligible': False}
        completed.write_json(path.parent / 'reports/generic_completed_task_energy.json', energy)
        return energy
    monkeypatch.setattr(completed, 'prepare_energy_case', prepare_energy)
    monkeypatch.setattr(completed, 'run_energy_case', run_energy)
    outcome = adapter.run_generic(value, context)
    assert outcome['new_performance_process_count'] == outcome['new_performance_dispatch_attempt_count'] == 0
    assert outcome['retained_fresh_performance_processes'] == outcome['valid_repeats']['performance'] == 3
    assert outcome['previous_energy_attempts']['failed'] == outcome['previous_energy_attempts']['physical_collector_starts'] == 1
    assert outcome['new_capture_count'] == (0 if energy_fails else 3)
    assert outcome['fresh_hardware_executed'] is (not energy_fails)
    assert outcome['old_performance_raw_copied'] is False
    assert outcome['reused_current_phase_performance'] is True
    assert len(outcome['phase_energy_admissions']) == (0 if energy_fails else 3)
    if not energy_fails:
        assert outcome['phase_energy_admissions'][0]['remaining_single_starts_for_case'] == 8
        assert outcome['phase_energy_admissions'][0]['remaining_single_starts_for_slot'] == 2
        assert outcome['phase_energy_admissions'][-1]['prior_and_current_chains'] == {
            'repeat:0': 2, 'repeat:1': 1, 'repeat:2': 0}
    assert all(p.read_bytes() == payload for p, payload in original.items())
