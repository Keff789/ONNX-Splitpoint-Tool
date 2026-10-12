"""Finite Generic acceptance must create new normal evidence and retain failures."""
import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_generic as adapter
from onnx_splitpoint_tool.workflow import generic_completed_task as completed
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseJournal, RemoteProcessLeaseScope
from onnx_splitpoint_tool.process_control import ProcessTreeRegistry
from test_generic_completion_reporting import case, result, write_result


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    primary = tmp_path / 'primary'
    primary.mkdir()
    row = case('b074', 'test_setup')
    row.update(row['identity'])
    row.update(original_run=str(primary), case_output_relative='test_setup/resnet50/b074',
               warmup_count=100, task_count=1000, repetitions=3,
               build_allowed=False, fallback_allowed=False,
               result_path=str(tmp_path / 'historical' / 'generic_completion_results.json'),
               remote_root='/old/session', runtime_config={'execution_id': 'old'})
    source = tmp_path / 'historical_plan.json'
    completed.write_json(source, {'rows': [row], 'original_run': str(primary)})
    target = {'id': 'test_setup', 'remote': {'host': '192.0.2.1', 'user': 'fixture', 'port': 22},
              'energy': {'urecs_address': '192.0.2.2'}}
    completed.write_json(primary / 'run_manifest.json', {'hardware_targets': [target]})
    registry = tmp_path / 'registry.json'
    registry.write_text('{}')
    monkeypatch.setattr(adapter, 'source_and_target', lambda *a: ({}, target, ['dut:fixture'], {}, {}))
    monkeypatch.setattr(completed, 'original_generic_environment', lambda r: {
        'python': '/usr/bin/python3', 'environment': {}, 'source': 'normal.log', 'original_line': 'normal'})
    def runtime(row, remote_root, execution_id):
        config = copy.deepcopy(row)
        config['execution_id'] = execution_id
        return config, []
    monkeypatch.setattr(completed, 'runtime_configuration', runtime)
    def bundle(destination):
        destination.mkdir()
        path = destination.with_suffix('.tar.gz')
        path.write_bytes(b'normal generated runtime')
        return path
    monkeypatch.setattr(completed, 'prepare_runner_bundle', bundle)
    monkeypatch.setattr(completed, 'bind_runtime_source_closure', lambda row, root: row)
    cell = {'model_id': 'resnet50', 'case_id': 'b074', 'setup_id': 'test_setup',
            'generic_case_key': row['case_key'], 'generic_source_plan': {
                'path': str(source), 'sha256': completed.file_sha256(source)}}
    context = {'run_dir': str(tmp_path / 'acceptance'), 'source_primary_run': str(primary),
               'hardware_registry_path': str(registry), 'controller_root': str(tmp_path),
               'execute_allowed': True, 'log': lambda *a: None}
    value = adapter.prepare_generic(cell, context)
    return value, context, cell, row


def test_prepare_uses_one_fresh_normal_execution_without_historical_results(prepared):
    value, context, cell, historical = prepared
    plan = completed.read_json(Path(value['execution_plan']))
    row = plan['rows'][0]
    assert plan['authorized_row_count'] == len(plan['rows']) == 1
    assert plan['source_execution_plans'] == [cell['generic_source_plan']]
    assert row['runtime_config']['execution_id'] != 'old'
    assert row['remote_root'] != historical['remote_root']
    assert Path(row['result_path']).is_relative_to(Path(context['run_dir']))
    assert not Path(row['result_path']).exists()
    assert value['hardware_executed'] is False
    assert value['performance_processes'] == value['energy_captures'] == 3
    assert value['historical_result_reference'] == historical['result_path']
    assert not (Path(context['run_dir']) / 'energy').exists()


def test_selection_sha_and_ambiguity_fail_before_output(prepared, tmp_path):
    _, context, cell, row = prepared
    context['run_dir'] = str(tmp_path / 'other_acceptance')
    source = Path(cell['generic_source_plan']['path'])
    completed.write_json(source, {'rows': [row, row]})
    with pytest.raises(ValueError, match='source plan changed'):
        adapter.prepare_generic(cell, context)
    cell['generic_source_plan']['sha256'] = completed.file_sha256(source)
    with pytest.raises(ValueError, match='Exactly one'):
        adapter.prepare_generic(cell, context)
    assert not Path(context['run_dir']).exists()


def test_saved_supplement_run_binding_preserves_its_real_original_root(prepared, tmp_path):
    _, context, cell, row = prepared
    configured = Path(context['source_primary_run'])
    supplement = configured / '_supplements' / 'selected_run'
    supplement.mkdir(parents=True)
    (supplement / 'run_manifest.json').write_bytes((configured / 'run_manifest.json').read_bytes())
    row['original_run'] = str(supplement)
    row['historical_original_run'] = '/historical/supplement'
    source = Path(cell['generic_source_plan']['path'])
    plan = {'original_run': str(configured), 'rows': [row], 'source_run_bindings': [{
        'historical_root': row['historical_original_run'], 'archive_root': str(supplement),
        'run_manifest_sha256': completed.file_sha256(supplement / 'run_manifest.json')}]}
    completed.write_json(source, plan)
    cell['generic_source_plan']['sha256'] = completed.file_sha256(source)
    context['run_dir'] = str(tmp_path / 'supplement_acceptance')
    value = adapter.prepare_generic(cell, context)
    assert value['configured_primary_run'] == str(configured)
    assert value['selected_original_run'] == str(supplement)
    actual = completed.read_json(Path(value['execution_plan']))
    assert actual['original_run'] == actual['rows'][0]['runtime_config']['original_run'] == str(supplement)
    plan['source_run_bindings'][0]['run_manifest_sha256'] = 'invalid'
    completed.write_json(source, plan)
    cell['generic_source_plan']['sha256'] = completed.file_sha256(source)
    context['run_dir'] = str(tmp_path / 'bad_supplement_acceptance')
    with pytest.raises(ValueError, match='supplement run binding'):
        adapter.prepare_generic(cell, context)
    assert not Path(context['run_dir']).exists()


def test_only_selected_new_budget_is_initialized_and_old_budget_is_untouched(prepared, tmp_path):
    value, context, _, _ = prepared
    old = tmp_path / 'old_campaign' / 'energy' / 'energy_task_budget.json'
    completed.write_json(old, {'original_scientific_state': True})
    original = old.read_bytes()
    row = completed.read_json(Path(value['execution_plan']))['rows'][0]
    path = adapter._budget(value, row, context)
    budget = completed.read_json(path)
    assert len(budget['tasks']) == 1
    task = next(iter(budget['tasks'].values()))
    assert task['chains'] == []
    assert task['limits'] == {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20}
    # The controller can initialize this selected row before the adapter admits it.
    assert adapter._budget(value, row, context) == path
    assert completed.read_json(path) == budget
    assert old.read_bytes() == original
    context['budget_file'] = str(old)
    with pytest.raises(ValueError, match='new finite acceptance budget'):
        adapter._budget(value, row, context)
    assert old.read_bytes() == original


def test_profile_and_owner_checks_precede_dispatch(prepared, monkeypatch):
    value, context, _, _ = prepared
    monkeypatch.setattr(completed, 'transport_for', lambda *a, **k: pytest.fail('No transport without owner'))
    context['expected_duration_s'] = 30
    with pytest.raises(ValueError, match='regular'):
        adapter.run_generic(value, context)
    context['expected_duration_s'] = 60
    monkeypatch.setattr(RemoteProcessLeaseJournal, 'from_environment', classmethod(lambda cls, **k: SimpleNamespace(scope='scope')))
    monkeypatch.delenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES', raising=False)
    with pytest.raises(RuntimeError, match='resource broker'):
        adapter.run_generic(value, context)
    assert not (Path(value['output_root']) / 'production-start.json').exists()


def _owned_run(prepared, monkeypatch):
    value, context, _, _ = prepared
    scope = RemoteProcessLeaseScope('acceptance', 'session')
    context['remote_process_registry'] = SimpleNamespace(configured_scope=scope)
    context['process_registry'] = ProcessTreeRegistry()
    monkeypatch.setenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES', 'required')
    monkeypatch.setattr(RemoteProcessLeaseJournal, 'from_environment', classmethod(lambda cls, **k: SimpleNamespace(scope=scope)))
    monkeypatch.setattr(completed, 'transport_for', lambda *a, **k: SimpleNamespace(_ssh_base=lambda: ['ssh', 'fixture']))
    monkeypatch.setattr(adapter, 'ResourceTransport', lambda transport, **k: transport)
    monkeypatch.setattr(completed, 'observe_owners', lambda *a: {})
    monkeypatch.setattr(completed, 'capture_configuration', lambda *a, **k: {})
    monkeypatch.setattr(completed, 'verify_runtime_environment', lambda *a: {})
    monkeypatch.setattr(completed, 'configuration_comparison', lambda *a: {'status': 'observed_no_known_configuration_conflict'})
    monkeypatch.setattr(completed, 'upload_new_file', lambda *a, **k: {})
    monkeypatch.setattr(completed, 'checked_remote', lambda *a, **k: None)
    return value, context


def test_failed_new_process_preserves_valid_repeat_and_does_not_start_energy(prepared, monkeypatch):
    value, context = _owned_run(prepared, monkeypatch)
    def run_case(row, *a, **k):
        output = Path(row['result_path'])
        payload = result(row['runtime_config'], 1)
        write_result(output, payload)
        completed.write_json(output.parent / 'dispatch_attempts.json', {'attempts': [
            {'repetition_index': 1, 'returncode': 0}, {'repetition_index': 2, 'returncode': 1}]})
        row['dispatch_attempted'] = True
        raise RuntimeError('controlled normal process failure')
    monkeypatch.setattr(completed, 'run_case', run_case)
    monkeypatch.setattr(completed, 'prepare_energy_case', lambda *a, **k: pytest.fail('No energy after performance failure'))
    outcome = adapter.run_generic(value, context)
    assert outcome['status'] == 'failed'
    assert outcome['fresh_hardware_executed'] is True
    assert outcome['new_capture_count'] == 0
    assert outcome['new_performance_process_count'] == 1
    assert outcome['new_performance_dispatch_attempt_count'] == 2
    assert outcome['valid_repeats'] == {'performance': 1, 'energy': 0}
    assert outcome['failed_attempts'] == {'performance': 1, 'energy': 0}
    assert Path(value['output_root'], 'generic_result.json').is_file()


def test_normal_import_with_claim_exclusion_is_technical_success(prepared, monkeypatch):
    value, context = _owned_run(prepared, monkeypatch)
    def run_case(row, *a, **k):
        path = Path(row['result_path'])
        write_result(path, result(row['runtime_config']))
        completed.write_json(path.parent / 'dispatch_attempts.json', {'attempts': [
            {'repetition_index': index, 'returncode': 0} for index in (1, 2, 3)]})
        row['dispatch_attempted'] = True
    def prepare_energy(row, **kwargs):
        path = kwargs['output'] / 'generic_energy_plan.json'
        budget = completed.read_json(kwargs['campaign_budget_file'])
        completed.write_json(path, {'campaign_budget_file': str(kwargs['campaign_budget_file']),
                                   'campaign_row_id': next(iter(budget['tasks']))})
        assert kwargs['duration_s'] == 60.
        assert kwargs['require_environment_observation'] is True
        assert Path(kwargs['campaign_budget_file']).is_relative_to(Path(context['run_dir']))
        return {}
    def run_energy(path, row, transport, **kwargs):
        assert kwargs['process_registry'] is context['process_registry']
        for index in (1, 2, 3):
            completed.write_json(path.parent / f'capture_{index}_started.json', {'real_start': True})
            completed.write_json(path.parent / f'capture_{index}_process_receipt.json', {'returncode': 0})
        normal = completed.read_json(path)
        budget = completed.read_json(Path(normal['campaign_budget_file']))
        budget['tasks'][normal['campaign_row_id']]['chains'] = [
            {'collector_started': True, 'finished': True, 'valid': True,
             'run_directory': str(path.parent / f'capture_{index}' / 'run_000')}
            for index in (1, 2, 3)]
        completed.write_json(Path(normal['campaign_budget_file']), budget)
        return {'status': 'completed', 'normal_report_verified': True,
                'valid_repetitions': 3, 'energy_efficiency_claim_eligible': False}
    monkeypatch.setattr(completed, 'run_case', run_case)
    monkeypatch.setattr(completed, 'prepare_energy_case', prepare_energy)
    monkeypatch.setattr(completed, 'run_energy_case', run_energy)
    outcome = adapter.run_generic(value, context)
    assert outcome['status'] == 'completed'
    assert outcome['new_capture_count'] == 3
    assert outcome['valid_repeats'] == {'performance': 3, 'energy': 3}
    assert outcome['energy']['energy_efficiency_claim_eligible'] is False
    assert outcome['import_projection']['normal_report_verified'] is True
    # Its original start stays exclusive even after a successful import.
    with pytest.raises(ValueError, match='execution plan changed'):
        adapter.run_generic(value, context)


def test_normal_energy_collector_uses_gui_owner_and_cancellation(tmp_path, monkeypatch):
    """The normal energy API must never create an unowned collector registry."""
    from onnx_splitpoint_tool import native_progress
    from onnx_splitpoint_tool.runners import generic_energy
    from onnx_splitpoint_tool.energy.task_budget import EnergyTaskBudget
    registry = ProcessTreeRegistry()
    cancel = threading.Event()
    identity = {'model_id': 'resnet50', 'setup_id': 'test_setup', 'case_id': 'b074',
                'comparison_backend': 'hailo8'}
    budget = tmp_path / 'budget.json'
    limits = {'max_chains': 9, 'max_retries': 2, 'max_transport_failures': 20}
    with budget.open('a+') as stream:
        EnergyTaskBudget(stream, limits, campaign_row='selected', source_id='fixture')
    calls = []
    for index in (1, 2, 3):
        argv = ['controlled-collector', '--out', str(tmp_path / f'capture_{index}'),
                '--campaign-budget-file', str(budget), '--campaign-row-id', 'selected',
                '--campaign-repeats', '3', '--campaign-max-retries', '2',
                '--campaign-max-transport-failures', '20',
                '--task-logical-repeat', f'repeat:{index - 1}', '--invalid-repeat-max-retries', '0']
        calls.append({'repetition_index': index, 'argv': argv, 'command_sha256': generic_energy.canonical_hash(argv),
                      'workload_command_file': 'fixture', 'workload_command_sha256': 'bound',
                      'preflight_command_file': 'fixture', 'preflight_command_sha256': 'bound'})
    plan = {'config': {'identity': identity}, 'identity': identity, 'calls': calls,
            'config_path': 'fixture', 'config_sha256': 'bound',
            'performance_result': 'fixture', 'performance_result_sha256': 'bound',
            'collector_script': 'fixture', 'collector_script_sha256': 'bound',
            'hardware_setups_file': 'fixture', 'hardware_setups_file_sha256': 'bound',
            'campaign_budget_file': str(budget), 'campaign_row_id': 'selected',
            'campaign_source_id': 'fixture', 'campaign_limits': limits}
    path = tmp_path / 'generic_energy_plan.json'
    completed.write_json(path, plan)
    monkeypatch.setenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES', 'required')
    monkeypatch.setattr(RemoteProcessLeaseJournal, 'from_environment', classmethod(lambda cls, **k: object()))
    monkeypatch.setattr(generic_energy, 'validate_config', lambda *a, **k: {})
    monkeypatch.setattr(generic_energy, 'file_hash', lambda *a: 'bound')
    monkeypatch.setattr(completed, '_verified_energy_resume_command_context', lambda *a: True)
    monkeypatch.setattr(completed, 'check_energy_campaign', lambda *a, **k: {})
    stages = []
    monkeypatch.setattr(completed, 'stage_energy_case', lambda *a: stages.append(True))
    dispatched = []
    def collector(argv, **kwargs):
        assert kwargs['process_registry'] is registry
        assert kwargs['cancel_event'] is cancel
        assert kwargs['should_cancel']() is False
        dispatched.append(argv)
        cancel.set()
        return SimpleNamespace(returncode=130)
    monkeypatch.setattr(native_progress, 'run_streaming', collector)
    with pytest.raises(RuntimeError, match='collector failed'):
        completed.run_energy_case(path, {'identity': identity}, None,
                                  process_registry=registry, cancel_event=cancel)
    assert len(dispatched) == len(stages) == 1
    receipt = completed.read_json(tmp_path / 'capture_1_process_receipt.json')
    assert receipt['returncode'] == 130
    assert completed.read_json(tmp_path / 'controller_cleanup.json')['owned_local_remaining'] == 0
    assert not (tmp_path / 'capture_2_started.json').exists()


def test_preflight_blocked_cli_attempt_is_not_a_physical_capture(prepared, monkeypatch):
    value, context = _owned_run(prepared, monkeypatch)
    def run_case(row, *a, **k):
        path = Path(row['result_path'])
        write_result(path, result(row['runtime_config']))
        completed.write_json(path.parent / 'dispatch_attempts.json', {'attempts': [
            {'repetition_index': index, 'returncode': 0} for index in (1, 2, 3)]})
        row['dispatch_attempted'] = True
    def prepare_energy(row, **kwargs):
        budget = completed.read_json(kwargs['campaign_budget_file'])
        completed.write_json(kwargs['output'] / 'generic_energy_plan.json', {
            'campaign_budget_file': str(kwargs['campaign_budget_file']),
            'campaign_row_id': next(iter(budget['tasks']))})
        return {}
    def run_energy(path, *a, **k):
        completed.write_json(path.parent / 'capture_1_started.json', {'cli_attempt': True})
        raise RuntimeError('runtime preflight blocked before collector creation')
    monkeypatch.setattr(completed, 'run_case', run_case)
    monkeypatch.setattr(completed, 'prepare_energy_case', prepare_energy)
    monkeypatch.setattr(completed, 'run_energy_case', run_energy)
    outcome = adapter.run_generic(value, context)
    assert outcome['status'] == 'failed'
    assert outcome['collector_cli_attempts'] == 1
    assert outcome['new_capture_count'] == outcome['failed_capture_count'] == 0
    assert outcome['physical_execution_uncertain'] is True
    assert outcome['valid_repeats'] == {'performance': 3, 'energy': 0}
