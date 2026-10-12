"""Explicit one-process acceptance keeps ordinary scientific cohorts strict."""
import copy
import json
from pathlib import Path
import shutil

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance_generic as adapter
from onnx_splitpoint_tool.workflow import generic_completed_task as completed
from onnx_splitpoint_tool.generic_completion_reporting import load_valid_repetitions, validate_completion_case
from tests.test_generic_completion_dispatch import LocalTransport, prepared_case
from test_hardware_acceptance_generic import prepared, _owned_run
from test_generic_completion_reporting import result, write_result
from tests.test_generic_energy_generated_processes import generated_case
from onnx_splitpoint_tool.runners.generic_energy import canonical_hash, file_hash
from onnx_splitpoint_tool.generic_energy_reporting import collect_generic_energy_case

CLAIM_FLAGS = {'claim_eligible', 'claim_ok', 'semantic_claim_ok', 'eligible_for_ranking',
    'eligible_for_scientific_claim', 'eligible_for_scientific_primary', 'eligible_for_energy_results_import',
    'candidate_eligible_for_scientific_primary', 'eligible_for_claim_transfer', 'primary_energy_claim_gate'}


@pytest.fixture
def functional(prepared):
    _, context, cell, _ = prepared
    context = {**context, 'run_dir': str(Path(context['run_dir']).parent / 'functional'),
               'acceptance_mode': 'functional', 'requested_repetitions': 1, 'expected_repetitions': 1,
               'profile': {'hardware_acceptance': {'mode': 'functional'},
                   'native_producers': {'warmup': 100, 'frames': 1000, 'repetitions': 1,
                       'energy': {'task_budget': {'enabled': True, 'max_retries': 2, 'max_transport_failures': 20}}}}}
    return adapter.prepare_generic(cell, context), context, cell


def test_functional_prepare_requests_one_actual_process_and_capture(functional):
    value, context, _ = functional
    plan = completed.read_json(Path(value['execution_plan']))
    assert value['acceptance_mode'] == plan['acceptance_mode'] == 'functional'
    assert value['performance_processes'] == value['energy_captures'] == plan['planned_repetitions'] == 1
    assert plan['planned_measured_completions'] == 1000
    assert value['performance_recipe'] == {'warmup_count': 100, 'task_count': 1000, 'repetitions': 1,
                                          'process_repetition_policy': 'separate_processes'}
    assert value['energy_recipe']['duration_s'] == 60
    assert plan['rows'][0]['runtime_config']['repetitions'] == 3
    assert not (Path(context['run_dir']) / 'energy').exists()


def test_one_repeat_requires_explicit_functional_profile(prepared):
    _, context, cell, _ = prepared
    context['expected_repetitions'] = 1
    with pytest.raises(ValueError, match='acceptance recipe'):
        adapter.prepare_generic(cell, context)
    context['acceptance_mode'] = 'functional'
    with pytest.raises(ValueError, match='explicit profile'):
        adapter.prepare_generic(cell, context)


def test_normal_functional_dispatch_runs_one_and_preserves_scientific_partial(tmp_path):
    row = prepared_case(tmp_path)
    row.update(acceptance_mode='functional', requested_repetitions=1)
    transport = LocalTransport()
    completed.run_case(row, transport, tmp_path / 'output', 1)
    assert [index for _, index in transport.calls] == [1]
    path = Path(row['result_path'])
    projection = validate_completion_case(completed.read_json(path), row, result_root=path.parent)
    assert projection['status'] == 'partial'
    assert projection['valid_repetition_count'] == 1 and projection['completed_task_count'] == 1000
    assert projection['pending_repetition_indices'] == [2, 3]
    assert not any('median' in key or 'stddev' in key for key in projection)
    assert len(load_valid_repetitions(path, row)) == 1


@pytest.mark.parametrize('fault', ['oracle', 'sample'])
def test_functional_dispatch_rejects_failed_oracle_and_changed_sample(tmp_path, fault):
    row = prepared_case(tmp_path)
    row.update(acceptance_mode='functional', requested_repetitions=1)
    class BrokenOutput(LocalTransport):
        def run_streaming(self, command, **kwargs):
            status = super().run_streaming(command, **kwargs)
            import shlex
            argv = shlex.split(command)
            output = Path(argv[argv.index('--out-dir') + 1])
            path = output / 'generic_completion_results.json'
            raw = json.loads(path.read_text())
            if fault == 'oracle':
                raw['repetitions'][0]['functional_control']['status'] = 'failed'
                path.write_text(json.dumps(raw))
            else:
                sample = next(iter(raw['repetitions'][0]['raw_output_samples'].values()))
                (output / sample['path']).write_bytes(b'changed actual numeric output')
            return status
    transport = BrokenOutput()
    with pytest.raises(ValueError, match='functional control|output sample'):
        completed.run_case(row, transport, tmp_path / 'output', 1)
    assert len(transport.calls) == 1


@pytest.mark.parametrize('scope', [{'requested_repetitions': 1},
    {'acceptance_mode': 'functional'}, {'acceptance_mode': 'functional', 'requested_repetitions': 3},
    {'acceptance_mode': 'functional', 'requested_repetitions': True}])
def test_invalid_dispatch_role_cannot_shorten_default_work(scope, tmp_path):
    row = prepared_case(tmp_path)
    row.update(scope)
    transport = LocalTransport()
    with pytest.raises(ValueError, match='acceptance'):
        completed.run_case(row, transport, tmp_path / 'output', 1)
    assert transport.calls == []


def test_functional_result_has_one_actual_observation_and_no_scientific_statistics(functional, monkeypatch):
    value, context, _ = functional
    value, context = _owned_run((value, context, None, None), monkeypatch)
    def run_case(row, *args, **kwargs):
        path = Path(row['result_path'])
        write_result(path, result(row['runtime_config'], 1))
        completed.write_json(path.parent / 'dispatch_attempts.json', {'attempts': [{'repetition_index': 1, 'returncode': 0}]})
        row['dispatch_attempted'] = True
    def prepare_energy(row, **kwargs):
        budget = completed.read_json(kwargs['campaign_budget_file'])
        key = adapter._campaign_key(row['identity'])
        assert budget['tasks'][key]['limits'] == {'max_chains': 3, 'max_retries': 2, 'max_transport_failures': 20}
        plan = {'campaign_budget_file': str(kwargs['campaign_budget_file']), 'campaign_row_id': key}
        completed.write_json(kwargs['output'] / 'generic_energy_plan.json', plan)
        return plan
    def run_energy(path, row, transport, **kwargs):
        completed.write_json(path.parent / 'capture_1_started.json', {'cli_attempt': True})
        plan = completed.read_json(path)
        budget_path = Path(plan['campaign_budget_file'])
        budget = completed.read_json(budget_path)
        budget['tasks'][plan['campaign_row_id']]['chains'] = [{'logical_repeat': 'repeat:0',
            'collector_started': True, 'finished': True, 'valid': True, 'source_completion_verified': True,
            'run_directory': str(path.parent / 'capture_1/run_000')}]
        completed.write_json(budget_path, budget)
        observation = {'acceptance_mode': 'functional', 'status': 'completed', 'normal_report_verified': True,
            'valid_repetitions': 1, 'energy_efficiency_claim_eligible': False,
            'scientific_cohort_complete': False, 'observations': [{'completed_tasks': 1000}]}
        completed.write_json(path.parent / 'reports/generic_completed_task_energy.json', observation)
        return observation
    monkeypatch.setattr(completed, 'run_case', run_case)
    monkeypatch.setattr(completed, 'prepare_energy_case', prepare_energy)
    monkeypatch.setattr(completed, 'run_energy_case', run_energy)
    outcome = adapter.run_generic(value, context)
    assert outcome['status'] == 'completed'
    assert outcome['valid_repeats'] == {'performance': 1, 'energy': 1}
    assert outcome['new_performance_process_count'] == outcome['new_capture_count'] == 1
    assert outcome['performance']['status'] == 'partial'
    assert outcome['output_validation']['status'] == 'passed'
    assert outcome['scientific_cohort_complete'] is outcome['scientific_claim_eligible'] is False
    report = completed.read_json(Path(value['output_root']) / 'FUNCTIONAL_ACCEPTANCE.json')
    assert report['actual_valid_performance_processes'] == report['actual_valid_energy_captures'] == 1
    assert report['scientific_statistics_published'] is False
    assert report['output_oracle']['status'] == 'passed'
    assert report['output_oracle']['reference_binding'] == report['functional_reference']
    assert report['output_samples'] and report['completed_result_artifact']['sha256']
    assert not any('median' in key or 'stddev' in key for key in report['normal_performance_projection'])


def _functional_plan(value, context):
    row = completed.read_json(Path(value['execution_plan']))['rows'][0]
    path = adapter._budget(value, row, context)
    return {'acceptance_mode': 'functional', 'requested_repetitions': 1,
        'campaign_budget_file': str(path), 'campaign_row_id': adapter._campaign_key(row['identity']),
        'campaign_source_id': '192.0.2.2',
        'campaign_limits': {'max_chains': 3, 'max_retries': 2, 'max_transport_failures': 20}}


def test_functional_capture_admission_keeps_current_three_chain_limit(functional):
    value, context, _ = functional
    plan = _functional_plan(value, context)
    proof = adapter._phase_energy_admission(value, context, plan, {'repetition_index': 1})
    assert proof['remaining_single_starts_for_slot'] == proof['profile_max_chains'] == 3
    path = Path(plan['campaign_budget_file'])
    budget = completed.read_json(path)
    budget['tasks'][plan['campaign_row_id']]['chains'] = [{'logical_repeat': 'repeat:0',
        'finished': True, 'valid': False, 'source_completion_verified': True,
        'run_directory': str(Path(context['run_dir']) / f'attempt_{index}/run_000')} for index in range(3)]
    completed.write_json(path, budget)
    with pytest.raises(RuntimeError, match='three-chain retry limit'):
        adapter._phase_energy_admission(value, context, plan, {'repetition_index': 1})


@pytest.mark.parametrize('prior_mode, prior_valid', [('functional', False), ('scientific', False), ('scientific', True)])
def test_new_functional_workflow_cannot_reset_prior_slot_credit_or_recapture_valid_slot(functional, prior_mode, prior_valid):
    value, context, _ = functional
    plan = _functional_plan(value, context)
    previous = Path(context['run_dir']).parent / 'failed_functional'
    profile = copy.deepcopy(context['profile'])
    profile['hardware_acceptance']['mode'] = prior_mode
    profile['native_producers']['repetitions'] = 3 if prior_mode == 'scientific' else 1
    completed.write_json(previous / 'profile.yaml', profile)
    prior = completed.read_json(Path(plan['campaign_budget_file']))
    prior['tasks'][plan['campaign_row_id']]['limits']['max_chains'] = 9 if prior_mode == 'scientific' else 3
    prior['tasks'][plan['campaign_row_id']]['chains'] = [{'logical_repeat': 'repeat:0',
        'finished': True, 'valid': prior_valid, 'source_completion_verified': prior_valid,
        'run_directory': str(previous / 'attempt/run_000')}]
    path = previous / 'energy/energy_task_budget.json'
    completed.write_json(path, prior)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match='requires reviewed continuation'):
        adapter._phase_energy_admission(value, context, plan, {'repetition_index': 1})
    assert path.read_bytes() == original


@pytest.fixture
def single_capture(generated_case, tmp_path):
    """Local CPU/synthetic capture fixtures never enter a hardware corpus."""
    plan, descriptors, case = copy.deepcopy(generated_case)
    performance_root = tmp_path / 'performance'
    shutil.copytree(Path(plan['performance_result']).parent, performance_root)
    performance = performance_root / 'generic_completion_results.json'
    raw = completed.read_json(performance)
    raw['repetitions'] = raw['repetitions'][:1]
    raw['status'] = 'partial'
    completed.write_json(performance, raw)
    plan.update(acceptance_mode='functional', requested_repetitions=1,
                performance_result=str(performance), performance_result_sha256=file_hash(performance))
    case.update(acceptance_mode='functional', requested_repetitions=1)
    plan['config']['energy_capture']['performance_result_sha256'] = file_hash(performance)
    config = tmp_path / 'generic_energy_config.json'
    completed.write_json(config, plan['config'])
    plan.update(config_path=str(config), config_sha256=file_hash(config))
    plan['calls'] = plan['calls'][:1]
    descriptors = descriptors[:1]
    prior = completed.read_json(Path(plan['campaign_budget_file']))
    task = prior['tasks'][plan['campaign_row_id']]
    task['chains'] = task['chains'][:1]
    task['limits']['max_chains'] = 3
    plan['campaign_limits']['max_chains'] = 3
    budget = tmp_path / 'campaign.json'
    completed.write_json(budget, prior)
    plan['campaign_budget_file'] = str(budget)
    call, descriptor = plan['calls'][0], descriptors[0]
    for argv in (call['argv'], descriptor['command']):
        argv[argv.index('--campaign-repeats') + 1] = '1'
        argv[argv.index('--campaign-budget-file') + 1] = str(budget)
    call['command_sha256'] = canonical_hash(call['argv'])
    descriptor['command_sha256'] = canonical_hash(descriptor['command'])
    descriptor['process_receipt'].update(argv=descriptor['command'], command_sha256=descriptor['command_sha256'])
    return plan, descriptors, case


def test_functional_import_uses_one_complete_normal_source_and_no_cohort_statistics(single_capture):
    plan, descriptors, case = single_capture
    original = Path(descriptors[0]['raw_aggregate_path']).read_bytes()
    observation = collect_generic_energy_case(plan, descriptors, case)
    assert observation['normal_report_verified'] is True and observation['valid_repetitions'] == 1
    assert observation['acceptance_mode'] == 'functional'
    assert observation['scientific_cohort_complete'] is observation['energy_efficiency_claim_eligible'] is False
    assert len(observation['observations']) == 1 and 'statistics' not in observation
    assert observation['normal_scientific_performance_projection']['status'] == 'partial'
    def claims(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith('claim_eligible') or key in CLAIM_FLAGS:
                    yield item
                yield from claims(item)
        elif isinstance(value, list):
            for item in value:
                yield from claims(item)
    assert any(claim is True for claim in claims(json.loads(original)))
    assert all(claim is False for claim in claims(observation))
    assert observation['observations'][0]['evaluation_role'] == 'functional_acceptance'
    assert observation['observations'][0]['diagnostic_only'] is True
    assert Path(descriptors[0]['raw_aggregate_path']).read_bytes() == original


def test_functional_claim_projection_uses_normal_canonical_diagnostic_clamp():
    from onnx_splitpoint_tool.generic_energy_reporting import _functional_view
    original = {'nested': [{key: True for key in CLAIM_FLAGS}],
                'energy_efficiency_claim_eligible_run_count': 3, 'completed_tasks': 1000}
    projected = _functional_view(original)
    assert all(value is False for value in projected['nested'][0].values())
    assert projected['energy_efficiency_claim_eligible_run_count'] == 0
    assert projected['completed_tasks'] == 1000
    assert original['energy_efficiency_claim_eligible_run_count'] == 3
    assert all(value is True for value in original['nested'][0].values())


@pytest.mark.parametrize('fault', ['implicit_role', 'wrong_slot', 'unclosed_source', 'failed_oracle', 'foreign_source'])
def test_functional_import_cannot_bypass_role_or_ordinary_single_gates(single_capture, fault):
    plan, descriptors, case = single_capture
    if fault == 'implicit_role':
        plan.pop('acceptance_mode')
    elif fault == 'wrong_slot':
        descriptors[0]['repetition_index'] = 2
    elif fault == 'unclosed_source':
        path = Path(plan['campaign_budget_file'])
        budget = completed.read_json(path)
        budget['tasks'][plan['campaign_row_id']]['chains'][0]['source_completion_verified'] = False
        completed.write_json(path, budget)
    else:
        path = Path(plan['performance_result'])
        raw = completed.read_json(path)
        if fault == 'failed_oracle':
            raw['repetitions'][0]['functional_control']['status'] = 'failed'
        else:
            raw['repetitions'][0]['runtime_source_binding']['unchanged'] = False
        completed.write_json(path, raw)
        plan['performance_result_sha256'] = file_hash(path)
        plan['config']['energy_capture']['performance_result_sha256'] = file_hash(path)
    with pytest.raises(ValueError):
        collect_generic_energy_case(plan, descriptors, case)


def test_normal_functional_parent_reimports_one_retained_single_without_new_dispatch(single_capture, tmp_path, monkeypatch):
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseJournal
    from onnx_splitpoint_tool import native_progress
    plan, descriptors, case = single_capture
    plan['resume_reused_sources'] = descriptors
    output = tmp_path / 'parent'
    path = output / 'generic_energy_plan.json'
    completed.write_json(path, plan)
    monkeypatch.setenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES', 'required')
    monkeypatch.setattr(RemoteProcessLeaseJournal, 'from_environment', classmethod(lambda cls, **k: object()))
    monkeypatch.setattr(completed, 'stage_energy_case', lambda *a: pytest.fail('No staging for a retained accepted Single'))
    monkeypatch.setattr(native_progress, 'run_streaming', lambda *a, **k: pytest.fail('No recapture of accepted logical slot'))
    observation = completed.run_energy_case(path, case, None)
    assert observation['valid_repetitions'] == 1 and observation['normal_report_verified'] is True
    assert not (output / 'capture_1_started.json').exists()
    assert completed.read_json(output / 'controller_cleanup.json')['owned_local_remaining'] == 0


def test_functional_energy_planner_uses_one_single_with_three_chain_retry_envelope(single_capture, tmp_path):
    plan, _, case = single_capture
    fresh = completed.prepare_energy_case(case, performance_result=Path(plan['performance_result']),
        output=tmp_path / 'new_plan', collector_python='normal-python', collector_script=Path(plan['collector_script']),
        hardware_setups_file=Path(plan['hardware_setups_file']), remote_command_prefix=[],
        campaign_budget_file=Path(plan['campaign_budget_file']))
    assert fresh['acceptance_mode'] == 'functional' and fresh['requested_repetitions'] == 1
    assert fresh['campaign_limits'] == {'max_chains': 3, 'max_retries': 2, 'max_transport_failures': 20}
    assert len(fresh['calls']) == 1
    argv = fresh['calls'][0]['argv']
    assert argv[argv.index('--campaign-repeats') + 1] == '1'
    assert argv[argv.index('--task-logical-repeat') + 1] == 'repeat:0'
    assert argv[argv.index('--invalid-repeat-max-retries') + 1] == '0'
    assert argv[argv.index('--duration') + 1] == '60.0'
    assert fresh['config']['repetitions'] == fresh['config']['energy_capture']['contract']['repetitions'] == 3
