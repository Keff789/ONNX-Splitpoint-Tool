"""Persistent campaign policy with real local process leaves; no hardware."""
import json
from pathlib import Path
import shlex
import sys

import pytest

from onnx_splitpoint_tool.energy.task_budget import EnergyTaskBudget, source_completion
from test_v283_energy_task_budget import Rig


LIFECYCLE = (
    "Fast Firmware GO sent monotonic_ns=1 local=127.0.0.1:1234 requested_device_us=27000000\n"
    "Fast Firmware protocol end verified monotonic_ns=27000000001 local=127.0.0.1:1234 elapsed_us=27000000\n"
    "Fast Firmware source closed monotonic_ns=27000000002 local=127.0.0.1:1234 verified=true reason=Ok(())\n"
)


def campaign_rig(tmp_path, monkeypatch, modes):
    rig = Rig(tmp_path, monkeypatch, modes)
    fake = tmp_path / 'fake_collector.py'
    fake.write_text(fake.read_text().replace('raise SystemExit(proc.returncode)',
        f'print({LIFECYCLE!r}, end="")\nraise SystemExit(proc.returncode)'))
    return rig


def measure(rig, name, row='row-a', **kw):
    args = dict(task_budget_file=None, task_max_chains=None, task_max_transport_failures=None,
                campaign_budget_file=rig.root/'campaign.json', campaign_row_id=row,
                campaign_repeats=3, campaign_max_retries=1, campaign_max_transport_failures=2)
    args.update(kw)
    return rig.measure(name, **args)


def test_unresolved_source_blocks_other_rows_and_reentry_but_not_other_source(tmp_path, monkeypatch, record_property):
    rig = campaign_rig(tmp_path, monkeypatch, ['success', 'first_sample', 'success'])
    result = measure(rig, 'first')
    assert result['status'] == 'BLOCKED'
    assert result['error'] == 'campaign_source_completion_unresolved'
    assert rig.counts() == {'preflight': 2, 'collector': 2, 'workload': 1}
    for name in ['reentry', 'third']:
        blocked = measure(rig, name, row=name)
        assert blocked['execution_status'] == 'NOT_RUN' and not blocked['ok']
    assert rig.counts()['collector'] == 2
    rig.setup.urecs_address = 'OTHER_PHYSICAL_SOURCE'
    independent = measure(rig, 'independent', row='independent')
    assert independent['ok'] and independent['repeat_contract_complete']
    assert rig.counts() == {'preflight': 5, 'collector': 5, 'workload': 4}
    record_property('actual_process_starts', json.dumps(rig.counts()))


def test_two_transport_failures_with_verified_ends_accumulate_across_rows(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['drop', 'drop', 'success'])
    one = measure(rig, 'one', row='one', run_count=1, exact_run_count=True, invalid_repeat_max_retries=0)
    assert one['task_budget']['campaign_source']['transport_failures'] == 1
    two = measure(rig, 'two', row='two', run_count=1, exact_run_count=True, invalid_repeat_max_retries=0)
    assert two['error'] == 'campaign_source_transport_failure_limit'
    three = measure(rig, 'three', row='three')
    assert three['execution_status'] == 'NOT_RUN'
    assert rig.counts() == {'preflight': 2, 'collector': 2, 'workload': 2}


def test_verified_source_transport_retry_keeps_three_logical_repeats(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['drop', 'success', 'success', 'success'])
    result = measure(rig, 'retry')
    assert result['ok'] and result['repeat_contract_complete']
    assert result['task_budget']['counts']['begun_chains'] == 4
    assert result['task_budget']['counts']['valid_logical_repeats'] == 3
    assert rig.counts() == {'preflight': 4, 'collector': 4, 'workload': 4}


def test_outer_probe_preflights_consume_same_six_chain_budget(tmp_path, monkeypatch, record_property):
    rig = campaign_rig(tmp_path, monkeypatch, ['success'])
    rig.env['R3_PREFLIGHT_FAIL'] = '1'
    prepare = shlex.join([sys.executable, str(tmp_path/'fake_preflight.py'), 'prepare'])
    for logical in range(3):
        for attempt in range(2):
            result = measure(rig, f'{logical}-{attempt}', row='probe', run_count=1, exact_run_count=True,
                             invalid_repeat_max_retries=0, preflight_command=None,
                             preflight_prepare_command=prepare, _task_logical_repeat=f'repeat:{logical}')
            assert result['runs'][0]['status'] == 'preflight_failed'
    blocked = measure(rig, 'seventh', row='probe', run_count=1, exact_run_count=True,
                      invalid_repeat_max_retries=0, preflight_command=None,
                      preflight_prepare_command=prepare, _task_logical_repeat='repeat:3')
    assert blocked['error'] == 'task_chain_limit'
    assert rig.counts() == {'preflight': 6, 'collector': 0, 'workload': 0}
    assert blocked['task_budget']['counts']['begun_chains'] == 6
    record_property('actual_process_starts', json.dumps(rig.counts()))


@pytest.mark.parametrize('mutation', ['missing_go', 'different_port', 'duplicate_end', 'early_end', 'false_close'])
def test_only_complete_exact_lifecycle_releases_source(tmp_path, mutation):
    (tmp_path/'collector_stdout.log').write_text(LIFECYCLE)
    assert source_completion(tmp_path)['verified']
    lines = LIFECYCLE.splitlines(True)
    if mutation == 'missing_go': lines.pop(0)
    elif mutation == 'different_port': lines[1] = lines[1].replace(':1234', ':2345')
    elif mutation == 'duplicate_end': lines.insert(2, lines[1])
    elif mutation == 'early_end': lines[1] = lines[1].replace('elapsed_us=27000000', 'elapsed_us=1')
    else: lines[2] = lines[2].replace('verified=true', 'verified=false')
    (tmp_path/'collector_stdout.log').write_text(''.join(lines))
    assert not source_completion(tmp_path)['verified']


def test_finished_checkpoint_without_source_end_blocks_new_row(tmp_path):
    limits = {'max_chains': 6, 'max_retries': 1, 'max_transport_failures': 2}
    path = tmp_path/'campaign.json'
    path.write_text(json.dumps({'campaign': True, 'max_transport_failures': 2,
        'sources': {'source': {'stop_reason': ''}}, 'tasks': {'old': {
            'source_id': 'source', 'limits': limits, 'stop_reason': '',
            'chains': [{'finished': True, 'collector_started': True}]}}}))
    with path.open('r+') as handle:
        budget = EnergyTaskBudget(handle, limits, campaign_row='new', source_id='source')
        assert not budget.allowed('repeat:0')
        assert budget.data['stop_reason'] == 'campaign_source_interrupted_chain_unresolved'


def test_missing_result_field_cannot_erase_actual_collector_start(tmp_path):
    with (tmp_path/'budget').open('w+') as handle:
        budget = EnergyTaskBudget(handle, {'max_chains': 6, 'max_retries': 1, 'max_transport_failures': 2},
                                  campaign_row='row', source_id='source')
        chain = budget.reserve('repeat:0', tmp_path/'attempt', False)
        budget.collector_started(chain)
        budget.finish(chain, {'ok': False}, tmp_path/'attempt')
        assert chain['collector_started'] is True
        assert budget.source['stop_reason'] == 'campaign_source_completion_unresolved'


def test_budget_projection_keeps_scientific_import_fail_closed(tmp_path, monkeypatch):
    from scripts.run_native_producer_energy_from_summary import _campaign_budget_block_projection, _attach_energy_aggregate
    rig = campaign_rig(tmp_path, monkeypatch, ['first_sample'])
    measure(rig, 'failed')
    measure(rig, 'blocked', row='next')
    command = shlex.join(['energy_measurement_cli.py', '--campaign-budget-file', str(tmp_path/'campaign.json'), '--campaign-row-id', 'next'])
    projection = _campaign_budget_block_projection(command, tmp_path/'blocked')
    assert projection['status'] == 'BLOCKED' and projection['execution_status'] == 'NOT_RUN'
    result = _attach_energy_aggregate({'rc': 1}, command, tmp_path/'blocked', expected_runs=3)
    assert result['energy_aggregate_verified'] is False
    assert result['energy_aggregate_import_status'] == 'rejected_fail_closed'
    assert _campaign_budget_block_projection('energy_measurement_cli.py --runs 3', tmp_path) == {}


def test_variant_arguments_preserve_frozen_registry_over_environment(tmp_path, monkeypatch):
    from scripts.run_evalrun_native_producer_variants import _common_energy_args
    monkeypatch.setenv('ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE', str(tmp_path/'wrong.yaml'))
    registry = str(tmp_path/'frozen.yaml')
    cfg = {'_workflow_context': {'hardware_setups_file': registry}, 'energy': {
        'task_budget': {'enabled': True, 'max_retries': 1, 'max_transport_failures': 2}}}
    args = _common_energy_args(cfg, run_dir=tmp_path/'run', duration_s=1, timeout_s=60)
    assert args[args.index('--hardware-setups-file')+1] == registry
    assert args[args.index('--campaign-budget-file')+1] == str(tmp_path/'run/energy_task_budget.json')


def test_campaign_source_failure_does_not_block_independent_infrastructure():
    from scripts.run_native_producer_energy_from_summary import _energy_global_infrastructure_failure
    result = {'global_infrastructure_failure': 'urecs_transport_unavailable',
              'global_infrastructure_category': 'collector_initialization',
              'energy_task_budget': {'reason': 'campaign_source_completion_unresolved'}}
    assert _energy_global_infrastructure_failure(result) == {}
    result['global_infrastructure_failure'] = 'remote_authentication_failed'
    assert _energy_global_infrastructure_failure(result)['code'] == 'remote_authentication_failed'
