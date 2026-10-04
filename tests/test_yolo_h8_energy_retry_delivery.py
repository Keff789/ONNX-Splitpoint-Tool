"""Authorized H8 retry delivery through existing writers and local process leaves."""
import copy
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.energy import collector
from scripts import energy_measurement_cli as cli
from scripts import run_native_producer_energy_from_summary as executor
from test_v283_campaign_energy_budget import campaign_rig, measure
from test_yolo_authorized_energy_budget import case, budget, write
from test_yolo_h8_energy_budget_limits import h8, install, NEW, ROWS
from test_v267_energy_aggregate_import import _planned_row


def rig_for(tmp_path, monkeypatch, modes):
    root = tmp_path / 'local-processes'; root.mkdir()
    return campaign_rig(root, monkeypatch, modes)


def bounded(rig, current, name='measurement', **kwargs):
    return rig.measure(name, task_budget_file=None, task_max_chains=None,
        task_max_transport_failures=None, _task_budget=current,
        invalid_repeat_max_retries=5, invalid_repeat_reconnect_backoff_s=10,
        **kwargs)


def observed_pauses(monkeypatch):
    """Only accelerate the existing wait clock; keep its loop/cancel logic real."""
    pauses = []
    original = collector._collector_reconnect_backoff
    real_time = collector.time
    def backoff(seconds, **kwargs):
        elapsed = [0.0]
        fake = SimpleNamespace(**vars(real_time))
        fake.monotonic = lambda: elapsed[0]
        fake.sleep = lambda duration: elapsed.__setitem__(0, elapsed[0] + duration)
        with monkeypatch.context() as clock:
            clock.setattr(collector, 'time', fake)
            evidence = original(seconds, **kwargs)
        pauses.append(evidence)
        return evidence
    monkeypatch.setattr(collector, '_collector_reconnect_backoff', backoff)
    return pauses


def test_three_sixth_attempt_successes_use_eighteen_total_chains(h8, tmp_path, monkeypatch, record_property):
    install(h8)
    rig = rig_for(tmp_path, monkeypatch, ['drop'] * 3 + (['drop'] * 4 + ['success']) * 3)
    pauses = observed_pauses(monkeypatch)
    with budget(h8, row=ROWS[2], limits=NEW) as current:
        result = bounded(rig, current)
        assert result['ok'] and result['repeat_contract_complete']
        assert result['invalid_repeat_max_retries'] == 5
        assert result['invalid_repeat_reconnect_backoff_s'] == 10
        assert current.counts()['begun_chains'] == 18
        assert current.counts()['valid_logical_repeats'] == 3
        assert current.source['transport_failures'] == 15
        for logical in range(3):
            entries = [c for c in current.data['chains'] if c['logical_repeat'] == f'repeat:{logical}']
            assert len(entries) == 6 and [c['valid'] for c in entries] == [False] * 5 + [True]
        assert not current.allowed('repeat:3')
        assert current.counts()['collector_starts'] == 18
    assert rig.counts() == {'collector': 18, 'preflight': 18, 'workload': 18}
    assert len(pauses) == 15 and all(p['elapsed_backoff_s'] >= 10 for p in pauses)
    assert rig.defaults.invalid_repeat_reconnect_backoff_s == 0  # task override did not mutate defaults
    record_property('eighteen_physical_processes', json.dumps({'counts': rig.counts(), 'pauses': pauses}))


def prepared_partial(h8, tmp_path, monkeypatch, modes):
    rig = rig_for(tmp_path, monkeypatch, ['success', *modes])
    first = measure(rig, 'original-valid', row='seed', campaign_max_retries=2,
        campaign_max_transport_failures=20, run_count=1, exact_run_count=True,
        invalid_repeat_max_retries=0)
    assert first['ok']
    original = copy.deepcopy(first['task_budget']['chains'][0])
    h8.root['tasks'][h8.row]['chains'][0] = original
    write(h8.path, h8.root)
    install(h8)
    return rig, original


def test_resume_reuses_original_valid_repeat_and_pauses_before_first_new_capture(h8, tmp_path, monkeypatch):
    rig, original = prepared_partial(h8, tmp_path, monkeypatch, ['drop', 'success', 'success'])
    original_summary = Path(original['run_directory']) / 'energy_summary.json'
    original_bytes = original_summary.read_bytes()
    pauses = observed_pauses(monkeypatch)
    with budget(h8, limits=NEW) as current:
        result = bounded(rig, current, reuse_campaign_repeats=True)
        assert result['ok'] and result['repeat_contract_complete']
        assert result['runs'][0]['campaign_reused_repeat']['run_directory'] == original['run_directory']
        assert current.data['chains'][0] == original
        assert current.counts()['begun_chains'] == 5  # two old plus three new
        assert current.counts()['valid_logical_repeats'] == 3
    assert rig.counts()['collector'] == 4  # one retained fixture capture plus three new
    assert original_summary.read_bytes() == original_bytes
    resumed = json.loads((rig.root / 'measurement/run_001/energy_summary.json').read_text())
    assert resumed['collector_recovery_before_start']['trigger_reasons'] == ['campaign_repeat_reentry']
    assert len(pauses) == 2 and all(p['requested_backoff_s'] == 10 for p in pauses)


def test_old_unfinished_attempt_consumes_one_of_six(h8, tmp_path, monkeypatch):
    rig, _ = prepared_partial(h8, tmp_path, monkeypatch, ['drop', 'success'] + ['drop'] * 4)
    pauses = observed_pauses(monkeypatch)
    with budget(h8, limits=NEW) as current:
        result = bounded(rig, current, reuse_campaign_repeats=True)
        assert not result['ok'] and result['error'] == 'task_logical_retry_limit'
        attempts = [c for c in current.data['chains'] if c['logical_repeat'] == 'repeat:1']
        assert len(attempts) == 6 and attempts[0]['finished'] is False
        assert sum(c['collector_started'] for c in attempts) == 6
    assert rig.counts()['collector'] == 7  # original valid + five retries + independent repeat2
    assert len(pauses) == 5


def test_fresh_unconfirmed_source_end_never_retries(h8, tmp_path, monkeypatch):
    install(h8)
    rig = rig_for(tmp_path, monkeypatch, ['first_sample', 'success'])
    pauses = observed_pauses(monkeypatch)
    with budget(h8, row=ROWS[2], limits=NEW) as current:
        result = bounded(rig, current)
        assert result['error'] == 'campaign_source_completion_unresolved'
        assert current.is_set()
    assert rig.counts()['collector'] == 1 and pauses == []


@pytest.mark.parametrize('pause', [None, 0, 5])
def test_authorized_budget_requires_bound_pause_even_for_direct_cli(h8, tmp_path, monkeypatch, pause):
    install(h8)
    rig = rig_for(tmp_path, monkeypatch, ['success'])
    with budget(h8, row=ROWS[2], limits=NEW) as current:
        with pytest.raises(ValueError, match='bound 10 second reconnect pause'):
            rig.measure('wrong-pause', task_budget_file=None, task_max_chains=None,
                task_max_transport_failures=None, _task_budget=current,
                invalid_repeat_max_retries=5, invalid_repeat_reconnect_backoff_s=pause)
    assert rig.counts() == {'collector': 0, 'preflight': 0, 'workload': 0}


def execution_fixture(h8, tmp_path):
    root = tmp_path / 'execution'
    row, plan = _planned_row(root / 'measurements/row/plan', runs=3)
    row.update(backend='hailo8_to_trt', model='yolo26s', case='b038',
               setup_id='orin_nx_hailo8_01', measurement_setup_id='orin_nx_hailo8_01',
               precision='uint8_dequant_fp16', contract='uint8_dequant')
    parts = executor._replace_command_option(shlex.split(row['measure_command']), '--setup-id', row['setup_id'])
    parts += ['--campaign-budget-file', str(h8.path), '--campaign-row-id', h8.row,
              '--campaign-repeats', '3', '--campaign-max-retries', '2',
              '--campaign-max-transport-failures', '20', '--invalid-repeat-max-retries', '2']
    row['measure_command'] = shlex.join(parts)
    return row, plan, root


def test_normal_execution_writer_and_cli_deliver_only_authorized_five_and_ten(h8, tmp_path, monkeypatch):
    install(h8)
    for key, value in h8.env.items(): monkeypatch.setenv(key, value)
    row, plan, root = execution_fixture(h8, tmp_path)
    original = copy.deepcopy(row)
    result = executor._prepare_measurement_execution(row, plan, allowed_root=root)
    assert row == original
    assert result['row']['measure_command_planned'] == original['measure_command']
    assert executor._command_option(result['command'], '--campaign-max-retries') == '5'
    assert executor._command_option(result['command'], '--invalid-repeat-max-retries') == '5'
    assert executor._command_option(result['command'], '--invalid-repeat-reconnect-backoff-s') == '10.0'
    seen = []
    from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup
    monkeypatch.setattr(cli, '_measurement_context', lambda *args: (EnergyDefaults(), EnergySetup(setup_id=row['setup_id'])))
    def capture(**kwargs):
        seen.append(kwargs)
        return {'ok': True}
    monkeypatch.setattr(cli, 'run_fast_firmware_measurement', capture)  # argv delivery only; actual collector tested above
    assert cli.main(result['argv'][2:]) == 0
    assert seen[0]['invalid_repeat_max_retries'] == seen[0]['campaign_max_retries'] == 5
    assert seen[0]['campaign_repeats'] == 3 and seen[0]['invalid_repeat_reconnect_backoff_s'] == 10
    assert seen[0]['campaign_max_transport_failures'] == 20


@pytest.mark.parametrize('option,value', [('--campaign-max-retries','3'), ('--campaign-repeats','4'),
    ('--campaign-max-transport-failures','21'), ('--invalid-repeat-max-retries','1')])
def test_execution_rejects_unbound_baseline(h8, tmp_path, monkeypatch, option, value):
    install(h8)
    for key, item in h8.env.items(): monkeypatch.setenv(key, item)
    row, plan, root = execution_fixture(h8, tmp_path)
    row['measure_command'] = shlex.join(executor._replace_command_option(shlex.split(row['measure_command']), option, value))
    with pytest.raises(ValueError, match='baseline differs'):
        executor._prepare_measurement_execution(row, plan, allowed_root=root)
    assert not list(root.rglob('attempt_*'))


def test_defaults_and_mirrors_unchanged_outside_explicit_override():
    from onnx_splitpoint_tool.energy.config import EnergyDefaults
    assert EnergyDefaults().invalid_repeat_max_retries == 1
    assert EnergyDefaults().invalid_repeat_reconnect_backoff_s == 5
    root = Path(__file__).resolve().parents[1]
    for name in ('energy_measurement_cli.py', 'run_native_producer_energy_from_summary.py'):
        assert (root / 'scripts' / name).read_bytes() == (root / 'onnx_splitpoint_tool/resources/remote_scripts' / name).read_bytes()
