"""Real energy control path with independently checked local process leaves."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading

import pytest
from onnx_splitpoint_tool.energy import collector
from onnx_splitpoint_tool.energy.config import EnergyDefaults, EnergySetup

FIXTURES = Path(__file__).parent / 'fixtures/v283_energy_task_budget'


class Rig:
    def __init__(self, root, monkeypatch, scenario):
        self.root = root
        self.events = root / 'events.jsonl'
        self.scenario = root / 'scenario.json'
        self.scenario.write_text(json.dumps(scenario))
        for src in FIXTURES.glob('*.py'):
            dest = root / src.name
            shutil.copyfile(src, dest)
            dest.chmod(0o700)
        self.env = {'R3_FAKE_EVENTS': str(self.events), 'R3_FAKE_SCENARIO': str(self.scenario)}
        for key, value in self.env.items():
            monkeypatch.setenv(key, value)
        self.defaults = EnergyDefaults(collector_binary=str(root / 'fake_collector.py'),
            power_calculations_binary=str(root / 'fake_power.py'), command_startup_budget_s=15,
            sample_rate=2000, pre_duration_s=0, post_duration_s=0,
            run_count=3, physical_scope='MB', window_label='command', invalid_repeat_reconnect_backoff_s=0)
        self.setup = EnergySetup(setup_id='fake', enabled=True, urecs_address='FAKE_NO_SOCKET')
        self.command = shlex.join([sys.executable, str(root / 'fake_workload.py'),
                                   collector._ENERGY_PREFLIGHT_NONCE_TOKEN])
        self.preflight = shlex.join([sys.executable, str(root / 'fake_preflight.py'),
                                     collector._ENERGY_PREFLIGHT_NONCE_TOKEN])

    def counts(self):
        rows = [json.loads(line) for line in self.events.read_text().splitlines()] if self.events.exists() else []
        return {kind: sum(r['kind'] == kind for r in rows) for kind in ('preflight', 'collector', 'workload')}

    def measure(self, name='measurement', **overrides):
        kwargs = dict(setup=self.setup, defaults=self.defaults, duration_s=1., run_count=3,
            inference_count=100, physical_scope='MB', window_label='command', run_id='same-task',
            require_runtime_work_units=True, require_command_window_alignment=True,
            preflight_command=self.preflight, preflight_expected_command_contract_sha256='a'*64,
            task_budget_file=self.root / 'budget.json', task_max_chains=6,
            task_max_transport_failures=2, invalid_repeat_max_retries=1, subprocess_env=self.env)
        kwargs.update(overrides)
        return collector.run_fast_firmware_measurement(self.command, self.root / name, **kwargs)


def evidence(rig, result, record_property):
    actual = rig.counts()
    counts = result['task_budget']['counts']
    assert actual == {'preflight': counts['preflight_chains'], 'collector': counts['collector_starts'],
                      'workload': counts['workload_starts']}
    record_property('fake_evidence', json.dumps({'actual_process_starts': actual,
        'task_budget': result['task_budget'], 'status': result['status'], 'ok': result['ok']}))
    assert (rig.root / 'budget.json').is_file()


def test_fake_processes_independently(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['success', 'first_sample', 'closed', 'drop', 'coverage'])
    env = {**os.environ, **rig.env}
    p = subprocess.run([sys.executable, str(tmp_path/'fake_preflight.py'), 'nonce'], env=env, capture_output=True, text=True)
    assert p.returncode == 0
    data = json.loads(p.stdout)
    assert data['attestation_sha256'] == collector._preflight_attestation_seal(data)
    work = tmp_path/'direct_work.sh'
    work.write_text('#!/bin/sh\nexec ' + rig.command + '\n'); work.chmod(0o700)
    for i, expected in enumerate([0, 1, 1, 0, 0]):
        storage = tmp_path/f'direct_{i}'
        result = subprocess.run([str(tmp_path/'fake_collector.py'), f'-s={storage}', f'-c={work}',
            '-d=17s', '--run-id', 'fake', '--window-id', str(i)], env=env, capture_output=True, text=True)
        assert result.returncode == expected
        if i in (1, 2):
            assert 'sending half is closed' in result.stderr
        else:
            marker = json.loads((storage/'command_window_markers.json').read_text())
            assert marker['stream']['dropped_samples'] == (1 if i == 3 else 0)
            assert marker['stream']['trace_covers_window'] == (i != 4)
    out = tmp_path/'power_direct'
    result = subprocess.run([str(tmp_path/'fake_power.py'), f'--output-path={out}',
                            '-c', '-r', '--estimated-duration=3'], env=env)
    assert result.returncode == 0 and 'energy: 4.8' in (out/'results.yaml').read_text()
    assert rig.counts() == {'preflight': 1, 'collector': 5, 'workload': 3}
    record_property('fake_leaf_counts', json.dumps(rig.counts()))


@pytest.mark.parametrize('failure', ['first_sample', 'closed', 'drop', 'coverage'])
def test_no_fourth_chain_after_two_transport_failures(tmp_path, monkeypatch, failure, record_property):
    rig = Rig(tmp_path, monkeypatch, ['success', failure, failure, 'success'])
    result = rig.measure()
    assert rig.counts() == {'preflight': 3, 'collector': 3, 'workload': 1 if failure in ('first_sample', 'closed') else 3}
    assert result['error'] == 'task_transport_failure_limit'
    assert result['repeat_retry_attempt_count'] == 0
    assert result['task_budget']['counts']['valid_logical_repeats'] == 1
    assert result['terminal'] and not result['ok']
    assert result['runs'][1]['status'] == ('collector_failed' if failure in ('first_sample', 'closed') else 'collector_finished')
    # Repeated real summary writes do not drive counters or clear persisted stop.
    before = rig.counts()
    for _ in range(3):
        collector._write_json(tmp_path/'measurement/run_001/energy_summary.json', result['runs'][1])
    again = rig.measure('reentry')
    assert rig.counts() == before and again['error'] == result['error']
    evidence(rig, result, record_property)


def test_failed_preflights_share_six_chain_budget_across_entries(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['success'])
    rig.env['R3_PREFLIGHT_FAIL'] = '1'
    for i in range(2):
        result = rig.measure(f'preflight_{i}')
        assert result['runs'][0]['status'] == 'preflight_failed'
    assert rig.counts() == {'preflight': 2, 'collector': 0, 'workload': 0}
    rig.env.pop('R3_PREFLIGHT_FAIL')
    result = rig.measure('good', run_count=5)
    assert result['error'] == 'task_chain_limit'
    assert rig.counts() == {'preflight': 6, 'collector': 4, 'workload': 4}
    assert result['task_budget']['counts']['begun_chains'] == 6
    assert result['task_budget']['counts']['valid_logical_repeats'] == 4
    evidence(rig, result, record_property)


def test_transient_retry_keeps_valid_selection(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['first_sample', 'success', 'success', 'success'])
    result = rig.measure()
    assert result['ok'] and result['repeat_contract_complete']
    assert result['repeat_retry_recovered_count'] == 1
    assert [r['selected_repeat_attempt_index'] for r in result['runs']] == [1, 0, 0]
    assert result['task_budget']['counts']['valid_logical_repeats'] == 3
    assert json.loads((tmp_path/'measurement/run_000/energy_summary.json').read_text())['final_energy_gate_status'] == 'fail'
    evidence(rig, result, record_property)


def test_retry_cap_cannot_reset_on_reentry(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['first_sample', 'success', 'success', 'first_sample', 'success'])
    result = rig.measure(task_max_transport_failures=9)
    assert not result['ok'] and rig.counts()['collector'] == 4
    again = rig.measure('reentry', task_max_transport_failures=9)
    assert again['error'] == 'task_logical_retry_limit'
    assert rig.counts()['collector'] == 4
    evidence(rig, again, record_property)


def test_cancel_between_retry_check_and_reentry(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['first_sample', 'success', 'success', 'success'])
    cancel = threading.Event()
    original = collector._collector_reconnect_backoff
    calls = []
    def cancel_before_retry(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        if len(calls) == 2:  # first between initial repeats, then retry planning
            cancel.set()
        return result  # stale cancellation flag tests the actual next-start gate
    monkeypatch.setattr(collector, '_collector_reconnect_backoff', cancel_before_retry)
    result = rig.measure(cancel_event=cancel)
    assert result['error'] == 'task_cancelled'
    assert rig.counts() == {'preflight': 3, 'collector': 3, 'workload': 2}
    evidence(rig, result, record_property)


def test_cancel_after_log_open_prevents_popen(tmp_path, monkeypatch):
    cancel = threading.Event()
    stderr = tmp_path/'stderr.log'
    original = Path.open
    def opening(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if path == stderr:
            cancel.set()
        return handle
    monkeypatch.setattr(Path, 'open', opening)
    marker = tmp_path/'must_not_start'
    result = collector._run_one([sys.executable, '-c', f'open({str(marker)!r}, "w").close()'],
        cwd=None, stdout_path=tmp_path/'stdout.log', stderr_path=stderr, cancel_event=cancel)
    assert result['cancelled'] and result['process_started'] is False
    assert not marker.exists()


def test_cli_propagates_budget_into_real_control(tmp_path, monkeypatch, record_property, capsys):
    from scripts import energy_measurement_cli as cli
    rig = Rig(tmp_path, monkeypatch, ['success', 'first_sample', 'first_sample'])
    monkeypatch.setattr(cli, '_measurement_context', lambda *a: (rig.defaults, rig.setup))
    rc = cli.main(['measure', '--setup-id', 'fake', '--out', str(tmp_path/'cli'),
        '--command', rig.command, '--duration', '1', '--runs', '3', '--physical-scope', 'MB',
        '--window-label', 'command', '--preflight-command', rig.preflight,
        '--preflight-expected-command-contract-sha256', 'a'*64,
        '--task-budget-file', str(tmp_path/'budget.json'), '--task-max-chains', '6',
        '--task-max-transport-failures', '2', '--invalid-repeat-max-retries', '1'])
    result = json.loads(capsys.readouterr().out)
    assert rc != 0 and result['error'] == 'task_transport_failure_limit'
    assert rig.counts()['collector'] == 3
    evidence(rig, result, record_property)


def test_cleanup_uses_only_owned_process_registry(tmp_path, record_property):
    import time
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry, bind_process_registry
    cancel = threading.Event()
    registry = ProcessTreeRegistry()
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 start_new_session=True)
    child = []
    started = tmp_path/'owned_started'
    def on_started(proc, timestamp):
        child.append(proc)
        # Wait for the physical fake to write its own start event.
        deadline = time.monotonic() + 3
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert started.exists()
        cancel.set()
    try:
        with bind_process_registry(registry):
            result = collector._run_one([sys.executable, '-c',
                f'from pathlib import Path; import time; Path({str(started)!r}).write_text("started"); time.sleep(30)'],
                cwd=None, stdout_path=tmp_path/'out', stderr_path=tmp_path/'err',
                cancel_event=cancel, on_process_started=on_started, timeout=5)
        assert result['cancelled'] and result['process_started']
        assert len(child) == 1 and child[0].poll() is not None
        assert unrelated.poll() is None
        registry.assert_quiescent()
        record_property('owned_process_cleanup', json.dumps({'owned_starts': len(child),
            'owned_terminated': True, 'unrelated_preserved': True, 'registry_quiescent': True}))
    finally:
        # Both children were created by this test; only the owned one belongs
        # to the product registry. Dispose of the independent control here.
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_checkpoint_limits_and_interrupted_chain_cannot_reset(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['success'])
    result = rig.measure(task_max_chains=1)
    assert result['error'] == 'task_chain_limit' and rig.counts()['collector'] == 1
    with pytest.raises(ValueError, match='limits differ'):
        rig.measure('changed_limits', task_max_chains=6)
    evidence(rig, result, record_property)


def test_bounded_task_rejects_unbudgeted_duration_probe(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['success'])
    result = rig.measure(duration_s=None)
    assert result['error'] == 'task_requires_explicit_duration'
    assert rig.counts() == {'preflight': 0, 'collector': 0, 'workload': 0}
    assert result['terminal']
    evidence(rig, result, record_property)


def test_interrupted_checkpoint_is_terminal_without_another_start(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['success'])
    data = {'limits': {'max_chains': 6, 'max_transport_failures': 2, 'max_retries': 1},
            'chains': [{'logical_repeat': 'repeat:0', 'run_directory': str(tmp_path/'aborted'),
                        'preflight_requested': False, 'collector_started': False, 'finished': False}],
            'stop_reason': ''}
    (tmp_path/'budget.json').write_text(json.dumps(data))
    result = rig.measure()
    assert result['error'] == 'task_interrupted_chain_unresolved'
    assert result['task_budget']['counts']['begun_chains'] == 1
    assert rig.counts() == {'preflight': 0, 'collector': 0, 'workload': 0}
    evidence(rig, result, record_property)


def test_reentry_run_label_cannot_reset_logical_retry_budget(tmp_path, monkeypatch, record_property):
    rig = Rig(tmp_path, monkeypatch, ['first_sample', 'success', 'success', 'first_sample'])
    result = rig.measure(task_max_transport_failures=9)
    assert result['error'] == 'task_logical_retry_limit'
    again = rig.measure('different_directory', run_id='changed-label', task_max_transport_failures=9)
    assert again['error'] == 'task_logical_retry_limit' and rig.counts()['collector'] == 4
    evidence(rig, again, record_property)
