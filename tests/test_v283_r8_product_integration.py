"""R8 normal mode/configuration regressions; no private YAML dependencies."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config, validate_run_modes_config
from onnx_splitpoint_tool.native_execution_contract import resolve_native_execution_contract
from onnx_splitpoint_tool.energy.task_budget import campaign_budget_policy


def legacy_profile():
    config = default_run_modes_config()
    old = copy.deepcopy(config['modes']['standard'])
    old['runtime']['native'].update(frames=1000, warmup=100, repetitions=3)
    return {'name': 'portable_r8', 'execution_preset': {
        'id': 'standard', 'follow_tool_config': True, 'snapshot': old},
        'native_producers': {'enabled': True, 'frames': 1000, 'warmup': 100,
                             'energy': {'enabled': True, 'mode': 'measure'}}}


def test_legacy_standard_final_standard_uses_one_performance_contract():
    profile = legacy_profile()
    for mode, expected in [('standard', (100, 10, 1)), ('final', (1000, 100, 3)), ('standard', (100, 10, 1))]:
        profile, _ = apply_run_mode(profile, mode_id=mode, config=default_run_modes_config())
        contract = resolve_native_execution_contract(profile)
        assert tuple(contract[k] for k in ('frames', 'warmup', 'repetitions')) == expected
        assert profile['execution_preset']['effective']['native_performance_repetitions'] == contract['repetitions']
        assert profile['energy']['repeat_override'] == 3
        assert profile['energy']['generic_enabled'] is False


def test_missing_policy_is_product_default_and_disabled_override_is_visible():
    profile, _ = apply_run_mode(legacy_profile(), config=default_run_modes_config())
    assert campaign_budget_policy(profile['native_producers'])['max_retries'] == 1
    assert profile['native_producers']['energy']['task_budget_source'] == 'product_default'
    profile['native_producers']['energy']['task_budget'] = {'enabled': False}
    profile, _ = apply_run_mode(profile, config=default_run_modes_config())
    assert campaign_budget_policy(profile['native_producers']) == {}
    assert profile['native_producers']['energy']['task_budget_source'] == 'profile_override'


def test_custom_and_frozen_native_budgets_survive():
    profile = legacy_profile()
    profile['native_producers'].update(frames=217, warmup=0, repetitions=2)
    result, _ = apply_run_mode(profile, mode_id='final', config=default_run_modes_config())
    assert tuple(result['native_producers'][k] for k in ('frames', 'warmup', 'repetitions')) == (217, 0, 2)
    profile = legacy_profile()
    profile['execution_preset']['follow_tool_config'] = False
    result, _ = apply_run_mode(profile, config=default_run_modes_config())
    assert resolve_native_execution_contract(result)['repetitions'] == result['execution_preset']['effective']['native_performance_repetitions'] == 3


@pytest.mark.parametrize('custom', [False, True])
def test_versioned_migration_only_replaces_exact_standard_tuple(custom):
    config = default_run_modes_config()
    config['schema_version'] = 13
    config['modes']['standard']['runtime']['native'].update(frames=219 if custom else 1000, warmup=100, repetitions=3)
    result = validate_run_modes_config(config)
    assert result['modes']['standard']['runtime']['native']['frames'] == (219 if custom else 100)
    assert validate_run_modes_config(result) == result


def test_collector_resolves_once_and_rejects_changed_bytes(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.energy.config import resolve_collector_binding, EnergyDefaults, EnergySetup
    from onnx_splitpoint_tool.energy.collector import _collector_command, _energy_path_env
    binary = tmp_path / 'collector'
    binary.write_text('#!/bin/sh\nexit 0\n'); binary.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path))
    binding = resolve_collector_binding({'energy_defaults': {'collector_binary': 'collector'}})
    assert binding['collector_binary'] == str(binary)
    defaults = EnergyDefaults(collector_binary=str(binary), collector_sha256=binding['collector_sha256'])
    setup = EnergySetup(setup_id='test', urecs_address='127.0.0.1')
    monkeypatch.setenv('PATH', '/nowhere')
    args = dict(run_id='r8', window_id='w')
    assert _collector_command(defaults, setup, tmp_path, tmp_path/'command', 1, **args)[0] == str(binary)
    assert _energy_path_env()['URECS_RECEIVE_DIAGNOSTICS'] == '1'
    binary.write_text('changed')
    with pytest.raises(RuntimeError, match='bytes changed'):
        _collector_command(defaults, setup, tmp_path, tmp_path/'command', 1, **args)


def test_reviewed_install_is_idempotent_and_preserves_custom_fields(tmp_path, monkeypatch):
    import yaml
    from onnx_splitpoint_tool.energy import config
    registry, mirror = tmp_path/'hardware.yaml', tmp_path/'energy.yaml'
    source = tmp_path/'original'; source.write_bytes(b'approved collector'); source.chmod(0o755)
    monkeypatch.setattr(config, 'REVIEWED_COLLECTOR_SHA256', hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(config, 'default_registry_path', lambda: registry)
    monkeypatch.setattr(config, 'default_energy_config_file', lambda: mirror)
    payload = {'energy_defaults': {'collector_binary': 'urecs-data-collector', 'channel': 17}, 'custom': {'keep': True}}
    for path in (registry, mirror): path.write_text(yaml.safe_dump(payload)); path.chmod(0o640)
    result = config.install_reviewed_collector(source, tmp_path/'backups')
    assert len(result['changes']) == 2
    assert Path(result['collector_binary']).read_bytes() == source.read_bytes()
    for path in (registry, mirror):
        assert yaml.safe_load(path.read_text())['custom'] == payload['custom']
        assert path.stat().st_mode & 0o777 == 0o640
        assert yaml.safe_load((tmp_path/'backups'/path.name).read_text()) == payload
    assert config.install_reviewed_collector(source, tmp_path/'backups')['changes'] == []


def test_native_display_keeps_process_output_and_warnings(tmp_path):
    from onnx_splitpoint_tool.log_utils import NativeDisplayLog
    from onnx_splitpoint_tool.native_progress import run_streaming
    logs = []
    payload = json.dumps({'rows': [{'array': list(range(300))} for _ in range(20)]}, indent=2)
    command = [sys.executable, '-c', f'print({payload!r}); print("[WARN] preserved"); print("token=never_export")']
    diagnostic = tmp_path/'native.log'
    display = NativeDisplayLog(diagnostic, logs.append, label='test', command=command)
    result = run_streaming(command, label='test', line_callback=display)
    display.finish(returncode=result.returncode, elapsed_s=result.elapsed_s)
    reference = run_streaming(command, label='control', line_callback=lambda line: None)
    assert result.returncode == 0 and reference.stdout == result.stdout
    assert '[WARN] preserved' in '\n'.join(logs)
    assert len('\n'.join(logs)) < 2000
    assert 'never_export' not in diagnostic.read_text()
    assert '299' in diagnostic.read_text()


@pytest.mark.parametrize('rc', [0, 1, 124])
def test_ssh_probes_keep_full_result_only_in_diagnostic(tmp_path, monkeypatch, rc):
    from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport, HostConfig
    logs = []
    transport = SSHTransport(HostConfig(id='local', label='local', host='localhost', user='test'), log=logs.append)
    transport.diagnostics_dir = tmp_path
    output = 'FIRST_MARKER\n' + 'payload ' * 2000 + ('\nERROR: timeout\n' if rc else '\nok\n')
    def fake_capture(*a, **kw):
        kw['diagnostics']['command_argv'] = ['ssh', 'PAYLOAD'*1000]
        return rc, output
    monkeypatch.setattr(transport, '_run_capture', fake_capture)
    assert transport.run_read_only('PAYLOAD'*1000) == (rc, output)
    assert len('\n'.join(logs)) < 1500
    assert 'PAYLOAD' not in '\n'.join(logs)
    assert 'PAYLOAD'*1000 in next(tmp_path.glob('*.json')).read_text()
    assert json.loads(next(tmp_path.glob('*.json')).read_text())['output'] == output


def test_diagnostic_write_failure_keeps_short_failure(tmp_path):
    from onnx_splitpoint_tool.log_utils import NativeDisplayLog
    logs = []
    blocker = tmp_path/'file'; blocker.write_text('x')
    display = NativeDisplayLog(blocker/'diag', logs.append, label='test', command=['secret=hidden'])
    display('{"rows":' + 'x'*20000)
    display('[ERROR] workload failed')
    assert len('\n'.join(logs)) < 1000
    assert 'nicht schreibbar' in '\n'.join(logs)
    assert 'workload failed' in '\n'.join(logs)


def test_bearer_credentials_are_fully_redacted():
    from onnx_splitpoint_tool.log_utils import redact_diagnostic
    assert 'example_secret_value' not in redact_diagnostic('Authorization: Bearer example_secret_value')


def test_parallel_ssh_failures_preserve_separate_full_diagnostics(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport, HostConfig
    logs = []
    def probe(index):
        transport = SSHTransport(HostConfig(id=f'target{index}', label='test', host='localhost', user='test'), log=logs.append)
        transport.diagnostics_dir = tmp_path
        monkeypatch.setattr(transport, '_begin_remote_lease', lambda **kw: None)
        monkeypatch.setattr(transport, '_finish_remote_lease', lambda *a, **kw: False)
        monkeypatch.setattr(transport, '_run_capture', lambda *a, **kw: (124, f'ERROR target{index} timeout'))
        return transport.run('payload'*3000)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(probe, range(2)))
    assert all(rc == 70 and 'cleanup unproven' in output for rc, output in results)
    diagnostics = [json.loads(p.read_text()) for p in tmp_path.glob('*.json')]
    assert len(diagnostics) == 2
    assert {r['target_id'] for r in diagnostics} == {'target0', 'target1'}
    assert all(r['returncode'] == 70 and r['cleanup_proven'] is False for r in diagnostics)
    assert len('\n'.join(logs)) < 2000 and 'payload' not in '\n'.join(logs)


def test_original_gui_scalar_fixture_reproduces_then_resolves_budget_discrepancy():
    fixture=json.loads((Path(__file__).parent/'fixtures/v283_r8/original_gui_scalar_evidence.json').read_text())
    profile={'name':'original_scalar_fixture','execution_preset':fixture['preset'],
             'native_producers':fixture['native_producers']}
    assert profile['execution_preset']['effective']['native_performance_repetitions']==1
    assert resolve_native_execution_contract(profile)['repetitions']==3
    resolved,_=apply_run_mode(profile,config=default_run_modes_config())
    assert tuple(resolve_native_execution_contract(resolved)[k] for k in ('frames','warmup','repetitions'))==(100,10,1)
    assert resolved['execution_preset']['effective']['native_performance_repetitions']==1
    assert campaign_budget_policy(resolved['native_producers'])['max_retries']==1


def test_energy_display_never_adds_failed_attempts_to_valid_logical_repeats():
    from onnx_splitpoint_tool.workflow.evidence_status import _energy_accounting_projection
    fixture=json.loads((Path(__file__).parent/'fixtures/v283_r8/original_gui_scalar_evidence.json').read_text())['energy_original_counts']
    rows=[]
    for i in range(fixture['native_energy_rows']):
        row={'model':f'model{i}','case':'full','backend':'native_full_hailo8',
             'setup_id':'h8','comparison_backend':'hailo8',
             'energy_aggregate_valid_repeat_count':int(i<24),'energy_aggregate_requested_repeat_count':3}
        rows.append(row)
    # Old reports have no attempt counters: display unknown instead of deriving
    # attempts from log volume or mixing them with logical repetitions.
    old=_energy_accounting_projection({'present_expected_rows':rows},[],[],rows)
    assert old['valid_repetition_count']==24 and old['requested_repetition_count']==183
    assert old['collector_attempt_count'] is None
    for i,row in enumerate(rows):
        row['energy_collector_attempt_count']=342 if i==0 else 0
        row['energy_failed_collector_attempt_count']=318 if i==0 else 0
    current=_energy_accounting_projection({'present_expected_rows':rows},[],[],rows)
    assert (current['valid_repetition_count'],current['requested_repetition_count'],current['collector_attempt_count'],current['failed_collector_attempt_count'])==(24,183,342,318)


@pytest.mark.parametrize('transport', ['native','ssh'])
def test_finished_process_output_is_drained_despite_slow_observer(tmp_path, monkeypatch, transport):
    import time
    from onnx_splitpoint_tool.native_progress import run_streaming
    from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport, HostConfig
    expected=[f'line {index}' for index in range(300)]
    command=[sys.executable,'-c',f'print({chr(10).join(expected)!r})']
    seen=[]
    def slow(line):
        seen.append(line)
        time.sleep(.005)
    if transport=='native':
        result=run_streaming(command,timeout=1.0,line_callback=slow)
        assert result.returncode==0
        assert result.stdout=='\n'.join(expected)
    else:
        connection=SSHTransport(HostConfig(id='fake-ssh-leaf',label='local process',host='localhost',user='test'))
        connection.diagnostics_dir=tmp_path
        # Only the remote hardware process is replaced by an actual local
        # process. Transport drain/timeout/ownership logic remains real.
        monkeypatch.setattr(connection,'_ssh_cmd',lambda *a,**kw:command)
        assert connection.run_streaming('controlled fake hardware output',on_line=slow,timeout=1.0)==0
    assert seen==expected


@pytest.mark.parametrize('transport', ['native', 'ssh'])
def test_inherited_open_pipe_cannot_report_success(tmp_path, monkeypatch, transport):
    from onnx_splitpoint_tool.native_progress import run_streaming
    from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport, HostConfig
    from onnx_splitpoint_tool.process_control import ProcessTreeRegistry, bind_process_registry
    # The registered process owns this child. Its stdout deliberately outlives
    # a successful parent; regular product ownership performs bounded cleanup.
    command = [sys.executable, '-c',
               "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)']); print('parent done',flush=True); time.sleep(.1)"]
    seen = []
    registry = ProcessTreeRegistry()
    lease_finishes = []
    try:
        with bind_process_registry(registry):
            if transport == 'native':
                rc = run_streaming(command, timeout=5, line_callback=seen.append).returncode
            else:
                connection = SSHTransport(HostConfig(id='fake-ssh-pipe', label='local process', host='localhost', user='test'))
                connection.diagnostics_dir = tmp_path
                monkeypatch.setattr(connection, '_ssh_cmd', lambda *a, **kw: command)
                monkeypatch.setattr(connection, '_finish_remote_lease', lambda operation, **kw: lease_finishes.append(kw['abnormal']) or True)
                rc = connection.run_streaming('controlled inherited pipe', on_line=seen.append, timeout=5)
                assert lease_finishes == [True]
        assert rc == 70
        assert any('pipe_drain_incomplete' in line for line in seen)
        registry.assert_quiescent()
    finally:
        registry.terminate_all(grace_s=.2)
