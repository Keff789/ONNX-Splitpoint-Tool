"""One authorized GUI-source diagnostic gates the existing energy remainder."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import fcntl
import json
import os

import pytest

from test_physical_resource_recovery import recovery_case, _retry_evidence
from test_yolo_rest_capture_recovery import rest_case
from test_yolo_rest_energy_resume import run_fence_case
from test_yolo_augmentation_rest_blocks import phase_fixture, actual_plan
from onnx_splitpoint_tool.workflow import yolo_augmentation_rest as rest
from onnx_splitpoint_tool.energy import task_budget
from onnx_splitpoint_tool.remote import resource_recovery


def test_diagnostic_command_uses_normal_exact_one_capture_cli(actual_plan, tmp_path):
    run = Path(actual_plan['augmentation_run']); output = tmp_path / 'one_session'
    cfg = rest.io.read_json(run / 'reports/native_producer_stage_config.json')
    command, row = rest.source_test_command(run, cfg, output, run / 'reports/diagnostic')
    assert command[3] == 'test-setup'
    assert command[command.index('--setup-id') + 1] == 'orin_nx_hailo10_01'
    assert command[command.index('--campaign-budget-file') + 1] == str(run / 'energy_task_budget.json')
    assert command[command.index('--campaign-repeats') + 1] == '1'
    assert command[command.index('--campaign-max-retries') + 1] == '0'
    assert command[command.index('--campaign-max-transport-failures') + 1] == '20'
    assert command[command.index('--sleep-s') + 1] == '2'
    assert json.loads(row) == ['energy_setup_test', output.name, 'orin_nx_hailo10_01']


def test_budget_handoff_holds_all_exact_released_locks_without_rewriting_them(run_fence_case, monkeypatch):
    case, plan, output, lock, fence = run_fence_case
    case.kwargs.pop('operator_evidence')
    case.kwargs['retry_authorization'] = _retry_evidence(case)
    resource_recovery.recover_capture_resources(**case.kwargs)
    original = {resource: lease.lock_path.read_bytes() for resource, lease in case.locks.items()}
    calls = []
    def install(path, *, capture_request, recovery_reply, new_parent):
        for lease in case.locks.values():
            with lease.lock_path.open('r+b') as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert path == case.run_dir / 'energy_task_budget.json'
        assert new_parent['run_id'] == case.run_dir.name and new_parent['session_id'] == output.name
        assert new_parent['pid'] == os.getpid() and new_parent['start_ticks']
        assert capture_request['operation_id'] == recovery_reply['operation_id']
        calls.append(new_parent)
        return {'new_parent': new_parent}
    monkeypatch.setattr(task_budget, 'install_authorized_source_reentry', install)
    assert rest._recover_rest_run_fence(plan, case.rest_attempt, output, lock, fence, source_test=True)
    assert len(calls) == 1
    assert all(lease.lock_path.read_bytes() == original[resource] for resource, lease in case.locks.items())
    assert rest.io.read_json(output / 'reports/source-budget-handoff.json')['new_parent'] == calls[0]


def test_operator_action_alone_cannot_authorize_unknown_source_diagnostic(run_fence_case, monkeypatch):
    case, plan, output, lock, fence = run_fence_case
    resource_recovery.recover_capture_resources(**case.kwargs)
    monkeypatch.setattr(task_budget, 'install_authorized_source_reentry', lambda *args, **kwargs: pytest.fail('no authorization'))
    with pytest.raises(ValueError, match='explicit one-attempt retry authorization'):
        rest._recover_rest_run_fence(plan, case.rest_attempt, output, lock, fence, source_test=True)


@pytest.mark.parametrize('diagnostic_passes', [False, True])
def test_rest_energy_follows_only_a_passed_diagnostic_in_same_owned_session(phase_fixture, monkeypatch, diagnostic_passes):
    output, _, delivery, _ = phase_fixture
    previous = output.parent / 'old'; block = output.parent / 'old_block'
    descriptor = dict(previous_attempt=str(previous), previous_results=str(block / 'partial.json'),
        previous_block=str(block), inventory={'block1_energy': [('model', 'case', 'setup')]})
    monkeypatch.setattr(rest, 'energy_continuation', lambda *args: descriptor)
    monkeypatch.setattr(rest, '_energy_delivery_roots', lambda *args: [block])
    phases = []
    @contextmanager
    def owned(*args, **kwargs):
        assert kwargs == {'previous_attempt': str(previous), 'source_test': True}
        phases.append('owned')
        def execute(command, label, *args, **kwargs):
            phases.append(label)
            raise RuntimeError('controlled stop before remainder capture')
        yield execute, {}, SimpleNamespace(active_count=lambda: 0), None
    monkeypatch.setattr(rest, 'owned_execution', owned)
    def diagnostic(*args):
        phases.append('normal-source-diagnostic')
        if not diagnostic_passes: raise ValueError('source completion unknown')
    monkeypatch.setattr(rest, '_run_source_test', diagnostic)
    monkeypatch.setattr(rest, 'energy_command', lambda *args, **kwargs: ['normal-energy'])
    monkeypatch.setattr(rest, 'singlecase_command', lambda *args: pytest.fail('no premature Native'))
    assert rest.finish_rest('plan.json', output, include_b066=True,
        previous_energy_attempt=previous, source_test=True) == 1
    assert phases == ['owned', 'normal-source-diagnostic'] + (['block1-remaining-energy'] if diagnostic_passes else [])
    assert len(delivery['energy_results']) == 1


def test_one_start_guard_is_scoped_only_to_diagnostic_child(actual_plan, tmp_path, monkeypatch):
    run = Path(actual_plan['augmentation_run']); output = tmp_path / 'attempt'; output.mkdir()
    cfg = rest.io.read_json(run / 'reports/native_producer_stage_config.json')
    observed = []
    def execute(command, label, timeout, *, extra_env):
        guard = Path(extra_env['ONNX_SPLITPOINT_TEST_START_JOURNAL'])
        assert label == 'source-test' and timeout == 300
        observed.append(guard)
        task_budget.reserve_controlled_test_start('collector', tmp_path / 'source_test/run_000', journal_path=guard)
        with pytest.raises(RuntimeError):
            task_budget.reserve_controlled_test_start('collector', tmp_path / 'source_test/run_001', journal_path=guard)
    monkeypatch.setattr(rest, '_verify_source_test', lambda *args: {'status': 'passed'})
    before = os.environ.get('ONNX_SPLITPOINT_TEST_START_JOURNAL')
    rest._run_source_test(run, cfg, output, tmp_path, execute, SimpleNamespace(active_count=lambda: 0))
    assert len(observed) == 1 and os.environ.get('ONNX_SPLITPOINT_TEST_START_JOURNAL') == before
    with pytest.raises(FileExistsError):
        rest._run_source_test(run, cfg, output, tmp_path, execute, SimpleNamespace(active_count=lambda: 0))
    assert len(observed) == 1


@pytest.mark.parametrize('argv', [ ['--execute', 'plan.json', '--test-recovered-energy-source'],
    ['--finish-rest', 'plan.json', '--test-recovered-energy-source'] ])
def test_diagnostic_cli_requires_explicit_rest_predecessor(argv):
    with pytest.raises(SystemExit) as exc:
        rest.augmentation.main(argv)
    assert exc.value.code == 2


def test_normal_cli_collector_broker_and_source_gate_then_stored_evidence_negatives(tmp_path, monkeypatch):
    import yaml
    from test_yolo_energy_setup_budget import _rig
    from scripts import energy_measurement_cli as cli
    from onnx_splitpoint_tool.workflow import run_control
    from onnx_splitpoint_tool.remote.process_lease import ControllerResourceBroker
    rig = _rig(tmp_path, monkeypatch, ['success'])
    rig.setup.setup_id = setup_id = 'orin_nx_hailo10_01'
    # The existing software collector fixture predates the R6 window alias.
    # Declare it before the ordinary writer emits/hash-binds any samples.
    fake = tmp_path / 'fake_collector.py'
    fake.write_text(fake.read_text().replace("'dropped_samples': 0,", "'dropped_samples': 0, 'dropped_samples_in_window': 0,"))
    registry_path = tmp_path / 'hardware.yaml'
    registry_path.write_text(yaml.safe_dump({'schema_version': 1, 'energy_defaults': rig.defaults.to_dict(),
        'hardware_setups': [{'id': setup_id, 'label': 'Software diagnostic',
            'host': {'address': '127.0.0.2', 'user': 'fixture', 'port': 22, 'base_dir': '/unused'},
            'energy': {'enabled': True, 'urecs_address': rig.setup.urecs_address}}]}))
    cfg = {'_workflow_context': {'hardware_setups_file': str(registry_path)},
        'energy': {'task_budget': {'enabled': True, 'max_retries': 2, 'max_transport_failures': 20}}}
    run = tmp_path / 'run'; output = tmp_path / 'new_parent'; output.mkdir()
    rest.io.write_json(run / 'jobs/workflow_control.json', {})
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path', lambda: tmp_path / 'locks/interlock')
    monkeypatch.setattr(rest.io, 'bound_targets', lambda *args: {setup_id: {'host': '127.0.0.2'}})
    monkeypatch.setattr(rest.io, 'observe_owners', lambda *args: None)  # Device owner probe leaf only.
    broker_objects = []
    original = ControllerResourceBroker.__init__
    def broker_init(self, *args, **kwargs):
        original(self, *args, **kwargs); broker_objects.append(self)
    monkeypatch.setattr(ControllerResourceBroker, '__init__', broker_init)
    work = run / 'reports/continuation'; diagnostic = work / 'source_test'
    with rest.owned_execution({'augmentation_run': str(run), 'rows': []}, output, cfg) as (_, env, registry, cancel):
        for key, value in env.items(): monkeypatch.setenv(key, value)
        def execute(command, label, timeout, *, extra_env):
            with monkeypatch.context() as child:
                for key, value in extra_env.items(): child.setenv(key, value)
                assert cli.main(command[3:]) == 0
        proof = rest._run_source_test(run, cfg, output, work, execute, registry)
        assert proof['status'] == 'passed' and proof['physical_starts'] == 1
        assert rig.counts()['collector'] == 1 and not broker_objects[0].stopped_sources
        command, row_id = rest.source_test_command(run, cfg, output, diagnostic)
        guard = output / 'source-test-starts.json'
        def verify(): return rest._verify_source_test(run, cfg, output, diagnostic, row_id, guard)
        run_dir = diagnostic / 'run_000'
        row = rest.io.read_json(diagnostic / 'energy_summary.json')['runs'][0]
        # Each mutation is applied to preserved software evidence, never to a
        # production row or positive flag. No second collector is started.
        trace = Path(row['command_window_trace_binding']['trace_path'])
        trace_bytes = trace.read_bytes(); trace.write_bytes(b'corrupt trace')
        with pytest.raises(ValueError, match='stored sample/marker/result binding'): verify()
        trace.write_bytes(trace_bytes)
        log = run_dir / 'collector_stdout.log'; data = log.read_bytes()
        log.write_bytes(data.replace(b'verified=true', b'verified=false'))
        with pytest.raises(ValueError, match='closure evidence'): verify()
        log.write_bytes(data)
        requests = list((output / 'remote-process-leases').glob('*.resource-request.json'))
        request = next(rest.io.read_json(path) for path in requests if rest.io.read_json(path).get('purpose') == 'capture')
        reply_path = output / 'remote-process-leases' / (request['operation_id'] + '.resource-reply.json')
        raw = reply_path.read_bytes(); changed = json.loads(raw); changed['state'] = 'STOP'; rest.io.write_json(reply_path, changed)
        with pytest.raises(ValueError, match='resource release'): verify()
        reply_path.write_bytes(raw)
        changed = json.loads(raw); changed['session_id'] = 'foreign'; rest.io.write_json(reply_path, changed)
        with pytest.raises(ValueError, match='resource release'): verify()
        reply_path.write_bytes(raw)
        assert verify()['status'] == 'passed' and rig.counts()['collector'] == 1
