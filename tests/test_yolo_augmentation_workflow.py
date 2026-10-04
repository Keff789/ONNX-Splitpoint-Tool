"""Bounded workflow guards and real local controller admission; no hardware."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.generic_completion_plan import SETUPS
from onnx_splitpoint_tool.workflow import yolo_augmentation as workflow
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.process_control import controller_local_activity, workflow_resource_options
from tests.test_ap06_resource_journal import journal_rig  # actual broker/locks; external fixture storage


@pytest.fixture
def frozen(tmp_path):
    original = tmp_path / 'closed-run'; original.mkdir()
    output = tmp_path / 'new-cohort'; (output / 'plan').mkdir(parents=True)
    models = {'yolo11l': {'primary_cases': ['b120'], 'reserve_cases': []},
              'yolo26s': {'primary_cases': ['b038'], 'reserve_cases': []},
              'yolov7_paper': {'primary_cases': ['b044', 'b066'], 'reserve_cases': []}}
    rows = [{'model_id': model, 'case_id': case, 'setup_id': setup, 'backend': backend,
             'selection_role': 'primary', 'part2_external_inputs': ['boundary'],
             'warmup_count': 100, 'task_count': 1000, 'repetitions': 3,
             'case_key': '|'.join((model, case, setup))}
            for model, data in models.items() for case in data['primary_cases']
            for setup, backend in SETUPS.items()]
    base = tmp_path / 'base-execution-plan.json'
    base.write_text(json.dumps({'rows': [{'model_id': 'yolo11l', 'case_id': 'b003',
                                         'setup_id': setup} for setup in SETUPS]}))
    plan = {'schema': 'onnx-splitpoint/yolo-min3-augmentation-plan',
            'cohort_id': 'yolo_min3_capability_augmentation', 'frozen': True, 'force_rebuild': False,
            'rows': rows, 'models': models, 'primary_count': 12,
            'base_execution_plan': str(base), 'original_run': str(original),
            'output_root': str(output), 'augmentation_run': str(output / 'runs/new-run'),
            'base_completion_root': str(tmp_path / 'base-completion'),
            'start_case_keys': [row['case_key'] for row in rows[:3]]}
    path = output / 'plan/augmentation-plan.json'; path.write_text(json.dumps(plan))
    return path, plan


def test_frozen_scope_reads_only_its_exact_new_case_plan(frozen):
    path, plan = frozen
    before = path.read_bytes()
    assert workflow.load_frozen_plan(path) == plan
    assert path.read_bytes() == before
    assert len(plan['rows']) == 12
    assert not Path(plan['augmentation_run']).exists()


@pytest.mark.parametrize('mutation', ['cohort', 'force', 'unfrozen', 'duplicate', 'distribution',
    'setup', 'backend', 'multiinput', 'warmup', 'tasks', 'reps', 'repeat_base', 'original_output', 'base_output', 'outside_new_run', 'same_run_id'])
def test_plan_drift_is_rejected_before_workflow_or_hardware(frozen, mutation):
    path, plan = frozen
    if mutation == 'cohort': plan['cohort_id'] = 'other'
    elif mutation == 'force': plan['force_rebuild'] = True
    elif mutation == 'unfrozen': plan['frozen'] = False
    elif mutation == 'duplicate': plan['rows'][1] = copy.deepcopy(plan['rows'][0])
    elif mutation == 'distribution': plan['rows'][4]['model_id'] = 'yolo11l'
    elif mutation == 'setup': plan['rows'][4]['setup_id'] = 'outside_setup'
    elif mutation == 'backend': plan['rows'][4]['backend'] = 'deepx' if plan['rows'][4]['backend'] != 'deepx' else 'hailo8'
    elif mutation == 'multiinput': plan['rows'][4]['part2_external_inputs'].append('skip')
    elif mutation == 'warmup': plan['rows'][4]['warmup_count'] = 10
    elif mutation == 'tasks': plan['rows'][4]['task_count'] = 100
    elif mutation == 'reps': plan['rows'][4]['repetitions'] = 1
    elif mutation == 'repeat_base':
        plan['models']['yolo11l']['primary_cases'] = ['b003']
        for row in plan['rows'][:3]: row['case_id'] = 'b003'
    elif mutation == 'original_output': plan['output_root'] = plan['original_run']
    elif mutation == 'base_output':
        plan['output_root'] = plan['base_completion_root']
        plan['augmentation_run'] = str(Path(plan['output_root']) / 'runs/new-run')
    elif mutation == 'outside_new_run': plan['augmentation_run'] = str(path.parent / '../../outside')
    elif mutation == 'same_run_id': plan['augmentation_run'] = str(Path(plan['output_root']) / Path(plan['original_run']).name)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError): workflow.load_frozen_plan(path)


def options(plan):
    run = Path(plan['augmentation_run'])
    return {'out': str(run.parent), 'run_id': run.name,
            'profile': str(Path(plan['output_root']) / 'plan/augmentation-profile.yaml'),
            'execution_mode': 'generate_and_run', 'window_method_validation_probe_enabled': False,
            'hailo_build_full': False, 'hailo_build_part2': False}


@pytest.mark.parametrize('field,value', [('out', '/tmp/unbound-run'), ('run_id', 'closed-run'),
    ('profile', '/tmp/unbound-profile.yaml'), ('resume', True), ('resume_missing', True),
    ('resume_missing_full_quality_only', True), ('force_stage', ['generate_benchmark_set']),
    ('hailo_force_build', True), ('hailo_build_full', True), ('hailo_build_part2', True),
    ('result_sources', ['old-results.json']), ('execution_mode', 'contracts_only'),
    ('energy_enabled', True), ('window_method_validation_probe_enabled', True)])
def test_options_cannot_redirect_normal_writers_or_expand_work(frozen, field, value):
    _, plan = frozen
    prepared = options(plan)
    accepted = workflow.validated_workflow_options(plan, prepared)
    assert accepted.run_id == 'new-run'
    prepared[field] = value
    with pytest.raises(ValueError): workflow.validated_workflow_options(plan, prepared)


def test_preservation_snapshot_reads_real_central_quality_path(frozen):
    _, plan = frozen
    original = Path(plan['original_run'])
    central = original / 'quality_management/central_quality_summary.json'
    central.parent.mkdir(); central.write_text('{"original_548":true}')
    base = Path(plan['base_completion_root']); base.mkdir()
    before = central.read_bytes()
    inventory = workflow.preservation_inventory(plan)
    assert str(central) in inventory
    assert central.read_bytes() == before


def test_rejected_second_entry_preserves_previous_terminal_and_before_inventory(frozen, monkeypatch):
    path, plan = frozen
    output = Path(plan['output_root'])
    (output / 'plan/workflow-options.json').write_text(json.dumps(options(plan)))
    for name in ('production-start.json', 'preservation-before.json', 'terminal-status.json', 'STATUS.txt'):
        (output / name).write_text('preserved original ' + name)
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    monkeypatch.setattr(workflow.completion, 'observe_owners', lambda *a: pytest.fail('second entry reached hardware owner probe'))
    with pytest.raises(FileExistsError, match='already entered'):
        workflow.execute(path)
    assert {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()} == before


def test_bad_workflow_options_stop_before_owner_probe_or_start_marker(frozen, monkeypatch):
    path, plan = frozen
    prepared = options(plan); prepared['out'] = plan['original_run']
    output = Path(plan['output_root'])
    (output / 'plan/workflow-options.json').write_text(json.dumps(prepared))
    monkeypatch.setattr(workflow.completion, 'observe_owners', lambda *a: pytest.fail('invalid metadata reached hardware owner probe'))
    with pytest.raises(ValueError, match='workflow options differ'):
        workflow.execute(path)
    assert not (output / 'production-start.json').exists()


def test_native_ok_metadata_without_normal_generic_bindings_cannot_start_completion(frozen, monkeypatch):
    path, plan = frozen
    run = Path(plan['augmentation_run']); (run / 'reports/native_validation').mkdir(parents=True)
    (run / 'profile.yaml').write_text('{}')
    (run / 'run_manifest.json').write_text(json.dumps({'run_id': run.name}))
    row = plan['rows'][0]
    summary = run / 'reports/native_producer_combined_summary.json'
    summary.write_text(json.dumps({'rows': [{'model': row['model_id'], 'case': row['case_id'],
        'setup_id': row['setup_id'], 'status': 'ok'}]}))
    validation = run / 'reports/native_validation/native_producer_validation_summary.json'
    validation.write_text('{"rows":[]}')
    monkeypatch.setattr(workflow.completion, 'transport_for', lambda *a, **kw: pytest.fail('metadata-only Native row reached hardware'))
    with pytest.raises(ValueError, match='normal reporter could not join'):
        workflow.completion_barrier(path, summary, validation, row['case_key'])
    assert not (Path(plan['output_root']) / 'completion/execution-plan.json').exists()


_RESOURCE_CHILD = r'''
import json,sys
from pathlib import Path
from onnx_splitpoint_tool.workflow.yolo_augmentation import completion_setup_resources
from onnx_splitpoint_tool.process_control import controller_local_activity, workflow_resource_options
run, result, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
class OuterRemoteState:
    cancelled = False
    def active_count(self): return 1 if mode == 'pending' else 0
    def poison(self): self.cancelled = True
registry = OuterRemoteState()
try:
    with completion_setup_resources(run, 'setup-a', {}, registry, 'm|b001|setup-a'):
        opts = workflow_resource_options()
        assert opts['controller_setup_id'] == 'setup-a'
        assert Path(opts['controller_attempt_dir']).is_relative_to(run)
        # This is the real controller transfer admission used by SCP. No SDK,
        # SSH, inference or collector is run by this local resource test.
        with controller_local_activity('transfer'):
            (result / 'transfer-admitted.json').write_text(json.dumps(opts))
    (result / 'finished.json').write_text(json.dumps({'status':'released'}))
except BaseException as exc:
    (result / 'error.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc),'poisoned':registry.cancelled}))
    raise
'''


@pytest.mark.parametrize('mode', ['clean', 'pending'])
def test_actual_callback_child_uses_parent_dut_and_transfer_broker_without_double_lock(journal_rig, mode):
    rig = journal_rig
    result = rig.run / ('callback-' + mode); result.mkdir()
    env = dict(os.environ, **rig.journal.environment(), PYTHONDONTWRITEBYTECODE='1',
               ONNX_SPLITPOINT_CONTROLLER_RESOURCES='required')
    with (result / 'child.log').open('w') as log:
        child = subprocess.Popen([sys.executable, '-B', '-c', _RESOURCE_CHILD,
            str(rig.run), str(result), mode], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    rig.registry.register(child, label='yolo-augmentation-local-resource-test')
    rig.children.append((child, result))
    rig.until(lambda: child.poll() is not None)
    rig.broker.tick()
    assert (result / 'transfer-admitted.json').exists(), (result / 'child.log').read_text()
    records = [value['row'] for value in rig.broker.operations.values()]
    assert {row['purpose'] for row in records} == {'dut_job', 'transfer'}
    assert not any(row.get('continuation') for row in records)
    if mode == 'clean':
        assert child.returncode == 0
        assert (result / 'finished.json').exists()
        assert rig.broker.stopped_sources == set()
        assert all(row['terminal'] for row in rig.broker.operations.values())
    else:
        assert child.returncode != 0
        error = json.loads((result / 'error.json').read_text())
        assert error['poisoned'] is True
        assert 'cleanup remains unresolved' in error['error']
        assert 'dut:' + rig.root.name in rig.broker.stopped_sources
        evidence = rig.run / 'reports/augmentation_completion/setup-a/m_b001_setup-a/cleanup-unproven.json'
        assert json.loads(evidence.read_text())['broker_claim_left_owned_until_process_exit'] is True
        dut = next(value for value in rig.broker.operations.values() if value['row']['purpose'] == 'dut_job')
        reply = rig.journal._read_json_exact(rig.journal.directory / (dut['row']['operation_id'] + '.resource-reply.json'))
        assert reply['state'] == 'STOP' and reply['reason'] == 'controller_resource_owner_lost'


def test_standalone_callback_uses_real_physical_lock_and_fences_unproven_cleanup(tmp_path, monkeypatch):
    monkeypatch.delenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES', raising=False)
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path', lambda: tmp_path / 'locks/workflow.lock')
    class OuterRemoteState:
        cancelled = False
        def active_count(self): return 1
        def poison(self): self.cancelled = True
    registry = OuterRemoteState()
    run = tmp_path / 'run'; run.mkdir()
    target = {'physical_host_id': 'fixture-' + tmp_path.name,
              'remote': {'host': '127.0.0.1', 'user': 'fixture'}}
    with pytest.raises(RuntimeError, match='cleanup remains unresolved'):
        with workflow.completion_setup_resources(run, 'setup-a', target, registry, 'm|b001|setup-a'):
            assert workflow_resource_options()['controller_setup_id'] == 'setup-a'
    lock = run_control.EvaluationRunLock.for_resource(workflow.completion.physical_dut_key(target), owner={})
    assert lock.quarantine_path.is_file()
    assert registry.cancelled is True
    assert not workflow_resource_options().get('controller_setup_id')


@pytest.fixture
def completion_execution(tmp_path, monkeypatch):
    """Normal execution/parser/report chain, controlled external transport only."""
    import tarfile
    from tests.test_generic_completion_dispatch import LocalTransport, prepared_case
    monkeypatch.delenv('ONNX_SPLITPOINT_CONTROLLER_RESOURCES', raising=False)
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path', lambda: tmp_path / 'locks/workflow.lock')
    output = tmp_path / 'cohort'; output.mkdir()
    row = prepared_case(tmp_path)
    row['result_path'] = str(output / 'completion/results' / row['case_output_relative'] / 'generic_completion_results.json')
    original_reports = Path(row['original_run']) / 'reports'; original_reports.mkdir(parents=True)
    (original_reports / 'native_host_telemetry_summary.json').write_text('{"captures": []}')
    with tarfile.open(output / 'runner-bundle.tar.gz', 'w:gz'):
        pass
    plan = {'output_root': str(output), 'augmentation_run': str(output / 'runs/new-run'), 'primary_count': 12}
    accumulated = {'rows': [row], 'original_run': row['original_run']}
    target = {'id': row['setup_id'], 'accelerator': 'hailo8', 'physical_host_id': 'fixture-' + tmp_path.name,
              'remote': {'host': '127.0.0.1', 'user': 'fixture'}}
    transport = LocalTransport()
    monkeypatch.setattr(workflow.completion, 'transport_for', lambda *args, **kwargs: transport)
    def run():
        return workflow._run_completion_requests(plan, accumulated, {workflow.key(row)},
                                                  {row['setup_id']: target}, 'remaining')
    return output, row, accumulated, transport, run


def test_callback_reports_actual_partial_repetition_and_error_after_real_run_case(completion_execution):
    from onnx_splitpoint_tool.generic_completion_reporting import load_valid_repetitions
    output, row, accumulated, transport, run = completion_execution
    transport.stop_at = 2
    with pytest.raises(RuntimeError, match='Generic case stopped'):
        run()
    assert [index for _, index in transport.calls] == [1, 2]
    assert [rep['repetition_index'] for rep in load_valid_repetitions(row['result_path'], row)] == [1]
    summary = json.loads((output / 'completion/reports/generic_completion_summary.json').read_text())
    assert summary['status_counts']['partial'] == 1 and summary['valid_repetitions'] == 1
    assert summary['completed_task_count'] == 1000 and summary['technical_pairs'] == 0
    persisted = json.loads((output / 'completion/execution-plan.json').read_text())['rows'][0]
    assert persisted['execution_status'] == 'failed' and persisted['dispatch_attempted'] is True
    assert 'Generic case stopped' in persisted['execution_error']
    observation = json.loads((output / 'completion/reports/generic_completion_observations.json').read_text())['rows'][0]
    assert observation['status'] == 'partial' and observation['execution_error'] == persisted['execution_error']
    assert json.loads((output / 'completion/cleanup-status.json').read_text())['active_owned_remote_operations'] == 0


def test_callback_preflight_failure_without_result_is_blocked_not_unstarted(completion_execution, monkeypatch):
    output, row, _, transport, run = completion_execution
    normal_read = transport.run_read_only
    def failed_environment(command, **kwargs):
        if 'GENERIC_RUNTIME_ENVIRONMENT' in command:
            return 1, 'controlled unavailable interpreter'
        return normal_read(command, **kwargs)
    monkeypatch.setattr(transport, 'run_read_only', failed_environment)
    with pytest.raises(RuntimeError, match='dependency inspection'):
        run()
    assert transport.calls == [] and not Path(row['result_path']).exists()
    summary = json.loads((output / 'completion/reports/generic_completion_summary.json').read_text())
    assert summary['status_counts']['blocked'] == 1 and summary['status_counts']['unstarted'] == 0
    assert row['execution_status'] == 'blocked' and not row.get('dispatch_attempted')


def test_remaining_callback_preserves_completed_rows_without_any_transport(completion_execution, monkeypatch):
    output, row, _, transport, run = completion_execution
    assert run() == 0
    before = copy.deepcopy(row)
    raw = Path(row['result_path']).read_bytes()
    snapshots = {p: p.read_bytes() for p in (output / 'completion/configuration').rglob('*.json')}
    def forbidden(*args, **kwargs):
        raise AssertionError('completed cases must not capture or contact hardware')
    monkeypatch.setattr(workflow.completion, 'transport_for', forbidden)
    assert run() == 0
    assert row == before and Path(row['result_path']).read_bytes() == raw
    assert {p: p.read_bytes() for p in (output / 'completion/configuration').rglob('*.json')} == snapshots
    assert [index for _, index in transport.calls] == [1, 2, 3]


def test_completion_continuation_uses_new_runner_without_replacing_old_bundle(completion_execution):
    import shlex
    output, row, _, transport, run = completion_execution
    remote = Path(row['remote_root'])
    remote.mkdir(parents=True)
    old_bundle = remote / 'runner-bundle.tar.gz'
    old_bundle.write_bytes(b'original failed attempt runner bundle')
    original_config = copy.deepcopy(row['runtime_config'])
    assert run() == 0
    assert old_bundle.read_bytes() == b'original failed attempt runner bundle'
    runner_root = remote / 'attempts' / output.name
    assert (runner_root / 'runner-bundle.tar.gz').read_bytes() == (output / 'runner-bundle.tar.gz').read_bytes()
    attempts = json.loads(Path(row['result_path']).with_name('dispatch_attempts.json').read_text())['attempts']
    assert [item['repetition_index'] for item in attempts] == [1, 2, 3]
    for item in attempts:
        argv = shlex.split(item['command'])
        assert str(runner_root / 'runner/run_split_onnxruntime.py') in argv
        assert argv[2] == str(runner_root)
        assert argv[argv.index('--generic-completion-config') + 1].startswith(str(remote / 'configs'))
        assert argv[argv.index('--out-dir') + 1] == str(remote / 'results' / row['case_output_relative'])
    assert row['runtime_config'] == original_config
    # Completed cases never upload even a newer runner or repeat valid work.
    (output / 'runner-bundle.tar.gz').write_bytes(b'later unused bundle')
    assert run() == 0
    assert len(transport.calls) == 3


def test_completion_attempt_runner_conflict_still_blocks_dispatch(completion_execution):
    output, row, _, transport, run = completion_execution
    bundle = Path(row['remote_root']) / 'attempts' / output.name / 'runner-bundle.tar.gz'
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b'different bytes already at this exact attempt')
    with pytest.raises(ValueError, match='existing staging destination differs'):
        run()
    assert transport.calls == []
    assert bundle.read_bytes() == b'different bytes already at this exact attempt'


def test_partial_callback_reentry_retains_old_conditions_and_only_runs_missing_repetitions(completion_execution):
    output, row, _, transport, run = completion_execution
    transport.stop_at = 2
    with pytest.raises(RuntimeError):
        run()
    comparison = copy.deepcopy(row['runtime_environment_comparison'])
    first_snapshot = Path(comparison['current_snapshot'])
    first_bytes = first_snapshot.read_bytes()
    first_rep = copy.deepcopy(json.loads(Path(row['result_path']).read_text())['repetitions'][0])
    transport.stop_at = None
    assert run() == 0
    assert [index for _, index in transport.calls] == [1, 2, 2, 3]
    assert row['runtime_environment_comparison'] == comparison
    assert first_snapshot.read_bytes() == first_bytes
    assert json.loads(Path(row['result_path']).read_text())['repetitions'][0] == first_rep
    history = row['completion_environment_observations']
    assert len(history) == 2 and history[1]['preserved_repetition_indices'] == [1]
    assert history[1]['pending_repetition_indices'] == [2, 3]
    assert history[0]['comparison']['current_snapshot'] != history[1]['comparison']['current_snapshot']
    assert row['execution_status'] == 'completed' and 'execution_error' not in row
    assert len(row['execution_error_history']) == 1
    summary = json.loads((output / 'completion/reports/generic_completion_summary.json').read_text())
    assert summary['valid_repetitions'] == 3 and summary['technical_pairs'] == 1


def test_remaining_callback_counts_real_completions_and_updates_incremental_plan(completion_execution, monkeypatch, tmp_path):
    from tests.test_generic_completion_dispatch import prepared_case
    output, first, accumulated, transport, run = completion_execution
    assert run() == 0
    before = copy.deepcopy(first)
    second = prepared_case(tmp_path, index=2)
    second['result_path'] = str(output / 'completion/results' / second['case_output_relative'] / 'generic_completion_results.json')
    accumulated['rows'].append(second)
    plan = {'output_root': str(output), 'augmentation_run': str(output / 'runs/new-run'), 'primary_count': 12}
    target = {'id': second['setup_id'], 'accelerator': 'hailo8', 'physical_host_id': 'fixture-' + tmp_path.name,
              'remote': {'host': '127.0.0.1', 'user': 'fixture'}}
    observed_indices = []
    real_run_case = workflow.completion.run_case
    def observe(row, transport, output, index, **kwargs):
        observed_indices.append((index, kwargs['total']))
        return real_run_case(row, transport, output, index, **kwargs)
    monkeypatch.setattr(workflow.completion, 'run_case', observe)
    assert workflow._run_completion_requests(plan, accumulated, {workflow.key(first), workflow.key(second)},
        {second['setup_id']: target}, 'remaining') == 0
    assert observed_indices == [(2, 12)] and first == before
    persisted = json.loads((output / 'completion/execution-plan.json').read_text())
    assert persisted['row_count'] == 2 and persisted['planned_repetitions'] == 6
    assert persisted['planned_measured_completions'] == 6000 and persisted['authorized_primary_case_count'] == 12
