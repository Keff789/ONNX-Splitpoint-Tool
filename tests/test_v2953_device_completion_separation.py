"""Actual Split Writer -> independent safety and strict result Reader."""
from contextlib import nullcontext
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.native_deepx_async import NativeExecutionLifecycle
from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native
from test_v2952_split_reference_completion import actual_worker_report
from test_v2951_deepx_lifecycle import phase_prepared
from test_v295_numerical_product_acceptance import IDENTITY, split_contract, valid_split_numerical_raw


@pytest.fixture
def worker_raw(tmp_path, monkeypatch):
    code, raw, _ = actual_worker_report(tmp_path, monkeypatch)
    assert code == 0
    return raw


def test_actual_missing_redundant_field_proves_resource_completion_without_rewriting_report(worker_raw):
    raw = copy.deepcopy(worker_raw)
    del raw['deepx_numerical_preflight']['serial_reference_calls_completed']
    before = copy.deepcopy(raw)
    safety = numerical.device_completion_safety(raw)
    assert safety['device_completion_proven'] is True
    assert safety['report_contract_complete'] is False
    assert safety['evidence_gaps'] == ['serial_reference_calls_completed_missing']
    assert safety['independent_completion_proven'] is True
    assert raw == before
    # Scientific validation stays stricter than device completion.
    prepared = {'identity': IDENTITY, 'source_contract': split_contract()}
    # Source artifact bindings are fixture leaves; the actual writer payload
    # and both real readers remain active.
    bound = copy.deepcopy(raw)
    bound['case'] = IDENTITY['case']
    bound['native_command_contract'] = split_contract(warmup=0, frames=64)
    with pytest.raises(ValueError, match='report contract is incomplete'):
        numerical.validate_numerical_raw(bound, prepared)


@pytest.mark.parametrize('change', ['base_missing', 'base_mismatch', 'ref_mismatch', 'boolean', 'runtime',
                                   'constructor', 'pending', 'unreleased', 'wait', 'unsafe'])
def test_actual_closed_worker_does_not_hide_contradictions(worker_raw, change):
    raw = copy.deepcopy(worker_raw)
    refs = raw['deepx_numerical_preflight']
    del refs['serial_reference_calls_completed']
    if change == 'base_missing': del refs['serial_sdk_calls_completed']
    elif change == 'base_mismatch': refs['serial_sdk_calls_completed'] = 31
    elif change == 'ref_mismatch': refs['serial_reference_calls_completed'] = 31
    elif change == 'boolean': refs['serial_sdk_calls_completed'] = True
    elif change == 'runtime': raw['deepx_execution_lifecycle']['runtime_calls_completed'] -= 1
    elif change == 'constructor': raw['deepx_execution_lifecycle']['runtime_constructors_returned'] -= 1
    elif change == 'pending': raw['deepx_runtime_observation']['pending'] = 1
    elif change == 'unreleased': raw['deepx_runtime_observation']['unreleased'] = 1
    elif change == 'wait': raw['deepx_runtime_observation']['wait_completed'] = 63
    else: raw['nested'] = {'unsafe_device_state': True}
    assert numerical.device_completion_safety(raw)['device_completion_proven'] is False


def test_report_order_or_comparison_failure_is_not_unknown_sdk_completion(worker_raw):
    raw = copy.deepcopy(worker_raw)
    raw['ok'] = False
    refs = raw['deepx_numerical_preflight']
    refs['status'] = 'failed'
    refs['requests'].reverse()
    refs['requests'][0]['completed_task_equal'] = False
    assert numerical.device_completion_safety(raw)['device_completion_proven'] is True


@pytest.mark.parametrize('failure,safe', [('mapping', True), ('trt', False), ('serial_timeout', False)])
def test_actual_failure_after_sdk_return_is_separate_from_unknown_sdk(tmp_path, monkeypatch, failure, safe):
    code, raw, _ = actual_worker_report(tmp_path, monkeypatch, failure)
    assert code != 0 and raw['ok'] is False
    del raw['deepx_numerical_preflight']['serial_reference_calls_completed']
    assert raw['deepx_numerical_preflight']['serial_reference_requests'] == 0
    assert numerical.device_completion_safety(raw)['device_completion_proven'] is safe


@pytest.mark.parametrize('raw', [{}, {'accepted_work_drained': True}, {'pending': 0, 'unreleased': 0},
                                  {'accepted_work_drained': True, 'pending': 0, 'unreleased': 0}])
def test_zero_queue_or_flag_without_sdk_entry_return_evidence_never_proves_completion(raw):
    assert numerical.device_completion_safety(raw)['device_completion_proven'] is False


def test_pre_runtime_missing_redundant_zero_remains_report_gap_but_no_sdk_entry():
    raw = {'deepx_execution_lifecycle': NativeExecutionLifecycle().report(),
           'serial_reference_calls_started': 0}
    safety = numerical.device_completion_safety(raw)
    assert safety['device_completion_proven'] is safety['pre_runtime_proven'] is True
    assert safety['report_contract_complete'] is False
    assert 'serial_reference_calls_completed' not in raw


def test_independent_closed_child_never_covers_unknown_or_missing_counter_sibling(worker_raw):
    unknown = {'deepx_execution_lifecycle': NativeExecutionLifecycle().report()}
    unknown['deepx_execution_lifecycle']['runtime_construction_started'] = True
    raw = {'rows': [worker_raw, unknown]}
    assert numerical.device_completion_safety(raw)['device_completion_proven'] is False
    unknown.clear()
    unknown.update({'deepx_execution_lifecycle': copy.deepcopy(worker_raw['deepx_execution_lifecycle']),
                    'serial_reference_calls_started': 1})
    assert numerical.device_completion_safety(raw)['device_completion_proven'] is False


@pytest.mark.parametrize('condition,safe', [('closed', True), ('active', False), ('poisoned', False),
                                          ('no_cleanup', False), ('wrong_target', False), ('missing_dir', False)])
def test_owned_preparation_closure_is_exact_process_proof(tmp_path, condition, safe):
    transport = SimpleNamespace(diagnostics_dir=tmp_path, host=SimpleNamespace(id='dut'))
    registry = SimpleNamespace(active_count=lambda: 1 if condition == 'active' else 0,
        cancelled=condition == 'poisoned', configured_scope=object())
    native._write(tmp_path / 'ssh_stage.json', {'command': 'owned staging', 'target_id': 'other' if condition == 'wrong_target' else 'dut',
        'returncode': 1, 'cleanup_proven': condition != 'no_cleanup'})
    if condition == 'missing_dir': transport.diagnostics_dir = None
    assert numerical._owned_preparation_closure(transport, registry, set())['proven'] is safe


@pytest.mark.parametrize('condition,fenced', [('result_gap', False), ('no_cleanup', True),
    ('stage_closed', False), ('stage_unknown', True), ('comparison', False)])
def test_actual_normal_numerical_owner_requires_cleanup_for_every_proof(tmp_path, monkeypatch, condition, fenced):
    prepared, context = phase_prepared(tmp_path)
    output = Path(prepared['output'])
    selection = tmp_path / 'images.json'
    native._write(selection, {'images': [f'/original/{index}.jpg' for index in range(32)]})
    binding = {'path': str(selection), 'sha256': native._sha(selection)}
    context['remote_process_registry'] = SimpleNamespace(active_count=lambda: 0, cancelled=False, configured_scope=object())
    transport = SimpleNamespace(diagnostics_dir=output / 'diagnostics', host=SimpleNamespace(id='dut'))
    fences, dispatches = [], []
    class Lock:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def commit_quarantine_fence(self, evidence): fences.append(evidence)
    monkeypatch.setattr(native.EvaluationRunLock, 'for_resource', lambda *args, **kwargs: Lock())
    monkeypatch.setattr(native, '_verify_current_target', lambda *args: {})
    monkeypatch.setattr(native, 'bind_workflow_resource_options', lambda **kwargs: nullcontext())
    monkeypatch.setattr(native, '_transport', lambda *args: transport)
    monkeypatch.setattr(native, 'ensure_current_deepx_runtime', lambda *args: None)
    def stage(*args):
        if condition.startswith('stage_'):
            reserved = native._read(output / 'NUMERICAL_RESULT.json')
            assert reserved['preparation'] == {'phase': 'pre_worker_staging', 'worker_dispatched': False}
            native._write(transport.diagnostics_dir / 'ssh_stage.json', {'command': 'owned staging', 'target_id': 'dut',
                'returncode': 1, 'cleanup_proven': condition == 'stage_closed'})
            raise ValueError('local metadata staging failure')
    monkeypatch.setattr(native, '_stage_benchmark_set', stage)
    monkeypatch.setattr(numerical, '_stage_images', lambda *args: None)
    def execute(command, **kwargs):
        dispatches.append(command)
        native._write(transport.diagnostics_dir / 'ssh_owned.json', {'command': command, 'target_id': 'dut',
            'returncode': 0, 'cleanup_proven': condition != 'no_cleanup'})
        return 0, 'owned worker returned'
    transport.run = execute
    def collect(*args):
        raw = valid_split_numerical_raw()
        refs = raw['deepx_numerical_preflight']
        if condition in {'result_gap', 'no_cleanup'}:
            refs.update(serial_reference_calls_started=32, serial_sdk_calls_started=32, serial_sdk_calls_completed=32)
        else:
            refs['requests'][0]['completed_task_equal'] = False
        path = output / 'numerical/worker.json'
        native._write(path, raw)
        return path
    monkeypatch.setattr(native, '_collect_repeat', collect)
    result = numerical.run_numerical_native(prepared, context, binding)
    assert result['status'] == 'failed'
    assert bool(fences) is fenced
    assert result['resource_completion_proven'] is (not fenced)
    assert len(dispatches) == (0 if condition.startswith('stage_') else 1)


@pytest.mark.parametrize('remove', ['semantic_marker', 'measured_marker', 'semantic_child', 'measured_child'])
def test_final_full_attempt_requires_both_actual_child_lifecycles(remove):
    lifecycle = NativeExecutionLifecycle()
    lifecycle.construction_enter(); lifecycle.construction_return()
    lifecycle.inference_enter(); lifecycle.inference_return()
    child = {'deepx_execution_lifecycle': lifecycle.report()}
    row = {'backend': 'native_full_deepx', 'repetition_count_attempted': 1,
           'preparation_evidence': copy.deepcopy(child), 'deepx_prepared_feed_benchmark': copy.deepcopy(child)}
    assert numerical.device_completion_safety({'rows': [row]})['device_completion_proven'] is True
    key = 'preparation_evidence' if remove.startswith('semantic') else 'deepx_prepared_feed_benchmark'
    if remove.endswith('child'): del row[key]
    else: del row[key]['deepx_execution_lifecycle']
    assert numerical.device_completion_safety({'rows': [row]})['device_completion_proven'] is False


def test_full_preflight_failure_zero_attempts_needs_no_measured_child():
    raw = {'rows': [{'backend': 'native_full_deepx', 'repetition_count_attempted': 0,
        'preparation_evidence': {'deepx_execution_lifecycle': NativeExecutionLifecycle().report()}}]}
    assert numerical.device_completion_safety(raw)['device_completion_proven'] is True
