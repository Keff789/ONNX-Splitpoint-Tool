"""Native A4 through the actual profile, dispatch and replay boundaries."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.native_command_contract import (
    deepx_runtime_policy, successful_runtime_argv, split_energy_runtime_argv,
)
from onnx_splitpoint_tool.native_execution_contract import (
    build_native_execution_contract, enforce_native_variant_execution_contract,
    resolve_native_execution_contract, verify_native_execution_contract,
)
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.profile_options import workflow_options_from_profile_snapshot
from onnx_splitpoint_tool.workflow.start_snapshot import (
    materialize_runtime_profile, resolve_runtime_profile_start_snapshot,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from scripts import native_producer_e2e_eval_runner as dispatcher
from scripts import run_evalrun_native_producer_variants as variants


def profile(backend='deepx'):
    return {'name': 'a4_contracts', 'native_producers': {
        'enabled': True, 'backends': [backend], 'frames': 1000, 'warmup': 0,
        'repetitions': 3, 'queue_depth': 2, 'inflight': 8,
    }, 'hardware': {'selected_setups': []}, 'run_profiles': [{
        'id': backend + '_to_tensorrt', 'type': 'mixed_backend',
        'stage1': backend, 'stage2': 'tensorrt',
    }]}


def test_historical_policy_and_contract_remain_s1_without_rewriting():
    old = {'queue_depth': 2, 'inflight': 8, 'warmup': 0}
    before = deepcopy(old)
    assert deepx_runtime_policy(old)['deepx_runtime_mode'] == 'S1'
    assert old == before
    contract = build_native_execution_contract(old)
    assert 'deepx_runtime_mode' not in contract
    assert verify_native_execution_contract(contract) == contract


@pytest.mark.parametrize('bad', [None, '', 'A2', 4])
def test_only_missing_mode_is_historical(bad):
    with pytest.raises(ValueError, match='deepx_runtime_mode_invalid'):
        deepx_runtime_policy({'deepx_runtime_mode': bad})


@pytest.mark.parametrize('bad', [True, 4.0, 1, 8, '4'])
def test_a4_maximum_is_exactly_four(bad):
    with pytest.raises(ValueError, match='inflight'):
        deepx_runtime_policy({'deepx_runtime_mode': 'A4', 'deepx_max_inflight': bad})


def test_a4_split_queue_is_fixed_and_full_needs_no_artificial_queue():
    assert deepx_runtime_policy({'deepx_runtime_mode': 'A4'})['deepx_max_inflight'] == 4
    with pytest.raises(ValueError, match='queue_depth_3'):
        deepx_runtime_policy({'deepx_runtime_mode': 'A4', 'queue_depth': 2})
    with pytest.raises(ValueError, match='queue_depth_3'):
        build_native_execution_contract({'deepx_runtime_mode': 'A4', 'deepx_queue_depth': 4})


def test_fresh_gui_and_cli_snapshot_share_explicit_a4_and_keep_hailo_options(tmp_path):
    source = profile()
    resolved, snapshot = resolve_runtime_profile_start_snapshot(
        profile_request='a4.yaml', source_profile=source, resolved_profile=source,
        profile_id='a4_contracts', profile_path='', profile_source='test')
    options = workflow_options_from_profile_snapshot(
        profile_request='a4.yaml', out_root=str(tmp_path), start_snapshot=snapshot)
    assert options.native_producer_deepx_runtime_mode == 'A4'
    assert options.profile_start_snapshot['resolved_profile'] == resolved
    cfg = resolved['native_producers']
    assert cfg['deepx_runtime_mode'] == 'A4'
    assert cfg['deepx_max_inflight'] == 4 and cfg['deepx_queue_depth'] == 3
    assert (cfg['queue_depth'], cfg['inflight'], cfg['warmup']) == (2, 8, 0)
    assert 'deepx_runtime_mode' not in source['native_producers']
    contract = resolve_native_execution_contract(resolved)
    assert contract['deepx_runtime_mode'] == 'A4'
    assert contract['queue_depth'] == 2 and contract['inflight'] == 8


@pytest.mark.parametrize('backend', ['hailo8', 'hailo10h'])
def test_hailo_materialization_acquires_no_deepx_policy(backend):
    resolved, _ = materialize_runtime_profile(profile(backend))
    assert not any(key.startswith('deepx_') for key in resolved['native_producers'])
    assert resolved['native_producers']['inflight'] == 8


def test_archived_materialization_does_not_upgrade_missing_mode():
    resolved, _ = materialize_runtime_profile(profile(), options=SimpleNamespace(resume=True))
    assert 'deepx_runtime_mode' not in resolved['native_producers']


def config_for(source, *, resume=False, mode=''):
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.profile_payload = source
    runner.options = WorkflowOptions('profile.yaml', '/tmp/not-used', resume=resume,
                                    native_producer_deepx_runtime_mode=mode)
    return runner._native_producer_config()


def test_fresh_controller_selects_a4_and_historical_resume_is_blocked():
    assert config_for(profile())['deepx_runtime_mode'] == 'A4'
    with pytest.raises(ValueError, match='historical_s1_resume_requires_original_source'):
        config_for(profile(), resume=True)
    # An explicit new mode cannot migrate the old queue on resume either.
    with pytest.raises(ValueError, match='historical_s1_resume'):
        config_for(profile(), resume=True, mode='A4')


def test_explicit_s1_diagnostic_and_a4_resume_keep_their_mode():
    assert config_for(profile(), mode='S1')['deepx_runtime_mode'] == 'S1'
    source = profile()
    source['native_producers'].update(deepx_runtime_mode='A4', deepx_queue_depth=3)
    assert config_for(source, resume=True)['deepx_runtime_mode'] == 'A4'
    with pytest.raises(ValueError, match='resume_mode_mismatch'):
        config_for(source, resume=True, mode='S1')


def test_tensorrt_full_only_on_deepx_host_is_not_a_deepx_native_job():
    source = profile()
    source['run_profiles'] = [{'id': 'ort_tensorrt', 'type': 'same_backend_reference',
                               'full': 'tensorrt', 'stage1': 'tensorrt', 'stage2': 'tensorrt'}]
    source['native_producers']['full_baselines'] = {
        'enabled': True, 'backends_by_producer': {'deepx': ['tensorrt']}}
    config = config_for(source, resume=True)
    assert 'deepx_runtime_mode' not in config


def test_per_model_variant_cannot_change_deepx_mode():
    contract = build_native_execution_contract({'deepx_runtime_mode': 'A4'})
    with pytest.raises(ValueError, match='variant_override_forbidden:deepx_runtime_mode'):
        enforce_native_variant_execution_contract({'deepx_runtime_mode': 'S1'}, contract)


@pytest.mark.parametrize('field,value', [('deepx_max_inflight', 1), ('deepx_queue_depth', 1)])
def test_per_model_variant_cannot_reduce_fixed_a4_policy(field, value):
    contract = build_native_execution_contract({'deepx_runtime_mode': 'A4'})
    with pytest.raises(ValueError, match='variant_override_forbidden:' + field):
        enforce_native_variant_execution_contract({field: value}, contract)


def test_full_only_new_profile_writes_the_same_a4_policy():
    source = profile()
    source['run_profiles'] = [{'id': 'deepx_m1_full', 'type': 'same_backend_reference',
                               'full': 'deepx_m1', 'stage1': 'deepx_m1', 'stage2': 'deepx_m1'}]
    resolved, _ = materialize_runtime_profile(source)
    assert resolved['native_producers']['deepx_runtime_mode'] == 'A4'
    assert resolved['native_producers']['deepx_max_inflight'] == 4


def test_new_s1_diagnostic_keeps_requested_legacy_queue():
    source = profile()
    source['native_producers']['deepx_runtime_mode'] = 'S1'
    resolved, _ = materialize_runtime_profile(source)
    contract = resolve_native_execution_contract(resolved)
    assert contract['deepx_runtime_mode'] == 'S1'
    assert contract['deepx_queue_depth'] == source['native_producers']['queue_depth'] == 2


def replay_contract(mode):
    options = {'warmup': 0, 'queue_depth': 3, 'task': 'classification',
               'preprocess_mode_requested': 'auto', 'preprocess_mode_effective': 'resize',
               'letterbox_pad_value_requested': 114, 'letterbox_pad_value_effective': 0,
               'letterbox_pad_value': 0, 'prepared_input_bound': True}
    if mode:
        options.update(deepx_runtime_mode=mode, deepx_max_inflight=4 if mode == 'A4' else 1)
    return {'backend': 'deepx_to_trt', 'runtime_options': options,
            'boundary_contract': {'boundary_layout_effective': 'as_input'},
            'artifacts': {'dxnn': {'sha256': 'd'*64}, 'engine': {'sha256': 'e'*64},
                          'prepared_input': {'path': '/input.npy'}},
            'prepared_input_contract': {**options, 'shape': [1, 2, 2, 3], 'dtype': 'uint8'},
            'python_executable': '/python', 'runner': 'scripts/deepx.py',
            'benchmark_set': '/benchmark', 'case': 'b027', 'precision': 'uint8_cast_fp16',
            'input_image': '/image.jpg', 'runner_sha256': 'f'*64,
            'input_image_sha256': 'a'*64, 'contract_sha256': 'b'*64}


@pytest.mark.parametrize('mode,expected', [(None, 'S1'), ('S1', 'S1'), ('A4', 'A4')])
def test_performance_and_energy_replay_use_bound_mode_and_zero_warmup(mode, expected):
    contract = replay_contract(mode)
    before = deepcopy(contract)
    common = dict(duration_s=60, fresh_output_root='/fresh', remote_tool_dir='/tool')
    performance = successful_runtime_argv(contract, **common)
    energy = split_energy_runtime_argv(contract, **common, preflight_attestation_path='/proof.json')
    for argv in (performance, energy):
        if mode is None:
            assert '--deepx-runtime-mode' not in argv  # retained old source has no new option
        else:
            assert argv[argv.index('--deepx-runtime-mode') + 1] == expected
        assert argv[argv.index('--warmup') + 1] == '0'
        assert argv[argv.index('--queue-depth') + 1] == '3'
    assert contract == before


def test_child_dispatch_rejects_s1_report_for_requested_a4():
    with pytest.raises(ValueError, match='mode_mismatch'):
        dispatcher._deepx_runtime_projection({'ok': True}, 'A4')
    assert dispatcher._deepx_runtime_projection({'ok': True}, 'S1') == {}
    report = {'native_command_contract': {'runtime_options': {'deepx_runtime_mode': 'A4',
               'deepx_max_inflight': 4, 'queue_depth': 3}}, 'host_pending_max': 3}
    assert dispatcher._deepx_runtime_projection(report, 'A4')['host_pending_max'] == 3


def test_three_a4_repeats_launch_three_fresh_processes_and_preserve_all_reports(tmp_path, monkeypatch):
    import subprocess
    result_path = tmp_path / 'native' / 'deepx_native_fifo_e2e_results.json'
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        output = Path(argv[argv.index('--out-dir') + 1])
        output.mkdir(parents=True)
        record = {'ok': True, 'status': 'ok', 'fps_makespan': 100.0 + len(commands),
                  'runtime_instance_id': f'process-{len(commands)}',
                  'completed_frames': 1000, 'completed_work_units': 1000,
                  'deepx_runtime_mode': 'A4', 'deepx_max_inflight': 4,
                  'runtime_options': {'deepx_runtime_mode': 'A4', 'deepx_max_inflight': 4, 'queue_depth': 3}}
        (output / result_path.name).write_text(json.dumps(record))
        return subprocess.CompletedProcess(argv, 0, 'done', '')
    monkeypatch.setattr(dispatcher.subprocess, 'run', run)
    command = ['/python', 'deepx.py', '--repetitions', '3', '--deepx-runtime-mode', 'A4']
    result = dispatcher._run_deepx_a4_repeats(command, result_path, repetitions=3, timeout=10)
    assert result.returncode == 0
    report = json.loads(result_path.read_text())
    assert len(commands) == 3
    assert all(argv[argv.index('--repetitions') + 1] == '1' for argv in commands)
    assert len({argv[argv.index('--out-dir') + 1] for argv in commands}) == 3
    assert report['repetition_count_valid'] == 3
    assert report['repetition_independence_verified'] is True
    assert report['repetition_runtime_scope'] == 'fresh_process_per_repetition'
    assert report['fps_makespan'] == 102.0
    assert all(Path(leaf['report']).is_file() for leaf in report['fresh_process_launches'])


def test_failed_a4_repeat_preserves_failure_and_does_not_claim_complete(tmp_path, monkeypatch):
    import subprocess
    commands = []
    def fail(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 2, '', 'SDK wait failed')
    monkeypatch.setattr(dispatcher.subprocess, 'run', fail)
    result_path = tmp_path / 'native' / 'deepx_native_fifo_e2e_results.json'
    result = dispatcher._run_deepx_a4_repeats(
        ['/python', 'deepx.py', '--repetitions', '3'], result_path, repetitions=3, timeout=10)
    report = json.loads(result_path.read_text())
    assert result.returncode == 2 and report['ok'] is False
    assert len(commands) == report['repetition_count_attempted'] == 1
    assert report['repetition_count_requested'] == 3
    assert report['repetition_count_valid'] == 0
    assert report['repetition_status'] == 'incomplete'


def updater_options(mode, execution_contract=None):
    args = SimpleNamespace(
        hailo8_ssh='', hailo10_ssh='', deepx_ssh='', hailo8_env='',
        hailo10_env='', deepx_env='', backends='deepx', case_policy='case_map_only',
        precision='uint8_cast_fp16', frames=1000, warmup=0, repetitions=3,
        queue_depth=2, inflight=8, hailo_format='uint8', remote_root='/remote',
        remote_tool_dir='/tool', dump_outputs=True, no_build_missing_engines=True,
        engine_build_python='auto', no_copy=False, case_map='{"resnet50":["b045"]}',
        native_execution_contract_json=json.dumps(execution_contract) if execution_contract else '')
    if mode is not None:
        args.deepx_runtime_mode = mode
    return args


def test_normal_update_dispatch_accepts_a4_without_changing_shared_hailo_effort():
    from scripts import update_evalset_native_producers as updater
    contract = build_native_execution_contract({'deepx_runtime_mode': 'A4', 'queue_depth': 2,
                                               'inflight': 8, 'warmup': 0, 'frames': 1000, 'repetitions': 3})
    config = updater._native_cfg_from_args(updater_options('A4', contract))
    assert config['deepx_runtime_mode'] == 'A4'
    assert config['_native_execution_contract'] == contract
    assert (config['queue_depth'], config['inflight'], config['warmup']) == (2, 8, 0)


def test_normal_update_dispatch_rejects_mode_drift_from_parent():
    from scripts import update_evalset_native_producers as updater
    contract = build_native_execution_contract({'deepx_runtime_mode': 'A4', 'queue_depth': 2,
                                               'inflight': 8, 'warmup': 0, 'frames': 1000, 'repetitions': 3})
    for mode in ('S1', None):
        with pytest.raises(Exception, match='contract_cli_mismatch:deepx_runtime_mode'):
            updater._native_cfg_from_args(updater_options(mode, contract))


def test_normal_update_legacy_dispatch_retains_missing_mode():
    from scripts import update_evalset_native_producers as updater
    config = updater._native_cfg_from_args(updater_options(None))
    assert 'deepx_runtime_mode' not in config
    assert 'deepx_runtime_mode' not in config['_native_execution_contract']


def test_normal_full_report_projection_preserves_explicit_and_historical_mode(tmp_path):
    from scripts import native_producer_final_report as report
    root = tmp_path / 'native'
    output = root / 'analysis_tables/native_full_baseline_eval.json'
    output.parent.mkdir(parents=True)
    base = {'backend': 'native_full_deepx', 'model': 'resnet50', 'case': 'full',
            'ok': True, 'task': 'classification', 'outer_makespan_verified': True,
            'fps_makespan': 10.0, 'runtime_options': {'deepx_runtime_mode': 'A4', 'deepx_max_inflight': 4},
            'deepx_runtime_mode': 'A4', 'deepx_max_inflight': 4, 'actual_pending_max': 3}
    output.write_text(json.dumps({'rows': [base]}))
    row, = report._rows_from_native_full(root)
    assert row['deepx_runtime_mode'] == 'A4'
    assert row['deepx_max_inflight'] == 4
    assert row['actual_pending_max'] == 3
    for field in ('deepx_runtime_mode', 'deepx_max_inflight', 'actual_pending_max', 'runtime_options'):
        base.pop(field)
    output.write_text(json.dumps({'rows': [base]}))
    old, = report._rows_from_native_full(root)
    assert 'deepx_runtime_mode' not in old
    assert 'deepx_max_inflight' not in old


@pytest.mark.parametrize('phase', ['complete', 'numerical'])
def test_normal_profile_loader_accepts_full_and_split_acceptance_cells(tmp_path, phase):
    import yaml
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot
    root = Path(__file__).parents[1]
    source = yaml.safe_load((root / 'onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml').read_text())
    source['native_producers'] = {'enabled': True, 'backends': ['deepx'], 'deepx_runtime_mode': 'A4',
                                  'energy': {'enabled': False}}
    source['run_profiles'] = [{'id': 'deepx_m1_full', 'type': 'same_backend_reference',
                               'full': 'deepx_m1', 'stage1': 'deepx_m1', 'stage2': 'deepx_m1'},
                              {'id': 'deepx_m1_to_tensorrt', 'type': 'mixed_backend',
                               'stage1': 'deepx_m1', 'stage2': 'tensorrt'}]
    source['hardware_acceptance'] = {'enabled': True, 'source_report_root': str(tmp_path),
                                     'source_primary_run': str(tmp_path), 'phase': phase, 'cells': [
        {'model_id': 'resnet50', 'setup_id': 'setup', 'case_id': 'full', 'runtime_variants': ['native'],
         'native_backend': 'native_full_deepx'},
        {'model_id': 'resnet50', 'setup_id': 'setup', 'case_id': 'b045', 'runtime_variants': ['native'],
         'native_backend': 'deepx_to_trt'}]}
    path = tmp_path / 'full_and_split.yaml'
    path.write_text(yaml.safe_dump(source))
    resolved, snapshot = load_runtime_profile_snapshot(str(path))
    assert [cell['case_id'] for cell in resolved['hardware_acceptance']['cells']] == ['full', 'b045']
    assert resolved['hardware_acceptance']['phase'] == phase
    assert snapshot['resolved_profile']['native_producers']['deepx_runtime_mode'] == 'A4'


@pytest.mark.parametrize('runtime_python', [None, '/usr/bin/python3.10', '/opt/DeepX Runtime/bin/python'])
def test_remote_module_binding_uses_exact_optional_interpreter(runtime_python):
    import shlex
    import subprocess
    from onnx_splitpoint_tool.workflow.runner import _verify_remote_module_binding_v263
    commands = []
    def run(argv, **kwargs):
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps({'ok': True}), '')
    options = {} if runtime_python is None else {'runtime_python': runtime_python}
    result = _verify_remote_module_binding_v263(
        ssh='nx@host', remote_tool_dir='/remote/tool', remote_env='',
        module_name='onnx_splitpoint_tool.native_command_contract',
        relative_path='onnx_splitpoint_tool/native_command_contract.py',
        expected_sha256='a' * 64, process_runner=run, **options)
    assert result['verification']['ok'] is True
    assert f"{shlex.quote(runtime_python or 'python')} -c " in commands[0][-1]
