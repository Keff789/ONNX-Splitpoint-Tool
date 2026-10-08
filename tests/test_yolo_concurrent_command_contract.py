"""Portable checks for the bounded concurrent runtime, never measured data."""
import copy

import pytest

from onnx_splitpoint_tool.native_command_contract import (
    seal_native_command_contract, verify_native_energy_command_contract,
    split_energy_runtime_argv, verify_native_command_contract,
)
from tests.test_v272_hailo8_completed_detection_hotloop import _sealed_hailo8_command_contract


def contract():
    value = _sealed_hailo8_command_contract('hailo8_cpp_concurrent_three_stage')
    value.update(model='yolov7_paper', case='b066', precision='uint8_dequant_fp16')
    geometry = dict(task='detection', preprocess_mode_requested='letterbox',
                    preprocess_mode_effective='letterbox', letterbox_pad_value_requested=114,
                    letterbox_pad_value_effective=114, letterbox_pad_value=114)
    value['prepared_input_contract'] = {
        **geometry, 'dtype': 'uint8', 'layout': 'HWC', 'shape': [640, 640, 3],
        'format': 'raw_runtime_tensor', 'pad_value_effective': 114,
        'source_image_sha256': value['input_image_sha256'],
    }
    value['runtime_options'].update(
        **geometry, frames=1000, warmup=100, repetitions=3, post_queue_depth=4,
        measurement_endpoint='completed_task', completion_runtime_mode='fast_oracle_outside_timing',
        measurement_boundary='workers_ready_to_last_completed_task_frame',
        warmup_contract='fully_drained_before_worker_start', preprocessing_included=False,
        prepared_input_bound=True, energy_prepared_feed_capable=True,
        completion_execution_contract={'structural_fixture_only': True},
    )
    value['artifacts']['native_executable']['path'] = '/verified/three_stage_canary.so'
    for role in ('concurrent_helper', 'concurrent_canary', 'concurrent_fast_decode',
                 'native_three_stage', 'prepared_input'):
        value['artifacts'][role] = {'path': '/verified/' + role, 'sha256': 'a' * 64}
    return seal_native_command_contract(value)


def test_cpp_closure_is_distinct_from_python_runtime_and_replays_prepared_energy():
    value = contract()
    verified, reason = verify_native_energy_command_contract(value)
    assert verified == value, reason
    assert 'mixed_runtime_contract' not in value
    argv = split_energy_runtime_argv(value, duration_s=60, fresh_output_root='/fresh',
        remote_tool_dir='/tool', preflight_attestation_path='/actual-preflight.json')
    assert '--energy-workload-only' in argv and '--no-build' in argv
    assert argv[argv.index('--warmup') + 1] == '0'
    assert argv[argv.index('--duration-s') + 1] == '60.0'


@pytest.mark.parametrize('repetitions', [1, 3, 0, 2, 4, True, '1'])
def test_normal_concurrent_command_retains_strict_separate_process_budget(repetitions):
    value = copy.deepcopy(contract())
    value['runtime_options']['repetitions'] = repetitions
    value = seal_native_command_contract(value)
    verified, reason = verify_native_command_contract(value)
    if type(repetitions) is int and repetitions in {1, 3}:
        assert verified == value, reason
    else:
        assert verified is None and 'concurrent_repetitions_invalid' in reason


@pytest.mark.parametrize('mutation, expected', [
    ('missing_source', 'required_artifact_missing'),
    ('python_binary', 'library_invalid'),
    ('input_shape', 'prepared_input_invalid'),
    ('input_source', 'prepared_input_invalid'),
    ('includes_preprocess', 'measurement_boundary_invalid'),
    ('raw_endpoint', 'measurement_boundary_invalid'),
    ('missing_repetition', 'repetitions_invalid'),
    ('short_warmup', 'warmup_invalid'),
    ('wrong_case', 'identity_invalid'),
])
def test_resealed_incompatible_runtime_stays_negative(mutation, expected):
    value = copy.deepcopy(contract())
    options = value['runtime_options']
    if mutation == 'missing_source': del value['artifacts']['concurrent_fast_decode']
    elif mutation == 'python_binary': value['artifacts']['native_executable']['path'] = '/verified/python'
    elif mutation == 'input_shape': value['prepared_input_contract']['shape'] = [3, 640, 640]
    elif mutation == 'input_source': value['prepared_input_contract']['source_image_sha256'] = 'f' * 64
    elif mutation == 'includes_preprocess': options['preprocessing_included'] = True
    elif mutation == 'raw_endpoint': options['measurement_endpoint'] = 'raw_model_outputs'
    elif mutation == 'missing_repetition': options['repetitions'] = 2
    elif mutation == 'short_warmup': options['warmup'] = 99
    elif mutation == 'wrong_case': value['case'] = 'b044'
    verified, reason = verify_native_energy_command_contract(seal_native_command_contract(value))
    assert verified is None and expected in reason
