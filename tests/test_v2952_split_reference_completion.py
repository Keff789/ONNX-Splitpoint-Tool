"""Actual Split writer -> strict lifecycle reader, with only SDK/TRT leaves fake."""
import copy
import json
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from onnx_splitpoint_tool.workflow.hardware_acceptance_native import _deepx_completion_safety
from onnx_splitpoint_tool.workflow.hardware_acceptance_numerical import execution_lifecycle_safety
from scripts import native_deepx_trt_e2e_from_benchmarkset as runner
from test_v295_native_deepx_split_a4 import AsyncEngine, TRT


def actual_worker_report(tmp_path, monkeypatch, failure=None):
    observed = SimpleNamespace(engines=[], trts=[], serial_entries=0)

    class Engine(AsyncEngine):
        def __init__(self, path):
            super().__init__()
            observed.engines.append(self)

        def run(self, inputs):
            observed.serial_entries += 1
            # Entry1 is the separate setup SDK reference; Entry2 is oracle0.
            if observed.serial_entries == 2 and failure in {'serial_exception', 'serial_timeout'}:
                error = TimeoutError if failure == 'serial_timeout' else RuntimeError
                raise error('oracle SDK entered without a successful return')
            return [np.full((1, 4), float(inputs[0][0, 0, 0, 0]), dtype=np.float32)]

        def wait(self, job):
            _request_id, inputs, outputs = self.jobs.pop(job)
            outputs[0].fill(float(inputs[0][0, 0, 0, 0]))
            return outputs

        def close(self):
            assert not self.jobs

    class Consumer(TRT):
        outputs = ['logits']

        def __init__(self, path):
            super().__init__()
            self.outputs = ['logits']
            observed.trts.append(self)

        def run_prepared(self):
            return {'logits': self.staged.copy()}

        def run(self, feeds):
            if failure == 'trt':
                raise RuntimeError('oracle TRT entered without a successful return')
            return super().run(feeds)

        def close(self):
            pass

    benchmark = tmp_path / 'benchmark_set'
    benchmark.mkdir()
    (benchmark / 'benchmark_set.json').write_text(json.dumps({
        'model_id': 'mobilenet_v3_large', 'benchmark_task': 'classification'}))
    dxnn, engine = tmp_path / 'part1.dxnn', tmp_path / 'part2.engine'
    dxnn.write_bytes(b'existing fake SDK handle'); engine.write_bytes(b'existing fake TRT handle')
    contract = tmp_path / 'part1.json'
    contract.write_text(json.dumps({'input': {'name': 'image', 'shape': [2, 3, 3], 'dtype': 'uint8',
        'layout': 'HWC', 'task': 'classification', 'preprocess_mode_requested': 'auto',
        'preprocess_mode_effective': 'resize', 'letterbox_pad_value_requested': 114,
        'letterbox_pad_value_effective': 0}}))
    images = []
    for index in range(32):
        path = tmp_path / f'image{index:02d}.png'
        Image.fromarray(np.full((2, 3, 3), index, dtype=np.uint8)).save(path)
        images.append(str(path))
    manifest = tmp_path / 'images.json'
    manifest.write_text(json.dumps({'images': images}))
    monkeypatch.setitem(sys.modules, 'dx_engine', SimpleNamespace(InferenceEngine=Engine))
    monkeypatch.setattr(runner, 'NativeTRT', Consumer)
    monkeypatch.setattr(runner, '_STRICT_SPLIT_BOUNDARY', None)
    monkeypatch.setattr(runner, '_find_part1_dxnn', lambda *_: dxnn)
    monkeypatch.setattr(runner, '_engine_path', lambda *_: engine)
    monkeypatch.setattr(runner, '_part1_output_contract_path', lambda *_: contract)
    if failure == 'mapping':
        real_mapping = runner._map_a4_deepx_output_to_trt
        calls = 0
        def map_outputs(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('oracle mapping after the successful SDK return')
            return real_mapping(*args)
        monkeypatch.setattr(runner, '_map_a4_deepx_output_to_trt', map_outputs)
    output = tmp_path / 'result'
    monkeypatch.setattr(sys, 'argv', [runner.__file__, '--benchmark-set', str(benchmark),
        '--case', 'b027', '--image', images[0], '--out-dir', str(output),
        '--deepx-runtime-mode', 'A4', '--deepx-numerical-only', '--deepx-numerical-images', str(manifest),
        '--frames', '64', '--warmup', '0'])
    code = runner.main()
    return code, json.loads((output / 'deepx_native_fifo_e2e_results.json').read_text()), observed


@pytest.fixture
def passed_report(tmp_path, monkeypatch):
    code, raw, observed = actual_worker_report(tmp_path, monkeypatch)
    assert code == 0 and raw['ok'] is True
    assert observed.serial_entries == 33
    return raw


def test_actual_32_image_64_smoke_writer_has_complete_counter_pair_and_passes_strict_reader(passed_report):
    raw = passed_report
    reference = raw['deepx_numerical_preflight']
    assert reference['serial_reference_calls_started'] == reference['serial_reference_calls_completed'] == 32
    assert reference['serial_reference_requests'] == reference['mixed_completions'] == 32
    assert reference['serial_sdk_calls_started'] == reference['serial_sdk_calls_completed'] == 32
    assert reference['serial_trt_calls_started'] == reference['serial_trt_calls_completed'] == 32
    assert raw['completed_frames'] == 64
    lifecycle = raw['deepx_execution_lifecycle']
    assert lifecycle['runtime_constructors_entered'] == lifecycle['runtime_constructors_returned'] == 2
    assert lifecycle['runtime_calls_entered'] == lifecycle['runtime_calls_completed'] == 257
    assert execution_lifecycle_safety(raw)['device_completion_proven'] is True
    assert _deepx_completion_safety(raw)['device_completion_proven'] is True


@pytest.mark.parametrize('change', ['mismatch', 'boolean', 'unsafe', 'pending'])
def test_actual_worker_shape_keeps_missing_mismatched_and_unsafe_counter_vetoes(passed_report, change):
    raw = copy.deepcopy(passed_report)
    reference = raw['deepx_numerical_preflight']
    if change == 'mismatch':
        reference['serial_reference_calls_completed'] = 31
    elif change == 'boolean':
        reference['serial_reference_calls_completed'] = True
    elif change == 'unsafe':
        raw['nested'] = {'unsafe_device_state': True}
    else:
        raw['deepx_runtime_observation']['pending'] = 1
    # In particular requests=32 and lifecycle257/257 never backfill the
    # historical missing completion field that caused the real failed run.
    assert reference['serial_reference_requests'] == 32
    assert execution_lifecycle_safety(raw)['device_completion_proven'] is False
    assert _deepx_completion_safety(raw)['device_completion_proven'] is False


@pytest.mark.parametrize('failure', ['serial_exception', 'serial_timeout', 'trt', 'mapping'])
def test_actual_oracle_completion_counter_tracks_sdk_return_independently_of_task_failure(tmp_path, monkeypatch, failure):
    code, raw, _observed = actual_worker_report(tmp_path, monkeypatch, failure)
    assert code == 4 and raw['ok'] is False
    reference = raw['deepx_numerical_preflight']
    assert reference['serial_reference_calls_started'] == 1
    assert reference['serial_reference_calls_completed'] == (0 if failure.startswith('serial_') else 1)
    assert reference['serial_reference_requests'] == reference['mixed_completions'] == 0
    assert reference['status'] == 'failed'
    safe = _deepx_completion_safety(raw)['device_completion_proven']
    assert safe is (failure == 'mapping')
    if failure == 'trt':
        assert reference['serial_trt_calls_started'] == 1
        assert reference.get('serial_trt_calls_completed', 0) == 0


def test_reference_pair_initialized_zero_before_first_serial_reference(tmp_path, monkeypatch):
    original = runner._run_a4_serial
    first = []
    def observe(*args, **kwargs):
        if kwargs['phase'] == 'numerical_reference' and not first:
            progress = kwargs['progress']
            first.append((progress['serial_reference_calls_started'], progress['serial_reference_calls_completed']))
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, '_run_a4_serial', observe)
    code, raw, _ = actual_worker_report(tmp_path, monkeypatch)
    assert code == 0 and first == [(0, 0)]
    assert raw['deepx_numerical_preflight']['serial_reference_calls_completed'] == 32
