"""Exercise the shipped Native split adapter with synthetic SDK/TRT leaves."""
from __future__ import annotations

import threading
import time
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as postprocess
from scripts import native_deepx_trt_e2e_from_benchmarkset as runner
from test_v272_detection_hotloop_integration import _FakeCompletionRuntime


class AsyncEngine:
    def __init__(self, *, submit_failure=None, wait_failure=None):
        self.jobs = {}
        self.next_job = 0
        self.submissions = []
        self.events = []
        self.submit_failure = submit_failure
        self.wait_failure = wait_failure

    def get_input_tensors_info(self):
        return [dict(name='image', shape=[1, 2, 3, 3], dtype=np.uint8)]

    def get_output_tensors_info(self):
        return [dict(name='cut', shape=[1, 4], dtype=np.float32)]

    def run(self, _inputs):
        raise AssertionError('A4 must never fall back to synchronous inference')

    def run_async(self, inputs, *, user_arg, output_buffer):
        self.events.append(('submit', user_arg))
        self.submissions.append((time.perf_counter(), user_arg, threading.current_thread().name))
        if user_arg == self.submit_failure:
            raise RuntimeError('injected_submit_failure')
        job = self.next_job
        self.next_job += 1
        outputs = ([output_buffer.view(np.float32).reshape(1, 4)]
                   if isinstance(output_buffer, np.ndarray) else output_buffer)
        self.jobs[job] = (user_arg, inputs, outputs)
        return job

    def wait(self, job):
        request_id, inputs, outputs = self.jobs[job]
        self.events.append(('wait', request_id))
        if request_id == self.wait_failure:
            raise RuntimeError('injected_wait_failure')
        # SDK completions happen in reverse order; waits still return the
        # submitted request's own caller-owned allocation and retained input.
        for pending_id in reversed(self.jobs):
            identity, feed, buffer = self.jobs[pending_id]
            buffer[0].fill(float(identity + int(feed[0][0, 0, 0, 0])))
        del self.jobs[job]
        return outputs


class TRT:
    inputs = ['cut']
    shapes = {'cut': (1, 4)}
    dtypes = {'cut': np.dtype(np.float32)}
    host_memory_policy = 'owned_test_staging'
    output_materialization_policy = 'owned_test_outputs'

    def __init__(self, *, delay=0.0, failure_at=None, outputs=None):
        self.delay = delay
        self.failure_at = failure_at
        self.outputs = outputs
        self.seen = []
        self.threads = []

    def prepare_inputs(self, feeds):
        self.staged = feeds['cut'].copy()
        self.seen.append(float(self.staged[0, 0]))
        self.threads.append(threading.current_thread().name)

    def run_prepared(self):
        if len(self.seen) == self.failure_at:
            raise RuntimeError('injected_consumer_failure')
        if self.delay:
            time.sleep(self.delay)
        if self.outputs is not None:
            return self.outputs
        return {'logits': self.staged.copy()}

    def run(self, feeds):
        self.prepare_inputs(feeds)
        return self.run_prepared()


def execute(engine=None, trt=None, **options):
    return runner._deepx_fifo_run(
        engine or AsyncEngine(), np.zeros((1, 2, 3, 3), dtype=np.uint8), trt or TRT(),
        frames=options.pop('frames', 5), warmup=options.pop('warmup', 0),
        queue_depth=options.pop('queue_depth', 3), deepx_runtime_mode='A4', **options,
    )


@pytest.mark.parametrize('frames', [0, 1, 3, 4, 5, 32])
@pytest.mark.parametrize('warmup', [0, 100])
def test_actual_split_counts_fill_drain_and_warmup_separation(frames, warmup):
    engine, trt = AsyncEngine(), TRT()
    result = execute(engine, trt, frames=frames, warmup=warmup)
    assert result['completed_work_units'] == result['accepted_work_units'] == frames
    assert result['warmup_completed_frames'] == warmup
    assert result['deepx_actual_pending_max'] == min(frames, 4)
    assert result['deepx_final_drain_complete'] is True
    assert result['deepx_runtime_observation']['slot_count'] == 4
    assert result['deepx_runtime_observation']['unreleased'] == 0
    assert result['boundary_copy_total'] == frames
    assert trt.seen == list(range(warmup)) + list(range(frames))
    assert not engine.jobs
    assert all(name == 'native-deepx-a4-producer' for _, _, name in engine.submissions)
    assert all(name == 'native-deepx-trt-consumer' for name in trt.threads)
    if frames:
        assert result['request_latency']['status'] == 'complete'
        assert result['request_latency']['count'] == frames
    if frames > 4:
        main = engine.events[2 * warmup:]
        assert main[:4] == [('submit', index) for index in range(4)]
        assert main[4:6] == [('wait', 0), ('submit', 4)]


def test_duration_admission_and_drain_share_the_count_executor(monkeypatch):
    observations = []
    original = runner.run_a4

    def observe(*args, **kwargs):
        value = original(*args, **kwargs)
        observations.append(value)
        return value

    monkeypatch.setattr(runner, 'run_a4', observe)
    engine, trt = AsyncEngine(), TRT(delay=0.003)
    result = execute(engine, trt, duration_s=0.02)
    assert len(observations) == 2  # zero warmup and main use the same executor
    main = observations[-1]
    assert result['completed_work_units'] == len(engine.submissions) == len(trt.seen)
    assert result['completed_work_units'] > 0
    assert max(stamp for stamp, _, _ in engine.submissions) < main['timer_start_s'] + 0.02
    assert main['timer_end_s'] >= main['timer_start_s'] + 0.02
    assert not engine.jobs
    assert result['request_latency']['status'] == 'unavailable'


@pytest.mark.parametrize('failure', ['submit', 'wait', 'consumer', 'decoder'])
def test_errors_join_workers_and_fail_without_claiming_completion(failure):
    engine = AsyncEngine(submit_failure=4 if failure == 'submit' else None,
                         wait_failure=1 if failure == 'wait' else None)
    trt = TRT(delay=0.001, failure_at=1 if failure == 'consumer' else None)

    class BrokenDecoder(_FakeCompletionRuntime):
        def process(self, outputs):
            raise RuntimeError('injected_decoder_failure')

    options = dict(frames=32)
    if failure == 'decoder':
        options.update(task='detection', completion_runtime=BrokenDecoder())
    with pytest.raises(RuntimeError, match='injected_') as caught:
        execute(engine, trt, **options)
    assert not any(thread.name in {'native-deepx-a4-producer', 'native-deepx-trt-consumer'}
                   for thread in threading.enumerate())
    assert caught.value.result['drained'] is False
    if failure in {'consumer', 'decoder'}:
        assert not engine.jobs  # all admitted SDK jobs drained even on consumer error
    else:
        assert caught.value.unsafe_runtime_state is True


def test_failed_warmup_is_identified_before_any_measured_admission():
    engine = AsyncEngine(submit_failure=4)
    with pytest.raises(RuntimeError, match='injected_submit_failure') as caught:
        execute(engine, TRT(), frames=32, warmup=100)
    assert caught.value.deepx_workload_phase == 'warmup'
    assert [request_id for _, request_id, _ in engine.submissions] == list(range(5))


@pytest.mark.parametrize('conflict', ['queue', 'shape', 'dtype', 'multi_boundary', 'name'])
def test_contract_conflicts_reject_before_sdk_submit(conflict, monkeypatch):
    engine, trt = AsyncEngine(), TRT()
    kwargs = {}
    if conflict == 'queue':
        kwargs['queue_depth'] = 2
    elif conflict == 'shape':
        trt.shapes = {'cut': (2, 2)}  # matching element count is insufficient
    elif conflict == 'dtype':
        trt.dtypes = {'cut': np.dtype(np.int32)}
    elif conflict == 'multi_boundary':
        trt.inputs = ['cut', 'other']
    else:
        monkeypatch.setattr(runner, '_STRICT_SPLIT_BOUNDARY', dict(
            name='different', output_names=['different'], shape=[1, 4], dtype='float32'))
    with pytest.raises((ValueError, RuntimeError), match='deepx_a4_'):
        execute(engine, trt, **kwargs)
    assert not engine.submissions


def test_exact_sdk_input_binding_adds_only_the_declared_batch_dimension():
    adapter = runner.NativeDeepXAsync(AsyncEngine())
    source = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    prepared = adapter.prepare_input(source)
    assert prepared.shape == (1, 2, 3, 3)
    assert prepared.tobytes() == source.tobytes()
    with pytest.raises(ValueError, match='shape_dtype'):
        adapter.prepare_input(source.astype(np.float32))
    with pytest.raises(ValueError, match='shape_dtype'):
        adapter.prepare_input(source.reshape(3, 2, 3))


@pytest.mark.parametrize('family', ['yolov7', 'yolo11', 'yolo26'])
def test_all_detector_endpoints_keep_existing_decoder_on_same_two_workers(family, tmp_path):
    if family == 'yolov7':
        from test_v282_hailo8_sentinel_evidence import _yolov7_case
        outputs, contract = _yolov7_case()
    elif family == 'yolo11':
        from test_v27923_decoded_completion import _outputs, _runtime
        outputs = _outputs()
        contract = _runtime(tmp_path, outputs).execution_contract
    else:
        from test_v279_native_three_stage import _decoded_completion_contract
        outputs, contract = _decoded_completion_contract()
    serial = postprocess.create_detection_completion_runtime(contract)
    expected = serial.process(outputs)
    runtime = postprocess.create_detection_completion_runtime(contract)
    warmup = postprocess.create_detection_completion_runtime(contract)
    trt = TRT(outputs=outputs)
    result = execute(trt=trt, frames=3, warmup=1, task='detection',
                     completion_runtime=runtime, warmup_completion_runtime=warmup)
    assert {key: value for key, value in runtime.last_result.items() if key != 'completed_count'} == {
        key: value for key, value in expected.items() if key != 'completed_count'
    }
    assert runtime.completed_count == 3
    assert warmup.completed_count == 1
    assert result['postprocess_completion_verified'] is True
    assert result['postprocess_completed_frames'] == 3
    assert result['deepx_worker_policy'] == 'one_a4_producer_fifo3_one_serial_trt_completion_consumer'
    assert set(trt.threads) == {'native-deepx-trt-consumer'}
    if family == 'yolo26':
        assert len(runtime.last_result['detections']) == 2  # no second NMS


def test_split_remote_mirror_is_exact():
    source = Path(runner.__file__)
    mirror = source.parents[1] / 'onnx_splitpoint_tool/resources/remote_scripts' / source.name
    assert mirror.read_bytes() == source.read_bytes()


@pytest.mark.parametrize('corrupt', [False, True])
def test_numerical_preflight_checks_mixed_inputs_and_reused_buffers(corrupt, tmp_path):
    class ValueEngine(AsyncEngine):
        def __init__(self):
            super().__init__()
            self.scratch = np.empty((1, 4), dtype=np.float32)

        def run(self, inputs):
            self.scratch.fill(float(inputs[0][0, 0, 0, 0]))
            return [self.scratch]

        def wait(self, job):
            request_id, inputs, outputs = self.jobs.pop(job)
            outputs[0].fill(float(inputs[0][0, 0, 0, 0]) + int(corrupt and request_id == 1))
            return outputs

    engine, trt = ValueEngine(), TRT()
    inputs = [np.full((1, 2, 3, 3), value, dtype=np.uint8) for value in range(32)]
    progress = {}
    kwargs = dict(image_paths=[str(tmp_path / f'{value}.png') for value in range(32)],
        benchmark_set=tmp_path, model_id='resnet50', task='classification',
        preprocess_mode='resize', letterbox_pad_value=0, completion_runtime_mode='auto', progress=progress)
    if corrupt:
        with pytest.raises(RuntimeError, match='deepx_numerical_request_failed:1:.*output_mismatch'):
            runner.deepx_split_numerical_preflight(engine, trt, inputs, **kwargs)
        assert progress['mixed_completions'] == 32
        assert progress['mixed_drained'] is True
    else:
        result = runner.deepx_split_numerical_preflight(engine, trt, inputs, **kwargs)
        assert result['status'] == 'passed'
        assert result['serial_reference_requests'] == result['mixed_completions'] == 32
        assert result['actual_pending_max'] == 4
        assert all(row['completed_task_equal'] for row in result['requests'])
    assert trt.seen[:32] == list(range(32))
    assert trt.seen[32:] == [value + int(corrupt and value == 1) for value in range(32)]
    assert not engine.jobs


def test_numerical_images_require_distinct_paths_and_include_timing(tmp_path):
    paths = [tmp_path / f'{index}.jpg' for index in range(2)]
    for path in paths:
        path.write_bytes(b'owned_test_image_path')
    manifest = tmp_path / 'images.json'
    manifest.write_text(json.dumps({'images': [str(path) for path in paths]}))
    assert runner._numerical_image_paths(str(manifest), str(paths[0])) == [str(path) for path in paths]
    manifest.write_text(json.dumps({'images': [str(paths[0]), str(paths[0])]}))
    with pytest.raises(ValueError, match='distinct'):
        runner._numerical_image_paths(str(manifest), str(paths[0]))
    manifest.write_text(json.dumps({'images': [str(paths[0])]}))
    with pytest.raises(ValueError, match='include_timing'):
        runner._numerical_image_paths(str(manifest), str(paths[1]))


def test_explicit_s1_retains_historical_single_job_api():
    class Historical:
        def __init__(self):
            self.calls = 0

        def run(self, _inputs):
            self.calls += 1
            return [np.ones((1, 4), dtype=np.float32)]

    engine = Historical()
    result = runner._deepx_fifo_run(engine, np.zeros(1, np.float32), TRT(),
                                    frames=5, warmup=2, queue_depth=2, deepx_runtime_mode='S1')
    assert engine.calls == 7
    assert result['deepx_runtime_mode'] == 'S1'
    assert result['deepx_max_inflight'] == 1


@pytest.mark.parametrize('numerical', [False, True])
def test_normal_worker_uses_one_engine_and_dumps_final_async_output(tmp_path, monkeypatch, numerical):
    from PIL import Image

    instances = []

    class ResidentEngine(AsyncEngine):
        def __init__(self, path):
            super().__init__()
            instances.append(self)
            self.serial_calls = 0
            self.closed = False

        def run(self, inputs):
            self.serial_calls += 1
            assert self.serial_calls <= (3 if numerical else 1), 'output dump must use final async output'
            return [np.full((1, 4), float(inputs[0][0, 0, 0, 0]), dtype=np.float32)]

        def wait(self, job):
            if not numerical:
                return super().wait(job)
            request_id, inputs, outputs = self.jobs.pop(job)
            outputs[0].fill(float(inputs[0][0, 0, 0, 0]))
            return outputs

        def close(self):
            assert not self.jobs
            self.closed = True

    class NativeTRT:
        inputs = ['cut']
        outputs = ['logits']
        shapes = {'cut': (1, 4)}
        dtypes = {'cut': np.dtype(np.float32)}

        def __init__(self, path):
            self.closed = False

        def prepare_inputs(self, feeds):
            self.staged = feeds['cut'].copy()

        def run_prepared(self):
            return {'logits': self.staged.copy()}

        def run(self, feeds):
            self.prepare_inputs(feeds)
            return self.run_prepared()

        def close(self):
            self.closed = True

    benchmark_set = tmp_path / 'benchmark_set'
    benchmark_set.mkdir()
    (benchmark_set / 'benchmark_set.json').write_text(json.dumps(dict(
        model_id='resnet50', benchmark_task='classification')))
    dxnn = tmp_path / 'part1.dxnn'
    dxnn.write_bytes(b'existing_sdk_artifact')
    engine = tmp_path / 'part2.engine'
    engine.write_bytes(b'existing_trt_artifact')
    contract_path = tmp_path / 'part1.json'
    contract_path.write_text(json.dumps({'input': dict(
        name='image', shape=[2, 3, 3], dtype='uint8', layout='HWC',
        task='classification', preprocess_mode_requested='auto',
        preprocess_mode_effective='resize', letterbox_pad_value_requested=114,
        letterbox_pad_value_effective=0)}))
    image = tmp_path / 'image.png'
    Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8)).save(image)
    output = tmp_path / 'result'
    monkeypatch.setitem(sys.modules, 'dx_engine', SimpleNamespace(InferenceEngine=ResidentEngine))
    monkeypatch.setattr(runner, 'NativeTRT', NativeTRT)
    monkeypatch.setattr(runner, '_find_part1_dxnn', lambda *args: dxnn)
    monkeypatch.setattr(runner, '_engine_path', lambda *args: engine)
    monkeypatch.setattr(runner, '_part1_output_contract_path', lambda *args: contract_path)
    argv = [runner.__file__, '--benchmark-set', str(benchmark_set),
        '--case', 'b001', '--image', str(image), '--out-dir', str(output),
        '--deepx-runtime-mode', 'A4', '--frames', '5', '--warmup', '0', '--dump-outputs']
    if numerical:
        second_image = tmp_path / 'second.png'
        Image.fromarray(np.ones((2, 3, 3), dtype=np.uint8)).save(second_image)
        image_manifest = tmp_path / 'numerical_images.json'
        image_manifest.write_text(json.dumps({'images': [str(image), str(second_image)]}))
        argv += ['--deepx-numerical-only', '--deepx-numerical-images', str(image_manifest)]
    monkeypatch.setattr(sys, 'argv', argv)
    assert runner.main() == 0
    assert len(instances) == 1
    assert instances[0].closed
    assert instances[0].serial_calls == (3 if numerical else 1)
    report = json.loads((output / 'deepx_native_fifo_e2e_results.json').read_text())
    assert report['deepx_runtime_mode'] == 'A4'
    assert report['native_command_contract']['runtime_options']['deepx_runtime_mode'] == 'A4'
    assert report['native_command_contract']['runtime_options']['deepx_max_inflight'] == 4
    if numerical:
        assert report['deepx_numerical_preflight']['status'] == 'passed'
        assert report['deepx_numerical_preflight']['serial_reference_requests'] == 2
        assert report['deepx_numerical_preflight']['mixed_completions'] == 2
        assert report['completed_work_units'] == 5
    manifest = json.loads(Path(report['native_output_manifest']).read_text())
    entry = manifest['outputs'][0]
    payload_path = Path(report['native_output_manifest']).parent / entry['file']
    assert np.array_equal(np.fromfile(payload_path, dtype=np.float32).reshape(1, 4),
                          np.full((1, 4), 0 if numerical else 4, np.float32))
