"""The observed DX-RT async API, slab ownership and Generic completion."""
import threading
import time
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1, _UNRESOLVED_DEEPX_OWNERS
from onnx_splitpoint_tool.runners.task_completion import measure_split_completion
from tests.test_generic_async_stage1 import Processor


class DeepXEngine:
    def __init__(self, *, delay=.0005, fault=None):
        self.delay, self.fault = delay, fault
        self.jobs = {}
        self.submissions = []
        self.waits = []
        self.threads = []

    def get_input_tensors_info(self):
        return [{"name": "image", "shape": [1, 1], "dtype": "int64"}]

    def get_output_tensors_info(self):
        return [{"name": "first", "shape": [1], "dtype": "int64"},
                {"name": "second", "shape": [3], "dtype": "float32"}]

    def run(self, *args):
        raise AssertionError("synchronous fallback must not run")

    def run_async(self, arrays, *, user_arg, output_buffer):
        assert arrays[0].shape == (1, 1)
        assert output_buffer.shape == (20,) and output_buffer.dtype == np.uint8
        first = np.ndarray((1,), dtype=np.int64, buffer=output_buffer, offset=0)
        second = np.ndarray((3,), dtype=np.float32, buffer=output_buffer, offset=8)
        assert second.ctypes.data == first.ctypes.data + 8
        done = threading.Event()
        self.jobs[user_arg] = ([first, second], done)
        self.submissions.append(user_arg)

        def finish():
            time.sleep(self.delay * (2 if user_arg % 2 == 0 else 1))
            first[0] = arrays[0][0, 0]
            second[:] = [first[0], first[0] + 1, first[0] + 2]
            done.set()

        thread = threading.Thread(target=finish, name="deepx-async-device-stub")
        self.threads.append(thread)
        thread.start()
        if self.fault == "submit" and user_arg == 1:
            raise RuntimeError("submission outcome unknown")
        return 0 if self.fault == "duplicate" else user_arg

    def wait(self, job_id):
        self.waits.append(job_id)
        outputs, done = self.jobs[job_id]
        assert done.wait(2)
        if self.fault == "wait" and job_id == 1:
            raise RuntimeError("wait outcome unknown")
        return [value.copy() for value in outputs] if self.fault == "alias" else outputs


class DeepXSession:
    def __init__(self, **kwargs):
        self._engine = DeepXEngine(**kwargs)

    def _prepare_async_inputs(self, inputs):
        return [inputs["input"]]

    def _materialize_outputs(self, outputs):
        return {"cut": outputs[0], "auxiliary": outputs[1]}


def configured(*, inflight=4, **kwargs):
    session = DeepXSession(**kwargs)
    values = []

    def prepare():
        values.append(len(values))
        return {"input": np.array([values[-1]], dtype=np.int64)}

    return session, DeepXAsyncStage1(session, prepare, inflight=inflight), values


def measure(adapter, processor, **kwargs):
    return measure_split_completion(lambda: pytest.fail("sync producer used"), lambda x: x,
        lambda x: x, processor, stage1_async=adapter, **kwargs)


def finish_jobs(session):
    for thread in session._engine.threads:
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("inflight", [1, 4])
def test_exact_batch_axis_multioutput_slabs_order_and_owned_copies(inflight):
    session, adapter, values = configured(inflight=inflight)
    processor = Processor(delay=.001)
    report = measure(adapter, processor, warmup=3, tasks=20, queue_depth=1, timeout_s=2)
    assert processor.seen == values == list(range(23))
    assert session._engine.waits == list(range(23))
    assert report["stage1_execution"] == {"policy": "deepx_async_slots_v1", "inflight": inflight}
    assert report["stage1_peak_uncollected_jobs"] == inflight
    assert len({slot["buffer"].ctypes.data for slot in adapter.slots}) == inflight
    assert not adapter.pending
    finish_jobs(session)


def test_duration_drains_all_accepted_deepx_work():
    session, adapter, values = configured()
    processor = Processor(delay=.001)
    report = measure(adapter, processor, warmup=0, tasks=None, duration_s=.03,
                     queue_depth=1, timeout_s=2)
    assert processor.seen == values
    assert report["completed_work_units"] == report["admitted_task_count"] == len(values)
    assert report["measurement_end_ns"] >= report["admission_stopped_ns"] >= report["admission_deadline_ns"]
    assert not adapter.pending
    finish_jobs(session)


@pytest.mark.parametrize("fault", ["submit", "duplicate", "wait", "alias"])
def test_sdk_errors_fail_closed_without_reusing_unknown_buffers(fault):
    session, adapter, _ = configured(fault=fault)
    with pytest.raises(RuntimeError) as caught:
        measure(adapter, Processor(), warmup=0, tasks=10, timeout_s=2)
    assert caught.value.partial_report["status"] == "failed"
    assert "completed_task_fps" not in caught.value.partial_report
    finish_jobs(session)
    if fault != "alias":
        assert adapter in _UNRESOLVED_DEEPX_OWNERS
        assert adapter.pending
        assert any(slot["state"].endswith("unresolved") for slot in adapter.pending.values())
    else:
        assert not adapter.pending, "invalid output alias follows confirmed SDK completion"
    assert len(session._engine.waits) == len(set(session._engine.waits)), "unknown jobs must not be waited twice"


def test_consumer_failure_drains_known_jobs():
    session, adapter, _ = configured()
    with pytest.raises(RuntimeError):
        measure(adapter, Processor(fail=1), warmup=0, tasks=100, queue_depth=1, timeout_s=2)
    assert not adapter.pending
    assert set(session._engine.waits) == set(session._engine.submissions)
    finish_jobs(session)


def test_missing_async_api_and_changed_input_fail_without_sync_fallback():
    session = DeepXSession()
    session._engine.run_async = None
    with pytest.raises(ValueError, match="no synchronous fallback"):
        DeepXAsyncStage1(session, lambda: {}, inflight=4)
    session = DeepXSession()
    adapter = DeepXAsyncStage1(session, lambda: {"input": np.zeros((1,), dtype=np.float32)}, inflight=4)
    with pytest.raises(ValueError, match="shape/dtype/contiguity"):
        adapter.submit()
    assert not adapter.pending and not session._engine.submissions


def test_global_timeout_retains_owner_without_inventing_a_wait_timeout():
    session, adapter, _ = configured()
    release = threading.Event()
    original_wait = session._engine.wait

    def delayed_wait(job_id):
        assert release.wait(3)
        return original_wait(job_id)

    session._engine.wait = delayed_wait
    try:
        with pytest.raises(RuntimeError) as caught:
            measure(adapter, Processor(), warmup=0, tasks=10, timeout_s=.1)
        assert caught.value.partial_report["status"] == "failed"
        assert adapter in _UNRESOLVED_DEEPX_OWNERS
        assert adapter.pending
    finally:
        release.set()
        for thread in list(threading.enumerate()):
            if thread.name.startswith("generic-completion-"):
                thread.join(timeout=2)
        finish_jobs(session)
    assert not adapter.pending


def test_generated_main_uses_real_generic_deepx_input_and_output_adapters(tmp_path, monkeypatch):
    from tests.test_v2912_generic_split_completion import _module, _fixture
    from onnx_splitpoint_tool.runners.task_completion import valid_split_completion_repetitions
    from onnx_splitpoint_tool.runners import generic_async

    module = _module(monkeypatch)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_async", generic_async)
    config, logits = _fixture(tmp_path)
    config["stage1_execution"] = {"policy": "deepx_async_slots_v1", "inflight": 4}
    calls = {"submitted": 0, "waited": 0, "trt": 0}

    def forbidden(*args, **kwargs):
        raise AssertionError("async Generic cannot use sync inference, reference inference or a build")

    class Engine:
        def __init__(self, path):
            assert path == config["expected_runtime_artifacts"]["part1_runtime"]["path"]
            self.jobs = {}

        run = forbidden

        def get_input_tensors_info(self):
            return [{"name": "runtime_image", "shape": [1, 2, 2, 3], "dtype": "uint8"}]

        def get_output_tensors_info(self):
            return [{"name": "runtime_cut", "shape": [1, 6], "dtype": "float32"}]

        def run_async(self, arrays, *, user_arg, output_buffer):
            np.testing.assert_array_equal(arrays[0], np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3))
            output = np.ndarray((1, 6), dtype=np.float32, buffer=output_buffer)
            output[:] = logits
            self.jobs[user_arg] = output
            calls["submitted"] += 1
            return user_arg

        def wait(self, job_id):
            calls["waited"] += 1
            return [self.jobs.pop(job_id)]

    class TRT:
        def __init__(self, *args, **kwargs):
            assert kwargs["allow_build"] is False
            self.engine_path = Path(config["expected_runtime_artifacts"]["engine"]["path"])
            self.build_info = {"cache_hit": True, "build_disabled": True}
            self.input_names = ["cut"]

        def describe_io(self):
            return {"inputs": {"cut": {"shape": [1, 6], "dtype": "float32"}}}

        def get_outputs(self):
            return [SimpleNamespace(name="logits")]

        def run(self, _, feeds):
            calls["trt"] += 1
            return [feeds["cut"]]

    monkeypatch.setitem(sys.modules, "dx_engine", SimpleNamespace(InferenceEngine=Engine))
    monkeypatch.setattr(module, "NativeTRTSession", TRT)
    monkeypatch.setattr(module, "_create_session", forbidden)
    monkeypatch.setattr(module, "_bench", forbidden)
    monkeypatch.setattr(module, "_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(module, "ort", SimpleNamespace(SessionOptions=lambda: SimpleNamespace()))
    path = tmp_path / "async_config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "async_deepx_result"
    monkeypatch.setattr(sys, "argv", ["runner", "--generic-completion-config", str(path),
        "--out-dir", str(output), "--generic-completion-repetition", "1"])
    assert module.main() == 0
    result = json.loads((output / "generic_completion_results.json").read_text())
    assert len(valid_split_completion_repetitions(result, config, result_root=output)) == 1
    assert result["repetitions"][0]["stage1_peak_uncollected_jobs"] == 4
    assert calls == {"submitted": 1100, "waited": 1100, "trt": 1100}
