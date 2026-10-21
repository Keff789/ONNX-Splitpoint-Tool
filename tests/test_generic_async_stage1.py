"""Bounded Generic P1 admission, ownership, drain, and unchanged completion."""
from __future__ import annotations

import threading
import time
import json
import sys
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.generic_async import HailoAsyncStage1
from onnx_splitpoint_tool.runners.task_completion import measure_split_completion


class Processor:
    def __init__(self, *, fail=None, delay=0):
        self.completed_count = 0
        self.seen = []
        self.fail = fail
        self.delay = delay

    def process(self, outputs):
        if self.completed_count == self.fail:
            raise RuntimeError("consumer failed")
        if self.delay:
            time.sleep(self.delay)
        self.seen.append(int(outputs["cut"][0]))
        self.completed_count += 1


class SlotSession:
    """Driver stub: jobs can complete out of order and overwrite slot buffers."""
    timeout_ms = 100

    def __init__(self, *, submit_failure=None, completion_failure=None, delay=.0005):
        self.submit_failure = submit_failure
        self.completion_failure = completion_failure
        self.delay = delay
        self.submitted = []
        self.waited = []
        self.jobs = []
        self.pending = set()
        self.max_pending = 0

    def _create_reusable_binding_slot(self):
        done = threading.Event()
        done.set()
        return {"binding": {"cut": np.empty(1, dtype=np.int64)}, "done": done, "value": None}

    def _prepare_infer_inputs(self, inputs):
        return inputs

    def _submit_reusable_slot(self, slot, inputs, *, copy_inputs):
        assert copy_inputs is True
        assert slot["done"].is_set()
        value = inputs["input"]
        slot["value"] = value
        slot["done"].clear()
        self.pending.add(value)
        self.max_pending = max(self.max_pending, len(self.pending))
        self.submitted.append((value, time.perf_counter_ns()))

        def finish():
            # Even-numbered jobs deliberately finish after the next job.
            time.sleep(self.delay * (2 if value % 2 == 0 else 1))
            slot["binding"]["cut"][0] = value
            slot["done"].set()

        thread = threading.Thread(target=finish, name="generic-async-stub-device")
        self.jobs.append(thread)
        thread.start()
        if value == self.submit_failure:
            raise RuntimeError("submission failed after job admission")

    def _wait_reusable_slot(self, slot):
        if not slot["done"].wait(self.timeout_ms / 1000):
            raise TimeoutError("stub driver timeout")
        value = slot["value"]
        self.pending.discard(value)
        self.waited.append(value)
        if value == self.completion_failure:
            self.completion_failure = None
            raise RuntimeError("job callback failed")

    def _materialize_binding_outputs(self, binding, *, copy_outputs):
        assert copy_outputs is True
        return {name: value.copy() for name, value in binding.items()}


def producer(session, processor, *, inflight=3):
    admissions = []

    def prepare():
        value = len(admissions)
        admissions.append((value, processor.completed_count))
        return {"input": value}

    return HailoAsyncStage1(session, prepare, inflight=inflight), admissions


def run(adapter, processor, **kwargs):
    def forbidden():
        raise AssertionError("the synchronous producer must not run for the explicit async policy")

    return measure_split_completion(forbidden, lambda x: x, lambda x: x, processor,
                                    stage1_async=adapter, **kwargs)


def assert_drained(adapter, session):
    assert not adapter.pending
    assert not session.pending
    for thread in session.jobs:
        thread.join(timeout=1)
        assert not thread.is_alive()
    assert not any(thread.name.startswith("generic-completion-") for thread in threading.enumerate())


@pytest.mark.parametrize("inflight,depth", [(1, 1), (3, 1), (8, 2)])
def test_owned_outputs_order_and_warmup_drain(inflight, depth):
    session = SlotSession()
    processor = Processor(delay=.0005)
    adapter, admissions = producer(session, processor, inflight=inflight)
    report = run(adapter, processor, warmup=3, tasks=17, queue_depth=depth, timeout_s=3)
    assert processor.seen == list(range(20))
    assert admissions[3] == (3, 3), "measured device admission preceded complete warmup drain"
    assert session.submitted[3][1] >= report["measurement_start_ns"] >= report["warmup_end_ns"]
    assert session.max_pending == inflight
    assert report["stage1_execution"] == {"policy": "hailo_async_slots_v1", "inflight": inflight}
    assert report["worker_count"] == 2 and report["queue_depth"] == depth
    assert report["completed_task_count"] == 17
    assert report["processor_count_delta"] == 20
    assert_drained(adapter, session)


def test_duration_drains_every_admitted_job_including_pipeline_tail():
    session = SlotSession()
    processor = Processor(delay=.001)
    adapter, admissions = producer(session, processor, inflight=4)
    report = run(adapter, processor, warmup=0, tasks=None, duration_s=.035,
                 queue_depth=1, timeout_s=2)
    assert processor.seen == list(range(len(admissions)))
    assert report["completed_work_units"] == report["admitted_task_count"] == len(admissions)
    assert report["measurement_end_ns"] >= report["admission_stopped_ns"] >= report["admission_deadline_ns"]
    assert_drained(adapter, session)


@pytest.mark.parametrize("failure", ["submission", "callback", "consumer", "timeout"])
def test_failure_drains_owned_jobs_and_never_reports_success(failure):
    session = SlotSession(submit_failure=2 if failure == "submission" else None,
                          completion_failure=1 if failure == "callback" else None,
                          delay=.02 if failure == "timeout" else .001)
    processor = Processor(fail=1 if failure == "consumer" else None)
    adapter, _ = producer(session, processor, inflight=4)
    with pytest.raises(RuntimeError) as caught:
        run(adapter, processor, warmup=0, tasks=100, queue_depth=1,
            timeout_s=.005 if failure == "timeout" else 2)
    report = caught.value.partial_report
    assert report["status"] == "failed"
    assert "completed_task_fps" not in report
    assert report["stage1_execution"] == adapter.execution_policy
    assert_drained(adapter, session)


def test_unobserved_callback_cannot_be_reported_as_successful_cleanup():
    session = SlotSession()
    adapter, _ = producer(session, Processor(), inflight=1)
    slot = adapter.slots[0]
    slot["done"].clear()
    adapter.pending[id(slot)] = slot
    session._wait_reusable_slot = lambda slot: None
    with pytest.raises(RuntimeError, match="failed to drain"):
        adapter.drain()
    assert adapter.pending[id(slot)] is slot


def test_generated_main_selects_async_h10_and_keeps_generic_consumer(tmp_path, monkeypatch):
    from test_v2912_generic_split_completion import _module, _fixture, _descriptor
    from onnx_splitpoint_tool.runners.task_completion import valid_split_completion_repetitions
    import onnx_splitpoint_tool.runners.generic_async as async_module
    import glob
    import subprocess

    module = _module(monkeypatch, tmp_path / 'normal_runner_suite')
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_async", async_module)
    config, logits = _fixture(tmp_path)
    config["stage1_execution"] = {"policy": "hailo_async_slots_v1", "inflight": 4}
    config["original_run_cfg"]["stage1_provider"] = "hailo10h"
    config["input_binding"]["prepared_input_contract"]["normalization"] = "hef_quant_info_from_imagenet_float32"
    hef = Path(config["case_dir"]) / "hailo/hailo10h/part1/compiled.hef"
    hef.parent.mkdir(parents=True)
    hef.write_bytes(b"existing test HEF")
    config["expected_runtime_artifacts"]["part1_runtime"] = _descriptor(hef)
    calls = {"submitted": 0, "trt": 0, "pending": 0, "max_pending": 0}

    def forbidden(*args, **kwargs):
        raise AssertionError("async Generic cannot call synchronous P1, build, or reference inference")

    class Hailo:
        input_names = ["images"]
        timeout_ms = 100

        def __init__(self, path, **kwargs):
            self.hef_path = path

        def describe_io(self):
            return {}

        def dequantize_output(self, name, value):
            return value

        infer = forbidden

        def _create_reusable_binding_slot(self):
            event = threading.Event()
            event.set()
            return {"done": event, "binding": {"cut": logits.copy()}, "busy": False}

        def _prepare_infer_inputs(self, inputs):
            np.testing.assert_array_equal(inputs["images"], np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
            return inputs

        def _submit_reusable_slot(self, slot, inputs, *, copy_inputs):
            assert not slot["busy"] and copy_inputs
            slot["busy"] = True
            calls["submitted"] += 1
            calls["pending"] += 1
            calls["max_pending"] = max(calls["max_pending"], calls["pending"])

        def _wait_reusable_slot(self, slot):
            if slot["busy"]:
                calls["pending"] -= 1
                slot["busy"] = False

        def _materialize_binding_outputs(self, binding, *, copy_outputs):
            assert copy_outputs
            return {name: value.copy() for name, value in binding.items()}

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

    monkeypatch.setattr(module, "HailoInferModelSession", Hailo)
    monkeypatch.setattr(module, "NativeTRTSession", TRT)
    monkeypatch.setattr(module, "_create_session", forbidden)
    monkeypatch.setattr(module, "_bench", forbidden)
    monkeypatch.setattr(module, "_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(module, "ort", SimpleNamespace(SessionOptions=lambda: SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "hailo_platform", SimpleNamespace(VDevice=lambda: object()))
    original_glob = glob.glob
    monkeypatch.setattr(glob, "glob", lambda pattern, **kw: ["/dev/hailo-stub"] if pattern.startswith("/dev/") else original_glob(pattern, **kw))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="vdevice_ok", stderr=""))
    config_path = tmp_path / "async_config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "async_results"
    monkeypatch.setattr(sys, "argv", ["runner", "--generic-completion-config", str(config_path),
        "--out-dir", str(output), "--generic-completion-repetition", "1"])
    assert module.main() == 0
    assert Path(sys.modules['splitpoint_runners.generic_passthrough'].__file__).resolve().is_relative_to(
        tmp_path / 'normal_runner_suite')
    result = json.loads((output / "generic_completion_results.json").read_text())
    assert len(valid_split_completion_repetitions(result, config, result_root=output)) == 1
    assert result["stage1_execution"] == result["repetitions"][0]["stage1_execution"] == config["stage1_execution"]
    assert calls == {"submitted": 1100, "trt": 1100, "pending": 0, "max_pending": 4}
    assert module.main() == 0
    assert calls["submitted"] == 1100, "a same-policy validated repetition should remain resumable"
    frozen = copy.deepcopy(config)
    frozen["runtime_source_files"] = [{"path": "/bound/generic_async.py", "sha256": "a" * 64}]
    with pytest.raises(ValueError, match="runtime source binding"):
        valid_split_completion_repetitions(result, frozen)
    source_map = {"/bound/generic_async.py": "a" * 64}
    result["repetitions"][0]["runtime_source_binding"] = {
        "before": source_map, "after": source_map, "unchanged": True}
    assert len(valid_split_completion_repetitions(result, frozen)) == 1
    frozen["runtime_source_files"][0]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="runtime source binding"):
        valid_split_completion_repetitions(result, frozen)
