"""Exercise the real Native A4 adapter, faking only SDK/TRT device calls.

The borrowed-input, reverse-completion and reused-output cases come from the
completed DeepX diagnosis.  No report-directory code is imported by these tests.
"""
from __future__ import annotations

from collections import OrderedDict
import gc
import importlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from onnx_splitpoint_tool import native_deepx_async as a4
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion


def input_value(seed):
    return np.full((1, 1, 1, 1), seed, dtype=np.uint8)


def expected(seed, shape=(1, 6), dtype=np.float32, index=0):
    values = np.roll(np.arange(np.prod(shape)) + seed * 3 + index, seed % np.prod(shape))
    return values.astype(dtype).reshape(shape)


class SlotRuntime:
    """One fake engine; SDK waits finish outstanding jobs in reverse order."""
    def __init__(self, outputs=None, *, fail_submit=None, fail_wait=None,
                 returned_ids=None, bad_output=None, wait_delay=0):
        self.inputs_info = [{"name": "input", "shape": [1, 1, 1, 1], "dtype": np.uint8, "elem_size": 1}]
        self.outputs_info = outputs or [{"name": "sdk_boundary", "shape": [1, 6], "dtype": np.float32, "elem_size": 4}]
        self.pending = OrderedDict()
        self.events = []
        self.finished_ids = []
        self.submit_stamps = []
        self.slot_addresses = set()
        self.next_job = 100
        self.submit_count = self.wait_count = self.alive_checks = self.peak = 0
        self.fail_submit, self.fail_wait = fail_submit, fail_wait
        self.returned_ids = iter(returned_ids) if returned_ids is not None else None
        self.bad_output, self.wait_delay = bad_output, wait_delay
        self.shared_outputs = [np.empty(item["shape"], dtype=item["dtype"]) for item in self.outputs_info]

    def get_input_tensors_info(self):
        return self.inputs_info

    def get_output_tensors_info(self):
        return self.outputs_info

    def get_input_tensor_names(self):
        return [item["name"] for item in self.inputs_info]

    def get_output_tensor_names(self):
        return [item["name"] for item in self.outputs_info]

    def get_input_tensor_sizes(self):
        return [np.prod(item["shape"]) * np.dtype(item["dtype"]).itemsize for item in self.inputs_info]

    def get_output_tensor_sizes(self):
        return [np.prod(item["shape"]) * np.dtype(item["dtype"]).itemsize for item in self.outputs_info]

    def get_input_size(self):
        return sum(self.get_input_tensor_sizes())

    def get_output_size(self):
        return sum(self.get_output_tensor_sizes())

    def run_async(self, inputs, user_arg=None, output_buffer=None):
        # Capture entry before the fake's metadata and output-view preparation.
        # Those operations previously made this an accidental later timestamp.
        self.submit_stamps.append(time.perf_counter())
        assert isinstance(inputs, list) and all(isinstance(value, np.ndarray) for value in inputs)
        assert isinstance(output_buffer, np.ndarray) and output_buffer.dtype == np.uint8
        assert output_buffer.shape == (self.get_output_size(),) and output_buffer.flags.c_contiguous
        # Match the real SDK C++ ABI: one base pointer; each output is at its
        # cumulative native byte offset, including mixed element types.
        offsets = np.cumsum([0] + self.get_output_tensor_sizes()[:-1])
        outputs = [np.ndarray(info["shape"], dtype=info["dtype"], buffer=output_buffer, offset=int(offset))
                   for info, offset in zip(self.outputs_info, offsets)]
        self.submit_count += 1
        self.events.append(("submit", user_arg))
        if self.submit_count == self.fail_submit:
            raise RuntimeError("injected_submit_failure")
        for output, info in zip(outputs, self.outputs_info):
            assert output.shape == tuple(info["shape"]) and output.dtype == info["dtype"]
            assert output.flags.c_contiguous
        addresses = tuple(value.ctypes.data for value in outputs)
        assert all(addresses != tuple(value.ctypes.data for value in pending["outputs"])
                   for pending in self.pending.values()), "slot reused before wait"
        self.slot_addresses.add(addresses)
        job_id = next(self.returned_ids) if self.returned_ids is not None else self.next_job
        self.next_job += 1
        if type(job_id) is int and job_id >= 0 and job_id not in self.pending:
            self.pending[job_id] = {"input": weakref.ref(inputs[0]), "outputs": outputs,
                                    "request_id": user_arg, "done": False}
        self.peak = max(self.peak, len(self.pending))
        return job_id

    def wait(self, job_id):
        self.wait_count += 1
        self.events.append(("wait", job_id))
        if self.wait_count == self.fail_wait:
            raise TimeoutError("injected_wait_failure")
        if self.wait_delay:
            time.sleep(self.wait_delay)
        for finished_id, pending in reversed(list(self.pending.items())):
            if pending["done"]:
                continue
            source = pending["input"]()
            assert source is not None, "input died before SDK completion"
            self.alive_checks += 1
            seed = int(source.ravel()[0])
            for index, output in enumerate(pending["outputs"]):
                np.copyto(output, expected(seed, output.shape, output.dtype, index))
            pending["done"] = True
            self.finished_ids.append(finished_id)
        job = self.pending.pop(job_id)
        outputs = job["outputs"]
        if self.bad_output == "shared":
            for shared, value in zip(self.shared_outputs, outputs):
                np.copyto(shared, value)
            return self.shared_outputs
        if self.bad_output == "shape":
            return [outputs[0].reshape(2, 3)] + outputs[1:]
        if self.bad_output == "dtype":
            return [outputs[0].astype(np.float64)] + outputs[1:]
        if self.bad_output == "missing":
            return outputs[:-1]
        if self.bad_output == "order":
            return list(reversed(outputs))
        return outputs


class FakeTRT:
    def __init__(self, *, gate=None, entered=None, fail=None):
        self.inputs = ["boundary"]
        self.shapes = {"boundary": (1, 6)}
        self.dtypes = {"boundary": np.dtype(np.float32)}
        self.staging = np.empty((1, 6), dtype=np.float32)
        self.logits = np.empty((1, 6), dtype=np.float32)
        self.received = []
        self.gate, self.entered, self.fail = gate, entered, fail

    def prepare_inputs(self, feeds):
        value = feeds["boundary"]
        assert value.flags.owndata and value.flags.c_contiguous
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(3), "consumer test gate not released"
        if self.fail == "staging" and len(self.received) == 2:
            raise RuntimeError("injected_staging_failure")
        self.received.append(value.copy())
        np.copyto(self.staging, value)

    def run_prepared(self):
        if self.fail == "execute" and len(self.received) == 3:
            raise RuntimeError("injected_execute_failure")
        np.copyto(self.logits, self.staging)
        return {"logits": self.logits}


@pytest.fixture
def deep(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("native_deepx_trt_e2e_from_benchmarkset")
    monkeypatch.setattr(module, "_STRICT_SPLIT_BOUNDARY", {
        "name": "sdk_boundary", "output_names": ["sdk_boundary"],
        "shape": [1, 6], "dtype": "float32", "metadata_sha256": "existing-test-contract",
    })
    return module


def bounded(function, timeout=5):
    """A worker deadlock is a bounded test failure, with live-peer checking."""
    box = {}
    before = set(threading.enumerate())
    def execute():
        try:
            box["result"] = function()
        except BaseException as exc:
            box["error"] = exc
    thread = threading.Thread(target=execute, name="offline-a4-test", daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "A4 returned no result; possible deadlock"
    assert not [peer for peer in threading.enumerate() if peer not in before and peer.is_alive()]
    if "error" in box:
        raise box["error"]
    return box["result"]


def split_callbacks(deep, trt, processor):
    def mapping(request_id, raw):
        name, value, evidence = deep._map_deepx_output_to_trt(raw, trt)
        assert evidence["boundary_copy_count"] == 1
        assert not any(np.shares_memory(value, output) for output in raw)
        return name, value
    def consume(request_id, payload):
        name, value = payload
        trt.prepare_inputs({name: value})
        return trt.run_prepared()
    def complete(request_id, outputs):
        return processor.process(outputs)
    return {"map_outputs": mapping, "consume": consume, "complete": complete}


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 23])
@pytest.mark.parametrize("split", [False, True])
def test_counts_order_refill_owned_copy_and_task_completion(deep, count, split):
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    processor = ClassificationCompletion()
    trt = FakeTRT()
    callbacks = (split_callbacks(deep, trt, processor) if split else {
        "complete": lambda request_id, raw: processor.process({"logits": raw[0]})})
    result = bounded(lambda: a4.run_a4(adapter, lambda request_id: input_value(request_id + 1),
                                      count=count, capture=True, **callbacks))
    assert result["submitted_count"] == result["completed_count"] == processor.completed_count == count
    assert result["request_ids"] == list(range(count))
    assert result["actual_pending_max"] == min(4, count)
    assert result["drained"] is True and result["errors"] == []
    assert result["observation"]["pending"] == result["observation"]["unreleased"] == 0
    assert result["observation"]["queue_depth"] == (3 if split else 0)
    assert len(adapter.slots) == 4 and runtime.peak == min(4, count)
    assert runtime.wait_count == runtime.alive_checks == count
    assert not runtime.pending
    for request_id, record in result["captured"].items():
        value = record["logits"] if split else record[0]
        np.testing.assert_array_equal(value, expected(request_id + 1))
    if count:
        assert result["request_latency"]["status"] == "complete"
        assert result["request_latency"]["count"] == count
        assert result["timer_end_s"] == result["completion_times_s"][-1]
        assert result["timer_start_s"] <= min(runtime.submit_stamps)
        assert result["drain_end_s"] >= result["timer_end_s"]
    else:
        assert result["fps_makespan"] is None and result["makespan_s"] == 0
        assert runtime.events == []
    if count > 4:
        assert [kind for kind, _ in runtime.events[:5]] == ["submit"] * 4 + ["wait"]
        assert runtime.events[5][0] == "submit", "artificial four-job batch drain"
        assert runtime.finished_ids[:4] == [103, 102, 101, 100]


@pytest.mark.parametrize("warmup", [0, 100])
def test_warmup_drains_separately_and_reuses_exact_four_slots(warmup):
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    slots = [tuple(value.ctypes.data for value in slot.outputs) for slot in adapter.slots]
    warm = a4.run_a4(adapter, lambda request_id: input_value(11), count=warmup)
    assert warm["completed_count"] == warmup and not runtime.pending
    main = a4.run_a4(adapter, lambda request_id: input_value(31 + request_id), count=5, capture=True)
    assert main["completed_count"] == main["observation"]["submitted"] == 5
    assert main["request_ids"] == list(range(5))
    assert [tuple(value.ctypes.data for value in slot.outputs) for slot in adapter.slots] == slots
    for request_id, raw in main["captured"].items():
        np.testing.assert_array_equal(raw[0], expected(31 + request_id))
    assert runtime.submit_count == warmup + 5


@pytest.mark.parametrize("split", [False, True])
def test_regular_thousand_task_phase_completes_every_accepted_request(deep, split):
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    callbacks = split_callbacks(deep, FakeTRT(), ClassificationCompletion()) if split else {}
    result = bounded(lambda: a4.run_a4(adapter, lambda request_id: input_value(request_id % 200),
                                      count=1000, **callbacks))
    assert result["completed_count"] == result["submitted_count"] == 1000
    assert result["observation"]["wait_completed"] == 1000
    assert result["request_latency"]["count"] == 1000
    assert result["drained"] is True and result["actual_pending_max"] == 4


def test_full_multioutput_heterogeneous_shapes_dtypes_and_order():
    dtypes = [np.float32, np.float16, np.uint8, np.int8, np.uint16, np.int16, np.int32, np.int64, np.float64]
    outputs = [{"name": f"head{index}", "shape": [1, index + 1, 2], "dtype": dtype,
                "elem_size": np.dtype(dtype).itemsize} for index, dtype in enumerate(dtypes)]
    runtime = SlotRuntime(outputs)
    adapter = a4.NativeDeepXAsync(runtime, output_names=[item["name"] for item in outputs])
    result = a4.run_a4(adapter, lambda request_id: input_value(request_id + 2), count=9, capture=True)
    for request_id, raw in result["captured"].items():
        for index, (value, contract) in enumerate(zip(raw, adapter.output_contracts)):
            assert value.shape == contract.shape and value.dtype == contract.dtype
            np.testing.assert_array_equal(value, expected(request_id + 2, value.shape, value.dtype, index))
    assert len(runtime.slot_addresses) == 4
    assert adapter.output_offsets == tuple(np.cumsum([0] + runtime.get_output_tensor_sizes()[:-1]))
    for slot in adapter.slots:
        assert slot.buffer.nbytes == sum(contract.nbytes for contract in adapter.output_contracts)
        for offset, value in zip(adapter.output_offsets, slot.outputs):
            assert value.ctypes.data == slot.buffer.ctypes.data + offset
    for slot in adapter.slots:
        for value in slot.outputs:
            value.fill(0)
    np.testing.assert_array_equal(result["last_outputs"][0], expected(10, (1, 1, 2)))


def test_prepare_input_only_adds_exact_batch_dimension_and_submit_is_strict():
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    hwc = np.array([[[17]]], dtype=np.uint8)
    bound = adapter.prepare_input(hwc)
    assert bound.shape == (1, 1, 1, 1) and bound.dtype == hwc.dtype
    assert bound.tobytes() == hwc.tobytes() and np.shares_memory(bound, hwc)
    with pytest.raises(ValueError, match="shape_dtype"):
        adapter.submit(0, hwc)
    for bad in (bound.astype(np.float32), bound.reshape(1), np.zeros((1, 1, 1, 2), dtype=np.uint8)):
        with pytest.raises(ValueError, match="shape_dtype"):
            adapter.prepare_input(bad)
    assert runtime.submit_count == 0


def test_missing_dtype_and_nonunit_batch_are_explicitly_unsupported():
    runtime = SlotRuntime()
    runtime.outputs_info[0]["dtype"] = None
    with pytest.raises(ValueError, match="dtype_invalid"):
        a4.NativeDeepXAsync(runtime)
    runtime = SlotRuntime()
    runtime.inputs_info[0]["shape"] = [2, 1, 1, 1]
    with pytest.raises(ValueError, match="batch_one"):
        a4.NativeDeepXAsync(runtime)
    assert runtime.submit_count == 0


def test_named_inputs_use_sdk_order_and_reject_unknown_names_before_submission():
    runtime = SlotRuntime()
    runtime.inputs_info += [{"name": "extra", "shape": [1, 2], "dtype": np.int32, "elem_size": 4}]
    adapter = a4.NativeDeepXAsync(runtime, input_names=["input", "extra"])
    extra = np.array([[11, 12]], dtype=np.int32)
    bound = adapter.prepare_input({"extra": extra, "input": input_value(7)})
    assert len(bound) == 2 and bound[1] is extra
    with pytest.raises(ValueError, match="input_names"):
        adapter.submit(0, {"input": input_value(7), "wrong": extra})
    assert runtime.submit_count == 0
    result = a4.run_a4(adapter, lambda request_id: bound, count=1)
    np.testing.assert_array_equal(result["last_outputs"][0], expected(7))


@pytest.mark.parametrize("conflict", ["name", "order", "size", "shape", "dtype", "dynamic", "duplicate"])
def test_preflight_contract_conflicts_never_submit(conflict):
    runtime = SlotRuntime()
    kwargs = {}
    if conflict == "name":
        kwargs["output_names"] = ["other"]
    elif conflict == "order":
        runtime.outputs_info *= 2
        runtime.outputs_info = [dict(item, name=name) for item, name in zip(runtime.outputs_info, ("a", "b"))]
        kwargs["output_names"] = ["b", "a"]
    elif conflict == "size":
        runtime.get_output_tensor_sizes = lambda: [25]
    elif conflict == "shape":
        kwargs["output_templates"] = [np.empty((2, 3), dtype=np.float32)]
    elif conflict == "dtype":
        kwargs["output_templates"] = [np.empty((1, 6), dtype=np.float64)]
    elif conflict == "dynamic":
        runtime.outputs_info[0]["shape"] = [1, -1]
    else:
        runtime.outputs_info *= 2
    with pytest.raises(ValueError):
        a4.NativeDeepXAsync(runtime, **kwargs)
    assert runtime.submit_count == 0


@pytest.mark.parametrize("bad_output", ["shared", "shape", "dtype", "missing", "order"])
def test_wait_metadata_and_exact_slot_alias_failures_are_safely_drained(bad_output):
    outputs = [{"name": name, "shape": [1, 6], "dtype": np.float32, "elem_size": 4} for name in ("a", "b")]
    runtime = SlotRuntime(outputs, bad_output=bad_output)
    adapter = a4.NativeDeepXAsync(runtime)
    with pytest.raises(a4.NativeDeepXAsyncError) as caught:
        a4.run_a4(adapter, lambda request_id: input_value(request_id), count=5)
    result = caught.value.result
    assert result["drained"] is False
    assert result["observation"]["pending"] == result["observation"]["unreleased"] == 0
    assert result["unsafe_runtime_state"] is False
    assert not runtime.pending


@pytest.mark.parametrize("job_id", [-1, None, True, "100", 1.5, np.int64(100)])
def test_invalid_job_id_retains_uncertain_inputs_and_slots(job_id):
    runtime = SlotRuntime(returned_ids=[job_id])
    adapter = a4.NativeDeepXAsync(runtime)
    source = input_value(7)
    ref = weakref.ref(source)
    with pytest.raises(a4.NativeDeepXAsyncError) as caught:
        a4.run_a4(adapter, lambda request_id: source, count=5)
    del source
    gc.collect()
    assert ref() is not None
    assert caught.value.result["unsafe_runtime_state"] is True
    assert caught.value.result["observation"]["pending"] == 1
    with pytest.raises(RuntimeError, match="before_full_drain"):
        adapter.reset()


def test_duplicate_active_job_id_and_request_lifecycle_are_rejected():
    runtime = SlotRuntime(returned_ids=[100, 100])
    adapter = a4.NativeDeepXAsync(runtime)
    adapter.submit(0, input_value(1))
    with pytest.raises(ValueError, match="request_id"):
        adapter.submit(0, input_value(2))
    with pytest.raises(RuntimeError, match="release_before"):
        adapter.release(0)
    with pytest.raises(ValueError, match="job_id_duplicate"):
        adapter.submit(1, input_value(2))
    assert adapter.drain()
    assert adapter.report()["unsafe_runtime_state"] is True


def test_four_ready_slots_cannot_be_reused_until_owned_mapping_release():
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    for request_id in range(4):
        adapter.submit(request_id, input_value(request_id))
        completed_id, _ = adapter.wait_oldest()
        assert completed_id == request_id
    with pytest.raises(RuntimeError, match="reused_before_release"):
        adapter.submit(4, input_value(5))
    adapter.release(0)
    adapter.submit(4, input_value(5))
    request_id, _ = adapter.wait_oldest()
    adapter.release(request_id)
    assert adapter.drain() == []


@pytest.mark.parametrize("failure", ["submit", "wait", "mapping", "staging", "execute", "completion"])
def test_failure_paths_drain_known_jobs_and_join_both_workers(deep, failure):
    runtime = SlotRuntime(fail_submit=3 if failure == "submit" else None,
                          fail_wait=3 if failure == "wait" else None)
    adapter = a4.NativeDeepXAsync(runtime)
    trt = FakeTRT(fail=failure)
    processor = ClassificationCompletion()
    callbacks = split_callbacks(deep, trt, processor)
    if failure == "mapping":
        real = callbacks["map_outputs"]
        def mapping(request_id, raw):
            if request_id == 2:
                raise RuntimeError("injected_mapping_failure")
            return real(request_id, raw)
        callbacks["map_outputs"] = mapping
    elif failure == "completion":
        def completion(request_id, outputs):
            if request_id == 2:
                raise RuntimeError("injected_completion_failure")
            return processor.process(outputs)
        callbacks["complete"] = completion
    with pytest.raises(a4.NativeDeepXAsyncError) as caught:
        bounded(lambda: a4.run_a4(adapter, lambda request_id: input_value(request_id + 1),
                                   count=50, **callbacks))
    result = caught.value.result
    assert not result["drained"] and result["errors"]
    assert result["completed_count"] < 50
    assert any("injected_" in error for error in result["errors"])
    assert result["unsafe_runtime_state"] is (failure in {"submit", "wait", "staging", "execute"})
    assert result["observation"]["pending"] == (1 if failure in {"submit", "wait"} else 0)
    assert len(runtime.pending) == (1 if failure == "wait" else 0)


def test_aliasing_mapper_fails_before_slot_refill():
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    with pytest.raises(a4.NativeDeepXAsyncError, match="aliases_runtime_output"):
        a4.run_a4(adapter, lambda request_id: input_value(request_id), count=5,
                  map_outputs=lambda request_id, raw: raw)
    assert runtime.submit_count == 4 and not runtime.pending


def test_queue_three_backpressure_and_cancel_cannot_deadlock_sentinel(deep):
    gate, entered, cancelled = threading.Event(), threading.Event(), threading.Event()
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    callbacks = split_callbacks(deep, FakeTRT(gate=gate, entered=entered), ClassificationCompletion())
    box = {}
    def execute():
        try:
            a4.run_a4(adapter, lambda request_id: input_value(request_id), count=100,
                      cancel_event=cancelled, **callbacks)
        except a4.NativeDeepXAsyncError as exc:
            box["result"] = exc.result
    outer = threading.Thread(target=execute)
    outer.start()
    try:
        assert entered.wait(3)
        time.sleep(.15)
        assert 4 <= runtime.submit_count < 100
        assert runtime.peak == 4
        cancelled.set()
    finally:
        gate.set()
        outer.join(5)
    assert not outer.is_alive()
    assert box["result"]["drained"] is False
    assert box["result"]["observation"]["observed_max_queue"] == 3
    assert box["result"]["observation"]["pending"] == 0
    assert not runtime.pending
    assert not any(peer.name.startswith("native-deepx-") for peer in threading.enumerate())


@pytest.mark.parametrize("split", [False, True])
def test_duration_admission_stops_before_deadline_and_all_accepted_jobs_drain(deep, split):
    runtime = SlotRuntime(wait_delay=.001)
    adapter = a4.NativeDeepXAsync(runtime)
    callbacks = split_callbacks(deep, FakeTRT(), ClassificationCompletion()) if split else {}
    result = bounded(lambda: a4.run_a4(adapter, lambda request_id: input_value(request_id % 200),
                                      count=1, duration_s=.025, **callbacks))
    assert result["completed_count"] == result["submitted_count"] > 1
    assert result["drained"] is True and not runtime.pending
    assert len(result["submit_admission_times_s"]) == runtime.submit_count
    assert max(result["submit_admission_times_s"]) < result["admission_deadline_s"]
    assert result["admission_deadline_s"] == result["timer_start_s"] + .025
    assert all(admitted <= entered for admitted, entered in zip(result["submit_admission_times_s"], runtime.submit_stamps))
    assert result["timer_end_s"] >= max(runtime.submit_stamps)
    assert result["request_latency"]["status"] == "unavailable"
    assert result["request_latency"]["reason"] == "duration_workload_not_instrumented"


def test_deadline_crossed_during_input_acquisition_never_submits():
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    def slow_input(request_id):
        time.sleep(.02)
        return input_value(1)
    result = a4.run_a4(adapter, slow_input, duration_s=.005)
    assert result["submitted_count"] == result["completed_count"] == 0
    assert runtime.submit_count == 0


def test_deadline_crossed_during_submit_validation_does_not_admit_or_hold_slot(monkeypatch):
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    real = adapter._inputs
    def slow_validation(*args, **kwargs):
        result = real(*args, **kwargs)
        time.sleep(.01)
        return result
    monkeypatch.setattr(adapter, "_inputs", slow_validation)
    result = a4.run_a4(adapter, lambda request_id: input_value(1), duration_s=.005)
    assert result["submitted_count"] == result["completed_count"] == 0
    assert runtime.submit_count == 0 and adapter.report()["unreleased"] == 0


@pytest.mark.parametrize("crossing", ["input", "validation", "sdk_lookup", "instrumentation", "refill"])
def test_deadline_admission_decisions_with_deterministic_clock(monkeypatch, crossing):
    class Clock:
        now = 100.0
        def perf_counter(self):
            return self.now
    clock = Clock()
    entered = []
    class ClockedRuntime(SlotRuntime):
        @property
        def run_async(self):
            if crossing == "sdk_lookup" and self.armed:
                clock.now = 101.0
            return self.submit_native
        def submit_native(self, *args, **kwargs):
            entered.append(clock.now)
            return super().run_async(*args, **kwargs)
        def wait(self, job_id):
            result = super().wait(job_id)
            if crossing == "refill":
                clock.now = 101.0
            return result
    runtime = ClockedRuntime()
    runtime.armed = False
    adapter = a4.NativeDeepXAsync(runtime)
    runtime.armed = True
    real_inputs = adapter._inputs
    def validate(*args, **kwargs):
        result = real_inputs(*args, **kwargs)
        if crossing == "validation":
            clock.now = 101.0
        return result
    def source(request_id):
        if crossing == "input":
            clock.now = 101.0
        return input_value(request_id)
    def instrument(request_id):
        if crossing == "instrumentation":
            clock.now = 101.0
    monkeypatch.setattr(adapter, "_inputs", validate)
    monkeypatch.setattr(a4, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    result = a4.run_a4(adapter, source, duration_s=1.0, on_submit=instrument)
    expected_count = 4 if crossing == "refill" else 0
    assert result["submitted_count"] == result["completed_count"] == expected_count
    assert entered == [100.0] * expected_count
    assert result["submit_admission_times_s"] == [100.0] * expected_count
    assert result["admission_deadline_s"] == 101.0
    assert result["drained"] is True and not runtime.pending
    assert adapter.report()["unreleased"] == adapter.report()["pending"] == 0


def test_host_admission_timestamp_does_not_claim_atomic_sdk_entry(monkeypatch):
    class Clock:
        now = 100.0
        def perf_counter(self):
            return self.now
    clock = Clock()
    entered = []
    class DescheduledRuntime(SlotRuntime):
        def run_async(self, *args, **kwargs):
            # Model host descheduling between the admission clock check and
            # SDK entry.  This is deliberately not claimed as device timing.
            clock.now = 102.0
            entered.append(clock.now)
            return super().run_async(*args, **kwargs)
    runtime = DescheduledRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    monkeypatch.setattr(a4, "time", SimpleNamespace(perf_counter=clock.perf_counter))
    result = a4.run_a4(adapter, lambda request_id: input_value(7), duration_s=1.0)
    assert result["submitted_count"] == result["completed_count"] == 1
    assert result["submit_admission_times_s"] == [100.0]
    assert entered == [102.0] and result["admission_deadline_s"] == 101.0
    assert "SDK/device entry time unobserved" in result["observation"]["admission_timestamp_scope"]


def test_cooperative_timeout_drains_accepted_work_without_false_success():
    runtime = SlotRuntime(wait_delay=.005)
    adapter = a4.NativeDeepXAsync(runtime)
    with pytest.raises(a4.NativeDeepXAsyncError, match="timeout") as caught:
        bounded(lambda: a4.run_a4(adapter, lambda request_id: input_value(request_id),
                                   count=100, timeout_s=.003))
    assert caught.value.result["drained"] is False
    assert caught.value.result["unsafe_runtime_state"] is False
    assert caught.value.result["observation"]["pending"] == 0
    assert not runtime.pending


def test_abort_during_failure_drain_keeps_wait_failure_uncertain():
    cancelled = threading.Event()
    class AbortDrain(SlotRuntime):
        def wait(self, job_id):
            if self.wait_count == 1:
                cancelled.set()
                self.fail_wait = 2
            return super().wait(job_id)
    runtime = AbortDrain()
    adapter = a4.NativeDeepXAsync(runtime)
    with pytest.raises(a4.NativeDeepXAsyncError) as caught:
        bounded(lambda: a4.run_a4(adapter, lambda request_id: input_value(request_id),
                                   count=10, cancel_event=cancelled,
                                   map_outputs=lambda request_id, raw: (_ for _ in ()).throw(RuntimeError("mapping_error"))))
    assert caught.value.result["unsafe_runtime_state"] is True
    assert caught.value.result["observation"]["pending"] == 1
    assert not any(peer.name.startswith("native-deepx-") for peer in threading.enumerate())


def test_unreturned_serial_sdk_call_retains_engine_and_inputs_without_inventing_async_jobs():
    class SerialRuntime(SlotRuntime):
        def run(self, inputs):
            self.serial_input = weakref.ref(inputs[0])
            raise RuntimeError("entered_serial_sdk_failed")
    runtime = SerialRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    source = input_value(7)
    engine_ref, source_ref = weakref.ref(runtime), weakref.ref(source)
    try:
        runtime.run([source])
    except RuntimeError as exc:
        adapter.retain_unresolved_serial([source], str(exc))
    report = adapter.report()
    assert report["unsafe_runtime_state"] is report["serial_completion_unresolved"] is True
    assert report["pending"] == report["unreleased"] == report["submitted"] == 0
    assert "serial_sdk_completion_unproven" in ";".join(adapter.drain())
    with pytest.raises(RuntimeError, match="before_full_drain"):
        adapter.reset()
    with pytest.raises(RuntimeError, match="completion_unresolved"):
        adapter.submit(0, input_value(1))
    del source, runtime, adapter
    gc.collect()
    assert engine_ref() is not None and source_ref() is not None
    assert engine_ref().serial_input() is source_ref()


@pytest.mark.parametrize("failure", ["engine", "trt", "mapping"])
def test_actual_split_serial_oracle_distinguishes_sdk_return_from_later_task_failure(deep, tmp_path, monkeypatch, failure):
    class SerialRuntime(SlotRuntime):
        def run(self, inputs):
            if failure == "engine":
                raise RuntimeError("entered_serial_engine_failed")
            return [expected(int(inputs[0].ravel()[0]))]
    class SerialTRT(FakeTRT):
        def run(self, feeds):
            if failure == "trt":
                raise RuntimeError("entered_serial_trt_failed")
            self.prepare_inputs(feeds)
            return self.run_prepared()
    if failure == "mapping":
        def bad_mapping(*args):
            raise RuntimeError("post_sdk_mapping_failed")
        monkeypatch.setattr(deep, "_map_a4_deepx_output_to_trt", bad_mapping)
    runtime, trt = SerialRuntime(), SerialTRT()
    progress = {}
    kwargs = dict(image_paths=[str(tmp_path / "image.jpg")], benchmark_set=tmp_path,
                  model_id="resnet50", task="classification", preprocess_mode="resize",
                  letterbox_pad_value=0, completion_runtime_mode="auto", progress=progress)
    owner_count = len(a4._UNRESOLVED_OWNERS)
    with pytest.raises(RuntimeError) as caught:
        deep.deepx_split_numerical_preflight(runtime, trt, [input_value(7)], **kwargs)
    assert progress["serial_sdk_calls_started"] == 1
    assert progress.get("serial_sdk_calls_completed", 0) == (0 if failure == "engine" else 1)
    assert progress["serial_reference_requests"] == 0
    assert runtime.submit_count == runtime.wait_count == 0
    if failure == "mapping":
        assert not isinstance(caught.value, a4.NativeDeepXAsyncError)
        assert len(a4._UNRESOLVED_OWNERS) == owner_count
    else:
        error = caught.value
        assert isinstance(error, a4.NativeDeepXAsyncError)
        assert error.unsafe_runtime_state is True and error.result["drained"] is False
        assert error.result["observation"]["pending"] == error.result["observation"]["unreleased"] == 0
        assert error.result["observation"]["serial_completion_unresolved"] is True
        assert error.deepx_workload_phase == ("numerical_reference_trt" if failure == "trt" else "numerical_reference")
        assert error.adapter in a4._UNRESOLVED_OWNERS
        if failure == "trt":
            assert progress["serial_trt_calls_started"] == 1
            assert progress.get("serial_trt_calls_completed", 0) == 0


def test_last_outputs_owned_before_next_inference_and_final_copy_only_once(deep, monkeypatch):
    runtime = SlotRuntime()
    adapter = a4.NativeDeepXAsync(runtime)
    trt = FakeTRT()
    callbacks = split_callbacks(deep, trt, ClassificationCompletion())
    snapshots = []
    real = a4._snapshot
    def snapshot(value):
        if isinstance(value, dict):
            assert not runtime.pending
            assert not any(peer.name.startswith("native-deepx-") for peer in threading.enumerate())
            snapshots.append(time.perf_counter())
        return real(value)
    monkeypatch.setattr(a4, "_snapshot", snapshot)
    result = bounded(lambda: a4.run_a4(adapter, lambda request_id: input_value(request_id + 1), count=8, **callbacks))
    assert len(snapshots) == 1 and snapshots[0] >= result["timer_end_s"]
    last = result["last_outputs"]["logits"].copy()
    trt.logits.fill(-999)
    a4.run_a4(adapter, lambda request_id: input_value(99), count=1)
    np.testing.assert_array_equal(result["last_outputs"]["logits"], last)
    np.testing.assert_array_equal(last, expected(8))
