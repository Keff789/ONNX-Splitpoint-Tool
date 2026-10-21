"""CPU-only mixed control against actual Generic scheduler/DeepX slot adapter.

Every input, SDK event and artifact descriptor here is synthetic test data.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1, HailoAsyncStage1
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from onnx_splitpoint_tool.runners.task_completion import measure_split_completion


CANDIDATE = Path(__file__).resolve().parents[1] / "onnx_splitpoint_tool/runners/generic_mixed_control.py"
SPEC = importlib.util.spec_from_file_location("onnx_splitpoint_tool.runners._mixed_control_candidate", CANDIDATE)
mixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mixed)


def logits(value):
    result = np.arange(6, dtype=np.float32)
    return np.roll(result, int(value) % 6)[None]


def input_index(index):
    # Every FIFO slot changes discriminator payload in the second 16 tasks,
    # for both prescribed even slot-pool sizes, 4 and 8.
    return (index % 2) ^ int(index >= 16)


class Engine:
    """SDK fixture completes out of order into the real adapter's owned slabs."""

    def __init__(self, fault=None):
        self.fault = fault
        self.submissions, self.waits, self.completions, self.threads = [], [], [], []
        self.jobs = {}
        self.used_slabs = set()

    def get_input_tensors_info(self):
        return [{"name": "images", "shape": [1, 2, 2, 3], "dtype": "uint8"}]

    def get_output_tensors_info(self):
        return [{"name": "cut", "shape": [1, 6], "dtype": "float32"}]

    def run_async(self, arrays, *, user_arg, output_buffer):
        value = int(arrays[0].flat[0])
        output = np.ndarray((1, 6), dtype=np.float32, buffer=output_buffer)
        done = threading.Event()
        self.submissions.append(user_arg)
        self.jobs[user_arg] = output, done
        first_generation = output_buffer.ctypes.data not in self.used_slabs
        self.used_slabs.add(output_buffer.ctypes.data)

        def finish():
            time.sleep(.008 if user_arg % 2 == 0 else .0001)
            if self.fault != "stale_slot_reuse" or first_generation:
                output[:] = logits(0 if self.fault == "stale" and user_arg == 1 else value)
            self.completions.append(user_arg)
            done.set()

        thread = threading.Thread(target=finish, name="mixed-test-sdk")
        self.threads.append(thread)
        thread.start()
        return 0 if self.fault == "duplicate" else user_arg

    def wait(self, job_id):
        self.waits.append(job_id)
        output, done = self.jobs[job_id]
        assert done.wait(2)
        if self.fault == "lost" and job_id == 1:
            raise RuntimeError("SDK lost output fixture")
        return [output]

    def finish(self):
        for thread in self.threads:
            thread.join(timeout=2)
            assert not thread.is_alive()


class Session:
    def __init__(self, fault=None):
        self._engine = Engine(fault)

    def _prepare_async_inputs(self, feed):
        return [feed["images"]]

    def _materialize_outputs(self, arrays):
        return {"cut": arrays[0]}


class HailoSession:
    """Synthetic device lifecycle behind the actual Hailo slot adapter."""
    timeout_ms = 2000

    def __init__(self, fault=None):
        self.fault, self.jobs, self.used_slots = fault, [], set()

    def _create_reusable_binding_slot(self):
        done = threading.Event()
        done.set()
        return {"binding": {"cut": np.empty((1, 6), dtype=np.float32)}, "done": done}

    def _prepare_infer_inputs(self, feed):
        return feed

    def _submit_reusable_slot(self, slot, feed, *, copy_inputs):
        assert copy_inputs and slot["done"].is_set()
        value = int(np.array(feed["images"], copy=True).flat[0])
        first_generation = id(slot) not in self.used_slots
        self.used_slots.add(id(slot))
        slot["done"].clear()

        def finish():
            time.sleep(.001)
            if self.fault != "stale_slot_reuse" or first_generation:
                slot["binding"]["cut"][:] = logits(value)
            slot["done"].set()

        thread = threading.Thread(target=finish, name="mixed-test-hailo-sdk")
        self.jobs.append(thread)
        thread.start()

    def _wait_reusable_slot(self, slot):
        assert slot["done"].wait(self.timeout_ms / 1000)

    def _materialize_binding_outputs(self, binding, *, copy_outputs):
        assert copy_outputs
        return {name: value.copy() for name, value in binding.items()}

    def finish(self):
        for thread in self.jobs:
            thread.join(timeout=2)
            assert not thread.is_alive()


@pytest.fixture
def recipe():
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones_like(first)
    specs = [{"path": "/verified/fixture_" + str(index) + ".bin",
              "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
              "shape": [2, 2, 3], "dtype": "uint8", "input_name": "images"}
             for index, value in enumerate((first, second))]
    tasks = [{"task_id": "task_" + str(index), "prepared_input": deepcopy(specs[input_index(index)]),
              "feed": {"images": (first, second)[input_index(index)]}} for index in range(32)]
    config = {"stage1_execution": {"policy": "deepx_async_slots_v1", "inflight": 4},
              "queue_depth": 1, "timeout_s": 3,
              "input_binding": {"prepared_input": specs[0]},
              "identity": {"model_id": "synthetic_cpu_fixture"},
              "runtime_source_files": [{"path": "/verified/fixture_runner.py", "sha256": "a" * 64}]}
    return tasks, config


def callbacks():
    # Reuse final TRT-like output storage on every call: the helper must copy
    # the actual raw result before a following invocation overwrites it.
    final_storage = np.empty((1, 6), dtype=np.float32)
    calls = {"serial_p1": 0, "transfer": 0, "stage2": 0}

    def first(feed):
        calls["serial_p1"] += 1
        return {"cut": logits(feed["images"].flat[0])}

    def transfer(boundary, feed):
        calls["transfer"] += 1
        return {"cut": boundary["cut"] + np.float32(feed["images"].flat[0])}

    def second(inputs, feed):
        calls["stage2"] += 1
        final_storage[:] = inputs["cut"] + np.float32(feed["images"].flat[0])
        return {"logits": final_storage}

    return first, transfer, second, final_storage, calls


def run(recipe, *, fault=None, factory=None, comparator=None, synchronous=False, clone_payload=None,
        backend="deepx", inflight=None):
    tasks, config = recipe
    config = deepcopy(config)
    if backend == "hailo":
        config["stage1_execution"] = {"policy": "hailo_async_slots_v1", "inflight": 8}
    if inflight is not None:
        config["stage1_execution"]["inflight"] = inflight
    first, transfer, second, storage, calls = callbacks()
    session = HailoSession(fault) if backend == "hailo" else Session(fault)
    adapters = []

    def async_factory(prepare):
        cls = HailoAsyncStage1 if backend == "hailo" else DeepXAsyncStage1
        adapter = cls(session, prepare, inflight=config["stage1_execution"]["inflight"])
        adapters.append(adapter)
        return adapter

    if synchronous:
        config = deepcopy(config)
        config.pop("stage1_execution")
    try:
        result = mixed.run_mixed_control(tasks, config=config,
            original_sources={"fixture": "synthetic CPU-only inputs", "runtime_source_files": config["runtime_source_files"]},
            serial_stage1=first, transfer=transfer, stage2=second,
            processor_factory=factory or ClassificationCompletion,
            stage1_async_factory=None if synchronous else async_factory, compare=comparator,
            clone_payload=clone_payload)
        return result, session, adapters, storage, calls
    finally:
        session.finish() if backend == "hailo" else session._engine.finish()


def test_two_real_payloads_thirty_two_ids_out_of_order_sdk_and_normal_completion(recipe):
    result, session, adapters, storage, calls = run(recipe)
    assert result["status"] == "passed"
    assert result["serial_reference_count"] == result["mixed_completed_count"] == 32
    assert result["distinct_prepared_tensor_count"] == 2
    assert result["stage1_peak_uncollected_jobs"] == 4
    assert result["drain_completed"] and not adapters[0].pending
    assert session._engine.submissions == session._engine.waits == list(range(32))
    assert session._engine.completions != sorted(session._engine.completions)
    assert calls == {"serial_p1": 32, "transfer": 64, "stage2": 64}
    assert [item["task_id"] for item in result["observations"]] == ["task_" + str(i) for i in range(32)]
    for index, item in enumerate(result["observations"]):
        expected = logits(input_index(index)) + np.float32(2 * input_index(index))
        np.testing.assert_array_equal(item["serial"]["outputs"]["logits"], expected)
        np.testing.assert_array_equal(item["mixed"]["outputs"]["logits"], expected)
        np.testing.assert_array_equal(item["serial"]["completed_result"]["top1"], item["mixed"]["completed_result"]["top1"])
        assert item["input_sha256"] == recipe[0][index]["prepared_input"]["sha256"]
    # Neither TRT output reuse, adapter slab reuse, nor caller mutation changes
    # snapshots which have already completed and been returned.
    storage.fill(-9)
    for slot in adapters[0].slots:
        slot["buffer"].fill(255)
    for task in recipe[0]:
        task["feed"]["images"].fill(11)
    np.testing.assert_array_equal(result["observations"][0]["mixed"]["outputs"]["logits"], logits(0))
    assert "completed_task_fps" not in result
    assert "measurement_start_ns" not in result
    assert result["scientific_statistics_published"] is False


def test_synchronous_control_uses_same_two_workers_and_compiled_callbacks(recipe):
    result, session, adapters, _, calls = run(recipe, synchronous=True)
    assert result["stage1_execution"] == {"policy": "synchronous", "inflight": 1}
    assert result["stage1_peak_uncollected_jobs"] == 1 and result["drain_completed"]
    assert calls["serial_p1"] == 64 and not session._engine.submissions and not adapters


@pytest.mark.parametrize("fault", ["stale", "duplicate", "lost"])
def test_stale_duplicate_or_lost_outputs_never_pass_or_publish_a_rate(recipe, fault):
    with pytest.raises(mixed.MixedControlError) as caught:
        run(recipe, fault=fault)
    report = caught.value.partial_report
    assert report["status"] == "failed" and report["serial_reference_count"] == 32
    assert "completed_task_fps" not in report
    if fault == "stale":
        assert report["mixed_completed_count"] == 32 and report["drain_completed"]
        assert report["observations"][-1]["task_id"] == "task_1"
    else:
        assert not report["drain_completed"]


def test_consumer_exception_drains_all_known_async_jobs(recipe):
    count = 0

    class Broken(ClassificationCompletion):
        def process(self, outputs):
            raise RuntimeError("consumer failure fixture")

    def factory():
        nonlocal count
        count += 1
        return ClassificationCompletion() if count == 1 else Broken()

    with pytest.raises(mixed.MixedControlError) as caught:
        run(recipe, factory=factory)
    assert caught.value.partial_report["drain_completed"]
    assert caught.value.partial_report["mixed_completed_count"] == 0


@pytest.mark.parametrize("fault", ["too_few", "too_many", "duplicate_id", "same_payload", "shape", "dtype", "conflicting_sha"])
def test_bad_input_recipe_fails_before_runtime_or_factory_calls(recipe, fault):
    tasks, config = recipe
    if fault == "too_few":
        tasks.pop()
    elif fault == "too_many":
        tasks.append({**tasks[0], "task_id": "unexpected_33rd_task"})
    elif fault == "duplicate_id":
        tasks[-1]["task_id"] = tasks[0]["task_id"]
    elif fault == "same_payload":
        for task in tasks:
            task["feed"]["images"] = tasks[0]["feed"]["images"]
    elif fault == "shape":
        tasks[1]["feed"]["images"] = np.zeros((3, 4), dtype=np.uint8)
    elif fault == "dtype":
        tasks[1]["feed"]["images"] = tasks[1]["feed"]["images"].astype(np.float32)
    else:
        tasks[1]["prepared_input"]["sha256"] = tasks[0]["prepared_input"]["sha256"]
    def forbidden(*args):
        pytest.fail("invalid control entered runtime")
    with pytest.raises(ValueError):
        mixed.run_mixed_control(tasks, config=config, original_sources={"fixture": True},
            serial_stage1=forbidden, transfer=forbidden, stage2=forbidden,
            processor_factory=forbidden, stage1_async_factory=forbidden)


def test_unchanged_config_and_source_descriptors_are_bound_and_not_mutated(recipe):
    before = deepcopy(recipe[1])
    result, *_ = run(recipe)
    assert recipe[1] == before
    assert result["runtime_binding"]["runtime_source_files"] == before["runtime_source_files"]
    assert result["original_sources"]["runtime_source_files"] == before["runtime_source_files"]


def test_existing_comparator_is_used_without_new_tolerance(recipe):
    calls = []

    def existing(reference, actual):
        passed = np.array_equal(reference["outputs"]["logits"], actual["outputs"]["logits"])
        passed = passed and all(np.array_equal(reference["completed_result"][key], actual["completed_result"][key])
                                for key in ("top1", "top5"))
        calls.append(passed)
        return {"existing_fixture_comparison": True}, bool(passed)

    result, *_ = run(recipe, comparator=existing)
    assert result["comparison"] == "existing_product_comparator"
    assert calls == [False] * 4 + [True] * 32


def test_non_discriminating_reference_or_comparator_fails_before_async_work(recipe):
    def overly_permissive(reference, actual):
        return {}, True
    with pytest.raises(mixed.MixedControlError, match="cannot distinguish") as caught:
        run(recipe, comparator=overly_permissive)
    assert caught.value.partial_report["serial_reference_count"] == 32
    assert caught.value.partial_report["mixed_completed_count"] == 0


def test_mismatched_factory_policy_cannot_silently_run_synchronous_control(recipe):
    tasks, config = recipe
    first, transfer, second, _, calls = callbacks()

    def wrong_factory(prepare):
        return type("Wrong", (), {"execution_policy": {"policy": "synchronous", "inflight": 1}, "pending": {}})()

    with pytest.raises(mixed.MixedControlError, match="prepared policy"):
        mixed.run_mixed_control(tasks, config=config, original_sources={"fixture": True},
            serial_stage1=first, transfer=transfer, stage2=second,
            processor_factory=ClassificationCompletion, stage1_async_factory=wrong_factory)
    assert calls == {"serial_p1": 0, "transfer": 0, "stage2": 0}


def test_processor_lost_completion_is_a_failure_even_with_correct_logits(recipe):
    count = 0

    class Lost(ClassificationCompletion):
        def process(self, outputs):
            super().process(outputs)
            self.completed_count -= 1

    def factory():
        nonlocal count
        count += 1
        return ClassificationCompletion() if count == 1 else Lost()

    with pytest.raises(mixed.MixedControlError, match="counter mismatch") as caught:
        run(recipe, factory=factory)
    assert caught.value.partial_report["drain_completed"]


@pytest.mark.parametrize("synchronous", [False, True])
def test_normal_handoff_copy_callback_receives_only_tensor_map_and_preserves_ids(recipe, synchronous):
    copied = []

    def normal_copy(outputs):
        assert set(outputs) == {"cut"}
        copied.append(outputs["cut"].copy())
        return {name: value.copy() for name, value in outputs.items()}

    result, *_ = run(recipe, synchronous=synchronous, clone_payload=normal_copy)
    assert result["status"] == "passed" and result["handoff_payload_copy"]
    assert len(copied) == 32
    for index, value in enumerate(copied):
        np.testing.assert_array_equal(value, logits(input_index(index)))
    assert [row["task_id"] for row in result["observations"]] == ["task_" + str(i) for i in range(32)]


def test_owned_writable_inputs_are_checked_for_mutation(recipe):
    tasks, config = recipe
    config = deepcopy(config)
    config.pop("stage1_execution")

    def mutating_runtime(feed):
        assert feed["images"].flags.writeable and feed["images"].flags.c_contiguous
        feed["images"].flat[0] = 6
        return {"logits": logits(0)}

    with pytest.raises(mixed.MixedControlError, match="mutated the prepared input"):
        mixed.run_mixed_control(tasks, config=config, original_sources={"fixture": True},
            serial_stage1=mutating_runtime, transfer=lambda outputs, feed: outputs,
            stage2=lambda inputs, feed: inputs, processor_factory=ClassificationCompletion)
    # The helper owns its writable snapshots, and never modifies caller data.
    assert tasks[0]["feed"]["images"].flat[0] == 0
    assert tasks[1]["feed"]["images"].flat[0] == 1


def test_existing_detection_completion_snapshots_are_retained_per_input(recipe):
    from tests.test_v272_detection_completion_runtime import _decoded_outputs, _runtime
    tasks, config = recipe
    session = Session()
    adapters = []

    def factory(prepare):
        adapter = DeepXAsyncStage1(session, prepare, inflight=4)
        adapters.append(adapter)
        return adapter

    def detection_tail(inputs, feed):
        outputs = _decoded_outputs()
        outputs["detections"][..., [0, 2]] += inputs["cut"][0, 0]
        return outputs

    try:
        result = mixed.run_mixed_control(tasks, config=config,
            original_sources={"fixture": "synthetic detector control"},
            serial_stage1=lambda feed: {"cut": logits(feed["images"].flat[0])},
            transfer=lambda outputs, feed: outputs, stage2=detection_tail,
            processor_factory=lambda: _runtime(_decoded_outputs()), stage1_async_factory=factory)
    finally:
        session._engine.finish()
    assert result["status"] == "passed" and result["drain_completed"]
    assert result["observations"][0]["mixed"]["completed_result"]["host_nms_applied"] is False
    first = result["observations"][0]["mixed"]["completed_result"]["detections"]
    second = result["observations"][1]["mixed"]["completed_result"]["detections"]
    assert first != second
    assert len(result["observations"]) == 32 and not adapters[0].pending


def test_deadline_does_not_claim_a_blocked_sdk_wait_was_drained(recipe):
    tasks, config = recipe
    config = deepcopy(config)
    config["timeout_s"] = .02
    first, transfer, second, _, _ = callbacks()
    session = Session()
    release = threading.Event()
    entered = threading.Event()
    original_wait = session._engine.wait
    adapters = []

    def blocked_wait(job_id):
        entered.set()
        assert release.wait(5)
        return original_wait(job_id)

    session._engine.wait = blocked_wait

    def factory(prepare):
        adapter = DeepXAsyncStage1(session, prepare, inflight=4)
        adapters.append(adapter)
        return adapter

    try:
        with pytest.raises(mixed.MixedControlError) as caught:
            mixed.run_mixed_control(tasks, config=config, original_sources={"fixture": True},
                serial_stage1=first, transfer=transfer, stage2=second,
                processor_factory=ClassificationCompletion, stage1_async_factory=factory)
        assert entered.is_set() and adapters[0].pending and adapters[0].retained
        assert caught.value.partial_report["drain_completed"] is False
        assert "completed_task_fps" not in caught.value.partial_report
    finally:
        release.set()
        session._engine.finish()
        # Test fixture only: release the deliberately blocked wait, then prove
        # its own producer finally exits. This is not a hardware cleanup claim.
        deadline = time.monotonic() + 2
        while adapters and adapters[0].pending and time.monotonic() < deadline:
            time.sleep(.005)
        assert not adapters[0].pending


@pytest.mark.parametrize("backend,inflight", [("deepx", 4), ("hailo", 8)])
def test_phase_inversion_exercises_both_inputs_on_every_actual_adapter_slot(recipe, backend, inflight):
    result, _, adapters, _, _ = run(recipe, backend=backend)
    coverage = result["stage1_slot_input_coverage"]
    assert len(coverage) == inflight
    assert result["stage1_peak_uncollected_jobs"] == inflight
    assert all(slot["distinct_prepared_tensor_count"] == 2
               and slot["serial_outputs_discriminating"] for slot in coverage)
    for slot in coverage:
        assert slot["task_ids"] == ["task_" + str(index)
            for index in range(slot["slot_index"], 32, inflight)]
    assert not adapters[0].pending


@pytest.mark.parametrize("inflight", [4, 8, 17, 32])
def test_periodic_or_insufficient_slot_reuse_fails_before_any_runtime(recipe, inflight):
    tasks, config = recipe
    config["stage1_execution"]["inflight"] = inflight
    if inflight in (4, 8):
        choices = [deepcopy(tasks[0]), deepcopy(tasks[1])]
        tasks[:] = [{**choices[index % 2], "task_id": "task_" + str(index)} for index in range(32)]

    def forbidden(*args):
        pytest.fail("unsupported slot reuse entered runtime")

    with pytest.raises(ValueError, match="every reused stage1 slot"):
        mixed.run_mixed_control(tasks, config=config, original_sources={"fixture": True},
            serial_stage1=forbidden, transfer=forbidden, stage2=forbidden,
            processor_factory=forbidden, stage1_async_factory=forbidden)


@pytest.mark.parametrize("backend,inflight", [("deepx", 4), ("hailo", 8)])
def test_previous_generation_per_slot_fault_is_detected_after_phase_inversion(recipe, backend, inflight):
    # Reproduce the blind spot directly through the real product scheduler and
    # adapter: old period-two inputs remain identical in every reused slot, so
    # a device which never updates reused output buffers appears correct.
    session = HailoSession("stale_slot_reuse") if backend == "hailo" else Session("stale_slot_reuse")
    count = 0
    seen = []

    def naive_prepare():
        nonlocal count
        feed = {"images": np.full((2, 2, 3), count % 2, dtype=np.uint8)}
        count += 1
        return feed

    class Recorder(ClassificationCompletion):
        def process(self, outputs):
            seen.append(outputs["logits"].copy())
            return super().process(outputs)

    cls = HailoAsyncStage1 if backend == "hailo" else DeepXAsyncStage1
    adapter = cls(session, naive_prepare, inflight=inflight)
    try:
        unchecked = measure_split_completion(lambda: pytest.fail("sync fallback"), lambda output: output,
            lambda output: {"logits": output["cut"]}, Recorder(),
            warmup=0, tasks=32, queue_depth=1, timeout_s=3, stage1_async=adapter)
    finally:
        session.finish() if backend == "hailo" else session._engine.finish()
    assert unchecked["completed_task_count"] == 32
    for index, value in enumerate(seen):
        np.testing.assert_array_equal(value, logits(index % 2))
    assert not adapter.pending

    # The corrected helper's fixed phase inversion makes the same defect
    # visible on the first changed slot generation (task 16), after full drain.
    with pytest.raises(mixed.MixedControlError, match="task_16") as caught:
        run(recipe, backend=backend, fault="stale_slot_reuse")
    report = caught.value.partial_report
    assert report["serial_reference_count"] == report["mixed_completed_count"] == 32
    assert report["drain_completed"]
    assert all(slot["distinct_prepared_tensor_count"] == 2 for slot in report["stage1_slot_input_coverage"])
    assert report["observations"][-1]["task_id"] == "task_16"
    assert report["observations"][-1]["passed"] is False


def test_each_slots_references_must_discriminate_even_when_other_slots_do(recipe):
    tasks, config = recipe
    third = np.full((2, 2, 3), 2, dtype=np.uint8)
    descriptor = {**deepcopy(tasks[0]["prepared_input"]), "path": "/verified/third.bin",
                  "sha256": hashlib.sha256(third.tobytes()).hexdigest()}
    for index in range(1, 32, 4):
        choice = {"prepared_input": descriptor, "feed": {"images": third}} if index >= 16 else tasks[0]
        tasks[index] = {"task_id": "task_" + str(index),
                        "prepared_input": deepcopy(choice["prepared_input"]), "feed": choice["feed"]}
    session = Session()

    def serial(feed):
        value = int(feed["images"].flat[0])
        return {"logits": logits(0 if value == 2 else value)}

    with pytest.raises(mixed.MixedControlError, match="slot 1"):
        mixed.run_mixed_control(tasks, config=config, original_sources={"fixture": True},
            serial_stage1=serial, transfer=lambda output, feed: output,
            stage2=lambda inputs, feed: inputs, processor_factory=ClassificationCompletion,
            stage1_async_factory=lambda prepare: DeepXAsyncStage1(session, prepare, inflight=4))
    assert not session._engine.submissions
