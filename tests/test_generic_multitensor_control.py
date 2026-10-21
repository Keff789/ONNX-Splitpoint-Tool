"""CPU fixtures prove actual multi-tensor observation, never hardware PASS."""
from copy import deepcopy
import threading

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_mixed_control as mixed
from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1, HailoAsyncStage1
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from tests.test_generic_mixed_control import input_index, logits, recipe


def boundary(value):
    return {"cut_a": logits(value),
            "cut_b": np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4) + np.uint8(3 * value)}


def contract(*, passthrough=False):
    value = {"schema_version": 1, "minimum_boundary_tensors": 2,
        "part1_outputs": {"cut_a": {"shape": [1, 6], "dtype": "float32"},
                          "cut_b": {"shape": [1, 2, 3, 4], "dtype": "uint8"}},
        "part2_inputs": {"trt_a": {"shape": [1, 6], "dtype": "float32"},
                         "trt_b": {"shape": [1, 4, 2, 3], "dtype": "float16"}},
        "part2_to_part1": {"trt_a": "cut_a", "trt_b": "cut_b"}, "passthrough_inputs": {},
        "transfer_contract": {"part1_layouts": {"cut_a": "NC", "cut_b": "NHWC"},
            "part2_layouts": {"trt_a": "NC", "trt_b": "NCHW"},
            "quantization": {"cut_b": {"scale": .25, "zero_point": 5}}}}
    if passthrough:
        value["part2_inputs"]["image_passthrough"] = {"shape": [2, 2, 3], "dtype": "uint8"}
        value["passthrough_inputs"]["image_passthrough"] = "images"
    return value


def runtime_metadata(value):
    return {"part1_io": {"backend": "synthetic-only", "canonical_outputs": list(value["part1_outputs"]),
                "runtime_output_quantization": value["transfer_contract"]["quantization"]},
            "part2_io": {"backend": "synthetic-tensorrt", "inputs": deepcopy(value["part2_inputs"])},
            **{key: deepcopy(value[key]) for key in ("part2_to_part1", "passthrough_inputs", "transfer_contract")}}


class Engine:
    def __init__(self, fault=None):
        self.fault, self.jobs, self.calls = fault, {}, []

    def get_input_tensors_info(self):
        return [{"name": "images", "shape": [1, 2, 2, 3], "dtype": "uint8"}]

    def get_output_tensors_info(self):
        return [{"name": name, **spec} for name, spec in contract()["part1_outputs"].items()]

    def run_async(self, arrays, *, user_arg, output_buffer):
        a = np.ndarray((1, 6), dtype=np.float32, buffer=output_buffer)
        b = np.ndarray((1, 2, 3, 4), dtype=np.uint8, buffer=output_buffer, offset=24)
        expected = boundary(int(arrays[0].flat[0]))
        a[:], b[:] = expected["cut_a"], expected["cut_b"]
        if self.fault == "wrong_second":
            b += 1  # Final logits deliberately do not consume this tensor.
        self.jobs[user_arg] = [a, b]
        self.calls.append(user_arg)
        return user_arg

    def wait(self, job_id):
        self.last_wait = self.jobs.pop(job_id)
        return self.last_wait


class DeepXSession:
    def __init__(self, fault=None):
        self._engine = Engine(fault)
        self.fault = fault

    def _prepare_async_inputs(self, feed):
        return [feed["images"]]

    def _materialize_outputs(self, values):
        if self.fault == "borrowed":
            values = self._engine.last_wait  # Defect: discard production-owned copies.
        return dict(zip(("cut_a", "cut_b"), values))


class HailoSession:
    timeout_ms = 2000

    def __init__(self, fault=None):
        self.fault, self.calls = fault, []
        self.shared = boundary(0)

    def _create_reusable_binding_slot(self):
        done = threading.Event()
        done.set()
        arrays = self.shared if self.fault == "shared_slots" else boundary(0)
        return {"binding": arrays, "output_buffers": arrays,
                "input_buffers": {"images": np.empty((2, 2, 3), dtype=np.uint8)}, "done": done}

    def _prepare_infer_inputs(self, feed):
        return feed

    def _submit_reusable_slot(self, slot, feed, *, copy_inputs):
        assert copy_inputs and slot["done"].is_set()
        np.copyto(slot["input_buffers"]["images"], feed["images"])
        values = boundary(int(slot["input_buffers"]["images"].flat[0]))
        for name, value in values.items():
            slot["binding"][name][:] = value
        if self.fault == "wrong_second":
            slot["binding"]["cut_b"] += 1
        self.calls.append(id(slot))

    def _wait_reusable_slot(self, slot):
        assert slot["done"].is_set()

    def _materialize_binding_outputs(self, binding, *, copy_outputs):
        assert copy_outputs
        return binding if self.fault == "borrowed" else {name: value.copy() for name, value in binding.items()}


def callbacks(value, fault=None):
    metadata = runtime_metadata(value)

    def serial(feed):
        return boundary(int(feed["images"].flat[0]))

    def transfer(outputs, feed):
        result = {"trt_a": outputs["cut_a"], "trt_b": np.ascontiguousarray(
            ((outputs["cut_b"].astype(np.float32) - 5) * .25).transpose(0, 3, 1, 2), dtype=np.float16)}
        if "image_passthrough" in value["part2_inputs"]:
            result["image_passthrough"] = feed["images"]
        if fault == "transfer_mutation":
            outputs["cut_b"] += 1
        elif fault == "missing_second":
            result.pop("trt_b")
        elif fault == "wrong_dtype":
            result["trt_b"] = result["trt_b"].astype(np.float32)
        return result

    def second(inputs, feed):
        if fault == "p2_mutation":
            inputs["trt_b"] += 1
        return {"logits": inputs["trt_a"]}

    def runtime():
        if fault == "wrong_mapping":
            return {**metadata, "part2_to_part1": {"trt_a": "cut_b", "trt_b": "cut_a"}}
        if fault == "layout_metadata":
            return {**metadata, "transfer_contract": {"layout": "unbound"}}
        return metadata

    return serial, transfer, second, runtime


def execute(recipe, *, backend="deepx", fault=None, clone=False, passthrough=False):
    tasks, config = deepcopy(recipe)
    config["stage1_execution"] = {"policy": f"{backend}_async_slots_v1", "inflight": 8 if backend == "hailo" else 4}
    value = contract(passthrough=passthrough)
    first, transfer, second, runtime = callbacks(value, fault)
    session = HailoSession(fault) if backend == "hailo" else DeepXSession(fault)
    adapters = []

    def factory(prepare):
        cls = HailoAsyncStage1 if backend == "hailo" else DeepXAsyncStage1
        adapter = cls(session, prepare, inflight=config["stage1_execution"]["inflight"])
        adapters.append(adapter)
        return adapter

    def cloned(outputs):
        if fault == "queued_mutation":
            outputs["cut_b"] += 1
        return {name: array.copy() for name, array in outputs.items()}

    result = mixed.run_mixed_control(tasks, config=config, original_sources={"synthetic_only": True},
        serial_stage1=first, transfer=transfer, stage2=second, processor_factory=ClassificationCompletion,
        stage1_async_factory=factory, clone_payload=cloned if clone else None,
        boundary_observation=value, boundary_runtime=runtime)
    return result, adapters[0], session


@pytest.mark.parametrize("backend,inflight", [("hailo", 8), ("deepx", 4)])
@pytest.mark.parametrize("clone", [False, True])
def test_actual_two_boundary_maps_and_transformed_p2_feeds_all_tasks(recipe, backend, inflight, clone):
    report, adapter, _ = execute(recipe, backend=backend, clone=clone, passthrough=True)
    proof = report["boundary_observation"]
    assert proof["observed_task_count"] == report["mixed_completed_count"] == 32
    assert proof["observed_boundary_tensor_count"] == 2
    assert proof["slot_buffer_ownership"]["slot_count"] == inflight
    assert proof["slot_buffer_ownership"]["slots_disjoint"]
    assert report["stage1_peak_uncollected_jobs"] == inflight and report["drain_completed"]
    assert not adapter.pending and all(item["distinct_prepared_tensor_count"] == 2
                                      for item in report["stage1_slot_input_coverage"])
    for index, item in enumerate(report["observations"]):
        observed = item["boundary_observation"]
        assert observed["passed"]
        for phase in ("serial", "mixed"):
            snapshot = observed[phase]
            assert set(snapshot["part1_at_collection"]) == {"cut_a", "cut_b"}
            assert set(snapshot["part2_before_inference"]) == {"trt_a", "trt_b", "image_passthrough"}
            assert snapshot["part1_at_collection"]["cut_b"]["dtype"] == "uint8"
            assert snapshot["part2_before_inference"]["trt_b"]["dtype"] == "float16"
            assert snapshot["part2_before_inference"]["trt_b"]["shape"] == [1, 4, 2, 3]
            assert snapshot["runtime"]["part2_to_part1"] == {"trt_a": "cut_a", "trt_b": "cut_b"}
            assert snapshot["part2_to_part1_aliases"]["trt_a"] == ["cut_a"]
            assert snapshot["part2_to_part1_aliases"]["trt_b"] == []
            assert snapshot["no_reusable_sdk_buffer_alias"]
        np.testing.assert_array_equal(item["mixed"]["outputs"]["logits"], logits(input_index(index)))
    assert "completed_task_fps" not in report


@pytest.mark.parametrize("backend", ["hailo", "deepx"])
def test_second_tensor_error_detected_even_when_final_outputs_are_identical(recipe, backend):
    with pytest.raises(mixed.MixedControlError, match="serial/async tensor content") as caught:
        execute(recipe, backend=backend, fault="wrong_second")
    report = caught.value.partial_report
    assert report["drain_completed"] and report["mixed_completed_count"] == 32
    assert all(np.array_equal(left["outputs"]["logits"], right["outputs"]["logits"])
               for left, right in zip(report["serial_references"], report["mixed_results"]))


@pytest.mark.parametrize("backend", ["hailo", "deepx"])
def test_returned_sdk_borrowed_output_fails_before_consumption(recipe, backend):
    with pytest.raises(mixed.MixedControlError, match="aliases a reusable SDK slot"):
        execute(recipe, backend=backend, fault="borrowed")


@pytest.mark.parametrize("fault,reason", [("transfer_mutation", "transfer mutated"),
    ("missing_second", "tensor names"), ("wrong_dtype", "shape/dtype"),
    ("p2_mutation", "P2 mutated"), ("wrong_mapping", "live mapping"),
    ("layout_metadata", "live mapping"), ("queued_mutation", "changed while queued")])
def test_incomplete_mapping_and_mutation_rejected(recipe, fault, reason):
    with pytest.raises(mixed.MixedControlError, match=reason):
        execute(recipe, fault=fault, clone=fault == "queued_mutation")


def test_sdk_slots_must_have_private_buffers(recipe):
    with pytest.raises(mixed.MixedControlError, match="slots share buffers"):
        execute(recipe, backend="hailo", fault="shared_slots")


@pytest.mark.parametrize("fault", ["minimum", "one_source", "uncovered_input", "bad_dtype", "bool_shape"])
def test_invalid_declared_boundary_contract_fails_before_runtime(fault):
    value = contract()
    if fault == "minimum":
        value["minimum_boundary_tensors"] = True
    elif fault == "one_source":
        value["part2_to_part1"]["trt_b"] = "cut_a"
    elif fault == "uncovered_input":
        value["part2_to_part1"].pop("trt_b")
    elif fault == "bad_dtype":
        value["part1_outputs"]["cut_a"]["dtype"] = "object"
    else:
        value["part1_outputs"]["cut_a"]["shape"][0] = True
    with pytest.raises(ValueError):
        mixed.normalize_boundary_observation(value)
