"""Bounded, CPU-only proof and provenance checks for optional mixed controls."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_mixed_control_binding as binding
from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1


def write_json(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": binding._sha(path)}


@pytest.fixture
def bound(tmp_path):
    tensors, inputs = [], []
    for index in (0, 1):
        value = np.full((2, 2, 3), index, dtype=np.uint8)
        path = tmp_path / f"prepared_{index}.npy"
        np.save(path, value)
        tensors.append(value)
        inputs.append({"prepared_input": {"path": str(path), "sha256": binding._sha(path),
            "input_name": "images", "shape": [2, 2, 3], "dtype": "uint8", "format": "npy"},
            "provenance": {"source": "synthetic CPU-only fixture"}})
    source = tmp_path / "runner.py"
    source.write_text("# synthetic CPU-only runner source\n")
    config = {"task": "classification", "identity": {"model_id": "fixture", "case_id": "b001"},
        "execution_id": "synthetic-test-only", "original_run": "/synthetic/closed",
        "expected_runtime_artifacts": {"fixture": {"sha256": "a" * 64}},
        "input_binding": {"prepared_input": deepcopy(inputs[0]["prepared_input"])},
        "completion_contract": {"task": "classification", "operation": "top1_top5", "label_offset": 0},
        "stage1_execution": {"policy": "deepx_async_slots_v1", "inflight": 4},
        "queue_depth": 1, "timeout_s": 3}
    payload = {key: deepcopy(config[key]) for key in ("identity", "execution_id", "expected_runtime_artifacts", "stage1_execution")}
    payload.update(task_count=32, inputs=inputs, provenance={"synthetic_fixture": True},
        task_sequence_policy="first16_alternate_second16_invert_v1", task_input_indices=binding._input_sequence())
    config["mixed_input_control"] = write_json(tmp_path / "control.json", payload)
    config["runtime_source_files"] = [config["mixed_input_control"],
        {"path": str(source), "sha256": binding._sha(source)},
        *[{key: item["prepared_input"][key] for key in ("path", "sha256")} for item in inputs]]
    return config, payload, tensors


def rebind(config, payload):
    descriptor = write_json(Path(config["mixed_input_control"]["path"]), payload)
    config["mixed_input_control"] = descriptor
    config["runtime_source_files"][0] = descriptor


class Engine:
    def __init__(self):
        self.jobs = {}

    def get_input_tensors_info(self):
        return [{"name": "images", "shape": [1, 2, 2, 3], "dtype": "uint8"}]

    def get_output_tensors_info(self):
        return [{"name": "cut", "shape": [1, 6], "dtype": "float32"}]

    def run_async(self, arrays, *, user_arg, output_buffer):
        array = np.ndarray((1, 6), dtype=np.float32, buffer=output_buffer)
        array[:] = logits(arrays[0].flat[0])
        self.jobs[user_arg] = array
        return user_arg

    def wait(self, job_id):
        return [self.jobs.pop(job_id)]


class Session:
    def __init__(self):
        self._engine = Engine()

    def _prepare_async_inputs(self, feed):
        return [feed["images"]]

    def _materialize_outputs(self, values):
        return {"cut": values[0]}


def logits(value):
    return np.roll(np.arange(6, dtype=np.float32), int(value))[None]


def execute(config, tmp_path, *, stage2=None):
    session = Session()
    original_prepare = lambda: pytest.fail("the original throughput feed must be restored, not consumed")
    adapter = DeepXAsyncStage1(session, original_prepare, inflight=4)
    result = binding.run_bound_mixed_control(config, tmp_path, 1,
        serial_stage1=lambda feed: {"cut": logits(feed["images"].flat[0])},
        transfer=lambda outputs, feed: outputs,
        stage2=stage2 or (lambda inputs, feed: {"logits": inputs["cut"]}),
        stage1_async=adapter, prepare_async_feed=lambda feed: feed,
        clone_payload=None, runner_source=__file__)
    assert adapter.prepare_inputs is original_prepare
    assert not adapter.pending and not session._engine.jobs
    return result


def test_bound_sidecar_reads_exact_existing_arrays_without_mutating_config(bound):
    config, payload, tensors = bound
    before = deepcopy(config)
    actual, loaded = binding.load_bound_mixed_control(config)
    assert actual == payload and config == before
    for left, right in zip(loaded, tensors):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("fault", ["identity", "artifact", "policy", "count", "sequence", "shape", "baseline", "provenance"])
def test_foreign_or_malformed_sidecar_rejected_before_runtime(bound, fault):
    config, payload, _ = bound
    if fault == "identity":
        payload["identity"]["case_id"] = "other"
    elif fault == "artifact":
        payload["expected_runtime_artifacts"] = {"other": True}
    elif fault == "policy":
        payload["stage1_execution"]["inflight"] = 1
    elif fault == "count":
        payload["task_count"] = True
    elif fault == "sequence":
        payload["task_input_indices"] = [index % 2 for index in range(32)]
    elif fault == "shape":
        payload["inputs"][1]["prepared_input"]["shape"] = [3, 2, 2]
    elif fault == "baseline":
        payload["inputs"].reverse()
    else:
        payload["inputs"][1]["provenance"] = "not a mapping"
    rebind(config, payload)
    with pytest.raises(ValueError):
        binding.load_bound_mixed_control(config)


@pytest.mark.parametrize("fault", ["source_changed", "sidecar_changed", "input_changed", "duplicate_source", "outside_source", "energy"])
def test_changed_or_unbound_sources_rejected(bound, fault):
    config, payload, _ = bound
    if fault == "source_changed":
        Path(config["runtime_source_files"][1]["path"]).write_text("changed")
    elif fault == "sidecar_changed":
        Path(config["mixed_input_control"]["path"]).write_text("{}")
    elif fault == "input_changed":
        Path(payload["inputs"][1]["prepared_input"]["path"]).write_bytes(b"changed")
    elif fault == "duplicate_source":
        config["runtime_source_files"].append(config["runtime_source_files"][0])
    elif fault == "outside_source":
        config["runtime_source_files"].pop()
    else:
        config["energy_capture"] = {"fixture": True}
    with pytest.raises(ValueError):
        binding.load_bound_mixed_control(config)


def test_actual_adapter_pipeline_report_and_normal_reader_proof(bound, tmp_path):
    config, _, _ = bound
    proof = execute(config, tmp_path)
    repetition = {"stage1_execution": config["stage1_execution"], "handoff_payload_copy": False, "mixed_input_control": proof}
    assert binding.validate_bound_mixed_result(repetition, config, result_root=tmp_path) == proof
    report = json.loads((tmp_path / proof["report"]["path"]).read_text())
    assert report["task_input_indices"] == [index % 2 if index < 16 else 1 - index % 2 for index in range(32)]
    assert "completed_task_fps" not in report
    assert len(report["observations"]) == 32
    assert report["observations"][0]["serial"] == report["observations"][0]["mixed"]


def test_failed_control_is_serialized_and_does_not_contaminate_throughput_partial_report(bound, tmp_path):
    config, _, _ = bound
    session = Session()
    original = lambda: None
    adapter = DeepXAsyncStage1(session, original, inflight=4)
    calls = 0
    def broken(inputs, feed):
        nonlocal calls
        calls += 1
        if calls > 32:
            raise RuntimeError("synthetic consumer error")
        return {"logits": inputs["cut"]}
    with pytest.raises(RuntimeError, match="diagnostic report") as caught:
        binding.run_bound_mixed_control(config, tmp_path, 1,
            serial_stage1=lambda feed: {"cut": logits(feed["images"].flat[0])},
            transfer=lambda outputs, feed: outputs, stage2=broken, stage1_async=adapter,
            prepare_async_feed=lambda feed: feed, clone_payload=None, runner_source=__file__)
    assert not hasattr(caught.value, "partial_report")
    assert adapter.prepare_inputs is original and not adapter.pending
    report = json.loads((tmp_path / "mixed_input_control/repetition_1.json").read_text())
    assert report["status"] == "failed" and report["serial_reference_count"] == 32
    assert report["serial_references"][0]["outputs"]["logits"]["tensor_sha256"]


@pytest.mark.parametrize("fault", ["summary", "report_bytes", "escape", "symlink", "task_id", "sequence", "raw", "source_binding", "slot_coverage"])
def test_reader_rejects_tampered_summary_and_bound_report(bound, tmp_path, fault):
    config, _, _ = bound
    proof = execute(config, tmp_path)
    path = tmp_path / proof["report"]["path"]
    if fault == "summary":
        proof["mixed_completed_count"] = 31
    elif fault == "report_bytes":
        path.write_text("{}")
    elif fault == "escape":
        proof["report"]["path"] = "../escaped.json"
    elif fault == "symlink":
        outside = tmp_path.parent / (tmp_path.name + "_outside.json")
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    else:
        report = json.loads(path.read_text())
        if fault == "task_id":
            report["observations"][1]["task_id"] = "task_00"
        elif fault == "sequence":
            report["task_input_indices"] = [index % 2 for index in range(32)]
        elif fault == "raw":
            report["observations"][0]["mixed"]["outputs"]["logits"]["tensor_sha256"] = "b" * 64
        elif fault == "slot_coverage":
            report["stage1_slot_input_coverage"][0]["task_ids"] = ["task_00"]
        else:
            report["runtime_source_binding"]["after"] = {}
        path.write_text(json.dumps(report))
        proof["report"]["sha256"] = binding._sha(path)
    repetition = {"stage1_execution": config["stage1_execution"], "handoff_payload_copy": False, "mixed_input_control": proof}
    with pytest.raises(ValueError):
        binding.validate_bound_mixed_result(repetition, config, result_root=tmp_path)


def test_historical_absence_and_unrequested_proof():
    assert binding.load_bound_mixed_control({}) is None
    assert binding.validate_bound_mixed_result({}, {}) is None
    with pytest.raises(ValueError):
        binding.validate_bound_mixed_result({"mixed_input_control": {}}, {})


def test_deployed_bundle_import_uses_same_level_detection_module(tmp_path):
    from onnx_splitpoint_tool.workflow.generic_completed_task import prepare_runner_bundle
    bundle = tmp_path / "generated"
    prepare_runner_bundle(bundle)
    program = """import importlib,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
m=importlib.import_module('splitpoint_runners.generic_mixed_control_binding')
assert m._detection_module().__name__=='splitpoint_runners.native_detection_postprocess'
from splitpoint_runners.harness.classification import ClassificationCompletion
assert ClassificationCompletion().completed_count==0
print('deployed binding imports passed')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(bundle)],
                            cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "deployed binding imports passed" in result.stdout


def test_geometry_validation_uses_normal_builder_and_rejects_wrong_image_geometry():
    from tests.test_v272_detection_completion_runtime import _decoded_outputs, _decoded_source
    product = binding._detection_module()
    outputs = _decoded_outputs()
    def contract(wh):
        return product.build_detection_completion_execution_contract(model_id="yolo26s", outputs=outputs,
            input_hw=[64, 64], original_wh=wh, source_endpoint_contract=_decoded_source(outputs),
            preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114})
    base, alternate = contract([80, 60]), contract([60, 80])
    metadata = {"shape": [64, 64, 3], "dtype": "uint8", "original_wh": [60, 80]}
    assert binding._detection_contract(alternate, base, metadata) == alternate
    with pytest.raises(ValueError, match="geometry"):
        binding._detection_contract(alternate, base, {**metadata, "original_wh": [80, 60]})
    with pytest.raises(ValueError, match="geometry"):
        binding._detection_contract(alternate, base, {**metadata, "geometry": {"mode": "resize"}})
    changed = deepcopy(alternate)
    changed["processor_contract"]["confidence_threshold"] = .99
    with pytest.raises((ValueError, product.FrozenPostprocessError)):
        binding._detection_contract(changed, base, metadata)


def test_two_fresh_normal_detection_factories_have_exact_comparable_results():
    from tests.test_v272_detection_completion_runtime import _decoded_outputs, _decoded_source
    product = binding._detection_module()
    outputs = _decoded_outputs()
    contract = product.build_detection_completion_execution_contract(model_id="yolo26s", outputs=outputs,
        input_hw=[64, 64], original_wh=[80, 60], source_endpoint_contract=_decoded_source(outputs),
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114})
    processors = [product.create_detection_completion_runtime(contract, source_paths=(__file__,),
        sentinel_context={"prepared_input_sha256": "a" * 64}) for _ in range(2)]
    for _ in range(2):
        results = [processor.process(outputs) for processor in processors]
        assert binding._report_value(results[0]) == binding._report_value(results[1])
    assert [processor.completed_count for processor in processors] == [2, 2]
