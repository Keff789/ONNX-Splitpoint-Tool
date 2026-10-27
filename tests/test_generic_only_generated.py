"""Actual generated completion hook, scheduler and detector processors on CPU.

Only accelerator SDK/storage and compiled-subnet arithmetic are fixtures. No
validator, timer, mixed-control, completion processor or saved-proof checker is
stubbed. This is generated-hook integration, not compiled-hardware acceptance.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as detection
from onnx_splitpoint_tool import native_three_stage  # Load the actual postflight oracle before binding sources.
from onnx_splitpoint_tool.runners import generic_energy, generic_only, generic_mixed_control, generic_mixed_control_binding
from onnx_splitpoint_tool.runners.generic_async import DeepXAsyncStage1, HailoAsyncStage1
from onnx_splitpoint_tool.runners.generic_mixed_control_binding import run_bound_mixed_control
from onnx_splitpoint_tool.runners.task_completion import valid_split_completion_repetitions
from tests.test_v272_detection_completion_runtime import _decoded_outputs, _runtime
from tests.test_v2912_generic_split_completion import TEMPLATE, _module


def descriptor(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def boundary(value):
    return {"cut_a": np.arange(6, dtype=np.float32).reshape(1, 6) + value,
            "cut_b": np.asarray([[10 + value, 20 + value]], dtype=np.uint8)}


def image_value(array):
    return int(np.asarray(array).reshape(64, 64, 3)[32, 32, 0])


class Engine:
    def __init__(self):
        self.jobs, self.submitted, self.collected = {}, [], []

    def get_input_tensors_info(self):
        return [{"name": "images", "shape": [1, 64, 64, 3], "dtype": "uint8"}]

    def get_output_tensors_info(self):
        return [{"name": "cut_a", "shape": [1, 6], "dtype": "float32"},
                {"name": "cut_b", "shape": [1, 2], "dtype": "uint8"}]

    def run_async(self, arrays, *, user_arg, output_buffer):
        arrays_out = [np.ndarray((1, 6), dtype=np.float32, buffer=output_buffer),
                      np.ndarray((1, 2), dtype=np.uint8, buffer=output_buffer, offset=24)]
        values = boundary(image_value(arrays[0]))
        for target, source in zip(arrays_out, values.values()):
            np.copyto(target, source)
        self.jobs[user_arg] = arrays_out
        self.submitted.append(user_arg)
        return user_arg

    def wait(self, job_id):
        self.collected.append(job_id)
        return self.jobs.pop(job_id)


class DeepXSession:
    def __init__(self):
        self._engine = Engine()

    def _prepare_async_inputs(self, feed):
        return [feed["images"]]

    def _materialize_outputs(self, arrays):
        return dict(zip(("cut_a", "cut_b"), arrays))


class HailoSession:
    timeout_ms = 2000

    def __init__(self):
        self.submitted, self.collected = [], []

    def _create_reusable_binding_slot(self):
        done = threading.Event()
        done.set()
        values = boundary(0)
        return {"binding": values, "output_buffers": values,
                "input_buffers": {"images": np.empty((64, 64, 3), dtype=np.uint8)}, "done": done}

    def _prepare_infer_inputs(self, feed):
        return feed

    def _submit_reusable_slot(self, slot, feed, *, copy_inputs):
        assert copy_inputs and slot["done"].is_set()
        np.copyto(slot["input_buffers"]["images"], feed["images"])
        for name, value in boundary(image_value(slot["input_buffers"]["images"])).items():
            np.copyto(slot["binding"][name], value)
        self.submitted.append(id(slot))

    def _wait_reusable_slot(self, slot):
        self.collected.append(id(slot))
        assert slot["done"].is_set()

    def _materialize_binding_outputs(self, binding, *, copy_outputs):
        assert copy_outputs
        return {name: value.copy() for name, value in binding.items()}


def prepared_case(tmp_path, backend):
    case = tmp_path / "b001"
    case.mkdir()
    output = tmp_path / "result"
    output.mkdir()
    outputs = _decoded_outputs()
    completion = _runtime(outputs).execution_contract
    inputs = []
    for index in (0, 1):
        # Constant 80x60 source: exact 64x48 letterbox with eight padded rows.
        array = np.full((64, 64, 3), 114, dtype=np.uint8)
        array[8:56] = index
        path = case / f"prepared_{index}.npy"
        np.save(path, array, allow_pickle=False)
        image = case / f"image_{index}.png"
        Image.fromarray(np.full((60, 80, 3), index, dtype=np.uint8)).save(image)
        inputs.append({"prepared_input": {**descriptor(path), "input_name": "images", "shape": [64, 64, 3],
            "dtype": "uint8", "format": "npy"}, "provenance": {"synthetic_image": descriptor(image)},
            "original_wh": [80, 60], "completion_contract": completion})
    artifacts = {}
    for role in generic_only.ARTIFACT_ROLES:
        path = case / (role + ".fixture")
        path.write_bytes(("synthetic-compiled-storage-only-" + role).encode())
        artifacts[role] = descriptor(path)
    observation = {"schema_version": 1, "minimum_boundary_tensors": 2,
        "part1_outputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                          for name, value in boundary(0).items()},
        "part2_inputs": {"trt_a": {"shape": [1, 6], "dtype": "float32"},
                         "trt_b": {"shape": [1, 2], "dtype": "uint8"}},
        "part2_to_part1": {"trt_a": "cut_a", "trt_b": "cut_b"}, "passthrough_inputs": {},
        "transfer_contract": {"boundary_mode": "synthetic_identity", "fixture_only": True}}
    first_image = inputs[0]["provenance"]["synthetic_image"]
    config = {"measurement_mode": generic_only.MODE, "admission_mode": "generic_only", "task": "detection",
        "identity": {"model_id": "yolo26s", "case_id": "b001", "setup_id": "synthetic_cpu_only",
                     "precision": "float32_layout_fp16"}, "execution_id": "synthetic-generated-integration",
        "original_run": str(case), "case_dir": str(case),
        "original_run_cfg": {"stage1_provider": "hailo10h" if backend == "hailo" else "deepx_m1",
            "stage2_provider": "tensorrt", "provider": "tensorrt", "native_trt_precision": "fp16",
            "streaming_impl": "auto"},
        "stage1_execution": {"policy": f"{backend}_async_slots_v1", "inflight": 8 if backend == "hailo" else 4},
        "input_binding": {"prepared_input": inputs[0]["prepared_input"], "runtime_input_format": "uint8",
            "source_image": first_image["path"], "source_image_sha256": first_image["sha256"],
            "prepared_input_contract": {"shape": [64, 64, 3], "dtype": "uint8", "layout": "HWC",
                "normalization": "hef_quant_info_from_unit_float32" if backend == "hailo" else "none",
                "task": "detection", "preprocess_mode_effective": "letterbox",
                "source_image": first_image["path"], "source_image_sha256": first_image["sha256"]}},
        "functional_reference": {"kind": "serial_compiled_subnets", "input_sha256": inputs[0]["prepared_input"]["sha256"],
            "outputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in outputs.items()},
            "tolerances": {"match_iou_threshold": .5, "min_match_ratio": 1., "min_mean_iou": 1.}},
        "expected_runtime_artifacts": {role: artifacts[role] for role in ("part1_runtime", "engine")},
        "completion_contract": completion, "boundary_observation": observation,
        "generic_artifact_binding": {"schema": generic_only.ARTIFACT_SCHEMA,
            "runtime_precision": "float32_layout_fp16", "artifacts": artifacts,
            "boundary_contract": observation, "quality_reference": {"synthetic_cpu_fixture": True},
            "producer_source": {"synthetic_cpu_fixture": True}},
        "repetitions": 3, "warmup_count": 100, "task_count": 1000, "queue_depth": 1, "timeout_s": 30,
        "measurement_boundary": "prepared_input_before_p1_to_last_completed_task"}
    payload = {key: deepcopy(config[key]) for key in ("identity", "execution_id", "expected_runtime_artifacts",
                                                   "stage1_execution", "boundary_observation")}
    payload.update(task_count=32, inputs=inputs, provenance={"synthetic_cpu_only": True})
    sidecar = case / "control.json"
    sidecar.write_text(json.dumps(payload))
    config["mixed_input_control"] = descriptor(sidecar)
    sources = detection.native_full_runtime_source_hashes(TEMPLATE, __file__)
    sources.update({item["prepared_input"]["path"]: item["prepared_input"]["sha256"] for item in inputs})
    sources[str(sidecar)] = config["mixed_input_control"]["sha256"]
    config["runtime_source_files"] = [{"path": path, "sha256": digest} for path, digest in sorted(sources.items())]
    return config, output, np.load(inputs[0]["prepared_input"]["path"], allow_pickle=False)


def execute(monkeypatch, tmp_path, backend, processor_mode, *, configure_energy=None):
    module = _module(monkeypatch)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_energy", generic_energy)
    actual_factory = detection.create_detection_completion_runtime
    processors = []

    def factory(*args, **kwargs):
        runtime = actual_factory(*args, **kwargs, runtime_mode=processor_mode)
        processors.append(runtime)
        return runtime

    monkeypatch.setattr(detection, "create_detection_completion_runtime", factory)
    config, output, baseline = prepared_case(tmp_path, backend)
    if configure_energy is not None:
        configure_energy(config, output)
    events = []
    calls = {"transfer": 0, "stage2": 0}
    session = HailoSession() if backend == "hailo" else DeepXSession()
    cls = HailoAsyncStage1 if backend == "hailo" else DeepXAsyncStage1
    adapter = cls(session, lambda: {"images": baseline}, inflight=config["stage1_execution"]["inflight"])
    reusable_output = _decoded_outputs()["detections"].copy()

    def stage1(feed=None):
        events.append(("serial_p1", time.perf_counter_ns()))
        return boundary(image_value((feed or {"images": baseline})["images"]))

    def transfer(outputs, feed=None):
        calls["transfer"] += 1
        return {"trt_a": outputs["cut_a"], "trt_b": outputs["cut_b"]}

    def stage2(inputs, feed=None):
        calls["stage2"] += 1
        # Both actual boundary tensors affect the completed task.
        discriminator = int(inputs["trt_a"][0, 0]) + int(inputs["trt_b"][0, 0]) - 10
        reusable_output[:] = _decoded_outputs()["detections"]
        reusable_output[0, 0, (0, 2)] += discriminator
        return {"detections": reusable_output}

    def live_boundary():
        return {"part1_io": {"canonical_outputs": ["cut_a", "cut_b"], "fixture_only": True},
            "part2_io": {"inputs": config["boundary_observation"]["part2_inputs"]},
            **{key: config["boundary_observation"][key] for key in (
                "part2_to_part1", "passthrough_inputs", "transfer_contract")}}

    def mixed(index):
        events.append(("mixed_start", time.perf_counter_ns()))
        proof = run_bound_mixed_control(config, output, index, serial_stage1=stage1, transfer=transfer,
            stage2=stage2, stage1_async=adapter, prepare_async_feed=lambda feed: feed,
            clone_payload=None, runner_source=str(TEMPLATE), boundary_runtime=live_boundary)
        events.append(("mixed_end", time.perf_counter_ns()))
        return proof

    args = {"clone_payload": None, "repetition_index": 1, "stage1_async": adapter,
            "mixed_control": mixed if config.get('mixed_input_control') else None}
    runtime_artifacts = {role: item["path"] for role, item in config["expected_runtime_artifacts"].items()}
    try:
        status = module._run_generic_split_completion(config, output, stage1, transfer, stage2, runtime_artifacts, **args)
    except Exception:
        if len(processors) >= 2:
            left = processors[1].last_result
            right = processors[0].last_result
            differences = {key: {"serial": left.get(key), "measured": right.get(key)}
                           for key in set(left) | set(right) if left.get(key) != right.get(key)}
            (tmp_path / "GENERATED_FAILURE.json").write_text(json.dumps({"backend": backend,
                "processor_mode": processor_mode, "different_completion_fields": differences}, indent=2))
        raise
    assert status == 0
    result = json.loads((output / ("generic_energy_results.json" if config.get('energy_capture')
                                  else "generic_completion_results.json")).read_text())
    return SimpleNamespace(module=module, config=config, output=output, events=events, adapter=adapter,
        session=session, processors=processors, result=result, callbacks=(stage1, transfer, stage2),
        runtime_artifacts=runtime_artifacts, kwargs=args, calls=calls)


@pytest.mark.parametrize("backend", ["hailo", "deepx"])
@pytest.mark.parametrize("processor_mode", ["auto", "strict"])
def test_generated_generic_only_real_pipeline_counts_serial_proof_and_reentry(monkeypatch, tmp_path, backend, processor_mode):
    case = execute(monkeypatch, tmp_path, backend, processor_mode)
    valid = valid_split_completion_repetitions(case.result, case.config, result_root=case.output)
    assert len(valid) == 1
    row = valid[0]
    assert row["warmup_completed_count"] == 100 and row["completed_task_count"] == 1000
    assert row["processor_count_delta"] == 1100
    assert row["stage1_peak_uncollected_jobs"] == (8 if backend == "hailo" else 4)
    counter = "invocation_index" if processor_mode == "strict" else "completed_count"
    assert row["serial_reference"]["last_completed_result"][counter] == 1
    assert row["last_completed_result"][counter] == 1100
    assert row["serial_reference"]["last_completed_result"]["detections"] == row["last_completed_result"]["detections"]
    assert row["completion_execution_attestation"]["completed_work_units"] == 1100
    assert row["mixed_input_control"]["boundary_observation"]["observed_boundary_tensor_count"] == 2
    assert row["mixed_input_control"]["mixed_completed_count"] == 32
    assert len([name for name, _ in case.events if name == "serial_p1"]) == 33
    assert all(stamp < row["measurement_start_ns"] for _, stamp in case.events)
    assert not case.adapter.pending
    sdk = case.session if backend == "hailo" else case.session._engine
    assert len(sdk.submitted) == len(sdk.collected) == 1132
    assert case.calls == {"transfer": 1165, "stage2": 1165}
    assert case.result["status"] == "partial" and case.result["measurement_mode"] == generic_only.MODE
    serial_files = {path: path.read_bytes() for path in (case.output / "serial_reference").rglob("*") if path.is_file()}
    old_events, old_processors = list(case.events), len(case.processors)
    assert case.module._run_generic_split_completion(case.config, case.output, *case.callbacks,
        case.runtime_artifacts, **case.kwargs) == 0
    assert case.events == old_events and len(case.processors) == old_processors
    assert all(path.read_bytes() == value for path, value in serial_files.items())
    assert len(sdk.submitted) == 1132
    assert case.calls == {"transfer": 1165, "stage2": 1165}


def test_generated_result_normal_reader_rejects_serial_and_mode_tampering(monkeypatch, tmp_path):
    case = execute(monkeypatch, tmp_path, "deepx", "auto")
    for fault in ("serial_count", "measured_count", "semantic", "serial_missing", "old_mode", "old_admission", "process_policy", "execution", "repeat_execution"):
        altered = deepcopy(case.result)
        row = altered["repetitions"][0]
        if fault == "serial_count": row["serial_reference"]["last_completed_result"]["completed_count"] = 2
        elif fault == "measured_count": row["last_completed_result"]["completed_count"] = 1000
        elif fault == "semantic": row["last_completed_result"]["detections"][0]["class_id"] += 1
        elif fault == "serial_missing": row.pop("serial_reference")
        elif fault == "old_mode": altered["measurement_mode"] = "generic_completed_task"
        elif fault == "process_policy": altered.pop("process_repetition_policy")
        elif fault == "execution": altered["execution_id"] = "foreign-execution"
        elif fault == "repeat_execution": row["execution_id"] = "foreign-execution"
        else: altered.pop("admission_mode")
        with pytest.raises(ValueError):
            valid_split_completion_repetitions(altered, case.config, result_root=case.output)
    old_config = deepcopy(case.config)
    old_config.pop("admission_mode")
    old_config["measurement_mode"] = "generic_completed_task"
    with pytest.raises(ValueError):
        valid_split_completion_repetitions(case.result, old_config, result_root=case.output)
    old_result = deepcopy(case.result)
    old_result.pop("admission_mode")
    old_result["measurement_mode"] = "generic_completed_task"
    with pytest.raises(ValueError, match="paired Generic result cannot adopt"):
        valid_split_completion_repetitions(old_result, old_config, result_root=case.output)


@pytest.mark.parametrize("backend", ["hailo", "deepx"])
def test_configure_generic_only_precision_binds_producer_identity(monkeypatch, tmp_path, backend):
    module = _module(monkeypatch)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_mixed_control_binding", generic_mixed_control_binding)
    monkeypatch.setenv("ONNX_SPLITPOINT_ARTIFACT_POLICY", "synthetic_scope_before_configure")
    config, output, _ = prepared_case(tmp_path, backend)
    config["original_run_cfg"]["generic_quality_completion_config"] = "/historical/quality-only.json"
    config["original_run_cfg"]["quality_source_only"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    args = SimpleNamespace(generic_completion_config=str(path), generic_completion_repetition=1,
        generic_completion_energy=False, out_dir=str(output), model_id="yolo26s", quality_evidence_model_id="",
        quality_evidence_setup_id="synthetic_cpu_only", native_trt_precision="fp16", provider="tensorrt",
        stage1_provider="deepx_m1", stage2_provider="tensorrt", quality_evidence_only=False, build_only=False,
        generic_quality_completion_config="", quality_source_only=False)
    actual = module._configure_generic_completion(args, argparse.ArgumentParser())
    assert actual["identity"]["precision"] == "float32_layout_fp16"
    assert args.native_trt_precision == "fp16"
    assert args.native_trt_build is False and args.native_trt_no_fallback is True
    assert args.trt_runtime == "native"
    assert args.quality_source_only is False and args.generic_quality_completion_config == ""
    assert actual["original_run_cfg"]["generic_quality_completion_config"] == "/historical/quality-only.json"
    config["generic_artifact_binding"]["runtime_precision"] = "fp16"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit):
        module._configure_generic_completion(args, argparse.ArgumentParser())
