"""Synthetic byte/endpoint controls; these tests do not claim hardware quality."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_only as control


def fixture(tmp_path):
    from test_v272_detection_completion_runtime import _decoded_outputs, _runtime
    outputs = _decoded_outputs()
    processor = _runtime(outputs)
    artifacts = {}
    for role in control.ARTIFACT_ROLES:
        path = tmp_path / (role + ".fixture")
        path.write_bytes(("synthetic-" + role).encode())
        artifacts[role] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    config = {"measurement_mode": control.MODE, "admission_mode": "generic_only", "task": "detection",
              "identity": {"precision": "float32_layout_fp16"},
              "stage1_execution": {"policy": "deepx_async_slots_v1", "inflight": 4},
              "original_run_cfg": {"stage1_provider": "deepx_m1"},
              "mixed_input_control": {"path": "/synthetic/control.json", "sha256": "a" * 64},
              "input_binding": {"prepared_input": {"sha256": "b" * 64}},
              "functional_reference": {"kind": "serial_compiled_subnets", "input_sha256": "b" * 64,
                  "outputs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                              for name, value in outputs.items()}},
              "expected_runtime_artifacts": {role: artifacts[role] for role in ("part1_runtime", "engine")},
              "completion_contract": processor.execution_contract,
              "generic_artifact_binding": {"schema": control.ARTIFACT_SCHEMA, "artifacts": artifacts,
                  "runtime_precision": "float32_layout_fp16",
                  "boundary_contract": {"fixture_only": True}, "quality_reference": {"fixture_only": True},
                  "producer_source": {"fixture_only": True}}}
    return config, outputs, processor


def capture(tmp_path):
    config, outputs, processor = fixture(tmp_path)
    events = []
    def stage1():
        events.append("p1")
        return {name: value.copy() for name, value in outputs.items()}
    def transfer(boundary):
        events.append("transfer")
        return boundary
    def stage2(inputs):
        events.append("p2")
        return inputs
    arrays, proof = control.capture_serial_reference(config, tmp_path, 1, stage1=stage1,
        transfer=transfer, stage2=stage2, processor=processor)
    assert events == ["p1", "transfer", "p2"]
    assert processor.completed_count == 1
    measured = {}
    for index, (name, array) in enumerate(arrays.items()):
        path = tmp_path / f"measured_{index}.npy"
        np.save(path, array, allow_pickle=False)
        measured[name] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "shape": list(array.shape), "dtype": str(array.dtype)}
    from test_v272_detection_completion_runtime import _runtime
    measured_processor = _runtime(arrays)
    for _ in range(1100):
        measured_processor.process(arrays)
    repetition = {"serial_reference": proof, "raw_output_samples": measured,
                  "last_completed_result": deepcopy(measured_processor.last_result)}
    return config, arrays, proof, repetition


def test_serial_owns_buffers_and_exact_endpoint(tmp_path):
    config, outputs, processor = fixture(tmp_path)
    arrays, proof = control.capture_serial_reference(config, tmp_path, 1, stage1=lambda: outputs,
        transfer=lambda value: value, stage2=lambda value: value, processor=processor)
    saved = {name: value.copy() for name, value in arrays.items()}
    for name in outputs:
        assert not np.shares_memory(outputs[name], arrays[name])
        outputs[name].fill(0)
        np.testing.assert_array_equal(arrays[name], saved[name])
    assert proof["observed_task_count"] == 1
    assert not any("fps" in key.lower() for key in proof)
    with pytest.raises(FileExistsError):
        from test_v272_detection_completion_runtime import _runtime
        control.capture_serial_reference(config, tmp_path, 1, stage1=lambda: outputs,
            transfer=lambda value: value, stage2=lambda value: value, processor=_runtime(outputs))


def test_serial_saved_samples_are_independently_rechecked(tmp_path):
    config, _, _, repetition = capture(tmp_path)
    control.validate_serial_reference(repetition, config, result_root=tmp_path)


@pytest.mark.parametrize("fault", ["raw", "completed", "serial_hash", "serial_escape", "serial_metadata",
                                   "missing", "input_binding", "artifact_binding", "not_untimed"])
def test_serial_rejects_altered_or_foreign_evidence(tmp_path, fault):
    config, _, proof, repetition = capture(tmp_path)
    first = next(iter(proof["raw_output_samples"]))
    if fault == "raw":
        desc = repetition["raw_output_samples"][first]
        path = tmp_path / desc["path"]
        data = np.load(path); data.flat[0] += 1
        np.save(path, data); desc["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == "completed":
        repetition["last_completed_result"]["detection_count"] += 1
    elif fault == "serial_hash":
        proof["raw_output_samples"][first]["sha256"] = "0" * 64
    elif fault == "serial_escape":
        proof["raw_output_samples"][first]["path"] = "../elsewhere.npy"
    elif fault == "serial_metadata":
        proof["raw_output_samples"][first]["dtype"] = "float64"
    elif fault == "missing":
        repetition.pop("serial_reference")
    elif fault == "input_binding":
        proof["reference_binding"] = deepcopy(proof["reference_binding"])
        proof["reference_binding"]["input_sha256"] = "c" * 64
    elif fault == "artifact_binding":
        proof["runtime_artifacts"] = {}
    elif fault == "not_untimed":
        proof["checked_outside_timer"] = False
    with pytest.raises(ValueError):
        control.validate_serial_reference(repetition, config, result_root=tmp_path)


@pytest.mark.parametrize("fault", ["native", "energy", "sync", "slots", "backend_alias", "mode",
                                   "outputs", "quality", "artifact", "mixed"])
def test_explicit_mode_does_not_weaken_paired_or_artifact_admission(tmp_path, fault):
    config, _, _ = fixture(tmp_path)
    if fault == "native": config["original_run_cfg"]["native_split_quality_binding"] = "/fake-native.json"
    elif fault == "energy": config["energy_capture"] = {"duration_s": 60}
    elif fault == "sync": config.pop("stage1_execution")
    elif fault == "slots": config["stage1_execution"]["inflight"] = 3
    elif fault == "backend_alias": config["original_run_cfg"]["stage1_provider"] = "deepx"
    elif fault == "mode": config.pop("admission_mode")
    elif fault == "outputs": next(iter(config["functional_reference"]["outputs"].values()))["path"] = "/fake.npy"
    elif fault == "quality": config["generic_artifact_binding"].pop("quality_reference")
    elif fault == "artifact": config["generic_artifact_binding"]["artifacts"]["engine"]["sha256"] = "not-a-sha"
    elif fault == "mixed": config.pop("mixed_input_control")
    with pytest.raises(ValueError): control.validate_generic_only_config(config)


def test_config_checks_actual_bytes_before_runtime(tmp_path):
    config, _, _ = fixture(tmp_path)
    paths = control.validate_generic_only_config(config, check_files=True)
    paths["engine"].write_bytes(b"different synthetic compiled fixture")
    with pytest.raises(ValueError, match="bytes changed"):
        control.validate_generic_only_config(config, check_files=True)


def test_mode_markers_must_agree_and_legacy_stays_legacy():
    assert control.is_generic_only({}) is False
    assert control.is_generic_only({"measurement_mode": "generic_completed_task"}) is False
    for value in ({"measurement_mode": control.MODE}, {"admission_mode": "generic_only"}):
        with pytest.raises(ValueError): control.is_generic_only(value)


@pytest.mark.parametrize("fault", ["artifact_count", "artifact_hash", "artifact_content"])
def test_strict_projection_retains_and_checks_artifact_semantics(tmp_path, fault):
    config, _, _, repetition = capture(tmp_path)
    completed = repetition["last_completed_result"]
    if fault == "artifact_count":
        completed["artifact"]["invocation_index"] = 1
    elif fault == "artifact_hash":
        completed["artifact_sha256"] = "0" * 64
    else:
        from onnx_splitpoint_tool.native_detection_postprocess import canonical_json_sha256
        completed["artifact"]["completion_mode"] = "changed_semantic_endpoint"
        completed["artifact_sha256"] = canonical_json_sha256(completed["artifact"])
    with pytest.raises(ValueError):
        control.validate_serial_reference(repetition, config, result_root=tmp_path)
