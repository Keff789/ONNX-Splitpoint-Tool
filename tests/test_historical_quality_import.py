"""Import-only historical relations; synthetic files, no devices or services."""
from copy import deepcopy
from dataclasses import replace
import json
from unittest.mock import patch

import pytest

from onnx_splitpoint_tool import native_detection_postprocess as native
from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract
from onnx_splitpoint_tool.native_three_stage import FastDetectionCompletionRuntime, NativeThreeStageError
from onnx_splitpoint_tool.workflow import historical_quality_import as subject
from test_generic_quality_completion import descriptor
from test_v272_detection_completion_runtime import _decoded_outputs, _decoded_source
from test_v272_hailo8_completed_detection_hotloop import _sealed_hailo8_command_contract


OLD_IMPLEMENTATION = "e4999c38f46ed4736c83899d01d2ef220a8fb41b32d19d117d6d8d67fde1995b"


def contract(*, historical=False, confidence=.25, implementation=None, original_wh=(80, 60)):
    artifacts = deepcopy(native._implementation_artifacts())
    if historical or implementation:
        artifacts["native_detection_postprocess"]["sha256"] = implementation or OLD_IMPLEMENTATION
    outputs = _decoded_outputs()
    with patch.object(native, "_implementation_artifacts", return_value=artifacts):
        return native.build_detection_completion_execution_contract(
            model_id="yolo26s", outputs=outputs, input_hw=[64, 64], original_wh=original_wh,
            source_endpoint_contract=_decoded_source(outputs),
            preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114},
            confidence_threshold=confidence)


def postflight(execution, count, *, prepared="a" * 64, image="b" * 64):
    runtime = FastDetectionCompletionRuntime(execution)
    runtime.bind_sentinel_context({"prepared_input_sha256": prepared, "input_image_sha256": image})
    for _ in range(count):
        runtime.process(_decoded_outputs())
    return runtime.attestation()


class Fixture:
    def __init__(self, tmp_path, monkeypatch, case_id="b321"):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.tmp_path = tmp_path
        self.old, self.current = contract(historical=True), contract()
        self.identity = {"model_id": "yolo26s", "case_id": case_id, "setup_id": "orin_nx_hailo8_01",
            "direction": "hailo8_to_trt", "comparison_backend": "hailo8", "precision": "uint8_dequant_fp16",
            "output_endpoint_id": self.old["source_endpoint"]["output_endpoint_id"]}
        self.uid = "generic|yolo26s|H8|hailo8_to_trt|" + case_id
        self.artifacts = {}
        for role in ("source_part1_onnx", "source_part2_onnx", "part1_runtime", "engine",
                     "build_part2_onnx", "boundary_metadata"):
            path = tmp_path / role
            path.write_bytes(("test-only " + role).encode())
            self.artifacts[role] = descriptor(path)
        self.prepared = {"sha256": "a" * 64, "shape": [64, 64, 3], "dtype": "uint8"}
        self.input = {"prepared_input": self.prepared, "source_image_sha256": "b" * 64,
                      "preprocessing_contract_sha256": "c" * 64}
        self.config = {"identity": self.identity, "completion_contract": self.current,
                       "generic_artifact_binding": {"artifacts": self.artifacts}}
        command = _sealed_hailo8_command_contract("hailo8_python_vstreams_fifo")
        command.update(model="yolo26s", case=case_id, precision=self.identity["precision"])
        command["input_image_sha256"] = "b" * 64
        command["artifacts"].update(hef=self.artifacts["part1_runtime"], engine=self.artifacts["engine"])
        self.command = seal_native_command_contract(command)
        self.native = {"ok": True, "task": "detection", "case": case_id,
            "setup_id": self.identity["setup_id"], "precision": self.identity["precision"],
            "completion_execution_contract": self.old, "native_command_contract": self.command,
            "native_command_contract_sha256": self.command["contract_sha256"],
            "timing_input_binding": {"canonical_rgb_sha256": "a" * 64, "input_image_sha256": "b" * 64,
                "preprocessing_contract_sha256": "c" * 64, "executed_rgb_matches_common_preprocessing": True},
            "completed_work_units": 2, "requested_frames": 2, "postprocess_completed_frames": 2,
            "frames": 2, "warmup": 1,
            "postprocess_completion_verified": True, "completed_work_units_status": "exact_runtime_counter",
            "completion_execution_attestation": postflight(self.old, 2)}
        self.source_map = {"/original/runner.py": "d" * 64}
        self.fps = {"identity": self.identity, "completion_contract": self.old, "input_binding": self.input,
            "expected_runtime_artifacts": self.artifacts, "status": "ok", "task": "detection",
            "measurement_mode": "generic_completed_task", "measurement_boundary": "prepared_input_before_p1_to_last_completed_task",
            "process_repetition_policy": "separate_processes", "completed_task_fps": 12.5,
            "repetitions": [{"repetition_index": index, "status": "ok", "boot_id": "test-boot",
                "pid": 100 + index, "process_start_ticks": index, "runtime_instance_id": "test-" + str(index),
                "completed_task_count": 2, "requested_task_count": 2,
                "requested_warmup_count": 1, "warmup_completed_count": 1, "processor_count_delta": 3,
                "queue_depth": 2, "worker_count": 2, "measurement_boundary": "prepared_input_before_p1_to_last_completed_task",
                "fill_and_drain_included": True, "preprocessing_timed": False, "handoff_timed": True,
                "prepared_input_adapter_timed": True, "postprocess_timed": True, "output_materialization_timed": True,
                "completion_runtime_mode": "fast_oracle_outside_timing",
                "runtime_source_binding": {"before": self.source_map, "after": self.source_map, "unchanged": True},
                "runtime_artifacts": self.artifacts,
                "completion_execution_attestation": postflight(self.old, 3)} for index in (1, 2, 3)]}
        self.original = {"identity": self.identity, "model_sha256": "e" * 64, "completion_contract": self.old,
            "native_command_contract": self.command, "result_path": str(tmp_path / "fps.json"),
            "native_source": str(tmp_path / "native.json"), "expected_runtime_artifacts": self.artifacts,
            "runtime_config": {"completion_contract": self.old, "input_binding": self.input,
                "measurement_boundary": self.fps["measurement_boundary"], "task_count": 2, "warmup_count": 1,
                "repetitions": 3, "queue_depth": 2, "runtime_source_files": [{"path": k, "sha256": v}
                    for k, v in self.source_map.items()]}}
        self.plan = {"rows": [self.original]}
        self.result = {"status": "completed", "technical_status": "completed", "task": "detection",
            "variant": "composed", "model_id": "yolo26s", "case_id": case_id,
            "setup_id": self.identity["setup_id"], "model_sha256": "e" * 64,
            "preprocessing_contract_sha256": "c" * 64, "decision": "FAIL", "n": 2,
            "accuracy_assessment": {"status": "accuracy_loss", "quality_eligible": False}}
        self.proof = {"request_sha256": "f" * 64, "result_sha256": "1" * 64,
            "status": "accuracy_loss", "quality_eligible": False,
            "completed_quality_binding": {"base_execution_contract_sha256": self.current["contract_sha256"],
                "consumer_observer_implementation": {"path": "/old/product.py", "sha256": "2" * 64, "size_bytes": 12},
                "representative_input": {"prepared_input": self.prepared,
                    "source_image_sha256": "b" * 64, "original_wh": [80, 60]}}}
        self.qrow = {"identity": self.identity, "completion_contract": self.current,
            "generic_artifact_binding": self.config["generic_artifact_binding"],
            "producer_run_binding": {"run_dir": "/own/quality", "session_id": "own-session"},
            "quality_reference": {"result": self.save("quality_result", self.result)}}
        self.target = {"generic_uid": self.uid, "model_id": "yolo26s", "case_id": case_id,
            "prior_generic_sourceversion_reuse_admitted": True, "admission_mode": "reused_prior_generic_sourceversion_h8",
            "full_product_identity": self.identity, "model_sha256": "e" * 64, "effective_generic_fps": "12.5",
            "effective_repeats": [{"original": True}], "attempts": {"fps": ["original-attempt"]},
            "actual_artifact_authority": {"source_part1_onnx": self.artifacts["source_part1_onnx"],
                "source_part2_onnx": self.artifacts["source_part2_onnx"], "part1": self.artifacts["part1_runtime"],
                "part2_engine_observation": {"artifact": self.artifacts["engine"]}}}
        self.authority = object()
        self.normal_calls = []

        def normal_validate(row, *, producer_source_authority):
            # The unrelated normal Q producer is admitted against its own base.
            assert row == self.qrow
            assert row["completion_contract"] == self.current != self.old
            assert producer_source_authority is self.authority
            self.normal_calls.append(deepcopy(row))
            proof = deepcopy(self.proof)
            proof["completed_quality_binding"]["consumer_observer_implementation"]["path"] = "/new/product.py"
            return proof

        self.normal_validate = normal_validate
        monkeypatch.setattr(subject.quality_admission, "validate_quality_reference", normal_validate)

    def save(self, name, value):
        path = self.tmp_path / (name + ".json")
        path.write_text(json.dumps(value))
        return descriptor(path)

    def bind(self):
        fps = self.save("fps", self.fps)
        plan = self.save("plan", self.plan)
        self.target["source_authorities"] = {"reused_original_result": fps, "reused_original_plan": plan}
        return subject.HistoricalQualityImport(fps_result=fps, fps_plan=plan,
            native_record=self.save("native", self.native), quality_config=self.save("config", self.config),
            quality_row=self.qrow, quality_admission=self.save("admission", self.proof),
            producer_source_authority=self.authority)

    def run(self):
        binding = self.bind()
        return subject.import_historical_operator_quality([self.target], {self.uid: binding})


def test_two_normal_projections_keep_other_418_rows_and_every_original_fps_field(tmp_path, monkeypatch):
    fixtures = [Fixture(tmp_path / name, monkeypatch, case) for name, case in (("first", "b321"), ("second", "b322"))]
    imports = {f.uid: f.bind() for f in fixtures}
    def dispatch(row, **kwargs):
        return next(f for f in fixtures if f.identity == row["identity"]).normal_validate(row, **kwargs)
    monkeypatch.setattr(subject.quality_admission, "validate_quality_reference", dispatch)
    others = [{"generic_uid": "unrelated-" + str(i), "nested": {"fps_repeats": [i]}} for i in range(418)]
    original = deepcopy([*(f.target for f in fixtures), *others])
    result = subject.import_historical_operator_quality(original, imports)
    assert result[2:] == original[2:] and original[:2] == [f.target for f in fixtures]
    for index, fixture in enumerate(fixtures):
        assert len(fixture.normal_calls) == 1
        relation = result[index]["historical_completed_operator_quality_relation"]
        assert relation["historical_observation_relation"] == "postflight_oracle_sentinel"
        assert relation["historical_capture_ownership"] == "owned_postflight_after_fifo_drain"
        assert relation["original_source_attestation_body_verified"] is False
        assert relation["current_execution_admission"] is False
        assert relation["fps_completion_contract_sha256"] != relation["quality_completion_contract_sha256"]
        assert result[index]["central_quality_join_mode"] == subject.RELATION_MODE
        assert result[index]["quality_status"] == "accuracy_loss" and result[index]["quality_eligible"] is False
        for key, value in original[index].items():
            assert result[index][key] == value


@pytest.mark.parametrize("fault", ["threshold", "operator_geometry", "geometry", "preprocess", "dtype", "full_model",
    "artifact", "q_base", "q_admission", "native_missing", "native_contract", "native_command",
    "native_selfhash", "native_ownership", "native_count", "native_input", "native_artifact",
    "fps_selfhash", "fps_count", "fps_queue", "fps_process", "fps_source", "fps_artifact", "fps_value"])
def test_scientific_or_actual_backing_mismatch_rejected(tmp_path, monkeypatch, fault):
    f = Fixture(tmp_path, monkeypatch)
    if fault == "threshold":
        f.current["processor_contract"]["confidence_threshold"] = .3
    elif fault == "operator_geometry":
        f.current = contract(original_wh=[81, 60]); f.config["completion_contract"] = f.current; f.qrow["completion_contract"] = f.current
    elif fault == "geometry": f.proof["completed_quality_binding"]["representative_input"]["original_wh"] = [81, 60]
    elif fault == "preprocess": f.fps["input_binding"]["preprocessing_contract_sha256"] = "9" * 64
    elif fault == "dtype": f.proof["completed_quality_binding"]["representative_input"]["prepared_input"] = {**f.prepared, "dtype": "float32"}
    elif fault == "full_model": f.target["model_sha256"] = "9" * 64
    elif fault == "artifact": f.target["actual_artifact_authority"]["source_part1_onnx"] = {"sha256": "9" * 64, "size_bytes": 1}
    elif fault == "q_base": f.qrow["completion_contract"] = f.old
    elif fault == "q_admission":
        # A producer/config evidence change is not an allowed path relocation.
        saved = deepcopy(f.proof); saved["completed_quality_binding"]["base_execution_contract_sha256"] = "9" * 64
        binding = f.bind(); binding = replace(binding, quality_admission=f.save("bad_admission", saved))
        with pytest.raises(ValueError, match="own normal revalidation"):
            subject.import_historical_operator_quality([f.target], {f.uid: binding})
        return
    elif fault == "native_missing": f.native.pop("completion_execution_attestation")
    elif fault == "native_contract": f.native["completion_execution_contract"] = f.current
    elif fault == "native_command": f.native["native_command_contract_sha256"] = "9" * 64
    elif fault == "native_selfhash": f.native["completion_execution_attestation"]["attestation_sha256"] = "9" * 64
    elif fault == "native_ownership":
        att = f.native["completion_execution_attestation"]; att["sentinel_identity"].pop("source_snapshot_ownership")
        att.pop("attestation_sha256"); att["attestation_sha256"] = native.canonical_json_sha256(att)
    elif fault == "native_count": f.native["completed_work_units"] = 1
    elif fault == "native_input": f.native["timing_input_binding"]["canonical_rgb_sha256"] = "9" * 64
    elif fault == "native_artifact":
        att = f.native["completion_execution_attestation"]; att["artifact"]["detections"] = []
        att.pop("attestation_sha256"); att["attestation_sha256"] = native.canonical_json_sha256(att)
    elif fault == "fps_selfhash": f.fps["repetitions"][0]["completion_execution_attestation"]["attestation_sha256"] = "9" * 64
    elif fault == "fps_count": f.fps["repetitions"][0]["completed_task_count"] = 1
    elif fault == "fps_queue": f.fps["repetitions"][0]["queue_depth"] = 3
    elif fault == "fps_process": f.fps["repetitions"][1]["runtime_instance_id"] = f.fps["repetitions"][0]["runtime_instance_id"]
    elif fault == "fps_source": f.fps["repetitions"][0]["runtime_source_binding"]["unchanged"] = False
    elif fault == "fps_artifact": f.fps["repetitions"][0]["runtime_artifacts"] = {**f.artifacts, "engine": {"sha256": "9" * 64}}
    elif fault == "fps_value": f.target["effective_generic_fps"] = "12.6"
    with pytest.raises((ValueError, native.FrozenPostprocessError, NativeThreeStageError)):
        f.run()


def test_unknown_revision_and_untyped_or_missing_backing_rejected(tmp_path, monkeypatch):
    f = Fixture(tmp_path, monkeypatch)
    unknown = contract(historical=True, implementation="9" * 64)
    with pytest.raises(native.FrozenPostprocessError):
        subject._operator(unknown, model_sha256="e" * 64)
    binding = f.bind()
    with pytest.raises(ValueError, match="typed"):
        subject.import_historical_operator_quality([f.target], {f.uid: dict(binding.__dict__)})
    with pytest.raises(ValueError):
        subject.import_historical_operator_quality([f.target], {f.uid: replace(binding,
            native_record={**binding.native_record, "path": str(tmp_path / "absent-native.json")})})


def test_normal_quality_rejection_propagates_without_projection(tmp_path, monkeypatch):
    f = Fixture(tmp_path, monkeypatch)
    def reject(row, **kwargs):
        assert row["completion_contract"] == f.current
        raise ValueError("actual normal producer admission failed")
    monkeypatch.setattr(subject.quality_admission, "validate_quality_reference", reject)
    original = deepcopy(f.target)
    with pytest.raises(ValueError, match="normal producer"):
        f.run()
    assert "task_quality_gate" not in f.target
    assert f.target["effective_repeats"] == original["effective_repeats"]


def test_current_execution_verifier_still_rejects_historical_postflight():
    execution = contract(historical=True)
    attestation = postflight(execution, 2)
    with pytest.raises(native.FrozenPostprocessError, match="observation_relation"):
        native.verify_detection_completion_execution_attestation(attestation, execution_contract=execution)


def test_duplicate_or_unknown_target_rejected(tmp_path, monkeypatch):
    f = Fixture(tmp_path, monkeypatch); binding = f.bind()
    for rows, imports in (([f.target, f.target], {f.uid: binding}), ([f.target], {"other": binding})):
        with pytest.raises(ValueError, match="absent, duplicate or ambiguous"):
            subject.import_historical_operator_quality(rows, imports)
