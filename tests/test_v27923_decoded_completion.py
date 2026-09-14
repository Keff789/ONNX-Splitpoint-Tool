"""Regression: overnight YOLO11 decoded pre-NMS output needs a real host NMS.

The fixture is the unmodified cuda_ort declaration from the collected night's
output_contracts.json (other backends omitted). GPU execution alone is stubbed;
source declaration loading, processor, Full dispatch/hotloop and counts are real.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as pp
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract, runtime_output_contract,
)

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = Path(__file__).parent / "fixtures/v27923/yolo11l_overnight_output_contracts.json"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _outputs(layout="bcn"):
    values = np.zeros((1, 84, 8400), dtype=np.float32)
    # Same-class overlap must be suppressed; a different class is retained.
    values[0, :4, :3] = np.array([[320, 321, 320], [320, 320, 320],
                                 [100, 100, 100], [100, 100, 100]])
    values[0, 4, 0] = .9
    values[0, 4, 1] = .8
    values[0, 5, 2] = .7
    if layout == "bnc":
        values = values.transpose(0, 2, 1)
    elif layout == "cn":
        values = values[0]
    elif layout == "nc":
        values = values[0].T
    return {"output0": values}


def _suite(tmp_path):
    (tmp_path / "output_contracts.json").write_bytes(DECLARATION.read_bytes())
    return tmp_path


def _source(tmp_path, outputs):
    declaration = load_authoritative_output_contract(
        _suite(tmp_path), backend="tensorrt", model_id="yolo11l", task="detection",
    )
    assert declaration["contract_resolution_status"] == "attested"
    source = runtime_output_contract(
        "detection", outputs, raw_fallback=False, declared_contract=declaration,
    )
    assert source["endpoint_contract_complete"] is True
    assert source["stage"] == "decoded_pre_nms"
    return source


def _runtime(tmp_path, outputs):
    return pp.build_detection_completion_runtime(
        model_id="yolo11l", outputs=outputs, input_hw=[640, 640],
        original_wh=[640, 480], source_endpoint_contract=_source(tmp_path, outputs),
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114},
    )


@pytest.mark.parametrize("layout", ["bcn", "bnc"])
def test_attested_split_output_runs_nms_and_deletterbox(tmp_path, layout):
    outputs = _outputs(layout)
    runtime = _runtime(tmp_path, outputs)
    assert runtime.completed_count == 0
    result = runtime.process(outputs)
    assert result["detection_count"] == 2
    assert [d["class_id"] for d in result["detections"]] == [0, 1]
    assert [result["detections"][0][k] for k in ["x1", "y1", "x2", "y2"]] == [270, 190, 370, 290]
    assert result["host_nms_applied"] is True
    assert result["source_nms_attested"] is False
    assert result["completion_mode"] == "decoded_pre_nms_frozen_nms"
    attestation = runtime.attestation()
    assert attestation["completion_count_verified"] is True
    assert attestation["completion_count"] == 1
    pp.verify_detection_completion_execution_attestation(
        attestation, execution_contract=runtime.execution_contract,
    )


@pytest.mark.parametrize("layout", ["cn", "nc"])
def test_batchless_tensor_keeps_existing_runtime_endpoint_guard(tmp_path, layout):
    declaration = load_authoritative_output_contract(
        _suite(tmp_path), backend="tensorrt", model_id="yolo11l", task="detection",
    )
    source = runtime_output_contract(
        "detection", _outputs(layout), raw_fallback=False, declared_contract=declaration,
    )
    assert source["endpoint_contract_complete"] is False


@pytest.mark.parametrize("mutation", ["missing_attestation", "hash", "stage", "shape"])
def test_completion_still_rejects_unattested_or_drifting_source(tmp_path, mutation):
    outputs = _outputs()
    source = _source(tmp_path, outputs)
    if mutation == "missing_attestation":
        source.pop("output_endpoint_attestation")
    elif mutation == "hash":
        source["endpoint_contract_hash"] = "f" * 64
    elif mutation == "stage":
        source["stage"] = "raw_head"
    else:
        outputs = {"output0": outputs["output0"][:, :, :-1]}
    with pytest.raises(pp.FrozenPostprocessError):
        pp.build_detection_completion_runtime(
            model_id="yolo11l", outputs=outputs, input_hw=[640, 640],
            original_wh=[640, 480], source_endpoint_contract=source,
        )


@pytest.mark.parametrize("value", [np.nan, 1.1, -.1])
def test_invalid_decoded_values_never_increment_counter(tmp_path, value):
    outputs = _outputs()
    runtime = _runtime(tmp_path, outputs)
    runtime.process(outputs)
    invalid = copy.deepcopy(outputs)
    invalid["output0"][0, 4, 0] = value
    with pytest.raises(pp.FrozenPostprocessError, match="decoded_pre_nms"):
        runtime.process(invalid)
    assert runtime.completed_count == 1


def test_execution_source_stage_cannot_be_relabelled(tmp_path):
    runtime = _runtime(tmp_path, _outputs())
    contract = copy.deepcopy(runtime.execution_contract)
    contract.pop("contract_sha256")
    contract["completion_mode"] = "raw_head_frozen_decode_nms"
    contract["contract_sha256"] = pp.canonical_json_sha256(contract)
    with pytest.raises(pp.FrozenPostprocessError, match="source_mode_mismatch"):
        pp.verify_detection_completion_execution_contract(contract)


def test_raw_signature_guard_is_not_relaxed_for_decoded_tensor():
    with pytest.raises(pp.FrozenPostprocessError, match="exact_six_tensors"):
        pp.build_frozen_postprocess_contract(
            model_id="yolo11l", outputs=_outputs(), input_hw=[640, 640], original_wh=[640, 480],
        )


def test_trt_full_semantic_probe_and_measured_dispatch_complete_nms(tmp_path, monkeypatch):
    from PIL import Image
    from scripts import native_full_semantic_dump as dump
    from scripts import native_full_baseline_eval_runner as full

    outputs = _outputs()
    instances = []

    class NativeTRT:
        def __init__(self, _engine):
            self.inputs = ["images"]
            self.shapes = {"images": (1, 3, 640, 640)}
            self.dtypes = {"images": np.dtype("float32")}
            self.count = 0
            self.closed = False
            instances.append(self)

        def prepare_inputs(self, feeds):
            assert feeds["images"].shape == self.shapes["images"]

        def run(self, feeds):
            self.prepare_inputs(feeds)
            return self.run_prepared()

        def run_prepared(self):
            self.count += 1
            return outputs

        def close(self):
            self.closed = True

    native = ModuleType("native_hailo10_trt_e2e_from_benchmarkset")
    native.NativeTRT = NativeTRT
    monkeypatch.setitem(sys.modules, native.__name__, native)
    suite = _suite(tmp_path)
    image = tmp_path / "input.png"
    Image.new("RGB", (640, 480)).save(image)
    engine = tmp_path / "full.engine"
    engine.write_bytes(b"offline test engine; no device/compiler access")
    identity = {"paths": {"engine": str(engine)}, "hashes": {"source_onnx": "a" * 64},
                "quality_first_producer_identity_sha256": "b" * 64}
    probe = dump._run_tensorrt(
        suite, image, tmp_path / "dump", "yolo11l", "detection", "fp16", "offline", "tensorrt", identity,
    )
    assert probe["ok"] is True
    manifest = json.loads(Path(probe["output_manifest"]).read_text())
    assert manifest["contract_family"] == "decoded_pre_nms"
    assert probe["frozen_host_postprocess_contract"]["source_contract_family"] == "decoded_pre_nms"
    assert probe["frozen_host_postprocess_result"]["detection_count"] == 2

    hotloop_path = ROOT / "scripts/native_trt_full_completed_hotloop.py"
    spec = importlib.util.spec_from_file_location("v27923_test_trt_hotloop", hotloop_path)
    hotloop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hotloop)
    measured_calls = []

    def run_actual_hotloop(command, **_kwargs):
        measured_calls.append(command)
        monkeypatch.setattr(sys, "argv", command[1:])
        return {"rc": hotloop.main(), "timed_out": False}

    monkeypatch.setattr(full, "_run", run_actual_hotloop)
    row = full._attach_trt_completed_task_hotloop(
        {"ok": True, "status": "ok", "task": "detection", "model": "yolo11l",
         "backend": "native_full_tensorrt", "contract_family": "decoded_pre_nms",
         "endpoint_contract_hash": manifest["endpoint_contract_hash"],
         "input_manifest": probe["input_manifest"],
         "quality_first_producer_identity": {"engine": {"path": str(engine), "sha256": _sha(engine)}},
         "quality_first_producer_identity_sha256": "b" * 64,
         "frozen_host_postprocess_contract": probe["frozen_host_postprocess_contract"],
         "fps_makespan": 999.0},
        suite, "yolo11l", SimpleNamespace(engine_python_selected=sys.executable, frames=2, warmup=1,
                                          duration_s=0.0, timeout=30),
    )
    assert row["ok"] is True, row
    assert len(measured_calls) == 1
    assert row["fps_makespan"] != 999.0
    assert row["accelerator_only_diagnostic"]["fps_makespan"] == 999.0
    assert row["e2e_scope"] == "full_task_pipeline"
    assert row["completed_work_units"] == row["postprocess_completed_frames"] == 2
    assert row["frozen_host_postprocess_result"]["detection_count"] == 2
    assert row["completed_task_endpoint_attestation"]["source_stage"] == "decoded_pre_nms"
    assert row["completed_task_result_artifact_saved"] is True
    assert instances[-1].count == 4  # probe + warmup + two completed iterations
    assert all(instance.closed for instance in instances)
    from scripts import native_producer_validate_visualize as validator
    assert validator._completed_frozen_nms_attestation_passed(row) is True
    verified, comparison = validator._verified_completed_v2_frozen_contract(row, native_tensors=outputs)
    assert verified["source_contract_family"] == "decoded_pre_nms"
    assert comparison["contract_family"] == "decoded_nms"
    tampered = copy.deepcopy(row)
    tampered["frozen_host_postprocess_contract"]["source_contract_family"] = "raw_head"
    assert validator._completed_frozen_nms_attestation_passed(tampered) is False


def test_fast_split_runtime_keeps_nms_oracle_and_counter_guards(tmp_path):
    from onnx_splitpoint_tool.native_three_stage import (
        FastDetectionCompletionRuntime, infer_p2_output_contract_family,
        adapter_id_for_contract_family,
    )
    outputs = _outputs()
    reference = _runtime(tmp_path, outputs)
    fast = FastDetectionCompletionRuntime(reference.execution_contract)
    result = fast.process(outputs)
    assert result["detections"] == reference.process(outputs)["detections"]
    assert result["detection_count"] == 2
    source = _source(tmp_path, outputs)
    family = infer_p2_output_contract_family(source, model_id="yolo11l", task="detection")
    assert family == "ultralytics_decoded_pre_nms"
    assert adapter_id_for_contract_family(family) == fast.adapter_id
    assert fast.attestation()["status"] == "passed"
    invalid = copy.deepcopy(outputs)
    invalid["output0"][0, 4, 0] = np.nan
    with pytest.raises(pp.FrozenPostprocessError, match="nonfinite"):
        fast.process(invalid)
    assert fast.completed_count == 1


@pytest.mark.parametrize("mutation", [None, "threshold", "integrated_nms", "source_stage", "physical_hash"])
def test_central_quality_projection_requires_same_decoded_nms_semantics(tmp_path, mutation):
    from scripts import native_producer_validate_visualize as validator
    runtime = _runtime(tmp_path, _outputs())
    comparison = runtime.execution_contract["comparison_endpoint_contract"]
    source_hash = runtime.execution_contract["source_endpoint"]["endpoint_contract_hash"]
    record_endpoint = "decoded_xyxy_score_class_detections"
    decoder = {"canonical_record_endpoint": record_endpoint,
               "source_output_format": "ultralytics_decoded",
               "source_endpoint_semantics": "ultralytics_decoded",
               "source_endpoint_has_integrated_nms": False,
               "confidence_threshold": .25}
    nms = {"detr_or_bn6_confidence_threshold": .25, "detr_or_bn6_iou_threshold": .45,
           "detr_or_bn6_max_detections": 300}
    decoder_sha = pp.canonical_json_sha256(decoder)
    nms_sha = pp.canonical_json_sha256(nms)
    producer = {"task": "detection", "model_id": "yolo11l", "endpoint_contract_hash": source_hash,
                "source_onnx": {"sha256": "a" * 64},
                "endpoint": {"identity": {"stage": "decoded_pre_nms"}},
                "decoder_contract_sha256": decoder_sha, "nms_contract_sha256": nms_sha,
                "quality_record_endpoint": {"identity": {"canonical_record_endpoint": record_endpoint,
                    "decoder_contract_sha256": decoder_sha, "nms_contract_sha256": nms_sha}},
                "quality_contract": {"task": "detection", "model": {"sha256": "a" * 64},
                    "canonical_record_endpoint": record_endpoint, "source_endpoint_is_raw": False,
                    "preprocessing": {"identity": {"target_hw": [640, 640]}},
                    "decoder": {"sha256": decoder_sha, "identity": decoder},
                    "nms": {"sha256": nms_sha, "identity": nms}}}
    if mutation == "threshold":
        nms["detr_or_bn6_iou_threshold"] = .5
    elif mutation == "integrated_nms":
        decoder["source_endpoint_has_integrated_nms"] = True
    elif mutation == "source_stage":
        producer["endpoint"]["identity"]["stage"] = "decoded_nms"
    elif mutation == "physical_hash":
        producer["endpoint_contract_hash"] = "f" * 64
    passed, _reason = validator._native_trt_completed_quality_projection(
        producer, comparison, physical_endpoint_hash=source_hash,
    )
    assert passed is (mutation is None)
