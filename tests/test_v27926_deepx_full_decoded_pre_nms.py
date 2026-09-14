"""Replay the supplied D contracts; all engine/ONNX bytes below are synthetic."""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import onnx_splitpoint_tool.native_detection_postprocess as postprocess
import onnx_splitpoint_tool.native_output_endpoint as endpoint
import onnx_splitpoint_tool.runners.native_full_input as full_input
import onnx_splitpoint_tool.workflow.deepx_build_binding as binding

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/v27926"


def _suite():
    path = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    module = types.ModuleType("v27926_deepx_suite")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _recorded():
    return json.loads((FIXTURES / "yolo11l_recorded_output_contracts.json").read_text())


@pytest.fixture
def case(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    model_dir = run_root / "models/yolo11l"
    root = model_dir / "benchmark_set/legacy_suite"
    root.mkdir(parents=True)
    recorded = _recorded()
    _write_json(root / "output_contracts.json", recorded)
    _write_json(model_dir / "full_baselines/output_contracts.json", recorded)
    source = tmp_path / "synthetic.onnx"
    source.write_bytes(b"synthetic ONNX; first-input parser is mocked")
    cached = tmp_path / "cache/model.dxnn"
    cached.parent.mkdir()
    cached.write_bytes(b"synthetic DXNN; cache identity and engine are mocked")
    monkeypatch.setattr(binding, "inspect_deepx_environment", lambda **kw: {
        "compiler_ready": False, "runtime_ready": True, "cache_dir": str(cached.parent),
    })
    monkeypatch.setattr(binding, "_onnx_first_input_info", lambda path: ("images", [1, 3, 640, 640]))
    monkeypatch.setattr(binding, "_lookup_full_deepx_cache_candidate", lambda **kw: cached)
    monkeypatch.setattr(binding, "deepx_cached_artifact_identity_compatible", lambda **kw: (True, "", {}))
    result = binding.materialize_deepx_build_binding(
        run_dir=run_root, model_id="yolo11l", model_path=str(source),
        row={"task": "detection", "input_shape": [1, 3, 640, 640]},
        profile_payload={"deepx_build": {"mode": "reuse_only", "cache_dir": str(cached.parent)}},
        targets=["deepx_m1"], benchmark_set_contract={"suite_dir": str(root)},
    )
    assert result["status"] == "ok", result
    contract = json.loads((root / "deepx/deepx_m1/full/output_contract.json").read_text())
    return SimpleNamespace(root=root, model_dir=model_dir, contract=contract, cached=cached, result=result)


def test_actual_contract_is_rebound_from_cache_without_recompilation(case):
    suite = _suite()
    attestation = binding.recorded_deepx_full_endpoint_attestation(
        model_dir=case.model_dir, suite_dir=case.root, model_id="yolo11l",
    )
    remote = suite._deepx_authoritative_endpoint_attestation(case.root, {"benchmark_task": "detection"})
    assert attestation["pass"] is True
    assert attestation["authoritative_contract"] == remote["authoritative_contract"]
    assert attestation["authoritative_contract_sha256"] == remote["authoritative_contract_sha256"]
    assert attestation["source_endpoint_semantics"] == "decoded_pre_nms"
    assert attestation["source_endpoint_has_integrated_nms"] is False
    assert case.contract["contract_family"] == "decoded_pre_nms"
    assert case.contract["postprocessing"]["host_required"] is True
    bound = suite._deepx_bind_authoritative_endpoint_contract(
        case.root, {"benchmark_task": "detection"}, case.contract,
    )
    assert bound["endpoint_contract_binding_status"] == "attested"
    assert bound["contract_family"] == "decoded_pre_nms"
    assert bound["postprocessing"]["nms_on_host"] is True
    assert suite._deepx_contract_model_id(case.root, {}, bound) == "yolo11l"
    status = json.loads(Path(case.result["artifacts"]["deepx_artifact_status_json"]).read_text())
    assert status["cache_lookup"]["outcome"] == "HIT"
    assert case.cached.read_bytes().startswith(b"synthetic DXNN")


def test_original_failed_artifact_is_not_silently_promoted(case):
    old = json.loads((FIXTURES / "yolo11l_v27925_failed_artifact_contract.json").read_text())
    suite = _suite()
    bound = suite._deepx_bind_authoritative_endpoint_contract(case.root, {"benchmark_task": "detection"}, old)
    assert bound["endpoint_contract_binding_status"] == "conflict"
    assert "endpoint_mode" in bound["endpoint_contract_binding_conflicts"]
    assert suite._deepx_contract_model_id(case.root, {}, bound) == ""


@pytest.mark.parametrize("field,value", [
    ("endpoint_mode", "decoded"), ("host_tail_required", False),
    ("postprocessing_required", False), ("model_id", "yolo26m"),
])
def test_remote_binding_still_rejects_genuine_conflicts(case, field, value):
    contract = copy.deepcopy(case.contract)
    contract[field] = value
    bound = _suite()._deepx_bind_authoritative_endpoint_contract(case.root, {"benchmark_task": "detection"}, contract)
    assert bound["endpoint_contract_binding_status"] == "conflict"
    assert field in bound["endpoint_contract_binding_conflicts"]


@pytest.mark.parametrize("field", ["host_tail_required", "postprocessing_required"])
def test_local_pre_nms_requires_host_nms(case, field):
    recorded = _recorded()
    next(row for row in recorded["contracts"] if row["backend"] == "deepx_m1")[field] = False
    _write_json(case.root / "output_contracts.json", recorded)
    result = binding.recorded_deepx_full_endpoint_attestation(model_dir=case.model_dir, suite_dir=case.root, model_id="yolo11l")
    assert result["pass"] is False
    assert result["reason"] == "recorded_deepx_full_endpoint_contract_invalid"


def test_local_disagreeing_recorded_copies_stay_rejected(case):
    recorded = _recorded()
    row = next(row for row in recorded["contracts"] if row["backend"] == "deepx_m1")
    row.update(endpoint_mode="decoded", host_tail_required=False, postprocessing_required=False)
    _write_json(case.model_dir / "full_baselines/output_contracts.json", recorded)
    result = binding.recorded_deepx_full_endpoint_attestation(model_dir=case.model_dir, suite_dir=case.root, model_id="yolo11l")
    assert result["pass"] is False
    assert result["reason"] == "recorded_deepx_full_endpoint_contract_copies_disagree"


@pytest.mark.parametrize("model", ["yolo11l", "yolo26m", "yolov7_paper"])
def test_explicit_decoder_identity_resolved_under_numeric_transport_path(tmp_path, model):
    suite = _suite()
    assert suite._deepx_contract_model_id(tmp_path / "run/1/suite", {}, {"model_id": model}) == model
    assert suite._deepx_contract_model_id(tmp_path / model / "1/suite", {"model": "1"}, {}) == ""


def test_yolo11_model_family_conflict_stays_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="model_identity_conflict"):
        _suite()._deepx_contract_model_id(tmp_path, {"model_id": "yolo26m"}, {"model_id": "yolo11l"})


def _decoded_outputs():
    # Two overlapping candidates of class 0, one overlapping different class,
    # and one low-confidence candidate. Host NMS must leave exactly two boxes.
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, :4, :4] = np.array([[100, 101, 100, 300], [100, 101, 100, 300], [40, 40, 40, 10], [40, 40, 40, 10]])
    output[0, 4, :2] = [.9, .8]
    output[0, 5, 2] = .7
    output[0, 6, 3] = .1
    return output


def _install_runtime(monkeypatch, output):
    count = []
    class Engine:
        def __init__(self, path):
            assert Path(path).is_file()
        def run(self, feeds):
            assert feeds[0].shape == (640, 640, 3)
            assert feeds[0].dtype == np.uint8
            count.append(1)
            return [value.copy() for value in output] if isinstance(output, list) else [output.copy()]
    dx = types.ModuleType("dx_engine")
    dx.InferenceEngine = Engine
    cv = types.ModuleType("cv2")
    cv.imread = lambda path: np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()
    package = types.ModuleType("splitpoint_runners")
    package.__path__ = []
    for name, module in {
        "dx_engine": dx, "cv2": cv, "splitpoint_runners": package,
        "splitpoint_runners.native_detection_postprocess": postprocess,
        "splitpoint_runners.native_output_endpoint": endpoint,
        "splitpoint_runners.native_full_input": full_input,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return count


def test_whole_prepared_feed_uses_existing_decoded_nms_completion(case, monkeypatch):
    suite = _suite()
    count = _install_runtime(monkeypatch, _decoded_outputs())
    image = case.root / "image.png"
    Image.new("RGB", (640, 640), "black").save(image)
    results = case.root / "results"
    results.mkdir()
    monkeypatch.setattr(suite, "_deepx_find_prepared_feed_image", lambda *a: (image, "fixture"))
    result = suite._run_deepx_prepared_feed_benchmark(
        case.root, case.cached,
        {"benchmark_task": "detection", "model_id": "yolo11l", "setup_id": "test_deepx"},
        SimpleNamespace(runs=3, warmup=1, energy_measurement_only=False,
                        prepared_input_manifest="", quality_evidence_model_id="yolo11l"),
        results,
    )
    assert result["status"] == "ok", result
    assert len(count) == 5  # 1 untimed probe + 1 warmup + 3 timed completed tasks
    assert result["completed_frames"] == result["postprocess_completed_frames"] == 3
    assert result["runtime_endpoint_contract_family"] == "decoded_pre_nms"
    assert result["runtime_endpoint_contract_complete"] is True
    assert result["prepared_input_binding_verified"] is True
    assert Path(result["prepared_input_manifest"]).is_file()
    frozen = result["frozen_host_postprocess_contract"]
    assert frozen["decoder_id"] == "ultralytics_decoded_classaware_nms_v1"
    assert frozen["source_contract_family"] == "decoded_pre_nms"
    assert result["completed_task_endpoint_attestation"]["attested"] is True
    assert result["completed_task_result_artifact_verification_status"] == "verified_exact"
    saved = json.loads(Path(result["completed_task_result_artifact_path"]).read_text())
    assert len(saved["detections"]) == 2, saved
    assert {row["class_id"] for row in saved["detections"]} == {0, 1}


def test_quality_decode_uses_same_frozen_pre_nms_completion(case, monkeypatch):
    suite = _suite()
    _install_runtime(monkeypatch, _decoded_outputs())
    bound = suite._deepx_bind_authoritative_endpoint_contract(case.root, {"benchmark_task": "detection"}, case.contract)
    detections, result = suite._deepx_detection_decode(
        root=case.root, run={"model_id": "yolo11l"}, contract=bound,
        outputs=[_decoded_outputs()], orig_shape=(640, 640, 3), scale=1., pad_x=0, pad_y=0,
    )
    assert result["pass"] is True, result
    assert result["decoder_id"] == "ultralytics_decoded_classaware_nms_v1"
    assert result["source_endpoint_semantics"] == "decoded_pre_nms"
    assert result["source_endpoint_has_integrated_nms"] is False
    assert result["host_nms_applied"] is True
    assert len(detections) == 2
    assert {row["class_id"] for row in detections} == {0, 1}


def test_pre_nms_declaration_does_not_accept_final_bn6_shape(case, monkeypatch):
    suite = _suite()
    _install_runtime(monkeypatch, np.zeros((1, 2, 6), dtype=np.float32))
    bound = suite._deepx_bind_authoritative_endpoint_contract(case.root, {"benchmark_task": "detection"}, case.contract)
    detections, result = suite._deepx_detection_decode(
        root=case.root, run={"model_id": "yolo11l"}, contract=bound,
        outputs=[np.zeros((1, 2, 6), dtype=np.float32)], orig_shape=(640, 640, 3), scale=1., pad_x=0, pad_y=0,
    )
    assert not detections
    assert result["pass"] is False


@pytest.mark.parametrize("model", ["yolo26m", "yolov7_paper"])
def test_whole_prepared_raw_models_keep_their_existing_decoder(case, monkeypatch, model):
    suite = _suite()
    if model == "yolo26m":
        outputs = []
        for side in (80, 40, 20):
            outputs.extend([
                np.zeros((side, side, 4), dtype=np.float32),
                np.full((side, side, 80), -20., dtype=np.float32),
            ])
        expected_decoder = "yolo26_regcls_ltrb_classaware_nms_v1"
    else:
        outputs = [np.full((1, 3, side, side, 85), -20., dtype=np.float32) for side in (80, 40, 20)]
        from onnx_splitpoint_tool.runners.harness.yolo import YOLOV7_PAPER_DECODER_ID
        expected_decoder = YOLOV7_PAPER_DECODER_ID
    count = _install_runtime(monkeypatch, outputs)
    recorded = {
        "schema": "onnx-splitpoint/output-contracts", "schema_version": 1,
        "model_id": model, "task": "detection",
        "contracts": [{
            "model_id": model, "backend": "deepx_m1", "variant": "full", "task": "detection",
            "endpoint_mode": "raw_detection_head", "contract_status": "recorded",
            "host_tail_required": True, "postprocessing_required": True,
        }],
    }
    _write_json(case.root / "output_contracts.json", recorded)
    contract = copy.deepcopy(case.contract)
    contract.update(model_id=model, endpoint_mode="raw_detection_head", contract_family="raw_head")
    for key in ("endpoint_semantic_attestation", "source_onnx_sha256", "build_onnx_sha256"):
        contract.pop(key, None)
    contract["postprocessing"] = {"type": "yolo_host_decode", "host_required": True, "nms_on_host": True}
    _write_json(case.root / "deepx/deepx_m1/full/output_contract.json", contract)
    bound = suite._deepx_bind_authoritative_endpoint_contract(case.root, {"benchmark_task": "detection"}, contract)
    assert bound["endpoint_contract_binding_status"] == "attested"
    monkeypatch.setattr(suite, "_deepx_input_size_from_contract", lambda *a, **k: (640, bound))
    image = case.root / "raw_input.png"
    Image.new("RGB", (640, 640), "black").save(image)
    monkeypatch.setattr(suite, "_deepx_find_prepared_feed_image", lambda *a: (image, "fixture"))
    result = suite._run_deepx_prepared_feed_benchmark(
        case.root, case.cached, {"benchmark_task": "detection", "model_id": model},
        SimpleNamespace(runs=2, warmup=1, energy_measurement_only=False, prepared_input_manifest=""),
        case.root / "raw_results",
    )
    assert result["status"] == "ok", result
    assert len(count) == 4
    assert result["completed_frames"] == result["postprocess_completed_frames"] == 2
    assert result["runtime_endpoint_contract_family"] == "raw_head"
    assert result["frozen_host_postprocess_contract"]["decoder_id"] == expected_decoder
    assert result["completed_task_endpoint_attestation"]["attested"] is True
