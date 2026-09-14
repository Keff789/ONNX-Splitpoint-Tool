from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    FrozenPostprocessError,
    build_frozen_postprocess_contract,
    canonical_json_sha256,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError,
    quality_request_from_manifest,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_DECODER_ID,
    YOLOV7_PAPER_ONNX_SHA256,
)


ROOT = Path(__file__).resolve().parents[1]
GENERIC_RUNNER = ROOT / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
DEEPX_SUITE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


def _yolov7_raw_heads() -> dict[str, np.ndarray]:
    return {
        "output": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
        "clone_1": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
        "clone_2": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
    }


def test_yolov7_frozen_decoder_binds_geometry_but_has_invariant_identity() -> None:
    outputs = _yolov7_raw_heads()
    wide = build_frozen_postprocess_contract(
        model_id="yolov7_paper", outputs=outputs, input_hw=[640, 640],
        original_wh=[1280, 720],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    square = build_frozen_postprocess_contract(
        model_id="yolov7_paper", outputs=outputs, input_hw=[640, 640],
        original_wh=[640, 640],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )

    assert wide["decoder_id"] == YOLOV7_PAPER_DECODER_ID
    assert wide["invariant_contract_sha256"] == square["invariant_contract_sha256"]
    assert wide["contract_sha256"] != square["contract_sha256"]
    assert wide["invariant_identity"] == square["invariant_identity"]

    processor = FrozenDetectionPostprocessor(wide)
    result = processor.process(outputs, original_wh=[1280, 720])
    assert result["contract_family"] == "decoded_nms"
    assert result["postprocess_contract_sha256"] == wide["contract_sha256"]
    assert processor.completed_count == 1

    with pytest.raises(FrozenPostprocessError, match="original_wh_mismatch"):
        processor.process(outputs, original_wh=[640, 640])
    assert processor.completed_count == 1

    tampered = json.loads(json.dumps(wide))
    tampered["original_wh"] = [1920, 1080]
    with pytest.raises(FrozenPostprocessError, match="sha256_mismatch"):
        verify_frozen_postprocess_contract(tampered, outputs=outputs)

    implementation_drift = json.loads(json.dumps(wide))
    implementation_drift["implementation_artifacts"]["yolo_harness_sha256"] = "0" * 64
    implementation_drift["contract_sha256"] = canonical_json_sha256({
        key: value for key, value in implementation_drift.items()
        if key != "contract_sha256"
    })
    with pytest.raises(FrozenPostprocessError, match="implementation_sha256_mismatch"):
        verify_frozen_postprocess_contract(implementation_drift, outputs=outputs)


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function not found: {name}")


def test_canonical_quality_contract_is_backend_independent_and_completion_is_separate(
    tmp_path: Path,
) -> None:
    source = GENERIC_RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    canonical_builder = _function_source(source, "_build_detection_quality_contract")
    exporter = _function_source(source, "_export_central_quality_inputs")

    for backend_token in (
        "hailo", "deepx", "tensorrt", "hef_sha256",
        "candidate_execution_completion_contract",
    ):
        assert backend_token not in canonical_builder.lower()
    assert "candidate_execution_completion_contract" in exporter
    assert '"candidate_execution_completion_contract": completion_contract' in exporter
    assert '"candidate_execution_completion_contract_sha256": completion_contract_sha' in exporter

    def _calls(name: str) -> list[ast.Call]:
        return [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]

    def _keywords(call: ast.Call) -> dict[str, str]:
        return {
            str(keyword.arg): ast.unparse(keyword.value)
            for keyword in call.keywords if keyword.arg
        }

    quality_calls = _calls("_build_detection_quality_contract")
    assert len(quality_calls) == 1
    quality_keywords = _keywords(quality_calls[0])
    assert quality_keywords["output_format"] == "str(output_format or 'unknown')"
    assert quality_keywords["output_names"] == "list(output_names)"

    endpoint_calls = [
        _keywords(call)
        for call in _calls("_central_quality_endpoint_contract")
        if _keywords(call).get("task") == "'detection'"
    ]
    assert len(endpoint_calls) == 1
    assert endpoint_calls[0]["output_names"] == "_quality_output_names"
    assert endpoint_calls[0]["outputs"] == "_quality_output_values"

    export_calls = [
        _keywords(call)
        for call in _calls("_export_central_quality_inputs")
        if _keywords(call).get("task") == "'detection'"
    ]
    assert len(export_calls) == 1
    assert export_calls[0]["quality_contract"] == "detection_quality_contract"
    assert export_calls[0]["endpoint_contract"] == "_quality_endpoint"
    assert export_calls[0]["candidate_completion_contract"] == (
        "runtime_completion_contract"
    )
    assert export_calls[0]["completed_task_evidence"] == (
        "_quality_completed_task_evidence"
    )

    namespace: dict[str, object] = {
        "Path": Path,
        "_quality_contract_sha256": canonical_json_sha256,
        "_quality_file_sha256": lambda path: hashlib.sha256(
            Path(path).read_bytes()
        ).hexdigest(),
        "_portable_dataset_manifest_sha256": lambda path: hashlib.sha256(
            Path(path).read_bytes()
        ).hexdigest(),
    }
    exec("from __future__ import annotations\n" + canonical_builder, namespace)
    build_quality = namespace["_build_detection_quality_contract"]
    model = tmp_path / "yolov7_paper.onnx"
    runner = tmp_path / "run_split_onnxruntime.py"
    validation = tmp_path / "validation"
    validation.mkdir()
    model.write_bytes(b"canonical-source-model")
    runner.write_bytes(b"canonical-quality-runner")
    (validation / "manifest.json").write_text(
        json.dumps({"dataset": "frozen-smoke"}), encoding="utf-8",
    )
    kwargs = {
        "model_path": model,
        "validation_source": validation,
        "image_ids": ["sample.jpg"],
        "ground_truth_by_image": {"sample.jpg": []},
        "image_scale": "norm",
        "letterbox": True,
        "input_hw": (640, 640),
        "input_dtype": "float32",
        "output_format": "multiscale_head",
        "output_names": ["output", "clone_1", "clone_2"],
        "yolo_conf": 0.25,
        "yolo_iou": 0.45,
        "yolo_max_det": 300,
        "det_conf": 0.25,
        "det_iou": 0.45,
        "det_max_det": 300,
        "labels": ["person"],
        "runner_path": runner,
        "endpoint_attestor_sha256": "a" * 64,
    }
    cpu_quality_for_hailo = build_quality(**kwargs)
    cpu_quality_for_deepx = build_quality(**kwargs)
    completion_hailo_sha = canonical_json_sha256({
        "backend": "hailo8", "timing_scope": "raw_head_plus_frozen_decode_nms",
    })
    completion_deepx_sha = canonical_json_sha256({
        "backend": "deepx", "timing_scope": "raw_head_plus_frozen_decode_nms",
    })
    assert completion_hailo_sha != completion_deepx_sha
    assert cpu_quality_for_hailo["quality_contract_sha256"] == (
        cpu_quality_for_deepx["quality_contract_sha256"]
    )
    physical_raw_contract = build_quality(**{
        **kwargs,
        "output_format": "ultralytics_regcls",
        "output_names": [
            "/model.23/one2one_cv2.0/one2one_cv2.0.2/Conv",
            "/model.23/one2one_cv3.0/one2one_cv3.0.2/Conv",
        ],
    })
    assert physical_raw_contract["decoder"]["sha256"] != (
        cpu_quality_for_hailo["decoder"]["sha256"]
    )
    assert physical_raw_contract["quality_contract_sha256"] != (
        cpu_quality_for_hailo["quality_contract_sha256"]
    )


def test_generic_hailo_full_uses_graph_tail_then_frozen_yolov7_fallback() -> None:
    source = GENERIC_RUNNER.read_text(encoding="utf-8")
    assert "_run_full_host_tail_map" in source
    assert "hailo_raw_head_plus_hash_bound_onnx_host_tail" in source
    assert "_run_full_frozen_detection_hotloop" in source
    assert "hailo_raw_head_plus_frozen_decode_nms" in source
    assert "lacks any verified host completion path" in source

    deepx_source = DEEPX_SUITE.read_text(encoding="utf-8")
    assert 'item.get("frozen_postprocess_invariant_contract_sha256")' in deepx_source
    assert '"per_image_original_wh", "frozen_postprocess_contract_sha256"' in deepx_source


def _write_json_artifact(path: Path, payload: dict) -> dict:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": path.name,
        "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _completion_quality_request(tmp_path: Path) -> Path:
    reference_payload = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "classification",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "records": [{
            "image_id": "sample", "label_id": 0,
            "reference": {"top1_hit": True, "top5_hit": True},
        }],
    }
    completion_identity = {
        "schema": "onnx-splitpoint/hailo-full-host-tail-completion",
        "schema_version": 1,
        "source_endpoint": "hailo_raw_detection_head",
        "completed_endpoint": "decoded_xyxy_score_class_detection_records",
        "timing_scope": "hailo_raw_head_plus_frozen_decode_nms",
        "postprocess_included_per_timed_frame": True,
    }
    completion_sha = hashlib.sha256(json.dumps(
        completion_identity, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    completion_contract = {**completion_identity, "contract_sha256": completion_sha}
    candidate_payload = {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "task": "classification",
        "variant": "composed",
        "pairing_key": "image_id",
        "candidate_execution_completion_contract": completion_contract,
        "candidate_execution_completion_contract_sha256": completion_sha,
        "records": [{
            "image_id": "sample", "label_id": 0,
            "candidate": {"top1_hit": True, "top5_hit": True},
        }],
    }
    reference = _write_json_artifact(tmp_path / "reference.json", reference_payload)
    candidate = _write_json_artifact(tmp_path / "candidate.json", candidate_payload)
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "status": "pending_central_evaluation",
        "task": "classification",
        "variant": "composed",
        "pairing_key": "image_id",
        "execution_location": "management_node",
        "requested_by": "central_management",
        "reference": reference,
        "candidate": candidate,
        "candidate_execution_completion_contract": completion_contract,
        "candidate_execution_completion_contract_sha256": completion_sha,
        "record_count": 1,
        "reference_record_count": 1,
        "policy_sha256": "a" * 64,
        "metric_gate_config": {"primary_metric": "top1_accuracy"},
        "statistics": {"bootstrap_repetitions": 25},
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return request_path


def test_management_requires_identical_request_candidate_completion_contracts(
    tmp_path: Path,
) -> None:
    request_path = _completion_quality_request(tmp_path)
    loaded = quality_request_from_manifest(request_path)
    assert len(loaded.candidate_execution_completion_contract_sha256) == 64

    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate_path = tmp_path / request["candidate"]["path"]
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["candidate_execution_completion_contract"]["timing_scope"] = (
        "accelerator_only"
    )
    request["candidate"] = _write_json_artifact(candidate_path, candidate)
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="different execution completion contracts",
    ):
        quality_request_from_manifest(request_path)
