from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from onnx_splitpoint_tool.quality_cache import (
    canonical_json,
    image_ids_fingerprint,
    json_fingerprint,
)
from onnx_splitpoint_tool.quality_service import (
    QualityArtifactIntegrityError,
    _validate_candidate_execution_contract,
    quality_request_from_manifest,
)


SHA_RUNNER = hashlib.sha256(b"suite-runner-v269d").hexdigest()
SHA_ATTESTOR = hashlib.sha256(b"endpoint-attestor-v269d").hexdigest()
SHA_MANIFEST = hashlib.sha256(b"classification-manifest").hexdigest()
SHA_SOURCE = hashlib.sha256(b"full-source-onnx").hexdigest()
SHA_ENGINE = hashlib.sha256(b"setup-local-full-fp16-engine").hexdigest()
SHA_TRTEXEC = hashlib.sha256(b"setup-local-trtexec").hexdigest()
SHA_POLICY = hashlib.sha256(b"policy").hexdigest()
SHA_SOURCE_CONTRACTS = hashlib.sha256(b"suite-output-contracts-container").hexdigest()
SHA_RECORDED_CONTRACT = hashlib.sha256(b"recorded-full-endpoint-row").hexdigest()


def _component(identity: dict[str, Any]) -> dict[str, Any]:
    return {"identity": identity, "sha256": json_fingerprint(identity)}


def _endpoint_authority(endpoint: dict[str, Any]) -> dict[str, Any]:
    endpoint_sha = str(endpoint["sha256"])
    endpoint_stage = str(endpoint["identity"]["stage"])
    return _component({
        "schema": "onnx-splitpoint/tensorrt-endpoint-authority",
        "schema_version": 1,
        "graph_binding_source": (
            "authoritative_suite_output_contract_plus_exact_onnx_endpoint:v2"
        ),
        "source_contracts_sha256": SHA_SOURCE_CONTRACTS,
        "recorded_contract_sha256": SHA_RECORDED_CONTRACT,
        "full_model_sha256": SHA_SOURCE,
        "terminal_model_sha256": SHA_SOURCE,
        "endpoint_contract_hash": endpoint_sha,
        "endpoint_contract_complete": True,
        "contract_resolution_status": "attested",
        "stage": endpoint_stage,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "stage": endpoint_stage,
            "endpoint_contract_hash": endpoint_sha,
        },
    })


def _quality_contract() -> dict[str, Any]:
    preprocessing = _component({
        "schema": "onnx-splitpoint/classification-preprocessing-contract",
        "schema_version": 1,
        "image_scale": "imagenet",
        "letterbox": False,
        "input_hw": [224, 224],
        "input_dtype": "float32",
    })
    postprocessor = _component({
        "schema": "onnx-splitpoint/classification-topk-postprocessor-contract",
        "schema_version": 1,
        "implementation_runner_sha256": SHA_RUNNER,
        "implementation_functions": [
            "_classification_logits_vector", "_classification_gt_metrics",
        ],
        "canonical_record_endpoint": "classification_topk_hits",
        "topk": 5,
        "sort_order": "score_descending",
    })
    quality_record = _component({
        "schema": (
            "onnx-splitpoint/"
            "classification-quality-record-endpoint-contract"
        ),
        "schema_version": 1,
        "canonical_record_endpoint": "classification_topk_hits",
        "postprocessor_contract_sha256": postprocessor["sha256"],
        "implementation_runner_sha256": SHA_RUNNER,
        "vendored_endpoint_attestor_sha256": SHA_ATTESTOR,
    })
    ground_truth = [{
        "image_id": "a.jpg", "label_id": 3, "label_name": "three",
    }]
    contract = {
        "schema": "onnx-splitpoint/central-classification-quality-contract",
        "schema_version": 1,
        "task": "classification",
        "model": {"artifact_name": "source.onnx", "sha256": SHA_SOURCE},
        "dataset": {
            "manifest_name": "manifest.json",
            "manifest_sha256": SHA_MANIFEST,
            "image_ids_sha256": image_ids_fingerprint(["a.jpg"]),
            "ground_truth_sha256": json_fingerprint(ground_truth),
            "image_count": 1,
        },
        "preprocessing": preprocessing,
        "postprocessor": postprocessor,
        "quality_record_endpoint": quality_record,
        "quality_record_endpoint_contract_sha256": quality_record[
            "sha256"
        ],
        "contract_scope": "canonical_quality_record_semantics",
        "canonical_record_endpoint": "classification_topk_hits",
    }
    contract["quality_contract_sha256"] = json_fingerprint(contract)
    return contract


def _detection_quality_contract() -> dict[str, Any]:
    preprocessing = _component({
        "schema": "onnx-splitpoint/detection-preprocessing-contract",
        "schema_version": 1,
        "image_scale": "norm",
        "letterbox": True,
        "input_hw": [640, 640],
        "input_dtype": "float32",
    })
    decoder = _component({
        "schema": "onnx-splitpoint/detection-decoder-contract",
        "schema_version": 1,
        "adapter_id": "vendored_yolo_harness_with_local_fallback_v1",
        "adapter_location": "quality_record_host_postprocessor",
        "implementation_runner_sha256": SHA_RUNNER,
        "source_output_format": "multiscale_head",
        "source_output_names": ["p3", "p4", "p5"],
        "source_endpoint_semantics": "raw_multiscale_head",
        "source_endpoint_has_integrated_nms": False,
        "canonical_record_endpoint": "decoded_xyxy_score_class_detections",
        "confidence_threshold": 0.25,
        "labels_sha256": hashlib.sha256(b"labels").hexdigest(),
    })
    nms = _component({
        "schema": "onnx-splitpoint/detection-nms-contract",
        "schema_version": 1,
        "adapter_id": "vendored_yolo_harness_with_local_fallback_v1",
        "implementation_runner_sha256": SHA_RUNNER,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "detr_or_bn6_confidence_threshold": 0.25,
        "detr_or_bn6_iou_threshold": 0.45,
        "detr_or_bn6_max_detections": 300,
    })
    quality_record = _component({
        "schema": (
            "onnx-splitpoint/"
            "detection-quality-record-endpoint-contract"
        ),
        "schema_version": 1,
        "canonical_record_endpoint": (
            "decoded_xyxy_score_class_detections"
        ),
        "decoder_contract_sha256": decoder["sha256"],
        "nms_contract_sha256": nms["sha256"],
        "implementation_runner_sha256": SHA_RUNNER,
        "vendored_endpoint_attestor_sha256": SHA_ATTESTOR,
    })
    ground_truth = [{
        "image_id": "a.jpg",
        "ground_truth": [{
            "x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0,
            "score": 1.0, "class_id": 0, "class_name": "person",
        }],
    }]
    contract = {
        "schema": "onnx-splitpoint/central-detection-quality-contract",
        "schema_version": 1,
        "task": "detection",
        "model": {"artifact_name": "source.onnx", "sha256": SHA_SOURCE},
        "dataset": {
            "manifest_name": "manifest.json",
            "manifest_sha256": SHA_MANIFEST,
            "image_ids_sha256": image_ids_fingerprint(["a.jpg"]),
            "ground_truth_sha256": json_fingerprint(ground_truth),
            "image_count": 1,
        },
        "preprocessing": preprocessing,
        "decoder": decoder,
        "nms": nms,
        "quality_record_endpoint": quality_record,
        "quality_record_endpoint_contract_sha256": quality_record[
            "sha256"
        ],
        "source_endpoint_role": "canonical_reference_model_output",
        "source_endpoint_is_raw": True,
        "contract_scope": "canonical_quality_record_semantics",
        "canonical_record_endpoint": "decoded_xyxy_score_class_detections",
    }
    contract["quality_contract_sha256"] = json_fingerprint(contract)
    return contract


def _seal_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt = copy.deepcopy(receipt)
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = json_fingerprint(receipt)
    return receipt


def _receipt_binding(receipt: dict[str, Any], *, path: str) -> dict[str, Any]:
    receipt = _seal_receipt(receipt)
    return {
        "path": path,
        "sha256": json_fingerprint(receipt),
        "size_bytes": len(canonical_json(receipt).encode("utf-8")),
        "receipt": receipt,
    }


def _seal_producer(producer: dict[str, Any]) -> dict[str, Any]:
    producer = copy.deepcopy(producer)
    producer.pop("producer_identity_sha256", None)
    producer["producer_identity_sha256"] = json_fingerprint(producer)
    return producer


def _producer() -> dict[str, Any]:
    source = {
        "path": "/remote/eval/models/source.onnx",
        "sha256": SHA_SOURCE,
        "size_bytes": len(b"full-source-onnx"),
    }
    build = {
        "path": "/remote/cache/full/source.onnx",
        "sha256": SHA_SOURCE,
        "size_bytes": len(b"full-source-onnx"),
        "source_onnx_sha256": SHA_SOURCE,
    }
    engine = {
        "path": "/remote/cache/full/full_fp16.engine",
        "sha256": SHA_ENGINE,
        "size_bytes": len(b"setup-local-full-fp16-engine"),
        "source_onnx_sha256": SHA_SOURCE,
        "build_onnx_sha256": SHA_SOURCE,
    }
    trtexec = {
        "path": "/usr/src/tensorrt/bin/trtexec",
        "sha256": SHA_TRTEXEC,
        "size_bytes": len(b"setup-local-trtexec"),
    }
    receipt = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1,
        "build_returncode": 0,
        "dry_run": False,
        "command": [
            trtexec["path"], f"--onnx={build['path']}",
            f"--saveEngine={engine['path']}", "--fp16",
            "--memPoolSize=workspace:4096",
        ],
        "source_onnx": build["path"],
        "source_onnx_sha256": build["sha256"],
        "engine": engine["path"],
        "engine_sha256": engine["sha256"],
        "trtexec": trtexec["path"],
        "trtexec_sha256": trtexec["sha256"],
    }
    receipt_binding = _receipt_binding(
        receipt, path="/remote/cache/full/engine_build_receipt.json",
    )
    receipt_file_sha = hashlib.sha256(b"persisted-receipt-file").hexdigest()
    quality = _quality_contract()
    endpoint = _component({
        "schema": "onnx-splitpoint/output-endpoint-contract",
        "schema_version": 3,
        "task": "classification",
        "stage": "classification_logits",
        "tensor_signature": {
            "outputs": [{"name": "logits", "dtype": "float32", "shape": [1, 1000]}],
        },
    })
    precision = _component({
        "schema": "onnx-splitpoint/tensorrt-runtime-precision-contract",
        "schema_version": 1,
        "runtime_precision_identity": "fp16",
        "source_onnx_sha256": SHA_SOURCE,
        "build_onnx_sha256": SHA_SOURCE,
        "engine_sha256": SHA_ENGINE,
    })
    quality_record = copy.deepcopy(quality["quality_record_endpoint"])
    endpoint_attestor_identity = {
        "schema": "onnx-splitpoint/vendored-endpoint-attestor-identity",
        "schema_version": 1,
        "source": "suite_vendored",
        "sha256": SHA_ATTESTOR,
        "expected_sha256": SHA_ATTESTOR,
    }
    endpoint_attestor = {
        **_component(endpoint_attestor_identity),
        "path": "/remote/suite/vendored_endpoint_attestor.py",
    }
    return _seal_producer({
        "schema": "onnx-splitpoint/tensorrt-central-quality-producer-identity",
        "schema_version": 1,
        "execution_role": "full_quality_only",
        "backend": "native_tensorrt",
        "variant": "full",
        "case_id": "full",
        "task": "classification",
        "eval_run_id": "eval-20260721",
        "model_id": "resnet50",
        "setup_id": "orin_nx_hailo8_01",
        "source_run_id": "hailo8_to_trt",
        "source_onnx": source,
        "build_onnx": build,
        "engine": engine,
        "trtexec": trtexec,
        "engine_build_receipt": receipt_binding,
        "engine_build_receipt_file_sha256": receipt_file_sha,
        "model": {
            "source_onnx_sha256": SHA_SOURCE,
            "source_onnx_size_bytes": source["size_bytes"],
            "build_onnx_sha256": SHA_SOURCE,
            "build_onnx_size_bytes": build["size_bytes"],
            "runtime_artifact_sha256": SHA_ENGINE,
            "runtime_artifact_size_bytes": engine["size_bytes"],
        },
        "dataset": copy.deepcopy(quality["dataset"]),
        "preprocessing": copy.deepcopy(quality["preprocessing"]),
        "endpoint": endpoint,
        "endpoint_authority": _endpoint_authority(endpoint),
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint["sha256"],
        "precision": precision,
        "runtime_precision_identity": "fp16",
        "quality_record_endpoint": quality_record,
        "quality_record_endpoint_contract_sha256": quality_record["sha256"],
        "endpoint_attestor": endpoint_attestor,
        "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"],
        "preprocessing_contract_sha256": quality["preprocessing"]["sha256"],
        "decoder_contract_sha256": "",
        "nms_contract_sha256": "",
        "implementation_runner_sha256": SHA_RUNNER,
        "vendored_endpoint_attestor_sha256": SHA_ATTESTOR,
        "policy_sha256": SHA_POLICY,
        "performance_claims_emitted": False,
    })


def _detection_producer() -> dict[str, Any]:
    producer = copy.deepcopy(_producer())
    quality = _detection_quality_contract()
    endpoint = _component({
        "schema": "onnx-splitpoint/output-endpoint-contract",
        "schema_version": 3,
        "task": "detection",
        "stage": "raw_head",
        "tensor_signature": {
            "outputs": [
                {"name": "p3", "dtype": "float32", "shape": [1, 84, 80, 80]},
                {"name": "p4", "dtype": "float32", "shape": [1, 84, 40, 40]},
                {"name": "p5", "dtype": "float32", "shape": [1, 84, 20, 20]},
            ],
        },
    })
    quality_record = copy.deepcopy(quality["quality_record_endpoint"])
    producer.update({
        "task": "detection",
        "model_id": "yolo26s",
        "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"],
        "dataset": copy.deepcopy(quality["dataset"]),
        "preprocessing": copy.deepcopy(quality["preprocessing"]),
        "preprocessing_contract_sha256": quality["preprocessing"]["sha256"],
        "decoder_contract_sha256": quality["decoder"]["sha256"],
        "nms_contract_sha256": quality["nms"]["sha256"],
        "endpoint": endpoint,
        "endpoint_authority": _endpoint_authority(endpoint),
        "endpoint_contract_hash": endpoint["sha256"],
        "quality_record_endpoint": quality_record,
        "quality_record_endpoint_contract_sha256": quality_record["sha256"],
    })
    return _seal_producer(producer)


def _producer_duplicates(producer: dict[str, Any]) -> dict[str, Any]:
    receipt_binding = producer["engine_build_receipt"]
    return {
        "eval_run_id": producer["eval_run_id"],
        "model_id": producer["model_id"],
        "setup_id": producer["setup_id"],
        "source_run_id": producer["source_run_id"],
        "execution_role": producer["execution_role"],
        "backend": producer["backend"],
        "variant": producer["variant"],
        "case_id": producer["case_id"],
        "task": producer["task"],
        "source_model_sha256": producer["source_onnx"]["sha256"],
        "build_onnx_sha256": producer["build_onnx"]["sha256"],
        "runtime_artifact_sha256": producer["engine"]["sha256"],
        "trtexec_sha256": producer["trtexec"]["sha256"],
        "engine_build_receipt_sha256": receipt_binding["sha256"],
        "engine_build_receipt_file_sha256": producer[
            "engine_build_receipt_file_sha256"
        ],
        "trt_engine_build_receipt_sha256": receipt_binding["receipt"]["receipt_sha256"],
        "quality_contract": producer["quality_contract"],
        "quality_contract_sha256": producer["quality_contract_sha256"],
        "preprocessing_contract_sha256": producer["preprocessing_contract_sha256"],
        "decoder_contract_sha256": producer["decoder_contract_sha256"],
        "nms_contract_sha256": producer["nms_contract_sha256"],
        "quality_record_endpoint_contract_sha256": producer[
            "quality_record_endpoint_contract_sha256"
        ],
        "endpoint_contract_hash": producer["endpoint_contract_hash"],
        "runtime_precision_identity": producer["runtime_precision_identity"],
    }


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = canonical_json(payload).encode("utf-8")
    path.write_bytes(encoded)
    return {
        "path": path.name, "size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _write_request_bundle(tmp_path: Path, producer: dict[str, Any]) -> tuple[Path, Path]:
    quality = producer["quality_contract"]
    prediction = {
        "top1": 3, "top1_hit": True,
        "top5": [3, 2, 1, 4, 5], "top5_hit": True,
    }
    reference = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "classification",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "provenance_required": True,
        "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"],
        "records": [{
            "image_id": "a.jpg", "label_id": 3, "label_name": "three",
            "reference": prediction,
        }],
    }
    candidate = {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "pairing_key": "image_id",
        "producer_provenance_required": True,
        "producer_identity": producer,
        "producer_identity_sha256": producer["producer_identity_sha256"],
        **_producer_duplicates(producer),
        "records": [{
            "image_id": "a.jpg", "label_id": 3, "label_name": "three",
            "candidate": prediction,
        }],
    }
    reference_descriptor = _write_json_artifact(tmp_path / "reference.json", reference)
    candidate_descriptor = _write_json_artifact(tmp_path / "candidate.json", candidate)
    reference_descriptor["quality_contract_sha256"] = quality["quality_contract_sha256"]
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "status": "pending_central_evaluation",
        "pairing_key": "image_id",
        "execution_location": "management_node",
        "producer_provenance_required": True,
        "producer_identity": producer,
        "producer_identity_sha256": producer["producer_identity_sha256"],
        **_producer_duplicates(producer),
        "reference": {
            **reference_descriptor,
            "expected_image_ids": ["a.jpg"],
            "expected_image_ids_sha256": image_ids_fingerprint(["a.jpg"]),
        },
        "candidate": candidate_descriptor,
        "record_count": 1,
        "reference_record_count": 1,
        "policy_sha256": SHA_POLICY,
        "metric_gate_config": {
            "primary_metric": "top1_accuracy", "non_inferiority_margin": 0.01,
        },
        "statistics": {
            "method": "paired_bootstrap", "bootstrap_repetitions": 5,
            "seed": 7, "confidence_level": 0.95,
            "decision": "lower_one_sided_bound",
        },
    }
    request_path = tmp_path / "full_request.json"
    request_path.write_text(canonical_json(request), encoding="utf-8")
    return request_path, tmp_path / "candidate.json"


def _write_detection_request_bundle(
    tmp_path: Path,
    producer: dict[str, Any],
    *,
    reference_quality: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    quality = producer["quality_contract"]
    reference_quality = copy.deepcopy(reference_quality or quality)
    ground_truth = [{
        "x1": 10.0, "y1": 20.0, "x2": 40.0, "y2": 60.0,
        "score": 1.0, "class_id": 0, "class_name": "person",
    }]
    prediction = [{
        "x1": 11.0, "y1": 20.0, "x2": 39.0, "y2": 59.0,
        "score": 0.9, "class_id": 0, "class_name": "person",
    }]
    reference = {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "pairing_key": "image_id",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "provenance_required": True,
        "quality_contract": reference_quality,
        "quality_contract_sha256": reference_quality["quality_contract_sha256"],
        "records": [{
            "image_id": "a.jpg", "ground_truth": ground_truth,
            "reference": prediction,
        }],
    }
    candidate = {
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "pairing_key": "image_id",
        "producer_provenance_required": True,
        "producer_identity": producer,
        "producer_identity_sha256": producer["producer_identity_sha256"],
        **_producer_duplicates(producer),
        "records": [{
            "image_id": "a.jpg", "ground_truth": ground_truth,
            "candidate": prediction,
        }],
    }
    reference_descriptor = _write_json_artifact(
        tmp_path / "reference.json", reference,
    )
    candidate_descriptor = _write_json_artifact(
        tmp_path / "candidate.json", candidate,
    )
    reference_descriptor["quality_contract_sha256"] = reference_quality[
        "quality_contract_sha256"
    ]
    request = {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1,
        "status": "pending_central_evaluation",
        "pairing_key": "image_id",
        "execution_location": "management_node",
        "producer_provenance_required": True,
        "producer_identity": producer,
        "producer_identity_sha256": producer["producer_identity_sha256"],
        **_producer_duplicates(producer),
        "reference": {
            **reference_descriptor,
            "expected_image_ids": ["a.jpg"],
            "expected_image_ids_sha256": image_ids_fingerprint(["a.jpg"]),
        },
        "candidate": candidate_descriptor,
        "record_count": 1,
        "reference_record_count": 1,
        "policy_sha256": SHA_POLICY,
        "metric_gate_config": {
            "primary_metric": "coco_ap_50_95", "non_inferiority_margin": 0.01,
        },
        "statistics": {
            "method": "paired_bootstrap", "bootstrap_repetitions": 5,
            "seed": 7, "confidence_level": 0.95,
            "decision": "lower_one_sided_bound",
        },
    }
    request_path = tmp_path / "full_request.json"
    request_path.write_text(canonical_json(request), encoding="utf-8")
    return request_path, tmp_path / "candidate.json"


def test_trt_full_quality_producer_loads_without_deepx_transform_evidence(
    tmp_path: Path,
) -> None:
    producer = _producer()
    validated, producer_sha = _validate_candidate_execution_contract(
        producer, role="request", task="classification",
    )
    assert validated == producer
    assert producer_sha == producer["producer_identity_sha256"]

    request_path, _ = _write_request_bundle(tmp_path, producer)
    loaded = quality_request_from_manifest(request_path)

    assert loaded.metric_gate_config["producer_identity_sha256"] == producer_sha
    assert loaded.metric_gate_config["task"] == "classification"
    assert loaded.metric_gate_config["variant"] == "full"

    performance_owner = copy.deepcopy(producer)
    performance_owner["execution_role"] = "full_performance_owner"
    performance_owner = _seal_producer(performance_owner)
    validated_owner, _ = _validate_candidate_execution_contract(
        performance_owner, role="request", task="classification",
    )
    assert validated_owner["execution_role"] == "full_performance_owner"


def test_trt_detection_raw_head_quality_producer_loads_with_decoder_and_nms(
    tmp_path: Path,
) -> None:
    producer = _detection_producer()
    validated, producer_sha = _validate_candidate_execution_contract(
        producer, role="request", task="detection",
    )
    assert validated["endpoint"]["identity"]["stage"] == "raw_head"
    assert validated["decoder_contract_sha256"] == producer[
        "quality_contract"
    ]["decoder"]["sha256"]
    assert validated["nms_contract_sha256"] == producer[
        "quality_contract"
    ]["nms"]["sha256"]

    request_path, _ = _write_detection_request_bundle(tmp_path, producer)
    loaded = quality_request_from_manifest(request_path)

    assert loaded.metric_gate_config["producer_identity_sha256"] == producer_sha
    assert loaded.metric_gate_config["task"] == "detection"
    assert loaded.metric_gate_config["quality_contract_sha256"] == producer[
        "quality_contract_sha256"
    ]


def test_trt_detection_recorded_pre_nms_endpoint_is_supported() -> None:
    producer = _detection_producer()
    producer["endpoint"]["identity"]["stage"] = "decoded_pre_nms"
    producer["endpoint"]["sha256"] = json_fingerprint(
        producer["endpoint"]["identity"]
    )
    producer["endpoint_authority"] = _endpoint_authority(producer["endpoint"])
    producer["endpoint_contract_hash"] = producer["endpoint"]["sha256"]
    producer = _seal_producer(producer)

    validated, _ = _validate_candidate_execution_contract(
        producer, role="request", task="detection",
    )

    assert validated["endpoint"]["identity"]["stage"] == "decoded_pre_nms"


def _mutate_detection_adapter(
    producer: dict[str, Any], component_name: str,
) -> dict[str, Any]:
    quality = producer["quality_contract"]
    component = quality[component_name]
    if component_name == "decoder":
        component["identity"]["confidence_threshold"] = 0.73
    else:
        component["identity"]["iou_threshold"] = 0.73
    component["sha256"] = json_fingerprint(component["identity"])
    quality.pop("quality_contract_sha256", None)
    quality["quality_contract_sha256"] = json_fingerprint(quality)
    producer["quality_contract_sha256"] = quality["quality_contract_sha256"]
    producer[f"{component_name}_contract_sha256"] = component["sha256"]
    producer["quality_record_endpoint"]["identity"][
        f"{component_name}_contract_sha256"
    ] = component["sha256"]
    producer["quality_record_endpoint"]["sha256"] = json_fingerprint(
        producer["quality_record_endpoint"]["identity"]
    )
    producer["quality_record_endpoint_contract_sha256"] = producer[
        "quality_record_endpoint"
    ]["sha256"]
    return _seal_producer(producer)


@pytest.mark.parametrize("component_name", ["decoder", "nms"])
def test_trt_detection_cannot_borrow_quality_with_changed_decoder_or_nms(
    tmp_path: Path, component_name: str,
) -> None:
    canonical_producer = _detection_producer()
    canonical_quality = copy.deepcopy(canonical_producer["quality_contract"])
    changed_producer = _mutate_detection_adapter(
        copy.deepcopy(canonical_producer), component_name,
    )
    request_path, _ = _write_detection_request_bundle(
        tmp_path, changed_producer, reference_quality=canonical_quality,
    )

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="detection quality-record endpoint binding is inconsistent",
    ):
        quality_request_from_manifest(request_path)


def test_trt_outer_quality_endpoint_must_equal_sealed_quality_contract() -> None:
    producer = _detection_producer()
    outer = producer["quality_record_endpoint"]
    outer["identity"]["canonical_record_endpoint"] = "different_endpoint"
    outer["sha256"] = json_fingerprint(outer["identity"])
    producer["quality_record_endpoint_contract_sha256"] = outer["sha256"]
    producer = _seal_producer(producer)

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="quality-record endpoint differs from quality contract",
    ):
        _validate_candidate_execution_contract(
            producer, role="request producer", task="detection",
        )


def test_trt_endpoint_attestor_crosslink_is_fail_closed() -> None:
    producer = _detection_producer()
    attestor = producer["endpoint_attestor"]
    attestor["identity"]["expected_sha256"] = "f" * 64
    attestor["sha256"] = json_fingerprint(attestor["identity"])
    producer = _seal_producer(producer)

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="endpoint-attestor binding is inconsistent",
    ):
        _validate_candidate_execution_contract(
            producer, role="request producer", task="detection",
        )


def _mutate_engine(producer: dict[str, Any]) -> None:
    producer["engine"]["sha256"] = "a" * 64


def _mutate_source_onnx(producer: dict[str, Any]) -> None:
    producer["source_onnx"]["sha256"] = "f" * 64


def _mutate_build_onnx(producer: dict[str, Any]) -> None:
    producer["build_onnx"]["sha256"] = "b" * 64


def _mutate_receipt_command(producer: dict[str, Any]) -> None:
    binding = producer["engine_build_receipt"]
    receipt = binding["receipt"]
    receipt["command"][1] = "--onnx=/remote/cache/full/other.onnx"
    producer["engine_build_receipt"] = _receipt_binding(
        receipt, path=binding["path"],
    )


def _mutate_endpoint(producer: dict[str, Any]) -> None:
    producer["endpoint"]["identity"]["stage"] = "raw_head"
    producer["endpoint"]["sha256"] = json_fingerprint(
        producer["endpoint"]["identity"]
    )
    producer["endpoint_contract_hash"] = producer["endpoint"]["sha256"]


def _mutate_precision(producer: dict[str, Any]) -> None:
    producer["runtime_precision_identity"] = "fp32"
    producer["precision"]["identity"]["runtime_precision_identity"] = "fp32"
    producer["precision"]["sha256"] = json_fingerprint(
        producer["precision"]["identity"]
    )


def _mutate_dataset(producer: dict[str, Any]) -> None:
    producer["dataset"]["manifest_sha256"] = "c" * 64


def _mutate_model_size(producer: dict[str, Any]) -> None:
    producer["model"]["runtime_artifact_size_bytes"] += 1


def _mutate_quality_only_claim_status(producer: dict[str, Any]) -> None:
    producer["performance_claims_emitted"] = True


def _mutate_endpoint_authority_graph(producer: dict[str, Any]) -> None:
    identity = producer["endpoint_authority"]["identity"]
    identity["terminal_model_sha256"] = "e" * 64
    producer["endpoint_authority"] = _component(identity)


def _mutate_endpoint_authority_attestation(producer: dict[str, Any]) -> None:
    identity = producer["endpoint_authority"]["identity"]
    identity["output_endpoint_attestation"]["endpoint_contract_hash"] = "e" * 64
    producer["endpoint_authority"] = _component(identity)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (_mutate_engine, "engine|runtime_artifact"),
        (_mutate_source_onnx, "source model|source_onnx|build ONNX"),
        (_mutate_build_onnx, "engine model binding|build ONNX|build_onnx"),
        (_mutate_receipt_command, "receipt source/engine command mismatch"),
        (_mutate_endpoint, "endpoint stage is invalid"),
        (_mutate_precision, "identity|precision|--fp16"),
        (_mutate_dataset, "dataset differs"),
        (_mutate_model_size, "runtime_artifact_size_bytes binding mismatch"),
        (_mutate_quality_only_claim_status, "quality-only.*performance claims"),
        (_mutate_endpoint_authority_graph, "authority targets a different ONNX graph"),
        (_mutate_endpoint_authority_attestation, "runtime attestation is inconsistent"),
    ],
)
def test_trt_producer_mutations_fail_closed(
    mutator: Callable[[dict[str, Any]], None], match: str,
) -> None:
    producer = _producer()
    mutator(producer)
    producer = _seal_producer(producer)

    with pytest.raises(QualityArtifactIntegrityError, match=match):
        _validate_candidate_execution_contract(
            producer, role="request", task="classification",
        )


def test_trt_request_top_level_engine_binding_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    request_path, _ = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["runtime_artifact_sha256"] = "d" * 64
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="request top-level runtime_artifact_sha256 differs",
    ):
        quality_request_from_manifest(request_path)


def test_trt_request_policy_conflict_fails_closed(tmp_path: Path) -> None:
    request_path, _ = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["policy_sha256"] = "d" * 64
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="request policy SHA-256 differs from TensorRT producer identity",
    ):
        quality_request_from_manifest(request_path)


def test_trt_reference_descriptor_quality_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    request_path, _ = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["reference"]["quality_contract_sha256"] = "d" * 64
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="reference descriptor quality contract differs",
    ):
        quality_request_from_manifest(request_path)


def test_trt_candidate_top_level_receipt_binding_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    request_path, candidate_path = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["engine_build_receipt_sha256"] = "e" * 64
    request["candidate"] = _write_json_artifact(candidate_path, candidate)
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="candidate top-level engine_build_receipt_sha256 differs",
    ):
        quality_request_from_manifest(request_path)


def test_trt_request_receipt_file_hash_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    request_path, _ = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["engine_build_receipt_file_sha256"] = "f" * 64
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="request top-level engine_build_receipt_file_sha256 differs",
    ):
        quality_request_from_manifest(request_path)


def test_trt_request_missing_top_level_full_case_fails_closed(
    tmp_path: Path,
) -> None:
    request_path, _ = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request.pop("case_id")
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="request lacks duplicated TensorRT producer binding case_id",
    ):
        quality_request_from_manifest(request_path)


def test_trt_candidate_top_level_full_case_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    request_path, candidate_path = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["case_id"] = "b044"
    request["candidate"] = _write_json_artifact(candidate_path, candidate)
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="candidate top-level case_id differs from TensorRT producer identity",
    ):
        quality_request_from_manifest(request_path)


def test_trt_request_and_candidate_producers_must_be_identical(
    tmp_path: Path,
) -> None:
    request_path, candidate_path = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    different = copy.deepcopy(candidate["producer_identity"])
    different["eval_run_id"] = "other-eval-run"
    different = _seal_producer(different)
    candidate["producer_identity"] = different
    candidate["producer_identity_sha256"] = different["producer_identity_sha256"]
    candidate.update(_producer_duplicates(different))
    request["candidate"] = _write_json_artifact(candidate_path, candidate)
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="request and candidate use different candidate execution contracts",
    ):
        quality_request_from_manifest(request_path)


def test_trt_request_and_candidate_endpoint_authority_must_be_identical(
    tmp_path: Path,
) -> None:
    request_path, candidate_path = _write_request_bundle(tmp_path, _producer())
    request = json.loads(request_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    different = copy.deepcopy(candidate["producer_identity"])
    authority_identity = different["endpoint_authority"]["identity"]
    authority_identity["source_contracts_sha256"] = hashlib.sha256(
        b"different-suite-output-contracts-container"
    ).hexdigest()
    different["endpoint_authority"] = _component(authority_identity)
    different = _seal_producer(different)
    candidate["producer_identity"] = different
    candidate["producer_identity_sha256"] = different[
        "producer_identity_sha256"
    ]
    request["candidate"] = _write_json_artifact(candidate_path, candidate)
    request_path.write_text(canonical_json(request), encoding="utf-8")

    with pytest.raises(
        QualityArtifactIntegrityError,
        match="request and candidate use different candidate execution contracts",
    ):
        quality_request_from_manifest(request_path)
