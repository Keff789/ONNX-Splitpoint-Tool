from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import onnx_splitpoint_tool.dataset_provisioning as dataset_provisioning
import onnx_splitpoint_tool.deepx.calibration_size_canary as canary
from onnx_splitpoint_tool.deepx.calibration_size_canary import (
    BASELINE_CALIBRATION_COUNT,
    BASELINE_PROFILE_NAME,
    B1000_CACHE_NAMESPACE,
    B500_CACHE_NAMESPACE,
    BOOTSTRAP_REPETITIONS,
    CANDIDATE_CALIBRATION_COUNT,
    CANDIDATE_PROFILE_NAME,
    EXPECTED_QUALITY_ALGORITHM,
    EXPECTED_QUALITY_POLICY_SHA256,
    CalibrationArmEvidence,
    QualityEndpointEvidence,
    compare_calibration_arms,
)
from onnx_splitpoint_tool.deepx.config import (
    CLASSIFICATION_PREPROCESSING_IMAGENET,
)
from onnx_splitpoint_tool.deepx.preprocessing_ab import ArmEvidence
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
)


SHA = {
    name: hashlib.sha256(name.encode("utf-8")).hexdigest()
    for name in (
        "source", "build", "compiler", "compiler-contract", "validation",
        "image-ids", "ground-truth", "prepared", "preprocessing",
        "dxcom-500", "dxcom-1000", "cache-500", "cache-1000", "dxnn",
        "calibration-500", "calibration-1000",
    )
}


def _records(top1_hits: int, top5_hits: int) -> tuple[dict[str, object], ...]:
    return tuple({
        "image_id": f"val-{index:04d}",
        "label_id": index % 1000,
        "candidate": {
            "top1_hit": index < top1_hits,
            "top5_hit": index < top5_hits,
        },
    } for index in range(500))


_VALIDATION_IDENTITY_RECORDS = _records(0, 0)
_PREPARED_TRANSFORMS = [{
    "image_id": row["image_id"],
    "prepared_input_sha256": hashlib.sha256(
        f"prepared-{row['image_id']}".encode("utf-8")
    ).hexdigest(),
} for row in _VALIDATION_IDENTITY_RECORDS]
SHA["image-ids"] = canary.image_ids_fingerprint([
    row["image_id"] for row in _VALIDATION_IDENTITY_RECORDS
])
SHA["ground-truth"] = canary.json_fingerprint(sorted([{
    "image_id": str(row["image_id"]), "label_id": int(row["label_id"]),
} for row in _VALIDATION_IDENTITY_RECORDS], key=lambda row: row["image_id"]))
SHA["prepared"] = canary.json_fingerprint(_PREPARED_TRANSFORMS)
_PREPROCESSING_IDENTITY = canonical_image_preprocessing_contract(
    "classification", (224, 224),
)
SHA["preprocessing"] = canary._canonical_sha256(_PREPROCESSING_IDENTITY)

_RELEASE_AUTHORITY_SNAPSHOT = {
    name: getattr(canary, name) for name in (
        "EXPECTED_SOURCE_ONNX_SHA256",
        "EXPECTED_BUILD_ONNX_SHA256",
        "EXPECTED_COMPILER_IDENTITY_SHA256",
        "EXPECTED_COMPILER_CONTRACT_SHA256",
        "EXPECTED_VALIDATION_MANIFEST_SHA256",
        "EXPECTED_VALIDATION_IMAGE_IDS_SHA256",
        "EXPECTED_VALIDATION_GROUND_TRUTH_SHA256",
        "EXPECTED_PREPARED_INPUT_EVIDENCE_SHA256",
        "EXPECTED_PREPROCESSING_CONTRACT_SHA256",
        "EXPECTED_B500_CALIBRATION_MANIFEST_SHA256",
        "EXPECTED_B500_CALIBRATION_ITEMS_SHA256",
        "EXPECTED_B500_CALIBRATION_IDENTITY_SHA256",
        "EXPECTED_B500_CALIBRATION_CONTRACT_SHA256",
        "EXPECTED_B500_DXCOM_CONFIG_SHA256",
        "EXPECTED_B500_BUILD_OPTIONS_SHA256",
        "EXPECTED_B500_CACHE_KEY",
        "EXPECTED_B500_CACHE_CONTRACT_SHA256",
        "EXPECTED_B500_DXNN_SHA256",
        "EXPECTED_SETUP_ID",
        "EXPECTED_B500_DEEPX_TOP1_HITS",
        "EXPECTED_B500_DEEPX_TOP5_HITS",
        "EXPECTED_REFERENCE_TOP1_HITS",
        "EXPECTED_REFERENCE_TOP5_HITS",
    )
}


@pytest.fixture(autouse=True)
def _bind_synthetic_fixture_authorities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit evidence synthetic while exercising every release pin."""

    synthetic = {
        "EXPECTED_SOURCE_ONNX_SHA256": SHA["source"],
        "EXPECTED_BUILD_ONNX_SHA256": SHA["build"],
        "EXPECTED_COMPILER_IDENTITY_SHA256": SHA["compiler"],
        "EXPECTED_COMPILER_CONTRACT_SHA256": SHA["compiler-contract"],
        "EXPECTED_VALIDATION_MANIFEST_SHA256": SHA["validation"],
        "EXPECTED_VALIDATION_IMAGE_IDS_SHA256": SHA["image-ids"],
        "EXPECTED_VALIDATION_GROUND_TRUTH_SHA256": SHA["ground-truth"],
        "EXPECTED_PREPARED_INPUT_EVIDENCE_SHA256": SHA["prepared"],
        "EXPECTED_PREPROCESSING_CONTRACT_SHA256": SHA["preprocessing"],
        "EXPECTED_B500_CALIBRATION_ITEMS_SHA256": hashlib.sha256(
            b"calibration-items-500"
        ).hexdigest(),
        "EXPECTED_B500_CALIBRATION_IDENTITY_SHA256": SHA["calibration-500"],
        "EXPECTED_B500_CALIBRATION_CONTRACT_SHA256": hashlib.sha256(
            b"calibration-contract-500"
        ).hexdigest(),
        "EXPECTED_B500_DXCOM_CONFIG_SHA256": SHA["dxcom-500"],
        "EXPECTED_B500_BUILD_OPTIONS_SHA256": hashlib.sha256(
            b"options-500"
        ).hexdigest(),
        "EXPECTED_B500_CACHE_KEY": (
            "deepx_m1_full_imagenet_mean_std_v2_key500"
        ),
        "EXPECTED_B500_CACHE_CONTRACT_SHA256": SHA["cache-500"],
        "EXPECTED_B500_DXNN_SHA256": SHA["dxnn"],
    }
    for name, value in synthetic.items():
        monkeypatch.setattr(canary, name, value)


def _result(
    *, source_run_id: str, backend: str, records: tuple[dict[str, object], ...],
    decision: str = "pass", technical_status: str = "completed",
) -> dict[str, object]:
    top1 = sum(bool(dict(row["candidate"])["top1_hit"]) for row in records)
    top5 = sum(bool(dict(row["candidate"])["top5_hit"]) for row in records)
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "status": "completed",
        "technical_status": technical_status,
        "decision": decision,
        "source_run_id": source_run_id,
        "case_id": "full",
        "variant": "full",
        "task": "classification",
        "execution_role": "full_quality_only",
        "backend": backend,
        "algorithm_version": EXPECTED_QUALITY_ALGORITHM,
        "source_setup_id": "orin_nx_deepx_m1_01",
        "performance_claims_emitted": False,
        "n": 500,
        "primary": {
            "metric": "top1_accuracy", "decision": decision,
            "margin": 0.01, "sample_count": 500,
            "candidate_hits": top1,
            "reference_hits": 406,
            "bootstrap_repetitions_requested": 500,
        },
        "guardrails": {
            "top5_accuracy": {
                "metric": "top5_accuracy", "decision": decision,
                "margin": 0.01, "sample_count": 500,
                "candidate_hits": top5,
                "reference_hits": 476,
                "bootstrap_repetitions_requested": 500,
            },
        },
    }


def _with_result_decision(
    result: dict[str, object], decision: str,
) -> dict[str, object]:
    changed = json.loads(json.dumps(result))
    changed["decision"] = decision
    changed["primary"]["decision"] = decision
    changed["guardrails"]["top5_accuracy"]["decision"] = decision
    return changed


def _sealed_component(identity: dict[str, object]) -> dict[str, object]:
    return {
        "identity": identity,
        "sha256": canary._canonical_sha256(identity),
    }


def _full_only_fields(
    *, source_run_id: str, physical_backend: str,
) -> dict[str, object]:
    identity = {
        "schema": "onnx-splitpoint/full-only-quality-request-identity",
        "schema_version": 1,
        "quality_canary_id": (
            "tensorrt_at_deepx_m1_full"
            if source_run_id == "native_full_tensorrt"
            else "deepx_m1_full"
        ),
        "eval_run_id": "synthetic-evaluation-run",
        "model_id": "resnet50",
        "setup_id": "orin_nx_deepx_m1_01",
        "source_run_id": source_run_id,
        "backend": (
            "tensorrt" if source_run_id == "native_full_tensorrt"
            else "deepx_m1"
        ),
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }
    return {
        "full_only_plan_identity_required": True,
        "full_only_plan_identity": identity,
        "full_only_plan_identity_sha256": canary._canonical_sha256(identity),
        "quality_canary_id": identity["quality_canary_id"],
        "eval_run_id": identity["eval_run_id"],
        "model_id": identity["model_id"],
        "setup_id": identity["setup_id"],
        "source_run_id": identity["source_run_id"],
        "backend": physical_backend,
        "variant": "full",
        "execution_role": "full_quality_only",
        "performance_claims_emitted": False,
    }


def _producer_execution_contract(
    *, source_run_id: str, backend: str, dxnn_sha: str,
) -> dict[str, object]:
    is_trt = source_run_id == "native_full_tensorrt"
    runner_sha = hashlib.sha256(f"{source_run_id}-runner".encode()).hexdigest()
    endpoint = _sealed_component({
        "schema": "onnx-splitpoint/output-endpoint-contract",
        "schema_version": 3,
        "task": "classification",
        "stage": "classification_logits",
        "output_format": "classification_logits",
        "semantic": {},
        "tensor_signature": {
            "tensor_count": 1,
            "tensors": [{"index": 0, "rank": 1, "shape": [1000]}],
        },
    })
    preprocessing = _sealed_component(dict(_PREPROCESSING_IDENTITY))
    dataset = {
        "manifest_name": "imagenet_validation_manifest.json",
        "manifest_sha256": SHA["validation"],
        "image_ids_sha256": SHA["image-ids"],
        "ground_truth_sha256": SHA["ground-truth"],
        "image_count": 500,
    }
    if is_trt:
        runtime_input_encoding = {
            "schema": "onnx-splitpoint/model-input-encoding-contract",
            "schema_version": 1,
            "image_scale": "imagenet",
            "input_dtype": "float32",
        }
        preprocessing = {
            **preprocessing,
            "runtime_input_encoding": runtime_input_encoding,
            "runtime_input_encoding_sha256": canary._canonical_sha256(
                runtime_input_encoding
            ),
        }
        dataset["class_identity"] = "label_id"
        source_onnx = {
            "path": "/sealed/resnet50.onnx",
            "sha256": SHA["source"],
            "size_bytes": 1000,
        }
        build_onnx = {
            "path": "/sealed/resnet50-build.onnx",
            "sha256": SHA["source"],
            "size_bytes": 1000,
            "source_onnx_sha256": SHA["source"],
        }
        engine_sha = hashlib.sha256(b"trt-engine").hexdigest()
        engine = {
            "path": "/sealed/resnet50.engine",
            "sha256": engine_sha,
            "size_bytes": 2000,
            "source_onnx_sha256": SHA["source"],
            "build_onnx_sha256": SHA["source"],
        }
        trtexec = {
            "path": "/usr/src/tensorrt/bin/trtexec",
            "sha256": hashlib.sha256(b"trtexec").hexdigest(),
            "size_bytes": 3000,
        }
        receipt: dict[str, object] = {
            "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
            "schema_version": 1,
            "build_returncode": 0,
            "dry_run": False,
            "source_onnx": build_onnx["path"],
            "source_onnx_sha256": build_onnx["sha256"],
            "engine": engine["path"],
            "engine_sha256": engine["sha256"],
            "trtexec": trtexec["path"],
            "trtexec_sha256": trtexec["sha256"],
            "command": [
                trtexec["path"],
                f"--onnx={build_onnx['path']}",
                f"--saveEngine={engine['path']}",
                "--fp16",
            ],
        }
        receipt["receipt_sha256"] = canary._canonical_sha256(receipt)
        receipt_binding = {
            "path": "/sealed/resnet50.engine.receipt.json",
            "receipt": receipt,
            "sha256": canary._canonical_sha256(receipt),
            "size_bytes": len(canary._canonical_bytes(receipt)),
        }
        postprocessor = _sealed_component({
            "schema": "onnx-splitpoint/classification-topk-postprocessor-contract",
            "schema_version": 1,
            "canonical_record_endpoint": "classification_topk_hits",
            "implementation_functions": [
                "_classification_logits_vector", "_classification_gt_metrics",
            ],
            "implementation_runner_sha256": runner_sha,
            "sort_order": "score_descending",
            "topk": 5,
        })
        quality_record_endpoint = _sealed_component({
            "schema": (
                "onnx-splitpoint/"
                "classification-quality-record-endpoint-contract"
            ),
            "schema_version": 1,
            "canonical_record_endpoint": "classification_topk_hits",
            "postprocessor_contract_sha256": postprocessor["sha256"],
            "implementation_runner_sha256": runner_sha,
            "vendored_endpoint_attestor_sha256": hashlib.sha256(
                b"vendored-endpoint-attestor"
            ).hexdigest(),
        })
        quality_contract: dict[str, object] = {
            "schema": "onnx-splitpoint/central-classification-quality-contract",
            "schema_version": 1,
            "task": "classification",
            "contract_scope": "canonical_quality_record_semantics",
            "canonical_record_endpoint": "classification_topk_hits",
            "model": {"artifact_name": "resnet50.onnx", "sha256": SHA["source"]},
            "dataset": dataset,
            "preprocessing": preprocessing,
            "postprocessor": postprocessor,
            "quality_record_endpoint": quality_record_endpoint,
            "quality_record_endpoint_contract_sha256": (
                quality_record_endpoint["sha256"]
            ),
        }
        quality_contract["quality_contract_sha256"] = canary._canonical_sha256(
            quality_contract
        )
        precision = _sealed_component({
            "schema": "onnx-splitpoint/tensorrt-runtime-precision-contract",
            "schema_version": 1,
            "runtime_precision_identity": "fp16",
            "source_onnx_sha256": SHA["source"],
            "build_onnx_sha256": SHA["source"],
            "engine_sha256": engine_sha,
        })
        vendored_attestor_sha = hashlib.sha256(
            b"vendored-endpoint-attestor"
        ).hexdigest()
        endpoint_attestor_identity = {
            "schema": "onnx-splitpoint/vendored-endpoint-attestor-identity",
            "schema_version": 1,
            "source": "suite_vendored",
            "sha256": vendored_attestor_sha,
            "expected_sha256": vendored_attestor_sha,
        }
        endpoint_attestor = {
            **_sealed_component(endpoint_attestor_identity),
            "path": "/sealed/native_output_endpoint.py",
        }
        endpoint_attestation = {
            "attested": True,
            "status": "passed",
            "stage": "classification_logits",
            "endpoint_contract_hash": endpoint["sha256"],
        }
        endpoint_authority = _sealed_component({
            "schema": "onnx-splitpoint/tensorrt-endpoint-authority",
            "schema_version": 1,
            "graph_binding_source": (
                "authoritative_suite_output_contract_plus_exact_onnx_endpoint:v2"
            ),
            "endpoint_contract_complete": True,
            "contract_resolution_status": "attested",
            "source_contracts_sha256": hashlib.sha256(
                b"source-contract-container"
            ).hexdigest(),
            "recorded_contract_sha256": hashlib.sha256(
                b"source-contract-row"
            ).hexdigest(),
            "full_model_sha256": SHA["source"],
            "terminal_model_sha256": SHA["source"],
            "endpoint_contract_hash": endpoint["sha256"],
            "stage": "classification_logits",
            "output_endpoint_attestation": endpoint_attestation,
        })
        runtime_precision = "fp16"
        prepared_evidence = None
        prepared_join = None
        prepared_join_sha = None
    else:
        quality_record_endpoint = _sealed_component({
            "schema": "onnx-splitpoint/deepx-classification-endpoint-contract",
            "schema_version": 1,
            "canonical_record_endpoint": "classification_topk_hits",
            "postprocessor": "numpy_argsort_descending_top5",
            "implementation_runner_sha256": runner_sha,
            "implementation_functions": ["_run_deepx_semantic_validation"],
            "output_contract_sha256": hashlib.sha256(
                b"deepx-output-contract"
            ).hexdigest(),
            "runtime_observed_outputs": {
                "outputs": [{"index": 0, "shape": [1, 1000]}],
            },
        })
        runtime_numeric_identity = {
            "schema": "onnx-splitpoint/runtime-numeric-input-identity",
            "schema_version": 1,
            "backend": "deepx_m1",
            "task": "classification",
            "runtime_input_name": "input",
            "runtime_input_shape": [1, 224, 224, 3],
            "runtime_input_dtype": "uint8",
            "runtime_input_layout": "NHWC",
            "preprocessing_contract_sha256": SHA["preprocessing"],
        }
        runtime_numeric_sha = canary._canonical_sha256(runtime_numeric_identity)
        prepared_join = {
            "schema": "onnx-splitpoint/deepx-performance-quality-input-binding",
            "schema_version": 1,
            "binding_verified": True,
            "source_image_id": "val-0000",
            "source_image_sha256": hashlib.sha256(b"source-image").hexdigest(),
            "prepared_input_sha256": hashlib.sha256(b"prepared-input").hexdigest(),
            "prepared_input_bytes": 1 * 224 * 224 * 3,
            "prepared_input_name": "input",
            "prepared_input_shape": [1, 224, 224, 3],
            "prepared_input_dtype": "uint8",
            "prepared_input_layout": "NHWC",
            "runtime_preprocessing_sha256": SHA["preprocessing"],
            "runtime_numeric_input_sha256": runtime_numeric_sha,
        }
        prepared_join_sha = canary._canonical_sha256(prepared_join)
        prepared_evidence = {
            "schema": "onnx-splitpoint/deepx-quality-prepared-input-set",
            "schema_version": 1,
            "record_count": 500,
            "records_sha256": SHA["prepared"],
            "runtime_numeric_input_identity": runtime_numeric_identity,
            "runtime_numeric_input_sha256": runtime_numeric_sha,
            "performance_quality_input_binding": prepared_join,
            "performance_quality_input_binding_sha256": prepared_join_sha,
        }
        quality_contract = {
            "schema": "onnx-splitpoint/deepx-central-quality-record-contract",
            "schema_version": 1,
            "contract_scope": "canonical_quality_record_semantics",
            "task": "classification",
            "variant": "full",
            "model": {
                "source_onnx_name": "resnet50.onnx",
                "source_onnx_sha256": SHA["source"],
            },
            "dataset": {**dataset, "class_identity": "label_id"},
            "preprocessing": preprocessing,
            "prepared_input_evidence": prepared_evidence,
            "quality_record_endpoint": quality_record_endpoint,
            "quality_record_endpoint_contract_sha256": (
                quality_record_endpoint["sha256"]
            ),
            "canonical_record_endpoint": "classification_topk_hits",
        }
        quality_contract["quality_contract_sha256"] = canary._canonical_sha256(
            quality_contract
        )
        precision = _sealed_component({
            "schema": "onnx-splitpoint/deepx-runtime-precision-contract",
            "schema_version": 1,
            "artifact_kind": "dxnn",
            "artifact_name": "resnet50.dxnn",
            "artifact_sha256": dxnn_sha,
            "artifact_size_bytes": 123456,
            "precision_semantics": "opaque_vendor_compiled_artifact_identity",
            "declared_precision": None,
        })
        runtime_precision = f"deepx_dxnn_sha256:{dxnn_sha}"

    producer: dict[str, object] = {
        "schema": (
            "onnx-splitpoint/tensorrt-central-quality-producer-identity"
            if is_trt else "onnx-splitpoint/central-quality-producer-identity"
        ),
        "schema_version": 1,
        "model_id": "resnet50",
        "source_run_id": source_run_id,
        "case_id": "full",
        "variant": "full",
        "task": "classification",
        "execution_role": "full_quality_only",
        "backend": backend,
        "setup_id": "orin_nx_deepx_m1_01",
        "performance_claims_emitted": False,
        "model": {
            "source_onnx_sha256": SHA["source"],
            **({
                "source_onnx_size_bytes": source_onnx["size_bytes"],
                "build_onnx_sha256": build_onnx["sha256"],
                "build_onnx_size_bytes": build_onnx["size_bytes"],
                "runtime_artifact_sha256": engine["sha256"],
                "runtime_artifact_size_bytes": engine["size_bytes"],
            } if is_trt else {
                "runtime_artifact_sha256": dxnn_sha,
            }),
        },
        "dataset": dataset,
        "preprocessing": preprocessing,
        "endpoint": endpoint,
        "quality_record_endpoint": quality_record_endpoint,
        "precision": precision,
        "quality_contract": quality_contract,
        "quality_contract_sha256": quality_contract["quality_contract_sha256"],
        "preprocessing_contract_sha256": preprocessing["sha256"],
        "endpoint_contract_hash": endpoint["sha256"],
        "quality_record_endpoint_contract_sha256": (
            quality_record_endpoint["sha256"]
        ),
        "runtime_precision_identity": runtime_precision,
        "implementation_runner_sha256": runner_sha,
    }
    if is_trt:
        producer.update({
            "policy_sha256": EXPECTED_QUALITY_POLICY_SHA256,
            "eval_run_id": "synthetic-evaluation-run",
            "source_onnx": source_onnx,
            "build_onnx": build_onnx,
            "engine": engine,
            "trtexec": trtexec,
            "engine_build_receipt": receipt_binding,
            "engine_build_receipt_file_sha256": hashlib.sha256(
                b"engine-build-receipt-file"
            ).hexdigest(),
            "endpoint_contract_complete": True,
            "endpoint_authority": endpoint_authority,
            "endpoint_attestor": endpoint_attestor,
            "vendored_endpoint_attestor_sha256": vendored_attestor_sha,
            "decoder_contract_sha256": "",
            "nms_contract_sha256": "",
        })
    else:
        producer.update({
            "prepared_input_evidence": prepared_evidence,
            "prepared_input_evidence_sha256": SHA["prepared"],
            "prepared_input_join_binding": prepared_join,
            "prepared_input_join_binding_sha256": prepared_join_sha,
            "decoder_contract_sha256": "",
            "nms_contract_sha256": "",
            **_full_only_fields(
                source_run_id=source_run_id, physical_backend=backend,
            ),
        })
    producer["producer_identity_sha256"] = canary._canonical_sha256(producer)
    return producer


def _endpoint(
    tmp_path: Path, *, source_run_id: str, backend: str,
    records: tuple[dict[str, object], ...], decision: str = "pass",
    dxnn_sha: str = SHA["dxnn"],
) -> QualityEndpointEvidence:
    is_trt = source_run_id == "native_full_tensorrt"
    producer = _producer_execution_contract(
        source_run_id=source_run_id, backend=backend, dxnn_sha=dxnn_sha,
    )
    full_only = _full_only_fields(
        source_run_id=source_run_id, physical_backend=backend,
    )
    request: dict[str, object] = {
        "setup_id": "orin_nx_deepx_m1_01",
        "policy_sha256": EXPECTED_QUALITY_POLICY_SHA256,
        "expected_image_ids_sha256": SHA["image-ids"],
        "preprocessing_contract_sha256": SHA["preprocessing"],
        "statistics": {
            "method": "paired_bootstrap", "bootstrap_repetitions": 500,
            "seed": 20260710, "confidence_level": 0.95,
            "decision": "lower_one_sided_bound",
        },
        "metric_gate_config": {
            "primary_metric": "top1_accuracy",
            "non_inferiority_margin": 0.01,
            "guardrails": {"top5_accuracy_margin": 0.01},
        },
        "quality_contract": producer["quality_contract"],
        "quality_contract_sha256": producer["quality_contract_sha256"],
        "endpoint_contract_hash": producer["endpoint_contract_hash"],
        "quality_record_endpoint_contract_sha256": producer[
            "quality_record_endpoint_contract_sha256"
        ],
        "runtime_precision_identity": producer["runtime_precision_identity"],
        "decoder_contract_sha256": producer["decoder_contract_sha256"],
        "nms_contract_sha256": producer["nms_contract_sha256"],
        **full_only,
    }
    if is_trt:
        receipt_binding = producer["engine_build_receipt"]
        receipt = receipt_binding["receipt"]
        trt_duplicates = {
            "source_model_sha256": producer["source_onnx"]["sha256"],
            "build_onnx_sha256": producer["build_onnx"]["sha256"],
            "runtime_artifact_sha256": producer["engine"]["sha256"],
            "trtexec_sha256": producer["trtexec"]["sha256"],
            "engine_build_receipt_sha256": receipt_binding["sha256"],
            "engine_build_receipt_file_sha256": producer[
                "engine_build_receipt_file_sha256"
            ],
            "trt_engine_build_receipt_sha256": receipt["receipt_sha256"],
            "case_id": "full",
            "task": "classification",
        }
        request.update({
            **trt_duplicates,
        })
    else:
        request.update({
            "prepared_input_evidence_sha256": SHA["prepared"],
            "prepared_input_join_binding": producer[
                "prepared_input_join_binding"
            ],
            "prepared_input_join_binding_sha256": producer[
                "prepared_input_join_binding_sha256"
            ],
        })
    candidate: dict[str, object] = {
        "quality_contract": producer["quality_contract"],
        "quality_contract_sha256": producer["quality_contract_sha256"],
        "preprocessing_contract_sha256": producer[
            "preprocessing_contract_sha256"
        ],
        "quality_record_endpoint_contract_sha256": producer[
            "quality_record_endpoint_contract_sha256"
        ],
        "runtime_precision_identity": producer["runtime_precision_identity"],
        "decoder_contract_sha256": producer["decoder_contract_sha256"],
        "nms_contract_sha256": producer["nms_contract_sha256"],
        **full_only,
    }
    if is_trt:
        candidate["endpoint_contract_hash"] = producer["endpoint_contract_hash"]
        candidate.update(trt_duplicates)
    if not is_trt:
        candidate.update({
            "per_image_transforms": [
                dict(row) for row in _PREPARED_TRANSFORMS
            ],
            "prepared_input_evidence_sha256": SHA["prepared"],
            "prepared_input_join_binding": producer[
                "prepared_input_join_binding"
            ],
            "prepared_input_join_binding_sha256": producer[
                "prepared_input_join_binding_sha256"
            ],
        })
    result_backend = "tensorrt" if is_trt else backend
    return QualityEndpointEvidence(
        source_run_id=source_run_id,
        backend=backend,
        setup_id="orin_nx_deepx_m1_01",
        request_path=tmp_path / f"{source_run_id}-request.json",
        candidate_path=tmp_path / f"{source_run_id}-candidate.json",
        request=request,
        producer=producer,
        candidate=candidate,
        records=records,
        result=_result(
            source_run_id=source_run_id, backend=result_backend,
            records=records, decision=decision,
        ),
    )


def _manifest_items(count: int) -> tuple[dict[str, object], ...]:
    return tuple({
        "sample_id": f"train-{index:04d}",
        "relative_path": f"n{index % 1000:08d}/train-{index:04d}.JPEG",
        "sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
        "size_bytes": 100 + index,
        "class_name": f"n{index % 1000:08d}",
    } for index in range(count))


def _provisioning_manifest(count: int) -> dict[str, object]:
    return {
        "dataset_id": "ilsvrc2012-train-calibration",
        "task": "classification",
        "role": "calibration",
        "split": "train",
        "source_kind": "directory_scan",
        "hash_mode": "content",
        "population_count": count,
        "selection": {
            "strategy": "all", "seed": 20260710,
            "requested_max_items": 0,
            "selection_uses_model_predictions": False,
        },
        "provisioning_selection": {
            "source_split": "train",
            "source_population": "ILSVRC2012 train",
            "strategy": "class_stratified_deterministic_hash",
            "seed": 20260710,
            "requested_items": count,
            "selected_items": count,
            "selected_class_count": count,
            "selection_manifest": f"selection-n{count}.json",
            "selection_manifest_sha256": hashlib.sha256(
                f"selection-{count}".encode()
            ).hexdigest(),
            "kernel_ref": (
                "fixture-owner/"
                + dataset_provisioning.IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG
            ),
            "selection_uses_model_predictions": False,
        },
        "annotations": {"sha256": ""},
        "labels": {"sha256": hashlib.sha256(b"labels").hexdigest()},
    }


def _sealed_manifest(tmp_path: Path, *, count: int = 500) -> tuple[Path, dict[str, object]]:
    root = tmp_path / f"calibration-{count}"
    items: list[dict[str, object]] = []
    for index in range(count):
        content = f"image-{index}".encode()
        relative_path = f"n{index:08d}/train-{index:04d}.JPEG"
        image_path = root / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(content)
        items.append({
            "sample_id": f"train-{index:04d}",
            "relative_path": relative_path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
            "class_name": f"n{index:08d}",
        })
    identity_rows = [{
        "sample_id": row["sample_id"],
        "relative_path": row["relative_path"],
        "sha256": row["sha256"],
        "class_name": row["class_name"],
    } for row in items]
    selection_rows = [{
        "class_name": row["class_name"],
        "source_relative_path": row["relative_path"],
        "archive_path": row["relative_path"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
    } for row in items]
    selection_payload: dict[str, object] = {
        "schema": "onnx-splitpoint/imagenet-train-calibration-selection",
        "schema_version": 1,
        "generated_at_unix": 1786665600,
        "competition": "imagenet-object-localization-challenge",
        "source_root": "/kaggle/input/imagenet/ILSVRC/Data/CLS-LOC/train",
        "strategy": "class_stratified_deterministic_hash",
        "seed": 20260710,
        "requested_count": count,
        "selected_count": count,
        "available_class_count": 1000,
        "selected_class_count": count,
        "class_counts": {str(row["class_name"]): 1 for row in items},
        "selection_uses_model_predictions": False,
        "items": selection_rows,
    }
    selection_payload["selection_payload_sha256"] = canary._canonical_sha256(
        selection_payload
    )
    selection_path = tmp_path / f"selection-n{count}.json"
    selection_path.write_text(
        json.dumps(selection_payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    provisioning = _provisioning_manifest(count)
    provisioning_selection = dict(provisioning["provisioning_selection"])
    provisioning_selection.update({
        "selection_manifest": selection_path.name,
        "selection_manifest_sha256": hashlib.sha256(
            selection_path.read_bytes()
        ).hexdigest(),
    })
    provisioning["provisioning_selection"] = provisioning_selection
    manifest: dict[str, object] = {
        **provisioning,
        "schema": "onnx-splitpoint/dataset-manifest",
        "schema_version": 1,
        "created_at": "2026-08-14T00:00:00Z",
        "root": str(root),
        "root_name": f"calibration-{count}",
        "hash_mode": "content",
        "item_count": count,
        "items": items,
        "items_identity_sha256": "sha256:" + hashlib.sha256(
            json.dumps(
                identity_rows, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "annotations": {"path": "", "sha256": ""},
        "labels": {"path": "", "sha256": ""},
        "final_use_note": (
            "A manifest is content-addressed evidence. Final status still "
            "depends on the campaign profile and a disjointness check."
        ),
    }
    manifest["manifest_payload_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            manifest, indent=2, sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / f"manifest-{count}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    inventory = sorted(({
        "relative_path": row["relative_path"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
    } for row in items), key=lambda row: str(row["relative_path"]))
    contract: dict[str, object] = {
        "schema": "onnx-splitpoint/deepx-calibration-manifest-contract",
        "schema_version": 1,
        "status": "resolved",
        "task": "classification",
        "effective_count": count,
        "item_count": count,
        "root_inventory_count": count,
        "manifest_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "items_identity_sha256": manifest["items_identity_sha256"],
        "root_inventory_sha256": hashlib.sha256(
            json.dumps(
                inventory, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "dataset_id": "ilsvrc2012-train-calibration",
        "split": "train",
        "role": "calibration",
        "hash_mode": "content",
        "manifest_file_name": path.name,
        "dataset_root_name": root.name,
        "manifest_verification": {
            "ok": True,
            "schema_ok": True,
            "payload_hash_ok": True,
            "identity_hash_ok": True,
            "item_count_ok": True,
            "verification_mode": "full",
            "manifest_item_count": count,
            "checked_item_count": count,
            "missing_count": 0,
            "mismatch_count": 0,
        },
        "dataset_registry_binding_sha256": "sha256:" + hashlib.sha256(
            b"dataset-registry-binding"
        ).hexdigest(),
    }
    contract["identity_sha256"] = canary._canonical_sha256(contract)
    return path, contract


def _reseal_manifest(
    path: Path, contract: dict[str, object], payload: dict[str, object],
) -> dict[str, object]:
    identity_rows = [{
        "sample_id": row["sample_id"],
        "relative_path": row["relative_path"],
        "sha256": str(row["sha256"]).removeprefix("sha256:"),
        "class_name": row["class_name"],
    } for row in payload["items"]]
    payload["items_identity_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            identity_rows, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    payload.pop("manifest_payload_sha256", None)
    payload["manifest_payload_sha256"] = "sha256:" + hashlib.sha256(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False,
        ).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    changed = dict(contract)
    changed["manifest_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    changed["manifest_payload_sha256"] = payload["manifest_payload_sha256"]
    changed["items_identity_sha256"] = payload["items_identity_sha256"]
    changed.pop("identity_sha256", None)
    changed["identity_sha256"] = canary._canonical_sha256(changed)
    return changed


def _reseal_selection_receipt(
    path: Path, payload: dict[str, object],
) -> str:
    payload.pop("selection_payload_sha256", None)
    payload["selection_payload_sha256"] = canary._canonical_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile(*, count: int) -> dict[str, object]:
    label = "B500" if count == 500 else "B1000"
    _, profile = canary._materialized_frozen_profile(label)
    from onnx_splitpoint_tool.workflow.start_snapshot import snapshot_payload_sha256

    target = {
        "id": "orin_nx_deepx_m1_01",
        "label": "DeepX Orin NX",
        "accelerator": "deepx_m1",
        "provider": "deepx_m1",
        "enabled": True,
        "runtime": {},
        "remote": {},
        "build_environment_id": "deepx_dxcom_x86",
        "build_environment": {},
        "setup_source": "/frozen/hardware_setups.yaml",
        "setup_lock_id": "orin_nx_deepx_m1_01",
        "energy": {},
        "tags": [],
        "note": "Synthetic sealed unit target.",
    }
    profile["hardware"] = {
        **dict(profile["hardware"]),
        "resolved_targets": [target],
        "resolved_targets_sha256": snapshot_payload_sha256([target]),
        "resolution_frozen_at_start": True,
    }
    return profile


def _profile_provenance(
    *, count: int, profile: dict[str, object],
    runtime_bindings: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    from onnx_splitpoint_tool.workflow.start_snapshot import (
        build_profile_start_snapshot,
        public_start_snapshot_metadata,
    )

    label = "B500" if count == 500 else "B1000"
    name = BASELINE_PROFILE_NAME if count == 500 else CANDIDATE_PROFILE_NAME
    source_path, source = canary._frozen_profile_source(label)
    targets = list(dict(profile["hardware"])["resolved_targets"])
    bindings = runtime_bindings or {
        "runtime_materialized": True,
        "hardware_targets": targets,
        "hardware_targets_sha256": dict(profile["hardware"])[
            "resolved_targets_sha256"
        ],
    }
    snapshot = build_profile_start_snapshot(
        profile_request=str(source_path),
        source_profile=source,
        resolved_profile=profile,
        profile_id=name,
        profile_path=str(source_path),
        profile_source="file",
        runtime_bindings=bindings,
    )
    return source, public_start_snapshot_metadata(snapshot)


def _replace_profile_with_provenance(
    arm: CalibrationArmEvidence, profile: dict[str, object],
    runtime_bindings: dict[str, object] | None = None,
) -> CalibrationArmEvidence:
    source, snapshot = _profile_provenance(
        count=arm.expected_calibration_count, profile=profile,
        runtime_bindings=runtime_bindings,
    )
    return replace(
        arm, profile=profile, profile_source=source,
        profile_start_snapshot=snapshot,
    )


def _base(tmp_path: Path, *, count: int, dxnn_sha: str = SHA["dxnn"]) -> ArmEvidence:
    suffix = "500" if count == 500 else "1000"
    return ArmEvidence(
        run_dir=tmp_path / f"b{suffix}",
        mode=CLASSIFICATION_PREPROCESSING_IMAGENET,
        profile_name=(BASELINE_PROFILE_NAME if count == 500 else CANDIDATE_PROFILE_NAME),
        cache_dir=(B500_CACHE_NAMESPACE if count == 500 else B1000_CACHE_NAMESPACE),
        cache_key=f"deepx_m1_full_imagenet_mean_std_v2_key{suffix}",
        cache_contract_sha256=SHA[f"cache-{suffix}"],
        task="classification",
        target="deepx_m1",
        source_onnx_sha256=SHA["source"],
        build_onnx_sha256=SHA["build"],
        dxcom_config_sha256=SHA[f"dxcom-{suffix}"],
        build_options_sha256=hashlib.sha256(f"options-{suffix}".encode()).hexdigest(),
        dxnn_sha256=dxnn_sha,
        calibration_identity_sha256=SHA[f"calibration-{suffix}"],
        calibration_contract_sha256=hashlib.sha256(
            f"calibration-contract-{suffix}".encode()
        ).hexdigest(),
        calibration_items_identity_sha256=hashlib.sha256(
            f"calibration-items-{suffix}".encode()
        ).hexdigest(),
        compiler_identity_sha256=SHA["compiler"],
        compiler_contract_sha256=SHA["compiler-contract"],
        validation_manifest_sha256=SHA["validation"],
        validation_image_ids_sha256=SHA["image-ids"],
        validation_ground_truth_sha256=SHA["ground-truth"],
        prepared_input_evidence_sha256=SHA["prepared"],
        preprocessing_contract_sha256=SHA["preprocessing"],
        policy_sha256=EXPECTED_QUALITY_POLICY_SHA256,
        request_path=tmp_path / f"b{suffix}-request.json",
        candidate_path=tmp_path / f"b{suffix}-candidate.json",
        records=(),
        deepx_result={},
        trt_result={},
        workflow_status="ok",
    )


def _arm(
    tmp_path: Path, *, count: int, deepx_hits: tuple[int, int],
    trt_hits: tuple[int, int] = (406, 476), dxnn_sha: str = SHA["dxnn"],
) -> CalibrationArmEvidence:
    label = "B500" if count == 500 else "B1000"
    suffix = "500" if count == 500 else "1000"
    items = _manifest_items(count)
    manifest = _provisioning_manifest(count)
    manifest_path = tmp_path / f"manifest-{suffix}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    build_options = {
        "calibration_method": "ema", "calibration_count": count,
        "opt_level": 0,
    }
    contract = {
        "schema": "onnx-splitpoint/deepx-full-cache-contract",
        "schema_version": 2,
        "target": "deepx_m1", "variant": "full", "task": "classification",
        "classification_preprocessing": CLASSIFICATION_PREPROCESSING_IMAGENET,
        "source_onnx_sha256": SHA["source"],
        "build_onnx_sha256": SHA["build"],
        "dxcom_config_sha256": SHA[f"dxcom-{suffix}"],
        "preprocessing_contract": {"sha256": SHA["preprocessing"]},
        "calibration_manifest_contract": {
            "identity_sha256": SHA[f"calibration-{suffix}"], "count": count,
        },
        "compiler_identity": {"identity_sha256": SHA["compiler"]},
        "build_options": build_options,
        "contract_sha256": SHA[f"cache-{suffix}"],
    }
    dxcom = {
        "inputs": {"input": [1, 3, 224, 224]},
        "calibration_num": count,
        "calibration_method": "ema",
        "default_loader": {
            "dataset_path": f"/calibration/b{suffix}",
            "file_extensions": ["jpg", "jpeg", "png"],
            "preprocessings": [{"resize": {"width": 224, "height": 224}}],
        },
    }
    deepx_records = _records(*deepx_hits)
    trt_records = _records(*trt_hits)
    profile = _profile(count=count)
    profile_source, profile_start_snapshot = _profile_provenance(
        count=count, profile=profile,
    )
    return CalibrationArmEvidence(
        label=label,
        expected_calibration_count=count,
        evaluation_run_id="synthetic-evaluation-run",
        base=_base(tmp_path, count=count, dxnn_sha=dxnn_sha),
        profile=profile,
        profile_source=profile_source,
        profile_start_snapshot=profile_start_snapshot,
        cache_contract=contract,
        calibration_contract=dict(contract["calibration_manifest_contract"]),
        calibration_manifest_path=manifest_path,
        calibration_manifest=manifest,
        calibration_items=items,
        dxcom_config_path=tmp_path / f"dxcom-{suffix}.json",
        dxcom_config=dxcom,
        build_options=build_options,
        artifact_status="ready_reused" if count == 500 else "ready_built",
        receipt_path=tmp_path / f"receipt-{suffix}.json",
        deepx_endpoint=_endpoint(
            tmp_path, source_run_id="deepx_m1_full", backend="deepx_m1",
            records=deepx_records,
            decision="inconclusive" if count == 500 else "pass",
            dxnn_sha=dxnn_sha,
        ),
        trt_endpoint=_endpoint(
            tmp_path, source_run_id="native_full_tensorrt",
            backend="native_tensorrt", records=trt_records,
        ),
    )


def _pair(tmp_path: Path) -> tuple[CalibrationArmEvidence, CalibrationArmEvidence]:
    return (
        _arm(tmp_path, count=500, deepx_hits=(401, 474)),
        _arm(tmp_path, count=1000, deepx_hits=(410, 480)),
    )


def test_exact_pair_is_verified_and_readiness_uses_b1000_and_trt(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)
    result = compare_calibration_arms(baseline, candidate)

    assert result["status"] == "verified"
    assert result["calibration_cohort"]["b500_is_subset_of_b1000"] is True
    assert result["calibration_cohort"]["additional_b1000_items"] == 500
    assert result["paired_b1000_minus_b500"]["top1"]["repetitions"] == 500
    assert result["paired_b1000_minus_b500"]["top1"]["seed"] == 20260710
    assert result["paired_b1000_minus_b500"]["top5"]["seed"] == 20260711
    assert result["authoritative_b1000_vs_setup_local_trt_guardrail"]["decision"] == "pass"
    assert result["central_b1000_deepx_quality_pass"] is True
    assert result["standard_plus_ready"] is True


def test_release_scientific_and_historical_b500_authorities_are_pinned() -> None:
    assert _RELEASE_AUTHORITY_SNAPSHOT == {
        "EXPECTED_SOURCE_ONNX_SHA256": "cebd9d5879ddd304a18f51e559ab74fba5cbd1f610b9359321bc5228acbe4f50",
        "EXPECTED_BUILD_ONNX_SHA256": "b5921d9f75da27c05b6a70fda31f4a0b119383eaf1ddc263fa5e20ec6bd61528",
        "EXPECTED_COMPILER_IDENTITY_SHA256": "298a0f8649edb95255129db076f3ffbee8a804698386b1f46077a3c0e4b970c2",
        "EXPECTED_COMPILER_CONTRACT_SHA256": "f5a67a84946d2f44f4578e1737a67778783f5c7c6ef47fb7e1de0beb6e31bb86",
        "EXPECTED_VALIDATION_MANIFEST_SHA256": "64a7ef3bf55352bb39ef86f25fe73e6ff18a51ed1a02c8aca4940178f69e7b6c",
        "EXPECTED_VALIDATION_IMAGE_IDS_SHA256": "71032a98e158ca71711a5567de5d46fbf04f05777baa940f74f3c39fdf0c083f",
        "EXPECTED_VALIDATION_GROUND_TRUTH_SHA256": "87775e86ef3bcf3ec1e0d0a79696bf2f58fa215d5320ce80167bb1f417c7a177",
        "EXPECTED_PREPARED_INPUT_EVIDENCE_SHA256": "0833140afa44181b7163b2234197c8e64b308653e1e2656e46f1707e9e875724",
        "EXPECTED_PREPROCESSING_CONTRACT_SHA256": "ea28cf5ac35bd4c9a3321ac97fd559f32fd7a661dc4a93c324fe3ffc54188fa9",
        "EXPECTED_B500_CALIBRATION_MANIFEST_SHA256": "6d6978cafd1b828d6fa9438b232120f2a769311c28492bea5a141af586c3d58c",
        "EXPECTED_B500_CALIBRATION_ITEMS_SHA256": "943d1d1506250f3b6e29d0fb06b6118f01c3e70d769b00939d74e439cda9686c",
        "EXPECTED_B500_CALIBRATION_IDENTITY_SHA256": "913d753f34e6ab545ec38c5dcd27177b748eecf0c200625d579b7e6a6413efef",
        "EXPECTED_B500_CALIBRATION_CONTRACT_SHA256": "bea432f0cfbb7ec225a63e60c56c32f5c4c0fdf6e75c00e9852ec014e10677a7",
        "EXPECTED_B500_DXCOM_CONFIG_SHA256": "65b18a1721afea8432f2ac01631655bece6b8fb81326f78462491cb6aa771f9f",
        "EXPECTED_B500_BUILD_OPTIONS_SHA256": "55f9fade1ec02c5754a96f736c3233f5e3d4bad46b4980a01b1880e0340968b5",
        "EXPECTED_B500_CACHE_KEY": "deepx_m1_full_imagenet_mean_std_v2_313fece50a4acf14bf4fc4076b10b063",
        "EXPECTED_B500_CACHE_CONTRACT_SHA256": "3804283b15e4b26040ec9775023d822a54df0111e564836054ba1e9f6cf34f49",
        "EXPECTED_B500_DXNN_SHA256": "a8bb14689d2d160f0b66c0f4b614c8e214808ca28adc25dce8cb6fa419af05bb",
        "EXPECTED_SETUP_ID": "orin_nx_deepx_m1_01",
        "EXPECTED_B500_DEEPX_TOP1_HITS": 401,
        "EXPECTED_B500_DEEPX_TOP5_HITS": 474,
        "EXPECTED_REFERENCE_TOP1_HITS": 406,
        "EXPECTED_REFERENCE_TOP5_HITS": 476,
    }


@pytest.mark.parametrize(
    "field", ("source_onnx_sha256", "validation_manifest_sha256"),
)
def test_joint_scientific_authority_substitution_fails_closed(
    tmp_path: Path, field: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    baseline = replace(baseline, base=replace(baseline.base, **{field: "f" * 64}))
    candidate = replace(candidate, base=replace(candidate.base, **{field: "f" * 64}))
    with pytest.raises(ValueError, match="frozen v2.75.40-B scientific authority"):
        compare_calibration_arms(baseline, candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("cache_key", "substitute-cache-key"),
        ("cache_contract_sha256", "f" * 64),
        ("dxnn_sha256", "e" * 64),
        ("calibration_items_identity_sha256", "d" * 64),
    ),
)
def test_historical_b500_substitution_fails_closed(
    tmp_path: Path, field: str, value: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    baseline = replace(baseline, base=replace(baseline.base, **{field: value}))
    with pytest.raises(ValueError, match="frozen v2.75.40-B calibration authority"):
        compare_calibration_arms(baseline, candidate)
    assert BOOTSTRAP_REPETITIONS == 500
    assert BASELINE_CALIBRATION_COUNT == 500
    assert CANDIDATE_CALIBRATION_COUNT == 1000


def test_actual_manifest_items_payload_and_inventory_are_bound(tmp_path: Path) -> None:
    path, contract = _sealed_manifest(tmp_path)
    payload, items = canary._load_manifest(
        path, contract=contract, expected_count=500, label="B500",
    )
    assert payload["item_count"] == 500
    assert len(items) == 500

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["items"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="self-verification"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )


def test_production_writer_kernel_ref_is_accepted_without_pinning_owner(
    tmp_path: Path,
) -> None:
    path, contract = _sealed_manifest(tmp_path / "manifest")
    job = dataset_provisioning._write_imagenet_calibration_export_kernel(
        tmp_path / "production-writer",
        username="independent-owner",
        competition=dataset_provisioning.IMAGENET_KAGGLE_COMPETITION,
        calibration_items=500,
        seed=20260710,
    )
    assert job["kernel_ref"] == (
        "independent-owner/"
        + dataset_provisioning.IMAGENET_CALIBRATION_EXPORT_KERNEL_SLUG
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["provisioning_selection"]["kernel_ref"] = job["kernel_ref"]
    contract = _reseal_manifest(path, contract, manifest)

    payload, items = canary._load_manifest(
        path, contract=contract, expected_count=500, label="B500",
    )

    assert payload["provisioning_selection"]["kernel_ref"] == job["kernel_ref"]
    assert len(items) == 500


@pytest.mark.parametrize(
    ("kernel_ref", "message"),
    (
        (
            "private/imagenet-calibration",
            "kernel slug mismatch.*onnx-splitpoint-imagenet-calibration-export-v60j",
        ),
        (
            "valid-owner/lookalike-calibration-export",
            "kernel slug mismatch",
        ),
        (
            "@noncanonical/onnx-splitpoint-imagenet-calibration-export-v60j",
            "kernel owner is not canonical",
        ),
        (
            "owner/onnx-splitpoint-imagenet-calibration-export-v60j/extra",
            "canonical owner/slug form",
        ),
    ),
)
def test_production_calibration_kernel_ref_substitution_fails_closed(
    kernel_ref: str, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        canary._production_calibration_kernel_ref(kernel_ref, label="B500")


def test_frozen_b500_manifest_file_and_item_authorities_are_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _sealed_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    items_sha = str(payload["items_identity_sha256"]).removeprefix("sha256:")
    monkeypatch.setattr(
        canary, "EXPECTED_B500_CALIBRATION_MANIFEST_SHA256", manifest_sha,
    )
    monkeypatch.setattr(
        canary, "EXPECTED_B500_CALIBRATION_ITEMS_SHA256", items_sha,
    )
    canary._validate_frozen_b500_manifest_authority(path, payload)

    monkeypatch.setattr(
        canary, "EXPECTED_B500_CALIBRATION_MANIFEST_SHA256", "f" * 64,
    )
    with pytest.raises(ValueError, match="file authority.*expected=.*observed="):
        canary._validate_frozen_b500_manifest_authority(path, payload)

    monkeypatch.setattr(
        canary, "EXPECTED_B500_CALIBRATION_MANIFEST_SHA256", manifest_sha,
    )
    monkeypatch.setattr(
        canary, "EXPECTED_B500_CALIBRATION_ITEMS_SHA256", "e" * 64,
    )
    with pytest.raises(ValueError, match="cohort authority.*expected=.*observed="):
        canary._validate_frozen_b500_manifest_authority(path, payload)


def test_manifest_contract_count_and_inventory_tamper_fail_closed(tmp_path: Path) -> None:
    path, contract = _sealed_manifest(tmp_path)
    wrong_count = {**contract, "effective_count": 1000}
    with pytest.raises(ValueError, match="effective_count mismatch"):
        canary._load_manifest(
            path, contract=wrong_count, expected_count=500, label="B500",
        )
    wrong_inventory = {**contract, "root_inventory_sha256": "f" * 64}
    wrong_inventory.pop("identity_sha256")
    wrong_inventory["identity_sha256"] = canary._canonical_sha256(
        wrong_inventory
    )
    with pytest.raises(ValueError, match="inventory identity mismatch"):
        canary._load_manifest(
            path, contract=wrong_inventory, expected_count=500, label="B500",
        )
    extra = {**contract, "unreviewed_active_field": True}
    extra.pop("identity_sha256")
    extra["identity_sha256"] = canary._canonical_sha256(extra)
    with pytest.raises(ValueError, match="shape is not exact-v2"):
        canary._load_manifest(
            path, contract=extra, expected_count=500, label="B500",
        )


def test_identical_dxnn_hash_is_scientifically_valid(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)
    result = compare_calibration_arms(baseline, candidate)
    assert result["dxnn_identity"] == {
        "same_sha256": True,
        "scientifically_valid": True,
        "interpretation": "calibration_count_change_produced_identical_dxnn_bytes",
    }


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("compiler_identity_sha256", "invariant run evidence"),
        ("compiler_contract_sha256", "invariant run evidence"),
        ("build_onnx_sha256", "invariant run evidence"),
        ("preprocessing_contract_sha256", "invariant run evidence"),
        ("validation_image_ids_sha256", "invariant run evidence"),
        ("validation_ground_truth_sha256", "invariant run evidence"),
        ("prepared_input_evidence_sha256", "invariant run evidence"),
        ("policy_sha256", "invariant run evidence"),
    ],
)
def test_invariant_contract_drift_fails_closed(
    tmp_path: Path, field: str, message: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    candidate = replace(
        candidate, base=replace(candidate.base, **{field: "f" * 64}),
    )
    with pytest.raises(ValueError, match=message):
        compare_calibration_arms(baseline, candidate)


def test_non_subset_and_selection_drift_fail_closed(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)
    items = list(candidate.calibration_items)
    items[0] = {**items[0], "sha256": "f" * 64}
    with pytest.raises(ValueError, match="not a subset"):
        compare_calibration_arms(
            baseline, replace(candidate, calibration_items=tuple(items)),
        )

    changed = dict(candidate.calibration_manifest)
    changed["selection"] = {
        **dict(changed["selection"]), "seed": 1,
    }
    with pytest.raises(ValueError, match="selection semantics"):
        compare_calibration_arms(
            baseline, replace(candidate, calibration_manifest=changed),
        )


def test_only_calibration_derived_cache_and_dxcom_fields_may_change(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    changed_dxcom = json.loads(json.dumps(candidate.dxcom_config))
    changed_dxcom["inputs"]["input"] = [1, 3, 256, 256]
    with pytest.raises(ValueError, match="DX-COM configs differ"):
        compare_calibration_arms(
            baseline, replace(candidate, dxcom_config=changed_dxcom),
        )

    changed_contract = json.loads(json.dumps(candidate.cache_contract))
    changed_contract["compiler_identity"] = {"identity_sha256": "f" * 64}
    with pytest.raises(ValueError, match="outside allowed"):
        compare_calibration_arms(
            baseline, replace(candidate, cache_contract=changed_contract),
        )

    same_key = replace(
        candidate,
        base=replace(candidate.base, cache_key=baseline.base.cache_key),
    )
    with pytest.raises(ValueError, match="cache contract, key and root"):
        compare_calibration_arms(baseline, same_key)


def test_validation_order_and_incomplete_controls_fail_closed(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)
    _, ids_sha, gt_sha = canary._classification_record_identity(
        candidate.deepx_endpoint.records, label="ordered",
    )
    reversed_records = tuple(reversed(candidate.deepx_endpoint.records))
    reversed_ids, reversed_ids_sha, reversed_gt_sha = (
        canary._classification_record_identity(reversed_records, label="reversed")
    )
    assert reversed_ids != [row["image_id"] for row in candidate.deepx_endpoint.records]
    assert reversed_ids_sha == ids_sha
    assert reversed_gt_sha == gt_sha
    reordered = replace(
        candidate,
        deepx_endpoint=replace(
            candidate.deepx_endpoint,
            records=reversed_records,
        ),
    )
    with pytest.raises(ValueError, match="sample order"):
        compare_calibration_arms(baseline, reordered)

    failed_trt_result = _with_result_decision(candidate.trt_endpoint.result, "fail")
    failed_trt = replace(
        candidate,
        trt_endpoint=replace(candidate.trt_endpoint, result=failed_trt_result),
    )
    result = compare_calibration_arms(baseline, failed_trt)
    assert result["setup_local_tensorrt_controls_pass"] is False
    assert result["standard_plus_ready"] is False


def test_prepared_input_payload_is_recomputed_not_only_cross_compared(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)

    def drift(endpoint: QualityEndpointEvidence) -> QualityEndpointEvidence:
        payload = json.loads(json.dumps(endpoint.candidate))
        payload["per_image_transforms"][0]["prepared_input_sha256"] = "f" * 64
        return replace(endpoint, candidate=payload)

    baseline = replace(baseline, deepx_endpoint=drift(baseline.deepx_endpoint))
    candidate = replace(candidate, deepx_endpoint=drift(candidate.deepx_endpoint))
    with pytest.raises(ValueError, match="prepared-input transform identity"):
        compare_calibration_arms(baseline, candidate)


def test_setup_identity_cannot_be_jointly_spoofed(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)

    def drift(endpoint: QualityEndpointEvidence) -> QualityEndpointEvidence:
        producer = json.loads(json.dumps(endpoint.producer))
        request = json.loads(json.dumps(endpoint.request))
        result = json.loads(json.dumps(endpoint.result))
        producer["setup_id"] = "other_setup"
        request["setup_id"] = "other_setup"
        result["source_setup_id"] = "other_setup"
        return replace(
            endpoint, setup_id="other_setup", producer=producer,
            request=request, result=result,
        )

    baseline = replace(
        baseline,
        deepx_endpoint=drift(baseline.deepx_endpoint),
        trt_endpoint=drift(baseline.trt_endpoint),
    )
    candidate = replace(
        candidate,
        deepx_endpoint=drift(candidate.deepx_endpoint),
        trt_endpoint=drift(candidate.trt_endpoint),
    )
    with pytest.raises(ValueError, match="setup identity is not frozen"):
        compare_calibration_arms(baseline, candidate)


def test_trt_per_image_control_vector_must_be_identical_across_arms(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    records = json.loads(json.dumps(candidate.trt_endpoint.records))
    records[0]["candidate"]["top1_hit"] = False
    records[406]["candidate"]["top1_hit"] = True
    records[1]["candidate"]["top5_hit"] = False
    records[476]["candidate"]["top5_hit"] = True
    changed = replace(
        candidate,
        trt_endpoint=replace(candidate.trt_endpoint, records=tuple(records)),
    )
    with pytest.raises(ValueError, match="TensorRT hit vectors"):
        compare_calibration_arms(baseline, changed)


def test_all_central_results_use_frozen_cpu_ort_reference_hits(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    result = json.loads(json.dumps(candidate.deepx_endpoint.result))
    result["primary"]["reference_hits"] = 405
    changed = replace(
        candidate,
        deepx_endpoint=replace(candidate.deepx_endpoint, result=result),
    )
    with pytest.raises(ValueError, match="CPU/ORT reference authority"):
        compare_calibration_arms(baseline, changed)

    incomplete_result = {
        **candidate.deepx_endpoint.result,
        "technical_status": "failed",
    }
    incomplete = replace(
        candidate,
        deepx_endpoint=replace(candidate.deepx_endpoint, result=incomplete_result),
    )
    with pytest.raises(ValueError, match="not exact/complete"):
        compare_calibration_arms(baseline, incomplete)


def test_joint_endpoint_record_drift_cannot_reuse_frozen_validation_digests(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)

    def drift(endpoint: QualityEndpointEvidence) -> QualityEndpointEvidence:
        records = tuple({
            **row,
            "image_id": "other-" + str(row["image_id"]),
            "label_id": (int(row["label_id"]) + 1) % 1000,
        } for row in endpoint.records)
        return replace(endpoint, records=records)

    baseline = replace(
        baseline,
        deepx_endpoint=drift(baseline.deepx_endpoint),
        trt_endpoint=drift(baseline.trt_endpoint),
    )
    candidate = replace(
        candidate,
        deepx_endpoint=drift(candidate.deepx_endpoint),
        trt_endpoint=drift(candidate.trt_endpoint),
    )
    with pytest.raises(ValueError, match="frozen validation ID/GT authority"):
        compare_calibration_arms(baseline, candidate)


def test_b1000_central_quality_and_direct_trt_guardrail_both_gate_readiness(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    central_inconclusive = replace(
        candidate,
        deepx_endpoint=replace(
            candidate.deepx_endpoint,
            result=_with_result_decision(
                candidate.deepx_endpoint.result, "inconclusive",
            ),
        ),
    )
    result = compare_calibration_arms(baseline, central_inconclusive)
    assert result["authoritative_b1000_vs_setup_local_trt_guardrail"]["decision"] == "pass"
    assert result["central_b1000_deepx_quality_pass"] is False
    assert result["standard_plus_ready"] is False

    worse = _arm(
        tmp_path, count=1000, deepx_hits=(390, 465), trt_hits=(406, 476),
    )
    # Preserve an ostensibly passing central label: the direct setup-local TRT
    # guardrail must still catch the record-level regression.
    result = compare_calibration_arms(baseline, worse)
    assert result["authoritative_b1000_vs_setup_local_trt_guardrail"]["decision"] == "fail"
    assert result["standard_plus_ready"] is False


def test_b1000_restart_safe_reuse_is_accepted_and_reported(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)
    candidate = replace(candidate, artifact_status="ready_reused")
    result = compare_calibration_arms(baseline, candidate)
    assert result["candidate_b1000"]["artifact_status"] == "ready_reused"
    assert result["standard_plus_ready"] is True

    with pytest.raises(ValueError, match="built/reused"):
        compare_calibration_arms(
            baseline, replace(candidate, artifact_status="ready_fallback"),
        )


def test_profile_force_build_full_only_and_namespace_are_frozen(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)
    profile = json.loads(json.dumps(candidate.profile))
    profile["deepx_build"]["force_build"] = True
    with pytest.raises(ValueError, match="profile axis changed: force_build"):
        compare_calibration_arms(
            baseline, _replace_profile_with_provenance(candidate, profile),
        )

    profile = json.loads(json.dumps(candidate.profile))
    profile["quality_canary"]["execution_scope"] = "all"
    with pytest.raises(ValueError, match="full_only_quality_canary"):
        compare_calibration_arms(
            baseline, _replace_profile_with_provenance(candidate, profile),
        )

    wrong_cache = replace(
        candidate, base=replace(candidate.base, cache_dir="/cache/not-v27541-b1000"),
    )
    with pytest.raises(ValueError, match="profile/base cache binding"):
        compare_calibration_arms(baseline, wrong_cache)

    profile = json.loads(json.dumps(candidate.profile))
    profile["workflow"]["unreviewed_active_switch"] = True
    with pytest.raises(ValueError, match="calibration-size allowlist"):
        compare_calibration_arms(
            baseline, _replace_profile_with_provenance(candidate, profile),
        )


def test_official_profile_loader_projection_is_accepted_with_only_reviewed_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
        load_evaluation_profile,
    )
    from onnx_splitpoint_tool.workflow.start_snapshot import (
        materialize_runtime_profile,
    )

    profile_root = Path(__file__).parents[1] / "profiles"
    run_modes = (
        Path(__file__).parents[1]
        / "onnx_splitpoint_tool/resources/run_modes/default_run_modes.yaml"
    )
    monkeypatch.setenv("ONNX_SPLITPOINT_RUN_MODES_FILE", str(run_modes))
    hardware_registry = tmp_path / "hardware_setups.yaml"
    hardware_registry.write_text(yaml.safe_dump({
        "schema": "onnx-splitpoint/hardware-setups",
        "schema_version": 1,
        "hardware_setups": [{
            "id": "orin_nx_deepx_m1_01",
            "label": "DeepX Orin NX",
            "accelerator": "deepx_m1",
            "provider": "deepx_m1",
            "enabled": True,
            "host": "127.0.0.1",
        }],
        "hardware_groups": {},
        "build_environments": [],
    }, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_HARDWARE_SETUPS_FILE", str(hardware_registry),
    )
    baseline_loaded = load_evaluation_profile(
        profile_root / f"{BASELINE_PROFILE_NAME}.yaml"
    ).raw_profile
    candidate_loaded = load_evaluation_profile(
        profile_root / f"{CANDIDATE_PROFILE_NAME}.yaml"
    ).raw_profile
    baseline_profile, baseline_bindings = materialize_runtime_profile(
        baseline_loaded,
        profile_path=str(profile_root / f"{BASELINE_PROFILE_NAME}.yaml"),
    )
    candidate_profile, candidate_bindings = materialize_runtime_profile(
        candidate_loaded,
        profile_path=str(profile_root / f"{CANDIDATE_PROFILE_NAME}.yaml"),
    )
    canary._validate_resolved_profile(
        baseline_profile, label="B500", expected_count=500,
    )
    canary._validate_resolved_profile(
        candidate_profile, label="B1000", expected_count=1000,
    )
    assert canary._normalized_profile_for_pair(
        baseline_profile
    ) == canary._normalized_profile_for_pair(candidate_profile)

    baseline, candidate = _pair(tmp_path)
    result = compare_calibration_arms(
        _replace_profile_with_provenance(
            baseline, baseline_profile, baseline_bindings,
        ),
        _replace_profile_with_provenance(
            candidate, candidate_profile, candidate_bindings,
        ),
    )
    assert result["standard_plus_ready"] is True


def test_joint_profile_extra_is_rejected_by_frozen_source_shape(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    baseline_profile = json.loads(json.dumps(baseline.profile))
    candidate_profile = json.loads(json.dumps(candidate.profile))
    baseline_profile["joint_unreviewed_runtime_switch"] = True
    candidate_profile["joint_unreviewed_runtime_switch"] = True
    with pytest.raises(ValueError, match="snapshot binding differs"):
        compare_calibration_arms(
            replace(baseline, profile=baseline_profile),
            replace(candidate, profile=candidate_profile),
        )


def test_evaluation_run_envelope_status_and_schema_are_fail_closed(
    tmp_path: Path,
) -> None:
    from onnx_splitpoint_tool.workflow.artifacts import sha256_json

    profile = _profile(count=500)
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": "b500-run",
        "profile_id": BASELINE_PROFILE_NAME,
        "profile_hash": sha256_json(profile),
        "status": "ok",
    }
    canary._validate_run_envelope(
        tmp_path, manifest, label="B500",
        expected_profile_name=BASELINE_PROFILE_NAME, profile=profile,
    )
    for field, value in (
        ("status", "failed"), ("schema", "untrusted/run"),
        ("run_id", ""), ("run_id", 123),
    ):
        with pytest.raises(ValueError, match="EvaluationRun envelope"):
            canary._validate_run_envelope(
                tmp_path, {**manifest, field: value}, label="B500",
                expected_profile_name=BASELINE_PROFILE_NAME, profile=profile,
            )

    target = tmp_path / "run-status-target.json"
    target.write_text("{}", encoding="utf-8")
    (tmp_path / "run_status.json").symlink_to(target.name)
    with pytest.raises(ValueError, match="must not be a symlink"):
        canary._validate_run_envelope(
            tmp_path, manifest, label="B500",
            expected_profile_name=BASELINE_PROFILE_NAME, profile=profile,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_sha256", "f" * 64, "frozen v2.75.40 quality policy"),
        ("statistics.seed", 1, "statistics seed changed"),
        ("statistics.bootstrap_repetitions", 499, "statistics bootstrap_repetitions changed"),
        ("metric_gate_config.non_inferiority_margin", 1.0, "margins/metrics changed"),
        ("metric_gate_config.guardrails.top5_accuracy_margin", 1.0, "margins/metrics changed"),
    ],
)
def test_quality_policy_cannot_be_jointly_relaxed(
    tmp_path: Path, field: str, value: object, message: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    request = json.loads(json.dumps(candidate.deepx_endpoint.request))
    cursor = request
    parts = field.split(".")
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    changed = replace(
        candidate,
        deepx_endpoint=replace(candidate.deepx_endpoint, request=request),
    )
    with pytest.raises(ValueError, match=message):
        compare_calibration_arms(baseline, changed)


def test_deepx_v27540_writer_without_producer_policy_uses_request_authority(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    for arm in (baseline, candidate):
        assert "policy_sha256" not in arm.deepx_endpoint.producer
        assert "endpoint_contract_hash" not in arm.deepx_endpoint.candidate
        assert "endpoint_contract_hash" in arm.deepx_endpoint.request
    assert compare_calibration_arms(baseline, candidate)["standard_plus_ready"] is True


@pytest.mark.parametrize("mutation", ["postprocessor", "runner"])
def test_deepx_quality_execution_contract_cannot_change_with_calibration_size(
    tmp_path: Path, mutation: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    endpoint = candidate.deepx_endpoint
    producer = json.loads(json.dumps(endpoint.producer))
    quality_endpoint = producer["quality_record_endpoint"]
    quality_identity = quality_endpoint["identity"]
    if mutation == "postprocessor":
        quality_identity["postprocessor"] = "unreviewed_topk_implementation"
    else:
        changed_runner = hashlib.sha256(b"changed-deepx-runner").hexdigest()
        quality_identity["implementation_runner_sha256"] = changed_runner
        producer["implementation_runner_sha256"] = changed_runner
    quality_endpoint["sha256"] = canary._canonical_sha256(quality_identity)
    quality_contract = producer["quality_contract"]
    quality_contract["quality_record_endpoint"] = quality_endpoint
    quality_contract["quality_record_endpoint_contract_sha256"] = (
        quality_endpoint["sha256"]
    )
    quality_contract.pop("quality_contract_sha256")
    quality_contract["quality_contract_sha256"] = canary._canonical_sha256(
        quality_contract
    )
    producer["quality_record_endpoint_contract_sha256"] = quality_endpoint[
        "sha256"
    ]
    producer["quality_contract_sha256"] = quality_contract[
        "quality_contract_sha256"
    ]
    producer.pop("producer_identity_sha256")
    producer["producer_identity_sha256"] = canary._canonical_sha256(producer)

    request = json.loads(json.dumps(endpoint.request))
    payload = json.loads(json.dumps(endpoint.candidate))
    for owner in (request, payload):
        owner["quality_contract"] = quality_contract
        owner["quality_contract_sha256"] = quality_contract[
            "quality_contract_sha256"
        ]
        owner["quality_record_endpoint_contract_sha256"] = quality_endpoint[
            "sha256"
        ]
    changed_endpoint = replace(
        endpoint, producer=producer, request=request, candidate=payload,
    )
    with pytest.raises(ValueError, match="quality execution contracts differ"):
        compare_calibration_arms(
            baseline, replace(candidate, deepx_endpoint=changed_endpoint),
        )


@pytest.mark.parametrize(
    ("endpoint_name", "field"),
    [
        ("deepx_endpoint", "decoder_contract_sha256"),
        ("deepx_endpoint", "prepared_input_join_binding"),
        ("trt_endpoint", "runtime_artifact_sha256"),
    ],
)
def test_portable_request_candidate_duplicates_must_match_signed_producer(
    tmp_path: Path, endpoint_name: str, field: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    endpoint = getattr(candidate, endpoint_name)
    request = json.loads(json.dumps(endpoint.request))
    payload = json.loads(json.dumps(endpoint.candidate))
    if field == "prepared_input_join_binding":
        request[field]["binding_verified"] = False
    else:
        payload[field] = "f" * 64
    changed = replace(endpoint, request=request, candidate=payload)
    with pytest.raises(ValueError, match="duplicate differs|contract_sha256 differs"):
        compare_calibration_arms(
            baseline, replace(candidate, **{endpoint_name: changed}),
        )


@pytest.mark.parametrize("field", ["quality_canary_id", "eval_run_id"])
def test_full_only_plan_is_bound_to_profile_and_evaluation_run_authority(
    tmp_path: Path, field: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    endpoint = candidate.deepx_endpoint
    producer = json.loads(json.dumps(endpoint.producer))
    request = json.loads(json.dumps(endpoint.request))
    payload = json.loads(json.dumps(endpoint.candidate))
    changed_value = "invented-canary" if field == "quality_canary_id" else "invented-run"
    for owner in (producer, request, payload):
        owner["full_only_plan_identity"][field] = changed_value
        owner["full_only_plan_identity_sha256"] = canary._canonical_sha256(
            owner["full_only_plan_identity"]
        )
        owner[field] = changed_value
    producer.pop("producer_identity_sha256")
    producer["producer_identity_sha256"] = canary._canonical_sha256(producer)
    changed_endpoint = replace(
        endpoint, producer=producer, request=request, candidate=payload,
    )
    with pytest.raises(ValueError, match="Full-only plan identity differs"):
        compare_calibration_arms(
            baseline, replace(candidate, deepx_endpoint=changed_endpoint),
        )


def test_trt_runtime_input_encoding_is_real_writer_bound_and_arm_invariant(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    endpoint = candidate.trt_endpoint
    assert set(endpoint.producer["preprocessing"]) == {
        "identity", "sha256", "runtime_input_encoding",
        "runtime_input_encoding_sha256",
    }
    producer = json.loads(json.dumps(endpoint.producer))
    preprocessing = producer["preprocessing"]
    preprocessing["runtime_input_encoding"]["input_dtype"] = "float16"
    preprocessing["runtime_input_encoding_sha256"] = canary._canonical_sha256(
        preprocessing["runtime_input_encoding"]
    )
    quality_contract = producer["quality_contract"]
    quality_contract["preprocessing"] = preprocessing
    quality_contract.pop("quality_contract_sha256")
    quality_contract["quality_contract_sha256"] = canary._canonical_sha256(
        quality_contract
    )
    producer["quality_contract_sha256"] = quality_contract[
        "quality_contract_sha256"
    ]
    producer.pop("producer_identity_sha256")
    producer["producer_identity_sha256"] = canary._canonical_sha256(producer)
    request = json.loads(json.dumps(endpoint.request))
    payload = json.loads(json.dumps(endpoint.candidate))
    for owner in (request, payload):
        owner["quality_contract"] = quality_contract
        owner["quality_contract_sha256"] = quality_contract[
            "quality_contract_sha256"
        ]
    changed_endpoint = replace(
        endpoint, producer=producer, request=request, candidate=payload,
    )
    with pytest.raises(ValueError, match="quality execution contracts differ"):
        compare_calibration_arms(
            baseline, replace(candidate, trt_endpoint=changed_endpoint),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("model", "source ONNX differs"),
        ("dataset", "validation manifest_sha256 differs"),
        ("preprocessing", "preprocessing contract differs"),
        ("policy", "producer/request policy differs"),
        ("schema", "producer schema is not frozen"),
    ],
)
def test_trt_control_is_bound_to_deepx_scientific_authority(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    baseline, candidate = _pair(tmp_path)
    producer = json.loads(json.dumps(candidate.trt_endpoint.producer))
    if mutation == "model":
        producer["model"]["source_onnx_sha256"] = "f" * 64
    elif mutation == "dataset":
        producer["dataset"]["manifest_sha256"] = "f" * 64
    elif mutation == "preprocessing":
        producer["preprocessing_contract_sha256"] = "f" * 64
    elif mutation == "policy":
        producer["policy_sha256"] = "f" * 64
    else:
        producer["schema"] = "untrusted/producer"
    changed = replace(
        candidate,
        trt_endpoint=replace(candidate.trt_endpoint, producer=producer),
    )
    with pytest.raises(ValueError, match=message):
        compare_calibration_arms(baseline, changed)


def test_result_algorithm_technical_and_aggregate_decisions_are_exact(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    aggregate = {
        **candidate.deepx_endpoint.result, "decision": "pass",
        "primary": {
            **dict(candidate.deepx_endpoint.result["primary"]),
            "decision": "fail",
        },
    }
    with pytest.raises(ValueError, match="aggregate/component"):
        compare_calibration_arms(
            baseline,
            replace(
                candidate,
                deepx_endpoint=replace(candidate.deepx_endpoint, result=aggregate),
            ),
        )

    for field, value in (
        ("algorithm_version", "old-evaluator"),
        ("technical_status", None),
    ):
        result = dict(candidate.deepx_endpoint.result)
        if value is None:
            result.pop(field)
        else:
            result[field] = value
        with pytest.raises(ValueError, match="not exact/complete"):
            compare_calibration_arms(
                baseline,
                replace(
                    candidate,
                    deepx_endpoint=replace(candidate.deepx_endpoint, result=result),
                ),
            )


def test_direct_trt_guardrail_uses_unchanged_single_policy_seed(tmp_path: Path) -> None:
    baseline, candidate = _pair(tmp_path)
    result = compare_calibration_arms(baseline, candidate)
    guard = result["authoritative_b1000_vs_setup_local_trt_guardrail"]
    assert guard["top1"]["seed"] == 20260710
    assert guard["top5"]["seed"] == 20260710
    assert result["paired_b1000_minus_b500"]["top5"]["seed"] == 20260711


def test_manifest_requires_actual_one_item_per_selected_class(tmp_path: Path) -> None:
    path, contract = _sealed_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][-1]["class_name"] = payload["items"][0]["class_name"]
    payload["provisioning_selection"]["selected_class_count"] = 499
    contract = _reseal_manifest(path, contract, payload)
    with pytest.raises(ValueError, match="500 distinct classes"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )


def test_manifest_root_inventory_rejects_extra_images_and_symlinks(tmp_path: Path) -> None:
    path, contract = _sealed_manifest(tmp_path / "extra")
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = Path(payload["root"]) / "n99999999" / "extra.JPEG"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"extra")
    with pytest.raises(ValueError, match="root inventory differs"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )

    path, contract = _sealed_manifest(tmp_path / "symlink")
    payload = json.loads(path.read_text(encoding="utf-8"))
    first = Path(payload["root"]) / payload["items"][0]["relative_path"]
    link = Path(payload["root"]) / "n99999998" / "link.JPEG"
    link.parent.mkdir(parents=True)
    link.symlink_to(first)
    with pytest.raises(ValueError, match="contains a symlink"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )


def test_explicit_manifest_leaf_symlink_is_rejected_before_resolution(
    tmp_path: Path,
) -> None:
    path, contract = _sealed_manifest(tmp_path)
    link = tmp_path / "manifest-link.json"
    link.symlink_to(path)
    with pytest.raises(FileNotFoundError, match="calibration manifest not found"):
        canary._resolve_manifest_path(
            root=tmp_path, profile={}, contract=contract,
            explicit_path=link, label="B500",
        )


def test_quality_candidate_leaf_symlink_is_rejected_before_resolution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    request_dir = root / "models/resnet50/evidence/deepx_m1_full"
    request_dir.mkdir(parents=True)
    records = _records(410, 480)
    endpoint = _endpoint(
        tmp_path, source_run_id="deepx_m1_full", backend="deepx_m1",
        records=records,
    )
    producer = json.loads(json.dumps(endpoint.producer))
    ids_sha = canary.image_ids_fingerprint([row["image_id"] for row in records])
    ground_truth_sha = canary.json_fingerprint([{
        "image_id": row["image_id"], "label_id": int(row["label_id"]),
    } for row in records])
    producer["dataset"]["image_ids_sha256"] = ids_sha
    producer["dataset"]["ground_truth_sha256"] = ground_truth_sha
    producer.pop("producer_identity_sha256")
    producer_sha = canary._canonical_sha256(producer)
    producer["producer_identity_sha256"] = producer_sha
    candidate = {
        **endpoint.candidate,
        "schema": "onnx-splitpoint/task-quality-candidate-input",
        "schema_version": 1,
        "task": "classification", "variant": "full",
        "pairing_key": "image_id", "producer_provenance_required": True,
        "producer_identity": producer,
        "producer_identity_sha256": producer_sha,
        "record_count": 500, "records": list(records),
    }
    candidate_path = request_dir / "full_candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    request = {
        **endpoint.request,
        "expected_image_ids_sha256": ids_sha,
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "schema_version": 1, "status": "pending_central_evaluation",
        "execution_location": "management_node",
        "producer_provenance_required": True,
        "full_only_plan_identity_required": True,
        "model_id": "resnet50", "source_run_id": "deepx_m1_full",
        "variant": "full", "task": "classification",
        "execution_role": "full_quality_only", "backend": "deepx_m1",
        "setup_id": "orin_nx_deepx_m1_01",
        "performance_claims_emitted": False,
        "reference_record_count": 500, "record_count": 500,
        "expected_image_ids": [row["image_id"] for row in records],
        "producer_identity": producer,
        "producer_identity_sha256": producer_sha,
        "candidate": {
            "path": candidate_path.name,
            "size_bytes": candidate_path.stat().st_size,
            "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        },
    }
    request_path = request_dir / "full_request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    result = {
        **endpoint.result,
        "source_request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
    }
    loaded = canary._load_quality_endpoint(
        root, source_run_id="deepx_m1_full", expected_backend="deepx_m1",
        expected_records=500, result=result,
    )
    assert loaded.candidate_path == candidate_path

    original_request = request_path.read_bytes()
    for mutation in ("missing_embedded", "conflicting_request"):
        tampered_request = json.loads(original_request)
        if mutation == "missing_embedded":
            tampered_request["producer_identity"].pop("producer_identity_sha256")
        else:
            tampered_request["producer_identity_sha256"] = "f" * 64
        request_path.write_text(json.dumps(tampered_request), encoding="utf-8")
        with pytest.raises(ValueError, match="producer identity"):
            canary._load_quality_endpoint(
                root, source_run_id="deepx_m1_full", expected_backend="deepx_m1",
                expected_records=500, result=result,
            )
        request_path.write_bytes(original_request)

    tampered_request = json.loads(original_request)
    unsigned_producer = dict(tampered_request["producer_identity"])
    unsigned_producer.pop("producer_identity_sha256")
    unsigned_producer.pop("backend")
    changed_sha = canary._canonical_sha256(unsigned_producer)
    tampered_request["producer_identity"] = {
        **unsigned_producer, "producer_identity_sha256": changed_sha,
    }
    tampered_request["producer_identity_sha256"] = changed_sha
    request_path.write_text(json.dumps(tampered_request), encoding="utf-8")
    with pytest.raises(ValueError, match="exact Full-only quality evidence: backend"):
        canary._load_quality_endpoint(
            root, source_run_id="deepx_m1_full", expected_backend="deepx_m1",
            expected_records=500, result=result,
        )
    request_path.write_bytes(original_request)

    target = request_dir / "candidate-target.json"
    candidate_path.rename(target)
    candidate_path.symlink_to(target.name)
    with pytest.raises(ValueError, match="candidate escapes"):
        canary._load_quality_endpoint(
            root, source_run_id="deepx_m1_full", expected_backend="deepx_m1",
            expected_records=500, result=result,
        )


def test_provisioning_population_and_selection_semantics_are_count_specific(
    tmp_path: Path,
) -> None:
    baseline, candidate = _pair(tmp_path)
    assert baseline.calibration_manifest["population_count"] == 500
    assert candidate.calibration_manifest["population_count"] == 1000
    changed = json.loads(json.dumps(candidate.calibration_manifest))
    changed["provisioning_selection"]["strategy"] = "random"
    with pytest.raises(ValueError, match="selection semantics"):
        compare_calibration_arms(
            baseline, replace(candidate, calibration_manifest=changed),
        )

    path, contract = _sealed_manifest(tmp_path / "selection-extra")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provisioning_selection"]["unreviewed_sampling_axis"] = True
    contract = _reseal_manifest(path, contract, payload)
    with pytest.raises(ValueError, match="provisioning semantics"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )

    path, contract = _sealed_manifest(tmp_path / "manifest-extra")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unreviewed_sampling_axis"] = {"enabled": True}
    contract = _reseal_manifest(path, contract, payload)
    with pytest.raises(ValueError, match="manifest shape is not frozen"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )

    path, contract = _sealed_manifest(tmp_path / "item-extra")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["preprocessing_override"] = "unreviewed"
    contract = _reseal_manifest(path, contract, payload)
    with pytest.raises(ValueError, match="manifest item shape is not frozen"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )

    path, contract = _sealed_manifest(tmp_path / "wrong-corpus")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_id"] = "lookalike-train-calibration"
    contract = _reseal_manifest(path, contract, payload)
    with pytest.raises(ValueError, match="dataset_id mismatch"):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "selection manifest is unavailable"),
        ("wrong_file_sha", "selection manifest file hash mismatch"),
        ("competition", "selection receipt is not frozen"),
        ("source_item", "selection item is invalid"),
    ],
)
def test_private_imagenet_selection_receipt_is_byte_and_item_bound(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    path, contract = _sealed_manifest(tmp_path / mutation)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    provisioning = manifest["provisioning_selection"]
    selection_path = path.parent / provisioning["selection_manifest"]
    if mutation == "missing":
        selection_path.unlink()
    elif mutation == "wrong_file_sha":
        provisioning["selection_manifest_sha256"] = "f" * 64
        contract = _reseal_manifest(path, contract, manifest)
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if mutation == "competition":
            selection["competition"] = "lookalike-imagenet-corpus"
        else:
            selection["items"][0]["source_relative_path"] = (
                "lookalike/n00000000.JPEG"
            )
        provisioning["selection_manifest_sha256"] = _reseal_selection_receipt(
            selection_path, selection,
        )
        contract = _reseal_manifest(path, contract, manifest)
    with pytest.raises(ValueError, match=message):
        canary._load_manifest(
            path, contract=contract, expected_count=500, label="B500",
        )
