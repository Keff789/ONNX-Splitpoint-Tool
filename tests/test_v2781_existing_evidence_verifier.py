from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from onnx_splitpoint_tool.existing_evidence_verifier import (
    ExistingEvidenceError,
    create_receipt,
    verify_receipt,
)
from onnx_splitpoint_tool.quality_cache import (
    canonical_json,
    image_ids_fingerprint,
    json_fingerprint,
    prediction_fingerprint,
)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ExistingEvidenceFixture:
    run_dir: Path
    canary_dir: Path
    self_reference: Path
    output: Path
    benchmark_set: Path
    onnx: Path
    hef: Path
    hef_receipt: Path
    quality_request: Path
    candidate: Path
    boundary_manifest: Path
    output_manifest: Path
    native_report: Path
    completed_result: Path
    model_sha256: str
    hef_sha256: str
    preprocessing_sha256: str
    endpoint_sha256: str


def _implementation_artifacts() -> dict[str, dict[str, str]]:
    # These are the three adapter sources sealed by the real positive YOLO11
    # self-reference evidence.  The fixture deliberately uses their package
    # relative paths instead of test-only stand-ins.
    paths = {
        "harness_base": "onnx_splitpoint_tool/runners/harness/base.py",
        "native_detection_postprocess": (
            "onnx_splitpoint_tool/native_detection_postprocess.py"
        ),
        "yolo_harness": "onnx_splitpoint_tool/runners/harness/yolo.py",
    }
    root = Path(__file__).resolve().parents[1]
    return {
        name: {
            "relative_path": relative,
            "sha256": _file_sha256(root / relative),
        }
        for name, relative in paths.items()
    }


def _build_existing_evidence(
    tmp_path: Path,
    *,
    label: str = "a",
    run_id: str | None = None,
    output_name: str = "verification-receipt.json",
) -> ExistingEvidenceFixture:
    run_id = run_id or f"evaluation-run-{label}"
    run_dir = tmp_path / f"run-{label}"
    canary_dir = tmp_path / f"canary-{label}"
    self_reference = tmp_path / f"self-reference-{label}.json"
    output = tmp_path / output_name
    benchmark_set = (
        run_dir / "models/yolo11l/benchmark_set/legacy_suite"
    )

    onnx = benchmark_set / "models/yolo11l.onnx"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"real-layout-yolo11l-full-onnx\n")
    model_sha256 = _file_sha256(onnx)

    hef = benchmark_set / "hailo/hailo8/full/compiled.hef"
    hef.parent.mkdir(parents=True, exist_ok=True)
    hef.write_bytes(b"real-layout-yolo11l-hailo8-full-hef\n")
    hef_sha256 = _file_sha256(hef)

    preprocessing = {
        "schema": "onnx-splitpoint/image-preprocessing-contract",
        "schema_version": 2,
        "contract_scope": "prepared_rgb_uint8_semantics",
        "task": "detection",
        "preprocess_mode": "letterbox",
        "spatial_transform": "centered_letterbox",
        "target_hw": [640, 640],
        "color_space": "RGB",
        "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear",
        "resize_rounding": "python_round_ties_to_even",
        "placement": "floor_top_left_remainder_bottom_right",
        "pad_value": 114,
        "letterbox_pad_value": 114,
        "image_scale": "norm",
        "letterbox": True,
    }
    preprocessing_sha256 = _canonical_sha256(preprocessing)
    decoder = {
        "schema": "onnx-splitpoint/detection-decoder-contract",
        "schema_version": 1,
        "adapter_id": "vendored_yolo_harness_with_local_fallback_v1",
        "source_output_format": "ultralytics_decoded",
        "confidence_threshold": 0.25,
    }
    decoder_sha256 = _canonical_sha256(decoder)
    nms = {
        "schema": "onnx-splitpoint/detection-nms-contract",
        "schema_version": 1,
        "adapter_id": "vendored_yolo_harness_with_local_fallback_v1",
        "iou_threshold": 0.45,
        "max_detections": 300,
    }
    nms_sha256 = _canonical_sha256(nms)
    implementation_artifacts = _implementation_artifacts()
    raw_output_tensor_signature = {
        "tensor_count": 1,
        "tensors": [{
            "index": 0,
            "name": "output0",
            "shape": [1, 3, 4],
            "rank": 3,
            "dtype": "float32",
        }],
    }
    postprocess_invariant_identity = {
        "schema": "onnx-splitpoint/frozen-native-detection-postprocess",
        "schema_version": 1,
        "task": "detection",
        "model_id": "yolo11l",
        "model_family": "yolo11",
        "decoder_format": "ultralytics_regcls",
        "decoder_id": "yolo11_regcls_dfl16_classaware_nms_v1",
        "class_aware": True,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "host_postprocess_frozen": True,
        "implementation_artifacts": implementation_artifacts,
        "raw_output_tensor_signature": raw_output_tensor_signature,
    }
    postprocess_invariant_sha256 = _canonical_sha256(
        postprocess_invariant_identity
    )
    postprocess_contract = {
        **postprocess_invariant_identity,
        "contract_sha256": decoder_sha256,
        "invariant_identity": postprocess_invariant_identity,
        "invariant_contract_sha256": postprocess_invariant_sha256,
    }
    endpoint_contract = {
        "schema": "onnx-splitpoint/completed-task-comparison-endpoint",
        "schema_version": 2,
        "model_id": "yolo11l",
        "model_family": "yolo11",
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "coordinate_space": "original_image_xyxy_pixels",
        "output_record_format": "xyxy_score_class",
        "class_aware": True,
        "score_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
    }
    endpoint_sha256 = _canonical_sha256(endpoint_contract)
    endpoint_contract["endpoint_contract_complete"] = True
    endpoint_contract["endpoint_contract_hash"] = endpoint_sha256
    endpoint_contract["output_endpoint_id"] = (
        f"detection:decoded_nms:comparison:{endpoint_sha256}"
    )

    hef_receipt = hef.parent / "hailo_hef_build_receipt.json"
    _write_json(hef_receipt, {
        "schema": "onnx-splitpoint/hailo-hef-build-receipt/v2",
        "hw_arch": "hailo8",
        "source_onnx_sha256": model_sha256,
        "compiler_onnx_sha256": model_sha256,
        "hef_sha256": hef_sha256,
        "hef_size_bytes": hef.stat().st_size,
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha256,
    })

    image_ids = [f"{index:012d}.jpg" for index in range(500)]
    image_ids_sha256 = image_ids_fingerprint(image_ids)
    ground_truth_identity = [
        {"image_id": image_id, "ground_truth": []}
        for image_id in image_ids
    ]
    ground_truth_sha256 = json_fingerprint(ground_truth_identity)
    annotations_sha256 = json_fingerprint({
        "schema": "quality-annotations-v1",
        "records": sorted(ground_truth_identity, key=canonical_json),
    })
    dataset_manifest = {
        "schema": "onnx-splitpoint/dataset-manifest",
        "schema_version": 1,
        "dataset_id": "coco-val500-development",
        "image_count": 500,
        "image_ids": image_ids,
        "image_ids_sha256": image_ids_sha256,
    }
    dataset_path = (
        run_dir / "campaign/inputs/dataset_detection_validation.json"
    )
    _write_json(dataset_path, dataset_manifest)
    # The run registry is not the canonical 500-item quality manifest.  The
    # real Quality Contract records only that external manifest's content
    # identity and basename, not a safely resolvable physical run path.
    dataset_sha256 = hashlib.sha256(
        b"fixture-external-canonical-500-item-manifest"
    ).hexdigest()
    assert dataset_sha256 != _file_sha256(dataset_path)
    dataset_identity = {
        "coordinate_space": "original_image_xyxy_pixels",
        "ground_truth_sha256": ground_truth_sha256,
        "image_count": 500,
        "image_ids_sha256": image_ids_sha256,
        "manifest_name": "manifest.json",
        "manifest_sha256": dataset_sha256,
    }
    quality_contract: dict[str, Any] = {
        "schema": "onnx-splitpoint/central-detection-quality-contract",
        "schema_version": 1,
        "task": "detection",
        "contract_scope": "canonical_quality_record_semantics",
        "quality_record_endpoint": (
            endpoint_contract["output_endpoint_id"]
        ),
        "quality_record_endpoint_contract_sha256": endpoint_sha256,
        "source_endpoint_is_raw": False,
        "source_endpoint_role": "completed_task_comparison_endpoint",
        "canonical_coordinate_space": "original_image_xyxy_pixels",
        "canonical_record_endpoint": (
            "decoded_xyxy_score_class_detections"
        ),
        "dataset": dataset_identity,
        "model": {
            "artifact_name": onnx.name,
            "sha256": model_sha256,
        },
        "preprocessing": {
            "identity": preprocessing,
            "sha256": preprocessing_sha256,
        },
        "decoder": {"identity": decoder, "sha256": decoder_sha256},
        "nms": {"identity": nms, "sha256": nms_sha256},
    }
    quality_contract_sha256 = _canonical_sha256(quality_contract)
    quality_contract["quality_contract_sha256"] = quality_contract_sha256

    reference_records = [
        {"image_id": image_id, "reference": [], "ground_truth": []}
        for image_id in image_ids
    ]
    reference_predictions_sha256 = prediction_fingerprint(
        reference_records,
        payload_field="reference",
    )
    reference_path = run_dir / (
        "quality_management/references/yolo11l/"
        "canonical_cpu_reference.json"
    )
    _write_json(reference_path, {
        "schema": "onnx-splitpoint/task-quality-reference-input",
        "schema_version": 1,
        "task": "detection",
        "model_id": "yolo11l",
        "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True,
        "pairing_key": "image_id",
        "record_count": 500,
        "image_ids_sha256": image_ids_sha256,
        "quality_contract": quality_contract,
        "quality_contract_sha256": quality_contract_sha256,
        "prediction_sha256": reference_predictions_sha256,
        "records": reference_records,
    })
    reference_file_sha256 = _file_sha256(reference_path)
    policy = {
        "schema": "onnx-splitpoint/task-quality-policy",
        "schema_version": 3,
        "name": "fixture_task_quality",
        "profile_id": "fixture_task_quality",
        "native_self_reference_policy_id": (
            "class_aware_iou50_postnms_v2"
        ),
        "native_self_reference_min_match": 0.8,
        "native_self_reference_min_mean_iou": 0.85,
        "native_self_reference_iou_threshold": 0.5,
        "native_self_reference_confidence_threshold": 0.25,
        "native_self_reference_denominator": "reference_detections",
        "native_self_reference_class_aware": True,
        "numerical_similarity_required_for_claim": True,
    }
    policy_sha256 = _canonical_sha256(policy)

    def write_quality_source(
        backend: str,
    ) -> tuple[Path, Path, str]:
        is_hailo = backend == "hailo8"
        source_run_id = "hailo8" if is_hailo else "native_full_tensorrt"
        case_id = "b602" if is_hailo else "full"
        result_leaf = "b602/results_hailo8" if is_hailo else (
            "results_native_full_tensorrt"
        )
        request_dir = run_dir / (
            "models/yolo11l/benchmark_results/quality_inputs/"
            "orin_nx_hailo8_01/results/"
            f"{result_leaf}/task_quality_inputs"
        )
        # Distinct physical prediction sets ensure that accidentally sharing
        # the Hailo request with TensorRT cannot satisfy the fixture.
        detection = {
            "class_id": 1 if is_hailo else 2,
            "score": 0.9,
            "x1": 1.0,
            "y1": 2.0,
            "x2": 10.0,
            "y2": 12.0,
        }
        candidate_records = [
            {
                "image_id": image_id,
                "candidate": [detection] if index == 0 else [],
                "ground_truth": [],
            }
            for index, image_id in enumerate(image_ids)
        ]
        candidate_predictions_sha256 = prediction_fingerprint(
            candidate_records,
            payload_field="candidate",
        )
        runtime_precision = (
            f"hailo_hef_sha256:{hef_sha256}" if is_hailo else "fp16"
        )
        producer_backend = "hailo8" if is_hailo else "native_tensorrt"
        quality_canary_id = (
            "hailo8_full" if is_hailo else "tensorrt_at_hailo8_full"
        )
        full_only_plan_identity = {
            "schema": "onnx-splitpoint/full-only-plan-identity",
            "schema_version": 1,
            "model_id": "yolo11l",
            "backend": producer_backend,
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        }
        full_only_plan_identity_sha256 = _canonical_sha256(
            full_only_plan_identity
        )
        completion_fields: dict[str, Any] = {}
        if is_hailo:
            completion_body = {
                "schema": (
                    "onnx-splitpoint/hailo-full-host-tail-completion"
                ),
                "schema_version": 1,
                "model_id": "yolo11l",
                "task": "detection",
                "variant": "full",
                "completed_endpoint": (
                    "decoded_xyxy_score_class_detection_records"
                ),
                "completed_task_endpoint_contract_hash": endpoint_sha256,
                "hailo_hef_sha256": hef_sha256,
                "frozen_postprocess_invariant_contract_sha256": (
                    postprocess_invariant_sha256
                ),
            }
            completion_sha256 = _canonical_sha256(completion_body)
            completion_contract = {
                **completion_body,
                "contract_sha256": completion_sha256,
            }
            endpoint_attestation = {
                "schema": (
                    "onnx-splitpoint/completed-task-endpoint-attestation"
                ),
                "schema_version": 1,
                "attested": True,
                "status": "passed",
                "model_id": "yolo11l",
                "task": "detection",
                "stage": "decoded_nms",
                "completed_endpoint_contract": endpoint_contract,
                "endpoint_contract_hash": endpoint_sha256,
                "output_endpoint_id": endpoint_contract[
                    "output_endpoint_id"
                ],
            }
            completion_fields = {
                "candidate_execution_completion_contract": (
                    completion_contract
                ),
                "candidate_execution_completion_contract_sha256": (
                    completion_sha256
                ),
                "completed_task_endpoint_contract": endpoint_contract,
                "completed_task_endpoint_contract_hash": endpoint_sha256,
                "completed_task_output_endpoint_id": endpoint_contract[
                    "output_endpoint_id"
                ],
                "completed_task_endpoint_attestation": (
                    endpoint_attestation
                ),
                "completed_task_endpoint_attestation_sha256": (
                    _canonical_sha256(endpoint_attestation)
                ),
                "quality_join_endpoint": "completed_task_decoded_nms",
            }
        candidate = request_dir / "full_candidate.json"
        _write_json(candidate, {
            "schema": "onnx-splitpoint/task-quality-candidate-input",
            "schema_version": 1,
            "task": "detection",
            "variant": "full",
            "model_id": "yolo11l",
            "case_id": case_id,
            "eval_run_id": run_id,
            "source_run_id": source_run_id,
            "setup_id": "orin_nx_hailo8_01",
            "backend": producer_backend,
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
            "quality_canary_id": quality_canary_id,
            "pairing_key": "image_id",
            "runtime_precision_identity": runtime_precision,
            "quality_contract": quality_contract,
            "quality_contract_sha256": quality_contract_sha256,
            "full_only_plan_identity_required": True,
            "full_only_plan_identity": full_only_plan_identity,
            "full_only_plan_identity_sha256": (
                full_only_plan_identity_sha256
            ),
            "record_count": 500,
            "image_ids_sha256": image_ids_sha256,
            "prediction_sha256": candidate_predictions_sha256,
            "records": candidate_records,
            **completion_fields,
        })
        request = request_dir / "full_request.json"
        request_payload: dict[str, Any] = {
            "schema": "onnx-splitpoint/central-quality-evaluation-request",
            "schema_version": 1,
            "status": "pending_central_evaluation",
            "requested_by": "central_management",
            "eval_run_id": run_id,
            "source_run_id": source_run_id,
            "setup_id": "orin_nx_hailo8_01",
            "model_id": "yolo11l",
            "backend": producer_backend,
            "variant": "full",
            "task": "detection",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
            "quality_canary_id": quality_canary_id,
            "execution_location": "management_node",
            "pairing_key": "image_id",
            "record_count": 500,
            "reference_record_count": 500,
            "expected_image_ids": image_ids,
            "expected_image_ids_sha256": image_ids_sha256,
            "image_ids_sha256": image_ids_sha256,
            "candidate": {
                "path": candidate.name,
                "sha256": _file_sha256(candidate),
                "size_bytes": candidate.stat().st_size,
                "prediction_sha256": candidate_predictions_sha256,
            },
            "reference": {
                "expected_image_ids": image_ids,
                "expected_image_ids_sha256": image_ids_sha256,
                "record_count": 500,
                "quality_contract_sha256": quality_contract_sha256,
                "reference_role": "canonical_cpu_ort",
                "semantic_reference_only": True,
                "required": True,
                "source": "management_cpu_reference",
            },
            "quality_contract": quality_contract,
            "quality_contract_sha256": quality_contract_sha256,
            "full_only_plan_identity_required": True,
            "full_only_plan_identity": full_only_plan_identity,
            "full_only_plan_identity_sha256": (
                full_only_plan_identity_sha256
            ),
            "policy_sha256": policy_sha256,
            "preprocessing_contract_sha256": preprocessing_sha256,
            "decoder_contract_sha256": decoder_sha256,
            "nms_contract_sha256": nms_sha256,
            "completed_task_endpoint_contract": endpoint_contract,
            "completed_task_endpoint_contract_hash": endpoint_sha256,
            "completed_task_output_endpoint_id": (
                endpoint_contract["output_endpoint_id"]
            ),
            "runtime_precision_identity": runtime_precision,
            **completion_fields,
        }
        _write_json(request, request_payload)
        return request, candidate, candidate_predictions_sha256

    quality_sources = {
        backend: write_quality_source(backend)
        for backend in ("hailo8", "tensorrt")
    }
    quality_request, candidate, _hailo_candidate_prediction_sha256 = (
        quality_sources["hailo8"]
    )

    def quality_row(backend: str) -> dict[str, Any]:
        is_hailo = backend == "hailo8"
        source_run_id = (
            "hailo8" if is_hailo else "native_full_tensorrt"
        )
        request, _candidate, candidate_predictions_sha256 = quality_sources[
            backend
        ]
        row_decision = "fail" if is_hailo else "pass"

        def quality_component(
            metric: str,
            candidate_value: float,
            reference_value: float,
        ) -> dict[str, Any]:
            delta = candidate_value - reference_value
            ci_low = delta if is_hailo else delta - 0.001
            ci_high = delta if is_hailo else delta + 0.001
            return {
                "metric": metric,
                "candidate": candidate_value,
                "reference": reference_value,
                "delta": delta,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "margin": 0.01,
                "n": 500,
                "decision": row_decision,
                "status": row_decision,
            }

        primary = quality_component(
            "coco_ap_50_95",
            0.47 if is_hailo else 0.485,
            0.49,
        )
        ap50 = quality_component(
            "ap50",
            0.57 if is_hailo else 0.585,
            0.59,
        )
        ap75 = quality_component(
            "ap75",
            0.42 if is_hailo else 0.435,
            0.44,
        )
        row: dict[str, Any] = {
            "schema": "onnx-splitpoint/management-paired-quality-result",
            "schema_version": 1,
            "status": "completed",
            "technical_status": "completed",
            "scientific_status": row_decision,
            "decision": row_decision,
            "task": "detection",
            "model_id": "yolo11l",
            "model_sha256": model_sha256,
            "source_model_sha256": model_sha256,
            "variant": "full",
            "backend": backend,
            "case_id": "b602" if backend == "hailo8" else "full",
            # eval_run_id is the EvaluationRun binding. source_run_id names
            # the physical producer and intentionally differs by backend.
            "eval_run_id": run_id,
            "source_run_id": source_run_id,
            "source_setup_id": "orin_nx_hailo8_01",
            "execution_role": "full_quality_only",
            "quality_canary_id": (
                "hailo8_full"
                if backend == "hailo8"
                else "tensorrt_at_hailo8_full"
            ),
            "n": 500,
            "source_request": str(request.relative_to(run_dir)),
            "source_request_sha256": f"sha256:{_file_sha256(request)}",
            "candidate_predictions_sha256": candidate_predictions_sha256,
            "reference_predictions_sha256": reference_predictions_sha256,
            "management_cpu_reference": {
                "reference_path": str(reference_path.resolve()),
                "reference_sha256": reference_file_sha256,
                "prediction_sha256": reference_predictions_sha256,
                "record_count": 500,
            },
            "runtime_precision_identity": (
                f"hailo_hef_sha256:{hef_sha256}"
                if backend == "hailo8"
                else "fp16"
            ),
            "full_only_plan_identity_sha256": _canonical_sha256({
                "schema": "onnx-splitpoint/full-only-plan-identity",
                "schema_version": 1,
                "model_id": "yolo11l",
                "backend": (
                    "hailo8"
                    if backend == "hailo8"
                    else "native_tensorrt"
                ),
                "variant": "full",
                "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            }),
            "quality_contract_sha256": quality_contract_sha256,
            "policy_sha256": policy_sha256,
            "preprocessing_contract_sha256": preprocessing_sha256,
            "decoder_contract_sha256": decoder_sha256,
            "nms_contract_sha256": nms_sha256,
            "quality_record_endpoint_contract_sha256": endpoint_sha256,
            "validation_dataset_sha256": dataset_sha256,
            "validation_ground_truth_sha256": ground_truth_sha256,
            "annotations_sha256": annotations_sha256,
            "validation_image_ids_sha256": image_ids_sha256,
            "completed_task_endpoint_contract_hash": endpoint_sha256,
            "completed_task_output_endpoint_id": (
                endpoint_contract["output_endpoint_id"]
            ),
            "metric_gate_config": {
                "primary_metric": "coco_ap_50_95",
                "non_inferiority_margin": 0.01,
                "guardrails": {
                    "ap50_margin": 0.01,
                    "ap75_margin": 0.01,
                },
            },
            "configured_guardrails": ["ap50", "ap75"],
            "guardrail_contract_complete": True,
            "primary": primary,
            "guardrails": {"ap50": ap50, "ap75": ap75},
            "task_quality_metric": primary["metric"],
            "task_quality_candidate": primary["candidate"],
            "task_quality_reference": primary["reference"],
            "task_quality_delta": primary["delta"],
            "task_quality_ap50_candidate": ap50["candidate"],
            "task_quality_ap50_reference": ap50["reference"],
            "task_quality_ap50_delta": ap50["delta"],
            "task_quality_ap50_decision": ap50["decision"],
            "task_quality_ap75_candidate": ap75["candidate"],
            "task_quality_ap75_reference": ap75["reference"],
            "task_quality_ap75_delta": ap75["delta"],
            "task_quality_ap75_decision": ap75["decision"],
        }
        if backend == "tensorrt":
            row["source_onnx_sha256"] = model_sha256
        return row

    _write_json(
        run_dir / "quality_management/central_quality_summary.json",
        {
            "schema": "onnx-splitpoint/central-quality-summary",
            "schema_version": 1,
            "status": "ok",
            "technical_status": "ok",
            "scientific_status": "fail",
            "aggregate_quality_decision": "fail",
            "quality_decision": "fail",
            "scientific_pass": False,
            "decision_counts": {"fail": 1, "pass": 1},
            "request_count": 2,
            "quality_result_count": 2,
            "completed_count": 2,
            "summary_only_full_quality_expected_count": 2,
            "summary_only_full_quality_completed_count": 2,
            "results": [quality_row("hailo8"), quality_row("tensorrt")],
        },
    )

    _write_json(run_dir / "run_manifest.json", {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        # A retrospective receipt must not promote these original outcomes.
        "status": "partial",
        "technical_status": "partial",
        "scientific_status": "not_evaluated",
        "aggregate_quality_decision": "fail",
        "models": {
            "yolo11l": {
                "model_id": "yolo11l",
                "family": "yolo11",
                "family_id": "yolo11",
                "evaluation_role": "development",
                "generalization_scope": "development",
                "model_sha256": f"sha256:{model_sha256}",
            },
        },
    })

    _write_json(run_dir / "reports/results_bundle_manifest.json", {
        "schema": "onnx-splitpoint/results-bundle-manifest",
        "schema_version": 2,
        "run_id": run_id,
        "missing_reports": [],
        "missing_outputs": {},
    })
    scientific_report = run_dir / "reports/scientific/scientific_report.json"
    _write_json(scientific_report, {
        "schema": "onnx-splitpoint/scientific-report",
        "schema_version": 1,
        "run_id": run_id,
        "scientific_status": "fail",
    })
    _write_json(run_dir / "reports/scientific/report_manifest.json", {
        "schema": "onnx-splitpoint/scientific-report-manifest",
        "schema_version": 2,
        "artifacts": [{
            "path": scientific_report.name,
            "sha256": _file_sha256(scientific_report),
            "size_bytes": scientific_report.stat().st_size,
        }],
    })
    _write_json(benchmark_set / "output_contracts.json", {
        "schema": "onnx-splitpoint/output-contracts",
        "schema_version": 1,
        "model_id": "yolo11l",
        "task": "detection",
        "contracts": [{
            "schema": "onnx-splitpoint/output-contract",
            "schema_version": 1,
            "model_id": "yolo11l",
            "backend": "hailo8",
            "variant": "full",
            "task": "detection",
            "recorded_artifact_sha256": hef_sha256,
            "recorded_artifact_size_bytes": hef.stat().st_size,
        }],
    })

    indexed_paths = (
        "quality_management/central_quality_summary.json",
        "reports/results_bundle_manifest.json",
        "reports/scientific/report_manifest.json",
        "reports/scientific/scientific_report.json",
        (
            "models/yolo11l/benchmark_set/legacy_suite/"
            "output_contracts.json"
        ),
        (
            "models/yolo11l/benchmark_set/legacy_suite/"
            "hailo/hailo8/full/compiled.hef"
        ),
        (
            "models/yolo11l/benchmark_set/legacy_suite/"
            "hailo/hailo8/full/hailo_hef_build_receipt.json"
        ),
    )
    _write_json(run_dir / "artifact_index.json", {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": run_id,
        "artifacts": [
            {
                "kind": "fixture_artifact",
                "path": relative,
                "producer_stage": "fixture",
                "sha256": f"sha256:{_file_sha256(run_dir / relative)}",
                "size_bytes": (run_dir / relative).stat().st_size,
            }
            for relative in indexed_paths
        ],
    })

    result_dir = canary_dir / "results/01_yolo11l/00"
    dump_dir = result_dir / "dump"
    image = canary_dir / "inputs/000000000632.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fixture-canary-image\n")
    image_sha256 = _file_sha256(image)

    boundary_manifest = dump_dir / "native_full_input_manifest.json"
    input_dump = dump_dir / "input_rgb_uint8.bin"
    input_dump.parent.mkdir(parents=True, exist_ok=True)
    input_dump.write_bytes(b"\x01\x02\x03" * 16)
    _write_json(boundary_manifest, {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "model": "yolo11l",
        "task": "detection",
        "case": "full",
        "setup_id": "orin_nx_hailo8_01",
        "input_image": str(image.resolve()),
        "input_image_sha256": image_sha256,
        # The official producer records this as absolute provenance.  The
        # verifier must still open only the fixed basename below dump_root.
        "input_dump": str(input_dump.resolve()),
        "input_dump_sha256": _file_sha256(input_dump),
        "input_dump_bytes": input_dump.stat().st_size,
        "preprocessing_contract": preprocessing,
        "preprocessing_contract_sha256": preprocessing_sha256,
    })

    resolution = {
        "model_id": "yolo11l",
        "backend": "hailo8",
        "variant": "full",
        "artifact_binding_status": "verified",
        "artifact_sha256": hef_sha256,
        "compiler_onnx_sha256": model_sha256,
        "source_onnx_sha256": model_sha256,
        "preprocessing_contract_sha256": preprocessing_sha256,
    }
    detections = [{
        "class_id": 59,
        "score": 0.94,
        "x1": 1.0,
        "y1": 273.0,
        "x2": 400.0,
        "y2": 478.0,
    }]
    detections_sha256 = _canonical_sha256(detections)
    completed_result_payload = {
        "schema": "onnx-splitpoint/frozen-completed-detection-result-artifact",
        "schema_version": 1,
        "record_schema": "xyxy_score_class_id_v1",
        "coordinate_space": "original_image_xyxy_pixels",
        "sort_policy": (
            "score_desc_class_id_asc_xyxy_lexicographic_v1"
        ),
        "detections": detections,
    }
    completed_result_sha256 = _canonical_sha256(completed_result_payload)
    postprocess_result = {
        "task": "detection",
        "contract_family": "decoded_nms",
        "decoder_format": "ultralytics_regcls",
        "coordinate_space": "original_image_xyxy_pixels",
        "detection_count": len(detections),
        "detections": detections,
        "detections_sha256": detections_sha256,
        "record_schema": "xyxy_score_class_id_v1",
        "canonical_sort_policy": (
            "score_desc_class_id_asc_xyxy_lexicographic_v1"
        ),
        "completed_result_artifact": completed_result_payload,
        "completed_result_artifact_sha256": completed_result_sha256,
        "postprocess_contract_sha256": decoder_sha256,
    }
    output_manifest = dump_dir / "native_full_outputs_manifest.json"
    output_tensor = dump_dir / "output0.bin"
    output_tensor.parent.mkdir(parents=True, exist_ok=True)
    output_tensor.write_bytes(b"\x00\x00\x00\x00" * 12)
    _write_json(output_manifest, {
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 4,
        "model": "yolo11l",
        "task": "detection",
        "case": "full",
        "backend": "native_full_hailo8",
        "comparison_backend": "hailo8",
        "setup_id": "orin_nx_hailo8_01",
        "input_image": str(image.resolve()),
        "input_image_sha256": image_sha256,
        "provenance": {
            "image": str(image.resolve()),
            "image_sha256": image_sha256,
        },
        "decoder_contract_sha256": decoder_sha256,
        "nms_contract_sha256": nms_sha256,
        "frozen_host_postprocess_contract_sha256": decoder_sha256,
        "completed_task_endpoint_contract_hash": endpoint_sha256,
        "completed_task_output_endpoint_id": endpoint_contract[
            "output_endpoint_id"
        ],
        "authoritative_output_contract_resolution": resolution,
        "frozen_host_postprocess_contract": postprocess_contract,
        "frozen_host_postprocess_result": postprocess_result,
        "outputs": [{
            "name": "output0",
            "file": output_tensor.name,
            "sha256": _file_sha256(output_tensor),
            "bytes": output_tensor.stat().st_size,
            "shape": [1, 3, 4],
            "dtype": "float32",
        }],
    })

    completed_result = result_dir / "report.completed_task_result_artifact.json"
    completed_result.parent.mkdir(parents=True, exist_ok=True)
    completed_result.write_bytes(json.dumps(
        completed_result_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8"))
    assert _file_sha256(completed_result) == completed_result_sha256
    native_report = result_dir / "report.json"
    _write_json(native_report, {
        "ok": True,
        "model": "yolo11l",
        "task": "detection",
        "backend": "native_full_hailo8",
        "comparison_backend": "hailo8",
        "setup_id": "orin_nx_hailo8_01",
        "diagnostic_only": True,
        "claim_eligible": False,
        "input_image_sha256": image_sha256,
        "preprocessing_contract_sha256": preprocessing_sha256,
        "decoder_contract_sha256": decoder_sha256,
        "nms_contract_sha256": nms_sha256,
        "frozen_host_postprocess_contract": postprocess_contract,
        "frozen_host_postprocess_contract_sha256": decoder_sha256,
        "frozen_host_postprocess_invariant_contract_sha256": (
            postprocess_invariant_sha256
        ),
        "frozen_host_postprocess_result": postprocess_result,
        "completed_task_endpoint_contract": endpoint_contract,
        "completed_task_endpoint_contract_hash": endpoint_sha256,
        "completed_task_comparison_endpoint_contract_hash": endpoint_sha256,
        "completed_task_output_endpoint_id": endpoint_contract[
            "output_endpoint_id"
        ],
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact": completed_result_payload,
        "completed_task_result_artifact_sha256": completed_result_sha256,
        "completed_task_result_artifact_path": str(completed_result.resolve()),
        "completed_task_result_artifact_file_sha256": completed_result_sha256,
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
    })

    _write_json(canary_dir / "canary_result.json", {
        "schema": "onnx-splitpoint/v27713-hailo8-artifact-canary-result",
        "schema_version": 1,
        "status": "PASS",
        "result_receipt_written": True,
        "setup_id": "orin_nx_hailo8_01",
        "canary_scope": "structural_artifact_and_runtime_only",
        "numerical_correctness_attested": False,
        "backend_parity_attested": False,
        "source_snapshot_before_sha256": "a" * 64,
        "source_snapshot_after_sha256": "a" * 64,
        "attested_source_files_modified": False,
        "attested_source_files_changed": [],
        "benchmark_sets": [str(benchmark_set.resolve())],
        "model_results": [{
            "model_id": "yolo11l",
            "task": "detection",
            "status": "PASS",
            "hef_sha256": hef_sha256,
            "image": image.name,
            "image_sha256": image_sha256,
            "preprocessing_contract_sha256": preprocessing_sha256,
            "decoder_contract_sha256": decoder_sha256,
            "nms_contract_sha256": nms_sha256,
            "completed_task_endpoint_contract_hash": endpoint_sha256,
            "completed_task_result_artifact_sha256": (
                completed_result_sha256
            ),
            "completed_detections_sha256": detections_sha256,
            "structural_output_dump_attested": True,
            "structural_postprocess_completed": True,
            "structural_preprocessing_binding_attested": True,
            "numerical_correctness_attested": False,
            "backend_parity_attested": False,
        }],
        "runtime": {
            "status": "COMPLETED",
            "payload_verified": True,
            "runtime_admitted": True,
            "failures": [],
            "invocations": [{
                "model_id": "yolo11l",
                "expected_hef_sha256": hef_sha256,
                "image": image.name,
                "image_sha256": image_sha256,
                "preprocessing_contract_sha256": preprocessing_sha256,
                "completed_task_endpoint_contract_hash": endpoint_sha256,
                "result_relative": "results/01_yolo11l/00",
                "returncode": 0,
            }],
        },
    })

    output_meta = json.loads(output_manifest.read_text(encoding="utf-8"))
    _write_json(self_reference, {
        "schema": "onnx-splitpoint/native-yolo-full-self-reference-probe",
        "schema_version": 6,
        "ok": True,
        "semantic_available": True,
        "semantic_ok": True,
        "diagnosis": "native_semantic_matches_full_self_reference",
        "model": "yolo11l",
        "case": "full",
        "benchmark_set": str(benchmark_set.resolve()),
        "full_onnx": str(onnx.resolve()),
        "full_onnx_sha256": model_sha256,
        "boundary_manifest": str(boundary_manifest.resolve()),
        "boundary_manifest_sha256": _file_sha256(boundary_manifest),
        "native_output_manifest": str(output_manifest.resolve()),
        "native_output_manifest_sha256": _file_sha256(output_manifest),
        "native_report": str(native_report.resolve()),
        "native_report_sha256": _file_sha256(native_report),
        "native_output_manifest_meta": output_meta,
        "expected_contract_family": "decoded_nms",
        "expected_contract_source": (
            "verified_completed_task_comparison_endpoint_v2"
        ),
        "contract_family_match": True,
        "numerical_similarity_pass": True,
        "numerical_similarity_status": "passed",
        "native_detections": detections,
        "reference_detections": detections,
        "numerical_similarity_value": 1.0,
        "numerical_similarity_threshold": 0.8,
        "numerical_similarity_mean_iou": 1.0,
        "numerical_similarity_mean_iou_threshold": 0.85,
        "numerical_similarity_scope": "class_aware_postnms_detection",
        "numerical_similarity_metric": (
            "reference_match_ratio_and_mean_matched_iou"
        ),
        "numerical_similarity_policy_id": (
            policy["native_self_reference_policy_id"]
        ),
        "numerical_similarity_iou_threshold": 0.5,
        "numerical_similarity_confidence_threshold": 0.25,
        "numerical_similarity_matched_count": 1,
        "numerical_similarity_reference_count": 1,
        "numerical_similarity_policy": policy,
        "numerical_similarity_policy_sha256": policy_sha256,
        "best": {
            "contract_family_match": True,
            "full_count": 1,
            "native_count": 1,
            "full_mode": "completed_v2:full_ultralytics_decoded_decoded_nms",
            "native_mode": "completed_v2:frozen_host_tail",
            "match": {
                "ref_count": 1,
                "pred_count": 1,
                "matched": 1,
                "match_ratio": 1.0,
                "mean_iou": 1.0,
                "iou_threshold": 0.5,
            },
        },
        "completed_v2_evidence": {
            "available": True,
            "completed_v2_verified": True,
            "expected_contract_family": "decoded_nms",
            "completed_v2_semantic_evidence_tier": (
                "exact_same_hotloop_completed_artifact"
            ),
            "full_mode": (
                "completed_v2:full_ultralytics_decoded_decoded_nms"
            ),
            "native_mode": "completed_v2:frozen_host_tail",
            "completed_task_comparison_endpoint_contract_hash": (
                endpoint_sha256
            ),
            "completed_task_comparison_endpoint_contract": endpoint_contract,
            "completed_task_comparison_output_endpoint_id": (
                endpoint_contract["output_endpoint_id"]
            ),
            "native_completed_result_sha256": detections_sha256,
            "performance_hotloop_result_sha256": detections_sha256,
            "full_reference_result_sha256": detections_sha256,
            "exact_completed_result_identity_bound": True,
            "completed_v2_exact_result_claim_binding": True,
            "portable_result_hash_mismatch": False,
            "semantic_result_binding_status": (
                "exact_same_hotloop_completed_artifact"
            ),
        },
    })

    return ExistingEvidenceFixture(
        run_dir=run_dir,
        canary_dir=canary_dir,
        self_reference=self_reference,
        output=output,
        benchmark_set=benchmark_set,
        onnx=onnx,
        hef=hef,
        hef_receipt=hef_receipt,
        quality_request=quality_request,
        candidate=candidate,
        boundary_manifest=boundary_manifest,
        output_manifest=output_manifest,
        native_report=native_report,
        completed_result=completed_result,
        model_sha256=model_sha256,
        hef_sha256=hef_sha256,
        preprocessing_sha256=preprocessing_sha256,
        endpoint_sha256=endpoint_sha256,
    )


def _snapshot_files(*roots: Path) -> dict[str, tuple[bytes, int, int]]:
    snapshot: dict[str, tuple[bytes, int, int]] = {}
    for root in roots:
        if root.is_file():
            paths = (root,)
        else:
            paths = tuple(path for path in root.rglob("*") if path.is_file())
        for path in paths:
            key = f"{root.name}/{path.relative_to(root) if path != root else '.'}"
            info = path.stat()
            snapshot[key] = (path.read_bytes(), info.st_mtime_ns, info.st_mode)
    return snapshot


def _make_files_read_only(*roots: Path) -> None:
    for root in roots:
        paths = (root,) if root.is_file() else tuple(root.rglob("*"))
        for path in paths:
            if path.is_file():
                path.chmod(0o444)


def _receipt(fixture: ExistingEvidenceFixture, *, output: Path | None = None):
    return create_receipt(
        run_dir=fixture.run_dir,
        canary_dir=fixture.canary_dir,
        self_references=(fixture.self_reference,),
        output=output or fixture.output,
    )


def _reason_text(result: dict[str, Any]) -> str:
    return json.dumps(result.get("reason_codes", []), sort_keys=True).lower()


def _reseal_artifact_index_path(
    fixture: ExistingEvidenceFixture,
    relative: str,
) -> None:
    index_path = fixture.run_dir / "artifact_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload["artifacts"]
        if row.get("path") == relative
    ]
    assert len(rows) == 1
    source = fixture.run_dir / relative
    rows[0]["sha256"] = f"sha256:{_file_sha256(source)}"
    rows[0]["size_bytes"] = source.stat().st_size
    _write_json(index_path, payload)


def _rewrite_hailo_request_reference(
    fixture: ExistingEvidenceFixture,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    request = json.loads(
        fixture.quality_request.read_text(encoding="utf-8")
    )
    reference = request.get("reference")
    assert isinstance(reference, dict)
    mutate(reference)
    _write_json(fixture.quality_request, request)

    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hailo = next(
        row for row in summary["results"] if row["backend"] == "hailo8"
    )
    hailo["source_request_sha256"] = (
        f"sha256:{_file_sha256(fixture.quality_request)}"
    )
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)


def _reseal_output_manifest_self_reference(
    fixture: ExistingEvidenceFixture,
) -> None:
    payload = json.loads(
        fixture.self_reference.read_text(encoding="utf-8")
    )
    payload["native_output_manifest_sha256"] = _file_sha256(
        fixture.output_manifest
    )
    payload["native_output_manifest_meta"] = json.loads(
        fixture.output_manifest.read_text(encoding="utf-8")
    )
    _write_json(fixture.self_reference, payload)


def _assert_retrospective_scope(result: dict[str, Any]) -> None:
    assert result["claim_scope"] == "retrospective_evaluated_matrix"
    assert result["claim_scope_unchanged"] is True
    assert result["prospective_freeze"] is False
    assert result["entire_run_pass_attested"] is False
    assert result["hardware_rerun"] is False
    assert result["compiler_rerun"] is False
    assert result["quality_rerun"] is False


def test_verified_receipt_is_read_only_and_strictly_retrospective(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    _make_files_read_only(
        fixture.run_dir,
        fixture.canary_dir,
        fixture.self_reference,
    )
    before = _snapshot_files(
        fixture.run_dir,
        fixture.canary_dir,
        fixture.self_reference,
    )
    files_before_create = {
        path.resolve() for path in tmp_path.rglob("*") if path.is_file()
    }

    receipt = _receipt(fixture)

    files_after_create = {
        path.resolve() for path in tmp_path.rglob("*") if path.is_file()
    }
    assert files_after_create - files_before_create == {
        fixture.output.resolve()
    }
    assert receipt == json.loads(fixture.output.read_text(encoding="utf-8"))
    assert receipt["status"] == "VERIFIED"
    assert receipt["ok"] is True
    assert receipt["reason_codes"] == []
    assert len(receipt["evidence_projection_sha256"]) == 64
    assert len(receipt["result_payload_sha256"]) == 64
    _assert_retrospective_scope(receipt)
    source_run = receipt["evidence_projection"]["source_run"]
    assert source_run["recorded_status"] == "partial"
    assert source_run["recorded_technical_status"] == "partial"
    assert source_run["recorded_scientific_status"] == "not_evaluated"
    assert source_run["quality_status"] == "ok"
    assert source_run["quality_scientific_status"] == "fail"
    assert source_run["quality_decision"] == "fail"
    assert receipt["evidence_projection"]["quality_item_count"] == 500
    output_payload = json.loads(
        fixture.output_manifest.read_text(encoding="utf-8")
    )
    assert "preprocessing_contract_sha256" not in output_payload
    assert Path(json.loads(
        fixture.boundary_manifest.read_text(encoding="utf-8")
    )["input_dump"]).is_absolute()
    assert _snapshot_files(
        fixture.run_dir,
        fixture.canary_dir,
        fixture.self_reference,
    ) == before
    assert not fixture.output.is_relative_to(fixture.run_dir)
    assert not fixture.output.is_relative_to(fixture.canary_dir)

    receipt_bytes = fixture.output.read_bytes()
    receipt_mtime_ns = fixture.output.stat().st_mtime_ns
    verified = verify_receipt(
        fixture.output,
        run_dir=fixture.run_dir,
        canary_dir=fixture.canary_dir,
        self_references=(fixture.self_reference,),
    )
    assert verified["status"] == "VERIFIED"
    assert verified["ok"] is True
    assert verified["reason_codes"] == []
    assert fixture.output.read_bytes() == receipt_bytes
    assert fixture.output.stat().st_mtime_ns == receipt_mtime_ns
    assert _snapshot_files(
        fixture.run_dir,
        fixture.canary_dir,
        fixture.self_reference,
    ) == before


def test_real_historical_reference_placeholder_schema_is_verified(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    request = json.loads(
        fixture.quality_request.read_text(encoding="utf-8")
    )

    assert set(request["reference"]) == {
        "source",
        "reference_role",
        "semantic_reference_only",
        "required",
        "expected_image_ids",
        "expected_image_ids_sha256",
        "record_count",
        "quality_contract_sha256",
    }
    receipt = _receipt(fixture)
    assert receipt["status"] == "VERIFIED"
    assert receipt["ok"] is True


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    (
        ("source", "quality_request_binding_mismatch"),
        ("reference_role", "quality_request_binding_mismatch"),
        ("semantic_reference_only", "quality_request_binding_mismatch"),
        ("required", "quality_request_binding_mismatch"),
        ("record_count", "quality_request_binding_mismatch"),
        ("quality_contract_sha256", "quality_request_binding_mismatch"),
        ("expected_image_ids", "quality_identity_mismatch"),
        ("expected_image_ids_sha256", "quality_identity_mismatch"),
    ),
)
def test_historical_reference_placeholder_fields_remain_fail_closed(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)

    def mutate(reference: dict[str, Any]) -> None:
        if mutation == "source":
            reference[mutation] = "candidate_local_reference"
        elif mutation == "reference_role":
            reference[mutation] = "foreign_reference"
        elif mutation in {"semantic_reference_only", "required"}:
            reference[mutation] = False
        elif mutation == "record_count":
            reference[mutation] = 499
        elif mutation == "quality_contract_sha256":
            reference[mutation] = "f" * 64
        elif mutation == "expected_image_ids":
            reference[mutation][0] = "999999999999.jpg"
        else:
            reference[mutation] = "f" * 64

    _rewrite_hailo_request_reference(fixture, mutate)
    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == [reason_code]


@pytest.mark.parametrize(
    "fields",
    (
        ("path",),
        ("sha256",),
        ("size_bytes",),
        ("prediction_sha256",),
        ("path", "sha256", "size_bytes", "prediction_sha256"),
    ),
)
def test_optional_post_execution_reference_identities_are_verified(
    tmp_path: Path,
    fields: tuple[str, ...],
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    reference_path = fixture.run_dir / (
        "quality_management/references/yolo11l/"
        "canonical_cpu_reference.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    def mutate(descriptor: dict[str, Any]) -> None:
        identities = {
            "path": reference_path.relative_to(fixture.run_dir).as_posix(),
            "sha256": _file_sha256(reference_path),
            "size_bytes": reference_path.stat().st_size,
            "prediction_sha256": reference["prediction_sha256"],
        }
        descriptor.update({field: identities[field] for field in fields})

    _rewrite_hailo_request_reference(fixture, mutate)
    receipt = _receipt(fixture)

    assert receipt["status"] == "VERIFIED"
    assert receipt["ok"] is True


@pytest.mark.parametrize(
    ("mutation", "status", "reason_code"),
    (
        ("path", "CONFLICT", "source_run_binding_mismatch"),
        (
            "sha256",
            "INCOMPLETE",
            "historical_reference_version_unavailable",
        ),
        (
            "size_bytes",
            "INCOMPLETE",
            "historical_reference_version_unavailable",
        ),
        ("prediction_sha256", "CONFLICT", "quality_request_binding_mismatch"),
    ),
)
def test_optional_post_execution_reference_identities_fail_closed(
    tmp_path: Path,
    mutation: str,
    status: str,
    reason_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    reference_path = fixture.run_dir / (
        "quality_management/references/yolo11l/"
        "canonical_cpu_reference.json"
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    def mutate(descriptor: dict[str, Any]) -> None:
        descriptor.update({
            "path": reference_path.relative_to(fixture.run_dir).as_posix(),
            "sha256": _file_sha256(reference_path),
            "size_bytes": reference_path.stat().st_size,
            "prediction_sha256": reference["prediction_sha256"],
        })
        if mutation == "path":
            descriptor[mutation] = (
                "quality_management/references/foreign/"
                "canonical_cpu_reference.json"
            )
        elif mutation == "size_bytes":
            descriptor[mutation] += 1
        else:
            descriptor[mutation] = "f" * 64

    _rewrite_hailo_request_reference(fixture, mutate)
    receipt = _receipt(fixture)

    assert receipt["status"] == status
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == [reason_code]


@pytest.mark.parametrize("relative_path", (False, True))
def test_immutable_by_source_contract_reference_path_is_verified(
    tmp_path: Path,
    relative_path: bool,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    source_contract_sha256 = "a" * 64
    legacy = fixture.run_dir / (
        "quality_management/references/yolo11l/"
        "canonical_cpu_reference.json"
    )
    immutable = fixture.run_dir / (
        "quality_management/references/yolo11l/by_source_contract/"
        f"{source_contract_sha256}/canonical_cpu_reference.json"
    )
    immutable.parent.mkdir(parents=True)
    legacy.rename(immutable)

    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["results"]:
        management_reference = row["management_cpu_reference"]
        management_reference.update({
            "reference_path": (
                immutable.relative_to(fixture.run_dir).as_posix()
                if relative_path
                else str(immutable.resolve())
            ),
            "reference_sha256": (
                "sha256:"
                + str(management_reference["reference_sha256"]).removeprefix(
                    "sha256:"
                )
            ),
            "source_contract_sha256": f"sha256:{source_contract_sha256}",
            "reference_storage": "immutable_source_contract",
            "reference_immutable": True,
            "reference_size_bytes": immutable.stat().st_size,
        })
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "VERIFIED"
    assert receipt["ok"] is True


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("source_contract_sha256", "b" * 64),
        ("reference_storage", "mutable_alias"),
        ("reference_immutable", False),
    ),
)
def test_immutable_reference_markers_are_strictly_bound(
    tmp_path: Path,
    mutation: str,
    value: object,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    source_contract_sha256 = "a" * 64
    legacy = fixture.run_dir / (
        "quality_management/references/yolo11l/"
        "canonical_cpu_reference.json"
    )
    immutable = fixture.run_dir / (
        "quality_management/references/yolo11l/by_source_contract/"
        f"{source_contract_sha256}/canonical_cpu_reference.json"
    )
    immutable.parent.mkdir(parents=True)
    legacy.rename(immutable)

    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["results"]:
        management_reference = row["management_cpu_reference"]
        management_reference.update({
            "reference_path": str(immutable.resolve()),
            "reference_sha256": (
                "sha256:"
                + str(management_reference["reference_sha256"]).removeprefix(
                    "sha256:"
                )
            ),
            "source_contract_sha256": f"sha256:{source_contract_sha256}",
            "reference_storage": "immutable_source_contract",
            "reference_immutable": True,
            "reference_size_bytes": immutable.stat().st_size,
        })
    summary["results"][0]["management_cpu_reference"][mutation] = value
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == ["source_run_binding_mismatch"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reference_storage", "immutable_source_contract"),
        ("reference_immutable", True),
    ),
)
def test_legacy_reference_cannot_claim_immutable_storage(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["results"][0]["management_cpu_reference"][field] = value
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == ["source_run_binding_mismatch"]


def test_moved_historical_absolute_reference_path_rebases_exact_suffix(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    moved_legacy = (
        "/detached/original/evaluation-run/quality_management/"
        "references/yolo11l/canonical_cpu_reference.json"
    )
    for row in summary["results"]:
        row["management_cpu_reference"]["reference_path"] = moved_legacy
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "VERIFIED"
    assert receipt["ok"] is True


def test_moved_immutable_absolute_reference_path_rebases_exact_suffix(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    source_contract_sha256 = "a" * 64
    legacy = fixture.run_dir / (
        "quality_management/references/yolo11l/"
        "canonical_cpu_reference.json"
    )
    immutable = fixture.run_dir / (
        "quality_management/references/yolo11l/by_source_contract/"
        f"{source_contract_sha256}/canonical_cpu_reference.json"
    )
    immutable.parent.mkdir(parents=True)
    legacy.rename(immutable)

    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    moved_immutable = (
        "/detached/original/evaluation-run/quality_management/"
        "references/yolo11l/by_source_contract/"
        f"{source_contract_sha256}/canonical_cpu_reference.json"
    )
    for row in summary["results"]:
        management_reference = row["management_cpu_reference"]
        management_reference.update({
            "reference_path": moved_immutable,
            "reference_sha256": (
                "sha256:"
                + str(management_reference["reference_sha256"]).removeprefix(
                    "sha256:"
                )
            ),
            "source_contract_sha256": f"sha256:{source_contract_sha256}",
            "reference_storage": "immutable_source_contract",
            "reference_immutable": True,
            "reference_size_bytes": immutable.stat().st_size,
        })
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "VERIFIED"
    assert receipt["ok"] is True


@pytest.mark.parametrize(
    ("declared_path", "reason_code"),
    (
        (
            "quality_management/references/foreign/"
            "canonical_cpu_reference.json",
            "source_run_binding_mismatch",
        ),
        (
            "quality_management/references/yolo11l/by_source_contract/"
            "not-a-digest/canonical_cpu_reference.json",
            "source_run_binding_mismatch",
        ),
        (
            "quality_management/references/yolo11l/../yolo11l/"
            "canonical_cpu_reference.json",
            "unsafe_manifest_path",
        ),
    ),
)
def test_management_reference_path_shapes_fail_closed(
    tmp_path: Path,
    declared_path: str,
    reason_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hailo = next(
        row for row in summary["results"] if row["backend"] == "hailo8"
    )
    hailo["management_cpu_reference"]["reference_path"] = declared_path
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == [reason_code]


def test_non_target_classification_rows_bind_global_summary_structurally(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    run_manifest_path = fixture.run_dir / "run_manifest.json"
    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    run_manifest["models"]["mobilenet_v3_large"] = {
        "model_id": "mobilenet_v3_large",
        "family": "mobilenet_v3",
        "family_id": "mobilenet_v3",
        "evaluation_role": "development",
        "generalization_scope": "development",
        "model_sha256": f"sha256:{fixture.model_sha256}",
    }
    _write_json(run_manifest_path, run_manifest)

    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    classification_rows = []
    for source in summary["results"]:
        row = json.loads(json.dumps(source))
        row["model_id"] = "mobilenet_v3_large"
        row["task"] = "classification"
        for field in (
            "decoder_contract_sha256",
            "nms_contract_sha256",
        ):
            row.pop(field, None)
        for field in tuple(row):
            if field.startswith("task_quality_"):
                row.pop(field)
        classification_rows.append(row)
    summary["results"].extend(classification_rows)
    summary.update({
        "decision_counts": {"fail": 2, "pass": 2},
        "request_count": 4,
        "quality_result_count": 4,
        "completed_count": 4,
        "summary_only_full_quality_expected_count": 4,
        "summary_only_full_quality_completed_count": 4,
    })
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "VERIFIED"
    source_run = receipt["evidence_projection"]["source_run"]
    assert source_run["validated_quality_decision_counts"] == {
        "fail": 2,
        "pass": 2,
    }
    assert [
        model["model_id"]
        for model in receipt["evidence_projection"]["models"]
    ] == ["yolo11l"]

def test_missing_existing_evidence_is_incomplete_and_receipted(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    fixture.native_report.unlink()

    receipt = _receipt(fixture)

    assert receipt["status"] == "INCOMPLETE"
    assert receipt["ok"] is False
    assert receipt == json.loads(fixture.output.read_text(encoding="utf-8"))
    assert "missing" in _reason_text(receipt)
    _assert_retrospective_scope(receipt)


def test_verify_receipt_detects_equal_length_hash_tamper_with_restored_mtime(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    receipt = _receipt(fixture)
    original = fixture.boundary_manifest.read_bytes()
    original_stat = fixture.boundary_manifest.stat()
    tampered = original.replace(b'"case": "full"', b'"case": "evil"')
    assert len(tampered) == len(original)
    fixture.boundary_manifest.write_bytes(tampered)
    os.utime(
        fixture.boundary_manifest,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )

    verified = verify_receipt(fixture.output)

    assert verified["status"] == "CONFLICT"
    assert verified["ok"] is False
    assert any(
        token in _reason_text(verified)
        for token in ("hash", "sha256", "identity")
    )


def test_physical_candidate_prediction_identity_mismatch_is_conflict(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    candidate = json.loads(fixture.candidate.read_text(encoding="utf-8"))
    candidate["records"][0]["candidate"][0]["class_id"] = 9
    _write_json(fixture.candidate, candidate)

    # Keep both physical file descriptors current so the verifier has to
    # reject the stale prediction fingerprint, not merely a stale file hash.
    request = json.loads(fixture.quality_request.read_text(encoding="utf-8"))
    request["candidate"]["sha256"] = _file_sha256(fixture.candidate)
    request["candidate"]["size_bytes"] = fixture.candidate.stat().st_size
    _write_json(fixture.quality_request, request)
    summary_path = (
        fixture.run_dir
        / "quality_management/central_quality_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hailo_row = next(
        row for row in summary["results"] if row["backend"] == "hailo8"
    )
    hailo_row["source_request_sha256"] = (
        f"sha256:{_file_sha256(fixture.quality_request)}"
    )
    _write_json(summary_path, summary)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert "candidate_predictions_sha256_mismatch" in _reason_text(receipt)


@pytest.mark.parametrize("mutation", ["hef_receipt", "semantic_model_hash"])
def test_identity_conflicts_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    if mutation == "hef_receipt":
        payload = json.loads(fixture.hef_receipt.read_text(encoding="utf-8"))
        payload["source_onnx_sha256"] = "0" * 64
        _write_json(fixture.hef_receipt, payload)
    else:
        payload = json.loads(
            fixture.self_reference.read_text(encoding="utf-8")
        )
        payload["full_onnx_sha256"] = "f" * 64
        _write_json(fixture.self_reference, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"]


@pytest.mark.parametrize("symlink_kind", ["file", "parent", "root"])
def test_input_symlinks_fail_closed(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    try:
        if symlink_kind == "file":
            target = tmp_path / "outside-boundary.json"
            target.write_bytes(fixture.boundary_manifest.read_bytes())
            fixture.boundary_manifest.unlink()
            fixture.boundary_manifest.symlink_to(target)
        elif symlink_kind == "parent":
            dump = fixture.boundary_manifest.parent
            target = dump.with_name("real-dump")
            dump.rename(target)
            dump.symlink_to(target, target_is_directory=True)
        else:
            target = fixture.run_dir.with_name("real-run-a")
            fixture.run_dir.rename(target)
            fixture.run_dir.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert any(
        token in _reason_text(receipt)
        for token in (
            "symlink",
            "unsafe",
            "no_follow",
            "nofollow",
            "not_regular_file",
        )
    )


def test_self_reference_from_another_source_run_is_conflict(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path, label="a")
    foreign = _build_existing_evidence(
        tmp_path,
        label="b",
        run_id="evaluation-run-b",
        output_name="unused-foreign-receipt.json",
    )
    assert _file_sha256(foreign.onnx) == fixture.model_sha256

    payload = json.loads(
        fixture.self_reference.read_text(encoding="utf-8")
    )
    payload["benchmark_set"] = str(foreign.benchmark_set.resolve())
    payload["full_onnx"] = str(foreign.onnx.resolve())
    # Content identities remain valid; only the source-run binding changed.
    assert payload["full_onnx_sha256"] == _file_sha256(foreign.onnx)
    _write_json(fixture.self_reference, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert "source_run_binding_mismatch" in _reason_text(receipt)


def test_projection_and_result_identity_are_deterministic(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)

    first = _receipt(fixture, output=tmp_path / "receipt-one.json")
    second = _receipt(fixture, output=tmp_path / "receipt-two.json")

    assert first["status"] == second["status"] == "VERIFIED"
    assert first["evidence_projection"] == second["evidence_projection"]
    assert (
        first["evidence_projection_sha256"]
        == second["evidence_projection_sha256"]
    )
    assert first["result_payload_sha256"] == second["result_payload_sha256"]
    assert first == second


def test_receipt_projection_forgery_is_rejected_even_with_new_outer_hash(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    receipt = _receipt(fixture)
    receipt["evidence_projection"]["source_run"]["recorded_status"] = (
        "complete"
    )
    receipt["result_payload_sha256"] = _canonical_sha256({
        key: value
        for key, value in receipt.items()
        if key != "result_payload_sha256"
    })
    _write_json(fixture.output, receipt)

    with pytest.raises(ExistingEvidenceError) as caught:
        verify_receipt(fixture.output)

    assert caught.value.code == "evidence_projection_sha256_mismatch"


def test_completed_artifact_tamper_is_conflict(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(fixture.completed_result.read_text(encoding="utf-8"))
    payload["detections"][0]["class_id"] = 7
    fixture.completed_result.write_bytes(json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8"))

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert any(
        token in _reason_text(receipt)
        for token in ("completed", "artifact", "sha256")
    )


def test_canary_image_binding_mismatch_is_conflict(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    path = fixture.canary_dir / "canary_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_results"][0]["image_sha256"] = "f" * 64
    _write_json(path, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert "image" in _reason_text(receipt)


def test_preprocessing_cross_binding_mismatch_is_conflict(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(
        fixture.boundary_manifest.read_text(encoding="utf-8")
    )
    payload["preprocessing_contract_sha256"] = "e" * 64
    _write_json(fixture.boundary_manifest, payload)

    semantic = json.loads(
        fixture.self_reference.read_text(encoding="utf-8")
    )
    semantic["boundary_manifest_sha256"] = _file_sha256(
        fixture.boundary_manifest
    )
    _write_json(fixture.self_reference, semantic)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert "quality_identity_mismatch" in _reason_text(receipt)


@pytest.mark.parametrize(
    "mutation",
    ("candidate_range", "delta_arithmetic", "decision", "flattened_alias"),
)
def test_central_quality_metric_and_decision_tamper_is_conflict(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(
        item for item in payload["results"] if item["backend"] == "hailo8"
    )
    if mutation == "candidate_range":
        row["primary"]["candidate"] = 999.0
    elif mutation == "delta_arithmetic":
        row["primary"]["delta"] = 123.0
    elif mutation == "decision":
        row["primary"]["decision"] = "pass"
        row["primary"]["status"] = "pass"
    else:
        row["task_quality_candidate"] = 999.0
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert any(
        code in receipt["reason_codes"]
        for code in (
            "quality_metric_inconsistent",
            "quality_decision_mismatch",
        )
    )


def test_central_quality_summary_cannot_override_failing_rows(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in (
        "scientific_status",
        "aggregate_quality_decision",
        "quality_decision",
    ):
        payload[field] = "pass"
    payload["scientific_pass"] = True
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == [
        "central_quality_summary_decision_mismatch"
    ]


@pytest.mark.parametrize("field", ("scientific_pass", "decision_counts"))
def test_central_quality_summary_derived_fields_are_bound(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = (
        True if field == "scientific_pass" else {"pass": 2}
    )
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == [
        "central_quality_summary_decision_mismatch"
    ]


@pytest.mark.parametrize(
    "field",
    ("decision", "scientific_status"),
)
def test_central_quality_row_required_decision_aliases_cannot_be_removed(
    tmp_path: Path,
    field: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(
        item for item in payload["results"]
        if item["backend"] == "hailo8"
    )
    del row[field]
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == ["quality_decision_mismatch"]


def test_optional_task_quality_decision_alias_is_strict_when_present(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = next(
        item for item in payload["results"]
        if item["backend"] == "hailo8"
    )
    row["task_quality_decision"] = "pass"
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["quality_decision_mismatch"]


def test_missing_expected_quality_row_is_incomplete(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"] = [
        row for row in payload["results"]
        if row["backend"] != "hailo8"
    ]
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "INCOMPLETE"
    assert receipt["reason_codes"] == ["required_quality_row_missing"]


def test_unexpected_quality_row_is_conflict(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = json.loads(json.dumps(payload["results"][0]))
    extra["model_id"] = "unregistered_model"
    payload["results"].append(extra)
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["unexpected_quality_row"]


def test_canary_scope_must_be_exactly_structural_runtime_only(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    path = fixture.canary_dir / "canary_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["canary_scope"] = "structural_plus_unregistered_semantics"
    _write_json(path, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["canary_scope_mismatch"]


@pytest.mark.parametrize("mutation", ("ratio", "best_count", "best_match"))
def test_self_reference_counts_and_ratio_must_be_consistent(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(fixture.self_reference.read_text(encoding="utf-8"))
    if mutation == "ratio":
        payload["numerical_similarity_matched_count"] = 0
        payload["best"]["match"]["matched"] = 0
    elif mutation == "best_count":
        payload["best"]["full_count"] = 999
    else:
        payload["best"]["match"]["match_ratio"] = 0.5
    _write_json(fixture.self_reference, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["self_reference_count_mismatch"]


@pytest.mark.parametrize("mutation", ("body", "binding"))
def test_self_reference_policy_body_and_threshold_binding_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(fixture.self_reference.read_text(encoding="utf-8"))
    policy = payload["numerical_similarity_policy"]
    if mutation == "body":
        policy["native_self_reference_min_match"] = 0.7
    else:
        policy["native_self_reference_min_match"] = 0.7
        payload["numerical_similarity_policy_sha256"] = _canonical_sha256(
            policy
        )
    _write_json(fixture.self_reference, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["self_reference_policy_mismatch"]


@pytest.mark.parametrize("mutation", ("full_reference_hash", "native_list"))
def test_completed_v2_result_hashes_bind_real_detection_lists(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(fixture.self_reference.read_text(encoding="utf-8"))
    if mutation == "full_reference_hash":
        payload["completed_v2_evidence"][
            "full_reference_result_sha256"
        ] = "f" * 64
    else:
        payload["native_detections"][0]["class_id"] = 7
    _write_json(fixture.self_reference, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["completed_v2_evidence_invalid"]


def test_completed_comparison_endpoint_body_must_match_declared_hash(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(fixture.self_reference.read_text(encoding="utf-8"))
    payload["completed_v2_evidence"][
        "completed_task_comparison_endpoint_contract"
    ]["score_threshold"] = 0.5
    _write_json(fixture.self_reference, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["completed_v2_evidence_invalid"]


def test_unavailable_historical_reference_container_is_incomplete(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    reference = (
        fixture.run_dir
        / "quality_management/references/yolo11l/"
        "canonical_cpu_reference.json"
    )
    # Preserve semantic JSON while changing the physical container identity.
    reference.write_bytes(reference.read_bytes() + b"\n")

    receipt = _receipt(fixture)

    assert receipt["status"] == "INCOMPLETE"
    assert receipt["ok"] is False
    assert receipt["reason_codes"] == [
        "historical_reference_version_unavailable"
    ]


@pytest.mark.parametrize("mutation", ["empty", "stale_binding"])
def test_artifact_index_cannot_be_empty_or_stale(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    path = fixture.run_dir / "artifact_index.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "empty":
        payload["artifacts"] = []
    else:
        row = next(
            item
            for item in payload["artifacts"]
            if item["path"]
            == "quality_management/central_quality_summary.json"
        )
        row["sha256"] = f"sha256:{'0' * 64}"
    _write_json(path, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["ok"] is False
    assert "artifact_index" in _reason_text(receipt)


def test_two_historical_reference_hashes_at_one_path_are_incomplete(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = "quality_management/central_quality_summary.json"
    summary_path = fixture.run_dir / relative
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hailo = next(row for row in summary["results"] if row["backend"] == "hailo8")
    tensorrt = next(
        row for row in summary["results"] if row["backend"] == "tensorrt"
    )
    assert (
        hailo["management_cpu_reference"]["reference_path"]
        == tensorrt["management_cpu_reference"]["reference_path"]
    )
    tensorrt["management_cpu_reference"]["reference_sha256"] = "f" * 64
    assert (
        hailo["reference_predictions_sha256"]
        == tensorrt["reference_predictions_sha256"]
    )
    _write_json(summary_path, summary)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "INCOMPLETE"
    assert receipt["reason_codes"] == [
        "historical_reference_version_unavailable"
    ]
    assert "yolo11l:tensorrt" in json.dumps(
        receipt.get("reason_details", [])
    )


def test_incomplete_receipt_can_be_reverified_stably(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    fixture.native_report.unlink()
    created = _receipt(fixture)
    assert created["status"] == "INCOMPLETE"

    verified = verify_receipt(fixture.output)

    assert verified["status"] == "INCOMPLETE"
    assert verified["ok"] is False
    assert verified["reason_codes"] == ["required_evidence_missing"]


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    original = b"preexisting-user-output\n"
    fixture.output.write_bytes(original)

    with pytest.raises(ExistingEvidenceError) as caught:
        _receipt(fixture)

    assert caught.value.code == "output_already_exists"
    assert fixture.output.read_bytes() == original


def test_output_inside_attested_tool_tree_is_rejected(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    root = Path(__file__).resolve().parents[1]
    forbidden = root / f"forbidden-receipt-{tmp_path.name}.json"
    assert not forbidden.exists()

    with pytest.raises(ExistingEvidenceError) as caught:
        create_receipt(
            run_dir=fixture.run_dir,
            canary_dir=fixture.canary_dir,
            self_references=(fixture.self_reference,),
            output=forbidden,
        )

    assert caught.value.code == "output_inside_source"
    assert not forbidden.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_and_extra_canary_invocations_are_distinguished(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    path = fixture.canary_dir / "canary_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload["runtime"]["invocations"] = []
    else:
        extra = dict(payload["runtime"]["invocations"][0])
        extra["model_id"] = "foreign-model"
        payload["runtime"]["invocations"].append(extra)
    _write_json(path, payload)

    receipt = _receipt(fixture)

    if mutation == "missing":
        assert receipt["status"] == "INCOMPLETE"
        assert receipt["reason_codes"] == [
            "required_canary_model_missing"
        ]
    else:
        assert receipt["status"] == "CONFLICT"
        assert receipt["reason_codes"] == [
            "unexpected_canary_invocation"
        ]


@pytest.mark.parametrize(
    ("field", "value", "reason_code"),
    (
        ("schema", "onnx-splitpoint/foreign-output", "schema_mismatch"),
        ("schema_version", 3, "schema_version_mismatch"),
        ("model", "foreign-model", "output_manifest_identity_mismatch"),
        ("task", "classification", "output_manifest_identity_mismatch"),
        ("case", "split", "output_manifest_identity_mismatch"),
        (
            "backend",
            "native_full_tensorrt",
            "output_manifest_identity_mismatch",
        ),
        (
            "comparison_backend",
            "tensorrt",
            "output_manifest_identity_mismatch",
        ),
        (
            "setup_id",
            "foreign-setup",
            "output_manifest_identity_mismatch",
        ),
    ),
)
def test_output_manifest_base_identity_is_strictly_bound(
    tmp_path: Path,
    field: str,
    value: object,
    reason_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(
        fixture.output_manifest.read_text(encoding="utf-8")
    )
    payload[field] = value
    _write_json(fixture.output_manifest, payload)
    _reseal_output_manifest_self_reference(fixture)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [reason_code]


@pytest.mark.parametrize(
    ("mutation", "reason_code"),
    (
        ("missing_name", "output_tensor_metadata_invalid"),
        ("empty_shape", "output_tensor_metadata_invalid"),
        ("zero_dimension", "output_tensor_metadata_invalid"),
        ("oversized_dimension", "output_tensor_metadata_invalid"),
        ("excessive_rank", "output_tensor_metadata_invalid"),
        ("object_dtype", "output_tensor_metadata_invalid"),
        ("shape_dtype_size", "output_tensor_size_mismatch"),
    ),
)
def test_output_tensor_shape_dtype_and_size_are_plausible_and_bound(
    tmp_path: Path,
    mutation: str,
    reason_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(
        fixture.output_manifest.read_text(encoding="utf-8")
    )
    row = payload["outputs"][0]
    if mutation == "missing_name":
        row["name"] = ""
    elif mutation == "empty_shape":
        row["shape"] = []
    elif mutation == "zero_dimension":
        row["shape"] = [1, 0, 12]
    elif mutation == "oversized_dimension":
        row["shape"] = [1 << 31]
    elif mutation == "excessive_rank":
        row["shape"] = [1] * 9
    elif mutation == "object_dtype":
        row["dtype"] = "object"
    else:
        row["shape"] = [1, 3, 5]
    _write_json(fixture.output_manifest, payload)
    _reseal_output_manifest_self_reference(fixture)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [reason_code]


@pytest.mark.parametrize("mutation", ("missing", "extra", "duplicate", "wrong"))
def test_canary_benchmark_sets_are_the_exact_source_run_set(
    tmp_path: Path,
    mutation: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    path = fixture.canary_dir / "canary_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing":
        payload["benchmark_sets"] = []
    elif mutation == "duplicate":
        payload["benchmark_sets"].append(payload["benchmark_sets"][0])
    else:
        foreign = tmp_path / "foreign-benchmark-set"
        foreign.mkdir()
        if mutation == "extra":
            payload["benchmark_sets"].append(str(foreign.resolve()))
        else:
            payload["benchmark_sets"] = [str(foreign.resolve())]
    _write_json(path, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["canary_benchmark_sets_mismatch"]


def test_canary_source_benchmark_set_symlink_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    real_suite = fixture.benchmark_set.with_name("real-legacy-suite")
    fixture.benchmark_set.rename(real_suite)
    try:
        fixture.benchmark_set.symlink_to(
            real_suite,
            target_is_directory=True,
        )
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert any(
        token in _reason_text(receipt)
        for token in ("symlink", "unsafe_input_directory")
    )


@pytest.mark.parametrize("source", ("candidate", "reference"))
def test_physical_quality_ground_truth_records_are_cross_bound(
    tmp_path: Path,
    source: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    summary_path = (
        fixture.run_dir
        / "quality_management/central_quality_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hailo_row = next(
        row for row in summary["results"] if row["backend"] == "hailo8"
    )
    if source == "candidate":
        candidate = json.loads(fixture.candidate.read_text(encoding="utf-8"))
        candidate["records"][0]["ground_truth"] = [{"class_id": 7}]
        _write_json(fixture.candidate, candidate)
        request = json.loads(
            fixture.quality_request.read_text(encoding="utf-8")
        )
        request["candidate"]["sha256"] = _file_sha256(fixture.candidate)
        request["candidate"]["size_bytes"] = fixture.candidate.stat().st_size
        _write_json(fixture.quality_request, request)
        hailo_row["source_request_sha256"] = (
            f"sha256:{_file_sha256(fixture.quality_request)}"
        )
    else:
        reference_path = fixture.run_dir / (
            "quality_management/references/yolo11l/"
            "canonical_cpu_reference.json"
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference["records"][0]["ground_truth"] = [{"class_id": 7}]
        _write_json(reference_path, reference)
        hailo_row["management_cpu_reference"]["reference_sha256"] = (
            _file_sha256(reference_path)
        )
    _write_json(summary_path, summary)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["quality_ground_truth_mismatch"]


def test_physical_candidate_image_ids_are_bound_to_reference_records(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    candidate = json.loads(fixture.candidate.read_text(encoding="utf-8"))
    candidate["records"][0]["image_id"] = "999999999999.jpg"
    candidate["image_ids_sha256"] = image_ids_fingerprint([
        row["image_id"] for row in candidate["records"]
    ])
    candidate_prediction_sha = prediction_fingerprint(
        candidate["records"],
        payload_field="candidate",
    )
    candidate["prediction_sha256"] = candidate_prediction_sha
    _write_json(fixture.candidate, candidate)

    request = json.loads(
        fixture.quality_request.read_text(encoding="utf-8")
    )
    request["candidate"].update({
        "sha256": _file_sha256(fixture.candidate),
        "size_bytes": fixture.candidate.stat().st_size,
        "prediction_sha256": candidate_prediction_sha,
    })
    _write_json(fixture.quality_request, request)
    summary_path = (
        fixture.run_dir
        / "quality_management/central_quality_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hailo_row = next(
        row for row in summary["results"] if row["backend"] == "hailo8"
    )
    hailo_row["candidate_predictions_sha256"] = candidate_prediction_sha
    hailo_row["source_request_sha256"] = (
        f"sha256:{_file_sha256(fixture.quality_request)}"
    )
    _write_json(summary_path, summary)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["prediction_payload_invalid"]


@pytest.mark.parametrize(
    ("field", "reason_code"),
    (
        ("annotations_sha256", "quality_ground_truth_mismatch"),
        ("validation_dataset_sha256", "quality_contract_binding_mismatch"),
    ),
)
def test_quality_annotation_and_contract_manifest_identities_are_bound(
    tmp_path: Path,
    field: str,
    reason_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    summary_path = (
        fixture.run_dir
        / "quality_management/central_quality_summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    hailo_row = next(
        row for row in summary["results"] if row["backend"] == "hailo8"
    )
    hailo_row[field] = "0" * 64
    _write_json(summary_path, summary)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [reason_code]


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    (
        (
            "stale_hash",
            "frozen_postprocess_invariant_sha256_mismatch",
        ),
        (
            "outer_divergence",
            "frozen_postprocess_invariant_identity_mismatch",
        ),
    ),
)
def test_frozen_postprocess_invariant_identity_is_canonically_bound(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    output_payload = json.loads(
        fixture.output_manifest.read_text(encoding="utf-8")
    )
    native_payload = json.loads(
        fixture.native_report.read_text(encoding="utf-8")
    )
    semantic = json.loads(
        fixture.self_reference.read_text(encoding="utf-8")
    )
    contracts = (
        output_payload["frozen_host_postprocess_contract"],
        native_payload["frozen_host_postprocess_contract"],
        semantic["native_output_manifest_meta"][
            "frozen_host_postprocess_contract"
        ],
    )
    for contract in contracts:
        invariant = contract["invariant_identity"]
        invariant["confidence_threshold"] = 0.35
        if mutation == "stale_hash":
            contract["confidence_threshold"] = 0.35
        else:
            contract["invariant_contract_sha256"] = (
                _canonical_sha256(invariant)
            )
    if mutation == "outer_divergence":
        native_payload[
            "frozen_host_postprocess_invariant_contract_sha256"
        ] = contracts[1]["invariant_contract_sha256"]

    _write_json(fixture.output_manifest, output_payload)
    _write_json(fixture.native_report, native_payload)
    semantic["native_output_manifest_sha256"] = _file_sha256(
        fixture.output_manifest
    )
    semantic["native_report_sha256"] = _file_sha256(
        fixture.native_report
    )
    _write_json(fixture.self_reference, semantic)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [expected_code]


def test_hef_receipt_preprocessing_body_is_canonically_hashed(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(fixture.hef_receipt.read_text(encoding="utf-8"))
    payload["preprocessing_contract"]["pad_value"] = 115
    _write_json(fixture.hef_receipt, payload)
    relative = str(fixture.hef_receipt.relative_to(fixture.run_dir))
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [
        "preprocessing_contract_sha256_mismatch"
    ]


@pytest.mark.parametrize(
    ("scope", "field", "value", "expected_code"),
    (
        (
            "document",
            "model_id",
            "foreign-model",
            "output_contract_identity_mismatch",
        ),
        (
            "document",
            "task",
            "classification",
            "output_contract_identity_mismatch",
        ),
        (
            "contract",
            "model_id",
            "foreign-model",
            "output_contract_missing",
        ),
        (
            "contract",
            "task",
            "classification",
            "output_contract_identity_mismatch",
        ),
        (
            "contract",
            "backend",
            "tensorrt",
            "output_contract_missing",
        ),
        (
            "contract",
            "variant",
            "split",
            "output_contract_missing",
        ),
    ),
)
def test_output_contract_semantic_identity_is_strict(
    tmp_path: Path,
    scope: str,
    field: str,
    value: str,
    expected_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = str(
        (fixture.benchmark_set / "output_contracts.json").relative_to(
            fixture.run_dir
        )
    )
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload if scope == "document" else payload["contracts"][0]
    target[field] = value
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [expected_code]


def test_output_contract_recorded_hef_size_is_strict(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    relative = str(
        (fixture.benchmark_set / "output_contracts.json").relative_to(
            fixture.run_dir
        )
    )
    path = fixture.run_dir / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contracts"][0]["recorded_artifact_size_bytes"] += 1
    _write_json(path, payload)
    _reseal_artifact_index_path(fixture, relative)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [
        "output_contract_artifact_size_mismatch"
    ]


def test_canary_completed_hashes_are_optional_for_legacy_canary(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    path = fixture.canary_dir / "canary_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_result = payload["model_results"][0]
    for field in (
        "completed_task_endpoint_contract_hash",
        "completed_task_comparison_endpoint_contract_hash",
        "completed_task_result_artifact_sha256",
        "completed_detections_sha256",
    ):
        model_result.pop(field, None)
    _write_json(path, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "VERIFIED"
    assert receipt["reason_codes"] == []


@pytest.mark.parametrize(
    ("field", "expected_code"),
    (
        (
            "completed_task_comparison_endpoint_contract_hash",
            "canary_completed_endpoint_binding_mismatch",
        ),
        (
            "completed_task_result_artifact_sha256",
            "canary_completed_artifact_binding_mismatch",
        ),
        (
            "completed_detections_sha256",
            "canary_completed_detections_binding_mismatch",
        ),
    ),
)
def test_present_canary_completed_hashes_are_strictly_bound(
    tmp_path: Path,
    field: str,
    expected_code: str,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    path = fixture.canary_dir / "canary_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_results"][0][field] = "f" * 64
    _write_json(path, payload)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [expected_code]


def test_output_tensors_bind_the_frozen_raw_signature(tmp_path: Path) -> None:
    fixture = _build_existing_evidence(tmp_path)
    payload = json.loads(
        fixture.output_manifest.read_text(encoding="utf-8")
    )
    payload["outputs"][0]["name"] = "plausible-but-unsealed-output"
    _write_json(fixture.output_manifest, payload)
    _reseal_output_manifest_self_reference(fixture)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == [
        "raw_output_tensor_signature_mismatch"
    ]


def test_embedded_output_manifest_meta_must_equal_physical_manifest(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    semantic = json.loads(
        fixture.self_reference.read_text(encoding="utf-8")
    )
    semantic["native_output_manifest_meta"]["setup_id"] = "forged-setup"
    _write_json(fixture.self_reference, semantic)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["self_reference_meta_mismatch"]


def test_absolute_input_dump_provenance_cannot_redirect_the_read(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    outside = tmp_path / "outside/input_rgb_uint8.bin"
    outside.parent.mkdir()
    outside.write_bytes(b"untrusted-outside-tensor")
    boundary = json.loads(
        fixture.boundary_manifest.read_text(encoding="utf-8")
    )
    boundary["input_dump"] = str(outside.resolve())
    _write_json(fixture.boundary_manifest, boundary)
    semantic = json.loads(
        fixture.self_reference.read_text(encoding="utf-8")
    )
    semantic["boundary_manifest_sha256"] = _file_sha256(
        fixture.boundary_manifest
    )
    _write_json(fixture.self_reference, semantic)

    receipt = _receipt(fixture)

    assert receipt["status"] == "VERIFIED"
    input_rows = [
        row
        for row in receipt["evidence_projection"]["artifacts"]
        if row["logical_path"] == "canary/yolo11l/input_rgb_uint8.bin"
    ]
    assert len(input_rows) == 1
    assert input_rows[0]["sha256"] != _file_sha256(outside)


def test_input_dump_provenance_with_wrong_basename_fails_closed(
    tmp_path: Path,
) -> None:
    fixture = _build_existing_evidence(tmp_path)
    boundary = json.loads(
        fixture.boundary_manifest.read_text(encoding="utf-8")
    )
    boundary["input_dump"] = str(
        (tmp_path / "outside/foreign_tensor.bin").resolve()
    )
    _write_json(fixture.boundary_manifest, boundary)
    semantic = json.loads(
        fixture.self_reference.read_text(encoding="utf-8")
    )
    semantic["boundary_manifest_sha256"] = _file_sha256(
        fixture.boundary_manifest
    )
    _write_json(fixture.self_reference, semantic)

    receipt = _receipt(fixture)

    assert receipt["status"] == "CONFLICT"
    assert receipt["reason_codes"] == ["unsafe_manifest_path"]
