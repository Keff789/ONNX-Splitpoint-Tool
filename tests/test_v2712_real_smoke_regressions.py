from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDecodedNmsPostprocessor,
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_decoded_nms_normalization_contract,
    build_frozen_postprocess_contract,
    build_normalized_detection_endpoint_attestation,
    canonical_json_sha256,
    tensor_signature,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
)
from scripts import native_producer_validate_visualize as validator


# Exact Full-row identities from the 20260727_153940 Smoke pack.  Tensor
# dimensions are reduced below so the regression remains hardware-independent.
_YOLOV7_FULL_ROWS = (
    ("native_full_deepx", "orin_nx_deepx_m1_01", "deepx"),
    ("native_full_hailo10h", "orin_nx_hailo10_01", "hailo10h"),
    ("native_full_hailo8", "orin_nx_hailo8_01", "hailo8"),
    ("native_full_tensorrt", "orin_nx_deepx_m1_01", "deepx"),
    ("native_full_tensorrt", "orin_nx_hailo10_01", "hailo10h"),
    ("native_full_tensorrt", "orin_nx_hailo8_01", "hailo8"),
)
_SPLIT_CASES = ("b044", "b064", "b066")
_SPLIT_IDENTITIES = tuple(
    (backend, case, setup_id, comparison, decision)
    for backend, setup_id, comparison, decision in (
        (
            "deepx_to_trt",
            "orin_nx_deepx_m1_01",
            "deepx",
            "inconclusive",
        ),
        (
            "hailo10h_to_trt",
            "orin_nx_hailo10_01",
            "hailo10h",
            "pass",
        ),
        (
            "hailo8_to_trt",
            "orin_nx_hailo8_01",
            "hailo8",
            "fail",
        ),
    )
    for case in _SPLIT_CASES
)
_DEEPX_FULL_ROWS = (
    ("resnet50", "classification"),
    ("yolo26s", "detection"),
    ("yolov7_paper", "detection"),
)
_ENDPOINT_SHA256 = (
    "051f0d9e489355d312ad656532acf3ed8a2719774d8ef9843278818f5bd15adc"
)
_RAW_HEAD_GAP = "raw_head_decoder_postprocess_contract_unresolved"


def _raw_yolov7_outputs() -> dict[str, np.ndarray]:
    outputs = {
        "output": np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
        "clone_1": np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
        "clone_2": np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
    }
    candidate = outputs["output"][0, 0, 0, 0]
    candidate[:4] = 0.0
    candidate[4] = 20.0
    candidate[7] = 20.0
    return outputs


def _completed_multiscale_evidence(
    outputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    frozen = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
        confidence_threshold=0.25,
        iou_threshold=0.45,
        max_detections=300,
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    processor = FrozenDetectionPostprocessor(frozen)
    result = processor.process(outputs, original_wh=[80, 60])
    attestation = build_completed_detection_endpoint_attestation(
        frozen,
        result,
        completed_frames=100,
        postprocess_completed_frames=100,
        source_endpoint_contract_hash="b" * 64,
    )
    # The real Smoke pack used separate Performance and Semantic invocations.
    # Model that portable hash-only mismatch while preserving every other
    # archived completed-v2 field and the local replay result.  Explicitly
    # remove the newer exact sentinel artifact so this remains a V1 fixture.
    sealed = attestation["frozen_postprocess_result"]
    for field in (
        "coordinate_space",
        "record_schema",
        "canonical_sort_policy",
        "detections",
        "completed_result_artifact",
        "completed_result_artifact_sha256",
    ):
        sealed.pop(field, None)
    sealed["detections_sha256"] = "c" * 64
    comparison = attestation[
        "completed_task_comparison_endpoint_contract"
    ]
    return {
        "frozen_host_postprocess_contract": frozen,
        "frozen_host_postprocess_contract_sha256": frozen[
            "contract_sha256"
        ],
        "frozen_host_postprocess_result": copy.deepcopy(sealed),
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash": comparison[
            "endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": comparison[
            "output_endpoint_id"
        ],
    }


def _screening_policy() -> AccuracyGatePolicy:
    return AccuracyGatePolicy(
        dataset_tier="screening",
        frozen_before_final_campaign=False,
        screening_eligible_for_ranking=False,
        contract_only_eligible_for_ranking=False,
    )


@pytest.mark.parametrize(
    ("backend", "setup_id", "comparison_backend"),
    _YOLOV7_FULL_ROWS,
)
def test_real_smoke_yolov7_full_multiscale_completed_v2_is_consumed(
    backend: str,
    setup_id: str,
    comparison_backend: str,
) -> None:
    outputs = _raw_yolov7_outputs()
    evidence = _completed_multiscale_evidence(outputs)

    result = validator._completed_v2_self_reference_detection(
        outputs,
        outputs,
        evidence,
        policy=_screening_policy(),
    )
    row = {
        "backend": backend,
        "model": "yolov7_paper",
        "case": "full",
        "setup_id": setup_id,
        "comparison_backend": comparison_backend,
        "claim_eligible": True,
        "e2e_claim_eligible": True,
        "performance_claim_eligible": True,
        "energy_claim_eligible": True,
        "scientific_claim_eligible": True,
        "thesis_claim_eligible": True,
        "eligible_for_ranking": True,
        "ranking_eligible": True,
        "performance_eligible": True,
        "energy_eligible": True,
        "pareto_eligible": True,
        "thesis_comparison_eligible": True,
        "thesis_valid": True,
    }
    validator._apply_completed_v2_semantic_binding(row, result)

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["full_reference_format"] == "multiscale_head"
    assert result["full_mode"] == (
        "completed_v2:full_multiscale_head_decoded_nms"
    )
    assert result["native_mode"] == "completed_v2:frozen_host_tail"
    assert result["portable_result_hash_mismatch"] is True
    assert result["completed_v2_exact_result_claim_binding"] is False
    assert result["completed_v2_semantic_evidence_tier"] == (
        "development_screening_portable_replay"
    )
    assert row["semantic_result_binding_status"] == (
        "portable_result_hash_mismatch"
    )
    assert all(
        row[field] is False
        for field in (
            "claim_eligible",
            "e2e_claim_eligible",
            "performance_claim_eligible",
            "energy_claim_eligible",
            "scientific_claim_eligible",
            "thesis_claim_eligible",
            "eligible_for_ranking",
            "ranking_eligible",
            "performance_eligible",
            "energy_eligible",
            "pareto_eligible",
            "thesis_comparison_eligible",
            "thesis_valid",
        )
    )


def _direct_bn6_evidence(
    artifact_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
]:
    outputs = {
        "detections": np.asarray(
            [[[8.0, 16.0, 32.0, 48.0, 0.9, 2.0]]],
            dtype=np.float32,
        ),
    }
    source_hash = "a" * 64
    source_attestation = {
        "schema": "onnx-splitpoint/runtime-output-endpoint-attestation",
        "schema_version": 3,
        "attested": True,
        "status": "passed",
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "endpoint_contract_hash": source_hash,
        "tensor_signature": tensor_signature(outputs),
        "declared_contract": {
            "model_id": "yolov7_paper",
            "source_coordinate_space": (
                "model_input_letterbox_xyxy_pixels"
            ),
        },
    }
    contract = build_frozen_decoded_nms_normalization_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[64, 64],
        original_wh=[80, 60],
        preprocess={
            "mode": "letterbox_rgb_uint8",
            "rgb": True,
            "pad_value": 114,
        },
        source_coordinate_space="model_input_letterbox_xyxy_pixels",
        source_endpoint_contract_hash=source_hash,
        source_output_endpoint_attestation=source_attestation,
    )
    processor = FrozenDecodedNmsPostprocessor(contract)
    result = processor.process(outputs, original_wh=[80, 60])
    completion = build_normalized_detection_endpoint_attestation(
        contract,
        result,
        completed_frames=100,
        postprocess_completed_frames=100,
    )
    comparison = completion[
        "completed_task_comparison_endpoint_contract"
    ]
    workload = {
        "frozen_decoded_nms_normalization_contract": contract,
        "frozen_decoded_nms_normalization_contract_sha256": contract[
            "contract_sha256"
        ],
        "source_endpoint_contract_hash": source_hash,
        "source_output_endpoint_id": contract["source_output_endpoint_id"],
        "source_output_tensor_signature": contract[
            "source_output_tensor_signature"
        ],
        "completed_task_endpoint_attestation": completion,
    }
    full_command_contract = {"energy_workload": workload}
    full_command_contract["contract_sha256"] = canonical_json_sha256(
        full_command_contract
    )
    evidence = {
        "full_command_contract": full_command_contract,
        "full_command_contract_sha256": full_command_contract[
            "contract_sha256"
        ],
        "output_endpoint_id": contract["source_output_endpoint_id"],
        "physical_output_endpoint_id": contract[
            "source_output_endpoint_id"
        ],
        "endpoint_contract_hash": source_hash,
        "physical_endpoint_contract_hash": source_hash,
        "output_endpoint_attestation": source_attestation,
        "completed_task_endpoint_attestation": completion,
        "frozen_decoded_nms_normalization_result": result,
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": (
            "source_attestation_preserved"
        ),
        "completed_task_completion_mode": (
            "integrated_accelerator_plus_frozen_normalization"
        ),
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash": comparison[
            "endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": comparison[
            "output_endpoint_id"
        ],
    }
    artifact = result["completed_result_artifact"]
    artifact_path = artifact_dir / "direct_completed_result.json"
    artifact_path.write_text(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    artifact_sha256 = canonical_json_sha256(artifact)
    evidence.update({
        "completed_task_result_artifact": artifact,
        "completed_task_result_artifact_path": str(
            artifact_path.resolve()
        ),
        "completed_task_result_artifact_saved": True,
        "completed_task_result_artifact_sha256": artifact_sha256,
        "completed_task_result_artifact_file_sha256": artifact_sha256,
    })
    return evidence, outputs


def test_smoke_direct_bn6_completion_rejects_multiscale_full_reference(
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _direct_bn6_evidence(tmp_path)

    result = validator._completed_v2_self_reference_detection(
        _raw_yolov7_outputs(),
        native_outputs,
        evidence,
        policy=_screening_policy(),
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False
    assert result["reason"].endswith(
        "completed_v2_full_reference_format_not_contract_compatible:"
        "multiscale_head"
    )


def _raw_head_attestation() -> dict[str, Any]:
    return {
        "attested": True,
        "status": "passed",
        "stage": "raw_head",
        "endpoint": "raw_head",
        "reason": "raw_detection_tensor_structure_verified",
        "endpoint_contract_hash": _ENDPOINT_SHA256,
        "tensor_signature": {
            "tensor_count": 3,
            "tensors": [
                {
                    "index": index,
                    "name": name,
                    "rank": 5,
                    "shape": shape,
                    "dtype": "float32",
                }
                for index, (name, shape) in enumerate(
                    (
                        ("output", [1, 3, 80, 80, 85]),
                        ("clone_1", [1, 3, 40, 40, 85]),
                        ("clone_2", [1, 3, 20, 20, 85]),
                    )
                )
            ],
        },
    }


def _real_split_row(
    backend: str,
    case: str,
    setup_id: str,
    comparison_backend: str,
    decision: str,
) -> dict[str, Any]:
    policy = AccuracyGatePolicy().as_dict()
    policy_sha256 = validator._canonical_json_sha256(policy)
    semantic_ok = True
    decision_alias: bool | str = (
        True
        if decision == "pass"
        else False
        if decision == "fail"
        else "inconclusive"
    )
    row: dict[str, Any] = {
        "backend": backend,
        "model": "yolov7_paper",
        "case": case,
        "setup_id": setup_id,
        "comparison_backend": comparison_backend,
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "accelerator_output_stage": "raw_head",
        "accelerator_output_contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": _ENDPOINT_SHA256,
        "accelerator_endpoint_contract_hash": _ENDPOINT_SHA256,
        "output_endpoint_attestation": _raw_head_attestation(),
        "accelerator_output_endpoint_attestation": (
            _raw_head_attestation()
        ),
        "output_manifest_sha256": "1" * 64,
        "native_command_contract_sha256": "2" * 64,
        "native_split_quality_binding_sha256": "3" * 64,
        "native_command_contract": {"fixture": "sealed-command"},
        "native_split_quality_binding": {"fixture": "sealed-binding"},
        "native_split_quality_required": True,
        "native_split_quality_binding_required": True,
        "native_split_quality_consumer_status": (
            "exact_quality_native_engine_command_and_boundary_match"
        ),
        "quality_first_binding_status": (
            "central_native_exact_engine_command_boundary_match"
        ),
        "quality_first_binding_errors": [],
        "central_quality_binding_status": "exact_identity_match",
        "central_quality_evidence_verified": True,
        "native_row_ok": True,
        "report_ok": True,
        "buildable": True,
        "runtime_executable": True,
        "execution_validation_status": "passed",
        "tensor_ok": True,
        "strict_tensor_ok": True,
        "semantic_available": True,
        "semantic_ok": semantic_ok,
        "semantic_validation_status": "passed",
        "self_reference_available": True,
        "self_reference_ok": semantic_ok,
        "numerical_similarity_pass": semantic_ok,
        "numerical_similarity_status": "passed",
        "structural_contract_pass": False,
        "contract_consistent": False,
        "structural_contract_reason": _RAW_HEAD_GAP,
        "contract_gate_reason": _RAW_HEAD_GAP,
        "claim_structural_gate_pass": False,
        "claim_structural_gate_reason": _RAW_HEAD_GAP,
        "claim_ok_structural_clamped": True,
        "accuracy_gate_policy": policy,
        "accuracy_gate_policy_sha256": policy_sha256,
        "task_quality_policy_sha256": policy_sha256,
        "runtime_quality_gate_policy_sha256": policy_sha256,
        "accuracy_gate_policy_match": True,
        "accuracy_gate_tier": "screening",
        "accuracy_gate_decision": decision,
        "accuracy_gate_pass": decision == "pass",
        "task_quality_gate": {
            "tier": "screening",
            "decision": decision,
            "status": decision,
            "policy": policy,
            "policy_sha256": policy_sha256,
        },
        "task_quality_status": decision,
        "task_quality_pass": decision_alias,
        "task_valid": decision_alias,
        "precision_quality_verified": decision == "pass",
        "host_postprocessing_evidence_status": "unavailable",
        "host_postprocessing_evidence_source": "none",
        "host_postprocessing_legacy_alias_conflict": False,
        "completed_task_endpoint_attested": None,
        "completed_task_endpoint_attestation_status": "",
        "completed_task_endpoint_attestation": {},
        "completed_task_endpoint_contract": None,
        "completed_task_stage": "",
        "completed_task_contract_family": "",
        "completed_task_endpoint_contract_hash": "",
        "completed_task_output_endpoint_id": "",
        "status": "structural_contract_failed",
        "ok": False,
        "claim_ok": False,
        "eligible_for_ranking": False,
        "e2e_claim_eligible": True,
    }
    if backend in {"hailo10h_to_trt", "hailo8_to_trt"}:
        row.update({
            "interface_contract_pass": True,
            "interface_check_pass": True,
            "interface_contract_status": (
                "verified_native_command_metadata_boundary_and_bridge"
            ),
        })
    return row


def _portable_binding_pass(**_: Any) -> tuple[dict[str, bool], str]:
    return (
        {"portable_engine_command_boundary_output_binding": True},
        "portable_binding_command_and_consumer_attestation_exact_match",
    )


@pytest.mark.parametrize(
    (
        "backend",
        "case",
        "setup_id",
        "comparison_backend",
        "decision",
    ),
    _SPLIT_IDENTITIES,
)
def test_all_nine_real_raw_head_smoke_rows_are_non_technical(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    case: str,
    setup_id: str,
    comparison_backend: str,
    decision: str,
) -> None:
    monkeypatch.setattr(
        validator,
        "bind_quality_to_native_split",
        _portable_binding_pass,
    )
    row = _real_split_row(
        backend,
        case,
        setup_id,
        comparison_backend,
        decision,
    )

    validator._apply_smoke_diagnostic_policy([row])

    assert row["ok"] is True
    assert row["status"] == (
        "diagnostic_metric_threshold_warning"
        if decision == "fail"
        else "diagnostic_technical_pass"
    )
    assert validator._technical_quality_error(row) is False
    assert row["structural_contract_pass"] is False
    assert all(
        row.get(field) is False
        for field in (
            "claim_ok",
            "eligible_for_ranking",
            "ranking_eligible",
            "performance_eligible",
            "energy_eligible",
            "performance_claim_eligible",
            "energy_claim_eligible",
            "pareto_eligible",
            "thesis_valid",
        )
    )


@pytest.mark.parametrize(("model", "task"), _DEEPX_FULL_ROWS)
def test_all_three_real_deepx_full_input_modes_survive_projection(
    tmp_path: Path,
    model: str,
    task: str,
) -> None:
    # The producer summary carried performance_input_contract_mode=explicit;
    # the corresponding archived dump independently carried
    # input_contract_mode=explicit.
    producer_row = {
        "backend": "native_full_deepx",
        "model": model,
        "case": "full",
        "setup_id": "orin_nx_deepx_m1_01",
        "comparison_backend": "deepx",
        "performance_input_contract_mode": "explicit",
    }
    dump_payload = {
        "model": model,
        "backend": "native_full_deepx",
        "input_contract_mode": "explicit",
        "contract_family": "raw_head" if task == "detection" else "logits",
    }
    performance_mode, projection_status = (
        validator._project_performance_input_contract_mode(
            producer_row,
            {},
            dump_payload,
        )
    )
    projected_row = {
        **producer_row,
        "performance_input_contract_mode": performance_mode,
        "performance_input_contract_mode_projection_status": (
            projection_status
        ),
    }
    manifest = tmp_path / f"{model}_native_full_outputs_manifest.json"
    manifest.write_text(json.dumps(dump_payload), encoding="utf-8")

    gate = validator._native_full_e2e_contract_gate(
        manifest,
        projected_row,
        task,
    )

    assert performance_mode == "explicit"
    assert projection_status == "projected_consistent"
    assert projected_row["performance_input_contract_mode"] == "explicit"
    assert gate["e2e_contract_reason"] != (
        "deepx_performance_input_contract_not_explicit"
    )
