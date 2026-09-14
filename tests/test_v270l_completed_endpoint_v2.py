from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    FrozenDecodedNmsPostprocessor,
    FrozenPostprocessError,
    build_completed_detection_comparison_endpoint_contract,
    build_completed_detection_endpoint_attestation,
    build_frozen_decoded_nms_normalization_contract,
    build_frozen_postprocess_contract,
    build_normalized_detection_endpoint_attestation,
    canonical_json_sha256,
    frozen_postprocess_invariant_identity,
    tensor_signature,
    verify_completed_detection_comparison_endpoint_contract,
    verify_frozen_decoded_nms_normalization_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from scripts import native_producer_validate_visualize as validator


def _persist_completed_result_artifact(
    evidence: dict,
    result: dict,
    artifact_path: Path,
) -> None:
    artifact = result["completed_result_artifact"]
    artifact_json = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
    )
    artifact_path.write_text(artifact_json, encoding="utf-8")
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


def _raw_yolov7_outputs() -> dict[str, np.ndarray]:
    return {
        "output": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "clone_1": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "clone_2": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }


def _raw_yolov7_outputs_with_detection() -> dict[str, np.ndarray]:
    outputs = _raw_yolov7_outputs()
    strong = outputs["output"][0, 0, 0, 0]
    strong[:4] = 0.0
    strong[4] = 20.0
    strong[7] = 20.0
    return outputs


def _raw_yolo11_outputs_with_detection() -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for level, side in enumerate((80, 40, 20)):
        outputs[f"reg{level}"] = np.zeros(
            (side, side, 64), dtype=np.float32,
        )
        outputs[f"cls{level}"] = np.full(
            (side, side, 80), -20.0, dtype=np.float32,
        )
    outputs["cls0"][10, 10, 11] = 20.0
    return outputs


def _full_yolo11_ultralytics_decoded(
    *, class_id: int = 11,
) -> dict[str, np.ndarray]:
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    # The synthetic raw DFL16 head above decodes cell (10,10) at stride 8
    # to xyxy=[24,24,144,144].  Express the same record as Full-ONNX
    # decoded-pre-NMS xywh + 80 class probabilities.
    output[0, 0:4, 0] = [84.0, 84.0, 120.0, 120.0]
    output[0, 4 + int(class_id), 0] = 1.0
    return {"output0": output}


def _raw_contract() -> dict:
    return build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        outputs=_raw_yolov7_outputs(),
        input_hw=[640, 640],
        original_wh=[80, 60],
    )


def _direct_source_attestation(
    outputs: dict[str, np.ndarray],
    source_hash: str,
) -> dict:
    return {
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


def _direct_contract() -> tuple[dict, dict[str, np.ndarray]]:
    outputs = {
        "detections": np.asarray(
            [[
                [80.0, 160.0, 320.0, 480.0, 0.90, 2.0],
                [80.0, 160.0, 320.0, 480.0, 0.80, 2.0],
                [0.0, 0.0, 40.0, 40.0, 0.10, 1.0],
            ]],
            dtype=np.float32,
        ),
    }
    source_hash = "a" * 64
    source_attestation = _direct_source_attestation(
        outputs, source_hash,
    )
    contract = build_frozen_decoded_nms_normalization_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
        preprocess={
            "mode": "letterbox_rgb_uint8",
            "rgb": True,
            "pad_value": 114,
        },
        source_coordinate_space=(
            "model_input_letterbox_xyxy_pixels"
        ),
        source_endpoint_contract_hash=source_hash,
        source_output_endpoint_attestation=source_attestation,
    )
    return contract, outputs


def _archived_v27546_raw_contract() -> dict:
    """Exact read-only pre-model-bound decoder identity for v1 evidence."""
    outputs = {
        "output": np.full((1, 3, 2, 2, 85), -20.0, dtype=np.float32),
        "clone_1": np.full((1, 3, 1, 1, 85), -20.0, dtype=np.float32),
        "clone_2": np.full((1, 3, 1, 1, 85), -20.0, dtype=np.float32),
    }
    contract = {
        "schema": "onnx-splitpoint/frozen-native-detection-postprocess",
        "schema_version": 1,
        "task": "detection",
        "source_contract_family": "raw_head",
        "output_contract_family": "decoded_nms",
        "model_id": "yolov7_paper",
        "model_family": "yolov7",
        "decoder_id": "yolov7_multiscale_anchor_classaware_nms_v1",
        "decoder_format": "multiscale_head",
        "multiscale_activation_mode": "logits",
        "nms_implementation": "numpy_class_aware_nms_xyxy_v1",
        "class_aware": True,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "input_hw": [64, 64],
        "original_wh": [80, 60],
        "raw_output_tensor_signature": tensor_signature(outputs),
        "implementation_artifacts": {
            "harness_base": {
                "relative_path": "onnx_splitpoint_tool/runners/harness/base.py",
                "sha256": "938b3713f32e412dd1f57fcc07a397fa1f3b26ed30afbc945d22f1bb082db06c",
            },
            "native_detection_postprocess": {
                "relative_path": "onnx_splitpoint_tool/native_detection_postprocess.py",
                "sha256": "2ab5b42dfe741e084e8575da943c22c26f372202879f62070ad566e09dd65a8e",
            },
            "yolo_harness": {
                "relative_path": "onnx_splitpoint_tool/runners/harness/yolo.py",
                "sha256": "53228a2ce59c2f15f3ea025e3d52a2312ea121a88fc59982e7536372c05cbfcd",
            },
        },
        "execution_policy": "one_serial_host_tail_per_successful_accelerator_output",
        "performance_scope": "prepared_input_to_post_nms_detections",
        "energy_scope": "same_frozen_prepared_input_to_post_nms_hotloop",
        "host_postprocess_frozen": True,
        "postprocess_included": True,
        "invariant_identity": {},
    }
    contract["invariant_identity"] = frozen_postprocess_invariant_identity(
        contract
    )
    contract["invariant_contract_sha256"] = canonical_json_sha256(
        contract["invariant_identity"]
    )
    contract["contract_sha256"] = canonical_json_sha256(contract)
    return contract


def _reseal_comparison(contract: dict) -> dict:
    identity = {
        key: value
        for key, value in contract.items()
        if key not in {
            "endpoint_contract_complete",
            "endpoint_contract_hash",
            "output_endpoint_id",
        }
    }
    digest = canonical_json_sha256(identity)
    return {
        **identity,
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": digest,
        "output_endpoint_id": (
            f"detection:decoded_nms:comparison:{digest}"
        ),
    }


def test_raw_and_direct_v2_share_semantic_endpoint_only() -> None:
    raw = _raw_contract()
    direct, _outputs = _direct_contract()

    raw_endpoint = (
        build_completed_detection_comparison_endpoint_contract(raw)
    )
    direct_endpoint = (
        build_completed_detection_comparison_endpoint_contract(
            direct_normalization_contract=direct,
        )
    )

    assert raw_endpoint == direct_endpoint
    assert raw_endpoint["schema_version"] == 2
    assert direct["source_output_endpoint_id"] != raw_endpoint[
        "output_endpoint_id"
    ]
    assert direct["source_output_tensor_signature"] != raw[
        "raw_output_tensor_signature"
    ]


def test_direct_bn6_is_filtered_nmsed_and_deletterboxed() -> None:
    contract, outputs = _direct_contract()
    processor = FrozenDecodedNmsPostprocessor(contract)

    result = processor.process(outputs, original_wh=[80, 60])

    assert result["detection_count"] == 1
    assert processor.completed_count == 1
    assert [
        processor.last_detections[0][key]
        for key in ("x1", "y1", "x2", "y2")
    ] == pytest.approx([10.0, 10.0, 40.0, 50.0])
    assert processor.last_detections[0]["score"] == pytest.approx(0.9)
    assert processor.last_detections[0]["class_id"] == 2


def test_direct_geometry_and_source_space_tamper_fail_closed() -> None:
    contract, _outputs = _direct_contract()

    geometry_tamper = copy.deepcopy(contract)
    geometry_tamper["letterbox_geometry_contract"]["pad_top"] += 1
    unsigned = dict(geometry_tamper)
    unsigned.pop("contract_sha256")
    geometry_tamper["contract_sha256"] = canonical_json_sha256(unsigned)
    with pytest.raises(FrozenPostprocessError):
        verify_frozen_decoded_nms_normalization_contract(
            geometry_tamper
        )

    source_tamper = copy.deepcopy(contract)
    source_tamper["source_coordinate_space"] = (
        "original_image_xyxy_pixels"
    )
    unsigned = dict(source_tamper)
    unsigned.pop("contract_sha256")
    source_tamper["contract_sha256"] = canonical_json_sha256(unsigned)
    with pytest.raises(FrozenPostprocessError):
        verify_frozen_decoded_nms_normalization_contract(source_tamper)


def test_direct_completion_count_is_exact_and_fail_closed() -> None:
    contract, outputs = _direct_contract()
    result = FrozenDecodedNmsPostprocessor(contract).process(
        outputs, original_wh=[80, 60],
    )
    attestation = build_normalized_detection_endpoint_attestation(
        contract,
        result,
        completed_frames=3,
        postprocess_completed_frames=3,
    )
    assert attestation["attested"] is True
    assert attestation["postprocess_completion_verified"] is True
    assert attestation["completed_task_completion_mode"] == (
        "integrated_accelerator_plus_frozen_normalization"
    )

    with pytest.raises(FrozenPostprocessError):
        build_normalized_detection_endpoint_attestation(
            contract,
            result,
            completed_frames=3,
            postprocess_completed_frames=2,
        )


def test_v1_identity_remains_exactly_reproducible() -> None:
    endpoint = build_completed_detection_comparison_endpoint_contract(
        _archived_v27546_raw_contract(),
        schema_version=1,
    )
    assert endpoint["endpoint_contract_hash"] == (
        "db6ef57537b2a539de6b443d65f2282d9511fa41db1c6f33793c7b7805be2553"
    )
    assert verify_completed_detection_comparison_endpoint_contract(
        endpoint,
        frozen_contract=_archived_v27546_raw_contract(),
    ) == endpoint


def test_v1_resealed_threshold_and_extra_key_tamper_fail_closed() -> None:
    endpoint = build_completed_detection_comparison_endpoint_contract(
        _archived_v27546_raw_contract(),
        schema_version=1,
    )

    threshold_tamper = copy.deepcopy(endpoint)
    threshold_tamper["score_threshold"] = 0.20
    threshold_tamper = _reseal_comparison(threshold_tamper)
    with pytest.raises(FrozenPostprocessError):
        verify_completed_detection_comparison_endpoint_contract(
            threshold_tamper
        )

    key_tamper = copy.deepcopy(endpoint)
    key_tamper["unexpected"] = "accepted-only-by-hash"
    key_tamper = _reseal_comparison(key_tamper)
    with pytest.raises(FrozenPostprocessError):
        verify_completed_detection_comparison_endpoint_contract(
            key_tamper
        )


@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
def test_comparison_schema_version_never_coerces(value: object) -> None:
    endpoint = build_completed_detection_comparison_endpoint_contract(
        _archived_v27546_raw_contract(),
        schema_version=1,
    )
    endpoint["schema_version"] = value
    endpoint = _reseal_comparison(endpoint)
    with pytest.raises(FrozenPostprocessError):
        verify_completed_detection_comparison_endpoint_contract(endpoint)


def test_validator_accepts_yolo26_c4_c80_canonical_decoder() -> None:
    outputs: dict[str, np.ndarray] = {}
    conv = 61
    for size in (8, 4, 2):
        outputs[f"yolo26s_full/conv{conv}"] = np.zeros(
            (size, size, 4), dtype=np.float32,
        )
        outputs[f"yolo26s_full/conv{conv + 3}"] = np.full(
            (size, size, 80), -20.0, dtype=np.float32,
        )
        conv += 16

    candidates = validator._decode_layout_candidates(
        outputs, img_w=64, img_h=64, conf=0.25,
    )
    canonical = [
        candidate
        for candidate in candidates
        if (
            candidate.get("debug") or {}
        ).get("decoder_format") == "ultralytics_regcls"
    ]

    assert len(canonical) == 1
    assert canonical[0]["mode"] == "canonical_multiscale:raw"
    assert validator._detection_contract_family(
        canonical[0]["mode"]
    ) == "raw_head"


def test_validator_nms_cap_matches_canonical_policy() -> None:
    detections = [
        {
            "x1": float(index * 2),
            "y1": 0.0,
            "x2": float(index * 2 + 1),
            "y2": 1.0,
            "score": float(400 - index) / 400.0,
            "class_id": 0,
        }
        for index in range(350)
    ]
    assert len(validator._nms(detections)) == 300


def _completed_raw_endpoint_evidence(
    artifact_dir: Path,
    outputs: dict[str, np.ndarray] | None = None,
    *,
    model_id: str = "yolov7_paper",
    model_sha256: str = YOLOV7_PAPER_ONNX_SHA256,
    original_wh: tuple[int, int] = (80, 60),
) -> tuple[
    dict, dict[str, np.ndarray],
]:
    outputs = (
        outputs
        if outputs is not None
        else _raw_yolov7_outputs()
    )
    frozen = build_frozen_postprocess_contract(
        model_id=model_id,
        model_sha256=model_sha256,
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=list(original_wh),
    )
    processor = FrozenDetectionPostprocessor(frozen)
    result = processor.process(
        outputs, original_wh=list(original_wh),
    )
    attestation = build_completed_detection_endpoint_attestation(
        frozen,
        result,
        completed_frames=3,
        postprocess_completed_frames=3,
        source_endpoint_contract_hash="b" * 64,
    )
    comparison = attestation[
        "completed_task_comparison_endpoint_contract"
    ]
    evidence = {
        "frozen_host_postprocess_contract": frozen,
        "frozen_host_postprocess_contract_sha256": frozen[
            "contract_sha256"
        ],
        "frozen_host_postprocess_result": result,
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
    _persist_completed_result_artifact(
        evidence,
        result,
        artifact_dir / "raw_completed_result.json",
    )
    return evidence, outputs


def _completed_direct_endpoint_evidence(
    artifact_dir: Path,
    *,
    legacy_hash_only_v1: bool = False,
) -> tuple[
    dict, dict[str, np.ndarray],
]:
    direct, outputs = _direct_contract()
    source_hash = str(direct["source_endpoint_contract_hash"])
    source_attestation = _direct_source_attestation(
        outputs, source_hash,
    )
    processor = FrozenDecodedNmsPostprocessor(direct)
    result = processor.process(outputs, original_wh=[80, 60])
    if legacy_hash_only_v1:
        result = {
            key: result[key]
            for key in (
                "task",
                "contract_family",
                "decoder_format",
                "coordinate_space",
                "detection_count",
                "detections_sha256",
                "normalization_contract_sha256",
            )
        }
    completion = build_normalized_detection_endpoint_attestation(
        direct,
        result,
        completed_frames=3,
        postprocess_completed_frames=3,
        allow_legacy_hash_only_v1=legacy_hash_only_v1,
    )
    comparison = completion[
        "completed_task_comparison_endpoint_contract"
    ]
    workload = {
        "frozen_decoded_nms_normalization_contract": direct,
        "frozen_decoded_nms_normalization_contract_sha256": direct[
            "contract_sha256"
        ],
        "source_endpoint_contract_hash": source_hash,
        "source_output_endpoint_id": direct[
            "source_output_endpoint_id"
        ],
        "source_output_tensor_signature": direct[
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
        "output_endpoint_id": direct["source_output_endpoint_id"],
        "physical_output_endpoint_id": direct[
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
    if not legacy_hash_only_v1:
        _persist_completed_result_artifact(
            evidence,
            result,
            artifact_dir / "direct_completed_result.json",
        )
    return evidence, outputs


def test_validator_projects_raw_and_bn6_only_through_completed_v2(
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_raw_endpoint_evidence(tmp_path)
    full_outputs = {
        "detections": np.asarray(
            [[
                [80.0, 160.0, 320.0, 480.0, 0.9, 2.0],
                [80.0, 160.0, 320.0, 480.0, 0.8, 2.0],
                [0.0, 0.0, 40.0, 40.0, 0.1, 1.0],
            ]],
            dtype=np.float32,
        ),
    }

    result = validator._completed_v2_self_reference_detection(
        full_outputs,
        native_outputs,
        evidence,
    )

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["full_mode"] == (
        "completed_v2:full_bn6_normalized"
    )
    assert result["native_mode"] == (
        "completed_v2:frozen_host_tail"
    )
    assert result["expected_contract_family"] == "decoded_nms"
    assert result[
        "completed_task_comparison_output_endpoint_id"
    ] == evidence[
        "completed_task_comparison_output_endpoint_id"
    ]
    assert result["reference_detections"][0]["y1"] == pytest.approx(
        10.0
    )


def test_validator_consumes_yolov7_multiscale_full_reference_via_completed_v2(
    tmp_path: Path,
) -> None:
    raw_outputs = _raw_yolov7_outputs_with_detection()
    evidence, native_outputs = _completed_raw_endpoint_evidence(
        tmp_path,
        raw_outputs
    )

    result = validator._completed_v2_self_reference_detection(
        raw_outputs,
        native_outputs,
        evidence,
    )

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["full_reference_format"] == "multiscale_head"
    assert result["full_mode"] == (
        "completed_v2:full_multiscale_head_decoded_nms"
    )
    assert result["native_mode"] == (
        "completed_v2:frozen_host_tail"
    )
    assert result["reference_detections"]
    assert result["reference_detections"] == result[
        "native_detections"
    ]


def test_validator_consumes_yolo11_decoded_full_reference_via_completed_v2(
    tmp_path: Path,
) -> None:
    native_outputs = _raw_yolo11_outputs_with_detection()
    evidence, native_outputs = _completed_raw_endpoint_evidence(
        tmp_path,
        native_outputs,
        model_id="yolo11l",
        model_sha256="",
        original_wh=(640, 640),
    )

    result = validator._completed_v2_self_reference_detection(
        _full_yolo11_ultralytics_decoded(),
        native_outputs,
        evidence,
    )

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["full_reference_format"] == "ultralytics_decoded"
    assert result["full_mode"] == (
        "completed_v2:full_ultralytics_decoded_decoded_nms"
    )
    assert result["native_mode"] == (
        "completed_v2:frozen_host_tail"
    )
    parity = validator._match_detections(
        result["reference_detections"],
        result["native_detections"],
    )
    assert parity["match_ratio"] == 1.0
    assert parity["mean_iou"] == pytest.approx(1.0)

    mismatched = validator._completed_v2_self_reference_detection(
        _full_yolo11_ultralytics_decoded(class_id=12),
        native_outputs,
        evidence,
    )
    assert mismatched["available"] is True
    mismatch = validator._match_detections(
        mismatched["reference_detections"],
        mismatched["native_detections"],
    )
    assert mismatch["match_ratio"] == 0.0
    assert mismatch["matched"] == 0


def test_validator_yolov7_does_not_borrow_yolo11_full_format_allowance(
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_raw_endpoint_evidence(
        tmp_path,
        _raw_yolov7_outputs_with_detection(),
    )

    result = validator._completed_v2_self_reference_detection(
        _full_yolo11_ultralytics_decoded(),
        native_outputs,
        evidence,
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False
    assert result["reason"].endswith(
        "completed_v2_full_reference_format_not_contract_compatible:"
        "ultralytics_decoded"
    )


def test_validator_projects_direct_bn6_through_completed_v2(
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_direct_endpoint_evidence(
        tmp_path
    )

    result = validator._completed_v2_self_reference_detection(
        native_outputs,
        native_outputs,
        evidence,
    )

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["native_mode"] == (
        "completed_v2:integrated_accelerator_plus_"
        "frozen_normalization"
    )
    assert result["reference_detections"] == result[
        "native_detections"
    ]
    assert result[
        "completed_task_comparison_output_endpoint_id"
    ] == evidence[
        "completed_task_comparison_output_endpoint_id"
    ]


def test_validator_replays_only_historical_direct_v1_hash_result(
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_direct_endpoint_evidence(
        tmp_path,
        legacy_hash_only_v1=True,
    )

    result = validator._completed_v2_self_reference_detection(
        native_outputs,
        native_outputs,
        evidence,
    )

    assert result["available"] is True
    assert result["completed_v2_verified"] is True
    assert result["semantic_result_binding_status"] == (
        "exact_completed_result_identity_match"
    )
    assert result["native_detections"] == result[
        "reference_detections"
    ]
    assert result[
        "completed_task_comparison_output_endpoint_id"
    ] == evidence[
        "completed_task_comparison_output_endpoint_id"
    ]


def test_validator_direct_bn6_rejects_multiscale_full_reference(
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_direct_endpoint_evidence(
        tmp_path
    )

    result = validator._completed_v2_self_reference_detection(
        _raw_yolov7_outputs_with_detection(),
        native_outputs,
        evidence,
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False
    assert result["reason"].endswith(
        "completed_v2_full_reference_format_not_contract_compatible:"
        "multiscale_head"
    )


def test_validator_rejects_malformed_completed_v2_full_detections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_raw_endpoint_evidence(tmp_path)

    class _MalformedFullHarness:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def postprocess(
            self,
            _outputs: dict[str, np.ndarray],
            _context: dict[str, object],
        ) -> dict:
            return {
                "schema_version": 1,
                "task": "detection",
                "json": {
                    "format": "multiscale_head",
                    "detections": [{"class_id": 2}, "not-a-detection"],
                },
            }

    monkeypatch.setattr(
        validator,
        "_CanonicalYoloHarness",
        _MalformedFullHarness,
    )

    result = validator._completed_v2_self_reference_detection(
        _raw_yolov7_outputs(),
        native_outputs,
        evidence,
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False
    assert result["reason"].endswith(
        "completed_v2_full_reference_detections_invalid"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "full_contract_sha",
        "source_attestation",
        "normalization_contract",
        "completion_result",
        "comparison_id",
    ],
)
def test_validator_direct_completed_v2_tamper_fails_closed(
    mutation: str,
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_direct_endpoint_evidence(
        tmp_path
    )
    if mutation == "full_contract_sha":
        evidence["full_command_contract"]["contract_sha256"] = "c" * 64
    elif mutation == "source_attestation":
        evidence["output_endpoint_attestation"]["status"] = "failed"
    elif mutation == "normalization_contract":
        evidence["full_command_contract"]["energy_workload"][
            "frozen_decoded_nms_normalization_contract"
        ]["contract_sha256"] = "c" * 64
    elif mutation == "completion_result":
        evidence["completed_task_endpoint_attestation"][
            "frozen_decoded_nms_normalization_result"
        ]["detections_sha256"] = "c" * 64
    else:
        evidence[
            "completed_task_comparison_output_endpoint_id"
        ] = "detection:decoded_nms:comparison:" + "c" * 64

    result = validator._completed_v2_self_reference_detection(
        native_outputs,
        native_outputs,
        evidence,
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False


@pytest.mark.parametrize(
    "field",
    [
        "completed_task_comparison_endpoint_contract_hash",
        "completed_task_comparison_output_endpoint_id",
        "completed_task_completion_mode",
    ],
)
def test_validator_completed_v2_tamper_never_falls_back_to_raw(
    field: str,
    tmp_path: Path,
) -> None:
    evidence, native_outputs = _completed_raw_endpoint_evidence(tmp_path)
    if field.endswith("_hash"):
        evidence[field] = "c" * 64
    elif field.endswith("_id"):
        evidence[field] = (
            "detection:decoded_nms:comparison:" + "c" * 64
        )
    else:
        evidence[field] = "integrated_accelerator"

    result = validator._completed_v2_self_reference_detection(
        {
            "detections": np.asarray(
                [[
                    [80.0, 160.0, 320.0, 480.0, 0.9, 2.0],
                    [80.0, 160.0, 320.0, 480.0, 0.8, 2.0],
                    [0.0, 0.0, 40.0, 40.0, 0.1, 1.0],
                ]],
                dtype=np.float32,
            ),
        },
        native_outputs,
        evidence,
    )

    assert result["available"] is False
    assert result["completed_v2_verified"] is False
