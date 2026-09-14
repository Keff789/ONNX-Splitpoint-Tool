from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as postprocess
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    FrozenPostprocessError,
    build_detection_completion_execution_contract,
    build_frozen_postprocess_contract,
    canonical_json_sha256,
    tensor_signature,
    verify_detection_completion_execution_contract,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_DECODER_ID,
    YOLOV7_PAPER_ONNX_SHA256,
    build_yolov7_decoder_contract,
)


V2776_NATIVE_POSTPROCESS_SHA256 = (
    postprocess._V2776_NATIVE_DETECTION_POSTPROCESS_SHA256
)


def _yolo11_outputs() -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for level, side in enumerate((80, 40, 20)):
        reg_name = (
            f"/model.23/cv2.{level}/cv2.{level}.2/Conv_output_0"
        )
        cls_name = (
            f"/model.23/cv3.{level}/cv3.{level}.2/Conv_output_0"
        )
        outputs[reg_name] = np.zeros(
            (1, 64, side, side), dtype=np.float32,
        )
        outputs[cls_name] = np.full(
            (1, 80, side, side), -20.0, dtype=np.float32,
        )
    outputs[
        "/model.23/cv3.0/cv3.0.2/Conv_output_0"
    ][0, 11, 10, 10] = 20.0
    return outputs


def _yolo26_outputs() -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    for level, side in enumerate((80, 40, 20)):
        outputs[f"reg{level}"] = np.zeros(
            (side, side, 4), dtype=np.float32,
        )
        outputs[f"cls{level}"] = np.full(
            (side, side, 80), -20.0, dtype=np.float32,
        )
    return outputs


def _yolov7_outputs() -> dict[str, np.ndarray]:
    return {
        "p3": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
        "p4": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
        "p5": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
    }


def _raw_source(outputs: dict[str, np.ndarray]) -> dict[str, object]:
    signature = tensor_signature(outputs)
    endpoint_hash = "e" * 64
    return {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "endpoint": "raw_head",
            "stage": "raw_head",
            "endpoint_contract_hash": endpoint_hash,
            "tensor_signature": signature,
        },
    }


def _reseal_frozen(contract: dict[str, object]) -> dict[str, object]:
    resealed = copy.deepcopy(contract)
    resealed.pop("contract_sha256", None)
    resealed["invariant_identity"] = (
        postprocess.frozen_postprocess_invariant_identity(resealed)
    )
    resealed["invariant_contract_sha256"] = canonical_json_sha256(
        resealed["invariant_identity"]
    )
    resealed["contract_sha256"] = canonical_json_sha256(resealed)
    return resealed


def test_yolo11_family_has_explicit_dfl16_decoder_identity() -> None:
    assert postprocess._model_family("yolo11l") == "yolo11"
    assert postprocess._model_family("YOLO-11_L") == "yolo11"
    assert postprocess._expected_decoder("yolo11l") == (
        "yolo11_regcls_dfl16_classaware_nms_v1",
        "ultralytics_regcls",
    )
    with pytest.raises(
        FrozenPostprocessError,
        match="unsupported_native_full_raw_detection_model:yolo10l",
    ):
        postprocess._expected_decoder("yolo10l")


def test_yolo11_exact_endpoint_builds_and_runs_frozen_adapter() -> None:
    outputs = _yolo11_outputs()
    contract = build_frozen_postprocess_contract(
        model_id="yolo11l",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 640],
    )

    assert contract["schema_version"] == postprocess.SCHEMA_LEGACY_VERSION
    assert contract["model_id"] == "yolo11l"
    assert contract["model_family"] == "yolo11"
    assert contract["decoder_id"] == (
        "yolo11_regcls_dfl16_classaware_nms_v1"
    )
    assert contract["decoder_format"] == "ultralytics_regcls"
    assert contract["raw_output_tensor_signature"] == tensor_signature(
        outputs
    )

    processor = FrozenDetectionPostprocessor(contract)
    assert processor._harness.model_id == "yolo11l"
    result = processor.process(outputs, original_wh=[640, 640])

    assert result["detection_count"] == 1
    assert result["detections"][0]["class_id"] == 11
    assert result["detections"][0]["score"] > 0.99
    assert processor.completed_count == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda outputs: outputs.pop(
                "/model.23/cv3.2/cv3.2.2/Conv_output_0"
            ),
            "exact_six_tensors_required",
        ),
        (
            lambda outputs: outputs.__setitem__(
                "/model.23/cv2.0/cv2.0.2/Conv_output_0",
                np.zeros((1, 4, 80, 80), dtype=np.float32),
            ),
            "tensor_geometry_invalid",
        ),
        (
            lambda outputs: outputs.__setitem__(
                "/model.23/cv2.2/cv2.2.2/Conv_output_0",
                np.zeros((1, 64, 40, 40), dtype=np.float32),
            ),
            "tensor_geometry_duplicate",
        ),
    ],
    ids=["missing-level", "yolo26-direct-ltrb", "duplicate-level"],
)
def test_yolo11_endpoint_guard_fails_closed(
    mutation: object, reason: str,
) -> None:
    outputs = _yolo11_outputs()
    mutation(outputs)

    with pytest.raises(FrozenPostprocessError, match=reason):
        build_frozen_postprocess_contract(
            model_id="yolo11l",
            outputs=outputs,
            input_hw=[640, 640],
            original_wh=[640, 640],
        )


def test_yolo11_endpoint_guard_requires_parity_attested_640_geometry() -> None:
    with pytest.raises(
        FrozenPostprocessError,
        match="yolo11_raw_endpoint_640_input_required",
    ):
        build_frozen_postprocess_contract(
            model_id="yolo11l",
            outputs=_yolo11_outputs(),
            input_hw=[608, 608],
            original_wh=[608, 608],
        )


def test_yolo11_endpoint_guard_rejects_swapped_canonical_branches() -> None:
    outputs = _yolo11_outputs()
    swapped: dict[str, np.ndarray] = {}
    for level in range(3):
        reg_name = f"/model.23/cv2.{level}/cv2.{level}.2/Conv_output_0"
        cls_name = f"/model.23/cv3.{level}/cv3.{level}.2/Conv_output_0"
        swapped[reg_name] = outputs[cls_name]
        swapped[cls_name] = outputs[reg_name]

    with pytest.raises(
        FrozenPostprocessError,
        match="semantic_dfl16_coco_pairing_invalid",
    ):
        build_frozen_postprocess_contract(
            model_id="yolo11l",
            outputs=swapped,
            input_hw=[640, 640],
            original_wh=[640, 640],
        )


def test_yolo11_resealed_decoder_or_family_drift_is_rejected() -> None:
    contract = build_frozen_postprocess_contract(
        model_id="yolo11l",
        outputs=_yolo11_outputs(),
        input_hw=[640, 640],
        original_wh=[640, 640],
    )

    wrong_decoder = copy.deepcopy(contract)
    wrong_decoder["decoder_id"] = (
        "yolo26_regcls_ltrb_classaware_nms_v1"
    )
    with pytest.raises(
        FrozenPostprocessError,
        match="frozen_postprocess_contract_fields_invalid",
    ):
        verify_frozen_postprocess_contract(_reseal_frozen(wrong_decoder))

    wrong_family = copy.deepcopy(contract)
    wrong_family["model_family"] = "yolo26"
    with pytest.raises(
        FrozenPostprocessError,
        match="frozen_postprocess_contract_fields_invalid",
    ):
        verify_frozen_postprocess_contract(_reseal_frozen(wrong_family))


def test_yolo11_completion_reuses_existing_attested_endpoint_contract() -> None:
    outputs = _yolo11_outputs()
    contract = build_detection_completion_execution_contract(
        model_id="yolo11l",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 640],
        source_endpoint_contract=_raw_source(outputs),
    )

    verified = verify_detection_completion_execution_contract(contract)
    assert verified["model_family"] == "yolo11"
    assert verified["source_endpoint"]["endpoint_contract_hash"] == "e" * 64
    assert verified["processor_contract"]["decoder_id"] == (
        "yolo11_regcls_dfl16_classaware_nms_v1"
    )

    drifted = dict(outputs)
    drifted["/model.23/cv2.2/cv2.2.2/Conv_output_0"] = np.zeros(
        (1, 64, 40, 40), dtype=np.float32,
    )
    with pytest.raises(FrozenPostprocessError):
        postprocess.DetectionCompletionRuntime(contract).process(drifted)


def test_existing_yolo26_and_yolov7_decoder_identities_are_unchanged() -> None:
    assert postprocess._model_family("yolo26s") == "yolo26"
    assert postprocess._expected_decoder("yolo26s") == (
        "yolo26_regcls_ltrb_classaware_nms_v1",
        "ultralytics_regcls",
    )
    assert postprocess._model_family("yolov7_paper") == "yolov7"
    assert postprocess._expected_decoder("yolov7_paper") == (
        YOLOV7_PAPER_DECODER_ID,
        "multiscale_head",
    )

    yolo26 = build_frozen_postprocess_contract(
        model_id="yolo26s",
        outputs=_yolo26_outputs(),
        input_hw=[640, 640],
        original_wh=[640, 640],
    )
    assert yolo26["model_family"] == "yolo26"
    assert yolo26["decoder_id"] == (
        "yolo26_regcls_ltrb_classaware_nms_v1"
    )


@pytest.mark.parametrize("family", ["yolo26", "yolov7"])
def test_v2776_native_source_hash_remains_contract_compatible(
    family: str,
) -> None:
    if family == "yolo26":
        contract = build_frozen_postprocess_contract(
            model_id="yolo26s",
            outputs=_yolo26_outputs(),
            input_hw=[640, 640],
            original_wh=[640, 640],
        )
    else:
        contract = build_frozen_postprocess_contract(
            model_id="yolov7_paper",
            outputs=_yolov7_outputs(),
            input_hw=[640, 640],
            original_wh=[640, 640],
            model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        )

    archived = copy.deepcopy(contract)
    if family == "yolov7":
        # v2.77.6 could only emit decoder schema 1. Since v2.82 the default
        # builder emits schema 2 with explicitly changed sigmoid arithmetic;
        # attaching an old implementation hash to that new declaration must
        # remain invalid. Build the historical contract this test represents.
        legacy_decoder = build_yolov7_decoder_contract(
            model_id="yolov7_paper",
            model_sha256=YOLOV7_PAPER_ONNX_SHA256,
            activation_mode=archived["multiscale_activation_mode"],
            schema_version=1,
        )
        archived["model_bound_decoder_contract"] = legacy_decoder
        archived["model_bound_decoder_contract_sha256"] = (
            legacy_decoder["decoder_contract_sha256"]
        )
    archived["implementation_artifacts"][
        "native_detection_postprocess"
    ]["sha256"] = V2776_NATIVE_POSTPROCESS_SHA256
    verified = verify_frozen_postprocess_contract(_reseal_frozen(archived))

    assert verified["model_family"] == family
    if family == "yolov7":
        assert verified["model_bound_decoder_contract"]["schema_version"] == 1
        assert "sigmoid_arithmetic" not in verified["model_bound_decoder_contract"]
    assert verified.get("legacy_decoder_contract_unbound") is not True
    assert V2776_NATIVE_POSTPROCESS_SHA256 in (
        postprocess._IMPLEMENTATION_SHA256_COMPATIBILITY[
            "native_detection_postprocess"
        ]
    )
    assert V2776_NATIVE_POSTPROCESS_SHA256 not in (
        postprocess._LEGACY_UNBOUND_DECODER_NATIVE_SHA256
    )


def test_v2776_native_hash_cannot_claim_yolo11_support() -> None:
    contract = build_frozen_postprocess_contract(
        model_id="yolo11l",
        outputs=_yolo11_outputs(),
        input_hw=[640, 640],
        original_wh=[640, 640],
    )
    archived = copy.deepcopy(contract)
    archived["implementation_artifacts"][
        "native_detection_postprocess"
    ]["sha256"] = V2776_NATIVE_POSTPROCESS_SHA256

    with pytest.raises(
        FrozenPostprocessError,
        match="frozen_postprocess_implementation_sha256_mismatch",
    ):
        verify_frozen_postprocess_contract(_reseal_frozen(archived))


def test_v2776_native_hash_cannot_claim_new_yolov7_sigmoid_arithmetic() -> None:
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=_yolov7_outputs(),
        input_hw=[640, 640],
        original_wh=[640, 640],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    assert contract["model_bound_decoder_contract"]["schema_version"] == 2
    contract["implementation_artifacts"]["native_detection_postprocess"]["sha256"] = (
        V2776_NATIVE_POSTPROCESS_SHA256
    )
    with pytest.raises(
        FrozenPostprocessError,
        match="yolov7_sigmoid_arithmetic_implementation_mismatch",
    ):
        verify_frozen_postprocess_contract(_reseal_frozen(contract))
