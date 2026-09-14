from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as postprocess
from onnx_splitpoint_tool.runners.harness import yolo as yolo_harness
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenPostprocessError,
    build_detection_completion_execution_attestation,
    build_detection_completion_runtime,
    canonical_json_bytes,
    canonical_json_sha256,
    tensor_signature,
    verify_detection_completion_execution_attestation,
    verify_detection_completion_execution_contract,
)


def _decoded_outputs() -> dict[str, np.ndarray]:
    return {
        "detections": np.asarray(
            [[
                [8.0, 16.0, 32.0, 48.0, 0.90, 2.0],
                [8.0, 16.0, 32.0, 48.0, 0.80, 2.0],
                [0.0, 0.0, 4.0, 4.0, 0.10, 1.0],
            ]],
            dtype=np.float32,
        ),
    }


def _decoded_source(
    outputs: dict[str, np.ndarray],
    *,
    endpoint_hash: str = "d" * 64,
) -> dict:
    signature = tensor_signature(outputs)
    attestation = {
        "schema": "onnx-splitpoint/runtime-output-endpoint-attestation",
        "schema_version": 3,
        "attested": True,
        "status": "passed",
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "declared_contract": {
            "model_id": "yolo26s",
            "source_coordinate_space": (
                "model_input_letterbox_xyxy_pixels"
            ),
        },
    }
    return {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "output_endpoint_attestation": attestation,
    }


def _runtime(outputs: dict[str, np.ndarray]):
    return build_detection_completion_runtime(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[64, 64],
        original_wh=[80, 60],
        preprocess={
            "mode": "letterbox",
            "rgb": True,
            "pad_value": 114,
        },
        source_endpoint_contract=_decoded_source(outputs),
    )


def test_attested_decoded_nms_is_materialized_without_second_nms() -> None:
    outputs = _decoded_outputs()
    runtime = _runtime(outputs)

    result = runtime.process(outputs)

    # The first two rows overlap exactly and share a class. A second host NMS
    # would collapse them to one row; attested normalization must retain both.
    assert result["detection_count"] == 2
    assert [row["score"] for row in result["detections"]] == pytest.approx(
        [0.9, 0.8]
    )
    assert [
        result["detections"][0][key]
        for key in ("x1", "y1", "x2", "y2")
    ] == pytest.approx([10.0, 10.0, 40.0, 50.0])
    assert result["source_nms_attested"] is True
    assert result["host_nms_applied"] is False
    assert runtime.execution_contract["processor_contract"][
        "host_nms_applied"
    ] is False


def test_completion_counter_advances_only_after_full_success() -> None:
    outputs = _decoded_outputs()
    runtime = _runtime(outputs)
    runtime.process(outputs)
    assert runtime.completed_count == 1

    invalid = {
        "detections": outputs["detections"].copy(),
    }
    invalid["detections"][0, 0, 0] = np.nan
    with pytest.raises(
        FrozenPostprocessError,
        match="nonfinite",
    ):
        runtime.process(invalid)

    assert runtime.completed_count == 1
    assert runtime.last_result["invocation_index"] == 1


def test_completion_attestation_exposes_five_independent_hash_layers() -> None:
    outputs = _decoded_outputs()
    runtime = _runtime(outputs)
    result = runtime.process(outputs)
    attestation = runtime.attestation()

    hashes = [
        attestation["artifact_sha256"],
        attestation["schema_sha256"],
        attestation["content_sha256"],
        attestation["invocation_sha256"],
        attestation["relation_sha256"],
    ]
    assert all(len(value) == 64 for value in hashes)
    assert len(set(hashes)) == 5
    assert attestation["completion_count"] == 1
    assert attestation["completed_work_units"] == 1
    assert attestation["completion_count_verified"] is True
    assert attestation["artifact"] == result["artifact"]
    assert attestation["artifact_sha256"] == hashlib.sha256(
        canonical_json_bytes(result["artifact"])
    ).hexdigest()
    assert (
        attestation["implementation_sha256"]
        == runtime.execution_contract["implementation_sha256"]
    )
    assert attestation["content_sha256"] == result["content_sha256"]
    assert attestation["comparison_endpoint_contract"][
        "output_endpoint_id"
    ].startswith("detection:decoded_nms:comparison:")


def test_execution_and_result_hash_tampering_fail_closed() -> None:
    outputs = _decoded_outputs()
    runtime = _runtime(outputs)
    result = runtime.process(outputs)

    tampered_contract = copy.deepcopy(runtime.execution_contract)
    tampered_contract.pop("contract_sha256")
    tampered_contract["relation_sha256"] = "f" * 64
    tampered_contract["contract_sha256"] = canonical_json_sha256(
        tampered_contract
    )
    with pytest.raises(
        FrozenPostprocessError,
        match="hash_layers_mismatch",
    ):
        verify_detection_completion_execution_contract(tampered_contract)

    tampered_result = copy.deepcopy(result)
    tampered_result["detections"][0]["score"] = 0.7
    with pytest.raises(
        FrozenPostprocessError,
        match="attestation_result_invalid",
    ):
        build_detection_completion_execution_attestation(
            runtime.execution_contract,
            tampered_result,
            completed_work_units=1,
            completion_count=1,
            invocation_chain_sha256=runtime.invocation_chain_sha256,
        )

    tampered_transport = copy.deepcopy(result)
    tampered_transport["artifact"]["detections"][0]["score"] = 0.7
    tampered_transport["artifact_sha256"] = canonical_json_sha256(
        tampered_transport["artifact"]
    )
    with pytest.raises(
        FrozenPostprocessError,
        match="attestation_result_invalid",
    ):
        build_detection_completion_execution_attestation(
            runtime.execution_contract,
            tampered_transport,
            completed_work_units=1,
            completion_count=1,
            invocation_chain_sha256=runtime.invocation_chain_sha256,
        )


def test_legacy_static_artifact_alias_cannot_be_promoted_to_v2() -> None:
    outputs = _decoded_outputs()
    runtime = _runtime(outputs)
    legacy = copy.deepcopy(runtime.execution_contract)
    legacy.pop("contract_sha256")
    legacy["schema_version"] = 1
    legacy["artifact_sha256"] = legacy.pop("implementation_sha256")
    legacy.pop("artifact_hash_policy")
    legacy["hash_layers"]["artifact"] = legacy["hash_layers"].pop(
        "implementation"
    )
    legacy["contract_sha256"] = canonical_json_sha256(legacy)

    with pytest.raises(
        FrozenPostprocessError,
        match="contract_fields_invalid",
    ):
        verify_detection_completion_execution_contract(legacy)


def test_count_mismatch_never_attests() -> None:
    outputs = _decoded_outputs()
    runtime = _runtime(outputs)
    runtime.process(outputs)

    with pytest.raises(
        FrozenPostprocessError,
        match="attestation_count_invalid",
    ):
        runtime.attestation(completed_work_units=2)


def test_observation_relation_distinguishes_hotloop_from_replay() -> None:
    outputs = _decoded_outputs()
    hotloop = _runtime(outputs)
    hotloop.process(outputs)
    hotloop_attestation = hotloop.attestation()
    verified_hotloop = verify_detection_completion_execution_attestation(
        hotloop_attestation,
        execution_contract=hotloop.execution_contract,
        expected_observation_relation="same_hotloop_sentinel",
        reference_content_sha256=hotloop_attestation["content_sha256"],
    )
    assert verified_hotloop["same_hotloop_sentinel"] is True
    assert verified_hotloop["exact_result_claim_bound"] is True
    assert verified_hotloop["reference_content_match"] is True

    replay_outputs = _decoded_outputs()
    replay_outputs["detections"][0, 0, 4] = np.float32(0.85)
    replay = type(hotloop)(
        hotloop.execution_contract,
        observation_relation="independent_replay",
    )
    replay.process(replay_outputs)
    replay_attestation = replay.attestation()
    verified_replay = verify_detection_completion_execution_attestation(
        replay_attestation,
        execution_contract=hotloop.execution_contract,
        expected_observation_relation="independent_replay",
        reference_content_sha256=hotloop_attestation["content_sha256"],
    )
    assert verified_replay["reference_content_match"] is False
    assert verified_replay["exact_result_claim_bound"] is False
    assert (
        verified_replay["comparison_endpoint_contract"]
        == verified_hotloop["comparison_endpoint_contract"]
    )
    assert verified_replay["artifact_sha256"] != verified_hotloop[
        "artifact_sha256"
    ]
    assert verified_replay["implementation_sha256"] == verified_hotloop[
        "implementation_sha256"
    ]
    assert verified_replay["schema_sha256"] == verified_hotloop[
        "schema_sha256"
    ]

    with pytest.raises(
        FrozenPostprocessError,
        match="same_hotloop_content_mismatch",
    ):
        verify_detection_completion_execution_attestation(
            hotloop_attestation,
            execution_contract=hotloop.execution_contract,
            reference_content_sha256=replay_attestation["content_sha256"],
        )


def test_yolov7_completion_uses_one_contract_bound_activation_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = {
        "stride32": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
        "stride8": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
        "stride16": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
    }
    signature = tensor_signature(outputs)
    endpoint_hash = "e" * 64
    source = {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "model_sha256": YOLOV7_PAPER_ONNX_SHA256,
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
    contract = postprocess.build_detection_completion_execution_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 640],
        source_endpoint_contract=source,
    )
    processor = contract["processor_contract"]
    bound_mode = processor["multiscale_activation_mode"]
    assert bound_mode in {
        "logits", "activated", "objcls_activated",
    }

    def forbidden_runtime_inference(_outputs):
        raise AssertionError(
            "runtime activation inference must not run"
        )

    decode_calls: list[str] = []
    original_decode = yolo_harness._decode_multiscale_head_once

    def counted_decode(*args, **kwargs):
        decode_calls.append(str(kwargs["activation_mode"]))
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(
        yolo_harness,
        "_infer_multiscale_head_activation_mode",
        forbidden_runtime_inference,
    )
    monkeypatch.setattr(
        yolo_harness,
        "_decode_multiscale_head_once",
        counted_decode,
    )
    runtime = postprocess.DetectionCompletionRuntime(contract)
    result = runtime.process(outputs)

    assert result["detection_count"] == 0
    assert runtime.completed_count == 1
    assert decode_calls == [bound_mode]


def test_yolov7_completion_rejects_missing_activation_strategy() -> None:
    outputs = {
        "p3": np.zeros((1, 3, 80, 80, 85), dtype=np.float32),
        "p4": np.zeros((1, 3, 40, 40, 85), dtype=np.float32),
        "p5": np.zeros((1, 3, 20, 20, 85), dtype=np.float32),
    }
    contract = postprocess.build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 640],
    )
    contract.pop("contract_sha256")
    contract["multiscale_activation_mode"] = None
    contract["invariant_identity"][
        "multiscale_activation_mode"
    ] = None
    contract["invariant_contract_sha256"] = canonical_json_sha256(
        contract["invariant_identity"]
    )
    contract["contract_sha256"] = canonical_json_sha256(contract)

    with pytest.raises(
        FrozenPostprocessError,
        match="contract_fields_invalid",
    ):
        postprocess.verify_frozen_postprocess_contract(contract)


def test_yolo26_invariant_does_not_bind_yolov7_activation_field() -> None:
    outputs: dict[str, np.ndarray] = {}
    conv = 61
    for side in (80, 40, 20):
        outputs[f"yolo26s_full/conv{conv}"] = np.zeros(
            (side, side, 4), dtype=np.float32,
        )
        outputs[f"yolo26s_full/conv{conv + 3}"] = np.full(
            (side, side, 80), -20.0, dtype=np.float32,
        )
        conv += 16
    contract = postprocess.build_frozen_postprocess_contract(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[640, 483],
    )

    assert "multiscale_activation_mode" not in contract[
        "invariant_identity"
    ]

    archived = copy.deepcopy(contract)
    archived.pop("contract_sha256")
    archived.pop("multiscale_activation_mode")
    archived["invariant_contract_sha256"] = canonical_json_sha256(
        archived["invariant_identity"]
    )
    archived["contract_sha256"] = canonical_json_sha256(archived)

    verified = postprocess.verify_frozen_postprocess_contract(
        archived,
        outputs=outputs,
    )
    assert verified["contract_sha256"] == archived["contract_sha256"]
    assert "legacy_activation_strategy_unbound" not in verified
