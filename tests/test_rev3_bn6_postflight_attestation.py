"""BN6 Full postflight evidence must survive the strict normal attestation."""
from __future__ import annotations

import copy

import pytest

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDecodedNmsPostprocessor,
    FrozenPostprocessError,
    build_normalized_detection_endpoint_attestation,
)
from test_v2711_direct_bn6_producer_binding import _direct_evidence, full_runner


def _postflight_evidence():
    row, outputs = _direct_evidence()
    contract = row["frozen_decoded_nms_normalization_contract"]
    processor = FrozenDecodedNmsPostprocessor(contract)
    for _ in range(3):
        processor.process_task(outputs, original_wh=[1280, 720])
    return row, processor.finalize_evidence()


def _attest(row, result):
    return build_normalized_detection_endpoint_attestation(
        row["frozen_decoded_nms_normalization_contract"],
        result,
        completed_frames=3,
        postprocess_completed_frames=3,
    )


def test_bn6_full_postflight_survives_normal_attestation_and_binding():
    row, result = _postflight_evidence()
    original = copy.deepcopy(result)
    assert result["postflight_oracle"] == {
        "status": "passed",
        "observation_relation": "last_completed_task_after_drain",
        "compared_tasks": 1,
        "per_frame_parity_claimed": False,
    }
    attestation = _attest(row, result)
    assert result == original
    assert attestation["frozen_decoded_nms_normalization_result"] == original
    row["frozen_decoded_nms_normalization_result"] = result
    row["completed_task_endpoint_attestation"] = attestation
    binding = full_runner._verified_direct_completed_task_workload(
        row,
        row["frozen_decoded_nms_normalization_contract"],
        completed_frames=3,
        postprocess_completed_frames=3,
    )
    assert binding["completed_task_endpoint_attestation"] == attestation


@pytest.mark.parametrize("mutation", [
    "not_mapping", "missing_key", "unknown_key", "failed_status",
    "wrong_relation", "multiple_tasks", "boolean_count", "float_count",
    "per_frame_claim", "numeric_false",
])
def test_bn6_postflight_rejects_invalid_oracle(mutation):
    row, result = _postflight_evidence()
    oracle = result["postflight_oracle"]
    if mutation == "not_mapping":
        result["postflight_oracle"] = None
    elif mutation == "missing_key":
        oracle.pop("status")
    elif mutation == "unknown_key":
        oracle["invented_proof"] = True
    elif mutation == "failed_status":
        oracle["status"] = "failed"
    elif mutation == "wrong_relation":
        oracle["observation_relation"] = "every_completed_task"
    elif mutation == "multiple_tasks":
        oracle["compared_tasks"] = 2
    elif mutation == "boolean_count":
        oracle["compared_tasks"] = True
    elif mutation == "float_count":
        oracle["compared_tasks"] = 1.0
    elif mutation == "per_frame_claim":
        oracle["per_frame_parity_claimed"] = True
    elif mutation == "numeric_false":
        oracle["per_frame_parity_claimed"] = 0
    with pytest.raises(
        FrozenPostprocessError,
        match="direct_normalization_postflight_oracle_invalid",
    ):
        _attest(row, result)


def test_bn6_postflight_does_not_admit_unknown_normalization_keys():
    row, result = _postflight_evidence()
    result["invented_normalization_proof"] = True
    with pytest.raises(
        FrozenPostprocessError,
        match="direct_normalization_runtime_attestation_invalid",
    ):
        _attest(row, result)


def test_bn6_postflight_still_rejects_real_oracle_mismatch():
    row, outputs = _direct_evidence()
    processor = FrozenDecodedNmsPostprocessor(
        row["frozen_decoded_nms_normalization_contract"]
    )
    processor.process_task(outputs, original_wh=[1280, 720])
    assert processor.last_detections
    processor.last_detections = []
    with pytest.raises(
        FrozenPostprocessError,
        match="full_completion_postflight_oracle_mismatch",
    ):
        processor.finalize_evidence()
