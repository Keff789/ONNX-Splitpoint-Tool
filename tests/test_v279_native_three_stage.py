from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx_splitpoint_tool.native_three_stage import (
    ADAPTER_CLASSIFICATION_LOGITS,
    ADAPTER_YOLO11_DFL16,
    ADAPTER_YOLO26_DECODED,
    ADAPTER_YOLOV7_SPARSE,
    NativeThreeStageError,
    adapter_id_for_contract_family,
    infer_p2_output_contract_family,
    project_three_stage_endpoints,
    summarize_ms,
)

def test_stage_summary_is_deterministic() -> None:
    value = summarize_ms([3.0, 1.0, 2.0, float("nan"), -1.0])
    assert value["count"] == 3
    assert value["mean_ms"] == 2.0
    assert value["p50_ms"] == 2.0
    assert value["p95_ms"] == pytest.approx(2.9)
    assert value["min_ms"] == 1.0
    assert value["max_ms"] == 3.0

@pytest.mark.parametrize(("family", "adapter"), [
    ("classification_logits", ADAPTER_CLASSIFICATION_LOGITS),
    ("yolov7_anchor_multiscale_raw", ADAPTER_YOLOV7_SPARSE),
    ("yolo26_decoded_nms", ADAPTER_YOLO26_DECODED),
    ("yolo11_regcls_dfl16_raw", ADAPTER_YOLO11_DFL16),
])
def test_adapter_registry(family: str, adapter: str) -> None:
    assert adapter_id_for_contract_family(family) == adapter

def test_contract_family_resolution_is_attestation_first_and_fail_closed() -> None:
    assert infer_p2_output_contract_family({"task": "classification"}) == "classification_logits"
    assert infer_p2_output_contract_family({"stage": "decoded_nms"}, model_id="yolo26s", task="detection") == "yolo26_decoded_nms"
    assert infer_p2_output_contract_family({"stage": "raw_head"}, model_id="yolov7_paper", task="detection") == "yolov7_anchor_multiscale_raw"
    assert infer_p2_output_contract_family({"stage": "raw_head"}, model_id="yolo11l", task="detection") == "yolo11_regcls_dfl16_raw"
    with pytest.raises(NativeThreeStageError, match="unresolved"):
        infer_p2_output_contract_family({"stage": "raw_head"}, model_id="unknown", task="detection")

def test_projection_exposes_two_endpoints_without_erasing_legacy_fields() -> None:
    source = {"model_id": "yolov7_paper", "fps_makespan": 15.0}
    result = project_three_stage_endpoints(
        source,
        p2_output_fps=95.9,
        completed_detection_fps=95.88,
        p2_output_contract_family="yolov7_anchor_multiscale_raw",
        postprocess_adapter_id=ADAPTER_YOLOV7_SPARSE,
        postprocess_location="host_cpu_numpy",
        p1_samples_ms=[10.4, 10.5],
        p2_samples_ms=[8.7, 8.8],
        post_samples_ms=[1.3, 2.0],
    )
    assert source == {"model_id": "yolov7_paper", "fps_makespan": 15.0}
    assert result["performance_endpoint"] == "p2_output"
    assert result["application_performance_endpoint"] == "completed_detection"
    assert result["throughput_primary_fps"] == 95.9
    assert result["application_throughput_fps"] == 95.88
    assert result["completed_to_p2_ratio"] == pytest.approx(95.88 / 95.9)
    assert result["stage_timings"]["postprocess"]["p95_ms"] == pytest.approx(1.965)
    assert result["legacy_performance_endpoint"] == "raw_model_outputs"
    assert result["legacy_application_performance_endpoint"] == "completed_task"

def test_historical_projection_needs_source_and_never_fabricates_completed_endpoint() -> None:
    with pytest.raises(NativeThreeStageError, match="projection_source_required"):
        project_three_stage_endpoints(
            {}, p2_output_fps=10.0, completed_detection_fps=None,
            p2_output_contract_family="classification_logits",
            postprocess_adapter_id=ADAPTER_CLASSIFICATION_LOGITS,
            postprocess_location="not_applicable", directly_measured=False,
        )
    result = project_three_stage_endpoints(
        {}, p2_output_fps=10.0, completed_detection_fps=None,
        p2_output_contract_family="classification_logits",
        postprocess_adapter_id=ADAPTER_CLASSIFICATION_LOGITS,
        postprocess_location="not_applicable", directly_measured=False,
        projection_source="legacy_single_endpoint",
    )
    assert result["completed_detection_fps"] is None
    assert result["application_performance_endpoint"] == ""


def _decoded_completion_contract():
    from onnx_splitpoint_tool.native_detection_postprocess import (
        build_detection_completion_runtime,
        tensor_signature,
    )

    outputs = {
        "detections": np.asarray(
            [[
                [8.0, 16.0, 32.0, 48.0, 0.90, 2.0],
                [8.0, 16.0, 32.0, 48.0, 0.80, 2.0],
                [0.0, 0.0, 4.0, 4.0, 0.10, 1.0],
            ]],
            dtype=np.float32,
        ),
    }
    signature = tensor_signature(outputs)
    endpoint_hash = "d" * 64
    source = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "tensor_signature": signature,
        "output_endpoint_attestation": {
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
        },
    }
    oracle = build_detection_completion_runtime(
        model_id="yolo26s",
        outputs=outputs,
        input_hw=[64, 64],
        original_wh=[80, 60],
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114},
        source_endpoint_contract=source,
    )
    return outputs, oracle.execution_contract


def test_fast_completion_keeps_crypto_out_of_hotloop_and_attests_postflight() -> None:
    from onnx_splitpoint_tool.native_three_stage import (
        FastDetectionCompletionRuntime,
    )

    outputs, contract = _decoded_completion_contract()
    runtime = FastDetectionCompletionRuntime(contract)
    fast_result = runtime.process(outputs)

    assert fast_result["evidence_mode"] == (
        "task_only_no_crypto_in_timed_hotloop"
    )
    assert fast_result["detection_count"] == 2
    assert "artifact_sha256" not in fast_result

    attestation = runtime.attestation(completed_work_units=1)
    assert attestation["attested"] is True
    assert attestation["status"] == "passed"
    assert attestation["observation_relation"] == (
        "postflight_oracle_sentinel"
    )
    assert attestation["quality_oracle_location"] == (
        "outside_performance_timing"
    )
    assert attestation["fast_content_sha256"] == (
        attestation["oracle_content_sha256"]
    )
    assert len(attestation["artifact_sha256"]) == 64


def test_fast_completion_attestation_fails_closed_on_count_mismatch() -> None:
    from onnx_splitpoint_tool.native_three_stage import (
        FastDetectionCompletionRuntime,
    )

    outputs, contract = _decoded_completion_contract()
    runtime = FastDetectionCompletionRuntime(contract)
    runtime.process(outputs)
    with pytest.raises(NativeThreeStageError, match="count_mismatch"):
        runtime.attestation(completed_work_units=2)
