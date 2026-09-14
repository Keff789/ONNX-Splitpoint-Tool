from __future__ import annotations

import copy

import numpy as np

from onnx_splitpoint_tool.benchmark.accuracy_gate import (
    infer_contract_consistent,
)
from onnx_splitpoint_tool.benchmark.accuracy_gates import (
    _native_contract_consistent,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_comparison_endpoint_contract,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (
    _detection_postprocess_contract,
)
from onnx_splitpoint_tool.validation.gates import _contract_gate
from onnx_splitpoint_tool.validation.host_postprocess import (
    apply_host_postprocess_aliases,
    resolve_host_postprocess_evidence,
)
from scripts.native_producer_final_report import (
    _apply_comparison_claim_gates,
)


def _outputs(prefix: str = "") -> dict[str, np.ndarray]:
    names = (
        ("output", "clone_1", "clone_2")
        if not prefix
        else (f"{prefix}_small", f"{prefix}_medium", f"{prefix}_large")
    )
    return {
        names[0]: np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
        names[1]: np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
        names[2]: np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
    }


def _frozen(
    outputs: dict[str, np.ndarray],
    *,
    input_hw: tuple[int, int] = (640, 640),
) -> tuple[dict, dict]:
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=list(input_hw),
        original_wh=[80, 60],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    result = FrozenDetectionPostprocessor(contract).process(
        outputs,
        original_wh=[80, 60],
    )
    return contract, result


def _canonical_row(
    *,
    backend: str = "native_full_hailo8",
    output_prefix: str = "",
    source_endpoint_hash: str = "a" * 64,
) -> dict:
    contract, result = _frozen(_outputs(output_prefix))
    attestation = build_completed_detection_endpoint_attestation(
        contract,
        result,
        completed_frames=11,
        postprocess_completed_frames=11,
        source_endpoint_contract_hash=source_endpoint_hash,
    )
    return {
        "backend": backend,
        "model": "yolov7_paper",
        "case": "full",
        "execution_mode": "native_full_baseline",
        "setup_id": "setup-a",
        "comparison_backend": "ort_tensorrt",
        "execution_precision": "fp16",
        "ok": True,
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": source_endpoint_hash,
        "output_endpoint_id": (
            f"detection:raw_head:{source_endpoint_hash}"
        ),
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "task": "detection",
            "stage": "raw_head",
            "endpoint": "raw_head",
            "endpoint_contract_hash": source_endpoint_hash,
            "output_endpoint_id": (
                f"detection:raw_head:{source_endpoint_hash}"
            ),
        },
        "raw_head_contract_present": True,
        "requires_host_decode_nms": True,
        "semantic_reference_source": "full_onnx_self_reference",
        "self_reference_ok": True,
        "self_reference_diagnosis": (
            "native_semantic_matches_full_self_reference"
        ),
        "semantic_ok": True,
        "ap50_proxy": 1.0,
        "tensor_ok": True,
        "host_postprocess_frozen": True,
        "postprocess_included": True,
        "postprocess_completed_frames": 11,
        "postprocess_completion_verified": True,
        "frozen_host_postprocess_contract": contract,
        "frozen_host_postprocess_contract_sha256": contract[
            "contract_sha256"
        ],
        "frozen_host_postprocess_result": result,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_contract": attestation[
            "completed_endpoint_contract"
        ],
        "completed_task_endpoint_contract_hash": attestation[
            "endpoint_contract_hash"
        ],
        "completed_task_output_endpoint_id": attestation[
            "output_endpoint_id"
        ],
        "completed_task_comparison_endpoint_contract": attestation[
            "completed_task_comparison_endpoint_contract"
        ],
        "completed_task_comparison_endpoint_contract_hash": attestation[
            "completed_task_comparison_endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": attestation[
            "completed_task_comparison_output_endpoint_id"
        ],
        "completed_task_completion_mode": attestation[
            "completed_task_completion_mode"
        ],
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_endpoint_attestation_status": "passed",
    }


def test_canonical_host_tail_without_legacy_field_passes_every_gate() -> None:
    row = _canonical_row()
    assert "host_tail_available" not in row

    resolved = resolve_host_postprocess_evidence(row)
    assert resolved["available"] is True
    assert resolved["status"] == "passed"
    assert resolved["decoder_id"]
    assert _detection_postprocess_contract(row)[0] is True
    assert _contract_gate(row, "detection", {})["contract_consistent"] is True
    assert infer_contract_consistent(row) == "pass"
    assert _native_contract_consistent(row, {}, "detection")[0] is True

    normalized = apply_host_postprocess_aliases(row)
    assert normalized["host_postprocessing_available"] is True
    assert normalized["host_tail_available"] is True
    assert normalized["decoder_contract_pass"] is True
    assert normalized["nms_ok"] is True
    assert normalized["decoder_id"]


def test_host_tail_tamper_and_legacy_conflict_fail_closed() -> None:
    tampered = _canonical_row()
    tampered["completed_task_endpoint_attestation"] = copy.deepcopy(
        tampered["completed_task_endpoint_attestation"]
    )
    tampered["completed_task_endpoint_attestation"][
        "postprocess_completed_frames"
    ] = 10
    tampered_result = resolve_host_postprocess_evidence(tampered)
    assert tampered_result["available"] is False
    assert tampered_result["status"] == (
        "failed_incomplete_canonical_host_postprocess_evidence"
    )

    conflict = _canonical_row()
    conflict["host_tail_available"] = False
    resolved = resolve_host_postprocess_evidence(conflict)
    assert resolved["available"] is False
    assert resolved["status"] == "host_postprocess_evidence_conflict"


def test_classification_host_postprocess_is_not_required() -> None:
    row = {
        "task": "classification",
        "stage": "classification_logits",
        "contract_family": "classification_logits",
        "host_postprocess_frozen": False,
        "postprocess_included": False,
        "postprocess_completion_verified": True,
        "completed_task_endpoint_attested": False,
        "completed_task_endpoint_attestation_status": "not_required",
        # These legacy False aliases previously turned N/A into a failure.
        "host_postprocessing_available": False,
        "host_tail_available": False,
    }

    resolved = resolve_host_postprocess_evidence(row)
    assert resolved["available"] is None
    assert resolved["status"] == "not_required"
    assert resolved["legacy_alias_conflict"] is False

    normalized = apply_host_postprocess_aliases(row)
    assert normalized["host_postprocess_required"] is False
    assert normalized["host_tail_required"] is False
    assert normalized["host_postprocessing_available"] is False
    assert normalized["host_tail_available"] is False
    assert normalized["host_postprocessing_evidence_status"] == "not_required"


def test_accelerator_integrated_decoded_nms_host_tail_is_not_required() -> None:
    row = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "accelerator_output_stage": "decoded_nms",
        "accelerator_output_contract_family": "decoded_nms",
        "requires_host_decode_nms": False,
        "host_postprocess_required": False,
        "host_tail_required": False,
        "host_postprocess_frozen": False,
        "postprocess_included": True,
        "postprocess_location": "accelerator_runtime",
        "postprocess_completion_verified": True,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "host_postprocessing_available": False,
        "host_tail_available": False,
    }

    resolved = resolve_host_postprocess_evidence(row)
    assert resolved["available"] is None
    assert resolved["status"] == "not_required"
    assert resolved["legacy_alias_conflict"] is False

    normalized = apply_host_postprocess_aliases(row)
    assert normalized["host_postprocess_required"] is False
    assert normalized["host_tail_required"] is False
    assert normalized["host_postprocessing_available"] is False
    assert normalized["host_tail_available"] is False
    assert normalized["host_postprocessing_evidence_status"] == "not_required"


def test_raw_head_required_missing_host_tail_remains_failed() -> None:
    row = {
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "requires_host_decode_nms": True,
        "host_postprocess_required": True,
        "host_tail_required": True,
        "host_postprocess_frozen": False,
        "postprocess_included": False,
        "postprocess_completion_verified": False,
        "completed_task_endpoint_attested": False,
    }

    resolved = resolve_host_postprocess_evidence(row)
    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_incomplete_canonical_host_postprocess_evidence"
    )
    assert resolved["legacy_alias_conflict"] is False


def test_comparison_endpoint_ignores_physical_raw_signature() -> None:
    first, _ = _frozen(_outputs())
    second, _ = _frozen(_outputs("vendor"))

    assert first["raw_output_tensor_signature"] != second[
        "raw_output_tensor_signature"
    ]
    first_endpoint = build_completed_detection_comparison_endpoint_contract(
        first
    )
    second_endpoint = build_completed_detection_comparison_endpoint_contract(
        second
    )
    assert first_endpoint == second_endpoint
    assert first_endpoint["output_endpoint_id"].startswith(
        "detection:decoded_nms:comparison:"
    )


def test_comparison_endpoint_changes_with_task_semantics() -> None:
    outputs = {}
    conv = 61
    for size in (8, 4, 2):
        outputs[f"yolo26s_full/conv{conv}"] = np.zeros(
            (size, size, 4), dtype=np.float32,
        )
        outputs[f"yolo26s_full/conv{conv + 3}"] = np.full(
            (size, size, 80), -20.0, dtype=np.float32,
        )
        conv += 16
    first = build_frozen_postprocess_contract(
        model_id="yolo26s", outputs=outputs, input_hw=[64, 64],
        original_wh=[80, 60],
    )
    second = build_frozen_postprocess_contract(
        model_id="yolo26s", outputs=outputs, input_hw=[96, 96],
        original_wh=[80, 60],
    )

    assert build_completed_detection_comparison_endpoint_contract(
        first
    )["output_endpoint_id"] != build_completed_detection_comparison_endpoint_contract(
        second
    )["output_endpoint_id"]


def test_final_report_compares_completed_semantics_not_physical_raw_hash() -> None:
    rows = [
        _canonical_row(
            backend="native_full_hailo8",
            source_endpoint_hash="a" * 64,
        ),
        _canonical_row(
            backend="native_full_tensorrt",
            output_prefix="trt",
            source_endpoint_hash="b" * 64,
        ),
    ]
    gated = _apply_comparison_claim_gates(rows)

    assert len({
        row["completed_task_output_endpoint_id"] for row in gated
    }) == 2
    assert len({
        row["comparison_output_endpoint_id"] for row in gated
    }) == 1
    assert all(row["comparison_endpoint_match"] is True for row in gated)
    assert {
        row["physical_output_endpoint_id"] for row in gated
    } == {
        f"detection:raw_head:{'a' * 64}",
        f"detection:raw_head:{'b' * 64}",
    }
