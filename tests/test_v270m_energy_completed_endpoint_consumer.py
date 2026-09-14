from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_energy_reporting import (
    _resolved_energy_endpoint_identity,
    build_native_energy_pairs,
    collect_native_energy,
)


PHYSICAL_HASH = "a" * 64
OTHER_PHYSICAL_HASH = "b" * 64
COMPARISON_HASH = "c" * 64
PHYSICAL_ID = f"detection:raw_head:{PHYSICAL_HASH}"
OTHER_PHYSICAL_ID = (
    f"detection:decoded_nms:{OTHER_PHYSICAL_HASH}"
)
COMPARISON_ID = (
    f"detection:decoded_nms:comparison:{COMPARISON_HASH}"
)


def _comparison_contract() -> dict[str, object]:
    return {
        "schema": (
            "onnx-splitpoint/completed-task-comparison-endpoint"
        ),
        "stage": "decoded_nms",
        "endpoint_contract_hash": COMPARISON_HASH,
        "output_endpoint_id": COMPARISON_ID,
    }


def _validation(
    *,
    physical_id: str = PHYSICAL_ID,
    physical_hash: str = PHYSICAL_HASH,
    physical_stage: str = "raw_head",
) -> dict[str, object]:
    comparison = _comparison_contract()
    completion = {
        "attested": True,
        "status": "passed",
        "stage": "decoded_nms",
        "endpoint": "decoded_nms",
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash": (
            COMPARISON_HASH
        ),
        "completed_task_comparison_output_endpoint_id": COMPARISON_ID,
    }
    return {
        "backend": "native_full_hailo8",
        "model": "yolov7_paper",
        "case": "full",
        "precision": "uint8_cast_fp16",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "task": "detection",
        "stage": physical_stage,
        "contract_family": physical_stage,
        "endpoint_contract_hash": physical_hash,
        "endpoint_contract_complete": True,
        "output_endpoint_id": physical_id,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation_status": "passed",
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_endpoint_attestation": completion,
        "completed_task_comparison_endpoint_contract": comparison,
        "completed_task_comparison_endpoint_contract_hash": (
            COMPARISON_HASH
        ),
        "completed_task_comparison_output_endpoint_id": COMPARISON_ID,
        "comparison_output_endpoint_id": COMPARISON_ID,
    }


def _plan(
    *,
    physical_id: str = PHYSICAL_ID,
    physical_hash: str = PHYSICAL_HASH,
    physical_stage: str = "raw_head",
) -> dict[str, object]:
    return {
        "backend": "native_full_hailo8",
        "model": "yolov7_paper",
        "case": "full",
        "precision": "uint8_cast_fp16",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "task": "detection",
        "physical_output_endpoint_id": physical_id,
        "physical_endpoint_contract_hash": physical_hash,
        "physical_endpoint_stage": physical_stage,
        "physical_endpoint_contract_complete": True,
        "comparison_output_endpoint_id": COMPARISON_ID,
        "comparison_endpoint_contract_hash": COMPARISON_HASH,
        "comparison_endpoint_stage": "decoded_nms",
        "output_endpoint_id": COMPARISON_ID,
        "endpoint_contract_hash": COMPARISON_HASH,
        "endpoint_stage": "decoded_nms",
        "endpoint_contract_complete": True,
        "output_endpoint_match": True,
        "completion_pairing_eligible": True,
        "completion_pairing_status": (
            "strict_completed_detection_endpoint_verified"
        ),
    }


def test_detection_consumer_uses_canonical_endpoint_and_preserves_physical() -> None:
    resolved = _resolved_energy_endpoint_identity(
        _plan(), _validation(), "detection",
    )

    assert resolved["physical_output_endpoint_id"] == PHYSICAL_ID
    assert (
        resolved["physical_endpoint_contract_hash"] == PHYSICAL_HASH
    )
    assert resolved["physical_endpoint_stage"] == "raw_head"
    assert resolved["physical_output_endpoint_match"] is True
    assert resolved["comparison_output_endpoint_id"] == COMPARISON_ID
    assert (
        resolved["comparison_endpoint_contract_hash"]
        == COMPARISON_HASH
    )
    assert resolved["comparison_endpoint_stage"] == "decoded_nms"
    assert resolved["output_endpoint_id"] == COMPARISON_ID
    assert resolved["endpoint_contract_hash"] == COMPARISON_HASH
    assert resolved["completion_pairing_eligible"] is True
    assert resolved["output_endpoint_match"] is True


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("plan", "completion_pairing_eligible", False),
        ("plan", "completion_pairing_status", "unverified"),
        ("plan", "comparison_endpoint_contract_hash", "d" * 64),
        (
            "plan",
            "comparison_output_endpoint_id",
            "detection:decoded_nms:comparison:" + "d" * 64,
        ),
        (
            "validation",
            "completed_task_endpoint_attestation_status",
            "failed",
        ),
        (
            "validation",
            "completed_task_completion_mode",
            "integrated_accelerator",
        ),
        (
            "nested",
            "completed_task_comparison_endpoint_contract_hash",
            "d" * 64,
        ),
    ],
)
def test_detection_consumer_tamper_is_fail_closed_but_keeps_physical(
    target: str,
    field: str,
    value: object,
) -> None:
    plan = _plan()
    validation = _validation()
    if target == "plan":
        plan[field] = value
    elif target == "validation":
        validation[field] = value
    else:
        completion = copy.deepcopy(
            validation["completed_task_endpoint_attestation"]
        )
        assert isinstance(completion, dict)
        completion[field] = value
        validation["completed_task_endpoint_attestation"] = completion

    resolved = _resolved_energy_endpoint_identity(
        plan, validation, "detection",
    )

    assert resolved["physical_output_endpoint_id"] == PHYSICAL_ID
    assert resolved["physical_endpoint_contract_hash"] == PHYSICAL_HASH
    assert resolved["comparison_output_endpoint_id"] == ""
    assert resolved["comparison_endpoint_contract_hash"] == ""
    assert resolved["output_endpoint_id"] == ""
    assert resolved["endpoint_contract_hash"] == ""
    assert resolved["completion_pairing_eligible"] is False
    assert resolved["output_endpoint_match"] is False


def test_detection_split_without_measured_tail_is_never_relabelled() -> None:
    plan = {
        **_plan(),
        "backend": "hailo8_to_trt",
        "case": "b044",
        "comparison_output_endpoint_id": "",
        "comparison_endpoint_contract_hash": "",
        "comparison_endpoint_stage": "",
        "output_endpoint_id": PHYSICAL_ID,
        "endpoint_contract_hash": PHYSICAL_HASH,
        "endpoint_stage": "raw_head",
        "endpoint_contract_complete": False,
        "output_endpoint_match": False,
        "completion_pairing_eligible": False,
        "completion_pairing_status": (
            "detection_completed_endpoint_attestation_missing"
        ),
    }
    validation = {
        key: value
        for key, value in _validation().items()
        if not key.startswith("completed_task_")
        and key != "comparison_output_endpoint_id"
    }
    validation.update({
        "backend": "hailo8_to_trt",
        "case": "b044",
    })

    resolved = _resolved_energy_endpoint_identity(
        plan, validation, "detection",
    )

    assert resolved["physical_output_endpoint_id"] == PHYSICAL_ID
    assert resolved["physical_output_endpoint_match"] is True
    assert resolved["comparison_output_endpoint_id"] == ""
    assert resolved["output_endpoint_id"] == ""
    assert resolved["completion_pairing_eligible"] is False
    assert (
        resolved["completion_pairing_validation_status"]
        == "plan_completion_not_strictly_eligible"
    )


def test_physical_plan_validation_mismatch_blocks_canonical_pairing() -> None:
    validation = _validation(
        physical_id=(
            "detection:raw_head:" + "d" * 64
        ),
        physical_hash="d" * 64,
    )

    resolved = _resolved_energy_endpoint_identity(
        _plan(), validation, "detection",
    )

    assert resolved["physical_output_endpoint_id"] == PHYSICAL_ID
    assert resolved["physical_output_endpoint_match"] is False
    assert (
        resolved["physical_endpoint_identity_status"]
        == "plan_validation_mismatch"
    )
    assert resolved["comparison_output_endpoint_id"] == ""
    assert resolved["completion_pairing_eligible"] is False
    assert (
        resolved["completion_pairing_validation_status"]
        == "plan_completion_not_strictly_eligible"
    )


def test_collect_native_energy_emits_dual_identity_without_hash_conflict(
    tmp_path: Path,
) -> None:
    plan = _plan()
    validation = _validation()
    validation.update({
        "runtime_precision_identity": "uint8_cast_fp16",
        "execution_precision": "uint8_cast_fp16",
        "full_runtime_precision": "uint8_cast_fp16",
        "claim_ok": False,
        "semantic_ok": False,
        "contract_consistent": True,
    })
    validation_path = (
        tmp_path / "reports" / "native_validation"
        / "native_producer_validation_summary.json"
    )
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        json.dumps({"rows": [validation]}), encoding="utf-8",
    )
    result_path = (
        tmp_path / "reports" / "native_energy_measurements"
        / "native_producer_energy_results.json"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({
            "rows": [{
                "row": plan,
                "ok": False,
                "run": {"rc": 1, "error": "diagnostic"},
            }],
        }),
        encoding="utf-8",
    )

    row = collect_native_energy(tmp_path)[0]

    assert row["native_validation_join_status"] == "exact_unique"
    assert row["native_identity_evidence_conflicts"] == []
    assert row["physical_output_endpoint_id"] == PHYSICAL_ID
    assert row["physical_endpoint_contract_hash"] == PHYSICAL_HASH
    assert row["comparison_output_endpoint_id"] == COMPARISON_ID
    assert row["comparison_endpoint_contract_hash"] == COMPARISON_HASH
    assert row["output_endpoint_id"] == COMPARISON_ID
    assert row["endpoint_contract_hash"] == COMPARISON_HASH
    assert row["completion_pairing_eligible"] is True


def _pair_row(
    *,
    backend: str,
    mode: str,
    physical_id: str,
    physical_hash: str,
    physical_stage: str,
    energy: float,
) -> dict[str, object]:
    return {
        "backend": backend,
        "model": "yolov7_paper",
        "case": "full" if mode == "native_full_baseline" else "b044",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "execution_mode": mode,
        "task": "detection",
        "task_source": "declared",
        "evaluation_role": "development",
        "direction": "hailo8_to_trt",
        "precision": "uint8_cast_fp16",
        "split_boundary_precision": (
            "uint8_cast_fp16" if mode == "native_split" else ""
        ),
        "full_runtime_precision": (
            "uint8_cast_fp16"
            if mode == "native_full_baseline" else ""
        ),
        "pipeline_contract_sha256": "1" * 64,
        "pipeline_preprocessing_sha256": "2" * 64,
        "pipeline_decoder_sha256": "3" * 64,
        "pipeline_nms_sha256": "4" * 64,
        "quality_contract_sha256": "5" * 64,
        "preprocessing_contract_sha256": "6" * 64,
        "decoder_contract_sha256": "7" * 64,
        "nms_contract_sha256": "8" * 64,
        "source_request_sha256": "9" * 64,
        "model_sha256": "a" * 64,
        "validation_dataset_sha256": "b" * 64,
        "validation_dataset_image_ids_sha256": "c" * 64,
        "validation_dataset_ground_truth_sha256": "d" * 64,
        "accuracy_gate_policy_sha256": "e" * 64,
        "task_quality_policy_sha256": "e" * 64,
        "runtime_quality_gate_policy_sha256": "e" * 64,
        "validation_input_or_image_sha256": "f" * 64,
        "prepared_feed_task": "detection",
        "prepared_feed_preprocess_mode": "letterbox",
        "prepared_feed_letterbox_pad_value": "114",
        "prepared_feed_source_image_sha256": "f" * 64,
        "energy_scope": "MB",
        "energy_window_effective": "command_window",
        "duration_s": 60.0,
        "target_duration_s": 60.0,
        "active_duration_s": 60.0,
        "claim_eligible": True,
        "semantic_claim_ok": True,
        "contract_consistent": True,
        "runtime_work_units_exact": True,
        "energy_per_work_j": energy,
        "energy_primary_metric": "raw_input_energy",
        "energy_raw_primary": True,
        "ok": True,
        "physical_output_endpoint_id": physical_id,
        "physical_endpoint_contract_hash": physical_hash,
        "physical_endpoint_stage": physical_stage,
        "physical_endpoint_contract_complete": True,
        "physical_output_endpoint_match": True,
        "comparison_output_endpoint_id": COMPARISON_ID,
        "comparison_endpoint_contract_hash": COMPARISON_HASH,
        "comparison_endpoint_stage": "decoded_nms",
        "completion_pairing_eligible": True,
        "output_endpoint_id": COMPARISON_ID,
        "endpoint_contract_hash": COMPARISON_HASH,
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "endpoint_contract_complete": True,
        "output_endpoint_match": True,
    }


def test_pairing_compares_canonical_and_reports_distinct_physical_ids() -> None:
    split = _pair_row(
        backend="hailo8_to_trt",
        mode="native_split",
        physical_id=PHYSICAL_ID,
        physical_hash=PHYSICAL_HASH,
        physical_stage="raw_head",
        energy=0.10,
    )
    full = _pair_row(
        backend="native_full_hailo8",
        mode="native_full_baseline",
        physical_id=OTHER_PHYSICAL_ID,
        physical_hash=OTHER_PHYSICAL_HASH,
        physical_stage="decoded_nms",
        energy=0.12,
    )

    pair = build_native_energy_pairs([split, full])[0]

    assert pair["comparable"] is True
    assert pair["split_output_endpoint_id"] == COMPARISON_ID
    assert pair["baseline_output_endpoint_id"] == COMPARISON_ID
    assert (
        pair["split_comparison_output_endpoint_id"] == COMPARISON_ID
    )
    assert (
        pair["baseline_comparison_output_endpoint_id"]
        == COMPARISON_ID
    )
    assert pair["split_physical_output_endpoint_id"] == PHYSICAL_ID
    assert (
        pair["baseline_physical_output_endpoint_id"]
        == OTHER_PHYSICAL_ID
    )

    tampered = dict(
        full,
        comparison_output_endpoint_id=(
            "detection:decoded_nms:comparison:" + "d" * 64
        ),
        comparison_endpoint_contract_hash="d" * 64,
        output_endpoint_id=(
            "detection:decoded_nms:comparison:" + "d" * 64
        ),
        endpoint_contract_hash="d" * 64,
    )
    rejected = build_native_energy_pairs([split, tampered])[0]
    assert rejected["comparable"] is False
    assert (
        "output_endpoint_mismatch_or_missing"
        in rejected["comparison_reasons"]
    )
    assert (
        "endpoint_contract_hash_mismatch_or_missing"
        in rejected["comparison_reasons"]
    )
