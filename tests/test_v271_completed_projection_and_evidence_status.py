from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDetectionPostprocessor,
    build_completed_detection_endpoint_attestation,
    build_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_ONNX_SHA256,
)
from onnx_splitpoint_tool.workflow.evidence_status import (
    blocking_status,
    derive_native_evidence_status,
)
from scripts import native_full_baseline_eval_runner as full_runner
from scripts import native_producer_final_report as final_report
from scripts import native_producer_validate_visualize as validator


def _completed_host_tail_row() -> dict:
    outputs = {
        "output": np.full((1, 3, 80, 80, 85), -20.0, dtype=np.float32),
        "clone_1": np.full((1, 3, 40, 40, 85), -20.0, dtype=np.float32),
        "clone_2": np.full((1, 3, 20, 20, 85), -20.0, dtype=np.float32),
    }
    contract = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[1280, 720],
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
    )
    result = FrozenDetectionPostprocessor(contract).process(
        outputs, original_wh=[1280, 720],
    )
    attestation = build_completed_detection_endpoint_attestation(
        contract,
        result,
        completed_frames=5,
        postprocess_completed_frames=5,
        source_endpoint_contract_hash="a" * 64,
    )
    return {
        "backend": "native_full_tensorrt",
        "model": "yolov7_paper",
        "case": "full",
        "precision": "fp16",
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "a" * 64,
        "host_postprocess_frozen": True,
        "postprocess_included": True,
        "postprocess_completed_frames": 5,
        "postprocess_completion_verified": True,
        "completed_frames": 5,
        "frozen_host_postprocess_contract": contract,
        "frozen_host_postprocess_contract_sha256":
            contract["contract_sha256"],
        "frozen_host_postprocess_result": result,
        "completed_task_endpoint_attestation": attestation,
        "completed_task_completion_mode": "frozen_host_tail",
        "completed_task_comparison_endpoint_contract": attestation[
            "completed_task_comparison_endpoint_contract"
        ],
        "completed_task_comparison_endpoint_contract_hash": attestation[
            "completed_task_comparison_endpoint_contract_hash"
        ],
        "completed_task_comparison_output_endpoint_id": attestation[
            "completed_task_comparison_output_endpoint_id"
        ],
        "ok": True,
        "fps_makespan": 8.5,
        "outer_makespan_verified": True,
    }


def test_nested_completed_attestation_projects_canonical_aliases(
    tmp_path: Path,
) -> None:
    row = _completed_host_tail_row()
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    (analysis / "native_full_baseline_eval.json").write_text(
        json.dumps({"rows": [row]}),
        encoding="utf-8",
    )

    projected = final_report._rows_from_native_full(tmp_path)[0]

    assert projected["completed_task_endpoint_attested"] is True
    assert projected["completed_task_endpoint_attestation_status"] == "passed"
    assert projected["completed_task_endpoint_projection_status"] == (
        "passed_existing_attestation_verified"
    )
    assert validator._completed_frozen_nms_attestation_passed(projected)


def test_missing_completed_aliases_derive_but_conflicts_fail_closed() -> None:
    nested_only = _completed_host_tail_row()
    assert validator._completed_frozen_nms_attestation_passed(nested_only)

    conflict = copy.deepcopy(nested_only)
    conflict["completed_task_endpoint_attested"] = False
    assert not validator._completed_frozen_nms_attestation_passed(conflict)

    conflict = copy.deepcopy(nested_only)
    conflict["completed_task_endpoint_attestation_status"] = (
        "passed_existing_attestation_verified"
    )
    assert not validator._completed_frozen_nms_attestation_passed(conflict)


def test_native_full_runner_projects_only_verified_nested_aliases() -> None:
    row = _completed_host_tail_row()
    attestation = row["completed_task_endpoint_attestation"]

    assert full_runner._completed_attestation_aliases(attestation) == (
        True,
        "passed",
    )
    failed = copy.deepcopy(attestation)
    failed["status"] = "failed"
    assert full_runner._completed_attestation_aliases(failed) == (
        False,
        "failed",
    )


def test_empty_quality_aliases_do_not_erase_physical_completion() -> None:
    timing = _completed_host_tail_row()
    timing.update({
        "setup_id": "setup",
        "comparison_backend": "tensorrt",
        "execution_precision": "fp16",
        "output_endpoint_id": "physical",
    })
    quality = {
        key: timing.get(key)
        for key in (
            "backend",
            "model",
            "case",
            "precision",
            "execution_precision",
            "setup_id",
            "comparison_backend",
            "task",
            "stage",
            "output_endpoint_id",
        )
    }
    quality.update({
        "completed_task_endpoint_attestation": {},
        "completed_task_endpoint_attestation_status": "",
        "completed_task_comparison_endpoint_contract": None,
    })
    summary = {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 9,
        "status": "complete",
        "row_count": 1,
        "rows": [quality],
    }

    attached, _status = final_report._attach_quality_evidence(
        [copy.deepcopy(timing)],
        summary,
        quality_summary_status="loaded",
    )

    assert attached[0]["completed_task_endpoint_attestation"] == timing[
        "completed_task_endpoint_attestation"
    ]
    assert attached[0]["completed_task_comparison_endpoint_contract"] == timing[
        "completed_task_comparison_endpoint_contract"
    ]


def _matrix(count: int = 2) -> dict:
    return {
        "expected_row_count": count,
        "present_expected_row_count": count,
        "successful_expected_row_count": count,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
    }


def test_complete_negative_semantic_result_is_not_technical_failure() -> None:
    validation = {
        "technical_error_count": 0,
        "technical_chain_complete": True,
        "rows": [
            {
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": True,
            },
            {
                "semantic_available": True,
                "semantic_ok": False,
                "claim_ok": False,
            },
        ],
    }

    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(),
        validation_payload=validation,
        validation_requested=True,
        energy_requested=False,
    )

    assert evidence["semantics"]["status"] == "complete_fail"
    assert evidence["technical_quality_failure"] is False
    assert evidence["evidence_complete"] is True
    assert evidence["scientific_ready"] is True
    assert evidence["positive_performance_claim_available"] is True
    assert blocking_status(evidence, run_mode="standard") == ""


def test_non_strict_incomplete_energy_is_partial_not_technical() -> None:
    validation = {
        "technical_error_count": 0,
        "technical_chain_complete": True,
        "rows": [
            {
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": True,
            },
        ],
    }
    non_strict = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(1),
        validation_payload=validation,
        validation_requested=True,
        energy_requested=True,
        energy_status="blocked_no_complete_pairs",
        energy_strict=False,
    )
    strict = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=_matrix(1),
        validation_payload=validation,
        validation_requested=True,
        energy_requested=True,
        energy_status="blocked_no_complete_pairs",
        energy_strict=True,
    )

    assert non_strict["technical_quality_failure"] is False
    assert non_strict["evidence_complete"] is False
    assert blocking_status(non_strict, run_mode="standard") == "partial"
    assert strict["technical_quality_failure"] is True
    assert blocking_status(strict, run_mode="standard") == "partial"  # v2.82: local gap; scientific readiness remains closed.
