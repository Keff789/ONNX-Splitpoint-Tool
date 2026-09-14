from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.native_performance_reporting import (
    collect_native_performance_matrix,
)
from onnx_splitpoint_tool.validation.accuracy_gates import (
    AccuracyGatePolicy,
    apply_accuracy_gate_to_row,
    evaluate_detection_similarity,
)
from onnx_splitpoint_tool.workflow.runner import (
    _native_concise_summary_v60w,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import _write_reports


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> Any:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"_v270j_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("matched", "mean_iou", "expected"),
    [
        (9, 0.984, True),
        (8, 0.984, False),
        (9, 0.850, True),
        (9, 0.849, False),
    ],
)
def test_versioned_detection_similarity_v2(
    matched: int, mean_iou: float, expected: bool,
) -> None:
    policy = AccuracyGatePolicy()
    result = evaluate_detection_similarity(
        {
            "ref_count": 11,
            "matched": matched,
            "mean_iou": mean_iou,
            "iou_threshold": 0.50,
        },
        {"ref_count": 11, "matched": 11, "match_ratio": 1.0},
        policy,
    )

    assert policy.schema_version == 3
    assert policy.native_self_reference_policy_id == (
        "class_aware_iou50_postnms_v2"
    )
    assert policy.native_self_reference_min_match == 0.80
    assert policy.native_self_reference_min_mean_iou == 0.85
    assert result["numerical_similarity_pass"] is expected
    assert result["numerical_similarity_policy_id"] == (
        "class_aware_iou50_postnms_v2"
    )


@pytest.mark.parametrize("include_explicit_threshold", [False, True])
def test_archived_v1_policy_keeps_090_decision(
    include_explicit_threshold: bool,
) -> None:
    native_contract = {}
    if include_explicit_threshold:
        native_contract[
            "detection_self_reference_min_match_ratio"
        ] = 0.90
    policy = AccuracyGatePolicy.from_mapping({
        "schema": "onnx-splitpoint/task-quality-policy",
        "schema_version": 2,
        "name": "thesis_task_quality_v1",
        "profile_id": "thesis_task_quality_v1",
        "native_contract": native_contract,
    })
    result = evaluate_detection_similarity(
        {
            "ref_count": 11,
            "matched": 9,
            "mean_iou": 0.984,
            "iou_threshold": 0.50,
        },
        {},
        policy,
    )

    assert policy.schema_version == 2
    assert policy.native_self_reference_min_match == 0.90
    assert result["numerical_similarity_pass"] is False
    assert "native_self_reference_policy_id" not in policy.as_dict()


def test_archived_v1_policy_does_not_require_new_mean_iou_guard() -> None:
    policy = AccuracyGatePolicy.from_mapping({
        "schema_version": 2,
        "native_contract": {
            "detection_self_reference_min_match_ratio": 0.90,
        },
    })
    result = evaluate_detection_similarity(
        {
            "ref_count": 10,
            "matched": 9,
            "iou_threshold": 0.50,
        },
        {},
        policy,
    )

    assert result["numerical_similarity_pass"] is True
    assert result["numerical_similarity_mean_iou"] is None


def test_explicit_archived_v2_mean_iou_threshold_stays_090() -> None:
    archived_shapes = (
        {"native_self_reference_min_mean_iou": 0.90},
        {
            "native_contract": {
                "detection_self_reference": {
                    "min_mean_matched_iou": 0.90,
                },
            },
        },
        {
            "native_contract": {
                "detection_self_reference_min_mean_iou": 0.90,
            },
        },
        {"native_min_mean_iou": 0.90},
    )
    for archived in archived_shapes:
        policy = AccuracyGatePolicy.from_mapping({
            "schema_version": 3,
            **archived,
        })
        result = evaluate_detection_similarity(
            {
                "ref_count": 10,
                "matched": 9,
                "mean_iou": 0.887,
                "iou_threshold": 0.50,
            },
            {},
            policy,
        )

        assert policy.native_self_reference_min_mean_iou == 0.90
        assert result["numerical_similarity_pass"] is False


def test_gui_profile_export_declares_new_policy_schema() -> None:
    source = (
        ROOT / "onnx_splitpoint_tool" / "gui" / "profile_editor.py"
    ).read_text(encoding="utf-8")
    quality_start = source.index('"quality_gate": {')
    quality_end = source.index('"ranking_validation": {', quality_start)
    quality_block = source[quality_start:quality_end]

    assert '"schema_version": 3' in quality_block
    assert '"class_aware_iou50_postnms_v2"' in quality_block
    assert '"min_reference_match_ratio": 0.80' in quality_block
    assert '"min_mean_matched_iou": 0.85' in quality_block


def test_structure_numerical_and_task_quality_are_independent() -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    row = {
        "task": "detection",
        "buildable": True,
        "runtime_executable": True,
        "interface_contract_pass": True,
        "strict_boundary_numeric_pass": False,
        "task_quality_gate": {
            "decision": "pass",
            "tier": "final",
            "primary": {"metric": "coco_ap_50_95"},
        },
    }

    apply_accuracy_gate_to_row(row, AccuracyGatePolicy())

    assert row["structural_contract_pass"] is True
    assert row["contract_consistent"] is True
    assert row["numerical_similarity_pass"] is False
    assert row["task_quality_pass"] is True
    assert row["gate_status"] == "numerical_similarity_failed"
    assert validator._technical_quality_error(row) is False

    structurally_broken = dict(row)
    structurally_broken["structural_contract_pass"] = False
    structurally_broken["contract_consistent"] = False
    structurally_broken["numerical_similarity_pass"] = True
    assert validator._technical_quality_error(structurally_broken) is True


@pytest.mark.parametrize(
    ("schema_version", "expected_valid"),
    [(8, True), (9, True), (10, False)],
)
def test_quality_summary_schema_and_lossless_contract_join(
    schema_version: int,
    expected_valid: bool,
) -> None:
    reporter = _load_script("native_producer_final_report.py")
    performance = {
        "backend": "hailo10h_to_trt",
        "model": "yolov7_paper",
        "case": "b044",
        "precision": "fp16",
        "setup_id": "orin-1",
        "comparison_backend": "hailo10h",
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "claim_ok": True,
        "structural_contract_pass": True,
    }
    quality = {
        **performance,
        "status": "claim_ok",
        "gate_status": "eligible",
        "numerical_similarity_mean_iou": 0.0,
        "numerical_similarity_mean_iou_threshold": 0.9,
        "e2e_scope": "full_task_pipeline",
        "e2e_claim_eligible": False,
        "host_postprocessing_available": False,
        "host_tail_available": False,
        "postprocess_completed_frames": 0,
        "completed_task_stage": "decoded_nms",
        "completed_task_endpoint_attested": False,
    }
    joined, metadata = reporter._attach_quality_evidence(
        [dict(performance)],
        {
            "schema": (
                "onnx-splitpoint/native-producer-validation-summary"
            ),
            "schema_version": schema_version,
            "status": "complete",
            "row_count": 1,
            "rows": [quality],
        },
        quality_summary_status="loaded",
    )

    assert metadata["schema_valid"] is expected_valid
    row = joined[0]
    assert row["stage"] == "raw_head"
    if expected_valid:
        assert row["numerical_similarity_mean_iou"] == 0.0
        assert row["numerical_similarity_mean_iou_threshold"] == 0.9
        assert row["e2e_claim_eligible"] is False
        assert row["host_postprocessing_available"] is False
        assert row["postprocess_completed_frames"] == 0
        assert row["completed_task_stage"] == "decoded_nms"
        assert row["completed_task_endpoint_attested"] is False
        assert row["quality_physical_evidence_conflict"] is False
    else:
        assert "numerical_similarity_mean_iou" not in row
        assert metadata["quality_row_count"] == 0


def test_quality_summary_schema_v10_requires_exact_boolean_axis_counts() -> None:
    reporter = _load_script("native_producer_final_report.py")
    row = {
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "quality_claim_result_verified": False,
    }
    summary = {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 10,
        "status": "complete",
        "row_count": 1,
        "rows": [row],
        "precision_quality_binding_verified_count": 1,
        "task_quality_observation_valid_count": 1,
        "quality_claim_result_verified_count": 0,
    }

    assert reporter._quality_summary_schema_valid(summary) is True

    malformed_axis = {
        **summary,
        "rows": [{**row, "task_quality_observation_valid": "True"}],
    }
    assert reporter._quality_summary_schema_valid(malformed_axis) is False

    inconsistent_count = {
        **summary,
        "quality_claim_result_verified_count": 1,
    }
    assert reporter._quality_summary_schema_valid(inconsistent_count) is False


def test_quality_join_preserves_physical_stage_and_fails_on_conflict(
    tmp_path: Path,
) -> None:
    reporter = _load_script("native_producer_final_report.py")
    performance = {
        "backend": "hailo10h_to_trt",
        "model": "yolov7_paper",
        "case": "b044",
        "precision": "fp16",
        "setup_id": "orin-1",
        "comparison_backend": "hailo10h",
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "claim_ok": True,
        "structural_contract_pass": True,
    }
    quality = {
        **performance,
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "completed_task_stage": "decoded_nms",
        "claim_ok": True,
        "structural_contract_pass": True,
        "status": "claim_ok",
        "gate_status": "eligible",
    }
    joined, metadata = reporter._attach_quality_evidence(
        [dict(performance)],
        {
            "schema": (
                "onnx-splitpoint/native-producer-validation-summary"
            ),
            "schema_version": 9,
            "status": "complete",
            "row_count": 1,
            "rows": [quality],
        },
        quality_summary_status="loaded",
    )

    row = joined[0]
    assert metadata["schema_valid"] is True
    assert row["stage"] == "raw_head"
    assert row["contract_family"] == "raw_head"
    assert row["completed_task_stage"] == "decoded_nms"
    assert row["quality_physical_evidence_conflict"] is True
    assert set(row["quality_physical_evidence_conflict_fields"]) == {
        "stage", "contract_family",
    }
    assert row["structural_contract_pass"] is False
    assert row["claim_ok"] is False

    reports = tmp_path / "reports"
    _write(
        reports / "native_producer_combined_summary.json",
        {"rows": [row]},
    )
    _write(
        reports
        / "native_validation"
        / "native_producer_validation_summary.json",
        {"rows": [quality]},
    )
    _, concise_rows = _native_concise_summary_v60w(reports)
    assert concise_rows[0]["stage"] == "raw_head"
    assert concise_rows[0]["contract_family"] == "raw_head"
    assert concise_rows[0]["completed_task_stage"] == "decoded_nms"
    assert concise_rows[0]["structural_contract_pass"] is False
    assert concise_rows[0]["claim_ok"] is False
    assert (
        concise_rows[0]["quality_physical_evidence_conflict"] is True
    )


def test_claim_ok_is_fail_closed_without_mixing_evidence_axes() -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    row = {
        "task": "detection",
        "buildable": True,
        "runtime_executable": True,
        "claim_ok": True,
        "ok": True,
        "status": "claim_ok",
        "interface_contract_pass": False,
        "strict_boundary_numeric_pass": True,
        "task_quality_gate": {
            "decision": "pass",
            "tier": "final",
            "primary": {"metric": "coco_ap_50_95"},
        },
    }

    apply_accuracy_gate_to_row(row, AccuracyGatePolicy())

    assert row["claim_ok"] is False
    assert row["claim_ok_source"] is True
    assert row["claim_ok_structural_clamped"] is True
    assert row["structural_contract_pass"] is False
    assert row["numerical_similarity_pass"] is True
    assert row["task_quality_pass"] is True
    assert row["eligible_for_ranking"] is False
    assert row["status"] == "structural_contract_failed"

    validator._finalize_structural_claim_contract(row)
    assert row["claim_ok"] is False
    assert row["ok"] is False
    assert row["status"] == "structural_contract_failed"

    late_claim = {
        "task": "detection",
        "buildable": True,
        "runtime_executable": True,
        "claim_ok": False,
        "interface_contract_pass": False,
    }
    apply_accuracy_gate_to_row(late_claim, AccuracyGatePolicy())
    assert late_claim["claim_ok_source"] is False
    late_claim.update({"claim_ok": True, "status": "claim_ok"})
    apply_accuracy_gate_to_row(late_claim, AccuracyGatePolicy())
    assert late_claim["claim_ok_source"] is True
    assert late_claim["claim_ok"] is False
    assert late_claim["claim_ok_structural_clamped"] is True


@pytest.mark.parametrize(
    ("family_backend", "source_run_id", "expected_backend"),
    [
        ("deepx", "deepx_m1_to_tensorrt", "deepx_to_trt"),
        ("hailo10h", "hailo10_to_trt", "hailo10h_to_trt"),
    ],
)
@pytest.mark.parametrize("case_id", ["b044", "b116", "b216"])
def test_final_report_uses_sealed_split_pipeline_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family_backend: str,
    source_run_id: str,
    expected_backend: str,
    case_id: str,
) -> None:
    reporter = _load_script("native_producer_final_report.py")
    manifest = tmp_path / "native_outputs_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    manifest_sha = reporter._hash_file(manifest)
    h1, h2, h3 = "1" * 64, "2" * 64, "3" * 64
    binding = {
        "binding_sha256": "4" * 64,
        "eval_run_id": "run-1",
        "source_run_id": source_run_id,
        "source_request_sha256": h1,
        "central_result_sha256": h2,
        "central_quality_selection_sha256": h3,
        "preselection": {
            "backend": expected_backend,
            "model": "yolov7_paper",
            "case": case_id,
            "precision": "fp16",
        },
    }
    command = {
        "backend": expected_backend,
        "contract_sha256": "5" * 64,
        "native_split_quality_binding": binding,
        "artifacts": {
            "semantic_output_manifest": {"sha256": manifest_sha},
        },
    }
    attestation = {
        "semantic_output_manifest_sha256": manifest_sha,
        "source_request_sha256": h1,
        "native_split_quality_source_request_sha256": h1,
        "native_split_quality_central_result_sha256": h2,
        "native_split_quality_selection_sha256": h3,
    }
    source = {
        "backend": family_backend,
        "source_run_id": source_run_id,
        "model": "yolov7_paper",
        "case": case_id,
        "precision": "fp16",
        "native_command_contract": command,
        "native_split_quality_binding": binding,
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_split_quality_consumer_attestation": attestation,
        "source_request_sha256": h1,
        "native_split_quality_source_request_sha256": h1,
        "native_split_quality_central_result_sha256": h2,
        "native_split_quality_selection_sha256": h3,
    }
    observed: dict[str, Any] = {}

    def _bind(**kwargs: Any) -> tuple[dict[str, Any], str]:
        observed.update(kwargs["native_row"])
        assert kwargs["verification_mode"] == "portable"
        return binding, (
            "portable_binding_command_and_consumer_attestation_exact_match"
        )

    monkeypatch.setattr(reporter, "_bind_quality_to_native_split", _bind)
    monkeypatch.setattr(
        reporter,
        "_verify_manifest_payload_files",
        lambda _path: (True, "payload_files_rehashed"),
    )

    result = reporter._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        sources=[source],
    )

    assert observed["backend"] == expected_backend
    assert result["native_split_final_portable_binding_valid"] is True
    assert result["native_split_semantic_binding_valid"] is True


def _full_detection_row() -> dict[str, Any]:
    completed_contract = {
        "schema": "onnx-splitpoint/completed-task-endpoint-contract",
        "stage": "decoded_nms",
    }
    comparison_contract = {
        "schema": "onnx-splitpoint/completed-task-comparison-endpoint",
        "stage": "decoded_nms",
    }
    frozen_contract = {
        "schema": "onnx-splitpoint/frozen-native-detection-postprocess",
        "decoder_id": "sentinel-decoder",
    }
    frozen_result = {
        "task": "detection",
        "contract_family": "decoded_nms",
        "detection_count": 0,
    }
    completed_attestation = {
        "attested": True,
        "stage": "decoded_nms",
        "output_endpoint_id": "decoded_nms:sha256:abc",
        "endpoint_contract_hash": "a" * 64,
        "completed_endpoint_contract": completed_contract,
    }
    return {
        "backend": "native_full_tensorrt",
        "model": "yolov7_paper",
        "task": "detection",
        "ok": True,
        "fps_makespan": 100.0,
        "precision": "fp16",
        "setup_id": "jetson-1",
        "comparison_backend": "hailo10h",
        "stage": "decoded_nms",
        "output_format": "bn6_detections",
        "contract_family": "decoded_nms",
        "contract_source": "sentinel",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "b" * 64,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "endpoint_contract_hash": "b" * 64,
        },
        "accelerator_output_stage": "decoded_nms",
        "accelerator_output_contract_family": "decoded_nms",
        "accelerator_endpoint_contract_hash": "b" * 64,
        "accelerator_output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "endpoint_contract_hash": "b" * 64,
        },
        "e2e_scope": "full_task_pipeline",
        "e2e_claim_eligible": True,
        "e2e_contract_reason": "completed_endpoint_attested",
        "comparison_endpoint_stratum": "decoded_nms",
        "measurement_concurrency": 1,
        "requires_host_decode_nms": False,
        "postprocess_included": True,
        "postprocess_location": "accelerator_runtime",
        "host_postprocess_frozen": False,
        "host_postprocessing_available": False,
        "host_tail_available": False,
        "host_postprocess_required": False,
        "host_tail_required": False,
        "host_postprocessing_evidence_status": (
            "failed_incomplete_canonical_host_postprocess_evidence"
        ),
        "host_postprocessing_evidence_source": (
            "completed_task_host_postprocess_attestation_v1"
        ),
        "host_postprocessing_legacy_alias_conflict": False,
        "decoder_contract_pass": False,
        "nms_ok": False,
        "decoder_id": "",
        "postprocess_completed_frames": 100,
        "postprocess_completion_verified": True,
        "frozen_host_postprocess_contract": frozen_contract,
        "frozen_host_postprocess_contract_sha256": "c" * 64,
        "frozen_host_postprocess_result": frozen_result,
        "completed_task_stage": "decoded_nms",
        "completed_task_contract_family": "decoded_nms",
        "completed_task_endpoint_contract": completed_contract,
        "completed_task_endpoint_contract_hash": "a" * 64,
        "completed_task_output_endpoint_id": "decoded_nms:sha256:abc",
        "completed_task_comparison_endpoint_contract": (
            comparison_contract
        ),
        "completed_task_comparison_endpoint_contract_hash": "d" * 64,
        "completed_task_comparison_output_endpoint_id": (
            "detection:decoded_nms:comparison:" + "d" * 64
        ),
        "completed_task_completion_mode": "accelerator_runtime",
        "completed_task_endpoint_attested": True,
        "completed_task_endpoint_attestation": completed_attestation,
        "completed_task_endpoint_attestation_status": "passed",
        "structural_contract_pass": True,
        "structural_contract_status": "pass",
        "structural_contract_reason": "completed_endpoint_contract_complete",
        "numerical_similarity_pass": True,
        "numerical_similarity_status": "passed",
        "numerical_similarity_reason": "class_aware_similarity_passed",
        "numerical_similarity_policy_id": (
            "class_aware_iou50_postnms_v2"
        ),
        "numerical_similarity_scope": (
            "class_aware_postnms_detection"
        ),
        "numerical_similarity_metric": (
            "reference_match_ratio_and_mean_matched_iou"
        ),
        "numerical_similarity_value": 9 / 11,
        "numerical_similarity_threshold": 0.80,
        "numerical_similarity_mean_iou": 0.984,
        "numerical_similarity_mean_iou_threshold": 0.90,
        "task_quality_pass": True,
        "task_quality_status": "pass",
        "task_quality_reason": "central_dataset_gate_passed",
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
        "repetition_count_valid": 1,
        "repetition_status": "complete",
        "repetition_aggregation": "single_observation",
    }


def test_measurement_contract_reaches_combined_concise_and_scientific_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = _load_script("native_producer_final_report.py")
    producer_root = tmp_path / "producer"
    _write(
        producer_root / "analysis_tables" / "native_full_baseline_eval.json",
        {"rows": [_full_detection_row()]},
    )
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_producer_final_report.py",
            "--root",
            str(producer_root),
            "--out-dir",
            str(reports),
        ],
    )
    assert reporter.main() == 0

    combined = json.loads(
        (reports / "native_producer_combined_summary.json").read_text(
            encoding="utf-8"
        )
    )
    combined_row = combined["rows"][0]
    expected = _full_detection_row()
    for field in (
        "e2e_scope",
        "e2e_claim_eligible",
        "e2e_contract_reason",
        "comparison_endpoint_stratum",
        "measurement_concurrency",
        "postprocess_location",
        "host_postprocess_frozen",
        "host_postprocessing_available",
        "host_tail_available",
        "host_postprocessing_legacy_alias_conflict",
        "decoder_contract_pass",
        "nms_ok",
        "frozen_host_postprocess_contract",
        "frozen_host_postprocess_result",
        "completed_task_stage",
        "completed_task_endpoint_contract",
        "completed_task_endpoint_contract_hash",
        "completed_task_output_endpoint_id",
        "completed_task_comparison_endpoint_contract",
        "completed_task_comparison_endpoint_contract_hash",
        "completed_task_comparison_output_endpoint_id",
        "completed_task_endpoint_attested",
        "completed_task_endpoint_attestation",
        "structural_contract_pass",
        "numerical_similarity_pass",
        "numerical_similarity_mean_iou",
        "numerical_similarity_mean_iou_threshold",
        "task_quality_pass",
    ):
        assert combined_row[field] == expected[field]
    assert combined["schema_version"] == 7

    with (
        reports / "native_producer_combined_summary.csv"
    ).open(encoding="utf-8", newline="") as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["e2e_scope"] == "full_task_pipeline"
    assert json.loads(
        csv_row["completed_task_endpoint_contract"]
    ) == expected["completed_task_endpoint_contract"]
    assert json.loads(
        csv_row["frozen_host_postprocess_contract"]
    ) == expected["frozen_host_postprocess_contract"]
    assert csv_row["host_postprocessing_available"] == "False"
    assert csv_row["numerical_similarity_mean_iou"] == "0.984"
    combined_md = (
        reports / "native_producer_combined_summary.md"
    ).read_text(encoding="utf-8")
    assert "E2E scope" in combined_md
    assert "decoded_nms" in combined_md

    _write(
        reports
        / "native_validation"
        / "native_producer_validation_summary.json",
        {
            "rows": [{
                **combined_row,
                "semantic_ok": True,
                "claim_ok": False,
                "status": "screening_only",
            }],
        },
    )
    _, concise_rows = _native_concise_summary_v60w(reports)
    concise = concise_rows[0]
    assert concise["e2e_scope"] == "full_task_pipeline"
    assert concise["completed_task_endpoint_attestation"] == (
        expected["completed_task_endpoint_attestation"]
    )
    assert concise["host_postprocess_frozen"] is False
    assert concise["host_postprocessing_available"] is False
    assert concise["numerical_similarity_mean_iou"] == 0.984
    assert concise["numerical_similarity_mean_iou_threshold"] == 0.9
    assert concise["numerical_similarity_policy_id"] == (
        "class_aware_iou50_postnms_v2"
    )
    with (
        reports / "native_stage_concise_summary.csv"
    ).open(encoding="utf-8", newline="") as handle:
        concise_csv = next(csv.DictReader(handle))
    assert json.loads(
        concise_csv["frozen_host_postprocess_contract"]
    ) == expected["frozen_host_postprocess_contract"]
    assert json.loads(
        concise_csv["completed_task_endpoint_attestation"]
    ) == expected["completed_task_endpoint_attestation"]
    assert concise_csv["host_postprocessing_available"] == "False"
    assert concise_csv["numerical_similarity_mean_iou"] == "0.984"

    _write(
        reports / "native_expected_matrix.json",
        {
            "expected_row_count": 1,
            "present_expected_row_count": 1,
            "successful_expected_row_count": 1,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
            "present_expected_rows": [combined_row],
        },
    )
    matrix = collect_native_performance_matrix(tmp_path)
    observation = matrix["observations"][0]
    assert matrix["schema_version"] == 5
    assert observation["e2e_scope"] == "full_task_pipeline"
    assert observation["completed_task_endpoint_contract"] == (
        expected["completed_task_endpoint_contract"]
    )
    assert observation["completed_task_endpoint_attestation"] == (
        expected["completed_task_endpoint_attestation"]
    )
    assert observation["postprocess_location"] == "accelerator_runtime"
    assert observation["host_postprocess_frozen"] is False
    assert observation["host_postprocessing_available"] is False
    assert observation["numerical_similarity_mean_iou"] == 0.984
    assert observation["numerical_similarity_mean_iou_threshold"] == 0.9

    report_root = reports / "scientific"
    _write_reports(
        report_root,
        {
            "created_at": "2026-07-25T00:00:00+02:00",
            "profile_id": "p1-test",
            "rows": [],
            "summary": {},
            "native_performance_matrix": matrix,
        },
    )
    scientific_observations = json.loads(
        (
            report_root / "native_performance_observations.json"
        ).read_text(encoding="utf-8")
    )
    scientific_row = scientific_observations[0]
    assert scientific_row["e2e_scope"] == "full_task_pipeline"
    assert scientific_row["completed_task_endpoint_contract"] == (
        expected["completed_task_endpoint_contract"]
    )
    with (
        report_root / "native_performance_observations.csv"
    ).open(encoding="utf-8", newline="") as handle:
        scientific_csv = next(csv.DictReader(handle))
    assert json.loads(
        scientific_csv["completed_task_endpoint_attestation"]
    ) == expected["completed_task_endpoint_attestation"]
    assert json.loads(
        scientific_csv["frozen_host_postprocess_contract"]
    ) == expected["frozen_host_postprocess_contract"]
    assert scientific_csv["host_postprocessing_available"] == "False"
    assert scientific_csv["numerical_similarity_mean_iou"] == "0.984"
    scientific_md = (
        report_root / "native_performance_matrix.md"
    ).read_text(encoding="utf-8")
    assert "E2E scope" in scientific_md
    assert "Completed endpoint" in scientific_md
    assert "Host tail frozen" in scientific_md
    assert "Mean IoU" in scientific_md
    assert "E2E eligible" in scientific_md
    scientific_tex = (
        report_root
        / "thesis_tables"
        / "native_performance_observations.tex"
    ).read_text(encoding="utf-8")
    assert "Mean IoU" in scientific_tex
    assert "Completed comparison endpoint" in scientific_tex


def test_missing_legacy_host_postprocess_evidence_remains_unknown(
    tmp_path: Path,
) -> None:
    reporter = _load_script("native_producer_final_report.py")
    root = tmp_path / "producer"
    _write(
        root / "analysis_tables" / "native_full_baseline_eval.json",
        {
            "rows": [{
                "backend": "native_full_tensorrt",
                "model": "resnet50",
                "task": "classification",
                "ok": True,
                "fps_makespan": 10.0,
            }],
        },
    )
    row = reporter._rows_from_native_full(root)[0]

    assert row["host_postprocess_frozen"] is None
    assert row["postprocess_included"] is None
    assert row["postprocess_completed_frames"] is None
    assert row["postprocess_completion_verified"] is None
    assert row["completed_task_endpoint_attestation"] is None


def test_false_zero_and_unknown_survive_every_reporting_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reporter = _load_script("native_producer_final_report.py")
    explicit = {
        **_full_detection_row(),
        "model": "sentinel_false_zero",
        "claim_ok": False,
        "semantic_ok": False,
        "contract_consistent": False,
        "structural_contract_pass": False,
        "numerical_similarity_pass": False,
        "numerical_similarity_mean_iou": 0.0,
        "e2e_claim_eligible": False,
        "postprocess_completed_frames": 0,
        "postprocess_completion_verified": False,
        "completed_task_endpoint_attested": False,
        "completed_task_endpoint_attestation": {
            "attested": False,
            "status": "failed",
        },
    }
    unknown = {
        "backend": "native_full_tensorrt",
        "model": "sentinel_unknown",
        "task": "classification",
        "ok": True,
        "fps_makespan": 1.0,
        "precision": "fp16",
        "setup_id": "jetson-2",
        "comparison_backend": "deepx",
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
        "repetition_count_valid": 1,
        "repetition_status": "complete",
        "repetition_aggregation": "single_observation",
    }
    producer_root = tmp_path / "producer"
    _write(
        producer_root / "analysis_tables" / "native_full_baseline_eval.json",
        {"rows": [explicit, unknown]},
    )
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_producer_final_report.py",
            "--root",
            str(producer_root),
            "--out-dir",
            str(reports),
        ],
    )
    assert reporter.main() == 0

    combined = json.loads(
        (reports / "native_producer_combined_summary.json").read_text(
            encoding="utf-8"
        )
    )
    combined_by_model = {
        row["model"]: row for row in combined["rows"]
    }
    false_row = combined_by_model["sentinel_false_zero"]
    unknown_row = combined_by_model["sentinel_unknown"]
    for field in (
        "claim_ok",
        "semantic_ok",
        "contract_consistent",
        "structural_contract_pass",
        "numerical_similarity_pass",
        "e2e_claim_eligible",
        "host_postprocessing_available",
        "host_tail_available",
        "postprocess_completion_verified",
        "completed_task_endpoint_attested",
    ):
        assert false_row[field] is False
    assert false_row["numerical_similarity_mean_iou"] == 0.0
    assert false_row["postprocess_completed_frames"] == 0
    for field in (
        "numerical_similarity_mean_iou",
        "e2e_claim_eligible",
        "postprocess_completed_frames",
        "postprocess_completion_verified",
        "completed_task_endpoint_attested",
    ):
        assert unknown_row[field] is None
    assert unknown_row["host_postprocessing_available"] is False
    assert unknown_row["host_tail_available"] is False
    assert unknown_row["host_postprocessing_evidence_status"] == (
        "not_required"
    )

    _write(
        reports
        / "native_validation"
        / "native_producer_validation_summary.json",
        {
            "rows": [
                {**row, "status": "not_claimable"}
                for row in combined["rows"]
            ],
        },
    )
    _, concise_rows = _native_concise_summary_v60w(reports)
    concise_by_model = {
        row["model"]: row for row in concise_rows
    }
    for field in (
        "e2e_claim_eligible",
        "postprocess_completion_verified",
        "completed_task_endpoint_attested",
    ):
        assert concise_by_model["sentinel_false_zero"][field] is False
        assert concise_by_model["sentinel_unknown"][field] is None
    assert concise_by_model["sentinel_unknown"][
        "host_postprocessing_available"
    ] is False
    assert concise_by_model["sentinel_unknown"][
        "host_tail_available"
    ] is False
    assert (
        concise_by_model["sentinel_false_zero"][
            "numerical_similarity_mean_iou"
        ]
        == 0.0
    )
    assert (
        concise_by_model["sentinel_false_zero"][
            "postprocess_completed_frames"
        ]
        == 0
    )

    _write(
        reports / "native_expected_matrix.json",
        {
            "expected_row_count": 2,
            "present_expected_row_count": 2,
            "successful_expected_row_count": 2,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
            "present_expected_rows": combined["rows"],
        },
    )
    matrix = collect_native_performance_matrix(tmp_path)
    matrix_by_model = {
        row["model"]: row for row in matrix["observations"]
    }
    for field in (
        "e2e_claim_eligible",
        "postprocess_completion_verified",
        "completed_task_endpoint_attested",
    ):
        assert matrix_by_model["sentinel_false_zero"][field] is False
        assert matrix_by_model["sentinel_unknown"][field] is None
    assert matrix_by_model["sentinel_unknown"][
        "host_postprocessing_available"
    ] is False
    assert matrix_by_model["sentinel_unknown"][
        "host_tail_available"
    ] is False
    assert (
        matrix_by_model["sentinel_false_zero"][
            "numerical_similarity_mean_iou"
        ]
        == 0.0
    )
    assert (
        matrix_by_model["sentinel_false_zero"][
            "postprocess_completed_frames"
        ]
        == 0
    )

    report_root = reports / "scientific"
    _write_reports(
        report_root,
        {
            "created_at": "2026-07-26T00:00:00+02:00",
            "profile_id": "sentinel-test",
            "rows": [],
            "summary": {},
            "native_performance_matrix": matrix,
        },
    )
    scientific_rows = json.loads(
        (
            report_root / "native_performance_observations.json"
        ).read_text(encoding="utf-8")
    )
    scientific_by_model = {
        row["model"]: row for row in scientific_rows
    }
    assert (
        scientific_by_model["sentinel_false_zero"][
            "numerical_similarity_mean_iou"
        ]
        == 0.0
    )
    assert (
        scientific_by_model["sentinel_unknown"][
            "numerical_similarity_mean_iou"
        ]
        is None
    )

    for relative in (
        "native_producer_combined_summary.csv",
        "native_stage_concise_summary.csv",
        "scientific/native_performance_observations.csv",
    ):
        with (reports / relative).open(
            encoding="utf-8", newline="",
        ) as handle:
            csv_rows = {
                row.get("model") or row.get("model_id"): row
                for row in csv.DictReader(handle)
            }
        assert (
            csv_rows["sentinel_false_zero"][
                "numerical_similarity_mean_iou"
            ]
            in {"0.0", "0"}
        )
        assert (
            csv_rows["sentinel_false_zero"][
                "host_postprocessing_available"
            ]
            == "False"
        )
        assert (
            csv_rows["sentinel_false_zero"][
                "postprocess_completed_frames"
            ]
            == "0"
        )
        assert (
            csv_rows["sentinel_unknown"][
                "numerical_similarity_mean_iou"
            ]
            == ""
        )
        assert (
            csv_rows["sentinel_unknown"][
                "host_postprocessing_available"
            ]
            == "False"
        )


def test_p1_remote_script_mirrors_are_byte_identical() -> None:
    for name in (
        "native_producer_final_report.py",
        "native_producer_validate_visualize.py",
        "native_yolo_full_self_reference_probe.py",
        "update_evalset_native_producers.py",
    ):
        assert (ROOT / "scripts" / name).read_bytes() == (
            ROOT
            / "onnx_splitpoint_tool"
            / "resources"
            / "remote_scripts"
            / name
        ).read_bytes()
