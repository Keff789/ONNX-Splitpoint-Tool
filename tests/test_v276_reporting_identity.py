from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _enrich_archived_setup_identity,
    _native_ranking_audit_summary,
    _performance_cohort_projection,
    _ranking_audit_request,
    _scientific_row,
    build_benchmarkset_scientific_report,
)


def test_archived_row_restores_one_exact_central_request_identity(
    tmp_path: Path,
) -> None:
    quality = tmp_path / "quality_management"
    quality.mkdir()
    identity = {
        "variant": "composed",
        "setup_id": "orin_nx_hailo8_01",
        "source_run_id": "hailo8_to_trt",
        "runtime_precision_identity": "uint8_dequant_fp16",
        "task": "detection",
        "stage": "decoded_nms",
        "endpoint_contract_hash": "a" * 64,
        "endpoint_contract_complete": True,
        "identity_valid": True,
    }
    (quality / "central_quality_summary.json").write_text(
        json.dumps({
            "results": [{
                "model_id": "yolo26s",
                "case_id": "b002",
                "variant": "composed",
                "backend": "hailo8_to_trt",
                "source_setup_id": "orin_nx_hailo8_01",
                "request_identity": identity,
            }],
        }),
        encoding="utf-8",
    )

    rows = _enrich_archived_setup_identity(tmp_path, [{
        "model_id": "yolo26s",
        "case_id": "b002",
        "variant": "split",
        "backend": "hailo8_to_tensorrt",
    }])

    assert rows[0]["setup_id"] == "orin_nx_hailo8_01"
    assert rows[0]["quality_request_identity_status"] == (
        "central_quality_exact_join"
    )
    assert rows[0]["quality_request_identities_by_variant"] == {
        "composed": identity,
    }


def test_archived_row_does_not_choose_between_conflicting_identities(
    tmp_path: Path,
) -> None:
    quality = tmp_path / "quality_management"
    quality.mkdir()
    results = []
    for precision in ("float16", "uint8_dequant_fp16"):
        results.append({
            "model_id": "m",
            "case_id": "b001",
            "variant": "composed",
            "backend": "hailo8_to_trt",
            "source_setup_id": "setup",
            "request_identity": {
                "variant": "composed",
                "setup_id": "setup",
                "source_run_id": "hailo8_to_trt",
                "runtime_precision_identity": precision,
            },
        })
    (quality / "central_quality_summary.json").write_text(
        json.dumps({"results": results}), encoding="utf-8",
    )

    row = _enrich_archived_setup_identity(tmp_path, [{
        "model_id": "m",
        "case_id": "b001",
        "variant": "split",
        "backend": "hailo8_to_tensorrt",
    }])[0]

    assert row["quality_request_identity_status"] == (
        "ambiguous_central_quality_join"
    )
    assert "quality_request_identities_by_variant" not in row


def _request(*, quality_contract: dict) -> dict:
    endpoint_hash = "a" * 64
    return {
        "backend": "hailo8_to_trt",
        "case_id": "b001",
        "setup_id": "orin_nx_hailo8_01",
        "task": "classification",
        "variant": "composed",
        "runtime_precision_identity": "float32_layout_fp16",
        "endpoint_contract_hash": endpoint_hash,
        "endpoint_contract": {
            "task": "classification",
            "stage": "classification_logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": endpoint_hash,
        },
        "quality_contract": quality_contract,
    }


def _row(request: dict) -> dict:
    return {
        "case_id": "b001",
        "variant": "split",
        "primary_variant": "composed",
        "run_id": "hailo8_to_trt",
        "backend": "hailo8_to_trt",
        "runtime_ok": True,
        "total_latency_ms": 10.0,
        "task_quality_input_requests_by_variant": {"composed": request},
    }


def test_runtime_input_encoding_is_retained_without_inventing_numeric_identity(
    tmp_path: Path,
) -> None:
    encoding = {
        "schema": "onnx-splitpoint/model-input-encoding-contract",
        "schema_version": 1,
        "image_scale": "imagenet",
        "input_dtype": "float32",
    }
    normalized = normalize_benchmark_row(
        _row(_request(quality_contract={
            "preprocessing": {
                "runtime_input_encoding": encoding,
                "runtime_input_encoding_sha256": "b" * 64,
            },
        })),
        model_id="resnet50",
        source_path=tmp_path / "result.json",
    )
    identity = normalized["quality_request_identities_by_variant"]["composed"]
    assert identity["runtime_input_encoding_identity"] == encoding
    assert identity["runtime_input_encoding_sha256"] == "b" * 64
    assert identity["runtime_numeric_input_identity"] == {}
    assert identity["runtime_numeric_input_sha256"] == ""
    assert (
        identity["runtime_numeric_input_identity_status"]
        == "runtime_input_encoding_only"
    )
    assert normalized["runtime_numeric_input_identity"] == {}
    assert normalized["runtime_input_encoding_identity"] == encoding


def test_nested_prepared_input_projects_complete_numeric_identity(
    tmp_path: Path,
) -> None:
    numeric = {
        "schema": "onnx-splitpoint/runtime-numeric-input-identity",
        "schema_version": 1,
        "backend": "deepx",
        "task": "classification",
        "runtime_input_name": "input",
        "runtime_input_shape": [1, 224, 224, 3],
        "runtime_input_dtype": "float32",
        "runtime_input_layout": "NHWC",
    }
    normalized = normalize_benchmark_row(
        _row(_request(quality_contract={
            "prepared_input_evidence": {
                "runtime_numeric_input_identity": numeric,
                "runtime_numeric_input_sha256": "c" * 64,
            },
        })),
        model_id="resnet50",
        source_path=tmp_path / "result.json",
    )
    identity = normalized["quality_request_identities_by_variant"]["composed"]
    assert identity["identity_valid"] is True
    assert identity["runtime_numeric_input_identity"] == numeric
    assert identity["runtime_numeric_input_sha256"] == "c" * 64
    assert identity["runtime_numeric_input_identity_status"] == "complete"
    assert normalized["runtime_numeric_input_identity"] == numeric
    assert normalized["runtime_numeric_input_sha256"] == "c" * 64
    report_row = _scientific_row(normalized)
    assert report_row["runtime_numeric_input_identity"] == numeric
    assert report_row["runtime_numeric_input_identity_status"] == "complete"


def test_numeric_identity_conflict_is_fail_closed(tmp_path: Path) -> None:
    embedded = {"schema": "numeric", "runtime_input_dtype": "float16"}
    source = _row(_request(quality_contract={
        "prepared_input_evidence": {
            "runtime_numeric_input_identity": embedded,
            "runtime_numeric_input_sha256": "d" * 64,
        },
    }))
    source["runtime_numeric_input_identity"] = {
        "schema": "numeric",
        "runtime_input_dtype": "float32",
    }
    source["runtime_numeric_input_sha256"] = "e" * 64
    normalized = normalize_benchmark_row(
        source,
        model_id="resnet50",
        source_path=tmp_path / "result.json",
    )
    identity = normalized["quality_request_identities_by_variant"]["composed"]
    assert identity["identity_valid"] is False
    assert identity["runtime_numeric_input_identity_status"] == "conflict"
    assert any(
        "runtime_numeric_input_identity_conflict" in error
        for error in identity["identity_errors"]
    )
    assert any(
        "runtime_numeric_input_sha256_conflict" in error
        for error in identity["identity_errors"]
    )


def test_requested_native_audit_reports_explicit_native_disabled() -> None:
    profile = {
        "selection_policy": {
            "selection_strategy": "score_independent_audit",
        },
        "native_producers": {"enabled": False},
    }
    request = _ranking_audit_request(profile, {})
    assert request["requested"] is True
    assert request["native_execution_enabled"] is False
    summary = _native_ranking_audit_summary([], request)
    assert summary["status"] == "native_disabled"
    assert summary["audit_requested"] is True
    assert summary["native_execution_enabled"] is False


def test_missing_native_config_remains_requested_but_unavailable() -> None:
    request = {
        "requested": True,
        "source": "selection_plan",
        "native_execution_enabled": None,
    }
    summary = _native_ranking_audit_summary([], request)
    assert summary["status"] == "requested_but_unavailable"
    assert summary["native_execution_enabled"] is None


def test_performance_cohorts_are_explicit_and_do_not_rewrite_claim_gate() -> None:
    rows = [
        {
            "model_id": "m",
            "case_id": "b001",
            "runtime_executable": True,
            "latency_ms": 1.0,
            "task_quality_decision": "pass",
            "performance_claim_eligible": False,
        },
        {
            "model_id": "m",
            "case_id": "b002",
            "runtime_executable": True,
            "latency_ms": 2.0,
            "task_quality_decision": "fail",
            "performance_claim_eligible": False,
        },
        {
            "model_id": "m",
            "case_id": "b003",
            "runtime_executable": False,
            "task_quality_decision": "not_evaluated",
            "performance_claim_eligible": False,
        },
        {
            "model_id": "m",
            "case_id": "b004",
            "runtime_executable": True,
            "throughput_fps": 100.0,
            "task_quality_decision": "pass",
            "performance_claim_eligible": True,
        },
        {
            "model_id": "m",
            "case_id": "b005",
            "runtime_executable": True,
            "latency_ms": 3.0,
            "measurement_valid": False,
            "terminal_failure": True,
            "task_quality_decision": "pass",
            "performance_claim_eligible": False,
        },
    ]
    projected, summary = _performance_cohort_projection(rows)
    assert summary["performance_row_count"] == 5
    assert summary["technical_cohort_count"] == 3
    assert summary["quality_cohort_count"] == 2
    assert summary["claim_cohort_count"] == 1
    assert summary["cohort_consistency_error_count"] == 0
    assert projected[0]["cohort_membership"] == ["technical", "quality"]
    assert projected[1]["cohort_membership"] == ["technical"]
    assert projected[2]["cohort_membership"] == []
    assert projected[3]["cohort_membership"] == [
        "technical", "quality", "claim",
    ]
    assert projected[4]["cohort_membership"] == []


def test_cohort_projection_flags_claim_without_quality_without_demoting_it() -> None:
    projected, summary = _performance_cohort_projection([{
        "runtime_executable": True,
        "latency_ms": 1.0,
        "task_quality_decision": "fail",
        "performance_claim_eligible": True,
    }])
    assert projected[0]["claim_cohort_eligible"] is True
    assert projected[0]["quality_cohort_eligible"] is False
    assert projected[0]["cohort_consistency_errors"] == [
        "claim_without_quality",
    ]
    assert summary["cohort_consistency_error_count"] == 1


def test_scientific_report_writes_explicit_cohort_artifacts(
    tmp_path: Path,
) -> None:
    build_benchmarkset_scientific_report(
        tmp_path,
        [{
            "model_id": "m",
            "case_id": "b001",
            "task": "classification",
            "backend": "hailo8_to_tensorrt",
            "variant": "composed",
            "runtime_executable": True,
            "total_latency_ms": 2.0,
            "accuracy_gate_decision": "pass",
            "performance_claim_eligible": False,
        }],
        plan={
            "model_id": "m",
            "model_suite": {
                "primary": [{"id": "m", "evaluation_role": "development"}],
                "reserve": [],
            },
        },
    )
    report_root = tmp_path / "scientific_report"
    cohorts = json.loads(
        (report_root / "performance_cohorts.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (report_root / "performance_cohort_summary.json").read_text(
            encoding="utf-8",
        )
    )
    assert len(cohorts) == 1
    assert cohorts[0]["technical_cohort_eligible"] is True
    # A raw benchmark flag is not central-quality evidence. The report keeps
    # the row technical-only instead of promoting it into the quality cohort.
    assert cohorts[0]["quality_cohort_eligible"] is False
    assert cohorts[0]["claim_cohort_eligible"] is False
    assert summary["technical_cohort_count"] == 1
    assert summary["quality_cohort_count"] == 0
    assert summary["claim_cohort_count"] == 0
