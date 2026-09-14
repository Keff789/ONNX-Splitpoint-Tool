from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.artifacts import write_json
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    _sealed_full_only_quality_normalized_contract_v27538,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _full_quality_only_stale_hardware_smoke_projection,
    project_central_quality_status,
)


NA_STATUS = "not_applicable_quality_evidence_only"


def _normalized_contract() -> dict:
    return {
        "schema": "onnx-splitpoint/normalized-benchmark-results",
        "schema_version": 2,
        "model_id": "resnet50",
        "status": "quality_evidence_only_complete",
        "results": [],
        "result_count": 0,
        "matrix_complete": True,
        "performance_matrix_applicable": False,
        "performance_matrix_status": NA_STATUS,
        "quality_evidence_only_complete": True,
        "quality_evidence_count": 2,
        "expected_full_quality_count": 2,
    }


def _aggregate(tmp_path: Path, normalized: dict) -> dict[str, str]:
    model_dir = tmp_path / "models" / "resnet50"
    write_json(model_dir / "analysis" / "analysis.json", {
        "task": "classification",
        "model_resolved": True,
        "node_count": 177,
    })
    write_json(model_dir / "analysis" / "prediction.json", {
        "candidates": [{
            "case_id": "b053",
            "split_index": 53,
            "rank": 1,
        }],
    })
    write_json(model_dir / "analysis" / "final_candidate_plan.json", {
        "selected_candidates": [{
            "case_id": "b053",
            "split_index": 53,
            "rank": 1,
        }],
    })
    write_json(
        model_dir / "benchmark_results" / "normalized_results.json",
        normalized,
    )

    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.manifest = {"models": {"resnet50": {}}}
    workflow.report_paths = []
    workflow.outputs = {}
    _, _, _, status = workflow._stage_aggregate_results()
    assert status == "ok"

    with (tmp_path / "reports" / "summary.csv").open(
        newline="", encoding="utf-8",
    ) as handle:
        return next(csv.DictReader(handle))


def test_exact_rowless_quality_contract_is_reported_as_not_applicable(
    tmp_path: Path,
) -> None:
    row = _aggregate(tmp_path, _normalized_contract())
    assert row["measured_result_count"] == "0"
    assert row["benchmark_status"] == NA_STATUS
    assert row["split_measurement_status"] == NA_STATUS
    assert row["performance_matrix_applicable"] == "False"
    assert row["performance_matrix_status"] == NA_STATUS
    assert row["quality_evidence_only_complete"] == "True"
    assert row["quality_evidence_count"] == "2"
    assert row["expected_full_quality_count"] == "2"


def test_ordinary_zero_row_contract_remains_pending(tmp_path: Path) -> None:
    normalized = {
        "status": "pending_benchmark_execution",
        "results": [],
        "result_count": 0,
        "matrix_complete": False,
        "performance_matrix_applicable": True,
        "quality_evidence_only_complete": False,
        "quality_evidence_count": 0,
        "expected_full_quality_count": 0,
    }
    row = _aggregate(tmp_path, normalized)
    assert row["benchmark_status"] == "pending_execution"
    assert row["split_measurement_status"] == "pending_benchmark_execution"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "quality_evidence_only_incomplete"),
        ("performance_matrix_applicable", True),
        ("quality_evidence_only_complete", False),
        ("matrix_complete", False),
        ("results", [{"variant": "full"}]),
        ("result_count", 1),
        ("result_count", False),
        ("quality_evidence_count", 1),
        ("quality_evidence_count", True),
        ("expected_full_quality_count", 0),
    ],
)
def test_quality_only_aggregate_exception_is_fail_closed(
    key: str,
    value: object,
) -> None:
    contract = copy.deepcopy(_normalized_contract())
    contract[key] = value
    assert (
        _sealed_full_only_quality_normalized_contract_v27538(contract)
        is False
    )


def test_exact_quality_only_normalized_contract_is_recognized() -> None:
    assert (
        _sealed_full_only_quality_normalized_contract_v27538(
            _normalized_contract(),
        )
        is True
    )


def test_final_status_stays_ok_when_exact_quality_evidence_scientifically_fails(
    tmp_path: Path,
) -> None:
    quality_dir = tmp_path / "quality_management"
    quality_dir.mkdir(parents=True)
    (quality_dir / "central_quality_summary.json").write_text(
        json.dumps({
            "status": "ok",
            "request_count": 2,
            "completed_count": 2,
            "failed_count": 0,
            "merge": {"unmatched_result_count": 0},
            "results": [
                {
                    "status": "completed",
                    "technical_status": "completed",
                    "model_id": "resnet50",
                    "task": "classification",
                    "case_id": "full",
                    "variant": "full",
                    "run_id": "deepx_m1_full",
                    "source_run_id": "deepx_m1_full",
                    "backend": "deepx_m1",
                    "setup_id": "orin_nx_deepx_m1_01",
                    "execution_role": "full_quality_only",
                    "performance_claims_emitted": False,
                    "decision": "fail",
                    "primary": {
                        "metric": "top1",
                        "decision": "fail",
                    },
                    "guardrails": {
                        "top5": {"metric": "top5", "decision": "pass"},
                    },
                },
                {
                    "status": "completed",
                    "technical_status": "completed",
                    "model_id": "resnet50",
                    "task": "classification",
                    "case_id": "full",
                    "variant": "full",
                    "run_id": "native_full_tensorrt",
                    "source_run_id": "native_full_tensorrt",
                    "backend": "native_tensorrt",
                    "setup_id": "orin_nx_deepx_m1_01",
                    "execution_role": "full_quality_only",
                    "performance_claims_emitted": False,
                    "decision": "pass",
                    "primary": {
                        "metric": "top1",
                        "decision": "pass",
                    },
                    "guardrails": {
                        "top5": {"metric": "top5", "decision": "pass"},
                    },
                },
            ],
        }),
        encoding="utf-8",
    )

    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.profile_payload = {
        "execution_preset": {"id": "standard"},
        "quality_gate": {"dataset_tier": "screening"},
    }
    workflow.stage_results = [
        {"model_id": "resnet50", "stage": stage, "status": "ok"}
        for stage in (
            "run_benchmarks",
            "validate_outputs",
            "hardware_smoke",
            "evaluate_quality",
            "aggregate_results",
            "run_native_producers",
            "generate_report",
        )
    ]

    final_status, status_summary = workflow._derive_final_status()
    axes = workflow._quality_reporting_axes(final_status)

    assert final_status == "ok"
    assert status_summary["blocking_reason_count"] == 0
    assert axes["technical_status"] == "ok"
    assert axes["quality_evaluation_technical_status"] == "ok"
    assert axes["quality_decision"] == "fail"
    assert axes["scientific_status"] == "fail"
    assert axes["scientific_pass"] is False


def _write_exact_full_only_quality_acceptance(tmp_path: Path) -> None:
    model_id = "resnet50"
    setup_id = "orin_nx_hailo8_01"
    normalized = _normalized_contract()
    # A targeted missing-quality resume may execute only one previously
    # absent endpoint even though the sealed Central Quality postcondition is
    # the complete two-endpoint model set.
    normalized["quality_evidence_count"] = 1
    normalized["expected_full_quality_count"] = 1
    write_json(
        tmp_path
        / "models"
        / model_id
        / "benchmark_results"
        / "normalized_results.json",
        normalized,
    )
    expected = [
        {
            "id": "hailo8_full",
            "source_run_id": "hailo8",
            "setup_id": setup_id,
            "backend": "hailo8",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
        {
            "id": "tensorrt_at_hailo8_full",
            "source_run_id": "native_full_tensorrt",
            "setup_id": setup_id,
            "backend": "tensorrt",
            "variant": "full",
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False,
        },
    ]
    results = []
    for identity, decision in zip(expected, ("fail", "pass")):
        results.append({
            **identity,
            "model_id": model_id,
            "status": "completed",
            "technical_status": "completed",
            "decision": decision,
            "primary": {
                "metric": "top1_accuracy",
                "candidate": 0.7,
                "reference": 0.8,
                "delta": -0.1,
                "ci_low": -0.1,
                "ci_high": -0.1,
                "margin": 0.01,
                "decision": decision,
            },
        })
    write_json(
        tmp_path / "quality_management" / "central_quality_summary.json",
        {
            "status": "ok",
            "request_count": 2,
            "completed_count": 2,
            "failed_count": 0,
            "merge": {"unmatched_result_count": 0},
            "quality_acceptance_identity_contract": {
                "schema": (
                    "onnx-splitpoint/"
                    "full-only-quality-acceptance-identity-contract"
                ),
                "schema_version": 1,
                "execution_scope": "full_only",
                "identity_key_fields": [
                    "model_id",
                    "source_run_id",
                    "setup_id",
                    "backend",
                    "variant",
                    "execution_role",
                    "performance_claims_emitted",
                ],
                "model_ids": [model_id],
                "expected_identities": expected,
            },
            "results": results,
        },
    )


def test_exact_full_only_quality_makes_stale_hardware_smoke_non_blocking(
    tmp_path: Path,
) -> None:
    _write_exact_full_only_quality_acceptance(tmp_path)
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.profile_payload = {
        "execution_preset": {"id": "standard"},
        "quality_gate": {"dataset_tier": "screening"},
    }
    workflow.stage_results = [
        {"model_id": "resnet50", "stage": "run_benchmarks", "status": "ok"},
        {
            "model_id": "resnet50",
            "stage": "hardware_smoke",
            "status": "partial",
            "state": "completed",
            "complete": True,
            "error_class": "",
            "error_detail": "",
            "notes": ["pending_hardware_execution"],
        },
        {"model_id": "", "stage": "evaluate_quality", "status": "ok"},
    ]

    final_status, status_summary = workflow._derive_final_status()
    axes = workflow._quality_reporting_axes(final_status)

    assert final_status == "ok"
    assert status_summary["blocking_reason_count"] == 0
    assert status_summary["non_blocking_reason_count"] == 1
    assert axes["technical_status"] == "ok"
    assert axes["quality_evaluation_technical_status"] == "ok"
    assert axes["quality_decision"] == "fail"
    assert axes["scientific_status"] == "fail"


def test_exact_full_only_quality_does_not_hide_failed_hardware_stage(
    tmp_path: Path,
) -> None:
    _write_exact_full_only_quality_acceptance(tmp_path)
    workflow = object.__new__(EvaluationWorkflowRunner)
    workflow.run_dir = tmp_path
    workflow.profile_payload = {
        "execution_preset": {"id": "standard"},
        "quality_gate": {"dataset_tier": "screening"},
    }
    workflow.stage_results = [{
        "model_id": "resnet50",
        "stage": "hardware_smoke",
        "status": "failed",
        "error_detail": "runtime_transport_failed",
    }]

    final_status, status_summary = workflow._derive_final_status()

    assert final_status == "partial"  # v2.82: model-local transport failure remains visible.
    assert status_summary["blocking_reasons"][0]["original_stage_status"] == "failed"
    assert status_summary["blocking_reason_count"] == 1
    assert status_summary["non_blocking_reason_count"] == 0


def test_offline_projection_repairs_only_completed_stale_hardware_smoke(
    tmp_path: Path,
) -> None:
    _write_exact_full_only_quality_acceptance(tmp_path)
    central = project_central_quality_status(json.loads(
        (
            tmp_path
            / "quality_management"
            / "central_quality_summary.json"
        ).read_text(encoding="utf-8")
    ))
    run_status = {
        "status": "partial",
        "technical_status": "partial",
        "blocking_reason_count": 1,
        "blocking_reasons": [{
            "kind": "stage_status",
            "model_id": "resnet50",
            "stage": "hardware_smoke",
            "status": "partial",
            "blocking": True,
        }],
    }
    manifest = {
        "status": "partial",
        "technical_status": "partial",
        "models": {
            "resnet50": {
                "stages": {
                    "hardware_smoke": {
                        "stage": "hardware_smoke",
                        "status": "partial",
                        "state": "completed",
                        "complete": True,
                        "error_class": "",
                        "error_detail": "",
                    },
                },
            },
        },
        "root_stages": {},
    }

    projection = _full_quality_only_stale_hardware_smoke_projection(
        run_status=run_status,
        run_manifest=manifest,
        central_quality_reporting=central,
    )
    assert projection["applied"] is True
    assert projection["projected_technical_status"] == "ok"
    assert projection["stale_hardware_smoke_models"] == ["resnet50"]

    manifest["root_stages"] = {
        "aggregate_results": {
            "stage": "aggregate_results",
            "status": "failed",
        },
    }
    rejected = _full_quality_only_stale_hardware_smoke_projection(
        run_status=run_status,
        run_manifest=manifest,
        central_quality_reporting=central,
    )
    assert rejected["applied"] is False
    assert rejected["projected_technical_status"] == ""
