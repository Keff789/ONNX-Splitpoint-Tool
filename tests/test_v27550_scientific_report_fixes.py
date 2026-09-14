from __future__ import annotations

import csv
import json
from pathlib import Path

from onnx_splitpoint_tool.ranking_methods import canonical_direction
from onnx_splitpoint_tool.workflow.cross_runner_reporting import (
    _exact_identity,
    _identity_resolution_errors,
    compute_cross_runner_report,
)
from onnx_splitpoint_tool.workflow.results import (
    _row_identity_for_dedupe_v58f,
    normalize_benchmark_row,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _ranking_method_comparison,
    _scientific_decision_axes,
    build_benchmarkset_scientific_report,
)


def _nested_setup_row(setup_id: str) -> dict:
    return {
        "case_id": "b001",
        "variant": "composed",
        "run_id": "hailo8_to_trt",
        "backend": "hailo8_to_tensorrt",
        "runtime_ok": True,
        "total_latency_ms": 10.0,
        "task_quality_gates_by_variant": {
            "composed": {
                "quality_input_request": {
                    "producer_identity": {"setup_id": setup_id},
                }
            }
        },
    }


def _realistic_composed_request(
    setup_id: str,
    *,
    backend: str = "hailo8_to_trt",
    endpoint_hash: str = "3af2122851f19c5bf2b16b9c3ef78a6ba11452db33ff8fb915b76e3461bbcb28",
) -> dict:
    """Relevant identity surface of the real v2.75.49 composed request."""
    return {
        "schema": "onnx-splitpoint/central-quality-evaluation-request",
        "backend": backend,
        "case_id": "b001",
        "setup_id": setup_id,
        "task": "classification",
        "variant": "composed",
        "runtime_precision_identity": "float32_layout_fp16",
        "endpoint_contract_hash": endpoint_hash,
        "endpoint_contract": {
            "task": "classification",
            "stage": "classification_logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": endpoint_hash,
            "output_endpoint_attestation": {
                "attested": True,
                "endpoint": "classification_logits",
                "stage": "classification_logits",
                "status": "passed",
                "endpoint_contract_hash": endpoint_hash,
            },
        },
    }


def test_nested_setup_identity_is_preserved_before_dedupe(tmp_path: Path) -> None:
    first = normalize_benchmark_row(
        _nested_setup_row("orin_nx_hailo8_01"),
        model_id="m",
        source_path=tmp_path / "first.json",
    )
    second = normalize_benchmark_row(
        _nested_setup_row("orin_nx_hailo8_02"),
        model_id="m",
        source_path=tmp_path / "second.json",
    )
    assert first["setup_id"] == "orin_nx_hailo8_01"
    assert second["setup_id"] == "orin_nx_hailo8_02"
    assert _row_identity_for_dedupe_v58f(first) != _row_identity_for_dedupe_v58f(second)


def test_real_nested_request_projects_complete_variant_identity(
    tmp_path: Path,
) -> None:
    row = {
        "case_id": "b001",
        "variant": "split",
        "primary_variant": "composed",
        "run_id": "hailo8_to_trt",
        "backend": "hailo8_to_trt",
        "runtime_ok": True,
        "total_latency_ms": 10.0,
        "task_quality_input_requests_by_variant": {
            "composed": _realistic_composed_request(
                "orin_nx_hailo8_01",
            ),
        },
    }
    normalized = normalize_benchmark_row(
        row, model_id="resnet50", source_path=tmp_path / "result.json",
    )
    identity = normalized["quality_request_identities_by_variant"]["composed"]
    assert identity["identity_valid"] is True
    assert identity["setup_id"] == "orin_nx_hailo8_01"
    assert identity["direction"] == "hailo8_to_tensorrt"
    assert identity["producer_backend"] == "hailo8"
    assert identity["comparison_backend"] == "hailo8"
    assert identity["stage"] == "classification_logits"
    assert identity["endpoint_contract_complete"] is True
    assert identity["output_endpoint_attestation"]["status"] == "passed"
    assert identity["runtime_precision_identity"] == "float32_layout_fp16"
    assert _identity_resolution_errors(normalized) == []
    exact = _exact_identity(normalized)
    assert exact[2] == "hailo8_to_tensorrt"
    assert exact[3] == "float32_layout_fp16"
    assert exact[4] == "orin_nx_hailo8_01"
    assert exact[5] == "hailo8"
    assert exact[6].startswith("classification:classification_logits:")


def test_selected_variant_setup_prevents_cross_variant_dedupe_collapse(
    tmp_path: Path,
) -> None:
    def source(composed_setup: str, name: str) -> dict:
        return normalize_benchmark_row(
            {
                "case_id": "b001",
                "variant": "split",
                "primary_variant": "composed",
                "run_id": "hailo8_to_trt",
                "backend": "hailo8_to_trt",
                "runtime_ok": True,
                "total_latency_ms": 10.0,
                "task_quality_input_requests_by_variant": {
                    "composed": _realistic_composed_request(composed_setup),
                    "full": {
                        **_realistic_composed_request(
                            "shared_full_setup", backend="tensorrt",
                        ),
                        "variant": "full",
                    },
                },
            },
            model_id="resnet50",
            source_path=tmp_path / name,
        )

    first = source("composed_setup_a", "a.json")
    second = source("composed_setup_c", "c.json")
    assert first["setup_id"] == "composed_setup_a"
    assert second["setup_id"] == "composed_setup_c"
    assert _row_identity_for_dedupe_v58f(first) != _row_identity_for_dedupe_v58f(second)


def test_explicit_embedded_variant_setup_conflict_is_fail_closed(
    tmp_path: Path,
) -> None:
    normalized = normalize_benchmark_row(
        {
            "case_id": "b001",
            "variant": "split",
            "primary_variant": "composed",
            "run_id": "hailo8_to_trt",
            "backend": "hailo8_to_trt",
            "setup_id": "explicit_other_setup",
            "runtime_ok": True,
            "total_latency_ms": 10.0,
            "task_quality_input_requests_by_variant": {
                "composed": _realistic_composed_request(
                    "orin_nx_hailo8_01",
                ),
            },
        },
        model_id="resnet50",
        source_path=tmp_path / "conflict.json",
    )
    identity = normalized["quality_request_identities_by_variant"]["composed"]
    assert identity["identity_valid"] is False
    assert any(
        "explicit_embedded_setup_id_conflict" in error
        for error in identity["identity_errors"]
    )
    assert normalized["quality_identity_valid"] is False


def test_performance_results_keep_screening_observations_separate_from_claims(
    tmp_path: Path,
) -> None:
    build_benchmarkset_scientific_report(
        tmp_path,
        [{
            "model_id": "m",
            "task": "classification",
            "case_id": "b001",
            "backend": "hailo8_to_tensorrt",
            "variant": "composed",
            "buildable": True,
            "runtime_executable": True,
            "contract_consistent": False,
            "total_latency_ms": 10.0,
            "pipeline_cycle_selected_ms": 8.0,
        }],
        plan={
            "model_id": "m",
            "model_suite": {
                "primary": [{"id": "m", "evaluation_role": "development"}],
                "reserve": [],
            },
        },
    )
    root = tmp_path / "scientific_report"
    with (root / "performance_results.csv").open(newline="", encoding="utf-8") as handle:
        compatibility_claims = list(csv.DictReader(handle))
    with (root / "performance_observations.csv").open(newline="", encoding="utf-8") as handle:
        observations = list(csv.DictReader(handle))
    with (root / "claim_eligible_performance.csv").open(newline="", encoding="utf-8") as handle:
        claims = list(csv.DictReader(handle))
    assert len(observations) == 1
    assert compatibility_claims == []
    assert claims == []


def test_partial_declared_universe_has_diagnostic_correlation_only() -> None:
    audit_ids = ["b001", "b002", "b003", "b004"]
    method_rows = [
        {
            "model_id": "m",
            "case_id": case_id,
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "generic",
            "method_id": "cut_bytes_only",
            "predicted_value": float(index),
            "prediction_available": True,
            "prediction_unit": "bytes",
        }
        for index, case_id in enumerate(audit_ids, start=1)
    ]
    predictions = {
        "m": {
            "_ranking_method_predictions": method_rows,
            "_prediction_freeze": {
                "valid": True,
                "ranking_predictions_valid": True,
                "candidate_universe_complete": True,
                "candidate_universe_valid": True,
                "candidate_universe_scope": "predeclared_audit_universe",
                "candidate_universe_selected_case_ids": audit_ids,
            },
        }
    }
    measured = [
        {
            "model_id": "m",
            "evaluation_role": "development",
            "case_id": case_id,
            "backend": "hailo8_to_tensorrt",
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "generic",
            "variant": "composed",
            "runtime_executable": True,
            "pipeline_cycle_selected_ms": float(index),
            "ranking_eligible": False,
            "task_quality_decision": "fail",
        }
        for index, case_id in enumerate(audit_ids[:3], start=1)
    ]
    details, _macro, _summary = _ranking_method_comparison(
        measured,
        predictions,
        {
            "model_suite": {
                "primary": [{"id": "m", "evaluation_role": "development"}],
                "reserve": [],
            },
        },
        {
            "methods": ["cut_bytes_only"],
            "k_values": [1, 3],
            "elite_q_values": [1],
            "primary_k": 3,
            "minimum_candidates_for_correlation": 3,
            "require_complete_candidate_universe": True,
            "require_frozen_predictions": True,
            "near_optimal_relative_epsilon": 0.01,
            "method_policy": {},
        },
    )
    row = details[0]
    assert row["candidate_universe_declared_complete"] is True
    assert row["candidate_measurement_coverage_complete"] is False
    assert row["diagnostic_paired_candidate_count"] == 3
    assert row["diagnostic_spearman_rho"] == 1.0
    assert row["spearman_rho"] is None
    assert row["hit_at_1"] is None
    assert row["regret_at_3"] is None
    assert _summary["best_method_id"] == ""
    assert _summary["development_diagnostic_leader_method_id"] == ""


def test_cross_runner_retains_screening_pairs_by_eligibility_tier(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    (reports / "native_validation").mkdir(parents=True)
    endpoint_hash = "a" * 64
    generic_rows: list[dict] = []
    native_rows: list[dict] = []
    validation_rows: list[dict] = []
    for index in range(1, 4):
        case_id = f"b{index:03d}"
        exact = {
            "task": "classification",
            "stage": "logits",
            "endpoint_contract_hash": endpoint_hash,
            "precision": "fp16",
            "setup_id": "orin_nx_hailo10_01",
            "comparison_backend": "hailo10",
        }
        generic_rows.append({
            "model_id": "m",
            "case_id": case_id,
            "direction": "hailo10_to_tensorrt",
            "backend": "hailo10_to_tensorrt",
            "variant": "split",
            "runner_regime": "generic",
            "cycle_ms": float(index * 10),
            "task_quality_decision": "pass",
            "contract_consistent": False,
            "ranking_eligible": False,
            "quality_request_identities_by_variant": {
                "composed": {
                    "identity_valid": True,
                    "setup_id": exact["setup_id"],
                    "setup_ids": [exact["setup_id"]],
                    "direction": "hailo10_to_tensorrt",
                    "producer_backend": "hailo10",
                    "comparison_backend": "hailo10",
                    "task": exact["task"],
                    "stage": exact["stage"],
                    "endpoint_contract_hash": endpoint_hash,
                    "endpoint_contract_complete": True,
                    "output_endpoint_attestation": {
                        "attested": True,
                        "status": "passed",
                        "endpoint_contract_hash": endpoint_hash,
                    },
                    "runtime_precision_identity": exact["precision"],
                },
            },
        })
        native_rows.append({
            **exact,
            "model": "m",
            "case": case_id,
            "backend": "hailo10h_to_trt",
            "ok": True,
            "fps_makespan": 1000.0 / float(index * 5),
            "output_endpoint_match": True,
            "comparison_stratum_explicit": True,
            "quality_evidence_verified": True,
            "performance_claim_eligible": False,
        })
        validation_rows.append({
            **exact,
            "model": "m",
            "case": case_id,
            "backend": "hailo10h_to_trt",
            "semantic_ok": True,
            "task_valid": True,
            "accuracy_gate_pass": True,
            "contract_consistent": False,
        })
    (reports / "native_producer_combined_summary.json").write_text(
        json.dumps({"rows": native_rows}), encoding="utf-8",
    )
    (
        reports / "native_validation" / "native_producer_validation_summary.json"
    ).write_text(json.dumps({"rows": validation_rows}), encoding="utf-8")
    result = compute_cross_runner_report(
        tmp_path, generic_rows, minimum_candidates=3,
    )
    assert result["technical_pair_count"] == 3
    assert result["quality_pair_count"] == 3
    assert result["claim_pair_count"] == 0
    group = result["groups"][0]
    assert group["status"] == "quality_screening_only"
    assert group["technical_spearman_rho"] == 1.0
    assert group["quality_spearman_rho"] == 1.0
    assert group["spearman_rho"] is None


def test_general_gate_status_alone_does_not_pass_cross_runner_quality(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    (reports / "native_validation").mkdir(parents=True)
    endpoint_hash = "b" * 64
    exact = {
        "task": "classification",
        "stage": "classification_logits",
        "endpoint_contract_hash": endpoint_hash,
        "precision": "float32_layout_fp16",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
    }
    generic_rows = []
    native_rows = []
    validation_rows = []
    for index in range(1, 4):
        case_id = f"b{index:03d}"
        generic_rows.append({
            **exact,
            "model_id": "m",
            "case_id": case_id,
            "direction": "hailo8_to_tensorrt",
            "variant": "split",
            "runner_regime": "generic",
            "cycle_ms": float(index),
            "gate_status": "pass",
        })
        native_rows.append({
            **exact,
            "model": "m",
            "case": case_id,
            "backend": "hailo8_to_tensorrt",
            "ok": True,
            "fps_makespan": 1000.0 / float(index),
            "output_endpoint_match": True,
            "comparison_stratum_explicit": True,
            "quality_evidence_verified": True,
        })
        validation_rows.append({
            **exact,
            "model": "m",
            "case": case_id,
            "backend": "hailo8_to_tensorrt",
            "semantic_ok": True,
            "task_valid": True,
            "accuracy_gate_pass": True,
        })
    (reports / "native_producer_combined_summary.json").write_text(
        json.dumps({"rows": native_rows}), encoding="utf-8",
    )
    (
        reports / "native_validation" / "native_producer_validation_summary.json"
    ).write_text(json.dumps({"rows": validation_rows}), encoding="utf-8")
    result = compute_cross_runner_report(
        tmp_path, generic_rows, minimum_candidates=3,
    )
    assert result["technical_pair_count"] == 3
    assert result["quality_pair_count"] == 0
    assert result["claim_pair_count"] == 0
    assert result["groups"][0]["status"] == "technical_diagnostic_only"


def test_cross_runner_preserves_reverse_direction(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    (reports / "native_validation").mkdir(parents=True)
    endpoint_hash = "c" * 64
    exact = {
        "task": "classification",
        "stage": "classification_logits",
        "endpoint_contract_hash": endpoint_hash,
        "precision": "fp16",
        "setup_id": "orin_nx_hailo10_01",
        "comparison_backend": "hailo10h",
    }
    generic = {
        **exact,
        "model_id": "m",
        "case_id": "b001",
        "backend": "tensorrt_to_hailo10",
        "variant": "split",
        "runner_regime": "generic",
        "cycle_ms": 10.0,
    }
    native = {
        **exact,
        "model": "m",
        "case": "b001",
        "backend": "tensorrt_to_hailo10h",
        "ok": True,
        "fps_makespan": 200.0,
        "output_endpoint_match": True,
        "comparison_stratum_explicit": True,
    }
    validation = {
        **exact,
        "model": "m",
        "case": "b001",
        "backend": "tensorrt_to_hailo10h",
    }
    (reports / "native_producer_combined_summary.json").write_text(
        json.dumps({"rows": [native]}), encoding="utf-8",
    )
    (
        reports / "native_validation" / "native_producer_validation_summary.json"
    ).write_text(json.dumps({"rows": [validation]}), encoding="utf-8")
    result = compute_cross_runner_report(
        tmp_path, [generic], minimum_candidates=1,
    )
    assert result["pair_count"] == 1
    assert result["pairs"][0]["direction"] == "tensorrt_to_hailo10h"


def test_scientific_status_cannot_pass_without_claim_rows() -> None:
    axes = _scientific_decision_axes(
        {
            "rows": [{
                "runtime_executable": True,
                "latency_ms": 1.0,
                "performance_claim_eligible": False,
                "energy_claim_eligible": False,
            }],
            "summary": {"ranking": {"status": "development_only"}},
            "ranking_method_comparison": [],
        },
        technical_execution_status="ok",
        central_quality_reporting={
            "technical_status": "ok",
            "quality_decision": "pass",
        },
    )
    assert axes["technical_execution_status"] == "ok"
    assert axes["quality_evaluation_status"] == "evaluated"
    assert axes["aggregate_quality_status"] == "pass"
    assert axes["development_analysis_status"] == "unavailable"
    assert axes["claim_readiness_status"] == "blocked"
    assert axes["claim_eligible_row_count"] == 0
    assert axes["scientific_status"] == "not_claim_ready"
    assert axes["scientific_pass"] is False


def test_hailo10_family_alias_is_canonical_hailo10h() -> None:
    assert canonical_direction("hailo10_to_tensorrt") == "hailo10h_to_tensorrt"
    assert canonical_direction("hailo10h_to_tensorrt") == "hailo10h_to_tensorrt"
    assert canonical_direction("tensorrt_to_hailo10") == "tensorrt_to_hailo10h"
