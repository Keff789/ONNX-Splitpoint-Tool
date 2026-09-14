from __future__ import annotations

from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _ranking_method_comparison,
)


METHODS = ("cut_bytes_only", "weighted_score")
AUDIT_IDS = tuple(f"b{index:03d}" for index in range(1, 6))


def _frozen_prediction(
    model_id: str,
    *,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    method_rows = []
    for method_id in methods:
        predicted = (
            range(1, len(AUDIT_IDS) + 1)
            if method_id == "cut_bytes_only"
            else range(len(AUDIT_IDS), 0, -1)
        )
        for case_id, value in zip(AUDIT_IDS, predicted):
            method_rows.append({
                "model_id": model_id,
                "case_id": case_id,
                "direction": "hailo8_to_tensorrt",
                "runner_regime": "generic",
                "method_id": method_id,
                "predicted_value": float(value),
                "prediction_available": True,
                "prediction_unit": "bytes",
            })
    return {
        "_ranking_method_predictions": method_rows,
        "_prediction_freeze": {
            "valid": True,
            "ranking_predictions_valid": True,
            "candidate_universe_complete": True,
            "candidate_universe_valid": True,
            "candidate_universe_scope": "predeclared_audit_universe",
            "candidate_universe_selected_case_ids": list(AUDIT_IDS),
        },
    }


def _measurements(model_id: str, count: int) -> list[dict]:
    return [
        {
            "model_id": model_id,
            "evaluation_role": "development",
            "case_id": case_id,
            "backend": "hailo8_to_tensorrt",
            "direction": "hailo8_to_tensorrt",
            "runner_regime": "generic",
            "variant": "composed",
            "runtime_executable": True,
            "pipeline_cycle_selected_ms": float(index),
            "ranking_eligible": True,
            "task_quality_decision": "pass",
        }
        for index, case_id in enumerate(AUDIT_IDS[:count], start=1)
    ]


def _profile(*model_ids: str) -> dict:
    return {
        "model_suite": {
            "primary": [
                {"id": model_id, "evaluation_role": "development"}
                for model_id in model_ids
            ],
            "reserve": [],
        },
    }


def _policy() -> dict:
    return {
        "methods": list(METHODS),
        "k_values": [1, 3],
        "elite_q_values": [1],
        "primary_k": 3,
        "minimum_candidates_for_correlation": 3,
        "require_complete_candidate_universe": True,
        "require_frozen_predictions": True,
        "near_optimal_relative_epsilon": 0.01,
        "method_policy": {},
    }


def test_development_leader_uses_identical_qualified_actual_strata() -> None:
    qualified = "qualified_model"
    shortfall = "shortfall_model"
    predictions = {
        qualified: _frozen_prediction(qualified),
        shortfall: _frozen_prediction(shortfall),
    }
    details, macros, summary = _ranking_method_comparison(
        [
            *_measurements(qualified, 4),  # exactly 80% diagnostic coverage
            *_measurements(shortfall, 2),  # below min-n and 80% coverage
        ],
        predictions,
        _profile(qualified, shortfall),
        _policy(),
    )

    by_method = {row["method_id"]: row for row in macros}
    assert all(
        row["development_group_count"] == 2
        and row["development_diagnostic_eligible_group_count"] == 1
        for row in by_method.values()
    )
    assert (
        by_method["cut_bytes_only"][
            "development_diagnostic_macro_spearman_rho"
        ]
        == 1.0
    )
    assert summary["development_diagnostic_identical_actual_strata"] is True
    assert summary["development_diagnostic_leader_method_id"] == "cut_bytes_only"
    assert summary["development_diagnostic_leader_spearman_rho"] == 1.0
    assert summary["development_diagnostic_leader_min_coverage_fraction"] == 0.8
    assert len(summary["development_diagnostic_leader_actual_strata"]) == 1
    assert summary["development_diagnostic_leader_claim_eligible"] is False

    # A development/screening leader must never relax the confirmatory gate.
    assert summary["best_method_id"] == ""
    assert summary["best_method_basis"] == ""

    shortfall_rows = [
        row for row in details if row["model_id"] == shortfall
    ]
    assert all(
        "below_effective_minimum_candidate_count"
        in row["development_diagnostic_leader_group_exclusion_reasons"]
        and "diagnostic_coverage_below_80_percent"
        in row["development_diagnostic_leader_group_exclusion_reasons"]
        for row in shortfall_rows
    )


def test_development_leader_rejects_nonidentical_actual_strata() -> None:
    first = "first_model"
    second = "second_model"
    predictions = {
        first: _frozen_prediction(first, methods=("cut_bytes_only",)),
        second: _frozen_prediction(second, methods=("weighted_score",)),
    }
    _details, macros, summary = _ranking_method_comparison(
        [*_measurements(first, 4), *_measurements(second, 4)],
        predictions,
        _profile(first, second),
        _policy(),
    )

    assert all(
        row["development_diagnostic_eligible_group_count"] == 1
        for row in macros
    )
    assert summary["development_diagnostic_identical_actual_strata"] is False
    assert summary["development_diagnostic_leader_method_id"] == ""
    assert summary["best_method_id"] == ""
