from __future__ import annotations

from dataclasses import replace
import math

from onnx_splitpoint_tool.quality_service import (
    QUALITY_ALGORITHM_VERSION,
    QualityEvaluationRequest,
    _evaluate_payload,
    prepare_evaluation,
)


def _classification_request(*, candidate_hits: int, reference_hits: int) -> QualityEvaluationRequest:
    image_count = 500
    reference = [
        {
            "image_id": index,
            "reference": {
                "top1_hit": index < reference_hits,
                "top5_hit": True,
            },
        }
        for index in range(image_count)
    ]
    candidate = [
        {
            "image_id": index,
            "candidate": {
                "top1_hit": index < candidate_hits,
                "top5_hit": True,
            },
        }
        for index in range(image_count)
    ]
    return QualityEvaluationRequest(
        reference_records=reference,
        candidate_records=candidate,
        annotations=[{"image_id": index, "label_id": index} for index in range(image_count)],
        metric_gate_config={
            "primary_metric": "top1_accuracy",
            "non_inferiority_margin": 0.01,
            "guardrails": {"top5_accuracy_margin": 0.01},
        },
        repetitions=13,
        seed=20260731,
        confidence_level=0.95,
        non_inferiority_margin=0.01,
        evaluator_factory="onnx_splitpoint_tool.quality_metrics:classification_quality_evaluator",
        reference_prediction_field="reference",
        candidate_prediction_field="candidate",
        algorithm_version=f"{QUALITY_ALGORITHM_VERSION}:classification-accuracy-v2",
    )


def _evaluate(request: QualityEvaluationRequest) -> dict:
    _, payload = prepare_evaluation(request)
    return _evaluate_payload(payload)


def test_exact_398_of_500_vs_403_of_500_boundary_runs_bootstrap() -> None:
    result = _evaluate(_classification_request(candidate_hits=398, reference_hits=403))
    primary = result["primary"]

    assert primary["candidate_hits"] == 398
    assert primary["reference_hits"] == 403
    assert primary["sample_count"] == 500
    assert primary["delta"] == -0.01
    assert primary["point_estimate_comparison_basis"] == "integer_hit_counts"
    assert primary["bootstrap_skipped_reason"] == ""
    assert primary["bootstrap_repetitions"] == 13
    assert primary["decision"] in {"pass", "inconclusive"}


def test_one_ulp_below_inclusive_boundary_still_runs_bootstrap() -> None:
    boundary = -0.01
    one_ulp_below = math.nextafter(boundary, -math.inf)
    request = QualityEvaluationRequest(
        reference_records=[{"image_id": "only", "value": 0.0}],
        candidate_records=[{"image_id": "only", "value": one_ulp_below}],
        annotations=[{"image_id": "only"}],
        metric_gate_config={"primary_metric": "synthetic_delta"},
        repetitions=7,
        seed=11,
        confidence_level=0.95,
        non_inferiority_margin=0.01,
        evaluator_factory="paired_mean",
        value_field="value",
    )

    primary = _evaluate(request)["primary"]

    assert primary["delta"] == one_ulp_below
    assert primary["point_estimate_comparison_basis"] == "float_inclusive_threshold_tolerance"
    assert primary["bootstrap_skipped_reason"] == ""
    assert primary["bootstrap_repetitions"] == 7
    assert primary["decision"] == "pass"


def test_397_of_500_vs_403_of_500_remains_an_immediate_fail() -> None:
    result = _evaluate(_classification_request(candidate_hits=397, reference_hits=403))
    primary = result["primary"]

    assert primary["delta"] == -0.012
    assert primary["decision"] == "fail"
    assert primary["bootstrap_skipped_reason"] == "point_estimate_below_non_inferiority_margin"
    assert primary["bootstrap_repetitions"] == 0


def test_quality_algorithm_version_invalidates_previous_cache_identity() -> None:
    request = _classification_request(candidate_hits=398, reference_hits=403)
    current_key, _ = prepare_evaluation(request)
    previous_key, _ = prepare_evaluation(
        replace(
            request,
            algorithm_version="management_paired_quality_v1:classification-accuracy-v1",
        )
    )

    assert QUALITY_ALGORITHM_VERSION == "management_paired_quality_v2"
    assert request.algorithm_version.endswith(":classification-accuracy-v2")
    assert current_key != previous_key
