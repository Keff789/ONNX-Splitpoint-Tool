from __future__ import annotations

import hashlib

import pytest

from onnx_splitpoint_tool.workflow.scientific_reporting import (
    project_central_quality_status,
)


FULL_ONLY_SCHEMA = (
    "onnx-splitpoint/full-only-quality-acceptance-identity-contract"
)
STANDARD_SCHEMA = (
    "onnx-splitpoint/standard-setup-local-tensorrt-quality-"
    "acceptance-identity-contract"
)
IDENTITY_KEY_FIELDS = [
    "model_id",
    "source_run_id",
    "setup_id",
    "backend",
    "variant",
    "execution_role",
    "performance_claims_emitted",
]
MODELS = ["resnet50", "yolo26s", "yolov7_paper"]
SETUPS = [
    ("orin_nx_hailo8_01", "hailo8"),
    ("orin_nx_hailo10_01", "hailo10h"),
    ("orin_nx_deepx_m1_01", "deepx"),
]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quality_result(
    *,
    model_id: str,
    run_id: str,
    setup_id: str,
    backend: str,
    variant: str,
    execution_role: str,
    performance_claims_emitted: object,
) -> dict[str, object]:
    identity = "|".join(
        (model_id, run_id, setup_id, backend, variant, execution_role)
    )
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "model_id": model_id,
        "task": "classification" if model_id == "resnet50" else "detection",
        "case_id": "full" if variant == "full" else "selected",
        "variant": variant,
        "run_id": run_id,
        "source_run_id": run_id,
        "backend": backend,
        "setup_id": setup_id,
        "source_setup_id": setup_id,
        "execution_role": execution_role,
        "performance_claims_emitted": performance_claims_emitted,
        "evaluation_fingerprint": _digest("evaluation:" + identity),
        "source_request_sha256": "sha256:" + _digest("request:" + identity),
        "producer_identity_sha256": _digest("producer:" + identity),
        "producer_binding_eligible": True,
        "runtime_precision_identity": "fp16",
        "n": 500,
        "primary": {
            "metric": (
                "top1_accuracy"
                if model_id == "resnet50"
                else "coco_ap_50_95"
            ),
            "candidate": 0.8,
            "reference": 0.8,
            "delta": 0.0,
            "ci_low": -0.001,
            "ci_high": 0.001,
            "margin": 0.01,
            "decision": "pass",
            "bootstrap_repetitions_requested": 500,
            "bootstrap_repetitions": 500,
        },
        "guardrails": {
            ("top5_accuracy" if model_id == "resnet50" else "ap50"): {
                "metric": (
                    "top5_accuracy" if model_id == "resnet50" else "ap50"
                ),
                "decision": "pass",
            }
        },
    }


def _summary(
    results: list[dict[str, object]],
    contract: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "onnx-splitpoint/central-quality-summary",
        "status": "ok",
        "request_count": len(results),
        "completed_count": len(results),
        "technical_completed_count": len(results),
        "failed_count": 0,
        "merge": {"unmatched_result_count": 0},
        "quality_acceptance_identity_contract": contract,
        "results": results,
    }


def _contract(*, schema: str, scope: str) -> dict[str, object]:
    return {
        "schema": schema,
        "schema_version": 1,
        "execution_scope": scope,
        "identity_key_fields": list(IDENTITY_KEY_FIELDS),
        "model_ids": list(MODELS),
        "expected_identities": [
            {
                "id": f"tensorrt_at_{producer}_full",
                "source_run_id": "native_full_tensorrt",
                "setup_id": setup_id,
                "backend": "tensorrt",
                "variant": "full",
                "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            }
            for setup_id, producer in SETUPS
        ],
    }


def _standard_results() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    generic_rows = [
        ("deepx_m1_full", "orin_nx_deepx_m1_01", "deepx_m1", "full"),
        ("deepx_to_trt", "orin_nx_deepx_m1_01", "deepx_to_trt", "composed"),
        ("hailo10", "orin_nx_hailo10_01", "hailo10", "full"),
        ("hailo10h_to_trt", "orin_nx_hailo10_01", "hailo10h_to_trt", "composed"),
        ("hailo8", "orin_nx_hailo8_01", "hailo8", "full"),
        ("hailo8_to_trt", "orin_nx_hailo8_01", "hailo8_to_trt", "composed"),
        ("ort_tensorrt", "orin_nx_deepx_m1_01", "ort_tensorrt", "full"),
        ("ort_tensorrt", "orin_nx_deepx_m1_01", "ort_tensorrt", "composed"),
    ]
    for model_id in MODELS:
        for setup_id, _producer in SETUPS:
            results.append(_quality_result(
                model_id=model_id,
                run_id="native_full_tensorrt",
                setup_id=setup_id,
                backend="tensorrt",
                variant="full",
                execution_role="full_quality_only",
                performance_claims_emitted=False,
            ))
        for run_id, setup_id, backend, variant in generic_rows:
            results.append(_quality_result(
                model_id=model_id,
                run_id=run_id,
                setup_id=setup_id,
                backend=backend,
                variant=variant,
                execution_role="",
                performance_claims_emitted=(
                    False if variant == "composed" else None
                ),
            ))
    return results


def test_standard_contract_aggregates_nine_companions_and_keeps_generic_diagnostics() -> None:
    results = _standard_results()
    assert len(results) == 33

    projected = project_central_quality_status(_summary(
        results,
        _contract(
            schema=STANDARD_SCHEMA,
            scope="standard_quality_setup_local_tensorrt",
        ),
    ))

    assert projected["technical_status"] == "ok"
    assert projected["result_count"] == 33
    assert projected["aggregate_all_full_result_count"] == 21
    assert projected["aggregate_expected_full_result_count"] == 9
    assert projected["aggregate_full_result_count"] == 9
    assert projected["aggregate_usable_full_result_count"] == 9
    assert projected["aggregate_excluded_diagnostic_full_result_count"] == 12
    assert projected["aggregate_decision_counts"] == {"pass": 9}
    assert projected["aggregate_identity_contract_definition_errors"] == []
    assert projected["aggregate_missing_identity_count"] == 0
    assert projected["aggregate_duplicate_identity_count"] == 0
    assert projected["aggregate_unexpected_identity_count"] == 0
    assert projected["aggregate_identity_contract_issue_count"] == 0
    assert projected["aggregate_identity_contract_complete"] is True
    assert projected["quality_decision"] == "pass"
    assert projected["scientific_pass"] is True


@pytest.mark.parametrize(
    ("schema", "scope"),
    [
        (FULL_ONLY_SCHEMA, "full_only"),
        (STANDARD_SCHEMA, "standard_quality_setup_local_tensorrt"),
    ],
)
def test_projection_accepts_exactly_the_two_known_schema_scope_pairs(
    schema: str,
    scope: str,
) -> None:
    contract = _contract(schema=schema, scope=scope)
    contract["model_ids"] = ["resnet50"]
    contract["expected_identities"] = contract["expected_identities"][:1]
    result = _quality_result(
        model_id="resnet50",
        run_id="native_full_tensorrt",
        setup_id="orin_nx_hailo8_01",
        backend="tensorrt",
        variant="full",
        execution_role="full_quality_only",
        performance_claims_emitted=False,
    )

    projected = project_central_quality_status(_summary([result], contract))

    assert projected["aggregate_identity_contract_complete"] is True
    assert projected["aggregate_identity_contract_issue_count"] == 0
    assert projected["quality_decision"] == "pass"


@pytest.mark.parametrize(
    ("schema", "scope", "version", "expected_error"),
    [
        (
            FULL_ONLY_SCHEMA,
            "standard_quality_setup_local_tensorrt",
            1,
            "quality_acceptance_identity_contract_scope_must_be_full_only",
        ),
        (
            STANDARD_SCHEMA,
            "full_only",
            1,
            (
                "quality_acceptance_identity_contract_scope_must_be_"
                "standard_quality_setup_local_tensorrt"
            ),
        ),
        (
            STANDARD_SCHEMA,
            "standard_quality_setup_local_tensorrt",
            2,
            "quality_acceptance_identity_contract_schema_version_mismatch",
        ),
        (
            STANDARD_SCHEMA,
            "standard_quality_setup_local_tensorrt",
            True,
            "quality_acceptance_identity_contract_schema_version_mismatch",
        ),
    ],
)
def test_standard_contract_pair_and_version_validation_remain_fail_closed(
    schema: str,
    scope: str,
    version: object,
    expected_error: str,
) -> None:
    contract = _contract(schema=schema, scope=scope)
    contract["schema_version"] = version
    contract["model_ids"] = ["resnet50"]
    contract["expected_identities"] = contract["expected_identities"][:1]
    result = _quality_result(
        model_id="resnet50",
        run_id="native_full_tensorrt",
        setup_id="orin_nx_hailo8_01",
        backend="tensorrt",
        variant="full",
        execution_role="full_quality_only",
        performance_claims_emitted=False,
    )

    projected = project_central_quality_status(_summary([result], contract))

    assert projected["aggregate_identity_contract_complete"] is False
    assert expected_error in projected[
        "aggregate_identity_contract_definition_errors"
    ]
    assert projected["quality_decision"] == "not_evaluated"


@pytest.mark.parametrize(
    ("schema", "scope"),
    [
        (FULL_ONLY_SCHEMA, "full_only"),
        (STANDARD_SCHEMA, "standard_quality_setup_local_tensorrt"),
    ],
)
def test_expected_producer_prefix_cannot_hide_malformed_role_as_diagnostic(
    schema: str,
    scope: str,
) -> None:
    contract = _contract(schema=schema, scope=scope)
    contract["model_ids"] = ["resnet50"]
    contract["expected_identities"] = contract["expected_identities"][:1]
    valid = _quality_result(
        model_id="resnet50",
        run_id="native_full_tensorrt",
        setup_id="orin_nx_hailo8_01",
        backend="tensorrt",
        variant="full",
        execution_role="full_quality_only",
        performance_claims_emitted=False,
    )
    malformed = _quality_result(
        model_id="resnet50",
        run_id="native_full_tensorrt",
        setup_id="orin_nx_hailo8_01",
        backend="tensorrt",
        variant="full",
        execution_role="",
        performance_claims_emitted=False,
    )

    projected = project_central_quality_status(
        _summary([valid, malformed], contract)
    )

    assert projected["aggregate_full_result_count"] == 1
    assert projected["aggregate_unexpected_identity_count"] == 1
    assert projected["aggregate_identity_contract_issue_count"] == 1
    assert projected["aggregate_identity_contract_complete"] is False
    assert projected["quality_decision"] == "not_evaluated"
