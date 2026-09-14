from __future__ import annotations

import copy

import pytest

from onnx_splitpoint_tool.trt_quality_chain import (
    TensorRTQualityChainError,
    _producer_flat_identity,
    producer_set_from_central_quality_summary,
)
from tests.test_v269d_trt_central_quality_producer import _producer, _seal_producer


def _strict_producer(model: str = "resnet50") -> dict:
    producer = copy.deepcopy(_producer())
    producer.update({
        "eval_run_id": "eval-20260721",
        "setup_id": "orin_nx_hailo8_01",
        "model_id": model,
        "source_run_id": "native_full_tensorrt",
        "case_id": "full",
        "execution_role": "full_quality_only",
        "backend": "native_tensorrt",
        "variant": "full",
        "performance_claims_emitted": False,
    })
    return _seal_producer(producer)


def _result(producer: dict, *, request_sha: str = "a" * 64) -> dict:
    flat = {
        **_producer_flat_identity(producer),
        "source_request_sha256": request_sha,
    }
    nested = {
        "schema": "onnx-splitpoint/central-quality-request-identity",
        "schema_version": 4,
        "identity_valid": True,
        "producer_identity_validated": True,
        "producer_binding_eligible": True,
        "producer_identity": copy.deepcopy(producer),
        **flat,
    }
    return {
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "passed",
        "decision": "passed",
        "producer_binding_eligible": True,
        "producer_identity": copy.deepcopy(producer),
        "request_identity": nested,
        **flat,
    }


def _summary(results: list[dict]) -> dict:
    return {
        "schema": "onnx-splitpoint/central-quality-summary",
        "schema_version": 1,
        "merge": {"summary_only_native_full_quality_conflict_count": 0},
        "results": results,
    }


def _build(summary: dict) -> dict:
    return producer_set_from_central_quality_summary(
        summary,
        eval_run_id="eval-20260721",
        setup_id="orin_nx_hailo8_01",
        model_ids=["resnet50"],
    )


def test_builds_exact_multi_model_set_from_completed_central_result() -> None:
    producer = _strict_producer()
    payload = _build(_summary([_result(producer)]))
    assert payload["schema"] == "onnx-splitpoint/tensorrt-quality-producer-set"
    assert payload["producers_by_model"] == {"resnet50": producer}


def test_byte_identical_request_and_producer_mirror_is_tolerated() -> None:
    producer = _strict_producer()
    row = _result(producer)
    payload = _build(_summary([row, copy.deepcopy(row)]))
    assert list(payload["producers_by_model"]) == ["resnet50"]


def test_accepts_logical_tensorrt_alias_around_signed_native_producer() -> None:
    producer = _strict_producer()
    row = _result(producer)
    row["backend"] = "tensorrt"
    row["request_identity"]["backend"] = "tensorrt"
    row["request_identity"]["producer_backend"] = "native_tensorrt"

    payload = _build(_summary([row]))

    assert payload["producers_by_model"] == {"resnet50": producer}


def test_rejects_non_tensorrt_backend_alias() -> None:
    row = _result(_strict_producer())
    row["backend"] = "cpu"
    row["request_identity"]["backend"] = "cpu"

    with pytest.raises(TensorRTQualityChainError, match="backend differs"):
        _build(_summary([row]))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.update(status="failed"),
        lambda row: row.update(producer_binding_eligible=False),
        lambda row: row["request_identity"].update(schema_version=3),
        lambda row: row["request_identity"].update(producer_binding_eligible=False),
        lambda row: row["request_identity"].update(producer_identity={}),
        lambda row: row.update(performance_claims_emitted=True),
    ],
)
def test_rejects_unbindable_or_conflicting_result(mutation) -> None:
    row = _result(_strict_producer())
    mutation(row)
    with pytest.raises(TensorRTQualityChainError):
        _build(_summary([row]))


def test_rejects_second_independent_request_even_for_same_producer() -> None:
    producer = _strict_producer()
    with pytest.raises(TensorRTQualityChainError, match="ambiguous"):
        _build(_summary([
            _result(producer, request_sha="a" * 64),
            _result(producer, request_sha="b" * 64),
        ]))


def test_rejects_producer_scope_tamper_instead_of_ignoring_result() -> None:
    row = _result(_strict_producer())
    tampered = copy.deepcopy(row["producer_identity"])
    tampered["setup_id"] = "different-host"
    row["producer_identity"] = _seal_producer(tampered)
    with pytest.raises(TensorRTQualityChainError):
        _build(_summary([row]))


def test_rejects_missing_model_and_stage_reported_conflict() -> None:
    with pytest.raises(TensorRTQualityChainError, match="no completed"):
        _build(_summary([]))
    summary = _summary([_result(_strict_producer())])
    summary["merge"]["summary_only_native_full_quality_conflict_count"] = 1
    with pytest.raises(TensorRTQualityChainError, match="reports Native Full"):
        _build(summary)
