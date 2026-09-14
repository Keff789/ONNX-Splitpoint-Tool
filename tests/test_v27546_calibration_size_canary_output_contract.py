from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import onnx_splitpoint_tool.deepx.calibration_size_canary as canary
from onnx_splitpoint_tool.deepx.calibration_size_canary import (
    QualityEndpointEvidence,
    compare_calibration_arms,
)
from tests import test_v27541_deepx_calibration_size_canary as v27541


@pytest.fixture(autouse=True)
def _bind_synthetic_release_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v27541._bind_synthetic_fixture_authorities.__wrapped__(monkeypatch)


def _with_output_contract_artifact(
    endpoint: QualityEndpointEvidence,
    *,
    output_contract_sha256: str,
    observed_output_shape: list[int] | None = None,
) -> QualityEndpointEvidence:
    """Reseal one otherwise-identical, internally valid DeepX producer."""

    producer = json.loads(json.dumps(endpoint.producer))
    quality_endpoint = producer["quality_record_endpoint"]
    quality_endpoint["identity"][
        "output_contract_sha256"
    ] = output_contract_sha256
    if observed_output_shape is not None:
        quality_endpoint["identity"]["runtime_observed_outputs"][
            "outputs"
        ][0]["shape"] = observed_output_shape
    quality_endpoint["sha256"] = canary._canonical_sha256(
        quality_endpoint["identity"]
    )

    quality_contract = producer["quality_contract"]
    quality_contract["quality_record_endpoint"] = json.loads(
        json.dumps(quality_endpoint)
    )
    quality_contract[
        "quality_record_endpoint_contract_sha256"
    ] = quality_endpoint["sha256"]
    quality_contract.pop("quality_contract_sha256")
    quality_contract["quality_contract_sha256"] = canary._canonical_sha256(
        quality_contract
    )
    producer["quality_record_endpoint_contract_sha256"] = quality_endpoint[
        "sha256"
    ]
    producer["quality_contract_sha256"] = quality_contract[
        "quality_contract_sha256"
    ]
    producer.pop("producer_identity_sha256")
    producer["producer_identity_sha256"] = canary._canonical_sha256(producer)

    request = json.loads(json.dumps(endpoint.request))
    candidate = json.loads(json.dumps(endpoint.candidate))
    for owner in (request, candidate):
        owner["quality_contract"] = json.loads(json.dumps(quality_contract))
        owner["quality_contract_sha256"] = quality_contract[
            "quality_contract_sha256"
        ]
        owner[
            "quality_record_endpoint_contract_sha256"
        ] = quality_endpoint["sha256"]
    return replace(
        endpoint,
        producer=producer,
        request=request,
        candidate=candidate,
    )


def _with_runtime_endpoint_semantics(
    endpoint: QualityEndpointEvidence,
    *,
    mutation: str,
) -> QualityEndpointEvidence:
    producer = json.loads(json.dumps(endpoint.producer))
    runtime_endpoint = producer["endpoint"]
    identity = runtime_endpoint["identity"]
    if mutation == "stage":
        identity["stage"] = "classification_probabilities"
        identity["output_format"] = "classification_probabilities"
    else:
        identity["tensor_signature"]["tensors"][0]["shape"] = [999]
    runtime_endpoint["sha256"] = canary._canonical_sha256(identity)
    producer["endpoint_contract_hash"] = runtime_endpoint["sha256"]
    producer.pop("producer_identity_sha256")
    producer["producer_identity_sha256"] = canary._canonical_sha256(producer)

    request = json.loads(json.dumps(endpoint.request))
    request["endpoint_contract_hash"] = runtime_endpoint["sha256"]
    return replace(endpoint, producer=producer, request=request)


def test_b500_b1000_arm_local_output_contract_artifacts_compare(
    tmp_path: Path,
) -> None:
    baseline, candidate = v27541._pair(tmp_path)
    b500_output_contract = hashlib.sha256(
        b"b500-deepx-output-contract.json"
    ).hexdigest()
    b1000_output_contract = hashlib.sha256(
        b"b1000-deepx-output-contract.json"
    ).hexdigest()
    baseline_endpoint = _with_output_contract_artifact(
        baseline.deepx_endpoint,
        output_contract_sha256=b500_output_contract,
    )
    candidate_endpoint = _with_output_contract_artifact(
        candidate.deepx_endpoint,
        output_contract_sha256=b1000_output_contract,
    )
    baseline = replace(baseline, deepx_endpoint=baseline_endpoint)
    candidate = replace(candidate, deepx_endpoint=candidate_endpoint)

    assert b500_output_contract != b1000_output_contract
    assert (
        baseline_endpoint.producer["quality_record_endpoint"]["sha256"]
        != candidate_endpoint.producer["quality_record_endpoint"]["sha256"]
    )
    assert (
        baseline_endpoint.producer["quality_contract_sha256"]
        != candidate_endpoint.producer["quality_contract_sha256"]
    )
    assert compare_calibration_arms(
        baseline, candidate
    )["standard_plus_ready"] is True


@pytest.mark.parametrize(
    "mutation", ["stage", "tensor_signature", "runtime_observed_outputs"],
)
def test_output_contract_projection_keeps_runtime_endpoint_semantics_strict(
    tmp_path: Path,
    mutation: str,
) -> None:
    baseline, candidate = v27541._pair(tmp_path)
    baseline = replace(
        baseline,
        deepx_endpoint=_with_output_contract_artifact(
            baseline.deepx_endpoint,
            output_contract_sha256=hashlib.sha256(b"b500-contract").hexdigest(),
        ),
    )
    changed_endpoint = _with_output_contract_artifact(
        candidate.deepx_endpoint,
        output_contract_sha256=hashlib.sha256(b"b1000-contract").hexdigest(),
        observed_output_shape=(
            [1, 999] if mutation == "runtime_observed_outputs" else None
        ),
    )
    if mutation != "runtime_observed_outputs":
        changed_endpoint = _with_runtime_endpoint_semantics(
            changed_endpoint, mutation=mutation,
        )

    with pytest.raises(ValueError, match="quality execution contracts differ"):
        compare_calibration_arms(
            baseline,
            replace(candidate, deepx_endpoint=changed_endpoint),
        )


def test_output_contract_projection_does_not_mask_invalid_arm_seal(
    tmp_path: Path,
) -> None:
    baseline, candidate = v27541._pair(tmp_path)
    producer = json.loads(json.dumps(candidate.deepx_endpoint.producer))
    producer["quality_record_endpoint"]["identity"][
        "output_contract_sha256"
    ] = hashlib.sha256(b"unsealed-output-contract").hexdigest()
    changed_endpoint = replace(candidate.deepx_endpoint, producer=producer)

    with pytest.raises(
        ValueError,
        match="producer execution contract is invalid|component hash mismatch",
    ):
        compare_calibration_arms(
            baseline,
            replace(candidate, deepx_endpoint=changed_endpoint),
        )
