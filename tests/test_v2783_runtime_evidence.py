from __future__ import annotations

import copy

import pytest

from onnx_splitpoint_tool.runtime_evidence import (
    ABORTED_UNKNOWN,
    RUNTIME_FAILED,
    RUNTIME_PASS,
    TRANSIENT_INFRASTRUCTURE,
    RuntimeEvidenceError,
    canonical_runtime_key,
    lookup_runtime_evidence,
    make_runtime_evidence_record,
    runtime_evidence_index,
    runtime_key_sha256,
)


def _runtime_kwargs() -> dict[str, str]:
    return {
        "artifact_sha256": "1" * 64,
        "backend": "hailo_runtime",
        "hw_arch": "hailo8",
        "setup_id": "orin_nx_hailo8_01",
        "runtime_version": "hailort:4.23.0",
        "driver_version": "hailo-pci:4.23.0",
        "firmware_version": "4.23.0",
        "runner_entrypoint_sha256": "2" * 64,
        "input_binding_contract_sha256": "3" * 64,
        "boundary_endpoint_contract_sha256": "4" * 64,
        "preprocessing_contract_sha256": "5" * 64,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_sha256", "a" * 64),
        ("backend", "deepx_runtime"),
        ("hw_arch", "hailo10h"),
        ("setup_id", "other_setup"),
        ("runtime_version", "hailort:4.24.0"),
        ("driver_version", "hailo-pci:4.24.0"),
        ("firmware_version", "4.24.0"),
        ("runner_entrypoint_sha256", "a" * 64),
        ("input_binding_contract_sha256", "a" * 64),
        ("boundary_endpoint_contract_sha256", "a" * 64),
        ("preprocessing_contract_sha256", "a" * 64),
    ],
)
def test_v2783_runtime_identity_axes_are_exact(field: str, value: str) -> None:
    baseline_args = _runtime_kwargs()
    baseline = runtime_key_sha256(canonical_runtime_key(**baseline_args))
    changed_args = copy.deepcopy(baseline_args)
    changed_args[field] = value
    changed = runtime_key_sha256(canonical_runtime_key(**changed_args))
    assert changed != baseline


def test_v2783_runtime_reuse_is_separate_and_exact() -> None:
    key = canonical_runtime_key(**_runtime_kwargs())
    record = make_runtime_evidence_record(
        key,
        RUNTIME_PASS,
        evidence_origin={"result": "runtime-result.json"},
    )
    index = runtime_evidence_index([record], source_label="b5-runtime")
    assert index["build_evidence_included"] is False
    decision = lookup_runtime_evidence(index, key)
    assert decision.status == "HIT"
    assert decision.state == RUNTIME_PASS

    changed = _runtime_kwargs()
    changed["firmware_version"] = "4.24.0"
    assert lookup_runtime_evidence(
        index, canonical_runtime_key(**changed)
    ).status == "MISS"


@pytest.mark.parametrize(
    "state", [TRANSIENT_INFRASTRUCTURE, ABORTED_UNKNOWN]
)
def test_v2783_transient_runtime_states_are_never_reusable(state: str) -> None:
    key = canonical_runtime_key(**_runtime_kwargs())
    record = make_runtime_evidence_record(
        key,
        state,
        evidence_origin={"result": "runtime-result.json"},
    )
    index = runtime_evidence_index([record])
    assert record["reusable"] is False
    assert lookup_runtime_evidence(index, key).status == "MISS"


def test_v2783_conflicting_runtime_pass_and_fail_are_not_reused() -> None:
    key = canonical_runtime_key(**_runtime_kwargs())
    records = [
        make_runtime_evidence_record(
            key,
            RUNTIME_PASS,
            evidence_origin={"result": "pass.json"},
        ),
        make_runtime_evidence_record(
            key,
            RUNTIME_FAILED,
            evidence_origin={"result": "fail.json"},
        ),
    ]
    decision = lookup_runtime_evidence(runtime_evidence_index(records), key)
    assert decision.status == "CONFLICT"
    assert decision.reusable is False


def test_v2783_missing_runtime_stack_identity_is_fail_closed() -> None:
    values = _runtime_kwargs()
    values["firmware_version"] = "unknown"
    with pytest.raises(RuntimeEvidenceError, match="incomplete_identity"):
        canonical_runtime_key(**values)


def test_v2783_build_state_cannot_enter_runtime_ledger() -> None:
    key = canonical_runtime_key(**_runtime_kwargs())
    with pytest.raises(RuntimeEvidenceError, match="invalid_runtime_state"):
        make_runtime_evidence_record(
            key,
            "ARTIFACT_PASS",
            evidence_origin={"result": "build.json"},
        )
