from __future__ import annotations

"""Runtime evidence ledger, intentionally separate from build evidence.

A compiled artifact proves only compilation. Runtime reuse additionally binds
the physical setup, firmware/driver/runtime stack, runner implementation and
input/endpoint contracts. Missing identity is fail-closed.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .build_evidence import (
    BuildEvidenceError,
    _atomic_write_json_exclusive,
    _digest,
    _json_clone,
    _lexical_absolute,
    _normalize_backend,
    _normalize_hw_arch,
    _read_regular_nofollow,
    _strict_json,
    _token,
    canonical_sha256,
)


RUNTIME_KEY_SCHEMA = "onnx-splitpoint/exact-runtime-key/v1"
RUNTIME_RECORD_SCHEMA = "onnx-splitpoint/runtime-evidence-record/v1"
RUNTIME_INDEX_SCHEMA = "onnx-splitpoint/runtime-evidence-ledger/v1"

RUNTIME_PASS = "RUNTIME_PASS"
RUNTIME_FAILED = "RUNTIME_FAILED"
TRANSIENT_INFRASTRUCTURE = "TRANSIENT_INFRASTRUCTURE"
ABORTED_UNKNOWN = "ABORTED_UNKNOWN"

RUNTIME_STATES = frozenset({
    RUNTIME_PASS,
    RUNTIME_FAILED,
    TRANSIENT_INFRASTRUCTURE,
    ABORTED_UNKNOWN,
})
RUNTIME_REUSABLE_STATES = frozenset({RUNTIME_PASS, RUNTIME_FAILED})


# Both ledgers use the same stable fail-closed error envelope. Keeping an
# alias (rather than a subclass) also preserves the exact reason when a shared
# canonical/path primitive rejects runtime identity.
RuntimeEvidenceError = BuildEvidenceError


def _need(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise RuntimeEvidenceError(code, detail)


@dataclass(frozen=True)
class RuntimeEvidenceDecision:
    status: str
    key_sha256: str
    reusable: bool
    state: str | None
    reason: str
    record: dict[str, Any] | None = None
    evidence_origin: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "key_sha256": self.key_sha256,
            "reusable": self.reusable,
            "state": self.state,
            "reason": self.reason,
            "record": self.record,
            "evidence_origin": self.evidence_origin,
        }


def canonical_runtime_key(
    *,
    artifact_sha256: str,
    backend: str,
    hw_arch: str,
    setup_id: str,
    runtime_version: str,
    driver_version: str,
    firmware_version: str,
    runner_entrypoint_sha256: str,
    input_binding_contract_sha256: str,
    boundary_endpoint_contract_sha256: str,
    preprocessing_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": RUNTIME_KEY_SCHEMA,
        "schema_version": 1,
        "artifact_sha256": _digest(
            artifact_sha256, field="artifact_sha256"
        ),
        "backend": _normalize_backend(backend),
        "hw_arch": _normalize_hw_arch(hw_arch),
        "setup_id": _token(setup_id, field="setup_id"),
        "runtime_version": _token(
            runtime_version, field="runtime_version"
        ),
        "driver_version": _token(driver_version, field="driver_version"),
        "firmware_version": _token(
            firmware_version, field="firmware_version"
        ),
        "runner_entrypoint_sha256": _digest(
            runner_entrypoint_sha256,
            field="runner_entrypoint_sha256",
        ),
        "input_binding_contract_sha256": _digest(
            input_binding_contract_sha256,
            field="input_binding_contract_sha256",
        ),
        "boundary_endpoint_contract_sha256": _digest(
            boundary_endpoint_contract_sha256,
            field="boundary_endpoint_contract_sha256",
        ),
        "preprocessing_contract_sha256": _digest(
            preprocessing_contract_sha256,
            field="preprocessing_contract_sha256",
        ),
    }


def validate_runtime_key(value: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "invalid_runtime_key")
    _need(value.get("schema") == RUNTIME_KEY_SCHEMA, "runtime_key_schema_mismatch")
    _need(value.get("schema_version") == 1, "runtime_key_schema_version_mismatch")
    canonical = canonical_runtime_key(
        artifact_sha256=value.get("artifact_sha256"),
        backend=value.get("backend"),
        hw_arch=value.get("hw_arch"),
        setup_id=value.get("setup_id"),
        runtime_version=value.get("runtime_version"),
        driver_version=value.get("driver_version"),
        firmware_version=value.get("firmware_version"),
        runner_entrypoint_sha256=value.get("runner_entrypoint_sha256"),
        input_binding_contract_sha256=value.get(
            "input_binding_contract_sha256"
        ),
        boundary_endpoint_contract_sha256=value.get(
            "boundary_endpoint_contract_sha256"
        ),
        preprocessing_contract_sha256=value.get(
            "preprocessing_contract_sha256"
        ),
    )
    _need(dict(value) == canonical, "noncanonical_runtime_key")
    return canonical


def runtime_key_sha256(value: Mapping[str, Any]) -> str:
    return canonical_sha256(validate_runtime_key(value))


def make_runtime_evidence_record(
    key: Mapping[str, Any],
    state: str,
    *,
    evidence_origin: Mapping[str, Any],
    reason_code: str = "",
) -> dict[str, Any]:
    key_body = validate_runtime_key(key)
    state_eff = str(state or "").strip().upper()
    _need(state_eff in RUNTIME_STATES, "invalid_runtime_state", state_eff)
    origin = _json_clone(dict(evidence_origin), field="evidence_origin")
    _need(isinstance(origin, dict) and origin, "missing_evidence_origin")
    body: dict[str, Any] = {
        "schema": RUNTIME_RECORD_SCHEMA,
        "schema_version": 1,
        "key": key_body,
        "key_sha256": canonical_sha256(key_body),
        "state": state_eff,
        "deterministic": state_eff in RUNTIME_REUSABLE_STATES,
        "reusable": state_eff in RUNTIME_REUSABLE_STATES,
        "reason_code": str(reason_code or "").strip(),
        "evidence_origin": origin,
    }
    body["record_sha256"] = canonical_sha256(body)
    return body


def validate_runtime_evidence_record(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "invalid_runtime_record")
    body = dict(value)
    expected_hash = _digest(body.pop("record_sha256", None), field="record_sha256")
    _need(body.get("schema") == RUNTIME_RECORD_SCHEMA, "runtime_record_schema_mismatch")
    _need(body.get("schema_version") == 1, "runtime_record_schema_version_mismatch")
    key = validate_runtime_key(body.get("key") or {})
    _need(
        _digest(body.get("key_sha256"), field="key_sha256")
        == canonical_sha256(key),
        "key_sha256_mismatch",
    )
    state = str(body.get("state") or "").strip().upper()
    _need(state in RUNTIME_STATES, "invalid_runtime_state", state)
    expected_reusable = state in RUNTIME_REUSABLE_STATES
    _need(
        body.get("deterministic") is expected_reusable
        and body.get("reusable") is expected_reusable,
        "runtime_record_reuse_flag_mismatch",
    )
    canonical = make_runtime_evidence_record(
        key,
        state,
        evidence_origin=body.get("evidence_origin") or {},
        reason_code=str(body.get("reason_code") or ""),
    )
    _need(canonical["record_sha256"] == expected_hash, "record_sha256_mismatch")
    _need(canonical == dict(value), "noncanonical_runtime_record")
    return canonical


def runtime_evidence_index(
    records: Sequence[Mapping[str, Any]] = (),
    *,
    source_label: str = "runtime-ledger",
) -> dict[str, Any]:
    validated = [validate_runtime_evidence_record(record) for record in records]
    validated.sort(
        key=lambda record: (
            record["key_sha256"],
            record["state"],
            record["record_sha256"],
        )
    )
    body: dict[str, Any] = {
        "schema": RUNTIME_INDEX_SCHEMA,
        "schema_version": 1,
        "claim_scope": "exact_runtime_identity_only",
        "source_label": _token(source_label, field="source_label"),
        "records": validated,
        "record_count": len(validated),
        "reusable_record_count": sum(
            1 for record in validated if record["reusable"] is True
        ),
        "build_evidence_included": False,
    }
    body["ledger_payload_sha256"] = canonical_sha256(body)
    return body


def validate_runtime_evidence_index(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "invalid_runtime_index")
    supplied = dict(value)
    expected_hash = _digest(
        supplied.pop("ledger_payload_sha256", None),
        field="ledger_payload_sha256",
    )
    _need(supplied.get("schema") == RUNTIME_INDEX_SCHEMA, "runtime_index_schema_mismatch")
    _need(supplied.get("schema_version") == 1, "runtime_index_schema_version_mismatch")
    _need(supplied.get("build_evidence_included") is False, "build_ledger_not_separate")
    records = supplied.get("records")
    _need(isinstance(records, list), "invalid_runtime_records")
    canonical = runtime_evidence_index(
        records,
        source_label=supplied.get("source_label"),
    )
    _need(canonical["ledger_payload_sha256"] == expected_hash, "ledger_payload_sha256_mismatch")
    _need(canonical == dict(value), "noncanonical_runtime_index")
    return canonical


def lookup_runtime_evidence(
    index_or_path: Mapping[str, Any] | str | Path,
    exact_key: Mapping[str, Any],
) -> RuntimeEvidenceDecision:
    index = (
        validate_runtime_evidence_index(index_or_path)
        if isinstance(index_or_path, Mapping)
        else load_runtime_evidence_index(index_or_path)
    )
    key = validate_runtime_key(exact_key)
    key_hash = canonical_sha256(key)
    matches = [
        record for record in index["records"]
        if record["key_sha256"] == key_hash and record["reusable"] is True
    ]
    if not matches:
        return RuntimeEvidenceDecision(
            status="MISS",
            key_sha256=key_hash,
            reusable=False,
            state=None,
            reason="exact_runtime_key_not_found_or_nonreusable",
        )
    states = {record["state"] for record in matches}
    if len(states) != 1:
        return RuntimeEvidenceDecision(
            status="CONFLICT",
            key_sha256=key_hash,
            reusable=False,
            state=None,
            reason="conflicting_exact_runtime_outcomes",
        )
    selected = sorted(matches, key=lambda row: row["record_sha256"])[0]
    return RuntimeEvidenceDecision(
        status="HIT",
        key_sha256=key_hash,
        reusable=True,
        state=selected["state"],
        reason="exact_runtime_outcome",
        record=selected,
        evidence_origin=dict(selected.get("evidence_origin") or {}),
    )


def write_runtime_evidence_index(
    output: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    validated = validate_runtime_evidence_index(payload)
    path = _lexical_absolute(output, label="runtime_output")
    _atomic_write_json_exclusive(path, validated)
    return path


def load_runtime_evidence_index(path: str | Path) -> dict[str, Any]:
    observed = _read_regular_nofollow(
        path,
        label="runtime_evidence_index",
        collect=True,
        size_limit=64 * 1024 * 1024,
    )
    payload = _strict_json(
        observed.data or b"", label="runtime_evidence_index"
    )
    _need(isinstance(payload, Mapping), "json_object_required", "runtime_evidence_index")
    return validate_runtime_evidence_index(payload)


__all__ = [
    "ABORTED_UNKNOWN",
    "RUNTIME_FAILED",
    "RUNTIME_INDEX_SCHEMA",
    "RUNTIME_KEY_SCHEMA",
    "RUNTIME_PASS",
    "TRANSIENT_INFRASTRUCTURE",
    "RuntimeEvidenceDecision",
    "RuntimeEvidenceError",
    "canonical_runtime_key",
    "load_runtime_evidence_index",
    "lookup_runtime_evidence",
    "make_runtime_evidence_record",
    "runtime_evidence_index",
    "runtime_key_sha256",
    "validate_runtime_evidence_index",
    "validate_runtime_evidence_record",
    "validate_runtime_key",
    "write_runtime_evidence_index",
]
