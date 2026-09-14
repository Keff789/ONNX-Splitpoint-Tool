"""Fail-closed diagnostics for native detection semantic drift.

The helpers in this module classify already captured observations.  They never
change a quality threshold and never turn a diagnostic observation into claim
evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "onnx-splitpoint/hailo10-yolo26-full-semantic-diagnostic"
SCHEMA_VERSION = 1
PROSPECTIVE_EXCLUSION_SCHEMA = (
    "onnx-splitpoint/prospective-detection-claim-exclusions"
)
PROSPECTIVE_EXCLUSION_VERSION = 1
SOURCE_RUN_ID = "resnet_yolo26s_yolo7_20260728_211947"
_SHA256_LENGTH = 64


class DetectionDiagnosticContractError(ValueError):
    """Raised when a diagnostic fixture is ambiguous or incomparable."""


def build_hailo10_yolo26_prospective_exclusion(
    *,
    setup_id: str,
    diagnostic: Mapping[str, Any],
    diagnostic_fixture_sha256: str,
) -> dict[str, Any]:
    """Seal one setup-local exclusion from a verified diagnostic result."""
    setup = str(setup_id or "").strip()
    if not setup:
        raise DetectionDiagnosticContractError("exclusion_setup_id_missing")
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("model_id") != "yolo26s"
        or diagnostic.get("comparable") is not True
        or diagnostic.get("threshold_crossing") is not True
        or diagnostic.get("classification")
        != "compiled_raw_head_numeric_threshold_crossing"
        or diagnostic.get("scientific_claim_eligible") is not False
    ):
        raise DetectionDiagnosticContractError(
            "exclusion_diagnostic_not_verified"
        )
    diagnostic_sha = _sha256(
        diagnostic.get("diagnostic_sha256"),
        field="diagnostic_sha256",
    )
    fixture_sha = _sha256(
        diagnostic_fixture_sha256,
        field="diagnostic_fixture_sha256",
    )
    entry = {
        "schema": PROSPECTIVE_EXCLUSION_SCHEMA,
        "schema_version": PROSPECTIVE_EXCLUSION_VERSION,
        "setup_id": setup,
        "backend": "hailo10h_to_trt",
        "comparison_backend": "hailo10h",
        "model_id": "yolo26s",
        "reason": (
            "hailo10_yolo26_compiled_raw_head_numeric_threshold_crossing"
        ),
        "diagnostic_sha256": diagnostic_sha,
        "diagnostic_fixture_sha256": fixture_sha,
        "diagnostic_classification": str(
            diagnostic["classification"]
        ),
        "confidence_threshold": 0.25,
        "prospective": True,
        "scientific_claim_eligible": False,
    }
    entry["entry_sha256"] = _canonical_sha256(entry)
    return entry


def build_hailo10_yolo26_exclusion_from_fixture(
    *,
    setup_id: str,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    """Analyze the exact fixture and bind both fixture and result hashes."""
    if not isinstance(fixture, Mapping):
        raise DetectionDiagnosticContractError("fixture_invalid")
    diagnostic = analyze_hailo10_yolo26_full_fixture(fixture)
    return build_hailo10_yolo26_prospective_exclusion(
        setup_id=setup_id,
        diagnostic=diagnostic,
        diagnostic_fixture_sha256=_canonical_sha256(dict(fixture)),
    )


def seal_prospective_detection_exclusion_set(
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal an unambiguous set of setup/model/backend exclusions."""
    verified = [
        verify_hailo10_yolo26_prospective_exclusion(entry)
        for entry in entries
    ]
    identities = [
        (
            row["setup_id"],
            row["backend"],
            row["model_id"],
        )
        for row in verified
    ]
    if not verified:
        raise DetectionDiagnosticContractError(
            "prospective_exclusion_set_empty"
        )
    if len(set(identities)) != len(identities):
        raise DetectionDiagnosticContractError(
            "prospective_exclusion_identity_duplicate"
        )
    payload = {
        "schema": PROSPECTIVE_EXCLUSION_SCHEMA,
        "schema_version": PROSPECTIVE_EXCLUSION_VERSION,
        "entries": sorted(
            verified,
            key=lambda row: (
                row["setup_id"], row["backend"], row["model_id"],
            ),
        ),
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def verify_hailo10_yolo26_prospective_exclusion(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DetectionDiagnosticContractError(
            "prospective_exclusion_invalid"
        )
    entry = dict(raw)
    declared = _sha256(
        entry.pop("entry_sha256", ""),
        field="entry_sha256",
    )
    expected_keys = {
        "schema", "schema_version", "setup_id", "backend",
        "comparison_backend", "model_id", "reason",
        "diagnostic_sha256", "diagnostic_classification",
        "diagnostic_fixture_sha256",
        "confidence_threshold", "prospective",
        "scientific_claim_eligible",
    }
    if (
        set(entry) != expected_keys
        or entry.get("schema") != PROSPECTIVE_EXCLUSION_SCHEMA
        or int(entry.get("schema_version") or 0)
        != PROSPECTIVE_EXCLUSION_VERSION
        or not str(entry.get("setup_id") or "").strip()
        or entry.get("backend") != "hailo10h_to_trt"
        or entry.get("comparison_backend") != "hailo10h"
        or entry.get("model_id") != "yolo26s"
        or entry.get("reason")
        != "hailo10_yolo26_compiled_raw_head_numeric_threshold_crossing"
        or entry.get("diagnostic_classification")
        != "compiled_raw_head_numeric_threshold_crossing"
        or float(entry.get("confidence_threshold") or 0.0) != 0.25
        or entry.get("prospective") is not True
        or entry.get("scientific_claim_eligible") is not False
        or _canonical_sha256(entry) != declared
    ):
        raise DetectionDiagnosticContractError(
            "prospective_exclusion_invalid"
        )
    _sha256(
        entry.get("diagnostic_sha256"),
        field="diagnostic_sha256",
    )
    _sha256(
        entry.get("diagnostic_fixture_sha256"),
        field="diagnostic_fixture_sha256",
    )
    return {**entry, "entry_sha256": declared}


def verify_prospective_detection_exclusion_set(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DetectionDiagnosticContractError(
            "prospective_exclusion_set_invalid"
        )
    payload = dict(raw)
    declared = _sha256(
        payload.pop("contract_sha256", ""),
        field="contract_sha256",
    )
    if (
        set(payload) != {"schema", "schema_version", "entries"}
        or payload.get("schema") != PROSPECTIVE_EXCLUSION_SCHEMA
        or int(payload.get("schema_version") or 0)
        != PROSPECTIVE_EXCLUSION_VERSION
        or _canonical_sha256(payload) != declared
    ):
        raise DetectionDiagnosticContractError(
            "prospective_exclusion_set_invalid"
        )
    rebuilt = seal_prospective_detection_exclusion_set(
        list(payload.get("entries") or [])
    )
    if rebuilt["contract_sha256"] != declared:
        raise DetectionDiagnosticContractError(
            "prospective_exclusion_set_invalid"
        )
    return rebuilt


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_score(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise DetectionDiagnosticContractError(f"{field}_invalid")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DetectionDiagnosticContractError(f"{field}_invalid") from exc
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise DetectionDiagnosticContractError(f"{field}_invalid")
    return score


def _sha256(value: Any, *, field: str) -> str:
    token = str(value or "").strip().lower()
    if (
        len(token) != _SHA256_LENGTH
        or any(char not in "0123456789abcdef" for char in token)
    ):
        raise DetectionDiagnosticContractError(f"{field}_invalid")
    return token


def _sha256_list(
    value: Any,
    *,
    field: str,
    expected_count: int,
) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != expected_count
    ):
        raise DetectionDiagnosticContractError(f"{field}_invalid")
    return tuple(
        _sha256(item, field=f"{field}_{index}")
        for index, item in enumerate(value)
    )


def _pair(
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = payload.get("reference")
    candidate = payload.get("candidate")
    if not isinstance(reference, Mapping) or not isinstance(
        candidate, Mapping
    ):
        raise DetectionDiagnosticContractError("observation_pair_missing")
    return dict(reference), dict(candidate)


def _require_equal_hash(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    key: str,
    reason: str,
) -> str:
    first = _sha256(reference.get(key), field=f"reference_{key}")
    second = _sha256(candidate.get(key), field=f"candidate_{key}")
    if first != second:
        raise DetectionDiagnosticContractError(reason)
    return first


def _grid_cell(value: Any, *, field: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise DetectionDiagnosticContractError(f"{field}_invalid")
    try:
        y, x = (int(value[0]), int(value[1]))
    except (TypeError, ValueError, OverflowError) as exc:
        raise DetectionDiagnosticContractError(f"{field}_invalid") from exc
    if y < 0 or x < 0:
        raise DetectionDiagnosticContractError(f"{field}_invalid")
    return y, x


def analyze_hailo10_yolo26_full_fixture(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and classify the frozen Hailo-10H YOLO26s observation.

    Layout, scale, head mapping, input, preprocessing and decoder identity are
    comparability contracts.  A mismatch in any of them fails closed instead of
    being reported as numeric drift.
    """
    if not isinstance(payload, Mapping):
        raise DetectionDiagnosticContractError("fixture_invalid")
    if payload.get("schema") != SCHEMA or int(
        payload.get("schema_version") or 0
    ) != SCHEMA_VERSION:
        raise DetectionDiagnosticContractError("fixture_schema_invalid")
    if str(payload.get("model_id") or "").strip().lower() != "yolo26s":
        raise DetectionDiagnosticContractError("fixture_model_invalid")
    if str(payload.get("source_run_id") or "").strip() != SOURCE_RUN_ID:
        raise DetectionDiagnosticContractError("fixture_source_run_invalid")
    threshold = _finite_score(
        payload.get("confidence_threshold"),
        field="confidence_threshold",
    )
    if threshold != 0.25:
        raise DetectionDiagnosticContractError(
            "fixed_confidence_threshold_changed"
        )

    reference, candidate = _pair(payload)
    if str(reference.get("backend") or "") != "hailo8":
        raise DetectionDiagnosticContractError("reference_backend_invalid")
    if str(candidate.get("backend") or "") != "hailo10h":
        raise DetectionDiagnosticContractError("candidate_backend_invalid")
    if str(reference.get("setup_id") or "") != "orin_nx_hailo8_01":
        raise DetectionDiagnosticContractError("reference_setup_invalid")
    if str(candidate.get("setup_id") or "") != "orin_nx_hailo10_01":
        raise DetectionDiagnosticContractError("candidate_setup_invalid")

    input_sha256 = _require_equal_hash(
        reference,
        candidate,
        key="input_sha256",
        reason="exact_input_mismatch",
    )
    input_image_sha256 = _require_equal_hash(
        reference,
        candidate,
        key="input_image_sha256",
        reason="exact_input_image_mismatch",
    )
    preprocess_sha256 = _require_equal_hash(
        reference,
        candidate,
        key="preprocess_contract_sha256",
        reason="preprocess_contract_mismatch",
    )
    layout_sha256 = _require_equal_hash(
        reference,
        candidate,
        key="raw_head_layout_sha256",
        reason="raw_head_layout_mismatch",
    )
    scale_sha256 = _require_equal_hash(
        reference,
        candidate,
        key="raw_head_scale_contract_sha256",
        reason="raw_head_scale_contract_mismatch",
    )
    head_mapping_sha256 = _require_equal_hash(
        reference,
        candidate,
        key="head_mapping_sha256",
        reason="head_mapping_mismatch",
    )
    decoder_sha256 = _require_equal_hash(
        reference,
        candidate,
        key="decoder_contract_sha256",
        reason="decoder_contract_mismatch",
    )
    reference_manifest_sha256 = _sha256(
        reference.get("evidence_manifest_sha256"),
        field="reference_evidence_manifest_sha256",
    )
    candidate_manifest_sha256 = _sha256(
        candidate.get("evidence_manifest_sha256"),
        field="candidate_evidence_manifest_sha256",
    )
    reference_raw_head_sha256 = _sha256_list(
        reference.get("raw_head_payload_sha256"),
        field="reference_raw_head_payload_sha256",
        expected_count=6,
    )
    candidate_raw_head_sha256 = _sha256_list(
        candidate.get("raw_head_payload_sha256"),
        field="candidate_raw_head_payload_sha256",
        expected_count=6,
    )
    if reference_raw_head_sha256 == candidate_raw_head_sha256:
        raise DetectionDiagnosticContractError(
            "raw_head_payload_difference_not_observed"
        )

    reference_head = str(reference.get("selected_head") or "")
    candidate_head = str(candidate.get("selected_head") or "")
    if not reference_head or reference_head != candidate_head:
        raise DetectionDiagnosticContractError("selected_head_mismatch")
    reference_cell = _grid_cell(
        reference.get("selected_grid_cell"),
        field="reference_selected_grid_cell",
    )
    candidate_cell = _grid_cell(
        candidate.get("selected_grid_cell"),
        field="candidate_selected_grid_cell",
    )
    if reference_cell != candidate_cell:
        raise DetectionDiagnosticContractError("selected_grid_cell_mismatch")

    reference_score = _finite_score(
        reference.get("score"), field="reference_score"
    )
    candidate_score = _finite_score(
        candidate.get("score"), field="candidate_score"
    )
    reference_pass = reference_score >= threshold
    candidate_pass = candidate_score >= threshold
    threshold_crossing = bool(reference_pass and not candidate_pass)
    classification = (
        "compiled_raw_head_numeric_threshold_crossing"
        if threshold_crossing
        else "no_reference_to_candidate_threshold_crossing"
    )
    identity = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_run_id": SOURCE_RUN_ID,
        "model_id": "yolo26s",
        "confidence_threshold": threshold,
        "input_sha256": input_sha256,
        "input_image_sha256": input_image_sha256,
        "preprocess_contract_sha256": preprocess_sha256,
        "raw_head_layout_sha256": layout_sha256,
        "raw_head_scale_contract_sha256": scale_sha256,
        "head_mapping_sha256": head_mapping_sha256,
        "decoder_contract_sha256": decoder_sha256,
        "reference_evidence_manifest_sha256": reference_manifest_sha256,
        "candidate_evidence_manifest_sha256": candidate_manifest_sha256,
        "reference_raw_head_payload_sha256": list(
            reference_raw_head_sha256
        ),
        "candidate_raw_head_payload_sha256": list(
            candidate_raw_head_sha256
        ),
        "selected_head": reference_head,
        "selected_grid_cell": list(reference_cell),
        "reference_score": reference_score,
        "candidate_score": candidate_score,
    }
    return {
        **identity,
        "diagnostic_sha256": _canonical_sha256(identity),
        "comparable": True,
        "reference_threshold_pass": reference_pass,
        "candidate_threshold_pass": candidate_pass,
        "threshold_crossing": threshold_crossing,
        "score_delta": candidate_score - reference_score,
        "raw_head_payload_identity_match": False,
        "classification": classification,
        "localized_layer": "compiled_raw_detection_heads",
        "root_cause_proven": False,
        "quantization_or_compiler_optimization": "plausible_hypothesis",
        "confidence_threshold_change_permitted": False,
        "scientific_claim_eligible": False,
    }


__all__ = (
    "DetectionDiagnosticContractError",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SOURCE_RUN_ID",
    "analyze_hailo10_yolo26_full_fixture",
)
