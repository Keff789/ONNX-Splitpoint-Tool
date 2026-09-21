"""Shared fail-closed Native Energy quality-admission verifier.

The planner and the measurement executor must use the same predicate.  The
planner may validate an unsealed candidate before pair construction; the
executor additionally verifies the immutable row-local seal before it creates
any measurement output state.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping


SCHEMA = "onnx-splitpoint/native-energy-quality-admission"
SCHEMA_VERSION = 1

BOOLEAN_FIELDS = (
    "central_quality_evidence_verified",
    "precision_quality_binding_verified",
    "task_quality_observation_valid",
    "accuracy_gate_pass",
    "quality_provenance_complete",
    "quality_claim_result_verified",
    "diagnostic_only",
    "screening_comparable",
    "claim_comparable",
    "energy_claim_eligible",
)

REQUIRED_VERIFIED_AXES = (
    "central_quality_evidence_verified",
    "precision_quality_binding_verified",
    "task_quality_observation_valid",
    "quality_provenance_complete",
)

DIAGNOSTIC_FALSE_ROW_FIELDS = (
    "claim_ok",
    "claim_eligible",
    "eligible_for_energy_results_import",
    "eligible_for_scientific_claim",
    "energy_claim_eligible",
)


def energy_quality_reason_projection(admission: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only report axes; never authorize a measurement or a claim.

    A legacy false accuracy Boolean does not distinguish FAIL from
    INCONCLUSIVE. Retain that uncertainty instead of inventing model failures.
    """
    decision = str(admission.get("local_task_quality_decision") or "").lower()
    observation = admission.get("task_quality_observation_valid") is True
    if decision not in {"pass", "fail", "inconclusive", "reference", "reference_close", "accuracy_loss", "not_estimable"}:
        decision = ("pass" if admission.get("accuracy_gate_pass") is True else "not_pass") if observation else "unavailable"
    reason = str(admission.get("runtime_observation_reason") or "")
    campaign = list(admission.get("campaign_exclusion_reasons") or [])
    if reason == "incomplete_expected_native_matrix" and reason not in campaign:
        campaign.append(reason)
    binding = [field + "_not_verified" for field in (
        "central_quality_evidence_verified", "precision_quality_binding_verified",
        "task_quality_observation_valid", "quality_provenance_complete",
    ) if admission.get(field) is not True]
    local = [] if decision in {"pass", "reference", "reference_close", "accuracy_loss", "not_estimable"} else ["task_quality_" + decision]
    exclusion = str(admission.get("scientific_claim_exclusion_reason") or "")
    if exclusion and exclusion not in local:
        local.append(exclusion)
    other = [reason] if reason and reason not in campaign else []
    return {
        "local_task_quality_decision": decision,
        "local_task_quality_observation_valid": observation,
        "local_accuracy_gate_pass": admission.get("accuracy_gate_pass"),
        "energy_quality_exclusion_reasons": {
            "local_quality": local,
            "binding_or_completion": binding,
            "pairing_or_runtime_admission": other,
            "campaign": campaign,
        },
        "campaign_comparison_released": bool(
            not campaign and not exclusion and admission.get("energy_claim_eligible") is True
            and admission.get("claim_comparable") is True
            and admission.get("diagnostic_only") is False),
    }


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def strict_sha256(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token[7:]
    return token if re.fullmatch(r"[0-9a-f]{64}", token) else ""


def validate_energy_quality_admission_axes(
    admission: Mapping[str, Any],
    *,
    row: Mapping[str, Any] | None = None,
) -> None:
    """Validate the exact quality predicate shared by both execution layers."""

    admission_scope = str(
        admission.get("admission_scope") or "native_energy"
    ).strip()
    if admission_scope not in {
        "native_energy",
        "native_runtime_observation",
        "window_method_validation_probe",
    }:
        raise ValueError(
            "energy_quality_admission_scope_invalid"
        )
    for field in BOOLEAN_FIELDS:
        value = admission.get(field)
        if not isinstance(value, bool):
            raise ValueError(
                f"energy_quality_admission_{field}_not_boolean"
            )
        if row is not None and field in row and row.get(field) is not value:
            raise ValueError(
                f"energy_quality_admission_{field}_drift"
            )

    if admission_scope == "window_method_validation_probe":
        if (
            admission["diagnostic_only"] is not True
            or admission["accuracy_gate_pass"] is not False
            or admission["quality_claim_result_verified"] is not False
            or admission["screening_comparable"] is not False
            or admission["claim_comparable"] is not False
            or admission["energy_claim_eligible"] is not False
        ):
            raise ValueError(
                "window_probe_quality_admission_not_diagnostic_only"
            )
    elif admission_scope == "native_runtime_observation":
        if (
            admission["diagnostic_only"] is not True
            or admission["claim_comparable"] is not False
            or admission["energy_claim_eligible"] is not False
            or not str(
                admission.get("runtime_observation_reason")
                or ""
            ).strip()
        ):
            raise ValueError(
                "native_runtime_observation_not_diagnostic_only"
            )
        if (
            row is not None
            and row.get("semantic_claim_ok") is not False
        ):
            raise ValueError(
                "native_runtime_observation_semantic_claim_must_be_false"
            )
    else:
        for field in REQUIRED_VERIFIED_AXES:
            if admission.get(field) is not True:
                raise ValueError(
                    f"energy_quality_admission_{field}_not_verified"
                )

    if admission["accuracy_gate_pass"] is False:
        if (
            admission["diagnostic_only"] is not True
            or admission["claim_comparable"] is not False
            or admission["energy_claim_eligible"] is not False
        ):
            raise ValueError(
                "bound_negative_accuracy_observation_not_diagnostic_only"
            )
    elif admission["diagnostic_only"] is False and (
        admission["quality_claim_result_verified"] is not True
        or admission["claim_comparable"] is not True
        or admission["energy_claim_eligible"] is not True
    ):
        raise ValueError(
            "claim_energy_quality_result_not_verified"
        )

    if admission["diagnostic_only"] is True:
        if (
            admission["claim_comparable"] is not False
            or admission["energy_claim_eligible"] is not False
        ):
            raise ValueError(
                "diagnostic_energy_quality_claim_axes_must_be_false"
            )
    if admission["diagnostic_only"] is True and row is not None:
        for field in DIAGNOSTIC_FALSE_ROW_FIELDS:
            if row.get(field) is not False:
                raise ValueError(
                    f"diagnostic_energy_row_{field}_must_be_false"
                )


def verify_sealed_energy_quality_admission(
    row: Mapping[str, Any],
    *,
    required: bool,
) -> tuple[str, str]:
    """Verify a row-local quality admission and its immutable SHA-256 seal."""

    raw = row.get("energy_quality_admission")
    if not isinstance(raw, Mapping):
        if not required:
            return "", "legacy_energy_quality_admission_not_declared"
        raise ValueError("energy_quality_admission_missing")
    admission = dict(raw)
    declared = strict_sha256(admission.pop("admission_sha256", ""))
    if (
        not declared
        or canonical_json_sha256(admission) != declared
        or strict_sha256(
            row.get("energy_quality_admission_sha256")
        ) != declared
    ):
        raise ValueError(
            "energy_quality_admission_sha256_mismatch"
        )
    admission["admission_sha256"] = declared
    if (
        admission.get("schema") != SCHEMA
        or admission.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(
            "energy_quality_admission_schema_invalid"
        )

    for field in (
        "backend",
        "model",
        "case",
        "setup_id",
        "precision",
        "comparison_backend",
    ):
        admission_value = str(admission.get(field) or "").strip()
        row_value = str(row.get(field) or "").strip()
        if not admission_value or admission_value != row_value:
            raise ValueError(
                f"energy_quality_admission_{field}_drift"
            )

    admission_contract_sha = strict_sha256(
        admission.get("successful_command_contract_sha256")
    )
    row_contract_sha = strict_sha256(
        row.get("successful_command_contract_sha256")
    )
    if (
        not admission_contract_sha
        or not row_contract_sha
        or admission_contract_sha != row_contract_sha
    ):
        raise ValueError(
            "energy_quality_admission_command_contract_drift"
        )

    validate_energy_quality_admission_axes(
        admission,
        row=row,
    )
    return declared, "sealed_energy_quality_admission_verified"


__all__ = (
    "BOOLEAN_FIELDS",
    "DIAGNOSTIC_FALSE_ROW_FIELDS",
    "REQUIRED_VERIFIED_AXES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "canonical_json_sha256",
    "strict_sha256",
    "validate_energy_quality_admission_axes",
    "verify_sealed_energy_quality_admission",
)
