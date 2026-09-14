"""Quality-result v3 uncertainty semantics and read-only legacy projections.

This module does not evaluate metrics, change margins, or touch model artifacts.
Historical result bytes and recorded overall decisions remain authoritative;
projections identify uncomputed uncertainty instead of exposing pseudo intervals.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

QUALITY_RESULT_CONTRACT_VERSION = 3
POINT_FAIL_REASON = "point_estimate_below_non_inferiority_margin"
IDENTITY_REASON = "candidate_reference_identical"
UNCERTAINTY_FIELDS = (
    "decision_basis", "gate_bound_value", "ci_computed", "uncertainty_status",
    "prediction_identity_verified", "legacy_uncertainty_projection",
)


def _finite(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def project_quality_component(component: Mapping[str, Any] | None) -> dict[str, Any]:
    """Copy one component; documented bootstrap skips never expose CI bounds.

    The source decision is retained explicitly when an old pseudo-bound passed a
    component despite a different component stopping the complete bootstrap.
    """
    out = dict(component or {})
    reason = str(out.get("bootstrap_skipped_reason") or "")
    if "ci_computed" in out:
        if out.get("ci_computed") is False:
            out["ci_low"] = out["ci_high"] = None
        return out
    try:
        repetitions = int(out.get("bootstrap_repetitions"))
    except (TypeError, ValueError):
        repetitions = None
    if reason or repetitions == 0:
        out["legacy_uncertainty_projection"] = True
        out["ci_low"] = out["ci_high"] = None
        out["ci_computed"] = False
        out["gate_bound_value"] = None
        if reason == POINT_FAIL_REASON:
            failed = str(out.get("decision") or out.get("status") or "") == "fail"
            out["uncertainty_status"] = "not_computed_fast_fail"
            out["decision_basis"] = POINT_FAIL_REASON if failed else "bootstrap_not_computed_other_component_point_fail"
            if failed:
                out["gate_bound_value"] = out.get("delta")
            elif str(out.get("decision") or out.get("status") or "") == "pass":
                out["source_component_decision"] = "pass"
                out["decision"] = out["status"] = "inconclusive"
        elif reason == IDENTITY_REASON:
            out["uncertainty_status"] = "deterministic_identity_legacy_not_reverified"
            out["decision_basis"] = "recorded_candidate_reference_identity"
            out["prediction_identity_verified"] = False
        else:
            out["uncertainty_status"] = "not_computed"
            out["decision_basis"] = reason or "bootstrap_not_computed"
    elif repetitions is not None and repetitions > 0:
        out["ci_computed"] = _finite(out.get("ci_low")) and _finite(out.get("ci_high"))
        out["uncertainty_status"] = "computed_bootstrap" if out["ci_computed"] else "bootstrap_bounds_incomplete"
        out["decision_basis"] = "paired_bootstrap_lower_bound"
        out["gate_bound_value"] = out.get("ci_low") if out["ci_computed"] else None
    return out


def project_quality_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Version-aware, in-memory view; never rewrite source/result fingerprints."""
    out = dict(result)
    version = out.get("quality_result_contract_version")
    out["quality_result_source_contract_version"] = version
    out["quality_result_projection_contract_version"] = QUALITY_RESULT_CONTRACT_VERSION
    out["legacy_quality_result"] = version != QUALITY_RESULT_CONTRACT_VERSION
    if isinstance(out.get("primary"), Mapping):
        out["primary"] = project_quality_component(out["primary"])
    if isinstance(out.get("guardrails"), Mapping):
        out["guardrails"] = {
            str(name): project_quality_component(component) if isinstance(component, Mapping) else component
            for name, component in out["guardrails"].items()
        }
    return out


def project_flat_quality_uncertainty(row: Mapping[str, Any], prefix: str = "task_quality_") -> dict[str, Any]:
    """Apply the same presentation semantics to an existing flattened row."""
    out = dict(row)
    names = ("delta", "ci_low", "ci_high", "margin", "decision", "status",
             "bootstrap_repetitions", "bootstrap_skipped_reason", *UNCERTAINTY_FIELDS)
    component = {name: out[prefix + name] for name in names if prefix + name in out}
    for name, value in project_quality_component(component).items():
        out[prefix + name] = value
    return out
