"""Authoritative Quality-FIRST policy for Native Split claim boundaries.

Native result rows and Central Quality result rows are evidence, not policy
authorities.  In particular, removing a self-declared Quality-FIRST marker
from both rows must never make a current run eligible for the historical
legacy join.  The immutable EvaluationRun manifest and the Native stage plan
decide whether the Quality-FIRST chain is mandatory.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW = (
    "v2.69f-hardware-smoke-native-energy-repair"
)
SELECTION_FINGERPRINT_WORKFLOW = (
    "v2.70d-targeted-console-smoke-native-authority-repair"
)

CANONICAL_NATIVE_SPLIT_BACKENDS = frozenset({
    "hailo8_to_trt",
    "hailo10h_to_trt",
    "deepx_to_trt",
})
NATIVE_SPLIT_BACKEND_ALIASES = {
    "hailo8_to_trt": "hailo8_to_trt",
    "hailo8_to_tensorrt": "hailo8_to_trt",
    "hailo10_to_trt": "hailo10h_to_trt",
    "hailo10h_to_trt": "hailo10h_to_trt",
    "hailo10_to_tensorrt": "hailo10h_to_trt",
    "hailo10h_to_tensorrt": "hailo10h_to_trt",
    "deepx_to_trt": "deepx_to_trt",
    "deepx_to_tensorrt": "deepx_to_trt",
    "deepx_m1_to_trt": "deepx_to_trt",
    "deepx_m1_to_tensorrt": "deepx_to_trt",
}
NATIVE_SPLIT_BACKENDS = frozenset(NATIVE_SPLIT_BACKEND_ALIASES)

_HISTORICAL_WORKFLOW_RE = re.compile(
    r"^(?:"
    r"v(?:5[0-9]|6[01])[-a-z0-9._+]*"
    r"|v2\.(?:6[0-8](?:[a-z])?|69[abcde])[-a-z0-9._+]*"
    r"|benchmark_suite"
    r")$",
    re.IGNORECASE,
)
_CURRENT_OR_LATER_WORKFLOW_RE = re.compile(
    r"^v2\.(?:(?:69f)|(?:69[g-z])|(?:[7-9][0-9]))[-a-z0-9._+]*$",
    re.IGNORECASE,
)


def canonical_native_split_backend(value: Any) -> str:
    """Return the one canonical backend token for every supported alias."""
    token = str(value or "").strip().lower().replace("-", "_")
    return NATIVE_SPLIT_BACKEND_ALIASES.get(token, token)


def is_native_split_backend(value: Any) -> bool:
    """Return whether *value* names a managed Native Split backend."""
    return canonical_native_split_backend(value) in CANONICAL_NATIVE_SPLIT_BACKENDS


def _strict_json_object(path: Path) -> tuple[dict[str, Any] | None, str]:
    duplicate = False

    def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicate = True
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    except Exception as exc:
        return None, f"{path.name}_unreadable:{type(exc).__name__}"
    if duplicate:
        return None, f"{path.name}_duplicate_json_key"
    if not isinstance(value, dict):
        return None, f"{path.name}_root_not_object"
    return value, "ok"


def _effective_manifest_workflow(manifest: Mapping[str, Any]) -> tuple[str, list[str]]:
    errors: list[str] = []
    sessions = [
        row for row in list(manifest.get("execution_sessions") or [])
        if isinstance(row, Mapping)
    ]
    session_workflow = str(
        sessions[-1].get("workflow_version") if sessions else ""
    ).strip()
    current_workflow = str(manifest.get("current_workflow_version") or "").strip()
    creation_workflow = str(manifest.get("workflow_version") or "").strip()
    if current_workflow and session_workflow and current_workflow != session_workflow:
        errors.append("manifest_current_and_session_workflow_mismatch")
    effective = current_workflow or session_workflow or creation_workflow
    if not effective:
        errors.append("manifest_workflow_version_missing")
    return effective, errors


def _workflow_mode(workflow_version: str) -> str:
    normalized = str(workflow_version or "").strip()
    if normalized == CURRENT_NATIVE_SPLIT_QUALITY_WORKFLOW:
        return "required"
    if _CURRENT_OR_LATER_WORKFLOW_RE.fullmatch(normalized):
        return "required"
    if _HISTORICAL_WORKFLOW_RE.fullmatch(normalized):
        return "legacy"
    return "invalid"


def _requires_explicit_selection_fingerprint(workflow_version: str) -> bool:
    """Return whether the Native stage must copy the canonical fingerprint.

    v2.70c already records the fingerprint in the immutable start snapshot,
    but its Native stage only copied the requested-selection snapshot SHA.
    Keep that archived run readable while requiring the explicit field from
    v2.70d onward.
    """
    normalized = str(workflow_version or "").strip().lower()
    if normalized == SELECTION_FINGERPRINT_WORKFLOW:
        return True
    match = re.match(r"^v2\.(\d+)([a-z]?)", normalized)
    if not match:
        return False
    minor = int(match.group(1))
    suffix = match.group(2)
    return minor > 70 or (minor == 70 and suffix >= "d")


def _sha256_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    return text if re.fullmatch(r"[0-9a-f]{64}", text) else ""


def resolve_native_split_quality_authority(
    *,
    run_manifest_path: Path | str,
    stage_path: Path | str,
) -> dict[str, Any]:
    """Resolve the run-wide Native Split Quality-FIRST requirement.

    Unknown or malformed context deliberately resolves to ``required`` with an
    invalid authority status.  Callers therefore fail closed for split rows;
    only an explicitly version-bound historical manifest can enter legacy
    compatibility mode.
    """
    manifest_path = Path(run_manifest_path).expanduser().resolve()
    native_stage_path = Path(stage_path).expanduser().resolve()
    errors: list[str] = []

    manifest, manifest_status = _strict_json_object(manifest_path)
    if manifest is None:
        errors.append(manifest_status)
        workflow_version = ""
        run_id = ""
        mode = "invalid"
    else:
        if manifest.get("schema") != "onnx-splitpoint/evaluation-run-manifest":
            errors.append("run_manifest_schema_invalid")
        workflow_version, workflow_errors = _effective_manifest_workflow(manifest)
        errors.extend(workflow_errors)
        run_id = str(manifest.get("run_id") or "").strip()
        if not run_id:
            errors.append("run_manifest_run_id_missing")
        mode = _workflow_mode(workflow_version)
        if mode == "invalid":
            errors.append("run_manifest_workflow_version_unknown")
    manifest_snapshot = (
        manifest.get("profile_start_snapshot")
        if isinstance(manifest, Mapping)
        and isinstance(manifest.get("profile_start_snapshot"), Mapping)
        else {}
    )
    profile_start_snapshot_sha256 = _sha256_token(
        manifest_snapshot.get("snapshot_sha256")
    )
    requested_selection = (
        manifest_snapshot.get("requested_selection")
        if isinstance(manifest_snapshot.get("requested_selection"), Mapping)
        else {}
    )
    resolved_selection = (
        manifest_snapshot.get("resolved_selection")
        if isinstance(manifest_snapshot.get("resolved_selection"), Mapping)
        else {}
    )
    requested_selection_sha = _sha256_token(
        requested_selection.get("selection_snapshot_sha256")
        or requested_selection.get("snapshot_sha256")
    )
    resolved_selection_sha = _sha256_token(
        resolved_selection.get("selection_snapshot_sha256")
        or resolved_selection.get("snapshot_sha256")
    )
    profile_selection_fingerprint = _sha256_token(
        manifest_snapshot.get("selection_fingerprint")
    )
    selection_identity_mode = (
        "canonical_selection_fingerprint"
        if profile_selection_fingerprint
        else "legacy_equal_selection_snapshots"
    )
    if mode == "required":
        if not profile_start_snapshot_sha256:
            errors.append("run_manifest_profile_snapshot_sha256_missing_or_invalid")
        if not requested_selection_sha or not resolved_selection_sha:
            errors.append("run_manifest_selection_snapshot_sha256_missing_or_invalid")
        elif (
            not profile_selection_fingerprint
            and requested_selection_sha != resolved_selection_sha
        ):
            errors.append("run_manifest_selection_snapshot_sha256_mismatch")
        if (
            _requires_explicit_selection_fingerprint(workflow_version)
            and not profile_selection_fingerprint
        ):
            errors.append("run_manifest_selection_fingerprint_missing_or_invalid")
    tool_version = str(
        (manifest or {}).get("current_tool_version")
        or (manifest or {}).get("tool_version")
        or ""
    ).strip() if isinstance(manifest, Mapping) else ""

    stage, stage_status = _strict_json_object(native_stage_path)
    stage_required: bool | None = None
    stage_applicable: bool | None = None
    stage_run_id = ""
    if stage is None:
        if mode == "required":
            errors.append(stage_status)
    else:
        if stage.get("schema") not in {
            "onnx-splitpoint/native-producer-stage",
            "onnx-splitpoint/native-producer-variant-stage",
        }:
            errors.append("native_stage_schema_invalid")
        stage_run_id = str(stage.get("run_id") or "").strip()
        if not stage_run_id:
            eval_run_dir = str(stage.get("eval_run_dir") or "").strip()
            stage_run_id = Path(eval_run_dir).name if eval_run_dir else ""
        if run_id and stage_run_id and stage_run_id != run_id:
            errors.append("native_stage_run_id_mismatch")
        stage_workflow = str(stage.get("workflow_version") or "").strip()
        if mode == "required" and not stage_workflow:
            errors.append("native_stage_workflow_version_missing")
        elif stage_workflow and workflow_version and stage_workflow != workflow_version:
            errors.append("native_stage_workflow_version_mismatch")
        manifest_snapshot_sha = profile_start_snapshot_sha256
        stage_snapshot_sha = _sha256_token(
            stage.get("profile_start_snapshot_sha256")
        )
        if mode == "required" and manifest_snapshot_sha:
            if not stage_snapshot_sha:
                errors.append("native_stage_profile_snapshot_sha256_missing")
            elif stage_snapshot_sha != manifest_snapshot_sha:
                errors.append("native_stage_profile_snapshot_sha256_mismatch")
        stage_selection_sha = _sha256_token(
            stage.get("profile_selection_snapshot_sha256")
        )
        if mode == "required" and requested_selection_sha:
            if not stage_selection_sha:
                errors.append("native_stage_selection_snapshot_sha256_missing")
            elif stage_selection_sha != requested_selection_sha:
                errors.append("native_stage_selection_snapshot_sha256_mismatch")
        stage_selection_fingerprint = _sha256_token(
            stage.get("profile_selection_fingerprint")
        )
        if mode == "required" and profile_selection_fingerprint:
            if not stage_selection_fingerprint:
                if _requires_explicit_selection_fingerprint(workflow_version):
                    errors.append("native_stage_selection_fingerprint_missing")
            elif stage_selection_fingerprint != profile_selection_fingerprint:
                errors.append("native_stage_selection_fingerprint_mismatch")
        split_plan = stage.get("native_split_quality_first")
        if isinstance(split_plan, Mapping):
            required_value = split_plan.get("required")
            if isinstance(required_value, bool):
                stage_required = required_value
            else:
                errors.append("native_stage_split_quality_required_not_boolean")
            # ``required`` is the workflow policy.  ``applicable`` records
            # whether the sealed Native matrix actually contains Split rows.
            # Full-only campaigns keep the policy enabled, but have nothing
            # to bind and therefore must not be reported as an invalid legacy
            # or missing-Quality run.
            applicable_value = split_plan.get("applicable", True)
            if isinstance(applicable_value, bool):
                stage_applicable = applicable_value
            else:
                errors.append(
                    "native_stage_split_quality_applicable_not_boolean"
                )
        elif mode == "required":
            errors.append("native_stage_split_quality_plan_missing")

    # v2.69f and every later workflow requires QF for every managed split row.
    # The stage plan is an independent cross-check, never a downgrade switch.
    if mode == "required" and stage_required is not True:
        errors.append("native_stage_split_quality_not_required")

    required = mode != "legacy"
    valid = bool(mode in {"required", "legacy"} and not errors)
    return {
        "schema": "onnx-splitpoint/native-split-quality-authority",
        "schema_version": 1,
        "mode": mode,
        "valid": valid,
        "native_split_quality_required": required,
        "workflow_version": workflow_version,
        "tool_version": tool_version,
        "run_id": run_id,
        "profile_start_snapshot_sha256": profile_start_snapshot_sha256,
        # Compatibility name retained as the requested-selection snapshot SHA.
        # Requested and resolved SHAs legitimately differ when follow_tool_config
        # resolves a newer central run-mode definition.
        "profile_selection_snapshot_sha256": requested_selection_sha,
        "profile_requested_selection_snapshot_sha256": requested_selection_sha,
        "profile_resolved_selection_snapshot_sha256": resolved_selection_sha,
        "profile_selection_fingerprint": profile_selection_fingerprint,
        "selection_identity_mode": selection_identity_mode,
        "stage_run_id": stage_run_id,
        "stage_required": stage_required,
        "stage_applicable": stage_applicable,
        "native_split_quality_applicable": stage_applicable is not False,
        "run_manifest": str(manifest_path),
        "native_stage": str(native_stage_path),
        "errors": list(dict.fromkeys(errors)),
    }


def native_split_quality_required_for_row(
    row: Mapping[str, Any], authority: Mapping[str, Any] | None,
) -> bool:
    """Return the authoritative requirement for one row.

    A missing authority is unknown, not legacy.  Only a valid ``legacy``
    authority may disable the current Quality-FIRST chain.
    """
    if not is_native_split_backend(row.get("backend")):
        return False
    if not isinstance(authority, Mapping):
        return True
    return not bool(
        authority.get("valid") is True
        and str(authority.get("mode") or "") == "legacy"
        and authority.get("native_split_quality_required") is False
    )


def apply_native_split_quality_authority(
    row: dict[str, Any], authority: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach authority diagnostics and force the split requirement if needed."""
    if not is_native_split_backend(row.get("backend")):
        return row
    required = native_split_quality_required_for_row(row, authority)
    context = dict(authority or {})
    row["native_split_quality_authority"] = context
    row["native_split_quality_authority_valid"] = bool(context.get("valid"))
    row["native_split_quality_authority_mode"] = str(
        context.get("mode") or "invalid"
    )
    row["native_split_quality_authority_workflow_version"] = str(
        context.get("workflow_version") or ""
    )
    row["native_split_quality_authority_errors"] = list(
        context.get("errors") or ["native_split_quality_authority_missing"]
    )
    if required:
        row["native_split_quality_required"] = True
        row["native_split_quality_binding_required"] = True
    elif context.get("valid") is True and context.get("mode") == "legacy":
        row["native_split_quality_legacy_status"] = "historical_diagnostic_only"
        row["execution_role"] = str(
            row.get("execution_role") or "legacy_manual_diagnostic"
        )
        row["performance_claims_emitted"] = False
        row["status"] = "historical_diagnostic_only"
        row["ranking_eligible"] = False
        row["performance_eligible"] = False
        row["energy_eligible"] = False
        row["thesis_valid"] = False
    return row
