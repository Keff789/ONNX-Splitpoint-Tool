"""Versioned, fail-closed protocol freezes for confirmatory evaluation.

The campaign archive in :mod:`onnx_splitpoint_tool.campaign` is a convenient
container for all evidence produced by a run.  A protocol freeze has a
different purpose: it records the small set of decisions that must be fixed
*before* confirmatory measurements are inspected.  Keeping that projection in
one module makes the rule explicit and avoids treating an arbitrary profile
file checksum as a scientific protocol.

Version 2.63 uses the term ``confirmatory_holdout``.  The historic ``holdout``
spelling remains an accepted alias so profiles written for 2.62 continue to
load, but newly created freeze artefacts always contain the unambiguous role.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import __version__ as TOOL_PACKAGE_VERSION
from .workflow.artifacts import now_iso, sha256_file, sha256_json, write_json

PROTOCOL_FREEZE_SCHEMA = "onnx-splitpoint/protocol-freeze"
PROTOCOL_FREEZE_VERIFICATION_SCHEMA = "onnx-splitpoint/protocol-freeze-verification"
PROTOCOL_FREEZE_SCHEMA_VERSION = 1

DEVELOPMENT_ROLE = "development"
CONFIRMATORY_HOLDOUT_ROLE = "confirmatory_holdout"
LEGACY_HOLDOUT_ROLE = "holdout"
PROTOCOL_MANIFEST_KINDS = ("candidate", "dag", "prediction", "policy", "energy")


def _canonical_role(value: Any) -> str:
    role = str(value or "").strip().lower().replace("-", "_")
    if role == LEGACY_HOLDOUT_ROLE:
        return CONFIRMATORY_HOLDOUT_ROLE
    return role


def normalize_evaluation_role(value: Any, *, strict: bool = False) -> str:
    """Return the v2.63 protocol role, accepting the 2.62 holdout alias."""
    role = _canonical_role(value)
    if role in {DEVELOPMENT_ROLE, CONFIRMATORY_HOLDOUT_ROLE}:
        return role
    if strict:
        raise ValueError(
            "evaluation_role must be development or confirmatory_holdout "
            "(legacy alias: holdout)"
        )
    return role


def is_confirmatory_holdout(value: Any) -> bool:
    return normalize_evaluation_role(value) == CONFIRMATORY_HOLDOUT_ROLE


def _load_mapping(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], Path | None]:
    if isinstance(value, Mapping):
        return dict(value), None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"structured artefact not found: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except Exception:
        try:
            import yaml

            payload = yaml.safe_load(text)
        except Exception as exc:  # pragma: no cover - PyYAML is a package dependency
            raise ValueError(f"could not parse structured artefact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"structured artefact must contain a mapping: {path}")
    return dict(payload), path


def _campaign(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = profile.get("campaign")
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _freeze_config(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = _campaign(profile).get("protocol_freeze")
    return dict(value or {}) if isinstance(value, Mapping) else {}


def _active_models(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    suite = profile.get("model_suite") if isinstance(profile.get("model_suite"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for tier in ("primary", "reserve"):
        for raw in list(suite.get(tier) or []):
            row = dict(raw) if isinstance(raw, Mapping) else {"id": str(raw)}
            if isinstance(raw, Mapping) and raw.get("enabled") is False:
                continue
            if str(row.get("id") or "").strip():
                row["suite_tier"] = tier
                rows.append(row)
    return rows


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _manifest_source(config: Mapping[str, Any], kind: str) -> Any:
    manifests = config.get("manifests") if isinstance(config.get("manifests"), Mapping) else {}
    value = manifests.get(kind)
    if value in (None, ""):
        value = manifests.get(f"{kind}_manifest")
    if value in (None, ""):
        value = config.get(f"{kind}_manifest")
    return value


def _manifest_record(value: Any, *, kind: str, base_dir: Path) -> dict[str, Any]:
    declared_sha = ""
    if isinstance(value, Mapping):
        declared_sha = str(value.get("sha256") or "")
        value = value.get("path") or value.get("artifact")
    if not value:
        return {"kind": kind, "path": "", "sha256": "", "exists": False}
    path = _resolve_path(value, base_dir)
    actual = sha256_file(path) if path.is_file() else ""
    return {
        "kind": kind,
        "path": str(path),
        "sha256": actual or "",
        "declared_sha256": declared_sha,
        "declared_sha256_matches": bool(not declared_sha or declared_sha == actual),
        "exists": path.is_file(),
    }


def _protocol_version(config: Mapping[str, Any], value: Any = None) -> str:
    return str(value if value not in (None, "") else config.get("version") or "").strip()


def _protocol_amendment(config: Mapping[str, Any], value: Any = None) -> str:
    return str(value if value not in (None, "") else config.get("amendment") or "initial").strip()


def build_protocol_projection(
    profile: Mapping[str, Any],
    *,
    profile_path: str | Path | None = None,
    version: str | None = None,
    amendment: str | int | None = None,
) -> dict[str, Any]:
    """Build the timestamp-free protocol projection used for sealing.

    Only identities and prospective protocol inputs are included.  Runtime
    results, measured scores and report timestamps are intentionally excluded.
    """
    campaign = _campaign(profile)
    config = _freeze_config(profile)
    base_dir = Path(profile_path).expanduser().resolve().parent if profile_path else Path.cwd()
    protocol_version = _protocol_version(config, version)
    protocol_amendment = _protocol_amendment(config, amendment)

    release_cfg = config.get("release_identity") if isinstance(config.get("release_identity"), Mapping) else {}
    evaluation_cfg = config.get("evaluation_identity") if isinstance(config.get("evaluation_identity"), Mapping) else {}
    release_identity = {
        "release_id": str(release_cfg.get("release_id") or config.get("release_id") or TOOL_PACKAGE_VERSION),
        "tool_version": str(release_cfg.get("tool_version") or TOOL_PACKAGE_VERSION),
    }
    campaign_id = str(campaign.get("id") or campaign.get("campaign_id") or profile.get("name") or "campaign")
    evaluation_identity = {
        "evaluation_id": str(evaluation_cfg.get("evaluation_id") or config.get("evaluation_id") or campaign_id),
        "campaign_id": campaign_id,
        "profile_name": str(profile.get("name") or campaign_id),
    }

    model_roles: list[dict[str, Any]] = []
    for row in _active_models(profile):
        role = normalize_evaluation_role(row.get("evaluation_role"))
        model_roles.append({
            "model_id": str(row.get("id") or ""),
            "model_sha256": str(row.get("model_sha256") or ""),
            "family_id": str(row.get("family_id") or ""),
            "evaluation_role": role,
            "generalization_scope": str(row.get("generalization_scope") or ""),
            "candidate_universe": dict(row.get("candidate_universe") or {})
            if isinstance(row.get("candidate_universe"), Mapping)
            else {},
        })
    model_roles.sort(key=lambda row: str(row.get("model_id") or ""))

    manifests = {
        kind: _manifest_record(_manifest_source(config, kind), kind=kind, base_dir=base_dir)
        for kind in PROTOCOL_MANIFEST_KINDS
    }
    return {
        "protocol_version": protocol_version,
        "amendment": protocol_amendment,
        "amendment_reason": str(config.get("amendment_reason") or ""),
        "supersedes_sha256": str(config.get("supersedes_sha256") or ""),
        "release_identity": release_identity,
        "evaluation_identity": evaluation_identity,
        "model_roles": model_roles,
        "manifests": manifests,
    }


def _projection_errors(projection: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not str(projection.get("protocol_version") or "").strip():
        errors.append({"field": "protocol_version", "reason": "missing"})
    amendment = str(projection.get("amendment") or "").strip()
    if not amendment:
        errors.append({"field": "amendment", "reason": "missing"})
    if amendment.lower() not in {"initial", "0", "none"}:
        if not str(projection.get("amendment_reason") or "").strip():
            errors.append({"field": "amendment_reason", "reason": "required_for_amendment"})
        if not str(projection.get("supersedes_sha256") or "").strip():
            errors.append({"field": "supersedes_sha256", "reason": "required_for_amendment"})

    roles = [dict(row) for row in list(projection.get("model_roles") or []) if isinstance(row, Mapping)]
    ids = [str(row.get("model_id") or "") for row in roles]
    if not roles:
        errors.append({"field": "model_roles", "reason": "empty"})
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        errors.append({"field": "model_roles", "reason": "missing_or_duplicate_model_id"})
    for row in roles:
        if normalize_evaluation_role(row.get("evaluation_role")) not in {
            DEVELOPMENT_ROLE,
            CONFIRMATORY_HOLDOUT_ROLE,
        }:
            errors.append({
                "field": f"model_roles.{row.get('model_id')}.evaluation_role",
                "reason": "invalid_or_missing",
            })

    manifests = projection.get("manifests") if isinstance(projection.get("manifests"), Mapping) else {}
    for kind in PROTOCOL_MANIFEST_KINDS:
        row = manifests.get(kind) if isinstance(manifests.get(kind), Mapping) else {}
        if not row.get("exists") or not str(row.get("sha256") or ""):
            errors.append({"field": f"manifests.{kind}", "reason": "missing_or_unreadable"})
        if row.get("declared_sha256_matches") is False:
            errors.append({"field": f"manifests.{kind}.sha256", "reason": "declared_hash_mismatch"})
    return errors


def create_protocol_freeze(
    *,
    profile: str | Path | Mapping[str, Any],
    output: str | Path,
    version: str | None = None,
    amendment: str | int | None = None,
    signer: str = "",
) -> Path:
    """Create a versioned protocol freeze and refuse incomplete projections."""
    payload, source_path = _load_mapping(profile)
    projection = build_protocol_projection(
        payload,
        profile_path=source_path,
        version=version,
        amendment=amendment,
    )
    errors = _projection_errors(projection)
    if errors:
        raise ValueError(f"protocol freeze refused: {errors}")
    projection_sha = sha256_json(projection)
    freeze = {
        "schema": PROTOCOL_FREEZE_SCHEMA,
        "schema_version": PROTOCOL_FREEZE_SCHEMA_VERSION,
        "created_at": now_iso(),
        "signer": str(signer or ""),
        "freeze_status": "frozen",
        "projection": projection,
        "projection_sha256": projection_sha,
    }
    freeze["manifest_payload_sha256"] = sha256_json({
        key: value for key, value in freeze.items() if key != "manifest_payload_sha256"
    })
    return write_json(output, freeze)


def _diff(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected:
                diffs.append({"field": child, "reason": "unexpected", "actual": actual.get(key)})
            elif key not in actual:
                diffs.append({"field": child, "reason": "missing", "expected": expected.get(key)})
            else:
                diffs.extend(_diff(expected.get(key), actual.get(key), child))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if expected != actual:
            diffs.append({"field": path, "reason": "changed", "expected": expected, "actual": actual})
        return diffs
    if expected != actual:
        diffs.append({"field": path, "reason": "changed", "expected": expected, "actual": actual})
    return diffs


def verify_protocol_freeze(
    freeze: str | Path | Mapping[str, Any],
    *,
    profile: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the seal and, when supplied, the current prospective protocol.

    A profile comparison is intentionally exact for the frozen projection.  A
    changed policy, prediction, candidate set, DAG or energy manifest therefore
    fails instead of being silently accepted as a resume-time profile edit.
    """
    try:
        payload, freeze_path = _load_mapping(freeze)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "schema": PROTOCOL_FREEZE_VERIFICATION_SCHEMA,
            "schema_version": 1,
            "created_at": now_iso(),
            "ok": False,
            "error": str(exc),
            "mismatches": [],
        }
    projection = payload.get("projection") if isinstance(payload.get("projection"), Mapping) else {}
    projection_sha = sha256_json(projection) if projection else ""
    projection_hash_ok = bool(projection and projection_sha == str(payload.get("projection_sha256") or ""))
    manifest_hash_ok = str(payload.get("manifest_payload_sha256") or "") == sha256_json({
        key: value for key, value in payload.items() if key != "manifest_payload_sha256"
    })
    errors = _projection_errors(projection) if projection else [{"field": "projection", "reason": "missing"}]
    mismatches: list[dict[str, Any]] = []
    current_projection_sha = ""
    if profile is not None:
        current, profile_path = _load_mapping(profile)
        current_projection = build_protocol_projection(
            current,
            profile_path=profile_path,
        )
        current_projection_sha = sha256_json(current_projection)
        mismatches = _diff(projection, current_projection)
    ok = bool(
        payload.get("schema") == PROTOCOL_FREEZE_SCHEMA
        and int(payload.get("schema_version") or 0) == PROTOCOL_FREEZE_SCHEMA_VERSION
        and payload.get("freeze_status") == "frozen"
        and projection_hash_ok
        and manifest_hash_ok
        and not errors
        and not mismatches
    )
    return {
        "schema": PROTOCOL_FREEZE_VERIFICATION_SCHEMA,
        "schema_version": 1,
        "created_at": now_iso(),
        "ok": ok,
        "freeze_path": str(freeze_path or ""),
        "protocol_version": projection.get("protocol_version", ""),
        "amendment": projection.get("amendment", ""),
        "projection_hash_ok": projection_hash_ok,
        "manifest_payload_hash_ok": manifest_hash_ok,
        "frozen_projection_sha256": str(payload.get("projection_sha256") or ""),
        "current_projection_sha256": current_projection_sha,
        "projection_errors": errors,
        "mismatches": mismatches,
        "manifest": payload,
    }


def configured_protocol_freeze_path(
    profile: Mapping[str, Any], *, profile_path: str | Path | None = None
) -> Path | None:
    config = _freeze_config(profile)
    value = config.get("artifact") or config.get("path")
    if not value:
        return None
    base = Path(profile_path).expanduser().resolve().parent if profile_path else Path.cwd()
    return _resolve_path(value, base)


def verify_configured_protocol_freeze(
    profile: Mapping[str, Any], *, profile_path: str | Path | None = None
) -> dict[str, Any]:
    """Verify a configured freeze; absence remains valid for legacy profiles."""
    path = configured_protocol_freeze_path(profile, profile_path=profile_path)
    required = bool(_campaign(profile).get("require_protocol_freeze"))
    if path is None:
        return {
            "schema": PROTOCOL_FREEZE_VERIFICATION_SCHEMA,
            "schema_version": 1,
            "created_at": now_iso(),
            "ok": not required,
            "configured": False,
            "required": required,
            "status": "missing_required" if required else "not_configured_legacy_compatible",
            "mismatches": [],
        }
    result = verify_protocol_freeze(path, profile=profile_path or profile)
    result.update({"configured": True, "required": required, "status": "verified" if result.get("ok") else "changed_or_invalid"})
    return result


__all__: Sequence[str] = (
    "CONFIRMATORY_HOLDOUT_ROLE",
    "DEVELOPMENT_ROLE",
    "LEGACY_HOLDOUT_ROLE",
    "PROTOCOL_FREEZE_SCHEMA",
    "PROTOCOL_MANIFEST_KINDS",
    "build_protocol_projection",
    "configured_protocol_freeze_path",
    "create_protocol_freeze",
    "is_confirmatory_holdout",
    "normalize_evaluation_role",
    "verify_configured_protocol_freeze",
    "verify_protocol_freeze",
)
