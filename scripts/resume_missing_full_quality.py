#!/usr/bin/env python3
"""Repair missing Phase-5 Full-quality endpoints or audit an exact retry."""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

from onnx_splitpoint_tool import __version__ as TOOL_VERSION
from onnx_splitpoint_tool.workflow.artifacts import read_json
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner,
    WORKFLOW_VERSION,
)


EXPECTED_FULL_QUALITY_RESULTS = 8
MIN_PRESERVED_RESULTS = 5
MAX_PRESERVED_RESULTS = 7

_ALLOWED_MODEL_REBUILD_STAGES = frozenset({
    "run_benchmarks",
    "validate_outputs",
    "hardware_smoke",
})
_ALLOWED_ROOT_REBUILD_STAGES = frozenset({
    "evaluate_quality",
    "aggregate_results",
    "generate_report",
})
_ORIGINAL_REPAIR_SCOPE = frozenset({
    ("yolo26m", "hailo8"),
    ("yolo11l", "hailo8"),
    ("yolo11l", "native_full_tensorrt"),
})

_PHASE5_MODELS = (
    "mobilenet_v3_large",
    "regnet_x_1_6gf",
    "yolo26m",
    "yolo11l",
)
_MODEL_STAGE_SURFACE = (
    "resolve_model",
    "check_validation_assets",
    "prepare_model",
    "analyze_model",
    "select_split_candidates",
    "prepare_full_baselines",
    "generate_benchmark_set",
    "build_backend_artifacts",
    "run_benchmarks",
    "validate_outputs",
    "hardware_smoke",
)
_ROOT_STAGE_SURFACE = (
    "resolve_profile",
    "campaign_preflight",
    "evaluate_quality",
    "aggregate_results",
    "run_native_producers",
    "generate_report",
)
_LEGACY_V27711_WORKFLOW_VERSION = (
    "v2.77.11-targeted-quality-resume-historical-artifact-closure"
)
_LEGACY_V27711_FAILED_TARGET_KEYS = frozenset({
    ("yolo11l", "run_benchmarks"),
    ("", "evaluate_quality"),
})
_SUPPORTED_POSTHOC_WORKFLOWS = {
    _LEGACY_V27711_WORKFLOW_VERSION: "2.77.11",
    WORKFLOW_VERSION: TOOL_VERSION,
}
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SESSION_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


def _options_from_manifest(run_dir: Path) -> WorkflowOptions:
    manifest = read_json(run_dir / "run_manifest.json", default={}) or {}
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != "onnx-splitpoint/evaluation-run-manifest"
    ):
        raise ValueError("target run_manifest.json is missing or invalid")
    archived = manifest.get("options")
    if not isinstance(archived, Mapping):
        raise ValueError("target run has no archived workflow options")
    allowed = {field.name for field in fields(WorkflowOptions)}
    unknown = sorted(set(archived) - allowed)
    if unknown:
        raise ValueError(
            "target run contains unsupported workflow options: "
            + ", ".join(unknown)
        )
    payload = dict(archived)
    payload.update({
        "out": str(run_dir.parent),
        "run_id": run_dir.name,
        "resume": True,
        "require_fresh_run": False,
        "resume_missing_full_quality_only": True,
        "rerun_generated_only": False,
        "force_stage": [],
        "stop_after": None,
        "only_model": None,
        "max_models": None,
        "dry_run": False,
        "skip_benchmarks": False,
        "no_remote": False,
        "execution_mode": "generate_and_run",
        "remote_resume": False,
        "remote_no_resume": True,
        "remote_reuse_bundle": False,
        "remote_no_reuse_bundle": True,
        "hailo_force_build": False,
        "force_build_confirmed_backends": (),
        "force_build_confirmation_source": "",
        "energy_enabled": False,
        "native_producer_enabled": False,
        "native_producer_energy_enabled": False,
    })
    profile = Path(str(payload.get("profile") or "")).expanduser()
    if not profile.is_file():
        raise ValueError(
            "archived source profile is no longer available: " + str(profile)
        )
    return WorkflowOptions(**payload)


def _load_quality_summary(run_dir: Path) -> Mapping[str, Any]:
    summary = read_json(
        run_dir / "quality_management" / "central_quality_summary.json",
        default={},
    ) or {}
    if not isinstance(summary, Mapping):
        raise RuntimeError("central Full-quality summary is missing or invalid")
    return summary


def _completed_quality_results(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_results = summary.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("central Full-quality result list is missing or invalid")
    results = [row for row in raw_results if isinstance(row, Mapping)]
    if len(results) != len(raw_results) or any(
        str(row.get("status") or "").strip().lower() != "completed"
        or str(row.get("technical_status") or "").strip().lower()
        != "completed"
        for row in results
    ):
        raise RuntimeError(
            "central Full-quality summary contains a non-completed result"
        )
    return results


def _source_run_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_")
    for prefix in ("benchmark_results_", "results_"):
        if token.startswith(prefix):
            token = token[len(prefix):]
    if token.endswith("_auto"):
        token = token[:-5]
    return token.replace("_to_tensorrt", "_to_trt")


def _result_repair_identity(result: Mapping[str, Any]) -> tuple[str, str]:
    nested = (
        result.get("request_identity")
        if isinstance(result.get("request_identity"), Mapping)
        else {}
    )
    model_tokens = {
        str(value or "").strip().lower()
        for value in (result.get("model_id"), nested.get("model_id"))
        if str(value or "").strip()
    }
    source_tokens = {
        _source_run_token(value)
        for value in (
            result.get("source_run_id"),
            result.get("run_id"),
            nested.get("source_run_id"),
            nested.get("run_id"),
        )
        if _source_run_token(value)
    }
    if len(model_tokens) != 1 or len(source_tokens) != 1:
        raise RuntimeError(
            "preserved Full-quality result identity is missing or conflicting"
        )
    return next(iter(model_tokens)), next(iter(source_tokens))


def _initial_resume_scope(run_dir: Path) -> dict[str, Any]:
    """Freeze the supported 5/8, 6/8, or 7/8 repair cardinality and models."""

    summary = _load_quality_summary(run_dir)
    contract = (
        summary.get("quality_acceptance_identity_contract")
        if isinstance(summary.get("quality_acceptance_identity_contract"), Mapping)
        else {}
    )
    postcondition = (
        contract.get("postcondition")
        if isinstance(contract, Mapping) else {}
    )
    try:
        expected_count = int(postcondition.get("expected_count") or -1)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "initial Full-quality expected result count is invalid"
        ) from exc
    if expected_count != EXPECTED_FULL_QUALITY_RESULTS:
        raise RuntimeError(
            "initial Full-quality identity contract does not expect exactly 8 results"
        )
    results = _completed_quality_results(summary)
    preserved_count = len(results)
    if not MIN_PRESERVED_RESULTS <= preserved_count <= MAX_PRESERVED_RESULTS:
        raise RuntimeError(
            "targeted Full-quality resume requires exactly 5, 6, or 7 "
            f"preserved results; observed {preserved_count}"
        )
    completed_scope = {
        identity
        for identity in (_result_repair_identity(result) for result in results)
        if identity in _ORIGINAL_REPAIR_SCOPE
    }
    expected_completed_repairs = preserved_count - MIN_PRESERVED_RESULTS
    if len(completed_scope) != expected_completed_repairs:
        raise RuntimeError(
            "preserved Full-quality results do not form a valid subset of the "
            "original three-endpoint repair scope"
        )
    missing_scope = _ORIGINAL_REPAIR_SCOPE - completed_scope
    if len(missing_scope) != EXPECTED_FULL_QUALITY_RESULTS - preserved_count:
        raise RuntimeError(
            "remaining Full-quality repair scope does not match preserved count"
        )
    return {
        "preserved_results": preserved_count,
        "missing_repair_scope": sorted(missing_scope),
        "allowed_model_rebuilds": sorted({model for model, _run in missing_scope}),
    }


def _initial_preserved_result_count(run_dir: Path) -> int:
    return int(_initial_resume_scope(run_dir)["preserved_results"])


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _integer_field(payload: Mapping[str, Any], name: str, default: int = -1) -> int:
    value = payload.get(name, default)
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _strict_json_snapshot(
    run_dir: Path,
    path: Path,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    """Read one run-owned JSON object without following symlinks.

    The post-hoc path deliberately does not acquire the workflow lock because
    acquiring it writes ownership metadata.  Stable inode/size/mtime checks,
    terminal-session admission, and a second byte-for-byte pass at the end
    provide a read-only fail-closed boundary instead.
    """

    root = run_dir.resolve(strict=True)
    try:
        logical = path.relative_to(run_dir)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside the target run") from exc
    cursor = run_dir
    for part in logical.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError(f"{label} is symlinked")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label} is missing or unsafe") from exc

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"{label} is missing or unsafe") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise RuntimeError(f"{label} changed while it was read")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise RuntimeError(f"{label} changed while it was read")

    def _unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in items:
            if key in payload:
                raise RuntimeError(f"{label} contains duplicate JSON keys")
            payload[key] = value
        return payload

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload, raw


def _artifact_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _require_artifact_binding(
    artifact_index: Mapping[str, Any],
    *,
    logical_path: str,
    raw: bytes,
    kind: str,
    producer_stage: str,
    binding_session_id: str | None = None,
) -> None:
    records = artifact_index.get("artifacts")
    if not isinstance(records, list) or not all(
        isinstance(row, Mapping) for row in records
    ):
        raise RuntimeError("artifact_index.json artifact rows are invalid")
    matching = [
        row for row in records
        if str(row.get("path") or "") == logical_path
        and str(row.get("kind") or "") == kind
        and str(row.get("producer_stage") or "") == producer_stage
        and not str(row.get("model_id") or "")
        and (
            binding_session_id is None
            or str(row.get("binding_session_id") or "")
            == binding_session_id
        )
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"artifact index has no unique binding for {logical_path}"
        )
    row = matching[0]
    if (
        type(row.get("size_bytes")) is not int
        or row.get("size_bytes") != len(raw)
        or str(row.get("sha256") or "") != _artifact_digest(raw)
    ):
        raise RuntimeError(
            f"artifact index SHA/size mismatch for {logical_path}"
        )


def _parse_timestamp(value: Any, *, label: str) -> datetime:
    token = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} timestamp is missing or invalid") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} timestamp is not timezone-aware")
    return parsed


def _quality_result_row_count(run_dir: Path) -> int:
    summary = _load_quality_summary(run_dir)
    results = summary.get("results")
    if not isinstance(results, list):
        raise RuntimeError("central Full-quality result list is missing or invalid")
    return len(results)


def _verify_workflow_result(result: Any) -> str:
    """Reject an unsuccessful workflow before inspecting a possibly stale 8/8."""

    status = str(_result_field(result, "status", "") or "").strip().lower()
    completed = _result_field(result, "completed", None)
    if status not in {"ok", "partial"} or completed is False:
        raise RuntimeError(
            "targeted Full-quality workflow did not complete successfully: "
            f"status={status or 'missing'}, completed={completed!r}"
        )
    return status


def _verify_must_reuse_stages(
    stage_results: Sequence[Any], *, allowed_model_rebuilds: Sequence[str],
) -> dict[str, Any]:
    """Permit only the six narrowly targeted rebuild stages.

    Every other stage must carry the runner's explicit reuse decision and reuse
    skip marker.  In particular, a forged targeted flag cannot authorize model
    preparation, BenchmarkSet generation, or backend artifact construction.
    """

    if not isinstance(stage_results, (list, tuple)) or not stage_results:
        raise RuntimeError(
            "targeted Full-quality workflow returned no auditable stage results"
        )

    allowed_models = {
        str(model_id or "").strip().lower()
        for model_id in allowed_model_rebuilds
        if str(model_id or "").strip()
    }
    if not allowed_models or not allowed_models.issubset({"yolo26m", "yolo11l"}):
        raise RuntimeError(
            "targeted Full-quality model rebuild scope is missing or invalid"
        )

    violations: list[str] = []
    targeted_rebuilds: list[str] = []
    reused_count = 0
    for raw in stage_results:
        if not isinstance(raw, Mapping):
            violations.append("invalid_stage_result")
            continue
        stage = str(raw.get("stage") or "").strip()
        model_id = str(raw.get("model_id") or "").strip()
        label = f"{model_id}/{stage}" if model_id else f"workflow/{stage}"
        details = raw.get("details")
        decision = (
            details.get("resume_decision")
            if isinstance(details, Mapping)
            and isinstance(details.get("resume_decision"), Mapping)
            else {}
        )
        reused = bool(
            str(raw.get("skip_reason") or "")
            == "resume_reused_existing_stage_result"
            and decision.get("reusable") is True
        )
        if reused:
            reused_count += 1
            continue

        targeted = decision.get("missing_full_quality_targeted_rebuild") is True
        allowed = (
            model_id.lower() in allowed_models
            and stage in _ALLOWED_MODEL_REBUILD_STAGES
            if model_id
            else stage in _ALLOWED_ROOT_REBUILD_STAGES
        )
        if targeted and allowed:
            targeted_rebuilds.append(label)
            continue

        reason = str(decision.get("reason") or "missing_resume_decision")
        violations.append(f"{label}:{reason}")

    if violations:
        raise RuntimeError(
            "must-reuse stage rebuild detected: " + ", ".join(violations[:20])
        )
    return {
        "stage_results_checked": len(stage_results),
        "must_reuse_stages_reused": reused_count,
        "targeted_rebuild_stage_count": len(targeted_rebuilds),
        "targeted_rebuild_stages": targeted_rebuilds,
        "must_reuse_stage_rebuilds": 0,
    }


def _verify_final_summary(
    run_dir: Path,
    *,
    preserved_results: int,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not MIN_PRESERVED_RESULTS <= preserved_results <= MAX_PRESERVED_RESULTS:
        raise RuntimeError(
            "preserved Full-quality result count must be between 5 and 7"
        )
    summary = summary if summary is not None else _load_quality_summary(run_dir)
    contract = (
        summary.get("quality_acceptance_identity_contract")
        if isinstance(summary.get("quality_acceptance_identity_contract"), Mapping)
        else {}
    )
    postcondition = (
        contract.get("postcondition")
        if isinstance(contract, Mapping) else {}
    )
    results = _completed_quality_results(summary)
    exact = bool(
        isinstance(postcondition, Mapping)
        and str(postcondition.get("status") or "") == "verified_exact"
        and _integer_field(postcondition, "expected_count")
        == EXPECTED_FULL_QUALITY_RESULTS
        and _integer_field(postcondition, "completed_count")
        == EXPECTED_FULL_QUALITY_RESULTS
        and _integer_field(postcondition, "failed_count") == 0
        and _integer_field(postcondition, "missing_count") == 0
        and _integer_field(postcondition, "duplicate_count") == 0
        and _integer_field(postcondition, "contract_error_count") == 0
        and len(results) == EXPECTED_FULL_QUALITY_RESULTS
    )
    if not exact:
        raise RuntimeError("final Full-quality identity contract is not exact 8/8")
    return {
        "schema": "onnx-splitpoint/missing-full-quality-resume-result/v1",
        "status": "PASS",
        "run_dir": str(run_dir),
        "expected_results": EXPECTED_FULL_QUALITY_RESULTS,
        "completed_results": EXPECTED_FULL_QUALITY_RESULTS,
        "preserved_results": preserved_results,
        "repaired_results": EXPECTED_FULL_QUALITY_RESULTS - preserved_results,
        "quality_decision": str(summary.get("quality_decision") or ""),
        "technical_status": str(summary.get("technical_status") or ""),
        "builds_requested": 0,
        "ranking_enabled": False,
        "native_enabled": False,
        "energy_enabled": False,
    }


def _verify_resume_outcome(
    run_dir: Path,
    result: Any,
    *,
    preserved_results: int,
    allowed_model_rebuilds: Sequence[str],
) -> dict[str, Any]:
    # Lifecycle comes first: an old exact summary must never mask a failed or
    # cancelled invocation.  Stage reuse is then audited before PASS is minted.
    workflow_status = _verify_workflow_result(result)
    stage_audit = _verify_must_reuse_stages(
        _result_field(result, "stage_results", []),
        allowed_model_rebuilds=allowed_model_rebuilds,
    )
    payload = _verify_final_summary(
        run_dir, preserved_results=preserved_results,
    )
    payload["workflow_status"] = workflow_status
    payload["repair_models"] = sorted(
        {str(value).strip().lower() for value in allowed_model_rebuilds}
    )
    payload.update(stage_audit)
    return payload


def _audit_persisted_resume_decisions(
    resume_summary: Mapping[str, Any],
    *,
    run_id: str,
    profile_id: str,
    workflow_version: str,
    session_started: datetime,
    session_finished: datetime,
    allowed_model_rebuilds: Sequence[str],
    expected_must_reuse_count: int,
) -> dict[str, Any]:
    """Audit the persisted decision matrix without executing a stage."""

    raw_decisions = resume_summary.get("decisions")
    if not isinstance(raw_decisions, list) or not all(
        isinstance(row, Mapping) for row in raw_decisions
    ):
        raise RuntimeError("jobs/resume_summary.json decisions are invalid")
    expected_keys = {
        ("", stage) for stage in _ROOT_STAGE_SURFACE
    } | {
        (model_id, stage)
        for model_id in _PHASE5_MODELS
        for stage in _MODEL_STAGE_SURFACE
    }
    if (
        type(resume_summary.get("decision_count")) is not int
        or resume_summary.get("decision_count") != len(raw_decisions)
        or len(raw_decisions) != len(expected_keys)
    ):
        raise RuntimeError("persisted resume decision cardinality is invalid")

    target_models = {
        str(value or "").strip().lower()
        for value in allowed_model_rebuilds
        if str(value or "").strip()
    }
    targeted_keys = {
        ("", stage) for stage in _ALLOWED_ROOT_REBUILD_STAGES
    } | {
        (model_id, stage)
        for model_id in target_models
        for stage in _ALLOWED_MODEL_REBUILD_STAGES
    }
    observed_keys: set[tuple[str, str]] = set()
    observed_targeted_keys: set[tuple[str, str]] = set()
    targeted: list[str] = []
    reused_count = 0
    legacy_count = 0
    derived_counts: dict[str, int] = {}
    violations: list[str] = []

    for decision in raw_decisions:
        model_id = str(decision.get("model_id") or "").strip().lower()
        stage = str(decision.get("stage") or "").strip()
        key = (model_id, stage)
        label = f"{model_id}/{stage}" if model_id else f"workflow/{stage}"
        if key in observed_keys:
            violations.append(f"duplicate:{label}")
            continue
        observed_keys.add(key)
        expected_result_path = (
            f"models/{model_id}/stages/{stage}/stage_result.json"
            if model_id else f"stages/{stage}/stage_result.json"
        )
        try:
            created = _parse_timestamp(
                decision.get("created_at"), label=f"decision {label}",
            )
        except RuntimeError as exc:
            violations.append(f"{label}:{exc}")
            continue
        if not session_started <= created <= session_finished:
            violations.append(f"{label}:outside_current_session")
            continue
        if (
            decision.get("schema")
            != "onnx-splitpoint/evaluation-stage-resume-decision"
            or decision.get("schema_version") != 1
            or str(decision.get("run_id") or "") != run_id
            or str(decision.get("profile_id") or "") != profile_id
            or str(decision.get("stage_result_path") or "")
            != expected_result_path
            or decision.get("resume_requested") is not True
            or decision.get("force_stage") is not False
        ):
            violations.append(f"{label}:decision_identity_invalid")
            continue

        reusable = decision.get("reusable") is True
        reason = str(decision.get("reason") or "")
        count_key = "reused" if reusable else (reason or "not_reused")
        derived_counts[count_key] = derived_counts.get(count_key, 0) + 1

        if key in targeted_keys:
            canonical = bool(
                not reusable
                and reason == "missing_full_quality_targeted_rebuild"
                and decision.get("missing_full_quality_targeted_rebuild") is True
            )
            legacy = bool(
                workflow_version == _LEGACY_V27711_WORKFLOW_VERSION
                and key in _LEGACY_V27711_FAILED_TARGET_KEYS
                and not reusable
                and reason == "previous_stage_state_not_reusable:failed"
                and str(decision.get("previous_status") or "").strip().lower()
                == "failed"
                and decision.get("missing_full_quality_targeted_rebuild")
                is not True
            )
            if canonical or legacy:
                observed_targeted_keys.add(key)
                targeted.append(label)
                legacy_count += int(legacy)
                continue
            violations.append(f"{label}:{reason or 'missing_resume_decision'}")
            continue

        if (
            reusable
            and reason
            == "missing_full_quality_reuse_existing_stage_ignore_hash"
            and decision.get("missing_full_quality_reuse") is True
            and decision.get("missing_full_quality_targeted_rebuild") is not True
            and decision.get("artifacts_complete") is True
            and str(decision.get("previous_status") or "").strip().lower()
            in {"ok", "warn", "partial", "skipped"}
        ):
            reused_count += 1
            continue
        violations.append(f"{label}:{reason or 'missing_resume_decision'}")

    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        violations.append(
            "decision_surface_mismatch:"
            f"missing={missing[:5]}:unexpected={unexpected[:5]}"
        )
    if observed_targeted_keys != targeted_keys:
        violations.append("targeted_rebuild_surface_incomplete")
    if reused_count != expected_must_reuse_count:
        violations.append(
            "must_reuse_count_mismatch:"
            f"{reused_count}:{expected_must_reuse_count}"
        )
    raw_counts = resume_summary.get("counts")
    if (
        not isinstance(raw_counts, Mapping)
        or not all(
            isinstance(key, str) and type(value) is int and value >= 0
            for key, value in raw_counts.items()
        )
        or dict(raw_counts) != derived_counts
    ):
        violations.append("resume_summary_counts_mismatch")
    if violations:
        raise RuntimeError(
            "persisted resume decision audit failed: "
            + ", ".join(violations[:20])
        )
    return {
        "stage_results_checked": len(raw_decisions),
        "must_reuse_stages_reused": reused_count,
        "targeted_rebuild_stage_count": len(targeted),
        "targeted_rebuild_stages": targeted,
        "legacy_v27711_targeted_decision_count": legacy_count,
        "must_reuse_stage_rebuilds": 0,
    }


def _verify_completed_resume_read_only(run_dir: Path) -> dict[str, Any]:
    """Mint PASS for an already exact run only from its sealed resume trail."""

    manifest_path = run_dir / "run_manifest.json"
    index_path = run_dir / "artifact_index.json"
    quality_path = (
        run_dir / "quality_management" / "central_quality_summary.json"
    )
    manifest, manifest_raw = _strict_json_snapshot(
        run_dir, manifest_path, label="run_manifest.json",
    )
    artifact_index, index_raw = _strict_json_snapshot(
        run_dir, index_path, label="artifact_index.json",
    )
    quality_summary, quality_raw = _strict_json_snapshot(
        run_dir, quality_path, label="central Full-quality summary",
    )
    snapshots: list[tuple[Path, str, bytes]] = [
        (manifest_path, "run_manifest.json", manifest_raw),
        (index_path, "artifact_index.json", index_raw),
        (quality_path, "central Full-quality summary", quality_raw),
    ]

    run_id = str(manifest.get("run_id") or "").strip()
    profile_id = str(manifest.get("profile_id") or "").strip()
    if (
        manifest.get("schema") != "onnx-splitpoint/evaluation-run-manifest"
        or not run_id
        or run_id != run_dir.name
        or not profile_id
        or artifact_index.get("schema") != "onnx-splitpoint/artifact-index"
        or str(artifact_index.get("run_id") or "") != run_id
        or str(artifact_index.get("profile_id") or "") != profile_id
    ):
        raise RuntimeError("terminal run manifest/artifact-index identity is invalid")
    _require_artifact_binding(
        artifact_index,
        logical_path="quality_management/central_quality_summary.json",
        raw=quality_raw,
        kind="stage_artifact",
        producer_stage="evaluate_quality",
    )

    session_id = str(manifest.get("current_session_id") or "").strip()
    sessions = manifest.get("execution_sessions")
    if (
        _SESSION_ID_PATTERN.fullmatch(session_id) is None
        or not isinstance(sessions, list)
        or not sessions
        or not all(isinstance(row, Mapping) for row in sessions)
        or len({str(row.get("session_id") or "") for row in sessions})
        != len(sessions)
        or str(sessions[-1].get("session_id") or "") != session_id
    ):
        raise RuntimeError("current terminal resume execution session is invalid")
    current = sessions[-1]
    workflow_version = str(current.get("workflow_version") or "").strip()
    tool_version = str(current.get("tool_version") or "").strip()
    expected_tool_version = _SUPPORTED_POSTHOC_WORKFLOWS.get(workflow_version)
    workflow_status = str(current.get("status") or "").strip().lower()
    if (
        current.get("schema")
        != "onnx-splitpoint/evaluation-execution-session"
        or current.get("schema_version") != 1
        or expected_tool_version is None
        or tool_version != expected_tool_version
        or current.get("resume_requested") is not True
        or current.get("resumed_existing_manifest") is not True
        or workflow_status not in {"ok", "partial"}
        or str(current.get("error_class") or "").strip()
        or str(current.get("error_detail") or "").strip()
        or str(manifest.get("status") or "").strip().lower()
        != workflow_status
        or str(manifest.get("current_workflow_version") or "")
        != workflow_version
        or str(manifest.get("current_tool_version") or "") != tool_version
    ):
        raise RuntimeError("current resume execution session is not terminal-successful")
    session_started = _parse_timestamp(
        current.get("started_at"), label="current session start",
    )
    session_finished = _parse_timestamp(
        current.get("finished_at"), label="current session finish",
    )
    if session_finished < session_started:
        raise RuntimeError("current resume execution session timestamps are invalid")

    attestation_logical = (
        "jobs/missing_full_quality_reuse_attestations/"
        f"{session_id}.json"
    )
    attestation_path = run_dir / attestation_logical
    attestation, attestation_raw = _strict_json_snapshot(
        run_dir,
        attestation_path,
        label="current-session missing Full-quality reuse attestation",
    )
    snapshots.append((attestation_path, "reuse attestation", attestation_raw))
    target_values = attestation.get("targeted_rebuild_models")
    target_models = (
        [str(value or "").strip().lower() for value in target_values]
        if isinstance(target_values, list) else []
    )
    preserved_count = _integer_field(
        attestation, "preserved_quality_result_count",
    )
    remaining_count = _integer_field(
        attestation, "remaining_quality_result_count",
    )
    capacity = {"yolo26m": 1, "yolo11l": 2}
    if (
        attestation.get("schema")
        != "onnx-splitpoint/missing-full-quality-reuse-cohort-attestation"
        or attestation.get("schema_version") != 1
        or attestation.get("status") != "verified"
        or str(attestation.get("run_id") or "") != run_id
        or str(attestation.get("session_id") or "") != session_id
        or str(attestation.get("workflow_version") or "") != workflow_version
        or target_models != sorted(set(target_models))
        or not target_models
        or not set(target_models).issubset(capacity)
        or not MIN_PRESERVED_RESULTS <= preserved_count <= MAX_PRESERVED_RESULTS
        or preserved_count + remaining_count != EXPECTED_FULL_QUALITY_RESULTS
        or remaining_count < len(target_models)
        or remaining_count > sum(capacity[model] for model in target_models)
        or _SHA256_PATTERN.fullmatch(
            str(attestation.get("archived_artifact_index_sha256") or "")
        ) is None
        or _SHA256_PATTERN.fullmatch(
            str(
                attestation.get("archived_artifact_index_payload_sha256") or ""
            )
        ) is None
        or not all(
            isinstance(attestation.get(name), list)
            and all(isinstance(row, Mapping) for row in attestation.get(name))
            for name in ("supersessions", "historical_omissions", "rebound_paths")
        )
    ):
        raise RuntimeError("current-session reuse attestation is invalid")
    if _parse_timestamp(
        attestation.get("created_at"), label="reuse attestation",
    ) > session_finished:
        raise RuntimeError("reuse attestation postdates the terminal session")
    expected_decision_count = (
        len(_ROOT_STAGE_SURFACE)
        + len(_PHASE5_MODELS) * len(_MODEL_STAGE_SURFACE)
    )
    expected_targeted_count = (
        len(_ALLOWED_ROOT_REBUILD_STAGES)
        + len(target_models) * len(_ALLOWED_MODEL_REBUILD_STAGES)
    )
    expected_must_reuse_count = (
        expected_decision_count - expected_targeted_count
    )
    if (
        type(attestation.get("must_reuse_stage_count")) is not int
        or attestation.get("must_reuse_stage_count")
        != expected_must_reuse_count
    ):
        raise RuntimeError("reuse attestation must-reuse cardinality is invalid")
    _require_artifact_binding(
        artifact_index,
        logical_path=attestation_logical,
        raw=attestation_raw,
        kind="resume_attestation",
        producer_stage="missing_full_quality_reuse_attestation",
        binding_session_id=session_id,
    )

    resume_logical = str(manifest.get("resume_summary") or "").strip()
    if resume_logical != "jobs/resume_summary.json":
        raise RuntimeError("current run has no canonical persisted resume summary")
    resume_path = run_dir / resume_logical
    resume_summary, resume_raw = _strict_json_snapshot(
        run_dir, resume_path, label="jobs/resume_summary.json",
    )
    snapshots.append((resume_path, "jobs/resume_summary.json", resume_raw))
    summary_created = _parse_timestamp(
        resume_summary.get("created_at"), label="resume summary",
    )
    if (
        resume_summary.get("schema")
        != "onnx-splitpoint/evaluation-resume-summary"
        or resume_summary.get("schema_version") != 1
        or str(resume_summary.get("run_id") or "") != run_id
        or str(resume_summary.get("profile_id") or "") != profile_id
        or str(resume_summary.get("workflow_version") or "")
        != workflow_version
        or str(resume_summary.get("tool_version") or "") != tool_version
        or not session_started <= summary_created <= session_finished
    ):
        raise RuntimeError("jobs/resume_summary.json is not bound to current session")
    _require_artifact_binding(
        artifact_index,
        logical_path=resume_logical,
        raw=resume_raw,
        kind="job_artifact",
        producer_stage="workflow_resume",
    )
    decision_audit = _audit_persisted_resume_decisions(
        resume_summary,
        run_id=run_id,
        profile_id=profile_id,
        workflow_version=workflow_version,
        session_started=session_started,
        session_finished=session_finished,
        allowed_model_rebuilds=target_models,
        expected_must_reuse_count=expected_must_reuse_count,
    )

    payload = _verify_final_summary(
        run_dir,
        preserved_results=preserved_count,
        summary=quality_summary,
    )
    payload.update({
        "workflow_status": workflow_status,
        "repair_models": target_models,
        "verification_mode": "read_only_exact_8_of_8",
        "runner_invoked": False,
        "resume_session_id": session_id,
        "reuse_attestation": attestation_logical,
        "resume_summary": resume_logical,
    })
    payload.update(decision_audit)

    for path, label, original in snapshots:
        _payload, current_raw = _strict_json_snapshot(
            run_dir, path, label=label,
        )
        if current_raw != original:
            raise RuntimeError(f"{label} changed during read-only verification")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resume one interrupted Phase-5 Full-only quality run and repair "
            "only its sealed missing 1--3 of 8 endpoints, or verify an "
            "already exact 8/8 retry without invoking the workflow."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--confirm-force-build", action="append", choices=["hailo", "deepx"], default=[], help="Confirm inherited Force for this Resume only; repeat for each affected backend. Does not change the archived profile.")
    ns = parser.parse_args()
    run_dir = Path(ns.run_dir).expanduser().resolve()

    try:
        if _quality_result_row_count(run_dir) == EXPECTED_FULL_QUALITY_RESULTS:
            # A completed targeted retry must never dispatch hardware merely
            # because its v2.77.11 wrapper rejected two legacy decision labels.
            # Admit it only from the stable, current-session evidence above.
            payload = _verify_completed_resume_read_only(run_dir)
        else:
            initial_scope = _initial_resume_scope(run_dir)
            options = _options_from_manifest(run_dir)
            if ns.confirm_force_build:
                options.force_build_confirmed_backends = tuple(dict.fromkeys(ns.confirm_force_build))
                options.force_build_confirmation_source = "cli_explicit_option"
            result = EvaluationWorkflowRunner(
                options,
                log=(lambda line: None if ns.json else print(line, flush=True)),
            ).run()
            payload = _verify_resume_outcome(
                run_dir,
                result,
                preserved_results=int(initial_scope["preserved_results"]),
                allowed_model_rebuilds=list(
                    initial_scope["allowed_model_rebuilds"]
                ),
            )
    except Exception as exc:
        payload = {
            "schema": "onnx-splitpoint/missing-full-quality-resume-result/v1",
            "status": "FAIL",
            "run_dir": str(run_dir),
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
        print("MISSING_FULL_QUALITY_RESUME=FAIL", flush=True)
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print("MISSING_FULL_QUALITY_RESUME=PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
