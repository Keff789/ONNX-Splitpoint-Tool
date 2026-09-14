"""Canonical compact EvaluationRun debug-pack builder.

The normal debug pack keeps complete workflow diagnostics, exact bounded
ranking-audit metadata and declared decoded quality predictions/reference
records. Tensors, figures, model binaries and image datasets stay excluded.
Prediction files are included only through existing producer declarations and
file hashes, never by recursively collecting an arbitrary payload directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from ..filesystem_admission import (
    require_output_outside_source,
    require_write_target,
)
from .artifacts import package_build_snapshot, sha256_file
from .debug_pack_policy import (
    STRUCTURED_RESULT_MAX_FILE_BYTES,
    STRUCTURED_RESULT_MAX_TOTAL_BYTES,
    CENTRAL_DESCRIPTOR_MAX_FILE_BYTES,
    CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES,
    BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES,
    BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES,
    MAIN_WORKFLOW_LOG,
    MAIN_WORKFLOW_TAIL,
    MODEL_VALIDATION_SUMMARY_MAX_FILE_BYTES,
    MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES,
    OFFLINE_REPLAY_CORE,
    RANKING_AUDIT_MAX_FILE_BYTES,
    discover_ranking_audit_evidence,
    is_backend_artifact_diagnostic,
    is_candidate_body_mirror,
    is_cancelled_diagnostic_priority,
    is_model_validation_summary,
    is_offline_replay_core,
    is_ranking_audit_evidence,
    is_remote_execution_failure_diagnostic,
)
from .zip_utils import (
    iter_safe_pack_files,
    publish_verified_zip,
    require_safe_pack_directory,
    require_safe_pack_source,
    temporary_zip_path,
    zipinfo_for_path,
)

try:
    from .. import __version__ as TOOL_VERSION
except Exception:  # pragma: no cover - source-tree import fallback
    TOOL_VERSION = "unknown"


DEBUG_PACK_SCHEMA_VERSION = 13
DEFAULT_MAX_SMALL_FILE_BYTES = 2 * 1024 * 1024
DEFAULT_TAIL_BYTES = 1024 * 1024
MAX_TAIL_MEMBERS = 16
MAX_TAIL_TOTAL_BYTES = 16 * 1024 * 1024
EXACT_METADATA_MAX_FILE_BYTES = STRUCTURED_RESULT_MAX_FILE_BYTES
INDEX_VALIDATION_MAX_TOTAL_BYTES = STRUCTURED_RESULT_MAX_TOTAL_BYTES
RUNTIME_FALLBACK_PARSE_MAX_TOTAL_BYTES = STRUCTURED_RESULT_MAX_TOTAL_BYTES
RUNTIME_IDENTITY_OVERVIEW_MAX_BYTES = 16 * 1024
CENTRAL_DESCRIPTOR_MAX_REQUESTS = 1024
# Existing full-quality export envelope, separate from descriptor/index budgets.
# A campaign can contain many 5,000-record classification JSONs.
QUALITY_PREDICTION_MAX_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
# Reference status/stdout are compact diagnostics, with the same bounded
# allowance as the existing backend diagnostics. Never admit reference bodies.
MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_FILE_BYTES = BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES
MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_TOTAL_BYTES = BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES
MANAGEMENT_REFERENCE_DIAGNOSTIC_NAMES = frozenset({
    "management_cpu_reference_status.json",
    "management_cpu_reference_stdout.txt",
})
PROBE_MAX_FILE_BYTES = 64 * 1024 * 1024
PROBE_MAX_TOTAL_BYTES = 32 * 1024 * 1024
NATIVE_RAW_MAX_FILE_BYTES = PROBE_MAX_FILE_BYTES
NATIVE_RAW_MAX_TOTAL_BYTES = 512 * 1024 * 1024
NATIVE_MEASUREMENTS_ROOT = "reports/native_energy_measurements/measurements/"
RUNTIME_DIAGNOSTIC_FILES = frozenset({
    "normalized_results.json", "missing_measurements.json",
    "selected_run_completeness.json", "benchmark_execution_plan.json",
    "run_benchmarks_request.json",
})

# Kept public for the two historical CLI adapters and release-contract tests.
INCLUDE_ROOT_FILES = (
    MAIN_WORKFLOW_LOG,
    "run_manifest.json",
    "artifact_index.json",
    "profile.yaml",
    "profile_source.yaml",
    "profile_start_snapshot.json",
    "profile_resolution.json",
    "effective_execution_plan.json",
    "hardware_matrix.json",
)

INCLUDE_REPORT_FILES = (
    "reports/summary.csv",
    "reports/model_summary.csv",
    "reports/hardware_summary.csv",
    "reports/result_dashboard.json",
    "reports/result_dashboard.md",
    "reports/run_status_summary.json",
    "reports/run_status_summary.md",
    "reports/results_bundle_manifest.json",
    "reports/validation_summary.csv",
    "reports/native_producer_stage.json",
    "reports/native_producer_stage_config.json",
    "reports/native_progress.json",
    "reports/native_progress.jsonl",
    "reports/native_producer_summary.json",
    "reports/native_producer_summary.csv",
    "reports/native_producer_summary.md",
    "reports/native_producer_combined_summary.json",
    "reports/native_producer_combined_summary.csv",
    "reports/native_producer_combined_summary.md",
    "reports/native_stage_concise_summary.json",
    "reports/native_stage_concise_summary.csv",
    "reports/native_evidence_status.json",
    "reports/native_expected_matrix.json",
    "reports/native_expected_matrix_missing.csv",
    "reports/native_host_telemetry_summary.json",
    "reports/native_energy_model_hash_map.json",
    "reports/native_energy_plan/native_producer_energy_plan.json",
    "reports/native_energy_plan/native_producer_energy_plan.md",
    "reports/native_energy_measurements/plan/native_producer_energy_plan.json",
    "reports/native_energy_measurements/plan/native_producer_energy_plan.md",
    "reports/native_energy_measurements/native_producer_energy_results.json",
    "reports/native_energy_measurements/native_producer_energy_results.md",
    "reports/native_energy_measurements/native_producer_energy_results.partial.json",
    "reports/native_validation/native_producer_validation_summary.json",
    "reports/native_validation/native_producer_validation_summary.csv",
    "reports/native_validation/native_producer_validation_summary.md",
    "quality_management/central_quality_summary.json",
    "jobs/job_plan.json",
    "jobs/job_summary.json",
    "jobs/job_events.jsonl",
    "jobs/job_timeline.md",
)

INCLUDE = list(dict.fromkeys((*INCLUDE_ROOT_FILES, *INCLUDE_REPORT_FILES)))
REPLAY_CORE = OFFLINE_REPLAY_CORE

PROVENANCE_EVIDENCE_NAMES = frozenset({
    "hailo_hef_build_receipt.json",
    "setup_local_tensorrt_dispatch_preflight.json",
    "remote_hardware_matrix_status.json",
})

ALLOWED_SUFFIXES = frozenset({
    ".json", ".jsonl", ".csv", ".md", ".txt", ".tex", ".yaml", ".yml",
    ".log",
})
BLOCKED_SUFFIXES = frozenset({
    ".bin", ".onnx", ".hef", ".har", ".engine", ".dxnn", ".npz", ".npy",
    ".png", ".jpg", ".jpeg", ".pdf", ".svg", ".tar", ".gz", ".zip",
    ".whl", ".so", ".dll", ".dylib",
})
BLOCKED_DIRECTORY_PARTS = frozenset({
    "resources",
    "lean_bundle",
    "dist",
    "activation_proxy_cache",
    "evaluation_cache",
    "cpu_reference_store",
})
SCANNED_TOP_LEVELS = frozenset({
    "campaign", "jobs", "native_producers", "quality_management", "reports",
    "stages",
})


def _relative(run_dir: Path, path: Path) -> str:
    return path.relative_to(run_dir).as_posix()


def _is_runtime_diagnostic(relative: str) -> bool:
    """Exact canonical runtime diagnostics; never a broad model-tree rule."""
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts) or "\\" in relative:
        return False
    if len(parts) < 3 or parts[0] != "models" or not parts[1]:
        return False
    tail = parts[2:]
    if tail == ["model_manifest.json"]:
        return True
    if tail[0] == "benchmark_results":
        if len(tail) == 2:
            name = tail[-1]
            return name in RUNTIME_DIAGNOSTIC_FILES or name in {"benchmark_results.json", "results.json"} or (
                name.startswith("benchmark_results_") and name.endswith(".json")
            )
        if len(tail) >= 3 and tail[1] == "remote_diagnostics":
            remote = tail[2:]
            # Canonical and host-specific diagnostics remain separate. Only
            # this status matrix may pass through a lean_bundle directory.
            if len(remote) == 3:
                if remote[0] in BLOCKED_DIRECTORY_PARTS:
                    return False
                remote = remote[1:]
            return remote in (
                ["logs", "runner.log"],
                ["lean_bundle", "benchmark_suite_status_matrix.json"],
            )
    if tail[0] == "validation":
        return len(tail) == 2 and tail[1] in {
            "validation_case_matrix.json", "validation_case_matrix.csv",
        }
    if tail == ["full_baselines", "output_contracts.json"]:
        return True
    if tail[:2] == ["benchmark_set", "legacy_suite"]:
        suite = tail[2:]
        if suite in (
            ["results", "deepx_m1_full", "original_full_failure.json"],
            ["results", "deepx_m1_full", "prepared_input", "native_full_input_manifest.json"],
        ):
            return True
        return suite in (["benchmark_plan.json"], ["output_contracts.json"]) or (
            len(suite) == 4 and suite[0] in {"deepx", "hailo", "tensorrt"}
            and bool(suite[1]) and suite[2:] == ["full", "output_contract.json"]
        )
    return False


def _runtime_diagnostic_inventory(
    run_dir: Path, by_relative: Mapping[str, Path], *, max_bytes: int,
    index_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retain exact files and notice missing backend JSONs in existing indexes.

    Read only bounded canonical indexes; a copied run can retain its original
    absolute paths. Match their exact model-relative suffix and never open a
    declared external path or infer an artefact from a filename alone.
    """
    present = {rel for rel in by_relative if _is_runtime_diagnostic(rel)}
    expected = set(present) | set(index_records or {})
    failures: list[dict[str, str]] = []
    # Enumerate names only inside verified directories. Unsafe file symlinks
    # remain visible as rejected sources without reading their destinations.
    try:
        models = require_safe_pack_directory(run_dir / "models", run_dir)
        for model in sorted(models.iterdir()):
            try:
                model = require_safe_pack_directory(model, run_dir)
                results = require_safe_pack_directory(model / "benchmark_results", run_dir)
                for source in results.iterdir():
                    relative = _relative(run_dir, source)
                    if _is_runtime_diagnostic(relative):
                        expected.add(relative)
            except (OSError, RuntimeError, ValueError):
                continue
    except (OSError, RuntimeError, ValueError):
        pass
    # A historical debug export may retain the inventory while the originals
    # were deliberately omitted. Keep those exact names, never fabricate rows.
    old_manifest = run_dir / "debug_pack_manifest.json"
    try:
        old_manifest = require_safe_pack_source(old_manifest, run_dir)
        if old_manifest.stat().st_size <= EXACT_METADATA_MAX_FILE_BYTES:
            old = json.loads(old_manifest.read_text(encoding="utf-8"))
            inventory = old.get("runtime_execution_diagnostics", {})
            for relative in inventory.get("expected_members", []):
                if isinstance(relative, str) and _is_runtime_diagnostic(relative):
                    expected.add(relative)
    except FileNotFoundError:
        pass
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        failures.append({"path": "debug_pack_manifest.json",
                         "reason": f"historical_inventory_unreadable:{type(exc).__name__}"})
    for relative in sorted(present):
        parts = relative.split("/")
        if len(parts) != 4 or parts[2] != "benchmark_results":
            continue
        if parts[-1] not in RUNTIME_DIAGNOSTIC_FILES:
            continue
        source = by_relative[relative]
        if source.stat().st_size > STRUCTURED_RESULT_MAX_FILE_BYTES:
            continue  # The admission/omission manifest records this limit.
        try:
            payload = _read_compact_json(source, run_dir, STRUCTURED_RESULT_MAX_FILE_BYTES)
            pending = [payload]
            while pending:
                node = pending.pop()
                if isinstance(node, list):
                    pending.extend(node)
                elif isinstance(node, Mapping):
                    pending.extend(
                        value for value in node.values()
                        if isinstance(value, (Mapping, list))
                    )
                    for key in ("source_path", "primary_result_path"):
                        value = node.get(key)
                        if not isinstance(value, str) or not value:
                            continue
                        declared = Path(value)
                        member = "/".join(parts[:3] + [declared.name])
                        if not (
                            (declared.name.startswith("benchmark_results_")
                             and declared.suffix == ".json")
                            or declared.name in {"benchmark_results.json", "results.json"}
                        ):
                            continue
                        if ".." in declared.parts or not (
                            value == declared.name or value == member
                            or value.endswith("/" + member)
                        ):
                            failures.append({
                                "path": relative,
                                "reason": "backend_result_reference_outside_model",
                            })
                            continue
                        expected.add(member)
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            failures.append({
                "path": relative,
                "reason": f"runtime_index_invalid:{type(exc).__name__}",
            })
    return {
        "expected_members": sorted(expected),
        "present_source_members": sorted(present),
        "missing_source_members": sorted(expected - present),
        "reference_failures": failures,
        "source_records": dict(index_records or {}),
    }


def _is_runtime_summary(relative: str) -> bool:
    parts = relative.split("/")
    return bool(len(parts) == 5 and parts[0] == "models"
                and all(part not in {"", ".", ".."} for part in parts)
                and "\\" not in relative
                and parts[2:4] == ["benchmark_results", "diagnostic_summaries"]
                and (parts[4] in {"normalized_results.summary.json", "benchmark_results.summary.json", "results.summary.json"}
                     or (parts[4].startswith("benchmark_results_")
                         and parts[4].endswith(".summary.json"))))


class DebugSizeLimit(ValueError):
    def __init__(self, size: int, limit: int):
        self.details = {"size_bytes": size, "limit_bytes": limit}
        super().__init__(f"size_limit_exceeded:size_bytes={size}:limit_bytes={limit}")


def _read_compact_json(source: Path, root: Path, limit: int) -> Any:
    """Bound bytes before JSON decoding; never read a large file's tail."""
    source = require_safe_pack_source(source, root)
    size = source.stat().st_size
    if size > limit:
        raise DebugSizeLimit(size, limit)
    with source.open("rb") as stream:
        body = stream.read(limit + 1)
    if len(body) > limit:
        raise DebugSizeLimit(len(body), limit)
    return json.loads(body)


def _runtime_compact_export(
    root: Path, inventory: Mapping[str, Any], written_paths: set[str],
    skipped: list[dict[str, Any]], *, max_bytes: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare bounded export-only summaries without mutating historical runs.

    Original archive coverage is based on actual ZIP writes. Companions and
    fallback projections are separate derived evidence, never original rows.
    """
    from .compact_runtime_diagnostics import (
        FALLBACK_PARSE_MAX_BYTES, SUMMARY_MAX_BYTES, SUMMARY_TOTAL_MAX_BYTES,
        companion_matches, companion_path, project_runtime_payload,
        source_observation, summary_bytes,
    )

    omitted = {row["path"]: row.get("reason", "omitted") for row in skipped}
    coverage: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    summary_total = 0
    parse_total = 0
    for relative in inventory["expected_members"]:
        row: dict[str, Any] = {"source_path": relative,
                               "original_archived": relative in written_paths,
                               "derived_summary_archived": False}
        coverage.append(row)
        if relative in written_paths:
            row["status"] = "original_archived"
            continue
        source = root / relative
        try:
            source = require_safe_pack_source(source, root)
            observation = source_observation(source)
            row["observed_size_bytes"] = observation.get("observed_size_bytes")
        except FileNotFoundError:
            row.update(status="source_missing", summary_status="summary_unavailable",
                       summary_reason="source_missing")
            continue
        except (RuntimeError, ValueError) as exc:
            row.update(status="source_unsafe", summary_status="summary_unavailable",
                       summary_reason=f"source_unsafe:{type(exc).__name__}")
            continue
        except OSError as exc:
            row.update(status="source_unreadable", summary_status="summary_unavailable",
                       summary_reason=f"source_unreadable:{type(exc).__name__}")
            continue
        reason = omitted.get(relative, "source_not_archived")
        row["original_omission_reason"] = reason
        source_limit = (STRUCTURED_RESULT_MAX_FILE_BYTES if source.suffix.lower() in {".json", ".csv"}
                        else max(0, int(max_bytes)))
        oversized = int(source.stat().st_size) > source_limit
        row["status"] = "original_omitted_size_limit" if oversized else "original_not_archived"
        declared = inventory.get("source_records", {}).get(relative, {})
        if declared.get("sha256"):
            row["declared_original_sha256"] = declared["sha256"]
            row["declared_original_sha256_verification"] = "declared_not_reverified"
        summary_source = companion_path(source)
        summary_relative = _relative(root, summary_source)
        if not oversized or not _is_runtime_summary(summary_relative):
            row.update(summary_status="summary_unavailable",
                       summary_reason=reason if not oversized else "unsupported_source_kind")
            continue
        origin = "existing_companion"
        projection: dict[str, Any] | None = None
        try:
            # lstat notices symlinks as existing invalid companions: a stale or
            # conflicting companion is not hidden by silently replacing it.
            try:
                summary_source.lstat()
                has_companion = True
            except FileNotFoundError:
                has_companion = False
            if has_companion:
                projection = _read_compact_json(summary_source, root, SUMMARY_MAX_BYTES)
                if not isinstance(projection, Mapping):
                    raise ValueError("companion_invalid_schema")
                matches, match_reason = companion_matches(
                    projection, source_path=relative, source_stat=observation)
                if not matches:
                    raise ValueError("companion_unconfirmed:" + match_reason)
            else:
                size = int(source.stat().st_size)
                if size > FALLBACK_PARSE_MAX_BYTES:
                    raise ValueError("fallback_not_checked_size_limit")
                if parse_total + size > RUNTIME_FALLBACK_PARSE_MAX_TOTAL_BYTES:
                    raise ValueError("fallback_not_checked_budget_limit")
                parse_total += size
                origin = "export_only_fallback"
                payload = _read_compact_json(source, root, FALLBACK_PARSE_MAX_BYTES)
                if source_observation(source) != observation:
                    raise ValueError("source_changed_during_projection")
                projection = project_runtime_payload(
                    payload, source_path=relative, source_stat=observation,
                    declared_source_sha256=declared.get("sha256"), max_bytes=SUMMARY_MAX_BYTES)
                origin = "export_only_fallback"
            if projection.get("projection_role") != "derived_summary":
                raise ValueError("companion_projection_role_invalid")
            if projection.get("status") != "projected":
                raise ValueError(str(projection.get("reason") or "projection_unavailable"))
            body = summary_bytes(projection)
            if len(body) > SUMMARY_MAX_BYTES:
                raise ValueError("summary_size_limit")
            if summary_total + len(body) > SUMMARY_TOTAL_MAX_BYTES:
                raise ValueError("summary_total_budget_limit")
            if source_observation(source) != observation:
                raise ValueError("source_changed_during_projection")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, RecursionError) as exc:
            row.update(summary_status="summary_unavailable",
                       summary_reason=f"{type(exc).__name__}:{exc}")
            if origin == "export_only_fallback" and isinstance(exc, OSError):
                row.update(status="source_unreadable", summary_reason=f"source_unreadable:{type(exc).__name__}")
            if isinstance(exc, (json.JSONDecodeError, UnicodeError, RecursionError)):
                row["summary_reason"] = "source_invalid_json" if origin == "export_only_fallback" or not has_companion else "companion_invalid_json"
                if origin == "export_only_fallback":
                    row["status"] = "source_invalid"
            if projection is not None:
                from .compact_runtime_diagnostics import IDENTITY_FIELDS, STATUS_FIELDS
                rows = projection.get("rows", [])
                overview = projection.get("identity_failure_overview", [
                    {key: value for key, value in item.items()
                     if key in IDENTITY_FIELDS | STATUS_FIELDS | {"source_pointer"}}
                    for item in rows[:64] if isinstance(item, Mapping)
                ])
                admitted_overview: list[dict[str, Any]] = []
                overview_bytes = 0
                for item in overview[:64]:
                    item_bytes = len(json.dumps(item, ensure_ascii=False).encode("utf-8"))
                    if overview_bytes + item_bytes > RUNTIME_IDENTITY_OVERVIEW_MAX_BYTES:
                        break
                    admitted_overview.append(item)
                    overview_bytes += item_bytes
                row["identity_failure_overview"] = admitted_overview
                row["identity_failure_overview_complete"] = (
                    len(admitted_overview) == projection.get("source_row_count"))
                row["identity_failure_overview_limit_bytes"] = RUNTIME_IDENTITY_OVERVIEW_MAX_BYTES
                row["summary_source_row_count"] = projection.get("source_row_count")
            continue
        summary_total += len(body)
        row.update(summary_status="derived_summary_ready", summary_path=summary_relative,
                   summary_origin=origin, summary_size_bytes=len(body))
        summaries.append({"path": summary_relative, "body": body, "source_path": relative,
                          "source_observation": observation, "coverage": row})
    return coverage, summaries


def _sha_token(value: object) -> str:
    token = str(value or "").strip().lower()
    if token.startswith("sha256:"):
        token = token.split(":", 1)[1]
    if len(token) == 64 and all(ch in "0123456789abcdef" for ch in token):
        return token
    return ""


def _is_central_candidate_body(relative: str) -> bool:
    """Return whether a JSON is a replay body rather than a descriptor."""

    parts = relative.split("/")
    name = parts[-1].lower() if parts else ""
    if parts[:2] == ["quality_management", "references"]:
        return not is_management_reference_diagnostic(relative)
    in_quality_inputs = any(
        part in {"quality_inputs", "task_quality_inputs"} for part in parts[:-1]
    )
    if in_quality_inputs and (
        name.endswith("_candidate.json")
        or name in {
            "candidate.json", "candidate_predictions.json", "reference.json",
            "reference_predictions.json", "annotations.json",
        }
    ):
        return True
    return False


def is_management_reference_diagnostic(relative: str) -> bool:
    """Match only the two canonical per-model diagnostics, without recursion."""
    parts = relative.split("/")
    return bool(
        len(parts) == 4
        and parts[:2] == ["quality_management", "references"]
        and all(part and part not in {".", ".."} for part in parts)
        and "\\" not in relative
        and parts[2] not in {
            "workspaces", "by_source_contract", "venv", ".venv",
            *BLOCKED_DIRECTORY_PARTS,
        }
        and parts[3] in MANAGEMENT_REFERENCE_DIAGNOSTIC_NAMES
    )


def _management_reference_diagnostic_inventory(
    run_dir: Path, by_relative: Mapping[str, Path], *, max_bytes: int,
) -> dict[str, Any]:
    """Inventory existing or index-registered diagnostics, never JSON pointers.

    Do not infer a missing job from the model population. Inspect only the two
    exact filenames in safe, immediate model directories; this also records a
    rejected file symlink that the ordinary safe enumerator intentionally skips.
    The status contents may be interrupted/invalid and are never parsed here.
    """
    expected = {rel for rel in by_relative if is_management_reference_diagnostic(rel)}
    failures: list[dict[str, str]] = []
    references = run_dir / "quality_management" / "references"
    try:
        references = require_safe_pack_directory(references, run_dir)
        for model_dir in sorted(references.iterdir()):
            try:
                model_dir = require_safe_pack_directory(model_dir, run_dir)
            except (OSError, RuntimeError, ValueError):
                continue  # Indexed members below unsafe directories are handled below.
            for name in MANAGEMENT_REFERENCE_DIAGNOSTIC_NAMES:
                path = model_dir / name
                relative = _relative(run_dir, path)
                if not is_management_reference_diagnostic(relative):
                    continue
                try:
                    path.lstat()
                except FileNotFoundError:
                    continue
                except OSError:
                    pass  # Keep the known name so admission records the read failure.
                expected.add(relative)
    except FileNotFoundError:
        pass
    except (OSError, RuntimeError, ValueError) as exc:
        failures.append({
            "path": "quality_management/references",
            "reason": f"source_enumeration_failed:{type(exc).__name__}",
        })

    runtime_source_records: dict[str, dict[str, Any]] = {}
    index = run_dir / "artifact_index.json"
    index_validation: dict[str, Any] = {
        "status": "missing", "source_path": "artifact_index.json",
        "observed_size_bytes": None, "limit_bytes": EXACT_METADATA_MAX_FILE_BYTES,
        "index_coverage_verified": False,
    }
    try:
        index = require_safe_pack_source(index, run_dir)
        index_validation["observed_size_bytes"] = index.stat().st_size
        if index_validation["observed_size_bytes"] > EXACT_METADATA_MAX_FILE_BYTES:
            index_validation["status"] = "size_limit_exceeded"
        elif index_validation["observed_size_bytes"] > INDEX_VALIDATION_MAX_TOTAL_BYTES:
            index_validation["status"] = "not_checked_budget_limit"
            index_validation["budget_limit_bytes"] = INDEX_VALIDATION_MAX_TOTAL_BYTES
        else:
            index_validation["status"] = "unreadable"
            with index.open("rb") as stream:
                data = stream.read(EXACT_METADATA_MAX_FILE_BYTES + 1)
            if len(data) > EXACT_METADATA_MAX_FILE_BYTES:
                index_validation["status"] = "size_limit_exceeded"
                raise ValueError("index_grew_beyond_read_budget")
            index_validation["status"] = "invalid_json"
            payload = json.loads(data)
            index_validation["status"] = "invalid_schema"
            if not isinstance(payload, Mapping) or not isinstance(payload.get("artifacts"), list):
                raise TypeError("artifact index artifacts is not a list")
            records = payload["artifacts"]
            if any(not isinstance(record, Mapping) or not isinstance(record.get("path"), str)
                   for record in records):
                raise TypeError("artifact index record has no string path")
            for record in records:
                declared = record["path"]
                if _is_runtime_diagnostic(declared):
                    runtime_source_records[declared] = {key: record[key] for key in ("sha256", "size_bytes") if key in record}
                if is_management_reference_diagnostic(declared):
                    expected.add(declared)
                elif declared.replace("\\", "/").split("/")[-1] in MANAGEMENT_REFERENCE_DIAGNOSTIC_NAMES:
                    failures.append({
                        "path": declared,
                        "reason": "reference_diagnostic_index_path_not_canonical",
                    })
            index_validation.update(status="verified", index_coverage_verified=not failures,
                                    inspected_record_count=len(records))
    except FileNotFoundError:
        index_validation["status"] = "missing"
    except (OSError, RuntimeError, ValueError, TypeError, AttributeError, RecursionError) as exc:
        if isinstance(exc, (RuntimeError, ValueError)) and index_validation["status"] == "missing":
            index_validation["status"] = "unsafe_path"
        if isinstance(exc, OSError):
            index_validation["status"] = "unreadable"
        if isinstance(exc, (UnicodeError, RecursionError)):
            index_validation["status"] = "invalid_json"
        index_validation["error"] = f"{type(exc).__name__}:{exc}"
        failures.append({
            "path": "artifact_index.json",
            "reason": f"reference_diagnostic_index_{index_validation['status']}:{type(exc).__name__}",
        })

    admitted: list[str] = []
    present: list[str] = []
    omitted: list[dict[str, Any]] = []
    missing: list[str] = []
    total = 0
    for relative in sorted(expected):
        source = run_dir / relative
        try:
            source = require_safe_pack_source(source, run_dir)
            present.append(relative)
            size = source.stat().st_size
            include, reason = should_include_debug_file(run_dir, source, max_bytes)
            if not include:
                omitted.append({"path": relative, "reason": reason, "size_bytes": size})
                continue
            # Check readability before admission to required archive members.
            with source.open("rb") as stream:
                stream.read(1)
            if total + size > MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_TOTAL_BYTES:
                omitted.append({
                    "path": relative,
                    "reason": "management reference diagnostic total limit exceeded",
                    "size_bytes": size,
                    "limit_bytes": MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_TOTAL_BYTES,
                })
                continue
            admitted.append(relative)
            total += size
        except FileNotFoundError:
            missing.append(relative)
            omitted.append({"path": relative, "reason": "source_missing"})
        except (RuntimeError, ValueError) as exc:
            omitted.append({"path": relative, "reason": f"source_unsafe:{type(exc).__name__}"})
        except OSError as exc:
            omitted.append({"path": relative, "reason": f"source_unreadable:{type(exc).__name__}"})
    return {
        "expected_members": sorted(expected),
        "present_source_members": present,
        "admitted_source_members": admitted,
        "index_validation": index_validation,
        "_runtime_source_records": runtime_source_records,
        "missing_source_members": missing,
        "omitted_source_members": omitted,
        "reference_failures": failures,
        "max_file_bytes": MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_FILE_BYTES,
        "max_total_bytes": MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_TOTAL_BYTES,
    }


def _probe_include_raw(run_dir: Path) -> bool:
    try:
        profile = yaml.safe_load(
            require_safe_pack_source(
                run_dir / "profile.yaml", run_dir
            ).read_text(encoding="utf-8")
        ) or {}
        native = profile.get("native_producers") if isinstance(profile, dict) else {}
        energy = native.get("energy") if isinstance(native, dict) else {}
        probe = (
            energy.get("window_method_validation_probe")
            if isinstance(energy, dict) else {}
        )
        if isinstance(probe, dict) and probe:
            return probe.get("include_raw_parquet") is True
    except Exception:
        pass
    try:
        stage = json.loads(
            require_safe_pack_source(
                run_dir / "reports" / "native_producer_stage.json", run_dir
            ).read_text(encoding="utf-8")
        )
        probe = stage.get("window_method_validation_probe") if isinstance(stage, dict) else {}
        resolved = (
            probe.get("resolved_config")
            if isinstance(probe, dict) and isinstance(probe.get("resolved_config"), dict)
            else probe if isinstance(probe, dict) else {}
        )
        if isinstance(resolved, dict) and resolved:
            return resolved.get("include_raw_parquet") is True
    except Exception:
        pass
    try:
        report = json.loads(
            require_safe_pack_source(
                run_dir / "reports" / "window_method_validation_probe"
                / "window_method_validation_probe.json",
                run_dir,
            ).read_text(encoding="utf-8")
        )
        return bool(report.get("raw_parquet_debug_pack_requested"))
    except Exception:
        return False


def _native_include_raw(run_dir: Path) -> bool:
    """Read the existing effective-profile opt-in; probe policy is separate."""
    try:
        profile = yaml.safe_load(require_safe_pack_source(
            run_dir / "profile.yaml", run_dir
        ).read_text(encoding="utf-8"))
        # This setting belongs to top-level ``energy`` in the GUI, schema,
        # run-mode resolver and effective profile. The method probe has its
        # own setting below ``native_producers.energy``.
        return profile["energy"].get("include_raw_parquet_in_debug_pack") is True
    except (
        OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError,
        yaml.YAMLError,
    ):
        return False


def _is_native_raw_trace(relative: str) -> bool:
    path = Path(relative)
    return bool(
        relative.startswith(NATIVE_MEASUREMENTS_ROOT)
        and path.parent.name == "collector_storage"
        and path.suffix.lower() == ".parquet"
    )


def _native_raw_inventory(
    run_dir: Path, by_relative: Mapping[str, Path], *, requested: bool,
) -> dict[str, Any]:
    """Inventory local traces and their existing collector references only.

    Absolute paths in a copied run may retain its old root.  Match the exact
    run-relative suffix beside the receipt, never open the declared path.
    """
    present = {rel for rel in by_relative if _is_native_raw_trace(rel)}
    expected = set(present) if requested else set()
    failures: list[dict[str, str]] = []
    if requested:
        for relative, source in sorted(by_relative.items()):
            receipt = Path(relative)
            if not relative.startswith(NATIVE_MEASUREMENTS_ROOT) or receipt.name not in {
                "energy_summary.json", "command_window_request.json",
            }:
                continue
            try:
                if source.stat().st_size > EXACT_METADATA_MAX_FILE_BYTES:
                    raise ValueError("collector reference exceeds exact limit")
                payload = json.loads(require_safe_pack_source(
                    source, run_dir
                ).read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("collector reference is not an object")
                references = (
                    [payload.get("trace_path")]
                    if receipt.name == "command_window_request.json"
                    else payload.get("parquet_files", [])
                )
                if not isinstance(references, list):
                    raise ValueError("parquet_files is not an array")
                for raw in references:
                    if not raw:
                        continue
                    declared = Path(str(raw))
                    member = receipt.parent / "collector_storage" / declared.name
                    text = declared.as_posix()
                    if (
                        ".." in declared.parts
                        or not _is_native_raw_trace(member.as_posix())
                        or not (
                            text == member.as_posix()
                            or text.endswith("/" + member.as_posix())
                            or text == "collector_storage/" + declared.name
                        )
                    ):
                        failures.append({
                            "path": relative,
                            "reason": "raw_reference_outside_local_collector_storage",
                        })
                        continue
                    expected.add(member.as_posix())
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                failures.append({
                    "path": relative,
                    "reason": f"raw_reference_invalid:{type(exc).__name__}",
                })
    return {
        "include_raw_parquet_requested": requested,
        "requested_members": sorted(expected),
        "present_source_members": sorted(present),
        "missing_source_members": sorted(expected - present),
        "reference_failures": failures,
    }


def _energy_attempt_log_inventory(root: Path, by_relative: Mapping[str, Path]) -> dict[str, Any]:
    """Inventory declared attempts without choosing or validating a measurement.

    The existing collector history is kept byte-for-byte in the archive. This
    view only checks that both discarded and selected stdout evidence survives
    export. An archived absolute path is projected by its unique canonical
    measurements suffix; external paths and symlinks are never opened.
    """
    expected: set[str] = set()
    selected: set[str] = set()
    failures: list[dict[str, str]] = []
    for relative, source in sorted(by_relative.items()):
        if not relative.startswith(NATIVE_MEASUREMENTS_ROOT) or source.name != "energy_aggregate.json":
            continue
        try:
            payload = _read_compact_json(source, root, EXACT_METADATA_MAX_FILE_BYTES)
            runs = payload.get("runs", [])
            for run in runs:
                history = run.get("repeat_attempt_history", [])
                if not isinstance(history, list):
                    raise ValueError("attempt_history_not_list")
                for attempt in history:
                    declared = attempt.get("run_directory")
                    if not isinstance(declared, str) or not declared or "\\" in declared:
                        raise ValueError("attempt_directory_missing")
                    if ".." in Path(declared).parts:
                        raise ValueError("attempt_directory_traversal")
                    if declared.count(NATIVE_MEASUREMENTS_ROOT) == 1:
                        directory = NATIVE_MEASUREMENTS_ROOT + declared.split(NATIVE_MEASUREMENTS_ROOT, 1)[1]
                    elif not Path(declared).is_absolute():
                        directory = (Path(relative).parent / declared).as_posix()
                    else:
                        raise ValueError("attempt_directory_outside_measurement")
                    candidate = root / directory / "workload_stdout.log"
                    # Must remain inside this aggregate's measurement root.
                    candidate.relative_to(source.parent)
                    member = _relative(root, candidate)
                    expected.add(member)
                    if attempt.get("selected") is True:
                        selected.add(member)
                    if member in by_relative:
                        require_safe_pack_source(candidate, root)
        except (OSError, ValueError, TypeError, AttributeError, RuntimeError) as exc:
            failures.append({"path": relative, "reason": str(exc)})
    return {"expected_members": sorted(expected), "selected_stdout_members": sorted(selected),
            "missing_source_members": sorted(expected - set(by_relative)), "reference_failures": failures}


def should_include_debug_file(
    run_dir: Path,
    path: Path,
    max_bytes: int = DEFAULT_MAX_SMALL_FILE_BYTES,
    *,
    probe_include_raw: bool = False,
    native_include_raw: bool = False,
    central_request_descriptor: bool = False,
    quality_prediction_payload: bool = False,
    cancelled_diagnostic: bool = False,
) -> tuple[bool, str]:
    """Apply the compact default policy to one safe run member."""

    try:
        relative = _relative(run_dir, path)
    except ValueError:
        return False, "outside run"
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts) or "\\" in relative:
        return False, "unsafe relative path"
    if _is_runtime_summary(relative):
        return False, "derived summary requires source-bound export admission"
    reference_diagnostic = is_management_reference_diagnostic(relative)
    if reference_diagnostic:
        try:
            require_safe_pack_source(path, run_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"unsafe reference diagnostic source:{type(exc).__name__}"
    provenance = path.name in PROVENANCE_EVIDENCE_NAMES
    blocked = set(parts[:-1]) & BLOCKED_DIRECTORY_PARTS
    if _is_runtime_diagnostic(relative):
        blocked.discard("lean_bundle")
    if blocked and not provenance:
        return False, "blocked duplicate/data directory"
    if is_candidate_body_mirror(relative):
        return False, "duplicated candidate-body mirror"
    if _is_central_candidate_body(relative) and not (central_request_descriptor or quality_prediction_payload):
        return False, "central-quality replay body"

    suffix = path.suffix.lower()
    probe_root = relative.startswith("reports/window_method_validation_probe/")
    probe_parquet = bool(
        probe_include_raw and probe_root and suffix == ".parquet"
    )
    native_parquet = native_include_raw and _is_native_raw_trace(relative)
    probe_command = bool(probe_root and suffix == ".sh")
    if suffix in BLOCKED_SUFFIXES:
        return False, "blocked binary/figure suffix"
    if suffix not in ALLOWED_SUFFIXES and not (
        probe_parquet or probe_command or native_parquet
    ):
        return False, "suffix not allowed"

    try:
        size = int(path.stat().st_size)
    except OSError as exc:
        return False, f"stat failed:{type(exc).__name__}"
    if relative == MAIN_WORKFLOW_LOG:
        return True, ""
    if reference_diagnostic:
        effective_limit = (MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_FILE_BYTES
                           if suffix == ".json" else min(max(0, int(max_bytes)), MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_FILE_BYTES))
        return (
            (True, "") if size <= effective_limit
            else (False, "management reference diagnostic exceeds compact file limit")
        )
    if native_parquet:
        return (
            (True, "") if size <= NATIVE_RAW_MAX_FILE_BYTES
            else (False, "native energy raw trace exceeds file limit")
        )
    # Cancellation/outage diagnostics are control-plane evidence, not bulk
    # result rows.  Give them the bounded exact-metadata allowance so a tiny
    # caller ``max_bytes`` budget cannot evict the very status/lease records
    # needed to explain why an interrupted run stopped.  The per-file hard
    # limit still prevents an unexpectedly embedded payload from consuming the
    # archive budget.
    if cancelled_diagnostic and is_cancelled_diagnostic_priority(relative):
        return (
            (True, "") if size <= EXACT_METADATA_MAX_FILE_BYTES
            else (
                False,
                "cancelled control-plane evidence exceeds exact limit",
            )
        )
    if is_ranking_audit_evidence(relative):
        return (
            (True, "") if size <= RANKING_AUDIT_MAX_FILE_BYTES
            else (False, "ranking audit evidence exceeds exact limit")
        )
    if is_model_validation_summary(relative):
        effective_limit = MODEL_VALIDATION_SUMMARY_MAX_FILE_BYTES
        return (
            (True, "") if size <= effective_limit
            else (
                False,
                "model validation summary exceeds compact file limit",
            )
        )
    if is_backend_artifact_diagnostic(relative):
        effective_limit = BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES
        return (
            (True, "") if size <= effective_limit
            else (
                False,
                "backend artifact diagnostic exceeds compact file limit",
            )
        )
    if quality_prediction_payload:
        return ((True, "") if size <= STRUCTURED_RESULT_MAX_FILE_BYTES
                else (False, "size_limit_exceeded"))
    if central_request_descriptor:
        return (
            (True, "") if size <= CENTRAL_DESCRIPTOR_MAX_FILE_BYTES
            else (False, "central request descriptor exceeds exact limit")
        )
    if provenance or relative in INCLUDE:
        return (
            (True, "") if size <= EXACT_METADATA_MAX_FILE_BYTES
            else (False, "exact diagnostic metadata exceeds exact limit")
        )
    if _is_runtime_diagnostic(relative) or is_remote_execution_failure_diagnostic(relative):
        effective_limit = (STRUCTURED_RESULT_MAX_FILE_BYTES if suffix in {".json", ".csv"}
                           else max(0, int(max_bytes)))
        return (
            (True, "") if size <= effective_limit
            else (False, "too large")
        )
    if is_offline_replay_core(relative):
        return (
            (True, "") if size <= EXACT_METADATA_MAX_FILE_BYTES
            else (False, "offline summary exceeds exact limit")
        )
    if probe_parquet or probe_command or (
        probe_root and path.name in {
            "window_method_comparison.json", "window_method_validation_probe.json",
        }
    ):
        return (
            (True, "") if size <= PROBE_MAX_FILE_BYTES
            else (False, "window-method probe evidence exceeds exact limit")
        )
    if size > max(0, int(max_bytes)):
        return False, "too large"
    return True, ""


def _known_exact_duplicate_aliases(
    run_dir: Path,
    by_relative: Mapping[str, Path],
    *,
    max_small_file_bytes: int,
) -> list[dict[str, Any]]:
    """Find only the byte-identical aliases whose semantics are known.

    This intentionally is not a content-addressed, archive-wide deduplicator.
    Paths can carry meaning even when their current bytes happen to match, so
    every unrecognised path remains independently archived.  Likewise, a
    known alias is retained when its canonical peer is missing, differs, or
    would not itself be admitted by the compact-pack policy.
    """

    digest_cache: dict[str, str] = {}
    size_cache: dict[str, int | None] = {}

    def _size(relative: str) -> int | None:
        if relative not in size_cache:
            try:
                size_cache[relative] = int(
                    by_relative[relative].stat().st_size
                )
            except OSError:
                size_cache[relative] = None
        return size_cache[relative]

    def _digest(relative: str) -> str:
        if relative not in digest_cache:
            digest_cache[relative] = str(
                sha256_file(by_relative[relative]) or ""
            )
        return digest_cache[relative]

    def _record(
        alias_path: str,
        canonical_path: str,
        *,
        alias_kind: str,
    ) -> dict[str, Any] | None:
        alias = by_relative.get(alias_path)
        canonical = by_relative.get(canonical_path)
        if alias is None or canonical is None:
            return None
        alias_size = _size(alias_path)
        canonical_size = _size(canonical_path)
        if alias_size is None or canonical_size is None:
            return None
        if alias_size != canonical_size:
            return None
        canonical_admitted, _ = should_include_debug_file(
            run_dir,
            canonical,
            max_small_file_bytes,
        )
        if not canonical_admitted:
            return None
        alias_sha256 = _digest(alias_path)
        if not alias_sha256 or alias_sha256 != _digest(canonical_path):
            return None
        return {
            "omitted_path": alias_path,
            "canonical_path": canonical_path,
            "retained_path": canonical_path,
            "size_bytes": alias_size,
            "sha256": alias_sha256,
            "canonical_sha256": alias_sha256,
            "retained_sha256": alias_sha256,
            "match": "byte_identical_sha256",
            "alias_kind": alias_kind,
        }

    aliases: list[dict[str, Any]] = []
    summary_alias = _record(
        "reports/native_producer_combined_summary.json",
        "reports/native_producer_summary.json",
        alias_kind="legacy_combined_native_summary",
    )
    if summary_alias is not None:
        aliases.append(summary_alias)

    binding_alias_paths = sorted(
        relative
        for relative in by_relative
        if (
            len(relative.split("/")) == 4
            and relative.split("/")[0] == "native_producers"
            and relative.split("/")[2] == "quality_first"
            and relative.split("/")[3]
            == "native_split_quality_binding_set.json"
        )
    )
    binding_canonical_paths = sorted(
        relative
        for relative in by_relative
        if (
            len(relative.split("/")) == 4
            and relative.split("/")[:2]
            == ["reports", "native_quality_first"]
            and relative.split("/")[3]
            == "native_split_quality_binding_set.json"
        )
    )
    for alias_path in binding_alias_paths:
        alias_size = _size(alias_path)
        if alias_size is None:
            continue
        matching_canonicals = [
            canonical_path
            for canonical_path in binding_canonical_paths
            if _size(canonical_path) == alias_size
            and _digest(canonical_path) == _digest(alias_path)
        ]
        # Ambiguous equal report copies are retained.  Avoid inventing a
        # canonical path when the run itself does not identify one uniquely.
        if len(matching_canonicals) != 1:
            continue
        binding_alias = _record(
            alias_path,
            matching_canonicals[0],
            alias_kind="remote_native_split_quality_binding_mirror",
        )
        if binding_alias is not None:
            aliases.append(binding_alias)
    return aliases


def _confined_member(
    run_dir: Path, raw_path: object
) -> tuple[Path | None, str, str]:
    text = str(raw_path or "").strip()
    if not text:
        return None, "", "path_missing"
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(run_dir).as_posix()
    except ValueError:
        return None, "", "path_outside_run"
    try:
        return require_safe_pack_source(lexical, run_dir), relative, ""
    except FileNotFoundError:
        return None, relative, "source_missing"
    except (OSError, RuntimeError, ValueError) as exc:
        return None, relative, f"unsafe_or_unreadable:{type(exc).__name__}"


def _descriptor_reference_summary(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in ("candidate", "reference", "annotations"):
        value = payload.get(kind)
        if not value and kind == "annotations":
            value = payload.get("annotations_file")
        if isinstance(value, Mapping):
            rows.append({
                "kind": kind,
                "declared_path": str(
                    value.get("path") or value.get("file") or value.get("source") or ""
                ),
                "declared_sha256": str(
                    value.get("sha256") or value.get("file_sha256") or ""
                ),
                "declared_size_bytes": value.get("size_bytes"),
                "archived": False,
                "reason": "replay_payload_excluded_from_compact_debug_pack",
            })
        elif value:
            rows.append({
                "kind": kind,
                "declared_path": str(value),
                "declared_sha256": "",
                "declared_size_bytes": None,
                "archived": False,
                "reason": "replay_payload_excluded_from_compact_debug_pack",
            })
    return rows


def _discover_quality_prediction_payloads(root: Path, central: Mapping[str, Any]) -> dict[str, Any]:
    """Admit only declared decoded JSON bodies, one file at a time.

    These are existing prediction/annotation records, never raw tensor dumps.
    File identities come from the producing request/reference result, not a new
    registry. Missing historical bodies are reported without preventing logs
    and the remaining evidence from being exported.
    """
    references = [dict(row) for row in central.get("referenced_replay_payloads", [])]
    for item in central.get("management_reference_payloads", []):
        references.append(dict(item))
    files: dict[str, dict[str, Any]] = {}
    omitted: list[dict[str, Any]] = []
    total = 0
    for row in references:
        raw = str(row.get("declared_path") or "")
        # management_cpu_reference is a semantic source marker, not a filename.
        if not raw or (row.get("kind") == "reference" and raw == "management_cpu_reference"):
            continue
        expected_sha = _sha_token(row.get("declared_sha256"))
        if not expected_sha:
            omitted.append({**row, "path": raw, "reason": "declared_file_sha256_missing", "archived": False})
            continue
        declared = Path(raw)
        if declared.is_absolute():
            source, relative, reason = _confined_member(root, declared)
        else:
            # Normal requests use candidate.json beside the request; reference
            # results can instead name a canonical run-relative path.
            relative_base = Path(str(row.get("request_path") or "")).parent
            source, relative, reason = _confined_member(root, relative_base / declared)
            if source is None and reason == "source_missing":
                source, relative, reason = _confined_member(root, declared)
        record = {**row, "path": relative, "archived": False}
        if source is None:
            omitted.append({**record, "reason": reason})
            continue
        if source.suffix.lower() != ".json" or not _is_central_candidate_body(relative):
            omitted.append({**record, "reason": "not_decoded_quality_json"})
            continue
        if relative in files:
            if _sha_token(files[relative]["sha256"]) != expected_sha:
                omitted.append({**record, "reason": "conflicting_declared_sha256"})
            continue
        size = source.stat().st_size
        if size > STRUCTURED_RESULT_MAX_FILE_BYTES:
            omitted.append({**record, "reason": "size_limit_exceeded", "size_bytes": size,
                            "limit_bytes": STRUCTURED_RESULT_MAX_FILE_BYTES})
            continue
        if total + size > QUALITY_PREDICTION_MAX_TOTAL_BYTES:
            omitted.append({**record, "reason": "size_limit_exceeded", "size_bytes": size,
                            "observed_total_bytes": total + size,
                            "limit_bytes": QUALITY_PREDICTION_MAX_TOTAL_BYTES, "limit_scope": "quality_predictions_total"})
            continue
        try:
            if row.get("declared_size_bytes") is not None and int(row["declared_size_bytes"]) != size:
                raise ValueError("declared_size_mismatch")
            digest = str(sha256_file(source) or "")
            if _sha_token(digest) != expected_sha:
                raise ValueError("declared_hash_mismatch")
            payload = _read_compact_json(source, root, STRUCTURED_RESULT_MAX_FILE_BYTES)
            if not isinstance(payload, Mapping) or not any(isinstance(payload.get(key), list) for key in ("records", "images", "annotations")):
                raise ValueError("decoded_quality_schema_invalid")
            del payload  # Do not retain or duplicate the entire prediction population.
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            reason = "json_invalid" if isinstance(exc, (json.JSONDecodeError, UnicodeError)) else str(exc)
            omitted.append({**record, "reason": reason, "size_bytes": size,
                            **(exc.details if isinstance(exc, DebugSizeLimit) else {})})
            continue
        files[relative] = {**record, "sha256": digest, "size_bytes": size, "reason": "validated_declared_decoded_json"}
        total += size
    return {"files": list(files.values()), "admitted_members": sorted(files),
            "omitted": omitted, "total_size_bytes": total,
            "max_file_bytes": STRUCTURED_RESULT_MAX_FILE_BYTES,
            "max_total_bytes": QUALITY_PREDICTION_MAX_TOTAL_BYTES,
            "budget_unit": "uncompressed_bytes"}


def discover_central_request_descriptors(run_dir: Path) -> dict[str, Any]:
    """Discover request JSONs without following their replay-body references."""

    summary_path = run_dir / "quality_management" / "central_quality_summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        return {
            "summary_present": False,
            "expected_members": [],
            "present_source_members": [],
            "missing_source_members": [],
            "files": [],
            "referenced_replay_payloads": [],
            "failures": [],
            "oversized": [],
            "total_size_bytes": 0,
            "source_contract_ok": True,
        }
    try:
        safe_summary = require_safe_pack_source(summary_path, run_dir)
        summary = _read_compact_json(safe_summary, run_dir, EXACT_METADATA_MAX_FILE_BYTES)
    except Exception as exc:
        return {
            "summary_present": True,
            "expected_members": [],
            "present_source_members": [],
            "missing_source_members": [],
            "files": [],
            "referenced_replay_payloads": [],
            "failures": [{
                "path": "quality_management/central_quality_summary.json",
                "reason": ("size_limit_exceeded" if isinstance(exc, DebugSizeLimit) else f"summary_invalid:{type(exc).__name__}"),
                **(exc.details if isinstance(exc, DebugSizeLimit) else {}),
            }],
            "oversized": [],
            "total_size_bytes": 0,
            "source_contract_ok": False,
        }
    results = summary.get("results") if isinstance(summary, Mapping) else None
    if not isinstance(results, list):
        results = []

    expected: set[str] = set()
    present: set[str] = set()
    missing: set[str] = set()
    records_by_path: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    management_references: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    oversized: list[dict[str, Any]] = []
    total = 0
    if len(results) > CENTRAL_DESCRIPTOR_MAX_REQUESTS:
        failures.append({
            "path": "quality_management/central_quality_summary.json",
            "reason": "request_count_exceeds_limit",
            "count": len(results),
            "limit": CENTRAL_DESCRIPTOR_MAX_REQUESTS,
        })
        oversized.append({
            "path": "<central_request_descriptor_count>",
            "count": len(results),
            "limit": CENTRAL_DESCRIPTOR_MAX_REQUESTS,
        })

    for index, result in enumerate(results[:CENTRAL_DESCRIPTOR_MAX_REQUESTS]):
        if not isinstance(result, Mapping):
            failures.append({"result_index": index, "reason": "result_not_object"})
            continue
        reference = result.get("management_cpu_reference")
        if isinstance(reference, Mapping) and reference.get("reference_path"):
            management_references.append({
                "kind": "reference", "request_path": "",
                "declared_path": str(reference["reference_path"]),
                "declared_sha256": reference.get("reference_sha256", ""),
                "declared_size_bytes": reference.get("reference_size_bytes"),
            })
        source, relative, reason = _confined_member(
            run_dir, result.get("source_request")
        )
        if relative:
            expected.add(relative)
        if source is None:
            if relative:
                missing.add(relative)
            failures.append({
                "result_index": index,
                "path": relative,
                "reason": reason,
            })
            continue
        if relative in records_by_path:
            previous = records_by_path[relative]
            previous["result_indexes"].append(index)
            declared = _sha_token(result.get("source_request_sha256"))
            if declared and _sha_token(previous["sha256"]) != declared:
                failures.append({"path": relative, "result_index": index,
                                 "reason": "declared_hash_mismatch"})
            elif result.get("source_request_sha256") and not declared:
                failures.append({"path": relative, "result_index": index,
                                 "reason": "declared_hash_invalid"})
            continue
        size = int(source.stat().st_size)
        digest = str(sha256_file(source) or "") if size <= CENTRAL_DESCRIPTOR_MAX_FILE_BYTES else ""
        declared = _sha_token(result.get("source_request_sha256"))
        matches = None if not declared else digest.removeprefix("sha256:") == declared
        if result.get("source_request_sha256") and not declared:
            matches = False
            failures.append({"path": relative, "result_index": index,
                             "reason": "declared_hash_invalid"})
        record = {
            "path": relative,
            "size_bytes": size,
            "sha256": digest,
            "declared_sha256": ("sha256:" + declared) if declared else "",
            "declared_hash_matches": matches,
            "result_indexes": [index],
        }
        records_by_path[relative] = record
        present.add(relative)
        total += size
        if size > CENTRAL_DESCRIPTOR_MAX_FILE_BYTES:
            oversized.append({
                "path": relative,
                "size_bytes": size,
                "limit_bytes": CENTRAL_DESCRIPTOR_MAX_FILE_BYTES,
                "reason": "size_limit_exceeded",
            })
            continue
        if total > CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES:
            continue  # Count remaining sizes, but never parse beyond the aggregate budget.
        try:
            request = _read_compact_json(source, run_dir, CENTRAL_DESCRIPTOR_MAX_FILE_BYTES)
            if isinstance(request, Mapping) and matches is not False:
                references.extend(
                    {"request_path": relative, **row}
                    for row in _descriptor_reference_summary(request)
                )
            else:
                failures.append({"path": relative, "reason": "request_not_object"})
        except Exception as exc:
            failures.append({
                "path": relative,
                "reason": ("size_limit_exceeded" if isinstance(exc, DebugSizeLimit) else f"request_invalid:{type(exc).__name__}"),
                **(exc.details if isinstance(exc, DebugSizeLimit) else {}),
            })
        if matches is False:
            failures.append({"path": relative, "reason": "declared_hash_mismatch"})
    if total > CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES:
        oversized.append({
            "path": "<central_request_descriptors_total>",
            "size_bytes": total,
            "limit_bytes": CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES,
            "reason": "size_limit_exceeded",
        })
    return {
        "summary_present": True,
        "expected_members": sorted(expected),
        "present_source_members": sorted(present),
        "missing_source_members": sorted(missing),
        "files": [records_by_path[key] for key in sorted(records_by_path)],
        "referenced_replay_payloads": references,
        "management_reference_payloads": management_references,
        "failures": failures,
        "oversized": oversized,
        "total_size_bytes": total,
        "source_contract_ok": bool(not missing and not failures and not oversized),
    }


def _diagnostic_kind(relative: str, *, central_descriptors: set[str]) -> str:
    if is_management_reference_diagnostic(relative):
        return "management_cpu_reference_diagnostic"
    if _is_native_raw_trace(relative):
        return "native_energy_raw_trace"
    if relative == MAIN_WORKFLOW_LOG:
        return "main_workflow_log"
    if relative == MAIN_WORKFLOW_TAIL:
        return "main_workflow_log_tail"
    if _is_runtime_diagnostic(relative):
        return "runtime_execution_diagnostic"
    if is_ranking_audit_evidence(relative):
        return "ranking_audit_evidence"
    if is_model_validation_summary(relative):
        return "model_validation_summary"
    if is_backend_artifact_diagnostic(relative):
        return "backend_artifact_diagnostic"
    if is_remote_execution_failure_diagnostic(relative):
        return "remote_execution_failure_diagnostic"
    if is_cancelled_diagnostic_priority(relative):
        return "cancelled_control_plane"
    if relative in central_descriptors:
        return "central_quality_request_descriptor"
    if Path(relative).name in PROVENANCE_EVIDENCE_NAMES:
        return "execution_provenance_evidence"
    if is_offline_replay_core(relative):
        return "offline_replay_core"
    if relative.startswith("reports/window_method_validation_probe/"):
        if relative.endswith(".parquet"):
            return "window_probe_raw_trace"
        if relative.endswith(".sh"):
            return "window_probe_command"
        if Path(relative).name == "window_method_comparison.json":
            return "window_method_A/B_comparison"
        return "window_probe_evidence"
    if "/analysis_tables/" in relative:
        return "native_analysis"
    return "diagnostic_metadata"


def _write_source_member(
    archive: zipfile.ZipFile,
    source: Path,
    relative: str,
    *,
    run_dir: Path,
    diagnostic_kind: str,
) -> dict[str, Any]:
    source = require_safe_pack_source(source, run_dir)
    info = zipinfo_for_path(source, relative, compress_type=zipfile.ZIP_DEFLATED)
    digest = hashlib.sha256()
    size = 0
    initial_size = source.stat().st_size
    with source.open("rb") as src, archive.open(info, "w") as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            size += len(block)
            if size > initial_size:
                raise RuntimeError("source_grew_during_export:" + relative)
            dst.write(block)
            digest.update(block)
    if size != initial_size:
        raise RuntimeError("source_shrank_during_export:" + relative)
    return {
        "path": relative,
        "size_bytes": size,
        "sha256": "sha256:" + digest.hexdigest(),
        "diagnostic_kind": diagnostic_kind,
    }


def _tail_payload(path: Path, max_bytes: int) -> bytes:
    size = int(path.stat().st_size)
    keep = max(1, int(max_bytes))
    with path.open("rb") as handle:
        if size > keep:
            handle.seek(size - keep)
        payload = handle.read(keep)
    prefix = (
        f"[tail only: last {len(payload)} of {size} bytes from "
        f"{path.name}]\n"
    ).encode("utf-8")
    return prefix + payload


def _default_output(run_dir: Path) -> Path:
    configured = str(os.environ.get("ONNX_SPLITPOINT_EXPORT_DIR", "") or "").strip()
    root = Path(configured).expanduser() if configured else Path.home() / "Downloads"
    return root / f"{run_dir.name}_debug_pack.zip"


def _pack_source_identity(
    run_dir: Path,
    *,
    tool_build: Mapping[str, Any],
    selection_policy: str,
) -> dict[str, Any]:
    try:
        manifest_path = require_safe_pack_source(
            run_dir / "run_manifest.json", run_dir
        )
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(run_manifest, Mapping):
            run_manifest = {}
    except Exception:
        run_manifest = {}
    return {
        "schema": "onnx-splitpoint/pack-source-identity",
        "schema_version": 1,
        "run_id": str(run_manifest.get("run_id") or run_dir.name),
        "run_dir": str(run_dir),
        "tool_version_recorded": str(run_manifest.get("tool_version") or ""),
        "workflow_version_recorded": str(
            run_manifest.get("workflow_version") or ""
        ),
        "run_status_recorded": str(run_manifest.get("status") or ""),
        "pack_tool_version": str(TOOL_VERSION),
        "pack_tool_build": dict(tool_build),
        "selection_policy": str(selection_policy or "explicit_evaluation_run_argument"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def create_evaluation_debug_pack(
    run_dir: str | Path,
    out_zip: str | Path | None = None,
    *,
    max_small_file_bytes: int = DEFAULT_MAX_SMALL_FILE_BYTES,
    tail_bytes: int = DEFAULT_TAIL_BYTES,
    source_selection_policy: str = "explicit_evaluation_run_argument",
) -> dict[str, Any]:
    """Create, verify and atomically publish one compact diagnostic ZIP.

    Missing source evidence is represented in the manifest and never prevents
    an interrupted run from being packed.  Conversely, every expected source
    that *does* exist is required in the closed ZIP; admission or write failure
    therefore aborts publication rather than silently weakening the pack.
    """

    raw_run = Path(run_dir).expanduser()
    if raw_run.is_symlink():
        raise RuntimeError(f"EvaluationRun source must not be a symlink: {raw_run}")
    root = raw_run.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    destination = Path(out_zip).expanduser() if out_zip else _default_output(root)
    destination = destination.resolve(strict=False)
    require_output_outside_source(
        root, destination, operation="EvaluationRun debug pack export"
    )
    require_write_target(
        destination.parent,
        operation="EvaluationRun debug pack export",
        minimum_free_bytes=16 * 1024 * 1024,
        minimum_free_inodes=16,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cancelled_diagnostic_mode = str(source_selection_policy).startswith(
        "cancelled_diagnostic"
    )

    audit = discover_ranking_audit_evidence(root)
    central = discover_central_request_descriptors(root)
    quality_payloads = _discover_quality_prediction_payloads(root, central)
    quality_payload_members = set(quality_payloads["admitted_members"])
    # Budget uses uncompressed source bytes; the actual compressed ZIP size is
    # reported separately after publication. This conservatively admits disk.
    require_write_target(destination.parent, operation="EvaluationRun quality evidence export",
                         minimum_free_bytes=max(16 * 1024 * 1024,
                             int(quality_payloads["total_size_bytes"]) + int(central.get("total_size_bytes", 0))),
                         minimum_free_inodes=16)
    if audit.get("oversized"):
        raise RuntimeError(
            "ranking_audit_evidence_not_publishable:oversized="
            + json.dumps(audit["oversized"], sort_keys=True)
        )
    if audit.get("unpublishable"):
        raise RuntimeError(
            "ranking_audit_evidence_not_publishable:existing_source_error="
            + json.dumps(audit["unpublishable"], sort_keys=True)
        )
    if central.get("oversized"):
        raise RuntimeError(
            "central_quality_request_descriptors_not_publishable:oversized="
            + json.dumps(central["oversized"], sort_keys=True)
        )

    try:
        all_files = iter_safe_pack_files(root, root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"EvaluationRun cannot be enumerated safely: {type(exc).__name__}: {exc}"
        ) from exc
    by_relative = {_relative(root, path): path for path in all_files}
    probe_include_raw = _probe_include_raw(root)
    native_include_raw = _native_include_raw(root)
    native_raw = _native_raw_inventory(
        root, by_relative, requested=native_include_raw,
    )
    energy_attempts = _energy_attempt_log_inventory(root, by_relative)
    reference_diagnostics = _management_reference_diagnostic_inventory(
        root, by_relative, max_bytes=max_small_file_bytes,
    )
    runtime_diagnostics = _runtime_diagnostic_inventory(
        root, by_relative, max_bytes=max_small_file_bytes,
        index_records=reference_diagnostics.pop("_runtime_source_records"),
    )
    runtime_present = set(runtime_diagnostics["present_source_members"])
    reference_admitted = set(reference_diagnostics["admitted_source_members"])
    exact_duplicate_aliases = _known_exact_duplicate_aliases(
        root,
        by_relative,
        max_small_file_bytes=max_small_file_bytes,
    )
    exact_duplicate_aliases_by_path = {
        str(row["omitted_path"]): row for row in exact_duplicate_aliases
    }
    alias_canonical_members = {
        str(row["canonical_path"]) for row in exact_duplicate_aliases
    }
    central_present = set(central.get("present_source_members") or [])
    audit_present = set(audit.get("present_source_members") or [])
    audit_present.update(audit.get("optional_present_source_members") or [])
    provenance_present = {
        relative for relative in by_relative
        if Path(relative).name in PROVENANCE_EVIDENCE_NAMES
    }
    offline_present = set(OFFLINE_REPLAY_CORE) & set(by_relative)
    cancelled_priority_present = (
        {
            relative for relative in by_relative
            if is_cancelled_diagnostic_priority(relative)
        }
        if cancelled_diagnostic_mode
        else set()
    )
    model_validation_present = sorted(
        relative
        for relative in by_relative
        if is_model_validation_summary(relative)
    )
    model_validation_admitted: list[str] = []
    model_validation_omitted: list[dict[str, Any]] = []
    model_validation_total_bytes = 0
    for relative in model_validation_present:
        source = by_relative[relative]
        include, reason = should_include_debug_file(
            root,
            source,
            max_small_file_bytes,
            cancelled_diagnostic=cancelled_diagnostic_mode,
        )
        source_size = int(source.stat().st_size)
        if not include:
            model_validation_omitted.append({
                "path": relative,
                "reason": reason,
                "size_bytes": source_size,
            })
            continue
        if (
            model_validation_total_bytes + source_size
            > MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES
        ):
            model_validation_omitted.append({
                "path": relative,
                "reason": "model validation summary total limit exceeded",
                "size_bytes": source_size,
                "limit_bytes": MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES,
            })
            continue
        model_validation_admitted.append(relative)
        model_validation_total_bytes += source_size
    model_validation_admitted_set = set(model_validation_admitted)
    backend_artifact_diagnostic_present = sorted(
        relative
        for relative in by_relative
        if is_backend_artifact_diagnostic(relative)
    )
    backend_artifact_diagnostic_admitted: list[str] = []
    backend_artifact_diagnostic_omitted: list[dict[str, Any]] = []
    backend_artifact_diagnostic_total_bytes = 0
    for relative in backend_artifact_diagnostic_present:
        source = by_relative[relative]
        include, reason = should_include_debug_file(
            root,
            source,
            max_small_file_bytes,
            cancelled_diagnostic=cancelled_diagnostic_mode,
        )
        source_size = int(source.stat().st_size)
        if not include:
            backend_artifact_diagnostic_omitted.append({
                "path": relative,
                "reason": reason,
                "size_bytes": source_size,
            })
            continue
        if (
            cancelled_diagnostic_mode
            and is_cancelled_diagnostic_priority(relative)
        ):
            backend_artifact_diagnostic_admitted.append(relative)
            backend_artifact_diagnostic_total_bytes += source_size
            continue
        if (
            backend_artifact_diagnostic_total_bytes + source_size
            > BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES
        ):
            backend_artifact_diagnostic_omitted.append({
                "path": relative,
                "reason": "backend artifact diagnostic total limit exceeded",
                "size_bytes": source_size,
                "limit_bytes": BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES,
            })
            continue
        backend_artifact_diagnostic_admitted.append(relative)
        backend_artifact_diagnostic_total_bytes += source_size
    backend_artifact_diagnostic_admitted_set = set(
        backend_artifact_diagnostic_admitted
    )
    main_present = MAIN_WORKFLOW_LOG in by_relative
    tail_fallback_present = bool(
        not main_present and MAIN_WORKFLOW_TAIL in by_relative
    )
    critical_members = set(central_present)
    critical_members.update(quality_payload_members)
    critical_members.update(audit_present)
    critical_members.update(provenance_present)
    critical_members.update(offline_present)
    critical_members.update(cancelled_priority_present)
    critical_members.update(model_validation_admitted)
    critical_members.update(backend_artifact_diagnostic_admitted)
    critical_members.update(alias_canonical_members)
    critical_members.update(reference_admitted)
    if main_present:
        critical_members.add(MAIN_WORKFLOW_LOG)
    elif tail_fallback_present:
        critical_members.add(MAIN_WORKFLOW_TAIL)

    exact_candidates = set(INCLUDE)
    exact_candidates.update(critical_members)
    # Enumerate every canonical summary for an explicit archived/omitted
    # decision.  The precomputed admission set remains authoritative below,
    # even if ``models`` is added to the broad scanner in a future release.
    exact_candidates.update(model_validation_present)
    exact_candidates.update(backend_artifact_diagnostic_present)
    exact_candidates.update(runtime_present)
    exact_candidates.update(
        relative
        for relative in by_relative
        if is_remote_execution_failure_diagnostic(relative)
    )
    candidates: list[str] = sorted(exact_candidates)
    for relative in sorted(by_relative):
        parts = relative.split("/")
        if parts and parts[0] in SCANNED_TOP_LEVELS:
            candidates.append(relative)
        elif Path(relative).name in PROVENANCE_EVIDENCE_NAMES:
            candidates.append(relative)

    tool_build = package_build_snapshot()
    source_identity = _pack_source_identity(
        root,
        tool_build=tool_build,
        selection_policy=source_selection_policy,
    )
    temporary = temporary_zip_path(destination)
    written: list[dict[str, Any]] = []
    written_paths: set[str] = set()
    skipped: list[dict[str, Any]] = [
        *quality_payloads["omitted"],
        *model_validation_omitted,
        *backend_artifact_diagnostic_omitted,
        *reference_diagnostics["omitted_source_members"],
    ]
    tail_sources: list[tuple[str, Path]] = []
    required_members = {
        "debug_pack_manifest.json", "pack_source_identity.json", *critical_members
    }
    verification: dict[str, Any]

    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            source_identity_bytes = (
                json.dumps(
                    source_identity,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "\n"
            ).encode("utf-8")
            archive.writestr("pack_source_identity.json", source_identity_bytes)
            written.append({
                "path": "pack_source_identity.json",
                "size_bytes": len(source_identity_bytes),
                "sha256": (
                    "sha256:" + hashlib.sha256(source_identity_bytes).hexdigest()
                ),
                "diagnostic_kind": "pack_source_identity",
            })
            written_paths.add("pack_source_identity.json")
            if tail_fallback_present:
                tail_source = by_relative[MAIN_WORKFLOW_TAIL]
                source_size = int(tail_source.stat().st_size)
                if source_size > max(1, int(tail_bytes)):
                    payload = _tail_payload(tail_source, tail_bytes)
                    archive.writestr(MAIN_WORKFLOW_TAIL, payload)
                    written.append({
                        "path": MAIN_WORKFLOW_TAIL,
                        "source_path": MAIN_WORKFLOW_TAIL,
                        "source_size_bytes": source_size,
                        "size_bytes": len(payload),
                        "sha256": (
                            "sha256:" + hashlib.sha256(payload).hexdigest()
                        ),
                        "diagnostic_kind": "main_workflow_log_tail",
                    })
                    written_paths.add(MAIN_WORKFLOW_TAIL)
                else:
                    written.append(_write_source_member(
                        archive,
                        tail_source,
                        MAIN_WORKFLOW_TAIL,
                        run_dir=root,
                        diagnostic_kind="main_workflow_log_tail",
                    ))
                    written_paths.add(MAIN_WORKFLOW_TAIL)
            probe_raw_total_bytes = 0
            native_raw_total_bytes = 0
            for relative in dict.fromkeys(candidates):
                source = by_relative.get(relative)
                if source is None or relative in written_paths:
                    continue
                if is_management_reference_diagnostic(relative) and relative not in reference_admitted:
                    continue  # Its exact bounded-admission reason is already inventoried.
                if (
                    is_model_validation_summary(relative)
                    and relative not in model_validation_admitted_set
                ):
                    # Its one bounded-policy reason was recorded before the
                    # archive loop.  Do not re-admit or duplicate that record.
                    continue
                if (
                    is_backend_artifact_diagnostic(relative)
                    and relative
                    not in backend_artifact_diagnostic_admitted_set
                ):
                    # This exact control-plane JSON already has one bounded
                    # omission record from the precomputed aggregate policy.
                    continue
                alias_record = exact_duplicate_aliases_by_path.get(relative)
                if alias_record is not None:
                    skipped.append({
                        "path": relative,
                        "reason": "byte-identical known alias omitted",
                        "canonical_path": alias_record["canonical_path"],
                        "sha256": alias_record["sha256"],
                        "size_bytes": alias_record["size_bytes"],
                    })
                    continue
                include, reason = should_include_debug_file(
                    root,
                    source,
                    max_small_file_bytes,
                    probe_include_raw=probe_include_raw,
                    native_include_raw=native_include_raw,
                    central_request_descriptor=relative in central_present,
                    quality_prediction_payload=relative in quality_payload_members,
                    cancelled_diagnostic=cancelled_diagnostic_mode,
                )
                if not include:
                    effective_limit = (CENTRAL_DESCRIPTOR_MAX_FILE_BYTES if relative in central_present
                                       else STRUCTURED_RESULT_MAX_FILE_BYTES if (
                                           relative in quality_payload_members or (_is_runtime_diagnostic(relative) and source.suffix.lower() in {".json", ".csv"})
                                           or relative in INCLUDE or source.name in PROVENANCE_EVIDENCE_NAMES or is_offline_replay_core(relative)
                                       ) else max(0, int(max_small_file_bytes)))
                    skipped.append({"path": relative, "reason": reason,
                                    "size_bytes": int(source.stat().st_size), "limit_bytes": effective_limit})
                    if relative in critical_members:
                        raise RuntimeError(
                            "required_debug_member_not_publishable:"
                            f"{relative}:{reason}"
                        )
                    if (
                        reason == "too large"
                        and source.suffix.lower() in {".log", ".jsonl", ".txt"}
                        and relative != MAIN_WORKFLOW_LOG
                    ):
                        tail_sources.append((relative, source))
                    continue
                diagnostic_kind = ("decoded_quality_prediction" if relative in quality_payload_members else _diagnostic_kind(
                    relative, central_descriptors=central_present
                ))
                if diagnostic_kind == "native_energy_raw_trace":
                    source_size = int(source.stat().st_size)
                    if native_raw_total_bytes + source_size > NATIVE_RAW_MAX_TOTAL_BYTES:
                        skipped.append({
                            "path": relative,
                            "reason": "native energy raw total limit exceeded",
                            "size_bytes": source_size,
                            "limit_bytes": NATIVE_RAW_MAX_TOTAL_BYTES,
                        })
                        continue
                if diagnostic_kind == "window_probe_raw_trace":
                    source_size = int(source.stat().st_size)
                    if (
                        probe_raw_total_bytes + source_size
                        > PROBE_MAX_TOTAL_BYTES
                    ):
                        skipped.append({
                            "path": relative,
                            "reason": "window probe raw total limit exceeded",
                            "size_bytes": source_size,
                            "limit_bytes": PROBE_MAX_TOTAL_BYTES,
                        })
                        continue
                # An unreadable optional source can be inventoried before an
                # archive member exists. Once writing starts, every exception
                # aborts publication: ZIP cannot roll back a partial member.
                try:
                    with source.open("rb") as readability:
                        readability.read(1)
                except OSError as exc:
                    if relative in critical_members:
                        raise RuntimeError(f"required_debug_member_read_failed:{relative}:{exc}") from exc
                    skipped.append({"path": relative, "reason": f"source_unreadable:{type(exc).__name__}:{exc}"})
                    continue
                try:
                    record = _write_source_member(
                        archive,
                        source,
                        relative,
                        run_dir=root,
                        diagnostic_kind=diagnostic_kind,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "required_debug_member_write_failed:"
                        f"{relative}:{type(exc).__name__}:{exc}"
                    ) from exc
                written.append(record)
                written_paths.add(relative)
                if diagnostic_kind == "window_probe_raw_trace":
                    probe_raw_total_bytes += int(record.get("size_bytes") or 0)
                if diagnostic_kind == "native_energy_raw_trace":
                    native_raw_total_bytes += int(record.get("size_bytes") or 0)

            tail_total = 0
            for relative, source in tail_sources:
                if len([row for row in written if row.get("diagnostic_kind") == "bounded_log_tail"]) >= MAX_TAIL_MEMBERS:
                    break
                payload = _tail_payload(source, tail_bytes)
                if tail_total + len(payload) > MAX_TAIL_TOTAL_BYTES:
                    break
                tail_name = f"debug_tails/{relative}.tail"
                if tail_name in written_paths:
                    continue
                archive.writestr(tail_name, payload)
                written.append({
                    "path": tail_name,
                    "source_path": relative,
                    "size_bytes": len(payload),
                    "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
                    "diagnostic_kind": "bounded_log_tail",
                })
                written_paths.add(tail_name)
                tail_total += len(payload)

            runtime_coverage, derived_summaries = _runtime_compact_export(
                root, runtime_diagnostics, written_paths, skipped,
                max_bytes=max_small_file_bytes,
            )
            from .compact_runtime_diagnostics import (
                FALLBACK_PARSE_MAX_BYTES, SUMMARY_MAX_BYTES, SUMMARY_TOTAL_MAX_BYTES,
                source_observation,
            )
            for summary in derived_summaries:
                relative = summary["path"]
                source = require_safe_pack_source(root / summary["source_path"], root)
                if source_observation(source) != summary["source_observation"]:
                    raise RuntimeError("runtime_summary_source_changed_before_archive:" + summary["source_path"])
                archive.writestr(relative, summary["body"])
                written.append({
                    "path": relative, "source_path": summary["source_path"],
                    "size_bytes": len(summary["body"]),
                    "sha256": "sha256:" + hashlib.sha256(summary["body"]).hexdigest(),
                    "diagnostic_kind": "derived_runtime_summary",
                })
                written_paths.add(relative)
                required_members.add(relative)
                summary["coverage"].update(derived_summary_archived=True,
                                          summary_status="derived_summary_archived")
                if source_observation(source) != summary["source_observation"]:
                    raise RuntimeError("runtime_summary_source_changed_during_archive:" + summary["source_path"])

            missing_required_archive = sorted(critical_members - written_paths)
            if missing_required_archive:
                raise RuntimeError(
                    "required_debug_members_not_written:"
                    + ",".join(missing_required_archive)
                )

            by_written = {str(row["path"]): row for row in written}
            for original in [*central.get("files", []), *quality_payloads["files"]]:
                actual = by_written.get(original["path"], {})
                if actual.get("sha256") != original.get("sha256") or actual.get("size_bytes") != original.get("size_bytes"):
                    raise RuntimeError("declared_quality_source_changed_during_export:" + original["path"])
            audit_archived = sorted(audit_present & written_paths)
            central_archived = sorted(central_present & written_paths)
            provenance_archived = sorted(provenance_present & written_paths)
            offline_archived = sorted(offline_present & written_paths)
            model_validation_archived = sorted(
                set(model_validation_admitted) & written_paths
            )
            backend_artifact_diagnostic_archived = sorted(
                set(backend_artifact_diagnostic_admitted) & written_paths
            )
            cancelled_priority_archived = sorted(
                cancelled_priority_present & written_paths
            )
            main_record = by_written.get(MAIN_WORKFLOW_LOG, {})
            tail_record = by_written.get(MAIN_WORKFLOW_TAIL, {})
            exact_duplicate_alias_records: list[dict[str, Any]] = []
            for alias_record in exact_duplicate_aliases:
                canonical_path = str(alias_record["canonical_path"])
                canonical_record = by_written.get(canonical_path, {})
                canonical_archive_sha256 = str(
                    canonical_record.get("sha256") or ""
                )
                canonical_hash_verified = bool(
                    canonical_archive_sha256
                    and canonical_archive_sha256 == alias_record["sha256"]
                )
                exact_duplicate_alias_records.append({
                    **alias_record,
                    "canonical_archived": canonical_path in written_paths,
                    "canonical_archive_sha256": canonical_archive_sha256,
                    "canonical_hash_verified": canonical_hash_verified,
                })
            if not all(
                row["canonical_archived"] and row["canonical_hash_verified"]
                for row in exact_duplicate_alias_records
            ):
                raise RuntimeError(
                    "exact_duplicate_alias_canonical_verification_failed"
                )
            audit_missing = list(audit.get("missing_source_members") or [])
            central_missing = list(central.get("missing_source_members") or [])
            audit_complete = bool(
                (not audit.get("enabled"))
                or (
                    not audit_missing
                    and not audit.get("oversized")
                    and not audit.get("unpublishable")
                    and audit_present <= written_paths
                )
            )
            native_raw_archived = sorted(
                set(native_raw["present_source_members"]) & written_paths
            )
            native_raw_omitted = [
                row for row in skipped if _is_native_raw_trace(str(row["path"]))
            ] + [
                {"path": path, "reason": "source_missing"}
                for path in native_raw["missing_source_members"]
            ]
            native_raw_complete = bool(
                not native_include_raw or (
                    not native_raw_omitted and not native_raw["reference_failures"]
                )
            )
            runtime_omitted = [
                row for row in skipped if row["path"] in runtime_present
            ]
            runtime_complete = bool(
                not runtime_diagnostics["missing_source_members"]
                and not runtime_diagnostics["reference_failures"]
                and runtime_present <= written_paths
            )
            reference_archived = sorted(reference_admitted & written_paths)
            reference_complete = bool(
                not reference_diagnostics["omitted_source_members"]
                and not reference_diagnostics["reference_failures"]
                and set(reference_diagnostics["expected_members"]) <= written_paths
                and (reference_diagnostics["index_validation"]["index_coverage_verified"]
                     or (not reference_diagnostics["expected_members"]
                         and reference_diagnostics["index_validation"]["status"] == "missing"))
            )
            manifest: dict[str, Any] = {
                "schema": "onnx-splitpoint/evaluation-debug-pack-manifest",
                "schema_version": DEBUG_PACK_SCHEMA_VERSION,
                "tool_version": TOOL_VERSION,
                "tool_build": tool_build,
                "run_id": root.name,
                "run_dir": str(root),
                "source_identity": source_identity,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "complete": bool(
                    main_present
                    and audit_complete
                    and not central_missing
                    and not central.get("failures")
                    and not quality_payloads["omitted"]
                    and native_raw_complete
                    and runtime_complete
                    and reference_complete
                    and not energy_attempts["missing_source_members"]
                    and not energy_attempts["reference_failures"]
                    and set(energy_attempts["expected_members"]) <= written_paths
                ),
                "archive_publication": {
                    "mode": "verified_temporary_then_atomic_replace",
                    "verification": "central_directory_and_all_member_crc",
                    "final_identity_manifest": destination.name + ".manifest.json",
                },
                "policy": {
                    "purpose": "compact diagnostic and audit-evidence pack",
                    "max_small_file_bytes": int(max_small_file_bytes),
                    "structured_result_max_file_bytes": STRUCTURED_RESULT_MAX_FILE_BYTES,
                    "structured_result_max_total_bytes_per_class": STRUCTURED_RESULT_MAX_TOTAL_BYTES,
                    "decoded_quality_prediction_max_total_bytes": QUALITY_PREDICTION_MAX_TOTAL_BYTES,
                    "budget_unit": "uncompressed_bytes",
                    "audit_max_file_bytes": RANKING_AUDIT_MAX_FILE_BYTES,
                    "window_probe_raw_max_total_bytes": PROBE_MAX_TOTAL_BYTES,
                    "native_energy_raw_max_file_bytes": NATIVE_RAW_MAX_FILE_BYTES,
                    "native_energy_raw_max_total_bytes": NATIVE_RAW_MAX_TOTAL_BYTES,
                    "excluded_payload_classes": [
                        "tensor .bin and model/runtime binaries",
                        "PNG/JPEG/SVG/PDF figures",
                        "resources and lean_bundle copies except exact status matrices",
                        "central-quality candidate/reference/annotation bodies",
                        "Native producer candidate-body mirrors",
                    ],
                    "offline_replay": (
                        "not included; use the explicit replay export for payload bodies"
                    ),
                    "main_workflow_log": "full canonical root member; never tailed",
                },
                "main_workflow_log": {
                    "expected_member": MAIN_WORKFLOW_LOG,
                    "source_present": main_present,
                    "tail_fallback_source_present": tail_fallback_present,
                    "archived_members": (
                        [MAIN_WORKFLOW_LOG] if main_record
                        else [MAIN_WORKFLOW_TAIL] if tail_record else []
                    ),
                    "missing_source_members": [] if main_present else [MAIN_WORKFLOW_LOG],
                    "size_bytes": (
                        main_record or tail_record
                    ).get("size_bytes"),
                    "sha256": (main_record or tail_record).get("sha256", ""),
                    "completeness": (
                        "full" if main_record
                        else "tail_only" if tail_record else "missing"
                    ),
                    "complete": bool(main_record),
                },
                "exact_duplicate_aliases": {
                    "mode": "known_paths_and_byte_identical_sha256_only",
                    "omitted_alias_count": len(
                        exact_duplicate_alias_records
                    ),
                    "omitted_aliases": exact_duplicate_alias_records,
                    "canonical_members": sorted(alias_canonical_members),
                    "all_canonical_members_archived": all(
                        row["canonical_archived"]
                        for row in exact_duplicate_alias_records
                    ),
                    "all_canonical_hashes_verified": all(
                        row["canonical_hash_verified"]
                        for row in exact_duplicate_alias_records
                    ),
                },
                "ranking_audit_evidence": {
                    **audit,
                    "required_archive_members": sorted(audit_present),
                    "archived_members": audit_archived,
                    "missing_source_members": audit_missing,
                    "complete": audit_complete,
                    "all_archived_members_sha256_recorded": all(
                        bool(by_written.get(path, {}).get("sha256"))
                        for path in audit_archived
                    ),
                },
                "central_quality_replay_inputs": {
                    **central,
                    "mode": "request_descriptors_and_declared_decoded_predictions",
                    "replay_payloads_included": bool(quality_payload_members),
                    "decoded_prediction_payloads": {
                        **quality_payloads,
                        "archived_members": sorted(quality_payload_members & written_paths),
                        "complete": not quality_payloads["omitted"] and quality_payload_members <= written_paths,
                    },
                    "required_archive_members": sorted(central_present),
                    "archived_members": central_archived,
                    "complete": bool(
                        not central_missing
                        and not central.get("failures")
                        and central_present <= written_paths
                    ),
                    "limits": {
                        "max_requests": CENTRAL_DESCRIPTOR_MAX_REQUESTS,
                        "max_file_bytes": CENTRAL_DESCRIPTOR_MAX_FILE_BYTES,
                        "max_total_bytes": CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES,
                    },
                },
                "cancelled_diagnostic_priority": {
                    "policy": "control_plane_before_bulky_case_evidence",
                    "source_members": sorted(cancelled_priority_present),
                    "archived_members": cancelled_priority_archived,
                    "missing_archive_members": sorted(
                        cancelled_priority_present - written_paths
                    ),
                    "complete": bool(
                        cancelled_priority_present <= written_paths
                    ),
                },
                "execution_provenance_evidence": {
                    "required_members": sorted(provenance_present),
                    "present_members": provenance_archived,
                    "missing_members": sorted(provenance_present - written_paths),
                    "all_present_members_sha256_recorded": all(
                        bool(by_written.get(path, {}).get("sha256"))
                        for path in provenance_archived
                    ),
                },
                "offline_replay_core": {
                    "expected_members": sorted(OFFLINE_REPLAY_CORE),
                    "present_source_members": sorted(offline_present),
                    "archived_members": offline_archived,
                    "missing_source_members": sorted(
                        set(OFFLINE_REPLAY_CORE) - offline_present
                    ),
                    "all_archived_members_sha256_recorded": all(
                        bool(by_written.get(path, {}).get("sha256"))
                        for path in offline_archived
                    ),
                },
                "model_validation_summaries": {
                    "present_source_members": model_validation_present,
                    "admitted_source_members": model_validation_admitted,
                    "archived_members": model_validation_archived,
                    "omitted_source_members": model_validation_omitted,
                    "max_file_bytes": MODEL_VALIDATION_SUMMARY_MAX_FILE_BYTES,
                    "max_total_bytes": (
                        MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES
                    ),
                    "archived_total_bytes": sum(
                        int(by_written.get(path, {}).get("size_bytes") or 0)
                        for path in model_validation_archived
                    ),
                    "all_admitted_members_archived": (
                        set(model_validation_admitted) <= written_paths
                    ),
                    "all_archived_members_sha256_recorded": all(
                        bool(by_written.get(path, {}).get("sha256"))
                        for path in model_validation_archived
                    ),
                },
                "backend_artifact_diagnostics": {
                    "present_source_members": (
                        backend_artifact_diagnostic_present
                    ),
                    "admitted_source_members": (
                        backend_artifact_diagnostic_admitted
                    ),
                    "archived_members": backend_artifact_diagnostic_archived,
                    "omitted_source_members": (
                        backend_artifact_diagnostic_omitted
                    ),
                    "max_file_bytes": BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES,
                    "max_total_bytes": (
                        BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES
                    ),
                    "archived_total_bytes": sum(
                        int(by_written.get(path, {}).get("size_bytes") or 0)
                        for path in backend_artifact_diagnostic_archived
                    ),
                    "all_admitted_members_archived": (
                        set(backend_artifact_diagnostic_admitted)
                        <= written_paths
                    ),
                    "all_archived_members_sha256_recorded": all(
                        bool(by_written.get(path, {}).get("sha256"))
                        for path in backend_artifact_diagnostic_archived
                    ),
                },
                "runtime_execution_diagnostics": {
                    **runtime_diagnostics,
                    "archived_members": sorted(runtime_present & written_paths),
                    "original_archived_count": len(runtime_present & written_paths),
                    "source_coverage": runtime_coverage,
                    "derived_summary_members": [summary["path"] for summary in derived_summaries],
                    "derived_summary_count": len(derived_summaries),
                    "compact_view_complete_for_discovered_sources": all(
                        row["original_archived"] or row["derived_summary_archived"]
                        for row in runtime_coverage
                    ),
                    "summary_max_file_bytes": SUMMARY_MAX_BYTES,
                    "summary_max_total_bytes": SUMMARY_TOTAL_MAX_BYTES,
                    "summary_fallback_parse_max_file_bytes": FALLBACK_PARSE_MAX_BYTES,
                    "summary_fallback_parse_max_total_bytes": RUNTIME_FALLBACK_PARSE_MAX_TOTAL_BYTES,
                    "summary_provenance_verification": "source_path_size_mtime; declared_hashes_not_reverified",
                    "source_count": len(runtime_coverage),
                    "omitted_source_members": runtime_omitted,
                    "bounded_log_tails": [
                        row["path"] for row in written
                        if row.get("diagnostic_kind") == "bounded_log_tail"
                        and row.get("source_path") in runtime_present
                    ],
                    "max_file_bytes": max(0, int(max_small_file_bytes)),
                    "complete": runtime_complete,
                    "status": "complete" if runtime_complete else "partial",
                },
                "management_cpu_reference_diagnostics": {
                    **reference_diagnostics,
                    "scope": (
                        "quality_management/references/<model_id>/"
                        "{management_cpu_reference_status.json,management_cpu_reference_stdout.txt}"
                    ),
                    "content_policy": "original_bytes_only; diagnostic_status_is_not_semantic_success",
                    "archived_members": reference_archived,
                    "archived_total_bytes": sum(int(by_written[path]["size_bytes"]) for path in reference_archived),
                    "all_admitted_members_archived": reference_admitted <= written_paths,
                    "all_discovered_members_archived": (
                        set(reference_diagnostics["present_source_members"]) <= written_paths
                    ),
                    "complete": reference_complete,
                    "status": (
                        "partial" if not reference_complete
                        else "complete" if reference_archived else "not_present"
                    ),
                },
                "native_energy_attempt_diagnostics": {
                    **energy_attempts,
                    "archived_members": sorted(set(energy_attempts["expected_members"]) & written_paths),
                    "complete": (not energy_attempts["missing_source_members"]
                                 and not energy_attempts["reference_failures"]
                                 and set(energy_attempts["expected_members"]) <= written_paths),
                    "measurement_verification": "not_performed_by_exporter",
                },
                "native_energy_raw_traces": {
                    **native_raw,
                    "scope": NATIVE_MEASUREMENTS_ROOT + "**/collector_storage/*.parquet",
                    "requested_count": len(native_raw["requested_members"]),
                    "present_count": len(native_raw["present_source_members"]),
                    "archived_count": len(native_raw_archived),
                    "omitted_count": len(native_raw_omitted),
                    "archived_members": native_raw_archived,
                    "omitted_members": native_raw_omitted,
                    "archived_total_bytes": native_raw_total_bytes,
                    "max_file_bytes": NATIVE_RAW_MAX_FILE_BYTES,
                    "max_total_bytes": NATIVE_RAW_MAX_TOTAL_BYTES,
                    "complete": native_raw_complete,
                    "status": (
                        "not_requested" if not native_include_raw
                        else "partial" if not native_raw_complete
                        else "complete" if native_raw_archived
                        else "no_source_traces"
                    ),
                },
                "window_method_validation_probe": {
                    "present": any(
                        str(row.get("path") or "").startswith(
                            "reports/window_method_validation_probe/"
                        ) for row in written
                    ),
                    "include_raw_parquet_resolved": probe_include_raw,
                    "file_count": sum(
                        1 for row in written
                        if str(row.get("path") or "").startswith(
                            "reports/window_method_validation_probe/"
                        )
                    ),
                    "raw_parquet_count": sum(
                        row.get("diagnostic_kind") == "window_probe_raw_trace"
                        for row in written
                    ),
                    "raw_parquet_total_bytes": sum(
                        int(row.get("size_bytes") or 0) for row in written
                        if row.get("diagnostic_kind")
                        == "window_probe_raw_trace"
                    ),
                    "raw_parquet_max_total_bytes": PROBE_MAX_TOTAL_BYTES,
                    "window_method_comparison_file_count": sum(
                        row.get("diagnostic_kind") == "window_method_A/B_comparison"
                        for row in written
                    ),
                    "command_file_count": sum(
                        row.get("diagnostic_kind") == "window_probe_command"
                        for row in written
                    ),
                    "all_probe_members_sha256_recorded": all(
                        bool(row.get("sha256")) for row in written
                        if str(row.get("path") or "").startswith(
                            "reports/window_method_validation_probe/"
                        )
                    ),
                },
                "required_member_verification": {
                    "required_members": sorted(required_members),
                    "source_present_members_are_required": True,
                    "source_missing_members_block_publication": False,
                    "verification_scope": "closed ZIP member names plus all-member CRC",
                },
                "portable_zip_timestamps": True,
                "file_count": len(written),
                "uncompressed_size_bytes": sum(
                    int(row.get("size_bytes") or 0) for row in written
                ),
                "native_analysis_file_count": sum(
                    row.get("diagnostic_kind") == "native_analysis"
                    for row in written
                ),
                "skipped_count": len(skipped),
                "files": written,
                "skipped_examples": skipped[:500],
            }
            archive.writestr(
                "debug_pack_manifest.json",
                json.dumps(
                    manifest, indent=2, ensure_ascii=False, sort_keys=True
                ) + "\n",
            )

        for summary in derived_summaries:
            source = require_safe_pack_source(root / summary["source_path"], root)
            if source_observation(source) != summary["source_observation"]:
                raise RuntimeError("runtime_summary_source_changed_before_publication:" + summary["source_path"])
        verification = publish_verified_zip(
            temporary,
            destination,
            required_members=sorted(required_members),
            build_identity=tool_build,
            manifest_extra={
                "run_id": root.name,
                "pack_kind": "evaluation_debug_pack",
                "debug_pack_schema_version": DEBUG_PACK_SCHEMA_VERSION,
            },
        )
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "ok": True,
        "run_dir": str(root),
        "debug_pack": str(destination),
        "out_zip": str(destination),
        "file_count": len(written),
        "archive_size_bytes": verification["archive_size_bytes"],
        "archive_sha256": verification["archive_sha256"],
        "archive_verification": verification["status"],
        "verification_manifest": verification["verification_manifest"],
        "tool_build": tool_build,
    }


create_debug_pack = create_evaluation_debug_pack


__all__ = [
    "ALLOWED_SUFFIXES",
    "BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES",
    "BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES",
    "BLOCKED_DIRECTORY_PARTS",
    "BLOCKED_SUFFIXES",
    "CENTRAL_DESCRIPTOR_MAX_FILE_BYTES",
    "CENTRAL_DESCRIPTOR_MAX_REQUESTS",
    "CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES",
    "DEBUG_PACK_SCHEMA_VERSION",
    "DEFAULT_MAX_SMALL_FILE_BYTES",
    "DEFAULT_TAIL_BYTES",
    "INCLUDE",
    "INCLUDE_ROOT_FILES",
    "MODEL_VALIDATION_SUMMARY_MAX_FILE_BYTES",
    "MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES",
    "MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_FILE_BYTES",
    "MANAGEMENT_REFERENCE_DIAGNOSTIC_MAX_TOTAL_BYTES",
    "PROVENANCE_EVIDENCE_NAMES",
    "PROBE_MAX_TOTAL_BYTES",
    "REPLAY_CORE",
    "create_debug_pack",
    "create_evaluation_debug_pack",
    "discover_central_request_descriptors",
    "is_management_reference_diagnostic",
    "should_include_debug_file",
]
