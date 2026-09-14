"""Shared, bounded policy for compact EvaluationRun debug packs.

The pack is intentionally a diagnostic/evidence archive, not a second copy of
the generated benchmark set.  Exact small JSON/CSV/Markdown artefacts needed
to reconstruct the ranking audit are admitted explicitly.  Tensor payloads,
figures and unreferenced candidate bodies are not part of the default pack.
Declared decoded quality JSON is separately admitted with existing file hashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


MAIN_WORKFLOW_LOG = "evaluation_workflow.log"
MAIN_WORKFLOW_TAIL = "evaluation_workflow_tail.log"

# These limits are independent from the generic two-MiB diagnostic-file cap.
# Candidate plans can legitimately exceed that cap for a 20-case audit, while
# still being tiny compared with tensor dumps.  Keep the exception exact and
# bounded both per file and for the complete audit evidence set.
STRUCTURED_RESULT_MAX_FILE_BYTES = 256 * 1024 * 1024
STRUCTURED_RESULT_MAX_TOTAL_BYTES = 512 * 1024 * 1024
CENTRAL_DESCRIPTOR_MAX_FILE_BYTES = 64 * 1024 * 1024
CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES = 512 * 1024 * 1024
RANKING_AUDIT_MAX_FILE_BYTES = STRUCTURED_RESULT_MAX_FILE_BYTES
RANKING_AUDIT_MAX_TOTAL_BYTES = STRUCTURED_RESULT_MAX_TOTAL_BYTES

# Per-model validation summaries are compact diagnostic indexes, not replay
# payloads.  Include them for every model so a lean pack can explain invalid
# normalized rows without restoring tensors or prediction bodies.  Keep their
# allowance consistent with the explicitly admitted structured result class;
# its aggregate budget is recorded independently of the generic small cap.
MODEL_VALIDATION_SUMMARY_MAX_FILE_BYTES = STRUCTURED_RESULT_MAX_FILE_BYTES
MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES = STRUCTURED_RESULT_MAX_TOTAL_BYTES

# Backend-build decisions are small control-plane receipts needed to explain
# cache reuse and early compiler-stage failures.  Admit only the exact JSON
# names below; generated suites and their model/runtime binaries remain out of
# the compact pack.  The aggregate bound prevents a malformed multi-model run
# from turning this narrow exception into an unbounded directory walk.
BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES = STRUCTURED_RESULT_MAX_FILE_BYTES
BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES = STRUCTURED_RESULT_MAX_TOTAL_BYTES
BACKEND_ARTIFACT_BENCHMARK_FILES = frozenset({
    "artifact_reuse_manifest.json",
    "backend_artifact_decisions.json",
    "hailo_artifact_service_plan.json",
    "hailo_build_service_status.json",
    "deepx_prefetch_v60s.json",
})

CANCELLED_DIAGNOSTIC_ROOT_FILES = frozenset({
    "run_manifest.json",
    "artifact_index.json",
    "primary_failure.json",
    "effective_execution_plan.json",
})

CANCELLED_DIAGNOSTIC_REMOTE_CONTROL_FILES = frozenset({
    "run_status.json",
    "run_results.json",
    "suite_bundle_status.json",
    "benchmark_suite_status.json",
    "remote_cleanup.json",
    "preflight.json",
})

RANKING_AUDIT_ANALYSIS_FILES = frozenset({
    "analysis.json",
    "candidate_ranking.json",
    "prediction.json",
    "predictions_frozen.csv",
    "ranking_predictions_frozen.csv",
    "prediction_freeze_manifest.json",
    "candidate_universe_manifest.json",
    "candidate_universe.csv",
    "selection_input.json",
    "audit_plan.json",
    "final_candidate_plan.json",
})

# Hold-out campaigns may produce these alongside the common audit contract,
# but they are conditional rather than universally required.  Archive them
# when present without making a development audit permanently incomplete.
RANKING_AUDIT_OPTIONAL_ANALYSIS_FILES = frozenset({
    "holdout_predictions_frozen.csv",
    "holdout_ranking_predictions_frozen.csv",
    "prediction_freeze_conflict.json",
    "prediction_freeze_approval.json",
    "prediction_freeze_approval_verification.json",
    "prediction_freeze_approval_request.md",
})

RANKING_AUDIT_BENCHMARK_FILES = frozenset({
    "benchmark_cases.csv",
    "benchmark_set.json",
    "generation_decisions.json",
})

RANKING_AUDIT_REPORT_FILES = frozenset({
    "reports/scientific/native_ranking_audit.csv",
    "reports/scientific/native_ranking_audit.json",
    "reports/scientific/native_ranking_audit.md",
    "reports/scientific/ranking_method_comparison.csv",
    "reports/scientific/ranking_method_macro.csv",
    "reports/scientific/row_eligibility.csv",
    "reports/scientific/task_quality.csv",
    "reports/scientific/report_manifest.json",
})

# These files are copied into every Native producer benchmark tree.  Their
# canonical bytes already live below models/<model>/analysis and retaining the
# mirrors makes pack size grow with the number of hardware targets.
CANDIDATE_BODY_MIRROR_FILES = frozenset({
    "analysis.json",
    "candidate_ranking.json",
    "prediction.json",
    "predictions_frozen.csv",
    "holdout_predictions_frozen.csv",
    "ranking_predictions_frozen.csv",
    "holdout_ranking_predictions_frozen.csv",
    "prediction_freeze_manifest.json",
    "candidate_universe_manifest.json",
    "candidate_universe.csv",
    "audit_plan.json",
    "final_candidate_plan.json",
})


def _normalized(relative_path: str) -> str:
    normalized = str(relative_path or "").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def is_ranking_audit_evidence(relative_path: str) -> bool:
    """Return whether *relative_path* is exact canonical audit evidence."""

    normalized = _normalized(relative_path)
    if normalized in RANKING_AUDIT_REPORT_FILES:
        return True
    parts = normalized.split("/")
    if len(parts) == 4 and parts[0] == "models":
        if parts[2] == "analysis":
            return parts[3] in (
                RANKING_AUDIT_ANALYSIS_FILES
                | RANKING_AUDIT_OPTIONAL_ANALYSIS_FILES
            )
        if parts[2] == "benchmark_set":
            return parts[3] in RANKING_AUDIT_BENCHMARK_FILES
    return False


def is_candidate_body_mirror(relative_path: str) -> bool:
    """Identify predictor/audit bodies replicated below Native producers."""

    normalized = _normalized(relative_path)
    parts = normalized.split("/")
    return bool(
        len(parts) >= 4
        and parts[0] == "native_producers"
        and "benchmark_set" in parts[1:-1]
        and parts[-1] in CANDIDATE_BODY_MIRROR_FILES
    )


def is_model_validation_summary(relative_path: str) -> bool:
    """Identify the one compact row-validation index kept per model."""

    normalized = _normalized(relative_path)
    parts = normalized.split("/")
    return bool(
        len(parts) == 4
        and parts[0] == "models"
        and bool(parts[1])
        and parts[2:] == ["validation", "validation_summary.json"]
    )


def is_backend_artifact_diagnostic(relative_path: str) -> bool:
    """Identify exact, compact backend-build control-plane evidence.

    ``deepx_prefetch_v60s.json`` is produced inside ``legacy_suite`` by the
    normal generator, while direct/formal adapters can place it at the
    benchmark-set root.  Matching only that filename keeps all other suite
    contents excluded.
    """

    normalized = _normalized(relative_path)
    parts = normalized.split("/")
    if len(parts) >= 4 and parts[0] == "models" and bool(parts[1]):
        if parts[2] == "benchmark_set":
            if len(parts) == 4:
                return parts[3] in BACKEND_ARTIFACT_BENCHMARK_FILES
            return bool(
                parts[-1] == "deepx_prefetch_v60s.json"
                and parts[3:-1] == ["legacy_suite"]
            )
        return bool(
            len(parts) == 5
            and parts[2:]
            == ["stages", "build_backend_artifacts", "stage_result.json"]
        )
    return False


def is_remote_execution_failure_diagnostic(relative_path: str) -> bool:
    """Identify the compact control-plane evidence for remote-stage failures.

    The model tree is intentionally not scanned wholesale by compact debug
    packs.  These exact paths are the bounded exception needed to retain the
    primary workflow failure, the failing benchmark stage result, and the
    management-side remote status/stdout/stderr without admitting generated
    suites or model/runtime payloads.
    """

    normalized = _normalized(relative_path)
    if normalized == "primary_failure.json":
        return True
    parts = normalized.split("/")
    if (
        len(parts) == 5
        and parts[0] == "models"
        and bool(parts[1])
        and parts[2:]
        == ["stages", "run_benchmarks", "stage_result.json"]
    ):
        return True
    if not (
        len(parts) == 4
        and parts[0] == "models"
        and bool(parts[1])
        and parts[2] == "benchmark_results"
    ):
        return False
    name = parts[3]
    return bool(
        (name == "remote_hardware_matrix_status.json")
        or (
            name.startswith("remote_benchmark_status")
            and name.endswith(".json")
        )
        or (
            name.startswith("remote_benchmark_stdout")
            and name.endswith(".txt")
        )
        or (
            name.startswith("remote_benchmark_stderr")
            and name.endswith(".txt")
        )
    )


def is_cancelled_diagnostic_priority(relative_path: str) -> bool:
    """Identify small control-plane evidence that must precede bulky rows.

    Interrupted campaigns can contain thousands of per-case validation bodies.
    A byte-budgeted, lexicographic collector used to fill its archive with
    those bodies and omit the terminal run/status/lease JSON needed to explain
    the interruption.  This predicate intentionally admits only bounded
    control-plane documents; tensors and candidate bodies remain excluded.
    """

    normalized = _normalized(relative_path)
    if normalized in CANCELLED_DIAGNOSTIC_ROOT_FILES:
        return True
    parts = normalized.split("/")
    if any(part in {"lean_bundle", "resources"} for part in parts):
        return False
    if (
        len(parts) >= 3
        and parts[:2] == ["jobs", "remote_process_leases"]
        and parts[-1].endswith(".json")
    ):
        return True
    if normalized in {
        "jobs/job_plan.json",
        "jobs/job_summary.json",
        "reports/run_status_summary.json",
        "reports/result_dashboard.json",
        "quality_management/central_quality_summary.json",
        "quality_management/central_quality_queue_status.json",
    }:
        return True
    if (
        len(parts) == 5
        and parts[0] == "models"
        and bool(parts[1])
        and parts[2] == "stages"
        and bool(parts[3])
        and parts[4] == "stage_result.json"
    ):
        return True
    if (
        len(parts) >= 5
        and parts[0] == "models"
        and bool(parts[1])
        and parts[2:4] == ["benchmark_results", "remote_diagnostics"]
        and parts[-1] in CANCELLED_DIAGNOSTIC_REMOTE_CONTROL_FILES
    ):
        return True
    if (
        len(parts) == 4
        and parts[0] == "models"
        and bool(parts[1])
        and parts[2] == "benchmark_results"
        and (
            parts[3] == "remote_hardware_matrix_status.json"
            or (
                parts[3].startswith("remote_benchmark_status")
                and parts[3].endswith(".json")
            )
        )
    ):
        return True
    return False


def _load_object(path: Path) -> dict[str, Any]:
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink():
            return {}
        cursor = cursor.parent
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _load_profile(root: Path) -> dict[str, Any]:
    for name in ("profile.yaml", "profile_source.yaml"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if isinstance(value, dict):
            return dict(value)
    return {}


def _safe_regular_member(root: Path, relative: str) -> Path | None:
    candidate = root / relative
    cursor = root
    try:
        for part in Path(relative).parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
        return resolved if resolved.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _profile_audit_model_ids(profile: dict[str, Any]) -> tuple[bool, list[str]]:
    selection = profile.get("selection_policy")
    selection = selection if isinstance(selection, dict) else {}
    strategy = str(selection.get("selection_strategy") or "").strip().lower()
    strategy = strategy.replace("-", "_")
    requested = bool(
        selection.get("score_independent_audit_enabled") is True
        or strategy in {
            "score_independent_audit",
            "ranking_audit",
            "deterministic_audit",
        }
    )
    suite = profile.get("model_suite")
    suite = suite if isinstance(suite, dict) else {}
    primary = suite.get("primary")
    model_ids: list[str] = []
    if isinstance(primary, list):
        for row in primary:
            if not isinstance(row, dict) or row.get("enabled") is False:
                continue
            model_id = str(row.get("id") or "").strip()
            if model_id and Path(model_id).name == model_id:
                model_ids.append(model_id)
    return requested, sorted(set(model_ids))


def discover_ranking_audit_evidence(run_dir: str | Path) -> dict[str, Any]:
    """Discover the exact fail-closed evidence contract for an audit run.

    Audit intent comes only from the resolved profile or from a canonical
    ``analysis/audit_plan.json`` whose ``enabled`` flag is true.  Scientific
    reports are also emitted for ``not_requested`` runs, so their existence is
    diagnostic only and must never create audit intent.  Once intent exists,
    all canonical model evidence and the run-level Native ranking reports are
    required fail-closed.
    """

    root = Path(run_dir)
    models_root = root / "models"
    discovered_model_ids: list[str] = []
    enabled_plan_model_ids: list[str] = []
    if models_root.is_dir():
        for model_dir in sorted(models_root.iterdir(), key=lambda item: item.name):
            if not model_dir.is_dir() or model_dir.is_symlink():
                continue
            discovered_model_ids.append(model_dir.name)
            plan = _load_object(model_dir / "analysis" / "audit_plan.json")
            if plan.get("enabled") is True:
                enabled_plan_model_ids.append(model_dir.name)

    profile_requested, profile_model_ids = _profile_audit_model_ids(
        _load_profile(root)
    )
    explicit_report_present = any(
        _safe_regular_member(root, relative) is not None
        for relative in RANKING_AUDIT_REPORT_FILES
    )
    if profile_requested:
        # A run can fail before analysis/audit_plan.json is materialised.  The
        # resolved profile remains authoritative for which evidence was
        # expected and lets the diagnostic archive expose those gaps.
        model_ids = sorted(set(profile_model_ids or discovered_model_ids))
    elif enabled_plan_model_ids:
        # Once one canonical plan declares the run-level audit strategy, use
        # every materialised model directory.  This catches a sibling model
        # that failed before writing its own plan while still requiring a real
        # enabled plan (rather than report existence) to create audit intent.
        model_ids = sorted(set(discovered_model_ids or enabled_plan_model_ids))
    else:
        model_ids = []

    enabled = bool(profile_requested or enabled_plan_model_ids)

    expected: list[str] = []
    for model_id in model_ids:
        expected.extend(
            f"models/{model_id}/analysis/{name}"
            for name in sorted(RANKING_AUDIT_ANALYSIS_FILES)
        )
        expected.extend(
            f"models/{model_id}/benchmark_set/{name}"
            for name in sorted(RANKING_AUDIT_BENCHMARK_FILES)
        )
    if enabled:
        expected.extend(sorted(RANKING_AUDIT_REPORT_FILES))

    expected = sorted(set(expected))
    present: list[str] = []
    missing: list[str] = []
    files: list[dict[str, Any]] = []
    total_bytes = 0
    oversized: list[dict[str, Any]] = []
    unpublishable: list[dict[str, Any]] = []
    for relative in expected:
        path = _safe_regular_member(root, relative)
        if path is None:
            lexical = root / relative
            if lexical.exists() or lexical.is_symlink():
                unpublishable.append({
                    "path": relative,
                    "reason": "existing_source_unsafe_or_unreadable",
                })
            else:
                missing.append(relative)
            continue
        size = int(path.stat().st_size)
        total_bytes += size
        present.append(relative)
        files.append({"path": relative, "size_bytes": size})
        if size > RANKING_AUDIT_MAX_FILE_BYTES:
            oversized.append({
                "path": relative,
                "size_bytes": size,
                "reason": "size_limit_exceeded",
                "limit_bytes": RANKING_AUDIT_MAX_FILE_BYTES,
            })
    if total_bytes > RANKING_AUDIT_MAX_TOTAL_BYTES:
        oversized.append({
            "path": "<ranking_audit_total>",
            "size_bytes": total_bytes,
            "reason": "size_limit_exceeded",
                "limit_bytes": RANKING_AUDIT_MAX_TOTAL_BYTES,
        })
    optional_present: list[str] = []
    optional_files: list[dict[str, Any]] = []
    for model_id in model_ids:
        for name in sorted(RANKING_AUDIT_OPTIONAL_ANALYSIS_FILES):
            relative = f"models/{model_id}/analysis/{name}"
            path = _safe_regular_member(root, relative)
            if path is None:
                lexical = root / relative
                if lexical.exists() or lexical.is_symlink():
                    unpublishable.append({
                        "path": relative,
                        "reason": "existing_optional_source_unsafe_or_unreadable",
                    })
                continue
            size = int(path.stat().st_size)
            total_bytes += size
            optional_present.append(relative)
            optional_files.append({"path": relative, "size_bytes": size})
            if size > RANKING_AUDIT_MAX_FILE_BYTES:
                oversized.append({
                    "path": relative,
                    "size_bytes": size,
                    "reason": "size_limit_exceeded",
                "limit_bytes": RANKING_AUDIT_MAX_FILE_BYTES,
                })
    if total_bytes > RANKING_AUDIT_MAX_TOTAL_BYTES:
        total_row = next(
            (
                row for row in oversized
                if row.get("path") == "<ranking_audit_total>"
            ),
            None,
        )
        if total_row is None:
            oversized.append({
                "path": "<ranking_audit_total>",
                "size_bytes": total_bytes,
                "reason": "size_limit_exceeded",
                "limit_bytes": RANKING_AUDIT_MAX_TOTAL_BYTES,
            })
        else:
            total_row["size_bytes"] = total_bytes
    return {
        "enabled": enabled,
        "explicit_report_present": explicit_report_present,
        "profile_requested": profile_requested,
        "enabled_plan_model_ids": sorted(enabled_plan_model_ids),
        "model_ids": model_ids,
        "expected_members": expected,
        "present_source_members": present,
        "optional_present_source_members": optional_present,
        "missing_source_members": missing,
        "files": files + optional_files,
        "total_size_bytes": total_bytes,
        "max_file_bytes": RANKING_AUDIT_MAX_FILE_BYTES,
        "max_total_bytes": RANKING_AUDIT_MAX_TOTAL_BYTES,
        "oversized": oversized,
        "unpublishable": unpublishable,
        "source_contract_ok": bool(
            not missing and not oversized and not unpublishable
        ),
    }


OFFLINE_REPLAY_CORE = frozenset({
    "reports/native_producer_summary.json",
    "reports/native_validation/native_producer_validation_summary.json",
    "quality_management/central_quality_summary.json",
    "reports/native_energy_measurements/native_producer_energy_results.json",
})


def is_offline_replay_core(relative_path: str) -> bool:
    """Return whether *relative_path* is a size-exempt replay-core report.

    Only four exact, text-JSON paths are exempt.  This deliberately does not
    turn a directory prefix into an unbounded archive allowance.
    """

    normalized = _normalized(relative_path)
    return normalized in OFFLINE_REPLAY_CORE


__all__ = [
    "BACKEND_ARTIFACT_BENCHMARK_FILES",
    "STRUCTURED_RESULT_MAX_FILE_BYTES",
    "STRUCTURED_RESULT_MAX_TOTAL_BYTES",
    "CENTRAL_DESCRIPTOR_MAX_FILE_BYTES",
    "CENTRAL_DESCRIPTOR_MAX_TOTAL_BYTES",
    "BACKEND_ARTIFACT_DIAGNOSTIC_MAX_FILE_BYTES",
    "BACKEND_ARTIFACT_DIAGNOSTIC_MAX_TOTAL_BYTES",
    "CANDIDATE_BODY_MIRROR_FILES",
    "MAIN_WORKFLOW_LOG",
    "MAIN_WORKFLOW_TAIL",
    "MODEL_VALIDATION_SUMMARY_MAX_FILE_BYTES",
    "MODEL_VALIDATION_SUMMARY_MAX_TOTAL_BYTES",
    "OFFLINE_REPLAY_CORE",
    "RANKING_AUDIT_ANALYSIS_FILES",
    "RANKING_AUDIT_BENCHMARK_FILES",
    "RANKING_AUDIT_MAX_FILE_BYTES",
    "RANKING_AUDIT_MAX_TOTAL_BYTES",
    "RANKING_AUDIT_OPTIONAL_ANALYSIS_FILES",
    "RANKING_AUDIT_REPORT_FILES",
    "discover_ranking_audit_evidence",
    "is_candidate_body_mirror",
    "is_backend_artifact_diagnostic",
    "is_model_validation_summary",
    "is_offline_replay_core",
    "is_ranking_audit_evidence",
    "is_remote_execution_failure_diagnostic",
]
