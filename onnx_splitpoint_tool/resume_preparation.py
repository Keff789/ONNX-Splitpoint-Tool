from __future__ import annotations

"""Prepare a later Native Energy resume without starting a measurement.

The original plan rows, command contracts, runner scripts, and remote
preflights remain authoritative.  This coordinator only:

1. derives the exact data-artifact requirements from the selected contracts;
2. probes their frozen remote destinations read-only;
3. resolves and atomically restores only missing or byte-different artifacts;
4. runs every selected preflight as one all-pass cohort.

The returned ``measurement_wrapper_allowed`` flag is the sole hand-off to the
measurement coordinator.  This module never invokes a measurement wrapper,
collector, or workload.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .resume_artifact_contract import (
    HAILO8_PYTHON_DETECTION_SOURCE_RECOVERY_PROFILE,
    resume_artifact_requirements,
)
from .resume_artifact_rehydration import (
    ArtifactRequirement,
    ResumeArtifactResolutionError,
    build_resume_artifact_stage_map,
)
from .resume_cohort_preflight import run_resume_cohort_preflight
from .resume_hailo8_source_recovery import (
    HAILO8_SOURCE_RECOVERY_PROFILE,
    Hailo8ResumeSourceRecoveryError,
    materialize_hailo8_literal_sources,
)
from .resume_remote_rehydration import (
    RemoteArtifactRehydrationError,
    probe_remote_requirements,
    rehydrate_remote_stage_map,
    validate_resume_ssh_target,
)


RESUME_PREPARATION_SCHEMA = "onnx-splitpoint/resume-measurement-preparation"
RESUME_PREPARATION_SCHEMA_VERSION = 1
_HAILO8_RECOVERABLE_SOURCE_ROLES = frozenset(
    {"cmake", "generated_cpp"}
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _strict_directory(value: str | Path, *, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label}_is_symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label}_not_directory")
    return resolved


def _strict_summary(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError("summary_is_symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("summary_not_regular")
    return resolved


def _run_mirror_root(summary: Path) -> Path:
    # Workflow summaries live at <EvaluationRun>/reports/*.json.  Keeping the
    # complete EvaluationRun as the mirror root permits longest-suffix matching
    # between local native_producers/* and the frozen remote benchmark layout.
    return summary.parent.parent if summary.parent.name == "reports" else summary.parent


def _row_selector(row: Mapping[str, Any]) -> str:
    values = [
        str(row.get(field) or "").strip()
        for field in ("backend", "model", "case", "setup_id")
    ]
    if any(not value or "|" in value for value in values):
        raise ValueError("resume_row_identity_invalid")
    return "|".join(values)


def _row_ssh_target(
    row: Mapping[str, Any],
    context: Mapping[str, str],
) -> tuple[str, str]:
    identity = " ".join(
        str(row.get(field) or "").strip().lower()
        for field in ("backend", "setup_id")
    )
    if "hailo10" in identity:
        field = "hailo10_ssh"
    elif "hailo8" in identity:
        field = "hailo8_ssh"
    elif "deepx" in identity:
        field = "deepx_ssh"
    else:
        raise ValueError(
            f"resume_row_ssh_family_unknown:{_row_selector(row)}"
        )
    return field, validate_resume_ssh_target(context.get(field))


def _requirement_report(
    requirement: ArtifactRequirement,
) -> dict[str, Any]:
    return {
        "role": requirement.role,
        "remote_path": requirement.remote_path,
        "sha256": requirement.sha256,
        "size_bytes": requirement.size_bytes,
    }


def _hailo8_source_recovery_allowed(
    group: Mapping[str, Any],
    error: ResumeArtifactResolutionError,
) -> bool:
    """Authorise the narrow fallback only for its sealed contract profile."""
    if (
        error.code != "exact_source_not_found"
        or HAILO8_SOURCE_RECOVERY_PROFILE
        != HAILO8_PYTHON_DETECTION_SOURCE_RECOVERY_PROFILE
    ):
        return False
    missing_roles = {
        role.strip()
        for role in str(error.role or "").split(",")
        if role.strip()
    }
    if (
        not missing_roles
        or not missing_roles.issubset(
            _HAILO8_RECOVERABLE_SOURCE_ROLES
        )
    ):
        return False
    reports = group.get("contract_reports")
    if not isinstance(reports, list) or not reports:
        return False
    for raw_report in reports:
        if not isinstance(raw_report, Mapping):
            return False
        if (
            str(raw_report.get("source_recovery_profile") or "")
            != HAILO8_SOURCE_RECOVERY_PROFILE
        ):
            return False
        declared_roles = {
            str(role)
            for role in raw_report.get("source_recovery_roles") or []
        }
        if not missing_roles.issubset(declared_roles):
            return False
    return True


def _base_report(
    *,
    report_path: Path,
    attempt_id: str,
    selected_rows: Sequence[Mapping[str, Any]],
    remote_root: str,
) -> dict[str, Any]:
    return {
        "schema": RESUME_PREPARATION_SCHEMA,
        "schema_version": RESUME_PREPARATION_SCHEMA_VERSION,
        "ok": False,
        "status": "in_progress",
        "resume_attempt_id": attempt_id,
        "selected_row_count": len(selected_rows),
        "selected_rows": [_row_selector(row) for row in selected_rows],
        "frozen_remote_root": remote_root,
        "measurement_wrapper_allowed": False,
        "measurement_wrapper_started_count": 0,
        "started_measurement_count": 0,
        "collector_started_repeat_count": 0,
        "workload_started_repeat_count": 0,
        "remote_groups": [],
        "cohort_preflight": None,
        "report_path": str(report_path),
    }


def _finalise_report(
    report: dict[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    report["report_sha256"] = _canonical_sha256(unsigned)
    _atomic_write_json(report_path, report)
    return report


def prepare_resume_measurement_cohort(
    selected_rows: Iterable[Mapping[str, Any]],
    *,
    attempt_dir: str | Path,
    plan_root: str | Path,
    summary_path: str | Path,
    canonical_execution_context: Mapping[str, str],
    resume_attempt_id: str,
    timeout_s: float = 900.0,
    preflight_max_age_s: float = 300.0,
    artifact_store_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Restore exact legacy inputs and obtain an all-pass preflight gate."""
    rows = [dict(row) for row in selected_rows]
    attempt = _strict_directory(attempt_dir, label="attempt_dir")
    plan = _strict_directory(plan_root, label="plan_root")
    try:
        plan.relative_to(attempt)
    except ValueError as exc:
        raise ValueError("plan_root_outside_attempt_dir") from exc
    summary = _strict_summary(summary_path)
    preparation_dir = attempt / "resume_preparation"
    if preparation_dir.exists() or preparation_dir.is_symlink():
        raise ValueError("resume_preparation_directory_already_exists")
    preparation_dir.mkdir(mode=0o700)
    report_path = preparation_dir / "resume_preparation.json"
    remote_root = str(
        canonical_execution_context.get("remote_root") or ""
    ).strip()
    report = _base_report(
        report_path=report_path,
        attempt_id=str(resume_attempt_id),
        selected_rows=rows,
        remote_root=remote_root,
    )
    try:
        if not rows:
            raise ValueError("selected_rows_empty")
        if not remote_root:
            raise ValueError("frozen_remote_root_missing")

        grouped: dict[str, dict[str, Any]] = {}
        for row in rows:
            ssh_field, ssh_target = _row_ssh_target(
                row, canonical_execution_context,
            )
            contract = resume_artifact_requirements(
                row,
                plan_root=plan,
                frozen_remote_root=remote_root,
            )
            requirements = list(contract["requirements"])
            group = grouped.setdefault(
                ssh_target,
                {
                    "ssh_field": ssh_field,
                    "ssh_target": ssh_target,
                    "selectors": [],
                    "requirements": [],
                    "contract_reports": [],
                },
            )
            if group["ssh_field"] != ssh_field:
                raise ValueError("resume_ssh_target_family_collision")
            group["selectors"].append(_row_selector(row))
            group["requirements"].extend(requirements)
            group["contract_reports"].append({
                key: value
                for key, value in contract.items()
                if key != "requirements"
            })

        mirror_roots = [_run_mirror_root(summary)]
        stores = list(
            artifact_store_roots
            if artifact_store_roots is not None
            else (
                Path.home()
                / ".onnx_splitpoint_tool"
                / "artifact_store",
            )
        )

        # Complete all read-only probes and all local resolution before the
        # first remote mutation.  This prevents a later missing local source
        # from producing a partially restaged selection.
        prepared_groups: list[dict[str, Any]] = []
        for index, ssh_target in enumerate(sorted(grouped)):
            group = grouped[ssh_target]
            probe = probe_remote_requirements(
                group["requirements"],
                ssh_target=ssh_target,
                remote_run_root=remote_root,
                timeout_s=min(float(timeout_s), 120.0),
            )
            probe_path = preparation_dir / f"group_{index:02d}_probe.json"
            _atomic_write_json(probe_path, probe)
            needed_paths = {
                str(entry["remote_path"])
                for entry in probe["entries"]
                if entry.get("exact") is not True
            }
            needed = [
                requirement
                for requirement in group["requirements"]
                if requirement.remote_path in needed_paths
            ]
            stage_map: dict[str, Any] | None = None
            stage_map_path: Path | None = None
            source_recovery: dict[str, Any] | None = None
            source_recovery_path: Path | None = None
            if needed:
                try:
                    stage_map = build_resume_artifact_stage_map(
                        needed,
                        run_mirror_roots=mirror_roots,
                        artifact_store_roots=stores,
                        allowed_remote_roots=[remote_root],
                        expand_payload_manifests=True,
                    )
                except ResumeArtifactResolutionError as exc:
                    if not _hailo8_source_recovery_allowed(group, exc):
                        raise
                    derived_parent = preparation_dir / "derived_sources"
                    derived_parent.mkdir(mode=0o700, exist_ok=True)
                    source_recovery_path = (
                        preparation_dir
                        / f"group_{index:02d}_source_recovery.json"
                    )
                    try:
                        source_recovery = (
                            materialize_hailo8_literal_sources(
                                needed,
                                tool_root=Path(__file__).resolve().parents[1],
                                destination=(
                                    derived_parent / f"group_{index:02d}"
                                ),
                            )
                        )
                    except Hailo8ResumeSourceRecoveryError as recovery_exc:
                        source_recovery = {
                            "schema": (
                                "onnx-splitpoint/"
                                "hailo8-resume-source-recovery"
                            ),
                            "schema_version": 1,
                            "profile": HAILO8_SOURCE_RECOVERY_PROFILE,
                            "ok": False,
                            "status": "failed",
                            "local_only": True,
                            "code_executed": False,
                            "build_started": False,
                            "remote_mutation_performed": False,
                            "failure_code": recovery_exc.code,
                            "failure_role": recovery_exc.role,
                            "failure_detail": recovery_exc.detail,
                        }
                        _atomic_write_json(
                            source_recovery_path, source_recovery,
                        )
                        raise
                    _atomic_write_json(
                        source_recovery_path, source_recovery,
                    )
                    stage_map = build_resume_artifact_stage_map(
                        needed,
                        run_mirror_roots=[
                            *mirror_roots,
                            Path(str(source_recovery["source_root"])),
                        ],
                        artifact_store_roots=stores,
                        allowed_remote_roots=[remote_root],
                        expand_payload_manifests=True,
                    )
                stage_map_path = (
                    preparation_dir / f"group_{index:02d}_stage_map.json"
                )
                _atomic_write_json(stage_map_path, stage_map)
            prepared_groups.append({
                **group,
                "probe": probe,
                "probe_path": str(probe_path),
                "needed_requirements": needed,
                "stage_map": stage_map,
                "stage_map_path": (
                    str(stage_map_path) if stage_map_path is not None else ""
                ),
                "source_recovery": source_recovery,
                "source_recovery_path": (
                    str(source_recovery_path)
                    if source_recovery_path is not None else ""
                ),
            })

        remote_reports: list[dict[str, Any]] = []
        for index, group in enumerate(prepared_groups):
            stage_map = group["stage_map"]
            if stage_map is None:
                rehydration: dict[str, Any] = {
                    "ok": True,
                    "status": "all_remote_artifacts_already_exact",
                    "entry_count": 0,
                    "no_op_count": int(
                        group["probe"].get("requirement_count") or 0
                    ),
                    "rehydrated_count": 0,
                    "backup_count": 0,
                    "transferred_bytes": 0,
                }
            else:
                try:
                    rehydration = rehydrate_remote_stage_map(
                        stage_map,
                        ssh_target=group["ssh_target"],
                        remote_run_root=remote_root,
                        resume_attempt_id=resume_attempt_id,
                        timeout_s=min(float(timeout_s), 300.0),
                    )
                except RemoteArtifactRehydrationError as exc:
                    if isinstance(exc.report, Mapping):
                        partial_path = (
                            preparation_dir
                            / f"group_{index:02d}_rehydration_partial.json"
                        )
                        _atomic_write_json(partial_path, exc.report)
                    raise
            rehydration_path = (
                preparation_dir / f"group_{index:02d}_rehydration.json"
            )
            _atomic_write_json(rehydration_path, rehydration)
            remote_reports.append({
                "ssh_field": group["ssh_field"],
                "ssh_target": group["ssh_target"],
                "selectors": sorted(group["selectors"]),
                "contract_reports": group["contract_reports"],
                "requirements": [
                    _requirement_report(requirement)
                    for requirement in group["requirements"]
                ],
                "probe": group["probe"],
                "probe_path": group["probe_path"],
                "stage_map_path": group["stage_map_path"],
                "source_recovery": group["source_recovery"],
                "source_recovery_path": group["source_recovery_path"],
                "rehydration": rehydration,
                "rehydration_path": str(rehydration_path),
            })
        report["remote_groups"] = remote_reports

        cohort = run_resume_cohort_preflight(
            rows,
            attempt_dir=attempt,
            plan_root=plan,
            timeout_s=float(timeout_s),
            max_age_s=float(preflight_max_age_s),
        )
        report["cohort_preflight"] = cohort
        allowed = (
            cohort.get("ok") is True
            and cohort.get("measurement_wrapper_allowed") is True
        )
        report["ok"] = bool(allowed)
        report["status"] = (
            "ready_for_measurement"
            if allowed
            else "resume_cohort_preflight_failed"
        )
        report["measurement_wrapper_allowed"] = bool(allowed)
    except Exception as exc:
        report["ok"] = False
        report["status"] = "resume_preparation_failed"
        report["failure_type"] = type(exc).__name__
        report["failure"] = str(exc)
        if isinstance(
            exc,
            (
                Hailo8ResumeSourceRecoveryError,
                RemoteArtifactRehydrationError,
                ResumeArtifactResolutionError,
            ),
        ):
            report["failure_code"] = exc.code
            report["failure_detail"] = exc.detail
            if getattr(exc, "role", ""):
                report["failure_role"] = exc.role
        if isinstance(exc, RemoteArtifactRehydrationError):
            if isinstance(exc.report, Mapping):
                report["remote_rehydration_partial"] = dict(exc.report)
    return _finalise_report(report, report_path=report_path)


__all__ = [
    "RESUME_PREPARATION_SCHEMA",
    "RESUME_PREPARATION_SCHEMA_VERSION",
    "prepare_resume_measurement_cohort",
]
