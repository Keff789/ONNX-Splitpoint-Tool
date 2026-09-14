from __future__ import annotations

"""Read-only, purpose-specific EvaluationRun discovery.

An interrupted workflow is useful for debugging and can contain valid stage
checkpoints, but it is not automatically a finished result.  Earlier releases
collapsed those meanings into a single ``is_evaluation_run_dir`` predicate and
could therefore let a newer empty/partial directory hide the most recent
completed result.  This module keeps identification, analysis and resume
admission separate and performs no filesystem writes.
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..energy.comparison import resolve_energy_comparison


RunPurpose = Literal["debug", "latest_completed", "analysis", "resume"]

_RUN_DIR_PATTERNS = (
    re.compile(r"^\s*Run directory:\s*(?P<path>.+?)\s*$", re.IGNORECASE),
    re.compile(r"run directory created:\s*(?P<path>.+?)\s*$", re.IGNORECASE),
    re.compile(r"\brun_dir\s*[=:]\s*(?P<path>/\S+)", re.IGNORECASE),
)
_RUN_MANIFEST_SCHEMA = "onnx-splitpoint/evaluation-run-manifest"
_RUN_STATUS_SCHEMA = "onnx-splitpoint/run-status-summary"
_RESULTS_BUNDLE_SCHEMA = "onnx-splitpoint/results-bundle-manifest"
_TERMINAL_STATUSES = frozenset({"ok", "partial", "failed", "cancelled"})
_RESUMABLE_STATUSES = frozenset({"partial", "failed", "cancelled"})
_REQUIRED_REPORT_ARTIFACTS = frozenset({
    "scientific_report.json",
    "row_eligibility.csv",
    "task_quality.csv",
    "performance_results.csv",
    "energy_results.csv",
})
_RESUME_CONTRACT_HASH_FIELDS = (
    "profile_sha256",
    "effective_execution_plan_sha256",
    "stable_options_sha256",
    "plan_sha256",
    "resume_contract_sha256",
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _read_json_mapping(path: Path) -> tuple[dict[str, Any] | None, str]:
    if not path.is_file() or path.is_symlink():
        return None, "missing"
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return None, "malformed"
    if not isinstance(payload, dict):
        return None, "not_object"
    return payload, "ok"


def _schema_version_at_least(payload: Mapping[str, Any], minimum: int) -> bool:
    value = payload.get("schema_version")
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= minimum)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_regular_descendant(root: Path, relative: object) -> Path | None:
    raw = str(relative or "").strip().replace("\\", "/")
    logical = Path(raw)
    if not raw or logical.is_absolute() or ".." in logical.parts:
        return None
    try:
        root_resolved = root.resolve(strict=True)
        candidate = root / logical
        cursor = root
        for part in logical.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return None
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root_resolved)
        if not resolved.is_file() or candidate.is_symlink():
            return None
        return resolved
    except (OSError, RuntimeError, ValueError):
        return None


def _resume_contract_complete(manifest: Mapping[str, Any]) -> bool:
    contract = manifest.get("resume_contract")
    if not isinstance(contract, Mapping):
        return False
    if contract.get("schema") != "onnx-splitpoint/evaluation-resume-contract":
        return False
    if not _schema_version_at_least(contract, 1):
        return False
    return all(
        bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(contract.get(field) or "")))
        for field in _RESUME_CONTRACT_HASH_FIELDS
    )


def build_measurement_set_contract(
    run_dir: Path | str,
    *,
    allow_legacy_missing_run_id: bool = False,
) -> dict[str, Any]:
    """Hash-bind every complete normalized measurement file in one Run.

    Releases before 2.75.16 did not write ``evaluation_run_id`` into the
    normalized payload.  The optional legacy mode accepts only that one
    missing field; a present but different id, an incomplete matrix, or a
    missing model still fails closed.
    """

    root = Path(run_dir).expanduser()
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    manifest, manifest_state = _read_json_mapping(root / "run_manifest.json")
    run_id = str(manifest.get("run_id") or "") if manifest else ""
    profile_id = str(manifest.get("profile_id") or "") if manifest else ""
    manifest_models = manifest.get("models") if manifest else None
    expected_model_ids = sorted(
        str(model_id)
        for model_id in manifest_models
        if str(model_id).strip()
    ) if isinstance(manifest_models, Mapping) else []
    if (
        manifest_state != "ok"
        or not manifest
        or manifest.get("schema") != _RUN_MANIFEST_SCHEMA
        or not _schema_version_at_least(manifest, 1)
        or run_id != root.name
        or not profile_id
        or not expected_model_ids
        or not isinstance(manifest.get("model_count"), int)
        or isinstance(manifest.get("model_count"), bool)
        or manifest.get("model_count") != len(expected_model_ids)
    ):
        errors.append("measurement_manifest_identity_incomplete")
    models_root = root / "models"
    if models_root.is_symlink():
        errors.append("models_root_symlink")
    elif models_root.is_dir():
        try:
            model_dirs = sorted(path for path in models_root.iterdir() if path.is_dir())
        except OSError:
            model_dirs = []
            errors.append("models_root_unreadable")
        discovered_model_ids = {path.name for path in model_dirs}
        for unexpected in sorted(discovered_model_ids - set(expected_model_ids)):
            path = models_root / unexpected / "benchmark_results" / "normalized_results.json"
            if path.exists() or path.is_symlink():
                errors.append(f"unexpected_model_measurements:{unexpected}")
        for model_id in expected_model_ids:
            model_dir = models_root / model_id
            if model_id not in discovered_model_ids:
                errors.append(f"model_measurements_missing:{model_id}")
                continue
            if model_dir.is_symlink():
                errors.append(f"model_dir_symlink:{model_id}")
                continue
            relative = Path("models") / model_id / "benchmark_results" / "normalized_results.json"
            raw_path = root / relative
            if not raw_path.exists() and not raw_path.is_symlink():
                errors.append(f"normalized_results_missing:{model_id}")
                continue
            path = _safe_regular_descendant(root, relative)
            if path is None:
                errors.append(f"normalized_results_unsafe:{model_id}")
                continue
            payload, state = _read_json_mapping(path)
            if state != "ok" or payload is None:
                errors.append(f"normalized_results_{state}:{model_id}")
                continue
            rows = payload.get("results")
            declared_count = payload.get("result_count")
            valid_rows = bool(
                payload.get("schema") == "onnx-splitpoint/normalized-benchmark-results"
                and _schema_version_at_least(payload, 2)
                and str(payload.get("model_id") or "") == model_id
                and (
                    str(payload.get("evaluation_run_id") or "") == run_id
                    or (
                        allow_legacy_missing_run_id
                        and not str(payload.get("evaluation_run_id") or "")
                    )
                )
                and isinstance(rows, list)
                and rows
                and all(isinstance(row, Mapping) and bool(row) for row in rows)
                and isinstance(declared_count, int)
                and not isinstance(declared_count, bool)
                and declared_count == len(rows)
                and str(payload.get("status") or "") == "measured"
                and payload.get("matrix_complete") is True
                and all(
                    isinstance(payload.get(field), int)
                    and not isinstance(payload.get(field), bool)
                    and payload.get(field) == 0
                    for field in (
                        "missing_measurement_count",
                        "missing_required_profile_result_count",
                        "duplicate_required_profile_result_count",
                        "validation_cardinality_mismatch_count",
                    )
                )
            )
            if not valid_rows:
                errors.append(f"normalized_results_contract_mismatch:{model_id}")
                continue
            try:
                records.append({
                    "path": relative.as_posix(),
                    "model_id": model_id,
                    "result_count": len(rows),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })
            except OSError:
                errors.append(f"normalized_results_unreadable:{model_id}")

    total = sum(int(row["result_count"]) for row in records)
    material = {
        "schema": "onnx-splitpoint/measurement-set-contract",
        "schema_version": 1,
        "run_id": run_id,
        "profile_id": profile_id,
        "expected_model_ids": expected_model_ids,
        "model_count": len(records),
        "result_count": total,
        "artifacts": records,
    }
    digest = hashlib.sha256(
        json.dumps(
            material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        **material,
        "measurement_set_sha256": digest,
        "valid": bool(
            records
            and len(records) == len(expected_model_ids)
            and total > 0
            and not errors
        ),
        "errors": errors,
    }


def _recorded_tool_version(manifest: Mapping[str, Any]) -> tuple[int, ...]:
    raw = str(
        manifest.get("current_tool_version")
        or manifest.get("tool_version")
        or ""
    ).strip()
    match = re.fullmatch(r"v?(\d+(?:\.\d+){1,3})(?:[-+].*)?", raw)
    if not match:
        return ()
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return ()


def _measurement_report_projection_matches(
    run_dir: Path,
    measurement_set: Mapping[str, Any],
    report_rows: object,
    *,
    legacy: bool = False,
) -> bool:
    """Read-only cross-check of normalized rows and scientific projections.

    Current reports bind the precision, numeric-input and repetition identity
    as well as every projected measurement value.  A v2.75.15 report cannot
    represent the newer identity fields, so legacy admission is allowed only
    when each normalized row is unique under the old six-field identity.  In
    either mode the performance rows must match with exact cardinality and
    exact projected values.  Derived native-energy rows may coexist because
    they carry a different explicit ``row_role`` and are excluded here.
    """

    if not isinstance(report_rows, list) or not report_rows:
        return False

    evaluation_run_id = str(measurement_set.get("run_id") or "").strip()

    def canonical(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return format(value, ".17g")
        if isinstance(value, (Mapping, list, tuple)):
            if not value:
                return ""
            try:
                return json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            except (TypeError, ValueError):
                return "<unrepresentable>"
        return str(value).strip()

    def first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
        for name in names:
            value = row.get(name)
            if value not in (None, ""):
                return value
        return None

    def base_key(row: Mapping[str, Any], *, model_id: str = "") -> tuple[str, ...]:
        backend = str(row.get("backend") or "").strip()
        return (
            str(row.get("model_id") or model_id).strip(),
            backend,
            str(row.get("case_id") or "").strip(),
            str(row.get("variant") or "").strip(),
            str(
                row.get("setup_id")
                or row.get("measurement_setup_id")
                or row.get("source_setup_id")
                or ""
            ).strip(),
            str(
                row.get("run_id")
                or row.get("source_tag")
                or row.get("benchmark_run_id")
                or row.get("run_profile_id")
                or backend
                or evaluation_run_id
            ).strip(),
        )

    def key(
        row: Mapping[str, Any],
        *,
        model_id: str = "",
        report: bool,
    ) -> tuple[str, ...]:
        identity = base_key(row, model_id=model_id)
        extended_identity: tuple[str, ...] = ()
        if not legacy:
            extended_identity = tuple(canonical(value) for value in (
                first(
                    row,
                    ("runtime_precision_identity", "execution_precision", "precision"),
                ),
                row.get("runtime_numeric_input_sha256"),
                row.get("runtime_numeric_input_identity"),
                first(
                    row,
                    ("repetition_index", "process_local_repetition_index", "repeat_idx", "repeat_index"),
                ),
                row.get("repetition_id"),
                row.get("repetition_count_requested"),
                row.get("repetition_count_attempted"),
                row.get("repetition_count_valid"),
                row.get("repetition_status"),
                row.get("repetition_aggregation"),
                row.get("repetition_runtime_scope"),
                row.get("repetition_independence_verified"),
            ))

        if report:
            projected = (
                row.get("latency_ms"),
                row.get("cycle_ms"),
                row.get("throughput_fps"),
                row.get("average_power_w"),
                row.get("average_power_ci_low_w"),
                row.get("average_power_ci_high_w"),
                row.get("energy_per_work_j"),
                row.get("energy_per_work_sample_stddev_j"),
                row.get("energy_per_work_ci_low_j"),
                row.get("energy_per_work_ci_high_j"),
                row.get("energy_repeat_n"),
                row.get("energy_confidence_level"),
                row.get("energy_scope"),
                row.get("energy_window"),
            )
        else:
            energy_comparison = resolve_energy_comparison(row)
            projected = (
                first(row, ("split_latency_e2e_ms", "total_latency_ms", "full_e2e_latency_ms")),
                first(row, ("pipeline_cycle_selected_ms", "total_latency_ms")),
                first(row, ("throughput_primary_fps", "pipeline_fps_selected", "heterogeneous_pipeline_fps")),
                energy_comparison.get("comparison_average_power_w"),
                first(row, ("energy_streaming_avg_power_w_ci_low", "avg_power_w_ci_low")),
                first(row, ("energy_streaming_avg_power_w_ci_high", "avg_power_w_ci_high")),
                energy_comparison.get("comparison_energy_per_work_j"),
                energy_comparison.get("comparison_energy_per_work_sample_stddev_j"),
                energy_comparison.get("comparison_energy_per_work_ci_low_j"),
                energy_comparison.get("comparison_energy_per_work_ci_high_j"),
                first(row, ("energy_streaming_repeat_n", "energy_latency_repeat_n", "energy_valid_window_count")),
                row.get("energy_confidence_level"),
                first(row, ("energy_physical_scope", "measurement_scope", "energy_scope")),
                first(row, ("energy_window_label", "energy_measurement_window", "energy_window", "window_type")),
            )
        return identity + extended_identity + tuple(
            canonical(value) for value in projected
        )

    report_keys = Counter(
        key(row, report=True)
        for row in report_rows
        if isinstance(row, Mapping)
        and str(row.get("row_role") or "performance_observation")
        == "performance_observation"
    )
    expected: Counter[tuple[str, ...]] = Counter()
    legacy_identities: Counter[tuple[str, ...]] = Counter()
    artifacts = measurement_set.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            return False
        relative = artifact.get("path")
        path = _safe_regular_descendant(run_dir, relative)
        payload, state = _read_json_mapping(path) if path is not None else (None, "missing")
        if state != "ok" or payload is None:
            return False
        rows = payload.get("results")
        model_id = str(artifact.get("model_id") or "").strip()
        if not isinstance(rows, list) or not rows:
            return False
        normalized_rows = [row for row in rows if isinstance(row, Mapping)]
        expected.update(
            key(row, model_id=model_id, report=False)
            for row in normalized_rows
        )
        if legacy:
            legacy_identities.update(
                base_key(row, model_id=model_id) for row in normalized_rows
            )
    return bool(
        expected
        and (not legacy or all(count == 1 for count in legacy_identities.values()))
        and report_keys == expected
    )


def _report_manifest_binds(
    manifest: Mapping[str, Any] | None,
    report_path: Path,
) -> bool:
    if not manifest or manifest.get("schema") != "onnx-splitpoint/scientific-report-manifest":
        return False
    if not _schema_version_at_least(manifest, 1):
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return False
    report_root = report_path.parent
    seen: set[str] = set()
    for row in artifacts:
        if not isinstance(row, Mapping):
            return False
        logical = str(row.get("path") or "").replace("\\", "/")
        if not logical or logical in seen:
            return False
        seen.add(logical)
        digest = str(row.get("sha256") or "").removeprefix("sha256:").lower()
        size = row.get("size_bytes")
        path = _safe_regular_descendant(report_root, logical)
        if (
            path is None
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            return False
        try:
            if path.stat().st_size != size or _sha256_file(path) != digest:
                return False
        except OSError:
            return False
    return bool(_REQUIRED_REPORT_ARTIFACTS <= seen)


def _path_mode_allows_write(path: Path) -> bool:
    """Conservative, non-mutating directory permission check.

    Checking mode bits as well as ``os.access`` is intentional: release tests
    can run as root, where ``os.access`` alone would incorrectly accept a 0555
    directory.  ACL/mount checks are enforced again by the actual writer.
    """

    import os
    import stat

    try:
        st = path.stat()
        mode = st.st_mode
        uid = os.geteuid()
        groups = set(os.getgroups()) | {os.getegid()}
        if uid == st.st_uid:
            bits_ok = bool(mode & stat.S_IWUSR) and bool(mode & stat.S_IXUSR)
        elif st.st_gid in groups:
            bits_ok = bool(mode & stat.S_IWGRP) and bool(mode & stat.S_IXGRP)
        else:
            bits_ok = bool(mode & stat.S_IWOTH) and bool(mode & stat.S_IXOTH)
        statvfs = os.statvfs(path)
        read_only = bool(
            getattr(os, "ST_RDONLY", 1)
            and statvfs.f_flag & getattr(os, "ST_RDONLY", 1)
        )
        return bool(bits_ok and not read_only and os.access(path, os.W_OK | os.X_OK))
    except OSError:
        return False


def _nonempty_regular_file(path: Path) -> bool:
    try:
        return bool(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
        )
    except OSError:
        return False


@dataclass(frozen=True)
class EvaluationRunInspection:
    path: Path
    identified: bool
    debug_ready: bool
    lifecycle_status: str
    completed: bool
    analysis_ready: bool
    resumable: bool
    writable: bool
    reason_codes: tuple[str, ...]

    def usable_for(self, purpose: RunPurpose) -> bool:
        if purpose == "debug":
            return self.debug_ready
        if purpose in {"latest_completed", "analysis"}:
            return self.analysis_ready
        if purpose == "resume":
            return self.resumable
        raise ValueError(f"Unsupported EvaluationRun discovery purpose: {purpose!r}")

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "identified": self.identified,
            "debug_ready": self.debug_ready,
            "lifecycle_status": self.lifecycle_status,
            "completed": self.completed,
            "analysis_ready": self.analysis_ready,
            "resumable": self.resumable,
            "writable": self.writable,
            "reason_codes": list(self.reason_codes),
        }


def inspect_evaluation_run(path: Path | str) -> EvaluationRunInspection:
    """Classify one EvaluationRun without creating, touching or deleting files."""

    run_dir = Path(path).expanduser()
    reasons: list[str] = []
    if not run_dir.is_dir():
        return EvaluationRunInspection(
            path=run_dir,
            identified=False,
            debug_ready=False,
            lifecycle_status="",
            completed=False,
            analysis_ready=False,
            resumable=False,
            writable=False,
            reason_codes=("run_dir_missing",),
        )
    if run_dir.is_symlink():
        reasons.append("run_dir_symlink")

    manifest, manifest_state = _read_json_mapping(run_dir / "run_manifest.json")
    if manifest_state != "ok" or manifest is None:
        reasons.append(f"run_manifest_{manifest_state}")
        return EvaluationRunInspection(
            path=run_dir,
            identified=False,
            debug_ready=False,
            lifecycle_status="",
            completed=False,
            analysis_ready=False,
            resumable=False,
            writable=_path_mode_allows_write(run_dir),
            reason_codes=tuple(reasons),
        )

    schema = str(manifest.get("schema") or "").strip()
    run_id = str(manifest.get("run_id") or "").strip()
    status = str(manifest.get("status") or "").strip().lower()
    if schema != _RUN_MANIFEST_SCHEMA:
        reasons.append("run_manifest_schema_mismatch")
    if not run_id:
        reasons.append("run_id_missing")
    elif run_id != run_dir.name:
        reasons.append("run_id_directory_mismatch")
    if status not in _TERMINAL_STATUSES | {"created", "running"}:
        reasons.append("run_status_unknown")

    identified = not any(
        code in reasons
        for code in (
            "run_dir_symlink",
            "run_manifest_schema_mismatch",
            "run_id_missing",
            "run_id_directory_mismatch",
            "run_status_unknown",
        )
    )

    status_summary, summary_state = _read_json_mapping(
        run_dir / "reports" / "run_status_summary.json"
    )
    bundle, bundle_state = _read_json_mapping(
        run_dir / "reports" / "results_bundle_manifest.json"
    )
    if summary_state != "ok":
        reasons.append(f"run_status_summary_{summary_state}")
    if bundle_state != "ok":
        reasons.append(f"results_bundle_manifest_{bundle_state}")

    summary_matches = bool(
        status_summary
        and status_summary.get("schema") == _RUN_STATUS_SCHEMA
        and str(status_summary.get("run_id") or "") == run_id
        and str(status_summary.get("status") or "").strip().lower() == status
    )
    if status_summary is not None and not summary_matches:
        reasons.append("run_status_summary_identity_mismatch")

    bundle_matches = bool(
        bundle
        and bundle.get("schema") == _RESULTS_BUNDLE_SCHEMA
        and str(bundle.get("run_id") or "") == run_id
        and _schema_version_at_least(bundle, 2)
    )
    if bundle is not None and not bundle_matches:
        reasons.append("results_bundle_manifest_identity_mismatch")

    recorded_version = _recorded_tool_version(manifest)
    legacy_analysis_contract = bool(
        recorded_version and recorded_version < (2, 75, 16)
    )
    measurement_set = build_measurement_set_contract(
        run_dir,
        allow_legacy_missing_run_id=legacy_analysis_contract,
    )
    declared_measurement_set = bundle.get("measurement_set") if bundle else None
    current_measurement_set_matches = bool(
        bundle_matches
        and isinstance(declared_measurement_set, Mapping)
        and dict(declared_measurement_set) == measurement_set
        and measurement_set.get("valid") is True
    )
    legacy_measurement_set_matches = bool(
        legacy_analysis_contract
        and bundle_matches
        and declared_measurement_set is None
        and measurement_set.get("valid") is True
    )
    measurement_set_matches = bool(
        current_measurement_set_matches or legacy_measurement_set_matches
    )
    measured_results = bool(
        bundle_matches
        and bundle
        and bundle.get("contains_measured_benchmarks") is True
        and measurement_set_matches
    )
    if bundle_matches and not measured_results:
        reasons.append("measurement_set_incomplete_or_unbound")

    completed = bool(
        identified
        and status == "ok"
        and summary_matches
        and bundle_matches
        and measured_results
    )
    scientific_report = run_dir / "reports" / "scientific" / "scientific_report.json"
    report_payload, report_state = _read_json_mapping(scientific_report)
    report_manifest, report_manifest_state = _read_json_mapping(
        run_dir / "reports" / "scientific" / "report_manifest.json"
    )
    if report_state != "ok":
        reasons.append(f"scientific_report_{report_state}")
    if report_manifest_state != "ok":
        reasons.append(f"scientific_report_manifest_{report_manifest_state}")
    report_rows = report_payload.get("rows") if report_payload else None
    current_report_bound = bool(
        report_state == "ok"
        and report_payload
        and report_payload.get("schema") == "onnx-splitpoint/scientific-report"
        and _schema_version_at_least(report_payload, 3)
        and str(report_payload.get("run_id") or "") == run_id
        and isinstance(report_rows, list)
        and any(isinstance(row, Mapping) and row for row in report_rows)
        and str(report_payload.get("measurement_set_sha256") or "")
        == str(measurement_set.get("measurement_set_sha256") or "")
        and isinstance(report_payload.get("measurement_result_count"), int)
        and not isinstance(report_payload.get("measurement_result_count"), bool)
        and report_payload.get("measurement_result_count")
        == measurement_set.get("result_count")
        and _report_manifest_binds(report_manifest, scientific_report)
        and _measurement_report_projection_matches(
            run_dir,
            measurement_set,
            report_rows,
            legacy=False,
        )
    )
    legacy_report_bound = bool(
        legacy_measurement_set_matches
        and report_state == "ok"
        and report_payload
        and report_payload.get("schema") == "onnx-splitpoint/scientific-report"
        and _schema_version_at_least(report_payload, 3)
        and str(report_payload.get("source_kind") or "") == "evaluation_run"
        and not str(report_payload.get("run_id") or "")
        and str(report_payload.get("profile_id") or "")
        == str(manifest.get("profile_id") or "")
        and _report_manifest_binds(report_manifest, scientific_report)
        and _measurement_report_projection_matches(
            run_dir,
            measurement_set,
            report_rows,
            legacy=True,
        )
    )
    report_bound = bool(current_report_bound or legacy_report_bound)
    if report_payload is not None and not report_bound:
        reasons.append("scientific_report_contract_mismatch")
    outputs = bundle.get("outputs") if bundle else None
    bundle_claims_report = bool(
        bundle
        and bundle.get("contains_scientific_report") is True
        and isinstance(outputs, Mapping)
        and "reports/scientific/scientific_report.json"
        in {str(value) for value in outputs.values()}
    )
    profile_path = _safe_regular_descendant(run_dir, "profile.yaml")
    profile_safe = profile_path is not None
    if not profile_safe:
        reasons.append("profile_yaml_unsafe_or_missing")
    analysis_ready = bool(
        completed and report_bound and bundle_claims_report and profile_safe
    )
    if completed and not analysis_ready:
        reasons.append("scientific_report_incomplete")

    debug_evidence_paths = (
        run_dir / "evaluation_workflow.log",
        run_dir / "artifact_index.json",
        run_dir / "reports" / "run_status_summary.json",
        run_dir / "reports" / "native_producer_stage.json",
    )
    debug_ready = bool(
        identified
        and any(_nonempty_regular_file(path) for path in debug_evidence_paths)
    )
    writable = _path_mode_allows_write(run_dir)
    resumable = bool(
        identified
        and status in _RESUMABLE_STATUSES
        and writable
        and (run_dir / "profile.yaml").is_file()
        and not (run_dir / "profile.yaml").is_symlink()
        and _resume_contract_complete(manifest)
    )
    if identified and status in _RESUMABLE_STATUSES and not writable:
        reasons.append("resume_target_read_only")

    return EvaluationRunInspection(
        path=run_dir,
        identified=identified,
        debug_ready=debug_ready,
        lifecycle_status=status,
        completed=completed,
        analysis_ready=analysis_ready,
        resumable=resumable,
        writable=writable,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True)
class RunDiscoveryResult:
    selected: Path
    status: str
    candidates: tuple[Path, ...]
    searched_roots: tuple[Path, ...]
    log_candidates: tuple[Path, ...]
    reason: str = ""
    purpose: RunPurpose = "debug"
    ignored: tuple[EvaluationRunInspection, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "selected": str(self.selected),
            "status": self.status,
            "candidates": [str(p) for p in self.candidates],
            "searched_roots": [str(p) for p in self.searched_roots],
            "log_candidates": [str(p) for p in self.log_candidates],
            "reason": self.reason,
            "purpose": self.purpose,
            "ignored": [item.as_dict() for item in self.ignored],
        }


def is_evaluation_run_dir(
    path: Path | str, *, purpose: RunPurpose = "debug"
) -> bool:
    """Return whether *path* is admitted for the requested use."""

    return inspect_evaluation_run(path).usable_for(purpose)


def _parse_timestamp(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _run_sort_key(path: Path) -> tuple[float, float, str]:
    created = 0.0
    for candidate in (
        path / "reports" / "run_status_summary.json",
        path / "reports" / "results_bundle_manifest.json",
        path / "run_manifest.json",
    ):
        payload, state = _read_json_mapping(candidate)
        if state != "ok" or payload is None:
            continue
        created = max(
            created,
            _parse_timestamp(payload.get("created_at")),
            _parse_timestamp(payload.get("finished_at")),
            _parse_timestamp(payload.get("started_at")),
            _parse_timestamp(payload.get("updated_at")),
        )
    try:
        mtime = max(
            path.stat().st_mtime,
            (path / "run_manifest.json").stat().st_mtime
            if (path / "run_manifest.json").exists()
            else 0.0,
        )
    except OSError:
        mtime = 0.0
    return created, mtime, path.name


def run_dirs_from_latest_log(
    path: Path | str, *, max_bytes: int = 2_000_000
) -> list[Path]:
    """Extract run-directory references from a workflow log, newest first."""

    log_path = Path(path).expanduser()
    if not log_path.is_file():
        return []
    try:
        size = int(log_path.stat().st_size)
        with log_path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(max(0, size - max_bytes))
            text = handle.read(max_bytes + 1).decode("utf-8", errors="replace")
    except OSError:
        return []

    found: list[Path] = []
    for line in text.splitlines():
        for pattern in _RUN_DIR_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            raw = str(match.group("path") or "").strip().strip("'\"")
            raw = raw.rstrip(".,;")
            if raw:
                found.append(Path(raw).expanduser())
            break
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in reversed(found):
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _iter_root_children(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return []
    try:
        return [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        return []


def discover_evaluation_run(
    *,
    preferred: Sequence[Path | str] = (),
    output_roots: Sequence[Path | str] = (),
    latest_logs: Sequence[Path | str] = (),
    prefer_valid_explicit: bool = True,
    purpose: RunPurpose = "debug",
) -> RunDiscoveryResult:
    """Resolve the newest run admitted for *purpose*, without writing.

    ``debug`` retains failed/partial runs. ``analysis`` and
    ``latest_completed`` only accept a terminal ``ok`` run whose status summary,
    bundle identity and canonical scientific report agree. ``resume`` accepts
    only a writable terminal non-successful run; the workflow's separate exact
    profile/options contract is still authoritative before any resume write.
    """

    if purpose not in {"debug", "latest_completed", "analysis", "resume"}:
        raise ValueError(f"Unsupported EvaluationRun discovery purpose: {purpose!r}")

    preferred_paths = [
        Path(value).expanduser()
        for value in preferred
        if str(value or "").strip()
    ]
    roots = [
        Path(value).expanduser()
        for value in output_roots
        if str(value or "").strip()
    ]
    log_paths = [
        Path(value).expanduser()
        for value in latest_logs
        if str(value or "").strip()
    ]
    inspections: dict[str, EvaluationRunInspection] = {}

    def inspect(candidate: Path) -> EvaluationRunInspection:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in inspections:
            inspections[key] = inspect_evaluation_run(candidate)
        return inspections[key]

    def explicit_usable(item: EvaluationRunInspection) -> bool:
        # A user may explicitly request a manifest-bound stub for forensic
        # inspection.  Automatic Debug selection still requires substantive
        # log/status evidence so an empty newer directory cannot hide the last
        # useful interrupted run.
        return item.identified if purpose == "debug" else item.usable_for(purpose)

    if prefer_valid_explicit:
        for candidate in preferred_paths:
            item = inspect(candidate)
            if explicit_usable(item):
                return RunDiscoveryResult(
                    selected=candidate,
                    status="explicit",
                    candidates=(candidate,),
                    searched_roots=tuple(roots),
                    log_candidates=tuple(log_paths),
                    reason=f"The explicitly selected path is valid for {purpose}.",
                    purpose=purpose,
                    ignored=tuple(
                        value for value in inspections.values()
                        if not value.usable_for(purpose)
                    ),
                )

    raw_candidates: list[Path] = []
    for candidate in preferred_paths:
        item = inspect(candidate)
        if item.usable_for(purpose):
            raw_candidates.append(candidate)
        elif candidate.is_dir() and not item.identified:
            roots.append(candidate)

    for root in list(roots):
        if inspect(root).usable_for(purpose):
            raw_candidates.append(root)
        for child in _iter_root_children(root):
            if inspect(child).usable_for(purpose):
                raw_candidates.append(child)
        implicit_log = root / "_latest_evaluation_workflow.log"
        if implicit_log.is_file():
            log_paths.append(implicit_log)

    for log_path in list(log_paths):
        for candidate in run_dirs_from_latest_log(log_path):
            if inspect(candidate).usable_for(purpose):
                raw_candidates.append(candidate)
            for root in roots:
                rebased = root / candidate.name
                if inspect(rebased).usable_for(purpose):
                    raw_candidates.append(rebased)

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)

    ignored = tuple(
        sorted(
            (
                value
                for value in inspections.values()
                if value.path.is_dir() and not value.usable_for(purpose)
            ),
            key=lambda item: _run_sort_key(item.path),
            reverse=True,
        )
    )
    if deduped:
        ordered = sorted(deduped, key=_run_sort_key, reverse=True)
        selected = ordered[0]
        return RunDiscoveryResult(
            selected=selected,
            status="discovered",
            candidates=tuple(ordered),
            searched_roots=tuple(dict.fromkeys(roots)),
            log_candidates=tuple(dict.fromkeys(log_paths)),
            reason=(
                f"Selected the newest EvaluationRun valid for {purpose}; "
                f"ignored {len(ignored)} unusable candidate(s)."
            ),
            purpose=purpose,
            ignored=ignored,
        )

    fallback = (
        preferred_paths[0]
        if preferred_paths
        else (roots[0] if roots else Path())
    )
    return RunDiscoveryResult(
        selected=fallback,
        status="not_found",
        candidates=(),
        searched_roots=tuple(dict.fromkeys(roots)),
        log_candidates=tuple(dict.fromkeys(log_paths)),
        reason=(
            f"No EvaluationRun satisfying the read-only {purpose} contract was found; "
            f"ignored {len(ignored)} unusable candidate(s)."
        ),
        purpose=purpose,
        ignored=ignored,
    )
