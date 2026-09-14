"""Read-only verification of already produced adapter and quality evidence.

This module deliberately does *not* create a prospective protocol freeze.  It
cross-checks an existing evaluation run, a structural Hailo canary and one
Full-ONNX self-reference result per model, then writes a path-neutral,
hash-bound retrospective receipt outside the source runs.  The receipt does
not promote Development data, change a scientific decision or make source
files physically immutable.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from . import __version__ as TOOL_VERSION
from .quality_result_contract import UNCERTAINTY_FIELDS, project_quality_component
from .quality_cache import (
    QualityFingerprintError,
    canonical_image_id,
    canonical_json,
    image_ids_fingerprint,
    json_fingerprint,
    prediction_fingerprint,
)


SCHEMA = "onnx-splitpoint/read-only-existing-evidence-verification"
SCHEMA_VERSION = 1
VERIFY_SCHEMA = (
    "onnx-splitpoint/read-only-existing-evidence-verification-check"
)
VERIFIED = "VERIFIED"
INCOMPLETE = "INCOMPLETE"
CONFLICT = "CONFLICT"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JSON_LIMIT = 64 * 1024 * 1024
_READ_CHUNK = 1024 * 1024
_QUALITY_BACKENDS = ("hailo8", "tensorrt")
_TENSOR_DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
}
_TENSOR_MAX_RANK = 8
_TENSOR_MAX_DIMENSION = (1 << 31) - 1
_TENSOR_MAX_ELEMENTS = (1 << 63) - 1
_MISSING_CODES = {
    "required_evidence_missing",
    "required_quality_row_missing",
    "required_self_reference_missing",
    "required_canary_model_missing",
    "prediction_payload_unavailable",
    "historical_reference_version_unavailable",
}


class ExistingEvidenceError(RuntimeError):
    """Structured refusal raised by the read-only verifier."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"{self.code}{':' + self.detail if self.detail else ''}")


@dataclass(frozen=True)
class FileObservation:
    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    data: bytes | None = None

    def stable_identity(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.device,
            self.inode,
            self.mode,
            self.size_bytes,
            self.mtime_ns,
            self.ctime_ns,
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExistingEvidenceError("noncanonical_payload", type(exc).__name__) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: Any, *, field: str) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw[7:]
    if not _HEX64.fullmatch(raw):
        raise ExistingEvidenceError("invalid_sha256", field)
    return raw


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExistingEvidenceError("invalid_integer", field)
    return value


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExistingEvidenceError("invalid_number", field)
    result = float(value)
    if not math.isfinite(result):
        raise ExistingEvidenceError("invalid_number", field)
    return result


def _need(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise ExistingEvidenceError(code, detail)


def _strict_json(data: bytes, *, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ExistingEvidenceError("duplicate_json_key", f"{label}:{key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ExistingEvidenceError("nonfinite_json_number", f"{label}:{value}")

    try:
        text = data.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except ExistingEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExistingEvidenceError("invalid_json", label) from exc


def _lexical_absolute(value: str | Path, *, label: str) -> Path:
    raw = os.fspath(Path(value).expanduser())
    if "\x00" in raw or any(part == ".." for part in Path(raw).parts):
        raise ExistingEvidenceError("unsafe_path", label)
    if not os.path.isabs(raw):
        raw = os.path.abspath(raw)
    return Path(os.path.normpath(raw))


def _logical_path(value: Any, *, field: str) -> str:
    raw = str(value or "")
    logical = PurePosixPath(raw)
    if (
        not raw
        or logical.is_absolute()
        or "\\" in raw
        or any(ord(char) < 32 for char in raw)
        or any(part in {"", ".", ".."} for part in logical.parts)
        or logical.as_posix() != raw
    ):
        raise ExistingEvidenceError("unsafe_manifest_path", field)
    return raw


def _open_directory_nofollow(path: Path, *, label: str) -> int:
    if not all(hasattr(os, name) for name in ("O_NOFOLLOW", "O_DIRECTORY")):
        raise ExistingEvidenceError("nofollow_platform_unsupported", label)
    absolute = _lexical_absolute(path, label=label)
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    fd = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                child = os.open(component, flags, dir_fd=fd)
            except FileNotFoundError as exc:
                raise ExistingEvidenceError("required_evidence_missing", label) from exc
            except OSError as exc:
                code = (
                    "unsafe_symlink_component"
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                    else "unsafe_input_directory"
                )
                raise ExistingEvidenceError(code, label) from exc
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise ExistingEvidenceError("unsafe_input_directory", label)
        return fd
    except Exception:
        os.close(fd)
        raise


def _read_regular_nofollow(
    value: str | Path,
    *,
    label: str,
    collect: bool,
) -> FileObservation:
    path = _lexical_absolute(value, label=label)
    parent_fd = _open_directory_nofollow(path.parent, label=f"{label}.parent")
    file_fd = -1
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            before_path = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISREG(before_path.st_mode):
                raise ExistingEvidenceError("input_not_regular_file", label)
            file_fd = os.open(path.name, flags, dir_fd=parent_fd)
        except FileNotFoundError as exc:
            raise ExistingEvidenceError("required_evidence_missing", label) from exc
        except OSError as exc:
            code = (
                "unsafe_symlink_component"
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}
                else "unsafe_input_file"
            )
            raise ExistingEvidenceError(code, label) from exc
        before_fd = os.fstat(file_fd)
        if not stat.S_ISREG(before_fd.st_mode):
            raise ExistingEvidenceError("input_not_regular_file", label)
        if (before_path.st_dev, before_path.st_ino) != (
            before_fd.st_dev,
            before_fd.st_ino,
        ):
            raise ExistingEvidenceError("input_path_replaced_during_open", label)
        if collect and before_fd.st_size > _JSON_LIMIT:
            raise ExistingEvidenceError("json_too_large", label)

        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if collect else None
        total = 0
        remaining = before_fd.st_size
        while remaining:
            block = os.read(file_fd, min(_READ_CHUNK, remaining))
            if not block:
                raise ExistingEvidenceError("input_modified_during_read", label)
            total += len(block)
            remaining -= len(block)
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
        if os.read(file_fd, 1):
            raise ExistingEvidenceError("input_modified_during_read", label)

        after_fd = os.fstat(file_fd)
        try:
            after_path = os.stat(
                path.name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError as exc:
            raise ExistingEvidenceError(
                "input_path_replaced_after_read",
                label,
            ) from exc
        identity_before = (
            before_fd.st_dev,
            before_fd.st_ino,
            before_fd.st_mode,
            before_fd.st_size,
            before_fd.st_mtime_ns,
            before_fd.st_ctime_ns,
        )
        identity_after = (
            after_fd.st_dev,
            after_fd.st_ino,
            after_fd.st_mode,
            after_fd.st_size,
            after_fd.st_mtime_ns,
            after_fd.st_ctime_ns,
        )
        if identity_before != identity_after or total != before_fd.st_size:
            raise ExistingEvidenceError("input_modified_during_read", label)
        if (after_path.st_dev, after_path.st_ino) != (
            after_fd.st_dev,
            after_fd.st_ino,
        ):
            raise ExistingEvidenceError("input_path_replaced_after_read", label)
        return FileObservation(
            path=path,
            sha256=digest.hexdigest(),
            size_bytes=total,
            device=after_fd.st_dev,
            inode=after_fd.st_ino,
            mode=after_fd.st_mode,
            mtime_ns=after_fd.st_mtime_ns,
            ctime_ns=after_fd.st_ctime_ns,
            data=b"".join(chunks) if chunks is not None else None,
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _path_inside(path: Path, root: Path) -> bool:
    try:
        return (
            os.path.commonpath((os.fspath(path), os.fspath(root)))
            == os.fspath(root)
        )
    except ValueError:
        return False


class _Attestor:
    def __init__(self) -> None:
        self._observations: dict[str, FileObservation] = {}
        self._logical: dict[str, set[str]] = {}

    def _remember(self, observation: FileObservation, logical: str) -> None:
        key = os.fspath(observation.path)
        previous = self._observations.get(key)
        if previous is not None and (
            previous.sha256 != observation.sha256
            or previous.stable_identity() != observation.stable_identity()
        ):
            raise ExistingEvidenceError("input_changed_between_reads", logical)
        self._observations[key] = observation
        self._logical.setdefault(key, set()).add(logical)

    def json(self, path: Path, *, logical: str) -> Mapping[str, Any]:
        observation = _read_regular_nofollow(path, label=logical, collect=True)
        self._remember(observation, logical)
        payload = _strict_json(observation.data or b"", label=logical)
        if not isinstance(payload, Mapping):
            raise ExistingEvidenceError("json_object_required", logical)
        return dict(payload)

    def file(self, path: Path, *, logical: str) -> FileObservation:
        observation = _read_regular_nofollow(path, label=logical, collect=False)
        self._remember(observation, logical)
        return observation

    def add(self, observation: FileObservation, *, logical: str) -> None:
        self._remember(observation, logical)

    def observation(self, path: Path, *, logical: str) -> FileObservation:
        key = os.fspath(_lexical_absolute(path, label=logical))
        observation = self._observations.get(key)
        if observation is None:
            raise ExistingEvidenceError("unattested_file_lookup", logical)
        return observation

    def stable_snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw_path, first in sorted(self._observations.items()):
            second = _read_regular_nofollow(
                first.path,
                label="snapshot_recheck",
                collect=False,
            )
            if (
                first.sha256 != second.sha256
                or first.stable_identity() != second.stable_identity()
            ):
                raise ExistingEvidenceError(
                    "attested_source_files_modified",
                    ",".join(sorted(self._logical[raw_path])),
                )
            for logical in sorted(self._logical[raw_path]):
                rows.append({
                    "logical_path": logical,
                    "sha256": first.sha256,
                    "size_bytes": first.size_bytes,
                })
        rows.sort(key=lambda row: (row["logical_path"], row["sha256"]))
        return rows


def _schema(
    payload: Mapping[str, Any],
    *,
    name: str,
    versions: Iterable[int],
    label: str,
) -> None:
    _need(payload.get("schema") == name, "schema_mismatch", label)
    version = payload.get("schema_version")
    _need(
        not isinstance(version, bool) and isinstance(version, int),
        "schema_version_mismatch",
        label,
    )
    _need(version in set(versions), "schema_version_mismatch", label)


def _canonical_location(value: Any, *, label: str) -> Path:
    return _lexical_absolute(str(value or ""), label=label)


def _same_location(value: Any, expected: Path, *, code: str, label: str) -> None:
    actual = _canonical_location(value, label=label)
    _need(actual == expected, code, label)


def _management_reference_location(
    value: Any,
    *,
    run_dir: Path,
    model: str,
) -> tuple[str, Path]:
    """Return the strictly run-bound legacy or immutable reference path."""

    label = f"{model}.management_reference.reference_path"
    raw = str(value or "").strip()
    _need(bool(raw), "source_run_binding_mismatch", label)
    legacy_parts = (
        "quality_management",
        "references",
        model,
        "canonical_cpu_reference.json",
    )
    run_root = _lexical_absolute(run_dir, label="run_dir")
    if os.path.isabs(raw):
        declared = _lexical_absolute(raw, label=label)
        try:
            relative = declared.relative_to(run_root)
        except ValueError:
            # Receipts embed the absolute source location.  A moved run may
            # be rebased only through one exact, complete allowed suffix.
            if declared.parts[-len(legacy_parts):] == legacy_parts:
                relative = Path(*legacy_parts)
            else:
                immutable_parts = declared.parts[-6:]
                _need(
                    len(immutable_parts) == 6
                    and immutable_parts[:3] == legacy_parts[:3]
                    and immutable_parts[3] == "by_source_contract"
                    and _HEX64.fullmatch(immutable_parts[4]) is not None
                    and immutable_parts[5:]
                    == ("canonical_cpu_reference.json",),
                    "source_run_binding_mismatch",
                    label,
                )
                relative = Path(*immutable_parts)
    else:
        logical = _logical_path(raw, field=label)
        relative = Path(*PurePosixPath(logical).parts)

    parts = relative.parts
    legacy = parts == legacy_parts
    immutable = (
        len(parts) == 6
        and parts[:3] == legacy_parts[:3]
        and parts[3] == "by_source_contract"
        and _HEX64.fullmatch(parts[4]) is not None
        and parts[5:] == ("canonical_cpu_reference.json",)
    )
    _need(
        legacy or immutable,
        "source_run_binding_mismatch",
        label,
    )
    reference_path = run_root.joinpath(*parts)
    _need(
        reference_path.is_relative_to(run_root),
        "source_run_binding_mismatch",
        label,
    )
    return PurePosixPath(*parts).as_posix(), reference_path


def _close_number(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _strictly_below_inclusive_threshold(value: float, threshold: float) -> bool:
    """Mirror the quality service's inclusive floating-point margin rule."""

    numeric_value = float(value)
    numeric_threshold = float(threshold)
    _need(
        math.isfinite(numeric_value) and math.isfinite(numeric_threshold),
        "quality_metric_inconsistent",
        "nonfinite_threshold",
    )
    if numeric_value >= numeric_threshold:
        return False
    tolerance = max(
        1e-15,
        8 * math.ulp(numeric_value),
        8 * math.ulp(numeric_threshold),
    )
    return (numeric_threshold - numeric_value) > tolerance


def _quality_component_projection(
    raw: Any,
    *,
    label: str,
    expected_metric: str,
    expected_margin: float,
    expected_n: int,
    contract_version: int = 0,
    prediction_identity_verified: bool = False,
) -> dict[str, Any]:
    _need(isinstance(raw, Mapping), "quality_metric_inconsistent", label)
    metric = str(raw.get("metric") or "")
    _need(metric == expected_metric, "quality_metric_inconsistent", f"{label}:metric")
    candidate = _finite_number(raw.get("candidate"), field=f"{label}.candidate")
    reference = _finite_number(raw.get("reference"), field=f"{label}.reference")
    delta = _finite_number(raw.get("delta"), field=f"{label}.delta")
    margin = _finite_number(raw.get("margin"), field=f"{label}.margin")
    _need(
        0.0 <= candidate <= 1.0 and 0.0 <= reference <= 1.0
        and -1.0 <= delta <= 1.0 and 0.0 <= margin <= 1.0,
        "quality_metric_inconsistent", f"{label}:range",
    )
    _need(_close_number(delta, candidate - reference), "quality_metric_inconsistent", f"{label}:delta")
    _need(_close_number(margin, expected_margin), "quality_metric_inconsistent", f"{label}:margin")
    _need(_strict_int(raw.get("n"), field=f"{label}.n", minimum=1) == expected_n,
          "quality_metric_inconsistent", f"{label}:n")
    decision = str(raw.get("decision") or "").strip().lower()
    status = str(raw.get("status") or "").strip().lower()
    below = _strictly_below_inclusive_threshold(delta, -margin)
    if contract_version >= 3 and raw.get("ci_computed") is False:
        _need(raw.get("ci_low") is None and raw.get("ci_high") is None
              and _strict_int(raw.get("bootstrap_repetitions"), field=f"{label}.bootstrap_repetitions", minimum=0) == 0,
              "quality_metric_inconsistent", f"{label}:uncomputed_confidence_bounds")
        ci_low = ci_high = None
        basis = str(raw.get("decision_basis") or "")
        gate = raw.get("gate_bound_value")
        if basis == "point_estimate_below_non_inferiority_margin":
            _need(below and _close_number(_finite_number(gate, field=f"{label}.gate_bound_value"), delta),
                  "quality_decision_mismatch", f"{label}:point_fail")
            expected_decision = "fail"
        elif basis == "bootstrap_not_computed_other_component_point_fail":
            _need(not below and gate is None, "quality_decision_mismatch", f"{label}:uncomputed_sibling")
            expected_decision = "inconclusive"
        elif basis == "candidate_reference_identical":
            _need(prediction_identity_verified and raw.get("prediction_identity_verified") is True
                  and delta == 0.0 and candidate == reference and gate == 0.0,
                  "quality_decision_mismatch", f"{label}:unbound_identity")
            expected_decision = "pass"
        else:
            raise ExistingEvidenceError("quality_decision_mismatch", f"{label}:unsupported_uncertainty_basis")
    else:
        ci_low = _finite_number(raw.get("ci_low"), field=f"{label}.ci_low")
        ci_high = _finite_number(raw.get("ci_high"), field=f"{label}.ci_high")
        _need(-1.0 <= ci_low <= ci_high <= 1.0, "quality_metric_inconsistent", f"{label}:range")
        if contract_version >= 3:
            _need(raw.get("ci_computed") is True
                  and _strict_int(raw.get("bootstrap_repetitions"), field=f"{label}.bootstrap_repetitions", minimum=1) > 0
                  and not raw.get("bootstrap_skipped_reason")
                  and raw.get("decision_basis") == "paired_bootstrap_lower_bound"
                  and _close_number(_finite_number(raw.get("gate_bound_value"), field=f"{label}.gate_bound_value"), ci_low),
                  "quality_metric_inconsistent", f"{label}:unattested_bootstrap")
        expected_decision = "fail" if below else (
            "pass" if not _strictly_below_inclusive_threshold(ci_low, -margin) else "inconclusive"
        )
    _need(decision == status == expected_decision, "quality_decision_mismatch", label)
    projected = project_quality_component(raw)
    return {
        "metric": metric, "candidate": candidate, "reference": reference,
        "delta": delta, "ci_low": projected.get("ci_low", ci_low),
        "ci_high": projected.get("ci_high", ci_high), "margin": margin, "n": expected_n,
        "decision": projected.get("decision", decision),
        **{field: projected[field] for field in (*UNCERTAINTY_FIELDS, "source_component_decision") if field in projected},
    }


def _quality_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    metric_fields = (
        "task_quality_metric",
        "task_quality_candidate",
        "task_quality_reference",
        "task_quality_delta",
        "task_quality_decision",
        "task_quality_ap50_candidate",
        "task_quality_ap50_reference",
        "task_quality_ap50_delta",
        "task_quality_ap50_decision",
        "task_quality_ap75_candidate",
        "task_quality_ap75_reference",
        "task_quality_ap75_delta",
        "task_quality_ap75_decision",
    )
    n = _strict_int(row.get("n"), field="quality.n", minimum=1)
    metric_config = row.get("metric_gate_config")
    _need(
        isinstance(metric_config, Mapping),
        "quality_metric_inconsistent",
        "metric_gate_config",
    )
    primary_metric = str(metric_config.get("primary_metric") or "")
    primary_margin = _finite_number(
        metric_config.get("non_inferiority_margin"),
        field="quality.metric_gate_config.non_inferiority_margin",
    )
    contract_version = int(row.get("quality_result_contract_version") or 0)
    reference_sha = str(row.get("reference_predictions_sha256") or "")
    identity_verified = bool(_HEX64.fullmatch(reference_sha) and row.get("candidate_predictions_sha256") == reference_sha)
    primary = _quality_component_projection(
        row.get("primary"),
        label="quality.primary",
        expected_metric=primary_metric,
        expected_margin=primary_margin,
        expected_n=n,
        contract_version=contract_version,
        prediction_identity_verified=identity_verified,
    )
    guardrail_config = metric_config.get("guardrails")
    guardrails = row.get("guardrails")
    configured_guardrails = row.get("configured_guardrails")
    _need(
        isinstance(guardrail_config, Mapping)
        and isinstance(guardrails, Mapping)
        and isinstance(configured_guardrails, list)
        and all(isinstance(name, str) and name for name in configured_guardrails)
        and len(configured_guardrails) == len(set(configured_guardrails)),
        "quality_metric_inconsistent",
        "quality.guardrails",
    )
    expected_guardrails = {
        str(key)[:-7]
        for key in guardrail_config
        if str(key).endswith("_margin")
    }
    _need(
        set(configured_guardrails) == expected_guardrails
        and set(str(key) for key in guardrails) == expected_guardrails
        and row.get("guardrail_contract_complete") is True,
        "quality_metric_inconsistent",
        "quality.guardrail_contract",
    )
    projected_guardrails = {
        name: _quality_component_projection(
            guardrails.get(name),
            label=f"quality.guardrails.{name}",
            expected_metric=name,
            expected_margin=_finite_number(
                guardrail_config.get(f"{name}_margin"),
                field=f"quality.metric_gate_config.guardrails.{name}_margin",
            ),
            expected_n=n,
            contract_version=contract_version,
            prediction_identity_verified=identity_verified,
        )
        for name in sorted(expected_guardrails)
    }
    component_decisions = {primary["decision"]} | {
        component["decision"] for component in projected_guardrails.values()
    }
    expected_decision = (
        "fail" if "fail" in component_decisions
        else "pass" if component_decisions == {"pass"}
        else "inconclusive"
    )
    # These are the authoritative aggregate aliases emitted by the preserved
    # management-paired-quality-result contract.  They must not be removable:
    # otherwise a resealed row could retain valid components while erasing the
    # row-level scientific conclusion from the attested projection.
    for field in (
        "decision",
        "scientific_status",
    ):
        _need(
            field in row,
            "quality_decision_mismatch",
            f"quality.{field}:missing",
        )
    for field in (
        "decision",
        "scientific_status",
        "task_quality_decision",
        "task_quality_status",
        "quality_decision",
        "aggregate_quality_decision",
    ):
        if field in row:
            _need(
                str(row.get(field) or "").strip().lower() == expected_decision,
                "quality_decision_mismatch",
                f"quality.{field}",
            )
    aliases: dict[str, tuple[Any, bool]] = {
        "task_quality_metric": (primary["metric"], False),
        "task_quality_candidate": (primary["candidate"], True),
        "task_quality_reference": (primary["reference"], True),
        "task_quality_delta": (primary["delta"], True),
    }
    for name in ("ap50", "ap75"):
        component = projected_guardrails.get(name)
        if component is None:
            continue
        aliases.update({
            f"task_quality_{name}_candidate": (component["candidate"], True),
            f"task_quality_{name}_reference": (component["reference"], True),
            f"task_quality_{name}_delta": (component["delta"], True),
            f"task_quality_{name}_decision": (component.get("source_component_decision", component["decision"]), False),
        })
    for field, (expected, numeric) in aliases.items():
        if field not in row:
            continue
        observed = row.get(field)
        matches = (
            _close_number(
                _finite_number(observed, field=f"quality.{field}"),
                float(expected),
            )
            if numeric else str(observed or "") == str(expected)
        )
        _need(matches, "quality_metric_inconsistent", f"quality.{field}")
    return {
        "backend": str(row.get("backend") or ""),
        "case_id": str(row.get("case_id") or ""),
        "n": n,
        "technical_status": str(row.get("technical_status") or ""),
        "scientific_status": expected_decision,
        "decision": expected_decision,
        "row_sha256": _sha256_json(row),
        "primary": primary,
        "guardrails": projected_guardrails,
        "metrics": {key: row.get(key) for key in metric_fields if key in row},
        "identities": {
            key: _digest(row.get(key), field=f"quality.{key}")
            for key in (
                "model_sha256",
                "preprocessing_contract_sha256",
                "decoder_contract_sha256",
                "nms_contract_sha256",
                "candidate_predictions_sha256",
                "reference_predictions_sha256",
                "annotations_sha256",
                "validation_dataset_sha256",
                "validation_ground_truth_sha256",
                "validation_image_ids_sha256",
                "quality_contract_sha256",
            )
        },
    }


def _find_one_quality_row(
    rows: Sequence[Any],
    *,
    model: str,
    backend: str,
    expected_items: int,
    run_id: str,
) -> Mapping[str, Any]:
    matches = [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("model_id") or "") == model
        and str(row.get("backend") or "") == backend
        and row.get("status") == "completed"
        and row.get("n") == expected_items
    ]
    _need(bool(matches), "required_quality_row_missing", f"{model}:{backend}")
    _need(len(matches) == 1, "duplicate_quality_row", f"{model}:{backend}")
    row = matches[0]
    _need(row.get("task") == "detection", "quality_task_mismatch", model)
    _need(
        str(row.get("eval_run_id") or "") == run_id,
        "source_run_binding_mismatch",
        f"quality:{model}:{backend}",
    )
    _strict_int(row.get("n"), field=f"quality.{model}.{backend}.n", minimum=1)
    return row


def _prediction_digest(
    records: Sequence[Any],
    *,
    payload_field: str,
    label: str,
) -> tuple[str, str]:
    _need(bool(records), "prediction_payload_invalid", label)
    _need(
        all(isinstance(record, Mapping) for record in records),
        "prediction_payload_invalid",
        label,
    )
    typed_records = [dict(record) for record in records]
    try:
        predictions_sha = prediction_fingerprint(
            typed_records,
            image_id_field="image_id",
            payload_field=payload_field,
        )
        image_ids_sha = image_ids_fingerprint(
            [record.get("image_id") for record in typed_records],
        )
    except QualityFingerprintError as exc:
        raise ExistingEvidenceError(
            "prediction_payload_invalid",
            label,
        ) from exc
    return predictions_sha, image_ids_sha


def _ground_truth_digests(
    reference_records: Sequence[Any],
    candidate_records: Sequence[Any],
    *,
    label: str,
) -> tuple[str, str]:
    try:
        reference_by_id = {
            canonical_image_id(record["image_id"]): dict(record)
            for record in reference_records
            if isinstance(record, Mapping) and "image_id" in record
        }
        candidate_by_id = {
            canonical_image_id(record["image_id"]): dict(record)
            for record in candidate_records
            if isinstance(record, Mapping) and "image_id" in record
        }
    except (KeyError, QualityFingerprintError) as exc:
        raise ExistingEvidenceError("prediction_payload_invalid", label) from exc
    _need(
        len(reference_by_id) == len(reference_records)
        and len(candidate_by_id) == len(candidate_records)
        and set(reference_by_id) == set(candidate_by_id),
        "prediction_payload_invalid",
        f"{label}:pairing",
    )
    identity: list[dict[str, Any]] = []
    for token in sorted(reference_by_id):
        reference = reference_by_id[token]
        candidate = candidate_by_id[token]
        reference_ground_truth = reference.get("ground_truth")
        candidate_ground_truth = candidate.get("ground_truth")
        _need(
            isinstance(reference_ground_truth, list)
            and isinstance(candidate_ground_truth, list)
            and canonical_json(reference_ground_truth)
            == canonical_json(candidate_ground_truth),
            "quality_ground_truth_mismatch",
            f"{label}:{token}",
        )
        identity.append({
            "image_id": candidate.get("image_id"),
            "ground_truth": candidate_ground_truth,
        })
    ground_truth_sha = json_fingerprint(identity)
    annotations_sha = json_fingerprint({
        "schema": "quality-annotations-v1",
        "records": sorted(identity, key=canonical_json),
    })
    return ground_truth_sha, annotations_sha


def _quality_contract_projection(
    contract: Any,
    *,
    declared_sha256: Any,
    model: str,
    model_sha256: str,
    row: Mapping[str, Any],
    label: str,
    bind_physical_components: bool,
) -> dict[str, Any]:
    _need(
        isinstance(contract, Mapping),
        "quality_contract_invalid",
        label,
    )
    payload = dict(contract)
    _schema(
        payload,
        name="onnx-splitpoint/central-detection-quality-contract",
        versions={1},
        label=label,
    )
    embedded_sha = _digest(
        payload.pop("quality_contract_sha256", ""),
        field=f"{label}.quality_contract_sha256",
    )
    declared_sha = _digest(
        declared_sha256,
        field=f"{label}.declared_quality_contract_sha256",
    )
    _need(
        embedded_sha == declared_sha == json_fingerprint(payload),
        "quality_contract_sha256_mismatch",
        label,
    )
    model_contract = contract.get("model")
    dataset = contract.get("dataset")
    preprocessing = contract.get("preprocessing")
    decoder = contract.get("decoder")
    nms = contract.get("nms")
    _need(
        all(
            isinstance(value, Mapping)
            for value in (model_contract, dataset, preprocessing, decoder, nms)
        ),
        "quality_contract_invalid",
        label,
    )
    _need(
        _digest(model_contract.get("sha256"), field=f"{label}.model")
        == model_sha256
        and str(model_contract.get("artifact_name") or "") == f"{model}.onnx"
        and _strict_int(
            dataset.get("image_count"),
            field=f"{label}.dataset.image_count",
            minimum=1,
        )
        == int(row.get("n") or 0)
        and _digest(
            dataset.get("manifest_sha256"),
            field=f"{label}.dataset.manifest_sha256",
        )
        == _digest(
            row.get("validation_dataset_sha256"),
            field=f"{label}.row.validation_dataset_sha256",
        )
        and _digest(
            dataset.get("image_ids_sha256"),
            field=f"{label}.dataset.image_ids_sha256",
        )
        == _digest(
            row.get("validation_image_ids_sha256"),
            field=f"{label}.row.validation_image_ids_sha256",
        )
        and _digest(
            dataset.get("ground_truth_sha256"),
            field=f"{label}.dataset.ground_truth_sha256",
        )
        == _digest(
            row.get("validation_ground_truth_sha256"),
            field=f"{label}.row.validation_ground_truth_sha256",
        )
        and _digest(
            preprocessing.get("sha256"),
            field=f"{label}.preprocessing.sha256",
        )
        == _digest(
            row.get("preprocessing_contract_sha256"),
            field=f"{label}.row.preprocessing_contract_sha256",
        ),
        "quality_contract_binding_mismatch",
        label,
    )
    if bind_physical_components:
        _need(
            _digest(decoder.get("sha256"), field=f"{label}.decoder.sha256")
            == _digest(
                row.get("decoder_contract_sha256"),
                field=f"{label}.row.decoder_contract_sha256",
            )
            and _digest(nms.get("sha256"), field=f"{label}.nms.sha256")
            == _digest(
                row.get("nms_contract_sha256"),
                field=f"{label}.row.nms_contract_sha256",
            )
            and _digest(
                contract.get("quality_record_endpoint_contract_sha256"),
                field=f"{label}.quality_record_endpoint_contract_sha256",
            )
            == _digest(
                row.get("quality_record_endpoint_contract_sha256"),
                field=f"{label}.row.quality_record_endpoint_contract_sha256",
            ),
            "quality_contract_binding_mismatch",
            f"{label}:physical_components",
        )
    return {
        "quality_contract_sha256": embedded_sha,
        "preprocessing_contract_sha256": _digest(
            preprocessing.get("sha256"),
            field=f"{label}.preprocessing.sha256",
        ),
        "dataset_manifest_sha256": _digest(
            dataset.get("manifest_sha256"),
            field=f"{label}.dataset.manifest_sha256",
        ),
        "image_ids_sha256": _digest(
            dataset.get("image_ids_sha256"),
            field=f"{label}.dataset.image_ids_sha256",
        ),
        "ground_truth_sha256": _digest(
            dataset.get("ground_truth_sha256"),
            field=f"{label}.dataset.ground_truth_sha256",
        ),
    }


def _quality_payload_projection(
    attestor: _Attestor,
    *,
    run_dir: Path,
    run_id: str,
    model: str,
    model_sha256: str,
    backend: str,
    row: Mapping[str, Any],
    expected_items: int,
) -> dict[str, Any]:
    """Verify the physical request, candidate and CPU-reference payloads."""
    source_request = _logical_path(
        row.get("source_request"),
        field=f"{model}.{backend}.source_request",
    )
    _need(
        source_request.startswith(f"models/{model}/benchmark_results/"),
        "source_run_binding_mismatch",
        f"{model}:{backend}:source_request",
    )
    request_path = run_dir / source_request
    request = attestor.json(
        request_path,
        logical=f"run/{source_request}",
    )
    request_observed = attestor.observation(
        request_path,
        logical=f"run/{source_request}",
    )
    _need(
        request_observed.sha256
        == _digest(
            row.get("source_request_sha256"),
            field=f"{model}.{backend}.source_request_sha256",
        ),
        "source_request_sha256_mismatch",
        f"{model}:{backend}",
    )
    _schema(
        request,
        name="onnx-splitpoint/central-quality-evaluation-request",
        versions={1},
        label=f"{model}.{backend}.quality_request",
    )
    _need(
        request.get("task") == "detection"
        and request.get("variant") == "full"
        and request.get("model_id") == model
        and request.get("eval_run_id") == run_id
        and request.get("source_run_id") == row.get("source_run_id")
        and request.get("setup_id") == row.get("source_setup_id")
        and request.get("backend")
        == ("hailo8" if backend == "hailo8" else "native_tensorrt")
        and request.get("pairing_key") == "image_id"
        and request.get("execution_location")
        in {"management_node", "central_management"}
        and request.get("status") == "pending_central_evaluation"
        and request.get("execution_role") == "full_quality_only"
        and request.get("performance_claims_emitted") is False
        and request.get("quality_canary_id") == row.get("quality_canary_id"),
        "quality_request_binding_mismatch",
        f"{model}:{backend}",
    )
    for field in ("full_only_plan_identity_sha256", "policy_sha256"):
        _need(
            _digest(request.get(field), field=f"{model}.{backend}.request.{field}")
            == _digest(row.get(field), field=f"{model}.{backend}.row.{field}"),
            "quality_request_binding_mismatch",
            f"{model}:{backend}:{field}",
        )
    _need(
        _strict_int(
            request.get("record_count"),
            field=f"{model}.{backend}.request.record_count",
            minimum=1,
        )
        == expected_items
        and _strict_int(
            request.get("reference_record_count"),
            field=f"{model}.{backend}.request.reference_record_count",
            minimum=1,
        )
        == expected_items,
        "quality_record_count_mismatch",
        f"{model}:{backend}:request",
    )
    _need(
        str(request.get("runtime_precision_identity") or "")
        == str(row.get("runtime_precision_identity") or ""),
        "quality_identity_mismatch",
        f"{model}:{backend}:runtime_precision_identity",
    )
    request_quality_sha = _digest(
        request.get("quality_contract_sha256"),
        field=f"{model}.{backend}.request.quality_contract_sha256",
    )
    _need(
        request_quality_sha
        == _digest(
            row.get("quality_contract_sha256"),
            field=f"{model}.{backend}.quality_contract_sha256",
        ),
        "quality_identity_mismatch",
        f"{model}:{backend}:quality_contract",
    )
    request_contract_projection = _quality_contract_projection(
        request.get("quality_contract"),
        declared_sha256=request_quality_sha,
        model=model,
        model_sha256=model_sha256,
        row=row,
        label=f"{model}.{backend}.request.quality_contract",
        bind_physical_components=True,
    )

    candidate_descriptor = request.get("candidate")
    _need(
        isinstance(candidate_descriptor, Mapping),
        "prediction_payload_unavailable",
        f"{model}:{backend}:candidate_descriptor",
    )
    candidate_name = _logical_path(
        candidate_descriptor.get("path"),
        field=f"{model}.{backend}.candidate.path",
    )
    _need(
        "/" not in candidate_name,
        "unsafe_manifest_path",
        f"{model}:{backend}:candidate",
    )
    candidate_path = request_path.parent / candidate_name
    candidate = attestor.json(
        candidate_path,
        logical=f"run/{source_request.rsplit('/', 1)[0]}/{candidate_name}",
    )
    candidate_observed = attestor.observation(
        candidate_path,
        logical=f"{model}.{backend}.candidate",
    )
    _need(
        candidate_observed.sha256
        == _digest(
            candidate_descriptor.get("sha256"),
            field=f"{model}.{backend}.candidate.sha256",
        ),
        "candidate_payload_sha256_mismatch",
        f"{model}:{backend}",
    )
    _need(
        candidate_observed.size_bytes
        == _strict_int(
            candidate_descriptor.get("size_bytes"),
            field=f"{model}.{backend}.candidate.size_bytes",
            minimum=1,
        ),
        "candidate_payload_size_mismatch",
        f"{model}:{backend}",
    )
    _schema(
        candidate,
        name="onnx-splitpoint/task-quality-candidate-input",
        versions={1},
        label=f"{model}.{backend}.candidate",
    )
    _need(
        candidate.get("task") == "detection"
        and candidate.get("variant") == "full"
        and candidate.get("pairing_key") == "image_id"
        and candidate.get("model_id") == model
        and candidate.get("eval_run_id") == run_id
        and candidate.get("source_run_id") == row.get("source_run_id")
        and candidate.get("setup_id") == row.get("source_setup_id")
        and candidate.get("backend")
        == ("hailo8" if backend == "hailo8" else "native_tensorrt")
        and candidate.get("execution_role") == "full_quality_only"
        and candidate.get("performance_claims_emitted") is False
        and candidate.get("quality_canary_id") == row.get("quality_canary_id")
        and candidate.get("runtime_precision_identity")
        == row.get("runtime_precision_identity")
        and candidate.get("quality_contract_sha256")
        == request.get("quality_contract_sha256")
        and candidate.get("full_only_plan_identity_sha256")
        == request.get("full_only_plan_identity_sha256"),
        "quality_request_binding_mismatch",
        f"{model}:{backend}:candidate",
    )
    _need(
        candidate.get("quality_contract") == request.get("quality_contract"),
        "quality_contract_binding_mismatch",
        f"{model}:{backend}:candidate",
    )
    _quality_contract_projection(
        candidate.get("quality_contract"),
        declared_sha256=candidate.get("quality_contract_sha256"),
        model=model,
        model_sha256=model_sha256,
        row=row,
        label=f"{model}.{backend}.candidate.quality_contract",
        bind_physical_components=True,
    )
    candidate_records = candidate.get("records")
    _need(
        isinstance(candidate_records, list)
        and len(candidate_records) == expected_items,
        "quality_record_count_mismatch",
        f"{model}:{backend}:candidate",
    )
    candidate_predictions_sha, candidate_image_ids_sha = _prediction_digest(
        candidate_records,
        payload_field="candidate",
        label=f"{model}:{backend}:candidate",
    )
    _need(
        candidate_predictions_sha
        == _digest(
            row.get("candidate_predictions_sha256"),
            field=f"{model}.{backend}.candidate_predictions_sha256",
        ),
        "candidate_predictions_sha256_mismatch",
        f"{model}:{backend}",
    )

    management_reference = row.get("management_cpu_reference")
    _need(
        isinstance(management_reference, Mapping),
        "prediction_payload_unavailable",
        f"{model}:management_reference",
    )
    reference_relative, reference_path = _management_reference_location(
        management_reference.get("reference_path"),
        run_dir=run_dir,
        model=model,
    )
    reference_parts = PurePosixPath(reference_relative).parts
    immutable_reference = "by_source_contract" in reference_parts
    if immutable_reference:
        _need(
            _digest(
                management_reference.get("source_contract_sha256"),
                field=f"{model}.management_reference.source_contract_sha256",
            )
            == reference_parts[4]
            and management_reference.get("reference_storage")
            == "immutable_source_contract"
            and management_reference.get("reference_immutable") is True,
            "source_run_binding_mismatch",
            f"{model}:{backend}:immutable_management_reference",
        )
    else:
        _need(
            management_reference.get("reference_storage")
            != "immutable_source_contract"
            and management_reference.get("reference_immutable") is not True,
            "source_run_binding_mismatch",
            f"{model}:{backend}:legacy_management_reference",
        )
    reference = attestor.json(
        reference_path,
        logical=f"run/{reference_relative}",
    )
    reference_observed = attestor.observation(
        reference_path,
        logical=f"run/{reference_relative}",
    )
    _need(
        reference_observed.sha256
        == _digest(
            management_reference.get("reference_sha256"),
            field=f"{model}.management_reference.sha256",
        ),
        "historical_reference_version_unavailable",
        f"{model}:{backend}",
    )
    if immutable_reference or "reference_size_bytes" in management_reference:
        _need(
            reference_observed.size_bytes
            == _strict_int(
                management_reference.get("reference_size_bytes"),
                field=f"{model}.management_reference.reference_size_bytes",
                minimum=1,
            ),
            "historical_reference_version_unavailable",
            f"{model}:{backend}:management_reference_size",
        )
    _schema(
        reference,
        name="onnx-splitpoint/task-quality-reference-input",
        versions={1},
        label=f"{model}.management_reference",
    )
    _need(
        reference.get("task") == "detection"
        and reference.get("pairing_key") == "image_id"
        and str(reference.get("reference_role") or "canonical_cpu_ort")
        in {"canonical_cpu_ort", "onnxruntime_cpu", "canonical_full_onnx"}
        and reference.get("semantic_reference_only") is not False,
        "quality_request_binding_mismatch",
        f"{model}:management_reference",
    )
    reference_contract_projection = _quality_contract_projection(
        reference.get("quality_contract"),
        declared_sha256=reference.get("quality_contract_sha256"),
        model=model,
        model_sha256=model_sha256,
        row=row,
        label=f"{model}.management_reference.quality_contract",
        bind_physical_components=False,
    )
    reference_records = reference.get("records")
    _need(
        isinstance(reference_records, list)
        and len(reference_records) == expected_items
        and _strict_int(
            management_reference.get("record_count"),
            field=f"{model}.management_reference.record_count",
            minimum=1,
        )
        == expected_items,
        "quality_record_count_mismatch",
        f"{model}:management_reference",
    )
    reference_predictions_sha, reference_image_ids_sha = _prediction_digest(
        reference_records,
        payload_field="reference",
        label=f"{model}:management_reference",
    )
    _need(
        reference_predictions_sha
        == _digest(
            row.get("reference_predictions_sha256"),
            field=f"{model}.{backend}.reference_predictions_sha256",
        ),
        "reference_predictions_sha256_mismatch",
        f"{model}:{backend}",
    )
    ground_truth_sha, annotations_sha = _ground_truth_digests(
        reference_records,
        candidate_records,
        label=f"{model}:{backend}",
    )
    _need(
        ground_truth_sha
        == _digest(
            row.get("validation_ground_truth_sha256"),
            field=f"{model}.{backend}.validation_ground_truth_sha256",
        ),
        "quality_ground_truth_mismatch",
        f"{model}:{backend}",
    )
    _need(
        annotations_sha
        == _digest(
            row.get("annotations_sha256"),
            field=f"{model}.{backend}.annotations_sha256",
        ),
        "quality_ground_truth_mismatch",
        f"{model}:{backend}:annotations",
    )

    reference_descriptor = request.get("reference")
    _need(
        isinstance(reference_descriptor, Mapping),
        "quality_request_binding_mismatch",
        f"{model}:{backend}:reference_descriptor",
    )
    _need(
        str(reference_descriptor.get("source") or "")
        == "management_cpu_reference"
        and str(reference_descriptor.get("reference_role") or "")
        in {"canonical_cpu_ort", "onnxruntime_cpu", "canonical_full_onnx"}
        and reference_descriptor.get("semantic_reference_only") is True
        and reference_descriptor.get("required") is True,
        "quality_request_binding_mismatch",
        f"{model}:{backend}:reference_role",
    )
    _need(
        _strict_int(
            reference_descriptor.get("record_count"),
            field=f"{model}.{backend}.reference.record_count",
            minimum=1,
        )
        == expected_items
        and _digest(
            reference_descriptor.get("quality_contract_sha256"),
            field=f"{model}.{backend}.reference.quality_contract_sha256",
        )
        == reference_contract_projection["quality_contract_sha256"],
        "quality_request_binding_mismatch",
        f"{model}:{backend}:reference_descriptor",
    )

    # Historical requests deliberately carry only the semantic placeholder
    # above.  The management row and the physical reference (validated before
    # this block) are authoritative.  Newer producers may additionally seal
    # post-execution file identities; every identity that is present remains
    # fail-closed without making those fields mandatory retroactively.
    if "path" in reference_descriptor:
        descriptor_relative, descriptor_path = (
            _management_reference_location(
                reference_descriptor.get("path"),
                run_dir=run_dir,
                model=model,
            )
        )
        _need(
            descriptor_relative == reference_relative
            and descriptor_path == reference_path,
            "source_run_binding_mismatch",
            f"{model}:{backend}:request_reference_path",
        )
    if "sha256" in reference_descriptor:
        _need(
            reference_observed.sha256
            == _digest(
                reference_descriptor.get("sha256"),
                field=f"{model}.{backend}.reference.sha256",
            ),
            "historical_reference_version_unavailable",
            f"{model}:{backend}:request_reference_sha256",
        )
    if "size_bytes" in reference_descriptor:
        _need(
            reference_observed.size_bytes
            == _strict_int(
                reference_descriptor.get("size_bytes"),
                field=f"{model}.{backend}.reference.size_bytes",
                minimum=1,
            ),
            "historical_reference_version_unavailable",
            f"{model}:{backend}:request_reference_size",
        )
    if "prediction_sha256" in reference_descriptor:
        _need(
            reference_predictions_sha
            == _digest(
                reference_descriptor.get("prediction_sha256"),
                field=(
                    f"{model}.{backend}.reference.prediction_sha256"
                ),
            ),
            "quality_request_binding_mismatch",
            f"{model}:{backend}:request_reference_prediction",
        )
    expected_ids = reference_descriptor.get("expected_image_ids")
    _need(
        isinstance(expected_ids, list) and len(expected_ids) == expected_items,
        "quality_record_count_mismatch",
        f"{model}:{backend}:expected_image_ids",
    )
    try:
        expected_image_ids_sha = image_ids_fingerprint(expected_ids)
    except QualityFingerprintError as exc:
        raise ExistingEvidenceError(
            "prediction_payload_invalid",
            f"{model}:{backend}:expected_image_ids",
        ) from exc
    declared_image_ids_sha = _digest(
        reference_descriptor.get("expected_image_ids_sha256"),
        field=f"{model}.{backend}.expected_image_ids_sha256",
    )
    row_image_ids_sha = _digest(
        row.get("validation_image_ids_sha256"),
        field=f"{model}.{backend}.validation_image_ids_sha256",
    )
    _need(
        len(
            {
                expected_image_ids_sha,
                declared_image_ids_sha,
                row_image_ids_sha,
                candidate_image_ids_sha,
                reference_image_ids_sha,
            }
        )
        == 1,
        "quality_identity_mismatch",
        f"{model}:{backend}:image_ids",
    )
    _need(
        _digest(
            request.get("expected_image_ids_sha256"),
            field=f"{model}.{backend}.request.expected_image_ids_sha256",
        )
        == declared_image_ids_sha
        and request.get("expected_image_ids") == expected_ids,
        "quality_identity_mismatch",
        f"{model}:{backend}:request_image_ids",
    )

    completed_endpoint = str(
        request.get("completed_task_endpoint_contract_hash") or ""
    ).strip()
    invariant_contract = ""
    completion = request.get("candidate_execution_completion_contract")
    if backend == "hailo8":
        _need(
            isinstance(completion, Mapping),
            "quality_request_binding_mismatch",
            f"{model}:hailo8:completion",
        )
        invariant_contract = _digest(
            completion.get("frozen_postprocess_invariant_contract_sha256"),
            field=f"{model}.hailo8.invariant_contract",
        )
        completion_body = dict(completion)
        embedded_completion_sha = _digest(
            completion_body.pop("contract_sha256", ""),
            field=f"{model}.hailo8.completion.contract_sha256",
        )
        request_completion_sha = _digest(
            request.get("candidate_execution_completion_contract_sha256"),
            field=f"{model}.hailo8.completion_sha256",
        )
        _need(
            embedded_completion_sha
            == request_completion_sha
            == json_fingerprint(completion_body),
            "quality_request_binding_mismatch",
            f"{model}:hailo8:completion_sha256",
        )
        completed_fields = (
            "candidate_execution_completion_contract",
            "candidate_execution_completion_contract_sha256",
            "completed_task_endpoint_contract",
            "completed_task_endpoint_contract_hash",
            "completed_task_output_endpoint_id",
            "completed_task_endpoint_attestation",
            "completed_task_endpoint_attestation_sha256",
            "quality_join_endpoint",
        )
        _need(
            all(candidate.get(field) == request.get(field) for field in completed_fields),
            "quality_request_binding_mismatch",
            f"{model}:hailo8:candidate_completion",
        )
        endpoint_attestation = request.get("completed_task_endpoint_attestation")
        _need(
            isinstance(endpoint_attestation, Mapping)
            and _digest(
                request.get("completed_task_endpoint_attestation_sha256"),
                field=f"{model}.hailo8.endpoint_attestation_sha256",
            )
            == json_fingerprint(endpoint_attestation),
            "quality_request_binding_mismatch",
            f"{model}:hailo8:endpoint_attestation",
        )
        _need(
            _digest(
                completion.get("hailo_hef_sha256"),
                field=f"{model}.hailo8.completion.hef",
            )
            == str(row.get("runtime_precision_identity") or "").removeprefix(
                "hailo_hef_sha256:"
            ),
            "quality_identity_mismatch",
            f"{model}:hailo8:completion_hef",
        )
        _need(
            _digest(
                completed_endpoint,
                field=f"{model}.hailo8.completed_endpoint",
            )
            == _digest(
                row.get("completed_task_endpoint_contract_hash"),
                field=f"{model}.hailo8.row_completed_endpoint",
            ),
            "quality_identity_mismatch",
            f"{model}:hailo8:completed_endpoint",
        )

    return {
        "source_request_sha256": request_observed.sha256,
        "source_request_size_bytes": request_observed.size_bytes,
        "candidate_file_sha256": candidate_observed.sha256,
        "candidate_file_size_bytes": candidate_observed.size_bytes,
        "candidate_predictions_sha256": candidate_predictions_sha,
        "reference_file_sha256": reference_observed.sha256,
        "reference_file_size_bytes": reference_observed.size_bytes,
        "reference_predictions_sha256": reference_predictions_sha,
        "image_ids_sha256": row_image_ids_sha,
        "ground_truth_sha256": ground_truth_sha,
        "annotations_sha256": annotations_sha,
        "completed_task_endpoint_contract_hash": completed_endpoint,
        "frozen_postprocess_invariant_contract_sha256": invariant_contract,
        "full_only_plan_identity_sha256": _digest(
            row.get("full_only_plan_identity_sha256"),
            field=f"{model}.{backend}.full_only_plan_identity_sha256",
        ),
        "request_quality_contract": request_contract_projection,
        "reference_quality_contract": reference_contract_projection,
    }


def _report_artifacts(
    attestor: _Attestor,
    report_root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    rows = manifest.get("artifacts")
    _need(isinstance(rows, list), "report_manifest_invalid", "artifacts")
    relative_count = 0
    provenance_only = 0
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        _need(isinstance(raw, Mapping), "report_manifest_invalid", str(index))
        path_value = str(raw.get("path") or "")
        if PurePosixPath(path_value).is_absolute():
            provenance_only += 1
            continue
        logical = _logical_path(path_value, field=f"report.artifacts[{index}].path")
        _need(logical not in seen, "duplicate_report_artifact", logical)
        seen.add(logical)
        observed = attestor.file(
            report_root / logical,
            logical=f"run/reports/scientific/{logical}",
        )
        _need(
            observed.sha256 == _digest(raw.get("sha256"), field=logical),
            "report_artifact_sha256_mismatch",
            logical,
        )
        _need(
            observed.size_bytes
            == _strict_int(raw.get("size_bytes"), field=f"{logical}.size"),
            "report_artifact_size_mismatch",
            logical,
        )
        relative_count += 1
    _need(relative_count > 0, "report_manifest_empty", "relative artifacts")
    return {
        "verified_relative_artifact_count": relative_count,
        "absolute_provenance_entry_count": provenance_only,
    }


def _artifact_index_projection(
    payload: Mapping[str, Any],
    *,
    attested_artifacts: Sequence[Mapping[str, Any]],
    models: Iterable[str],
) -> dict[str, Any]:
    rows = payload.get("artifacts")
    _need(
        isinstance(rows, list) and rows,
        "artifact_index_empty",
        "artifacts",
    )
    bindings: dict[str, set[tuple[str, int]]] = {}
    for index, raw in enumerate(rows):
        _need(
            isinstance(raw, Mapping),
            "artifact_index_invalid",
            str(index),
        )
        relative = _logical_path(
            raw.get("path"),
            field=f"artifact_index.artifacts[{index}].path",
        )
        digest = _digest(
            raw.get("sha256"),
            field=f"artifact_index.artifacts[{index}].sha256",
        )
        size_bytes = _strict_int(
            raw.get("size_bytes"),
            field=f"artifact_index.artifacts[{index}].size_bytes",
        )
        bindings.setdefault(relative, set()).add((digest, size_bytes))

    required = {
        "quality_management/central_quality_summary.json",
        "reports/results_bundle_manifest.json",
        "reports/scientific/report_manifest.json",
    }
    for model in models:
        suite = f"models/{model}/benchmark_set/legacy_suite"
        required.update({
            f"{suite}/output_contracts.json",
            f"{suite}/hailo/hailo8/full/compiled.hef",
            f"{suite}/hailo/hailo8/full/hailo_hef_build_receipt.json",
        })

    observed: dict[str, tuple[str, int]] = {}
    for raw in attested_artifacts:
        logical = str(raw.get("logical_path") or "")
        if not logical.startswith("run/"):
            continue
        relative = logical.removeprefix("run/")
        if relative.startswith("reports/scientific/"):
            required.add(relative)
        if relative in required:
            observed[relative] = (
                _digest(raw.get("sha256"), field=f"attested.{relative}.sha256"),
                _strict_int(
                    raw.get("size_bytes"),
                    field=f"attested.{relative}.size_bytes",
                ),
            )

    missing_observations = required - set(observed)
    _need(
        not missing_observations,
        "artifact_index_binding_missing",
        ",".join(sorted(missing_observations)),
    )
    for relative in sorted(required):
        _need(
            observed[relative] in bindings.get(relative, set()),
            "artifact_index_binding_mismatch",
            relative,
        )
    return {
        "indexed_row_count": len(rows),
        "indexed_unique_path_count": len(bindings),
        "verified_binding_count": len(required),
        "verified_paths": sorted(required),
    }


def _manifest_dump_members(
    attestor: _Attestor,
    dump_root: Path,
    payload: Mapping[str, Any],
    *,
    model: str,
    frozen_contract: Mapping[str, Any],
) -> int:
    outputs = payload.get("outputs")
    _need(isinstance(outputs, list) and outputs, "output_manifest_invalid", model)
    signature = frozen_contract.get("raw_output_tensor_signature")
    _need(
        isinstance(signature, Mapping),
        "raw_output_tensor_signature_mismatch",
        model,
    )
    signature_rows = signature.get("tensors")
    _need(
        isinstance(signature_rows, list)
        and _strict_int(
            signature.get("tensor_count"),
            field=f"{model}.raw_output_tensor_signature.tensor_count",
            minimum=1,
        )
        == len(signature_rows)
        == len(outputs),
        "raw_output_tensor_signature_mismatch",
        f"{model}:tensor_count",
    )
    seen_files: set[str] = set()
    seen_names: set[str] = set()
    for index, row in enumerate(outputs):
        _need(isinstance(row, Mapping), "output_manifest_invalid", model)
        signature_row = signature_rows[index]
        _need(
            isinstance(signature_row, Mapping),
            "raw_output_tensor_signature_mismatch",
            f"{model}:{index}",
        )
        name = str(row.get("name") or "").strip()
        _need(
            bool(name) and name not in seen_names,
            "output_tensor_metadata_invalid",
            f"{model}:outputs[{index}].name",
        )
        seen_names.add(name)
        logical = _logical_path(row.get("file"), field=f"{model}.outputs[{index}]")
        _need("/" not in logical, "unsafe_manifest_path", logical)
        _need(logical not in seen_files, "duplicate_output_member", logical)
        seen_files.add(logical)
        shape = row.get("shape")
        _need(
            isinstance(shape, list)
            and 0 < len(shape) <= _TENSOR_MAX_RANK
            and all(
                not isinstance(dimension, bool)
                and isinstance(dimension, int)
                and 0 < dimension <= _TENSOR_MAX_DIMENSION
                for dimension in shape
            ),
            "output_tensor_metadata_invalid",
            f"{model}:{logical}:shape",
        )
        element_count = 1
        for dimension in shape:
            _need(
                element_count <= _TENSOR_MAX_ELEMENTS // dimension,
                "output_tensor_metadata_invalid",
                f"{model}:{logical}:shape_product",
            )
            element_count *= dimension
        dtype = str(row.get("dtype") or "").strip()
        dtype_bytes = _TENSOR_DTYPE_BYTES.get(dtype)
        _need(
            dtype_bytes is not None,
            "output_tensor_metadata_invalid",
            f"{model}:{logical}:dtype",
        )
        declared_bytes = _strict_int(
            row.get("bytes"),
            field=f"{logical}.bytes",
            minimum=1,
        )
        _need(
            element_count * dtype_bytes == declared_bytes,
            "output_tensor_size_mismatch",
            f"{model}:{logical}:shape_dtype",
        )
        _need(
            _strict_int(
                signature_row.get("index"),
                field=f"{model}.raw_output_tensor_signature[{index}].index",
            )
            == index
            and signature_row.get("name") == name
            and signature_row.get("shape") == shape
            and signature_row.get("dtype") == dtype
            and _strict_int(
                signature_row.get("rank"),
                field=f"{model}.raw_output_tensor_signature[{index}].rank",
                minimum=1,
            )
            == len(shape),
            "raw_output_tensor_signature_mismatch",
            f"{model}:{index}",
        )
        observed = attestor.file(
            dump_root / logical,
            logical=f"canary/{model}/outputs/{logical}",
        )
        _need(
            observed.sha256 == _digest(row.get("sha256"), field=logical),
            "output_tensor_sha256_mismatch",
            f"{model}:{logical}",
        )
        _need(
            observed.size_bytes
            == declared_bytes,
            "output_tensor_size_mismatch",
            f"{model}:{logical}",
        )
    return len(seen_files)


def _verify_output_manifest_identity(
    payload: Mapping[str, Any],
    *,
    model: str,
    setup_id: str,
) -> None:
    _schema(
        payload,
        name="onnx-splitpoint/runner-output-dump",
        versions={4},
        label=f"{model}.output_manifest",
    )
    _need(
        payload.get("model") == model
        and payload.get("task") == "detection"
        and payload.get("case") == "full"
        and payload.get("backend") == "native_full_hailo8"
        and payload.get("comparison_backend") == "hailo8"
        and payload.get("setup_id") == setup_id,
        "output_manifest_identity_mismatch",
        model,
    )


def _verify_canary_benchmark_sets(
    canary: Mapping[str, Any],
    *,
    run_dir: Path,
    models: Iterable[str],
) -> list[str]:
    expected_relatives = [
        f"models/{model}/benchmark_set/legacy_suite"
        for model in sorted(models)
    ]
    expected_paths = [run_dir / relative for relative in expected_relatives]
    raw = canary.get("benchmark_sets")
    _need(
        isinstance(raw, list)
        and len(raw) == len(expected_paths)
        and all(isinstance(value, str) and value.strip() for value in raw),
        "canary_benchmark_sets_mismatch",
        "cardinality",
    )
    declared_paths = [
        _canonical_location(value, label=f"canary.benchmark_sets[{index}]")
        for index, value in enumerate(raw)
    ]
    _need(
        len(set(declared_paths)) == len(declared_paths)
        and set(declared_paths) == set(expected_paths),
        "canary_benchmark_sets_mismatch",
        "exact_set",
    )
    for index, path in enumerate(expected_paths):
        descriptor = _open_directory_nofollow(
            path,
            label=f"canary.benchmark_sets[{index}]",
        )
        os.close(descriptor)
    return expected_relatives


def _self_reference_projection(
    payload: Mapping[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    _schema(
        payload,
        name="onnx-splitpoint/native-yolo-full-self-reference-probe",
        versions={6},
        label=f"self_reference:{model}",
    )
    _need(payload.get("model") == model, "self_reference_model_mismatch", model)
    for field in ("ok", "semantic_available", "semantic_ok"):
        _need(payload.get(field) is True, "self_reference_not_passed", f"{model}:{field}")
    _need(
        payload.get("diagnosis") == "native_semantic_matches_full_self_reference",
        "self_reference_not_passed",
        f"{model}:diagnosis",
    )
    _need(payload.get("case") == "full", "self_reference_case_mismatch", model)
    _need(
        payload.get("expected_contract_family") == "decoded_nms"
        and payload.get("contract_family_match") is True,
        "self_reference_contract_mismatch",
        model,
    )
    _need(
        payload.get("numerical_similarity_pass") is True
        and payload.get("numerical_similarity_status") == "passed",
        "self_reference_not_passed",
        f"{model}:similarity",
    )
    value = _finite_number(
        payload.get("numerical_similarity_value"),
        field=f"{model}.similarity",
    )
    threshold = _finite_number(
        payload.get("numerical_similarity_threshold"),
        field=f"{model}.similarity_threshold",
    )
    mean_iou = _finite_number(
        payload.get("numerical_similarity_mean_iou"),
        field=f"{model}.mean_iou",
    )
    mean_iou_threshold = _finite_number(
        payload.get("numerical_similarity_mean_iou_threshold"),
        field=f"{model}.mean_iou_threshold",
    )
    _need(
        0.0 <= value <= 1.0
        and 0.0 <= threshold <= 1.0
        and 0.0 <= mean_iou <= 1.0
        and 0.0 <= mean_iou_threshold <= 1.0,
        "self_reference_count_mismatch",
        f"{model}:similarity_range",
    )
    policy = payload.get("numerical_similarity_policy")
    _need(
        isinstance(policy, Mapping)
        and policy.get("schema") == "onnx-splitpoint/task-quality-policy"
        and policy.get("schema_version") == 3
        and policy.get("native_self_reference_denominator")
        == "reference_detections"
        and policy.get("native_self_reference_class_aware") is True
        and policy.get("numerical_similarity_required_for_claim") is True,
        "self_reference_policy_mismatch",
        model,
    )
    policy_sha = _digest(
        payload.get("numerical_similarity_policy_sha256"),
        field=f"{model}.policy",
    )
    _need(
        _sha256_json(policy) == policy_sha,
        "self_reference_policy_mismatch",
        f"{model}:policy_sha256",
    )
    policy_match_threshold = _finite_number(
        policy.get("native_self_reference_min_match"),
        field=f"{model}.policy.min_match",
    )
    policy_iou_threshold = _finite_number(
        policy.get("native_self_reference_iou_threshold"),
        field=f"{model}.policy.iou_threshold",
    )
    policy_confidence_threshold = _finite_number(
        policy.get("native_self_reference_confidence_threshold"),
        field=f"{model}.policy.confidence_threshold",
    )
    policy_mean_iou_threshold = _finite_number(
        policy.get("native_self_reference_min_mean_iou"),
        field=f"{model}.policy.min_mean_iou",
    )
    _need(
        _close_number(threshold, policy_match_threshold)
        and _close_number(mean_iou_threshold, policy_mean_iou_threshold)
        and _close_number(
            _finite_number(
                payload.get("numerical_similarity_iou_threshold"),
                field=f"{model}.similarity_iou_threshold",
            ),
            policy_iou_threshold,
        )
        and _close_number(
            _finite_number(
                payload.get("numerical_similarity_confidence_threshold"),
                field=f"{model}.similarity_confidence_threshold",
            ),
            policy_confidence_threshold,
        )
        and payload.get("numerical_similarity_policy_id")
        == policy.get("native_self_reference_policy_id")
        and payload.get("numerical_similarity_metric")
        == "reference_match_ratio_and_mean_matched_iou"
        and payload.get("numerical_similarity_scope")
        == "class_aware_postnms_detection",
        "self_reference_policy_mismatch",
        f"{model}:policy_binding",
    )
    _need(value >= threshold, "self_reference_threshold_mismatch", model)
    _need(mean_iou >= mean_iou_threshold, "self_reference_threshold_mismatch", model)
    matched = _strict_int(
        payload.get("numerical_similarity_matched_count"),
        field=f"{model}.matched",
    )
    reference_count = _strict_int(
        payload.get("numerical_similarity_reference_count"),
        field=f"{model}.reference_count",
        minimum=1,
    )
    _need(
        matched <= reference_count
        and _close_number(value, matched / reference_count),
        "self_reference_count_mismatch",
        model,
    )
    reference_detections = payload.get("reference_detections")
    native_detections = payload.get("native_detections")
    _need(
        isinstance(reference_detections, list)
        and isinstance(native_detections, list)
        and all(isinstance(row, Mapping) for row in reference_detections)
        and all(isinstance(row, Mapping) for row in native_detections)
        and len(reference_detections) == reference_count
        and matched <= len(native_detections),
        "self_reference_count_mismatch",
        f"{model}:detections",
    )
    best = payload.get("best")
    _need(isinstance(best, Mapping), "self_reference_best_missing", model)
    _need(best.get("contract_family_match") is True, "self_reference_contract_mismatch", model)
    best_match = best.get("match")
    _need(
        isinstance(best_match, Mapping)
        and _strict_int(
            best.get("full_count"), field=f"{model}.best.full_count",
        ) == len(reference_detections)
        and _strict_int(
            best.get("native_count"), field=f"{model}.best.native_count",
        ) == len(native_detections)
        and _strict_int(
            best_match.get("ref_count"), field=f"{model}.best.match.ref_count",
        ) == reference_count
        and _strict_int(
            best_match.get("pred_count"), field=f"{model}.best.match.pred_count",
        ) == len(native_detections)
        and _strict_int(
            best_match.get("matched"), field=f"{model}.best.match.matched",
        ) == matched
        and _close_number(
            _finite_number(
                best_match.get("match_ratio"),
                field=f"{model}.best.match.match_ratio",
            ),
            value,
        )
        and _close_number(
            _finite_number(
                best_match.get("mean_iou"),
                field=f"{model}.best.match.mean_iou",
            ),
            mean_iou,
        ),
        "self_reference_count_mismatch",
        f"{model}:best",
    )
    return {
        "diagnosis": payload.get("diagnosis"),
        "scope": str(payload.get("numerical_similarity_scope") or ""),
        "policy_sha256": policy_sha,
        "match_ratio": value,
        "match_threshold": threshold,
        "mean_iou": mean_iou,
        "mean_iou_threshold": mean_iou_threshold,
        "matched": matched,
        "reference_count": reference_count,
        "best_full_mode": str(best.get("full_mode") or ""),
        "best_native_mode": str(best.get("native_mode") or ""),
    }


def _verify_output_contract(
    payload: Mapping[str, Any],
    *,
    model: str,
    hef_sha256: str,
    hef_size_bytes: int,
) -> None:
    _schema(
        payload,
        name="onnx-splitpoint/output-contracts",
        versions={1},
        label=f"{model}.output_contracts",
    )
    _need(
        payload.get("model_id") == model
        and payload.get("task") == "detection",
        "output_contract_identity_mismatch",
        f"{model}:document",
    )
    contracts = payload.get("contracts")
    _need(isinstance(contracts, list), "output_contract_missing", model)
    matches = [
        row for row in contracts
        if isinstance(row, Mapping)
        and row.get("model_id") == model
        and row.get("backend") == "hailo8"
        and row.get("variant") == "full"
    ]
    _need(len(matches) == 1, "output_contract_missing", model)
    contract = matches[0]
    _schema(
        contract,
        name="onnx-splitpoint/output-contract",
        versions={1},
        label=f"{model}.output_contract",
    )
    _need(
        contract.get("model_id") == model
        and contract.get("task") == "detection"
        and contract.get("backend") == "hailo8"
        and contract.get("variant") == "full",
        "output_contract_identity_mismatch",
        f"{model}:contract",
    )
    _need(
        _digest(contract.get("recorded_artifact_sha256"), field=f"{model}.contract.hef")
        == hef_sha256,
        "hef_identity_mismatch",
        f"{model}:output_contract",
    )
    _need(
        _strict_int(
            contract.get("recorded_artifact_size_bytes"),
            field=f"{model}.contract.hef_size_bytes",
            minimum=1,
        )
        == hef_size_bytes,
        "output_contract_artifact_size_mismatch",
        model,
    )


def _recursive_values(payload: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            if str(raw_key) == key:
                values.append(value)
            values.extend(_recursive_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_recursive_values(value, key))
    return values


def _implementation_artifacts_projection(
    attestor: _Attestor,
    *,
    model: str,
    frozen_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    artifacts = frozen_contract.get("implementation_artifacts")
    invariant_identity = frozen_contract.get("invariant_identity")
    _need(
        isinstance(invariant_identity, Mapping),
        "frozen_postprocess_invariant_identity_mismatch",
        model,
    )
    invariant_payload = dict(invariant_identity)
    invariant_sha = _digest(
        frozen_contract.get("invariant_contract_sha256"),
        field=f"{model}.frozen_postprocess_invariant_contract_sha256",
    )
    _need(
        _sha256_json(invariant_payload) == invariant_sha,
        "frozen_postprocess_invariant_sha256_mismatch",
        model,
    )
    _need(
        all(
            key in frozen_contract and frozen_contract.get(key) == value
            for key, value in invariant_payload.items()
        ),
        "frozen_postprocess_invariant_identity_mismatch",
        model,
    )
    _need(
        isinstance(artifacts, Mapping)
        and invariant_payload.get("implementation_artifacts") == artifacts,
        "implementation_artifact_binding_mismatch",
        model,
    )
    allowed = {
        "onnx_splitpoint_tool/native_detection_postprocess.py",
        "onnx_splitpoint_tool/runners/harness/base.py",
        "onnx_splitpoint_tool/runners/harness/yolo.py",
    }
    _need(
        len(artifacts) == len(allowed),
        "implementation_artifact_binding_mismatch",
        model,
    )
    tool_root = _lexical_absolute(
        Path(__file__).parent.parent,
        label="tool_source_root",
    )
    projection: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for name, raw in sorted(artifacts.items(), key=lambda item: str(item[0])):
        _need(
            isinstance(raw, Mapping),
            "implementation_artifact_binding_mismatch",
            f"{model}:{name}",
        )
        relative = _logical_path(
            raw.get("relative_path"),
            field=f"{model}.implementation_artifacts.{name}.relative_path",
        )
        _need(
            relative in allowed and relative not in observed_paths,
            "implementation_artifact_binding_mismatch",
            f"{model}:{relative}",
        )
        observed_paths.add(relative)
        observed = attestor.file(
            tool_root / relative,
            logical=f"tool/{relative}",
        )
        declared = _digest(
            raw.get("sha256"),
            field=f"{model}.implementation_artifacts.{name}.sha256",
        )
        _need(
            observed.sha256 == declared,
            "implementation_artifact_sha256_mismatch",
            f"{model}:{relative}",
        )
        projection.append({
            "name": str(name),
            "relative_path": relative,
            "sha256": observed.sha256,
            "size_bytes": observed.size_bytes,
        })
    _need(
        observed_paths == allowed,
        "implementation_artifact_binding_mismatch",
        model,
    )
    return projection, invariant_sha


def _completed_evidence_projection(
    *,
    model: str,
    semantic: Mapping[str, Any],
    semantic_meta: Mapping[str, Any],
    native_report: Mapping[str, Any],
    completed_payload: Mapping[str, Any],
    completed_observed: FileObservation,
) -> dict[str, Any]:
    completed = semantic.get("completed_v2_evidence")
    best = semantic.get("best")
    _need(
        isinstance(completed, Mapping)
        and isinstance(best, Mapping)
        and completed.get("available") is True
        and completed.get("completed_v2_verified") is True
        and completed.get("exact_completed_result_identity_bound") is True
        and completed.get("completed_v2_exact_result_claim_binding") is True
        and completed.get("portable_result_hash_mismatch") is False,
        "completed_v2_evidence_invalid",
        model,
    )
    _need(
        completed.get("expected_contract_family") == "decoded_nms"
        and completed.get("semantic_result_binding_status")
        == "exact_same_hotloop_completed_artifact"
        and completed.get("completed_v2_semantic_evidence_tier")
        == "exact_same_hotloop_completed_artifact"
        and completed.get("full_mode") == best.get("full_mode")
        and completed.get("native_mode") == best.get("native_mode"),
        "completed_v2_evidence_invalid",
        f"{model}:binding_scope",
    )
    comparison_hash = _digest(
        completed.get("completed_task_comparison_endpoint_contract_hash"),
        field=f"{model}.completed_v2.comparison_endpoint",
    )
    comparison_contract = completed.get(
        "completed_task_comparison_endpoint_contract"
    )
    comparison_body = dict(comparison_contract) if isinstance(
        comparison_contract, Mapping
    ) else {}
    comparison_body.pop("endpoint_contract_complete", None)
    comparison_body.pop("endpoint_contract_hash", None)
    comparison_body.pop("output_endpoint_id", None)
    expected_comparison_output_id = (
        f"detection:decoded_nms:comparison:{comparison_hash}"
    )
    _need(
        isinstance(comparison_contract, Mapping)
        and comparison_contract.get("model_id") == model
        and comparison_contract.get("task") == "detection"
        and comparison_contract.get("contract_family") == "decoded_nms"
        and comparison_contract.get("endpoint_contract_complete") is True
        and _digest(
            comparison_contract.get("endpoint_contract_hash"),
            field=f"{model}.completed_v2.comparison_contract",
        )
        == comparison_hash,
        "completed_v2_evidence_invalid",
        f"{model}:comparison_endpoint",
    )
    _need(
        _sha256_json(comparison_body) == comparison_hash
        and comparison_contract.get("output_endpoint_id")
        == expected_comparison_output_id
        and completed.get("completed_task_comparison_output_endpoint_id")
        == expected_comparison_output_id,
        "completed_v2_evidence_invalid",
        f"{model}:comparison_endpoint_hash",
    )
    reference_detections = semantic.get("reference_detections")
    native_detections = semantic.get("native_detections")
    _need(
        isinstance(reference_detections, list)
        and isinstance(native_detections, list)
        and all(isinstance(row, Mapping) for row in reference_detections)
        and all(isinstance(row, Mapping) for row in native_detections),
        "completed_v2_evidence_invalid",
        f"{model}:semantic_detections",
    )
    full_reference_sha = _sha256_json(reference_detections)
    native_detections_sha = _sha256_json(native_detections)
    _need(
        full_reference_sha
        == _digest(
            completed.get("full_reference_result_sha256"),
            field=f"{model}.completed_v2.full_reference_result",
        ),
        "completed_v2_evidence_invalid",
        f"{model}:full_reference_result",
    )
    sealed = semantic_meta.get("frozen_host_postprocess_result")
    _need(
        isinstance(sealed, Mapping),
        "completed_v2_evidence_invalid",
        f"{model}:frozen_host_result",
    )
    detections_sha = _digest(
        sealed.get("detections_sha256"),
        field=f"{model}.frozen_host_result.detections_sha256",
    )
    _need(
        native_detections_sha
        == detections_sha
        == _digest(
            completed.get("native_completed_result_sha256"),
            field=f"{model}.completed_v2.native_result",
        )
        == _digest(
            completed.get("performance_hotloop_result_sha256"),
            field=f"{model}.completed_v2.hotloop_result",
        ),
        "completed_v2_evidence_invalid",
        f"{model}:native_result",
    )
    _need(
        sealed.get("detections") == native_detections
        and _strict_int(
            sealed.get("detection_count"),
            field=f"{model}.frozen_host_result.detection_count",
        ) == len(native_detections),
        "completed_v2_evidence_invalid",
        f"{model}:native_detections",
    )
    embedded_artifact = sealed.get("completed_result_artifact")
    _need(
        isinstance(embedded_artifact, Mapping)
        and dict(embedded_artifact) == dict(completed_payload),
        "completed_result_artifact_mismatch",
        model,
    )
    _schema(
        completed_payload,
        name="onnx-splitpoint/frozen-completed-detection-result-artifact",
        versions={1},
        label=f"{model}.completed_result_artifact",
    )
    _need(
        completed_payload.get("detections") == native_detections
        and _sha256_json(completed_payload.get("detections"))
        == native_detections_sha,
        "completed_result_artifact_mismatch",
        f"{model}:detections",
    )
    artifact_sha = _digest(
        sealed.get("completed_result_artifact_sha256"),
        field=f"{model}.completed_result_artifact_sha256",
    )
    _need(
        completed_observed.sha256 == artifact_sha,
        "completed_result_artifact_sha256_mismatch",
        model,
    )
    report_artifact_hashes = {
        str(value).strip().lower().removeprefix("sha256:")
        for key in (
            "completed_task_result_artifact_file_sha256",
            "completed_task_result_artifact_sha256",
            "completed_result_artifact_sha256",
        )
        for value in _recursive_values(native_report, key)
        if value not in (None, "")
    }
    _need(
        artifact_sha in report_artifact_hashes,
        "completed_result_artifact_sha256_mismatch",
        f"{model}:native_report",
    )
    report_detection_hashes = {
        str(value).strip().lower().removeprefix("sha256:")
        for value in _recursive_values(native_report, "detections_sha256")
        if value not in (None, "")
    }
    _need(
        detections_sha in report_detection_hashes,
        "completed_v2_evidence_invalid",
        f"{model}:native_report_detections",
    )
    return {
        "comparison_endpoint_contract_sha256": comparison_hash,
        "native_detections_sha256": detections_sha,
        "completed_result_artifact_sha256": artifact_sha,
        "completed_result_artifact_size_bytes": completed_observed.size_bytes,
    }


def _verify_canary_completed_bindings(
    *,
    model: str,
    canary_model: Mapping[str, Any],
    native_report: Mapping[str, Any],
    completed_projection: Mapping[str, Any],
    completed_observed: FileObservation,
) -> None:
    for field in (
        "completed_task_comparison_endpoint_contract_hash",
        "completed_task_result_artifact_sha256",
        "completed_task_result_artifact_file_sha256",
    ):
        _need(
            native_report.get(field) not in (None, ""),
            "required_evidence_missing",
            f"{model}:native_report.{field}",
        )
    native_completed_result = native_report.get(
        "frozen_host_postprocess_result"
    )
    _need(
        isinstance(native_completed_result, Mapping)
        and native_completed_result.get("detections_sha256") not in (None, ""),
        "required_evidence_missing",
        f"{model}:native_report.frozen_host_postprocess_result.detections_sha256",
    )

    report_endpoint_sha = _digest(
        native_report.get("completed_task_comparison_endpoint_contract_hash"),
        field=(
            f"{model}.native_report."
            "completed_task_comparison_endpoint_contract_hash"
        ),
    )
    _need(
        report_endpoint_sha
        == completed_projection["comparison_endpoint_contract_sha256"],
        "canary_completed_endpoint_binding_mismatch",
        model,
    )
    canary_endpoint_value = canary_model.get(
        "completed_task_comparison_endpoint_contract_hash"
    )
    if canary_endpoint_value not in (None, ""):
        _need(
            _digest(
                canary_endpoint_value,
                field=(
                    f"{model}.canary."
                    "completed_task_comparison_endpoint_contract_hash"
                ),
            )
            == report_endpoint_sha,
            "canary_completed_endpoint_binding_mismatch",
            model,
        )

    report_artifact_sha = _digest(
        native_report.get("completed_task_result_artifact_sha256"),
        field=f"{model}.native_report.completed_task_result_artifact_sha256",
    )
    report_artifact_file_sha = _digest(
        native_report.get("completed_task_result_artifact_file_sha256"),
        field=(
            f"{model}.native_report."
            "completed_task_result_artifact_file_sha256"
        ),
    )
    _need(
        report_artifact_sha
        == report_artifact_file_sha
        == completed_projection["completed_result_artifact_sha256"]
        == completed_observed.sha256,
        "canary_completed_artifact_binding_mismatch",
        model,
    )
    canary_artifact_value = canary_model.get(
        "completed_task_result_artifact_sha256"
    )
    if canary_artifact_value not in (None, ""):
        _need(
            _digest(
                canary_artifact_value,
                field=(
                    f"{model}.canary."
                    "completed_task_result_artifact_sha256"
                ),
            )
            == report_artifact_sha,
            "canary_completed_artifact_binding_mismatch",
            model,
        )

    report_detections_sha = _digest(
        native_completed_result.get("detections_sha256"),
        field=(
            f"{model}.native_report."
            "frozen_host_postprocess_result.detections_sha256"
        ),
    )
    _need(
        report_detections_sha
        == completed_projection["native_detections_sha256"],
        "canary_completed_detections_binding_mismatch",
        model,
    )
    canary_detections_value = canary_model.get(
        "completed_detections_sha256"
    )
    if canary_detections_value not in (None, ""):
        _need(
            _digest(
                canary_detections_value,
                field=f"{model}.canary.completed_detections_sha256",
            )
            == report_detections_sha,
            "canary_completed_detections_binding_mismatch",
            model,
        )


def _verification_projection(
    *,
    run_dir: Path,
    canary_dir: Path,
    self_reference_paths: Sequence[Path],
    expected_quality_items: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attestor = _Attestor()
    run_manifest = attestor.json(run_dir / "run_manifest.json", logical="run/run_manifest.json")
    _schema(
        run_manifest,
        name="onnx-splitpoint/evaluation-run-manifest",
        versions={1},
        label="run_manifest",
    )
    run_id = str(run_manifest.get("run_id") or "")
    _need(bool(run_id), "run_id_missing", "run_manifest")
    _same_location(
        run_manifest.get("run_dir"),
        run_dir,
        code="source_run_binding_mismatch",
        label="run_manifest.run_dir",
    )
    models_payload = run_manifest.get("models")
    _need(isinstance(models_payload, Mapping), "run_models_invalid", "run_manifest")

    artifact_index = attestor.json(
        run_dir / "artifact_index.json",
        logical="run/artifact_index.json",
    )
    _schema(
        artifact_index,
        name="onnx-splitpoint/artifact-index",
        versions={1},
        label="artifact_index",
    )
    _need(artifact_index.get("run_id") == run_id, "source_run_binding_mismatch", "artifact_index")

    quality = attestor.json(
        run_dir / "quality_management/central_quality_summary.json",
        logical="run/quality_management/central_quality_summary.json",
    )
    _schema(
        quality,
        name="onnx-splitpoint/central-quality-summary",
        versions={1},
        label="central_quality_summary",
    )
    quality_rows = quality.get("results")
    _need(isinstance(quality_rows, list), "central_quality_results_invalid", "results")
    summary_decisions = {
        str(quality.get(field) or "").strip().lower()
        for field in (
            "scientific_status",
            "aggregate_quality_decision",
            "quality_decision",
        )
    }
    _need(
        len(summary_decisions) == 1
        and summary_decisions <= {"pass", "fail", "inconclusive"},
        "quality_decision_mismatch",
        "central_quality_summary",
    )
    summary_decision = next(iter(summary_decisions))

    bundle_manifest = attestor.json(
        run_dir / "reports/results_bundle_manifest.json",
        logical="run/reports/results_bundle_manifest.json",
    )
    _schema(
        bundle_manifest,
        name="onnx-splitpoint/results-bundle-manifest",
        versions={2},
        label="results_bundle_manifest",
    )
    _need(bundle_manifest.get("run_id") == run_id, "source_run_binding_mismatch", "results_bundle")
    _need(bundle_manifest.get("missing_reports") == [], "required_evidence_missing", "missing_reports")
    _need(bundle_manifest.get("missing_outputs") == {}, "required_evidence_missing", "missing_outputs")

    report_manifest = attestor.json(
        run_dir / "reports/scientific/report_manifest.json",
        logical="run/reports/scientific/report_manifest.json",
    )
    _schema(
        report_manifest,
        name="onnx-splitpoint/scientific-report-manifest",
        versions={2},
        label="scientific_report_manifest",
    )
    report_summary = _report_artifacts(
        attestor,
        run_dir / "reports/scientific",
        report_manifest,
    )

    canary = attestor.json(
        canary_dir / "canary_result.json",
        logical="canary/canary_result.json",
    )
    _schema(
        canary,
        name="onnx-splitpoint/v27713-hailo8-artifact-canary-result",
        versions={1},
        label="canary_result",
    )
    _need(canary.get("status") == "PASS", "canary_not_passed", "status")
    _need(canary.get("result_receipt_written") is True, "canary_not_passed", "receipt")
    _need(
        canary.get("canary_scope")
        == "structural_artifact_and_runtime_only",
        "canary_scope_mismatch",
        "canary_scope",
    )
    _need(
        canary.get("source_snapshot_before_sha256")
        == canary.get("source_snapshot_after_sha256"),
        "canary_source_modified",
        "snapshot",
    )
    _need(canary.get("attested_source_files_modified") is False, "canary_source_modified", "flag")
    _need(canary.get("attested_source_files_changed") == [], "canary_source_modified", "changed")
    _need(canary.get("numerical_correctness_attested") is False, "canary_scope_overclaim", "numerical")
    _need(canary.get("backend_parity_attested") is False, "canary_scope_overclaim", "parity")
    runtime = canary.get("runtime")
    _need(isinstance(runtime, Mapping), "canary_runtime_invalid", "runtime")
    _need(
        runtime.get("status") == "COMPLETED"
        and runtime.get("payload_verified") is True
        and runtime.get("runtime_admitted") is True
        and runtime.get("failures") == [],
        "canary_runtime_invalid",
        "runtime",
    )
    model_results_raw = canary.get("model_results")
    invocations_raw = runtime.get("invocations")
    _need(isinstance(model_results_raw, list), "canary_models_invalid", "model_results")
    _need(isinstance(invocations_raw, list), "canary_runtime_invalid", "invocations")
    canary_models = {
        str(row.get("model_id") or ""): dict(row)
        for row in model_results_raw
        if isinstance(row, Mapping) and str(row.get("model_id") or "")
    }
    canary_invocations = {
        str(row.get("model_id") or ""): dict(row)
        for row in invocations_raw
        if isinstance(row, Mapping) and str(row.get("model_id") or "")
    }
    _need(
        len(canary_models) == len(model_results_raw),
        "duplicate_canary_model",
        "model_results",
    )
    _need(
        len(canary_invocations) == len(invocations_raw),
        "duplicate_canary_model",
        "invocations",
    )

    self_refs: dict[str, tuple[Mapping[str, Any], FileObservation]] = {}
    for index, path in enumerate(self_reference_paths):
        observed = _read_regular_nofollow(
            path,
            label=f"self_reference[{index}]",
            collect=True,
        )
        payload_raw = _strict_json(observed.data or b"", label=f"self_reference[{index}]")
        _need(isinstance(payload_raw, Mapping), "json_object_required", f"self_reference[{index}]")
        payload = dict(payload_raw)
        model = str(payload.get("model") or "")
        _need(bool(model), "self_reference_model_missing", str(index))
        _need(model not in self_refs, "duplicate_self_reference", model)
        attestor.add(observed, logical=f"self_references/{model}.json")
        self_refs[model] = (payload, observed)
    _need(bool(self_refs), "required_self_reference_missing", "all")
    missing_invocations = set(canary_models) - set(canary_invocations)
    extra_invocations = set(canary_invocations) - set(canary_models)
    _need(
        not missing_invocations,
        "required_canary_model_missing",
        ",".join(sorted(missing_invocations)),
    )
    _need(
        not extra_invocations,
        "unexpected_canary_invocation",
        ",".join(sorted(extra_invocations)),
    )
    missing_self_refs = set(canary_models) - set(self_refs)
    extra_self_refs = set(self_refs) - set(canary_models)
    _need(
        not missing_self_refs,
        "required_self_reference_missing",
        ",".join(sorted(missing_self_refs)),
    )
    _need(
        not extra_self_refs,
        "unexpected_self_reference",
        ",".join(sorted(extra_self_refs)),
    )
    _need(
        all(isinstance(row, Mapping) for row in quality_rows),
        "central_quality_results_invalid",
        "row_type",
    )
    expected_quality_keys = {
        (str(model), backend)
        for model in models_payload
        for backend in _QUALITY_BACKENDS
    }
    observed_quality_keys = [
        (
            str(row.get("model_id") or ""),
            str(row.get("backend") or ""),
        )
        for row in quality_rows
    ]
    _need(
        len(observed_quality_keys) == len(set(observed_quality_keys)),
        "duplicate_quality_row",
        "central_quality_summary",
    )
    missing_quality_keys = expected_quality_keys - set(
        observed_quality_keys
    )
    extra_quality_keys = set(observed_quality_keys) - expected_quality_keys
    _need(
        not missing_quality_keys,
        "required_quality_row_missing",
        ",".join(
            f"{model}:{backend}"
            for model, backend in sorted(missing_quality_keys)
        ),
    )
    _need(
        not extra_quality_keys,
        "unexpected_quality_row",
        ",".join(
            f"{model}:{backend}"
            for model, backend in sorted(extra_quality_keys)
        ),
    )
    quality_rows_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    global_quality_projections: dict[
        tuple[str, str], dict[str, Any]
    ] = {}
    global_quality_decisions: dict[tuple[str, str], str] = {}
    for raw_row in quality_rows:
        row = dict(raw_row)
        key = (
            str(row.get("model_id") or ""),
            str(row.get("backend") or ""),
        )
        _schema(
            row,
            name="onnx-splitpoint/management-paired-quality-result",
            versions={1},
            label=f"quality:{key[0]}:{key[1]}",
        )
        _need(
            row.get("status") == "completed"
            and row.get("technical_status") == "completed",
            "required_quality_row_missing",
            f"{key[0]}:{key[1]}:not_completed",
        )
        _need(
            str(row.get("eval_run_id") or "") == run_id,
            "source_run_binding_mismatch",
            f"quality:{key[0]}:{key[1]}",
        )
        _need(
            row.get("n") == expected_quality_items,
            "quality_metric_inconsistent",
            f"quality:{key[0]}:{key[1]}:n",
        )
        run_model = models_payload.get(key[0])
        _need(
            isinstance(run_model, Mapping),
            "run_model_invalid",
            key[0],
        )
        global_model_sha = _digest(
            run_model.get("model_sha256"),
            field=f"{key[0]}.model_sha256",
        )
        for field in ("model_sha256", "source_model_sha256"):
            _need(
                _digest(
                    row.get(field),
                    field=f"quality.{key[0]}.{key[1]}.{field}",
                )
                == global_model_sha,
                "model_identity_mismatch",
                f"quality:{key[0]}:{key[1]}:{field}",
            )
        if key[1] == "tensorrt":
            _need(
                _digest(
                    row.get("source_onnx_sha256"),
                    field=(
                        f"quality.{key[0]}.tensorrt."
                        "source_onnx_sha256"
                    ),
                )
                == global_model_sha,
                "model_identity_mismatch",
                f"quality:{key[0]}:tensorrt:source_onnx_sha256",
            )
        declared_decisions = {
            str(row.get(field) or "").strip().lower()
            for field in ("decision", "scientific_status")
        }
        _need(
            len(declared_decisions) == 1
            and declared_decisions <= {"pass", "fail", "inconclusive"},
            "quality_decision_mismatch",
            f"quality:{key[0]}:{key[1]}:declared_decision",
        )
        declared_decision = next(iter(declared_decisions))
        if "task_quality_decision" in row:
            _need(
                str(row.get("task_quality_decision") or "").strip().lower()
                == declared_decision,
                "quality_decision_mismatch",
                f"quality:{key[0]}:{key[1]}:task_quality_decision",
            )
        quality_rows_by_key[key] = row
        global_quality_decisions[key] = declared_decision
        # Only the models with independent Self-Reference evidence are inside
        # the deep numerical/physical claim.  Other official run rows remain
        # structurally bound into the recorded global summary without
        # retroactively promoting their task-specific metric evidence.
        if key[0] in self_refs:
            global_quality_projections[key] = _quality_projection(row)
    benchmark_set_relatives = _verify_canary_benchmark_sets(
        canary,
        run_dir=run_dir,
        models=canary_models,
    )

    model_projections: list[dict[str, Any]] = []
    for model in sorted(self_refs):
        _need(model in models_payload, "required_model_missing", model)
        _need(model in canary_models, "required_canary_model_missing", model)
        _need(model in canary_invocations, "required_canary_model_missing", f"{model}:invocation")
        run_model = models_payload[model]
        _need(isinstance(run_model, Mapping), "run_model_invalid", model)
        model_sha = _digest(run_model.get("model_sha256"), field=f"{model}.model_sha256")

        suite_relative = f"models/{model}/benchmark_set/legacy_suite"
        suite = run_dir / suite_relative
        onnx_relative = f"{suite_relative}/models/{model}.onnx"
        onnx_path = run_dir / onnx_relative
        onnx_observed = attestor.file(onnx_path, logical=f"run/{onnx_relative}")
        _need(onnx_observed.sha256 == model_sha, "model_identity_mismatch", model)

        hef_relative = f"{suite_relative}/hailo/hailo8/full/compiled.hef"
        hef_observed = attestor.file(run_dir / hef_relative, logical=f"run/{hef_relative}")
        receipt_relative = (
            f"{suite_relative}/hailo/hailo8/full/hailo_hef_build_receipt.json"
        )
        hef_receipt = attestor.json(
            run_dir / receipt_relative,
            logical=f"run/{receipt_relative}",
        )
        _need(
            hef_receipt.get("schema") == "onnx-splitpoint/hailo-hef-build-receipt/v2",
            "hef_receipt_schema_mismatch",
            model,
        )
        _need(
            _digest(hef_receipt.get("hef_sha256"), field=f"{model}.receipt.hef")
            == hef_observed.sha256,
            "hef_identity_mismatch",
            f"{model}:receipt",
        )
        for field in ("source_onnx_sha256", "compiler_onnx_sha256"):
            _need(
                _digest(hef_receipt.get(field), field=f"{model}.receipt.{field}")
                == model_sha,
                "model_identity_mismatch",
                f"{model}:{field}",
            )
        receipt_preprocessing_sha = _digest(
            hef_receipt.get("preprocessing_contract_sha256"),
            field=f"{model}.receipt.preprocessing_contract_sha256",
        )
        receipt_preprocessing_contract = hef_receipt.get(
            "preprocessing_contract"
        )
        _need(
            isinstance(receipt_preprocessing_contract, Mapping),
            "required_evidence_missing",
            f"{model}:hef_receipt.preprocessing_contract",
        )
        _need(
            _sha256_json(dict(receipt_preprocessing_contract))
            == receipt_preprocessing_sha,
            "preprocessing_contract_sha256_mismatch",
            f"{model}:hef_receipt",
        )
        output_contract_relative = f"{suite_relative}/output_contracts.json"
        output_contracts = attestor.json(
            run_dir / output_contract_relative,
            logical=f"run/{output_contract_relative}",
        )
        _verify_output_contract(
            output_contracts,
            model=model,
            hef_sha256=hef_observed.sha256,
            hef_size_bytes=hef_observed.size_bytes,
        )

        quality_by_backend: dict[str, Mapping[str, Any]] = {}
        quality_payloads: dict[str, dict[str, Any]] = {}
        quality_projections: list[dict[str, Any]] = []
        for backend in _QUALITY_BACKENDS:
            row = quality_rows_by_key[(model, backend)]
            quality_by_backend[backend] = row
            for field in ("model_sha256", "source_model_sha256"):
                _need(
                    _digest(row.get(field), field=f"{model}.{backend}.{field}")
                    == model_sha,
                    "model_identity_mismatch",
                    f"{model}:{backend}:{field}",
                )
            if backend == "tensorrt":
                _need(
                    _digest(row.get("source_onnx_sha256"), field=f"{model}.trt.onnx")
                    == model_sha,
                    "model_identity_mismatch",
                    f"{model}:tensorrt:source_onnx",
                )
            if backend == "hailo8":
                _need(
                    str(row.get("runtime_precision_identity") or "")
                    == f"hailo_hef_sha256:{hef_observed.sha256}",
                    "quality_identity_mismatch",
                    f"{model}:hailo8:runtime_precision_identity",
                )
            physical_payloads = _quality_payload_projection(
                attestor,
                run_dir=run_dir,
                run_id=run_id,
                model=model,
                model_sha256=model_sha,
                backend=backend,
                row=row,
                expected_items=expected_quality_items,
            )
            quality_payloads[backend] = physical_payloads
            projected_quality = dict(
                global_quality_projections[(model, backend)]
            )
            projected_quality["physical_payloads"] = physical_payloads
            quality_projections.append(projected_quality)
        for field in (
            "preprocessing_contract_sha256",
            "reference_predictions_sha256",
            "annotations_sha256",
            "validation_dataset_sha256",
            "validation_ground_truth_sha256",
            "validation_image_ids_sha256",
        ):
            _need(
                _digest(quality_by_backend["hailo8"].get(field), field=f"hailo8.{field}")
                == _digest(quality_by_backend["tensorrt"].get(field), field=f"tensorrt.{field}"),
                "quality_identity_mismatch",
                f"{model}:{field}",
            )
        _need(
            _digest(
                quality_by_backend["hailo8"].get(
                    "preprocessing_contract_sha256"
                ),
                field=f"{model}.hailo8.preprocessing_contract_sha256",
            )
            == receipt_preprocessing_sha,
            "quality_identity_mismatch",
            f"{model}:receipt_preprocessing",
        )

        canary_model = canary_models[model]
        _need(
            canary_model.get("status") == "PASS"
            and canary_model.get("task") == "detection"
            and canary_model.get("structural_output_dump_attested") is True
            and canary_model.get("structural_postprocess_completed") is True
            and canary_model.get("structural_preprocessing_binding_attested") is True
            and canary_model.get("numerical_correctness_attested") is False
            and canary_model.get("backend_parity_attested") is False,
            "canary_model_invalid",
            model,
        )
        canary_hef_sha = _digest(canary_model.get("hef_sha256"), field=f"{model}.canary.hef")
        _need(canary_hef_sha == hef_observed.sha256, "hef_identity_mismatch", f"{model}:canary")
        invocation = canary_invocations[model]
        _need(invocation.get("returncode") == 0, "canary_runtime_invalid", f"{model}:returncode")
        _need(
            _digest(invocation.get("expected_hef_sha256"), field=f"{model}.invocation.hef")
            == hef_observed.sha256,
            "hef_identity_mismatch",
            f"{model}:invocation",
        )
        result_relative = _logical_path(
            invocation.get("result_relative"),
            field=f"{model}.result_relative",
        )

        semantic, _semantic_observed = self_refs[model]
        semantic_projection = _self_reference_projection(semantic, model=model)
        _same_location(
            semantic.get("benchmark_set"),
            suite,
            code="source_run_binding_mismatch",
            label=f"{model}.benchmark_set",
        )
        _same_location(
            semantic.get("full_onnx"),
            onnx_path,
            code="source_run_binding_mismatch",
            label=f"{model}.full_onnx",
        )
        _need(
            _digest(semantic.get("full_onnx_sha256"), field=f"{model}.semantic.onnx")
            == model_sha,
            "model_identity_mismatch",
            f"{model}:semantic",
        )
        meta = semantic.get("native_output_manifest_meta")
        _need(isinstance(meta, Mapping), "self_reference_meta_missing", model)
        resolution = meta.get("authoritative_output_contract_resolution")
        _need(isinstance(resolution, Mapping), "self_reference_meta_missing", model)
        _need(
            _digest(resolution.get("artifact_sha256"), field=f"{model}.semantic.hef")
            == hef_observed.sha256,
            "hef_identity_mismatch",
            f"{model}:semantic",
        )
        _need(
            _digest(resolution.get("source_onnx_sha256"), field=f"{model}.semantic.source_onnx")
            == model_sha,
            "model_identity_mismatch",
            f"{model}:semantic_source",
        )
        _need(
            _digest(
                resolution.get("preprocessing_contract_sha256"),
                field=f"{model}.semantic.preprocessing_contract_sha256",
            )
            == receipt_preprocessing_sha,
            "quality_identity_mismatch",
            f"{model}:semantic_preprocessing",
        )

        result_root = canary_dir / result_relative
        dump_root = result_root / "dump"
        boundary_path = dump_root / "native_full_input_manifest.json"
        output_manifest_path = dump_root / "native_full_outputs_manifest.json"
        native_report_path = result_root / "report.json"
        _same_location(
            semantic.get("boundary_manifest"),
            boundary_path,
            code="canary_source_binding_mismatch",
            label=f"{model}.boundary_manifest",
        )
        _same_location(
            semantic.get("native_output_manifest"),
            output_manifest_path,
            code="canary_source_binding_mismatch",
            label=f"{model}.native_output_manifest",
        )
        _same_location(
            semantic.get("native_report"),
            native_report_path,
            code="canary_source_binding_mismatch",
            label=f"{model}.native_report",
        )
        boundary = attestor.json(
            boundary_path,
            logical=f"canary/{model}/native_full_input_manifest.json",
        )
        output_manifest = attestor.json(
            output_manifest_path,
            logical=f"canary/{model}/native_full_outputs_manifest.json",
        )
        native_report = attestor.json(
            native_report_path,
            logical=f"canary/{model}/report.json",
        )
        _need(
            dict(meta) == dict(output_manifest),
            "self_reference_meta_mismatch",
            model,
        )
        native_report_observed = attestor.observation(
            native_report_path,
            logical=f"canary/{model}/report.json",
        )
        boundary_observed = attestor.observation(
            boundary_path,
            logical=f"canary/{model}/native_full_input_manifest.json",
        )
        _need(
            _digest(semantic.get("boundary_manifest_sha256"), field=f"{model}.boundary.sha")
            == boundary_observed.sha256,
            "canary_source_sha256_mismatch",
            f"{model}:boundary",
        )
        output_manifest_observed = attestor.observation(
            output_manifest_path,
            logical=f"canary/{model}/native_full_outputs_manifest.json",
        )
        _need(
            _digest(semantic.get("native_output_manifest_sha256"), field=f"{model}.outputs.sha")
            == output_manifest_observed.sha256,
            "canary_source_sha256_mismatch",
            f"{model}:outputs",
        )
        _need(
            _digest(semantic.get("native_report_sha256"), field=f"{model}.report.sha")
            == native_report_observed.sha256,
            "canary_source_sha256_mismatch",
            f"{model}:report",
        )
        _schema(
            boundary,
            name="onnx-splitpoint/native-full-input-dump",
            versions={1, 2},
            label=f"{model}.boundary",
        )
        _need(boundary.get("model") == model, "canary_model_invalid", f"{model}:boundary")
        canary_image_sha = _digest(
            canary_model.get("image_sha256"),
            field=f"{model}.canary.image",
        )
        _need(
            canary_image_sha
            == _digest(
                invocation.get("image_sha256"),
                field=f"{model}.invocation.image",
            )
            == _digest(
                boundary.get("input_image_sha256"),
                field=f"{model}.boundary.input_image",
            )
            == _digest(
                output_manifest.get("input_image_sha256"),
                field=f"{model}.outputs.input_image",
            )
            == _digest(
                meta.get("input_image_sha256"),
                field=f"{model}.semantic.input_image",
            ),
            "canary_image_binding_mismatch",
            model,
        )
        provenance = meta.get("provenance")
        _need(
            isinstance(provenance, Mapping)
            and _digest(
                provenance.get("image_sha256"),
                field=f"{model}.semantic.provenance.image_sha256",
            )
            == canary_image_sha,
            "canary_image_binding_mismatch",
            f"{model}:provenance",
        )
        report_image_hashes = {
            str(value).strip().lower().removeprefix("sha256:")
            for value in _recursive_values(native_report, "input_image_sha256")
            if value not in (None, "")
        }
        _need(
            canary_image_sha in report_image_hashes,
            "canary_image_binding_mismatch",
            f"{model}:native_report",
        )
        _need(
            _digest(
                boundary.get("preprocessing_contract_sha256"),
                field=f"{model}.boundary.preprocessing",
            )
            == receipt_preprocessing_sha,
            "quality_identity_mismatch",
            f"{model}:canary_preprocessing",
        )
        output_resolution = output_manifest.get(
            "authoritative_output_contract_resolution"
        )
        _need(
            isinstance(output_resolution, Mapping)
            and output_resolution == resolution
            and _digest(
                output_resolution.get("preprocessing_contract_sha256"),
                field=f"{model}.outputs.resolution.preprocessing",
            )
            == receipt_preprocessing_sha,
            "quality_identity_mismatch",
            f"{model}:output_resolution_preprocessing",
        )
        report_preprocessing_hashes = {
            str(value).strip().lower().removeprefix("sha256:")
            for value in _recursive_values(
                native_report,
                "preprocessing_contract_sha256",
            )
            if value not in (None, "")
        }
        _need(
            receipt_preprocessing_sha in report_preprocessing_hashes,
            "quality_identity_mismatch",
            f"{model}:native_report_preprocessing",
        )
        frozen_contract = meta.get("frozen_host_postprocess_contract")
        _need(
            isinstance(frozen_contract, Mapping)
            and output_manifest.get("frozen_host_postprocess_contract")
            == frozen_contract,
            "implementation_artifact_binding_mismatch",
            f"{model}:frozen_host_contract",
        )
        implementation_projection, invariant_contract_sha = (
            _implementation_artifacts_projection(
                attestor,
                model=model,
                frozen_contract=frozen_contract,
            )
        )
        _need(
            invariant_contract_sha
            == quality_payloads["hailo8"][
                "frozen_postprocess_invariant_contract_sha256"
            ],
            "quality_identity_mismatch",
            f"{model}:frozen_postprocess_invariant",
        )
        input_dump_provenance = str(boundary.get("input_dump") or "")
        input_dump_parts = PurePosixPath(input_dump_provenance)
        _need(
            bool(input_dump_provenance)
            and "\\" not in input_dump_provenance
            and all(ord(character) >= 32 for character in input_dump_provenance)
            and ".." not in input_dump_parts.parts
            and input_dump_parts.name == "input_rgb_uint8.bin"
            and (
                input_dump_parts.is_absolute()
                or input_dump_provenance == "input_rgb_uint8.bin"
            ),
            "unsafe_manifest_path",
            f"{model}.input_dump",
        )
        input_dump_name = "input_rgb_uint8.bin"
        input_dump = attestor.file(
            dump_root / input_dump_name,
            logical=f"canary/{model}/{input_dump_name}",
        )
        _need(
            input_dump.sha256
            == _digest(boundary.get("input_dump_sha256"), field=f"{model}.input_dump.sha"),
            "input_dump_sha256_mismatch",
            model,
        )
        _need(
            input_dump.size_bytes
            == _strict_int(boundary.get("input_dump_bytes"), field=f"{model}.input_dump.bytes"),
            "input_dump_size_mismatch",
            model,
        )
        _verify_output_manifest_identity(
            output_manifest,
            model=model,
            setup_id=str(canary.get("setup_id") or ""),
        )
        output_count = _manifest_dump_members(
            attestor,
            dump_root,
            output_manifest,
            model=model,
            frozen_contract=frozen_contract,
        )
        completed_path = result_root / "report.completed_task_result_artifact.json"
        completed_payload = attestor.json(
            completed_path,
            logical=f"canary/{model}/report.completed_task_result_artifact.json",
        )
        completed_observed = attestor.observation(
            completed_path,
            logical=f"canary/{model}/report.completed_task_result_artifact.json",
        )
        _schema(
            completed_payload,
            name="onnx-splitpoint/frozen-completed-detection-result-artifact",
            versions={1},
            label=f"{model}.completed_result_artifact",
        )
        completed_projection = _completed_evidence_projection(
            model=model,
            semantic=semantic,
            semantic_meta=meta,
            native_report=native_report,
            completed_payload=completed_payload,
            completed_observed=completed_observed,
        )
        _verify_canary_completed_bindings(
            model=model,
            canary_model=canary_model,
            native_report=native_report,
            completed_projection=completed_projection,
            completed_observed=completed_observed,
        )
        _need(
            completed_projection["comparison_endpoint_contract_sha256"]
            == quality_payloads["hailo8"][
                "completed_task_endpoint_contract_hash"
            ],
            "quality_identity_mismatch",
            f"{model}:completed_comparison_endpoint",
        )

        model_projections.append({
            "model_id": model,
            "model_sha256": model_sha,
            "onnx_size_bytes": onnx_observed.size_bytes,
            "hef_sha256": hef_observed.sha256,
            "hef_size_bytes": hef_observed.size_bytes,
            "canary_image_sha256": canary_image_sha,
            "canary_output_tensor_count": output_count,
            "completed_evidence": completed_projection,
            "frozen_postprocess_invariant_contract_sha256": (
                invariant_contract_sha
            ),
            "implementation_artifacts": implementation_projection,
            "quality_evidence": sorted(
                quality_projections,
                key=lambda row: (row["backend"], row["case_id"]),
            ),
            "self_reference": semantic_projection,
        })

    decision_severity = {"pass": 0, "inconclusive": 1, "fail": 2}
    selected_quality_decisions = [
        str(evidence["decision"])
        for model_projection in model_projections
        for evidence in model_projection["quality_evidence"]
    ]
    _need(
        bool(selected_quality_decisions),
        "required_evidence_missing",
        "validated_quality_decisions",
    )
    validated_decision_floor = max(
        selected_quality_decisions,
        key=decision_severity.__getitem__,
    )
    all_recorded_quality_decisions = [
        global_quality_decisions[key]
        for key in sorted(global_quality_decisions)
    ]
    expected_decision_counts = {
        decision: all_recorded_quality_decisions.count(decision)
        for decision in sorted(set(all_recorded_quality_decisions))
    }
    observed_decision_counts = quality.get("decision_counts")
    _need(
        isinstance(observed_decision_counts, Mapping)
        and dict(observed_decision_counts) == expected_decision_counts,
        "central_quality_summary_decision_mismatch",
        "decision_counts",
    )
    for field in ("request_count", "quality_result_count", "completed_count"):
        _need(
            _strict_int(
                quality.get(field),
                field=f"central_quality_summary.{field}",
                minimum=1,
            )
            == len(global_quality_decisions),
            "central_quality_summary_decision_mismatch",
            field,
        )
    expected_summary_decision = (
        "fail"
        if expected_decision_counts.get("fail")
        else "inconclusive"
        if expected_decision_counts.get("inconclusive")
        else "pass"
    )
    _need(
        summary_decision == expected_summary_decision,
        "central_quality_summary_decision_mismatch",
        (
            f"summary={summary_decision}:"
            f"recalculated={expected_summary_decision}"
        ),
    )
    _need(
        type(quality.get("scientific_pass")) is bool
        and quality.get("scientific_pass")
        is (expected_summary_decision == "pass"),
        "central_quality_summary_decision_mismatch",
        "scientific_pass",
    )
    # The central summary may be stricter because it also covers rows outside
    # the scientifically selected components, but it may never be more
    # permissive than any row whose components we independently recalculated.
    _need(
        decision_severity[summary_decision]
        >= decision_severity[validated_decision_floor],
        "central_quality_summary_decision_mismatch",
        (
            f"summary={summary_decision}:"
            f"validated_floor={validated_decision_floor}"
        ),
    )

    artifacts = attestor.stable_snapshot()
    artifact_index_binding = _artifact_index_projection(
        artifact_index,
        attested_artifacts=artifacts,
        models=self_refs,
    )
    projection = {
        "claim_scope": "retrospective_evaluated_matrix",
        "claim_scope_unchanged": True,
        "prospective_freeze": False,
        "entire_run_pass_attested": False,
        "source_run": {
            "run_id": run_id,
            "profile_id": str(run_manifest.get("profile_id") or ""),
            "recorded_status": str(run_manifest.get("status") or ""),
            "recorded_technical_status": str(run_manifest.get("technical_status") or ""),
            "recorded_scientific_status": str(run_manifest.get("scientific_status") or ""),
            "quality_status": str(quality.get("status") or ""),
            "quality_scientific_status": str(quality.get("scientific_status") or ""),
            "quality_decision": str(quality.get("quality_decision") or ""),
            "validated_quality_decision_floor": validated_decision_floor,
            "validated_quality_decision_counts": expected_decision_counts,
        },
        "canary": {
            "setup_id": str(canary.get("setup_id") or ""),
            "scope": str(canary.get("canary_scope") or ""),
            "status": "PASS",
            "benchmark_sets": benchmark_set_relatives,
            "numerical_correctness_attested": False,
            "backend_parity_attested": False,
        },
        "quality_item_count": expected_quality_items,
        "artifact_index_binding": artifact_index_binding,
        "models": model_projections,
        "scientific_report_artifacts": report_summary,
        "artifacts": artifacts,
    }
    return projection, artifacts


def _status_for(code: str) -> str:
    return INCOMPLETE if code in _MISSING_CODES else CONFLICT


def _source_locations(
    run_dir: Path,
    canary_dir: Path,
    self_references: Sequence[Path],
    expected_quality_items: int,
) -> dict[str, Any]:
    return {
        "run_dir": os.fspath(run_dir),
        "canary_dir": os.fspath(canary_dir),
        "self_references": [os.fspath(path) for path in self_references],
        "expected_quality_items": expected_quality_items,
    }


def _receipt_payload(
    *,
    status: str,
    source_locations: Mapping[str, Any],
    projection: Mapping[str, Any] | None,
    reason_codes: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "status": status,
        "ok": status == VERIFIED,
        "claim_scope": "retrospective_evaluated_matrix",
        "claim_scope_unchanged": True,
        "prospective_freeze": False,
        "entire_run_pass_attested": False,
        "hardware_rerun": False,
        "compiler_rerun": False,
        "quality_rerun": False,
        "source_locations": dict(source_locations),
        "evidence_projection": dict(projection or {}),
        "evidence_projection_sha256": _sha256_json(projection or {}),
        "reason_codes": sorted(dict.fromkeys(str(code) for code in reason_codes)),
    }
    result["result_payload_sha256"] = _sha256_json(result)
    return result


def _verify_payload_selfhash(payload: Mapping[str, Any]) -> None:
    expected = _digest(payload.get("result_payload_sha256"), field="result_payload_sha256")
    body = dict(payload)
    body.pop("result_payload_sha256", None)
    _need(expected == _sha256_json(body), "result_payload_sha256_mismatch", "receipt")


def _exclusive_write(path: Path, payload: Mapping[str, Any]) -> None:
    output = _lexical_absolute(path, label="output")
    parent_fd = _open_directory_nofollow(output.parent, label="output.parent")
    write_fd = -1
    read_fd = -1
    created_identity: tuple[int, int] | None = None
    data = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            write_fd = os.open(output.name, flags, 0o600, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise ExistingEvidenceError("output_already_exists", os.fspath(output)) from exc
        created = os.fstat(write_fd)
        created_identity = (created.st_dev, created.st_ino)
        offset = 0
        while offset < len(data):
            written_count = os.write(write_fd, data[offset:])
            _need(
                written_count > 0,
                "output_short_write",
                os.fspath(output),
            )
            offset += written_count
        os.fsync(write_fd)
        written = os.fstat(write_fd)
        path_stat = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        _need(stat.S_ISREG(path_stat.st_mode), "output_not_regular", os.fspath(output))
        _need(
            (written.st_dev, written.st_ino) == (path_stat.st_dev, path_stat.st_ino),
            "output_replaced_during_write",
            os.fspath(output),
        )
        read_fd = os.open(
            output.name,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        read_stat = os.fstat(read_fd)
        _need(
            (written.st_dev, written.st_ino) == (read_stat.st_dev, read_stat.st_ino),
            "output_replaced_before_readback",
            os.fspath(output),
        )
        chunks: list[bytes] = []
        while True:
            block = os.read(read_fd, _READ_CHUNK)
            if not block:
                break
            chunks.append(block)
        _need(b"".join(chunks) == data, "output_readback_mismatch", os.fspath(output))
        os.fsync(parent_fd)
    except BaseException:
        if created_identity is not None:
            try:
                current = os.stat(
                    output.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(output.name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
        os.close(parent_fd)


def create_receipt(
    *,
    run_dir: str | Path,
    canary_dir: str | Path,
    self_references: Sequence[str | Path],
    output: str | Path,
    expected_quality_items: int = 500,
) -> dict[str, Any]:
    """Create one exclusive retrospective verification receipt.

    Input roots are never written.  Evidence conflicts are represented in the
    receipt; unsafe or non-exclusive output creation still raises.
    """
    run_root = _lexical_absolute(run_dir, label="run_dir")
    canary_root = _lexical_absolute(canary_dir, label="canary_dir")
    refs = sorted(
        (_lexical_absolute(path, label="self_reference") for path in self_references),
        key=os.fspath,
    )
    expected_items = _strict_int(
        expected_quality_items,
        field="expected_quality_items",
        minimum=1,
    )
    out = _lexical_absolute(output, label="output")
    tool_root = _lexical_absolute(
        Path(__file__).parent.parent,
        label="tool_source_root",
    )
    _need(
        not _path_inside(out, run_root)
        and not _path_inside(out, canary_root)
        and not _path_inside(out, tool_root),
        "output_inside_source",
        os.fspath(out),
    )
    _need(out not in refs, "output_is_source", os.fspath(out))
    locations = _source_locations(run_root, canary_root, refs, expected_items)
    try:
        projection, _artifacts = _verification_projection(
            run_dir=run_root,
            canary_dir=canary_root,
            self_reference_paths=refs,
            expected_quality_items=expected_items,
        )
        result = _receipt_payload(
            status=VERIFIED,
            source_locations=locations,
            projection=projection,
            reason_codes=[],
        )
    except ExistingEvidenceError as exc:
        result = _receipt_payload(
            status=_status_for(exc.code),
            source_locations=locations,
            projection=None,
            reason_codes=[exc.code],
        )
        result["reason_details"] = [str(exc)]
        result["result_payload_sha256"] = _sha256_json({
            key: value for key, value in result.items()
            if key != "result_payload_sha256"
        })
    _exclusive_write(out, result)
    return result


def verify_receipt(
    receipt: str | Path,
    *,
    run_dir: str | Path | None = None,
    canary_dir: str | Path | None = None,
    self_references: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Verify a receipt's self-hash and re-evaluate its source evidence."""
    receipt_path = _lexical_absolute(receipt, label="receipt")
    observed = _read_regular_nofollow(receipt_path, label="receipt", collect=True)
    raw = _strict_json(observed.data or b"", label="receipt")
    _need(isinstance(raw, Mapping), "json_object_required", "receipt")
    payload = dict(raw)
    _schema(payload, name=SCHEMA, versions={SCHEMA_VERSION}, label="receipt")
    _verify_payload_selfhash(payload)
    _need(
        payload.get("tool_version") == TOOL_VERSION
        and payload.get("claim_scope") == "retrospective_evaluated_matrix"
        and payload.get("claim_scope_unchanged") is True
        and payload.get("prospective_freeze") is False
        and payload.get("entire_run_pass_attested") is False
        and payload.get("hardware_rerun") is False
        and payload.get("compiler_rerun") is False
        and payload.get("quality_rerun") is False,
        "receipt_claim_scope_mismatch",
        "receipt",
    )
    embedded_projection = payload.get("evidence_projection")
    _need(
        isinstance(embedded_projection, Mapping),
        "receipt_projection_invalid",
        "evidence_projection",
    )
    expected_projection_sha = _digest(
        payload.get("evidence_projection_sha256"),
        field="evidence_projection_sha256",
    )
    _need(
        _sha256_json(embedded_projection) == expected_projection_sha,
        "evidence_projection_sha256_mismatch",
        "embedded_projection",
    )
    locations = payload.get("source_locations")
    _need(isinstance(locations, Mapping), "receipt_locations_invalid", "source_locations")
    selected_run = run_dir if run_dir is not None else locations.get("run_dir")
    selected_canary = canary_dir if canary_dir is not None else locations.get("canary_dir")
    selected_refs = (
        list(self_references)
        if self_references is not None
        else list(locations.get("self_references") or [])
    )
    expected_items = _strict_int(
        locations.get("expected_quality_items"),
        field="expected_quality_items",
        minimum=1,
    )
    try:
        projection, _artifacts = _verification_projection(
            run_dir=_lexical_absolute(selected_run, label="run_dir"),
            canary_dir=_lexical_absolute(selected_canary, label="canary_dir"),
            self_reference_paths=[
                _lexical_absolute(path, label="self_reference")
                for path in selected_refs
            ],
            expected_quality_items=expected_items,
        )
        _need(
            expected_projection_sha == _sha256_json(projection),
            "evidence_projection_sha256_mismatch",
            "receipt",
        )
        _need(payload.get("status") == VERIFIED and payload.get("ok") is True, "receipt_not_verified", "status")
        status = VERIFIED
        reasons: list[str] = []
    except ExistingEvidenceError as exc:
        status = _status_for(exc.code)
        reasons = [exc.code]
    return {
        "schema": VERIFY_SCHEMA,
        "schema_version": 1,
        "status": status,
        "ok": status == VERIFIED,
        "receipt_sha256": observed.sha256,
        "reason_codes": reasons,
        "claim_scope_unchanged": True,
        "prospective_freeze": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only verification of existing evaluation, canary and "
            "Full-ONNX self-reference evidence."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Create an exclusive verification receipt.")
    create.add_argument("--run-dir", required=True)
    create.add_argument("--canary-dir", required=True)
    create.add_argument("--self-reference", action="append", required=True)
    create.add_argument("--expected-quality-items", type=int, default=500)
    create.add_argument("--out", required=True)
    verify = sub.add_parser("verify", help="Recheck a previously created receipt.")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--run-dir")
    verify.add_argument("--canary-dir")
    verify.add_argument("--self-reference", action="append")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            result = create_receipt(
                run_dir=args.run_dir,
                canary_dir=args.canary_dir,
                self_references=args.self_reference,
                output=args.out,
                expected_quality_items=args.expected_quality_items,
            )
        else:
            result = verify_receipt(
                args.receipt,
                run_dir=args.run_dir,
                canary_dir=args.canary_dir,
                self_references=args.self_reference,
            )
    except ExistingEvidenceError as exc:
        result = {
            "schema": VERIFY_SCHEMA,
            "schema_version": 1,
            "status": _status_for(exc.code),
            "ok": False,
            "reason_codes": [exc.code],
            "claim_scope_unchanged": True,
            "prospective_freeze": False,
        }
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == VERIFIED else 2


__all__ = [
    "CONFLICT",
    "INCOMPLETE",
    "SCHEMA",
    "SCHEMA_VERSION",
    "VERIFIED",
    "ExistingEvidenceError",
    "create_receipt",
    "main",
    "verify_receipt",
]


if __name__ == "__main__":
    raise SystemExit(main())
