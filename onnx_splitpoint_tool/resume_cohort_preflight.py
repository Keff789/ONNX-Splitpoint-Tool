from __future__ import annotations

"""Selection-wide Native Energy resume preflight.

This local coordinator validates every selected plan row before invoking the
existing collector preflight helper.  It never starts an energy collector,
workload, or measurement wrapper.  A caller may proceed to measurement only
when ``measurement_wrapper_allowed`` is exactly ``True`` in the returned
report.

The module remains separate from the release-bound Native runners and remote
preflight implementations.  It consumes their plan-local shell templates and
delegates nonce creation, remote preflight execution, and attestation
validation to :func:`onnx_splitpoint_tool.energy.collector._run_energy_preflight`.
"""

import hashlib
import json
import os
import re
import shlex
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .energy.collector import _run_energy_preflight


COHORT_PREFLIGHT_SCHEMA = "onnx-splitpoint/resume-cohort-preflight"
COHORT_PREFLIGHT_SCHEMA_VERSION = 1
_NONCE_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_NONCE__"
_ATTESTATION_TOKEN = "__ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_COMMAND_BYTES = 2 * 1024 * 1024
_IDENTITY_FIELDS = ("backend", "model", "case", "setup_id")


class ResumeCohortPreflightError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = str(reason)
        self.detail = str(detail)
        message = self.reason
        if self.detail:
            message += f":{self.detail}"
        super().__init__(message)


@dataclass(frozen=True)
class _LoadedCommand:
    path: Path
    text: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int


@dataclass(frozen=True)
class _PreparedRow:
    index: int
    identity: tuple[str, str, str, str]
    selector: str
    command: _LoadedCommand
    preflight_command: _LoadedCommand
    command_contract: _LoadedCommand
    runtime_attestation_path_template: str
    expected_command_contract_sha256: str


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _normalise_existing_directory(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if raw.is_symlink():
        raise ResumeCohortPreflightError(f"{label}_is_symlink")
    try:
        resolved = raw.resolve(strict=True)
        mode = os.lstat(resolved).st_mode
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResumeCohortPreflightError(
            f"{label}_not_directory", str(raw),
        ) from exc
    if not stat.S_ISDIR(mode):
        raise ResumeCohortPreflightError(
            f"{label}_not_directory", str(raw),
        )
    return resolved


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _has_symlink_component(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    cursor = root
    try:
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                return True
    except OSError:
        return True
    return False


def _load_command_file(
    value: Any,
    *,
    field: str,
    plan_root: Path,
) -> _LoadedCommand:
    raw = str(value or "").strip()
    if not raw or "\x00" in raw or "\n" in raw or "\r" in raw:
        raise ResumeCohortPreflightError(f"{field}_path_invalid")
    lexical = Path(raw).expanduser()
    if not lexical.is_absolute():
        lexical = plan_root / lexical
    if lexical.is_symlink() or _has_symlink_component(lexical, plan_root):
        raise ResumeCohortPreflightError(f"{field}_not_regular")
    try:
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise ResumeCohortPreflightError(f"{field}_not_regular") from exc
    if not _path_is_within(resolved, plan_root):
        raise ResumeCohortPreflightError(f"{field}_outside_plan_root")

    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ResumeCohortPreflightError(f"{field}_not_regular") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_COMMAND_BYTES
        ):
            raise ResumeCohortPreflightError(f"{field}_not_regular")
        chunks: list[bytes] = []
        remaining = _MAX_COMMAND_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw_bytes = b"".join(chunks)
    if (
        len(raw_bytes) != before.st_size
        or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
    ):
        raise ResumeCohortPreflightError(f"{field}_changed_while_reading")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResumeCohortPreflightError(f"{field}_not_utf8") from exc
    if not text.strip() or "\x00" in text:
        raise ResumeCohortPreflightError(f"{field}_content_invalid")
    return _LoadedCommand(
        path=resolved,
        text=text,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        size_bytes=len(raw_bytes),
        device=int(after.st_dev),
        inode=int(after.st_ino),
        mtime_ns=int(after.st_mtime_ns),
    )


def _normalise_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    values: list[str] = []
    for field in _IDENTITY_FIELDS:
        value = str(row.get(field) or "").strip()
        if (
            not value
            or len(value) > 256
            or "|" in value
            or any(
                ord(character) < 32 or ord(character) == 127
                for character in value
            )
        ):
            raise ResumeCohortPreflightError(
                "row_identity_invalid", field,
            )
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def _normalise_runtime_attestation_template(value: Any) -> str:
    raw = str(value or "").strip()
    if (
        not raw
        or "\x00" in raw
        or "\n" in raw
        or "\r" in raw
        or not raw.startswith("/")
        or raw.startswith("//")
    ):
        raise ResumeCohortPreflightError(
            "runtime_attestation_template_invalid"
        )
    path = PurePosixPath(raw)
    if (
        not path.is_absolute()
        or path == PurePosixPath("/")
        or "." in path.parts
        or ".." in path.parts
        or str(path) != raw
        or not path.name
        or raw.count(_NONCE_TOKEN) != 1
    ):
        raise ResumeCohortPreflightError(
            "runtime_attestation_template_invalid"
        )
    return raw


def _normalise_sha256(value: Any, *, reason: str) -> str:
    raw = str(value or "").strip()
    if _SHA256_RE.fullmatch(raw) is None:
        raise ResumeCohortPreflightError(reason)
    return raw


def _validate_command_template_binding(
    loaded: _LoadedCommand,
    *,
    field: str,
    expected_sha256: str,
    require_expected_sha256_literal: bool,
) -> None:
    if _NONCE_TOKEN not in loaded.text:
        raise ResumeCohortPreflightError(f"{field}_nonce_token_missing")
    if _ATTESTATION_TOKEN not in loaded.text:
        raise ResumeCohortPreflightError(
            f"{field}_attestation_token_missing"
        )
    if require_expected_sha256_literal and expected_sha256 not in loaded.text:
        raise ResumeCohortPreflightError(
            f"{field}_expected_contract_sha256_missing"
        )


def _strict_command_contract(
    loaded: _LoadedCommand,
    *,
    identity: tuple[str, str, str, str],
    expected_sha256: str,
) -> dict[str, Any]:
    duplicate = False

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                duplicate = True
            value[key] = item
        return value

    try:
        payload = json.loads(loaded.text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ResumeCohortPreflightError(
            "command_contract_file_json_invalid"
        ) from exc
    if duplicate or not isinstance(payload, Mapping):
        raise ResumeCohortPreflightError(
            "command_contract_file_json_invalid"
        )
    if str(payload.get("contract_sha256") or "") != expected_sha256:
        raise ResumeCohortPreflightError(
            "command_contract_logical_sha256_mismatch"
        )
    for position, field in enumerate(_IDENTITY_FIELDS):
        if str(payload.get(field) or "").strip() != identity[position]:
            raise ResumeCohortPreflightError(
                "command_contract_row_identity_mismatch", field,
            )
    return dict(payload)


def _validate_preflight_contract_file_binding(
    preflight: _LoadedCommand,
    contract: _LoadedCommand,
    *,
    expected_file_sha256: str,
) -> None:
    contract_path = str(contract.path)
    candidates = {contract_path, shlex.quote(contract_path)}
    if not any(candidate in preflight.text for candidate in candidates):
        raise ResumeCohortPreflightError(
            "preflight_command_contract_file_binding_missing"
        )
    if expected_file_sha256 not in preflight.text:
        raise ResumeCohortPreflightError(
            "preflight_command_contract_file_sha256_binding_missing"
        )


def _prepare_row(
    row: Mapping[str, Any],
    *,
    index: int,
    plan_root: Path,
) -> _PreparedRow:
    identity = _normalise_identity(row)
    selector = "|".join(identity)
    expected_sha = _normalise_sha256(
        row.get("successful_command_contract_sha256"),
        reason="expected_command_contract_sha256_invalid",
    )
    command = _load_command_file(
        row.get("command_file"),
        field="command_file",
        plan_root=plan_root,
    )
    preflight = _load_command_file(
        row.get("preflight_command_file"),
        field="preflight_command_file",
        plan_root=plan_root,
    )
    command_contract = _load_command_file(
        row.get("command_contract_file"),
        field="command_contract_file",
        plan_root=plan_root,
    )
    expected_file_sha = _normalise_sha256(
        row.get("command_contract_file_sha256"),
        reason="command_contract_file_sha256_invalid",
    )
    if command_contract.sha256 != expected_file_sha:
        raise ResumeCohortPreflightError(
            "command_contract_file_sha256_mismatch"
        )
    _strict_command_contract(
        command_contract,
        identity=identity,
        expected_sha256=expected_sha,
    )
    _validate_command_template_binding(
        command,
        field="command_file",
        expected_sha256=expected_sha,
        require_expected_sha256_literal=True,
    )
    _validate_command_template_binding(
        preflight,
        field="preflight_command_file",
        expected_sha256=expected_sha,
        # Full Native preflights bind and transport the exact local contract
        # bytes, but the logical contract SHA is authoritative only in the
        # nonce-bound attestation returned by the remote preflight.
        require_expected_sha256_literal=False,
    )
    _validate_preflight_contract_file_binding(
        preflight,
        command_contract,
        expected_file_sha256=expected_file_sha,
    )
    return _PreparedRow(
        index=index,
        identity=identity,
        selector=selector,
        command=command,
        preflight_command=preflight,
        command_contract=command_contract,
        runtime_attestation_path_template=(
            _normalise_runtime_attestation_template(
                row.get(
                    "preflight_runtime_attestation_path_template"
                )
            )
        ),
        expected_command_contract_sha256=expected_sha,
    )


def _row_slug(row: _PreparedRow) -> str:
    readable = "__".join(
        re.sub(r"[^A-Za-z0-9_.-]", "_", value) or "unknown"
        for value in row.identity
    )
    digest = hashlib.sha256(row.selector.encode("utf-8")).hexdigest()[:12]
    return f"{readable}__{digest}"


def _loaded_command_report(command: _LoadedCommand) -> dict[str, Any]:
    return {
        "path": str(command.path),
        "sha256": command.sha256,
        "size_bytes": command.size_bytes,
        "source_stat": {
            "device": command.device,
            "inode": command.inode,
            "mtime_ns": command.mtime_ns,
            "size_bytes": command.size_bytes,
        },
    }


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


def _base_report(
    *,
    report_path: Path,
    selected_row_count: int,
) -> dict[str, Any]:
    return {
        "schema": COHORT_PREFLIGHT_SCHEMA,
        "schema_version": COHORT_PREFLIGHT_SCHEMA_VERSION,
        "ok": False,
        "status": "invalid_selection",
        "selected_row_count": selected_row_count,
        "prepared_row_count": 0,
        "preflight_attempted_count": 0,
        "preflight_passed_count": 0,
        "preflight_failed_count": 0,
        "measurement_wrapper_allowed": False,
        "measurement_wrapper_started": False,
        "collector_started": False,
        "workload_started": False,
        "selection_errors": [],
        "rows": [],
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


def _duplicate_selection_errors(
    prepared: Iterable[_PreparedRow],
) -> list[dict[str, Any]]:
    rows = list(prepared)
    errors: list[dict[str, Any]] = []
    uniqueness_fields = {
        "row_identity": lambda row: row.selector,
        "command_file": lambda row: str(row.command.path),
        "preflight_command_file": lambda row: str(row.preflight_command.path),
        "command_contract_file": lambda row: str(row.command_contract.path),
        "runtime_attestation_template": (
            lambda row: row.runtime_attestation_path_template
        ),
    }
    for name, getter in uniqueness_fields.items():
        indexes: dict[str, list[int]] = {}
        selectors: dict[str, list[str]] = {}
        for row in rows:
            key = getter(row)
            indexes.setdefault(key, []).append(row.index)
            selectors.setdefault(key, []).append(row.selector)
        for key, duplicate_indexes in indexes.items():
            if len(duplicate_indexes) < 2:
                continue
            errors.append(
                {
                    "reason": f"duplicate_{name}",
                    "value_sha256": hashlib.sha256(
                        key.encode("utf-8")
                    ).hexdigest(),
                    "row_indexes": duplicate_indexes,
                    "selectors": selectors[key],
                }
            )
    return errors


def run_resume_cohort_preflight(
    selected_rows: Iterable[Mapping[str, Any]],
    *,
    attempt_dir: str | Path,
    plan_root: str | Path,
    timeout_s: float = 900.0,
    max_age_s: float = 300.0,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Validate and execute all selected preflights as one fail-closed cohort."""
    attempt = _normalise_existing_directory(
        attempt_dir, label="attempt_dir",
    )
    plan = _normalise_existing_directory(plan_root, label="plan_root")
    if not _path_is_within(plan, attempt):
        raise ResumeCohortPreflightError("plan_root_outside_attempt_dir")
    if isinstance(timeout_s, bool) or float(timeout_s) <= 0:
        raise ResumeCohortPreflightError("timeout_s_invalid")
    if isinstance(max_age_s, bool) or float(max_age_s) <= 0:
        raise ResumeCohortPreflightError("max_age_s_invalid")

    rows = list(selected_rows)
    evidence_root = attempt / "resume_cohort_preflight"
    if evidence_root.exists() or evidence_root.is_symlink():
        raise ResumeCohortPreflightError(
            "cohort_evidence_directory_already_exists",
            str(evidence_root),
        )
    evidence_root.mkdir(mode=0o700)
    row_evidence_root = evidence_root / "rows"
    row_evidence_root.mkdir(mode=0o700)
    report_path = evidence_root / "resume_cohort_preflight.json"
    report = _base_report(
        report_path=report_path,
        selected_row_count=len(rows),
    )
    if not rows:
        report["selection_errors"].append(
            {"reason": "selected_rows_empty"}
        )
        return _finalise_report(report, report_path=report_path)

    prepared: list[_PreparedRow] = []
    preparation_results: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            preparation_results[index] = {
                "index": index,
                "ok": False,
                "status": "invalid",
                "reason": "selected_row_not_object",
            }
            continue
        try:
            item = _prepare_row(raw, index=index, plan_root=plan)
        except ResumeCohortPreflightError as exc:
            identity_values = [
                str(raw.get(field) or "").strip()
                for field in _IDENTITY_FIELDS
            ]
            preparation_results[index] = {
                "index": index,
                "identity": {
                    field: identity_values[position]
                    for position, field in enumerate(_IDENTITY_FIELDS)
                },
                "selector": "|".join(identity_values),
                "ok": False,
                "status": "invalid",
                "reason": exc.reason,
                "detail": exc.detail,
            }
            continue
        prepared.append(item)
        preparation_results[index] = {
            "index": index,
            "identity": dict(zip(_IDENTITY_FIELDS, item.identity)),
            "selector": item.selector,
            "ok": True,
            "status": "prepared",
            "command_file": _loaded_command_report(item.command),
            "preflight_command_file": _loaded_command_report(
                item.preflight_command
            ),
            "command_contract_file": _loaded_command_report(
                item.command_contract
            ),
            "runtime_attestation_path_template": (
                item.runtime_attestation_path_template
            ),
            "expected_command_contract_sha256": (
                item.expected_command_contract_sha256
            ),
        }
    report["prepared_row_count"] = len(prepared)
    report["rows"] = [
        preparation_results[index] for index in range(len(rows))
    ]
    preparation_failed = len(prepared) != len(rows)
    duplicate_errors = _duplicate_selection_errors(prepared)
    report["selection_errors"].extend(duplicate_errors)
    if preparation_failed or duplicate_errors:
        # Validate the whole local selection before any remote preflight.  This
        # guarantees that missing/symlinked files or ambiguous evidence paths
        # cannot result in a partially preflighted cohort.
        if preparation_failed:
            report["selection_errors"].append(
                {"reason": "row_preparation_failed"}
            )
        for result in report["rows"]:
            if result.get("status") == "prepared":
                result["ok"] = False
                result["status"] = "blocked_by_invalid_selection"
                result["reason"] = "cohort_selection_invalid"
        return _finalise_report(report, report_path=report_path)

    final_rows: list[dict[str, Any]] = []
    passed_count = 0
    for item in prepared:
        result = dict(preparation_results[item.index])
        evidence_dir = row_evidence_root / _row_slug(item)
        try:
            evidence, rendered_workload = _run_energy_preflight(
                item.preflight_command.text,
                workload_command_template=item.command.text,
                evidence_dir=evidence_dir,
                repeat_index="cohort",
                timeout_s=float(timeout_s),
                max_age_s=float(max_age_s),
                expected_command_contract_sha256=(
                    item.expected_command_contract_sha256
                ),
                runtime_attestation_path_template=(
                    item.runtime_attestation_path_template
                ),
                cancel_event=cancel_event,
            )
            if not isinstance(evidence, Mapping):
                raise ResumeCohortPreflightError(
                    "preflight_evidence_not_object"
                )
            evidence_payload = dict(evidence)
            evidence_ok = (
                evidence_payload.get("ok") is True
                and evidence_payload.get("collector_started") is False
                and evidence_payload.get("workload_started") is not True
                and str(
                    evidence_payload.get(
                        "expected_command_contract_sha256"
                    )
                    or ""
                )
                == item.expected_command_contract_sha256
                and isinstance(
                    evidence_payload.get("validation"), Mapping
                )
                and evidence_payload["validation"].get("ok") is True
            )
            result.update(
                {
                    "ok": bool(evidence_ok),
                    "status": (
                        "verified" if evidence_ok else "failed"
                    ),
                    "reason": (
                        "" if evidence_ok
                        else "preflight_evidence_not_verified"
                    ),
                    "evidence_dir": str(evidence_dir),
                    "preflight": evidence_payload,
                    "rendered_workload_sha256": hashlib.sha256(
                        str(rendered_workload).encode("utf-8")
                    ).hexdigest(),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "ok": False,
                    "status": "failed",
                    "reason": "preflight_runner_exception",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "evidence_dir": str(evidence_dir),
                }
            )
        if result["ok"]:
            passed_count += 1
        final_rows.append(result)

    attempted_count = len(prepared)
    failed_count = attempted_count - passed_count
    overall_ok = attempted_count == len(rows) and failed_count == 0
    report.update(
        {
            "ok": overall_ok,
            "status": "verified" if overall_ok else "failed",
            "preflight_attempted_count": attempted_count,
            "preflight_passed_count": passed_count,
            "preflight_failed_count": failed_count,
            "measurement_wrapper_allowed": overall_ok,
            "rows": final_rows,
        }
    )
    return _finalise_report(report, report_path=report_path)


__all__ = [
    "COHORT_PREFLIGHT_SCHEMA",
    "COHORT_PREFLIGHT_SCHEMA_VERSION",
    "ResumeCohortPreflightError",
    "run_resume_cohort_preflight",
]
