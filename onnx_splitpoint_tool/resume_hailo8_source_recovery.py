from __future__ import annotations

"""Recover exact generated Hailo-8 source bytes for a legacy resume.

The Hailo-8 Python detection runner writes two deterministic build-source
files before producing its sealed command contract.  Older result mirrors
retain ``CMakeLists.txt`` but may omit ``main.cpp``.  This module extracts only
the two literal source constants from the local runner without importing or
executing it, verifies the resulting bytes against the sealed contract hash,
and materialises attempt-local read-only source candidates.

No remote path is written here and no build, runner, SSH command, collector or
workload is executed.
"""

import ast
import hashlib
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .resume_artifact_rehydration import ArtifactRequirement


HAILO8_SOURCE_RECOVERY_SCHEMA = (
    "onnx-splitpoint/hailo8-resume-source-recovery"
)
HAILO8_SOURCE_RECOVERY_SCHEMA_VERSION = 1
HAILO8_SOURCE_RECOVERY_PROFILE = (
    "hailo8_python_detection_literal_sources_v1"
)
_RUNNER_RELATIVE_PATH = Path(
    "scripts/native_hailo_trt_fifo_from_benchmarkset.py"
)
_ROLE_CONSTANTS = {
    "cmake": "CMAKE_TXT",
    "generated_cpp": "CPP_SOURCE",
}
_MAX_RUNNER_BYTES = 8 * 1024 * 1024


class Hailo8ResumeSourceRecoveryError(ValueError):
    """Stable fail-closed error raised before any remote mutation."""

    def __init__(
        self,
        code: str,
        *,
        role: str = "",
        detail: str = "",
    ) -> None:
        self.code = str(code)
        self.role = str(role)
        self.detail = str(detail)
        message = self.code
        if self.role:
            message += f":role={self.role}"
        if self.detail:
            message += f":{self.detail}"
        super().__init__(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _strict_directory(value: str | Path, *, label: str) -> Path:
    lexical = Path(value).expanduser()
    if lexical.is_symlink():
        raise Hailo8ResumeSourceRecoveryError(f"{label}_is_symlink")
    try:
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise Hailo8ResumeSourceRecoveryError(
            f"{label}_not_directory"
        ) from exc
    if not resolved.is_dir():
        raise Hailo8ResumeSourceRecoveryError(f"{label}_not_directory")
    return resolved


def _safe_regular_file(path: Path, *, root: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Hailo8ResumeSourceRecoveryError(
            "runner_outside_tool_root"
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Hailo8ResumeSourceRecoveryError(
                "runner_path_contains_symlink"
            )
    try:
        resolved = path.resolve(strict=True)
        mode = os.lstat(resolved).st_mode
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        raise Hailo8ResumeSourceRecoveryError(
            "runner_not_regular"
        ) from exc
    if (
        not _path_is_within(resolved, root)
        or not stat.S_ISREG(mode)
    ):
        raise Hailo8ResumeSourceRecoveryError("runner_not_regular")
    return resolved


def _stable_read(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Hailo8ResumeSourceRecoveryError(
            "runner_open_failed"
        ) from exc
    try:
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size <= 0
                or before.st_size > _MAX_RUNNER_BYTES
            ):
                raise Hailo8ResumeSourceRecoveryError(
                    "runner_size_invalid"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
        except Hailo8ResumeSourceRecoveryError:
            raise
        except OSError as exc:
            raise Hailo8ResumeSourceRecoveryError(
                "runner_read_failed"
            ) from exc
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
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
        raise Hailo8ResumeSourceRecoveryError(
            "runner_changed_while_reading"
        )
    return raw


def _literal_source_constants(raw: bytes) -> dict[str, bytes]:
    try:
        text = raw.decode("utf-8")
        module = ast.parse(text)
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise Hailo8ResumeSourceRecoveryError(
            "runner_ast_invalid",
            detail=type(exc).__name__,
        ) from exc

    wanted = set(_ROLE_CONSTANTS.values())
    values: dict[str, bytes] = {}
    for node in module.body:
        targets: list[ast.expr] = []
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value
        if value_node is None:
            continue
        names = [
            target.id
            for target in targets
            if isinstance(target, ast.Name) and target.id in wanted
        ]
        for name in names:
            if name in values:
                raise Hailo8ResumeSourceRecoveryError(
                    "runner_literal_duplicate",
                    detail=name,
                )
            try:
                literal = ast.literal_eval(value_node)
            except (ValueError, TypeError, SyntaxError) as exc:
                raise Hailo8ResumeSourceRecoveryError(
                    "runner_literal_not_static",
                    detail=name,
                ) from exc
            if not isinstance(literal, str):
                raise Hailo8ResumeSourceRecoveryError(
                    "runner_literal_not_text",
                    detail=name,
                )
            values[name] = literal.encode("utf-8")

    missing = sorted(wanted - set(values))
    if missing:
        raise Hailo8ResumeSourceRecoveryError(
            "runner_literal_missing",
            detail=",".join(missing),
        )
    return values


def _requirement_fields(
    value: ArtifactRequirement | Mapping[str, Any],
) -> tuple[str, str, str, int | None]:
    if isinstance(value, ArtifactRequirement):
        return (
            str(value.role),
            str(value.remote_path),
            str(value.sha256).strip().lower(),
            value.size_bytes,
        )
    return (
        str(value.get("role") or ""),
        str(value.get("remote_path") or value.get("path") or ""),
        str(value.get("sha256") or "").strip().lower(),
        value.get("size_bytes"),
    )


def _write_read_only(path: Path, payload: bytes) -> None:
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise Hailo8ResumeSourceRecoveryError(
            "derived_source_parent_create_failed",
            detail=str(path.parent),
        ) from exc
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o400)
    except OSError as exc:
        raise Hailo8ResumeSourceRecoveryError(
            "derived_source_create_failed",
            detail=str(path),
        ) from exc
    try:
        try:
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise Hailo8ResumeSourceRecoveryError(
                        "derived_source_write_incomplete",
                        detail=str(path),
                    )
                written += count
            os.fsync(descriptor)
        except Hailo8ResumeSourceRecoveryError:
            raise
        except OSError as exc:
            raise Hailo8ResumeSourceRecoveryError(
                "derived_source_write_failed",
                detail=str(path),
            ) from exc
    finally:
        os.close(descriptor)


def materialize_hailo8_literal_sources(
    requirements: Iterable[
        ArtifactRequirement | Mapping[str, Any]
    ],
    *,
    tool_root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Create exact attempt-local candidates for authorised source roles."""
    tool = _strict_directory(tool_root, label="tool_root")
    destination_path = Path(destination).expanduser()
    if destination_path.exists() or destination_path.is_symlink():
        raise Hailo8ResumeSourceRecoveryError(
            "destination_already_exists"
        )
    parent = _strict_directory(
        destination_path.parent,
        label="destination_parent",
    )

    runner = _safe_regular_file(
        tool / _RUNNER_RELATIVE_PATH,
        root=tool,
    )
    runner_raw = _stable_read(runner)
    literals = _literal_source_constants(runner_raw)

    selected: list[tuple[str, str, str, int | None]] = []
    for requirement in requirements:
        fields = _requirement_fields(requirement)
        if fields[0] in _ROLE_CONSTANTS:
            selected.append(fields)
    if not selected:
        raise Hailo8ResumeSourceRecoveryError(
            "recoverable_requirements_empty"
        )

    pending: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for role, remote_path, expected_sha, expected_size in sorted(selected):
        identity = (role, remote_path, expected_sha)
        if identity in seen:
            continue
        seen.add(identity)
        constant = _ROLE_CONSTANTS[role]
        payload = literals[constant]
        actual_sha = _sha256_bytes(payload)
        if actual_sha != expected_sha:
            raise Hailo8ResumeSourceRecoveryError(
                "derived_source_sha256_mismatch",
                role=role,
                detail=(
                    f"expected={expected_sha};actual={actual_sha}"
                ),
            )
        if (
            expected_size is not None
            and (
                isinstance(expected_size, bool)
                or not isinstance(expected_size, int)
                or int(expected_size) != len(payload)
            )
        ):
            raise Hailo8ResumeSourceRecoveryError(
                "derived_source_size_mismatch",
                role=role,
            )
        basename = PurePosixPath(remote_path).name
        if not basename or basename in {".", ".."}:
            raise Hailo8ResumeSourceRecoveryError(
                "derived_source_remote_basename_invalid",
                role=role,
            )
        pending.append({
            "role": role,
            "remote_path": remote_path,
            "constant": constant,
            "sha256": expected_sha,
            "basename": basename,
            "payload": payload,
        })

    try:
        destination_path.mkdir(mode=0o700)
        root = destination_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Hailo8ResumeSourceRecoveryError(
            "destination_create_failed"
        ) from exc
    if not _path_is_within(root, parent):
        raise Hailo8ResumeSourceRecoveryError(
            "destination_outside_parent"
        )

    entries: list[dict[str, Any]] = []
    for pending_row in pending:
        role = str(pending_row["role"])
        remote_path = str(pending_row["remote_path"])
        constant = str(pending_row["constant"])
        expected_sha = str(pending_row["sha256"])
        basename = str(pending_row["basename"])
        payload = pending_row["payload"]
        if not isinstance(payload, bytes):
            raise Hailo8ResumeSourceRecoveryError(
                "derived_source_internal_payload_invalid",
                role=role,
            )
        local_path = (
            root
            / role
            / expected_sha[:16]
            / basename
        )
        _write_read_only(local_path, payload)
        try:
            post_write_valid = bool(
                local_path.is_file()
                and not local_path.is_symlink()
                and local_path.stat().st_size == len(payload)
                and _sha256_bytes(local_path.read_bytes()) == expected_sha
            )
        except OSError as exc:
            raise Hailo8ResumeSourceRecoveryError(
                "derived_source_post_write_read_failed",
                role=role,
            ) from exc
        if not post_write_valid:
            raise Hailo8ResumeSourceRecoveryError(
                "derived_source_post_write_verification_failed",
                role=role,
            )
        entries.append({
            "role": role,
            "remote_path": remote_path,
            "source_path": str(local_path),
            "constant": constant,
            "sha256": expected_sha,
            "size_bytes": len(payload),
            "status": "exact",
        })

    return {
        "schema": HAILO8_SOURCE_RECOVERY_SCHEMA,
        "schema_version": HAILO8_SOURCE_RECOVERY_SCHEMA_VERSION,
        "profile": HAILO8_SOURCE_RECOVERY_PROFILE,
        "ok": True,
        "status": "exact_sources_materialized",
        "local_only": True,
        "code_executed": False,
        "build_started": False,
        "remote_mutation_performed": False,
        "source_root": str(root),
        "runner_source_path": str(runner),
        "runner_source_sha256": _sha256_bytes(runner_raw),
        "artifact_count": len(entries),
        "total_bytes": sum(int(row["size_bytes"]) for row in entries),
        "entries": entries,
    }


__all__ = [
    "HAILO8ResumeSourceRecoveryError",
    "HAILO8_SOURCE_RECOVERY_PROFILE",
    "HAILO8_SOURCE_RECOVERY_SCHEMA",
    "HAILO8_SOURCE_RECOVERY_SCHEMA_VERSION",
    "materialize_hailo8_literal_sources",
]
