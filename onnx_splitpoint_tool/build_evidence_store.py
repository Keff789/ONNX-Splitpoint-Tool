from __future__ import annotations

"""Persistent, exact compiler outcomes shared by normal tool runs.

The existing build-evidence key, record and index contracts remain authoritative.
Only this module's live index is updated; recovered snapshots are read in place.
No original run, receipt, artifact or historical index is removed or rewritten.
"""

import contextlib
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

from .build_evidence import (
    BUILD_INDEX_SCHEMA,
    BuildEvidenceDecision,
    BuildEvidenceError,
    _ensure_directory_nofollow,
    _lexical_absolute,
    _open_directory_nofollow,
    _strict_json,
    build_evidence_index,
    canonical_json_bytes,
    canonical_sha256,
    lookup_build_evidence,
    make_build_evidence_record,
    validate_build_evidence_index,
    validate_build_key,
)

STORE_ROOT_ENV = "ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT"
LIVE_INDEX_FILENAME = "live_build_evidence.json"
LOCK_FILENAME = ".live_build_evidence.lock"
_JSON_LIMIT = 64 * 1024 * 1024
_LOCK_TIMEOUT_SECONDS = 30.0
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def default_build_evidence_root() -> Path:
    configured = os.environ.get(STORE_ROOT_ENV, "").strip()
    return _lexical_absolute(
        configured or Path.home() / ".onnx_splitpoint_tool" / "build_evidence",
        label="build_evidence_store.root",
    )


def _read_snapshot(parent_fd: int, filename: str) -> Any:
    """Read one opened generation; a concurrent atomic rename is harmless."""
    descriptor = os.open(
        filename,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BuildEvidenceError("index_not_regular_file", filename)
        if before.st_size > _JSON_LIMIT:
            raise BuildEvidenceError("input_too_large", filename)
        parts: list[bytes] = []
        length = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            length += len(block)
            if length > _JSON_LIMIT:
                raise BuildEvidenceError("input_too_large", filename)
            parts.append(block)
        after = os.fstat(descriptor)
        # Unlinking an old generation after replace changes ctime/nlink, but
        # does not change that opened immutable snapshot's bytes or mtime.
        if (
            (before.st_size, before.st_mtime_ns, before.st_mode)
            != (after.st_size, after.st_mtime_ns, after.st_mode)
            or length != after.st_size
        ):
            raise BuildEvidenceError("index_modified_during_read", filename)
        return _strict_json(b"".join(parts), label=filename)
    finally:
        os.close(descriptor)


def _thread_lock(root: Path) -> threading.Lock:
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(str(root), threading.Lock())


@contextlib.contextmanager
def _exclusive_writer(root: Path) -> Iterator[int]:
    """Hold a stable lock inode across reading, merging and replacing the index."""
    try:
        import fcntl
    except ImportError as exc:  # Existing exact-evidence I/O requires POSIX too.
        raise BuildEvidenceError("build_evidence_lock_platform_unsupported") from exc
    lock = _thread_lock(root)
    if not lock.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        raise BuildEvidenceError("build_evidence_lock_timeout", str(root))
    parent_fd = -1
    lock_fd = -1
    locked = False
    try:
        parent_fd = _ensure_directory_nofollow(root, label="build_evidence_store.root")
        lock_fd = os.open(
            LOCK_FILENAME,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise BuildEvidenceError("build_evidence_lock_not_regular")
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while not locked:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise BuildEvidenceError("build_evidence_lock_timeout", str(root))
                time.sleep(0.05)
        yield parent_fd
    finally:
        if locked:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        if lock_fd >= 0:
            os.close(lock_fd)
        if parent_fd >= 0:
            os.close(parent_fd)
        lock.release()


def _publish_snapshot(parent_fd: int, index: Mapping[str, Any]) -> None:
    """Fsync a complete replacement before the single atomic publication."""
    payload = canonical_json_bytes(index) + b"\n"
    if len(payload) > _JSON_LIMIT:
        raise BuildEvidenceError("live_index_too_large")
    temporary_name = f".{LIVE_INDEX_FILENAME}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise BuildEvidenceError("live_index_short_write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        # Only the owned live index is replaced. Historic snapshots are never
        # copied, truncated, renamed, or deleted by automatic recording.
        os.replace(
            temporary_name,
            LIVE_INDEX_FILENAME,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.fsync(parent_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name, dir_fd=parent_fd)


class BuildEvidenceStore:
    """Automatic shared terminal evidence plus readonly recovered snapshots.

    Lookup never creates files or directories. It discovers immediate ``*.json``
    children in the dedicated store root on every call, so newly recovered
    indices and records become visible without restarting a workflow. Unrelated
    valid JSON schemas are ignored. Malformed or invalid evidence is an explicit
    ERROR decision, never a negative HIT or a silent promise of protection.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = (
            default_build_evidence_root()
            if root is None
            else _lexical_absolute(root, label="build_evidence_store.root")
        )
        self.index_path = self.root / LIVE_INDEX_FILENAME

    def lookup(self, key: Mapping[str, Any]) -> BuildEvidenceDecision:
        exact_key = validate_build_key(key)
        key_hash = canonical_sha256(exact_key)
        try:
            parent_fd = _open_directory_nofollow(self.root, label="build_evidence_store.root")
        except BuildEvidenceError as exc:
            if exc.code == "required_path_missing":
                return BuildEvidenceDecision("MISS", key_hash, False, None, "exact_key_not_found")
            return self._error_decision(key_hash, [{"path": str(self.root), "reason": exc.code}])
        except OSError as exc:
            return self._error_decision(key_hash, [{"path": str(self.root), "reason": type(exc).__name__}])
        records: dict[str, dict[str, Any]] = {}
        diagnostics: list[dict[str, str]] = []
        try:
            filenames = sorted(name for name in os.listdir(parent_fd) if name.endswith(".json"))
            for filename in filenames:
                try:
                    value = _read_snapshot(parent_fd, filename)
                    recognized = (
                        filename == LIVE_INDEX_FILENAME
                        or isinstance(value, Mapping) and (
                            value.get("schema") == BUILD_INDEX_SCHEMA
                            or str(value.get("schema") or "").startswith("onnx-splitpoint/build-evidence-index/")
                            or "index_payload_sha256" in value
                        )
                    )
                    if not recognized:
                        continue
                    index = validate_build_evidence_index(value)
                    for record in index["records"]:
                        records[record["record_sha256"]] = record
                except (BuildEvidenceError, OSError) as exc:
                    diagnostics.append({
                        "path": str(self.root / filename),
                        "reason": exc.code if isinstance(exc, BuildEvidenceError) else type(exc).__name__,
                    })
        except OSError as exc:
            diagnostics.append({"path": str(self.root), "reason": type(exc).__name__})
        finally:
            os.close(parent_fd)
        if diagnostics:
            return self._error_decision(key_hash, diagnostics)
        index = build_evidence_index(list(records.values()), source_run_name="shared-build-evidence")
        return lookup_build_evidence(index, exact_key)

    @staticmethod
    def _error_decision(key_hash: str, diagnostics: list[dict[str, str]]) -> BuildEvidenceDecision:
        return BuildEvidenceDecision(
            status="ERROR",
            key_sha256=key_hash,
            reusable=False,
            state=None,
            reason="build_evidence_index_unreadable_or_invalid",
            evidence_origin={"index_errors": diagnostics},
        )

    def record(
        self,
        key: Mapping[str, Any],
        state: str,
        *,
        evidence_origin: Mapping[str, Any] | None = None,
        reason_code: str = "",
        artifact: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist one complete terminal record, retaining all prior outcomes.

        Deterministic failures can subsequently be reused without their original
        raw files. Infrastructure errors and unknown/aborted attempts remain
        visible but nonreusable according to the existing evidence contract.
        Distinct outcomes for the same exact request are retained as a conflict.
        Corrupt existing live data is never overwritten with an empty index.
        """
        record = make_build_evidence_record(
            key,
            state,
            evidence_origin=(
                {"kind": "automatic_terminal_build"}
                if evidence_origin is None else evidence_origin
            ),
            reason_code=reason_code,
            artifact=artifact,
        )
        with _exclusive_writer(self.root) as parent_fd:
            try:
                previous = validate_build_evidence_index(_read_snapshot(parent_fd, LIVE_INDEX_FILENAME))
            except FileNotFoundError:
                previous = build_evidence_index(source_run_name="shared-build-evidence")
            existing = {row["record_sha256"]: row for row in previous["records"]}
            if record["record_sha256"] in existing:
                return record
            existing[record["record_sha256"]] = record
            merged = build_evidence_index(
                list(existing.values()),
                source_run_name=previous["source_run"]["basename"],
                unresolved_observations=previous["unresolved_observations"],
                source_observations=previous["source_observations"],
            )
            _publish_snapshot(parent_fd, merged)
        return record
