from __future__ import annotations

"""ZIP helpers with portable timestamps and predictable metadata.

Python's :mod:`zipfile` rejects source mtimes before 1980. Some generated or
copied accelerator artifacts legitimately retain Unix epoch timestamps, so
Debug Pack creation must clamp metadata instead of dropping the diagnostic.
"""

import os
import hashlib
import json
import math
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

ZIP_MIN_EPOCH = 315532800.0  # 1980-01-01 00:00:00 UTC
ZIP_MAX_EPOCH = 4354819198.0  # 2107-12-31 23:59:58 UTC
ZIP_MAX_YEAR = 2107


def portable_zip_datetime(timestamp: float | int | None) -> tuple[int, int, int, int, int, int]:
    try:
        value = float(timestamp if timestamp is not None else time.time())
    except Exception:
        value = time.time()
    if math.isnan(value):
        value = time.time()
    elif value == math.inf:
        value = ZIP_MAX_EPOCH
    elif value == -math.inf:
        value = ZIP_MIN_EPOCH
    value = min(ZIP_MAX_EPOCH, max(ZIP_MIN_EPOCH, value))
    # UTC makes archive metadata deterministic across collector hosts and the
    # epoch clamp before conversion also handles values that ``localtime``
    # cannot represent on narrower platforms.
    try:
        parts = time.gmtime(value)
    except (OSError, OverflowError, ValueError):
        parts = time.gmtime(ZIP_MIN_EPOCH)
    year = min(ZIP_MAX_YEAR, max(1980, int(parts.tm_year)))
    return (year, int(parts.tm_mon), int(parts.tm_mday), int(parts.tm_hour), int(parts.tm_min), int(parts.tm_sec))


def zipinfo_for_path(path: Path, arcname: str, *, compress_type: int = zipfile.ZIP_DEFLATED) -> zipfile.ZipInfo:
    path = Path(path)
    st = path.stat()
    info = zipfile.ZipInfo(str(arcname).replace(os.sep, "/"), date_time=portable_zip_datetime(st.st_mtime))
    info.compress_type = compress_type
    info.external_attr = (int(st.st_mode) & 0xFFFF) << 16
    info.create_system = 3
    return info


def require_safe_pack_source(path: Path, allowed_root: Path) -> Path:
    """Return a regular source file contained below *allowed_root*.

    Every lexical component below the root is checked with ``lstat`` semantics;
    symlinks are rejected even when they ultimately resolve back into the run.
    This keeps an uploaded pack from following attacker- or accident-controlled
    links to unrelated local files.
    """

    root = Path(allowed_root).expanduser().resolve(strict=True)
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"pack source is outside the allowed root: {candidate} not below {root}"
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"pack source symlink is not allowed: {cursor}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"resolved pack source escapes the allowed root: {resolved}"
        ) from exc
    if not resolved.is_file():
        raise ValueError(f"pack source is not a regular file: {resolved}")
    return resolved


def require_safe_pack_directory(path: Path, allowed_root: Path) -> Path:
    """Return a real directory contained below *allowed_root*.

    Pack builders must validate a directory before enumerating it.  Validating
    only the files discovered by ``rglob`` is too late: a symlinked directory
    can expose names from outside the EvaluationRun even when the later file
    copy is correctly rejected.
    """

    root = Path(allowed_root).expanduser().resolve(strict=True)
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"pack directory is outside the allowed root: {candidate} not below {root}"
        ) from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"pack directory symlink is not allowed: {cursor}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"resolved pack directory escapes the allowed root: {resolved}"
        ) from exc
    if not resolved.is_dir():
        raise ValueError(f"pack source is not a directory: {resolved}")
    return resolved


def iter_safe_pack_files(path: Path, allowed_root: Path) -> list[Path]:
    """Return files below *path* without following or enumerating symlinks."""

    root = Path(allowed_root).expanduser().resolve(strict=True)
    start = require_safe_pack_directory(path, root)
    pending = [start]
    files: list[Path] = []
    while pending:
        current = pending.pop()
        try:
            children = sorted(current.iterdir(), key=lambda item: item.name)
        except OSError:
            continue
        directories: list[Path] = []
        for child in children:
            try:
                if child.is_symlink():
                    continue
                if child.is_dir():
                    directories.append(require_safe_pack_directory(child, root))
                elif child.is_file():
                    files.append(require_safe_pack_source(child, root))
            except (OSError, RuntimeError, ValueError):
                continue
        pending.extend(reversed(directories))
    return files


def write_path_portable(
    zf: zipfile.ZipFile,
    path: Path,
    arcname: str,
    *,
    compress_type: int = zipfile.ZIP_DEFLATED,
    chunk_size: int = 1024 * 1024,
    allowed_root: Path | None = None,
) -> int:
    """Write *path* with a ZIP-valid timestamp and return its byte size."""
    path = Path(path)
    if allowed_root is not None:
        path = require_safe_pack_source(path, Path(allowed_root))
    elif path.is_symlink():
        raise ValueError(f"pack source symlink is not allowed: {path}")
    info = zipinfo_for_path(path, arcname, compress_type=compress_type)
    size = 0
    with path.open("rb") as src, zf.open(info, "w") as dst:
        while True:
            block = src.read(chunk_size)
            if not block:
                break
            dst.write(block)
            size += len(block)
    return size


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def temporary_zip_path(destination: Path) -> Path:
    """Reserve a same-directory temporary path for atomic ZIP publication.

    Keeping the temporary archive next to its destination guarantees that the
    final :func:`os.replace` does not cross filesystems.  The reservation file
    is removed before :class:`zipfile.ZipFile` opens the path so callers retain
    the usual exclusive ``"w"`` creation semantics.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(descriptor)
    path = Path(raw_path)
    path.unlink()
    return path


def verify_zip_archive(
    path: Path,
    *,
    required_members: Iterable[str] = (),
) -> dict[str, Any]:
    """Fully validate a closed ZIP and return its immutable file identity.

    Opening the central directory alone would still miss truncated compressed
    members.  ``testzip`` therefore reads and CRC-checks every member before a
    pack can be published.  Duplicate member names are rejected because they
    make later evidence extraction ambiguous.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ZIP candidate does not exist: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        seen_names: set[str] = set()
        duplicate_names: set[str] = set()
        for name in names:
            if name in seen_names:
                duplicate_names.add(name)
            else:
                seen_names.add(name)
        if duplicate_names:
            raise ValueError(
                f"ZIP contains duplicate members: {sorted(duplicate_names)[:10]}"
            )
        missing = sorted(set(str(name) for name in required_members) - set(names))
        if missing:
            raise ValueError(f"ZIP is missing required members: {missing}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"CRC/decompression check failed for {bad_member}")
    return {
        "schema": "onnx-splitpoint/archive-verification",
        "schema_version": 1,
        "status": "verified",
        "structural_check": "central_directory_and_all_member_crc",
        "archive_size_bytes": int(path.stat().st_size),
        "archive_sha256": _sha256_file(path),
        "member_count": len(names),
        "required_members": sorted(str(name) for name in required_members),
    }


def publish_verified_zip(
    temporary_path: Path,
    destination: Path,
    *,
    required_members: Iterable[str] = (),
    build_identity: Mapping[str, Any] | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify and atomically publish a ZIP plus an adjacent identity manifest.

    A ZIP cannot contain its own final byte size and SHA-256 without creating a
    self-reference.  Those final values are therefore recorded in the adjacent
    ``*.manifest.json`` after the closed temporary archive has been verified.
    The in-archive manifest should retain the build identity and point to this
    sidecar.  No partially written destination is ever exposed.
    """
    temporary_path = Path(temporary_path)
    destination = Path(destination)
    try:
        verification = verify_zip_archive(temporary_path, required_members=required_members)
    except Exception:
        # A failed candidate must remain invisible to consumers and must not
        # leave an apparently publishable temporary archive behind.  Callers
        # still receive the original verification exception.
        temporary_path.unlink(missing_ok=True)
        raise
    sidecar = destination.with_name(destination.name + ".manifest.json")
    sidecar_payload: dict[str, Any] = {
        **verification,
        "archive_path": str(destination),
        "archive_name": destination.name,
        "tool_build": dict(build_identity or {}),
        **dict(manifest_extra or {}),
    }
    sidecar_tmp = temporary_zip_path(sidecar)
    try:
        sidecar_tmp.write_text(
            json.dumps(sidecar_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, destination)
        os.replace(sidecar_tmp, sidecar)
    finally:
        temporary_path.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
    return {
        **sidecar_payload,
        "archive_path": str(destination),
        "verification_manifest": str(sidecar),
    }
