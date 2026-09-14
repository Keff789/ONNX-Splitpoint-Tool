"""Crash-safe publication of the existing HEF/receipt/cache-meta contract.

The public filenames are compatibility links through one directory pointer.
Only that pointer is committed, after all three files have been flushed and
validated. Readers resolve the HEF once and read its siblings in that immutable
generation. Previous generations remain available as complete local backups.
No additional artifact identity or hash contract is introduced.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterator, Mapping
import uuid

RECEIPT_NAME = "hailo_hef_build_receipt.json"
META_NAME = "cache_meta.json"
GENERATIONS_NAME = ".hailo-generations"
POINTER_NAME = ".hailo-current"


def snapshot_hef(path: Path) -> Path:
    """Pin all subsequent sibling reads to the same published generation."""
    return Path(path).resolve()


def resolve_published_bundle_member(
    path: Path, *, root: Path,
    hash_fn: Callable[[Path], str | None] | None = None,
) -> Path | None:
    """Resolve only the local compatibility links made by ``publish_bundle``.

    Return a real generation file (or its directory for ``.hailo-current``).
    Unrelated, redirected, incomplete or invalid bundles return None, so callers
    can retain their existing symlink rejection. This is a read-only admission
    check using the existing receipt and metadata contract, not a cache lookup.
    """
    from .hailo_backend import _load_valid_hailo_receipt

    path, root = Path(path), Path(root)
    if not path.is_symlink():
        return None
    if path.name not in {POINTER_NAME, META_NAME, RECEIPT_NAME} and path.suffix != ".hef":
        return None
    directory = path.parent
    try:
        ancestor = root
        if ancestor.is_symlink():
            return None
        for part in directory.relative_to(root).parts:
            if part in {"", ".", ".."}:
                return None
            ancestor = ancestor / part
            if ancestor.is_symlink():
                return None
        pointer = directory / POINTER_NAME
        if not pointer.is_symlink():
            return None
        target = os.readlink(pointer)
        if not re.fullmatch(re.escape(GENERATIONS_NAME) + r"/[0-9a-f]{32}", target):
            return None
        generations = directory / GENERATIONS_NAME
        generation = directory / target
        if (generations.is_symlink() or generation.is_symlink()
                or not generation.is_dir()):
            return None
        generation.resolve(strict=True).relative_to(root.resolve(strict=True))
        members = list(generation.iterdir())
        hefs = [member for member in members if member.suffix == ".hef"]
        if len(hefs) != 1:
            return None
        names = {hefs[0].name, RECEIPT_NAME, META_NAME}
        if {member.name for member in members} != names:
            return None
        for name in names:
            member = generation / name
            alias = directory / name
            if member.is_symlink() or not member.is_file():
                return None
            if (not alias.is_symlink()
                    or os.readlink(alias) != f"{POINTER_NAME}/{name}"):
                return None
        if path.name != POINTER_NAME and path.name not in names:
            return None
        if _load_valid_hailo_receipt(hefs[0], allow_legacy_v2=True, hash_fn=hash_fn) is None:
            return None
        return generation if path.name == POINTER_NAME else generation / path.name
    except (OSError, ValueError, RuntimeError):
        return None


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return  # Windows does not support opening directories with os.open.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _publication_lock(directory: Path) -> Iterator[None]:
    """Serialize independent threads/processes adopting or publishing a bundle."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".hailo-publish.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt
            handle.write(b"\0")
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _copy_fsynced(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _write_json_fsynced(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_link(path: Path, target: str) -> None:
    pending = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        os.symlink(target, pending, target_is_directory=(path.name == POINTER_NAME))
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def publish_bundle(
    *, source_hef: Path, destination: Path, receipt: Mapping[str, Any],
    cache_meta: Mapping[str, Any], validator: Callable[[Path], Any],
) -> Path:
    """Commit a complete generation, preserving the old one on failure.

    The validator must check the staged HEF against its receipt and metadata.
    Symlink support is required; unsupported filesystems fail before changing
    existing public files. Returning the resolved HEF also gives callers a stable
    snapshot for registration/backup even if another publisher runs immediately.
    """
    source = snapshot_hef(source_hef)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if (source.stat().st_size != receipt.get("hef_size_bytes")
            or digest.hexdigest() != str(receipt.get("hef_sha256") or "")):
        raise ValueError("hailo_bundle_validation_failed:source_receipt_mismatch")
    destination = Path(destination).absolute()
    directory = destination.parent
    names = (destination.name, RECEIPT_NAME, META_NAME)
    with _publication_lock(directory):
        current = snapshot_hef(destination)
        if current.is_file() and current.parent.parent.name == GENERATIONS_NAME:
            try:
                current_receipt = json.loads((current.parent / RECEIPT_NAME).read_text(encoding="utf-8"))
                current_meta = json.loads((current.parent / META_NAME).read_text(encoding="utf-8"))
                comparable_meta = lambda value: {
                    key: item for key, item in value.items()
                    if key not in {"created_at", "source"}
                }
                if (current_receipt == dict(receipt)
                        and comparable_meta(current_meta) == comparable_meta(cache_meta)
                        and validator(current)):
                    return current
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        generations = directory / GENERATIONS_NAME
        generations.mkdir(exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".pending-", dir=generations))
        generation: Path | None = None
        try:
            _copy_fsynced(source, stage / destination.name)
            _write_json_fsynced(stage / RECEIPT_NAME, receipt)
            _write_json_fsynced(stage / META_NAME, cache_meta)
            if not validator(stage / destination.name):
                raise ValueError("hailo_bundle_validation_failed")
            _fsync_directory(stage)
            generation = generations / uuid.uuid4().hex
            os.replace(stage, generation)
            _fsync_directory(generations)

            pointer = directory / POINTER_NAME
            if not pointer.is_symlink():
                if pointer.exists():
                    raise ValueError("hailo_bundle_pointer_is_not_a_symlink")
                # Test support before touching any existing compatibility file.
                probe = directory / (".hailo-link-probe-" + uuid.uuid4().hex)
                try:
                    os.symlink(os.path.relpath(generation, directory), probe,
                               target_is_directory=True)
                finally:
                    probe.unlink(missing_ok=True)
                previous = generations / ("previous-" + uuid.uuid4().hex)
                previous.mkdir()
                for name in names:
                    old = directory / name
                    if old.is_file():
                        _copy_fsynced(old.resolve(), previous / name)
                _fsync_directory(previous)
                _fsync_directory(generations)
                # This initial pointer still denotes the prior bytes, including
                # incomplete legacy state. Replacing each alias cannot expose
                # any part of the new generation before the final commit.
                _replace_link(pointer, os.path.relpath(previous, directory))

            for name in names:
                alias = directory / name
                target = str(Path(POINTER_NAME) / name)
                if not alias.is_symlink() or os.readlink(alias) != target:
                    _replace_link(alias, target)
            _fsync_directory(directory)
            _replace_link(pointer, os.path.relpath(generation, directory))
            _fsync_directory(directory)
            return generation / destination.name
        finally:
            # Validated generations are retained even if publication fails: a
            # costly successful compiler output must not disappear with an I/O
            # failure during the pointer commit.
            if stage.exists():
                shutil.rmtree(stage)
