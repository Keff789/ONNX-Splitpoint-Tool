from __future__ import annotations

"""Non-mutating filesystem admission checks for workflow writers."""

import os
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FilesystemWriteInspection:
    requested_path: Path
    probe_path: Path
    writable: bool
    read_only_mount: bool
    permission_bits_allow: bool
    os_access_allow: bool
    free_bytes: int
    free_inodes: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_path": str(self.requested_path),
            "probe_path": str(self.probe_path),
            "writable": self.writable,
            "read_only_mount": self.read_only_mount,
            "permission_bits_allow": self.permission_bits_allow,
            "os_access_allow": self.os_access_allow,
            "free_bytes": self.free_bytes,
            "free_inodes": self.free_inodes,
            "reason": self.reason,
        }


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.is_file():
        candidate = candidate.parent
    return candidate


def _mode_allows_directory_write(path: Path) -> bool:
    info = path.stat()
    mode = info.st_mode
    uid = os.geteuid()
    groups = set(os.getgroups()) | {os.getegid()}
    if uid == info.st_uid:
        return bool(mode & stat.S_IWUSR) and bool(mode & stat.S_IXUSR)
    if info.st_gid in groups:
        return bool(mode & stat.S_IWGRP) and bool(mode & stat.S_IXGRP)
    return bool(mode & stat.S_IWOTH) and bool(mode & stat.S_IXOTH)


def inspect_write_target(path: str | Path) -> FilesystemWriteInspection:
    """Inspect a directory target without creating a probe file or directory."""

    requested = Path(path).expanduser()
    try:
        probe = _nearest_existing_directory(requested)
        if not probe.is_dir():
            raise OSError(f"no existing directory ancestor for {requested}")
        fs = os.statvfs(probe)
        read_only = bool(
            getattr(os, "ST_RDONLY", 1)
            and fs.f_flag & getattr(os, "ST_RDONLY", 1)
        )
        mode_ok = _mode_allows_directory_write(probe)
        access_ok = os.access(probe, os.W_OK | os.X_OK)
        free_bytes = int(fs.f_bavail) * int(fs.f_frsize or fs.f_bsize)
        free_inodes = int(fs.f_favail)
        if read_only:
            reason = "read_only_mount"
        elif not mode_ok:
            reason = "directory_permissions_read_only"
        elif not access_ok:
            reason = "directory_access_denied"
        else:
            reason = "ok"
        return FilesystemWriteInspection(
            requested_path=requested,
            probe_path=probe,
            writable=bool(not read_only and mode_ok and access_ok),
            read_only_mount=read_only,
            permission_bits_allow=mode_ok,
            os_access_allow=access_ok,
            free_bytes=free_bytes,
            free_inodes=free_inodes,
            reason=reason,
        )
    except OSError as exc:
        return FilesystemWriteInspection(
            requested_path=requested,
            probe_path=requested,
            writable=False,
            read_only_mount=False,
            permission_bits_allow=False,
            os_access_allow=False,
            free_bytes=0,
            free_inodes=0,
            reason=f"filesystem_inspection_failed:{type(exc).__name__}",
        )


def require_write_target(
    path: str | Path,
    *,
    operation: str,
    minimum_free_bytes: int = 0,
    minimum_free_inodes: int = 0,
) -> FilesystemWriteInspection:
    """Fail before mutation when a target cannot safely accept new output."""

    inspection = inspect_write_target(path)
    problems: list[str] = []
    if not inspection.writable:
        problems.append(inspection.reason)
    if inspection.free_bytes < max(0, int(minimum_free_bytes)):
        problems.append(
            f"free_bytes={inspection.free_bytes}<required={int(minimum_free_bytes)}"
        )
    if inspection.free_inodes < max(0, int(minimum_free_inodes)):
        problems.append(
            f"free_inodes={inspection.free_inodes}<required={int(minimum_free_inodes)}"
        )
    if problems:
        raise RuntimeError(
            f"{operation} requires a writable output directory before any files "
            f"are created: requested={inspection.requested_path}, "
            f"checked={inspection.probe_path}, reason={';'.join(problems)}. "
            "Select a writable output root; existing measurement runs remain unchanged."
        )
    return inspection


def require_output_outside_source(
    source_root: str | Path,
    output_path: str | Path,
    *,
    operation: str,
) -> tuple[Path, Path]:
    """Reject an export whose destination would modify its source tree."""

    source = Path(source_root).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve(strict=False)
    try:
        output.relative_to(source)
    except ValueError:
        return source, output
    raise RuntimeError(
        f"{operation} output must be outside the read-only source run: "
        f"source={source}, output={output}"
    )
