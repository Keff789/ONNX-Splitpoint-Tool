from __future__ import annotations

import json
import os
import secrets
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schema import dumps_json_safe


SCHEMA = "onnx-splitpoint/generation-state"
SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")



def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default



def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_DIRECTORY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    return flags


def _open_parent_dir_nofollow(path: Path) -> tuple[int, str]:
    """Open (and, when needed, create) ``path.parent`` without symlinks.

    Every component is resolved relative to an already-open directory.  This
    avoids switching back to pathname traversal after a component has been
    checked and gives the writer a stable parent directory for all following
    operations.
    """

    path = Path(path)
    target_name = path.name
    if not target_name or target_name in {".", ".."}:
        raise ValueError(f"invalid generation-state path: {path}")

    parts = path.parts
    if path.is_absolute():
        base = path.anchor
        parent_parts = parts[1:-1]
    else:
        base = "."
        parent_parts = parts[:-1]
    if any(part in {"", ".", ".."} for part in parent_parts):
        raise ValueError(f"unsafe generation-state parent path: {path.parent}")

    flags = _directory_open_flags()
    current_fd = os.open(base, flags)
    try:
        for component in parent_parts:
            try:
                child_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o777, dir_fd=current_fd)
                except FileExistsError:
                    # Another writer may have created it.  The no-follow open
                    # below still verifies that it is a real directory.
                    pass
                child_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = child_fd
        return current_fd, target_name
    except BaseException:
        os.close(current_fd)
        raise


def _validate_existing_target(parent_fd: int, target_name: str) -> None:
    try:
        target_stat = os.stat(
            target_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return
    if not stat.S_ISREG(target_stat.st_mode):
        raise RuntimeError(
            "generation-state destination exists but is not a regular file: "
            f"{target_name}"
        )


def _open_random_temp(parent_fd: int) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    for _attempt in range(64):
        temp_name = f".generation-state-{secrets.token_hex(16)}.tmp"
        try:
            return os.open(temp_name, flags, 0o600, dir_fd=parent_fd), temp_name
        except FileExistsError:
            # O_EXCL makes a pre-planted file or symlink harmless.  Pick a new
            # random name rather than opening or removing the attacker's path.
            continue
    raise FileExistsError("could not reserve a unique generation-state temp file")


def _write_all(fd: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("short write while persisting generation state")
        remaining = remaining[written:]


def write_json_atomic(path: Path, data: Any) -> None:
    payload = dumps_json_safe(data, indent=2).encode("utf-8")
    parent_fd, target_name = _open_parent_dir_nofollow(Path(path))
    temp_fd: Optional[int] = None
    temp_name: Optional[str] = None
    try:
        _validate_existing_target(parent_fd, target_name)
        temp_fd, temp_name = _open_random_temp(parent_fd)
        try:
            _write_all(temp_fd, payload)
            os.fsync(temp_fd)
        finally:
            os.close(temp_fd)
            temp_fd = None

        # Recheck immediately before replacement so a destination that was
        # changed to a symlink or special file during the write is rejected.
        _validate_existing_target(parent_fd, target_name)
        os.replace(
            temp_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temp_name = None
        os.fsync(parent_fd)
    finally:
        if temp_fd is not None:
            os.close(temp_fd)
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)



def init_state(
    *,
    model_name: str,
    model_source: str,
    requested_cases: int,
    ranked_candidates: Iterable[int],
    candidate_search_pool: Iterable[int],
    hef_full_policy: str,
    run_mode: str,
) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "model_name": str(model_name),
        "model_source": str(model_source),
        "requested_cases": int(requested_cases),
        "ranked_candidates": [int(x) for x in ranked_candidates],
        "candidate_search_pool": [int(x) for x in candidate_search_pool],
        "completed_boundaries": [],
        "accepted_boundaries": [],
        "discarded_boundaries": [],
        "generated_cases": 0,
        "discarded_cases": 0,
        "shortfall": max(0, int(requested_cases)),
        "current_boundary": None,
        "hef_full_policy": str(hef_full_policy),
        "run_mode": str(run_mode),
        "suite_full_hefs": {},
    }



def update_state(path: Path, state: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
    state = dict(state or {})
    state.update(fields)
    state["updated_at"] = now_iso()
    write_json_atomic(path, state)
    return state



def find_latest_resumable_set(parent_dir: Path, model_base: str) -> Optional[Path]:
    parent_dir = Path(parent_dir)
    prefix = f"{model_base}_benchmark_"
    candidates: list[tuple[float, Path]] = []
    try:
        for child in parent_dir.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith(prefix):
                continue
            state_path = child / "generation_state.json"
            if not state_path.exists():
                continue
            state = read_json(state_path, default={}) or {}
            if not isinstance(state, dict):
                continue
            if str(state.get("status") or "").strip().lower() in {"complete", "completed", "ok"}:
                continue
            try:
                mtime = float(state_path.stat().st_mtime)
            except Exception:
                mtime = 0.0
            candidates.append((mtime, child))
    except Exception:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
