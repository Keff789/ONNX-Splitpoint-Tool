"""Log retention / cleanup helpers.

The tool produces several logs:
  - gui.log (tool's own log)
  - 3rd party logs (e.g. Hailo SDK: hailo_sdk.*.log)

Without cleanup these can accumulate over time and clutter the user's disk.

Design goals
------------
- Safe by default: only deletes files inside explicit log roots.
- Best-effort: never crash the app if deletion fails (Windows file locks, etc.).
- Dependency-free: usable from both GUI and backend code paths.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class LogRetentionPolicy:
    """Retention policy for log cleanup."""

    enabled: bool = True
    # Delete logs older than this many days. Use None to disable age-based cleanup.
    max_age_days: Optional[int] = 30
    # Keep at most this many log files (newest kept). Use None to disable.
    max_files: Optional[int] = 300
    # File glob patterns to include.
    patterns: Tuple[str, ...] = ("*.log",)
    # File names to always keep.
    keep_names: Tuple[str, ...] = ("gui.log",)


def _parse_int_env(name: str, default: Optional[int]) -> Optional[int]:
    v = os.environ.get(name)
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    if s.lower() in {"none", "null", "off", "false"}:
        return None
    try:
        return int(s)
    except Exception:
        return default


def policy_from_env() -> LogRetentionPolicy:
    """Build a policy from environment variables.

    Supported env vars:
      - ONNX_SPLITPOINT_LOG_RETENTION: 0/1 (default: 1)
      - ONNX_SPLITPOINT_LOG_RETENTION_DAYS: int (default: 30)
      - ONNX_SPLITPOINT_LOG_RETENTION_MAX_FILES: int (default: 300)

    Notes:
      - Set *_DAYS or *_MAX_FILES to "none" to disable that constraint.
    """

    enabled_raw = os.environ.get("ONNX_SPLITPOINT_LOG_RETENTION", "1").strip().lower()
    enabled = enabled_raw not in {"0", "false", "off", "no"}
    days = _parse_int_env("ONNX_SPLITPOINT_LOG_RETENTION_DAYS", 30)
    max_files = _parse_int_env("ONNX_SPLITPOINT_LOG_RETENTION_MAX_FILES", 300)
    return LogRetentionPolicy(enabled=enabled, max_age_days=days, max_files=max_files)


def _iter_candidates(root: Path, patterns: Sequence[str], *, recursive: bool) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    try:
        if recursive:
            for pat in patterns:
                out.extend(root.rglob(pat))
        else:
            for pat in patterns:
                out.extend(root.glob(pat))
    except Exception:
        return []
    # Deduplicate (same file could match multiple patterns)
    uniq: Dict[str, Path] = {}
    for p in out:
        try:
            uniq[str(p.resolve())] = p
        except Exception:
            uniq[str(p)] = p
    return list(uniq.values())


def apply_log_retention(
    roots: Iterable[Path],
    *,
    policy: Optional[LogRetentionPolicy] = None,
    recursive: bool = True,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Apply log cleanup to the given roots.

    Returns a stats dict:
      {"removed": int, "kept": int, "freed_bytes": int, "errors": int}
    """

    pol = policy or policy_from_env()
    stats: Dict[str, object] = {
        "removed": 0,
        "kept": 0,
        "freed_bytes": 0,
        "errors": 0,
        "roots": [str(r) for r in roots],
        "policy": {
            "enabled": bool(pol.enabled),
            "max_age_days": pol.max_age_days,
            "max_files": pol.max_files,
            "patterns": list(pol.patterns),
            "keep_names": list(pol.keep_names),
        },
    }

    if not pol.enabled:
        return stats

    # Gather candidates.
    files: List[Path] = []
    for root in roots:
        files.extend(_iter_candidates(Path(root), pol.patterns, recursive=recursive))

    now = time.time()
    max_age_s: Optional[float] = None
    if pol.max_age_days is not None:
        try:
            max_age_s = float(max(0, int(pol.max_age_days))) * 86400.0
        except Exception:
            max_age_s = None

    # Compute metadata.
    meta: List[Tuple[float, int, Path]] = []  # (mtime, size, path)
    for p in files:
        try:
            st = p.stat()
            meta.append((float(st.st_mtime), int(st.st_size), p))
        except Exception:
            # If we can't stat, leave it alone.
            continue

    # Sort newest first.
    meta.sort(key=lambda t: t[0], reverse=True)

    keep_set: set[str] = set()
    for _mtime, _sz, p in meta:
        if p.name in pol.keep_names:
            try:
                keep_set.add(str(p.resolve()))
            except Exception:
                keep_set.add(str(p))

    # Determine deletions.
    to_delete: List[Tuple[float, int, Path]] = []
    kept: List[Tuple[float, int, Path]] = []

    for mtime, sz, p in meta:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)

        if rp in keep_set:
            kept.append((mtime, sz, p))
            continue

        # Age-based deletion
        if max_age_s is not None and (now - mtime) > max_age_s:
            to_delete.append((mtime, sz, p))
        else:
            kept.append((mtime, sz, p))

    # Count-based deletion (oldest first)
    if pol.max_files is not None:
        try:
            max_files = int(pol.max_files)
            if max_files >= 0:
                # kept is still sorted newest->oldest because meta was.
                # Move oldest beyond max_files into deletion set.
                if len(kept) > max_files:
                    extra = kept[max_files:]
                    kept = kept[:max_files]
                    to_delete.extend(extra)
        except Exception:
            pass

    # Execute deletions.
    removed = 0
    freed = 0
    errors = 0
    for _mtime, sz, p in to_delete:
        try:
            if dry_run:
                removed += 1
                freed += max(0, int(sz))
                continue
            p.unlink(missing_ok=True)  # type: ignore[call-arg]
            removed += 1
            freed += max(0, int(sz))
        except Exception:
            errors += 1

    stats["removed"] = removed
    stats["kept"] = len(kept)
    stats["freed_bytes"] = freed
    stats["errors"] = errors
    return stats
