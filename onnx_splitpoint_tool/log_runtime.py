"""Runtime helpers for active GUI log files.

These helpers keep the log discovery logic out of the UI modules and make the
currently active GUI log easier to find on disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

from .paths import ensure_dir, splitpoint_home, splitpoint_logs_dir

ACTIVE_LOG_ENV = "ONNX_SPLITPOINT_ACTIVE_LOG_PATH"
CWD_LOG_ENV = "ONNX_SPLITPOINT_CWD_LOG_PATH"
ACTIVE_LOG_PATH_TXT = "active_log_path.txt"
CURRENT_LOG_ALIAS = "current.log"


def _norm_path(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except Exception:
        return str(path)



def discover_live_logging_paths(*, logger_obj: Optional[logging.Logger] = None) -> List[Path]:
    """Return file paths currently attached to logging handlers."""

    seen: set[str] = set()
    out: List[Path] = []

    def _add(path_like: object) -> None:
        if not path_like:
            return
        try:
            path = Path(str(path_like)).expanduser()
        except Exception:
            return
        key = _norm_path(path)
        if key in seen:
            return
        seen.add(key)
        out.append(path)

    loggers: List[logging.Logger] = [logging.getLogger()]
    if logger_obj is not None:
        loggers.append(logger_obj)

    for lg in loggers:
        try:
            for handler in getattr(lg, "handlers", []):
                _add(getattr(handler, "baseFilename", None))
        except Exception:
            continue
    return out



def active_log_alias_candidates(*, include_cwd: bool = True) -> List[Path]:
    candidates = [
        splitpoint_logs_dir() / CURRENT_LOG_ALIAS,
        splitpoint_logs_dir() / ACTIVE_LOG_PATH_TXT,
        splitpoint_home() / ACTIVE_LOG_PATH_TXT,
    ]
    if include_cwd:
        candidates.extend([Path.cwd() / CURRENT_LOG_ALIAS, Path.cwd() / ACTIVE_LOG_PATH_TXT])
    return candidates



def resolve_active_log_path() -> Optional[Path]:
    """Best-effort resolution of the active GUI log file."""

    env_path = (os.environ.get(ACTIVE_LOG_ENV) or "").strip()
    if env_path:
        return Path(env_path).expanduser()

    for candidate in discover_live_logging_paths():
        if candidate.exists():
            return candidate

    for meta in active_log_alias_candidates(include_cwd=True):
        try:
            if meta.name == ACTIVE_LOG_PATH_TXT and meta.is_file():
                txt = meta.read_text(encoding="utf-8").strip()
                if txt:
                    return Path(txt).expanduser()
            if meta.name == CURRENT_LOG_ALIAS and meta.exists():
                return meta.resolve() if meta.is_symlink() else meta
        except Exception:
            continue

    defaults = [
        splitpoint_logs_dir() / "gui.log",
        splitpoint_home() / "gui.log",
        Path.cwd() / "gui.log",
    ]
    for candidate in defaults:
        if candidate.exists():
            return candidate
    return None



def _safe_unlink(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
    except Exception:
        pass



def _publish_alias(alias_path: Path, target_path: Path) -> None:
    """Try to create/update a symlink alias for the active log.

    On some platforms (or locked-down desktops) symlink creation can fail. This
    helper is intentionally best-effort.
    """

    try:
        alias_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    try:
        if alias_path.is_symlink():
            try:
                if _norm_path(alias_path.resolve()) == _norm_path(target_path):
                    return
            except Exception:
                pass
            _safe_unlink(alias_path)
        elif alias_path.exists():
            # Do not clobber a regular file that the user may inspect manually.
            return
        alias_path.symlink_to(target_path)
    except Exception:
        # Best-effort only. A text pointer is written separately.
        pass



def publish_active_log_metadata(log_path: Path, *, publish_cwd_alias: bool = False) -> None:
    """Publish metadata for the currently active GUI log.

    This writes:
      - the active log env var
      - ``~/.onnx_splitpoint_tool/logs/active_log_path.txt``
      - best-effort ``current.log`` symlink aliases
    """

    try:
        log_path = Path(log_path).expanduser()
    except Exception:
        return

    os.environ[ACTIVE_LOG_ENV] = str(log_path)

    meta_dirs = [ensure_dir(splitpoint_logs_dir()), ensure_dir(splitpoint_home())]
    for meta_dir in meta_dirs:
        try:
            (meta_dir / ACTIVE_LOG_PATH_TXT).write_text(str(log_path), encoding="utf-8")
        except Exception:
            pass
        _publish_alias(meta_dir / CURRENT_LOG_ALIAS, log_path)

    if publish_cwd_alias:
        try:
            cwd_meta = Path.cwd() / ACTIVE_LOG_PATH_TXT
            cwd_meta.write_text(str(log_path), encoding="utf-8")
        except Exception:
            pass
        _publish_alias(Path.cwd() / CURRENT_LOG_ALIAS, log_path)



def active_log_description() -> str:
    path = resolve_active_log_path()
    if path is None:
        return "Active log: unknown"
    return f"Active log: {path}"
