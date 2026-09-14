from __future__ import annotations

"""Atomic, periodically refreshable launcher status files."""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping


def atomic_write_launcher_status(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data.setdefault("updated_at_epoch_s", time.time())
    data.setdefault("schema", "onnx-splitpoint/launcher-status")
    data.setdefault("schema_version", 1)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for key in sorted(data):
                value = data[key]
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
                handle.write(f"{str(key).upper()}={value}\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
    return path


def status_payload(
    *, state: str, phase: str, worker_pid: int | None = None,
    monitor_pid: int | None = None, log_path: Path | None = None,
    progress_completed: int | None = None, progress_total: int | None = None,
    workflow_rc: int | None = None, detail: str = "",
) -> dict[str, Any]:
    now = time.time()
    log_age = None
    if log_path is not None and Path(log_path).is_file():
        log_age = max(0.0, now - Path(log_path).stat().st_mtime)
    return {
        "state": str(state), "phase": str(phase),
        "worker_pid": "" if worker_pid is None else int(worker_pid),
        "monitor_pid": "" if monitor_pid is None else int(monitor_pid),
        "log_path": "" if log_path is None else str(log_path),
        "log_age_s": "" if log_age is None else round(log_age, 3),
        "progress_completed": "" if progress_completed is None else int(progress_completed),
        "progress_total": "" if progress_total is None else int(progress_total),
        "workflow_rc": "" if workflow_rc is None else int(workflow_rc),
        "detail": str(detail),
        "updated_at_epoch_s": now,
    }
