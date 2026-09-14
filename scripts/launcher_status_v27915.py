#!/usr/bin/env python3
from __future__ import annotations

"""Atomic launcher-status writer and monitor for v2.79.15 campaigns."""

import argparse
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.workflow.launcher_status import (
    atomic_write_launcher_status,
    status_payload,
)

_WORKFLOW_START = re.compile(r"\[workflow\]\s+start\s+([A-Za-z0-9_.:-]+)")
_WORKFLOW_END = re.compile(r"\[workflow\]\s+(?:ok|failed)\s+([A-Za-z0-9_.:-]+)")
_NATIVE_STAGE = re.compile(r"\[native-progress\]\s+STAGE_START\s+([A-Za-z0-9_.:-]+)")
_PROGRESS = re.compile(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)\s+completed")

# Explicitly document the periodically refreshed launcher_status.txt contract.
# ``status_payload`` computes ``log_age_s`` from the current log mtime.
_PERIODIC_STATUS_FIELDS = (
    "phase", "worker_pid", "monitor_pid", "log_age_s",
    "progress_completed", "progress_total", "workflow_rc",
)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inspect_log(path: Path, *, max_bytes: int = 2 * 1024 * 1024) -> dict[str, Any]:
    if not path.is_file():
        return {"phase": "starting", "progress_completed": None, "progress_total": None, "detail": "log_not_created"}
    with path.open("rb") as handle:
        size = path.stat().st_size
        if size > max_bytes:
            handle.seek(size - max_bytes)
            handle.readline()
        text = handle.read().decode("utf-8", "replace")
    phase = "running"
    progress_done = None
    progress_total = None
    detail = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        detail = line[-600:]
        match = _WORKFLOW_START.search(line) or _NATIVE_STAGE.search(line)
        if match:
            phase = match.group(1)
        ended = _WORKFLOW_END.search(line)
        if ended and ended.group(1) == phase:
            phase = f"after_{phase}"
        progress = _PROGRESS.search(line)
        if progress:
            progress_done = int(progress.group("done"))
            progress_total = int(progress.group("total"))
    return {
        "phase": phase,
        "progress_completed": progress_done,
        "progress_total": progress_total,
        "detail": detail,
    }


def write_once(args: argparse.Namespace) -> int:
    log = Path(args.log).expanduser() if args.log else None
    observed = inspect_log(log) if log else {
        "phase": args.phase or "unknown",
        "progress_completed": args.progress_completed,
        "progress_total": args.progress_total,
        "detail": args.detail or "",
    }
    phase = args.phase or observed["phase"]
    progress_completed = (
        args.progress_completed if args.progress_completed is not None
        else observed["progress_completed"]
    )
    progress_total = (
        args.progress_total if args.progress_total is not None
        else observed["progress_total"]
    )
    detail = args.detail or observed["detail"]
    payload = status_payload(
        state=args.state,
        phase=phase,
        worker_pid=args.worker_pid,
        monitor_pid=args.monitor_pid,
        log_path=log,
        progress_completed=progress_completed,
        progress_total=progress_total,
        workflow_rc=args.workflow_rc,
        detail=detail,
    )
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if args.run_root:
        payload["run_root"] = str(Path(args.run_root).expanduser())
    atomic_write_launcher_status(Path(args.status), payload)
    return 0


def monitor(args: argparse.Namespace) -> int:
    status = Path(args.status).expanduser()
    log = Path(args.log).expanduser()
    interval = max(1.0, float(args.interval_s))
    while _pid_alive(int(args.worker_pid)):
        observed = inspect_log(log)
        payload = status_payload(
            state="RUNNING",
            phase=observed["phase"],
            worker_pid=int(args.worker_pid),
            monitor_pid=os.getpid(),
            log_path=log,
            progress_completed=observed["progress_completed"],
            progress_total=observed["progress_total"],
            workflow_rc=None,
            detail=observed["detail"],
        )
        payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if args.run_root:
            payload["run_root"] = str(Path(args.run_root).expanduser())
        atomic_write_launcher_status(status, payload)
        time.sleep(interval)
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--status", required=True)
    write.add_argument("--state", required=True)
    write.add_argument("--phase", default="")
    write.add_argument("--worker-pid", type=int)
    write.add_argument("--monitor-pid", type=int)
    write.add_argument("--log", default="")
    write.add_argument("--run-root", default="")
    write.add_argument("--progress-completed", type=int)
    write.add_argument("--progress-total", type=int)
    write.add_argument("--workflow-rc", type=int)
    write.add_argument("--detail", default="")
    write.set_defaults(func=write_once)

    mon = sub.add_parser("monitor")
    mon.add_argument("--status", required=True)
    mon.add_argument("--log", required=True)
    mon.add_argument("--worker-pid", required=True, type=int)
    mon.add_argument("--run-root", default="")
    mon.add_argument("--interval-s", type=float, default=30.0)
    mon.set_defaults(func=monitor)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
