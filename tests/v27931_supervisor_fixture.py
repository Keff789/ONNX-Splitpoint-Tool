"""Readiness-controlled test harness around the unchanged production supervisor.

Only this stdlib fixture prestarts its process. The real worker's 180 s budget,
descendant tracking, signals and reaping are never patched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--never-ready", action="store_true")
    parser.add_argument("--startup-delay", type=float, default=1.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import deepx_full_workflow_smoke_worker_v27930 as worker
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    pidfile, ready = directory / "grandchild.pid", directory / "ready.json"
    grandchild = directory / "grandchild.py"
    grandchild.write_text(
        "import json,os,signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        f"Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
        f"time.sleep({args.startup_delay!r})\n"
        + ("" if args.never_ready else
           f"Path({str(ready)!r}).write_text(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),term_ignored=True)))\n")
        + "while True: time.sleep(.05)\n", encoding="utf-8")
    parent = directory / "parent.py"
    parent.write_text(
        "import subprocess,sys,time\n"
        f"subprocess.Popen([sys.executable,'-I','-S','-B',{str(grandchild)!r}],start_new_session=True)\n"
        "while True: time.sleep(.05)\n", encoding="utf-8")
    command = [sys.executable, "-I", "-S", "-B", str(parent)]
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=directory, start_new_session=True)
    report = None
    alive = False
    readiness = None
    signals = []
    try:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not ready.is_file():
            if proc.poll() is not None:
                raise RuntimeError("fixture_parent_exited_before_readiness")
            time.sleep(.02)
        if ready.is_file():
            readiness = json.loads(ready.read_text())
            os.kill(readiness["pid"], 0)
            alive = readiness["pid"] == readiness["sid"] and readiness["term_ignored"] is True
            if not alive:
                raise RuntimeError("fixture_readiness_contract_invalid")
        elif not args.never_ready:
            raise RuntimeError("fixture_readiness_timeout")
    finally:
        # Start the tested supervisor clock only after bounded readiness. Its
        # Popen boundary receives the actual, already running owned process.
        original_popen, original_signal = worker.subprocess.Popen, worker._signal_owned
        calls = []
        def adopted_popen(argv, **kwargs):
            assert argv == command and kwargs["start_new_session"] is True
            assert not calls
            calls.append(True)
            return proc
        def observe_signal(tracked, sig):
            signals.append({"signal": int(sig), "pids": list(tracked)})
            return original_signal(tracked, sig)
        worker.subprocess.Popen, worker._signal_owned = adopted_popen, observe_signal
        try:
            report = worker.supervised_run(command, cwd=directory,
                log_path=directory / "child.log", timeout=.3, grace=.2)
        finally:
            worker.subprocess.Popen, worker._signal_owned = original_popen, original_signal
        pid = int(pidfile.read_text()) if pidfile.is_file() else None
        absent = False
        if pid is not None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                absent = True
        report.update(readiness_status="ready" if readiness else "timeout",
            grandchild_seen_alive=alive, readiness=readiness, grandchild_pid=pid,
            grandchild_reaped=absent, signal_trace=signals,
            total_elapsed_s=time.monotonic() - started)
        (directory / "supervision.json").write_text(json.dumps(report), encoding="utf-8")
    return 0 if readiness else 2


if __name__ == "__main__":
    raise SystemExit(main())
