from __future__ import annotations

"""Live progress journal and streaming subprocess helpers for Native stages."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Callable, Mapping, Sequence
import json
import os
import subprocess
import time


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NativeProgressJournal:
    def __init__(self, jsonl_path: str | Path, state_path: str | Path | None = None):
        self.jsonl_path = Path(jsonl_path)
        self.state_path = Path(state_path) if state_path else self.jsonl_path.with_suffix('.json')
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dir(cls, directory: str | Path) -> "NativeProgressJournal":
        d=Path(directory); d.mkdir(parents=True, exist_ok=True)
        return cls(d/'native_progress.jsonl', d/'native_progress.json')

    @classmethod
    def from_env(cls) -> "NativeProgressJournal | None":
        d=os.environ.get('ONNX_SPLITPOINT_NATIVE_PROGRESS_DIR','').strip()
        return cls.from_dir(d) if d else None

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        row={'timestamp':_now(),'event':str(event),**fields}
        line=json.dumps(row, ensure_ascii=False, sort_keys=True)
        # O_APPEND keeps each short record atomic on POSIX for cooperating processes.
        fd=os.open(self.jsonl_path, os.O_WRONLY|os.O_CREAT|os.O_APPEND, 0o644)
        try:
            os.write(fd,(line+'\n').encode('utf-8'))
        finally:
            os.close(fd)
        # Use a process-specific temporary path: several nested Native helpers
        # may publish progress concurrently into the same journal.
        tmp=self.state_path.with_name(self.state_path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
        tmp.write_text(json.dumps(row,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        os.replace(tmp,self.state_path)
        return row


@dataclass
class StreamingCompletedProcess:
    args: Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float


def run_streaming(
    cmd: Sequence[str], *, timeout: float | None = None, cwd: str | Path | None = None,
    env: Mapping[str,str] | None = None, label: str = 'native', heartbeat_s: float = 30.0,
    journal: NativeProgressJournal | None = None,
    line_callback: Callable[[str],None] | None = None,
) -> StreamingCompletedProcess:
    """Run a process while forwarding every output line and emitting heartbeats."""
    merged_env=dict(os.environ)
    if env: merged_env.update({str(k):str(v) for k,v in env.items()})
    merged_env.setdefault('PYTHONUNBUFFERED','1')
    start=time.time(); q: Queue[str|None]=Queue(); tail=deque(maxlen=800)
    if journal: journal.emit('START', label=label, cmd=list(map(str,cmd)))
    proc=subprocess.Popen(list(map(str,cmd)), cwd=str(cwd) if cwd else None, env=merged_env,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          bufsize=1, universal_newlines=True)
    def reader():
        assert proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline,''):
                q.put(line)
        finally:
            q.put(None)
    Thread(target=reader,daemon=True).start()
    last_output=start; last_heartbeat=start; eof=False; timed_out=False
    while True:
        now=time.time()
        try:
            item=q.get(timeout=0.25)
            if item is None:
                eof=True
            else:
                clean=item.rstrip('\n')
                tail.append(clean); last_output=now
                if line_callback: line_callback(clean)
                else: print(clean,flush=True)
        except Empty:
            pass
        if heartbeat_s>0 and now-last_heartbeat>=heartbeat_s and proc.poll() is None:
            msg=f'[native-progress] HEARTBEAT label={label} elapsed={now-start:.1f}s silence={now-last_output:.1f}s'
            if line_callback: line_callback(msg)
            else: print(msg,flush=True)
            if journal: journal.emit('HEARTBEAT',label=label,elapsed_s=now-start,silence_s=now-last_output,pid=proc.pid)
            last_heartbeat=now
        if timeout is not None and now-start>float(timeout) and proc.poll() is None:
            timed_out=True; proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill()
        if proc.poll() is not None and eof and q.empty(): break
    rc=proc.wait(); elapsed=time.time()-start
    output='\n'.join(tail)
    if timed_out and rc==0: rc=124
    if timed_out: output += f'\nTimeoutExpired after {timeout}s'
    if journal: journal.emit('END',label=label,returncode=rc,elapsed_s=elapsed,timed_out=timed_out)
    return StreamingCompletedProcess(list(map(str,cmd)),rc,output,'',elapsed)


def stream_command(
    cmd: Sequence[str], *, timeout: float | None = None, cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None, label: str = "native", heartbeat_s: float = 15.0,
    progress_jsonl: str | Path | None = None, progress_json: str | Path | None = None,
    line_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper returning the dict contract used by helper scripts.

    Unlike :func:`subprocess.run`, output is forwarded immediately and a parent
    heartbeat is emitted even when the child is completely silent.  This is the
    common execution primitive for Native split, Native Full, validation and
    energy subprocesses.
    """
    journal = None
    if progress_jsonl:
        journal = NativeProgressJournal(progress_jsonl, progress_json)
    elif progress_json:
        pp = Path(progress_json)
        journal = NativeProgressJournal(pp.with_suffix('.jsonl'), pp)
    else:
        journal = NativeProgressJournal.from_env()
    completed = run_streaming(
        cmd, timeout=timeout, cwd=cwd, env=env, label=label,
        heartbeat_s=heartbeat_s, journal=journal, line_callback=line_callback,
    )
    return {
        "cmd": list(map(str, completed.args)),
        "rc": int(completed.returncode),
        "returncode": int(completed.returncode),
        "elapsed_s": float(completed.elapsed_s),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
        "timed_out": int(completed.returncode) == 124,
    }
