"""Exact remote-process ownership and cancellation for workflow SSH jobs.

Killing a local ``ssh`` process does not prove that its remote workload was
stopped.  This module gives every workflow-owned remote command a unique
lease.  The lease binds the workflow run/session and operation token to the
remote process' PID, process group and Linux ``/proc`` start time.  Cleanup is
performed through a second SSH control command which validates that identity
before signalling exactly the recorded process tree.

The feature is opt-in.  Standalone/non-workflow SSHTransport users retain the
historic behaviour when no :class:`RemoteProcessLeaseScope` is supplied.
"""

from __future__ import annotations

import hashlib
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFE_OPERATION_RE = re.compile(r"^[A-Za-z0-9_.-]{1,160}$")

REMOTE_LEASE_ENV_RUN_ID = "ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID"
REMOTE_LEASE_ENV_SESSION_ID = "ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID"
REMOTE_LEASE_ENV_JOURNAL_DIR = "ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR"
REMOTE_LEASE_ENV_REMOTE_ROOT = "ONNX_SPLITPOINT_REMOTE_LEASE_REMOTE_ROOT"
REMOTE_LEASE_JOURNAL_SCHEMA = "onnx-splitpoint/remote-process-lease-journal"
REMOTE_LEASE_JOURNAL_SCHEMA_VERSION = 1
REMOTE_LEASE_UNPROVEN_RC = 70
# The embedded cleanup budget is 15 seconds with default settings.  Outer
# owners reserve two additional seconds so a valid cleanup that reaches its
# own bound is not killed by the surrounding timeout at the same instant.
REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S = 17.0

_ACTIVE_REMOTE_PROCESS_REGISTRY: ContextVar[Any] = ContextVar(
    "onnx_splitpoint_active_remote_process_registry",
    default=None,
)


@contextmanager
def bind_remote_process_registry(registry: Any):
    """Bind cross-process SSH ownership to synchronous nested helpers."""

    token = _ACTIVE_REMOTE_PROCESS_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _ACTIVE_REMOTE_PROCESS_REGISTRY.reset(token)


def current_remote_process_registry() -> Any:
    return _ACTIVE_REMOTE_PROCESS_REGISTRY.get()


def remote_process_lease_cleanup_timeout_s(
    *,
    grace_s: float = 3.0,
    launch_wait_s: float = 1.0,
) -> float:
    """Bound the embedded cleanup phases plus local SSH-control overhead."""

    grace = max(0.0, float(grace_s))
    launch_wait = max(0.0, float(launch_wait_s))
    kill_phase = max(1.0, min(5.0, grace + 1.0))
    guardian_phase = max(1.0, min(5.0, grace + 1.0))
    final_guardian_phase = 1.0
    ssh_control_overhead = 2.0
    return max(
        5.0,
        launch_wait
        + grace
        + kill_phase
        + guardian_phase
        + final_guardian_phase
        + ssh_control_overhead,
    )


def _remote_process_lease_cancel_join_budget_s(*, grace_s: float) -> float:
    """Bound a parallel registry cleanup pass with control-path slack."""

    return remote_process_lease_cleanup_timeout_s(grace_s=grace_s) + 2.0


class RemoteProcessLeaseConfigurationError(ValueError):
    """An SSH option would break exact leased-process ownership."""


_REMOTE_LEASE_SAFE_SSH_OPTIONS = (
    "-o",
    "ForkAfterAuthentication=no",
    "-o",
    "SessionType=default",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPersist=no",
)


def validate_remote_process_lease_ssh_prefix(argv: Sequence[str]) -> None:
    """Reject SSH lifecycle modes incompatible with an exact remote lease.

    This validation is deliberately limited to workload-executing SSH.  SCP
    and rsync transfer paths are outside this lease contract.
    """

    values = [str(value) for value in argv]
    if not values or os.path.basename(values[0]).lower() not in {"ssh", "ssh.exe"}:
        raise RemoteProcessLeaseConfigurationError(
            "remote process leasing requires an ssh control prefix"
        )

    options: List[str] = []
    index = 1
    while index < len(values):
        token = values[index]
        if token == "-o":
            if index + 1 >= len(values):
                raise RemoteProcessLeaseConfigurationError(
                    "leased SSH has -o without an option value"
                )
            options.append(values[index + 1])
            index += 2
            continue
        if token.startswith("-o") and len(token) > 2:
            options.append(token[2:])
            index += 1
            continue
        if token.startswith("-") and token != "-":
            short = token[1:]
            forbidden_short = [flag for flag in ("f", "N", "M", "O") if flag in short]
            if forbidden_short:
                raise RemoteProcessLeaseConfigurationError(
                    "leased SSH forbids lifecycle-changing option(s): -"
                    + " -".join(forbidden_short)
                )
        index += 1

    for raw in options:
        key, separator, raw_value = str(raw).partition("=")
        if not separator:
            pieces = key.split(None, 1)
            key = pieces[0] if pieces else ""
            if len(pieces) == 2:
                raw_value = pieces[1]
                separator = " "
        normalized_key = key.strip().lower()
        value = raw_value.strip().lower() if separator else "yes"
        if normalized_key == "forkafterauthentication" and value not in {
            "no",
            "false",
        }:
            raise RemoteProcessLeaseConfigurationError(
                "leased SSH forbids ForkAfterAuthentication"
            )
        if normalized_key == "sessiontype" and value != "default":
            raise RemoteProcessLeaseConfigurationError(
                "leased SSH requires SessionType=default"
            )
        if normalized_key == "controlpersist" and value not in {
            "no",
            "false",
            "0",
            "0s",
        }:
            raise RemoteProcessLeaseConfigurationError(
                "leased SSH forbids persistent control masters"
            )
        if normalized_key == "controlmaster" and value not in {"no", "false"}:
            raise RemoteProcessLeaseConfigurationError(
                "leased SSH forbids ControlMaster modes"
            )


def harden_remote_process_lease_ssh_argv(argv: Sequence[str]) -> List[str]:
    """Force foreground, non-multiplexed SSH semantics for leased work.

    Explicit command-line values take precedence over user and system
    ``ssh_config``.  Validation still rejects a conflicting value supplied by
    the caller instead of silently changing that caller's requested mode.
    """

    values = [str(value) for value in argv]
    validate_remote_process_lease_ssh_prefix(values)
    safe = list(_REMOTE_LEASE_SAFE_SSH_OPTIONS)
    if values[1 : 1 + len(safe)] == safe:
        return values
    return [values[0], *safe, *values[1:]]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _safe_label(value: str, fallback: str = "remote") -> str:
    label = _SAFE_LABEL_RE.sub("_", str(value or "")).strip("._-")
    return (label or fallback)[:48]


# The leased root is a long-lived Linux child-subreaper, not the payload shell
# itself.  Therefore a payload that daemonises/double-forks remains an exact
# child of this guardian instead of being reparented outside the lease.  A
# nominal SSH return is possible only after the payload and every adopted
# descendant have exited.
_REMOTE_GUARDIAN_PY = r'''
import ctypes, os, pathlib, signal, subprocess, sys, time

gate_raw, cancel_raw, drained_raw, ready_raw, launcher_raw, payload, token = sys.argv[1:]
gate = pathlib.Path(gate_raw)
cancel = pathlib.Path(cancel_raw)
drained = pathlib.Path(drained_raw)
ready = pathlib.Path(ready_raw)
launcher_pid = int(launcher_raw)

try:
    proc_self_pid = int(pathlib.Path("/proc/self/stat").read_text().split("(", 1)[0].strip())
except (OSError, ValueError):
    proc_self_pid = -1
if proc_self_pid != os.getpid():
    print("__SPLITPOINT_REMOTE_GUARDIAN__=unsafe_procfs_namespace", flush=True)
    raise SystemExit(69)

try:
    libc = ctypes.CDLL(None, use_errno=True)
    if int(libc.prctl(36, 1, 0, 0, 0)) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER) failed")
except Exception as exc:
    print("__SPLITPOINT_REMOTE_GUARDIAN__=subreaper_unavailable:%s" % type(exc).__name__, flush=True)
    raise SystemExit(69)

def exact_cancel_requested():
    try:
        return cancel.read_text(encoding="ascii").strip() == token
    except OSError:
        return False

def publish_drained():
    temporary = drained.with_name(drained.name + ".%d.tmp" % os.getpid())
    with temporary.open("w", encoding="ascii") as handle:
        handle.write(token + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(drained))

stop_requested = False
def request_stop(_signum, _frame):
    global stop_requested
    stop_requested = True
for signum in (signal.SIGINT, signal.SIGTERM):
    signal.signal(signum, request_stop)

# The launcher must not publish a cancellable PID until the subreaper and its
# non-terminating TERM handler are both installed.
ready.touch(exist_ok=True)

deadline = time.monotonic() + 10.0
while not gate.exists():
    if exact_cancel_requested() or stop_requested:
        publish_drained()
        raise SystemExit(130)
    try:
        os.kill(launcher_pid, 0)
    except OSError:
        publish_drained()
        raise SystemExit(130)
    if time.monotonic() >= deadline:
        publish_drained()
        raise SystemExit(130)
    time.sleep(0.02)

payload_proc = subprocess.Popen(["bash", "-lc", payload])
payload_rc = None

def direct_children():
    children = set()
    task_root = pathlib.Path("/proc/%d/task" % os.getpid())
    try:
        tasks = list(task_root.iterdir())
    except OSError:
        tasks = []
    for task in tasks:
        try:
            text = (task / "children").read_text()
        except OSError:
            continue
        children.update(int(value) for value in text.split() if value.isdigit())
    return children

while True:
    if payload_rc is None:
        payload_rc = payload_proc.poll()
    if payload_rc is not None:
        # Reap adopted descendants after Popen has reaped its own direct child.
        while True:
            try:
                adopted_pid, _status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if adopted_pid == 0:
                break
        if not direct_children():
            break
    time.sleep(0.02)

publish_drained()
raise SystemExit(130 if stop_requested else int(payload_rc or 0))
'''.strip()


# This launcher deliberately places only a tiny gate in the new session before
# publishing the lease.  The actual payload cannot start until the atomic lease
# exists.  If the SSH-side launcher dies in the publication window, the gate
# notices that its exact parent PID disappeared and exits rather than leaking.
_REMOTE_LAUNCH_PY = r'''
import json, os, pathlib, subprocess, sys, time

run_hash, session_hash, operation_id, token, root_arg, guardian_script, payload = sys.argv[1:]
# The launcher and independent cleanup use different SSH/login-shell
# invocations.  XDG_RUNTIME_DIR/TMPDIR are therefore not a stable shared
# identity.  The uid-scoped /tmp root is deterministic across both sessions;
# callers may still supply an explicit root.
base = pathlib.Path(root_arg) if root_arg else pathlib.Path("/tmp") / (
    "onnx_splitpoint-process-leases-%d" % os.getuid()
)
root = base / run_hash / session_hash
root.mkdir(mode=0o700, parents=True, exist_ok=True)
try:
    root.chmod(0o700)
except OSError:
    pass
lease = root / (operation_id + ".lease.json")
cancel = root / (operation_id + ".cancel")
gate = root / (operation_id + ".go")
drained = root / (operation_id + ".drained")
ready = root / (operation_id + ".guardian-ready")
def exact_cancel_requested():
    try:
        return cancel.read_text(encoding="ascii").strip() == token
    except OSError:
        return False
if exact_cancel_requested():
    print("__SPLITPOINT_REMOTE_LEASE__=cancelled_before_start", flush=True)
    raise SystemExit(130)
try:
    proc_self_pid = int(pathlib.Path("/proc/self/stat").read_text().split("(", 1)[0].strip())
except (OSError, ValueError):
    proc_self_pid = -1
if proc_self_pid != os.getpid():
    print("__SPLITPOINT_REMOTE_LEASE__=unsafe_procfs_namespace", flush=True)
    raise SystemExit(69)

parent_pid = os.getpid()
child = subprocess.Popen(
    [sys.executable, "-c", guardian_script, str(gate), str(cancel),
     str(drained), str(ready), str(parent_pid), payload, token],
    start_new_session=True,
)

def identity(pid):
    text = pathlib.Path("/proc/%d/stat" % pid).read_text()
    close = text.rfind(")")
    fields = text[close + 2:].split()
    return int(fields[2]), int(fields[19])  # pgrp (field 5), starttime (22)

try:
    ready_deadline = time.monotonic() + 10.0
    while not ready.exists() and time.monotonic() < ready_deadline:
        if child.poll() is not None:
            raise RuntimeError("remote lease guardian exited before readiness")
        time.sleep(0.02)
    if not ready.exists():
        raise RuntimeError("remote lease guardian readiness timed out")
    pgid, start_time = identity(child.pid)
    if pgid != child.pid:
        raise RuntimeError("leased child is not its process-group leader")
    record = {
        "schema": "onnx-splitpoint/remote-process-lease",
        "schema_version": 1,
        "run_sha256": run_hash,
        "session_sha256": session_hash,
        "operation_id": operation_id,
        "token": token,
        "pid": child.pid,
        "pgid": pgid,
        "start_time_ticks": start_time,
    }
    temporary = lease.with_name(lease.name + ".%d.tmp" % parent_pid)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(lease))
    if exact_cancel_requested():
        # The exact control command owns termination.  Do not release the gate.
        print("__SPLITPOINT_REMOTE_LEASE__=cancelled_after_publish", flush=True)
    else:
        gate.touch(exist_ok=True)
    return_code = child.wait()
finally:
    try:
        current = json.loads(lease.read_text(encoding="utf-8"))
        # The guardian's fsynced token proves every adopted descendant exited.
        # A concurrent cleanup controller may already have read this lease;
        # retain the terminal token for every concurrent cleanup controller.
        # Before proof, rc=70 retains the retryable identity record.
        drained_ok = False
        try:
            drained_ok = drained.read_text(encoding="ascii").strip() == token
        except OSError:
            pass
        if current.get("token") == token and drained_ok:
            lease.unlink()
    except (OSError, ValueError):
        pass
    try:
        gate.unlink()
    except OSError:
        pass
    try:
        ready.unlink()
    except OSError:
        pass
raise SystemExit(return_code)
'''.strip()


# argv: run hash, session hash, operation id, token, root override,
# TERM grace seconds, launch-race wait seconds.
_REMOTE_CLEANUP_PY = r'''
import json, os, pathlib, signal, sys, time

run_hash, session_hash, operation_id, token, root_arg, grace_raw, launch_raw = sys.argv[1:]
grace = max(0.0, float(grace_raw))
launch_wait = max(0.0, float(launch_raw))
base = pathlib.Path(root_arg) if root_arg else pathlib.Path("/tmp") / (
    "onnx_splitpoint-process-leases-%d" % os.getuid()
)
root = base / run_hash / session_hash
root.mkdir(mode=0o700, parents=True, exist_ok=True)
try:
    root.chmod(0o700)
except OSError:
    pass
lease = root / (operation_id + ".lease.json")
cancel = root / (operation_id + ".cancel")
drained = root / (operation_id + ".drained")
temporary = cancel.with_name(cancel.name + ".%d.tmp" % os.getpid())
temporary.write_text(token + "\n", encoding="ascii")
os.replace(str(temporary), str(cancel))
try:
    proc_self_pid = int(pathlib.Path("/proc/self/stat").read_text().split("(", 1)[0].strip())
except (OSError, ValueError):
    proc_self_pid = -1
if proc_self_pid != os.getpid():
    if not lease.exists():
        print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=cancelled_before_lease_unsafe_procfs")
        raise SystemExit(0)
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=unsafe_procfs_namespace")
    raise SystemExit(69)

deadline = time.monotonic() + launch_wait
while not lease.exists() and time.monotonic() < deadline:
    time.sleep(0.02)
if not lease.exists():
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=cancelled_before_lease")
    raise SystemExit(0)
try:
    record = json.loads(lease.read_text(encoding="utf-8"))
except FileNotFoundError:
    # A sibling exact controller/launcher may retire the lease after our
    # existence check. Only the matching guardian proof closes that race.
    try:
        proven = drained.read_text(encoding="ascii").strip() == token
    except OSError:
        proven = False
    if proven:
        print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=already_exited_drained")
        raise SystemExit(0)
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=lease_disappeared_without_drain")
    raise SystemExit(70)
except Exception as exc:
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=invalid_lease:%s" % type(exc).__name__)
    raise SystemExit(65)

expected = {
    "run_sha256": run_hash,
    "session_sha256": session_hash,
    "operation_id": operation_id,
    "token": token,
}
if any(str(record.get(key, "")) != value for key, value in expected.items()):
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=identity_mismatch")
    raise SystemExit(66)
try:
    root_pid = int(record["pid"])
    root_pgid = int(record["pgid"])
    root_start = int(record["start_time_ticks"])
except Exception:
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=invalid_numeric_identity")
    raise SystemExit(65)
if root_pid <= 1 or root_pgid != root_pid:
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=unsafe_process_group")
    raise SystemExit(65)

def identity(pid):
    try:
        text = pathlib.Path("/proc/%d/stat" % pid).read_text()
        close = text.rfind(")")
        fields = text[close + 2:].split()
        return {
            "pid": pid, "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]),
            "start": int(fields[19]),
        }
    except (OSError, ValueError, IndexError):
        return None

root_identity = identity(root_pid)
if root_identity is None or root_identity["state"] == "Z":
    try:
        drained_ok = drained.read_text(encoding="ascii").strip() == token
    except OSError:
        drained_ok = False
    if not drained_ok:
        print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=root_exited_without_drain")
        raise SystemExit(70)
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=already_exited_drained")
    try:
        if json.loads(lease.read_text()).get("token") == token:
            lease.unlink()
    except Exception:
        pass
    raise SystemExit(0)
if root_identity["start"] != root_start or root_identity["pgid"] != root_pgid:
    print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=pid_reused_or_pgid_changed")
    raise SystemExit(66)

owned = {root_pid: root_identity}

def exact_alive(item):
    current = identity(item["pid"])
    return (
        current is not None
        and current["state"] != "Z"
        and current["start"] == item["start"]
    )

def capture_descendants():
    # Traverse only Linux' exact per-parent children lists.  No global /proc
    # scan, process-name lookup or broad run-id match participates in
    # ownership discovery.
    changed = True
    while changed:
        changed = False
        for parent_pid, parent_item in list(owned.items()):
            if not exact_alive(parent_item):
                continue
            task_root = pathlib.Path("/proc/%d/task" % parent_pid)
            try:
                task_dirs = list(task_root.iterdir())
            except OSError:
                task_dirs = []
            child_pids = set()
            for task_dir in task_dirs:
                try:
                    text = (task_dir / "children").read_text()
                except OSError:
                    continue
                child_pids.update(
                    int(value) for value in text.split() if value.isdigit()
                )
            for pid in child_pids:
                if pid in owned:
                    # PID reuse during fork churn must not pin a stale start
                    # time forever.  Replacement is safe only because the new
                    # identity is again an exact child of this still-owned
                    # parent.
                    previous = owned[pid]
                    current = identity(pid)
                    if (not exact_alive(previous) and current is not None
                            and current["ppid"] == parent_pid):
                        owned[pid] = current
                        changed = True
                    continue
                item = identity(pid)
                if item is not None and item["ppid"] == parent_pid:
                    owned[pid] = item
                    changed = True

def signal_term_all():
    capture_descendants()
    # A process group is signalled only when its exact leader is in the owned
    # descendant set and still has the captured start time.  Individual exact
    # PIDs cover descendants that deliberately created a nested session.
    groups = []
    for pid, item in list(owned.items()):
        if item["pgid"] == pid and exact_alive(item):
            groups.append(pid)
    for pgid in sorted(set(groups), reverse=True):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    for pid, item in sorted(owned.items(), reverse=True):
        if not exact_alive(item):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

def signal_descendants(sig):
    # The subreaper guardian is the ownership barrier.  Never include its
    # process group in the KILL fan-out: payloads in that group are addressed
    # by exact PID while the guardian remains alive to adopt and expose churn.
    capture_descendants()
    groups = []
    for pid, item in list(owned.items()):
        if pid != root_pid and item["pgid"] == pid and exact_alive(item):
            groups.append(pid)
    for pgid in sorted(set(groups), reverse=True):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass
    for pid, item in sorted(owned.items(), reverse=True):
        if pid == root_pid or not exact_alive(item):
            continue
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass

def alive_count():
    capture_descendants()
    return sum(1 for item in owned.values() if exact_alive(item))

def alive_descendant_count():
    capture_descendants()
    return sum(
        1 for pid, item in owned.items()
        if pid != root_pid and exact_alive(item)
    )

def drain_proven():
    try:
        return drained.read_text(encoding="ascii").strip() == token
    except OSError:
        return False

signal_term_all()
term_deadline = time.monotonic() + grace
while alive_count() and time.monotonic() < term_deadline:
    # Re-signal exact late descendants without widening ownership.
    signal_term_all()
    time.sleep(0.02)

# KILL descendants only.  Repeated capture through the living subreaper sees
# children forked from TERM handlers and children adopted after their original
# parent exits.  Killing the guardian in this phase would destroy that proof.
kill_deadline = time.monotonic() + max(1.0, min(5.0, grace + 1.0))
while alive_descendant_count() and time.monotonic() < kill_deadline:
    signal_descendants(signal.SIGKILL)
    time.sleep(0.02)

# With no observed descendants, allow the guardian to reap, publish its
# drained token, and exit naturally.  If publication wins the race but the
# guardian has not returned yet, it may finally be killed as a separate exact
# PID; otherwise it remains alive and cleanup fails closed for a later retry.
guardian_deadline = time.monotonic() + max(1.0, min(5.0, grace + 1.0))
while exact_alive(owned[root_pid]) and time.monotonic() < guardian_deadline:
    if alive_descendant_count():
        signal_descendants(signal.SIGKILL)
    # Even after the drained token appears, prefer the guardian's natural
    # exit.  Exact SIGKILL is only the last action after this bounded wait.
    time.sleep(0.02)
if exact_alive(owned[root_pid]) and drain_proven():
    try:
        os.kill(root_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass
    final_deadline = time.monotonic() + 1.0
    while exact_alive(owned[root_pid]) and time.monotonic() < final_deadline:
        time.sleep(0.02)

remaining = alive_count()
drained_ok = drain_proven()
if remaining == 0 and drained_ok:
    try:
        if json.loads(lease.read_text()).get("token") == token:
            lease.unlink()
    except Exception:
        pass
success = remaining == 0 and drained_ok
print("__SPLITPOINT_REMOTE_LEASE_CLEANUP__=%s owned=%d remaining=%d drained=%d" %
      ("terminated" if success else "survivors", len(owned), remaining,
       1 if drained_ok else 0))
raise SystemExit(0 if success else 70)
'''.strip()


@dataclass(frozen=True)
class RemoteProcessLeaseScope:
    """Run/session identity used to mint unique remote operation leases."""

    run_id: str
    session_id: str
    lease_root: str = ""

    def __post_init__(self) -> None:
        if not str(self.run_id).strip():
            raise ValueError("remote process lease requires a non-empty run_id")
        if not str(self.session_id).strip():
            raise ValueError("remote process lease requires a non-empty session_id")

    @property
    def run_sha256(self) -> str:
        return _sha256_text(self.run_id)

    @property
    def session_sha256(self) -> str:
        return _sha256_text(self.session_id)

    def operation(
        self,
        *,
        label: str,
        control_argv_prefix: Sequence[str],
    ) -> "RemoteProcessLeaseOperation":
        operation_id = f"{_safe_label(label)}-{uuid.uuid4().hex}"
        return RemoteProcessLeaseOperation(
            scope=self,
            operation_id=operation_id,
            token=uuid.uuid4().hex + uuid.uuid4().hex,
            control_argv_prefix=tuple(str(value) for value in control_argv_prefix),
        )


class RemoteProcessLeaseOperation:
    """One exact remote command and the independent control path that owns it."""

    def __init__(
        self,
        *,
        scope: RemoteProcessLeaseScope,
        operation_id: str,
        token: str,
        control_argv_prefix: Sequence[str],
    ) -> None:
        self.scope = scope
        self.operation_id = str(operation_id)
        self.token = str(token)
        prefix = [str(value) for value in control_argv_prefix]
        if prefix and os.path.basename(prefix[0]).lower() in {"ssh", "ssh.exe"}:
            prefix = harden_remote_process_lease_ssh_argv(prefix)
        self.control_argv_prefix = tuple(prefix)
        if _SAFE_OPERATION_RE.fullmatch(self.operation_id) is None:
            raise ValueError("invalid remote lease operation_id")
        if not self.token or any(character.isspace() for character in self.token):
            raise ValueError("invalid remote lease token")
        if not self.control_argv_prefix:
            raise ValueError("remote lease control argv prefix must not be empty")
        self._cancel_lock = threading.Lock()
        self._cancel_result: Optional[Dict[str, Any]] = None

    def _identity_argv(self) -> List[str]:
        return [
            self.scope.run_sha256,
            self.scope.session_sha256,
            self.operation_id,
            self.token,
            str(self.scope.lease_root or ""),
        ]

    def wrap_remote_command(self, payload: str) -> str:
        argv = ["python3", "-c", _REMOTE_LAUNCH_PY]
        argv.extend(self._identity_argv())
        argv.append(_REMOTE_GUARDIAN_PY)
        argv.append(str(payload))
        return " ".join(shlex.quote(value) for value in argv)

    def wrap_ssh_argv(self, command: Sequence[str]) -> List[str]:
        values = harden_remote_process_lease_ssh_argv(command)
        if not values or os.path.basename(values[0]).lower() not in {"ssh", "ssh.exe"}:
            raise ValueError("remote lease wrapping requires an ssh command")
        if len(values) < 3:
            raise ValueError("ssh command has no remote payload")
        return values[:-1] + [self.wrap_remote_command(values[-1])]

    def cleanup_remote_command(
        self,
        *,
        grace_s: float = 3.0,
        launch_wait_s: float = 1.0,
    ) -> str:
        argv = ["python3", "-c", _REMOTE_CLEANUP_PY]
        argv.extend(self._identity_argv())
        argv.extend([str(max(0.0, float(grace_s))), str(max(0.0, float(launch_wait_s)))])
        return " ".join(shlex.quote(value) for value in argv)

    def cancel_remote(
        self,
        *,
        grace_s: float = 3.0,
        launch_wait_s: float = 1.0,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute the second, exact remote control command once."""

        with self._cancel_lock:
            if self._cancel_result is not None:
                return dict(self._cancel_result)
            command = list(self.control_argv_prefix) + [
                self.cleanup_remote_command(
                    grace_s=grace_s,
                    launch_wait_s=launch_wait_s,
                )
            ]
            timeout = (
                float(timeout_s)
                if timeout_s is not None
                else remote_process_lease_cleanup_timeout_s(
                    grace_s=grace_s,
                    launch_wait_s=launch_wait_s,
                )
            )
            started = time.monotonic()
            try:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=(os.name == "posix"),
                )
                try:
                    output, _ = proc.communicate(timeout=timeout)
                    returncode = int(proc.returncode or 0)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except (OSError, ProcessLookupError):
                            try:
                                proc.kill()
                            except OSError:
                                pass
                    else:
                        proc.kill()
                    try:
                        output, _ = proc.communicate(timeout=2.0)
                    except subprocess.TimeoutExpired:
                        output = "remote cleanup control process did not exit after SIGKILL"
                    returncode = 124
            except Exception as exc:
                output = f"{type(exc).__name__}: {exc}"
                returncode = 1
            cleanup_states = [
                line.strip().split("=", 1)[1]
                for line in str(output or "").splitlines()
                if line.strip().startswith(
                    "__SPLITPOINT_REMOTE_LEASE_CLEANUP__="
                )
            ]
            proven_cleanup = any(
                state == "cancelled_before_lease"
                or state == "cancelled_before_lease_unsafe_procfs"
                or state == "already_exited_drained"
                or (
                    state.startswith("terminated ")
                    and "remaining=0" in state.split()
                    and "drained=1" in state.split()
                )
                for state in cleanup_states
            )
            result = {
                "operation_id": self.operation_id,
                "returncode": returncode,
                "output": str(output or ""),
                "elapsed_s": time.monotonic() - started,
                # rc=0 alone is not proof that the intended remote Python
                # control program ran (a wrapper or shell could swallow it).
                # Every proven terminal path emits this exact sentinel.
                "ok": bool(
                    returncode == 0
                    and proven_cleanup
                ),
            }
            # Network/control failures are retryable.  Cache only a proven
            # remote stop (including the tombstoned-before-start outcome).
            if result["ok"]:
                self._cancel_result = dict(result)
            return dict(result)


class RemoteProcessLeaseJournalError(RuntimeError):
    """A local lease descriptor could not be trusted or persisted."""


class RemoteProcessLeaseLaunchRejected(RuntimeError):
    """Sticky cancellation forbids starting the registered main SSH path."""

    def __init__(
        self,
        operation: RemoteProcessLeaseOperation,
        report: Mapping[str, Any],
    ) -> None:
        self.operation = operation
        self.cleanup_report = dict(report)
        self.cleanup_proven = bool(report.get("ok"))
        self.returncode = 130 if self.cleanup_proven else REMOTE_LEASE_UNPROVEN_RC
        detail = "cleanup proven" if self.cleanup_proven else "cleanup unproven"
        super().__init__(
            "remote lease launch rejected after terminal cancellation "
            f"({detail}, operation={operation.operation_id})"
        )


class RemoteProcessLeaseJournal:
    """Atomic, session-local hand-off between nested SSH helpers and a parent.

    Nested helpers may run in child Python interpreters and therefore cannot
    register an operation in the parent's in-memory registry.  This journal is
    the durable ownership boundary: the helper publishes the exact operation
    token and SSH control prefix before it starts SSH; the parent scans those
    descriptors during cancellation and reconstructs only matching run/session
    operations.
    """

    _DESCRIPTOR_SUFFIX = ".remote-lease.json"
    _CANCEL_FILE = "session.cancelled.json"

    def __init__(self, *, scope: RemoteProcessLeaseScope, directory: os.PathLike) -> None:
        self.scope = scope
        supplied = Path(directory).expanduser()
        if supplied.exists() and supplied.is_symlink():
            raise RemoteProcessLeaseJournalError(
                f"remote lease journal must not be a symlink: {supplied}"
            )
        supplied.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            supplied.chmod(0o700)
        except OSError:
            pass
        self.directory = supplied.resolve()
        self._lock = threading.RLock()

    @classmethod
    def from_environment(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        required: bool = False,
    ) -> Optional["RemoteProcessLeaseJournal"]:
        values = os.environ if env is None else env
        names = (
            REMOTE_LEASE_ENV_RUN_ID,
            REMOTE_LEASE_ENV_SESSION_ID,
            REMOTE_LEASE_ENV_JOURNAL_DIR,
        )
        present = [bool(str(values.get(name, "")).strip()) for name in names]
        if not any(present):
            if required:
                raise RemoteProcessLeaseJournalError(
                    "remote lease journal environment is not configured"
                )
            return None
        if not all(present):
            missing = [name for name, available in zip(names, present) if not available]
            raise RemoteProcessLeaseJournalError(
                "incomplete remote lease journal environment; missing "
                + ", ".join(missing)
            )
        scope = RemoteProcessLeaseScope(
            str(values[REMOTE_LEASE_ENV_RUN_ID]),
            str(values[REMOTE_LEASE_ENV_SESSION_ID]),
            str(values.get(REMOTE_LEASE_ENV_REMOTE_ROOT, "") or ""),
        )
        return cls(scope=scope, directory=str(values[REMOTE_LEASE_ENV_JOURNAL_DIR]))

    def environment(self) -> Dict[str, str]:
        result = {
            REMOTE_LEASE_ENV_RUN_ID: str(self.scope.run_id),
            REMOTE_LEASE_ENV_SESSION_ID: str(self.scope.session_id),
            REMOTE_LEASE_ENV_JOURNAL_DIR: str(self.directory),
        }
        if self.scope.lease_root:
            result[REMOTE_LEASE_ENV_REMOTE_ROOT] = str(self.scope.lease_root)
        else:
            # Clear an inherited override when callers merge this mapping into
            # a subprocess environment for a scope that uses the remote default.
            result[REMOTE_LEASE_ENV_REMOTE_ROOT] = ""
        return result

    def _descriptor_path(self, operation_id: str) -> Path:
        value = str(operation_id)
        if _SAFE_OPERATION_RE.fullmatch(value) is None:
            raise RemoteProcessLeaseJournalError("unsafe remote lease operation_id")
        return self.directory / (value + self._DESCRIPTOR_SUFFIX)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        if os.name != "posix":
            return
        try:
            descriptor = os.open(str(directory), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    def _write_json_atomic(self, destination: Path, payload: Mapping[str, Any]) -> None:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        descriptor = -1
        try:
            descriptor = os.open(
                str(temporary),
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                json.dump(dict(payload), handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(destination))
            self._fsync_directory(self.directory)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except OSError:
                pass

    @staticmethod
    def _read_json_exact(path: Path) -> Dict[str, Any]:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RemoteProcessLeaseJournalError(
                f"cannot stat remote lease descriptor {path}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RemoteProcessLeaseJournalError(
                f"remote lease descriptor is not a regular file: {path}"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(str(path), flags)
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, ValueError, TypeError) as exc:
            raise RemoteProcessLeaseJournalError(
                f"invalid remote lease descriptor {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise RemoteProcessLeaseJournalError(
                f"remote lease descriptor is not an object: {path}"
            )
        return value

    def _payload(self, operation: RemoteProcessLeaseOperation) -> Dict[str, Any]:
        return {
            "schema": REMOTE_LEASE_JOURNAL_SCHEMA,
            "schema_version": REMOTE_LEASE_JOURNAL_SCHEMA_VERSION,
            "run_sha256": self.scope.run_sha256,
            "session_sha256": self.scope.session_sha256,
            "operation_id": operation.operation_id,
            "token": operation.token,
            "control_argv_prefix": list(operation.control_argv_prefix),
            "remote_lease_root": str(operation.scope.lease_root or ""),
        }

    def record_operation(self, operation: RemoteProcessLeaseOperation) -> Path:
        if (
            operation.scope.run_sha256 != self.scope.run_sha256
            or operation.scope.session_sha256 != self.scope.session_sha256
            or str(operation.scope.lease_root or "")
            != str(self.scope.lease_root or "")
        ):
            raise RemoteProcessLeaseJournalError(
                "remote lease operation does not belong to this journal scope"
            )
        if os.path.basename(operation.control_argv_prefix[0]).lower() not in {
            "ssh",
            "ssh.exe",
        }:
            raise RemoteProcessLeaseJournalError(
                "remote lease journal accepts only an exact ssh control prefix"
            )
        validate_remote_process_lease_ssh_prefix(operation.control_argv_prefix)
        destination = self._descriptor_path(operation.operation_id)
        payload = self._payload(operation)
        with self._lock:
            if destination.exists():
                current = self._read_json_exact(destination)
                if current != payload:
                    raise RemoteProcessLeaseJournalError(
                        f"conflicting remote lease descriptor: {destination}"
                    )
                return destination
            self._write_json_atomic(destination, payload)
        return destination

    def _operation_from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        path: Path,
    ) -> RemoteProcessLeaseOperation:
        expected = {
            "schema": REMOTE_LEASE_JOURNAL_SCHEMA,
            "schema_version": REMOTE_LEASE_JOURNAL_SCHEMA_VERSION,
            "run_sha256": self.scope.run_sha256,
            "session_sha256": self.scope.session_sha256,
            "remote_lease_root": str(self.scope.lease_root or ""),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise RemoteProcessLeaseJournalError(
                    f"remote lease journal identity mismatch for {path}: {key}"
                )
        operation_id = str(payload.get("operation_id", ""))
        if path != self._descriptor_path(operation_id):
            raise RemoteProcessLeaseJournalError(
                f"remote lease descriptor filename mismatch: {path}"
            )
        token = str(payload.get("token", ""))
        control = payload.get("control_argv_prefix")
        if not isinstance(control, list) or not control or not all(
            isinstance(value, str) and value for value in control
        ):
            raise RemoteProcessLeaseJournalError(
                f"invalid remote lease control argv in {path}"
            )
        if os.path.basename(control[0]).lower() not in {"ssh", "ssh.exe"}:
            raise RemoteProcessLeaseJournalError(
                f"non-ssh remote lease control argv in {path}"
            )
        try:
            validate_remote_process_lease_ssh_prefix(control)
        except RemoteProcessLeaseConfigurationError as exc:
            raise RemoteProcessLeaseJournalError(
                f"unsafe remote lease SSH options in {path}: {exc}"
            ) from exc
        try:
            return RemoteProcessLeaseOperation(
                scope=self.scope,
                operation_id=operation_id,
                token=token,
                control_argv_prefix=control,
            )
        except ValueError as exc:
            raise RemoteProcessLeaseJournalError(
                f"invalid remote lease operation in {path}: {exc}"
            ) from exc

    def _scan_with_errors(
        self,
    ) -> List[Tuple[Path, Optional[RemoteProcessLeaseOperation], Optional[str]]]:
        results: List[
            Tuple[Path, Optional[RemoteProcessLeaseOperation], Optional[str]]
        ] = []
        with self._lock:
            for path in sorted(self.directory.glob("*" + self._DESCRIPTOR_SUFFIX)):
                try:
                    payload = self._read_json_exact(path)
                    operation = self._operation_from_payload(payload, path=path)
                except RemoteProcessLeaseJournalError as exc:
                    # A normal helper may atomically finish and unlink between
                    # glob() and lstat/open.  Absence is no pending ownership;
                    # still-present corrupt, unreadable or symlink entries
                    # remain fail-closed.
                    if not os.path.lexists(str(path)):
                        continue
                    results.append((path, None, str(exc)))
                else:
                    results.append((path, operation, None))
        return results

    def scan_operations(self) -> List[RemoteProcessLeaseOperation]:
        operations: List[RemoteProcessLeaseOperation] = []
        errors: List[str] = []
        for _path, operation, error in self._scan_with_errors():
            if error is not None:
                errors.append(error)
            elif operation is not None:
                operations.append(operation)
        if errors:
            raise RemoteProcessLeaseJournalError("; ".join(errors))
        return operations

    def remove_operation(
        self,
        operation: RemoteProcessLeaseOperation,
        *,
        reason: str,
    ) -> bool:
        if reason not in {"normal_exit", "cleanup_proven", "not_started"}:
            raise RemoteProcessLeaseJournalError(
                "lease descriptor removal requires a proven terminal reason"
            )
        path = self._descriptor_path(operation.operation_id)
        with self._lock:
            if not path.exists():
                return False
            payload = self._read_json_exact(path)
            if (
                str(payload.get("token", "")) != operation.token
                or str(payload.get("run_sha256", "")) != self.scope.run_sha256
                or str(payload.get("session_sha256", ""))
                != self.scope.session_sha256
            ):
                raise RemoteProcessLeaseJournalError(
                    f"refusing to remove non-matching lease descriptor: {path}"
                )
            path.unlink()
            self._fsync_directory(self.directory)
        return True

    def mark_cancelled(self) -> None:
        payload = {
            "schema": REMOTE_LEASE_JOURNAL_SCHEMA,
            "schema_version": REMOTE_LEASE_JOURNAL_SCHEMA_VERSION,
            "run_sha256": self.scope.run_sha256,
            "session_sha256": self.scope.session_sha256,
            "cancelled": True,
        }
        with self._lock:
            self._write_json_atomic(self.directory / self._CANCEL_FILE, payload)

    def is_cancelled(self) -> bool:
        path = self.directory / self._CANCEL_FILE
        if not path.exists():
            return False
        try:
            payload = self._read_json_exact(path)
        except RemoteProcessLeaseJournalError:
            # A corrupt cancellation tombstone must fail closed.
            return True
        if (
            payload.get("schema") != REMOTE_LEASE_JOURNAL_SCHEMA
            or payload.get("schema_version") != REMOTE_LEASE_JOURNAL_SCHEMA_VERSION
            or payload.get("run_sha256") != self.scope.run_sha256
            or payload.get("session_sha256") != self.scope.session_sha256
        ):
            return True
        return payload.get("cancelled") is True

    def pending_count(self) -> int:
        return len(self._scan_with_errors())


class RemoteProcessLeaseRegistry:
    """Thread-safe sticky ownership registry for all workflow remote leases."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._operations: Dict[int, RemoteProcessLeaseOperation] = {}
        self._cancelled = False
        self._journal: Optional[RemoteProcessLeaseJournal] = None

    @staticmethod
    def _operation_key(operation: RemoteProcessLeaseOperation) -> Tuple[str, str]:
        return operation.operation_id, operation.token

    def configure_journal(
        self,
        *,
        scope: RemoteProcessLeaseScope,
        journal_dir: os.PathLike,
    ) -> RemoteProcessLeaseJournal:
        journal = RemoteProcessLeaseJournal(scope=scope, directory=journal_dir)
        with self._lock:
            if self._journal is not None and (
                self._journal.scope.run_sha256 != scope.run_sha256
                or self._journal.scope.session_sha256 != scope.session_sha256
                or self._journal.directory != journal.directory
            ):
                raise RemoteProcessLeaseJournalError(
                    "remote lease registry journal is already configured"
                )
            self._journal = journal
            operations = list(self._operations.values())
            cancelled = self._cancelled
        for operation in operations:
            journal.record_operation(operation)
        if cancelled:
            journal.mark_cancelled()
        return journal

    def journal_environment(self) -> Dict[str, str]:
        with self._lock:
            journal = self._journal
        if journal is None:
            raise RemoteProcessLeaseJournalError(
                "remote lease registry journal is not configured"
            )
        return journal.environment()

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def configured_scope(self) -> Optional[RemoteProcessLeaseScope]:
        """Return the immutable parent workflow scope, when journaled.

        Remote benchmark helpers also have their own result/run identifiers;
        those must never be substituted for the owning EvaluationRun identity
        once a registry journal is configured.
        """

        with self._lock:
            return self._journal.scope if self._journal is not None else None

    def poison(self) -> None:
        """Forbid every later launch after an unproven remote cleanup.

        The in-memory barrier is committed before the durable tombstone.  If
        persisting the tombstone fails, callers still observe a terminal
        registry while the exception tells them that cross-process proof is
        missing.
        """

        with self._lock:
            self._cancelled = True
            journal = self._journal
        if journal is not None:
            journal.mark_cancelled()

    def register(self, operation: RemoteProcessLeaseOperation) -> None:
        cancel_late = False
        with self._lock:
            if self._journal is not None:
                self._journal.record_operation(operation)
            self._operations[id(operation)] = operation
            cancel_late = self._cancelled or bool(
                self._journal is not None and self._journal.is_cancelled()
            )
            if cancel_late:
                self._cancelled = True
        if cancel_late:
            try:
                report = operation.cancel_remote()
            except Exception as exc:
                report = {
                    "operation_id": operation.operation_id,
                    "returncode": REMOTE_LEASE_UNPROVEN_RC,
                    "output": f"{type(exc).__name__}: {exc}",
                    "ok": False,
                }
            if bool(report.get("ok")):
                self.unregister(operation, reason="cleanup_proven")
            # Terminal cancellation is a start barrier, not a cleanup hint.
            # Even a proven tombstone must never return control to a caller
            # that will proceed to the main Popen; an unproven cleanup keeps
            # both the in-memory operation and its journal descriptor.
            raise RemoteProcessLeaseLaunchRejected(operation, report)

    def unregister(
        self,
        operation: RemoteProcessLeaseOperation,
        *,
        reason: str = "normal_exit",
    ) -> None:
        with self._lock:
            if self._journal is not None:
                self._journal.remove_operation(operation, reason=reason)
            self._operations.pop(id(operation), None)

    def recover_journaled_operations(self) -> List[RemoteProcessLeaseOperation]:
        with self._lock:
            journal = self._journal
        if journal is None:
            return []
        recovered = journal.scan_operations()
        with self._lock:
            existing = {
                self._operation_key(operation)
                for operation in self._operations.values()
            }
            for operation in recovered:
                if self._operation_key(operation) not in existing:
                    self._operations[id(operation)] = operation
                    existing.add(self._operation_key(operation))
        return recovered

    def _recover_journal_for_cancel(self) -> List[Dict[str, Any]]:
        """Recover valid descriptors and report invalid ones without deleting.

        An unreadable descriptor represents unproven remote state.  It remains
        on disk and therefore keeps ``active_count`` non-zero for the parent's
        final fail-closed retry/reporting path.
        """

        with self._lock:
            journal = self._journal
        if journal is None:
            return []
        invalid: List[Dict[str, Any]] = []
        scanned = journal._scan_with_errors()
        with self._lock:
            existing = {
                self._operation_key(operation)
                for operation in self._operations.values()
            }
            for path, operation, error in scanned:
                if error is not None:
                    invalid.append(
                        {
                            "operation_id": "",
                            "descriptor_path": str(path),
                            "returncode": 65,
                            "output": error,
                            "elapsed_s": 0.0,
                            "ok": False,
                        }
                    )
                    continue
                assert operation is not None
                key = self._operation_key(operation)
                if key not in existing:
                    self._operations[id(operation)] = operation
                    existing.add(key)
        return invalid

    def cancel_all(self, *, grace_s: float = 3.0) -> List[Dict[str, Any]]:
        """Set sticky cancellation and clean all active leases in parallel."""

        persistence_reports: List[Dict[str, Any]] = []
        tombstone_persisted = True
        try:
            self.poison()
        except Exception as exc:
            # ``poison`` sets the in-memory barrier first.  A failed durable
            # tombstone must not abort exact cleanup of operations that are
            # already known in memory or can still be recovered below.
            tombstone_persisted = False
            with self._lock:
                journal = self._journal
            persistence_reports.append(
                {
                    "operation_id": "",
                    "descriptor_path": str(
                        journal.directory / journal._CANCEL_FILE
                    )
                    if journal is not None
                    else "",
                    "returncode": 74,
                    "output": (
                        "remote lease cancellation tombstone could not be "
                        f"persisted: {type(exc).__name__}: {exc}"
                    ),
                    "elapsed_s": 0.0,
                    "ok": False,
                    "error_code": "remote_lease_tombstone_write_failed",
                }
            )
        try:
            invalid_reports = self._recover_journal_for_cancel()
        except Exception as exc:
            # Keep cleaning the operations already registered in memory.  A
            # journal scan failure remains an explicit unproven result.
            with self._lock:
                journal = self._journal
            invalid_reports = [
                {
                    "operation_id": "",
                    "descriptor_path": str(journal.directory)
                    if journal is not None
                    else "",
                    "returncode": 74,
                    "output": (
                        "remote lease journal scan failed during cancellation: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    "elapsed_s": 0.0,
                    "ok": False,
                    "error_code": "remote_lease_journal_scan_failed",
                }
            ]
        with self._lock:
            operations = list(self._operations.values())
        if not operations:
            return persistence_reports + invalid_reports
        reports: List[Optional[Dict[str, Any]]] = [None] * len(operations)
        report_lock = threading.Lock()

        def _cancel(index: int, operation: RemoteProcessLeaseOperation) -> None:
            try:
                report = operation.cancel_remote(grace_s=grace_s)
            except Exception as exc:
                report = {
                    "operation_id": operation.operation_id,
                    "returncode": REMOTE_LEASE_UNPROVEN_RC,
                    "output": f"{type(exc).__name__}: {exc}",
                    "elapsed_s": 0.0,
                    "ok": False,
                }
            with report_lock:
                # The owner may already have committed a timeout report.  A
                # late daemon result cannot retroactively prove that the
                # bounded cancellation pass completed in time.
                if reports[index] is None:
                    reports[index] = report

        threads = [
            threading.Thread(
                target=_cancel,
                args=(index, operation),
                name=f"remote-lease-cancel-{index}",
                daemon=True,
            )
            for index, operation in enumerate(operations)
        ]
        for thread in threads:
            thread.start()
        join_started = time.monotonic()
        join_budget_s = _remote_process_lease_cancel_join_budget_s(
            grace_s=grace_s
        )
        join_deadline = join_started + join_budget_s
        for thread in threads:
            remaining_s = max(0.0, join_deadline - time.monotonic())
            if remaining_s <= 0.0:
                break
            thread.join(timeout=remaining_s)
        with report_lock:
            for index, (thread, operation) in enumerate(zip(threads, operations)):
                if reports[index] is not None:
                    continue
                timed_out = thread.is_alive()
                reports[index] = {
                    "operation_id": operation.operation_id,
                    "returncode": REMOTE_LEASE_UNPROVEN_RC,
                    "output": (
                        "exact remote lease cleanup worker exceeded the "
                        "bounded join budget"
                        if timed_out
                        else "exact remote lease cleanup worker exited without a report"
                    ),
                    "elapsed_s": max(0.0, time.monotonic() - join_started),
                    "ok": False,
                    "error_code": (
                        "remote_lease_cleanup_worker_timeout"
                        if timed_out
                        else "remote_lease_cleanup_worker_no_report"
                    ),
                }
        completed = persistence_reports + invalid_reports + [
            dict(report) for report in reports if report is not None
        ]
        successful: List[RemoteProcessLeaseOperation] = [
            operation
            for operation, report in zip(operations, reports)
            if report is not None and bool(report.get("ok"))
        ]
        with self._lock:
            for operation in successful:
                if not tombstone_persisted:
                    # Individual remote cleanup may be proven, but without a
                    # durable session barrier a late helper can still race a
                    # new launch.  Retain ownership until a later pass can
                    # persist the tombstone and remove the descriptor.
                    continue
                removal_ok = True
                if self._journal is not None:
                    try:
                        self._journal.remove_operation(
                            operation, reason="cleanup_proven"
                        )
                    except Exception as exc:
                        removal_ok = False
                        for report in completed:
                            if (
                                report.get("operation_id")
                                == operation.operation_id
                                and bool(report.get("ok"))
                            ):
                                report["ok"] = False
                                report["returncode"] = 74
                                report["output"] = (
                                    str(report.get("output", ""))
                                    + f"\nlocal journal removal failed: {exc}"
                                ).strip()
                                break
                if removal_ok:
                    self._operations.pop(id(operation), None)
        return completed

    def active_count(self) -> int:
        with self._lock:
            operations = {
                self._operation_key(operation)
                for operation in self._operations.values()
            }
            journal = self._journal
        if journal is None:
            return len(operations)
        scanned = journal._scan_with_errors()
        valid = {
            self._operation_key(operation)
            for _path, operation, error in scanned
            if error is None and operation is not None
        }
        invalid_count = sum(1 for _path, _operation, error in scanned if error)
        return len(operations | valid) + invalid_count


def resolve_remote_process_lease_scope(
    *,
    registry: Optional[RemoteProcessLeaseRegistry],
    fallback_run_id: str,
    workflow_session_id: str,
) -> Optional[RemoteProcessLeaseScope]:
    """Resolve ownership without confusing a benchmark sub-run with its run.

    A configured parent registry is authoritative.  The fallback preserves
    standalone remote-benchmark callers which have no EvaluationRun journal.
    """

    if registry is not None and registry.configured_scope is not None:
        return registry.configured_scope
    if str(workflow_session_id or "").strip():
        return RemoteProcessLeaseScope(
            run_id=str(fallback_run_id),
            session_id=str(workflow_session_id),
        )
    return None


def cancel_journaled_remote_processes_from_environment(
    *,
    env: Optional[Mapping[str, str]] = None,
    grace_s: float = 3.0,
) -> List[Dict[str, Any]]:
    """Drain a nested helper's session journal before its local tree stops.

    With no workflow lease environment this is a compatibility no-op.  Once a
    journal is configured, cleanup is fail-closed: its cancellation tombstone
    blocks later SSH starts, and unproven descriptors remain for the owning
    EvaluationWorkflowRunner's retry loop.
    """

    journal = RemoteProcessLeaseJournal.from_environment(env, required=False)
    if journal is None:
        return []
    registry = RemoteProcessLeaseRegistry()
    registry.configure_journal(
        scope=journal.scope,
        journal_dir=journal.directory,
    )
    reports = registry.cancel_all(grace_s=grace_s)
    remaining = registry.active_count()
    failures = [report for report in reports if not bool(report.get("ok"))]
    if remaining or failures:
        raise RemoteProcessLeaseJournalError(
            "remote cleanup is not proven for "
            f"{remaining} journaled operation(s); "
            f"{len(failures)} cancellation failure report(s)"
        )
    return reports


def _validate_raw_ssh_argv(argv: Sequence[str]) -> List[str]:
    values = [str(value) for value in argv]
    if not values or os.path.basename(values[0]).lower() not in {"ssh", "ssh.exe"}:
        raise RemoteProcessLeaseJournalError(
            "journaled remote execution requires a raw ssh argv"
        )
    if len(values) < 3:
        raise RemoteProcessLeaseJournalError("raw ssh argv has no remote payload")
    prefix = harden_remote_process_lease_ssh_argv(values[:-1])
    return prefix + [values[-1]]


def journaled_ssh_wrapper_argv(
    argv: Sequence[str],
    *,
    label: str,
    env: Optional[Mapping[str, str]] = None,
    timeout_s: Optional[float] = None,
) -> List[str]:
    """Return an argv wrapper suitable for existing streaming subprocess code.

    The returned helper inherits stdio, so callers can continue using their
    existing heartbeat/progress parser.  The configured journal environment
    must also be inherited or passed to that subprocess.
    """

    values = [str(value) for value in argv]
    # This transform is intentionally safe at broad execution boundaries: it
    # leaves local/scp/rsync commands untouched, and preserves historical
    # standalone SSH behaviour when no workflow lease environment exists.
    if not values or os.path.basename(values[0]).lower() not in {"ssh", "ssh.exe"}:
        return values
    journal = RemoteProcessLeaseJournal.from_environment(env, required=False)
    if journal is None:
        return values
    _validate_raw_ssh_argv(values)
    wrapped = [
        sys.executable,
        "-m",
        "onnx_splitpoint_tool.remote.process_lease_cli",
        "exec",
        "--label",
        _safe_label(label),
    ]
    if timeout_s is not None:
        wrapped.extend(["--timeout-s", str(max(0.0, float(timeout_s)))])
    return wrapped + ["--"] + values


def _stop_local_ssh(proc: subprocess.Popen, *, grace_s: float = 1.0) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except OSError:
                pass
    else:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=max(0.0, float(grace_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                proc.kill()
            except OSError:
                pass
    else:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass


def run_journaled_ssh(
    argv: Sequence[str],
    *,
    label: str,
    env: Optional[Mapping[str, str]] = None,
    timeout_s: Optional[float] = None,
) -> int:
    """Run one raw SSH argv under an exact, cross-process lease.

    The descriptor is published before SSH can start.  SIGINT/SIGTERM and an
    abnormal SSH exit invoke the independent exact remote cleanup command
    before the local SSH process is terminated.  A descriptor is removed only
    after a normal SSH exit, a definitely-not-started path, or proven cleanup;
    otherwise it remains for the parent registry's retry loop.
    """

    values = _validate_raw_ssh_argv(argv)
    journal = RemoteProcessLeaseJournal.from_environment(env, required=True)
    assert journal is not None
    if journal.is_cancelled():
        return 130
    operation = journal.scope.operation(
        label=label,
        control_argv_prefix=values[:-1],
    )

    received_signal = threading.Event()
    signal_number = [0]
    previous_handlers: Dict[int, Any] = {}

    def _on_signal(signum: int, _frame: Any) -> None:
        signal_number[0] = int(signum)
        received_signal.set()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _on_signal)
            except (OSError, ValueError):
                pass

    proc: Optional[subprocess.Popen] = None
    try:
        journal.record_operation(operation)
        # Close the publication/cancellation race.  At this point no local SSH
        # process exists, so removing the descriptor is a proven not-started
        # outcome; the session tombstone prevents later helpers from starting.
        if journal.is_cancelled() or received_signal.is_set():
            if received_signal.is_set():
                try:
                    journal.mark_cancelled()
                except Exception:
                    # The main SSH path is proven not started, but without a
                    # durable session tombstone cross-process cancellation is
                    # not proven.  Best-effort the exact remote tombstone,
                    # retain the descriptor, and surface the fail-closed rc.
                    try:
                        operation.cancel_remote(grace_s=3.0)
                    except Exception:
                        pass
                    return REMOTE_LEASE_UNPROVEN_RC
            journal.remove_operation(operation, reason="not_started")
            return 130
        leased_argv = operation.wrap_ssh_argv(values)
        try:
            proc = subprocess.Popen(
                leased_argv,
                start_new_session=(os.name == "posix"),
            )
        except Exception:
            journal.remove_operation(operation, reason="not_started")
            raise

        returncode: Optional[int] = None
        timed_out = False
        started = time.monotonic()
        while returncode is None and not received_signal.is_set() and not timed_out:
            try:
                returncode = int(proc.wait(timeout=0.05))
            except subprocess.TimeoutExpired:
                timed_out = bool(
                    timeout_s is not None
                    and time.monotonic() - started >= max(0.0, float(timeout_s))
                )

        if received_signal.is_set() or timed_out:
            # A local timeout/signal is terminal for this workflow session.
            # Poison before cleanup so no parallel/later helper can start a
            # new physical workload while exact stop is being established.
            tombstone_error: Optional[Exception] = None
            try:
                journal.mark_cancelled()
            except Exception as exc:
                tombstone_error = exc
            cleanup_ok = False
            journal_remove_ok = True
            try:
                try:
                    report = operation.cancel_remote(grace_s=3.0)
                except Exception:
                    report = {"ok": False}
                cleanup_ok = bool(report.get("ok"))
                if cleanup_ok and tombstone_error is None:
                    try:
                        journal.remove_operation(
                            operation, reason="cleanup_proven"
                        )
                    except Exception:
                        journal_remove_ok = False
            finally:
                # Ordering matters: do not terminate the primary SSH path
                # until the independent exact cleanup command returned.
                _stop_local_ssh(proc)
            if (
                tombstone_error is not None
                or not cleanup_ok
                or not journal_remove_ok
            ):
                return REMOTE_LEASE_UNPROVEN_RC
            return 124 if timed_out else 130

        assert returncode is not None
        if returncode == 0:
            journal.remove_operation(operation, reason="normal_exit")
            return 0

        # A failed local SSH connection is not proof that the remote launcher
        # and its descendants stopped.  Retain the descriptor unless the
        # independent exact control path proves cleanup.
        report = operation.cancel_remote(grace_s=3.0)
        if bool(report.get("ok")):
            journal.remove_operation(operation, reason="cleanup_proven")
            return returncode
        try:
            journal.mark_cancelled()
        except Exception:
            # The descriptor deliberately remains.  rc=70 communicates that
            # neither remote cleanup nor the durable session barrier is
            # proven; never leak the journal I/O exception as an ordinary SSH
            # execution failure.
            pass
        return REMOTE_LEASE_UNPROVEN_RC
    finally:
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, ValueError):
                pass


def process_lease_cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m onnx_splitpoint_tool.remote.process_lease_cli"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("exec", help="run raw SSH under an exact lease")
    execute.add_argument("--label", required=True)
    execute.add_argument("--timeout-s", type=float, default=None)
    execute.add_argument("ssh_argv", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    raw = list(arguments.ssh_argv)
    if raw and raw[0] == "--":
        raw = raw[1:]
    try:
        return run_journaled_ssh(
            raw,
            label=str(arguments.label),
            timeout_s=arguments.timeout_s,
        )
    except (RemoteProcessLeaseJournalError, ValueError) as exc:
        print(f"remote lease configuration error: {exc}", file=sys.stderr)
        return 64
    except OSError as exc:
        print(f"remote lease execution error: {exc}", file=sys.stderr)
        return 71


class ControllerResourceClaim:
    """Child side of the existing journal's controller capture handoff."""

    def __init__(self, journal, *, setup_id, attempt_dir, cancel_event=None, purpose="capture", continuation=None):
        from onnx_splitpoint_tool.process_control import _proc_start_time
        self.journal = journal
        self.cancel_event = cancel_event
        self.config = journal._read_json_exact(journal.directory / "controller.resources.json")
        self.operation = "capture-" + uuid.uuid4().hex
        self.token = uuid.uuid4().hex
        self.payload = {"operation_id": self.operation, "token": self.token,
            "run_id": journal.scope.run_id, "session_id": journal.scope.session_id,
            "owner_pid": os.getpid(), "owner_start_ticks": _proc_start_time(os.getpid()),
            "setup_id": str(setup_id), "attempt_dir": str(Path(attempt_dir).resolve()),
            "sequence": 0, "phase": "RESERVING", "purpose": purpose, "continuation": continuation}
        self.acquired = False
        self.process_started = False

    @classmethod
    def from_environment(cls, *, setup_id, attempt_dir, cancel_event=None, env=None, purpose="capture", continuation=None):
        values = {**os.environ, **(env or {})}
        if values.get("ONNX_SPLITPOINT_CONTROLLER_RESOURCES") != "required":
            return None
        journal = RemoteProcessLeaseJournal.from_environment(values, required=True)
        return cls(journal, setup_id=setup_id, attempt_dir=attempt_dir, cancel_event=cancel_event, purpose=purpose, continuation=continuation)

    def _parent_alive(self):
        from onnx_splitpoint_tool.process_control import _proc_start_time
        return _proc_start_time(int(self.config["parent_pid"])) == self.config["parent_start_ticks"]

    def _publish(self, phase, **fields):
        self.payload.update(fields, phase=phase, sequence=self.payload["sequence"] + 1)
        self.journal._write_json_atomic(self.journal.directory / (self.operation + ".resource-request.json"), self.payload)

    def _wait(self, expected, *, allow_cancel=False):
        path = self.journal.directory / (self.operation + ".resource-reply.json")
        while True:
            if not self._parent_alive():
                raise RemoteProcessLeaseJournalError("controller_resource_owner_lost")
            if not allow_cancel and self.cancel_event is not None and self.cancel_event.is_set():
                raise RemoteProcessLeaseJournalError("controller_resource_cancelled_before_dispatch")
            if path.exists():
                row = self.journal._read_json_exact(path)
                if any(row.get(key) != self.payload[key] for key in ("operation_id", "token", "run_id", "session_id")):
                    raise RemoteProcessLeaseJournalError("controller_resource_reply_binding_mismatch")
                if row.get("state") == "STOP":
                    raise RemoteProcessLeaseJournalError(str(row.get("reason") or "controller_resource_stopped"))
                if row.get("state") == expected:
                    return row
            time.sleep(0.025)

    def acquire(self):
        self._publish("RESERVING")
        try:
            row = self._wait("QUIET_CONFIRMED" if self.payload["purpose"] == "capture" else "ACTIVE")
        except BaseException:
            self._publish("UNUSED")
            raise
        self.acquired = True
        return row

    def started(self, proc):
        from onnx_splitpoint_tool.process_control import _proc_start_time
        self.process_started = True
        self._publish("ACQUIRING", collector_pid=int(proc.pid), collector_start_ticks=_proc_start_time(proc.pid))

    def finish(self, result):
        self._publish("FINALIZING", process_started=bool(result.get("process_started", self.process_started)))
        return self._wait("RELEASED", allow_cancel=True)


class ControllerResourceBroker:
    """Parent admission over one ResourcePauseGate and existing physical leases.

    tick never waits for work, SSH or a child. Journal replies are the dispatch
    gate; a failed or lost collector owner leaves the source's durable fence.
    """

    def __init__(self, journal, gate, *, run_dir, setups, parent_pid=None):
        from onnx_splitpoint_tool.process_control import _proc_start_time
        self.journal, self.gate = journal, gate
        self.run_dir = Path(run_dir).resolve()
        self.setups = {str(key): tuple(sorted(set(value))) for key, value in setups.items()}
        self.operations = {}
        self.stopped_sources = set()
        self.parent_pid = int(parent_pid or os.getpid())
        self.config = {"run_id": journal.scope.run_id, "session_id": journal.scope.session_id,
            "parent_pid": self.parent_pid, "parent_start_ticks": _proc_start_time(self.parent_pid),
            "run_dir": str(self.run_dir), "setups": {key: list(value) for key, value in self.setups.items()}}
        path = journal.directory / "controller.resources.json"
        if path.exists():
            previous = journal._read_json_exact(path)
            if previous != self.config:
                raise RemoteProcessLeaseJournalError("controller_resource_owner_changed")
        journal._write_json_atomic(path, self.config)

    def _reply(self, row, state, **details):
        result = {key: row[key] for key in ("operation_id", "token", "run_id", "session_id")}
        result.update(state=state, observed_monotonic=time.monotonic(), **details)
        self.journal._write_json_atomic(self.journal.directory / (row["operation_id"] + ".resource-reply.json"), result)
        return result

    def _validate(self, path, row):
        if (row.get("run_id") != self.journal.scope.run_id
                or row.get("session_id") != self.journal.scope.session_id
                or not _SAFE_OPERATION_RE.fullmatch(str(row.get("operation_id") or ""))
                or path.name != row["operation_id"] + ".resource-request.json"
                or not re.fullmatch(r"[0-9a-f]{32}", str(row.get("token") or ""))
                or row.get("setup_id") not in self.setups
                or row.get("purpose", "capture") not in {"capture", "dut_job", "transfer", "prepare", "postcalc"}):
            raise RemoteProcessLeaseJournalError("controller_resource_request_binding_invalid")
        attempt = Path(str(row.get("attempt_dir") or ""))
        if not attempt.is_absolute() or not attempt.resolve().is_relative_to(self.run_dir) or attempt.is_symlink():
            raise RemoteProcessLeaseJournalError("controller_resource_attempt_outside_run")
        return attempt

    def _close(self, operation, *, stopped=False, reason=""):
        from onnx_splitpoint_tool.process_control import _proc_start_time
        row = operation["row"]
        if operation.get("activity") is not None:
            self.gate.release_activity(operation.pop("activity"))
        if not stopped:
            for key in ("reservation", "dut_reservation"):
                if operation.get(key) is not None:
                    self.gate.end_quiet(operation.pop(key))
        for resource, lease in reversed(operation.get("leases", [])):
            if stopped and resource.startswith(("dut:", "source:", "controller:capture", "controller:nic")):
                lease.commit_quarantine_fence({"reason": reason, "resource_request": row,
                    "source_completion_unproven": True})
                self.stopped_sources.add(resource)
            lease.release()
        operation["leases"] = []
        operation["terminal"] = True
        self._reply(row, "STOP" if stopped else "RELEASED", reason=reason)

    def tick(self):
        from onnx_splitpoint_tool.process_control import _proc_start_time
        from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock, WorkflowRunControlError
        from onnx_splitpoint_tool.energy.task_budget import source_completion
        for path in sorted(self.journal.directory.glob("*.resource-request.json")):
            row = self.journal._read_json_exact(path)
            attempt = self._validate(path, row)
            key = row["operation_id"]
            operation = self.operations.get(key)
            if operation is not None:
                identity_fields = ("token", "owner_pid", "owner_start_ticks", "setup_id", "attempt_dir", "purpose", "continuation")
                if any(operation["row"].get(field) != row.get(field) for field in identity_fields):
                    raise RemoteProcessLeaseJournalError("controller_resource_owner_changed")
                if int(row.get("sequence") or 0) < int(operation["row"].get("sequence") or 0):
                    raise RemoteProcessLeaseJournalError("controller_resource_sequence_regressed")
                if operation.get("terminal"):
                    continue
                operation["row"] = row
            else:
                if row["phase"] not in {"RESERVING", "UNUSED"}:
                    raise RemoteProcessLeaseJournalError("controller_resource_missing_reservation")
                from onnx_splitpoint_tool.process_control import _capture_process_tree
                owned = {identity.pid: identity.start_time_ticks for identity in _capture_process_tree(self.parent_pid)}
                if owned.get(int(row["owner_pid"])) != row["owner_start_ticks"]:
                    raise RemoteProcessLeaseJournalError("controller_resource_unowned_request_process")
                resources = set(self.setups[row["setup_id"]])
                purpose = row.get("purpose", "capture")
                if purpose == "dut_job":
                    resources = {key for key in resources if key.startswith("dut:")}
                elif purpose in {"transfer", "prepare", "postcalc"}:
                    resources = {key for key in resources if key in {"controller:cpu", "controller:io"}
                                 or (purpose == "transfer" and key.startswith("controller:nic"))}
                if purpose in {"transfer", "postcalc"}:
                    resources.add("controller:" + purpose)
                continuation = self.operations.get(row.get("continuation"))
                if row.get("continuation"):
                    if (continuation is None or not continuation.get("granted") or continuation.get("terminal")
                            or any(continuation["row"].get(k) != row.get(k) for k in ("owner_pid", "owner_start_ticks", "setup_id"))
                            or continuation["row"].get("purpose") not in {"prepare", "transfer", "postcalc"}):
                        raise RemoteProcessLeaseJournalError("controller_activity_continuation_invalid")
                    resources -= continuation["resources"]
                operation = {"row": row, "resources": resources, "leases": [], "terminal": False,
                             "continuation_token": continuation.get("activity") if continuation else None}
                self.operations[key] = operation
                if resources.intersection(self.stopped_sources):
                    operation["terminal"] = True
                    self._reply(row, "STOP", reason="campaign_source_completion_unresolved")
                    continue
            alive = _proc_start_time(int(row["owner_pid"])) == row["owner_start_ticks"]
            if not alive:
                self._close(operation, stopped=operation.get("granted", False), reason="controller_resource_owner_lost")
                continue
            if row["phase"] == "UNUSED":
                self._close(operation, stopped=operation.get("started", False), reason="cancelled_before_dispatch")
                continue
            if row["phase"] == "RESERVING":
                purpose = row.get("purpose", "capture")
                if purpose != "capture":
                    if "activity" not in operation:
                        operation["activity"] = self.gate.begin_activity(key, resources=operation["resources"],
                            cpu=0 if purpose == "dut_job" or operation["continuation_token"] else 1,
                            memory_bytes=0 if purpose == "dut_job" or operation["continuation_token"] else 256 * 1024**2,
                            ignore_pause_reasons=("urecs_energy_acquisition",), continuation_token=operation["continuation_token"])
                    if operation.get("granted") or not self.gate.poll_activity(operation["activity"]):
                        continue
                    quiet = {"state": "ACTIVE"}
                else:
                    quiet = None
                if purpose == "capture":
                    # Close DUT admission first. An already running job must
                    # finish its uploads/downloads before controller quietness
                    # can block these nested activities without a wait cycle.
                    if "dut_reservation" not in operation:
                        operation["dut_reservation"] = self.gate.begin_quiet(key + ":dut",
                            resources={r for r in operation["resources"] if r.startswith("dut:")}, owner=key)
                    if self.gate.poll_quiet(operation["dut_reservation"])["state"] != "QUIET_CONFIRMED":
                        continue
                    if "reservation" not in operation:
                        operation["reservation"] = self.gate.begin_quiet(key,
                            resources={r for r in operation["resources"] if not r.startswith("dut:")}, owner=key)
                    quiet = self.gate.poll_quiet(operation["reservation"])
                    if quiet["state"] != "QUIET_CONFIRMED" or operation.get("granted"):
                        continue
                try:
                    for resource in sorted(operation["resources"]):
                        if not resource.startswith(("dut:", "source:", "controller:capture", "controller:nic")):
                            continue
                        if purpose != "capture" and not resource.startswith("dut:"):
                            continue
                        lease = EvaluationRunLock.for_resource(resource, owner={
                            "run_id": self.journal.scope.run_id, "session_id": self.journal.scope.session_id,
                            "operation_id": key, "owner_pid": row["owner_pid"]})
                        lease.acquire()
                        operation["leases"].append((resource, lease))
                except WorkflowRunControlError as exc:
                    self._close(operation, reason="physical_resource_conflict:" + str(exc))
                    self._reply(row, "STOP", reason="physical_resource_conflict:" + str(exc))
                    continue
                operation["granted"] = True
                self._reply(row, "QUIET_CONFIRMED" if purpose == "capture" else "ACTIVE", reservation=quiet)
            elif row["phase"] == "ACQUIRING":
                if not operation.get("granted"):
                    raise RemoteProcessLeaseJournalError("collector_started_without_quiet_grant")
                operation["started"] = True
            elif row["phase"] == "FINALIZING":
                if row.get("purpose", "capture") != "capture":
                    if not operation.get("granted"):
                        raise RemoteProcessLeaseJournalError("activity_finished_without_admission")
                    self._close(operation)
                    continue
                started = bool(row.get("process_started") or operation.get("started"))
                cleanup_path = attempt / "collector_stdout.log.cleanup.json"
                cleanup = self.journal._read_json_exact(cleanup_path) if cleanup_path.is_file() else {}
                if not operation.get("granted"):
                    raise RemoteProcessLeaseJournalError("collector_finished_without_quiet_grant")
                cleanup_ok = (cleanup.get("owned_tree_quiescent") is True
                    and cleanup.get("collector_pid") == row.get("collector_pid")
                    and cleanup.get("collector_start_ticks") == row.get("collector_start_ticks")
                    and (not started or (cleanup.get("process_started") is True and bool(row.get("collector_start_ticks")))))
                verified = source_completion(attempt)["verified"] if started else True
                self._close(operation, stopped=not (verified and (cleanup_ok or not started)),
                            reason="" if verified and (cleanup_ok or not started) else "campaign_source_completion_unresolved")
            else:
                raise RemoteProcessLeaseJournalError("controller_resource_unknown_phase")

    def close(self):
        self.tick()
        for operation in self.operations.values():
            if not operation.get("terminal"):
                self._close(operation, stopped=operation.get("granted", False), reason="controller_resource_shutdown_unresolved")


@contextmanager
def controller_activity(*, setup_id, attempt_dir, purpose, cancel_event=None, env=None, continuation=None):
    claim = ControllerResourceClaim.from_environment(setup_id=setup_id, attempt_dir=attempt_dir,
        purpose=purpose, cancel_event=cancel_event, env=env, continuation=continuation)
    if claim is None:
        yield None
        return
    claim.acquire()
    try:
        yield claim
    finally:
        claim.finish({"process_started": False})
