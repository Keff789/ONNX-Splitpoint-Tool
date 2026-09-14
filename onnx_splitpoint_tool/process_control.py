from __future__ import annotations

"""Scoped process-tree cancellation for workflow-owned subprocesses.

The Evaluation Workflow deliberately starts long-running helpers in their own
sessions.  Some of those helpers start another session for an Energy workload.
Killing only the direct child's process group therefore leaves the nested
workload alive.  This module repeatedly captures the active descendant tree
before and during cancellation and terminates each captured process group
without ever signalling the caller's own group.  Workflow helpers are
foreground processes; deliberately self-daemonising third-party programs must
use a dedicated ownership broker instead of relying on this local tracker.
"""

from dataclasses import dataclass
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
import os
import signal
import subprocess
import threading
import time


_ACTIVE_PROCESS_REGISTRY: ContextVar[Any] = ContextVar(
    "onnx_splitpoint_active_process_registry",
    default=None,
)


@contextmanager
def bind_process_registry(registry: Any):
    """Bind workflow child ownership to synchronous nested helper calls."""

    token = _ACTIVE_PROCESS_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _ACTIVE_PROCESS_REGISTRY.reset(token)


def current_process_registry() -> Any:
    return _ACTIVE_PROCESS_REGISTRY.get()


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_time_ticks: str
    depth: int


class ProcessTreeCapabilityError(RuntimeError):
    """Raised when nested-session cancellation cannot be proved safely."""

    error_code = "nested_process_tree_cancellation_unavailable"

    def __init__(self, message: str, *, capabilities: dict[str, Any]) -> None:
        super().__init__(message)
        self.capabilities = dict(capabilities)


class ProcessTreeCancellationError(RuntimeError):
    """Raised when workflow-owned processes remain after bounded cancellation."""

    error_code = "workflow_process_tree_not_quiescent"

    def __init__(
        self,
        message: str,
        *,
        reports: list[dict[str, Any]],
        capabilities: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.reports = [dict(report) for report in reports]
        self.capabilities = dict(capabilities)


@lru_cache(maxsize=1)
def _procfs_matches_active_pid_namespace() -> bool:
    """Return whether mounted procfs uses the caller's PID namespace.

    Container launchers can expose a host ``/proc`` while Python sees a PID
    from a nested namespace.  Traversing numeric paths in that situation can
    select an unrelated host process, so descendant discovery must be disabled
    unless ``/proc/self`` and ``os.getpid()`` agree.
    """

    if os.name != "posix":
        return False
    try:
        raw = Path("/proc/self/stat").read_text(encoding="utf-8")
        procfs_pid = int(raw.split("(", 1)[0].strip())
    except Exception:
        return False
    return procfs_pid == os.getpid()


def process_tree_capabilities() -> dict[str, Any]:
    """Describe which cancellation guarantees are safe in this process.

    A POSIX process group is sufficient for ordinary descendants.  Descendants
    that call ``setsid()`` require trustworthy procfs traversal; using a host
    procfs from a nested PID namespace would be unsafe because numeric PIDs can
    refer to unrelated processes.  Windows uses ``taskkill /T`` instead.
    """

    procfs_match = bool(_procfs_matches_active_pid_namespace())
    root_group_available = os.name == "posix"
    nested_safe = bool((os.name == "posix" and procfs_match) or os.name == "nt")
    if nested_safe:
        reason = "nested_process_tree_cancellation_available"
    elif os.name == "posix":
        reason = "procfs_pid_namespace_mismatch_or_unavailable"
    else:
        reason = "unsupported_process_tree_platform"
    return {
        "process_control_platform": os.name,
        "procfs_pid_namespace_matches": procfs_match,
        "descendant_discovery_available": bool(os.name == "posix" and procfs_match),
        "root_process_group_cancellation_available": root_group_available,
        "nested_session_cancellation_safe": nested_safe,
        "cancellation_capability_reason": reason,
        "cancellation_assurance": (
            "full_descendant_tree" if nested_safe else "root_process_group_only"
        ),
        "unverified_nested_descendants_possible": not nested_safe,
    }


def require_nested_session_cancellation() -> dict[str, Any]:
    """Fail closed before starting a workflow that creates nested sessions."""

    capabilities = process_tree_capabilities()
    if not bool(capabilities.get("nested_session_cancellation_safe")):
        raise ProcessTreeCapabilityError(
            "Nested-session cancellation is unavailable: mounted procfs does "
            "not match the active PID namespace. Refusing to traverse numeric "
            "procfs paths or claim that separately-sessioned descendants can "
            "be cancelled safely.",
            capabilities=capabilities,
        )
    return capabilities


def _proc_start_time(pid: int) -> str:
    """Return Linux' immutable process start tick, or an empty fallback."""

    if not _procfs_matches_active_pid_namespace():
        return ""
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        # comm may contain spaces and parentheses.  Fields after the final ')'
        # start at stat field 3; starttime is field 22, hence index 19.
        tail = raw[raw.rfind(")") + 2 :].split()
        return str(tail[19])
    except Exception:
        return ""


def _direct_children(pid: int) -> set[int]:
    """Read children created by every thread of a Linux process."""

    children: set[int] = set()
    if not _procfs_matches_active_pid_namespace():
        return children
    try:
        task_root = Path(f"/proc/{int(pid)}/task")
        for child_file in task_root.glob("*/children"):
            try:
                children.update(
                    int(token)
                    for token in child_file.read_text(encoding="utf-8").split()
                    if token.isdigit()
                )
            except Exception:
                continue
    except Exception:
        pass
    return children


def _capture_process_tree(root_pid: int) -> list[ProcessIdentity]:
    if os.name != "posix":
        return []
    queue: list[tuple[int, int]] = [(int(root_pid), 0)]
    seen: set[int] = set()
    captured: list[ProcessIdentity] = []
    while queue:
        pid, depth = queue.pop(0)
        if pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        try:
            pgid = int(os.getpgid(pid))
        except (ProcessLookupError, PermissionError, OSError):
            continue
        captured.append(
            ProcessIdentity(
                pid=pid,
                pgid=pgid,
                start_time_ticks=_proc_start_time(pid),
                depth=depth,
            )
        )
        queue.extend((child, depth + 1) for child in _direct_children(pid))
    return captured


def _identity_alive(identity: ProcessIdentity) -> bool:
    if identity.pid <= 0 or identity.pid == os.getpid():
        return False
    try:
        os.kill(identity.pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    current_start = _proc_start_time(identity.pid)
    if identity.start_time_ticks and current_start != identity.start_time_ticks:
        return False
    # A zombie cannot execute or write any more data.  Numeric procfs paths are
    # consulted only when procfs is proven to use our active PID namespace.
    # Otherwise the same number may describe an unrelated host process.
    if _procfs_matches_active_pid_namespace():
        try:
            raw = Path(f"/proc/{identity.pid}/stat").read_text(encoding="utf-8")
            tail = raw[raw.rfind(")") + 2 :].split()
            if tail and tail[0] == "Z":
                return False
        except Exception:
            pass
    return True


def _signal_tree(identities: list[ProcessIdentity], sig: int) -> None:
    identities = [item for item in identities if _identity_alive(item)]
    if not identities:
        return
    own_pid = os.getpid()
    try:
        own_pgid = os.getpgrp()
    except Exception:  # pragma: no cover - non-POSIX fallback
        own_pgid = -1

    # A group signal is required for shells/SSH commands that fork without
    # retaining a direct Python-visible Popen.  Never signal our own group.
    # Only signal a group when its session/group leader is itself one of the
    # captured, still-identical processes.  This avoids acting on a raw PGID
    # that has been recycled for an unrelated process.
    owned_group_leaders = {
        item.pid
        for item in identities
        if item.pid == item.pgid and _identity_alive(item)
    }
    groups = sorted(
        {
            item.pgid
            for item in identities
            if item.pgid > 1
            and item.pgid != own_pgid
            and item.pid != own_pid
            and item.pgid in owned_group_leaders
        },
        reverse=True,
    )
    signalled_groups: set[int] = set()
    for pgid in groups:
        try:
            os.killpg(pgid, sig)
            signalled_groups.add(pgid)
        except (ProcessLookupError, PermissionError, OSError):
            continue

    # Processes that inherited the caller's group must be addressed by PID;
    # signalling that group would terminate the GUI/CLI itself.
    for item in sorted(identities, key=lambda row: row.depth, reverse=True):
        if item.pid == own_pid or item.pgid in signalled_groups:
            continue
        if not _identity_alive(item):
            continue
        try:
            os.kill(item.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            continue


def terminate_process_tree(
    proc: subprocess.Popen[Any],
    *,
    grace_s: float = 3.0,
    known_identities: list[ProcessIdentity] | None = None,
) -> dict[str, Any]:
    """Terminate ``proc`` plus descendants, including nested process groups.

    The returned diagnostic is intentionally small and contains no command or
    environment data.  On non-Linux platforms the capture naturally falls back
    to the direct process/process group.
    """

    root_pid = int(getattr(proc, "pid", 0) or 0)
    capabilities = process_tree_capabilities()
    root_completed = proc.poll() is not None
    seed_identities = list(known_identities or [])
    # A completed Popen is authoritative.  Never inspect its numeric PID after
    # completion because that PID may already belong to a different process.
    # Previously tracked descendants remain safe to terminate because their
    # immutable start identities are checked before every signal.
    if root_completed and not any(
        _identity_alive(item) for item in seed_identities
    ):
        return {
            "root_pid": root_pid,
            "captured_process_count": 0,
            "captured_process_group_count": 0,
            "remaining_process_count": 0,
            **capabilities,
        }
    identities = list(seed_identities)
    if not root_completed and root_pid > 0:
        identities.extend(_capture_process_tree(root_pid))
    if not identities and root_pid > 0 and not root_completed:
        try:
            identities = [
                ProcessIdentity(
                    pid=root_pid,
                    pgid=int(os.getpgid(root_pid)) if os.name == "posix" else -1,
                    start_time_ticks=_proc_start_time(root_pid),
                    depth=0,
                )
            ]
        except Exception:
            identities = []

    identity_by_key = {
        (item.pid, item.start_time_ticks): item for item in identities
    }

    def _recapture_owned_descendants() -> list[ProcessIdentity]:
        if os.name != "posix" or not _procfs_matches_active_pid_namespace():
            return list(identity_by_key.values())
        # Re-scan every still-owned process while TERM handlers run.  A helper
        # may create its workload only as shutdown begins; a one-shot snapshot
        # would miss that late nested session.
        scan_roots = [
            item.pid for item in list(identity_by_key.values())
            if _identity_alive(item)
        ]
        if proc.poll() is None and root_pid > 0 and root_pid not in scan_roots:
            scan_roots.insert(0, root_pid)
        for scan_root in scan_roots:
            for item in _capture_process_tree(scan_root):
                identity_by_key.setdefault(
                    (item.pid, item.start_time_ticks), item
                )
        return list(identity_by_key.values())

    if os.name == "posix":
        _signal_tree(identities, signal.SIGTERM)
    elif os.name == "nt" and proc.poll() is None:  # pragma: no cover
        try:
            subprocess.run(
                ["taskkill", "/PID", str(root_pid), "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=max(1.0, float(grace_s)),
            )
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    deadline = time.monotonic() + max(0.0, float(grace_s))
    while time.monotonic() < deadline:
        identities = _recapture_owned_descendants()
        # Reap the direct child promptly.  This also makes cancellation robust
        # in PID-namespace containers whose mounted /proc cannot expose the
        # child's zombie state.
        proc.poll()
        if not any(_identity_alive(item) for item in identities):
            break
        time.sleep(0.02)

    identities = _recapture_owned_descendants()
    remaining = [item for item in identities if _identity_alive(item)]
    if remaining:
        if os.name == "posix":
            _signal_tree(remaining, signal.SIGKILL)
        elif os.name == "nt":  # pragma: no cover
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(root_pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=2.0,
                )
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    # SIGKILL is asynchronous.  Keep reaping and rechecking the captured tree
    # until every owned process is unable to execute, rather than returning a
    # misleading success as soon as the direct Popen exits.
    kill_deadline = time.monotonic() + max(
        0.5, min(2.0, max(0.0, float(grace_s)) + 0.5)
    )
    while time.monotonic() < kill_deadline:
        proc.poll()
        identities = _recapture_owned_descendants()
        remaining = [item for item in identities if _identity_alive(item)]
        if not remaining:
            break
        if os.name == "posix":
            _signal_tree(remaining, signal.SIGKILL)
        time.sleep(0.02)

    try:
        proc.wait(timeout=max(0.2, min(1.0, float(grace_s) + 0.2)))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

    return {
        "root_pid": root_pid,
        "captured_process_count": len(identities),
        "captured_process_group_count": len(
            {item.pgid for item in identities if item.pgid > 0}
        ),
        "remaining_process_count": sum(
            1 for item in identities if _identity_alive(item)
        ),
        **capabilities,
    }


class ProcessTreeRegistry:
    """Thread-safe registry of direct children owned by one workflow session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._termination_lock = threading.Lock()
        # Key by the Popen object's registration token, never only by numeric
        # PID.  A completed/reaped PID can be reused while an older entry still
        # owns a separately-sessioned descendant.
        self._processes: dict[int, _RegisteredProcess] = {}
        self._last_reports: dict[int, dict[str, Any]] = {}
        # Python worker threads cannot be forcibly terminated.  A call-site
        # that exhausts its bounded cancellation grace records that uncertainty
        # here so the runner's final quiescence assertion must quarantine even
        # when every currently captured OS process has exited.
        self._cleanup_uncertainties: list[dict[str, Any]] = []
        self._cancelled = False
        self._cancel_grace_s = 3.0

    def record_cleanup_uncertainty(self, *, label: str, detail: str = "") -> None:
        """Permanently make this run's final process proof fail closed."""

        record = {
            "root_pid": 0,
            "label": str(label or "workflow-worker")[:160],
            "registration_token": 0,
            "known_alive_process_count": 0,
            "remaining_process_count": 1,
            "cleanup_uncertainty": True,
            "termination_error": str(
                detail or "bounded worker cancellation did not prove exit"
            )[:1000],
            **process_tree_capabilities(),
        }
        with self._lock:
            self._cancelled = True
            self._cleanup_uncertainties.append(record)

    @staticmethod
    def _root_identity(proc: subprocess.Popen[Any]) -> ProcessIdentity | None:
        if proc.poll() is not None:
            return None
        pid = int(getattr(proc, "pid", 0) or 0)
        for item in _capture_process_tree(pid):
            if item.pid == pid:
                return item
        if pid <= 0 or os.name != "posix":
            return None
        try:
            return ProcessIdentity(
                pid=pid,
                pgid=int(os.getpgid(pid)),
                start_time_ticks=_proc_start_time(pid),
                depth=0,
            )
        except (AttributeError, ProcessLookupError, PermissionError, OSError):
            return None

    @staticmethod
    def _same_registered_process(
        proc: subprocess.Popen[Any], identity: ProcessIdentity | None
    ) -> bool:
        if proc.poll() is not None:
            return False
        if identity is None or not identity.start_time_ticks:
            return True
        current = _proc_start_time(identity.pid)
        return bool(current and current == identity.start_time_ticks)

    def register(
        self, proc: subprocess.Popen[Any], *, label: str = ""
    ) -> dict[str, Any] | None:
        token = id(proc)
        identity = self._root_identity(proc)
        if proc.poll() is not None:
            return None
        known = {
            (item.pid, item.start_time_ticks): item
            for item in _capture_process_tree(int(proc.pid))
        }
        if identity is not None:
            known.setdefault((identity.pid, identity.start_time_ticks), identity)
        entry = _RegisteredProcess(
            registration_token=token,
            proc=proc,
            label=str(label or ""),
            root_identity=identity,
            known_identities=known,
            tracker_stop=threading.Event(),
        )

        def _track_descendants() -> None:
            while not entry.tracker_stop.wait(0.01):
                if proc.poll() is not None:
                    return
                captured = _capture_process_tree(int(proc.pid))
                if not captured:
                    continue
                with self._lock:
                    if self._processes.get(token) is not entry:
                        return
                    for item in captured:
                        entry.known_identities.setdefault(
                            (item.pid, item.start_time_ticks), item
                        )

        terminate_now = False
        grace_s = 3.0
        with self._lock:
            if self._cancelled:
                terminate_now = True
                grace_s = self._cancel_grace_s
                self._processes[token] = entry
            else:
                entry.tracker_thread = threading.Thread(
                    target=_track_descendants,
                    name=f"track-process-tree-{int(proc.pid)}",
                    daemon=True,
                )
                self._processes[token] = entry
                entry.tracker_thread.start()
        # Sticky cancellation closes the Popen -> register race: a process
        # created after the cancellation snapshot is killed before register()
        # returns to its owner.
        if terminate_now:
            try:
                report = self._terminate_entry(entry, grace_s=grace_s)
            except BaseException as exc:
                report = self._termination_exception_report(entry, exc)
            with self._lock:
                self._last_reports[token] = dict(report)
                if int(report.get("remaining_process_count") or 0) == 0:
                    if self._processes.get(token) is entry:
                        self._processes.pop(token, None)
                    self._last_reports.pop(token, None)
            if int(report.get("remaining_process_count") or 0) > 0:
                raise ProcessTreeCancellationError(
                    "A process registered after workflow cancellation did not "
                    "reach quiescence.",
                    reports=[report],
                    capabilities=process_tree_capabilities(),
                )
            return report
        return None

    def unregister(self, proc: subprocess.Popen[Any]) -> None:
        token = id(proc)
        tracker: threading.Thread | None = None
        cleanup_entry: _RegisteredProcess | None = None
        with self._lock:
            current = self._processes.get(token)
            if current is not None and current.proc is proc:
                root_alive = self._same_registered_process(
                    current.proc, current.root_identity
                )
                descendants_alive = any(
                    _identity_alive(item)
                    for item in current.known_identities.values()
                    if current.root_identity is None
                    or item != current.root_identity
                )
                if root_alive or descendants_alive:
                    cleanup_entry = current
                else:
                    self._processes.pop(token, None)
                    self._last_reports.pop(token, None)
                    current.tracker_stop.set()
                    tracker = current.tracker_thread
        if cleanup_entry is not None:
            # unregister() is an ownership hand-off, not permission for a live
            # root, inherited-pipe child or captured separately-sessioned
            # descendant to escape the workflow registry.  Retain the entry if bounded termination
            # cannot prove quiescence so the runner's final gate can fail hard.
            self.terminate_registered(proc, grace_s=0.5)
            return
        if tracker is not None and tracker is not threading.current_thread():
            tracker.join(timeout=0.2)

    def active_pids(self) -> list[int]:
        with self._lock:
            return sorted({
                int(entry.proc.pid)
                for entry in self._processes.values()
                if self._same_registered_process(
                    entry.proc, entry.root_identity
                )
            })

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @staticmethod
    def _entry_alive(entry: "_RegisteredProcess") -> bool:
        if ProcessTreeRegistry._same_registered_process(
            entry.proc, entry.root_identity
        ):
            return True
        return any(
            _identity_alive(item)
            for item in entry.known_identities.values()
            if entry.root_identity is None or item != entry.root_identity
        )

    @staticmethod
    def _terminate_entry(
        entry: "_RegisteredProcess", *, grace_s: float
    ) -> dict[str, Any]:
        entry.tracker_stop.set()
        tracker = entry.tracker_thread
        if tracker is not None and tracker is not threading.current_thread():
            tracker.join(timeout=0.2)
        report = terminate_process_tree(
            entry.proc,
            grace_s=grace_s,
            known_identities=list(entry.known_identities.values()),
        )
        report["label"] = entry.label
        report["registration_token"] = entry.registration_token
        return report

    @staticmethod
    def _termination_exception_report(
        entry: "_RegisteredProcess", exc: BaseException
    ) -> dict[str, Any]:
        return {
            "root_pid": int(getattr(entry.proc, "pid", 0) or 0),
            "label": entry.label,
            "registration_token": entry.registration_token,
            "captured_process_count": len(entry.known_identities),
            "captured_process_group_count": len(
                {
                    item.pgid
                    for item in entry.known_identities.values()
                    if item.pgid > 0
                }
            ),
            # An exception is uncertainty, never proof of successful cleanup.
            "remaining_process_count": 1,
            "termination_error": f"{type(exc).__name__}: {exc}",
            **process_tree_capabilities(),
        }

    @staticmethod
    def _termination_worker_join_budget_s(grace_s: float) -> float:
        """Bound one parallel cancellation pass, including the KILL/reap tail."""

        # terminate_process_tree() spends at most ``grace_s`` in its TERM loop,
        # followed by a bounded KILL/reap tail.  Keep a small scheduling margin,
        # but never let an unexpectedly stuck OS/procfs call retain the run lock.
        return max(1.0, max(0.0, float(grace_s)) + 4.0)

    def terminate_registered(
        self, proc: subprocess.Popen[Any], *, grace_s: float = 3.0
    ) -> dict[str, Any]:
        """Terminate one registered tree without closing the whole registry."""

        token = id(proc)
        with self._termination_lock:
            with self._lock:
                entry = self._processes.get(token)
                if entry is None or entry.proc is not proc:
                    return terminate_process_tree(proc, grace_s=grace_s)
            try:
                report = self._terminate_entry(entry, grace_s=grace_s)
            except BaseException as exc:
                report = self._termination_exception_report(entry, exc)
            with self._lock:
                self._last_reports[token] = dict(report)
                if (
                    self._processes.get(token) is entry
                    and int(report.get("remaining_process_count") or 0) == 0
                ):
                    self._processes.pop(token, None)
                    self._last_reports.pop(token, None)
            return report

    def terminate_all(self, *, grace_s: float = 3.0) -> list[dict[str, Any]]:
        # GUI, signal watcher, runner exception handling and runner finally may
        # all converge here.  Serialize them and make cancellation terminal.
        with self._termination_lock:
            with self._lock:
                self._cancelled = True
                self._cancel_grace_s = max(0.0, float(grace_s))
                rows_with_tokens: list[tuple[int, _RegisteredProcess]] = []
                for token, entry in list(self._processes.items()):
                    if not self._entry_alive(entry):
                        if self._processes.get(token) is entry:
                            self._processes.pop(token, None)
                            self._last_reports.pop(token, None)
                        continue
                    entry.tracker_stop.set()
                    rows_with_tokens.append((token, entry))

                rows = [entry for _token, entry in rows_with_tokens]

            reports: list[dict[str, Any] | None] = [None] * len(rows)

            def _terminate_one(
                index: int,
                entry: _RegisteredProcess,
            ) -> None:
                try:
                    reports[index] = self._terminate_entry(
                        entry, grace_s=grace_s
                    )
                except BaseException as exc:
                    reports[index] = self._termination_exception_report(entry, exc)

            workers = [
                threading.Thread(
                    target=_terminate_one,
                    args=(index, entry),
                    name=f"cancel-process-tree-{int(entry.proc.pid)}",
                    # A timed-out worker leaves its entry registered and forces
                    # the runner's durable quarantine.  It must not itself keep
                    # the application alive after that bounded hand-off.
                    daemon=True,
                )
                for index, entry in enumerate(rows)
            ]
            started_workers: list[tuple[int, threading.Thread]] = []
            for index, worker in enumerate(workers):
                try:
                    worker.start()
                except BaseException as exc:
                    reports[index] = self._termination_exception_report(
                        rows[index], exc
                    )
                else:
                    started_workers.append((index, worker))
            join_budget_s = self._termination_worker_join_budget_s(grace_s)
            join_deadline = time.monotonic() + join_budget_s
            for _index, worker in started_workers:
                worker.join(timeout=max(0.0, join_deadline - time.monotonic()))
            for index, worker in started_workers:
                if worker.is_alive() and reports[index] is None:
                    reports[index] = self._termination_exception_report(
                        rows[index],
                        TimeoutError(
                            "process-tree termination worker exceeded bounded "
                            f"cleanup budget ({join_budget_s:.3f}s)"
                        ),
                    )

            with self._lock:
                for index, (token, entry) in enumerate(rows_with_tokens):
                    current = self._processes.get(token)
                    report = reports[index]
                    if report is not None:
                        self._last_reports[token] = dict(report)
                    if (
                        current is entry
                        and report is not None
                        and int(report.get("remaining_process_count") or 0) == 0
                    ):
                        self._processes.pop(token, None)
                        self._last_reports.pop(token, None)
            with self._lock:
                uncertainties = [
                    dict(report) for report in self._cleanup_uncertainties
                ]
            return [
                dict(report) for report in reports if report is not None
            ] + uncertainties

    def survivor_reports(self) -> list[dict[str, Any]]:
        """Return sanitized diagnostics for every still-owned live tree."""

        survivors: list[dict[str, Any]] = []
        with self._lock:
            survivors.extend(
                dict(report) for report in self._cleanup_uncertainties
            )
            for token, entry in list(self._processes.items()):
                if not self._entry_alive(entry):
                    if self._processes.get(token) is entry:
                        self._processes.pop(token, None)
                        self._last_reports.pop(token, None)
                        entry.tracker_stop.set()
                    continue
                previous = dict(self._last_reports.get(token) or {})
                known_alive = sum(
                    1
                    for item in entry.known_identities.values()
                    if _identity_alive(item)
                )
                survivors.append(
                    {
                        **previous,
                        "root_pid": int(getattr(entry.proc, "pid", 0) or 0),
                        "label": entry.label,
                        "registration_token": entry.registration_token,
                        "known_alive_process_count": known_alive,
                        "remaining_process_count": max(
                            1,
                            int(previous.get("remaining_process_count") or 0),
                            known_alive,
                        ),
                        **process_tree_capabilities(),
                    }
                )
        return survivors

    def remaining_owned_count(self) -> int:
        """Count live registered roots or descendant-only owned trees."""

        return len(self.survivor_reports())

    def has_owned_processes(self) -> bool:
        return self.remaining_owned_count() > 0

    def is_quiescent(
        self, *, require_nested_session_safety: bool = False
    ) -> bool:
        capabilities = process_tree_capabilities()
        if require_nested_session_safety and not bool(
            capabilities.get("nested_session_cancellation_safe")
        ):
            return False
        return not self.survivor_reports()

    def assert_quiescent(
        self, *, require_nested_session_safety: bool = False
    ) -> None:
        """Raise unless no owned tree remains and required guarantees exist."""

        capabilities = process_tree_capabilities()
        reports = self.survivor_reports()
        capability_missing = bool(
            require_nested_session_safety
            and not capabilities.get("nested_session_cancellation_safe")
        )
        if not reports and not capability_missing:
            return
        reasons: list[str] = []
        if reports:
            reasons.append(f"{len(reports)} owned process tree(s) remain")
        if capability_missing:
            reasons.append(
                "nested-session cancellation cannot be verified in this PID namespace"
            )
        raise ProcessTreeCancellationError(
            "; ".join(reasons) + ".",
            reports=reports,
            capabilities=capabilities,
        )


@dataclass
class _RegisteredProcess:
    registration_token: int
    proc: subprocess.Popen[Any]
    label: str
    root_identity: ProcessIdentity | None
    known_identities: dict[tuple[int, str], ProcessIdentity]
    tracker_stop: threading.Event
    tracker_thread: threading.Thread | None = None
