"""SSH/SCP transport helpers.

We intentionally use the *system* ssh/scp binaries instead of a Python SSH
implementation (e.g. paramiko) because:

* it automatically reuses the user's existing SSH config, agent and keys
* it works cross-platform (Linux, Windows OpenSSH) without extra deps
* it keeps this tool lightweight

This module is used by the remote benchmarking flow.
"""

from __future__ import annotations

from onnx_splitpoint_tool.process_control import budget_controller_work

import os
import json
import re
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path
from ..log_utils import redact_diagnostic, compact_diagnostic_cause
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Tuple

from ..process_control import ProcessTreeRegistry, current_process_registry, _proc_start_time
from .process_lease import (
    RemoteProcessLeaseConfigurationError,
    RemoteProcessLeaseLaunchRejected,
    RemoteProcessLeaseOperation,
    RemoteProcessLeaseRegistry,
    RemoteProcessLeaseScope,
    harden_remote_process_lease_ssh_argv,
    validate_remote_process_lease_ssh_prefix,
)


def shell_quote(s: str) -> str:
    """Quote a string for safe use in a remote bash command.

    We use single quotes and escape embedded single quotes using the classic
    POSIX pattern:  'foo'"'"'bar'
    """

    if s == "":
        return "''"
    return "'" + s.replace("'", "'\"'\"'") + "'"


@dataclass
class HostConfig:
    """Serializable remote host configuration (no secrets)."""

    id: str
    label: str
    host: str
    user: str = ""
    port: int = 22
    remote_base_dir: str = "~/splitpoint_runs"
    ssh_extra_args: str = ""  # e.g. "-o StrictHostKeyChecking=accept-new"

    @property
    def user_host(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @property
    def user_host_pretty(self) -> str:
        """Human‑friendly host string including port."""
        return f"{self.user_host}:{int(self.port or 22)}"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, object]) -> "HostConfig":
        # Be resilient to missing/new keys.
        return HostConfig(
            id=str(d.get("id", "")),
            label=str(d.get("label", "")),
            host=str(d.get("host", "")),
            user=str(d.get("user", "")),
            port=int(d.get("port", 22) or 22),
            remote_base_dir=str(d.get("remote_base_dir", "~/splitpoint_runs")),
            ssh_extra_args=str(d.get("ssh_extra_args", "")),
        )


class SSHTransport:
    """Thin wrapper around system ssh/scp.

    Public API intentionally matches what benchmark.remote_run expects.
    """

    def __init__(
        self,
        host: HostConfig,
        log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[object] = None,
        remote_lease_scope: Optional[RemoteProcessLeaseScope] = None,
        remote_lease_registry: Optional[RemoteProcessLeaseRegistry] = None,
    ):
        self.host = host
        self._read_only_state = threading.local()
        self._log = log
        self._cancel_event = cancel_event
        self._remote_lease_scope = remote_lease_scope
        self._remote_lease_registry = remote_lease_registry
        self.diagnostics_dir = None

    # -------------------------
    # command builders
    # -------------------------
    def _extra_args(self) -> List[str]:
        extra = (self.host.ssh_extra_args or "").strip()
        if not extra:
            return []
        try:
            return shlex.split(extra)
        except Exception:
            # If parsing fails, fall back to a naive split.
            return extra.split()

    def _ssh_base(self) -> List[str]:
        cmd: List[str] = ["ssh", "-p", str(int(self.host.port or 22))]
        cmd += self._extra_args()
        cmd.append(self.host.user_host)
        if self._remote_lease_scope is not None:
            return harden_remote_process_lease_ssh_argv(cmd)
        return cmd

    def _scp_base(self) -> List[str]:
        cmd: List[str] = ["scp", "-P", str(int(self.host.port or 22))]
        cmd += self._extra_args()
        return cmd

    def _ssh_cmd(self, bash_cmd: str, env: Optional[dict] = None) -> List[str]:
        # Compose env exports inside the remote bash -lc context.
        if env:
            exports = []
            for k, v in env.items():
                if v is None:
                    continue
                exports.append(f"export {k}={shell_quote(str(v))}")
            if exports:
                bash_cmd = "; ".join(exports) + "; " + bash_cmd
        return self._ssh_base() + ["bash", "-lc", shell_quote(bash_cmd)]

    # -------------------------
    # helpers
    # -------------------------
    def _log_line(self, s: str) -> None:
        if self._log:
            try:
                self._log(s)
            except Exception:
                pass

    def _diagnostic(self, report: dict, *, path=None) -> str:
        destination = Path(path) if path else Path(self.diagnostics_dir or (Path.home() / ".onnx_splitpoint_tool" / "logs" / "diagnostics")) / ("ssh_" + uuid.uuid4().hex + ".json")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(redact_diagnostic(report), indent=2, ensure_ascii=False), encoding="utf-8")
            return str(destination)
        except OSError as exc:
            self._log_line(f"[warn] SSH-Diagnose nicht schreibbar: {destination} ({type(exc).__name__})")
            return "unavailable"

    def _cancel_requested(self, explicit: Optional[object] = None) -> bool:
        event = explicit if explicit is not None else self._cancel_event
        try:
            return bool(
                (event is not None and event.is_set())
                or (
                    self._remote_lease_registry is not None
                    and self._remote_lease_registry.cancelled
                )
            )
        except Exception:
            return False

    def _begin_remote_lease(
        self,
        *,
        label: str,
    ) -> Optional[RemoteProcessLeaseOperation]:
        scope = self._remote_lease_scope
        if scope is None:
            return None
        control_prefix = self._ssh_base()
        validate_remote_process_lease_ssh_prefix(control_prefix)
        operation = scope.operation(
            label=label,
            control_argv_prefix=control_prefix,
        )
        if self._remote_lease_registry is not None:
            self._remote_lease_registry.register(operation)
        return operation

    def _finish_remote_lease(
        self,
        operation: Optional[RemoteProcessLeaseOperation],
        *,
        abnormal: bool,
    ) -> bool:
        if operation is None:
            return True
        cleanup_ok = True
        # A non-zero local SSH result cannot prove that a detached remote
        # descendant stopped.  Resolve it through the exact control path.
        if abnormal:
            cleanup_ok = bool(operation.cancel_remote(grace_s=3.0).get("ok"))
            if not cleanup_ok and self._remote_lease_registry is not None:
                self._remote_lease_registry.poison()
        if not abnormal or cleanup_ok:
            if self._remote_lease_registry is not None:
                self._remote_lease_registry.unregister(operation)
        return cleanup_ok

    def _run_capture(
        self,
        cmd: List[str],
        *,
        timeout: Optional[int],
        remote_operation: Optional[RemoteProcessLeaseOperation] = None,
        diagnostics: Optional[dict] = None,
    ) -> Tuple[int, str]:
        """Run SSH/SCP with cancellation polling even when output is silent."""

        entered = time.monotonic()
        diagnostic = diagnostics if diagnostics is not None else {}
        diagnostic.update(command_argv=list(cmd), timeout_s=timeout,
                          owner_pid=os.getpid(), phase="local_admission",
                          remote_completion_proven=False)
        registry = current_process_registry() or ProcessTreeRegistry()

        def cancellation_requested() -> bool:
            return self._cancel_requested() or registry.cancelled

        if cancellation_requested():
            if remote_operation is not None:
                remote_operation.cancel_remote(grace_s=3.0)
            return 130, "cancelled before process start"
        diagnostic["phase"] = "local_spawn"
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        diagnostic.update(local_pid=proc.pid, local_start_time_ticks=_proc_start_time(proc.pid), local_pgid=os.getpgid(proc.pid) if os.name == "posix" else None,
                          spawn_elapsed_s=time.monotonic() - entered, phase="local_registration")
        registry.register(proc, label="ssh-capture")
        chunks: List[str] = []
        reader_done = threading.Event()

        def _read_output() -> None:
            assert proc.stdout is not None
            try:
                while True:
                    chunk = proc.stdout.read(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
            except Exception:
                pass
            finally:
                reader_done.set()

        reader = threading.Thread(
            target=_read_output,
            name=f"ssh-capture-{proc.pid}",
            daemon=True,
        )
        reader.start()
        started = time.monotonic()
        diagnostic.update(phase="process_wait", pre_wait_elapsed_s=started - entered)
        forced_rc: Optional[int] = None
        try:
            while True:
                if cancellation_requested():
                    forced_rc = 130
                    diagnostic["phase"] = "probe_cancelled"
                    if remote_operation is not None:
                        remote_operation.cancel_remote(grace_s=3.0)
                    registry.terminate_registered(proc, grace_s=3.0)
                    break
                if (
                    timeout is not None
                    and time.monotonic() - started > float(timeout)
                ):
                    forced_rc = 124
                    diagnostic["phase"] = "pipe_drain_timeout" if proc.poll() is not None else "process_wait_timeout"
                    if remote_operation is not None:
                        remote_operation.cancel_remote(grace_s=3.0)
                    registry.terminate_registered(proc, grace_s=3.0)
                    break
                if proc.poll() is not None:
                    # Use the existing bounded reader-join allowance before
                    # classifying EOF as missing. Under host load the reader
                    # can finish after root exit without any surviving child.
                    # A pipe still held open after this wait remains an error.
                    diagnostic["phase"] = "pipe_drain"
                    if reader_done.wait(timeout=1.0):
                        break
                    diagnostic["phase"] = "pipe_drain_incomplete"
                    registry.terminate_registered(proc, grace_s=0.5)
                    forced_rc = 70
                    break
                time.sleep(0.05)
            try:
                rc = int(proc.wait(timeout=5))
            except Exception:
                registry.terminate_registered(proc, grace_s=0.2)
                rc = 1
                diagnostic["phase"] = "process_exit_unknown"
        except BaseException:
            if remote_operation is not None:
                remote_operation.cancel_remote(grace_s=0.5)
            registry.terminate_registered(proc, grace_s=0.5)
            raise
        finally:
            reader.join(timeout=1.0)
            try:
                if not reader.is_alive() and proc.stdout is not None:
                    proc.stdout.close()
            except Exception:
                pass
            diagnostic.update(local_exit_code=proc.poll(), reader_finished=not reader.is_alive(),
                              elapsed_s=time.monotonic() - entered)
            diagnostic["local_root_and_pipe_completion_proven"] = proc.poll() is not None and not reader.is_alive()
            registry.unregister(proc)
            diagnostic["local_tree_survivors"] = [row for row in registry.survivor_reports()
                                                  if row.get("root_pid") in (0, proc.pid)]
            diagnostic["local_completion_proven"] = bool(
                diagnostic["local_root_and_pipe_completion_proven"] and not diagnostic["local_tree_survivors"])
            if not diagnostic["local_completion_proven"] and forced_rc is None:
                forced_rc = 70
                diagnostic["phase"] = "process_completion_unknown"
            if forced_rc is None:
                diagnostic["phase"] = "completed"
            diagnostic["returncode"] = forced_rc if forced_rc is not None else locals().get("rc")
        return int(forced_rc if forced_rc is not None else rc), "".join(chunks)

    # -------------------------
    # public API
    # -------------------------
    def test_connection(self, timeout_s: int = 10) -> Tuple[bool, str]:
        """Test SSH connectivity and return (ok, message).

        Message contains both stdout and stderr for easier debugging.
        """

        cmd = self._ssh_cmd("echo __SPLITPOINT_OK__ && uname -a")
        try:
            rc, combined = self._run_capture(cmd, timeout=timeout_s)
        except FileNotFoundError as e:
            return False, f"ssh not found: {e}"
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s}s"
        except Exception as e:
            return False, f"ssh failed: {e}"

        out_raw = combined.strip()
        err = ""

        # Keep a sentinel in the command so we can robustly detect success,
        # but do *not* show it in the popup (it's an internal marker).
        out_lines = [ln.strip() for ln in out_raw.splitlines() if ln.strip()]
        ok = rc == 0 and "__SPLITPOINT_OK__" in out_lines
        out_lines = [ln for ln in out_lines if ln != "__SPLITPOINT_OK__"]

        msg_lines = [
            f"remote: {self.host.user_host_pretty}",
            f"rc={rc}",
        ]
        if out_lines:
            # Usually the first line is `uname -a`.
            msg_lines.append(f"uname: {out_lines[0]}")
            if len(out_lines) > 1:
                msg_lines += ["stdout:"] + out_lines
        else:
            msg_lines.append("stdout: <empty>")
        if err:
            msg_lines += ["stderr:", err]
        return ok, "\n".join(msg_lines)

    def resolve_path(self, remote_path: str, timeout_s: int = 10) -> str:
        """Resolve a remote path to an absolute, expanded POSIX path.

        We expand '~' on the remote side (so it uses the *remote* user's HOME)
        and convert to an absolute path. This is important because Python
        itself does not expand '~' in file paths.
        """

        # Use an env var to avoid quoting issues if the path contains spaces
        # or other special characters.
        cmd = (
            "python3 -c \"import os; "
            "p=os.environ.get('SPLITPOINT_PATH',''); "
            "print(os.path.abspath(os.path.expanduser(p)))\""
        )
        rc, out = self.run(cmd, timeout_s=timeout_s, env={"SPLITPOINT_PATH": remote_path})
        if rc != 0:
            raise RuntimeError(f"resolve_path failed (rc={rc})\n{out}")

        # Be robust to extra output: return the last non-empty line.
        for line in reversed((out or "").splitlines()):
            line = line.strip()
            if line:
                return line
        return remote_path

    def resolve_path_read_only(self, remote_path: str, timeout_s: int = 10) -> str:
        """Resolve a remote path without creating a remote lease artefact.

        Storage admission has to run before *any* remote filesystem mutation.
        A normal leased ``run()`` cannot satisfy that ordering because its
        launcher publishes control files below ``/tmp``.  This narrowly scoped
        helper executes only the path-expansion probe through the unleased,
        capture-only transport.
        """

        cmd = (
            "python3 -c \"import os; "
            "p=os.environ.get('SPLITPOINT_PATH',''); "
            "print(os.path.abspath(os.path.expanduser(p)))\""
        )
        rc, out = self.run_read_only(
            cmd,
            timeout_s=timeout_s,
            env={"SPLITPOINT_PATH": remote_path},
        )
        if rc != 0:
            raise RuntimeError(f"resolve_path_read_only failed (rc={rc})\n{out}")
        for line in reversed((out or "").splitlines()):
            line = line.strip()
            if line:
                return line
        return remote_path

    @property
    def read_only_diagnostics(self) -> dict:
        """Last probe evidence for this caller thread; never shared across targets."""
        return dict(getattr(self._read_only_state, "report", {}) or {})

    def run_read_only(
        self,
        bash_cmd: str,
        timeout: Optional[int] = None,
        timeout_s: Optional[int] = None,
        env: Optional[dict] = None,
    ) -> Tuple[int, str]:
        """Run non-mutating admission without a remote lease or automatic retry.

        Local SSH exit does not prove the end of unleased remote work.
        """
        if timeout is None:
            timeout = timeout_s
        cmd = self._ssh_cmd(bash_cmd, env=env)
        report = {"target_id": self.host.id, "command": bash_cmd,
                  "lease": "not_created_read_only", "phase": "local_spawn",
                  "owner_pid": os.getpid(), "timeout_s": timeout,
                  "remote_completion_proven": False, "retry_permitted": False}
        self._read_only_state.report = report
        started = time.monotonic()
        diag = self._diagnostic(report)
        self._log_line(f"[ssh-read-only] setup={self.host.id} gestartet · Diagnose: {diag}")
        try:
            result = self._run_capture(cmd, timeout=timeout, diagnostics=report)
        except FileNotFoundError as exc:
            result = (127, f"ssh not found: {exc}")
        except subprocess.TimeoutExpired:
            result = (124, f"timeout after {timeout}s")
        except Exception as exc:
            result = (1, f"ssh failed: {type(exc).__name__}")
        report["returncode"] = result[0]
        report["output_tail"] = result[1][-12000:]
        if diag != "unavailable":
            self._diagnostic(dict(report, output=result[1]), path=diag)
        self._log_line(f"[ssh-read-only-result] setup={self.host.id} rc={result[0]} · {time.monotonic()-started:.1f}s · phase={report.get('phase')} · Diagnose: {diag}" + (" · " + compact_diagnostic_cause(result[1]) if result[0] else ""))
        return result

    def run(
        self,
        bash_cmd: str,
        timeout: Optional[int] = None,
        timeout_s: Optional[int] = None,
        env: Optional[dict] = None,
    ) -> Tuple[int, str]:
        """Run a remote bash command and return (rc, combined_output)."""

        if timeout is None:
            timeout = timeout_s
        try:
            operation = self._begin_remote_lease(label="ssh-capture")
        except RemoteProcessLeaseLaunchRejected as exc:
            return exc.returncode, str(exc)
        except RemoteProcessLeaseConfigurationError as exc:
            return 64, f"leased SSH configuration rejected: {exc}"
        remote_command = (
            operation.wrap_remote_command(bash_cmd)
            if operation is not None
            else bash_cmd
        )
        cmd = self._ssh_cmd(remote_command, env=env)
        started = time.monotonic()
        diag = self._diagnostic({"target_id": self.host.id, "command": bash_cmd, "command_argv": cmd})
        self._log_line(f"[ssh] setup={self.host.id} gestartet · Diagnose: {diag}")
        result: Tuple[int, str]
        try:
            result = self._run_capture(
                cmd,
                timeout=timeout,
                remote_operation=operation,
            )
        except FileNotFoundError as e:
            result = (127, f"ssh not found: {e}")
        except subprocess.TimeoutExpired:
            result = (124, f"timeout after {timeout}s")
        except Exception as e:
            result = (1, f"ssh failed: {type(e).__name__}")
        finally:
            # ``result`` is unavailable only for a BaseException, which must
            # also resolve the remote lease fail-closed.
            abnormal = "result" not in locals() or int(result[0]) != 0
            cleanup_proven = self._finish_remote_lease(
                operation, abnormal=abnormal
            )
        if abnormal and not cleanup_proven:
            result = (
                70,
                str(result[1])
                + "\nremote cleanup unproven; lease session poisoned",
            )
        if diag != "unavailable":
            self._diagnostic({"target_id": self.host.id, "command": bash_cmd, "command_argv": cmd,
                              "returncode": result[0], "output": result[1],
                              "cleanup_proven": cleanup_proven}, path=diag)
        self._log_line(f"[ssh-result] setup={self.host.id} rc={result[0]} · {time.monotonic()-started:.1f}s · Diagnose: {diag}" + (" · " + compact_diagnostic_cause(result[1]) if result[0] else ""))
        return result

    def run_streaming(
        self,
        bash_cmd: str,
        on_line: Callable[[str], None],
        timeout: Optional[int] = None,
        timeout_s: Optional[int] = None,
        env: Optional[dict] = None,
        cancel_event: Optional[object] = None,
    ) -> int:
        """Run a remote command and stream combined stdout/stderr line-by-line."""

        if timeout is None:
            timeout = timeout_s

        registry = current_process_registry() or ProcessTreeRegistry()

        def cancellation_requested() -> bool:
            return self._cancel_requested(cancel_event) or registry.cancelled

        if cancellation_requested():
            on_line("[remote] cancelled before process start")
            return 130

        try:
            operation = self._begin_remote_lease(label="ssh-stream")
        except RemoteProcessLeaseLaunchRejected as exc:
            on_line(str(exc))
            return exc.returncode
        except RemoteProcessLeaseConfigurationError as exc:
            on_line(f"leased SSH configuration rejected: {exc}")
            return 64
        remote_command = (
            operation.wrap_remote_command(bash_cmd)
            if operation is not None
            else bash_cmd
        )
        cmd = self._ssh_cmd(remote_command, env=env)
        diag = self._diagnostic({"target_id": self.host.id, "command": bash_cmd, "command_argv": cmd})
        self._log_line(f"[ssh-stream] setup={self.host.id} gestartet · Diagnose: {diag}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=(os.name == "posix"),
            )
        except FileNotFoundError as e:
            cleanup_proven = self._finish_remote_lease(
                operation, abnormal=True
            )
            on_line(f"ssh not found: {e}")
            return 127 if cleanup_proven else 70
        except Exception as e:
            cleanup_proven = self._finish_remote_lease(
                operation, abnormal=True
            )
            on_line(f"ssh failed: {type(e).__name__} · Diagnose: {diag}")
            return 1 if cleanup_proven else 70

        registry.register(proc, label="ssh-stream")

        start = time.time()
        timed_out = False
        cancelled = False
        assert proc.stdout is not None
        output_queue: Queue[str | None] = Queue()

        def _read_output() -> None:
            assert proc.stdout is not None
            try:
                for raw_line in proc.stdout:
                    output_queue.put(raw_line)
            finally:
                output_queue.put(None)

        reader = threading.Thread(
            target=_read_output,
            name=f"ssh-stream-{proc.pid}",
            daemon=True,
        )
        reader.start()
        pipe_drain_incomplete = False
        try:
            ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
            ts_re = re.compile(r"(?<!^)(?<!\n)(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
            eof = False
            root_exit_seen: Optional[float] = None
            while True:
                try:
                    raw = output_queue.get(timeout=0.1)
                except Empty:
                    raw = ""
                if raw is None:
                    eof = True
                elif raw:
                    # Progress output can contain CR updates, ANSI colour and
                    # concatenated timestamped log records.
                    chunk = ansi_re.sub("", raw.replace("\r", "\n"))
                    chunk = ts_re.sub(r"\n\1", chunk)
                    for line in chunk.splitlines():
                        on_line(line)

                if cancellation_requested():
                    cancelled = True
                    on_line("[remote] cancel requested; terminating…")
                    try:
                        if operation is not None:
                            operation.cancel_remote(grace_s=3.0)
                    finally:
                        registry.terminate_registered(proc, grace_s=3.0)
                    # Termination is bounded and fail-closed.  Do not spin
                    # forever if a surviving process refuses to exit; retain
                    # it in the shared registry for the runner quarantine gate.
                    break
                if (
                    timeout is not None
                    and (time.time() - start) > timeout
                    and (proc.poll() is None or reader.is_alive())
                ):
                    timed_out = True
                    on_line(f"[remote] timeout ({timeout}s) exceeded; terminating…")
                    try:
                        if operation is not None:
                            operation.cancel_remote(grace_s=3.0)
                    finally:
                        registry.terminate_registered(proc, grace_s=3.0)
                    break
                if proc.poll() is not None:
                    if eof and output_queue.empty():
                        break
                    if root_exit_seen is None:
                        root_exit_seen = time.monotonic()
                    # A finished reader has queued EOF: drain its finite
                    # output even when the observer is slower than 250 ms.
                    if reader.is_alive() and time.monotonic() - root_exit_seen >= 0.25:
                        pipe_drain_incomplete = True
                        on_line('[remote] pipe_drain_incomplete: inherited output pipe remained open')
                        registry.terminate_registered(proc, grace_s=0.5)
                        break
        except BaseException:
            if operation is not None:
                operation.cancel_remote(grace_s=0.5)
            registry.terminate_registered(proc, grace_s=0.5)
            self._finish_remote_lease(operation, abnormal=True)
            raise
        finally:
            reader.join(timeout=1.0)
            try:
                if not reader.is_alive():
                    proc.stdout.close()
            except Exception:
                pass
            registry.unregister(proc)

        try:
            rc = proc.wait(timeout=5)
        except Exception:
            registry.terminate_registered(proc, grace_s=0.2)
            rc = 1

        final_rc = 130 if cancelled else 124 if timed_out else 70 if pipe_drain_incomplete else int(rc)
        cleanup_proven = self._finish_remote_lease(
            operation, abnormal=final_rc != 0
        )
        if final_rc != 0 and not cleanup_proven:
            on_line("[remote] cleanup unproven; lease session poisoned")
            return 70
        return final_rc

    @budget_controller_work("transfer")
    def scp_upload(
        self,
        local_path: str,
        remote_path: str,
        recursive: bool = False,
        timeout: Optional[int] = None,
        timeout_s: Optional[int] = None,
    ) -> Tuple[int, str]:
        """Upload file/dir to remote (rc, combined_output)."""

        if timeout is None:
            timeout = timeout_s
        cmd = self._scp_base()
        if recursive:
            cmd.append("-r")

        dest = f"{self.host.user_host}:{remote_path}"
        cmd += [local_path, dest]
        self._log_line(f"[scp-upload] {local_path} -> {dest}")
        try:
            return self._run_capture(cmd, timeout=timeout)
        except FileNotFoundError as e:
            return 127, f"scp not found: {e}"
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s"
        except Exception as e:
            return 1, f"scp failed: {e}"

    @budget_controller_work("transfer")
    def scp_download(
        self,
        remote_path: str,
        local_path: str,
        recursive: bool = False,
        timeout: Optional[int] = None,
        timeout_s: Optional[int] = None,
    ) -> Tuple[int, str]:
        """Download file/dir from remote (rc, combined_output)."""

        if timeout is None:
            timeout = timeout_s
        cmd = self._scp_base()
        if recursive:
            cmd.append("-r")

        src = f"{self.host.user_host}:{remote_path}"
        try:
            os.makedirs(os.path.dirname(os.path.abspath(local_path)) or ".", exist_ok=True)
        except Exception:
            pass

        cmd += [src, local_path]
        self._log_line(f"[scp-download] {src} -> {local_path}")
        try:
            return self._run_capture(cmd, timeout=timeout)
        except FileNotFoundError as e:
            return 127, f"scp not found: {e}"
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s"
        except Exception as e:
            return 1, f"scp failed: {e}"
