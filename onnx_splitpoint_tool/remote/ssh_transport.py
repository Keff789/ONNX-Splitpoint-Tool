"""SSH/SCP transport helpers.

We intentionally use the *system* ssh/scp binaries instead of a Python SSH
implementation (e.g. paramiko) because:

* it automatically reuses the user's existing SSH config, agent and keys
* it works cross-platform (Linux, Windows OpenSSH) without extra deps
* it keeps this tool lightweight

This module is used by the remote benchmarking flow.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from queue import Empty, Queue
from typing import Callable, Dict, List, Optional, Tuple

from ..process_control import ProcessTreeRegistry, current_process_registry
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
        self._log = log
        self._cancel_event = cancel_event
        self._remote_lease_scope = remote_lease_scope
        self._remote_lease_registry = remote_lease_registry

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
    ) -> Tuple[int, str]:
        """Run SSH/SCP with cancellation polling even when output is silent."""

        registry = current_process_registry() or ProcessTreeRegistry()

        def cancellation_requested() -> bool:
            return self._cancel_requested() or registry.cancelled

        if cancellation_requested():
            if remote_operation is not None:
                remote_operation.cancel_remote(grace_s=3.0)
            return 130, "cancelled before process start"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name == "posix"),
        )
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
        forced_rc: Optional[int] = None
        try:
            root_exit_seen: Optional[float] = None
            while True:
                if cancellation_requested():
                    forced_rc = 130
                    if remote_operation is not None:
                        remote_operation.cancel_remote(grace_s=3.0)
                    registry.terminate_registered(proc, grace_s=3.0)
                    break
                if (
                    timeout is not None
                    and time.monotonic() - started > float(timeout)
                ):
                    forced_rc = 124
                    if remote_operation is not None:
                        remote_operation.cancel_remote(grace_s=3.0)
                    registry.terminate_registered(proc, grace_s=3.0)
                    break
                if proc.poll() is not None:
                    if reader_done.is_set():
                        break
                    if root_exit_seen is None:
                        root_exit_seen = time.monotonic()
                    if time.monotonic() - root_exit_seen >= 0.25:
                        registry.terminate_registered(proc, grace_s=0.5)
                        break
                time.sleep(0.05)
            try:
                rc = int(proc.wait(timeout=5))
            except Exception:
                registry.terminate_registered(proc, grace_s=0.2)
                rc = 1
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
            registry.unregister(proc)
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

    def run_read_only(
        self,
        bash_cmd: str,
        timeout: Optional[int] = None,
        timeout_s: Optional[int] = None,
        env: Optional[dict] = None,
    ) -> Tuple[int, str]:
        """Run a non-mutating admission probe without a remote lease.

        Callers must use this only for commands that are themselves read-only.
        Bypassing the lease is intentional here: lease publication writes
        remote control files and would invalidate a before-first-mutation
        storage check.
        """

        if timeout is None:
            timeout = timeout_s
        cmd = self._ssh_cmd(bash_cmd, env=env)
        self._log_line(f"[ssh-read-only] $ {bash_cmd}")
        try:
            return self._run_capture(cmd, timeout=timeout)
        except FileNotFoundError as exc:
            return 127, f"ssh not found: {exc}"
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s"
        except Exception as exc:
            return 1, f"ssh failed: {exc}"

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
        self._log_line(f"[ssh] $ {bash_cmd}")
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
            result = (1, f"ssh failed: {e}")
        finally:
            # ``result`` is unavailable only for a BaseException, which must
            # also resolve the remote lease fail-closed.
            abnormal = "result" not in locals() or int(result[0]) != 0
            cleanup_proven = self._finish_remote_lease(
                operation, abnormal=abnormal
            )
        if abnormal and not cleanup_proven:
            return (
                70,
                str(result[1])
                + "\nremote cleanup unproven; lease session poisoned",
            )
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
        self._log_line(f"[ssh-stream] $ {bash_cmd}")
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
            on_line(f"ssh failed: {e}")
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
                    if time.monotonic() - root_exit_seen >= 0.25:
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

        final_rc = 130 if cancelled else 124 if timed_out else int(rc)
        cleanup_proven = self._finish_remote_lease(
            operation, abnormal=final_rc != 0
        )
        if final_rc != 0 and not cleanup_proven:
            on_line("[remote] cleanup unproven; lease session poisoned")
            return 70
        return final_rc

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
