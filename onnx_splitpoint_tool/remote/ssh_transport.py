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
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple


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

    def __init__(self, host: HostConfig, log: Optional[Callable[[str], None]] = None):
        self.host = host
        self._log = log

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

    # -------------------------
    # public API
    # -------------------------
    def test_connection(self, timeout_s: int = 10) -> Tuple[bool, str]:
        """Test SSH connectivity and return (ok, message).

        Message contains both stdout and stderr for easier debugging.
        """

        cmd = self._ssh_cmd("echo __SPLITPOINT_OK__ && uname -a")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except FileNotFoundError as e:
            return False, f"ssh not found: {e}"
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout_s}s"
        except Exception as e:
            return False, f"ssh failed: {e}"

        out_raw = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()

        # Keep a sentinel in the command so we can robustly detect success,
        # but do *not* show it in the popup (it's an internal marker).
        out_lines = [ln.strip() for ln in out_raw.splitlines() if ln.strip()]
        ok = proc.returncode == 0 and "__SPLITPOINT_OK__" in out_lines
        out_lines = [ln for ln in out_lines if ln != "__SPLITPOINT_OK__"]

        msg_lines = [
            f"remote: {self.host.user_host_pretty}",
            f"rc={proc.returncode}",
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
        cmd = self._ssh_cmd(bash_cmd, env=env)
        self._log_line(f"[ssh] $ {bash_cmd}")
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return int(proc.returncode), out
        except FileNotFoundError as e:
            return 127, f"ssh not found: {e}"
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s"
        except Exception as e:
            return 1, f"ssh failed: {e}"

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

        cmd = self._ssh_cmd(bash_cmd, env=env)
        self._log_line(f"[ssh-stream] $ {bash_cmd}")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            on_line(f"ssh not found: {e}")
            return 127
        except Exception as e:
            on_line(f"ssh failed: {e}")
            return 1

        start = time.time()
        assert proc.stdout is not None
        try:
            ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
            ts_re = re.compile(r"(?<!^)(?<!\n)(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

            for raw in proc.stdout:
                # raw is a line terminated by \n, but may contain \r progress updates / ANSI colors
                chunk = raw.replace("\r", "\n")
                chunk = ansi_re.sub("", chunk)
                # If progress output and another log line got concatenated, split before timestamps.
                chunk = ts_re.sub(r"\n\1", chunk)

                for line in chunk.splitlines():
                    on_line(line)

                if cancel_event and cancel_event.is_set():
                    on_line("[remote] cancel requested; terminating…")
                    proc.terminate()
                    break

                if timeout is not None and (time.time() - t0) > timeout:
                    on_line(f"[remote] timeout ({timeout}s) exceeded; terminating…")
                    proc.terminate()
                    break
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

        try:
            rc = proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            rc = 1
        return int(rc)

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
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return int(proc.returncode), out
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
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return int(proc.returncode), out
        except FileNotFoundError as e:
            return 127, f"scp not found: {e}"
        except subprocess.TimeoutExpired:
            return 124, f"timeout after {timeout}s"
        except Exception as e:
            return 1, f"scp failed: {e}"
