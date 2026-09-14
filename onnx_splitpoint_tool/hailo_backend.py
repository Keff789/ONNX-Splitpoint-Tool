"""Hailo backend helpers (optional).

This module is intentionally *optional*: the main tool should run without any
Hailo SDK installed. All imports of `hailo_sdk_client` happen at call time.

Current use cases
-----------------
- "Parse-only" feasibility check: can the given ONNX be translated by the Hailo
  DFC/SDK (i.e., `translate_onnx_model`)?

The goal is not to fully compile to HEF in the split-ranking stage, but to
quickly reject candidates that cannot be parsed/translated at all.
"""

from __future__ import annotations

from .config_values import parse_config_bool

import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import sys
import time
import tempfile
import logging
import platform
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
try:  # ONNX is required for actual graph translation, but cache-only lookup remains usable without it.
    import onnx  # type: ignore
    from onnx import AttributeProto, helper  # type: ignore
except Exception:  # pragma: no cover - optional in lightweight report/test environments
    onnx = None  # type: ignore
    AttributeProto = Any  # type: ignore
    helper = None  # type: ignore

# Optional (pure python) helper to resolve multiple DFC versions (Hailo-8 vs Hailo-10)
from .hailo.backend_mode import auto_prefers_subprocess, normalize_hailo_backend, subprocess_backend_for_platform
from .cache_verify_policy import (
    cache_miss_blocked_message,
    compiler_dispatch_forbidden,
)
from .filesystem_admission import inspect_write_target
from .runners.backends.hailo_utils import get_dfc_manager
from .process_control import (
    current_process_registry,
    terminate_process_tree,
)
from .remote.process_lease import (
    REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S,
    current_remote_process_registry,
    journaled_ssh_wrapper_argv,
)
from .preprocessing_contract import (
    canonical_image_preprocessing_contract,
    normalize_image_task,
    prepare_rgb_uint8_image,
    preprocessing_contract_sha256,
    resolve_image_preprocessing_contract,
    target_hw_from_shape,
)
from .hailo_timeout_policy import (
    is_hailo_timeout_unlimited,
    parse_hailo_timeout_seconds,
)


log = logging.getLogger(__name__)

# Successful component target checks are reused in this process, keyed by
# selected venv/component stat identity and actual GPU UUID/architecture.
# This is bounded transient state, never a model/cache hash or persistent file.
_HAILO_COMPILER_PROBE_CACHE: Dict[Any, Any] = {}


def _run_owned_subprocess(
    args: Any,
    *,
    input: Any = None,
    capture_output: bool = False,
    timeout: float | None = None,
    check: bool = False,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Use workflow ownership for otherwise synchronous Hailo helpers.

    Outside an Evaluation Workflow this is exactly ``subprocess.run``.  During
    a workflow the child gets its own process group, is registered before the
    blocking communicate call, and is therefore interruptible by GUI/CLI
    cancellation even while a DFC probe or compiler is completely silent.
    """

    registry = current_process_registry()
    remote_registry = current_remote_process_registry()
    raw_args = [str(value) for value in args] if isinstance(args, (list, tuple)) else []
    if (
        remote_registry is not None
        and raw_args
        and os.path.basename(raw_args[0]).lower() in {"ssh", "ssh.exe"}
    ):
        lease_env = remote_registry.journal_environment()
        merged_env = dict(os.environ)
        if kwargs.get("env") is not None:
            merged_env.update(
                {str(key): str(value) for key, value in kwargs["env"].items()}
            )
        merged_env.update(lease_env)
        kwargs["env"] = merged_env
        args = journaled_ssh_wrapper_argv(
            raw_args,
            label="hailo-activation-proxy",
            env=merged_env,
            timeout_s=timeout,
        )
        if timeout is not None:
            timeout = float(timeout) + REMOTE_LEASE_OUTER_CLEANUP_MARGIN_S
    if registry is None:
        return subprocess.run(
            args,
            input=input,
            capture_output=capture_output,
            timeout=timeout,
            check=check,
            **kwargs,
        )
    if bool(getattr(registry, "cancelled", False)):
        text_mode = bool(
            kwargs.get("text")
            or kwargs.get("universal_newlines")
            or kwargs.get("encoding")
            or kwargs.get("errors")
        )
        empty: Any = "" if text_mode else b""
        completed = subprocess.CompletedProcess(
            args=args,
            returncode=130,
            stdout=(
                empty
                if capture_output or kwargs.get("stdout") == subprocess.PIPE
                else None
            ),
            stderr=(
                empty
                if capture_output or kwargs.get("stderr") == subprocess.PIPE
                else None
            ),
        )
        if check:
            completed.check_returncode()
        return completed
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output"
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    if os.name == "posix":
        kwargs.setdefault("start_new_session", True)
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
        kwargs.setdefault(
            "creationflags", subprocess.CREATE_NEW_PROCESS_GROUP
        )
    proc = subprocess.Popen(args, **kwargs)

    def _terminate_owned(grace_s: float) -> None:
        registry.terminate_registered(proc, grace_s=grace_s)

    def _cancel_leased_remote_best_effort() -> None:
        if (
            remote_registry is None
            or not raw_args
            or os.path.basename(raw_args[0]).lower() not in {"ssh", "ssh.exe"}
        ):
            return
        try:
            remote_registry.cancel_all(grace_s=3.0)
        except BaseException:
            # The descriptor remains registered.  The runner's final
            # quarantine gate will retry and fail closed if proof is absent.
            pass

    try:
        registry.register(proc, label="hailo-owned-subprocess")
    except BaseException:
        _cancel_leased_remote_best_effort()
        _terminate_owned(0.5)
        raise

    def _bounded_collect_after_stop() -> tuple[Any, Any]:
        try:
            return proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired as exc:
            partial_stdout = exc.output
            partial_stderr = exc.stderr
            try:
                _terminate_owned(0.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                return proc.communicate(timeout=0.5)
            except subprocess.TimeoutExpired as final_exc:
                return (
                    final_exc.output if final_exc.output is not None else partial_stdout,
                    final_exc.stderr if final_exc.stderr is not None else partial_stderr,
                )

    try:
        deadline = (
            time.monotonic() + max(0.0, float(timeout))
            if timeout is not None else None
        )
        pending_input = input
        cancelled = False
        while True:
            if bool(getattr(registry, "cancelled", False)):
                cancelled = True
                _cancel_leased_remote_best_effort()
                _terminate_owned(0.5)
                stdout, stderr = _bounded_collect_after_stop()
                break
            wait_s = 0.2
            if deadline is not None:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    _cancel_leased_remote_best_effort()
                    _terminate_owned(0.5)
                    stdout, stderr = _bounded_collect_after_stop()
                    raise subprocess.TimeoutExpired(
                        args,
                        timeout,
                        output=stdout,
                        stderr=stderr,
                    )
                wait_s = min(wait_s, remaining_s)
            try:
                stdout, stderr = proc.communicate(
                    input=pending_input,
                    timeout=wait_s,
                )
                break
            except subprocess.TimeoutExpired:
                # communicate() retains a partially written input buffer; it
                # must not be submitted a second time on the next poll.
                pending_input = None
                continue
        completed = subprocess.CompletedProcess(
            args=args,
            returncode=130 if cancelled else int(proc.returncode or 0),
            stdout=stdout,
            stderr=stderr,
        )
        if check:
            completed.check_returncode()
        return completed
    except BaseException:
        # Unexpected Python/pipe failures must preserve the same ordering as
        # cooperative workflow cancellation: exact remote control first,
        # then bounded local process-tree teardown.
        _cancel_leased_remote_best_effort()
        _terminate_owned(0.5)
        raise
    finally:
        registry.unregister(proc)


def _owned_check_output(args: Any, **kwargs: Any) -> Any:
    if "stdout" in kwargs:
        raise ValueError("stdout argument not allowed; it will be overridden")
    return _run_owned_subprocess(
        args,
        stdout=subprocess.PIPE,
        check=True,
        **kwargs,
    ).stdout


def _hailo_heartbeat_interval_s() -> float:
    try:
        return max(0.0, float(os.environ.get("ONNX_SPLITPOINT_HAILO_HEARTBEAT_S", "30")))
    except Exception:
        return 30.0


def _run_with_hailo_heartbeat(label: str, fn: Callable[[], Any]) -> Any:
    """Run a potentially long blocking Hailo SDK call with periodic stdout
    heartbeats.  Hailo optimization/compile can be silent for minutes, which
    looks like a hung benchmark-set generator in the GUI.  This helper does not
    change SDK semantics; it only makes long phases observable.
    """
    interval = _hailo_heartbeat_interval_s()
    if interval <= 0:
        return fn()
    stop = threading.Event()
    started = time.time()

    def _beat() -> None:
        # Print only after the first interval so short calls stay quiet.
        while not stop.wait(interval):
            elapsed = time.time() - started
            try:
                print(f"[hailo][heartbeat] {label} still running after {elapsed:.0f}s", flush=True)
            except Exception:
                pass

    t = threading.Thread(target=_beat, name=f"hailo-heartbeat-{label[:20]}", daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop.set()
        elapsed = time.time() - started
        try:
            print(f"[hailo][heartbeat] {label} finished after {elapsed:.1f}s", flush=True)
        except Exception:
            pass


def _parse_simple_version(ver: str) -> Optional[Tuple[int, int]]:
    """Parse a simple 'major.minor' version string into a tuple.

    Returns None if parsing fails.
    """

    s = str(ver or "").strip()
    if not s:
        return None
    # Accept "2.35" and also "glibc 2.35" (from getconf output).
    if " " in s:
        s = s.split()[-1].strip()
    if "." not in s:
        return None
    try:
        a, b = s.split(".", 1)
        return int(a), int(re.match(r"^(\d+)", b).group(1) if re.match(r"^(\d+)", b) else b)
    except Exception:
        return None


def _version_lt(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return (a[0] < b[0]) or (a[0] == b[0] and a[1] < b[1])


def _default_glibc_min_for_hw_arch(hw_arch: str) -> Optional[Tuple[int, int]]:
    hw = str(hw_arch or "").strip().lower()
    if hw.startswith("hailo8") or hw.startswith("hailo10"):
        # Both current DFC wheels (3.33 / 5.2) are built on a baseline that
        # requires glibc >= 2.34.
        return (2, 34)
    return None


_HAILO_HW_ARCH_ALIASES: Dict[str, str] = {
    # DFC 5.x expects explicit variants (hailo10h / hailo10p). Older configs
    # (and earlier versions of this tool) used the generic "hailo10" string.
    #
    # We intentionally map to *hailo10h* as a sensible default for most Hailo-10
    # modules. Users targeting Hailo-10P should select "hailo10p" explicitly.
    "hailo10": "hailo10h",
    # Future-proofing: if users pass "hailo15" as a family name, pick a default.
    "hailo15": "hailo15h",
}


def _normalize_hailo_hw_arch(hw_arch: str) -> str:
    """Normalize user-provided Hailo `hw_arch` strings for the DFC.

    The Hailo DFC (`hailo_sdk_client.ClientRunner`) validates `hw_arch` against
    a fixed set of strings. Some DFC versions changed these identifiers (e.g.
    "hailo10" -> "hailo10h"/"hailo10p").

    We keep backwards compatibility by mapping known legacy aliases.
    """

    hw = str(hw_arch or "").strip().lower()
    if not hw:
        return "hailo8"
    return _HAILO_HW_ARCH_ALIASES.get(hw, hw)


def _clean_opt_str(val: object) -> Optional[str]:
    """Return a cleaned optional string.

    - None -> None
    - "" / whitespace -> None
    - "None" / "null" (case-insensitive) -> None

    This avoids the common bug where `str(None)` becomes the literal "None",
    which then gets passed to `wsl.exe -d None`.
    """
    if val is None:
        return None
    try:
        s = str(val).strip()
    except Exception:
        return None
    if not s:
        return None
    if s.lower() in {"none", "null"}:
        return None
    return s


def _sanitize_wsl_text(s: str) -> str:
    """Sanitize WSL stdout/stderr for GUI/log consumption.

    Some `wsl.exe` service errors are emitted as UTF-16LE and end up decoded
    with embedded NULs when read as UTF-8. Tk message boxes may truncate at
    NUL characters, so we remove them.
    """
    if not s:
        return ""
    return s.replace("\x00", "")


def _truncate_log_text(s: str, *, max_chars: int = 20000) -> str:
    """Truncate very long subprocess output for readable logs.

    We keep a head+tail window and insert a truncation marker in the middle.
    """
    if not s:
        return ""
    try:
        s = str(s)
    except Exception:
        return ""
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    # 30% head, 70% tail (tail usually contains the real error)
    head_n = max(2000, int(max_chars * 0.3))
    tail_n = max(2000, max_chars - head_n)
    head = s[:head_n]
    tail = s[-tail_n:]
    return head + "\n\n… [TRUNCATED: output too long for gui.log] …\n\n" + tail



def _write_wsl_debug_log(
    outdir_win: Optional[Union[str, Path]],
    *,
    filename: str,
    wsl_cmd: List[str],
    stdout: str,
    stderr: str,
) -> Optional[str]:
    """Write a debug log file (optional) OR pipe details into the main gui.log.

    Earlier versions wrote one debug log file per failing WSL call next to the
    split outputs. That quickly becomes noisy.

    Current behaviour:
    - Default: do **not** create extra files. Instead, emit the full command +
      stdout/stderr into the main logger (gui.log).
    - Opt-in: set ONNX_SPLITPOINT_HAILO_DEBUG_FILES=1 to re-enable per-call
      debug log files.
    """

    debug_files = str(os.environ.get("ONNX_SPLITPOINT_HAILO_DEBUG_FILES", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    # Always pipe details into gui.log.
    try:
        cmd_s = " ".join(map(str, wsl_cmd))
        out_s = _truncate_log_text(_sanitize_wsl_text(stdout or ""), max_chars=50000)
        err_s = _truncate_log_text(_sanitize_wsl_text(stderr or ""), max_chars=50000)
        log.error(
            "[hailo][debug] %s\ncmd: %s\n\n--- stdout ---\n%s\n\n--- stderr ---\n%s\n",
            filename,
            cmd_s,
            out_s or "<empty>",
            err_s or "<empty>",
        )
    except Exception:
        pass

    if not debug_files:
        return None

    def _wsl_to_win(p: str) -> str:
        """Convert a common WSL path (/mnt/<drive>/...) back to a Windows path.

        Some error paths pass WSL paths ("/mnt/c/...") into this function.
        On Windows, Path("/mnt/c/...") resolves to "\\mnt\\c...", which is
        typically not writable, and we'd silently fail to write the debug log.
        """

        s = str(p)
        m = re.match(r"^/mnt/([a-zA-Z])/(.*)$", s)
        if not m:
            return s
        drive = m.group(1).upper()
        rest = m.group(2).replace("/", "\\")
        return f"{drive}:\\{rest}"

    try:
        # Prefer writing next to the failing output (outdir), but always fall
        # back to a stable location if that path is not writable.
        if outdir_win:
            out_s = str(outdir_win)
            if sys.platform == "win32" and out_s.startswith("/mnt/"):
                out_s = _wsl_to_win(out_s)
            base = Path(out_s).expanduser().resolve()
        else:
            from .paths import splitpoint_wsl_debug_dir
            base = splitpoint_wsl_debug_dir()

        # If the chosen base is not usable, fall back to the project-local wsl_debug dir
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            from .paths import splitpoint_wsl_debug_dir
            base = splitpoint_wsl_debug_dir()
            base.mkdir(parents=True, exist_ok=True)

        p = base / filename

        lines: List[str] = []
        lines.append("# ONNX Splitpoint Tool - WSL debug log")
        lines.append("")
        lines.append(f"cmd: {' '.join(map(str, wsl_cmd))}")
        lines.append("")
        lines.append("--- stdout ---")
        lines.append(_sanitize_wsl_text(stdout) or "<empty>")
        lines.append("")
        lines.append("--- stderr ---")
        lines.append(_sanitize_wsl_text(stderr) or "<empty>")
        lines.append("")
        p.write_text("\n".join(lines), encoding="utf-8", errors="replace")
        return str(p)
    except Exception:
        return None


def hailo_sdk_available() -> bool:
    """Return True if the Hailo SDK python module can be imported."""
    try:
        import hailo_sdk_client  # noqa: F401

        return True
    except Exception:
        return False


@dataclass
class HailoProbeResult:
    """Result of a quick Hailo backend availability check."""

    ok: bool
    backend: str  # "local" or "wsl"
    reason: str = ""
    details: Optional[Dict[str, Any]] = None


# ------------------------------- WSL bridge -------------------------------

_WSL_RESULT_MARKER = "__SPLITPOINT_HAILO_RESULT__"

# Printed by our CUDA probe helper (see onnx_splitpoint_tool/cuda_probe.py).
_CUDA_PROBE_MARKER = "__SPLITPOINT_CUDA_PROBE__"


def _find_marker_json(text: str, marker: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON payload following a marker from mixed output."""
    if not text or not marker:
        return None
    idx = text.rfind(marker)
    if idx < 0:
        return None
    payload_all = text[idx + len(marker):]
    first_line = payload_all.strip().splitlines()[0].strip() if payload_all else ""
    if not first_line:
        return None

    # 1) Best case: pure JSON
    try:
        return json.loads(first_line)
    except Exception:
        pass

    # 2) Fallback: extract the first {...} block
    try:
        a = first_line.find("{")
        b = first_line.rfind("}")
        if a >= 0 and b > a:
            return json.loads(first_line[a : b + 1])
    except Exception:
        return None

    return None


def hailo_wsl_available() -> bool:
    """Return True if we appear to be on Windows and `wsl` is callable."""
    if sys.platform != "win32":
        return False
    return (shutil.which("wsl.exe") is not None) or (shutil.which("wsl") is not None)


def _wsl_exe() -> str:
    """Pick the best WSL executable name for subprocess calls."""
    return "wsl.exe" if shutil.which("wsl.exe") is not None else "wsl"


def wsl_list_distros(*, timeout_s: float = 3.0) -> List[str]:
    """Return available WSL distribution names (best-effort).

    This is used to make the GUI less error-prone (avoid typos like
    "Ubuntu_22.04" vs "Ubuntu-22.04").

    Returns an empty list if WSL is not available.
    """

    if not hailo_wsl_available():
        return []
    exe = _wsl_exe()
    try:
        proc = _run_owned_subprocess(
            [exe, "-l", "-q"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        out = _sanitize_wsl_text((proc.stdout or "") + (proc.stderr or ""))
        # `wsl -l -q` prints one distro name per line.
        distros = [ln.strip() for ln in out.replace("\r", "\n").split("\n") if ln.strip()]
        # De-duplicate while preserving order.
        seen = set()
        uniq: List[str] = []
        for d in distros:
            if d in seen:
                continue
            seen.add(d)
            uniq.append(d)
        return uniq
    except Exception:
        return []


def normalize_wsl_distro_name(distro: str) -> str:
    """Best-effort normalization of a user-provided WSL distro name.

    - strips whitespace
    - accepts empty string (meaning "default")
    - auto-fixes a common typo: underscores instead of hyphens
      if the fixed name exists in `wsl -l -q`.
    """

    s = (distro or "").strip()
    if not s:
        return ""

    # If exact match exists, keep it.
    distros = wsl_list_distros(timeout_s=2.0)
    if s in distros:
        return s

    # Case-insensitive match.
    for d in distros:
        if d.lower() == s.lower():
            return d

    # Common typo: "Ubuntu_22.04" instead of "Ubuntu-22.04".
    if "_" in s and "-" not in s:
        alt = s.replace("_", "-")
        if alt in distros:
            return alt
        for d in distros:
            if d.lower() == alt.lower():
                return d

    return s


def windows_path_to_wsl(path: Union[str, Path]) -> str:
    """Best-effort conversion of a Windows path to a WSL path.

    Examples
    --------
    "C:\\temp\\a b\\x.onnx" -> "/mnt/c/temp/a b/x.onnx"
    "D:/data/model.onnx"     -> "/mnt/d/data/model.onnx"

    If the input already looks like a Linux path ("/mnt/..." or "/home/..."),
    it is returned unchanged.
    """

    s = str(path)
    if s.startswith("/mnt/") or s.startswith("/home/") or s.startswith("/"):
        return s

    # Normalize backslashes
    s = s.replace("\\", "/")

    # Drive letter path
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        return f"/mnt/{drive}/{rest}"

    return s


def _bash_quote(s: str) -> str:
    """Quote a string for `bash -lc` commands."""
    return shlex.quote(str(s))


# --------------------------- Managed venv bridge ---------------------------

def _resolve_managed_venv_python(
    *,
    hw_arch: str,
    venv_activate: str = "auto",
) -> Tuple[str, Path, str]:
    """Resolve the managed DFC venv python for a given hw_arch.

    On Windows, the managed venv lives *inside WSL*, so you must use the WSL
    bridge. On Linux (native or WSL), we can run the venv python directly.

    Returns: (profile_id, venv_python_path, activate_path)
    """

    mgr = get_dfc_manager()
    resolved = mgr.resolve_wsl_runtime(
        hw_arch=str(hw_arch),
        wsl_distro=None,
        wsl_venv_activate=(_clean_opt_str(venv_activate) or "auto"),
    )
    act = str(resolved.wsl_venv_activate or "").strip()
    if not act:
        raise RuntimeError(
            f"No managed DFC profile found for hw_arch={hw_arch!r}. "
            "Set an explicit venv activate path, or add a profile in resources/hailo/profiles.json."
        )

    act_expanded = os.path.expanduser(act) if act.startswith("~") else act
    act_path = Path(act_expanded).expanduser()
    if not act_path.is_absolute():
        # Be conservative; resolve relative paths against current working dir.
        act_path = act_path.resolve()

    # .../<venv>/bin/activate -> parents[1] == <venv>
    venv_dir = act_path.parents[1]
    py = venv_dir / "bin" / "python"
    if not py.exists():
        py = venv_dir / "bin" / "python3"
    if not py.exists():
        raise RuntimeError(f"Managed venv python not found (expected {venv_dir}/bin/python)")

    return str(resolved.profile_id), py, str(act_path)


def _managed_venv_child_env(
    python_path: Union[str, Path],
    base_env: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Project normal venv activation semantics into a direct child launch.

    Managed DFC helpers intentionally invoke the venv interpreter by absolute
    path instead of sourcing ``bin/activate``.  Python imports then work, but
    vendor subprocesses such as ``onnxsim`` still resolve through the inherited
    host ``PATH`` unless the venv's ``bin`` directory is projected explicitly.
    """

    # Keep the lexical venv path.  POSIX venv interpreters are normally
    # symlinks to a system interpreter; ``resolve()`` would therefore turn
    # ``<venv>/bin/python`` into (for example) ``/usr/bin/python3.10`` and
    # project the system ``bin`` directory instead of the managed venv.
    py = Path(
        os.path.abspath(
            os.fspath(Path(str(python_path)).expanduser())
        )
    )
    bin_dir = py.parent
    venv_dir = bin_dir.parent
    source = os.environ if base_env is None else base_env
    env = {str(key): str(value) for key, value in dict(source).items()}
    inherited_path = str(env.get("PATH") or "")
    path_parts = [str(bin_dir)]
    path_parts.extend(
        part
        for part in inherited_path.split(os.pathsep)
        if part
        and Path(
            os.path.abspath(os.fspath(Path(part).expanduser()))
        ) != bin_dir
    )
    env["PATH"] = os.pathsep.join(path_parts)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env.pop("PYTHONHOME", None)
    return env


def hailo_probe_via_venv(
    *,
    hw_arch: str = "hailo8",
    venv_activate: str = "auto",
    timeout_s: int = 90,
) -> "HailoProbeResult":
    """Probe a managed DFC venv *directly* (Linux / WSL).

    This is the Linux counterpart of :func:`hailo_probe_via_wsl`.
    """

    if sys.platform == "win32":
        return HailoProbeResult(ok=False, backend="venv", reason="Managed venv probe is not available on Windows (use WSL backend).")

    def _summarize(out_text: str) -> str:
        t = (out_text or "").strip()
        if not t:
            return "Probe failed"
        for line in reversed(t.splitlines()):
            if "__HAILO_PROBE_ERR__" in line:
                msg = line.split("__HAILO_PROBE_ERR__", 1)[1].strip()
                if msg:
                    # Map common low-level errors to helpful guidance.
                    if "pkg_resources" in msg:
                        return "pkg_resources missing (setuptools>=82 removed it). Install setuptools<82 or re-run provisioning."
                    if "GLIBC_" in msg and "libc.so.6" in msg:
                        return "glibc too old for this DFC wheel (needs >= 2.34). Use a newer distro / environment."
                    if "Descriptors cannot be created directly" in msg or "CheckCalledFromGeneratedFile" in msg:
                        return "protobuf version mismatch (env drift). Re-run provisioning."
                    return msg[:240]
        if "Descriptors cannot be created directly" in t or "CheckCalledFromGeneratedFile" in t:
            return "protobuf version mismatch (env drift). Re-run provisioning."

        if "GLIBC_" in t and "libc.so.6" in t:
            return "glibc too old for this DFC wheel (needs >= 2.34). Use a newer distro / environment."
        if "No module named" in t and "pkg_resources" in t:
            return "pkg_resources missing (setuptools>=82 removed it). Install setuptools<82 or re-run provisioning."
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        return (lines[-1] if lines else "Probe failed")[:240]

    try:
        profile_id, py, act_path = _resolve_managed_venv_python(hw_arch=str(hw_arch), venv_activate=(_clean_opt_str(venv_activate) or "auto"))
    except Exception as e:
        return HailoProbeResult(ok=False, backend="venv", reason=f"Failed to resolve DFC profile: {e}")

    # Fast pre-flight: glibc version check.
    #
    # Some Hailo SDK wheels ship native libraries that require newer glibc
    # symbols (e.g. GLIBC_2.34). On older distros (Ubuntu 20.04 glibc 2.31),
    # imports may appear to work, but HEF build fails later when the binary is
    # actually loaded. We check up-front and provide a clear error.
    try:
        mgr = get_dfc_manager()
        prof = mgr.get_profile(str(profile_id))
        req = _parse_simple_version(getattr(prof, "glibc_min", "") or "") if prof is not None else None
        req_tuple = req if req is not None else _default_glibc_min_for_hw_arch(str(hw_arch))
        if req_tuple is not None:
            out_glibc = _owned_check_output(["getconf", "GNU_LIBC_VERSION"], text=True, stderr=subprocess.STDOUT).strip()
            cur = _parse_simple_version(out_glibc)
            if cur is not None and _version_lt(cur, req_tuple):
                details = {
                    "profile_id": profile_id,
                    "venv_activate": act_path,
                    "venv_python": str(py),
                    "glibc": f"{cur[0]}.{cur[1]}",
                    "glibc_required": f"{req_tuple[0]}.{req_tuple[1]}",
                }
                return HailoProbeResult(
                    ok=False,
                    backend="venv",
                    reason=f"glibc too old (have {cur[0]}.{cur[1]}, need >= {req_tuple[0]}.{req_tuple[1]})",
                    details=details,
                )
    except Exception:
        # If the check fails for any reason (missing getconf etc.), do not
        # block probing; the import probe will still provide a useful error.
        pass

    # Self-heal: setuptools 82+ removed pkg_resources, but some Hailo SDK
    # components still import it.
    try:
        _ = _owned_check_output([str(py), "-c", "import pkg_resources"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        try:
            log.info("[hailo][probe][venv] pkg_resources missing -> installing setuptools<82 (self-heal)")
            _run_owned_subprocess(
                [str(py), "-m", "pip", "install", "--force-reinstall", "setuptools<82"],
                capture_output=True,
                text=True,
                timeout=min(120, max(10, int(timeout_s))),
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            # Keep probing even if the fix step fails; the probe error will
            # provide a useful reason.
            pass

    # Also run a lightweight CUDA probe (for GUI visibility). We do it before
    # forcing CPU for the import probe, so the reported capability reflects the
    # *system* and not the probe's CUDA_VISIBLE_DEVICES override.
    py_probe = (
        "import sys, os, json; "
        "\n# CUDA probe (best-effort)\n"
        "try:\n"
        "  from onnx_splitpoint_tool.cuda_probe import probe_cuda_environment\n"
        f"  print('{_CUDA_PROBE_MARKER}' + json.dumps(probe_cuda_environment(), ensure_ascii=False))\n"
        "except Exception as _e:\n"
        f"  print('{_CUDA_PROBE_MARKER}' + json.dumps({{'error': str(_e)}}, ensure_ascii=False))\n"
        "\n# Import probe (keep quiet / avoid GPU init)\n"
        "os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1'); "
        "os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3'); "
        "\ntry:\n"
        "  import pkg_resources, hailo_sdk_client, onnx, google.protobuf\n"
        "  # Try to load the emulator module as an early indicator for binary/GLIBC issues.\n"
        "  try:\n"
        "    import hailo_sdk_client.emulator.emulator  # noqa\n"
        "  except Exception as _e:\n"
        "    s = str(_e)\n"
        "    if ('GLIBC_' in s) or ('libc.so.6' in s):\n"
        "      raise\n"
        "  print('__HAILO_PROBE_OK__', getattr(hailo_sdk_client,'__version__','?'), onnx.__version__, google.protobuf.__version__)\n"
        "except Exception as e:\n"
        "  print('__HAILO_PROBE_ERR__', type(e).__name__ + ':', str(e))\n"
        "  sys.exit(2)\n"
    )

    # Pass an explicit environment to the probe. This keeps behavior
    # deterministic and avoids crashes like: "name 'env' is not defined".
    env = _managed_venv_child_env(py)
    env.setdefault("PYTHONUNBUFFERED", "1")

    # Make the tool package importable inside the managed venv probe so we can
    # run the CUDA probe helper.
    try:
        repo_root = Path(__file__).resolve().parents[1]
        pp = str(env.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = (str(repo_root) + (os.pathsep + pp if pp else "")).strip()
    except Exception:
        pass

    cmd = [str(py), "-c", py_probe]

    try:
        log.info("[hailo][probe][venv] hw_arch=%s profile=%s python=%s", hw_arch, profile_id, str(py))
        # Importing hailo_sdk_client can trigger an *interactive* system requirements check
        # on first use ("Continue? [Y/n]"). In a GUI / non-interactive context this would
        # block forever and end in a timeout. We proactively feed "y".
        proc = _run_owned_subprocess(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
            env=env,
            input="y\n",
        )
        out = _sanitize_wsl_text((proc.stdout or "") + (proc.stderr or ""))

        cuda_probe = _find_marker_json(out, _CUDA_PROBE_MARKER)

        glibc_seen: Optional[str] = None
        try:
            for ln in out.splitlines():
                if ln.strip().startswith("__SPLITPOINT_GLIBC__"):
                    parts = ln.strip().split()
                    if len(parts) >= 2:
                        glibc_seen = parts[1].strip()
        except Exception:
            glibc_seen = None
        ok = "__HAILO_PROBE_OK__" in out
        if not ok:
            log.warning(
                "[hailo][probe][venv] failed rc=%s tail=%s",
                proc.returncode,
                " ".join((out.strip().splitlines()[-5:] if out else ["<no output>"])),
            )
        details = {
            "returncode": proc.returncode,
            "profile_id": profile_id,
            "venv_activate": act_path,
            "venv_python": str(py),
            "cuda_probe": cuda_probe,
            "output_tail": "\n".join(out.strip().splitlines()[-30:]),
        }
        return HailoProbeResult(ok=ok, backend="venv", reason=("" if ok else _summarize(out)), details=details)
    except subprocess.TimeoutExpired:
        return HailoProbeResult(ok=False, backend="venv", reason=f"Venv probe timed out after {timeout_s}s")
    except Exception as e:
        return HailoProbeResult(ok=False, backend="venv", reason=str(e))


def hailo_probe_local() -> HailoProbeResult:
    """Check whether the local Python environment can import the Hailo DFC SDK."""
    try:
        import hailo_sdk_client  # type: ignore

        details: Dict[str, Any] = {}
        details["hailo_sdk_client"] = getattr(hailo_sdk_client, "__file__", None)
        details["hailo_sdk_client_version"] = getattr(hailo_sdk_client, "__version__", None)
        return HailoProbeResult(ok=True, backend="local", details=details)
    except Exception as e:
        return HailoProbeResult(ok=False, backend="local", reason=str(e), details=None)


def hailo_probe_via_wsl(
    *,
    hw_arch: str = "hailo8",
    wsl_distro: str = "",
    wsl_venv_activate: str = "auto",
    timeout_s: int = 90,
) -> HailoProbeResult:
    """Check whether Hailo DFC is reachable inside WSL.

    This is meant for the Windows GUI, where the DFC lives in WSL.

    If `wsl_venv_activate` is set to "auto" (or empty), we resolve a managed
    venv based on `hw_arch` via :class:`~onnx_splitpoint_tool.hailo.dfc_manager.DfcManager`.
    """
    if not hailo_wsl_available():
        return HailoProbeResult(ok=False, backend="wsl", reason="wsl.exe not found (WSL not available)")

    def _summarize(out_text: str) -> str:
        """Extract a short, human-friendly failure reason from probe output."""

        t = (out_text or "").strip()
        if not t:
            return "Probe failed"

        # Explicit error marker from our probe snippet.
        for line in reversed(t.splitlines()):
            if "__HAILO_GLIBC_TOO_OLD__" in line:
                # Example: "__HAILO_GLIBC_TOO_OLD__ glibc=2.31 required=2.34"
                s = line.split("__HAILO_GLIBC_TOO_OLD__", 1)[1].strip()
                # Keep it compact; full details are shown on badge click.
                if s:
                    return f"WSL distro too old ({s})"
                return "WSL distro too old (glibc too old)"
            if "__HAILO_PROBE_ERR__" in line:
                msg = line.split("__HAILO_PROBE_ERR__", 1)[1].strip()
                if msg:
                    # Map common low-level errors to helpful guidance.
                    if "pkg_resources" in msg:
                        return "pkg_resources missing (setuptools>=82 removed it). Install setuptools<82 or re-run provisioning."
                    if "GLIBC_" in msg and "libc.so.6" in msg:
                        return "glibc too old for this DFC wheel (needs >= 2.34). Use a newer distro / environment."
                    if "Descriptors cannot be created directly" in msg or "CheckCalledFromGeneratedFile" in msg:
                        return "protobuf version mismatch (env drift). Re-run provisioning."
                    return msg[:240]

        # Common protobuf mismatch symptom.
        if "Descriptors cannot be created directly" in t or "CheckCalledFromGeneratedFile" in t:
            return "protobuf version mismatch (env drift). Re-run provisioning."

        if "GLIBC_" in t and "libc.so.6" in t:
            return "glibc too old for this DFC wheel (needs >= 2.34). Use a newer distro / environment."

        if "__HAILO_GLIBC_TOO_OLD__" in t:
            return "WSL distro too old (glibc too old)"

        if "No module named" in t and "hailo_sdk_client" in t:
            return "hailo_sdk_client not importable (DFC not installed)"

        if "No module named" in t and "pkg_resources" in t:
            return "pkg_resources missing (setuptools>=82 removed it). Install setuptools<82 or re-run provisioning."

        if "No such file or directory" in t and "activate" in t:
            return "WSL venv activate script not found"

        # Fallback: last non-empty line.
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        return (lines[-1] if lines else "Probe failed")[:240]

    # Resolve managed venv/distro if requested.
    try:
        mgr = get_dfc_manager()
        resolved = mgr.resolve_wsl_runtime(
            hw_arch=str(hw_arch),
            wsl_distro=_clean_opt_str(wsl_distro),
            wsl_venv_activate=(_clean_opt_str(wsl_venv_activate) or "auto"),
        )
    except Exception as e:
        return HailoProbeResult(ok=False, backend="wsl", reason=f"Failed to resolve DFC profile: {e}")

    distro_eff = str(resolved.wsl_distro or "").strip()
    venv_eff = str(resolved.wsl_venv_activate or "").strip()

    # Keep a dedicated environment dict for subprocess calls. This also lets
    # us feed stdin defaults (the first DFC import may prompt interactively).
    env = os.environ.copy()

    if not venv_eff:
        return HailoProbeResult(
            ok=False,
            backend="wsl",
            reason=(
                f"No managed DFC profile found for hw_arch={hw_arch!r}. "
                "Set an explicit WSL venv path, or add a profile in resources/hailo/profiles.json."
            ),
            details={"hw_arch": str(hw_arch), "wsl_distro": distro_eff or None},
        )

    wsl_exe = shutil.which("wsl.exe") or shutil.which("wsl") or "wsl.exe"

    # Determine the minimum required glibc for this profile (if known).
    # We embed the check into the WSL probe bash script so we can fail fast
    # (before importing heavy Python deps / touching the DFC).
    glibc_req_tuple: Optional[Tuple[int, int]] = None
    try:
        prof = mgr.get_profile(str(resolved.profile_id or "")) if getattr(resolved, "profile_id", None) else None
        if prof is not None and getattr(prof, "glibc_min", None):
            glibc_req_tuple = _parse_simple_version(str(getattr(prof, "glibc_min")))
    except Exception:
        glibc_req_tuple = None
    if glibc_req_tuple is None:
        glibc_req_tuple = _default_glibc_min_for_hw_arch(str(hw_arch))

    cmd = [wsl_exe]
    if distro_eff:
        cmd += ["-d", distro_eff]

    # NOTE: do not quote paths starting with '~' here; quoting prevents tilde expansion.
    act = venv_eff
    # Print a unique marker so we can reliably detect success.
    # Compact probe: import hailo_sdk_client and also print versions of onnx/protobuf.
    # We print a single-line marker to make it easy to show a meaningful error in the GUI.
    py_probe = (
        "import sys; "
        "\ntry:\n"
        "  import hailo_sdk_client, onnx, google.protobuf\n"
        "  # Try to load the emulator module as an early indicator for binary/GLIBC issues.\n"
        "  try:\n"
        "    import hailo_sdk_client.emulator.emulator  # noqa\n"
        "  except Exception as _e:\n"
        "    s = str(_e)\n"
        "    if ('GLIBC_' in s) or ('libc.so.6' in s):\n"
        "      raise\n"
        "  print('__HAILO_PROBE_OK__', getattr(hailo_sdk_client,'__version__','?'), onnx.__version__, google.protobuf.__version__)\n"
        "except Exception as e:\n"
        "  print('__HAILO_PROBE_ERR__', type(e).__name__ + ':', str(e))\n"
        "  sys.exit(2)\n"
    )

    # Lightweight CUDA probe for GUI visibility (best-effort). This runs *before*
    # we force CUDA_VISIBLE_DEVICES=-1 for the import probe, so the result
    # reflects whether GPU acceleration is actually usable.
    py_cuda_probe = (
        "import json, os, shutil, subprocess, sys; "
        "from pathlib import Path; "
        "def _run(cmd, t=2.5):\n"
        "  try:\n"
        "    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=t)\n"
        "    return int(p.returncode), ((p.stdout or '') + (p.stderr or '')).strip()\n"
        "  except Exception as e:\n"
        "    return 127, f'{type(e).__name__}: {e}'\n"
        "exe = shutil.which('nvidia-smi'); "
        "smi_found = bool(exe); smi_ok = False; gpus = []; smi_err = ''; "
        "\nif exe:\n"
        "  rc, out = _run([exe, '-L']);\n"
        "  if rc == 0:\n"
        "    gpus = [ln.strip() for ln in out.splitlines() if ln.strip()];\n"
        "    smi_ok = bool(gpus);\n"
        "  else:\n"
        "    smi_err = out or f'nvidia-smi rc={rc}';\n"
        "\nroots = [];\n"
        "for k in ('CUDA_HOME','CUDA_PATH','CUDA_DIR'):\n"
        "  v = (os.environ.get(k) or '').strip();\n"
        "  if v: roots.append(Path(v).expanduser());\n"
        "roots.append(Path('/usr/local/cuda'));\n"
        "try:\n"
        "  roots += sorted(Path('/usr/local').glob('cuda-*'));\n"
        "except Exception:\n"
        "  pass\n"
        "def _find_lib(root: Path):\n"
        "  d = root/'nvvm'/'libdevice'\n"
        "  if d.is_dir():\n"
        "    for f in sorted(d.glob('libdevice*.bc')):\n"
        "      if f.is_file():\n"
        "        return str(f)\n"
        "  return None\n"
        "lib = None; root_ok = None;\n"
        "for r in roots:\n"
        "  try:\n"
        "    lib = _find_lib(r)\n"
        "  except Exception:\n"
        "    lib = None\n"
        "  if lib:\n"
        "    root_ok = str(r);\n"
        "    break\n"
        "gpu_ok = bool(smi_ok) and bool(lib);\n"
        "payload = {\n"
        "  'nvidia_smi': {'found': smi_found, 'ok': smi_ok, 'gpus': gpus[:8], 'error': smi_err},\n"
        "  'cuda_root': root_ok,\n"
        "  'libdevice_path': lib,\n"
        "  'gpu_ok': gpu_ok,\n"
        "  'summary': ('Compute: GPU (auto)' if gpu_ok else 'Compute: CPU (auto: CUDA/libdevice missing)')\n"
        "};\n"
        f"print('{_CUDA_PROBE_MARKER}' + json.dumps(payload, ensure_ascii=False))\n"
    )

    # Use `python` (not `python3`) after activation. Some venvs do not provide a
    # `python3` shim, which would accidentally run the *system* python3.
    #
    # IMPORTANT: Do NOT embed a glibc preflight check inside the `bash -lc` script.
    # On Windows, quoting/argument translation through `wsl.exe` can result in the
    # command substitution output (e.g. "2.35") being treated as a standalone shell
    # command, yielding: "bash: line 1: 2.35: command not found".
    #
    # Instead, perform the glibc check as a separate `wsl.exe -- getconf ...` call
    # below (before sourcing the venv).

    bash = (
        "set -e; "
        "echo __SPLITPOINT_WSL_BEGIN__; "
        f"source {act}; "
        "echo __SPLITPOINT_WSL_VENV_OK__; "
        # CUDA probe (best-effort, non-fatal)
        f"python -c {shlex.quote(py_cuda_probe)} || true; "
        # Keep probe fast/quiet: avoid GPU probing on import.
        "export CUDA_VISIBLE_DEVICES=-1; "
        "export TF_CPP_MIN_LOG_LEVEL=3; "
        # Self-heal: setuptools 82+ removed pkg_resources. Some Hailo SDK
        # components still import it.
        "python -c 'import pkg_resources' >/dev/null 2>&1 || "
        "python -m pip install --force-reinstall 'setuptools<82' >/dev/null 2>&1 || true; "
        f"python -c {shlex.quote(py_probe)}; "
        "(hailo --version 2>/dev/null || true)"
    )
    cmd += ["--", "bash", "-lc", bash]

    # Separate glibc pre-flight (WSL).
    glibc_seen: Optional[str] = None
    if glibc_req_tuple is not None:
        try:
            glibc_cmd = ["wsl.exe"]
            if distro_eff:
                glibc_cmd += ["-d", distro_eff]
            glibc_cmd += ["--", "getconf", "GNU_LIBC_VERSION"]
            gproc = _run_owned_subprocess(
                glibc_cmd,
                capture_output=True,
                text=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
            )
            gout = _sanitize_wsl_text((gproc.stdout or "") + (gproc.stderr or "")).strip()
            cur = _parse_simple_version(gout)
            if cur is not None:
                glibc_seen = f"{cur[0]}.{cur[1]}"
                if (cur[0], cur[1]) < (int(glibc_req_tuple[0]), int(glibc_req_tuple[1])):
                    return HailoProbeResult(
                        ok=False,
                        backend="wsl",
                        reason=f"glibc too old (have {cur[0]}.{cur[1]}, need >= {int(glibc_req_tuple[0])}.{int(glibc_req_tuple[1])})",
                        details={
                            "profile_id": resolved.profile_id,
                            "wsl_distro": distro_eff or None,
                            "wsl_venv_activate": venv_eff,
                            "glibc": glibc_seen,
                            "glibc_required": f"{int(glibc_req_tuple[0])}.{int(glibc_req_tuple[1])}",
                            "output_tail": gout,
                        },
                    )
        except Exception:
            glibc_seen = None

    try:
        log.info("[hailo][probe][wsl] hw_arch=%s profile=%s distro=%s activate=%s", hw_arch, resolved.profile_id, distro_eff or "", venv_eff)
        log.debug("[hailo][probe][wsl] cmd=%s", cmd)

        # Importing hailo_sdk_client can trigger an *interactive* system requirements check
        # on first use ("Continue? [Y/n]"). In a non-interactive WSL probe this would block
        # forever and end in a timeout. We proactively feed "y".
        proc = _run_owned_subprocess(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            errors="replace",
            env=env,
            input="y\n",
        )
        out = _sanitize_wsl_text((proc.stdout or "") + (proc.stderr or ""))

        cuda_probe = _find_marker_json(out, _CUDA_PROBE_MARKER)

        ok = "__HAILO_PROBE_OK__" in out
        if not ok:
            log.warning(
                "[hailo][probe][wsl] failed rc=%s tail=%s",
                proc.returncode,
                " ".join((out.strip().splitlines()[-5:] if out else ["<no output>"]))
            )
        details = {
            "returncode": proc.returncode,
            "profile_id": resolved.profile_id,
            "wsl_distro": distro_eff or None,
            "wsl_venv_activate": venv_eff,
            "glibc": glibc_seen,
            "glibc_required": (f"{glibc_req_tuple[0]}.{glibc_req_tuple[1]}" if glibc_req_tuple is not None else None),
            "cuda_probe": cuda_probe,
            "output_tail": "\n".join(out.strip().splitlines()[-30:]),
        }
        return HailoProbeResult(ok=ok, backend="wsl", reason=("" if ok else _summarize(out)), details=details)
    except subprocess.TimeoutExpired:
        return HailoProbeResult(ok=False, backend="wsl", reason=f"WSL probe timed out after {timeout_s}s")
    except Exception as e:
        return HailoProbeResult(ok=False, backend="wsl", reason=str(e))


def hailo_probe_auto(
    *,
    backend: str = "auto",
    hw_arch: str = "hailo8",
    wsl_distro: str = "",
    wsl_venv_activate: str = "auto",
    timeout_s: int = 30,
) -> HailoProbeResult:
    mode = normalize_hailo_backend(backend)

    if mode == "subprocess":
        mode = subprocess_backend_for_platform()

    if mode == "local":
        return hailo_probe_local()
    if mode == "venv":
        return hailo_probe_via_venv(hw_arch=hw_arch, venv_activate=wsl_venv_activate, timeout_s=timeout_s)
    if mode == "wsl":
        return hailo_probe_via_wsl(hw_arch=hw_arch, wsl_distro=wsl_distro, wsl_venv_activate=wsl_venv_activate, timeout_s=timeout_s)

    # auto
    prefer_subprocess = auto_prefers_subprocess()
    if not prefer_subprocess and hailo_sdk_available():
        return hailo_probe_local()

    if sys.platform == "win32":
        res = hailo_probe_via_wsl(hw_arch=hw_arch, wsl_distro=wsl_distro, wsl_venv_activate=wsl_venv_activate, timeout_s=timeout_s)
    else:
        res = hailo_probe_via_venv(hw_arch=hw_arch, venv_activate=wsl_venv_activate, timeout_s=timeout_s)

    if bool(getattr(res, "ok", False)) or not hailo_sdk_available():
        return res
    if hailo_sdk_available():
        return hailo_probe_local()
    return res


def _find_result_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON payload following our marker from mixed output."""
    if not text:
        return None
    idx = text.rfind(_WSL_RESULT_MARKER)
    if idx < 0:
        return None

    # The helper prints a single marker line. Other libraries may print to
    # stdout/stderr before/after, so we only parse the first line after marker.
    payload_all = text[idx + len(_WSL_RESULT_MARKER):]
    first_line = payload_all.strip().splitlines()[0].strip() if payload_all else ""
    if not first_line:
        return None

    # 1) Best case: pure JSON
    try:
        return json.loads(first_line)
    except Exception:
        pass

    # 2) Fallback: extract the first {...} block (guards against accidental
    # trailing logs appended on the same line).
    try:
        a = first_line.find("{")
        b = first_line.rfind("}")
        if a >= 0 and b > a:
            return json.loads(first_line[a : b + 1])
    except Exception:
        return None

    return None


@dataclass
class _StreamedSubprocessResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    timeout_kind: Optional[str] = None
    last_stage: Optional[str] = None
    stage_history: Optional[List[Dict[str, Any]]] = None
    elapsed_s: float = 0.0
    cleanup: Optional[Dict[str, Any]] = None


def _hailo_stage_from_line(line: str) -> Optional[str]:
    s = str(line or '').strip().lower()
    # Structured result recipe keys include "calibration" and "part1" even
    # after publication. They are data, not compiler progress messages.
    if not s or s.startswith('__splitpoint_hailo_result__'):
        return None
    if 'statistics collector' in s or s.startswith('calibration:'):
        return 'statistics_collector'
    if 'bias correction' in s:
        return 'bias_correction'
    if 'layer noise analysis' in s or 'full quant analysis' in s:
        return 'layer_noise_analysis'
    if 'searching for a better partition' in s or 'found valid partition' in s or 'iteration #' in s:
        return 'partition_search'
    if 'building optimization options' in s:
        return 'compile_prep'
    if 'single context flow' in s or 'multi context flow' in s or 'allocat' in s or 'validating layers feasibility' in s or 'context:' in s or 'mapping prepost' in s:
        return 'allocation'
    if 'model optimization' in s or 'optimization level' in s:
        return 'optimization'
    if (
        'activation calibration' in s
        or '[hailo][activation' in s
        or 'activation_from_part1' in s
        or ('calib' in s and 'part1' in s)
        or ('[hailo][calib]' in s and 'image_preprocess' in s)
    ):
        return 'activation_calibration'
    if 'translate' in s or 'translation' in s or 'parsing' in s:
        return 'translation'
    if 'compiling kernels' in s or 'building hef' in s or 'successful compilation' in s or 'compiled.hef' in s or 'hef written' in s:
        return 'compile'
    return None


def _env_int_first_positive(*names: str) -> Optional[int]:
    for name in names:
        raw = str(os.environ.get(name) or '').strip()
        if not raw:
            continue
        try:
            val = int(raw)
        except Exception:
            continue
        if val > 0:
            return int(val)
    return None


def _timeout_env_token(*names: str) -> Optional[str]:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return None


def _timeout_disabled_token(value: Any) -> bool:
    """Compatibility wrapper around the canonical timeout-token parser."""

    return is_hailo_timeout_unlimited(value)


def _resolve_hef_timeout_policy(requested_timeout_s: Any) -> Tuple[int, Optional[int]]:
    """Return ``(hard_timeout_s, idle_timeout_s)`` for HEF helpers.

    The historic default remains a 10,800-second hard watchdog.  A caller may
    now explicitly disable only the hard timeout with any canonical unlimited
    token (``0``, ``off``, ``none``, ``unlimited`` or ``disabled``).
    Heartbeats, process ownership, GUI/manual cancellation, and an explicitly
    configured idle watchdog remain active.
    """

    env_hard_raw = _timeout_env_token(
        'ONNX_SPLITPOINT_HAILO_HEF_TIMEOUT_S',
        'OSP_HAILO_HARD_TIMEOUT_S',
    )
    env_idle_raw = _timeout_env_token(
        'ONNX_SPLITPOINT_HAILO_HEF_IDLE_TIMEOUT_S',
        'OSP_HAILO_IDLE_TIMEOUT_S',
    )

    selected_hard = (
        env_hard_raw if env_hard_raw is not None else requested_timeout_s
    )
    hard_timeout_s = parse_hailo_timeout_seconds(
        selected_hard,
        default=3600,
        minimum_enabled_s=60,
        label="Hailo hard timeout",
    )
    # Preserve the historic backend default expansion, but only for the
    # default request.  An explicit zero/token always remained zero above.
    if env_hard_raw is None and hard_timeout_s == 3600:
        hard_timeout_s = 10800

    idle_timeout_s: Optional[int]
    parsed_idle = parse_hailo_timeout_seconds(
        env_idle_raw,
        default=0,
        minimum_enabled_s=1,
        label="Hailo idle timeout",
    )
    idle_timeout_s = None if parsed_idle == 0 else parsed_idle
    return int(hard_timeout_s), idle_timeout_s


def _parse_hailo_duration_to_s(text: str) -> Optional[float]:
    """Parse common Hailo duration formats into seconds.

    Supported examples:
    - ``2m 4s 589ms``
    - ``1h 2m 39s``
    - ``00:08:38.97``
    """

    s = str(text or "").strip()
    if not s:
        return None

    m_hms = re.match(r"^(?:(\d+):)?(\d{2}):(\d{2})(?:\.(\d+))?$", s)
    if m_hms is not None:
        hours = int(m_hms.group(1) or 0)
        minutes = int(m_hms.group(2) or 0)
        seconds = int(m_hms.group(3) or 0)
        frac_s = float(f"0.{m_hms.group(4)}") if m_hms.group(4) else 0.0
        return float(hours * 3600 + minutes * 60 + seconds) + frac_s

    total = 0.0
    matched = False
    for pat, scale in ((r"(\d+)h", 3600.0), (r"(\d+)m", 60.0), (r"(\d+)s", 1.0), (r"(\d+)ms", 0.001)):
        m = re.search(pat, s)
        if m is None:
            continue
        total += float(int(m.group(1))) * scale
        matched = True
    if matched:
        return total
    return None


def _merge_detail_dict(base: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Merge nested result-detail dictionaries without losing existing keys."""

    out = dict(base or {})
    if not extra:
        return out or None
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            merged = dict(out[key])
            merged.update(value)
            out[key] = merged
        else:
            out[key] = value
    return out or None


def _capture_command_snapshot(cmd: List[str], *, timeout_s: float = 2.0, max_chars: int = 4000) -> Optional[Dict[str, Any]]:
    try:
        proc = _run_owned_subprocess(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=timeout_s,
            check=False,
        )
        out = _truncate_log_text(_sanitize_wsl_text(proc.stdout or ''), max_chars=max_chars)
        return {
            'cmd': list(map(str, cmd)),
            'returncode': int(proc.returncode or 0),
            'output': out,
        }
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired as e:
        out = _truncate_log_text(_sanitize_wsl_text((getattr(e, 'stdout', '') or '') + (getattr(e, 'stderr', '') or '')), max_chars=max_chars)
        return {
            'cmd': list(map(str, cmd)),
            'returncode': 124,
            'output': out,
            'timed_out': True,
        }
    except Exception as e:
        return {
            'cmd': list(map(str, cmd)),
            'returncode': None,
            'error': f'{type(e).__name__}: {e}',
        }


def _capture_parent_system_snapshot() -> Dict[str, Any]:
    """Best-effort system diagnostics for timeout/failure reports."""

    snap: Dict[str, Any] = {
        'platform': sys.platform,
        'platform_release': platform.release(),
        'python': sys.version.split()[0],
        'pid': int(os.getpid()),
        'captured_at': float(time.time()),
    }
    commands: Dict[str, Any] = {}

    if os.name == 'nt':
        for name, cmd in (
            ('nvidia_smi', ['nvidia-smi']),
            ('os_info', ['cmd', '/c', 'ver']),
        ):
            res = _capture_command_snapshot(cmd)
            if res is not None:
                commands[name] = res
    else:
        for name, cmd in (
            ('nvidia_smi', ['nvidia-smi']),
            ('free_m', ['free', '-m']),
            ('df_h', ['df', '-h', '.']),
        ):
            res = _capture_command_snapshot(cmd)
            if res is not None:
                commands[name] = res

    if commands:
        snap['commands'] = commands
    return snap


def _extract_hailo_process_summary(
    stdout: str,
    stderr: str,
    *,
    stage_history: Optional[List[Dict[str, Any]]] = None,
    elapsed_s: Optional[float] = None,
    last_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize useful timing/debug signals from Hailo helper stdout/stderr."""

    text = "\n".join([stdout or "", stderr or ""]).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    summary: Dict[str, Any] = {}
    detected: Dict[str, Any] = {}
    algo_times_s: Dict[str, float] = {}
    snr_db: Dict[str, float] = {}
    row_per_cut_hints: List[str] = []
    validator_failed_nodes: List[str] = []

    if stage_history:
        hist: List[Dict[str, Any]] = []
        stage_durations: Dict[str, float] = {}
        prev_t: Optional[float] = None
        prev_stage: Optional[str] = None
        for item in stage_history:
            stage = str(item.get('stage') or '').strip()
            try:
                t_s = float(item.get('t_s') or 0.0)
            except Exception:
                t_s = 0.0
            if not stage:
                continue
            hist.append({'stage': stage, 't_s': round(t_s, 3)})
            if prev_stage is not None and prev_t is not None:
                stage_durations[prev_stage] = round(max(0.0, t_s - prev_t), 3)
            prev_stage = stage
            prev_t = t_s
        if prev_stage is not None and prev_t is not None and elapsed_s is not None:
            stage_durations[prev_stage] = round(max(0.0, float(elapsed_s) - prev_t), 3)
        if hist:
            summary['stage_history'] = hist
        if stage_durations:
            summary['stage_durations_s'] = stage_durations

    if last_stage:
        summary['last_stage'] = str(last_stage)
    if elapsed_s is not None:
        summary['elapsed_s_observed'] = round(float(elapsed_s), 3)

    context_count: Optional[int] = None
    for line in lines:
        low = line.lower()
        if 'calibration set seems to not be normalized' in low:
            detected['normalization_warning'] = True
        if 'single context flow failed' in low:
            detected['single_context_failed'] = True
            summary['single_context_failure'] = line
        if 'using multi-context flow' in low:
            detected['multi_context_used'] = True
        if 'using single-context flow' in low:
            detected['single_context_used'] = True
        if 'watchdog expired' in low:
            detected['watchdog_expired'] = True
        if 'mapping failed' in low:
            detected['mapping_failed'] = True

        m_ctx = re.search(r'found valid partition to\s+(\d+)\s+contexts', low)
        if m_ctx is not None:
            context_count = int(m_ctx.group(1))
        m_apply = re.search(r'applying selected partition to\s+(\d+)\s+contexts', low)
        if m_apply is not None:
            context_count = int(m_apply.group(1))

        m_part = re.search(r'partitioner finished after\s+(\d+)\s+iterations,\s*time it took:\s*(.+)$', line, flags=re.I)
        if m_part is not None:
            summary['partition_iterations'] = int(m_part.group(1))
            dur = _parse_hailo_duration_to_s(m_part.group(2))
            if dur is not None:
                summary['partition_time_s'] = round(dur, 3)

        m_alloc = re.search(r'successful mapping \(allocation time:\s*(.+?)\)$', line, flags=re.I)
        if m_alloc is not None:
            dur = _parse_hailo_duration_to_s(m_alloc.group(1))
            if dur is not None:
                summary['allocation_time_s'] = round(dur, 3)

        m_comp = re.search(r'successful compilation \(compilation time:\s*(.+?)\)$', line, flags=re.I)
        if m_comp is not None:
            dur = _parse_hailo_duration_to_s(m_comp.group(1))
            if dur is not None:
                summary['compilation_time_s'] = round(dur, 3)

        m_opt = re.search(r'Model Optimization Algorithm\s+(.+?)\s+is done \(completion time is\s+(.+?)\)', line, flags=re.I)
        if m_opt is not None:
            dur = _parse_hailo_duration_to_s(m_opt.group(2))
            if dur is not None:
                algo_times_s[str(m_opt.group(1)).strip()] = round(dur, 3)

        m_snr = re.search(r'([^\s]+)\s+SNR:\s*([0-9]+(?:\.[0-9]+)?)\s*dB', line, flags=re.I)
        if m_snr is not None:
            snr_db[str(m_snr.group(1)).strip()] = float(m_snr.group(2))

        m_row_cut = re.search(r'Node needed ROW_PER_CUT due to low halts FPS but was not set:\s*(.+)$', line, flags=re.I)
        if m_row_cut is not None:
            raw_names = [str(x).strip() for x in m_row_cut.group(1).split(',')]
            for name in raw_names:
                if name and name not in row_per_cut_hints:
                    row_per_cut_hints.append(name)

        m_validator = re.search(r'Validator failed on node:\s*([A-Za-z0-9_./-]+)\s+with Agent infeasible', line, flags=re.I)
        if m_validator is not None:
            node_name = str(m_validator.group(1)).strip()
            if node_name and node_name not in validator_failed_nodes:
                validator_failed_nodes.append(node_name)

    if context_count is not None:
        summary['context_count'] = int(context_count)
    if algo_times_s:
        summary['algo_times_s'] = algo_times_s
    if snr_db:
        summary['snr_db'] = snr_db
    if row_per_cut_hints:
        summary['row_per_cut_hints'] = list(row_per_cut_hints)
    if validator_failed_nodes:
        summary['validator_failed_nodes'] = list(validator_failed_nodes)
    if detected:
        summary['detected'] = detected
    return summary


def _build_subprocess_detail_bundle(
    run: _StreamedSubprocessResult,
    stdout: str,
    stderr: str,
    *,
    include_system_snapshot: bool,
) -> Optional[Dict[str, Any]]:
    details: Dict[str, Any] = {}
    if getattr(run, "cleanup", None) is not None:
        details["process_cleanup"] = dict(run.cleanup)
    proc_summary = _extract_hailo_process_summary(
        stdout,
        stderr,
        stage_history=run.stage_history,
        elapsed_s=run.elapsed_s,
        last_stage=run.last_stage,
    )
    if proc_summary:
        details['process_summary'] = proc_summary
    if include_system_snapshot:
        details['system_snapshot'] = _capture_parent_system_snapshot()
    return details or None


def _kill_process_tree(proc: subprocess.Popen[Any], *, grace_s: float = 10.0) -> None:
    terminate_process_tree(proc, grace_s=grace_s)


def _run_streamed_subprocess(
    cmd: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    stdin_yes: bool = False,
    on_log: Optional[Callable[[str, str], None]] = None,
    hard_timeout_s: Optional[int] = None,
    idle_timeout_s: Optional[int] = None,
) -> _StreamedSubprocessResult:
    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    state: Dict[str, Any] = {
        'last_output_ts': time.monotonic(),
        'last_stage': None,
        'stage_history': [],
    }
    t0 = time.monotonic()
    try:
        heartbeat_s = max(0.0, float(os.environ.get('ONNX_SPLITPOINT_HAILO_SUBPROCESS_HEARTBEAT_S', os.environ.get('ONNX_SPLITPOINT_HAILO_HEARTBEAT_S', '60')) or '60'))
    except Exception:
        heartbeat_s = 60.0
    next_heartbeat_ts = t0 + heartbeat_s if heartbeat_s > 0 else 0.0

    def _emit(stream_name: str, line: str) -> None:
        if on_log is None:
            return
        try:
            on_log(stream_name, line)
        except Exception:
            return

    from .process_control import ProcessTreeRegistry
    registry = current_process_registry() or ProcessTreeRegistry()
    cleanup: Dict[str, Any] = {}
    if registry is not None and bool(getattr(registry, 'cancelled', False)):
        message = 'CANCELLED before Hailo subprocess start'
        _emit('status', message)
        return _StreamedSubprocessResult(
            returncode=130,
            stdout='',
            stderr=message,
            last_stage='cancelled',
            stage_history=[],
            elapsed_s=float(time.monotonic() - t0),
        )

    popen_kwargs: Dict[str, Any] = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE,
        'stdin': subprocess.PIPE,
        'text': True,
        'encoding': 'utf-8',
        'errors': 'replace',
        'bufsize': 1,
        'universal_newlines': True,
    }
    if cwd is not None:
        popen_kwargs['cwd'] = str(cwd)
    if env is not None:
        popen_kwargs['env'] = env

    if os.name == 'nt':
        popen_kwargs['creationflags'] = int(getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) or 0)
    else:
        popen_kwargs['start_new_session'] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    if registry is not None:
        registry.register(proc, label='hailo-streamed-subprocess')

    def _terminate_owned(grace_s: float = 3.0) -> None:
        nonlocal cleanup
        if registry is not None:
            cleanup = registry.terminate_registered(proc, grace_s=grace_s)
        else:
            _kill_process_tree(proc, grace_s=grace_s)

    if stdin_yes:
        try:
            if proc.stdin is not None:
                proc.stdin.write('y\n')
                proc.stdin.flush()
        except Exception:
            pass

    def _reader(stream: Any, sink: List[str], stream_name: str) -> None:
        if stream is None:
            return
        try:
            for raw in iter(stream.readline, ''):
                if raw == '' and proc.poll() is not None:
                    break
                line = raw.rstrip('\n')
                if line.endswith('\r'):
                    line = line.rstrip('\r')
                line = _sanitize_wsl_text(line)
                sink.append(line)
                state['last_output_ts'] = time.monotonic()
                stage = _hailo_stage_from_line(line)
                if stage:
                    prev_stage = state.get('last_stage')
                    if prev_stage != stage:
                        state['stage_history'].append({
                            'stage': str(stage),
                            't_s': float(round(time.monotonic() - t0, 3)),
                        })
                    state['last_stage'] = stage
                _emit(stream_name, line)
        finally:
            try:
                stream.close()
            except Exception:
                pass

    t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_lines, 'stdout'), daemon=True)
    t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_lines, 'stderr'), daemon=True)
    t_out.start()
    t_err.start()

    timed_out = False
    cancelled = False
    timeout_kind: Optional[str] = None
    try:
        while True:
            if registry is not None and bool(getattr(registry, 'cancelled', False)):
                cancelled = True
                state['last_stage'] = 'cancelled'
                _emit('status', 'CANCELLED by Evaluation Workflow')
                if proc.poll() is None:
                    _terminate_owned(3.0)
                break
            rc = proc.poll()
            if rc is not None:
                break
            now = time.monotonic()
            if hard_timeout_s is not None and hard_timeout_s > 0 and (now - t0) > float(hard_timeout_s):
                timed_out = True
                timeout_kind = 'hard'
                break
            if heartbeat_s > 0 and now >= next_heartbeat_ts:
                silence_s = now - float(state.get('last_output_ts') or t0)
                elapsed_s = now - t0
                stage = str(state.get('last_stage') or 'unknown')
                _emit('status', f"[hailo][subprocess][heartbeat] still running after {elapsed_s:.0f}s; stage={stage}; silence={silence_s:.0f}s; hard_timeout_s={hard_timeout_s if hard_timeout_s else 'off'}; idle_timeout_s={idle_timeout_s if idle_timeout_s else 'off'}")
                next_heartbeat_ts = now + heartbeat_s
            if idle_timeout_s is not None and idle_timeout_s > 0 and (now - float(state['last_output_ts'])) > float(idle_timeout_s):
                timed_out = True
                timeout_kind = 'idle'
                break
            time.sleep(0.2)

        if timed_out:
            _terminate_owned(10.0)
        try:
            proc.wait(timeout=5)
        except Exception:
            _terminate_owned(0.5)
        # SDK roots may exit while their captured workers still own scratch
        # files or inherited pipes. Retain the component view until all owned
        # children have been cleaned through the existing identity registry.
        _terminate_owned(0.5)
        if int(cleanup.get("remaining_process_count", 0)):
            raise RuntimeError("hailo_owned_children_not_quiescent:" + json.dumps(cleanup, sort_keys=True))
    except BaseException:
        _terminate_owned(0.5)
        t_out.join(timeout=2)
        t_err.join(timeout=2)
        if registry is not None:
            registry.unregister(proc)
        raise

    t_out.join(timeout=2)
    t_err.join(timeout=2)

    if registry is not None:
        registry.unregister(proc)

    stderr_text = _sanitize_wsl_text('\n'.join(stderr_lines))
    if cancelled:
        stderr_text = (stderr_text + '\nCANCELLED by Evaluation Workflow').strip()

    return _StreamedSubprocessResult(
        returncode=130 if cancelled else int(getattr(proc, 'returncode', 0) or 0),
        stdout=_sanitize_wsl_text('\n'.join(stdout_lines)),
        stderr=stderr_text,
        timed_out=bool(timed_out),
        timeout_kind=timeout_kind,
        last_stage=(str(state.get('last_stage')) if state.get('last_stage') else None),
        stage_history=list(state.get('stage_history') or []),
        elapsed_s=float(time.monotonic() - t0),
        cleanup=cleanup,
    )


def hailo_parse_check_via_wsl(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    save_har: bool = True,
    disable_rt_metadata_extraction: bool = True,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
    # WSL bridge settings
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 180,
) -> "HailoParseResult":
    """Run the parse-check inside WSL (Windows host -> WSL2 Linux backend).

    This is intended for the common situation where the Hailo DFC is only
    available as a Linux wheel, but the GUI is running on Windows.

    The function calls a tiny helper script *inside WSL* and parses a structured
    JSON result from mixed stdout/stderr using a marker token.
    """

    t0 = time.time()
    onnx_path = Path(onnx_path)
    if net_name is None:
        net_name = onnx_path.stem

    if sys.platform != "win32":
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error="WSL backend is only available when running on Windows.",
        )

    if not hailo_wsl_available():
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error="WSL backend not available (wsl.exe not found).",
        )

    # Resolve managed venv/distro if requested.
    try:
        mgr = get_dfc_manager()
        resolved = mgr.resolve_wsl_runtime(
            hw_arch=str(hw_arch),
            wsl_distro=_clean_opt_str(wsl_distro),
            wsl_venv_activate=(_clean_opt_str(wsl_venv_activate) or "auto"),
        )
    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"Failed to resolve DFC profile: {e}",
        )

    distro_eff = str(resolved.wsl_distro or "").strip() or None
    venv_eff = str(resolved.wsl_venv_activate or "").strip()
    if not venv_eff:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=(
                f"No managed DFC profile found for hw_arch={hw_arch!r}. "
                "Set an explicit WSL venv path, or add a profile in resources/hailo/profiles.json."
            ),
        )

    # Resolve helper script path on Windows and convert to WSL path.
    helper_win = (Path(__file__).resolve().parent / "wsl_inline_check")
    helper_wsl = windows_path_to_wsl(str(helper_win))

    # Convert ONNX and outdir paths.
    onnx_wsl = windows_path_to_wsl(str(onnx_path.resolve()))
    outdir_wsl = None
    if outdir is not None:
        outdir_wsl = windows_path_to_wsl(str(Path(outdir).resolve()))

    # Build a bash command that activates the venv and runs the helper.
    # NOTE: do not quote paths starting with '~' here; quoting prevents tilde expansion.
    venv_activate = venv_eff

    cmd_parts = [
        "set -e",  # fail fast
        "echo __SPLITPOINT_WSL_BEGIN__",
        f"source {venv_activate}",
        "echo __SPLITPOINT_WSL_VENV_OK__",
        # Make sure we always flush output.
        "export PYTHONUNBUFFERED=1",
        # Self-heal: setuptools 82+ removed pkg_resources, but some Hailo SDK
        # components still import it.
        "python -c \"import pkg_resources\" >/dev/null 2>&1 || "
        "python -m pip install --force-reinstall \"setuptools<82\" >/dev/null 2>&1 || true",
        # Use `python` after venv activation to ensure we run the venv interpreter.
        f"python {_bash_quote(helper_wsl)}"
        f" --onnx {_bash_quote(onnx_wsl)}"
        f" --hw-arch {_bash_quote(str(hw_arch))}"
        f" --net-name {_bash_quote(str(net_name))}"
        f" --fixup {'1' if fixup else '0'}"
        f" --add-conv-defaults {'1' if add_conv_defaults else '0'}"
        f" --save-har {'1' if save_har else '0'}"
        f" --disable-rt-metadata-extraction {'1' if disable_rt_metadata_extraction else '0'}",
    ]

    if outdir_wsl is not None:
        cmd_parts[-1] += f" --outdir {_bash_quote(outdir_wsl)}"
    if start_node_names:
        cmd_parts[-1] += f" --start-node-names-json {_bash_quote(json.dumps(list(start_node_names)))}"
    if end_node_names:
        cmd_parts[-1] += f" --end-node-names-json {_bash_quote(json.dumps(list(end_node_names)))}"

    # net_input_shapes is optional; for now we only support the default inference
    # on the WSL side. (Passing large dicts through CLI quoting is possible but
    # not necessary for the current use case.)

    bash_cmd = " && ".join(cmd_parts)

    wsl_cmd: List[str] = [_wsl_exe()]
    if distro_eff:
        wsl_cmd += ["-d", str(distro_eff)]
    wsl_cmd += ["--", "bash", "-lc", bash_cmd]

    try:
        log.info(
            "[hailo][parse][wsl] hw_arch=%s profile=%s distro=%s activate=%s onnx=%s outdir=%s",
            hw_arch,
            resolved.profile_id,
            distro_eff or "",
            venv_eff,
            onnx_wsl,
            outdir_wsl or "",
        )
        log.debug("[hailo][parse][wsl] cmd=%s", wsl_cmd)

        proc = _run_owned_subprocess(
            wsl_cmd,
            capture_output=True,
            text=True,
            timeout=int(wsl_timeout_s),
            env=dict(os.environ),
            encoding="utf-8",
            errors="replace",
            input="y\n",
        )
    except subprocess.TimeoutExpired:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"WSL hailo check timed out after {wsl_timeout_s}s.",
        )
    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"WSL hailo check failed to launch: {type(e).__name__}: {e}",
        )

    stdout = _sanitize_wsl_text(proc.stdout or "")
    stderr = _sanitize_wsl_text(proc.stderr or "")
    mixed = "\n".join([stdout, stderr]).strip()
    payload = _find_result_json(mixed)

    if payload is None:
        # Provide a short tail for debugging.
        tail = mixed[-4000:] if mixed else "<no stdout/stderr captured>"
        dbg_path = _write_wsl_debug_log(
            outdir,
            filename=f"hailo_wsl_parse_{net_name}_{int(time.time())}.log",
            wsl_cmd=wsl_cmd,
            stdout=stdout,
            stderr=stderr,
        )
        if dbg_path:
            tail = tail + f"\n\n[debug_log] {dbg_path}"
        log.warning(
            "[hailo][parse][wsl] no structured result rc=%s debug_log=%s",
            proc.returncode,
            dbg_path or "-",
        )
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=(
                "WSL hailo check did not return a structured result. "
                f"exit_code={proc.returncode}. tail=\n{tail}\n\n"
                "Details were written to gui.log (Logs tab)."
            ),
        )

    # Map JSON payload back to our result object.
    return HailoParseResult(
        ok=bool(payload.get("ok")),
        elapsed_s=float(payload.get("elapsed_s", time.time() - t0)),
        hw_arch=str(payload.get("hw_arch", hw_arch)),
        net_name=str(payload.get("net_name", net_name)),
        backend=str(payload.get("backend") or "wsl"),
        error=payload.get("error"),
        har_path=payload.get("har_path"),
        fixed_onnx_path=payload.get("fixed_onnx_path"),
        fixup_report=payload.get("fixup_report"),
    )


def hailo_parse_check_via_venv(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    save_har: bool = True,
    disable_rt_metadata_extraction: bool = True,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
    venv_activate: str = "auto",
    timeout_s: int = 180,
) -> "HailoParseResult":
    """Run the parse-check helper inside a managed DFC venv (Linux / WSL).

    This avoids installing Hailo SDK deps into the tool's own Python env.
    """

    if sys.platform == "win32":
        return HailoParseResult(
            ok=False,
            elapsed_s=0.0,
            hw_arch=str(hw_arch),
            net_name=str(net_name or Path(str(onnx_path)).stem),
            backend="venv",
            error="Managed venv backend is not available on Windows (use WSL backend).",
        )

    t0 = time.time()
    onnx_path = Path(str(onnx_path)).expanduser().resolve()
    net_name_eff = str(net_name or onnx_path.stem)
    outdir_path = Path(str(outdir)).expanduser().resolve() if outdir else None
    if outdir_path is not None:
        outdir_path.mkdir(parents=True, exist_ok=True)

    # Resolve managed venv python.
    try:
        profile_id, py, _act = _resolve_managed_venv_python(hw_arch=str(hw_arch), venv_activate=(_clean_opt_str(venv_activate) or "auto"))
    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Failed to resolve managed DFC venv: {e}",
        )

    # Self-heal: setuptools 82+ removed pkg_resources, but some Hailo SDK
    # components still import it.
    try:
        _ = _owned_check_output([str(py), "-c", "import pkg_resources"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        try:
            log.info("[hailo][hef][venv] pkg_resources missing -> installing setuptools<82 (self-heal)")
            _run_owned_subprocess(
                [str(py), "-m", "pip", "install", "--force-reinstall", "setuptools<82"],
                capture_output=True,
                text=True,
                timeout=min(300, max(10, int(timeout_s))),
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass

    # Self-heal: setuptools 82+ removed pkg_resources, but some Hailo SDK
    # components still import it.
    try:
        _ = _owned_check_output([str(py), "-c", "import pkg_resources"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        try:
            log.info("[hailo][parse][venv] pkg_resources missing -> installing setuptools<82 (self-heal)")
            _run_owned_subprocess(
                [str(py), "-m", "pip", "install", "--force-reinstall", "setuptools<82"],
                capture_output=True,
                text=True,
                timeout=min(180, max(10, int(timeout_s))),
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass

    helper = Path(__file__).resolve().parent / "wsl_inline_check"
    if not helper.exists():
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Helper script missing: {helper}",
        )

    cmd: List[str] = [
        str(py),
        str(helper),
        "--onnx",
        str(onnx_path),
        "--hw-arch",
        str(hw_arch),
        "--net-name",
        str(net_name_eff),
        "--fixup",
        "1" if fixup else "0",
        "--add-conv-defaults",
        "1" if add_conv_defaults else "0",
        "--save-har",
        "1" if save_har else "0",
        "--disable-rt-metadata-extraction",
        "1" if disable_rt_metadata_extraction else "0",
    ]
    if outdir_path is not None:
        cmd += ["--outdir", str(outdir_path)]
    if start_node_names:
        cmd += ["--start-node-names-json", json.dumps(list(start_node_names))]
    if end_node_names:
        cmd += ["--end-node-names-json", json.dumps(list(end_node_names))]

    # NOTE: net_input_shapes is ignored for now (same as WSL helper path).
    if net_input_shapes is not None:
        log.debug("[hailo][parse][venv] net_input_shapes ignored (not yet wired through helper)")

    try:
        log.info(
            "[hailo][parse][venv] hw_arch=%s profile=%s python=%s onnx=%s outdir=%s",
            hw_arch,
            profile_id,
            str(py),
            str(onnx_path),
            str(outdir_path or ""),
        )

        # The Hailo SDK drops multiple `hailo_sdk.*.log` files into the current
        # working directory. Run helpers from a dedicated log folder to avoid
        # cluttering the user's project/repo directory.
        from .paths import ensure_dir, splitpoint_logs_dir

        # Hailo-8 and Hailo-10 may compile concurrently.  Give each physical
        # architecture its own SDK cwd and HailoRT log instead of sharing the
        # profile directory, where vendor-created hailo_sdk.* files can collide.
        hailo_log_cwd = ensure_dir(
            splitpoint_logs_dir()
            / "hailo_sdk"
            / str(profile_id)
            / _normalize_hailo_hw_arch(hw_arch)
        )

        # Best-effort log retention for Hailo SDK logs. The SDK tends to drop
        # multiple rotating log files into the working directory.
        try:
            from .log_retention import LogRetentionPolicy, apply_log_retention

            apply_log_retention(
                [hailo_log_cwd],
                policy=LogRetentionPolicy(
                    enabled=True,
                    max_age_days=14,
                    max_files=80,
                    patterns=("*.log",),
                    keep_names=(),
                ),
                recursive=False,
            )
        except Exception:
            pass
        env = _managed_venv_child_env(py)
        # HailoRT can be configured to write logs to a single file.
        env["HAILORT_LOGGER_PATH"] = str(hailo_log_cwd / "hailort.log")

        proc = _run_owned_subprocess(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
            env=env,
            cwd=str(hailo_log_cwd),
            encoding="utf-8",
            errors="replace",
            input="y\n",
        )
    except subprocess.TimeoutExpired:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Venv hailo check timed out after {timeout_s}s.",
        )
    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Venv hailo check failed to launch: {type(e).__name__}: {e}",
        )

    stdout = _sanitize_wsl_text(proc.stdout or "")
    stderr = _sanitize_wsl_text(proc.stderr or "")
    mixed = "\n".join([stdout, stderr]).strip()
    payload = _find_result_json(mixed)

    if payload is None:
        tail = mixed[-4000:] if mixed else "<no stdout/stderr captured>"
        dbg_path = _write_wsl_debug_log(
            str(outdir_path) if outdir_path is not None else None,
            filename=f"hailo_venv_parse_{net_name_eff}_{int(time.time())}.log",
            wsl_cmd=cmd,
            stdout=stdout,
            stderr=stderr,
        )
        if dbg_path:
            tail = tail + f"\n\n[debug_log] {dbg_path}"
        log.warning(
            "[hailo][parse][venv] no structured result rc=%s debug_log=%s",
            proc.returncode,
            dbg_path or "-",
        )
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=(
                "Venv hailo check did not return a structured result. "
                f"exit_code={proc.returncode}. tail=\n{tail}\n\n"
                "Details were written to gui.log (Logs tab)."
            ),
        )

    return HailoParseResult(
        ok=bool(payload.get("ok")),
        elapsed_s=float(payload.get("elapsed_s", time.time() - t0)),
        hw_arch=str(payload.get("hw_arch", hw_arch)),
        net_name=str(payload.get("net_name", net_name_eff)),
        backend="venv",
        error=payload.get("error"),
        har_path=payload.get("har_path"),
        fixed_onnx_path=payload.get("fixed_onnx_path"),
        fixup_report=payload.get("fixup_report"),
    )


def hailo_parse_check_auto(
    onnx_path: Union[str, Path],
    *,
    backend: str = "auto",
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    save_har: bool = True,
    disable_rt_metadata_extraction: bool = True,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
    # WSL bridge settings
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 180,
) -> "HailoParseResult":
    """Convenience wrapper: pick the best available backend."""

    mode = normalize_hailo_backend(backend)
    if mode == "subprocess":
        mode = subprocess_backend_for_platform()

    def _run_local() -> HailoParseResult:
        return hailo_parse_check(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            save_har=save_har,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
        )

    def _run_venv() -> HailoParseResult:
        return hailo_parse_check_via_venv(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            save_har=save_har,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            venv_activate=wsl_venv_activate,
            timeout_s=wsl_timeout_s,
        )

    def _run_wsl() -> HailoParseResult:
        return hailo_parse_check_via_wsl(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            save_har=save_har,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            wsl_distro=wsl_distro,
            wsl_venv_activate=wsl_venv_activate,
            wsl_timeout_s=wsl_timeout_s,
        )

    if mode == "local":
        return _run_local()
    if mode == "venv":
        return _run_venv()
    if mode == "wsl":
        return _run_wsl()

    # auto
    prefer_subprocess = auto_prefers_subprocess()
    if not prefer_subprocess and hailo_sdk_available():
        return _run_local()

    res = _run_wsl() if sys.platform == "win32" else _run_venv()
    if bool(getattr(res, "ok", False)) or bool(getattr(res, "skipped", False)) or not hailo_sdk_available():
        return res
    if hailo_sdk_available():
        return _run_local()
    return res


# ------------------------------- ONNX fixups -------------------------------

def _get_attr(node: onnx.NodeProto, name: str) -> Optional[onnx.AttributeProto]:
    for a in node.attribute:
        if a.name == name:
            return a
    return None


def _set_or_patch_ints_attr(node: onnx.NodeProto, name: str, values: List[int]) -> None:
    """Ensure an INTS attribute exists and has the desired values."""
    a = _get_attr(node, name)
    if a is None:
        node.attribute.append(helper.make_attribute(name, list(values)))
        return
    if a.type == AttributeProto.INTS:
        # Patch in-place
        del a.ints[:]
        a.ints.extend([int(v) for v in values])
        return

    # Wrong type: replace
    node.attribute.remove(a)
    node.attribute.append(helper.make_attribute(name, list(values)))


def _prune_unused_graph_inputs_for_hailo(model: onnx.ModelProto) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    """Remove dangling ONNX graph inputs that are not consumed by the graph.

    Hailo DFC is stricter than ONNX Runtime for split subgraphs.  It can reject
    a model with an error like ``Couldn't find inputs from ONNX proto. Number of
    expected inputs: N, Inputs found: M`` when graph.input still lists boundary
    tensors that are no longer used after a Part2-prefix / host-tail cut.

    The operation is intentionally narrow: initializer inputs are kept, and an
    input that is also a graph output is kept for valid pass-through graphs.
    """

    patched = onnx.ModelProto()
    patched.CopyFrom(model)
    g = patched.graph

    init_names = {str(getattr(t, "name", "") or "") for t in getattr(g, "initializer", [])}
    try:
        init_names.update(str(getattr(t, "name", "") or "") for t in getattr(g, "sparse_initializer", []))
    except Exception:
        pass

    used_by_nodes = set()
    for node in getattr(g, "node", []):
        for name in getattr(node, "input", []):
            if name:
                used_by_nodes.add(str(name))
    graph_outputs = {str(getattr(o, "name", "") or "") for o in getattr(g, "output", [])}

    keep_inputs = []
    removed: List[str] = []
    for vi in list(getattr(g, "input", [])):
        name = str(getattr(vi, "name", "") or "")
        if (not name) or name in init_names or name in used_by_nodes or name in graph_outputs:
            keep_inputs.append(vi)
        else:
            removed.append(name)

    if removed:
        del g.input[:]
        g.input.extend(keep_inputs)

    return patched, {
        "unused_graph_inputs_pruned": int(len(removed)),
        "unused_graph_input_names": list(removed),
    }


def fix_onnx_for_hailo(
    model: onnx.ModelProto,
    *,
    add_conv_defaults: bool = True,
) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    """Apply a small set of pragmatic ONNX fixups that help the Hailo parser.

    This is intentionally conservative: it fills in some missing Conv/
    ConvTranspose attributes, optionally adds a few default attributes, and
    prunes dangling graph inputs that can appear in generated split-prefix
    ONNXs.  The latter is important for YOLO Part2 accelerator-prefix models:
    after cutting before the DFL/decode tail, some original boundary tensors may
    no longer feed any node, while Hailo still treats them as required parser
    inputs.

    Returns (patched_model, report).
    """

    patched = onnx.ModelProto()
    patched.CopyFrom(model)

    report: Dict[str, Any] = {
        "kernel_shape_patched": 0,
        "conv_defaults_added": 0,
        "unused_graph_inputs_pruned": 0,
        "unused_graph_input_names": [],
        "notes": [],
    }

    patched, prune_report = _prune_unused_graph_inputs_for_hailo(patched)
    report["unused_graph_inputs_pruned"] = int(prune_report.get("unused_graph_inputs_pruned") or 0)
    report["unused_graph_input_names"] = list(prune_report.get("unused_graph_input_names") or [])
    if report["unused_graph_inputs_pruned"]:
        report["notes"].append(
            "Pruned unused graph.input entries before Hailo parsing: "
            + ", ".join(report["unused_graph_input_names"][:12])
            + (" ..." if len(report["unused_graph_input_names"]) > 12 else "")
        )

    g = patched.graph
    # Initializers are needed to infer some attributes (kernel from weight shape)
    init_by_name = {i.name: i for i in g.initializer}

    for n in g.node:
        if n.op_type not in {"Conv", "ConvTranspose"}:
            continue

        # ---- kernel_shape ----
        if _get_attr(n, "kernel_shape") is None:
            # Try to infer from weight tensor W (second input)
            if len(n.input) >= 2 and n.input[1] in init_by_name:
                W = init_by_name[n.input[1]]
                # Conv: [M, C/group, kH, kW]
                # ConvTranspose: [C, M/group, kH, kW]
                if len(W.dims) >= 4:
                    kh = int(W.dims[-2])
                    kw = int(W.dims[-1])
                    _set_or_patch_ints_attr(n, "kernel_shape", [kh, kw])
                    report["kernel_shape_patched"] += 1
                else:
                    report["notes"].append(
                        f"Could not infer kernel_shape for {n.op_type} '{n.name or '(unnamed)'}' (W dims={list(W.dims)})"
                    )

        if add_conv_defaults:
            # Add defaults if missing (ONNX spec defaults, but some parsers want explicit)
            # strides default: [1, 1]
            if _get_attr(n, "strides") is None:
                _set_or_patch_ints_attr(n, "strides", [1, 1])
                report["conv_defaults_added"] += 1

            # dilations default: [1, 1]
            if _get_attr(n, "dilations") is None:
                _set_or_patch_ints_attr(n, "dilations", [1, 1])
                report["conv_defaults_added"] += 1

            # pads default: [0, 0, 0, 0]
            if _get_attr(n, "pads") is None:
                _set_or_patch_ints_attr(n, "pads", [0, 0, 0, 0])
                report["conv_defaults_added"] += 1

    return patched, report


def infer_net_input_shapes_from_model(model: onnx.ModelProto) -> Optional[Union[List[int], Dict[str, List[int]]]]:
    """Infer a net_input_shapes structure from ONNX graph inputs.

    Hailo's `translate_onnx_model` accepts either:
      - a single shape list (for single-input networks)
      - a dict input_name -> shape list (for multi-input networks)

    We return None if all shapes are already fully static (no unknown dims),
    because in that case the translator usually does not need an override.
    """

    g = model.graph
    init_names = {i.name for i in g.initializer}
    inputs = [vi for vi in g.input if vi.name not in init_names]
    if not inputs:
        return None

    def _shape_of_vi(vi: onnx.ValueInfoProto) -> List[int]:
        tt = vi.type.tensor_type
        dims: List[int] = []
        if not tt.HasField("shape"):
            return []
        for d in tt.shape.dim:
            if d.HasField("dim_value") and int(d.dim_value) > 0:
                dims.append(int(d.dim_value))
            else:
                # Unknown/param dimension -> replace with 1 (safe default for feasibility checks)
                dims.append(1)
        return dims

    shapes: Dict[str, List[int]] = {vi.name: _shape_of_vi(vi) for vi in inputs}

    # If all dims are static already, return None.
    # (We cannot reliably detect if original was "unknown"; but if we inserted 1s,
    #  the shape will still look static. That's fine: it's a feasibility check.)
    if len(shapes) == 1:
        return list(next(iter(shapes.values())))
    return shapes



def _normalize_optional_node_name_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        raw = str(value).strip()
        if not raw:
            return None
        parts = [part.strip() for part in raw.split(',') if part.strip()]
        return parts or None
    out: List[str] = []
    try:
        for item in list(value):
            s = str(item or '').strip()
            if s and s not in out:
                out.append(s)
    except Exception:
        s = str(value or '').strip()
        if s:
            out.append(s)
    return out or None


def _apply_translate_node_overrides(kwargs: Dict[str, Any], *, start_node_names: Any = None, end_node_names: Any = None) -> Dict[str, Any]:
    start_nodes = _normalize_optional_node_name_list(start_node_names)
    end_nodes = _normalize_optional_node_name_list(end_node_names)
    if start_nodes:
        kwargs['start_node_names'] = start_nodes
    if end_nodes:
        kwargs['end_node_names'] = end_nodes
    return kwargs


# ------------------------------- Parse check -------------------------------


@dataclass
class HailoParseResult:
    ok: bool
    elapsed_s: float
    hw_arch: str
    net_name: str
    backend: Optional[str] = None
    error: Optional[str] = None
    har_path: Optional[str] = None
    fixed_onnx_path: Optional[str] = None
    fixup_report: Optional[Dict[str, Any]] = None


def hailo_parse_check(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    save_har: bool = True,
    disable_rt_metadata_extraction: bool = True,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
) -> HailoParseResult:
    """Run a *parse/translate-only* feasibility check via the Hailo SDK.

    This calls `ClientRunner.translate_onnx_model(...)` and treats success as
    "Hailo can translate this ONNX graph".
    """

    t0 = time.time()
    onnx_path = Path(onnx_path)
    if net_name is None:
        net_name = onnx_path.stem

    out_dir = Path(outdir) if outdir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from hailo_sdk_client import ClientRunner
    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(_normalize_hailo_hw_arch(hw_arch)),
            net_name=str(net_name),
            backend="local",
            error=f"Hailo SDK not available: {e}",
        )

    hw_arch_eff = _normalize_hailo_hw_arch(hw_arch)
    if hw_arch_eff != str(hw_arch or "").strip().lower():
        log.warning("[hailo] hw_arch alias: '%s' -> '%s'", str(hw_arch), hw_arch_eff)

    fixed_path: Optional[Path] = None
    fixup_report: Optional[Dict[str, Any]] = None
    model_for_parse = onnx_path
    if fixup:
        try:
            m = onnx.load(str(onnx_path))
            m2, rep = fix_onnx_for_hailo(m, add_conv_defaults=add_conv_defaults)
            fixup_report = rep
            if out_dir is not None:
                fixed_path = out_dir / (onnx_path.stem + "_hailo_fixed.onnx")
            else:
                fixed_path = onnx_path.parent / (onnx_path.stem + "_hailo_fixed.onnx")
            onnx.save(m2, str(fixed_path))
            model_for_parse = fixed_path
        except Exception as e:
            # Fixup failed; fall back to original
            fixup_report = {"error": str(e)}
            model_for_parse = onnx_path

    # If user didn't provide shapes, try to infer.
    inferred_default_net_input_shapes = None
    try:
        m_tmp = onnx.load(str(model_for_parse))
        inferred_default_net_input_shapes = infer_net_input_shapes_from_model(
            m_tmp
        )
        if net_input_shapes is None:
            net_input_shapes = inferred_default_net_input_shapes
    except Exception:
        if net_input_shapes is None:
            net_input_shapes = None

    try:
        runner = ClientRunner(hw_arch=str(hw_arch_eff))
        translate_kwargs = _apply_translate_node_overrides({
            'model': str(model_for_parse),
            'net_name': str(net_name),
            'net_input_shapes': net_input_shapes,
            # Keep this on by default: avoids parsing issues with missing RT metadata
            'disable_rt_metadata_extraction': bool(disable_rt_metadata_extraction),
        }, start_node_names=start_node_names, end_node_names=end_node_names)
        runner.translate_onnx_model(**translate_kwargs)

        har_path = None
        if save_har and out_dir is not None:
            har_path = str(out_dir / "parsed.har")
            runner.save_har(har_path)

        return HailoParseResult(
            ok=True,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend="local",
            har_path=har_path,
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
        )

    except Exception as e:
        return HailoParseResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend="local",
            error=str(e),
            har_path=None,
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
        )


# ------------------------------- HEF build -------------------------------


@dataclass
class HailoHefBuildResult:
    ok: bool
    elapsed_s: float
    hw_arch: str
    net_name: str
    backend: Optional[str] = None
    error: Optional[str] = None
    hef_path: Optional[str] = None
    parsed_har_path: Optional[str] = None
    quant_har_path: Optional[str] = None
    fixed_onnx_path: Optional[str] = None
    fixup_report: Optional[Dict[str, Any]] = None
    skipped: bool = False
    calib_info: Optional[Dict[str, Any]] = None
    returncode: Optional[int] = None
    debug_log: Optional[str] = None
    timed_out: bool = False
    timeout_kind: Optional[str] = None
    last_stage: Optional[str] = None
    failure_kind: Optional[str] = None
    unsupported_reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


_HAILO_HEF_RESULT_FIELDS = {f.name for f in fields(HailoHefBuildResult)}


def _make_hef_result(**kwargs: Any) -> HailoHefBuildResult:
    """Build a HEF result without crashing on newly added metadata keys."""

    data: Dict[str, Any] = {}
    extras: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _HAILO_HEF_RESULT_FIELDS:
            data[key] = value
        else:
            extras[key] = value

    detail_dict = dict(data.get("details") or {})
    if extras:
        detail_dict.setdefault("extra_fields", {}).update(extras)
    if detail_dict:
        data["details"] = detail_dict

    return HailoHefBuildResult(**data)


def _hailo_workspace_env_int(name: str, default: int) -> int:
    try:
        return max(0, int(str(os.environ.get(name, default)).strip()))
    except (TypeError, ValueError, OverflowError):
        return max(0, int(default))


def _hailo_workspace_env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(str(os.environ.get(name, default)).strip()))
    except (TypeError, ValueError, OverflowError):
        return max(0.0, float(default))


def hailo_dfc_workspace_preflight(
    path: Union[str, Path],
    *,
    calibration_count: int,
    input_shapes: Sequence[Sequence[int]],
) -> Dict[str, Any]:
    """Check local DFC workspace bytes and inodes before SDK dispatch.

    DFC Bias Correction can materialize roughly 24 float32 calibration-set
    equivalents.  For the observed 500 x 640 x 640 x 3 campaign this predicts
    about 54.9 GiB, matching the DFC's own runtime estimate.  The preflight
    adds a bounded reserve and records the exact calculation so an admission
    failure is distinguishable from a compiler/model failure.
    """

    enabled = str(
        os.environ.get(
            "ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_PREFLIGHT", "1",
        )
    ).strip().lower() not in {"0", "false", "no", "off"}
    contract_problems: List[str] = []
    try:
        count = int(calibration_count)
        if isinstance(calibration_count, bool) or count <= 0:
            raise ValueError("nonpositive calibration count")
    except (TypeError, ValueError, OverflowError):
        count = 0
        contract_problems.append("effective_calibration_count_unresolved")
    elements_per_sample = 0
    normalized_shapes: List[List[int]] = []
    for raw_shape in input_shapes:
        try:
            shape = [int(value) for value in raw_shape]
        except (TypeError, ValueError, OverflowError):
            contract_problems.append("calibration_identity_shapes_unresolved")
            continue
        if not shape or any(value <= 0 for value in shape):
            contract_problems.append("calibration_identity_shapes_unresolved")
            continue
        elements = 1
        for value in shape:
            elements *= int(value)
        elements_per_sample += elements
        normalized_shapes.append(shape)

    if not normalized_shapes:
        contract_problems.append("calibration_identity_shapes_unresolved")
    if not str(path or "").strip():
        contract_problems.append("workspace_target_unresolved")
    calibration_bytes = int(count * elements_per_sample * 4)
    multiplier = _hailo_workspace_env_float(
        "ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_MULTIPLIER", 24.0,
    )
    reserve_bytes = _hailo_workspace_env_int(
        "ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_RESERVE_BYTES",
        2 * 1024 * 1024 * 1024,
    )
    floor_bytes = _hailo_workspace_env_int(
        "ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_BYTES",
        4 * 1024 * 1024 * 1024,
    )
    required_bytes = max(
        floor_bytes,
        int(calibration_bytes * multiplier) + reserve_bytes,
    )
    required_inodes = _hailo_workspace_env_int(
        "ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_INODES", 50_000,
    )
    inspection = inspect_write_target(Path(path))
    problems: List[str] = []
    if enabled:
        if not inspection.writable:
            problems.append(str(inspection.reason or "workspace_not_writable"))
        if int(inspection.free_bytes) < required_bytes:
            problems.append(
                f"free_bytes={inspection.free_bytes}<required={required_bytes}"
            )
        if int(inspection.free_inodes) < required_inodes:
            problems.append(
                f"free_inodes={inspection.free_inodes}"
                f"<required={required_inodes}"
            )
    return {
        "schema": "onnx-splitpoint/hailo-dfc-workspace-preflight",
        "schema_version": 1,
        "status": (
            "disabled" if not enabled else "unknown" if contract_problems
            else "failed" if problems else "passed"
        ),
        "estimate_complete": not contract_problems,
        "enabled": enabled,
        "workspace": inspection.as_dict(),
        "calculation": {
            "calibration_count": count,
            "input_shapes": normalized_shapes,
            "float32_calibration_bytes": calibration_bytes,
            "workspace_multiplier": multiplier,
            "reserve_bytes": reserve_bytes,
            "floor_bytes": floor_bytes,
            "required_free_bytes": required_bytes if not contract_problems else None,
            "required_free_inodes": required_inodes,
        },
        "problems": list(dict.fromkeys(contract_problems)) + problems,
    }



def _classify_hailo_failure_text(text: str) -> Dict[str, Any]:
    """Return structured failure metadata for common Hailo DFC host failures."""
    low = str(text or "").lower()
    out: Dict[str, Any] = {}
    if any(tok in low for tok in (
        "no space left on device",
        "errno 28",
        "disk quota exceeded",
    )):
        out.update({
            "failure_kind": "local_dfc_workspace_exhausted",
            "error_class": "local_dfc_workspace_exhausted",
            "root_cause_hint": "local_dfc_workspace_enospc",
            "diagnostic_hint": (
                "The local filesystem ran out of bytes or inodes during DFC "
                "execution. This is infrastructure capacity exhaustion, not "
                "evidence that the splitpoint is semantically invalid."
            ),
            "timed_out": False,
        })
    elif any(tok in low for tok in (
        "mapping failed (timeout",
        "watchdog expired after",
        "mapping failed: timeout",
    )):
        out.update({
            "failure_kind": "hailo_dfc_mapping_timeout",
            "error_class": "hailo_dfc_mapping_timeout",
            "root_cause_hint": "hailo_dfc_internal_mapping_watchdog",
            "diagnostic_hint": (
                "The Hailo compiler's internal mapper exhausted its watchdog "
                "budget. The compiler invocation returned normally, but the "
                "artifact outcome is a mapping timeout."
            ),
            "timed_out": True,
            "timeout_kind": "hailo_dfc_mapping_watchdog",
        })
    elif any(tok in low for tok in (
        "cudnn_status_execution_failed",
        "no algorithm worked",
        "unknown cudnn status",
        "cuda_dnn.cc",
        "xla/stream_executor/cuda",
    )):
        out.update({
            "failure_kind": "hailo_dfc_cuda_cudnn_failure",
            "error_class": "hailo_dfc_cuda_cudnn_failure",
            "root_cause_hint": "tensorflow_xla_conv2d_no_algorithm_or_cudnn_failure",
            "diagnostic_hint": "Hailo DFC failed in TensorFlow/XLA/cuDNN Conv2D profiling. This is a host CUDA/cuDNN environment issue, not proof that the splitpoint is semantically invalid. Prefer CPU-only DFC build or repair the DFC CUDA stack.",
        })
    elif "hailo sdk not available" in low and "no module named" in low:
        out.update({
            "failure_kind": "hailo_dfc_import_failed",
            "error_class": "hailo_dfc_import_failed",
            "root_cause_hint": "hailo_sdk_client_import_failed",
        })
    return out

def _hef_result_from_payload(
    payload: Optional[Dict[str, Any]],
    *,
    elapsed_default: float,
    hw_arch: str,
    net_name: str,
    backend_default: str,
    returncode: Optional[int] = None,
    last_stage: Optional[str] = None,
    debug_log: Optional[str] = None,
) -> HailoHefBuildResult:
    body = dict(payload or {})
    body["ok"] = bool(body.get("ok"))
    body["elapsed_s"] = float(body.get("elapsed_s", elapsed_default))
    body["hw_arch"] = str(body.get("hw_arch", hw_arch))
    body["net_name"] = str(body.get("net_name", net_name))
    body["backend"] = str(body.get("backend") or backend_default)

    if returncode is not None and body.get("returncode") is None:
        body["returncode"] = int(returncode)
    if debug_log and not body.get("debug_log"):
        body["debug_log"] = str(debug_log)
    if body["ok"] and not body.get("last_stage"):
        # The managed child's completed phase is more precise than a trailing
        # SDK prose line. Failed results keep their primary failure stage.
        for section in (body.get("details"), body.get("calib_info")):
            events = section.get("phase_events") if isinstance(section, Mapping) else None
            if isinstance(events, list) and events:
                event = events[-1]
                if (isinstance(event, Mapping) and event.get("state") == "completed"
                        and event.get("phase") in {"sdk_initialization", "translate",
                            "calibration_materialization", "optimize", "compile", "publication"}):
                    body["last_stage"] = str(event["phase"])
                    break
    if last_stage and not body.get("last_stage"):
        body["last_stage"] = str(last_stage)

    # Classify common host-environment failures before returning.  Older helper
    # code surfaced TensorFlow/cuDNN build failures as "Hailo SDK not available",
    # which made reports look like a missing SDK instead of a host CUDA/cuDNN
    # failure.  Preserve the raw error but add machine-readable metadata.
    if not bool(body.get("ok")):
        combined_err = "\n".join(str(body.get(k) or "") for k in ("error", "stderr", "stdout"))
        try:
            details_for_text = body.get("details")
            if isinstance(details_for_text, dict):
                combined_err += "\n" + json.dumps(details_for_text, ensure_ascii=False)[:12000]
        except Exception:
            pass
        cls = _classify_hailo_failure_text(combined_err)
        if cls:
            if not body.get("failure_kind"):
                body["failure_kind"] = cls.get("failure_kind")
            if cls.get("timed_out") is True:
                body["timed_out"] = True
            if cls.get("timeout_kind") and not body.get("timeout_kind"):
                body["timeout_kind"] = cls.get("timeout_kind")
            details = dict(body.get("details") or {})
            details.setdefault("error_class", cls.get("error_class"))
            details.setdefault("root_cause_hint", cls.get("root_cause_hint"))
            details.setdefault("diagnostic_hint", cls.get("diagnostic_hint"))
            body["details"] = details

    # Best-effort: if the helper already persisted a result JSON and that path is
    # directly accessible from this process (managed venv / native Linux), refresh
    # it with any parent-side metadata we merged in here (e.g. stage summary).
    try:
        p_raw = body.get("result_json_path")
        if isinstance(p_raw, str) and p_raw.strip():
            p_json = Path(p_raw).expanduser()
            if p_json.exists():
                p_json.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return _make_hef_result(**body)


def _recover_hef_result_from_compiled_artifact(
    outdir: Optional[Union[str, Path]],
    *,
    elapsed_s: float,
    hw_arch: str,
    net_name: str,
    backend: str,
    returncode: Optional[int],
    debug_log: Optional[str] = None,
    last_stage: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Optional[HailoHefBuildResult]:
    """Recover a successful HEF build when helper stdout lost the marker.

    Some long DFC/Hailo builds complete successfully and leave a valid
    ``compiled.hef`` in the requested output directory, but the wrapper process
    does not return the single-line structured marker.  Treating that as a hard
    compile failure discards expensive, valid benchmark candidates.  Recovery is
    deliberately conservative: rc==0, a non-empty compiled.hef, and a valid
    v2 build receipt are all required.  A bare HEF cannot prove which image
    geometry was used during calibration.
    """
    try:
        if returncode not in (0, None):
            return None
        if outdir is None:
            return None
        out_p = Path(str(outdir)).expanduser().resolve()
        hef = out_p / "compiled.hef"
        if not hef.is_file() or hef.stat().st_size <= 0:
            return None
        snapshot = hef.resolve()
        receipt = _load_valid_hailo_receipt(
            snapshot, expected_net_name=str(net_name), allow_legacy_v2=True,
        )
        if receipt is None:
            return None
        if str(receipt.get("hw_arch") or "") != str(_normalize_hailo_hw_arch(hw_arch)):
            return None
        snapshot = _publish_hailo_bundle(
            source_hef=snapshot, destination=hef, receipt=receipt,
            source="structured_result_recovery",
        )
        parsed = out_p / "parsed.har"
        quant = out_p / "quantized.har"
        fixed = None
        try:
            fixed_hits = sorted(out_p.glob("*_hailo_fixed.onnx"))
            fixed = fixed_hits[0] if fixed_hits else None
        except Exception:
            fixed = None
        rec_details = _merge_detail_dict(
            details,
            {
                "recovered_from_missing_structured_result": True,
                "recovery_reason": "compiled.hef exists after helper returncode 0",
                "recovered_hef_size_bytes": int(snapshot.stat().st_size),
                "build_receipt": receipt,
            },
        )
        result = _make_hef_result(
            ok=True,
            elapsed_s=float(elapsed_s),
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend=str(backend),
            error=None,
            hef_path=str(hef),
            parsed_har_path=(str(parsed) if parsed.is_file() else None),
            quant_har_path=(str(quant) if quant.is_file() else None),
            fixed_onnx_path=(str(fixed) if fixed is not None and fixed.is_file() else None),
            returncode=int(returncode or 0),
            debug_log=debug_log,
            last_stage=last_stage,
            failure_kind=None,
            details=rec_details,
        )
        try:
            (out_p / "hailo_hef_build_result.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return result
    except Exception:
        return None


def _safe_filename(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return "model"
    # Keep it cross-platform
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("._") or "model"


_CALIB_ITEM_EXTS = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_npy_any(path: Path) -> np.ndarray:
    arr = np.load(path)
    # Support .npz
    if isinstance(arr, np.lib.npyio.NpzFile):
        keys = list(arr.keys())
        if not keys:
            raise ValueError(f"Empty npz: {path}")
        # common keys
        for k in ["image", "images", "input", "data", "arr_0"]:
            if k in keys:
                return np.asarray(arr[k])
        return np.asarray(arr[keys[0]])
    return np.asarray(arr)


def _iter_calib_items(calib_dir: Path, *, recursive: bool = True) -> List[Path]:
    if not calib_dir.exists():
        return []
    try:
        if recursive:
            items = [p for p in calib_dir.rglob('*') if p.is_file() and p.suffix.lower() in _CALIB_ITEM_EXTS]
        else:
            items = [p for p in calib_dir.iterdir() if p.is_file() and p.suffix.lower() in _CALIB_ITEM_EXTS]
    except Exception:
        return []
    return sorted(items)


def _scan_calib_dir(calib_dir: Path, *, recursive: bool = True, limit: Optional[int] = None) -> Dict[str, Any]:
    if recursive:
        items = _iter_calib_items(calib_dir, recursive=True)
    else:
        items = _iter_calib_items(calib_dir, recursive=False)
    if limit is not None and int(limit) > 0:
        items = items[: int(limit)]

    array_items = [p for p in items if p.suffix.lower() in {'.npy', '.npz'}]
    image_items = [p for p in items if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}]

    kind = 'empty'
    if array_items and image_items:
        kind = 'mixed'
    elif array_items:
        kind = 'array'
    elif image_items:
        kind = 'image'

    return {
        'dir': str(calib_dir),
        'recursive': bool(recursive),
        'kind': kind,
        'count': int(len(items)),
        'array_count': int(len(array_items)),
        'image_count': int(len(image_items)),
        'suffixes': sorted({p.suffix.lower() for p in items}),
        'items': items,
        'preview': [str(p) for p in items[: min(8, len(items))]],
    }


def _load_calib_item_any(path: Path) -> np.ndarray:
    if path.suffix.lower() in ('.npy', '.npz'):
        return _load_npy_any(path)

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f'Pillow is required to load image calibration items: {type(exc).__name__}: {exc}')

    with Image.open(path) as im:
        if im.mode != 'RGB':
            im = im.convert('RGB')
        return np.asarray(im)


def _coerce_hwc_image(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3:
        raise ValueError(f'Calibration item is not image-like. shape={tuple(x.shape)}')
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.shape[-1] not in (1, 3):
        raise ValueError(f'Unsupported calibration image channels: shape={tuple(x.shape)}')
    return np.ascontiguousarray(x)


def _resize_hwc_image(
    arr: np.ndarray,
    target_h: Optional[int],
    target_w: Optional[int],
    *,
    target_c: Optional[int] = None,
) -> np.ndarray:
    x = _coerce_hwc_image(arr)

    if target_c is not None:
        if x.shape[-1] == target_c:
            pass
        elif x.shape[-1] == 1 and target_c == 3:
            x = np.repeat(x, 3, axis=-1)
        elif x.shape[-1] == 3 and target_c == 1:
            x = np.mean(x, axis=-1, keepdims=True)
        else:
            raise ValueError(f'Cannot convert calibration channels {x.shape[-1]} -> {target_c}')

    if target_h is None or target_w is None or (x.shape[0] == int(target_h) and x.shape[1] == int(target_w)):
        return np.ascontiguousarray(x)

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(f'Pillow is required to resize image calibration items: {type(exc).__name__}: {exc}')

    is_float = np.issubdtype(x.dtype, np.floating)
    unit_float = bool(is_float and x.size and float(np.nanmax(x)) <= 1.5)
    if is_float:
        if unit_float:
            x8 = np.clip(x, 0.0, 1.0) * 255.0
        else:
            x8 = np.clip(x, 0.0, 255.0)
        x8 = x8.astype(np.uint8)
    elif x.dtype != np.uint8:
        x8 = np.clip(x, 0, 255).astype(np.uint8)
    else:
        x8 = x

    if x8.shape[-1] == 1:
        im = Image.fromarray(x8[..., 0], mode='L')
        im = im.resize((int(target_w), int(target_h)), resample=Image.BILINEAR)
        y = np.asarray(im)[..., None]
    else:
        im = Image.fromarray(x8, mode='RGB')
        im = im.resize((int(target_w), int(target_h)), resample=Image.BILINEAR)
        y = np.asarray(im)

    if is_float:
        y = y.astype(np.float32)
        if unit_float:
            y /= 255.0
    return np.ascontiguousarray(y)


def _hn_get_shape(meta: Dict[str, Any]) -> Optional[List[int]]:
    """Best-effort extract of an input-layer shape from HN metadata.

    Returns the *per-sample* shape (no dataset dim). For many CV models, this is
    [H, W, C] (NHWC).
    """

    cand_keys = ["input_shape", "output_shape", "shape", "output_shapes", "input_shapes"]
    shape = None
    for k in cand_keys:
        if k not in meta:
            continue
        v = meta.get(k)
        if isinstance(v, list) and v and isinstance(v[0], list):
            v = v[0]
        if isinstance(v, list) and v:
            shape = v
            break
    if shape is None:
        return None

    dims: List[Optional[int]] = []
    for d in shape:
        if d is None:
            dims.append(None)
        elif isinstance(d, int):
            dims.append(None if d <= 0 else int(d))
        else:
            dims.append(None)

    # Drop a leading batch dim if present
    if len(dims) >= 2 and (dims[0] is None or dims[0] == 1):
        dims = dims[1:]

    if any(d is None for d in dims):
        return None
    return [int(d) for d in dims]  # type: ignore


def _sort_hn_input_layers(hn_layers: Dict[str, Any]) -> List[str]:
    inputs = []
    for name, meta in hn_layers.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("type") == "input_layer":
            inputs.append(str(name))

    def keyfn(name: str) -> Tuple[int, str]:
        m = re.search(r"input_layer(\d+)$", name)
        if m:
            try:
                return (int(m.group(1)), name)
            except Exception:
                return (9999, name)
        return (9999, name)

    return sorted(inputs, key=keyfn)


def _estimate_bytes(shape: List[int], n: int, dtype_bytes: int = 4) -> int:
    total = dtype_bytes
    for d in shape:
        try:
            total *= int(d)
        except Exception:
            return 0
    return int(total * int(n))


def _calibration_storage_mode() -> str:
    raw = str(os.environ.get("ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE") or os.environ.get("ONNX_SPLITPOINT_HAILO_CALIB_STORAGE") or "memory").strip().lower()
    return raw if raw in {"memory", "memmap"} else "memory"


def _calibration_memory_cap_bytes(default_mb: int = 256) -> int:
    try:
        mb = max(32, int(float(os.environ.get("ONNX_SPLITPOINT_HAILO_CALIB_CAP_MB", default_mb))))
    except Exception:
        mb = default_mb
    return int(mb) * 1024 * 1024


def _clamp_calib_count(shape: List[int], requested: int, *, cap_bytes: int | None = None, storage: str | None = None) -> int:
    """Return the executable calibration count for the selected storage mode.

    Standard/Final use a disk-backed memmap, so the requested sample count is
    not silently reduced to 445/54 by a 256 MiB in-memory array cap.
    """
    n = max(1, int(requested))
    mode = str(storage or _calibration_storage_mode()).strip().lower()
    if mode == "memmap":
        return n
    cap = int(cap_bytes or _calibration_memory_cap_bytes())
    est = _estimate_bytes(shape, n, dtype_bytes=4)
    if est <= 0 or est <= cap:
        return n
    per = max(1, _estimate_bytes(shape, 1, dtype_bytes=4))
    if per <= 0:
        return n
    return min(n, max(1, int(cap // per)))


def _resolve_hailo_calibration_storage(requested: int) -> str:
    """Resolve ``auto`` before the cache identity is computed.

    The selected storage mode changes whether the requested calibration count
    is executable under the memory cap.  It is therefore part of the build
    identity, not an implementation detail that may be chosen after a cache
    lookup.
    """

    raw = str(
        os.environ.get("ONNX_SPLITPOINT_HAILO_CALIBRATION_STORAGE") or "auto"
    ).strip().lower()
    if raw == "auto":
        return "memmap" if int(requested) > 64 else "memory"
    if raw not in {"memory", "memmap"}:
        raise ValueError(f"unsupported_hailo_calibration_storage:{raw}")
    return raw


def _hailo_calibration_shape_candidates(
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]],
    preprocessing_contract: Mapping[str, Any],
) -> List[List[int]]:
    """Return deterministic positive shapes for pre-cache count clamping."""

    raw_shapes: List[Any]
    if isinstance(net_input_shapes, Mapping):
        raw_shapes = list(net_input_shapes.values())
    elif isinstance(net_input_shapes, list):
        raw_shapes = [net_input_shapes]
    else:
        raw_shapes = []
    shapes: List[List[int]] = []
    for raw_shape in raw_shapes:
        if not isinstance(raw_shape, (list, tuple)):
            continue
        try:
            shape = [int(value) for value in raw_shape]
        except (TypeError, ValueError, OverflowError):
            continue
        if shape and all(value > 0 for value in shape):
            shapes.append(shape)
    if not shapes:
        target_hw = preprocessing_contract.get("target_hw")
        if isinstance(target_hw, (list, tuple)) and len(target_hw) == 2:
            try:
                height, width = (int(target_hw[0]), int(target_hw[1]))
            except (TypeError, ValueError, OverflowError):
                height = width = 0
            if height > 0 and width > 0:
                shapes.append([height, width, 3])
    if not shapes:
        raise ValueError("hailo_calibration_shape_identity_unavailable")
    return shapes


def _effective_hailo_calibration_count(
    *,
    requested: int,
    shapes: Sequence[Sequence[int]],
    storage: str,
    cap_bytes: int,
) -> int:
    """Compute the exact count sealed into both cache payload and receipt."""

    count = max(1, int(requested))
    if str(storage) == "memory":
        for raw_shape in shapes:
            count = min(
                count,
                _clamp_calib_count(
                    [int(value) for value in raw_shape],
                    int(requested),
                    cap_bytes=int(cap_bytes),
                    storage="memory",
                ),
            )
    return int(count)


def _hailo_authoritative_calibration_sample_count(
    calib_dir: Path | None,
) -> tuple[int | None, str]:
    """Return a deterministic source-sample ceiling when one is available.

    An explicitly selected dataset manifest is authoritative.  Without one,
    individual image files are also one-sample records by construction.  NPY
    and NPZ files may contain arbitrary batches, so their sample count is left
    to the post-materialisation guard instead of being guessed from filenames.
    """

    manifest_hint = str(
        os.environ.get("ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST") or ""
    ).strip()
    if manifest_hint:
        manifest_path = Path(os.path.expanduser(manifest_hint))
        if not manifest_path.is_file():
            raise ValueError(
                f"hailo_calibration_manifest_missing:{manifest_path}"
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("hailo_calibration_manifest_not_object")
        listed_items = payload.get("items")
        listed_count = (
            len(listed_items) if isinstance(listed_items, list) else None
        )
        declared_counts = [
            value for value in (
                payload.get("item_count"),
                payload.get("sample_count"),
                payload.get("count"),
            )
            if value is not None
        ]
        normalized_counts: list[int] = []
        for value in declared_counts:
            if type(value) is not int or value <= 0:
                raise ValueError(
                    "hailo_calibration_manifest_sample_count_invalid"
                )
            normalized_counts.append(int(value))
        if listed_count is not None:
            if listed_count <= 0:
                raise ValueError("hailo_calibration_manifest_items_empty")
            normalized_counts.append(int(listed_count))
        if not normalized_counts or len(set(normalized_counts)) != 1:
            raise ValueError(
                "hailo_calibration_manifest_sample_count_ambiguous"
            )
        return normalized_counts[0], "manifest"

    if calib_dir is None or not calib_dir.is_dir():
        return None, "unavailable"
    items = _iter_calib_items(calib_dir, recursive=True)
    if not items:
        return None, "unavailable"
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    if all(path.suffix.lower() in image_suffixes for path in items):
        return len(items), "image_files"
    return None, "materialization_required"


def _verify_hailo_materialized_calibration_count(
    *,
    sealed_effective_count: int,
    calib_inputs: Mapping[str, np.ndarray],
) -> int:
    """Reject a sample-count drift before optimize/compile consumes it."""

    counts: dict[str, int] = {}
    for name, array in calib_inputs.items():
        shape = getattr(array, "shape", ())
        try:
            count = int(shape[0])
        except (IndexError, TypeError, ValueError, OverflowError):
            count = 0
        counts[str(name)] = count
    distinct = set(counts.values())
    if (
        not counts
        or len(distinct) != 1
        or next(iter(distinct)) != int(sealed_effective_count)
    ):
        raise RuntimeError(
            "hailo_calibration_materialized_count_mismatch:"
            f"sealed={int(sealed_effective_count)}:observed={counts}"
        )
    return next(iter(distinct))


def _verify_hailo_calibration_count_after_translation(
    *,
    sealed_effective_count: int,
    requested: int,
    expected_shapes: Mapping[str, Sequence[int]],
    storage: str,
    cap_bytes: int,
    available_sample_count: int | None = None,
) -> int:
    """Fail closed if the translated HN changes the sealed memory footprint."""

    observed = _effective_hailo_calibration_count(
        requested=int(requested),
        shapes=list(expected_shapes.values()),
        storage=str(storage),
        cap_bytes=int(cap_bytes),
    )
    if available_sample_count is not None:
        observed = min(observed, max(1, int(available_sample_count)))
    if observed != int(sealed_effective_count):
        raise RuntimeError(
            "hailo_calibration_effective_count_changed_after_translation:"
            f"sealed={int(sealed_effective_count)}:observed={observed}"
        )
    return observed



def _infer_hailo_image_preprocess(
    *,
    model_path: Optional[Path],
    expected_shapes: Dict[str, List[int]],
    activation_part1_onnx: Optional[Path] = None,
) -> str:
    """Infer preprocessing for image calibration data.

    Defaults:
    - classification-like 224x224 RGB models -> ImageNet mean/std
    - detector-like / larger RGB models -> 0..1 norm

    Can be overridden with ONNX_SPLITPOINT_HAILO_IMAGE_PREPROCESS=norm|raw|imagenet|clip.
    """
    env = str(os.environ.get('ONNX_SPLITPOINT_HAILO_IMAGE_PREPROCESS') or os.environ.get('SPLITPOINT_HAILO_IMAGE_PREPROCESS') or '').strip().lower()
    if env in {'raw', 'norm', 'imagenet', 'clip'}:
        return env
    if activation_part1_onnx is not None:
        path_hint = activation_part1_onnx
    else:
        path_hint = model_path
    name = str(getattr(path_hint, 'stem', '') or '').lower()
    detector_markers = ('yolo', 'detr', 'detect', 'seg', 'pose', 'obb', 'scrfd', 'retina', 'ssd')
    class_markers = ('resnet', 'mobilenet', 'regnet', 'efficientnet', 'convnext', 'densenet', 'vit', 'swin', 'inception', 'vgg', 'classification', 'classifier')
    if any(m in name for m in detector_markers):
        return 'norm'
    if any(m in name for m in class_markers):
        return 'imagenet'

    for shp in (expected_shapes or {}).values():
        try:
            dims = [int(x) for x in shp]
        except Exception:
            continue
        if len(dims) == 3:
            # HWC or CHW image shapes.
            if dims[-1] in (1, 3) and dims[0] <= 384 and dims[1] <= 384:
                return 'imagenet' if dims[-1] == 3 else 'norm'
            if dims[0] in (1, 3) and dims[1] <= 384 and dims[2] <= 384:
                return 'imagenet' if dims[0] == 3 else 'norm'
    return 'norm'


def _apply_image_preprocess_for_model_input(x: np.ndarray, preprocess: str) -> np.ndarray:
    """Convert HWC image array to the model input preprocessing domain."""
    mode = str(preprocess or 'norm').strip().lower()
    y = np.asarray(x)
    if y.dtype == np.uint8 or (np.issubdtype(y.dtype, np.integer)):
        y = y.astype(np.float32)
    else:
        y = y.astype(np.float32, copy=False)
    # Convert obvious raw 0..255 floats to 0..1 for normalized modes.
    if mode in {'norm', 'imagenet', 'clip'} and y.size and float(np.nanmax(y)) > 1.5:
        y = y / 255.0
    if mode == 'raw':
        # Preserve raw 0..255 scale if present.  If the input was 0..1, upscale.
        if y.size and float(np.nanmax(y)) <= 1.5:
            y = y * 255.0
        return np.ascontiguousarray(y.astype(np.float32, copy=False))
    if mode == 'imagenet' and y.ndim == 3 and y.shape[-1] == 3:
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        y = (y - mean) / std
    elif mode == 'clip' and y.ndim == 3 and y.shape[-1] == 3:
        mean = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 3)
        y = (y - mean) / std
    # norm -> already 0..1
    return np.ascontiguousarray(y.astype(np.float32, copy=False))
def _try_build_calib_from_dir(
    *,
    calib_dir: Path,
    expected_shape: List[int],
    limit: int,
    preprocess: str = 'norm',
    storage: str | None = None,
    storage_mode: str | None = None,
    memmap_dir: Path | None = None,
    memmap_path: Path | None = None,
    preprocessing_contract: Mapping[str, Any] | None = None,
) -> Optional[np.ndarray]:
    if not calib_dir.exists():
        return None
    calib_scan = _scan_calib_dir(calib_dir, recursive=True, limit=int(limit))
    items = list(calib_scan.get('items') or [])
    if not items:
        return None

    tgt = [int(v) for v in expected_shape]
    contract_eff: Optional[Dict[str, Any]] = None
    if preprocessing_contract is not None:
        if len(tgt) != 3 or tgt[-1] not in (1, 3):
            raise ValueError(
                "Canonical image preprocessing can only feed an HWC image input; "
                f"expected_shape={tgt}"
            )
        contract_eff, _ = resolve_image_preprocessing_contract(
            task=preprocessing_contract.get("task"),
            target_hw=[tgt[0], tgt[1]],
            declared=preprocessing_contract,
        )
    mode = str(storage_mode or storage or _calibration_storage_mode()).strip().lower()
    mmap: np.memmap | None = None
    mmap_path_local: Path | None = None
    batches: List[np.ndarray] = []
    total = 0
    if mode == 'memmap':
        if memmap_path is not None:
            mmap_path_local = Path(memmap_path)
            mmap_path_local.parent.mkdir(parents=True, exist_ok=True)
            mmap_path_local.unlink(missing_ok=True)
        else:
            tmp_dir = Path(memmap_dir) if memmap_dir is not None else Path(tempfile.gettempdir())
            tmp_dir.mkdir(parents=True, exist_ok=True)
            fd, raw_path = tempfile.mkstemp(prefix='splitpoint_hailo_calib_', suffix='.mmap', dir=str(tmp_dir))
            os.close(fd)
            mmap_path_local = Path(raw_path)
        mmap = np.memmap(mmap_path_local, dtype=np.float32, mode='w+', shape=(int(limit), *tgt))

    for pth in items:
        try:
            a = _load_calib_item_any(pth)
        except Exception:
            continue
        a = np.asarray(a)
        if len(tgt) == 3 and a.ndim in (2, 3, 4):
            if contract_eff is not None:
                a, _geometry = prepare_rgb_uint8_image(a, contract_eff)
                a = a[None, ...]
            else:
                try:
                    a = _resize_hwc_image(a, tgt[0], tgt[1], target_c=tgt[2])[None, ...]
                except Exception:
                    pass
        if a.ndim == len(tgt):
            a = a[None, ...]
        if a.ndim != len(tgt) + 1:
            continue
        if len(tgt) == 3 and a.ndim == 4 and a.shape[-1] in (1, 3):
            a = np.stack([_apply_image_preprocess_for_model_input(ss, preprocess) for ss in a], axis=0)
        else:
            a = a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32, copy=False)
        sample = list(a.shape[1:])
        if sample == tgt:
            pass
        elif len(tgt) == 3 and sample == [tgt[2], tgt[0], tgt[1]]:
            a = np.transpose(a, (0, 2, 3, 1))
        elif len(tgt) == 3:
            try:
                a = np.stack([_resize_hwc_image(ss, tgt[0], tgt[1], target_c=tgt[2]) for ss in a], axis=0)
            except Exception:
                continue
        else:
            continue
        a = np.ascontiguousarray(a.astype(np.float32, copy=False))
        take = min(int(a.shape[0]), int(limit) - total)
        if take <= 0:
            break
        if mmap is not None:
            mmap[total:total + take] = a[:take]
        else:
            batches.append(a[:take])
        total += take
        if total >= int(limit):
            break

    if total <= 0:
        if mmap is not None:
            try:
                mmap._mmap.close()
            except Exception:
                pass
        if mmap_path_local is not None:
            mmap_path_local.unlink(missing_ok=True)
        return None
    if mmap is not None:
        mmap.flush()
        # Return a memmap view; caller removes its backing file after optimize.
        return mmap[:total]
    ds = np.concatenate(batches, axis=0)
    return np.ascontiguousarray(ds[: int(limit)].astype(np.float32, copy=False))



def _nname(s: str) -> str:
    s = (s or '').replace('\\', '/')
    s = s.split('/')[-1]
    s = s.split(':')[0]
    return s.lower().strip()


def _load_onnx_input_names(model_path: Path) -> List[str]:
    try:
        m = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return []
    init_names = {str(getattr(x, 'name', '') or '') for x in getattr(m.graph, 'initializer', [])}
    out: List[str] = []
    for vi in getattr(m.graph, 'input', []):
        name = str(getattr(vi, 'name', '') or '')
        if not name or name in init_names:
            continue
        out.append(name)
    return out


def _load_onnx_output_names(model_path: Path) -> List[str]:
    try:
        m = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return []
    out: List[str] = []
    for vi in getattr(m.graph, 'output', []):
        name = str(getattr(vi, 'name', '') or '')
        if not name:
            continue
        out.append(name)
    return out


def _map_part2_inputs_to_part1_outputs(
    part1_outputs: List[str],
    part2_inputs: List[str],
) -> Tuple[Dict[str, str], List[str], Dict[str, Any]]:
    mapping: Dict[str, str] = {}
    mapping_how: Dict[str, str] = {}
    p1_out_norm = {_nname(n): n for n in part1_outputs}

    for raw in part2_inputs:
        nrm = _nname(raw)
        if raw in part1_outputs:
            mapping[raw] = raw
            mapping_how[raw] = 'exact'
        elif nrm in p1_out_norm:
            mapping[raw] = p1_out_norm[nrm]
            mapping_how[raw] = 'normalized'

    if len(mapping) != len(part2_inputs) and len(part2_inputs) == len(part1_outputs):
        for idx, raw in enumerate(part2_inputs):
            if raw in mapping:
                continue
            mapping[raw] = part1_outputs[idx]
            mapping_how[raw] = 'positional_fallback'

    missing = [n for n in part2_inputs if n not in mapping]
    debug = {
        'mapping': dict(mapping),
        'mapping_how': dict(mapping_how),
        'part1_outputs': list(part1_outputs),
        'part2_inputs': list(part2_inputs),
        'missing_inputs': list(missing),
    }
    return mapping, missing, debug


def hailo_part2_activation_precheck_from_io(
    *,
    part1_outputs: List[str],
    part2_inputs: List[str],
    original_inputs: Optional[List[str]] = None,
    part1_inputs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Classify whether Part2 activation-calibration can be driven from Part1.

    This is the compatibility rule used by the Hailo Part2 activation-calib
    path: every Part2 external input must be producible by Part1 outputs.
    Missing inputs that also belong to the original full-model inputs are
    surfaced as ``likely_original_inputs`` for clearer diagnostics.
    """

    p1_out = [str(x) for x in (part1_outputs or []) if str(x)]
    p2_in = [str(x) for x in (part2_inputs or []) if str(x)]
    p1_inputs_eff = [str(x) for x in (part1_inputs or original_inputs or []) if str(x)]

    mapping, missing, debug = _map_part2_inputs_to_part1_outputs(p1_out, p2_in)
    p1_input_norm = {_nname(n) for n in p1_inputs_eff}
    likely_original_inputs = [n for n in missing if _nname(n) in p1_input_norm]

    info: Dict[str, Any] = dict(debug)
    info.update(
        {
            'inspect_ok': True,
            'compatible': not bool(missing),
            'part1_inputs': list(p1_inputs_eff),
            'likely_original_inputs': list(likely_original_inputs),
        }
    )
    return info


def hailo_part2_activation_precheck_from_manifest(split_manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Run the cheap Part2 activation-calibration precheck from split metadata.

    The benchmark exporter already has ``split_manifest`` in memory, so it does
    not need to re-open ONNX files just to detect the classic
    ``missing=['images']`` cases.
    """

    manifest = dict(split_manifest or {})
    part1_outputs = manifest.get('cut_tensors_full') or manifest.get('part1_cut_names') or []
    part2_inputs = manifest.get('part2_external_inputs') or []
    original_inputs = manifest.get('orig_inputs') or []

    info = hailo_part2_activation_precheck_from_io(
        part1_outputs=list(part1_outputs) if isinstance(part1_outputs, list) else [],
        part2_inputs=list(part2_inputs) if isinstance(part2_inputs, list) else [],
        original_inputs=list(original_inputs) if isinstance(original_inputs, list) else [],
    )
    info['source'] = 'split_manifest'
    return info


def format_hailo_part2_activation_precheck_error(info: Dict[str, Any]) -> str:
    return _format_activation_calib_preflight_error(info)


_HAILO_PART2_HARD_PARSER_BLOCKER_OPS = {"TopK", "GatherElements"}
_HAILO_PART2_SOFT_PARSER_BLOCKER_OPS = {"ReduceMax"}


def _path_parent_prefix(name: str) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    if "/" not in s.strip("/"):
        return s.rsplit("/", 1)[0] if "/" in s else s
    parent = s.rsplit("/", 1)[0]
    return parent or s


def _dominant_parser_blocked_prefix(names: List[str]) -> Optional[str]:
    counts: Dict[str, int] = {}
    best_prefix: Optional[str] = None
    best_count = -1
    for raw in names or []:
        prefix = _path_parent_prefix(str(raw))
        if not prefix:
            continue
        counts[prefix] = int(counts.get(prefix, 0)) + 1
        if counts[prefix] > best_count or (counts[prefix] == best_count and best_prefix is not None and len(prefix) > len(best_prefix)):
            best_prefix = prefix
            best_count = counts[prefix]
    return best_prefix


def _suggest_yolo_one2one_raw_head_end_nodes(nodes: List[Any]) -> List[str]:
    """Return YOLO26 one2one raw-head Conv endpoints when the full set exists.

    Hailo DFC reports YOLO26-like heads as a YOLOv6-equivalent NMS structure and
    recommends the six ``one2one_cv2/cv3`` Conv nodes as accelerator endpoints.
    The generic parser-blocker precheck should prefer those raw endpoints over a
    late Transpose/Concat suggestion, because the latter still leaves layout-heavy
    post-processing inside the Hailo prefix.
    """
    groups: Dict[str, Dict[int, Dict[int, Tuple[int, str, int]]]] = {}
    rx = re.compile(
        r"^(?P<prefix>(?:.*?/)?model(?:/model)?\.\d+)/one2one_cv(?P<branch>[23])\.(?P<scale>[0-2])/(?P<body>.+)/Conv$"
    )
    for idx, node in enumerate(nodes or []):
        name = str(getattr(node, "name", "") or "").strip()
        if not name:
            continue
        m = rx.match(name)
        if not m:
            continue
        try:
            branch = int(m.group("branch"))
            scale = int(m.group("scale"))
        except Exception:
            continue
        body = str(m.group("body") or "")
        suffix = [int(x) for x in re.findall(rf"one2one_cv{branch}\.{scale}\.(\d+)(?:/|$)", body)]
        score = max(suffix) if suffix else -1
        prefix = str(m.group("prefix") or "").strip()
        existing = groups.setdefault(prefix, {}).setdefault(scale, {}).get(branch)
        if existing is None or score > existing[0] or (score == existing[0] and idx > existing[2]):
            groups[prefix].setdefault(scale, {})[branch] = (int(score), name, int(idx))

    best_prefix = None
    best_count = 0
    for prefix, by_scale in groups.items():
        count = sum(1 for scale in (0, 1, 2) for branch in (2, 3) if branch in by_scale.get(scale, {}))
        if count > best_count or (count == best_count and best_prefix is not None and prefix > best_prefix):
            best_prefix = prefix
            best_count = count
    if not best_prefix or best_count < 6:
        return []

    by_scale = groups[best_prefix]
    items: List[Tuple[int, str]] = []
    for scale in (0, 1, 2):
        for branch in (2, 3):
            item = by_scale.get(scale, {}).get(branch)
            if item is None:
                return []
            items.append((int(item[2]), item[1]))
    return [name for _, name in sorted(items, key=lambda x: x[0])]


def _suggest_yolo_decode_tail_end_nodes(nodes: List[Any], first_blocked_idx: Optional[int], blocked_names: List[str]) -> List[str]:
    """Suggest safe Hailo parser end nodes before a YOLO DFL/decode tail.

    Hailo DFC commonly reports YOLO11/YOLO10 Part2 failures around
    ``/model.*/dfl/Reshape`` and suggests ending the accelerator subgraph at
    tensors such as ``/model.23/Sigmoid`` and ``/model.23/Concat``.  This
    helper reproduces that suggestion statically so benchmark generation can
    materialize a Hailo-prefix + host-tail deployment instead of spending a
    full compile attempt on the unsupported decoded tail.
    """
    if first_blocked_idx is None:
        return []
    blocked_blob = " ".join(str(x or "") for x in blocked_names).lower()
    if "dfl" not in blocked_blob and "decode" not in blocked_blob:
        return []

    head_prefix = ""
    for raw in blocked_names:
        s = str(raw or "").strip()
        low = s.lower()
        marker = "/dfl/"
        pos = low.find(marker)
        if pos >= 0:
            head_prefix = s[:pos].rstrip("/")
            break
    if not head_prefix:
        # Fallback to the parent of the first blocked node; this is less exact
        # but still keeps us near the detection head.
        head_prefix = _path_parent_prefix(blocked_names[0]) if blocked_names else ""
    if not head_prefix:
        return []

    preferred: Dict[str, str] = {}
    fallback: List[str] = []
    for idx, node in enumerate(nodes or []):
        if idx >= int(first_blocked_idx):
            break
        name = str(getattr(node, "name", "") or "").strip()
        op_type = str(getattr(node, "op_type", "") or "").strip()
        if not name or not name.startswith(head_prefix.rstrip("/") + "/"):
            continue
        low_name = name.lower()
        if "/dfl/" in low_name:
            continue
        if op_type in {"Sigmoid", "Concat"}:
            preferred[op_type] = name
        elif op_type in {"Conv", "Transpose", "Reshape"}:
            fallback.append(name)

    out: List[str] = []
    # Preserve the order usually expected by Hailo's own suggestion: class/object
    # activation branch plus box-distribution branch.
    for op_type in ("Sigmoid", "Concat"):
        name = preferred.get(op_type)
        if name and name not in out:
            out.append(name)
    if out:
        return out

    # Last-resort: take a small tail-near set, newest first, then restore graph order.
    tail = []
    seen: set[str] = set()
    for name in reversed(fallback):
        if name in seen:
            continue
        seen.add(name)
        tail.append(name)
        if len(tail) >= 6:
            break
    return list(reversed(tail))


def hailo_part2_parser_blocker_precheck_from_model(
    part2_model: onnx.ModelProto,
    *,
    split_manifest: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cheap static scan for known Hailo Part2 parser blockers.

    This intentionally does *not* invoke the Hailo parser. It scans the split's
    Part2 ONNX graph for operator types that are known to fail translation with
    the currently supported Hailo DFC stack (for example ``TopK`` and
    ``GatherElements`` in YOLO-style post-processing heads).

    The precheck is conservative: only hard blockers make the split incompatible.
    Soft blockers are reported for diagnostics but do not reject on their own.
    """

    info: Dict[str, Any] = {
        'inspect_ok': False,
        'compatible': None,
        'source': 'part2_model_scan',
    }
    try:
        g = getattr(part2_model, 'graph', None)
        nodes = list(getattr(g, 'node', []) or [])
        hard_blockers: List[Dict[str, Any]] = []
        soft_blockers: List[Dict[str, Any]] = []
        transpose_candidates: List[Tuple[int, str]] = []
        for idx, node in enumerate(nodes):
            name = str(getattr(node, 'name', '') or '')
            op_type = str(getattr(node, 'op_type', '') or '')
            rec = {'name': name, 'op_type': op_type, 'index': int(idx)}
            name_l = name.lower()
            yolo_dfl_shuffle = bool(
                op_type in {'Reshape', 'Transpose'}
                and ('/dfl/' in name_l or name_l.endswith('/dfl') or 'dfl/' in name_l)
            )
            if op_type in _HAILO_PART2_HARD_PARSER_BLOCKER_OPS or yolo_dfl_shuffle:
                hard_blockers.append(rec)
            elif op_type in _HAILO_PART2_SOFT_PARSER_BLOCKER_OPS:
                soft_blockers.append(rec)
            if op_type == 'Transpose' and name:
                transpose_candidates.append((int(idx), name))

        blocked_names = [str(rec.get('name') or '') for rec in hard_blockers if str(rec.get('name') or '')]
        dominant_prefix = _dominant_parser_blocked_prefix(blocked_names)
        first_blocked_idx = min((int(rec.get('index', 0)) for rec in hard_blockers), default=None)
        suggested_end_nodes: List[str] = []

        # YOLO26/end-to-end exports often have Hailo-friendly one2one raw heads
        # before the final Transpose/Concat/NMS-style tail.  Prefer those six
        # Conv endpoints over a late Transpose suggestion; otherwise the actual
        # compiler can still fail during allocation on concat/format-conversion
        # layers even though the static parser-blocker scan looked acceptable.
        one2one_head_suggestion = _suggest_yolo_one2one_raw_head_end_nodes(nodes)
        yolo_tail_suggestion = _suggest_yolo_decode_tail_end_nodes(nodes, first_blocked_idx, blocked_names)
        if one2one_head_suggestion:
            suggested_end_nodes = list(one2one_head_suggestion)
        elif yolo_tail_suggestion:
            suggested_end_nodes = list(yolo_tail_suggestion)
        elif dominant_prefix and first_blocked_idx is not None:
            prefix_eff = dominant_prefix.rstrip('/') + '/'
            candidates = [
                name for idx, name in transpose_candidates
                if idx < int(first_blocked_idx) and str(name).startswith(prefix_eff)
            ]
            if candidates:
                # Prefer the latest transpose nodes before the blocked tail.
                ordered = list(reversed(candidates))
                dedup: List[str] = []
                seen: set[str] = set()
                for cand in ordered:
                    if cand not in seen:
                        dedup.append(cand)
                        seen.add(cand)
                suggested_end_nodes = dedup[:4]

        manifest = dict(split_manifest or {}) if isinstance(split_manifest, dict) else {}
        info.update({
            'inspect_ok': True,
            'compatible': not bool(hard_blockers),
            'hard_blocker_count': int(len(hard_blockers)),
            'soft_blocker_count': int(len(soft_blockers)),
            'blocked_ops': sorted({str(rec.get('op_type') or '') for rec in hard_blockers if str(rec.get('op_type') or '')}),
            'blocked_nodes': blocked_names,
            'soft_blocker_ops': sorted({str(rec.get('op_type') or '') for rec in soft_blockers if str(rec.get('op_type') or '')}),
            'soft_blocker_nodes': [str(rec.get('name') or '') for rec in soft_blockers if str(rec.get('name') or '')],
            'blocked_prefix': dominant_prefix,
            'suggested_end_nodes': list(suggested_end_nodes),
            'part2_inputs': list(manifest.get('part2_external_inputs') or []),
            'part1_outputs': list(manifest.get('cut_tensors_full') or manifest.get('part1_cut_names') or []),
        })
        return info
    except Exception as exc:
        info['error'] = f'{type(exc).__name__}: {exc}'
        return info


def format_hailo_part2_parser_blocker_error(info: Dict[str, Any]) -> str:
    blocked_ops = list(info.get('blocked_ops') or [])
    blocked_nodes = list(info.get('blocked_nodes') or [])
    blocked_prefix = str(info.get('blocked_prefix') or '').strip()
    suggested_end_nodes = list(info.get('suggested_end_nodes') or [])

    msg = 'Unsupported Hailo Part2 parser-blocking head detected'
    if blocked_prefix:
        msg += f': blocked_prefix={blocked_prefix}'
    if blocked_ops:
        msg += f' blocked_ops={blocked_ops}'
    if suggested_end_nodes:
        msg += f' suggested_end_nodes={suggested_end_nodes}'
    if blocked_nodes:
        preview = blocked_nodes[:6]
        msg += f' blocked_nodes={preview}'
        if len(blocked_nodes) > len(preview):
            msg += f' (+{len(blocked_nodes) - len(preview)} more)'
    return msg


def _activation_calib_preflight(
    *,
    part1_onnx: Path,
    part2_onnx: Path,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        'inspect_ok': False,
        'compatible': None,
        'part1_onnx': str(part1_onnx),
        'part2_onnx': str(part2_onnx),
    }

    try:
        part1_inputs = _load_onnx_input_names(part1_onnx)
        part1_outputs = _load_onnx_output_names(part1_onnx)
        part2_inputs = _load_onnx_input_names(part2_onnx)
    except Exception as exc:
        info['error'] = f'{type(exc).__name__}: {exc}'
        return info

    if not part1_outputs or not part2_inputs:
        info.update(
            {
                'inspect_ok': True,
                'compatible': True,
                'part1_inputs': list(part1_inputs),
                'part1_outputs': list(part1_outputs),
                'part2_inputs': list(part2_inputs),
            }
        )
        return info

    info.update(
        hailo_part2_activation_precheck_from_io(
            part1_outputs=part1_outputs,
            part2_inputs=part2_inputs,
            part1_inputs=part1_inputs,
        )
    )
    return info


def _format_activation_calib_preflight_error(info: Dict[str, Any]) -> str:
    missing = list(info.get('missing_inputs') or [])
    likely_original_inputs = list(info.get('likely_original_inputs') or [])
    part1_outputs = list(info.get('part1_outputs') or [])
    part2_inputs = list(info.get('part2_inputs') or [])
    msg = (
        'Unsupported Hailo Part2 activation-calibration splitpoint: '
        'part2 inputs are not fully produced by part1 outputs. '
        f'missing={missing}'
    )
    if likely_original_inputs:
        msg += f' likely_original_inputs={likely_original_inputs}'
    msg += f' part1_outputs={part1_outputs} part2_inputs={part2_inputs}'
    return msg


def _detect_part1_layout_ort(inp: Any) -> str:
    shp = list(getattr(inp, 'shape', []) or [])
    if len(shp) == 4:
        if isinstance(shp[1], int) and shp[1] == 3:
            return 'NCHW'
        if isinstance(shp[3], int) and shp[3] == 3:
            return 'NHWC'
    return 'NCHW'


def _ort_input_hwc_spec(inp: Any) -> Tuple[Optional[int], Optional[int], Optional[int], str]:
    shp = list(getattr(inp, 'shape', []) or [])
    layout = _detect_part1_layout_ort(inp)

    def _to_dim(v: Any) -> Optional[int]:
        try:
            iv = int(v)
        except Exception:
            return None
        return iv if iv > 0 else None

    dims = [_to_dim(v) for v in shp]
    if len(dims) == 4:
        if layout == 'NCHW':
            return dims[2], dims[3], dims[1], layout
        return dims[1], dims[2], dims[3], layout
    if len(dims) == 3:
        if layout == 'NCHW':
            return dims[1], dims[2], dims[0], layout
        return dims[0], dims[1], dims[2], layout
    return None, None, None, layout


def _prepare_part1_input_for_activation_calib(
    arr: np.ndarray,
    inp: Any,
    preprocess: str = 'norm',
    preprocessing_contract: Mapping[str, Any] | None = None,
) -> np.ndarray:
    tgt_h, tgt_w, tgt_c, layout = _ort_input_hwc_spec(inp)
    if tgt_h is None or tgt_w is None:
        raise ValueError("Part1 image input requires static height and width")
    if preprocessing_contract is not None:
        if tgt_c not in (3, None):
            raise ValueError(
                "Canonical RGB preprocessing cannot feed a non-RGB Part1 input; "
                f"channels={tgt_c}"
            )
        contract_eff, _ = resolve_image_preprocessing_contract(
            task=preprocessing_contract.get("task"),
            target_hw=[tgt_h, tgt_w],
            declared=preprocessing_contract,
        )
        x, _geometry = prepare_rgb_uint8_image(arr, contract_eff)
    else:
        x = _resize_hwc_image(arr, tgt_h, tgt_w, target_c=tgt_c)

    exp_type = str(getattr(inp, 'type', '') or '').lower()
    if 'float' in exp_type:
        x = _apply_image_preprocess_for_model_input(x, preprocess)
    else:
        if np.issubdtype(x.dtype, np.floating):
            if x.size and float(np.nanmax(x)) <= 1.5:
                x = x * 255.0
            x = np.clip(x, 0.0, 255.0)
        x = x.astype(np.uint8)

    if layout == 'NCHW':
        x = np.transpose(x, (2, 0, 1))[None, ...]
    else:
        x = x[None, ...]
    return np.ascontiguousarray(x)


def _convert_calib_dataset_to_hn_shape(ds: np.ndarray, hn_shape: List[int]) -> Tuple[np.ndarray, str]:
    x = np.asarray(ds)
    if x.ndim < 2:
        raise ValueError(f'Bad calib dataset rank: {x.ndim}')
    while x.ndim >= 3 and x.shape[1] == 1 and (x.ndim - 1) > len(hn_shape):
        x = np.squeeze(x, axis=1)
    sample_shape = list(x.shape[1:])
    target = [int(v) for v in hn_shape]
    if sample_shape == target:
        return np.ascontiguousarray(x.astype(np.float32, copy=False)), 'ok (already matches)'
    if len(target) == 3 and x.ndim == 4:
        if sample_shape == [target[2], target[0], target[1]]:
            y = np.transpose(x, (0, 2, 3, 1))
            return np.ascontiguousarray(y.astype(np.float32, copy=False)), 'transpose NCHW->NHWC'
        if sample_shape == [target[0], target[1], target[2]]:
            return np.ascontiguousarray(x.astype(np.float32, copy=False)), 'ok (NHWC)'
    raise ValueError(f'Cannot convert calib dataset sample_shape={sample_shape} to hn_shape={target}')



def _activation_proxy_strict_enabled() -> bool:
    """Return whether activation proxy fallback is forbidden.

    This is intentionally checked in the backend code, not only in the GUI, so
    CLI runs and background GUI jobs behave the same way.
    """
    raw = str(
        os.environ.get('ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT')
        or os.environ.get('SPLITPOINT_ACTIVATION_PROXY_STRICT')
        or os.environ.get('ONNX_SPLITPOINT_ACTIVATION_PROXY_NO_FALLBACK')
        or ''
    ).strip().lower()
    return raw in {'1', 'true', 'yes', 'on', 'strict', 'fail', 'no_fallback'}


def _activation_proxy_backend_request() -> str:
    """Return requested activation proxy producer backend.

    The default prefers CUDA ORT when available and falls back to ORT CPU.  This is a pragmatic proxy for Stage2 accelerator calibration on a CUDA build workstation.  For faster/better proxy
    calibration on a build workstation, set::

        ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND=cuda_ort
        ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND=tensorrt_ort

    The selected backend is still a *proxy* for Stage2 calibration unless it is
    generated by the actual runtime producer.
    """
    raw = str(os.environ.get('ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND') or os.environ.get('SPLITPOINT_ACTIVATION_PROXY_BACKEND') or 'cuda_ort').strip().lower().replace('-', '_')
    aliases = {
        '': 'ort_cpu',
        'auto': 'cuda_ort',
        'cpu': 'ort_cpu',
        'ort': 'ort_cpu',
        'ort_cpu': 'ort_cpu',
        'cpu_ort': 'ort_cpu',
        'cuda': 'cuda_ort',
        'gpu': 'cuda_ort',
        'ort_cuda': 'cuda_ort',
        'cuda_ort': 'cuda_ort',
        'trt': 'tensorrt_ort',
        'tensorrt': 'tensorrt_ort',
        'ort_tensorrt': 'tensorrt_ort',
        'tensorrt_ort': 'tensorrt_ort',
        'trt_ort': 'tensorrt_ort',
        'remote': 'remote_deepx_tensorrt',
        'remote_deepx': 'remote_deepx_tensorrt',
        'remote_trt': 'remote_deepx_tensorrt',
        'remote_tensorrt': 'remote_deepx_tensorrt',
        'remote_deepx_trt': 'remote_deepx_tensorrt',
        'remote_deepx_tensorrt': 'remote_deepx_tensorrt',
        'remote_cuda': 'remote_deepx_cuda',
        'remote_deepx_cuda': 'remote_deepx_cuda',
    }
    return aliases.get(raw, 'ort_cpu')





def _activation_proxy_is_remote_backend(backend: str) -> bool:
    return str(backend or '').strip().lower().replace('-', '_').startswith('remote_')


def _load_activation_proxy_deepx_remote_setup() -> Dict[str, Any]:
    """Resolve the DeepX NX runtime setup from ~/.onnx_splitpoint_tool/hardware_setups.yaml."""
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyYAML is required to read hardware_setups.yaml: {type(exc).__name__}: {exc}") from exc
    try:
        from .workflow.hardware_matrix import ensure_hardware_setups_file, canon_accelerator  # type: ignore
    except Exception:
        from onnx_splitpoint_tool.workflow.hardware_matrix import ensure_hardware_setups_file, canon_accelerator  # type: ignore
    p = ensure_hardware_setups_file()
    data = yaml.safe_load(Path(p).read_text(encoding='utf-8')) or {}
    setups = list(data.get('hardware_setups') or []) if isinstance(data, dict) else []
    candidates: List[Dict[str, Any]] = []
    for raw in setups:
        if not isinstance(raw, dict):
            continue
        acc = canon_accelerator(raw.get('accelerator') or raw.get('backend') or raw.get('target') or raw.get('id'))
        if acc == 'deepx_m1' or 'deepx' in str(raw.get('id') or '').lower():
            candidates.append(dict(raw))
    if not candidates:
        raise RuntimeError(f"No DeepX hardware setup found in {p}")
    selected = None
    for c in candidates:
        h = c.get('host') if isinstance(c.get('host'), dict) else {}
        addr = str(h.get('address') or h.get('host') or c.get('address') or c.get('host') or '').strip()
        if addr:
            selected = c
            break
    selected = selected or candidates[0]
    host_map = selected.get('host') if isinstance(selected.get('host'), dict) else {}
    runtime = selected.get('runtime') if isinstance(selected.get('runtime'), dict) else {}
    remote = selected.get('remote') if isinstance(selected.get('remote'), dict) else {}
    host = str(host_map.get('address') or host_map.get('host') or remote.get('host') or runtime.get('host') or selected.get('host') or '').strip()
    user = str(host_map.get('user') or remote.get('user') or runtime.get('user') or selected.get('user') or 'nx').strip() or 'nx'
    port = int(host_map.get('port') or remote.get('port') or runtime.get('port') or selected.get('port') or 22)
    base_dir = str(host_map.get('base_dir') or remote.get('remote_base_dir') or runtime.get('remote_base_dir') or selected.get('remote_base_dir') or '~/splitpoint_runs')
    activate = str(runtime.get('activate') or runtime.get('venv_activate') or runtime.get('venv') or remote.get('remote_venv') or selected.get('remote_venv') or 'source ~/venvs/deepx-runtime/bin/activate')
    provider = str(runtime.get('provider') or remote.get('provider') or selected.get('provider') or 'deepx_m1')
    if not host:
        raise RuntimeError(f"DeepX hardware setup {selected.get('id') or '<unknown>'} has no host configured in {p}")
    return {'id': str(selected.get('id') or 'orin_nx_deepx_m1_01'), 'label': str(selected.get('label') or selected.get('name') or 'Orin NX + DeepX DX-M1'), 'host': host, 'user': user, 'port': port, 'base_dir': base_dir, 'activate': activate, 'provider': provider, 'registry': str(p)}


_REMOTE_ACTIVATION_PROXY_SCRIPT = '#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse, json, re, tarfile\nfrom pathlib import Path\nimport numpy as np\n\ndef _safe_key(name: str) -> str:\n    s = re.sub(r\'[^A-Za-z0-9_]+\', \'_\', str(name)).strip(\'_\')\n    return s or \'tensor\'\n\ndef _load_item(path: Path):\n    suf = path.suffix.lower()\n    if suf == \'.npy\':\n        return np.asarray(np.load(str(path)))\n    if suf == \'.npz\':\n        data = np.load(str(path)); return np.asarray(data[list(data.files)[0]])\n    import cv2\n    im = cv2.imread(str(path), cv2.IMREAD_COLOR)\n    if im is None: raise RuntimeError(f\'cv2.imread failed for {path}\')\n    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)\n\ndef _onnx_io_meta(path: Path):\n    import onnx\n    m = onnx.load(str(path))\n    def dims(v):\n        out=[]\n        for d in v.type.tensor_type.shape.dim:\n            if getattr(d,\'dim_value\',0): out.append(int(d.dim_value))\n            elif getattr(d,\'dim_param\',\'\'): out.append(str(d.dim_param))\n            else: out.append(None)\n        return out\n    def typ(v):\n        try:\n            code=int(v.type.tensor_type.elem_type)\n            try:\n                import onnx\n                return str(onnx.TensorProto.DataType.Name(code))\n            except Exception:\n                return str(code)\n        except Exception: return \'\'\n    return {\'inputs\':[{\'name\':i.name,\'shape\':dims(i),\'type\':typ(i)} for i in m.graph.input], \'outputs\':[{\'name\':o.name,\'shape\':dims(o),\'type\':typ(o)} for o in m.graph.output]}\n\ndef _is_float_type(t):\n    s=str(t or \'\').strip().lower()\n    return s in {\'1\',\'float\',\'float32\',\'tensor_float\',\'tensor(float)\',\'float16\',\'tensor(float16)\',\'float64\',\'double\',\'tensor(double)\'} or \'float\' in s or \'double\' in s\n\ndef _hwc_spec(meta):\n    shape=list(meta.get(\'shape\') or [])\n    if len(shape)==4:\n        if shape[1] in (1,3): return int(shape[2]), int(shape[3]), int(shape[1]), \'NCHW\'\n        return int(shape[1]), int(shape[2]), int(shape[3]), \'NHWC\'\n    if len(shape)==3:\n        if shape[0] in (1,3): return int(shape[1]), int(shape[2]), int(shape[0]), \'NCHW\'\n        return int(shape[0]), int(shape[1]), int(shape[2]), \'NHWC\'\n    return 224,224,3,\'NHWC\'\n\ndef _prepare_input(arr, inp_meta, preprocess):\n    import cv2\n    h,w,c,layout=_hwc_spec(inp_meta)\n    x=np.asarray(arr)\n    if x.ndim==2: x=np.stack([x,x,x], axis=-1)\n    if x.ndim==4 and x.shape[0]==1: x=x[0]\n    if x.ndim==3 and x.shape[-1] not in (1,3) and x.shape[0] in (1,3): x=np.transpose(x,(1,2,0))\n    if x.ndim != 3: raise RuntimeError(f\'bad calibration input rank {x.ndim}\')\n    if c==1 and x.shape[-1]!=1: x=cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY)[...,None]\n    elif c==3 and x.shape[-1]==1: x=np.repeat(x,3,axis=-1)\n    x=cv2.resize(x,(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32)\n    mode=str(preprocess or \'norm\').lower()\n    exp_type=str(inp_meta.get(\'type\') or \'\').lower()\n    if _is_float_type(exp_type):\n        if mode in {\'norm\',\'imagenet\',\'clip\'} and x.size and float(np.nanmax(x))>1.5: x=x/255.0\n        if mode==\'imagenet\' and x.shape[-1]==3:\n            mean=np.asarray([0.485,0.456,0.406],np.float32).reshape(1,1,3); std=np.asarray([0.229,0.224,0.225],np.float32).reshape(1,1,3); x=(x-mean)/std\n        elif mode==\'clip\' and x.shape[-1]==3:\n            mean=np.asarray([0.48145466,0.4578275,0.40821073],np.float32).reshape(1,1,3); std=np.asarray([0.26862954,0.26130258,0.27577711],np.float32).reshape(1,1,3); x=(x-mean)/std\n        elif mode==\'raw\' and x.size and float(np.nanmax(x))<=1.5: x=x*255.0\n    else:\n        if x.size and float(np.nanmax(x))<=1.5: x=x*255.0\n        x=np.clip(x,0,255).astype(np.uint8)\n    if layout==\'NCHW\': x=np.transpose(x,(2,0,1))[None,...]\n    else: x=x[None,...]\n    return np.ascontiguousarray(x)\n\ndef main():\n    ap=argparse.ArgumentParser(); ap.add_argument(\'--meta\', required=True); ns=ap.parse_args()\n    meta=json.loads(Path(ns.meta).read_text())\n    root=Path(meta.get(\'work_dir\') or \'.\').resolve()\n    part1=root/\'part1.onnx\'; part2=root/\'part2.onnx\'; images=root/\'images\'\n    provider_mode=str(meta.get(\'provider_mode\') or \'tensorrt\').lower()\n    preprocess=str(meta.get(\'input_preprocess\') or \'norm\')\n    limit=int(meta.get(\'limit\') or 100)\n    import onnxruntime as ort\n    try:\n        if hasattr(ort,\'preload_dlls\'): ort.preload_dlls(directory=\'\')\n    except Exception: pass\n    p1m=_onnx_io_meta(part1); p2m=_onnx_io_meta(part2)\n    p1_in=p1m[\'inputs\'][0]\n    p1_out_names=[o[\'name\'] for o in p1m[\'outputs\']]\n    p2_in_names=[i[\'name\'] for i in p2m[\'inputs\']]\n    missing=[n for n in p2_in_names if n not in set(p1_out_names)]\n    if missing: raise RuntimeError(\'Part2 inputs are not Part1 outputs: \'+\', \'.join(missing))\n    providers=[\'CPUExecutionProvider\']\n    if provider_mode in {\'tensorrt\',\'trt\',\'remote_deepx_tensorrt\'}: providers=[\'TensorrtExecutionProvider\',\'CUDAExecutionProvider\',\'CPUExecutionProvider\']\n    elif provider_mode in {\'cuda\',\'remote_deepx_cuda\'}: providers=[\'CUDAExecutionProvider\',\'CPUExecutionProvider\']\n    so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1; so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL\n    sess=ort.InferenceSession(str(part1), sess_options=so, providers=providers)\n    used=list(sess.get_providers() or [])\n    primary=\'TensorrtExecutionProvider\' if \'TensorrtExecutionProvider\' in providers else (\'CUDAExecutionProvider\' if \'CUDAExecutionProvider\' in providers else \'CPUExecutionProvider\')\n    if bool(meta.get(\'strict\')) and primary not in used: raise RuntimeError(f\'remote strict proxy requested {primary}, but session providers are {used}\')\n    exts={\'.jpg\',\'.jpeg\',\'.png\',\'.bmp\',\'.npy\',\'.npz\'}\n    items=[p for p in sorted(images.rglob(\'*\')) if p.is_file() and p.suffix.lower() in exts][:limit]\n    if not items: raise RuntimeError(\'no calibration items in remote images directory\')\n    per={n: [] for n in p2_in_names}\n    for p in items:\n        x=_prepare_input(_load_item(p), p1_in, preprocess)\n        outs=sess.run(p2_in_names, {p1_in[\'name\']: x})\n        for n,o in zip(p2_in_names, outs):\n            arr=np.asarray(o)\n            if arr.ndim>=1 and arr.shape[0]==x.shape[0]: per[n].append(np.asarray(arr[0], dtype=np.float32))\n            else: per[n].append(np.asarray(arr, dtype=np.float32))\n    arrays={n:np.stack(v,axis=0).astype(np.float32) for n,v in per.items()}\n    key_map={}; payload={}\n    for n,a in arrays.items():\n        k=_safe_key(n)\n        while k in payload: k=k+\'_x\'\n        key_map[k]=n; payload[k]=a\n    np.savez_compressed(str(root/\'arrays.npz\'), **payload)\n    (root/\'key_map.json\').write_text(json.dumps(key_map, indent=2), encoding=\'utf-8\')\n    producer=\'remote_deepx_cpu\'\n    if \'TensorrtExecutionProvider\' in used: producer=\'remote_deepx_tensorrt\'\n    elif \'CUDAExecutionProvider\' in used: producer=\'remote_deepx_cuda\'\n    debug={\'remote_activation_proxy\': True, \'remote_host\': meta.get(\'remote_host\'), \'remote_setup_id\': meta.get(\'remote_setup_id\'), \'activation_proxy_requested_backend\': meta.get(\'requested_backend\'), \'activation_proxy_producer_backend\': producer, \'activation_proxy_source\': producer + \'_reference_proxy\', \'activation_proxy_provider_fallback\': \'\' if producer==meta.get(\'requested_backend\') else f\'remote requested {meta.get("requested_backend")}, session providers={used}\', \'remote_session_providers\': used, \'calib_items_total\': len(items), \'calib_source_dir\': str(images), \'input_preprocess\': preprocess, \'part1_input\': p1_in, \'part2_inputs\': p2m[\'inputs\'], \'calib_shapes_before_hn\': {k:{\'dataset_shape\':list(v.shape),\'dtype\':str(v.dtype)} for k,v in arrays.items()}}\n    (root/\'debug.json\').write_text(json.dumps(debug, indent=2), encoding=\'utf-8\')\n    with tarfile.open(root/\'output.tgz\',\'w:gz\') as tf:\n        for name in [\'arrays.npz\',\'key_map.json\',\'debug.json\']: tf.add(root/name, arcname=name)\n    return 0\nif __name__==\'__main__\': raise SystemExit(main())\n'

# Canonical inputs are materialized as RGB uint8 .npy files before transfer.
# A same-size OpenCV resize would normally be a copy, but skipping it makes the
# byte-preservation guarantee explicit and independent of OpenCV versions.
_REMOTE_ACTIVATION_PROXY_SCRIPT = _REMOTE_ACTIVATION_PROXY_SCRIPT.replace(
    "    x=cv2.resize(x,(w,h),interpolation=cv2.INTER_LINEAR).astype(np.float32)\n",
    "    if tuple(x.shape[:2]) != (h,w): x=cv2.resize(x,(w,h),interpolation=cv2.INTER_LINEAR)\n"
    "    x=x.astype(np.float32)\n",
)


def _ssh_run_for_activation_proxy(setup: Dict[str, Any], command: str, *, timeout_s: int = 3600) -> tuple[int, str, str]:
    host = f"{setup['user']}@{setup['host']}"
    # Send a single quoted remote command.  Passing ['bash','-lc', command]
    # directly to ssh can lose quoting because ssh joins remote argv with
    # spaces before the remote shell sees it.
    remote_cmd = 'bash -lc ' + shlex.quote(str(command))
    ssh_cmd = ['ssh', '-p', str(setup.get('port') or 22), host, remote_cmd]
    proc = _run_owned_subprocess(ssh_cmd, text=True, capture_output=True, timeout=int(timeout_s))
    return int(proc.returncode), proc.stdout or '', proc.stderr or ''


def _scp_for_activation_proxy(setup: Dict[str, Any], src: str | Path, dst_remote: str) -> tuple[int, str, str]:
    """Copy a local file to the remote DeepX host.

    ``dst_remote`` must be an already-expanded absolute path returned by the
    remote shell.  Do not pass shlex-quoted strings here: scp does not perform a
    second shell expansion for single-quoted ``~/...`` paths, which previously
    produced paths such as ``'~/splitpoint_runs/...'`` and failed with
    "No such file or directory".
    """
    host = f"{setup['user']}@{setup['host']}"
    proc = _run_owned_subprocess(['scp', '-P', str(setup.get('port') or 22), str(src), f'{host}:{str(dst_remote)}'], text=True, capture_output=True)
    return int(proc.returncode), proc.stdout or '', proc.stderr or ''


def _scp_from_activation_proxy(setup: Dict[str, Any], src_remote: str, dst: str | Path) -> tuple[int, str, str]:
    """Copy a remote file back from the DeepX host.

    ``src_remote`` must be an expanded absolute path.  Keep it unquoted for scp.
    """
    host = f"{setup['user']}@{setup['host']}"
    proc = _run_owned_subprocess(['scp', '-P', str(setup.get('port') or 22), f'{host}:{str(src_remote)}', str(dst)], text=True, capture_output=True)
    return int(proc.returncode), proc.stdout or '', proc.stderr or ''


def _build_activation_calib_from_part1_onnx_remote_deepx(
    *,
    requested_backend: str,
    part1_onnx: Path,
    part2_onnx: Path,
    calib_dir: Path,
    limit: int,
    gen_batch: int,
    input_preprocess: str = 'norm',
    preprocessing_contract: Mapping[str, Any] | None = None,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Any]]:
    import tempfile, tarfile, uuid
    if compiler_dispatch_forbidden() and (
        "tensorrt" in str(requested_backend).lower()
        or "trt" in str(requested_backend).lower()
    ):
        raise RuntimeError(cache_miss_blocked_message(
            "tensorrt_ort_ep",
            "remote activation proxy requested TensorRTExecutionProvider",
        ))
    setup = _load_activation_proxy_deepx_remote_setup()
    calib_scan = _scan_calib_dir(calib_dir, recursive=True, limit=max(1, int(limit)))
    calib_items = list(calib_scan.get('items') or [])
    if not calib_items:
        raise FileNotFoundError(f'No calibration items (.npy/.npz/images) in: {calib_dir}')
    strict_proxy = _activation_proxy_strict_enabled()
    provider_mode = 'tensorrt' if 'tensorrt' in str(requested_backend) or 'trt' in str(requested_backend) else 'cuda'
    base = str(setup.get('base_dir') or '~/splitpoint_runs').strip()
    # Normalize stale paths that accidentally persisted as /home/<user>/~/...
    # before we hand them to the remote shell.  The remote shell expansion below
    # then returns a real absolute path for scp, e.g. /home/nx/splitpoint_runs.
    if '/~/' in base:
        base = '~/' + base.split('/~/')[-1]
    elif base.endswith('/~'):
        base = '~'
    prefix = 'splitpoint_actproxy_' + uuid.uuid4().hex[:10]
    # Resolve the remote base directory on the remote shell.  Do not pass
    # quoted '~' paths to scp: scp will not expand them and then fails with
    # "dest open '~/...': No such file or directory".  The mktemp command
    # below always returns an absolute path such as /home/nx/splitpoint_runs/...
    mkcmd = f'''set -e
raw_base={shlex.quote(base)}
case "$raw_base" in
  "") base="$HOME/splitpoint_runs" ;;
  "~") base="$HOME" ;;
  "~/"*) base="$HOME/${{raw_base#~/}}" ;;
  */~/*) base="$HOME/${{raw_base#*~/}}" ;;
  */~) base="$HOME" ;;
  /*) base="$raw_base" ;;
  *) base="$HOME/$raw_base" ;;
esac
mkdir -p "$base/activation_proxy"
mktemp -d "$base/activation_proxy/{prefix}.XXXXXX"
'''
    rc, out, err = _ssh_run_for_activation_proxy(setup, mkcmd, timeout_s=60)
    if rc != 0 or not out.strip():
        raise RuntimeError(f'remote mktemp failed rc={rc}: {err[-1000:]}')
    remote_dir = out.strip().splitlines()[-1].strip()
    with tempfile.TemporaryDirectory(prefix='splitpoint_remote_proxy_') as td:
        tdp = Path(td); payload = tdp / 'payload'; (payload / 'images').mkdir(parents=True, exist_ok=True)
        shutil.copy2(part1_onnx, payload / 'part1.onnx'); shutil.copy2(part2_onnx, payload / 'part2.onnx')
        copied = 0
        for i, item in enumerate(calib_items[:max(1, int(limit))]):
            try:
                src = Path(item)
                if preprocessing_contract is not None:
                    prepared, _geometry = prepare_rgb_uint8_image(
                        _load_calib_item_any(src), preprocessing_contract
                    )
                    np.save(payload / 'images' / f'{i:06d}.npy', prepared)
                else:
                    shutil.copy2(src, payload / 'images' / f'{i:06d}_{src.name}')
                copied += 1
            except Exception:
                if preprocessing_contract is not None:
                    raise
        meta = {'work_dir': str(remote_dir), 'limit': max(1, int(limit)), 'gen_batch': max(1, int(gen_batch)), 'input_preprocess': str(input_preprocess), 'provider_mode': provider_mode, 'strict': bool(strict_proxy), 'requested_backend': str(requested_backend), 'remote_host': f"{setup.get('user')}@{setup.get('host')}:{setup.get('port')}", 'remote_setup_id': setup.get('id'), 'preprocessing_contract': dict(preprocessing_contract or {}), 'preprocessing_contract_sha256': preprocessing_contract_sha256(preprocessing_contract) if preprocessing_contract else None, 'inputs_prepared_before_transfer': bool(preprocessing_contract), 'inputs_copied': int(copied)}
        (payload / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        (payload / 'remote_activation_proxy.py').write_text(_REMOTE_ACTIVATION_PROXY_SCRIPT, encoding='utf-8')
        archive = tdp / 'input.tgz'
        with tarfile.open(archive, 'w:gz') as tf:
            for p in payload.rglob('*'):
                tf.add(p, arcname=str(p.relative_to(payload)))
        rc, _out, err = _scp_for_activation_proxy(setup, archive, remote_dir + '/input.tgz')
        if rc != 0:
            raise RuntimeError(f'remote scp input failed rc={rc}: {err[-1000:]}')
        activate = str(setup.get('activate') or 'source ~/venvs/deepx-runtime/bin/activate')
        # Execute with the Python from the activated remote venv.  Some remote
        # shells keep a stale PATH or do not provide 'python'; use VIRTUAL_ENV
        # directly when available.
        remote_cmd = f'''
set -e
cd {shlex.quote(remote_dir)}
tar -xzf input.tgz
ACTIVATE_CMD={shlex.quote(activate)}
eval "$ACTIVATE_CMD"
export LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:${{LD_LIBRARY_PATH:-}}
PYBIN=python3
if [ -n "${{VIRTUAL_ENV:-}}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYBIN="$VIRTUAL_ENV/bin/python"
elif [ -n "${{VIRTUAL_ENV:-}}" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
  PYBIN="$VIRTUAL_ENV/bin/python3"
fi
"$PYBIN" remote_activation_proxy.py --meta meta.json
'''
        rc, out, err = _ssh_run_for_activation_proxy(setup, remote_cmd, timeout_s=7200)
        if rc != 0:
            raise RuntimeError(f'remote activation proxy failed rc={rc}; dir={remote_dir}; stdout_tail={out[-1500:]}; stderr_tail={err[-2500:]}')
        out_archive = tdp / 'output.tgz'
        rc, _out2, err2 = _scp_from_activation_proxy(setup, remote_dir + '/output.tgz', out_archive)
        if rc != 0 or not out_archive.exists():
            raise RuntimeError(f'remote scp output failed rc={rc}: {err2[-1000:]}')
        outdir = tdp / 'out'; outdir.mkdir()
        with tarfile.open(out_archive, 'r:gz') as tf:
            tf.extractall(outdir)
        key_map = json.loads((outdir / 'key_map.json').read_text(encoding='utf-8'))
        debug = json.loads((outdir / 'debug.json').read_text(encoding='utf-8'))
        npz = np.load(str(outdir / 'arrays.npz'))
        arrays: Dict[str, np.ndarray] = {str(name): np.asarray(npz[key]).astype(np.float32, copy=False) for key, name in key_map.items()}
        names = [str(v) for v in key_map.values()]
        debug.update({'activation_proxy_remote_work_dir': remote_dir, 'activation_proxy_remote_setup': setup, 'activation_proxy_remote_requested_backend': str(requested_backend), 'activation_proxy_remote_stdout_tail': out[-2000:], 'activation_proxy_calib_items_copied': int(copied), 'preprocessing_contract': dict(preprocessing_contract or {}), 'preprocessing_contract_sha256': preprocessing_contract_sha256(preprocessing_contract) if preprocessing_contract else None, 'calib_scan': {k: v for k, v in calib_scan.items() if k != 'items'}})
        return names, arrays, debug




def _preload_nvidia_cuda_wheel_libs_for_ort() -> Dict[str, Any]:
    """Best-effort CUDA/cuDNN/cuBLAS preload for ONNX Runtime provider sessions.

    This mirrors scripts/check_activation_proxy_backend.py.  It helps when the
    build-host activation proxy is cuda_ort/tensorrt_ort and the CUDA runtime
    libraries are installed via NVIDIA Python wheels rather than system ldconfig.
    """
    try:
        import ctypes
        import site
        import sys as _sys
        roots: List[Path] = []
        try:
            roots.extend(Path(x) for x in site.getsitepackages())
        except Exception:
            pass
        try:
            roots.append(Path(site.getusersitepackages()))
        except Exception:
            pass
        for x in _sys.path:
            if x and 'site-packages' in x:
                roots.append(Path(x))
        uniq: List[Path] = []
        for r in roots:
            if r and r.exists() and r not in uniq:
                uniq.append(r)
        patterns = [
            '**/libcudart.so*', '**/libnvrtc.so*', '**/libnvJitLink.so*',
            '**/libcublas.so*', '**/libcublasLt.so*', '**/libcudnn.so*',
            '**/libcufft.so*', '**/libcurand.so*', '**/libcusparse.so*', '**/libcusolver.so*',
            '**/libnvinfer.so*', '**/libnvinfer_plugin.so*', '**/libnvonnxparser.so*',
        ]
        found: List[Path] = []
        for root in uniq:
            for pat in patterns:
                try:
                    for q in root.glob(pat):
                        if q.is_file() and q not in found:
                            found.append(q)
                except Exception:
                    pass
        order = ['libcudart','libnvrtc','libnvJitLink','libcublas','libcublasLt','libcudnn','libcufft','libcurand','libcusparse','libcusolver','libnvinfer','libnvinfer_plugin','libnvonnxparser']
        def key(q: Path):
            for i,prefix in enumerate(order):
                if q.name.startswith(prefix):
                    return (i,q.name)
            return (999,q.name)
        found = sorted(found, key=key)
        dirs: List[str] = []
        for q in found:
            d=str(q.parent)
            if d not in dirs:
                dirs.append(d)
        if dirs:
            os.environ['LD_LIBRARY_PATH'] = ':'.join(dirs + [os.environ.get('LD_LIBRARY_PATH','')])
        loaded: List[str] = []
        errors: List[str] = []
        for q in found:
            try:
                ctypes.CDLL(str(q), mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
                loaded.append(str(q))
            except Exception as exc:
                errors.append(f'{q}: {type(exc).__name__}: {exc}')
        return {'attempted': True, 'found_count': len(found), 'loaded_count': len(loaded), 'loaded_tail': loaded[-20:], 'dirs': dirs, 'errors': errors[:10]}
    except Exception as exc:
        return {'attempted': True, 'error': f'{type(exc).__name__}: {exc}'}

def _activation_proxy_provider_session_smoke(ort: Any, provider: str) -> Dict[str, Any]:
    """Return whether an ORT provider can actually create a session.

    Provider listing alone is not enough on workstations: ORT can list the
    TensorRT/CUDA EPs while session creation silently falls back to CUDA/CPU
    because a shared library such as libnvinfer, libcublas, or libcudnn is not
    loadable. The activation proxy must record the provider that *actually*
    produced calibration tensors.
    """
    if (
        provider == 'TensorrtExecutionProvider'
        and compiler_dispatch_forbidden()
    ):
        return {
            'ok': False,
            'requested_provider': provider,
            'blocked': True,
            'error': cache_miss_blocked_message(
                'tensorrt_ort_ep', 'activation proxy provider smoke'
            ),
        }
    try:
        import tempfile
        import numpy as _np
        _cuda_preload_info = _preload_nvidia_cuda_wheel_libs_for_ort()
        import onnx as _onnx  # type: ignore
        from onnx import TensorProto as _TensorProto, helper as _helper  # type: ignore
    except Exception as exc:
        return {'ok': False, 'error': f'smoke prerequisites missing: {type(exc).__name__}: {exc}'}
    try:
        x = _helper.make_tensor_value_info('x', _TensorProto.FLOAT, [1, 4])
        y = _helper.make_tensor_value_info('y', _TensorProto.FLOAT, [1, 4])
        node = _helper.make_node('Relu', ['x'], ['y'])
        graph = _helper.make_graph([node], 'splitpoint_activation_proxy_provider_smoke', [x], [y])
        model = _helper.make_model(graph, producer_name='splitpoint-activation-proxy-smoke', opset_imports=[_helper.make_opsetid('', 17)])
        model.ir_version = min(getattr(model, 'ir_version', 9), 9)
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            path = f.name
        _onnx.save(model, path)
        providers = [provider]
        if provider == 'TensorrtExecutionProvider':
            providers.extend(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        elif provider == 'CUDAExecutionProvider':
            providers.append('CPUExecutionProvider')
        sess = ort.InferenceSession(path, providers=providers)
        got = list(sess.get_providers() or [])
        out = sess.run(None, {'x': _np.array([[-1.0, 0.0, 2.0, 3.0]], dtype=_np.float32)})[0]
        active = provider in set(got)
        conv_smoke = {'ok': None, 'skipped': True}
        if active and provider in {'CUDAExecutionProvider', 'TensorrtExecutionProvider'}:
            try:
                x2 = _helper.make_tensor_value_info('x', _TensorProto.FLOAT, [1, 3, 16, 16])
                w2 = _helper.make_tensor_value_info('w', _TensorProto.FLOAT, [4, 3, 3, 3])
                y2 = _helper.make_tensor_value_info('y', _TensorProto.FLOAT, [1, 4, 14, 14])
                node2 = _helper.make_node('Conv', ['x', 'w'], ['y'], pads=[0, 0, 0, 0], strides=[1, 1])
                graph2 = _helper.make_graph([node2], 'splitpoint_activation_proxy_provider_conv_smoke', [x2, w2], [y2])
                model2 = _helper.make_model(graph2, producer_name='splitpoint-activation-proxy-conv-smoke', opset_imports=[_helper.make_opsetid('', 17)])
                model2.ir_version = min(getattr(model2, 'ir_version', 9), 9)
                with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f2:
                    path2 = f2.name
                _onnx.save(model2, path2)
                sess2 = ort.InferenceSession(path2, providers=providers)
                got2 = list(sess2.get_providers() or [])
                yarr = sess2.run(None, {'x': _np.zeros((1, 3, 16, 16), dtype=_np.float32), 'w': _np.ones((4, 3, 3, 3), dtype=_np.float32)})[0]
                conv_smoke = {'ok': bool(provider in set(got2) and tuple(yarr.shape) == (1, 4, 14, 14)), 'session_providers': got2}
            except Exception as conv_exc:
                conv_smoke = {'ok': False, 'error': f'{type(conv_exc).__name__}: {conv_exc}'}
        return {
            'ok': bool(active and tuple(out.shape) == (1, 4) and conv_smoke.get('ok') is not False),
            'requested_provider': provider,
            'session_providers': got,
            'provider_active': bool(active),
            'conv_smoke': conv_smoke,
            'cuda_preload': _cuda_preload_info,
        }
    except Exception as exc:
        return {'ok': False, 'requested_provider': provider, 'error': f'{type(exc).__name__}: {exc}'}

def _activation_proxy_provider_selection(ort: Any) -> Dict[str, Any]:
    requested = _activation_proxy_backend_request()
    try:
        available = list(ort.get_available_providers() or [])
    except Exception:
        available = []

    def has(ep: str) -> bool:
        return ep in set(available)

    def smoke(ep: str) -> Dict[str, Any]:
        return _activation_proxy_provider_session_smoke(ort, ep)

    fallback_reason = ''
    producer = 'ort_cpu'
    providers = ['CPUExecutionProvider']
    smoke_results: Dict[str, Any] = {}

    if compiler_dispatch_forbidden() and requested == 'tensorrt_ort':
        return {
            'requested_backend': requested,
            'producer_backend': 'ort_cpu',
            'source': 'ort_cpu_reference_proxy',
            'providers_requested': ['CPUExecutionProvider'],
            'available_providers': available,
            'fallback_reason': cache_miss_blocked_message(
                'tensorrt_ort_ep',
                'activation proxy forced to CPU without creating a TensorRT EP session',
            ),
            'provider_session_smoke': smoke_results,
        }

    if requested == 'tensorrt_ort':
        if has('TensorrtExecutionProvider'):
            smoke_results['TensorrtExecutionProvider'] = smoke('TensorrtExecutionProvider')
            if bool(smoke_results['TensorrtExecutionProvider'].get('ok')):
                producer = 'tensorrt_ort'
                providers = [p for p in ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if has(p)]
            elif has('CUDAExecutionProvider'):
                smoke_results['CUDAExecutionProvider'] = smoke('CUDAExecutionProvider')
                if bool(smoke_results['CUDAExecutionProvider'].get('ok')):
                    producer = 'cuda_ort'
                    providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has(p)]
                    fallback_reason = 'requested tensorrt_ort but TensorrtExecutionProvider session smoke failed; using cuda_ort proxy'
                else:
                    producer = 'ort_cpu'
                    providers = ['CPUExecutionProvider'] if has('CPUExecutionProvider') else list(available or ['CPUExecutionProvider'])
                    fallback_reason = 'requested tensorrt_ort but TensorRT and CUDA session smoke failed; using ort_cpu proxy'
            else:
                producer = 'ort_cpu'
                providers = ['CPUExecutionProvider'] if has('CPUExecutionProvider') else list(available or ['CPUExecutionProvider'])
                fallback_reason = 'requested tensorrt_ort but TensorRT session smoke failed and CUDAExecutionProvider is unavailable; using ort_cpu proxy'
        elif has('CUDAExecutionProvider'):
            smoke_results['CUDAExecutionProvider'] = smoke('CUDAExecutionProvider')
            if bool(smoke_results['CUDAExecutionProvider'].get('ok')):
                producer = 'cuda_ort'
                providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has(p)]
                fallback_reason = 'requested tensorrt_ort but TensorrtExecutionProvider is unavailable; using cuda_ort proxy'
            else:
                producer = 'ort_cpu'
                providers = ['CPUExecutionProvider'] if has('CPUExecutionProvider') else list(available or ['CPUExecutionProvider'])
                fallback_reason = 'requested tensorrt_ort but CUDA session smoke failed; using ort_cpu proxy'
        else:
            producer = 'ort_cpu'
            providers = ['CPUExecutionProvider'] if has('CPUExecutionProvider') else list(available or ['CPUExecutionProvider'])
            fallback_reason = 'requested tensorrt_ort but TensorRT/CUDA EPs are unavailable; using ort_cpu proxy'
    elif requested == 'cuda_ort':
        if has('CUDAExecutionProvider'):
            smoke_results['CUDAExecutionProvider'] = smoke('CUDAExecutionProvider')
            if bool(smoke_results['CUDAExecutionProvider'].get('ok')):
                producer = 'cuda_ort'
                providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if has(p)]
            else:
                producer = 'ort_cpu'
                providers = ['CPUExecutionProvider'] if has('CPUExecutionProvider') else list(available or ['CPUExecutionProvider'])
                fallback_reason = 'requested cuda_ort but CUDAExecutionProvider session smoke failed; using ort_cpu proxy'
        else:
            producer = 'ort_cpu'
            providers = ['CPUExecutionProvider'] if has('CPUExecutionProvider') else list(available or ['CPUExecutionProvider'])
            fallback_reason = 'requested cuda_ort but CUDAExecutionProvider is unavailable; using ort_cpu proxy'
    else:
        producer = 'ort_cpu'
        providers = ['CPUExecutionProvider'] if has('CPUExecutionProvider') else list(available or ['CPUExecutionProvider'])

    if not providers:
        providers = ['CPUExecutionProvider']
        producer = 'ort_cpu'
        fallback_reason = fallback_reason or 'no ONNX Runtime providers reported; using CPUExecutionProvider by name'

    source = f'{producer}_reference_proxy'
    return {
        'requested_backend': requested,
        'producer_backend': producer,
        'source': source,
        'providers_requested': providers,
        'available_providers': available,
        'fallback_reason': fallback_reason,
        'provider_session_smoke': smoke_results,
    }

def _build_activation_calib_from_part1_onnx(
    *,
    part1_onnx: Path,
    part2_onnx: Path,
    calib_dir: Path,
    limit: int,
    gen_batch: int,
    input_preprocess: str = 'norm',
    preprocessing_contract: Mapping[str, Any] | None = None,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Any]]:
    requested_backend_for_proxy = _activation_proxy_backend_request()
    if _activation_proxy_is_remote_backend(requested_backend_for_proxy):
        try:
            return _build_activation_calib_from_part1_onnx_remote_deepx(
                requested_backend=requested_backend_for_proxy,
                part1_onnx=part1_onnx,
                part2_onnx=part2_onnx,
                calib_dir=calib_dir,
                limit=limit,
                gen_batch=gen_batch,
                input_preprocess=input_preprocess,
                preprocessing_contract=preprocessing_contract,
            )
        except Exception as remote_exc:
            if _activation_proxy_strict_enabled():
                raise RuntimeError(
                    'Activation proxy strict mode: remote DeepX proxy failed and fallback is disabled; '
                    f'requested={requested_backend_for_proxy}; error={type(remote_exc).__name__}: {remote_exc}'
                ) from remote_exc
            old_env = os.environ.get('ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND')
            old_env2 = os.environ.get('SPLITPOINT_ACTIVATION_PROXY_BACKEND')
            os.environ['ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND'] = 'ort_cpu'
            os.environ['SPLITPOINT_ACTIVATION_PROXY_BACKEND'] = 'ort_cpu'
            try:
                names, arrays, debug = _build_activation_calib_from_part1_onnx(
                    part1_onnx=part1_onnx, part2_onnx=part2_onnx, calib_dir=calib_dir,
                    limit=limit, gen_batch=gen_batch, input_preprocess=input_preprocess,
                    preprocessing_contract=preprocessing_contract,
                )
            finally:
                if old_env is None: os.environ.pop('ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND', None)
                else: os.environ['ONNX_SPLITPOINT_ACTIVATION_PROXY_BACKEND'] = old_env
                if old_env2 is None: os.environ.pop('SPLITPOINT_ACTIVATION_PROXY_BACKEND', None)
                else: os.environ['SPLITPOINT_ACTIVATION_PROXY_BACKEND'] = old_env2
            debug['activation_proxy_requested_backend'] = requested_backend_for_proxy
            debug['activation_proxy_provider_fallback'] = (str(debug.get('activation_proxy_provider_fallback') or '') + ('; ' if debug.get('activation_proxy_provider_fallback') else '') + f'remote proxy failed ({type(remote_exc).__name__}: {remote_exc}); using ort_cpu proxy').strip()
            debug['activation_proxy_remote_fallback'] = True
            debug['activation_proxy_remote_fallback_error'] = f'{type(remote_exc).__name__}: {remote_exc}'
            return names, arrays, debug
    try:
        _cuda_preload_info = _preload_nvidia_cuda_wheel_libs_for_ort()
        import onnxruntime as ort  # type: ignore
        if hasattr(ort, 'preload_dlls'):
            try:
                ort.preload_dlls(directory='')
            except Exception:
                pass
    except Exception as exc:
        raise RuntimeError(f'onnxruntime is required to generate multi-input activation calibration: {type(exc).__name__}: {exc}')

    if not calib_dir.exists():
        raise FileNotFoundError(f'Calibration dir not found: {calib_dir}')
    calib_scan = _scan_calib_dir(calib_dir, recursive=True, limit=max(1, int(limit)))
    calib_items = list(calib_scan.get('items') or [])
    if not calib_items:
        raise FileNotFoundError(f'No calibration items (.npy/.npz/images) in: {calib_dir}')

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider_selection = _activation_proxy_provider_selection(ort)
    strict_proxy = _activation_proxy_strict_enabled()
    try:
        provider_selection.setdefault('cuda_wheel_preload', _cuda_preload_info)
        provider_selection['strict_proxy'] = bool(strict_proxy)
    except Exception:
        pass
    _req_backend = str(provider_selection.get('requested_backend') or 'ort_cpu')
    _prod_backend = str(provider_selection.get('producer_backend') or 'ort_cpu')
    _fb_reason = str(provider_selection.get('fallback_reason') or '')
    if strict_proxy and _req_backend in {'cuda_ort', 'tensorrt_ort'} and _prod_backend != _req_backend:
        raise RuntimeError(
            'Activation proxy strict mode: requested accelerated proxy did not pass provider selection; '
            f'requested={_req_backend}; selected={_prod_backend}; reason={_fb_reason or "provider unavailable or session smoke failed"}'
        )
    p1_providers = list(provider_selection.get('providers_requested') or ['CPUExecutionProvider'])
    # Part2 is inspected for input metadata only.  Keep this on CPU to avoid
    # building/optimizing the suffix while generating activation calibration.
    p2_providers = ['CPUExecutionProvider'] if 'CPUExecutionProvider' in set(provider_selection.get('available_providers') or []) else p1_providers
    if compiler_dispatch_forbidden() and any(
        str(provider) == 'TensorrtExecutionProvider'
        for provider in [*p1_providers, *p2_providers]
    ):
        raise RuntimeError(cache_miss_blocked_message(
            'tensorrt_ort_ep',
            'activation proxy attempted to create a TensorRT EP session',
        ))
    p1_sess = ort.InferenceSession(str(part1_onnx), sess_options=so, providers=p1_providers)
    p2_sess = ort.InferenceSession(str(part2_onnx), providers=p2_providers)

    p1_in = p1_sess.get_inputs()[0]
    p1_in_name = p1_in.name
    p1_out_names = [o.name for o in p1_sess.get_outputs()]

    p2_inputs = list(p2_sess.get_inputs())
    p2_in_names = [m.name for m in p2_inputs]
    mapping, missing, mapping_debug = _map_part2_inputs_to_part1_outputs(p1_out_names, p2_in_names)
    if missing:
        preflight = _activation_calib_preflight(part1_onnx=part1_onnx, part2_onnx=part2_onnx)
        raise RuntimeError(_format_activation_calib_preflight_error(preflight if preflight.get('inspect_ok') else mapping_debug))
    req_out_names = [mapping[n] for n in p2_in_names]
    per_input_samples: Dict[str, List[np.ndarray]] = {n: [] for n in p2_in_names}

    debug: Dict[str, Any] = {
        'activation_proxy_requested_backend': provider_selection.get('requested_backend'),
        'activation_proxy_producer_backend': provider_selection.get('producer_backend'),
        'activation_proxy_source': provider_selection.get('source'),
        'activation_proxy_providers_requested': list(provider_selection.get('providers_requested') or []),
        'activation_proxy_available_providers': list(provider_selection.get('available_providers') or []),
        'activation_proxy_provider_fallback': provider_selection.get('fallback_reason') or '',
        'activation_proxy_strict': bool(_activation_proxy_strict_enabled()),
        'activation_proxy_env_strict': str(os.environ.get('ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT') or os.environ.get('SPLITPOINT_ACTIVATION_PROXY_STRICT') or ''),
        'activation_proxy_part1_session_providers': list(p1_sess.get_providers() or []),
        'activation_proxy_part2_session_providers': list(p2_sess.get_providers() or []),
        'part1_input': {'name': p1_in_name, 'type': str(getattr(p1_in, 'type', '') or ''), 'shape': list(getattr(p1_in, 'shape', []) or [])},
        'mapping_part2in_to_part1out': dict(mapping),
        'mapping_debug': mapping_debug,
        'part2_inputs': {m.name: {'type': str(getattr(m, 'type', '') or ''), 'shape': [d if isinstance(d, int) else None for d in (getattr(m, 'shape', []) or [])]} for m in p2_inputs},
        'calib_source_dir': str(calib_dir),
        'input_preprocess': str(input_preprocess),
        'preprocessing_contract': dict(preprocessing_contract or {}),
        'preprocessing_contract_sha256': (
            preprocessing_contract_sha256(preprocessing_contract)
            if preprocessing_contract
            else None
        ),
        'calib_items_total': int(len(calib_items)),
        'calib_items_preview': list(calib_scan.get('preview') or []),
        'calib_item_suffixes': list(calib_scan.get('suffixes') or []),
        'calib_scan_recursive': bool(calib_scan.get('recursive', True)),
        'calib_scan_kind': str(calib_scan.get('kind') or 'unknown'),
        'calib_scan': {k: v for k, v in calib_scan.items() if k != 'items'},
    }

    bdim = None
    try:
        bdim = getattr(p1_in, 'shape', [None])[0]
    except Exception:
        bdim = None
    gen_batch = max(1, int(gen_batch))
    if isinstance(bdim, int) and bdim == 1 and gen_batch != 1:
        gen_batch = 1

    idx = 0
    N = len(calib_items)
    runtime_fallback_done = False
    while idx < N:
        batch_paths = calib_items[idx: idx + gen_batch]
        xs: List[np.ndarray] = []
        for pp in batch_paths:
            arr = _load_calib_item_any(pp)
            xs.append(
                _prepare_part1_input_for_activation_calib(
                    arr,
                    p1_in,
                    preprocess=input_preprocess,
                    preprocessing_contract=preprocessing_contract,
                )
            )
        xb = np.concatenate(xs, axis=0) if len(xs) > 1 else xs[0]
        try:
            outs = p1_sess.run(req_out_names, {p1_in_name: xb})
        except Exception as run_exc:
            # CUDA/TensorRT provider listing and tiny smoke tests can still pass while
            # the real Part1 graph fails later, e.g. with CUDNN_FE errors on specific
            # convolution shapes.  By default we fall back to CPU ORT for broad-screening
            # continuity.  Set ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT=1 to make such
            # fallback a hard error; this is useful when verifying that the selected
            # proxy backend was actually used.
            cur_providers = list(p1_sess.get_providers() or [])
            if (not runtime_fallback_done) and any(p != 'CPUExecutionProvider' for p in cur_providers):
                strict_proxy = _activation_proxy_strict_enabled()
                fallback_reason = f'{type(run_exc).__name__}: {run_exc}'
                if strict_proxy:
                    raise RuntimeError(
                        'Activation proxy strict mode: requested accelerated proxy failed during real Part1 forward; '
                        f'env_strict={os.environ.get("ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT")!r}; providers={cur_providers}; error={fallback_reason}'
                    ) from run_exc
                runtime_fallback_done = True
                debug['activation_proxy_runtime_fallback'] = True
                debug['activation_proxy_runtime_fallback_from_providers'] = cur_providers
                debug['activation_proxy_runtime_fallback_reason'] = fallback_reason
                debug['activation_proxy_requested_backend_before_runtime_fallback'] = debug.get('activation_proxy_requested_backend')
                debug['activation_proxy_producer_backend_before_runtime_fallback'] = debug.get('activation_proxy_producer_backend')
                debug['activation_proxy_source_before_runtime_fallback'] = debug.get('activation_proxy_source')
                debug['activation_proxy_producer_backend'] = 'ort_cpu'
                debug['activation_proxy_source'] = 'ort_cpu_reference_proxy'
                debug['activation_proxy_provider_fallback'] = (str(debug.get('activation_proxy_provider_fallback') or '') + ('; ' if debug.get('activation_proxy_provider_fallback') else '') + 'runtime inference failed on accelerated provider; using ort_cpu proxy').strip()
                p1_sess = ort.InferenceSession(str(part1_onnx), sess_options=so, providers=['CPUExecutionProvider'])
                debug['activation_proxy_part1_session_providers_after_runtime_fallback'] = list(p1_sess.get_providers() or [])
                per_input_samples = {n: [] for n in p2_in_names}
                idx = 0
                continue
            raise
        for p2_name, out_arr in zip(p2_in_names, outs):
            oa = np.asarray(out_arr)
            if oa.ndim >= 1 and oa.shape[0] == xb.shape[0]:
                for bi in range(int(oa.shape[0])):
                    feat = np.asarray(oa[bi]).astype(np.float32, copy=False)
                    per_input_samples[p2_name].append(np.ascontiguousarray(feat))
            else:
                feat = np.asarray(oa).astype(np.float32, copy=False)
                per_input_samples[p2_name].append(np.ascontiguousarray(feat))
        idx += gen_batch

    calib_arrays: Dict[str, np.ndarray] = {}
    for p2_name, samples in per_input_samples.items():
        if not samples:
            raise RuntimeError(f'No activation calibration samples generated for part2 input {p2_name}')
        calib_arrays[p2_name] = np.stack(samples, axis=0).astype(np.float32)

    debug['calib_shapes_before_hn'] = {k: {'dataset_shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in calib_arrays.items()}
    return p2_in_names, calib_arrays, debug



def _array_stats_for_manifest(arr: np.ndarray) -> Dict[str, Any]:
    """Small, JSON-safe statistics for activation proxy calibration tensors."""
    try:
        x = np.asarray(arr)
        if x.size <= 0:
            return {"shape": list(x.shape), "dtype": str(x.dtype), "empty": True}
        xf = x.astype(np.float64, copy=False).reshape(-1)
        return {
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "min": float(np.min(xf)),
            "max": float(np.max(xf)),
            "mean": float(np.mean(xf)),
            "std": float(np.std(xf)),
            "p01": float(np.percentile(xf, 1.0)),
            "p50": float(np.percentile(xf, 50.0)),
            "p99": float(np.percentile(xf, 99.0)),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _write_activation_proxy_cache_manifest(
    *,
    out_dir: Path,
    part1_onnx: Path,
    part2_onnx: Path,
    calib_dir: Optional[Path],
    calib_arrays_by_part2_input: Dict[str, np.ndarray],
    activation_debug: Optional[Dict[str, Any]],
    eff_count: int,
    gen_batch: int,
    stage2_backend: str = "hailo",
    store_samples_env: str = "ONNX_SPLITPOINT_ACTIVATION_PROXY_STORE_SAMPLES",
    force_store_samples: int = 0,
) -> Optional[str]:
    """Persist a lightweight ORT-CPU activation proxy calibration manifest.

    The Hailo DFC compile path already generated the cut tensors in memory and
    consumed them for optimizer calibration.  Persisting the full activation
    dataset for every split would be expensive, so the default is a manifest +
    statistics only.  Set ``ONNX_SPLITPOINT_ACTIVATION_PROXY_STORE_SAMPLES`` to
    a small positive integer to store a few sample NPZ files for debugging.
    """
    try:
        cache_dir = out_dir / "activation_proxy_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stats: Dict[str, Any] = {}
        tensors: List[Dict[str, Any]] = []
        for name, arr in calib_arrays_by_part2_input.items():
            x = np.asarray(arr)
            st = _array_stats_for_manifest(x)
            stats[str(name)] = st
            sample_shape = list(x.shape[1:]) if x.ndim >= 1 else list(x.shape)
            tensors.append({
                "name": str(name),
                "dataset_shape": list(x.shape),
                "sample_shape": sample_shape,
                "dtype": str(x.dtype),
                "layout": "unknown",
                "stats": st,
            })
        store_n = 0
        try:
            store_n = max(0, int(os.environ.get(store_samples_env, "0") or "0"))
        except Exception:
            store_n = 0
        try:
            store_n = max(store_n, int(force_store_samples or 0))
        except Exception:
            pass
        stored_sample_count = 0
        if store_n > 0 and calib_arrays_by_part2_input:
            first = next(iter(calib_arrays_by_part2_input.values()))
            try:
                n_avail = int(np.asarray(first).shape[0])
            except Exception:
                n_avail = 0
            stored_sample_count = min(store_n, n_avail)
            for i in range(stored_sample_count):
                payload = {str(name): np.asarray(arr)[i].astype(np.float32, copy=False) for name, arr in calib_arrays_by_part2_input.items() if np.asarray(arr).shape[0] > i}
                np.savez_compressed(str(cache_dir / f"sample_{i:06d}.npz"), **payload)
        producer_backend = str((activation_debug or {}).get('activation_proxy_producer_backend') or 'ort_cpu')
        proxy_source = str((activation_debug or {}).get('activation_proxy_source') or f'{producer_backend}_reference_proxy')
        requested_backend = str((activation_debug or {}).get('activation_proxy_requested_backend') or 'ort_cpu')
        fallback_reason = str((activation_debug or {}).get('activation_proxy_provider_fallback') or '')
        manifest = {
            "schema_version": 2,
            "cache_kind": "activation_calibration",
            "cache_version": "v52u_proxy",
            "source": proxy_source,
            "calibration_source": proxy_source,
            "producer_backend": producer_backend,
            "requested_backend": requested_backend,
            "producer_exact": False,
            "trust_level": "proxy",
            "status": "ready",
            "compiler_consumed_direct": True,
            "note": "Cut tensors were generated from an ONNX Runtime Part1 proxy during accelerator Part2 compilation. This is a fast proxy calibration source for broad screening; producer-exact calibration is reserved for final candidates or failed proxy cases.",
            "provider_fallback_reason": fallback_reason,
            "stage2_backend": str(stage2_backend),
            "part1_onnx": str(part1_onnx),
            "part2_onnx": str(part2_onnx),
            "calib_source_dir": str(calib_dir) if calib_dir is not None else None,
            "requested_count": int(eff_count),
            "sample_count": int(next(iter(calib_arrays_by_part2_input.values())).shape[0]) if calib_arrays_by_part2_input else int(eff_count),
            "stored_sample_count": int(stored_sample_count),
            "sample_storage": "npz_debug_samples" if stored_sample_count else "manifest_stats_only",
            "activation_gen_batch": int(gen_batch),
            "preprocessing_contract": dict(
                (activation_debug or {}).get("preprocessing_contract") or {}
            ),
            "preprocessing_contract_sha256": (
                (activation_debug or {}).get("preprocessing_contract_sha256")
            ),
            "tensors": tensors,
            "stats": stats,
            "debug": activation_debug or {},
        }
        p_manifest = cache_dir / "manifest.json"
        p_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            (cache_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return str(p_manifest)
    except Exception:
        log.debug("Failed to persist activation proxy cache manifest", exc_info=True)
        return None

def _bare_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


_HAILO_HEF_RECEIPT_NAME = "hailo_hef_build_receipt.json"
_HAILO_HEF_RECEIPT_SCHEMA = "onnx-splitpoint/hailo-hef-build-receipt/v2"
_HAILO_HEF_CACHE_SCHEMA_V2 = "onnx-splitpoint/hailo-hef-cache-key-v2"
_HAILO_HEF_CACHE_SCHEMA_V3 = "onnx-splitpoint/hailo-hef-cache-key-v3"
_HAILO_HEF_CACHE_SCHEMA = _HAILO_HEF_CACHE_SCHEMA_V3
_HAILO_HEF_CACHE_SCHEMAS = {
    _HAILO_HEF_CACHE_SCHEMA_V2, _HAILO_HEF_CACHE_SCHEMA_V3,
}


def _first_shape(value: Any) -> Optional[List[int]]:
    if isinstance(value, dict):
        for shape in value.values():
            if isinstance(shape, (list, tuple)):
                return [int(v) for v in shape]
        return None
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return None


def _onnx_first_input_shape(path: Path) -> Optional[List[int]]:
    if onnx is None or not path.is_file():
        return None
    try:
        model = onnx.load(str(path), load_external_data=False)
        shapes = infer_net_input_shapes_from_model(model)
        return _first_shape(shapes)
    except Exception:
        return None


def _resolve_hailo_image_contract(
    *,
    model_path: Path,
    activation_part1: Path | None,
    net_input_shapes: Any,
    task: Any = None,
    declared: Mapping[str, Any] | str | None = None,
) -> tuple[dict[str, Any], str]:
    """Resolve the semantic calibration contract before cache/DFC activity."""

    declared_eff: Mapping[str, Any] | str | None = declared
    if declared_eff is None:
        declared_eff = str(
            os.environ.get("ONNX_SPLITPOINT_HAILO_PREPROCESSING_CONTRACT_JSON")
            or os.environ.get("SPLITPOINT_HAILO_PREPROCESSING_CONTRACT_JSON")
            or ""
        ).strip() or None
    task_declared = str(
        task
        or os.environ.get("ONNX_SPLITPOINT_HAILO_CALIB_TASK")
        or os.environ.get("SPLITPOINT_HAILO_CALIB_TASK")
        or ""
    ).strip()
    declared_payload: Mapping[str, Any] | None = None
    if declared_eff is not None:
        if isinstance(declared_eff, Mapping):
            declared_payload = declared_eff
        elif isinstance(declared_eff, str):
            text = declared_eff.strip()
            if text:
                parsed = json.loads(text)
                if not isinstance(parsed, Mapping):
                    raise ValueError(
                        "Declared Hailo image preprocessing contract must be a JSON object"
                    )
                declared_payload = parsed
        else:
            raise ValueError(
                "Declared Hailo image preprocessing contract must be a mapping or JSON object"
            )
    if not task_declared and declared_payload is not None:
        task_declared = str(declared_payload.get("task") or "").strip()
    if not task_declared:
        raise ValueError(
            "Hailo image preprocessing requires an explicit task='detection' or "
            "task='classification' (or a declared preprocessing contract containing "
            "that task). Model names and input dimensions must not infer semantics."
        )
    task_eff = normalize_image_task(task_declared)
    semantic_model = activation_part1 if activation_part1 is not None else model_path
    shape = (
        _onnx_first_input_shape(semantic_model)
        if activation_part1 is not None
        else _first_shape(net_input_shapes) or _onnx_first_input_shape(model_path)
    )
    if shape is None:
        raise ValueError(
            "Cannot seal Hailo preprocessing: the semantic image input shape is unavailable"
        )
    target_hw = target_hw_from_shape(shape)
    return resolve_image_preprocessing_contract(
        task=task_eff, target_hw=target_hw, declared=declared_eff
    )


def _hailo_receipt_path(hef_path: Path) -> Path:
    # Resolve once so a concurrent bundle publication cannot mix generations.
    return Path(hef_path).resolve().parent / _HAILO_HEF_RECEIPT_NAME


def _hailo_cache_meta_path(hef_path: Path) -> Path:
    return Path(hef_path).resolve().parent / "cache_meta.json"


def _hailo_cache_meta_from_receipt(
    receipt: Mapping[str, Any], *, source: str = "hailo_build_hef",
) -> Dict[str, Any]:
    return {
        "schema": "onnx-splitpoint/hailo-hef-cache-meta-v2",
        "cache_key": str(receipt.get("cache_key") or ""),
        "payload": dict(receipt.get("cache_payload") or {}),
        "created_at": time.time(),
        "hef_size": receipt.get("hef_size_bytes"),
        "hef_sha256": str(receipt.get("hef_sha256") or ""),
        "preprocessing_contract_sha256": str(
            receipt.get("preprocessing_contract_sha256") or ""
        ),
        "net_name": str(receipt.get("net_name") or ""),
        "hw_arch": str(receipt.get("hw_arch") or ""),
        "source": source,
    }


def _hailo_cache_meta_matches_receipt(
    metadata: Mapping[str, Any], receipt: Mapping[str, Any],
) -> bool:
    expected = _hailo_cache_meta_from_receipt(receipt)
    return all(
        metadata.get(key) == value
        for key, value in expected.items()
        if key not in {"created_at", "source"}
    )


def _publish_hailo_bundle(
    *, source_hef: Path, destination: Path, receipt: Mapping[str, Any],
    source: str = "hailo_build_hef",
) -> Path:
    from .hailo_cache_bundle import publish_bundle
    payload = dict(receipt.get("cache_payload") or {})
    return publish_bundle(
        source_hef=source_hef, destination=destination, receipt=receipt,
        cache_meta=_hailo_cache_meta_from_receipt(receipt, source=source),
        validator=lambda staged: _load_valid_hailo_receipt(
            staged,
            cache_key=str(receipt.get("cache_key") or ""),
            cache_payload=payload,
            source_onnx_sha256=str(receipt.get("source_onnx_sha256") or ""),
            preprocessing_sha256=str(receipt.get("preprocessing_contract_sha256") or ""),
            expected_net_name=str(receipt.get("net_name") or ""),
            expected_net_input_shapes=payload.get("net_input_shapes"),
            expected_disable_rt_metadata_extraction=payload.get("disable_rt_metadata_extraction"),
            allow_legacy_v2=payload.get("schema") == _HAILO_HEF_CACHE_SCHEMA_V2,
            allow_diagnostic=receipt.get("diagnostic_only") is True,
        ),
    )


def _hailo_cache_bundle_status(hef_path: Path) -> Dict[str, Any]:
    """Read-only classification; HEF-only generations never count as hits."""
    hef = Path(hef_path).resolve()
    receipt_path = _hailo_receipt_path(hef)
    meta_path = _hailo_cache_meta_path(hef)
    result: Dict[str, Any] = {
        "hef_path": str(hef), "receipt_path": str(receipt_path),
        "cache_meta_path": str(meta_path),
        "hef_present": hef.is_file(), "receipt_present": receipt_path.is_file(),
        "cache_meta_present": meta_path.is_file(), "reusable": False,
    }
    if not hef.is_file():
        result["status"] = "missing"
    elif not receipt_path.is_file():
        result["status"] = "legacy_unsealed"
    else:
        receipt = _load_valid_hailo_receipt(hef, validate_cache_meta=False)
        if receipt is None:
            result["status"] = "receipt_invalid"
        elif not meta_path.is_file():
            result.update(status="metadata_missing", reusable=(
                hef.parent.parent.name != ".hailo-generations"
            ))
        else:
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                valid = isinstance(metadata, dict) and _hailo_cache_meta_matches_receipt(metadata, receipt)
            except Exception:
                valid = False
            result.update(status="sealed" if valid else "metadata_invalid", reusable=bool(valid))
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _load_valid_hailo_receipt(
    hef_path: Path,
    *,
    preprocessing_sha256: str | None = None,
    source_onnx_sha256: str | None = None,
    cache_key: str | None = None,
    cache_payload: Mapping[str, Any] | None = None,
    expected_net_name: str | None = None,
    expected_net_input_shapes: Any = None,
    expected_disable_rt_metadata_extraction: bool | None = None,
    allow_legacy_v2: bool = False,
    validate_cache_meta: bool = True,
    allow_diagnostic: bool = False,
    receipt_override: Mapping[str, Any] | None = None,
    hash_fn: Any = None,
) -> Optional[Dict[str, Any]]:
    hef_path = Path(hef_path).resolve()
    receipt_path = _hailo_receipt_path(hef_path)
    try:
        receipt = (dict(receipt_override) if receipt_override is not None
                   else json.loads(receipt_path.read_text(encoding="utf-8")))
        if not isinstance(receipt, dict):
            return None
        if not allow_diagnostic and (receipt.get("diagnostic_only") is True or receipt.get("publish_artifacts") is False):
            return None

        def _sha_token(value: Any) -> str:
            token = str(value or "").strip().lower()
            if token.startswith("sha256:"):
                token = token[7:]
            if len(token) != 64 or any(
                character not in "0123456789abcdef" for character in token
            ):
                return ""
            return token

        if receipt.get("schema") != _HAILO_HEF_RECEIPT_SCHEMA:
            return None
        if not hef_path.is_file() or hef_path.stat().st_size <= 0:
            return None
        actual_hef_sha = _sha_token((
            _bare_file_sha256 if hash_fn is None else hash_fn
        )(hef_path))
        if not actual_hef_sha or _sha_token(receipt.get("hef_sha256")) != actual_hef_sha:
            return None
        receipt_hef_size = receipt.get("hef_size_bytes")
        if (
            type(receipt_hef_size) is not int
            or receipt_hef_size <= 0
            or receipt_hef_size != int(hef_path.stat().st_size)
        ):
            return None
        source_sha = _sha_token(receipt.get("source_onnx_sha256"))
        compiler_sha = _sha_token(receipt.get("compiler_onnx_sha256"))
        compiler_filename = str(
            receipt.get("compiler_onnx_filename") or ""
        ).strip()
        if (
            not source_sha
            or not compiler_sha
            or not compiler_filename
            or Path(compiler_filename).name != compiler_filename
            or not compiler_filename.lower().endswith(".onnx")
        ):
            return None
        if source_onnx_sha256 and source_sha != _sha_token(source_onnx_sha256):
            return None
        if not str(receipt.get("hw_arch") or "").strip():
            return None
        receipt_net_name = str(receipt.get("net_name") or "").strip()
        if not receipt_net_name:
            return None
        if (
            expected_net_name is not None
            and receipt_net_name != str(expected_net_name).strip()
        ):
            return None

        contract = receipt.get("preprocessing_contract")
        if not isinstance(contract, dict):
            return None
        resolved_contract, resolved_preprocessing_sha = (
            resolve_image_preprocessing_contract(
                task=contract.get("task"),
                target_hw=contract.get("target_hw"),
                declared=contract,
            )
        )
        receipt_preprocessing_sha = _sha_token(
            receipt.get("preprocessing_contract_sha256")
        )
        if (
            dict(resolved_contract) != contract
            or receipt_preprocessing_sha != resolved_preprocessing_sha
        ):
            return None
        if (
            preprocessing_sha256
            and receipt_preprocessing_sha != _sha_token(preprocessing_sha256)
        ):
            return None

        sealed_cache_payload = receipt.get("cache_payload")
        if not isinstance(sealed_cache_payload, dict):
            return None
        cache_schema = sealed_cache_payload.get("schema")
        if cache_schema not in _HAILO_HEF_CACHE_SCHEMAS:
            return None
        receipt_cache_key = _sha_token(receipt.get("cache_key"))
        calculated_cache_key = hashlib.sha256(
            json.dumps(
                sealed_cache_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if not receipt_cache_key or receipt_cache_key != calculated_cache_key:
            return None
        if cache_key and receipt_cache_key != _sha_token(cache_key):
            return None
        if cache_payload is not None and sealed_cache_payload != dict(cache_payload):
            return None
        if _sha_token(sealed_cache_payload.get("model_sha256")) != compiler_sha:
            return None
        if dict(sealed_cache_payload.get("preprocessing_contract") or {}) != contract:
            return None
        if _sha_token(
            sealed_cache_payload.get("preprocessing_contract_sha256")
        ) != receipt_preprocessing_sha:
            return None
        if str(sealed_cache_payload.get("hw_arch") or "") != str(
            receipt.get("hw_arch") or ""
        ):
            return None
        if cache_schema == _HAILO_HEF_CACHE_SCHEMA_V3:
            try:
                sealed_shapes = _normalize_hailo_net_input_shapes(
                    sealed_cache_payload.get("net_input_shapes")
                )
            except ValueError:
                return None
            if (
                str(sealed_cache_payload.get("net_name") or "").strip()
                != receipt_net_name
                or sealed_cache_payload.get("net_input_shapes")
                != sealed_shapes
                or type(
                    sealed_cache_payload.get(
                        "disable_rt_metadata_extraction"
                    )
                ) is not bool
            ):
                return None
            if (
                expected_net_name is not None
                and str(sealed_cache_payload.get("net_name") or "").strip()
                != str(expected_net_name).strip()
            ):
                return None
            if (
                expected_disable_rt_metadata_extraction is not None
                and sealed_cache_payload.get(
                    "disable_rt_metadata_extraction"
                ) is not bool(expected_disable_rt_metadata_extraction)
            ):
                return None
            if (
                expected_net_input_shapes is not None
                and sealed_cache_payload.get("net_input_shapes")
                != _normalize_hailo_net_input_shapes(
                    expected_net_input_shapes
                )
            ):
                return None
        elif (
            not allow_legacy_v2
            and (
                expected_net_name is not None
                or expected_net_input_shapes is not None
                or expected_disable_rt_metadata_extraction is not None
            )
        ):
            return None
        receipt_sdk = str(receipt.get("hailo_sdk_version") or "").strip()
        if not receipt_sdk or receipt_sdk != str(
            sealed_cache_payload.get("hailo_sdk_version") or ""
        ).strip():
            return None
        receipt_count = receipt.get("calibration_count")
        cache_count = sealed_cache_payload.get("calibration_count")
        receipt_requested_count = receipt.get(
            "requested_calibration_count"
        )
        cache_requested_count = sealed_cache_payload.get(
            "requested_calibration_count"
        )
        receipt_storage = str(
            receipt.get("calibration_storage") or ""
        ).strip().lower()
        cache_storage = str(
            sealed_cache_payload.get("calibration_storage") or ""
        ).strip().lower()
        receipt_memory_cap = receipt.get(
            "calibration_memory_cap_bytes"
        )
        cache_memory_cap = sealed_cache_payload.get(
            "calibration_memory_cap_bytes"
        )
        cache_batch_size = sealed_cache_payload.get(
            "calibration_batch_size"
        )
        cache_integrity = str(
            sealed_cache_payload.get("integrity") or ""
        )
        cache_nodes_valid = all(
            isinstance(values, list)
            and all(
                isinstance(value, str)
                and value
                and value == value.strip()
                for value in values
            )
            and len(values) == len(set(values))
            for values in (
                sealed_cache_payload.get("start_nodes"),
                sealed_cache_payload.get("end_nodes"),
            )
        )
        if (
            type(receipt_count) is not int
            or type(cache_count) is not int
            or receipt_count <= 0
            or cache_count <= 0
            or receipt_count != cache_count
            or type(receipt_requested_count) is not int
            or type(cache_requested_count) is not int
            or receipt_requested_count < receipt_count
            or receipt_requested_count != cache_requested_count
            or receipt_storage not in {"memory", "memmap"}
            or receipt_storage != cache_storage
            or type(receipt_memory_cap) is not int
            or type(cache_memory_cap) is not int
            or receipt_memory_cap <= 0
            or receipt_memory_cap != cache_memory_cap
            or type(cache_batch_size) is not int
            or cache_batch_size <= 0
            or cache_integrity not in {"strict", "relaxed"}
            or not cache_nodes_valid
        ):
            return None
        receipt_calibration = str(
            receipt.get("calibration_identity") or ""
        ).strip()
        cache_calibration = str(
            sealed_cache_payload.get("calibration_identity") or ""
        ).strip()
        if not receipt_calibration or receipt_calibration != cache_calibration:
            return None
        prepared_calibration_sha = hashlib.sha256(
            json.dumps(
                {
                    "calibration_identity": cache_calibration,
                    "preprocessing_contract_sha256": receipt_preprocessing_sha,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        if _sha_token(
            receipt.get("prepared_calibration_identity_sha256")
        ) != prepared_calibration_sha:
            return None
        if _sha_token(
            sealed_cache_payload.get("prepared_calibration_identity_sha256")
        ) != prepared_calibration_sha:
            return None
        if validate_cache_meta:
            meta_path = _hailo_cache_meta_path(hef_path)
            if meta_path.is_file():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict) or not _hailo_cache_meta_matches_receipt(metadata, receipt):
                    return None
            elif hef_path.parent.parent.name == ".hailo-generations":
                return None
        return dict(receipt)
    except Exception:
        return None


def _write_hailo_receipt(
    *,
    hef_path: Path,
    source_onnx: Path,
    compiler_onnx: Path,
    hw_arch: str,
    net_name: str,
    preprocessing_contract: Mapping[str, Any],
    preprocessing_sha256: str,
    cache_key: str,
    cache_payload: Mapping[str, Any],
    calibration_identity: str,
    calibration_count: int,
) -> Dict[str, Any]:
    sealed_cache_payload = dict(cache_payload)
    try:
        resolved_contract, resolved_preprocessing_sha = (
            resolve_image_preprocessing_contract(
                task=preprocessing_contract.get("task"),
                target_hw=preprocessing_contract.get("target_hw"),
                declared=preprocessing_contract,
            )
        )
    except Exception as exc:
        raise ValueError(
            "inconsistent_hailo_cache_receipt_preprocessing"
        ) from exc
    calculated_cache_key = hashlib.sha256(
        json.dumps(
            sealed_cache_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    cache_count = sealed_cache_payload.get("calibration_count")
    requested_count = sealed_cache_payload.get(
        "requested_calibration_count"
    )
    storage = str(
        sealed_cache_payload.get("calibration_storage") or ""
    ).strip().lower()
    memory_cap_bytes = sealed_cache_payload.get(
        "calibration_memory_cap_bytes"
    )
    sdk_version = _hailo_sdk_version_token()
    cache_calibration_identity = str(
        sealed_cache_payload.get("calibration_identity") or ""
    )
    prepared_calibration_identity = hashlib.sha256(
        json.dumps(
            {
                "calibration_identity": cache_calibration_identity,
                "preprocessing_contract_sha256": resolved_preprocessing_sha,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    start_nodes = sealed_cache_payload.get("start_nodes")
    end_nodes = sealed_cache_payload.get("end_nodes")
    nodes_valid = all(
        isinstance(values, list)
        and all(
            isinstance(value, str)
            and value
            and value == value.strip()
            for value in values
        )
        and len(values) == len(set(values))
        for values in (start_nodes, end_nodes)
    )
    if (
        sealed_cache_payload.get("schema") not in _HAILO_HEF_CACHE_SCHEMAS
        or str(cache_key) != calculated_cache_key
        or type(cache_count) is not int
        or cache_count <= 0
        or cache_count != int(calibration_count)
        or type(requested_count) is not int
        or requested_count < cache_count
        or storage not in {"memory", "memmap"}
        or type(memory_cap_bytes) is not int
        or memory_cap_bytes <= 0
        or str(sealed_cache_payload.get("hw_arch") or "")
        != str(hw_arch)
        or str(sealed_cache_payload.get("hailo_sdk_version") or "")
        != sdk_version
        or str(sealed_cache_payload.get("calibration_identity") or "")
        != str(calibration_identity)
        or str(sealed_cache_payload.get("model_sha256") or "")
        != _bare_file_sha256(compiler_onnx)
        or dict(resolved_contract) != dict(preprocessing_contract)
        or str(preprocessing_sha256) != resolved_preprocessing_sha
        or dict(sealed_cache_payload.get("preprocessing_contract") or {})
        != dict(resolved_contract)
        or str(
            sealed_cache_payload.get("preprocessing_contract_sha256") or ""
        ) != resolved_preprocessing_sha
        or str(
            sealed_cache_payload.get(
                "prepared_calibration_identity_sha256"
            ) or ""
        ) != prepared_calibration_identity
        or not nodes_valid
    ):
        raise ValueError("inconsistent_hailo_cache_receipt_identity")
    if sealed_cache_payload.get("schema") == _HAILO_HEF_CACHE_SCHEMA_V3:
        try:
            normalized_shapes = _normalize_hailo_net_input_shapes(
                sealed_cache_payload.get("net_input_shapes")
            )
        except ValueError:
            normalized_shapes = object()
        if (
            str(sealed_cache_payload.get("net_name") or "").strip()
            != str(net_name).strip()
            or type(
                sealed_cache_payload.get(
                    "disable_rt_metadata_extraction"
                )
            ) is not bool
            or "net_input_shapes" not in sealed_cache_payload
            or sealed_cache_payload.get("net_input_shapes")
            != normalized_shapes
        ):
            raise ValueError("inconsistent_hailo_cache_receipt_identity")
    receipt = {
        "schema": _HAILO_HEF_RECEIPT_SCHEMA,
        "created_at_unix_s": time.time(),
        "source_onnx_sha256": _bare_file_sha256(source_onnx),
        "compiler_onnx_sha256": _bare_file_sha256(compiler_onnx),
        "compiler_onnx_filename": compiler_onnx.name,
        "hef_sha256": _bare_file_sha256(hef_path),
        "hef_size_bytes": int(hef_path.stat().st_size),
        "hw_arch": str(hw_arch),
        "net_name": str(net_name),
        "hailo_sdk_version": sdk_version,
        "calibration_identity": str(calibration_identity),
        "prepared_calibration_identity_sha256": str(
            prepared_calibration_identity
        ),
        "calibration_count": int(calibration_count),
        "requested_calibration_count": int(requested_count),
        "calibration_storage": storage,
        "calibration_memory_cap_bytes": int(memory_cap_bytes),
        "preprocessing_contract": dict(preprocessing_contract),
        "preprocessing_contract_sha256": str(preprocessing_sha256),
        "cache_key": str(cache_key),
        "cache_payload": sealed_cache_payload,
    }
    _atomic_write_json(_hailo_receipt_path(hef_path), receipt)
    return receipt


def _calibration_identity(calib_dir: Path | None, *, strict: bool) -> str:
    if calib_dir is None or not calib_dir.exists():
        return 'none'
    manifest_hint = str(os.environ.get('ONNX_SPLITPOINT_HAILO_CALIB_MANIFEST') or '').strip()
    if manifest_hint:
        mp = Path(os.path.expanduser(manifest_hint))
        if mp.is_file():
            return 'manifest:' + _bare_file_sha256(mp)
    rows = []
    for path in sorted(p for p in calib_dir.rglob('*') if p.is_file()):
        try:
            st = path.stat()
            rel = str(path.relative_to(calib_dir)).replace('\\', '/')
            identity = _bare_file_sha256(path) if strict else f'{st.st_size}:{st.st_mtime_ns}'
            rows.append((rel, identity))
        except Exception:
            continue
    return hashlib.sha256(json.dumps(rows, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()


def _hailo_sdk_version_token_from_controller_metadata() -> str | None:
    """Return controller-side DFC identity without importing compiler code."""

    try:
        from importlib.metadata import version
        for package in ("hailo-dataflow-compiler", "hailo_sdk_client", "hailo-model-zoo"):
            try:
                value = str(version(package) or "").strip()
                if value:
                    return f"{package}:{value}"
            except Exception:
                continue
    except Exception:
        pass
    return None


def _hailo_sdk_version_token() -> str:
    """Best-effort DFC version token for build-cache invalidation."""

    metadata_token = _hailo_sdk_version_token_from_controller_metadata()
    if metadata_token:
        return metadata_token
    try:
        import hailo_sdk_client  # type: ignore
        value = str(getattr(hailo_sdk_client, "__version__", "") or "").strip()
        if value:
            return f"hailo_sdk_client:{value}"
    except Exception:
        pass
    return "unknown"


def _hailo_sdk_version_token_from_managed_venv(
    *,
    hw_arch: str,
    venv_activate: str = "auto",
) -> str | None:
    """Read the managed DFC distribution version without executing its Python.

    Cache lookup happens in the controller interpreter, while Hailo builds run
    in a dedicated managed venv.  Using the controller's package metadata made
    an existing managed-venv cache invisible whenever the controller did not
    itself have DFC installed.  Reading ``*.dist-info`` through
    :mod:`importlib.metadata` is process-free and preserves the exact token the
    compiler interpreter writes into receipts.
    """

    try:
        from importlib.metadata import distributions

        _profile_id, python_path, _activate = _resolve_managed_venv_python(
            hw_arch=str(hw_arch),
            venv_activate=str(venv_activate or "auto"),
        )
    except Exception:
        return None

    venv_dir = Path(python_path).parent.parent
    site_packages = sorted({
        path.resolve()
        for lib_name in ("lib", "lib64")
        for path in (venv_dir / lib_name).glob("python*/site-packages")
        if path.is_dir()
    })
    if not site_packages:
        return None

    def _normalise_distribution_name(value: Any) -> str:
        return re.sub(r"[-_.]+", "-", str(value or "").strip().lower())

    found: dict[str, set[str]] = {}
    try:
        for distribution in distributions(
            path=[str(path) for path in site_packages]
        ):
            name = str(distribution.metadata.get("Name") or "").strip()
            version = str(distribution.version or "").strip()
            if not name or not version:
                continue
            found.setdefault(
                _normalise_distribution_name(name), set()
            ).add(version)
    except Exception:
        return None

    for package in (
        "hailo-dataflow-compiler",
        "hailo_sdk_client",
        "hailo-model-zoo",
    ):
        versions = found.get(_normalise_distribution_name(package), set())
        if len(versions) == 1:
            return f"{package}:{next(iter(versions))}"
        if len(versions) > 1:
            # Multiple versions for the authoritative distribution are an
            # ambiguous compiler identity.  Fail closed instead of guessing.
            return None
    return None


def _normalize_hailo_net_input_shapes(value: Any) -> Any:
    """Return a canonical JSON form for the compiler's input-shape override."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        normalized: Dict[str, List[int]] = {}
        for name in sorted(value, key=lambda item: str(item)):
            shape = value[name]
            if not isinstance(shape, (list, tuple)) or not shape:
                raise ValueError("invalid_hailo_net_input_shapes")
            values = []
            for dimension in shape:
                if isinstance(dimension, bool) or not isinstance(dimension, int):
                    raise ValueError("invalid_hailo_net_input_shapes")
                values.append(int(dimension))
            normalized[str(name)] = values
        return normalized
    if isinstance(value, (list, tuple)) and value:
        values = []
        for dimension in value:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise ValueError("invalid_hailo_net_input_shapes")
            values.append(int(dimension))
        return values
    raise ValueError("invalid_hailo_net_input_shapes")


def _hailo_net_input_shapes_semantically_equal(left: Any, right: Any) -> bool:
    normalized_left = _normalize_hailo_net_input_shapes(left)
    normalized_right = _normalize_hailo_net_input_shapes(right)
    if normalized_left == normalized_right:
        return True
    if isinstance(normalized_left, dict) and len(normalized_left) == 1:
        return next(iter(normalized_left.values())) == normalized_right
    if isinstance(normalized_right, dict) and len(normalized_right) == 1:
        return next(iter(normalized_right.values())) == normalized_left
    return False


def _hailo_legacy_contract_translate_axes_match(
    contract: Any,
    *,
    net_name: str,
    net_input_shapes: Any,
    disable_rt_metadata_extraction: bool,
) -> bool:
    """Reject legacy records that contradict a requested translate identity.

    Early v2.75 ArtifactStore records did not always persist these axes.  Their
    absence therefore remains eligible for the narrowly defined historical
    defaults, but an axis that *is* present must agree exactly.  This prevents
    a coarse historical contract from overriding the stronger v3 request.
    """

    if not isinstance(contract, Mapping):
        return True
    candidates: list[Mapping[str, Any]] = [contract]
    for key in ("build_contract", "compiler", "hailo", "translate"):
        nested = contract.get(key)
        if isinstance(nested, Mapping):
            candidates.append(nested)
    expected_name = str(net_name).strip()
    for candidate in candidates:
        if "net_name" in candidate:
            actual_name = str(candidate.get("net_name") or "").strip()
            if not actual_name or actual_name != expected_name:
                return False
        if "net_input_shapes" in candidate:
            try:
                if not _hailo_net_input_shapes_semantically_equal(
                    candidate.get("net_input_shapes"), net_input_shapes,
                ):
                    return False
            except ValueError:
                return False
        if "disable_rt_metadata_extraction" in candidate:
            actual_rt_flag = candidate.get(
                "disable_rt_metadata_extraction"
            )
            if (
                type(actual_rt_flag) is not bool
                or actual_rt_flag is not bool(
                    disable_rt_metadata_extraction
                )
            ):
                return False
    return True


def _hailo_cache_key(
    *,
    model_path: Path,
    activation_part1: Path | None,
    hw_arch: str,
    opt_level: int,
    calib_dir: Path | None,
    calib_count: int,
    calib_batch_size: int,
    extra_model_script: str,
    start_nodes: Sequence[str] | None,
    end_nodes: Sequence[str] | None,
    preprocessing_contract: Mapping[str, Any] | None = None,
    effective_calib_count: int | None = None,
    calibration_storage: str | None = None,
    calibration_memory_cap_bytes: int | None = None,
    net_name: str | None = None,
    net_input_shapes: Any = None,
    disable_rt_metadata_extraction: bool | None = None,
    hailo_sdk_version_token: str | None = None,
) -> tuple[str, dict[str, Any]]:
    strict = str(os.environ.get('ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY') or 'relaxed').lower() == 'strict'
    calibration_identity = _calibration_identity(calib_dir, strict=strict)
    preprocessing_sha = (
        preprocessing_contract_sha256(preprocessing_contract)
        if preprocessing_contract
        else 'unsealed-legacy-call'
    )
    prepared_calibration_identity = hashlib.sha256(
        json.dumps(
            {
                'calibration_identity': calibration_identity,
                'preprocessing_contract_sha256': preprocessing_sha,
            },
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        ).encode('utf-8')
    ).hexdigest()
    requested_count = max(1, int(calib_count))
    effective_count = (
        max(1, int(effective_calib_count))
        if effective_calib_count is not None else requested_count
    )
    if effective_count > requested_count:
        raise ValueError(
            "effective Hailo calibration count cannot exceed requested count"
        )
    storage = str(
        calibration_storage
        or _resolve_hailo_calibration_storage(requested_count)
    ).strip().lower()
    if storage not in {"memory", "memmap"}:
        raise ValueError(f"unsupported_hailo_calibration_storage:{storage}")
    memory_cap_bytes = int(
        calibration_memory_cap_bytes
        if calibration_memory_cap_bytes is not None
        else _calibration_memory_cap_bytes()
    )
    if memory_cap_bytes <= 0:
        raise ValueError("hailo calibration memory cap must be positive")
    semantic_v3 = (
        net_name is not None
        and disable_rt_metadata_extraction is not None
    )
    payload = {
        'schema': (
            _HAILO_HEF_CACHE_SCHEMA_V3
            if semantic_v3 else _HAILO_HEF_CACHE_SCHEMA_V2
        ),
        'model_sha256': _bare_file_sha256(model_path),
        'activation_part1_sha256': _bare_file_sha256(activation_part1) if activation_part1 and activation_part1.is_file() else '',
        'hw_arch': str(hw_arch),
        'hailo_sdk_version': str(
            hailo_sdk_version_token or _hailo_sdk_version_token()
        ),
        'optimization_level': int(opt_level),
        'calibration_identity': calibration_identity,
        'prepared_calibration_identity_sha256': prepared_calibration_identity,
        'calibration_count': effective_count,
        'requested_calibration_count': requested_count,
        'calibration_storage': storage,
        'calibration_memory_cap_bytes': memory_cap_bytes,
        'calibration_batch_size': int(calib_batch_size),
        'extra_model_script': str(extra_model_script or ''),
        'start_nodes': list(start_nodes or []),
        'end_nodes': list(end_nodes or []),
        'integrity': 'strict' if strict else 'relaxed',
        'preprocessing_contract': dict(preprocessing_contract or {}),
        'preprocessing_contract_sha256': preprocessing_sha,
    }
    if semantic_v3:
        normalized_net_name = str(net_name or "").strip()
        if not normalized_net_name:
            raise ValueError("hailo net_name must be non-empty")
        payload.update({
            'net_name': normalized_net_name,
            'net_input_shapes': _normalize_hailo_net_input_shapes(
                net_input_shapes
            ),
            'disable_rt_metadata_extraction': bool(
                disable_rt_metadata_extraction
            ),
        })
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
    return key, payload


def _hailo_cache_root() -> Path:
    return Path(os.path.expanduser(os.environ.get('ONNX_SPLITPOINT_HAILO_CACHE_ROOT') or '~/.cache/onnx_splitpoint/hailo_hef')).resolve()


def _migrate_hailo_receipt_to_cache_contract(
    receipt: Mapping[str, Any], *, cache_key: str,
    cache_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reseal a verified v2 receipt under a stronger version-neutral key."""
    migrated = dict(receipt)
    old_key = str(migrated.get("cache_key") or "")
    migrated["cache_key"] = str(cache_key)
    migrated["cache_payload"] = dict(cache_payload)
    if old_key and old_key != str(cache_key):
        migrated["migrated_from_cache_key"] = old_key
    return migrated


def _load_migrated_hailo_v2_receipt(
    hef_path: Path, *, legacy_cache_key: str,
    legacy_cache_payload: Mapping[str, Any], cache_key: str,
    cache_payload: Mapping[str, Any], preprocessing_sha256: str,
    source_onnx_sha256: str, net_name: str,
    net_input_shapes: Any,
    disable_rt_metadata_extraction: bool,
    allow_legacy_v2: bool,
) -> Optional[Dict[str, Any]]:
    if not allow_legacy_v2:
        return None
    receipt = _load_valid_hailo_receipt(
        hef_path,
        preprocessing_sha256=preprocessing_sha256,
        source_onnx_sha256=source_onnx_sha256,
        cache_key=legacy_cache_key,
        cache_payload=legacy_cache_payload,
        expected_net_name=net_name,
        expected_net_input_shapes=net_input_shapes,
        expected_disable_rt_metadata_extraction=(
            disable_rt_metadata_extraction
        ),
        allow_legacy_v2=True,
    )
    if receipt is None:
        return None
    migrated = _migrate_hailo_receipt_to_cache_contract(
        receipt, cache_key=cache_key, cache_payload=cache_payload,
    )
    return migrated


def _backfill_hailo_exact_cache(
    *,
    cache_dir: Path,
    hef_path: Path,
    receipt: Mapping[str, Any],
    cache_key: str,
    cache_payload: Mapping[str, Any],
    preprocessing_sha256: str,
    source_onnx_sha256: str,
    net_name: str,
    hw_arch: str,
) -> bool:
    """Atomically back up all three files under the existing exact cache key."""
    try:
        if (
            str(receipt.get("cache_key") or "") != str(cache_key)
            or receipt.get("cache_payload") != dict(cache_payload)
            or str(receipt.get("preprocessing_contract_sha256") or "") != str(preprocessing_sha256)
            or str(receipt.get("source_onnx_sha256") or "") != str(source_onnx_sha256)
            or str(receipt.get("net_name") or "") != str(net_name)
            or str(receipt.get("hw_arch") or "") != str(hw_arch)
        ):
            raise ValueError("hailo_cache_backfill_identity_mismatch")
        _publish_hailo_bundle(
            source_hef=hef_path, destination=cache_dir / "compiled.hef",
            receipt=receipt, source="artifact_store_exact_v2_restore",
        )
        return True
    except Exception as exc:
        log.warning("[hailo][cache] atomic bundle backup failed: %s", exc)
        return False


def _restore_hailo_v2_artifact_store_exact(
    *,
    destination: Path,
    cache_dir: Path | None,
    cache_key: str,
    cache_payload: Mapping[str, Any],
    preprocessing_sha256: str,
    source_onnx_sha256: str,
    net_name: str,
    hw_arch: str,
    legacy_cache_key: str = "",
    legacy_cache_payload: Mapping[str, Any] | None = None,
    allow_legacy_v2: bool = False,
    diagnostics: Optional[Dict[str, Any]] = None,
    read_only: bool = False,
) -> Optional[Dict[str, Any]]:
    """Restore a byte- and build-identical ArtifactStore HEF.

    The historical unified-library contract is intentionally not consulted:
    it was computed before ONNX fixup, effective calibration clamping and SDK
    resolution.  Existing v2.75 records remain addressable through the exact
    ``legacy_cache_key`` plus their embedded canonical v2 receipt.
    """

    try:
        from .artifact_store import ArtifactStore, artifact_store_enabled

        if not artifact_store_enabled():
            return None
        store = ArtifactStore(
            os.environ.get("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT") or None,
            read_only=bool(read_only),
        )
        exact_candidates: List[Tuple[Any, Dict[str, Any], bool]] = []
        audit = diagnostics if diagnostics is not None else {}
        audit.update(matched_candidates=0, valid_candidates=0, rejected_candidates=0,
                     legacy_unsealed_candidates=0,
                     status="missing", selection_order="artifact_id_ascending")
        for record in store.list(kind="hailo_hef", limit=1_000_000):
            try:
                metadata = record.metadata if isinstance(record.metadata, dict) else {}
                record_key = str(metadata.get("legacy_cache_key") or "")
                candidate_is_legacy = bool(
                    allow_legacy_v2
                    and legacy_cache_key
                    and record_key == str(legacy_cache_key)
                )
                expected_key = (
                    str(legacy_cache_key) if candidate_is_legacy
                    else str(cache_key)
                )
                expected_payload = (
                    dict(legacy_cache_payload or {})
                    if candidate_is_legacy else dict(cache_payload)
                )
                if record_key != expected_key:
                    continue
                audit["matched_candidates"] += 1
                if (
                    candidate_is_legacy
                    and not _hailo_legacy_contract_translate_axes_match(
                        record.contract,
                        net_name=net_name,
                        net_input_shapes=cache_payload.get(
                            "net_input_shapes"
                        ),
                        disable_rt_metadata_extraction=bool(
                            cache_payload.get(
                                "disable_rt_metadata_extraction"
                            )
                        ),
                    )
                ):
                    continue
                receipt = metadata.get("build_receipt")
                if not isinstance(receipt, dict):
                    audit["legacy_unsealed_candidates"] += 1
                    continue
                if receipt.get("schema") != _HAILO_HEF_RECEIPT_SCHEMA:
                    continue
                if str(receipt.get("cache_key") or "") != expected_key:
                    continue
                if receipt.get("cache_payload") != expected_payload:
                    continue
                if str(receipt.get("net_name") or "").strip() != str(
                    net_name
                ).strip():
                    continue
                if str(receipt.get("hw_arch") or "") != str(hw_arch):
                    continue
                if str(receipt.get("source_onnx_sha256") or "") != str(
                    source_onnx_sha256
                ):
                    continue
                if str(receipt.get("preprocessing_contract_sha256") or "") != str(
                    preprocessing_sha256
                ):
                    continue
                if str(receipt.get("hef_sha256") or "") != str(record.artifact_hash):
                    continue
                object_path = Path(record.object_path)
                if (
                    not object_path.is_file()
                    or int(object_path.stat().st_size) != int(record.size_bytes)
                    or _bare_file_sha256(object_path) != str(record.artifact_hash)
                ):
                    continue
                # Fully validate every candidate before conflict detection or
                # ordering. A corrupt low-id duplicate must not mask a valid one.
                if metadata.get("bundle_status") == "sealed":
                    valid_record, _reason = store.validate_record(record, verify="strict")
                    if not valid_record:
                        continue
                validated = _load_valid_hailo_receipt(
                    object_path, preprocessing_sha256=preprocessing_sha256,
                    source_onnx_sha256=source_onnx_sha256,
                    cache_key=expected_key, cache_payload=expected_payload,
                    expected_net_name=net_name,
                    expected_net_input_shapes=cache_payload.get("net_input_shapes"),
                    expected_disable_rt_metadata_extraction=cache_payload.get("disable_rt_metadata_extraction"),
                    allow_legacy_v2=candidate_is_legacy,
                    receipt_override=receipt, validate_cache_meta=False,
                )
                if validated is None:
                    continue
                exact_candidates.append((record, dict(validated), candidate_is_legacy))
            except Exception as candidate_error:
                log.warning("[hailo][artifact-store] invalid candidate id=%s: %s",
                            getattr(record, "artifact_id", "?"), candidate_error)
                continue
        audit["valid_candidates"] = len(exact_candidates)
        audit["rejected_candidates"] = audit["matched_candidates"] - len(exact_candidates)

        if any(not is_legacy for _record, _receipt, is_legacy in exact_candidates):
            exact_candidates = [
                item for item in exact_candidates if not item[2]
            ]
        identities = {
            (
                str(record.artifact_hash),
                json.dumps(
                    {
                        key: value for key, value in receipt.items()
                        if key not in {
                            "created_at_unix_s",
                            "migrated_from_cache_key",
                        }
                    },
                    sort_keys=True, separators=(",", ":"),
                ),
            )
            for record, receipt, _is_legacy in exact_candidates
        }
        if len(identities) > 1:
            audit["status"] = "conflicting_valid_duplicates"
            audit["conflicting_artifact_ids"] = sorted(int(item[0].artifact_id) for item in exact_candidates)
            log.warning(
                "[hailo][artifact-store] conflicting exact records for key=%s",
                cache_key[:12],
            )
            return None
        if not exact_candidates:
            if audit["matched_candidates"]:
                audit["status"] = (
                    "legacy_unsealed"
                    if audit["legacy_unsealed_candidates"] == audit["matched_candidates"]
                    else "all_candidates_invalid"
                )
            return None
        record, receipt, candidate_is_legacy = sorted(
            exact_candidates, key=lambda item: int(item[0].artifact_id),
        )[0]
        if candidate_is_legacy:
            receipt = _migrate_hailo_receipt_to_cache_contract(
                receipt, cache_key=cache_key, cache_payload=cache_payload,
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".hailo-artifact-restore-", dir=str(destination.parent)
        ) as temporary_dir:
            staged_hef = Path(temporary_dir) / "compiled.hef"
            if read_only:
                shutil.copy2(Path(record.object_path), staged_hef)
                materialize_method = "copy_read_only_probe"
            elif dict(record.metadata or {}).get("bundle_status") != "sealed":
                # The complete historical receipt was validated above; the
                # generic store intentionally refuses HEF-only materialization.
                shutil.copy2(Path(record.object_path), staged_hef)
                materialize_method = "copy_verified_legacy_receipt"
            else:
                materialize_method = store.materialize(
                    record, staged_hef, reference=f"hailo-exact-v2:{cache_key}",
                )
            committed_hef = _publish_hailo_bundle(
                source_hef=staged_hef, destination=destination, receipt=receipt,
                source="artifact_store_exact_v2_restore",
            )
        # Publication already validated the immutable snapshot. Do not reread
        # the public pointer, which another successful publisher may advance.
        validated_receipt = dict(receipt)
        audit.update(status="selected", selected_artifact_id=int(record.artifact_id))
        cache_backfilled = bool(
            not read_only and cache_dir is not None
            and _backfill_hailo_exact_cache(
                cache_dir=cache_dir,
                hef_path=committed_hef,
                receipt=validated_receipt,
                cache_key=cache_key,
                cache_payload=cache_payload,
                preprocessing_sha256=preprocessing_sha256,
                source_onnx_sha256=source_onnx_sha256,
                net_name=net_name,
                hw_arch=hw_arch,
            )
        )
        restored = {
            "artifact_id": int(record.artifact_id),
            "contract_hash": str(record.contract_hash),
            "artifact_hash": str(record.artifact_hash),
            "source": str(record.object_path),
            "destination": str(committed_hef),
            "materialize_method": materialize_method,
            "cache_key": str(cache_key),
            "cache_backfilled": cache_backfilled,
            "build_receipt": validated_receipt,
            "duplicate_selection": dict(audit),
        }
        try:
            _atomic_write_json(
                destination.parent / "artifact_store_restore.json", restored
            )
        except Exception:
            pass
        return restored
    except Exception as exc:
        log.warning("[hailo][artifact-store] exact v2 restore failed: %s", exc)
        return None


def _inspect_hailo_diagnostic_hef(path: Path) -> Dict[str, Any]:
    """Read the compiled container through the installed HailoRT API only.

    A missing reader is explicit evidence, never a synthetic HEF-header PASS.
    No accelerator is opened by these metadata queries.
    """
    try:
        from hailo_platform import HEF
    except ImportError as exc:
        return {"status": "unavailable", "reader": "hailo_platform.HEF", "error": str(exc)}
    try:
        container = HEF(str(path))
        inputs = list(container.get_input_vstream_infos())
        outputs = list(container.get_output_vstream_infos())
        if not inputs or not outputs:
            raise ValueError("HEF has no readable input or output stream metadata")
        return {"status": "passed", "reader": "hailo_platform.HEF",
                "input_count": len(inputs), "output_count": len(outputs),
                "inputs": [str(v.name) for v in inputs], "outputs": [str(v.name) for v in outputs]}
    except Exception as exc:
        return {"status": "failed", "reader": "hailo_platform.HEF", "error": f"{type(exc).__name__}: {exc}"}


def _hailo_build_hef_legacy(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    disable_rt_metadata_extraction: bool = True,
    opt_level: int = 1,
    calib_dir: Optional[Union[str, Path]] = None,
    calib_count: int = 64,
    calib_batch_size: int = 8,
    activation_part1_onnx: Optional[Union[str, Path]] = None,
    activation_gen_batch: int = 8,
    force: bool = False,
    cache_only: bool = False,
    keep_artifacts: bool = False,
    publish_artifacts: bool = True,
    extra_model_script: Optional[str] = None,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    preprocessing_contract: Optional[Union[Mapping[str, Any], str]] = None,
    sdk_version_token: Optional[str] = None,
    read_only_cache_probe: Optional[bool] = None,
    negative_evidence_identity_authoritative: bool = True,
) -> HailoHefBuildResult:
    """Translate + optimize + compile an ONNX to a HEF.

    Notes
    -----
    - This function requires `hailo_sdk_client` (DFC) to be importable.
    - For single-input networks, `calib_dir` can point to image/input `.npy` / `.npz` samples.
- For multi-input split stage2 networks, pass `activation_part1_onnx` together with
  `calib_dir` containing image calibration samples. The tool will run Part1 with
  ONNXRuntime to generate activation calibration for Part2 (splitbench-style).
- If calibration data cannot be used, a *random* calibration set is generated based
  on HN input-layer shapes.
    """

    force = parse_config_bool(force, field="hailo_build.force_build")
    publish_artifacts = parse_config_bool(publish_artifacts, field="hailo_build.publish_artifacts")
    if not publish_artifacts:
        _validate_hailo_diagnostic_environment(outdir)
    t0 = time.time()
    policy_cache_verify = compiler_dispatch_forbidden()
    read_only_probe = bool(
        policy_cache_verify or (cache_only if read_only_cache_probe is None else read_only_cache_probe)
    )
    cache_only = bool(cache_only or policy_cache_verify)
    onnx_path = Path(onnx_path)
    net_name = str(net_name or onnx_path.stem).strip()
    if not net_name:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name="",
            backend="local",
            failure_kind="invalid_build_contract",
            error="Hailo net_name must be non-empty after normalization",
            last_stage="cache_contract",
            timed_out=False,
        )

    if bool(cache_only) and bool(force):
        return _make_hef_result(
            ok=False,
            elapsed_s=0.0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            skipped=True,
            failure_kind="cache_miss_blocked",
            unsupported_reason="cache_verify_only_force_conflict",
            error=(
                "cache_miss_blocked[hailo_dfc]: cache_only and force are "
                "mutually exclusive; DFC dispatch was not started"
            ),
            last_stage="cache_lookup",
            timed_out=False,
        )

    hw_arch_eff = _normalize_hailo_hw_arch(hw_arch)
    if hw_arch_eff != str(hw_arch or "").strip().lower():
        log.warning("[hailo] hw_arch alias: '%s' -> '%s'", str(hw_arch), hw_arch_eff)

    out_dir = Path(outdir) if outdir is not None else onnx_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # A new probe/recipe supersedes the prior run-local negative diagnostic.
    # The persistent evidence index is append-only and remains untouched.
    prior_negative = out_dir / "hailo_negative_evidence.json"
    if prior_negative.is_file():
        _atomic_write_json(prior_negative, {
            "net_name": net_name, "hw_arch": hw_arch_eff,
            "negative_evidence_hit": False,
            "build_evidence": {"status": "UNAVAILABLE", "reusable": False,
                               "reason": "superseded_by_current_probe"},
        })

    activation_part1_onnx_p = Path(activation_part1_onnx).expanduser().resolve() if activation_part1_onnx else None
    calib_dir_p = Path(calib_dir).expanduser().resolve() if calib_dir else None

    try:
        preprocessing_contract_eff, preprocessing_sha256 = _resolve_hailo_image_contract(
            model_path=onnx_path.expanduser().resolve(),
            activation_part1=activation_part1_onnx_p,
            net_input_shapes=net_input_shapes,
            task=task,
            declared=preprocessing_contract,
        )
    except Exception as exc:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend="local",
            error=f"Invalid Hailo image preprocessing contract: {type(exc).__name__}: {exc}",
            failure_kind="invalid_preprocessing_contract",
            last_stage="preprocessing_contract_preflight",
            timed_out=False,
        )

    if activation_part1_onnx_p is not None:
        preflight = _activation_calib_preflight(part1_onnx=activation_part1_onnx_p, part2_onnx=onnx_path)
        if preflight.get('inspect_ok') and preflight.get('compatible') is False:
            return _make_hef_result(
                ok=False,
                elapsed_s=time.time() - t0,
                hw_arch=str(hw_arch_eff),
                net_name=str(net_name),
                backend='local',
                skipped=True,
                failure_kind='unsupported_splitpoint',
                unsupported_reason='activation_preflight_missing_inputs',
                error=_format_activation_calib_preflight_error(preflight),
                calib_info={
                    'source': 'activation_from_part1',
                    'requested_count': int(calib_count),
                    'preflight': preflight,
                },
            )

    hef_path = out_dir / "compiled.hef"

    fixed_path: Optional[Path] = None
    fixup_report: Optional[Dict[str, Any]] = None
    model_for_parse = onnx_path
    if fixup:
        try:
            m = onnx.load(str(onnx_path))
            m2, rep = fix_onnx_for_hailo(m, add_conv_defaults=add_conv_defaults)
            fixup_report = rep
            fixed_path = out_dir / (onnx_path.stem + "_hailo_fixed.onnx")
            onnx.save(m2, str(fixed_path))
            model_for_parse = fixed_path
        except Exception as e:
            fixup_report = {"error": str(e)}
            model_for_parse = onnx_path
            fixed_path = None

    inferred_default_net_input_shapes = None
    try:
        m_tmp = onnx.load(str(model_for_parse))
        inferred_default_net_input_shapes = infer_net_input_shapes_from_model(
            m_tmp
        )
        if net_input_shapes is None:
            net_input_shapes = inferred_default_net_input_shapes
    except Exception:
        if net_input_shapes is None:
            net_input_shapes = None

    try:
        calibration_storage = _resolve_hailo_calibration_storage(
            int(calib_count)
        )
        calibration_memory_cap = _calibration_memory_cap_bytes()
        calibration_identity_shapes = _hailo_calibration_shape_candidates(
            net_input_shapes, preprocessing_contract_eff,
        )
        effective_calib_count = _effective_hailo_calibration_count(
            requested=int(calib_count),
            shapes=calibration_identity_shapes,
            storage=calibration_storage,
            cap_bytes=calibration_memory_cap,
        )
        memory_limited_calib_count = int(effective_calib_count)
        (
            calibration_available_count,
            calibration_available_count_source,
        ) = _hailo_authoritative_calibration_sample_count(calib_dir_p)
        if calibration_available_count is not None:
            effective_calib_count = min(
                int(effective_calib_count),
                int(calibration_available_count),
            )
    except Exception as calibration_identity_exc:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend="local",
            failure_kind="invalid_build_contract",
            error=(
                "Unable to seal Hailo calibration count identity: "
                f"{type(calibration_identity_exc).__name__}: "
                f"{calibration_identity_exc}"
            ),
            last_stage="calibration_count_contract",
            timed_out=False,
        )

    # v60o: content-addressed HEF build cache.  Building Hailo artifacts
    # dominates repeated development runs, so identical model/configuration
    # requests reuse the compiled HEF.  Final mode uses strict calibration
    # content identity; relaxed modes use path/size/mtime identity.
    cache_enabled = publish_artifacts and str(os.environ.get('ONNX_SPLITPOINT_HAILO_CACHE_ENABLED', '1')).strip().lower() not in {'0', 'false', 'no', 'off'}
    cache_key = ''
    cache_payload: dict[str, Any] = {}
    legacy_cache_key = ''
    legacy_cache_payload: dict[str, Any] = {}
    cache_dir: Path | None = None
    cache_lookup_error = ""
    try:
        cache_key, cache_payload = _hailo_cache_key(
            model_path=model_for_parse,
            activation_part1=activation_part1_onnx_p,
            hw_arch=str(hw_arch_eff),
            opt_level=int(opt_level),
            calib_dir=calib_dir_p,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            extra_model_script=str(extra_model_script or ''),
            start_nodes=start_node_names,
            end_nodes=end_node_names,
            preprocessing_contract=preprocessing_contract_eff,
            effective_calib_count=effective_calib_count,
            calibration_storage=calibration_storage,
            calibration_memory_cap_bytes=calibration_memory_cap,
            net_name=str(net_name),
            net_input_shapes=net_input_shapes,
            disable_rt_metadata_extraction=bool(
                disable_rt_metadata_extraction
            ),
            hailo_sdk_version_token=sdk_version_token,
        )
        legacy_cache_key, legacy_cache_payload = _hailo_cache_key(
            model_path=model_for_parse,
            activation_part1=activation_part1_onnx_p,
            hw_arch=str(hw_arch_eff),
            opt_level=int(opt_level),
            calib_dir=calib_dir_p,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            extra_model_script=str(extra_model_script or ''),
            start_nodes=start_node_names,
            end_nodes=end_node_names,
            preprocessing_contract=preprocessing_contract_eff,
            effective_calib_count=effective_calib_count,
            calibration_storage=calibration_storage,
            calibration_memory_cap_bytes=calibration_memory_cap,
            hailo_sdk_version_token=sdk_version_token,
        )
    except Exception as cache_exc:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend='local',
            failure_kind='invalid_build_contract',
            error=f'Unable to seal Hailo cache/build contract: {type(cache_exc).__name__}: {cache_exc}',
            last_stage='cache_contract',
            timed_out=False,
        )

    # Force bypasses lookup, not backup. Establish the destination independently
    # so forced successful builds also seal and retain the complete cache tuple.
    if cache_enabled:
        cache_dir = _hailo_cache_root() / cache_key

    try:
        legacy_v2_eligible = bool(
            disable_rt_metadata_extraction
            and _hailo_net_input_shapes_semantically_equal(
                net_input_shapes, inferred_default_net_input_shapes,
            )
        )
    except ValueError:
        legacy_v2_eligible = False

    artifact_store_diagnostics: Dict[str, Any] = {}
    source_onnx_sha256 = _bare_file_sha256(onnx_path)
    if hef_path.is_file() and not bool(force):
        existing_hef = hef_path.resolve()
        existing_receipt = _load_valid_hailo_receipt(
            existing_hef,
            preprocessing_sha256=preprocessing_sha256,
            source_onnx_sha256=source_onnx_sha256,
            cache_key=cache_key,
            cache_payload=cache_payload,
            expected_net_name=str(net_name),
            expected_net_input_shapes=net_input_shapes,
            expected_disable_rt_metadata_extraction=bool(
                disable_rt_metadata_extraction
            ),
        )
        if existing_receipt is None:
            existing_receipt = _load_migrated_hailo_v2_receipt(
                existing_hef,
                legacy_cache_key=legacy_cache_key,
                legacy_cache_payload=legacy_cache_payload,
                cache_key=cache_key,
                cache_payload=cache_payload,
                preprocessing_sha256=preprocessing_sha256,
                source_onnx_sha256=source_onnx_sha256,
                net_name=str(net_name),
                net_input_shapes=net_input_shapes,
                disable_rt_metadata_extraction=bool(
                    disable_rt_metadata_extraction
                ),
                allow_legacy_v2=legacy_v2_eligible,
            )
        if existing_receipt is not None:
            if not read_only_probe:
                _publish_hailo_bundle(
                    source_hef=existing_hef, destination=hef_path,
                    receipt=existing_receipt, source="validated_existing_artifact",
                )
            return HailoHefBuildResult(
                ok=True,
                elapsed_s=time.time() - t0,
                hw_arch=str(hw_arch_eff),
                net_name=str(net_name),
                backend="local",
                hef_path=str(hef_path),
                skipped=True,
                calib_info={
                    "source": "validated_existing_artifact",
                    "compiler_dispatch_count": 0, "cache_hit": True,
                    "preprocessing_contract": preprocessing_contract_eff,
                    "preprocessing_contract_sha256": preprocessing_sha256,
                    "build_receipt": existing_receipt,
                },
            )
        log.warning(
            "[hailo][reuse] ignoring unsealed or stale compiled.hef at %s", hef_path
        )

    if cache_enabled and not bool(force):
        try:
            cache_dir = _hailo_cache_root() / cache_key
            cached_hef = (cache_dir / 'compiled.hef').resolve()
            cached_meta = _hailo_cache_meta_path(cached_hef)
            cached_receipt = _load_valid_hailo_receipt(
                cached_hef,
                preprocessing_sha256=preprocessing_sha256,
                source_onnx_sha256=source_onnx_sha256,
                cache_key=cache_key,
                cache_payload=cache_payload,
                expected_net_name=str(net_name),
                expected_net_input_shapes=net_input_shapes,
                expected_disable_rt_metadata_extraction=bool(
                    disable_rt_metadata_extraction
                ),
            )
            if cached_receipt is not None:
                _publish_hailo_bundle(
                    source_hef=cached_hef, destination=hef_path,
                    receipt=cached_receipt, source="local_exact_cache",
                )
                hit_payload = {
                    'compiler_dispatch_count': 0, 'cache_hit': True,
                    'cache_key': cache_key,
                    'cache_dir': str(cache_dir),
                    'requested_count': int(calib_count),
                    'payload': cache_payload,
                    'preprocessing_contract': preprocessing_contract_eff,
                    'preprocessing_contract_sha256': preprocessing_sha256,
                    'build_receipt': cached_receipt,
                }
                try:
                    if cached_meta.is_file():
                        hit_payload['cache_meta'] = json.loads(cached_meta.read_text(encoding='utf-8'))
                except Exception:
                    pass
                try:
                    (out_dir / 'hailo_cache_hit.json').write_text(json.dumps(hit_payload, indent=2), encoding='utf-8')
                except Exception:
                    pass
                print(
                    f"[hailo-cache] HIT role=hef model={net_name} "
                    f"identity={cache_key} reason=receipt_verified "
                    f"artifact={cached_hef} hw_arch={hw_arch_eff}",
                    flush=True,
                )
                return HailoHefBuildResult(
                    ok=True,
                    elapsed_s=time.time() - t0,
                    hw_arch=str(hw_arch_eff),
                    net_name=str(net_name),
                    backend='local',
                    hef_path=str(hef_path),
                    fixed_onnx_path=str(fixed_path) if fixed_path else None,
                    fixup_report=fixup_report,
                    skipped=True,
                    calib_info=hit_payload,
                    details={'compiler_dispatch_count': 0, 'cache_hit': True, 'cache_key': cache_key, 'cache_dir': str(cache_dir)},
                )
            legacy_cache_dir = _hailo_cache_root() / legacy_cache_key
            legacy_cached_hef = (legacy_cache_dir / 'compiled.hef').resolve()
            migrated_receipt = _load_migrated_hailo_v2_receipt(
                legacy_cached_hef,
                legacy_cache_key=legacy_cache_key,
                legacy_cache_payload=legacy_cache_payload,
                cache_key=cache_key,
                cache_payload=cache_payload,
                preprocessing_sha256=preprocessing_sha256,
                source_onnx_sha256=source_onnx_sha256,
                net_name=str(net_name),
                net_input_shapes=net_input_shapes,
                disable_rt_metadata_extraction=bool(
                    disable_rt_metadata_extraction
                ),
                allow_legacy_v2=legacy_v2_eligible,
            )
            if migrated_receipt is not None:
                committed_hef = _publish_hailo_bundle(
                    source_hef=legacy_cached_hef, destination=hef_path,
                    receipt=migrated_receipt, source="local_exact_v2_migrated",
                )
                cache_backfilled = not read_only_probe and _backfill_hailo_exact_cache(
                    cache_dir=cache_dir,
                    hef_path=committed_hef,
                    receipt=migrated_receipt,
                    cache_key=cache_key,
                    cache_payload=cache_payload,
                    preprocessing_sha256=preprocessing_sha256,
                    source_onnx_sha256=source_onnx_sha256,
                    net_name=str(net_name),
                    hw_arch=str(hw_arch_eff),
                )
                hit_payload = {
                    'compiler_dispatch_count': 0, 'cache_hit': True,
                    'cache_source': 'local_exact_v2_migrated',
                    'cache_key': cache_key,
                    'legacy_cache_key': legacy_cache_key,
                    'cache_dir': str(cache_dir),
                    'cache_backfilled': bool(cache_backfilled),
                    'requested_count': int(calib_count),
                    'payload': cache_payload,
                    'preprocessing_contract': preprocessing_contract_eff,
                    'preprocessing_contract_sha256': preprocessing_sha256,
                    'build_receipt': migrated_receipt,
                }
                print(
                    f"[hailo-cache] HIT role=hef model={net_name} "
                    f"identity={cache_key} reason=legacy_receipt_migrated "
                    f"artifact={legacy_cached_hef} hw_arch={hw_arch_eff}",
                    flush=True,
                )
                return HailoHefBuildResult(
                    ok=True,
                    elapsed_s=time.time() - t0,
                    hw_arch=str(hw_arch_eff),
                    net_name=str(net_name),
                    backend='local',
                    hef_path=str(hef_path),
                    fixed_onnx_path=str(fixed_path) if fixed_path else None,
                    fixup_report=fixup_report,
                    skipped=True,
                    calib_info=hit_payload,
                    details={
                        'compiler_dispatch_count': 0, 'cache_hit': True,
                        'cache_source': 'local_exact_v2_migrated',
                        'cache_key': cache_key,
                        'legacy_cache_key': legacy_cache_key,
                        'cache_dir': str(cache_dir),
                    },
                )
        except Exception as cache_exc:
            log.warning("[hailo][cache] lookup failed: %s", cache_exc)
            cache_lookup_error = f"{type(cache_exc).__name__}: {cache_exc}"
            cache_dir = None

    if publish_artifacts and not bool(force):
        exact_cache_dir = cache_dir
        if exact_cache_dir is None and cache_enabled:
            try:
                exact_cache_dir = _hailo_cache_root() / cache_key
            except Exception:
                exact_cache_dir = None
        restored = _restore_hailo_v2_artifact_store_exact(
            destination=hef_path,
            cache_dir=exact_cache_dir,
            cache_key=cache_key,
            cache_payload=cache_payload,
            preprocessing_sha256=preprocessing_sha256,
            source_onnx_sha256=source_onnx_sha256,
            net_name=str(net_name),
            hw_arch=str(hw_arch_eff),
            legacy_cache_key=legacy_cache_key,
            legacy_cache_payload=legacy_cache_payload,
            allow_legacy_v2=legacy_v2_eligible,
            diagnostics=artifact_store_diagnostics, read_only=read_only_probe,
        )
        if restored is not None:
            restored_receipt = dict(restored.get("build_receipt") or {})
            hit_payload = {
                "source": "artifact_store_exact_v2",
                "compiler_dispatch_count": 0, "cache_hit": True,
                "cache_source": "artifact_store_exact_v2",
                "cache_key": cache_key,
                "cache_dir": str(exact_cache_dir or ""),
                "cache_backfilled": bool(restored.get("cache_backfilled")),
                "requested_count": int(calib_count),
                "payload": cache_payload,
                "preprocessing_contract": preprocessing_contract_eff,
                "preprocessing_contract_sha256": preprocessing_sha256,
                "artifact_id": restored.get("artifact_id"),
                "contract_hash": restored.get("contract_hash"),
                "artifact_hash": restored.get("artifact_hash"),
                "build_receipt": restored_receipt,
            }
            try:
                _atomic_write_json(out_dir / "hailo_cache_hit.json", hit_payload)
            except Exception:
                pass
            print(
                f"[hailo-cache] HIT role=hef model={net_name} "
                f"identity={cache_key} reason=artifact_store_receipt_verified "
                f"artifact={hef_path} hw_arch={hw_arch_eff}",
                flush=True,
            )
            return HailoHefBuildResult(
                ok=True,
                elapsed_s=time.time() - t0,
                hw_arch=str(hw_arch_eff),
                net_name=str(net_name),
                backend="artifact_store",
                hef_path=str(hef_path),
                fixed_onnx_path=str(fixed_path) if fixed_path else None,
                fixup_report=fixup_report,
                skipped=True,
                calib_info=hit_payload,
                details={
                    "compiler_dispatch_count": 0, "cache_hit": True,
                    "cache_source": "artifact_store_exact_v2",
                    "cache_key": cache_key,
                    "cache_dir": str(exact_cache_dir or ""),
                    "artifact_store_restore": dict(restored),
                },
            )

    bundle_statuses = {
        "destination": _hailo_cache_bundle_status(hef_path),
        "exact_v3": _hailo_cache_bundle_status(_hailo_cache_root() / cache_key / "compiled.hef"),
        "legacy_v2": _hailo_cache_bundle_status(_hailo_cache_root() / legacy_cache_key / "compiled.hef"),
    }
    cache_miss_reason = (
        "force_rebuild_requested" if bool(force)
        else "cache_disabled" if not cache_enabled
        else str(artifact_store_diagnostics.get("status"))
        if artifact_store_diagnostics.get("status") in {"conflicting_valid_duplicates", "all_candidates_invalid", "legacy_unsealed"}
        else "legacy_unsealed"
        if any(item.get("status") == "legacy_unsealed" for item in bundle_statuses.values())
        else "not_found_or_receipt_invalid"
    )
    print(
        f"[hailo-cache] MISS role=hef model={net_name} "
        f"identity={cache_key} reason={cache_miss_reason} "
        f"artifact={hef_path} hw_arch={hw_arch_eff}",
        flush=True,
    )

    # A verified positive HEF above always wins.  Negative evidence uses the
    # same exact v3 payload, including compiler/recipe/calibration identity.
    # force=True is used by automatic retries too and must not bypass it.
    from .hailo_negative_evidence import lookup_before_compile
    negative_info = lookup_before_compile(
        cache_payload=cache_payload, cache_key=cache_key,
        source_onnx=onnx_path, net_name=net_name, hw_arch=hw_arch_eff,
        force=bool(force),
        authoritative_sdk=bool(negative_evidence_identity_authoritative),
        publish_artifacts=publish_artifacts,
    )
    if negative_info.get("compiler_dispatch_allowed") is False:
        negative_hit = bool(negative_info.get("negative_evidence_hit"))
        info = {
            "cache_hit": False, "cache_only": bool(cache_only),
            "negative_evidence_hit": negative_hit,
            "build_evidence": negative_info, "cache_key": cache_key,
            "payload": cache_payload, "net_name": net_name, "hw_arch": hw_arch_eff,
            "reason": str(negative_info.get("state") or negative_info.get("reason") or ""),
            "compiler_dispatch_allowed": False,
            "compiler_dispatch_count": 0,
        }
        try:
            _atomic_write_json(out_dir / "hailo_negative_evidence.json", info)
        except Exception as exc:
            log.warning("[build-evidence] failed to write run diagnostic: %s", exc)
        return _make_hef_result(
            ok=False, elapsed_s=time.time() - t0, hw_arch=str(hw_arch_eff),
            net_name=str(net_name), backend="local", skipped=True,
            failure_kind=("known_negative_build_evidence" if negative_hit else
                          "build_evidence_" + str(negative_info.get("status") or "error").lower()),
            unsupported_reason=str(negative_info.get("state") or "") if negative_hit else None,
            error=(f"Exact build evidence prevents repeated compiler attempt: "
                   f"{negative_info.get('state') or negative_info.get('status')} "
                   f"({negative_info.get('reason')})"),
            last_stage="build_evidence_lookup", timed_out=False,
            calib_info=info, details=info,
            fixed_onnx_path=str(fixed_path) if fixed_path else None,
            fixup_report=fixup_report,
        )

    if bool(cache_only):
        cache_root_text = ""
        exact_cache_dir_text = str(cache_dir or "")
        legacy_cache_dir_text = ""
        exact_cache_hef_present = False
        exact_cache_receipt_present = False
        legacy_cache_hef_present = False
        legacy_cache_receipt_present = False
        try:
            cache_root = _hailo_cache_root()
            cache_root_text = str(cache_root)
            exact_cache_dir = cache_dir or (cache_root / cache_key)
            legacy_cache_dir = cache_root / legacy_cache_key
            exact_cache_dir_text = str(exact_cache_dir)
            legacy_cache_dir_text = str(legacy_cache_dir)
            exact_cached_hef = exact_cache_dir / "compiled.hef"
            legacy_cached_hef = legacy_cache_dir / "compiled.hef"
            exact_cache_hef_present = exact_cached_hef.is_file()
            exact_cache_receipt_present = _hailo_receipt_path(
                exact_cached_hef
            ).is_file()
            legacy_cache_hef_present = legacy_cached_hef.is_file()
            legacy_cache_receipt_present = _hailo_receipt_path(
                legacy_cached_hef
            ).is_file()
        except Exception as diagnostic_exc:
            if not cache_lookup_error:
                cache_lookup_error = (
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )

        diagnostic_path = out_dir / "hailo_cache_miss.json"
        diagnostic = {
            "schema": "onnx-splitpoint/hailo-cache-miss/v1",
            "status": (
                "cache_miss_blocked" if policy_cache_verify else "cache_miss"
            ),
            "artifact_kind": "hailo_hef",
            "reason": cache_miss_reason,
            "bundle_statuses": bundle_statuses,
            "artifact_store_candidates": artifact_store_diagnostics,
            "net_name": str(net_name),
            "hw_arch": str(hw_arch_eff),
            "cache_key_v3": cache_key,
            "cache_payload_v3": cache_payload,
            "cache_key_v2": legacy_cache_key,
            "cache_payload_v2": legacy_cache_payload,
            "cache_root": cache_root_text,
            "probe_outcomes": {
                "destination_hef_present": hef_path.is_file(),
                "destination_receipt_present": _hailo_receipt_path(
                    hef_path
                ).is_file(),
                "exact_v3_cache_dir": exact_cache_dir_text,
                "exact_v3_hef_present": exact_cache_hef_present,
                "exact_v3_receipt_present": exact_cache_receipt_present,
                "legacy_v2_cache_dir": legacy_cache_dir_text,
                "legacy_v2_hef_present": legacy_cache_hef_present,
                "legacy_v2_receipt_present": legacy_cache_receipt_present,
                "artifact_store_restored": False,
                "cache_lookup_error": cache_lookup_error,
            },
            "compiler_dispatch_allowed": False,
            # Execution provenance only: copied from the effective prepared
            # builder contract, never added to either cache identity.
            "workspace_contract": {
                "out_dir": str(out_dir.resolve()),
                "effective_calibration_count": int(effective_calib_count),
                "calibration_identity_shapes": calibration_identity_shapes,
                "calibration_storage": calibration_storage,
                "source": "prepared_hailo_build_contract",
            },
        }
        try:
            _atomic_write_json(diagnostic_path, diagnostic)
            diagnostic_path_text = str(diagnostic_path)
        except Exception:
            diagnostic_path_text = ""
        info = {
            'cache_hit': False,
            'cache_only': True,
            'reason': cache_miss_reason,
            'bundle_statuses': bundle_statuses,
            'artifact_store_candidates': artifact_store_diagnostics,
            'cache_key': cache_key,
            'cache_dir': str(cache_dir or ''),
            'requested_count': int(calib_count),
            'payload': cache_payload,
            'diagnostic_path': diagnostic_path_text,
            'build_evidence': negative_info,
            'workspace_contract': diagnostic['workspace_contract'],
            'compiler_dispatch_count': 0,
        }
        if policy_cache_verify:
            return _make_hef_result(
                ok=False,
                elapsed_s=time.time() - t0,
                hw_arch=str(hw_arch_eff),
                net_name=str(net_name),
                backend="local",
                skipped=True,
                failure_kind="cache_miss_blocked",
                unsupported_reason="cache_verify_only_policy",
                error=cache_miss_blocked_message(
                    "hailo_dfc",
                    f"exact cache miss key={cache_key[:12]} "
                    f"net={net_name} hw_arch={hw_arch_eff}",
                ),
                last_stage="cache_lookup",
                timed_out=False,
                calib_info=info,
                details=info,
                fixed_onnx_path=str(fixed_path) if fixed_path else None,
                fixup_report=fixup_report,
            )
        return _make_hef_result(
            ok=False, elapsed_s=time.time() - t0, hw_arch=str(hw_arch_eff),
            net_name=str(net_name), backend='local', skipped=True,
            failure_kind='deferred_cold_full_cache_miss',
            unsupported_reason='cache_only_policy',
            error='Smoke cold-build policy deferred a Hailo Full cache miss.',
            last_stage='cache_lookup', timed_out=False, calib_info=info,
            details=info,
            fixed_onnx_path=str(fixed_path) if fixed_path else None, fixup_report=fixup_report,
        )

    workspace_preflight = hailo_dfc_workspace_preflight(
        out_dir,
        calibration_count=int(effective_calib_count),
        input_shapes=calibration_identity_shapes,
    )
    workspace_preflight_path = out_dir / "hailo_dfc_workspace_preflight.json"
    try:
        _atomic_write_json(workspace_preflight_path, workspace_preflight)
    except Exception:
        pass
    if workspace_preflight.get("status") in {"failed", "unknown"}:
        problems = ";".join(
            str(value) for value in workspace_preflight.get("problems") or []
        )
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend="local",
            error=(
                "Local Hailo DFC workspace preflight failed before compiler "
                f"dispatch: {problems}"
            ),
            failure_kind=("local_dfc_workspace_unresolved"
                          if workspace_preflight.get("status") == "unknown"
                          else "local_dfc_workspace_insufficient"),
            last_stage="local_dfc_workspace_preflight",
            timed_out=False,
            fixed_onnx_path=str(fixed_path) if fixed_path else None,
            fixup_report=fixup_report,
            details={
                "workspace_preflight": workspace_preflight,
                "workspace_preflight_path": str(workspace_preflight_path),
                "compiler_dispatch_count": 0,
                "compiler_dispatched": False,
            },
        )

    # Keep the proprietary SDK import call-local.  Controller hosts without DFC
    # must still be able to import this module, and cache-only Smoke runs must
    # be able to defer a cold Full build before the SDK is imported.
    compiler_context_effective: Dict[str, Any] = {}
    try:
        from .hailo_compiler_context import validate_compiler_child_environment
        if "ONNX_SPLITPOINT_HAILO_RESOLVED_COMPILER_CONTEXT" in os.environ:
            compiler_context_effective = validate_compiler_child_environment()
        from hailo_sdk_client import ClientRunner  # type: ignore
    except Exception as e:
        return _make_hef_result(
            ok=False, elapsed_s=time.time() - t0, hw_arch=str(hw_arch_eff),
            net_name=str(net_name), backend='local',
            error=f'Hailo SDK initialization failed: {e}', failure_kind=getattr(e, 'reason', 'sdk_unavailable'),
            last_stage='compiler_context' if hasattr(e, 'reason') else 'sdk_import', timed_out=False,
            details={'compiler_dispatch_count': 0, 'compiler_context': compiler_context_effective, 'context_error': getattr(e, 'details', {})},
            fixed_onnx_path=str(fixed_path) if fixed_path else None,
            fixup_report=fixup_report,
        )

    phase_events: List[Dict[str, Any]] = []
    phase_started = time.monotonic()
    def phase_start(name: str) -> None:
        nonlocal phase_started
        phase_started = time.monotonic()
        phase_events.append({"phase": name, "state": "started", "event": "started", "timestamp": time.time(),
                             "monotonic_s": phase_started, "monotonic": phase_started, "pid": os.getpid()})
        try:
            _atomic_write_json(out_dir / "hailo_build_phases.json", {"events": phase_events})
        except OSError as exc:
            log.warning("[hailo][phases] could not persist phase evidence: %s", exc)
    def phase_finish(name: str, state: str = "completed") -> None:
        phase_events.append({"phase": name, "state": state, "event": state, "timestamp": time.time(),
                             "monotonic_s": time.monotonic(), "monotonic": time.monotonic(), "elapsed_s": max(0.0, time.monotonic()-phase_started),
                             "pid": os.getpid()})
        try:
            _atomic_write_json(out_dir / "hailo_build_phases.json", {"events": phase_events})
        except OSError as exc:
            log.warning("[hailo][phases] could not persist phase evidence: %s", exc)
    active_dfc_stage = "sdk_initialization"
    phase_start(active_dfc_stage)
    try:
        runner = ClientRunner(hw_arch=str(hw_arch_eff))
        translate_kwargs = _apply_translate_node_overrides({
            'model': str(model_for_parse),
            'net_name': str(net_name),
            'net_input_shapes': net_input_shapes,
            'disable_rt_metadata_extraction': bool(disable_rt_metadata_extraction),
        }, start_node_names=start_node_names, end_node_names=end_node_names)
        phase_finish(active_dfc_stage)
        active_dfc_stage = "translate"
        phase_start(active_dfc_stage)
        print(f"[hailo][translate] start net={net_name} hw_arch={hw_arch_eff}", flush=True)
        _run_with_hailo_heartbeat(f"translate net={net_name}", lambda: runner.translate_onnx_model(**translate_kwargs))
        print(f"[hailo][translate] done net={net_name}", flush=True)
        phase_finish(active_dfc_stage)

        parsed_har = out_dir / "parsed.har"
        if keep_artifacts:
            try:
                runner.save_har(str(parsed_har))
            except Exception:
                pass

        # Build calibration dataset
        active_dfc_stage = "calibration_materialization"
        phase_start(active_dfc_stage)
        hn = runner.get_hn_dict() or {}
        hn_layers = hn.get("layers") or {}
        if not isinstance(hn_layers, dict):
            hn_layers = {}
        input_layers = _sort_hn_input_layers(hn_layers)
        if not input_layers:
            raise RuntimeError("No HN input layers found after translate")

        # Choose calibration data (dir -> fallback random)
        calib_inputs: Dict[str, np.ndarray] = {}
        calib_meta: Dict[str, Any] = {
            "source": None,
            "requested_count": int(calib_count),
            "used_count": None,
            "batch_size": None,
            "inputs": {},
        }

        # Prepare expected shapes per input
        expected_shapes: Dict[str, List[int]] = {}
        for in_name in input_layers:
            meta = hn_layers.get(in_name) if isinstance(hn_layers, dict) else None
            shp = _hn_get_shape(meta) if isinstance(meta, dict) else None
            if shp is None:
                # fallback to net_input_shapes if available
                if isinstance(net_input_shapes, dict) and in_name in net_input_shapes:
                    shp = list(net_input_shapes[in_name])
                elif isinstance(net_input_shapes, list) and len(input_layers) == 1:
                    shp = list(net_input_shapes)
            if shp is None:
                # last resort
                shp = [1]
            expected_shapes[in_name] = [int(x) for x in shp]

        image_preprocess_eff = _infer_hailo_image_preprocess(
            model_path=onnx_path,
            expected_shapes=expected_shapes,
            activation_part1_onnx=activation_part1_onnx_p,
        )
        if str(image_preprocess_eff) != str(
            preprocessing_contract_eff.get("image_scale") or ""
        ):
            raise RuntimeError(
                "Hailo numeric image preprocessing contradicts the sealed task contract: "
                f"resolved={image_preprocess_eff!r} "
                f"contract={preprocessing_contract_eff.get('image_scale')!r}"
            )
        try:
            print(f"[hailo][calib] image_preprocess={image_preprocess_eff}")
        except Exception:
            pass

        # v60o: Smoke keeps the memory cap; Standard/Final use disk-backed
        # calibration tensors so requested counts (e.g. 500 detector images)
        # are not silently reduced to ~54 samples.
        eff_count = _verify_hailo_calibration_count_after_translation(
            sealed_effective_count=effective_calib_count,
            requested=int(calib_count),
            expected_shapes=expected_shapes,
            storage=calibration_storage,
            cap_bytes=calibration_memory_cap,
            available_sample_count=calibration_available_count,
        )
        calib_temp_paths: List[Path] = []

        used_dir = False
        used_activation = False
        activation_debug: Optional[Dict[str, Any]] = None
        activation_part2_names: Optional[List[str]] = None

        if calib_dir_p is not None and calib_dir_p.exists():
            # IMPORTANT:
            # For exported Part2 models we prefer activation calibration whenever
            # Part1 is available, even when HN exposes only a single input layer.
            #
            # Rationale:
            # - The single-input HN case often still corresponds to an internal
            #   cut-tensor / activation tensor, not to a raw image input.
            # - Resizing image samples directly into that tensor shape can produce
            #   a syntactically valid but semantically wrong calibration set.
            # - Several real splits (for example late YOLOv7 boundaries such as
            #   b066) only fail on Hailo-part2 because the previous logic routed
            #   single-input Part2 models through `_try_build_calib_from_dir()`.
            #
            # Therefore: whenever `activation_part1_onnx` is available, treat the
            # model as a Part2 split and generate calibration activations from
            # Part1 first. Direct image calibration from `calib_dir` remains the
            # fallback only for models without a Part1 activation source.
            if activation_part1_onnx_p is not None:
                try:
                    print(
                        f"[hailo][activation] start net={net_name} hw_arch={hw_arch_eff} "
                        f"source=part1 limit={eff_count} batch={max(1, int(activation_gen_batch))}",
                        flush=True,
                    )
                    activation_part2_names, calib_by_part2_input, activation_debug = _run_with_hailo_heartbeat(
                        f"activation_calibration net={net_name}",
                        lambda: _build_activation_calib_from_part1_onnx(
                            part1_onnx=activation_part1_onnx_p,
                            part2_onnx=model_for_parse,
                            calib_dir=calib_dir_p,
                            limit=eff_count,
                            gen_batch=max(1, int(activation_gen_batch)),
                            input_preprocess=str(image_preprocess_eff),
                            preprocessing_contract=preprocessing_contract_eff,
                        ),
                    )
                    print(
                        f"[hailo][activation] done net={net_name} inputs={len(activation_part2_names or [])}",
                        flush=True,
                    )
                    if len(activation_part2_names) != len(input_layers):
                        raise RuntimeError(
                            f'Multi-input activation calib count mismatch: part2_onnx_inputs={len(activation_part2_names)} hn_inputs={len(input_layers)}'
                        )
                    conv_info: Dict[str, Any] = {'hn_inputs': list(input_layers), 'conversions': {}}
                    for i, hn_in in enumerate(input_layers):
                        onnx_in = activation_part2_names[i]
                        ds = calib_by_part2_input[onnx_in]
                        hn_shape = expected_shapes[hn_in]
                        before = list(ds.shape)
                        ds2, how = _convert_calib_dataset_to_hn_shape(ds, hn_shape)
                        calib_inputs[hn_in] = ds2
                        conv_info['conversions'][hn_in] = {
                            'onnx_input': onnx_in,
                            'before': before,
                            'hn_shape': list(hn_shape),
                            'after': list(ds2.shape),
                            'how': how,
                        }
                    if activation_debug is None:
                        activation_debug = {}
                    activation_debug['strategy'] = 'activation_from_part1_preferred'
                    activation_debug['hn_inputs'] = list(input_layers)
                    activation_debug['hn_input_shapes'] = {hn_in: list(expected_shapes[hn_in]) for hn_in in input_layers}
                    activation_debug['calib_shapes_after_hn'] = {k: {'dataset_shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in calib_inputs.items()}
                    activation_debug['calib_conversion'] = conv_info
                    try:
                        (out_dir / 'calib_activations_shapes.json').write_text(json.dumps(activation_debug, indent=2), encoding='utf-8')
                    except Exception:
                        pass
                    proxy_manifest_path = _write_activation_proxy_cache_manifest(
                        out_dir=out_dir,
                        part1_onnx=activation_part1_onnx_p,
                        part2_onnx=model_for_parse,
                        calib_dir=calib_dir_p,
                        calib_arrays_by_part2_input=calib_by_part2_input,
                        activation_debug=activation_debug,
                        eff_count=int(eff_count),
                        gen_batch=max(1, int(activation_gen_batch)),
                        stage2_backend=str(hw_arch_eff),
                    )
                    if proxy_manifest_path:
                        activation_debug['activation_proxy_cache_manifest'] = proxy_manifest_path
                    used_activation = True
                except Exception as exc:
                    msg = (
                        'Failed to build activation calibration for multi-input Hailo model. '
                        f'part1={activation_part1_onnx_p} part2={model_for_parse} calib_dir={calib_dir_p}. '
                        f'Details: {type(exc).__name__}: {exc}'
                    )
                    preflight = _activation_calib_preflight(part1_onnx=activation_part1_onnx_p, part2_onnx=model_for_parse)
                    failure_kind = 'invalid_calibration_set'
                    skipped = False
                    unsupported_reason = None
                    if preflight.get('inspect_ok') and preflight.get('compatible') is False:
                        msg = _format_activation_calib_preflight_error(preflight)
                        failure_kind = 'unsupported_splitpoint'
                        skipped = True
                        unsupported_reason = 'activation_preflight_missing_inputs'
                    return _make_hef_result(
                        ok=False,
                        elapsed_s=time.time() - t0,
                        hw_arch=str(hw_arch_eff),
                        net_name=str(net_name),
                        backend='local',
                        fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
                        fixup_report=fixup_report,
                        error=msg,
                        skipped=skipped,
                        failure_kind=failure_kind,
                        unsupported_reason=unsupported_reason,
                        calib_info={
                            'source': 'activation_from_part1',
                            'requested_count': int(calib_count),
                            'preflight': preflight,
                            'calib_dir': str(calib_dir_p) if calib_dir_p is not None else None,
                        },
                    )
            elif len(input_layers) == 1:
                in0 = input_layers[0]
                memmap_path = out_dir / f"{net_name}_calibration_float32.mmap" if calibration_storage == 'memmap' else None
                ds = _try_build_calib_from_dir(
                    calib_dir=calib_dir_p, expected_shape=expected_shapes[in0], limit=eff_count,
                    preprocess=str(image_preprocess_eff), storage_mode=calibration_storage, memmap_path=memmap_path,
                    preprocessing_contract=preprocessing_contract_eff,
                )
                if ds is not None and memmap_path is not None:
                    calib_temp_paths.append(memmap_path)
                if ds is not None:
                    calib_inputs[in0] = ds
                    used_dir = True

        if not used_dir and not used_activation:
            rng = np.random.default_rng(0)
            for in_name in input_layers:
                shp = expected_shapes[in_name]
                if calibration_storage == 'memmap':
                    fd, raw_path = tempfile.mkstemp(prefix='splitpoint_hailo_random_', suffix='.mmap', dir=str(out_dir))
                    os.close(fd)
                    ds = np.memmap(raw_path, dtype=np.float32, mode='w+', shape=(eff_count, *shp))
                    chunk = max(1, min(16, eff_count))
                    for pos in range(0, eff_count, chunk):
                        stop = min(eff_count, pos + chunk)
                        ds[pos:stop] = rng.random((stop - pos, *shp), dtype=np.float32)
                    ds.flush()
                    calib_temp_paths.append(Path(raw_path))
                    calib_inputs[in_name] = ds
                else:
                    ds = rng.random((eff_count, *shp), dtype=np.float32)
                    calib_inputs[in_name] = np.ascontiguousarray(ds)

        actual_count = _verify_hailo_materialized_calibration_count(
            sealed_effective_count=int(eff_count),
            calib_inputs=calib_inputs,
        )
        # Determine batch size from the materialised dataset, not the requested count.
        bs = max(1, min(int(calib_batch_size), int(actual_count)))
        if used_activation:
            calib_meta['source'] = 'activation_from_part1'
            calib_meta['activation_part1_onnx'] = str(activation_part1_onnx_p) if activation_part1_onnx_p is not None else None
            calib_meta['activation_gen_batch'] = int(max(1, int(activation_gen_batch)))
            if activation_debug is not None:
                calib_meta['activation_debug_path'] = str(out_dir / 'calib_activations_shapes.json')
                if activation_debug.get('activation_proxy_cache_manifest'):
                    calib_meta['activation_proxy_cache_manifest'] = str(activation_debug.get('activation_proxy_cache_manifest'))
                _ap_source = str(activation_debug.get('activation_proxy_source') or 'ort_cpu_reference_proxy')
                calib_meta['activation_calibration_source'] = _ap_source
                calib_meta['trust_level'] = 'proxy'
                calib_meta['producer_backend'] = str(activation_debug.get('activation_proxy_producer_backend') or 'ort_cpu')
                calib_meta['requested_backend'] = str(activation_debug.get('activation_proxy_requested_backend') or 'ort_cpu')
                calib_meta['producer_exact'] = False
                calib_meta['providers_requested'] = list(activation_debug.get('activation_proxy_providers_requested') or [])
                calib_meta['available_providers'] = list(activation_debug.get('activation_proxy_available_providers') or [])
                if activation_debug.get('activation_proxy_provider_fallback'):
                    calib_meta['provider_fallback_reason'] = str(activation_debug.get('activation_proxy_provider_fallback'))
        else:
            calib_meta['source'] = str(calib_dir_p) if used_dir and calib_dir_p is not None else 'random'
        calib_meta['image_preprocess'] = str(image_preprocess_eff)
        calib_meta['preprocessing_contract'] = dict(preprocessing_contract_eff)
        calib_meta['preprocessing_contract_sha256'] = str(preprocessing_sha256)
        calib_meta['requested_count'] = int(calib_count)
        calib_meta['effective_count'] = int(eff_count)
        calib_meta['available_sample_count'] = (
            int(calibration_available_count)
            if calibration_available_count is not None else None
        )
        calib_meta['available_sample_count_source'] = str(
            calibration_available_count_source
        )
        calib_meta['clamped'] = bool(int(eff_count) < int(calib_count))
        clamp_reasons: List[str] = []
        if memory_limited_calib_count < int(calib_count):
            clamp_reasons.append('memory_cap')
        if (
            calibration_available_count is not None
            and int(calibration_available_count)
            < memory_limited_calib_count
        ):
            clamp_reasons.append(
                f'available_samples:{calibration_available_count_source}'
            )
        calib_meta['clamped_reason'] = '+'.join(clamp_reasons)
        calib_meta['storage_mode'] = calibration_storage
        calib_meta['calibration_memory_cap_bytes'] = int(
            calibration_memory_cap
        )
        calib_meta['effective_requested_count'] = int(eff_count)
        calib_meta['used_count'] = int(actual_count)
        calib_meta['batch_size'] = int(bs)
        for k, shp in expected_shapes.items():
            calib_meta['inputs'][k] = {'shape': list(shp)}

        model_script = (
            f"model_optimization_flavor(optimization_level={int(opt_level)}, batch_size={int(bs)})\n"
            f"model_optimization_config(calibration, batch_size={int(bs)}, calibset_size={int(actual_count)})\n"
        )
        extra_script = str(extra_model_script or "").strip()
        if extra_script:
            model_script += extra_script
            if not model_script.endswith("\n"):
                model_script += "\n"
        runner.load_model_script(model_script)

        phase_finish(active_dfc_stage)
        active_dfc_stage = "optimize"
        phase_start(active_dfc_stage)
        print(
            f"[hailo][optimize] start net={net_name} hw_arch={hw_arch_eff} "
            f"calib_source={calib_meta.get('source')} used_count={calib_meta.get('used_count')} "
            f"batch_size={calib_meta.get('batch_size')} opt_level={int(opt_level)}",
            flush=True,
        )
        _run_with_hailo_heartbeat(f"optimize net={net_name}", lambda: runner.optimize(calib_inputs))
        print(f"[hailo][optimize] done net={net_name}", flush=True)
        phase_finish(active_dfc_stage)
        # The SDK consumed calibration synchronously. Release disk-backed arrays
        # before compilation so multi-model runs do not accumulate GiB-sized files.
        if calib_temp_paths:
            import gc
            for _arr in list(calib_inputs.values()):
                try:
                    if isinstance(_arr, np.memmap):
                        _arr.flush()
                        if getattr(_arr, '_mmap', None) is not None:
                            _arr._mmap.close()
                except Exception:
                    pass
            calib_inputs.clear()
            gc.collect()
            for _tmp in calib_temp_paths:
                try:
                    _tmp.unlink(missing_ok=True)
                except TypeError:
                    if _tmp.exists():
                        _tmp.unlink()
                except Exception:
                    pass

        quant_har = out_dir / "quantized.har"
        if keep_artifacts:
            try:
                runner.save_har(str(quant_har))
            except Exception:
                pass

        active_dfc_stage = "compile"
        phase_start(active_dfc_stage)
        print(f"[hailo][compile] start net={net_name} hw_arch={hw_arch_eff}", flush=True)
        hef_bytes = _run_with_hailo_heartbeat(f"compile net={net_name}", lambda: runner.compile())
        print(f"[hailo][compile] done net={net_name} bytes={len(hef_bytes) if hef_bytes is not None else 'none'}", flush=True)
        if not hef_bytes:
            raise RuntimeError("hailo_compiler_empty_hef")
        phase_finish(active_dfc_stage)
        active_dfc_stage = "publication"
        phase_start(active_dfc_stage)
        # Never write through the public HEF symlink: a forced rebuild must
        # preserve the prior immutable generation until the full tuple commits.
        with tempfile.TemporaryDirectory(prefix=".hailo-compile-", dir=out_dir) as stage_dir:
            staged_hef = Path(stage_dir) / "compiled.hef"
            staged_hef.write_bytes(hef_bytes)
            hef_validation = (_inspect_hailo_diagnostic_hef(staged_hef) if not publish_artifacts
                              else {"status": "not_run", "reason": "regular_compiler_receipt_contract"})
            calib_meta["hef_validation"] = hef_validation
            if hef_validation["status"] == "failed":
                raise RuntimeError("hailo_diagnostic_hef_validation_failed:" + json.dumps(hef_validation, sort_keys=True))
            build_receipt = _write_hailo_receipt(
                hef_path=staged_hef,
                source_onnx=onnx_path,
                compiler_onnx=model_for_parse,
                hw_arch=str(hw_arch_eff),
                net_name=str(net_name),
                preprocessing_contract=preprocessing_contract_eff,
                preprocessing_sha256=preprocessing_sha256,
                cache_key=cache_key,
                cache_payload=cache_payload,
                calibration_identity=str(cache_payload.get('calibration_identity') or 'none'),
                calibration_count=int(actual_count),
            )
            build_receipt.update({"compiler_context": compiler_context_effective,
                                  "phase_events": list(phase_events),
                                  "publish_artifacts": publish_artifacts,
                                  "diagnostic_only": not publish_artifacts,
                                  "hef_validation": hef_validation})
            _atomic_write_json(_hailo_receipt_path(staged_hef), build_receipt)
            committed_hef = _publish_hailo_bundle(
                source_hef=staged_hef, destination=hef_path, receipt=build_receipt,
                source="compiler",
            )
        phase_finish(active_dfc_stage)
        calib_meta.update({"compiler_dispatch_count": 1, "compiler_context": compiler_context_effective,
                           "phase_events": phase_events, "publish_artifacts": publish_artifacts,
                           "gpu_execution_status": "gpu_execution_unproven" if compiler_context_effective.get("device") == "gpu" else "not_requested"})
        calib_meta['build_receipt'] = build_receipt
        calib_meta['build_receipt_path'] = str(_hailo_receipt_path(committed_hef))
        calib_meta['cache_hit'] = False
        calib_meta['cache_key'] = cache_key
        if cache_enabled and cache_dir is not None and cache_key:
            cache_backfilled = _backfill_hailo_exact_cache(
                cache_dir=cache_dir, hef_path=committed_hef,
                receipt=build_receipt, cache_key=cache_key,
                cache_payload=cache_payload, preprocessing_sha256=preprocessing_sha256,
                source_onnx_sha256=source_onnx_sha256, net_name=str(net_name),
                hw_arch=str(hw_arch_eff),
            )
            calib_meta['cache_bundle_backup'] = "sealed" if cache_backfilled else "failed"
            if not cache_backfilled:
                calib_meta['cache_backup_error'] = "atomic_cache_bundle_backup_failed"
                log.warning("[hailo][cache] HEF build succeeded but bundle backup failed: %s", cache_dir)

        print(
            f"[hailo-cache] BUILD role=hef model={net_name} "
            f"identity={cache_key} reason={cache_miss_reason} "
            f"artifact={hef_path} hw_arch={hw_arch_eff}",
            flush=True,
        )

        return HailoHefBuildResult(
            ok=True,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend="local",
            hef_path=str(hef_path),
            parsed_har_path=(str(parsed_har) if keep_artifacts else None),
            quant_har_path=(str(quant_har) if keep_artifacts else None),
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
            skipped=False,
            calib_info=calib_meta,
            details={
                'hef_validation': hef_validation,
                'compiler_dispatch_count': 1,
                'phase_events': phase_events,
                'compiler_context': compiler_context_effective,
                'publish_artifacts': publish_artifacts,
                'cache_hit': False,
                'cache_key': cache_key or None,
                'cache_dir': str(cache_dir) if cache_dir is not None else None,
                'preprocessing_contract': preprocessing_contract_eff,
                'preprocessing_contract_sha256': preprocessing_sha256,
                'build_receipt': build_receipt,
            },
        )

    except Exception as e:
        phase_finish(active_dfc_stage, "failed")
        for array in list(locals().get("calib_inputs", {}).values()):
            try:
                if isinstance(array, np.memmap) and getattr(array, "_mmap", None) is not None:
                    array._mmap.close()
            except Exception:
                pass
        for path in locals().get("calib_temp_paths", []):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        err = str(e)
        # Helpful hint for a very common binary-compatibility issue in WSL/Linux.
        # Example: "libc.so.6: version GLIBC_2.34 not found".
        if "GLIBC_" in err and "libc.so.6" in err:
            err = (
                err
                + "\n\n"
                + "Hint: Your Linux/WSL distro ships an older glibc than the Hailo DFC wheel expects. "
                + "Use a newer distro (e.g. Ubuntu 22.04/24.04) and provision the DFC venv there, "
                + "or (on Windows) set the GUI 'WSL distro' field to that newer distro."
            )
        classification = _classify_hailo_failure_text(err)
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
            net_name=str(net_name),
            backend="local",
            error=err,
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
            last_stage=active_dfc_stage,
            details={"compiler_dispatch_count": 1, "phase_events": phase_events, "compiler_context": compiler_context_effective, "publish_artifacts": publish_artifacts},
            **classification,
        )


def hailo_build_hef_via_wsl(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    disable_rt_metadata_extraction: bool = True,
    opt_level: int = 1,
    calib_dir: Optional[Union[str, Path]] = None,
    calib_count: int = 64,
    calib_batch_size: int = 8,
    activation_part1_onnx: Optional[Union[str, Path]] = None,
    activation_gen_batch: int = 8,
    force: bool = False,
    cache_only: bool = False,
    keep_artifacts: bool = False,
    publish_artifacts: bool = True,
    compute_device: Optional[str] = None,
    gpu_selector: Optional[str] = None,
    compute_by_family: Optional[Mapping[str, Any]] = None,
    compiler_context: Optional[Mapping[str, Any]] = None,
    extra_model_script: Optional[str] = None,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    preprocessing_contract: Optional[Union[Mapping[str, Any], str]] = None,
    # WSL bridge settings
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 3600,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> HailoHefBuildResult:
    """Build a HEF inside WSL (Windows host -> WSL2 backend)."""

    force = parse_config_bool(force, field="hailo_build.force_build")
    publish_artifacts = parse_config_bool(publish_artifacts, field="hailo_build.publish_artifacts")
    if not publish_artifacts or compute_device is not None or gpu_selector is not None or compute_by_family is not None or compiler_context is not None:
        return _make_hef_result(ok=False, elapsed_s=0.0, hw_arch=str(hw_arch),
            net_name=str(net_name or Path(onnx_path).stem), backend="wsl",
            failure_kind="hailo_compute_requires_managed_child", last_stage="compiler_context",
            error="Explicit family compute and isolated diagnostics require running backend='venv' inside Linux/WSL.")
    t0 = time.time()
    onnx_path = Path(onnx_path)
    net_name = str(net_name or onnx_path.stem).strip()
    if not net_name:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name="",
            backend="wsl",
            failure_kind="invalid_build_contract",
            error="Hailo net_name must be non-empty after normalization",
            last_stage="cache_contract",
            timed_out=False,
        )
    try:
        net_input_shapes_eff = (
            _normalize_hailo_net_input_shapes(net_input_shapes)
            if net_input_shapes is not None else None
        )
    except ValueError as exc:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name,
            backend="wsl",
            failure_kind="invalid_build_contract",
            error=f"Invalid Hailo net_input_shapes: {exc}",
            last_stage="cache_contract",
            timed_out=False,
        )

    activation_part1_onnx_p = Path(activation_part1_onnx).expanduser().resolve() if activation_part1_onnx else None
    if activation_part1_onnx_p is not None:
        preflight = _activation_calib_preflight(part1_onnx=activation_part1_onnx_p, part2_onnx=onnx_path)
        if preflight.get('inspect_ok') and preflight.get('compatible') is False:
            return _make_hef_result(
                ok=False,
                elapsed_s=time.time() - t0,
                hw_arch=str(hw_arch),
                net_name=str(net_name),
                backend='wsl',
                skipped=True,
                failure_kind='unsupported_splitpoint',
                unsupported_reason='activation_preflight_missing_inputs',
                error=_format_activation_calib_preflight_error(preflight),
                calib_info={'source': 'activation_from_part1', 'preflight': preflight},
            )

    hard_timeout_s, idle_timeout_s = _resolve_hef_timeout_policy(wsl_timeout_s)

    if sys.platform != "win32":
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error="WSL backend is only available when running on Windows.",
        )

    if not hailo_wsl_available():
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error="WSL backend not available (wsl.exe not found).",
        )

    # Resolve managed venv/distro if requested.
    try:
        mgr = get_dfc_manager()
        resolved = mgr.resolve_wsl_runtime(
            hw_arch=str(hw_arch),
            wsl_distro=_clean_opt_str(wsl_distro),
            wsl_venv_activate=(_clean_opt_str(wsl_venv_activate) or "auto"),
        )
    except Exception as e:
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"Failed to resolve DFC profile: {e}",
        )

    distro_eff = str(resolved.wsl_distro or "").strip() or None
    venv_eff = str(resolved.wsl_venv_activate or "").strip()
    if not venv_eff:
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=(
                f"No managed DFC profile found for hw_arch={hw_arch!r}. "
                "Set an explicit WSL venv path, or add a profile in resources/hailo/profiles.json."
            ),
        )

    helper_win = (Path(__file__).resolve().parent / "wsl_inline_build_hef")
    helper_wsl = windows_path_to_wsl(str(helper_win))

    onnx_wsl = windows_path_to_wsl(str(onnx_path.resolve()))
    outdir_wsl = None
    if outdir is not None:
        outdir_wsl = windows_path_to_wsl(str(Path(outdir).resolve()))
    calib_wsl = None
    if calib_dir is not None:
        calib_wsl = windows_path_to_wsl(str(Path(calib_dir).resolve()))
    activation_part1_onnx_wsl = None
    if activation_part1_onnx is not None:
        activation_part1_onnx_wsl = windows_path_to_wsl(str(Path(activation_part1_onnx).resolve()))

    venv_activate = venv_eff  # do not quote '~'

    preprocessing_json = (
        json.dumps(dict(preprocessing_contract), sort_keys=True, separators=(',', ':'))
        if isinstance(preprocessing_contract, Mapping)
        else str(
            preprocessing_contract
            or os.environ.get("ONNX_SPLITPOINT_HAILO_PREPROCESSING_CONTRACT_JSON")
            or ""
        )
    )
    task_env_value = str(
        task or os.environ.get("ONNX_SPLITPOINT_HAILO_CALIB_TASK") or ""
    )

    from .hailo_negative_evidence import child_context
    evidence_context = child_context()
    if evidence_context.get("full_source_onnx_path"):
        evidence_context["full_source_onnx_path"] = windows_path_to_wsl(
            str(evidence_context["full_source_onnx_path"])
        )
    evidence_root = Path(os.environ.get("ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT")
                         or (Path.home() / ".onnx_splitpoint_tool" / "build_evidence"))
    evidence_root_wsl = windows_path_to_wsl(str(evidence_root.resolve()))
    evidence_json = json.dumps(evidence_context, sort_keys=True, separators=(",", ":"))

    cmd = (
        "set -e; "
        "echo __SPLITPOINT_WSL_BEGIN__; "
        f"source {venv_activate}; "
        "echo __SPLITPOINT_WSL_VENV_OK__; "
        "export PYTHONUNBUFFERED=1; "
        f"export ONNX_SPLITPOINT_HAILO_CALIB_TASK={_bash_quote(task_env_value)}; "
        f"export ONNX_SPLITPOINT_HAILO_PREPROCESSING_CONTRACT_JSON={_bash_quote(preprocessing_json)}; "
        f"export ONNX_SPLITPOINT_BUILD_EVIDENCE_CONTEXT_JSON={_bash_quote(evidence_json)}; "
        f"export ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT={_bash_quote(evidence_root_wsl)}; "
        # Self-heal: setuptools 82+ removed pkg_resources, but some Hailo SDK
        # components still import it.
        "python -c \"import pkg_resources\" >/dev/null 2>&1 || "
        "python -m pip install --force-reinstall \"setuptools<82\" >/dev/null 2>&1 || true; "
        # Use `python` after venv activation to ensure we run the venv interpreter.
        f"python {_bash_quote(helper_wsl)}"
        f" --onnx {_bash_quote(onnx_wsl)}"
        f" --hw-arch {_bash_quote(str(hw_arch))}"
        f" --net-name {_bash_quote(str(net_name))}"
        f" --fixup {'1' if fixup else '0'}"
        f" --add-conv-defaults {'1' if add_conv_defaults else '0'}"
        f" --disable-rt-metadata-extraction {'1' if disable_rt_metadata_extraction else '0'}"
        f" --opt-level {int(opt_level)}"
        f" --calib-count {int(calib_count)}"
        f" --calib-batch-size {int(calib_batch_size)}"
        f" --force {'1' if force else '0'}"
        f" --cache-only {'1' if cache_only else '0'}"
        f" --keep-artifacts {'1' if keep_artifacts else '0'}"
    )
    if outdir_wsl is not None:
        cmd += f" --outdir {_bash_quote(outdir_wsl)}"
    if net_input_shapes_eff is not None:
        cmd += (
            " --net-input-shapes-json "
            + _bash_quote(json.dumps(
                net_input_shapes_eff,
                sort_keys=True,
                separators=(",", ":"),
            ))
        )
    if start_node_names:
        cmd += f" --start-node-names-json {_bash_quote(json.dumps(list(start_node_names)))}"
    if end_node_names:
        cmd += f" --end-node-names-json {_bash_quote(json.dumps(list(end_node_names)))}"
    if calib_wsl is not None:
        cmd += f" --calib-dir {_bash_quote(calib_wsl)}"
    if activation_part1_onnx_wsl is not None:
        cmd += f" --activation-part1-onnx {_bash_quote(activation_part1_onnx_wsl)}"
    cmd += f" --activation-gen-batch {int(activation_gen_batch)}"
    extra_script = str(extra_model_script or "").strip()
    if extra_script:
        encoded = base64.b64encode(extra_script.encode('utf-8')).decode('ascii')
        cmd += f" --extra-model-script-b64 {_bash_quote(encoded)}"

    wsl_cmd: List[str] = [_wsl_exe()]
    if distro_eff:
        wsl_cmd += ["-d", str(distro_eff)]
    wsl_cmd += ["--", "bash", "-lc", cmd]

    try:
        log.info(
            "[hailo][hef][wsl] hw_arch=%s profile=%s distro=%s activate=%s onnx=%s outdir=%s hard_timeout_s=%s idle_timeout_s=%s",
            hw_arch,
            resolved.profile_id,
            distro_eff or "",
            venv_eff,
            onnx_wsl,
            outdir_wsl or "",
            hard_timeout_s,
            idle_timeout_s if idle_timeout_s is not None else "off",
        )
        log.debug("[hailo][hef][wsl] cmd=%s", wsl_cmd)
        run = _run_streamed_subprocess(
            wsl_cmd,
            stdin_yes=True,
            on_log=on_log,
            hard_timeout_s=hard_timeout_s,
            idle_timeout_s=idle_timeout_s,
        )
    except Exception as e:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"WSL HEF build failed to launch: {type(e).__name__}: {e}",
            failure_kind='launch_error',
        )

    stdout = run.stdout
    stderr = run.stderr
    rc = int(run.returncode or 0)
    proc_details = _build_subprocess_detail_bundle(
        run,
        stdout,
        stderr,
        include_system_snapshot=bool(run.timed_out),
    )

    if run.timed_out:
        dbg_path = _write_wsl_debug_log(
            outdir,
            filename=f"hailo_wsl_hef_timeout_{hw_arch}_{net_name}_{int(time.time())}.log",
            wsl_cmd=wsl_cmd,
            stdout=stdout,
            stderr=stderr,
        )
        timeout_budget = hard_timeout_s if run.timeout_kind != 'idle' or idle_timeout_s is None else idle_timeout_s
        err = f"WSL HEF build timed out ({run.timeout_kind or 'hard'}) after {timeout_budget}s."
        if run.last_stage:
            err += f" Last active stage: {run.last_stage}."
        if dbg_path:
            err += f"\n\n[debug_log] {dbg_path}"
        timeout_details = _merge_detail_dict(
            proc_details,
            {
                'hard_timeout_s': int(hard_timeout_s),
                'idle_timeout_s': int(idle_timeout_s) if idle_timeout_s is not None else None,
                'effective_timeout_kind': run.timeout_kind,
            },
        )
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=err,
            returncode=(124 if rc == 0 else rc),
            debug_log=dbg_path,
            timed_out=True,
            timeout_kind=run.timeout_kind,
            last_stage=run.last_stage,
            failure_kind='timeout',
            details=timeout_details,
        )
    mixed = "\n".join([stdout, stderr]).strip()
    payload = _find_result_json(mixed)
    if payload is None:
        # Convert Windows unsigned return code (e.g. 0xFFFFFFFF) to signed for readability.
        if isinstance(rc, int) and rc > 0x7FFFFFFF:
            rc_signed = rc - 0x100000000
        else:
            rc_signed = rc
        tail = mixed[-4000:] if mixed else "<no stdout/stderr captured>"
        dbg_path = _write_wsl_debug_log(
            outdir,
            filename=f"hailo_wsl_hef_{hw_arch}_{net_name}_{int(time.time())}.log",
            wsl_cmd=wsl_cmd,
            stdout=stdout,
            stderr=stderr,
        )
        if dbg_path:
            tail = tail + f"\n\n[debug_log] {dbg_path}"
        log.warning(
            "[hailo][hef][wsl] no structured result rc=%s signed=%s debug_log=%s",
            rc,
            rc_signed,
            dbg_path or "-",
        )
        recovered = _recover_hef_result_from_compiled_artifact(
            outdir,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            returncode=rc,
            debug_log=dbg_path,
            last_stage=run.last_stage,
            details=_build_subprocess_detail_bundle(run, stdout, stderr, include_system_snapshot=False),
        )
        if recovered is not None:
            log.info("[hailo][hef][wsl] recovered structured HEF result from compiled.hef: %s", recovered.hef_path)
            return recovered
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=(
                "WSL HEF build did not return a structured result. "
                f"exit_code={rc} (signed {rc_signed}). tail=\n{tail}\n\n"
                "Details were written to gui.log (Logs tab)."
            ),
            returncode=rc,
            debug_log=dbg_path,
            last_stage=run.last_stage,
            failure_kind='missing_structured_result',
            details=_build_subprocess_detail_bundle(run, stdout, stderr, include_system_snapshot=True),
        )

    # Structured result present. Still write a debug log on failures so users can
    # inspect the full stdout/stderr from the DFC.
    if not bool(payload.get("ok")):
        dbg_path = _write_wsl_debug_log(
            outdir,
            filename=f"hailo_wsl_hef_fail_{hw_arch}_{net_name}_{int(time.time())}.log",
            wsl_cmd=wsl_cmd,
            stdout=stdout,
            stderr=stderr,
        )
        err_txt = str(payload.get("error") or "").rstrip()
        if err_txt:
            err_txt += "\n\n"
        if dbg_path:
            err_txt += f"[debug_log] {dbg_path}\n"
        err_txt += "Details were written to gui.log (Logs tab)."
        payload["error"] = err_txt
        payload.setdefault('debug_log', dbg_path)
        payload.setdefault('returncode', rc)
        payload.setdefault('last_stage', run.last_stage)

    merged_payload_details = _build_subprocess_detail_bundle(
        run,
        stdout,
        stderr,
        include_system_snapshot=not bool(payload.get("ok")),
    )
    if merged_payload_details:
        payload['details'] = _merge_detail_dict(payload.get('details'), merged_payload_details)

    return _hef_result_from_payload(
        payload,
        elapsed_default=time.time() - t0,
        hw_arch=str(hw_arch),
        net_name=str(net_name),
        backend_default='wsl',
        returncode=rc,
        last_stage=run.last_stage,
    )



_HAILO_DIAGNOSTIC_PATHS = {
    "HOME": "home", "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data", "TFHUB_CACHE_DIR": "tfhub",
    "TORCH_EXTENSIONS_DIR": "torch-extensions", "JOBLIB_TEMP_FOLDER": "joblib",
    "XDG_STATE_HOME": "state", "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
    "PYTHONPYCACHEPREFIX": "pycache", "NUMBA_CACHE_DIR": "numba", "TORCH_HOME": "torch",
    "TRITON_CACHE_DIR": "triton", "CUDA_CACHE_PATH": "cuda-cache", "KERAS_HOME": "keras",
    "MPLCONFIGDIR": "matplotlib", "ONNX_SPLITPOINT_HAILO_CACHE_ROOT": "hef-cache",
    "ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT": "artifact-store",
    "ONNX_SPLITPOINT_BUILD_EVIDENCE_ROOT": "build-evidence",
    "ONNX_SPLITPOINT_TOOL_ROOT": "tool-state",
}


def _hailo_diagnostic_child_environment(parent_env: Mapping[str, str], outdir: Path) -> Dict[str, str]:
    """Scope all backend/framework writable state to one explicit diagnostic job.

    Called in the controller but only returns a child environment. No parent
    HOME, config, cache roots or TensorFlow state are changed.
    """
    root = Path(outdir).absolute()
    if any(c.isspace() for c in str(root)):
        raise ValueError("hailo_diagnostic_outdir_requires_whitespace_free_path")
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("hailo_diagnostic_outdir_must_not_traverse_symlinks")
    root.mkdir(parents=True, exist_ok=True)
    env = dict(parent_env)
    state = root / "diagnostic_state"
    for key, directory in _HAILO_DIAGNOSTIC_PATHS.items():
        path = state / directory
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("hailo_diagnostic_state_symlink:" + str(path))
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env.update(ONNX_SPLITPOINT_HAILO_PUBLISH_ARTIFACTS="0",
               ONNX_SPLITPOINT_HAILO_DIAGNOSTIC_ROOT=str(root),
               ONNX_SPLITPOINT_HAILO_CACHE_ENABLED="0",
               ONNX_SPLITPOINT_ARTIFACT_STORE_ENABLED="0",
               PYTHONDONTWRITEBYTECODE="1")
    return env


def _validate_hailo_diagnostic_environment(outdir: Any) -> None:
    raw = os.environ.get("ONNX_SPLITPOINT_HAILO_DIAGNOSTIC_ROOT", "")
    if not outdir or not raw:
        raise ValueError("hailo_diagnostic_requires_isolated_managed_child")
    root = Path(raw).absolute()
    if root.resolve() != root or Path(outdir).resolve() != root:
        raise ValueError("hailo_diagnostic_outdir_mismatch")
    state = root / "diagnostic_state"
    for key, directory in _HAILO_DIAGNOSTIC_PATHS.items():
        expected = state / directory
        if os.environ.get(key) != str(expected) or expected.resolve() != expected:
            raise ValueError("hailo_diagnostic_write_target_not_isolated:" + key)


def hailo_build_hef_via_venv(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    disable_rt_metadata_extraction: bool = True,
    opt_level: int = 1,
    calib_dir: Optional[Union[str, Path]] = None,
    calib_count: int = 64,
    calib_batch_size: int = 8,
    activation_part1_onnx: Optional[Union[str, Path]] = None,
    activation_gen_batch: int = 8,
    force: bool = False,
    cache_only: bool = False,
    keep_artifacts: bool = False,
    publish_artifacts: bool = True,
    compute_device: Optional[str] = None,
    gpu_selector: Optional[str] = None,
    compute_by_family: Optional[Mapping[str, Any]] = None,
    compiler_context: Optional[Mapping[str, Any]] = None,
    extra_model_script: Optional[str] = None,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    preprocessing_contract: Optional[Union[Mapping[str, Any], str]] = None,
    venv_activate: str = "auto",
    timeout_s: int = 3600,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> HailoHefBuildResult:
    """Build a HEF inside a managed DFC venv (Linux / WSL)."""

    force = parse_config_bool(force, field="hailo_build.force_build")
    publish_artifacts = parse_config_bool(publish_artifacts, field="hailo_build.publish_artifacts")
    if not publish_artifacts and outdir is None:
        raise ValueError("hailo_diagnostic_outdir_required")
    t0 = time.time()
    onnx_path = Path(str(onnx_path)).expanduser().resolve()
    net_name_eff = str(net_name or onnx_path.stem).strip()
    if not net_name_eff:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name="",
            backend="venv",
            failure_kind="invalid_build_contract",
            error="Hailo net_name must be non-empty after normalization",
            last_stage="cache_contract",
            timed_out=False,
        )
    try:
        net_input_shapes_eff = (
            _normalize_hailo_net_input_shapes(net_input_shapes)
            if net_input_shapes is not None else None
        )
    except ValueError as exc:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            failure_kind="invalid_build_contract",
            error=f"Invalid Hailo net_input_shapes: {exc}",
            last_stage="cache_contract",
            timed_out=False,
        )
    outdir_path = Path(str(outdir)).expanduser().resolve() if outdir else None
    if outdir_path is not None:
        outdir_path.mkdir(parents=True, exist_ok=True)

    activation_part1_onnx_p = Path(str(activation_part1_onnx)).expanduser().resolve() if activation_part1_onnx else None
    if activation_part1_onnx_p is not None:
        preflight = _activation_calib_preflight(part1_onnx=activation_part1_onnx_p, part2_onnx=onnx_path)
        if preflight.get('inspect_ok') and preflight.get('compatible') is False:
            return _make_hef_result(
                ok=False,
                elapsed_s=time.time() - t0,
                hw_arch=str(hw_arch),
                net_name=net_name_eff,
                backend='venv',
                skipped=True,
                failure_kind='unsupported_splitpoint',
                unsupported_reason='activation_preflight_missing_inputs',
                error=_format_activation_calib_preflight_error(preflight),
                calib_info={'source': 'activation_from_part1', 'preflight': preflight},
            )

    hard_timeout_s, idle_timeout_s = _resolve_hef_timeout_policy(timeout_s)

    if sys.platform == "win32":
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error="Managed venv HEF build is not available on Windows (use WSL backend).",
        )

    try:
        profile_id, py, _act = _resolve_managed_venv_python(hw_arch=str(hw_arch), venv_activate=(_clean_opt_str(venv_activate) or "auto"))
    except Exception as e:
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Failed to resolve managed DFC venv: {e}",
        )

    # Direct managed callers receive the same compiler-free reuse path as
    # auto dispatch. Even an explicit GPU request must not probe hardware for
    # an already verified artifact.
    if publish_artifacts and not force:
        import inspect
        values = locals().copy()
        probe_kwargs = {name: values[name] for name in inspect.signature(_hailo_build_hef_legacy).parameters
                        if name in values and name != "onnx_path"}
        probe_kwargs.update(cache_only=True, read_only_cache_probe=bool(cache_only),
                            sdk_version_token=_hailo_sdk_version_token_from_managed_venv(
                                hw_arch=str(hw_arch), venv_activate=venv_activate) or "unknown")
        probe = _hailo_build_hef_legacy(onnx_path, **probe_kwargs)
        if (probe.ok and (probe.calib_info or {}).get("cache_hit")) or cache_only or compiler_dispatch_forbidden():
            return probe
        if probe.failure_kind in {"known_negative_build_evidence", "build_evidence_error", "build_evidence_conflict",
                                  "invalid_preprocessing_contract", "invalid_build_contract"}:
            return probe

    helper = Path(__file__).resolve().parent / "wsl_inline_build_hef"
    if not helper.exists():
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Helper script missing: {helper}",
        )

    cmd: List[str] = [
        str(py),
        str(helper),
        "--onnx",
        str(onnx_path),
        "--hw-arch",
        str(hw_arch),
        "--net-name",
        str(net_name_eff),
        "--fixup",
        "1" if fixup else "0",
        "--add-conv-defaults",
        "1" if add_conv_defaults else "0",
        "--disable-rt-metadata-extraction",
        "1" if disable_rt_metadata_extraction else "0",
        "--opt-level",
        str(int(opt_level)),
        "--calib-count",
        str(int(calib_count)),
        "--calib-batch-size",
        str(int(calib_batch_size)),
        "--force",
        "1" if force else "0",
        "--cache-only",
        "1" if cache_only else "0",
        "--publish-artifacts",
        "1" if publish_artifacts else "0",
        "--keep-artifacts",
        "1" if keep_artifacts else "0",
    ]
    if outdir_path is not None:
        cmd += ["--outdir", str(outdir_path)]
    if net_input_shapes_eff is not None:
        cmd += [
            "--net-input-shapes-json",
            json.dumps(
                net_input_shapes_eff,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ]
    if start_node_names:
        cmd += ["--start-node-names-json", json.dumps(list(start_node_names))]
    if end_node_names:
        cmd += ["--end-node-names-json", json.dumps(list(end_node_names))]
    if calib_dir is not None:
        cmd += ["--calib-dir", str(Path(str(calib_dir)).expanduser().resolve())]
    if activation_part1_onnx_p is not None:
        cmd += ["--activation-part1-onnx", str(activation_part1_onnx_p)]
    cmd += ["--activation-gen-batch", str(int(activation_gen_batch))]
    extra_script = str(extra_model_script or "").strip()
    if extra_script:
        encoded = base64.b64encode(extra_script.encode('utf-8')).decode('ascii')
        cmd += ["--extra-model-script-b64", encoded]

    try:
        log.info(
            "[hailo][hef][venv] hw_arch=%s profile=%s python=%s onnx=%s outdir=%s hard_timeout_s=%s idle_timeout_s=%s",
            hw_arch,
            profile_id,
            str(py),
            str(onnx_path),
            str(outdir_path or ""),
            hard_timeout_s,
            idle_timeout_s if idle_timeout_s is not None else "off",
        )
        log.debug("[hailo][hef][venv] cmd=%s", cmd)

        # The Hailo SDK drops multiple `hailo_sdk.*.log` files into the current
        # working directory. Run helpers from a dedicated log folder to avoid
        # cluttering the user's project/repo directory.
        from .paths import ensure_dir, splitpoint_logs_dir

        # Target-local SDK cwd is required when the evaluation scheduler runs
        # Hailo-8 and Hailo-10 helpers at the same time.
        hailo_log_cwd = ensure_dir(
            (outdir_path / "sdk_logs") if not publish_artifacts
            else (splitpoint_logs_dir() / "hailo_sdk" / str(profile_id) / _normalize_hailo_hw_arch(hw_arch))
        )

        # Best-effort log retention for Hailo SDK logs.
        try:
            from .log_retention import LogRetentionPolicy, apply_log_retention

            apply_log_retention(
                [hailo_log_cwd] if publish_artifacts else [],
                policy=LogRetentionPolicy(
                    enabled=True,
                    max_age_days=14,
                    max_files=80,
                    patterns=("*.log",),
                    keep_names=(),
                ),
                recursive=False,
            )
        except Exception:
            pass
        env = _managed_venv_child_env(py)
        from .hailo_negative_evidence import CONTEXT_ENV, child_context
        env[CONTEXT_ENV] = json.dumps(child_context(), sort_keys=True, separators=(",", ":"))
        env["HAILORT_LOGGER_PATH"] = str(hailo_log_cwd / "hailort.log")
        env.setdefault("ONNX_SPLITPOINT_HAILO_HELPER_BACKEND", "venv")
        if task is not None:
            env["ONNX_SPLITPOINT_HAILO_CALIB_TASK"] = str(task)
        if preprocessing_contract is not None:
            env["ONNX_SPLITPOINT_HAILO_PREPROCESSING_CONTRACT_JSON"] = (
                json.dumps(dict(preprocessing_contract), sort_keys=True, separators=(",", ":"))
                if isinstance(preprocessing_contract, Mapping)
                else str(preprocessing_contract)
            )
        from .hailo_compiler_context import (
            resolve_hailo_compiler_context, compiler_child_environment,
        )
        job_override = None
        if compute_device is not None or gpu_selector is not None:
            job_override = {"device": compute_device or "gpu"}
            if gpu_selector is not None:
                job_override["gpu_selector"] = str(gpu_selector)
        if not publish_artifacts:
            env = _hailo_diagnostic_child_environment(env, outdir_path)
            env[CONTEXT_ENV] = "{}"
        if len(_HAILO_COMPILER_PROBE_CACHE) > 128:
            _HAILO_COMPILER_PROBE_CACHE.clear()
        resolved_context = resolve_hailo_compiler_context(
            str(py), str(hw_arch), job_override=job_override,
            compute_by_family=compute_by_family, explicit_context=compiler_context,
            parent_env=env, probe_cache=_HAILO_COMPILER_PROBE_CACHE, work_dir=outdir_path,
        )
        with compiler_child_environment(resolved_context, parent_env=env,
                                        work_dir=outdir_path) as (env, effective_context):
            if outdir_path is not None:
                _atomic_write_json(outdir_path / "hailo_compiler_context.json", effective_context)
            run = _run_streamed_subprocess(
                cmd, cwd=str(hailo_log_cwd), env=env, stdin_yes=True, on_log=on_log,
                hard_timeout_s=hard_timeout_s, idle_timeout_s=idle_timeout_s,
            )
    except Exception as e:
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Venv HEF build failed to launch: {type(e).__name__}: {e}",
            failure_kind=getattr(e, 'reason', 'launch_error'),
            last_stage='compiler_context' if hasattr(e, 'reason') or 'resolved_context' in locals() else 'launch',
            details={'publish_artifacts': publish_artifacts, 'compiler_context': locals().get('resolved_context', {}), 'error_details': getattr(e, 'details', {})},
        )

    stdout = run.stdout
    stderr = run.stderr
    rc = int(run.returncode or 0)
    proc_details = _build_subprocess_detail_bundle(
        run,
        stdout,
        stderr,
        include_system_snapshot=bool(run.timed_out),
    )

    proc_details.update(compiler_context=effective_context, publish_artifacts=publish_artifacts)

    if run.timed_out:
        dbg_path = _write_wsl_debug_log(
            str(outdir_path) if outdir_path is not None else None,
            filename=f"hailo_venv_hef_timeout_{hw_arch}_{net_name_eff}_{int(time.time())}.log",
            wsl_cmd=cmd,
            stdout=stdout,
            stderr=stderr,
        )
        timeout_budget = hard_timeout_s if run.timeout_kind != 'idle' or idle_timeout_s is None else idle_timeout_s
        err = f"Venv HEF build timed out ({run.timeout_kind or 'hard'}) after {timeout_budget}s."
        if run.last_stage:
            err += f" Last active stage: {run.last_stage}."
        if dbg_path:
            err += f"\n\n[debug_log] {dbg_path}"
        timeout_details = _merge_detail_dict(
            proc_details,
            {
                'hard_timeout_s': int(hard_timeout_s),
                'idle_timeout_s': int(idle_timeout_s) if idle_timeout_s is not None else None,
                'effective_timeout_kind': run.timeout_kind,
            },
        )
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=err,
            returncode=(124 if rc == 0 else rc),
            debug_log=dbg_path,
            timed_out=True,
            timeout_kind=run.timeout_kind,
            last_stage=run.last_stage,
            failure_kind='timeout',
            details=timeout_details,
        )
    mixed = "\n".join([stdout, stderr]).strip()
    payload = _find_result_json(mixed)
    if payload is None:
        tail = mixed[-4000:] if mixed else "<no stdout/stderr captured>"
        dbg_path = _write_wsl_debug_log(
            str(outdir_path) if outdir_path is not None else None,
            filename=f"hailo_venv_hef_{hw_arch}_{net_name_eff}_{int(time.time())}.log",
            wsl_cmd=cmd,
            stdout=stdout,
            stderr=stderr,
        )
        if dbg_path:
            tail = tail + f"\n\n[debug_log] {dbg_path}"
        log.warning(
            "[hailo][hef][venv] no structured result rc=%s debug_log=%s",
            rc,
            dbg_path or "-",
        )
        recovered = _recover_hef_result_from_compiled_artifact(
            outdir_path,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            returncode=rc,
            debug_log=dbg_path,
            last_stage=run.last_stage,
            details=_build_subprocess_detail_bundle(run, stdout, stderr, include_system_snapshot=False),
        )
        if recovered is not None:
            log.info("[hailo][hef][venv] recovered structured HEF result from compiled.hef: %s", recovered.hef_path)
            return recovered
        _err_msg = (
            "Venv HEF build did not return a structured result. "
            f"exit_code={rc}. tail=\n{tail}\n\n"
            "Details were written to gui.log (Logs tab)."
        )
        _cls = _classify_hailo_failure_text(_err_msg + "\n" + str(stdout or "") + "\n" + str(stderr or ""))
        _details = _build_subprocess_detail_bundle(run, stdout, stderr, include_system_snapshot=True)
        if _cls:
            _details = _merge_detail_dict(_details, {
                "error_class": _cls.get("error_class"),
                "root_cause_hint": _cls.get("root_cause_hint"),
                "diagnostic_hint": _cls.get("diagnostic_hint"),
            })
        return _make_hef_result(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=_err_msg,
            returncode=rc,
            debug_log=dbg_path,
            last_stage=run.last_stage,
            failure_kind=_cls.get("failure_kind") if _cls else 'missing_structured_result',
            details=_details,
        )

    # Structured result present. Still write a debug log on failures so users can
    # inspect the full stdout/stderr from the DFC.
    if not bool(payload.get("ok")):
        dbg_path = _write_wsl_debug_log(
            str(outdir_path) if outdir_path is not None else None,
            filename=f"hailo_venv_hef_fail_{hw_arch}_{net_name_eff}_{int(time.time())}.log",
            wsl_cmd=cmd,
            stdout=stdout,
            stderr=stderr,
        )
        err_txt = str(payload.get("error") or "").rstrip()
        if err_txt:
            err_txt += "\n\n"
        if dbg_path:
            err_txt += f"[debug_log] {dbg_path}\n"
        err_txt += "Details were written to gui.log (Logs tab)."
        payload["error"] = err_txt
        payload.setdefault('debug_log', dbg_path)
        payload.setdefault('returncode', rc)
        payload.setdefault('last_stage', run.last_stage)

    merged_payload_details = _build_subprocess_detail_bundle(
        run,
        stdout,
        stderr,
        include_system_snapshot=not bool(payload.get("ok")),
    )
    merged_payload_details.update(compiler_context=effective_context, publish_artifacts=publish_artifacts)
    if merged_payload_details:
        payload['details'] = _merge_detail_dict(payload.get('details'), merged_payload_details)

    return _hef_result_from_payload(
        payload,
        elapsed_default=time.time() - t0,
        hw_arch=str(hw_arch),
        net_name=net_name_eff,
        backend_default='venv',
        returncode=rc,
        last_stage=run.last_stage,
    )


def hailo_build_hef_auto(
    onnx_path: Union[str, Path],
    *,
    backend: str = "auto",
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    net_input_shapes: Optional[Union[List[int], Dict[str, List[int]]]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    disable_rt_metadata_extraction: bool = True,
    opt_level: int = 1,
    calib_dir: Optional[Union[str, Path]] = None,
    calib_count: int = 64,
    calib_batch_size: int = 8,
    activation_part1_onnx: Optional[Union[str, Path]] = None,
    activation_gen_batch: int = 8,
    force: bool = False,
    cache_only: bool = False,
    keep_artifacts: bool = False,
    publish_artifacts: bool = True,
    compute_device: Optional[str] = None,
    gpu_selector: Optional[str] = None,
    compute_by_family: Optional[Mapping[str, Any]] = None,
    compiler_context: Optional[Mapping[str, Any]] = None,
    extra_model_script: Optional[str] = None,
    start_node_names: Optional[Sequence[str]] = None,
    end_node_names: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
    preprocessing_contract: Optional[Union[Mapping[str, Any], str]] = None,
    # WSL bridge
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 3600,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> HailoHefBuildResult:
    force = parse_config_bool(force, field="hailo_build.force_build")
    publish_artifacts = parse_config_bool(publish_artifacts, field="hailo_build.publish_artifacts")
    if not publish_artifacts and outdir is None:
        raise ValueError("hailo_diagnostic_outdir_required")
    policy_cache_verify = compiler_dispatch_forbidden()
    cache_only = bool(cache_only or policy_cache_verify)
    mode = normalize_hailo_backend(backend)
    if mode == "subprocess":
        mode = subprocess_backend_for_platform()
    net_name = str(net_name or Path(str(onnx_path)).stem).strip()
    if not net_name:
        return _make_hef_result(
            ok=False,
            elapsed_s=0.0,
            hw_arch=str(hw_arch),
            net_name="",
            backend=str(mode),
            failure_kind="invalid_build_contract",
            error="Hailo net_name must be non-empty after normalization",
            last_stage="cache_contract",
            timed_out=False,
        )

    if bool(cache_only) and bool(force):
        return _make_hef_result(
            ok=False,
            elapsed_s=0.0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend=str(mode),
            skipped=True,
            failure_kind="cache_miss_blocked",
            unsupported_reason="cache_verify_only_force_conflict",
            error=(
                "cache_miss_blocked[hailo_dfc]: cache_only and force are "
                "mutually exclusive; DFC dispatch was not started"
            ),
            last_stage="cache_lookup",
            timed_out=False,
        )

    # Every backend must reject an undeclared image task before backend cache
    # reuse, environment/profile resolution, or compiler launch.  In
    # particular, a 320x320 detector must never be reclassified as a classifier
    # by size.
    try:
        _resolve_hailo_image_contract(
            model_path=Path(onnx_path).expanduser().resolve(),
            activation_part1=(
                Path(activation_part1_onnx).expanduser().resolve()
                if activation_part1_onnx
                else None
            ),
            net_input_shapes=net_input_shapes,
            task=task,
            declared=preprocessing_contract,
        )
    except Exception as exc:
        return _make_hef_result(
            ok=False,
            elapsed_s=0.0,
            hw_arch=str(hw_arch),
            net_name=str(net_name or Path(onnx_path).stem),
            backend=str(mode),
            error=f"Invalid Hailo image preprocessing contract: {type(exc).__name__}: {exc}",
            failure_kind="invalid_preprocessing_contract",
            last_stage="preprocessing_contract_preflight",
            timed_out=False,
        )

    # A subprocess/venv dispatcher must not be launched merely to discover a
    # reusable artifact.  Run the compiler-free legacy cache path first; it
    # may return only an exact destination/local-cache/ArtifactStore v2 hit.
    # A miss continues through the selected backend unchanged.
    if publish_artifacts and (mode != "local" or bool(cache_only)) and not bool(force):
        exact_probe_sdk_token: Optional[str] = None
        exact_probe_sdk_source = ""
        # Cache verification must not call backend discovery merely to learn
        # the compiler identity: some discovery paths import or probe DFC.
        # ``auto`` uses the managed Linux venv when one is present, and the
        # metadata reader below is itself filesystem-only and returns ``None``
        # when that venv is absent.
        managed_backend_selected = bool(
            mode == "venv"
            or (mode == "auto" and sys.platform != "win32")
        )
        if managed_backend_selected:
            exact_probe_sdk_token = _hailo_sdk_version_token_from_managed_venv(
                hw_arch=str(hw_arch),
                venv_activate=str(wsl_venv_activate or "auto"),
            )
            if exact_probe_sdk_token:
                exact_probe_sdk_source = "managed_venv"
            if not exact_probe_sdk_token and mode == "auto":
                # ``auto`` may legitimately select a controller-local DFC.
                # Distribution metadata is still process-free and does not
                # import the compiler module.
                exact_probe_sdk_token = (
                    _hailo_sdk_version_token_from_controller_metadata()
                )
                if exact_probe_sdk_token:
                    exact_probe_sdk_source = "controller_metadata"
        elif mode == "local":
            exact_probe_sdk_token = (
                _hailo_sdk_version_token_from_controller_metadata()
            )
            if exact_probe_sdk_token:
                exact_probe_sdk_source = "controller_metadata"

        if exact_probe_sdk_token:
            print(
                "[hailo][cache] compiler identity "
                f"source={exact_probe_sdk_source} "
                f"token={exact_probe_sdk_token}",
                flush=True,
            )

        if (policy_cache_verify or bool(cache_only)) and not exact_probe_sdk_token:
            return _make_hef_result(
                ok=False,
                elapsed_s=0.0,
                hw_arch=str(hw_arch),
                net_name=str(net_name),
                backend=str(mode),
                skipped=True,
                failure_kind="cache_miss_blocked",
                unsupported_reason="compiler_identity_unavailable",
                error=cache_miss_blocked_message(
                    "hailo_dfc",
                    "managed compiler identity unavailable; filesystem-only "
                    "package metadata did not identify the DFC version",
                ),
                last_stage="cache_identity",
                timed_out=False,
                details={
                    "cache_hit": False,
                    "compiler_identity_available": False,
                    "compiler_dispatch_allowed": False,
                },
            )
        exact_probe = _hailo_build_hef_legacy(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            opt_level=int(opt_level),
            calib_dir=calib_dir,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            activation_part1_onnx=activation_part1_onnx,
            activation_gen_batch=int(activation_gen_batch),
            force=False,
            cache_only=True,
            read_only_cache_probe=bool(cache_only),
            keep_artifacts=bool(keep_artifacts),
            publish_artifacts=publish_artifacts,
            extra_model_script=extra_model_script,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            task=task,
            preprocessing_contract=preprocessing_contract,
            # A controller prelookup never imports the compiler to discover
            # its identity. Missing metadata remains explicitly unknown; a
            # permitted real build can identify its SDK in the managed child.
            sdk_version_token=exact_probe_sdk_token or "unknown",
            negative_evidence_identity_authoritative=bool(
                exact_probe_sdk_token
                and (exact_probe_sdk_source == "managed_venv" or mode == "local")
            ),
        )
        exact_probe_info = getattr(exact_probe, "calib_info", None)
        if (
            bool(getattr(exact_probe, "ok", False))
            and bool(getattr(exact_probe, "skipped", False))
            and isinstance(exact_probe_info, Mapping)
            and exact_probe_info.get("cache_hit") is True
        ):
            return exact_probe
        if str(getattr(exact_probe, "failure_kind", "") or "") in {
            "known_negative_build_evidence", "build_evidence_error", "build_evidence_conflict",
        }:
            return exact_probe
        if bool(cache_only):
            # The parent process has already exhausted destination, exact v3,
            # legacy-v2 migration and ArtifactStore restore. Never start a
            # managed venv/WSL/local compiler child merely to repeat the miss.
            return exact_probe

    def _run_local() -> HailoHefBuildResult:
        return hailo_build_hef(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            opt_level=int(opt_level),
            calib_dir=calib_dir,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            activation_part1_onnx=activation_part1_onnx,
            activation_gen_batch=int(activation_gen_batch),
            force=bool(force),
            cache_only=bool(cache_only),
            keep_artifacts=bool(keep_artifacts),
            publish_artifacts=publish_artifacts,
            extra_model_script=extra_model_script,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            task=task,
            preprocessing_contract=preprocessing_contract,
        )

    def _run_venv() -> HailoHefBuildResult:
        return hailo_build_hef_via_venv(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            opt_level=int(opt_level),
            calib_dir=calib_dir,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            activation_part1_onnx=activation_part1_onnx,
            activation_gen_batch=int(activation_gen_batch),
            force=bool(force),
            cache_only=bool(cache_only),
            keep_artifacts=bool(keep_artifacts),
            publish_artifacts=publish_artifacts,
            extra_model_script=extra_model_script,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            task=task,
            preprocessing_contract=preprocessing_contract,
            **{key: value for key, value in {
                "compute_device": compute_device, "gpu_selector": gpu_selector,
                "compute_by_family": compute_by_family, "compiler_context": compiler_context,
            }.items() if value is not None},
            venv_activate=wsl_venv_activate,
            timeout_s=int(wsl_timeout_s),
            on_log=on_log,
        )

    explicit_compute = bool(not publish_artifacts or compute_device is not None or gpu_selector is not None
                            or compute_by_family is not None or compiler_context is not None
                            or os.environ.get("ONNX_SPLITPOINT_HAILO_COMPUTE_BY_FAMILY")
                            or os.environ.get("ONNX_SPLITPOINT_HAILO_COMPUTE_OVERRIDE"))
    if mode == "auto" and explicit_compute and sys.platform != "win32":
        # A failed selected compiler environment must not become a differently
        # configured in-process SDK build via auto fallback.
        return _run_venv()
    if mode == "local" and explicit_compute:
        return _make_hef_result(ok=False, elapsed_s=0.0, hw_arch=str(hw_arch),
            net_name=str(net_name or Path(onnx_path).stem), backend="local",
            failure_kind="hailo_compute_requires_managed_child",
            last_stage="compiler_context", error="Explicit compute selection and diagnostic isolation require backend='venv'.")
    if mode == "local":
        return _run_local()
    if mode == "venv":
        return _run_venv()
    if mode == "wsl":
        return hailo_build_hef_via_wsl(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            opt_level=int(opt_level),
            calib_dir=calib_dir,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            activation_part1_onnx=activation_part1_onnx,
            activation_gen_batch=int(activation_gen_batch),
            force=bool(force),
            cache_only=bool(cache_only),
            keep_artifacts=bool(keep_artifacts),
            publish_artifacts=publish_artifacts,
            extra_model_script=extra_model_script,
            start_node_names=start_node_names,
            end_node_names=end_node_names,
            task=task,
            preprocessing_contract=preprocessing_contract,
            **{key: value for key, value in {
                "compute_device": compute_device, "gpu_selector": gpu_selector,
                "compute_by_family": compute_by_family, "compiler_context": compiler_context,
            }.items() if value is not None},
            wsl_distro=wsl_distro,
            wsl_venv_activate=wsl_venv_activate,
            wsl_timeout_s=int(wsl_timeout_s),
            on_log=on_log,
        )

    if mode == "auto" and sys.platform != "win32":
        prefer_subprocess = auto_prefers_subprocess()
        if prefer_subprocess:
            res = _run_venv()
            err_text = str(getattr(res, "error", "") or "")
            if bool(getattr(res, "ok", False)) or bool(getattr(res, "skipped", False)) or not hailo_sdk_available():
                return res
            if "Failed to resolve managed DFC venv" not in err_text:
                return res
        if hailo_sdk_available():
            return _run_local()

    if mode in {"auto", "local"} and hailo_sdk_available():
        return _run_local()

    if mode == "auto":
        # Linux: auto should prefer the managed venv.
        if sys.platform != "win32":
            return _run_venv()
        return hailo_build_hef_via_wsl(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            net_input_shapes=net_input_shapes,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            opt_level=int(opt_level),
            calib_dir=calib_dir,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            activation_part1_onnx=activation_part1_onnx,
            activation_gen_batch=int(activation_gen_batch),
            force=bool(force),
            cache_only=bool(cache_only),
            keep_artifacts=bool(keep_artifacts),
            publish_artifacts=publish_artifacts,
            extra_model_script=extra_model_script,
            task=task,
            preprocessing_contract=preprocessing_contract,
            **{key: value for key, value in {
                "compute_device": compute_device, "gpu_selector": gpu_selector,
                "compute_by_family": compute_by_family, "compiler_context": compiler_context,
            }.items() if value is not None},
            wsl_distro=wsl_distro,
            wsl_venv_activate=wsl_venv_activate,
            wsl_timeout_s=int(wsl_timeout_s),
            on_log=on_log,
        )

    return HailoHefBuildResult(
        ok=False,
        elapsed_s=0.0,
        hw_arch=str(hw_arch),
        net_name=str(net_name or Path(str(onnx_path)).stem),
        error=(
            "No usable Hailo backend available. "
            "Install hailo_sdk_client in this Python env, or use the managed Hailo DFC venv/backend (auto/subprocess/venv). On Windows you can also configure the WSL backend."
        ),
    )


# v60s unified artifact-library bridge ---------------------------------------
def _v60s_hailo_extract_hef(value):
    from pathlib import Path as _Path
    if isinstance(value, (_Path, str)):
        p = _Path(value).expanduser()
        if p.suffix.lower() == ".hef" and p.is_file():
            return p.resolve()
    if isinstance(value, dict):
        for key in ("hef", "hef_path", "artifact", "artifact_path", "compiled_hef", "path"):
            if key in value:
                found = _v60s_hailo_extract_hef(value[key])
                if found:
                    return found
        for item in value.values():
            found = _v60s_hailo_extract_hef(item)
            if found:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _v60s_hailo_extract_hef(item)
            if found:
                return found
    for attr in ("hef_path", "artifact_path", "compiled_hef", "output_path", "path"):
        try:
            found = _v60s_hailo_extract_hef(getattr(value, attr))
            if found:
                return found
        except Exception:
            pass
    return None


def _v60s_hailo_bound(args, kwargs):
    import inspect as _inspect
    try:
        return dict(_inspect.signature(_hailo_build_hef_legacy).bind_partial(*args, **kwargs).arguments)
    except Exception:
        return dict(kwargs)


def _v60s_hailo_contract(bound):
    from pathlib import Path as _Path
    import hashlib as _hashlib
    import json as _json
    # Execution policy (cache-only/build-missing/timeouts) must not change the
    # identity of the generated compiler artifact.  Otherwise a Full HEF built
    # in Standard/Final could never satisfy a Smoke cache-only lookup.
    skip = {"outdir", "out_dir", "output_dir", "workdir", "log_path", "logger", "progress", "callback",
            "timeout_s", "wsl_timeout_s", "hard_timeout_s", "idle_timeout_s", "cache_dir", "cache_root", "force",
            "cache_only", "publish_artifacts", "compute_device", "gpu_selector", "compute_by_family", "compiler_context", "tool_version", "workflow_version", "release", "release_id", "build_id",
            "source_run", "run_id", "backend", "wsl_distro", "wsl_venv_activate", "venv_activate",
            "on_log", "build_evidence_context", "negative_evidence_identity_authoritative"}
    payload = {"schema": "onnx-splitpoint/hailo-build-contract/v2"}
    try:
        model_path = _Path(bound.get("onnx_path")).expanduser().resolve()
        activation_part1 = (
            _Path(bound.get("activation_part1_onnx")).expanduser().resolve()
            if bound.get("activation_part1_onnx")
            else None
        )
        canonical_preprocess, canonical_preprocess_sha = _resolve_hailo_image_contract(
            model_path=model_path,
            activation_part1=activation_part1,
            net_input_shapes=bound.get("net_input_shapes"),
            task=bound.get("task"),
            declared=bound.get("preprocessing_contract"),
        )
        payload["preprocessing_contract"] = canonical_preprocess
        payload["preprocessing_contract_sha256"] = canonical_preprocess_sha
    except Exception as exc:
        # Keep the contract non-reusable.  The legacy builder returns the
        # structured fail-closed preprocessing error immediately afterwards.
        payload["preprocessing_contract_error"] = f"{type(exc).__name__}: {exc}"
    for key, value in sorted(bound.items()):
        name = str(key)
        if name in skip or name in {"task", "preprocessing_contract"} or callable(value):
            continue
        if name == "calib_dir" and value:
            p = _Path(value).expanduser()
            manifest = next(
                (
                    candidate
                    for candidate in (
                        p / "selection.json",
                        p / "manifest.json",
                        p / "dataset_manifest.json",
                    )
                    if candidate.is_file()
                ),
                None,
            )
            if manifest is not None:
                # Preserve the exact 2.69f contract representation so existing
                # ArtifactStore records remain reusable across the upgrade.
                payload[name] = {
                    "name": p.name,
                    "manifest_sha256": _hashlib.sha256(manifest.read_bytes()).hexdigest(),
                }
            else:
                strict = str(os.environ.get('ONNX_SPLITPOINT_HAILO_CACHE_INTEGRITY') or 'relaxed').lower() == 'strict'
                payload[name] = {
                    "name": p.name,
                    "identity": _calibration_identity(p, strict=strict),
                    "integrity": "strict" if strict else "relaxed",
                }
            continue
        if isinstance(value, _Path) or (
            isinstance(value, str)
            and (
                "path" in name
                or "onnx" in name
                or name.endswith("file")
                or name in {"model"}
            )
        ):
            p = _Path(value).expanduser()
            if p.is_file():
                h = _hashlib.sha256()
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                        h.update(chunk)
                payload[name] = {"name": p.name, "sha256": h.hexdigest(), "size": p.stat().st_size}
            elif p.is_dir():
                manifest = next((x for x in (p / "selection.json", p / "manifest.json", p / "dataset_manifest.json") if x.is_file()), None)
                payload[name] = {"name": p.name, "manifest_sha256": _hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest else ""}
            else:
                payload[name] = str(value)
            continue
        try:
            _json.dumps(value)
            payload[name] = value
        except Exception:
            payload[name] = repr(value)
    for env_name in ("ONNX_SPLITPOINT_HAILO_PROFILE", "ONNX_SPLITPOINT_HAILO_TARGET_ARCHES", "ONNX_SPLITPOINT_RUN_MODE"):
        if os.environ.get(env_name):
            payload[env_name.lower()] = os.environ.get(env_name)
    return payload


def _v60s_hailo_destination(bound):
    from pathlib import Path as _Path
    for name in ("hef_path", "output_path", "artifact_path"):
        value = bound.get(name)
        if value:
            p = _Path(value).expanduser()
            if p.suffix.lower() == ".hef":
                return p
    for name in ("outdir", "out_dir", "output_dir"):
        value = bound.get(name)
        if value:
            return _Path(value).expanduser() / "compiled.hef"
    return None


def _v60s_hailo_restore(bound, contract):
    """Disabled compatibility shim for the former coarse pre-key restore.

    Restore is now performed by ``_hailo_build_hef_legacy`` only after the
    compiler ONNX, effective calibration settings, SDK token and canonical
    preprocessing have produced the exact v2 cache key and payload.
    """

    return None


def _v60s_hailo_restored_result(bound, restored):
    result = HailoHefBuildResult(
        ok=True,
        elapsed_s=0.0,
        hw_arch=str(bound.get("hw_arch") or bound.get("target_arch") or bound.get("arch") or "hailo8"),
        net_name=str(bound.get("net_name") or Path(str(bound.get("onnx_path") or "model")).stem),
        backend="artifact_store",
        hef_path=str(restored["destination"]),
        skipped=True,
    )
    calib_info = {
        "source": "artifact_store",
        "compiler_dispatch_count": 0, "cache_hit": True,
        "cache_source": "artifact_store",
        "artifact_id": restored["artifact_id"],
        "contract_hash": restored["contract_hash"],
        "artifact_hash": restored["artifact_hash"],
        "preprocessing_contract": restored.get("build_receipt", {}).get("preprocessing_contract"),
        "preprocessing_contract_sha256": restored.get("build_receipt", {}).get("preprocessing_contract_sha256"),
        "build_receipt": restored.get("build_receipt"),
    }
    result.calib_info = calib_info
    result.details = {
        "compiler_dispatch_count": 0, "cache_hit": True,
        "cache_source": "artifact_store",
        "artifact_store_restore": dict(restored),
    }
    return result


def _v60s_hailo_register(result, bound, contract):
    # Compiler-free probes may materialize their temporary output, but must
    # never register/touch persistent store records or pin references.
    if not bound.get("publish_artifacts", True) or bool(bound.get("cache_only")) or compiler_dispatch_forbidden():
        return

    def record_status(status, error=None):
        detail = dict((result.get("details") if isinstance(result, dict)
                       else getattr(result, "details", None)) or {})
        detail["artifact_store_bundle_backup"] = status
        if error:
            detail["artifact_store_backup_error"] = str(error)
        if isinstance(result, dict):
            result["details"] = detail
        else:
            result.details = detail

    try:
        from .artifact_store import ArtifactStore, artifact_store_enabled
        if not artifact_store_enabled():
            return
        result_ok = result.get("ok") if isinstance(result, dict) else getattr(result, "ok", None)
        if result_ok is not True:
            return
        path = _v60s_hailo_extract_hef(result)
        if path is None:
            destination = _v60s_hailo_destination(bound)
            if destination is not None and destination.is_file():
                path = destination.resolve()
        if path is None:
            return
        source_identity = contract.get("onnx_path")
        expected_source_sha = (
            str(source_identity.get("sha256") or "")
            if isinstance(source_identity, Mapping)
            else ""
        )
        receipt = _load_valid_hailo_receipt(
            path,
            preprocessing_sha256=str(contract.get("preprocessing_contract_sha256") or ""),
            source_onnx_sha256=expected_source_sha,
        )
        if receipt is None:
            record_status("failed", "invalid_hailo_receipt")
            log.warning("[hailo][artifact-store] bundle backup rejected: invalid receipt at %s", path)
            return
        if not _hailo_cache_meta_path(path).is_file():
            # Historical receipt-bearing outputs can be upgraded safely;
            # HEF-only outputs were rejected above and are never resealed.
            destination = _v60s_hailo_destination(bound) or path
            path = _publish_hailo_bundle(
                source_hef=path, destination=destination, receipt=receipt,
                source="historical_receipt_upgrade",
            )
        store = ArtifactStore(os.environ.get("ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT") or None)
        pin = str(os.environ.get("ONNX_SPLITPOINT_ARTIFACT_PIN_FINAL", "0")).lower() in {"1","true","yes","on"}
        store.register_hailo_bundle(source_path=path,
                       receipt_path=_hailo_receipt_path(path),
                       cache_meta_path=_hailo_cache_meta_path(path), contract=contract,
                       metadata={"target": str(bound.get("hw_arch") or bound.get("target_arch") or bound.get("arch") or ""),
                                 "legacy_cache_key": str(receipt.get("cache_key") or ""),
                                 "preprocessing_contract_sha256": str(receipt.get("preprocessing_contract_sha256") or ""),
                                 "build_receipt": receipt,
                                 "bridge": "hailo_build_hef"},
                       source_run=os.environ.get("ONNX_SPLITPOINT_RUN_ID", ""),
                       pin=pin, pin_label="final-campaign" if pin else "")
        record_status("sealed")
    except Exception as exc:
        record_status("failed", f"{type(exc).__name__}: {exc}")
        log.warning("[hailo][artifact-store] atomic bundle backup failed: %s", exc)
        return


def hailo_build_hef(*args, **kwargs):
    kwargs["force"] = parse_config_bool(kwargs.get("force", False), field="hailo_build.force_build")
    kwargs["publish_artifacts"] = parse_config_bool(kwargs.get("publish_artifacts", True), field="hailo_build.publish_artifacts")
    from .hailo_negative_evidence import attach_and_record, build_evidence_scope
    context = kwargs.pop("build_evidence_context", None)
    bound = _v60s_hailo_bound(args, kwargs)
    contract = _v60s_hailo_contract(bound)
    with build_evidence_scope(context, bound):
        result = _hailo_build_hef_legacy(*args, **kwargs)
        result = attach_and_record(result, bound) if bound.get("publish_artifacts", True) else result
        _v60s_hailo_register(result, bound, contract)
    return result


_hailo_build_hef_auto_dispatch = hailo_build_hef_auto


def _v60s_hailo_auto_bound(args, kwargs):
    import inspect as _inspect
    try:
        call = _inspect.signature(_hailo_build_hef_auto_dispatch).bind_partial(*args, **kwargs)
        call.apply_defaults()
        legacy_names = set(_inspect.signature(_hailo_build_hef_legacy).parameters)
        provenance_names = {"wsl_timeout_s", "timeout_s", "compute_device", "gpu_selector", "compute_by_family", "compiler_context"}
        return {key: value for key, value in call.arguments.items() if key in legacy_names | provenance_names}
    except Exception:
        return dict(kwargs)


def _hailo_build_hef_auto_with_attempt(*args, **kwargs):
    """Dispatch a HEF build and retain every terminal compiler attempt."""

    from .hailo_attempt_receipts import (
        begin_hailo_attempt, finalize_hailo_attempt,
    )

    bound = _v60s_hailo_auto_bound(args, kwargs)
    contract = _v60s_hailo_contract(bound)
    outdir = Path(
        bound.get("outdir") or bound.get("out_dir")
        or bound.get("output_dir") or Path.cwd()
    ).expanduser()
    requested_timeout = bound.get("wsl_timeout_s", bound.get("timeout_s", 3600))
    try:
        hard_timeout_s, idle_timeout_s = _resolve_hef_timeout_policy(
            requested_timeout
        )
    except Exception:
        hard_timeout_s, idle_timeout_s = None, None
    receipt_bound = dict(bound)
    net_token = str(bound.get("net_name") or "").strip().lower()
    has_start = bool(bound.get("start_node_names"))
    has_end = bool(bound.get("end_node_names"))
    if "full" in net_token and has_end:
        attempt_endpoint = "raw_head_fallback"
    elif "full" in net_token and not has_start and not has_end:
        attempt_endpoint = "decoded_full"
    elif has_end and not has_start:
        attempt_endpoint = "split_part1"
    elif has_start:
        attempt_endpoint = "split_part2"
    else:
        attempt_endpoint = "full_or_unspecified"
    receipt_bound.update({
        "hard_timeout_s": hard_timeout_s,
        "idle_timeout_s": idle_timeout_s,
        "timeout_s": requested_timeout,
        "endpoint": attempt_endpoint,
        # Activation Part1 is a calibration producer, not the compiler input.
        "compiler_onnx_path": str(bound.get("onnx_path") or ""),
    })
    attempt = begin_hailo_attempt(outdir=outdir, bound=receipt_bound)
    try:
        result = _hailo_build_hef_auto_dispatch(*args, **kwargs)
    except BaseException as exc:
        from .hailo_negative_evidence import current_evidence_info
        attempt["metadata"] = {**dict(attempt.get("metadata") or {}),
                               "build_evidence": current_evidence_info()}
        finalize_hailo_attempt(attempt=attempt, error=exc)
        raise
    from .hailo_negative_evidence import attach_and_record
    result = attach_and_record(result, bound) if bound.get("publish_artifacts", True) else result
    finalize_hailo_attempt(attempt=attempt, result=result)
    _v60s_hailo_register(result, bound, contract)
    return result


def hailo_build_hef_auto(*args, **kwargs):
    """Use shared exact negative evidence for every normal Hailo backend."""
    kwargs["force"] = parse_config_bool(kwargs.get("force", False), field="hailo_build.force_build")
    kwargs["publish_artifacts"] = parse_config_bool(kwargs.get("publish_artifacts", True), field="hailo_build.publish_artifacts")
    if kwargs["force"] and kwargs.get("publish_artifacts", True):
        raise ValueError("force_build_disabled_for_productive_jobs: use reuse_and_build_missing; force is restricted to an isolated publish_artifacts=False diagnostic job")
    publish = parse_config_bool(kwargs.get("publish_artifacts", True), field="hailo_build.publish_artifacts")
    if not publish:
        if not kwargs.get("outdir"):
            raise ValueError("hailo_diagnostic_outdir_required")
        raw_root = Path(kwargs["outdir"]).expanduser().absolute()
        if raw_root.resolve() != raw_root or any(c.isspace() for c in str(raw_root)):
            raise ValueError("hailo_diagnostic_outdir_requires_whitespace_free_path_without_symlinks")
    from .hailo_negative_evidence import build_evidence_scope
    context = kwargs.pop("build_evidence_context", None)
    bound = _v60s_hailo_auto_bound(args, kwargs)
    with build_evidence_scope(context, bound):
        return _hailo_build_hef_auto_with_attempt(*args, **kwargs)


def _with_hailo_evidence_context(function):
    # Direct managed-backend callers also pass the same context into the
    # helper, where the real compiler version is known before SDK import.
    from functools import wraps
    import inspect

    @wraps(function)
    def wrapped(*args, **kwargs):
        kwargs["publish_artifacts"] = parse_config_bool(kwargs.get("publish_artifacts", True), field="hailo_build.publish_artifacts")
        from .hailo_negative_evidence import attach_and_record, build_evidence_scope
        context = kwargs.pop("build_evidence_context", None)
        call = inspect.signature(function).bind_partial(*args, **kwargs)
        call.apply_defaults()
        bound = dict(call.arguments)
        with build_evidence_scope(context, bound):
            result = function(*args, **kwargs)
            return attach_and_record(result, bound) if bound.get("publish_artifacts", True) else result
    return wrapped


hailo_build_hef_via_venv = _with_hailo_evidence_context(hailo_build_hef_via_venv)
hailo_build_hef_via_wsl = _with_hailo_evidence_context(hailo_build_hef_via_wsl)
