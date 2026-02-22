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

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import sys
import time
import logging
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import AttributeProto, helper

# Optional (pure python) helper to resolve multiple DFC versions (Hailo-8 vs Hailo-10)
from .hailo.dfc_manager import get_dfc_manager


log = logging.getLogger(__name__)


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
            base = Path.home() / ".onnx_splitpoint_tool" / "wsl_debug"

        # If the chosen base is not usable, fall back to ~/.onnx_splitpoint_tool/wsl_debug
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception:
            base = Path.home() / ".onnx_splitpoint_tool" / "wsl_debug"
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
        proc = subprocess.run(
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


def hailo_probe_via_venv(
    *,
    hw_arch: str = "hailo8",
    venv_activate: str = "auto",
    timeout_s: int = 30,
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
            out_glibc = subprocess.check_output(["getconf", "GNU_LIBC_VERSION"], text=True, stderr=subprocess.STDOUT).strip()
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
        _ = subprocess.check_output([str(py), "-c", "import pkg_resources"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        try:
            log.info("[hailo][probe][venv] pkg_resources missing -> installing setuptools<82 (self-heal)")
            subprocess.run(
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

    py_probe = (
        "import sys, os; "
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

    cmd = [str(py), "-c", py_probe]

    try:
        log.info("[hailo][probe][venv] hw_arch=%s profile=%s python=%s", hw_arch, profile_id, str(py))
        # Importing hailo_sdk_client can trigger an *interactive* system requirements check
        # on first use ("Continue? [Y/n]"). In a GUI / non-interactive context this would
        # block forever and end in a timeout. We proactively feed "y".
        proc = subprocess.run(
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
    timeout_s: int = 30,
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
            gproc = subprocess.run(
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
        proc = subprocess.run(
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
    backend = (backend or "auto").strip().lower()
    if backend not in ("auto", "local", "wsl", "venv"):
        return HailoProbeResult(ok=False, backend=backend, reason=f"Unknown backend: {backend!r}")

    if backend == "local":
        return hailo_probe_local()
    if backend == "venv":
        return hailo_probe_via_venv(hw_arch=hw_arch, venv_activate=wsl_venv_activate, timeout_s=timeout_s)
    if backend == "wsl":
        return hailo_probe_via_wsl(hw_arch=hw_arch, wsl_distro=wsl_distro, wsl_venv_activate=wsl_venv_activate, timeout_s=timeout_s)

    # auto
    if hailo_sdk_available():
        return hailo_probe_local()

    # Windows: use WSL bridge; Linux: prefer managed venv probe (no need to
    # install hailo_sdk_client into the tool's own Python env).
    if sys.platform == "win32":
        return hailo_probe_via_wsl(hw_arch=hw_arch, wsl_distro=wsl_distro, wsl_venv_activate=wsl_venv_activate, timeout_s=timeout_s)

    return hailo_probe_via_venv(hw_arch=hw_arch, venv_activate=wsl_venv_activate, timeout_s=timeout_s)


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
    helper_win = (Path(__file__).resolve().parent / "wsl_hailo_check.py")
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

        proc = subprocess.run(
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
        _ = subprocess.check_output([str(py), "-c", "import pkg_resources"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        try:
            log.info("[hailo][hef][venv] pkg_resources missing -> installing setuptools<82 (self-heal)")
            subprocess.run(
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
        _ = subprocess.check_output([str(py), "-c", "import pkg_resources"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        try:
            log.info("[hailo][parse][venv] pkg_resources missing -> installing setuptools<82 (self-heal)")
            subprocess.run(
                [str(py), "-m", "pip", "install", "--force-reinstall", "setuptools<82"],
                capture_output=True,
                text=True,
                timeout=min(180, max(10, int(timeout_s))),
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            pass

    helper = Path(__file__).resolve().parent / "wsl_hailo_check.py"
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
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
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
    # WSL bridge settings
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 180,
) -> "HailoParseResult":
    """Convenience wrapper: pick the best available backend.

    backend:
      - "auto": local SDK if importable, else WSL (Windows only)
      - "local": require local SDK
      - "wsl": require WSL bridge
    """

    mode = str(backend or "auto").strip().lower()
    if mode not in {"auto", "local", "wsl", "venv"}:
        mode = "auto"

    if mode in {"auto", "local"} and hailo_sdk_available():
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
        )

    if mode == "venv":
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
            venv_activate=wsl_venv_activate,
            timeout_s=wsl_timeout_s,
        )

    if mode in {"auto", "wsl"}:
        # Linux: auto should prefer the managed venv (no WSL bridge exists).
        if mode == "auto" and sys.platform != "win32":
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
                venv_activate=wsl_venv_activate,
                timeout_s=wsl_timeout_s,
            )
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
            wsl_distro=wsl_distro,
            wsl_venv_activate=wsl_venv_activate,
            wsl_timeout_s=wsl_timeout_s,
        )

    # Nothing available.
    return HailoParseResult(
        ok=False,
        elapsed_s=0.0,
        hw_arch=str(hw_arch),
        net_name=str(net_name or Path(str(onnx_path)).stem),
        error=(
            "No usable Hailo backend available. "
            "Install hailo_sdk_client in this Python env, or configure the WSL backend on Windows."
        ),
    )


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


def fix_onnx_for_hailo(
    model: onnx.ModelProto,
    *,
    add_conv_defaults: bool = True,
) -> Tuple[onnx.ModelProto, Dict[str, Any]]:
    """Apply a small set of pragmatic ONNX fixups that help the Hailo parser.

    This is intentionally conservative: it only fills in some missing Conv/
    ConvTranspose attributes (most commonly `kernel_shape`) and optionally adds
    a few default attributes when they are absent.

    Returns (patched_model, report).
    """

    patched = onnx.ModelProto()
    patched.CopyFrom(model)

    report: Dict[str, Any] = {
        "kernel_shape_patched": 0,
        "conv_defaults_added": 0,
        "notes": [],
    }

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
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            error=f"Hailo SDK not available: {e}",
        )

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
    if net_input_shapes is None:
        try:
            m_tmp = onnx.load(str(model_for_parse))
            net_input_shapes = infer_net_input_shapes_from_model(m_tmp)
        except Exception:
            net_input_shapes = None

    try:
        runner = ClientRunner(hw_arch=str(hw_arch))
        runner.translate_onnx_model(
            model=str(model_for_parse),
            net_name=str(net_name),
            net_input_shapes=net_input_shapes,
            # Keep this on by default: avoids parsing issues with missing RT metadata
            disable_rt_metadata_extraction=bool(disable_rt_metadata_extraction),
        )

        har_path = None
        if save_har and out_dir is not None:
            har_path = str(out_dir / "parsed.har")
            runner.save_har(har_path)

        return HailoParseResult(
            ok=True,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
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
            hw_arch=str(hw_arch),
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


def _clamp_calib_count(shape: List[int], requested: int, *, cap_bytes: int = 256 * 1024 * 1024) -> int:
    """Avoid accidentally allocating multi-GB random calibration sets."""

    n = max(1, int(requested))
    est = _estimate_bytes(shape, n, dtype_bytes=4)
    if est <= 0:
        return n
    if est <= cap_bytes:
        return n
    per = max(1, _estimate_bytes(shape, 1, dtype_bytes=4))
    if per <= 0:
        return n
    n2 = max(1, cap_bytes // per)
    return min(n, int(n2))


def _try_build_calib_from_dir(
    *,
    calib_dir: Path,
    expected_shape: List[int],
    limit: int,
) -> Optional[np.ndarray]:
    if not calib_dir.exists():
        return None
    items = sorted([p for p in calib_dir.iterdir() if p.suffix.lower() in (".npy", ".npz")])
    if not items:
        return None

    batches: List[np.ndarray] = []
    total = 0
    for p in items:
        a = _load_npy_any(p)
        if a.ndim == len(expected_shape):
            a = a[None, ...]
        if a.ndim != len(expected_shape) + 1:
            continue

        # Convert dtype to float32 (Hailo optimize typically expects float)
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        else:
            a = a.astype(np.float32, copy=False)

        # Try to match shape; special-case NCHW->NHWC for 3D image shapes
        sample = list(a.shape[1:])
        tgt = list(expected_shape)
        if sample == tgt:
            pass
        elif len(tgt) == 3 and sample == [tgt[2], tgt[0], tgt[1]]:
            # NCHW -> NHWC
            a = np.transpose(a, (0, 2, 3, 1))
        else:
            continue

        batches.append(np.ascontiguousarray(a))
        total += int(a.shape[0])
        if total >= int(limit):
            break

    if not batches:
        return None

    ds = np.concatenate(batches, axis=0)
    if ds.shape[0] > int(limit):
        ds = ds[: int(limit)]
    return np.ascontiguousarray(ds.astype(np.float32, copy=False))


def hailo_build_hef(
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
    force: bool = False,
    keep_artifacts: bool = False,
) -> HailoHefBuildResult:
    """Translate + optimize + compile an ONNX to a HEF.

    Notes
    -----
    - This function requires `hailo_sdk_client` (DFC) to be importable.
    - If `calib_dir` is not provided or cannot be used, a *random* calibration
      set is generated based on HN input-layer shapes.
    """

    t0 = time.time()
    onnx_path = Path(onnx_path)
    if net_name is None:
        net_name = onnx_path.stem

    out_dir = Path(outdir) if outdir is not None else onnx_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    hef_path = out_dir / "compiled.hef"
    if hef_path.exists() and not bool(force):
        return HailoHefBuildResult(
            ok=True,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            hef_path=str(hef_path),
            skipped=True,
        )

    try:
        from hailo_sdk_client import ClientRunner
    except Exception as e:
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            error=f"Hailo SDK not available: {e}",
        )

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

    if net_input_shapes is None:
        try:
            m_tmp = onnx.load(str(model_for_parse))
            net_input_shapes = infer_net_input_shapes_from_model(m_tmp)
        except Exception:
            net_input_shapes = None

    try:
        runner = ClientRunner(hw_arch=str(hw_arch))
        runner.translate_onnx_model(
            model=str(model_for_parse),
            net_name=str(net_name),
            net_input_shapes=net_input_shapes,
            disable_rt_metadata_extraction=bool(disable_rt_metadata_extraction),
        )

        parsed_har = out_dir / "parsed.har"
        if keep_artifacts:
            try:
                runner.save_har(str(parsed_har))
            except Exception:
                pass

        # Build calibration dataset
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

        # Decide an effective calib_count that won't explode memory
        eff_count = int(calib_count)
        for shp in expected_shapes.values():
            eff_count = min(eff_count, _clamp_calib_count(shp, int(calib_count)))
        eff_count = max(1, eff_count)

        calib_dir_p = Path(calib_dir).expanduser().resolve() if calib_dir else None
        used_dir = False
        if calib_dir_p is not None and calib_dir_p.exists() and len(input_layers) == 1:
            # For now, only support directory calibration for single-input networks.
            in0 = input_layers[0]
            ds = _try_build_calib_from_dir(calib_dir=calib_dir_p, expected_shape=expected_shapes[in0], limit=eff_count)
            if ds is not None:
                calib_inputs[in0] = ds
                used_dir = True

        if not used_dir:
            rng = np.random.default_rng(0)
            for in_name in input_layers:
                shp = expected_shapes[in_name]
                ds = rng.random((eff_count, *shp), dtype=np.float32)
                calib_inputs[in_name] = np.ascontiguousarray(ds)

        # Determine batch size
        bs = max(1, min(int(calib_batch_size), int(eff_count)))
        calib_meta["source"] = str(calib_dir_p) if used_dir and calib_dir_p is not None else "random"
        calib_meta["used_count"] = int(eff_count)
        calib_meta["batch_size"] = int(bs)
        for k, shp in expected_shapes.items():
            calib_meta["inputs"][k] = {"shape": list(shp)}

        model_script = (
            f"model_optimization_flavor(optimization_level={int(opt_level)}, batch_size={int(bs)})\n"
            f"model_optimization_config(calibration, batch_size={int(bs)}, calibset_size={int(eff_count)})\n"
        )
        runner.load_model_script(model_script)

        runner.optimize(calib_inputs)

        quant_har = out_dir / "quantized.har"
        if keep_artifacts:
            try:
                runner.save_har(str(quant_har))
            except Exception:
                pass

        hef_bytes = runner.compile()
        hef_path.write_bytes(hef_bytes)

        return HailoHefBuildResult(
            ok=True,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            hef_path=str(hef_path),
            parsed_har_path=(str(parsed_har) if keep_artifacts else None),
            quant_har_path=(str(quant_har) if keep_artifacts else None),
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
            skipped=False,
            calib_info=calib_meta,
        )

    except Exception as e:
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
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="local",
            error=err,
            fixed_onnx_path=str(fixed_path) if fixed_path is not None else None,
            fixup_report=fixup_report,
        )


def hailo_build_hef_via_wsl(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    disable_rt_metadata_extraction: bool = True,
    opt_level: int = 1,
    calib_dir: Optional[Union[str, Path]] = None,
    calib_count: int = 64,
    calib_batch_size: int = 8,
    force: bool = False,
    keep_artifacts: bool = False,
    # WSL bridge settings
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 3600,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> HailoHefBuildResult:
    """Build a HEF inside WSL (Windows host -> WSL2 backend)."""

    t0 = time.time()
    onnx_path = Path(onnx_path)
    if net_name is None:
        net_name = onnx_path.stem

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

    helper_win = (Path(__file__).resolve().parent / "wsl_hailo_build_hef.py")
    helper_wsl = windows_path_to_wsl(str(helper_win))

    onnx_wsl = windows_path_to_wsl(str(onnx_path.resolve()))
    outdir_wsl = None
    if outdir is not None:
        outdir_wsl = windows_path_to_wsl(str(Path(outdir).resolve()))
    calib_wsl = None
    if calib_dir is not None:
        calib_wsl = windows_path_to_wsl(str(Path(calib_dir).resolve()))

    venv_activate = venv_eff  # do not quote '~'

    cmd = (
        "set -e; "
        "echo __SPLITPOINT_WSL_BEGIN__; "
        f"source {venv_activate}; "
        "echo __SPLITPOINT_WSL_VENV_OK__; "
        "export PYTHONUNBUFFERED=1; "
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
        f" --keep-artifacts {'1' if keep_artifacts else '0'}"
    )
    if outdir_wsl is not None:
        cmd += f" --outdir {_bash_quote(outdir_wsl)}"
    if calib_wsl is not None:
        cmd += f" --calib-dir {_bash_quote(calib_wsl)}"

    wsl_cmd: List[str] = [_wsl_exe()]
    if distro_eff:
        wsl_cmd += ["-d", str(distro_eff)]
    wsl_cmd += ["--", "bash", "-lc", cmd]

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    timed_out = False

    def _emit(stream_name: str, line: str) -> None:
        if on_log is None:
            return
        try:
            on_log(stream_name, line)
        except Exception:
            # Never let UI callbacks crash compilation.
            return

    try:
        log.info(
            "[hailo][hef][wsl] hw_arch=%s profile=%s distro=%s activate=%s onnx=%s outdir=%s",
            hw_arch,
            resolved.profile_id,
            distro_eff or "",
            venv_eff,
            onnx_wsl,
            outdir_wsl or "",
        )
        log.debug("[hailo][hef][wsl] cmd=%s", wsl_cmd)

        popen = subprocess.Popen(
            wsl_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
        )

        # The first DFC invocation may prompt for a system requirements check.
        # Feed a default "yes" so the process never blocks in a GUI context.
        try:
            if popen.stdin is not None:
                popen.stdin.write("y\n")
                popen.stdin.flush()
        except Exception:
            pass

        def _reader(stream, sink: List[str], stream_name: str) -> None:
            if stream is None:
                return
            try:
                for raw in iter(stream.readline, ""):
                    if raw == "" and popen.poll() is not None:
                        break
                    line = raw.rstrip("\n")
                    if line.endswith("\r"):
                        line = line.rstrip("\r")
                    line = _sanitize_wsl_text(line)
                    sink.append(line)
                    _emit(stream_name, line)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t_out = threading.Thread(target=_reader, args=(popen.stdout, stdout_lines, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(popen.stderr, stderr_lines, "stderr"), daemon=True)
        t_out.start()
        t_err.start()
        try:
            popen.wait(timeout=float(wsl_timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                popen.kill()
            except Exception:
                pass
            try:
                popen.wait(timeout=10)
            except Exception:
                pass
        t_out.join(timeout=2)
        t_err.join(timeout=2)
    except Exception as e:
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=f"WSL HEF build failed to launch: {type(e).__name__}: {e}",
        )

    stdout = _sanitize_wsl_text("\n".join(stdout_lines))
    stderr = _sanitize_wsl_text("\n".join(stderr_lines))
    rc = int(getattr(popen, "returncode", 0) or 0)

    if timed_out:
        dbg_path = _write_wsl_debug_log(
            outdir,
            filename=f"hailo_wsl_hef_timeout_{hw_arch}_{net_name}_{int(time.time())}.log",
            wsl_cmd=wsl_cmd,
            stdout=stdout,
            stderr=stderr,
        )
        err = f"WSL HEF build timed out after {wsl_timeout_s}s."
        if dbg_path:
            err += f"\n\n[debug_log] {dbg_path}"
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend="wsl",
            error=err,
            returncode=124,
            debug_log=dbg_path,
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
        return HailoHefBuildResult(
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

    return HailoHefBuildResult(
        ok=bool(payload.get("ok")),
        elapsed_s=float(payload.get("elapsed_s", time.time() - t0)),
        hw_arch=str(payload.get("hw_arch", hw_arch)),
        net_name=str(payload.get("net_name", net_name)),
        backend=str(payload.get("backend") or "wsl"),
        error=payload.get("error"),
        hef_path=payload.get("hef_path"),
        parsed_har_path=payload.get("parsed_har_path"),
        quant_har_path=payload.get("quant_har_path"),
        fixed_onnx_path=payload.get("fixed_onnx_path"),
        fixup_report=payload.get("fixup_report"),
        skipped=bool(payload.get("skipped", False)),
        calib_info=payload.get("calib_info"),
    )


def hailo_build_hef_via_venv(
    onnx_path: Union[str, Path],
    *,
    hw_arch: str = "hailo8",
    net_name: Optional[str] = None,
    outdir: Optional[Union[str, Path]] = None,
    fixup: bool = True,
    add_conv_defaults: bool = True,
    disable_rt_metadata_extraction: bool = True,
    opt_level: int = 1,
    calib_dir: Optional[Union[str, Path]] = None,
    calib_count: int = 64,
    calib_batch_size: int = 8,
    force: bool = False,
    keep_artifacts: bool = False,
    venv_activate: str = "auto",
    timeout_s: int = 3600,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> HailoHefBuildResult:
    """Build a HEF inside a managed DFC venv (Linux / WSL)."""

    t0 = time.time()
    onnx_path = Path(str(onnx_path)).expanduser().resolve()
    net_name_eff = str(net_name or onnx_path.stem)
    outdir_path = Path(str(outdir)).expanduser().resolve() if outdir else None
    if outdir_path is not None:
        outdir_path.mkdir(parents=True, exist_ok=True)

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

    helper = Path(__file__).resolve().parent / "wsl_hailo_build_hef.py"
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
        "--keep-artifacts",
        "1" if keep_artifacts else "0",
    ]
    if outdir_path is not None:
        cmd += ["--outdir", str(outdir_path)]
    if calib_dir is not None:
        cmd += ["--calib-dir", str(Path(str(calib_dir)).expanduser().resolve())]

    stdout_lines: List[str] = []
    stderr_lines: List[str] = []
    timed_out = False

    def _emit(stream_name: str, line: str) -> None:
        if on_log is None:
            return
        try:
            on_log(stream_name, line)
        except Exception:
            return

    try:
        log.info(
            "[hailo][hef][venv] hw_arch=%s profile=%s python=%s onnx=%s outdir=%s",
            hw_arch,
            profile_id,
            str(py),
            str(onnx_path),
            str(outdir_path or ""),
        )
        log.debug("[hailo][hef][venv] cmd=%s", cmd)

        popen = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
        )

        # The first DFC invocation may prompt for a system requirements check.
        # Feed a default "yes" so the process never blocks in a GUI context.
        try:
            if popen.stdin is not None:
                popen.stdin.write("y\n")
                popen.stdin.flush()
        except Exception:
            pass

        def _reader(stream, sink: List[str], stream_name: str) -> None:
            if stream is None:
                return
            try:
                for raw in iter(stream.readline, ""):
                    if raw == "" and popen.poll() is not None:
                        break
                    line = raw.rstrip("\n")
                    if line.endswith("\r"):
                        line = line.rstrip("\r")
                    line = _sanitize_wsl_text(line)
                    sink.append(line)
                    _emit(stream_name, line)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t_out = threading.Thread(target=_reader, args=(popen.stdout, stdout_lines, "stdout"), daemon=True)
        t_err = threading.Thread(target=_reader, args=(popen.stderr, stderr_lines, "stderr"), daemon=True)
        t_out.start()
        t_err.start()
        try:
            popen.wait(timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                popen.kill()
            except Exception:
                pass
            try:
                popen.wait(timeout=10)
            except Exception:
                pass
        t_out.join(timeout=2)
        t_err.join(timeout=2)
    except Exception as e:
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=f"Venv HEF build failed to launch: {type(e).__name__}: {e}",
        )

    stdout = _sanitize_wsl_text("\n".join(stdout_lines))
    stderr = _sanitize_wsl_text("\n".join(stderr_lines))
    rc = int(getattr(popen, "returncode", 0) or 0)

    if timed_out:
        dbg_path = _write_wsl_debug_log(
            str(outdir_path) if outdir_path is not None else None,
            filename=f"hailo_venv_hef_timeout_{hw_arch}_{net_name_eff}_{int(time.time())}.log",
            wsl_cmd=cmd,
            stdout=stdout,
            stderr=stderr,
        )
        err = f"Venv HEF build timed out after {timeout_s}s."
        if dbg_path:
            err += f"\n\n[debug_log] {dbg_path}"
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=err,
            returncode=124,
            debug_log=dbg_path,
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
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch),
            net_name=net_name_eff,
            backend="venv",
            error=(
                "Venv HEF build did not return a structured result. "
                f"exit_code={rc}. tail=\n{tail}\n\n"
                "Details were written to gui.log (Logs tab)."
            ),
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

    return HailoHefBuildResult(
        ok=bool(payload.get("ok")),
        elapsed_s=float(payload.get("elapsed_s", time.time() - t0)),
        hw_arch=str(payload.get("hw_arch", hw_arch)),
        net_name=str(payload.get("net_name", net_name_eff)),
        backend="venv",
        error=payload.get("error"),
        hef_path=payload.get("hef_path"),
        parsed_har_path=payload.get("parsed_har_path"),
        quant_har_path=payload.get("quant_har_path"),
        fixed_onnx_path=payload.get("fixed_onnx_path"),
        fixup_report=payload.get("fixup_report"),
        skipped=bool(payload.get("skipped", False)),
        calib_info=payload.get("calib_info"),
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
    force: bool = False,
    keep_artifacts: bool = False,
    # WSL bridge
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 3600,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> HailoHefBuildResult:
    mode = str(backend or "auto").strip().lower()
    if mode not in {"auto", "local", "wsl", "venv"}:
        mode = "auto"

    if mode in {"auto", "local"} and hailo_sdk_available():
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
            force=bool(force),
            keep_artifacts=bool(keep_artifacts),
        )

    if mode == "venv":
        return hailo_build_hef_via_venv(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            opt_level=int(opt_level),
            calib_dir=calib_dir,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            force=bool(force),
            keep_artifacts=bool(keep_artifacts),
            venv_activate=wsl_venv_activate,
            timeout_s=int(wsl_timeout_s),
        )

    if mode in {"auto", "wsl"}:
        # Linux: auto should prefer the managed venv.
        if mode == "auto" and sys.platform != "win32":
            return hailo_build_hef_via_venv(
                onnx_path,
                hw_arch=hw_arch,
                net_name=net_name,
                outdir=outdir,
                fixup=fixup,
                add_conv_defaults=add_conv_defaults,
                disable_rt_metadata_extraction=disable_rt_metadata_extraction,
                opt_level=int(opt_level),
                calib_dir=calib_dir,
                calib_count=int(calib_count),
                calib_batch_size=int(calib_batch_size),
                force=bool(force),
                keep_artifacts=bool(keep_artifacts),
                venv_activate=wsl_venv_activate,
                timeout_s=int(wsl_timeout_s),
                on_log=on_log,
            )
        return hailo_build_hef_via_wsl(
            onnx_path,
            hw_arch=hw_arch,
            net_name=net_name,
            outdir=outdir,
            fixup=fixup,
            add_conv_defaults=add_conv_defaults,
            disable_rt_metadata_extraction=disable_rt_metadata_extraction,
            opt_level=int(opt_level),
            calib_dir=calib_dir,
            calib_count=int(calib_count),
            calib_batch_size=int(calib_batch_size),
            force=bool(force),
            keep_artifacts=bool(keep_artifacts),
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
            "Install hailo_sdk_client in this Python env, or configure the WSL backend on Windows."
        ),
    )
