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

import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import signal
import threading
import sys
import time
import logging
import platform
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import AttributeProto, helper

# Optional (pure python) helper to resolve multiple DFC versions (Hailo-8 vs Hailo-10)
from .hailo.backend_mode import auto_prefers_subprocess, normalize_hailo_backend, subprocess_backend_for_platform
from .runners.backends.hailo_utils import get_dfc_manager


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
    env = os.environ.copy()
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


def _hailo_stage_from_line(line: str) -> Optional[str]:
    s = str(line or '').strip().lower()
    if not s:
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
    if 'activation calibration' in s or ('calib' in s and 'part1' in s):
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


def _resolve_hef_timeout_policy(requested_timeout_s: int) -> Tuple[int, Optional[int]]:
    """Return ``(hard_timeout_s, idle_timeout_s)`` for HEF helper processes.

    Legacy GUI code passes ``3600`` unconditionally. Treat that as the old
    default and upgrade it to a safer 3h hard timeout unless the user overrides
    it via environment variables.
    """

    env_hard = _env_int_first_positive('ONNX_SPLITPOINT_HAILO_HEF_TIMEOUT_S', 'OSP_HAILO_HARD_TIMEOUT_S')
    env_idle = _env_int_first_positive('ONNX_SPLITPOINT_HAILO_HEF_IDLE_TIMEOUT_S', 'OSP_HAILO_IDLE_TIMEOUT_S')

    try:
        requested = int(requested_timeout_s)
    except Exception:
        requested = 0

    if env_hard is not None:
        hard_timeout_s = int(env_hard)
    elif requested <= 0:
        hard_timeout_s = 10800
    elif requested == 3600:
        hard_timeout_s = 10800
    else:
        hard_timeout_s = max(60, int(requested))

    if env_idle is not None:
        idle_timeout_s: Optional[int] = int(env_idle)
    else:
        idle_timeout_s = min(3600, max(1800, int(hard_timeout_s // 3)))

    if idle_timeout_s is not None and idle_timeout_s <= 0:
        idle_timeout_s = None
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
        proc = subprocess.run(
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
    if proc.poll() is not None:
        return

    if os.name == 'nt':
        try:
            subprocess.run(
                ['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        return

    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass

    deadline = time.monotonic() + max(0.1, float(grace_s))
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)

    if proc.poll() is not None:
        return

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


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

    def _emit(stream_name: str, line: str) -> None:
        if on_log is None:
            return
        try:
            on_log(stream_name, line)
        except Exception:
            return

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
    timeout_kind: Optional[str] = None
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.monotonic()
        if hard_timeout_s is not None and hard_timeout_s > 0 and (now - t0) > float(hard_timeout_s):
            timed_out = True
            timeout_kind = 'hard'
            break
        if idle_timeout_s is not None and idle_timeout_s > 0 and (now - float(state['last_output_ts'])) > float(idle_timeout_s):
            timed_out = True
            timeout_kind = 'idle'
            break
        time.sleep(0.2)

    if timed_out:
        _kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
    else:
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    t_out.join(timeout=2)
    t_err.join(timeout=2)

    return _StreamedSubprocessResult(
        returncode=int(getattr(proc, 'returncode', 0) or 0),
        stdout=_sanitize_wsl_text('\n'.join(stdout_lines)),
        stderr=_sanitize_wsl_text('\n'.join(stderr_lines)),
        timed_out=bool(timed_out),
        timeout_kind=timeout_kind,
        last_stage=(str(state.get('last_stage')) if state.get('last_stage') else None),
        stage_history=list(state.get('stage_history') or []),
        elapsed_s=float(time.monotonic() - t0),
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

        hailo_log_cwd = ensure_dir(splitpoint_logs_dir() / "hailo_sdk" / str(profile_id))

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
        env = dict(os.environ)
        # HailoRT can be configured to write logs to a single file.
        env.setdefault("HAILORT_LOGGER_PATH", str(hailo_log_cwd / "hailort.log"))

        proc = subprocess.run(
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
    if net_input_shapes is None:
        try:
            m_tmp = onnx.load(str(model_for_parse))
            net_input_shapes = infer_net_input_shapes_from_model(m_tmp)
        except Exception:
            net_input_shapes = None

    try:
        runner = ClientRunner(hw_arch=str(hw_arch_eff))
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
    if last_stage and not body.get("last_stage"):
        body["last_stage"] = str(last_stage)

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
    calib_scan = _scan_calib_dir(calib_dir, recursive=True, limit=int(limit))
    items = list(calib_scan.get('items') or [])
    if not items:
        return None

    batches: List[np.ndarray] = []
    total = 0
    tgt = [int(v) for v in expected_shape]
    for p in items:
        try:
            a = _load_calib_item_any(p)
        except Exception:
            continue

        a = np.asarray(a)
        if len(tgt) == 3 and a.ndim in (2, 3, 4):
            try:
                a = _resize_hwc_image(a, tgt[0], tgt[1], target_c=tgt[2])[None, ...]
            except Exception:
                pass

        if a.ndim == len(tgt):
            a = a[None, ...]
        if a.ndim != len(tgt) + 1:
            continue

        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        else:
            a = a.astype(np.float32, copy=False)

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
            if op_type in _HAILO_PART2_HARD_PARSER_BLOCKER_OPS:
                hard_blockers.append(rec)
            elif op_type in _HAILO_PART2_SOFT_PARSER_BLOCKER_OPS:
                soft_blockers.append(rec)
            if op_type == 'Transpose' and name:
                transpose_candidates.append((int(idx), name))

        blocked_names = [str(rec.get('name') or '') for rec in hard_blockers if str(rec.get('name') or '')]
        dominant_prefix = _dominant_parser_blocked_prefix(blocked_names)
        first_blocked_idx = min((int(rec.get('index', 0)) for rec in hard_blockers), default=None)
        suggested_end_nodes: List[str] = []
        if dominant_prefix and first_blocked_idx is not None:
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


def _prepare_part1_input_for_activation_calib(arr: np.ndarray, inp: Any) -> np.ndarray:
    tgt_h, tgt_w, tgt_c, layout = _ort_input_hwc_spec(inp)
    x = _resize_hwc_image(arr, tgt_h, tgt_w, target_c=tgt_c)

    exp_type = str(getattr(inp, 'type', '') or '').lower()
    if 'float' in exp_type:
        x = x.astype(np.float32)
        if x.size and float(np.nanmax(x)) > 1.5:
            x = x / 255.0
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


def _build_activation_calib_from_part1_onnx(
    *,
    part1_onnx: Path,
    part2_onnx: Path,
    calib_dir: Path,
    limit: int,
    gen_batch: int,
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, Any]]:
    try:
        import onnxruntime as ort  # type: ignore
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
    p1_sess = ort.InferenceSession(str(part1_onnx), sess_options=so, providers=['CPUExecutionProvider'])
    p2_sess = ort.InferenceSession(str(part2_onnx), providers=['CPUExecutionProvider'])

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
        'part1_input': {'name': p1_in_name, 'type': str(getattr(p1_in, 'type', '') or ''), 'shape': list(getattr(p1_in, 'shape', []) or [])},
        'mapping_part2in_to_part1out': dict(mapping),
        'mapping_debug': mapping_debug,
        'part2_inputs': {m.name: {'type': str(getattr(m, 'type', '') or ''), 'shape': [d if isinstance(d, int) else None for d in (getattr(m, 'shape', []) or [])]} for m in p2_inputs},
        'calib_source_dir': str(calib_dir),
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
    while idx < N:
        batch_paths = calib_items[idx: idx + gen_batch]
        xs: List[np.ndarray] = []
        for pp in batch_paths:
            arr = _load_calib_item_any(pp)
            xs.append(_prepare_part1_input_for_activation_calib(arr, p1_in))
        xb = np.concatenate(xs, axis=0) if len(xs) > 1 else xs[0]
        outs = p1_sess.run(req_out_names, {p1_in_name: xb})
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
    activation_part1_onnx: Optional[Union[str, Path]] = None,
    activation_gen_batch: int = 8,
    force: bool = False,
    keep_artifacts: bool = False,
    extra_model_script: Optional[str] = None,
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

    t0 = time.time()
    onnx_path = Path(onnx_path)
    if net_name is None:
        net_name = onnx_path.stem

    hw_arch_eff = _normalize_hailo_hw_arch(hw_arch)
    if hw_arch_eff != str(hw_arch or "").strip().lower():
        log.warning("[hailo] hw_arch alias: '%s' -> '%s'", str(hw_arch), hw_arch_eff)

    out_dir = Path(outdir) if outdir is not None else onnx_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    activation_part1_onnx_p = Path(activation_part1_onnx).expanduser().resolve() if activation_part1_onnx else None
    calib_dir_p = Path(calib_dir).expanduser().resolve() if calib_dir else None

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
    if hef_path.exists() and not bool(force):
        return HailoHefBuildResult(
            ok=True,
            elapsed_s=time.time() - t0,
            hw_arch=str(hw_arch_eff),
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
            hw_arch=str(hw_arch_eff),
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
        runner = ClientRunner(hw_arch=str(hw_arch_eff))
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
                    activation_part2_names, calib_by_part2_input, activation_debug = _build_activation_calib_from_part1_onnx(
                        part1_onnx=activation_part1_onnx_p,
                        part2_onnx=model_for_parse,
                        calib_dir=calib_dir_p,
                        limit=eff_count,
                        gen_batch=max(1, int(activation_gen_batch)),
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
                ds = _try_build_calib_from_dir(calib_dir=calib_dir_p, expected_shape=expected_shapes[in0], limit=eff_count)
                if ds is not None:
                    calib_inputs[in0] = ds
                    used_dir = True

        if not used_dir and not used_activation:
            rng = np.random.default_rng(0)
            for in_name in input_layers:
                shp = expected_shapes[in_name]
                ds = rng.random((eff_count, *shp), dtype=np.float32)
                calib_inputs[in_name] = np.ascontiguousarray(ds)

        # Determine batch size
        bs = max(1, min(int(calib_batch_size), int(eff_count)))
        if used_activation:
            calib_meta['source'] = 'activation_from_part1'
            calib_meta['activation_part1_onnx'] = str(activation_part1_onnx_p) if activation_part1_onnx_p is not None else None
            calib_meta['activation_gen_batch'] = int(max(1, int(activation_gen_batch)))
            if activation_debug is not None:
                calib_meta['activation_debug_path'] = str(out_dir / 'calib_activations_shapes.json')
        else:
            calib_meta['source'] = str(calib_dir_p) if used_dir and calib_dir_p is not None else 'random'
        calib_meta['used_count'] = int(next(iter(calib_inputs.values())).shape[0]) if calib_inputs else int(eff_count)
        calib_meta['batch_size'] = int(bs)
        for k, shp in expected_shapes.items():
            calib_meta['inputs'][k] = {'shape': list(shp)}

        model_script = (
            f"model_optimization_flavor(optimization_level={int(opt_level)}, batch_size={int(bs)})\n"
            f"model_optimization_config(calibration, batch_size={int(bs)}, calibset_size={int(eff_count)})\n"
        )
        extra_script = str(extra_model_script or "").strip()
        if extra_script:
            model_script += extra_script
            if not model_script.endswith("\n"):
                model_script += "\n"
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
            hw_arch=str(hw_arch_eff),
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
    activation_part1_onnx: Optional[Union[str, Path]] = None,
    activation_gen_batch: int = 8,
    force: bool = False,
    keep_artifacts: bool = False,
    extra_model_script: Optional[str] = None,
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

    hard_timeout_s, idle_timeout_s = _resolve_hef_timeout_policy(int(wsl_timeout_s))

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
    activation_part1_onnx: Optional[Union[str, Path]] = None,
    activation_gen_batch: int = 8,
    force: bool = False,
    keep_artifacts: bool = False,
    extra_model_script: Optional[str] = None,
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

    hard_timeout_s, idle_timeout_s = _resolve_hef_timeout_policy(int(timeout_s))

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
        "--keep-artifacts",
        "1" if keep_artifacts else "0",
    ]
    if outdir_path is not None:
        cmd += ["--outdir", str(outdir_path)]
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

        hailo_log_cwd = ensure_dir(splitpoint_logs_dir() / "hailo_sdk" / str(profile_id))

        # Best-effort log retention for Hailo SDK logs.
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
        env = dict(os.environ)
        env.setdefault("HAILORT_LOGGER_PATH", str(hailo_log_cwd / "hailort.log"))
        env.setdefault("ONNX_SPLITPOINT_HAILO_HELPER_BACKEND", "venv")

        run = _run_streamed_subprocess(
            cmd,
            cwd=str(hailo_log_cwd),
            env=env,
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
            net_name=net_name_eff,
            backend="venv",
            error=f"Venv HEF build failed to launch: {type(e).__name__}: {e}",
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
        return _make_hef_result(
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
    keep_artifacts: bool = False,
    extra_model_script: Optional[str] = None,
    # WSL bridge
    wsl_distro: Optional[str] = None,
    wsl_venv_activate: str = "auto",
    wsl_timeout_s: int = 3600,
    on_log: Optional[Callable[[str, str], None]] = None,
) -> HailoHefBuildResult:
    mode = normalize_hailo_backend(backend)
    if mode == "subprocess":
        mode = subprocess_backend_for_platform()

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
            keep_artifacts=bool(keep_artifacts),
            extra_model_script=extra_model_script,
        )

    def _run_venv() -> HailoHefBuildResult:
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
            activation_part1_onnx=activation_part1_onnx,
            activation_gen_batch=int(activation_gen_batch),
            force=bool(force),
            keep_artifacts=bool(keep_artifacts),
            extra_model_script=extra_model_script,
            venv_activate=wsl_venv_activate,
            timeout_s=int(wsl_timeout_s),
            on_log=on_log,
        )

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
            keep_artifacts=bool(keep_artifacts),
            extra_model_script=extra_model_script,
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
            keep_artifacts=bool(keep_artifacts),
            extra_model_script=extra_model_script,
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
