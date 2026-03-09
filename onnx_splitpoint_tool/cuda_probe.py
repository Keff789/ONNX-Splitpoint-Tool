"""CUDA environment probing for Hailo DFC (TensorFlow/XLA).

Why this exists
--------------
The Hailo Dataflow Compiler (DFC) can use TensorFlow/XLA GPU kernels during
optimization. On many systems (especially WSL or "driver-only" installs),
an NVIDIA GPU may be present but the full CUDA toolkit (specifically
``nvvm/libdevice`` bitcode) is missing. In that case the DFC often fails with
errors like:

- ``libdevice is required ... but was not found``
- PTX compilation failures / driver fallback

This module provides:

1) A **probe** that checks for a usable NVIDIA GPU *and* a CUDA toolkit that
   contains the required ``libdevice*.bc`` files.
2) An **auto-configure** helper that, when CUDA is incomplete, forces CPU by
   setting ``CUDA_VISIBLE_DEVICES=-1``.

The main tool can run without any Hailo SDK installed; therefore this module is
pure stdlib and safe to import in all environments.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _uniq_paths(paths: List[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        try:
            s = str(p)
        except Exception:
            continue
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(p)
    return out


def _run_quick(cmd: List[str], *, timeout_s: float = 2.0) -> Tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return int(proc.returncode), out.strip()
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:
        return 127, f"{type(e).__name__}: {e}"


def _probe_nvidia_smi() -> Dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"found": False, "ok": False, "gpus": [], "error": "nvidia-smi not found"}
    rc, out = _run_quick([exe, "-L"], timeout_s=2.5)
    if rc != 0:
        return {"found": True, "ok": False, "gpus": [], "error": out or f"nvidia-smi rc={rc}"}
    gpus = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    ok = any("GPU" in ln for ln in gpus) or bool(gpus)
    return {"found": True, "ok": bool(ok), "gpus": gpus[:8], "error": ""}


def _candidate_cuda_roots() -> List[Path]:
    roots: List[Path] = []

    # 1) Environment variables
    for k in ("CUDA_HOME", "CUDA_PATH", "CUDA_DIR"):
        v = (os.environ.get(k) or "").strip()
        if v:
            roots.append(Path(v).expanduser())

    # 2) Standard toolkit locations
    roots.append(Path("/usr/local/cuda"))
    try:
        roots.extend(sorted(Path("/usr/local").glob("cuda-*")))
    except Exception:
        pass

    # 3) NVIDIA pip packages sometimes ship a minimal nvcc tree
    #    (nvidia/cuda_nvcc). If present and contains nvvm/libdevice it can be
    #    enough for XLA.
    try:
        for sp in list(sys.path):
            if not sp:
                continue
            p = Path(sp)
            cand = p / "nvidia" / "cuda_nvcc"
            if cand.exists():
                roots.append(cand)
    except Exception:
        pass

    return _uniq_paths([p for p in roots if p is not None])


def _find_libdevice(cuda_root: Path) -> Optional[Path]:
    try:
        d = cuda_root / "nvvm" / "libdevice"
        if d.is_dir():
            # Most common: libdevice.10.bc, but accept any libdevice*.bc
            for f in sorted(d.glob("libdevice*.bc")):
                if f.is_file():
                    return f
    except Exception:
        pass
    # Rare fallback: directly in the root
    try:
        for f in sorted(cuda_root.glob("libdevice*.bc")):
            if f.is_file():
                return f
    except Exception:
        pass
    return None


def probe_cuda_environment() -> Dict[str, Any]:
    """Probe whether GPU acceleration is likely to work for TF/XLA.

    Returns a JSON-serializable dict.

    The main decision points are:
    - Is an NVIDIA GPU accessible?  (nvidia-smi)
    - Is a CUDA toolkit with nvvm/libdevice available?
    """

    info: Dict[str, Any] = {}

    # User override is always respected.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    info["cuda_visible_devices"] = cvd
    info["user_override"] = cvd is not None

    smi = _probe_nvidia_smi()
    info["nvidia_smi"] = smi

    roots = _candidate_cuda_roots()
    info["cuda_roots_checked"] = [str(p) for p in roots[:12]]

    found_root: Optional[Path] = None
    found_libdevice: Optional[Path] = None
    for r in roots:
        lib = _find_libdevice(r)
        if lib is not None:
            found_root = r
            found_libdevice = lib
            break

    info["cuda_root"] = str(found_root) if found_root else None
    info["libdevice_path"] = str(found_libdevice) if found_libdevice else None

    ptxas = None
    if found_root is not None:
        try:
            cand = found_root / "bin" / "ptxas"
            if cand.is_file():
                ptxas = cand
        except Exception:
            ptxas = None
    info["ptxas_path"] = str(ptxas) if ptxas else None

    # Determine "GPU-usable" (auto) based on both checks.
    gpu_ok = bool(smi.get("ok")) and bool(found_libdevice)
    info["gpu_ok"] = bool(gpu_ok)

    # Human summary (kept short for GUI labels).
    if cvd is not None:
        if str(cvd).strip() == "-1":
            info["summary"] = "Compute: CPU (forced by CUDA_VISIBLE_DEVICES=-1)"
        else:
            info["summary"] = f"Compute: GPU (user override CUDA_VISIBLE_DEVICES={cvd})"
    else:
        if gpu_ok:
            info["summary"] = "Compute: GPU (auto)"
        else:
            # Provide the most useful reason.
            if not bool(smi.get("found")):
                info["summary"] = "Compute: CPU (auto: no NVIDIA GPU detected)"
            elif not bool(smi.get("ok")):
                info["summary"] = "Compute: CPU (auto: NVIDIA GPU not usable)"
            else:
                info["summary"] = "Compute: CPU (auto: CUDA toolkit/libdevice missing)"

    # Longer reason for logs/popups.
    if gpu_ok:
        info["reason"] = f"NVIDIA GPU detected and libdevice found at {info['libdevice_path']}"
    else:
        parts = []
        if not bool(smi.get("found")):
            parts.append("nvidia-smi not found")
        elif not bool(smi.get("ok")):
            parts.append(f"nvidia-smi failed: {smi.get('error') or 'unknown'}")
        if not found_libdevice:
            parts.append("CUDA libdevice*.bc not found")
        info["reason"] = "; ".join([p for p in parts if p]) or "CUDA GPU acceleration not available"

    # Recommended XLA flag when GPU is usable.
    info["xla_cuda_data_dir"] = str(found_root) if (gpu_ok and found_root is not None) else None
    return info


def _append_xla_flag(flag: str) -> None:
    if not flag:
        return
    cur = (os.environ.get("XLA_FLAGS") or "").strip()
    if flag in cur:
        return
    os.environ["XLA_FLAGS"] = (cur + " " + flag).strip() if cur else flag


def auto_configure_cuda(*, prefer_gpu: bool = True) -> Dict[str, Any]:
    """Auto-configure env vars for stable DFC runs.

    Behaviour
    ---------
    - If the user already set CUDA_VISIBLE_DEVICES: **respect it**.
    - Otherwise:
        - if ``prefer_gpu`` and GPU looks usable -> enable GPU (no env forcing)
          and set XLA CUDA data dir.
        - else -> force CPU (CUDA_VISIBLE_DEVICES=-1).
    - Users can also force the mode via ``ONNX_SPLITPOINT_HAILO_COMPUTE``:
        - "cpu"  -> CPU forced
        - "gpu"  -> GPU forced (best effort)
        - "auto" -> default behaviour
    """

    mode_override = (os.environ.get("ONNX_SPLITPOINT_HAILO_COMPUTE") or "").strip().lower()
    if mode_override in {"cpu", "force_cpu"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        info = probe_cuda_environment()
        info["mode"] = "cpu_forced"
        info["summary"] = "Compute: CPU (forced by ONNX_SPLITPOINT_HAILO_COMPUTE=cpu)"
        return info
    if mode_override in {"gpu", "force_gpu"}:
        # Do not override CUDA_VISIBLE_DEVICES; just try to make XLA find the toolkit.
        info = probe_cuda_environment()
        if info.get("xla_cuda_data_dir"):
            _append_xla_flag(f"--xla_gpu_cuda_data_dir={info['xla_cuda_data_dir']}")
        info["mode"] = "gpu_forced"
        info["summary"] = "Compute: GPU (forced by ONNX_SPLITPOINT_HAILO_COMPUTE=gpu)"
        return info

    # Respect explicit user choice.
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        info = probe_cuda_environment()
        info["mode"] = "user"
        # summary already reflects override
        return info

    # Auto mode.
    # Probe *before* forcing CPU so the returned summary/reason reflect the
    # system capability (not our temporary CUDA_VISIBLE_DEVICES override).
    info = probe_cuda_environment()
    if prefer_gpu and bool(info.get("gpu_ok")):
        # Enable GPU; help XLA find the toolkit.
        if info.get("xla_cuda_data_dir"):
            _append_xla_flag(f"--xla_gpu_cuda_data_dir={info['xla_cuda_data_dir']}")
        info["mode"] = "gpu_auto"
        info["cuda_visible_devices_effective"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return info

    # Fallback: CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    info["mode"] = "cpu_auto"
    info["forced_cpu"] = True
    info["cuda_visible_devices_effective"] = "-1"
    return info
