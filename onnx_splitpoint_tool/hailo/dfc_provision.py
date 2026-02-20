"""Provision managed Hailo DFC virtualenvs from bundled wheel(s).

This script is intended to run on Linux (native or inside WSL).

Typical usage (inside WSL):

    python3 -m onnx_splitpoint_tool.hailo.dfc_provision --all

It will:
  - create per-profile venvs (hailo8 / hailo10)
  - install the DFC wheel(s) placed in `onnx_splitpoint_tool/resources/hailo/<profile>/`

Notes
-----
- The wheels are not shipped in this repo by default.
- Put your Hailo wheels into the directories mentioned above.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

from .dfc_manager import DfcProfile, get_dfc_manager


def _resources_hailo_root() -> Path:
    # .../onnx_splitpoint_tool/hailo/dfc_provision.py -> parents[1] == onnx_splitpoint_tool
    return Path(__file__).resolve().parents[1] / "resources" / "hailo"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python3"


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _py_version(py: str) -> Optional[Tuple[int, int]]:
    """Return (major, minor) for a python executable, or None if not runnable."""
    try:
        out = subprocess.check_output(
            [py, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        parts = out.split(".")
        if len(parts) < 2:
            return None
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def _pick_venv_python(min_version: Tuple[int, int] = (3, 10)) -> str:
    """Pick a python executable suitable for creating the managed DFC venvs.

    Hailo DFC wheels (3.33 / 5.2) pull in dependencies (e.g. jax) that require
    Python >= 3.10. On many WSL distros, /usr/bin/python3 may still be 3.8.
    """

    # 1) If the current interpreter is already new enough, prefer it.
    if (sys.version_info.major, sys.version_info.minor) >= min_version:
        return sys.executable

    # 2) Otherwise look for common commands.
    candidates = [
        "python3.10",
        "python3",
        "python",
    ]
    for cand in candidates:
        path = shutil.which(cand)
        if not path:
            continue
        ver = _py_version(path)
        if ver and ver >= min_version:
            return path

    raise RuntimeError(
        "Python >= 3.10 is required to provision the Hailo DFC venvs (DFC deps like jax have no wheels for Python 3.8).\n\n"
        "If your WSL distro is Ubuntu 20.04 (python3=3.8), you have three practical options:\n"
        "  1) Upgrade WSL distro to Ubuntu 22.04/24.04 (recommended if possible).\n"
        "  2) Install Python 3.10 via a user-space tool (pyenv or uv).\n"
        "  3) Provide a custom python path: --python /path/to/python3.10\n"
    )


def _find_wheels(wheel_dir: Path) -> List[Path]:
    if not wheel_dir.exists():
        return []
    wheels = sorted([p for p in wheel_dir.glob("*.whl") if p.is_file()])
    return wheels


def _choose_dfc_wheel(wheels: List[Path]) -> Optional[Path]:
    if not wheels:
        return None
    # Prefer something that looks like the DFC wheel name.
    for p in wheels:
        if "dataflow" in p.name and "compiler" in p.name:
            return p
    return wheels[-1]


def provision_profile(
    profile: DfcProfile,
    *,
    venv_python: str,
    force_reinstall: bool = False,
    upgrade_onnx: bool = False,
    offline: bool = False,
) -> Tuple[bool, str]:
    """Provision a single profile. Returns (ok, message)."""

    hailo_root = _resources_hailo_root()
    wheel_sub = profile.wheel_dir or profile.profile_id
    wheel_dir = hailo_root / wheel_sub

    wheels = _find_wheels(wheel_dir)
    wheel = _choose_dfc_wheel(wheels)
    if wheel is None:
        return False, f"No .whl found in {wheel_dir} (expected a Hailo DFC wheel)."

    venv_activate = profile.wsl_venv_activate
    # Convert a tilde path to an absolute path on Linux.
    # (This script runs inside Linux/WSL, so ~ expansion is valid.)
    if venv_activate.startswith("~"):
        venv_dir = Path(os.path.expanduser(venv_activate)).parents[1]
    else:
        venv_dir = Path(venv_activate).expanduser().resolve().parents[1]

    # Create venv if missing.
    if not venv_dir.exists():
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        _run([venv_python, "-m", "venv", str(venv_dir)])

    py = _venv_python(venv_dir)
    if not py.exists():
        return False, f"Venv python not found at {py}"

    # Ensure the venv uses a sufficiently new python.
    ver = _py_version(str(py))
    if ver is not None and ver < (3, 10):
        # Common when the first run was made with the default python3=3.8.
        # Recreate the venv automatically to keep UX smooth.
        try:
            print(f"[WARN] Existing venv for {profile.profile_id} uses Python {ver[0]}.{ver[1]} (<3.10). Recreating…", flush=True)
            shutil.rmtree(venv_dir)
        except Exception as e:
            return False, f"Venv python is too old ({ver[0]}.{ver[1]}). Could not remove {venv_dir}: {type(e).__name__}: {e}"

        _run([venv_python, "-m", "venv", str(venv_dir)])
        py = _venv_python(venv_dir)
        if not py.exists():
            return False, f"Venv python not found at {py} after recreate"

    # Upgrade pip tooling
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    # Install wheel (offline, from directory)
    pip_cmd = [str(py), "-m", "pip", "install"]
    if force_reinstall:
        pip_cmd += ["--force-reinstall"]

    # Prefer local wheels, but allow pip index downloads unless --offline is set.
    pip_cmd += ["--find-links", str(wheel_dir)]
    if offline:
        pip_cmd += ["--no-index"]
    pip_cmd += [str(wheel)]
    _run(pip_cmd)

    # IMPORTANT:
    # Do NOT upgrade ONNX by default.
    # The Hailo DFC wheels pin specific versions (e.g. hailo8 DFC 3.33 pins onnx==1.16.0, protobuf==3.20.3).
    # Upgrading ONNX often upgrades protobuf too, which breaks hailo_sdk_client imports.
    if upgrade_onnx:
        print(
            f"[WARN] Upgrading onnx inside the managed venv for '{profile.profile_id}'. "
            "This is NOT recommended and may break DFC pinned dependencies.",
            flush=True,
        )
        _run([str(py), "-m", "pip", "install", "--upgrade", "onnx"])

    # Fail fast if dependencies are inconsistent.
    try:
        chk = subprocess.check_output([str(py), "-m", "pip", "check"], text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        return False, f"pip check failed for {profile.profile_id}:\n{e.output}"
    if chk:
        return False, f"pip check reported dependency issues for {profile.profile_id}:\n{chk}"

    # Sanity import
    code = (
        "import hailo_sdk_client, onnx, google.protobuf; "
        "print(getattr(hailo_sdk_client,'__version__','unknown'), onnx.__version__, google.protobuf.__version__)"
    )
    try:
        out = subprocess.check_output([str(py), "-c", code], text=True).strip()
    except Exception as e:
        return False, f"Installed wheel but import failed: {type(e).__name__}: {e}"

    return (
        True,
        f"OK: {profile.profile_id} (hailo_sdk_client / onnx / protobuf: {out})\n  venv: {venv_dir}\n  wheel: {wheel.name}",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="List known profiles and exit")
    ap.add_argument("--all", action="store_true", help="Provision all profiles")
    ap.add_argument("--profile", action="append", default=None, help="Provision only this profile_id (repeatable)")
    ap.add_argument("--force-reinstall", action="store_true", help="Force reinstall the wheel into the venv")
    # Default: do NOT upgrade onnx. (Upgrading can break DFC pinned deps.)
    ap.add_argument(
        "--upgrade-onnx",
        action="store_true",
        help="Upgrade onnx inside the managed venv (NOT recommended; can break DFC pinned deps)",
    )
    ap.add_argument(
        "--no-onnx-upgrade",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy / backward compatibility
    )
    ap.add_argument("--offline", action="store_true", help="Do not use pip index; install only from local wheel directories")
    ap.add_argument(
        "--python",
        dest="python_exe",
        default=None,
        help="Python executable (>=3.10) to use for venv creation. Useful with pyenv/uv on older WSL distros.",
    )

    args = ap.parse_args()

    # Decide which python to use for creating the managed venvs.
    if args.python_exe:
        venv_python = str(args.python_exe).strip()
        ver = _py_version(venv_python)
        if ver is None:
            print(f"[ERR] --python '{venv_python}' is not runnable.", file=sys.stderr)
            return 4
        if ver < (3, 10):
            print(f"[ERR] --python must be >=3.10 (got {ver[0]}.{ver[1]}).", file=sys.stderr)
            return 4
    else:
        try:
            venv_python = _pick_venv_python()
        except Exception as e:
            print(f"[ERR] {e}", file=sys.stderr)
            return 4

    mgr = get_dfc_manager()

    if args.list:
        for p in mgr.profiles:
            print(
                f"{p.profile_id}: prefixes={list(p.hw_arch_prefixes)} venv={p.wsl_venv_activate} wheel_dir={p.wheel_dir or ''}"
            )
        return 0

    wanted: List[str] = []
    if args.profile:
        wanted = [str(x).strip().lower() for x in args.profile if str(x).strip()]
    elif args.all:
        wanted = [p.profile_id for p in mgr.profiles]
    else:
        ap.error("Specify --all or --profile <id> (or --list)")

    ok_all = True
    for pid in wanted:
        prof = mgr.get_profile(pid)
        if prof is None:
            print(f"[ERR] Unknown profile_id: {pid}", file=sys.stderr)
            ok_all = False
            continue

        print(f"[INFO] Provisioning profile '{prof.profile_id}' …", flush=True)
        ok, msg = provision_profile(
            prof,
            venv_python=venv_python,
            force_reinstall=bool(args.force_reinstall),
            upgrade_onnx=(bool(getattr(args, "upgrade_onnx", False)) and (not bool(getattr(args, "no_onnx_upgrade", False)))),
            offline=bool(args.offline),
        )
        print(msg)
        ok_all = ok_all and ok

    return 0 if ok_all else 3


if __name__ == "__main__":
    raise SystemExit(main())
