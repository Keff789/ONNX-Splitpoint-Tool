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
import zipfile
from email.parser import Parser
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


def _extract_exact_pins_from_wheel(wheel_path: Path) -> List[str]:
    """Return a list of `name==version` constraints found in the wheel metadata.

    Hailo DFC wheels commonly pin critical dependencies exactly
    (e.g. `onnx==1.16.0`, `protobuf==3.20.3`). If users accidentally upgrade
    ONNX/protobuf inside the managed venv, hailo_sdk_client imports can break.
    """

    if not wheel_path.exists():
        return []

    try:
        with zipfile.ZipFile(str(wheel_path), "r") as zf:
            meta_name = None
            for n in zf.namelist():
                if n.endswith(".dist-info/METADATA"):
                    meta_name = n
                    break
            if not meta_name:
                return []

            raw = zf.read(meta_name)
            text = raw.decode("utf-8", errors="replace")

        msg = Parser().parsestr(text)
        req_lines = msg.get_all("Requires-Dist") or []

        # Try to use packaging for proper requirement parsing (available via pip).
        try:
            from packaging.requirements import Requirement  # type: ignore
        except Exception:
            Requirement = None  # type: ignore

        pins: List[str] = []

        for req in req_lines:
            s = str(req).strip()
            if not s:
                continue

            name = None
            ver = None

            if Requirement is not None:
                try:
                    r = Requirement(s)
                    name = str(r.name).strip()
                    # Only accept exact pins (==)
                    exact = [sp for sp in r.specifier if getattr(sp, "operator", None) == "=="]
                    if len(exact) == 1 and str(exact[0]).startswith("=="):
                        ver = str(exact[0]).lstrip("=")
                except Exception:
                    name = None
                    ver = None

            if name is None or ver is None:
                # Fallback regex for simple 'name==version' forms.
                if "==" in s and ";" not in s:
                    parts = s.split("==", 1)
                    if len(parts) == 2:
                        n0 = parts[0].strip()
                        v0 = parts[1].strip()
                        if n0 and v0 and " " not in n0 and " " not in v0:
                            name = n0
                            ver = v0

            if name and ver:
                pins.append(f"{name}=={ver}")

        # De-duplicate while preserving order.
        seen = set()
        uniq: List[str] = []
        for p in pins:
            key = p.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(p)
        return uniq
    except Exception:
        return []


def _write_constraints_file(venv_dir: Path, pins: List[str]) -> Optional[Path]:
    """Write a pip constraints file into the managed venv."""

    if not pins:
        return None
    try:
        path = venv_dir / "pip_constraints.txt"
        lines = [
            "# Auto-generated by ONNX Splitpoint Tool (Hailo DFC manager)",
            "# Re-run provisioning to regenerate.",
            "",
        ] + [str(x).strip() for x in pins if str(x).strip()]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except Exception:
        return None


def _patch_activate_for_constraints(venv_dir: Path, constraints_path: Path) -> None:
    """Patch venv/bin/activate so pip automatically uses the constraints file."""

    act = venv_dir / "bin" / "activate"
    if not act.exists():
        return

    marker = "# ONNX_SPLITPOINT_PIP_CONSTRAINTS"
    export_line = f'export PIP_CONSTRAINT="{constraints_path}"'

    try:
        txt = act.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    if marker in txt:
        return

    # Do not override if the user already set PIP_CONSTRAINT.
    if "PIP_CONSTRAINT=" in txt:
        return

    try:
        with act.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write(marker + "\n")
            f.write(export_line + "\n")
    except Exception:
        return


def _get_installed_versions(py: Path, names: List[str]) -> dict:
    """Return a mapping name->version (or '' if not installed)."""

    if not names:
        return {}

    import json as _json

    code = (
        "import importlib.metadata as m, json, sys; "
        "names=json.loads(sys.argv[1]); out={}; "
        "for n in names:\n"
        "  key=str(n);\n"
        "  try:\n"
        "    out[key]=m.version(key)\n"
        "  except Exception:\n"
        "    out[key]=''\n"
        "print(json.dumps(out))"
    )

    try:
        raw = subprocess.check_output([str(py), "-c", code, _json.dumps(names)], text=True, stderr=subprocess.STDOUT).strip()
        return _json.loads(raw) if raw else {}
    except Exception:
        return {}


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

    # Write a constraints file based on exact pins inside the DFC wheel.
    # Additionally, we pin a few fragile runtime deps (onnx/protobuf/ml-dtypes)
    # to the versions that were actually installed by the DFC wheel.
    #
    # This serves as a guardrail: if someone runs `pip install onnx` inside the
    # managed venv, pip will keep the pinned versions instead of upgrading and
    # breaking hailo_sdk_client imports.
    pins = _extract_exact_pins_from_wheel(wheel)

    # Ensure critical runtime deps are pinned, even if the wheel only specifies
    # ranges via transitive dependencies.
    critical_names = ["onnx", "protobuf", "ml-dtypes", "ml_dtypes"]
    installed = _get_installed_versions(py, critical_names)

    def _has_pin(name_lc: str) -> bool:
        for p in pins:
            if p.split("==", 1)[0].strip().lower() == name_lc:
                return True
        return False

    if installed.get("onnx") and (not _has_pin("onnx")):
        pins.append(f"onnx=={installed['onnx']}")
    if installed.get("protobuf") and (not _has_pin("protobuf")):
        pins.append(f"protobuf=={installed['protobuf']}")
    ml_ver = installed.get("ml-dtypes") or installed.get("ml_dtypes")
    if ml_ver and (not _has_pin("ml-dtypes")) and (not _has_pin("ml_dtypes")):
        # Prefer the canonical package name.
        pins.append(f"ml-dtypes=={ml_ver}")
    constraints_path = _write_constraints_file(venv_dir, pins)
    if constraints_path is not None:
        _patch_activate_for_constraints(venv_dir, constraints_path)

        # If a managed venv already existed and someone upgraded packages
        # manually, make a best-effort attempt to restore pinned versions.
        # We only enforce a small set of known fragile deps to avoid excessive
        # downloads.
        critical = [
            "onnx",
            "protobuf",
            "ml-dtypes",
            "ml_dtypes",
        ]
        desired: List[str] = []
        # Collect exact pins for the critical packages.
        for p in pins:
            try:
                name, ver = p.split("==", 1)
            except Exception:
                continue
            if name.strip().lower() in {c.lower() for c in critical}:
                desired.append(f"{name.strip()}=={ver.strip()}")

        if desired:
            # Check installed versions first.
            names = [d.split("==", 1)[0] for d in desired]
            vers = _get_installed_versions(py, names)
            mismatched: List[str] = []
            for d in desired:
                n, v = d.split("==", 1)
                got = str(vers.get(n, "") or "")
                if got != str(v):
                    mismatched.append(d)

            if mismatched:
                print(f"[WARN] Detected dependency drift in '{profile.profile_id}' venv. Restoring pins: {', '.join(mismatched)}", flush=True)
                fix_cmd = [str(py), "-m", "pip", "install", "--force-reinstall", "--no-deps"]
                if offline:
                    fix_cmd += ["--no-index"]
                fix_cmd += mismatched
                _run(fix_cmd)

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
