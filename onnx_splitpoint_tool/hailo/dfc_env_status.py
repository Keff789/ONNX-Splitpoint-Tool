"""Status helpers for managed Hailo DFC environments.

The tool can provision separate DFC virtualenvs for Hailo-8 and Hailo-10 from
wheels placed in ``onnx_splitpoint_tool/resources/hailo/<profile>/``.  This
module keeps GUI/CLI status reporting independent from the actual installer.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..resources_utils import persistent_resource_path
from ..runners.backends.hailo_utils import DfcProfile, get_dfc_manager


def hailo_resources_root() -> Path:
    """Return the package/resource directory that should contain Hailo wheels."""

    return persistent_resource_path("resources", "hailo")


def _venv_dir_from_activate(path_text: str) -> Path:
    p = Path(os.path.expanduser(str(path_text or "").strip()))
    # Expected: <venv>/bin/activate.  Be defensive for hand-edited profiles.
    if p.name == "activate" and p.parent.name == "bin":
        return p.parent.parent
    if p.name == "bin":
        return p.parent
    return p


def _venv_python(venv_dir: Path) -> Path:
    py = venv_dir / "bin" / "python"
    return py if py.exists() else (venv_dir / "bin" / "python3")

def _last_provision_status(venv_dir: Path) -> Dict[str, Any]:
    path = venv_dir / "splitpoint_dfc_provision_status.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _package_version(py: Path, package_name: str) -> str:
    if not py.exists():
        return ""
    code = (
        "import importlib.metadata as m, sys; "
        "name=sys.argv[1]; "
        "\ntry:\n print(m.version(name))\nexcept Exception:\n print('')"
    )
    try:
        return subprocess.check_output([str(py), "-c", code, package_name], text=True, stderr=subprocess.STDOUT, timeout=8).strip()
    except Exception:
        return ""


def _pip_check_no_broken_requirements(text: str) -> bool:
    msg = str(text or "").strip().lower()
    return (not msg) or ("no broken requirements found" in msg and "requires" not in msg and "incompatible" not in msg)


def _module_import_version(py: Path, module_name: str, *, timeout: int = 20) -> tuple[bool, str]:
    """Return (import_ok, version_or_output).  Used only as a fallback when
    package metadata is not available.  Several Hailo wheels expose the DFC
    distribution as ``hailo-dataflow-compiler`` while the importable module is
    ``hailo_sdk_client``; older status code looked only for the module's
    distribution metadata and reported a false NOT READY state.
    """
    if not py.exists():
        return False, ""
    code = (
        "import importlib, sys; "
        "m=importlib.import_module(sys.argv[1]); "
        "print(getattr(m, '__version__', 'import_ok'))"
    )
    try:
        out = subprocess.check_output([str(py), "-c", code, module_name], text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()
        return True, out
    except Exception as exc:
        return False, str(exc)


def inspect_profile(profile: DfcProfile, *, probe_import: bool = False) -> Dict[str, Any]:
    """Inspect wheel presence and managed venv status for one DFC profile."""

    root = hailo_resources_root()
    wheel_sub = profile.wheel_dir or profile.profile_id
    wheel_dir = root / wheel_sub
    wheels = sorted([p for p in wheel_dir.glob("*.whl") if p.is_file()]) if wheel_dir.exists() else []
    activate = str(profile.wsl_venv_activate or "")
    venv_dir = _venv_dir_from_activate(activate)
    py = _venv_python(venv_dir)
    result: Dict[str, Any] = {
        "profile_id": profile.profile_id,
        "hw_arch_prefixes": list(profile.hw_arch_prefixes),
        "notes": profile.notes,
        "glibc_min": profile.glibc_min,
        "wheel_dir": str(wheel_dir),
        "wheel_count": len(wheels),
        "wheels": [p.name for p in wheels],
        "venv_activate": activate,
        "venv_dir": str(venv_dir),
        "venv_exists": venv_dir.exists(),
        "python": str(py),
        "python_exists": py.exists(),
        "last_provision_status": _last_provision_status(venv_dir),
        "hailo_sdk_client_version": _package_version(py, "hailo_sdk_client"),
        "hailo_dataflow_compiler_version": _package_version(py, "hailo-dataflow-compiler"),
        "onnx_version": _package_version(py, "onnx"),
        "protobuf_version": _package_version(py, "protobuf"),
        "status": "unknown",
        "ready": False,
        "error_class": "",
        "hint": "",
    }
    if not wheel_dir.exists():
        result.update(
            status="wheel_dir_missing",
            error_class="hailo_dfc_wheel_dir_missing",
            hint=f"Create {wheel_dir} and place the matching Hailo DFC .whl there.",
        )
        return result
    if not wheels:
        result.update(
            status="wheel_missing",
            error_class="hailo_dfc_wheel_missing",
            hint=f"Place the matching hailo_dataflow_compiler-*.whl in {wheel_dir}.",
        )
        return result
    if not venv_dir.exists() or not py.exists():
        result.update(
            status="venv_missing",
            error_class="hailo_dfc_venv_missing",
            hint="Click 'Install/Repair DFC envs' or run ./scripts/provision_hailo_dfcs_wsl.sh --all.",
        )
        return result
    # Hailo DFC wheels expose different metadata names across versions.  The
    # authoritative install signal is either the DFC distribution metadata
    # (hailo-dataflow-compiler) or a successful hailo_sdk_client import.
    dfc_version = str(result.get("hailo_dataflow_compiler_version") or "").strip()
    sdk_version = str(result.get("hailo_sdk_client_version") or "").strip()
    import_ok = False
    import_version = ""
    if not sdk_version and not dfc_version:
        # Lightweight fallback for environments where metadata is missing but
        # the importable SDK is actually present.
        import_ok, import_version = _module_import_version(py, "hailo_sdk_client", timeout=20)
        result["hailo_sdk_client_import_ok"] = import_ok
        result["hailo_sdk_client_import_output"] = import_version
        if import_ok and import_version and import_version != "import_ok":
            result["hailo_sdk_client_version"] = import_version
            sdk_version = import_version

    installed = bool(sdk_version or dfc_version or import_ok)
    if not installed:
        last = result.get("last_provision_status") if isinstance(result.get("last_provision_status"), dict) else {}
        last_status = str(last.get("status") or "").strip()
        last_msg = str(last.get("message") or "").strip()
        if last_status:
            hint = "Last installer state: " + last_status
            if last_msg:
                short = last_msg.replace("\n", " ")
                if len(short) > 360:
                    short = short[:357] + "..."
                hint += ". " + short
            if last_status == "pip_check_issues" and _pip_check_no_broken_requirements(last_msg):
                hint = (
                    "Last installer state was a false-positive pip_check_issues marker "
                    "('No broken requirements found'), but no DFC package/import could be confirmed. "
                    "Click Install/Repair DFC envs once to refresh the status."
                )
            if "graphviz" in (last_msg + " " + last_status).lower():
                hint += " Install Graphviz headers first: sudo apt update && sudo apt install -y graphviz libgraphviz-dev pkg-config build-essential"
            result.update(
                status="venv_present_install_incomplete",
                error_class="hailo_dfc_install_incomplete",
                hint=hint,
            )
        else:
            result.update(
                status="venv_present_install_incomplete",
                error_class="hailo_dfc_install_incomplete",
                hint="Managed venv exists but no Hailo DFC package/import could be confirmed. Use Hardware → Install/Repair DFC.",
            )
        return result

    result.update(status="ready", ready=True, error_class="", hint="Managed DFC venv is present and Hailo DFC/SDK is installed.")
    last = result.get("last_provision_status") if isinstance(result.get("last_provision_status"), dict) else {}
    if str(last.get("status") or "").strip() == "pip_check_issues" and _pip_check_no_broken_requirements(str(last.get("message") or "")):
        result["stale_false_positive_pip_check_marker"] = True
        result["hint"] = "Previous pip_check_issues marker said 'No broken requirements found' and is ignored."

    if probe_import:
        code = "import hailo_sdk_client; print(getattr(hailo_sdk_client,'__version__','unknown'))"
        try:
            out = subprocess.check_output([str(py), "-c", code], text=True, stderr=subprocess.STDOUT, timeout=20).strip()
            result["import_ok"] = True
            result["import_output"] = out
        except Exception as exc:
            result["import_ok"] = False
            result["import_output"] = str(exc)
            result["ready"] = False
            result["status"] = "import_failed"
            result["error_class"] = "hailo_dfc_import_failed"
            result["hint"] = "Re-run provisioning; the managed venv may have drifted or miss pinned dependencies."
    return result


def inspect_profiles(*, probe_import: bool = False) -> List[Dict[str, Any]]:
    mgr = get_dfc_manager()
    return [inspect_profile(p, probe_import=probe_import) for p in mgr.profiles]


def format_status_lines(statuses: Iterable[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    lines.append("Hailo DFC managed environments")
    lines.append(f"Resources: {hailo_resources_root()}")
    lines.append("")
    for st in statuses:
        pid = str(st.get("profile_id") or "?")
        ready = bool(st.get("ready"))
        status = str(st.get("status") or "unknown")
        mark = "OK" if ready else "NOT READY"
        lines.append(f"{pid}: {mark} ({status})")
        prefixes = ", ".join(str(x) for x in st.get("hw_arch_prefixes") or [])
        if prefixes:
            lines.append(f"  arch prefixes: {prefixes}")
        lines.append(f"  wheels: {st.get('wheel_count', 0)} in {st.get('wheel_dir')}")
        wheels = list(st.get("wheels") or [])
        if wheels:
            for name in wheels[:5]:
                lines.append(f"    - {name}")
            if len(wheels) > 5:
                lines.append(f"    … +{len(wheels) - 5} more")
        lines.append(f"  venv: {st.get('venv_dir')}")
        lines.append(f"  activate: {st.get('venv_activate')}")
        last = st.get("last_provision_status") if isinstance(st.get("last_provision_status"), dict) else {}
        if last:
            ls = str(last.get("status") or "").strip()
            lp = str(last.get("python_version") or "").strip()
            exp = str(last.get("expected_python") or "").strip()
            bits = []
            if ls:
                bits.append(f"last={ls}")
            if lp:
                bits.append(f"python={lp}")
            if exp:
                bits.append(f"expected={exp}")
            if bits:
                lines.append("  provision: " + ", ".join(bits))
        versions = []
        for key, label in (("hailo_dataflow_compiler_version", "hailo_dfc"), ("hailo_sdk_client_version", "hailo_sdk_client"), ("onnx_version", "onnx"), ("protobuf_version", "protobuf")):
            val = str(st.get(key) or "").strip()
            if val:
                versions.append(f"{label}={val}")
        if versions:
            lines.append("  versions: " + ", ".join(versions))
        hint = str(st.get("hint") or "").strip()
        if hint and not ready:
            lines.append(f"  hint: {hint}")
        lines.append("")
    return lines


def format_status_text(*, probe_import: bool = False) -> str:
    return "\n".join(format_status_lines(inspect_profiles(probe_import=probe_import))).rstrip() + "\n"
