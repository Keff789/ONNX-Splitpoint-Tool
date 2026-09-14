"""Explicit, offline Hailo8 CUDA supplement planning.

This module does not import vendor frameworks, invoke pip, download packages,
modify vendor venvs, or declare GPU readiness. Only ``stage_reviewed_plan``
writes an additive private directory; callers must explicitly authorize it.
"""
from __future__ import annotations

import email
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

SCHEMA = "hailo8_dependency_plan_v27934"
OVERLAY_SCHEMA = "hailo8_private_cuda_overlay_v27934"
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_INVENTORY_BYTES = 16 * 1024 * 1024
MAX_WHEEL_BYTES = 4 * 1024 * 1024 * 1024
MAX_WHEEL_FILES = 20000

# -I -S prevents sitecustomize/.pth execution and user PYTHONPATH inheritance.
# The lexical venv root is explicit because older Python -S resets sys.prefix.
_METADATA_WORKER = r'''
import email, json, os, pathlib, platform, sys
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
rows, seen, total = [], set(), 0
sites = []
for lib in (root / "lib", root / "lib64"):
    for site in sorted(lib.glob("python" + '.'.join(str(x) for x in sys.version_info[:2]) + "/site-packages")):
        resolved = site.resolve(strict=True)
        resolved.relative_to(root)
        if resolved not in sites:
            sites.append(resolved)
for site in sites:
    for item in sorted(site.glob("*.dist-info")):
        item.resolve(strict=True).relative_to(root)
        path = item / "METADATA"
        path.resolve(strict=True).relative_to(root)
        if not path.is_file() or path.stat().st_size > 2097152:
            raise RuntimeError("metadata_missing_or_over_budget:" + str(path))
        data = path.read_bytes()
        total += len(data)
        if total > 16777216 or len(rows) >= 2048:
            raise RuntimeError("metadata_inventory_over_budget")
        msg = email.message_from_bytes(data)
        name, version = msg.get("Name"), msg.get("Version")
        if not name or not version:
            raise RuntimeError("metadata_identity_missing:" + str(path))
        rows.append(dict(name=name, version=version,
            requires_dist=msg.get_all("Requires-Dist", []),
            provides_extra=msg.get_all("Provides-Extra", []),
            requires_python=msg.get("Requires-Python", ""),
            metadata_path=str(path)))
v = sys.implementation.version
iv = '.'.join(str(x) for x in (v.major, v.minor, v.micro))
if v.releaselevel != 'final': iv += v.releaselevel[0] + str(v.serial)
marker = dict(implementation_name=sys.implementation.name,
 implementation_version=iv, os_name=os.name,
 platform_machine=platform.machine(), platform_release=platform.release(),
 platform_system=platform.system(), platform_version=platform.version(),
 python_full_version=platform.python_version(), platform_python_implementation=platform.python_implementation(),
 python_version='.'.join(platform.python_version_tuple()[:2]), sys_platform=sys.platform)
print(json.dumps(dict(selected_venv=str(root), python_executable=sys.argv[2],
 marker_environment=marker, libc=list(platform.libc_ver()), distributions=rows, vendor_imports=False), sort_keys=True))
'''


def _absolute(path):
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def collect_inventory(selected_python, *, timeout_s=30):
    """Read complete local distribution requirements in a bounded fresh process."""
    py = _absolute(selected_python)  # deliberately do not resolve bin/python
    if py.parent.name != "bin" or not py.is_file() or not os.access(py, os.X_OK):
        raise ValueError("selected_hailo8_venv_python_invalid")
    root = py.parent.parent.resolve(strict=True)
    if not (root / "pyvenv.cfg").is_file():
        raise ValueError("selected_hailo8_venv_configuration_missing")
    cfg = (root / "pyvenv.cfg").read_text()
    if re.search(r"(?im)^include-system-site-packages\s*=\s*true\s*$", cfg):
        raise ValueError("selected_hailo8_venv_inherits_unbounded_system_packages")
    if not 0 < float(timeout_s) <= 60:
        raise ValueError("metadata_timeout_must_be_bounded")
    # File-backed output limits memory even if the selected interpreter misbehaves.
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        result = subprocess.run(
            [str(py), "-I", "-B", "-S", "-c", _METADATA_WORKER, str(root), str(py)],
            stdout=out, stderr=err, timeout=float(timeout_s), check=False,
            env={"PATH": str(py.parent), "HOME": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        if out.tell() > MAX_INVENTORY_BYTES or err.tell() > MAX_INVENTORY_BYTES:
            raise ValueError("metadata_output_over_budget")
        out.seek(0); err.seek(0)
        if result.returncode:
            raise ValueError("metadata_collection_failed:" + err.read(8192).decode(errors="replace"))
        inventory = json.loads(out.read())
    if inventory.get("selected_venv") != str(root) or inventory.get("python_executable") != str(py):
        raise ValueError("metadata_selected_venv_mismatch")
    return inventory


def _packaging():
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version
        from packaging.specifiers import SpecifierSet
    except ImportError as exc:
        raise RuntimeError("environment_blocked:packaging_required_for_metadata_plan") from exc
    return Requirement, canonicalize_name, Version, SpecifierSet


def _distributions(inventory):
    _, canonical, Version, _ = _packaging()
    result = {}
    for row in inventory["distributions"]:
        name = canonical(row["name"])
        Version(row["version"])
        if name in result:
            raise ValueError("duplicate_distribution:" + name)
        result[name] = row
    sdk = result.get("hailo-dataflow-compiler")
    if not sdk or Version(sdk["version"]).major != 3:
        raise ValueError("hailo8_dfc3_metadata_required_no_hailo10_borrowing")
    if "tensorflow" not in result:
        raise ValueError("hailo8_tensorflow_metadata_required")
    return result


def _active_requirement(raw, marker_environment, extras=("",)):
    Requirement, _, _, _ = _packaging()
    req = Requirement(raw)
    if req.url:
        raise ValueError("direct_url_requirement_needs_manual_review:" + req.name)
    return req, req.marker is None or any(
        req.marker.evaluate({**marker_environment, "extra": extra}) for extra in extras
    )


def _supplement_name(name):
    return bool(re.fullmatch(r"nvidia-[a-z0-9-]+-cu12", name))


def _safe_target(target, inventory):
    target = _absolute(target)
    root = Path(inventory["selected_venv"]).resolve(strict=True)
    # A dedicated *new sibling* cannot accidentally overwrite any vendor/tool venv.
    if target.parent.resolve(strict=True) != root.parent or not target.name.startswith("hailo8_cuda_"):
        raise ValueError("private_hailo8_target_must_be_new_named_sibling")
    if target.exists() or target.is_symlink():
        raise ValueError("private_hailo8_target_already_exists")
    return target


def _wheel_compatible(path, inventory):
    """Check supported Linux CUDA wheel tags against the selected Python/host.

    Generic linux tags are intentionally not accepted: they do not express a
    minimum loader ABI. Unsupported tags stay a manual-review block.
    """
    from packaging.utils import parse_wheel_filename
    _, _, _, tags = parse_wheel_filename(Path(path).name)
    env = inventory["marker_environment"]
    py = env["python_version"].replace(".", "")
    arch = env["platform_machine"]
    libc, version = inventory.get("libc", ["", ""])
    try:
        glibc = tuple(int(x) for x in version.split(".")[:2]) if libc == "glibc" else (0, 0)
    except ValueError:
        glibc = (0, 0)
    for tag in tags:
        if tag.interpreter not in ("py3", "py" + py, "cp" + py) or tag.abi not in ("none", "cp" + py):
            continue
        if tag.platform == "any":
            return True
        if env["sys_platform"] != "linux":
            continue
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)_(.+)", tag.platform)
        if match and match[3] == arch and glibc >= (int(match[1]), int(match[2])):
            return True
        legacy = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}
        if any(tag.platform == prefix + "_" + arch and glibc >= minimum for prefix, minimum in legacy.items()):
            return True
    return False


def _inspect_wheel(path):
    _, canonical, Version, _ = _packaging()
    path = _absolute(path)
    if not path.is_file() or path.is_symlink() or path.suffix != ".whl" or path.stat().st_size > MAX_WHEEL_BYTES:
        raise ValueError("local_wheel_invalid:" + str(path))
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > MAX_WHEEL_FILES or sum(i.file_size for i in members) > MAX_WHEEL_BYTES:
            raise ValueError("wheel_contents_over_budget")
        metadata = [i for i in members if i.filename.endswith(".dist-info/METADATA")]
        if len(metadata) != 1 or metadata[0].file_size > MAX_METADATA_BYTES:
            raise ValueError("wheel_metadata_not_unique_or_over_budget")
        names = set()
        for item in members:
            name = item.filename
            p = PurePosixPath(name)
            mode = item.external_attr >> 16
            if (name in names or p.is_absolute() or ".." in p.parts or "\\" in name or
                    not p.parts or stat.S_ISLNK(mode) or (mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_IFMT(mode) == 0))):
                raise ValueError("unsafe_wheel_member:" + name)
            names.add(name)
            if (p.suffix == ".pth" or p.name in ("sitecustomize.py", "usercustomize.py") or
                    any(part.endswith(".data") for part in p.parts)):
                raise ValueError("wheel_global_hook_or_install_scheme_rejected:" + name)
            if p.parts[0] != "nvidia" and not p.parts[0].endswith(".dist-info"):
                raise ValueError("wheel_outside_nvidia_namespace:" + name)
        msg = email.message_from_bytes(archive.read(metadata[0]))
        name, version = canonical(msg["Name"] or ""), msg["Version"]
        if not _supplement_name(name) or not version:
            raise ValueError("wheel_framework_substitution_rejected:" + name)
        Version(version)
        row = dict(name=name, version=version, requires_dist=msg.get_all("Requires-Dist", []),
                   provides_extra=msg.get_all("Provides-Extra", []), requires_python=msg.get("Requires-Python", ""))
        return {**row, "path": str(path), "sha256": _digest(path), "size": path.stat().st_size}


def build_plan(inventory, target, *, packages=(), wheels=()):
    """Plan explicitly selected packages; optional framework extras are proposals only.

    Full metadata is retained. A concrete stage plan additionally requires local
    exact wheels, complete dependency closure and a user-selected package list.
    """
    Requirement, canonical, Version, SpecifierSet = _packaging()
    installed = _distributions(inventory)
    target = _safe_target(target, inventory)
    env = inventory["marker_environment"]
    proposals, active_constraints, errors = {}, {}, []
    for name, row in installed.items():
        extras = ("", "and-cuda") if name == "tensorflow" else ("",)
        for raw in row.get("requires_dist", []):
            try:
                req, active = _active_requirement(raw, env, extras)
            except Exception as exc:
                errors.append({"reason": "invalid_or_unreviewed_requirement", "source": name, "requirement": raw, "detail": str(exc)})
                continue
            if not active:
                continue
            dep = canonical(req.name)
            active_constraints.setdefault(dep, []).append({"source": name, "requirement": raw})
            if name == "tensorflow" and _supplement_name(dep):
                pins = [s.version for s in req.specifier if s.operator == "==" and "*" not in s.version]
                if len(pins) != 1:
                    errors.append({"reason": "tensorflow_supplement_without_exact_pin", "requirement": raw})
                else:
                    proposals[dep] = {"name": dep, "version": pins[0], "source_requirement": raw,
                                      "installed_version": installed.get(dep, {}).get("version"),
                                      "necessity": "NOT_PROVEN_BY_METADATA"}
    selected = [canonical(x) for x in packages]
    if len(set(selected)) != len(selected):
        errors.append({"reason": "duplicate_package_selection"})
    selected = sorted(set(selected))
    expected = {}
    for name in selected:
        if name not in proposals:
            errors.append({"reason": "package_not_in_current_tensorflow_cuda_requirements", "name": name})
            continue
        expected[name] = proposals[name]["version"]
        if name in installed:
            errors.append({"reason": "existing_package_must_not_be_shadowed", "name": name,
                           "installed_version": installed[name]["version"]})
    wheel_rows = []
    for path in wheels:
        try:
            wheel_rows.append(_inspect_wheel(path))
        except Exception as exc:
            errors.append({"reason": "wheel_rejected", "detail": str(exc)})
    wheel_map = {}
    for row in wheel_rows:
        name = row["name"]
        if name in wheel_map:
            errors.append({"reason": "duplicate_wheel", "name": name})
        wheel_map[name] = row
        if expected.get(name) != row["version"]:
            errors.append({"reason": "unreviewed_solver_change_or_wrong_version", "name": name, "version": row["version"]})
        if not _wheel_compatible(row["path"], inventory):
            errors.append({"reason": "wheel_target_python_or_platform_incompatible", "name": name})
        if row["requires_python"] and not SpecifierSet(row["requires_python"]).contains(env["python_full_version"]):
            errors.append({"reason": "wheel_python_requirement_conflict", "name": name})
        for raw in row["requires_dist"]:
            try:
                req, active = _active_requirement(raw, env)
                if not active:
                    continue
                dep = canonical(req.name)
                version = expected.get(dep) or installed.get(dep, {}).get("version")
                if not version or not req.specifier.contains(version):
                    errors.append({"reason": "supplement_dependency_missing_or_conflicting", "source": name, "requirement": raw})
            except Exception as exc:
                errors.append({"reason": "supplement_requirement_invalid", "detail": str(exc)})
    for name, version in expected.items():
        for constraint in active_constraints.get(name, []):
            req = Requirement(constraint["requirement"])
            if not req.specifier.contains(version):
                errors.append({"reason": "installed_framework_or_sdk_constraint_conflict", "name": name, **constraint})
    missing_wheels = sorted(set(expected) - set(wheel_map))
    status = "BLOCKED" if errors else ("REVIEW_REQUIRED" if not selected else ("LOCAL_WHEELS_REQUIRED" if missing_wheels else "READY_FOR_EXPLICIT_OFFLINE_STAGE"))
    return {"schema": SCHEMA, "family": "hailo8", "status": status,
            "inventory": inventory, "target": str(target), "proposals": sorted(proposals.values(), key=lambda x: x["name"]),
            "selected_packages": selected, "wheels": sorted(wheel_rows, key=lambda x: (x["name"], x["path"])),
            "missing_wheels": missing_wheels, "conflicts": errors,
            "expected_changes": ["Create only the new private target; selected exact local NVIDIA wheels only.",
                                 "No vendor venv, framework, registry, shell, GUI, or cache writes."],
            "rollback": "Deselect the overlay manifest; once no child uses it, remove only the private target.",
            "network_used": False, "package_manager_used": False,
            "hardware_execution": "NOT_RUN", "gpu_readiness": "NOT_PROVEN",
            "next_gates": ["fresh_hailo8_sdk_import", "hailo8_eager_matmul_and_conv2d", "hailo8_xla",
                           "hailo8_model_build", "normal_hailo8_reuse_after_restart"]}


def stage_reviewed_plan(plan, *, expected_plan_sha256, plan_file):
    """Explicit offline action; exact reviewed bytes and fresh metadata are required.

    This is deliberately separate from the release installer and build preflight.
    No resolver runs, so it cannot opportunistically replace a framework.
    """
    if _digest(plan_file) != expected_plan_sha256 or json.loads(Path(plan_file).read_text()) != plan:
        raise ValueError("reviewed_plan_bytes_changed")
    if plan.get("schema") != SCHEMA or plan.get("status") != "READY_FOR_EXPLICIT_OFFLINE_STAGE":
        raise ValueError("plan_not_ready_for_explicit_stage")
    current = collect_inventory(plan["inventory"]["python_executable"])
    if current != plan["inventory"]:
        raise ValueError("selected_venv_metadata_changed_since_review")
    rebuilt = build_plan(current, plan["target"], packages=plan["selected_packages"], wheels=[w["path"] for w in plan["wheels"]])
    if rebuilt != plan:
        raise ValueError("reviewed_plan_or_wheel_changed")
    target = _safe_target(plan["target"], current)
    target.mkdir(mode=0o700)
    try:
        destination = target / "packages"
        destination.mkdir(mode=0o700)
        members_written = {}
        for row in plan["wheels"]:
            with zipfile.ZipFile(row["path"]) as archive:
                for item in archive.infolist():
                    path = destination / item.filename
                    if item.is_dir():
                        path.mkdir(parents=True, exist_ok=True)
                        continue
                    # NVIDIA namespace init files may be shared, but their bytes
                    # must agree; no silent last-wheel-wins content substitution.
                    if item.filename in members_written:
                        if item.file_size > 1024 * 1024 or path.read_bytes() != archive.read(item):
                            raise ValueError("wheel_file_collision:" + item.filename)
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, path.open("xb") as stream:
                        shutil.copyfileobj(source, stream, length=1024 * 1024)
                    path.chmod(0o700 if ((item.external_attr >> 16) & 0o111) else 0o600)
                    members_written[item.filename] = row["name"]
        lib_dirs = sorted({str(p.parent.relative_to(target)) for p in destination.rglob("*.so*") if p.is_file()})
        if not lib_dirs:
            raise ValueError("supplement_has_no_runtime_libraries_loader_gate_open")
        manifest = {"schema": OVERLAY_SCHEMA, "family": "hailo8", "selected_venv": current["selected_venv"],
                    "python_executable": current["python_executable"], "inventory": current,
                    "packages": [{k: w[k] for k in ("name", "version", "sha256")} for w in plan["wheels"]],
                    "library_dirs": lib_dirs, "gpu_readiness": "NOT_PROVEN", "hardware_execution": "NOT_RUN",
                    "namespace_imports_enabled": False, "rollback": plan["rollback"]}
        manifest_path = target / "overlay_manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        return manifest_path
    except BaseException:
        shutil.rmtree(target)
        raise


def child_library_environment(base_env, *, family, selected_python, manifest_path):
    """Explicit H8 selection only; returned LD path affects the supplied child.

    No PYTHONPATH is added. Libraries are resolved by exact inspected directories;
    optional namespace-import requirements must be diagnosed, not globally hooked.
    """
    if family != "hailo8":
        raise ValueError("hailo8_overlay_cannot_be_used_by_other_family")
    path = _absolute(manifest_path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INVENTORY_BYTES:
        raise ValueError("hailo8_overlay_manifest_invalid")
    root = path.parent.resolve(strict=True)
    manifest = json.loads(path.read_text())
    selected = _absolute(selected_python)
    venv = selected.parent.parent.resolve(strict=True)
    if (manifest.get("schema") != OVERLAY_SCHEMA or manifest.get("family") != "hailo8" or
            manifest.get("selected_venv") != str(venv) or manifest.get("python_executable") != str(selected) or
            root.parent != venv.parent or not root.name.startswith("hailo8_cuda_")):
        raise ValueError("hailo8_overlay_family_or_venv_binding_mismatch")
    # Metadata-only validation, no SDK/GPU process and no installation on HIT.
    inventory = manifest["inventory"]
    installed = _distributions(inventory)
    current_metadata = set()
    for lib in (venv / "lib", venv / "lib64"):
        for metadata in lib.glob("python" + inventory["marker_environment"]["python_version"] + "/site-packages/*.dist-info/METADATA"):
            current_metadata.add(str(metadata.resolve(strict=True)))
    if current_metadata != {str(Path(row["metadata_path"]).resolve(strict=True)) for row in installed.values()}:
        raise ValueError("hailo8_overlay_selected_venv_distribution_inventory_changed")
    for name, row in installed.items():
        metadata = Path(row["metadata_path"])
        metadata.resolve(strict=True).relative_to(venv)
        if metadata.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError("hailo8_overlay_metadata_over_budget")
        msg = email.message_from_bytes(metadata.read_bytes())
        if (msg.get("Name") != row["name"] or msg.get("Version") != row["version"] or
                msg.get_all("Requires-Dist", []) != row.get("requires_dist", []) or
                msg.get_all("Provides-Extra", []) != row.get("provides_extra", []) or
                msg.get("Requires-Python", "") != row.get("requires_python", "")):
            raise ValueError("hailo8_overlay_selected_venv_metadata_changed:" + name)
    _, canonical, _, _ = _packaging()
    observed = {}
    package_root = root / "packages"
    if package_root.is_symlink():
        raise ValueError("hailo8_overlay_package_root_symlink")
    for item in package_root.rglob("*"):
        if item.is_symlink() or item.suffix == ".pth" or item.name in ("sitecustomize.py", "usercustomize.py"):
            raise ValueError("hailo8_overlay_hook_or_symlink_rejected")
        if item.name == "METADATA" and item.parent.name.endswith(".dist-info"):
            msg = email.message_from_bytes(item.read_bytes())
            name = canonical(msg.get("Name", ""))
            if name in observed or not _supplement_name(name):
                raise ValueError("hailo8_overlay_unreviewed_distribution")
            observed[name] = msg.get("Version")
    expected = {p["name"]: p["version"] for p in manifest["packages"]}
    if observed != expected or not expected:
        raise ValueError("hailo8_overlay_package_inventory_changed")
    dirs = []
    for relative in manifest["library_dirs"]:
        candidate = root / relative
        candidate.resolve(strict=True).relative_to(package_root.resolve(strict=True))
        if not candidate.is_dir() or not any(p.is_file() for p in candidate.glob("*.so*")):
            raise ValueError("hailo8_overlay_library_directory_invalid")
        dirs.append(str(candidate))
    if not dirs:
        raise ValueError("hailo8_overlay_libraries_missing")
    env = {str(k): str(v) for k, v in base_env.items()}
    old = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + ([old] if old else []))
    return env


def validated_overlay_components(*, family, selected_python, manifest_path):
    """Return the reviewed H8 nvcc pair, or None for a library-only supplement.

    Selection reuses the entire family/venv/package inspection, then checks exact
    package-relative component paths. It never admits arbitrary external roots.
    The caller must still run its real bounded assembler target probe.
    """
    child_library_environment({}, family=family, selected_python=selected_python,
                              manifest_path=manifest_path)
    manifest_path = _absolute(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if "nvidia-cuda-nvcc-cu12" not in {p["name"] for p in manifest["packages"]}:
        return None
    root = manifest_path.parent / "packages/nvidia/cuda_nvcc"
    ptxas, libdevice = root / "bin/ptxas", root / "nvvm/libdevice/libdevice.10.bc"
    for path in (ptxas, libdevice):
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("hailo8_overlay_nvcc_component_missing_or_empty:" + str(path))
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    if not os.access(ptxas, os.X_OK):
        raise ValueError("hailo8_overlay_ptxas_not_executable")
    return {"ptxas_path": str(ptxas), "libdevice_path": str(libdevice), "component_root": str(root)}
