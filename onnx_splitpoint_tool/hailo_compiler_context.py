"""Family-local, process-local Hailo compiler selection (stdlib only).

Resolve this *after* positive/negative artifact lookup.  A CPU request never
starts a GPU probe.  The returned execution context is diagnostic provenance,
not part of the existing artifact/recipe identity.  No package installation or
framework import takes place here.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Mapping, MutableMapping

CONTEXT_ENV = "ONNX_SPLITPOINT_HAILO_RESOLVED_COMPILER_CONTEXT"
MASK_SOURCE_ENV = "ONNX_SPLITPOINT_CUDA_MASK_SOURCE"
DEPENDENCY_MANIFEST_ENV = "ONNX_SPLITPOINT_HAILO8_DEPENDENCY_MANIFEST"
_XLA_ROOT = "--xla_gpu_cuda_data_dir"
_UNPROVIDED_COMPILERS = ("nvcc", "nvlink", "fatbinary", "cicc", "cudafe++")


class CompilerContextError(ValueError):
    """An infrastructure/configuration error, never model infeasibility."""
    def __init__(self, reason: str, message: str, **details: Any):
        self.reason = reason
        self.details = details
        super().__init__(f"{reason}: {message}")


def _error(reason: str, message: str, **details: Any):
    raise CompilerContextError(reason, message, **details)


def normalize_hailo_family(family: str) -> str:
    name = str(family).strip().lower()
    if name == "hailo10":
        name = "hailo10h"
    if name == "hailo8l":
        # DFC compute family only. Backend hw_arch and receipt identity keep
        # the exact physical SKU; an H8L HEF is not an H8 artifact alias.
        name = "hailo8"
    if name not in {"hailo8", "hailo10h"}:
        _error("hailo_compute_family_invalid", f"Unsupported DFC family {family!r}")
    return name


def _device(value: Any, source: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"device": value}
    if not isinstance(value, Mapping):
        _error("hailo_compute_invalid", f"{source} must contain a cpu/gpu device")
    result = dict(value)
    device = result.get("device")
    if not isinstance(device, str) or device.strip().lower() not in {"cpu", "gpu"}:
        _error("hailo_compute_invalid", f"{source}.device must be cpu or gpu")
    result["device"] = device.strip().lower()
    selector = result.get("gpu_selector")
    if selector is not None:
        if not isinstance(selector, str) or not selector.strip():
            _error("hailo_gpu_selection_invalid", f"{source}.gpu_selector must be a nonempty string")
        result["gpu_selector"] = selector.strip()
    if "dependency_manifest" in result:
        # Omission inherits the legacy explicit environment opt-in. An empty
        # string is a deliberate deselection, so never replace it with None.
        manifest = result["dependency_manifest"]
        if not isinstance(manifest, str) or "\x00" in manifest or "\n" in manifest or "\r" in manifest:
            _error("hailo_dependency_manifest_invalid", f"{source}.dependency_manifest must be a path string or an explicit empty string")
        result["dependency_manifest"] = manifest.strip()
    return result


def normalize_compute_by_family(value: Mapping | None) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        _error("hailo_compute_invalid", "compute_by_family must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for family, entry in value.items():
        normalized = normalize_hailo_family(family)
        setting = _device(entry, f"compute_by_family.{family}")
        if normalized != "hailo8" and "dependency_manifest" in setting:
            _error("hailo_dependency_manifest_family_invalid", "Dependency overlays are supported only for hailo8")
        if normalized in result and result[normalized] != setting:
            _error("hailo_compute_alias_conflict", f"Conflicting hailo10/hailo10h entries for {normalized}")
        result[normalized] = setting
    return result


def _env_bool(value: str, key: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    _error("hailo_compute_invalid", f"{key} has an invalid boolean value")


def resolve_compute_selection(family: str, *, job_override: Any = None,
                              compute_by_family: Mapping | None = None,
                              env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve one immutable job decision, preserving its source and conflicts.

    Explicit current job > family setting > legacy environment > CPU default.
    Explicit environment CPU prohibitions and user GPU masks are constraints;
    they cannot silently be overridden by a GPU request.  A historical global
    GPU opt-in alone does not grant Hailo8 GPU execution.
    """
    environ = dict(os.environ if env is None else env)
    family = normalize_hailo_family(family)
    if job_override is None:
        job_override = environ.get("ONNX_SPLITPOINT_HAILO_COMPUTE_OVERRIDE") or None
        if isinstance(job_override, str) and job_override.lstrip().startswith("{"):
            try:
                job_override = json.loads(job_override)
            except ValueError:
                _error("hailo_compute_invalid", "Malformed job compute override JSON")
    if compute_by_family is None and environ.get("ONNX_SPLITPOINT_HAILO_COMPUTE_BY_FAMILY"):
        try:
            compute_by_family = json.loads(environ["ONNX_SPLITPOINT_HAILO_COMPUTE_BY_FAMILY"])
        except ValueError:
            _error("hailo_compute_invalid", "Malformed compute_by_family environment JSON")
    settings = normalize_compute_by_family(compute_by_family)
    legacy: list[tuple[str, str]] = []
    ignored: list[str] = []
    mode = environ.get("ONNX_SPLITPOINT_HAILO_COMPUTE", "").strip().lower()
    if mode and mode != "auto":
        modes = {"cpu": "cpu", "force_cpu": "cpu", "gpu": "gpu", "force_gpu": "gpu"}
        if mode not in modes:
            _error("hailo_compute_invalid", "ONNX_SPLITPOINT_HAILO_COMPUTE must be cpu or gpu")
        legacy.append(("ONNX_SPLITPOINT_HAILO_COMPUTE", modes[mode]))
    elif mode == "auto":
        ignored.append("ONNX_SPLITPOINT_HAILO_COMPUTE=auto: no implicit GPU fallback/activation")
    for key in ("ONNX_SPLITPOINT_HAILO_ALLOW_GPU", "SPLITPOINT_HAILO_ALLOW_GPU"):
        if key in environ:
            legacy.append((key, "gpu" if _env_bool(environ[key], key) else "cpu"))
    if len({choice for _, choice in legacy}) > 1:
        _error("hailo_compute_conflict", "Legacy CPU/GPU environment choices disagree", choices=legacy)
    job_setting = _device(job_override, "job_override") if job_override is not None else {}
    if job_override is not None:
        setting, source = job_setting, "job_override"
    elif family in settings:
        setting, source = settings[family], f"compute_by_family.{family}"
    elif legacy and not (family == "hailo8" and legacy[0][1] == "gpu"):
        setting, source = {"device": legacy[0][1]}, "environment:" + ",".join(key for key, _ in legacy)
    else:
        setting, source = {"device": "cpu"}, "tool_default"
        if legacy and family == "hailo8":
            ignored.append("global legacy GPU opt-in does not grant hailo8 GPU execution")
    device = setting["device"]
    mask = environ.get("CUDA_VISIBLE_DEVICES")
    tool_mask = environ.get(MASK_SOURCE_ENV) == "tool_default" and mask == "-1"
    if device == "gpu":
        prohibitions = [key for key, choice in legacy if choice == "cpu"]
        if mask is not None and not tool_mask and mask.strip() in {"", "-1"}:
            prohibitions.append("CUDA_VISIBLE_DEVICES")
        if prohibitions:
            _error("hailo_compute_conflict", "GPU request conflicts with explicit CPU prohibition/mask", sources=prohibitions)
    selector = setting.get("gpu_selector")
    if device == "gpu":
        if selector is not None and mask is not None and not tool_mask and selector != mask.strip():
            _error("hailo_compute_conflict", "Requested GPU differs from explicitly inherited GPU mask", gpu_selector=selector, inherited_mask=mask)
        selector = selector or (mask.strip() if mask is not None and not tool_mask else "0")
        if not re.fullmatch(r"(?:\d+|GPU-[a-zA-Z0-9-]+)", selector):
            _error("hailo_gpu_selection_invalid", "Select exactly one GPU index or full GPU UUID", gpu_selector=selector)
    # Device and overlay have independent precedence: a CPU/GPU-only job
    # override must not discard an explicitly saved family manifest.
    manifest_inputs = []
    if family == "hailo8":
        for manifest_setting, manifest_source in (
                (job_setting, "job_override"),
                (settings.get(family, {}), f"compute_by_family.{family}")):
            if "dependency_manifest" in manifest_setting:
                manifest_inputs.append({"source": manifest_source, "value": manifest_setting["dependency_manifest"]})
        if DEPENDENCY_MANIFEST_ENV in environ:
            manifest_inputs.append({"source": "environment:" + DEPENDENCY_MANIFEST_ENV,
                                    "value": _device({"device": device, "dependency_manifest": environ[DEPENDENCY_MANIFEST_ENV]}, DEPENDENCY_MANIFEST_ENV)["dependency_manifest"]})
    elif "dependency_manifest" in job_setting:
        _error("hailo_dependency_manifest_family_invalid", "Dependency overlays are supported only for hailo8")
    chosen_manifest = manifest_inputs[0] if manifest_inputs else {"source": "unset", "value": None}
    return {"family": family, "device": device, "source": source,
            "dependency_manifest": chosen_manifest["value"],
            "dependency_manifest_source": chosen_manifest["source"],
            "dependency_manifest_usage": "not_used_for_cpu" if device == "cpu" else ("selected" if chosen_manifest["value"] else "no_overlay"),
            "ignored_dependency_manifests": [dict(row, reason="ignored_lower_priority") for row in manifest_inputs[1:]],
            "gpu_selector": selector if device == "gpu" else None,
            "ignored_legacy": ignored, "legacy_inputs": [{"source": key, "device": choice} for key, choice in legacy], "conflicts": [],
            "inherited_cuda_mask_source": "tool_default" if tool_mask else ("environment" if mask is not None else "unset")}


def parse_xla_flags(value: str) -> tuple[list[str], str | None]:
    """Parse the exact root flag (both syntaxes), preserving all other flags."""
    try:
        tokens = shlex.split(value or "", posix=True)
    except ValueError as exc:
        _error("hailo_xla_flags_invalid", str(exc))
    other, roots = [], []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            _error("hailo_xla_flags_invalid", f"Unexpected XLA token {token!r}")
        if token == _XLA_ROOT:
            i += 1
            if i >= len(tokens) or tokens[i].startswith("--") or not tokens[i]:
                _error("hailo_xla_flags_invalid", "Missing XLA CUDA root value")
            roots.append(tokens[i])
        elif token.startswith(_XLA_ROOT + "="):
            root = token[len(_XLA_ROOT) + 1:]
            if not root:
                _error("hailo_xla_flags_invalid", "Empty XLA CUDA root value")
            roots.append(root)
        else:
            other.append(token)
            # Preserve a separate argument for other syntactically valid flags.
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 1
                other.append(tokens[i])
        i += 1
    if len(set(roots)) > 1:
        _error("hailo_xla_root_conflict", "Multiple conflicting explicit XLA CUDA roots", roots=roots)
    return other, roots[0] if roots else None


def replace_xla_cuda_root(value: str, root: str) -> str:
    other, _ = parse_xla_flags(value)
    if not root or any(char.isspace() for char in root):
        _error("hailo_xla_root_invalid", "Private CUDA view needs a path without whitespace")
    return shlex.join(other + [f"{_XLA_ROOT}={root}"])


def _lexical_venv(venv_python: str | Path) -> tuple[Path, Path]:
    python = Path(os.path.abspath(os.path.expanduser(str(venv_python))))
    # Do NOT resolve bin/python: normally it is a symlink to system Python.
    if python.parent.name not in {"bin", "Scripts"}:
        _error("hailo_compiler_venv_invalid", "Interpreter must belong to the selected lexical venv bin", venv_python=str(python))
    root = python.parent.parent
    if not (root / "pyvenv.cfg").is_file() or not python.is_file() or not os.access(python, os.X_OK):
        _error("hailo_compiler_venv_invalid", "Selected venv/interpreter is missing or not executable", venv_python=str(python))
    return python, root


def _component(path: Path, root: Path, *, executable: bool) -> Path:
    if not path.is_absolute():
        _error("hailo_compiler_component_invalid", "Component paths must be absolute", path=str(path))
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except ValueError:
        _error("hailo_compiler_foreign_venv", "Component escapes the selected DFC venv", path=str(path), venv_root=str(root))
    except OSError as exc:
        _error("hailo_compiler_component_missing", str(exc), path=str(path))
    if not resolved.is_file() or resolved.stat().st_size == 0 or not os.access(resolved, os.R_OK):
        _error("hailo_compiler_component_invalid", "Component must be a readable nonempty regular file", path=str(path))
    if executable and not os.access(resolved, os.X_OK):
        _error("hailo_compiler_component_not_executable", "ptxas is not executable", path=str(path))
    return resolved


def _locate_components(root: Path, explicit: Mapping | None) -> tuple[Path, Path, str]:
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            _error("hailo_compiler_context_invalid", "explicit_context must be a mapping")
        try:
            ptxas = Path(str(explicit["ptxas_path"]))
            libdevice = Path(str(explicit["libdevice_path"]))
        except KeyError:
            _error("hailo_compiler_context_invalid", "Explicit context needs ptxas_path and libdevice_path")
        # Explicit context wins; unused discovered alternatives are not inspected.
        ptxas = _component(ptxas, root, executable=True)
        libdevice = _component(libdevice, root, executable=False)
        supplied_python = explicit.get("venv_python")
        if supplied_python and _lexical_venv(supplied_python)[1] != root:
            _error("hailo_compiler_foreign_venv", "Explicit context names a different DFC venv")
        expected_roots = {ptxas.parent.parent}
        provided_root = explicit.get("component_root")
        if provided_root:
            expected_roots = {Path(str(provided_root)).resolve(strict=True)}
        coherent = any(libdevice in {r / "lib" / "libdevice.10.bc", r / "nvvm" / "libdevice" / "libdevice.10.bc"} for r in expected_roots)
        if not coherent:
            _error("hailo_compiler_pair_mismatch", "ptxas and libdevice must come from the same component package/root")
        return ptxas, libdevice, "explicit_context"
    site_packages = sorted(set(root.glob("lib/python*/site-packages")) | set(root.glob("lib64/python*/site-packages")) | set(root.glob("Lib/site-packages")))
    for relative, lib_relative, source in [
        ("triton/backends/nvidia/bin/ptxas", "triton/backends/nvidia/lib/libdevice.10.bc", "selected_venv_triton"),
        ("nvidia/cuda_nvcc/bin/ptxas", "nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc", "selected_venv_cuda_nvcc"),
    ]:
        candidates = [(sp / relative, sp / lib_relative) for sp in site_packages if (sp / relative).exists() or (sp / relative).is_symlink() or (sp / lib_relative).exists() or (sp / lib_relative).is_symlink()]
        candidates = list({(str(a.absolute()), str(b.absolute())): (a, b) for a, b in candidates}.values())
        if not candidates:
            continue
        resolved = [(_component(a, root, executable=True), _component(b, root, executable=False)) for a, b in candidates]
        unique = list(dict.fromkeys(resolved))
        if len(unique) != 1:
            _error("hailo_compiler_context_ambiguous", "Multiple component pairs in selected DFC venv", candidates=[[str(x), str(y)] for x, y in unique])
        return unique[0][0], unique[0][1], source
    _error("hailo_compiler_components_missing", "No complete local ptxas/libdevice pair in selected DFC venv", venv_root=str(root))


def _run(command: list[str], *, env: Mapping[str, str], timeout_s: float) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, env=dict(env), capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        _error("hailo_compiler_probe_timeout", "Bounded compiler preflight timed out", command=command, timeout_s=timeout_s, stderr=str(exc.stderr or ""))
    except OSError as exc:
        _error("hailo_compiler_probe_failed", str(exc), command=command)


def _gpu_target(selector: str, env: Mapping[str, str]) -> dict[str, str]:
    smi = shutil.which("nvidia-smi", path=env.get("PATH", os.defpath))
    if not smi:
        _error("hailo_gpu_unavailable", "nvidia-smi is unavailable for selected GPU")
    command = [str(Path(smi).absolute()), "--query-gpu=index,uuid,compute_cap", "--format=csv,noheader,nounits", "--id=" + selector]
    result = _run(command, env=env, timeout_s=5.0)
    rows = [row for row in csv.reader(io.StringIO(result.stdout)) if any(x.strip() for x in row)]
    if result.returncode != 0 or len(rows) != 1 or len(rows[0]) != 3:
        _error("hailo_gpu_selection_unresolved", "Cannot identify exactly the selected GPU and compute capability", command=command, stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
    index, uuid, capability = [field.strip() for field in rows[0]]
    match = re.fullmatch(r"(\d+)\.(\d+)", capability)
    if not match or not index.isdigit() or not uuid.startswith("GPU-"):
        _error("hailo_gpu_architecture_unresolved", "Selected GPU returned invalid architecture/UUID", row=rows[0])
    if selector.isdigit() and selector != index or selector.startswith("GPU-") and selector != uuid:
        _error("hailo_gpu_selection_unresolved", "nvidia-smi returned a different device", requested=selector, returned=rows[0])
    return {"gpu_index": index, "gpu_uuid": uuid, "compute_capability": capability, "target_arch": "sm_" + match.group(1) + match.group(2), "nvidia_smi": command[0]}


def _stat_key(path: Path) -> tuple:
    st = path.stat()
    return str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def resolve_hailo_compiler_context(venv_python: str | Path, family: str, *,
        compute_selection: Mapping | None = None, job_override: Any = None,
        compute_by_family: Mapping | None = None, explicit_context: Mapping | None = None,
        parent_env: Mapping[str, str] | None = None, probe_cache: MutableMapping | None = None,
        work_dir: str | Path | None = None) -> dict[str, Any]:
    env = dict(os.environ if parent_env is None else parent_env)
    selection = dict(compute_selection) if compute_selection is not None else resolve_compute_selection(family, job_override=job_override, compute_by_family=compute_by_family, env=env)
    if selection.get("family") != normalize_hailo_family(family) or selection.get("device") not in {"cpu", "gpu"}:
        _error("hailo_compiler_context_invalid", "Compute selection differs from requested DFC family")
    python, root = _lexical_venv(venv_python)
    context = {**selection, "schema_version": 1, "venv_python": str(python), "venv_root": str(root), "compiler_component_view": selection["device"] == "gpu", "full_cuda_toolkit": False}
    if context.get("dependency_manifest") is None:
        context.pop("dependency_manifest", None)
    if selection["device"] == "cpu":
        return context
    # Recheck hard inherited constraints even for a selection passed by a caller.
    resolve_compute_selection(family, job_override={"device": "gpu", "gpu_selector": selection.get("gpu_selector", "0")}, env=env)
    parse_xla_flags(env.get("XLA_FLAGS", ""))
    overlay_manifest = selection.get("dependency_manifest") if selection["family"] == "hailo8" else None
    if explicit_context is not None and selection.get("dependency_manifest_source", "").startswith("environment:"):
        # Preserve the historical explicit component-context override. A saved
        # family or job overlay remains explicit; only an unused legacy env
        # overlay is displaced by this already supported job context.
        context["ignored_dependency_manifests"] = list(context.get("ignored_dependency_manifests", [])) + [{
            "source": selection["dependency_manifest_source"], "value": overlay_manifest,
            "reason": "ignored_by_explicit_component_context"}]
        context.update(dependency_manifest_source="explicit_context", dependency_manifest_usage="no_overlay")
        context.pop("dependency_manifest", None)
        overlay_manifest = None
    overlay_pair = None
    if overlay_manifest:
        # Only a deliberately selected, inspected H8 overlay may extend the
        # selected venv. Never permit arbitrary external component roots or
        # borrow a Hailo10/DeepX package pair.
        from .hailo_dependency_plan import child_library_environment, validated_overlay_components
        overlay_manifest = os.path.abspath(os.path.expanduser(overlay_manifest))
        context["dependency_manifest"] = overlay_manifest
        try:
            env = child_library_environment(env, family="hailo8", selected_python=str(python), manifest_path=overlay_manifest)
            if explicit_context is None:
                overlay_pair = validated_overlay_components(family="hailo8", selected_python=str(python), manifest_path=overlay_manifest)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            _error("hailo_dependency_manifest_invalid", str(exc), family="hailo8", manifest_path=overlay_manifest)
    if explicit_context is not None:
        ptxas, libdevice, source = _locate_components(root, explicit_context)
    elif overlay_pair is not None:
        overlay_root = Path(overlay_pair["component_root"])
        ptxas = _component(Path(overlay_pair["ptxas_path"]), overlay_root, executable=True)
        libdevice = _component(Path(overlay_pair["libdevice_path"]), overlay_root, executable=False)
        source = "selected_hailo8_dependency_overlay"
    else:
        ptxas, libdevice, source = _locate_components(root, None)
    gpu = _gpu_target(str(selection.get("gpu_selector") or "0"), env)
    key = (str(root), _stat_key(ptxas), _stat_key(libdevice), gpu["gpu_uuid"], gpu["target_arch"])
    cached = probe_cache.get(key) if probe_cache is not None else None
    if cached is None:
        version = _run([str(ptxas), "--version"], env=env, timeout_s=5.0)
        if version.returncode != 0 or not (version.stdout + version.stderr).strip():
            _error("hailo_gpu_ptxas_version_failed", "Selected ptxas version call failed", binary=str(ptxas), returncode=version.returncode, stderr=version.stderr)
        probe_root = Path(work_dir).absolute() if work_dir is not None else None
        if probe_root is not None:
            probe_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="onnx_hailo_target_", dir=probe_root) as scratch:
            src, out = Path(scratch) / "target.ptx", Path(scratch) / "target.cubin"
            # A no-op needs only an old, forward-compatible PTX feature set.
            # .target sm_89 with .version 7.0 is itself invalid (sm_89 was
            # introduced later), even when this ptxas supports the real GPU.
            # Keep the *output* -arch exactly the selected GPU; the source
            # .target only declares its minimal feature requirements.
            # NVIDIA PTX ISA, "PTX Module Directives: .target":
            # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-module-directives-target
            source_target = "sm_" + str(min(50, int(gpu["target_arch"][3:])))
            source_ptx = f".version 4.0\n.target {source_target}\n.address_size 64\n.visible .entry splitpoint_target_probe() {{ ret; }}\n"
            src.write_text(source_ptx, encoding="utf-8")
            command = [str(ptxas), str(src), "-arch=" + gpu["target_arch"], "-o", str(out)]
            result = _run(command, env=env, timeout_s=15.0)
            output_size = out.stat().st_size if out.is_file() else 0
            if result.returncode != 0 or output_size == 0:
                failure_text = (result.stdout + "\n" + result.stderr).lower()
                if result.returncode == 0:
                    failure_reason = "hailo_gpu_ptxas_target_output_missing"
                elif (gpu["target_arch"].lower() in failure_text and any(term in failure_text for term in ("not defined", "unsupported", "not supported", "invalid value"))):
                    failure_reason = "hailo_gpu_ptxas_target_unsupported"
                elif "unsupported .version" in failure_text or "unsupported ptx version" in failure_text:
                    failure_reason = "hailo_gpu_ptxas_source_unsupported"
                else:
                    failure_reason = "hailo_gpu_ptxas_target_probe_failed"
                _error(failure_reason, "Selected assembler did not produce a nonempty target object", binary=str(ptxas), version=(version.stdout + version.stderr).strip(), target_arch=gpu["target_arch"], command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, output_size=output_size, source_ptx=source_ptx)
            cached = {"ptxas_version": (version.stdout + version.stderr).strip(), "target_probe": {"kind": "preflight_target", "binary": str(ptxas), "target_arch": gpu["target_arch"], "source_ptx": source_ptx, "returncode": result.returncode, "output_size": output_size, "stdout": result.stdout, "stderr": result.stderr}}
        if probe_cache is not None:
            probe_cache[key] = cached
    context.update({**gpu, **cached, "ptxas_path": str(ptxas), "libdevice_path": str(libdevice), "component_source": source})
    return context


_FORWARDER = '''import json, os, subprocess, sys, time
binary, trace = sys.argv[1:3]
args = sys.argv[3:]
started = time.time()
kind = "version" if "--version" in args or "-V" in args else "assembly"
record = {"kind": kind, "binary": binary, "args": args, "pid": os.getpid(), "parent_pid": os.getppid(), "started_unix": started, "phase": os.environ.get("ONNX_SPLITPOINT_HAILO_PHASE", "compiler_child"), "target_arches": [a.split("=",1)[1] for a in args if a.startswith("-arch=") or a.startswith("--gpu-name=")]}
try:
    result = subprocess.run([binary] + args)
    record["returncode"] = result.returncode
except OSError as exc:
    record.update(returncode=127, error=str(exc))
record.update(finished_unix=time.time(), duration_s=time.time()-started)
fd = os.open(trace, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
try:
    os.write(fd, (json.dumps(record, sort_keys=True)+"\\n").encode())
finally:
    os.close(fd)
sys.exit(record["returncode"] if record["returncode"] >= 0 else 128-record["returncode"])
'''


@contextmanager
def compiler_child_environment(context: Mapping[str, Any], *, parent_env: Mapping[str, str] | None = None, work_dir: str | Path | None = None):
    """Keep the private view alive around the existing supervised child call.

    The caller must wait/reap its supervised process group before leaving this
    context, including timeout and cancellation.  No persistent config/cache
    contains the view path.  The trace survives in the supplied job directory.
    """
    env = dict(os.environ if parent_env is None else parent_env)
    effective = json.loads(json.dumps(dict(context)))
    if effective.get("device") == "cpu":
        env.update(CUDA_VISIBLE_DEVICES="-1", ONNX_SPLITPOINT_HAILO_ALLOW_GPU="0", ONNX_SPLITPOINT_HAILO_COMPUTE="cpu")
        env[MASK_SOURCE_ENV] = "tool_default" if effective.get("source") == "tool_default" else "resolved_job"
        env[CONTEXT_ENV] = json.dumps(effective, sort_keys=True)
        yield env, effective
        return
    if effective.get("device") != "gpu" or not effective.get("target_probe", {}).get("output_size"):
        _error("hailo_compiler_context_invalid", "GPU child requires a successful resolved target probe")
    # All callers (normal, deferred and diagnostic) use the already resolved
    # selection here. Re-reading the parent's manifest would undo saved
    # family/job precedence at the actual spawn boundary.
    overlay_manifest = effective.get("dependency_manifest")
    if overlay_manifest:
        if effective.get("family") != "hailo8":
            _error("hailo_dependency_manifest_family_invalid", "Dependency overlays are supported only for hailo8")
        from .hailo_dependency_plan import child_library_environment
        try:
            overlay = child_library_environment({}, family="hailo8", selected_python=effective["venv_python"], manifest_path=overlay_manifest)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            _error("hailo_dependency_manifest_invalid", str(exc), family="hailo8", manifest_path=overlay_manifest)
        library_dirs = overlay["LD_LIBRARY_PATH"].split(os.pathsep)
        inherited = [entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry and entry not in library_dirs]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs + inherited)
        effective["dependency_library_dirs"] = library_dirs
    if effective.get("family") == "hailo8":
        env[DEPENDENCY_MANIFEST_ENV] = overlay_manifest or ""
    work = Path(work_dir).absolute() if work_dir is not None else None
    if work is not None:
        work.mkdir(parents=True, exist_ok=True)
    # XLA itself has inconsistent quote handling. Do not silently escape a
    # diagnostic job's declared filesystem boundary to another scratch root.
    if work is not None and any(c.isspace() for c in str(work)):
        _error("hailo_xla_root_invalid", "Choose a compiler job directory without whitespace", work_dir=str(work))
    view_parent = str(work) if work is not None else None
    with tempfile.TemporaryDirectory(prefix="onnx_hailo_cuda_view_", dir=view_parent) as directory:
        view = Path(directory)
        binary_dir, lib_dir = view / "bin", view / "nvvm" / "libdevice"
        binary_dir.mkdir()
        lib_dir.mkdir(parents=True)
        trace = (work if work is not None else view) / (view.name + "_ptxas_invocations.jsonl")
        wrapper_source = view / "forward_ptxas.py"
        wrapper_source.write_text(_FORWARDER, encoding="utf-8")
        command = [effective["venv_python"], str(wrapper_source), effective["ptxas_path"], str(trace)]
        forwarder = binary_dir / "ptxas"
        forwarder.write_text("#!/bin/sh\nexec " + " ".join(shlex.quote(x) for x in command) + ' "$@"\n', encoding="utf-8")
        forwarder.chmod(0o700)
        # A component view is not a complete toolkit. An SDK asking for an
        # additional compiler must not discover an unrelated system CUDA
        # executable later in inherited PATH and silently mix toolchains.
        for missing_name in _UNPROVIDED_COMPILERS:
            guard = binary_dir / missing_name
            message = "hailo_gpu_compiler_component_missing:" + missing_name + "; selected compiler_component_view provides ptxas/libdevice only; system CUDA fallback is disabled"
            guard.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(message) + " >&2\nexit 127\n", encoding="utf-8")
            guard.chmod(0o700)
        (lib_dir / "libdevice.10.bc").symlink_to(effective["libdevice_path"])
        env.update(CUDA_VISIBLE_DEVICES=effective["gpu_uuid"], ONNX_SPLITPOINT_HAILO_ALLOW_GPU="1", ONNX_SPLITPOINT_HAILO_COMPUTE="gpu", CUDA_HOME=str(view), CUDA_PATH=str(view))
        env[MASK_SOURCE_ENV] = "resolved_job"
        env["XLA_FLAGS"] = replace_xla_cuda_root(env.get("XLA_FLAGS", ""), str(view))
        env["PATH"] = os.pathsep.join([str(binary_dir), str(Path(effective["venv_python"]).parent), env.get("PATH", os.defpath)])
        env.setdefault("TF_NUM_INTEROP_THREADS", "2")
        env.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        effective.update(view_root=str(view), ptxas_trace_path=str(trace), available_components=["ptxas", "libdevice.10.bc"], unprovided_compiler_components=list(_UNPROVIDED_COMPILERS), thread_settings={key: env.get(key) for key in ("TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_FORCE_GPU_ALLOW_GROWTH")})
        env[CONTEXT_ENV] = json.dumps(effective, sort_keys=True)
        validate_compiler_child_environment(env)
        yield env, effective


def validate_compiler_child_environment(env: Mapping[str, str] | None = None) -> dict[str, Any] | None:
    """Validate immediately before SDK import; never probe/remask a resolved job."""
    environ = dict(os.environ if env is None else env)
    raw = environ.get(CONTEXT_ENV)
    if raw is None:
        return None
    try:
        context = json.loads(raw)
    except (ValueError, TypeError):
        _error("hailo_compiler_context_invalid", "Malformed resolved child context")
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        _error("hailo_compiler_context_invalid", "Unsupported resolved child context")
    if context.get("device") == "cpu":
        if environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            _error("hailo_compiler_context_changed", "CPU job mask changed before SDK import")
        return context
    if context.get("device") != "gpu" or environ.get("CUDA_VISIBLE_DEVICES") != context.get("gpu_uuid"):
        _error("hailo_compiler_context_changed", "GPU selection changed before SDK import")
    if context.get("family") == "hailo8":
        if environ.get(DEPENDENCY_MANIFEST_ENV, "") != (context.get("dependency_manifest") or ""):
            _error("hailo_compiler_context_changed", "Dependency manifest selection changed before SDK import")
        expected_dirs = context.get("dependency_library_dirs", [])
        if expected_dirs and environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)[:len(expected_dirs)] != expected_dirs:
            _error("hailo_compiler_context_changed", "Dependency libraries changed before SDK import")
    view = Path(context.get("view_root", ""))
    _, root = parse_xla_flags(environ.get("XLA_FLAGS", ""))
    if not view.is_absolute() or root != str(view) or any(environ.get(k) != str(view) for k in ("CUDA_HOME", "CUDA_PATH")):
        _error("hailo_compiler_context_changed", "Private CUDA root changed before SDK import")
    if not (view / "bin" / "ptxas").is_file() or not (view / "nvvm" / "libdevice" / "libdevice.10.bc").is_file():
        _error("hailo_compiler_component_missing", "Private compiler view was removed before SDK/child completion")
    if shutil.which("ptxas", path=environ.get("PATH", "")) != str(view / "bin" / "ptxas"):
        _error("hailo_compiler_context_changed", "PATH no longer selects the private ptxas forwarder")
    return context
