from __future__ import annotations

from ..config_values import parse_config_bool, validate_profile_config_booleans

import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..cache_verify_policy import compiler_dispatch_forbidden
from ..paths import ensure_dir, splitpoint_provisioning_logs_dir
from ..process_control import current_process_registry, terminate_process_tree, ProcessTreeRegistry


def _expand(p: str | os.PathLike[str] | None) -> Path:
    raw = Path(os.path.expandvars(os.path.expanduser(str(p or ""))))
    try:
        return raw.resolve()
    except PermissionError:
        return raw.absolute()
    except OSError:
        return raw.absolute()


def _is_auto(v: Any) -> bool:
    return str(v or "").strip().lower() in {"", "auto", "default", "none"}


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except PermissionError:
        return False
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except PermissionError:
        return False
    except OSError:
        return False


def _path_status(path: Path) -> str:
    try:
        if path.is_dir():
            return "directory"
        if path.exists():
            return "file"
        return "missing"
    except PermissionError:
        return "permission_denied"
    except OSError as exc:
        return f"os_error:{type(exc).__name__}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dx_all_suite_root() -> Path:
    for key in ("DEEPX_DX_ALL_SUITE_ROOT", "DX_ALL_SUITE_ROOT", "DX_ALL_SUITE"):
        v = os.environ.get(key)
        if not _is_auto(v):
            return _expand(v)

    # Prefer the current user's home.  Do not probe /home/jasmin first: on
    # shared systems that path can raise PermissionError during pathlib.exists().
    candidates = [
        "~/dx-all-suite",
        "~/deepx/dx-all-suite",
        "/opt/dx-all-suite",
        "/home/jasmin/dx-all-suite",
    ]
    for p in candidates:
        q = _expand(p)
        if _safe_exists(q):
            return q
    return _expand("~/dx-all-suite")


def default_compiler_venv(root: Optional[Path] = None) -> Path:
    if not _is_auto(os.environ.get("DEEPX_COMPILER_VENV")):
        return _expand(os.environ.get("DEEPX_COMPILER_VENV"))
    r = root or default_dx_all_suite_root()
    candidates = [
        r / "dx-compiler" / "venv-dx-compiler-local",
        r / "dx-compiler" / "venv",
        r / "venv-dx-compiler-local",
    ]
    for c in candidates:
        if _safe_exists(c / "bin" / "activate") or _safe_exists(c / "bin" / "python"):
            return c
    return candidates[0]


def default_runtime_venv(root: Optional[Path] = None) -> Path:
    if os.environ.get("DEEPX_RUNTIME_VENV"):
        return _expand(os.environ.get("DEEPX_RUNTIME_VENV"))
    r = root or default_dx_all_suite_root()
    candidates = [
        _expand("~/venvs/deepx-runtime"),
        _expand("~/venvs/deepx-runtime"),
        r / "dx-runtime" / "venv-dx-runtime-local",
    ]
    for c in candidates:
        if _safe_exists(c / "bin" / "activate") or _safe_exists(c / "bin" / "python"):
            return c
    return candidates[0]


def default_compiler_venv_activate(root: Optional[Path] = None) -> str:
    return f"source {default_compiler_venv(root)}/bin/activate"


def default_runtime_venv_activate(root: Optional[Path] = None) -> str:
    return f"source {default_runtime_venv(root)}/bin/activate"


def default_cache_dir() -> Path:
    return _expand("~/Models/BackendArtifacts/deepx")


def _venv_from_activate(cmd: Any) -> Optional[Path]:
    raw = str(cmd or "").strip()
    if not raw:
        return None
    # Common forms: source /path/venv/bin/activate, . ~/venv/bin/activate
    try:
        tokens = shlex.split(raw)
    except ValueError:
        tokens = raw.split()
    for token in tokens:
        if token.endswith("/bin/activate"):
            return _expand(token[:-len("/bin/activate")])
    return None


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _compiler_config_values(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize aliases within one priority level, before merging levels."""
    cfg = {key: value for key, value in config.items() if not _is_auto(value)}
    for key, aliases in (("dx_all_suite_root", ("dx_all_suite_root", "compiler_root")),
                         ("compiler_venv", ("compiler_venv", "venv"))):
        value = next((cfg[name] for name in aliases if name in cfg), None)
        if value is not None:
            cfg[key] = value
    if "compiler_venv" not in cfg:
        venv = _venv_from_activate(cfg.get("venv_activate"))
        if venv is not None:
            cfg["compiler_venv"] = str(venv)
    return cfg


def _local_compiler_candidates(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [_compiler_config_values(row) for row in payload.get("build_environments", []) or []
            if isinstance(row, Mapping) and row.get("enabled", True)
            and str(row.get("kind") or "") == "deepx_dxcom"
            and str(row.get("host") or "local") in {"local", "localhost"}]


def _saved_compiler_candidates() -> list[dict[str, Any]]:
    # Read the existing tool configuration without creating or rewriting it.
    from .. import backend_build_environments
    path = Path(backend_build_environments.CONFIG_PATH).expanduser()
    if not path.is_file():
        return []
    import yaml
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _local_compiler_candidates(payload)


def _compiler_high_priority(config: Mapping[str, Any], *, root: Any = None) -> dict[str, Any]:
    cfg = _compiler_config_values(config)
    if not _is_auto(root):
        cfg["dx_all_suite_root"] = root
    for field, keys in (
        ("dx_all_suite_root", ("DEEPX_DX_ALL_SUITE_ROOT", "DX_ALL_SUITE_ROOT", "DX_ALL_SUITE")),
        ("compiler_venv", ("DEEPX_COMPILER_VENV",)),
        ("compiler_overlay", ("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY",)),
    ):
        value = next((os.environ[key] for key in keys if not _is_auto(os.environ.get(key))), None)
        if value is not None:
            cfg[field] = value
    return cfg


def _compiler_python_path(python: Any) -> Path:
    # Resolve directories for comparisons, but never dereference bin/python.
    return Path(os.path.expandvars(os.path.expanduser(str(python)))).absolute()


def _compiler_fallback_needed(cfg: Mapping[str, Any], python: Any = None) -> bool:
    return not (cfg.get("dx_all_suite_root") and (cfg.get("compiler_venv") or python)
                and cfg.get("compiler_overlay"))


def _select_compiler_fallback(
    cfg: Mapping[str, Any], candidates: list[dict[str, Any]], *, python: Any = None,
) -> dict[str, Any]:
    """Select one compatible record only when the call still needs its fields."""
    compatible = []
    for row in candidates:
        if any(cfg.get(key) and row.get(key) and _expand(cfg[key]) != _expand(row[key])
               for key in ("dx_all_suite_root", "compiler_venv")):
            continue
        if python and row.get("compiler_venv"):
            selected_python = _compiler_python_path(python)
            row_python = _venv_python(_expand(row["compiler_venv"]))
            if (_expand(selected_python.parent), selected_python.name) != (row_python.parent, row_python.name):
                continue
        compatible.append(row)
    if _compiler_fallback_needed(cfg, python):
        if len(compatible) > 1:
            raise ValueError("deepx_compiler_configuration_ambiguous: multiple compatible local DX-COM configurations")
    # Keep ancillary fields of an unambiguous profile record (for example
    # runtime_venv); the higher-priority fields still win in the merge.
    selected = dict(compatible[0]) if len(compatible) == 1 else {}
    # A complete compiler choice needs no stored compiler alternative. Cache
    # locations are independent: prefer a compatible configured target, but do
    # not discard the sole existing cache when the caller overrides the entire
    # compiler installation. This preserves v31's custom artifact location.
    if not cfg.get("cache_dir") and not selected.get("cache_dir"):
        cache_candidates = compatible if any(row.get("cache_dir") for row in compatible) else candidates
        caches = {str(_expand(row["cache_dir"])) for row in cache_candidates if row.get("cache_dir")}
        if len(caches) > 1:
            raise ValueError("deepx_compiler_configuration_ambiguous: cache_dir requires an explicit selection")
        if caches:
            selected["cache_dir"] = caches.pop()
    return selected


def profile_compiler_configuration(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse the profile's existing local build environment and explicit overrides."""
    validate_profile_config_booleans(profile)
    explicit = _compiler_config_values(profile.get("deepx_build") or {})
    high = _compiler_high_priority(explicit)
    selected = _select_compiler_fallback(high, _local_compiler_candidates(profile))
    # Force is owned by the effective profile, never by compiler discovery.
    # _is_auto historically filters False; restore this typed switch after
    # fallback merging so a local environment cannot silently enable Force.
    force_build = parse_config_bool(
        profile.get("deepx_build", {}).get("force_build", False),
        field="deepx_build.force_build",
    )
    return {**selected, **explicit, "force_build": force_build}


def resolve_compiler_context(
    config: Optional[Mapping[str, Any]] = None, *,
    root: str | os.PathLike[str] | None = None,
    python: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Resolve one compiler-local context, without importing or starting it."""
    high = _compiler_high_priority(config or {}, root=root)
    python = None if _is_auto(python) else python
    # Invalid explicit overlays fail on their own terms, before any optional
    # saved alternative can obscure the actual configuration error.
    if high.get("compiler_overlay"):
        explicit_path = _expand(high["compiler_overlay"])
        if os.pathsep in str(explicit_path) or not (explicit_path / "torch" / "__init__.py").is_file():
            raise ValueError(f"deepx_compiler_overlay_invalid: {explicit_path}")
    saved = (_select_compiler_fallback(high, _saved_compiler_candidates(), python=python)
             if _compiler_fallback_needed(high, python) or not high.get("cache_dir") else {})
    cfg = {**saved, **high}
    dx_root = _expand(cfg.get("dx_all_suite_root") or default_dx_all_suite_root())
    python_venv = None
    if python and _compiler_python_path(python).parent.name in {"bin", "Scripts"}:
        python_venv = _compiler_python_path(python).parent.parent
    venv = _expand(cfg.get("compiler_venv") or python_venv or default_compiler_venv(dx_root))
    # Keep the venv path: resolving a bin/python symlink loses its site-packages.
    compiler_python = str(_compiler_python_path(python or _venv_python(venv)))
    explicit_overlay = os.environ.get("ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY", "").strip()
    explicit_overlay = "" if _is_auto(explicit_overlay) else explicit_overlay
    configured_overlay = str(cfg.get("compiler_overlay") or "").strip()
    source = ("process_environment" if explicit_overlay else "explicit_configuration" if high.get("compiler_overlay")
              else "saved_build_configuration" if configured_overlay else "unconfigured")
    overlay_value = explicit_overlay or configured_overlay
    if not overlay_value:
        # The already used cu126 installation only; no recursive/system search.
        candidates = sorted({p.resolve() for p in (dx_root / "dx-compiler").glob("pytorch-*-cu126-overlay") if (p / "torch" / "__init__.py").is_file()})
        if len(candidates) > 1:
            raise ValueError("deepx_compiler_overlay_ambiguous: " + ", ".join(map(str, candidates)))
        if candidates:
            overlay_value, source = str(candidates[0]), "known_dx_all_suite_installation"
    overlay = str(_expand(overlay_value)) if overlay_value else ""
    if overlay and (os.pathsep in overlay or not (Path(overlay) / "torch" / "__init__.py").is_file()):
        raise ValueError(f"deepx_compiler_overlay_invalid: {overlay}")
    child_env = dict(os.environ)
    if overlay:
        existing = [p for p in child_env.get("PYTHONPATH", "").split(os.pathsep) if p and p != overlay]
        child_env["PYTHONPATH"] = os.pathsep.join([overlay, *existing])
        child_env["ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY"] = overlay
    return {"compiler_python": compiler_python, "compiler_venv": str(venv),
            "dx_all_suite_root": str(dx_root), "compiler_overlay": overlay,
            "compiler_selection_source": source, "cache_dir": str(_expand(cfg.get("cache_dir") or default_cache_dir())), "child_environment": child_env}


def compiler_subprocess_environment(
    config: Optional[Mapping[str, Any]] = None, *, context: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, str]]:
    resolved = dict(context or resolve_compiler_context(config))
    return dict(resolved["child_environment"]) if resolved.get("compiler_overlay") else None


def compiler_python_command(context: Mapping[str, Any], *, code: str, isolated: bool = False) -> list[str]:
    """An explicit sys.path bootstrap also works when Python ignores PYTHONPATH (-I)."""
    overlay = str(context.get("compiler_overlay") or "")
    bootstrap = ("import sys; sys.path.insert(0, " + repr(overlay) + "); " if overlay else "")
    bootstrap += "exec(compile(" + repr(code) + ", '<deepx-compiler-child>', 'exec'))"
    return [str(context["compiler_python"]), *(["-I"] if isolated else []), "-c", bootstrap]


def _cancel_requested(cancel_event: Any, registry: Any) -> bool:
    try:
        if cancel_event is not None and cancel_event.is_set():
            return True
    except Exception:
        pass
    return bool(registry is not None and getattr(registry, "cancelled", False))


def _bounded_collect_after_stop(
    proc: subprocess.Popen[str],
    terminate: Any,
    *,
    timeout_s: float = 2.0,
) -> str:
    """Drain a stopped probe without waiting forever on inherited pipes."""

    try:
        output, _ = proc.communicate(timeout=max(0.1, float(timeout_s)))
        return output or ""
    except subprocess.TimeoutExpired as exc:
        partial = exc.output or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        try:
            terminate(0.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            output, _ = proc.communicate(timeout=0.5)
            return output or str(partial)
        except subprocess.TimeoutExpired as final_exc:
            final_output = final_exc.output or partial or ""
            if isinstance(final_output, bytes):
                final_output = final_output.decode("utf-8", "replace")
            return str(final_output) + "\n[output drain remained open after process-tree stop]"


def _run_owned_probe(
    args: list[str],
    *,
    timeout_s: int,
    process_registry: Any = None,
    cancel_event: Any = None,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    """Run a short environment probe as an owned, cancellable process tree."""

    registry = process_registry or current_process_registry()
    local_registry = registry is None
    if local_registry:
        registry = ProcessTreeRegistry()
    if _cancel_requested(cancel_event, registry):
        return subprocess.CompletedProcess(args, 130, "CANCELLED before DeepX probe start", "")
    popen_kwargs: dict[str, Any] = {
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if env is not None:
        popen_kwargs["env"] = dict(env)
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(args, **popen_kwargs)
    if registry is not None:
        try:
            registry.register(proc, label="deepx-environment-probe")
        except BaseException:
            terminate_process_tree(proc, grace_s=0.5)
            raise

    def _terminate(grace_s: float) -> None:
        if registry is not None:
            registry.terminate_registered(proc, grace_s=grace_s)
        else:
            terminate_process_tree(proc, grace_s=grace_s)

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    output = ""
    try:
        while True:
            if _cancel_requested(cancel_event, registry):
                if proc.poll() is None:
                    _terminate(1.0)
                output = _bounded_collect_after_stop(proc, _terminate)
                return subprocess.CompletedProcess(
                    args,
                    130,
                    (output or "") + "\nCANCELLED by Evaluation Workflow",
                    "",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate(0.5)
                output = _bounded_collect_after_stop(proc, _terminate)
                raise subprocess.TimeoutExpired(args, timeout_s, output=output)
            try:
                output, _ = proc.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        return subprocess.CompletedProcess(args, int(proc.returncode or 0), output or "", "")
    except BaseException:
        if proc.poll() is None:
            _terminate(0.5)
        raise
    finally:
        if registry is not None:
            registry.unregister(proc)
            if local_registry:
                registry.assert_quiescent()


def _probe_python_import(
    python: Path,
    module: str,
    timeout_s: int = 10,
    *,
    process_registry: Any = None,
    cancel_event: Any = None,
    compiler_scope: bool = False,
    compiler_context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if not _safe_exists(python):
        return {"module": module, "ok": False, "reason": f"python not found: {python}"}
    try:
        probe_script = f"""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
module_name = {module!r}
loaded = importlib.import_module(module_name)
module_file = str(getattr(loaded, "__file__", "") or "")
module_sha256 = ""
if module_file and Path(module_file).is_file():
    module_sha256 = hashlib.sha256(Path(module_file).read_bytes()).hexdigest()
distributions = list(importlib.metadata.packages_distributions().get(module_name, []) or [])
distribution = distributions[0] if distributions else ""
version = str(getattr(loaded, "__version__", "") or "")
if not version and distribution:
    try:
        version = importlib.metadata.version(distribution)
    except Exception:
        version = ""
print(json.dumps({{
    "status": "OK",
    "module": module_name,
    "module_file": module_file,
    "module_file_sha256": module_sha256,
    "distribution": distribution,
    "package_version": version,
}}, sort_keys=True))
"""
        cp = _run_owned_probe(
            compiler_python_command(compiler_context or resolve_compiler_context(python=python), code=probe_script) if compiler_scope else [str(python), "-c", probe_script],
            timeout_s=timeout_s,
            process_registry=process_registry,
            cancel_event=cancel_event,
            env=compiler_subprocess_environment(context=compiler_context) if compiler_scope else None,
        )
        output = cp.stdout or ""
        identity: dict[str, Any] = {}
        if cp.returncode == 0:
            for line in reversed(output.splitlines()):
                try:
                    candidate = json.loads(line)
                except Exception:
                    continue
                if isinstance(candidate, Mapping) and candidate.get("status") == "OK":
                    identity = dict(candidate)
                    break
        return {
            "module": module,
            "ok": cp.returncode == 0,
            "returncode": cp.returncode,
            "output_tail": output[-1000:],
            "distribution": str(identity.get("distribution") or ""),
            "package_version": str(identity.get("package_version") or ""),
            "module_file": str(identity.get("module_file") or ""),
            "module_file_sha256": str(identity.get("module_file_sha256") or ""),
        }
    except Exception as exc:
        return {"module": module, "ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def _which_many(names: list[str]) -> dict[str, Optional[str]]:
    return {n: shutil.which(n) for n in names}


def _which_in_venv(venv: Path, names: list[str]) -> dict[str, Optional[str]]:
    """Return executable paths inside a venv without relying on global PATH.

    DeepX compiler tools are installed as console entry points inside the
    dx-compiler venv.  A missing global ``dxcom`` binary is acceptable; a
    missing venv entry point is not acceptable for ONNX -> DXNN builds.
    """
    bindir = venv / ("Scripts" if os.name == "nt" else "bin")
    out: dict[str, Optional[str]] = {}
    for name in names:
        cand = bindir / name
        try:
            out[name] = str(cand) if cand.exists() else None
        except Exception:
            out[name] = None
    return out



def _python_tag(
    python: Path,
    *,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> str:
    if not _safe_exists(python):
        return ""
    try:
        cp = _run_owned_probe(
            [str(python), "-c", "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"],
            timeout_s=5,
            process_registry=process_registry,
            cancel_event=cancel_event,
            env=compiler_subprocess_environment(),
        )
        return (cp.stdout or "").strip() if cp.returncode == 0 else ""
    except Exception:
        return ""


def _cuda_architecture_status(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Classify metadata only; absence of evidence is never an incompatibility."""
    result = dict(payload)
    result.update(status="unknown", reason="deepx_compiler_cuda_probe_unavailable")
    if payload.get("cuda_available") is False:
        result.update(status="not_applicable", reason="deepx_compiler_cpu_path")
        return result
    capability = payload.get("device_capability")
    arches = payload.get("compiled_architectures")
    if payload.get("cuda_available") is not True or not isinstance(capability, (list, tuple)) or len(capability) != 2 or not isinstance(arches, list):
        return result
    try:
        major, minor = (int(value) for value in capability)
        if major < 1 or minor < 0 or minor > 9:
            return result
    except (TypeError, ValueError):
        return result
    known: list[tuple[str, int]] = []
    for arch in arches:
        match = re.fullmatch(r"(sm|compute)_(\d+)", str(arch))
        if match:
            known.append((match.group(1), int(match.group(2))))
    if not known:
        return result
    target = major * 10 + minor
    # Cubins are compatible within a major version, towards newer minor
    # versions. PTX may be JIT compiled for newer GPU architectures.
    supported = any(
        (kind == "sm" and value // 10 == major and value <= target)
        or (kind == "compute" and value <= target)
        for kind, value in known
    )
    if supported:
        result.update(status="compatible", reason="deepx_compiler_cuda_architecture_compatible")
    elif len(known) != len(arches):
        # Architecture-specific suffixes cannot be classified from these
        # metadata alone. Do not turn a partial parse into a false failure.
        return result
    else:
        result.update(status="incompatible", reason="deepx_compiler_cuda_architecture_unsupported")
    return result


def probe_compiler_cuda_architecture(
    python: Path,
    *,
    process_registry: Any = None,
    cancel_event: Any = None,
    compiler_context: Optional[Mapping[str, Any]] = None,
    operation_probe: bool = False,
) -> dict[str, Any]:
    """Inspect the selected compiler Python without allocating GPU tensors."""
    base: dict[str, Any] = {"compiler_python": str(python)}
    if compiler_dispatch_forbidden():
        return {**base, "status": "not_probed", "reason": "artifact_policy_forbids_compiler_probe"}
    try:
        context = dict(compiler_context or resolve_compiler_context(python=python))
        child_env = compiler_subprocess_environment(context=context)
        base.update({key: context[key] for key in ("compiler_python", "compiler_overlay", "compiler_selection_source") if context.get(key)})
    except (OSError, ValueError) as exc:
        return {**base, "status": "incompatible", "reason": str(exc).split(":")[0], "detail": str(exc)}
    if child_env is not None:
        base["compiler_overlay"] = child_env["PYTHONPATH"].split(os.pathsep)[0]
    if not _safe_exists(python):
        return {**base, "status": "unknown", "reason": "deepx_compiler_python_missing"}
    marker = "DEEPX_COMPILER_CUDA_METADATA="
    script = """
import json
import os
import torch
p = {"torch_version": str(torch.__version__), "cuda_available": bool(torch.cuda.is_available()),
     "torch_module_file": str(torch.__file__), "torch_cuda_version": str(torch.version.cuda),
     "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
if p["cuda_available"]:
    i = int(torch.cuda.current_device())
    p.update(device_index=i, device_name=torch.cuda.get_device_name(i),
             device_capability=list(torch.cuda.get_device_capability(i)),
             compiled_architectures=list(torch.cuda.get_arch_list()))
if OPERATION_PROBE:
    p["operations_probe_status"] = "not_applicable" if not p["cuda_available"] else "pending"
    if p["cuda_available"]:
        try:
            a = torch.ones((128, 128), device="cuda", dtype=torch.float32)
            b = torch.full((128, 128), 2.0, device="cuda", dtype=torch.float32)
            c = a @ b
            if not bool(torch.allclose(c, torch.full_like(c, 256.0))):
                raise RuntimeError("matmul_result_invalid")
            x = torch.ones((1, 3, 16, 16), device="cuda", dtype=torch.float32)
            w = torch.ones((4, 3, 3, 3), device="cuda", dtype=torch.float32)
            y = torch.nn.functional.conv2d(x, w)
            if tuple(y.shape) != (1, 4, 14, 14) or not bool(torch.allclose(y, torch.full_like(y, 27.0))):
                raise RuntimeError("convolution_result_invalid")
            torch.cuda.synchronize()
            p["operations_probe_status"] = "pass"
        except Exception as exc:
            p["operations_probe_status"] = "failed"
            p["operations_probe_error"] = type(exc).__name__ + ": " + str(exc)
    p["dxcom_compile_status"] = "not_run"
print("DEEPX_COMPILER_CUDA_METADATA=" + json.dumps(p, sort_keys=True))
"""
    script = script.replace("OPERATION_PROBE", repr(bool(operation_probe)))
    try:
        cp = _run_owned_probe(
            compiler_python_command(context, code=script), timeout_s=120 if operation_probe else 15,
            process_registry=process_registry, cancel_event=cancel_event,
            env=child_env,
        )
        if cp.returncode == 0:
            for line in reversed((cp.stdout or "").splitlines()):
                if line.startswith(marker):
                    payload = json.loads(line[len(marker):])
                    if isinstance(payload, dict):
                        result = {**base, **_cuda_architecture_status(payload)}
                        if context.get("compiler_overlay") and not Path(str(payload.get("torch_module_file") or "")).resolve().is_relative_to(Path(context["compiler_overlay"]).resolve()):
                            result.update(status="incompatible", reason="deepx_compiler_overlay_not_effective")
                        if payload.get("operations_probe_status") == "failed" and result.get("status") != "incompatible":
                            result.update(status="incompatible", reason="deepx_compiler_gpu_operation_failed")
                        return result
        return {**base, "status": "unknown", "reason": "deepx_compiler_probe_cancelled" if cp.returncode == 130 else "deepx_compiler_cuda_probe_unavailable",
                "output_tail": (cp.stdout or "")[-1000:]}
    except Exception as exc:
        return {**base, "status": "unknown", "reason": "deepx_compiler_probe_timeout" if isinstance(exc, subprocess.TimeoutExpired) else "deepx_compiler_cuda_probe_unavailable",
                "detail": f"{type(exc).__name__}: {exc}"}


def compiler_cuda_preflight_message(status: Mapping[str, Any]) -> str:
    capability = status.get("device_capability") or []
    target = "sm_" + "".join(str(value) for value in capability) if capability else "unknown"
    return (
        f"{status.get('reason', 'deepx_compiler_cuda_probe_unavailable')}: "
        f"GPU={status.get('device_name', 'unknown')} ({target}); "
        f"Torch={status.get('torch_version', 'unknown')}; "
        f"supported={status.get('compiled_architectures', [])}; "
        f"compiler_python={status.get('compiler_python', '')}"
    )


def _available_dx_com_wheels(root: Path, limit: int = 20) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    try:
        for p in sorted(root.glob("**/dx_com-*.whl")):
            name = p.name
            tags = []
            for part in name.split("-"):
                if part.startswith("cp") and part[2:5].isdigit():
                    tags.append(part)
            out.append({"path": str(p), "name": name, "python_tags": ",".join(sorted(set(tags)))})
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


def _static_compiler_identity(venv: Path, context: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Read installed distribution identity without executing vendor Python."""
    import hashlib
    import importlib.metadata
    sites = sorted((venv / "lib").glob("python*/site-packages"))
    tag = ""
    try:
        cfg = (venv / "pyvenv.cfg").read_text(encoding="utf-8")
        match = re.search(r"(?m)^version\s*=\s*(\d+)\.(\d+)", cfg)
        if match:
            tag = "cp" + match[1] + match[2]
    except OSError:
        pass
    if not tag and len(sites) == 1:
        match = re.fullmatch(r"python(\d+)\.(\d+)", sites[0].parent.name)
        if match:
            tag = "cp" + match[1] + match[2]
    search = [Path(str(context["compiler_overlay"]))] if context.get("compiler_overlay") else []
    search += sites
    for site in search:
        module = site / "dx_com" / "__init__.py"
        if not module.is_file():
            module = site / "dx_com.py"
        if not module.is_file():
            continue
        distribution, version = "", ""
        for dist in importlib.metadata.distributions(path=[str(site)]):
            if str(dist.metadata.get("Name", "")).lower().replace("-", "_") == "dx_com":
                distribution, version = str(dist.metadata["Name"]), str(dist.version)
                break
        return tag, [{"module": "dx_com", "ok": True, "module_file": str(module),
                      "module_file_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
                      "distribution": distribution, "package_version": version,
                      "identity_source": "installed_files_without_import"}]
    return tag, []


def inspect_deepx_environment(
    *,
    root: str | os.PathLike[str] | None = None,
    probe: bool = False,
    probe_import: bool = False,
    path_only: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Inspect local DeepX build/runtime evidence.

    This is deliberately a status/config MVP. It does not install or modify
    DX-RT drivers/firmware.  The caller can pass ``root`` from the GUI entry;
    ``auto``/empty falls back to environment variables and common paths.
    """
    cfg = dict(config or {})
    context_error = ""
    try:
        compiler_context = resolve_compiler_context(cfg, root=root)
    except (OSError, ValueError) as exc:
        context_error = str(exc)
        compiler_context = {}
    root_val = root if root is not None else cfg.get("dx_all_suite_root")
    dx_root = Path(compiler_context["dx_all_suite_root"]) if compiler_context else (_expand(root_val) if not _is_auto(root_val) else default_dx_all_suite_root())
    dx_root_status = _path_status(dx_root)
    comp_venv = _expand(cfg.get("compiler_venv") or cfg.get("venv") or (_venv_from_activate(cfg.get("venv_activate")) or default_compiler_venv(dx_root)))
    run_venv = _expand(cfg.get("runtime_venv") or (_venv_from_activate(cfg.get("runtime_venv_activate")) or default_runtime_venv(dx_root)))
    cache_dir = _expand(cfg.get("cache_dir") or compiler_context.get("cache_dir") or default_cache_dir())

    tools = _which_many(["dxrt-cli", "parse_model", "run_model", "dxcom", "dx_com"])
    compiler_venv_tools = _which_in_venv(comp_venv, ["dxcom", "dx_com"])
    runtime_venv_tools = _which_in_venv(run_venv, ["dxrt-cli", "parse_model", "run_model"])
    dxrt_devs = []
    try:
        dxrt_devs = sorted(str(p) for p in Path("/dev").glob("dxrt*"))
    except Exception:
        pass

    comp_py = Path(compiler_context.get("compiler_python") or _venv_python(comp_venv))
    comp_venv = Path(compiler_context.get("compiler_venv") or comp_venv)
    compiler_venv_tools = _which_in_venv(comp_venv, ["dxcom", "dx_com"])
    run_py = _venv_python(run_venv)
    compiler_probe_blocked = compiler_dispatch_forbidden() or path_only or bool(context_error)
    # A cache-verification canary may inspect paths and receipts only.  Even a
    # seemingly harmless version/import probe starts the compiler-venv Python;
    # importing dx_com can initialise compiler components before the later
    # compile fence is reached.  Keep this inspection entirely process-free.
    comp_py_tag = "" if compiler_probe_blocked else _python_tag(
        comp_py,
        process_registry=process_registry,
        cancel_event=cancel_event,
    )
    dx_com_wheels = _available_dx_com_wheels(dx_root) if _safe_is_dir(dx_root) else []
    do_probe = bool(probe or probe_import) and not compiler_probe_blocked
    compiler_modules: list[dict[str, Any]] = []
    runtime_modules: list[dict[str, Any]] = []
    if compiler_probe_blocked:
        comp_py_tag, compiler_modules = _static_compiler_identity(comp_venv, compiler_context)
    if do_probe:
        # Required for the tool's ONNX -> DXNN path. DX-COM exposes
        # the public compiler package as ``dx_com`` and the console entry point
        # as ``dxcom``/``dx_com``.
        compiler_modules.append(
            _probe_python_import(
                comp_py,
                "dx_com",
                process_registry=process_registry,
                cancel_event=cancel_event,
                compiler_scope=True,
                compiler_context=compiler_context,
            )
        )
        for mod in ("dx_engine",):
            runtime_modules.append(
                _probe_python_import(
                    run_py,
                    mod,
                    process_registry=process_registry,
                    cancel_event=cancel_event,
                )
            )

    compiler_cli = compiler_venv_tools.get("dxcom") or compiler_venv_tools.get("dx_com") or tools.get("dxcom") or tools.get("dx_com")
    compiler_base_ready = _safe_is_dir(dx_root) and _safe_exists(comp_py)
    compiler_required_import_ok = True
    if do_probe and compiler_modules:
        compiler_required_import_ok = any(bool(x.get("ok")) for x in compiler_modules)
    compiler_ready = bool(compiler_base_ready and compiler_required_import_ok and compiler_cli)
    # This metadata check also runs during the ordinary non-import preflight:
    # a valid dxcom path does not establish that its Torch supports this GPU.
    compiler_cuda = ({"status": "not_probed", "reason": "cache_lookup_before_compiler_probe"}
                     if compiler_probe_blocked else probe_compiler_cuda_architecture(
                         comp_py, process_registry=process_registry, cancel_event=cancel_event,
                         compiler_context=compiler_context))
    if context_error:
        compiler_cuda = {"status": "incompatible", "reason": context_error.split(":")[0], "detail": context_error}
    if compiler_cuda.get("status") == "incompatible":
        compiler_ready = False

    runtime_tool_ok = any(runtime_venv_tools.get(x) or tools.get(x) for x in ("dxrt-cli", "run_model", "parse_model"))
    runtime_ready = _safe_exists(run_py) or runtime_tool_ok or bool(dxrt_devs)
    if do_probe and runtime_modules:
        runtime_ready = runtime_ready and any(bool(x.get("ok")) for x in runtime_modules)

    ready = bool(compiler_ready and runtime_ready)
    ok = bool(compiler_ready or runtime_ready)
    status = "ok" if ready else ("partial" if ok else "missing")
    hints = []
    if compiler_cuda.get("status") in {"incompatible", "unknown"}:
        hints.append(compiler_cuda_preflight_message(compiler_cuda))
    if not _safe_is_dir(dx_root):
        if dx_root_status == "permission_denied":
            hints.append(f"dx-all-suite root is not accessible due to permissions: {dx_root}. Use a writable path such as ~/dx-all-suite or edit ~/.onnx_splitpoint_tool/build_environments.yaml.")
        else:
            hints.append(f"dx-all-suite root not found: {dx_root}. Set DEEPX_DX_ALL_SUITE_ROOT or edit ~/.onnx_splitpoint_tool/build_environments.yaml.")
    if not _safe_exists(comp_py):
        hints.append(f"DeepX compiler venv python not found: {comp_py}")
    if not runtime_ready:
        hints.append("No DeepX runtime evidence found. On the NX, check /dev/dxrt*, dxrt-cli, parse_model/run_model, and dx_engine.")
    if do_probe and compiler_modules and not any(bool(x.get("ok")) for x in compiler_modules):
        hints.append("Required DeepX compiler import dx_com failed. Verify the dx-compiler venv or run Install compiler.")
        if dx_com_wheels:
            tags = sorted({t for w in dx_com_wheels for t in str(w.get("python_tags") or "").split(",") if t})
            if tags and comp_py_tag and comp_py_tag not in tags:
                hints.append(f"DX-COM wheel Python tag mismatch: compiler venv is {comp_py_tag}, available dx_com wheel tags are {', '.join(tags)}. Use Python 3.11 if only cp311 wheels are present.")
            elif tags:
                hints.append(f"Available dx_com wheel tags: {', '.join(tags)}. If imports still fail, rerun Install compiler with force-recreate.")
        else:
            hints.append("No dx_com-*.whl was found under dx-all-suite yet. Run Install compiler; the tool will download/install the compatible DX-COM wheel directly and only uses upstream install.sh when explicitly requested.")
    if compiler_base_ready and not compiler_cli:
        hints.append("Required DeepX compiler executable was not found in the compiler venv. The tool builds DXNN through the dxcom console entry point, so compiler status is partial until venv/bin/dxcom or venv/bin/dx_com exists.")
    if do_probe and runtime_modules and not any(bool(x.get("ok")) for x in runtime_modules):
        hints.append("dx_engine import failed. Verify the DX-RT Python package in the runtime venv.")

    return {
        "backend": "deepx_m1",
        "status": status,
        "ok": ok,
        "ready": ready,
        "compiler_ready": bool(compiler_ready),
        "runtime_ready": bool(runtime_ready),
        "dx_all_suite_root": str(dx_root),
        "dx_all_suite_root_status": dx_root_status,
        "compiler_venv": str(comp_venv),
        "compiler_python": str(comp_py),
        "compiler_overlay": compiler_context.get("compiler_overlay", ""),
        "compiler_selection_source": compiler_context.get("compiler_selection_source", "unresolved"),
        "compiler_python_tag": comp_py_tag,
        "probe_blocked_by_artifact_policy": bool(
            compiler_probe_blocked and (probe or probe_import)
        ),
        "compiler_cli": str(compiler_cli or ""),
        "compiler_venv_tools": compiler_venv_tools,
        "available_dx_com_wheels": dx_com_wheels,
        "runtime_venv": str(run_venv),
        "runtime_python": str(run_py),
        "runtime_venv_tools": runtime_venv_tools,
        "cache_dir": str(cache_dir),
        "tools": tools,
        "device_paths": dxrt_devs,
        "compiler_imports": compiler_modules,
        "compiler_cuda_preflight": compiler_cuda,
        "runtime_imports": runtime_modules,
        "hints": hints,
    }


def format_deepx_status_text(*, probe_import: bool = False, status: Optional[Mapping[str, Any]] = None) -> str:
    st = dict(status or inspect_deepx_environment(probe_import=probe_import))
    lines = [
        "DeepX DX-M1 environment status",
        "",
        f"Status: {st.get('status')}  ready={bool(st.get('ready'))}  ok={bool(st.get('ok'))}",
        f"Compiler ready: {bool(st.get('compiler_ready'))}",
        f"Runtime ready : {bool(st.get('runtime_ready'))}",
        f"dx-all-suite : {st.get('dx_all_suite_root')}",
        f"Compiler venv: {st.get('compiler_venv')}",
        f"Compiler tag : {st.get('compiler_python_tag') or '-'}",
        f"Compiler CLI : {st.get('compiler_cli') or '-'}",
        f"Compiler CUDA: {(st.get('compiler_cuda_preflight') or {}).get('status', 'not_probed')}",
        f"Compiler Torch: {(st.get('compiler_cuda_preflight') or {}).get('torch_version', '-')}",
        f"Compiler overlay: {(st.get('compiler_cuda_preflight') or {}).get('compiler_overlay', '-')}",
        "Compiler API : dx_com Python package + dxcom entry point",
        f"Runtime venv : {st.get('runtime_venv')}",
        f"Cache dir    : {st.get('cache_dir')}",
        "",
        "Global tools:",
    ]
    for name, path in (st.get("tools") or {}).items():
        if name in {"dxcom", "dx_com"} and not path:
            lines.append(f"  {name}: - (global CLI not required; compiler venv entry point is checked separately)")
        else:
            lines.append(f"  {name}: {path or '-'}")
    vtools = st.get("compiler_venv_tools") or {}
    if vtools:
        lines.append("")
        lines.append("Compiler venv tools:")
        for name in ("dxcom", "dx_com"):
            lines.append(f"  {name}: {vtools.get(name) or '-'}")
    rtools = st.get("runtime_venv_tools") or {}
    if rtools:
        lines.append("")
        lines.append("Runtime venv tools:")
        for name in ("dxrt-cli", "parse_model", "run_model"):
            lines.append(f"  {name}: {rtools.get(name) or '-'}")
    wheels = st.get("available_dx_com_wheels") or []
    if wheels:
        lines.append("")
        lines.append("DX-COM wheels:")
        seen: set[str] = set()
        for w in wheels[:10]:
            name = str(w.get('name') or '')
            key = f"{name}|{w.get('python_tags') or '-'}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"  {name}  tags={w.get('python_tags') or '-'}")
    lines.append("")
    lines.append("Device paths:")
    devs = st.get("device_paths") or []
    if devs:
        for d in devs:
            lines.append(f"  {d}")
    else:
        lines.append("  -")
    records = st.get("compiler_imports") or []
    if records:
        lines.append("")
        lines.append("Required compiler import probes:")
        for r in records:
            lines.append(f"  {r.get('module')}: {'OK' if r.get('ok') else 'FAIL'}")
            tail = str(r.get("output_tail") or r.get("reason") or "").strip()
            if tail:
                lines.append("    " + tail.replace("\n", "\n    ")[-1200:])
    records = st.get("runtime_imports") or []
    if records:
        lines.append("")
        lines.append("Runtime import probes:")
        for r in records:
            lines.append(f"  {r.get('module')}: {'OK' if r.get('ok') else 'FAIL'}")
            tail = str(r.get("output_tail") or r.get("reason") or "").strip()
            if tail:
                lines.append("    " + tail.replace("\n", "\n    ")[-1200:])
    if st.get("compiler_ready"):
        lines.append("")
        lines.append("Note:")
        lines.append("  The required DeepX compiler path is dx_com import + dxcom/dx_com entry point in the compiler venv.")
        lines.append("  A global dxcom binary is not required.")
        if not (st.get("device_paths") or []):
            lines.append("  No local /dev/dxrt* device was found. That is OK for a build host; runtime hardware can still be verified on the remote NX.")
    hints = st.get("hints") or []
    if hints:
        lines.append("")
        lines.append("Hints:")
        for h in hints:
            lines.append(f"  - {h}")
    return "\n".join(lines)


def default_build_environment_yaml() -> str:
    dx_root = default_dx_all_suite_root()
    comp = default_compiler_venv(dx_root)
    runv = default_runtime_venv(dx_root)
    cache = default_cache_dir()
    return f"""build_environments:
  - id: hailo8_dfc_managed
    kind: hailo8_dfc
    host: local
    shell: bash
    workdir: ~/.onnx_splitpoint_tool/hailo/builds/hailo8
    venv_activate: "source ~/.onnx_splitpoint_tool/hailo/venv_hailo8/bin/activate"
    cache_dir: ~/Models/BackendArtifacts/hailo

  - id: hailo10_dfc_managed
    kind: hailo10_dfc
    host: local
    shell: bash
    workdir: ~/.onnx_splitpoint_tool/hailo/builds/hailo10
    venv_activate: "source ~/.onnx_splitpoint_tool/hailo/venv_hailo10/bin/activate"
    cache_dir: ~/Models/BackendArtifacts/hailo

  - id: deepx_dxcom_x86
    kind: deepx_dxcom
    host: local
    shell: bash
    dx_all_suite_root: {dx_root}
    venv_activate: "source {comp}/bin/activate"
    compiler_venv: {comp}
    runtime_venv: {runv}
    runtime_venv_activate: "source {runv}/bin/activate"
    cache_dir: {cache}
"""


def provision_deepx_runtime_venv(
    *,
    root: str | os.PathLike[str] | None = None,
    line_callback: Optional[Callable[[str], None]] = None,
    timeout_s: int = 7200,
    run_compiler_install: bool = False,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Best-effort DeepX env repair wrapper used by the GUI.

    The function streams installer output to a central project-local log file
    while the command is running.  This avoids the confusing "empty log folder"
    state during long git checkouts/venv preparation.
    """
    script = repo_root() / "scripts" / "provision_deepx_env.sh"
    log_path = ensure_dir(splitpoint_provisioning_logs_dir()) / "deepx_provision_last.log"

    def _emit(line: str) -> None:
        try:
            if line_callback is not None:
                line_callback(str(line))
        except Exception:
            pass

    logs: list[str] = []
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("DeepX DX-M1 provisioning log\n", encoding="utf-8")
    except Exception:
        pass

    if not _safe_exists(script):
        msg = f"script missing: {script}"
        payload = {"ok": False, "status": "failed", "exit_code": 127, "logs": [msg], "log_path": str(log_path)}
        try:
            log_path.write_text(msg + "\n", encoding="utf-8")
        except Exception:
            pass
        _emit(msg)
        return payload

    cmd = ["bash", str(script), "--repair"]
    effective_timeout_s = int(timeout_s or 0)
    if run_compiler_install:
        cmd += ["--run-compiler-install", "--compiler-install-mode", os.environ.get("DEEPX_COMPILER_INSTALL_MODE", "direct")]
        # DX-COM downloads and installation can legitimately take a while.
        # The default is direct wheel installation, not upstream install.sh, so
        # no hidden sudo prompt should block the GUI. Keep a hard timeout anyway.
        effective_timeout_s = max(effective_timeout_s, int(os.environ.get("DEEPX_COMPILER_INSTALL_TIMEOUT", "3600") or 3600))
    if not _is_auto(root):
        cmd += ["--root", str(root)]
    logs.append("[cmd] " + " ".join(cmd))
    _emit(logs[-1])

    registry = process_registry or current_process_registry()
    if _cancel_requested(cancel_event, registry):
        line = "[cancelled] DeepX provisioning cancelled before process start"
        logs.append(line)
        _emit(line)
        return {
            "ok": False,
            "status": "cancelled",
            "exit_code": 130,
            "cmd": cmd,
            "logs": logs,
            "log_path": str(log_path),
        }

    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as lf:
            lf.write(logs[-1] + "\n")
            lf.flush()
            popen_kwargs: dict[str, Any] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
            }
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True
            elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            proc = subprocess.Popen(cmd, **popen_kwargs)
            if registry is not None:
                try:
                    registry.register(proc, label="deepx-provisioning")
                except BaseException:
                    terminate_process_tree(proc, grace_s=0.5)
                    raise

            def _terminate(grace_s: float) -> None:
                if registry is not None:
                    registry.terminate_registered(proc, grace_s=grace_s)
                else:
                    terminate_process_tree(proc, grace_s=grace_s)

            output_queue: queue.Queue[Any] = queue.Queue()
            reader_done = threading.Event()

            def _read_output() -> None:
                try:
                    assert proc.stdout is not None
                    for raw in proc.stdout:
                        output_queue.put(raw)
                except Exception:
                    pass
                finally:
                    reader_done.set()

            reader = threading.Thread(
                target=_read_output,
                name=f"deepx-provision-output-{proc.pid}",
                daemon=True,
            )
            reader.start()
            start_time = time.monotonic()
            root_exit_seen: float | None = None
            terminal_status = ""
            try:
                while True:
                    try:
                        raw = output_queue.get(timeout=0.1)
                    except queue.Empty:
                        raw = ""
                    batch = [raw] if raw else []
                    while True:
                        try:
                            batch.append(output_queue.get_nowait())
                        except queue.Empty:
                            break
                    for output_line in batch:
                        line = str(output_line).rstrip("\n")
                        if line:
                            logs.append(line)
                            lf.write(line + "\n")
                            lf.flush()
                            _emit(line)

                    if _cancel_requested(cancel_event, registry):
                        terminal_status = "cancelled"
                        _terminate(1.0)
                        break
                    if (
                        int(effective_timeout_s or 0) > 0
                        and time.monotonic() - start_time > int(effective_timeout_s)
                    ):
                        terminal_status = "timeout"
                        _terminate(1.0)
                        break

                    if proc.poll() is not None:
                        if reader_done.is_set() and output_queue.empty():
                            break
                        if root_exit_seen is None:
                            root_exit_seen = time.monotonic()
                        elif time.monotonic() - root_exit_seen >= 0.5:
                            _terminate(0.5)
                            break

                try:
                    rc = int(proc.wait(timeout=2.0))
                except subprocess.TimeoutExpired:
                    _terminate(0.0)
                    try:
                        rc = int(proc.wait(timeout=0.5))
                    except subprocess.TimeoutExpired:
                        rc = 1
                reader.join(timeout=1.0)
            finally:
                if registry is not None:
                    registry.unregister(proc)

            if terminal_status:
                exit_code = 130 if terminal_status == "cancelled" else 124
                line = (
                    "[cancelled] DeepX provisioning cancelled by Evaluation Workflow"
                    if terminal_status == "cancelled"
                    else f"[timeout] provisioning exceeded {effective_timeout_s}s"
                )
                logs.append(line)
                lf.write(line + "\n")
                lf.flush()
                _emit(line)
                return {
                    "ok": False,
                    "status": terminal_status,
                    "exit_code": exit_code,
                    "cmd": cmd,
                    "logs": logs,
                    "log_path": str(log_path),
                    "timeout_s": effective_timeout_s,
                }
            done = f"[done] exit_code={rc}"
            logs.append(done)
            lf.write(done + "\n")
            lf.flush()
            _emit(done)
            # rc=2 is a deliberate "partial" state from the script:
            # repo/runtime can be usable while DX-COM compiler imports are
            # missing.  Keep that truth in the payload so the GUI does not say
            # "provisioned" for a compiler-incomplete environment.
            status = "ok" if rc == 0 else ("partial" if rc in (2, 4, 5) else "failed")
            return {"ok": rc == 0, "status": status, "exit_code": rc, "cmd": cmd, "logs": logs, "log_path": str(log_path), "run_compiler_install": bool(run_compiler_install)}
    except Exception as exc:
        line = f"{type(exc).__name__}: {exc}"
        logs.append(line)
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as lf:
                lf.write(line + "\n")
        except Exception:
            pass
        _emit(line)
        return {"ok": False, "status": "failed", "exit_code": 1, "cmd": cmd, "logs": logs, "log_path": str(log_path)}

def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Inspect DeepX DX-M1 build/runtime environment")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--probe-import", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--root", default="auto")
    ns = ap.parse_args(argv)
    st = inspect_deepx_environment(root=ns.root, probe_import=bool(ns.probe_import))
    if ns.json:
        print(json.dumps(st, indent=2))
    else:
        print(format_deepx_status_text(probe_import=bool(ns.probe_import), status=st))
    if st.get("ready"):
        return 0
    if st.get("ok"):
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
