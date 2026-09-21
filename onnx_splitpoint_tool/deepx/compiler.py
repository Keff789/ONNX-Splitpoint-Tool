from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .env_status import (
    compiler_cuda_preflight_message,
    compiler_subprocess_environment,
    default_compiler_venv,
    default_dx_all_suite_root,
    probe_compiler_cuda_architecture,
    resolve_compiler_context,
    compiler_python_command,
)
from ..cache_verify_policy import (
    cache_miss_blocked_message,
    compiler_dispatch_forbidden,
)
from ..process_control import current_process_registry, terminate_process_tree, ProcessTreeRegistry


@dataclass
class DeepXBuildResult:
    ok: bool
    status: str
    onnx_path: str
    config_path: str
    output_dir: str
    dxnn_path: str = ""
    log_path: str = ""
    returncode: int | None = None
    command: str = ""
    message: str = ""


def _expand(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def _compiler_shell_prefix(activate: Path, context: Optional[Mapping[str, Any]] = None) -> str:
    prefix = f"set -e; source {shlex.quote(str(activate))};"
    child_env = compiler_subprocess_environment(context=context)
    if child_env is not None:
        # Reapply after login-shell/venv startup, which may reset PYTHONPATH.
        # The tensor loader then prepends its existing sitecustomize directory.
        prefix += f" export PYTHONPATH={shlex.quote(child_env['PYTHONPATH'])};"
    return prefix


def _find_dxnn(out_dir: Path) -> Optional[Path]:
    try:
        matches = sorted([p for p in out_dir.rglob("*.dxnn") if p.is_file()], key=lambda p: (p.stat().st_mtime, str(p)))
        return matches[-1] if matches else None
    except Exception:
        return None


def _dxnn_snapshot(out_dir: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in out_dir.rglob("*.dxnn") if p.is_file()}


def _fresh_dxnn(out_dir: Path, before: Mapping[str, tuple[int, int]]) -> Optional[Path]:
    candidate = _find_dxnn(out_dir)
    if candidate is None or candidate.stat().st_size <= 0:
        return None
    current = (candidate.stat().st_size, candidate.stat().st_mtime_ns)
    return candidate if before.get(str(candidate)) != current else None


def _cuda_preflight_failure(
    *, venv: Path, onnx_path: Path, config_path: Path, output_dir: Path,
    log_path: Path, process_registry: Any = None, cancel_event: Any = None,
    audit: Optional[dict[str, Any]] = None,
    compiler_context: Optional[Mapping[str, Any]] = None,
) -> Optional[DeepXBuildResult]:
    # Cache lookup happens before these compiler entry points. This is an
    # infrastructure guard only; it must never become negative split evidence.
    status = probe_compiler_cuda_architecture(
        venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        process_registry=process_registry, cancel_event=cancel_event,
        compiler_context=compiler_context, operation_probe=True,
    )
    if audit is not None:
        audit.update(status)
    if status.get("status") == "not_probed" or (status.get("status") in {"compatible", "not_applicable"} and status.get("operations_probe_status") in {"pass", "not_applicable"}):
        return None
    message = compiler_cuda_preflight_message(status)
    result = DeepXBuildResult(
        ok=False, status=("cancelled" if status.get("reason") == "deepx_compiler_probe_cancelled" else "compiler_environment_unverified" if status.get("status") == "unknown" else "compiler_environment_incompatible"), onnx_path=str(onnx_path),
        config_path=str(config_path), output_dir=str(output_dir),
        log_path=str(log_path), message=message,
    )
    log_path.write_text(message + "\n", encoding="utf-8")
    manifest = {**asdict(result), "compiler_cuda_preflight": status, "compiler_dispatched": False}
    (output_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return result


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
    """Drain captured output without trusting descendants to close the pipe."""

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


def _run_owned_compiler(
    cmd: list[str],
    *,
    timeout_s: int,
    process_registry: Any = None,
    cancel_event: Any = None,
    context: Optional[Mapping[str, Any]] = None,
) -> subprocess.CompletedProcess[str]:
    """Capture a DX-COM command without leaving its process tree unowned."""

    if compiler_dispatch_forbidden():
        return subprocess.CompletedProcess(
            cmd,
            78,
            cache_miss_blocked_message("deepx_dx_com"),
            "",
        )

    registry = process_registry or current_process_registry()
    local_registry = registry is None
    if local_registry:
        registry = ProcessTreeRegistry()
    if _cancel_requested(cancel_event, registry):
        return subprocess.CompletedProcess(cmd, 130, "CANCELLED before DX-COM start", "")
    popen_kwargs: dict[str, Any] = {
        "text": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    child_env = compiler_subprocess_environment(context=context)
    if child_env is not None:
        popen_kwargs["env"] = child_env
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    from ..backend_backfill import reserve_active_build
    reserve_active_build(cmd)
    proc = subprocess.Popen(cmd, **popen_kwargs)
    if registry is not None:
        try:
            registry.register(proc, label="deepx-compiler")
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
                    _terminate(3.0)
                output = _bounded_collect_after_stop(proc, _terminate)
                return subprocess.CompletedProcess(
                    cmd,
                    130,
                    (output or "") + "\nCANCELLED by Evaluation Workflow",
                    "",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate(0.5)
                output = _bounded_collect_after_stop(proc, _terminate)
                raise subprocess.TimeoutExpired(
                    cmd,
                    timeout_s,
                    output=output,
                )
            try:
                output, _ = proc.communicate(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        returncode = int(proc.returncode or 0)
        if _cancel_requested(cancel_event, registry):
            returncode = 130
            output = (output or "") + "\nCANCELLED by Evaluation Workflow"
        return subprocess.CompletedProcess(cmd, returncode, output or "", "")
    except BaseException:
        if proc.poll() is None:
            _terminate(0.5)
        raise
    finally:
        if registry is not None:
            registry.unregister(proc)
            if local_registry:
                registry.assert_quiescent()


def compile_dxnn(
    *,
    onnx_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    compiler_root: str | Path | None = None,
    compiler_venv: str | Path | None = None,
    compiler_overlay: str | Path | None = None,
    build_config: Optional[Mapping[str, Any]] = None,
    opt_level: int = 0,
    timeout_s: int = 7200,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> DeepXBuildResult:
    """Compile ONNX -> DXNN with dxcom.

    This wrapper is intentionally artifact-first and non-GUI.  It writes
    deepx_build.log and build_manifest.json.  It does not install drivers or
    firmware; provisioning is handled by scripts/provision_deepx_env.sh.
    """
    onnx_p = _expand(onnx_path)
    cfg_p = _expand(config_path)
    out_dir = _expand(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "deepx_build.log"
    selected_cfg = {**dict(build_config or {})}
    if compiler_venv:
        selected_cfg["compiler_venv"] = str(compiler_venv)
    if compiler_overlay:
        selected_cfg["compiler_overlay"] = str(compiler_overlay)
    try:
        context = resolve_compiler_context(selected_cfg, root=compiler_root)
    except (ValueError, OSError) as exc:
        message = str(exc)
        log_path.write_text(message + "\n", encoding="utf-8")
        result = DeepXBuildResult(False, "compiler_environment_incompatible", str(onnx_p), str(cfg_p), str(out_dir), log_path=str(log_path), message=message)
        (out_dir / "build_manifest.json").write_text(json.dumps({**asdict(result), "compiler_dispatched": False, "compiler_cuda_preflight": {"status": "incompatible", "reason": message.split(":")[0], "detail": message}}) + "\n", encoding="utf-8")
        return result
    root = Path(context["dx_all_suite_root"])
    venv = Path(context["compiler_venv"])
    compiler_environment: dict[str, Any] = {}
    preflight_failure = _cuda_preflight_failure(
        venv=venv, onnx_path=onnx_p, config_path=cfg_p, output_dir=out_dir,
        log_path=log_path, process_registry=process_registry, cancel_event=cancel_event,
        audit=compiler_environment, compiler_context=context,
    )
    if preflight_failure is not None:
        return preflight_failure
    activate = venv / "bin" / "activate"
    dxcom_exe = venv / "bin" / "dxcom"
    if not dxcom_exe.exists():
        alt = venv / "bin" / "dx_com"
        dxcom_exe = alt if alt.exists() else Path("dxcom")
    cli_command = shlex.join(compiler_python_command(context, code="import runpy, sys; entry=sys.argv.pop(1); sys.argv[0]=entry; runpy.run_path(entry, run_name='__main__')") + [str(dxcom_exe)])
    cmd = f"{_compiler_shell_prefix(activate, context)} {cli_command} -m {shlex.quote(str(onnx_p))} -c {shlex.quote(str(cfg_p))} -o {shlex.quote(str(out_dir))}"
    if opt_level is not None:
        cmd += f" --opt_level={int(opt_level)}"

    before_dxnn = _dxnn_snapshot(out_dir)
    start = time.time()
    try:
        cp = _run_owned_compiler(
            ["bash", "-lc", cmd], timeout_s=int(timeout_s),
            process_registry=process_registry, cancel_event=cancel_event, context=context,
        )
        output = cp.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        dxnn = _fresh_dxnn(out_dir, before_dxnn)
        ok = bool(cp.returncode == 0 and dxnn)
        status = "ok" if ok else ("dxnn_missing" if cp.returncode == 0 else "cancelled" if cp.returncode == 130 else "compile_failed")
        res = DeepXBuildResult(ok=ok, status=status, onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), dxnn_path=str(dxnn or ""), log_path=str(log_path), returncode=int(cp.returncode), command=cmd, message=("DXNN built" if ok else output[-1200:]))
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        log_path.write_text(str(output) + f"\n[timeout after {timeout_s}s]\n", encoding="utf-8")
        res = DeepXBuildResult(ok=False, status="timeout", onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), log_path=str(log_path), command=cmd, message=f"timeout after {timeout_s}s")
    except Exception as exc:
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        res = DeepXBuildResult(ok=False, status="exception", onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), log_path=str(log_path), command=cmd, message=f"{type(exc).__name__}: {exc}")

    manifest = asdict(res)
    manifest["compiler_cuda_preflight"] = compiler_environment
    manifest["elapsed_s"] = round(time.time() - start, 3)
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return res


def compile_dxnn_with_npz_cv2_shim(
    *,
    onnx_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    input_name: str,
    input_shape: list[int] | tuple[int, ...],
    compiler_root: str | Path | None = None,
    compiler_venv: str | Path | None = None,
    compiler_overlay: str | Path | None = None,
    build_config: Optional[Mapping[str, Any]] = None,
    opt_level: int = 0,
    timeout_s: int = 7200,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> DeepXBuildResult:
    """Compile ONNX -> DXNN while teaching DX-COM's image default_loader to read NPZ/NPY tensors.

    This is an experimental bridge for single-input split Stage2 models.  It runs
    DX-COM through its Python API in the compiler venv and monkey-patches
    ``cv2.imread`` inside that process so ``default_loader`` can consume
    activation-proxy NPZ/NPY files instead of returning ``None``.
    """
    onnx_p = _expand(onnx_path)
    cfg_p = _expand(config_path)
    out_dir = _expand(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "deepx_build.log"
    selected_cfg = {**dict(build_config or {})}
    if compiler_venv:
        selected_cfg["compiler_venv"] = str(compiler_venv)
    if compiler_overlay:
        selected_cfg["compiler_overlay"] = str(compiler_overlay)
    try:
        context = resolve_compiler_context(selected_cfg, root=compiler_root)
    except (ValueError, OSError) as exc:
        message = str(exc)
        log_path.write_text(message + "\n", encoding="utf-8")
        result = DeepXBuildResult(False, "compiler_environment_incompatible", str(onnx_p), str(cfg_p), str(out_dir), log_path=str(log_path), message=message)
        (out_dir / "build_manifest.json").write_text(json.dumps({**asdict(result), "compiler_dispatched": False, "compiler_cuda_preflight": {"status": "incompatible", "reason": message.split(":")[0], "detail": message}}) + "\n", encoding="utf-8")
        return result
    root = Path(context["dx_all_suite_root"])
    venv = Path(context["compiler_venv"])
    compiler_environment: dict[str, Any] = {}
    preflight_failure = _cuda_preflight_failure(
        venv=venv, onnx_path=onnx_p, config_path=cfg_p, output_dir=out_dir,
        log_path=log_path, process_registry=process_registry, cancel_event=cancel_event,
        audit=compiler_environment, compiler_context=context,
    )
    if preflight_failure is not None:
        return preflight_failure
    activate = venv / "bin" / "activate"
    script_path = out_dir / "dxcom_npz_cv2_shim_compile.py"
    shape = [int(x) for x in list(input_shape or [])]
    shim_payload = {
        "onnx_path": str(onnx_p),
        "config_path": str(cfg_p),
        "output_dir": str(out_dir),
        "input_name": str(input_name),
        "input_shape": shape,
        "opt_level": int(opt_level),
    }
    # Build the runtime script as plain text. It is intentionally self-contained
    # because it executes inside the dx-compiler venv, not this tool's venv.
    script = """
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

PAYLOAD = __PAYLOAD__
ONNX_PATH = PAYLOAD["onnx_path"]
CONFIG_PATH = PAYLOAD["config_path"]
OUTPUT_DIR = PAYLOAD["output_dir"]
INPUT_NAME = PAYLOAD["input_name"]
INPUT_SHAPE = list(PAYLOAD.get("input_shape") or [])
OPT_LEVEL = int(PAYLOAD.get("opt_level") or 0)

try:
    import cv2  # type: ignore
except Exception as exc:
    raise RuntimeError(f"cv2 import failed in dx-compiler venv: {type(exc).__name__}: {exc}")

_orig_imread = cv2.imread

def _activation_tensor_from_file(path):
    p = str(path)
    low = p.lower()
    if low.endswith(".npz"):
        data = np.load(p)
        files = list(getattr(data, "files", []) or [])
        if not files:
            raise RuntimeError(f"NPZ sample has no tensors: {p}")
        key = INPUT_NAME if INPUT_NAME in files else files[0]
        arr = np.asarray(data[key], dtype=np.float32)
    elif low.endswith(".npy"):
        arr = np.asarray(np.load(p), dtype=np.float32)
    else:
        return None
    target = list(INPUT_SHAPE or [])
    if target and len(target) == arr.ndim + 1 and int(target[0]) == 1:
        arr = arr.reshape([1] + list(arr.shape))
    if target and len(target) == arr.ndim:
        try:
            known = [int(x) for x in target]
            if all(v > 0 for v in known) and int(np.prod(known)) == int(arr.size):
                arr = arr.reshape(known)
        except Exception:
            pass
    return np.ascontiguousarray(arr.astype(np.float32, copy=False))

def _patched_imread(path, flags=cv2.IMREAD_COLOR):
    arr = _activation_tensor_from_file(path)
    if arr is not None:
        return arr
    return _orig_imread(path, flags)

cv2.imread = _patched_imread

import dx_com  # type: ignore
print("[dxcom-npz-shim] model=", ONNX_PATH)
print("[dxcom-npz-shim] config=", CONFIG_PATH)
print("[dxcom-npz-shim] output=", OUTPUT_DIR)
print("[dxcom-npz-shim] input=", INPUT_NAME, INPUT_SHAPE)
try:
    result = dx_com.compile(model=ONNX_PATH, config=CONFIG_PATH, output_dir=OUTPUT_DIR, opt_level=OPT_LEVEL)
except TypeError:
    result = dx_com.compile(model=ONNX_PATH, config=CONFIG_PATH, output_dir=OUTPUT_DIR)
print("[dxcom-npz-shim] compile_result=", result)
""".replace("__PAYLOAD__", json.dumps(shim_payload, ensure_ascii=False))
    script_path.write_text(script, encoding="utf-8")
    cmd = f"{_compiler_shell_prefix(activate, context)} python {shlex.quote(str(script_path))}"
    before_dxnn = _dxnn_snapshot(out_dir)
    start = time.time()
    try:
        cp = _run_owned_compiler(
            ["bash", "-lc", cmd], timeout_s=int(timeout_s),
            process_registry=process_registry, cancel_event=cancel_event, context=context,
        )
        output = cp.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        dxnn = _fresh_dxnn(out_dir, before_dxnn)
        ok = bool(cp.returncode == 0 and dxnn)
        status = "ok" if ok else ("dxnn_missing" if cp.returncode == 0 else "cancelled" if cp.returncode == 130 else "compile_failed")
        res = DeepXBuildResult(ok=ok, status=status, onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), dxnn_path=str(dxnn or ""), log_path=str(log_path), returncode=int(cp.returncode), command=cmd, message=("DXNN built" if ok else output[-1600:]))
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        log_path.write_text(str(output) + f"\n[timeout after {timeout_s}s]\n", encoding="utf-8")
        res = DeepXBuildResult(ok=False, status="timeout", onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), log_path=str(log_path), command=cmd, message=f"timeout after {timeout_s}s")
    except Exception as exc:
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        res = DeepXBuildResult(ok=False, status="exception", onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), log_path=str(log_path), command=cmd, message=f"{type(exc).__name__}: {exc}")
    manifest = asdict(res)
    manifest["compiler_cuda_preflight"] = compiler_environment
    manifest["elapsed_s"] = round(time.time() - start, 3)
    manifest["npz_cv2_shim"] = True
    manifest["npz_cv2_shim_input_name"] = str(input_name)
    manifest["npz_cv2_shim_input_shape"] = shape
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return res


def compile_dxnn_with_tensor_loader(
    *,
    onnx_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    activation_manifest: str | Path,
    compiler_root: str | Path | None = None,
    compiler_venv: str | Path | None = None,
    compiler_overlay: str | Path | None = None,
    build_config: Optional[Mapping[str, Any]] = None,
    opt_level: int = 0,
    timeout_s: int = 7200,
    process_registry: Any = None,
    cancel_event: Any = None,
) -> DeepXBuildResult:
    """Compile ONNX -> DXNN using an experimental feature-tensor calibration loader.

    This wrapper runs DX-COM through a Python shim inside the DX-COM venv.  The
    shim installs best-effort patches for DX-COM's image-oriented default loader
    so ``sample_*.npz/.npy`` activation proxy samples can be used for split
    Stage2 calibration.
    """
    onnx_p = _expand(onnx_path)
    cfg_p = _expand(config_path)
    manifest_p = _expand(activation_manifest)
    out_dir = _expand(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "deepx_stage2_tensor_loader_build.log"
    selected_cfg = {**dict(build_config or {})}
    if compiler_venv:
        selected_cfg["compiler_venv"] = str(compiler_venv)
    if compiler_overlay:
        selected_cfg["compiler_overlay"] = str(compiler_overlay)
    try:
        context = resolve_compiler_context(selected_cfg, root=compiler_root)
    except (ValueError, OSError) as exc:
        message = str(exc)
        log_path.write_text(message + "\n", encoding="utf-8")
        result = DeepXBuildResult(False, "compiler_environment_incompatible", str(onnx_p), str(cfg_p), str(out_dir), log_path=str(log_path), message=message)
        (out_dir / "build_manifest.json").write_text(json.dumps({**asdict(result), "compiler_dispatched": False, "compiler_cuda_preflight": {"status": "incompatible", "reason": message.split(":")[0], "detail": message}}) + "\n", encoding="utf-8")
        return result
    root = Path(context["dx_all_suite_root"])
    venv = Path(context["compiler_venv"])
    compiler_environment: dict[str, Any] = {}
    preflight_failure = _cuda_preflight_failure(
        venv=venv, onnx_path=onnx_p, config_path=cfg_p, output_dir=out_dir,
        log_path=log_path, process_registry=process_registry, cancel_event=cancel_event,
        audit=compiler_environment, compiler_context=context,
    )
    if preflight_failure is not None:
        return preflight_failure
    activate = venv / "bin" / "activate"
    py_exe = venv / "bin" / "python"
    if not py_exe.exists():
        py_exe = Path("python3")

    # v54: add a sitecustomize hook next to the DX-COM shim.  Some DX-COM
    # versions spawn dataloader workers; monkey-patches inside the main shim do
    # not automatically propagate to those worker interpreters.  sitecustomize
    # is imported by Python at startup when present on PYTHONPATH, so it lets
    # worker processes read .npz/.npy activation samples through the otherwise
    # image-oriented cv2.imread path as well.
    site_dir = out_dir / "tensor_loader_sitecustomize"
    site_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = site_dir / "sitecustomize.py"
    sitecustomize = r'''
from __future__ import annotations
import json
import os
import pathlib

try:
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None


def _san(name: str) -> str:
    return str(name).replace('/', '_').replace(':', '_').replace('.', '_').replace(' ', '_')


def _load_json(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _tensor_names(manifest, config):
    names = []
    for t in manifest.get('tensors') or []:
        if isinstance(t, dict) and t.get('name'):
            names.append(str(t.get('name')))
    for n in (config.get('inputs') or {}).keys():
        if str(n) not in names:
            names.append(str(n))
    cfg_names = list((config.get('inputs') or {}).keys())
    return [n for n in names if n in cfg_names] + [n for n in cfg_names if n not in names]


def _fit(arr, shape):
    if _np is None:
        return arr
    a = _np.asarray(arr, dtype=_np.float32)
    shp = [int(x) for x in (shape or [])]
    if shp:
        if len(a.shape) == len(shp) - 1 and shp and shp[0] == 1:
            a = a.reshape([1] + list(a.shape))
        try:
            if list(a.shape) != shp and int(_np.prod(a.shape)) == int(_np.prod(shp)):
                a = a.reshape(shp)
        except Exception:
            pass
    return _np.ascontiguousarray(a.astype(_np.float32, copy=False))


def _load_np(path: str, input_names, input_shapes):
    if _np is None:
        return None
    p = pathlib.Path(path)
    if p.suffix.lower() == '.npy':
        arr = _np.load(str(p))
        name = input_names[0] if input_names else 'input'
        return {name: _fit(arr, input_shapes.get(name))}
    data = _np.load(str(p), allow_pickle=False)
    keys = list(data.files)
    out = {}
    for i, name in enumerate(input_names):
        candidates = [name, _san(name), f'input_{i}', f'arr_{i}', str(i)]
        chosen = next((c for c in candidates if c in data), None)
        if chosen is None and len(keys) == 1 and len(input_names) == 1:
            chosen = keys[0]
        if chosen is None:
            sn = _san(name).lower()
            for k in keys:
                sk = _san(k).lower()
                if sk == sn or sn.endswith(sk) or sk.endswith(sn):
                    chosen = k
                    break
        if chosen is None:
            raise KeyError(f'No tensor for input {name!r} in {p.name}; available={keys}')
        out[name] = _fit(data[chosen], input_shapes.get(name))
    return out


def _install_cv2_patch():
    cfg_path = os.environ.get('SPLITPOINT_DXCOM_TENSOR_CONFIG') or ''
    manifest_path = os.environ.get('SPLITPOINT_DXCOM_TENSOR_MANIFEST') or ''
    config = _load_json(cfg_path)
    manifest = _load_json(manifest_path)
    input_shapes = {str(k): [int(x) for x in v] for k, v in (config.get('inputs') or {}).items()}
    input_names = _tensor_names(manifest, config)
    if not input_names:
        input_names = list(input_shapes.keys())
    try:
        import cv2  # type: ignore
    except Exception:
        return
    orig = getattr(cv2, 'imread', None)

    def imread(path, flags=None):
        sp = str(path)
        if sp.lower().endswith(('.npz', '.npy')):
            sample = _load_np(sp, input_names, input_shapes)
            if isinstance(sample, dict):
                if len(sample) == 1:
                    return next(iter(sample.values()))
                return sample.get(input_names[0])
        if orig is None:
            return None
        return orig(path, flags) if flags is not None else orig(path)

    cv2.imread = imread


_install_cv2_patch()
'''
    sitecustomize_path.write_text(sitecustomize, encoding="utf-8")

    shim_path = out_dir / "dxcom_tensor_activation_compile.py"
    shim = r'''#!/usr/bin/env python3
import argparse
import importlib
import json
import pathlib
import sys
import traceback
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except Exception:
    torch = None


def _to_torch_tensor(value):
    if torch is None:
        return value
    try:
        return torch.as_tensor(value, dtype=torch.float32).contiguous()
    except Exception:
        return value


def _to_torch_dict(obj):
    if not isinstance(obj, dict):
        return obj
    return {str(k): _to_torch_tensor(v) for k, v in obj.items()}


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _san(name: str) -> str:
    return str(name).replace('/', '_').replace(':', '_').replace('.', '_').replace(' ', '_')


def _manifest_tensor_names(manifest: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    names = []
    for t in manifest.get('tensors') or []:
        if isinstance(t, dict) and t.get('name'):
            names.append(str(t.get('name')))
    for n in (config.get('inputs') or {}).keys():
        if str(n) not in names:
            names.append(str(n))
    return names


def _fit_shape(arr, shape):
    a = np.asarray(arr, dtype=np.float32)
    if shape:
        shp = [int(x) for x in shape]
        if len(a.shape) == len(shp) - 1 and shp and shp[0] == 1:
            a = a.reshape([1] + list(a.shape))
        try:
            if list(a.shape) != shp and int(np.prod(a.shape)) == int(np.prod(shp)):
                a = a.reshape(shp)
        except Exception:
            pass
    return np.ascontiguousarray(a.astype(np.float32, copy=False))


def _load_np_sample(path: str, input_names: List[str], input_shapes: Dict[str, List[int]]):
    p = pathlib.Path(path)
    if p.suffix.lower() == '.npy':
        arr = np.load(str(p))
        if len(input_names) <= 1:
            name = input_names[0] if input_names else 'input'
            return {name: _fit_shape(arr, input_shapes.get(name))}
        return {name: _fit_shape(arr, input_shapes.get(name)) for name in input_names}
    data = np.load(str(p), allow_pickle=False)
    out = {}
    keys = list(data.files)
    for i, name in enumerate(input_names):
        candidates = [name, _san(name), f'input_{i}', f'arr_{i}', str(i)]
        chosen = None
        for c in candidates:
            if c in data:
                chosen = c
                break
        if chosen is None and len(keys) == 1 and len(input_names) == 1:
            chosen = keys[0]
        if chosen is None:
            sn = _san(name).lower()
            for k in keys:
                sk = _san(k).lower()
                if sk == sn or sn.endswith(sk) or sk.endswith(sn):
                    chosen = k
                    break
        if chosen is None:
            raise KeyError(f'No tensor for input {name!r} in {p.name}; available={keys}')
        out[name] = _fit_shape(data[chosen], input_shapes.get(name))
    return out


class TensorActivationDataset:
    def __init__(self, *args, **kwargs):
        self.files = list(TENSOR_SAMPLE_FILES)
        self.input_names = list(TENSOR_INPUT_NAMES)
        self.input_shapes = dict(TENSOR_INPUT_SHAPES)
        # DX-COM / quant_fx sometimes passes the dataset object directly as
        # a dataloader and expects DataLoader-like attributes.  Keep batches
        # at one sample because activation proxy samples already include the
        # model batch dimension, e.g. [1,C,H,W].
        self.batch_size = int(kwargs.get('batch_size') or kwargs.get('calib_batch_size') or 1)
        if self.batch_size < 1:
            self.batch_size = 1
        self.calib_batch_size = self.batch_size
        self.dataset = self
        self.drop_last = False
        self.shuffle = False
        self.num_workers = 0
        self.pin_memory = False
        self.sample_count = len(self.files)
        self.calibration_num = self.sample_count
        self.num_batches = max(1, (self.sample_count + self.batch_size - 1) // self.batch_size) if self.files else 0
        if not self.files:
            raise RuntimeError('No activation proxy samples found for tensor dataset')
    def __len__(self):
        return len(self.files)
    def __iter__(self):
        for i in range(len(self.files)):
            yield self[i]
    def __getitem__(self, idx):
        fp = self.files[int(idx) % len(self.files)]
        obj = _load_np_sample(fp, self.input_names, self.input_shapes)
        return _to_torch_dict(obj)


def _patch_cv2_imread(input_names, input_shapes):
    try:
        import cv2
    except Exception:
        return False
    orig = getattr(cv2, 'imread', None)
    def imread(path, flags=None):
        sp = str(path)
        if sp.lower().endswith(('.npz', '.npy')):
            data = _load_np_sample(sp, input_names, input_shapes)
            if len(data) == 1:
                return next(iter(data.values()))
            return data[input_names[0]]
        if orig is None:
            return None
        return orig(path, flags) if flags is not None else orig(path)
    cv2.imread = imread
    return True


def _patch_default_dataset():
    patched = []
    mod_names = [
        'dx_com.dataloader.default_dataset',
        'dx_com.dataloader.default_dataloader',
        'dx_com.dataloader',
        'dxcom.dataloader.default_dataset',
        'dxcom.dataloader',
    ]
    for mn in mod_names:
        try:
            mod = importlib.import_module(mn)
        except Exception:
            continue
        for attr in list(vars(mod).keys()):
            low = attr.lower()
            if ('dataset' in low or 'loader' in low) and attr[:1].isupper():
                obj = getattr(mod, attr)
                if isinstance(obj, type):
                    try:
                        setattr(mod, attr, TensorActivationDataset)
                        patched.append(f'{mn}.{attr}')
                    except Exception:
                        pass
        for attr in ('DefaultDataset', 'DefaultDataSet', 'DefaultLoader', 'DefaultDataLoader'):
            if hasattr(mod, attr):
                try:
                    setattr(mod, attr, TensorActivationDataset)
                    patched.append(f'{mn}.{attr}')
                except Exception:
                    pass
    return sorted(set(patched))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--opt-level', type=int, default=0)
    args = ap.parse_args()
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    global TENSOR_INPUT_NAMES, TENSOR_INPUT_SHAPES, TENSOR_SAMPLE_FILES
    TENSOR_INPUT_SHAPES = {str(k): [int(x) for x in v] for k, v in (config.get('inputs') or {}).items()}
    TENSOR_INPUT_NAMES = _manifest_tensor_names(manifest, config)
    cfg_names = list(TENSOR_INPUT_SHAPES.keys())
    TENSOR_INPUT_NAMES = [n for n in TENSOR_INPUT_NAMES if n in TENSOR_INPUT_SHAPES] + [n for n in cfg_names if n not in TENSOR_INPUT_NAMES]
    sample_dir = pathlib.Path(args.manifest).resolve().parent
    TENSOR_SAMPLE_FILES = sorted(str(p) for p in sample_dir.glob('sample_*.npz')) + sorted(str(p) for p in sample_dir.glob('sample_*.npy'))
    if not TENSOR_SAMPLE_FILES:
        raise RuntimeError(f'No sample_*.npz/.npy files next to {args.manifest}')
    print('[tensor-loader] inputs:', TENSOR_INPUT_NAMES)
    print('[tensor-loader] input_shapes:', TENSOR_INPUT_SHAPES)
    print('[tensor-loader] samples:', len(TENSOR_SAMPLE_FILES), 'dir=', sample_dir)
    print('[tensor-loader] cv2 patched:', _patch_cv2_imread(TENSOR_INPUT_NAMES, TENSOR_INPUT_SHAPES))
    import dx_com
    patched = _patch_default_dataset()
    print('[tensor-loader] dataset patches:', patched)
    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(pathlib.Path(args.out) / 'tensor_loader_runtime_manifest.json', 'w', encoding='utf-8') as f:
        json.dump({'inputs': TENSOR_INPUT_NAMES, 'input_shapes': TENSOR_INPUT_SHAPES, 'sample_count': len(TENSOR_SAMPLE_FILES), 'batch_size': 1, 'patched': patched}, f, indent=2)
    if not hasattr(dx_com, 'compile'):
        raise RuntimeError('dx_com.compile API not found in this DX-COM environment')
    kwargs = dict(model=str(args.model), config=str(args.config), output_dir=str(args.out))
    try:
        kwargs['opt_level'] = int(args.opt_level)
        dx_com.compile(**kwargs)
    except TypeError:
        kwargs.pop('opt_level', None)
        dx_com.compile(**kwargs)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
'''
    shim_path.write_text(shim, encoding="utf-8")
    cmd = (
        f"{_compiler_shell_prefix(activate, context)} "
        f"export SPLITPOINT_DXCOM_TENSOR_CONFIG={shlex.quote(str(cfg_p))}; "
        f"export SPLITPOINT_DXCOM_TENSOR_MANIFEST={shlex.quote(str(manifest_p))}; "
        f"export PYTHONPATH={shlex.quote(str(site_dir))}:$PYTHONPATH; "
        f"{shlex.quote(str(py_exe))} {shlex.quote(str(shim_path))} "
        f"--model {shlex.quote(str(onnx_p))} --config {shlex.quote(str(cfg_p))} "
        f"--out {shlex.quote(str(out_dir))} --manifest {shlex.quote(str(manifest_p))} "
        f"--opt-level {int(opt_level)}"
    )
    before_dxnn = _dxnn_snapshot(out_dir)
    start = time.time()
    try:
        cp = _run_owned_compiler(
            ["bash", "-lc", cmd], timeout_s=int(timeout_s),
            process_registry=process_registry, cancel_event=cancel_event, context=context,
        )
        output = cp.stdout or ""
        log_path.write_text(output, encoding="utf-8")
        dxnn = _fresh_dxnn(out_dir, before_dxnn)
        ok = bool(cp.returncode == 0 and dxnn)
        status = "ok" if ok else ("dxnn_missing" if cp.returncode == 0 else "cancelled" if cp.returncode == 130 else "compile_failed")
        res = DeepXBuildResult(ok=ok, status=status, onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), dxnn_path=str(dxnn or ""), log_path=str(log_path), returncode=int(cp.returncode), command=cmd, message=("DXNN built with tensor loader" if ok else output[-1600:]))
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        log_path.write_text(str(output) + f"\n[timeout after {timeout_s}s]\n", encoding="utf-8")
        res = DeepXBuildResult(ok=False, status="timeout", onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), log_path=str(log_path), command=cmd, message=f"timeout after {timeout_s}s")
    except Exception as exc:
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        res = DeepXBuildResult(ok=False, status="exception", onnx_path=str(onnx_p), config_path=str(cfg_p), output_dir=str(out_dir), log_path=str(log_path), command=cmd, message=f"{type(exc).__name__}: {exc}")
    manifest = asdict(res)
    manifest["compiler_cuda_preflight"] = compiler_environment
    manifest["elapsed_s"] = round(time.time() - start, 3)
    manifest["tensor_activation_loader"] = True
    manifest["activation_manifest"] = str(manifest_p)
    manifest["sitecustomize_cv2_npz_loader"] = True
    manifest["sitecustomize_path"] = str(sitecustomize_path)
    (out_dir / "build_manifest_tensor_loader.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return res
