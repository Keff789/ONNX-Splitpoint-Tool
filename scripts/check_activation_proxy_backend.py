#!/usr/bin/env python3
"""Check or install local ONNX Runtime providers for activation-proxy generation.

Examples:
  python scripts/check_activation_proxy_backend.py --backend cuda_ort
  python scripts/check_activation_proxy_backend.py --backend cuda_ort --install --replace
  python scripts/check_activation_proxy_backend.py --backend tensorrt_ort --install --replace

For CUDA/TensorRT providers, --install now repairs the Python NVIDIA CUDA/cuDNN
runtime wheels even when the provider is listed but cannot create a session.

The script intentionally probes the *current* Python interpreter so it can be
run from the Splitpoint .venv.  CUDA/TensorRT availability still depends on the
local NVIDIA driver/CUDA/cuDNN/TensorRT libraries being installed and visible.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


# --- Splitpoint project-venv bootstrap -------------------------------------------------
# Users often run this helper from a shell whose prompt still says "(.venv)" while
# `python` actually resolves to /usr/bin/python because VIRTUAL_ENV is stale or the
# shell was re-used.  On Debian/Ubuntu with PEP 668 this makes pip installs fail with
# "externally-managed-environment".  To make the CLI robust, re-exec into the local
# project .venv automatically when possible.  The GUI already launches the helper with
# the correct interpreter; this guard is mostly for manual terminal use.
def _inside_venv() -> bool:
    try:
        return bool(sys.prefix and sys.prefix != getattr(sys, "base_prefix", sys.prefix))
    except Exception:
        return False


def _maybe_reexec_project_venv() -> None:
    if os.environ.get("SPLITPOINT_PROXY_BACKEND_NO_REEXEC") == "1":
        return
    if os.environ.get("SPLITPOINT_PROXY_BACKEND_REEXECED") == "1":
        return
    if _inside_venv():
        return
    try:
        script_path = Path(__file__).resolve()
        # Prefer the repository root derived from this script path, then cwd.
        candidates = [script_path.parents[1] / ".venv" / "bin" / "python", Path.cwd() / ".venv" / "bin" / "python"]
        for py in candidates:
            if py.exists() and os.access(py, os.X_OK):
                env = dict(os.environ)
                env["SPLITPOINT_PROXY_BACKEND_REEXECED"] = "1"
                env.pop("VIRTUAL_ENV", None)  # avoid stale prompt/env confusion
                os.execve(str(py), [str(py), str(script_path), *sys.argv[1:]], env)
    except Exception:
        # Fall through and report a clear error later if pip would be system-managed.
        return


_maybe_reexec_project_venv()
# --------------------------------------------------------------------------------------


def _actual_venv() -> str | None:
    # sys.prefix is the interpreter that is actually being probed; VIRTUAL_ENV
    # may be stale when the GUI was launched from another shell/venv.
    try:
        base = getattr(sys, "base_prefix", sys.prefix)
        if sys.prefix and sys.prefix != base:
            return str(sys.prefix)
    except Exception:
        pass
    return None


PROBE_CODE = r'''
import json
preload = {"attempted": False, "ok": None, "error": ""}
try:
    import onnxruntime as ort
    # ORT >= 1.21 can preload CUDA/cuDNN libraries installed as NVIDIA Python
    # packages. This often fixes provider listing-vs-session mismatches on x86
    # workstations where libcublas/libcudnn are not in the system loader path.
    if hasattr(ort, "preload_dlls"):
        try:
            preload["attempted"] = True
            ort.preload_dlls(directory="")
            preload["ok"] = True
        except Exception as _exc:
            preload["ok"] = False
            preload["error"] = f"{type(_exc).__name__}: {_exc}"
    out = {
        "ok": True,
        "onnxruntime_version": getattr(ort, "__version__", None),
        "available_providers": list(ort.get_available_providers() or []),
        "module_file": getattr(ort, "__file__", None),
        "preload_dlls": preload,
    }
except Exception as exc:
    out = {"ok": False, "error": f"onnxruntime import failed: {type(exc).__name__}: {exc}"}
print(json.dumps(out))
'''


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    return int(proc.returncode), proc.stdout or "", proc.stderr or ""


def _probe() -> dict[str, Any]:
    rc, out, err = _run([sys.executable, "-c", PROBE_CODE])
    try:
        data = json.loads((out or "").strip().splitlines()[-1])
    except Exception:
        data = {"ok": False, "error": "probe JSON parse failed", "stdout": out, "stderr": err, "rc": rc}
    data.setdefault("probe_rc", rc)
    if err.strip():
        data.setdefault("probe_stderr", err.strip())
    return data


def _pip_install(packages: list[str], *, replace: bool = False) -> dict[str, Any]:
    logs: list[dict[str, Any]] = []
    if not _inside_venv():
        return {
            "install_rc": 2,
            "install_logs": [{
                "cmd": "<refused>",
                "rc": 2,
                "stdout_tail": "",
                "stderr_tail": (
                    "Refusing to install into a non-venv interpreter: " + sys.executable +
                    ". Run ./start_gui.sh once or use .venv/bin/python scripts/check_activation_proxy_backend.py ..."
                ),
            }],
        }
    base = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    if replace:
        rc, out, err = _run([*base, "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"])
        logs.append({"cmd": sys.executable + " -m pip uninstall -y onnxruntime onnxruntime-gpu", "rc": rc, "stdout_tail": out[-4000:], "stderr_tail": err[-4000:]})
    # Upgrade pip tooling first. Newer pip resolves the NVIDIA CUDA wheels more reliably.
    rc_tools, out_tools, err_tools = _run([*base, "install", "--upgrade", "pip", "setuptools", "wheel"])
    logs.append({"cmd": sys.executable + " -m pip install --upgrade pip setuptools wheel", "rc": rc_tools, "stdout_tail": out_tools[-4000:], "stderr_tail": err_tools[-4000:]})
    rc, out, err = _run([*base, "install", "--upgrade", *packages])
    logs.append({"cmd": sys.executable + " -m pip install --upgrade " + " ".join(packages), "rc": rc, "stdout_tail": out[-4000:], "stderr_tail": err[-4000:]})
    return {"install_rc": rc, "install_logs": logs, "install_packages": packages}


def _required_provider(backend: str) -> str | None:
    if backend == "cuda_ort":
        return "CUDAExecutionProvider"
    if backend == "tensorrt_ort":
        return "TensorrtExecutionProvider"
    return None




def _normalize_backend(value: str) -> str:
    s = str(value or "auto").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "cpu": "ort_cpu",
        "ort": "ort_cpu",
        "onnxruntime": "ort_cpu",
        "ort_cpu": "ort_cpu",
        "cpu_ort": "ort_cpu",
        "cuda": "cuda_ort",
        "gpu": "cuda_ort",
        "ort_cuda": "cuda_ort",
        "cuda_ort": "cuda_ort",
        "onnxruntime_cuda": "cuda_ort",
        "trt": "tensorrt_ort",
        "tensor_rt": "tensorrt_ort",
        "tensorrt": "tensorrt_ort",
        "ort_tensorrt": "tensorrt_ort",
        "tensorrt_ort": "tensorrt_ort",
        "onnxruntime_tensorrt": "tensorrt_ort",
        "remote": "remote_deepx_tensorrt", "remote_trt": "remote_deepx_tensorrt", "remote_tensorrt": "remote_deepx_tensorrt", "remote_deepx": "remote_deepx_tensorrt", "remote_deepx_trt": "remote_deepx_tensorrt", "remote_deepx_tensorrt": "remote_deepx_tensorrt",
        "remote_cuda": "remote_deepx_cuda", "remote_deepx_cuda": "remote_deepx_cuda",
    }
    return aliases.get(s, s if s in {"auto", "ort_cpu", "cuda_ort", "tensorrt_ort", "remote_deepx_tensorrt", "remote_deepx_cuda"} else "auto")





def _preload_nvidia_cuda_libs_for_process() -> dict[str, Any]:
    """Preload CUDA/cuDNN/cuBLAS libraries from pip NVIDIA wheels.

    ORT's preload_dlls(directory="") is helpful but on some Linux installs it
    still leaves libonnxruntime_providers_cuda.so unable to resolve
    libcublasLt.so.12/libcublas.so.12 at dlopen time.  Load the wheel-shipped
    libraries explicitly with RTLD_GLOBAL before constructing ORT sessions.
    """
    import ctypes
    import site
    import sys as _sys
    roots: list[Path] = []
    try:
        roots.extend(Path(x) for x in site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    for x in _sys.path:
        if x and "site-packages" in x:
            roots.append(Path(x))
    unique_roots: list[Path] = []
    for r in roots:
        if r and r.exists() and r not in unique_roots:
            unique_roots.append(r)

    patterns = [
        "**/libcuda.so*",  # normally system-provided; harmless if absent
        "**/libcudart.so*",
        "**/libnvrtc.so*",
        "**/libnvJitLink.so*",
        "**/libcublas.so*",
        "**/libcublasLt.so*",
        "**/libcudnn.so*",
        "**/libcufft.so*",
        "**/libcurand.so*",
        "**/libcusparse.so*",
        "**/libcusolver.so*",
    ]
    found: list[Path] = []
    for root in unique_roots:
        for pat in patterns:
            try:
                for q in root.glob(pat):
                    if q.is_file() and q not in found:
                        found.append(q)
            except Exception:
                pass

    # Load in a dependency-friendly order.  Keep exact paths so dlopen does not
    # depend on LD_LIBRARY_PATH being honoured after process startup.
    order_names = [
        "libcudart", "libnvrtc", "libnvJitLink",
        "libcublas", "libcublasLt", "libcudnn", "libcufft", "libcurand", "libcusparse", "libcusolver",
    ]
    def key(q: Path) -> tuple[int, str]:
        name = q.name
        for i, prefix in enumerate(order_names):
            if name.startswith(prefix):
                return (i, name)
        return (999, name)
    found_sorted = sorted(found, key=key)
    dirs: list[str] = []
    for q in found_sorted:
        d=str(q.parent)
        if d not in dirs:
            dirs.append(d)
    if dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")])

    loaded: list[str] = []
    errors: list[str] = []
    for q in found_sorted:
        # libcuda.so from driver may be a stub in wheels; avoid forcing it.
        if q.name.startswith("libcuda.so"):
            continue
        try:
            ctypes.CDLL(str(q), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            loaded.append(str(q))
        except Exception as exc:
            errors.append(f"{q}: {type(exc).__name__}: {exc}")
    return {
        "attempted": True,
        "found_count": len(found_sorted),
        "loaded_count": len(loaded),
        "loaded": loaded[-20:],
        "dirs": dirs,
        "errors": errors[:20],
        "ok": bool(loaded) or not found_sorted,
    }

def _preload_tensorrt_libs_for_process() -> dict[str, Any]:
    """Load pip-installed TensorRT libraries into the current process.

    onnxruntime-gpu does not bundle TensorRT itself.  NVIDIA's pip TensorRT
    packages place libnvinfer*.so under site-packages.  ORT's TensorRT EP is a
    shared library loaded by dlopen; preloading the TensorRT libraries with
    RTLD_GLOBAL makes their SONAMEs visible to the EP loader.
    """
    import ctypes
    import site
    import sys as _sys
    roots: list[Path] = []
    try:
        for x in site.getsitepackages():
            roots.append(Path(x))
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    for x in _sys.path:
        if x and "site-packages" in x:
            roots.append(Path(x))
    seen_roots: list[Path] = []
    for r in roots:
        if r and r not in seen_roots:
            seen_roots.append(r)
    found: list[Path] = []
    for root in seen_roots:
        try:
            for pat in ("**/libnvinfer.so*", "**/libnvinfer_plugin.so*", "**/libnvonnxparser.so*", "**/libnvinfer_lean.so*", "**/libnvinfer_dispatch.so*"):
                for p in root.glob(pat):
                    if p.is_file() and p not in found:
                        found.append(p)
        except Exception:
            pass
    dirs: list[str] = []
    for p in found:
        d = str(p.parent)
        if d not in dirs:
            dirs.append(d)
    for d in ("/usr/lib/x86_64-linux-gnu", "/usr/local/TensorRT/lib", "/usr/lib/aarch64-linux-gnu"):
        if Path(d).exists() and d not in dirs:
            dirs.append(d)
    if dirs:
        os.environ["LD_LIBRARY_PATH"] = ":".join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")])
    loaded: list[str] = []
    errors: list[str] = []
    for p in sorted(found, key=lambda x: ("plugin" in x.name, "parser" in x.name, x.name)):
        try:
            ctypes.CDLL(str(p), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
            loaded.append(str(p))
        except Exception as exc:
            errors.append(f"{p}: {type(exc).__name__}: {exc}")
    return {"attempted": True, "loaded": loaded, "dirs": dirs, "errors": errors[:10], "ok": bool(loaded) or not errors}

def _session_smoke(provider: str | None) -> dict[str, Any]:
    """Try to instantiate and run a tiny ONNX model with the requested provider.

    Provider listing alone is not enough on some systems: ORT may list the
    TensorRT provider but fail when creating a session if CUDA/cuDNN/TensorRT
    libraries are missing. This smoke test catches that case early. If the
    optional `onnx` Python package is missing we report a skipped smoke, but do
    not fail CPU checks.
    """
    if not provider:
        return {"ok": True, "skipped": True, "reason": "no accelerated provider requested"}
    try:
        import tempfile
        import numpy as np
        import onnx
        _cuda_preload_info = _preload_nvidia_cuda_libs_for_process()
        _trt_preload_info = _preload_tensorrt_libs_for_process()
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(directory="")
                _preload_tensorrt_libs_for_process()
            except Exception:
                pass
        from onnx import TensorProto, helper
    except Exception as exc:
        return {
            "ok": False,
            "skipped": True,
            "reason": f"session smoke requires onnx+numpy+onnxruntime: {type(exc).__name__}: {exc}",
        }
    try:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Relu", ["x"], ["y"])
        graph = helper.make_graph([node], "splitpoint_provider_smoke", [x], [y])
        model = helper.make_model(graph, producer_name="splitpoint-provider-smoke", opset_imports=[helper.make_opsetid("", 17)])
        # Keep IR version conservative for older ORT builds.
        model.ir_version = min(getattr(model, "ir_version", 9), 9)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            path = f.name
        onnx.save(model, path)
        providers = [provider]
        if provider == "TensorrtExecutionProvider":
            providers.extend(["CUDAExecutionProvider", "CPUExecutionProvider"])
        elif provider == "CUDAExecutionProvider":
            providers.append("CPUExecutionProvider")
        sess = ort.InferenceSession(path, providers=providers)
        got = list(sess.get_providers() or [])
        out = sess.run(None, {"x": np.array([[-1.0, 0.0, 2.0, 3.0]], dtype=np.float32)})[0]
        provider_active = provider in got

        conv_smoke = {"ok": None, "skipped": True}
        if provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"} and provider_active:
            try:
                # A Relu-only smoke can pass even when the CUDA/cuDNN path used by
                # real CNNs fails later. Run a tiny Conv as well so CUDNN_FE
                # failures are caught before activation-proxy generation.
                x2 = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 16, 16])
                w2 = helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 3, 3, 3])
                y2 = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 14, 14])
                node2 = helper.make_node("Conv", ["x", "w"], ["y"], pads=[0, 0, 0, 0], strides=[1, 1])
                graph2 = helper.make_graph([node2], "splitpoint_provider_conv_smoke", [x2, w2], [y2])
                model2 = helper.make_model(graph2, producer_name="splitpoint-provider-conv-smoke", opset_imports=[helper.make_opsetid("", 17)])
                model2.ir_version = min(getattr(model2, "ir_version", 9), 9)
                with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f2:
                    path2 = f2.name
                onnx.save(model2, path2)
                sess2 = ort.InferenceSession(path2, providers=providers)
                got2 = list(sess2.get_providers() or [])
                xarr = np.zeros((1, 3, 16, 16), dtype=np.float32)
                warr = np.ones((4, 3, 3, 3), dtype=np.float32)
                yarr = sess2.run(None, {"x": xarr, "w": warr})[0]
                conv_smoke = {"ok": bool(provider in got2 and tuple(yarr.shape) == (1, 4, 14, 14)), "session_providers": got2}
            except Exception as conv_exc:
                conv_smoke = {"ok": False, "error": f"{type(conv_exc).__name__}: {conv_exc}"}

        ok = bool(provider_active and out.shape == (1, 4) and (conv_smoke.get("ok") is not False))
        return {
            "ok": ok,
            "requested_provider": provider,
            "session_providers": got,
            "provider_active": bool(provider_active),
            "output_preview": [float(v) for v in out.reshape(-1).tolist()],
            "conv_smoke": conv_smoke,
            "cuda_preload": _cuda_preload_info,
            "tensorrt_preload": _trt_preload_info,
        }
    except Exception as exc:
        return {
            "ok": False,
            "requested_provider": provider,
            "error": f"{type(exc).__name__}: {exc}",
        }

def _packages_for_backend(backend: str) -> list[str]:
    # Official ONNX Runtime PyPI GPU package is onnxruntime-gpu. Since ORT
    # 1.19 the default CUDA build line is CUDA 12.x on PyPI.  ORT provider
    # listing is not enough: session creation can still fail if CUDA/cuDNN/cuBLAS
    # shared libraries are not installed.  Therefore GPU repair explicitly
    # installs the NVIDIA CUDA 12 runtime wheels as well as ORT.  ORT >= 1.21 can
    # preload these wheels with ort.preload_dlls(directory="").
    if backend in {"cuda_ort", "tensorrt_ort"}:
        env = os.environ.get("ONNX_SPLITPOINT_ORT_GPU_PACKAGE", "").strip()
        base = [env or "onnxruntime-gpu[cuda,cudnn]"]
        explicit = [
            "onnx",
            "numpy",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cuda-nvrtc-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12",
            "nvidia-cufft-cu12",
            "nvidia-curand-cu12",
            "nvidia-cusolver-cu12",
            "nvidia-cusparse-cu12",
            "nvidia-nvjitlink-cu12",
        ]
        if backend == "tensorrt_ort":
            # ONNX Runtime TensorRT EP needs TensorRT itself.  NVIDIA's official
            # pip install supports CUDA-major-specific TensorRT packages.
            explicit.extend(["tensorrt-cu12", "tensorrt-lean-cu12", "tensorrt-dispatch-cu12"])
        extra = [x.strip() for x in os.environ.get("ONNX_SPLITPOINT_ORT_GPU_EXTRA_PACKAGES", "").replace(",", " ").split() if x.strip()]
        out: list[str] = []
        for pkg in [*base, *explicit, *extra]:
            if pkg and pkg not in out:
                out.append(pkg)
        return out
    return [os.environ.get("ONNX_SPLITPOINT_ORT_CPU_PACKAGE", "onnxruntime").strip() or "onnxruntime", "onnx", "numpy"]




def _check_remote_deepx_backend(requested: str) -> dict[str, Any]:
    """Check the centrally configured DeepX remote activation-proxy host.

    Reports the actual remote Python executable, import errors, ORT provider
    list, dx_engine status, and a tiny provider session-smoke. This is more
    useful than the older providers=[]/dx_engine=false output.
    """
    import shlex as _shlex
    out: dict[str, Any] = {"requested_backend": requested, "remote": True, "ok": False}
    try:
        import yaml  # type: ignore
        p = Path.home() / ".onnx_splitpoint_tool" / "hardware_setups.yaml"
        out["hardware_setups_file"] = str(p)
        if not p.exists():
            out["reason"] = "hardware_setups.yaml not found; configure DeepX remote setup in Tool Config"
            return out
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        setup = None
        for item in list(data.get("hardware_setups") or []):
            if not isinstance(item, dict):
                continue
            hay = (str(item.get("accelerator") or "") + " " + str(item.get("id") or "")).lower()
            if "deepx" in hay or "dx_m1" in hay or "dx-m1" in hay:
                setup = item
                h0 = item.get("host") if isinstance(item.get("host"), dict) else {}
                if str(h0.get("address") or h0.get("host") or "").strip():
                    break
        if not setup:
            out["reason"] = "no DeepX hardware setup found"
            return out
        h = setup.get("host") if isinstance(setup.get("host"), dict) else {}
        rt = setup.get("runtime") if isinstance(setup.get("runtime"), dict) else {}
        host = str(h.get("address") or h.get("host") or setup.get("host") or "").strip()
        user = str(h.get("user") or setup.get("user") or "nx")
        port = int(h.get("port") or setup.get("port") or 22)
        activate = str(rt.get("activate") or rt.get("venv") or setup.get("remote_venv") or "source ~/venvs/deepx-runtime/bin/activate")
        out.update({"setup_id": setup.get("id"), "host": host, "user": user, "port": port, "activate": activate})
        if not host:
            out["reason"] = "DeepX hardware setup has no host"
            return out

        want = "TensorrtExecutionProvider" if "tensorrt" in requested else "CUDAExecutionProvider"
        remote_probe = r"""
set +e
ACTIVATE_CMD=__ACTIVATE_CMD__
if [ -n "$ACTIVATE_CMD" ]; then
  eval "$ACTIVATE_CMD"
  ACTIVATE_RC=$?
else
  ACTIVATE_RC=0
fi
export ACTIVATE_RC
export LD_LIBRARY_PATH=/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH:-}
PYBIN="$(command -v python3 || command -v python || true)"
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PYBIN="$VIRTUAL_ENV/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
  PYBIN="$VIRTUAL_ENV/bin/python3"
fi
if [ -z "$PYBIN" ]; then
  printf '%s\n' '{"fatal":"no python found after activation"}'
  exit 0
fi
"$PYBIN" - <<'PYREMOTE'
import json, os, sys, tempfile
res = {"python": sys.executable, "prefix": sys.prefix, "cwd": os.getcwd(), "activate_rc": int(os.environ.get("ACTIVATE_RC", "0") or 0)}
try:
    import numpy as np
    res["numpy"] = getattr(np, "__version__", "")
except Exception as exc:
    res["numpy_error"] = f"{type(exc).__name__}: {exc}"
try:
    import onnxruntime as ort
    try:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
            res["preload_dlls"] = True
    except Exception as exc:
        res["preload_dlls_error"] = f"{type(exc).__name__}: {exc}"
    res["onnxruntime"] = getattr(ort, "__version__", "")
    res["providers"] = list(ort.get_available_providers() or [])
    want = os.environ.get("SPLITPOINT_WANT_PROVIDER", "")
    if want:
        try:
            import onnx
            from onnx import TensorProto, helper
            X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
            Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
            graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "remote_proxy_smoke", [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = min(getattr(model, "ir_version", 7), 7)
            fd, path = tempfile.mkstemp(suffix=".onnx")
            os.close(fd)
            onnx.save(model, path)
            providers = [want, "CUDAExecutionProvider", "CPUExecutionProvider"] if want == "TensorrtExecutionProvider" else [want, "CPUExecutionProvider"]
            sess = ort.InferenceSession(path, providers=providers)
            used = list(sess.get_providers() or [])
            res["session_smoke"] = {"ok": want in used, "requested_provider": want, "session_providers": used}
            try:
                os.remove(path)
            except Exception:
                pass
        except Exception as exc:
            res["session_smoke"] = {"ok": False, "requested_provider": want, "error": f"{type(exc).__name__}: {exc}"}
except Exception as exc:
    res["providers"] = []
    res["ort_error"] = f"{type(exc).__name__}: {exc}"
try:
    from dx_engine import InferenceEngine  # noqa: F401
    res["dx_engine"] = True
except Exception as exc:
    res["dx_engine"] = False
    res["dx_engine_error"] = f"{type(exc).__name__}: {exc}"
try:
    import tensorrt as trt
    res["tensorrt"] = getattr(trt, "__version__", "")
except Exception as exc:
    res["tensorrt_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(res, sort_keys=True))
PYREMOTE
"""
        remote_probe = remote_probe.replace("__ACTIVATE_CMD__", _shlex.quote(activate))
        remote_cmd = "SPLITPOINT_WANT_PROVIDER=" + _shlex.quote(want) + " bash -lc " + _shlex.quote(remote_probe)
        proc = subprocess.run(["ssh", "-p", str(port), f"{user}@{host}", remote_cmd], text=True, capture_output=True, timeout=60)
        out["ssh_rc"] = int(proc.returncode)
        out["ssh_stdout_tail"] = (proc.stdout or "")[-6000:]
        out["ssh_stderr_tail"] = (proc.stderr or "")[-6000:]
        out["required_remote_provider"] = want
        if proc.returncode != 0:
            out["reason"] = "SSH/runtime preflight failed"
            return out
        try:
            js = json.loads((proc.stdout or "").strip().splitlines()[-1])
        except Exception as exc:
            js = {"parse_error": f"{type(exc).__name__}: {exc}", "raw_tail": (proc.stdout or "")[-2000:]}
        out["remote_probe"] = js
        providers = set(js.get("providers") or []) if isinstance(js, dict) else set()
        smoke = js.get("session_smoke") if isinstance(js, dict) else None
        smoke_ok = bool(isinstance(smoke, dict) and smoke.get("ok")) if want in providers else False
        out["ok"] = bool(want in providers and smoke_ok and js.get("dx_engine") is True)
        if not out["ok"]:
            reasons = []
            if want not in providers:
                reasons.append(f"remote provider {want} not listed")
            elif not smoke_ok:
                reasons.append(f"remote provider {want} listed but session smoke failed")
            if js.get("dx_engine") is not True:
                reasons.append("dx_engine import failed on remote")
            for key in ("ort_error", "dx_engine_error", "tensorrt_error", "preload_dlls_error"):
                if js.get(key):
                    reasons.append(f"{key}: {js.get(key)}")
            out["reason"] = "; ".join(reasons) or "remote DeepX proxy check failed"
        return out
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out

def main() -> int:
    ap = argparse.ArgumentParser(description="Check/install activation-proxy ONNX Runtime provider in the current Python environment.")
    ap.add_argument("--backend", default="auto", choices=["auto", "ort_cpu", "cuda_ort", "tensorrt_ort", "remote_deepx_tensorrt", "remote_deepx_cuda", "remote", "remote_tensorrt", "remote_cuda", "cpu", "cuda", "tensorrt", "ort_tensorrt", "ort_cuda", "cpu_ort"])
    ap.add_argument("--install", action="store_true", help="Install missing ORT package into the current interpreter environment.")
    ap.add_argument("--replace", action="store_true", help="Uninstall onnxruntime/onnxruntime-gpu before installing the requested package.")
    ap.add_argument("--json", action="store_true", help="Print JSON only. Default output is already JSON; kept for compatibility.")
    ap.add_argument("--no-session-smoke", action="store_true", help="Only check provider listing, do not instantiate a tiny test session.")
    args = ap.parse_args()

    requested_raw = str(args.backend or "auto")
    requested = _normalize_backend(requested_raw)
    virtual_env_env = os.environ.get("VIRTUAL_ENV")
    actual_venv = _actual_venv()
    result: dict[str, Any] = {
        "requested_backend": requested,
        "requested_backend_raw": requested_raw,
        "python": sys.executable,
        "venv": actual_venv or virtual_env_env,
        "actual_venv": actual_venv,
        "env_virtual_env": virtual_env_env,
        "cwd": str(Path.cwd()),
        "inside_venv": _inside_venv(),
        "reexeced_project_venv": os.environ.get("SPLITPOINT_PROXY_BACKEND_REEXECED") == "1",
    }
    if actual_venv and virtual_env_env and Path(actual_venv).resolve() != Path(virtual_env_env).resolve():
        result["venv_warning"] = (
            "VIRTUAL_ENV points to a different environment than the Python interpreter. "
            "The check uses sys.executable/sys.prefix, not the stale VIRTUAL_ENV value."
        )

    if requested in {"remote_deepx_tensorrt", "remote_deepx_cuda"}:
        result.update(_check_remote_deepx_backend(requested))
        result.setdefault("hint", "Remote activation proxy uses the DeepX hardware setup from Tool Config. The real cache generation copies ONNX/images to the host, runs ORT there, and copies NPZ tensors back.")
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    probe = _probe()
    result["initial_probe"] = probe

    need_install = not bool(probe.get("ok"))
    provider = _required_provider(requested)
    providers = list(probe.get("available_providers") or []) if probe.get("ok") else []
    if provider and provider not in providers:
        result["provider_missing_before_install"] = provider
        if args.install:
            need_install = True

    if args.install and requested == "ort_cpu":
        result["install_note"] = (
            "Selected backend is ort_cpu, so this installs/checks CPU ONNX Runtime only. "
            "Choose cuda_ort or tensorrt_ort in Tool Config before pressing Install/repair ORT backend "
            "if you want to install/check GPU providers."
        )

    # In earlier versions we only installed when the provider was missing from
    # get_available_providers().  That was not enough: ORT can list CUDA/TensorRT
    # but fail when creating a session because libcublas/libcudnn/etc. are
    # missing.  If --install was requested, always repair the requested backend.
    if args.install:
        packages = _packages_for_backend(requested)
        result["install_packages"] = packages
        result.update(_pip_install(packages, replace=bool(args.replace)))
        probe = _probe()
        result["final_probe"] = probe
    else:
        result["final_probe"] = probe

    final = result.get("final_probe") or {}
    final_providers = list(final.get("available_providers") or []) if final.get("ok") else []
    ok = bool(final.get("ok"))
    reason = ""
    if ok and provider and provider not in final_providers:
        ok = False
        reason = f"{provider} not available"
    elif not ok:
        reason = str(final.get("error") or probe.get("error") or "onnxruntime unavailable")

    smoke = None
    if ok and provider and not args.no_session_smoke:
        smoke = _session_smoke(provider)
        result["session_smoke"] = smoke
        if not smoke.get("ok"):
            ok = False
            reason = str(smoke.get("error") or smoke.get("reason") or f"{provider} session smoke failed")
    elif provider and args.no_session_smoke:
        result["session_smoke"] = {"ok": None, "skipped": True, "reason": "--no-session-smoke"}

    result.update({
        "ok": ok,
        "required_provider": provider,
        "available_providers": final_providers,
        "reason": reason,
        "hint": (
            "ort_cpu only checks CPUExecutionProvider. For the RTX/CUDA proxy, choose cuda_ort or run: "
            ".venv/bin/python scripts/check_activation_proxy_backend.py --backend cuda_ort --install --replace. "
            "For TensorRT proxy use --backend tensorrt_ort. The helper auto-reexecs into the project .venv when run with system Python. "
            "If providers are listed but session_smoke fails, install/repair with --install --replace; this installs onnxruntime-gpu[cuda,cudnn] plus explicit NVIDIA CUDA/cuDNN/cuBLAS/cuFFT runtime wheels and preloads them. TensorRT EP also requires TensorRT libraries (libnvinfer/libnvinfer_plugin); --install for tensorrt_ort now installs NVIDIA tensorrt-cu12 packages and preloads their libs when possible."
        ),
    })
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
