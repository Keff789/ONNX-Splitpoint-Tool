from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import PreparedHandle
from .hailo_utils import get_dfc_manager
from .._types import BackendCaps, BackendRunOut, RunCfg

_RESULT_MARKER = "__SPLITPOINT_HAILO_RESULT__"


def _import_hailo_module() -> Any:
    """Import Hailo Python bindings.

    Hailo packages have used different top-level module names over time.
    Prefer `hailo_platform` when available, and fall back to `hailort`.
    """
    try:
        import hailo_platform as hpf  # type: ignore
        return hpf
    except Exception as exc_platform:
        try:
            import hailort as hpf  # type: ignore
            return hpf
        except Exception as exc_hailort:
            raise RuntimeError(
                "Hailo runtime is unavailable: cannot import 'hailo_platform' or 'hailort'. "
                "Install HailoRT Python bindings in the runtime environment."
            ) from exc_hailort


@dataclass
class _HailoSession:
    hef_path: Path
    quantized_inputs: bool
    quantized_outputs: bool
    _hpf: Any = None
    _vdevice: Any = None
    _network_group: Any = None
    _network_group_params: Any = None
    _pipe: Any = None
    input_names: list[str] = None  # type: ignore[assignment]
    output_names: list[str] = None  # type: ignore[assignment]
    input_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]
    output_shapes: dict[str, tuple[int, ...]] = None  # type: ignore[assignment]

    _activation_lock = threading.RLock()

    def __post_init__(self) -> None:
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        self._open()

    def _format_type(self, *, want_uint8: bool):
        fmt = getattr(self._hpf, "FormatType", None)
        if fmt is None:
            return None
        if want_uint8:
            return getattr(fmt, "UINT8", getattr(fmt, "AUTO", None))
        return getattr(fmt, "FLOAT32", getattr(fmt, "AUTO", None))

    def _open(self) -> None:
        hpf = _import_hailo_module()

        if not self.hef_path.exists() or self.hef_path.stat().st_size <= 0:
            raise RuntimeError(f"Invalid HEF file: {self.hef_path}")

        self._hpf = hpf
        hef = hpf.HEF(str(self.hef_path))
        self._vdevice = hpf.VDevice()
        if hasattr(self._vdevice, "__enter__"):
            self._vdevice.__enter__()

        cfg = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_groups = self._vdevice.configure(hef, cfg)
        if not network_groups:
            raise RuntimeError("No network groups returned by Hailo VDevice.configure")
        self._network_group = network_groups[0]
        self._network_group_params = self._network_group.create_params()

        in_infos = list(hef.get_input_vstream_infos())
        out_infos = list(hef.get_output_vstream_infos())
        self.input_names = [x.name for x in in_infos]
        self.output_names = [x.name for x in out_infos]
        self.input_shapes = {x.name: tuple(x.shape) for x in in_infos}
        self.output_shapes = {x.name: tuple(x.shape) for x in out_infos}

        in_params = hpf.InputVStreamParams.make(
            self._network_group,
            quantized=bool(self.quantized_inputs),
            format_type=self._format_type(want_uint8=bool(self.quantized_inputs)) or hpf.FormatType.AUTO,
        )
        out_params = hpf.OutputVStreamParams.make(
            self._network_group,
            quantized=bool(self.quantized_outputs),
            format_type=self._format_type(want_uint8=bool(self.quantized_outputs)) or hpf.FormatType.AUTO,
        )
        self._pipe = hpf.InferVStreams(self._network_group, in_params, out_params)
        if hasattr(self._pipe, "__enter__"):
            self._pipe.__enter__()

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._pipe is None or self._network_group is None:
            raise RuntimeError("Hailo session is closed")

        infer_inputs: Dict[str, np.ndarray] = {}
        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing required Hailo input '{name}'")
            arr = np.asarray(inputs[name])
            if arr.ndim == len(self.input_shapes.get(name, ())):
                arr = arr[None, ...]
            infer_inputs[name] = arr

        with _HailoSession._activation_lock:
            activation = self._network_group.activate(self._network_group_params)
            entered = False
            try:
                if hasattr(activation, "__enter__"):
                    activation.__enter__()
                    entered = True
                out_raw = self._pipe.infer(infer_inputs)
            finally:
                try:
                    if entered and hasattr(activation, "__exit__"):
                        activation.__exit__(None, None, None)
                    elif hasattr(activation, "release"):
                        activation.release()  # type: ignore[attr-defined]
                    elif hasattr(activation, "close"):
                        activation.close()  # type: ignore[attr-defined]
                except Exception:
                    pass

        out: dict[str, np.ndarray] = {}
        for key, value in out_raw.items():
            arr = np.asarray(value)
            if arr.ndim > 0 and arr.shape[0] == 1:
                try:
                    arr = np.squeeze(arr, axis=0)
                except Exception:
                    pass
            out[str(key)] = arr
        return out

    def close(self) -> None:
        try:
            if self._pipe is not None and hasattr(self._pipe, "__exit__"):
                self._pipe.__exit__(None, None, None)
        finally:
            self._pipe = None
        try:
            if self._vdevice is not None and hasattr(self._vdevice, "__exit__"):
                self._vdevice.__exit__(None, None, None)
        finally:
            self._vdevice = None


@dataclass
class _HailoPrepared:
    hef_path: Path
    session: Optional[_HailoSession]
    input_names: list[str]
    output_names: list[str]
    input_shapes: dict[str, tuple[int, ...]]
    output_shapes: dict[str, tuple[int, ...]]
    quantized_inputs: bool
    quantized_outputs: bool


class HailoBackend:
    name = "hailo"
    capabilities = BackendCaps(
        needs_compiler=True,
        supports_two_stage=True,
        supports_cache_dir=True,
        supports_fp16=False,
    )

    def __init__(self, *, strict: bool = True, **cfg: Any) -> None:
        self.cfg = dict(cfg)
        self.default_compile_backend = str(self.cfg.get("compile_backend", "auto"))
        if strict:
            self._check_available()

    def _check_available(self) -> None:
        backend = str(self.cfg.get("compile_backend", "auto") or "auto").strip().lower()
        if backend in {"wsl"}:
            if shutil.which("wsl.exe") is None:
                raise RuntimeError("Hailo backend compile_backend='wsl' requires wsl.exe in PATH.")
            return

        if backend in {"local", "auto", "venv"}:
            dev_ok = any(Path(p).exists() for p in ("/dev/hailo0", "/dev/hailo1")) or shutil.which("hailortcli") is not None
            if not dev_ok and backend == "local":
                raise RuntimeError("Hailo runtime device not detected (/dev/hailo0). Install HailoRT driver.")
            try:
                __import__("hailo_platform")
            except Exception:
                try:
                    __import__("hailort")
                except Exception:
                    if backend == "local":
                        raise RuntimeError("Cannot import hailo_platform/hailort. Install HailoRT python bindings.")

    def prepare(self, run_cfg: RunCfg, artifacts_dir: Path) -> PreparedHandle:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        options = dict(self.cfg)
        options.update(run_cfg.options or {})

        model_path = Path(run_cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        hw_arch = str(options.get("hw_arch", "hailo8"))
        net_name = str(options.get("net_name") or model_path.stem)
        opt_level = int(options.get("opt_level", 1))
        calib_dir = options.get("calib_dir")
        calib_count = int(options.get("calib_count", 64))
        calib_batch_size = int(options.get("calib_batch_size", 8))
        fixup = bool(options.get("fixup", True))
        force_rebuild = bool(options.get("force_rebuild", False))
        keep_artifacts = bool(options.get("keep_artifacts", False))
        compile_backend = str(options.get("compile_backend", self.default_compile_backend) or "auto").lower().strip()
        timeout_s = int(options.get("timeout_s", 1800))
        wsl_timeout_s = int(options.get("wsl_timeout_s", timeout_s))
        wsl_distro = options.get("wsl_distro")
        wsl_venv_activate = str(options.get("wsl_venv_activate", "auto"))

        hef_path = artifacts_dir / "compiled.hef"
        sha1 = hashlib.sha1(model_path.read_bytes()).hexdigest()
        cache_root = Path.home() / ".onnx_splitpoint_tool" / "hailo" / "hef_cache"
        cache_dir = cache_root / hw_arch / f"sha1_{sha1}" / f"opt{opt_level}_cal{calib_count}_bs{calib_batch_size}_fix{int(fixup)}"
        cached_hef = cache_dir / "compiled.hef"

        if force_rebuild:
            if hef_path.exists():
                hef_path.unlink()

        if not hef_path.exists() and cached_hef.exists() and cached_hef.stat().st_size > 0:
            hef_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_hef, hef_path)

        if not hef_path.exists() or hef_path.stat().st_size <= 0:
            result = self._compile_hef(
                model_path=model_path,
                hef_path=hef_path,
                hw_arch=hw_arch,
                net_name=net_name,
                opt_level=opt_level,
                calib_dir=Path(calib_dir).expanduser().resolve() if calib_dir else None,
                calib_count=calib_count,
                calib_batch_size=calib_batch_size,
                fixup=fixup,
                keep_artifacts=keep_artifacts,
                compile_backend=compile_backend,
                wsl_distro=wsl_distro,
                wsl_venv_activate=wsl_venv_activate,
                timeout_s=timeout_s,
                wsl_timeout_s=wsl_timeout_s,
            )
            if not result.get("ok"):
                raise RuntimeError(result.get("error") or "Hailo compilation failed")

            if not hef_path.exists() or hef_path.stat().st_size <= 0:
                raise RuntimeError(f"Hailo compilation did not create valid HEF: {hef_path}")

            cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(hef_path, cached_hef)

        if not hef_path.exists() or hef_path.stat().st_size <= 0:
            raise RuntimeError(f"Compiled HEF missing or empty: {hef_path}")

        quantized_inputs = bool(options.get("quantized_inputs", False))
        quantized_outputs = bool(options.get("quantized_outputs", False))

        session = _HailoSession(
            hef_path=hef_path,
            quantized_inputs=quantized_inputs,
            quantized_outputs=quantized_outputs,
        )

        prepared = _HailoPrepared(
            hef_path=hef_path,
            session=session,
            input_names=list(session.input_names),
            output_names=list(session.output_names),
            input_shapes=dict(session.input_shapes),
            output_shapes=dict(session.output_shapes),
            quantized_inputs=quantized_inputs,
            quantized_outputs=quantized_outputs,
        )

        return PreparedHandle(
            input_names=list(prepared.input_names),
            output_names=list(prepared.output_names),
            handle=prepared,
        )

    def _compile_hef(
        self,
        *,
        model_path: Path,
        hef_path: Path,
        hw_arch: str,
        net_name: str,
        opt_level: int,
        calib_dir: Optional[Path],
        calib_count: int,
        calib_batch_size: int,
        fixup: bool,
        keep_artifacts: bool,
        compile_backend: str,
        wsl_distro: Optional[str],
        wsl_venv_activate: str,
        timeout_s: int,
        wsl_timeout_s: int,
    ) -> dict[str, Any]:
        mode = compile_backend
        if mode == "auto":
            mode = "local"
            try:
                __import__("hailo_sdk_client")
            except Exception:
                mode = "wsl" if shutil.which("wsl.exe") else "venv"

        if mode == "local":
            return self._compile_local(
                model_path=model_path,
                hef_path=hef_path,
                hw_arch=hw_arch,
                net_name=net_name,
                opt_level=opt_level,
                calib_dir=calib_dir,
                calib_count=calib_count,
                calib_batch_size=calib_batch_size,
                fixup=fixup,
                keep_artifacts=keep_artifacts,
            )
        if mode == "venv":
            return self._compile_subprocess(
                python_bin=sys.executable,
                model_path=model_path,
                hef_path=hef_path,
                hw_arch=hw_arch,
                net_name=net_name,
                opt_level=opt_level,
                calib_dir=calib_dir,
                calib_count=calib_count,
                calib_batch_size=calib_batch_size,
                fixup=fixup,
                keep_artifacts=keep_artifacts,
                timeout_s=timeout_s,
            )
        if mode == "wsl":
            return self._compile_wsl(
                model_path=model_path,
                hef_path=hef_path,
                hw_arch=hw_arch,
                net_name=net_name,
                opt_level=opt_level,
                calib_dir=calib_dir,
                calib_count=calib_count,
                calib_batch_size=calib_batch_size,
                fixup=fixup,
                keep_artifacts=keep_artifacts,
                timeout_s=wsl_timeout_s,
                wsl_distro=wsl_distro,
                wsl_venv_activate=wsl_venv_activate,
            )

        raise RuntimeError(f"Unsupported compile_backend '{compile_backend}'")

    def _compile_local(self, **kwargs: Any) -> dict[str, Any]:
        return _compile_hailo_hef_core(**kwargs)

    def _compile_subprocess(self, *, python_bin: str, timeout_s: int, **kwargs: Any) -> dict[str, Any]:
        payload = base64.b64encode(json.dumps(_serialize_kwargs(kwargs), sort_keys=True).encode("utf-8")).decode("ascii")
        code = (
            "import base64,json,sys;"
            "from pathlib import Path;"
            "from onnx_splitpoint_tool.runners.backends.hailo_backend import _compile_hailo_hef_core;"
            "d=json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'));"
            "d['model_path']=Path(d['model_path']);d['hef_path']=Path(d['hef_path']);"
            "d['calib_dir']=Path(d['calib_dir']) if d.get('calib_dir') else None;"
            "r=_compile_hailo_hef_core(**d);"
            f"print('{_RESULT_MARKER}'+json.dumps(r,sort_keys=True))"
        )
        proc = subprocess.run(
            [python_bin, "-c", code, payload],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
        mix = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = _find_marker_json(mix)
        if parsed is None:
            raise RuntimeError(f"VENV compile failed (rc={proc.returncode}): no structured result marker found")
        return parsed

    def _compile_wsl(
        self,
        *,
        model_path: Path,
        hef_path: Path,
        hw_arch: str,
        net_name: str,
        opt_level: int,
        calib_dir: Optional[Path],
        calib_count: int,
        calib_batch_size: int,
        fixup: bool,
        keep_artifacts: bool,
        timeout_s: int,
        wsl_distro: Optional[str],
        wsl_venv_activate: str,
    ) -> dict[str, Any]:
        wsl = shutil.which("wsl.exe")
        if not wsl:
            raise RuntimeError("compile_backend='wsl' requested but wsl.exe not found")

        resolved = get_dfc_manager().resolve_wsl_runtime(
            hw_arch=hw_arch,
            wsl_distro=wsl_distro,
            wsl_venv_activate=wsl_venv_activate,
        )
        activate = resolved.wsl_venv_activate
        if not activate:
            raise RuntimeError("No WSL venv activation path resolved for selected hw_arch.")

        payload = {
            "model_path": str(model_path),
            "hef_path": str(hef_path),
            "hw_arch": hw_arch,
            "net_name": net_name,
            "opt_level": int(opt_level),
            "calib_dir": str(calib_dir) if calib_dir else None,
            "calib_count": int(calib_count),
            "calib_batch_size": int(calib_batch_size),
            "fixup": bool(fixup),
            "keep_artifacts": bool(keep_artifacts),
        }
        b64 = base64.b64encode(json.dumps(payload, sort_keys=True).encode("utf-8")).decode("ascii")
        code = (
            "import base64,json,sys;"
            "from pathlib import Path;"
            "from onnx_splitpoint_tool.runners.backends.hailo_backend import _compile_hailo_hef_core;"
            "d=json.loads(base64.b64decode(sys.argv[1]).decode('utf-8'));"
            "d['model_path']=Path(d['model_path']);d['hef_path']=Path(d['hef_path']);"
            "d['calib_dir']=Path(d['calib_dir']) if d.get('calib_dir') else None;"
            "r=_compile_hailo_hef_core(**d);"
            f"print('{_RESULT_MARKER}'+json.dumps(r,sort_keys=True))"
        )
        bash_cmd = (
            "set -e; "
            f"source {shlex.quote(activate)}; "
            f"python -c {shlex.quote(code)} {shlex.quote(b64)}"
        )
        cmd = [wsl]
        if resolved.wsl_distro:
            cmd += ["-d", str(resolved.wsl_distro)]
        cmd += ["--", "bash", "-lc", bash_cmd]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=float(timeout_s))
        mix = (proc.stdout or "") + "\n" + (proc.stderr or "")
        parsed = _find_marker_json(mix)
        if parsed is None:
            tail = mix[-4000:] if mix else "<no output>"
            raise RuntimeError(f"WSL compile failed (rc={proc.returncode}): no result marker. Output:\n{tail}")
        return parsed

    def run(self, prepared: PreparedHandle, inputs: dict) -> BackendRunOut:
        prep: _HailoPrepared = prepared.handle
        if prep.session is None:
            raise RuntimeError("Hailo prepared session already cleaned up")

        missing = [name for name in prep.input_names if name not in inputs]
        if missing:
            raise KeyError(f"Missing required inputs for Hailo backend: {missing}")

        t0 = time.perf_counter()
        outputs = prep.session.infer({k: np.asarray(v) for k, v in inputs.items()})
        t1 = time.perf_counter()
        return BackendRunOut(outputs=outputs, metrics={"infer_ms": (t1 - t0) * 1000.0})

    def cleanup(self, prepared: PreparedHandle) -> None:
        prep: _HailoPrepared = prepared.handle
        try:
            if prep.session is not None:
                prep.session.close()
        finally:
            prep.session = None


def _serialize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _find_marker_json(text: str) -> Optional[dict[str, Any]]:
    marker = _RESULT_MARKER
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line.startswith(marker):
            continue
        payload = line[len(marker):].strip()
        try:
            data = json.loads(payload)
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return None


def _compile_hailo_hef_core(
    *,
    model_path: Path,
    hef_path: Path,
    hw_arch: str,
    net_name: str,
    opt_level: int,
    calib_dir: Optional[Path],
    calib_count: int,
    calib_batch_size: int,
    fixup: bool,
    keep_artifacts: bool,
) -> dict[str, Any]:
    del calib_dir, fixup
    try:
        from hailo_sdk_client import ClientRunner  # type: ignore
    except Exception as exc:
        return {
            "ok": False,
            "error": (
                "Cannot import hailo_sdk_client for HEF compilation. "
                "Install Hailo DFC SDK in the selected compile environment. "
                f"Details: {type(exc).__name__}: {exc}"
            ),
        }

    hef_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        runner = ClientRunner(hw_arch=str(hw_arch))
        runner.translate_onnx_model(
            model=str(model_path),
            net_name=str(net_name),
            net_input_shapes=None,
            disable_rt_metadata_extraction=True,
        )
        bs = max(1, min(int(calib_batch_size), int(calib_count)))
        script = (
            f"model_optimization_flavor(optimization_level={int(opt_level)}, batch_size={int(bs)})\n"
            f"model_optimization_config(calibration, batch_size={int(bs)}, calibset_size={int(calib_count)})\n"
        )
        runner.load_model_script(script)

        hn = runner.get_hn_dict() or {}
        layers = hn.get("layers") or {}
        input_names = [name for name, meta in layers.items() if isinstance(meta, dict) and meta.get("type") == "input_layer"]
        if not input_names:
            raise RuntimeError("No input layers found in HN metadata")

        rng = np.random.default_rng(0)
        calib_inputs: dict[str, np.ndarray] = {}
        for name in input_names:
            shape = _hn_shape(layers.get(name) or {})
            if not shape:
                shape = [1]
            calib_inputs[name] = np.ascontiguousarray(rng.random((int(calib_count), *shape), dtype=np.float32))

        runner.optimize(calib_inputs)
        if keep_artifacts:
            try:
                runner.save_har(str(hef_path.parent / "quantized.har"))
            except Exception:
                pass

        hef_bytes = runner.compile()
        hef_path.write_bytes(hef_bytes)
        return {"ok": True, "hef_path": str(hef_path)}
    except Exception as exc:
        return {"ok": False, "error": f"HEF compile failed: {type(exc).__name__}: {exc}"}


def _hn_shape(meta: dict[str, Any]) -> list[int]:
    for key in ("input_shape", "shape", "output_shapes", "input_shapes"):
        value = meta.get(key)
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        if not isinstance(value, list):
            continue
        dims = [int(x) for x in value if isinstance(x, int) and x > 0]
        if dims and dims[0] == 1 and len(dims) > 1:
            dims = dims[1:]
        if dims:
            return dims
    return []
