#!/usr/bin/env python3
"""Experimental Hailo10H -> Native TensorRT E2E FIFO runner.

This is the first real E2E producer path for Hailo10H.  It is intentionally
Python-based because Hailo10H uses the HailoRT InferModel/run_async style API;
the Hailo8 production fastpath remains the C++ VStreams->TensorRT runner.

The script executes:
  producer thread: Hailo10H Part1 HEF -> FIFO
  consumer thread: FIFO -> native TensorRT Part2 engine

It writes a native_pipeline JSON report with the same key timing fields as the
Hailo8 native FIFO runner.  v59eb also writes Hailo8-compatible output and
boundary/input dump manifests when requested, so the normal Full-ONNX
self-reference validation can judge Hailo10H rows.
"""
from __future__ import annotations

import argparse, ctypes, ctypes.util, hashlib, json, os, queue, re, shutil, subprocess, sys, threading, time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.runners._types import RunCfg
from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend
from onnx_splitpoint_tool.native_command_contract import (
    hailo10_preprocess_binding,
    load_split_energy_workload_binding,
    seal_native_command_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    DetectionCompletionRuntime,
    build_detection_completion_execution_contract,
    persist_detection_completion_execution_artifacts,
    verify_detection_completion_execution_attestation,
    verify_detection_completion_execution_contract,
)
from onnx_splitpoint_tool.native_split_quality import (
    bind_quality_to_native_split,
    canonical_native_split_backend,
    canonical_json_sha256,
    native_split_quality_selection_duplicates,
    validate_native_split_quality_binding,
)
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)

_STRICT_SPLIT_BOUNDARY: dict[str, Any] | None = None
_STRICT_SPLIT_BOUNDARY_EVIDENCE: dict[str, Any] = {}


def _case_id(x: str) -> str:
    s = str(x)
    return s if s.startswith('b') else f'b{int(s):03d}'

def _load_json(p: Path) -> Any:
    try:
        return json.loads(
            p.read_text(encoding='utf-8'),
            object_pairs_hook=_no_duplicate_json_keys,
        )
    except Exception: return None

def _no_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate_json_key:{key}')
        value[key] = item
    return value

def _load_split_quality_binding(ns: argparse.Namespace, *, case: str, task: str) -> dict[str, Any] | None:
    if not str(ns.native_split_quality_binding or '').strip():
        return None
    raw = _load_json(Path(ns.native_split_quality_binding).expanduser().resolve())
    binding,status = validate_native_split_quality_binding(
        raw,
        expected_identity={
            'backend':'hailo10h_to_trt', 'model':str(ns.model_id or ''),
            'case':case, 'setup_id':str(ns.setup_id or ''), 'task':task,
        },
        verification_mode='local',
    )
    if binding is None:
        raise RuntimeError(f'native_split_quality_local_validation_failed:{status}')
    if str(binding.get('eval_run_id') or '') != str(ns.eval_run_id or ''):
        raise RuntimeError('native_split_quality_eval_run_id_mismatch')
    if canonical_native_split_backend(
        binding.get('source_run_id'), ns.setup_id,
    ) != canonical_native_split_backend(ns.source_run_id, ns.setup_id):
        raise RuntimeError('native_split_quality_source_run_id_mismatch')
    try:
        native_split_quality_selection_duplicates(binding)
    except ValueError as exc:
        raise RuntimeError(
            f'native_split_quality_central_selection_invalid:{exc}'
        ) from exc
    return binding

def _binding_path(binding: Mapping[str, Any], name: str) -> Path:
    row=(binding.get('artifacts') or {}).get(name)
    if not isinstance(row,Mapping):
        raise RuntimeError(f'native_split_quality_binding_artifact_missing:{name}')
    return Path(str(row.get('path') or '')).resolve()

def _seal_split_consumer_attestation(binding: Mapping[str, Any], command: Mapping[str, Any], ns: argparse.Namespace, *, case: str, task: str) -> dict[str, Any]:
    artifacts=command.get('artifacts') if isinstance(command.get('artifacts'),Mapping) else {}
    payload={
        'schema':'onnx-splitpoint/native-split-quality-consumer-attestation',
        'schema_version':1,
        'status':'local_files_rehashed_and_exact_command_join_verified',
        'binding_sha256':str(binding.get('binding_sha256') or ''),
        'command_contract_sha256':str(command.get('contract_sha256') or ''),
        'eval_run_id':str(binding.get('eval_run_id') or ''),
        'source_run_id':canonical_native_split_backend(
            binding.get('source_run_id'), ns.setup_id,
        ),
        'backend':'hailo10h_to_trt', 'model_id':str(ns.model_id or ''),
        'case_id':case, 'setup_id':str(ns.setup_id or ''), 'task':task,
        'precision':str(ns.precision),
        'local_artifact_verification_sha256':canonical_json_sha256(dict(binding.get('local_artifact_verification') or {})),
        'semantic_output_manifest_sha256':str((artifacts.get('semantic_output_manifest') or {}).get('sha256') or ''),
        'semantic_boundary_manifest_sha256':str((artifacts.get('semantic_boundary_manifest') or {}).get('sha256') or ''),
        **native_split_quality_selection_duplicates(binding),
    }
    payload['attestation_sha256']=canonical_json_sha256(payload)
    return payload

def _file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ''
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def _assert_expected_sha256(label: str, path: str | Path, expected: str) -> str:
    """Fail closed before hardware access when replay bytes differ."""
    actual = _file_sha256(path)
    wanted = str(expected or '').strip().lower()
    if wanted and actual != wanted:
        raise RuntimeError(
            f'replay artifact SHA-256 mismatch for {label}: '
            f'expected={wanted} actual={actual or "<missing>"} path={path}'
        )
    return actual

def _seal_manifest_payload_files(manifest_path: Any) -> None:
    path=Path(str(manifest_path or '')).expanduser()
    if not path.is_file(): return
    payload=_load_json(path)
    if not isinstance(payload,dict): raise RuntimeError(f'native_semantic_manifest_invalid:{path}')
    sealed=[]
    def _one(value: Any, role: str) -> dict[str,Any]:
        target=Path(str(value or ''))
        if not target.is_absolute(): target=path.parent/target
        target=target.resolve()
        if not target.is_file() or target.stat().st_size <= 0: raise RuntimeError(f'native_semantic_payload_missing:{role}:{target}')
        row={'role':role,'path':str(target),'sha256':_file_sha256(target),'size_bytes':int(target.stat().st_size)}; sealed.append(row); return row
    for name in ('outputs','tensors'):
        rows=payload.get(name)
        if isinstance(rows,list):
            for index,row in enumerate(rows):
                if isinstance(row,dict) and (row.get('file') or row.get('path')):
                    identity=_one(row.get('file') or row.get('path'),f'{name}[{index}]')
                    row['sha256']=identity['sha256']; row['size_bytes']=identity['size_bytes']
    for key in ('file','input_dump','selected_input_dump'):
        if payload.get(key):
            identity=_one(payload[key],key); payload[f'{key}_sha256']=identity['sha256']; payload[f'{key}_size_bytes']=identity['size_bytes']
    if not sealed: raise RuntimeError(f'native_semantic_manifest_has_no_payload_files:{path}')
    payload['payload_artifacts']=sealed; payload['payload_artifacts_sha256']=canonical_json_sha256(sealed)
    path.write_text(json.dumps(payload,indent=2),encoding='utf-8')

def _find_hef(case_dir: Path) -> Path | None:
    pats = [
        'hailo/hailo10h/part1/compiled.hef',
        'hailo/hailo10/part1/compiled.hef',
        'hailo/hailo10n/part1/compiled.hef',
        'hailo/hailo15h/part1/compiled.hef',
    ]
    for r in pats:
        p = case_dir / r
        if p.is_file(): return p
    xs = sorted(case_dir.glob('hailo/hailo10*/part1/*.hef')) + sorted(case_dir.glob('hailo/hailo15*/part1/*.hef'))
    return xs[0] if xs else None

def _find_part1_onnx(case_dir: Path, case: str) -> Path | None:
    xs = sorted(case_dir.glob(f'*part1*{case[1:]}*.onnx')) or sorted(case_dir.glob('*part1*.onnx'))
    return xs[0] if xs else None

def _engine_path(bs: Path, case: str, precision: str) -> Path:
    if precision == 'uint8_cast_fp16':
        name = 'part2_uint8_cast_fp16.engine'
    elif precision == 'uint8_dequant_fp16':
        name = 'part2_uint8_dequant_fp16.engine'
    elif precision == 'fp16':
        name = 'part2_fp16.engine'
    else:
        name = f'part2_{precision}.engine'
    return bs/'native_trt'/case/'part2'/precision/name


def _engine_boundary_contract(engine: Path) -> dict[str, Any]:
    """Bind the exact native TRT metadata/bridge used by this engine."""
    metadata_path = engine.parent / 'native_trt_meta.json'
    metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    bridge = metadata.get('uint8_cast_bridge')
    bridge = bridge if isinstance(bridge, Mapping) else {}
    layout = bridge.get('boundary_layout')
    layout = layout if isinstance(layout, Mapping) else {}
    requested = str(layout.get('requested') or '').strip()
    effective = str(layout.get('effective') or requested or '').strip()
    return {
        'metadata_path': str(metadata_path.resolve()) if metadata_path.is_file() else '',
        'metadata_sha256': _file_sha256(metadata_path),
        'boundary_layout_requested': requested,
        'boundary_layout_effective': effective,
        'dequant_scale': bridge.get('scale'),
        'dequant_zero_point': bridge.get('zero_point'),
        'bridge_schema': str(bridge.get('schema') or ''),
    }

def _build_engine(bs: Path, case: str, precision: str, engine_build_python: str = 'auto', *, boundary_layout: str = 'as_input', dequant_scale: float = 0.0, dequant_zero_point: float = 0.0) -> list[dict[str, Any]]:
    """Build the native TensorRT Part2 engine using a Python that can import ONNX.

    Hailo10 runs normally with PYTHONNOUSERSITE=1 so that numpy 2.x from
    ~/.local does not shadow the venv.  That is correct for runtime, but it can
    hide the `onnx` package used only for creating the uint8_cast bridge ONNX.
    The builder is therefore allowed to run with a different Python and with
    user-site re-enabled.
    """
    base = [str(ROOT/'scripts'/'native_trt_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', case, '--variants', 'part2', '--precision', precision, '--run-smoke', '--iterations', '100', '--workspace-mb', '4096', '--workspace-mode', 'auto']
    if precision in {'uint8_dequant_fp16', 'float32_layout_fp16'} and str(boundary_layout or 'as_input') != 'as_input':
        base += ['--boundary-layout', str(boundary_layout)]
    if precision == 'uint8_dequant_fp16':
        if float(dequant_scale or 0.0) > 0.0:
            base += ['--dequant-scale', str(float(dequant_scale))]
        base += ['--dequant-zero-point', str(float(dequant_zero_point or 0.0))]
    if str(engine_build_python or '').strip() and str(engine_build_python).strip() != 'auto':
        candidates = [str(engine_build_python).strip()]
    else:
        candidates = [sys.executable, '/usr/bin/python3', '/usr/bin/python', 'python3', 'python']
    seen = set(); out=[]
    for py in candidates:
        if not py or py in seen:
            continue
        seen.add(py)
        cmd = [py] + base
        env = os.environ.copy()
        # Re-enable user site just for the ONNX bridge builder.
        env.pop('PYTHONNOUSERSITE', None)
        t0=time.time(); pr=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        rec={'name':'build_native_trt_part2','cmd':cmd,'python':py,'rc':pr.returncode,'elapsed_s':time.time()-t0,'stdout_tail':pr.stdout[-4000:],'stderr_tail':pr.stderr[-4000:]}
        out.append(rec)
        if pr.returncode == 0:
            break
    return out

class _CudaCompat:
    class cudaMemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2
    def __init__(self):
        lib = ctypes.util.find_library('cudart') or 'libcudart.so'
        self.lib = ctypes.CDLL(lib)
        self.lib.cudaStreamCreate.argtypes=[ctypes.POINTER(ctypes.c_void_p)]; self.lib.cudaStreamCreate.restype=ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes=[ctypes.c_void_p]; self.lib.cudaStreamDestroy.restype=ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes=[ctypes.c_void_p]; self.lib.cudaStreamSynchronize.restype=ctypes.c_int
        self.lib.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]; self.lib.cudaMalloc.restype=ctypes.c_int
        self.lib.cudaFree.argtypes=[ctypes.c_void_p]; self.lib.cudaFree.restype=ctypes.c_int
        self.lib.cudaMallocHost.argtypes=[ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]; self.lib.cudaMallocHost.restype=ctypes.c_int
        self.lib.cudaFreeHost.argtypes=[ctypes.c_void_p]; self.lib.cudaFreeHost.restype=ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes=[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]; self.lib.cudaMemcpyAsync.restype=ctypes.c_int
    def cudaStreamCreate(self):
        s=ctypes.c_void_p(); rc=int(self.lib.cudaStreamCreate(ctypes.byref(s))); return rc, int(s.value or 0)
    def cudaStreamDestroy(self,s): return int(self.lib.cudaStreamDestroy(ctypes.c_void_p(int(s))))
    def cudaStreamSynchronize(self,s): return int(self.lib.cudaStreamSynchronize(ctypes.c_void_p(int(s))))
    def cudaMalloc(self,n):
        p=ctypes.c_void_p(); rc=int(self.lib.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(int(n)))); return rc, int(p.value or 0)
    def cudaFree(self,p): return int(self.lib.cudaFree(ctypes.c_void_p(int(p))))
    def cudaMallocHost(self,n):
        p=ctypes.c_void_p(); rc=int(self.lib.cudaMallocHost(ctypes.byref(p), ctypes.c_size_t(int(n)))); return rc, int(p.value or 0)
    def cudaFreeHost(self,p): return int(self.lib.cudaFreeHost(ctypes.c_void_p(int(p))))
    def cudaMemcpyAsync(self,dst,src,n,kind,stream): return int(self.lib.cudaMemcpyAsync(ctypes.c_void_p(int(dst)), ctypes.c_void_p(int(src)), ctypes.c_size_t(int(n)), int(kind), ctypes.c_void_p(int(stream))))

def _pointer_value(ptr: Any) -> int:
    """Return a stable integer address for cuda-python and ctypes pointers."""
    if isinstance(ptr, (int, np.integer)):
        return int(ptr)
    value = getattr(ptr, 'value', None)
    if value is not None:
        return int(value)
    return int(ptr)

class NativeTRT:
    """Shared native TensorRT Part2 consumer for Hailo10H and DeepX.

    All bindings use page-locked host memory, matching the Hailo8 C++ path.
    ``run`` returns stable views of those output buffers instead of adding a
    second pageable NumPy copy after the synchronized D2H transfer.  A caller
    that needs to retain an output across the next invocation must copy it
    explicitly; the measured FIFO consumers do not retain Part2 outputs.
    """
    def __init__(self, engine_path: Path):
        import tensorrt as trt
        self.trt=trt
        try:
            from cuda import cudart  # type: ignore
            self.cudart=cudart
        except Exception:
            self.cudart=_CudaCompat()
        logger=trt.Logger(trt.Logger.ERROR)
        runtime=trt.Runtime(logger)
        engine=runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if engine is None: raise RuntimeError(f'failed to deserialize engine {engine_path}')
        ctx=engine.create_execution_context()
        if ctx is None: raise RuntimeError('failed to create TRT execution context')
        err, stream = self.cudart.cudaStreamCreate()
        if int(err)!=0: raise RuntimeError(f'cudaStreamCreate failed: {err}')
        self.runtime=runtime; self.engine=engine; self.ctx=ctx; self.stream=stream
        self.inputs=[]; self.outputs=[]; self.shapes={}; self.dtypes={}; self.dev={}
        self.host_in={}; self.host_out={}; self.host_ptr={}; self._host_backing={}
        self.host_memory_policy='cuda_pinned_all_bindings'
        self.output_materialization_policy='synchronized_d2h_pinned_view_no_post_copy'
        def np_dtype(dt):
            try: return np.dtype(trt.nptype(dt))
            except Exception:
                s=str(dt).lower()
                if 'uint8' in s: return np.dtype(np.uint8)
                if 'int8' in s: return np.dtype(np.int8)
                if 'half' in s or 'float16' in s: return np.dtype(np.float16)
                return np.dtype(np.float32)
        if hasattr(engine, 'num_io_tensors'):
            for i in range(int(engine.num_io_tensors)):
                n=str(engine.get_tensor_name(i)); is_in = engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
                sh=tuple(int(x) if int(x)>0 else 1 for x in engine.get_tensor_shape(n)); dt=np_dtype(engine.get_tensor_dtype(n))
                self._reg(n, sh, dt, is_in)
        else:
            for i in range(int(engine.num_bindings)):
                n=str(engine.get_binding_name(i)); is_in=bool(engine.binding_is_input(i)); sh=tuple(int(x) if int(x)>0 else 1 for x in engine.get_binding_shape(i)); dt=np_dtype(engine.get_binding_dtype(i)); self._reg(n, sh, dt, is_in)
    def _reg(self,n,sh,dt,is_in):
        nb=int(np.prod(sh))*np.dtype(dt).itemsize; err, ptr=self.cudart.cudaMalloc(nb)
        if int(err)!=0: raise RuntimeError(f'cudaMalloc failed {n}: {err}')
        self.dev[n]=_pointer_value(ptr); self.shapes[n]=tuple(sh); self.dtypes[n]=np.dtype(dt)
        try:
            err, host_ptr = self.cudart.cudaMallocHost(nb)
        except (AttributeError, TypeError) as exc:
            self.cudart.cudaFree(self.dev[n])
            self.dev.pop(n, None)
            raise RuntimeError(
                'native TensorRT Part2 requires cudaMallocHost so every backend '
                'uses the same pinned-host transfer policy'
            ) from exc
        if int(err)!=0:
            self.cudart.cudaFree(self.dev[n])
            self.dev.pop(n, None)
            raise RuntimeError(f'cudaMallocHost failed {n}: {err}')
        host_addr=_pointer_value(host_ptr)
        backing=(ctypes.c_ubyte * nb).from_address(host_addr)
        host_arr=np.ctypeslib.as_array(backing).view(np.dtype(dt)).reshape(tuple(sh))
        self.host_ptr[n]=host_addr; self._host_backing[n]=backing
        (self.host_in if is_in else self.host_out)[n]=host_arr
        try: self.ctx.set_tensor_address(n, self.dev[n])
        except Exception: pass
        (self.inputs if is_in else self.outputs).append(n)
    def prepare_inputs(self, feeds: dict[str,np.ndarray]) -> None:
        """Materialize the boundary once into the pinned TRT input buffer."""
        for n in self.inputs:
            # exact first, else if only one input use first feed
            arr = feeds.get(n)
            if arr is None and len(self.inputs)==1 and feeds: arr=next(iter(feeds.values()))
            if arr is None: raise KeyError(f'missing TRT input {n}, feeds={list(feeds)}')
            arr=np.asarray(arr)
            if tuple(arr.shape)!=self.shapes[n]: arr=arr.reshape(self.shapes[n])
            if arr.dtype!=self.dtypes[n]: arr=arr.astype(self.dtypes[n], copy=False)
            arr=np.ascontiguousarray(arr)
            host=self.host_in[n]
            np.copyto(host, arr, casting='no')
    def run_prepared(self) -> dict[str,np.ndarray]:
        """Execute H2D, TensorRT, D2H and synchronize."""
        for n in self.inputs:
            host=self.host_in[n]
            rc=self.cudart.cudaMemcpyAsync(int(self.dev[n]), int(host.ctypes.data), int(host.nbytes), self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
            if int(rc)!=0: raise RuntimeError(f'H2D failed {n}: {rc}')
        if not self.ctx.execute_async_v3(self.stream): raise RuntimeError('execute_async_v3 returned False')
        for n in self.outputs:
            out=self.host_out[n]
            rc=self.cudart.cudaMemcpyAsync(int(out.ctypes.data), int(self.dev[n]), int(out.nbytes), self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
            if int(rc)!=0: raise RuntimeError(f'D2H failed {n}: {rc}')
        rc=self.cudart.cudaStreamSynchronize(self.stream)
        if int(rc)!=0: raise RuntimeError(f'cudaStreamSynchronize failed {rc}')
        return {n:self.host_out[n] for n in self.outputs}
    def run(self, feeds: dict[str,np.ndarray]) -> dict[str,np.ndarray]:
        self.prepare_inputs(feeds)
        return self.run_prepared()
    def close(self):
        for p in list(self.dev.values()):
            try: self.cudart.cudaFree(p)
            except Exception: pass
        for p in list(self.host_ptr.values()):
            try: self.cudart.cudaFreeHost(p)
            except Exception: pass
        self.dev.clear(); self.host_in.clear(); self.host_out.clear(); self._host_backing.clear(); self.host_ptr.clear()
        try: self.cudart.cudaStreamDestroy(self.stream)
        except Exception: pass
        self.stream=0


def _image_to_shape(image_path: str, shape: tuple[int, ...], quantized: bool, *, preprocess_mode: str = 'resize', letterbox_pad_value: int = 114, task: str = '') -> np.ndarray:
    if Image is None:
        raise RuntimeError('PIL/Pillow is required for --image input preprocessing')
    im = Image.open(image_path).convert('RGB')
    sh = tuple(int(x) for x in shape)
    # Hailo runtime commonly exposes HWC or NHWC input for HEFs, but accept NCHW too.
    if len(sh) == 3:
        h,w,c = sh if sh[-1] in (1,3,4) else (sh[1], sh[2], sh[0])
        if preprocess_mode == 'letterbox':
            scale=min(float(w)/float(im.width), float(h)/float(im.height)); nw=max(1,int(round(im.width*scale))); nh=max(1,int(round(im.height*scale)))
            resized=im.resize((nw,nh)); canvas=Image.new('RGB',(int(w),int(h)),(int(letterbox_pad_value),)*3); canvas.paste(resized,((int(w)-nw)//2,(int(h)-nh)//2)); arr=np.asarray(canvas)
        else:
            arr = np.asarray(im.resize((int(w), int(h))))
        if c == 1:
            arr = arr[..., :1]
        if sh[-1] in (1,3,4):
            out = arr
        else:
            out = np.transpose(arr, (2,0,1))
    elif len(sh) == 4 and sh[0] == 1:
        inner = sh[1:]
        if inner[-1] in (1,3,4):
            h,w,c = inner
            if preprocess_mode == 'letterbox':
                scale=min(float(w)/float(im.width), float(h)/float(im.height)); nw=max(1,int(round(im.width*scale))); nh=max(1,int(round(im.height*scale)))
                resized=im.resize((nw,nh)); canvas=Image.new('RGB',(int(w),int(h)),(int(letterbox_pad_value),)*3); canvas.paste(resized,((int(w)-nw)//2,(int(h)-nh)//2)); arr=np.asarray(canvas)
            else:
                arr = np.asarray(im.resize((int(w), int(h))))
            if c == 1: arr = arr[..., :1]
            out = arr[None, ...]
        elif inner[0] in (1,3,4):
            c,h,w = inner
            if preprocess_mode == 'letterbox':
                scale=min(float(w)/float(im.width), float(h)/float(im.height)); nw=max(1,int(round(im.width*scale))); nh=max(1,int(round(im.height*scale)))
                resized=im.resize((nw,nh)); canvas=Image.new('RGB',(int(w),int(h)),(int(letterbox_pad_value),)*3); canvas.paste(resized,((int(w)-nw)//2,(int(h)-nh)//2)); arr=np.asarray(canvas)
            else:
                arr = np.asarray(im.resize((int(w), int(h))))
            if c == 1: arr = arr[..., :1]
            out = np.transpose(arr, (2,0,1))[None, ...]
        else:
            raise RuntimeError(f'unsupported image input shape {sh}')
    else:
        raise RuntimeError(f'unsupported image input shape {sh}')
    if quantized:
        return np.ascontiguousarray(out.astype(np.uint8, copy=False))
    floating = out.astype(np.float32) / 255.0
    if str(task or '').strip().lower() == 'classification':
        mean=np.asarray([0.485,0.456,0.406],dtype=np.float32)
        std=np.asarray([0.229,0.224,0.225],dtype=np.float32)
        if floating.shape[-1] in (3,4):
            floating[..., :3]=(floating[..., :3]-mean)/std
        elif len(floating.shape) >= 3 and floating.shape[-3] in (3,4):
            reshape=(1,)*(floating.ndim-3)+(3,1,1)
            floating[..., :3, :, :]=(floating[..., :3, :, :]-mean.reshape(reshape))/std.reshape(reshape)
    return np.ascontiguousarray(floating)

def _make_input(prepared, quantized: bool, seed: int=0, image: str | None = None, *, preprocess_mode: str = 'resize', letterbox_pad_value: int = 114, task: str = '') -> dict[str,np.ndarray]:
    prep=prepared.handle
    shapes=dict(prep.runtime_input_shapes or prep.input_shapes)
    rng=np.random.default_rng(seed)
    out={}
    for i,n in enumerate(prepared.input_names):
        sh=tuple(int(x) for x in shapes.get(n, (1,)))
        if image:
            source=_image_to_shape(image, sh, False, preprocess_mode=preprocess_mode, letterbox_pad_value=letterbox_pad_value, task=task)
        elif quantized:
            source=np.ascontiguousarray(rng.random(sh).astype(np.float32))
        else:
            source=np.ascontiguousarray(rng.random(sh).astype(np.float32))
        if quantized:
            quantizer=getattr(prep.session, 'quantize_input', None)
            if not callable(quantizer):
                raise RuntimeError('Hailo-10 native UINT8 input requires exact HEF QuantInfo quantization')
            out[n]=np.ascontiguousarray(quantizer(n, source), dtype=np.uint8)
        else:
            out[n]=np.ascontiguousarray(source, dtype=np.float32)
    return out

def _selected_hailo_output_format(prepared, quantized_outputs: bool) -> str:
    """Report the actual HEF-selected host output type; require UINT8 here."""
    session=getattr(getattr(prepared, 'handle', None), 'session', None)
    describe=getattr(session, 'describe_io', None)
    if callable(describe):
        value=str(dict(describe() or {}).get('runtime_output_format') or '').strip().lower()
        if value:
            if quantized_outputs and value != 'uint8':
                raise RuntimeError(f'Hailo-10 native output format is not the required UINT8 HEF stream: {value}')
            return value
    raise RuntimeError('Hailo-10 runtime did not report its selected HEF output format')


def _exact_hailo10_output_quantization(prepared) -> dict[str, Any]:
    """Return the single Part1-boundary QuantInfo read from the loaded HEF."""
    session=getattr(getattr(prepared, 'handle', None), 'session', None)
    describe=getattr(session, 'describe_io', None)
    if not callable(describe):
        raise RuntimeError('Hailo-10 exact HEF output QuantInfo is unavailable')
    io=dict(describe() or {})
    formats=dict(io.get('runtime_output_formats') or {})
    quantization=dict(io.get('runtime_output_quantization') or {})
    if not formats or set(str(value).strip().upper() for value in formats.values()) != {'UINT8'}:
        raise RuntimeError('Hailo-10 Part1 boundary must expose only native UINT8 HEF output streams')
    if len(quantization) != 1:
        raise RuntimeError(
            'Hailo-10 native Part2 bridge requires exactly one HEF output QuantInfo; '
            f'observed={len(quantization)}'
        )
    name,row=next(iter(quantization.items()))
    if not isinstance(row, Mapping):
        raise RuntimeError('Hailo-10 HEF output QuantInfo payload is invalid')
    try:
        scale=float(row.get('scale'))
        zero_point=float(row.get('zero_point'))
    except (TypeError, ValueError) as exc:
        raise RuntimeError('Hailo-10 HEF output QuantInfo is non-numeric') from exc
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(zero_point):
        raise RuntimeError('Hailo-10 HEF output QuantInfo is invalid')
    return {
        'name':str(name), 'scale':scale, 'zero_point':zero_point,
        'source':'exact_loaded_hef_quant_info',
    }


def _require_exact_hailo10_engine_quantization(
    engine: Path, quantization: Mapping[str, Any], *, boundary_layout: str,
    boundary_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed unless the existing Part2 bridge matches this HEF exactly."""
    contract=(
        dict(boundary_contract)
        if isinstance(boundary_contract, Mapping)
        else _engine_boundary_contract(engine)
    )
    try:
        scale=float(contract.get('dequant_scale'))
        zero_point=float(contract.get('dequant_zero_point'))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            'Hailo-10 Part2 engine has no exact dequantization metadata; rebuild it'
        ) from exc
    expected_scale=float(quantization['scale'])
    expected_zero_point=float(quantization['zero_point'])
    if str(contract.get('bridge_schema') or '') != 'onnx-splitpoint/uint8-dequant-bridge':
        raise RuntimeError(
            'Hailo-10 Part2 engine does not declare the required UINT8 dequantization bridge; rebuild it'
        )
    aliases={
        '':'as_input','auto':'as_input','none':'as_input','identity':'as_input',
        'nchw':'as_input','as_target_shape':'as_input','as_manifest_shape':'as_input',
        'nhwc_to_nchw':'memory_nhwc_to_nchw','hwcn_to_nchw':'memory_hwcn_to_nchw',
        'nwhc_to_nchw':'memory_nwhc_to_nchw','ncwh_to_nchw':'memory_ncwh_to_nchw',
        'chwn_to_nchw':'memory_chwn_to_nchw',
    }
    requested=str(boundary_layout or 'as_input').strip().lower().replace('-','_')
    expected_layout=aliases.get(requested,requested)
    actual_layout=str(contract.get('boundary_layout_effective') or '').strip().lower()
    if actual_layout != expected_layout:
        raise RuntimeError(
            'Hailo-10 Part2 engine boundary layout mismatch: '
            f'engine={actual_layout or "missing"} selected={expected_layout}; rebuild it'
        )
    if not np.isclose(scale, expected_scale, rtol=0.0, atol=1e-12) or not np.isclose(
        zero_point, expected_zero_point, rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError(
            'Hailo-10 Part2 engine QuantInfo mismatch: '
            f'engine=(scale={scale}, zero_point={zero_point}) '
            f'hef=(scale={expected_scale}, zero_point={expected_zero_point}); rebuild it'
        )
    return contract

def _pick_hailo_output(outputs: dict[str,Any], trt: NativeTRT) -> tuple[str,np.ndarray]:
    global _STRICT_SPLIT_BOUNDARY_EVIDENCE
    if _STRICT_SPLIT_BOUNDARY is not None:
        if len(trt.inputs) != 1 or len(outputs) != 1:
            raise RuntimeError('native_split_quality_runtime_boundary_tensor_count_mismatch')
        expected_names={
            str(_STRICT_SPLIT_BOUNDARY.get('name') or ''),
            str(_STRICT_SPLIT_BOUNDARY.get('runtime_name') or ''),
        }
        expected_names.discard('')
        runtime_name,raw=next(iter(outputs.items()))
        if str(runtime_name) not in expected_names:
            raise RuntimeError('native_split_quality_runtime_boundary_name_mismatch')
        arr=np.asarray(raw)
        expected_shape=tuple(int(x) for x in list(_STRICT_SPLIT_BOUNDARY.get('shape') or []))
        expected_dtype=str(_STRICT_SPLIT_BOUNDARY.get('dtype') or '').strip().lower()
        if tuple(arr.shape) != expected_shape:
            raise RuntimeError(
                'native_split_quality_runtime_boundary_shape_mismatch:'
                f'name={runtime_name},observed={list(arr.shape)},expected={list(expected_shape)}'
            )
        if str(arr.dtype).lower() != expected_dtype:
            raise RuntimeError('native_split_quality_runtime_boundary_dtype_mismatch')
        target=trt.inputs[0]
        if int(arr.size) != int(np.prod(trt.shapes[target])):
            raise RuntimeError('native_split_quality_runtime_boundary_engine_size_mismatch')
        _STRICT_SPLIT_BOUNDARY_EVIDENCE={
            'status':'exact_runtime_boundary_verified',
            'runtime_name':str(runtime_name), 'shape':[int(x) for x in arr.shape],
            'dtype':str(arr.dtype), 'element_count':int(arr.size),
            'trt_input_name':str(target),
            'trt_input_shape':[int(x) for x in trt.shapes[target]],
            'binding_boundary_metadata_sha256':str(_STRICT_SPLIT_BOUNDARY.get('metadata_sha256') or ''),
        }
        return target,arr
    if len(trt.inputs)==1:
        ti=trt.inputs[0]
        if ti in outputs: return ti, np.asarray(outputs[ti])
        target_numel=int(np.prod(trt.shapes[ti]))
        for k,v in outputs.items():
            arr=np.asarray(v)
            if int(arr.size)==target_numel: return ti, arr
        if outputs: return ti, np.asarray(next(iter(outputs.values())))
    raise RuntimeError('cannot map Hailo output to TensorRT input')



def _extract_slot_outputs(sess: Any, slot: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract a completed slot without rewriting sealed boundary memory.

    Quality-FIRST binds the exact HailoRT VStream shape.  The TensorRT bridge
    owns the layout conversion, so adapting HWC to NCHW here would invalidate
    that evidence and apply the transform twice.
    """
    # Import private adapter defensively; if it ever moves, fall back to raw arrays.
    try:
        from onnx_splitpoint_tool.runners.backends.hailo_backend import _try_adapt_tensor  # type: ignore
    except Exception:
        def _try_adapt_tensor(x, target_shape):  # type: ignore
            return False, x
    binding = slot["binding"]
    out: dict[str, np.ndarray] = {}
    for raw_name in list(getattr(sess, '_hef_output_names', []) or []):
        arr = np.asarray(sess._binding_output(binding, str(raw_name)).get_buffer())  # type: ignore[attr-defined]
        arr = np.array(arr, copy=True)
        cname = getattr(sess, '_output_name_hef_to_canonical', {}).get(str(raw_name), str(raw_name))
        # A leading 1 can be a real HWC height (e.g. MobileNet [1,1,960]).
        # Remove only an extra batch axis proved by the runtime contract.
        runtime_shape = (
            _STRICT_SPLIT_BOUNDARY.get('shape')
            if _STRICT_SPLIT_BOUNDARY is not None
            else getattr(sess, 'runtime_output_shapes', {}).get(str(cname))
            or getattr(sess, '_hef_output_shapes', {}).get(str(raw_name))
        )
        if runtime_shape and tuple(arr.shape) == (1, *tuple(runtime_shape)):
            arr = arr[0]
        if _STRICT_SPLIT_BOUNDARY is not None:
            out[str(cname)] = np.ascontiguousarray(arr)
            continue
        cshape = getattr(sess, 'output_shapes', {}).get(str(cname))
        ok, adapted = _try_adapt_tensor(arr, cshape)
        out[str(cname)] = np.ascontiguousarray(adapted if ok else arr)
    return out


def _capture_raw_hailo10_sample(
    sess: Any, inputs: dict[str, np.ndarray], *, diagnostic_capture: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Run one unmeasured sample through the same raw-slot path as the hotloop.

    ``HailoBackend.run`` intentionally adapts HWC runtime outputs to canonical
    ONNX/NCHW shapes.  The Native TensorRT Part2 engine, however, already owns
    the sealed HWC->NCHW input bridge.  Semantic dumps must therefore capture
    the raw reusable binding slot exactly like the measured producer path.
    """
    infer_inputs = sess._prepare_infer_inputs(inputs)  # type: ignore[attr-defined]
    slot = sess._create_reusable_binding_slot()  # type: ignore[attr-defined]
    sess._fill_reusable_slot_inputs(slot, infer_inputs)  # type: ignore[attr-defined]
    sess._submit_reusable_slot(  # type: ignore[attr-defined]
        slot, infer_inputs, copy_inputs=False,
    )
    sess._wait_reusable_slot(slot)  # type: ignore[attr-defined]
    if diagnostic_capture is not None:
        # Explicit, unmeasured probe only: capture physical names and buffers
        # before canonical-name/layout adaptation. The normal hotloop has no
        # copy, hashing, or diagnostics added by this opt-in branch.
        diagnostic_capture['prepared_inputs'] = {
            str(name): np.array(value, copy=True) for name, value in infer_inputs.items()
        }
        diagnostic_capture['raw_outputs'] = {}
        diagnostic_capture['raw_output_storage'] = {}
        for name in list(getattr(sess, '_hef_output_names', []) or []):
            raw = np.asarray(sess._binding_output(slot['binding'], str(name)).get_buffer())
            diagnostic_capture['raw_output_storage'][str(name)] = {
                'shape': list(raw.shape), 'strides': list(raw.strides),
                'dtype': str(raw.dtype), 'c_contiguous': bool(raw.flags.c_contiguous),
                'nbytes': int(raw.nbytes),
            }
            diagnostic_capture['raw_outputs'][str(name)] = np.array(raw, copy=True)
        diagnostic_capture['runtime_io'] = dict(sess.describe_io() or {})
    return _extract_slot_outputs(sess, slot)


_REPETITION_MEDIAN_KEYS = (
    'hailo_latency_ms', 'p1_ms', 'producer_makespan_ms',
    'consumer_makespan_ms', 'p1_effective_cycle_ms',
    'consumer_effective_cycle_ms', 'output_extract_ms', 'payload_copy_ms',
    'fifo_put_block_ms', 'queue_wait_ms', 'trt_input_copy_ms',
    'handoff_ms', 'p2_run_ms', 'completion_tail_ms', 'postprocess_ms',
    'deepx_run_ms', 'output_map_ms',
    'p1_thread_ms', 'p2_thread_ms', 'paper_equivalent_cycle_ms',
    'paper_equivalent_fps', 'makespan_ms', 'fps_makespan',
)


def _median_ci95(values: list[float]) -> tuple[float, float, float]:
    """Deterministic percentile-bootstrap CI for a repetition-level median."""
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return float('nan'), float('nan'), float('nan')
    med = float(np.median(vals))
    if vals.size == 1:
        return med, med, med
    rng = np.random.default_rng(0x261F)
    draws = vals[rng.integers(0, vals.size, size=(20000, vals.size))]
    medians = np.median(draws, axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return med, float(low), float(high)


def _aggregate_repetition_metrics(
    runs: list[dict[str, Any]], *,
    repetition_runtime_scope: str = 'shared_initialized_runtime',
    repetition_independence_verified: bool = False,
) -> dict[str, Any]:
    """Aggregate independent measurement intervals by median, never best-of."""
    if not runs:
        raise ValueError('at least one repetition result is required')
    out = dict(runs[-1])
    for key in _REPETITION_MEDIAN_KEYS:
        vals = [float(row[key]) for row in runs if row.get(key) is not None and np.isfinite(float(row[key]))]
        if vals:
            out[key] = float(np.median(vals))
    fps_values = [float(row['fps_makespan']) for row in runs if row.get('fps_makespan') is not None and np.isfinite(float(row['fps_makespan']))]
    fps_median, fps_low, fps_high = _median_ci95(fps_values)
    records=[]
    for i,row in enumerate(runs):
        record=dict(row, repetition_index=i + 1)
        completed=int(record.get('completed_work_units') or record.get('completed_frames') or record.get('consumed_frames') or record.get('frames') or 0)
        record.update({'frames': completed, 'completed_frames': completed, 'completed_work_units': completed, 'ok': True, 'status': 'ok'})
        records.append(record)
    runtime_instance_ids = [str(row.get('runtime_instance_id') or '') for row in records]
    unique_runtime_instances = bool(
        runtime_instance_ids
        and all(runtime_instance_ids)
        and len(set(runtime_instance_ids)) == len(runtime_instance_ids)
    )
    completed_counts = [int(row.get('completed_work_units') or 0) for row in records]
    identical_positive_work = bool(
        completed_counts and all(count > 0 for count in completed_counts)
        and len(set(completed_counts)) == 1
    )
    independence_verified = bool(
        repetition_independence_verified
        and repetition_runtime_scope in {'fresh_runtime_per_repetition', 'fresh_process_per_repetition'}
        and unique_runtime_instances
        and identical_positive_work
    )
    out.update({
        'repetitions_requested': len(runs),
        'repetitions_completed': len(runs),
        'repetition_count_requested': len(runs),
        'repetition_count_attempted': len(runs),
        'repetition_count_valid': len(runs),
        'repetition_status': 'complete',
        'repetition_aggregation': 'median_never_best_of',
        'fps_makespan': fps_median,
        'fps_makespan_median': fps_median,
        'fps_makespan_ci95_low': fps_low,
        'fps_makespan_ci95_high': fps_high,
        'fps_median': fps_median,
        'fps_ci95_low': fps_low,
        'fps_ci95_high': fps_high,
        'fps_makespan_ci95_method': 'deterministic_percentile_bootstrap_of_repetition_medians_20000',
        'fps_ci95_method': 'deterministic_percentile_bootstrap_of_repetition_medians_20000',
        'fps_ci95_level': 0.95,
        'aggregate_total_completed_frames': int(sum(int(row.get('completed_frames') or row.get('consumed_frames') or row.get('frames') or 0) for row in runs)),
        'repetition_evidence': records,
        'repetition_records': records,
        'repetition_runtime_scope': str(repetition_runtime_scope),
        'repetition_independence_verified': independence_verified,
        'repetition_runtime_instance_ids': runtime_instance_ids,
    })
    return out


def _hailo10_async_fifo_run(
    sess: Any,
    inputs: dict[str, np.ndarray],
    trt: NativeTRT,
    *,
    frames: int,
    warmup: int,
    inflight: int,
    queue_depth: int,
    duration_s: float = 0.0,
    task: str = 'classification',
    completion_runtime: Any | None = None,
    warmup_completion_runtime: Any | None = None,
) -> dict[str, Any]:
    """Run a real Hailo10 InferModel async producer feeding a TRT consumer.

    Important metric semantics:
      * queue_wait_ms is *not* handoff work; it is backlog/consumer scheduling wait.
      * handoff_ms is the single producer-side boundary materialization plus
        FIFO push work.  There is no second payload copy.
      * p2_run_ms is Native TensorRT H2D+execute+D2H.
      * paper_equivalent_cycle_ms uses the producer effective cycle and p2_run_ms,
        not queue waiting.  This makes the reported model comparable to the
        Hailo8 native C++ FIFO runner and the paper-style two-thread model.
    """
    frames = max(0, int(frames)); warmup = max(0, int(warmup)); inflight = max(1, int(inflight)); queue_depth = max(1, int(queue_depth)); duration_s = max(0.0, float(duration_s or 0.0))
    task_value = str(task or '').strip().lower()
    if task_value not in {'classification', 'detection'}:
        raise RuntimeError('native FIFO task must be classification or detection')
    if task_value == 'detection' and completion_runtime is None:
        raise RuntimeError(
            'detection completion runtime missing from measured hotloop'
        )
    if (
        task_value == 'detection'
        and warmup > 0
        and warmup_completion_runtime is None
    ):
        raise RuntimeError(
            'detection completion runtime missing from warmup hotloop'
        )
    infer_inputs = sess._prepare_infer_inputs(inputs)  # type: ignore[attr-defined]
    slots = [sess._create_reusable_binding_slot() for _ in range(inflight)]  # type: ignore[attr-defined]
    for slot in slots:
        sess._fill_reusable_slot_inputs(slot, infer_inputs)  # type: ignore[attr-defined]

    # Warmup producer+consumer sequentially but with reusable bindings to stabilize caches.
    for i in range(warmup):
        slot = slots[i % inflight]
        sess._submit_reusable_slot(slot, infer_inputs, copy_inputs=False)  # type: ignore[attr-defined]
        sess._wait_reusable_slot(slot)  # type: ignore[attr-defined]
        outs = _extract_slot_outputs(sess, slot)
        name, arr = _pick_hailo_output(outs, trt)
        warm_outputs = trt.run({name: arr})
        if task_value == 'detection':
            warmup_completion_runtime.process(warm_outputs)

    q: queue.Queue[Any] = queue.Queue(maxsize=queue_depth)
    sentinel = object()
    errors: list[str] = []
    hailo_lat_ms: list[float] = []
    p2_ms: list[float] = []
    completion_tail_ms: list[float] = []
    trt_input_copy_ms: list[float] = []
    queue_wait_ms: list[float] = []
    extract_ms: list[float] = []
    fifo_put_ms: list[float] = []
    submit_t: dict[int, float] = {}
    counters = {'produced': 0, 'consumed': 0}
    timing = {'producer_start': 0.0, 'producer_end': 0.0, 'consumer_start': 0.0, 'consumer_end': 0.0}
    ready = threading.Barrier(3)
    start_event = threading.Event()
    cancel_event = threading.Event()
    measurement = {'start': 0.0, 'last_completion': 0.0}

    def put_payload(payload: Any) -> bool:
        while not cancel_event.is_set():
            try:
                q.put(payload, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def submit_slot(slot: dict[str, Any]) -> None:
        submit_t[id(slot)] = time.perf_counter()
        sess._submit_reusable_slot(slot, infer_inputs, copy_inputs=False)  # type: ignore[attr-defined]

    def producer() -> None:
        submitted = 0; completed = 0
        busy: set[int] = set()
        duration_mode = duration_s > 0.0
        ready.wait()
        start_event.wait()
        timing['producer_start'] = time.perf_counter()
        deadline = measurement['start'] + duration_s if duration_mode else 0.0
        try:
            # Prime the Hailo scheduler.
            while (
                not cancel_event.is_set()
                and (duration_mode or submitted < frames)
                and len(busy) < inflight
            ):
                if duration_mode and time.perf_counter() >= deadline:
                    break
                slot = slots[submitted % inflight]
                submit_slot(slot); busy.add(id(slot)); submitted += 1
            while not cancel_event.is_set() and (
                busy or (not duration_mode and completed < frames)
            ):
                progressed = False
                for slot in slots:
                    sid = id(slot)
                    if sid not in busy:
                        continue
                    done = slot.get('done')
                    if done is not None and hasattr(done, 'is_set') and not done.is_set():
                        continue
                    # Completed.  Wait also collects exceptions and releases the job.
                    sess._wait_reusable_slot(slot)  # type: ignore[attr-defined]
                    t_done = time.perf_counter()
                    t_sub = submit_t.pop(sid, t_done)
                    hailo_lat_ms.append((t_done - t_sub) * 1000.0)

                    t_ex0 = time.perf_counter()
                    outs = _extract_slot_outputs(sess, slot)
                    name, arr = _pick_hailo_output(outs, trt)
                    t_ex1 = time.perf_counter()
                    extract_ms.append((t_ex1 - t_ex0) * 1000.0)

                    # _extract_slot_outputs already owns one contiguous copy of the
                    # reusable Hailo binding.  Copying it again here disadvantaged
                    # Hailo10H relative to the Hailo8 C++ handoff.
                    payload = arr
                    t_put0 = time.perf_counter()
                    if not put_payload(
                        (name, payload, time.perf_counter())
                    ):
                        break
                    t_put1 = time.perf_counter()
                    fifo_put_ms.append((t_put1 - t_put0) * 1000.0)

                    completed += 1; counters['produced'] = completed; busy.remove(sid); progressed = True
                    if (duration_mode or submitted < frames) and not (duration_mode and time.perf_counter() >= deadline):
                        submit_slot(slot); busy.add(id(slot)); submitted += 1
                if not progressed:
                    time.sleep(0.00005)
        except Exception as e:
            errors.append('producer: ' + repr(e))
            cancel_event.set()
        finally:
            try:
                # Ensure no in-flight job remains.
                for slot in slots:
                    try:
                        sess._wait_reusable_slot(slot)  # type: ignore[attr-defined]
                    except Exception as e:
                        errors.append('producer_wait: ' + repr(e))
            finally:
                timing['producer_end'] = time.perf_counter()
                if cancel_event.is_set():
                    try:
                        q.put_nowait(sentinel)
                    except queue.Full:
                        pass
                else:
                    q.put(sentinel)

    def consumer() -> None:
        ready.wait()
        start_event.wait()
        timing['consumer_start'] = time.perf_counter()
        try:
            while True:
                try:
                    item = q.get(timeout=0.05)
                except queue.Empty:
                    if cancel_event.is_set():
                        break
                    continue
                if item is sentinel:
                    break
                name, payload, t_put = item
                t0 = time.perf_counter()
                queue_wait_ms.append((t0 - t_put) * 1000.0)
                trt.prepare_inputs({name: payload})
                t1 = time.perf_counter()
                trt_outputs = trt.run_prepared()
                t2 = time.perf_counter()
                if task_value == 'detection':
                    completion_runtime.process(trt_outputs)
                    t3 = time.perf_counter()
                else:
                    t3 = t2
                trt_input_copy_ms.append((t1 - t0) * 1000.0)
                p2_ms.append((t2 - t1) * 1000.0)
                completion_tail_ms.append((t3 - t2) * 1000.0)
                counters['consumed'] += 1
                measurement['last_completion'] = t3
        except Exception as e:
            errors.append('consumer: ' + repr(e))
            cancel_event.set()
        finally:
            timing['consumer_end'] = time.perf_counter()

    th_p = threading.Thread(target=producer, daemon=True); th_c = threading.Thread(target=consumer, daemon=True)
    th_p.start(); th_c.start()
    # Both workers exist and are parked before the measured interval begins.
    # This excludes Python thread construction/scheduling from every backend's
    # makespan while retaining FIFO fill and drain.
    ready.wait()
    t0 = time.perf_counter()
    measurement['start'] = t0
    start_event.set()
    th_p.join(); th_c.join()
    if errors:
        raise RuntimeError('; '.join(errors))

    measured_frames = int(counters.get('consumed') or 0)
    if measured_frames <= 0 or measurement['last_completion'] <= 0.0:
        raise RuntimeError('no measured Hailo10H->TensorRT frames completed')
    t1 = float(measurement['last_completion'])
    makespan_ms = max(0.0, (t1 - t0) * 1000.0)
    producer_ms = max(0.0, (timing.get('producer_end', t1) - timing.get('producer_start', t0)) * 1000.0)
    consumer_ms = max(0.0, (timing.get('consumer_end', t1) - timing.get('consumer_start', t0)) * 1000.0)
    fps_makespan = (measured_frames / (makespan_ms / 1000.0)) if makespan_ms > 0 and measured_frames > 0 else 0.0
    producer_effective_ms = (producer_ms / measured_frames) if producer_ms > 0 and measured_frames > 0 else 0.0
    consumer_effective_ms = (consumer_ms / measured_frames) if consumer_ms > 0 and measured_frames > 0 else 0.0
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    p2 = mean(p2_ms)
    qwait = mean(queue_wait_ms)
    extract = mean(extract_ms)
    copy = 0.0
    fifo_put = mean(fifo_put_ms)
    input_copy = mean(trt_input_copy_ms)
    completion_tail = mean(completion_tail_ms)
    hlat = mean(hailo_lat_ms)
    # Actual handoff work excludes queue backlog wait.
    handoff_work = extract + copy + fifo_put + input_copy
    p2_thread = input_copy + p2 + completion_tail
    cycle = max(producer_effective_ms, p2_thread)
    result = {
        'producer_impl': 'hailo10_infermodel_async_fifo',
        'frames': measured_frames,
        'requested_frames': frames,
        'duration_s': duration_s,
        'warmup': warmup,
        'inflight': inflight,
        'queue_depth': queue_depth,
        'produced_frames': int(counters['produced']),
        'consumed_frames': int(counters['consumed']),
        'hailo_latency_ms': hlat,
        'p1_ms': hlat,
        'producer_makespan_ms': producer_ms,
        'consumer_makespan_ms': consumer_ms,
        'p1_effective_cycle_ms': producer_effective_ms,
        'consumer_effective_cycle_ms': consumer_effective_ms,
        'output_extract_ms': extract,
        'payload_copy_ms': copy,
        'boundary_copy_count': 1,
        'boundary_copy_policy': 'single_owned_contiguous_copy_from_reusable_accelerator_binding',
        'fifo_put_block_ms': fifo_put,
        'queue_wait_ms': qwait,
        'trt_input_copy_ms': input_copy,
        'handoff_ms': handoff_work,
        'p2_run_ms': p2,
        'completion_tail_ms': completion_tail,
        'postprocess_ms': completion_tail,
        'p1_thread_ms': producer_effective_ms,
        'p2_thread_ms': p2_thread,
        'paper_equivalent_cycle_ms': cycle,
        'paper_equivalent_fps': (1000.0 / cycle) if cycle > 0 else 0.0,
        'makespan_ms': makespan_ms,
        'fps_makespan': fps_makespan,
        'measurement_boundary': (
            'workers_ready_to_last_completed_task_frame'
            if task_value == 'detection'
            else 'workers_ready_to_last_completed_trt_frame'
        ),
        'last_completion_source': (
            'same_hotloop_completed_task_sentinel'
            if task_value == 'detection'
            else 'native_trt_synchronized_output'
        ),
        'warmup_contract': 'fully_drained_before_worker_start',
        'trt_host_memory_policy': getattr(trt, 'host_memory_policy', 'unknown'),
        'trt_output_materialization_policy': getattr(trt, 'output_materialization_policy', 'unknown'),
        'trt_copy_outputs': True,
        'completed_frames': int(counters['consumed']),
        'completed_work_units': int(counters['consumed']),
        'completed_work_units_source': (
            'same_hotloop_detection_completion_counter'
            if task_value == 'detection'
            else 'trt_consumer_successful_completion_counter'
        ),
        'completed_work_units_status': 'exact_runtime_counter',
        'metric_note': 'For async_fifo, queue_wait_ms is backlog wait and is not counted as handoff work. Detection p2_thread_ms and makespan include the serial completed-task tail.',
    }
    if task_value == 'detection':
        result.update(_completion_attestation_fields(
            completion_runtime,
            completed_work_units=int(counters['consumed']),
        ))
    else:
        result.update({
            'postprocess_included': False,
            'postprocess_completed_frames': 0,
            'postprocess_completion_verified': False,
        })
    return result




def _benchmark_task(bs: Path) -> str:
    try:
        payload = _load_json(bs / 'benchmark_set.json') or {}
        return str(payload.get('benchmark_task') or payload.get('task') or payload.get('model_task') or '').strip().lower()
    except Exception:
        return ''

def _benchmark_model_id(bs: Path) -> str:
    payload = _load_json(bs / 'benchmark_set.json')
    if isinstance(payload, Mapping):
        return str(payload.get('model_id') or payload.get('model_name') or '').strip()
    return ''


def _native_output_contract(
    task: str, outputs: dict[str, np.ndarray],
    declared_contract: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    return runtime_output_contract(
        task, outputs, raw_fallback=False, declared_contract=declared_contract,
    )


def _endpoint_report_fields(manifest_path: str | Path) -> dict[str, Any]:
    payload = _load_json(Path(manifest_path).expanduser())
    payload = dict(payload) if isinstance(payload, Mapping) else {}
    return {
        'task': str(payload.get('task') or ''),
        'output_format': str(payload.get('output_format') or ''),
        'contract_family': str(payload.get('contract_family') or ''),
        'stage': str(payload.get('stage') or ''),
        'contract_source': str(payload.get('contract_source') or ''),
        'endpoint_contract_complete': payload.get('endpoint_contract_complete') is True,
        'endpoint_contract_hash': str(payload.get('endpoint_contract_hash') or ''),
        'tensor_signature': payload.get('tensor_signature') if isinstance(payload.get('tensor_signature'), Mapping) else {},
        'output_endpoint_attestation': payload.get('output_endpoint_attestation') if isinstance(payload.get('output_endpoint_attestation'), Mapping) else {},
    }


def _image_model_geometry(
    inputs: Mapping[str, Any],
    image_path: str,
) -> tuple[list[int], list[int]]:
    """Resolve the single image-input H/W and exact source-image W/H."""
    if Image is None:
        raise RuntimeError(
            'detection completion requires PIL/Pillow for exact image geometry'
        )
    image = Path(str(image_path or '')).expanduser()
    if not image.is_file():
        raise RuntimeError(
            'detection completion requires an exact source image'
        )
    candidates: list[tuple[int, int]] = []
    for value in inputs.values():
        shape = tuple(int(dim) for dim in np.asarray(value).shape)
        if len(shape) == 4 and shape[0] == 1:
            if shape[1] in (1, 3, 4):
                candidates.append((shape[2], shape[3]))
            if shape[3] in (1, 3, 4):
                candidates.append((shape[1], shape[2]))
        elif len(shape) == 3:
            if shape[0] in (1, 3, 4):
                candidates.append((shape[1], shape[2]))
            if shape[2] in (1, 3, 4):
                candidates.append((shape[0], shape[1]))
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1 or min(candidates[0]) <= 0:
        raise RuntimeError(
            'detection completion image-input geometry is ambiguous'
        )
    with Image.open(image) as source:
        original_wh = [int(source.width), int(source.height)]
    return [int(candidates[0][0]), int(candidates[0][1])], original_wh


def _build_detection_completion_contract(
    *,
    benchmark_set: Path,
    model_id: str,
    outputs: Mapping[str, Any],
    inputs: Mapping[str, Any],
    image_path: str,
    preprocess_mode: str,
    letterbox_pad_value: int,
) -> dict[str, Any]:
    input_hw, original_wh = _image_model_geometry(inputs, image_path)
    declaration = load_authoritative_output_contract(
        benchmark_set,
        backend='tensorrt',
        model_id=str(model_id),
        variant='full',
        task='detection',
    )
    source_endpoint = runtime_output_contract(
        'detection',
        outputs,
        raw_fallback=False,
        declared_contract=declaration,
    )
    contract = build_detection_completion_execution_contract(
        model_id=str(model_id),
        outputs=outputs,
        input_hw=input_hw,
        original_wh=original_wh,
        source_endpoint_contract=source_endpoint,
        preprocess={
            'mode': str(preprocess_mode),
            'rgb': True,
            'pad_value': int(letterbox_pad_value),
        },
    )
    return verify_detection_completion_execution_contract(contract)


def _bound_detection_completion_contract(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    options = dict(binding.get('runtime_options') or {})
    contract = verify_detection_completion_execution_contract(
        options.get('completion_execution_contract')
    )
    if str(
        options.get('completion_execution_contract_sha256') or ''
    ).strip().lower() != str(contract.get('contract_sha256') or ''):
        raise RuntimeError(
            'split_energy_completion_execution_contract_sha256_mismatch'
        )
    return contract


def _completion_attestation_fields(
    completion_runtime: Any,
    *,
    completed_work_units: int,
) -> dict[str, Any]:
    if int(getattr(completion_runtime, 'completed_count', -1)) != int(
        completed_work_units
    ):
        raise RuntimeError(
            'detection completion count does not match completed work units'
        )
    raw = completion_runtime.attestation(
        completed_work_units=int(completed_work_units)
    )
    if not isinstance(raw, Mapping):
        raise RuntimeError('detection completion attestation missing')
    attestation = (
        verify_detection_completion_execution_attestation(
            raw,
            execution_contract=completion_runtime.execution_contract,
            expected_observation_relation='same_hotloop_sentinel',
        )
        if isinstance(completion_runtime, DetectionCompletionRuntime)
        else dict(raw)
    )
    if (
        attestation.get('attested') is not True
        or attestation.get('status') != 'passed'
        or attestation.get('observation_relation')
        != 'same_hotloop_sentinel'
        or attestation.get('exact_result_claim_bound') is not True
        or int(attestation.get('completed_work_units') or 0)
        != int(completed_work_units)
        or int(attestation.get('completion_count') or 0)
        != int(completed_work_units)
    ):
        raise RuntimeError('detection completion attestation invalid')
    completed_endpoint = dict(
        attestation.get('completed_endpoint_contract') or {}
    )
    comparison_endpoint = dict(
        attestation.get('comparison_endpoint_contract') or {}
    )
    return {
        'completed_work_units': int(completed_work_units),
        'postprocess_included': True,
        'postprocess_completed_frames': int(completed_work_units),
        'postprocess_completion_verified': True,
        'completion_execution_attestation': attestation,
        'completion_execution_contract_sha256': str(
            attestation.get('execution_contract_sha256') or ''
        ),
        'completion_observation_relation': 'same_hotloop_sentinel',
        'completion_exact_result_claim_bound': True,
        'completed_task_endpoint_contract': completed_endpoint,
        'completed_task_endpoint_contract_hash': str(
            completed_endpoint.get('endpoint_contract_hash') or ''
        ),
        'completed_task_output_endpoint_id': str(
            completed_endpoint.get('output_endpoint_id') or ''
        ),
        'comparison_endpoint_contract': comparison_endpoint,
        'comparison_endpoint_contract_hash': str(
            comparison_endpoint.get('endpoint_contract_hash') or ''
        ),
        'completed_task_comparison_endpoint_contract':
            comparison_endpoint,
        'completed_task_comparison_endpoint_contract_hash': str(
            comparison_endpoint.get('endpoint_contract_hash') or ''
        ),
        'completed_task_comparison_output_endpoint_id': str(
            comparison_endpoint.get('output_endpoint_id') or ''
        ),
        'completed_task_stage': 'decoded_nms',
        'completed_task_contract_family': 'decoded_nms',
        'completed_task_completion_mode':
            'detection_completion_execution_v1',
        'completed_task_endpoint_attested': True,
        'completed_task_endpoint_attestation': attestation,
        'completed_task_endpoint_attestation_status': 'passed',
        'completion_artifact_sha256': str(
            attestation.get('artifact_sha256') or ''
        ),
        'completion_schema_sha256': str(
            attestation.get('schema_sha256') or ''
        ),
        'completion_content_sha256': str(
            attestation.get('content_sha256') or ''
        ),
        'completion_invocation_sha256': str(
            attestation.get('invocation_sha256') or ''
        ),
        'completion_relation_sha256': str(
            attestation.get('relation_sha256') or ''
        ),
    }


def _dump_trt_outputs(outputs: dict[str, np.ndarray], out_dir: Path, *, producer: str, case: str, backend: str, benchmark_set: Path, input_image: str = "", task: str = "") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries=[]
    for i,(name, arr) in enumerate(outputs.items()):
        a=np.ascontiguousarray(np.asarray(arr))
        fname=f"output_{i:02d}_{''.join(c if c.isalnum() or c in '._-' else '_' for c in str(name))}.bin"
        (out_dir/fname).write_bytes(a.tobytes())
        entries.append({'name':str(name),'file':fname,'dtype':str(a.dtype),'shape':[int(x) for x in a.shape],'bytes':int(a.nbytes)})
    declaration = load_authoritative_output_contract(
        benchmark_set, backend='tensorrt', model_id=_benchmark_model_id(benchmark_set),
        variant='full', task=task,
    )
    manifest={'schema':'onnx-splitpoint/runner-output-dump','schema_version':4,'producer':producer,'backend':backend,'case':case,'input_image':str(input_image or ''),'input_image_sha256':_file_sha256(input_image) if input_image else '', 'provenance':{'image':str(input_image or ''),'image_sha256':_file_sha256(input_image) if input_image else '', 'image_source':'exact_file' if input_image else 'synthetic_or_runtime_input'},'authoritative_output_contract_resolution':declaration,**_native_output_contract(task, outputs, declaration),'outputs':entries}
    mp=out_dir/'native_outputs_manifest.json'
    mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return str(mp)


def _input_hwc_uint8(inputs: dict[str, np.ndarray]) -> tuple[np.ndarray | None, list[int]]:
    """Return first native input as HWC uint8 for Full-ONNX self-reference feeds."""
    if not inputs:
        return None, []
    arr = np.asarray(next(iter(inputs.values())))
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        return None, []
    if arr.shape[-1] in (1, 3, 4):
        hwc = arr[..., :3]
    elif arr.shape[0] in (1, 3, 4):
        hwc = np.transpose(arr[:3, ...], (1, 2, 0))
    else:
        return None, []
    if hwc.dtype == np.uint8:
        out = hwc
    else:
        f = hwc.astype(np.float32, copy=False)
        if f.size and float(np.nanmax(f)) <= 1.5:
            f = f * 255.0
        out = np.clip(np.nan_to_num(f), 0, 255).astype(np.uint8)
    out = np.ascontiguousarray(out)
    return out, [int(out.shape[0]), int(out.shape[1]), int(out.shape[2])]


def _reference_input_hwc_uint8(*, input_image: str, inputs: dict[str, np.ndarray], preprocess_mode: str, letterbox_pad_value: int) -> tuple[np.ndarray | None, list[int]]:
    """Return RGB pixels, never HEF quantization codes, for Quality reference."""
    if input_image and inputs:
        runtime_shape=tuple(int(x) for x in np.asarray(next(iter(inputs.values()))).shape)
        raw_rgb=_image_to_shape(
            input_image, runtime_shape, True,
            preprocess_mode=preprocess_mode,
            letterbox_pad_value=int(letterbox_pad_value),
        )
        return _input_hwc_uint8({'input':raw_rgb})
    return _input_hwc_uint8(inputs)


def _dump_hailo10_boundary(out_dir: Path, *, boundary_name: str, boundary: np.ndarray, inputs: dict[str, np.ndarray], trt: NativeTRT, case: str, precision: str, boundary_layout: str = "as_input", input_image: str = "", preprocess_mode: str = "resize", letterbox_pad_value: int = 114, quantized_inputs: bool = True, quantized_outputs: bool = True) -> str:
    """Write a Hailo8-compatible boundary manifest for Hailo10H E2E rows.

    The boundary tensor is the exact Hailo10H Part1 output selected for the
    TensorRT Part2 input.  The input_rgb_uint8.bin dump is used by the normal
    Full-ONNX self-reference validators.
    """
    bdir = out_dir / 'native_fifo_boundary'
    bdir.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(np.asarray(boundary))
    dtype = str(arr.dtype)
    safe_dtype = ''.join(c if c.isalnum() else '_' for c in dtype)
    bfile = bdir / f'boundary_{safe_dtype}.bin'
    bfile.write_bytes(arr.tobytes())
    inp_hwc, inp_shape = _reference_input_hwc_uint8(
        input_image=input_image,
        inputs=inputs,
        preprocess_mode=preprocess_mode,
        letterbox_pad_value=letterbox_pad_value,
    )
    input_dump = ''
    if inp_hwc is not None:
        ip = bdir / 'input_rgb_uint8.bin'
        ip.write_bytes(np.ascontiguousarray(inp_hwc).tobytes())
        input_dump = str(ip)
    trt_input_name = trt.inputs[0] if trt.inputs else str(boundary_name)
    trt_shape = list(trt.shapes.get(trt_input_name, arr.shape)) if trt_input_name else list(arr.shape)
    trt_dtype = str(trt.dtypes.get(trt_input_name, arr.dtype)) if trt_input_name else dtype
    manifest = {
        'schema': 'onnx-splitpoint/native-boundary-dump',
        'schema_version': 3,
        'backend': 'hailo10h_to_trt',
        'producer_impl': 'hailo10_infermodel_async_fifo',
        'dtype': dtype,
        'nbytes': int(arr.nbytes),
        'seq': 0,
        'hailo_output_name': str(boundary_name),
        'trt_input_name': str(trt_input_name),
        'trt_input_dtype': str(trt_dtype),
        'trt_input_bytes': int(np.prod(trt_shape) * np.dtype(trt.dtypes.get(trt_input_name, arr.dtype)).itemsize) if trt_input_name else int(arr.nbytes),
        'shape': [int(x) for x in arr.shape],
        'runtime_boundary_shape': [int(x) for x in arr.shape],
        'trt_input_shape': [int(x) for x in trt_shape],
        'boundary_shape_source': 'hailo_runtime_output_binding',
        'boundary_layout': str(boundary_layout or 'as_input'),
        'layout_transform_owner': 'tensorrt_part2_input_bridge',
        'image': str(input_image or ''),
        'image_sha256': _file_sha256(input_image) if input_image else '',
        'file': str(bfile),
        'input_dump': input_dump,
        'input_shape_hwc': inp_shape,
        'preprocess': {
            'mode': (preprocess_mode + '_rgb_uint8') if input_image else 'synthetic_or_prepared_input',
            'pad_value': int(letterbox_pad_value) if input_image and preprocess_mode == 'letterbox' else 0,
            'rgb': True,
            'ort_model_scale': 'norm',
            'quantized_inputs': bool(quantized_inputs),
            'quantized_outputs': bool(quantized_outputs),
        },
        'input_image': str(input_image or ''),
        'provenance': {
            'image': str(input_image or ''),
            'image_sha256': _file_sha256(input_image) if input_image else '',
            'image_source': 'exact_file' if input_image else 'synthetic_or_runtime_input',
            'seq': 0,
            'precision': str(precision),
            'backend': 'hailo10h_to_trt',
        }
    }
    mp = bdir / 'native_fifo_boundary_manifest.json'
    mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return str(mp)


def _hailo10_energy_binding(ns: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    binding, reason = load_split_energy_workload_binding(
        ns.energy_preflight_attestation,
        expected_nonce=str(ns.energy_preflight_nonce or ''),
        expected_command_contract_sha256=str(ns.source_contract_sha256 or ''),
        expected_backend='hailo10h_to_trt',
        max_age_s=float(ns.energy_preflight_max_age_s),
    )
    if binding is None:
        return None, reason
    options = dict(binding.get('runtime_options') or {})
    boundary = dict(binding.get('boundary_contract') or {})
    prepared_contract = dict(binding.get('prepared_input_contract') or {})
    errors=[]
    if int(ns.warmup) != 0 or int(options.get('warmup') if options.get('warmup') is not None else -1) != 0:
        errors.append('split_energy_warmup_must_be_zero')
    if ns.build_missing_engine or ns.force_rebuild_engine:
        errors.append('split_energy_build_must_be_disabled')
    if ns.dump_outputs or ns.dump_boundary or options.get('dump_outputs') or options.get('dump_boundary'):
        errors.append('split_energy_dumps_must_be_disabled')
    if int(ns.repetitions) != 1:
        errors.append('split_energy_repetitions_must_be_one')
    if float(ns.duration_s or 0.0) <= 0.0:
        errors.append('split_energy_duration_must_be_positive')
    checks={
        'benchmark_set': (str(ns.benchmark_set), str(binding.get('benchmark_set') or '')),
        'case': (str(ns.case), str(binding.get('case') or '')),
        'precision': (str(ns.precision), str(binding.get('precision') or '')),
        'image': (str(Path(ns.image).expanduser().resolve()), str(Path(str(binding.get('input_image') or '')).expanduser().resolve())),
        'hw_arch': (str(ns.hw_arch), str(binding.get('hw_arch') or 'hailo10h')),
        'queue_depth': (str(int(ns.queue_depth)), str(int(options.get('queue_depth') or 0))),
        'inflight': (str(int(ns.inflight)), str(int(options.get('inflight') or 0))),
        'producer_impl': (str(ns.producer_impl), str(options.get('producer_impl') or '')),
        'quantized_inputs': (str(bool(ns.quantized_inputs)), str(bool(options.get('quantized_inputs')))),
        'quantized_outputs': (str(bool(ns.quantized_outputs)), str(bool(options.get('quantized_outputs')))),
        'copy_outputs': (str(bool(ns.copy_outputs)), str(bool(options.get('copy_outputs')))),
        'boundary_layout': (str(ns.boundary_layout), str(boundary.get('boundary_layout_effective') or '')),
        'task': (str(ns.task), str(options.get('task') or '')),
        'preprocess_mode': (str(ns.preprocess_mode), str(options.get('preprocess_mode_requested') or '')),
        'letterbox_pad_value': (
            str(int(ns.letterbox_pad_value)),
            str(int(options.get('letterbox_pad_value_requested')))
            if options.get('letterbox_pad_value_requested') is not None else '-1',
        ),
    }
    for label,(actual,expected) in checks.items():
        if actual != expected:
            errors.append(f'split_energy_cli_{label}_mismatch')
    if str(options.get('producer_impl') or '').lower() not in {'auto','async_fifo'}:
        errors.append('split_energy_hailo10_async_fifo_required')
    if not list(options.get('canonical_input_slot_names') or []):
        errors.append('split_energy_canonical_input_slot_names_missing')
    if not list(options.get('canonical_output_slot_names') or []):
        errors.append('split_energy_canonical_output_slot_names_missing')
    if not hailo10_preprocess_binding(options, prepared_contract):
        errors.append('split_energy_prepared_preprocess_contract_mismatch')
    if str(options.get('task') or '') == 'detection':
        try:
            _bound_detection_completion_contract(binding)
        except Exception as exc:
            errors.append(
                'split_energy_detection_completion_contract_invalid:'
                f'{type(exc).__name__}:{exc}'
            )
    if errors:
        return None, ';'.join(errors)
    return binding, reason

def main() -> int:
    ap=argparse.ArgumentParser(description='Experimental Hailo10H->NativeTRT Python E2E FIFO runner')
    ap.add_argument('--benchmark-set', required=True); ap.add_argument('--case', required=True)
    ap.add_argument('--hw-arch', default='hailo10h'); ap.add_argument('--precision', default='uint8_dequant_fp16')
    ap.add_argument('--build-missing-engine', action='store_true')
    ap.add_argument('--engine-build-python', default='auto', help='Python interpreter used only to build the native TRT Part2 engine. auto retries with user-site enabled.')
    ap.add_argument('--force-rebuild-engine', action='store_true', help='Remove an existing Part2 engine before rebuilding. Needed when layout/dequant contract changes.')
    ap.add_argument('--boundary-layout', default='as_input', choices=['as_input','memory_nwc_to_ncw','memory_nhwc_to_nchw','memory_hwcn_to_nchw','memory_nwhc_to_nchw','memory_ncwh_to_nchw','memory_chwn_to_nchw'], help='Raw Hailo10 boundary layout before the Part2 input bridge.')
    ap.add_argument('--dequant-scale', type=float, default=0.0, help='Optional uint8_dequant_fp16 scale forwarded to TRT bridge builder.')
    ap.add_argument('--dequant-zero-point', type=float, default=0.0, help='Optional uint8_dequant_fp16 zero point forwarded to TRT bridge builder.')
    ap.add_argument('--frames', type=int, default=1000); ap.add_argument('--warmup', type=int, default=100); ap.add_argument('--duration-s', type=float, default=0.0, help='Run measured workload for this many seconds instead of fixed frame count (energy mode).'); ap.add_argument('--queue-depth', type=int, default=3); ap.add_argument('--inflight', type=int, default=8)
    ap.add_argument('--repetitions', type=int, default=1, help='Independent performance intervals; reported as median with a 95%% repetition-level CI (never best-of).')
    quantized_inputs = ap.add_mutually_exclusive_group()
    quantized_inputs.add_argument('--quantized-inputs', dest='quantized_inputs', action='store_true', default=True)
    quantized_inputs.add_argument('--no-quantized-inputs', dest='quantized_inputs', action='store_false')
    quantized_outputs = ap.add_mutually_exclusive_group()
    quantized_outputs.add_argument('--quantized-outputs', dest='quantized_outputs', action='store_true', default=True)
    quantized_outputs.add_argument('--no-quantized-outputs', dest='quantized_outputs', action='store_false')
    copy_outputs = ap.add_mutually_exclusive_group()
    copy_outputs.add_argument('--copy-outputs', dest='copy_outputs', action='store_true', default=True)
    copy_outputs.add_argument('--no-copy-outputs', dest='copy_outputs', action='store_false')
    ap.add_argument('--dump-outputs', action='store_true', help='Dump one final NativeTRT output set for shared native validation/visualization.')
    ap.add_argument('--dump-boundary', action='store_true', help='Dump selected Hailo10 Part1 boundary plus input_rgb_uint8.bin for Full-ONNX self-reference validation.')
    ap.add_argument('--producer-impl', choices=['sync','async_fifo','auto'], default='auto', help='Hailo10 producer implementation. auto uses async_fifo.')
    ap.add_argument('--image', default='', help='Optional exact validation image to feed for dump/validation runs. If set, synthetic random input is not used.')
    ap.add_argument('--task', default='', choices=['','classification','detection'])
    ap.add_argument('--preprocess-mode', default='auto', choices=['auto','resize','letterbox'])
    ap.add_argument('--letterbox-pad-value', type=int, default=114)
    ap.add_argument('--out-dir', default='', help='Fresh result/output root for energy or A/B replay; defaults to the canonical native_pipeline directory.')
    ap.add_argument('--expected-runner-sha256', default='', help='Replay guard for this staged runner.')
    ap.add_argument('--expected-image-sha256', default='', help='Replay guard for the exact input image.')
    ap.add_argument('--expected-hef-sha256', default='', help='Replay guard for the exact Part1 HEF.')
    ap.add_argument('--expected-engine-sha256', default='', help='Replay guard for the exact Part2 engine.')
    ap.add_argument('--source-contract-sha256', default='', help='Archived successful command contract being replayed.')
    ap.add_argument('--setup-id', default=os.environ.get('ONNX_SPLITPOINT_SETUP_ID',''))
    ap.add_argument('--eval-run-id', default='')
    ap.add_argument('--source-run-id', default='hailo10h_to_trt')
    ap.add_argument('--model-id', default='')
    ap.add_argument('--native-split-quality-binding', default='')
    ap.add_argument('--energy-workload-only', action='store_true', help='Run only the preflight-attested warmup-free measured FIFO hotloop.')
    ap.add_argument('--energy-preflight-attestation', default='', help='Fresh nonce-bound split-energy preflight attestation JSON.')
    ap.add_argument('--energy-preflight-nonce', default='', help='Fresh collector repeat nonce expected in the preflight attestation.')
    ap.add_argument('--energy-preflight-max-age-s', type=float, default=300.0)
    ns=ap.parse_args()
    if ns.force_rebuild_engine:
        ap.error('productive_force_build_disabled: --force-rebuild-engine is disabled; --build-missing-engine remains available')
    if int(ns.repetitions) < 1:
        ap.error('--repetitions must be >= 1')
    energy_binding: dict[str, Any] | None = None
    energy_binding_status = ''
    if ns.energy_workload_only:
        if not ns.energy_preflight_attestation or not ns.energy_preflight_nonce:
            print('split_energy_preflight_attestation_or_nonce_missing', file=sys.stderr)
            return 6
        energy_binding, energy_binding_status = _hailo10_energy_binding(ns)
        if energy_binding is None:
            print(f'split_energy_attestation_rejected:{energy_binding_status}', file=sys.stderr)
            return 6
    elif not bool(ns.copy_outputs):
        print('native_claim_run_requires_copy_outputs', file=sys.stderr)
        return 6
    bs=Path(ns.benchmark_set).expanduser().resolve(); case=_case_id(ns.case); cdir=bs/case
    task_effective=str(ns.task or _benchmark_task(bs) or '').strip().lower()
    if task_effective not in {'classification', 'detection'}:
        print('benchmark_task_missing_or_invalid', file=sys.stderr)
        return 6
    resolved_model_id = str(
        ns.model_id or _benchmark_model_id(bs) or bs.parent.name
    ).strip()
    if task_effective == 'detection' and not resolved_model_id:
        print('detection_completion_model_id_missing', file=sys.stderr)
        return 6
    preprocess_mode_effective=(str(ns.preprocess_mode) if str(ns.preprocess_mode) != 'auto' else ('letterbox' if task_effective == 'detection' else 'resize'))
    quality_binding=_load_split_quality_binding(ns,case=case,task=task_effective)
    if quality_binding is not None:
        if ns.build_missing_engine or ns.force_rebuild_engine:
            raise RuntimeError('native_split_quality_engine_rebuild_forbidden')
        selection=quality_binding['preselection']
        ns.precision=str(selection['precision'])
        ns.boundary_layout=str(selection['boundary_layout'])
        ns.quantized_outputs=str(selection['hailo_format']) == 'uint8'
        ns.preprocess_mode=str(selection['preprocess_mode'])
        preprocess_mode_effective=str(selection['preprocess_mode'])
        ns.letterbox_pad_value=int(selection['letterbox_pad_value'])
        boundary_metadata=_load_json(_binding_path(quality_binding,'boundary_metadata'))
        boundary_tensor=boundary_metadata.get('boundary_tensor') if isinstance(boundary_metadata,dict) else None
        if not isinstance(boundary_tensor,dict):
            raise RuntimeError('native_split_quality_boundary_metadata_tensor_missing')
        global _STRICT_SPLIT_BOUNDARY
        _STRICT_SPLIT_BOUNDARY={
            **boundary_tensor,
            'metadata_sha256':str(boundary_metadata.get('metadata_sha256') or ''),
        }
    out_dir=(Path(ns.out_dir).expanduser().resolve() if ns.out_dir else bs/'native_pipeline'/case/'hailo10h_to_trt'/ns.precision); out_dir.mkdir(parents=True, exist_ok=True)
    out_path=out_dir/'hailo10_native_fifo_e2e_results.json'
    if energy_binding is not None:
        energy_artifacts = dict(energy_binding.get('artifacts') or {})
        hef = Path(str((energy_artifacts.get('hef') or {}).get('path') or ''))
        engine = Path(str((energy_artifacts.get('engine') or {}).get('path') or ''))
        attested_input_sha = str(energy_binding.get('input_image_sha256') or '')
    elif quality_binding is not None:
        hef=_binding_path(quality_binding,'part1_runtime')
        engine=_binding_path(quality_binding,'engine')
        attested_input_sha=_file_sha256(ns.image) if ns.image else ''
    else:
        hef=_find_hef(cdir); engine=_engine_path(bs, case, ns.precision)
        attested_input_sha = _file_sha256(ns.image) if ns.image else ''
    report={'ok':False,'mode':'hailo10h_python_native_fifo_e2e','benchmark_set':str(bs),'case':case,'hw_arch':ns.hw_arch,'precision':ns.precision,'producer_impl_requested':ns.producer_impl,'input_image':str(ns.image or ''),'input_image_sha256':attested_input_sha, 'input_image_source':'exact_file' if ns.image else 'synthetic_or_runtime_input','boundary_layout':str(ns.boundary_layout),'dequant_scale':float(ns.dequant_scale or 0.0),'dequant_zero_point':float(ns.dequant_zero_point or 0.0),'hef':str(hef) if hef else '', 'engine':str(engine) if engine.exists() else '', 'energy_workload_only':bool(ns.energy_workload_only), 'energy_preflight_status':energy_binding_status}
    if not hef: report['error']='missing_hailo10_part1_hef'; out_path.write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2)); return 2
    if ns.force_rebuild_engine and engine.exists():
        try:
            engine.unlink()
            report['engine_force_rebuilt'] = True
        except Exception as _e:
            report['engine_force_rebuild_warning'] = f'{type(_e).__name__}: {_e}'
    if energy_binding is not None:
        report['replay_artifact_verification'] = {
            'status': 'verified_by_nonce_bound_preflight_before_collector',
            'source_contract_sha256': str(ns.source_contract_sha256 or ''),
            'input_image_sha256': str(energy_binding.get('input_image_sha256') or ''),
            'artifact_sha256': {
                str(name): str(row.get('sha256') or '')
                for name, row in dict(energy_binding.get('artifacts') or {}).items()
                if isinstance(row, Mapping)
            },
        }
    elif quality_binding is not None:
        part1_onnx=None
    else:
        try:
            report['replay_artifact_verification'] = {
                'runner_sha256': _assert_expected_sha256('runner', Path(__file__).resolve(), ns.expected_runner_sha256),
                'input_image_sha256': _assert_expected_sha256('input_image', ns.image, ns.expected_image_sha256),
                'hef_sha256': _assert_expected_sha256('hef', hef, ns.expected_hef_sha256),
                'engine_sha256': _assert_expected_sha256('engine', engine, ns.expected_engine_sha256),
                'source_contract_sha256': str(ns.source_contract_sha256 or ''),
            }
        except Exception as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            report['replay_verification_failed'] = True
            out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2))
            return 4
    opts={'hef_path':str(hef),'hw_arch':ns.hw_arch,'runtime_api':'infer_model','quantized_inputs':bool(ns.quantized_inputs),'quantized_outputs':bool(ns.quantized_outputs),'hotloop':True,'copy_outputs':bool(ns.copy_outputs)}
    if energy_binding is not None:
        bound_options = dict(energy_binding.get('runtime_options') or {})
        bound_artifacts = dict(energy_binding.get('artifacts') or {})
        part1_row = bound_artifacts.get('part1_onnx')
        part1_onnx = (
            Path(str(part1_row.get('path') or ''))
            if bound_options.get('part1_onnx_used') is True and isinstance(part1_row, Mapping)
            else None
        )
    else:
        part1_onnx=_find_part1_onnx(cdir, case)
    if energy_binding is not None:
        opts['canonical_input_slot_names'] = [str(x) for x in (bound_options.get('canonical_input_slot_names') or [])]
        opts['canonical_output_slot_names'] = [str(x) for x in (bound_options.get('canonical_output_slot_names') or [])]
    elif part1_onnx:
        opts['onnx_model_path']=str(part1_onnx)
    backend=HailoBackend(strict=True, **opts)
    prepared=None; trt=None
    try:
        prepared=backend.prepare(RunCfg(model_path=hef, options=opts), out_dir/'hailo_artifacts')
        hailo_output_format=_selected_hailo_output_format(
            prepared, bool(ns.quantized_outputs),
        )
        if bool(ns.quantized_outputs):
            if str(ns.precision) != 'uint8_dequant_fp16':
                raise RuntimeError(
                    'Hailo-10 native UINT8 output requires the uint8_dequant_fp16 Part2 bridge'
                )
            exact_output_quantization=_exact_hailo10_output_quantization(prepared)
            ns.dequant_scale=float(exact_output_quantization['scale'])
            ns.dequant_zero_point=float(exact_output_quantization['zero_point'])
            report['hef_output_quantization']=dict(exact_output_quantization)
            report['dequant_scale']=float(ns.dequant_scale)
            report['dequant_zero_point']=float(ns.dequant_zero_point)
        else:
            exact_output_quantization={}
        if not engine.exists() and ns.build_missing_engine:
            report['engine_build_steps']=_build_engine(
                bs, case, ns.precision, ns.engine_build_python,
                boundary_layout=ns.boundary_layout,
                dequant_scale=ns.dequant_scale,
                dequant_zero_point=ns.dequant_zero_point,
            )
        if not engine.exists():
            raise RuntimeError('missing_native_trt_part2_engine')
        if exact_output_quantization:
            bound_boundary_contract = (
                energy_binding.get('boundary_contract')
                if isinstance(energy_binding, Mapping) else None
            )
            report['boundary_contract']=_require_exact_hailo10_engine_quantization(
                engine, exact_output_quantization,
                boundary_layout=str(ns.boundary_layout),
                boundary_contract=bound_boundary_contract,
            )
        report['engine']=str(engine)
        prepared_input_contract: dict[str, Any] = {}
        prepared_input_artifacts: dict[str, dict[str, str]] = {}
        if energy_binding is not None:
            prepared_input_contract = dict(energy_binding.get('prepared_input_contract') or {})
            bound_artifacts = dict(energy_binding.get('artifacts') or {})
            inputs={}
            for entry in list(prepared_input_contract.get('entries') or []):
                if not isinstance(entry, Mapping):
                    raise RuntimeError('split_energy_prepared_input_entry_invalid')
                name = str(entry.get('name') or '')
                artifact_name = str(entry.get('artifact_name') or '')
                row = bound_artifacts.get(artifact_name)
                if not name or not isinstance(row, Mapping):
                    raise RuntimeError('split_energy_prepared_input_artifact_missing')
                arr = np.load(Path(str(row.get('path') or '')), allow_pickle=False)
                if list(arr.shape) != [int(x) for x in list(entry.get('shape') or [])]:
                    raise RuntimeError('split_energy_prepared_input_shape_mismatch')
                if str(arr.dtype) != str(entry.get('dtype') or ''):
                    raise RuntimeError('split_energy_prepared_input_dtype_mismatch')
                if not arr.flags.c_contiguous:
                    raise RuntimeError('split_energy_prepared_input_not_c_contiguous')
                inputs[name] = arr
            if list(inputs) != [str(x) for x in list(prepared.input_names or [])]:
                raise RuntimeError('split_energy_prepared_input_slot_order_mismatch')
        else:
            inputs=_make_input(
                prepared, bool(ns.quantized_inputs), image=(ns.image or None),
                preprocess_mode=preprocess_mode_effective,
                letterbox_pad_value=int(ns.letterbox_pad_value),
                task=task_effective,
            )
            prepared_dir = out_dir / 'contract_artifacts'
            prepared_dir.mkdir(parents=True, exist_ok=True)
            entries=[]
            for index,name in enumerate(list(prepared.input_names or [])):
                arr=np.ascontiguousarray(np.asarray(inputs[name]))
                safe=''.join(c if c.isalnum() or c in '._-' else '_' for c in str(name)) or f'input_{index}'
                path=prepared_dir/f'hailo10_prepared_input_{index:02d}_{safe}.npy'
                np.save(path, arr, allow_pickle=False)
                artifact_name=f'prepared_input_{index:02d}'
                prepared_input_artifacts[artifact_name]={'path':str(path),'sha256':_file_sha256(path)}
                entries.append({'name':str(name),'artifact_name':artifact_name,'shape':[int(x) for x in arr.shape],'dtype':str(arr.dtype),'c_contiguous':bool(arr.flags.c_contiguous)})
            prepared_input_contract={
                'format':'numpy_npy_v1',
                'entries':entries,
                'slot_order':[str(x) for x in list(prepared.input_names or [])],
                'preprocess':'exact_performance_prepared_tensor_persisted',
                'task':task_effective,
                'preprocess_mode_requested':str(ns.preprocess_mode),
                'preprocess_mode':preprocess_mode_effective,
                'preprocess_mode_effective':preprocess_mode_effective,
                'normalization':(
                    'hef_quant_info_from_imagenet_float32'
                    if ns.quantized_inputs and task_effective == 'classification'
                    else 'hef_quant_info_from_unit_float32'
                    if ns.quantized_inputs else 'imagenet_float32'
                    if task_effective == 'classification' else 'divide_255_float32'
                ),
                'letterbox':preprocess_mode_effective == 'letterbox',
                'pad_value':int(ns.letterbox_pad_value) if preprocess_mode_effective == 'letterbox' else None,
                'letterbox_pad_value_requested':int(ns.letterbox_pad_value),
                'letterbox_pad_value_effective':int(ns.letterbox_pad_value) if preprocess_mode_effective == 'letterbox' else 0,
                'letterbox_pad_value':int(ns.letterbox_pad_value) if preprocess_mode_effective == 'letterbox' else 0,
                'pad_value_effective':int(ns.letterbox_pad_value) if preprocess_mode_effective == 'letterbox' else 0,
                'source_image':str(ns.image or ''),
                'source_image_sha256':_file_sha256(ns.image) if ns.image else '',
            }
        completion_execution_contract: dict[str, Any] | None = None
        if task_effective == 'detection':
            if energy_binding is not None:
                completion_execution_contract = (
                    _bound_detection_completion_contract(energy_binding)
                )
                if (
                    completion_execution_contract.get('model_id')
                    != resolved_model_id
                ):
                    raise RuntimeError(
                        'split_energy_detection_completion_model_mismatch'
                    )
                observed_input_hw, observed_original_wh = (
                    _image_model_geometry(inputs, ns.image)
                )
                processor_contract = dict(
                    completion_execution_contract.get(
                        'processor_contract'
                    ) or {}
                )
                if (
                    list(processor_contract.get('input_hw') or [])
                    != observed_input_hw
                    or list(processor_contract.get('original_wh') or [])
                    != observed_original_wh
                ):
                    raise RuntimeError(
                        'split_energy_detection_completion_geometry_mismatch'
                    )
            else:
                probe_trt = NativeTRT(engine)
                try:
                    raw_probe = _capture_raw_hailo10_sample(
                        prepared.handle.session, inputs,
                    )
                    probe_name, probe_boundary = _pick_hailo_output(
                        raw_probe, probe_trt,
                    )
                    probe_outputs = probe_trt.run({
                        probe_name: probe_boundary
                    })
                    completion_execution_contract = (
                        _build_detection_completion_contract(
                            benchmark_set=bs,
                            model_id=resolved_model_id,
                            outputs=probe_outputs,
                            inputs=inputs,
                            image_path=str(ns.image or ''),
                            preprocess_mode=preprocess_mode_effective,
                            letterbox_pad_value=(
                                int(ns.letterbox_pad_value)
                                if preprocess_mode_effective
                                == 'letterbox' else 0
                            ),
                        )
                    )
                finally:
                    probe_trt.close()
            report.update({
                'completion_execution_contract':
                    completion_execution_contract,
                'completion_execution_contract_sha256': str(
                    completion_execution_contract.get(
                        'contract_sha256'
                    ) or ''
                ),
                'completed_task_endpoint_contract': dict(
                    completion_execution_contract.get(
                        'completed_endpoint_contract'
                    ) or {}
                ),
                'comparison_endpoint_contract': dict(
                    completion_execution_contract.get(
                        'comparison_endpoint_contract'
                    ) or {}
                ),
                'stage': str(
                    (
                        completion_execution_contract.get(
                            'source_endpoint'
                        ) or {}
                    ).get('stage') or ''
                ),
                'contract_family': str(
                    (
                        completion_execution_contract.get(
                            'source_endpoint'
                        ) or {}
                    ).get('contract_family') or ''
                ),
                'endpoint_contract_complete': True,
                'endpoint_contract_hash': str(
                    (
                        completion_execution_contract.get(
                            'source_endpoint'
                        ) or {}
                    ).get('endpoint_contract_hash') or ''
                ),
                'output_endpoint_id': str(
                    (
                        completion_execution_contract.get(
                            'source_endpoint'
                        ) or {}
                    ).get('output_endpoint_id') or ''
                ),
            })
        impl = 'async_fifo' if ns.producer_impl == 'auto' else ns.producer_impl
        expected_input_names = [str(x) for x in list(prepared.input_names or [])]
        expected_output_names = [str(x) for x in list(prepared.output_names or [])]
        backend.cleanup(prepared)
        prepared = None

        def _open_fresh_runtime() -> tuple[Any, Any, Any, str]:
            fresh_backend = HailoBackend(strict=True, **opts)
            fresh_prepared = fresh_backend.prepare(
                RunCfg(model_path=hef, options=opts), out_dir/'hailo_artifacts'
            )
            if [str(x) for x in list(fresh_prepared.input_names or [])] != expected_input_names:
                fresh_backend.cleanup(fresh_prepared)
                raise RuntimeError('fresh Hailo runtime input-slot contract changed between repetitions')
            if [str(x) for x in list(fresh_prepared.output_names or [])] != expected_output_names:
                fresh_backend.cleanup(fresh_prepared)
                raise RuntimeError('fresh Hailo runtime output-slot contract changed between repetitions')
            fresh_trt = NativeTRT(engine)
            token = hashlib.sha256(
                f"hailo10-runtime:{time.time_ns()}:{id(fresh_prepared.handle.session)}:{id(fresh_trt)}".encode('utf-8')
            ).hexdigest()
            return fresh_backend, fresh_prepared, fresh_trt, f'fresh_runtime:{token}'

        if impl == 'async_fifo':
            repetition_runs=[]
            for repetition_index in range(int(ns.repetitions)):
                if prepared is not None:
                    backend.cleanup(prepared)
                    prepared = None
                if trt is not None:
                    trt.close()
                    trt = None
                backend, prepared, trt, runtime_instance_id = _open_fresh_runtime()
                sess = prepared.handle.session
                measured_completion_runtime = (
                    DetectionCompletionRuntime(
                        completion_execution_contract,
                        observation_relation='same_hotloop_sentinel',
                    )
                    if task_effective == 'detection'
                    and completion_execution_contract is not None
                    else None
                )
                warmup_completion_runtime = (
                    DetectionCompletionRuntime(
                        completion_execution_contract,
                        observation_relation='same_hotloop_sentinel',
                    )
                    if task_effective == 'detection'
                    and completion_execution_contract is not None
                    and int(ns.warmup) > 0
                    else None
                )
                metrics = _hailo10_async_fifo_run(
                    sess,
                    inputs,
                    trt,
                    frames=ns.frames,
                    warmup=ns.warmup,
                    inflight=ns.inflight,
                    queue_depth=ns.queue_depth,
                    duration_s=ns.duration_s,
                    task=task_effective,
                    completion_runtime=measured_completion_runtime,
                    warmup_completion_runtime=warmup_completion_runtime,
                )
                repetition_runs.append(dict(
                    metrics,
                    repetition_index=repetition_index + 1,
                    runtime_instance_id=runtime_instance_id,
                ))
            report.update(_aggregate_repetition_metrics(
                repetition_runs,
                repetition_runtime_scope='fresh_runtime_per_repetition',
                repetition_independence_verified=True,
            ))
        else:
            # Legacy synchronous producer path kept for comparison/debug.
            if int(ns.repetitions) != 1:
                raise RuntimeError('--repetitions > 1 requires --producer-impl async_fifo (or auto)')
            backend, prepared, trt, runtime_instance_id = _open_fresh_runtime()
            measured_completion_runtime = (
                DetectionCompletionRuntime(
                    completion_execution_contract,
                    observation_relation='same_hotloop_sentinel',
                )
                if task_effective == 'detection'
                and completion_execution_contract is not None
                else None
            )
            warmup_completion_runtime = (
                DetectionCompletionRuntime(
                    completion_execution_contract,
                    observation_relation='same_hotloop_sentinel',
                )
                if task_effective == 'detection'
                and completion_execution_contract is not None
                and int(ns.warmup) > 0
                else None
            )
            for _ in range(max(0, ns.warmup)):
                hout=backend.run(prepared, inputs).outputs
                name, arr=_pick_hailo_output(hout, trt)
                warm_outputs=trt.run({name:arr})
                if task_effective == 'detection':
                    warmup_completion_runtime.process(warm_outputs)
            q: queue.Queue[Any]=queue.Queue(maxsize=max(1, ns.queue_depth))
            p1_times=[]; p2_times=[]; handoff_times=[]; completion_tail_times=[]; errors=[]
            sentinel=object()
            ready=threading.Barrier(3); start_event=threading.Event(); cancel_event=threading.Event()
            measured={'start':0.0, 'last_completion':0.0}
            def put_payload(payload):
                while not cancel_event.is_set():
                    try:
                        q.put(payload,timeout=0.05); return True
                    except queue.Full:
                        continue
                return False
            def prod():
                ready.wait(); start_event.wait()
                try:
                    _n = 0; _start = measured['start']
                    while not cancel_event.is_set() and ((ns.duration_s and time.perf_counter() - _start < ns.duration_s) or ((not ns.duration_s) and _n < ns.frames)):
                        _n += 1
                        t0=time.perf_counter(); out=backend.run(prepared, inputs); t1=time.perf_counter()
                        name, arr=_pick_hailo_output(out.outputs, trt)
                        payload=np.ascontiguousarray(arr)
                        p1_times.append((t1-t0)*1000.0)
                        if not put_payload((name,payload,time.perf_counter())):
                            break
                except Exception as e:
                    errors.append('producer: '+repr(e))
                    cancel_event.set()
                finally:
                    if cancel_event.is_set():
                        try: q.put_nowait(sentinel)
                        except queue.Full: pass
                    else:
                        q.put(sentinel)
            def cons():
                ready.wait(); start_event.wait()
                try:
                    while True:
                        try:
                            item=q.get(timeout=0.05)
                        except queue.Empty:
                            if cancel_event.is_set(): break
                            continue
                        if item is sentinel: break
                        name, payload, t_put=item
                        t0=time.perf_counter(); trt_outputs=trt.run({name:payload}); t1=time.perf_counter()
                        if task_effective == 'detection':
                            measured_completion_runtime.process(trt_outputs)
                            t2=time.perf_counter()
                        else:
                            t2=t1
                        handoff_times.append((t0-t_put)*1000.0)
                        p2_times.append((t1-t0)*1000.0)
                        completion_tail_times.append((t2-t1)*1000.0)
                        measured['last_completion']=t2
                except Exception as e:
                    errors.append('consumer: '+repr(e))
                    cancel_event.set()
            th1=threading.Thread(target=prod, daemon=True); th2=threading.Thread(target=cons, daemon=True)
            th1.start(); th2.start(); ready.wait()
            t_start=time.perf_counter(); measured['start']=t_start; start_event.set()
            th1.join(); th2.join(); t_end=measured['last_completion']
            if errors: report['errors']=errors; raise RuntimeError('; '.join(errors))
            if not p2_times or t_end <= 0.0: raise RuntimeError('no measured Hailo10H->TensorRT frames completed')
            mean=lambda xs: float(np.mean(xs)) if xs else 0.0
            p1=mean(p1_times); p2=mean(p2_times); handoff=mean(handoff_times); completion_tail=mean(completion_tail_times)
            cycle=max(p1,p2+handoff+completion_tail); fps=1000.0/cycle if cycle>0 else 0.0
            measured_frames=len(p2_times)
            makespan_ms=(t_end-t_start)*1000.0; fps_makespan=measured_frames/(makespan_ms/1000.0) if makespan_ms>0 and measured_frames>0 else 0.0
            report.update({'producer_impl':'hailo10_sync_fifo','frames':measured_frames,'completed_frames':measured_frames,'completed_work_units':measured_frames,'produced_frames':len(p1_times),'consumed_frames':measured_frames,'requested_frames':ns.frames,'duration_s':float(ns.duration_s or 0.0),'warmup':ns.warmup,'queue_depth':ns.queue_depth,'inflight':ns.inflight,'hailo_format':hailo_output_format,'trt_input_dtype':str(trt.dtypes[trt.inputs[0]]) if trt.inputs else '', 'p1_ms':p1, 'p1_effective_cycle_ms':p1, 'handoff_ms':handoff, 'p2_run_ms':p2,'completion_tail_ms':completion_tail,'postprocess_ms':completion_tail, 'p1_thread_ms':p1, 'p2_thread_ms':p2+handoff+completion_tail, 'paper_equivalent_cycle_ms':cycle, 'paper_equivalent_fps':fps, 'makespan_ms':makespan_ms,'fps_makespan':fps_makespan,'measurement_boundary':'workers_ready_to_last_completed_task_frame' if task_effective == 'detection' else 'workers_ready_to_last_completed_trt_frame','last_completion_source':'same_hotloop_completed_task_sentinel' if task_effective == 'detection' else 'native_trt_synchronized_output','warmup_contract':'fully_drained_before_worker_start','trt_host_memory_policy':getattr(trt,'host_memory_policy','unknown'),'trt_output_materialization_policy':getattr(trt,'output_materialization_policy','unknown'),'trt_copy_outputs':True,'runtime_instance_id':runtime_instance_id,'repetition_runtime_scope':'fresh_runtime_per_repetition','repetition_independence_verified':True})
            if task_effective == 'detection':
                report.update(_completion_attestation_fields(
                    measured_completion_runtime,
                    completed_work_units=measured_frames,
                ))
            else:
                report.update({
                    'postprocess_included': False,
                    'postprocess_completed_frames': 0,
                    'postprocess_completion_verified': False,
                })

        if bool(getattr(ns, 'dump_outputs', False) or getattr(ns, 'dump_boundary', False)):
            try:
                # Run one deterministic sample through the same producer->TRT contract and dump TRT outputs/boundary.
                hout = _capture_raw_hailo10_sample(
                    prepared.handle.session, inputs,
                )
                dname, darr = _pick_hailo_output(hout, trt)
                if bool(getattr(ns, 'dump_boundary', False)):
                    runtime_name = str(
                        _STRICT_SPLIT_BOUNDARY_EVIDENCE.get('runtime_name')
                        or dname
                    )
                    bm = _dump_hailo10_boundary(out_dir, boundary_name=runtime_name, boundary=np.asarray(darr), inputs=inputs, trt=trt, case=case, precision=ns.precision, boundary_layout=ns.boundary_layout, input_image=(ns.image or ''), preprocess_mode=preprocess_mode_effective, letterbox_pad_value=int(ns.letterbox_pad_value), quantized_inputs=bool(ns.quantized_inputs), quantized_outputs=bool(ns.quantized_outputs))
                    report['native_fifo_boundary_manifest'] = bm
                    report['boundary_manifest'] = bm
                if bool(getattr(ns, 'dump_outputs', False)):
                    douts = trt.run({dname: darr})
                    om = _dump_trt_outputs(douts, out_dir / 'native_outputs', producer='hailo10h_python_native_fifo_e2e', case=case, backend='hailo10h_to_trt', benchmark_set=bs, input_image=(ns.image or ''), task=task_effective)
                    report['output_manifest'] = om
                    report['native_output_manifest'] = om
                    report['native_fifo_output_manifest'] = om
                    report.update(_endpoint_report_fields(om))
            except Exception as _dump_exc:
                report['output_dump_error'] = f'{type(_dump_exc).__name__}: {_dump_exc}'

        report.update({'ok': True, 'hailo_format': hailo_output_format, 'trt_input_dtype': str(trt.dtypes[trt.inputs[0]]) if trt and trt.inputs else '', 'trt_input_bytes': int(np.prod(trt.shapes[trt.inputs[0]]) * trt.dtypes[trt.inputs[0]].itemsize) if trt and trt.inputs else 0, 'trt_inputs': trt.inputs if trt else [], 'trt_outputs': trt.outputs if trt else []})
        if energy_binding is not None:
            exact_count = int(
                report.get('completed_work_units')
                or (
                    report.get('consumed_frames')
                    if task_effective == 'classification' else 0
                )
                or 0
            )
            produced_count = int(report.get('produced_frames') or 0)
            if exact_count <= 0:
                raise RuntimeError('split_energy_exact_completed_frames_missing')
            if produced_count != exact_count:
                raise RuntimeError('split_energy_produced_consumed_count_mismatch')
            if task_effective == 'detection' and (
                report.get('postprocess_completion_verified') is not True
                or int(report.get('postprocess_completed_frames') or 0)
                != exact_count
                or report.get('completion_observation_relation')
                != 'same_hotloop_sentinel'
                or report.get('completion_exact_result_claim_bound')
                is not True
            ):
                raise RuntimeError(
                    'split_energy_detection_completion_attestation_missing'
                )
            report.update({
                'completed_frames': exact_count,
                'completed_work_units': exact_count,
                'warmup': 0,
                'energy_workload_only': True,
                'source_contract_sha256': str(ns.source_contract_sha256 or ''),
            })
        else:
            _seal_manifest_payload_files(report.get('native_fifo_output_manifest'))
            _seal_manifest_payload_files(report.get('native_fifo_boundary_manifest'))
            if quality_binding is not None and _STRICT_SPLIT_BOUNDARY_EVIDENCE.get('status') != 'exact_runtime_boundary_verified':
                raise RuntimeError('native_split_quality_runtime_boundary_evidence_missing')
            input_sha = _file_sha256(ns.image) if ns.image else ''
            contract_artifacts = {
                'python_executable': {'path': str(sys.executable), 'sha256': _file_sha256(sys.executable)},
                'hef': {'path': str(hef), 'sha256': _file_sha256(hef)},
                'engine': {'path': str(engine), 'sha256': _file_sha256(engine)},
                **prepared_input_artifacts,
            }
            native_trt_meta = engine.parent / 'native_trt_meta.json'
            if native_trt_meta.is_file():
                contract_artifacts['native_trt_meta'] = {
                    'path': str(native_trt_meta),
                    'sha256': _file_sha256(native_trt_meta),
                }
            if part1_onnx is not None:
                contract_artifacts['part1_onnx'] = {
                    'path': str(part1_onnx), 'sha256': _file_sha256(part1_onnx),
                }
            if quality_binding is not None:
                role_map={
                    'part1_runtime':'hef', 'boundary_metadata':'boundary_metadata',
                    'source_part2_onnx':'source_part2_onnx', 'build_part2_onnx':'build_part2_onnx',
                    'engine':'engine', 'native_trt_meta':'native_trt_meta',
                    'engine_build_receipt':'engine_build_receipt', 'trtexec':'trtexec',
                }
                for source_name,contract_name in role_map.items():
                    row=(quality_binding.get('artifacts') or {}).get(source_name)
                    if isinstance(row,Mapping): contract_artifacts[contract_name]=dict(row)
            for report_name,artifact_name in (
                ('native_fifo_output_manifest','semantic_output_manifest'),
                ('native_fifo_boundary_manifest','semantic_boundary_manifest'),
            ):
                manifest=Path(str(report.get(report_name) or ''))
                if manifest.is_file():
                    contract_artifacts[artifact_name]={'path':str(manifest.resolve()),'sha256':_file_sha256(manifest),'size_bytes':int(manifest.stat().st_size)}
            contract_payload = {
                'backend': 'hailo10h_to_trt', 'model': str(ns.model_id or bs.parent.name),
                'setup_id': str(ns.setup_id or 'orin_nx_hailo10_01'), 'comparison_backend': 'hailo10h',
                'runner': 'scripts/native_hailo10_trt_e2e_from_benchmarkset.py',
                'python_executable': str(sys.executable),
                'interpreter_identity': {
                    'executable': str(sys.executable),
                    'resolved_executable': str(Path(sys.executable).resolve()),
                    'executable_sha256': _file_sha256(sys.executable),
                    'prefix': str(sys.prefix), 'base_prefix': str(getattr(sys, 'base_prefix', '')),
                    'version': str(sys.version),
                },
                'runner_sha256': _file_sha256(Path(__file__).resolve()),
                'benchmark_set': str(bs), 'case': case, 'hw_arch': str(ns.hw_arch),
                'precision': str(ns.precision), 'input_image': str(ns.image or ''),
                'input_image_source': 'exact_file' if ns.image else 'synthetic_or_runtime_input',
                'input_image_sha256': input_sha,
                'artifacts': contract_artifacts,
                'prepared_input_contract': prepared_input_contract,
                'runtime_options': {
                    'frames': int(ns.frames), 'duration_s': float(ns.duration_s or 0.0),
                    'warmup': int(ns.warmup), 'queue_depth': int(ns.queue_depth),
                    'repetitions': int(ns.repetitions),
                    'inflight': int(ns.inflight), 'producer_impl': str(ns.producer_impl),
                    'quantized_inputs': bool(ns.quantized_inputs),
                    'quantized_outputs': bool(ns.quantized_outputs),
                    'copy_outputs': bool(ns.copy_outputs), 'dump_outputs': bool(ns.dump_outputs),
                    'dump_boundary': bool(ns.dump_boundary), 'build': bool(ns.build_missing_engine),
                    'part1_onnx_used': part1_onnx is not None,
                    'canonical_input_slot_names': [str(x) for x in list(prepared.input_names or [])],
                    'canonical_output_slot_names': [str(x) for x in list(prepared.output_names or [])],
                    'runtime_input_shapes': {
                        str(name): [int(x) for x in shape]
                        for name,shape in dict(prepared.handle.runtime_input_shapes or {}).items()
                    },
                    'runtime_output_shapes': {
                        str(name): [int(x) for x in shape]
                        for name,shape in dict(prepared.handle.runtime_output_shapes or {}).items()
                    },
                    'prepared_input_bound': True,
                    'task': task_effective,
                    'preprocess_mode_requested': str(ns.preprocess_mode),
                    'preprocess_mode_effective': preprocess_mode_effective,
                    'letterbox_pad_value_requested': int(ns.letterbox_pad_value),
                    'letterbox_pad_value_effective': int(ns.letterbox_pad_value) if preprocess_mode_effective == 'letterbox' else 0,
                    'letterbox_pad_value': int(ns.letterbox_pad_value) if preprocess_mode_effective == 'letterbox' else 0,
                    'completion_execution_contract': (
                        completion_execution_contract
                        if task_effective == 'detection' else None
                    ),
                    'completion_execution_contract_sha256': (
                        str(
                            completion_execution_contract.get(
                                'contract_sha256'
                            ) or ''
                        )
                        if completion_execution_contract is not None
                        else ''
                    ),
                },
                'boundary_contract': _engine_boundary_contract(engine),
                'source_contract_sha256': str(ns.source_contract_sha256 or ''),
                'runtime_boundary_evidence': dict(_STRICT_SPLIT_BOUNDARY_EVIDENCE),
            }
            if quality_binding is not None:
                selection=dict(quality_binding.get('preselection') or {})
                quality_source_run_id=canonical_native_split_backend(
                    quality_binding.get('source_run_id'), ns.setup_id,
                )
                contract_payload.update({
                    'eval_run_id':str(quality_binding.get('eval_run_id') or ''),
                    'source_run_id':quality_source_run_id,
                    'native_split_quality_binding':quality_binding,
                    'native_split_quality_binding_sha256':str(quality_binding.get('binding_sha256') or ''),
                    'native_split_quality_eval_run_id':str(quality_binding.get('eval_run_id') or ''),
                    'native_split_quality_source_run_id':quality_source_run_id,
                    **native_split_quality_selection_duplicates(quality_binding),
                    'native_split_quality_local_verification':dict(quality_binding.get('local_artifact_verification') or {}),
                    'quality_preselection':selection,
                    'quality_preselection_sha256':str(selection.get('selection_sha256') or ''),
                    'quality_boundary_contract':dict(quality_binding.get('boundary_contract') or {}),
                    'quality_boundary_contract_sha256':str(quality_binding.get('boundary_contract_sha256') or ''),
                })
            engine_artifact = contract_payload['artifacts'].get('engine')
            if not isinstance(engine_artifact, Mapping):
                raise RuntimeError('native_split_engine_artifact_binding_missing')
            engine_path_bound = str(engine_artifact.get('path') or '')
            engine_sha_bound = str(engine_artifact.get('sha256') or '')
            if not engine_path_bound or not engine_sha_bound:
                raise RuntimeError('native_split_engine_artifact_binding_incomplete')
            # Energy admission deliberately verifies these top-level duplicates
            # against the authoritative Quality-FIRST artifact row.  Project
            # them from that row instead of recomputing an independent value.
            contract_payload['engine'] = engine_path_bound
            contract_payload['engine_sha256'] = engine_sha_bound
            boundary_contract = contract_payload['boundary_contract']
            contract_payload['complete'] = bool(
                input_sha and contract_payload['runner_sha256']
                and all(str(row.get('sha256') or '') for row in contract_payload['artifacts'].values())
                and str(boundary_contract.get('metadata_path') or '')
                and str(boundary_contract.get('metadata_sha256') or '')
                and str(boundary_contract.get('bridge_schema') or '')
                and str(boundary_contract.get('boundary_layout_requested') or '')
                and str(boundary_contract.get('boundary_layout_effective') or '')
                and hailo10_preprocess_binding(
                    contract_payload['runtime_options'], prepared_input_contract,
                )
                and (
                    task_effective != 'detection'
                    or (
                        completion_execution_contract is not None
                        and str(
                            completion_execution_contract.get(
                                'contract_sha256'
                            ) or ''
                        )
                        == str(
                            contract_payload['runtime_options'].get(
                                'completion_execution_contract_sha256'
                            ) or ''
                        )
                    )
                )
            )
            report['native_command_contract'] = seal_native_command_contract(contract_payload)
            report['native_command_contract_sha256'] = report['native_command_contract']['contract_sha256']
            report['workload_contract_sha256'] = report['native_command_contract_sha256']
            if quality_binding is not None:
                report.update({
                    'model_id':str(ns.model_id), 'setup_id':str(ns.setup_id),
                    'eval_run_id':str(ns.eval_run_id),
                    'source_run_id':canonical_native_split_backend(
                        ns.source_run_id, ns.setup_id,
                    ),
                    'native_split_quality_binding':quality_binding,
                    'native_split_quality_binding_sha256':str(quality_binding.get('binding_sha256') or ''),
                    'native_split_quality_eval_run_id':str(quality_binding.get('eval_run_id') or ''),
                    'native_split_quality_source_run_id':canonical_native_split_backend(
                        quality_binding.get('source_run_id'), ns.setup_id,
                    ),
                    **native_split_quality_selection_duplicates(quality_binding),
                })
                joined,join_status=bind_quality_to_native_split(
                    native_row={**report,'backend':'hailo10h_to_trt','model_id':str(ns.model_id),'case_id':case,'task':task_effective,'precision':str(ns.precision),'setup_id':str(ns.setup_id),'comparison_backend':'hailo10h'},
                    quality_binding=quality_binding,
                )
                if joined is None: raise RuntimeError(f'native_split_quality_consumer_join_failed:{join_status}')
                report['native_split_quality_consumer_attestation']=_seal_split_consumer_attestation(joined,report['native_command_contract'],ns,case=case,task=task_effective)
                report['native_split_quality_consumer_status']=join_status
            for records_key in ('repetition_records', 'repetition_evidence'):
                records = report.get(records_key)
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            record['workload_contract_sha256'] = report['workload_contract_sha256']
    except Exception as e:
        report['ok']=False; report['error']=repr(e)
    finally:
        try:
            if trt: trt.close()
        except Exception: pass
        try:
            if prepared: backend.cleanup(prepared)
        except Exception: pass
    if report.get('ok') and task_effective == 'detection':
        try:
            report.update(
                persist_detection_completion_execution_artifacts(
                    report,
                    output_path=out_path.with_name(
                        f'{out_path.stem}.completed_task_result_artifact.json'
                    ),
                )
            )
        except Exception as exc:
            report['ok'] = False
            report['error'] = (
                'completion_execution_artifact_persistence_failed:'
                f'{type(exc).__name__}:{exc}'
            )
    out_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'ok':report.get('ok'), 'fps_makespan':report.get('fps_makespan'), 'paper_fps':report.get('paper_equivalent_fps'), 'producer_impl':report.get('producer_impl'), 'report':str(out_path), 'error':report.get('error','')}, indent=2))
    if ns.energy_workload_only and report.get('ok'):
        print(f"__SPLITPOINT_WORK_UNITS__={int(report.get('completed_work_units') or 0)}")
        print("__SPLITPOINT_WORK_UNITS_SOURCE__=completed_work_units")
        print("__SPLITPOINT_WORK_UNITS_EXACT__=1")
    return 0 if report.get('ok') else 5
if __name__ == '__main__':
    raise SystemExit(main())
