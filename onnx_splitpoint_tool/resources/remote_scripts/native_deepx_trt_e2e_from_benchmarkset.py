#!/usr/bin/env python3
"""DeepX -> Native TensorRT E2E FIFO runner.

This is the first real DeepX producer adapter.  It keeps the same strict native
FIFO contract used by the Hailo fastpaths: DeepX Part1 must produce a single
boundary tensor that can be mapped to the native TensorRT Part2 input.  Complex
multi-boundary contracts are reported as unsupported instead of silently falling
back to the generic runner.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'scripts'
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    from native_hailo10_trt_e2e_from_benchmarkset import (  # type: ignore
        NativeTRT,
        _aggregate_repetition_metrics,
    )
except Exception as exc:  # pragma: no cover
    NativeTRT = None  # type: ignore
    _aggregate_repetition_metrics = None  # type: ignore
    _NATIVE_TRT_IMPORT_ERROR = exc
else:
    _NATIVE_TRT_IMPORT_ERROR = None

from onnx_splitpoint_tool.native_command_contract import (
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


def _case_id(x: str) -> str:
    s = str(x)
    return s if s.startswith('b') else f'b{int(s):03d}'


def _load_json(p: Path) -> Any:
    try:
        return json.loads(
            p.read_text(encoding='utf-8'),
            object_pairs_hook=_no_duplicate_json_keys,
        )
    except Exception:
        return None


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
            'backend':'deepx_to_trt', 'model':str(ns.model_id or ''),
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
        'backend':'deepx_to_trt', 'model_id':str(ns.model_id or ''),
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


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def _find_part1_dxnn(bs: Path, case: str) -> Path | None:
    case_dir = bs / case
    cands = [
        case_dir / 'deepx' / 'deepx_m1' / 'part1' / 'model.dxnn',
        case_dir / 'deepx' / 'deepx_m1' / 'part1' / f'{bs.name}_part1.dxnn',
    ]
    cands += sorted((case_dir / 'deepx').glob('**/part1/*.dxnn')) if (case_dir / 'deepx').exists() else []
    for p in cands:
        if p.exists():
            return p
    return None


def _engine_path(bs: Path, case: str, precision: str) -> Path:
    if precision == 'uint8_cast_fp16':
        name = 'part2_uint8_cast_fp16.engine'
    elif precision == 'uint8_dequant_fp16':
        name = 'part2_uint8_dequant_fp16.engine'
    elif precision == 'fp16':
        name = 'part2_fp16.engine'
    elif precision == 'float32_layout_fp16':
        name = 'part2_float32_layout_fp16.engine'
    else:
        name = f'part2_{precision}.engine'
    return bs / 'native_trt' / case / 'part2' / precision / name


def _run(cmd: list[str], timeout: float | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, env=env)
    return {
        'cmd': cmd,
        'rc': p.returncode,
        'elapsed_s': time.time() - t0,
        'stdout_tail': (p.stdout or '')[-8000:],
        'stderr_tail': (p.stderr or '')[-8000:],
    }


def _python_can_import(py: str, module: str, *, env: dict[str, str] | None = None) -> bool:
    try:
        p = subprocess.run([py, '-c', f'import {module}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20, env=env)
        return p.returncode == 0
    except Exception:
        return False


def _candidate_engine_pythons(requested: str) -> list[tuple[str, dict[str, str] | None, str]]:
    out: list[tuple[str, dict[str, str] | None, str]] = []
    base_env = os.environ.copy()
    def add(py: str | None, env: dict[str, str] | None, label: str) -> None:
        if not py:
            return
        p = shutil.which(py) or py
        if p and p not in [x[0] for x in out]:
            out.append((p, env, label))
    if requested and requested != 'auto':
        add(requested, base_env, 'requested')
        return out
    add(sys.executable, base_env, 'current')
    env_user = base_env.copy(); env_user.pop('PYTHONNOUSERSITE', None)
    add(sys.executable, env_user, 'current_user_site_enabled')
    add('/usr/bin/python3', env_user, 'system_python_user_site_enabled')
    add('/usr/bin/python', env_user, 'system_python_user_site_enabled')
    add('python3', env_user, 'python3_user_site_enabled')
    return out


def _build_native_trt_part2(bs: Path, case: str, precision: str, requested_python: str, timeout: float, *, boundary_layout: str = 'as_input') -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for py, env, label in _candidate_engine_pythons(requested_python):
        can_onnx = _python_can_import(py, 'onnx', env=env)
        prefix = {'engine_build_python': py, 'engine_build_python_label': label, 'engine_build_python_can_import_onnx': can_onnx}
        if not can_onnx:
            steps.append({'name': 'build_native_trt_part2_skip_python_missing_onnx', **prefix, 'rc': 127, 'elapsed_s': 0.0, 'stdout_tail': '', 'stderr_tail': 'python cannot import onnx'})
            continue
        cmd = [py, str(SCRIPTS / 'native_trt_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', case, '--variants', 'part2', '--precision', precision, '--run-smoke', '--iterations', '100', '--workspace-mb', '4096', '--workspace-mode', 'auto']
        if precision in {'float32_layout_fp16', 'uint8_dequant_fp16'} and boundary_layout:
            cmd += ['--boundary-layout', str(boundary_layout)]
        st = _run(cmd, timeout=timeout, env=env); st.update(prefix); st['name'] = 'build_native_trt_part2'; steps.append(st)
        if st.get('rc') == 0:
            break
    return steps


def _find_contract(bs: Path, case: str) -> dict[str, Any]:
    for p in [bs / 'io_contracts' / case / 'io_contract.json', bs / case / 'io_contract.json']:
        j = _load_json(p)
        if isinstance(j, dict):
            return j
    summ = _load_json(bs / 'io_contracts' / 'summary.json')
    if isinstance(summ, dict):
        for c in summ.get('cases', []) or []:
            if str(c.get('case_id')) == case:
                return c
    return {}


def _part1_output_contract_path(bs: Path, case: str) -> Path:
    return bs / case / 'deepx' / 'deepx_m1' / 'part1' / 'output_contract.json'


def _part1_output_contract(bs: Path, case: str) -> dict[str, Any]:
    p = _part1_output_contract_path(bs, case)
    j = _load_json(p)
    return j if isinstance(j, dict) else {}


def _validate_part1_preprocess_contract(
    contract: Mapping[str, Any],
    *,
    task: str,
    requested_mode: str,
    effective_mode: str,
    requested_pad: int,
    effective_pad: int,
) -> dict[str, Any]:
    inp = contract.get('input') if isinstance(contract.get('input'), Mapping) else {}
    source_mode = str(
        inp.get('preprocess_mode_effective') or inp.get('preprocess_mode') or ''
    ).strip().lower()
    source_pad = int(
        inp.get('letterbox_pad_value_effective')
        if inp.get('letterbox_pad_value_effective') is not None
        else inp.get('letterbox_pad_value')
        if inp.get('letterbox_pad_value') is not None
        else -1
    )
    source_task = str(inp.get('task') or '').strip().lower()
    source_requested_mode = str(
        inp.get('preprocess_mode_requested') or ''
    ).strip().lower()
    source_requested_pad = int(
        inp.get('letterbox_pad_value_requested')
        if inp.get('letterbox_pad_value_requested') is not None else -1
    )
    if (
        source_task != task
        or source_requested_mode != requested_mode
        or source_mode != effective_mode
        or source_requested_pad != int(requested_pad)
        or source_pad != int(effective_pad)
    ):
        raise RuntimeError('deepx_part1_task_preprocess_contract_mismatch')
    return dict(inp)


def _input_shape_from_contract(contract: dict[str, Any]) -> tuple[int, ...]:
    shape = (((contract.get('input') or {}).get('shape')) or [1, 3, 640, 640])
    return tuple(int(x) for x in shape)


def _resolve_preprocess_contract(
    task: str,
    requested_mode: str,
    letterbox_pad_value: int,
) -> tuple[str, int]:
    task_value = str(task or '').strip().lower()
    if task_value not in {'classification', 'detection'}:
        raise ValueError('benchmark_task_missing_or_invalid')
    requested = str(requested_mode or 'auto').strip().lower()
    if requested not in {'auto', 'resize', 'letterbox'}:
        raise ValueError('preprocess_mode_invalid')
    effective = (
        'letterbox' if task_value == 'detection' else 'resize'
    ) if requested == 'auto' else requested
    pad_requested = int(letterbox_pad_value)
    if pad_requested < 0 or pad_requested > 255:
        raise ValueError('letterbox_pad_value_out_of_range')
    return effective, pad_requested if effective == 'letterbox' else 0


def _letterbox_rgb_image(image: Any, width: int, height: int, pad_value: int) -> Any:
    source_width, source_height = image.size
    scale = min(float(width) / float(source_width), float(height) / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = image.resize((resized_width, resized_height))
    canvas = Image.new('RGB', (int(width), int(height)), (int(pad_value),) * 3)
    canvas.paste(
        resized,
        ((int(width) - resized_width) // 2, (int(height) - resized_height) // 2),
    )
    return canvas


def _image_variants_for_shape(
    shape: Sequence[int],
    image_path: str,
    *,
    preprocess_mode: str,
    letterbox_pad_value: int,
) -> list[np.ndarray]:
    if Image is None:
        raise RuntimeError('PIL/Pillow is required for --image input preprocessing')
    sh = tuple(int(x) for x in shape)
    im = Image.open(image_path).convert('RGB')
    out: list[np.ndarray] = []
    # Generate common HWC/NHWC/NCHW u8 and float variants; the existing candidate selector chooses the variant whose DeepX output can feed TRT.
    if len(sh) == 4 and sh[0] == 1:
        if sh[1] in (1,3,4):
            c,h,w = sh[1], sh[2], sh[3]
        elif sh[-1] in (1,3,4):
            h,w,c = sh[1], sh[2], sh[3]
        else:
            h = int(sh[-2]); w = int(sh[-1]); c = 3
    elif len(sh) == 3:
        if sh[0] in (1,3,4): c,h,w = sh[0], sh[1], sh[2]
        else: h,w,c = sh[0], sh[1], sh[2]
    else:
        h,w,c = 224,224,3
    if str(preprocess_mode) == 'letterbox':
        prepared_image = _letterbox_rgb_image(
            im, int(w), int(h), int(letterbox_pad_value),
        )
    elif str(preprocess_mode) == 'resize':
        prepared_image = im.resize((int(w), int(h)))
    else:
        raise ValueError(f'unsupported_preprocess_mode:{preprocess_mode}')
    hwc = np.asarray(prepared_image, dtype=np.uint8)
    if c == 1: hwc = hwc[..., :1]
    nhwc = hwc[None, ...]
    nchw_u8 = np.transpose(hwc, (2,0,1))[None, ...].astype(np.uint8, copy=False)
    out.extend([np.ascontiguousarray(hwc.astype(np.uint8, copy=False)), np.ascontiguousarray(nhwc.astype(np.uint8, copy=False)), np.ascontiguousarray(nchw_u8), np.ascontiguousarray(nchw_u8.astype(np.float32)/255.0), np.ascontiguousarray(nhwc.astype(np.float32)/255.0)])
    return out

def _make_candidate_inputs(
    shape: Sequence[int],
    seed: int = 0,
    image: str | None = None,
    *,
    preprocess_mode: str = 'resize',
    letterbox_pad_value: int = 0,
) -> list[np.ndarray]:
    if image:
        return _image_variants_for_shape(
            shape,
            image,
            preprocess_mode=preprocess_mode,
            letterbox_pad_value=letterbox_pad_value,
        )
    rng = np.random.default_rng(seed)
    sh = tuple(int(x) for x in shape)
    out: list[np.ndarray] = []
    if len(sh) == 4 and sh[0] == 1 and sh[1] in (1, 3):
        _, c, h, w = sh
        hwc = rng.integers(0, 256, size=(h, w, c), dtype=np.uint8)
        nhwc = hwc[None, ...]
        nchw_u8 = np.transpose(hwc, (2, 0, 1))[None, ...].astype(np.uint8, copy=False)
        nchw_f = nchw_u8.astype(np.float32) / 255.0
        out.extend([np.ascontiguousarray(hwc), np.ascontiguousarray(nhwc), np.ascontiguousarray(nchw_u8), np.ascontiguousarray(nchw_f)])
    else:
        out.append(np.ascontiguousarray(rng.random(sh).astype(np.float32)))
        out.append(np.ascontiguousarray(rng.integers(0, 256, size=sh, dtype=np.uint8)))
    final=[]; seen=set()
    for a in out:
        k=(tuple(a.shape), str(a.dtype))
        if k not in seen:
            seen.add(k); final.append(a)
    return final


def _exact_input_from_quality_contract(
    contract_input: Mapping[str, Any], image: str, *, preprocess_mode: str,
    letterbox_pad_value: int,
) -> np.ndarray:
    """Create one input fixed before performance; never probe alternatives."""
    expected_shape=tuple(int(x) for x in list(contract_input.get('shape') or []))
    expected_dtype=str(contract_input.get('dtype') or '').strip().lower()
    expected_layout=str(contract_input.get('layout') or '').strip().upper()
    candidates=_image_variants_for_shape(
        expected_shape,image,preprocess_mode=preprocess_mode,
        letterbox_pad_value=letterbox_pad_value,
    )
    matches=[
        value for value in candidates
        if tuple(value.shape) == expected_shape and str(value.dtype).lower() == expected_dtype
    ]
    if expected_layout not in {'HWC','NHWC','CHW','NCHW'} or len(matches) != 1:
        raise RuntimeError('native_split_quality_deepx_exact_input_contract_unresolved')
    return np.ascontiguousarray(matches[0])


def _run_deepx(engine: Any, inp: np.ndarray) -> list[np.ndarray]:
    # dx_engine accepts a positional list in the existing generated runner.
    if hasattr(engine, 'run'):
        outs = engine.run([np.ascontiguousarray(inp)])
    elif hasattr(engine, 'Run'):
        outs = engine.Run([np.ascontiguousarray(inp)])
    else:
        raise RuntimeError('dx_engine.InferenceEngine has no run/Run method')
    if not isinstance(outs, (list, tuple)):
        outs = [outs]
    return [np.asarray(o) for o in outs]


def _select_input_candidate(engine: Any, candidates: list[np.ndarray], trt: Any) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    errors=[]
    target_size = int(np.prod(trt.shapes[trt.inputs[0]])) if len(trt.inputs) == 1 else None
    for candidate_index, cand in enumerate(candidates):
        try:
            outs = _run_deepx(engine, cand)
            # Prefer a candidate whose output can feed the TRT input.
            if target_size is None or any(int(np.asarray(o).size) == int(target_size) for o in outs):
                return cand, outs, {'candidate_index': int(candidate_index), 'selected_shape': list(cand.shape), 'selected_dtype': str(cand.dtype), 'output_shapes': [list(np.asarray(o).shape) for o in outs], 'output_dtypes': [str(np.asarray(o).dtype) for o in outs], 'selection_method': 'performance_probe_then_persist_exact_prepared_input'}
            errors.append({'shape': list(cand.shape), 'dtype': str(cand.dtype), 'error': 'outputs do not match TRT input size', 'output_shapes': [list(np.asarray(o).shape) for o in outs]})
        except Exception as e:
            errors.append({'shape': list(cand.shape), 'dtype': str(cand.dtype), 'error': f'{type(e).__name__}: {e}'})
    raise RuntimeError('No DeepX input candidate worked: ' + json.dumps(errors)[-4000:])


def _map_deepx_output_to_trt(outs: list[np.ndarray], trt: Any) -> tuple[str, np.ndarray, dict[str, Any]]:
    if len(trt.inputs) != 1:
        raise RuntimeError(f'DeepX E2E currently supports single TRT input, got {trt.inputs}')
    name = trt.inputs[0]
    shape = tuple(trt.shapes[name])
    dtype = np.dtype(trt.dtypes[name])
    target_size = int(np.prod(shape))
    selected = None
    strict_evidence: dict[str, Any] = {}
    if _STRICT_SPLIT_BOUNDARY is not None:
        output_names=[str(x) for x in list(_STRICT_SPLIT_BOUNDARY.get('output_names') or [])]
        expected_name=str(_STRICT_SPLIT_BOUNDARY.get('name') or '')
        if len(outs) != len(output_names) or output_names.count(expected_name) != 1:
            raise RuntimeError('native_split_quality_runtime_boundary_name_or_count_mismatch')
        selected=np.asarray(outs[output_names.index(expected_name)])
        expected_shape=tuple(int(x) for x in list(_STRICT_SPLIT_BOUNDARY.get('shape') or []))
        expected_dtype=str(_STRICT_SPLIT_BOUNDARY.get('dtype') or '').strip().lower()
        if tuple(selected.shape) != expected_shape:
            raise RuntimeError('native_split_quality_runtime_boundary_shape_mismatch')
        if str(selected.dtype).lower() != expected_dtype:
            raise RuntimeError('native_split_quality_runtime_boundary_dtype_mismatch')
        strict_evidence={
            'status':'exact_runtime_boundary_verified', 'runtime_name':expected_name,
            'output_index':int(output_names.index(expected_name)),
            'output_count':len(outs), 'shape':[int(x) for x in selected.shape],
            'dtype':str(selected.dtype), 'element_count':int(selected.size),
            'binding_boundary_metadata_sha256':str(_STRICT_SPLIT_BOUNDARY.get('metadata_sha256') or ''),
        }
    else:
        for o in outs:
            arr = np.asarray(o)
            if int(arr.size) == target_size:
                selected = arr; break
        if selected is None:
            selected = np.asarray(outs[0])
    source = np.asarray(selected)
    if int(source.size) != target_size:
        raise RuntimeError(f'DeepX output size {source.size} does not match TRT input {name} size {target_size}; shape={source.shape} target={shape}')

    # dx_engine may reuse its output allocation on the very next invocation.
    # A view (including np.ascontiguousarray on an already-contiguous tensor)
    # is therefore not a safe FIFO payload.  Allocate the final TRT-shaped,
    # TRT-typed buffer once and copy into it once.  This is the single owned
    # boundary copy; the consumer's pinned H2D staging remains part of the
    # measured Part-2 side of the pipeline.
    arr = np.empty(shape, dtype=dtype, order='C')
    np.copyto(arr, source.reshape(shape), casting='unsafe')
    converted = source.dtype != dtype
    copy_meta = {
        'source_name': str(_STRICT_SPLIT_BOUNDARY.get('name') or '') if _STRICT_SPLIT_BOUNDARY is not None else '',
        'trt_input': name,
        'target_shape': list(shape),
        'target_dtype': str(dtype),
        'source_shape': list(source.shape),
        'source_dtype': str(source.dtype),
        'converted': converted,
        'nbytes': int(arr.nbytes),
        'boundary_copy_count': 1,
        'boundary_copy_policy': 'one_owned_contiguous_typed_copy_before_fifo_enqueue',
        'fifo_payload_owns_memory': bool(arr.flags.owndata),
        'fifo_payload_c_contiguous': bool(arr.flags.c_contiguous),
        'fifo_payload_shares_deepx_output': bool(np.shares_memory(arr, source)),
        'strict_quality_boundary': strict_evidence,
    }
    if not copy_meta['fifo_payload_owns_memory'] or not copy_meta['fifo_payload_c_contiguous'] or copy_meta['fifo_payload_shares_deepx_output']:
        raise RuntimeError('DeepX FIFO payload ownership contract failed')
    return name, arr, copy_meta




def _best_hwc_uint8_from_input(selected_input: np.ndarray, image_path: str = "") -> tuple[np.ndarray | None, list[int], str]:
    """Return an HWC uint8 image dump for Full-ONNX self-reference.

    DeepX input contracts can be HWC/NHWC/NCHW and uint8/float.  The validation
    oracle needs the exact image semantics that fed DeepX.  Never recreate this
    evidence from the source image: doing so silently replaced letterboxing with
    a stretch-resize in v2.66.  Derive the compatibility dump from the selected
    tensor itself and fail closed when its layout cannot be represented.
    """
    arr = np.asarray(selected_input)
    h = w = c = 0
    if arr.ndim == 4 and arr.shape[0] == 1:
        if arr.shape[1] in (1, 3, 4):
            c, h, w = int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3])
        elif arr.shape[-1] in (1, 3, 4):
            h, w, c = int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3])
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3, 4):
            c, h, w = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        elif arr.shape[-1] in (1, 3, 4):
            h, w, c = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    if c <= 0:
        c = 3
    if h <= 0 or w <= 0:
        h, w = 640, 640
    try:
        a = np.asarray(selected_input)
        if a.ndim == 4 and a.shape[0] == 1:
            a = a[0]
        if a.ndim == 3 and a.shape[0] in (1, 3, 4):
            a = np.transpose(a, (1, 2, 0))
        if a.ndim == 3 and a.shape[-1] in (1, 3, 4):
            if a.dtype.kind == 'f':
                aa = np.clip(a, 0.0, 1.0) * 255.0
            else:
                aa = np.asarray(a)
            hwc = np.ascontiguousarray(np.clip(aa, 0, 255).astype(np.uint8))
            return hwc, [int(hwc.shape[0]), int(hwc.shape[1]), int(hwc.shape[2])], 'deepx_selected_input_exact_hwc_uint8'
    except Exception:
        pass
    return None, [], 'unavailable'


def _dump_boundary(
    boundary: np.ndarray,
    selected_input: np.ndarray,
    out_dir: Path,
    *,
    producer: str,
    backend: str,
    case: str,
    precision: str,
    trt_input_name: str,
    trt_input_dtype: str,
    input_image: str = "",
    boundary_layout: str = "as_input",
    map_meta: dict[str, Any] | None = None,
    preprocess_mode: str,
    letterbox_pad_value: int,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(np.asarray(boundary))
    dtype = str(arr.dtype)
    suffix = dtype.replace('float32', 'float32').replace('uint8', 'uint8')
    fname = f"boundary_{suffix}.bin"
    (out_dir / fname).write_bytes(arr.tobytes())
    exact_selected = np.ascontiguousarray(np.asarray(selected_input))
    selected_input_dump = out_dir / 'deepx_selected_input.npy'
    np.save(selected_input_dump, exact_selected, allow_pickle=False)
    hwc, hwc_shape, input_source = _best_hwc_uint8_from_input(exact_selected, input_image)
    input_dump = ""
    if hwc is not None:
        input_dump = str(out_dir / 'input_rgb_uint8.bin')
        (out_dir / 'input_rgb_uint8.bin').write_bytes(np.ascontiguousarray(hwc).tobytes())
    manifest = {
        'schema': 'onnx-splitpoint/native-boundary-dump',
        'schema_version': 3,
        'backend': backend,
        'producer_impl': producer,
        'dtype': dtype,
        'nbytes': int(arr.nbytes),
        'seq': 0,
        'deepx_output_name': str((map_meta or {}).get('source_name') or ''),
        'trt_input_name': str(trt_input_name),
        'trt_input_dtype': str(trt_input_dtype),
        'trt_input_bytes': int(arr.nbytes),
        'shape': [int(x) for x in arr.shape],
        'boundary_shape_source': 'tensorrt_input_binding',
        'boundary_layout': str(boundary_layout or 'as_input'),
        'image': str(input_image or ''),
        'image_sha256': _file_sha256(input_image) if input_image else '',
        'file': str(out_dir / fname),
        'input_dump': input_dump,
        'input_shape_hwc': hwc_shape,
        'selected_input_dump': str(selected_input_dump),
        'selected_input_dump_sha256': _file_sha256(selected_input_dump),
        'selected_input_shape': [int(x) for x in exact_selected.shape],
        'selected_input_dtype': str(exact_selected.dtype),
        'selected_input_c_contiguous': bool(exact_selected.flags.c_contiguous),
        'preprocess': {
            'mode': f'{str(preprocess_mode)}_rgb_uint8',
            'mode_effective': str(preprocess_mode),
            'pad_value': int(letterbox_pad_value) if str(preprocess_mode) == 'letterbox' else 0,
            'letterbox_pad_value_effective': int(letterbox_pad_value) if str(preprocess_mode) == 'letterbox' else 0,
            'rgb': True,
            'ort_model_scale': 'norm',
            'input_source': input_source,
            'deepx_selected_input_shape': [int(x) for x in np.asarray(selected_input).shape],
            'deepx_selected_input_dtype': str(np.asarray(selected_input).dtype),
        },
        'input_image': str(input_image or ''),
        'provenance': {
            'image': str(input_image or ''),
            'image_sha256': _file_sha256(input_image) if input_image else '',
            'image_source': 'exact_file' if input_image else 'synthetic_or_unknown',
            'precision': str(precision),
            'backend': backend,
        }
    }
    mp = out_dir / 'native_fifo_boundary_manifest.json'
    mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return str(mp)


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
    input_value: Any,
    image_path: str,
) -> tuple[list[int], list[int]]:
    """Resolve model H/W and exact source-image W/H without guessing."""
    if Image is None:
        raise RuntimeError(
            'detection completion requires PIL/Pillow for exact image geometry'
        )
    image = Path(str(image_path or '')).expanduser()
    if not image.is_file():
        raise RuntimeError(
            'detection completion requires an exact source image'
        )
    shape = tuple(int(value) for value in np.asarray(input_value).shape)
    candidates: list[tuple[int, int]] = []
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
            f'detection completion input geometry is ambiguous: {shape}'
        )
    with Image.open(image) as source:
        original_wh = [int(source.width), int(source.height)]
    return [int(candidates[0][0]), int(candidates[0][1])], original_wh


def _build_detection_completion_contract(
    *,
    benchmark_set: Path,
    model_id: str,
    outputs: Mapping[str, Any],
    selected_input: Any,
    image_path: str,
    preprocess_mode: str,
    letterbox_pad_value: int,
) -> dict[str, Any]:
    input_hw, original_wh = _image_model_geometry(
        selected_input, image_path,
    )
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
    raw = options.get('completion_execution_contract')
    contract = verify_detection_completion_execution_contract(raw)
    declared = str(
        options.get('completion_execution_contract_sha256') or ''
    ).strip().lower()
    if declared != str(contract.get('contract_sha256') or ''):
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
    manifest={'schema':'onnx-splitpoint/runner-output-dump','schema_version':4,'producer':producer,'backend':backend,'case':case,'input_image':str(input_image or ''),'input_image_sha256':_file_sha256(input_image) if input_image else '', 'provenance':{'image':str(input_image or ''),'image_sha256':_file_sha256(input_image) if input_image else '', 'image_source':'exact_file' if input_image else 'synthetic_or_unknown'},'authoritative_output_contract_resolution':declaration,**_native_output_contract(task, outputs, declaration),'outputs':entries}
    mp=out_dir/'native_outputs_manifest.json'
    mp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return str(mp)

def _deepx_fifo_run(
    engine: Any,
    input_arr: np.ndarray,
    trt: Any,
    *,
    frames: int,
    warmup: int,
    queue_depth: int,
    duration_s: float = 0.0,
    task: str = 'classification',
    completion_runtime: Any | None = None,
    warmup_completion_runtime: Any | None = None,
) -> dict[str, Any]:
    task_value = str(task or '').strip().lower()
    if task_value not in {'classification', 'detection'}:
        raise RuntimeError('native FIFO task must be classification or detection')
    if task_value == 'detection' and completion_runtime is None:
        raise RuntimeError(
            'detection completion runtime missing from measured hotloop'
        )
    if (
        task_value == 'detection'
        and int(warmup) > 0
        and warmup_completion_runtime is None
    ):
        raise RuntimeError(
            'detection completion runtime missing from warmup hotloop'
        )
    strict_evidence: dict[str, Any] = {}

    def record_strict_evidence(meta: Mapping[str, Any]) -> None:
        current = dict(meta.get('strict_quality_boundary') or {})
        if not current:
            return
        if strict_evidence and current != strict_evidence:
            raise RuntimeError('native_split_quality_runtime_boundary_evidence_drift')
        strict_evidence.update(current)

    # Warmup synchronously.
    for _ in range(max(0, int(warmup))):
        outs = _run_deepx(engine, input_arr)
        n, arr, warmup_meta = _map_deepx_output_to_trt(outs, trt)
        record_strict_evidence(warmup_meta)
        warm_outputs = trt.run({n: arr})
        if task_value == 'detection':
            warmup_completion_runtime.process(warm_outputs)
    q: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_depth)))
    sentinel = object()
    prod_times=[]; map_times=[]; put_times=[]; input_copy_times=[]; p2_times=[]
    completion_tail_times: list[float] = []
    boundary_copy_counts: list[int] = []
    errors=[]
    ready=threading.Barrier(3)
    start_event=threading.Event()
    cancel_event=threading.Event()
    measurement={'start':0.0, 'last_completion':0.0}

    def put_payload(payload: Any) -> bool:
        while not cancel_event.is_set():
            try:
                q.put(payload, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def producer():
        ready.wait(); start_event.wait()
        try:
            _n = 0
            _start = measurement['start']
            duration_mode = float(duration_s or 0.0) > 0.0
            while not cancel_event.is_set() and (
                (duration_mode and time.perf_counter() - _start < float(duration_s))
                or ((not duration_mode) and _n < int(frames))
            ):
                _n += 1
                t0=_now_ms(); outs=_run_deepx(engine, input_arr); t1=_now_ms()
                n, arr, meta = _map_deepx_output_to_trt(outs, trt); t2=_now_ms()
                record_strict_evidence(meta)
                boundary_copy_counts.append(int(meta.get('boundary_copy_count') or 0))
                q0=_now_ms()
                if not put_payload((n, arr)):
                    break
                q1=_now_ms()
                prod_times.append(t1-t0); map_times.append(t2-t1); put_times.append(q1-q0)
        except Exception as e:
            errors.append('producer: '+repr(e))
            cancel_event.set()
        finally:
            if cancel_event.is_set():
                try:
                    q.put_nowait(sentinel)
                except queue.Full:
                    pass
            else:
                q.put(sentinel)
    def consumer():
        ready.wait(); start_event.wait()
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
                n, arr = item
                t0=_now_ms(); trt.prepare_inputs({n: arr}); t1=_now_ms()
                trt_outputs = trt.run_prepared(); t2=_now_ms()
                if task_value == 'detection':
                    completion_runtime.process(trt_outputs)
                    t3=_now_ms()
                else:
                    t3=t2
                input_copy_times.append(t1-t0); p2_times.append(t2-t1)
                completion_tail_times.append(t3-t2)
                measurement['last_completion']=t3 / 1000.0
        except Exception as e:
            errors.append('consumer: '+repr(e))
            cancel_event.set()
    pt=threading.Thread(target=producer, daemon=True); ct=threading.Thread(target=consumer, daemon=True)
    pt.start(); ct.start(); ready.wait()
    t_start=time.perf_counter(); measurement['start']=t_start; start_event.set()
    pt.join(); ct.join()
    if errors:
        raise RuntimeError('; '.join(errors))
    if not p2_times or measurement['last_completion'] <= 0.0:
        raise RuntimeError('no measured DeepX->TensorRT frames completed')
    if len(boundary_copy_counts) != len(p2_times) or any(count != 1 for count in boundary_copy_counts):
        raise RuntimeError('DeepX FIFO did not preserve exactly one owned boundary copy per completed frame')
    makespan_ms = (measurement['last_completion'] - t_start) * 1000.0
    def mean(x: list[float]) -> float: return float(np.mean(x)) if x else float('nan')
    p1_ms = mean(prod_times)
    input_copy_ms = mean(input_copy_times)
    handoff_ms = mean(map_times) + mean(put_times) + input_copy_ms
    p2_ms = mean(p2_times)
    completion_tail_ms = mean(completion_tail_times)
    p1_thread = p1_ms
    p2_thread = input_copy_ms + p2_ms + completion_tail_ms
    cycle = max(p1_thread, p2_thread)
    result = {
        'frames': len(p2_times),
        'completed_frames': len(p2_times),
        'completed_work_units': len(p2_times),
        'requested_frames': int(frames),
        'duration_s': float(duration_s or 0.0),
        'warmup': int(warmup),
        'queue_depth': int(queue_depth),
        'p1_ms': p1_ms,
        'deepx_run_ms': p1_ms,
        'output_map_ms': mean(map_times),
        'fifo_put_block_ms': mean(put_times),
        'trt_input_copy_ms': input_copy_ms,
        'handoff_ms': handoff_ms,
        'p2_run_ms': p2_ms,
        'completion_tail_ms': completion_tail_ms,
        'postprocess_ms': completion_tail_ms,
        'p1_thread_ms': p1_thread,
        'p2_thread_ms': p2_thread,
        'paper_equivalent_cycle_ms': cycle,
        'paper_equivalent_fps': 1000.0/cycle if cycle > 0 else None,
        'makespan_ms': makespan_ms,
        'fps_makespan': float(len(p2_times)) / (makespan_ms/1000.0) if makespan_ms > 0 and len(p2_times) > 0 else None,
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
        'boundary_copy_count': 1,
        'boundary_copy_total': int(sum(boundary_copy_counts)),
        'boundary_copy_policy': 'one_owned_contiguous_typed_copy_before_fifo_enqueue',
        'fifo_payload_ownership': 'owned_no_alias_to_deepx_runtime_output',
        'strict_quality_boundary': strict_evidence,
    }
    if task_value == 'detection':
        result.update(_completion_attestation_fields(
            completion_runtime,
            completed_work_units=len(p2_times),
        ))
    else:
        result.update({
            'postprocess_included': False,
            'postprocess_completed_frames': 0,
            'postprocess_completion_verified': False,
        })
    return result


def _deepx_energy_binding(ns: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    binding, reason = load_split_energy_workload_binding(
        ns.energy_preflight_attestation,
        expected_nonce=str(ns.energy_preflight_nonce or ''),
        expected_command_contract_sha256=str(ns.source_contract_sha256 or ''),
        expected_backend='deepx_to_trt',
        max_age_s=float(ns.energy_preflight_max_age_s),
    )
    if binding is None:
        return None, reason
    options = dict(binding.get('runtime_options') or {})
    boundary = dict(binding.get('boundary_contract') or {})
    artifacts = dict(binding.get('artifacts') or {})
    prepared_contract = binding.get('prepared_input_contract')
    errors=[]
    if int(ns.warmup) != 0 or int(options.get('warmup') if options.get('warmup') is not None else -1) != 0:
        errors.append('split_energy_warmup_must_be_zero')
    if ns.build_missing_engine:
        errors.append('split_energy_build_must_be_disabled')
    if ns.dump_outputs or ns.dump_boundary or options.get('dump_outputs') or options.get('dump_boundary'):
        errors.append('split_energy_dumps_must_be_disabled')
    if int(ns.repetitions) != 1:
        errors.append('split_energy_repetitions_must_be_one')
    if float(ns.duration_s or 0.0) <= 0.0:
        errors.append('split_energy_duration_must_be_positive')
    if not isinstance(artifacts.get('prepared_input'), Mapping):
        errors.append('split_energy_prepared_input_artifact_missing')
    if not isinstance(prepared_contract, Mapping):
        errors.append('split_energy_prepared_input_contract_missing')
    task = str(options.get('task') or '')
    requested_mode = str(options.get('preprocess_mode_requested') or '')
    effective_mode = str(options.get('preprocess_mode_effective') or '')
    try:
        requested_pad = int(options.get('letterbox_pad_value_requested'))
        effective_pad = int(options.get('letterbox_pad_value_effective'))
        resolved_mode, resolved_pad = _resolve_preprocess_contract(
            task, requested_mode, requested_pad,
        )
    except (TypeError, ValueError, OverflowError):
        requested_pad = -1
        effective_pad = -1
        resolved_mode = ''
        resolved_pad = -1
    if (
        task not in {'classification', 'detection'}
        or requested_mode not in {'auto', 'resize', 'letterbox'}
        or effective_mode not in {'resize', 'letterbox'}
        or effective_mode != resolved_mode
        or effective_pad != resolved_pad
        or int(options.get('letterbox_pad_value') if options.get('letterbox_pad_value') is not None else -1) != effective_pad
        or not isinstance(prepared_contract, Mapping)
        or str(prepared_contract.get('task') or '') != task
        or str(prepared_contract.get('preprocess_mode_requested') or '') != requested_mode
        or str(prepared_contract.get('preprocess_mode_effective') or '') != effective_mode
        or int(prepared_contract.get('letterbox_pad_value_requested') if prepared_contract.get('letterbox_pad_value_requested') is not None else -1) != requested_pad
        or int(prepared_contract.get('letterbox_pad_value_effective') if prepared_contract.get('letterbox_pad_value_effective') is not None else -1) != effective_pad
        or int(prepared_contract.get('letterbox_pad_value') if prepared_contract.get('letterbox_pad_value') is not None else -1) != effective_pad
    ):
        errors.append('split_energy_deepx_task_preprocess_contract_mismatch')
    if task == 'detection':
        try:
            _bound_detection_completion_contract(binding)
        except Exception as exc:
            errors.append(
                'split_energy_detection_completion_contract_invalid:'
                f'{type(exc).__name__}:{exc}'
            )
    checks={
        'benchmark_set': (str(ns.benchmark_set), str(binding.get('benchmark_set') or '')),
        'case': (str(ns.case), str(binding.get('case') or '')),
        'precision': (str(ns.precision), str(binding.get('precision') or '')),
        'image': (str(Path(ns.image).expanduser().resolve()), str(Path(str(binding.get('input_image') or '')).expanduser().resolve())),
        'queue_depth': (str(int(ns.queue_depth)), str(int(options.get('queue_depth') or 0))),
        'boundary_layout': (str(ns.boundary_layout), str(boundary.get('boundary_layout_effective') or '')),
        'task': (str(ns.task), task),
        'preprocess_mode': (str(ns.preprocess_mode), requested_mode),
        'letterbox_pad_value': (str(int(ns.letterbox_pad_value)), str(requested_pad)),
    }
    for label,(actual,expected) in checks.items():
        if actual != expected:
            errors.append(f'split_energy_cli_{label}_mismatch')
    if errors:
        return None, ';'.join(errors)
    return binding, reason


def main() -> int:
    ap = argparse.ArgumentParser(description='DeepX->NativeTRT Python FIFO E2E runner')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--precision', default='uint8_cast_fp16', choices=['uint8_cast_fp16','uint8_dequant_fp16','fp16','float32_layout_fp16'])
    ap.add_argument('--build-missing-engine', action='store_true')
    ap.add_argument('--engine-build-python', default='auto')
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--duration-s', type=float, default=0.0, help='Run measured workload for this many seconds instead of fixed frame count (energy mode).')
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--repetitions', type=int, default=1, help='Independent performance intervals; reported as median with a 95%% repetition-level CI (never best-of).')
    ap.add_argument('--timeout', type=float, default=1800)
    ap.add_argument('--dump-outputs', action='store_true')
    ap.add_argument('--dump-boundary', action='store_true')
    ap.add_argument('--boundary-layout', default='as_input', help='DeepX output boundary layout metadata; as_input is the initial bring-up contract.')
    ap.add_argument('--image', default='', help='Optional exact validation image to feed for dump/validation runs. If set, synthetic random input is not used.')
    ap.add_argument('--task', default='', choices=['', 'classification', 'detection'], help='Benchmark task. If omitted it is read from benchmark_set.json outside energy mode.')
    ap.add_argument('--preprocess-mode', default='auto', choices=['auto', 'resize', 'letterbox'], help='Task-specific image geometry; auto resolves classification to resize and detection to letterbox.')
    ap.add_argument('--letterbox-pad-value', type=int, default=114, help='RGB padding used when the effective preprocessing mode is letterbox (0..255).')
    ap.add_argument('--out-dir', default='', help='Fresh result/output root for energy replay; defaults to the canonical native_pipeline directory.')
    ap.add_argument('--expected-runner-sha256', default='', help='Replay guard for this staged runner.')
    ap.add_argument('--expected-image-sha256', default='', help='Replay guard for the exact input image.')
    ap.add_argument('--expected-dxnn-sha256', default='', help='Replay guard for the exact Part1 DXNN.')
    ap.add_argument('--expected-engine-sha256', default='', help='Replay guard for the exact Part2 engine.')
    ap.add_argument('--source-contract-sha256', default='', help='Archived successful command contract being replayed.')
    ap.add_argument('--setup-id', default=os.environ.get('ONNX_SPLITPOINT_SETUP_ID',''))
    ap.add_argument('--eval-run-id', default='')
    ap.add_argument('--source-run-id', default='deepx_to_trt')
    ap.add_argument('--model-id', default='')
    ap.add_argument('--native-split-quality-binding', default='')
    ap.add_argument('--energy-workload-only', action='store_true', help='Run only the preflight-attested prepared-input hotloop.')
    ap.add_argument('--energy-preflight-attestation', default='', help='Fresh nonce-bound split-energy preflight attestation JSON.')
    ap.add_argument('--energy-preflight-nonce', default='', help='Fresh collector repeat nonce expected in the preflight attestation.')
    ap.add_argument('--energy-preflight-max-age-s', type=float, default=300.0)
    ns = ap.parse_args()
    if int(ns.repetitions) < 1:
        ap.error('--repetitions must be >= 1')
    energy_binding: dict[str, Any] | None = None
    energy_binding_status = ''
    if ns.energy_workload_only:
        if not ns.energy_preflight_attestation or not ns.energy_preflight_nonce:
            print('split_energy_preflight_attestation_or_nonce_missing', file=sys.stderr)
            return 6
        energy_binding, energy_binding_status = _deepx_energy_binding(ns)
        if energy_binding is None:
            print(f'split_energy_attestation_rejected:{energy_binding_status}', file=sys.stderr)
            return 6
    bs = Path(ns.benchmark_set).expanduser().resolve(); case = _case_id(ns.case)
    task = str(ns.task or ('' if energy_binding is not None else _benchmark_task(bs))).strip().lower()
    if task not in {'classification', 'detection'}:
        print('benchmark_task_missing_or_invalid', file=sys.stderr)
        return 6
    resolved_model_id = str(
        ns.model_id or _benchmark_model_id(bs) or bs.parent.name
    ).strip()
    if task == 'detection' and not resolved_model_id:
        print('detection_completion_model_id_missing', file=sys.stderr)
        return 6
    try:
        preprocess_mode_effective, letterbox_pad_value_effective = _resolve_preprocess_contract(
            task, str(ns.preprocess_mode), int(ns.letterbox_pad_value),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        print(f'deepx_preprocess_contract_invalid:{exc}', file=sys.stderr)
        return 6
    quality_binding=_load_split_quality_binding(ns,case=case,task=task)
    if quality_binding is not None:
        if ns.build_missing_engine:
            raise RuntimeError('native_split_quality_engine_rebuild_forbidden')
        selection=quality_binding['preselection']
        ns.precision=str(selection['precision'])
        ns.boundary_layout=str(selection['boundary_layout'])
        ns.preprocess_mode=str(selection['preprocess_mode'])
        preprocess_mode_effective=str(selection['preprocess_mode'])
        ns.letterbox_pad_value=int(selection['letterbox_pad_value'])
        letterbox_pad_value_effective=int(selection['letterbox_pad_value']) if preprocess_mode_effective == 'letterbox' else 0
        boundary_metadata_path=_binding_path(quality_binding,'boundary_metadata')
        boundary_metadata=_load_json(boundary_metadata_path)
        boundary_tensor=boundary_metadata.get('boundary_tensor') if isinstance(boundary_metadata,dict) else None
        if not isinstance(boundary_tensor,dict) or not isinstance(boundary_tensor.get('deepx_input_contract'),dict):
            raise RuntimeError('native_split_quality_deepx_embedded_input_contract_missing')
        global _STRICT_SPLIT_BOUNDARY
        _STRICT_SPLIT_BOUNDARY={
            **boundary_tensor,
            'output_names':[str(x) for x in list((boundary_metadata.get('boundary_tensor') or {}).get('output_names') or [boundary_tensor.get('runtime_name') or boundary_tensor.get('name')])],
            'metadata_sha256':str(boundary_metadata.get('metadata_sha256') or ''),
        }
    out_dir = Path(ns.out_dir).expanduser().resolve() if ns.out_dir else bs/'native_pipeline'/case/'deepx_to_trt'/ns.precision; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/'deepx_native_fifo_e2e_results.json'
    attested_input_sha = str(energy_binding.get('input_image_sha256') or '') if energy_binding is not None else (_file_sha256(ns.image) if ns.image else '')
    report: dict[str, Any] = {'ok': False, 'mode': 'deepx_python_native_fifo_e2e', 'benchmark_set': str(bs), 'case': case, 'precision': ns.precision, 'input_image': str(ns.image or ''), 'input_image_sha256': attested_input_sha, 'input_image_source': 'exact_file' if ns.image else 'synthetic_or_unknown', 'task': task, 'preprocess_mode_requested': str(ns.preprocess_mode), 'preprocess_mode_effective': preprocess_mode_effective, 'letterbox_pad_value_requested': int(ns.letterbox_pad_value), 'letterbox_pad_value_effective': int(letterbox_pad_value_effective), 'steps': [], 'energy_workload_only': bool(ns.energy_workload_only), 'energy_preflight_status': energy_binding_status}
    try:
        if energy_binding is not None:
            energy_artifacts = dict(energy_binding.get('artifacts') or {})
            dxnn = Path(str((energy_artifacts.get('dxnn') or {}).get('path') or ''))
            engine_path = Path(str((energy_artifacts.get('engine') or {}).get('path') or ''))
        elif quality_binding is not None:
            dxnn=_binding_path(quality_binding,'part1_runtime')
            engine_path=_binding_path(quality_binding,'engine')
        else:
            dxnn = _find_part1_dxnn(bs, case)
            if not dxnn:
                raise FileNotFoundError(f'DeepX part1 model.dxnn not found for {case}')
            engine_path = _engine_path(bs, case, ns.precision)
        if not engine_path.exists() and ns.build_missing_engine:
            report['steps'].extend(_build_native_trt_part2(bs, case, ns.precision, ns.engine_build_python, ns.timeout, boundary_layout=ns.boundary_layout))
        if not engine_path.exists():
            raise FileNotFoundError(f'Native TRT part2 engine not found: {engine_path}')
        map_meta: dict[str, Any] = {}
        if energy_binding is not None:
            report['replay_artifact_verification'] = {
                'status': 'verified_by_nonce_bound_preflight_before_collector',
                'source_contract_sha256': str(ns.source_contract_sha256 or ''),
                'input_image_sha256': str(energy_binding.get('input_image_sha256') or ''),
                'artifact_sha256': {
                    str(name): str(row.get('sha256') or '')
                    for name,row in dict(energy_binding.get('artifacts') or {}).items()
                    if isinstance(row, Mapping)
                },
            }
        else:
            report['replay_artifact_verification'] = {
                'runner_sha256': _assert_expected_sha256('runner', Path(__file__).resolve(), ns.expected_runner_sha256),
                'input_image_sha256': _assert_expected_sha256('input_image', ns.image, ns.expected_image_sha256),
                'dxnn_sha256': _assert_expected_sha256('dxnn', dxnn, ns.expected_dxnn_sha256),
                'engine_sha256': _assert_expected_sha256('engine', engine_path, ns.expected_engine_sha256),
                'source_contract_sha256': str(ns.source_contract_sha256 or ''),
                'runtime_boundary_evidence': dict((map_meta or {}).get('strict_quality_boundary') or {}),
            }
        if NativeTRT is None:
            raise RuntimeError(f'NativeTRT import failed: {_NATIVE_TRT_IMPORT_ERROR!r}')
        from dx_engine import InferenceEngine  # type: ignore
        dx_engine = InferenceEngine(str(dxnn))
        trt = NativeTRT(engine_path)
        prepared_input_path: Path | None = None
        prepared_input_contract: dict[str, Any] = {}
        completion_execution_contract: dict[str, Any] | None = None
        if energy_binding is not None:
            energy_artifacts = dict(energy_binding.get('artifacts') or {})
            prepared_row = energy_artifacts.get('prepared_input') or {}
            prepared_input_path = Path(str(prepared_row.get('path') or ''))
            prepared_input_contract = dict(energy_binding.get('prepared_input_contract') or {})
            selected_input = np.load(prepared_input_path, allow_pickle=False)
            expected_shape = [int(x) for x in list(prepared_input_contract.get('shape') or [])]
            expected_dtype = str(prepared_input_contract.get('dtype') or '')
            if list(selected_input.shape) != expected_shape:
                raise RuntimeError('split_energy_prepared_input_shape_mismatch')
            if str(selected_input.dtype) != expected_dtype:
                raise RuntimeError('split_energy_prepared_input_dtype_mismatch')
            if not selected_input.flags.c_contiguous:
                raise RuntimeError('split_energy_prepared_input_not_c_contiguous')
            select_meta = {
                **prepared_input_contract,
                'selection_method': 'preflight_attested_prepared_input_no_probe',
            }
            map_meta = {'status': 'not_probed_outside_counted_hotloop'}
            if task == 'detection':
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
                    _image_model_geometry(selected_input, ns.image)
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
            if quality_binding is not None:
                contract_path=boundary_metadata_path
                source_input_contract=dict(_STRICT_SPLIT_BOUNDARY.get('deepx_input_contract') or {})
                if (
                    str(source_input_contract.get('task') or '') != task
                    or str(source_input_contract.get('preprocess_mode_effective') or source_input_contract.get('preprocess_mode') or '') != preprocess_mode_effective
                    or int(source_input_contract.get('letterbox_pad_value_effective') if source_input_contract.get('letterbox_pad_value_effective') is not None else source_input_contract.get('letterbox_pad_value') or 0) != int(letterbox_pad_value_effective)
                ):
                    raise RuntimeError('native_split_quality_deepx_embedded_input_contract_policy_mismatch')
                contract={'input':source_input_contract}
            else:
                contract_path = _part1_output_contract_path(bs, case)
                contract = _part1_output_contract(bs, case)
                source_input_contract = _validate_part1_preprocess_contract(
                    contract,
                    task=task,
                    requested_mode=str(ns.preprocess_mode),
                    effective_mode=preprocess_mode_effective,
                    requested_pad=int(ns.letterbox_pad_value),
                    effective_pad=int(letterbox_pad_value_effective),
                )
            shape = _input_shape_from_contract(contract)
            if quality_binding is not None:
                if not ns.image:
                    raise RuntimeError('native_split_quality_deepx_exact_input_image_required')
                selected_input=_exact_input_from_quality_contract(
                    source_input_contract,str(ns.image),
                    preprocess_mode=preprocess_mode_effective,
                    letterbox_pad_value=letterbox_pad_value_effective,
                )
                first_outs=_run_deepx(dx_engine,selected_input)
                select_meta={
                    'candidate_index':0,
                    'selected_shape':[int(x) for x in selected_input.shape],
                    'selected_dtype':str(selected_input.dtype),
                    'selection_method':'quality_bound_exact_input_contract_no_probe_or_fallback',
                    'source_input_contract':source_input_contract,
                }
            else:
                cands = _make_candidate_inputs(
                    shape,
                    image=(ns.image or None),
                    preprocess_mode=preprocess_mode_effective,
                    letterbox_pad_value=letterbox_pad_value_effective,
                )
                selected_input, first_outs, select_meta = _select_input_candidate(dx_engine, cands, trt)
            _n0, _boundary0, map_meta = _map_deepx_output_to_trt(first_outs, trt)
            report['replay_artifact_verification']['runtime_boundary_evidence'] = dict(
                map_meta.get('strict_quality_boundary') or {}
            )
            if task == 'detection':
                probe_outputs = trt.run({_n0: _boundary0})
                completion_execution_contract = (
                    _build_detection_completion_contract(
                        benchmark_set=bs,
                        model_id=resolved_model_id,
                        outputs=probe_outputs,
                        selected_input=selected_input,
                        image_path=str(ns.image or ''),
                        preprocess_mode=preprocess_mode_effective,
                        letterbox_pad_value=letterbox_pad_value_effective,
                    )
                )
            prepared_dir = out_dir / 'contract_artifacts'
            prepared_dir.mkdir(parents=True, exist_ok=True)
            prepared_input_path = prepared_dir / 'deepx_selected_input.npy'
            np.save(prepared_input_path, np.ascontiguousarray(selected_input), allow_pickle=False)
            prepared_input_contract = {
                'format': 'numpy_npy_v1',
                'shape': [int(x) for x in selected_input.shape],
                'dtype': str(selected_input.dtype),
                'c_contiguous': bool(selected_input.flags.c_contiguous),
                'candidate_index': int(select_meta.get('candidate_index') or 0),
                'selection_method': str(select_meta.get('selection_method') or ''),
                'task': task,
                'preprocess_mode_requested': str(ns.preprocess_mode),
                'preprocess_mode_effective': preprocess_mode_effective,
                'letterbox_pad_value_requested': int(ns.letterbox_pad_value),
                'letterbox_pad_value_effective': int(letterbox_pad_value_effective),
                'letterbox_pad_value': int(letterbox_pad_value_effective),
                'preprocess': f'exact_performance_{preprocess_mode_effective}_rgb_tensor_persisted_after_candidate_validation',
                'source_image': str(ns.image or ''),
                'source_image_sha256': _file_sha256(ns.image) if ns.image else '',
                'source_contract_shape': [int(x) for x in shape],
                'source_input_contract': source_input_contract,
                'source_input_contract_path': str(contract_path),
                'source_input_contract_sha256': _file_sha256(contract_path),
            }
            if bool(ns.dump_boundary):
                _bm = _dump_boundary(
                    _boundary0, selected_input, out_dir / 'native_fifo_boundary',
                    producer='deepx_python_dx_engine_fifo', backend='deepx_to_trt',
                    case=case, precision=ns.precision, trt_input_name=_n0,
                    trt_input_dtype=str(_boundary0.dtype), input_image=(ns.image or ''),
                    boundary_layout=ns.boundary_layout, map_meta=map_meta,
                    preprocess_mode=preprocess_mode_effective,
                    letterbox_pad_value=letterbox_pad_value_effective,
                )
                report['native_fifo_boundary_manifest'] = _bm
                report['boundary_manifest'] = _bm
        if task == 'detection':
            if completion_execution_contract is None:
                raise RuntimeError(
                    'detection completion execution contract missing'
                )
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
        # Input probing is setup, not a repetition.  Release both runtimes and
        # instantiate them anew for every measured interval so the raw repeat
        # records are genuinely independent runtime instances.
        try:
            trt.close()
        except Exception:
            pass
        trt = None
        try:
            if hasattr(dx_engine, 'close'):
                dx_engine.close()
        except Exception:
            pass
        dx_engine = None
        gc.collect()

        repetition_runs=[]
        for repetition_index in range(int(ns.repetitions)):
            if trt is not None:
                try:
                    trt.close()
                except Exception:
                    pass
                trt = None
            if dx_engine is not None:
                try:
                    if hasattr(dx_engine, 'close'):
                        dx_engine.close()
                except Exception:
                    pass
                dx_engine = None
            gc.collect()
            dx_engine = InferenceEngine(str(dxnn))
            trt = NativeTRT(engine_path)
            runtime_instance_id = 'fresh_runtime:' + hashlib.sha256(
                f"deepx-trt-runtime:{time.time_ns()}:{id(dx_engine)}:{id(trt)}".encode('utf-8')
            ).hexdigest()
            measured_completion_runtime = (
                DetectionCompletionRuntime(
                    completion_execution_contract,
                    observation_relation='same_hotloop_sentinel',
                )
                if task == 'detection'
                and completion_execution_contract is not None
                else None
            )
            warmup_completion_runtime = (
                DetectionCompletionRuntime(
                    completion_execution_contract,
                    observation_relation='same_hotloop_sentinel',
                )
                if task == 'detection'
                and completion_execution_contract is not None
                and int(ns.warmup) > 0
                else None
            )
            res = _deepx_fifo_run(
                dx_engine,
                selected_input,
                trt,
                frames=ns.frames,
                warmup=ns.warmup,
                queue_depth=ns.queue_depth,
                duration_s=ns.duration_s,
                task=task,
                completion_runtime=measured_completion_runtime,
                warmup_completion_runtime=warmup_completion_runtime,
            )
            repetition_runs.append(dict(
                res,
                repetition_index=repetition_index + 1,
                runtime_instance_id=runtime_instance_id,
            ))
        report.update(_aggregate_repetition_metrics(
            repetition_runs,
            repetition_runtime_scope='fresh_runtime_per_repetition',
            repetition_independence_verified=True,
        ))
        if bool(ns.dump_outputs):
            try:
                _outs = _run_deepx(dx_engine, selected_input)
                _n, _arr, _ = _map_deepx_output_to_trt(_outs, trt)
                _douts = trt.run({_n: _arr})
                _om = _dump_trt_outputs(_douts, out_dir / 'native_outputs', producer='deepx_python_dx_engine_fifo', case=case, backend='deepx_to_trt', benchmark_set=bs, input_image=(ns.image or ''), task=task)
                report['output_manifest'] = _om
                report['native_output_manifest'] = _om
                report.update(_endpoint_report_fields(_om))
            except Exception as _dump_exc:
                report['output_dump_error'] = f'{type(_dump_exc).__name__}: {_dump_exc}'
        report.update({
            'ok': True,
            'native_fifo_e2e_implemented': True,
            'producer_impl': 'dx_engine_python_fifo',
            'dxnn': str(dxnn),
            'engine': str(engine_path),
            'input_selection': select_meta,
            'boundary_map': map_meta,
            'boundary_layout': str(ns.boundary_layout),
            'trt_inputs': list(getattr(trt, 'inputs', [])),
            'trt_outputs': list(getattr(trt, 'outputs', [])),
        })
        if energy_binding is not None:
            exact_count = int(
                report.get('completed_work_units')
                or (
                    report.get('completed_frames')
                    if task == 'classification' else 0
                )
                or 0
            )
            if exact_count <= 0:
                raise RuntimeError('split_energy_exact_completed_frames_missing')
            if task == 'detection' and (
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
            _seal_manifest_payload_files(report.get('native_output_manifest'))
            _seal_manifest_payload_files(report.get('native_fifo_boundary_manifest'))
            if quality_binding is not None and str(((map_meta or {}).get('strict_quality_boundary') or {}).get('status') or '') != 'exact_runtime_boundary_verified':
                raise RuntimeError('native_split_quality_runtime_boundary_evidence_missing')
            input_sha = _file_sha256(ns.image) if ns.image else ''
            assert prepared_input_path is not None
            contract_payload = {
                'backend': 'deepx_to_trt', 'model': str(ns.model_id or bs.parent.name),
                'setup_id': str(ns.setup_id or 'orin_nx_deepx_m1_01'), 'comparison_backend': 'deepx',
                'runner': 'scripts/native_deepx_trt_e2e_from_benchmarkset.py',
                'python_executable': str(sys.executable),
                'interpreter_identity': {
                    'executable': str(sys.executable),
                    'resolved_executable': str(Path(sys.executable).resolve()),
                    'executable_sha256': _file_sha256(sys.executable),
                    'prefix': str(sys.prefix), 'base_prefix': str(getattr(sys, 'base_prefix', '')),
                    'version': str(sys.version),
                },
                'runner_sha256': _file_sha256(Path(__file__).resolve()),
                'benchmark_set': str(bs), 'case': case, 'precision': str(ns.precision),
                'input_image': str(ns.image or ''),
                'input_image_source': 'exact_file' if ns.image else 'synthetic_or_unknown',
                'input_image_sha256': input_sha,
                'artifacts': {
                    'python_executable': {'path': str(sys.executable), 'sha256': _file_sha256(sys.executable)},
                    'dxnn': {'path': str(dxnn), 'sha256': _file_sha256(dxnn)},
                    'engine': {'path': str(engine_path), 'sha256': _file_sha256(engine_path)},
                    'prepared_input': {'path': str(prepared_input_path), 'sha256': _file_sha256(prepared_input_path)},
                    'part1_input_contract': {'path': str(contract_path), 'sha256': _file_sha256(contract_path)},
                },
                'prepared_input_contract': prepared_input_contract,
                'runtime_options': {
                    'frames': int(ns.frames), 'duration_s': float(ns.duration_s or 0.0),
                    'warmup': int(ns.warmup), 'queue_depth': int(ns.queue_depth),
                    'repetitions': int(ns.repetitions),
                    'dump_outputs': bool(ns.dump_outputs),
                    'dump_boundary': bool(ns.dump_boundary),
                    'build': bool(ns.build_missing_engine),
                    'prepared_input_bound': True,
                    'task': task,
                    'preprocess_mode_requested': str(ns.preprocess_mode),
                    'preprocess_mode_effective': preprocess_mode_effective,
                    'letterbox_pad_value_requested': int(ns.letterbox_pad_value),
                    'letterbox_pad_value_effective': int(letterbox_pad_value_effective),
                    'letterbox_pad_value': int(letterbox_pad_value_effective),
                    'completion_execution_contract': (
                        completion_execution_contract
                        if task == 'detection' else None
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
                'boundary_contract': {
                    'boundary_layout_requested': str(ns.boundary_layout),
                    'boundary_layout_effective': str(ns.boundary_layout),
                    'dequant_scale': None, 'dequant_zero_point': None,
                },
                'source_contract_sha256': str(ns.source_contract_sha256 or ''),
            }
            if quality_binding is not None:
                role_map={
                    'part1_runtime':'dxnn', 'boundary_metadata':'boundary_metadata',
                    'source_part2_onnx':'source_part2_onnx', 'build_part2_onnx':'build_part2_onnx',
                    'engine':'engine', 'native_trt_meta':'native_trt_meta',
                    'engine_build_receipt':'engine_build_receipt', 'trtexec':'trtexec',
                }
                for source_name,contract_name in role_map.items():
                    row=(quality_binding.get('artifacts') or {}).get(source_name)
                    if isinstance(row,Mapping): contract_payload['artifacts'][contract_name]=dict(row)
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
            contract_payload['engine'] = engine_path_bound
            contract_payload['engine_sha256'] = engine_sha_bound
            metadata_artifact = contract_payload['artifacts'].get('native_trt_meta')
            if isinstance(metadata_artifact, Mapping):
                metadata_path_bound = str(metadata_artifact.get('path') or '')
                metadata_sha_bound = str(metadata_artifact.get('sha256') or '')
                if not metadata_path_bound or not metadata_sha_bound:
                    raise RuntimeError('native_split_metadata_artifact_binding_incomplete')
                contract_payload['boundary_contract']['metadata_path'] = (
                    metadata_path_bound
                )
                contract_payload['boundary_contract']['metadata_sha256'] = (
                    metadata_sha_bound
                )
            for report_name,artifact_name in (
                ('native_output_manifest','semantic_output_manifest'),
                ('native_fifo_boundary_manifest','semantic_boundary_manifest'),
            ):
                manifest=Path(str(report.get(report_name) or ''))
                if manifest.is_file():
                    contract_payload['artifacts'][artifact_name]={'path':str(manifest.resolve()),'sha256':_file_sha256(manifest),'size_bytes':int(manifest.stat().st_size)}
            contract_payload['complete'] = bool(
                input_sha and contract_payload['runner_sha256']
                and all(str(row.get('sha256') or '') for row in contract_payload['artifacts'].values())
                and prepared_input_contract.get('task') in {'classification', 'detection'}
                and prepared_input_contract.get('preprocess_mode_requested') in {'auto', 'resize', 'letterbox'}
                and prepared_input_contract.get('preprocess_mode_effective') == preprocess_mode_effective
                and prepared_input_contract.get('letterbox_pad_value_requested') == int(ns.letterbox_pad_value)
                and prepared_input_contract.get('letterbox_pad_value_effective') == int(letterbox_pad_value_effective)
                and prepared_input_contract.get('letterbox_pad_value') == int(letterbox_pad_value_effective)
                and (
                    task != 'detection'
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
                    native_row={**report,'backend':'deepx_to_trt','model_id':str(ns.model_id),'case_id':case,'task':task,'precision':str(ns.precision),'setup_id':str(ns.setup_id),'comparison_backend':'deepx'},
                    quality_binding=quality_binding,
                )
                if joined is None: raise RuntimeError(f'native_split_quality_consumer_join_failed:{join_status}')
                report['native_split_quality_consumer_attestation']=_seal_split_consumer_attestation(joined,report['native_command_contract'],ns,case=case,task=task)
                report['native_split_quality_consumer_status']=join_status
            for records_key in ('repetition_records', 'repetition_evidence'):
                records = report.get(records_key)
                if isinstance(records, list):
                    for record in records:
                        if isinstance(record, dict):
                            record['workload_contract_sha256'] = report['workload_contract_sha256']
        try: trt.close()
        except Exception: pass
    except Exception as e:
        report['ok'] = False
        report['error'] = f'{type(e).__name__}: {e}'
    if report.get('ok') and task == 'detection':
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
    print(json.dumps({'ok': bool(report.get('ok')), 'fps_makespan': report.get('fps_makespan'), 'paper_fps': report.get('paper_equivalent_fps'), 'report': str(out_path), 'error': report.get('error','')}, indent=2))
    if ns.energy_workload_only and report.get('ok'):
        print(f"__SPLITPOINT_WORK_UNITS__={int(report.get('completed_work_units') or 0)}")
        print("__SPLITPOINT_WORK_UNITS_SOURCE__=completed_work_units")
        print("__SPLITPOINT_WORK_UNITS_EXACT__=1")
    return 0 if report.get('ok') else 4

if __name__ == '__main__':
    raise SystemExit(main())
