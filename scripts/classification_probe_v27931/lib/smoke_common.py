"""Standalone bounded diagnostic support. No application/cache imports."""
from __future__ import annotations
import ast
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import types
import typing

SCHEMA = 'deepx-classification-input-probe/v27931'
MODELS = ('mobilenet_v3_large', 'resnet50', 'regnet_x_1_6gf')
DEFAULT_RUN = 'complete_set_20260907_161614'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    payload=json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    temporary=p.with_name(p.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(payload,encoding='utf-8')
    os.replace(temporary,p)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sha_token(value):
    value = str(value or '').lower().removeprefix('sha256:')
    if not re.fullmatch('[a-f0-9]{64}', value):
        raise ValueError('missing_or_invalid_expected_sha256')
    return value


def exact_artifact(candidates, expected):
    expected = sha_token(expected)
    checked = []
    seen = set()
    for value in candidates:
        if not value:
            continue
        p = Path(value).expanduser()
        if not p.is_file():
            checked.append({'path': str(p), 'status': 'missing'}); continue
        p = p.resolve(strict=True)
        if str(p) in seen:
            continue
        seen.add(str(p)); got = digest(p)
        checked.append({'path': str(p), 'sha256': got, 'size_bytes': p.stat().st_size})
        if got == expected:
            return p, checked
    raise ValueError('exact_artifact_unavailable:' + json.dumps(checked))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError('cannot_load_source:' + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BlockManagementPackage(importlib.abc.MetaPathFinder):
    """Force suite-local imports instead of activating editable GUI policies."""
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'onnx_splitpoint_tool' or fullname.startswith('onnx_splitpoint_tool.'):
            raise ModuleNotFoundError('diagnostic_uses_isolated_suite_not_installed_management_package', name=fullname)
        return None


def isolated_product(suite):
    suite = Path(suite)
    sys.path.insert(0, str(suite))
    for key in list(sys.modules):
        if key == 'splitpoint_runners' or key.startswith('splitpoint_runners.') or key == 'onnx_splitpoint_tool' or key.startswith('onnx_splitpoint_tool.'):
            del sys.modules[key]
    # These package shells only isolate imports. All called function bodies
    # below come from recorded, unmodified product source files.
    for name, directory in [('splitpoint_runners', suite/'splitpoint_runners'),
                            ('splitpoint_runners.harness', suite/'splitpoint_runners/harness')]:
        m = types.ModuleType(name); m.__path__ = [str(directory)]
        m.__file__ = str(directory/'__init__.py'); sys.modules[name] = m
    if not any(isinstance(x, BlockManagementPackage) for x in sys.meta_path):
        sys.meta_path.insert(0, BlockManagementPackage())
    return load_module('diagnostic_product_benchmark', suite/'benchmark_suite.py')


def extract_functions(path, names, namespace=None, method=None):
    """Execute unchanged AST function bodies, not module CLI/import side effects.

    Recursively include same-file function dependencies. Constants/imports are
    supplied explicitly. The whole original source is archived alongside it.
    """
    p = Path(path); tree = ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)
    selected = set(names); extra = []
    if method:
        cname, mname = method
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cname]
        if len(classes) != 1:
            raise ValueError('product_class_not_unique:' + cname)
        extra = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == mname]
        if len(extra) != 1:
            raise ValueError('product_method_not_unique:' + mname)
    for n in selected:
        if n not in definitions:
            raise ValueError('product_function_missing:' + n)
    while True:
        nodes = extra + [n for k in selected for n in definitions[k]]
        refs = {n.id for parent in nodes for n in ast.walk(parent) if isinstance(n, ast.Name)}
        newer = selected | (refs & set(definitions))
        if newer == selected:
            break
        selected = newer
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in selected]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    mod = ast.fix_missing_locations(ast.Module(body=[future, *nodes, *extra], type_ignores=[]))
    env = {'Path': Path, 'os': os, 'sys': sys, 'json': json, 'hashlib': hashlib,
           're': re, '__file__': str(p), '__name__': 'product_function_observation'}
    env.update({k:v for k,v in vars(typing).items() if not k.startswith('_') and k not in env}); env.update(namespace or {})
    exec(compile(mod, str(p), 'exec'), env)
    return env


def run_bounded(argv, log_path, timeout, env=None, cwd=None, heartbeat=None):
    """Timeout a newly owned process group; never signal unrelated jobs."""
    started = time.monotonic(); timed_out = False
    log_path = Path(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 cwd=cwd, start_new_session=True)
        next_notice = started + 15
        try:
            while child.poll() is None:
                now = time.monotonic()
                if now - started >= timeout:
                    timed_out = True; break
                if heartbeat and now >= next_notice:
                    print(f'{heartbeat}: noch aktiv ({int(now-started)} s)', flush=True); next_notice = now + 15
                time.sleep(0.1)
        except BaseException:
            terminate_group(child)
            raise
        if timed_out:
            terminate_group(child)
        else:
            child.wait()
    return {'returncode': child.returncode, 'timed_out': timed_out,
            'elapsed_s': round(time.monotonic()-started, 3), 'log': str(log_path)}


def terminate_group(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def clean_env(work):
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME'):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg',
               MPLCONFIGDIR=str(Path(work)/'mpl'),
               ONNX_SPLITPOINT_ARTIFACT_POLICY='cache_verify_only', ORT_DISABLE_TELEMETRY='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    return env


def stats(array):
    import numpy as np
    a = np.asarray(array); finite = np.isfinite(a); vals = a[finite]
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'size': int(a.size),
            'strides':list(a.strides),'c_contiguous':bool(a.flags.c_contiguous),
            'finite_count': int(finite.sum()), 'nan_count': int(np.isnan(a).sum()),
            'inf_count': int(np.isinf(a).sum()),
            'min': float(vals.min()) if vals.size else None,
            'max': float(vals.max()) if vals.size else None,
            'std':float(vals.astype(np.float64).std()) if vals.size else None,
            'zero_fraction':float(np.count_nonzero(a==0)/a.size) if a.size else None,
            'saturation_fraction':float(np.count_nonzero(a==255)/a.size) if a.size and a.dtype==np.uint8 else None,
            'mean': float(vals.astype(np.float64).mean()) if vals.size else None,
            'sha256_bytes': hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def compare(left, right):
    import numpy as np
    a, b = np.asarray(left), np.asarray(right)
    out = {'left_shape': list(a.shape), 'right_shape': list(b.shape),
           'left_dtype': str(a.dtype), 'right_dtype': str(b.dtype), 'same_shape': a.shape == b.shape}
    if a.shape != b.shape:
        return out
    out.update(exact_equal=bool(np.array_equal(a,b)), dtype_equal=a.dtype == b.dtype)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        out['finite'] = False; return out
    af, bf = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    delta = np.abs(af-bf); denom = float(np.linalg.norm(af)*np.linalg.norm(bf))
    out.update(finite=True, max_abs=float(delta.max()) if delta.size else 0.0,
               mean_abs=float(delta.mean()) if delta.size else 0.0,
               changed_elements=int(np.count_nonzero(delta)), mismatch_positions=np.argwhere(np.abs(a.astype(np.float64)-b.astype(np.float64))>0)[:8].tolist(),
               cosine=float(np.dot(af,bf)/denom) if denom else None)
    return out


def logits_record(outputs, names, label, harness=None):
    import numpy as np
    arrays = [np.asarray(x) for x in outputs]
    if len(arrays) != 1 or len(names) != 1:
        raise ValueError('ambiguous_output_count:' + str(len(arrays)))
    a = arrays[0]
    if not np.issubdtype(a.dtype,np.floating):
        raise ValueError('classification_nonfloating_output_requires_recorded_dequantization_contract')
    # Never flatten a multi-batch or multi-head tensor into pseudo classes.
    if a.shape not in ((1000,), (1,1000), (1,1,1000), (1,1000,1,1)):
        raise ValueError('classification_1000_single_sample_required:' + str(a.shape))
    vec = a.reshape(1000)
    if not np.isfinite(vec).all():
        raise ValueError('nonfinite_classification_output')
    order = np.argsort(-vec.astype(np.float64), kind='stable')[:5]
    top = [int(i) for i in order]
    out = {'output_name': str(names[0]), 'raw': stats(a), 'top1': top[0], 'top5': top,
           'top5_raw_values': [float(vec[i]) for i in top], 'label_id': label,
           'top1_correct': top[0] == label if label is not None else None,
           'top5_correct': label in top if label is not None else None,
           'probability_like': bool(np.all(vec >= 0) and np.all(vec <= 1) and np.isclose(vec.sum(),1,atol=.002)),
           'top5_tie_present': len(set(float(vec[i]) for i in top)) != 5}
    if harness:
        p = harness.postprocess({str(names[0]): a}, {})
        payload = p.json if hasattr(p,'json') else p['json']
        ids = [int(x['id']) for x in payload['topk']]
        out.update(product_harness=payload, harness_top1_match=ids[0] == top[0],
                   harness_top5_set_match=set(ids) == set(top), harness_top5_order_match=ids == top)
    return out


MAX_MODEL_PAYLOAD_BYTES=128*1024*1024
_WRITTEN_PAYLOAD_BYTES={}

def save_tensors(path, **arrays):
    """Bound uncompressed arrays before writing; copy runtime buffers immediately."""
    import numpy as np
    copied={str(k):np.asarray(v).copy() for k,v in arrays.items()}
    if any(v.dtype.hasobject for v in copied.values()):raise ValueError('pickle_tensor_forbidden')
    root=Path(path).parent
    if root.name=='trt_control':root=root.parent
    is_split=Path(path).name.startswith('split_control_')
    existing=sum(p.stat().st_size for p in (root.glob('split_control_*.npz') if is_split else root.rglob('*')) if p.is_file() and '_stage' not in p.relative_to(root).parts)
    budget_key=str(root/'split_control') if is_split else str(root)
    raw_bytes=sum(v.nbytes for v in copied.values())
    previous=_WRITTEN_PAYLOAD_BYTES.get(budget_key,existing)
    # CPU and remote each reserve 40 MiB; selected images are capped at 32 MiB.
    # The remaining 16 MiB covers bounded metadata/source/log collection.
    if previous+raw_bytes>min(MAX_MODEL_PAYLOAD_BYTES,(8 if is_split else 40)*1024*1024):raise ValueError('model_payload_budget_exceeded')
    np.savez_compressed(path,**copied)
    _WRITTEN_PAYLOAD_BYTES[budget_key]=previous+raw_bytes
