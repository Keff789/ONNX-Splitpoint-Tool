#!/usr/bin/env python3
"""Replay a native boundary dump through the Part2 bridge ONNX and compare it to native TensorRT output.

This separates two failure classes:

1. The bridge/dequant/layout contract is still wrong.  Then ORT replay of the
   same bridge ONNX is semantically bad too.
2. The bridge ONNX is fine, but the TensorRT engine/native output path differs.
   Then ORT replay is good while native FIFO output is bad/different.

The script is diagnostic.  It does not benchmark anything and does not require
Hailo hardware; it only needs the dumped boundary, the native output manifest and
onnxruntime.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    from validate_output_dumps import load_dump  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f'failed to import validate_output_dumps.load_dump: {e}')


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def _dtype_from_manifest(man: dict[str, Any]):
    dtype_s = str(man.get('dtype') or 'uint8').lower()
    return {
        'uint8': np.uint8, 'int8': np.int8,
        'float32': np.float32, 'fp32': np.float32,
        'float16': np.float16, 'fp16': np.float16,
    }.get(dtype_s, np.uint8)


def _summ(x: np.ndarray) -> dict[str, Any]:
    a = np.asarray(x)
    f = a.astype(np.float32, copy=False) if a.size else a
    finite = np.isfinite(f) if a.size else np.asarray([True])
    out: dict[str, Any] = {
        'shape': [int(v) for v in a.shape],
        'dtype': str(a.dtype),
        'size': int(a.size),
        'finite': bool(finite.all()),
    }
    if a.size:
        xf = f[np.isfinite(f)]
        if xf.size:
            out.update({
                'min': float(np.min(xf)),
                'max': float(np.max(xf)),
                'mean': float(np.mean(xf)),
                'std': float(np.std(xf)),
                'q001': float(np.quantile(xf, 0.001)),
                'q01': float(np.quantile(xf, 0.01)),
                'q50': float(np.quantile(xf, 0.50)),
                'q99': float(np.quantile(xf, 0.99)),
                'q999': float(np.quantile(xf, 0.999)),
                'ratio_ge_0999': float(np.mean(xf >= 0.999)),
            })
    return out


def _fit_compare(a: np.ndarray, b: np.ndarray, sample: int = 500000) -> dict[str, Any]:
    x = np.asarray(a).astype(np.float64).ravel()
    y = np.asarray(b).astype(np.float64).ravel()
    if x.size != y.size:
        return {'ok': False, 'reason': 'size_mismatch', 'candidate_size': int(x.size), 'reference_size': int(y.size)}
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size > sample:
        idx = np.linspace(0, x.size - 1, sample).astype(np.int64)
        x = x[idx]; y = y[idx]
    if x.size < 2:
        return {'ok': False, 'reason': 'too_few_points'}
    d = x - y
    xm = x.mean(); ym = y.mean()
    xv = ((x - xm) ** 2).mean(); yv = ((y - ym) ** 2).mean()
    if xv > 0 and yv > 0:
        corr = float(((x - xm) * (y - ym)).mean() / math.sqrt(xv * yv))
    else:
        corr = None
    return {
        'ok': True,
        'count': int(x.size),
        'max_abs': float(np.max(np.abs(d))) if d.size else 0.0,
        'mean_abs': float(np.mean(np.abs(d))) if d.size else 0.0,
        'median_abs': float(np.median(np.abs(d))) if d.size else 0.0,
        'p99_abs': float(np.quantile(np.abs(d), 0.99)) if d.size else 0.0,
        'corr': corr,
        'allclose_1e_2_1e_3': bool(np.allclose(x, y, rtol=1e-3, atol=1e-2)),
        'allclose_5e_2_1e_2': bool(np.allclose(x, y, rtol=1e-2, atol=5e-2)),
    }


def _find_bridge_onnx(bs: Path, case: str, explicit: str = '') -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p.resolve()
    pats = [
        f'native_trt/{case}/part2/float32_layout_fp16/*_float32_layout_bridge.onnx',
        f'native_trt/{case}/part2/fp16/*_float_boundary_layout_bridge.onnx',
        f'native_trt/{case}/part2/fp32/*_float_boundary_layout_bridge.onnx',
        f'native_trt/{case}/part2/uint8_dequant_fp16/*_uint8_dequant_bridge.onnx',
        f'native_trt/{case}/part2/*/*_float32_layout_bridge.onnx',
        f'native_trt/{case}/part2/*/*_float_boundary_layout_bridge.onnx',
        f'native_trt/{case}/part2/*/*_uint8_dequant_bridge.onnx',
        f'**/native_trt/{case}/part2/*/*_float32_layout_bridge.onnx',
        f'**/native_trt/{case}/part2/*/*_float_boundary_layout_bridge.onnx',
        f'**/native_trt/{case}/part2/*/*_uint8_dequant_bridge.onnx',
        f'**/*_float32_layout_bridge.onnx',
        f'**/*_float_boundary_layout_bridge.onnx',
        f'**/*_uint8_dequant_bridge.onnx',
    ]
    for pat in pats:
        xs = sorted(bs.glob(pat))
        if xs:
            return xs[0].resolve()
    raise FileNotFoundError(f'could not locate generated Part2 bridge ONNX under {bs} case={case}')


def _find_output_manifest(boundary_manifest: Path, explicit: str = '') -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p.resolve()
    work = boundary_manifest.parent.parent if boundary_manifest.parent.name == 'native_fifo_boundary' else boundary_manifest.parent
    res = work / 'native_fifo_results.json'
    if res.is_file():
        try:
            j = _read_json(res)
            for k in ('native_fifo_output_manifest', 'output_manifest', 'outputs_manifest'):
                v = str(j.get(k) or '').strip()
                if v:
                    p = Path(v).expanduser()
                    if p.is_file():
                        return p.resolve()
                    q = work / p.name
                    if q.is_file():
                        return q.resolve()
        except Exception:
            pass
    p = work / 'native_fifo_outputs' / 'native_fifo_outputs_manifest.json'
    if p.is_file():
        return p.resolve()
    xs = sorted(work.glob('**/*outputs_manifest.json'))
    if xs:
        return xs[0].resolve()
    raise FileNotFoundError(f'could not locate native output manifest near {boundary_manifest}')


def _ort_input_dtype(inp: Any):
    t = str(getattr(inp, 'type', '') or '').lower()
    if 'uint8' in t:
        return np.uint8
    if 'float16' in t:
        return np.float16
    return np.float32


def _semantic(outputs: list[np.ndarray], names: list[str], reference_report: str) -> dict[str, Any]:
    if not reference_report:
        return {'available': False, 'reason': 'no_reference_report'}
    try:
        from native_boundary_dequant_sweep import _semantic_score_outputs  # type: ignore
        return _semantic_score_outputs(outputs, names, reference_report)
    except Exception as e:
        return {'available': False, 'reason': f'semantic_helper_failed: {e!r}'}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--boundary-manifest', required=True)
    ap.add_argument('--output-manifest', default='')
    ap.add_argument('--bridge-onnx', default='')
    ap.add_argument('--reference-report', default='')
    ap.add_argument('--out', default='')
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = ns.case
    man_path = Path(ns.boundary_manifest).expanduser().resolve()
    man = _read_json(man_path)
    binp = Path(str(man.get('file') or '')).expanduser()
    if not binp.is_absolute():
        binp = (man_path.parent / binp).resolve()
    raw_shape = [int(x) for x in (man.get('shape') or man.get('boundary_shape') or [])]
    raw = np.fromfile(str(binp), dtype=_dtype_from_manifest(man))
    if raw_shape and int(np.prod(raw_shape)) == raw.size:
        raw = raw.reshape(raw_shape)

    bridge = _find_bridge_onnx(bs, case, ns.bridge_onnx)
    out_manifest = _find_output_manifest(man_path, ns.output_manifest)

    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f'onnxruntime required for replay: {e}')
    sess = ort.InferenceSession(str(bridge), providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    input_name = inp.name
    expected_shape = [int(x) if isinstance(x, int) or str(x).isdigit() else -1 for x in inp.shape]
    feed = raw.astype(_ort_input_dtype(inp), copy=False)
    if expected_shape and all(int(x) > 0 for x in expected_shape) and list(feed.shape) != expected_shape:
        if int(np.prod(expected_shape)) != feed.size:
            raise RuntimeError(f'bridge input size mismatch: raw shape={list(feed.shape)} expected={expected_shape}')
        feed = feed.reshape(expected_shape)
    ort_outs = [np.asarray(x) for x in sess.run(None, {input_name: feed})]
    ort_names = [o.name for o in sess.get_outputs()]

    native_tensors, native_meta = load_dump(str(out_manifest))
    native_items = list(native_tensors.items())

    rows = []
    for i, oo in enumerate(ort_outs):
        oname = ort_names[i] if i < len(ort_names) else f'output{i}'
        nname = oname if oname in native_tensors else (native_items[i][0] if i < len(native_items) else '')
        narr = native_tensors.get(nname) if nname else None
        row: dict[str, Any] = {'index': i, 'ort_name': oname, 'native_name': nname, 'ort_summary': _summ(oo)}
        if narr is None:
            row['compare'] = {'ok': False, 'reason': 'native_output_missing'}
        else:
            row['native_summary'] = _summ(np.asarray(narr))
            row['compare'] = _fit_compare(np.asarray(narr), oo)
        rows.append(row)

    native_names = [n for n, _a in native_items]
    native_outs = [np.asarray(a) for _n, a in native_items]
    ort_sem = _semantic(ort_outs, ort_names, ns.reference_report)
    native_sem = _semantic(native_outs, native_names, ns.reference_report)

    # Diagnostic classification.
    same_outputs = bool(rows) and all((r.get('compare') or {}).get('ok') and ((r.get('compare') or {}).get('corr') is None or float((r.get('compare') or {}).get('corr') or 0.0) > 0.999) and float((r.get('compare') or {}).get('mean_abs') or 999.0) < 1e-2 for r in rows)
    ort_match = float(((ort_sem.get('match') or {}).get('match_ratio') if ort_sem.get('available') else 0.0) or 0.0)
    native_match = float(((native_sem.get('match') or {}).get('match_ratio') if native_sem.get('available') else 0.0) or 0.0)
    if ort_sem.get('available') and ort_match >= 0.5 and native_match < 0.5:
        diagnosis = 'trt_or_native_output_path_suspect'
    elif ort_sem.get('available') and ort_match < 0.5:
        diagnosis = 'bridge_dequant_layout_or_preprocess_still_suspect'
    elif same_outputs:
        diagnosis = 'native_matches_ort_bridge_semantics_unavailable'
    else:
        diagnosis = 'tensor_mismatch_between_native_trt_and_ort_bridge'

    payload = {
        'schema': 'onnx-splitpoint/native-part2-boundary-replay-compare',
        'schema_version': 1,
        'ok': True,
        'benchmark_set': str(bs),
        'case': case,
        'boundary_manifest': str(man_path),
        'boundary_file': str(binp),
        'bridge_onnx': str(bridge),
        'output_manifest': str(out_manifest),
        'reference_report': str(Path(ns.reference_report).expanduser().resolve()) if ns.reference_report else '',
        'bridge_input': {'name': input_name, 'shape': expected_shape, 'type': str(getattr(inp, 'type', ''))},
        'raw_summary': _summ(raw),
        'native_output_meta': native_meta,
        'rows': rows,
        'ort_bridge_semantic': ort_sem,
        'native_trt_semantic': native_sem,
        'diagnosis': diagnosis,
    }
    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent.parent / 'native_part2_boundary_replay_compare.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    md = out.with_suffix('.md')
    lines = [
        '# Native Part2 boundary replay compare', '',
        f'Boundary manifest: `{man_path}`',
        f'Bridge ONNX: `{bridge}`',
        f'Native output manifest: `{out_manifest}`',
        f'Reference: `{ns.reference_report or ""}`',
        f'Diagnosis: `{diagnosis}`', '',
        '## Semantic summary', '',
        f"ORT bridge match: `{(ort_sem.get('match') or {}).get('match_ratio') if ort_sem.get('available') else ort_sem.get('reason')}` mode=`{ort_sem.get('decode_mode','')}` pred=`{ort_sem.get('pred_count','')}`",
        f"Native TRT match: `{(native_sem.get('match') or {}).get('match_ratio') if native_sem.get('available') else native_sem.get('reason')}` mode=`{native_sem.get('decode_mode','')}` pred=`{native_sem.get('pred_count','')}`",
        '', '## Tensor compare', '',
        '| idx | ORT output | native output | shape | mean abs | max abs | corr | close |',
        '|---:|---|---|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        cmp = r.get('compare') or {}
        lines.append(f"| {r.get('index')} | `{r.get('ort_name')}` | `{r.get('native_name')}` | `{(r.get('ort_summary') or {}).get('shape')}` | {cmp.get('mean_abs','')} | {cmp.get('max_abs','')} | {cmp.get('corr','')} | {cmp.get('allclose_5e_2_1e_2','')} |")
    md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({'ok': True, 'out': str(out), 'summary_md': str(md), 'diagnosis': diagnosis, 'ort_bridge_match': ort_match if ort_sem.get('available') else None, 'native_trt_match': native_match if native_sem.get('available') else None}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
