#!/usr/bin/env python3
"""Run a dumped native Part1 boundary through Part2 ONNX under multiple layout hypotheses.

This is an offline semantic diagnostic for Native FIFO failures.  It avoids
rebuilding TensorRT engines for each hypothesis:

  native boundary dump -> candidate layout/dequant -> original Part2 ONNX via ORT -> semantic proxy

If one candidate is semantically good, rebuild the TensorRT bridge with that
layout/dequant.  If no candidate is good, the issue is likely not only the
bridge layout; investigate VStream mapping, Part2 contract, or postprocess.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    from native_boundary_dequant_sweep import _semantic_score_outputs  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f'failed to import semantic helper from native_boundary_dequant_sweep: {e}')


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def _dtype_from_manifest(man: dict[str, Any]):
    s = str(man.get('dtype') or 'uint8').lower()
    return {
        'uint8': np.uint8,
        'int8': np.int8,
        'float32': np.float32,
        'fp32': np.float32,
        'float16': np.float16,
        'fp16': np.float16,
    }.get(s, np.uint8)


def _summ(x: np.ndarray) -> dict[str, Any]:
    a = np.asarray(x)
    out: dict[str, Any] = {'shape': [int(v) for v in a.shape], 'dtype': str(a.dtype), 'size': int(a.size)}
    if not a.size:
        return out | {'finite': True}
    f = a.astype(np.float32, copy=False)
    finite = np.isfinite(f)
    out['finite'] = bool(finite.all())
    vals = f[finite]
    if vals.size:
        qs = np.quantile(vals, [0, .001, .01, .1, .5, .9, .99, .999, 1.0]).tolist()
        out.update({
            'min': float(vals.min()),
            'max': float(vals.max()),
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'quantiles': [float(v) for v in qs],
            'ratio_ge_0999': float(np.mean(vals >= 0.999)),
            'ratio_le_0': float(np.mean(vals <= 0.0)),
        })
    return out


def _find_part2_onnx(bs: Path, case: str, explicit: str = '') -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p.resolve()
    roots = [bs / case, bs]
    pats = [
        f'{case}/*part2*.onnx',
        f'{case}/*_part2_*.onnx',
        f'**/{case}/*part2*.onnx',
        f'**/*part2*{case.lstrip("b")}*.onnx',
        '**/*part2*.onnx',
    ]
    seen: set[Path] = set()
    for pat in pats:
        for r in roots:
            for x in sorted(r.glob(pat)):
                if x.is_file() and x not in seen:
                    # Prefer the original Part2 ONNX, not generated bridge ONNX.
                    name = x.name.lower()
                    if 'bridge' in name or 'dequant' in name or 'layout' in name:
                        continue
                    return x.resolve()
                    seen.add(x)
    # Fallback may pick a bridge, but record that through filename.
    for pat in pats:
        for r in roots:
            xs = sorted([x for x in r.glob(pat) if x.is_file()])
            if xs:
                return xs[0].resolve()
    raise FileNotFoundError(f'part2 ONNX not found under {bs} case={case}')


def _ort_input_dtype(inp: Any):
    t = str(getattr(inp, 'type', '') or '').lower()
    if 'uint8' in t:
        return np.uint8
    if 'int8' in t:
        return np.int8
    if 'float16' in t:
        return np.float16
    return np.float32


def _shape_from_ort(shape: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for d in shape:
        try:
            iv = int(d)
            out.append(iv if iv > 0 else -1)
        except Exception:
            out.append(-1)
    return out


def _layout_candidates(raw: np.ndarray, target_shape: tuple[int, ...]) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    """Return arrays shaped as target_shape from common accelerator memory layouts."""
    out: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    if not target_shape or any(int(x) <= 0 for x in target_shape):
        return out
    prod = int(np.prod(target_shape))
    base = np.asarray(raw)
    flat = base.reshape(-1)
    if flat.size != prod:
        return out

    def add(name: str, fn, meta: dict[str, Any] | None = None) -> None:
        try:
            y = fn()
            if tuple(y.shape) != tuple(target_shape):
                return
            if any(n == name for n, _a, _m in out):
                return
            out.append((name, y, dict(meta or {})))
        except Exception:
            return

    add('as_target_shape', lambda: flat.reshape(target_shape), {'reshape': list(target_shape), 'perm': None})
    if tuple(base.shape) == tuple(target_shape):
        out.insert(0, ('as_manifest_shape', base, {'reshape': list(base.shape), 'perm': None}))

    if len(target_shape) == 4:
        n, c, h, w = [int(x) for x in target_shape]
        # Canonical Part2 input expected as NCHW.
        add('memory_nhwc_to_nchw', lambda: flat.reshape((n, h, w, c)).transpose(0, 3, 1, 2), {'reshape': [n, h, w, c], 'perm': [0, 3, 1, 2]})
        add('memory_nwhc_to_nchw', lambda: flat.reshape((n, w, h, c)).transpose(0, 3, 2, 1), {'reshape': [n, w, h, c], 'perm': [0, 3, 2, 1]})
        add('memory_chwn_to_nchw', lambda: flat.reshape((c, h, w, n)).transpose(3, 0, 1, 2), {'reshape': [c, h, w, n], 'perm': [3, 0, 1, 2]})
        add('memory_hwcn_to_nchw', lambda: flat.reshape((h, w, c, n)).transpose(3, 2, 0, 1), {'reshape': [h, w, c, n], 'perm': [3, 2, 0, 1]})
        add('memory_ncwh_to_nchw', lambda: flat.reshape((n, c, w, h)).transpose(0, 1, 3, 2), {'reshape': [n, c, w, h], 'perm': [0, 1, 3, 2]})
        add('memory_nchw_identity', lambda: flat.reshape((n, c, h, w)), {'reshape': [n, c, h, w], 'perm': None})

        # Some exported activations are channels-last.  These candidates only
        # produce target_shape if target_shape itself is NHWC.
        n2, h2, w2, c2 = [int(x) for x in target_shape]
        add('memory_nchw_to_nhwc', lambda: flat.reshape((n2, c2, h2, w2)).transpose(0, 2, 3, 1), {'reshape': [n2, c2, h2, w2], 'perm': [0, 2, 3, 1]})
        add('memory_hwcn_to_nhwc', lambda: flat.reshape((h2, w2, c2, n2)).transpose(3, 0, 1, 2), {'reshape': [h2, w2, c2, n2], 'perm': [3, 0, 1, 2]})
    return out


def _parse_floats(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s or '').replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _prep_raw(raw: np.ndarray, dtype_s: str, scales: list[float], zps: list[float]) -> list[tuple[str, np.ndarray, dict[str, Any]]]:
    """Return dequant/raw variants before layout conversion."""
    ds = str(dtype_s or '').lower()
    if ds in ('uint8', 'int8'):
        if not scales:
            # Include identity-cast as a diagnostic plus common fits.  For real
            # semantic use the user should pass explicit values from validator.
            scales = [1.0]
        if not zps:
            zps = [0.0]
        arr = np.asarray(raw)
        out=[]
        for s in scales:
            for zp in zps:
                y = (arr.astype(np.float32) - float(zp)) * float(s)
                out.append((f'dequant_s={s:g}_zp={zp:g}', y, {'scale': float(s), 'zero_point': float(zp)}))
        return out
    return [('native_float', np.asarray(raw).astype(np.float32, copy=False), {'scale': None, 'zero_point': None})]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--boundary-manifest', required=True)
    ap.add_argument('--part2-onnx', default='')
    ap.add_argument('--reference-report', default='')
    ap.add_argument('--dequant-scale', default='', help='comma-separated scales for uint8/int8 boundary')
    ap.add_argument('--dequant-zero-point', default='', help='comma-separated zero-points for uint8/int8 boundary')
    ap.add_argument('--out', default='')
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = str(ns.case)
    man_path = Path(ns.boundary_manifest).expanduser().resolve()
    man = _read_json(man_path)
    binp = Path(str(man.get('file') or '')).expanduser()
    if not binp.is_absolute():
        binp = (man_path.parent / binp).resolve()
    raw = np.fromfile(str(binp), dtype=_dtype_from_manifest(man))
    raw_shape = [int(x) for x in (man.get('shape') or man.get('boundary_shape') or [])]
    if raw_shape and int(np.prod(raw_shape)) == raw.size:
        raw = raw.reshape(raw_shape)

    part2 = _find_part2_onnx(bs, case, ns.part2_onnx)
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f'onnxruntime required: {e}')
    sess = ort.InferenceSession(str(part2), providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_shape = _shape_from_ort(inp.shape)
    if not input_shape or any(x <= 0 for x in input_shape):
        raise RuntimeError(f'Part2 input has dynamic/unknown shape unsupported by this diagnostic: {inp.shape}')
    ort_dtype = _ort_input_dtype(inp)
    output_names = [o.name for o in sess.get_outputs()]

    scales = _parse_floats(ns.dequant_scale)
    zps = _parse_floats(ns.dequant_zero_point)
    preps = _prep_raw(raw, str(man.get('dtype') or ''), scales, zps)

    rows=[]
    for prep_name, prepped, prep_meta in preps:
        for layout_name, cand, layout_meta in _layout_candidates(prepped, tuple(input_shape)):
            row: dict[str, Any] = {
                'prep': prep_name,
                'layout': layout_name,
                'layout_meta': layout_meta,
                'prep_meta': prep_meta,
                'input_summary': _summ(cand),
            }
            try:
                feed = cand.astype(ort_dtype, copy=False)
                outs = [np.asarray(x) for x in sess.run(None, {input_name: feed})]
                row['outputs_summary'] = [_summ(o) for o in outs]
                sem = _semantic_score_outputs(outs, output_names, ns.reference_report)
                row['semantic'] = sem
                m = sem.get('match') or {}
                cam = sem.get('class_agnostic_match') or {}
                row['score_key'] = {
                    'matched': int(m.get('matched') or 0),
                    'match_ratio': float(m.get('match_ratio') or 0.0),
                    'mean_iou': float(m.get('mean_iou') or 0.0),
                    'ca_matched': int(cam.get('matched') or 0),
                    'ca_match_ratio': float(cam.get('match_ratio') or 0.0),
                    'pred_count': int(sem.get('pred_count') or 0),
                    'decode_mode': sem.get('decode_mode'),
                }
            except Exception as e:
                row['error'] = repr(e)
                row['semantic'] = {'available': False, 'reason': 'ort_run_failed'}
                row['score_key'] = {'matched': 0, 'match_ratio': 0.0, 'mean_iou': 0.0, 'ca_matched': 0, 'ca_match_ratio': 0.0, 'pred_count': 0, 'decode_mode': ''}
            rows.append(row)

    def key(r: dict[str, Any]) -> tuple[float, float, float, float, float]:
        sk = r.get('score_key') or {}
        return (
            float(sk.get('match_ratio') or 0.0),
            float(sk.get('matched') or 0.0),
            float(sk.get('ca_match_ratio') or 0.0),
            float(sk.get('mean_iou') or 0.0),
            -abs(float(sk.get('pred_count') or 0.0) - 11.0),
        )
    rows.sort(key=key, reverse=True)
    best = rows[0] if rows else None
    best_ratio = float(((best or {}).get('score_key') or {}).get('match_ratio') or 0.0)
    if best is None:
        diagnosis = 'no_candidates'
    elif best_ratio >= 0.5:
        diagnosis = 'layout_candidate_semantically_plausible'
    else:
        diagnosis = 'no_layout_candidate_semantically_plausible'

    payload = {
        'schema': 'onnx-splitpoint/native-part2-boundary-layout-sweep',
        'schema_version': 1,
        'ok': True,
        'benchmark_set': str(bs),
        'case': case,
        'boundary_manifest': str(man_path),
        'boundary_file': str(binp),
        'boundary_dtype': str(man.get('dtype') or ''),
        'boundary_shape': [int(v) for v in raw.shape],
        'part2_onnx': str(part2),
        'part2_input': {'name': input_name, 'shape': input_shape, 'type': str(getattr(inp, 'type', ''))},
        'part2_outputs': output_names,
        'reference_report': str(Path(ns.reference_report).expanduser().resolve()) if ns.reference_report else '',
        'diagnosis': diagnosis,
        'best': best,
        'rows': rows,
    }
    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent.parent / 'native_part2_boundary_layout_sweep.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    md = out.with_suffix('.md')
    lines=[
        '# Native Part2 boundary layout sweep', '',
        f'Boundary manifest: `{man_path}`',
        f'Boundary dtype/shape: `{man.get("dtype")}` `{[int(v) for v in raw.shape]}`',
        f'Part2 ONNX: `{part2}`',
        f'Part2 input: `{input_name}` shape=`{input_shape}` type=`{getattr(inp, "type", "")}`',
        f'Reference: `{ns.reference_report or ""}`',
        f'Diagnosis: `{diagnosis}`', '',
        '| rank | prep | layout | match | class-agnostic | pred | mode | top score saturation | error |',
        '|---:|---|---|---:|---:|---:|---|---:|---|',
    ]
    for i, r in enumerate(rows[:30]):
        sk = r.get('score_key') or {}
        sem = r.get('semantic') or {}
        sat = sem.get('score_saturation') or {}
        lines.append(
            f"| {i} | `{r.get('prep')}` | `{r.get('layout')}` | "
            f"{sk.get('matched')}/{(sem.get('match') or {}).get('ref_count','')} ({sk.get('match_ratio')}) | "
            f"{sk.get('ca_matched')}/{(sem.get('class_agnostic_match') or {}).get('ref_count','')} ({sk.get('ca_match_ratio')}) | "
            f"{sk.get('pred_count')} | `{sk.get('decode_mode')}` | {sat.get('ratio_ge_0999','')} | `{r.get('error','')}` |"
        )
    if best:
        lines += ['', '## Best candidate', '', f"prep=`{best.get('prep')}` layout=`{best.get('layout')}`", '', '```json', json.dumps(best.get('score_key') or {}, indent=2), '```']
    md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({'ok': True, 'out': str(out), 'summary_md': str(md), 'diagnosis': diagnosis, 'best_layout': (best or {}).get('layout'), 'best_prep': (best or {}).get('prep'), 'best_score': (best or {}).get('score_key')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
