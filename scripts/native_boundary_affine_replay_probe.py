#!/usr/bin/env python3
"""Replay a native boundary through Part2 ONNX after affine correction probes.

This diagnostic answers a narrower question than the raw layout sweep:

  1. Does ORT Part1 activation -> original Part2 ONNX reproduce the semantic reference?
     If not, the Part2/reference/postprocess contract is suspect.
  2. Does native boundary -> layout -> Part2 fail, but native boundary after a
     fitted global/per-channel affine correction -> Part2 succeed?
     If yes, the native boundary is the same tensor up to numeric calibration.
  3. If even affine-corrected boundary fails while the ORT Part1 control passes,
     the native Hailo/accelerator boundary is not sufficient for the Part2
     ONNX contract through a simple layout/affine transform.

It intentionally uses the exact preprocessed input dump from the native boundary
manifest when available, so the positive ORT Part1 control and the native
boundary come from the same image tensor.
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
    from native_boundary_activation_compare import (  # type: ignore
        _find_part1_onnx,
        _fit_affine,
        _input_dump_feed_from_manifest,
        _onnx_io,
        _preprocess,
        _find_image,
        _image_from_boundary_manifest,
        _image_from_report,
        _read_json,
        _summ,
    )
    from native_boundary_interface_contract_validator import _contract_candidate_layouts, _channel_axis  # type: ignore
    from native_boundary_dequant_sweep import _semantic_score_outputs  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f'failed to import diagnostic helpers: {e}')


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
    for pat in pats:
        for r in roots:
            for x in sorted(r.glob(pat)):
                if x.is_file():
                    name = x.name.lower()
                    if 'bridge' in name or 'dequant' in name or 'layout' in name:
                        continue
                    return x.resolve()
    for pat in pats:
        for r in roots:
            xs = sorted([x for x in r.glob(pat) if x.is_file()])
            if xs:
                return xs[0].resolve()
    raise FileNotFoundError(f'part2 ONNX not found under {bs} case={case}')


def _shape_from_ort(shape: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for d in shape:
        try:
            iv = int(d)
            out.append(iv if iv > 0 else -1)
        except Exception:
            out.append(-1)
    return out


def _ort_input_dtype(inp: Any):
    t = str(getattr(inp, 'type', '') or '').lower()
    if 'uint8' in t:
        return np.uint8
    if 'int8' in t:
        return np.int8
    if 'float16' in t:
        return np.float16
    return np.float32


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _move_channel_first(x: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        return x.reshape(x.shape[0], x.shape[1], -1)[0]
    if axis == 3:
        return np.moveaxis(x, 3, 1).reshape(x.shape[0], x.shape[3], -1)[0]
    raise ValueError(f'unsupported channel axis {axis}')


def _apply_per_channel_affine(raw: np.ndarray, ref: np.ndarray, axis: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit and apply ref ~= a*raw + b per channel, preserving original shape."""
    x = np.asarray(raw, dtype=np.float32)
    y = np.asarray(ref, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f'shape mismatch raw={x.shape} ref={y.shape}')
    out = np.empty_like(x, dtype=np.float32)
    fits: list[dict[str, Any]] = []
    if axis == 1:
        cnum = x.shape[1]
        for c in range(cnum):
            f = _fit_affine(x[:, c, :, :], y[:, c, :, :], sample=20000)
            a = float(f.get('scale_fit') if f.get('ok') else 1.0)
            b = float(f.get('bias_fit') if f.get('ok') else 0.0)
            out[:, c, :, :] = x[:, c, :, :] * a + b
            fits.append(f)
    elif axis == 3:
        cnum = x.shape[3]
        for c in range(cnum):
            f = _fit_affine(x[:, :, :, c], y[:, :, :, c], sample=20000)
            a = float(f.get('scale_fit') if f.get('ok') else 1.0)
            b = float(f.get('bias_fit') if f.get('ok') else 0.0)
            out[:, :, :, c] = x[:, :, :, c] * a + b
            fits.append(f)
    else:
        raise ValueError(f'unsupported channel axis {axis}')

    def vals(key: str) -> np.ndarray:
        arr = [_safe_float(f.get(key)) for f in fits if f.get('ok')]
        arr = [v for v in arr if v is not None]
        return np.asarray(arr, dtype=np.float64)
    corrs = vals('corr'); r2s = vals('r2'); scales = vals('scale_fit'); biases = vals('bias_fit'); maes = vals('mae')
    meta = {
        'ok': bool(corrs.size),
        'channel_axis': int(axis),
        'channel_count': int(len(fits)),
        'valid_fit_count': int(corrs.size),
        'corr_median': float(np.median(corrs)) if corrs.size else None,
        'r2_median': float(np.median(r2s)) if r2s.size else None,
        'scale_median': float(np.median(scales)) if scales.size else None,
        'scale_p10': float(np.quantile(scales, 0.10)) if scales.size else None,
        'scale_p90': float(np.quantile(scales, 0.90)) if scales.size else None,
        'bias_median': float(np.median(biases)) if biases.size else None,
        'bias_p10': float(np.quantile(biases, 0.10)) if biases.size else None,
        'bias_p90': float(np.quantile(biases, 0.90)) if biases.size else None,
        'mae_median': float(np.median(maes)) if maes.size else None,
    }
    return out, meta


def _semantic(outputs: list[np.ndarray], names: list[str], reference_report: str) -> dict[str, Any]:
    if not reference_report:
        return {'available': False, 'reason': 'no_reference_report'}
    try:
        return _semantic_score_outputs(outputs, names, reference_report)
    except Exception as e:
        return {'available': False, 'reason': f'semantic_helper_failed: {e!r}'}


def _score_key(sem: dict[str, Any]) -> dict[str, Any]:
    m = sem.get('match') or {}
    cam = sem.get('class_agnostic_match') or {}
    return {
        'matched': int(m.get('matched') or 0),
        'match_ratio': float(m.get('match_ratio') or 0.0),
        'mean_iou': float(m.get('mean_iou') or 0.0),
        'ca_matched': int(cam.get('matched') or 0),
        'ca_match_ratio': float(cam.get('match_ratio') or 0.0),
        'pred_count': int(sem.get('pred_count') or 0),
        'decode_mode': sem.get('decode_mode'),
    }


def _run_part2(sess: Any, input_name: str, output_names: list[str], ort_dtype: Any, feed: np.ndarray, reference_report: str) -> dict[str, Any]:
    row: dict[str, Any] = {'input_summary': _summ(feed)}
    try:
        outs = [np.asarray(x) for x in sess.run(None, {input_name: feed.astype(ort_dtype, copy=False)})]
        sem = _semantic(outs, output_names, reference_report)
        row['outputs_summary'] = [_summ(o) for o in outs]
        row['semantic'] = sem
        row['score_key'] = _score_key(sem)
    except Exception as e:
        row['error'] = repr(e)
        row['semantic'] = {'available': False, 'reason': 'ort_run_failed'}
        row['score_key'] = {'matched': 0, 'match_ratio': 0.0, 'mean_iou': 0.0, 'ca_matched': 0, 'ca_match_ratio': 0.0, 'pred_count': 0, 'decode_mode': ''}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--boundary-manifest', required=True)
    ap.add_argument('--part2-onnx', default='')
    ap.add_argument('--reference-report', default='')
    ap.add_argument('--image-scale', default='native')
    ap.add_argument('--out', default='')
    ap.add_argument('--include-reference-control', action='store_true', default=True)
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

    part1 = _find_part1_onnx(bs, case)
    part1_inputs, _part1_outputs_meta = _onnx_io(part1)
    if not part1_inputs:
        raise RuntimeError(f'no inputs in {part1}')
    p1_input_name, p1_input_shape = part1_inputs[0]

    # Resolve image only for fallback preprocessing if the manifest does not have input_dump.
    ref_report = Path(ns.reference_report).expanduser().resolve() if ns.reference_report else None
    boundary_image = _image_from_boundary_manifest(man)
    ref_image = _image_from_report(ref_report)
    img_name = boundary_image or ref_image
    img_path = _find_image(bs, img_name) if img_name else None

    feed = _input_dump_feed_from_manifest(man, p1_input_shape, ns.image_scale)
    feed_source = 'boundary_manifest_input_dump' if feed is not None else f'image_preprocess:{ns.image_scale}'
    if feed is None:
        if img_path is None:
            raise FileNotFoundError('could not resolve input image and no input_dump is available in boundary manifest')
        feed = _preprocess(img_path, p1_input_shape, ns.image_scale)

    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f'onnxruntime required: {e}')

    p1_sess = ort.InferenceSession(str(part1), providers=['CPUExecutionProvider'])
    ref_outs = [np.asarray(x) for x in p1_sess.run(None, {p1_input_name: feed})]
    ref_out_names = [o.name for o in p1_sess.get_outputs()]

    part2 = _find_part2_onnx(bs, case, ns.part2_onnx)
    p2_sess = ort.InferenceSession(str(part2), providers=['CPUExecutionProvider'])
    p2_in = p2_sess.get_inputs()[0]
    p2_input_name = p2_in.name
    p2_shape = _shape_from_ort(p2_in.shape)
    p2_dtype = _ort_input_dtype(p2_in)
    p2_out_names = [o.name for o in p2_sess.get_outputs()]

    if not p2_shape or any(int(x) <= 0 for x in p2_shape):
        raise RuntimeError(f'Part2 input has dynamic/unknown shape unsupported by this diagnostic: {p2_in.shape}')

    # Choose ORT Part1 output that matches Part2 input by name first, then element count.
    ref_candidates: list[tuple[str, np.ndarray]] = []
    for name, arr in zip(ref_out_names, ref_outs):
        if name == p2_input_name:
            ref_candidates.insert(0, (name, arr))
        elif np.asarray(arr).size == int(np.prod(p2_shape)):
            ref_candidates.append((name, arr))
    if not ref_candidates:
        raise RuntimeError(f'No Part1 output matches Part2 input {p2_input_name} shape={p2_shape}')
    ref_name, ref_activation = ref_candidates[0]
    if list(ref_activation.shape) != p2_shape and ref_activation.size == int(np.prod(p2_shape)):
        ref_activation = ref_activation.reshape(p2_shape)

    rows: list[dict[str, Any]] = []

    # Positive control: canonical ORT Part1 activation into original Part2 ONNX.
    control = _run_part2(p2_sess, p2_input_name, p2_out_names, p2_dtype, ref_activation.astype(np.float32), ns.reference_report)
    control.update({
        'kind': 'ort_part1_reference_control',
        'layout': 'reference_activation',
        'transform': 'none',
        'fit': {'ok': True, 'source': 'ORT Part1 output'},
    })
    rows.append(control)

    axis = _channel_axis(tuple(ref_activation.shape))
    for layout_name, raw_layout in _contract_candidate_layouts(raw, tuple(ref_activation.shape)):
        raw_f = np.asarray(raw_layout, dtype=np.float32)
        # Identity/layout only.
        r = _run_part2(p2_sess, p2_input_name, p2_out_names, p2_dtype, raw_f, ns.reference_report)
        r.update({'kind': 'native_boundary', 'layout': layout_name, 'transform': 'identity', 'fit': _fit_affine(raw_f, ref_activation)})
        rows.append(r)

        # Global affine fitted to ORT Part1 for the exact same preprocessed input.
        gf = _fit_affine(raw_f, ref_activation)
        if gf.get('ok'):
            a = float(gf.get('scale_fit') or 1.0)
            b = float(gf.get('bias_fit') or 0.0)
            corrected = raw_f * a + b
            rr = _run_part2(p2_sess, p2_input_name, p2_out_names, p2_dtype, corrected.astype(np.float32), ns.reference_report)
            rr.update({'kind': 'native_boundary', 'layout': layout_name, 'transform': 'global_affine_fit_to_ort_part1', 'fit': gf})
            rows.append(rr)

        # Per-channel affine fitted to ORT Part1.
        if axis is not None:
            try:
                pc_corr, pc_meta = _apply_per_channel_affine(raw_f, ref_activation, axis)
                rr = _run_part2(p2_sess, p2_input_name, p2_out_names, p2_dtype, pc_corr.astype(np.float32), ns.reference_report)
                rr.update({'kind': 'native_boundary', 'layout': layout_name, 'transform': 'per_channel_affine_fit_to_ort_part1', 'fit': pc_meta})
                rows.append(rr)
            except Exception as e:
                rows.append({
                    'kind': 'native_boundary',
                    'layout': layout_name,
                    'transform': 'per_channel_affine_fit_to_ort_part1',
                    'error': repr(e),
                    'semantic': {'available': False, 'reason': 'per_channel_fit_failed'},
                    'score_key': {'matched': 0, 'match_ratio': 0.0, 'mean_iou': 0.0, 'ca_matched': 0, 'ca_match_ratio': 0.0, 'pred_count': 0, 'decode_mode': ''},
                })

    def sort_key(r: dict[str, Any]) -> tuple[float, float, float, float, float]:
        sk = r.get('score_key') or {}
        return (
            float(sk.get('match_ratio') or 0.0),
            float(sk.get('matched') or 0.0),
            float(sk.get('ca_match_ratio') or 0.0),
            float(sk.get('mean_iou') or 0.0),
            -abs(float(sk.get('pred_count') or 0.0) - 11.0),
        )

    sorted_rows = sorted(rows, key=sort_key, reverse=True)
    control_ratio = float((control.get('score_key') or {}).get('match_ratio') or 0.0)
    best = sorted_rows[0] if sorted_rows else None
    best_boundary = None
    for r in sorted_rows:
        if r.get('kind') == 'native_boundary':
            best_boundary = r
            break
    best_boundary_ratio = float(((best_boundary or {}).get('score_key') or {}).get('match_ratio') or 0.0)
    best_boundary_transform = str((best_boundary or {}).get('transform') or '')

    if control_ratio < 0.5:
        diagnosis = 'part2_or_reference_contract_suspect'
    elif best_boundary_ratio >= 0.5 and best_boundary_transform in {'identity'}:
        diagnosis = 'native_boundary_contract_semantically_ok_offline'
    elif best_boundary_ratio >= 0.5 and 'affine' in best_boundary_transform:
        diagnosis = 'native_boundary_affine_fixable'
    else:
        diagnosis = 'native_boundary_not_part2_contract_or_non_affine'

    payload = {
        'schema': 'onnx-splitpoint/native-boundary-affine-replay-probe',
        'schema_version': 1,
        'ok': True,
        'diagnosis': diagnosis,
        'benchmark_set': str(bs),
        'case': case,
        'part1_onnx': str(part1),
        'part2_onnx': str(part2),
        'part2_input': {'name': p2_input_name, 'shape': p2_shape, 'type': str(getattr(p2_in, 'type', ''))},
        'part2_outputs': p2_out_names,
        'reference_report': str(Path(ns.reference_report).expanduser().resolve()) if ns.reference_report else '',
        'boundary_manifest': str(man_path),
        'boundary_file': str(binp),
        'boundary_dtype': str(man.get('dtype') or ''),
        'boundary_shape': [int(v) for v in raw.shape],
        'feed_source': feed_source,
        'boundary_image': boundary_image,
        'reference_image': ref_image,
        'ort_part1_output_used': {'name': ref_name, 'shape': [int(v) for v in ref_activation.shape], 'summary': _summ(ref_activation)},
        'control': control,
        'best': best,
        'best_boundary': best_boundary,
        'rows': sorted_rows,
    }

    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent.parent / 'native_boundary_affine_replay_probe.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    md = out.with_suffix('.md')
    lines = [
        '# Native boundary affine replay probe', '',
        f'Diagnosis: `{diagnosis}`',
        f'Boundary manifest: `{man_path}`',
        f'Boundary dtype/shape: `{man.get("dtype")}` `{[int(v) for v in raw.shape]}`',
        f'Feed source: `{feed_source}`',
        f'Part1 ONNX: `{part1}`',
        f'Part2 ONNX: `{part2}`',
        f'Part2 input: `{p2_input_name}` shape=`{p2_shape}` type=`{getattr(p2_in, "type", "")}`',
        f'ORT Part1 output used: `{ref_name}` shape=`{[int(v) for v in ref_activation.shape]}`',
        f'Reference: `{ns.reference_report or ""}`', '',
        '## Interpretation guardrail', '',
        f'ORT Part1 reference-control match: `{control_ratio}`',
        f'Best native-boundary match: `{best_boundary_ratio}` transform=`{best_boundary_transform}` layout=`{(best_boundary or {}).get("layout")}`', '',
        '| rank | kind | layout | transform | match | class-agnostic | pred | mode | fit r2/corr | fit scale/bias | error |',
        '|---:|---|---|---|---:|---:|---:|---|---:|---:|---|',
    ]
    for i, r in enumerate(sorted_rows[:40]):
        sem = r.get('semantic') or {}
        sk = r.get('score_key') or {}
        fit = r.get('fit') or {}
        m = sem.get('match') or {}
        cam = sem.get('class_agnostic_match') or {}
        fit_r2 = fit.get('r2') if 'r2' in fit else fit.get('r2_median')
        fit_corr = fit.get('corr') if 'corr' in fit else fit.get('corr_median')
        fit_scale = fit.get('scale_fit') if 'scale_fit' in fit else fit.get('scale_median')
        fit_bias = fit.get('bias_fit') if 'bias_fit' in fit else fit.get('bias_median')
        lines.append(
            f"| {i} | `{r.get('kind')}` | `{r.get('layout')}` | `{r.get('transform')}` | "
            f"{sk.get('matched')}/{m.get('ref_count','')} ({sk.get('match_ratio')}) | "
            f"{sk.get('ca_matched')}/{cam.get('ref_count','')} ({sk.get('ca_match_ratio')}) | "
            f"{sk.get('pred_count')} | `{sk.get('decode_mode')}` | "
            f"{fit_r2}/{fit_corr} | {fit_scale}/{fit_bias} | `{r.get('error','')}` |"
        )
    md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({
        'ok': True,
        'out': str(out),
        'summary_md': str(md),
        'diagnosis': diagnosis,
        'control_match': control_ratio,
        'best_boundary_match': best_boundary_ratio,
        'best_boundary_layout': (best_boundary or {}).get('layout'),
        'best_boundary_transform': (best_boundary or {}).get('transform'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
