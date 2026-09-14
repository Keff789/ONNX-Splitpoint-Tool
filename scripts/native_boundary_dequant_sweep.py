#!/usr/bin/env python3
"""Offline boundary dequantization sweep for native split diagnostics.

Runs Part2 ONNX with a raw uint8 boundary dump using several candidate
scale/zero-point pairs. Besides tensor statistics, this version can optionally
rank candidates against an ORT/full detection reference using a robust
validation_report parser plus the same lightweight YOLO decode/NMS proxy used by native validation. This is still diagnostic: it is
intended to identify plausible dequantization parameters before rebuilding many
TensorRT engines.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding='utf-8'))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_sha256(value: Any) -> str:
    token = str(value or '').strip().lower()
    if token.startswith('sha256:'):
        token = token[7:]
    return token if len(token) == 64 and all(c in '0123456789abcdef' for c in token) else ''


def _resolve_reference_image(
    *, report: Path | None, image_name: str | None, benchmark_set: Path,
) -> Path | None:
    if not image_name:
        return None
    raw = Path(str(image_name)).expanduser()
    candidates = [raw]
    if report is not None:
        candidates.append(report.parent / raw.name)
    candidates.extend(sorted(benchmark_set.rglob(raw.name)))
    observed: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file() and resolved not in observed:
            observed.append(resolved)
    if not observed:
        return None
    digests = {_sha256_file(path) for path in observed}
    return observed[0] if len(digests) == 1 else None


def _same_input_identity(
    boundary_image: Any, boundary_sha256: Any,
    reference_image: Any, reference_sha256: Any,
) -> bool:
    return bool(
        boundary_image and reference_image
        and Path(str(boundary_image)).name == Path(str(reference_image)).name
        and _normalize_sha256(boundary_sha256)
        and _normalize_sha256(boundary_sha256)
        == _normalize_sha256(reference_sha256)
    )


def _find_part2_onnx(bs: Path, case: str) -> Path:
    c = bs / case
    pats = [f"*_part2_*{case.lstrip('b')}*.onnx", "*part2*.onnx"]
    for pat in pats:
        xs = sorted(c.glob(pat))
        if xs:
            return xs[0]
    raise FileNotFoundError(f"part2 ONNX not found under {c}")


def _input_shape_from_onnx(path: Path):
    import onnx
    model = onnx.load(str(path))
    inp = model.graph.input[0]
    dims=[]
    for d in inp.type.tensor_type.shape.dim:
        dims.append(int(d.dim_value) if d.dim_value else -1)
    return inp.name, dims


def _candidate_scales(bs: Path, case: str, user_scales: list[float]) -> list[dict[str, Any]]:
    out=[]
    for s in user_scales:
        out.append({'source':'user','scale':float(s),'zero_point':0.0})
    # common diagnostic values.  v59dd: include zero-point grid.
    # Hailo/DeepX raw activation tensors often reserve the raw minimum value
    # as an encoded zero (e.g. raw min=1).  Testing only zp=0 can make the
    # dequant bridge look falsely hopeless.
    common_scales = [1/255, 1/128, 1/64, 1/32, 1/16, 0.05, 0.1, 0.2, 0.5, 1.0]
    common_zps = [0.0, 1.0, 2.0, 5.0, 10.0, 16.0, 32.0, 64.0, 128.0]
    for s in common_scales:
        for zp in common_zps:
            out.append({'source':'common_grid','scale':float(s),'zero_point':float(zp)})
    # activation calibration min/max heuristic
    stats = bs / case / 'activation_calibration' / 'stats.json'
    if stats.exists():
        try:
            js = _read_json(stats)
            vals=[]
            def rec(x):
                if isinstance(x, dict):
                    mn=x.get('min'); mx=x.get('max')
                    if isinstance(mn,(int,float)) and isinstance(mx,(int,float)) and mx>mn:
                        vals.append((float(mn),float(mx)))
                    for v in x.values(): rec(v)
                elif isinstance(x, list):
                    for v in x: rec(v)
            rec(js)
            for mn,mx in vals[:20]:
                sc=(mx-mn)/255.0
                out.append({'source':'activation_minmax','scale':sc,'zero_point':round(-mn/sc) if sc else 0.0,'min':mn,'max':mx})
        except Exception as e:
            out.append({'source':'activation_minmax_error','error':repr(e),'scale':1/255,'zero_point':0.0})
    # de-duplicate
    seen=set(); ded=[]
    for c in out:
        key=(round(float(c.get('scale',0)),10), round(float(c.get('zero_point',0)),4))
        if key in seen: continue
        seen.add(key); ded.append(c)
    return ded


def _summ(x: np.ndarray):
    x = np.asarray(x)
    finite = np.isfinite(x)
    xf = x[finite]
    if xf.size == 0:
        return {'shape':list(x.shape),'dtype':str(x.dtype),'finite':False}
    qs=np.quantile(xf, [0, .001, .01, .1, .5, .9, .99, .999, 1.0]).tolist()
    return {
        'shape':list(x.shape), 'dtype':str(x.dtype), 'finite':bool(finite.all()),
        'min':float(xf.min()), 'max':float(xf.max()), 'mean':float(xf.mean()),
        'quantiles':qs, 'score_saturation_hint':float(np.mean(xf>=0.999)) if xf.size else None
    }


def _local_reference_detections(report: Path | str | None) -> tuple[list[dict[str, Any]], str | None, str]:
    """Robust fallback parser for validation_report.json detection references.

    v59dc: Some remote environments imported an older validator helper and
    reported reference_has_no_detections even though validation_report.json
    contained detections under viz/full/json/detections.  Keep the parser local
    to the sweep so semantic ranking does not depend on an imported helper
    version.
    """
    if not report:
        return [], None, 'no_reference_report'
    try:
        p = Path(report)
        if not p.is_file():
            return [], None, 'reference_report_missing'
        j = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        return [], None, f'reference_report_parse_failed:{e!r}'

    def norm_det(d: Any) -> dict[str, Any] | None:
        if not isinstance(d, dict):
            return None
        # Already normalized format used by visual reports.
        if all(k in d for k in ('x1','y1','x2','y2')):
            try:
                cid = d.get('class_id', d.get('cls', d.get('class', -1)))
                return {
                    'x1': float(d['x1']), 'y1': float(d['y1']),
                    'x2': float(d['x2']), 'y2': float(d['y2']),
                    'score': float(d.get('score', 1.0)),
                    'class_id': int(cid) if str(cid).strip() not in ('', 'None') else -1,
                    'label': d.get('label', ''),
                }
            except Exception:
                return None
        # COCO-ish bbox format: x,y,w,h
        bb = d.get('bbox') or d.get('box')
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            try:
                x, y, w, h = map(float, bb[:4])
                cid = d.get('class_id', d.get('category_id', d.get('cls', -1)))
                return {
                    'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h,
                    'score': float(d.get('score', 1.0)),
                    'class_id': int(cid) if str(cid).strip() not in ('', 'None') else -1,
                    'label': d.get('label', ''),
                }
            except Exception:
                return None
        return None

    candidates: list[tuple[str, Any, str | None]] = []
    viz = j.get('viz') if isinstance(j, dict) else {}
    if isinstance(viz, dict):
        for variant in ('composed', 'full', 'part2', 'part1'):
            node = (viz.get(variant) or {})
            js = (node.get('json') or {}) if isinstance(node, dict) else {}
            candidates.append((f'viz/{variant}/json/detections', js.get('detections'), ((js.get('provenance') or {}).get('image') if isinstance(js, dict) else None)))
        # Some reports use viz.variants.<variant> directly.
        variants = viz.get('variants') if isinstance(viz, dict) else {}
        if isinstance(variants, dict):
            for variant in ('composed', 'full', 'part2', 'part1'):
                js = variants.get(variant) or {}
                if isinstance(js, dict):
                    candidates.append((f'viz/variants/{variant}/detections', js.get('detections'), ((js.get('provenance') or {}).get('image'))))

    # Dataset validation image entries sometimes contain predictions/detections.
    vd = j.get('validation_dataset') if isinstance(j, dict) else {}
    if isinstance(vd, dict):
        variants = vd.get('variants') or {}
        if isinstance(variants, dict):
            for variant in ('composed', 'full', 'part2', 'part1'):
                v = variants.get(variant) or {}
                imgs = v.get('images') if isinstance(v, dict) else []
                if isinstance(imgs, list):
                    for im in imgs[:5]:
                        if isinstance(im, dict):
                            dets = im.get('detections') or im.get('predictions') or im.get('preds')
                            candidates.append((f'validation_dataset/{variant}/images', dets, im.get('image')))

    for source, dets_raw, image in candidates:
        if not isinstance(dets_raw, list) or not dets_raw:
            continue
        dets = [x for x in (norm_det(d) for d in dets_raw) if x is not None]
        if dets:
            return dets, image, source
    return [], None, 'no_detections_in_known_paths'


def _semantic_score_outputs(outputs: list[np.ndarray], output_names: list[str], ref_report: str | None) -> dict[str, Any]:
    """Best-effort detection semantic proxy for one dequant candidate.

    Uses native_producer_validate_visualize helpers if available. If no
    reference is supplied or helpers are unavailable, returns unavailable rather
    than failing the sweep.
    """
    if not ref_report:
        return {'available': False, 'reason': 'no_reference_report'}
    try:
        from native_producer_validate_visualize import (
            _reference_detections, _choose_detection_candidate, _match_detections,
            _match_detections_class_agnostic, _score_saturation,
        )
    except Exception as e:
        return {'available': False, 'reason': f'validator_import_failed: {e!r}'}
    try:
        # Prefer the local parser to avoid stale imported helper versions.
        ref_dets, image_name, ref_source = _local_reference_detections(Path(ref_report))
        if not ref_dets:
            try:
                ref_dets, image_name = _reference_detections(Path(ref_report))
                ref_source = 'imported_helper' if ref_dets else ref_source
            except Exception:
                pass
        if not ref_dets:
            return {'available': False, 'reason': 'reference_has_no_detections', 'reference_image': image_name, 'reference_source': ref_source}
        tensors = {str(output_names[i] if i < len(output_names) else f'output{i}'): np.asarray(o) for i,o in enumerate(outputs)}
        dets, mode, diag = _choose_detection_candidate(tensors, ref_dets, img_w=640, img_h=640, conf=0.25)
        match = _match_detections(ref_dets, dets) if dets else {'ref_count': len(ref_dets), 'pred_count': 0, 'matched': 0, 'match_ratio': 0.0, 'mean_iou': 0.0}
        match_ca = _match_detections_class_agnostic(ref_dets, dets) if dets else {'ref_count': len(ref_dets), 'pred_count': 0, 'matched': 0, 'match_ratio': 0.0, 'mean_iou': 0.0, 'class_agnostic': True}
        sat = _score_saturation(dets)
        return {
            'available': True,
            'decode_mode': mode,
            'reference_image': image_name,
            'reference_source': ref_source,
            'pred_count': len(dets),
            'match': match,
            'class_agnostic_match': match_ca,
            'score_saturation': sat,
            'decode_reason': (diag or {}).get('reason'),
        }
    except Exception as e:
        return {'available': False, 'reason': f'semantic_score_failed: {e!r}'}


def _candidate_key(row: dict[str, Any]) -> tuple:
    sem = row.get('semantic') or {}
    if sem.get('available'):
        m = sem.get('match') or {}
        ca = sem.get('class_agnostic_match') or {}
        sat = sem.get('score_saturation') or {}
        return (
            int(m.get('matched', 0)),
            float(m.get('match_ratio', 0.0)),
            float(m.get('mean_iou', 0.0)),
            int(ca.get('matched', 0)),
            float(ca.get('match_ratio', 0.0)),
            -float(sat.get('ratio_ge_0999', 0.0)),
            -float((row.get('outputs') or [{}])[0].get('score_saturation_hint') or 0.0),
        )
    out0 = (row.get('outputs') or [{}])[0]
    return (0, 0.0, 0.0, 0, 0.0, -float(out0.get('score_saturation_hint') or 1.0), 0.0)


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--boundary-manifest', required=True)
    ap.add_argument('--scales', default='', help='Comma separated additional scales (tested with --zero-points if provided)')
    ap.add_argument('--zero-points', default='', help='Comma separated zero-points to combine with --scales')
    ap.add_argument('--out', default='')
    ap.add_argument('--max-candidates', type=int, default=120)
    ap.add_argument('--reference-report', default='', help='Optional ORT/full validation_report.json for semantic candidate ranking')
    ap.add_argument('--require-reference-detections', action='store_true', help='Exit non-zero if the supplied reference report has no detections or cannot be used for semantic ranking')
    ap.add_argument('--require-image-match', action='store_true', help='Exit non-zero if boundary dump provenance image and reference report image differ. Prevents semantic sweeps on mismatched images.')
    ap.add_argument('--summary-md', default='', help='Optional markdown summary path')
    ns=ap.parse_args()
    bs=Path(ns.benchmark_set).expanduser().resolve(); case=ns.case
    man=_read_json(Path(ns.boundary_manifest).expanduser().resolve())
    binp=Path(man.get('file','')).expanduser()
    if not binp.is_absolute(): binp=(Path(ns.boundary_manifest).parent/binp).resolve()
    raw=np.fromfile(str(binp), dtype=np.uint8)
    part2=_find_part2_onnx(bs,case)
    input_name, shape=_input_shape_from_onnx(part2)
    if any(d<0 for d in shape):
        raise RuntimeError(f'dynamic part2 input shape unsupported for sweep: {shape}')
    numel=int(np.prod(shape))
    if raw.size != numel:
        raise RuntimeError(f'boundary size mismatch: raw elements={raw.size}, part2 input elements={numel}, shape={shape}')
    raw=raw.reshape(shape)
    prov = man.get('provenance') if isinstance(man.get('provenance'), dict) else {}
    boundary_image = man.get('input_image') or man.get('image') or prov.get('image')
    reference_report_path = Path(ns.reference_report).expanduser().resolve() if ns.reference_report else None
    ref_dets_probe, reference_image, reference_source = _local_reference_detections(reference_report_path) if reference_report_path else ([], None, 'no_reference_report')
    boundary_image_sha = _normalize_sha256(
        man.get('input_image_sha256') or man.get('image_sha256')
        or prov.get('image_sha256')
    )
    boundary_image_path = Path(str(boundary_image)).expanduser() if boundary_image else None
    if not boundary_image_sha and boundary_image_path and boundary_image_path.is_file():
        boundary_image_sha = _sha256_file(boundary_image_path)
    reference_image_path = _resolve_reference_image(
        report=reference_report_path, image_name=reference_image,
        benchmark_set=bs,
    )
    reference_image_sha = (
        _sha256_file(reference_image_path) if reference_image_path else ''
    )
    image_identity_verified = _same_input_identity(
        boundary_image, boundary_image_sha,
        reference_image, reference_image_sha,
    )
    image_mismatch = bool(
        boundary_image and reference_image and not image_identity_verified
    )
    if ns.require_image_match and ns.reference_report and not image_identity_verified:
        out=Path(ns.out).expanduser().resolve() if ns.out else (bs/case/'native_boundary_dequant_sweep.json')
        out.parent.mkdir(parents=True,exist_ok=True)
        error = 'boundary_reference_image_mismatch' if image_mismatch else 'boundary_reference_image_identity_unverified'
        payload={'ok':False,'diagnostic_only':True,'claim_eligible':False,'error':error,'boundary_image':boundary_image,'boundary_image_sha256':boundary_image_sha,'reference_image':reference_image,'reference_image_sha256':reference_image_sha,'reference_source':reference_source,'boundary_manifest':str(Path(ns.boundary_manifest).resolve()),'reference_report':str(reference_report_path) if reference_report_path else ''}
        out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
        print(json.dumps({'ok':False,'error':error,'out':str(out),'boundary_image':boundary_image,'reference_image':reference_image},indent=2))
        return 3
    # Semantic ranking is allowed only for a byte-identical input. Even then it
    # remains diagnostic fitting and can never select a production contract.
    semantic_reference_report = str(reference_report_path) if image_identity_verified and reference_report_path else ''
    scales=[]
    if ns.scales.strip():
        for s in ns.scales.split(','):
            if s.strip(): scales.append(float(s.strip()))
    user_zps=[]
    if ns.zero_points.strip():
        for z in ns.zero_points.split(','):
            if z.strip(): user_zps.append(float(z.strip()))
    cands=_candidate_scales(bs,case,[])
    if scales:
        zps = user_zps or [0.0]
        for sc in scales:
            for zp in zps:
                cands.insert(0, {'source':'user_grid','scale':float(sc),'zero_point':float(zp)})
    cands=cands[:ns.max_candidates]
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError(f'onnxruntime required for sweep: {e}')
    sess=ort.InferenceSession(str(part2), providers=['CPUExecutionProvider'])
    output_names=[o.name for o in sess.get_outputs()]
    rows=[]
    for c in cands:
        scale=float(c.get('scale',1.0)); zp=float(c.get('zero_point',0.0))
        x=(raw.astype(np.float32)-zp)*scale
        try:
            outs=sess.run(None,{input_name:x})
            row={'ok':True,'diagnostic_only':True,'claim_eligible':False,'scale':scale,'zero_point':zp,'source':c.get('source'),'input_summary':_summ(x),'outputs':[_summ(o) for o in outs]}
            row['semantic']=_semantic_score_outputs([np.asarray(o) for o in outs], output_names, semantic_reference_report or None)
            row['selection_key']=list(_candidate_key(row))
            rows.append(row)
        except Exception as e:
            rows.append({'ok':False,'diagnostic_only':True,'claim_eligible':False,'scale':scale,'zero_point':zp,'source':c.get('source'),'error':repr(e)})
    # Summarize whether semantic ranking really had a usable reference.
    sem_reasons=[]
    sem_available=False
    for r in rows:
        sem = r.get('semantic') or {}
        if sem.get('available'):
            sem_available=True
        elif sem.get('reason'):
            sem_reasons.append(str(sem.get('reason')))
    reference_status = {
        'semantic_ranking_available': bool(sem_available),
        'reasons': sorted(set(sem_reasons)),
        'reference_report': str(Path(ns.reference_report).resolve()) if ns.reference_report else '',
        'boundary_image': boundary_image or '',
        'reference_image': reference_image or '',
        'reference_source': reference_source,
        'input_image_mismatch': bool(image_mismatch),
        'input_image_identity_verified': bool(image_identity_verified),
        'boundary_image_sha256': boundary_image_sha,
        'reference_image_sha256': reference_image_sha,
    }
    if ns.require_reference_detections and ns.reference_report and not sem_available:
        out=Path(ns.out).expanduser().resolve() if ns.out else (bs/case/'native_boundary_dequant_sweep.json')
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps({'ok':False,'diagnostic_only':True,'claim_eligible':False,'error':'reference_detections_unavailable','reference_status':reference_status},indent=2),encoding='utf-8')
        print(json.dumps({'ok':False,'error':'reference_detections_unavailable','out':str(out),'reference_status':reference_status},indent=2))
        return 2
    best_idx = max(range(len(rows)), key=lambda i: _candidate_key(rows[i])) if rows else None
    best_basis = 'semantic' if sem_available else 'output_stats_only_no_usable_reference'
    out=Path(ns.out).expanduser().resolve() if ns.out else (bs/case/'native_boundary_dequant_sweep.json')
    payload={
        'schema':'onnx-splitpoint/native-boundary-dequant-sweep', 'schema_version':5,
        'diagnostic_only': True,
        'claim_eligible': False,
        'production_contract_eligible': False,
        'selection_export_permitted': False,
        'benchmark_set':str(bs),'case':case,'part2_onnx':str(part2),
        'boundary_manifest':str(Path(ns.boundary_manifest).resolve()),
        'reference_report':str(Path(ns.reference_report).resolve()) if ns.reference_report else '',
        'reference_status': reference_status,
        'best_candidate_basis': best_basis,
        'candidate_zero_points': user_zps if user_zps else [0.0,1.0,2.0,5.0,10.0,16.0,32.0,64.0,128.0],
        'input_name':input_name,'input_shape':shape,'raw_summary':_summ(raw),
        'best_candidate_index':best_idx,'best_candidate':(rows[best_idx] if best_idx is not None else None),
        'candidates':rows,
    }
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    md_path=Path(ns.summary_md).expanduser().resolve() if ns.summary_md else out.with_suffix('.md')
    md=['# Native boundary dequant sweep','', '> DIAGNOSTIC ONLY: no candidate from this sweep is claim-eligible or exportable as a production dequantization contract.', '', f'Benchmark set: `{bs}`',f'Case: `{case}`',f'Part2 ONNX: `{part2}`',f'Reference: `{ns.reference_report or ""}`',f'Boundary image: `{boundary_image or ""}`',f'Reference image: `{reference_image or ""}`',f'Exact input identity verified: `{bool(image_identity_verified)}`',f'Input image mismatch: `{bool(image_mismatch)}`','']
    if not reference_status.get('semantic_ranking_available'):
        md += ['> WARNING: Semantic ranking is not available for this sweep. The selected best candidate is based on output statistics only, not AP50/IoU semantics.', f'> Reasons: `{reference_status.get("reasons")}`', '']
    # Distinguish semantic *availability* from semantic pass.  A row can
    # have available=True while match_ratio is still zero; the markdown should
    # not display that as a semantic True.
    best_match = 0.0
    best_ca = 0.0
    for r in rows:
        sem = r.get('semantic') or {}
        m = sem.get('match') or {}
        ca = sem.get('class_agnostic_match') or {}
        best_match = max(best_match, float(m.get('match_ratio') or 0.0))
        best_ca = max(best_ca, float(ca.get('match_ratio') or 0.0))
    if reference_status.get('semantic_ranking_available') and best_match < 0.25 and best_ca < 0.25:
        md += ['> WARNING: No dequant candidate produced a meaningful semantic or class-agnostic match against the reference. This suggests that scale/zero-point alone is not sufficient; check boundary layout/name/contract against ONNX Part1 activation.', '']
    md += ['| idx | scale | zp | source | sem_available | decode/reason | match | class-agnostic | pred | score_sat | out_sat |','|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|']
    for i,r in enumerate(rows):
        sem=r.get('semantic') or {}; m=sem.get('match') or {}; ca=sem.get('class_agnostic_match') or {}; sat=sem.get('score_saturation') or {}; out0=(r.get('outputs') or [{}])[0]
        md.append(f'| {i} | {r.get("scale")} | {r.get("zero_point")} | {r.get("source")} | {bool(sem.get("available"))} | {sem.get("decode_mode", sem.get("reason",""))} | {m.get("match_ratio",0)} | {ca.get("match_ratio",0)} | {sem.get("pred_count","")} | {sat.get("ratio_ge_0999","")} | {out0.get("score_saturation_hint","")} |')
    if best_idx is not None:
        b=rows[best_idx]
        md += ['', f'Best candidate index: `{best_idx}` scale=`{b.get("scale")}` zero_point=`{b.get("zero_point")}` source=`{b.get("source")}`']
        md += [f'Best observed semantic match ratio: `{best_match}`; best class-agnostic match ratio: `{best_ca}`']
    md_path.write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'ok':True,'out':str(out),'summary_md':str(md_path),'candidates':len(rows),'best_candidate_index':best_idx},indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
