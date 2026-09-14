#!/usr/bin/env python3
"""Summarize native detection validation reports for likely boundary-contract issues.

This helper is intentionally conservative. It does not mark rows as valid; it scans
existing detection_visual_validation.json files and highlights patterns that usually
mean the native boundary contract is semantically wrong, for example a uint8 Cast
bridge feeding a float TensorRT part2 without the producer's affine dequantization.
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        return {'_load_error': f'{type(e).__name__}: {e}'}


def _best_candidate(rep: dict[str, Any]) -> dict[str, Any]:
    diag = rep.get('decode_diagnostics') or {}
    cands = diag.get('candidates') or []
    best = None
    best_key = (-1.0, -1.0, -10**9)
    for c in cands:
        m = c.get('match') or {}
        ma = c.get('class_agnostic_match') or {}
        sat = c.get('score_saturation') or {}
        key = (float(m.get('match_ratio') or 0.0), float(ma.get('match_ratio') or 0.0), -int(sat.get('score_ge_0999') or 0))
        if key > best_key:
            best_key, best = key, c
    return best or {}


def _diagnose(rep: dict[str, Any]) -> tuple[str, str]:
    if rep.get('_load_error'):
        return 'error', str(rep['_load_error'])
    if rep.get('claim_ok') is True or rep.get('semantic_ok') is True:
        return 'semantic_ok', 'semantic validation passed'
    if rep.get('input_image_mismatch'):
        return 'input_mismatch', 'native input image does not match reference image'
    best = _best_candidate(rep)
    m = best.get('match') or {}
    ma = best.get('class_agnostic_match') or {}
    sat = best.get('score_saturation') or {}
    mr = float(m.get('match_ratio') or 0.0)
    amr = float(ma.get('match_ratio') or 0.0)
    sat_ratio = float(sat.get('ratio_ge_0999') or 0.0)
    if mr <= 0.0 and amr <= 0.0:
        return 'postprocess_or_boundary_fail', 'no tested layout overlaps the reference; output contract/coordinates/postprocess suspect'
    if mr < 0.2 and amr < 0.2 and sat_ratio >= 0.8:
        return 'boundary_quantization_suspect', 'very low semantic and class-agnostic match plus score saturation; uint8_cast_fp16 likely not a valid semantic boundary'
    if mr < 0.2 and amr > mr:
        return 'class_or_score_mapping_suspect', 'some boxes overlap class-agnostically but class-aware match fails; class/score mapping or quantization suspect'
    if mr < 0.2:
        return 'semantic_fail', 'semantic match below acceptance threshold'
    return 'semantic_fail', 'semantic validation failed'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-run-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=None)
    ns = ap.parse_args(argv)
    run = ns.eval_run_dir
    root = run / 'reports' / 'native_validation'
    out_dir = ns.out_dir or (run / 'reports' / 'native_validation')
    out_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for p in sorted(root.rglob('detection_visual_validation.json')):
        rep = _load(p)
        status, reason = _diagnose(rep)
        rel = p.parent.name
        parts = rel.split('__')
        backend = parts[0] if len(parts)>0 else ''
        model = parts[1] if len(parts)>1 else ''
        case = parts[2] if len(parts)>2 else ''
        best = _best_candidate(rep)
        rows.append({
            'backend': backend,
            'model': model,
            'case': case,
            'status': status,
            'reason': reason,
            'decode_mode': rep.get('decode_mode',''),
            'warning': rep.get('decode_warning',''),
            'match_ratio': (rep.get('match') or {}).get('match_ratio',''),
            'class_agnostic_match_ratio': (rep.get('class_agnostic_match') or {}).get('match_ratio',''),
            'best_candidate': best.get('mode',''),
            'score_saturation_ratio': (best.get('score_saturation') or {}).get('ratio_ge_0999',''),
            'report': str(p),
        })
    jsonp = out_dir / 'native_boundary_contract_probe.json'
    csvp = out_dir / 'native_boundary_contract_probe.csv'
    mdp = out_dir / 'native_boundary_contract_probe.md'
    payload={'schema':'onnx-splitpoint/native-boundary-contract-probe','schema_version':1,'eval_run_dir':str(run),'rows':rows,'row_count':len(rows)}
    jsonp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    with csvp.open('w', newline='', encoding='utf-8') as f:
        fields=['backend','model','case','status','reason','decode_mode','warning','match_ratio','class_agnostic_match_ratio','best_candidate','score_saturation_ratio','report']
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    md=['# Native boundary contract probe','', '| backend | model | case | status | reason | match | class-agnostic | saturation |', '|---|---|---|---|---|---:|---:|---:|']
    for r in rows:
        md.append(f"| {r['backend']} | {r['model']} | {r['case']} | {r['status']} | {r['reason']} | {r['match_ratio']} | {r['class_agnostic_match_ratio']} | {r['score_saturation_ratio']} |")
    mdp.write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'ok':True,'rows':len(rows),'json':str(jsonp),'csv':str(csvp),'md':str(mdp)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
