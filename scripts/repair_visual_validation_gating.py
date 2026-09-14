#!/usr/bin/env python3
"""Repair validation tables so visual FAIL rows are surfaced as failures.

This helper is intentionally conservative. It does not invent semantic success:
- visual_verification PASS/FAIL rows are counted per model.
- if visual FAIL rows exist, model summary validation_ok_with_visual is false.
- Native validation debug artifacts remain artifacts; claim_ok remains false until a
  strict tensor or semantic validator exists.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def read_csv(p: Path):
    if not p.is_file(): return []
    with p.open(newline='', encoding='utf-8') as f: return list(csv.DictReader(f))

def write_csv(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r.keys():
            if k not in fields: fields.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--eval-run-dir', required=True)
    ns=ap.parse_args(); run=Path(ns.eval_run_dir); reports=run/'reports'
    vis=read_csv(reports/'visual_verification'/'visual_verification.csv')
    counts={}
    for r in vis:
        mid=(r.get('model_id') or '').strip(); st=(r.get('status') or '').strip().lower()
        if not mid: continue
        c=counts.setdefault(mid, {'total':0,'pass':0,'fail':0,'unknown':0})
        c['total']+=1
        if st in ('pass','passed','true','ok','success'): c['pass']+=1
        elif st in ('fail','failed','false','error'): c['fail']+=1
        else: c['unknown']+=1
    for name in ('validation_summary.csv','summary.csv','model_summary.csv'):
        p=reports/name; rows=read_csv(p)
        if not rows: continue
        for r in rows:
            mid=(r.get('model_id') or r.get('model') or '').strip(); c=counts.get(mid)
            if not c: continue
            r['visual_verification_total']=c['total']; r['visual_verification_pass']=c['pass']; r['visual_verification_invalid']=c['fail']; r['visual_verification_unvalidated']=c['unknown']
            r['visual_validation_ok']='False' if c['fail'] else 'True'
            r['validation_ok_with_visual']='False' if c['fail'] else str(r.get('validation_ok',''))
            if c['fail']:
                r['validation_status']='visual_failures_present'
                r['validation_ok']='False'
        write_csv(p, rows)
    out={'ok': True, 'visual_counts': counts, 'updated': ['validation_summary.csv','summary.csv','model_summary.csv']}
    (reports/'visual_verification'/'visual_gating_repair.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__=='__main__': main()
