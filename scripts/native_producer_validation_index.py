#!/usr/bin/env python3
"""Index native producer output dumps and validation artifacts.

This lightweight index is used before the task-specific visual validators are run.
It records which native rows have output dumps, generic tensor validation files and
which rows still need YOLO/classification semantic validation.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any


def _load_json(p: Path) -> Any:
    try:
        if p.is_file(): return json.loads(p.read_text(encoding='utf-8'))
    except Exception: pass
    return None


def _find_dump(report: Path) -> tuple[str, str]:
    j=_load_json(report) or {}
    cand = j.get('native_fifo_output_manifest') or j.get('output_manifest') or ''
    if cand and Path(cand).is_file():
        return cand, 'manifest_from_report'
    # search nearby
    for name in ('native_fifo_outputs_manifest.json','runner_outputs_manifest.json'):
        for p in report.parent.rglob(name):
            return str(p), 'nearby_search'
    return '', 'missing'


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', required=True, help='native_producer_summary.json / combined summary')
    ap.add_argument('--out-dir', required=True)
    ns=ap.parse_args()
    data=_load_json(Path(ns.summary).expanduser()) or {}
    rows=[]
    for r in data.get('rows', []) or []:
        if not r.get('ok'): continue
        report=Path(str(r.get('report') or '')).expanduser()
        dump, how = _find_dump(report) if report else ('','missing_report')
        val=''
        if dump:
            p=Path(dump)
            candidates=[p.parent/'native_fifo_output_validation.json', p.parent/'runner_output_validation.json', p.parent/'validation.json']
            val=next((str(x) for x in candidates if x.is_file()), '')
        model=str(r.get('model',''))
        task='detection' if model.lower().startswith('yolo') else 'classification'
        rows.append({**{k:r.get(k,'') for k in ('backend','model','case','producer_impl','fps_makespan','handoff_ms','report')},
                     'task':task,'dump_manifest':dump,'dump_status':how,'tensor_validation':val,
                     'next_validation':'visual_yolo_dump_validator' if task=='detection' else 'classification_topk_dump_validator'})
    out=Path(ns.out_dir).expanduser(); out.mkdir(parents=True, exist_ok=True)
    jsonp=out/'native_producer_validation_index.json'; csvp=out/'native_producer_validation_index.csv'; mdp=out/'native_producer_validation_index.md'
    jsonp.write_text(json.dumps({'rows':rows}, indent=2), encoding='utf-8')
    fields=['backend','model','case','producer_impl','task','fps_makespan','handoff_ms','dump_status','dump_manifest','tensor_validation','next_validation','report']
    with csvp.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    lines=['# Native producer validation index','','| backend | model | case | task | dump | next validation |','|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['backend']} | {r['model']} | {r['case']} | {r['task']} | {r['dump_status']} | {r['next_validation']} |")
    mdp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok':True,'rows':len(rows),'json':str(jsonp),'csv':str(csvp),'md':str(mdp)}, indent=2))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
