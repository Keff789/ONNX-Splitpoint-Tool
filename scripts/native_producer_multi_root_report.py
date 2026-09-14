#!/usr/bin/env python3
"""Create a combined Native Producer report from multiple collected backend roots.

The expected input is a local collection directory produced by
native_producer_collect_remote_results.py, for example:

  NativeFIFO/complete_set/hailo8/...
  NativeFIFO/complete_set/hailo10/...
  NativeFIFO/complete_set/deepx/...

Each subdirectory can either be the staged eval root itself or contain one.
The script scans recursively for native producer summaries and per-case result
JSONs and creates one thesis-facing combined table.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any

def load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None

def fnum(x: Any):
    try:
        if x is None or x == '': return None
        return float(x)
    except Exception:
        return None

def model_case_from_path(p: Path) -> tuple[str,str]:
    parts = list(p.parts)
    model=''; case=''
    if 'benchmark_set' in parts:
        idx = parts.index('benchmark_set')
        if idx >= 1:
            model = parts[idx-1]
        if 'native_pipeline' in parts:
            nidx = parts.index('native_pipeline')
            if nidx + 1 < len(parts): case = parts[nidx+1]
    return model, case

def row_from_result(p: Path, label: str) -> dict[str, Any] | None:
    j = load_json(p)
    if not isinstance(j, dict): return None
    model, case = model_case_from_path(p)
    name = p.name
    parent = str(p)
    backend=''; impl=''; status='ok' if j.get('ok') else str(j.get('error') or j.get('status') or 'failed')
    if 'hailo10' in name or 'hailo10h_to_trt' in parent:
        backend='hailo10h_to_trt'; impl=j.get('producer_impl') or 'hailo10_infermodel_async_fifo'
    elif 'deepx' in name or 'deepx_to_trt' in parent:
        backend='deepx_to_trt'; impl='deepx_python_dx_engine_fifo' if j.get('ok') else 'deepx_scaffold'
    elif 'native_fifo_results' in name or 'hailo_to_trt' in parent:
        backend='hailo8_to_trt'; impl='hailo8_cpp_vstreams_fifo'
    else:
        return None
    return {
        'source': label,
        'backend': backend,
        'impl': impl,
        'model': model or j.get('model',''),
        'case': case or j.get('case',''),
        'status': status,
        'ok': bool(j.get('ok')),
        'fps_makespan': fnum(j.get('fps_makespan')),
        'paper_fps': fnum(j.get('paper_equivalent_fps') or j.get('paper_fps')),
        'handoff_ms': fnum(j.get('handoff_ms')),
        'p1_ms': fnum(j.get('p1_ms') or j.get('deepx_run_ms')),
        'p2_run_ms': fnum(j.get('p2_run_ms')),
        'p1_thread_ms': fnum(j.get('p1_thread_ms')),
        'p2_thread_ms': fnum(j.get('p2_thread_ms')),
        'frames': j.get('frames'),
        'warmup': j.get('warmup'),
        'inflight': j.get('inflight'),
        'precision': j.get('precision',''),
        'report': str(p),
    }

def scan_root(root: Path, label: str) -> list[dict[str,Any]]:
    rows=[]
    patterns = ['*native_fifo_results.json','*hailo10_native_fifo_e2e_results.json','*deepx_native_fifo_e2e_results.json','*deepx_native_fifo_e2e_status.json']
    seen=set()
    for pat in patterns:
        for p in root.rglob(pat):
            if p in seen: continue
            seen.add(p)
            r = row_from_result(p, label)
            if r: rows.append(r)
    # Deduplicate by source/backend/model/case, keep ok and highest frames/fps if duplicates
    best={}
    for r in rows:
        k=(r.get('source'),r.get('backend'),r.get('model'),r.get('case'))
        old=best.get(k)
        if old is None:
            best[k]=r
        else:
            old_score=(1 if old.get('ok') else 0, old.get('frames') or 0, old.get('fps_makespan') or 0)
            new_score=(1 if r.get('ok') else 0, r.get('frames') or 0, r.get('fps_makespan') or 0)
            if new_score >= old_score: best[k]=r
    return list(best.values())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', action='append', required=True, help='label=/path or just /path. Repeatable.')
    ap.add_argument('--out-dir', required=True)
    ns=ap.parse_args()
    all_rows=[]
    for spec in ns.root:
        if '=' in spec:
            label, path = spec.split('=',1)
        else:
            path = spec; label = Path(path).name
        all_rows += scan_root(Path(path).expanduser().resolve(), label)
    all_rows=sorted(all_rows, key=lambda r:(r['backend'], r['model'], r['case'], r['source']))
    out=Path(ns.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    data={'rows':all_rows,'row_count':len(all_rows),'ok_count':sum(1 for r in all_rows if r.get('ok'))}
    jsonp=out/'native_producer_multiroot_summary.json'; csvp=out/'native_producer_multiroot_summary.csv'; mdp=out/'native_producer_multiroot_summary.md'
    jsonp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    fields=['source','backend','impl','model','case','status','ok','fps_makespan','paper_fps','handoff_ms','p1_ms','p2_run_ms','p1_thread_ms','p2_thread_ms','frames','warmup','inflight','precision','report']
    with csvp.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader();
        for r in all_rows: w.writerow({k:r.get(k,'') for k in fields})
    lines=['# Native producer multi-root summary','', '| source | backend | model | case | ok | FPS | handoff ms | impl |','|---|---|---|---|---:|---:|---:|---|']
    for r in all_rows:
        fps='' if r.get('fps_makespan') is None else f"{float(r['fps_makespan']):.3f}"
        h='' if r.get('handoff_ms') is None else f"{float(r['handoff_ms']):.3f}"
        lines.append(f"| {r.get('source','')} | {r.get('backend','')} | {r.get('model','')} | {r.get('case','')} | {r.get('ok')} | {fps} | {h} | {r.get('impl','')} |")
    mdp.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok':True,'rows':len(all_rows),'ok_count':data['ok_count'],'json':str(jsonp),'csv':str(csvp),'md':str(mdp)}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
