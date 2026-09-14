#!/usr/bin/env python3
"""Summarize Hailo10/DeepX producer probe reports and emit next-action commands.

This is a small planning helper. It does not claim E2E native FIFO execution for
Hailo10/DeepX; it turns probe outputs into a concise readiness table and tells
which producer implementation should be built/tested next.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _probe_files(root: Path, backend: str) -> list[Path]:
    if backend == 'hailo10h':
        return sorted(root.rglob('hailo10_native_fifo_probe.json'))
    if backend == 'deepx':
        return sorted(root.rglob('deepx_native_fifo_probe.json'))
    return sorted(root.rglob('*native_fifo_probe.json'))


def _case_model_from_path(root: Path, p: Path) -> tuple[str, str]:
    try:
        rel = p.relative_to(root)
        # <model>/benchmark_set/native_pipeline/<case>/...
        parts = rel.parts
        if len(parts) >= 4 and parts[1] == 'benchmark_set':
            return parts[0], parts[3]
    except Exception:
        pass
    data = _load(p)
    bs = str(data.get('benchmark_set',''))
    model = Path(bs).parent.name if bs else ''
    return model, str(data.get('case',''))


def main() -> int:
    ap = argparse.ArgumentParser(description='Summarize Hailo10/DeepX native FIFO producer probe readiness')
    ap.add_argument('--root', required=True)
    ap.add_argument('--backend', choices=['hailo10h','deepx','all'], default='all')
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    backends = ['hailo10h','deepx'] if args.backend == 'all' else [args.backend]
    rows: list[dict[str, Any]] = []
    for backend in backends:
        for p in _probe_files(root, backend):
            data = _load(p)
            model, case = _case_model_from_path(root, p)
            prod = bool(data.get('producer_ready') or data.get('hailo10_part1_probe') or (data.get('deepx_python_probe') or {}).get('modules'))
            trt = data.get('native_trt_part2_probe') or {}
            cons = bool(data.get('consumer_ready') or data.get('part2_engine') or (isinstance(trt, dict) and trt.get('latency_mean_ms') is not None))
            next_action = ''
            if not prod:
                next_action = 'fix producer runtime/artifacts'
            elif not cons:
                next_action = 'build native TensorRT part2 engine'
            else:
                next_action = 'ready_for_e2e_producer_impl'
            rows.append({
                'backend': backend,
                'model': model,
                'case_id': case,
                'probe_ok': bool(data.get('ok')),
                'producer_ready': prod,
                'consumer_ready': cons,
                'projected_fps_without_handoff': data.get('projected_fps_without_handoff',''),
                'producer_fps': ((data.get('hailo10_part1_probe') or {}).get('throughput') or {}).get('fps',''),
                'trt_latency_ms': trt.get('latency_mean_ms') if isinstance(trt, dict) else '',
                'e2e_implemented': bool(data.get('native_fifo_e2e_implemented')),
                'next_action': next_action,
                'report': str(p),
            })
    out = root / 'analysis_tables'
    out.mkdir(parents=True, exist_ok=True)
    j = out / 'native_producer_next_actions.json'
    c = out / 'native_producer_next_actions.csv'
    m = out / 'native_producer_next_actions.md'
    summary = {'root': str(root), 'rows': rows, 'row_count': len(rows)}
    j.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['backend','model','case_id','probe_ok','producer_ready','consumer_ready','projected_fps_without_handoff','producer_fps','trt_latency_ms','e2e_implemented','next_action','report']
    with c.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in fields})
    lines = ['# Native producer next actions','',f'Root: `{root}`','', '| backend | model | case | producer | consumer | projected FPS | next action |', '|---|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['backend']} | {r['model']} | {r['case_id']} | {r['producer_ready']} | {r['consumer_ready']} | {r.get('projected_fps_without_handoff') or ''} | {r['next_action']} |")
    m.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'rows': len(rows), 'json': str(j), 'csv': str(c), 'md': str(m)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
