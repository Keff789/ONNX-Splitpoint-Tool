#!/usr/bin/env python3
"""Run Hailo10H->TRT E2E for multiple inflight values and summarize.

This is a convenience wrapper around native_hailo10_trt_e2e_from_benchmarkset.py.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _case_id(x: str) -> str:
    s=str(x)
    return s if s.startswith('b') else f'b{int(s):03d}'

def _load(path: Path):
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return None

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', required=True)
    ap.add_argument('--hw-arch', default='hailo10h')
    ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--inflight-values', default='1,2,4,8,16')
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--build-missing-engine', action='store_true')
    ap.add_argument('--producer-impl', default='async_fifo', choices=['async_fifo','sync','auto'])
    ap.add_argument('--quantized-inputs', action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--quantized-outputs', action=argparse.BooleanOptionalAction, default=True)
    ns = ap.parse_args()
    bs=Path(ns.benchmark_set).expanduser().resolve(); case=_case_id(ns.case)
    rows=[]
    for item in ns.inflight_values.split(','):
        item=item.strip()
        if not item: continue
        inflight=int(item)
        cmd=[sys.executable, str(ROOT/'scripts'/'native_hailo10_trt_e2e_from_benchmarkset.py'),
             '--benchmark-set', str(bs), '--case', case, '--hw-arch', ns.hw_arch,
             '--precision', ns.precision, '--frames', str(ns.frames), '--warmup', str(ns.warmup),
             '--queue-depth', str(ns.queue_depth), '--inflight', str(inflight), '--producer-impl', ns.producer_impl]
        if ns.build_missing_engine: cmd.append('--build-missing-engine')
        cmd.append('--quantized-inputs' if ns.quantized_inputs else '--no-quantized-inputs')
        cmd.append('--quantized-outputs' if ns.quantized_outputs else '--no-quantized-outputs')
        print('[inflight-sweep]', ' '.join(cmd), flush=True)
        p=subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report_path=bs/'native_pipeline'/case/'hailo10h_to_trt'/ns.precision/'hailo10_native_fifo_e2e_results.json'
        rep=_load(report_path) or {}
        row={
            'inflight': inflight,
            'rc': p.returncode,
            'ok': bool(rep.get('ok')),
            'fps_makespan': rep.get('fps_makespan'),
            'paper_equivalent_fps': rep.get('paper_equivalent_fps'),
            'p1_effective_cycle_ms': rep.get('p1_effective_cycle_ms'),
            'p2_run_ms': rep.get('p2_run_ms'),
            'handoff_ms': rep.get('handoff_ms'),
            'fifo_queue_wait_ms': rep.get('fifo_queue_wait_ms'),
            'hailo_latency_ms': rep.get('hailo_latency_ms'),
            'report': str(report_path),
            'stdout_tail': p.stdout[-2000:],
            'stderr_tail': p.stderr[-2000:],
        }
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['inflight','ok','fps_makespan','paper_equivalent_fps','p1_effective_cycle_ms','p2_run_ms','handoff_ms','fifo_queue_wait_ms']}, indent=2), flush=True)
    out_dir=bs/'analysis_tables'; out_dir.mkdir(parents=True, exist_ok=True)
    out=out_dir/f'native_hailo10h_inflight_sweep_{case}.json'
    out.write_text(json.dumps({'benchmark_set':str(bs),'case':case,'rows':rows}, indent=2), encoding='utf-8')
    md=out.with_suffix('.md')
    lines=[f'# Hailo10 inflight sweep {case}', '', '| inflight | ok | FPS | paper FPS | p1 cycle ms | p2 run ms | handoff ms | fifo wait ms |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        def f(x):
            return '' if x is None else (f'{x:.3f}' if isinstance(x,(float,int)) else str(x))
        lines.append(f"| {r['inflight']} | {r['ok']} | {f(r.get('fps_makespan'))} | {f(r.get('paper_equivalent_fps'))} | {f(r.get('p1_effective_cycle_ms'))} | {f(r.get('p2_run_ms'))} | {f(r.get('handoff_ms'))} | {f(r.get('fifo_queue_wait_ms'))} |")
    md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok':True,'rows':len(rows),'json':str(out),'md':str(md)}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
