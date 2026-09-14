#!/usr/bin/env python3
"""Run Hailo10H/DeepX native-FIFO producer probes over staged EvalRun benchmark sets.

This is the probe-stage counterpart of native_fifo_eval_runner.py. It does not
claim E2E native FIFO measurements for Hailo10/DeepX yet. It verifies producer
readiness and TensorRT consumer readiness over a selected case map.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parse_models(s: str | None, root: Path) -> list[str]:
    if s:
        return [x.strip() for x in s.split(',') if x.strip()]
    return [p.name for p in sorted(root.iterdir() if root.exists() else []) if (p / 'benchmark_set').exists()]


def _case_map(s: str | None) -> dict[str, list[str]]:
    if not s:
        return {}
    data = json.loads(s)
    out: dict[str, list[str]] = {}
    for model, cases in data.items():
        if isinstance(cases, str):
            cases = [cases]
        out[str(model)] = [str(c) for c in cases]
    return out


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _find_bs(root: Path, model: str) -> Path | None:
    candidates = [root / model / 'benchmark_set', root / model / 'benchmark_set' / 'legacy_suite', root / model]
    for p in candidates:
        if (p / 'benchmark_set.json').exists():
            return p.resolve()
    base = root / model
    if base.exists():
        for p in sorted(base.rglob('benchmark_set.json')):
            return p.parent.resolve()
    return None


def _run(cmd: list[str], timeout: float | None = None) -> dict[str, Any]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {'cmd': cmd, 'rc': p.returncode, 'elapsed_s': time.time()-t0, 'stdout_tail': p.stdout[-8000:], 'stderr_tail': p.stderr[-8000:]}
    except subprocess.TimeoutExpired as e:
        return {'cmd': cmd, 'rc': 124, 'elapsed_s': time.time()-t0, 'stdout_tail': (e.stdout or '')[-8000:] if isinstance(e.stdout, str) else '', 'stderr_tail': (e.stderr or '')[-8000:] if isinstance(e.stderr, str) else '', 'timeout': True}


def _probe_report_path(bs: Path, backend: str, case: str, precision: str) -> Path:
    if backend == 'hailo10h':
        return bs / 'native_pipeline' / case / 'hailo10h_to_trt' / precision / 'hailo10_native_fifo_probe.json'
    if backend == 'deepx':
        return bs / 'native_pipeline' / case / 'deepx_to_trt' / precision / 'deepx_native_fifo_probe.json'
    raise ValueError(backend)


def _summarize_probe(model: str, bs: Path, backend: str, case: str, precision: str, step: dict[str, Any]) -> dict[str, Any]:
    report_path = _probe_report_path(bs, backend, case, precision)
    rep = _load_json(report_path)
    trt_probe = rep.get('native_trt_part2_probe') if isinstance(rep, dict) else {}
    consumer_ready = rep.get('consumer_ready', None)
    if consumer_ready is None:
        consumer_ready = bool(rep.get('part2_engine')) or (isinstance(trt_probe, dict) and trt_probe.get('latency_mean_ms') is not None)
    producer_ready = rep.get('producer_ready', None)
    if producer_ready is None:
        producer_ready = bool(rep.get('hailo10_part1_probe')) if backend == 'hailo10h' else bool((rep.get('deepx_python_probe') or {}).get('modules'))
    row = {
        'model': model,
        'case_id': case,
        'backend': backend,
        'benchmark_set': str(bs),
        'probe_report': str(report_path),
        'probe_ok': bool(rep.get('ok')),
        'producer_ready': producer_ready,
        'consumer_ready': consumer_ready,
        'native_fifo_e2e_implemented': rep.get('native_fifo_e2e_implemented'),
        'projected_fps_without_handoff': rep.get('projected_fps_without_handoff'),
        'projected_cycle_ms_without_handoff': rep.get('projected_cycle_ms_without_handoff'),
        'error': rep.get('error', ''),
        'next_action': rep.get('next_action', ''),
        'rc': step.get('rc'),
        'elapsed_s': step.get('elapsed_s'),
    }
    # extra Hailo10 fields
    if backend == 'hailo10h':
        h = rep.get('hailo10_part1_probe') or {}
        th = h.get('throughput') if isinstance(h, dict) else None
        if isinstance(th, dict):
            row['producer_fps'] = th.get('fps')
            row['producer_completion_interval_ms'] = th.get('completion_interval_mean_ms')
        trt = rep.get('native_trt_part2_probe') or {}
        row['trt_latency_ms'] = trt.get('latency_mean_ms') if isinstance(trt, dict) else None
    # extra DeepX fields
    if backend == 'deepx':
        row['deepx_modules'] = ','.join((rep.get('deepx_python_probe') or {}).get('modules', []))
        row['deepx_artifact_count'] = len(rep.get('deepx_artifacts') or [])
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description='Run Hailo10H/DeepX native FIFO producer probes over staged benchmark sets')
    ap.add_argument('--root', required=True)
    ap.add_argument('--backend', required=True, choices=['hailo10h','deepx'])
    ap.add_argument('--models', default='')
    ap.add_argument('--case-map', default='')
    ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--build-missing-engine', action='store_true')
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--inflight', type=int, default=8, help='Hailo10 only')
    ap.add_argument('--timeout', type=float, default=3600)
    ap.add_argument('--engine-build-python', default='auto')
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    models = _parse_models(args.models, root)
    cmap = _case_map(args.case_map)
    out_dir = root / 'analysis_tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    model_steps: list[dict[str, Any]] = []

    for model in models:
        bs = _find_bs(root, model)
        if not bs:
            rows.append({'model': model, 'backend': args.backend, 'probe_ok': False, 'error': 'benchmark_set_not_found'})
            continue
        cases = cmap.get(model) or []
        if not cases:
            # Default: all case directories.
            cases = [p.name for p in sorted(bs.glob('b*')) if p.is_dir()]
        for case in cases:
            if args.backend == 'hailo10h':
                cmd = [sys.executable, str(ROOT/'scripts'/'native_hailo10_trt_fifo_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', case, '--hw-arch', 'hailo10h', '--precision', args.precision, '--frames', str(args.frames), '--warmup', str(args.warmup), '--inflight', str(args.inflight), '--engine-build-python', args.engine_build_python]
                if args.build_missing_engine:
                    cmd.append('--build-missing-engine')
                cmd += ['--quantized-inputs', '--quantized-outputs']
            else:
                cmd = [sys.executable, str(ROOT/'scripts'/'native_deepx_trt_fifo_from_benchmarkset.py'), '--benchmark-set', str(bs), '--case', case, '--precision', args.precision, '--engine-build-python', args.engine_build_python]
                if args.build_missing_engine:
                    cmd.append('--build-missing-engine')
            print(f"[producer-probe-eval] {model}/{case}/{args.backend}: {' '.join(cmd)}", flush=True)
            step = _run(cmd, timeout=args.timeout)
            model_steps.append({'model': model, 'case_id': case, 'backend': args.backend, **step})
            rows.append(_summarize_probe(model, bs, args.backend, case, args.precision, step))

    summary = {'ok': True, 'root': str(root), 'backend': args.backend, 'precision': args.precision, 'rows': rows, 'steps': model_steps, 'row_count': len(rows), 'ok_count': sum(1 for r in rows if r.get('probe_ok'))}
    prefix = f'native_{args.backend}_producer_probe_eval'
    (out_dir / f'{prefix}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    fields = ['model','case_id','backend','probe_ok','producer_ready','consumer_ready','native_fifo_e2e_implemented','projected_fps_without_handoff','projected_cycle_ms_without_handoff','producer_fps','producer_completion_interval_ms','trt_latency_ms','deepx_modules','deepx_artifact_count','error','next_action','rc','elapsed_s','probe_report','benchmark_set']
    with (out_dir / f'{prefix}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})
    lines = [f'# Native {args.backend} producer probe eval', '', f'Root: `{root}`', f'Precision: `{args.precision}`', '', '| model | case | ok | producer | consumer | projected FPS | error / next action |', '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r.get('model')} | {r.get('case_id')} | {r.get('probe_ok')} | {r.get('producer_ready')} | {r.get('consumer_ready')} | {r.get('projected_fps_without_handoff') or ''} | {r.get('error') or r.get('next_action') or ''} |")
    (out_dir / f'{prefix}.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'rows': len(rows), 'ok_count': summary['ok_count'], 'json': str(out_dir/f'{prefix}.json'), 'csv': str(out_dir/f'{prefix}.csv'), 'md': str(out_dir/f'{prefix}.md')}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
