#!/usr/bin/env python3
"""Refresh an existing generated EvaluationRun/EvalSet with current runner code.

Use this when BenchmarkSets are already generated but the tool/runner scripts
changed.  It can refresh generated benchmark_suite.py/splitpoint runner files
for every model BenchmarkSet and optionally execute the native-producer stage in
place, producing reports/native_producer_*.  It is intentionally lighter than a
full EvalRun rerun.
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _models(eval_run: Path, explicit: list[str]) -> list[str]:
    if explicit: return explicit
    md = eval_run/'models'
    return sorted([p.name for p in md.iterdir() if (p/'benchmark_set'/'benchmark_set.json').is_file()]) if md.is_dir() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval-run-dir', required=True)
    ap.add_argument('--models', default='', help='Comma-separated model ids; default all models with benchmark_set.json')
    ap.add_argument('--refresh-suites', action='store_true', default=False, help='Refresh benchmark_suite.py and per-case runner scripts in generated BenchmarkSets')
    ap.add_argument('--run-native-producers', action='store_true', default=False, help='Run/update native producer stage after suite refresh')
    ap.add_argument('--native-backend', action='append', default=[], help='Native producer backend: hailo8, hailo10h, deepx. Repeatable.')
    ap.add_argument('--native-case-policy', default='all_accepted', choices=['all_accepted','preferred_then_backfill','case_map_only','first'])
    ap.add_argument('--native-case-map', default='')
    ap.add_argument('--native-remote-root', default='/home/nx/native_fifo_evalsets')
    ap.add_argument('--native-remote-tool-dir', default='/home/nx/ONNX-Splitpoint-Tool')
    ap.add_argument('--hailo8-ssh', default='')
    ap.add_argument('--hailo10-ssh', default='')
    ap.add_argument('--deepx-ssh', default='')
    ap.add_argument('--hailo8-env', default='')
    ap.add_argument('--hailo10-env', default='export PYTHONNOUSERSITE=1; source ~/venvs/hailo10/bin/activate')
    ap.add_argument('--deepx-env', default='source ~/venvs/deepx-runtime/bin/activate')
    ap.add_argument('--precision', default='uint8_cast_fp16')
    ap.add_argument('--frames', type=int, default=1000)
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--repetitions', type=int, default=3, help='Independent performance repetitions. Legacy safe default: Standard contract (3).')
    ap.add_argument('--queue-depth', type=int, default=3)
    ap.add_argument('--inflight', type=int, default=8)
    ap.add_argument('--hailo-format', default='uint8')
    ap.add_argument('--dump-outputs', action='store_true')
    ap.add_argument('--no-copy', action='store_true')
    ap.add_argument('--no-build-missing-engines', action='store_true')
    ap.add_argument('--native-telemetry-label', default='legacy_generated_standard', help='Stable label for canonical pre/post host telemetry evidence.')
    ns = ap.parse_args()
    if ns.repetitions < 1:
        ap.error('--repetitions must be >= 1')

    eval_run = Path(ns.eval_run_dir).expanduser().resolve()
    models = [m.strip() for m in ns.models.split(',') if m.strip()]
    models = _models(eval_run, models)
    if not ns.refresh_suites and not ns.run_native_producers:
        # Safe default: do both update steps.
        ns.refresh_suites = True; ns.run_native_producers = True
    results: dict[str, Any] = {'eval_run_dir': str(eval_run), 'models': models, 'suite_refresh': [], 'native_producers': None}
    if ns.refresh_suites:
        for m in models:
            bs = eval_run/'models'/m/'benchmark_set'
            cmd = [sys.executable, str(ROOT/'scripts'/'run_benchmark_suite_from_set.py'), '--benchmark-set', str(bs), '--refresh-only']
            pr = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            results['suite_refresh'].append({'model':m, 'benchmark_set':str(bs), 'rc':pr.returncode, 'stdout_tail':pr.stdout[-2000:], 'stderr_tail':pr.stderr[-2000:]})
    if ns.run_native_producers:
        cmd = [sys.executable, str(ROOT/'scripts'/'update_evalrun_native_producers.py'), '--eval-run-dir', str(eval_run), '--case-policy', ns.native_case_policy, '--remote-root', ns.native_remote_root, '--remote-tool-dir', ns.native_remote_tool_dir, '--precision', ns.precision, '--frames', str(ns.frames), '--warmup', str(ns.warmup), '--repetitions', str(ns.repetitions), '--queue-depth', str(ns.queue_depth), '--inflight', str(ns.inflight), '--hailo-format', ns.hailo_format, '--native-telemetry-label', ns.native_telemetry_label]
        if ns.models: cmd += ['--models', ns.models]
        for b in ns.native_backend: cmd += ['--backend', b]
        if ns.native_case_map: cmd += ['--case-map', ns.native_case_map]
        if ns.hailo8_ssh: cmd += ['--hailo8-ssh', ns.hailo8_ssh]
        if ns.hailo10_ssh: cmd += ['--hailo10-ssh', ns.hailo10_ssh]
        if ns.deepx_ssh: cmd += ['--deepx-ssh', ns.deepx_ssh]
        if ns.hailo8_env: cmd += ['--hailo8-env', ns.hailo8_env]
        if ns.hailo10_env: cmd += ['--hailo10-env', ns.hailo10_env]
        if ns.deepx_env: cmd += ['--deepx-env', ns.deepx_env]
        if ns.dump_outputs: cmd.append('--dump-outputs')
        if ns.no_copy: cmd.append('--no-copy')
        if ns.no_build_missing_engines: cmd.append('--no-build-missing-engines')
        pr = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results['native_producers'] = {'rc': pr.returncode, 'stdout_tail': pr.stdout[-4000:], 'stderr_tail': pr.stderr[-4000:], 'cmd': cmd}
    out = eval_run/'reports'/'generated_evalset_update.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    failed_refreshes = [row for row in results['suite_refresh'] if int(row.get('rc') or 0) != 0]
    native_rc = int((results.get('native_producers') or {}).get('rc') or 0)
    ok = not failed_refreshes and native_rc == 0
    print(json.dumps({'ok': ok, 'models': len(models), 'refresh_suites': bool(ns.refresh_suites), 'run_native_producers': bool(ns.run_native_producers), 'performance_repetitions': ns.repetitions, 'host_telemetry': 'canonical_pre_post_non_blocking', 'report': str(out)}, indent=2))
    return 0 if ok else 2

if __name__ == '__main__':
    raise SystemExit(main())
