#!/usr/bin/env python3
"""Attach native-producer Fastpath summaries to an existing EvaluationRun.

The script copies/merges native producer summaries into <eval-run>/reports so the
native Fastpath becomes a first-class reporting artifact next to the generic
EvalRun. It does not require all native artifacts to be on the same host, but the
summary roots must be locally available on the machine where this script runs.
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_json(p: Path) -> Any:
    try:
        if p.is_file():
            return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eval-run-dir', required=True)
    ap.add_argument('--native-root', action='append', required=True, help='One or more local native_fifo_evalsets roots or backend copies')
    ap.add_argument('--copy-native-artifacts', action='store_true', help='Copy analysis_tables and result jsons into reports/native_producers/artifacts')
    ap.add_argument('--recursive', action='store_true', help='Pass --recursive to native_producer_final_report')
    ap.add_argument(
        '--remote-execution-context-json', action='append', default=[],
        help=(
            'Repeatable schema-bound setup/root/tool allowlist record passed '
            'through to native_producer_final_report.'
        ),
    )
    ns = ap.parse_args()
    eval_dir = Path(ns.eval_run_dir).expanduser().resolve()
    reports = eval_dir / 'reports'; reports.mkdir(parents=True, exist_ok=True)
    native_report_dir = reports / 'native_producers'; native_report_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(ROOT/'scripts'/'native_producer_final_report.py')]
    for r in ns.native_root:
        cmd += ['--root', str(Path(r).expanduser().resolve())]
    if ns.recursive:
        cmd.append('--recursive')
    for context in ns.remote_execution_context_json:
        cmd += ['--remote-execution-context-json', str(context)]
    cmd += ['--out-dir', str(native_report_dir)]
    subprocess.check_call(cmd)

    src_json = native_report_dir / 'native_producer_combined_summary.json'
    src_csv = native_report_dir / 'native_producer_combined_summary.csv'
    src_md = native_report_dir / 'native_producer_combined_summary.md'
    # Convenience copies at reports root
    for src, name in [(src_json,'native_producer_summary.json'),(src_csv,'native_producer_summary.csv'),(src_md,'native_producer_summary.md')]:
        if src.is_file():
            shutil.copy2(src, reports / name)

    data = _load_json(src_json) or {'rows': []}
    rows = data.get('rows', []) or []
    idx = {
        'schema': 'onnx-splitpoint/native-producer-attach',
        'eval_run_dir': str(eval_dir),
        'native_roots': [str(Path(r).expanduser().resolve()) for r in ns.native_root],
        'row_count': len(rows),
        'ok_count': sum(1 for r in rows if r.get('ok')),
        'reports': {
            'json': str(reports/'native_producer_summary.json'),
            'csv': str(reports/'native_producer_summary.csv'),
            'md': str(reports/'native_producer_summary.md'),
        },
    }
    if ns.copy_native_artifacts:
        art = native_report_dir / 'artifacts'; art.mkdir(parents=True, exist_ok=True)
        copied = []
        for row in rows:
            rp = Path(str(row.get('report') or '')).expanduser()
            if rp.is_file():
                dst = art / row.get('backend','backend') / row.get('model','model') / row.get('case','case')
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rp, dst / rp.name)
                copied.append(str(dst / rp.name))
        idx['copied_artifacts'] = copied
    (native_report_dir / 'attach_manifest.json').write_text(json.dumps(idx, indent=2), encoding='utf-8')
    print(json.dumps({'ok': True, **idx}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
