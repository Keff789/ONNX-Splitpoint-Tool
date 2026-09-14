#!/usr/bin/env python3
"""Collect native producer result artefacts from remote Hailo/DeepX hosts.

This helper is intended to run on the orchestration host (e.g. Smartmirror2).
It copies compact native producer result artefacts from remote staged eval-set
roots into a local collection folder. It intentionally copies only summaries and
per-case result/dump manifests by default, not full BenchmarkSets or large output
.bin dumps, unless --include-dumps is requested.
"""
from __future__ import annotations
import argparse, json, subprocess, shlex, sys
from pathlib import Path
from typing import Any

DEFAULT_PATTERNS = [
    'analysis_tables/native_*producer*eval.*',
    'analysis_tables/native_fifo_eval_runner.*',
    'analysis_tables/native_fifo_summary.*',
    'analysis_tables/native_producer_combined_summary.*',
    '*/benchmark_set/native_pipeline/*/*/*/*results.json',
    '*/benchmark_set/native_pipeline/*/*/*/*status.json',
    '*/benchmark_set/native_pipeline/*/*/*/*probe.json',
    '*/benchmark_set/native_pipeline/*/*/*/native_fifo_report.md',
    '*/benchmark_set/benchmark_results_native_fifo_*.json',
]
DUMP_PATTERNS = [
    '*/benchmark_set/native_pipeline/*/*/*/native_fifo_outputs/*manifest.json',
    '*/benchmark_set/native_pipeline/*/*/*/native_fifo_outputs/*validation.json',
    '*/benchmark_set/native_pipeline/*/*/*/native_fifo_outputs/*.bin',
]

def run(cmd: list[str], dry: bool=False) -> int:
    print('[collect]', ' '.join(shlex.quote(x) for x in cmd), flush=True)
    if dry:
        return 0
    return subprocess.call(cmd)

def rsync_from(src: str, dst: Path, patterns: list[str], delete: bool=False, dry: bool=False) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    include_args: list[str] = []
    include_args += ['--include', '*/']
    for pat in patterns:
        include_args += ['--include', pat]
    include_args += ['--exclude', '*']
    cmd = ['rsync', '-a']
    if delete:
        cmd.append('--delete')
    cmd += include_args + [src.rstrip('/') + '/', str(dst) + '/']
    return run(cmd, dry=dry)

def parse_source(s: str) -> tuple[str, str, str]:
    # format: label=ssh:/path or label=/local/path
    if '=' not in s:
        raise SystemExit(f'Bad --source {s!r}; expected label=ssh:/path or label=/path')
    label, rest = s.split('=', 1)
    if ':' in rest and not rest.startswith('/'):
        ssh, path = rest.split(':', 1)
        return label, ssh + ':' + path, path
    return label, rest, rest

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', action='append', required=True,
                    help='Source as label=ssh@host:/remote/native_root or label=/local/native_root. Repeatable.')
    ap.add_argument('--out-root', required=True, help='Local collection root')
    ap.add_argument('--include-dumps', action='store_true', help='Also copy binary output dumps; can be large')
    ap.add_argument('--delete', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ns = ap.parse_args()
    out_root = Path(ns.out_root).expanduser().resolve()
    patterns = DEFAULT_PATTERNS + (DUMP_PATTERNS if ns.include_dumps else DUMP_PATTERNS[:-1])
    rows=[]; ok=True
    for spec in ns.source:
        label, src, _path = parse_source(spec)
        dst = out_root / label
        rc = rsync_from(src, dst, patterns, delete=ns.delete, dry=ns.dry_run)
        rows.append({'label':label,'src':src,'dst':str(dst),'rc':rc,'ok':rc==0})
        ok = ok and rc == 0
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root/'native_producer_collection_manifest.json').write_text(json.dumps({'ok':ok,'sources':rows}, indent=2), encoding='utf-8')
    print(json.dumps({'ok':ok,'out_root':str(out_root),'sources':rows}, indent=2))
    return 0 if ok else 2
if __name__ == '__main__':
    raise SystemExit(main())
