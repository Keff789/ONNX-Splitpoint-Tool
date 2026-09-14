#!/usr/bin/env python3
"""Locate and run a generated benchmark_suite.py inside a BenchmarkSet.

The tool package itself does not contain benchmark_suite.py; every BenchmarkSet
contains its own generated suite, usually under <benchmark_set>/legacy_suite/.
This wrapper avoids accidentally running from the wrong directory.

Example:
  python scripts/run_generated_suite.py --benchmark-set "$BS" -- \
    --run-id hailo8_to_trt --trt-runtime-mode native_preferred \
    --native-trt-precision fp16 --boundary-mode canonical_float32
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def find_suite(bs: Path) -> Path:
    candidates = [
        bs / 'legacy_suite' / 'benchmark_suite.py',
        bs / 'benchmark_suite.py',
    ]
    for c in candidates:
        if c.is_file():
            return c
    found = sorted(bs.glob('**/benchmark_suite.py'))
    if found:
        # Prefer the top-level generated suite over remote-result copies.
        found.sort(key=lambda p: (len(p.parts), str(p)))
        return found[0]
    raise FileNotFoundError(f"No benchmark_suite.py found below {bs}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Run generated benchmark_suite.py from a BenchmarkSet.')
    ap.add_argument('--benchmark-set', required=True, help='Path to BenchmarkSet root')
    ap.add_argument('--python', default=sys.executable, help='Python executable for the generated suite')
    ap.add_argument('suite_args', nargs=argparse.REMAINDER, help='Arguments after -- are passed to benchmark_suite.py')
    ns = ap.parse_args(argv)
    bs = Path(ns.benchmark_set).expanduser().resolve()
    suite = find_suite(bs)
    args = list(ns.suite_args or [])
    if args and args[0] == '--':
        args = args[1:]
    cmd = [ns.python, str(suite)] + args
    print('[run-generated-suite] benchmark_set:', bs)
    print('[run-generated-suite] suite:', suite)
    print('[run-generated-suite] cwd:', suite.parent)
    print('[run-generated-suite] cmd:', ' '.join(map(str, cmd)))
    return subprocess.call(cmd, cwd=str(suite.parent), env=os.environ.copy())


if __name__ == '__main__':
    raise SystemExit(main())
