#!/usr/bin/env python3
"""Smoke/diagnostic for Splitpoint boundary transfer modes.

This does not execute Hailo or TensorRT.  It isolates host-side payload size,
contiguity/cast costs and the amount of data each boundary contract would move.
Use it before implementing/validating a real quantized Stage1→Stage2 contract.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _norm_case(case: str) -> str:
    s = str(case).strip()
    if s.startswith('b'):
        n = s[1:]
    else:
        n = s
    return f"b{int(n):03d}" if n.isdigit() else s


def _load_contract(root: Path, case: str) -> dict[str, Any]:
    case = _norm_case(case)
    p = root / 'io_contracts' / case / 'io_contract.json'
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}; run scripts/export_split_io_contracts.py first")
    return json.loads(p.read_text(encoding='utf-8'))


def _bench(fn, warmup: int, runs: int) -> list[float]:
    for _ in range(max(0, warmup)):
        fn()
    vals = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return vals


def _stats(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, dtype=np.float64)
    return {
        'mean_ms': float(np.mean(a)),
        'min_ms': float(np.min(a)),
        'max_ms': float(np.max(a)),
        'p95_ms': float(np.percentile(a, 95)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Smoke boundary transfer/cast costs for split IO contracts.')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--case', action='append', required=True)
    ap.add_argument('--runs', type=int, default=200)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--json-out', default='')
    args = ap.parse_args()

    root = Path(args.benchmark_set).expanduser().resolve()
    results: dict[str, Any] = {'benchmark_set': str(root), 'cases': []}
    for case in args.case:
        c = _load_contract(root, case)
        tensors = c.get('producer_outputs') or c.get('consumer_inputs') or []
        if not tensors:
            raise RuntimeError(f'contract for {case} contains no tensors')
        # Current tool supports one boundary tensor best; for multiple tensors sum/report all.
        case_payload: dict[str, Any] = {'case': _norm_case(case), 'tensors': [], 'modes': []}
        total_elems = 0
        for t in tensors:
            shape = [int(x) for x in t.get('shape') or []]
            elems = int(np.prod(shape)) if shape else int(t.get('element_count') or 0)
            total_elems += elems
            case_payload['tensors'].append({'name': t.get('name'), 'shape': shape, 'element_count': elems})
        # representative shape: flatten all tensors into one contiguous payload
        modes = [
            ('canonical_float32', np.float32, np.float32, 'current safe generic runner contract'),
            ('canonical_float16', np.float16, np.float32, 'reduced host payload; current fp32 Stage2 would still need cast back'),
            ('quantized_int8_contract', np.uint8, np.uint8, 'target performance contract; valid only if Stage2 engine expects same quantized tensor'),
        ]
        for mode, src_dtype, dst_dtype, note in modes:
            src = (np.random.random(total_elems) * 255).astype(src_dtype)
            def copy_only():
                return np.array(src, copy=True)
            def cast_to_stage2():
                return np.ascontiguousarray(src.astype(dst_dtype, copy=False))
            copy_stats = _stats(_bench(copy_only, args.warmup, args.runs))
            cast_stats = _stats(_bench(cast_to_stage2, args.warmup, args.runs))
            payload = {
                'mode': mode,
                'src_dtype': str(np.dtype(src_dtype)),
                'stage2_dtype_for_smoke': str(np.dtype(dst_dtype)),
                'bytes': int(src.nbytes),
                'mib': float(src.nbytes / (1024.0 * 1024.0)),
                'copy_only': copy_stats,
                'cast_or_prepare_for_stage2': cast_stats,
                'note': note,
            }
            case_payload['modes'].append(payload)
            print(json.dumps({'case': _norm_case(case), **payload}, ensure_ascii=False))
        results['cases'].append(case_payload)
    out = Path(args.json_out) if args.json_out else (root / 'io_contracts' / 'boundary_transfer_smoke.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[boundary-smoke] wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
