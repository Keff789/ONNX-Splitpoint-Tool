#!/usr/bin/env python3
"""Inspect a BenchmarkSet for native FIFO fastpath eligibility.

This is a conservative static check. It does not run hardware. It answers:
- Which cases have Hailo/DeepX part1 artifacts?
- Which cases have TensorRT part2 engines?
- Which boundary contracts look simple enough for the native FIFO fastpath?
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path
from typing import Any


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return None


def _case_dirs(bs: Path):
    return sorted([p for p in bs.iterdir() if p.is_dir() and p.name.startswith('b') and p.name[1:].isdigit()])


def _find_contract(bs: Path, case: str) -> dict:
    for p in [bs/'io_contracts'/case/'io_contract.json', bs/case/'io_contract.json']:
        if p.exists():
            j=_load_json(p)
            if isinstance(j, dict):
                return j
    # minimal fallback from split manifest
    return {}


def _first_boundary(contract: dict) -> dict:
    cins = contract.get('consumer_inputs') or []
    pouts = contract.get('producer_outputs') or []
    if cins:
        return cins[0]
    if pouts:
        return pouts[0]
    return {}


def _has(p: Path) -> bool:
    return p.exists() and p.is_file()


def _scan_case(bs: Path, case_dir: Path, precision: str) -> dict:
    case = case_dir.name
    contract = _find_contract(bs, case)
    boundary = _first_boundary(contract)
    consumer_inputs = contract.get('consumer_inputs') or []
    producer_outputs = contract.get('producer_outputs') or []
    boundary_tensor_count = max(len(consumer_inputs), len(producer_outputs), 0)
    shape = boundary.get('shape')
    dtype = boundary.get('dtype') or boundary.get('elem_type')
    bytes_i8 = boundary.get('bytes_int8') or boundary.get('bytes_uint8')
    bytes_f32 = boundary.get('bytes_float32')

    paths = {
        'hailo8_part1_hef': case_dir/'hailo'/'hailo8'/'part1'/'compiled.hef',
        'hailo10h_part1_hef': case_dir/'hailo'/'hailo10h'/'part1'/'compiled.hef',
        'hailo10_part1_hef': case_dir/'hailo'/'hailo10'/'part1'/'compiled.hef',
        'deepx_part1': case_dir/'deepx'/'part1',
        'trt_part2_fp16': bs/'native_trt'/case/'part2'/'fp16'/'part2_fp16.engine',
        'trt_part2_uint8_cast': bs/'native_trt'/case/'part2'/'uint8_cast_fp16'/'part2_uint8_cast_fp16.engine',
    }
    simple_single_boundary = boundary_tensor_count == 1
    has_h8 = _has(paths['hailo8_part1_hef'])
    has_h10 = _has(paths['hailo10h_part1_hef']) or _has(paths['hailo10_part1_hef'])
    has_trt_fp16 = _has(paths['trt_part2_fp16'])
    has_trt_u8 = _has(paths['trt_part2_uint8_cast'])
    has_deepx = paths['deepx_part1'].exists()

    # conservative eligibility reasons
    def elig_hailo(arch: str) -> tuple[bool, str]:
        if arch == 'hailo8' and not has_h8:
            return False, 'missing_hailo8_part1_hef'
        if arch == 'hailo10h' and not has_h10:
            return False, 'missing_hailo10h_part1_hef'
        if not simple_single_boundary:
            return False, f'unsupported_boundary_count_{boundary_tensor_count}'
        if precision == 'uint8_cast_fp16':
            if not has_trt_u8:
                return False, 'missing_uint8_cast_part2_engine'
            return True, 'supported_raw_uint8_hailo_to_uint8_cast_trt'
        if precision == 'fp16':
            if not has_trt_fp16:
                return False, 'missing_fp16_part2_engine'
            return True, 'supported_float_or_raw_to_float32_bridge_diagnostic'
        return False, 'unsupported_precision'

    h8_ok, h8_reason = elig_hailo('hailo8')
    h10_ok, h10_reason = elig_hailo('hailo10h')
    deepx_ok = False
    deepx_reason = 'not_implemented_deepx_native_fifo_runner'
    if has_deepx:
        deepx_reason = 'deepx_artifact_present_but_native_fifo_runtime_not_implemented'

    return {
        'case': case,
        'boundary_tensor_count': boundary_tensor_count,
        'boundary_name': boundary.get('name'),
        'boundary_shape': shape,
        'boundary_dtype': dtype,
        'bytes_int8_or_uint8': bytes_i8,
        'bytes_float32': bytes_f32,
        'has_hailo8_part1_hef': has_h8,
        'has_hailo10h_part1_hef': has_h10,
        'has_deepx_part1_artifact': has_deepx,
        'has_trt_part2_fp16': has_trt_fp16,
        'has_trt_part2_uint8_cast_fp16': has_trt_u8,
        'hailo8_to_trt_native_fifo_ok': h8_ok,
        'hailo8_to_trt_reason': h8_reason,
        'hailo10h_to_trt_native_fifo_ok': h10_ok,
        'hailo10h_to_trt_reason': h10_reason,
        'deepx_to_trt_native_fifo_ok': deepx_ok,
        'deepx_to_trt_reason': deepx_reason,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Native FIFO fastpath capability matrix for a BenchmarkSet')
    ap.add_argument('--benchmark-set', required=True)
    ap.add_argument('--precision', default='uint8_cast_fp16', choices=['uint8_cast_fp16','fp16'])
    ap.add_argument('--out-dir', default='')
    ns = ap.parse_args()
    bs = Path(ns.benchmark_set).expanduser().resolve()
    rows = [_scan_case(bs, c, ns.precision) for c in _case_dirs(bs)]
    out_dir = Path(ns.out_dir).expanduser().resolve() if ns.out_dir else bs/'analysis_tables'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'benchmark_set': str(bs), 'precision': ns.precision, 'case_count': len(rows), 'rows': rows}
    (out_dir/'native_fifo_capability_matrix.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if rows:
        keys = list(rows[0].keys())
        with (out_dir/'native_fifo_capability_matrix.csv').open('w', newline='', encoding='utf-8') as f:
            w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    md = ['# Native FIFO capability matrix', '', f'BenchmarkSet: `{bs}`', f'Precision: `{ns.precision}`', '', '| Case | Hailo8→TRT | Hailo10H→TRT | DeepX→TRT | Boundary | TRT uint8 bridge |', '|---|---:|---:|---:|---|---:|']
    for r in rows:
        b = f"{r.get('boundary_name')} {r.get('boundary_shape')}"
        md.append(f"| {r['case']} | {r['hailo8_to_trt_reason']} | {r['hailo10h_to_trt_reason']} | {r['deepx_to_trt_reason']} | `{b}` | {r['has_trt_part2_uint8_cast_fp16']} |")
    (out_dir/'native_fifo_capability_matrix.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'out_dir': str(out_dir), 'case_count': len(rows), 'supported_hailo8': sum(1 for r in rows if r['hailo8_to_trt_native_fifo_ok']), 'supported_hailo10h': sum(1 for r in rows if r['hailo10h_to_trt_native_fifo_ok'])}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
