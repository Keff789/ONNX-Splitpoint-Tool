#!/usr/bin/env python3
"""Smoke-test boundary handoff representation costs for a split case.

This is a Phase-3 preparation tool. It does not change model semantics and it does
not claim that INT8 boundary is valid; it estimates host-side data volume and
conversion/copy cost for the same boundary tensor in float32/float16/int8-like
representations. A real INT8 boundary needs scale/zero-point/layout agreement
between the Hailo producer and TensorRT consumer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnx  # type: ignore
except Exception as exc:  # pragma: no cover
    onnx = None
    _ONNX_ERR = exc
else:
    _ONNX_ERR = None


def _case_id(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith('b'):
        s = s[1:]
    return f"b{int(s):03d}"


def _find_part2(root: Path, case: str) -> Path:
    c = root / _case_id(case)
    if not c.exists():
        c = root / f"b{int(str(case).lstrip('b'))}"
    hits = sorted(c.glob('*part2*.onnx'))
    if not hits:
        raise FileNotFoundError(f"part2 ONNX not found below {c}")
    return hits[0]


def _inputs(path: Path) -> list[dict[str, Any]]:
    if onnx is None:
        raise RuntimeError(f"onnx import failed: {_ONNX_ERR}")
    m = onnx.load(str(path), load_external_data=False)
    init_names = {i.name for i in m.graph.initializer}
    out = []
    for v in m.graph.input:
        if v.name in init_names:
            continue
        dims = []
        static = True
        for d in v.type.tensor_type.shape.dim:
            if d.dim_value and int(d.dim_value) > 0:
                dims.append(int(d.dim_value))
            else:
                dims.append(1)
                static = False
        out.append({"name": v.name, "shape": dims, "static": static})
    return out


def _measure(fn, runs: int) -> dict[str, float]:
    vals = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        _ = fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean_ms": float(np.mean(arr)), "min_ms": float(np.min(arr)), "max_ms": float(np.max(arr)), "std_ms": float(np.std(arr))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-set', required=True, type=Path)
    ap.add_argument('--case', required=True)
    ap.add_argument('--runs', type=int, default=200)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    root = args.benchmark_set.expanduser().resolve()
    part2 = _find_part2(root, args.case)
    ins = _inputs(part2)
    # For split part2, the feature boundary is usually the first input not named images.
    feat = next((x for x in ins if x['name'] not in {'images', 'input', 'input_0'}), ins[0])
    shape = [int(x) for x in feat['shape']]
    numel = int(np.prod(shape, dtype=np.int64))
    src_f32 = np.random.random(shape).astype(np.float32)
    # Simulate a quantized producer using a conservative scale. This is only a host-cost probe.
    scale = np.float32(1.0 / 255.0)
    zero_point = np.uint8(0)

    def f32_copy():
        return np.ascontiguousarray(src_f32, dtype=np.float32)
    def f16_cast():
        return np.ascontiguousarray(src_f32.astype(np.float16, copy=False))
    def u8_quant():
        return np.ascontiguousarray(np.clip(np.rint(src_f32 / scale) + int(zero_point), 0, 255).astype(np.uint8))
    def u8_dequant_to_f32():
        q = u8_quant()
        return np.ascontiguousarray((q.astype(np.float32) - int(zero_point)) * scale)

    rows = {
        'canonical_float32': {'bytes': int(numel * 4), **_measure(f32_copy, args.runs)},
        'canonical_float16': {'bytes': int(numel * 2), **_measure(f16_cast, args.runs)},
        'quantized_uint8_pack_only': {'bytes': int(numel), **_measure(u8_quant, args.runs)},
        'quantized_uint8_dequant_to_f32': {'bytes': int(numel * 4), 'wire_bytes': int(numel), **_measure(u8_dequant_to_f32, max(10, args.runs // 4))},
    }
    report = {
        'benchmark_set': str(root),
        'case': _case_id(args.case),
        'part2_onnx': str(part2),
        'boundary_input': feat,
        'numel': numel,
        'modes': rows,
        'notes': [
            'This is a host-side representation/copy/cast smoke test only.',
            'quantized_uint8 is not a valid model contract until scale/zero_point/layout are exported and the TensorRT stage2 engine is built for the same contract.',
        ],
    }
    out = args.out or (root / 'native_trt' / _case_id(args.case) / 'boundary_handoff_smoke.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(out), 'case': report['case'], 'boundary_input': feat, 'modes': rows}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
