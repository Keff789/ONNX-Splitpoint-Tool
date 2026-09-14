#!/usr/bin/env python3
"""Integrate native FIFO fastpath results into an EvaluationRun-style report.

The Evaluation workflow still produces the generic BenchmarkSet/runner results.
Native FIFO runs are an additional execution mode over staged BenchmarkSets.  This
script scans either an EvaluationRun directory or a staged native_fifo_evalsets
root, finds native FIFO result rows, and writes a consolidated summary with
fallback/support status.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _find_benchmark_sets(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob('benchmark_set.json'):
        b = p.parent
        if (b / 'benchmark_plan.json').exists() or any(b.glob('b*/split_manifest.json')):
            out.append(b)
    # staged roots often use .../<model>/benchmark_set
    return sorted(set(out))


def _model_name_from_bs(bs: Path) -> str:
    try:
        js = _load(bs / 'benchmark_set.json')
        for k in ('model_name', 'model_id', 'model'):
            if js.get(k):
                return str(js[k])
    except Exception:
        pass
    if bs.name == 'benchmark_set' and bs.parent.name:
        return bs.parent.name
    return bs.name


def _native_rows(bs: Path) -> list[dict[str, Any]]:
    model = _model_name_from_bs(bs)
    rows: list[dict[str, Any]] = []
    # benchmark-compatible rows
    for p in sorted(bs.glob('benchmark_results_native_fifo_*.json')):
        js = _load(p)
        for r in js.get('results') or []:
            if isinstance(r, dict):
                rr = dict(r)
                rr.setdefault('model_id', model)
                rr.setdefault('case_id', p.stem.replace('benchmark_results_native_fifo_', ''))
                rr['benchmark_set'] = str(bs)
                rr['native_fifo_row_json'] = str(p)
                rows.append(rr)
    # raw native_fifo_results.json not yet converted
    for p in sorted(bs.glob('native_pipeline/b*/hailo_to_trt/*/native_fifo_results.json')):
        js = _load(p)
        if not js.get('ok'):
            continue
        case = p.parts[-4]
        precision = p.parts[-2]
        if any(str(r.get('case_id')) == case and str(r.get('native_fifo_results_json')) == str(p) for r in rows):
            continue
        rows.append({
            'model_id': model,
            'case_id': case,
            'benchmark_set': str(bs),
            'backend': 'hailo8_to_tensorrt',
            'run_id': 'hailo8_to_trt_native_fifo',
            'native_fifo_enabled': True,
            'native_fifo_precision': precision,
            'native_fifo_boundary_mode': 'raw_uint8_hailo' if precision == 'uint8_cast_fp16' else '',
            'native_fifo_preprocess_ms': js.get('preprocess_ms'),
            'native_fifo_p1_ms': js.get('p1_ms'),
            'native_fifo_handoff_ms': js.get('handoff_ms'),
            'native_fifo_p2_run_ms': js.get('p2_run_ms'),
            'native_fifo_p1_thread_ms': js.get('p1_thread_ms'),
            'native_fifo_p2_thread_ms': js.get('p2_thread_ms'),
            'native_fifo_paper_equivalent_cycle_ms': js.get('paper_equivalent_cycle_ms'),
            'native_fifo_paper_equivalent_fps': js.get('paper_equivalent_fps'),
            'native_fifo_fps_makespan': js.get('fps_makespan'),
            'pipeline_fps_selected': js.get('fps_makespan') or js.get('paper_equivalent_fps'),
            'pipeline_cycle_selected_ms': js.get('paper_equivalent_cycle_ms'),
            'trt_input_dtype': js.get('trt_input_dtype'),
            'trt_input_bytes': js.get('trt_input_bytes'),
            'runtime_ok': True,
            'native_fifo_results_json': str(p),
            'native_fifo_output_manifest': js.get('native_fifo_output_manifest'),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description='Create integrated native FIFO report for EvaluationRun/staged BenchmarkSets')
    ap.add_argument('--root', '--run-dir', dest='root', required=True, help='EvaluationRun dir or staged native_fifo_evalsets root')
    ap.add_argument('--out-dir', default='', help='Defaults to <root>/reports or <root>/analysis_tables')
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    bs_list = _find_benchmark_sets(root)
    rows: list[dict[str, Any]] = []
    for bs in bs_list:
        rows.extend(_native_rows(bs))
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / 'reports' if (root / 'models').exists() else root / 'analysis_tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / 'native_fifo_eval_summary.json'
    csv_path = out_dir / 'native_fifo_eval_summary.csv'
    md_path = out_dir / 'native_fifo_eval_summary.md'
    payload = {'schema': 'onnx-splitpoint/native-fifo-eval-summary', 'schema_version': 1, 'root': str(root), 'benchmark_sets': [str(x) for x in bs_list], 'rows': rows}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with csv_path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    lines = ['# Native FIFO Evaluation Summary', '', f'Root: `{root}`', '', '| model | case | fps | handoff ms | p1 thread ms | p2 thread ms | input bytes | precision | result |', '|---|---|---:|---:|---:|---:|---:|---|---|']
    for r in sorted(rows, key=lambda x: (str(x.get('model_id')), str(x.get('case_id')))):
        lines.append(f"| {r.get('model_id','')} | {r.get('case_id','')} | {r.get('native_fifo_fps_makespan') or r.get('pipeline_fps_selected') or ''} | {r.get('native_fifo_handoff_ms') or ''} | {r.get('native_fifo_p1_thread_ms') or ''} | {r.get('native_fifo_p2_thread_ms') or ''} | {r.get('trt_input_bytes') or ''} | {r.get('native_fifo_precision') or ''} | `{r.get('native_fifo_results_json') or ''}` |")
    if not rows:
        lines.append('')
        lines.append('No native FIFO rows found. Run `native_fifo_smoke_matrix.py` on staged BenchmarkSets first.')
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'benchmark_sets': len(bs_list), 'rows': len(rows), 'json': str(json_path), 'csv': str(csv_path), 'md': str(md_path)}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
