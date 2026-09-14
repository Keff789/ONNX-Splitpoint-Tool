#!/usr/bin/env python3
"""Create a Native-FIFO exploration/scout report from an EvaluationRun or BenchmarkSet tree.

The script is intentionally conservative: it does not require the benchmark sets to
already be copied to the target NX. It discovers per-model BenchmarkSets locally,
extracts cases, and emits suggested copy/smoke commands.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _case_sort_key(name: str) -> Tuple[int, str]:
    if name.startswith("b") and name[1:].isdigit():
        return (int(name[1:]), name)
    return (10**9, name)


def _find_benchmark_sets(root: Path) -> List[Path]:
    """Find benchmark set roots. Prefer runnable legacy_suite roots over containers."""
    candidates: List[Path] = []

    for pat in [
        "models/*/benchmark_set/legacy_suite",
        "models/*/benchmark_set",
        "models/*/benchmark_results/remote_diagnostics",
        "models/*/benchmark_results/remote_diagnostics/lean_bundle",
        "models/*/benchmark_results/*/lean_bundle",
        "models/*/benchmark_results/*/results/lean_bundle",
    ]:
        for p in root.glob(pat):
            if (p / "benchmark_set.json").exists():
                candidates.append(p)

    if (root / "benchmark_set.json").exists():
        candidates.append(root)

    if not candidates:
        for js in root.rglob("benchmark_set.json"):
            s = str(js)
            if "/energy/" in s or "/collector_storage/" in s or "/processed/" in s:
                continue
            candidates.append(js.parent)

    # exact de-dup
    tmp: List[Path] = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            tmp.append(p)

    def mname(p: Path) -> str:
        try:
            parts = p.relative_to(root).parts
            if len(parts) >= 2 and parts[0] == "models":
                return parts[1]
        except Exception:
            pass
        return p.name

    # Prefer model/benchmark_set/legacy_suite if both parent and child are found.
    by_model: Dict[str, Path] = {}
    for p in tmp:
        m = mname(p)
        old = by_model.get(m)
        if old is None:
            by_model[m] = p
        elif p.name == "legacy_suite" and old.name != "legacy_suite":
            by_model[m] = p
        elif (p.name == old.name) and len(str(p)) < len(str(old)):
            by_model[m] = p
    return sorted(by_model.values(), key=lambda x: str(x))

def _model_name_from_bs(run_dir: Path, bs: Path) -> str:
    try:
        rel = bs.relative_to(run_dir)
        parts = rel.parts
        if len(parts) >= 2 and parts[0] == "models":
            return parts[1]
    except Exception:
        pass
    meta = _load_json(bs / "benchmark_set.json")
    for key in ("model_name", "model", "name"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return Path(v).stem
    return bs.name


def _find_cases(bs: Path) -> List[Path]:
    cases = [p for p in bs.iterdir() if p.is_dir() and p.name.startswith("b") and (p / "split_manifest.json").exists()]
    if not cases:
        # Some lean bundles only contain case_reports; still list b* dirs if present.
        cases = [p for p in bs.iterdir() if p.is_dir() and p.name.startswith("b")]
    return sorted(cases, key=lambda p: _case_sort_key(p.name))


def _find_first(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _boundary_info(case_dir: Path, bs: Path, case_id: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"boundary_tensor_count": 0, "boundary_tensor": "", "boundary_desc": "", "boundary_int8_bytes": None}
    # Prefer explicit IO contract if present.
    for p in [
        bs / "io_contracts" / case_id / "io_contract.json",
        case_dir / "io_contract.json",
    ]:
        d = _load_json(p)
        if d:
            prod = d.get("producer_outputs") or []
            cons = d.get("consumer_inputs") or []
            arr = prod or cons
            info["boundary_tensor_count"] = len(arr)
            if len(arr) == 1:
                t = arr[0]
                info["boundary_tensor"] = str(t.get("name", ""))
                shape = t.get("shape")
                info["boundary_desc"] = f"{t.get('dtype','?')} {shape}"
                info["boundary_int8_bytes"] = t.get("bytes_int8") or t.get("bytes_native")
            return info
    # Summary contract fallback.
    summary = _load_json(bs / "io_contracts" / "summary.json")
    for c in summary.get("cases", []) if isinstance(summary.get("cases"), list) else []:
        if c.get("case_id") == case_id:
            arr = c.get("producer_outputs") or c.get("consumer_inputs") or []
            info["boundary_tensor_count"] = len(arr)
            if len(arr) == 1:
                t = arr[0]
                info["boundary_tensor"] = str(t.get("name", ""))
                info["boundary_desc"] = f"{t.get('dtype','?')} {t.get('shape')}"
                info["boundary_int8_bytes"] = t.get("bytes_int8") or t.get("bytes_native")
            return info
    # Native TRT meta can prove single input.
    for prec in ["uint8_cast_fp16", "fp16", "fp32", "int8"]:
        meta = _load_json(bs / "native_trt" / case_id / "part2" / prec / "native_trt_meta.json")
        if meta:
            inputs = meta.get("inputs") or []
            info["boundary_tensor_count"] = len(inputs)
            if len(inputs) == 1:
                t = inputs[0]
                info["boundary_tensor"] = str(t.get("name", ""))
                shape = t.get("shape")
                elem = t.get("elem_type") or t.get("dtype") or "?"
                info["boundary_desc"] = f"{elem} {shape}"
                try:
                    n = 1
                    for x in shape:
                        n *= int(x)
                    info["boundary_int8_bytes"] = n
                except Exception:
                    pass
            return info
    return info


def _generic_metrics(case_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    # Read all benchmark result jsons near case dir.
    for p in case_dir.glob("benchmark_results_*_auto.json"):
        d = _load_json(p)
        if not d:
            continue
        # Schemas vary. Capture small number of useful fields if present.
        for k in ("pipeline_fps", "pipeline_fps_selected", "fps_makespan", "fps_cycle", "composed_ms", "latency_ms"):
            if k in d and k not in metrics:
                metrics[k] = d[k]
    for res in case_dir.glob("results_*/validation_report.json"):
        d = _load_json(res)
        if not d:
            continue
        # Often these nested keys exist.
        for path_key in [
            ("pipeline", "fps"),
            ("throughput_report", "fps_makespan"),
            ("timings", "composed_ms"),
        ]:
            cur: Any = d
            ok = True
            for kk in path_key:
                if isinstance(cur, dict) and kk in cur:
                    cur = cur[kk]
                else:
                    ok = False
                    break
            if ok:
                metrics["/".join(path_key)] = cur
    return metrics


def _native_fifo_result(bs: Path, case_id: str, precision: str) -> Tuple[Optional[Path], Dict[str, Any]]:
    p = bs / "native_pipeline" / case_id / "hailo_to_trt" / precision / "native_fifo_results.json"
    d = _load_json(p)
    if d:
        return p, d
    # Search if precision differs.
    for q in (bs / "native_pipeline" / case_id).rglob("native_fifo_results.json") if (bs / "native_pipeline" / case_id).exists() else []:
        d = _load_json(q)
        if d:
            return q, d
    return None, {}


def _score_row(row: Dict[str, Any]) -> float:
    score = 0.0
    if row.get("part1_hef_exists"):
        score += 3
    if row.get("part2_onnx_exists"):
        score += 1
    if row.get("part2_engine_exists"):
        score += 2
    if row.get("boundary_tensor_count") == 1:
        score += 3
    if row.get("native_fifo_ok"):
        score += 10
    b = row.get("boundary_int8_bytes")
    if isinstance(b, (int, float)) and b > 0:
        # prefer moderate/small boundary
        score += max(0.0, 3.5 - (float(b) / (1024 * 1024)))
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="EvaluationRun directory or BenchmarkSet root")
    ap.add_argument("--top-per-model", type=int, default=3)
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--precision", default="uint8_cast_fp16")
    ap.add_argument("--remote-root", default="/home/nx/native_fifo_evalsets", help="Suggested remote root used in generated copy/smoke commands")
    ap.add_argument("--remote", default="nx@192.168.0.104", help="Suggested SSH target used in generated copy commands")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    reports_dir = run_dir / "reports" if (run_dir / "models").exists() else run_dir / "analysis_tables"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bsets = _find_benchmark_sets(run_dir)
    rows: List[Dict[str, Any]] = []
    for bs in bsets:
        model = _model_name_from_bs(run_dir, bs)
        remote_bs = f"{args.remote_root.rstrip('/')}/{run_dir.name}/{model}/benchmark_set"
        for case_dir in _find_cases(bs):
            case_id = case_dir.name
            binfo = _boundary_info(case_dir, bs, case_id)
            part1_hef = _find_first([
                case_dir / "hailo" / args.hw_arch / "part1" / "compiled.hef",
                case_dir / "hailo" / "hailo8" / "part1" / "compiled.hef",
                case_dir / "hailo" / "hailo10h" / "part1" / "compiled.hef",
                case_dir / "hailo" / "hailo10" / "part1" / "compiled.hef",
            ])
            part2_onnx = _find_first([case_dir / f"{model}_part2_{case_id}.onnx", *case_dir.glob("*_part2_*.onnx")])
            engine = _find_first([
                bs / "native_trt" / case_id / "part2" / args.precision / f"part2_{args.precision}.engine",
                bs / "native_trt" / case_id / "part2" / args.precision / "model.engine",
            ])
            nf_path, nf = _native_fifo_result(bs, case_id, args.precision)
            row: Dict[str, Any] = {
                "model": model,
                "case_id": case_id,
                "benchmark_set": str(bs),
                "remote_benchmark_set": remote_bs,
                "part1_hef_exists": bool(part1_hef),
                "part1_hef": str(part1_hef or ""),
                "part2_onnx_exists": bool(part2_onnx),
                "part2_onnx": str(part2_onnx or ""),
                "part2_engine_exists": bool(engine),
                "part2_engine": str(engine or ""),
                "native_fifo_result": str(nf_path or ""),
                "native_fifo_ok": bool(nf.get("ok")),
                "native_fifo_fps_makespan": nf.get("fps_makespan"),
                "native_fifo_paper_fps": nf.get("paper_equivalent_fps"),
                "native_fifo_handoff_ms": nf.get("handoff_ms"),
                "copy_command": f"rsync -a --info=progress2 {bs}/ {args.remote}:{remote_bs}/",
                "smoke_command": f"python scripts/native_fifo_smoke_matrix.py --benchmark-set {remote_bs} --hw-arch {args.hw_arch} --precision {args.precision} --build-missing-engines --case {case_id} --frames 100 --warmup 10 --queue-depth 3 --hailo-format uint8 --dump-outputs",
            }
            row.update(binfo)
            row.update({f"generic_{k}": v for k, v in _generic_metrics(case_dir).items()})
            row["score"] = _score_row(row)
            rows.append(row)

    # Top per model
    selected: List[Dict[str, Any]] = []
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(str(r.get("model", "?")), []).append(r)
    for model, rs in sorted(by_model.items()):
        selected.extend(sorted(rs, key=lambda r: (-float(r.get("score") or 0), _case_sort_key(str(r.get("case_id", "")))))[: args.top_per_model])

    payload = {
        "run_dir": str(run_dir),
        "benchmark_sets_found": [str(p) for p in bsets],
        "rows_total": len(rows),
        "top_per_model": args.top_per_model,
        "rows": selected,
    }
    json_path = reports_dir / "native_fifo_exploration_candidates.json"
    csv_path = reports_dir / "native_fifo_exploration_candidates.csv"
    md_path = reports_dir / "native_fifo_exploration_candidates.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = [
        "model", "case_id", "score", "boundary_tensor_count", "boundary_tensor", "boundary_int8_bytes",
        "part1_hef_exists", "part2_onnx_exists", "part2_engine_exists", "native_fifo_ok",
        "native_fifo_fps_makespan", "native_fifo_handoff_ms", "benchmark_set", "remote_benchmark_set",
        "copy_command", "smoke_command",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(selected)

    lines = ["# Native FIFO Exploration Candidates", "", f"Benchmark sets found: {len(bsets)}", f"Rows selected: {len(selected)}", ""]
    if selected:
        lines.append("| Model | Case | Score | Boundary | HEF | Engine | Native OK | FPS | Handoff ms |")
        lines.append("|---|---:|---:|---|---|---|---|---:|---:|")
        for r in selected:
            lines.append(f"| {r.get('model')} | {r.get('case_id')} | {float(r.get('score') or 0):.2f} | {r.get('boundary_tensor_count')}× {r.get('boundary_tensor')} ({r.get('boundary_int8_bytes')}) | {r.get('part1_hef_exists')} | {r.get('part2_engine_exists')} | {r.get('native_fifo_ok')} | {r.get('native_fifo_fps_makespan') or ''} | {r.get('native_fifo_handoff_ms') or ''} |")
        lines.append("\n## Suggested copy/smoke commands\n")
        for r in selected:
            lines.append(f"### {r.get('model')} {r.get('case_id')}")
            lines.append("```bash")
            lines.append(str(r.get("copy_command")))
            lines.append(str(r.get("smoke_command")))
            lines.append("```\n")
    else:
        lines.append("No runnable BenchmarkSets/cases were discovered. Check whether the EvaluationRun contains `models/<model>/benchmark_set/legacy_suite` or a copied BenchmarkSet with `benchmark_set.json`.")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"ok": True, "benchmark_sets_found": len(bsets), "rows_total": len(rows), "rows": len(selected), "json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
