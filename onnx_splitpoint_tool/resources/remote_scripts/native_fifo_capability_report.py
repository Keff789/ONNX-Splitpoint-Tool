#!/usr/bin/env python3
"""Summarize native FIFO fastpath coverage for a BenchmarkSet.

The report is intentionally conservative, but it uses all available evidence:
  * explicit io_contracts/<case>/io_contract.json
  * io_contracts/summary.json
  * native_trt/<case>/part2/<precision>/native_trt_meta.json
  * existing native_pipeline/<case>/.../native_fifo_results.json

This avoids marking a case as unsupported merely because the standalone
io_contract export was not run, while still explaining what is missing for
cases that have not been prepared yet.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _case_dirs(bs: Path) -> list[Path]:
    return sorted([p for p in bs.iterdir() if p.is_dir() and p.name.startswith("b") and p.name[1:].isdigit()])


def _infer_contract(bs: Path, case: str) -> dict:
    p = bs / "io_contracts" / case / "io_contract.json"
    if p.exists():
        return _load_json(p)
    summ = _load_json(bs / "io_contracts" / "summary.json")
    for row in list(summ.get("cases") or []):
        if isinstance(row, dict) and str(row.get("case_id") or "") == case:
            return row
    return {}


def _meta_for(bs: Path, case: str, precision: str) -> dict:
    paths = [
        bs / "native_trt" / case / "part2" / precision / "native_trt_meta.json",
    ]
    if precision == "uint8_cast_fp16":
        paths.append(bs / "native_trt" / case / "part2" / "uint8_cast_fp16" / "native_trt_meta.json")
    for p in paths:
        if p.exists():
            return _load_json(p)
    return {}


def _engine_for(bs: Path, case: str, precision: str) -> Path:
    if precision == "uint8_cast_fp16":
        return bs / "native_trt" / case / "part2" / "uint8_cast_fp16" / "part2_uint8_cast_fp16.engine"
    return bs / "native_trt" / case / "part2" / precision / f"part2_{precision}.engine"


def _result_for(bs: Path, case: str, precision: str) -> Path:
    return bs / "native_pipeline" / case / "hailo_to_trt" / precision / "native_fifo_results.json"


def _fallback_result_json(bs: Path, case: str) -> dict:
    # Native FIFO runs also write a benchmark-compatible row in the BenchmarkSet root.
    br = bs / f"benchmark_results_native_fifo_{case}.json"
    payload = _load_json(br) if br.exists() else {}
    rows = payload.get("results") or []
    if rows and isinstance(rows[0], dict):
        r = rows[0]
        return {
            "ok": bool(r.get("runtime_ok", True)),
            "fps_makespan": r.get("native_fifo_fps_makespan") or r.get("pipeline_fps_selected"),
            "paper_equivalent_fps": r.get("native_fifo_paper_equivalent_fps"),
            "handoff_ms": r.get("native_fifo_handoff_ms"),
            "trt_input_dtype": r.get("trt_input_dtype"),
            "trt_input_bytes": r.get("trt_input_bytes"),
            "native_fifo_results_json": r.get("native_fifo_results_json"),
        }
    return {}


def _tensor_summary(contract: dict, meta: dict, result_json: dict) -> tuple[int, str, str, int, str]:
    # Preferred: explicit producer/consumer IO contract.
    outs = contract.get("producer_outputs") or contract.get("consumer_inputs") or []
    if outs:
        first = outs[0] if isinstance(outs[0], dict) else {}
        shape = first.get("shape") or []
        dtype = str(first.get("dtype") or first.get("elem_type") or "")
        bytes_int8 = int(first.get("bytes_int8") or first.get("bytes_uint8") or 0)
        return len(outs), str(first.get("name") or ""), f"{dtype} {shape}", bytes_int8, "io_contract"

    # Fallback: native TRT meta inputs.  For a part2 engine this describes the
    # boundary input that the native FIFO consumer will receive.
    inputs = meta.get("inputs") or []
    if inputs:
        first = inputs[0] if isinstance(inputs[0], dict) else {}
        shape = first.get("shape") or []
        dtype = str(first.get("elem_type") or first.get("dtype") or "")
        n = 1
        try:
            for d in shape:
                n *= int(d)
        except Exception:
            n = 0
        elem_bytes = 1 if dtype.upper() in ("UINT8", "INT8", "BOOL") else 2 if dtype.upper() in ("FLOAT16", "HALF", "FP16") else 4
        bytes_int8 = n if n else int(first.get("bytes_int8") or 0)
        return len(inputs), str(first.get("name") or ""), f"{dtype} {shape} (from native_trt_meta)", bytes_int8, "native_trt_meta"

    # Fallback for already executed native FIFO results.
    if result_json.get("ok"):
        nbytes = int(result_json.get("trt_input_bytes") or 0)
        dtype = str(result_json.get("trt_input_dtype") or "")
        return 1, str(result_json.get("trt_input_name") or ""), f"{dtype} input_bytes={nbytes} (from existing native_fifo_results)", nbytes if dtype in ("uint8", "int8") else 0, "native_fifo_results"

    return 0, "", "", 0, "missing"


def main() -> int:
    ap = argparse.ArgumentParser(description="Native FIFO capability report for a BenchmarkSet")
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--precision", default="uint8_cast_fp16")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--run-smoke-supported", action="store_true", help="Run short native FIFO smoke for supported cases that do not yet have results.")
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--queue-depth", type=int, default=3)
    ap.add_argument("--hailo-format", default="uint8")
    ns = ap.parse_args()
    bs = Path(ns.benchmark_set).expanduser().resolve()
    out_dir = Path(ns.out_dir).expanduser().resolve() if ns.out_dir else bs / "analysis_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cdir in _case_dirs(bs):
        cid = cdir.name
        contract = _infer_contract(bs, cid)
        engine = _engine_for(bs, cid, ns.precision)
        meta = _meta_for(bs, cid, ns.precision)
        result = _result_for(bs, cid, ns.precision)
        result_json = _load_json(result) if result.exists() else {}
        if not result_json:
            result_json = _fallback_result_json(bs, cid)
        n_tensors, tensor_name, tensor_desc, int8_bytes, contract_source = _tensor_summary(contract, meta, result_json)
        part1_hef = cdir / "hailo" / ns.hw_arch / "part1" / "compiled.hef"
        alias_paths = []
        if ns.hw_arch == "hailo8":
            alias_paths.append(cdir / "hailo" / "hailo" / "part1" / "compiled.hef")
        if ns.hw_arch in ("hailo10", "hailo10h"):
            alias_paths.extend([
                cdir / "hailo" / "hailo10" / "part1" / "compiled.hef",
                cdir / "hailo" / "hailo10h" / "part1" / "compiled.hef",
            ])
        if not part1_hef.exists():
            for p in alias_paths:
                if p.exists():
                    part1_hef = p; break
        native_fifo_ok = bool(result_json.get("ok"))
        supported = part1_hef.exists() and engine.exists() and (n_tensors == 1 or native_fifo_ok)
        reason_parts = []
        if not part1_hef.exists(): reason_parts.append("missing_hailo_part1_hef")
        if not engine.exists(): reason_parts.append("missing_native_trt_part2_engine")
        if not (n_tensors == 1 or native_fifo_ok): reason_parts.append(f"boundary_tensor_count={n_tensors}")
        reason = "ok" if supported else "; ".join(reason_parts)
        action = "ready"
        if not engine.exists() and part1_hef.exists():
            action = f"build_native_trt_part2:{ns.precision}"
        if supported and not native_fifo_ok:
            action = "run_native_fifo_smoke"
        if not part1_hef.exists():
            action = "build_hailo_part1_hef"
        row = {
            "case_id": cid,
            "native_fifo_supported": supported,
            "unsupported_reason": reason,
            "recommended_action": action,
            "boundary_tensor_count": n_tensors,
            "boundary_tensor": tensor_name,
            "boundary_desc": tensor_desc,
            "boundary_contract_source": contract_source,
            "boundary_int8_bytes": int8_bytes,
            "part1_hef": str(part1_hef) if part1_hef.exists() else "",
            "part2_engine": str(engine) if engine.exists() else "",
            "native_fifo_result": str(result) if result.exists() else str(result_json.get("native_fifo_results_json") or ""),
            "native_fifo_ok": native_fifo_ok,
            "native_fifo_fps_makespan": result_json.get("fps_makespan"),
            "native_fifo_paper_fps": result_json.get("paper_equivalent_fps"),
            "native_fifo_handoff_ms": result_json.get("handoff_ms"),
        }
        rows.append(row)

    json_path = out_dir / "native_fifo_capability_report.json"
    csv_path = out_dir / "native_fifo_capability_report.csv"
    md_path = out_dir / "native_fifo_capability_report.md"
    json_path.write_text(json.dumps({"benchmark_set": str(bs), "hw_arch": ns.hw_arch, "precision": ns.precision, "cases": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    lines = ["# Native FIFO capability report", "", f"BenchmarkSet: `{bs}`", f"HW arch: `{ns.hw_arch}`", f"Precision: `{ns.precision}`", "", "| case | supported | action | reason | boundary | source | int8 bytes | result fps | handoff ms |", "|---|---:|---|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case_id']} | {str(r['native_fifo_supported']).lower()} | {r['recommended_action']} | {r['unsupported_reason']} | {r['boundary_tensor']} {r['boundary_desc']} | {r['boundary_contract_source']} | {r['boundary_int8_bytes']} | {r['native_fifo_fps_makespan'] or ''} | {r['native_fifo_handoff_ms'] or ''} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "cases": len(rows), "supported": sum(1 for r in rows if r["native_fifo_supported"]), "json": str(json_path), "csv": str(csv_path), "md": str(md_path)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
