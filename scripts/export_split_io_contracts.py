#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import onnx  # type: ignore
except Exception as exc:  # pragma: no cover
    onnx = None  # type: ignore
    _ONNX_IMPORT_ERROR = exc
else:
    _ONNX_IMPORT_ERROR = None

_TYPE_BYTES = {
    "FLOAT": 4,
    "FLOAT16": 2,
    "BFLOAT16": 2,
    "DOUBLE": 8,
    "INT64": 8,
    "UINT64": 8,
    "INT32": 4,
    "UINT32": 4,
    "INT16": 2,
    "UINT16": 2,
    "INT8": 1,
    "UINT8": 1,
    "BOOL": 1,
}


def _norm_case(case: str) -> str:
    case = str(case).strip()
    if case.startswith("b") and case[1:].isdigit():
        return f"b{int(case[1:]):03d}"
    if case.isdigit():
        return f"b{int(case):03d}"
    return case


def _load(path: Path):
    if onnx is None:
        raise RuntimeError(f"onnx import failed: {_ONNX_IMPORT_ERROR}")
    return onnx.load(str(path), load_external_data=False)


def _value_info_map(model) -> dict[str, Any]:
    info: dict[str, Any] = {}
    items = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    for vi in items:
        if not vi.type.HasField("tensor_type"):
            continue
        t = vi.type.tensor_type
        elem = onnx.TensorProto.DataType.Name(t.elem_type) if t.elem_type else "UNKNOWN"
        shape: list[int | str] = []
        for d in t.shape.dim:
            if d.dim_value and int(d.dim_value) > 0:
                shape.append(int(d.dim_value))
            elif d.dim_param:
                shape.append(str(d.dim_param))
            else:
                shape.append("?")
        info[vi.name] = {"name": vi.name, "dtype": elem, "shape": shape}
    return info


def _tensor_specs(path: Path, which: str) -> list[dict[str, Any]]:
    model = _load(path)
    initializers = {i.name for i in model.graph.initializer}
    values = list(model.graph.output if which == "outputs" else model.graph.input)
    out: list[dict[str, Any]] = []
    for vi in values:
        if which == "inputs" and vi.name in initializers:
            continue
        if not vi.type.HasField("tensor_type"):
            continue
        t = vi.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(t.elem_type) if t.elem_type else "UNKNOWN"
        shape: list[int | str] = []
        known_shape: list[int] = []
        for d in t.shape.dim:
            if d.dim_value and int(d.dim_value) > 0:
                v = int(d.dim_value)
                shape.append(v)
                known_shape.append(v)
            elif d.dim_param:
                shape.append(str(d.dim_param))
            else:
                shape.append("?")
        elem_count = math.prod(known_shape) if known_shape and len(known_shape) == len(shape) else None
        dtype_bytes = _TYPE_BYTES.get(dtype)
        bytes_native = int(elem_count * dtype_bytes) if elem_count is not None and dtype_bytes else None
        out.append({
            "name": vi.name,
            "dtype": dtype,
            "shape": shape,
            "element_count": elem_count,
            "bytes_native": bytes_native,
            "bytes_float32": int(elem_count * 4) if elem_count is not None else None,
            "bytes_float16": int(elem_count * 2) if elem_count is not None else None,
            "bytes_int8": int(elem_count) if elem_count is not None else None,
        })
    return out


def _find_case_dirs(root: Path, cases: list[str]) -> list[Path]:
    if cases:
        result = []
        for c in cases:
            n = _norm_case(c)
            p = root / n
            if not p.is_dir() and n.startswith("b"):
                p = root / f"b{int(n[1:])}"
            if not p.is_dir():
                raise FileNotFoundError(f"case not found: {c}")
            result.append(p)
        return result
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("b")])


def _find_part(case_dir: Path, name: str) -> Path | None:
    cands = sorted(case_dir.glob(f"*{name}*.onnx"))
    return cands[0] if cands else None


def _fingerprints(tensors: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for t in tensors:
        elems = t.get("element_count")
        shape = [x for x in t.get("shape", []) if isinstance(x, int)]
        if elems == 3276800 or sorted(shape[-3:]) == sorted([512, 80, 80]):
            tags.append("yolov7_paper_512x80x80_candidate")
        if elems == 1638400 or sorted(shape[-3:]) == sorted([256, 80, 80]):
            tags.append("yolov7_nearby_256x80x80_candidate")
        if elems == 6553600 or sorted(shape[-3:]) == sorted([1024, 80, 80]):
            tags.append("large_1024x80x80_candidate")
    return sorted(set(tags))


def _recommend_modes(tensors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    total_f32 = sum(int(t.get("bytes_float32") or 0) for t in tensors)
    total_f16 = sum(int(t.get("bytes_float16") or 0) for t in tensors)
    total_i8 = sum(int(t.get("bytes_int8") or 0) for t in tensors)
    recs.append({"mode": "canonical_float32", "estimated_bytes": total_f32, "status": "always_supported_baseline"})
    recs.append({"mode": "canonical_float16", "estimated_bytes": total_f16, "status": "candidate_if_stage2_accepts_fp16_or_cast_is_cheap"})
    recs.append({
        "mode": "quantized_int8_contract",
        "estimated_bytes": total_i8,
        "status": "requires_explicit_stage1_scale_zero_point_and_stage2_int8_contract",
    })
    return recs


def main() -> int:
    ap = argparse.ArgumentParser(description="Export split IO/boundary contracts for a benchmark set.")
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--case", action="append", default=[], help="Case id such as b066. Repeatable. Default: all cases.")
    ap.add_argument("--write-per-case", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--json-out", default="", help="Default: <benchmark-set>/io_contracts/summary.json")
    args = ap.parse_args()
    root = Path(args.benchmark_set).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    out_root = root / "io_contracts"
    out_root.mkdir(parents=True, exist_ok=True)
    cases = _find_case_dirs(root, args.case)
    summary: dict[str, Any] = {"benchmark_set": str(root), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cases": []}
    for cdir in cases:
        part1 = _find_part(cdir, "part1")
        part2 = _find_part(cdir, "part2")
        if not part1 or not part2:
            continue
        p1_outputs = _tensor_specs(part1, "outputs")
        p2_inputs = _tensor_specs(part2, "inputs")
        # Direct-name matches are uncommon after Hailo/TensorRT export, but useful when present.
        p1_names = {t["name"] for t in p1_outputs}
        p2_names = {t["name"] for t in p2_inputs}
        matched = sorted(p1_names & p2_names)
        crossing = p1_outputs if p1_outputs else p2_inputs
        contract = {
            "case_id": cdir.name,
            "part1_onnx": str(part1),
            "part2_onnx": str(part2),
            "producer_outputs": p1_outputs,
            "consumer_inputs": p2_inputs,
            "name_matches": matched,
            "fingerprints": _fingerprints(crossing),
            "recommended_boundary_modes": _recommend_modes(crossing),
            "notes": [
                "canonical_float32 is the safe validation contract used by the generic runner.",
                "quantized_int8_contract is a performance target, not automatically valid unless stage2 was built for the same quantized boundary.",
            ],
        }
        if args.write_per_case:
            c_out = out_root / cdir.name
            c_out.mkdir(parents=True, exist_ok=True)
            (c_out / "io_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
        summary["cases"].append(contract)
    json_out = Path(args.json_out).expanduser().resolve() if args.json_out else out_root / "summary.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[io-contract] cases={len(summary['cases'])} wrote {json_out}")
    for c in summary["cases"]:
        rec = {r["mode"]: r["estimated_bytes"] for r in c.get("recommended_boundary_modes", [])}
        print(json.dumps({"case": c["case_id"], "fingerprints": c.get("fingerprints", []), "bytes": rec}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
