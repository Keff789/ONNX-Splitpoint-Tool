from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx import TensorProto

from .cache_verify_policy import (
    cache_miss_blocked_message,
    compiler_dispatch_forbidden,
)

_ONNX_DTYPE_TO_NP = {
    TensorProto.FLOAT: np.float32,
    TensorProto.FLOAT16: np.float16,
    TensorProto.DOUBLE: np.float64,
    TensorProto.INT8: np.int8,
    TensorProto.UINT8: np.uint8,
    TensorProto.INT16: np.int16,
    TensorProto.UINT16: np.uint16,
    TensorProto.INT32: np.int32,
    TensorProto.UINT32: np.uint32,
    TensorProto.INT64: np.int64,
    TensorProto.UINT64: np.uint64,
    TensorProto.BOOL: np.bool_,
}


def _dtype_name(elem_type: int) -> str:
    try:
        return TensorProto.DataType.Name(int(elem_type))
    except Exception:
        return str(elem_type)


def _dtype_nbytes(elem_type: int) -> Optional[int]:
    np_dt = _ONNX_DTYPE_TO_NP.get(int(elem_type))
    if np_dt is None:
        return None
    return int(np.dtype(np_dt).itemsize)


def tensor_value_info_to_dict(v: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": getattr(v, "name", "")}
    try:
        t = v.type.tensor_type
        elem = int(t.elem_type)
        dims: List[Any] = []
        static = True
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            elif d.HasField("dim_param"):
                dims.append(str(d.dim_param))
                static = False
            else:
                dims.append(None)
                static = False
        out.update({
            "elem_type": elem,
            "dtype": _dtype_name(elem),
            "shape": dims,
            "static_shape": static,
        })
        nb = _dtype_nbytes(elem)
        if static and nb is not None and dims:
            numel = 1
            for x in dims:
                numel *= int(x)
            out["numel"] = int(numel)
            out["bytes"] = int(numel * nb)
        else:
            out["numel"] = None
            out["bytes"] = None
    except Exception as exc:
        out.update({"error": f"{type(exc).__name__}: {exc}"})
    return out


def load_io_info(onnx_path: Path) -> Dict[str, Any]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return {
        "path": str(onnx_path),
        "inputs": [tensor_value_info_to_dict(v) for v in model.graph.input],
        "outputs": [tensor_value_info_to_dict(v) for v in model.graph.output],
    }


def canonical_case_id(case: str | int) -> str:
    s = str(case).strip().lower()
    if not s:
        raise ValueError("empty case id")
    if s.startswith("b"):
        s = s[1:]
    return f"b{int(s):03d}"


def find_case_dir(benchmark_set: Path, case: str | int) -> Path:
    cid = canonical_case_id(case)
    candidates = [benchmark_set / cid, benchmark_set / cid.replace("b0", "b", 1)]
    # Also tolerate b66 instead of b066 and nested legacy_suite directories.
    try:
        n = int(cid[1:])
        candidates.extend([benchmark_set / f"b{n}", benchmark_set / "legacy_suite" / cid, benchmark_set / "legacy_suite" / f"b{n}"])
    except Exception:
        pass
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    hits = sorted(benchmark_set.rglob(f"{cid}")) + sorted(benchmark_set.rglob(f"b{int(cid[1:])}"))
    for p in hits:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"Could not find case directory {cid} below {benchmark_set}")


def find_onnx_files(benchmark_set: Path, case: Optional[str | int] = None) -> Dict[str, Optional[Path]]:
    out: Dict[str, Optional[Path]] = {"full": None, "part1": None, "part2": None}
    # full model
    model_hits = sorted((benchmark_set / "models").glob("*.onnx")) if (benchmark_set / "models").exists() else []
    if not model_hits:
        model_hits = [p for p in sorted(benchmark_set.glob("*.onnx")) if "part" not in p.name.lower()]
    if model_hits:
        out["full"] = model_hits[0]
    if case is not None:
        cdir = find_case_dir(benchmark_set, case)
        cid_num = int(canonical_case_id(case)[1:])
        part1 = sorted(cdir.glob(f"*part1*b{cid_num}.onnx")) + sorted(cdir.glob("*part1*.onnx"))
        part2 = sorted(cdir.glob(f"*part2*b{cid_num}.onnx")) + sorted(cdir.glob("*part2*.onnx"))
        out["part1"] = part1[0] if part1 else None
        out["part2"] = part2[0] if part2 else None
    return out


def shape_arg_from_io(io_info: Dict[str, Any], overrides: Optional[Dict[str, str]] = None) -> str:
    overrides = overrides or {}
    parts: List[str] = []
    for inp in io_info.get("inputs") or []:
        name = str(inp.get("name"))
        if name in overrides:
            shape_s = overrides[name]
        else:
            dims = inp.get("shape") or []
            normalized: List[int] = []
            for i, d in enumerate(dims):
                if isinstance(d, int) and d > 0:
                    normalized.append(int(d))
                else:
                    # Conservative fallback: batch=1, unknown feature dims are not guessable.
                    normalized.append(1)
            shape_s = "x".join(str(x) for x in normalized)
        if name and shape_s:
            parts.append(f"{name}:{shape_s}")
    return ",".join(parts)


def parse_shape_overrides(items: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items or []:
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"shape override must be name:1xCxHxW, got {item!r}")
        name, val = item.split(":", 1)
        out[name.strip()] = val.strip().replace(",", "x")
    return out


def find_trtexec(explicit: str = "") -> str:
    if explicit:
        p = shutil.which(explicit) or explicit
        if Path(p).exists() or shutil.which(p):
            return p
        raise FileNotFoundError(f"trtexec not found: {explicit}")
    p = shutil.which("trtexec")
    if not p:
        for cand in ["/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec", "/usr/local/bin/trtexec"]:
            if Path(cand).exists():
                return cand
        raise FileNotFoundError("trtexec not found in PATH or common Jetson locations")
    return p


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    command = [str(value) for value in cmd]
    if compiler_dispatch_forbidden() and any(
        value == flag or value.startswith(flag + "=")
        for value in command[1:]
        for flag in ("--onnx", "--saveEngine", "--buildOnly")
    ):
        raise RuntimeError(cache_miss_blocked_message(
            "tensorrt_build", "native_trt_utils rejected a build command"
        ))
    t0 = time.time()
    proc = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    elapsed = time.time() - t0
    return {"returncode": int(proc.returncode), "elapsed_s": elapsed, "cmd": command, "stdout": proc.stdout}


def parse_trtexec_output(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    patterns = {
        "throughput_qps": r"Throughput:\s*([0-9.]+)\s*qps",
        "latency_mean_ms": r"Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "latency_median_ms": r"Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*[0-9.]+\s*ms,\s*median\s*=\s*([0-9.]+)\s*ms",
        "gpu_compute_mean_ms": r"GPU Compute Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "enqueue_mean_ms": r"Enqueue Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "h2d_mean_ms": r"H2D Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "d2h_mean_ms": r"D2H Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                out[key] = float(m.group(1))
            except Exception:
                pass
    # Fallbacks for newer / older trtexec output variants.
    m = re.search(r"mean\s*=\s*([0-9.]+)\s*ms", text, re.IGNORECASE)
    if m and "latency_mean_ms" not in out:
        out["latency_mean_ms"] = float(m.group(1))
    return out


def build_trtexec_build_cmd(
    trtexec: str,
    onnx_path: Path,
    engine_path: Path,
    shapes: str,
    precision: str = "fp16",
    workspace_mib: int = 4096,
    timing_cache: Optional[Path] = None,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    if shapes:
        cmd.append(f"--shapes={shapes}")
    precision = precision.lower().strip()
    if precision == "fp16":
        cmd.append("--fp16")
    elif precision == "int8":
        cmd.append("--int8")
    elif precision in {"fp32", "float32"}:
        pass
    else:
        raise ValueError(f"unsupported precision {precision!r}")
    if workspace_mib > 0:
        # TensorRT 10 removed --workspace; --memPoolSize is accepted on TRT10 and
        # keeps Phase-2 smoke tests aligned with the evaluation runner.
        cmd.append(f"--memPoolSize=workspace:{int(workspace_mib)}")
    if timing_cache:
        cmd.append(f"--timingCacheFile={timing_cache}")
    cmd += list(extra or [])
    return cmd


def build_trtexec_run_cmd(
    trtexec: str,
    engine_path: Path,
    duration_s: int = 10,
    warmup_ms: int = 500,
    iterations: int = 0,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd = [trtexec, f"--loadEngine={engine_path}", f"--warmUp={int(warmup_ms)}"]
    if iterations and iterations > 0:
        cmd.append(f"--iterations={int(iterations)}")
    else:
        cmd.append(f"--duration={int(duration_s)}")
    cmd += list(extra or [])
    return cmd


def boundary_contract_summary(part1_onnx: Path, part2_onnx: Path, full_onnx: Optional[Path] = None) -> Dict[str, Any]:
    p1 = load_io_info(part1_onnx)
    p2 = load_io_info(part2_onnx)
    full = load_io_info(full_onnx) if full_onnx else None
    full_inputs = {i.get("name") for i in (full or {}).get("inputs", [])} if full else set()
    p1_inputs = {i.get("name") for i in p1.get("inputs", [])}
    # Candidate feature inputs = part2 inputs not obvious original model inputs.
    feature_inputs = []
    passthrough_inputs = []
    for inp in p2.get("inputs") or []:
        name = inp.get("name")
        if name in full_inputs or name in p1_inputs:
            passthrough_inputs.append(inp)
        else:
            feature_inputs.append(inp)
    mode_rows = []
    for inp in feature_inputs:
        shape = inp.get("shape") or []
        static = bool(inp.get("static_shape"))
        numel = inp.get("numel")
        row = {"name": inp.get("name"), "shape": shape, "static_shape": static, "numel": numel, "onnx_dtype": inp.get("dtype")}
        if static and numel:
            row.update({
                "bytes_float32": int(numel) * 4,
                "bytes_float16": int(numel) * 2,
                "bytes_int8": int(numel),
                "mib_float32": (int(numel) * 4) / (1024 * 1024),
                "mib_float16": (int(numel) * 2) / (1024 * 1024),
                "mib_int8": int(numel) / (1024 * 1024),
            })
        mode_rows.append(row)
    return {
        "part1_onnx": str(part1_onnx),
        "part2_onnx": str(part2_onnx),
        "full_onnx": str(full_onnx) if full_onnx else None,
        "part1_outputs": p1.get("outputs"),
        "part2_inputs": p2.get("inputs"),
        "feature_input_count": len(feature_inputs),
        "passthrough_input_count": len(passthrough_inputs),
        "feature_inputs": feature_inputs,
        "passthrough_inputs": passthrough_inputs,
        "boundary_size_modes": mode_rows,
        "quantized_int8_candidate": bool(len(feature_inputs) == 1 and (mode_rows and mode_rows[0].get("bytes_int8"))),
        "notes": [
            "INT8 boundary is only safe when producer and consumer share an explicit quantization contract (scale/zero_point/layout/order).",
            "This probe estimates the potential transfer size reduction; it does not modify ONNX or build a quantized TensorRT engine.",
        ],
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
