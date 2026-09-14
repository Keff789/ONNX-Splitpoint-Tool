#!/usr/bin/env python3
"""Compare Generic Hailo stage1 output against a Native FIFO boundary dump.

Purpose
-------
When the Generic runner is semantically valid but the Native FIFO Hailo->TRT
fastpath is not, this probe separates three cases:

1. Python/Generic Hailo stage1 + ONNX Part2 is semantically good, but the
   Native C++ boundary is bad -> Native HailoRT invocation / vstream contract
   differs from Generic.
2. Both Generic Hailo stage1 + ONNX Part2 and Native boundary + ONNX Part2 are
   bad -> the split is not safe as a Hailo-Part1 -> float-Part2 contract, even
   if full/generic hardware runs are valid.
3. Generic and Native boundaries are close, but TensorRT native output differs
   -> use native_part2_boundary_replay_compare next.

This script runs on the Hailo host. It uses the existing Python HailoBackend
path, the exact input dump recorded by Native FIFO, and the original Part2 ONNX
under ORT for semantic scoring.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Reused by the bounded v31 probe; numeric proximity and exploratory layout
# sweeps do not establish a quality/benchmark binding.

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from hailo_boundary_diagnostics_v27931 import FLAGS as DIAGNOSTIC_FLAGS, json_safe

try:
    import onnxruntime as ort  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"onnxruntime import failed: {type(exc).__name__}: {exc}")

try:
    from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend  # type: ignore
    from onnx_splitpoint_tool.runners._types import RunCfg  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"HailoBackend import failed: {type(exc).__name__}: {exc}")

try:
    from native_boundary_activation_compare import (  # type: ignore
        _find_part1_onnx,
        _input_dump_feed_from_manifest,
        _onnx_io,
    )
    from native_hailo_trt_fifo_from_benchmarkset import _find_hef  # type: ignore
    from native_part2_boundary_layout_sweep import (  # type: ignore
        _find_part2_onnx,
        _layout_candidates,
        _summ,
    )
    from native_boundary_dequant_sweep import _semantic_score_outputs  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"helper import failed: {type(exc).__name__}: {exc}")


def _read_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _dtype_from_manifest(man: dict[str, Any]):
    s = str(man.get("dtype") or "uint8").lower()
    return {
        "uint8": np.uint8,
        "int8": np.int8,
        "float32": np.float32,
        "fp32": np.float32,
        "float16": np.float16,
        "fp16": np.float16,
    }.get(s, np.uint8)


def _load_native_boundary(man_path: Path) -> tuple[dict[str, Any], np.ndarray, Path]:
    man = _read_json(man_path)
    f = Path(str(man.get("file") or "")).expanduser()
    if not f.is_absolute():
        f = (man_path.parent / f).resolve()
    if not f.is_file():
        raise FileNotFoundError(f"boundary file not found: {f}")
    raw = np.fromfile(str(f), dtype=_dtype_from_manifest(man))
    shp = man.get("shape") or man.get("boundary_shape")
    if shp:
        shp_i = [int(x) for x in shp]
        if int(np.prod(shp_i)) != int(raw.size):
            raise RuntimeError(f"boundary shape {shp_i} does not match file elements {raw.size}")
        raw = raw.reshape(shp_i)
    return man, raw, f


def _load_input_dump_hwc(man: dict[str, Any]) -> np.ndarray | None:
    p = man.get("input_dump") or man.get("preprocessed_input_file")
    if not isinstance(p, str) or not p:
        return None
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        base = Path(str(man.get("file") or "")).expanduser().parent
        pp = (base / pp).resolve()
    if not pp.is_file():
        return None
    shp = man.get("input_shape_hwc") or (man.get("preprocess") or {}).get("input_shape_hwc")
    if not shp or len(shp) != 3:
        return None
    h, w, c = [int(x) for x in shp]
    raw = np.fromfile(str(pp), dtype=np.uint8)
    if raw.size != h * w * c:
        return None
    return raw.reshape((h, w, c))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return float("nan")
    sx = float(np.std(x)); sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _tensor_compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a)
    bb = np.asarray(b)
    out: dict[str, Any] = {
        "shape_a": [int(x) for x in aa.shape],
        "shape_b": [int(x) for x in bb.shape],
        "dtype_a": str(aa.dtype),
        "dtype_b": str(bb.dtype),
        "same_shape": bool(tuple(aa.shape) == tuple(bb.shape)),
    }
    if aa.size != bb.size:
        out.update({"same_size": False})
        return out
    x = aa.astype(np.float64, copy=False).reshape(-1)
    y = bb.astype(np.float64, copy=False).reshape(-1)
    d = x - y
    out.update({
        "same_size": True,
        "mean_abs": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(math.sqrt(float(np.mean(d * d)))),
        "corr": _corr(x, y),
        "close_1e_3_1e_2": bool(np.allclose(x, y, rtol=1e-3, atol=1e-2)),
        "a_summary": _summ(aa),
        "b_summary": _summ(bb),
    })
    # affine fit y ~= scale*x + bias, useful for quantization drift
    try:
        vx = float(np.var(x))
        if vx > 0:
            cov = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
            scale = cov / vx
            bias = float(np.mean(y) - scale * np.mean(x))
            pred = scale * x + bias
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            out["affine_a_to_b"] = {"scale": float(scale), "bias": float(bias), "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None}
    except Exception:
        pass
    return out


def _onnx_output_names(path: Path) -> list[str]:
    _inputs, outputs = _onnx_io(path)
    return [str(name) for name, _shape in outputs]


def _run_part2(part2: Path, input_name: str, feed: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    sess = ort.InferenceSession(str(part2), providers=["CPUExecutionProvider"])
    outs = sess.run(None, {input_name: np.asarray(feed, dtype=np.float32)})
    return [o.name for o in sess.get_outputs()], [np.asarray(x) for x in outs]


def _run_generic_hailo(
    *,
    hef: Path,
    part1: Path,
    input_name: str,
    input_feed: np.ndarray,
    input_hwc_u8: np.ndarray | None,
    output_names: list[str],
    hw_arch: str,
    quantized_inputs: bool,
    quantized_outputs: bool,
    artifacts_dir: Path,
) -> dict[str, np.ndarray]:
    hb = HailoBackend(strict=True, compile_backend="local")
    options = {
        "hw_arch": hw_arch,
        "hef_path": str(hef),
        "onnx_model_path": str(part1),
        "quantized_inputs": bool(quantized_inputs),
        "quantized_outputs": bool(quantized_outputs),
        "canonical_input_slot_names": [input_name],
        "canonical_output_slot_names": output_names,
        "persistent_activation": False,
    }
    prep = hb.prepare(RunCfg(model_path=part1, options=options), artifacts_dir=artifacts_dir)
    try:
        if quantized_inputs:
            if input_hwc_u8 is None:
                raise RuntimeError("quantized input mode requires input_dump/input_shape_hwc in boundary manifest")
            feed = {input_name: np.ascontiguousarray(input_hwc_u8.astype(np.uint8, copy=False))}
        else:
            feed = {input_name: np.ascontiguousarray(np.asarray(input_feed, dtype=np.float32))}
        out = hb.run(prep, feed).outputs
        return {str(k): np.asarray(v) for k, v in dict(out).items()}
    finally:
        try:
            hb.cleanup(prep)
        except Exception:
            pass


def _first_matching_tensor(tensors: dict[str, np.ndarray], preferred_name: str, target_shape: tuple[int, ...]) -> tuple[str, np.ndarray] | None:
    if preferred_name in tensors:
        return preferred_name, np.asarray(tensors[preferred_name])
    # A same-sized, same-shaped output with another name is not evidence for
    # this declared cut. Do not silently repair a broken mapping in a probe.
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--boundary-manifest", required=True)
    ap.add_argument("--reference-report", default="")
    ap.add_argument("--hw-arch", default="hailo8")
    ap.add_argument("--out", default="")
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = ns.case if str(ns.case).startswith("b") else f"b{int(ns.case):03d}"
    man_path = Path(ns.boundary_manifest).expanduser().resolve()
    man, native_raw, native_file = _load_native_boundary(man_path)
    input_hwc_u8 = _load_input_dump_hwc(man)

    part1 = _find_part1_onnx(bs, case)
    part2 = _find_part2_onnx(bs, case, "")
    hef = _find_hef(bs, case, ns.hw_arch)
    part1_inputs, part1_outputs = _onnx_io(part1)
    if not part1_inputs or not part1_outputs:
        raise RuntimeError(f"could not read part1 IO from {part1}")
    input_name, input_shape = part1_inputs[0]
    output_name, output_shape = part1_outputs[0]
    target_shape = tuple(int(x) for x in output_shape)
    output_names = [str(name) for name, _shp in part1_outputs]

    input_feed = _input_dump_feed_from_manifest(man, [int(x) for x in input_shape], "native")
    if input_feed is None:
        raise RuntimeError("boundary manifest has no usable input_dump; rerun native with --native-boundary-debug")

    # ORT Part1 reference from the exact same input dump.
    sess1 = ort.InferenceSession(str(part1), providers=["CPUExecutionProvider"])
    ort_p1_outs = sess1.run(None, {input_name: input_feed.astype(np.float32)})
    ort_p1_map = {o.name: np.asarray(a) for o, a in zip(sess1.get_outputs(), ort_p1_outs)}
    ort_ref = np.asarray(ort_p1_map.get(output_name, ort_p1_outs[0]))

    rows: list[dict[str, Any]] = []
    native_layout_compares = []
    for layout_name, cand, layout_meta in _layout_candidates(np.asarray(native_raw, dtype=np.float32), tuple(ort_ref.shape)):
        native_layout_compares.append({
            "layout": layout_name,
            "layout_meta": layout_meta,
            "compare_to_ort_part1": _tensor_compare(cand, ort_ref),
        })
    native_layout_compares.sort(key=lambda r: float((r.get("compare_to_ort_part1") or {}).get("corr") or -999), reverse=True)

    # Native boundary through original Part2 ONNX using the best corr layout.
    native_part2_semantic: dict[str, Any] = {"available": False, "reason": "no_native_layout_candidate"}
    if native_layout_compares:
        best_native_layout = native_layout_compares[0]["layout"]
        for layout_name, cand, _layout_meta in _layout_candidates(np.asarray(native_raw, dtype=np.float32), tuple(ort_ref.shape)):
            if layout_name == best_native_layout:
                out_names, outs = _run_part2(part2, output_name, cand)
                native_part2_semantic = _semantic_score_outputs(outs, out_names, ns.reference_report)
                break

    modes = [
        ("generic_float_in_float_out", False, False),
        ("generic_uint8_in_float_out", True, False),
    ]
    # Quantized-output modes are useful for raw-vstream comparison, but not for
    # feeding float ONNX Part2. Keep them optional through an environment flag.
    import os
    if str(os.environ.get("ONNX_SPLITPOINT_PROBE_RAW_HAILO_OUTPUTS", "") or "").lower() in {"1", "true", "yes", "on"}:
        modes.extend([
            ("generic_float_in_uint8_out", False, True),
            ("generic_uint8_in_uint8_out", True, True),
        ])

    for mode_name, qin, qout in modes:
        row: dict[str, Any] = {
            "mode": mode_name,
            "quantized_inputs": bool(qin),
            "quantized_outputs": bool(qout),
        }
        try:
            gmap = _run_generic_hailo(
                hef=hef,
                part1=part1,
                input_name=str(input_name),
                input_feed=input_feed,
                input_hwc_u8=input_hwc_u8,
                output_names=output_names,
                hw_arch=str(ns.hw_arch),
                quantized_inputs=bool(qin),
                quantized_outputs=bool(qout),
                artifacts_dir=man_path.parent / f"generic_hailo_probe_{mode_name}",
            )
            picked = _first_matching_tensor(gmap, str(output_name), tuple(ort_ref.shape))
            if picked is None:
                raise RuntimeError("generic Hailo produced no tensors")
            gname, gout = picked
            row["generic_output_name"] = gname
            row["generic_output_summary"] = _summ(gout)
            row["compare_generic_to_ort_part1"] = _tensor_compare(gout, ort_ref)
            # Compare native boundary under each layout to the generic output.
            ncmp = []
            for layout_name, cand, layout_meta in _layout_candidates(np.asarray(native_raw, dtype=np.float32), tuple(np.asarray(gout).shape)):
                ncmp.append({"layout": layout_name, "layout_meta": layout_meta, "compare_native_to_generic": _tensor_compare(cand, gout)})
            ncmp.sort(key=lambda r: float((r.get("compare_native_to_generic") or {}).get("corr") or -999), reverse=True)
            row["native_vs_generic_layouts"] = ncmp[:10]
            if not qout:
                out_names, outs = _run_part2(part2, str(output_name), np.asarray(gout, dtype=np.float32))
                row["part2_onnx_semantic"] = _semantic_score_outputs(outs, out_names, ns.reference_report)
                row["part2_output_summaries"] = [_summ(o) for o in outs]
            else:
                row["part2_onnx_semantic"] = {"available": False, "reason": "quantized_output_not_fed_to_float_part2"}
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["part2_onnx_semantic"] = {"available": False, "reason": "generic_hailo_run_failed"}
        rows.append(row)

    def _match_ratio(sem: dict[str, Any]) -> float:
        try:
            return float((sem.get("match") or {}).get("match_ratio") or 0.0)
        except Exception:
            return 0.0

    best_generic_ratio = max([_match_ratio(r.get("part2_onnx_semantic") or {}) for r in rows] or [0.0])
    native_ratio = _match_ratio(native_part2_semantic)
    best_g_vs_ort_corr = max([float(((r.get("compare_generic_to_ort_part1") or {}).get("corr")) or -999) for r in rows if not r.get("error")] or [-999])
    best_n_vs_g_corr = max([
        float(((x.get("compare_native_to_generic") or {}).get("corr")) or -999)
        for r in rows for x in (r.get("native_vs_generic_layouts") or [])
    ] or [-999])

    generic_attempted = bool(rows)
    generic_success = any(not r.get("error") for r in rows)
    generic_errors = [str(r.get("error") or "") for r in rows if r.get("error")]
    if generic_attempted and not generic_success:
        diagnosis = "generic_hailo_unavailable_cannot_compare"
    elif best_generic_ratio >= 0.5 and native_ratio < 0.5:
        diagnosis = "native_cpp_hailort_boundary_contract_suspect"
    elif best_generic_ratio < 0.5 and native_ratio < 0.5:
        diagnosis = "split_hailo_part1_to_float_part2_contract_suspect"
    elif best_generic_ratio >= 0.5 and native_ratio >= 0.5:
        diagnosis = "boundary_contract_semantically_plausible_check_trt_path_if_native_fails"
    else:
        diagnosis = "inconclusive"

    payload = {
        **DIAGNOSTIC_FLAGS,
        "schema": "onnx-splitpoint/generic-vs-native-hailo-boundary-probe",
        "schema_version": 1,
        "ok": bool(generic_success),
        "diagnosis": diagnosis,
        "benchmark_set": str(bs),
        "case": case,
        "hw_arch": ns.hw_arch,
        "part1_onnx": str(part1),
        "part2_onnx": str(part2),
        "hef": str(hef),
        "boundary_manifest": str(man_path),
        "boundary_file": str(native_file),
        "input_dump_present": bool(input_hwc_u8 is not None),
        "input_name": input_name,
        "part1_output_name": output_name,
        "part1_output_shape": [int(x) for x in target_shape],
        "native_boundary_summary": _summ(native_raw),
        "ort_part1_summary": _summ(ort_ref),
        "native_layout_compares_to_ort_part1": native_layout_compares[:10],
        "native_part2_onnx_semantic_best_corr_layout": native_part2_semantic,
        "best_generic_part2_match_ratio": best_generic_ratio,
        "native_part2_match_ratio": native_ratio,
        "best_generic_vs_ort_corr": best_g_vs_ort_corr,
        "best_native_vs_generic_corr": best_n_vs_g_corr,
        "generic_attempted": generic_attempted,
        "generic_success": generic_success,
        "generic_errors": generic_errors,
        "rows": rows,
    }

    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent / "generic_vs_native_hailo_boundary_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# Generic vs Native Hailo boundary probe",
        "",
        f"Diagnosis: `{diagnosis}`",
        f"Benchmark set: `{bs}`",
        f"Case: `{case}`",
        f"HEF: `{hef}`",
        f"Part1 ONNX: `{part1}`",
        f"Part2 ONNX: `{part2}`",
        f"Boundary manifest: `{man_path}`",
        "",
        "## Summary",
        "",
        f"- native Part2 ORT match ratio: `{native_ratio}`",
        f"- best Generic-Hailo Part1 -> ORT Part2 match ratio: `{best_generic_ratio}`",
        f"- best Generic-Hailo vs ORT-Part1 corr: `{best_g_vs_ort_corr}`",
        f"- best Native-boundary vs Generic-Hailo corr: `{best_n_vs_g_corr}`",
        "",
        "## Native boundary vs ORT Part1 layouts",
        "",
        "| rank | layout | corr | mean abs | max abs | affine r2 | affine scale | affine bias |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(native_layout_compares[:10]):
        c = r.get("compare_to_ort_part1") or {}
        aff = c.get("affine_a_to_b") or {}
        lines.append(f"| {i} | `{r.get('layout')}` | {c.get('corr')} | {c.get('mean_abs')} | {c.get('max_abs')} | {aff.get('r2')} | {aff.get('scale')} | {aff.get('bias')} |")
    lines += [
        "",
        "## Generic Hailo modes",
        "",
        "| mode | error | generic vs ORT corr | generic->Part2 match | pred | decode mode | best native vs generic layout | best native vs generic corr |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]
    for r in rows:
        cmp = r.get("compare_generic_to_ort_part1") or {}
        sem = r.get("part2_onnx_semantic") or {}
        m = sem.get("match") or {}
        nvg = (r.get("native_vs_generic_layouts") or [{}])[0] if r.get("native_vs_generic_layouts") else {}
        nvgc = nvg.get("compare_native_to_generic") or {}
        lines.append(
            f"| `{r.get('mode')}` | `{r.get('error','')}` | {cmp.get('corr')} | {m.get('match_ratio')} | {sem.get('pred_count')} | `{sem.get('decode_mode')}` | `{nvg.get('layout','')}` | {nvgc.get('corr')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `generic_hailo_unavailable_cannot_compare`: Generic Hailo could not run, usually because Python HailoRT bindings are missing; do not treat best_generic_*=-999/0 as a real Generic-Hailo failure.",
        "- `native_cpp_hailort_boundary_contract_suspect`: Generic-Hailo boundary is semantically good but Native C++ boundary is not; inspect native HailoRT vstream params/input/output selection.",
        "- `split_hailo_part1_to_float_part2_contract_suspect`: both Generic-Hailo and Native Hailo boundaries fail with the original float Part2; the split is unsafe for Hailo-Part1 -> float-Part2 without a stronger adapter/calibration strategy.",
        "- `boundary_contract_semantically_plausible_check_trt_path_if_native_fails`: boundary handoff is likely OK; compare TensorRT/native outputs.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(json_safe({
        **DIAGNOSTIC_FLAGS,
        "ok": bool(generic_success),
        "out": str(out),
        "summary_md": str(md),
        "diagnosis": diagnosis,
        "native_part2_match_ratio": native_ratio,
        "best_generic_part2_match_ratio": best_generic_ratio,
        "best_generic_vs_ort_corr": best_g_vs_ort_corr,
        "best_native_vs_generic_corr": best_n_vs_g_corr,
    }), indent=2, allow_nan=False))
    return 0 if generic_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
