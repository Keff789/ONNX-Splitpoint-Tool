#!/usr/bin/env python3
"""Boundary interface contract validator.

This diagnostic compares a native accelerator boundary dump (usually raw uint8
from Hailo/DeepX Part1) against the ONNX/ORT Part1 activation for the exact same
input image.  Unlike the simple activation compare, it performs:

* multiple memory-layout hypotheses
* global affine fit
* direct per-channel affine fit
* channel-reorder/correlation analysis

The goal is not to prove semantic correctness but to classify why a native
Part1->TensorRT Part2 split fails:

* global_dequant_candidate: one global scale/zp likely enough
* per_channel_dequant_candidate: per-channel quantization likely
* layout_reorder_suspect: memory/channel order mismatch likely
* not_same_tensor_or_transform: Hailo/DeepX output is not the ONNX cut tensor
* insufficient_evidence: no clear conclusion

This is intentionally conservative: it never makes an accuracy claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Reuse the robust image/ORT helpers from the existing activation diagnostic.
try:
    from native_boundary_activation_compare import (  # type: ignore
        _candidate_layouts,
        _find_image,
        _find_part1_onnx,
        _fit_affine,
        _image_from_boundary_manifest,
        _image_from_report,
        _input_dump_feed_from_manifest,
        _onnx_io,
        _preprocess,
        _read_json,
        _summ,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(f"failed to import native_boundary_activation_compare helpers: {e}")


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _channel_axis(shape: tuple[int, ...]) -> int | None:
    """Guess the channel axis for common 4D activation shapes."""
    if len(shape) != 4:
        return None
    # NCHW: [1, C, H, W]
    if shape[0] == 1 and shape[1] >= 4 and shape[2] >= 4 and shape[3] >= 4:
        return 1
    # NHWC: [1, H, W, C]
    if shape[0] == 1 and shape[-1] >= 4 and shape[1] >= 4 and shape[2] >= 4:
        return 3
    return None




def _contract_candidate_layouts(raw: np.ndarray, target_shape: tuple[int, ...]) -> list[tuple[str, np.ndarray]]:
    """Generate boundary-memory hypotheses that all end in target_shape.

    Native boundary manifests may only know byte count, while ONNX/ORT knows the
    intended Part1 output shape.  Always test the direct target-shape reshape
    first, then common accelerator memory layouts.
    """
    out: list[tuple[str, np.ndarray]] = []
    prod = int(np.prod(target_shape)) if target_shape else 0

    def add(name: str, fn) -> None:
        try:
            x = fn()
            if tuple(x.shape) == tuple(target_shape):
                # Avoid duplicate names with identical data layout.
                if not any(n == name for n, _ in out):
                    out.append((name, x))
        except Exception:
            pass

    if prod and raw.size == prod:
        add("as_target_shape", lambda: raw.reshape(target_shape))
    if tuple(raw.shape) == tuple(target_shape) and not any(n == "as_manifest_shape" for n, _ in out):
        out.insert(0, ("as_manifest_shape", raw))

    if len(target_shape) == 4 and prod and raw.size == prod:
        n, a, b, c = [int(x) for x in target_shape]
        # Target interpreted as NCHW: [N,C,H,W].
        n0, ch, h, w = n, a, b, c
        add("memory_nhwc_to_nchw", lambda: raw.reshape((n0, h, w, ch)).transpose(0, 3, 1, 2))
        add("memory_nwhc_to_nchw", lambda: raw.reshape((n0, w, h, ch)).transpose(0, 3, 2, 1))
        add("memory_chwn_to_nchw", lambda: raw.reshape((ch, h, w, n0)).transpose(3, 0, 1, 2))
        add("memory_hwcn_to_nchw", lambda: raw.reshape((h, w, ch, n0)).transpose(3, 2, 0, 1))
        add("memory_ncwh_to_nchw", lambda: raw.reshape((n0, ch, w, h)).transpose(0, 1, 3, 2))

        # Target interpreted as NHWC: [N,H,W,C]. Included for exported models
        # that carry channels-last activations.
        n1, h1, w1, ch1 = n, a, b, c
        add("memory_nchw_to_nhwc", lambda: raw.reshape((n1, ch1, h1, w1)).transpose(0, 2, 3, 1))
        add("memory_chwn_to_nhwc", lambda: raw.reshape((ch1, h1, w1, n1)).transpose(3, 1, 2, 0))
        add("memory_hwcn_to_nhwc", lambda: raw.reshape((h1, w1, ch1, n1)).transpose(3, 0, 1, 2))

    return out

def _move_channel_first(x: np.ndarray, axis: int) -> np.ndarray:
    if axis == 1:
        return x.reshape(x.shape[0], x.shape[1], -1)[0]
    if axis == 3:
        return np.moveaxis(x, 3, 1).reshape(x.shape[0], x.shape[3], -1)[0]
    raise ValueError(f"unsupported channel axis {axis}")


def _sample_spatial(x: np.ndarray, max_spatial: int) -> np.ndarray:
    # x: [C, S]
    if x.shape[1] <= max_spatial:
        return x
    idx = np.linspace(0, x.shape[1] - 1, max_spatial).astype(np.int64)
    return x[:, idx]


def _corr_matrix(raw_cf: np.ndarray, ref_cf: np.ndarray, max_spatial: int = 20000) -> np.ndarray:
    """Compute channel correlation matrix raw_channel x ref_channel."""
    x = _sample_spatial(raw_cf.astype(np.float64), max_spatial)
    y = _sample_spatial(ref_cf.astype(np.float64), max_spatial)
    # Center and normalize per channel.
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    xs = np.sqrt((x * x).sum(axis=1, keepdims=True))
    ys = np.sqrt((y * y).sum(axis=1, keepdims=True))
    xs[xs == 0] = np.nan
    ys[ys == 0] = np.nan
    xn = x / xs
    yn = y / ys
    c = xn @ yn.T
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return c


def _per_channel_fit(raw: np.ndarray, ref: np.ndarray, axis: int, max_channels: int = 1024) -> dict[str, Any]:
    raw_cf = _move_channel_first(raw, axis)
    ref_cf = _move_channel_first(ref, axis)
    c = min(raw_cf.shape[0], ref_cf.shape[0])
    if c == 0 or c > max_channels:
        return {"ok": False, "reason": f"unsupported_channel_count_{c}"}
    fits = []
    for i in range(c):
        f = _fit_affine(raw_cf[i], ref_cf[i], sample=20000)
        if f.get("ok"):
            fits.append(f)
        else:
            fits.append({"ok": False, "reason": f.get("reason")})
    def arr(k: str) -> np.ndarray:
        vals = [_safe_float(f.get(k)) for f in fits if f.get("ok")]
        vals = [v for v in vals if v is not None]
        return np.asarray(vals, dtype=np.float64)
    corrs = arr("corr")
    r2s = arr("r2")
    scales = arr("scale_fit")
    zps = arr("zero_point_fit")
    maes = arr("mae")
    summary = {
        "ok": bool(corrs.size),
        "channel_count": int(c),
        "valid_fit_count": int(corrs.size),
        "corr_mean": float(corrs.mean()) if corrs.size else None,
        "corr_median": float(np.median(corrs)) if corrs.size else None,
        "corr_p10": float(np.quantile(corrs, 0.10)) if corrs.size else None,
        "corr_p90": float(np.quantile(corrs, 0.90)) if corrs.size else None,
        "r2_mean": float(r2s.mean()) if r2s.size else None,
        "r2_median": float(np.median(r2s)) if r2s.size else None,
        "scale_median": float(np.median(scales)) if scales.size else None,
        "scale_p10": float(np.quantile(scales, 0.10)) if scales.size else None,
        "scale_p90": float(np.quantile(scales, 0.90)) if scales.size else None,
        "zero_point_median": float(np.median(zps)) if zps.size else None,
        "zero_point_p10": float(np.quantile(zps, 0.10)) if zps.size else None,
        "zero_point_p90": float(np.quantile(zps, 0.90)) if zps.size else None,
        "mae_median": float(np.median(maes)) if maes.size else None,
    }
    # Keep a few outlier channels for debugging.
    outliers = []
    for i, f in enumerate(fits):
        if not f.get("ok"):
            continue
        corr = _safe_float(f.get("corr"))
        r2 = _safe_float(f.get("r2"))
        if corr is not None and (corr < 0.2 or (r2 is not None and r2 < 0.0)):
            outliers.append({"channel": i, "corr": corr, "r2": r2, "scale": f.get("scale_fit"), "zero_point": f.get("zero_point_fit")})
        if len(outliers) >= 20:
            break
    summary["low_corr_examples"] = outliers
    return summary


def _channel_reorder(raw: np.ndarray, ref: np.ndarray, axis: int, max_spatial: int = 20000) -> dict[str, Any]:
    raw_cf = _move_channel_first(raw, axis)
    ref_cf = _move_channel_first(ref, axis)
    c = min(raw_cf.shape[0], ref_cf.shape[0])
    if c == 0 or raw_cf.shape[0] != ref_cf.shape[0]:
        return {"ok": False, "reason": f"channel_count_mismatch raw={raw_cf.shape[0]} ref={ref_cf.shape[0]}"}
    corr = _corr_matrix(raw_cf, ref_cf, max_spatial=max_spatial)
    diag = np.diag(corr)
    abs_corr = np.abs(corr)
    best_idx = abs_corr.argmax(axis=1)
    best = corr[np.arange(c), best_idx]
    direct_best_fraction = float((best_idx == np.arange(c)).mean())
    # Top suspicious channels where diagonal is weak but another channel is strong.
    examples = []
    gains = np.abs(best) - np.abs(diag)
    for i in np.argsort(-gains)[:20]:
        examples.append({
            "raw_channel": int(i),
            "best_ref_channel": int(best_idx[i]),
            "diag_corr": float(diag[i]),
            "best_corr": float(best[i]),
            "gain_abs": float(gains[i]),
        })
    return {
        "ok": True,
        "channel_count": int(c),
        "diag_corr_mean": float(diag.mean()),
        "diag_corr_median": float(np.median(diag)),
        "diag_abs_corr_mean": float(np.abs(diag).mean()),
        "diag_abs_corr_median": float(np.median(np.abs(diag))),
        "best_abs_corr_mean": float(np.abs(best).mean()),
        "best_abs_corr_median": float(np.median(np.abs(best))),
        "direct_best_fraction": direct_best_fraction,
        "reorder_gain_mean": float((np.abs(best) - np.abs(diag)).mean()),
        "reorder_gain_median": float(np.median(np.abs(best) - np.abs(diag))),
        "reorder_examples": examples,
    }


def _classify(global_fit: dict[str, Any], pc: dict[str, Any], reorder: dict[str, Any]) -> tuple[str, str]:
    g_corr = _safe_float(global_fit.get("corr")) or 0.0
    g_r2 = _safe_float(global_fit.get("r2")) or -999.0
    pc_corr = _safe_float(pc.get("corr_median")) or 0.0
    pc_r2 = _safe_float(pc.get("r2_median")) or -999.0
    best_abs = _safe_float(reorder.get("best_abs_corr_median")) or 0.0
    diag_abs = _safe_float(reorder.get("diag_abs_corr_median")) or 0.0
    direct_frac = _safe_float(reorder.get("direct_best_fraction")) or 0.0
    gain = _safe_float(reorder.get("reorder_gain_median")) or 0.0

    if g_corr >= 0.90 and g_r2 >= 0.80:
        return "global_dequant_candidate", "high global affine correlation/r2; one global scale/zp may be sufficient"
    if pc_corr >= 0.80 and pc_r2 >= 0.50 and g_r2 < 0.70:
        return "per_channel_dequant_candidate", "direct per-channel fits are much stronger than global fit"
    if best_abs >= 0.75 and (gain >= 0.25 or direct_frac < 0.50):
        return "layout_or_channel_reorder_suspect", "best channel correlations are much stronger than direct channel correlations"
    if best_abs < 0.50 and pc_corr < 0.50 and g_corr < 0.50:
        return "not_same_tensor_or_transform", "global, per-channel and reorder correlations are all weak"
    if diag_abs < 0.50 and best_abs >= 0.50:
        return "possible_reorder_or_layout_transform", "some channels correlate, but not on the expected direct axis"
    return "insufficient_evidence", "contract relation is weak or ambiguous; inspect full JSON and overlays"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--boundary-manifest", required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--reference-report", default="")
    ap.add_argument("--image-scale", default="native", choices=["native","native_letterbox_norm","native_letterbox_raw","letterbox0_norm","letterbox0_raw","yolo","yolo_letterbox_norm","yolo_letterbox_raw","letterbox114_norm","letterbox114_raw","resize_norm","resize_raw","norm","raw","auto","imagenet"])
    ap.add_argument("--out", default="")
    ap.add_argument("--max-spatial", type=int, default=20000)
    ap.add_argument("--require-image-match", action="store_true")
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = ns.case
    man_path = Path(ns.boundary_manifest).expanduser().resolve()
    if not man_path.is_file():
        out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent / "native_boundary_interface_contract_validator.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        work_dir = man_path.parent.parent if man_path.parent.name == "native_fifo_boundary" else man_path.parent
        nearby_results = work_dir / "native_fifo_results.json"
        nearby_cfg = work_dir / "native_fifo_config.json"
        payload = {
            "schema": "onnx-splitpoint/native-boundary-interface-contract-validator",
            "schema_version": 2,
            "ok": False,
            "error": "boundary_manifest_missing",
            "boundary_manifest": str(man_path),
            "benchmark_set": str(bs),
            "case": case,
            "nearby_native_fifo_results": str(nearby_results) if nearby_results.exists() else "",
            "nearby_native_fifo_config": str(nearby_cfg) if nearby_cfg.exists() else "",
            "hint": "Run the Hailo8 native FIFO path with --dump-boundary, or through update_evalset_native_producers.py with --native-boundary-debug, then rerun this validator.",
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md = out.with_suffix(".md")
        lines = [
            "# Native boundary interface contract validator",
            "",
            "Status: `boundary_manifest_missing`",
            "",
            f"Missing manifest: `{man_path}`",
            "",
            "The validator is wired correctly, but there is no raw Part1 boundary dump to compare.",
            "",
            "Run the native Hailo8 producer with `--native-boundary-debug` on the EvalRun updater, or `--dump-boundary` on the native FIFO runner.",
            "",
            f"Nearby native result JSON: `{payload['nearby_native_fifo_results']}`",
            f"Nearby native config JSON: `{payload['nearby_native_fifo_config']}`",
        ]
        md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({"ok": False, "error": "boundary_manifest_missing", "out": str(out), "summary_md": str(md)}, indent=2))
        return 2
    man = _read_json(man_path)
    binp = Path(man.get("file", "")).expanduser()
    if not binp.is_absolute():
        binp = (man_path.parent / binp).resolve()
    raw_shape = man.get("shape") or man.get("boundary_shape")
    dtype_s = str(man.get("dtype") or "uint8").lower()
    dtype_map = {"uint8": np.uint8, "int8": np.int8, "float32": np.float32, "fp32": np.float32, "float16": np.float16, "fp16": np.float16}
    raw_dtype = dtype_map.get(dtype_s, np.uint8)
    raw = np.fromfile(str(binp), dtype=raw_dtype)
    if raw_shape:
        try:
            raw = raw.reshape([int(x) for x in raw_shape])
        except Exception as e:
            raise RuntimeError(f"boundary manifest shape {raw_shape} does not match file {binp} dtype={dtype_s}: {e}")

    ref_report = Path(ns.reference_report).expanduser().resolve() if ns.reference_report else None
    boundary_image = _image_from_boundary_manifest(man)
    ref_image = _image_from_report(ref_report)
    img_name = ns.image or boundary_image or ref_image
    img_path = _find_image(bs, img_name) if img_name else None
    if not img_path:
        raise FileNotFoundError(f"Could not resolve image {img_name!r}; pass --image explicitly")
    mismatch = bool(boundary_image and ref_image and Path(str(boundary_image)).name != Path(str(ref_image)).name)
    if ns.require_image_match and mismatch:
        raise RuntimeError(f"boundary/reference image mismatch: boundary={boundary_image} ref={ref_image}")

    part1 = _find_part1_onnx(bs, case)
    inputs, _outs = _onnx_io(part1)
    if not inputs:
        raise RuntimeError(f"no inputs in {part1}")
    inp_name, inp_shape = inputs[0]

    import onnxruntime as ort
    sess = ort.InferenceSession(str(part1), providers=["CPUExecutionProvider"])
    feed = _input_dump_feed_from_manifest(man, inp_shape, ns.image_scale)
    feed_source = 'boundary_manifest_input_dump' if feed is not None else f'image_preprocess:{ns.image_scale}'
    if feed is None:
        feed = _preprocess(img_path, inp_shape, ns.image_scale)
    ort_outs = sess.run(None, {inp_name: feed})
    out_names = [o.name for o in sess.get_outputs()]

    rows = []
    for out_name, ref in zip(out_names, ort_outs):
        ref = np.asarray(ref)
        if ref.size != raw.size:
            continue
        for layout_name, raw_layout in _contract_candidate_layouts(raw, tuple(ref.shape)):
            axis = _channel_axis(tuple(ref.shape))
            global_fit = _fit_affine(raw_layout, ref)
            pc = _per_channel_fit(raw_layout, ref, axis) if axis is not None else {"ok": False, "reason": "no_channel_axis"}
            reorder = _channel_reorder(raw_layout, ref, axis, max_spatial=ns.max_spatial) if axis is not None else {"ok": False, "reason": "no_channel_axis"}
            status, reason = _classify(global_fit, pc, reorder)
            rows.append({
                "onnx_output": out_name,
                "onnx_shape": list(ref.shape),
                "layout": layout_name,
                "status": status,
                "reason": reason,
                "global_fit": global_fit,
                "per_channel_fit": pc,
                "channel_reorder": reorder,
                "reference_summary": _summ(ref),
                "raw_layout_summary": _summ(raw_layout),
            })

    def _score(row: dict[str, Any]) -> tuple[float, float, float]:
        status_order = {
            "global_dequant_candidate": 5,
            "per_channel_dequant_candidate": 4,
            "layout_or_channel_reorder_suspect": 3,
            "possible_reorder_or_layout_transform": 2,
            "insufficient_evidence": 1,
            "not_same_tensor_or_transform": 0,
        }
        gf = row.get("global_fit") or {}
        pc = row.get("per_channel_fit") or {}
        reor = row.get("channel_reorder") or {}
        return (
            status_order.get(row.get("status"), 0),
            _safe_float(gf.get("r2")) or -999.0,
            max(_safe_float(pc.get("corr_median")) or 0.0, _safe_float(reor.get("best_abs_corr_median")) or 0.0),
        )
    rows.sort(key=_score, reverse=True)

    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent / "native_boundary_interface_contract_validator.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "onnx-splitpoint/native-boundary-interface-contract-validator",
        "schema_version": 2,
        "ok": True,
        "benchmark_set": str(bs),
        "case": case,
        "part1_onnx": str(part1),
        "boundary_manifest": str(man_path),
        "boundary_file": str(binp),
        "image": str(img_path),
        "boundary_image": boundary_image,
        "reference_report_image": ref_image,
        "image_mismatch": mismatch,
        "ort_feed_source": feed_source,
        "raw_summary": _summ(raw),
        "rows": rows,
        "best": rows[0] if rows else None,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# Native boundary interface contract validator",
        "",
        f"Benchmark set: `{bs}`",
        f"Case: `{case}`",
        f"Boundary manifest: `{man_path}`",
        f"Image: `{img_path}`",
        f"Boundary image: `{boundary_image}`",
        f"Reference image: `{ref_image}`",
        f"Image mismatch: `{mismatch}`",
        f"ORT feed source: `{feed_source}`",
        "",
        "## Raw boundary summary",
        "",
        f"shape=`{list(raw.shape)}` dtype=`{raw.dtype}` min=`{payload['raw_summary'].get('min')}` max=`{payload['raw_summary'].get('max')}` mean=`{payload['raw_summary'].get('mean')}`",
        "",
        "## Candidate summary",
        "",
        "| rank | output | layout | status | global r2 | global corr | pc corr med | pc r2 med | best ch corr med | direct best frac | note |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(rows[:20]):
        gf = r.get("global_fit") or {}
        pc = r.get("per_channel_fit") or {}
        reor = r.get("channel_reorder") or {}
        lines.append(
            f"| {i} | `{r.get('onnx_output')}` | `{r.get('layout')}` | `{r.get('status')}` | "
            f"{gf.get('r2','')} | {gf.get('corr','')} | {pc.get('corr_median','')} | {pc.get('r2_median','')} | "
            f"{reor.get('best_abs_corr_median','')} | {reor.get('direct_best_fraction','')} | {r.get('reason','')} |"
        )
    if rows:
        b = rows[0]
        lines += [
            "",
            "## Best candidate details",
            "",
            f"Status: `{b.get('status')}`",
            "",
            f"Reason: {b.get('reason')}",
            "",
            "### Per-channel scale / zero-point summary",
            "",
        ]
        pc = b.get("per_channel_fit") or {}
        for k in ("channel_count", "valid_fit_count", "corr_median", "r2_median", "scale_median", "scale_p10", "scale_p90", "zero_point_median", "zero_point_p10", "zero_point_p90", "mae_median"):
            lines.append(f"- {k}: `{pc.get(k)}`")
        lines += ["", "### Channel reorder summary", ""]
        reor = b.get("channel_reorder") or {}
        for k in ("diag_abs_corr_median", "best_abs_corr_median", "direct_best_fraction", "reorder_gain_median"):
            lines.append(f"- {k}: `{reor.get(k)}`")
        ex = reor.get("reorder_examples") or []
        if ex:
            lines += ["", "Top reorder examples:", "", "| raw ch | best ref ch | diag corr | best corr | gain |", "|---:|---:|---:|---:|---:|"]
            for e in ex[:10]:
                lines.append(f"| {e.get('raw_channel')} | {e.get('best_ref_channel')} | {e.get('diag_corr')} | {e.get('best_corr')} | {e.get('gain_abs')} |")
    else:
        lines += ["", "No candidate with matching element count was found."]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "summary_md": str(md), "rows": len(rows), "best_status": rows[0].get("status") if rows else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
