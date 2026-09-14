#!/usr/bin/env python3
"""Compare native-boundary Part2 replay against an ORT Part1->Part2 tensor oracle.

Why this exists
---------------
The semantic/visual validator is useful only after its own reference and decode
contract is proven.  If the positive control

    ORT Part1 activation -> original Part2 ONNX -> semantic reference

already fails, the next question is tensor-level, not Hailo-level:

    Does native boundary -> Part2 ONNX produce the same tensors as
    ORT Part1 activation -> Part2 ONNX?

This probe uses the exact input_dump recorded by the native boundary manifest,
runs ORT Part1 and Part2 as the local oracle, then replays the native boundary
under layout/affine variants through the same Part2 ONNX and compares output
tensors directly.  Semantic scoring is kept as an optional diagnostic but it no
longer drives the main conclusion.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    import onnxruntime as ort  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"onnxruntime import failed: {type(exc).__name__}: {exc}")

try:
    from native_boundary_activation_compare import (  # type: ignore
        _find_part1_onnx,
        _fit_affine,
        _input_dump_feed_from_manifest,
        _onnx_io,
        _preprocess,
        _find_image,
        _image_from_boundary_manifest,
        _image_from_report,
        _read_json,
        _summ,
    )
    from native_boundary_interface_contract_validator import _contract_candidate_layouts, _channel_axis  # type: ignore
    from native_boundary_affine_replay_probe import (  # type: ignore
        _apply_per_channel_affine,
        _dtype_from_manifest,
        _find_part2_onnx,
        _ort_input_dtype,
        _shape_from_ort,
    )
    from native_boundary_dequant_sweep import _semantic_score_outputs  # type: ignore
    from validate_output_dumps import load_dump  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"helper import failed: {type(exc).__name__}: {exc}")


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size != bb.size or aa.size < 2:
        return None
    sa = float(np.std(aa)); sb = float(np.std(bb))
    if sa <= 0.0 or sb <= 0.0:
        return None
    try:
        c = float(np.corrcoef(aa, bb)[0, 1])
        return c if math.isfinite(c) else None
    except Exception:
        return None


def _r2_affine(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Fit b ~= scale*a + bias and return a compact diagnostic."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    out: dict[str, Any] = {"ok": False}
    if x.size != y.size or x.size < 2:
        out["reason"] = "size_mismatch"
        return out
    vx = float(np.var(x))
    if vx <= 0.0:
        out["reason"] = "zero_variance"
        return out
    cov = float(np.mean((x - np.mean(x)) * (y - np.mean(y))))
    scale = cov / vx
    bias = float(np.mean(y) - scale * np.mean(x))
    pred = scale * x + bias
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    out.update({
        "ok": True,
        "scale": float(scale),
        "bias": float(bias),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "corr": _corr(x, y),
    })
    return out


def _tensor_compare(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    c = np.asarray(candidate)
    r = np.asarray(reference)
    out: dict[str, Any] = {
        "candidate_shape": [int(x) for x in c.shape],
        "reference_shape": [int(x) for x in r.shape],
        "candidate_dtype": str(c.dtype),
        "reference_dtype": str(r.dtype),
        "same_shape": bool(tuple(c.shape) == tuple(r.shape)),
        "candidate_summary": _summ(c),
        "reference_summary": _summ(r),
    }
    if c.size != r.size:
        out["same_size"] = False
        return out
    cf = c.astype(np.float64, copy=False).reshape(-1)
    rf = r.astype(np.float64, copy=False).reshape(-1)
    diff = cf - rf
    absdiff = np.abs(diff)
    ref_abs_mean = float(np.mean(np.abs(rf))) if rf.size else 0.0
    ref_std = float(np.std(rf)) if rf.size else 0.0
    rmse = float(math.sqrt(float(np.mean(diff * diff)))) if diff.size else 0.0
    out.update({
        "same_size": True,
        "mean_abs": float(np.mean(absdiff)) if absdiff.size else 0.0,
        "median_abs": float(np.median(absdiff)) if absdiff.size else 0.0,
        "p95_abs": float(np.quantile(absdiff, 0.95)) if absdiff.size else 0.0,
        "max_abs": float(np.max(absdiff)) if absdiff.size else 0.0,
        "rmse": rmse,
        "rel_mean_abs": float(np.mean(absdiff) / max(ref_abs_mean, 1e-12)) if absdiff.size else 0.0,
        "rmse_over_ref_std": float(rmse / max(ref_std, 1e-12)),
        "corr": _corr(cf, rf),
        "allclose_1e_3_1e_2": bool(np.allclose(cf, rf, rtol=1e-3, atol=1e-2)),
        "allclose_1e_2_1e_1": bool(np.allclose(cf, rf, rtol=1e-2, atol=1e-1)),
        "affine_candidate_to_reference": _r2_affine(cf, rf),
    })
    return out


def _match_outputs(candidate_names: list[str], candidate_outputs: list[np.ndarray], reference_names: list[str], reference_outputs: list[np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    for ri, (rn, ro) in enumerate(zip(reference_names, reference_outputs)):
        ci: int | None = None
        if rn in candidate_names:
            idx = candidate_names.index(rn)
            if idx not in used:
                ci = idx
        if ci is None:
            # Prefer same shape, then same element count.
            for j, co in enumerate(candidate_outputs):
                if j in used:
                    continue
                if tuple(np.asarray(co).shape) == tuple(np.asarray(ro).shape):
                    ci = j
                    break
        if ci is None:
            for j, co in enumerate(candidate_outputs):
                if j in used:
                    continue
                if int(np.asarray(co).size) == int(np.asarray(ro).size):
                    ci = j
                    break
        if ci is None:
            rows.append({
                "reference_name": rn,
                "reference_index": ri,
                "candidate_status": "missing",
                "reference_summary": _summ(np.asarray(ro)),
            })
            continue
        used.add(ci)
        cn = candidate_names[ci] if ci < len(candidate_names) else f"candidate_{ci}"
        rows.append({
            "reference_name": rn,
            "reference_index": ri,
            "candidate_name": cn,
            "candidate_index": ci,
            "candidate_status": "matched",
            "compare": _tensor_compare(np.asarray(candidate_outputs[ci]), np.asarray(ro)),
        })
    extra = [i for i in range(len(candidate_outputs)) if i not in used]
    for i in extra:
        rows.append({
            "candidate_name": candidate_names[i] if i < len(candidate_names) else f"candidate_{i}",
            "candidate_index": i,
            "candidate_status": "extra",
            "candidate_summary": _summ(np.asarray(candidate_outputs[i])),
        })
    return rows


def _aggregate_compares(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comps = [r.get("compare") or {} for r in rows if r.get("candidate_status") == "matched"]
    comps = [c for c in comps if c.get("same_size")]
    if not comps:
        return {"matched_outputs": 0, "ok_like_control": False, "reason": "no_matched_same_size_outputs"}
    corrs = np.asarray([float(c.get("corr")) for c in comps if _safe_float(c.get("corr")) is not None], dtype=np.float64)
    rels = np.asarray([float(c.get("rel_mean_abs") or 0.0) for c in comps], dtype=np.float64)
    rmses = np.asarray([float(c.get("rmse_over_ref_std") or 0.0) for c in comps], dtype=np.float64)
    maxabs = np.asarray([float(c.get("max_abs") or 0.0) for c in comps], dtype=np.float64)
    allclose_loose = all(bool(c.get("allclose_1e_2_1e_1")) for c in comps)
    corr_min = float(np.min(corrs)) if corrs.size else None
    corr_mean = float(np.mean(corrs)) if corrs.size else None
    rel_mean = float(np.mean(rels)) if rels.size else None
    rmse_mean = float(np.mean(rmses)) if rmses.size else None
    # A deliberately practical tensor-level gate: not a formal proof of semantic
    # equality, but enough to say whether the native boundary produces the same
    # Part2 output family as the ORT oracle.
    ok_like = bool(
        allclose_loose or (
            corr_min is not None and corr_min >= 0.995 and
            rmse_mean is not None and rmse_mean <= 0.10 and
            rel_mean is not None and rel_mean <= 0.10
        )
    )
    return {
        "matched_outputs": int(len(comps)),
        "corr_min": corr_min,
        "corr_mean": corr_mean,
        "rel_mean_abs_mean": rel_mean,
        "rmse_over_ref_std_mean": rmse_mean,
        "max_abs_max": float(np.max(maxabs)) if maxabs.size else None,
        "allclose_1e_2_1e_1_all": bool(allclose_loose),
        "ok_like_control": ok_like,
    }


def _semantic(outputs: list[np.ndarray], names: list[str], reference_report: str) -> dict[str, Any]:
    if not reference_report:
        return {"available": False, "reason": "no_reference_report"}
    try:
        return _semantic_score_outputs(outputs, names, reference_report)
    except Exception as exc:
        return {"available": False, "reason": f"semantic_helper_failed:{type(exc).__name__}:{exc}"}


def _semantic_ratio(sem: dict[str, Any]) -> float:
    try:
        return float((sem.get("match") or {}).get("match_ratio") or 0.0)
    except Exception:
        return 0.0


def _run_ort_part2(sess: Any, input_name: str, feed: np.ndarray, dtype: Any) -> list[np.ndarray]:
    return [np.asarray(x) for x in sess.run(None, {input_name: np.asarray(feed).astype(dtype, copy=False)})]


def _resolve_native_output_manifest(man_path: Path, explicit: str = "") -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p.resolve() if p.is_file() else None
    work = man_path.parent.parent
    candidates = [
        work / "native_fifo_outputs" / "native_fifo_outputs_manifest.json",
        work / "native_outputs" / "native_outputs_manifest.json",
        work / "runner_outputs" / "runner_outputs_manifest.json",
    ]
    # native_fifo_results.json contains the authoritative link if present.
    res = work / "native_fifo_results.json"
    if res.is_file():
        try:
            js = json.loads(res.read_text(encoding="utf-8"))
            v = js.get("native_fifo_output_manifest") or js.get("output_manifest")
            if isinstance(v, str) and v:
                candidates.insert(0, Path(v).expanduser())
        except Exception:
            pass
    for p in candidates:
        if not p.is_absolute():
            p = (work / p).resolve()
        if p.is_file():
            return p.resolve()
    return None


def _output_dict_to_lists(tensors: dict[str, np.ndarray]) -> tuple[list[str], list[np.ndarray]]:
    names = list(tensors.keys())
    return names, [np.asarray(tensors[n]) for n in names]


def _score_candidate(row: dict[str, Any]) -> tuple[float, float, float, float]:
    ag = row.get("part2_output_compare_aggregate") or {}
    return (
        1.0 if ag.get("ok_like_control") else 0.0,
        float(ag.get("corr_min") or -999.0),
        -float(ag.get("rmse_over_ref_std_mean") or 999.0),
        -float(ag.get("rel_mean_abs_mean") or 999.0),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--boundary-manifest", required=True)
    ap.add_argument("--native-output-manifest", default="")
    ap.add_argument("--part2-onnx", default="")
    ap.add_argument("--reference-report", default="")
    ap.add_argument("--image-scale", default="native")
    ap.add_argument("--out", default="")
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = str(ns.case)
    man_path = Path(ns.boundary_manifest).expanduser().resolve()
    man = _read_json(man_path)

    bfile = Path(str(man.get("file") or "")).expanduser()
    if not bfile.is_absolute():
        bfile = (man_path.parent / bfile).resolve()
    if not bfile.is_file():
        raise FileNotFoundError(f"boundary file not found: {bfile}")
    raw = np.fromfile(str(bfile), dtype=_dtype_from_manifest(man))
    raw_shape = [int(x) for x in (man.get("shape") or man.get("boundary_shape") or [])]
    if raw_shape and int(np.prod(raw_shape)) == raw.size:
        raw = raw.reshape(raw_shape)

    part1 = _find_part1_onnx(bs, case)
    part2 = _find_part2_onnx(bs, case, ns.part2_onnx)
    p1_inputs, _p1_outputs_meta = _onnx_io(part1)
    if not p1_inputs:
        raise RuntimeError(f"no inputs in part1 ONNX: {part1}")
    p1_input_name, p1_input_shape = p1_inputs[0]

    ref_report_path = Path(ns.reference_report).expanduser().resolve() if ns.reference_report else None
    boundary_image = _image_from_boundary_manifest(man)
    ref_image = _image_from_report(ref_report_path)
    img_name = boundary_image or ref_image
    img_path = _find_image(bs, img_name) if img_name else None
    feed = _input_dump_feed_from_manifest(man, p1_input_shape, ns.image_scale)
    feed_source = "boundary_manifest_input_dump" if feed is not None else f"image_preprocess:{ns.image_scale}"
    if feed is None:
        if img_path is None:
            raise FileNotFoundError("could not resolve input image and no input_dump is available")
        feed = _preprocess(img_path, p1_input_shape, ns.image_scale)

    p1_sess = ort.InferenceSession(str(part1), providers=["CPUExecutionProvider"])
    p1_outs = [np.asarray(x) for x in p1_sess.run(None, {p1_input_name: np.asarray(feed, dtype=np.float32)})]
    p1_out_names = [o.name for o in p1_sess.get_outputs()]

    p2_sess = ort.InferenceSession(str(part2), providers=["CPUExecutionProvider"])
    p2_in = p2_sess.get_inputs()[0]
    p2_input_name = str(p2_in.name)
    p2_shape = _shape_from_ort(p2_in.shape)
    p2_dtype = _ort_input_dtype(p2_in)
    p2_output_names = [o.name for o in p2_sess.get_outputs()]
    if not p2_shape or any(int(x) <= 0 for x in p2_shape):
        raise RuntimeError(f"Part2 input has dynamic/unknown shape unsupported by this diagnostic: {p2_in.shape}")

    # Pick the canonical Part1 output for the Part2 input.
    ref_candidates: list[tuple[str, np.ndarray]] = []
    for name, arr in zip(p1_out_names, p1_outs):
        aa = np.asarray(arr)
        if name == p2_input_name:
            ref_candidates.insert(0, (name, aa))
        elif int(aa.size) == int(np.prod(p2_shape)):
            ref_candidates.append((name, aa))
    if not ref_candidates:
        raise RuntimeError(f"No Part1 output matches Part2 input {p2_input_name} shape={p2_shape}")
    ref_name, ref_activation = ref_candidates[0]
    if list(ref_activation.shape) != p2_shape and int(ref_activation.size) == int(np.prod(p2_shape)):
        ref_activation = ref_activation.reshape(p2_shape)

    control_outputs = _run_ort_part2(p2_sess, p2_input_name, ref_activation.astype(np.float32), p2_dtype)
    control_semantic = _semantic(control_outputs, p2_output_names, ns.reference_report)
    control_semantic_ratio = _semantic_ratio(control_semantic)

    rows: list[dict[str, Any]] = []
    axis = _channel_axis(tuple(ref_activation.shape))
    seen: set[tuple[str, str]] = set()
    for layout_name, raw_layout in _contract_candidate_layouts(raw, tuple(ref_activation.shape)):
        raw_f = np.asarray(raw_layout, dtype=np.float32)
        variants: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        variants.append(("identity", raw_f, _fit_affine(raw_f, ref_activation)))
        gf = _fit_affine(raw_f, ref_activation)
        if gf.get("ok"):
            a = float(gf.get("scale_fit") or 1.0)
            b = float(gf.get("bias_fit") or 0.0)
            variants.append(("global_affine_fit_to_ort_part1", (raw_f * a + b).astype(np.float32), gf))
        if axis is not None:
            try:
                pc, pc_meta = _apply_per_channel_affine(raw_f, ref_activation, axis)
                variants.append(("per_channel_affine_fit_to_ort_part1", pc.astype(np.float32), pc_meta))
            except Exception as exc:
                rows.append({
                    "kind": "native_boundary",
                    "layout": layout_name,
                    "transform": "per_channel_affine_fit_to_ort_part1",
                    "error": f"{type(exc).__name__}: {exc}",
                    "part2_output_compare_aggregate": {"matched_outputs": 0, "ok_like_control": False, "reason": "per_channel_fit_failed"},
                })
        for transform, candidate_feed, fit in variants:
            key = (layout_name, transform)
            if key in seen:
                continue
            seen.add(key)
            row: dict[str, Any] = {
                "kind": "native_boundary",
                "layout": layout_name,
                "transform": transform,
                "boundary_input_summary": _summ(candidate_feed),
                "fit_to_ort_part1": fit,
            }
            try:
                outs = _run_ort_part2(p2_sess, p2_input_name, candidate_feed, p2_dtype)
                compares = _match_outputs(p2_output_names, outs, p2_output_names, control_outputs)
                row["part2_output_compare"] = compares
                row["part2_output_compare_aggregate"] = _aggregate_compares(compares)
                row["part2_output_summaries"] = [_summ(o) for o in outs]
                row["semantic_vs_external_reference"] = _semantic(outs, p2_output_names, ns.reference_report)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["part2_output_compare_aggregate"] = {"matched_outputs": 0, "ok_like_control": False, "reason": "part2_run_failed"}
            rows.append(row)

    rows_sorted = sorted(rows, key=_score_candidate, reverse=True)
    best_boundary = rows_sorted[0] if rows_sorted else None

    native_output_manifest = _resolve_native_output_manifest(man_path, ns.native_output_manifest)
    native_output_compare: dict[str, Any] = {"available": False, "reason": "native_output_manifest_missing"}
    if native_output_manifest is not None:
        try:
            tensors, meta = load_dump(str(native_output_manifest))
            nnames, nouts = _output_dict_to_lists(tensors)
            comps = _match_outputs(nnames, nouts, p2_output_names, control_outputs)
            native_output_compare = {
                "available": True,
                "manifest": str(native_output_manifest),
                "manifest_meta_schema": meta.get("schema"),
                "output_names": nnames,
                "compare": comps,
                "aggregate": _aggregate_compares(comps),
                "semantic_vs_external_reference": _semantic(nouts, nnames, ns.reference_report),
            }
        except Exception as exc:
            native_output_compare = {"available": False, "manifest": str(native_output_manifest), "reason": f"native_output_load_or_compare_failed:{type(exc).__name__}:{exc}"}

    best_ok_like = bool(((best_boundary or {}).get("part2_output_compare_aggregate") or {}).get("ok_like_control"))
    native_ok_like = bool(((native_output_compare.get("aggregate") or {}) if native_output_compare.get("available") else {}).get("ok_like_control"))

    if control_semantic_ratio < 0.5:
        if best_ok_like or native_ok_like:
            diagnosis = "external_semantic_reference_or_decode_suspect_boundary_matches_oracle"
        else:
            diagnosis = "external_semantic_reference_or_decode_suspect_and_boundary_differs_from_oracle"
    else:
        if best_ok_like and native_ok_like:
            diagnosis = "native_boundary_and_native_trt_match_part2_oracle"
        elif best_ok_like and not native_ok_like:
            diagnosis = "boundary_part2_oracle_ok_but_native_trt_output_path_suspect"
        elif not best_ok_like:
            diagnosis = "native_boundary_does_not_reproduce_part2_oracle"
        else:
            diagnosis = "inconclusive"

    payload = {
        "schema": "onnx-splitpoint/native-part2-oracle-tensor-probe",
        "schema_version": 1,
        "ok": True,
        "diagnosis": diagnosis,
        "benchmark_set": str(bs),
        "case": case,
        "part1_onnx": str(part1),
        "part2_onnx": str(part2),
        "part2_input": {"name": p2_input_name, "shape": p2_shape, "type": str(getattr(p2_in, "type", ""))},
        "part2_outputs": p2_output_names,
        "reference_report": str(ref_report_path) if ref_report_path else "",
        "boundary_manifest": str(man_path),
        "boundary_file": str(bfile),
        "boundary_dtype": str(man.get("dtype") or ""),
        "boundary_shape": [int(x) for x in raw.shape],
        "feed_source": feed_source,
        "boundary_image": boundary_image,
        "reference_image": ref_image,
        "ort_part1_output_used": {"name": ref_name, "shape": [int(v) for v in ref_activation.shape], "summary": _summ(ref_activation)},
        "control_part2_output_summaries": [_summ(o) for o in control_outputs],
        "control_semantic_vs_external_reference": control_semantic,
        "control_semantic_match_ratio": control_semantic_ratio,
        "best_boundary": best_boundary,
        "rows": rows_sorted,
        "native_trt_output_compare_to_oracle": native_output_compare,
    }

    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent / "native_part2_oracle_tensor_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    lines: list[str] = [
        "# Native Part2 oracle tensor probe",
        "",
        f"Diagnosis: `{diagnosis}`",
        f"Boundary manifest: `{man_path}`",
        f"Boundary dtype/shape: `{man.get('dtype')}` `{[int(x) for x in raw.shape]}`",
        f"Feed source: `{feed_source}`",
        f"Part1 ONNX: `{part1}`",
        f"Part2 ONNX: `{part2}`",
        f"Part2 input: `{p2_input_name}` shape=`{p2_shape}` type=`{getattr(p2_in, 'type', '')}`",
        f"ORT Part1 output used: `{ref_name}` shape=`{[int(v) for v in ref_activation.shape]}`",
        f"External semantic reference: `{ns.reference_report or ''}`",
        "",
        "## Guardrail",
        "",
        f"ORT Part1 -> Part2 semantic match vs external reference: `{control_semantic_ratio}`",
        "",
        "If that value is low, the visual/semantic reference or decode path is not a reliable pass/fail oracle for this split replay. The tensor oracle below is then more important.",
        "",
        "## Best native-boundary replays vs ORT Part1->Part2 tensor oracle",
        "",
        "| rank | layout | transform | oracle-like | corr min | rel MAE mean | RMSE/std mean | matched outputs | semantic-vs-external | fit r2/corr | fit scale/bias | error |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(rows_sorted[:30]):
        ag = r.get("part2_output_compare_aggregate") or {}
        sem = r.get("semantic_vs_external_reference") or {}
        fit = r.get("fit_to_ort_part1") or {}
        m = sem.get("match") or {}
        fit_r2 = fit.get("r2") if "r2" in fit else fit.get("r2_median")
        fit_corr = fit.get("corr") if "corr" in fit else fit.get("corr_median")
        fit_scale = fit.get("scale_fit") if "scale_fit" in fit else fit.get("scale_median")
        fit_bias = fit.get("bias_fit") if "bias_fit" in fit else fit.get("bias_median")
        lines.append(
            f"| {i} | `{r.get('layout')}` | `{r.get('transform')}` | {ag.get('ok_like_control')} | "
            f"{ag.get('corr_min')} | {ag.get('rel_mean_abs_mean')} | {ag.get('rmse_over_ref_std_mean')} | "
            f"{ag.get('matched_outputs')} | {m.get('match_ratio')} | {fit_r2}/{fit_corr} | {fit_scale}/{fit_bias} | `{r.get('error','')}` |"
        )
    lines += [
        "",
        "## Native TRT output vs ORT Part1->Part2 tensor oracle",
        "",
    ]
    if native_output_compare.get("available"):
        ag = native_output_compare.get("aggregate") or {}
        sem = native_output_compare.get("semantic_vs_external_reference") or {}
        m = sem.get("match") or {}
        lines += [
            f"Manifest: `{native_output_compare.get('manifest')}`",
            "",
            "| oracle-like | corr min | rel MAE mean | RMSE/std mean | matched outputs | semantic-vs-external |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {ag.get('ok_like_control')} | {ag.get('corr_min')} | {ag.get('rel_mean_abs_mean')} | {ag.get('rmse_over_ref_std_mean')} | {ag.get('matched_outputs')} | {m.get('match_ratio')} |",
        ]
    else:
        lines.append(f"Native output compare unavailable: `{native_output_compare.get('reason')}`")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `external_semantic_reference_or_decode_suspect_boundary_matches_oracle`: the split replay tensors match the ORT oracle, so the current semantic validator/reference is likely misleading.",
        "- `external_semantic_reference_or_decode_suspect_and_boundary_differs_from_oracle`: the semantic reference is still suspect, but the native boundary also does not reproduce the ORT Part2 tensor oracle.",
        "- `boundary_part2_oracle_ok_but_native_trt_output_path_suspect`: ORT replay is OK, native TensorRT output is not; inspect engine bindings/output dump.",
        "- `native_boundary_does_not_reproduce_part2_oracle`: the boundary is numerically insufficient for original Part2, independent of the external semantic validator.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best_ag = (best_boundary or {}).get("part2_output_compare_aggregate") or {}
    native_ag = (native_output_compare.get("aggregate") or {}) if native_output_compare.get("available") else {}
    print(json.dumps({
        "ok": True,
        "out": str(out),
        "summary_md": str(md),
        "diagnosis": diagnosis,
        "control_semantic_match_ratio": control_semantic_ratio,
        "best_boundary_layout": (best_boundary or {}).get("layout"),
        "best_boundary_transform": (best_boundary or {}).get("transform"),
        "best_boundary_oracle_like": best_ag.get("ok_like_control"),
        "best_boundary_corr_min": best_ag.get("corr_min"),
        "best_boundary_rel_mae_mean": best_ag.get("rel_mean_abs_mean"),
        "native_trt_oracle_like": native_ag.get("ok_like_control"),
        "native_trt_corr_min": native_ag.get("corr_min"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
