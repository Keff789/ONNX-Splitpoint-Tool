#!/usr/bin/env python3
"""Probe the pure ORT split-chain contract against the full model/reference.

This diagnostic is intentionally hardware-free.  It answers whether the local
validation reference, the Part1/Part2 ONNX pair, and the YOLO decoder agree
before blaming Native Hailo/TensorRT.

It runs, using the exact native input dump when available:

  A) full ONNX with ORT -> semantic proxy against reference_report
  B) Part1 ONNX with ORT -> candidate Part1 boundary outputs -> Part2 ONNX with ORT
     -> semantic proxy against reference_report
  C) tensor compare: split-chain outputs vs full ONNX outputs

If A already fails, the Native/Hailo boundary is not yet the right target; the
reference/decode/full-model contract must be fixed first.  If A passes but B
fails, the split export/Part2 input contract is suspect.
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
    _ORT_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - helpers remain importable for contract-only diagnostics
    ort = None  # type: ignore
    _ORT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

try:
    from native_boundary_activation_compare import (  # type: ignore
        _find_part1_onnx,
        _input_dump_feed_from_manifest,
        _onnx_io,
        _preprocess,
        _find_image,
        _image_from_boundary_manifest,
        _image_from_report,
        _read_json,
        _summ,
    )
    from native_part2_boundary_layout_sweep import _find_part2_onnx  # type: ignore
    from native_boundary_dequant_sweep import _semantic_score_outputs  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"helper import failed: {type(exc).__name__}: {exc}")


def _shape_from_ort(shape: Iterable[Any]) -> list[int]:
    out: list[int] = []
    for d in shape:
        try:
            iv = int(d)
            out.append(iv if iv > 0 else -1)
        except Exception:
            out.append(-1)
    return out


def _ort_dtype(inp: Any):
    t = str(getattr(inp, "type", "") or "").lower()
    if "uint8" in t:
        return np.uint8
    if "int8" in t:
        return np.int8
    if "float16" in t:
        return np.float16
    return np.float32


def _find_full_onnx(bs: Path, explicit: str = "") -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(p)
        return p.resolve()
    models = bs / "models"
    if models.is_dir():
        cands = [p for p in sorted(models.glob("*.onnx")) if "part" not in p.name.lower() and "bridge" not in p.name.lower()]
        if cands:
            return cands[0].resolve()
        cands = sorted(models.glob("*.onnx"))
        if cands:
            return cands[0].resolve()
    cands = [p for p in sorted(bs.glob("*.onnx")) if "part" not in p.name.lower() and "bridge" not in p.name.lower()]
    if cands:
        return cands[0].resolve()
    cands = [p for p in sorted(bs.glob("**/*.onnx")) if "part" not in p.name.lower() and "bridge" not in p.name.lower()]
    # Avoid accidentally choosing exported split products under case dirs if a full model exists elsewhere.
    filtered = [p for p in cands if f"/{p.parent.name}/" not in str(p)]
    return (filtered[0] if filtered else (cands[0] if cands else None))


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return None
    sx = float(np.std(x)); sy = float(np.std(y))
    if sx <= 0 or sy <= 0:
        return None
    v = float(np.corrcoef(x, y)[0, 1])
    return v if math.isfinite(v) else None


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    aa = np.asarray(a); bb = np.asarray(b)
    out: dict[str, Any] = {
        "shape_a": [int(x) for x in aa.shape],
        "shape_b": [int(x) for x in bb.shape],
        "dtype_a": str(aa.dtype),
        "dtype_b": str(bb.dtype),
        "same_shape": bool(tuple(aa.shape) == tuple(bb.shape)),
        "same_size": bool(aa.size == bb.size),
    }
    if aa.size != bb.size:
        return out
    x = aa.astype(np.float64).reshape(-1)
    y = bb.astype(np.float64).reshape(-1)
    d = x - y
    out.update({
        "mean_abs": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(math.sqrt(float(np.mean(d * d)))),
        "corr": _corr(x, y),
        "allclose_1e_4_1e_3": bool(np.allclose(x, y, rtol=1e-4, atol=1e-3)),
        "allclose_1e_3_1e_2": bool(np.allclose(x, y, rtol=1e-3, atol=1e-2)),
        "summary_a": _summ(aa),
        "summary_b": _summ(bb),
    })
    return out


def _semantic(outputs: list[np.ndarray], names: list[str], reference_report: str) -> dict[str, Any]:
    if not reference_report:
        return {"available": False, "reason": "no_reference_report"}
    try:
        return _semantic_score_outputs(outputs, names, reference_report)
    except Exception as exc:
        return {"available": False, "reason": f"semantic_helper_failed:{type(exc).__name__}:{exc}"}


def _match_ratio(sem: dict[str, Any]) -> float:
    try:
        return float((sem.get("match") or {}).get("match_ratio") or 0.0)
    except Exception:
        return 0.0


def _feed_for_session(sess: Any, preferred_feed: np.ndarray) -> tuple[str, np.ndarray, list[int], str]:
    inp = sess.get_inputs()[0]
    name = inp.name
    shape = _shape_from_ort(inp.shape)
    dtype = _ort_dtype(inp)
    arr = np.asarray(preferred_feed)
    source = "preferred"
    if shape and all(int(x) > 0 for x in shape):
        if tuple(arr.shape) != tuple(shape):
            if arr.size == int(np.prod(shape)):
                arr = arr.reshape(shape)
                source = "preferred_reshaped"
            else:
                raise RuntimeError(f"feed shape {list(arr.shape)} incompatible with session input {name} shape={shape}")
    return name, arr.astype(dtype, copy=False), shape, source


def _run_session(path: Path, feed: np.ndarray) -> tuple[list[str], list[np.ndarray], dict[str, Any]]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name, arr, shape, source = _feed_for_session(sess, feed)
    outs = [np.asarray(x) for x in sess.run(None, {name: arr})]
    names = [o.name for o in sess.get_outputs()]
    meta = {"input_name": name, "input_shape": shape, "feed_source": source, "input_summary": _summ(arr)}
    return names, outs, meta


def _find_best_output_compare(split_names: list[str], split_outs: list[np.ndarray], full_names: list[str], full_outs: list[np.ndarray]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, so in enumerate(split_outs):
        sname = split_names[i] if i < len(split_names) else f"split_output_{i}"
        # Name match first, then same shape, then same element count.
        candidates: list[tuple[str, np.ndarray, str]] = []
        for j, fo in enumerate(full_outs):
            fname = full_names[j] if j < len(full_names) else f"full_output_{j}"
            if fname == sname:
                candidates.insert(0, (fname, fo, "name"))
            elif tuple(np.asarray(fo).shape) == tuple(np.asarray(so).shape):
                candidates.append((fname, fo, "shape"))
            elif np.asarray(fo).size == np.asarray(so).size:
                candidates.append((fname, fo, "size"))
        if not candidates:
            rows.append({"split_output": sname, "error": "no_full_output_candidate"})
            continue
        for fname, fo, why in candidates[:5]:
            rows.append({"split_output": sname, "full_output": fname, "match_by": why, "compare": _compare(so, fo)})
    def key(r: dict[str, Any]) -> tuple[float, float]:
        c = r.get("compare") or {}
        corr = c.get("corr")
        mae = c.get("mean_abs")
        return (float(corr) if corr is not None else -999.0, -float(mae) if mae is not None else -1e30)
    rows = sorted(rows, key=key, reverse=True)
    return {"best": rows[0] if rows else None, "rows": rows[:20]}


def main() -> int:
    if ort is None:
        raise RuntimeError(f"onnxruntime import failed: {_ORT_IMPORT_ERROR}")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--boundary-manifest", default="", help="Optional native boundary manifest; used only to get exact input_dump.")
    ap.add_argument("--full-onnx", default="")
    ap.add_argument("--part2-onnx", default="")
    ap.add_argument("--reference-report", default="")
    ap.add_argument("--image-scale", default="native")
    ap.add_argument("--out", default="")
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = str(ns.case)
    part1 = _find_part1_onnx(bs, case)
    part2 = _find_part2_onnx(bs, case, ns.part2_onnx)
    full = _find_full_onnx(bs, ns.full_onnx)

    p1_inputs, _ = _onnx_io(part1)
    if not p1_inputs:
        raise RuntimeError(f"Part1 has no inputs: {part1}")
    p1_input_name, p1_input_shape = p1_inputs[0]

    man: dict[str, Any] = {}
    man_path: Path | None = None
    feed = None
    feed_source = ""
    if ns.boundary_manifest:
        man_path = Path(ns.boundary_manifest).expanduser().resolve()
        man = _read_json(man_path)
        feed = _input_dump_feed_from_manifest(man, [int(x) for x in p1_input_shape], ns.image_scale)
        if feed is not None:
            feed_source = "boundary_manifest_input_dump"
    if feed is None:
        ref_report = Path(ns.reference_report).expanduser().resolve() if ns.reference_report else None
        img_name = _image_from_boundary_manifest(man) or _image_from_report(ref_report)
        img = _find_image(bs, img_name) if img_name else None
        if img is None:
            raise RuntimeError("no boundary input_dump and no reference/boundary image could be resolved")
        feed = _preprocess(img, [int(x) for x in p1_input_shape], ns.image_scale)
        feed_source = f"image_preprocess:{ns.image_scale}"

    # Part1 run.
    p1_sess = ort.InferenceSession(str(part1), providers=["CPUExecutionProvider"])
    p1_input = p1_sess.get_inputs()[0]
    p1_feed = np.asarray(feed).astype(_ort_dtype(p1_input), copy=False)
    p1_outs = [np.asarray(x) for x in p1_sess.run(None, {p1_input.name: p1_feed})]
    p1_out_names = [o.name for o in p1_sess.get_outputs()]

    # Part2 session and candidate boundaries.
    p2_sess = ort.InferenceSession(str(part2), providers=["CPUExecutionProvider"])
    p2_in = p2_sess.get_inputs()[0]
    p2_input_name = p2_in.name
    p2_shape = _shape_from_ort(p2_in.shape)
    p2_dtype = _ort_dtype(p2_in)
    p2_out_names = [o.name for o in p2_sess.get_outputs()]
    if not p2_shape or any(int(x) <= 0 for x in p2_shape):
        raise RuntimeError(f"Part2 dynamic/unknown input shape unsupported: {p2_in.shape}")

    boundary_candidates: list[tuple[str, np.ndarray, str]] = []
    for name, arr in zip(p1_out_names, p1_outs):
        a = np.asarray(arr)
        if name == p2_input_name:
            boundary_candidates.insert(0, (name, a, "name"))
        elif tuple(a.shape) == tuple(p2_shape):
            boundary_candidates.append((name, a, "shape"))
        elif a.size == int(np.prod(p2_shape)):
            boundary_candidates.append((name, a.reshape(p2_shape), "size_reshape"))
    if not boundary_candidates:
        raise RuntimeError(f"No Part1 output candidate for Part2 input {p2_input_name} shape={p2_shape}")

    split_rows: list[dict[str, Any]] = []
    for cname, carr, cwhy in boundary_candidates:
        try:
            split_outs = [np.asarray(x) for x in p2_sess.run(None, {p2_input_name: np.asarray(carr).astype(p2_dtype, copy=False)})]
            sem = _semantic(split_outs, p2_out_names, ns.reference_report)
            split_rows.append({
                "candidate_name": cname,
                "candidate_match_by": cwhy,
                "candidate_summary": _summ(carr),
                "semantic": sem,
                "outputs_summary": [_summ(x) for x in split_outs],
                "output_names": p2_out_names,
                "outputs": split_outs,  # removed before JSON serialization
            })
        except Exception as exc:
            split_rows.append({"candidate_name": cname, "candidate_match_by": cwhy, "error": f"{type(exc).__name__}: {exc}", "semantic": {"available": False, "reason": "part2_run_failed"}, "outputs": []})

    split_rows_sorted = sorted(split_rows, key=lambda r: _match_ratio(r.get("semantic") or {}), reverse=True)
    best_split = split_rows_sorted[0] if split_rows_sorted else None
    best_split_ratio = _match_ratio((best_split or {}).get("semantic") or {})

    full_names: list[str] = []
    full_outs: list[np.ndarray] = []
    full_meta: dict[str, Any] = {}
    full_sem: dict[str, Any] = {"available": False, "reason": "full_model_missing"}
    output_compare: dict[str, Any] = {"best": None, "rows": []}
    if full is not None:
        try:
            full_names, full_outs, full_meta = _run_session(full, np.asarray(feed))
            full_sem = _semantic(full_outs, full_names, ns.reference_report)
            if best_split and best_split.get("outputs"):
                output_compare = _find_best_output_compare(best_split.get("output_names") or [], best_split.get("outputs") or [], full_names, full_outs)
        except Exception as exc:
            full_sem = {"available": False, "reason": f"full_model_run_failed:{type(exc).__name__}:{exc}"}

    full_ratio = _match_ratio(full_sem)
    best_cmp = output_compare.get("best") or {}
    best_cmp_obj = best_cmp.get("compare") or {}
    split_full_corr = best_cmp_obj.get("corr")
    split_full_close = bool(best_cmp_obj.get("allclose_1e_3_1e_2")) if best_cmp_obj else False

    if full is None:
        diagnosis = "full_model_missing_cannot_decide_reference_contract"
    elif full_ratio < 0.5:
        if split_full_corr is not None and float(split_full_corr) > 0.999:
            diagnosis = "semantic_decoder_or_reference_suspect_split_matches_full"
        else:
            diagnosis = "reference_or_yolo_decode_contract_suspect"
    elif best_split_ratio >= 0.5:
        diagnosis = "ort_split_chain_contract_ok" if (split_full_close or (split_full_corr is not None and float(split_full_corr) > 0.999)) else "ort_split_semantic_ok_but_tensor_differs_from_full"
    else:
        diagnosis = "ort_split_chain_part2_contract_suspect"

    # Remove ndarray objects before serialization.
    json_split_rows = []
    for r in split_rows_sorted:
        rr = dict(r)
        rr.pop("outputs", None)
        json_split_rows.append(rr)

    payload = {
        "schema": "onnx-splitpoint/split-chain-reference-contract-probe",
        "schema_version": 1,
        "ok": True,
        "diagnosis": diagnosis,
        "benchmark_set": str(bs),
        "case": case,
        "feed_source": feed_source,
        "boundary_manifest": str(man_path) if man_path else "",
        "part1_onnx": str(part1),
        "part2_onnx": str(part2),
        "full_onnx": str(full) if full else "",
        "reference_report": str(Path(ns.reference_report).expanduser().resolve()) if ns.reference_report else "",
        "part1_input": {"name": p1_input.name, "shape": _shape_from_ort(p1_input.shape), "type": str(getattr(p1_input, "type", "")), "summary": _summ(p1_feed)},
        "part1_outputs": [{"name": n, "summary": _summ(a)} for n, a in zip(p1_out_names, p1_outs)],
        "part2_input": {"name": p2_input_name, "shape": p2_shape, "type": str(getattr(p2_in, "type", ""))},
        "full_input": full_meta,
        "full_semantic": full_sem,
        "best_split_semantic": (best_split or {}).get("semantic"),
        "best_split_candidate": {k: v for k, v in (best_split or {}).items() if k not in {"outputs", "semantic"}},
        "full_match_ratio": full_ratio,
        "best_split_match_ratio": best_split_ratio,
        "best_split_vs_full_output_compare": output_compare,
        "split_candidates": json_split_rows,
    }

    out = Path(ns.out).expanduser().resolve() if ns.out else (man_path.parent / "split_chain_reference_contract_probe.json" if man_path else bs / case / "split_chain_reference_contract_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# Split-chain reference contract probe",
        "",
        f"Diagnosis: `{diagnosis}`",
        f"Benchmark set: `{bs}`",
        f"Case: `{case}`",
        f"Feed source: `{feed_source}`",
        f"Part1 ONNX: `{part1}`",
        f"Part2 ONNX: `{part2}`",
        f"Full ONNX: `{full or ''}`",
        f"Reference: `{ns.reference_report or ''}`",
        "",
        "## Summary",
        "",
        f"- full ONNX semantic match ratio: `{full_ratio}`",
        f"- best ORT split-chain semantic match ratio: `{best_split_ratio}`",
        f"- best split-vs-full output corr: `{split_full_corr}`",
        f"- best split-vs-full allclose 1e-3/1e-2: `{split_full_close}`",
        "",
        "## Split candidates",
        "",
        "| rank | Part1 output | selected by | semantic match | pred | decode mode |",
        "|---:|---|---|---:|---:|---|",
    ]
    for i, r in enumerate(json_split_rows[:20]):
        sem = r.get("semantic") or {}
        m = sem.get("match") or {}
        lines.append(f"| {i} | `{r.get('candidate_name')}` | `{r.get('candidate_match_by')}` | {m.get('matched')}/{m.get('ref_count','')} ({m.get('match_ratio')}) | {sem.get('pred_count')} | `{sem.get('decode_mode')}` |")
    lines += ["", "## Best split vs full output compare", "", "| field | value |", "|---|---:|"]
    c = best_cmp_obj
    for k in ["same_shape", "same_size", "mean_abs", "max_abs", "rmse", "corr", "allclose_1e_4_1e_3", "allclose_1e_3_1e_2"]:
        lines.append(f"| `{k}` | `{c.get(k)}` |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- `reference_or_yolo_decode_contract_suspect`: even the full ONNX does not match the selected reference with this semantic decoder; fix reference/decode first.",
        "- `semantic_decoder_or_reference_suspect_split_matches_full`: split and full tensors agree, but semantic proxy fails; the decoder/reference is the issue.",
        "- `ort_split_chain_part2_contract_suspect`: full ONNX is good but ORT Part1→Part2 is bad; inspect split export, selected Part1 output, and Part2 input contract.",
        "- `ort_split_chain_contract_ok`: pure ORT split chain is good; then Native boundary/runner remains the target.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "out": str(out),
        "summary_md": str(md),
        "diagnosis": diagnosis,
        "full_match_ratio": full_ratio,
        "best_split_match_ratio": best_split_ratio,
        "best_split_vs_full_corr": split_full_corr,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
