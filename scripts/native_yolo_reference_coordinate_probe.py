#!/usr/bin/env python3
"""Probe whether YOLO split/native semantic failures are caused by reference coordinate-space mismatch.

This diagnostic starts with the strongest possible positive control:

    exact native input_rgb_uint8 dump -> ORT Part1 -> original Part2 ONNX

It then decodes the Part2 output once and evaluates the same decoded boxes against
reference detections in multiple coordinate contracts:

  * current/raw comparison: decoded boxes in model-input coordinates vs reference
  * prediction de-letterboxed back to the original image coordinates
  * reference boxes letterboxed into the model-input coordinate system
  * direct-resize variants as a fallback

If the positive control is only bad in the raw/current comparison but good after
letterbox coordinate conversion, the Native validation path was producing a false
semantic fail. If the positive control is bad in every coordinate contract, the
problem is not the Hailo boundary yet; the Part2 output decoder/reference report
contract must be fixed before native Hailo->TRT can be judged.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

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
        _input_dump_feed_from_manifest,
        _onnx_io,
        _read_json,
        _summ,
    )
    from native_boundary_affine_replay_probe import _find_part2_onnx, _shape_from_ort, _ort_input_dtype  # type: ignore
    from native_boundary_dequant_sweep import _local_reference_detections  # type: ignore
    from native_producer_validate_visualize import (  # type: ignore
        _decode_layout_candidates,
        _match_detections,
        _match_detections_class_agnostic,
        _nms,
        _reference_detections,
        _score_saturation,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"helper import failed: {type(exc).__name__}: {exc}")


def _find_existing_image(bs: Path, *candidates: str | None) -> Path | None:
    seen: set[str] = set()
    for c in candidates:
        if not c:
            continue
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        p = Path(s).expanduser()
        if p.is_file():
            return p.resolve()
        # Try exact basename and tail relative to benchmark-set.
        for q in [bs / s, bs / Path(s).name]:
            if q.is_file():
                return q.resolve()
        matches = list(bs.rglob(Path(s).name))[:20]
        for m in matches:
            if m.is_file():
                return m.resolve()
    return None


def _image_size(p: Path | None) -> tuple[int | None, int | None]:
    if p is None or not p.is_file():
        return None, None
    try:
        from PIL import Image
        with Image.open(p) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None, None


def _norm_det(d: dict[str, Any]) -> dict[str, Any]:
    out = dict(d)
    # Keep required keys and numeric values stable.
    for k in ("x1", "y1", "x2", "y2", "score"):
        if k in out:
            try:
                out[k] = float(out[k])
            except Exception:
                out[k] = 0.0
    try:
        out["class_id"] = int(out.get("class_id", out.get("cls", out.get("class", -1))))
    except Exception:
        out["class_id"] = -1
    return out


def _clip_det(d: dict[str, Any], w: float, h: float) -> dict[str, Any]:
    out = _norm_det(d)
    out["x1"] = float(min(max(out.get("x1", 0.0), 0.0), w))
    out["x2"] = float(min(max(out.get("x2", 0.0), 0.0), w))
    out["y1"] = float(min(max(out.get("y1", 0.0), 0.0), h))
    out["y2"] = float(min(max(out.get("y2", 0.0), 0.0), h))
    return out


def _letterbox_params(orig_w: int, orig_h: int, model_w: int, model_h: int) -> dict[str, float]:
    gain = min(float(model_w) / max(1.0, float(orig_w)), float(model_h) / max(1.0, float(orig_h)))
    new_w = float(orig_w) * gain
    new_h = float(orig_h) * gain
    pad_x = (float(model_w) - new_w) / 2.0
    pad_y = (float(model_h) - new_h) / 2.0
    return {"gain": gain, "pad_x": pad_x, "pad_y": pad_y, "new_w": new_w, "new_h": new_h}


def _ref_to_letterbox_model(dets: list[dict[str, Any]], orig_w: int, orig_h: int, model_w: int, model_h: int) -> list[dict[str, Any]]:
    p = _letterbox_params(orig_w, orig_h, model_w, model_h)
    g, px, py = p["gain"], p["pad_x"], p["pad_y"]
    out = []
    for d in dets:
        x = _norm_det(d)
        x["x1"] = x["x1"] * g + px
        x["x2"] = x["x2"] * g + px
        x["y1"] = x["y1"] * g + py
        x["y2"] = x["y2"] * g + py
        out.append(_clip_det(x, model_w, model_h))
    return out


def _pred_from_letterbox_model(dets: list[dict[str, Any]], orig_w: int, orig_h: int, model_w: int, model_h: int) -> list[dict[str, Any]]:
    p = _letterbox_params(orig_w, orig_h, model_w, model_h)
    g, px, py = p["gain"], p["pad_x"], p["pad_y"]
    out = []
    for d in dets:
        x = _norm_det(d)
        x["x1"] = (x["x1"] - px) / max(g, 1e-9)
        x["x2"] = (x["x2"] - px) / max(g, 1e-9)
        x["y1"] = (x["y1"] - py) / max(g, 1e-9)
        x["y2"] = (x["y2"] - py) / max(g, 1e-9)
        out.append(_clip_det(x, orig_w, orig_h))
    return out


def _ref_to_resize_model(dets: list[dict[str, Any]], orig_w: int, orig_h: int, model_w: int, model_h: int) -> list[dict[str, Any]]:
    sx = float(model_w) / max(1.0, float(orig_w)); sy = float(model_h) / max(1.0, float(orig_h))
    out = []
    for d in dets:
        x = _norm_det(d)
        x["x1"] *= sx; x["x2"] *= sx; x["y1"] *= sy; x["y2"] *= sy
        out.append(_clip_det(x, model_w, model_h))
    return out


def _pred_from_resize_model(dets: list[dict[str, Any]], orig_w: int, orig_h: int, model_w: int, model_h: int) -> list[dict[str, Any]]:
    sx = float(orig_w) / max(1.0, float(model_w)); sy = float(orig_h) / max(1.0, float(model_h))
    out = []
    for d in dets:
        x = _norm_det(d)
        x["x1"] *= sx; x["x2"] *= sx; x["y1"] *= sy; x["y2"] *= sy
        out.append(_clip_det(x, orig_w, orig_h))
    return out


def _ref_dets(report: Path) -> tuple[list[dict[str, Any]], str | None, str]:
    try:
        dets, img, src = _local_reference_detections(report)
        if dets:
            return [_norm_det(d) for d in dets], img, src
    except Exception:
        pass
    try:
        dets, img = _reference_detections(report)
        if dets:
            return [_norm_det(d) for d in dets], img, "native_producer_validate_visualize._reference_detections"
    except Exception:
        pass
    return [], None, "no_reference_detections"


def _match(ref: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    m = _match_detections(ref, pred) if ref else {"ref_count": 0, "pred_count": len(pred), "matched": 0, "match_ratio": 0.0, "mean_iou": 0.0, "iou_threshold": 0.5}
    ca = _match_detections_class_agnostic(ref, pred) if ref else {"ref_count": 0, "pred_count": len(pred), "matched": 0, "match_ratio": 0.0, "mean_iou": 0.0, "iou_threshold": 0.5, "class_agnostic": True}
    return {"match": m, "class_agnostic_match": ca, "score_saturation": _score_saturation(pred), "pred_count": len(pred)}


def _ratio(row: dict[str, Any]) -> tuple[float, float, float, int, int]:
    m = (row.get("match") or {})
    ca = (row.get("class_agnostic_match") or {})
    sat = (row.get("score_saturation") or {})
    return (
        float(m.get("match_ratio") or 0.0),
        float(ca.get("match_ratio") or 0.0),
        float(m.get("mean_iou") or 0.0),
        int(m.get("matched") or 0),
        -int(sat.get("score_ge_0999") or 0),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-set", required=True)
    ap.add_argument("--case", required=True)
    ap.add_argument("--boundary-manifest", required=True)
    ap.add_argument("--reference-report", required=True)
    ap.add_argument("--image-scale", default="native")
    ap.add_argument("--out", default="")
    ns = ap.parse_args()

    bs = Path(ns.benchmark_set).expanduser().resolve()
    case = str(ns.case)
    man_path = Path(ns.boundary_manifest).expanduser().resolve()
    man = _read_json(man_path)

    part1 = _find_part1_onnx(bs, case)
    part2 = _find_part2_onnx(bs, case, "")
    part1_inputs, part1_outputs = _onnx_io(part1)
    if not part1_inputs:
        raise RuntimeError(f"Part1 has no inputs: {part1}")
    p1_input_name, p1_input_shape = part1_inputs[0]
    feed = _input_dump_feed_from_manifest(man, [int(x) for x in p1_input_shape], ns.image_scale)
    if feed is None:
        raise RuntimeError("boundary manifest has no usable input_dump; rerun native with --native-boundary-debug")

    sess1 = ort.InferenceSession(str(part1), providers=["CPUExecutionProvider"])
    p1_outs = [np.asarray(x) for x in sess1.run(None, {p1_input_name: np.asarray(feed, dtype=np.float32)})]
    p1_names = [o.name for o in sess1.get_outputs()]

    sess2 = ort.InferenceSession(str(part2), providers=["CPUExecutionProvider"])
    p2_in = sess2.get_inputs()[0]
    p2_input_name = p2_in.name
    p2_shape = _shape_from_ort(p2_in.shape)
    p2_dtype = _ort_input_dtype(p2_in)
    p2_names = [o.name for o in sess2.get_outputs()]

    ref_activation = None
    ref_activation_name = ""
    for name, arr in zip(p1_names, p1_outs):
        a = np.asarray(arr)
        if name == p2_input_name:
            ref_activation_name, ref_activation = name, a
            break
    if ref_activation is None:
        for name, arr in zip(p1_names, p1_outs):
            a = np.asarray(arr)
            if a.size == int(np.prod(p2_shape)):
                ref_activation_name, ref_activation = name, a.reshape(p2_shape)
                break
    if ref_activation is None:
        raise RuntimeError(f"No Part1 output matches Part2 input {p2_input_name} shape={p2_shape}")
    if list(ref_activation.shape) != p2_shape:
        ref_activation = ref_activation.reshape(p2_shape)

    outs = [np.asarray(x) for x in sess2.run(None, {p2_input_name: np.asarray(ref_activation, dtype=p2_dtype)})]
    tensors = {str(p2_names[i] if i < len(p2_names) else f"output{i}"): np.asarray(o) for i, o in enumerate(outs)}

    ref_report = Path(ns.reference_report).expanduser().resolve()
    ref_dets, ref_image, ref_source = _ref_dets(ref_report)
    boundary_image = str(man.get("input_image") or (man.get("provenance") or {}).get("image") or man.get("image") or "")
    img_path = _find_existing_image(bs, ref_image, boundary_image)
    orig_w, orig_h = _image_size(img_path)
    input_shape_hwc = man.get("input_shape_hwc") or (man.get("preprocess") or {}).get("input_shape_hwc") or []
    if input_shape_hwc and len(input_shape_hwc) >= 2:
        model_h, model_w = int(input_shape_hwc[0]), int(input_shape_hwc[1])
    elif len(feed.shape) == 4 and feed.shape[1] in (1, 3):
        model_h, model_w = int(feed.shape[2]), int(feed.shape[3])
    elif len(feed.shape) == 4:
        model_h, model_w = int(feed.shape[1]), int(feed.shape[2])
    else:
        model_h, model_w = 640, 640

    ref_variants: dict[str, list[dict[str, Any]]] = {"ref_original": ref_dets}
    pred_transforms: dict[str, Any] = {"pred_identity": lambda d: d}
    geometry: dict[str, Any] = {"orig_w": orig_w, "orig_h": orig_h, "model_w": model_w, "model_h": model_h, "image_path": str(img_path) if img_path else ""}
    if orig_w and orig_h:
        lbp = _letterbox_params(int(orig_w), int(orig_h), int(model_w), int(model_h))
        geometry["letterbox"] = lbp
        ref_variants["ref_letterboxed_to_model"] = _ref_to_letterbox_model(ref_dets, int(orig_w), int(orig_h), int(model_w), int(model_h))
        ref_variants["ref_resized_to_model"] = _ref_to_resize_model(ref_dets, int(orig_w), int(orig_h), int(model_w), int(model_h))
        pred_transforms["pred_deletterboxed_to_original"] = lambda d, ow=int(orig_w), oh=int(orig_h), mw=int(model_w), mh=int(model_h): _pred_from_letterbox_model(d, ow, oh, mw, mh)
        pred_transforms["pred_resize_inverse_to_original"] = lambda d, ow=int(orig_w), oh=int(orig_h), mw=int(model_w), mh=int(model_h): _pred_from_resize_model(d, ow, oh, mw, mh)

    candidates = _decode_layout_candidates(tensors, img_w=int(model_w), img_h=int(model_h), conf=0.25)
    rows: list[dict[str, Any]] = []
    for c in candidates:
        base_dets = _nms(c.get("detections") or [])
        for pred_space, transform in pred_transforms.items():
            try:
                pred_dets = transform(base_dets)
            except TypeError:
                pred_dets = transform(base_dets)  # type: ignore[misc]
            for ref_space, rds in ref_variants.items():
                # Compare compatible spaces only, plus keep current raw baseline.
                compatible = (
                    (pred_space == "pred_identity" and ref_space in {"ref_original", "ref_letterboxed_to_model", "ref_resized_to_model"})
                    or (pred_space == "pred_deletterboxed_to_original" and ref_space == "ref_original")
                    or (pred_space == "pred_resize_inverse_to_original" and ref_space == "ref_original")
                )
                if not compatible:
                    continue
                scored = _match(rds, pred_dets)
                row = {
                    "mode": c.get("mode"),
                    "kind": c.get("kind"),
                    "pred_space": pred_space,
                    "ref_space": ref_space,
                    "candidate_count_before_nms": len(c.get("detections") or []),
                    "debug": c.get("debug") or {},
                    **scored,
                }
                rows.append(row)

    rows.sort(key=_ratio, reverse=True)
    best = rows[0] if rows else {"match": {"match_ratio": 0.0, "matched": 0, "ref_count": len(ref_dets)}, "class_agnostic_match": {"match_ratio": 0.0}}
    cur = None
    for r in rows:
        if r.get("pred_space") == "pred_identity" and r.get("ref_space") == "ref_original":
            cur = r; break

    best_ratio = float(((best.get("match") or {}).get("match_ratio")) or 0.0)
    cur_ratio = float((((cur or {}).get("match") or {}).get("match_ratio")) or 0.0)
    best_ref_space = str(best.get("ref_space") or "")
    best_pred_space = str(best.get("pred_space") or "")

    if not ref_dets:
        diagnosis = "reference_missing_or_unreadable"
    elif cur_ratio < 0.5 and best_ratio >= 0.5 and ("letterbox" in best_ref_space or "deletterbox" in best_pred_space):
        diagnosis = "semantic_validation_coordinate_space_mismatch"
    elif best_ratio >= 0.5:
        diagnosis = "part1_part2_control_semantically_ok_with_nondefault_decode_contract"
    else:
        diagnosis = "part2_output_decode_or_reference_report_contract_suspect"

    payload = {
        "schema": "onnx-splitpoint/native-yolo-reference-coordinate-probe",
        "schema_version": 1,
        "ok": True,
        "diagnosis": diagnosis,
        "benchmark_set": str(bs),
        "case": case,
        "part1_onnx": str(part1),
        "part2_onnx": str(part2),
        "part2_input": {"name": p2_input_name, "shape": p2_shape, "type": str(getattr(p2_in, "type", ""))},
        "part2_outputs": p2_names,
        "part1_output_used": {"name": ref_activation_name, "shape": [int(x) for x in ref_activation.shape], "summary": _summ(ref_activation)},
        "part2_output_summaries": [_summ(o) for o in outs],
        "boundary_manifest": str(man_path),
        "reference_report": str(ref_report),
        "reference_source": ref_source,
        "reference_image": ref_image,
        "boundary_image": boundary_image,
        "reference_count": len(ref_dets),
        "geometry": geometry,
        "current_raw_match": cur,
        "best": best,
        "rows": rows,
    }

    out = Path(ns.out).expanduser().resolve() if ns.out else man_path.parent / "native_yolo_reference_coordinate_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = out.with_suffix(".md")
    lines = [
        "# Native YOLO reference coordinate contract probe",
        "",
        f"Diagnosis: `{diagnosis}`",
        f"Part1 ONNX: `{part1}`",
        f"Part2 ONNX: `{part2}`",
        f"Boundary manifest: `{man_path}`",
        f"Reference report: `{ref_report}`",
        f"Reference source: `{ref_source}`",
        f"Reference image: `{ref_image}`",
        f"Boundary image: `{boundary_image}`",
        f"Resolved image: `{geometry.get('image_path','')}`",
        f"Original size: `{orig_w}x{orig_h}`  model size: `{model_w}x{model_h}`",
        f"Letterbox params: `{geometry.get('letterbox')}`",
        "",
        "## Summary",
        "",
        f"Current raw match: `{((cur or {}).get('match') or {}).get('matched',0)}/{((cur or {}).get('match') or {}).get('ref_count',len(ref_dets))}` ratio=`{cur_ratio}`",
        f"Best match: `{(best.get('match') or {}).get('matched',0)}/{(best.get('match') or {}).get('ref_count',len(ref_dets))}` ratio=`{best_ratio}` mode=`{best.get('mode')}` pred_space=`{best_pred_space}` ref_space=`{best_ref_space}`",
        "",
        "| rank | mode | pred space | ref space | match | class-agnostic | pred | top score saturation |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows[:30]):
        m = r.get("match") or {}; ca = r.get("class_agnostic_match") or {}; sat = r.get("score_saturation") or {}
        lines.append(
            f"| {i} | `{r.get('mode')}` | `{r.get('pred_space')}` | `{r.get('ref_space')}` | "
            f"{m.get('matched')}/{m.get('ref_count')} ({m.get('match_ratio')}) | "
            f"{ca.get('matched')}/{ca.get('ref_count')} ({ca.get('match_ratio')}) | "
            f"{r.get('pred_count')} | {sat.get('score_ge_0999')} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `semantic_validation_coordinate_space_mismatch`: the ORT Part1 -> Part2 control is good after letterbox/deletterbox conversion; patch native semantic validation coordinate handling before judging Hailo.",
        "- `part2_output_decode_or_reference_report_contract_suspect`: even the ORT control cannot reproduce the reference under tested coordinate spaces; fix Part2 decoder/reference selection first.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "out": str(out),
        "summary_md": str(md),
        "diagnosis": diagnosis,
        "current_raw_match_ratio": cur_ratio,
        "best_match_ratio": best_ratio,
        "best_mode": best.get("mode"),
        "best_pred_space": best_pred_space,
        "best_ref_space": best_ref_space,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
