from __future__ import annotations

import json
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._types import SampleCfg
from .base import Harness, PostprocessResult, PostprocessResultLike


def _load_labels_from_assets(filename: str) -> Optional[List[str]]:
    """Load a label list from ``splitpoint_runners/assets`` (best effort)."""

    try:
        runners_root = Path(__file__).resolve().parents[1]
        p = runners_root / "assets" / filename
        if not p.is_file():
            return None
        labels = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
        labels = [ln for ln in labels if ln]
        return labels or None
    except Exception:
        return None


# COCO 80 labels (common for YOLO models). Used for nicer overlays.
_COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    # Clip for numerical stability
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert [...,4] xywh (center) to xyxy."""

    x, y, w, h = xywh[..., 0], xywh[..., 1], xywh[..., 2], xywh[..., 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=-1)


def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between one box a[4] and many boxes b[N,4] (xyxy)."""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Very small, dependency-free NMS."""

    if boxes.size == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _bbox_iou_xyxy(boxes[i], boxes[rest])
        order = rest[ious < iou_thresh]
    return keep


@dataclass
class _Detections:
    boxes_xyxy: np.ndarray  # [N,4] in model-input pixel coords
    scores: np.ndarray  # [N]
    class_ids: np.ndarray  # [N]


def _decode_bn6(output: np.ndarray) -> _Detections:
    """Decode already-materialized detections [N,6] or [1,N,6]."""

    det = np.asarray(output)
    if det.ndim == 3 and det.shape[0] == 1:
        det = det[0]
    if det.ndim != 2 or det.shape[1] != 6:
        raise ValueError(f"bn6 detections expected shape [N,6] or [1,N,6], got {det.shape}")
    boxes = det[:, 0:4].astype(np.float32, copy=False)
    scores = det[:, 4].astype(np.float32, copy=False)
    cls = det[:, 5].astype(np.int64, copy=False)
    return _Detections(boxes_xyxy=boxes, scores=scores, class_ids=cls)


def _decode_concat_yolo(output: np.ndarray, conf_thresh: float) -> _Detections:
    """Decode concatenated YOLO output [1,N,5+nc] or [N,5+nc].

    Assumes output is already in (cx,cy,w,h,obj,cls...) where cls are per-class
    probabilities/logits.
    """

    out = np.asarray(output)
    if out.ndim == 3 and out.shape[0] == 1:
        out = out[0]
    if out.ndim != 2 or out.shape[1] < 6:
        raise ValueError(f"concat YOLO output expected [N,>=6], got {out.shape}")

    xywh = out[:, 0:4]
    obj = out[:, 4]
    cls_scores = out[:, 5:]

    # If class scores look like logits, softmax is expensive; use sigmoid for
    # typical YOLO (which uses sigmoid per class).
    cls_probs = _sigmoid(cls_scores)
    cls_id = np.argmax(cls_probs, axis=1)
    cls_conf = cls_probs[np.arange(cls_probs.shape[0]), cls_id]
    score = _sigmoid(obj) * cls_conf

    keep = score >= conf_thresh
    if not np.any(keep):
        return _Detections(boxes_xyxy=np.zeros((0, 4), np.float32), scores=np.zeros((0,), np.float32), class_ids=np.zeros((0,), np.int64))

    boxes = _xywh_to_xyxy(xywh[keep]).astype(np.float32)
    return _Detections(boxes_xyxy=boxes, scores=score[keep].astype(np.float32), class_ids=cls_id[keep].astype(np.int64))


# Default YOLOv5-style anchors for 640 input (works reasonably for many YOLOv5/7 exports).
_YOLOV5_ANCHORS_640 = {
    8: np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),
    16: np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),
    32: np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32),
}


def _yolo_multiscale_name_hint(name: str) -> Tuple[int, int]:
    import re as _re

    lname = str(name or "").strip().lower()
    if not lname:
        return (99, 9999)

    token_ranks = (
        (r"\bp3\b", 0),
        (r"\bp4\b", 1),
        (r"\bp5\b", 2),
        (r"\bsmall\b", 0),
        (r"\bmedium\b", 1),
        (r"\blarge\b", 2),
        (r"stride[_:/-]?8\b", 0),
        (r"stride[_:/-]?16\b", 1),
        (r"stride[_:/-]?32\b", 2),
        (r"\bs8\b", 0),
        (r"\bs16\b", 1),
        (r"\bs32\b", 2),
    )
    for pattern, rank in token_ranks:
        if _re.search(pattern, lname):
            return (rank, 0)

    m = _re.search(r"(?:layer|head|output|out|branch|scale)[_:/-]?(\d+)", lname)
    if m:
        try:
            return (10, int(m.group(1)))
        except Exception:
            pass

    return (99, 9999)


def _try_parse_suffix_int(name: str) -> Optional[int]:
    base = str(name or '').strip()
    if not base:
        return None
    i = len(base) - 1
    while i >= 0 and base[i].isdigit():
        i -= 1
    if i == len(base) - 1:
        return None
    try:
        return int(base[i + 1 :])
    except Exception:
        return None


def _infer_ch_sp_any(x: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    try:
        shp = tuple(int(s) for s in np.asarray(x).shape)
    except Exception:
        return None, None
    a = np.asarray(x)
    if a.ndim == 4 and len(shp) == 4:
        d1, d2, d3 = shp[1], shp[2], shp[3]
        if d1 <= 512 and d2 > 1 and d3 > 1:
            return d1, d2 * d3
        if d3 <= 512 and d1 > 1 and d2 > 1:
            return d3, d1 * d2
        if d1 <= d3:
            return d1, d2 * d3
        return d3, d1 * d2
    if a.ndim == 3 and len(shp) == 3:
        d1, d2 = shp[1], shp[2]
        if d1 <= 512 and d2 > d1:
            return d1, d2
        if d2 <= 512 and d1 > d2:
            return d2, d1
        return (d1, d2) if d1 <= d2 else (d2, d1)
    return None, None


def _get_ultralytics_regcls_pairs(
    output_names: Optional[Sequence[str]],
    outputs: Sequence[np.ndarray],
) -> List[Tuple[int, int, int]]:
    names = [str(n) for n in (list(output_names) if output_names is not None else [])]
    names_ok = bool(names) and len(names) == len(outputs)
    pairs: List[Tuple[int, int, int]] = []

    if names_ok:
        reg_map: Dict[int, int] = {}
        cls_map: Dict[int, int] = {}
        for i, n in enumerate(names):
            base = n.strip().split('/')[-1].split(':')[-1]
            low = base.lower()
            if low.startswith('reg'):
                k = _try_parse_suffix_int(low)
                if k is not None:
                    reg_map[k] = i
            elif low.startswith('cls'):
                k = _try_parse_suffix_int(low)
                if k is not None:
                    cls_map[k] = i
        common = sorted(set(reg_map.keys()) & set(cls_map.keys()))
        if len(common) >= 2:
            for k in common:
                ri = reg_map[k]
                ci = cls_map[k]
                r = np.asarray(outputs[ri])
                c = np.asarray(outputs[ci])
                if r.ndim not in (3, 4) or c.ndim not in (3, 4):
                    continue
                pairs.append((int(k), int(ri), int(ci)))
    if len(pairs) >= 2:
        return pairs

    descs: List[Tuple[int, int, int]] = []
    for i, out in enumerate(outputs):
        a = np.asarray(out)
        if a.ndim not in (3, 4):
            continue
        ch, sp = _infer_ch_sp_any(a)
        if ch is None or sp is None:
            continue
        descs.append((int(i), int(ch), int(sp)))
    if len(descs) < 4:
        return []

    by_ch: Dict[int, List[Tuple[int, int]]] = {}
    for out_idx, ch, sp in descs:
        by_ch.setdefault(ch, []).append((out_idx, sp))
    if len(by_ch) < 2:
        return []
    ch_sorted = sorted(by_ch.keys(), key=lambda c: len(by_ch[c]), reverse=True)
    ch_a, ch_b = ch_sorted[0], ch_sorted[1]

    def _sp_map(ch: int) -> Dict[int, List[int]]:
        m: Dict[int, List[int]] = {}
        for out_idx, sp in by_ch.get(ch, []):
            m.setdefault(sp, []).append(out_idx)
        return m

    sp_a = _sp_map(ch_a)
    sp_b = _sp_map(ch_b)
    common_sp = sorted(set(sp_a.keys()) & set(sp_b.keys()), reverse=True)
    if len(common_sp) < 2:
        return []

    def _reg_score(ch: int) -> int:
        if ch % 4 != 0:
            return -1
        reg_max = ch // 4
        if reg_max < 4 or reg_max > 64:
            return 0
        bonus = 0
        if reg_max in (8, 16, 24, 32):
            bonus += 2
        if reg_max in (12, 20):
            bonus += 1
        return 1 + bonus

    reg_ch, cls_ch = ch_a, ch_b
    if _reg_score(ch_b) > _reg_score(ch_a):
        reg_ch, cls_ch = ch_b, ch_a
    elif _reg_score(ch_a) == _reg_score(ch_b) and ch_b < ch_a:
        reg_ch, cls_ch = ch_b, ch_a

    sp_reg = _sp_map(reg_ch)
    sp_cls = _sp_map(cls_ch)
    pairs_fb: List[Tuple[int, int, int]] = []
    for level_idx, sp in enumerate(common_sp[:3], start=1):
        r_list = sp_reg.get(sp, [])
        c_list = sp_cls.get(sp, [])
        if len(r_list) != 1 or len(c_list) != 1:
            continue
        pairs_fb.append((int(level_idx), int(r_list[0]), int(c_list[0])))
    if len(pairs_fb) >= 2:
        return pairs_fb
    return []


def _is_ultralytics_regcls(outputs: Sequence[np.ndarray], output_names: Optional[Sequence[str]] = None) -> bool:
    pairs = _get_ultralytics_regcls_pairs(output_names, outputs)
    if len(pairs) < 2:
        return False
    for _lvl, ri, _ci in pairs:
        r = np.asarray(outputs[ri])
        if r.ndim == 4:
            if r.shape[1] >= 4 and r.shape[1] % 4 == 0:
                return True
            if r.shape[-1] >= 4 and r.shape[-1] % 4 == 0:
                return True
        elif r.ndim == 3:
            if r.shape[1] >= 4 and r.shape[1] % 4 == 0:
                return True
            if r.shape[-1] >= 4 and r.shape[-1] % 4 == 0:
                return True
    return False


def _normalize_multiscale_outputs(
    outputs: Sequence[np.ndarray],
    output_names: Optional[Sequence[str]] = None,
    *,
    na_hint: int = 3,
) -> Tuple[List[str], List[np.ndarray], Dict[str, Any]]:
    raw_names = [str(n) for n in (list(output_names) if output_names is not None else [])]
    if len(raw_names) < len(outputs):
        raw_names.extend([f"output_{i}" for i in range(len(raw_names), len(outputs))])

    records: List[Dict[str, Any]] = []
    for idx, out in enumerate(outputs):
        raw = np.asarray(out)
        can = _canonicalize_multiscale_head_output(raw, na_hint=na_hint)
        gh = int(can.shape[2])
        gw = int(can.shape[3])
        records.append(
            {
                "index": int(idx),
                "raw_name": raw_names[idx],
                "raw_shape": tuple(int(x) for x in raw.shape),
                "tensor": can.astype(np.float32, copy=False),
                "gh": gh,
                "gw": gw,
                "ch": int(can.shape[-1]),
            }
        )

    def _sort_key(rec: Dict[str, Any]) -> Tuple[int, int, int, int, int, int]:
        hint_rank, hint_index = _yolo_multiscale_name_hint(str(rec.get("raw_name") or ""))
        gh = int(rec["gh"])
        gw = int(rec["gw"])
        area = gh * gw
        return (-area, -gh, -gw, hint_rank, hint_index, int(rec["index"]))

    ordered = sorted(records, key=_sort_key)

    counts: Dict[str, int] = {}
    canonical_names: List[str] = []
    canonical_outputs: List[np.ndarray] = []
    mappings: List[Dict[str, Any]] = []

    for rec in ordered:
        base_name = f"yolo_head_{int(rec['gh'])}x{int(rec['gw'])}"
        counts[base_name] = counts.get(base_name, 0) + 1
        canonical_name = base_name if counts[base_name] == 1 else f"{base_name}_{counts[base_name]}"

        canonical_names.append(canonical_name)
        canonical_outputs.append(rec["tensor"])
        mappings.append(
            {
                "raw_name": str(rec["raw_name"]),
                "raw_shape": [int(x) for x in rec["raw_shape"]],
                "canonical_name": canonical_name,
                "canonical_shape": [int(x) for x in np.asarray(rec["tensor"]).shape],
                "grid_hw": [int(rec["gh"]), int(rec["gw"])],
                "channels": int(rec["ch"]),
            }
        )

    return canonical_names, canonical_outputs, {"format": "multiscale_head", "normalized": True, "mappings": mappings}


def _canonicalize_multiscale_head_output(output: np.ndarray, na_hint: int = 3) -> np.ndarray:
    """Normalize one YOLO multiscale tensor to ``[B,na,gh,gw,ch]``.

    Besides the usual ORT layouts, Hailo commonly returns channels-last tensors
    without an explicit batch dimension, e.g. ``(80, 80, 255)``. Those tensors
    are semantically fine, but older postprocessing rejected them and therefore
    produced no detection overlays.
    """

    a = np.asarray(output)

    if a.ndim == 5:
        if a.shape[1] == na_hint and a.shape[-1] >= 6:
            return a.astype(np.float32, copy=False)
        if a.shape[-2] == na_hint and a.shape[-1] >= 6:
            return a.transpose(0, 3, 1, 2, 4).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported multiscale head layout: {a.shape}")

    if a.ndim == 4:
        # [B,C,H,W]
        if a.shape[0] == 1 and a.shape[1] >= 6 * na_hint and a.shape[1] % na_hint == 0:
            b, ch, gh, gw = a.shape
            return a.reshape(b, na_hint, ch // na_hint, gh, gw).transpose(0, 1, 3, 4, 2).astype(np.float32, copy=False)

        # [B,H,W,C] (common for Hailo)
        if a.shape[0] == 1 and a.shape[-1] >= 6 * na_hint and a.shape[-1] % na_hint == 0:
            b, gh, gw, ch = a.shape
            return a.reshape(b, gh, gw, na_hint, ch // na_hint).transpose(0, 3, 1, 2, 4).astype(np.float32, copy=False)

        # [na,H,W,ch]
        if a.shape[0] == na_hint and a.shape[-1] >= 6:
            return a[None, ...].astype(np.float32, copy=False)

        # [H,W,na,ch]
        if a.shape[-2] == na_hint and a.shape[-1] >= 6:
            return a.transpose(2, 0, 1, 3)[None, ...].astype(np.float32, copy=False)

        raise ValueError(f"Unsupported multiscale head layout: {a.shape}")

    if a.ndim == 3:
        # [H,W,C] (common for Hailo outputs)
        gh, gw, ch = a.shape
        if ch >= 6 * na_hint and ch % na_hint == 0:
            return a.reshape(1, gh, gw, na_hint, ch // na_hint).transpose(0, 3, 1, 2, 4).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported multiscale head output rank: {a.shape}")

def _describe_outputs(outputs: Mapping[str, np.ndarray] | Dict[str, np.ndarray]) -> Dict[str, List[int]]:
    desc: Dict[str, List[int]] = {}
    for name, value in outputs.items():
        try:
            desc[str(name)] = [int(x) for x in np.asarray(value).shape]
        except Exception:
            desc[str(name)] = []
    return desc


def _clip_unit_interval(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def _infer_multiscale_head_activation_mode(outputs: Sequence[np.ndarray]) -> str:
    """Infer whether YOLO heads look like logits or already-sigmoid activations.

    Hailo-compiled exports frequently surface YOLO heads after the sigmoid step
    (or at least with objectness/class channels already in probability space).
    Decoding those tensors with the standard logits-based path applies sigmoid a
    second time, which lifts near-zero background scores to ~0.25 and causes a
    flood of false positives. Detect that case from simple value-range stats.
    """
    if not outputs:
        return "logits"

    samples: list[np.ndarray] = []
    for out in outputs:
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim < 5:
            continue
        flat = arr.reshape(-1, int(arr.shape[-1]))
        if flat.size == 0:
            continue
        if flat.shape[0] > 8192:
            flat = flat[:8192]
        samples.append(flat)
    if not samples:
        return "logits"

    flat = np.concatenate(samples, axis=0)
    xywh = flat[:, 0:4]
    obj = flat[:, 4]
    cls = flat[:, 5:] if flat.shape[1] > 5 else flat[:, 4:5]

    def _frac_unit(a: np.ndarray) -> float:
        aa = np.asarray(a, dtype=np.float32)
        if aa.size == 0:
            return 0.0
        return float(np.mean((aa >= -1e-4) & (aa <= 1.0 + 1e-4)))

    frac_xywh_unit = _frac_unit(xywh)
    frac_obj_unit = _frac_unit(obj)
    frac_cls_unit = _frac_unit(cls)
    q99_abs = float(np.nanpercentile(np.abs(flat), 99.0)) if flat.size else 0.0

    if frac_obj_unit >= 0.995 and frac_cls_unit >= 0.995:
        if frac_xywh_unit >= 0.98 and q99_abs <= 1.25:
            return "activated"
        return "objcls_activated"
    return "logits"


def _decode_multiscale_head_once(
    outputs: Sequence[np.ndarray],
    input_hw: Tuple[int, int],
    conf_thresh: float,
    *,
    activation_mode: str,
) -> _Detections:
    ih, iw = input_hw

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_cls: List[np.ndarray] = []

    for p in outputs:
        b, na, gh, gw, ch = p.shape
        if b != 1:
            raise ValueError("Only batch=1 supported for YOLO postprocess")
        if ch < 6:
            raise ValueError(f"Invalid multiscale head channel dim: {ch}")
        nc = ch - 5

        stride_h = ih / float(gh) if gh > 0 else 0.0
        stride = int(round(stride_h)) if stride_h > 0 else None
        if stride is None or stride <= 0:
            raise ValueError("Could not infer stride")

        anchors = _YOLOV5_ANCHORS_640.get(stride)
        if anchors is None:
            anchors = np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32) * (stride / 8.0)

        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack([xs, ys], axis=-1).astype(np.float32)[None, ...]

        raw = p[0]
        if activation_mode == "activated":
            txy = _clip_unit_interval(raw[..., 0:2])
            twh = _clip_unit_interval(raw[..., 2:4])
            obj = _clip_unit_interval(raw[..., 4])
            cls = _clip_unit_interval(raw[..., 5:])
        elif activation_mode == "objcls_activated":
            txy = _sigmoid(raw[..., 0:2])
            twh = _sigmoid(raw[..., 2:4])
            obj = _clip_unit_interval(raw[..., 4])
            cls = _clip_unit_interval(raw[..., 5:])
        else:
            txy = _sigmoid(raw[..., 0:2])
            twh = _sigmoid(raw[..., 2:4])
            obj = _sigmoid(raw[..., 4])
            cls = _sigmoid(raw[..., 5:])

        xy = (txy * 2.0 - 0.5 + grid) * float(stride)
        wh = (twh * 2.0) ** 2 * anchors[:, None, None, :]
        xywh = np.concatenate([xy, wh], axis=-1)
        boxes = _xywh_to_xyxy(xywh).reshape(-1, 4)

        cls_flat = cls.reshape(-1, nc)
        cls_id = np.argmax(cls_flat, axis=1)
        cls_conf = cls_flat[np.arange(cls_flat.shape[0]), cls_id]
        score = (obj.reshape(-1) * cls_conf).astype(np.float32)

        keep = score >= conf_thresh
        if not np.any(keep):
            continue

        all_boxes.append(boxes[keep])
        all_scores.append(score[keep])
        all_cls.append(cls_id[keep].astype(np.int64))

    if not all_boxes:
        return _Detections(
            boxes_xyxy=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            class_ids=np.zeros((0,), np.int64),
        )

    boxes_xyxy = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_cls, axis=0)
    return _Detections(boxes_xyxy=boxes_xyxy.astype(np.float32), scores=scores.astype(np.float32), class_ids=class_ids.astype(np.int64))


def _multiscale_candidate_quality(det: _Detections) -> float:
    n = int(det.scores.size)
    if n <= 0:
        return 0.0
    topk = np.sort(det.scores.astype(np.float32))[-min(10, n):]
    mean_top = float(np.mean(topk)) if topk.size else 0.0
    # Penalize floods of borderline detections. Correct decodes usually produce
    # fewer, stronger boxes than the double-sigmoid failure mode.
    return mean_top - 0.01 * float(min(n, 200))


def _decode_multiscale_head(outputs: Sequence[np.ndarray], input_hw: Tuple[int, int], conf_thresh: float) -> _Detections:
    """Decode YOLOv5/YOLOv7-style multi-scale head outputs.

    Supports common layouts:
    - [B,na,gh,gw,ch]
    - [B,ch,gh,gw] where ch = na*(5+nc)
    - Hailo packed channels-last heads such as [gh,gw,255]

    The decoder auto-detects whether the heads look like raw logits or already
    sigmoid-activated tensors (common with Hailo builds) and chooses the more
    plausible decoding path.
    """

    if len(outputs) < 3:
        raise ValueError("multiscale head expects >=3 outputs")

    _norm_names, norm, _norm_meta = _normalize_multiscale_outputs(outputs)

    inferred_mode = _infer_multiscale_head_activation_mode(norm)
    modes_to_try = [inferred_mode]
    if inferred_mode != "logits":
        modes_to_try.append("logits")
    modes_to_try = list(dict.fromkeys(modes_to_try))

    best_det: Optional[_Detections] = None
    best_quality = -1e18
    for mode in modes_to_try:
        det = _decode_multiscale_head_once(norm, input_hw=input_hw, conf_thresh=conf_thresh, activation_mode=mode)
        quality = _multiscale_candidate_quality(det)
        if best_det is None or quality > best_quality:
            best_det = det
            best_quality = quality

    if best_det is None:
        return _Detections(
            boxes_xyxy=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            class_ids=np.zeros((0,), np.int64),
        )
    return best_det

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _decode_ultralytics_regcls(outputs: Sequence[np.ndarray], input_hw: Tuple[int, int], conf_thresh: float, output_names: Optional[Sequence[str]] = None) -> _Detections:
    """Decode Ultralytics reg/cls head outputs (YOLOv8/10/11-style).

    Supports both aggregated two-tensor exports and split-per-level exports such
    as ``reg1, cls1, reg2, cls2, reg3, cls3``.
    """

    if len(outputs) < 2:
        raise ValueError("ultralytics_regcls expects at least 2 outputs")

    def _to_nchw_any(t: np.ndarray, H_img: int, W_img: int, expected_hw: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int]:
        a = np.asarray(t)
        if a.ndim == 4:
            h1, w1 = int(a.shape[2]), int(a.shape[3])
            ok1 = h1 > 1 and w1 > 1 and (H_img % h1 == 0) and (W_img % w1 == 0)
            h2, w2 = int(a.shape[1]), int(a.shape[2])
            ok2 = h2 > 1 and w2 > 1 and (H_img % h2 == 0) and (W_img % w2 == 0)
            if expected_hw is not None:
                ok1 = ok1 and (h1, w1) == expected_hw
                ok2 = ok2 and (h2, w2) == expected_hw
            if ok1 and not ok2:
                return a.astype(np.float32, copy=False), max(1, W_img // w1)
            if ok2 and not ok1:
                return np.transpose(a, (0, 3, 1, 2)).astype(np.float32, copy=False), max(1, W_img // w2)
            if ok1 and ok2:
                in_ar = float(H_img) / float(W_img)
                ar1 = float(h1) / float(w1)
                ar2 = float(h2) / float(w2)
                d1 = abs(ar1 - in_ar)
                d2 = abs(ar2 - in_ar)
                if d1 + 1e-6 < d2:
                    return a.astype(np.float32, copy=False), max(1, W_img // w1)
                if d2 + 1e-6 < d1:
                    return np.transpose(a, (0, 3, 1, 2)).astype(np.float32, copy=False), max(1, W_img // w2)
                if int(a.shape[1]) >= int(a.shape[3]):
                    return a.astype(np.float32, copy=False), max(1, W_img // w1)
                return np.transpose(a, (0, 3, 1, 2)).astype(np.float32, copy=False), max(1, W_img // w2)
            return a.astype(np.float32, copy=False), max(1, W_img // int(a.shape[-1]))
        if a.ndim == 3:
            b, d1, d2 = a.shape
            def _sqrt_int(n: int) -> Optional[int]:
                r = int(round(n ** 0.5))
                if r > 0 and r * r == n:
                    return r
                return None
            hw_a = _sqrt_int(int(d2))
            ok_a = hw_a is not None and (H_img % hw_a == 0) and (W_img % hw_a == 0)
            if expected_hw is not None:
                ok_a = ok_a and (hw_a, hw_a) == expected_hw
            hw_b = _sqrt_int(int(d1))
            ok_b = hw_b is not None and (H_img % hw_b == 0) and (W_img % hw_b == 0)
            if expected_hw is not None:
                ok_b = ok_b and (hw_b, hw_b) == expected_hw
            if ok_a and not ok_b:
                return a.reshape(b, d1, hw_a, hw_a).astype(np.float32, copy=False), max(1, W_img // hw_a)
            if ok_b and not ok_a:
                return np.transpose(a, (0, 2, 1)).reshape(b, d2, hw_b, hw_b).astype(np.float32, copy=False), max(1, W_img // hw_b)
            if ok_a and ok_b:
                if d1 <= d2:
                    return a.reshape(b, d1, hw_a, hw_a).astype(np.float32, copy=False), max(1, W_img // hw_a)
                return np.transpose(a, (0, 2, 1)).reshape(b, d2, hw_b, hw_b).astype(np.float32, copy=False), max(1, W_img // hw_b)
        raise ValueError(f"Unsupported tensor rank: {a.ndim}")

    pairs = _get_ultralytics_regcls_pairs(output_names, outputs)
    if len(pairs) >= 2:
        H_img, W_img = input_hw
        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_cls: List[np.ndarray] = []
        for _lvl, reg_i, cls_i in pairs:
            try:
                reg_nchw, stride = _to_nchw_any(np.asarray(outputs[reg_i]), H_img, W_img)
                h, w = int(reg_nchw.shape[2]), int(reg_nchw.shape[3])
                cls_nchw, _ = _to_nchw_any(np.asarray(outputs[cls_i]), H_img, W_img, expected_hw=(h, w))
            except Exception:
                continue
            if reg_nchw.shape[0] != 1 or cls_nchw.shape[0] != 1:
                continue
            reg_ch = int(reg_nchw.shape[1])
            if reg_ch % 4 != 0:
                continue
            reg_max = reg_ch // 4
            if reg_max > 1:
                x = reg_nchw.reshape(1, 4, reg_max, h, w)
                prob = _softmax(x, axis=2)
                bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
                dist = np.sum(prob * bins, axis=2) * float(stride)
            else:
                dist = np.maximum(reg_nchw.reshape(1, 4, h, w), 0.0) * float(stride)
            gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            cx = (gx.astype(np.float32) + 0.5) * float(stride)
            cy = (gy.astype(np.float32) + 0.5) * float(stride)
            l, t, r, btm = dist[0, 0], dist[0, 1], dist[0, 2], dist[0, 3]
            boxes = np.stack([np.clip(cx - l, 0.0, float(W_img)), np.clip(cy - t, 0.0, float(H_img)), np.clip(cx + r, 0.0, float(W_img)), np.clip(cy + btm, 0.0, float(H_img))], axis=-1).reshape(-1, 4)
            cls_prob = _sigmoid(cls_nchw[0])
            best_cls = np.argmax(cls_prob, axis=0).reshape(-1)
            best_score = np.max(cls_prob, axis=0).reshape(-1)
            keep = best_score >= float(conf_thresh)
            if not np.any(keep):
                continue
            all_boxes.append(boxes[keep].astype(np.float32))
            all_scores.append(best_score[keep].astype(np.float32))
            all_cls.append(best_cls[keep].astype(np.int64))
        if all_boxes:
            return _Detections(boxes_xyxy=np.concatenate(all_boxes, axis=0), scores=np.concatenate(all_scores, axis=0), class_ids=np.concatenate(all_cls, axis=0))

    a0 = np.asarray(outputs[0]).astype(np.float32, copy=False)
    a1 = np.asarray(outputs[1]).astype(np.float32, copy=False)

    # Heuristic: reg has last/second dim multiple of 4 and relatively small vs cls.
    reg, cls = (a0, a1) if a0.shape[-1] != a1.shape[-1] else (a0, a1)

    # Make reg shape [1, A, 4, reg_max]
    if reg.ndim != 3:
        raise ValueError(f"reg tensor expected rank-3, got {reg.shape}")
    if reg.shape[1] < reg.shape[2]:
        # likely [1, 4*reg_max, A]
        reg = reg.transpose(0, 2, 1)
    # now [1, A, 4*reg_max]
    if reg.shape[-1] % 4 != 0:
        raise ValueError(f"reg last dim must be divisible by 4, got {reg.shape}")
    reg_max = reg.shape[-1] // 4
    reg = reg.reshape(1, reg.shape[1], 4, reg_max)

    # Make cls shape [1, A, C]
    if cls.ndim != 3:
        raise ValueError(f"cls tensor expected rank-3, got {cls.shape}")
    if cls.shape[1] < cls.shape[2]:
        # likely [1, C, A]
        cls = cls.transpose(0, 2, 1)
    # now [1, A, C]
    num_classes = cls.shape[-1]

    # Figure out feature map shapes from A and input size. Common: 80x80 + 40x40 + 20x20 = 8400.
    ih, iw = input_hw
    # Candidate strides for typical 640 input. Fallback to {8,16,32}.
    strides = [8, 16, 32]
    grid_sizes: List[Tuple[int, int, int]] = []
    for s in strides:
        gh = int(round(ih / s))
        gw = int(round(iw / s))
        grid_sizes.append((gh, gw, s))
    A_expected = sum(gh * gw for gh, gw, _ in grid_sizes)
    if reg.shape[1] != A_expected:
        # Fallback: try 4-scale variant (adds stride 64)
        strides2 = [8, 16, 32, 64]
        grid_sizes = []
        for s in strides2:
            gh = int(round(ih / s))
            gw = int(round(iw / s))
            grid_sizes.append((gh, gw, s))
        A_expected2 = sum(gh * gw for gh, gw, _ in grid_sizes)
        if reg.shape[1] != A_expected2:
            # Best effort: treat A as a single flat grid
            grid_sizes = [(int(math.sqrt(reg.shape[1])), int(math.sqrt(reg.shape[1])), int(round(ih / math.sqrt(reg.shape[1]))))]

    # Build anchor points
    anchor_points: List[np.ndarray] = []
    for gh, gw, stride in grid_sizes:
        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        ap = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float32) + 0.5
        ap = ap * float(stride)
        anchor_points.append(ap)
    points = np.concatenate(anchor_points, axis=0)
    if points.shape[0] != reg.shape[1]:
        # As a last resort, truncate/pad
        points = points[: reg.shape[1]]

    # Decode DFL: softmax over reg_max bins
    prob = _softmax(reg, axis=-1)  # [1,A,4,reg_max]
    bins = np.arange(reg_max, dtype=np.float32)
    dist = (prob * bins).sum(axis=-1)  # [1,A,4]
    # dist = [l,t,r,b]
    lt = dist[:, :, 0:2]
    rb = dist[:, :, 2:4]
    xy1 = points[None, :, :] - lt
    xy2 = points[None, :, :] + rb
    boxes = np.concatenate([xy1, xy2], axis=-1)[0]  # [A,4]

    cls_prob = _sigmoid(cls[0])
    cls_id = np.argmax(cls_prob, axis=1)
    cls_conf = cls_prob[np.arange(cls_prob.shape[0]), cls_id]
    keep = cls_conf >= conf_thresh
    if not np.any(keep):
        return _Detections(
            boxes_xyxy=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            class_ids=np.zeros((0,), np.int64),
        )
    return _Detections(
        boxes_xyxy=boxes[keep].astype(np.float32),
        scores=cls_conf[keep].astype(np.float32),
        class_ids=cls_id[keep].astype(np.int64),
    )


def _detect_yolo_format(outputs: Sequence[np.ndarray], output_names: Optional[Sequence[str]] = None) -> str:
    if not outputs:
        return "unknown"
    if len(outputs) == 1:
        out = np.asarray(outputs[0])
        if out.shape[-1] == 6:
            return "bn6_detections"
        if out.ndim in (2, 3) and out.shape[-1] >= 6:
            return "concat"
        return "unknown"
    if _is_ultralytics_regcls(outputs, output_names=output_names):
        return "ultralytics_regcls"
    if len(outputs) == 2:
        return "ultralytics_regcls"
    if len(outputs) >= 3:
        return "multiscale_head"
    return "unknown"


class YoloHarness:
    def __init__(
        self,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        max_det: int = 300,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_det = int(max_det)

        # Class names are primarily a UI/UX concern. Prefer a versioned asset
        # file, allow the caller (manifest/CLI) to override, and fall back to a
        # baked-in COCO80 list.
        if class_names is None:
            class_names = _load_labels_from_assets("coco80_labels.txt") or list(_COCO80)
        self.class_names = list(class_names)

    def make_inputs(self, sample_cfg: SampleCfg) -> Dict[str, np.ndarray]:
        """Load and preprocess an image for YOLO-style detection models."""

        image_path = sample_cfg.image_path
        if not image_path:
            raise ValueError("YoloHarness.make_inputs: sample_cfg.image_path is empty")

        input_name = sample_cfg.input_name or "images"
        n, c, h, w = sample_cfg.input_shape
        if n != 1 or c != 3:
            raise ValueError(f"YoloHarness.make_inputs expects NCHW with N=1,C=3, got {sample_cfg.input_shape}")

        try:
            from PIL import Image

            im = Image.open(str(image_path)).convert("RGB")

            # Letterbox resize (preserve aspect ratio with padding). This is the
            # standard preprocessing for YOLO-family detectors and avoids
            # systematic box misalignment on non-square images.
            ow, oh = im.size
            if ow <= 0 or oh <= 0:
                raise ValueError(f"Invalid image size: {im.size}")
            gain = min(w / float(ow), h / float(oh))
            new_w = int(round(ow * gain))
            new_h = int(round(oh * gain))
            im_resized = im.resize((new_w, new_h), resample=Image.BILINEAR)
            canvas = Image.new("RGB", (w, h), color=(114, 114, 114))
            pad_left = (w - new_w) // 2
            pad_top = (h - new_h) // 2
            canvas.paste(im_resized, (pad_left, pad_top))
            im = canvas

            arr = np.asarray(im, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load image for YOLO inputs: {image_path}") from e

        # HWC -> CHW -> NCHW
        chw = np.transpose(arr, (2, 0, 1))
        nchw = chw[None, ...]

        scale = sample_cfg.input_scale
        if scale == "norm" or scale == "auto":
            nchw = nchw / 255.0
        elif scale == "raw":
            pass
        else:
            # Be permissive: unknown => treat as norm.
            nchw = nchw / 255.0

        return {input_name: nchw.astype(np.float32, copy=False)}

    def accuracy_proxy(
        self, outputs_full: Dict[str, np.ndarray], outputs_composed: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        # Not required for Phase 2 refactor (kept for future use).
        return {}

    def postprocess(self, outputs: Dict[str, np.ndarray], context: Optional[Dict[str, Any]] = None) -> PostprocessResultLike:
        if not outputs:
            raise ValueError("YoloHarness.postprocess: outputs is empty")

        # Preserve order (insertion order in dict is stable in py3.7+)
        raw_output_names = [str(k) for k in outputs.keys()]
        output_names = list(raw_output_names)
        out_list = [np.asarray(v) for v in outputs.values()]
        fmt = _detect_yolo_format(out_list, output_names=output_names)
        raw_output_shapes = _describe_outputs(outputs)
        decoded_output_shapes = dict(raw_output_shapes)
        output_normalization: Optional[Dict[str, Any]] = None
        multiscale_activation_hint: Optional[str] = None
        if fmt == "multiscale_head":
            try:
                output_names, out_list, output_normalization = _normalize_multiscale_outputs(
                    out_list,
                    output_names=output_names,
                )
                decoded_output_shapes = {name: [int(x) for x in np.asarray(arr).shape] for name, arr in zip(output_names, out_list)}
                multiscale_activation_hint = _infer_multiscale_head_activation_mode(out_list)
            except Exception:
                output_normalization = None

        input_hw = (640, 640)
        if context and context.get("input_hw") is not None:
            ihw = context["input_hw"]
            if isinstance(ihw, (tuple, list)) and len(ihw) == 2:
                input_hw = (int(ihw[0]), int(ihw[1]))

        det: _Detections
        if fmt == "bn6_detections":
            det = _decode_bn6(out_list[0])
        elif fmt == "concat":
            det = _decode_concat_yolo(out_list[0], conf_thresh=self.conf_thresh)
        elif fmt == "ultralytics_regcls":
            det = _decode_ultralytics_regcls(out_list, input_hw=input_hw, conf_thresh=self.conf_thresh, output_names=output_names)
        elif fmt == "multiscale_head":
            det = _decode_multiscale_head(out_list, input_hw=input_hw, conf_thresh=self.conf_thresh)
        else:
            # Unknown: don't crash; return empty.
            det = _Detections(
                boxes_xyxy=np.zeros((0, 4), np.float32),
                scores=np.zeros((0,), np.float32),
                class_ids=np.zeros((0,), np.int64),
            )

        # NMS per class
        boxes = det.boxes_xyxy
        scores = det.scores
        cls = det.class_ids

        keep_all: List[int] = []
        for c in np.unique(cls) if cls.size else []:
            idx = np.where(cls == c)[0]
            kept = _nms_xyxy(boxes[idx], scores[idx], iou_thresh=self.iou_thresh)
            keep_all.extend(idx[k] for k in kept)
        keep_all = sorted(set(keep_all), key=lambda i: float(scores[i]), reverse=True)
        if self.max_det > 0:
            keep_all = keep_all[: self.max_det]

        boxes = boxes[keep_all] if keep_all else boxes[:0]
        scores = scores[keep_all] if keep_all else scores[:0]
        cls = cls[keep_all] if keep_all else cls[:0]

        # Rescale boxes to original image if we have it.
        orig_wh: Optional[Tuple[int, int]] = None
        image_path = None
        if context:
            image_path = context.get("image_path")
        if image_path:
            try:
                from PIL import Image

                with Image.open(str(image_path)) as im:
                    orig_wh = im.size  # (w,h)
            except Exception:
                orig_wh = None

        boxes_out = boxes.copy()
        if orig_wh is not None:
            # Invert the same letterbox transform used in preprocessing:
            # 1) remove padding, 2) divide by uniform gain.
            ow, oh = orig_wh
            ih, iw = input_hw

            if ow > 0 and oh > 0 and iw > 0 and ih > 0:
                gain = min(iw / float(ow), ih / float(oh))
                new_w = int(round(ow * gain))
                new_h = int(round(oh * gain))
                pad_left = (iw - new_w) // 2
                pad_top = (ih - new_h) // 2

                boxes_out[:, 0] = (boxes_out[:, 0] - pad_left) / gain
                boxes_out[:, 2] = (boxes_out[:, 2] - pad_left) / gain
                boxes_out[:, 1] = (boxes_out[:, 1] - pad_top) / gain
                boxes_out[:, 3] = (boxes_out[:, 3] - pad_top) / gain

                boxes_out[:, 0] = np.clip(boxes_out[:, 0], 0, ow)
                boxes_out[:, 2] = np.clip(boxes_out[:, 2], 0, ow)
                boxes_out[:, 1] = np.clip(boxes_out[:, 1], 0, oh)
                boxes_out[:, 3] = np.clip(boxes_out[:, 3], 0, oh)

        # Build JSON-friendly list
        det_list: List[Dict[str, Any]] = []
        for i in range(boxes_out.shape[0]):
            cid = int(cls[i])
            label = str(cid)
            if 0 <= cid < len(self.class_names):
                label = self.class_names[cid]
            x1, y1, x2, y2 = [float(v) for v in boxes_out[i].tolist()]
            det_list.append(
                {
                    "class_id": cid,
                    "label": label,
                    "score": float(scores[i]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )

        variant_name = str(context.get("variant", "")) if context else ""
        provenance: Dict[str, Any] = {
            "source": "actual_variant_outputs",
            "variant": variant_name or None,
            "output_names": raw_output_names,
            "output_shapes": raw_output_shapes,
            "input_hw": [int(input_hw[0]), int(input_hw[1])],
            "decoder": {
                "format": fmt,
                "conf_thresh": float(self.conf_thresh),
                "iou_thresh": float(self.iou_thresh),
                "max_det": int(self.max_det),
            },
        }
        if multiscale_activation_hint is not None:
            try:
                provenance["decoder"]["activation_mode_hint"] = str(multiscale_activation_hint)
            except Exception:
                pass
        if output_normalization is not None:
            provenance["output_normalization"] = output_normalization
            provenance["decoded_output_names"] = list(output_names)
            provenance["decoded_output_shapes"] = decoded_output_shapes
        if image_path:
            try:
                provenance["image"] = Path(str(image_path)).name
            except Exception:
                provenance["image"] = str(image_path)

        artifacts: Dict[str, str] = {}
        json_obj: Dict[str, Any] = {
            "task": "detection",
            "format": fmt,
            "variant": (variant_name or None),
            "viz_enabled": True,
            "viz_error": None,
            "detections_generated_from_variant": (variant_name or None),
            "n_detections": int(len(det_list)),
            "detections": det_list,
            "provenance": provenance,
            "artifacts": artifacts,
        }
        overlays: Dict[str, str] = {}
        summary = f"n={len(det_list)}" if det_list else "n=0"

        if context and context.get("output_dir"):
            out_dir = Path(str(context.get("output_dir")))
            variant = variant_name
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            json_path = out_dir / (f"detections_{variant}.json" if variant else "detections.json")
            artifacts["json"] = json_path.name

            # Overlay PNG
            if image_path:
                try:
                    from PIL import Image, ImageDraw, ImageFont

                    im = Image.open(str(image_path)).convert("RGB")
                    draw = ImageDraw.Draw(im)
                    font = ImageFont.load_default()
                    for d in det_list[:200]:
                        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                        txt = f"{d['label']} {d['score']:.2f}"
                        draw.text((x1 + 2, y1 + 2), txt, fill=(255, 255, 255), font=font)
                    png_path = out_dir / (f"detections_{variant}.png" if variant else "detections.png")
                    im.save(png_path)
                    # Store a *relative* path so reports remain portable when
                    # transferred from remote to local machines.
                    overlays["main"] = png_path.name
                    artifacts["overlay_main"] = png_path.name
                except Exception:
                    pass

            try:
                json_path.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")
            except Exception:
                pass

        return PostprocessResult(task="detection", json=json_obj, overlays=overlays, summary_text=summary)
