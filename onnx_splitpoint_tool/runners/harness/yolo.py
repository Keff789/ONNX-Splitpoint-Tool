from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _decode_multiscale_head(outputs: Sequence[np.ndarray], input_hw: Tuple[int, int], conf_thresh: float) -> _Detections:
    """Decode YOLOv5/YOLOv7-style multi-scale head outputs.

    Supports common layouts:
    - [B,na,gh,gw,ch]
    - [B,ch,gh,gw] where ch = na*(5+nc)
    """

    if len(outputs) < 3:
        raise ValueError("multiscale head expects >=3 outputs")

    # Normalize outputs to [B,na,gh,gw,ch]
    norm: List[np.ndarray] = []
    for p in outputs:
        a = np.asarray(p)
        if a.ndim == 4:
            b, ch, gh, gw = a.shape
            # infer na=3
            na = 3
            if ch % na != 0:
                raise ValueError(f"Cannot reshape multiscale head with ch={ch} into na={na}")
            a = a.reshape(b, na, ch // na, gh, gw).transpose(0, 1, 3, 4, 2)  # [B,na,gh,gw,ch]
        elif a.ndim == 5:
            # Could be [B,na,gh,gw,ch] already or [B,gh,gw,na,ch]
            if a.shape[1] in (3, 4, 5) and a.shape[-1] >= 6:
                # likely [B,na,gh,gw,ch]
                pass
            elif a.shape[-2] in (3,):
                # [B,gh,gw,na,ch]
                a = a.transpose(0, 3, 1, 2, 4)
            else:
                # assume [B,na,gh,gw,ch]
                pass
        else:
            raise ValueError(f"Unsupported multiscale head output rank: {a.shape}")
        norm.append(a.astype(np.float32, copy=False))

    # Sort by grid size (largest first)
    norm.sort(key=lambda t: t.shape[2], reverse=True)

    ih, iw = input_hw

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_cls: List[np.ndarray] = []

    for p in norm:
        b, na, gh, gw, ch = p.shape
        if b != 1:
            raise ValueError("Only batch=1 supported for YOLO postprocess")

        # Typical: ch = 5 + nc
        if ch < 6:
            raise ValueError(f"Invalid multiscale head channel dim: {ch}")
        nc = ch - 5

        # Determine stride
        # For common YOLO heads: gh in {80,40,20} for input 640 -> strides {8,16,32}
        if gh > 0:
            stride_h = ih / float(gh)
        else:
            stride_h = 0.0
        stride = int(round(stride_h)) if stride_h > 0 else None
        if stride is None or stride <= 0:
            raise ValueError("Could not infer stride")

        anchors = _YOLOV5_ANCHORS_640.get(stride)
        if anchors is None:
            # Fallback: scale anchors roughly by stride
            anchors = np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32) * (stride / 8.0)

        # grid
        ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack([xs, ys], axis=-1).astype(np.float32)  # [gh,gw,2]
        grid = grid[None, ...]  # [1,gh,gw,2] broadcast over anchors

        raw = p[0]  # [na,gh,gw,ch]
        txy = _sigmoid(raw[..., 0:2])
        twh = _sigmoid(raw[..., 2:4])
        obj = _sigmoid(raw[..., 4])
        cls = _sigmoid(raw[..., 5:])  # [na,gh,gw,nc]

        # Decode centers
        xy = (txy * 2.0 - 0.5 + grid) * float(stride)
        # Decode sizes
        wh = (twh * 2.0) ** 2 * anchors[:, None, None, :]
        xywh = np.concatenate([xy, wh], axis=-1)  # [na,gh,gw,4]
        boxes = _xywh_to_xyxy(xywh).reshape(-1, 4)

        cls_id = np.argmax(cls.reshape(-1, nc), axis=1)
        cls_conf = cls.reshape(-1, nc)[np.arange(cls_id.size), cls_id]
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


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _decode_ultralytics_regcls(outputs: Sequence[np.ndarray], input_hw: Tuple[int, int], conf_thresh: float) -> _Detections:
    """Decode Ultralytics reg/cls head outputs (YOLOv8/10/11-style).

    The export layout varies; we support the common cases:
    - reg: [1, 4*reg_max, A] or [1, A, 4*reg_max]
    - cls: [1, C, A] or [1, A, C]
    where A = sum over feature maps (e.g. 8400 for 640 input)
    """

    if len(outputs) != 2:
        raise ValueError("ultralytics_regcls expects exactly 2 outputs")

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


def _detect_yolo_format(outputs: Sequence[np.ndarray]) -> str:
    if not outputs:
        return "unknown"
    if len(outputs) == 1:
        out = np.asarray(outputs[0])
        if out.shape[-1] == 6:
            return "bn6_detections"
        if out.ndim in (2, 3) and out.shape[-1] >= 6:
            return "concat"
        return "unknown"
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
        out_list = [np.asarray(v) for v in outputs.values()]
        fmt = _detect_yolo_format(out_list)

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
            det = _decode_ultralytics_regcls(out_list, input_hw=input_hw, conf_thresh=self.conf_thresh)
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

        json_obj: Dict[str, Any] = {"task": "detection", "format": fmt, "detections": det_list}
        overlays: Dict[str, str] = {}
        summary = f"n={len(det_list)}" if det_list else "n=0"

        if context and context.get("output_dir"):
            out_dir = Path(str(context.get("output_dir")))
            variant = str(context.get("variant", ""))
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # JSON
            json_path = out_dir / (f"detections_{variant}.json" if variant else "detections.json")
            try:
                json_path.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")
            except Exception:
                pass

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
                except Exception:
                    pass

        return PostprocessResult(task="detection", json=json_obj, overlays=overlays, summary_text=summary)
