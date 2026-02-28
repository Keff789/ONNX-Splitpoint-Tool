from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from .._types import SampleCfg


class YoloHarness:
    """Minimal YOLO-family harness.

    This harness focuses on:
    - building a single image input tensor (letterbox)
    - computing a simple accuracy proxy on raw outputs (max/mean abs diff)

    Advanced decoding/visualization stays outside Phase 1.
    """

    name = "yolo"

    def __init__(self, default_scale: Literal["raw", "norm"] = "norm") -> None:
        self.default_scale = default_scale

    def _resolve_scale_policy(self, sample_cfg: SampleCfg) -> tuple[Literal["raw", "norm"], list[str]]:
        warnings: list[str] = []
        if sample_cfg.input_scale == "auto":
            if not sample_cfg.allow_auto_scale:
                warnings.append(
                    "input_scale=auto requested but allow_auto_scale=False; "
                    f"falling back to '{self.default_scale}'."
                )
                return self.default_scale, warnings

            # Phase 1: Harness alone can't probe model outputs; keep robust.
            warnings.append(
                "input_scale=auto requested; Phase-1 YOLO harness does not probe outputs yet; "
                f"using fallback '{self.default_scale}'."
            )
            return self.default_scale, warnings

        return sample_cfg.input_scale, warnings

    def make_inputs(self, sample_cfg: SampleCfg) -> dict:
        scale, warns = self._resolve_scale_policy(sample_cfg)
        # Attach warnings to sample_cfg.extra so callers can collect them.
        if warns:
            sample_cfg.extra.setdefault("warnings", []).extend(warns)

        img_path = sample_cfg.image_path
        if img_path is None:
            raise ValueError("SampleCfg.image_path is required for YOLO harness")

        input_name = sample_cfg.input_name or "images"
        shape = tuple(sample_cfg.input_shape or (1, 3, 640, 640))
        if len(shape) != 4:
            raise ValueError(f"Expected NCHW shape, got {shape}")
        n, c, h, w = shape
        if n != 1 or c != 3:
            raise ValueError(f"Only batch=1, channels=3 supported in Phase-1 harness; got {shape}")

        img = _load_image_rgb(img_path)
        img_lb = _letterbox(img, (h, w))

        # HWC uint8 -> CHW float32
        arr = img_lb.astype(np.float32)
        if scale == "norm":
            arr = arr / 255.0
        # else raw keeps 0..255 in float32.

        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        arr = np.ascontiguousarray(arr)

        return {input_name: arr}

    def postprocess(self, outputs: Any, context: dict) -> dict:
        # Phase 1: no-op
        return {}

    def accuracy_proxy(self, ref: Any, out: Any) -> dict:
        # Compare raw tensors.
        ref_map = _to_tensor_map(ref)
        out_map = _to_tensor_map(out)

        common = [k for k in ref_map.keys() if k in out_map]
        if not common:
            return {
                "compared_outputs": 0,
                "max_abs": None,
                "mean_abs": None,
                "note": "no common output keys to compare",
            }

        max_abs = 0.0
        mean_abs_acc = 0.0
        count = 0
        for k in common:
            a = ref_map[k]
            b = out_map[k]
            diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
            max_abs = max(max_abs, float(np.max(diff)))
            mean_abs_acc += float(np.mean(diff))
            count += 1

        return {
            "compared_outputs": count,
            "max_abs": max_abs,
            "mean_abs": mean_abs_acc / max(1, count),
        }


def _to_tensor_map(outputs: Any) -> dict[str, np.ndarray]:
    if isinstance(outputs, dict):
        return {str(k): np.asarray(v) for k, v in outputs.items()}
    if isinstance(outputs, (list, tuple)):
        # Fall back to positional keys.
        return {str(i): np.asarray(v) for i, v in enumerate(outputs)}
    raise TypeError(f"Unsupported outputs type: {type(outputs).__name__}")


def _load_image_rgb(path: Path) -> np.ndarray:
    # Lazy import to keep import-time deps minimal.
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PIL/Pillow is required for YOLO harness image loading") from e

    img = Image.open(path).convert("RGB")
    return np.array(img)


def _letterbox(img: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
    """Letterbox resize to `new_shape` (H,W) with padding."""

    h0, w0 = img.shape[:2]
    h, w = new_shape

    # Scale ratio (new / old)
    r = min(h / h0, w / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))  # (w, h)

    # Resize
    if (w0, h0) != new_unpad:
        img_resized = _resize(img, new_unpad)
    else:
        img_resized = img

    # Compute padding
    dw = w - img_resized.shape[1]
    dh = h - img_resized.shape[0]

    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    # Padding color: 114 like common YOLO implementations
    img_padded = np.pad(
        img_resized,
        ((top, bottom), (left, right), (0, 0)),
        mode="constant",
        constant_values=114,
    )

    return img_padded


def _resize(img: np.ndarray, new_wh: tuple[int, int]) -> np.ndarray:
    """Resize using PIL (bilinear). new_wh=(w,h)."""
    try:
        from PIL import Image
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PIL/Pillow is required for YOLO harness resizing") from e

    w, h = new_wh
    im = Image.fromarray(img)
    im = im.resize((w, h), resample=Image.BILINEAR)
    return np.array(im)
