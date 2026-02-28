from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image

from .base import Harness, PostprocessResult, PostprocessResultLike


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class _Layout:
    """Represents how a model expects image tensors."""

    # Either "NCHW" or "NHWC".
    name: str
    # Spatial size expected by the model (H, W).
    hw: tuple[int, int]


def _infer_layout(input_shape: Sequence[int] | None) -> _Layout:
    """Infer tensor layout (NCHW vs NHWC) and spatial size from a model input shape.

    The tool primarily targets vision models with 4D inputs.
    If the shape is unknown or contains dynamic dims, fall back to ImageNet default.
    """

    default = _Layout(name="NCHW", hw=(224, 224))
    if not input_shape or len(input_shape) != 4:
        return default

    shape = [int(x) for x in input_shape]

    def _valid_dim(v: int) -> bool:
        return v > 0

    # NCHW: (N, C, H, W)
    if shape[1] == 3:
        h, w = shape[2], shape[3]
        if _valid_dim(h) and _valid_dim(w):
            return _Layout(name="NCHW", hw=(h, w))
        return default

    # NHWC: (N, H, W, C)
    if shape[3] == 3:
        h, w = shape[1], shape[2]
        if _valid_dim(h) and _valid_dim(w):
            return _Layout(name="NHWC", hw=(h, w))
        return _Layout(name="NHWC", hw=default.hw)

    # Unknown channel position. Prefer NCHW, keep default crop size.
    return default


def _load_image(source: str | Path | Image.Image | np.ndarray) -> Image.Image:
    """Load an image from a path, PIL image, or numpy array."""

    if isinstance(source, Image.Image):
        return source

    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim not in (2, 3):
            raise ValueError(f"Expected image ndarray with 2 or 3 dims, got shape={arr.shape}.")
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return Image.fromarray(arr.astype(np.uint8))

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    return Image.open(path)


def _resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    """Resize so that the shorter side == short_side, preserving aspect ratio."""

    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image size: {img.size}")

    if min(w, h) == short_side:
        return img

    if w < h:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))
    else:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))

    return img.resize((new_w, new_h), resample=Image.BILINEAR)


def _center_crop(img: Image.Image, crop_hw: tuple[int, int]) -> Image.Image:
    """Center crop without any padding/letterbox."""

    crop_h, crop_w = crop_hw
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"Invalid crop size: {crop_hw}")

    w, h = img.size
    if w < crop_w or h < crop_h:
        raise ValueError(
            f"Cannot center-crop {crop_hw} from resized image size {(h, w)}. "
            "Resize stage produced an image that is too small."
        )

    left = int(math.floor((w - crop_w) / 2))
    top = int(math.floor((h - crop_h) / 2))
    right = left + crop_w
    bottom = top + crop_h
    return img.crop((left, top, right, bottom))


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for 1D logits."""

    x = x.astype(np.float64, copy=False)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    if not np.isfinite(s) or s == 0.0:
        return np.full_like(ex, 1.0 / ex.size, dtype=np.float64)
    return ex / s


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors."""

    a64 = a.astype(np.float64, copy=False).ravel()
    b64 = b.astype(np.float64, copy=False).ravel()

    if not np.all(np.isfinite(a64)) or not np.all(np.isfinite(b64)):
        return float("nan")

    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na < eps and nb < eps:
        return 1.0
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


class ClassificationHarness(Harness):
    """Harness for ImageNet-style classification models."""

    name = "classification"

    def __init__(
        self,
        *,
        resize_short_side: int = 256,
        topk: int = 5,
        eps: float = 1e-4,
        cosine_threshold: float = 0.9999,
        labels: Sequence[str] | None = None,
    ) -> None:
        self._resize_short_side = int(resize_short_side)
        self._topk = int(topk)
        self._eps = float(eps)
        self._cosine_threshold = float(cosine_threshold)
        self._labels_override = list(labels) if labels is not None else None

    def make_inputs(self, sample_cfg: Any) -> dict[str, np.ndarray]:
        """Create model inputs using standard ImageNet preprocessing."""

        input_name = getattr(sample_cfg, "input_name", None)
        input_shape = getattr(sample_cfg, "input_shape", None)
        # Newer SampleCfg uses `image_path`, older suites used `sample_path`.
        sample_path = getattr(sample_cfg, "image_path", None)
        if sample_path is None:
            sample_path = getattr(sample_cfg, "sample_path", None)

        if not input_name or not isinstance(input_name, str):
            raise ValueError(
                "ClassificationHarness.make_inputs requires sample_cfg.input_name (str). "
                f"Got: {input_name!r}"
            )

        layout = _infer_layout(input_shape)
        crop_h, crop_w = layout.hw

        # Ensure the resize stage produces an image large enough for the crop.
        # Default ImageNet behavior: resize short side to 256 for 224 crops.
        # Generalize for other crop sizes.
        try:
            auto_resize_short = int(round(min(crop_h, crop_w) / 224.0 * 256.0))
        except Exception:
            auto_resize_short = self._resize_short_side
        resize_short = max(self._resize_short_side, auto_resize_short, crop_h, crop_w)

        if sample_path is None:
            raise ValueError(
                "ClassificationHarness.make_inputs requires a sample image path via "
                "sample_cfg.image_path (or legacy sample_cfg.sample_path)."
            )

        img = _load_image(sample_path)
        img = img.convert("RGB")
        img = _resize_short_side(img, resize_short)
        img = _center_crop(img, (crop_h, crop_w))

        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC, [0,1]
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected RGB image after conversion, got array shape={arr.shape}.")

        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD

        if layout.name == "NHWC":
            tensor = arr[None, :, :, :].astype(np.float32, copy=False)
        else:
            # NCHW
            tensor = np.transpose(arr, (2, 0, 1))[None, :, :, :].astype(np.float32, copy=False)

        return {input_name: tensor}

    def accuracy_proxy(self, ref: Sequence[np.ndarray], out: Sequence[np.ndarray]) -> dict[str, Any]:
        """Compare raw output tensors (logits) between full and composed models."""

        n = min(len(ref), len(out))
        if n == 0:
            raise ValueError("No outputs to compare: ref/out are empty.")

        per_out: list[dict[str, Any]] = []
        max_abs_all = 0.0
        mean_abs_all: list[float] = []
        cosine_all: list[float] = []

        for i in range(n):
            a = np.asarray(ref[i])
            b = np.asarray(out[i])
            diff = a.astype(np.float64, copy=False) - b.astype(np.float64, copy=False)
            absdiff = np.abs(diff)
            max_abs = float(np.max(absdiff))
            mean_abs = float(np.mean(absdiff))
            cos = _cosine_similarity(a, b)

            per_out.append(
                {
                    "index": i,
                    "shape_ref": list(a.shape),
                    "shape_out": list(b.shape),
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "cosine_similarity": cos,
                    "elementwise_pass": bool(max_abs <= self._eps),
                    "cosine_pass": bool(np.isfinite(cos) and cos >= self._cosine_threshold),
                }
            )

            max_abs_all = max(max_abs_all, max_abs)
            mean_abs_all.append(mean_abs)
            cosine_all.append(cos)

        elementwise_pass = bool(max_abs_all <= self._eps)
        cosine_pass = bool(
            all(np.isfinite(c) for c in cosine_all) and min(cosine_all) >= self._cosine_threshold
        )
        final_pass = elementwise_pass or cosine_pass

        return {
            "compared_outputs": n,
            "eps": self._eps,
            "cosine_threshold": self._cosine_threshold,
            "max_abs": float(max_abs_all),
            "mean_abs": float(np.mean(mean_abs_all)),
            "cosine_similarity": float(np.min(cosine_all)),
            "elementwise_pass": elementwise_pass,
            "cosine_pass": cosine_pass,
            "final_pass": final_pass,
            "outputs": per_out,
        }

    def postprocess(
        self, outputs: Dict[str, np.ndarray], context: dict[str, Any] | None = None
    ) -> PostprocessResultLike:
        """Turn logits into a stable top-k prediction dictionary.

        If an ``output_dir`` is provided via context, this will also emit:
        - classification_<variant>.json
        - classification_<variant>.png (best-effort)
        """

        if not outputs:
            raise ValueError("ClassificationHarness.postprocess: outputs is empty")

        first_key = next(iter(outputs.keys()))
        logits = np.asarray(outputs[first_key])
        logits = np.squeeze(logits)
        if logits.ndim != 1:
            raise ValueError(f"Expected logits as 1D tensor after squeeze, got shape={logits.shape}")

        probs = _softmax(logits)
        k = max(1, min(self._topk, probs.size))
        topk_idx = np.argsort(-probs)[:k]

        labels = self._labels_override
        if labels is None:
            labels = self._load_imagenet_labels()

        out: list[dict[str, Any]] = []
        for idx in topk_idx:
            idx_int = int(idx)
            label = str(idx_int)
            if labels is not None and 0 <= idx_int < len(labels):
                label = labels[idx_int]
            out.append({"id": idx_int, "label": label, "p": float(probs[idx_int])})

        json_obj: dict[str, Any] = {"task": "classification", "topk": out}
        overlays: dict[str, str] = {}
        summary = ""
        if out:
            summary = f"top1={out[0]['label']} (p={out[0]['p']:.3f})"

        if context and context.get("output_dir"):
            out_dir = Path(str(context.get("output_dir")))
            variant = str(context.get("variant", ""))
            image_path = context.get("image_path")

            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            # JSON
            json_path = out_dir / (f"classification_{variant}.json" if variant else "classification.json")
            try:
                json_path.write_text(json.dumps(json_obj, indent=2), encoding="utf-8")
            except Exception:
                pass

            # Overlay PNG (best-effort)
            if image_path:
                try:
                    from PIL import Image, ImageDraw, ImageFont

                    im = Image.open(str(image_path)).convert("RGB")
                    draw = ImageDraw.Draw(im)
                    font = ImageFont.load_default()
                    lines = ["Top-k:"] + [
                        f"{i+1}. {t['label']} ({t['p']:.3f})" for i, t in enumerate(out[:5])
                    ]
                    y = 5
                    for line in lines:
                        draw.text((5, y), line, fill=(255, 255, 255), font=font)
                        y += 12
                    png_path = out_dir / (f"classification_{variant}.png" if variant else "classification.png")
                    im.save(png_path)
                    # Store a *relative* path so reports remain portable when
                    # transferred from remote to local machines.
                    overlays["main"] = png_path.name
                except Exception:
                    pass

        return PostprocessResult(task="classification", json=json_obj, overlays=overlays, summary_text=summary)

    def _load_imagenet_labels(self) -> list[str] | None:
        """Load ImageNet label mapping.

        Expected format: JSON list of strings where index == class id.

        We *prefer* a file next to this harness module so it also works inside
        benchmark suites (which ship only ``splitpoint_runners``).
        """

        # 1) Canonical, versioned assets file next to the runner library.
        # This path also exists inside benchmark suites (which ship
        # ``splitpoint_runners``).
        try:
            runners_root = Path(__file__).resolve().parents[1]
            for p in [
                runners_root / "assets" / "imagenet_labels.txt",
                runners_root / "assets" / "imagenet_labels.json",
            ]:
                if not p.is_file():
                    continue

                if p.suffix.lower() == ".txt":
                    labels = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
                    labels = [ln for ln in labels if ln]
                    if labels:
                        return labels

                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return list(data)
        except Exception:
            pass

        # 2) Legacy: local file next to this module
        try:
            local_path = Path(__file__).with_name("imagenet_labels.json")
            if local_path.is_file():
                data = json.loads(local_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    return list(data)
        except Exception:
            pass

        # 3) Packaged tool resources (GUI/install)
        try:
            root = importlib_resources.files("onnx_splitpoint_tool")
            # Prefer new asset path, fall back to older resources/ location.
            for rel in [
                ("runners", "assets", "imagenet_labels.json"),
                ("resources", "imagenet_labels.json"),
            ]:
                path = root.joinpath(*rel)
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list) and all(isinstance(x, str) for x in data):
                        return list(data)
                except FileNotFoundError:
                    continue
        except Exception:
            return None

        return None
