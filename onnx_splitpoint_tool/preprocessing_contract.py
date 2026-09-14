"""Canonical image-preprocessing contracts shared by quality and accelerators.

The contract intentionally describes the prepared RGB ``uint8`` image, before
backend-specific layout conversion and numeric normalization.  This keeps the
semantic identity stable across CPU, TensorRT, DeepX and Hailo while runtime
tensor hashes can still attest the exact bytes bound to a device.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


PREPROCESSING_CONTRACT_SCHEMA = "onnx-splitpoint/image-preprocessing-contract"
PREPROCESSING_CONTRACT_SCHEMA_VERSION = 2
RUNTIME_NUMERIC_INPUT_SCHEMA = (
    "onnx-splitpoint/runtime-numeric-input-identity"
)
RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION = 1


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_image_task(task: Any) -> str:
    value = str(task or "").strip().lower().replace("-", "_")
    if value in {"detect", "detection", "object_detection", "instance_segmentation", "segmentation", "pose", "obb"}:
        return "detection"
    if value in {"classify", "classification", "image_classification", "classifier"}:
        return "classification"
    raise ValueError(
        "Image preprocessing requires an explicit task: detection or classification; "
        f"got {task!r}"
    )


def infer_image_task_from_hint(*hints: Any) -> str:
    text = " ".join(str(value or "").lower() for value in hints)
    detector_markers = (
        "yolo", "detr", "detect", "segment", "pose", "obb", "scrfd",
        "retina", "ssd", "fasterrcnn", "maskrcnn",
    )
    classifier_markers = (
        "resnet", "mobilenet", "regnet", "efficientnet", "convnext",
        "densenet", "inception", "vgg", "classifier", "classification",
        "vit", "swin",
    )
    if any(marker in text for marker in detector_markers):
        return "detection"
    if any(marker in text for marker in classifier_markers):
        return "classification"
    raise ValueError(
        "Unable to infer image task safely. Pass task='detection' or "
        "task='classification' so preprocessing cannot silently drift."
    )


def target_hw_from_shape(shape: Sequence[Any]) -> tuple[int, int]:
    dims: list[int | None] = []
    for raw in list(shape or []):
        try:
            value = int(raw)
        except Exception:
            value = 0
        dims.append(value if value > 0 else None)
    if len(dims) == 4:
        if dims[-1] in {1, 3, 4}:  # NHWC
            h, w = dims[1], dims[2]
        elif dims[1] in {1, 3, 4}:  # NCHW
            h, w = dims[2], dims[3]
        else:
            raise ValueError(f"Cannot identify image layout from shape {list(shape)!r}")
    elif len(dims) == 3:
        if dims[-1] in {1, 3, 4}:  # HWC
            h, w = dims[0], dims[1]
        elif dims[0] in {1, 3, 4}:  # CHW
            h, w = dims[1], dims[2]
        else:
            raise ValueError(f"Cannot identify image layout from shape {list(shape)!r}")
    else:
        raise ValueError(f"Expected a rank-3/4 image shape, got {list(shape)!r}")
    if h is None or w is None:
        raise ValueError(f"Image target height/width must be static, got {list(shape)!r}")
    return int(h), int(w)


def canonical_image_preprocessing_contract(
    task: Any, target_hw: Sequence[Any]
) -> dict[str, Any]:
    image_task = normalize_image_task(task)
    if len(list(target_hw or [])) != 2:
        raise ValueError(f"target_hw must be [height, width], got {target_hw!r}")
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_hw must be positive, got {target_hw!r}")
    detection = image_task == "detection"
    return {
        "schema": PREPROCESSING_CONTRACT_SCHEMA,
        "schema_version": PREPROCESSING_CONTRACT_SCHEMA_VERSION,
        "contract_scope": "prepared_rgb_uint8_semantics",
        "task": image_task,
        "preprocess_mode": "letterbox" if detection else "resize",
        "spatial_transform": "centered_letterbox" if detection else "direct_resize",
        "target_hw": [target_h, target_w],
        "color_space": "RGB",
        "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear",
        "resize_rounding": "python_round_ties_to_even",
        "placement": (
            "floor_top_left_remainder_bottom_right" if detection else "not_applicable"
        ),
        "pad_value": 114 if detection else 0,
        "letterbox_pad_value": 114 if detection else 0,
        # Compatibility aliases retained in the sealed identity.  They also
        # make the post-RGB numeric transform explicit for calibration.
        "image_scale": "norm" if detection else "imagenet",
        "letterbox": bool(detection),
    }


def preprocessing_contract_sha256(contract: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(contract))


def resolve_image_preprocessing_contract(
    *,
    task: Any,
    target_hw: Sequence[Any],
    declared: Mapping[str, Any] | str | None = None,
) -> tuple[dict[str, Any], str]:
    expected = canonical_image_preprocessing_contract(task, target_hw)
    if declared is not None:
        candidate: Any = declared
        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                candidate = None
            else:
                candidate = json.loads(text)
        if candidate is not None:
            if not isinstance(candidate, Mapping):
                raise ValueError("Declared image preprocessing contract must be a JSON object")
            actual = dict(candidate)
            if actual != expected:
                differing = sorted(
                    key
                    for key in set(actual) | set(expected)
                    if actual.get(key) != expected.get(key)
                )
                raise ValueError(
                    "Image preprocessing contract mismatch; inference/compilation is "
                    f"refused. differing_fields={differing}; expected={expected}; actual={actual}"
                )
    return expected, preprocessing_contract_sha256(expected)


def observed_runtime_preprocessing_identity(
    *,
    task: Any,
    target_hw: Sequence[Any],
    preprocess_mode: Any,
    color_space: Any,
    input_domain: Any,
    resize_interpolation: Any,
    resize_rounding: Any,
    placement: Any,
    pad_value: Any,
    image_scale: Any,
) -> tuple[dict[str, Any], str]:
    """Describe the semantic image transform actually used by a producer.

    The returned mapping deliberately has exactly the same schema and keys as
    :func:`canonical_image_preprocessing_contract`.  A conforming runtime
    therefore produces the same SHA as Central Quality, while a BGR input,
    different target size, resize algorithm, placement or pad changes the
    identity instead of being hidden behind a coarse ``letterbox`` label.
    """

    identity = canonical_image_preprocessing_contract(task, target_hw)
    mode_token = str(preprocess_mode or "").strip().lower()
    if mode_token.endswith("_rgb_uint8"):
        mode_token = mode_token[: -len("_rgb_uint8")]
    letterbox = mode_token == "letterbox"
    try:
        observed_pad = int(pad_value)
    except (TypeError, ValueError, OverflowError):
        observed_pad = -1
    identity.update({
        "preprocess_mode": mode_token,
        "spatial_transform": (
            "centered_letterbox" if letterbox
            else "direct_resize" if mode_token == "resize"
            else str(mode_token or "unavailable")
        ),
        "color_space": str(color_space or "").strip().upper(),
        "input_domain": str(input_domain or "").strip().lower(),
        "resize_interpolation": str(
            resize_interpolation or ""
        ).strip().lower(),
        "resize_rounding": str(resize_rounding or "").strip().lower(),
        "placement": str(placement or "").strip().lower(),
        "pad_value": observed_pad,
        "letterbox_pad_value": observed_pad if letterbox else 0,
        "image_scale": str(image_scale or "").strip().lower(),
        "letterbox": letterbox,
    })
    return identity, preprocessing_contract_sha256(identity)


def canonical_runtime_dtype(value: Any) -> str:
    token = str(value or "").strip().lower()
    return {
        "u8": "uint8",
        "fp16": "float16",
        "half": "float16",
        "float": "float32",
        "fp32": "float32",
    }.get(token, token)


def runtime_numeric_input_identity(
    *,
    backend: Any,
    task: Any,
    preprocessing_contract_sha256_value: Any,
    runtime_input_name: Any,
    runtime_input_shape: Sequence[Any],
    runtime_input_dtype: Any,
    runtime_input_layout: Any,
    runtime_color_space: Any,
    runtime_normalization: Any,
) -> tuple[dict[str, Any], str]:
    """Seal the backend-specific numeric transform separately from RGB pixels."""

    shape = [int(value) for value in list(runtime_input_shape or [])]
    dtype = canonical_runtime_dtype(runtime_input_dtype)
    normalization = str(runtime_normalization or "").strip().lower()
    if dtype == "uint8":
        numeric_domain = "uint8_0_255"
    elif normalization in {
        "imagenet_mean_std", "mean_std",
    }:
        numeric_domain = "float_imagenet_mean_std"
    elif normalization in {
        "scale_0_1", "divide_255", "divide_255_float32",
        "model_preprocessing_float32",
    }:
        numeric_domain = "float_0_1"
    elif normalization == "embedded_dxcom_preprocessing":
        numeric_domain = "backend_embedded_preprocessing"
    else:
        numeric_domain = "unclassified"
    identity = {
        "schema": RUNTIME_NUMERIC_INPUT_SCHEMA,
        "schema_version": RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION,
        "backend": str(backend or "").strip().lower(),
        "task": normalize_image_task(task),
        "preprocessing_contract_sha256": str(
            preprocessing_contract_sha256_value or ""
        ).strip().lower(),
        "runtime_input_name": str(runtime_input_name or "").strip(),
        "runtime_input_shape": shape,
        "runtime_input_dtype": dtype,
        "runtime_input_layout": str(
            runtime_input_layout or ""
        ).strip().upper(),
        "runtime_color_space": str(
            runtime_color_space or ""
        ).strip().upper(),
        "runtime_normalization": normalization,
        "runtime_numeric_domain": numeric_domain,
    }
    return identity, canonical_json_sha256(identity)


def runtime_numeric_input_identity_errors(
    identity: Mapping[str, Any],
    preprocessing_identity: Mapping[str, Any],
) -> list[str]:
    """Return fail-closed inconsistencies in a runtime numeric attestation."""

    errors: list[str] = []
    dtype = canonical_runtime_dtype(identity.get("runtime_input_dtype"))
    normalization = str(
        identity.get("runtime_normalization") or ""
    ).strip().lower()
    task = normalize_image_task(preprocessing_identity.get("task"))
    if identity.get("schema") != RUNTIME_NUMERIC_INPUT_SCHEMA:
        errors.append("runtime_numeric_input_schema_mismatch")
    try:
        schema_version = int(identity.get("schema_version") or 0)
    except (TypeError, ValueError, OverflowError):
        schema_version = -1
    if schema_version != RUNTIME_NUMERIC_INPUT_SCHEMA_VERSION:
        errors.append("runtime_numeric_input_schema_version_mismatch")
    preprocessing_sha = str(
        identity.get("preprocessing_contract_sha256") or ""
    ).strip().lower()
    if (
        preprocessing_sha
        != preprocessing_contract_sha256(preprocessing_identity)
    ):
        errors.append("runtime_numeric_input_preprocessing_sha256_mismatch")
    if not str(identity.get("backend") or "").strip():
        errors.append("runtime_numeric_input_backend_missing")
    if not str(identity.get("runtime_input_name") or "").strip():
        errors.append("runtime_numeric_input_name_missing")
    if str(identity.get("task") or "").strip().lower() != task:
        errors.append("runtime_numeric_input_task_mismatch")
    if str(identity.get("runtime_color_space") or "").strip().upper() != "RGB":
        errors.append("runtime_numeric_input_color_space_mismatch")
    shape = list(identity.get("runtime_input_shape") or [])
    try:
        if (
            not shape
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape
            )
        ):
            raise ValueError("runtime image shape must be positive integers")
        observed_hw = list(target_hw_from_shape(shape))
    except (TypeError, ValueError):
        observed_hw = []
        errors.append("runtime_numeric_input_shape_invalid")
    if observed_hw != list(preprocessing_identity.get("target_hw") or []):
        errors.append("runtime_numeric_input_target_hw_mismatch")
    expected_layout = ""
    if len(shape) == 4:
        if shape[-1] in {1, 3, 4}:
            expected_layout = "NHWC"
        elif shape[1] in {1, 3, 4}:
            expected_layout = "NCHW"
    elif len(shape) == 3:
        if shape[-1] in {1, 3, 4}:
            expected_layout = "HWC"
        elif shape[0] in {1, 3, 4}:
            expected_layout = "CHW"
    if (
        not expected_layout
        or str(identity.get("runtime_input_layout") or "").strip().upper()
        != expected_layout
    ):
        errors.append("runtime_numeric_input_layout_mismatch")
    allowed_uint8 = {
        "none", "raw", "none_uint8", "identity_uint8_0_255",
        "embedded_hailo_quantization", "embedded_dxcom_preprocessing",
    }
    allowed_float = (
        {"imagenet_mean_std", "mean_std"}
        if task == "classification"
        else {
            "scale_0_1", "divide_255", "divide_255_float32",
            "model_preprocessing_float32",
            "embedded_dxcom_preprocessing",
        }
    )
    if dtype == "uint8":
        if normalization not in allowed_uint8:
            errors.append("runtime_numeric_input_normalization_invalid")
    elif dtype in {"float16", "float32"}:
        if normalization not in allowed_float:
            errors.append("runtime_numeric_input_normalization_invalid")
    else:
        errors.append("runtime_numeric_input_dtype_invalid")
    expected_domain = (
        "uint8_0_255"
        if dtype == "uint8"
        else "float_imagenet_mean_std"
        if normalization in {"imagenet_mean_std", "mean_std"}
        else "float_0_1"
        if normalization in {
            "scale_0_1", "divide_255", "divide_255_float32",
            "model_preprocessing_float32",
        }
        else "backend_embedded_preprocessing"
        if normalization == "embedded_dxcom_preprocessing"
        else "unclassified"
    )
    if str(
        identity.get("runtime_numeric_domain") or ""
    ).strip().lower() != expected_domain:
        errors.append("runtime_numeric_input_domain_mismatch")
    return list(dict.fromkeys(errors))


def _coerce_rgb_uint8(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 4 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 3 and value.shape[0] in {1, 3, 4} and value.shape[-1] not in {1, 3, 4}:
        value = np.transpose(value, (1, 2, 0))
    if value.ndim == 2:
        value = value[..., None]
    if value.ndim != 3 or value.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Expected an image-like array, got shape={tuple(value.shape)}")
    if value.shape[-1] == 4:
        value = value[..., :3]
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    if np.issubdtype(value.dtype, np.floating):
        finite_max = float(np.nanmax(value)) if value.size else 0.0
        if finite_max <= 1.5:
            value = value * 255.0
    value = np.clip(value, 0, 255).astype(np.uint8, copy=False)
    return np.ascontiguousarray(value)


def prepare_rgb_uint8_image(
    image: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the exact semantic transform and return per-image geometry."""

    task = contract.get("task")
    target_hw = contract.get("target_hw")
    expected, _ = resolve_image_preprocessing_contract(
        task=task, target_hw=target_hw, declared=contract
    )
    source = _coerce_rgb_uint8(image)
    src_h, src_w = int(source.shape[0]), int(source.shape[1])
    target_h, target_w = map(int, expected["target_hw"])
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency failure
        raise RuntimeError(f"Pillow is required for canonical preprocessing: {exc}") from exc
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")

    if expected["preprocess_mode"] == "resize":
        resized = np.asarray(
            Image.fromarray(source, mode="RGB").resize(
                (target_w, target_h), resample=bilinear
            )
        )
        geometry = {
            "mode": "resize",
            "source_hw": [src_h, src_w],
            "target_hw": [target_h, target_w],
            "resized_hw": [target_h, target_w],
            "scale_xy": [target_w / float(src_w), target_h / float(src_h)],
            "pad_ltrb": [0, 0, 0, 0],
        }
        return np.ascontiguousarray(resized), geometry

    scale = min(target_w / float(src_w), target_h / float(src_h))
    resized_w = max(1, min(target_w, int(round(src_w * scale))))
    resized_h = max(1, min(target_h, int(round(src_h * scale))))
    resized = np.asarray(
        Image.fromarray(source, mode="RGB").resize(
            (resized_w, resized_h), resample=bilinear
        )
    )
    pad_w = target_w - resized_w
    pad_h = target_h - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    prepared = np.full(
        (target_h, target_w, 3), int(expected["pad_value"]), dtype=np.uint8
    )
    prepared[top : top + resized_h, left : left + resized_w] = resized
    geometry = {
        "mode": "letterbox",
        "source_hw": [src_h, src_w],
        "target_hw": [target_h, target_w],
        "resized_hw": [resized_h, resized_w],
        "scale_xy": [resized_w / float(src_w), resized_h / float(src_h)],
        "pad_ltrb": [left, top, right, bottom],
        "pad_value": int(expected["pad_value"]),
    }
    return np.ascontiguousarray(prepared), geometry
