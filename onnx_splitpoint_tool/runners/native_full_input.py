from __future__ import annotations

"""Prepare and seal the exact DeepX Native-Full runtime input.

The generated Generic suite and the later Native runner both consume this
primitive.  Image decoding and preprocessing happen before any timed loop;
consumers replay only the sealed ``runtime_input.bin`` bytes.
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover - target dependency
    Image = None  # type: ignore

if (__package__ or "").split(".", 1)[0] == "splitpoint_runners":
    from splitpoint_runners.preprocessing_contract import (  # type: ignore
        observed_runtime_preprocessing_identity,
        runtime_numeric_input_identity,
    )
else:
    from onnx_splitpoint_tool.preprocessing_contract import (
        observed_runtime_preprocessing_identity,
        runtime_numeric_input_identity,
    )


SCHEMA = "onnx-splitpoint/native-full-input-dump"
SCHEMA_VERSION = 2


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8"))


def _absolute_without_resolving(path: Path) -> Path:
    candidate = path.expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _contains_symlink(path: Path) -> bool:
    candidate = _absolute_without_resolving(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _contains_symlink(path.parent) or path.is_symlink():
        raise ValueError("prepared_input_output_path_contains_symlink")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _input_contract_fields(
    input_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[int], np.dtype[Any], str, str, str, str, int]:
    inp = (
        dict(input_contract.get("input") or {})
        if isinstance(input_contract.get("input"), Mapping) else {}
    )
    shape_raw = inp.get("shape")
    if (
        not isinstance(shape_raw, (list, tuple))
        or not shape_raw
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in shape_raw
        )
    ):
        raise ValueError("prepared_input_contract_shape_invalid")
    shape = [int(value) for value in shape_raw]
    aliases = {
        "float": "float32", "fp32": "float32", "float32": "float32",
        "u8": "uint8", "uint8": "uint8",
    }
    dtype_name = aliases.get(str(inp.get("dtype") or "").strip().lower(), "")
    if not dtype_name:
        raise ValueError("prepared_input_contract_dtype_invalid")
    dtype = np.dtype(dtype_name)
    layout = str(inp.get("layout") or "").strip().upper()
    if layout not in {"HWC", "NHWC", "CHW", "NCHW"}:
        raise ValueError("prepared_input_contract_layout_invalid")
    color_space = str(inp.get("color_space") or "").strip().upper()
    if color_space != "RGB":
        raise ValueError("prepared_input_contract_color_space_invalid")
    normalization = str(inp.get("normalization") or "").strip().lower()
    preprocess_mode = str(inp.get("preprocess_mode") or "").strip().lower()
    if preprocess_mode not in {"resize", "letterbox"}:
        raise ValueError("prepared_input_contract_preprocess_mode_invalid")
    pad_value = inp.get("letterbox_pad_value", 114 if preprocess_mode == "letterbox" else 0)
    if isinstance(pad_value, bool) or not isinstance(pad_value, int) or not 0 <= pad_value <= 255:
        raise ValueError("prepared_input_contract_pad_value_invalid")
    return (
        inp, shape, dtype, layout, color_space, normalization,
        preprocess_mode, int(pad_value),
    )


def _shape_hwc(shape: Sequence[int], layout: str) -> tuple[int, int, int]:
    values = [int(value) for value in shape]
    if layout == "NCHW" and len(values) == 4 and values[0] == 1:
        return values[2], values[3], values[1]
    if layout == "NHWC" and len(values) == 4 and values[0] == 1:
        return values[1], values[2], values[3]
    if layout == "CHW" and len(values) == 3:
        return values[1], values[2], values[0]
    if layout == "HWC" and len(values) == 3:
        return values[0], values[1], values[2]
    raise ValueError("prepared_input_contract_shape_layout_mismatch")


def _letterbox_rgb(image: Any, width: int, height: int, pad: int) -> np.ndarray:
    src_w, src_h = image.size
    scale = min(float(width) / max(1, src_w), float(height) / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = np.asarray(
        image.resize((new_w, new_h), resample=bilinear), dtype=np.uint8,
    )
    canvas = np.full((height, width, 3), int(pad), dtype=np.uint8)
    left = max(0, (width - new_w) // 2)
    top = max(0, (height - new_h) // 2)
    canvas[top:top + new_h, left:left + new_w, : resized.shape[2]] = resized[..., :3]
    return np.ascontiguousarray(canvas)


def _prepare_tensor(
    image_path: Path, *, shape: Sequence[int], dtype: np.dtype[Any],
    layout: str, task: str, normalization: str, preprocess_mode: str,
    pad_value: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if Image is None:
        raise RuntimeError("Pillow is required for DeepX prepared input")
    height, width, channels = _shape_hwc(shape, layout)
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
        if preprocess_mode == "letterbox":
            hwc = _letterbox_rgb(image, width, height, pad_value)
        else:
            bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
            hwc = np.asarray(
                image.resize((width, height), resample=bilinear),
                dtype=np.uint8,
            )
    if channels == 1:
        hwc = hwc[..., :1]
    elif channels < hwc.shape[-1]:
        hwc = hwc[..., :channels]
    base = hwc.astype(np.float32)
    if dtype.kind == "f":
        base /= 255.0
        if str(task).strip().lower() == "classification" and (
            "imagenet" in normalization or not normalization
        ) and base.shape[-1] == 3:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            base = (base - mean) / std
    else:
        base = hwc
    if layout == "NCHW":
        tensor = np.transpose(base, (2, 0, 1))[None, ...]
    elif layout == "NHWC":
        tensor = base[None, ...]
    elif layout == "CHW":
        tensor = np.transpose(base, (2, 0, 1))
    else:
        tensor = base
    tensor = np.ascontiguousarray(tensor.astype(dtype, copy=False))
    if list(tensor.shape) != [int(value) for value in shape]:
        raise ValueError("prepared_input_tensor_shape_mismatch")
    preprocess = {
        "mode": (
            "letterbox_rgb_uint8" if preprocess_mode == "letterbox"
            else "resize_rgb_uint8"
        ),
        "pad_value": int(pad_value if preprocess_mode == "letterbox" else 0),
        "rgb": True,
        "ort_model_scale": (
            "imagenet" if str(task).strip().lower() == "classification"
            else "norm"
        ),
        "layout": layout,
        "runtime_dtype": str(dtype),
        "normalization": normalization,
        "color_space": "RGB",
        "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear",
        "resize_rounding": "python_round_ties_to_even",
        "placement": (
            "floor_top_left_remainder_bottom_right"
            if preprocess_mode == "letterbox" else "not_applicable"
        ),
    }
    return tensor, np.ascontiguousarray(hwc), preprocess


def prepare_and_seal_deepx_native_full_input(
    *, image_path: str | Path, input_contract: Mapping[str, Any], task: str,
    out_dir: str | Path, model: str, setup_id: str = "",
    comparison_backend: str = "",
) -> dict[str, Any]:
    """Create a schema-v2 sealed input before any timed DeepX inference."""
    image = _absolute_without_resolving(Path(image_path))
    output = _absolute_without_resolving(Path(out_dir))
    if _contains_symlink(image) or not image.is_file():
        raise ValueError("prepared_input_source_image_invalid")
    if _contains_symlink(output):
        raise ValueError("prepared_input_output_path_contains_symlink")
    image = image.resolve(strict=True)
    output.mkdir(parents=True, exist_ok=True)
    inp, shape, dtype, layout, color_space, normalization, mode, pad = (
        _input_contract_fields(input_contract)
    )
    tensor, input_hwc, preprocess = _prepare_tensor(
        image, shape=shape, dtype=dtype, layout=layout, task=task,
        normalization=normalization, preprocess_mode=mode, pad_value=pad,
    )
    input_bytes = input_hwc.tobytes()
    runtime_bytes = tensor.tobytes()
    input_file = output / "input_rgb_uint8.bin"
    runtime_file = output / "runtime_input.bin"
    manifest_file = output / "native_full_input_manifest.json"
    _atomic_write(input_file, input_bytes)
    _atomic_write(runtime_file, runtime_bytes)
    semantic_identity, semantic_sha = observed_runtime_preprocessing_identity(
        task=str(task).strip().lower(),
        target_hw=[int(input_hwc.shape[0]), int(input_hwc.shape[1])],
        preprocess_mode=preprocess["mode"], color_space=color_space,
        input_domain="uint8_0_255",
        resize_interpolation=preprocess["resize_interpolation"],
        resize_rounding=preprocess["resize_rounding"],
        placement=preprocess["placement"],
        pad_value=preprocess["pad_value"],
        image_scale=preprocess["ort_model_scale"],
    )
    numeric_identity, numeric_sha = runtime_numeric_input_identity(
        backend="native_full_deepx", task=str(task).strip().lower(),
        preprocessing_contract_sha256_value=semantic_sha,
        runtime_input_name=str(inp.get("name") or "input"),
        runtime_input_shape=shape, runtime_input_dtype=str(dtype),
        runtime_input_layout=layout, runtime_color_space=color_space,
        runtime_normalization=normalization,
    )
    payload = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
        "backend": "native_full_deepx", "model": str(model),
        "setup_id": str(setup_id),
        "comparison_backend": str(comparison_backend),
        "case": "full", "task": str(task).strip().lower(),
        "image": str(image), "image_sha256": _sha256_file(image),
        "input_image": str(image), "input_image_sha256": _sha256_file(image),
        "input_dump": input_file.name,
        "input_dump_sha256": _sha256_bytes(input_bytes),
        "input_dump_bytes": len(input_bytes),
        "input_shape_hwc": [int(value) for value in input_hwc.shape],
        "runtime_input_name": str(inp.get("name") or "input"),
        "runtime_input_shape": shape,
        "runtime_input_dtype": str(dtype),
        "runtime_input_file": runtime_file.name,
        "runtime_input_sha256": _sha256_bytes(runtime_bytes),
        "runtime_input_bytes": len(runtime_bytes),
        "preprocess": preprocess,
        "runtime_preprocess_mode": preprocess["mode"],
        "runtime_input_layout": layout,
        "runtime_color_space": color_space,
        "runtime_normalization": normalization,
        "runtime_preprocessing_identity": semantic_identity,
        "runtime_preprocessing_sha256": semantic_sha,
        "runtime_numeric_input_identity": numeric_identity,
        "runtime_numeric_input_sha256": numeric_sha,
        "contract_source": "shared_pre_timing_deepx_input_sealer",
    }
    _atomic_write(
        manifest_file,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return {
        "manifest_path": manifest_file,
        "manifest_sha256": _sha256_file(manifest_file),
        "payload": payload,
        "runtime_input": tensor,
        "input_hwc": input_hwc,
        "preprocess": preprocess,
    }


def load_sealed_deepx_native_full_input(
    manifest_path: str | Path, *, image_path: str | Path,
    input_contract: Mapping[str, Any], task: str, expected_model: str = "",
    expected_setup_id: str = "", expected_comparison_backend: str = "",
) -> dict[str, Any]:
    """Verify and load a previously sealed DeepX tensor without rebuilding it."""
    manifest = _absolute_without_resolving(Path(manifest_path))
    image = _absolute_without_resolving(Path(image_path))
    if (
        _contains_symlink(manifest) or _contains_symlink(image)
        or manifest.name != "native_full_input_manifest.json"
        or not manifest.is_file() or not image.is_file()
    ):
        raise ValueError("prepared_input_manifest_missing")
    manifest = manifest.resolve(strict=True)
    image = image.resolve(strict=True)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("prepared_input_manifest_invalid") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("prepared_input_manifest_invalid")
    payload = dict(payload)
    if (
        payload.get("schema") != SCHEMA
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("backend") != "native_full_deepx"
        or str(payload.get("task") or "").strip().lower()
        != str(task).strip().lower()
    ):
        raise ValueError("prepared_input_manifest_identity_mismatch")
    expected_identity = {
        "model": str(expected_model or "").strip(),
        "setup_id": str(expected_setup_id or "").strip(),
        "comparison_backend": str(expected_comparison_backend or "").strip(),
    }
    if any(
        expected
        and str(payload.get(field) or "").strip() != expected
        for field, expected in expected_identity.items()
    ):
        raise ValueError("prepared_input_manifest_identity_mismatch")
    inp, shape, dtype, layout, color_space, normalization, mode, pad = (
        _input_contract_fields(input_contract)
    )
    if (
        list(payload.get("runtime_input_shape") or []) != shape
        or str(payload.get("runtime_input_dtype") or "").strip().lower()
        != str(dtype).lower()
        or str(payload.get("runtime_input_layout") or "").strip().upper()
        != layout
        or str(payload.get("runtime_color_space") or "").strip().upper()
        != color_space
        or str(payload.get("runtime_normalization") or "").strip().lower()
        != normalization
        or str(payload.get("runtime_input_name") or "").strip()
        != str(inp.get("name") or "input")
    ):
        raise ValueError("prepared_input_contract_mismatch")
    manifest_mode = str(payload.get("runtime_preprocess_mode") or "").strip().lower()
    if manifest_mode.endswith("_rgb_uint8"):
        manifest_mode = manifest_mode[:-len("_rgb_uint8")]
    if manifest_mode != mode:
        raise ValueError("prepared_input_preprocessing_contract_mismatch")
    if (
        str(payload.get("input_image_sha256") or "").strip().lower()
        != _sha256_file(image)
    ):
        raise ValueError("prepared_input_source_image_sha256_mismatch")
    if (
        str(payload.get("runtime_input_file") or "") != "runtime_input.bin"
        or str(payload.get("input_dump") or "") != "input_rgb_uint8.bin"
    ):
        raise ValueError("prepared_input_tensor_metadata_invalid")
    tensor = manifest.parent / "runtime_input.bin"
    input_dump = manifest.parent / "input_rgb_uint8.bin"
    for path in (tensor, input_dump):
        if _contains_symlink(path) or not path.is_file():
            raise ValueError("prepared_input_tensor_metadata_invalid")
    runtime_bytes = tensor.read_bytes()
    expected_bytes = math.prod(shape) * dtype.itemsize
    if (
        len(runtime_bytes) != expected_bytes
        or payload.get("runtime_input_bytes") != expected_bytes
        or str(payload.get("runtime_input_sha256") or "").strip().lower()
        != _sha256_bytes(runtime_bytes)
    ):
        raise ValueError("prepared_input_tensor_sha256_mismatch")
    input_bytes = input_dump.read_bytes()
    height, width, channels = _shape_hwc(shape, layout)
    expected_input_bytes = height * width * channels
    if (
        len(input_bytes) != expected_input_bytes
        or payload.get("input_dump_bytes") != expected_input_bytes
        or str(payload.get("input_dump_sha256") or "").strip().lower()
        != _sha256_bytes(input_bytes)
    ):
        raise ValueError("prepared_input_dump_sha256_mismatch")
    semantic = payload.get("runtime_preprocessing_identity")
    numeric = payload.get("runtime_numeric_input_identity")
    semantic_sha = str(payload.get("runtime_preprocessing_sha256") or "").strip().lower()
    numeric_sha = str(payload.get("runtime_numeric_input_sha256") or "").strip().lower()
    expected_semantic, expected_semantic_sha = (
        observed_runtime_preprocessing_identity(
            task=str(task).strip().lower(),
            target_hw=[height, width],
            preprocess_mode=(
                "letterbox_rgb_uint8" if mode == "letterbox"
                else "resize_rgb_uint8"
            ),
            color_space=color_space,
            input_domain="uint8_0_255",
            resize_interpolation="bilinear",
            resize_rounding="python_round_ties_to_even",
            placement=(
                "floor_top_left_remainder_bottom_right"
                if mode == "letterbox" else "not_applicable"
            ),
            pad_value=int(pad if mode == "letterbox" else 0),
            image_scale=(
                "imagenet"
                if str(task).strip().lower() == "classification"
                else "norm"
            ),
        )
    )
    expected_numeric, expected_numeric_sha = runtime_numeric_input_identity(
        backend="native_full_deepx",
        task=str(task).strip().lower(),
        preprocessing_contract_sha256_value=expected_semantic_sha,
        runtime_input_name=str(inp.get("name") or "input"),
        runtime_input_shape=shape,
        runtime_input_dtype=str(dtype),
        runtime_input_layout=layout,
        runtime_color_space=color_space,
        runtime_normalization=normalization,
    )
    if (
        not isinstance(semantic, Mapping) or not isinstance(numeric, Mapping)
        or dict(semantic) != expected_semantic
        or semantic_sha != expected_semantic_sha
        or _canonical_sha256(semantic) != semantic_sha
        or dict(numeric) != expected_numeric
        or numeric_sha != expected_numeric_sha
        or _canonical_sha256(numeric) != numeric_sha
    ):
        raise ValueError("prepared_input_runtime_identity_invalid")
    runtime_input = np.frombuffer(runtime_bytes, dtype=dtype).reshape(shape).copy(order="C")
    input_shape = payload.get("input_shape_hwc")
    if (
        not isinstance(input_shape, list) or len(input_shape) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in input_shape)
    ):
        raise ValueError("prepared_input_dump_shape_invalid")
    input_hwc = np.frombuffer(input_bytes, dtype=np.uint8).reshape(input_shape).copy(order="C")
    return {
        "manifest_path": manifest,
        "manifest_sha256": _sha256_file(manifest),
        "payload": payload,
        "runtime_input": runtime_input,
        "input_hwc": input_hwc,
        "preprocess": dict(payload.get("preprocess") or {}),
    }
