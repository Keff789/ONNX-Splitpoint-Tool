#!/usr/bin/env python3
"""Generate one exact-input semantic output dump for a Native Full baseline.

The script intentionally performs only a single untimed inference.  Performance is
measured by ``native_full_baseline_eval_runner.py``; this companion creates the
standard output/input manifests consumed by the normal Native self-reference
validator.  Supported here are TensorRT Full and DEEPX Full.  Hailo Full dumps are
created directly by ``smoke_hailo10_hef_runner.py`` so the Hailo runtime Python is
kept intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover - target dependency
    Image = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDecodedNmsPostprocessor,
    FrozenDetectionPostprocessor,
    build_frozen_decoded_nms_normalization_contract,
    build_frozen_postprocess_contract,
    build_letterbox_geometry_contract,
    verify_frozen_decoded_nms_normalization_contract,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    observed_runtime_preprocessing_identity,
    runtime_numeric_input_identity,
)
from onnx_splitpoint_tool.runners.native_full_input import (
    load_sealed_deepx_native_full_input,
)
from onnx_splitpoint_tool.runners.harness.yolo import (
    YOLOV7_PAPER_MODEL_ID,
    YOLOV7_PAPER_ONNX_SHA256,
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bound_model_sha256(
    *sources: Mapping[str, Any], model_id: str = "",
) -> str:
    """Resolve an optional transport hash against the decoder registry.

    Backend contracts may omit the source-model digest when the exact model is
    already registered centrally.  Any digest that *is* transported remains a
    strict assertion: malformed, mutually conflicting, or registry-conflicting
    values fail closed.
    """
    values: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in (
            "model_sha256", "full_model_sha256", "terminal_model_sha256",
            "source_onnx_sha256",
        ):
            raw = str(source.get(key) or "").strip().lower()
            if not raw:
                continue
            value = raw.removeprefix("sha256:")
            if (
                len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise RuntimeError("raw_head_model_sha256_invalid")
            values.add(value)
    if len(values) > 1:
        raise RuntimeError("raw_head_model_sha256_conflicting")

    registered = (
        YOLOV7_PAPER_ONNX_SHA256
        if str(model_id or "").strip() == YOLOV7_PAPER_MODEL_ID
        else ""
    )
    if registered:
        if values and next(iter(values)) != registered:
            raise RuntimeError("raw_head_model_sha256_registry_mismatch")
        return registered
    if not values:
        raise RuntimeError("raw_head_model_sha256_missing_or_conflicting")
    return next(iter(values))


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _task(benchmark_set: Path, model: str = "") -> str:
    for name in ("benchmark_set.json", "benchmark_plan.json"):
        payload = _load_json(benchmark_set / name)
        if not isinstance(payload, Mapping):
            continue
        value = str(payload.get("benchmark_task") or payload.get("model_task") or payload.get("task") or "").strip().lower()
        if value:
            return value
    low = str(model or benchmark_set.parent.name).lower()
    return "detection" if any(token in low for token in ("yolo", "detect", "coco")) else "classification"


def _model_id(benchmark_set: Path, requested: str = "") -> str:
    text = str(requested or "").strip()
    if text:
        return text
    payload = _load_json(benchmark_set / "benchmark_set.json")
    if isinstance(payload, Mapping):
        value = str(payload.get("model_id") or payload.get("model_name") or "").strip()
        if value:
            return value
    return str(benchmark_set.parent.name)


def _resolve_image(benchmark_set: Path, image: str) -> Path:
    text = str(image or "").strip()
    if text:
        p = Path(text).expanduser()
        if p.is_file():
            return p.resolve()
        for candidate in (benchmark_set / text, benchmark_set / "resources" / "validation" / text):
            if candidate.is_file():
                return candidate.resolve()
        basename = p.name
        if basename:
            hits = sorted((benchmark_set / "resources" / "validation").rglob(basename)) if (benchmark_set / "resources" / "validation").is_dir() else []
            if hits:
                return hits[0].resolve()
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jpeg"}
    root = benchmark_set / "resources" / "validation"
    if root.is_dir():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in image_exts:
                return p.resolve()
    raise FileNotFoundError(f"No semantic validation image found under {benchmark_set}; requested={image!r}")


def _letterbox_rgb(image: Any, width: int, height: int, pad: int = 114) -> np.ndarray:
    src_w, src_h = image.size
    scale = min(float(width) / max(1, src_w), float(height) / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = np.asarray(
        image.resize((new_w, new_h), resample=bilinear),
        dtype=np.uint8,
    )
    canvas = np.full((height, width, 3), int(pad), dtype=np.uint8)
    x = max(0, (width - new_w) // 2)
    y = max(0, (height - new_h) // 2)
    canvas[y:y + new_h, x:x + new_w, : resized.shape[2]] = resized[..., :3]
    return np.ascontiguousarray(canvas)


def _shape_hwc(shape: Sequence[int]) -> tuple[int, int, int, str]:
    sh = tuple(max(1, int(x)) for x in shape)
    if len(sh) == 4 and sh[0] == 1:
        if sh[1] in (1, 3, 4):
            return sh[2], sh[3], sh[1], "nchw"
        if sh[-1] in (1, 3, 4):
            return sh[1], sh[2], sh[3], "nhwc"
    if len(sh) == 3:
        if sh[0] in (1, 3, 4):
            return sh[1], sh[2], sh[0], "chw"
        if sh[-1] in (1, 3, 4):
            return sh[0], sh[1], sh[2], "hwc"
    raise RuntimeError(f"Unsupported image tensor shape: {sh}")


def _prepare_image_tensor(
    image_path: Path,
    shape: Sequence[int],
    dtype: np.dtype,
    *,
    task: str,
    normalization: str = "",
    preprocess_mode: str = "auto",
    letterbox_pad: int = 114,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if Image is None:
        raise RuntimeError("Pillow is required for Native Full semantic dumps")
    h, w, c, layout = _shape_hwc(shape)
    image = Image.open(image_path).convert("RGB")
    bilinear = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    mode = str(preprocess_mode or "auto").lower()
    if mode == "auto":
        mode = "letterbox" if task == "detection" else "resize"
    if mode == "letterbox":
        hwc = _letterbox_rgb(image, w, h, int(letterbox_pad))
        pad = int(letterbox_pad)
    else:
        hwc = np.asarray(
            image.resize((w, h), resample=bilinear), dtype=np.uint8,
        )
        pad = 0
    if c == 1:
        hwc = hwc[..., :1]
    elif c < hwc.shape[-1]:
        hwc = hwc[..., :c]
    base = hwc.astype(np.float32)
    norm = str(normalization or "").lower()
    if np.dtype(dtype).kind == "f":
        base /= 255.0
        if task == "classification" and ("imagenet" in norm or not norm):
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            if base.shape[-1] == 3:
                base = (base - mean) / std
    else:
        base = hwc
    if layout == "nchw":
        tensor = np.transpose(base, (2, 0, 1))[None, ...]
    elif layout == "nhwc":
        tensor = base[None, ...]
    elif layout == "chw":
        tensor = np.transpose(base, (2, 0, 1))
    else:
        tensor = base
    tensor = np.ascontiguousarray(tensor.astype(dtype, copy=False))
    meta = {
        "mode": "letterbox_rgb_uint8" if mode == "letterbox" else "resize_rgb_uint8",
        "pad_value": int(pad),
        "rgb": True,
        "ort_model_scale": "imagenet" if task == "classification" else "norm",
        "layout": layout.upper(),
        "runtime_dtype": str(np.dtype(dtype)),
        "normalization": normalization or ("imagenet_mean_std" if task == "classification" else "scale_0_1"),
        "color_space": "RGB",
        "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear",
        "resize_rounding": "python_round_ties_to_even",
        "placement": (
            "floor_top_left_remainder_bottom_right"
            if mode == "letterbox" else "not_applicable"
        ),
    }
    return tensor, np.ascontiguousarray(hwc), meta


def _contract(
    task: str, outputs: Mapping[str, np.ndarray],
    declared_contract: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    contract = runtime_output_contract(
        task, outputs, raw_fallback=True, declared_contract=declared_contract,
    )
    if contract.get("contract_family") == "decoded_nms":
        contract.update({
            "e2e_scope": "full_task_pipeline",
            "postprocess_included": True,
            "claim_eligible_e2e": True,
        })
    elif (
        str(task or "").strip().lower() == "classification"
        and contract.get("endpoint_contract_complete") is True
        and str(contract.get("contract_family") or "") in {
            "classification_logits", "classification_probabilities",
        }
    ):
        contract.update({
            "e2e_scope": "full_task_pipeline",
            "claim_eligible_e2e": True,
        })
    else:
        contract.update({
            "e2e_scope": "accelerator_only",
            "postprocess_included": False,
            "requires_host_decode_nms": str(task or "").strip().lower() == "detection",
            "host_postprocess_frozen": False,
            "claim_eligible_e2e": False,
        })
    return contract


def _direct_normalization_for_outputs(
    *,
    task: str,
    model: str,
    image: Path,
    input_hwc: np.ndarray,
    preprocess: Mapping[str, Any],
    outputs: Mapping[str, np.ndarray],
    observed_contract: Mapping[str, Any],
    declared_output_contract: Mapping[str, Any] | str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze Direct-BN6 normalization from authoritative geometry only."""
    if (
        str(task or "").strip().lower() != "detection"
        or str(observed_contract.get("contract_family") or "")
        != "decoded_nms"
    ):
        return {}, {}
    declaration = (
        dict(declared_output_contract)
        if isinstance(declared_output_contract, Mapping) else {}
    )
    source_coordinate_space = str(
        declaration.get("source_coordinate_space") or ""
    ).strip()
    if source_coordinate_space != "model_input_letterbox_xyxy_pixels":
        raise RuntimeError(
            "direct_bn6_source_coordinate_space_not_authoritatively_sealed"
        )
    attestation = observed_contract.get("output_endpoint_attestation")
    if not isinstance(attestation, Mapping):
        raise RuntimeError(
            "direct_bn6_runtime_endpoint_attestation_missing"
        )
    if Image is None:
        raise RuntimeError(
            "Pillow is required for Direct-BN6 normalization geometry"
        )
    with Image.open(image) as original_image:
        original_wh = [
            int(original_image.size[0]), int(original_image.size[1]),
        ]
    input_shape = list(np.asarray(input_hwc).shape)
    if len(input_shape) != 3 or input_shape[0] <= 0 or input_shape[1] <= 0:
        raise RuntimeError("direct_bn6_input_hwc_shape_invalid")
    direct_contract = build_frozen_decoded_nms_normalization_contract(
        source_completed=True,
        model_id=model,
        outputs=outputs,
        input_hw=[int(input_shape[0]), int(input_shape[1])],
        original_wh=original_wh,
        preprocess=preprocess,
        source_coordinate_space=source_coordinate_space,
        source_endpoint_contract_hash=str(
            observed_contract.get("endpoint_contract_hash") or ""
        ),
        source_output_endpoint_attestation=attestation,
    )
    direct_result = FrozenDecodedNmsPostprocessor(
        direct_contract
    ).process(outputs, original_wh=original_wh)
    return direct_contract, direct_result


def _write_manifests(
    *,
    outputs: Mapping[str, np.ndarray],
    out_dir: Path,
    backend: str,
    model: str,
    setup_id: str,
    comparison_backend: str,
    task: str,
    image: Path,
    input_hwc: np.ndarray,
    preprocess: Mapping[str, Any],
    input_tensor: np.ndarray,
    input_name: str,
    runtime_meta: Mapping[str, Any],
    declared_output_contract: Mapping[str, Any] | str | None = None,
    frozen_postprocess_contract: Mapping[str, Any] | None = None,
    frozen_postprocess_result: Mapping[str, Any] | None = None,
    frozen_decoded_nms_normalization_contract: (
        Mapping[str, Any] | None
    ) = None,
    frozen_decoded_nms_normalization_result: (
        Mapping[str, Any] | None
    ) = None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = out_dir / "input_rgb_uint8.bin"
    input_file.write_bytes(np.ascontiguousarray(input_hwc).tobytes())
    runtime_input_file = out_dir / "runtime_input.bin"
    runtime_input_array = np.ascontiguousarray(input_tensor)
    runtime_input_file.write_bytes(runtime_input_array.tobytes())
    input_manifest = out_dir / "native_full_input_manifest.json"
    runtime_color_space = str(
        preprocess.get("color_space")
        or ("RGB" if preprocess.get("rgb") is True else "")
    ).strip().upper()
    runtime_normalization = str(
        preprocess.get("normalization") or ""
    ).strip().lower()
    runtime_preprocess_identity, runtime_preprocess_sha = (
        observed_runtime_preprocessing_identity(
            task=task,
            target_hw=[int(input_hwc.shape[0]), int(input_hwc.shape[1])],
            preprocess_mode=(
                preprocess.get("mode")
                or preprocess.get("preprocess_mode")
            ),
            color_space=runtime_color_space,
            input_domain=(
                "uint8_0_255"
                if np.asarray(input_hwc).dtype == np.uint8
                else str(np.asarray(input_hwc).dtype)
            ),
            resize_interpolation=preprocess.get("resize_interpolation"),
            resize_rounding=preprocess.get("resize_rounding"),
            placement=preprocess.get("placement"),
            pad_value=preprocess.get("pad_value", 0),
            image_scale=preprocess.get("ort_model_scale"),
        )
    )
    runtime_numeric_identity, runtime_numeric_sha = (
        runtime_numeric_input_identity(
            backend=f"native_full_{backend}",
            task=task,
            preprocessing_contract_sha256_value=runtime_preprocess_sha,
            runtime_input_name=input_name,
            runtime_input_shape=[int(x) for x in input_tensor.shape],
            runtime_input_dtype=str(input_tensor.dtype),
            runtime_input_layout=str(preprocess.get("layout") or ""),
            runtime_color_space=runtime_color_space,
            runtime_normalization=runtime_normalization,
        )
    )
    input_payload = {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 2,
        "backend": f"native_full_{backend}",
        "model": model,
        "setup_id": str(setup_id or ""),
        "comparison_backend": str(comparison_backend or ""),
        "case": "full",
        "task": str(task or "").strip().lower(),
        "image": str(image),
        "image_sha256": _sha256(image),
        "input_image": str(image),
        "input_image_sha256": _sha256(image),
        "input_dump": (
            input_file.name if backend == "deepx" else str(input_file)
        ),
        "input_dump_sha256": _sha256(input_file),
        "input_dump_bytes": int(input_file.stat().st_size),
        "input_shape_hwc": [int(x) for x in input_hwc.shape],
        "runtime_input_name": str(input_name),
        "runtime_input_shape": [int(x) for x in input_tensor.shape],
        "runtime_input_dtype": str(input_tensor.dtype),
        "runtime_input_file": (
            runtime_input_file.name
            if backend == "deepx" else str(runtime_input_file)
        ),
        "runtime_input_sha256": _sha256(runtime_input_file),
        "runtime_input_bytes": int(runtime_input_array.nbytes),
        "preprocess": dict(preprocess),
        "runtime_preprocess_mode": str(preprocess.get("mode") or ""),
        "runtime_input_layout": str(preprocess.get("layout") or ""),
        "runtime_color_space": runtime_color_space,
        "runtime_normalization": runtime_normalization,
        "runtime_preprocessing_identity": runtime_preprocess_identity,
        "runtime_preprocessing_sha256": runtime_preprocess_sha,
        "runtime_numeric_input_identity": runtime_numeric_identity,
        "runtime_numeric_input_sha256": runtime_numeric_sha,
    }
    preprocess_mode = str(
        preprocess.get("mode") or preprocess.get("preprocess_mode") or ""
    ).strip().lower()
    if (
        str(task or "").strip().lower() == "detection"
        and preprocess_mode in {"letterbox", "letterbox_rgb_uint8"}
    ):
        if Image is None:
            raise RuntimeError(
                "Pillow is required for sealed letterbox geometry"
            )
        with Image.open(image) as original_image:
            original_wh = [
                int(original_image.size[0]), int(original_image.size[1]),
            ]
        geometry = build_letterbox_geometry_contract(
            input_hw=[
                int(input_hwc.shape[0]), int(input_hwc.shape[1]),
            ],
            original_wh=original_wh,
            preprocess=preprocess,
        )
        # Optional schema-v2 fields remain inside the manifest's own SHA-256
        # envelope.  Historical v2 manifests without these fields remain valid
        # physical evidence; Direct-BN6 V2 completion requires them.
        input_payload.update({
            "original_image_wh": original_wh,
            "letterbox_geometry_contract": geometry,
            "letterbox_geometry_contract_sha256": str(
                geometry["geometry_contract_sha256"]
            ),
        })
    input_manifest.write_text(json.dumps(input_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    entries: list[dict[str, Any]] = []
    for index, (name, value) in enumerate(outputs.items()):
        arr = np.ascontiguousarray(np.asarray(value))
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))
        file_name = f"output_{index:02d}_{safe or index}.bin"
        (out_dir / file_name).write_bytes(arr.tobytes())
        entries.append({
            "name": str(name),
            "file": file_name,
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "bytes": int(arr.nbytes),
        })
    output_manifest = out_dir / "native_full_outputs_manifest.json"
    endpoint_contract = _contract(task, outputs, declared_output_contract)
    if frozen_postprocess_contract:
        verified_frozen = verify_frozen_postprocess_contract(
            frozen_postprocess_contract, outputs=outputs,
        )
        if verified_frozen["source_contract_family"] != endpoint_contract.get("contract_family"):
            raise RuntimeError("semantic_frozen_postprocess_source_family_mismatch")
        result = dict(frozen_postprocess_result or {})
        if (
            str(result.get("postprocess_contract_sha256") or "")
            != str(verified_frozen.get("contract_sha256") or "")
            or result.get("contract_family") != "decoded_nms"
        ):
            raise RuntimeError("semantic_frozen_postprocess_result_binding_invalid")
        endpoint_contract.update({
            "e2e_scope": "full_task_pipeline",
            "postprocess_included": True,
            "requires_host_decode_nms": True,
            "host_postprocess_frozen": True,
            "claim_eligible_e2e": True,
            "frozen_host_postprocess_contract": verified_frozen,
            "frozen_host_postprocess_contract_sha256": str(
                verified_frozen["contract_sha256"]
            ),
            "frozen_host_postprocess_result": result,
        })
    if frozen_decoded_nms_normalization_contract:
        source_attestation = endpoint_contract.get(
            "output_endpoint_attestation"
        )
        verified_direct = (
            verify_frozen_decoded_nms_normalization_contract(
                frozen_decoded_nms_normalization_contract,
                outputs=outputs,
                source_output_endpoint_attestation=(
                    source_attestation
                    if isinstance(source_attestation, Mapping) else None
                ),
            )
        )
        direct_result = dict(
            frozen_decoded_nms_normalization_result or {}
        )
        if (
            str(
                direct_result.get("normalization_contract_sha256") or ""
            ) != str(verified_direct.get("contract_sha256") or "")
            or direct_result.get("contract_family") != "decoded_nms"
            or direct_result.get("coordinate_space")
            != "original_image_xyxy_pixels"
        ):
            raise RuntimeError(
                "semantic_direct_normalization_result_binding_invalid"
            )
        endpoint_contract.update({
            "e2e_scope": "full_task_pipeline",
            "postprocess_included": True,
            "requires_host_decode_nms": False,
            "host_postprocess_frozen": False,
            "normalization_frozen": True,
            "claim_eligible_e2e": True,
            "frozen_decoded_nms_normalization_contract": verified_direct,
            "frozen_decoded_nms_normalization_contract_sha256": str(
                verified_direct["contract_sha256"]
            ),
            "frozen_decoded_nms_normalization_result": direct_result,
        })
    output_payload = {
        "schema": "onnx-splitpoint/runner-output-dump",
        "schema_version": 4,
        "producer": f"native_full_{backend}",
        "backend": f"native_full_{backend}",
        "model": model,
        "setup_id": str(setup_id or ""),
        "comparison_backend": str(comparison_backend or ""),
        "case": "full",
        "execution_mode": "native_full_baseline",
        "quality_first_producer_identity_sha256": str(
            runtime_meta.get("quality_first_producer_identity_sha256") or ""
        ),
        "input_image": str(image),
        "input_image_sha256": _sha256(image),
        "input_manifest": str(input_manifest),
        "input_manifest_sha256": _sha256(input_manifest),
        "boundary_manifest": str(input_manifest),
        "native_boundary_manifest": str(input_manifest),
        "provenance": {
            "image": str(image),
            "image_sha256": _sha256(image),
            "image_source": "exact_file",
            "runtime": dict(runtime_meta),
        },
        "input_contract_mode": str(runtime_meta.get("input_contract_mode") or "explicit"),
        "authoritative_output_contract_resolution": (
            dict(declared_output_contract)
            if isinstance(declared_output_contract, Mapping) else {}
        ),
        **endpoint_contract,
        "outputs": entries,
    }
    output_manifest.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_manifest, input_manifest


def _find_engine(benchmark_set: Path, precision: str) -> Path:
    candidates = [
        benchmark_set / "native_trt" / "full" / precision / f"full_{precision}.engine",
        benchmark_set / "native_trt" / "full" / precision / "full.engine",
    ]
    candidates += sorted((benchmark_set / "native_trt" / "full").glob("**/*.engine")) if (benchmark_set / "native_trt" / "full").is_dir() else []
    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"Native TensorRT Full engine not found under {benchmark_set}")


def _verify_explicit_trt_identity(args: argparse.Namespace) -> dict[str, Any]:
    path_fields = {
        "source_onnx": "explicit_full_source_onnx",
        "build_onnx": "explicit_full_build_onnx",
        "engine": "explicit_full_engine",
        "trtexec": "explicit_full_trtexec",
        "engine_build_receipt": "explicit_full_build_receipt",
    }
    hash_fields = {
        "source_onnx": "expected_source_onnx_sha256",
        "build_onnx": "expected_build_onnx_sha256",
        "engine": "expected_engine_sha256",
        "trtexec": "expected_trtexec_sha256",
        "engine_build_receipt": "expected_engine_build_receipt_file_sha256",
    }
    size_fields = {
        "source_onnx": "expected_source_onnx_size_bytes",
        "build_onnx": "expected_build_onnx_size_bytes",
        "engine": "expected_engine_size_bytes",
        "trtexec": "expected_trtexec_size_bytes",
    }
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, field in path_fields.items():
        text = str(getattr(args, field, "") or "").strip()
        if not text:
            raise RuntimeError(f"explicit TensorRT semantic identity lacks --{field.replace('_', '-')}")
        path = Path(text).expanduser().resolve()
        expected = str(getattr(args, hash_fields[name], "") or "").strip().lower()
        if not path.is_file() or re.fullmatch(r"[0-9a-f]{64}", expected) is None or _sha256(path) != expected:
            raise RuntimeError(f"explicit TensorRT semantic {name} artifact mismatch")
        if name in size_fields:
            size = int(getattr(args, size_fields[name], 0) or 0)
            if size <= 0 or int(path.stat().st_size) != size:
                raise RuntimeError(f"explicit TensorRT semantic {name} size mismatch")
        paths[name] = path
        hashes[name] = expected
    if hashes["source_onnx"] != hashes["build_onnx"]:
        raise RuntimeError("explicit TensorRT semantic source/build ONNX mismatch")
    producer_sha = str(
        args.quality_first_producer_identity_sha256 or ""
    ).strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", producer_sha) is None:
        raise RuntimeError("explicit TensorRT semantic producer identity SHA-256 is invalid")
    receipt_raw = _load_json(paths["engine_build_receipt"])
    if not isinstance(receipt_raw, Mapping):
        raise RuntimeError("explicit TensorRT semantic build receipt is invalid")
    receipt = dict(receipt_raw)
    payload = dict(receipt)
    inner_sha = str(payload.pop("receipt_sha256", "") or "").strip().lower()
    outer_sha = _canonical_json_sha256(receipt)
    canonical_bytes = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    expected_receipt = {
        "source_onnx": str(paths["build_onnx"]),
        "source_onnx_sha256": hashes["build_onnx"],
        "engine": str(paths["engine"]),
        "engine_sha256": hashes["engine"],
        "trtexec": str(paths["trtexec"]),
        "trtexec_sha256": hashes["trtexec"],
    }
    command = receipt.get("command")
    argv = [str(value) for value in command] if isinstance(command, list) else []
    if (
        receipt.get("schema") != "onnx-splitpoint/tensorrt-engine-build-receipt"
        or int(receipt.get("schema_version") or 0) != 1
        or receipt.get("build_returncode") != 0
        or receipt.get("dry_run") is not False
        or re.fullmatch(r"[0-9a-f]{64}", inner_sha) is None
        or _canonical_json_sha256(payload) != inner_sha
        or any(str(receipt.get(key) or "") != value for key, value in expected_receipt.items())
        or not argv or argv[0] != str(paths["trtexec"])
        or [value for value in argv[1:] if value.startswith("--onnx=")]
        != [f"--onnx={paths['build_onnx']}"]
        or [value for value in argv[1:] if value.startswith("--saveEngine=")]
        != [f"--saveEngine={paths['engine']}"]
    ):
        raise RuntimeError("explicit TensorRT semantic receipt binding mismatch")
    if outer_sha != str(args.expected_engine_build_receipt_sha256 or "").strip().lower():
        raise RuntimeError("explicit TensorRT semantic outer receipt SHA-256 mismatch")
    if inner_sha != str(args.expected_trt_engine_build_receipt_sha256 or "").strip().lower():
        raise RuntimeError("explicit TensorRT semantic inner receipt SHA-256 mismatch")
    if len(canonical_bytes) != int(args.expected_engine_build_receipt_size_bytes or 0):
        raise RuntimeError("explicit TensorRT semantic canonical receipt size mismatch")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": hashes,
        "engine_build_receipt_sha256": outer_sha,
        "engine_build_receipt_file_sha256": hashes["engine_build_receipt"],
        "trt_engine_build_receipt_sha256": inner_sha,
        "engine_build_receipt_size_bytes": len(canonical_bytes),
        "quality_first_producer_identity_sha256": producer_sha,
        "status": "quality_first_identity_verified_exact",
    }


def _run_tensorrt(
    benchmark_set: Path, image: Path, out_dir: Path, model: str, task: str,
    precision: str, setup_id: str, comparison_backend: str,
    explicit_identity: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        from native_hailo10_trt_e2e_from_benchmarkset import NativeTRT  # type: ignore
    except Exception as exc:  # pragma: no cover - target dependency
        raise RuntimeError(f"NativeTRT import failed: {type(exc).__name__}: {exc}") from exc
    engine_path = Path(str((explicit_identity.get("paths") or {}).get("engine") or ""))
    if not engine_path.is_file():
        raise RuntimeError("quality-sealed TensorRT Full engine is missing")
    runtime = NativeTRT(engine_path)
    try:
        if not runtime.inputs:
            raise RuntimeError("TensorRT Full engine has no input bindings")
        input_name = str(runtime.inputs[0])
        shape = tuple(runtime.shapes[input_name])
        dtype = np.dtype(runtime.dtypes[input_name])
        tensor, hwc, prep = _prepare_image_tensor(image, shape, dtype, task=task)
        outputs = runtime.run({input_name: tensor})
        declaration = load_authoritative_output_contract(
            benchmark_set, backend="tensorrt", model_id=model,
            variant="full", task=task,
        )
        frozen_contract: dict[str, Any] = {}
        frozen_result: dict[str, Any] = {}
        direct_contract: dict[str, Any] = {}
        direct_result: dict[str, Any] = {}
        observed_contract = _contract(task, outputs, declaration)
        if str(observed_contract.get("contract_family") or "") in {"raw_head", "decoded_pre_nms"}:
            input_height, input_width, _channels, _layout = _shape_hwc(
                list(shape)
            )
            if Image is None:
                raise RuntimeError(
                    "Pillow is required for frozen detection geometry"
                )
            with Image.open(image) as original_image:
                original_wh = [
                    int(original_image.size[0]),
                    int(original_image.size[1]),
                ]
            frozen_contract = build_frozen_postprocess_contract(
                model_id=model,
                source_contract_family=str(observed_contract["contract_family"]),
                model_sha256=_bound_model_sha256(
                    {"source_onnx_sha256": str(
                        (explicit_identity.get("hashes") or {}).get(
                            "source_onnx"
                        ) or ""
                    )},
                    declaration,
                    model_id=model,
                ),
                outputs=outputs,
                input_hw=[int(input_height), int(input_width)],
                original_wh=original_wh,
            )
            frozen_result = FrozenDetectionPostprocessor(
                frozen_contract
            ).process(outputs, original_wh=original_wh)
        elif (
            task == "detection"
            and str(observed_contract.get("contract_family") or "")
            == "decoded_nms"
        ):
            direct_contract, direct_result = (
                _direct_normalization_for_outputs(
                    task=task,
                    model=model,
                    image=image,
                    input_hwc=hwc,
                    preprocess=prep,
                    outputs=outputs,
                    observed_contract=observed_contract,
                    declared_output_contract=declaration,
                )
            )
        manifest, input_manifest = _write_manifests(
            outputs=outputs,
            out_dir=out_dir,
            backend="tensorrt",
            model=model,
            setup_id=setup_id,
            comparison_backend=comparison_backend,
            task=task,
            image=image,
            input_hwc=hwc,
            preprocess=prep,
            input_tensor=tensor,
            input_name=input_name,
            runtime_meta={
                "engine": str(engine_path), "precision": precision,
                "quality_first_identity": dict(explicit_identity),
                "quality_first_producer_identity_sha256": str(
                    explicit_identity["quality_first_producer_identity_sha256"]
                ),
            },
            declared_output_contract=declaration,
            frozen_postprocess_contract=frozen_contract,
            frozen_postprocess_result=frozen_result,
            frozen_decoded_nms_normalization_contract=direct_contract,
            frozen_decoded_nms_normalization_result=direct_result,
        )
        return {
            "ok": True, "output_manifest": str(manifest),
            "input_manifest": str(input_manifest), "engine": str(engine_path),
            "quality_first_producer_identity_sha256": str(
                explicit_identity["quality_first_producer_identity_sha256"]
            ),
            "frozen_host_postprocess_contract": frozen_contract,
            "frozen_host_postprocess_contract_sha256": str(
                frozen_contract.get("contract_sha256") or ""
            ),
            "frozen_host_postprocess_result": frozen_result,
            "frozen_decoded_nms_normalization_contract": direct_contract,
            "frozen_decoded_nms_normalization_contract_sha256": str(
                direct_contract.get("contract_sha256") or ""
            ),
            "frozen_decoded_nms_normalization_result": direct_result,
        }
    finally:
        runtime.close()


def _find_deepx_contract(benchmark_set: Path) -> tuple[Path, dict[str, Any]]:
    candidates = [
        benchmark_set / "deepx" / "deepx_m1" / "full" / "model.dxnn",
    ]
    candidates += sorted((benchmark_set / "deepx").glob("**/full/*.dxnn")) if (benchmark_set / "deepx").is_dir() else []
    dxnn = next((p.resolve() for p in candidates if p.is_file()), None)
    if dxnn is None:
        raise FileNotFoundError(f"DEEPX Full DXNN not found under {benchmark_set}")
    contract_path = benchmark_set / "deepx" / "deepx_m1" / "full" / "output_contract.json"
    contract = _load_json(contract_path) if contract_path.is_file() else {}
    return dxnn, dict(contract) if isinstance(contract, Mapping) else {}


def _deepx_candidate_tensors(
    image: Path, contract: Mapping[str, Any], task: str, *,
    allow_diagnostic_probes: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, dict[str, Any], str]]:
    inp = contract.get("input") if isinstance(contract.get("input"), Mapping) else {}
    if not allow_diagnostic_probes:
        missing = [
            key for key in (
                "shape", "dtype", "layout", "normalization", "color_space",
                "preprocess_mode",
            )
            if inp.get(key) in (None, "", [])
        ]
        if missing:
            raise RuntimeError(
                "DeepX explicit input contract is incomplete; missing="
                + ",".join(missing)
            )
        layout = str(inp.get("layout") or "").strip().upper()
        if layout not in {"NCHW", "NHWC", "CHW", "HWC"}:
            raise RuntimeError(f"DeepX explicit input layout is unsupported: {layout!r}")
        if str(inp.get("color_space") or "").strip().upper() != "RGB":
            raise RuntimeError("DeepX explicit input color_space must be RGB")
    shape = list(inp.get("shape") or ([1, 3, 640, 640] if task == "detection" else [1, 3, 224, 224]))
    dtype_text = str(inp.get("dtype") or "float32").lower()
    dtype = np.uint8 if "uint8" in dtype_text else np.float32
    normalization = str(inp.get("normalization") or "")
    direct = _prepare_image_tensor(
        image, shape, np.dtype(dtype), task=task, normalization=normalization,
        preprocess_mode=str(inp.get("preprocess_mode") or "resize"),
        letterbox_pad=int(inp.get("letterbox_pad_value") or 114),
    )
    candidates: list[tuple[np.ndarray, np.ndarray, dict[str, Any], str]] = [(direct[0], direct[1], direct[2], "contract")]
    if not allow_diagnostic_probes:
        return candidates
    # Older DXNNs sometimes expose HWC uint8 despite an ONNX-derived NCHW
    # contract.  Keep deterministic fallback candidates, but record which one
    # actually worked in the semantic report.
    h, w, _c, _layout = _shape_hwc(shape)
    if Image is None:
        return candidates
    hwc = np.asarray(Image.open(image).convert("RGB").resize((w, h)), dtype=np.uint8)
    variants = [
        (np.ascontiguousarray(hwc), "hwc_uint8"),
        (np.ascontiguousarray(hwc[None, ...]), "nhwc_uint8"),
        (np.ascontiguousarray(np.transpose(hwc, (2, 0, 1))[None, ...]), "nchw_uint8"),
        (np.ascontiguousarray(np.transpose(hwc, (2, 0, 1))[None, ...].astype(np.float32) / 255.0), "nchw_float01"),
        (np.ascontiguousarray(hwc[None, ...].astype(np.float32) / 255.0), "nhwc_float01"),
    ]
    seen = {(tuple(direct[0].shape), str(direct[0].dtype))}
    for tensor, source in variants:
        key = (tuple(tensor.shape), str(tensor.dtype))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((tensor, np.ascontiguousarray(hwc), {"mode": "resize_rgb_uint8", "pad_value": 0, "rgb": True, "ort_model_scale": "imagenet" if task == "classification" else "norm", "layout": source}, source))
    return candidates


def _deepx_input_contract_mode(source: str) -> str:
    """Classify only verified DeepX input sources as explicit."""
    return (
        "explicit"
        if str(source or "").strip() in {
            "contract",
            "shared_pre_timing_sealed_manifest",
        }
        else "diagnostic_autodetect"
    )


def _run_deepx(
    benchmark_set: Path, image: Path, out_dir: Path, model: str, task: str,
    setup_id: str, comparison_backend: str, *,
    allow_diagnostic_input_probes: bool = False,
    prepared_input_manifest: Path | None = None,
) -> dict[str, Any]:
    try:
        from dx_engine import InferenceEngine  # type: ignore
    except Exception as exc:  # pragma: no cover - target dependency
        raise RuntimeError(f"dx_engine import failed: {type(exc).__name__}: {exc}") from exc
    dxnn, contract = _find_deepx_contract(benchmark_set)
    endpoint_declaration = load_authoritative_output_contract(
        benchmark_set, backend="deepx_m1", model_id=model,
        variant="full", task=task,
    )
    engine = InferenceEngine(str(dxnn))
    errors: list[dict[str, Any]] = []
    if prepared_input_manifest is not None:
        sealed = load_sealed_deepx_native_full_input(
            prepared_input_manifest,
            image_path=image,
            input_contract=contract,
            task=task,
            expected_model=model,
            expected_setup_id=setup_id,
            expected_comparison_backend=comparison_backend,
        )
        candidates = [(
            np.ascontiguousarray(sealed["runtime_input"]),
            np.ascontiguousarray(sealed["input_hwc"]),
            dict(sealed["preprocess"]),
            "shared_pre_timing_sealed_manifest",
        )]
    else:
        candidates = _deepx_candidate_tensors(
            image, contract, task,
            allow_diagnostic_probes=allow_diagnostic_input_probes,
        )
    for tensor, hwc, prep, source in candidates:
        try:
            if hasattr(engine, "run"):
                raw = engine.run([np.ascontiguousarray(tensor)])
            elif hasattr(engine, "Run"):
                raw = engine.Run([np.ascontiguousarray(tensor)])
            else:
                raise RuntimeError("dx_engine.InferenceEngine has no run/Run method")
            values = list(raw) if isinstance(raw, (list, tuple)) else [raw]
            names: list[str] = []
            declared = contract.get("outputs") if isinstance(contract.get("outputs"), list) else []
            for index, _value in enumerate(values):
                if index < len(declared) and isinstance(declared[index], Mapping):
                    names.append(str(declared[index].get("name") or f"output_{index}"))
                else:
                    names.append(f"output_{index}")
            outputs = {name: np.asarray(value) for name, value in zip(names, values)}
            frozen_contract: dict[str, Any] = {}
            frozen_result: dict[str, Any] = {}
            direct_contract: dict[str, Any] = {}
            direct_result: dict[str, Any] = {}
            observed_contract = _contract(task, outputs, endpoint_declaration)
            source_family = str(observed_contract.get("contract_family") or "")
            if (
                task == "detection"
                and endpoint_declaration.get("stage") == "decoded_pre_nms"
                and observed_contract.get("endpoint_contract_complete") is not True
            ):
                observed_attestation = observed_contract.get("output_endpoint_attestation") or {}
                raise RuntimeError(
                    "deepx_full_pre_nms_runtime_endpoint_invalid: "
                    + str(observed_attestation.get("reason") or "incomplete_endpoint")
                )
            if task == "detection" and source_family in {"raw_head", "decoded_pre_nms"}:
                inp = contract.get("input") if isinstance(contract.get("input"), Mapping) else {}
                shape = list(inp.get("shape") or list(tensor.shape))
                input_height, input_width, _channels, _layout = _shape_hwc(shape)
                if Image is None:
                    raise RuntimeError("Pillow is required for frozen detection geometry")
                with Image.open(image) as original_image:
                    original_wh = [
                        int(original_image.size[0]), int(original_image.size[1]),
                    ]
                frozen_contract = build_frozen_postprocess_contract(
                    model_id=model,
                    model_sha256=_bound_model_sha256(
                        contract, endpoint_declaration,
                        model_id=model,
                    ),
                    source_contract_family=source_family,
                    outputs=outputs,
                    input_hw=[int(input_height), int(input_width)],
                    original_wh=original_wh,
                )
                frozen_processor = FrozenDetectionPostprocessor(frozen_contract)
                frozen_result = frozen_processor.process(
                    outputs, original_wh=original_wh,
                )
            elif (
                task == "detection"
                and str(observed_contract.get("contract_family") or "")
                == "decoded_nms"
            ):
                direct_contract, direct_result = (
                    _direct_normalization_for_outputs(
                        task=task,
                        model=model,
                        image=image,
                        input_hwc=hwc,
                        preprocess=prep,
                        outputs=outputs,
                        observed_contract=observed_contract,
                        declared_output_contract=endpoint_declaration,
                    )
                )
            manifest, input_manifest = _write_manifests(
                outputs=outputs,
                out_dir=out_dir,
                backend="deepx",
                model=model,
                setup_id=setup_id,
                comparison_backend=comparison_backend,
                task=task,
                image=image,
                input_hwc=hwc,
                preprocess=prep,
                input_tensor=tensor,
                input_name=str((contract.get("input") or {}).get("name") or "input") if isinstance(contract.get("input"), Mapping) else "input",
                runtime_meta={
                    "dxnn": str(dxnn), "input_candidate": source,
                    "input_contract_mode": _deepx_input_contract_mode(source),
                },
                declared_output_contract=endpoint_declaration,
                frozen_postprocess_contract=frozen_contract,
                frozen_postprocess_result=frozen_result,
                frozen_decoded_nms_normalization_contract=direct_contract,
                frozen_decoded_nms_normalization_result=direct_result,
            )
            return {
                "ok": True, "output_manifest": str(manifest),
                "input_manifest": str(input_manifest), "dxnn": str(dxnn),
                "input_candidate": source, "attempt_errors": errors,
                "frozen_host_postprocess_contract": frozen_contract,
                "frozen_host_postprocess_contract_sha256": str(
                    frozen_contract.get("contract_sha256") or ""
                ),
                "frozen_host_postprocess_result": frozen_result,
                "frozen_decoded_nms_normalization_contract": (
                    direct_contract
                ),
                "frozen_decoded_nms_normalization_contract_sha256": str(
                    direct_contract.get("contract_sha256") or ""
                ),
                "frozen_decoded_nms_normalization_result": direct_result,
            }
        except Exception as exc:
            errors.append({"source": source, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "error": f"{type(exc).__name__}: {exc}"})
    raise RuntimeError("No DEEPX Full semantic input candidate succeeded: " + json.dumps(errors)[-6000:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-set", required=True)
    parser.add_argument("--backend", required=True, choices=["tensorrt", "deepx"])
    parser.add_argument("--model", default="")
    parser.add_argument("--task", default="")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--trt-precision", default="fp16")
    parser.add_argument("--explicit-full-source-onnx", default="")
    parser.add_argument("--explicit-full-build-onnx", default="")
    parser.add_argument("--explicit-full-engine", default="")
    parser.add_argument("--explicit-full-trtexec", default="")
    parser.add_argument("--explicit-full-build-receipt", default="")
    parser.add_argument("--expected-source-onnx-sha256", default="")
    parser.add_argument("--expected-build-onnx-sha256", default="")
    parser.add_argument("--expected-engine-sha256", default="")
    parser.add_argument("--expected-trtexec-sha256", default="")
    parser.add_argument("--expected-engine-build-receipt-sha256", default="")
    parser.add_argument("--expected-engine-build-receipt-file-sha256", default="")
    parser.add_argument("--expected-trt-engine-build-receipt-sha256", default="")
    parser.add_argument("--expected-source-onnx-size-bytes", type=int, default=0)
    parser.add_argument("--expected-build-onnx-size-bytes", type=int, default=0)
    parser.add_argument("--expected-engine-size-bytes", type=int, default=0)
    parser.add_argument("--expected-trtexec-size-bytes", type=int, default=0)
    parser.add_argument("--expected-engine-build-receipt-size-bytes", type=int, default=0)
    parser.add_argument("--quality-first-producer-identity-sha256", default="")
    parser.add_argument("--setup-id", default="")
    parser.add_argument("--comparison-backend", default="")
    parser.add_argument(
        "--prepared-input-manifest", default="",
        help=(
            "Transportable schema-v2 DeepX input manifest created before the "
            "Generic timed loop. Native Full verifies and replays its bytes."
        ),
    )
    parser.add_argument(
        "--diagnostic-deepx-input-probes", action="store_true",
        help="Allow legacy HWC/NHWC/NCHW probe permutations. Diagnostic only; final evidence uses the explicit archived contract.",
    )
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    benchmark_set = Path(args.benchmark_set).expanduser().resolve()
    model = _model_id(benchmark_set, args.model)
    task = str(args.task or _task(benchmark_set, model)).lower()
    image = _resolve_image(benchmark_set, args.image)
    out_dir = Path(args.out_dir).expanduser().resolve()
    try:
        if args.backend == "tensorrt":
            explicit_identity = _verify_explicit_trt_identity(args)
            result = _run_tensorrt(
                benchmark_set, image, out_dir, model, task, args.trt_precision,
                str(args.setup_id or ""), str(args.comparison_backend or ""),
                explicit_identity,
            )
        else:
            result = _run_deepx(
                benchmark_set, image, out_dir, model, task,
                str(args.setup_id or ""), str(args.comparison_backend or ""),
                allow_diagnostic_input_probes=bool(args.diagnostic_deepx_input_probes),
                prepared_input_manifest=(
                    Path(args.prepared_input_manifest).expanduser()
                    if str(args.prepared_input_manifest or "").strip()
                    else None
                ),
            )
        result.update({
            "backend": f"native_full_{args.backend}", "model": model,
            "setup_id": str(args.setup_id or ""),
            "comparison_backend": str(args.comparison_backend or ""),
            "task": task, "image": str(image), "image_sha256": _sha256(image),
        })
        rc = 0
    except Exception as exc:
        result = {
            "ok": False, "backend": f"native_full_{args.backend}",
            "model": model, "setup_id": str(args.setup_id or ""),
            "comparison_backend": str(args.comparison_backend or ""),
            "task": task, "image": str(image), "image_sha256": _sha256(image),
            "failure_reason": "native_full_semantic_dump_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        rc = 1
    report_path = Path(args.json_out).expanduser().resolve() if args.json_out else out_dir / "native_full_semantic_dump.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({**result, "report": str(report_path)}, indent=2, ensure_ascii=False), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
