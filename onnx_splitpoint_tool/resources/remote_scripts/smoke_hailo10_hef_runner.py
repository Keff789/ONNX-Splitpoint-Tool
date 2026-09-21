#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.runners.request_latency import RequestLatency
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from onnx_splitpoint_tool.runners._types import RunCfg
from onnx_splitpoint_tool.runners.backends.hailo_backend import HailoBackend
from onnx_splitpoint_tool.native_output_endpoint import (
    load_authoritative_output_contract,
    runtime_output_contract,
)
from onnx_splitpoint_tool.native_detection_postprocess import (
    FrozenDecodedNmsPostprocessor,
    FrozenDetectionPostprocessor,
    FrozenPostprocessError,
    build_completed_detection_endpoint_attestation,
    build_frozen_decoded_nms_normalization_contract,
    build_frozen_postprocess_contract,
    build_normalized_detection_endpoint_attestation,
    canonical_json_bytes,
    persist_completed_result_artifact as _persist_shared_completed_result_artifact,
    tensor_signature,
    verify_frozen_decoded_nms_normalization_contract,
    verify_frozen_postprocess_contract,
)
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    prepare_rgb_uint8_image,
    preprocessing_contract_sha256,
    resolve_image_preprocessing_contract,
    runtime_numeric_input_identity,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _input_tensor(shape: tuple[int, ...], *, quantized: bool, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if not shape:
        shape = (1,)
    if quantized:
        return np.ascontiguousarray(rng.integers(0, 256, size=shape, dtype=np.uint8))
    return np.ascontiguousarray(rng.random(shape).astype(np.float32, copy=False))


def _file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bound_model_sha256(*sources: Mapping[str, Any]) -> str:
    values = {
        str(source.get(key) or "").strip().lower().removeprefix("sha256:")
        for source in sources
        if isinstance(source, Mapping)
        for key in (
            "model_sha256", "full_model_sha256", "terminal_model_sha256",
            "source_onnx_sha256",
        )
        if str(source.get(key) or "").strip()
    }
    if len(values) != 1:
        raise RuntimeError("raw_head_model_sha256_missing_or_conflicting")
    value = next(iter(values))
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError("raw_head_model_sha256_invalid")
    return value


def _runtime_input_evidence(
    *,
    inputs: Mapping[str, np.ndarray],
    prepared_input_names: list[str],
    prepared_runtime_shapes: Mapping[str, tuple[int, ...]],
    quantized_inputs: bool,
    bound_runtime_contract: Mapping[str, Any] | None = None,
    preverified_sha256: str = "",
    runtime_input_file: str | Path | None = None,
) -> dict[str, Any]:
    """Attest the exact post-hotloop tensor bound to the first runtime slot."""
    if not inputs:
        raise RuntimeError("runtime input evidence is unavailable")
    input_name, raw_tensor = next(iter(inputs.items()))
    tensor = np.ascontiguousarray(np.asarray(raw_tensor))
    expected_shape = tuple(
        int(value)
        for value in tuple(prepared_runtime_shapes.get(input_name, ()))
    )
    expected_dtype = np.dtype("uint8" if quantized_inputs else "float32")
    if (
        input_name not in prepared_input_names
        or not expected_shape
        or tuple(tensor.shape) != expected_shape
        or np.dtype(tensor.dtype) != expected_dtype
    ):
        raise RuntimeError("runtime input name/shape/dtype binding mismatch")

    tensor_sha256 = hashlib.sha256(tensor.tobytes(order="C")).hexdigest()
    tensor_bytes = int(tensor.nbytes)
    contract = dict(bound_runtime_contract or {})
    preflight_bound = bool(contract)
    if preflight_bound:
        if (
            str(contract.get("runtime_input_name") or "") != input_name
            or tuple(contract.get("runtime_input_shape") or ())
            != tuple(tensor.shape)
            or np.dtype(str(contract.get("runtime_input_dtype") or ""))
            != np.dtype(tensor.dtype)
            or int(contract.get("runtime_input_bytes") or 0) != tensor_bytes
            or str(contract.get("runtime_input_sha256") or "").strip().lower()
            != tensor_sha256
            or str(preverified_sha256 or "").strip().lower()
            != tensor_sha256
        ):
            raise RuntimeError("preflight runtime input content binding mismatch")

    input_file = (
        Path(runtime_input_file).expanduser().resolve()
        if runtime_input_file else None
    )
    if input_file is not None:
        if (
            not input_file.is_file()
            or int(input_file.stat().st_size) != tensor_bytes
            or _file_sha256(input_file) != tensor_sha256
        ):
            raise RuntimeError("runtime input file/content binding mismatch")

    return {
        "runtime_input_name": str(input_name),
        "runtime_input_shape": [int(value) for value in tensor.shape],
        "runtime_input_dtype": str(tensor.dtype),
        "runtime_input_bytes": tensor_bytes,
        "runtime_input_sha256": tensor_sha256,
        "runtime_input_file": str(input_file) if input_file is not None else "",
        "runtime_input_binding_verified": True,
        "runtime_input_preflight_bound": preflight_bound,
        "runtime_input_hash_source": "post_hotloop_exact_tensor_bytes",
    }


def _persist_completed_result_artifact(
    artifact: Mapping[str, Any] | None,
    expected_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Atomically materialize and re-hash one canonical completed result."""
    if not isinstance(artifact, Mapping) or not artifact:
        return {
            "saved": False,
            "path": "",
            "file_sha256": "",
        }
    if str(expected_sha256 or "").strip().lower() != hashlib.sha256(
        canonical_json_bytes(dict(artifact))
    ).hexdigest():
        raise RuntimeError(
            "completed result artifact canonical SHA-256 mismatch"
        )
    evidence = _persist_shared_completed_result_artifact(
        artifact,
        expected_sha256=expected_sha256,
        output_path=output_path,
    )
    return {
        "saved": True,
        "path": str(evidence["completed_task_result_artifact_path"]),
        "file_sha256": str(
            evidence["completed_task_result_artifact_file_sha256"]
        ),
    }


_PREVERIFIED_RUNTIME_INPUT_SCHEMA = "onnx-splitpoint/preverified-runtime-input-tensor"
_PREVERIFIED_RUNTIME_INPUT_VERSION = 1
_BOUND_RUNTIME_DTYPES: dict[str, np.dtype[Any]] = {
    name: np.dtype(name)
    for name in (
        "uint8", "int8", "uint16", "int16", "uint32", "int32",
        "float16", "float32", "float64",
    )
}


def _json_string_list(raw: str, *, field: str) -> list[str]:
    try:
        value = json.loads(str(raw or ""))
    except Exception as exc:
        raise RuntimeError(f"{field} must be a JSON string list: {exc}") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise RuntimeError(f"{field} must be a non-empty JSON string list")
    result = [item.strip() for item in value]
    if len(set(result)) != len(result):
        raise RuntimeError(f"{field} must not contain duplicate names")
    return result


def _runtime_input_contract(raw: str) -> dict[str, Any]:
    """Parse the preflight-sealed tensor contract without reading its file."""
    try:
        value = json.loads(str(raw or ""))
    except Exception as exc:
        raise RuntimeError(f"runtime input contract JSON is invalid: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("runtime input contract must be a JSON object")
    contract = dict(value)
    if (
        contract.get("schema") != _PREVERIFIED_RUNTIME_INPUT_SCHEMA
        or int(contract.get("schema_version") or 0) != _PREVERIFIED_RUNTIME_INPUT_VERSION
    ):
        raise RuntimeError("runtime input contract schema/version is incompatible")
    name = str(contract.get("runtime_input_name") or "").strip()
    if not name:
        raise RuntimeError("runtime input contract has no input name")
    shape_raw = contract.get("runtime_input_shape")
    if (
        not isinstance(shape_raw, (list, tuple))
        or not shape_raw
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape_raw)
    ):
        raise RuntimeError("runtime input contract shape must contain positive integers")
    shape = [int(dim) for dim in shape_raw]
    dtype_name = str(contract.get("runtime_input_dtype") or "").strip().lower()
    if dtype_name not in _BOUND_RUNTIME_DTYPES:
        raise RuntimeError(f"unsupported runtime input dtype: {dtype_name or '<missing>'}")
    declared_bytes = contract.get("runtime_input_bytes")
    if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int) or declared_bytes <= 0:
        raise RuntimeError("runtime input contract byte count must be a positive integer")
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * int(_BOUND_RUNTIME_DTYPES[dtype_name].itemsize)
    if int(declared_bytes) != expected_bytes:
        raise RuntimeError(
            f"runtime input contract byte count mismatch: declared={declared_bytes} expected={expected_bytes}"
        )
    digest = str(contract.get("runtime_input_sha256") or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise RuntimeError("runtime input contract SHA-256 is invalid")
    preprocess = contract.get("preprocess")
    if preprocess is not None and not isinstance(preprocess, Mapping):
        raise RuntimeError("runtime input preprocess provenance must be an object")
    return {
        **contract,
        "runtime_input_name": name,
        "runtime_input_shape": shape,
        "runtime_input_dtype": dtype_name,
        "runtime_input_bytes": int(declared_bytes),
        "runtime_input_sha256": digest,
        "preprocess": dict(preprocess or {}),
    }


def _canonicalize_frozen_postprocess_outputs(
    outputs: Mapping[str, np.ndarray],
    contract: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Adapt only a sealed packed YOLO HWC/NHWC head to its rank-5 form."""
    signature = contract.get("raw_output_tensor_signature")
    raw_entries = (
        signature.get("tensors")
        if isinstance(signature, Mapping) else None
    )
    if (
        not isinstance(raw_entries, list)
        or not raw_entries
        or int(signature.get("tensor_count") or 0) != len(raw_entries)
        or len(outputs) != len(raw_entries)
    ):
        raise FrozenPostprocessError(
            "frozen_postprocess_tensor_signature_mismatch"
        )

    expected_names: list[str] = []
    expected_entries: list[tuple[str, tuple[int, ...], np.dtype[Any]]] = []
    for position, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise FrozenPostprocessError(
                "frozen_postprocess_tensor_signature_mismatch"
            )
        name = str(raw_entry.get("name") or "")
        raw_shape = raw_entry.get("shape")
        raw_rank = raw_entry.get("rank")
        raw_index = raw_entry.get("index")
        dtype_name = str(raw_entry.get("dtype") or "")
        if (
            not name
            or name in expected_names
            or isinstance(raw_index, bool)
            or not isinstance(raw_index, int)
            or raw_index != position
            or not isinstance(raw_shape, list)
            or not raw_shape
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in raw_shape
            )
            or isinstance(raw_rank, bool)
            or not isinstance(raw_rank, int)
            or raw_rank != len(raw_shape)
        ):
            raise FrozenPostprocessError(
                "frozen_postprocess_tensor_signature_mismatch"
            )
        try:
            expected_dtype = np.dtype(dtype_name)
        except (TypeError, ValueError):
            raise FrozenPostprocessError(
                "frozen_postprocess_tensor_signature_mismatch"
            ) from None
        if expected_dtype.kind not in "fiu":
            raise FrozenPostprocessError(
                "frozen_postprocess_tensor_signature_mismatch"
            )
        expected_names.append(name)
        expected_entries.append((
            name,
            tuple(int(dim) for dim in raw_shape),
            expected_dtype,
        ))

    if list(outputs) != expected_names:
        raise FrozenPostprocessError(
            "frozen_postprocess_tensor_signature_mismatch"
        )

    canonical: dict[str, np.ndarray] = {}
    for name, expected_shape, expected_dtype in expected_entries:
        array = np.asarray(outputs[name])
        if array.dtype != expected_dtype:
            raise FrozenPostprocessError(
                "frozen_postprocess_tensor_signature_mismatch"
            )
        if tuple(array.shape) == expected_shape:
            canonical[name] = array
            continue

        packed: np.ndarray | None = None
        if len(expected_shape) == 5 and expected_shape[0] == 1:
            _batch, anchors, height, width, channels = expected_shape
            packed_channels = anchors * channels
            if tuple(array.shape) == (height, width, packed_channels):
                packed = array
            elif tuple(array.shape) == (
                1, height, width, packed_channels,
            ):
                packed = array[0]
            if packed is not None:
                canonical[name] = np.ascontiguousarray(
                    packed.reshape(
                        height, width, anchors, channels,
                    ).transpose(2, 0, 1, 3)[None, ...]
                )
                continue

        raise FrozenPostprocessError(
            "frozen_postprocess_tensor_signature_mismatch"
        )

    if dict(signature) != tensor_signature(canonical):
        raise FrozenPostprocessError(
            "frozen_postprocess_tensor_signature_mismatch"
        )
    return canonical


def _load_preverified_runtime_input(
    path: str | Path,
    contract: Mapping[str, Any],
    *,
    preverified_sha256: str,
    prepared_input_names: list[str],
    prepared_runtime_shapes: Mapping[str, tuple[int, ...]],
    quantized_inputs: bool,
) -> dict[str, np.ndarray]:
    """Load a preflight-bound tensor exactly; deliberately performs no hash work."""
    runtime_path = Path(path).expanduser().resolve()
    if not runtime_path.is_file():
        raise RuntimeError(f"preverified runtime input is missing: {runtime_path}")
    declared_sha = str(contract.get("runtime_input_sha256") or "").strip().lower()
    supplied_sha = str(preverified_sha256 or "").strip().lower()
    if supplied_sha != declared_sha:
        raise RuntimeError("preverified runtime input SHA-256 does not match the sealed contract")
    if len(prepared_input_names) != 1:
        raise RuntimeError(
            f"bound runtime input requires exactly one prepared input, got {prepared_input_names}"
        )
    runtime_name = str(contract.get("runtime_input_name") or "")
    if prepared_input_names[0] != runtime_name:
        raise RuntimeError(
            f"prepared runtime input name mismatch: prepared={prepared_input_names[0]!r} contract={runtime_name!r}"
        )
    contract_shape = tuple(int(dim) for dim in list(contract.get("runtime_input_shape") or []))
    prepared_shape = tuple(int(dim) for dim in tuple(prepared_runtime_shapes.get(runtime_name, ())))
    if not prepared_shape or prepared_shape != contract_shape:
        raise RuntimeError(
            f"prepared runtime input shape mismatch: prepared={prepared_shape} contract={contract_shape}"
        )
    dtype_name = str(contract.get("runtime_input_dtype") or "").strip().lower()
    expected_dtype = "uint8" if quantized_inputs else "float32"
    if dtype_name != expected_dtype:
        raise RuntimeError(
            f"prepared runtime input dtype mismatch: runtime expects {expected_dtype}, contract has {dtype_name}"
        )
    declared_bytes = int(contract.get("runtime_input_bytes") or 0)
    actual_bytes = int(runtime_path.stat().st_size)
    if actual_bytes != declared_bytes:
        raise RuntimeError(
            f"runtime input file-size mismatch: file={actual_bytes} contract={declared_bytes}"
        )
    tensor = np.fromfile(runtime_path, dtype=_BOUND_RUNTIME_DTYPES[dtype_name])
    if int(tensor.nbytes) != declared_bytes or int(tensor.size) != int(np.prod(contract_shape, dtype=np.int64)):
        raise RuntimeError("runtime input tensor element/byte count does not match the sealed contract")
    return {runtime_name: np.ascontiguousarray(tensor.reshape(contract_shape))}


def _shape_hwc(shape: tuple[int, ...]) -> tuple[int, int, int, str]:
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
    raise RuntimeError(f"unsupported image input shape {sh}")


def _letterbox(image: Any, width: int, height: int, pad: int) -> np.ndarray:
    contract = canonical_image_preprocessing_contract(
        "detection", (int(height), int(width)),
    )
    if int(pad) != int(contract["pad_value"]):
        raise RuntimeError(
            "Hailo Native Full detection requires canonical letterbox pad 114"
        )
    prepared, _geometry = prepare_rgb_uint8_image(
        np.asarray(image.convert("RGB"), dtype=np.uint8), contract,
    )
    return prepared


def _image_tensor(
    image_path: Path,
    shape: tuple[int, ...],
    *,
    quantized: bool,
    task: str,
    preprocess_mode: str,
    letterbox_pad: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if Image is None:
        raise RuntimeError("Pillow is required for --image")
    h, w, c, layout = _shape_hwc(shape)
    image = Image.open(image_path).convert("RGB")
    mode = str(preprocess_mode or "auto").lower()
    if mode == "auto":
        mode = "letterbox" if task == "detection" else "resize"
    contract, contract_sha256 = resolve_image_preprocessing_contract(
        task=task, target_hw=(h, w),
    )
    if (
        mode != str(contract["preprocess_mode"])
        or int(letterbox_pad if mode == "letterbox" else 0)
        != int(contract["pad_value"])
    ):
        raise RuntimeError(
            "Hailo image preprocessing differs from the canonical task contract"
        )
    if c != 3:
        raise RuntimeError(
            "canonical RGB preprocessing requires a three-channel runtime input"
        )
    hwc, geometry = prepare_rgb_uint8_image(
        np.asarray(image, dtype=np.uint8), contract,
    )
    if quantized:
        base = hwc
    else:
        base = hwc.astype(np.float32) / 255.0
        if task == "classification" and base.shape[-1] == 3:
            mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
            std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
            base = (base - mean) / std
    if layout == "nchw":
        tensor = np.transpose(base, (2, 0, 1))[None, ...]
    elif layout == "nhwc":
        tensor = base[None, ...]
    elif layout == "chw":
        tensor = np.transpose(base, (2, 0, 1))
    else:
        tensor = base
    tensor = np.ascontiguousarray(tensor.astype(np.uint8 if quantized else np.float32, copy=False))
    prep = {
        "mode": "letterbox_rgb_uint8" if mode == "letterbox" else "resize_rgb_uint8",
        "pad_value": int(letterbox_pad if mode == "letterbox" else 0),
        "rgb": True,
        "color_space": "RGB",
        "ort_model_scale": "imagenet" if task == "classification" else "norm",
        "layout": layout.upper(),
        "quantized_inputs": bool(quantized),
        "preprocessing_contract": contract,
        "preprocessing_contract_sha256": contract_sha256,
        "preprocessing_geometry": geometry,
    }
    return tensor, np.ascontiguousarray(hwc), prep


def _verified_preprocessing_evidence(
    preprocess: Mapping[str, Any] | None,
    *,
    task: str,
    runtime_input_shape: tuple[int, ...],
) -> tuple[dict[str, Any], str]:
    """Verify the exact canonical prepared-pixel contract before inference."""
    if not isinstance(preprocess, Mapping):
        raise RuntimeError("canonical image preprocessing evidence is missing")
    h, w, channels, _layout = _shape_hwc(runtime_input_shape)
    if channels != 3:
        raise RuntimeError(
            "canonical RGB preprocessing requires a three-channel runtime input"
        )
    declared = preprocess.get("preprocessing_contract")
    if not isinstance(declared, Mapping):
        raise RuntimeError("canonical image preprocessing contract is missing")
    contract, contract_sha256 = resolve_image_preprocessing_contract(
        task=task, target_hw=(h, w), declared=declared,
    )
    if (
        str(preprocess.get("preprocessing_contract_sha256") or "")
        != contract_sha256
        or str(preprocess.get("mode") or "").strip().lower()
        != f"{contract['preprocess_mode']}_rgb_uint8"
        or preprocess.get("rgb") is not True
        or str(preprocess.get("color_space") or "").strip().upper()
        != "RGB"
        or int(preprocess.get("pad_value") or 0)
        != int(contract["pad_value"])
    ):
        raise RuntimeError(
            "runtime preprocessing metadata differs from the canonical contract"
        )
    return contract, contract_sha256


def _runtime_preprocessing_binding(
    *, backend: str, task: str, input_name: str,
    input_tensor: np.ndarray, preprocess: Mapping[str, Any],
    runtime_normalization: str,
) -> dict[str, Any]:
    semantic = preprocess.get("preprocessing_contract")
    semantic = dict(semantic) if isinstance(semantic, Mapping) else {}
    semantic_sha = str(
        preprocess.get("preprocessing_contract_sha256") or ""
    ).strip().lower()
    if (
        not semantic
        or preprocessing_contract_sha256(semantic) != semantic_sha
    ):
        raise RuntimeError(
            "runtime preprocessing identity is missing or inconsistent"
        )
    numeric, numeric_sha = runtime_numeric_input_identity(
        backend=backend,
        task=task,
        preprocessing_contract_sha256_value=semantic_sha,
        runtime_input_name=input_name,
        runtime_input_shape=[int(value) for value in input_tensor.shape],
        runtime_input_dtype=str(input_tensor.dtype),
        runtime_input_layout=str(preprocess.get("layout") or ""),
        runtime_color_space="RGB",
        runtime_normalization=runtime_normalization,
    )
    return {
        "runtime_preprocessing_identity": semantic,
        "runtime_preprocessing_sha256": semantic_sha,
        "runtime_numeric_input_identity": numeric,
        "runtime_numeric_input_sha256": numeric_sha,
    }


def _output_contract(
    task: str, outputs: dict[str, Any],
    declared_contract: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    contract = runtime_output_contract(
        task, outputs, raw_fallback=True, declared_contract=declared_contract,
    )
    if contract.get("contract_family") == "decoded_nms":
        contract.update({"e2e_scope": "full_task_pipeline", "postprocess_included": True, "postprocess_location": "accelerator_runtime", "claim_eligible_e2e": True})
    elif (
        str(task or "").strip().lower() == "classification"
        and contract.get("endpoint_contract_complete") is True
        and str(contract.get("contract_family") or "") in {
            "classification_logits", "classification_probabilities",
        }
    ):
        contract.update({"e2e_scope": "full_task_pipeline", "claim_eligible_e2e": True})
    else:
        contract.update({"e2e_scope": "accelerator_only", "postprocess_included": False, "requires_host_decode_nms": str(task or "").strip().lower() == "detection", "host_postprocess_frozen": False, "claim_eligible_e2e": False})
    return contract


def _resolve_declared_output_contract(
    path: str | Path, *, hw_arch: str, model: str, task: str,
) -> dict[str, Any]:
    declaration_path = Path(path).expanduser().resolve()
    payload = json.loads(declaration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("declared output contract must be a JSON object")
    if isinstance(payload.get("contracts"), list):
        # A suite-root container is resolved by exact hardware, model and task.
        # The returned metadata intentionally has no stage on any mismatch, so
        # runtime tensor shape alone cannot promote a decoded-NMS endpoint.
        return load_authoritative_output_contract(
            declaration_path,
            backend=hw_arch,
            model_id=model,
            variant="full",
            task=task,
        )
    # Standalone adjacent declarations cannot prove which suite model, backend
    # and task selected this artifact.  Keep diagnostic metadata, but omit any
    # stage so it cannot attest a scientific endpoint.
    return {
        "contract_resolution_status": "conflict",
        "contract_resolution_reason": "standalone_output_contract_not_authoritative",
        "declaration_source": str(declaration_path),
    }


_HAILO8_YOLOV7_SOURCE_OUTPUTS: tuple[tuple[str, list[int]], ...] = (
    ("output", [1, 3, 80, 80, 85]),
    ("clone_1", [1, 3, 40, 40, 85]),
    ("clone_2", [1, 3, 20, 20, 85]),
)


def _verified_hailo8_yolov7_source_output_shapes(
    declaration: Mapping[str, Any] | None,
    *,
    source_onnx: str | Path,
    hw_arch: str,
    backend_label: str,
    model: str,
    task: str,
) -> dict[str, list[int]]:
    """Project one hash-verified Source-ONNX output attestation.

    Hailo-8 exposes the known YOLOv7 raw heads as three packed HWC streams.
    Their physical names are not graph identities.  This exception is issued
    only for the exact Full backend/model/task and only after checking the
    loader-issued suite declaration, the nested canonical digest, and the
    actual Source-ONNX bytes.  Every mismatch fails before ``prepare`` can open
    a Hailo device.
    """

    is_target = bool(
        str(hw_arch or "").strip().lower().startswith("hailo8")
        and str(backend_label or "").strip() == "native_full_hailo8"
        and str(model or "").strip() == "yolov7_paper"
        and str(task or "").strip().lower() == "detection"
    )
    if not is_target:
        return {}
    contract = dict(declaration or {})
    required_semantics = bool(
        contract.get("contract_resolution_status") == "attested"
        and contract.get("authoritative_output_contract") is True
        and str(contract.get("backend") or "").strip().lower().startswith("hailo8")
        and str(contract.get("model_id") or "").strip() == "yolov7_paper"
        and str(contract.get("variant") or "").strip().lower() == "full"
        and str(contract.get("task") or "").strip().lower() == "detection"
        and str(contract.get("stage") or "").strip().lower() == "raw_head"
        and str(contract.get("contract_family") or "").strip().lower() == "raw_head"
        and str(contract.get("endpoint_mode") or "").strip().lower()
        == "raw_head"
        and contract.get("compiled_artifact_raw_head") is True
        and contract.get("host_tail_required") is True
        and contract.get("postprocessing_required") is True
        and contract.get("requires_external_postprocess") is True
        and contract.get("source_onnx_multiscale_raw_head") is True
        and str(contract.get("raw_endpoint_origin") or "").strip()
        == "source_onnx_graph_outputs"
        and contract.get("full_end_node_names") == []
    )
    if not required_semantics:
        raise RuntimeError(
            "Hailo-8 YOLOv7 Source-ONNX output attestation is not authoritative"
        )

    attestation_raw = contract.get("source_onnx_raw_head_attestation")
    if not isinstance(attestation_raw, Mapping):
        raise RuntimeError("Hailo-8 YOLOv7 Source-ONNX output attestation is missing")
    attestation = dict(attestation_raw)
    digest = str(attestation.get("attestation_sha256") or "").strip().lower()
    outer_digest = str(
        contract.get("source_onnx_raw_head_attestation_sha256") or ""
    ).strip().lower()
    source_digest = str(attestation.get("source_onnx_sha256") or "").strip().lower()
    compiler_digest = str(attestation.get("compiler_onnx_sha256") or "").strip().lower()
    is_sha256 = lambda value: len(value) == 64 and all(  # noqa: E731
        char in "0123456789abcdef" for char in value
    )
    body = {
        key: value for key, value in attestation.items()
        if key != "attestation_sha256"
    }
    if (
        attestation.get("schema")
        != "onnx-splitpoint/hailo-source-raw-head-attestation"
        or attestation.get("schema_version") != 1
        or attestation.get("raw_endpoint_origin")
        != "source_onnx_graph_outputs"
        or not is_sha256(digest)
        or digest != outer_digest
        or digest != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        or not is_sha256(source_digest)
        or not is_sha256(compiler_digest)
        or source_digest
        != str(contract.get("source_onnx_sha256") or "").strip().lower()
        or compiler_digest
        != str(contract.get("compiler_onnx_sha256") or "").strip().lower()
    ):
        raise RuntimeError("Hailo-8 YOLOv7 Source-ONNX attestation hash mismatch")

    expected_outputs = [
        {
            "name": name,
            "element_type": 1,
            "rank": 5,
            "shape": shape,
        }
        for name, shape in _HAILO8_YOLOV7_SOURCE_OUTPUTS
    ]
    if attestation.get("outputs") != expected_outputs:
        raise RuntimeError(
            "Hailo-8 YOLOv7 Source-ONNX output signature mismatch"
        )
    source_path = Path(source_onnx).expanduser()
    if (
        not source_path.is_file()
        or _file_sha256(source_path) != source_digest
    ):
        raise RuntimeError("Hailo-8 YOLOv7 Source-ONNX byte binding mismatch")
    return {name: list(shape) for name, shape in _HAILO8_YOLOV7_SOURCE_OUTPUTS}


def _write_output_dump(
    outputs: dict[str, Any],
    out_dir: Path,
    *,
    backend: str,
    model: str,
    setup_id: str,
    comparison_backend: str,
    task: str,
    image: Path,
    input_hwc: np.ndarray,
    input_tensor: np.ndarray,
    input_name: str,
    preprocess: dict[str, Any],
    declared_output_contract: Mapping[str, Any] | str | None = None,
    frozen_postprocess_contract: Mapping[str, Any] | None = None,
    frozen_postprocess_result: Mapping[str, Any] | None = None,
    frozen_decoded_nms_normalization_contract: (
        Mapping[str, Any] | None
    ) = None,
    frozen_decoded_nms_normalization_result: (
        Mapping[str, Any] | None
    ) = None,
    diagnostic_only: bool = False,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preprocess = dict(preprocess)
    if not isinstance(preprocess.get("preprocessing_contract"), Mapping):
        contract, contract_sha = resolve_image_preprocessing_contract(
            task=task,
            target_hw=(int(input_hwc.shape[0]), int(input_hwc.shape[1])),
        )
        observed_mode = str(preprocess.get("mode") or "").strip().lower()
        if observed_mode != f"{contract['preprocess_mode']}_rgb_uint8":
            raise RuntimeError(
                "runtime preprocessing mode differs from the canonical contract"
            )
        preprocess.update({
            "pad_value": int(contract["pad_value"]),
            "rgb": True,
            "color_space": "RGB",
            "layout": _shape_hwc(
                tuple(int(value) for value in input_tensor.shape)
            )[3].upper(),
            "quantized_inputs": np.asarray(input_tensor).dtype == np.uint8,
            "preprocessing_contract": contract,
            "preprocessing_contract_sha256": contract_sha,
        })
    input_file = out_dir / "input_rgb_uint8.bin"
    input_file.write_bytes(np.ascontiguousarray(input_hwc).tobytes())
    runtime_input_file = out_dir / "runtime_input.bin"
    runtime_input_array = np.ascontiguousarray(input_tensor)
    runtime_input_file.write_bytes(runtime_input_array.tobytes())
    input_manifest = out_dir / "native_full_input_manifest.json"
    runtime_normalization = (
        "embedded_hailo_quantization"
        if preprocess.get("quantized_inputs") is True
        else "imagenet_mean_std"
        if str(task or "").strip().lower() == "classification"
        else "divide_255"
    )
    runtime_binding = _runtime_preprocessing_binding(
        backend=backend,
        task=task,
        input_name=input_name,
        input_tensor=runtime_input_array,
        preprocess=preprocess,
        runtime_normalization=runtime_normalization,
    )
    input_payload = {
        "schema": "onnx-splitpoint/native-full-input-dump", "schema_version": 2,
        "backend": backend, "model": str(model or ""),
        "setup_id": str(setup_id or ""),
        "comparison_backend": str(comparison_backend or ""),
        "case": "full", "task": str(task or "").strip().lower(),
        "image": str(image), "image_sha256": _file_sha256(image),
        "input_image": str(image), "input_image_sha256": _file_sha256(image),
        "input_dump": str(input_file),
        "input_dump_sha256": _file_sha256(input_file),
        "input_dump_bytes": int(input_file.stat().st_size),
        "input_shape_hwc": [int(x) for x in input_hwc.shape],
        "runtime_input_name": input_name, "runtime_input_shape": [int(x) for x in input_tensor.shape],
        "runtime_input_dtype": str(input_tensor.dtype), "preprocess": preprocess,
        "runtime_input_file": str(runtime_input_file),
        "runtime_input_sha256": _file_sha256(runtime_input_file),
        "runtime_input_bytes": int(runtime_input_array.nbytes),
        "runtime_input_layout": str(preprocess.get("layout") or ""),
        "runtime_preprocess_mode": str(preprocess.get("mode") or ""),
        "runtime_normalization": runtime_normalization,
        "runtime_color_space": "RGB",
        "preprocessing_contract": dict(
            preprocess.get("preprocessing_contract") or {}
        ),
        "preprocessing_contract_sha256": str(
            preprocess.get("preprocessing_contract_sha256") or ""
        ),
        "preprocessing_geometry": dict(
            preprocess.get("preprocessing_geometry") or {}
        ),
        **runtime_binding,
    }
    if frozen_decoded_nms_normalization_contract:
        input_payload.update({
            "original_image_wh": list(
                frozen_decoded_nms_normalization_contract.get(
                    "original_wh"
                ) or []
            ),
            "letterbox_geometry_contract": dict(
                frozen_decoded_nms_normalization_contract.get(
                    "letterbox_geometry_contract"
                ) or {}
            ),
            "letterbox_geometry_contract_sha256": str(
                frozen_decoded_nms_normalization_contract.get(
                    "letterbox_geometry_contract_sha256"
                ) or ""
            ),
        })
    input_manifest.write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    entries=[]
    for idx,(name,value) in enumerate(outputs.items()):
        arr=np.ascontiguousarray(np.asarray(value))
        safe=''.join(c if c.isalnum() or c in '._-' else '_' for c in str(name))
        fname=f"output_{idx:02d}_{safe or idx}.bin"
        output_file = out_dir / fname
        output_file.write_bytes(arr.tobytes())
        entries.append({
            "name": str(name), "file": fname, "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape], "bytes": int(arr.nbytes),
            "sha256": _file_sha256(output_file),
        })
    manifest = out_dir / "native_full_outputs_manifest.json"
    output_contract = _output_contract(task, outputs, declared_output_contract)
    if frozen_postprocess_contract:
        verified_postprocess = verify_frozen_postprocess_contract(
            frozen_postprocess_contract, outputs=outputs,
        )
        postprocess_result = dict(frozen_postprocess_result or {})
        if (
            postprocess_result.get("contract_family") != "decoded_nms"
            or str(postprocess_result.get("postprocess_contract_sha256") or "")
            != str(verified_postprocess["contract_sha256"])
        ):
            raise RuntimeError("frozen Native Full postprocess result is missing or inconsistent")
        output_contract.update({
            "e2e_scope": "full_task_pipeline",
            "postprocess_included": True,
            "postprocess_location": "serialized_host_tail_inside_measured_interval",
            "requires_host_decode_nms": True,
            "host_postprocess_frozen": True,
            "claim_eligible_e2e": True,
            "frozen_host_postprocess_contract": verified_postprocess,
            "frozen_host_postprocess_contract_sha256": str(verified_postprocess["contract_sha256"]),
            "frozen_host_postprocess_result": postprocess_result,
            "decoder_id": str(verified_postprocess["decoder_id"]),
            "decoder_contract_sha256": str(verified_postprocess["contract_sha256"]),
            "nms_contract_sha256": str(verified_postprocess["contract_sha256"]),
        })
    if frozen_decoded_nms_normalization_contract:
        source_attestation = output_contract.get(
            "output_endpoint_attestation"
        )
        verified_direct = verify_frozen_decoded_nms_normalization_contract(
            frozen_decoded_nms_normalization_contract,
            outputs=outputs,
            source_output_endpoint_attestation=(
                source_attestation
                if isinstance(source_attestation, Mapping) else None
            ),
        )
        direct_result = dict(
            frozen_decoded_nms_normalization_result or {}
        )
        if (
            direct_result.get("contract_family") != "decoded_nms"
            or direct_result.get("coordinate_space")
            != "original_image_xyxy_pixels"
            or str(
                direct_result.get("normalization_contract_sha256") or ""
            ) != str(verified_direct["contract_sha256"])
        ):
            raise RuntimeError(
                "frozen Direct-BN6 normalization result is missing or inconsistent"
            )
        output_contract.update({
            "e2e_scope": "full_task_pipeline",
            "postprocess_included": True,
            "postprocess_location": (
                "serialized_host_normalization_inside_measured_interval"
            ),
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
    if diagnostic_only:
        output_contract.update({
            "diagnostic_only": True,
            "claim_eligible": False,
            "claim_eligible_e2e": False,
        })
    payload = {
        "schema":"onnx-splitpoint/runner-output-dump","schema_version":4,
        "producer":backend,"backend":backend,"case":"full","execution_mode":"native_full_baseline",
        "model":str(model or ""),"setup_id":str(setup_id or ""),
        "comparison_backend":str(comparison_backend or ""),
        "input_image":str(image),"input_image_sha256":_file_sha256(image),
        "input_manifest":str(input_manifest),"boundary_manifest":str(input_manifest),"native_boundary_manifest":str(input_manifest),
        "provenance":{"image":str(image),"image_sha256":_file_sha256(image),"image_source":"exact_file"},
        "authoritative_output_contract_resolution": dict(declared_output_contract or {}),
        **output_contract,"outputs":entries,
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, input_manifest


def _summarize_outputs(outputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, value in outputs.items():
        arr = np.asarray(value)
        summary[str(name)] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "min": float(np.min(arr)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test a precompiled Hailo-10H HEF through the tool's HailoBackend."
    )
    parser.add_argument(
        "--hef",
        default="/usr/local/hailo/resources/models/hailo10h/yolov6n.hef",
        help="Path to a Hailo-10H-compatible HEF.",
    )
    parser.add_argument("--onnx", default="", help="Optional ONNX model path for canonical IO-name mapping.")
    parser.add_argument("--hw-arch", default="hailo10h", help="Hailo architecture label used for metadata.")
    parser.add_argument(
        "--runtime-api",
        default="auto",
        choices=["auto", "infer_model", "vstreams"],
        help="Hailo runtime API. auto uses InferModel for Hailo-10/Hailo-15.",
    )
    parser.add_argument("--artifacts-dir", default="/tmp/splitpoint_hailo10_smoke", help="Temporary artifact directory.")
    parser.add_argument("--warmup", type=_positive_int, default=2, help="Warmup inferences before timing.")
    parser.add_argument("--runs", type=_positive_int, default=10, help="Timed inference runs.")
    parser.add_argument(
        "--throughput-mode",
        action="store_true",
        help="Measure pipelined async throughput instead of synchronous per-frame latency.",
    )
    parser.add_argument(
        "--counted-hotloop-only", action="store_true",
        help="Emit count/performance evidence only; never run an extra output-sampling inference after the counted hotloop.",
    )
    parser.add_argument(
        "--frames",
        type=_positive_int,
        default=500,
        help="Measured frames in throughput mode.",
    )
    parser.add_argument(
        "--duration-s", type=float, default=0.0,
        help="Minimum measured hotloop duration. Frames remain a minimum work budget.",
    )
    parser.add_argument(
        "--inflight",
        type=_positive_int,
        default=8,
        help="Number of InferModel jobs/bindings allowed in flight in throughput mode.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random input seed.")
    parser.add_argument("--image", default="", help="Exact semantic input image. When set, random input is not used for the first runtime input.")
    parser.add_argument("--preverified-input-image-sha256", default="", help="Image digest already verified by the pre-sampling energy preflight; avoids hashing the image in the measured hotloop.")
    parser.add_argument("--runtime-input-bin", default="", help="Exact runtime tensor sealed and verified before the energy command.")
    parser.add_argument("--runtime-input-contract-json", default="", help="Preflight-validated name/shape/dtype/byte-count contract for --runtime-input-bin.")
    parser.add_argument("--preverified-runtime-input-sha256", default="", help="Digest verified by the pre-sampling energy preflight; it is compared with the sealed contract but never recalculated here.")
    parser.add_argument("--canonical-input-slot-names-json", default="", help="Canonical input-slot names from the successful Full run.")
    parser.add_argument("--canonical-output-slot-names-json", default="", help="Canonical output-slot names from the successful Full run.")
    parser.add_argument("--task", default="", choices=["", "classification", "detection"], help="Task used for preprocessing and output-contract metadata.")
    parser.add_argument("--preprocess-mode", default="auto", choices=["auto", "resize", "letterbox"], help="Exact image preprocessing used for the semantic dump.")
    parser.add_argument("--letterbox-pad-value", type=int, default=114)
    parser.add_argument("--dump-outputs", action="store_true", help="Write a standard Native Full semantic output dump.")
    parser.add_argument("--dump-dir", default="", help="Directory for Native Full input/output dump manifests.")
    parser.add_argument("--backend-label", default="native_full_hailo10h")
    parser.add_argument("--model", default="")
    parser.add_argument("--setup-id", default="")
    parser.add_argument("--comparison-backend", default="")
    parser.add_argument(
        "--quantized-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send UINT8 input tensors to HailoRT. Model-zoo detection HEFs usually expect this.",
    )
    parser.add_argument(
        "--quantized-outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request quantized output tensors from HailoRT.",
    )
    parser.add_argument(
        "--persistent-activation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the Hailo network group activated across timed runs.",
    )
    parser.add_argument(
        "--hotloop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse InferModel bindings and IO buffers across timed runs.",
    )
    parser.add_argument(
        "--copy-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy outputs before returning them from reusable hotloop buffers.",
    )
    parser.add_argument(
        "--copy-inputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy input data into ring-buffer slots for every submitted frame in throughput mode.",
    )
    parser.add_argument("--json-out", default="", help="Optional path for a JSON smoke-test report.")
    parser.add_argument("--declared-output-contract-json", default="", help="Optional producer/export endpoint declaration JSON used for fail-closed endpoint attestation.")
    parser.add_argument("--frozen-postprocess-contract-json", default="", help="Sealed raw-head decoder/NMS contract for an energy replay.")
    parser.add_argument(
        "--frozen-decoded-nms-normalization-contract-json",
        default="",
        help=(
            "Sealed Direct-BN6 normalization contract for an energy "
            "replay. Mutually exclusive with --frozen-postprocess-contract-json."
        ),
    )
    parser.add_argument("--original-image-wh-json", default="", help="Preverified original image [width,height] used only to invert letterboxing.")
    parser.add_argument(
        "--diagnostic-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Suppress claim eligibility in every emitted child artifact.",
    )
    args = parser.parse_args()

    claim_performance_run = bool(
        args.throughput_mode
        and str(args.backend_label or "").startswith("native_full_")
        and not bool(args.counted_hotloop_only)
    )
    if claim_performance_run and not bool(args.copy_outputs):
        raise RuntimeError(
            "Native Full Hailo claim runs require --copy-outputs so every backend materializes equivalent outputs"
        )
    declared_output_contract: dict[str, Any] | None = None
    if str(args.declared_output_contract_json or "").strip():
        declared_output_contract = _resolve_declared_output_contract(
            args.declared_output_contract_json,
            hw_arch=str(args.hw_arch or ""),
            model=str(args.model or ""),
            task=str(args.task or ""),
        )

    hef_path = Path(args.hef).expanduser()
    if not hef_path.is_absolute():
        hef_path = hef_path.resolve()
    if not hef_path.is_file():
        raise FileNotFoundError(f"HEF not found: {hef_path}")

    bound_flags = (
        str(args.runtime_input_bin or "").strip(),
        str(args.runtime_input_contract_json or "").strip(),
        str(args.preverified_runtime_input_sha256 or "").strip(),
        str(args.canonical_input_slot_names_json or "").strip(),
        str(args.canonical_output_slot_names_json or "").strip(),
    )
    bound_runtime_mode = any(bound_flags)
    bound_runtime_contract: dict[str, Any] | None = None
    canonical_input_names: list[str] = []
    canonical_output_names: list[str] = []
    attested_source_output_shapes: dict[str, list[int]] = {}
    if bound_runtime_mode:
        if not all(bound_flags):
            raise RuntimeError("all preverified runtime-input binding arguments are required together")
        if args.onnx:
            raise RuntimeError("--onnx is forbidden for a preverified runtime-input energy hotloop")
        if bool(args.dump_outputs):
            raise RuntimeError("--dump-outputs is forbidden for a preverified runtime-input energy hotloop")
        if not bool(args.throughput_mode) or not bool(args.counted_hotloop_only) or int(args.warmup) != 0:
            raise RuntimeError(
                "preverified runtime-input mode requires --throughput-mode, --counted-hotloop-only, and --warmup 0"
            )
        bound_runtime_contract = _runtime_input_contract(args.runtime_input_contract_json)
        canonical_input_names = _json_string_list(
            args.canonical_input_slot_names_json, field="canonical input-slot names",
        )
        canonical_output_names = _json_string_list(
            args.canonical_output_slot_names_json, field="canonical output-slot names",
        )
        if canonical_input_names != [str(bound_runtime_contract["runtime_input_name"])]:
            raise RuntimeError("canonical input-slot names do not match the sealed runtime input")
        image_digest = str(args.preverified_input_image_sha256 or "").strip().lower()
        if str(args.image or "").strip() and (
            len(image_digest) != 64 or any(ch not in "0123456789abcdef" for ch in image_digest)
        ):
            raise RuntimeError("bound runtime-input mode requires a valid preverified image digest when --image is present")
    else:
        attested_source_output_shapes = (
            _verified_hailo8_yolov7_source_output_shapes(
                declared_output_contract,
                source_onnx=str(args.onnx or ""),
                hw_arch=str(args.hw_arch or ""),
                backend_label=str(args.backend_label or ""),
                model=str(args.model or ""),
                task=str(args.task or ""),
            )
        )

    options: dict[str, Any] = {
        "hef_path": str(hef_path),
        "hw_arch": str(args.hw_arch),
        "quantized_inputs": bool(args.quantized_inputs),
        "quantized_outputs": bool(args.quantized_outputs),
        "persistent_activation": bool(args.persistent_activation),
        "runtime_api": str(args.runtime_api),
        "hotloop": bool(args.hotloop),
        "copy_outputs": bool(args.copy_outputs),
    }
    if bound_runtime_mode:
        options["canonical_input_slot_names"] = canonical_input_names
        options["canonical_output_slot_names"] = canonical_output_names
    if args.onnx:
        options["onnx_model_path"] = str(Path(args.onnx).expanduser())
    if attested_source_output_shapes:
        canonical_output_names = list(attested_source_output_shapes)
        options["canonical_output_slot_names"] = canonical_output_names
        options["attested_source_output_shapes"] = attested_source_output_shapes

    backend = HailoBackend(strict=True, **options)
    prepared = None
    try:
        prepared = backend.prepare(
            RunCfg(model_path=hef_path, options=options),
            Path(args.artifacts_dir).expanduser(),
        )
        prep = prepared.handle

        input_shapes = dict(prep.runtime_input_shapes or prep.input_shapes)
        exact_image = Path(args.image).expanduser().resolve() if str(args.image or "").strip() else None
        input_hwc = None
        input_preprocess = None
        inputs: dict[str, np.ndarray] = {}
        if bound_runtime_mode:
            assert bound_runtime_contract is not None
            inputs = _load_preverified_runtime_input(
                args.runtime_input_bin,
                bound_runtime_contract,
                preverified_sha256=str(args.preverified_runtime_input_sha256),
                prepared_input_names=list(prepared.input_names),
                prepared_runtime_shapes=input_shapes,
                quantized_inputs=bool(args.quantized_inputs),
            )
            input_preprocess = dict(bound_runtime_contract.get("preprocess") or {})
        else:
            for idx, name in enumerate(prepared.input_names):
                shape = tuple(input_shapes.get(name, ()))
                if idx == 0 and exact_image is not None:
                    tensor, input_hwc, input_preprocess = _image_tensor(
                        exact_image, shape, quantized=bool(args.quantized_inputs), task=str(args.task or ""),
                        preprocess_mode=str(args.preprocess_mode), letterbox_pad=int(args.letterbox_pad_value),
                    )
                    inputs[name] = tensor
                else:
                    inputs[name] = _input_tensor(shape, quantized=bool(args.quantized_inputs), seed=args.seed + idx)

        image_task = str(args.task or "").strip().lower()
        native_full_detection = bool(
            image_task == "detection"
            and str(args.backend_label or "").startswith("native_full_")
        )
        preprocessing_contract: dict[str, Any] = {}
        preprocessing_contract_digest = ""
        if image_task in {"classification", "detection"} and input_preprocess:
            first_runtime_input = next(iter(inputs.values()))
            (
                preprocessing_contract,
                preprocessing_contract_digest,
            ) = _verified_preprocessing_evidence(
                input_preprocess,
                task=image_task,
                runtime_input_shape=tuple(first_runtime_input.shape),
            )
        elif native_full_detection:
            raise RuntimeError(
                "Native Full Hailo detection requires canonical prepared-pixel evidence"
            )

        print("[hailo10-smoke] HEF:", hef_path)
        print("[hailo10-smoke] inputs:", json.dumps({k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()}, sort_keys=True))
        print("[hailo10-smoke] outputs:", json.dumps({k: list(v) for k, v in prep.runtime_output_shapes.items()}, sort_keys=True))
        print(
            "[hailo10-smoke] quantized_inputs=%s quantized_outputs=%s persistent_activation=%s runtime_api=%s hotloop=%s copy_outputs=%s"
            % (
                bool(args.quantized_inputs),
                bool(args.quantized_outputs),
                bool(args.persistent_activation),
                str(args.runtime_api),
                bool(args.hotloop),
                bool(args.copy_outputs),
            )
        )

        original_wh: list[int] | None = None
        if str(args.original_image_wh_json or "").strip():
            parsed_wh = json.loads(str(args.original_image_wh_json))
            if (
                not isinstance(parsed_wh, list) or len(parsed_wh) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in parsed_wh)
            ):
                raise RuntimeError("--original-image-wh-json must be [positive_width,positive_height]")
            original_wh = [int(value) for value in parsed_wh]
        elif exact_image is not None and not bound_runtime_mode:
            if Image is None:
                raise RuntimeError("Pillow is required to bind Native Full detection image geometry")
            with Image.open(exact_image) as original_image:
                original_wh = [int(original_image.size[0]), int(original_image.size[1])]

        frozen_postprocess_contract: dict[str, Any] = {}
        frozen_postprocessor: FrozenDetectionPostprocessor | None = None
        frozen_decoded_nms_normalization_contract: dict[str, Any] = {}
        decoded_nms_normalizer: FrozenDecodedNmsPostprocessor | None = None
        frozen_postprocess_probe_outputs: dict[str, Any] = {}
        probe_contract: dict[str, Any] = {}
        frozen_raw_json = str(
            args.frozen_postprocess_contract_json or ""
        ).strip()
        frozen_direct_json = str(
            args.frozen_decoded_nms_normalization_contract_json or ""
        ).strip()
        if frozen_raw_json and frozen_direct_json:
            raise RuntimeError(
                "Raw-head and Direct-BN6 replay contracts are mutually exclusive"
            )
        if frozen_raw_json:
            raw_contract = json.loads(str(args.frozen_postprocess_contract_json))
            frozen_postprocess_contract = verify_frozen_postprocess_contract(raw_contract)
            frozen_postprocessor = FrozenDetectionPostprocessor(frozen_postprocess_contract)
        elif frozen_direct_json:
            raw_contract = json.loads(frozen_direct_json)
            frozen_decoded_nms_normalization_contract = (
                verify_frozen_decoded_nms_normalization_contract(
                    raw_contract
                )
            )
            if original_wh != list(
                frozen_decoded_nms_normalization_contract.get(
                    "original_wh"
                ) or []
            ):
                raise RuntimeError(
                    "Direct-BN6 replay original image geometry mismatch"
                )
            decoded_nms_normalizer = FrozenDecodedNmsPostprocessor(
                frozen_decoded_nms_normalization_contract
            )
        elif (
            str(args.task or "").strip().lower() == "detection"
            and str(args.backend_label or "").startswith("native_full_")
        ):
            # One untimed structural probe binds the physical endpoint.  Both a
            # raw head and Direct-BN6 still need one canonical completion step
            # inside every measured iteration.
            probe = backend.run(prepared, inputs)
            frozen_postprocess_probe_outputs = dict(probe.outputs)
            probe_contract = _output_contract(
                str(args.task or ""), frozen_postprocess_probe_outputs,
                declared_output_contract,
            )
            probe_family = str(
                probe_contract.get("contract_family") or ""
            ).strip()
            if probe_family == "raw_head":
                if original_wh is None:
                    raise RuntimeError("raw-head Native Full detection requires preverified original image geometry")
                first_input = next(iter(inputs.values()))
                if first_input.ndim < 3:
                    raise RuntimeError("raw-head Native Full detection input geometry is unavailable")
                input_h, input_w, _channels, _layout = _shape_hwc(
                    tuple(first_input.shape)
                )
                input_hw = [input_h, input_w]
                frozen_postprocess_contract = build_frozen_postprocess_contract(
                    model_id=str(args.model or ""),
                    model_sha256=_bound_model_sha256(
                        declared_output_contract, probe_contract,
                    ),
                    outputs=frozen_postprocess_probe_outputs,
                    input_hw=input_hw,
                    original_wh=original_wh,
                )
                frozen_postprocessor = FrozenDetectionPostprocessor(frozen_postprocess_contract)
            elif probe_family == "decoded_nms":
                if original_wh is None:
                    raise RuntimeError(
                        "Direct-BN6 Native Full detection requires preverified original image geometry"
                    )
                first_input = next(iter(inputs.values()))
                input_h, input_w, _channels, _layout = _shape_hwc(
                    tuple(first_input.shape)
                )
                source_attestation = probe_contract.get(
                    "output_endpoint_attestation"
                )
                declared_source = (
                    source_attestation.get("declared_contract")
                    if isinstance(source_attestation, Mapping) else None
                )
                source_coordinate_space = str(
                    (
                        declared_source.get("source_coordinate_space")
                        if isinstance(declared_source, Mapping) else ""
                    ) or ""
                ).strip()
                frozen_decoded_nms_normalization_contract = (
                    build_frozen_decoded_nms_normalization_contract(
                        source_completed=True,
                        model_id=str(args.model or ""),
                        outputs=frozen_postprocess_probe_outputs,
                        input_hw=[input_h, input_w],
                        original_wh=original_wh,
                        preprocess=dict(input_preprocess or {}),
                        source_coordinate_space=source_coordinate_space,
                        source_endpoint_contract_hash=str(
                            probe_contract.get("endpoint_contract_hash") or ""
                        ),
                        source_output_endpoint_attestation=(
                            source_attestation
                            if isinstance(source_attestation, Mapping) else {}
                        ),
                    )
                )
                decoded_nms_normalizer = FrozenDecodedNmsPostprocessor(
                    frozen_decoded_nms_normalization_contract
                )
            else:
                raise RuntimeError(
                    "Native Full Hailo detection has no authoritative raw-head or decoded-NMS endpoint"
                )

        classification_processor = ClassificationCompletion() if str(args.task) == "classification" else None
        completion_processor = frozen_postprocessor or decoded_nms_normalizer or classification_processor
        if (frozen_postprocessor or decoded_nms_normalizer) is not None and original_wh is None:
            raise RuntimeError("frozen Native Full postprocess requires preverified original image geometry")

        def _timed_postprocess(outputs: dict[str, np.ndarray]) -> dict[str, Any]:
            if classification_processor is not None:
                return classification_processor.process(outputs)
            if frozen_postprocessor is not None:
                canonical_outputs = _canonicalize_frozen_postprocess_outputs(
                    outputs, frozen_postprocess_contract,
                )
                return frozen_postprocessor.process(
                    canonical_outputs, original_wh=original_wh,
                )
            if decoded_nms_normalizer is not None:
                return decoded_nms_normalizer.process(
                    outputs, original_wh=original_wh,
                )
            raise RuntimeError("Native Full completion processor is missing")

        last_outputs: dict[str, Any] = {}
        timings_ms: list[float] = []
        request_latency = RequestLatency(
            int(args.frames) if args.throughput_mode else int(args.runs),
            task_complete=completion_processor is not None,
            start_anchor="hailo_full:before_backend_run",
            end_anchor="hailo_full:after_postprocess_or_host_outputs", enabled=not args.duration_s)
        throughput_result: dict[str, Any] | None = None
        if bool(args.throughput_mode):
            if hasattr(prep.session, "benchmark_throughput"):
                throughput_result = prep.session.benchmark_throughput(
                    inputs,
                    frames=int(args.frames),
                    inflight=max(1, int(args.inflight)),
                    warmup_frames=int(args.warmup),
                    copy_inputs=bool(args.copy_inputs),
                    duration_s=max(0.0, float(args.duration_s or 0.0)),
                    postprocess_callback=(
                        _timed_postprocess if completion_processor is not None else None
                    ),
                )
                requested_frames = int(throughput_result.get("requested_frames") or args.frames)
                completed_frames = int(throughput_result.get("completed_frames") or 0)
                if (
                    completed_frames < requested_frames
                    or str(throughput_result.get("completed_work_units_status") or "") != "exact_runtime_counter"
                    or (
                        float(args.duration_s or 0.0) > 0.0
                        and throughput_result.get("minimum_duration_satisfied") is not True
                    )
                ):
                    raise RuntimeError(
                        f"Hailo throughput completion mismatch: requested={requested_frames} observed={completed_frames}"
                    )
                if not bool(args.counted_hotloop_only):
                    out = backend.run(prepared, inputs)
                    last_outputs = dict(out.outputs)
            else:
                # Hailo-8 VStreams are synchronous in the common HailoRT API.
                # Measure a steady-state native loop instead of failing merely
                # because the InferModel-only async helper is unavailable.
                for _ in range(int(args.warmup)):
                    out = backend.run(prepared, inputs)
                    last_outputs = dict(out.outputs)
                    if completion_processor is not None:
                        _timed_postprocess(last_outputs)
                t0 = time.perf_counter()
                completed_frames = 0
                measured_postprocess_before = (
                    int(completion_processor.completed_count)
                    if completion_processor is not None else 0
                )
                target_duration_s = max(0.0, float(args.duration_s or 0.0))
                while (
                    completed_frames < int(args.frames)
                    or time.perf_counter() - t0 < target_duration_s
                ):
                    request_latency.start(completed_frames)
                    out = backend.run(prepared, inputs)
                    last_outputs = dict(out.outputs)
                    if completion_processor is not None:
                        _timed_postprocess(last_outputs)
                    request_latency.complete(completed_frames)
                    completed_frames += 1
                elapsed_s = max(0.0, time.perf_counter() - t0)
                if completed_frames < int(args.frames):
                    raise RuntimeError(
                        f"Hailo VStreams completion mismatch: requested={int(args.frames)} observed={completed_frames}"
                    )
                fps = (float(completed_frames) / elapsed_s) if completed_frames > 0 and elapsed_s > 0.0 else 0.0
                throughput_result = {
                    "request_latency": request_latency.report(),
                    "frames": int(completed_frames),
                    "requested_frames": int(args.frames),
                    "minimum_requested_frames": int(args.frames),
                    "completed_frames": int(completed_frames),
                    "completed_work_units": int(completed_frames),
                    "completed_work_units_source": (
                        "synchronous_vstreams_frozen_postprocess_success_counter"
                        if completion_processor is not None
                        else "synchronous_vstreams_successful_return_counter"
                    ),
                    "completed_work_units_status": "exact_runtime_counter",
                    "postprocess_included": completion_processor is not None,
                    "postprocess_completed_frames": (
                        int(completion_processor.completed_count) - measured_postprocess_before
                        if completion_processor is not None else 0
                    ),
                    "postprocess_completion_status": (
                        "exact_runtime_counter" if completion_processor is not None
                        else "not_requested"
                    ),
                    "warmup_frames": int(args.warmup),
                    "warmup_completed_frames": int(args.warmup),
                    "inflight": 1,
                    "copy_inputs": bool(args.copy_inputs),
                    "elapsed_s": elapsed_s,
                    "requested_duration_s": target_duration_s,
                    "minimum_duration_satisfied": bool(
                        target_duration_s <= 0.0 or elapsed_s >= target_duration_s
                    ),
                    "measurement_control": (
                        "minimum_frames_and_duration"
                        if target_duration_s > 0.0 else "exact_frames"
                    ),
                    "fps": fps,
                    "completion_interval_mean_ms": (1000.0 / fps) if fps > 0.0 else 0.0,
                    "completion_interval_semantics": "reciprocal_steady_state_observed_completion_throughput",
                    "latency_measured": False,
                    "measurement_mode": "synchronous_vstreams_native_loop",
                }
            print(
                "[hailo10-smoke] throughput fps=%.2f completion_interval_mean_ms=%.3f elapsed_s=%.3f completed/requested=%d/%d inflight=%d warmup=%d copy_inputs=%s mode=%s"
                % (
                    float(throughput_result.get("fps", 0.0)),
                    float(throughput_result.get("completion_interval_mean_ms", 0.0)),
                    float(throughput_result.get("elapsed_s", 0.0)),
                    int(throughput_result.get("completed_frames", 0)),
                    int(throughput_result.get("requested_frames", 0)),
                    int(throughput_result.get("inflight", 0)),
                    int(throughput_result.get("warmup_frames", 0)),
                    bool(throughput_result.get("copy_inputs", True)),
                    str(throughput_result.get("measurement_mode") or "async_infer_model"),
                )
            )
        else:
            for _ in range(int(args.warmup)):
                out = backend.run(prepared, inputs)
                last_outputs = dict(out.outputs)
                if completion_processor is not None:
                    _timed_postprocess(last_outputs)
            measured_postprocess_before = (
                int(completion_processor.completed_count)
                if completion_processor is not None else 0
            )
            for request_id in range(int(args.runs)):
                request_latency.start(request_id)
                iteration_started = time.perf_counter()
                out = backend.run(prepared, inputs)
                last_outputs = dict(out.outputs)
                if completion_processor is not None:
                    _timed_postprocess(last_outputs)
                    timings_ms.append(
                        (time.perf_counter() - iteration_started) * 1000.0
                    )
                else:
                    timings_ms.append(
                        float(out.metrics.get("infer_ms", 0.0))
                    )

                request_latency.complete(request_id)

            if timings_ms:
                print(
                    "[hailo10-smoke] infer_ms mean=%.3f min=%.3f max=%.3f runs=%d"
                    % (statistics.fmean(timings_ms), min(timings_ms), max(timings_ms), len(timings_ms))
                )
            else:
                print("[hailo10-smoke] no timed runs requested")

        if (
            bool(args.dump_outputs)
            and not last_outputs
            and frozen_postprocess_probe_outputs
        ):
            last_outputs = dict(frozen_postprocess_probe_outputs)
        output_summary = _summarize_outputs(last_outputs)
        print("[hailo10-smoke] output_summary:", json.dumps(output_summary, sort_keys=True))
        output_manifest = ""
        input_manifest = ""
        if bool(args.dump_outputs):
            if exact_image is None or input_hwc is None or input_preprocess is None:
                raise RuntimeError("--dump-outputs requires --image and a resolvable image input")
            dump_dir = Path(args.dump_dir).expanduser().resolve() if str(args.dump_dir or "").strip() else Path(args.artifacts_dir).expanduser().resolve() / "native_full_outputs"
            # The async throughput helper may obtain a fresh output after the
            # measured loop.  Decode that exact dumped tensor set with a
            # separate counter so the dump result cannot accidentally refer
            # to the previous measured callback output.
            dump_postprocess_result: dict[str, Any] = {}
            dump_direct_normalization_result: dict[str, Any] = {}
            if frozen_postprocess_contract:
                dump_postprocessor = FrozenDetectionPostprocessor(
                    frozen_postprocess_contract
                )
                dump_postprocess_result = dump_postprocessor.process(
                    last_outputs, original_wh=original_wh,
                )
            elif frozen_decoded_nms_normalization_contract:
                dump_normalizer = FrozenDecodedNmsPostprocessor(
                    frozen_decoded_nms_normalization_contract
                )
                dump_direct_normalization_result = dump_normalizer.process(
                    last_outputs, original_wh=original_wh,
                )
            om, im = _write_output_dump(
                last_outputs, dump_dir, backend=str(args.backend_label), model=str(args.model or ""),
                setup_id=str(args.setup_id or ""), comparison_backend=str(args.comparison_backend or ""),
                task=str(args.task or ""), image=exact_image,
                input_hwc=input_hwc, input_tensor=next(iter(inputs.values())), input_name=str(next(iter(inputs.keys()))), preprocess=dict(input_preprocess),
                declared_output_contract=declared_output_contract,
                frozen_postprocess_contract=frozen_postprocess_contract,
                frozen_postprocess_result=dump_postprocess_result,
                frozen_decoded_nms_normalization_contract=(
                    frozen_decoded_nms_normalization_contract
                ),
                frozen_decoded_nms_normalization_result=(
                    dump_direct_normalization_result
                ),
                diagnostic_only=bool(args.diagnostic_only),
            )
            output_manifest, input_manifest = str(om), str(im)
            print(f"[hailo10-smoke] semantic output manifest: {output_manifest}", flush=True)

        if native_full_detection and not str(args.json_out or "").strip():
            raise RuntimeError(
                "Native Full Hailo detection requires --json-out for Completed-v2 persistence"
            )
        if args.json_out:
            json_path = Path(args.json_out).expanduser()
            if not json_path.is_absolute():
                json_path = json_path.resolve()
            input_image_sha256 = str(args.preverified_input_image_sha256 or "").strip().lower()
            if exact_image is not None and not input_image_sha256 and not bound_runtime_mode:
                input_image_sha256 = _file_sha256(exact_image)
            runtime_input_file = (
                Path(args.runtime_input_bin).expanduser().resolve()
                if bound_runtime_mode else
                Path(input_manifest).resolve().parent / "runtime_input.bin"
                if input_manifest else None
            )
            runtime_input_evidence = _runtime_input_evidence(
                inputs=inputs,
                prepared_input_names=[str(value) for value in prepared.input_names],
                prepared_runtime_shapes=input_shapes,
                quantized_inputs=bool(args.quantized_inputs),
                bound_runtime_contract=bound_runtime_contract,
                preverified_sha256=str(args.preverified_runtime_input_sha256),
                runtime_input_file=runtime_input_file,
            )
            if bound_runtime_mode:
                runtime_normalization = str(
                    (input_preprocess or {}).get("normalization")
                    or (input_preprocess or {}).get("ort_model_scale")
                    or "preflight_bound_preprocessed_tensor"
                )
            else:
                runtime_normalization = (
                    "embedded_hailo_quantization" if bool(args.quantized_inputs)
                    else "imagenet_mean_std" if str(args.task or "") == "classification"
                    else "divide_255"
                )
            runtime_preprocessing_binding: dict[str, Any] = {}
            if preprocessing_contract and input_preprocess:
                runtime_preprocessing_binding = (
                    _runtime_preprocessing_binding(
                        backend=str(args.backend_label or ""),
                        task=str(args.task or ""),
                        input_name=str(
                            runtime_input_evidence["runtime_input_name"]
                        ),
                        input_tensor=next(iter(inputs.values())),
                        preprocess=input_preprocess,
                        runtime_normalization=runtime_normalization,
                    )
                )
            measured_completed_frames = (
                int(
                    (throughput_result or {}).get(
                        "completed_frames"
                    ) or 0
                )
                if bool(args.throughput_mode)
                else len(timings_ms)
            )
            measured_postprocess_frames = (
                int(
                    (throughput_result or {}).get(
                        "postprocess_completed_frames"
                    ) or 0
                )
                if bool(args.throughput_mode)
                else int(
                    completion_processor.completed_count
                    - measured_postprocess_before
                    if completion_processor is not None else 0
                )
            )
            completed_task_attestation: dict[str, Any] = {}
            if (
                completion_processor is not None
                and measured_completed_frames > 0
                and measured_postprocess_frames
                == measured_completed_frames
            ):
                if frozen_postprocessor is not None:
                    completed_task_attestation = (
                        build_completed_detection_endpoint_attestation(
                            frozen_postprocess_contract,
                            frozen_postprocessor.last_result,
                            completed_frames=measured_completed_frames,
                            postprocess_completed_frames=(
                                measured_postprocess_frames
                            ),
                            source_endpoint_contract_hash=str(
                                probe_contract.get(
                                    "endpoint_contract_hash"
                                ) or ""
                            ),
                        )
                    )
                elif decoded_nms_normalizer is not None:
                    completed_task_attestation = (
                        build_normalized_detection_endpoint_attestation(
                            frozen_decoded_nms_normalization_contract,
                            decoded_nms_normalizer.last_result,
                            completed_frames=measured_completed_frames,
                            postprocess_completed_frames=(
                                measured_postprocess_frames
                            ),
                        )
                    )
            completion_result = (
                dict(completion_processor.last_result)
                if completion_processor is not None and classification_processor is None else {}
            )
            completed_result_artifact = (
                dict(
                    completion_result.get(
                        "completed_result_artifact"
                    ) or {}
                )
                if isinstance(
                    completion_result.get("completed_result_artifact"),
                    Mapping,
                )
                else {}
            )
            completed_result_artifact_sha256 = str(
                completion_result.get(
                    "completed_result_artifact_sha256"
                ) or ""
            )
            completed_result_persistence = {
                "saved": False,
                "path": "",
                "file_sha256": "",
            }
            if completed_task_attestation.get("attested") is True:
                completed_result_persistence = (
                    _persist_completed_result_artifact(
                        completed_result_artifact,
                        completed_result_artifact_sha256,
                        json_path.with_name(
                            f"{json_path.stem}.completed_task_result_artifact.json"
                        ),
                    )
                )
            completed_v2_ok = bool(
                completed_task_attestation.get("attested") is True
                and completed_result_persistence["saved"] is True
                and measured_postprocess_frames == measured_completed_frames
                and measured_completed_frames > 0
            )
            if native_full_detection and not completed_v2_ok:
                raise RuntimeError(
                    "Native Full Hailo detection completed without a persisted Completed-v2 attestation"
                )
            report = {
                "hef": str(hef_path),
                "hw_arch": str(args.hw_arch),
                "quantized_inputs": bool(args.quantized_inputs),
                "quantized_outputs": bool(args.quantized_outputs),
                "persistent_activation": bool(args.persistent_activation),
                "runtime_api": str(args.runtime_api),
                "hotloop": bool(args.hotloop),
                "copy_outputs": bool(args.copy_outputs),
                "copy_inputs": bool(args.copy_inputs),
                "throughput_mode": bool(args.throughput_mode),
                "throughput": throughput_result,
                "request_latency": (throughput_result or {}).get("request_latency") or request_latency.report(),
                "completed_frames": (
                    measured_completed_frames
                ),
                "completed_work_units_status": (
                    str((throughput_result or {}).get("completed_work_units_status") or "")
                    if bool(args.throughput_mode) else "exact_synchronous_timing_count"
                ),
                "requested_duration_s": float(args.duration_s or 0.0),
                "measured_duration_s": (
                    float((throughput_result or {}).get("elapsed_s") or 0.0)
                    if bool(args.throughput_mode) else None
                ),
                "minimum_duration_satisfied": (
                    bool((throughput_result or {}).get("minimum_duration_satisfied"))
                    if float(args.duration_s or 0.0) > 0.0 else True
                ),
                "claim_copy_outputs_verified": bool(args.copy_outputs) if claim_performance_run else None,
                "session_hotloop": bool(getattr(prep.session, "hotloop", False)),
                "input_names": list(prepared.input_names),
                "output_names": list(prepared.output_names),
                "input_shapes": {k: list(v.shape) for k, v in inputs.items()},
                "input_dtypes": {k: str(v.dtype) for k, v in inputs.items()},
                "runtime_input_name": str(
                    runtime_input_evidence["runtime_input_name"]
                ),
                "runtime_input_shape": list(
                    runtime_input_evidence["runtime_input_shape"]
                ),
                "runtime_input_dtype": str(
                    runtime_input_evidence["runtime_input_dtype"]
                ),
                "runtime_input_layout": str((input_preprocess or {}).get("layout") or ""),
                "runtime_preprocess_mode": str((input_preprocess or {}).get("mode") or ""),
                "runtime_normalization": runtime_normalization,
                "runtime_color_space": "RGB" if exact_image is not None or (input_preprocess or {}).get("rgb") is True else "unavailable",
                "runtime_input_source": (
                    "preflight_bound_runtime_input_tensor" if bound_runtime_mode
                    else "runner_image_preprocessing" if exact_image is not None
                    else "generated_random_input"
                ),
                "runtime_input_file": str(
                    runtime_input_evidence["runtime_input_file"]
                ),
                "runtime_input_sha256": str(
                    runtime_input_evidence["runtime_input_sha256"]
                ),
                "runtime_input_bytes": int(
                    runtime_input_evidence["runtime_input_bytes"]
                ),
                "runtime_input_contract_schema": str((bound_runtime_contract or {}).get("schema") or ""),
                "runtime_input_contract_schema_version": int((bound_runtime_contract or {}).get("schema_version") or 0),
                "runtime_input_binding_verified": bool(
                    runtime_input_evidence["runtime_input_binding_verified"]
                ),
                "runtime_input_preflight_bound": bool(
                    runtime_input_evidence["runtime_input_preflight_bound"]
                ),
                "runtime_input_hash_source": str(
                    runtime_input_evidence["runtime_input_hash_source"]
                ),
                "preprocessing_contract": preprocessing_contract,
                "preprocessing_contract_sha256": (
                    preprocessing_contract_digest
                ),
                "preprocessing_contract_attested": bool(
                    preprocessing_contract
                    and preprocessing_contract_digest
                    == preprocessing_contract_sha256(
                        preprocessing_contract
                    )
                ),
                **runtime_preprocessing_binding,
                "preprocessing_geometry": dict(
                    (input_preprocess or {}).get(
                        "preprocessing_geometry"
                    ) or {}
                ),
                "canonical_input_slot_names": canonical_input_names,
                "canonical_output_slot_names": canonical_output_names,
                "image_decode_performed": bool(exact_image is not None and not bound_runtime_mode),
                "preprocessing_performed": bool(exact_image is not None and not bound_runtime_mode),
                "preprocessing_timed": False,
                "runtime_output_shapes": {k: list(v) for k, v in prep.runtime_output_shapes.items()},
                "timings_ms": timings_ms,
                "output_summary": output_summary,
                "ok": bool(
                    not native_full_detection or completed_v2_ok
                ),
                "task": str(args.task or ""),
                "backend": str(args.backend_label),
                "model": str(args.model or ""),
                "setup_id": str(args.setup_id or ""),
                "comparison_backend": str(args.comparison_backend or ""),
                "input_image": str(exact_image) if exact_image is not None else "",
                "input_image_sha256": input_image_sha256 if exact_image is not None else "",
                "output_manifest": output_manifest,
                "input_manifest": input_manifest,
                "frozen_host_postprocess_contract": frozen_postprocess_contract,
                "frozen_host_postprocess_contract_sha256": str(
                    frozen_postprocess_contract.get("contract_sha256") or ""
                ),
                "frozen_host_postprocess_result": (
                    dict(frozen_postprocessor.last_result)
                    if frozen_postprocessor is not None else {}
                ),
                "host_postprocess_frozen": frozen_postprocessor is not None,
                "normalization_frozen": decoded_nms_normalizer is not None,
                "frozen_decoded_nms_normalization_contract": (
                    frozen_decoded_nms_normalization_contract
                ),
                "frozen_decoded_nms_normalization_contract_sha256": str(
                    frozen_decoded_nms_normalization_contract.get(
                        "contract_sha256"
                    ) or ""
                ),
                "frozen_decoded_nms_normalization_result": (
                    dict(decoded_nms_normalizer.last_result)
                    if decoded_nms_normalizer is not None else {}
                ),
                "postprocess_included": completion_processor is not None,
                "postprocess_location": (
                    "serialized_host_tail_inside_measured_interval"
                    if frozen_postprocessor is not None
                    else (
                        "serialized_host_normalization_inside_measured_interval"
                        if decoded_nms_normalizer is not None else ""
                    )
                ),
                "postprocess_completed_frames": int(
                    measured_postprocess_frames
                ),
                "postprocess_completion_verified": bool(
                    completion_processor is None
                    or completed_task_attestation.get("attested") is True
                ),
                "completed_task_endpoint_contract": dict(
                    completed_task_attestation.get(
                        "completed_endpoint_contract"
                    ) or {}
                ),
                "completed_task_stage": str(
                    completed_task_attestation.get("stage") or ""
                ),
                "completed_task_contract_family": (
                    "decoded_nms"
                    if completed_task_attestation.get("attested") is True
                    else ""
                ),
                "completed_task_endpoint_contract_hash": str(
                    completed_task_attestation.get(
                        "endpoint_contract_hash"
                    ) or ""
                ),
                "completed_task_output_endpoint_id": str(
                    completed_task_attestation.get(
                        "output_endpoint_id"
                    ) or ""
                ),
                "completed_task_comparison_endpoint_contract": dict(
                    completed_task_attestation.get(
                        "completed_task_comparison_endpoint_contract"
                    ) or {}
                ),
                "completed_task_comparison_endpoint_contract_hash": str(
                    completed_task_attestation.get(
                        "completed_task_comparison_endpoint_contract_hash"
                    ) or ""
                ),
                "completed_task_comparison_output_endpoint_id": str(
                    completed_task_attestation.get(
                        "completed_task_comparison_output_endpoint_id"
                    ) or ""
                ),
                "completed_task_completion_mode": str(
                    completed_task_attestation.get(
                        "completed_task_completion_mode"
                    ) or ""
                ),
                "completed_task_endpoint_attested": (
                    completed_task_attestation.get("attested") is True
                ),
                "completed_task_endpoint_attestation_status": str(
                    completed_task_attestation.get("status") or ""
                ),
                "completed_task_endpoint_attestation": (
                    completed_task_attestation
                ),
                "completed_task_result_artifact_saved": bool(
                    completed_result_persistence["saved"]
                ),
                "completed_task_result_artifact": (
                    completed_result_artifact
                ),
                "completed_task_result_artifact_sha256": (
                    completed_result_artifact_sha256
                ),
                "completed_task_result_artifact_path": str(
                    completed_result_persistence["path"]
                ),
                "completed_task_result_artifact_file_sha256": str(
                    completed_result_persistence["file_sha256"]
                ),
                "original_image_wh": original_wh or [],
                "runtime_endpoint_contract_family": str(
                    probe_contract.get("contract_family")
                    or frozen_postprocess_contract.get("source_contract_family")
                    or frozen_decoded_nms_normalization_contract.get(
                        "source_contract_family"
                    )
                    or ""
                ),
            }
            if classification_processor is not None:
                report.update(classification_processor.report(measured_postprocess_frames))
            if args.diagnostic_only:
                report.update({
                    "diagnostic_only": True,
                    "claim_eligible": False,
                    "claim_eligible_e2e": False,
                })
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            print("[hailo10-smoke] wrote:", json_path)
    except Exception as exc:
        evidence = getattr(exc, "request_latency", None)
        if evidence is None and "request_latency" in locals():
            evidence = request_latency.report()
        if args.json_out and evidence is not None:
            failure_path = Path(args.json_out).expanduser()
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps({"ok": False, "status": "runtime_failed",
                "error": f"{type(exc).__name__}: {exc}", "request_latency": evidence}, indent=2))
        raise
    finally:
        if prepared is not None:
            backend.cleanup(prepared)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
