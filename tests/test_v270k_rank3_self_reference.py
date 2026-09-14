from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


def _load_validator():
    path = Path("scripts/native_producer_validate_visualize.py")
    spec = importlib.util.spec_from_file_location(
        "v270k_rank3_self_reference_validator",
        path,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def validator():
    return _load_validator()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sealed_manifest(
    tmp_path: Path,
    tensor: np.ndarray,
    *,
    layout: str = "HWC",
    scale: str = "norm",
    extra: dict[str, Any] | None = None,
) -> Path:
    runtime = np.ascontiguousarray(tensor)
    runtime_path = tmp_path / "runtime_input.bin"
    runtime_path.write_bytes(runtime.tobytes(order="C"))
    manifest = {
        "schema": "onnx-splitpoint/native-full-input-dump",
        "schema_version": 1,
        "runtime_input_file": runtime_path.name,
        "runtime_input_sha256": _sha256_bytes(runtime_path.read_bytes()),
        "runtime_input_bytes": runtime_path.stat().st_size,
        "runtime_input_shape": [int(value) for value in runtime.shape],
        "runtime_input_dtype": str(runtime.dtype),
        "runtime_input_name": "images",
        "preprocess": {
            "layout": layout,
            "ort_model_scale": scale,
        },
    }
    if extra:
        manifest.update(extra)
    manifest_path = tmp_path / "native_full_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _derive(
    validator: Any,
    manifest: Path,
    shape: list[int],
    *,
    target_dtype: Any = np.float32,
    image_scale: str = "native",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    evidence: dict[str, Any] = {}
    feed = validator._native_input_dump_feed(
        manifest,
        shape,
        image_scale=image_scale,
        target_dtype=target_dtype,
        evidence=evidence,
    )
    return feed, evidence


def test_detection_norm_hwc_uint8_derives_exact_rank4_nchw(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    manifest = _sealed_manifest(tmp_path, runtime, scale="norm")

    feed, evidence = _derive(validator, manifest, [1, 3, 2, 4])
    expected = np.transpose(
        runtime.astype(np.float32) / 255.0,
        (2, 0, 1),
    )[None]

    assert feed is not None
    np.testing.assert_array_equal(feed, expected)
    assert feed.shape == (1, 3, 2, 4)
    assert feed.dtype == np.float32
    assert feed.flags.c_contiguous
    assert evidence["status"] == "passed"
    assert (
        evidence["transformation_id"]
        == "sealed_runtime_to_onnx_reference_v1"
    )
    assert evidence["source_shape"] == [2, 4, 3]
    assert evidence["source_layout"] == "HWC"
    assert evidence["target_shape"] == [1, 3, 2, 4]
    assert evidence["target_layout"] == "NCHW"
    assert evidence["ort_model_scale"] == "norm"
    assert evidence["derived_reference_tensor_sha256"] == _sha256_bytes(
        expected.tobytes(order="C"),
    )


def test_resnet_imagenet_hwc_uint8_derives_exact_rank4_nchw(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.asarray(
        [
            [[0, 32, 64], [96, 128, 160], [192, 224, 255], [8, 16, 24]],
            [[40, 48, 56], [72, 80, 88], [104, 112, 120], [136, 144, 152]],
        ],
        dtype=np.uint8,
    )
    manifest = _sealed_manifest(tmp_path, runtime, scale="imagenet")

    feed, evidence = _derive(validator, manifest, [1, 3, 2, 4])
    work = runtime.astype(np.float32) / 255.0
    work = (
        work
        - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    expected = np.transpose(work, (2, 0, 1))[None]

    assert feed is not None
    np.testing.assert_array_equal(feed, expected)
    assert feed.flags.c_contiguous
    assert evidence["status"] == "passed"
    assert evidence["ort_model_scale"] == "imagenet"
    assert evidence["derived_reference_tensor_sha256"] == _sha256_bytes(
        expected.tobytes(order="C"),
    )


def test_rank4_nchw_runtime_tensor_is_preserved_byte_exactly(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.linspace(
        -2.0,
        2.0,
        num=1 * 3 * 2 * 4,
        dtype=np.float32,
    ).reshape(1, 3, 2, 4)
    manifest = _sealed_manifest(
        tmp_path,
        runtime,
        layout="NCHW",
        scale="norm",
    )

    feed, evidence = _derive(validator, manifest, [1, 3, 2, 4])

    assert feed is not None
    np.testing.assert_array_equal(feed, runtime)
    assert feed.tobytes(order="C") == runtime.tobytes(order="C")
    assert evidence["transformation_id"] == "sealed_runtime_identity_v1"
    assert evidence["source_runtime_input_sha256"] == _sha256_bytes(
        runtime.tobytes(order="C"),
    )
    assert evidence["derived_reference_tensor_sha256"] == _sha256_bytes(
        runtime.tobytes(order="C"),
    )


def test_rank4_runtime_dtype_mismatch_is_not_silently_cast(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4)
    manifest = _sealed_manifest(
        tmp_path,
        runtime,
        layout="NCHW",
        scale="norm",
    )

    feed, evidence = _derive(
        validator,
        manifest,
        [1, 3, 2, 4],
        target_dtype=np.float16,
    )

    assert feed is None
    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == "sealed_runtime_identity_dtype_mismatch"


def test_tampered_runtime_sha_fails_closed(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    manifest = _sealed_manifest(tmp_path, runtime)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["runtime_input_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    feed, evidence = _derive(validator, manifest, [1, 3, 2, 4])

    assert feed is None
    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == "sealed_runtime_tensor_binding_invalid"


def test_conflicting_runtime_layout_fails_closed(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    manifest = _sealed_manifest(
        tmp_path,
        runtime,
        layout="HWC",
        extra={"runtime_input_layout": "CHW"},
    )

    feed, evidence = _derive(validator, manifest, [1, 3, 2, 4])

    assert feed is None
    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == "sealed_runtime_layout_conflict"


def test_unsupported_reference_scale_fails_closed(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    manifest = _sealed_manifest(tmp_path, runtime, scale="raw")

    feed, evidence = _derive(validator, manifest, [1, 3, 2, 4])

    assert feed is None
    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == "sealed_runtime_onnx_scale_unsupported"


def test_rank3_non_uint8_runtime_dtype_fails_closed(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    manifest = _sealed_manifest(tmp_path, runtime, scale="norm")

    feed, evidence = _derive(validator, manifest, [1, 3, 2, 4])

    assert feed is None
    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == "sealed_runtime_rank3_dtype_unsupported"


def test_unsupported_onnx_input_dtype_is_not_coerced(
    validator: Any,
) -> None:
    assert validator._ort_input_dtype(
        SimpleNamespace(type="tensor(int8)"),
    ) is None


def test_incompatible_onnx_shape_fails_closed(
    validator: Any,
    tmp_path: Path,
) -> None:
    runtime = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    manifest = _sealed_manifest(tmp_path, runtime, scale="norm")

    feed, evidence = _derive(validator, manifest, [1, 3, 5, 4])

    assert feed is None
    assert evidence["status"] == "unavailable"
    assert evidence["reason"] == "sealed_runtime_derived_reference_invalid"
