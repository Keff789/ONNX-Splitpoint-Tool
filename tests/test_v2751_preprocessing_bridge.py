from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_module(path: Path, prefix: str) -> Any:
    name = f"{prefix}_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _quality_export_fixture(
    tmp_path: Path, preprocessing_identity: dict[str, Any],
) -> tuple[Any, Any, Path, Path, dict[str, Any], dict[str, Any]]:
    fixture = _load_module(
        ROOT / "tests/test_v269e_quality_export_vendored.py",
        "v2751_quality_fixture",
    )
    outputs = [("output0", [1, 300, 6])]
    module, case, _manifest_full, persistent_full = (
        fixture._load_self_contained_runner(
            tmp_path,
            model_id="yolo26s",
            endpoint_mode="decoded",
            outputs=outputs,
        )
    )
    quality, context = fixture._quality_contract_and_producer_context(
        module,
        root=tmp_path / "producer_context",
        source_onnx=persistent_full,
    )
    quality["preprocessing"] = {
        "identity": copy.deepcopy(preprocessing_identity),
        "sha256": module._quality_contract_sha256(preprocessing_identity),
    }
    quality.pop("quality_contract_sha256", None)
    quality["quality_contract_sha256"] = module._quality_contract_sha256(
        quality
    )
    return fixture, module, case, persistent_full, quality, context


def _export_detection_request(
    fixture: Any,
    module: Any,
    case: Path,
    persistent_full: Path,
    quality: dict[str, Any],
    context: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return fixture._full_detection_request(
        module,
        case=case,
        persistent_full=persistent_full,
        names=["output0"],
        arrays=[np.zeros((1, 300, 6), dtype=np.float32)],
        diagnostics={},
        producer_context=context,
        quality_contract=quality,
    )


def test_trt_quality_producer_accepts_canonical_v2_target_hw(
    tmp_path: Path,
) -> None:
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640)
    )
    assert preprocessing["schema_version"] == 2
    assert preprocessing["target_hw"] == [640, 640]
    assert "input_hw" not in preprocessing

    fixture, module, case, persistent_full, quality, context = (
        _quality_export_fixture(tmp_path, preprocessing)
    )
    _endpoint, request = _export_detection_request(
        fixture, module, case, persistent_full, quality, context
    )

    sealed = request["producer_identity"]["quality_contract"][
        "preprocessing"
    ]
    assert sealed["identity"] == preprocessing
    assert sealed["sha256"] == module._quality_contract_sha256(preprocessing)


def test_trt_quality_producer_rejects_resealed_v2_input_hw_alias(
    tmp_path: Path,
) -> None:
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640)
    )
    preprocessing["input_hw"] = preprocessing.pop("target_hw")
    fixture, module, case, persistent_full, quality, context = (
        _quality_export_fixture(tmp_path, preprocessing)
    )

    assert quality["preprocessing"]["sha256"] == (
        module._quality_contract_sha256(preprocessing)
    )
    quality_payload = copy.deepcopy(quality)
    declared_quality_sha = quality_payload.pop("quality_contract_sha256")
    assert declared_quality_sha == module._quality_contract_sha256(
        quality_payload
    )

    with pytest.raises(
        RuntimeError,
        match="quality-only TRT detection semantics differ",
    ):
        _export_detection_request(
            fixture, module, case, persistent_full, quality, context
        )


def test_completed_projection_rejects_target_input_geometry_mismatch() -> None:
    validator = _load_module(
        ROOT / "scripts/native_producer_validate_visualize.py",
        "v2751_validator",
    )
    physical_endpoint_sha = "1" * 64
    source_onnx_sha = "2" * 64
    decoder_sha = "3" * 64
    nms_sha = "4" * 64
    preprocessing = canonical_image_preprocessing_contract(
        "detection", (640, 640)
    )
    producer = {
        "task": "detection",
        "model_id": "yolov7_paper",
        "endpoint_contract_hash": physical_endpoint_sha,
        "source_onnx": {"sha256": source_onnx_sha},
        "endpoint": {"identity": {"stage": "raw_head"}},
        "decoder_contract_sha256": decoder_sha,
        "nms_contract_sha256": nms_sha,
        "quality_record_endpoint": {
            "identity": {
                "canonical_record_endpoint": (
                    "decoded_xyxy_score_class_detections"
                ),
                "decoder_contract_sha256": decoder_sha,
                "nms_contract_sha256": nms_sha,
            }
        },
        "quality_contract": {
            "task": "detection",
            "canonical_record_endpoint": (
                "decoded_xyxy_score_class_detections"
            ),
            "source_endpoint_is_raw": True,
            "model": {"sha256": source_onnx_sha},
            "preprocessing": {"identity": preprocessing},
            "decoder": {
                "sha256": decoder_sha,
                "identity": {
                    "canonical_record_endpoint": (
                        "decoded_xyxy_score_class_detections"
                    ),
                    "source_output_format": "multiscale_head",
                    "source_endpoint_semantics": "raw_multiscale_head",
                    "source_endpoint_has_integrated_nms": False,
                    "confidence_threshold": 0.25,
                },
            },
            "nms": {
                "sha256": nms_sha,
                "identity": {
                    "iou_threshold": 0.45,
                    "max_detections": 300,
                },
            },
        },
    }
    comparison = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "model_id": "yolov7_paper",
        "class_aware": True,
        "score_threshold": 0.25,
        "iou_threshold": 0.45,
        "max_detections": 300,
        "input_hw": [320, 320],
        "canonical_completion_policy_id": (
            "decoded_nms_xyxy_original_classaware_postfilter_v2"
        ),
        "nms_semantics_id": "class_aware_nms_xyxy_v1",
    }

    verified, status = validator._native_trt_completed_quality_projection(
        producer,
        comparison,
        physical_endpoint_hash=physical_endpoint_sha,
    )
    assert verified is False
    assert status == "native_trt_completed_quality_semantics_mismatch"


def test_native_validator_remote_mirror_is_byte_identical() -> None:
    local = ROOT / "scripts/native_producer_validate_visualize.py"
    remote = (
        ROOT
        / "onnx_splitpoint_tool/resources/remote_scripts"
        / "native_producer_validate_visualize.py"
    )
    assert local.read_bytes() == remote.read_bytes()
