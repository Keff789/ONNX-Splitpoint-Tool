"""Real preparation -> BN6 materialization regressions; no device/compiler.

Only SDK inference is synthetic. Preprocessing, declaration binding, runtime
value attestation, inverse geometry and result persistence use product code.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import onnx_splitpoint_tool.native_detection_postprocess as postprocess
import onnx_splitpoint_tool.native_output_endpoint as endpoint
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    prepare_rgb_uint8_image,
)


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"


@pytest.fixture
def suite(monkeypatch):
    module = types.ModuleType("fix2_deepx_suite")
    module.__file__ = str(TEMPLATE)
    exec(compile(TEMPLATE.read_text(), str(TEMPLATE), "exec"), module.__dict__)
    package = types.ModuleType("splitpoint_runners")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "splitpoint_runners", package)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.native_detection_postprocess", postprocess)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.native_output_endpoint", endpoint)
    return module


@pytest.fixture
def contract(tmp_path, suite):
    authority = {
        "model_id": "yolo26m", "backend": "deepx_m1", "variant": "full",
        "task": "detection", "contract_status": "recorded",
        "endpoint_mode": "decoded", "host_tail_required": False,
        "postprocessing_required": False,
        "source_coordinate_space": "model_input_letterbox_xyxy_pixels",
    }
    (tmp_path / "output_contracts.json").write_text(json.dumps({
        "model_id": "yolo26m", "task": "detection", "contracts": [authority],
    }))
    raw = {
        **authority,
        "input": {
            "name": "images", "shape": [640, 640, 3], "dtype": "uint8",
            "layout": "HWC", "normalization": "embedded_dxcom_preprocessing",
            "color_space": "RGB", "preprocess_mode": "letterbox",
            "letterbox_pad_value": 114,
        },
        "outputs": [{"name": "detections", "shape": [1, 1, 6], "dtype": "float32"}],
        "postprocessing": {"host_required": False, "nms_on_host": False},
    }
    bound = suite._deepx_bind_authoritative_endpoint_contract(
        tmp_path, {"model_id": "yolo26m", "benchmark_task": "detection"}, raw,
    )
    assert bound["endpoint_contract_binding_status"] == "attested", bound
    return bound


def _image(height, width):
    return np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)


def _output():
    return np.array([[[200, 200, 300, 300, .9, 1]]], dtype=np.float32)


def _decode(suite, root, contract, source, audit):
    return suite._deepx_detection_decode(
        root=root, run={"model_id": "yolo26m"}, contract=contract,
        outputs=[_output()], orig_shape=source.shape,
        scale=audit["scale"], pad_x=audit["pad_x"], pad_y=audit["pad_y"],
    )


@pytest.mark.parametrize("height,width", [(333, 500), (500, 333), (335, 500)])
def test_nominal_gain_reaches_real_decoder_without_changing_pixels(
    tmp_path, suite, contract, height, width,
):
    source = _image(height, width)
    source_before = source.tobytes()
    feed, audit = suite._deepx_prepare_image_from_contract(source, 640, contract, "detection")
    canonical = canonical_image_preprocessing_contract("detection", [640, 640])
    expected, effective = prepare_rgb_uint8_image(source[..., ::-1], canonical)
    geometry = postprocess.build_letterbox_geometry_contract(
        input_hw=[640, 640], original_wh=[width, height], preprocess=contract["input"],
    )
    assert feed.tobytes() == expected.tobytes()
    assert source.tobytes() == source_before
    assert audit["prepared_tensor_binding"]["prepared_input_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest()
    assert [audit["pad_x"], audit["pad_y"]] == effective["pad_ltrb"][:2]
    assert audit["scale"] == geometry["gain"] == 1.28
    assert effective["scale_xy"] == [effective["resized_hw"][1] / width, effective["resized_hw"][0] / height]
    detections, result = _decode(suite, tmp_path, contract, source, audit)
    assert result["pass"] is True, result
    assert len(detections) == 1
    assert result["host_nms_applied"] is False
    expected_box = [(200 - audit["pad_x"]) / 1.28, (200 - audit["pad_y"]) / 1.28,
                    (300 - audit["pad_x"]) / 1.28, (300 - audit["pad_y"]) / 1.28]
    assert [detections[0][key] for key in ("x1", "y1", "x2", "y2")] == pytest.approx(expected_box)


@pytest.mark.parametrize("field,delta", [("scale", -.01), ("pad_x", 1), ("pad_y", 1)])
def test_incorrect_geometry_still_blocks_exact_materialization(tmp_path, suite, contract, field, delta):
    source = _image(333, 500)
    _, audit = suite._deepx_prepare_image_from_contract(source, 640, contract, "detection")
    audit[field] += delta
    detections, result = _decode(suite, tmp_path, contract, source, audit)
    assert detections == []
    assert result["pass"] is False
    assert "preprocessing_geometry_mismatch" in result["error"]


def test_classification_resize_keeps_existing_pixels_and_scalar(suite):
    source = _image(333, 500)
    contract = {"input": {
        "shape": [224, 224, 3], "dtype": "uint8", "layout": "HWC",
        "normalization": "raw", "color_space": "RGB", "preprocess_mode": "resize",
        "letterbox_pad_value": 0,
    }}
    feed, audit = suite._deepx_prepare_image_from_contract(source, 224, contract, "classification")
    expected, geometry = prepare_rgb_uint8_image(source[..., ::-1], canonical_image_preprocessing_contract("classification", [224, 224]))
    assert feed.tobytes() == expected.tobytes()
    assert audit["scale"] == min(geometry["scale_xy"])
    assert audit["pad_x"] == audit["pad_y"] == 0


@pytest.mark.parametrize("corrupt_one", [False, True])
def test_semantic_loop_keeps_every_odd_image_or_reports_incomplete(
    tmp_path, suite, contract, monkeypatch, corrupt_one,
):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index, shape in enumerate([(333, 500), (500, 333), (335, 500)]):
        Image.fromarray(_image(*shape)[..., ::-1]).save(image_dir / f"{index}.png")
    dxnn = tmp_path / "model.dxnn"
    dxnn.write_bytes(b"synthetic SDK input; no model compiler")
    seen = []

    class Engine:
        def __init__(self, path):
            assert Path(path) == dxnn

        def run(self, feeds):
            seen.append(feeds[0].copy())
            return [_output()]

    dx = types.ModuleType("dx_engine")
    dx.InferenceEngine = Engine
    cv = types.ModuleType("cv2")
    cv.imread = lambda path: np.asarray(Image.open(path).convert("RGB"))[..., ::-1].copy()
    monkeypatch.setitem(sys.modules, "dx_engine", dx)
    monkeypatch.setitem(sys.modules, "cv2", cv)
    monkeypatch.setattr(suite, "_deepx_input_size_from_contract", lambda *args, **kwargs: (640, contract))
    original_prepare = suite._deepx_prepare_image_from_contract
    if corrupt_one:
        def prepare(source, *args):
            feed, audit = original_prepare(source, *args)
            if source.shape[:2] == (333, 500):
                audit["scale"] -= .01
            return feed, audit
        monkeypatch.setattr(suite, "_deepx_prepare_image_from_contract", prepare)
    results = tmp_path / "results"
    results.mkdir()
    result = suite._run_deepx_semantic_validation(
        tmp_path, dxnn, {"model_id": "yolo26m", "benchmark_task": "detection"},
        SimpleNamespace(validation_images=str(image_dir)), results,
    )
    assert len(seen) == 3
    assert result["image_count"] == 3
    assert result["validated_image_count"] == (2 if corrupt_one else 3)
    assert result["error_count"] == int(corrupt_one)
    assert result["status"] == ("semantic_runtime_incomplete" if corrupt_one else "ok")
    assert result["decoder_postprocess_contract"]["pass"] is (not corrupt_one)
    assert result["preprocessing_contract"]["all_samples_same_contract"] is True
    saved = json.loads((results / "detections.json").read_text())
    assert len(saved["images"]) == result["validated_image_count"]
    if corrupt_one:
        # The actual exporter rejects incomplete execution before it can create
        # a central request; no fabricated candidate/GT or relaxed gate.
        with pytest.raises(RuntimeError, match="deepx_quality_semantic_validation_incomplete"):
            suite._deepx_export_central_quality_request(
                tmp_path, dxnn,
                {"id": "deepx_m1_full", "model_id": "yolo26m", "variants": ["full"],
                 "backend": "deepx_m1", "benchmark_task": "detection",
                 "task_quality_gate": {"statistics": {"execution_location": "central_management"}}},
                result, results, expected_model_id="yolo26m", expected_backend="deepx_m1", expected_variant="full",
            )
