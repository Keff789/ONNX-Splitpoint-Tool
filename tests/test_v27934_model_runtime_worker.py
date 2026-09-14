"""Runtime diagnostic contract tests. Device inference remains a hardware gate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract, prepare_rgb_uint8_image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_worker_v34", ROOT / "scripts/hailo_model_runtime_worker_v27934.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


@pytest.fixture
def stage(tmp_path):
    value = tmp_path / "stage"
    value.mkdir()
    request = {"schema": "hailo_model_runtime_request_v27934", "model": "mobilenet_v3_large", "family": "hailo10h",
               "setup_id": "bound_hailo10h_setup", "input_name": "images", "input_shape": [1, 3, 6, 6],
               "output_name": "logits", "class_count": 1000,
               "expected_source_onnx_sha256": "1" * 64, "expected_compiler_sha256": "2" * 64,
               "preprocessing_contract": canonical_image_preprocessing_contract("classification", [6, 6]), "images": []}
    def file_row(path):
        return {"path": str(path.relative_to(value)), "sha256": worker._sha(path), "size_bytes": path.stat().st_size}
    for role in ("cpu_hef", "gpu_hef"):
        path = value / (role + ".hef")
        path.write_bytes(b"synthetic-not-runnable-HEF-" + role.encode())
        request[role] = file_row(path)
    for i in range(16):
        path = value / (str(i) + ".png")
        Image.fromarray(np.full((7, 9, 3), i, dtype=np.uint8)).save(path)
        request["images"].append({**file_row(path), "id": str(i), "label": i})
    (value / "request.json").write_text(json.dumps(request))
    return value, request


def overwrite(stage, request):
    (stage / "request.json").write_text(json.dumps(request))


def test_fixed16_request_binds_exact_files_names_graph_and_contract(stage):
    path, request = stage
    assert worker.validate_staged_request(path) == request
    assert not (path / "results").exists()


@pytest.mark.parametrize("role", ["cpu_hef", "gpu_hef", "images"])
def test_modified_source_file_rejected_before_runtime(stage, role):
    path, request = stage
    row = request[role][0] if role == "images" else request[role]
    (path / row["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity_mismatch"):
        worker.validate_staged_request(path)
    assert not (path / "results").exists()


@pytest.mark.parametrize("bad_path", ["../outside.hef", "/tmp/outside.hef", "sub\\outside.hef"])
def test_input_path_cannot_escape_private_stage(stage, bad_path):
    path, request = stage
    request["cpu_hef"]["path"] = bad_path
    overwrite(path, request)
    with pytest.raises(ValueError, match="inside_stage"):
        worker.validate_staged_request(path)


def test_input_symlink_cannot_point_to_other_artifact(stage):
    path, request = stage
    hef = path / request["gpu_hef"]["path"]
    hef.unlink()
    hef.symlink_to(path / request["cpu_hef"]["path"])
    with pytest.raises(ValueError, match="regular_local"):
        worker.validate_staged_request(path)


@pytest.mark.parametrize("key,value", [("family", "hailo8"), ("input_shape", [1, 4, 4, 3]),
    ("input_shape", [1, 3, 4096, 4096]), ("class_count", True), ("expected_compiler_sha256", ""), ("output_name", "")])
def test_arbitrary_scope_shape_names_and_graph_identity_rejected(stage, key, value):
    path, request = stage
    request[key] = value
    overwrite(path, request)
    with pytest.raises(ValueError):
        worker.validate_staged_request(path)


def test_duplicate_image_ids_and_wrong_normalization_rejected(stage):
    path, request = stage
    request["images"][1]["id"] = request["images"][0]["id"]
    overwrite(path, request)
    with pytest.raises(ValueError, match="distinct"):
        worker.validate_staged_request(path)
    request["images"][1]["id"] = "1"
    request["preprocessing_contract"]["image_scale"] = "raw"
    overwrite(path, request)
    with pytest.raises(ValueError, match="preprocessing contract mismatch"):
        worker.validate_staged_request(path)


@pytest.mark.parametrize("shape", [(1000,), (1, 1000), (1, 1, 1000), (1, 1000, 1, 1)])
def test_unique_class_axis_uses_explicit_graph_class_count(shape):
    value = np.arange(1000, dtype=np.float32).reshape(shape)
    actual = worker.classification_vector(value, 1000)
    assert np.array_equal(actual, np.arange(1000, dtype=np.float32))


@pytest.mark.parametrize("shape,dtype", [((1000, 1000), "float32"), ((2, 1000), "float32"), ((1000,), "uint8"), ((5,), "float32")])
def test_ambiguous_axes_or_quantized_output_do_not_turn_into_logits(shape, dtype):
    with pytest.raises(ValueError):
        worker.classification_vector(np.zeros(shape, dtype=dtype), 1000)


def test_nonfinite_logits_are_preserved_as_failure():
    x = np.zeros((1, 1000), dtype=np.float32)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite_dequantized"):
        worker.classification_vector(x, 1000)


def test_actual_existing_preprocessing_once_and_layout_roundtrip(stage):
    path, request = stage
    sys.path.insert(0, str(ROOT / "scripts"))
    from smoke_hailo10_hef_runner import _image_tensor
    original = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
    image_path = path / "gradient.png"
    Image.fromarray(original).save(image_path)
    prepared, _ = prepare_rgb_uint8_image(original, request["preprocessing_contract"])
    expected = (prepared.astype(np.float32) / 255.0 - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    feeds = []
    for shape in [(6, 6, 3), (1, 6, 6, 3), (3, 6, 6), (1, 3, 6, 6)]:
        logical, rgb, info = _image_tensor(image_path, shape, quantized=False, task="classification", preprocess_mode="resize", letterbox_pad=0)
        actual = worker._logical_hwc(logical, info["layout"])
        assert np.array_equal(actual, expected)
        assert np.array_equal(rgb, prepared)
        assert info["preprocessing_contract"] == request["preprocessing_contract"]
        feeds.append(actual)
    assert all(np.array_equal(feeds[0], x) for x in feeds[1:])


def test_unsafe_request_fails_before_backend_creation(stage):
    path, request = stage
    request["class_count"] = False
    overwrite(path, request)
    with pytest.raises(ValueError, match="explicit_class_count"):
        worker.collect(path)
    assert not (path / "results").exists()
