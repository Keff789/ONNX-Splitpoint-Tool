from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from onnx_splitpoint_tool.runners._types import SampleCfg
from onnx_splitpoint_tool.runners.harness.base import postprocess_result_to_dict
from onnx_splitpoint_tool.runners.harness.classification import ClassificationHarness


def _save_image(img: Image.Image, path):
    img.save(path)
    return path


def test_preprocessing_shapes_and_range_no_letterbox(tmp_path):
    # Create a wide image: left half red, right half green.
    w, h = 512, 256
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2, 0] = 255  # red
    arr[:, w // 2 :, 1] = 255  # green
    img = Image.fromarray(arr, mode="RGB")
    path = _save_image(img, tmp_path / "rg.png")

    cfg = SampleCfg(image_path=path, input_name="data", input_shape=(1, 3, 224, 224))
    harn = ClassificationHarness()
    inputs = harn.make_inputs(cfg)

    assert "data" in inputs
    x = inputs["data"]
    assert x.dtype == np.float32
    assert x.shape == (1, 3, 224, 224)
    assert np.all(np.isfinite(x))

    # Verify we did a center crop (no padding/letterbox):
    # left edge should be red-ish, right edge green-ish after normalization.
    red_left = float(x[0, 0, :, 0].mean())
    green_left = float(x[0, 1, :, 0].mean())
    red_right = float(x[0, 0, :, -1].mean())
    green_right = float(x[0, 1, :, -1].mean())

    assert red_left > 1.0
    assert green_left < 0.0
    assert green_right > 1.0
    assert red_right < 0.0


def test_preprocessing_normalization_known_pixel(tmp_path):
    # Solid black image -> (0/255 - mean)/std.
    img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), mode="RGB")
    path = _save_image(img, tmp_path / "black.png")

    cfg = SampleCfg(image_path=path, input_name="data", input_shape=(1, 3, 224, 224))
    harn = ClassificationHarness()
    x = harn.make_inputs(cfg)["data"]

    expected = np.array(
        [
            -0.485 / 0.229,
            -0.456 / 0.224,
            -0.406 / 0.225,
        ],
        dtype=np.float32,
    )

    got = x[0, :, 0, 0]
    assert np.allclose(got, expected, rtol=0, atol=1e-6)


def test_accuracy_proxy_metrics():
    harn = ClassificationHarness(eps=1e-4, cosine_threshold=0.9)
    ref = [np.array([1.0, 2.0, 3.0], dtype=np.float32)]
    out = [np.array([1.0, 2.0, 4.0], dtype=np.float32)]

    res = harn.accuracy_proxy(ref, out)
    assert math.isclose(res["max_abs"], 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(res["mean_abs"], 1.0 / 3.0, rel_tol=0.0, abs_tol=1e-12)

    # Expected cosine similarity.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    expected_cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert math.isclose(res["cosine_similarity"], expected_cos, rel_tol=0.0, abs_tol=1e-12)


def test_postprocess_topk_and_labels_and_fallback(monkeypatch):
    # Use a small, injected label list to keep the test independent of the 1000-class file.
    labels = ["zero", "one", "two", "three"]
    harn = ClassificationHarness(topk=3, labels=labels)
    logits = np.array([0.0, 10.0, -1.0, 9.0], dtype=np.float32)
    out = postprocess_result_to_dict(harn.postprocess({"logits": logits}, context={}))

    assert out["task"] == "classification"
    topk = out["json"]["topk"]
    assert [d["id"] for d in topk] == [1, 3, 0]
    assert [d["label"] for d in topk] == ["one", "three", "zero"]
    assert topk[0]["p"] > topk[1]["p"] > topk[2]["p"]

    # Now force missing labels -> fallback to stringified IDs.
    harn2 = ClassificationHarness(topk=2, labels=None)
    monkeypatch.setattr(harn2, "_load_imagenet_labels", lambda: None)
    out2 = postprocess_result_to_dict(harn2.postprocess({"logits": logits}, context={}))
    assert [d["label"] for d in out2["json"]["topk"]] == ["1", "3"]
