from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool.deepx.config import (
    CLASSIFICATION_PREPROCESSING_CURRENT,
    CLASSIFICATION_PREPROCESSING_IMAGENET,
)
from onnx_splitpoint_tool.deepx.preprocessing_probe import (
    classification_input_tensor,
    load_probe_samples,
    resolve_image_input_contract,
    select_classification_logits,
    topk,
)


def test_numeric_arms_differ_only_by_imagenet_normalization() -> None:
    rgb = np.asarray([[[0, 127, 255], [255, 64, 32]]], dtype=np.uint8)
    current = classification_input_tensor(
        rgb, mode=CLASSIFICATION_PREPROCESSING_CURRENT, layout="NCHW",
    )
    corrected = classification_input_tensor(
        rgb, mode=CLASSIFICATION_PREPROCESSING_IMAGENET, layout="NCHW",
    )
    assert current.shape == corrected.shape == (1, 3, 1, 2)
    expected_current = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
    expected_corrected = (
        expected_current
        - np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    ) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    np.testing.assert_allclose(current, expected_current, rtol=0, atol=1e-7)
    np.testing.assert_allclose(corrected, expected_corrected, rtol=0, atol=1e-6)


def test_input_layout_and_logits_contracts_are_fail_closed() -> None:
    assert resolve_image_input_contract([1, 3, 224, 224]) == ("NCHW", (224, 224))
    assert resolve_image_input_contract([1, 224, 224, 3]) == ("NHWC", (224, 224))
    with pytest.raises(ValueError, match="static 3-channel"):
        resolve_image_input_contract([1, 4, 224, 224])
    logits = select_classification_logits([np.asarray([[0.0, 0.1, 0.1, 0.8, 0.2]])])
    assert topk(logits, 5) == [3, 4, 1, 2, 0]
    with pytest.raises(ValueError, match="exactly one"):
        select_classification_logits([np.zeros((1, 5)), np.zeros((1, 5))])


def test_suite_manifest_identity_and_optional_content_verification(tmp_path: Path) -> None:
    image = tmp_path / "images" / "n00000001" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-decoded-by-this-test")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "onnx-splitpoint/classification-validation-manifest",
        "schema_version": 2,
        "samples": [{
            "image": "images/n00000001/sample.jpg",
            "sample_id": "n00000001/sample.jpg",
            "label_id": 7,
            "source_sha256": digest,
        }],
    }), encoding="utf-8")
    samples, identity = load_probe_samples(
        manifest, expected_count=1, verify_content=True,
    )
    assert samples[0].path == image.resolve()
    assert samples[0].label_id == 7
    assert identity["sample_count"] == 1
    assert identity["content_verified"] is True
    assert len(identity["ordered_sample_identity_sha256"]) == 64

    tampered = json.loads(manifest.read_text(encoding="utf-8"))
    tampered["samples"][0]["source_sha256"] = "0" * 64
    manifest.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_probe_samples(manifest, expected_count=1, verify_content=True)


def test_manifest_never_silently_reselects_or_accepts_duplicate_ids(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "samples": [
            {"image": "a.jpg", "sample_id": "same", "label_id": 0},
            {"image": "b.jpg", "sample_id": "same", "label_id": 1},
        ]
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicated"):
        load_probe_samples(manifest, expected_count=2)
    unique = json.loads(manifest.read_text(encoding="utf-8"))
    unique["samples"][1]["sample_id"] = "different"
    manifest.write_text(json.dumps(unique), encoding="utf-8")
    with pytest.raises(ValueError, match="count mismatch"):
        load_probe_samples(
            manifest,
            expected_count=1,
        )
