"""Timing selection must not depend on the case/backend visualization seed."""
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def suite():
    path = Path(__file__).resolve().parents[1] / "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt"
    loader = importlib.machinery.SourceFileLoader("fairness_timing_suite", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def test_frozen_manifest_order_wins_over_visual_sample_seed(suite, tmp_path):
    images = tmp_path / "validation"
    images.mkdir()
    for name in ("z.png", "a.png"):
        (images / name).write_bytes(name.encode())
    (images / "manifest.json").write_text(json.dumps({"samples": [
        {"image": "z.png", "image_id": 17}, {"image": "a.png", "image_id": 3},
    ]}))
    image, origin = suite._select_timing_sample_image(images, tmp_path, "detection")
    assert image == images / "z.png"
    assert origin == "frozen_validation_manifest_first_record"
    # The same shared selector is used by sealed Native-Full feeds.
    from argparse import Namespace
    assert (image, origin) == suite._deepx_find_prepared_feed_image(
        tmp_path, 640, "detection", {}, Namespace(validation_images=str(images)),
    )


def test_explicit_image_has_priority_and_missing_image_is_not_substituted(suite, tmp_path):
    selected = tmp_path / "exact.png"
    selected.write_bytes(b"exact")
    assert suite._select_timing_sample_image(None, tmp_path, "detection", str(selected))[0] == selected
    with pytest.raises(ValueError, match="timing_input_image_missing"):
        suite._select_timing_sample_image(None, tmp_path, "detection", "missing.png")


@pytest.mark.parametrize("task,preset", [("detection", "coco"), ("classification", "imagenet")])
def test_existing_suite_presets_remain_supported(suite, tmp_path, task, preset):
    image = tmp_path / "test_images" / ("test_image_" + preset + ".png")
    image.parent.mkdir()
    image.write_bytes(b"preset")
    assert suite._select_timing_sample_image(None, tmp_path, task, preset)[0] == image
