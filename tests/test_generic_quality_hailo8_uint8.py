"""The H8 observer follows the already declared normal SDK input variant."""
from copy import deepcopy
import json

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from test_generic_quality_hailo8 import h8_config, h8_capture, normal_adapter


def uint8_config(tmp_path, *, raw=False):
    fixture = h8_config(tmp_path, raw=raw)
    fixture["config"]["identity"]["precision"] = "uint8_dequant_fp16"
    fixture["config"]["representative_input"]["runtime_input_adapter"]["runtime_input_format"] = "uint8"
    fixture["path"].write_text(json.dumps(fixture["config"]))
    return fixture


@pytest.mark.parametrize("raw", [False, True])
def test_uint8_normal_capture_and_portable_proof_preserve_exact_input_bytes(tmp_path, raw):
    fixture = uint8_config(tmp_path, raw=raw)
    original = fixture["inputs"]["images"].copy()
    feeds = fixture["inputs"]
    actual = normal_adapter()({"task": "detection", "input_binding": {"runtime_input_format": "uint8"}}, "hailo8", feeds)
    assert actual is feeds and actual["images"] is feeds["images"]
    evidence, records = h8_capture(fixture)
    comparison = evidence["per_image"][0]["representative_input_comparison"]
    prepared = comparison["prepared_tensor"]
    observed = comparison["actual_runtime_input"]["images"]
    assert prepared["dtype"] == observed["dtype"] == "uint8"
    assert prepared["tensor_sha256"] == observed["tensor_sha256"]
    assert np.array_equal(fixture["inputs"]["images"], original)
    quality._validate_evidence(evidence, records, fixture["config"]["completion_contract"],
                               fixture["config"]["expected_runtime_artifacts"])


def test_n5000_uint8_metadata_preserves_frozen_ids_and_precision(tmp_path):
    fixture = uint8_config(tmp_path)
    ids = [f"{index:012d}.jpg" for index in range(5000)]
    fixture["config"]["expected_image_ids"] = ids
    fixture["path"].write_text(json.dumps(fixture["config"]))
    config = quality.load_quality_completion_config(fixture["path"], check_files=False)
    assert config["expected_image_ids"] == ids
    assert config["identity"]["precision"] == "uint8_dequant_fp16"
    assert config["representative_input"]["runtime_input_adapter"]["runtime_input_format"] == "uint8"


@pytest.mark.parametrize("fault", ["missing", "backend", "function", "task", "format", "extra",
                                   "precision", "unknown_precision", "prepared_dtype", "prepared_format"])
def test_inconsistent_or_missing_declared_uint8_adapter_rejects(tmp_path, fault):
    fixture = uint8_config(tmp_path)
    representative = fixture["config"]["representative_input"]
    adapter = representative["runtime_input_adapter"]
    if fault == "missing": representative.pop("runtime_input_adapter")
    elif fault == "backend": adapter["backend"] = "hailo10h"
    elif fault == "function": adapter["function"] = "other_adapter"
    elif fault == "task": adapter["task"] = "classification"
    elif fault == "format": adapter["runtime_input_format"] = "float32"
    elif fault == "extra": adapter["normalization"] = "none"
    elif fault == "precision": fixture["config"]["identity"]["precision"] = "float32_layout_fp16"
    elif fault == "unknown_precision": fixture["config"]["identity"]["precision"] = "unknown"
    elif fault == "prepared_dtype": representative["prepared_input"]["dtype"] = "float32"
    else: representative["prepared_input"]["format"] = "unknown"
    fixture["path"].write_text(json.dumps(fixture["config"]))
    with pytest.raises(ValueError):
        quality.load_quality_completion_config(fixture["path"], check_files=False)


@pytest.mark.parametrize("fault", ["float32", "altered_uint8"])
def test_actual_input_must_match_uint8_passthrough(tmp_path, fault):
    fixture = uint8_config(tmp_path)
    actual = fixture["inputs"]["images"].copy()
    if fault == "float32": actual = actual.astype(np.float32) / np.float32(255.0)
    else: actual.flat[0] ^= 1
    with pytest.raises(ValueError, match="normal timed input relation"):
        h8_capture(fixture, actual={"images": actual})


def test_resealed_uint8_proof_cannot_claim_a_float32_conversion(tmp_path):
    fixture = uint8_config(tmp_path)
    evidence, records = h8_capture(fixture)
    forged = deepcopy(evidence)
    comparison = forged["per_image"][0]["representative_input_comparison"]
    wrong = quality._input_snapshot({"images": fixture["inputs"]["images"].astype(np.float32) / np.float32(255.0)})
    comparison["runtime_input_adapter_proof"]["expected_runtime_input"] = wrong
    comparison["actual_runtime_input"] = wrong
    forged["per_image"][0]["runtime_inputs"] = wrong
    forged.pop("evidence_sha256")
    forged["evidence_sha256"] = quality._hash(forged)
    with pytest.raises(ValueError, match="uint8 passthrough relation"):
        quality._validate_evidence(forged, records, fixture["config"]["completion_contract"],
                                   fixture["config"]["expected_runtime_artifacts"])
