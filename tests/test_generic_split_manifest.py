"""Exact observed b177 staging relation and scientific-drift denials."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.generic_split_manifest import validate_generic_split_manifest_staging


FIXTURE = Path(__file__).parent / "fixtures/generic_split_manifest"


def bindings():
    args = json.loads((FIXTURE / "b177_bindings.json").read_text())
    return (FIXTURE / "b177_original.json").read_bytes(), (FIXTURE / "b177_staged.json").read_bytes(), args


def rewritten(document, args):
    raw = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    args["staged_source"].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    return raw


def test_actual_normal_staging_preserves_both_file_identities_and_all_bindings():
    original, staged, args = bindings()
    unchanged = deepcopy(args)
    proof = validate_generic_split_manifest_staging(original, staged, **args)
    assert args == unchanged
    assert proof["original_manifest"] == args["original_source"]
    assert proof["staged_manifest"] == args["staged_source"]
    assert proof["original_manifest"]["sha256"] != proof["staged_manifest"]["sha256"]
    assert proof["original_boundary_metadata_artifact"]["sha256"] == args["original_source"]["sha256"]
    assert set(proof["unchanged_artifact_bindings"]) == set(args["original_artifacts"])
    assert proof["prepared_input"]["sha256"] == args["row"]["functional_reference"]["input_sha256"]
    assert proof["scientific_manifest_fields_unchanged"] is True
    paths = {item["pointer"] for item in proof["allowed_differences"]}
    assert "/hailo/hefs/hailo10h" in paths
    assert "/hailo/hefs/hailo8/part1_build/source_onnx_path" in paths
    assert "/deepx/part1_dxnn/cache_lookup/outcome" in paths


@pytest.mark.parametrize("path,value", [
    (("boundary",), 178),
    (("part2",), "another_graph.onnx"),
    (("full_model",), "../models/another_model.onnx"),
    (("cut", "names_part1"), ["different_tensor"]),
    (("io", "part2_external_inputs"), ["different_input"]),
    (("pipeline", "stage2"), "another_stage"),
    (("hailo", "config", "calib_count"), 501),
    (("hailo", "config", "build", "full"), True),
    (("hailo", "hefs", "hailo8", "part1"), "different.hef"),
    (("hailo", "hefs", "hailo8", "part1_build", "source_onnx_path"), "/foreign/b177/yolo11l_part1_b177.onnx"),
    (("hailo", "hefs", "hailo8", "part1_build", "cache_hit"), False),
    (("deepx", "part1_dxnn", "dxnn_sha256"), "f" * 64),
    (("deepx", "part1_dxnn", "cache_contract", "input_shape"), [1, 3, 320, 320]),
    (("deepx", "part1_dxnn", "cache_contract", "calibration_count"), 501),
    (("unexpected_scientific_metadata",), {"source": "unreviewed"}),
])
def test_descriptor_valid_scientific_or_unreviewed_metadata_drift_is_rejected(path, value):
    original, staged, args = bindings()
    document = json.loads(staged)
    parent = document
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


def test_replacement_of_unused_full_contract_is_not_a_removal_exception():
    original, staged, args = bindings()
    document = json.loads(staged)
    document["hailo"]["hefs"]["hailo8"]["full_output_contract"] = {"artifact_sha256": "f" * 64}
    with pytest.raises(ValueError, match="unreviewed metadata or scientific"):
        validate_generic_split_manifest_staging(original, rewritten(document, args), **args)


@pytest.mark.parametrize("fault", [
    lambda a: a["row"]["identity"].update(comparison_backend="hailo10h"),
    lambda a: a["row"].update(admission_mode="paired"),
    lambda a: a["row"]["generic_artifact_binding"]["artifacts"]["boundary_metadata"].update(sha256="f" * 64),
    lambda a: a["row"]["generic_artifact_binding"]["artifacts"]["engine"].update(sha256="f" * 64),
    lambda a: a["row"]["generic_artifact_binding"]["artifacts"]["part1_runtime"].update(size_bytes=1),
    lambda a: a["producer_model"].update(part2_onnx_sha256="f" * 64),
    lambda a: a.update(full_model_sha256="f" * 64),
    lambda a: a["boundary_inputs"][0].update(shape=[1, 1]),
    lambda a: a["row"]["input_binding"]["prepared_input"].update(sha256="f" * 64),
    lambda a: a["row"]["input_binding"].update(runtime_input_format="uint8"),
])
def test_actual_artifact_graph_boundary_input_and_scope_bindings_are_required(fault):
    original, staged, args = bindings()
    fault(args)
    with pytest.raises(ValueError, match="split-manifest staging"):
        validate_generic_split_manifest_staging(original, staged, **args)


@pytest.mark.parametrize("which", ["original_source", "staged_source"])
def test_manifest_byte_descriptor_cannot_be_rewritten_or_omitted(which):
    original, staged, args = bindings()
    args[which]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bytes differ from descriptor"):
        validate_generic_split_manifest_staging(original, staged, **args)
