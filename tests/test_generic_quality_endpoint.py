"""Explicit composed source roles; actual file hashes, no inference or devices."""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from onnx_splitpoint_tool import native_detection_postprocess as detection
from onnx_splitpoint_tool.runners import generic_quality_endpoint as endpoint
from test_v272_detection_completion_runtime import _decoded_source


def artifact(path, content):
    path.write_bytes(content)
    return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}


def fixture(tmp_path):
    full = artifact(tmp_path / "full.onnx", b"test-only full source ONNX bytes")
    part2 = artifact(tmp_path / "part2.onnx", b"test-only distinct composed Part2 ONNX bytes")
    outputs = {"output0": np.zeros((1, 300, 6), dtype=np.float32)}
    authority = _decoded_source(outputs)
    authority.update(full_model_sha256=full["sha256"], terminal_model_sha256=part2["sha256"])
    fields = endpoint.project_generic_quality_endpoint(authority, full_model=full, part2_source=part2)
    config = {**fields, "expected_runtime_artifacts": {"source_part2_onnx": part2}}
    return full, part2, outputs, authority, config


def test_b039_shaped_roles_preserve_original_authority_and_existing_native_guard(tmp_path):
    full, part2, outputs, authority, config = fixture(tmp_path)
    original = deepcopy(authority)
    with pytest.raises(detection.FrozenPostprocessError, match="model_sha256_conflict"):
        detection.build_detection_completion_execution_contract(model_id="yolo26s", outputs=outputs,
            input_hw=[640, 640], original_wh=[640, 426], source_endpoint_contract=authority,
            preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114})
    checked = endpoint.verify_generic_quality_endpoint_projection(config, check_files=True, full_model_path=full["path"])
    contract = detection.build_detection_completion_execution_contract(model_id="yolo26s", outputs=outputs,
        input_hw=[640, 640], original_wh=[640, 426], source_endpoint_contract=config["source_endpoint_contract"],
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114})
    assert authority == original == config["source_endpoint_authority"]
    assert checked["part2_source"]["sha256"] == part2["sha256"]
    assert contract["source_endpoint"]["model_sha256"] == full["sha256"]
    assert config["source_endpoint_contract"] == {key: value for key, value in original.items() if key != "terminal_model_sha256"}


@pytest.mark.parametrize("fault", ["missing", "raw", "semantic", "part2", "full", "bytes", "nested"])
def test_projection_rejects_missing_roles_and_source_substitution(tmp_path, fault):
    full, part2, outputs, authority, config = fixture(tmp_path)
    kwargs = {"check_files": True, "full_model_path": full["path"]}
    if fault == "missing":
        config.pop("source_endpoint_authority"); config.pop("source_endpoint_projection")
    elif fault == "raw":
        config["source_endpoint_authority"]["terminal_model_sha256"] = "a" * 64
    elif fault == "semantic":
        config["source_endpoint_contract"]["output_endpoint_attestation"]["status"] = "failed"
    elif fault == "part2":
        config["expected_runtime_artifacts"]["source_part2_onnx"]["sha256"] = "b" * 64
    elif fault == "full":
        kwargs["full_model_path"] = part2["path"]
    elif fault == "bytes":
        from pathlib import Path
        Path(part2["path"]).write_bytes(b"changed source")
    else:
        config["source_endpoint_authority"]["output_endpoint_attestation"]["declared_contract"]["model_sha256"] = part2["sha256"]
    with pytest.raises(ValueError):
        endpoint.verify_generic_quality_endpoint_projection(config, **kwargs)


def test_creation_requires_independently_verified_full_and_part2_files(tmp_path):
    full, part2, outputs, authority, config = fixture(tmp_path)
    full["sha256"] = "e" * 64
    with pytest.raises(ValueError, match="source bytes changed"):
        endpoint.project_generic_quality_endpoint(authority, full_model=full, part2_source=part2)


def test_only_unidentified_synthetic_fixtures_can_omit_projection(tmp_path):
    outputs = {"output0": np.zeros((1, 300, 6), dtype=np.float32)}
    config = {"source_endpoint_contract": _decoded_source(outputs)}
    assert endpoint.verify_generic_quality_endpoint_projection(config) is None
    config["source_endpoint_contract"]["full_model_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="source-role authority missing"):
        endpoint.verify_generic_quality_endpoint_projection(config)
