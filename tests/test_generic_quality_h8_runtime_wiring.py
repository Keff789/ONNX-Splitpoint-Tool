"""Normal Generic Quality UINT8 wiring and durable failed-capture evidence."""
import ast
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from tests.test_generic_quality_existing_bridge import bridge_fixture
from tests.test_generic_quality_hailo8 import RUNNER
from tests.test_generic_quality_hailo8_uint8 import uint8_config


TREE = ast.parse(RUNNER.read_text())
HELPERS = {
    "_completion_hailo_inputs", "_completion_hailo_quantized_inputs",
    "_generic_quality_hailo8_input_adapter", "_generic_quality_hailo8_uint8_feed",
    "_finish_generic_quality_capture", "_hailo_stage1_interface_float_map",
    "_canonical_image_preprocessing_contract", "_image_preprocessing_contract_sha256",
    "_resolve_image_preprocessing_contract", "_hailo10_native_io",
    "_hailo_image_quantized_inputs_default", "_configure_quality_source_only",
}


def runner_helpers():
    nodes = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in HELPERS]
    assert {node.name for node in nodes} == HELPERS
    namespace = {"np": np, "Mapping": Mapping, "Path": Path, "json": json,
                 "os": os, "re": re, "hashlib": hashlib}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace


def nested_function(namespace, name, *, last=False):
    nodes = [node for node in ast.walk(TREE) if isinstance(node, ast.FunctionDef) and node.name == name]
    node = nodes[-1] if last else nodes[0]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace[name]


def test_strict_verified_bridge_is_available_only_after_local_file_guard(tmp_path):
    fixture, bound, _ = bridge_fixture(tmp_path)
    loaded = quality.load_quality_completion_config(fixture["path"], check_files=True)
    assert loaded["_verified_generic_interface_binding"] == bound
    assert "native_split_quality_binding" not in loaded
    assert "_verified_generic_interface_binding" not in quality.load_quality_completion_config(
        fixture["path"], check_files=False)


def test_original_bound_quantization_interface_ABI_and_invalid_scale():
    historical = json.loads((Path(__file__).parent / "fixtures/generic_quality_existing_bridge_metadata.json").read_text())
    bound = next(row["binding"] for row in historical["pairs"]
                 if row["model_id"] == "yolo26s" and row["case_id"] == "b003")
    helper = runner_helpers()["_hailo_stage1_interface_float_map"]
    metadata = bound["boundary_metadata_payload"]["boundary_tensor"]
    tensor = np.full(metadata["canonical_part2_shape"], 17, dtype=np.uint8)
    name = metadata["name"]
    converted = helper({name: tensor}, verified_binding=bound)
    quant = bound["boundary_contract"]
    assert np.array_equal(converted[name],
        (tensor.astype(np.float32) - np.float32(quant["dequant_zero_point"])) * np.float32(quant["dequant_scale"]))
    # The same existing interface ABI remains valid for genuine Native callers.
    assert np.array_equal(helper({name: tensor}, verified_binding=bound)[name], converted[name])
    damaged = deepcopy(bound)
    damaged["boundary_contract"]["dequant_scale"] *= 2
    with pytest.raises(ValueError, match="quantization_parameters_invalid"):
        helper({name: tensor}, verified_binding=damaged)


@pytest.mark.parametrize("fault", ["quantization", "HEF", "scope", "unverified_runtime_field", "Native_flag"])
def test_generic_interface_binding_cannot_bypass_original_guards(tmp_path, fault):
    fixture, _, paths = bridge_fixture(tmp_path)
    cfg = fixture["config"]
    if fault == "quantization":
        path = Path(cfg["existing_part2_bridge"]["binding_source"]["path"])
        payload = json.loads(path.read_text())
        payload["native_split_quality_binding"]["boundary_contract"]["dequant_scale"] *= 2
        path.write_text(json.dumps(payload))
    elif fault == "HEF":
        paths["part1_runtime"].write_bytes(b"different HEF")
    elif fault == "scope":
        cfg["identity"]["setup_id"] = "another_physical_setup"
    elif fault == "unverified_runtime_field":
        cfg["_verified_generic_interface_binding"] = {"forged": True}
    else:
        cfg["native_split_quality_binding"] = {"forged": True}
    fixture["path"].write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        quality.load_quality_completion_config(fixture["path"], check_files=True)


def test_uint8_dataset_feed_uses_each_real_image_and_reaches_original_infer(tmp_path):
    fixture = uint8_config(tmp_path)
    cfg = quality.load_quality_completion_config(fixture["path"], check_files=False)
    height, width = cfg["completion_contract"]["processor_contract"]["input_hw"]
    second = fixture["images"][1]
    Image.new("RGB", (60, 80), (91, 17, 203)).save(second)
    ns = runner_helpers()
    class Session:
        input_names = ["images"]
        def infer(self, feeds):
            assert feeds["images"].dtype == np.uint8
            return feeds
        def quantize_input(self, *args):
            pytest.fail("the UINT8 RGB contract must not enter the H10 QuantInfo converter")
    session = Session()
    ns.update(quality_h8_uint8_inputs=True, generic_quality_config=cfg,
        quality_h8_input_adapter=ns["_generic_quality_hailo8_input_adapter"](cfg),
        hailo_p1=session, generic_completion_config=None, p1_inputs_hailo=None,
        base_static_full_inputs={}, image_input_name="images", image_input_dtype=np.float32,
        img_hw=(height, width), resolved_image_scale="norm", val_letterbox=True,
        validation_preprocessing_contract={}, p1_inputs=["images"], stage1_tok="hailo8",
        _load_image_as_nchw=lambda *a, **k: np.zeros((1, 3, height, width), np.float32))
    prepare = nested_function(ns, "_build_feeds_for_validation_image", last=True)
    infer = nested_function(ns, "run_stage1_map")
    prepared = []
    for image in fixture["images"]:
        full, p1, _ = prepare(image)
        assert full["images"].dtype == np.float32
        actual = infer(p1)["images"]
        assert actual.flags.c_contiguous and actual.shape == (height, width, 3)
        prepared.append(actual)
    assert np.array_equal(prepared[0][height//2, width//2], [23, 53, 73])
    assert np.array_equal(prepared[1][height//2, width//2], [91, 17, 203])
    assert np.array_equal(prepared[0][0, 0], [114, 114, 114])
    assert not np.array_equal(prepared[0], prepared[1])


def test_uint8_initial_feed_uses_resolved_source_image(tmp_path):
    fixture = uint8_config(tmp_path)
    cfg = quality.load_quality_completion_config(fixture["path"], check_files=False)
    ns = runner_helpers()
    ns.update(quality_h8_uint8_inputs=True, generic_quality_config=cfg,
              hailo_p1=SimpleNamespace(input_names=["images"]),
              p_img=fixture["images"][0], args=SimpleNamespace(image="default"))
    block = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == "quality_h8_uint8_inputs"
        and any(isinstance(child, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "p1_inputs_hailo"
                        for target in child.targets) for child in node.body))
    module = ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[]))
    code = compile(module, str(RUNNER), "exec")
    exec(code, ns)
    prepared = ns["p1_inputs_hailo"]["images"]
    assert prepared.dtype == np.uint8
    assert np.array_equal(prepared[prepared.shape[0]//2, prepared.shape[1]//2], [23, 53, 73])
    ns["p_img"] = None
    with pytest.raises(ValueError, match="resolved source image"):
        exec(code, ns)


def test_float32_FPS_and_native_session_defaults_are_unchanged(monkeypatch):
    ns = runner_helpers()
    monkeypatch.delenv("ONNX_SPLITPOINT_HAILO_QUANTIZED_IMAGE_INPUTS", raising=False)
    monkeypatch.delenv("SPLITPOINT_HAILO_QUANTIZED_IMAGE_INPUTS", raising=False)
    assert ns["_generic_quality_hailo8_input_adapter"](None) is None
    assert ns["_generic_quality_hailo8_input_adapter"]({"identity": {"comparison_backend": "hailo10h"}}) is None
    assert ns["_hailo_image_quantized_inputs_default"]("hailo8") is False
    assert ns["_hailo_image_quantized_inputs_default"]("hailo10h") is True
    assert ns["_completion_hailo_quantized_inputs"]({"input_binding": {"runtime_input_format": "uint8"}}) is True
    assert ns["_completion_hailo_quantized_inputs"]({"input_binding": {"runtime_input_format": "float32"}}) is False
    array = np.array([1., 2.], np.float32)
    assert ns["_hailo_stage1_interface_float_map"]({"cut": array})["cut"] is array


def test_generic_quality_cannot_enable_native_flag():
    ns = runner_helpers()
    args = SimpleNamespace(quality_source_only=True, generic_quality_completion_config="quality.json",
        generic_completion_config="", variants="composed", native_split_quality_binding="native.json")
    class Parser:
        def error(self, message):
            raise ValueError(message)
    with pytest.raises(ValueError, match="exactly one Native binding or Generic"):
        ns["_configure_quality_source_only"](args, Parser())


def capture_clock(monkeypatch, completed=1):
    runtime = object.__new__(quality.GenericQualityCompletion)
    runtime.entries = {str(index): {} for index in range(completed)}
    runtime.config = {"expected_image_ids": ["a.png", "b.png"]}
    runtime._capture_started_ns = 1_000_000_000
    runtime._capture_timing = runtime._capture_timing_attempt = None
    monkeypatch.setattr(quality.time, "perf_counter_ns", lambda: 2_000_000_000)
    return runtime


def test_failed_capture_preserves_original_image_error_and_observed_count(tmp_path, monkeypatch):
    runtime = capture_clock(monkeypatch)
    rows = [{"image": "a.png"}, {"image": "b.png", "error": "ValueError: original UINT8 relation mismatch"}]
    finish = runner_helpers()["_finish_generic_quality_capture"]
    with pytest.raises(ValueError, match="scope/count/monotonic"):
        finish(runtime, rows, tmp_path)
    diagnostic = json.loads((tmp_path / "generic_quality_capture_failure.json").read_text())
    assert diagnostic["valid_capture"] is False
    assert diagnostic["expected_image_count"] == diagnostic["attempted_image_count"] == 2
    assert diagnostic["completed_image_count"] == diagnostic["image_error_count"] == 1
    assert diagnostic["first_image_errors"] == [rows[1]]
    assert diagnostic["unvalidated_capture_timing_attempt"]["image_count"] == 1
    assert runtime._capture_timing is None


def test_diagnostic_io_failure_keeps_original_capture_guard(tmp_path, monkeypatch):
    runtime = capture_clock(monkeypatch)
    with pytest.raises(ValueError, match="scope/count/monotonic"):
        runner_helpers()["_finish_generic_quality_capture"](runtime, [], tmp_path / "missing")


def test_successful_capture_is_unchanged_and_failure_diagnostic_is_bounded(tmp_path, monkeypatch):
    runtime = capture_clock(monkeypatch, completed=2)
    timing = runner_helpers()["_finish_generic_quality_capture"](runtime, [{}, {}], tmp_path)
    assert timing["image_count"] == 2 and timing["performance_measurement_executed"] is False
    assert not (tmp_path / "generic_quality_capture_failure.json").exists()
    rows = [{"image": str(index), "error": "original " + "x" * 4096} for index in range(5000)]
    diagnostic = runtime.capture_failure_diagnostic(rows, ValueError("count"))
    assert diagnostic["image_error_count"] == 5000
    assert len(diagnostic["first_image_errors"]) == 8 and diagnostic["image_errors_truncated"]
    assert len(json.dumps(diagnostic)) < 20000
