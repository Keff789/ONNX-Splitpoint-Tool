"""The existing sealed Part2 bridge consumes physical bytes in Generic Quality."""
from __future__ import annotations

import ast
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.runners import generic_quality_completion as quality
from onnx_splitpoint_tool.runners.backends.hailo_backend import _HailoSession
from onnx_splitpoint_tool.runners.native_split_quality_runtime import prepare_quality_first_boundary_input
from tests.test_generic_quality_existing_bridge import bridge_fixture


TEMPLATE = Path(__file__).parents[1] / "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt"
TARGET = (1, 4, 2, 3)


def _helper():
    names = {"prepare_generic_native_boundary_input", "_adapt_tensor", "_try_adapt_tensor"}
    nodes = [node for node in ast.parse(TEMPLATE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = {"np": np, "Mapping": Mapping,
                 "prepare_quality_first_boundary_input": prepare_quality_first_boundary_input}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TEMPLATE), "exec"), namespace)
    return namespace["prepare_generic_native_boundary_input"]


def _binding():
    return {"native_trt_meta": {"uint8_cast_bridge": {
        "input_name": "cut", "input_shape": list(TARGET),
        "boundary_layout": {"applied": True, "effective": "memory_nhwc_to_nchw",
                            "memory_shape": [1, 2, 3, 4], "perm": [0, 3, 1, 2]}}},
        "preselection": {"boundary_tensor_name": "cut", "boundary_layout": "memory_nhwc_to_nchw",
                         "boundary_transform": "uint8_dequant_then_layout",
                         "dequant_scale": .25, "dequant_zero_point": 17.0}}


def _canonical(raw):
    session = object.__new__(_HailoSession)
    session._hef_output_names = ["physical"]
    session._output_name_hef_to_canonical = {"physical": "cut"}
    session.output_shapes = {"cut": TARGET}
    session.attested_source_output_shapes = {}
    return session._canonical_outputs_from_slot_buffers({"physical": raw})["cut"]


@pytest.mark.parametrize("kind", ["raw_uint8", "dequantized_float", "float_layout"])
def test_generic_quality_uses_the_existing_physical_bridge_after_sdk_canonicalization(kind):
    raw = np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4)
    binding = _binding()
    target_dtype = np.uint8
    source = raw
    if kind == "dequantized_float":
        source = (raw.astype(np.float32) - 17) * .25
    elif kind == "float_layout":
        source = raw.astype(np.float32) / 7
        target_dtype = np.float32
        binding["preselection"]["boundary_transform"] = "layout_only"
    canonical = _canonical(source)
    events = []
    feed = _helper()("cut", canonical, TARGET, target_dtype,
        args=SimpleNamespace(_generic_quality_interface_binding=binding),
        _boundary_mode="canonical_float32" if kind == "float_layout" else "raw_uint8_hailo",
        _record_boundary_event=events.append,
        _cached_boundary_buffer=lambda _name, shape, dtype: np.empty(shape, dtype=dtype))
    expected = source if kind == "float_layout" else raw
    np.testing.assert_array_equal(feed, expected.reshape(TARGET))
    assert feed.flags.c_contiguous and feed.dtype == target_dtype
    assert not np.array_equal(feed, canonical)
    assert events[0]["action"] == "quality_first_internal_bridge_memory_order_preserved"
    # Execute the sealed graph's declared bridge; all canonical values survive.
    observed = feed.reshape(1, 2, 3, 4).transpose(0, 3, 1, 2)
    if kind != "float_layout":
        observed = (observed.astype(np.float32) - 17) * .25
        expected = (raw.transpose(0, 3, 1, 2).astype(np.float32) - 17) * .25
    else:
        expected = canonical
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("fault", ["input_name", "engine_shape", "layout", "permutation"])
def test_generic_quality_bridge_rejects_a_conflicting_live_contract(fault):
    binding = _binding()
    bridge = binding["native_trt_meta"]["uint8_cast_bridge"]
    if fault == "input_name":
        bridge["input_name"] = "foreign"
    elif fault == "engine_shape":
        bridge["input_shape"] = [1, 4, 3, 2]
    elif fault == "layout":
        bridge["boundary_layout"]["effective"] = "as_input"
    else:
        bridge["boundary_layout"]["perm"] = [0, 3, 3, 1]
    with pytest.raises(ValueError, match="bridge contract mismatch|boundary_layout_contract_invalid"):
        _helper()("cut", np.arange(24, dtype=np.uint8).reshape(TARGET), TARGET, np.uint8,
            args=SimpleNamespace(_generic_quality_interface_binding=binding),
            _boundary_mode="raw_uint8_hailo", _record_boundary_event=lambda _event: None,
            _cached_boundary_buffer=lambda _name, shape, dtype: np.empty(shape, dtype=dtype))


def test_native_binding_keeps_its_original_precedence_and_feed():
    binding = _binding()
    bridge = binding["native_trt_meta"]["uint8_cast_bridge"]
    selection = binding["preselection"]
    raw = np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4)
    args = SimpleNamespace(_native_split_quality_binding=binding,
        _native_split_boundary_layout=bridge["boundary_layout"], _native_split_boundary_input_name="cut",
        _native_split_boundary_transform=selection["boundary_transform"],
        _native_split_boundary_dequant_scale=selection["dequant_scale"],
        _native_split_boundary_dequant_zero_point=selection["dequant_zero_point"],
        _generic_quality_interface_binding={"conflicting": "unused"})
    feed = _helper()("cut", _canonical(raw), TARGET, np.uint8, args=args,
        _boundary_mode="raw_uint8_hailo", _record_boundary_event=lambda _event: None,
        _cached_boundary_buffer=lambda _name, shape, dtype: np.empty(shape, dtype=dtype))
    np.testing.assert_array_equal(feed, raw.reshape(TARGET))


def test_plain_generic_feed_remains_canonical_without_a_verified_bridge():
    source = np.arange(24, dtype=np.uint8).reshape(TARGET)
    feed = _helper()("cut", source, TARGET, np.uint8, args=SimpleNamespace(),
        _boundary_mode="raw_uint8_hailo", _record_boundary_event=lambda _event: None,
        _cached_boundary_buffer=lambda _name, shape, dtype: np.empty(shape, dtype=dtype))
    np.testing.assert_array_equal(feed, source)


def test_only_actual_local_eight_file_validation_supplies_the_runtime_bridge(tmp_path):
    fixture, bound, _ = bridge_fixture(tmp_path)
    original_bytes = fixture["path"].read_bytes()
    metadata = quality.load_quality_completion_config(fixture["path"], check_files=False)
    assert "_verified_generic_interface_binding" not in metadata
    loaded = quality.load_quality_completion_config(fixture["path"], check_files=True)
    verified = loaded["_verified_generic_interface_binding"]
    assert verified["binding_sha256"] == bound["binding_sha256"]
    assert "native_split_quality_binding" not in loaded
    assert fixture["path"].read_bytes() == original_bytes
    forged = deepcopy(fixture["config"])
    forged["_verified_generic_interface_binding"] = verified
    fixture["path"].write_text(json.dumps(forged))
    with pytest.raises(ValueError, match="cannot claim Native authority"):
        quality.load_quality_completion_config(fixture["path"], check_files=True)
