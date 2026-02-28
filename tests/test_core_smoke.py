from __future__ import annotations

from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from onnx_splitpoint_tool.api import analyze_model


def _write_tiny_onnx(path: Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])

    w_data = [1.0, 0.0, 0.0, 1.0] * 4
    w = helper.make_tensor("W", TensorProto.FLOAT, [4, 4], w_data)

    mm = helper.make_node("MatMul", ["x", "W"], ["mm_out"], name="matmul")
    relu = helper.make_node("Relu", ["mm_out"], ["y"], name="relu")

    graph = helper.make_graph([mm, relu], "tiny", [x], [y], initializer=[w])
    model = helper.make_model(graph)
    onnx.save(model, str(path))


def test_core_analyze_model_smoke(tmp_path: Path) -> None:
    model_path = tmp_path / "tiny.onnx"
    _write_tiny_onnx(model_path)

    result = analyze_model(str(model_path))

    assert isinstance(result.get("candidate_bounds"), list)
    assert len(result["candidate_bounds"]) > 0
    assert len(result.get("costs_bytes", [])) > 0
    assert len(result.get("peak_act_mem_max_bytes", [])) == len(result.get("costs_bytes", []))
    assert "flops_left_prefix" in result
