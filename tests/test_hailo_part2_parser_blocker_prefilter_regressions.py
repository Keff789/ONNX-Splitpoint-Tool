from __future__ import annotations

import pytest

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper

from onnx_splitpoint_tool.hailo_backend import (
    format_hailo_part2_parser_blocker_error,
    hailo_part2_parser_blocker_precheck_from_model,
)


def _make_part2_model_with_parser_blockers() -> onnx.ModelProto:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 4])
    k = helper.make_tensor('k', TensorProto.INT64, [1], [1])
    idx = helper.make_tensor_value_info('idx', TensorProto.INT64, [1, 4, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 1])
    nodes = [
        helper.make_node('Transpose', ['x'], ['t2'], perm=[0, 2, 1], name='/model.23/Transpose_2'),
        helper.make_node('Transpose', ['t2'], ['t3'], perm=[0, 2, 1], name='/model.23/Transpose_3'),
        helper.make_node('TopK', ['t3', 'k'], ['values', 'indices'], axis=-1, largest=1, sorted=1, name='/model.23/TopK'),
        helper.make_node('GatherElements', ['t3', 'indices'], ['g'], axis=-1, name='/model.23/GatherElements'),
        helper.make_node('Identity', ['indices'], ['idx'], name='/model.23/IdentityIdx'),
        helper.make_node('Identity', ['g'], ['y'], name='/model.23/IdentityOut'),
    ]
    graph = helper.make_graph(nodes, 'g', [x], [idx, y], initializer=[k])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])


def test_hailo_part2_parser_precheck_detects_yolo_style_blockers() -> None:
    model = _make_part2_model_with_parser_blockers()
    info = hailo_part2_parser_blocker_precheck_from_model(model)
    assert info['inspect_ok'] is True
    assert info['compatible'] is False
    assert 'TopK' in info['blocked_ops']
    assert 'GatherElements' in info['blocked_ops']
    assert str(info.get('blocked_prefix') or '').startswith('/model.23')
    assert any('Transpose_3' in s or 'Transpose_2' in s for s in list(info.get('suggested_end_nodes') or []))


def test_hailo_part2_parser_precheck_error_mentions_blocked_head() -> None:
    model = _make_part2_model_with_parser_blockers()
    info = hailo_part2_parser_blocker_precheck_from_model(model)
    msg = format_hailo_part2_parser_blocker_error(info)
    assert 'parser-blocking head' in msg
    assert 'blocked_prefix=' in msg
    assert 'TopK' in msg
