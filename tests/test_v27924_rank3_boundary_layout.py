from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx.reference import ReferenceEvaluator

from onnx_splitpoint_tool.native_split_quality import (
    canonical_json_sha256,
    known_native_split_policy,
    materialize_native_split_preselection,
    resolve_native_boundary_layout,
    validate_native_split_preselection,
)
from onnx_splitpoint_tool.runners.native_split_quality_runtime import prepare_quality_first_boundary_input
from scripts import native_trt_from_benchmarkset as builder


LAYOUT = 'memory_nwc_to_ncw'
TARGET = [1, 84, 8400]
PHYSICAL = [1, 8400, 84]


def _source(tmp_path: Path) -> Path:
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Identity', ['cut'], ['result'])], 'rank3_part2',
        [onnx.helper.make_tensor_value_info('cut', onnx.TensorProto.FLOAT, TARGET)],
        [onnx.helper.make_tensor_value_info('result', onnx.TensorProto.FLOAT, TARGET)],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 13)])
    source = tmp_path / 'part2.onnx'
    onnx.save(model, source)
    return source


def test_exact_overnight_rank3_shapes_resolve() -> None:
    assert resolve_native_boundary_layout(PHYSICAL, TARGET) == LAYOUT
    assert resolve_native_boundary_layout(TARGET, TARGET) == 'as_input'


@pytest.mark.parametrize('runtime,target', [
    ([1, 840, 840], TARGET),  # Equal element count is insufficient.
    ([8400, 84], TARGET),  # Undeclared missing batch dimension.
    ([2, 8400, 84], TARGET),
    ([1, 8400, 84], [1, 84, 0]),
    ([1, 84, 84], [1, 84, 84]),  # Same shape has two possible axis orders.
])
def test_rank3_layout_rejects_unproven_or_ambiguous_shapes(runtime, target) -> None:
    with pytest.raises(ValueError, match='native_split_quality_boundary_layout_'):
        resolve_native_boundary_layout(runtime, target)


@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('source_order', ['physical', 'canonical'])
def test_real_rank3_bridge_roundtrip_through_onnx(tmp_path: Path, quantized: bool, source_order: str) -> None:
    source = _source(tmp_path)
    scale, zero_point = .125, 93.
    raw = (np.arange(np.prod(PHYSICAL), dtype=np.uint32) % 256).astype(np.uint8).reshape(PHYSICAL)
    if quantized:
        bridge, meta = builder._make_uint8_dequant_bridge_onnx(
            source, tmp_path / 'bridge', root=tmp_path, case='b398',
            scale=scale, zero_point=zero_point, boundary_layout=LAYOUT,
        )
        expected = ((raw.astype(np.float32) - zero_point) * scale).transpose(0, 2, 1)
        source_array = raw if source_order == 'physical' else expected
        dtype, transform = np.uint8, 'uint8_dequant_then_layout'
    else:
        bridge, meta = builder._make_float32_layout_bridge_onnx(
            source, tmp_path / 'bridge', boundary_layout=LAYOUT,
        )
        physical_float = raw.astype(np.float32) / 32.
        expected = physical_float.transpose(0, 2, 1)
        source_array = physical_float if source_order == 'physical' else expected
        dtype, transform = np.float32, 'layout_only'
    layout = meta['boundary_layout']
    assert layout['memory_shape'] == PHYSICAL
    assert layout['perm'] == [0, 2, 1]
    assert layout['effective'] == LAYOUT
    assert meta['bridge_sha256'] == hashlib.sha256(bridge.read_bytes()).hexdigest()
    onnx.checker.check_model(onnx.load(bridge))
    feed = prepare_quality_first_boundary_input(
        source_array, target_shape=TARGET, target_dtype=dtype,
        boundary_layout=layout, boundary_transform=transform,
        dequant_scale=scale, dequant_zero_point=zero_point,
    )
    actual = ReferenceEvaluator(str(bridge)).run(None, {'cut': feed})[0]
    np.testing.assert_array_equal(actual, expected)
    assert feed.flags.c_contiguous
    assert list(actual.shape) == TARGET


@pytest.mark.parametrize('shape', [[1, 84, 84], [1, 84, 2, 4200], [1, 84, 0]])
def test_rank3_bridge_rejects_wrong_rank_or_ambiguous_shape(shape) -> None:
    with pytest.raises(RuntimeError):
        builder._make_boundary_layout_nodes('cut', 'cut', shape, LAYOUT)


def test_rank3_feed_rejects_same_element_count_wrong_source_shape() -> None:
    with pytest.raises(ValueError, match='boundary_source_shape_not_declared'):
        prepare_quality_first_boundary_input(
            np.zeros((1, 840, 840), np.float32), target_shape=TARGET,
            target_dtype=np.float32,
            boundary_layout={'applied': True, 'memory_shape': PHYSICAL,
                             'perm': [0, 2, 1], 'effective': LAYOUT},
            boundary_transform='layout_only',
        )


def test_rank3_layout_changes_existing_build_artifact_identity(tmp_path: Path) -> None:
    source = _source(tmp_path)
    _, before = builder._make_float32_layout_bridge_onnx(
        source, tmp_path / 'identity', boundary_layout='as_input',
    )
    _, after = builder._make_float32_layout_bridge_onnx(
        source, tmp_path / 'transpose', boundary_layout=LAYOUT,
    )
    assert before['source_sha256'] == after['source_sha256']
    assert before['bridge_sha256'] != after['bridge_sha256']
    assert before['boundary_layout']['effective'] == 'as_input'
    assert after['boundary_layout']['effective'] == LAYOUT


@pytest.mark.parametrize('model,case', [('yolo26m', 'b398'), ('yolo26s', 'b364')])
def test_rank3_layout_remains_bound_to_exact_selection(model: str, case: str) -> None:
    policy = known_native_split_policy(model_id=model, case_id=case,
        setup_id='orin_nx_hailo10_01', backend='hailo10h_to_trt')
    part1 = {'path': '/test/part1.hef', 'sha256': 'b' * 64, 'size_bytes': 123}
    metadata = {
        'schema': 'onnx-splitpoint/native-part1-boundary-metadata', 'schema_version': 1,
        'model_id': model, 'case_id': case, 'setup_id': 'orin_nx_hailo10_01',
        'backend': 'hailo10h_to_trt', 'part1_artifact_sha256': part1['sha256'],
        'part1_artifact_size_bytes': part1['size_bytes'], 'boundary_tensor_count': 1,
        'boundary_tensor': {'name': 'cut', 'shape': PHYSICAL,
            'canonical_part2_shape': TARGET, 'dtype': 'uint8',
            'quantization': {'source': 'hailort_hef_output_vstream_info',
                             'scale': .125, 'zero_point': 93.}},
        'boundary_layout': LAYOUT, 'boundary_transform': 'uint8_dequant_then_layout',
    }
    metadata['metadata_sha256'] = canonical_json_sha256(metadata)
    selection = materialize_native_split_preselection(policy=policy,
        part1_artifact=part1, boundary_metadata=metadata,
        boundary_metadata_artifact={'path': '/test/boundary.json', 'sha256': 'a'*64, 'size_bytes': 456})
    assert selection['boundary_layout'] == LAYOUT
    valid, _ = validate_native_split_preselection(selection)
    assert valid is not None
    changed = copy.deepcopy(selection)
    changed['boundary_layout'] = 'as_input'
    invalid, reason = validate_native_split_preselection(changed)
    assert invalid is None and reason.endswith('sha256_mismatch')
    changed.pop('selection_sha256')
    changed['selection_sha256'] = canonical_json_sha256(changed)
    invalid, reason = validate_native_split_preselection(changed)
    assert invalid is None and 'boundary_layout_mismatch' in reason
