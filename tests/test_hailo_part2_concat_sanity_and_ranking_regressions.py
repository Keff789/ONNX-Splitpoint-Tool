from __future__ import annotations

from dataclasses import dataclass

from onnx_splitpoint_tool.benchmark.hailo_policy import HailoFailureRecord, should_skip_from_failure_cluster
from onnx_splitpoint_tool.benchmark.hailo_scoring import rerank_candidates_for_hailo
from onnx_splitpoint_tool.benchmark.part2_sanity import (
    format_hailo_part2_concat_sanity_error,
    hailo_part2_concat_sanity_from_model,
)


class _FakeDim:
    def __init__(self, value=None, param=None):
        self.dim_value = value
        self.dim_param = param

    def HasField(self, name: str) -> bool:  # noqa: N802 - protobuf-style API
        if name == 'dim_value':
            return self.dim_value is not None
        if name == 'dim_param':
            return self.dim_param is not None
        return False


class _FakeShape:
    def __init__(self, dims):
        self.dim = [_FakeDim(v) for v in dims]


class _FakeTensorType:
    def __init__(self, dims):
        self.shape = _FakeShape(dims)


class _FakeType:
    def __init__(self, dims):
        self.tensor_type = _FakeTensorType(dims)


class _FakeValueInfo:
    def __init__(self, name, dims):
        self.name = name
        self.type = _FakeType(dims)


class _FakeAttr:
    def __init__(self, name, i):
        self.name = name
        self.i = i


class _FakeNode:
    def __init__(self, *, op_type, name, inputs, outputs, axis=1):
        self.op_type = op_type
        self.name = name
        self.input = list(inputs)
        self.output = list(outputs)
        self.attribute = [_FakeAttr('axis', axis)] if op_type == 'Concat' else []


class _FakeGraph:
    def __init__(self, *, nodes, value_infos):
        self.node = list(nodes)
        self.input = list(value_infos)
        self.output = []
        self.value_info = list(value_infos)


class _FakeModel:
    def __init__(self, graph):
        self.graph = graph


def test_part2_concat_sanity_detects_mismatched_head_concat_without_onnx_dependency() -> None:
    model = _FakeModel(
        _FakeGraph(
            nodes=[
                _FakeNode(
                    op_type='Concat',
                    name='/model.23/Concat_3',
                    inputs=['a', 'b', 'c'],
                    outputs=['out'],
                    axis=2,
                )
            ],
            value_infos=[
                _FakeValueInfo('a', [1, 1, 4, 6400]),
                _FakeValueInfo('b', [1, 1, 4, 1600]),
                _FakeValueInfo('c', [1, 1, 400, 4]),
            ],
        )
    )

    info = hailo_part2_concat_sanity_from_model(model, split_manifest={'part2_outputs_effective': ['/model.23/Transpose_3_output_0']})
    assert info['inspect_ok'] is True
    assert info['compatible'] is False
    assert info['reason'] == 'concat_shape_mismatch'
    assert info['node_name'] == '/model.23/Concat_3'
    assert 3 in list(info['mismatched_dims'])
    msg = format_hailo_part2_concat_sanity_error(info)
    assert 'Concat_3' in msg
    assert 'input_shapes=' in msg


def test_hailo8_rerank_penalizes_suggested_endnode_fallback_and_raw_head_boundary() -> None:
    analysis = {
        'scores': {10: 1.0, 20: 1.0},
        'costs_bytes': {10: 1024.0, 20: 1024.0},
        'peak_act_mem_right_bytes': {10: 1024.0, 20: 1024.0},
        'crossing_counts_all': {10: 2, 20: 2},
        'flops_left_prefix': {10: 5.0, 20: 5.0},
        'total_flops': 10.0,
        '_candidate_rows': [
            {
                'boundary': 10,
                'cut_tensors': ['/clean/Reshape_output_0'],
                'hailo_parse_checked': True,
                'hailo_parse_ok': True,
                'hailo_parse_strategy': 'original',
            },
            {
                'boundary': 20,
                'cut_tensors': [
                    '/model.22/Slice_output_0',
                    '/model.23/one2one_cv3.1/conv/Conv_output_0',
                    '/model.23/one2one_cv3.1/act/Sigmoid_output_0',
                ],
                'hailo_parse_checked': True,
                'hailo_parse_ok': True,
                'hailo_parse_strategy': 'hailo_parser_suggested_end_nodes',
                'hailo_parse_used_suggested_end_nodes': True,
            },
        ],
    }

    ranked, meta = rerank_candidates_for_hailo(analysis, [20, 10])
    assert ranked[0] == 10
    assert meta[20]['hailo_compile_risk_score'] > meta[10]['hailo_compile_risk_score']


def test_failure_cluster_skip_requires_same_failure_family() -> None:
    failures = [
        HailoFailureRecord(
            boundary=552,
            stage='part1',
            hw_arch='hailo8',
            failure_kind='allocator_agent_infeasible',
            clusterable=True,
            detail='format_conversion8 errors:\n\tAgent infeasible',
            family='format_conversion_agent_infeasible',
        ),
        HailoFailureRecord(
            boundary=553,
            stage='part1',
            hw_arch='hailo8',
            failure_kind='allocator_agent_infeasible',
            clusterable=True,
            detail='Validator failed on node: concat3 with Agent infeasible',
            family='validator_concat',
        ),
    ]
    decision = should_skip_from_failure_cluster(
        554,
        failures,
        candidate_policy={'hailo_clusterable_boundary': True},
        stage='part1',
        hw_archs=['hailo8'],
        radius=12,
        min_failures=2,
    )
    assert decision.skip is False
    assert decision.nearby_failures == 1
