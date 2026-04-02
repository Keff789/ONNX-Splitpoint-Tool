from __future__ import annotations

from pathlib import Path

import pytest

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationConfig,
    BenchmarkGenerationOrchestrationService,
    BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool.hailo_backend import (
    format_hailo_part2_activation_precheck_error,
    format_hailo_part2_parser_blocker_error,
    hailo_part2_activation_precheck_from_manifest,
    hailo_part2_parser_blocker_precheck_from_model,
)
from onnx_splitpoint_tool.split_export_graph import cut_tensors_for_boundary, split_model_on_cut_tensors


def _make_full_model_with_blocked_tail() -> onnx.ModelProto:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4, 4])
    k = helper.make_tensor('k', TensorProto.INT64, [1], [1])
    idx = helper.make_tensor_value_info('idx', TensorProto.INT64, [1, 4, 1])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4, 1])
    nodes = [
        helper.make_node('Identity', ['x'], ['feat'], name='/backbone/Identity'),
        helper.make_node('Transpose', ['feat'], ['t2'], perm=[0, 2, 1], name='/model.23/Transpose_2'),
        helper.make_node('Transpose', ['t2'], ['t3'], perm=[0, 2, 1], name='/model.23/Transpose_3'),
        helper.make_node('TopK', ['t3', 'k'], ['values', 'indices'], axis=-1, largest=1, sorted=1, name='/model.23/TopK'),
        helper.make_node('GatherElements', ['t3', 'indices'], ['g'], axis=-1, name='/model.23/GatherElements'),
        helper.make_node('Identity', ['indices'], ['idx'], name='/model.23/IdentityIdx'),
        helper.make_node('Identity', ['g'], ['y'], name='/model.23/IdentityOut'),
    ]
    graph = helper.make_graph(nodes, 'g', [x], [idx, y], initializer=[k])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])


def _make_runtime(tmp_path: Path) -> BenchmarkGenerationRuntime:
    return BenchmarkGenerationRuntime(
        out_dir=tmp_path,
        bench_log_path=tmp_path / 'benchmark_generation.log',
        state_path=tmp_path / '.benchmark_generation_state.json',
        requested_cases=1,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        model_name='toy',
        model_source='toy.onnx',
        hef_full_policy='end',
    )


def _make_exec_cfg(tmp_path: Path) -> BenchmarkGenerationExecutionConfig:
    model = _make_full_model_with_blocked_tail()
    nodes = list(model.graph.node)
    order = list(range(len(nodes)))
    return BenchmarkGenerationExecutionConfig(
        runtime=_make_runtime(tmp_path),
        target_cases=1,
        gap=0,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        strict_boundary=False,
        model=model,
        nodes=nodes,
        order=order,
        analysis_payload={},
        full_model_src='toy.onnx',
        full_model_dst='toy.onnx',
        hef_targets=['hailo8'],
        hef_part1=True,
        hef_part2=True,
        hailo_part2_precheck_fn=hailo_part2_activation_precheck_from_manifest,
        hailo_part2_precheck_error_fn=format_hailo_part2_activation_precheck_error,
        hailo_part2_parser_precheck_fn=hailo_part2_parser_blocker_precheck_from_model,
        hailo_part2_parser_precheck_error_fn=format_hailo_part2_parser_blocker_error,
    )


def test_split_model_on_cut_tensors_supports_part2_output_override() -> None:
    model = _make_full_model_with_blocked_tail()
    nodes = list(model.graph.node)
    order = list(range(len(nodes)))
    cut = cut_tensors_for_boundary(order, nodes, 0)
    _p1, p2, manifest = split_model_on_cut_tensors(
        model,
        cut_tensors=cut,
        part2_output_names=['t2', 't3'],
    )
    assert list(manifest.get('part2_outputs_effective') or []) == ['t2', 't3']
    assert str(manifest.get('part2_output_strategy') or '') == 'override'
    assert [str(vi.name) for vi in p2.graph.output] == ['t2', 't3']


def test_probe_hailo_part2_support_uses_suggested_end_nodes(tmp_path: Path) -> None:
    cfg = _make_exec_cfg(tmp_path)
    svc = BenchmarkGenerationExecutionService()
    probe = svc.probe_hailo_part2_support(cfg, 0)
    assert probe['inspect_ok'] is True
    assert probe['compatible'] is True
    assert probe['used_suggested_end_nodes'] is True
    assert str(probe.get('strategy') or '') == 'hailo_parser_suggested_end_nodes'
    assert list(probe.get('effective_part2_outputs') or [])
    p2_model = probe['p2_model']
    assert all(node.op_type not in {'TopK', 'GatherElements'} for node in p2_model.graph.node)


def test_probe_hailo_part2_support_can_disable_suggested_end_nodes(tmp_path: Path) -> None:
    cfg = _make_exec_cfg(tmp_path)
    cfg.hailo_part2_enable_suggested_endnode_fallback = False
    svc = BenchmarkGenerationExecutionService()
    probe = svc.probe_hailo_part2_support(cfg, 0)
    assert probe['inspect_ok'] is True
    assert probe['compatible'] is False
    assert probe.get('fallback_disabled') is True
    assert probe.get('reason') == 'parser'


def test_orchestration_service_downgrades_hailo_stage2_variants_without_part2(tmp_path: Path) -> None:
    exec_cfg = _make_exec_cfg(tmp_path)
    runtime = exec_cfg.runtime
    cb = BenchmarkGenerationExecutionCallbacks(
        log=lambda *args, **kwargs: None,
        queue_put=lambda *args, **kwargs: None,
        persist_state=lambda *args, **kwargs: None,
        publish_hailo_diagnostics=lambda *args, **kwargs: None,
        predicted_metrics_for_boundary=lambda payload, boundary: {},
        hailo_parse_entry_for_boundary=lambda payload, boundary: None,
        hailo_parse_scalar_fields=lambda entry: {},
    )
    bench_plan_runs = [
        {
            'id': 'hailo8',
            'type': 'hailo',
            'hw_arch': 'hailo8',
            'variants': ['composed', 'part1', 'part2'],
            'stage1': {'type': 'hailo', 'hw_arch': 'hailo8'},
            'stage2': {'type': 'hailo', 'hw_arch': 'hailo8'},
        },
        {
            'id': 'trt_to_hailo8',
            'type': 'matrix',
            'variants': ['part1', 'part2', 'composed'],
            'stage1': {'type': 'onnxruntime', 'provider': 'tensorrt'},
            'stage2': {'type': 'hailo', 'hw_arch': 'hailo8'},
        },
    ]
    cfg = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=exec_cfg,
        execution_callbacks=cb,
        target_cases=1,
        preferred_shortlist_original=[0],
        ranked_candidates=[0],
        candidate_search_pool=[0],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        full_model_src='toy.onnx',
        full_model_dst='toy.onnx',
        analysis_payload={},
        analysis_params_payload={},
        system_spec_payload=None,
        bench_log_path=str(tmp_path / 'benchmark_generation.log'),
        bench_plan_runs=bench_plan_runs,
        hef_targets=['hailo8'],
        hef_full=False,
        hef_part1=True,
        hef_part2=True,
        hef_backend='dataflow_compiler',
        hef_fixup=False,
        hef_opt_level=2,
        hef_calib_dir=None,
        hef_calib_count=1,
        hef_calib_bs=1,
        hef_force=False,
        hef_keep=False,
        hef_wsl_distro=None,
        hef_wsl_venv='',
        hef_timeout_s=0,
        full_hef_policy='end',
    )
    svc = BenchmarkGenerationOrchestrationService()
    downgraded = svc._downgrade_benchmark_plan_without_hailo_part2(cfg, log=lambda *args, **kwargs: None)
    runs = {str(r.get('id')): list(r.get('variants') or []) for r in downgraded.bench_plan_runs}
    assert runs['hailo8'] == ['full', 'part1']
    assert runs['trt_to_hailo8'] == ['part1']
    assert downgraded.hef_part2 is False
    assert downgraded.hef_part1 is True
