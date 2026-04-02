from __future__ import annotations

from pathlib import Path

import pytest

onnx = pytest.importorskip('onnx')
from onnx import TensorProto, helper

from onnx_splitpoint_tool.benchmark.hailo_policy import (
    HailoFailureRecord,
    analyze_cut_tensors,
    build_case_hailo_variant_availability,
    case_has_usable_hailo_variant,
    should_skip_from_failure_cluster,
)
from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationRuntime,
)
from onnx_splitpoint_tool.hailo_backend import HailoHefBuildResult


def _make_simple_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])
    nodes = [
        helper.make_node('Identity', ['x'], ['mid'], name='/backbone/Identity_0'),
        helper.make_node('Identity', ['mid'], ['y'], name='/head/Identity_1'),
    ]
    graph = helper.make_graph(nodes, 'g', [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])


def _make_runtime(tmp_path: Path) -> BenchmarkGenerationRuntime:
    return BenchmarkGenerationRuntime(
        out_dir=tmp_path,
        bench_log_path=tmp_path / 'benchmark_generation.log',
        state_path=tmp_path / 'generation_state.json',
        requested_cases=1,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        model_name='toy',
        model_source='toy.onnx',
        hef_full_policy='end',
    )


def _make_callbacks() -> BenchmarkGenerationExecutionCallbacks:
    return BenchmarkGenerationExecutionCallbacks(
        log=lambda *args, **kwargs: None,
        queue_put=lambda *args, **kwargs: None,
        persist_state=lambda *args, **kwargs: None,
        publish_hailo_diagnostics=lambda *args, **kwargs: None,
        predicted_metrics_for_boundary=lambda payload, boundary: {},
        hailo_parse_entry_for_boundary=lambda payload, boundary: None,
        hailo_parse_scalar_fields=lambda entry: {},
    )


def _make_exec_cfg(
    tmp_path: Path,
    *,
    bench_plan_runs,
    hailo_build_hef_fn,
) -> BenchmarkGenerationExecutionConfig:
    model = _make_simple_model()
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
        analysis_candidates=[
            {
                'boundary': 0,
                'cut_tensors': [
                    '/model.23/Reshape_output_0',
                    '/model.22/Slice_output_0',
                    '/model.23/one2one_cv3.1/conv/Conv_output_0',
                    '/model.23/one2one_cv3.1/act/Sigmoid_output_0',
                ],
            }
        ],
        bench_plan_runs=list(bench_plan_runs),
        full_model_src='toy.onnx',
        full_model_dst='toy.onnx',
        hef_targets=['hailo8'],
        hef_part1=True,
        hef_part2=True,
        hailo_build_hef_fn=hailo_build_hef_fn,
        hef_backend='dataflow_compiler',
        hef_fixup=False,
        hef_opt_level=1,
        hef_calib_dir=None,
        hef_calib_count=1,
        hef_calib_bs=1,
        hef_force=False,
        hef_keep=False,
        hef_wsl_distro=None,
        hef_wsl_venv='',
        hef_timeout_s=0,
    )


def _fake_hef_builder_factory(*, part1_ok: bool, part2_ok: bool):
    def _build(onnx_path, *, outdir, net_name, hw_arch, **kwargs):
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        is_part1 = 'part1' in str(net_name)
        ok = part1_ok if is_part1 else part2_ok
        if ok:
            hef_path = outdir_path / 'compiled.hef'
            hef_path.write_text('hef', encoding='utf-8')
            return HailoHefBuildResult(
                ok=True,
                elapsed_s=0.1,
                hw_arch=str(hw_arch),
                net_name=str(net_name),
                backend='venv',
                hef_path=str(hef_path),
                details={'process_summary': {'detected': {'single_context_used': True}}},
            )
        return HailoHefBuildResult(
            ok=False,
            elapsed_s=0.1,
            hw_arch=str(hw_arch),
            net_name=str(net_name),
            backend='venv',
            error='Compilation failed: No successful assignments: format_conversion8 errors:\n\tAgent infeasible',
            returncode=3,
            details={
                'process_summary': {
                    'detected': {
                        'single_context_failed': True,
                        'mapping_failed': True,
                    },
                    'single_context_failure': '[info] Single context flow failed: Recoverable single context error',
                }
            },
        )

    return _build


def test_hailo_cut_tensor_policy_flags_yolo26_style_late_head_boundary() -> None:
    profile = analyze_cut_tensors(
        [
            '/model.23/Reshape_output_0',
            '/model.23/Reshape_3_output_0',
            '/model.22/Slice_output_0',
            '/model.22/Slice_1_output_0',
            '/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/conv/Conv_output_0',
            '/model.22/m.0/m.0.0/cv1/act/Mul_output_0',
            '/model.23/one2one_cv3.1/one2one_cv3.1.1/one2one_cv3.1.1.1/act/Sigmoid_output_0',
        ]
    )
    assert profile.penalty >= 1.0
    assert profile.high_risk is True
    assert profile.clusterable is True
    assert profile.one2one_count >= 1
    assert profile.raw_activation_count >= 2


def test_failure_cluster_skip_triggers_for_repeated_allocator_failures() -> None:
    failures = [
        HailoFailureRecord(boundary=531, stage='part1', hw_arch='hailo8', failure_kind='allocator_agent_infeasible', clusterable=True, detail='format_conversion8'),
        HailoFailureRecord(boundary=538, stage='part1', hw_arch='hailo8', failure_kind='allocator_agent_infeasible', clusterable=True, detail='format_conversion9'),
    ]
    decision = should_skip_from_failure_cluster(
        534,
        failures,
        candidate_policy={'hailo_clusterable_boundary': True},
        stage='part1',
        hw_archs=['hailo8'],
        radius=12,
        min_failures=2,
    )
    assert decision.skip is True
    assert decision.nearby_failures == 2


def test_case_has_usable_hailo_variant_for_partial_matrix_case() -> None:
    availability = build_case_hailo_variant_availability(
        {},
        {
            'hailo8': {
                'part2': 'b001/hailo/hailo8/part2/compiled.hef',
                'part2_build': {'ok': True},
                'part1_error': 'failed',
            }
        },
    )
    runs = [
        {
            'id': 'trt_to_hailo8',
            'type': 'matrix',
            'variants': ['part1', 'part2', 'composed'],
            'stage1': {'type': 'onnxruntime', 'provider': 'tensorrt'},
            'stage2': {'type': 'hailo', 'hw_arch': 'hailo8'},
        }
    ]
    assert availability['hailo8']['part2'] is True
    assert availability['hailo8']['part1'] is False
    assert case_has_usable_hailo_variant(runs, availability) is True


def test_execution_service_keeps_partial_case_when_stage2_hailo_variant_is_still_usable(tmp_path: Path) -> None:
    bench_plan_runs = [
        {
            'id': 'trt_to_hailo8',
            'type': 'matrix',
            'variants': ['part1', 'part2', 'composed'],
            'stage1': {'type': 'onnxruntime', 'provider': 'tensorrt'},
            'stage2': {'type': 'hailo', 'hw_arch': 'hailo8'},
        }
    ]
    cfg = _make_exec_cfg(
        tmp_path,
        bench_plan_runs=bench_plan_runs,
        hailo_build_hef_fn=_fake_hef_builder_factory(part1_ok=False, part2_ok=True),
    )
    svc = BenchmarkGenerationExecutionService()
    svc.execute_case_build_loop(cfg, _make_callbacks())

    assert len(cfg.runtime.cases) == 1
    assert len(cfg.runtime.discarded_cases) == 0
    case = cfg.runtime.cases[0]
    availability = case.get('hailo_case_variant_availability') or {}
    assert availability['hailo8']['part1'] is False
    assert availability['hailo8']['part2'] is True
    assert availability['hailo8']['composed'] is False


def test_execution_service_rejects_partial_case_when_no_configured_hailo_variant_remains(tmp_path: Path) -> None:
    bench_plan_runs = [
        {
            'id': 'hailo8',
            'type': 'hailo',
            'hw_arch': 'hailo8',
            'variants': ['full', 'composed'],
            'stage1': {'type': 'hailo', 'hw_arch': 'hailo8'},
            'stage2': {'type': 'hailo', 'hw_arch': 'hailo8'},
        }
    ]
    cfg = _make_exec_cfg(
        tmp_path,
        bench_plan_runs=bench_plan_runs,
        hailo_build_hef_fn=_fake_hef_builder_factory(part1_ok=False, part2_ok=True),
    )
    svc = BenchmarkGenerationExecutionService()
    svc.execute_case_build_loop(cfg, _make_callbacks())

    assert len(cfg.runtime.cases) == 0
    assert len(cfg.runtime.discarded_cases) == 1
    rec = cfg.runtime.discarded_cases[0]
    assert str(rec.get('reason') or '') == 'hailo_hef_build_failed'
