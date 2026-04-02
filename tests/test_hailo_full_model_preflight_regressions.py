from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def _make_runtime(tmp_path: Path) -> BenchmarkGenerationRuntime:
    return BenchmarkGenerationRuntime(
        out_dir=tmp_path,
        bench_log_path=tmp_path / 'benchmark_generation.log',
        state_path=tmp_path / '.benchmark_generation_state.json',
        requested_cases=1,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        model_name='toy',
        model_source=str(tmp_path / 'toy.onnx'),
        hef_full_policy='end',
    )


def _make_exec_cfg(tmp_path: Path) -> BenchmarkGenerationExecutionConfig:
    x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 4, 4])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node('Identity', ['x'], ['y'], name='id')],
            'g',
            [x],
            [y],
        ),
        opset_imports=[helper.make_opsetid('', 13)],
    )
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
        nodes=list(model.graph.node),
        order=[0],
        analysis_payload={},
        full_model_src=str(tmp_path / 'toy.onnx'),
        full_model_dst=str(tmp_path / 'toy.onnx'),
        hef_targets=['hailo8'],
        hef_part1=True,
        hef_part2=True,
    )


class _NoCaseLoopExecutionService(BenchmarkGenerationExecutionService):
    def __init__(self) -> None:
        super().__init__()
        self.case_loop_called = False

    def execute_case_build_loop(self, cfg, callbacks):  # type: ignore[override]
        self.case_loop_called = True
        raise AssertionError('candidate loop must not run when full-model Hailo preflight aborts generation')


def test_orchestration_aborts_on_explicit_full_model_hailo_parser_blocker(tmp_path: Path) -> None:
    model_path = tmp_path / 'toy.onnx'
    model_path.write_bytes(b'onnx')

    exec_cfg = _make_exec_cfg(tmp_path)
    runtime = exec_cfg.runtime
    callbacks = BenchmarkGenerationExecutionCallbacks(
        log=lambda *args, **kwargs: None,
        queue_put=lambda *args, **kwargs: None,
        persist_state=lambda *args, **kwargs: None,
        publish_hailo_diagnostics=lambda *args, **kwargs: None,
        predicted_metrics_for_boundary=lambda payload, boundary: {},
        hailo_parse_entry_for_boundary=lambda payload, boundary: None,
        hailo_parse_scalar_fields=lambda entry: {},
    )

    def _write_harness(out_dir: str, bench_json: str) -> str:
        path = Path(out_dir) / 'benchmark_suite.py'
        path.write_text('# stub harness\n', encoding='utf-8')
        return str(path)

    def _fake_parse_check(*args, **kwargs):
        return SimpleNamespace(
            ok=False,
            elapsed_s=0.12,
            backend='venv',
            error=(
                'Parsing failed. The errors found in the graph are: '
                'UnsupportedActivationLayerError in op /convnext/stage0/act/Erf: '
                'Unexpected activation at /convnext/stage0/act/Erf, op=Erf '
                'UnsupportedOperationError in op /head/TopK: TopK operation is unsupported'
            ),
            fixed_onnx_path=None,
        )

    cfg = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=exec_cfg,
        execution_callbacks=callbacks,
        target_cases=1,
        preferred_shortlist_original=[0],
        ranked_candidates=[0],
        candidate_search_pool=[0],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        full_model_src=str(model_path),
        full_model_dst=str(model_path),
        analysis_payload={},
        analysis_params_payload={},
        system_spec_payload=None,
        bench_log_path=str(tmp_path / 'benchmark_generation.log'),
        bench_plan_runs=[],
        hef_targets=['hailo8'],
        hef_full=True,
        hef_part1=True,
        hef_part2=True,
        hef_backend='auto',
        hef_fixup=False,
        hef_opt_level=1,
        hef_calib_dir=None,
        hef_calib_count=1,
        hef_calib_bs=1,
        hef_force=False,
        hef_keep=False,
        hef_wsl_distro=None,
        hef_wsl_venv='auto',
        hef_timeout_s=30,
        full_hef_policy='end',
        hailo_parse_check_fn=_fake_parse_check,
        write_harness_script=_write_harness,
        hailo_selected=True,
    )

    exec_service = _NoCaseLoopExecutionService()
    svc = BenchmarkGenerationOrchestrationService(execution_service=exec_service)
    result = svc.run(cfg)

    preflight = result.summary_data.get('hailo_full_model_preflight')
    assert isinstance(preflight, dict)
    assert preflight.get('checked') is True
    assert preflight.get('aborted') is True
    assert preflight.get('failed_count') == 1
    assert preflight.get('unsupported_failure_count') == 1
    assert exec_service.case_loop_called is False
    assert result.final_status == 'warn'
    assert 'Hailo parser preflight' in str(result.final_msg)
    assert 'blocked benchmark generation' in str(result.summary_data.get('raw_text') or '')
