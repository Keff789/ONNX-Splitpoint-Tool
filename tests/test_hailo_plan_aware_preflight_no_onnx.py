from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def _make_exec_cfg(tmp_path: Path, runtime: BenchmarkGenerationRuntime) -> BenchmarkGenerationExecutionConfig:
    return BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        strict_boundary=False,
        model=None,
        nodes=[],
        order=[],
        analysis_payload={},
        full_model_src=str(tmp_path / 'toy.onnx'),
        full_model_dst=str(tmp_path / 'toy.onnx'),
        hef_targets=['hailo8'],
        hef_part1=True,
        hef_part2=True,
        bench_plan_runs=[
            {'id': 'ort_tensorrt', 'type': 'onnxruntime', 'provider': 'tensorrt', 'stage1': {'type': 'onnxruntime', 'provider': 'tensorrt'}, 'stage2': {'type': 'onnxruntime', 'provider': 'tensorrt'}},
            {'id': 'hailo8', 'type': 'hailo', 'hw_arch': 'hailo8', 'variants': ['full', 'composed', 'part1', 'part2'], 'stage1': {'type': 'hailo', 'hw_arch': 'hailo8'}, 'stage2': {'type': 'hailo', 'hw_arch': 'hailo8'}},
            {'id': 'trt_to_hailo8', 'type': 'matrix', 'provider': 'tensorrt', 'variants': ['part1', 'part2', 'composed'], 'stage1': {'type': 'onnxruntime', 'provider': 'tensorrt'}, 'stage2': {'type': 'hailo', 'hw_arch': 'hailo8'}},
            {'id': 'hailo8_to_trt', 'type': 'matrix', 'provider': 'tensorrt', 'variants': ['part1', 'part2', 'composed'], 'stage1': {'type': 'hailo', 'hw_arch': 'hailo8'}, 'stage2': {'type': 'onnxruntime', 'provider': 'tensorrt'}},
        ],
    )


class _RecordingExecutionService(BenchmarkGenerationExecutionService):
    def __init__(self) -> None:
        super().__init__()
        self.case_loop_called = False
        self.last_cfg = None

    def probe_hailo_part2_support(self, cfg, boundary, log_cb=None):  # type: ignore[override]
        return {'compatible': True, 'strategy': 'original'}

    def execute_case_build_loop(self, cfg, callbacks):  # type: ignore[override]
        self.case_loop_called = True
        self.last_cfg = cfg
        return None


def _callbacks():
    return BenchmarkGenerationExecutionCallbacks(
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


def test_plan_aware_preflight_keeps_trt_to_hailo_when_failure_is_near_input(tmp_path: Path) -> None:
    model_path = tmp_path / 'toy.onnx'
    model_path.write_bytes(b'fake-onnx')
    runtime = _make_runtime(tmp_path)
    exec_cfg = _make_exec_cfg(tmp_path, runtime)

    def _fake_parse_check(*args, **kwargs):
        return SimpleNamespace(
            ok=False,
            elapsed_s=0.1,
            backend='venv',
            error=(
                'Parsing failed. The errors found in the graph are: '
                'UnsupportedOperationError in op pixel_values_QuantizeLinear: DynamicQuantizeLinear operation is unsupported '
                'UnsupportedOperationError in op /convnext/embeddings/patch_embeddings/Conv_quant: ConvInteger operation is unsupported'
            ),
            fixed_onnx_path=None,
        )

    cfg = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=exec_cfg,
        execution_callbacks=_callbacks(),
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
        bench_plan_runs=list(exec_cfg.bench_plan_runs),
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

    exec_service = _RecordingExecutionService()
    svc = BenchmarkGenerationOrchestrationService(execution_service=exec_service)
    result = svc.run(cfg)

    preflight = result.summary_data.get('hailo_full_model_preflight')
    assert isinstance(preflight, dict)
    assert preflight.get('plan_adjusted') is True
    assert preflight.get('aborted') is False
    assert exec_service.case_loop_called is True
    assert exec_service.last_cfg is not None
    run_ids = [str(r.get('id')) for r in exec_service.last_cfg.bench_plan_runs]
    assert 'trt_to_hailo8' in run_ids
    assert 'hailo8_to_trt' not in run_ids
    hailo_run = next(r for r in exec_service.last_cfg.bench_plan_runs if str(r.get('id')) == 'hailo8')
    assert list(hailo_run.get('variants') or []) == ['part2']
    assert exec_service.last_cfg.hef_part1 is False
    assert exec_service.last_cfg.hef_part2 is True


def test_plan_aware_preflight_aborts_if_no_runs_remain(tmp_path: Path) -> None:
    model_path = tmp_path / 'toy.onnx'
    model_path.write_bytes(b'fake-onnx')
    runtime = _make_runtime(tmp_path)
    exec_cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[0],
        candidate_search_pool=[0],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        strict_boundary=False,
        model=None,
        nodes=[],
        order=[],
        analysis_payload={},
        full_model_src=str(model_path),
        full_model_dst=str(model_path),
        hef_targets=['hailo8'],
        hef_part1=True,
        hef_part2=True,
        bench_plan_runs=[],
    )

    def _fake_parse_check(*args, **kwargs):
        return SimpleNamespace(
            ok=False,
            elapsed_s=0.1,
            backend='venv',
            error='UnsupportedOperationError in op pixel_values_QuantizeLinear: DynamicQuantizeLinear operation is unsupported',
            fixed_onnx_path=None,
        )

    cfg = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=exec_cfg,
        execution_callbacks=_callbacks(),
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

    exec_service = _RecordingExecutionService()
    svc = BenchmarkGenerationOrchestrationService(execution_service=exec_service)
    result = svc.run(cfg)
    preflight = result.summary_data.get('hailo_full_model_preflight')
    assert isinstance(preflight, dict)
    assert preflight.get('aborted') is True
    assert exec_service.case_loop_called is False
