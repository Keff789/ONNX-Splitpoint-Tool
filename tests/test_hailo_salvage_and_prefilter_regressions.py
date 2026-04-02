from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import (
    BenchmarkGenerationExecutionCallbacks,
    BenchmarkGenerationExecutionConfig,
    BenchmarkGenerationExecutionService,
    BenchmarkGenerationOrchestrationConfig,
    BenchmarkGenerationOrchestrationService,
    BenchmarkGenerationRuntime,
)
from dataclasses import dataclass


@dataclass
class _FakeHefBuildResult:
    ok: bool
    elapsed_s: float
    hw_arch: str
    net_name: str
    backend: str = 'venv'
    error: str | None = None
    hef_path: str | None = None
    details: dict | None = None
    returncode: int | None = None
    skipped: bool = False



class _DummyModel:
    pass


def _make_runtime(tmp_path: Path, *, target_cases: int = 2) -> BenchmarkGenerationRuntime:
    bench_log = tmp_path / 'benchmark_generation.log'
    bench_log.write_text('', encoding='utf-8')
    return BenchmarkGenerationRuntime(
        out_dir=tmp_path,
        bench_log_path=bench_log,
        state_path=tmp_path / 'generation_state.json',
        requested_cases=target_cases,
        ranked_candidates=[0, 1],
        candidate_search_pool=[0, 1],
        model_name='toy',
        model_source=str(tmp_path / 'toy.onnx'),
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


def test_shortlist_prefilter_uses_concat_sanity_reason_key(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path, target_cases=1)
    full_model = tmp_path / 'toy.onnx'
    full_model.write_text('onnx', encoding='utf-8')
    exec_cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=1,
        gap=0,
        ranked_candidates=[12],
        candidate_search_pool=[12, 13],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        strict_boundary=False,
        model=_DummyModel(),
        nodes=[],
        order=[],
        analysis_payload={},
        analysis_candidates=[{'boundary': 12}, {'boundary': 13}],
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        hef_targets=['hailo8'],
        hef_part1=True,
        hef_part2=True,
    )
    orch_cfg = BenchmarkGenerationOrchestrationConfig(
        runtime=runtime,
        execution_cfg=exec_cfg,
        execution_callbacks=_make_callbacks(),
        target_cases=1,
        preferred_shortlist_original=[12],
        ranked_candidates=[12],
        candidate_search_pool=[12, 13],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        analysis_payload={},
        analysis_params_payload={},
        system_spec_payload=None,
        bench_log_path=str(runtime.bench_log_path),
        bench_plan_runs=[],
        hef_targets=['hailo8'],
        hef_full=False,
        hef_part1=True,
        hef_part2=True,
        hef_backend='venv',
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
        full_hef_policy='end',
    )
    svc = BenchmarkGenerationOrchestrationService(execution_service=BenchmarkGenerationExecutionService())

    def _probe(*args, **kwargs):
        return {
            'inspect_ok': True,
            'compatible': False,
            'reason': 'concat_sanity',
            'detail': 'Hailo Part2 concat sanity failed at /model.23/Concat_3',
            'concat_sanity_precheck': {
                'inspect_ok': True,
                'compatible': False,
                'reason': 'concat_shape_mismatch',
                'node_name': '/model.23/Concat_3',
                'input_shapes': [[1, 1, 4, 6400], [1, 1, 400, 4]],
            },
        }

    svc.execution_service.probe_hailo_part2_support = _probe  # type: ignore[method-assign]
    discarded_cases = []
    discarded_boundaries = set()
    ranked, pool, filtered = svc._prefilter_shortlist_for_hailo_part2(
        orch_cfg,
        discarded_cases=discarded_cases,
        discarded_boundaries=discarded_boundaries,
        log=lambda *args, **kwargs: None,
    )

    assert ranked == []
    assert 12 not in pool
    assert filtered == {12}
    assert discarded_cases[0]['reason'] == 'hailo_part2_concat_sanity_prefilter'
    assert discarded_cases[0]['node_name'] == '/model.23/Concat_3'


def test_execution_service_attempts_nearby_row_per_cut_salvage_retry(tmp_path: Path, monkeypatch) -> None:
    runtime = _make_runtime(tmp_path, target_cases=2)
    full_model = tmp_path / 'toy.onnx'
    full_model.write_text('onnx', encoding='utf-8')

    cfg = BenchmarkGenerationExecutionConfig(
        runtime=runtime,
        target_cases=2,
        gap=0,
        ranked_candidates=[0, 1],
        candidate_search_pool=[0, 1],
        out_dir=tmp_path,
        base='toy',
        pad=3,
        strict_boundary=False,
        model=_DummyModel(),
        nodes=['n0', 'n1'],
        order=[0, 1],
        analysis_payload={},
        analysis_candidates=[{'boundary': 0}, {'boundary': 1}],
        bench_plan_runs=[
            {
                'id': 'hailo8_to_trt',
                'type': 'matrix',
                'variants': ['part1', 'part2', 'composed'],
                'stage1': {'type': 'hailo', 'hw_arch': 'hailo8'},
                'stage2': {'type': 'onnxruntime', 'provider': 'tensorrt'},
            }
        ],
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        hef_targets=['hailo8'],
        hef_part1=True,
        hef_part2=True,
        hef_backend='venv',
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
        hailo_build_hef_fn=None,  # set below
    )

    fake_api = types.ModuleType('onnx_splitpoint_tool.api')
    fake_api.__version__ = 'test'
    fake_api.cut_tensors_for_boundary = lambda order, nodes, boundary: [f'cut_{int(boundary)}']

    def _split_model_on_cut_tensors(model, *, cut_tensors, strict_boundary=False, part2_output_names=None):
        boundary = int(str(list(cut_tensors)[0]).split('_')[-1])
        return (
            {'kind': 'part1', 'boundary': boundary},
            {'kind': 'part2', 'boundary': boundary, 'outputs': list(part2_output_names or [])},
            {
                'part1_inputs': ['x'],
                'part1_outputs': [f'p1_out_{boundary}'],
                'part2_inputs': [f'p2_in_{boundary}'],
                'part2_outputs': [f'p2_out_{boundary}'],
                'cut_tensors': list(cut_tensors),
            },
        )

    fake_api.split_model_on_cut_tensors = _split_model_on_cut_tensors
    fake_api.save_model = lambda model, path: Path(path).write_text(json.dumps(model), encoding='utf-8')
    def _write_runner(case_dir, manifest_filename, target='auto'):
        runner = Path(case_dir) / 'runner.py'
        runner.write_text('# runner', encoding='utf-8')
        return str(runner)

    fake_api.write_runner_skeleton_onnxruntime = _write_runner
    monkeypatch.setitem(sys.modules, 'onnx_splitpoint_tool.api', fake_api)

    build_calls = []
    attempt_counts = {}

    def _builder(onnx_path, *, outdir, net_name, hw_arch, extra_model_script=None, **kwargs):
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        net_name = str(net_name)
        is_part1 = 'part1' in net_name
        boundary = int(net_name.rsplit('_b', 1)[-1])
        key = (boundary, 'part1' if is_part1 else 'part2')
        attempt_counts[key] = attempt_counts.get(key, 0) + 1
        build_calls.append({
            'boundary': boundary,
            'kind': 'part1' if is_part1 else 'part2',
            'extra_model_script': extra_model_script,
            'keep_artifacts': kwargs.get('keep_artifacts'),
        })
        if not is_part1:
            hef_path = outdir_path / 'compiled.hef'
            hef_path.write_text('hef', encoding='utf-8')
            return _FakeHefBuildResult(
                ok=True,
                elapsed_s=0.1,
                hw_arch=str(hw_arch),
                net_name=net_name,
                backend='venv',
                hef_path=str(hef_path),
            )
        if boundary == 0:
            hef_path = outdir_path / 'compiled.hef'
            hef_path.write_text('hef', encoding='utf-8')
            return _FakeHefBuildResult(
                ok=True,
                elapsed_s=0.1,
                hw_arch=str(hw_arch),
                net_name=net_name,
                backend='venv',
                hef_path=str(hef_path),
                details={'process_summary': {'row_per_cut_hints': ['conv10', 'conv76']}},
            )
        # boundary 1: fail first, succeed only on salvage retry with model script
        if extra_model_script and 'performance_param(compiler_optimization_level=max)' in str(extra_model_script):
            hef_path = outdir_path / 'compiled.hef'
            hef_path.write_text('hef', encoding='utf-8')
            return _FakeHefBuildResult(
                ok=True,
                elapsed_s=0.2,
                hw_arch=str(hw_arch),
                net_name=net_name,
                backend='venv',
                hef_path=str(hef_path),
                details={'process_summary': {'row_per_cut_hints': ['conv76']}},
            )
        return _FakeHefBuildResult(
            ok=False,
            elapsed_s=0.2,
            hw_arch=str(hw_arch),
            net_name=net_name,
            backend='venv',
            error='Compilation failed: No successful assignments: format_conversion8 errors:\n\tAgent infeasible',
            returncode=3,
            details={
                'process_summary': {
                    'single_context_failure': '[info] Single context flow failed: Recoverable single context error',
                    'detected': {'single_context_failed': True, 'mapping_failed': True},
                }
            },
        )

    cfg.hailo_build_hef_fn = _builder

    svc = BenchmarkGenerationExecutionService()
    svc.execute_case_build_loop(cfg, _make_callbacks())

    assert len(cfg.runtime.cases) == 2
    part1_calls_b1 = [c for c in build_calls if c['boundary'] == 1 and c['kind'] == 'part1']
    assert len(part1_calls_b1) == 2
    assert 'performance_param(compiler_optimization_level=max)' in str(part1_calls_b1[1]['extra_model_script'] or '')
    assert part1_calls_b1[1]['keep_artifacts'] is True

    manifest_path = tmp_path / 'b001' / 'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    hefs = (((manifest.get('hailo') or {}).get('hefs') or {}).get('hailo8') or {})
    assert hefs['part1_salvage']['attempted'] is True
    assert hefs['part1_salvage']['ok'] is True
    assert hefs['part1_salvage']['donor_boundary'] == 0
