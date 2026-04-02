import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.analysis import load_benchmark_analysis
from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService


def test_finalize_generation_outputs_persists_objective(tmp_path: Path) -> None:
    svc = BenchmarkGenerationService()
    out_dir = tmp_path / "suite"
    out_dir.mkdir(parents=True)
    full_src = out_dir / "model.onnx"
    full_src.write_bytes(b"x")
    result = svc.finalize_generation_outputs(
        out_dir=out_dir,
        base='demo',
        full_model_src=str(full_src),
        full_model_dst=str(full_src),
        analysis_params={},
        system_spec=None,
        cases=[{'boundary': 7, 'predicted': {'score': 1.0, 'cut_mib': 2.0}}],
        errors=[],
        discarded_cases=[],
        requested_cases=1,
        preferred_shortlist_original=[7],
        ranked_candidates=[7],
        shortlist_prefiltered_boundaries=[],
        candidate_search_pool=[7],
        bench_log_path=str(out_dir / 'benchmark_generation.log'),
        analysis_payload={},
        bench_plan_runs=[],
        hef_targets=[],
        hef_full=False,
        hef_part1=False,
        hef_part2=False,
        hef_backend='hailo',
        hef_wsl_distro=None,
        hef_wsl_venv='',
        hef_opt_level=0,
        hef_calib_count=0,
        hef_calib_bs=1,
        hef_calib_dir=None,
        hef_fixup=False,
        hef_force=False,
        hef_keep=False,
        suite_hailo_hefs=None,
        write_harness_script=lambda *_args, **_kwargs: str(out_dir / 'benchmark_suite.py'),
        benchmark_objective='throughput',
    )
    bench = json.loads((out_dir / 'benchmark_set.json').read_text(encoding='utf-8'))
    plan = json.loads((out_dir / 'benchmark_plan.json').read_text(encoding='utf-8'))
    assert bench['objective'] == 'throughput'
    assert plan['objective'] == 'throughput'
    (out_dir / 'benchmark_results_demo.csv').write_text(
        'boundary,full_mean_ms,composed_mean_ms,final_pass_all,score_pred,cut_mib\n7,25.0,27.0,true,1.0,2.0\n',
        encoding='utf-8',
    )
    report = load_benchmark_analysis(out_dir, cache_base=tmp_path / 'cache')
    assert str(report.summary.get('objective')).lower() == 'throughput'
    assert result.bench_payload['objective'] == 'throughput'
