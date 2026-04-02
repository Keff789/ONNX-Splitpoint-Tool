from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService


def test_finalize_generation_outputs_refreshes_case_level_hailo_full_availability(tmp_path: Path) -> None:
    svc = BenchmarkGenerationService()
    out_dir = tmp_path / "suite"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_model = out_dir / "model.onnx"
    full_model.write_text("stub", encoding="utf-8")

    result = svc.finalize_generation_outputs(
        out_dir=out_dir,
        base='demo',
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        analysis_params={},
        system_spec=None,
        cases=[{
            'boundary': 42,
            'case_dir': 'b042',
            'hailo_case_variant_availability': {
                'hailo8': {'full': False, 'part1': True, 'part2': True, 'composed': True}
            },
        }],
        errors=[],
        discarded_cases=[],
        requested_cases=1,
        preferred_shortlist_original=[42],
        ranked_candidates=[42],
        shortlist_prefiltered_boundaries=[],
        candidate_search_pool=[42],
        bench_log_path=out_dir / 'benchmark_generation.log',
        analysis_payload={},
        bench_plan_runs=[],
        hef_targets=['hailo8'],
        hef_full=True,
        hef_part1=True,
        hef_part2=True,
        hef_backend='auto',
        hef_wsl_distro=None,
        hef_wsl_venv='',
        hef_opt_level=2,
        hef_calib_count=0,
        hef_calib_bs=1,
        hef_calib_dir=None,
        hef_fixup=False,
        hef_force=False,
        hef_keep=False,
        suite_hailo_hefs={'hailo8': {'full': 'hailo/hailo8/full/compiled.hef'}},
        write_harness_script=lambda *args, **kwargs: str(out_dir / 'benchmark_suite.py'),
        benchmark_objective='latency',
    )
    bench = json.loads((out_dir / 'benchmark_set.json').read_text(encoding='utf-8'))
    avail = bench['cases'][0]['hailo_case_variant_availability']['hailo8']
    assert avail['full'] is True
    assert result.plan_path == out_dir / 'benchmark_plan.json'
    assert (out_dir / 'benchmark_set.json').exists()
