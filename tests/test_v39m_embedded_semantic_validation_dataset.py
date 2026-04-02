from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService, _provision_embedded_semantic_validation_dataset
from onnx_splitpoint_tool.remote.bundle import REMOTE_MINIMAL_INCLUDES


def test_provision_embedded_semantic_validation_dataset_once(tmp_path: Path) -> None:
    suite = tmp_path / 'suite'
    suite.mkdir()
    rel = _provision_embedded_semantic_validation_dataset(suite)
    assert rel == 'resources/validation/coco_50_data'
    ds = suite / rel
    assert ds.is_dir()
    jpgs = list(ds.glob('*.jpg'))
    assert len(jpgs) >= 50
    # second call should be idempotent
    rel2 = _provision_embedded_semantic_validation_dataset(suite)
    assert rel2 == rel
    assert len(list(ds.glob('*.jpg'))) == len(jpgs)


def test_finalize_generation_outputs_wires_embedded_validation_dataset_when_empty(tmp_path: Path) -> None:
    svc = BenchmarkGenerationService()
    out_dir = tmp_path / 'suite'
    out_dir.mkdir(parents=True, exist_ok=True)
    full_model = out_dir / 'model.onnx'
    full_model.write_text('stub', encoding='utf-8')

    run = {
        'id': 'ort_cuda',
        'type': 'onnxruntime',
        'provider': 'cuda',
        'image_scale': 'auto',
        'stage1': {'type': 'onnxruntime', 'provider': 'cuda'},
        'stage2': {'type': 'onnxruntime', 'provider': 'cuda'},
        'validation_images': '',
        'validation_max_images': 0,
    }

    svc.finalize_generation_outputs(
        out_dir=out_dir,
        base='demo',
        full_model_src=str(full_model),
        full_model_dst=str(full_model),
        analysis_params={},
        system_spec=None,
        cases=[{'boundary': 1}],
        errors=[],
        discarded_cases=[],
        requested_cases=1,
        preferred_shortlist_original=[1],
        ranked_candidates=[1],
        shortlist_prefiltered_boundaries=[],
        candidate_search_pool=[1],
        bench_log_path=out_dir / 'benchmark_generation.log',
        analysis_payload={},
        bench_plan_runs=[run],
        hef_targets=[],
        hef_full=False,
        hef_part1=False,
        hef_part2=False,
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
        suite_hailo_hefs=None,
        write_harness_script=lambda *args, **kwargs: str(out_dir / 'benchmark_suite.py'),
        benchmark_objective='latency',
    )
    plan = json.loads((out_dir / 'benchmark_plan.json').read_text(encoding='utf-8'))
    run0 = plan['runs'][0]
    assert run0['validation_images'] == 'resources/validation/coco_50_data'
    assert int(run0['validation_max_images']) == 50
    assert (out_dir / 'resources' / 'validation' / 'coco_50_data').is_dir()


def test_remote_bundle_includes_suite_validation_dataset() -> None:
    assert 'resources/validation/**' in REMOTE_MINIMAL_INCLUDES


def test_benchmark_suite_template_resolves_validation_images_against_suite_root() -> None:
    src = Path('onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt').read_text(encoding='utf-8')
    assert 'validation paths against the suite root' in src
    assert 'val_img = (case_dir.parent / val_img).resolve()' in src
