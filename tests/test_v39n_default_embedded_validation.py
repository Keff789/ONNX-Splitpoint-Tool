from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationService


def test_build_run_plan_defaults_to_embedded_dataset_size() -> None:
    svc = BenchmarkGenerationService()
    plan = svc.build_run_plan(
        acc_cpu=False,
        acc_cuda=True,
        acc_trt=False,
        acc_h8=False,
        acc_h10=False,
        hailo8_hw="hailo8",
        hailo10_hw="hailo10h",
        image_scale="auto",
        validation_images="",
        validation_max_images=0,
        hailo_preset="End-to-end compare",
        hailo_custom_full=True,
        hailo_custom_composed=True,
        hailo_custom_part1=False,
        hailo_custom_part2=False,
        matrix_trt_to_hailo=False,
        matrix_hailo_to_trt=False,
    )
    assert plan.bench_plan_runs
    assert all(int(r.get("validation_max_images") or 0) == 50 for r in plan.bench_plan_runs)


def test_runner_template_has_embedded_default_fallback() -> None:
    tpl = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt").read_text(encoding="utf-8")
    assert 'coco_50_data' in tpl
    assert 'max_images = 50' in tpl
