from __future__ import annotations

from pathlib import Path


RUNNER_TEMPLATE = Path("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt")
SUITE_TEMPLATE = Path("onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt")
SERVICES = Path("onnx_splitpoint_tool/benchmark/services.py")
PANEL_VALIDATE = Path("onnx_splitpoint_tool/gui/panels/panel_validate.py")


def test_runner_template_exposes_validation_dataset_args() -> None:
    src = RUNNER_TEMPLATE.read_text(encoding="utf-8")
    assert '"--validation-images"' in src
    assert '"--validation-max-images"' in src
    assert '"--validation-pass-image-ratio"' in src
    assert '_collect_validation_image_paths(' in src
    assert '_aggregate_proxy_dataset_results(' in src


def test_runner_template_records_proxy_dataset_results() -> None:
    src = RUNNER_TEMPLATE.read_text(encoding="utf-8")
    assert 'comparisons[v]["proxy_dataset"]' in src
    assert 'comparisons[v]["passed_single_sample"]' in src
    assert 'validation_dataset=validation_dataset_summary' in src


def test_benchmark_suite_forwards_validation_dataset_args() -> None:
    src = SUITE_TEMPLATE.read_text(encoding="utf-8")
    assert 'validation_images: Optional[str] = None' in src
    assert 'validation_max_images: int = 0' in src
    assert '"--validation-images"' in src
    assert '"--validation-max-images"' in src
    assert 'validation_images=validation_images' in src
    assert 'validation_max_images=validation_max_images' in src


def test_services_and_panel_wire_validation_dataset_controls() -> None:
    svc = SERVICES.read_text(encoding="utf-8")
    panel = PANEL_VALIDATE.read_text(encoding="utf-8")
    assert 'validation_images: Optional[str]' in svc
    assert 'validation_max_images: int' in svc
    assert "'validation_images': plan_validation_images" in svc
    assert 'var_bench_validation_images' in panel
    assert 'var_bench_validation_max_images' in panel
