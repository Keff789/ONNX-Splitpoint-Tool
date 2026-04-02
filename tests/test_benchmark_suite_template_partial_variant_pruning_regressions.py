from __future__ import annotations

from pathlib import Path


ROOT = Path('onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt')


def test_template_knows_case_level_hailo_variant_availability() -> None:
    src = ROOT.read_text(encoding='utf-8')
    assert 'hailo_case_variant_availability' in src
    assert '_prune_case_variants_for_run' in src
    assert '_variant_hailo_requirements_for_run' in src


def test_template_prunes_variants_before_running_case() -> None:
    src = ROOT.read_text(encoding='utf-8')
    assert 'variants_eff = _prune_case_variants_for_run(run, variants, c, suite_hailo_hefs)' in src
    assert 'no usable variants after Hailo availability pruning' in src


def test_template_allows_suite_global_hailo_full_even_if_case_availability_is_stale() -> None:
    src = ROOT.read_text(encoding='utf-8')
    assert 'suite_hailo_hefs' in src
    assert 'suite_ok = bool(suite_meta.get("full"))' in src
    assert 'case_ok or suite_ok' in src
