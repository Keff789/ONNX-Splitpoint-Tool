from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
CANARY = ROOT / "onnx_splitpoint_tool/resources/native_concurrent_three_stage_yolov7/three_stage_canary.py"
FIXTURE = ROOT / "tests/fixtures/v2793/yolov7_b066_three_stage_report.json"


def _load_helper():
    spec = importlib.util.spec_from_file_location("v2793_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_product_helper_and_vendored_copy_match():
    vendored = ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py"
    assert HELPER.read_bytes() == vendored.read_bytes()


def test_canary_count_is_parameterized_and_no_longer_hard_coded():
    text = CANARY.read_text(encoding="utf-8")
    assert '--expected-corpus-count' in text
    assert 'exact_corpus_count_required' in text
    assert 'exact_32_image_corpus_required' not in text


def test_helper_uses_explicit_corpus_reference_and_out_root():
    text = HELPER.read_text(encoding="utf-8")
    for marker in (
        '_prepare_one_image_smoke_corpus',
        '_run_vendored_canary_explicit',
        '--expected-corpus-count',
        '--reference-report',
        '--out-root',
        'product_execution_context',
    ):
        assert marker in text


def test_real_successful_report_projects_non_empty_stage_and_oracle_evidence():
    module = _load_helper()
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    stage = module._v2793_stage_timings(report)
    oracle = module._v2793_oracle_parity(report)
    assert stage['P1']['mean_ms_by_repetition']
    assert stage['P2']['mean_ms_by_repetition']
    assert stage['Post']['p95_ms_by_repetition']
    assert stage['aggregate']['raw_fps_median'] > 90.0
    assert oracle['status'] == 'passed'
    assert oracle['postflight']['all_exact'] is True


def test_real_report_endpoint_values_match_hardware_smoke():
    module = _load_helper()
    report = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert module._pick_number(report, 'raw_fps_median') == 97.077267591
    assert module._pick_number(report, 'completed_fps_median') == 97.058748157
