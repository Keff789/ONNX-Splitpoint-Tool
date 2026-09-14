from pathlib import Path

import onnx_splitpoint_tool as pkg
from onnx_splitpoint_tool import v2794_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_historical_v2794_smoke_and_entrypoints_remain_available():
    assert v2794_smoke.VERSION == "2.79.4"
    assert v2794_smoke.LINEAGE == "v2.79"
    assert v2794_smoke.BUILD_ID == (
        "v2.79.4-native-productized-three-stage-release-consistency"
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'onnx-splitpoint-smoke-v2794 = '
        '"onnx_splitpoint_tool.v2794_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-79-4 = '
        '"onnx_splitpoint_tool.v2794_smoke:main"'
    ) in pyproject


def test_product_features_present():
    required = {
        "native_three_stage_productized_single_or_multi_image_corpus",
        "native_three_stage_explicit_corpus_reference_and_out_root",
        "native_three_stage_nonempty_stage_timing_projection",
        "native_three_stage_nonempty_oracle_parity_projection",
        "release_line_smoke_alias_tracks_current_maintenance_release",
    }
    assert required.issubset(set(pkg.__build_features__))


def test_runner_copies_match():
    assert (ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py").read_bytes() == (ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_fifo_from_benchmarkset.py").read_bytes()
    assert (ROOT / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py").read_bytes() == (ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py").read_bytes()
