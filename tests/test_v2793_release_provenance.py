from pathlib import Path
import onnx_splitpoint_tool as pkg
from onnx_splitpoint_tool import v2793_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_historical_v2793_smoke_and_entrypoints_remain_available():
    assert v2793_smoke.VERSION == "2.79.3"
    assert v2793_smoke.LINEAGE == "v2.79"
    assert v2793_smoke.BUILD_ID == (
        "v2.79.3-native-productized-three-stage-endpoint-evidence"
    )
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'onnx-splitpoint-smoke-v2793 = '
        '"onnx_splitpoint_tool.v2793_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-79-3 = '
        '"onnx_splitpoint_tool.v2793_smoke:main"'
    ) in pyproject


def test_product_features_present():
    required = {
        "native_three_stage_productized_single_or_multi_image_corpus",
        "native_three_stage_explicit_corpus_reference_and_out_root",
        "native_three_stage_nonempty_stage_timing_projection",
        "native_three_stage_nonempty_oracle_parity_projection",
    }
    assert required.issubset(set(pkg.__build_features__))


def test_runner_copies_match():
    assert (ROOT / "scripts/native_hailo_trt_fifo_from_benchmarkset.py").read_bytes() == (ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_fifo_from_benchmarkset.py").read_bytes()
    assert (ROOT / "scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py").read_bytes() == (ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_hailo_trt_concurrent_three_stage_from_benchmarkset.py").read_bytes()
