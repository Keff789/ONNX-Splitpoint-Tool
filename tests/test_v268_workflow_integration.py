from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.remote_runtime_closure import (
    native_remote_package_closure,
)


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_forwards_performance_repetitions_to_all_native_paths() -> None:
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
    assert 'repetitions = int(native_execution_contract["repetitions"])' in source
    assert "verify_native_execution_contract(" in source
    assert 'performance_timeout = timeout * repetitions' in source
    assert source.count('"--repetitions", str(repetitions)') >= 7
    assert '"native_performance_aggregation": "median_never_best_of"' in source


def test_full_performance_does_not_inherit_energy_duration() -> None:
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
    full_block = source.split('fargs = [', 1)[1].split('shell_full =', 1)[0]
    assert '"--duration-s"' not in full_block
    assert 'exact frame-count contract' in full_block


def test_workflow_captures_nonblocking_pre_post_host_telemetry() -> None:
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
    assert '"native_host_telemetry.py"' in source
    assert '_capture_remote_host_telemetry("pre")' in source
    assert '_capture_remote_host_telemetry("post")' in source
    assert 'telemetry_group = f"{b}_performance"' in source
    assert '"--capture-group", _q(telemetry_group)' in source
    assert 'collect_failure_host_telemetry' in source
    assert 'if callable(telemetry_capture) and not telemetry_post_captured' in source
    assert '"claim_gate": False' in source
    assert 'native_host_telemetry_summary.json' in source


def test_workflow_stages_runtime_endpoint_attestor_with_remote_runners() -> None:
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
    closure = {path: (module, tokens) for path, module, tokens in native_remote_package_closure()}
    module, tokens = closure["onnx_splitpoint_tool/native_output_endpoint.py"]
    assert module == "onnx_splitpoint_tool.native_output_endpoint"
    assert {
        "def attest_decoded_nms",
        "def runtime_output_contract",
        "def load_manifest_outputs",
    } <= set(tokens)
    assert "native_remote_package_closure()" in source
    preprocessing_module, preprocessing_tokens = closure[
        "onnx_splitpoint_tool/preprocessing_contract.py"
    ]
    assert preprocessing_module == (
        "onnx_splitpoint_tool.preprocessing_contract"
    )
    assert {
        "def canonical_image_preprocessing_contract",
        "def prepare_rgb_uint8_image",
        "def runtime_numeric_input_identity",
    } <= set(preprocessing_tokens)


def test_deepx_stages_shared_trt_consumer_before_split_execution() -> None:
    source = (ROOT / "onnx_splitpoint_tool/workflow/runner.py").read_text(encoding="utf-8")
    dependency_region = source.split("dependency_specs = []", 1)[1].split(
        "package_runtime_closure = [", 1
    )[0]
    deepx_region = dependency_region.split('elif b == "deepx":', 1)[1]

    assert '"native_deepx_trt_e2e_from_benchmarkset.py"' in deepx_region
    assert '"native_hailo10_trt_e2e_from_benchmarkset.py"' in deepx_region
    assert '"class NativeTRT"' in deepx_region
    assert '"_aggregate_repetition_metrics"' in deepx_region
    shared_consumer_position = source.index(
        '"native_hailo10_trt_e2e_from_benchmarkset.py"',
        source.index("dependency_specs = []"),
    )
    split_execution_position = source.index('label=f"split:{b}:generic"')
    assert shared_consumer_position < split_execution_position
