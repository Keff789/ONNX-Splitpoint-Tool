from __future__ import annotations

from pathlib import Path
import py_compile


def test_hailo_backend_prefers_activation_calib_even_for_single_input_part2() -> None:
    text = Path('onnx_splitpoint_tool/hailo_backend.py').read_text(encoding='utf-8')

    assert 'if activation_part1_onnx_p is not None:' in text
    assert 'elif len(input_layers) == 1:' in text
    assert 'activation_from_part1_preferred' in text


def test_benchmark_suite_template_supports_shared_trt_cache_root_and_session_flattening() -> None:
    text = Path('onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt').read_text(encoding='utf-8')

    assert '--trt-cache-root' in text
    assert 'trt_cache_root=args.trt_cache_root' in text
    assert 'session_build_s' in text
    assert 'baseline_ort_cpu' in text
    assert 'ort_runtime' in text


def test_run_split_template_records_runtime_session_build_info() -> None:
    text = Path('onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text(encoding='utf-8')

    assert 'ort_session_infos' in text
    assert 'baseline_ort_cpu' in text
    assert 'ort_runtime' in text
    compile(text, 'run_split_onnxruntime.py.txt', 'exec')


def test_remote_run_passes_shared_trt_cache_root_to_benchmark_suite() -> None:
    text = Path('onnx_splitpoint_tool/benchmark/remote_run.py').read_text(encoding='utf-8')

    assert 'remote_trt_cache_root' in text
    assert '--trt-cache-root' in text


def test_modified_python_sources_compile() -> None:
    py_compile.compile('onnx_splitpoint_tool/hailo_backend.py', doraise=True)
    py_compile.compile('onnx_splitpoint_tool/benchmark/remote_run.py', doraise=True)
