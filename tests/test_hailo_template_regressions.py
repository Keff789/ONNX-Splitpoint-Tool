from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip('numpy')


def _load_helper_ns() -> dict:
    text = Path('onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text(encoding='utf-8')
    start = text.index('def _adapt_tensor')
    end = text.index('class HailoSession:')
    snippet = text[start:end]
    ns = {
        '__name__': 'onnx_splitpoint_tool_template_helpers_test',
        'np': np,
        'Optional': __import__('typing').Optional,
        'Tuple': __import__('typing').Tuple,
        'List': __import__('typing').List,
        'Dict': __import__('typing').Dict,
    }
    exec(compile(snippet, 'run_splitpoint_helpers.py', 'exec'), ns)
    return ns


def test_template_adapt_tensor_supports_generic_nchw_nhwc_channels() -> None:
    ns = _load_helper_ns()
    adapt = ns['_adapt_tensor']

    x = np.zeros((1, 256, 40, 40), dtype=np.float32)
    y = adapt(x, (40, 40, 256))
    assert y.shape == (40, 40, 256)

    x2 = np.zeros((40, 40, 256), dtype=np.float32)
    y2 = adapt(x2, (1, 256, 40, 40))
    assert y2.shape == (1, 256, 40, 40)


def test_template_select_source_tensor_prefers_shape_compatible_output() -> None:
    ns = _load_helper_ns()
    pick = ns['_select_source_tensor']

    out1_map = {
        'o0': np.zeros((1, 128, 80, 80), dtype=np.float32),
        'o1': np.zeros((1, 256, 40, 40), dtype=np.float32),
    }
    hit = pick('input_layer1', (40, 40, 256), out1_map, {})
    assert hit is not None
    src_name, adapted = hit
    assert src_name == 'o1'
    assert adapted.shape == (40, 40, 256)


def test_remote_preflight_probe_inherits_extra_sites() -> None:
    src = Path('onnx_splitpoint_tool/benchmark/remote_run.py').read_text(encoding='utf-8')
    assert 'SPLITPOINT_EXTRA_SITES' in src
    assert 'site.addsitedir(_p)' in src


def test_runner_template_writes_provenance_artifacts() -> None:
    src = Path('onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text(encoding='utf-8')
    assert 'artifacts_manifest.json' in src
    assert 'visualization_skipped.txt' in src
    assert 'generated_from_actual_outputs' in src
    assert 'variant_records' in src


def test_suite_template_surfaces_structured_variant_status() -> None:
    src = Path('onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt').read_text(encoding='utf-8')
    assert 'variant_records' in src
    assert 'reason_code' in src
    assert 'measured' in src
