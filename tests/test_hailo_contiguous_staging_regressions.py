from __future__ import annotations

from pathlib import Path

import numpy as np

from onnx_splitpoint_tool.runners.backends import hailo_backend as hb


def _load_template_helper_ns() -> dict:
    text = Path('onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text(encoding='utf-8')
    start = text.index('def _ensure_frames_dim')
    end = text.index('class HailoSession:')
    snippet = text[start:end]
    ns = {
        '__name__': 'onnx_splitpoint_tool_template_contiguous_helpers_test',
        'np': np,
        'Dict': __import__('typing').Dict,
        'Tuple': __import__('typing').Tuple,
        'Optional': __import__('typing').Optional,
    }
    exec(compile(snippet, 'run_splitpoint_contiguous_helpers.py', 'exec'), ns)
    return ns


def test_template_contiguous_cache_reuses_buffers() -> None:
    ns = _load_template_helper_ns()
    helper = ns['_ensure_c_contiguous_cached']

    src = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2)
    assert src.flags.c_contiguous is False

    cache: dict[str, np.ndarray] = {}
    out1 = helper(cache, 'x', src)
    out2 = helper(cache, 'x', src)

    assert out1.flags.c_contiguous is True
    assert np.array_equal(out1, src)
    assert out1 is out2



def test_runner_hailo_contiguous_cache_reuses_buffers() -> None:
    src = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5).transpose(1, 0, 2)
    assert src.flags.c_contiguous is False

    cache: dict[str, np.ndarray] = {}
    out1 = hb._ensure_c_contiguous_cached(cache, 'input_layer1', src)
    out2 = hb._ensure_c_contiguous_cached(cache, 'input_layer1', src)

    assert out1.flags.c_contiguous is True
    assert np.array_equal(out1, src)
    assert out1 is out2



def test_template_freezes_fixed_hailo_inputs_once() -> None:
    src = Path('onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text(encoding='utf-8')
    assert 'p1_inputs_hailo = _make_tensor_map_contiguous(p1_inputs_hailo)' in src
    assert 'full_inputs_hailo = _make_tensor_map_contiguous(full_inputs_hailo)' in src
    assert 'p2_inputs_0 = _make_tensor_map_contiguous(p2_inputs_0)' in src
