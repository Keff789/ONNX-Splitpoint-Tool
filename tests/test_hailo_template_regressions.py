from __future__ import annotations

from pathlib import Path

import numpy as np


def _load_helper_ns() -> dict:
    text = Path('onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text(encoding='utf-8')
    start = text.index('def _adapt_tensor')
    end = text.index('class HailoSession:')
    snippet = text[start:end]
    ns = {
        '__name__': 'onnx_splitpoint_tool_template_helpers_test',
        'np': np,
        're': __import__('re'),
        'Optional': __import__('typing').Optional,
        'Tuple': __import__('typing').Tuple,
        'List': __import__('typing').List,
        'Dict': __import__('typing').Dict,
        'Sequence': __import__('typing').Sequence,
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



def test_remote_runner_self_check_rejects_missing_input_helpers(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.benchmark.remote_run import _assert_generated_runner_is_self_consistent

    bad = tmp_path / "run_split_onnxruntime.py"
    bad.write_text(
        "def main():\n"
        "    return _maybe_cast_for_onnx_input(x, 'tensor(float)')\n",
        encoding="utf-8",
    )

    try:
        _assert_generated_runner_is_self_consistent(bad)
    except RuntimeError as e:
        assert "_maybe_cast_for_onnx_input" in str(e)
    else:
        raise AssertionError("expected self-check to reject missing helper")


def test_template_hailo_layer_slot_parses_generic_stream_names() -> None:
    ns = _load_helper_ns()
    slot = ns['_hailo_layer_slot']

    assert slot('input_layer1') == 0
    assert slot('output_layer4') == 3
    assert slot('yolov7_part2_b89/input_layer3') == 2


def test_template_overlay_boundary_slot_mapping_bridges_manifest_and_hailo_generic_names() -> None:
    ns = _load_helper_ns()
    overlay = ns['_overlay_boundary_slot_mapping']

    expected_inputs = ['yolov7_part2_b89/input_layer1', 'yolov7_part2_b89/input_layer2']
    produced_outputs = ['cut_a', 'cut_b']
    mapping = overlay(
        {},
        expected_inputs=expected_inputs,
        produced_outputs=produced_outputs,
        part1_cut_names_manifest=['cut_a', 'cut_b'],
        part2_cut_names_manifest=['orig_in_a', 'orig_in_b'],
    )

    assert mapping[expected_inputs[0]] == 'cut_a'
    assert mapping[expected_inputs[1]] == 'cut_b'


def test_template_overlay_boundary_slot_mapping_bridges_hailo_outputs_into_manifest_inputs() -> None:
    ns = _load_helper_ns()
    overlay = ns['_overlay_boundary_slot_mapping']

    expected_inputs = ['orig_in_a', 'orig_in_b']
    produced_outputs = ['yolov7_part1_b89/output_layer1', 'yolov7_part1_b89/output_layer2']
    mapping = overlay(
        {},
        expected_inputs=expected_inputs,
        produced_outputs=produced_outputs,
        part1_cut_names_manifest=['cut_a', 'cut_b'],
        part2_cut_names_manifest=['orig_in_a', 'orig_in_b'],
    )

    assert mapping['orig_in_a'] == 'yolov7_part1_b89/output_layer1'
    assert mapping['orig_in_b'] == 'yolov7_part1_b89/output_layer2'


def test_template_imports_re_for_hailo_slot_and_yolo_name_helpers() -> None:
    src = Path('onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt').read_text(encoding='utf-8')
    assert 'import re' in src
    assert 're.search(' in src


def test_remote_runner_self_check_rejects_missing_re_import(tmp_path: Path) -> None:
    from onnx_splitpoint_tool.benchmark.remote_run import _assert_generated_runner_is_self_consistent

    bad = tmp_path / "run_split_onnxruntime.py"
    bad.write_text(
        "def _maybe_cast_for_onnx_input(x, t):\n"
        "    return x\n\n"
        "def _shape_from_ort_input(x):\n"
        "    return None\n\n"
        "def main():\n"
        "    return re.search('x', 'x')\n",
        encoding="utf-8",
    )

    try:
        _assert_generated_runner_is_self_consistent(bad)
    except RuntimeError as e:
        assert "module 're'" in str(e)
    else:
        raise AssertionError("expected self-check to reject missing re import")


def test_template_build_hailo_io_name_map_honors_preferred_slot_names() -> None:
    ns = _load_helper_ns()
    fn = ns['_build_hailo_io_name_map']

    mapping = fn(
        ['input_layer1', 'input_layer2', 'input_layer3', 'input_layer4'],
        ['mul_24', 'mul_25', 'mul_26', 'mul_58'],
        hailo_shapes={
            'input_layer1': (40, 40, 128),
            'input_layer2': (40, 40, 128),
            'input_layer3': (20, 20, 256),
            'input_layer4': (20, 20, 256),
        },
        onnx_shapes={
            'mul_24': (1, 128, 40, 40),
            'mul_25': (1, 128, 40, 40),
            'mul_26': (1, 256, 20, 20),
            'mul_58': (1, 256, 20, 20),
        },
        slot_names=['mul_58', 'mul_25', 'mul_24', 'mul_26'],
    )

    assert mapping['input_layer1'] == 'mul_58'
    assert mapping['input_layer2'] == 'mul_25'
    assert mapping['input_layer3'] == 'mul_24'
    assert mapping['input_layer4'] == 'mul_26'
