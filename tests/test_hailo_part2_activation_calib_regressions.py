from pathlib import Path


def test_hailo_backend_supports_activation_calib_for_multi_input_part2() -> None:
    text = Path('onnx_splitpoint_tool/hailo_backend.py').read_text(encoding='utf-8')
    assert 'activation_part1_onnx' in text
    assert 'activation_from_part1' in text
    assert '_build_activation_calib_from_part1_onnx' in text
    assert '_convert_calib_dataset_to_hn_shape' in text


def test_gui_part2_hef_export_passes_part1_onnx_for_activation_calib() -> None:
    text = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    assert 'activation_part1_onnx=p1_path' in text
    assert 'activation_gen_batch=int(hef_calib_bs)' in text


def test_wsl_inline_build_hef_accepts_activation_calib_args() -> None:
    text = Path('onnx_splitpoint_tool/wsl_inline_build_hef').read_text(encoding='utf-8')
    assert '--activation-part1-onnx' in text
    assert '--activation-gen-batch' in text
