from __future__ import annotations

import importlib
import py_compile
from pathlib import Path


def test_hailo_backend_source_mentions_recursive_image_calibration_support() -> None:
    text = Path('onnx_splitpoint_tool/hailo_backend.py').read_text(encoding='utf-8')
    assert '_CALIB_ITEM_EXTS' in text
    assert "'.png'" in text or '".png"' in text
    assert "'.jpg'" in text or '".jpg"' in text
    assert '_iter_calib_items(calib_dir, recursive=True)' in text
    assert '_load_calib_item_any' in text
    assert '_resize_hwc_image' in text
    assert 'No calibration items (.npy/.npz/images)' in text



def test_hailo_backend_python_compiles_after_image_calib_changes() -> None:
    py_compile.compile('onnx_splitpoint_tool/hailo_backend.py', doraise=True)



def test_gui_logging_source_exports_active_log_path_and_panel_uses_scrolledtext() -> None:
    gui_text = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    logs_text = Path('onnx_splitpoint_tool/gui/panels/panel_logs.py').read_text(encoding='utf-8')
    assert 'ONNX_SPLITPOINT_ACTIVE_LOG_PATH' in gui_text
    assert 'FileHandler(str(log_path)' in gui_text
    assert 'scrolledtext.ScrolledText' in logs_text
    assert '[log] {path}' in logs_text or '[log]' in logs_text



def test_panel_logs_discovers_active_log_path_and_read_tail_contains_header(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('HOME', str(tmp_path))
    active = tmp_path / 'custom_logs' / 'session.log'
    active.parent.mkdir(parents=True)
    active.write_text('hello\nworld\n', encoding='utf-8')
    monkeypatch.setenv('ONNX_SPLITPOINT_ACTIVE_LOG_PATH', str(active))

    from onnx_splitpoint_tool.gui.panels import panel_logs

    importlib.reload(panel_logs)

    logs = panel_logs._discover_logs()
    assert logs
    assert logs[0].resolve() == active.resolve()

    tail = panel_logs._read_tail_text(active)
    assert str(active) in tail
    assert 'hello' in tail
