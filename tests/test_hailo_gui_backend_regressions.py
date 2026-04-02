from __future__ import annotations

from pathlib import Path


def test_gui_params_accept_subprocess_backend_and_normalize_values() -> None:
    src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    assert 'backend_display_values' in src
    assert 'normalize_hailo_backend' in src
    assert 'Hailo backend must be one of: auto, subprocess, local, venv, wsl' in src
    hailo_src = Path('onnx_splitpoint_tool/hailo_backend.py').read_text(encoding='utf-8')
    assert 'subprocess_backend_for_platform' in hailo_src
    assert 'auto_prefers_subprocess' in hailo_src


def test_select_picks_uses_probe_auto_for_hailo_backend_precheck() -> None:
    src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    start = src.index('        if hailo_enabled:')
    end = src.index('        checks_budget =', start)
    block = src[start:end]
    assert 'from .hailo_backend import hailo_probe_auto' in block
    assert 'probe = hailo_probe_auto(' in block
    assert 'backend = normalize_hailo_backend' in block
    assert 'the configured backend is not ready' in block


def test_hardware_panel_exposes_subprocess_backend_option() -> None:
    src = Path('onnx_splitpoint_tool/gui/panels/panel_hardware.py').read_text(encoding='utf-8')
    assert 'backend_display_values' in src
    assert 'subprocess erzwingt immer den Subprozess-Backend' in src
    assert 'Venv override:' in src
