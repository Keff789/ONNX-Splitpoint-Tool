from __future__ import annotations

from pathlib import Path


def test_gui_params_accept_explicit_venv_backend_and_default_to_auto_override() -> None:
    src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    assert "{'auto', 'local', 'venv', 'wsl'}" in src
    assert 'hailo_wsl_venv_activate = (hailo_wsl_venv_activate or \'\').strip() or "auto"' in src


def test_select_picks_uses_probe_auto_for_hailo_backend_precheck() -> None:
    src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    start = src.index('        if hailo_enabled:')
    end = src.index('        checks_budget =', start)
    block = src[start:end]
    assert 'from .hailo_backend import hailo_probe_auto' in block
    assert 'probe = hailo_probe_auto(' in block
    assert 'backend not in {"auto", "local", "venv", "wsl"}' in block
    assert 'the configured backend is not ready' in block


def test_hardware_panel_exposes_venv_backend_option() -> None:
    src = Path('onnx_splitpoint_tool/gui/panels/panel_hardware.py').read_text(encoding='utf-8')
    assert 'values=["auto", "local", "venv", "wsl"]' in src
    assert 'Venv override:' in src
