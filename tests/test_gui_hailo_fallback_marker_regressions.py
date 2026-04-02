from __future__ import annotations

from pathlib import Path


def test_gui_app_contains_fallback_marker_logic() -> None:
    text = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    assert "symbol = \"↪\" if used_alt else \"✅\"" in text
    assert "hailo_parse_used_suggested_end_nodes" in text
    assert "OK via fallback" in text
