from __future__ import annotations

from pathlib import Path


def test_update_table_reads_predicted_latency_arrays_from_analysis_payload() -> None:
    src = Path('onnx_splitpoint_tool/gui_app.py').read_text(encoding='utf-8')
    assert 'pred_lat_total = a.get("pred_latency_total_ms") or []' in src
    assert 'pred_lat_link = a.get("pred_latency_link_ms") or []' in src
    assert 'scores = a.get("scores") or {}' in src
