from __future__ import annotations

from pathlib import Path


def test_candidate_table_surfaces_hailo_parse_status() -> None:
    panel_src = Path("onnx_splitpoint_tool/gui/panels/panel_analysis.py").read_text(encoding="utf-8")
    assert '"hailo_parse"' in panel_src
    assert '("hailo_parse", "Hailo")' in panel_src
    assert 'app.tree.tag_configure("hailo_ok"' in panel_src

    cand_src = Path("onnx_splitpoint_tool/gui/panels/panel_candidates.py").read_text(encoding="utf-8")
    assert '"hailo_parse": tk.StringVar' in cand_src
    assert '("Hailo parse", "hailo_parse")' in cand_src
    assert 'hailo_parse_message' in cand_src

    gui_src = Path("onnx_splitpoint_tool/gui_app.py").read_text(encoding="utf-8")
    assert '"hailo_parse",' in gui_src
    assert 'self.tree.heading("hailo_parse", text="Hailo")' in gui_src
    assert 'self.tree.column("hailo_parse", width=60, anchor=tk.CENTER)' in gui_src
    assert 'def _hailo_parse_scalar_fields' in gui_src
    assert 'self.tree.tag_configure("hailo_ok"' in gui_src
    assert 'self.tree.tag_configure("hailo_fail"' in gui_src


def test_benchmark_export_persists_hailo_parse_and_robust_predictions() -> None:
    src = Path("onnx_splitpoint_tool/gui_app.py").read_text(encoding="utf-8")
    assert 'pred = self._analysis_predicted_metrics_for_boundary(a, int(b))' in src
    assert "manifest_out['hailo_parse_check'] = hailo_parse_entry" in src
    assert 'manifest_out.update(hailo_parse_fields)' in src
    assert "case_entry['hailo_parse_check'] = hailo_parse_entry" in src
    assert 'case_entry.update(hailo_parse_fields)' in src
    assert '"hailo_parse_checked"' in src
    assert '"hailo_parse_target"' in src
    assert 'split_candidates.csv' in src and 'hailo_parse_message' in src
    assert "bench['hailo_parse_check'] = {" in src
