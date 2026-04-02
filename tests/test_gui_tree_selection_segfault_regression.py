from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUI_APP = ROOT / "onnx_splitpoint_tool" / "gui_app.py"
PANEL_ANALYSIS = ROOT / "onnx_splitpoint_tool" / "gui" / "panels" / "panel_analysis.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_candidate_selection_path_uses_non_rebuilding_memory_refresh() -> None:
    src = _read(GUI_APP)
    assert "def _handle_candidate_selected" in src
    assert "self._refresh_memory_forecast(rebuild_table=False)" in src



def test_memory_forecast_supports_explicit_rebuild_mode() -> None:
    src = _read(GUI_APP)
    assert "def _refresh_memory_forecast(self, rebuild_table: bool = False)" in src
    assert "base_picks = [int(x) for x in (self.current_picks or [])]" in src
    assert "self._update_table(a, base_picks, self._last_params)" in src
    assert "filtered_picks" in src



def test_initial_tree_selection_no_longer_calls_handler_directly() -> None:
    src = _read(PANEL_ANALYSIS)
    assert "app.tree.selection_set(first)" in src
    assert "app._on_tree_selection_changed()" not in src
    assert "app.tree.after_idle(_apply_initial_selection)" in src


def test_tree_selection_is_coalesced_onto_idle_queue() -> None:
    src = _read(GUI_APP)
    assert "def _schedule_tree_selection_sync" in src
    assert "self._tree_selection_sync_job = self.after_idle(self._flush_tree_selection_sync)" in src
    assert "self._schedule_tree_selection_sync(boundary)" in src


def test_tree_selection_handler_no_longer_retags_rows() -> None:
    src = _read(GUI_APP)
    assert "Legacy no-op" in src
    assert "tree.item(iid, tags=tuple(cur_tags))" not in src
    assert "self._apply_selected_row_table_tag(boundary)" not in src
