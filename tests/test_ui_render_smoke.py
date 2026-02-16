from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.gui.panels import panel_analysis


class _Events:
    def on_model_loaded(self, _cb):
        return None

    def on_candidate_selected(self, _cb):
        return None


class _MockApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        # vars used by ANALYSIS_PARAM_SPECS
        self.var_topk = tk.StringVar(master=root, value="10")
        self.var_min_gap = tk.StringVar(master=root, value="1")
        self.var_min_compute = tk.StringVar(master=root, value="1")
        self.var_batch = tk.StringVar(master=root, value="")
        self.var_bpe = tk.StringVar(master=root, value="")
        self.var_unknown_mb = tk.StringVar(master=root, value="2.0")
        self.var_exclude_trivial = tk.BooleanVar(master=root, value=True)
        self.var_only_one = tk.BooleanVar(master=root, value=False)
        self.var_strict_boundary = tk.BooleanVar(master=root, value=True)
        self.var_rank = tk.StringVar(master=root, value="score")
        self.var_llm_enable = tk.BooleanVar(master=root, value=False)
        self.var_llm_preset = tk.StringVar(master=root, value="Standard")
        self.var_llm_mode = tk.StringVar(master=root, value="decode")
        self.var_llm_prefill = tk.StringVar(master=root, value="512")
        self.var_llm_decode = tk.StringVar(master=root, value="2048")
        self.var_llm_use_ort_symbolic = tk.BooleanVar(master=root, value=True)

        self.events = _Events()
        self.selected_candidate = None
        self.analysis = {}
        self._candidate_rows = []
        self.gui_state = SimpleNamespace(current_model_path="", model_type="onnx")

    def _on_analyse(self):
        return None

    def _split_selected_boundary(self):
        return None

    def _generate_benchmark_set(self):
        return None

    def _export_tex_table(self):
        return None

    def _refresh_candidates_table(self, *_args, **_kwargs):
        return None

    def _configure_candidate_columns(self):
        return None

    def _on_tree_selection_changed(self, *_args, **_kwargs):
        return None

    def _on_tree_motion_clean_tooltip(self, *_args, **_kwargs):
        return None

    def _hide_tree_clean_tooltip(self, *_args, **_kwargs):
        return None

    def _on_plot_click_select_candidate(self, *_args, **_kwargs):
        return None

    def _export_overview(self, *_args, **_kwargs):
        return None

    def _export_single(self, *_args, **_kwargs):
        return None

    def _infer_ui_state(self):
        return "ANALYSED"

    def _set_ui_state(self, _state):
        return None

    def _refresh_memory_forecast(self):
        return None

    def _update_diagnostics(self, _analysis):
        return None

    def _update_table(self, _analysis, picks, _params):
        self.tree.delete(*self.tree.get_children(""))
        for i, b in enumerate(picks):
            self.tree.insert("", "end", values=(
                i + 1,
                "✅",
                b,
                f"semantic-{b}",
                0.1,
                1,
                0.1,
                0.1,
                "A",
                "B",
                1,
                1,
                1,
                "Y",
                "Y",
                1,
                1,
            ))

    def _update_plots(self, _analysis, _picks, _params):
        self.ax_comm.plot([0, 1], [1, 2])
        self.canvas.draw_idle()


def test_render_analysis_smoke() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk not available in environment: {exc}")

    root.withdraw()
    app = _MockApp(root)
    frame = panel_analysis.build_panel(root, app=app)
    frame.pack(fill="both", expand=True)

    mock_result = {
        "candidates": [0, 1],
        "analysis": {"costs_bytes": [100, 200], "unknown_crossing_counts": [0, 0]},
        "picks": [0, 1],
        "params": object(),
    }
    panel_analysis.render_analysis(frame, app, mock_result)

    assert len(app.tree.get_children("")) > 0
    assert app.canvas is not None

    root.destroy()


def test_render_analysis_accepts_flat_payload_without_picks_params() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk not available in environment: {exc}")

    root.withdraw()
    app = _MockApp(root)
    app._last_params = object()
    frame = panel_analysis.build_panel(root, app=app)
    frame.pack(fill="both", expand=True)

    mock_result = {
        "candidates": [2, 3],
        "analysis": {"costs_bytes": [100, 200, 300, 400], "unknown_crossing_counts": [0, 0, 0, 0]},
    }
    panel_analysis.render_analysis(frame, app, mock_result)

    assert len(app.tree.get_children("")) == 2

    root.destroy()
