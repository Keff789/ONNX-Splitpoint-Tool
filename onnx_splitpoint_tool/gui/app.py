"""GUI application entrypoint and incremental notebook shell."""

from __future__ import annotations

import os
import logging
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict

from .. import __version__ as TOOL_VERSION
from .. import api as asc
from ..gui_app import SplitPointAnalyserGUI as LegacySplitPointAnalyserGUI
from ..gui_app import _setup_gui_logging
from .panels import panel_analysis, panel_split_export, panel_hardware, panel_logs, panel_validation

__version__ = TOOL_VERSION
logger = logging.getLogger(__name__)


class SplitPointAnalyserGUI(LegacySplitPointAnalyserGUI):
    """Notebook-based shell around the legacy GUI.

    The heavy UI/logic still lives in :mod:`onnx_splitpoint_tool.gui_app` and is
    migrated incrementally into ``gui.panels`` modules.
    """

    TAB_LABELS = (
        ("analysis", "Analyse"),
        ("export", "Split & Export"),
        ("validate", "Validierung & Benchmark"),
        ("hardware", "Hardware"),
        ("logs", "Logs"),
    )

    def __init__(self):
        super().__init__()
        self._init_central_notebook()
        self._wire_model_type_state()
        self._apply_model_type_visibility()

    def _init_central_notebook(self) -> None:
        logger.info("Initializing central notebook UI")
        root_children = list(self.winfo_children())

        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.panel_frames: Dict[str, ttk.Frame] = {
            "analysis": panel_analysis.build_panel(self.main_notebook, app=self),
            "export": panel_split_export.build_panel(self.main_notebook, app=self),
            "validate": panel_validation.build_panel(self.main_notebook, app=self),
            "hardware": panel_hardware.build_panel(self.main_notebook, app=self),
            "logs": panel_logs.build_panel(self.main_notebook, app=self),
        }

        for key, label in self.TAB_LABELS:
            self.main_notebook.add(self.panel_frames[key], text=label)

        panel_analysis.hide_legacy_widgets(root_children, self)
        logger.info("Central notebook initialized")

    def _handle_analysis_done(self, analysis_result) -> None:
        """Route analysis rendering through panel_analysis in the Tk main thread."""

        def _render() -> None:
            panel = self.panel_frames.get("analysis") if hasattr(self, "panel_frames") else None
            if panel is None:
                logger.warning("Analysis panel not initialized; falling back to legacy handler")
                super()._handle_analysis_done(analysis_result)
                return
            panel_analysis.render_analysis(panel, self, analysis_result)

        if threading.current_thread() is threading.main_thread():
            _render()
        else:
            self.after(0, _render)


    def _wire_model_type_state(self) -> None:
        if not hasattr(self, "gui_state"):
            return
        try:
            self.gui_state.model_type = str(getattr(self.gui_state, "model_type", "cv") or "cv")
        except Exception:
            logger.exception("Failed to read gui_state.model_type, falling back to 'cv'")
            self.gui_state.model_type = "cv"

        if hasattr(self, "var_llm_enable"):
            self.var_llm_enable.trace_add("write", lambda *_: self._on_llm_toggle())

    def _on_llm_toggle(self) -> None:
        self.gui_state.model_type = "llm" if bool(self.var_llm_enable.get()) else "cv"
        self._apply_model_type_visibility()

    def _apply_model_type_visibility(self) -> None:
        model_type = str(getattr(self.gui_state, "model_type", "cv") or "cv").lower()
        is_llm = model_type == "llm"

        if hasattr(self, "adv_tabs"):
            try:
                self.adv_tabs.tab(0, state="normal" if is_llm else "disabled")
                if (not is_llm) and int(self.adv_tabs.index("current")) == 0:
                    self.adv_tabs.select(1)
            except Exception:
                logger.exception("Failed to update advanced tabs visibility for model_type=%s", model_type)

        for tab_key in ("export", "validate", "hardware"):
            if hasattr(self, "main_notebook"):
                try:
                    self.main_notebook.tab(self.panel_frames[tab_key], state="normal")
                except Exception:
                    logger.exception("Failed to set notebook tab state for '%s'", tab_key)


    def _refresh_memory_fit_inspector(self) -> None:
        """Refresh the Memory Fit widget in the Analyse tab (Candidate Inspector).

        This is used when hardware selections (accelerators) change without changing the selected
        candidate row, so the Memory Fit bars update immediately.
        """
        try:
            panel = self.panel_frames.get("analysis") if hasattr(self, "panel_frames") else None
            if panel is None or not hasattr(panel, "memory_fit"):
                return

            boundary = None
            try:
                boundary = self._selected_boundary_index()
            except Exception:
                boundary = None
            if boundary is None:
                return

            # Use the Hardware tab selection as the source of truth (fallback to
            # legacy memory-forecast vars if needed).
            left_name = ""
            right_name = ""
            for attr in ("var_hw_left_accel", "var_memf_left_accel"):
                try:
                    left_name = getattr(self, attr).get()
                    break
                except Exception:
                    pass
            for attr in ("var_hw_right_accel", "var_memf_right_accel"):
                try:
                    right_name = getattr(self, attr).get()
                    break
                except Exception:
                    pass

            estimate = self._get_memory_stats_for_boundary(
                boundary, left_accel_name=left_name, right_accel_name=right_name
            )
            panel.memory_fit.update(estimate)
        except Exception:
            # Keep UI responsive even if memory estimation fails.
            return


def main() -> None:
    """Start the Tk GUI application."""
    log_path = _setup_gui_logging()
    try:
        print(f"[GUI] {os.path.abspath(__file__)} (v{__version__})")
        print(f"[CORE] {os.path.abspath(getattr(asc, '__file__', ''))} (v{getattr(asc, '__version__', '?')})")
        if log_path:
            print(f"[LOG] {log_path}")
    except Exception:
        logger.exception("Failed to print GUI startup metadata")
    app = SplitPointAnalyserGUI()
    app.mainloop()
