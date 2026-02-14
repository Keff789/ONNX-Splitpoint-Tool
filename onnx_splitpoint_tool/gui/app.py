"""GUI application entrypoint and incremental notebook shell."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk
from typing import Dict

from .. import __version__ as TOOL_VERSION
from .. import api as asc
from ..gui_app import SplitPointAnalyserGUI as LegacySplitPointAnalyserGUI
from ..gui_app import _setup_gui_logging
from .panels import panel_analysis, panel_export, panel_hardware, panel_logs, panel_validate

__version__ = TOOL_VERSION


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
        root_children = list(self.winfo_children())

        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.panel_frames: Dict[str, ttk.Frame] = {
            "analysis": panel_analysis.build_panel(self.main_notebook, app=self),
            "export": panel_export.build_panel(self.main_notebook, app=self),
            "validate": panel_validate.build_panel(self.main_notebook, app=self),
            "hardware": panel_hardware.build_panel(self.main_notebook, app=self),
            "logs": panel_logs.build_panel(self.main_notebook, app=self),
        }

        for key, label in self.TAB_LABELS:
            self.main_notebook.add(self.panel_frames[key], text=label)

        analysis_mount = getattr(self.panel_frames["analysis"], "legacy_mount", self.panel_frames["analysis"])
        for widget in root_children:
            widget.pack_forget()
            widget.pack(in_=analysis_mount, fill=tk.BOTH if widget is getattr(self, "mid_pane", None) else tk.X, expand=widget is getattr(self, "mid_pane", None), padx=0, pady=(0, 8) if widget is getattr(self, "params_frame", None) else 0)

    def _wire_model_type_state(self) -> None:
        if not hasattr(self, "gui_state"):
            return
        try:
            self.gui_state.model_type = str(getattr(self.gui_state, "model_type", "cv") or "cv")
        except Exception:
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
                pass

        for tab_key in ("export", "validate", "hardware"):
            if hasattr(self, "main_notebook"):
                try:
                    self.main_notebook.tab(self.panel_frames[tab_key], state="normal")
                except Exception:
                    pass


def main() -> None:
    """Start the Tk GUI application."""
    log_path = _setup_gui_logging()
    try:
        print(f"[GUI] {os.path.abspath(__file__)} (v{__version__})")
        print(f"[CORE] {os.path.abspath(getattr(asc, '__file__', ''))} (v{getattr(asc, '__version__', '?')})")
        if log_path:
            print(f"[LOG] {log_path}")
    except Exception:
        pass
    app = SplitPointAnalyserGUI()
    app.mainloop()
