"""GUI application entrypoint.

This module is the new home for startup/window bootstrap logic.
The current implementation still reuses the large legacy class from
``onnx_splitpoint_tool.gui_app`` while migration to panel/widget modules
happens incrementally.
"""

from __future__ import annotations

import os

from .. import __version__ as TOOL_VERSION
from .. import api as asc
from ..gui_app import SplitPointAnalyserGUI, _setup_gui_logging

__version__ = TOOL_VERSION


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
