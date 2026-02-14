"""Logs panel with file shortcuts and error copy helper."""

from __future__ import annotations

import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List


def _log_candidates() -> List[Path]:
    return [Path.cwd() / "gui.log", Path.home() / ".onnx_splitpoint_tool" / "gui.log"]


def _open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        try:
            subprocess.Popen(["open", str(path)])
        except Exception:
            subprocess.Popen(["notepad", str(path)])


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    box = tk.Text(frame, wrap="word", state="disabled")
    box.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 8))

    actions = ttk.Frame(frame)
    actions.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))

    def _read_log() -> str:
        for p in _log_candidates():
            if p.exists():
                try:
                    return p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
        return "No gui.log found yet."

    def _refresh() -> None:
        content = _read_log()
        box.configure(state="normal")
        box.delete("1.0", tk.END)
        box.insert("1.0", content)
        box.configure(state="disabled")

    def _open_log() -> None:
        for p in _log_candidates():
            if p.exists():
                _open_path(p)
                return
        messagebox.showinfo("gui.log", "No gui.log found.")

    def _copy_last_error() -> None:
        lines = _read_log().splitlines()
        error_lines = [ln for ln in lines if "error" in ln.lower() or "traceback" in ln.lower()]
        text = "\n".join(error_lines[-20:]).strip() or "No error line found in gui.log."
        frame.clipboard_clear()
        frame.clipboard_append(text)
        messagebox.showinfo("Logs", "Last error copied to clipboard.")

    ttk.Button(actions, text="Refresh", command=_refresh).pack(side=tk.LEFT)
    ttk.Button(actions, text="Open gui.log", command=_open_log).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(actions, text="Copy last error", command=_copy_last_error).pack(side=tk.LEFT, padx=(8, 0))

    _refresh()
    return frame
