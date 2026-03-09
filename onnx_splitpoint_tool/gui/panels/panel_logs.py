"""Logs panel.

Historically the tool wrote a `./gui.log` into the current working directory.
After adding Hailo support we noticed that 3rd party libraries (Hailo SDK)
also drop multiple `hailo_sdk.*.log` files into the *current working
directory*, which can quickly clutter the repository.

This panel therefore supports browsing multiple log files from:
  - ~/.onnx_splitpoint_tool/logs (new default)
  - ~/.onnx_splitpoint_tool (legacy)
  - ./ (legacy / working directory)
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

from ...paths import splitpoint_home, splitpoint_logs_dir


def _open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    # Platform-specific open.
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return

    subprocess.Popen(["xdg-open", str(path)])


def _iter_log_files(root: Path, *, recursive: bool, max_files: int = 400) -> List[Path]:
    if not root.exists():
        return []
    try:
        if recursive:
            files = list(root.rglob("*.log"))
        else:
            files = list(root.glob("*.log"))
    except Exception:
        return []

    # Sort newest first.
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except Exception:
            return 0.0

    files.sort(key=_mtime, reverse=True)
    return files[:max_files]


def _discover_logs() -> List[Path]:
    """Return a de-duplicated list of candidate log files."""

    logs: List[Path] = []

    # New default: ~/.onnx_splitpoint_tool/logs/**
    logs += _iter_log_files(splitpoint_logs_dir(), recursive=True)

    # Legacy: ~/.onnx_splitpoint_tool/*.log (non-recursive)
    logs += _iter_log_files(splitpoint_home(), recursive=False)

    # Legacy: current working dir (non-recursive)
    logs += _iter_log_files(Path.cwd(), recursive=False)

    # Ensure gui.log candidates are present first (if they exist).
    preferred = [
        splitpoint_logs_dir() / "gui.log",
        splitpoint_home() / "gui.log",
        Path.cwd() / "gui.log",
    ]

    seen: set[str] = set()
    out: List[Path] = []

    for p in preferred + logs:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp in seen:
            continue
        seen.add(rp)
        if p.exists():
            out.append(p)

    return out


def _label_for(path: Path) -> str:
    """Pretty label for a log path used in the dropdown."""

    try:
        rel = path.resolve().relative_to(splitpoint_logs_dir().resolve())
        return f"logs/{rel.as_posix()}"
    except Exception:
        pass

    try:
        rel = path.resolve().relative_to(splitpoint_home().resolve())
        return f"~/.onnx_splitpoint_tool/{rel.as_posix()}"
    except Exception:
        pass

    try:
        rel = path.resolve().relative_to(Path.cwd().resolve())
        return f"./{rel.as_posix()}"
    except Exception:
        pass

    return str(path)


def _read_tail_text(path: Path, *, max_bytes: int = 2_000_000) -> str:
    """Read a log file (best-effort), limiting memory usage for huge files."""

    try:
        with path.open("rb") as f:
            if f.seekable():
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size > max_bytes:
                    f.seek(max(0, size - max_bytes), os.SEEK_SET)
                    data = f.read()
                    txt = data.decode("utf-8", errors="replace")
                    return (
                        f"[showing last {max_bytes/1_000_000:.1f} MB of {size/1_000_000:.1f} MB]\n"
                        + txt
                    )
            data = f.read()
            return data.decode("utf-8", errors="replace")
    except Exception as e:
        return f"Failed to read {path}: {e}"


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)

    top = ttk.Frame(frame)
    top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
    top.columnconfigure(1, weight=1)

    ttk.Label(top, text="Log file:").grid(row=0, column=0, sticky="w")

    selected = tk.StringVar(value="")
    combo = ttk.Combobox(top, textvariable=selected, state="readonly")
    combo.grid(row=0, column=1, sticky="ew", padx=(8, 8))

    box = tk.Text(frame, wrap="word", state="disabled")
    box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))

    actions = ttk.Frame(frame)
    actions.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    # Keep a mapping from label->path.
    log_map: Dict[str, Path] = {}

    def _refresh_list(select_first: bool = False) -> None:
        nonlocal log_map
        paths = _discover_logs()
        log_map = { _label_for(p): p for p in paths }
        labels = list(log_map.keys())

        combo["values"] = labels

        # Keep selection if still present.
        cur = selected.get()
        if cur in log_map:
            return
        if labels:
            if select_first or not cur:
                selected.set(labels[0])
        else:
            selected.set("")

    def _current_path() -> Optional[Path]:
        lbl = selected.get().strip()
        if not lbl:
            return None
        return log_map.get(lbl)

    def _render_text(text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", tk.END)
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _refresh_content() -> None:
        p = _current_path()
        if not p:
            _render_text("No log file found yet.\n\nTip: run an action (benchmark/HEF build) and check again.")
            return
        if not p.exists():
            _render_text(f"Log file not found anymore: {p}")
            return
        _render_text(_read_tail_text(p))

    def _open_selected() -> None:
        p = _current_path()
        if not p:
            messagebox.showinfo("Logs", "No log file selected.")
            return
        try:
            _open_path(p)
        except Exception as e:
            messagebox.showerror("Logs", f"Failed to open log file:\n{p}\n\n{e}")

    def _open_folder() -> None:
        p = _current_path()
        if not p:
            messagebox.showinfo("Logs", "No log file selected.")
            return
        try:
            _open_path(p.parent)
        except Exception as e:
            messagebox.showerror("Logs", f"Failed to open folder:\n{p.parent}\n\n{e}")

    def _copy_last_error() -> None:
        p = _current_path()
        if not p or not p.exists():
            messagebox.showinfo("Logs", "No log file selected.")
            return
        lines = _read_tail_text(p).splitlines()
        error_lines = [ln for ln in lines if "error" in ln.lower() or "traceback" in ln.lower()]
        text = "\n".join(error_lines[-40:]).strip() or "No error line found in the selected log."
        frame.clipboard_clear()
        frame.clipboard_append(text)
        messagebox.showinfo("Logs", "Last error copied to clipboard.")

    # ------------------------------------------------------------------
    # Log retention helpers (best-effort)
    # ------------------------------------------------------------------

    enabled_var = getattr(app, "var_log_retention_enabled", None)
    days_var = getattr(app, "var_log_retention_days", None)
    max_files_var = getattr(app, "var_log_retention_max_files", None)
    if enabled_var is None:
        enabled_var = tk.BooleanVar(value=True)
    if days_var is None:
        days_var = tk.IntVar(value=30)
    if max_files_var is None:
        max_files_var = tk.IntVar(value=300)

    def _clean_logs() -> None:
        try:
            if app is not None and hasattr(app, "_apply_log_retention"):
                app._apply_log_retention(show_popup=True)  # type: ignore[attr-defined]
            else:
                from ...log_retention import LogRetentionPolicy, apply_log_retention

                pol = LogRetentionPolicy(
                    enabled=bool(enabled_var.get()),
                    max_age_days=int(days_var.get()),
                    max_files=int(max_files_var.get()),
                )
                apply_log_retention([splitpoint_logs_dir()], policy=pol, recursive=True)
        except Exception as e:
            messagebox.showwarning("Log retention", f"Cleanup failed: {e}")
        try:
            _refresh_list(select_first=False)
            _refresh_content()
        except Exception:
            pass

    def _on_combo_change(_event=None):
        _refresh_content()

    combo.bind("<<ComboboxSelected>>", _on_combo_change)

    btns = ttk.Frame(actions)
    btns.pack(side=tk.LEFT, fill=tk.X, expand=True)

    ttk.Button(btns, text="Refresh", command=_refresh_content).pack(side=tk.LEFT)
    ttk.Button(btns, text="Refresh list", command=lambda: (_refresh_list(False), _refresh_content())).pack(
        side=tk.LEFT, padx=(8, 0)
    )
    ttk.Button(btns, text="Open", command=_open_selected).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(btns, text="Open folder", command=_open_folder).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(btns, text="Copy last error", command=_copy_last_error).pack(side=tk.LEFT, padx=(8, 0))

    ret = ttk.Frame(actions)
    ret.pack(side=tk.RIGHT)
    ttk.Separator(ret, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=(8, 8))
    ttk.Checkbutton(ret, text="Auto-clean", variable=enabled_var).pack(side=tk.LEFT)
    ttk.Label(ret, text="days").pack(side=tk.LEFT, padx=(6, 2))
    ttk.Spinbox(ret, from_=0, to=365, width=5, textvariable=days_var).pack(side=tk.LEFT)
    ttk.Label(ret, text="max files").pack(side=tk.LEFT, padx=(8, 2))
    ttk.Spinbox(ret, from_=0, to=5000, width=6, textvariable=max_files_var).pack(side=tk.LEFT)
    ttk.Button(ret, text="Clean now", command=_clean_logs).pack(side=tk.LEFT, padx=(8, 0))

    # Initial fill.
    _refresh_list(select_first=True)
    _refresh_content()
    return frame
