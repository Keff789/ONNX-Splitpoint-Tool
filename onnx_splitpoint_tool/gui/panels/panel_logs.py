"""Logs panel.

Historically the tool wrote a `./gui.log` into the current working directory.
After adding Hailo support we noticed that 3rd party libraries (Hailo SDK)
also drop multiple `hailo_sdk.*.log` files into the *current working
directory*, which can quickly clutter the repository.

This panel therefore supports browsing multiple log files from:
  - ~/.onnx_splitpoint_tool/logs (new default)
  - ~/.onnx_splitpoint_tool (legacy)
  - ./ (legacy / working directory)
  - the currently active GUI log path exported via
    ONNX_SPLITPOINT_ACTIVE_LOG_PATH
"""

from __future__ import annotations

import os
import subprocess
import sys
import logging
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, List, Optional

from ...log_runtime import active_log_description, discover_live_logging_paths as _discover_live_runtime_log_paths, resolve_active_log_path
from ...paths import splitpoint_home, splitpoint_logs_dir


def _open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return

    subprocess.Popen(["xdg-open", str(path)])


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


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

    def _sort_key(p: Path):
        try:
            st = p.stat()
            return (float(st.st_mtime), int(st.st_size), p.name)
        except Exception:
            return (0.0, 0, p.name)

    files.sort(key=_sort_key, reverse=True)
    return files[:max_files]


def _discover_logs() -> List[Path]:
    """Return a de-duplicated list of candidate log files."""

    logs: List[Path] = []

    active_env = (os.environ.get("ONNX_SPLITPOINT_ACTIVE_LOG_PATH") or "").strip()
    preferred: List[Path] = []
    if active_env:
        preferred.append(Path(active_env).expanduser())

    preferred += _discover_live_runtime_log_paths()
    preferred += [
        splitpoint_logs_dir() / "gui.log",
        splitpoint_home() / "gui.log",
        Path.cwd() / "gui.log",
    ]

    logs += _iter_log_files(splitpoint_logs_dir(), recursive=True)
    logs += _iter_log_files(splitpoint_home(), recursive=False)
    logs += _iter_log_files(Path.cwd(), recursive=False)

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


def _discover_live_logging_paths() -> List[Path]:
    return _discover_live_runtime_log_paths()


def _label_for(path: Path) -> str:
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
            header = f"[log] {path}\n\n"
            if f.seekable():
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size > max_bytes:
                    f.seek(max(0, size - max_bytes), os.SEEK_SET)
                    data = f.read()
                    txt = data.decode("utf-8", errors="replace").replace("\\x00", "")
                    if not txt.strip():
                        txt = "[log file exists but selected tail is empty]\n"
                    return (
                        header
                        + f"[showing last {max_bytes/1_000_000:.1f} MB of {size/1_000_000:.1f} MB]\n\n"
                        + txt
                    )
                f.seek(0, os.SEEK_SET)
            data = f.read()
            txt = data.decode("utf-8", errors="replace").replace("\\x00", "")
            if not txt.strip():
                txt = "[log file exists but is empty]\n"
            return header + txt
    except Exception as e:
        return f"[log] {path}\n\nFailed to read log file: {e}"


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(1, weight=1)

    top = ttk.Frame(frame)
    top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
    top.columnconfigure(1, weight=1)

    ttk.Label(top, text="Log file:").grid(row=0, column=0, sticky="w")

    selected = tk.StringVar(value="")
    active_var = tk.StringVar(value=active_log_description())
    combo = ttk.Combobox(top, textvariable=selected, state="readonly")
    combo.grid(row=0, column=1, sticky="ew", padx=(8, 8))
    ttk.Label(top, textvariable=active_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    box = scrolledtext.ScrolledText(
        frame,
        wrap="word",
        state="normal",
        font=("Courier New", 10),
        background="white",
        foreground="black",
        insertbackground="black",
        borderwidth=1,
        relief="sunken",
    )
    box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
    box.bind("<Key>", lambda _evt: "break")

    actions = ttk.Frame(frame)
    actions.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    log_map: Dict[str, Path] = {}

    def _update_active_desc() -> None:
        active = resolve_active_log_path()
        if active is None:
            active_var.set("Active log: unknown")
        else:
            active_var.set(f"Active log: {active}")

    def _choose_default_label(labels: List[str]) -> str:
        if not labels:
            return ""
        cur = selected.get().strip()
        if cur in log_map:
            return cur
        active_env = (os.environ.get("ONNX_SPLITPOINT_ACTIVE_LOG_PATH") or "").strip()
        if active_env:
            active_p = Path(active_env).expanduser()
            for lbl, path in log_map.items():
                try:
                    if path.resolve() == active_p.resolve():
                        return lbl
                except Exception:
                    if str(path) == str(active_p):
                        return lbl
        non_empty = [lbl for lbl in labels if _file_size(log_map[lbl]) > 0]
        return non_empty[0] if non_empty else labels[0]

    def _refresh_list(select_first: bool = False) -> None:
        nonlocal log_map
        _update_active_desc()
        paths = _discover_logs()
        log_map = {_label_for(p): p for p in paths}
        labels = list(log_map.keys())
        combo["values"] = labels

        cur = selected.get().strip()
        if cur in log_map and not select_first:
            return
        selected.set(_choose_default_label(labels))

    def _current_path() -> Optional[Path]:
        lbl = selected.get().strip()
        if not lbl:
            return None
        return log_map.get(lbl)

    def _render_text(text: str) -> None:
        box.configure(state="normal")
        box.delete("1.0", tk.END)
        box.insert("1.0", text)
        box.see("1.0")

    def _refresh_content() -> None:
        p = _current_path()
        if not p:
            _render_text("No log file found yet.\n\nTip: run an action (benchmark/HEF build) and click 'Refresh list'.")
            return
        if not p.exists():
            _render_text(f"Log file not found anymore:\n{p}")
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

    def _select_active() -> None:
        active = resolve_active_log_path()
        if active is None:
            messagebox.showinfo("Logs", "No active log file detected yet.")
            return
        _refresh_list(select_first=False)
        target_key = None
        for label, path_obj in log_map.items():
            try:
                if path_obj.resolve() == active.resolve():
                    target_key = label
                    break
            except Exception:
                if str(path_obj) == str(active):
                    target_key = label
                    break
        if target_key is None:
            messagebox.showinfo("Logs", f"Active log is not in the current list yet:\n{active}")
            return
        selected.set(target_key)
        _refresh_content()

    def _open_active() -> None:
        active = resolve_active_log_path()
        if active is None:
            messagebox.showinfo("Logs", "No active log file detected yet.")
            return
        try:
            _open_path(active)
        except Exception as e:
            messagebox.showerror("Logs", f"Failed to open active log file:\n{active}\n\n{e}")

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

    combo.bind("<<ComboboxSelected>>", lambda _event=None: _refresh_content())

    btns = ttk.Frame(actions)
    btns.pack(side=tk.LEFT, fill=tk.X, expand=True)

    ttk.Button(btns, text="Refresh", command=_refresh_content).pack(side=tk.LEFT)
    ttk.Button(btns, text="Refresh list", command=lambda: (_refresh_list(False), _refresh_content())).pack(
        side=tk.LEFT, padx=(8, 0)
    )
    ttk.Button(btns, text="Select active", command=_select_active).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(btns, text="Open active", command=_open_active).pack(side=tk.LEFT, padx=(8, 0))
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

    _refresh_list(select_first=True)
    _refresh_content()

    _auto_refresh_job = {"id": None, "last_label": None, "last_mtime_ns": None, "ticks": 0}

    def _schedule_auto_refresh() -> None:
        try:
            _auto_refresh_job["id"] = frame.after(1500, _auto_refresh)
        except Exception:
            _auto_refresh_job["id"] = None

    def _auto_refresh() -> None:
        _auto_refresh_job["id"] = None
        try:
            _auto_refresh_job["ticks"] = int(_auto_refresh_job.get("ticks") or 0) + 1
            # Refresh the file list occasionally so newly created logs show up.
            if _auto_refresh_job["ticks"] % 4 == 0:
                _refresh_list(select_first=False)
            else:
                _update_active_desc()
            p = _current_path()
            if p is not None and p.exists():
                try:
                    st = p.stat()
                    mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
                except Exception:
                    mtime_ns = None
                label = selected.get().strip()
                if label != _auto_refresh_job.get("last_label") or mtime_ns != _auto_refresh_job.get("last_mtime_ns"):
                    _refresh_content()
                    _auto_refresh_job["last_label"] = label
                    _auto_refresh_job["last_mtime_ns"] = mtime_ns
        finally:
            if frame.winfo_exists():
                _schedule_auto_refresh()

    def _cancel_auto_refresh(_event=None) -> None:
        job = _auto_refresh_job.get("id")
        if job is not None:
            try:
                frame.after_cancel(job)
            except Exception:
                pass
            _auto_refresh_job["id"] = None

    frame.bind("<Destroy>", _cancel_auto_refresh, add=True)
    _schedule_auto_refresh()
    return frame
