"""GUI application entrypoint and incremental notebook shell."""

from __future__ import annotations

import os
import logging
import shutil
import subprocess
import sys
import time
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from datetime import datetime
from tkinter import ttk
from tkinter import messagebox
from tkinter import scrolledtext
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .. import __version__ as TOOL_VERSION
from .. import api as asc
from ..settings import SettingsStore
from ..remote import HostConfig, SSHTransport
from ..benchmark.remote_run import RemoteBenchmarkArgs, run_remote_benchmark
from ..benchmark.suite_refresh import refresh_suite_harness
from ..benchmark.services import RemoteBenchmarkCallbacks, RemoteBenchmarkController, RemoteBenchmarkService
from ..workdir import ensure_workdir
from ..gui_app import SplitPointAnalyserGUI as LegacySplitPointAnalyserGUI
from ..gui_app import _setup_gui_logging
from .hailo_diagnostics import format_hailo_diagnostics_text, load_hailo_result_json
from .hailo_parse_budget import normalize_persisted_hailo_max_checks
from .panels import panel_analysis, panel_split_export, panel_benchmark_analysis, panel_hardware, panel_jobs, panel_logs, panel_validation
from .widgets.text_progress_dialog import TextProgressDialog

__version__ = TOOL_VERSION
logger = logging.getLogger(__name__)


@dataclass
class BackgroundJobRecord:
    job_id: str
    kind: str
    type_label: str
    title: str
    name: str
    output_dir: str = ""
    log_path: str = ""
    status: str = "running"
    status_text: str = "Starting…"
    progress_value: float = 0.0
    progress_maximum: float = 100.0
    progress_display: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    last_message: str = ""
    log_lines: list[str] = field(default_factory=list)
    cancel_callback: Optional[Callable[[], None]] = None
    can_cancel: bool = False
    geometry: str = "760x460"
    monitor: Optional[TextProgressDialog] = None
    dismissed: bool = False



class SplitPointAnalyserGUI(LegacySplitPointAnalyserGUI):
    """Notebook-based shell around the legacy GUI.

    The heavy UI/logic still lives in :mod:`onnx_splitpoint_tool.gui_app` and is
    migrated incrementally into ``gui.panels`` modules.
    """

    TAB_LABELS = (
        ("analysis", "Analyse"),
        ("export", "Split & Export"),
        ("validate", "Benchmark"),
        ("bench_analysis", "Benchmark-Analyse"),
        ("jobs", "Jobs"),
        ("hardware", "Hardware"),
        ("logs", "Logs"),
    )

    def __init__(self):
        super().__init__()

        # The legacy GUI uses `self` as the Tk root. Some newer modules expect
        # a `.root` attribute as well, so keep an alias for compatibility.
        self.root = self
        self._hailo_gui_diag_history: list[dict] = []
        self._remote_service = RemoteBenchmarkService()
        self._remote_controller = RemoteBenchmarkController(service=self._remote_service)
        self._background_jobs: dict[str, BackgroundJobRecord] = {}
        self._background_job_order: list[str] = []

        # ------------------------------------------------------------------
        # Persistent settings
        # ------------------------------------------------------------------
        self._settings_store = SettingsStore()

        # Remote benchmarking vars (must exist before applying persisted tk_vars)
        self.remote_hosts = []
        self.var_remote_host_id = tk.StringVar(value="")
        self.var_remote_benchmark_set = tk.StringVar(value="")
        self.var_remote_provider = tk.StringVar(value="auto")
        # Keep remote numeric entries as StringVars so temporarily empty fields do not
        # trigger TclError callbacks while the user edits them.
        self.var_remote_warmup = tk.StringVar(value="10")
        self.var_remote_iters = tk.StringVar(value="100")
        self.var_remote_repeats = tk.StringVar(value="1")
        self.var_remote_timeout = tk.StringVar(value="7200")
        self.var_remote_throughput_frames = tk.StringVar(value="24")
        self.var_remote_throughput_warmup_frames = tk.StringVar(value="6")
        self.var_remote_throughput_queue_depth = tk.StringVar(value="2")
        self.var_remote_add_args = tk.StringVar(value="")
        self.var_remote_venv = tk.StringVar(value="")
        self.var_remote_transfer_mode = tk.StringVar(value="bundle")
        self.var_remote_reuse_bundle = tk.BooleanVar(value=True)

        # Log retention (cleanup)
        # These are persisted automatically (all var_* Tk variables are saved).
        self.var_log_retention_enabled = tk.BooleanVar(value=True)
        self.var_log_retention_days = tk.IntVar(value=30)
        self.var_log_retention_max_files = tk.IntVar(value=300)

        # Persistent settings
        # NOTE: Some Tk variables are created by notebook panels. We therefore apply settings
        # twice: once early (for global vars like remote host / output dir), and once again
        # after the notebook panels have created their Tk variables.
        persisted = self._settings_store.load()
        self._apply_persistent_settings(persisted)

        # Ensure the working-directory structure exists under the selected
        # output folder. This makes remote benchmarking + artifact management
        # predictable.
        try:
            if getattr(self, "default_output_dir", None):
                ensure_workdir(Path(self.default_output_dir))
        except Exception:
            pass

        self._init_central_notebook()

        # Re-apply now that all panel Tk variables exist
        self._apply_persistent_settings(persisted)
        self._wire_model_type_state()
        self._apply_model_type_visibility()

        # Apply log retention shortly after startup (best-effort).
        try:
            self.after(800, lambda: self._apply_log_retention(show_popup=False))
        except Exception:
            pass

        # Save settings on close
        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        except Exception:
            try:
                self.protocol("WM_DELETE_WINDOW", self._on_close)
            except Exception:
                pass

        # Auto-probe Hailo availability on startup so users immediately see
        # whether Hailo-8 / Hailo-10 profiles are usable.
        try:
            self.after(250, getattr(self, "_hailo_refresh_status", lambda: None))
        except Exception:
            pass

    _JOB_STATUS_LABELS = {
        "queued": "Queued",
        "running": "Running",
        "cancelling": "Cancelling…",
        "success": "Finished",
        "warning": "Finished (warnings)",
        "error": "Failed",
        "cancelled": "Cancelled",
    }
    _JOB_STATUS_COLORS = {
        "idle": "#616161",
        "queued": "#616161",
        "running": "#1565c0",
        "cancelling": "#546e7a",
        "success": "#2e7d32",
        "warning": "#ef6c00",
        "error": "#c62828",
        "cancelled": "#757575",
    }

    # ------------------------------------------------------------------
    # Background jobs helpers
    # ------------------------------------------------------------------

    def _jobs_status_label(self, status: str) -> str:
        return str(self._JOB_STATUS_LABELS.get(str(status or "").strip().lower(), status or "Idle"))

    def _jobs_status_color(self, status: str) -> str:
        return str(self._JOB_STATUS_COLORS.get(str(status or "").strip().lower(), self._JOB_STATUS_COLORS["idle"]))

    def _jobs_format_progress(self, record: Optional[BackgroundJobRecord]) -> str:
        if record is None:
            return ""
        if str(record.progress_display or "").strip():
            return str(record.progress_display).strip()
        maxv = float(record.progress_maximum or 0.0)
        cur = float(record.progress_value or 0.0)
        if maxv <= 0:
            return ""
        if maxv <= 1.000001:
            pct = int(round(max(0.0, min(1.0, cur / maxv)) * 100.0))
            return f"{pct}%"
        if float(maxv).is_integer() and float(cur).is_integer():
            return f"{int(cur)}/{int(maxv)}"
        return f"{cur:.1f}/{maxv:.1f}"

    def _jobs_summary_text(self, kind: str, record: Optional[BackgroundJobRecord]) -> str:
        title = "Generate" if str(kind) == "generate" else "Remote run"
        if record is None:
            return f"{title}: idle"
        status = str(record.status or "").strip().lower()
        progress = self._jobs_format_progress(record)
        if status in {"running", "cancelling"}:
            if kind == "generate" and progress:
                return f"{title}: {status} ({progress})"
            if progress:
                return f"{title}: {progress}"
            return f"{title}: {status}"
        if progress:
            return f"{title}: {self._jobs_status_label(status).lower()} ({progress})"
        return f"{title}: {self._jobs_status_label(status).lower()}"

    def _jobs_latest_record(self, kind: str) -> Optional[BackgroundJobRecord]:
        wanted = str(kind or "").strip().lower()
        for job_id in reversed(list(getattr(self, "_background_job_order", []) or [])):
            record = (getattr(self, "_background_jobs", {}) or {}).get(job_id)
            if record is None or bool(getattr(record, "dismissed", False)):
                continue
            if str(getattr(record, "kind", "") or "").strip().lower() == wanted:
                return record
        return None

    def _jobs_register(
        self,
        *,
        job_id: str,
        kind: str,
        type_label: str,
        title: str,
        name: str,
        output_dir: str = "",
        log_path: str = "",
        initial_status: str = "Starting…",
        initial_lines: Optional[list[str]] = None,
        progress_maximum: float = 100.0,
        cancel_callback: Optional[Callable[[], None]] = None,
        can_cancel: bool = False,
        geometry: str = "760x460",
        show_monitor: bool = True,
    ) -> BackgroundJobRecord:
        record = BackgroundJobRecord(
            job_id=str(job_id),
            kind=str(kind),
            type_label=str(type_label),
            title=str(title),
            name=str(name),
            output_dir=str(output_dir or ""),
            log_path=str(log_path or ""),
            status="running",
            status_text=str(initial_status or "Starting…"),
            progress_value=0.0,
            progress_maximum=max(1.0 if float(progress_maximum or 0.0) <= 0.0 else float(progress_maximum), 1.0e-9),
            last_message=str(initial_status or "Starting…"),
            log_lines=[str(line).rstrip("\n") for line in list(initial_lines or [])],
            cancel_callback=cancel_callback,
            can_cancel=bool(can_cancel and callable(cancel_callback)),
            geometry=str(geometry or "760x460"),
        )
        self._background_jobs[record.job_id] = record
        if record.job_id not in self._background_job_order:
            self._background_job_order.append(record.job_id)
        self._jobs_refresh_views()
        if show_monitor:
            self._jobs_open_monitor(record.job_id)
        return record

    def _jobs_update_paths(self, job_id: str, *, output_dir: Optional[str] = None, log_path: Optional[str] = None) -> None:
        record = self._background_jobs.get(str(job_id))
        if record is None:
            return
        if output_dir is not None:
            record.output_dir = str(output_dir or "")
        if log_path is not None:
            record.log_path = str(log_path or "")
        self._jobs_refresh_views()

    def _jobs_append_log(self, job_id: str, line: str) -> None:
        record = self._background_jobs.get(str(job_id))
        if record is None:
            return
        text = str(line or "").rstrip("\n")
        if not text:
            return
        record.log_lines.append(text)
        if len(record.log_lines) > 5000:
            record.log_lines = record.log_lines[-5000:]
        stripped = text.strip()
        if stripped:
            record.last_message = stripped
        if record.monitor is not None and bool(getattr(record.monitor, "alive", False)):
            try:
                record.monitor.append(text)
            except Exception:
                logger.debug("Failed to append job log to monitor for %s", job_id, exc_info=True)
        self._jobs_refresh_views()

    def _jobs_set_progress(
        self,
        job_id: str,
        *,
        value: float,
        label: Optional[str] = None,
        display: Optional[str] = None,
        progress_maximum: Optional[float] = None,
    ) -> None:
        record = self._background_jobs.get(str(job_id))
        if record is None:
            return
        if progress_maximum is not None:
            try:
                record.progress_maximum = max(float(progress_maximum), 1.0e-9)
            except Exception:
                pass
        try:
            record.progress_value = max(0.0, min(float(record.progress_maximum or 1.0), float(value)))
        except Exception:
            pass
        if display is not None:
            record.progress_display = str(display or "")
        if label is not None:
            record.status_text = str(label or record.status_text)
            stripped = str(label or "").strip()
            if stripped:
                record.last_message = stripped
        if record.monitor is not None and bool(getattr(record.monitor, "alive", False)):
            try:
                record.monitor.set_absolute_progress(record.progress_value, record.status_text)
            except Exception:
                logger.debug("Failed to update job monitor progress for %s", job_id, exc_info=True)
        self._jobs_refresh_views()

    def _jobs_finish(
        self,
        job_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        output_dir: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> None:
        record = self._background_jobs.get(str(job_id))
        if record is None:
            return
        if output_dir is not None:
            record.output_dir = str(output_dir or "")
        if log_path is not None:
            record.log_path = str(log_path or "")
        record.status = str(status or "success").strip().lower()
        record.status_text = self._jobs_status_label(record.status)
        record.end_time = datetime.now()
        record.can_cancel = False
        if record.status in {"success", "warning"} and float(record.progress_maximum or 0.0) > 0.0:
            record.progress_value = max(record.progress_value, record.progress_maximum)
        if message is not None:
            stripped = str(message or "").strip()
            if stripped:
                record.last_message = stripped
        if record.monitor is not None and bool(getattr(record.monitor, "alive", False)):
            try:
                record.monitor.finish(status_text=record.status_text)
                record.monitor.btn_cancel.configure(state="disabled")
            except Exception:
                logger.debug("Failed to finalize job monitor for %s", job_id, exc_info=True)
        self._jobs_refresh_views()

    def _jobs_request_cancel(self, job_id: str) -> None:
        record = self._background_jobs.get(str(job_id))
        if record is None:
            return
        if str(record.status or "").strip().lower() not in {"queued", "running", "cancelling"}:
            return
        if not bool(record.can_cancel) or not callable(record.cancel_callback):
            messagebox.showinfo("Cancel job", "This job cannot be cancelled from the GUI.")
            return
        record.status = "cancelling"
        record.status_text = self._jobs_status_label("cancelling")
        record.last_message = "Cancel requested…"
        if record.monitor is not None and bool(getattr(record.monitor, "alive", False)):
            try:
                record.monitor.append("[ui] Cancel requested…")
                record.monitor.set_status(record.status_text)
                record.monitor.btn_cancel.configure(state="disabled")
            except Exception:
                logger.debug("Failed to push cancel request to monitor for %s", job_id, exc_info=True)
        try:
            record.cancel_callback()
        except Exception as exc:
            logger.exception("Failed to cancel job %s", job_id)
            messagebox.showerror("Cancel job", f"Could not request cancellation:\n\n{exc}")
        self._jobs_refresh_views()

    def _jobs_on_monitor_closed(self, job_id: str) -> None:
        record = self._background_jobs.get(str(job_id))
        if record is not None:
            record.monitor = None
        self._jobs_refresh_views()

    def _jobs_open_monitor(self, job_id: str) -> None:
        record = self._background_jobs.get(str(job_id))
        if record is None:
            return
        existing = getattr(record, "monitor", None)
        if existing is not None and bool(getattr(existing, "alive", False)):
            try:
                existing.window.deiconify()
                existing.window.lift()
                existing.window.focus_force()
            except Exception:
                pass
            return

        dialog = TextProgressDialog(
            self.root,
            title=record.title,
            initial_status=record.status_text,
            initial_lines=list(record.log_lines or []),
            progress_mode="determinate",
            progress_maximum=float(record.progress_maximum or 100.0),
            on_cancel=(lambda _job_id=record.job_id: self._jobs_request_cancel(_job_id)) if record.can_cancel else None,
            on_close=(lambda _job_id=record.job_id: self._jobs_on_monitor_closed(_job_id)),
            geometry=str(record.geometry or "760x460"),
        )
        record.monitor = dialog
        try:
            dialog.set_absolute_progress(record.progress_value, record.status_text)
        except Exception:
            pass
        if not bool(record.can_cancel):
            try:
                dialog.btn_cancel.configure(state="disabled")
            except Exception:
                pass
        if str(record.status or "").strip().lower() not in {"queued", "running", "cancelling"}:
            try:
                dialog.finish(status_text=record.status_text)
            except Exception:
                pass
        self._jobs_refresh_views()

    def _jobs_selected_job_id(self) -> Optional[str]:
        tree = getattr(self, "jobs_tree", None)
        if tree is None:
            return None
        try:
            selection = tree.selection()
        except Exception:
            selection = ()
        if not selection:
            return None
        return str(selection[0])

    def _jobs_refresh_tree(self) -> None:
        tree = getattr(self, "jobs_tree", None)
        if tree is None:
            return
        try:
            selected = self._jobs_selected_job_id()
        except Exception:
            selected = None

        try:
            tree.delete(*tree.get_children())
        except Exception:
            return

        try:
            tree.tag_configure("job_running", background="#e3f2fd")
            tree.tag_configure("job_cancelling", background="#eceff1")
            tree.tag_configure("job_success", background="#e8f5e9")
            tree.tag_configure("job_warning", background="#fff3e0")
            tree.tag_configure("job_error", background="#ffebee")
            tree.tag_configure("job_cancelled", background="#f5f5f5")
        except Exception:
            pass

        for job_id in reversed(list(self._background_job_order or [])):
            record = self._background_jobs.get(job_id)
            if record is None or bool(record.dismissed):
                continue
            name_target = str(record.name or "")
            if str(record.output_dir or "").strip():
                name_target = f"{name_target} — {record.output_dir}" if name_target else str(record.output_dir)
            last_message = str(record.last_message or "").replace("\n", " ").strip()
            values = (
                str(record.type_label or ""),
                name_target,
                self._jobs_status_label(record.status),
                self._jobs_format_progress(record),
                record.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                last_message[:220],
            )
            tag = f"job_{str(record.status or 'running').strip().lower()}"
            try:
                tree.insert("", "end", iid=record.job_id, values=values, tags=(tag,))
            except Exception:
                logger.debug("Failed to insert Jobs row for %s", job_id, exc_info=True)

        if selected and selected in self._background_jobs and not bool(self._background_jobs[selected].dismissed):
            try:
                tree.selection_set(selected)
                tree.focus(selected)
            except Exception:
                pass
        self._jobs_update_panel_buttons()

    def _jobs_refresh_status_bar(self) -> None:
        for kind, attr in (("generate", "job_bar_generate"), ("remote_run", "job_bar_remote")):
            widget = getattr(self, attr, None)
            if widget is None:
                continue
            record = self._jobs_latest_record(kind)
            status = str(getattr(record, "status", "idle") or "idle") if record is not None else "idle"
            text = self._jobs_summary_text(kind, record)
            try:
                widget.configure(text=text, bg=self._jobs_status_color(status), fg="white")
            except Exception:
                logger.debug("Failed to refresh jobs status bar for %s", kind, exc_info=True)

    def _jobs_refresh_views(self) -> None:
        self._jobs_refresh_status_bar()
        self._jobs_refresh_tree()

    def _jobs_update_panel_buttons(self) -> None:
        record = self._background_jobs.get(self._jobs_selected_job_id() or "")
        running = record is not None and str(record.status or "").strip().lower() in {"queued", "running", "cancelling"}
        finished = record is not None and not running
        for attr in ("btn_jobs_open_monitor", "btn_jobs_open_log", "btn_jobs_open_output"):
            btn = getattr(self, attr, None)
            if btn is not None:
                try:
                    btn.configure(state=("normal" if record is not None else "disabled"))
                except Exception:
                    pass
        btn_cancel = getattr(self, "btn_jobs_cancel", None)
        if btn_cancel is not None:
            try:
                btn_cancel.configure(state=("normal" if record is not None and running and bool(record.can_cancel) else "disabled"))
            except Exception:
                pass
        btn_dismiss = getattr(self, "btn_jobs_dismiss", None)
        if btn_dismiss is not None:
            try:
                btn_dismiss.configure(state=("normal" if finished else "disabled"))
            except Exception:
                pass

    def _jobs_open_monitor_selected(self) -> None:
        job_id = self._jobs_selected_job_id()
        if job_id:
            self._jobs_open_monitor(job_id)

    def _jobs_open_log_selected(self) -> None:
        record = self._background_jobs.get(self._jobs_selected_job_id() or "")
        if record is None:
            return
        path = str(record.log_path or "").strip()
        if not path:
            messagebox.showinfo("Open log", "This job has no log path yet.")
            return
        self._open_path(path)

    def _jobs_open_output_selected(self) -> None:
        record = self._background_jobs.get(self._jobs_selected_job_id() or "")
        if record is None:
            return
        path = str(record.output_dir or "").strip()
        if not path:
            messagebox.showinfo("Open output", "This job has no output folder yet.")
            return
        self._open_path(path)

    def _jobs_cancel_selected(self) -> None:
        job_id = self._jobs_selected_job_id()
        if job_id:
            self._jobs_request_cancel(job_id)

    def _jobs_dismiss_selected(self) -> None:
        job_id = self._jobs_selected_job_id()
        if not job_id:
            return
        record = self._background_jobs.get(job_id)
        if record is None:
            return
        if str(record.status or "").strip().lower() in {"queued", "running", "cancelling"}:
            messagebox.showinfo("Dismiss job", "Running jobs cannot be dismissed. Close the monitor window instead.")
            return
        existing = getattr(record, "monitor", None)
        if existing is not None and bool(getattr(existing, "alive", False)):
            try:
                existing.window.destroy()
            except Exception:
                pass
        record.monitor = None
        record.dismissed = True
        self._jobs_refresh_views()

    def _open_path(self, path: str) -> None:
        target = str(path or "").strip()
        if not target:
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(target)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", target])
            else:
                opener = shutil.which("xdg-open")
                if opener:
                    subprocess.Popen([opener, target])
                else:
                    raise RuntimeError("No desktop opener (xdg-open) available")
        except Exception as exc:
            try:
                self.clipboard_clear()
                self.clipboard_append(target)
                self.update_idletasks()
            except Exception:
                pass
            messagebox.showinfo(
                "Open path",
                f"Could not open this path automatically. The path was copied to the clipboard.\n\n{target}\n\n{exc}",
            )

    # ------------------------------------------------------------------
    # Persistent settings helpers
    # ------------------------------------------------------------------

    def _apply_persistent_settings(self, settings: dict) -> None:
        """Apply persisted settings to GUI state (best-effort)."""
        if not isinstance(settings, dict):
            return

        # Restore output/working dir
        out_dir = settings.get("output_dir")
        if isinstance(out_dir, str) and out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
                self.default_output_dir = out_dir
                if hasattr(self, "gui_state"):
                    self.gui_state.output_dir = out_dir
                disp = out_dir
                try:
                    disp = self._abbrev_path(out_dir)
                except Exception:
                    pass
                try:
                    self.output_dir_label_var.set(disp)
                except Exception:
                    pass
            except Exception:
                pass

        # Restore remote hosts
        hosts = settings.get("remote_hosts")
        if isinstance(hosts, list):
            self.remote_hosts = hosts

        # Restore tkinter variables
        tk_vars = settings.get("tk_vars")
        if isinstance(tk_vars, dict):
            for name, value in tk_vars.items():
                if name == "var_hailo_max_checks":
                    try:
                        value = normalize_persisted_hailo_max_checks(value)
                    except Exception:
                        pass
                var = getattr(self, name, None)
                if isinstance(var, tk.Variable):
                    try:
                        var.set(value)
                    except Exception:
                        pass

        sel = settings.get("remote_selected_host_id")
        if isinstance(sel, str) and sel:
            try:
                self.var_remote_host_id.set(sel)
            except Exception:
                pass

    def _collect_persistent_settings(self) -> dict:
        tk_vars = {}
        for name, value in self.__dict__.items():
            if name.startswith("var_") and isinstance(value, tk.Variable):
                try:
                    tk_vars[name] = value.get()
                except Exception:
                    pass

        data = self._settings_store.load()
        data["tk_vars"] = tk_vars
        data["output_dir"] = getattr(self, "default_output_dir", None)
        data["remote_hosts"] = getattr(self, "remote_hosts", [])
        try:
            data["remote_selected_host_id"] = self.var_remote_host_id.get()
        except Exception:
            pass
        return data

    def _persist_settings(self) -> None:
        try:
            self._settings_store.save(self._collect_persistent_settings())
        except Exception:
            logger.exception("Failed to save settings")

    def _on_close(self):
        self._persist_settings()
        try:
            self.destroy()
        except Exception:
            try:
                self.root.destroy()
            except Exception:
                pass

    def _on_pick_output_folder(self):
        super()._on_pick_output_folder()
        self._persist_settings()

    # ------------------------------------------------------------------
    # Log retention helpers (Logs tab)
    # ------------------------------------------------------------------

    def _apply_log_retention(self, *, show_popup: bool = False) -> None:
        """Best-effort cleanup of old log files.

        This is intentionally non-fatal. Any exception is swallowed.
        """

        try:
            enabled = True
            try:
                enabled = bool(self.var_log_retention_enabled.get())
            except Exception:
                enabled = True
            if not enabled:
                return

            try:
                days = int(self.var_log_retention_days.get())
            except Exception:
                days = 30
            try:
                max_files = int(self.var_log_retention_max_files.get())
            except Exception:
                max_files = 300

            from ..paths import splitpoint_logs_dir
            from ..log_retention import LogRetentionPolicy, apply_log_retention

            roots = [splitpoint_logs_dir(), Path.home() / ".onnx_splitpoint_tool" / "wsl_debug"]
            pol = LogRetentionPolicy(enabled=True, max_age_days=days, max_files=max_files)
            stats = apply_log_retention(roots, policy=pol, recursive=True)

            removed = int(stats.get("removed") or 0)
            freed_mb = float(stats.get("freed_bytes") or 0) / (1024.0 * 1024.0)
            errs = int(stats.get("errors") or 0)
            logger.info("Log retention: removed=%s freed=%.1fMB errors=%s", removed, freed_mb, errs)

            if show_popup:
                messagebox.showinfo(
                    "Log retention",
                    f"Removed {removed} log file(s) (freed {freed_mb:.1f} MB).\nErrors: {errs}",
                )
        except Exception:
            # Never crash the GUI for a cleanup failure.
            if show_popup:
                try:
                    messagebox.showwarning("Log retention", "Log cleanup failed (see gui.log).")
                except Exception:
                    pass
            return

    # ------------------------------------------------------------------
    # Remote benchmark helpers (Benchmark tab)
    # ------------------------------------------------------------------

    def _remote_host_configs(self):
        return list(self._remote_service.host_configs(getattr(self, "remote_hosts", []) or []))

    def _remote_get_selected_host(self) -> HostConfig | None:
        sel = ""
        try:
            sel = self.var_remote_host_id.get()
        except Exception:
            sel = ""
        return self._remote_service.get_selected_host(getattr(self, "remote_hosts", []) or [], sel)

    def _remote_hosts_values_for_combo(self):
        # Combobox values are "<id> — <label>"
        vals = []
        for h in self._remote_host_configs():
            vals.append(f"{h.id} — {h.label}")
        return vals

    def _remote_on_host_combo_selected(self, event=None):
        # Combobox sets the full string, we want to store id only.
        try:
            v = event.widget.get()
        except Exception:
            return
        if "—" in v:
            host_id = v.split("—", 1)[0].strip()
        else:
            host_id = v.strip()
        try:
            self.var_remote_host_id.set(host_id)
        except Exception:
            pass
        self._persist_settings()

    def _popup_text(self, title: str, body: str, *, width: int = 70, height: int = 8):
        """Show a scrollable text popup for debug/diagnostic output.

        Note: this is primarily used for the SSH connection test output.
        It must remain *readable* (the previous smaller HiDPI-compensated
        font was too tiny on some setups).
        """

        dlg = tk.Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill="both", expand=True)

        # Use a monospaced font (log-like output) and keep it comfortably
        # readable across platforms. v31 was the sweet spot.
        bfont = tkfont.nametofont("TkFixedFont").copy()
        try:
            base_size = int(bfont.cget("size"))
        except Exception:
            base_size = 10
        bfont.configure(size=max(base_size, 10), weight="normal")

        txt = scrolledtext.ScrolledText(frm, height=height, width=width, wrap="word", font=bfont)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", (body or "").strip() + "\n")
        txt.configure(state="disabled")

        btn = ttk.Button(frm, text="OK", command=dlg.destroy)
        btn.pack(anchor="e", pady=(10, 0))

        dlg.update_idletasks()
        dlg.minsize(dlg.winfo_reqwidth(), dlg.winfo_reqheight())

    def _hailo_gui_record_diagnostics(self, entry: dict) -> None:
        """Remember recent Hailo build diagnostics for quick GUI inspection."""

        if not isinstance(entry, dict) or not entry:
            return
        hist = list(getattr(self, "_hailo_gui_diag_history", []) or [])
        hist.append(dict(entry))
        if len(hist) > 80:
            hist = hist[-80:]
        self._hailo_gui_diag_history = hist

    def _hailo_gui_show_last_diagnostics(self) -> None:
        hist = list(getattr(self, "_hailo_gui_diag_history", []) or [])
        if not hist:
            messagebox.showinfo(
                "Hailo diagnostics",
                "No Hailo HEF diagnostics have been recorded in this GUI session yet.\n\n"
                "Run a HEF build first or open a hailo_hef_build_result.json from disk.",
            )
            return

        recent = hist[-12:]
        body_parts = []
        hidden = max(0, len(hist) - len(recent))
        if hidden:
            body_parts.append(f"Showing the last {len(recent)} Hailo build entries ({hidden} older entries hidden).\n")
        for idx, entry in enumerate(reversed(recent), start=1):
            if body_parts:
                body_parts.append("\n" + "=" * 96 + "\n")
            body_parts.append(f"[{idx}] {entry.get('label') or 'Hailo build'}\n")
            body_parts.append(format_hailo_diagnostics_text(entry))

        self._popup_text("Hailo diagnostics (recent builds)", "".join(body_parts), width=120, height=34)

    def _hailo_gui_open_result_json(self) -> None:
        initialdir = None
        try:
            mp = str(getattr(self, "model_path", "") or "").strip()
            if mp:
                initialdir = str(Path(mp).expanduser().resolve().parent)
        except Exception:
            initialdir = None

        path = filedialog.askopenfilename(
            title="Open hailo_hef_build_result.json",
            filetypes=[("Hailo result JSON", "hailo_hef_build_result.json"), ("JSON", "*.json"), ("All files", "*")],
            initialdir=initialdir,
        )
        if not path:
            return

        try:
            entry = load_hailo_result_json(path)
        except Exception as e:
            messagebox.showerror("Hailo diagnostics", f"Could not read result JSON:\n{e}")
            return

        self._hailo_gui_record_diagnostics(entry)
        self._popup_text(
            f"Hailo diagnostics — {Path(path).parent.name}",
            format_hailo_diagnostics_text(entry),
            width=120,
            height=34,
        )

    def _remote_test_connection(self):
        host = self._remote_get_selected_host()
        if host is None:
            messagebox.showwarning("Remote benchmark", "Please select a remote host first.")
            return
        ok, msg = self._remote_service.test_connection(host, timeout_s=10)

        # Always show stdout/stderr in the popup *and* log it. This makes SSH
        # debugging much easier (wrong key, host key prompt, config issues, ...).
        try:
            logger.info("[remote][test] ok=%s\n%s", ok, msg)
        except Exception:
            pass

        # Avoid huge messageboxes.
        msg_show = msg
        if len(msg_show) > 8000:
            msg_show = msg_show[:8000] + "\n... (truncated)"

        title = "Connection OK" if ok else "Connection FAILED"
        self._popup_text(title, msg_show)

    def _remote_open_hosts_dialog(self, refresh_callback=None):
        """Simple host manager (no secrets)."""
        win = tk.Toplevel(self.root)
        win.title("Remote hosts")
        win.geometry("700x360")
        win.transient(self.root)

        # Some window managers may open the dialog behind the main window.
        try:
            win.lift()
            win.focus_force()
        except Exception:
            pass

        # Left: list
        left = ttk.Frame(win)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=8, pady=8)
        right = ttk.Frame(win)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        lst = tk.Listbox(left, width=30, height=12)
        lst.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        def _reload_list(select_id: str | None = None):
            lst.delete(0, tk.END)
            for h in self._remote_host_configs():
                lst.insert(tk.END, f"{h.id} — {h.label}")
            if select_id:
                for i in range(lst.size()):
                    if lst.get(i).startswith(select_id + " ") or lst.get(i).startswith(select_id + "—") or lst.get(i).startswith(select_id + " —"):
                        lst.selection_set(i)
                        break

        # Right: fields
        fields = {}
        for row, (key, label) in enumerate(
            [
                ("id", "ID"),
                ("label", "Label"),
                ("user", "User"),
                ("host", "Host"),
                ("port", "Port"),
                ("remote_base_dir", "Remote base dir"),
                ("ssh_extra_args", "SSH extra args"),
            ]
        ):
            ttk.Label(right, text=label + ":").grid(row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="")
            ent = ttk.Entry(right, textvariable=var)
            ent.grid(row=row, column=1, sticky="ew", pady=2)
            fields[key] = var
        right.columnconfigure(1, weight=1)

        def _get_selected_id() -> str | None:
            sel = lst.curselection()
            if not sel:
                return None
            txt = lst.get(sel[0])
            return txt.split("—", 1)[0].strip() if "—" in txt else txt.strip()

        def _load_selected(event=None):
            hid = _get_selected_id()
            if not hid:
                return
            for h in getattr(self, "remote_hosts", []) or []:
                if str(h.get("id") or "") == hid:
                    for k, v in fields.items():
                        v.set(str(h.get(k) or ""))
                    return

        lst.bind("<<ListboxSelect>>", _load_selected)

        btns = ttk.Frame(left)
        btns.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        def _new_host():
            for v in fields.values():
                v.set("")
            fields["port"].set("22")
            fields["remote_base_dir"].set("~/splitpoint_runs")
            fields["ssh_extra_args"].set("")

        def _save_host():
            hid = fields["id"].get().strip() or fields["label"].get().strip()
            if not hid:
                messagebox.showwarning("Remote hosts", "ID or Label is required")
                return
            entry = {
                "id": hid,
                "label": fields["label"].get().strip() or hid,
                "user": fields["user"].get().strip(),
                "host": fields["host"].get().strip(),
                "port": int(fields["port"].get().strip() or 22),
                "remote_base_dir": fields["remote_base_dir"].get().strip() or "~/splitpoint_runs",
                "ssh_extra_args": fields["ssh_extra_args"].get().strip(),
            }
            if not entry["host"]:
                messagebox.showwarning("Remote hosts", "Host is required")
                return
            hosts = [h for h in (getattr(self, "remote_hosts", []) or []) if isinstance(h, dict) and str(h.get("id") or "") != hid]
            hosts.append(entry)
            self.remote_hosts = hosts
            self._persist_settings()
            _reload_list(select_id=hid)
            if refresh_callback:
                try:
                    refresh_callback()
                except Exception:
                    pass

        def _delete_host():
            hid = _get_selected_id()
            if not hid:
                return
            if not messagebox.askyesno("Remote hosts", f"Delete host '{hid}'?"):
                return
            self.remote_hosts = [h for h in (getattr(self, "remote_hosts", []) or []) if isinstance(h, dict) and str(h.get("id") or "") != hid]
            self._persist_settings()
            _reload_list()
            if refresh_callback:
                try:
                    refresh_callback()
                except Exception:
                    pass

        ttk.Button(btns, text="New", command=_new_host).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Save", command=_save_host).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Delete", command=_delete_host).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=2)

        _reload_list(select_id=self.var_remote_host_id.get() if hasattr(self, "var_remote_host_id") else None)
        _load_selected()

    def _remote_selected_suite_dir(self) -> Path | None:
        bench_json = self.var_remote_benchmark_set.get().strip() if hasattr(self, "var_remote_benchmark_set") else ""
        if not bench_json:
            return None
        p = Path(bench_json).expanduser()
        if p.is_dir():
            return p
        if p.suffix.lower() == ".json":
            return p.parent
        return p.parent if p.name else p

    def _remote_selected_benchmark_json(self) -> Path | None:
        bench_json = self.var_remote_benchmark_set.get().strip() if hasattr(self, "var_remote_benchmark_set") else ""
        if not bench_json:
            return None
        p = Path(bench_json).expanduser()
        if p.is_dir():
            cand = p / "benchmark_set.json"
            if cand.exists():
                return cand
            jsons = sorted([q for q in p.glob("*.json") if q.is_file()])
            return jsons[0] if jsons else None
        if p.suffix.lower() == ".json":
            return p
        return None

    @staticmethod
    def _parse_remote_int(value, *, default: int, label: str, minimum: int = 0) -> int:
        raw = str(value).strip()
        if raw == "":
            return default
        try:
            number = int(raw)
        except Exception as e:
            raise ValueError(f"{label} must be an integer.") from e
        if number < minimum:
            return minimum
        return number

    def _parse_remote_outer_timeout(self) -> int | None:
        raw = self.var_remote_timeout.get().strip() if hasattr(self, "var_remote_timeout") else "7200"
        if raw == "":
            return 7200
        try:
            number = int(raw)
        except Exception as e:
            raise ValueError("Remote outer timeout must be an integer number of seconds (0 = off).") from e
        if number <= 0:
            return None
        return number

    def _refresh_selected_suite_harness(self):
        suite_dir = self._remote_selected_suite_dir()
        if suite_dir is None:
            messagebox.showwarning("Refresh suite harness", "Please select a benchmark_set.json first.")
            return
        if not suite_dir.exists():
            messagebox.showerror("Refresh suite harness", f"Suite directory not found: {suite_dir}")
            return

        bench_json = self._remote_selected_benchmark_json()
        messages: list[str] = []
        try:
            stats = self._remote_service.refresh_suite_harness(
                suite_dir,
                benchmark_set_json=bench_json,
                log=lambda line: messages.append(str(line)),
            )
        except Exception as e:
            messagebox.showerror("Refresh suite harness", f"Could not refresh suite harness:\n{e}")
            return

        summary = [
            f"Suite: {suite_dir}",
            f"Benchmark JSON: {stats.get('bench_json_name')}",
            f"Cases scanned: {stats.get('case_count')}",
            f"benchmark_suite.py updated: {'yes' if stats.get('suite_script_updated') else 'no'}",
            f"splitpoint_runners files updated: {stats.get('runner_lib_files_updated', 0)}",
            f"Case runner files updated: {stats.get('case_runner_files_updated', 0)}",
            f"Case folders changed: {stats.get('case_runner_cases_updated', 0)}",
            "",
            "The cached suite bundle will rebuild automatically on the next remote run if files changed.",
        ]
        if messages:
            summary += ["", "Details:"] + messages
        messagebox.showinfo("Refresh suite harness", "\n".join(summary))

    def _remote_rebuild_suite_bundle(self):
        suite_dir = self._remote_selected_suite_dir()
        if suite_dir is None:
            messagebox.showwarning("Rebuild bundle", "Please select a benchmark_set.json first.")
            return
        if not suite_dir.exists():
            messagebox.showerror("Rebuild bundle", f"Suite directory not found: {suite_dir}")
            return

        dist_dir = suite_dir / "dist"
        targets = [
            dist_dir / "suite_bundle.tar.gz",
            dist_dir / "suite_bundle.tar.gz.manifest.json",
            dist_dir / "suite_bundle.tar.gz.tmp",
            dist_dir / "suite_bundle.tar.gz.manifest.json.tmp",
        ]
        removed: list[str] = []
        for target in targets:
            try:
                if target.exists():
                    target.unlink()
                    removed.append(target.name)
            except Exception as e:
                messagebox.showerror("Rebuild bundle", f"Could not remove {target}:\n{e}")
                return

        if removed:
            messagebox.showinfo(
                "Rebuild bundle",
                "Removed cached bundle artifacts from:\n"
                f"{dist_dir}\n\n"
                "Next remote benchmark run will rebuild the suite bundle.\n\n"
                f"Removed: {', '.join(removed)}",
            )
        else:
            messagebox.showinfo(
                "Rebuild bundle",
                "No cached suite bundle was found for the selected suite.\n\n"
                f"Checked: {dist_dir}",
            )

    def _remote_run_benchmark(self):
        if bool(getattr(self, "_remote_benchmark_active", False)):
            messagebox.showinfo(
                "Remote benchmark already running",
                "A remote benchmark is already running in the background.\n\n"
                "You can keep one benchmark-set generation running in parallel, but only one remote benchmark at a time.",
            )
            return

        host = self._remote_get_selected_host()
        if host is None:
            messagebox.showwarning("Remote benchmark", "Please select a remote host first.")
            return

        bench_json = self.var_remote_benchmark_set.get().strip() if hasattr(self, "var_remote_benchmark_set") else ""
        if not bench_json:
            messagebox.showwarning("Remote benchmark", "Please select a benchmark_set.json first.")
            return
        bench_path = Path(bench_json).expanduser()
        if bench_path.is_dir():
            bench_path = bench_path / "benchmark_set.json"
        if not bench_path.exists():
            messagebox.showerror("Remote benchmark", f"benchmark_set.json not found: {bench_path}")
            return

        run_id = time.strftime("%Y%m%d_%H%M%S")
        job_id = f"remote-run-{run_id}"
        cancel_event = threading.Event()

        try:
            warmup = self._parse_remote_int(
                self.var_remote_warmup.get() if hasattr(self, "var_remote_warmup") else "10",
                default=10,
                label="Remote warmup",
                minimum=0,
            )
            iters = self._parse_remote_int(
                self.var_remote_iters.get() if hasattr(self, "var_remote_iters") else "100",
                default=100,
                label="Remote runs",
                minimum=1,
            )
            repeats = self._parse_remote_int(
                self.var_remote_repeats.get() if hasattr(self, "var_remote_repeats") else "1",
                default=1,
                label="Remote repeats",
                minimum=1,
            )
            throughput_frames = self._parse_remote_int(
                self.var_remote_throughput_frames.get() if hasattr(self, "var_remote_throughput_frames") else "24",
                default=24,
                label="Streaming frames",
                minimum=0,
            )
            throughput_warmup_frames = self._parse_remote_int(
                self.var_remote_throughput_warmup_frames.get() if hasattr(self, "var_remote_throughput_warmup_frames") else "6",
                default=6,
                label="Streaming warmup frames",
                minimum=0,
            )
            throughput_queue_depth = self._parse_remote_int(
                self.var_remote_throughput_queue_depth.get() if hasattr(self, "var_remote_throughput_queue_depth") else "2",
                default=2,
                label="Streaming queue depth",
                minimum=1,
            )
            timeout_s = self._parse_remote_outer_timeout()
        except Exception as exc:
            messagebox.showerror("Remote benchmark", f"Invalid benchmark settings:\n\n{exc}")
            return

        local_working_dir = Path(getattr(self, "default_output_dir", "."))
        suite_name = bench_path.parent.name or bench_path.stem or f"remote_run_{run_id}"
        self._jobs_register(
            job_id=job_id,
            kind="remote_run",
            type_label="Remote run",
            title=f"Remote benchmark — {suite_name}",
            name=str(suite_name),
            output_dir=str(local_working_dir),
            initial_status="Starting…",
            initial_lines=[
                f"Remote host: {host.user_host}",
                f"Suite: {bench_path.parent}",
                "Runs in the background. Close this window any time and reopen it from the Jobs tab.",
            ],
            progress_maximum=1.0,
            cancel_callback=lambda: cancel_event.set(),
            can_cancel=True,
        )

        args = RemoteBenchmarkArgs(
            provider=self.var_remote_provider.get() if hasattr(self, "var_remote_provider") else "auto",
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            timeout_s=timeout_s,
            throughput_frames=throughput_frames,
            throughput_warmup_frames=throughput_warmup_frames,
            throughput_queue_depth=throughput_queue_depth,
            add_args=self.var_remote_add_args.get() if hasattr(self, "var_remote_add_args") else "",
            remote_venv=self.var_remote_venv.get() if hasattr(self, "var_remote_venv") else "",
            transfer_mode=self.var_remote_transfer_mode.get() if hasattr(self, "var_remote_transfer_mode") else "bundle",
            reuse_bundle=bool(self.var_remote_reuse_bundle.get()) if hasattr(self, "var_remote_reuse_bundle") else True,
        )

        callbacks = RemoteBenchmarkCallbacks(
            log=lambda s: self.root.after(0, lambda _s=s: self._jobs_append_log(job_id, _s)),
            progress=lambda p, lbl: self.root.after(
                0,
                lambda _p=p, _lbl=lbl: self._jobs_set_progress(
                    job_id,
                    value=max(0.0, min(1.0, float(_p))),
                    label=str(_lbl or ""),
                    display=f"{int(round(max(0.0, min(1.0, float(_p))) * 100.0))}%",
                    progress_maximum=1.0,
                ),
            ),
            finish=lambda status: self.root.after(
                0,
                lambda _status=status: self._jobs_set_progress(
                    job_id,
                    value=(self._background_jobs.get(job_id).progress_value if self._background_jobs.get(job_id) is not None else 0.0),
                    label=str(_status or ""),
                ),
            ),
            result=lambda kind, out: self.root.after(0, lambda _kind=kind, _out=out: self._finalize_remote_benchmark_job(job_id, _kind, _out)),
        )

        self._set_background_job_active("remote_run", True)
        try:
            self._remote_benchmark_thread = self._remote_controller.start_async(
                host=host,
                benchmark_set_json=bench_path,
                local_working_dir=local_working_dir,
                run_id=run_id,
                args=args,
                cancel_event=cancel_event,
                callbacks=callbacks,
            )
        except Exception:
            self._set_background_job_active("remote_run", False)
            self._jobs_finish(job_id, status="error", message="Failed to start remote benchmark thread")
            raise

    def _finalize_remote_benchmark_job(self, job_id: str, final_kind: str, out: Dict[str, Any]) -> None:
        self._set_background_job_active("remote_run", False)
        local_run_dir = str(out.get("local_run_dir") or "").strip()
        log_path = ""
        if local_run_dir:
            runner_log = Path(local_run_dir) / "logs" / "runner.log"
            stdout_log = Path(local_run_dir) / "logs" / "stdout.txt"
            if runner_log.exists():
                log_path = str(runner_log)
            elif stdout_log.exists():
                log_path = str(stdout_log)
        status_map = {
            "ok": "success",
            "partial": "warning",
            "cancelled": "cancelled",
            "failed": "error",
            "error": "error",
        }
        self._jobs_update_paths(job_id, output_dir=(local_run_dir or None), log_path=(log_path or None))
        self._jobs_finish(
            job_id,
            status=status_map.get(str(final_kind or "").strip().lower(), "error"),
            message=str(out.get("error") or out.get("local_run_dir") or ""),
            output_dir=(local_run_dir or None),
            log_path=(log_path or None),
        )
        self._handle_remote_benchmark_result(final_kind, out)

    def _handle_remote_benchmark_result(self, final_kind: str, out: Dict[str, Any]) -> None:
        if final_kind == "ok":
            messagebox.showinfo("Remote benchmark", f"Done. Results saved to:\n\n{out.get('local_run_dir')}")
        elif final_kind == "partial":
            messagebox.showwarning("Remote benchmark", f"Partial run. Results saved to:\n\n{out.get('local_run_dir')}\n\n{out.get('error')}")
        elif final_kind == "cancelled":
            messagebox.showwarning("Remote benchmark", f"Run cancelled:\n\n{out.get('error')}")
        elif final_kind == "failed":
            messagebox.showerror("Remote benchmark", f"Run failed:\n\n{out.get('error')}")
        elif final_kind == "error":
            messagebox.showerror("Remote benchmark", f"Run errored:\n\n{out.get('error')}")


    def _init_central_notebook(self) -> None:
        logger.info("Initializing central notebook UI")
        root_children = list(self.winfo_children())

        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.panel_frames: Dict[str, ttk.Frame] = {
            "analysis": panel_analysis.build_panel(self.main_notebook, app=self),
            "export": panel_split_export.build_panel(self.main_notebook, app=self),
            "validate": panel_validation.build_panel(self.main_notebook, app=self),
            "bench_analysis": panel_benchmark_analysis.build_panel(self.main_notebook, app=self),
            "jobs": panel_jobs.build_panel(self.main_notebook, app=self),
            "hardware": panel_hardware.build_panel(self.main_notebook, app=self),
            "logs": panel_logs.build_panel(self.main_notebook, app=self),
        }

        for key, label in self.TAB_LABELS:
            self.main_notebook.add(self.panel_frames[key], text=label)

        panel_analysis.hide_legacy_widgets(root_children, self)

        self.jobs_status_bar = ttk.Frame(self)
        self.jobs_status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 8))
        ttk.Label(self.jobs_status_bar, text="Background jobs:").pack(side=tk.LEFT, padx=(0, 8))
        self.job_bar_generate = tk.Label(self.jobs_status_bar, text="Generate: idle", padx=8, pady=2, bg=self._jobs_status_color("idle"), fg="white")
        self.job_bar_generate.pack(side=tk.LEFT, padx=(0, 8))
        self.job_bar_remote = tk.Label(self.jobs_status_bar, text="Remote run: idle", padx=8, pady=2, bg=self._jobs_status_color("idle"), fg="white")
        self.job_bar_remote.pack(side=tk.LEFT)

        # The notebook panels rebind several legacy button references (most
        # importantly the Benchmark button). Refresh the UI state once after all
        # panels are mounted so the freshly bound widgets receive the correct
        # enabled/disabled state immediately.
        try:
            self._set_ui_state(self._infer_ui_state())
        except Exception:
            logger.exception("Failed to refresh notebook action-button state")

        try:
            self._jobs_refresh_views()
        except Exception:
            logger.exception("Failed to refresh Jobs UI state")

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
            try:
                refresh = getattr(self, "_benchmark_refresh_hailo_compile_outlook", None)
                if callable(refresh):
                    refresh()
            except Exception:
                logger.debug("Benchmark Hailo compile outlook refresh failed in notebook shell", exc_info=True)

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
