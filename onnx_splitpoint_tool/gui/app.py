"""GUI application entrypoint and incremental notebook shell."""

from __future__ import annotations

import os
import posixpath
import csv
import json
import re
import logging
import shutil
import shlex
import subprocess
import sys
import time
import threading
import signal
import uuid
import yaml
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field, replace
from datetime import datetime
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from tkinter import scrolledtext
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from .. import __release__ as TOOL_VERSION
from .. import api as asc
from ..settings import SettingsStore
from ..remote import HostConfig, SSHTransport
from ..benchmark.remote_run import (
    RemoteBenchmarkArgs,
    preflight_remote_energy_dispatch,
    run_remote_benchmark,
)
from ..energy.comparison import resolve_energy_comparison
from ..benchmark.suite_refresh import refresh_suite_harness
from ..benchmark.services import RemoteBenchmarkCallbacks, RemoteBenchmarkController, RemoteBenchmarkService
from ..workdir import ensure_workdir
from ..paths import splitpoint_provisioning_logs_dir
from ..workflow.contracts import WorkflowOptions
from ..workflow.hardware_matrix import ensure_hardware_setups_file, default_hardware_setups_file, canon_accelerator
from ..workflow.runner import EvaluationWorkflowRunner
from ..workflow.run_control import WorkflowRunCancelledError
from ..workflow.status_reporting import blocking_reasons_for_display, energy_axis_description
from ..workflow.run_discovery import discover_evaluation_run, is_evaluation_run_dir
from ..filesystem_admission import (
    require_write_target,
)
from ..gui_app import SplitPointAnalyserGUI as LegacySplitPointAnalyserGUI
from ..gui_app import _setup_gui_logging
from .hailo_diagnostics import format_hailo_diagnostics_text, load_hailo_result_json
from .dashboard_projection import dashboard_summary_line
from .hailo_parse_budget import normalize_persisted_hailo_max_checks
from .panels import panel_analysis, panel_split_export, panel_benchmark_analysis, panel_evaluation_workflow, panel_hardware, panel_jobs, panel_logs, panel_validation
from .profile_campaign import (
    ProfileCampaignOptions,
    run_profile_campaign,
    snapshot_preparation_runtime,
    snapshot_remote_defaults,
)
from ..benchmark.evaluation_profiles import load_evaluation_profile, load_export_metadata_for_model
from ..execution_plan import build_effective_execution_plan
from ..benchmark.model_preparation import (
    normalize_model_preparation_mode,
    prepare_model_for_benchmark,
    preparation_result_is_selected_model,
)
from .widgets.text_progress_dialog import TextProgressDialog
from .profile_editor import open_evaluation_profile_editor

__version__ = TOOL_VERSION
logger = logging.getLogger(__name__)


def _evaluation_energy_not_started_messages(
    payload: Mapping[str, Any] | None,
) -> list[str]:
    """Return explicit Energy preflight diagnostics for the result surface."""

    messages: list[str] = []
    value = dict(payload or {}) if isinstance(payload, Mapping) else {}
    for stage in list(value.get("stage_results") or []):
        if not isinstance(stage, Mapping):
            continue
        details = (
            stage.get("details")
            if isinstance(stage.get("details"), Mapping) else {}
        )
        for container in (stage, details):
            message = str(
                container.get("energy_not_started_reason") or ""
            ).strip()
            if message and message not in messages:
                messages.append(message)
            for row in list(container.get("model_preflight_rows") or []):
                if not isinstance(row, Mapping):
                    continue
                row_message = str(
                    row.get("energy_not_started_reason") or ""
                ).strip()
                if row_message and row_message not in messages:
                    messages.append(row_message)
        stage_message = str(stage.get("message") or "").strip()
        if (
            stage_message.startswith("Energy requested, not started:")
            and stage_message not in messages
        ):
            messages.append(stage_message)
    return messages


def _persist_as_generic_gui_var(name: str) -> bool:
    """Return whether a Tk variable belongs in generic GUI settings.

    Hardware setup cards and platform-power status cards are backed by the
    central hardware registry.  Replaying a second GUI-settings copy after a
    lazy panel build could otherwise show or write stale setup configuration.
    """

    key = str(name or "")
    if key.startswith("var_hwsetup_"):
        return False
    if key.startswith("var_platform_power_"):
        return False
    return True


def _new_evaluation_workflow_job_id() -> str:
    """Return a collision-resistant GUI row id for one workflow launch."""

    return (
        f"evaluation-workflow-{time.strftime('%Y%m%d_%H%M%S')}-"
        f"{uuid.uuid4().hex}"
    )


def _evaluation_workflow_subjob_row_id(
    workflow_scope: str, workflow_job_id: str
) -> str:
    """Namespace a formal workflow job id under its GUI parent launch."""

    def _token(value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or ""))
        return cleaned.strip("-._") or fallback

    scope = _token(workflow_scope, "workflow")
    child = _token(workflow_job_id, "job")
    return f"eval-subjob-{scope}--{child}"


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
    cancel_callback: Optional[Callable[[], Optional[bool]]] = None
    can_cancel: bool = False
    geometry: str = "760x460"
    parent_job_id: str = ""
    workflow_scope: str = ""
    workflow_job_id: str = ""
    workflow_parent_job_id: str = ""
    workflow_queue_order: int = 0
    workflow_actual_started: bool = True
    workflow_duration_s: Optional[float] = None
    monitor: Optional[TextProgressDialog] = None
    dismissed: bool = False
    worker_thread: Optional[threading.Thread] = None



class SplitPointAnalyserGUI(LegacySplitPointAnalyserGUI):
    """Notebook-based shell around the legacy GUI.

    The heavy UI/logic still lives in :mod:`onnx_splitpoint_tool.gui_app` and is
    migrated incrementally into ``gui.panels`` modules.
    """

    TAB_LABELS = (
        ("analysis", "Analyse"),
        ("export", "Advanced Export"),
        ("evaluation_workflow", "Evaluation Workflow"),
        ("validate", "Benchmark"),
        ("bench_analysis", "Benchmark-Analyse"),
        ("jobs", "Jobs"),
        ("hardware", "Tool Config"),
        ("logs", "Logs"),
    )

    def __init__(self):
        super().__init__()

        # The legacy GUI uses `self` as the Tk root. Some newer modules expect
        # a `.root` attribute as well, so keep an alias for compatibility.
        self.root = self
        # v58ad: install expensive Tk visibility tracing only on demand.
        # Binding <Map>/<Expose> on the root can fire for hundreds of child
        # widgets and the debug file writes themselves caused minute-long
        # blank-window startups in v58ac.
        try:
            if str(os.environ.get("ONNX_SPLITPOINT_STARTUP_VISIBILITY_TRACE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                self._install_startup_visibility_tracing()
            else:
                logger.info("Startup visibility trace disabled (set ONNX_SPLITPOINT_STARTUP_VISIBILITY_TRACE=1 to enable)")
        except Exception:
            pass
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
        self.var_remote_hardware_setup_id = tk.StringVar(value="")
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
        self.var_remote_measure_energy = tk.BooleanVar(value=False)
        self.var_remote_energy_runs = tk.StringVar(value="")  # empty => Tool Config energy default

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
        self._persisted_settings_cache = persisted
        self._apply_persistent_settings(persisted)

        # Ensure the working-directory structure exists under the selected
        # output folder. This makes remote benchmarking + artifact management
        # predictable.
        try:
            if getattr(self, "default_output_dir", None):
                ensure_workdir(Path(self.default_output_dir))
        except Exception:
            pass

        # Suppress expensive trace-triggered status refreshes while the panels are
        # being constructed and persisted settings are replayed.  Without this,
        # Hailo/DeepX status callbacks can run before the first window paint and
        # make startup look blocked even though the work is technically scheduled
        # via Tk after().
        self._startup_status_refresh_suppressed = True
        self._init_central_notebook()

        # Re-apply now that all panel Tk variables exist
        self._apply_persistent_settings(persisted)
        # v58aa: do not use after_idle for startup-unblock state.  On busy
        # Tk queues after_idle may not fire for a long time although the GUI is
        # already usable.  Use a short timer instead so deferred status refreshes
        # are unblocked deterministically after the first event-loop tick.
        try:
            self.after(250, lambda: setattr(self, "_startup_status_refresh_suppressed", False))
        except Exception:
            self._startup_status_refresh_suppressed = False
        self._wire_model_type_state()
        self._apply_model_type_visibility()

        # v58aa: log-retention was still a likely startup stall source on
        # machines with large result/log trees.  Keep startup purely interactive
        # by default; users can run Clean now from the Logs tab or opt in.
        try:
            if str(os.environ.get("ONNX_SPLITPOINT_STARTUP_LOG_RETENTION", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                import threading as _startup_threading
                def _retention_worker() -> None:
                    try:
                        self._apply_log_retention(show_popup=False)
                    except Exception:
                        logger.debug("Deferred log-retention failed", exc_info=True)
                self.after(15000, lambda: _startup_threading.Thread(target=_retention_worker, name="splitpoint-log-retention", daemon=True).start())
                logger.info("Startup log retention enabled; scheduled after first GUI paint")
            else:
                logger.info("Startup log retention disabled (use Logs → Clean now or set ONNX_SPLITPOINT_STARTUP_LOG_RETENTION=1)")
        except Exception:
            pass
        try:
            self.after(100, lambda: logger.info("GUI ready event-loop tick reached; startup probes are deferred"))
        except Exception:
            pass
        # Do not use after_idle as a readiness signal.  It can be delayed by
        # recurring Tk work and made startup look slow even after the window is
        # responsive.  Keep an opt-in diagnostic for future profiling only.
        try:
            if str(os.environ.get("ONNX_SPLITPOINT_STARTUP_LOG_IDLE_MARKER", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                self.after_idle(lambda: logger.info("GUI idle queue drained"))
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
        self._install_evaluation_shutdown_signal_broker()

        # v58s: Hailo/DFC import probes are expensive and were able to delay
        # startup despite being nominally "background".  Default to no startup
        # auto-probe; users can trigger Status/Refresh in Tool Config or opt in.
        try:
            if str(os.environ.get("ONNX_SPLITPOINT_STARTUP_HAILO_PROBE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                probe_delay_ms = int(str(os.environ.get("ONNX_SPLITPOINT_STARTUP_HAILO_PROBE_DELAY_MS", "10000") or "10000").strip() or "10000")
                logger.info("Startup Hailo DFC auto-probe enabled; scheduling after %sms", probe_delay_ms)
                self.after(max(0, probe_delay_ms), getattr(self, "_hailo_refresh_status", lambda: None))
            else:
                logger.info("Startup Hailo DFC auto-probe disabled (set ONNX_SPLITPOINT_STARTUP_HAILO_PROBE=1 to enable)")
        except Exception:
            pass


    def _startup_trace_note(self, message: str) -> None:
        """Write startup visibility milestones to both gui.log and an optional trace file."""
        try:
            logger.info("[startup-visibility] %s", message)
        except Exception:
            pass
        try:
            trace_file = os.environ.get("ONNX_SPLITPOINT_STARTUP_TRACE_FILE")
            if trace_file:
                import time as _time
                shell_ts = os.environ.get("ONNX_SPLITPOINT_STARTUP_SHELL_TS")
                prefix = ""
                if shell_ts:
                    try:
                        prefix = f"[wall +{max(0.0, _time.time() - float(shell_ts)):.3f}s] "
                    except Exception:
                        prefix = ""
                with open(trace_file, "a", encoding="utf-8") as fh:
                    fh.write(prefix + message + "\n")
        except Exception:
            pass

    def _install_startup_visibility_tracing(self) -> None:
        """Trace when the Tk root is actually mapped/visible.

        v58ac diagnostic: Previous logs showed the event loop was active quickly, while the
        user still perceived the window as appearing much later.  These bindings
        tell us whether the window manager maps the root late, and the explicit
        deiconify/lift/update call nudges Tk to paint the first frame as early as
        possible.
        """
        try:
            def _note(event_name: str):
                def _cb(_event=None):
                    self._startup_trace_note(event_name)
                return _cb
            self.bind("<Map>", _note("Tk root <Map> event"), add="+")
            self.bind("<Visibility>", _note("Tk root <Visibility> event"), add="+")
            self.bind("<Expose>", _note("Tk root <Expose> event"), add="+")
        except Exception:
            pass
        try:
            self.after(10, lambda: self._startup_trace_note("Tk after(10) tick"))
        except Exception:
            pass

    def _force_initial_window_paint(self) -> None:
        """Best-effort first-paint nudge before entering mainloop.

        v58ac proved that calling ``update()`` and tracing every Map/Expose
        event can itself create a long blank-window stall.  The safe default is
        therefore only ``update_idletasks()`` after the notebook has been fully
        constructed.  This lets Tk compute geometry and hand the finished window
        to the window manager without recursively draining the event queue.

        Modes via ONNX_SPLITPOINT_INITIAL_PAINT_MODE:
          - idletasks (default): deiconify/lift/update_idletasks only
          - off: no explicit paint nudge
          - update: full update(), debug only, can be slow
        """
        try:
            mode = str(os.environ.get("ONNX_SPLITPOINT_INITIAL_PAINT_MODE", "off") or "off").strip().lower()
            if mode in {"0", "false", "no", "off", "disabled"}:
                logger.info("Startup initial paint nudge disabled; entering mainloop directly")
                return
            try:
                self.deiconify()
            except Exception:
                pass
            try:
                self.lift()
            except Exception:
                pass
            try:
                self.update_idletasks()
                logger.info("Startup initial Tk update_idletasks completed; entering mainloop")
            except Exception:
                logger.debug("Startup update_idletasks failed", exc_info=True)
            if mode in {"update", "full", "1", "true", "yes", "on"}:
                logger.warning("Startup full Tk update requested; this is debug-only and may delay first paint")
                try:
                    self.update()
                    logger.info("Startup full Tk update completed; entering mainloop")
                except Exception:
                    logger.debug("Startup full update failed", exc_info=True)
        except Exception:
            try:
                logger.debug("Initial Tk paint/update failed", exc_info=True)
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
        "skipped": "Skipped",
        "partial": "Partial",
        "warn": "Warning",
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
        "skipped": "#9e9e9e",
        "partial": "#ef6c00",
        "warn": "#ef6c00",
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

    def _jobs_parse_iso_datetime(self, value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            # Tk-side datetimes are naive local times; drop timezone only for
            # display-duration arithmetic. The event timestamps are produced on
            # the same host/session as the GUI in normal use.
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except Exception:
            return None

    def _jobs_format_seconds(self, seconds: Any) -> str:
        try:
            secs = max(0, int(float(seconds)))
        except Exception:
            return ""
        if secs < 60:
            return f"{secs}s"
        mins, sec = divmod(secs, 60)
        if mins < 60:
            return f"{mins}m {sec:02d}s"
        hrs, mins = divmod(mins, 60)
        if hrs < 48:
            return f"{hrs}h {mins:02d}m"
        days, hrs = divmod(hrs, 24)
        return f"{days}d {hrs}h"

    def _jobs_elapsed_text(self, record: Optional[BackgroundJobRecord]) -> str:
        if record is None:
            return ""
        try:
            if getattr(record, "workflow_duration_s", None) is not None:
                return self._jobs_format_seconds(getattr(record, "workflow_duration_s"))
            if str(record.kind or "") == "evaluation_workflow_subjob" and not bool(getattr(record, "workflow_actual_started", True)):
                return ""
            end = record.end_time or datetime.now()
            delta = end - record.start_time
            return self._jobs_format_seconds(delta.total_seconds())
        except Exception:
            return ""

    def _jobs_summary_text(self, kind: str, record: Optional[BackgroundJobRecord]) -> str:
        kind_norm = str(kind or "").strip().lower()
        if kind_norm == "generate":
            title = "Generate"
        elif kind_norm == "remote_run":
            title = "Remote run"
        elif kind_norm in {"profile_campaign", "evaluation_workflow"}:
            title = "Workflow"
        elif kind_norm == "validation_assets":
            title = "Validation assets"
        else:
            title = str(kind or "Job")
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
        cancel_callback: Optional[Callable[[], Optional[bool]]] = None,
        can_cancel: bool = False,
        geometry: str = "760x460",
        show_monitor: bool = True,
        record_status: str = "running",
    ) -> BackgroundJobRecord:
        record = BackgroundJobRecord(
            job_id=str(job_id),
            kind=str(kind),
            type_label=str(type_label),
            title=str(title),
            name=str(name),
            output_dir=str(output_dir or ""),
            log_path=str(log_path or ""),
            status=str(record_status or "running").strip().lower(),
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
        value: Optional[float] = None,
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
        if value is not None:
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
            if record.kind == "evaluation_workflow" and str(label).startswith("finalize_artifacts"):
                record.can_cancel = False
                if record.monitor is not None and bool(getattr(record.monitor, "alive", False)):
                    try:
                        record.monitor.btn_cancel.configure(state="disabled")
                    except Exception:
                        pass
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
        if record.kind == "evaluation_workflow":
            # The formal root row records the measurement decision before the
            # writer seals jobs/.  Its GUI mirror completes only on return from
            # that writer, without appending to any run-owned file.
            for child_id, child in list(self._background_jobs.items()):
                if (
                    child.kind == "evaluation_workflow_subjob"
                    and getattr(child, "workflow_scope", "") == str(job_id)
                    and child.type_label == "EvaluationWorkflowJob"
                ):
                    self._jobs_finish(child_id, status=record.status, message=message)
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
        try:
            accepted = record.cancel_callback()
        except Exception as exc:
            logger.exception("Failed to cancel job %s", job_id)
            messagebox.showerror("Cancel job", f"Could not request cancellation:\n\n{exc}")
            return
        if accepted is False:
            record.can_cancel = False
            record.last_message = "Abschlussprüfung läuft; Abbruch ist nicht mehr verfügbar."
            if record.monitor is not None and bool(getattr(record.monitor, "alive", False)):
                try:
                    record.monitor.btn_cancel.configure(state="disabled")
                except Exception:
                    pass
            self._jobs_refresh_views()
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
        self._jobs_refresh_views()


    def _jobs_workflow_status_to_gui(self, status: str) -> str:
        s = str(status or "").strip().lower()
        if s in {"ok", "success", "validated", "measured", "hardware_verified"}:
            return "success"
        if s in {"partial", "warn", "warning"}:
            return "warning"
        if s in {"failed", "error"}:
            return "error"
        if s in {"cancelled", "canceled"}:
            return "cancelled"
        if s in {"skipped", "resume_reused_existing_stage_result"}:
            return "success"
        if s in {"queued"}:
            return "queued"
        return "running" if s == "running" else "warning"

    def _jobs_handle_workflow_job_event(
        self,
        event: Mapping[str, Any],
        *,
        output_dir: str = "",
        log_path: str = "",
        workflow_scope: str = "",
    ) -> None:
        """Mirror formal EvaluationWorkflowRunner subjobs into the Jobs tab.

        The existing GUI still owns the actual background thread.  These rows are
        live status mirrors for the formal workflow job tree written under
        EvaluationRuns/<run>/jobs/*.json.
        """
        try:
            raw_job_id = str(event.get("job_id") or "").strip()
            if not raw_job_id:
                return
            scope = str(
                workflow_scope
                or event.get("workflow_scope")
                or event.get("session_id")
                or event.get("run_id")
                or "workflow"
            ).strip()
            job_id = _evaluation_workflow_subjob_row_id(scope, raw_job_id)
            raw_parent_id = str(event.get("parent_job_id") or "").strip()
            parent_row_id = (
                _evaluation_workflow_subjob_row_id(scope, raw_parent_id)
                if raw_parent_id
                else ""
            )
            raw_type = str(event.get("job_type") or "WorkflowJob")
            model_id = str(event.get("model_id") or "").strip()
            stage = str(event.get("stage") or "").strip()
            title = str(event.get("title") or event.get("job_id") or "Workflow job")
            name = title
            event_run_dir = str(event.get("run_dir") or "").strip()
            if event_run_dir:
                output_dir = event_run_dir
                log_path = str(Path(event_run_dir) / "evaluation_workflow.log")
            status = str(event.get("status") or "queued")
            event_kind = str(event.get("event") or "status")
            measurement_complete = (
                raw_type == "EvaluationWorkflowJob"
                and str(event.get("message") or "").startswith("measurement_phase_complete")
            )
            if measurement_complete:
                # Persistent formal job summaries describe the measurement
                # result.  The live root remains active until run() returns.
                status = "running"
            parent_workflow = self._background_jobs.get(scope)
            if (
                raw_type == "EvaluationWorkflowJob"
                and parent_workflow is not None
                and str(parent_workflow.status) not in {"queued", "running", "cancelling"}
            ):
                # A previously queued event may arrive after the worker's
                # return callback.  It must not resurrect the completed root.
                status = str(parent_workflow.status)
                event_kind = "workflow_returned"
            gui_status = self._jobs_workflow_status_to_gui(status) if event_kind != "planned" else "queued"
            if job_id not in self._background_jobs:
                self._jobs_register(
                    job_id=job_id,
                    kind="evaluation_workflow_subjob",
                    type_label=raw_type,
                    title=f"Evaluation Workflow — {title}",
                    name=name,
                    output_dir=str(output_dir or ""),
                    log_path=str(log_path or ""),
                    initial_status=self._jobs_status_label(gui_status),
                    initial_lines=[f"[{event_kind}] {title}", str(event.get("message") or "")],
                    progress_maximum=1.0,
                    can_cancel=False,
                    geometry="900x420",
                    show_monitor=False,
                    record_status=gui_status,
                )
            rec = self._background_jobs.get(job_id)
            if rec is None:
                return
            rec.kind = "evaluation_workflow_subjob"
            rec.type_label = raw_type
            rec.name = name
            rec.workflow_job_id = raw_job_id
            rec.workflow_scope = scope
            rec.workflow_parent_job_id = raw_parent_id
            rec.parent_job_id = parent_row_id
            try:
                rec.workflow_queue_order = int(event.get("queue_order") or 0)
            except Exception:
                rec.workflow_queue_order = 0
            if output_dir:
                rec.output_dir = str(output_dir)
            if log_path:
                rec.log_path = str(log_path)
            msg = str(event.get("message") or "").strip() or f"{event_kind}: {status}"
            details = dict(event.get("details") or {}) if isinstance(event.get("details"), Mapping) else {}
            detail_progress = str(details.get("subtask") or details.get("remote_status") or "").strip()
            event_time = self._jobs_parse_iso_datetime(event.get("time")) or datetime.now()
            started_at = self._jobs_parse_iso_datetime(event.get("started_at"))
            finished_at = self._jobs_parse_iso_datetime(event.get("finished_at"))
            duration_raw = event.get("duration_s")
            duration_s: Optional[float]
            try:
                duration_s = float(duration_raw) if duration_raw not in (None, "") else None
            except Exception:
                duration_s = None

            if event_kind == "planned":
                rec.status = "queued"
                rec.status_text = self._jobs_status_label("queued")
                rec.progress_value = 0.0
                rec.workflow_actual_started = False
                rec.workflow_duration_s = None
            elif event_kind == "log":
                rec.status = "running"
                rec.status_text = self._jobs_status_label("running")
                rec.progress_value = max(float(rec.progress_value or 0.0), 0.25)
                if not bool(getattr(rec, "workflow_actual_started", True)) or started_at is not None:
                    rec.start_time = started_at or event_time
                rec.workflow_actual_started = True
                rec.workflow_duration_s = None
            elif gui_status in {"success", "warning", "error", "cancelled"}:
                rec.status = gui_status
                rec.status_text = self._jobs_status_label(gui_status)
                if started_at is not None:
                    rec.start_time = started_at
                elif not bool(getattr(rec, "workflow_actual_started", True)):
                    rec.start_time = event_time
                rec.end_time = finished_at or event_time
                rec.workflow_actual_started = True
                if duration_s is not None:
                    rec.workflow_duration_s = max(0.0, duration_s)
                else:
                    try:
                        rec.workflow_duration_s = max(0.0, (rec.end_time - rec.start_time).total_seconds())
                    except Exception:
                        rec.workflow_duration_s = None
                rec.can_cancel = False
                rec.progress_value = 1.0
            else:
                rec.status = gui_status
                rec.status_text = self._jobs_status_label(gui_status)
                if gui_status == "running":
                    if not bool(getattr(rec, "workflow_actual_started", True)) or started_at is not None:
                        rec.start_time = started_at or event_time
                    rec.workflow_actual_started = True
                    rec.workflow_duration_s = None
                    rec.progress_value = 0.25
                else:
                    rec.progress_value = rec.progress_value
            rec.progress_display = detail_progress or str(status)
            rec.last_message = msg
            # Keep subjob logs small; the main workflow monitor has the full log.
            if msg and (not rec.log_lines or rec.log_lines[-1] != msg):
                rec.log_lines.append(msg)
                if len(rec.log_lines) > 100:
                    rec.log_lines = rec.log_lines[-100:]
            self._jobs_refresh_views()
        except Exception:
            logger.debug("Failed to mirror evaluation workflow subjob event", exc_info=True)

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

        records = [self._background_jobs.get(jid) for jid in list(self._background_job_order or [])]
        records = [r for r in records if r is not None and not bool(r.dismissed)]

        def _sort_key(record: BackgroundJobRecord):
            if str(record.kind or "") == "evaluation_workflow_subjob":
                return (0, int(record.workflow_queue_order or 0), record.start_time)
            return (1, -int(record.start_time.timestamp()), record.start_time)

        inserted: set[str] = set()

        def _insert_record(record: BackgroundJobRecord) -> None:
            if record.job_id in inserted:
                return
            parent = str(getattr(record, "parent_job_id", "") or "")
            if parent and parent in self._background_jobs and parent not in inserted:
                parent_rec = self._background_jobs.get(parent)
                if parent_rec is not None and not bool(parent_rec.dismissed):
                    _insert_record(parent_rec)
                else:
                    parent = ""
            if parent and parent not in inserted:
                parent = ""
            name_target = str(record.name or "")
            if str(record.output_dir or "").strip() and str(record.kind or "") != "evaluation_workflow_subjob":
                name_target = f"{name_target} — {record.output_dir}" if name_target else str(record.output_dir)
            last_message = str(record.last_message or "").replace("\n", " ").strip()
            values = (
                str(record.type_label or ""),
                name_target,
                self._jobs_status_label(record.status),
                self._jobs_format_progress(record),
                self._jobs_elapsed_text(record),
                record.start_time.strftime("%Y-%m-%d %H:%M:%S"),
                last_message[:220],
            )
            tag = f"job_{str(record.status or 'running').strip().lower()}"
            tree_label = str(record.title or record.name or record.job_id)
            if str(record.kind or "") == "evaluation_workflow_subjob":
                # Show the human job title in the tree; the stable job id remains
                # available in the detail panel/log and in jobs/job_plan.json.
                tree_label = str(record.name or record.title or record.workflow_job_id or record.job_id)
            try:
                tree.insert(parent, "end", iid=record.job_id, text=tree_label, values=values, tags=(tag,), open=True)
                inserted.add(record.job_id)
            except Exception:
                logger.debug("Failed to insert Jobs row for %s", record.job_id, exc_info=True)

        for record in sorted(records, key=_sort_key):
            _insert_record(record)

        if selected and selected in self._background_jobs and not bool(self._background_jobs[selected].dismissed):
            try:
                tree.selection_set(selected)
                tree.focus(selected)
            except Exception:
                pass
        self._jobs_update_panel_buttons()

    def _jobs_refresh_status_bar(self) -> None:
        for kind, attr in (("generate", "job_bar_generate"), ("remote_run", "job_bar_remote"), ("validation_assets", "job_bar_validation_assets")):
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

        workflow_widget = getattr(self, "job_bar_workflow", None)
        if workflow_widget is not None:
            workflow_records = [
                self._jobs_latest_record("evaluation_workflow"),
                self._jobs_latest_record("profile_campaign"),
            ]
            workflow_records = [r for r in workflow_records if r is not None]
            workflow_record = max(workflow_records, key=lambda r: r.start_time, default=None)
            workflow_kind = str(getattr(workflow_record, "kind", "evaluation_workflow") or "evaluation_workflow") if workflow_record is not None else "evaluation_workflow"
            workflow_status = str(getattr(workflow_record, "status", "idle") or "idle") if workflow_record is not None else "idle"
            workflow_text = self._jobs_summary_text(workflow_kind, workflow_record)
            try:
                workflow_widget.configure(text=workflow_text, bg=self._jobs_status_color(workflow_status), fg="white")
            except Exception:
                logger.debug("Failed to refresh workflow jobs status bar", exc_info=True)

    def _jobs_refresh_views(self) -> None:
        self._jobs_refresh_status_bar()
        self._jobs_refresh_tree()

    def _jobs_expand_all(self) -> None:
        tree = getattr(self, "jobs_tree", None)
        if tree is None:
            return
        try:
            def _walk(parent=""):
                for iid in tree.get_children(parent):
                    tree.item(iid, open=True)
                    _walk(iid)
            _walk("")
        except Exception:
            pass

    def _jobs_collapse_all(self) -> None:
        tree = getattr(self, "jobs_tree", None)
        if tree is None:
            return
        try:
            def _walk(parent=""):
                for iid in tree.get_children(parent):
                    _walk(iid)
                    tree.item(iid, open=False)
            _walk("")
        except Exception:
            pass

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
            # v58s: canonical profile YAML path wins over stale combobox defaults.
            if isinstance(settings.get("evaluation_profile_yaml"), str) and settings.get("evaluation_profile_yaml"):
                tk_vars = dict(tk_vars)
                tk_vars["var_eval_workflow_profile"] = settings.get("evaluation_profile_yaml")
            for name, value in tk_vars.items():
                # v59b: Hardware setup editor fields are backed by the central
                # ~/.onnx_splitpoint_tool/hardware_setups.yaml registry.  Older
                # GUI settings persisted these widget variables too; replaying
                # them after package updates could overwrite what is shown in
                # the Tool Config cards with stale values.  Do not apply them
                # from generic GUI settings anymore.
                if not _persist_as_generic_gui_var(str(name)):
                    continue
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

        # v56: restore canonical energy_defaults into the visible Tool Config
        # variables.  Older settings may have only stored tk_vars; canonical
        # defaults win once they exist because the scripts read this block.
        try:
            ed = settings.get("energy_defaults") if isinstance(settings.get("energy_defaults"), dict) else {}
            mapping = {
                "var_energy_enabled": ed.get("enabled"),
                "var_energy_collector_binary": ed.get("collector_binary"),
                "var_energy_power_calculations_binary": ed.get("power_calculations_binary"),
                "var_energy_data_port": ed.get("data_port"),
                "var_energy_channel": ed.get("channel"),
                "var_energy_sample_rate": ed.get("sample_rate"),
                "var_energy_environment": ed.get("environment"),
                "var_energy_pre_duration_s": ed.get("pre_duration_s"),
                "var_energy_post_duration_s": ed.get("post_duration_s"),
                "var_energy_duration_margin_s": ed.get("duration_margin_s"),
                "var_energy_min_active_duration_s": ed.get("min_active_duration_s"),
                "var_energy_run_count": ed.get("run_count"),
                "var_energy_keep_raw_parquet": ed.get("keep_raw_parquet"),
                "var_energy_include_raw_parquet_debug": ed.get("include_raw_parquet_in_debug_pack"),
                "var_energy_compare_legacy_window": ed.get("compare_legacy_window"),
                "var_energy_window_probe_enabled": ed.get("window_method_validation_probe_enabled"),
                "var_energy_window_probe_repeats": ed.get("window_method_validation_probe_repeats"),
                "var_energy_window_probe_include_raw": ed.get("window_method_validation_probe_include_raw_parquet"),
                "var_energy_window_probe_strict": ed.get("window_method_validation_probe_strict"),
            }
            for name, value in mapping.items():
                if value is None:
                    continue
                var = getattr(self, name, None)
                if isinstance(var, tk.Variable):
                    try:
                        var.set(value)
                    except Exception:
                        pass
        except Exception:
            pass

        sel = settings.get("remote_selected_host_id")
        if isinstance(sel, str) and sel:
            try:
                self.var_remote_host_id.set(sel)
            except Exception:
                pass
        try:
            # Keep old RemoteBenchmarkService paths working while the central
            # hardware registry becomes the source of truth.
            self._sync_remote_hosts_from_hardware_setups()
        except Exception:
            logger.debug("Failed to sync remote_hosts from hardware setups", exc_info=True)

        # Clean up stale semantic-validation paths loaded from persisted UI state.
        try:
            from ..benchmark.suite_refresh import normalize_semantic_validation_request

            var_imgs = getattr(self, "var_bench_validation_images", None)
            var_max = getattr(self, "var_bench_validation_max_images", None)
            var_task = getattr(self, "var_bench_task", None)
            if isinstance(var_imgs, tk.Variable):
                raw_imgs = (var_imgs.get() or "")
                raw_max = (var_max.get() if isinstance(var_max, tk.Variable) else "50") or "50"
                raw_task = (var_task.get() if isinstance(var_task, tk.Variable) else "auto") or "auto"
                norm_imgs, norm_max, use_embedded = normalize_semantic_validation_request(raw_imgs, raw_max, benchmark_task=raw_task)
                if use_embedded and str(raw_imgs).strip():
                    var_imgs.set("")
                if isinstance(var_max, tk.Variable):
                    var_max.set(str(int(norm_max or 0)))
        except Exception:
            pass

    def _collect_persistent_settings(self) -> dict:
        tk_vars = {}
        for name, value in self.__dict__.items():
            if name.startswith("var_") and isinstance(value, tk.Variable):
                # v59b: central hardware setup fields are persisted only in
                # hardware_setups.yaml.  Keeping a second copy in gui settings
                # made them appear to reset after tool updates / lazy tab rebuilds.
                if not _persist_as_generic_gui_var(name):
                    continue
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
        try:
            # v58s: keep visible energy-default controls stable across restart;
            # the canonical config file remains the source used by CLI scripts.
            data["energy_defaults"] = {
                "enabled": bool(self.var_energy_enabled.get()),
                "collector_binary": str(self.var_energy_collector_binary.get() or "urecs-data-collector"),
                "power_calculations_binary": str(self.var_energy_power_calculations_binary.get() or "power_calculations"),
                "data_port": int(float(str(self.var_energy_data_port.get() or 3000))),
                "channel": int(float(str(self.var_energy_channel.get() or 0))),
                "sample_rate": int(float(str(self.var_energy_sample_rate.get() or 2000))),
                "environment": str(self.var_energy_environment.get() or "Jetson"),
                "pre_duration_s": float(str(self.var_energy_pre_duration_s.get() or 5)),
                "post_duration_s": float(str(self.var_energy_post_duration_s.get() or 5)),
                "duration_margin_s": float(str(self.var_energy_duration_margin_s.get() or 1)),
                "min_active_duration_s": float(str(self.var_energy_min_active_duration_s.get() or 30)),
                "run_count": int(float(str(self.var_energy_run_count.get() or 3))),
                "keep_raw_parquet": bool(self.var_energy_keep_raw_parquet.get()),
                "include_raw_parquet_in_debug_pack": bool(self.var_energy_include_raw_parquet_debug.get()),
                "compare_legacy_window": bool(self.var_energy_compare_legacy_window.get()),
                "window_method_validation_probe_enabled": bool(self.var_energy_window_probe_enabled.get()),
                "window_method_validation_probe_repeats": max(1, int(float(str(self.var_energy_window_probe_repeats.get() or 3)))),
                "window_method_validation_probe_include_raw_parquet": bool(self.var_energy_window_probe_include_raw.get()),
                "window_method_validation_probe_strict": bool(self.var_energy_window_probe_strict.get()),
            }
        except Exception:
            pass
        try:
            # v58s: keep selected Evaluation Profile stable across restarts even
            # when the profile combobox default list order changes.
            prof = str(self.var_eval_workflow_profile.get() or "").strip()
            if prof:
                data["evaluation_profile_yaml"] = prof
        except Exception:
            pass
        return data

    def _persist_settings(self) -> None:
        try:
            self._settings_store.save(self._collect_persistent_settings())
        except Exception:
            logger.exception("Failed to save settings")

    def _install_evaluation_shutdown_signal_broker(self) -> None:
        """Route process signals through Tk's main loop without touching Tk in a handler."""

        self._gui_shutdown_signal_pending = 0
        self._gui_shutdown_signal_handlers: dict[int, Any] = {}
        if threading.current_thread() is not threading.main_thread():
            return

        def _request_shutdown(signum: int, _frame: Any) -> None:
            # Python invokes this callback on the main thread, but Tk calls are
            # still unsafe from a signal handler.  The periodic main-loop poll
            # below performs the actual workflow cancellation and close.
            self._gui_shutdown_signal_pending = int(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._gui_shutdown_signal_handlers[signum] = signal.getsignal(
                    signum
                )
                signal.signal(signum, _request_shutdown)
            except (AttributeError, OSError, RuntimeError, ValueError):
                continue

        def _poll_shutdown_signal() -> None:
            signum = int(
                getattr(self, "_gui_shutdown_signal_pending", 0) or 0
            )
            if signum:
                self._gui_shutdown_signal_pending = 0
                try:
                    signal_name = signal.Signals(signum).name.lower()
                except Exception:
                    signal_name = str(signum)
                logger.info(
                    "GUI shutdown requested by signal_%s; cancelling active "
                    "Evaluation Workflow workers",
                    signal_name,
                )
                self._on_close()
                return
            if bool(getattr(self, "_evaluation_close_pending", False)):
                return
            try:
                self.root.after(100, _poll_shutdown_signal)
            except Exception:
                self._restore_evaluation_shutdown_signal_broker()

        try:
            self.root.after(100, _poll_shutdown_signal)
        except Exception:
            self._restore_evaluation_shutdown_signal_broker()

    def _restore_evaluation_shutdown_signal_broker(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        previous = dict(
            getattr(self, "_gui_shutdown_signal_handlers", {}) or {}
        )
        self._gui_shutdown_signal_handlers = {}
        for signum, handler in previous.items():
            try:
                signal.signal(int(signum), handler)
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

    def _on_close(self):
        if bool(getattr(self, "_evaluation_close_pending", False)):
            return
        self._evaluation_close_pending = True
        self._gui_closing = True
        try:
            self.withdraw()
        except Exception:
            pass
        # Closing the entire GUI is a real workflow cancellation (closing only
        # a progress dialog remains a detach).  Invoke the parent workflow
        # callbacks synchronously before destroying Tk so registered Native /
        # Energy process trees are already gone when the UI disappears.
        for record in list(getattr(self, "_background_jobs", {}).values()):
            if str(getattr(record, "kind", "") or "") != "evaluation_workflow":
                continue
            if str(getattr(record, "status", "") or "").lower() not in {
                "queued",
                "running",
                "cancelling",
            }:
                continue
            callback = getattr(record, "cancel_callback", None)
            if not callable(callback):
                continue
            record.status = "cancelling"
            record.status_text = "Cancelling…"
            try:
                callback()
            except Exception:
                logger.exception("Failed to cancel Evaluation Workflow during GUI close")
        self._persist_settings()

        # Keep Tk's event loop responsive while the non-daemon workflow worker
        # finishes its bounded process/service cleanup and releases the run
        # lock.  Blocking join() here can deadlock a worker currently entering
        # root.after().
        def _destroy_after_workflows_finish() -> None:
            active_threads = []
            for record in list(
                getattr(self, "_background_jobs", {}).values()
            ):
                if str(getattr(record, "kind", "") or "") != "evaluation_workflow":
                    continue
                worker_thread = getattr(record, "worker_thread", None)
                if (
                    worker_thread is not None
                    and worker_thread is not threading.current_thread()
                    and worker_thread.is_alive()
                ):
                    active_threads.append(worker_thread)
            if active_threads:
                try:
                    self.root.after(50, _destroy_after_workflows_finish)
                    return
                except Exception:
                    return
            self._restore_evaluation_shutdown_signal_broker()
            try:
                self.destroy()
            except Exception:
                try:
                    self.root.destroy()
                except Exception:
                    pass

        try:
            self.root.after(0, _destroy_after_workflows_finish)
        except Exception:
            _destroy_after_workflows_finish()

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

            from ..paths import splitpoint_logs_dir, splitpoint_wsl_debug_dir
            from ..log_retention import LogRetentionPolicy, apply_log_retention

            roots = [splitpoint_logs_dir(), splitpoint_wsl_debug_dir()]
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
    # ------------------------------------------------------------------
    # Central hardware setup registry helpers
    # ------------------------------------------------------------------

    def _hardware_setups_path(self) -> Path:
        try:
            return ensure_hardware_setups_file()
        except Exception:
            return default_hardware_setups_file()

    def _hardware_registry_load(self) -> dict:
        """Load the central hardware_setup registry from ~/.onnx_splitpoint_tool.

        This registry is now the source of truth for the three accelerator
        runtime targets: Orin NX + Hailo-8, Orin NX + Hailo-10 and
        Orin NX + DeepX DX-M1.
        """
        p = self._hardware_setups_path()
        try:
            from ..energy.config import (
                HARDWARE_REGISTRY_REVISION_KEY,
                load_hardware_registry_with_revision,
            )

            data, revision = load_hardware_registry_with_revision(p)
            payload = dict(data) if isinstance(data, dict) else {}
            payload[HARDWARE_REGISTRY_REVISION_KEY] = revision
            return payload
        except Exception:
            logger.exception("Failed to load hardware setup registry: %s", p)
            # Fail closed if a caller nevertheless tries to save this payload:
            # no real SHA-256 can equal this sentinel.
            return {"__onnx_splitpoint_registry_file_sha256": "<load-failed>"}

    def _hardware_registry_save(self, data: Mapping[str, Any]) -> None:
        p = self._hardware_setups_path()
        payload = dict(data or {})
        from ..energy.config import (
            HARDWARE_REGISTRY_REVISION_KEY,
            HardwareRegistryConflictError,
            load_hardware_registry_with_revision,
            save_hardware_registry,
        )

        expected_revision = payload.pop(HARDWARE_REGISTRY_REVISION_KEY, None)
        # Keep the lightweight operator backup, but delegate the authoritative
        # write to the same lock + fsync + atomic-replace implementation used by
        # the CLI and platform-power backend.
        try:
            if p.exists():
                bak = p.with_suffix(p.suffix + ".bak")
                bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            logger.debug("Failed to write hardware registry backup", exc_info=True)
        try:
            if expected_revision is None:
                # Backwards-compatible path for non-GUI callers that construct
                # a payload manually.  Every in-product GUI load carries a
                # revision and therefore takes the guarded branch below.
                save_hardware_registry(payload, p)
            else:
                save_hardware_registry(
                    payload,
                    p,
                    expected_file_sha256=str(expected_revision),
                )
        except HardwareRegistryConflictError:
            # Retain the fresh snapshot for diagnostics/callers and then fail;
            # silently merging a stale full GUI payload could erase a newly
            # committed idle-power calibration and its provenance binding.
            try:
                fresh, fresh_revision = load_hardware_registry_with_revision(p)
                fresh_payload = dict(fresh)
                fresh_payload[HARDWARE_REGISTRY_REVISION_KEY] = fresh_revision
                self._hardware_registry_conflict_reload = fresh_payload
            except Exception:
                logger.exception("Failed to reload registry after stale GUI save")
            raise

    def _hardware_setup_by_id(self, setup_id: str) -> dict:
        sid = str(setup_id or "").strip()
        if not sid:
            return {}
        reg = self._hardware_registry_load()
        for raw in reg.get("hardware_setups") or []:
            if isinstance(raw, Mapping) and str(raw.get("id") or "") == sid:
                return dict(raw)
        return {}

    def _hardware_setups_values_for_combo(self) -> list[str]:
        reg = self._hardware_registry_load()
        values: list[str] = []
        for raw in reg.get("hardware_setups") or []:
            if not isinstance(raw, Mapping):
                continue
            sid = str(raw.get("id") or "").strip()
            if not sid:
                continue
            label = str(raw.get("label") or raw.get("name") or sid).strip()
            acc = str(raw.get("accelerator") or raw.get("backend") or "").strip()
            suffix = f" ({acc})" if acc else ""
            values.append(f"{sid} — {label}{suffix}")
        return values

    def _hardware_setup_id_from_display(self, value: str) -> str:
        s = str(value or "").strip()
        if "—" in s:
            return s.split("—", 1)[0].strip()
        return s

    def _hardware_setup_remote_payload(self, setup_id: str) -> dict:
        setup = self._hardware_setup_by_id(setup_id)
        if not setup:
            return {}
        host_raw = setup.get("host") or {}
        host = dict(host_raw) if isinstance(host_raw, Mapping) else {"address": str(host_raw or "")}
        runtime = dict(setup.get("runtime") or {}) if isinstance(setup.get("runtime"), Mapping) else {}
        remote = dict(setup.get("remote") or setup.get("remote_execution") or {}) if isinstance(setup.get("remote") or setup.get("remote_execution"), Mapping) else {}
        sid = str(setup.get("id") or setup_id or "").strip()
        label = str(setup.get("label") or setup.get("name") or sid).strip()
        accelerator = str(setup.get("accelerator") or setup.get("backend") or "").strip()
        provider = str(runtime.get("provider") or remote.get("provider") or setup.get("provider") or accelerator or "auto").strip() or "auto"
        venv = str(runtime.get("activate") or runtime.get("venv_activate") or runtime.get("remote_venv") or runtime.get("venv") or remote.get("remote_venv") or remote.get("venv") or "").strip()
        # A plain path like ~/venvs/deepx-runtime/bin/activate is accepted by
        # the remote runner, but the UI should prefer explicit source commands.
        if venv and venv.endswith("/activate") and not venv.strip().startswith(("source ", ". ")):
            venv = "source " + venv
        base = str(host.get("base_dir") or host.get("remote_base_dir") or remote.get("remote_base_dir") or runtime.get("remote_base_dir") or "~/splitpoint_runs").strip() or "~/splitpoint_runs"
        return {
            "id": sid,
            "label": label,
            "host": str(host.get("address") or host.get("host") or remote.get("host") or "").strip(),
            "user": str(host.get("user") or remote.get("user") or runtime.get("user") or "nx").strip(),
            "port": int(host.get("port") or remote.get("port") or runtime.get("port") or 22),
            "remote_base_dir": base,
            "ssh_extra_args": str(host.get("ssh_extra_args") or remote.get("ssh_extra_args") or "").strip(),
            "remote_venv": venv,
            "provider": provider,
            "accelerator": accelerator,
        }

    def _hardware_setup_summary(self, setup_id: str) -> str:
        p = self._hardware_setup_remote_payload(setup_id)
        if not p:
            return "No hardware setup selected. Configure setups in Tool Config."
        target = f"{p.get('user') + '@' if p.get('user') else ''}{p.get('host') or '(host missing)'}:{p.get('port') or 22}"
        return f"{p.get('label') or p.get('id')} · {target} · provider={p.get('provider') or 'auto'} · venv={p.get('remote_venv') or '(auto)'}"

    def _hardware_setup_host_config(self, setup_id: str) -> HostConfig | None:
        p = self._hardware_setup_remote_payload(setup_id)
        if not p or not str(p.get("host") or "").strip():
            return None
        return HostConfig(
            id=str(p.get("id") or setup_id),
            label=str(p.get("label") or setup_id),
            host=str(p.get("host") or ""),
            user=str(p.get("user") or "nx"),
            port=int(p.get("port") or 22),
            remote_base_dir=str(p.get("remote_base_dir") or "~/splitpoint_runs"),
            ssh_extra_args=str(p.get("ssh_extra_args") or ""),
        )

    def _hardware_setup_smoke_command(self, setup_id: str) -> str:
        p = self._hardware_setup_remote_payload(setup_id)
        provider = str(p.get("provider") or p.get("accelerator") or "auto").strip().lower()
        acc = str(p.get("accelerator") or provider).strip().lower()
        venv = str(p.get("remote_venv") or "").strip()
        lines = ["set -e"]
        if venv:
            if any(ch.isspace() for ch in venv):
                lines.append(venv)
            else:
                vv = venv
                if vv.startswith("~/"):
                    vv = "$HOME/" + vv[2:]
                lines.append(f'if [ -f "{vv}" ]; then source "{vv}"; fi')
        lines += [
            'echo "[preflight] host=$(hostname) user=$(whoami)"',
            'echo "[preflight] python=$(command -v python3 || command -v python || true)"',
            'python3 - <<\'PY\'\nimport sys\nprint("python", sys.executable)\ntry:\n import numpy as np; print("numpy", np.__version__)\nexcept Exception as e: print("numpy FAILED", repr(e))\ntry:\n import cv2; print("cv2", cv2.__version__)\nexcept Exception as e: print("cv2 FAILED", repr(e))\ntry:\n import onnx; print("onnx", onnx.__version__)\nexcept Exception as e: print("onnx FAILED", repr(e))\ntry:\n import onnxruntime as ort; print("onnxruntime", ort.__version__); print("providers", ort.get_available_providers())\nexcept Exception as e: print("onnxruntime FAILED", repr(e))\nPY',
        ]
        if "deepx" in acc or "deepx" in provider or "dx_m1" in provider:
            lines += [
                'echo "[deepx] device files:"; ls -l /dev/dxrt* 2>/dev/null || true',
                'echo "[deepx] dxrt-cli:"; command -v dxrt-cli || true; dxrt-cli -s 2>/dev/null || true',
                'echo "[deepx] tools:"; command -v parse_model || true; command -v run_model || true',
                'python3 - <<\'PY\'\ntry:\n from dx_engine import InferenceEngine\n print("dx_engine OK")\nexcept Exception as e:\n print("dx_engine FAILED", repr(e))\ntry:\n import tensorrt as trt\n print("TensorRT", trt.__version__)\nexcept Exception as e:\n print("TensorRT FAILED", repr(e))\nPY',
            ]
        if "hailo" in acc or "hailo" in provider:
            lines += [
                'echo "[hailo] tools:"; command -v hailortcli || true; command -v hailort || true',
                'hailortcli scan 2>/dev/null || hailortcli fw-control identify 2>/dev/null || true',
                'python3 - <<\'PY\'\ntry:\n import hailo_platform\n print("hailo_platform OK")\nexcept Exception as e:\n print("hailo_platform FAILED", repr(e))\nPY',
            ]
        return "\n".join(lines)

    def _test_hardware_setup_remote(self, setup_id: str) -> None:
        sid = self._hardware_setup_id_from_display(setup_id)
        host = self._hardware_setup_host_config(sid)
        if host is None:
            messagebox.showwarning("Hardware setup test", f"No SSH host configured for {sid}.")
            return
        title = f"Hardware setup test — {sid}"

        def _worker() -> None:
            chunks: list[str] = []
            try:
                ok, msg = SSHTransport(host).test_connection(timeout_s=10)
                chunks.append(f"=== SSH connection test ({host.user_host_pretty}) ===\nOK={ok}\n{msg}\n")
                if ok:
                    cmd = self._hardware_setup_smoke_command(sid)
                    rc, out = SSHTransport(host).run(cmd, timeout=90)
                    chunks.append(f"\n=== Runtime smoke test ===\nrc={rc}\n{out}\n")
            except Exception as exc:
                chunks.append(f"\nERROR: {type(exc).__name__}: {exc}\n")
            body = "\n".join(chunks)
            try:
                self.root.after(0, lambda: self._popup_text(title, body, width=120, height=34))
            except Exception:
                pass
        threading.Thread(target=_worker, name=f"hardware-setup-test-{sid}", daemon=True).start()

    def _test_all_hardware_setups_remote(self) -> None:
        reg = self._hardware_registry_load()
        ids = []
        for raw in reg.get("hardware_setups") or []:
            if isinstance(raw, Mapping):
                sid = str(raw.get("id") or "").strip()
                if sid and self._hardware_setup_remote_payload(sid).get("host"):
                    ids.append(sid)
        if not ids:
            messagebox.showwarning("Hardware setup test", "No configured hardware setup has an SSH host.")
            return

        def _worker() -> None:
            chunks: list[str] = []
            for sid in ids:
                host = self._hardware_setup_host_config(sid)
                if host is None:
                    continue
                chunks.append("\n" + "=" * 96 + f"\n{sid} — {host.user_host_pretty}\n")
                try:
                    ok, msg = SSHTransport(host).test_connection(timeout_s=10)
                    chunks.append(f"SSH OK={ok}\n{msg}\n")
                    if ok:
                        rc, out = SSHTransport(host).run(self._hardware_setup_smoke_command(sid), timeout=90)
                        chunks.append(f"Runtime smoke rc={rc}\n{out}\n")
                except Exception as exc:
                    chunks.append(f"ERROR: {type(exc).__name__}: {exc}\n")
            body = "\n".join(chunks)
            try:
                self.root.after(0, lambda: self._popup_text("Hardware setup tests", body, width=130, height=38))
            except Exception:
                pass
        threading.Thread(target=_worker, name="hardware-setup-test-all", daemon=True).start()

    def _apply_hardware_setup_to_remote_vars(self, setup_id: str, *, persist: bool = True) -> dict:
        """Resolve a central hardware setup into the legacy remote vars.

        The benchmark runner still consumes HostConfig + RemoteBenchmarkArgs.
        This method bridges the new central setup registry into those legacy
        variables so old code paths keep working while the UI no longer exposes
        duplicate host/venv fields.
        """
        sid = self._hardware_setup_id_from_display(setup_id)
        p = self._hardware_setup_remote_payload(sid)
        if not p:
            return {}
        try:
            self.var_remote_hardware_setup_id.set(sid)
            self.var_remote_host_id.set(sid)
            # Keep provider override under user control.  Most benchmark plans
            # should run with provider=auto so TensorRT references and the
            # selected accelerator rows execute together.
            if not str(self.var_remote_provider.get() or "").strip():
                self.var_remote_provider.set("auto")
            self.var_remote_venv.set(str(p.get("remote_venv") or ""))
        except Exception:
            pass
        # Keep remote_hosts compatible with the existing RemoteBenchmarkService.
        host_entry = {
            "id": str(p.get("id") or sid),
            "label": str(p.get("label") or sid),
            "host": str(p.get("host") or ""),
            "user": str(p.get("user") or ""),
            "port": int(p.get("port") or 22),
            "remote_base_dir": str(p.get("remote_base_dir") or "~/splitpoint_runs"),
            "ssh_extra_args": str(p.get("ssh_extra_args") or ""),
        }
        hosts = [dict(h) for h in (getattr(self, "remote_hosts", []) or []) if isinstance(h, Mapping) and str(h.get("id") or "") != str(host_entry["id"])]
        hosts.append(host_entry)
        self.remote_hosts = hosts
        if persist:
            self._persist_settings()
        return p

    def _open_hardware_setups_config(self) -> None:
        self._open_path(str(self._hardware_setups_path()))

    def _open_tool_config_tab(self) -> None:
        try:
            frame = self.panel_frames.get("hardware")
            if frame is not None:
                self.main_notebook.select(frame)
        except Exception:
            pass

    def _sync_remote_hosts_from_hardware_setups(self) -> None:
        """Populate legacy remote_hosts from the central hardware registry."""
        reg = self._hardware_registry_load()
        added = []
        for raw in reg.get("hardware_setups") or []:
            if not isinstance(raw, Mapping):
                continue
            payload = self._hardware_setup_remote_payload(str(raw.get("id") or ""))
            if payload.get("host"):
                added.append({
                    "id": str(payload.get("id") or ""),
                    "label": str(payload.get("label") or payload.get("id") or ""),
                    "host": str(payload.get("host") or ""),
                    "user": str(payload.get("user") or ""),
                    "port": int(payload.get("port") or 22),
                    "remote_base_dir": str(payload.get("remote_base_dir") or "~/splitpoint_runs"),
                    "ssh_extra_args": str(payload.get("ssh_extra_args") or ""),
                })
        if not added:
            return
        merged = [dict(h) for h in (getattr(self, "remote_hosts", []) or []) if isinstance(h, Mapping)]
        for entry in added:
            merged = [h for h in merged if str(h.get("id") or "") != str(entry.get("id") or "")]
            merged.append(entry)
        self.remote_hosts = merged

    # Remote benchmark helpers (Benchmark tab)
    # ------------------------------------------------------------------

    def _remote_host_configs(self):
        """Return central hardware setups as selectable remote targets.

        The Tool Config hardware_setup registry is now the preferred source of
        truth.  Legacy Remote Hosts are appended only for backwards-compatible
        manual runs.
        """
        hosts: list[HostConfig] = []
        try:
            reg = self._hardware_registry_load()
            for raw in reg.get("hardware_setups") or []:
                if not isinstance(raw, Mapping):
                    continue
                sid = str(raw.get("id") or "").strip()
                payload = self._hardware_setup_remote_payload(sid)
                if not payload or not str(payload.get("host") or "").strip():
                    continue
                hosts.append(HostConfig(
                    id=str(payload.get("id") or sid),
                    label=str(payload.get("label") or sid),
                    host=str(payload.get("host") or ""),
                    user=str(payload.get("user") or ""),
                    port=int(payload.get("port") or 22),
                    remote_base_dir=str(payload.get("remote_base_dir") or "~/splitpoint_runs"),
                    ssh_extra_args=str(payload.get("ssh_extra_args") or ""),
                ))
        except Exception:
            logger.exception("Failed to enumerate central hardware setups")
        seen = {h.id for h in hosts}
        for h in list(self._remote_service.host_configs(getattr(self, "remote_hosts", []) or [])):
            if h.id not in seen:
                hosts.append(h)
                seen.add(h.id)
        return hosts

    def _remote_get_selected_host(self) -> HostConfig | None:
        sel = ""
        try:
            sel = self.var_remote_host_id.get()
        except Exception:
            sel = ""
        host = self._remote_service.get_selected_host(getattr(self, "remote_hosts", []) or [], sel)
        if host is not None:
            return host
        # New central hardware setup path: the selected id may be a setup id
        # rather than a legacy remote-host id.  Resolve and mirror it into the
        # legacy host list before retrying.
        try:
            setup_id = self.var_remote_hardware_setup_id.get() or sel
        except Exception:
            setup_id = sel
        if setup_id:
            payload = self._apply_hardware_setup_to_remote_vars(setup_id, persist=False)
            if payload:
                sel2 = str(payload.get("id") or setup_id)
                return self._remote_service.get_selected_host(getattr(self, "remote_hosts", []) or [], sel2)
        return None

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
        # If the selected id is one of the central hardware setups, also fill
        # provider + runtime env from Tool Config.  This keeps the Benchmark tab
        # in sync without maintaining a second host/venv form.
        try:
            if self._hardware_setup_by_id(host_id):
                self._apply_hardware_setup_to_remote_vars(host_id, persist=False)
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


    # ------------------------------------------------------------------
    # Central hardware setup tests + automatic remote dispatch planning
    # ------------------------------------------------------------------

    def _host_config_from_hardware_payload(self, payload: Mapping[str, Any]) -> HostConfig:
        return HostConfig(
            id=str(payload.get("id") or payload.get("label") or "hardware"),
            label=str(payload.get("label") or payload.get("id") or "hardware"),
            host=str(payload.get("host") or ""),
            user=str(payload.get("user") or ""),
            port=int(payload.get("port") or 22),
            remote_base_dir=str(payload.get("remote_base_dir") or "~/splitpoint_runs"),
            ssh_extra_args=str(payload.get("ssh_extra_args") or ""),
        )

    def _hardware_setup_test_script(self, setup_id: str) -> str:
        payload = self._hardware_setup_remote_payload(setup_id)
        acc = canon_accelerator(payload.get("accelerator") or payload.get("provider") or setup_id)
        venv = str(payload.get("remote_venv") or "").strip()
        lines: list[str] = [
            "set -o pipefail",
            "echo '[splitpoint-test] host='$(hostname)' user='$(whoami)' pwd='$(pwd)",
        ]
        if venv:
            # Accept either 'source ...' or a raw activate-script path.
            if venv.startswith(("source ", ". ")):
                lines.append(f"{venv} >/dev/null 2>&1 || {{ echo '[err] failed to activate: {venv}'; exit 12; }}")
            else:
                lines.append(f"source {venv} >/dev/null 2>&1 || {{ echo '[err] failed to activate: {venv}'; exit 12; }}")
        lines.extend([
            "echo '[splitpoint-test] python='$(command -v python3 || command -v python || true)",
            "python3 - <<'PY'",
            "import sys",
            "print('python_executable:', sys.executable)",
            "try:",
            "    import numpy as np; print('numpy:', np.__version__)",
            "except Exception as e: print('numpy: FAILED', repr(e))",
            "try:",
            "    import cv2; print('cv2:', cv2.__version__)",
            "except Exception as e: print('cv2: FAILED', repr(e))",
            "try:",
            "    import onnxruntime as ort; print('onnxruntime:', ort.__version__, 'providers=', ort.get_available_providers())",
            "except Exception as e: print('onnxruntime: FAILED', repr(e))",
            "PY",
        ])
        if acc.startswith("hailo"):
            lines.extend([
                "echo '[splitpoint-test] Hailo checks'",
                "command -v hailortcli >/dev/null 2>&1 && hailortcli fw-control identify || true",
                "python3 - <<'PY'",
                "ok=False",
                "try:",
                "    import hailo_platform; print('hailo_platform: OK'); ok=True",
                "except Exception as e: print('hailo_platform: FAILED', repr(e))",
                "try:",
                "    import hailort; print('hailort: OK'); ok=True",
                "except Exception as e: print('hailort: FAILED', repr(e))",
                "raise SystemExit(0 if ok else 21)",
                "PY",
            ])
        elif acc == "deepx_m1":
            lines.extend([
                "echo '[splitpoint-test] DeepX checks'",
                "lspci -nn | grep -Ei '1ff4|deepx|processing' || true",
                "ls -l /dev/dxrt* 2>/dev/null || true",
                "command -v dxrt-cli >/dev/null 2>&1 && dxrt-cli -s || true",
                "command -v run_model >/dev/null 2>&1 && echo 'run_model:'$(command -v run_model) || true",
                "command -v parse_model >/dev/null 2>&1 && echo 'parse_model:'$(command -v parse_model) || true",
                "command -v trtexec >/dev/null 2>&1 && echo 'trtexec:'$(command -v trtexec) || true",
                "python3 - <<'PY'",
                "import ctypes.util",
                "try:",
                "    from dx_engine import InferenceEngine; print('dx_engine: OK')",
                "except Exception as e: print('dx_engine: FAILED', repr(e)); raise SystemExit(31)",
                "try:",
                "    import tensorrt as trt; print('tensorrt:', trt.__version__)",
                "except Exception as e: print('tensorrt: FAILED', repr(e))",
                "for lib in ('cublas','cublasLt','cudnn','cufft'):",
                "    print(f'{lib}:', ctypes.util.find_library(lib))",
                "PY",
            ])
        return "\n".join(lines)

    def _test_hardware_setup(self, setup_id: str, status_var: Any = None) -> None:
        """UI callback used by Tool Config setup cards.

        The actual SSH/runtime test runs asynchronously and shows a details
        popup when finished.  status_var is optional so older callers can use
        the same method without caring about Tk variables.
        """
        try:
            if status_var is not None:
                status_var.set("testing…")
        except Exception:
            pass
        self._test_hardware_setup_async(setup_id, status_var=status_var)

    def _test_hardware_setup_async(self, setup_id: str, status_var: Any = None) -> None:
        sid = self._hardware_setup_id_from_display(setup_id)
        payload = self._hardware_setup_remote_payload(sid)
        if not payload or not str(payload.get("host") or "").strip():
            try:
                if status_var is not None:
                    status_var.set("not configured")
            except Exception:
                pass
            messagebox.showwarning("Hardware setup test", f"No host configured for setup: {sid}")
            return
        host = self._host_config_from_hardware_payload(payload)
        script = self._hardware_setup_test_script(sid)

        def worker():
            parts: list[str] = [f"Setup: {sid}", f"Remote: {host.user_host_pretty}", ""]
            ok, msg = SSHTransport(host).test_connection(timeout_s=12)
            parts.append("--- SSH connectivity ---")
            parts.append(msg)
            warn = False
            if ok:
                rc, out = SSHTransport(host).run(script, timeout=45)
                parts.append("\n--- Runtime preflight ---")
                parts.append(f"rc={rc}")
                parts.append(out)
                ok2 = (rc == 0)
                low = (out or "").lower()
                # A Hailo-only venv may not carry cv2/onnx, while the runtime
                # itself is fine.  Keep that visible as a warning instead of
                # leaving the status label stuck at testing.
                warn = ok2 and (" failed" in low or "failed " in low or "module not found" in low or "warning" in low)
            else:
                ok2 = False
            if ok and ok2 and warn:
                title = "Hardware setup OK (warnings)"
                status_text = "OK (warnings)"
            elif ok and ok2:
                title = "Hardware setup OK"
                status_text = "OK"
            else:
                title = "Hardware setup FAILED"
                status_text = "FAILED"
            body = "\n".join(parts)
            def _finish_popup():
                try:
                    if status_var is not None:
                        status_var.set(status_text)
                except Exception:
                    pass
                self._popup_text(title, body, width=120, height=34)
            self.root.after(0, _finish_popup)

        threading.Thread(target=worker, daemon=True).start()


    # ------------------------------------------------------------------
    # Energy measurement helpers (v56a/b)
    # ------------------------------------------------------------------
    def _energy_test_tools_async(self) -> None:
        """Check local urecs-data-collector and power_calculations binaries."""
        def _worker() -> None:
            import json
            try:
                from onnx_splitpoint_tool.energy.config import load_hardware_registry, energy_defaults_from_registry
                from onnx_splitpoint_tool.energy.collector import check_energy_tools
                reg = load_hardware_registry(self._hardware_setups_path())
                defaults = energy_defaults_from_registry(reg)
                result = check_energy_tools(defaults)
                ok = bool(result.get("collector_found"))
                summary = "SUCCESS: u.RECS collector found" if ok else "FAIL: u.RECS collector not found"
                if result.get("power_calculations_found"):
                    summary += "\nPower calculations found."
                else:
                    summary += "\nPower calculations missing; raw parquet can still be collected."
                details = json.dumps(result, indent=2, ensure_ascii=False)
                self.root.after(0, lambda: show_diagnostic_dialog(
                    self.root,
                    title="Energy tools check",
                    heading=summary.splitlines()[0],
                    summary="\n".join(summary.splitlines()[1:]),
                    details=details,
                    severity="success" if ok else "error",
                ))
            except Exception as exc:
                self.root.after(0, lambda: show_diagnostic_dialog(self.root, title="Energy tools check", heading="FAIL: energy tool check crashed", summary=f"{type(exc).__name__}: {exc}", details=repr(exc), severity="error"))
        threading.Thread(target=_worker, name="energy-tools-check", daemon=True).start()

    def _energy_test_setup_async(self, setup_id: str, status_var=None) -> None:
        """Run a short u.RECS fast-firmware sleep measurement for one setup."""
        sid = self._hardware_setup_id_from_display(setup_id)
        def _set_status(text: str) -> None:
            try:
                if status_var is not None:
                    self.root.after(0, lambda: status_var.set(text))
            except Exception:
                pass
        def _worker() -> None:
            import json, time
            try:
                from pathlib import Path
                from onnx_splitpoint_tool.energy.config import load_hardware_registry, energy_defaults_from_registry, energy_setup_from_registry, energy_measurements_root
                from onnx_splitpoint_tool.energy.collector import test_fast_firmware_sleep
                reg = load_hardware_registry(self._hardware_setups_path())
                setup = energy_setup_from_registry(reg, sid)
                if not setup.urecs_address:
                    raise RuntimeError(f"u.RECS address missing for setup {sid}. Configure it in Tool Config → Accelerator envs.")
                defaults = energy_defaults_from_registry(reg)
                out_dir = energy_measurements_root(getattr(self, "default_output_dir", None)) / "Tests" / sid / time.strftime("%Y%m%d_%H%M%S")
                result = test_fast_firmware_sleep(sid, out_dir=out_dir, sleep_s=2.0, registry_path=self._hardware_setups_path())
                ok = bool(result.get("ok"))
                _set_status("energy OK" if ok else "energy FAIL")
                summary = f"Output: {out_dir}\n"
                if result.get("avg_power_w") is not None:
                    summary += f"Average power: {result.get('avg_power_w'):.3f} W\n"
                if result.get("avg_energy_total_j") is not None:
                    summary += f"Average energy: {result.get('avg_energy_total_j'):.3f} J\n"
                details = json.dumps(result, indent=2, ensure_ascii=False)
                self.root.after(0, lambda: show_diagnostic_dialog(self.root, title=f"Energy setup test — {sid}", heading=("SUCCESS" if ok else "FAIL") + f": {sid}", summary=summary, details=details, severity="success" if ok else "error"))
            except Exception as exc:
                _set_status("energy FAIL")
                self.root.after(0, lambda: show_diagnostic_dialog(self.root, title=f"Energy setup test — {sid}", heading=f"FAIL: {sid}", summary=f"{type(exc).__name__}: {exc}", details=repr(exc), severity="error"))
        threading.Thread(target=_worker, name=f"energy-test-{sid}", daemon=True).start()

    def _test_energy_setups_for_current_suite(self) -> None:
        """Run short u.RECS energy checks for setups used by the current suite.

        This intentionally reuses the central hardware registry.  It is a GUI
        convenience button for the Benchmark tab and does not affect benchmark
        results.
        """
        bench_json = self.var_remote_benchmark_set.get().strip() if hasattr(self, "var_remote_benchmark_set") else ""
        setups: list[str] = []
        try:
            if bench_json:
                bench_path = Path(bench_json).expanduser()
                if bench_path.is_dir():
                    bench_path = bench_path / "benchmark_set.json"
                if bench_path.exists():
                    setups = [str(d.get("setup_id") or "") for d in self._auto_remote_dispatch_plan(bench_path) if str(d.get("setup_id") or "")]
        except Exception:
            setups = []
        if not setups:
            try:
                reg = self._hardware_registry_load()
                for raw in reg.get("hardware_setups") or []:
                    if not isinstance(raw, Mapping):
                        continue
                    sid = str(raw.get("id") or "").strip()
                    energy = raw.get("energy") if isinstance(raw.get("energy"), Mapping) else {}
                    if sid and bool(energy.get("enabled")) and str(energy.get("urecs_address") or "").strip():
                        setups.append(sid)
            except Exception:
                setups = []
        # unique, stable order
        uniq: list[str] = []
        for sid in setups:
            if sid and sid not in uniq:
                uniq.append(sid)
        setups = uniq
        if not setups:
            messagebox.showwarning(
                "Energy setup test",
                "No energy-enabled hardware setup found. Enable Energy and set the u.RECS address in Tool Config → Accelerator envs.",
            )
            return
        if not messagebox.askyesno(
            "Energy setup test",
            "Run a short u.RECS fast-firmware sleep measurement for the selected setup(s)?\n\n"
            + "\n".join(f"• {sid}" for sid in setups)
            + "\n\nEach setup takes roughly pre + 2s + post seconds.",
        ):
            return
        for sid in setups:
            try:
                self._energy_test_setup_async(sid)
            except Exception:
                logger.exception("Failed to start energy setup test for %s", sid)

    def _test_hardware_setups_for_current_suite(self) -> None:
        bench_json = self.var_remote_benchmark_set.get().strip() if hasattr(self, "var_remote_benchmark_set") else ""
        if bench_json:
            bench_path = Path(bench_json).expanduser()
            if bench_path.is_dir():
                bench_path = bench_path / "benchmark_set.json"
        else:
            bench_path = None
        setups: list[str] = []
        try:
            if bench_path and bench_path.exists():
                setups = [str(d.get("setup_id") or "") for d in self._auto_remote_dispatch_plan(bench_path) if str(d.get("setup_id") or "")]
        except Exception:
            setups = []
        if not setups:
            reg = self._hardware_registry_load()
            setups = [str(x.get("id") or "") for x in reg.get("hardware_setups") or [] if isinstance(x, Mapping) and str(x.get("host", {}).get("address") if isinstance(x.get("host"), Mapping) else x.get("host") or "").strip()]
        if not setups:
            messagebox.showwarning("Hardware setup test", "No configured hardware setups with hosts found.")
            return
        # Test sequentially and show one combined result.
        def worker():
            all_parts: list[str] = []
            for sid in setups:
                payload = self._hardware_setup_remote_payload(sid)
                if not payload:
                    all_parts.append(f"=== {sid} ===\nmissing setup\n")
                    continue
                host = self._host_config_from_hardware_payload(payload)
                all_parts.append(f"=== {sid} — {payload.get('label') or ''} ===")
                ok, msg = SSHTransport(host).test_connection(timeout_s=12)
                all_parts.append(msg)
                if ok:
                    rc, out = SSHTransport(host).run(self._hardware_setup_test_script(sid), timeout=45)
                    all_parts.append(f"runtime rc={rc}")
                    all_parts.append(out)
                all_parts.append("")
            body = "\n".join(all_parts)
            self.root.after(0, lambda: self._popup_text("Hardware setups test", body, width=125, height=38))
        threading.Thread(target=worker, daemon=True).start()

    def _benchmark_plan_runs_for_suite(self, benchmark_set_json: Path) -> list[dict]:
        """Return the *authoritative* frozen run plan for a benchmark set.

        Important distinction:
        - benchmark_plan.json / benchmark_set.json["plan"]["runs"] is the frozen
          plan that was actually materialized when the benchmark set was created.
        - evaluation_profile.run_profile_ids or root-level legacy run_profiles are
          only the profile request / provenance and may contain targets that were
          later filtered, backfilled, or not materialized.

        v59g: prefer the frozen plan strictly.  This prevents the Benchmark tab
        from showing Hailo/CUDA dispatches just because they were requested in an
        old profile while the actual benchmark_plan.json only contains e.g.
        ort_tensorrt + deepx rows.
        """
        bench_path = Path(benchmark_set_json).expanduser()
        if bench_path.is_dir():
            bench_path = bench_path / "benchmark_set.json"
        suite_dir = bench_path.parent

        def _rows_from(data, source: str, *, root_legacy_ok: bool = False) -> list[dict]:
            if not isinstance(data, Mapping):
                return []
            # Authoritative top-level plan files.
            for key in ("runs", "plan_runs", "benchmark_runs"):
                value = data.get(key)
                if isinstance(value, list) and value:
                    rows = [dict(x) for x in value if isinstance(x, Mapping)]
                    for r in rows:
                        r.setdefault("_plan_source", source)
                    return rows
            # benchmark_set.json usually nests the authoritative plan here.
            plan = data.get("plan")
            if isinstance(plan, Mapping):
                for key in ("runs", "plan_runs", "benchmark_runs"):
                    value = plan.get(key)
                    if isinstance(value, list) and value:
                        rows = [dict(x) for x in value if isinstance(x, Mapping)]
                        for r in rows:
                            r.setdefault("_plan_source", source + ":plan")
                        return rows
            # Legacy fallback only.  Do not treat profile run_profiles as a frozen
            # plan if a real plan exists or should exist.
            if root_legacy_ok:
                for key in ("run_profiles",):
                    value = data.get(key)
                    if isinstance(value, list) and value:
                        rows = []
                        for x in value:
                            if isinstance(x, Mapping):
                                row = dict(x)
                            else:
                                rid = str(x).strip()
                                row = {"id": rid, "run_id": rid, "name": rid}
                            if row:
                                row.setdefault("_plan_source", source + ":legacy_run_profiles")
                                rows.append(row)
                        if rows:
                            return rows
            return []

        # 1) benchmark_plan.json is the source of truth if present.
        plan_path = suite_dir / "benchmark_plan.json"
        if plan_path.exists():
            try:
                data = json.loads(plan_path.read_text(encoding="utf-8"))
                rows = _rows_from(data, "benchmark_plan.json", root_legacy_ok=False)
                if rows:
                    return rows
            except Exception:
                logger.debug("Failed to read authoritative benchmark_plan.json", exc_info=True)

        # 2) fallback to the nested plan inside benchmark_set.json.
        if bench_path.exists():
            try:
                data = json.loads(bench_path.read_text(encoding="utf-8"))
                rows = _rows_from(data, "benchmark_set.json", root_legacy_ok=False)
                if rows:
                    return rows
                # Only if there is no nested/frozen plan at all, use legacy root
                # run_profiles.  This should be rare and is labelled explicitly.
                rows = _rows_from(data, "benchmark_set.json", root_legacy_ok=True)
                if rows:
                    return rows
            except Exception:
                logger.debug("Failed to read benchmark_set.json plan", exc_info=True)

        # v56h: legacy generated suites can have only a generation log.  Parse the
        # final Plan runs line as a last-resort dispatch hint.
        gen_log = suite_dir / "benchmark_generation.log"
        if gen_log.exists():
            try:
                lines = gen_log.read_text(encoding="utf-8", errors="replace").splitlines()
                plan_line = ""
                for line in lines:
                    if "Plan runs:" in line:
                        plan_line = line
                if plan_line:
                    raw = plan_line.split("Plan runs:", 1)[1]
                    ids = [x.strip() for x in raw.split(",") if x.strip()]
                    rows = []
                    for rid in ids:
                        low = rid.lower()
                        row = {"id": rid, "run_id": rid, "name": rid, "_plan_source": "benchmark_generation.log"}
                        if "deepx" in low or "dx_m1" in low or "dxm1" in low:
                            row["provider"] = "deepx_m1"
                            row["stage1"] = "deepx" if low.startswith("deepx") else "tensorrt"
                            row["stage2"] = "deepx" if "to_deepx" in low or low.startswith("tensorrt_to_deepx") else "tensorrt"
                        elif "hailo" in low:
                            row["provider"] = "hailo8"
                            row["stage1"] = "hailo8" if "hailo" in low and not low.startswith("trt_to") and not low.startswith("tensorrt_to") else "tensorrt"
                            row["stage2"] = "hailo8" if "to_hailo" in low or low.startswith("trt_to") or low.startswith("tensorrt_to") else "tensorrt"
                        elif "cuda" in low:
                            row["provider"] = "cuda"
                            row["type"] = "onnxruntime"
                        elif "trt" in low or "tensorrt" in low:
                            row["provider"] = "tensorrt"
                            row["type"] = "onnxruntime"
                        else:
                            row["provider"] = "cpu"
                            row["type"] = "onnxruntime"
                        rows.append(row)
                    if rows:
                        return rows
            except Exception:
                logger.debug("Failed to parse benchmark_generation.log for plan runs", exc_info=True)
        return []

    def _benchmark_profile_requested_run_ids_for_suite(self, benchmark_set_json: Path) -> list[str]:
        """Return run_profile_ids from provenance only; not authoritative."""
        bench_path = Path(benchmark_set_json).expanduser()
        if bench_path.is_dir():
            bench_path = bench_path / "benchmark_set.json"
        if not bench_path.exists():
            return []
        try:
            data = json.loads(bench_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        prof = data.get("evaluation_profile") if isinstance(data, Mapping) else None
        ids = []
        if isinstance(prof, Mapping):
            raw = prof.get("run_profile_ids")
            if isinstance(raw, list):
                ids.extend(str(x).strip() for x in raw if str(x).strip())
            ov = prof.get("overrides")
            if isinstance(ov, Mapping):
                raw = ov.get("run_profile_ids")
                if isinstance(raw, list):
                    ids.extend(str(x).strip() for x in raw if str(x).strip())
        # Preserve order, remove duplicates.
        out = []
        for x in ids:
            if x and x not in out:
                out.append(x)
        return out

    def _benchmark_run_id(self, row: Mapping[str, Any]) -> str:
        return str(row.get("id") or row.get("name") or row.get("run_id") or row.get("run") or row.get("run_name") or "").strip()

    def _benchmark_row_text(self, row: Mapping[str, Any]) -> str:
        try:
            return json.dumps(row, sort_keys=True).lower()
        except Exception:
            return str(row).lower()

    def _setup_id_for_accelerator(self, accelerator: str) -> str:
        acc = canon_accelerator(accelerator)
        reg = self._hardware_registry_load()
        entries = [dict(x) for x in reg.get("hardware_setups") or [] if isinstance(x, Mapping)]
        # Preferred exact accelerator match.
        for raw in entries:
            r_acc = canon_accelerator(raw.get("accelerator") or raw.get("backend") or raw.get("provider") or raw.get("id"))
            if r_acc == acc:
                return str(raw.get("id") or "")
        # TensorRT/CUDA/CPU references can run on any NX. Prefer DeepX because
        # it is the setup where TensorRT has been validated in this project.
        if acc in {"tensorrt", "cuda", "cpu", "cpu_ort", "cuda_ort", "ort_tensorrt"}:
            for wanted in ("deepx_m1", "hailo8", "hailo10"):
                for raw in entries:
                    if canon_accelerator(raw.get("accelerator") or raw.get("backend") or raw.get("provider") or raw.get("id")) == wanted:
                        return str(raw.get("id") or "")
        return ""

    def _accelerator_for_benchmark_run(self, row: Mapping[str, Any]) -> str:
        # Explicit mapping wins.
        hw_ref = str(row.get("hardware_setup_id") or row.get("hardware_setup") or row.get("target_setup") or "").strip()
        if hw_ref:
            setup = self._hardware_setup_by_id(hw_ref)
            if setup:
                return canon_accelerator(setup.get("accelerator") or setup.get("backend") or setup.get("provider") or hw_ref)
        text = self._benchmark_row_text(row)
        if "deepx" in text or "dx_m1" in text or "dxm1" in text:
            return "deepx_m1"
        if "hailo10" in text:
            return "hailo10"
        if "hailo8" in text or '"hailo"' in text or "hailo_to" in text or "to_hailo" in text:
            return "hailo8"
        if "tensorrt" in text or "trt" in text or "cuda" in text:
            return "tensorrt"
        return "cpu"

    def _row_is_plain_host_reference(self, row: Mapping[str, Any]) -> bool:
        text = self._benchmark_row_text(row)
        if "hailo" in text or "deepx" in text or "dx_m1" in text or "dxm1" in text:
            return False
        provider = str(row.get("provider") or row.get("full_provider") or "").strip().lower()
        typ = str(row.get("type") or "").strip().lower()
        return typ in {"onnxruntime", "ort"} or provider in {"cpu", "cuda", "tensorrt", "trt"}

    def _row_mentions_accelerator(self, row: Mapping[str, Any], accelerator: str) -> bool:
        acc = canon_accelerator(accelerator)
        text = self._benchmark_row_text(row)
        if acc == "deepx_m1":
            return "deepx" in text or "dx_m1" in text or "dxm1" in text
        # v59ab: Hailo-8 and Hailo-10 are not interchangeable.  Some benchmark
        # rows still contain generic provider fields such as "provider": "hailo".
        # Do not let those generic fields route hailo10h/hailo10_to_trt rows to
        # a Hailo-8 testbed.  Explicit architecture markers win.
        has_hailo10 = ("hailo10" in text) or ("hailo_10" in text) or ("hailo-10" in text) or ("h10" in text)
        has_hailo8 = ("hailo8" in text) or ("hailo_8" in text) or ("hailo-8" in text) or ("h8" in text)
        if acc.startswith("hailo10"):
            return has_hailo10
        if acc == "hailo8":
            return (not has_hailo10) and (has_hailo8 or '"hailo"' in text)
        return bool(acc and acc in text)

    def _configured_hardware_setups_with_payloads(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        try:
            reg = self._hardware_registry_load()
            for raw in reg.get("hardware_setups") or []:
                if not isinstance(raw, Mapping):
                    continue
                sid = str(raw.get("id") or "").strip()
                if not sid:
                    continue
                payload = self._hardware_setup_remote_payload(sid)
                if not payload or not str(payload.get("host") or "").strip():
                    continue
                item = dict(raw)
                item["setup_id"] = sid
                item["payload"] = payload
                item["accelerator"] = canon_accelerator(payload.get("accelerator") or raw.get("accelerator") or raw.get("backend") or raw.get("provider") or sid)
                out.append(item)
        except Exception:
            logger.debug("Failed to enumerate configured hardware setup payloads", exc_info=True)
        return out

    def _auto_remote_dispatch_plan(self, benchmark_set_json: Path) -> list[dict]:
        """Return per-hardware remote dispatches for a suite.

        v52o: no single "selected remote" is required anymore.  The generated
        benchmark_plan.json is authoritative: Hailo rows go to the configured
        Hailo NX, DeepX rows go to the configured DeepX NX.  Plain ORT/CUDA/TRT
        reference rows are repeated on each relevant NX so comparisons are
        host-local and the user does not need a separate manual choice.
        """
        runs = self._benchmark_plan_runs_for_suite(benchmark_set_json)
        setups = self._configured_hardware_setups_with_payloads()
        dispatch: dict[str, dict] = {}
        if runs:
            row_pairs = [(self._benchmark_run_id(r), r) for r in runs if self._benchmark_run_id(r)]
            plain_ref_ids = [rid for rid, row in row_pairs if self._row_is_plain_host_reference(row)]
            for setup in setups:
                setup_id = str(setup.get("setup_id") or "").strip()
                acc = str(setup.get("accelerator") or "").strip()
                explicit_ids: list[str] = []
                accel_ids: list[str] = []
                for rid, row in row_pairs:
                    explicit = str(row.get("hardware_setup_id") or row.get("hardware_setup") or row.get("target_setup") or "").strip()
                    if explicit and explicit == setup_id:
                        explicit_ids.append(rid)
                    elif self._row_mentions_accelerator(row, acc):
                        accel_ids.append(rid)
                if explicit_ids or accel_ids:
                    run_ids: list[str] = []
                    for rid in plain_ref_ids + explicit_ids + accel_ids:
                        if rid and rid not in run_ids:
                            run_ids.append(rid)
                    dispatch[setup_id] = {"setup_id": setup_id, "payload": setup.get("payload") or {}, "run_ids": run_ids}
            if not dispatch:
                # Pure ORT/CUDA/TRT benchmark set. Prefer DeepX/TensorRT-validated
                # setup if present, otherwise use the first configured NX.
                chosen = None
                for wanted in ("deepx_m1", "hailo8", "hailo10"):
                    for setup in setups:
                        if str(setup.get("accelerator") or "") == wanted:
                            chosen = setup
                            break
                    if chosen:
                        break
                if chosen is None and setups:
                    chosen = setups[0]
                if chosen is not None:
                    setup_id = str(chosen.get("setup_id") or "")
                    dispatch[setup_id] = {"setup_id": setup_id, "payload": chosen.get("payload") or {}, "run_ids": [rid for rid, _row in row_pairs] or [""]}
        else:
            # No plan: legacy suite. Use first configured setup as fallback.
            if setups:
                setup = setups[0]
                setup_id = str(setup.get("setup_id") or "")
                dispatch[setup_id] = {"setup_id": setup_id, "payload": setup.get("payload") or {}, "run_ids": []}
        return list(dispatch.values())

    def _benchmark_auto_dispatch_summary(self, benchmark_set_json: str | Path | None = None) -> str:
        try:
            p = Path(str(benchmark_set_json or (self.var_remote_benchmark_set.get() if hasattr(self, "var_remote_benchmark_set") else ""))).expanduser()
            if p.is_dir():
                p = p / "benchmark_set.json"
            dispatches = self._auto_remote_dispatch_plan(p) if p.exists() else []
            if not dispatches:
                return "Auto dispatch: no matching configured hardware setup found yet. Configure hosts in Tool Config."
            runs = self._benchmark_plan_runs_for_suite(p) if p.exists() else []
            plan_ids = [self._benchmark_run_id(r) for r in runs if self._benchmark_run_id(r)]
            requested_ids = self._benchmark_profile_requested_run_ids_for_suite(p) if p.exists() else []
            sources = []
            for r in runs:
                src = str(r.get("_plan_source") or "").strip()
                if src and src not in sources:
                    sources.append(src)
            title = "Auto dispatch from frozen benchmark plan"
            if sources:
                title += f" ({', '.join(sources)})"
            lines = [title + ":"]
            if requested_ids and set(requested_ids) != set(plan_ids):
                missing = [x for x in requested_ids if x not in plan_ids]
                extra = [x for x in plan_ids if x not in requested_ids]
                lines.append("  note: profile requested run_ids differ from the materialized benchmark_plan.")
                if missing:
                    lines.append("  requested but NOT in this benchmarkset: " + ", ".join(missing))
                if extra:
                    lines.append("  materialized additionally: " + ", ".join(extra))
                lines.append("  To run the requested targets, regenerate/resume the benchmark set with the current target selection.")
            for d in dispatches:
                payload = d.get("payload") or {}
                runs = [str(x) for x in (d.get("run_ids") or []) if str(x).strip()]
                run_txt = ", ".join(runs) if runs else "all/legacy"
                setup_id = str(d.get("setup_id") or "")
                label = str(payload.get("label") or payload.get("accelerator") or setup_id)
                remote = f"{payload.get('user') or ''}@{payload.get('host') or ''}:{payload.get('port') or 22}"
                lines.append(f"• {setup_id} — {label}")
                lines.append(f"  remote: {remote}")
                lines.append(f"  runs:   {run_txt}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Auto dispatch summary unavailable: {type(exc).__name__}"


    def _test_hardware_setup_remote(self, setup_id: str) -> tuple[bool, str]:
        """Run a focused SSH/runtime smoke test for one central hardware setup."""
        payload = self._hardware_setup_remote_payload(setup_id)
        if not payload or not str(payload.get("host") or "").strip():
            out = "Hardware setup has no remote host configured."
            try:
                self._popup_text("Hardware setup test", out)
            except Exception:
                pass
            return False, out
        host = self._host_config_from_hardware_payload(payload)
        venv = str(payload.get("remote_venv") or "").strip()
        provider = str(payload.get("provider") or payload.get("accelerator") or "auto").strip().lower()
        acc = canon_accelerator(payload.get("accelerator") or provider)
        prefix = ""
        if venv:
            prefix = venv if venv.startswith(("source ", ". ")) else ("source " + venv)
            prefix += " >/dev/null 2>&1 || true; "
        common = (
            "set -o pipefail; "
            "echo '[host]' $(hostname) $(uname -m); "
            "echo '[python]' $(command -v python3 || true); "
            f"{prefix}"
            "python - <<'PY'\n"
            "import sys\n"
            "print('python:', sys.executable)\n"
            "try:\n import onnxruntime as ort; print('onnxruntime:', ort.__version__, ort.get_available_providers())\nexcept Exception as e: print('onnxruntime: FAILED', repr(e))\n"
        )
        if acc == "deepx_m1":
            body = common + (
                "try:\n from dx_engine import InferenceEngine; print('dx_engine: OK')\nexcept Exception as e: print('dx_engine: FAILED', repr(e))\n"
                "try:\n import tensorrt as trt; print('tensorrt:', trt.__version__)\nexcept Exception as e: print('tensorrt: FAILED', repr(e))\n"
                "PY\n"
                "echo '[dxrt-cli]' $(command -v dxrt-cli || true); dxrt-cli -s 2>/dev/null | head -30 || true; "
                "echo '[run_model]' $(command -v run_model || true); "
                "echo '[parse_model]' $(command -v parse_model || true); "
                "echo '[trtexec]' $(command -v trtexec || true)"
            )
        elif acc.startswith("hailo"):
            body = common + (
                "try:\n import hailo_platform as hp; print('hailo_platform: OK', getattr(hp, '__version__', ''))\nexcept Exception as e1:\n"
                " try:\n  import hailort as hr; print('hailort: OK', getattr(hr, '__version__', ''))\n except Exception as e2: print('hailo import: FAILED', repr(e1), repr(e2))\n"
                "PY\n"
                "echo '[hailo devices]'; ls -l /dev/hailo* 2>/dev/null || true; "
                "echo '[hailortcli]' $(command -v hailortcli || true); hailortcli scan 2>/dev/null || true"
            )
        else:
            body = common + "PY\n"
        transport = SSHTransport(host)
        rc, out = transport.run(body, timeout=45)
        ok = (rc == 0)
        title = f"Hardware setup test — {'OK' if ok else 'FAILED'}"
        try:
            logger.info("[hardware-setup-test] %s rc=%s\n%s", setup_id, rc, out)
            self._popup_text(title, out, width=120, height=32)
        except Exception:
            pass
        return ok, out

    def _remote_test_connection(self):
        # Resolve the central Tool Config hardware setup before constructing
        # the legacy HostConfig. This keeps Benchmark runs in sync with the
        # Evaluation Workflow hardware matrix and avoids duplicate host/venv
        # configuration fields.
        try:
            setup_id = str(getattr(self, "var_remote_hardware_setup_id", tk.StringVar(value="")).get() or "").strip()
            if setup_id:
                self._apply_hardware_setup_to_remote_vars(setup_id, persist=False)
        except Exception:
            logger.debug("Failed to apply central hardware setup before remote benchmark", exc_info=True)

        host = self._remote_get_selected_host()
        if host is None:
            messagebox.showwarning("Remote benchmark", "Please select a hardware setup in Tool Config / Benchmark first.")
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
        validation_images = None
        validation_max_images = None
        validation_reference_mode = None
        benchmark_task = 'auto'
        mini_coco_ap50 = False
        mini_classification_eval = False
        try:
            validation_images = (getattr(self, 'var_bench_validation_images', None).get() if hasattr(self, 'var_bench_validation_images') else '') or ''
            validation_images = str(validation_images).strip() or None
        except Exception:
            validation_images = None
        try:
            raw_max = (getattr(self, 'var_bench_validation_max_images', None).get() if hasattr(self, 'var_bench_validation_max_images') else '') or ''
            validation_max_images = int(str(raw_max).strip()) if str(raw_max).strip() else 0
        except Exception:
            validation_max_images = 0
        try:
            validation_reference_mode = (getattr(self, 'var_bench_validation_reference_mode', None).get() if hasattr(self, 'var_bench_validation_reference_mode') else 'auto') or 'auto'
            validation_reference_mode = str(validation_reference_mode).strip() or 'auto'
        except Exception:
            validation_reference_mode = 'auto'
        try:
            benchmark_task = (getattr(self, 'var_bench_task', None).get() if hasattr(self, 'var_bench_task') else 'auto') or 'auto'
            benchmark_task = str(benchmark_task).strip() or 'auto'
        except Exception:
            benchmark_task = 'auto'
        try:
            mini_coco_ap50 = bool((getattr(self, 'var_bench_mini_coco_ap50', None).get() if hasattr(self, 'var_bench_mini_coco_ap50') else False))
        except Exception:
            mini_coco_ap50 = False
        try:
            mini_classification_eval = bool((getattr(self, 'var_bench_mini_classification_eval', None).get() if hasattr(self, 'var_bench_mini_classification_eval') else False))
        except Exception:
            mini_classification_eval = False
        try:
            stats = self._remote_service.refresh_suite_harness(
                suite_dir,
                benchmark_set_json=bench_json,
                validation_images=validation_images,
                validation_max_images=validation_max_images,
                validation_reference_mode=validation_reference_mode,
                mini_coco_ap50=mini_coco_ap50,
                benchmark_task=benchmark_task,
                mini_classification_eval=mini_classification_eval,
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
            f"Validation defaults normalized: {'yes' if stats.get('validation_changed') else 'no'}",
            f"Validation images: {stats.get('validation_images') or '(unchanged)'}",
            f"Validation max images: {stats.get('validation_max_images')}",
            f"Validation reference mode: {stats.get('validation_reference_mode')}",
            f"Benchmark task: {stats.get('benchmark_task')}",
            f"Mini-COCO AP50: {'enabled' if stats.get('mini_coco_ap50') else 'disabled'}",
            f"Mini-Classification eval: {'enabled' if stats.get('mini_classification_eval') else 'disabled'}",
            f"Validation resource provisioned: {'yes' if stats.get('validation_resource_provisioned') else 'no'}",
            f"Patched files: {', '.join(stats.get('validation_patched_files') or []) or '-'}",
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

    def _canon_remote_provider_token(self, value: object) -> str:
        s = str(value or "").strip().lower().replace("-", "_")
        if s in {"dx_m1", "dxm1", "deepx", "deepx_dx_m1"}:
            return "deepx_m1"
        if s in {"trt", "ort_trt", "ort_tensorrt", "tensorrt"}:
            return "tensorrt"
        if s in {"cuda", "cuda_ort", "ort_cuda"}:
            return "cuda"
        if s in {"cpu", "cpu_ort", "ort_cpu"}:
            return "cpu"
        if "hailo10" in s:
            return "hailo10"
        if "hailo8" in s:
            return "hailo8"
        if s == "hailo":
            return "hailo8"
        return s

    def _benchmark_plan_for_suite(self, benchmark_set_json: Path) -> dict:
        bench = {}
        try:
            bench = json.loads(Path(benchmark_set_json).read_text(encoding="utf-8"))
            if not isinstance(bench, Mapping):
                bench = {}
        except Exception:
            bench = {}
        plan = bench.get("plan") if isinstance(bench.get("plan"), Mapping) else None
        if plan:
            return dict(plan)
        plan_path = Path(benchmark_set_json).parent / "benchmark_plan.json"
        if plan_path.exists():
            try:
                p = json.loads(plan_path.read_text(encoding="utf-8"))
                if isinstance(p, Mapping):
                    return dict(p)
            except Exception:
                pass
        return {}

    def _plan_run_entries_for_suite(self, benchmark_set_json: Path) -> list[dict]:
        plan = self._benchmark_plan_for_suite(benchmark_set_json)
        for key in ("runs", "planned_runs", "run_profiles", "matrix_runs"):
            rows = plan.get(key)
            if isinstance(rows, list):
                return [dict(x) for x in rows if isinstance(x, Mapping)]
        return []

    def _run_id_from_plan_row(self, row: Mapping[str, Any]) -> str:
        return str(row.get("id") or row.get("name") or row.get("run_id") or row.get("backend") or row.get("type") or "").strip()

    def _plan_row_mentions_accelerator(self, row: Mapping[str, Any], accelerator: str) -> bool:
        acc = self._canon_remote_provider_token(accelerator)
        text = json.dumps(dict(row or {}), ensure_ascii=False).lower()
        if acc == "deepx_m1":
            return "deepx" in text or "dx_m1" in text or "dxm1" in text
        if acc == "hailo10":
            return "hailo10" in text
        if acc == "hailo8":
            return "hailo8" in text or '"hailo"' in text
        if acc == "tensorrt":
            return "tensorrt" in text or "trt" in text
        return bool(acc and acc in text)

    def _setup_id_for_accelerator(self, accelerator: str) -> str:
        acc = self._canon_remote_provider_token(accelerator)
        reg = self._hardware_registry_load()
        preferred = {
            "hailo8": "orin_nx_hailo8_01",
            "hailo10": "orin_nx_hailo10_01",
            "deepx_m1": "orin_nx_deepx_m1_01",
        }.get(acc, "")
        if preferred and self._hardware_setup_remote_payload(preferred).get("host"):
            return preferred
        for raw in reg.get("hardware_setups") or []:
            if not isinstance(raw, Mapping):
                continue
            sid = str(raw.get("id") or "").strip()
            raw_acc = self._canon_remote_provider_token(raw.get("accelerator") or raw.get("backend") or raw.get("provider"))
            if raw_acc == acc and self._hardware_setup_remote_payload(sid).get("host"):
                return sid
        return preferred

    def _configured_setup_ids(self) -> list[str]:
        reg = self._hardware_registry_load()
        ids: list[str] = []
        for raw in reg.get("hardware_setups") or []:
            if isinstance(raw, Mapping):
                sid = str(raw.get("id") or "").strip()
                if sid and self._hardware_setup_remote_payload(sid).get("host"):
                    ids.append(sid)
        return ids

    def _default_reference_setup_id(self, active_accelerators: set[str] | None = None) -> str:
        active_accelerators = active_accelerators or set()
        # If a suite contains exactly one accelerator family, run pure ORT/TRT
        # reference rows on the same NX.  If multiple families are present and
        # the plan did not specify hardware_setup_id, duplicate reference rows on
        # all active setups (handled by caller).
        if len(active_accelerators) == 1:
            return self._setup_id_for_accelerator(next(iter(active_accelerators)))
        for acc in ("deepx_m1", "hailo8", "hailo10"):
            sid = self._setup_id_for_accelerator(acc)
            if sid and self._hardware_setup_remote_payload(sid).get("host"):
                return sid
        ids = self._configured_setup_ids()
        return ids[0] if ids else ""

    def _benchmark_auto_dispatches(self, benchmark_set_json: Path) -> list[dict]:
        """Return per-run remote dispatches derived from benchmark_plan.json.

        Each dispatch executes exactly one run id on the hardware setup that owns
        that backend.  This is the central behavior the GUI should use: the user
        configures the three NX setups once in Tool Config, and generated run
        plans decide which setup each row runs on.
        """
        rows = self._plan_run_entries_for_suite(benchmark_set_json)
        if not rows:
            sid = self._default_reference_setup_id(set())
            return [{"setup_id": sid, "run_id": "", "reason": "no_plan_fallback"}] if sid else []

        # Determine accelerator families present in this plan.
        active_accs: set[str] = set()
        for row in rows:
            for acc in ("hailo8", "hailo10", "deepx_m1"):
                if self._plan_row_mentions_accelerator(row, acc):
                    active_accs.add(acc)

        dispatches: list[dict] = []
        for row in rows:
            rid = self._run_id_from_plan_row(row)
            if not rid:
                continue
            explicit = str(row.get("hardware_setup_id") or row.get("hardware_setup") or row.get("target_setup") or "").strip()
            setup_ids: list[str] = []
            if explicit:
                setup_ids = [explicit]
            else:
                for acc in ("deepx_m1", "hailo10", "hailo8"):
                    if self._plan_row_mentions_accelerator(row, acc):
                        sid = self._setup_id_for_accelerator(acc)
                        if sid:
                            setup_ids.append(sid)
                if not setup_ids:
                    # Pure CPU/CUDA/TensorRT reference rows.  With one active
                    # accelerator, run on that NX.  With multiple accelerators,
                    # duplicate the reference row on each active NX unless the
                    # profile already pinned the row with hardware_setup_id.
                    if len(active_accs) > 1:
                        for acc in sorted(active_accs):
                            sid = self._setup_id_for_accelerator(acc)
                            if sid:
                                setup_ids.append(sid)
                    else:
                        sid = self._default_reference_setup_id(active_accs)
                        if sid:
                            setup_ids.append(sid)
            # Keep order and de-dupe.
            seen: set[str] = set()
            for sid in setup_ids:
                if not sid or sid in seen:
                    continue
                seen.add(sid)
                if not self._hardware_setup_remote_payload(sid).get("host"):
                    dispatches.append({"setup_id": sid, "run_id": rid, "status": "missing_host"})
                else:
                    dispatches.append({"setup_id": sid, "run_id": rid, "status": "ready"})
        return dispatches

    def _collect_remote_benchmark_args_from_ui(self) -> RemoteBenchmarkArgs:
        warmup = self._parse_remote_int(self.var_remote_warmup.get() if hasattr(self, "var_remote_warmup") else "10", default=10, label="Remote warmup", minimum=0)
        iters = self._parse_remote_int(self.var_remote_iters.get() if hasattr(self, "var_remote_iters") else "100", default=100, label="Remote runs", minimum=1)
        repeats = self._parse_remote_int(self.var_remote_repeats.get() if hasattr(self, "var_remote_repeats") else "1", default=1, label="Remote repeats", minimum=1)
        throughput_frames = self._parse_remote_int(self.var_remote_throughput_frames.get() if hasattr(self, "var_remote_throughput_frames") else "24", default=24, label="Streaming frames", minimum=0)
        throughput_warmup_frames = self._parse_remote_int(self.var_remote_throughput_warmup_frames.get() if hasattr(self, "var_remote_throughput_warmup_frames") else "6", default=6, label="Streaming warmup frames", minimum=0)
        throughput_queue_depth = self._parse_remote_int(self.var_remote_throughput_queue_depth.get() if hasattr(self, "var_remote_throughput_queue_depth") else "2", default=2, label="Streaming queue depth", minimum=1)
        timeout_s = self._parse_remote_outer_timeout()
        return RemoteBenchmarkArgs(
            provider=self.var_remote_provider.get() if hasattr(self, "var_remote_provider") else "auto",
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            timeout_s=timeout_s,
            throughput_frames=throughput_frames,
            throughput_warmup_frames=throughput_warmup_frames,
            throughput_queue_depth=throughput_queue_depth,
            validation_images=(self.var_bench_validation_images.get() if hasattr(self, "var_bench_validation_images") else ""),
            validation_max_images=int(((self.var_bench_validation_max_images.get() if hasattr(self, "var_bench_validation_max_images") else "50") or "50")),
            validation_reference_mode=(self.var_bench_validation_reference_mode.get() if hasattr(self, "var_bench_validation_reference_mode") else "auto"),
            mini_coco_ap50=(bool(self.var_bench_mini_coco_ap50.get()) if hasattr(self, "var_bench_mini_coco_ap50") else False),
            benchmark_task=(self.var_bench_task.get() if hasattr(self, "var_bench_task") else "auto"),
            mini_classification_eval=(bool(self.var_bench_mini_classification_eval.get()) if hasattr(self, "var_bench_mini_classification_eval") else False),
            add_args=self.var_remote_add_args.get() if hasattr(self, "var_remote_add_args") else "",
            remote_venv=self.var_remote_venv.get() if hasattr(self, "var_remote_venv") else "",
            transfer_mode=self.var_remote_transfer_mode.get() if hasattr(self, "var_remote_transfer_mode") else "bundle",
            reuse_bundle=bool(self.var_remote_reuse_bundle.get()) if hasattr(self, "var_remote_reuse_bundle") else True,
            energy_enabled=bool(self.var_remote_measure_energy.get()) if hasattr(self, "var_remote_measure_energy") else False,
            energy_registry_path=str(self._hardware_setups_path()),
            energy_run_count=self._parse_remote_int(self.var_remote_energy_runs.get() if hasattr(self, "var_remote_energy_runs") and str(self.var_remote_energy_runs.get()).strip() else "0", default=0, label="Energy runs", minimum=0),
        )

    def _remote_result_group_name(self, run_id: object) -> str:
        raw = str(run_id or "").strip() or time.strftime("%Y%m%d_%H%M%S")
        safe = "".join((c if (c.isalnum() or c in ("-", "_", ".")) else "_") for c in raw).strip("._-")
        return "remote_" + (safe or time.strftime("%Y%m%d_%H%M%S"))

    def _cleanup_primary_remote_run_dir_after_energy(self, *, host, primary_out: dict[str, Any] | None, log) -> dict[str, Any]:
        """Best-effort cleanup of a deployed remote suite after energy windows finish.

        Energy measurement needs the primary remote_run_dir to stay alive while
        row/phase windows execute.  After all windows and local downloads/merges
        are finished, remove it from the NX to avoid filling ~/splitpoint_runs.
        """
        remote_dir = str((primary_out or {}).get("remote_run_dir") or "").strip()
        payload: dict[str, Any] = {
            "schema": "onnx-splitpoint/remote-cleanup",
            "schema_version": 1,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "remote_run_dir": remote_dir,
            "attempted": False,
            "ok": False,
            "reason": "",
        }
        if not remote_dir:
            payload["reason"] = "primary_remote_run_dir_missing"
            return payload
        if "/splitpoint_runs/" not in remote_dir:
            payload["reason"] = "refuse_unsafe_path"
            try:
                log(f"[cleanup][warn] refusing unsafe remote cleanup path: {remote_dir}")
            except Exception:
                pass
            return payload
        script = (
            "set -u\n"
            f"REMOTE_DIR={shlex.quote(remote_dir)}\n"
            "case \"$REMOTE_DIR\" in *'/splitpoint_runs/'*) ;; *) echo '[cleanup] unsafe path:' \"$REMOTE_DIR\"; exit 2;; esac\n"
            "if [ -d \"$REMOTE_DIR\" ]; then\n"
            "  echo '[cleanup] before:'; du -sh \"$REMOTE_DIR\" 2>/dev/null || true\n"
            "  rm -rf -- \"$REMOTE_DIR\"\n"
            "  if [ -e \"$REMOTE_DIR\" ]; then echo '[cleanup] path still exists'; exit 3; fi\n"
            "  echo '[cleanup] removed:' \"$REMOTE_DIR\"\n"
            "else\n"
            "  echo '[cleanup] already absent:' \"$REMOTE_DIR\"\n"
            "fi\n"
        )
        try:
            payload["attempted"] = True
            rc, out = SSHTransport(host).run("bash -lc " + shlex.quote(script), timeout=180)
            payload.update({"rc": rc, "ok": rc == 0, "output_tail": str(out or "")[-12000:]})
            if rc == 0:
                log(f"[cleanup] removed primary remote run dir after energy: {remote_dir}")
            else:
                log(f"[cleanup][warn] failed to remove primary remote run dir rc={rc}: {str(out or '')[-1000:]}")
        except Exception as exc:
            payload.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            try:
                log(f"[cleanup][warn] primary remote cleanup failed: {type(exc).__name__}: {exc}")
            except Exception:
                pass
        return payload


    def _remote_energy_enabled(self) -> bool:
        try:
            return bool(getattr(self, "var_remote_measure_energy", tk.BooleanVar(value=False)).get())
        except Exception:
            return False

    def _remote_energy_runs(self, default: int = 1) -> int:
        try:
            raw = str(getattr(self, "var_remote_energy_runs", tk.StringVar(value=str(default))).get() or "").strip()
            return max(1, int(raw or str(default)))
        except Exception:
            return max(1, int(default or 1))

    def _energy_rows_from_result_obj(self, obj: Any) -> list[dict[str, Any]]:
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            for key in ("rows", "results", "measurements", "records"):
                val = obj.get(key)
                if isinstance(val, list):
                    return [r for r in val if isinstance(r, dict)]
            return [obj]
        return []

    def _energy_collect_row_targets(self, primary_local_run_dir: str | Path, run_id: str) -> list[dict[str, Any]]:
        """Return row-level u.RECS measurement targets from primary benchmark rows.

        Energy should be measured for comparable result rows, not for a whole
        plan run that may contain multiple cases and variants.  For reference
        runs (e.g. ort_tensorrt) we measure the full variant per case.  For
        heterogeneous split runs we measure the composed variant per case.  Full
        accelerator runs keep their full variant.
        """
        base = Path(primary_local_run_dir).expanduser()
        if not base.exists():
            return []
        run_id_l = str(run_id or "").strip().lower()
        targets: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        files = sorted(base.rglob("benchmark_results*.json"))[:200]
        for fp in files:
            stem_l = fp.stem.lower()
            if run_id_l and run_id_l not in stem_l:
                # Keep broad fallback for odd result filenames, but avoid mixing
                # unrelated run ids when the stem already identifies the run.
                continue
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            for row in self._energy_rows_from_result_obj(obj):
                row_run = str(row.get("run_id") or row.get("backend") or row.get("provider") or run_id or "").lower()
                if run_id_l and run_id_l not in row_run and run_id_l not in stem_l:
                    continue
                if row.get("counts_as_split_benchmark") is False:
                    continue
                err = str(row.get("error_class") or row.get("skip_reason") or "").lower()
                if err and err not in {"none", "null"}:
                    # Do not spend u.RECS windows on known missing/unsupported rows.
                    continue
                status = str(row.get("deepx_stage2_contract_status") or row.get("stage2_contract_status") or "").lower()
                if status in {"all_candidates_rejected", "contract_rejected", "requires_feature_tensor_calibration", "native_unstable", "deepx_stage2_native_unstable"}:
                    continue
                case_id = str(row.get("case_id") or row.get("case") or "").strip()
                if not case_id:
                    b = row.get("boundary")
                    if b is not None:
                        try:
                            case_id = f"b{int(b):03d}"
                        except Exception:
                            case_id = str(b).strip()
                if not case_id:
                    case_id = "full"
                variant_hint = str(row.get("variant") or row.get("primary_variant") or "").strip().lower()
                rid = run_id_l or row_run
                if "_to_" in rid or "to_trt" in rid or "trt_to" in rid:
                    variant = "composed"
                elif "full" in rid or variant_hint == "full" or case_id.lower() == "full":
                    variant = "full"
                elif row.get("full_mean_ms") is not None and ("ort_tensorrt" in rid or "tensorrt" in rid or "cuda" in rid):
                    # Reference rows contain full + composed diagnostics.  For
                    # baseline energy we measure the full variant, not full+parts.
                    variant = "full"
                elif variant_hint in {"full", "composed", "part1", "part2"}:
                    variant = variant_hint
                else:
                    variant = "composed"
                # v57d: full baselines are canonical per run/backend, not per
                # split case.  A TensorRT/CUDA/DeepX full model is the same full
                # graph for b059, b054, ...; measuring it for every split only
                # wastes u.RECS windows and makes the dispatch totals misleading.
                # Keep the first concrete case as the executable target, but mark
                # the measurement as applicable to all full rows of this run_id.
                is_split_run = ("_to_" in rid or "to_trt" in rid or "trt_to" in rid)
                canonical_full = (variant == "full" and not is_split_run)
                key = (("__canonical_full__" if canonical_full else case_id), variant)
                if key in seen:
                    continue
                seen.add(key)
                # v57e: energy target throughput must match the measured target
                # variant.  The benchmark result row can contain same-backend
                # composed pipeline fields even when we are measuring the full
                # baseline.  For full-energy targets use the full-model latency
                # FPS, not row.pipeline_fps_selected from composed diagnostics.
                target_fps = row.get("pipeline_fps_selected")
                target_fps_source = "row_pipeline_fps_selected"
                if variant == "full":
                    try:
                        full_ms = float(row.get("full_mean_ms") or row.get("full_e2e_mean_ms") or row.get("total_latency_ms") or 0.0)
                    except Exception:
                        full_ms = 0.0
                    if full_ms > 0:
                        target_fps = 1000.0 / full_ms
                        target_fps_source = "full_latency_fps"
                targets.append({
                    "case_id": case_id,
                    "variant": variant,
                    "source_file": str(fp),
                    "run_id": str(run_id or row.get("run_id") or ""),
                    "final_pass": row.get("final_pass"),
                    "pipeline_fps_selected": target_fps,
                    "energy_reference_fps": target_fps,
                    "energy_reference_fps_source": target_fps_source,
                    "full_mean_ms": row.get("full_mean_ms"),
                    "full_e2e_mean_ms": row.get("full_e2e_mean_ms"),
                    "total_latency_ms": row.get("total_latency_ms"),
                    "part1_mean_ms": row.get("part1_mean_ms"),
                    "part2_mean_ms": row.get("part2_mean_ms"),
                    "composed_mean_ms": row.get("composed_mean_ms"),
                    "split_latency_e2e_ms": row.get("split_latency_e2e_ms"),
                    "canonical_full_baseline": bool(canonical_full),
                    "applies_to_all_cases": bool(canonical_full),
                    "canonical_full_target_case": case_id if canonical_full else None,
                })
        if not targets and run_id_l:
            # v57a: if the primary benchmark rows failed before reporting, build
            # row-level energy targets from benchmark_set.json instead of falling
            # back to a synthetic full/full target.  Synthetic full/full is only
            # valid for true full-only run ids; split/reference run ids still have
            # concrete case folders (b059, b054, ...).  The old fallback caused
            # --energy-target-case full to skip every real case and fail the
            # duration probe.
            try:
                bs_files = [base / "results" / "benchmark_set.json", base / "benchmark_set.json"]
                bs_obj = None
                for bs in bs_files:
                    if bs.exists():
                        bs_obj = json.loads(bs.read_text(encoding="utf-8"))
                        break
                cases = []
                if isinstance(bs_obj, dict):
                    for c in bs_obj.get("cases") or []:
                        if not isinstance(c, dict):
                            continue
                        cid = str(c.get("case_dir") or c.get("folder") or "").strip()
                        if not cid and c.get("boundary") is not None:
                            try:
                                cid = f"b{int(c.get('boundary')):03d}"
                            except Exception:
                                cid = str(c.get("boundary") or "").strip()
                        if cid:
                            cases.append(cid)
                if cases:
                    rid = run_id_l
                    if "_to_" in rid or "to_trt" in rid or "trt_to" in rid:
                        variant = "composed"
                    elif "full" in rid:
                        variant = "full"
                    elif "ort_tensorrt" in rid or "tensorrt" in rid or "cuda" in rid:
                        variant = "full"
                    else:
                        variant = "composed"
                    for cid in cases:
                        key = (cid, variant)
                        if key not in seen:
                            seen.add(key)
                            targets.append({"case_id": cid, "variant": variant, "run_id": str(run_id or ""), "source": "benchmark_set_fallback"})
            except Exception:
                pass
        if not targets and run_id_l:
            # Last-resort full-run fallback for genuine full-only runs.
            if "full" in run_id_l and not ("_to_" in run_id_l or "to_trt" in run_id_l or "trt_to" in run_id_l):
                targets.append({"case_id": "full", "variant": "full", "run_id": str(run_id or ""), "source": "full_run_fallback"})
        return targets

    def _remote_benchmark_cli_command(self, *, host, benchmark_set_json: Path, local_working_dir: Path, run_id: str, args: RemoteBenchmarkArgs, results_group_id: str | None = None) -> list[str]:
        """Build a CLI command equivalent to RemoteBenchmarkService.run.

        This is used by the u.RECS energy wrapper: the Rust collector can run a
        shell command, but it cannot call our in-process Python service.  The CLI
        path keeps the measured command reproducible and records stdout/stderr in
        the energy artifact folder.
        """
        cmd = [
            sys.executable,
            "-m",
            "onnx_splitpoint_tool.cli",
            "benchmark-remote",
            str(Path(benchmark_set_json).expanduser().resolve()),
            "--host",
            str(host.user_host),
            "--port",
            str(int(getattr(host, "port", 22) or 22)),
            "--remote-base-dir",
            str(getattr(host, "remote_base_dir", "~/splitpoint_runs") or "~/splitpoint_runs"),
            "--working-dir",
            str(Path(local_working_dir).expanduser().resolve()),
            "--run-id",
            str(run_id),
            "--provider",
            str(getattr(args, "provider", "auto") or "auto"),
            "--repeats",
            str(int(getattr(args, "repeats", 1) or 1)),
            "--warmup",
            str(int(getattr(args, "warmup", 0) or 0)),
            "--iters",
            str(int(getattr(args, "iters", 1) or 1)),
            "--timeout-s",
            str(int(getattr(args, "timeout_s", 0) or 0)),
            "--transfer-mode",
            str(getattr(args, "transfer_mode", "bundle") or "bundle"),
            "--throughput-frames",
            str(int(getattr(args, "throughput_frames", 0) or 0)),
            "--throughput-warmup-frames",
            str(int(getattr(args, "throughput_warmup_frames", 0) or 0)),
            "--throughput-queue-depth",
            str(max(1, int(getattr(args, "throughput_queue_depth", 1) or 1))),
        ]
        user = str(getattr(host, "user", "") or "").strip()
        if user:
            cmd += ["--user", user]
        ssh_extra = str(getattr(host, "ssh_extra_args", "") or "").strip()
        if ssh_extra:
            cmd += ["--ssh-extra-args", ssh_extra]
        remote_venv = str(getattr(args, "remote_venv", "") or "").strip()
        if remote_venv:
            cmd += ["--remote-venv", remote_venv]
        add_args = str(getattr(args, "add_args", "") or "").strip()
        if add_args:
            cmd += ["--add-args", add_args]
        if not bool(getattr(args, "reuse_bundle", True)):
            cmd += ["--no-reuse-bundle"]
        if not bool(getattr(args, "resume", True)):
            cmd += ["--no-resume"]
        if results_group_id:
            cmd += ["--results-group-id", str(results_group_id)]
        return cmd

    def _run_remote_dispatch_with_energy(self, *, setup_id: str, run_id: str, host, benchmark_set_json: Path, local_working_dir: Path, child_run_id: str, args: RemoteBenchmarkArgs, results_group_id: str, log, primary_out: dict[str, Any] | None = None, energy_target_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one remote dispatch wrapped by u.RECS fast-firmware measurements.

        v56i: energy follows the benchmark-tab semantics instead of wrapping the
        whole command once.  For each selected run-id we measure separate windows:

          * latency phase: one u.RECS window per benchmark repeat, each window
            runs exactly ``iters`` measured benchmark iterations and disables the
            streaming probe.
          * streaming phase: one u.RECS window per benchmark repeat, each window
            runs the configured streaming/interleaving frame probe and keeps the
            normal latency part as a minimal one-run sanity pass because older
            generated suite runners do not yet expose a pure streaming-only CLI.

        This means Repeats=2, Runs=100, Streaming frames=48 produces four u.RECS
        windows per run-id: latency_r000, latency_r001, streaming_r000,
        streaming_r001.  It never creates one u.RECS measurement per inference.
        """
        # Re-admit after the primary deployment as well.  This catches a
        # registry or installed-source change between the pre-primary gate and
        # the first collector window, before creating an energy output folder.
        admission = preflight_remote_energy_dispatch(
            host=host,
            args=args,
            registry_path=self._hardware_setups_path(),
        )
        from onnx_splitpoint_tool.energy.config import energy_defaults_from_registry
        from onnx_splitpoint_tool.energy.collector import (
            run_fast_firmware_measurement,
            select_host_normalization_role,
        )

        reg = admission["registry"]
        defaults = energy_defaults_from_registry(reg)
        setup = admission["setup"]
        if not setup.enabled:
            return {"status": "energy_disabled", "ok": False, "reason": f"energy disabled for setup {setup_id}"}
        if not setup.urecs_address:
            return {"status": "energy_unconfigured", "ok": False, "reason": f"u.RECS address missing for setup {setup_id}"}

        benchmark_repeats = max(1, int(getattr(args, "repeats", 1) or 1))
        override_repeats = int(getattr(args, "energy_run_count", 0) or 0)
        phase_repeats = max(1, override_repeats or benchmark_repeats)
        if override_repeats:
            repeat_source = "energy_override"
        else:
            repeat_source = "benchmark_repeats"

        group_dir = Path(local_working_dir).expanduser().resolve() / "Results" / Path(benchmark_set_json).parent.name / results_group_id
        energy_root = group_dir / str(child_run_id) / "energy"
        energy_root.mkdir(parents=True, exist_ok=True)

        # v57r: mark the primary remote run metadata as energy-augmented.  The
        # primary benchmark is executed without energy_enabled internally so the
        # suite can be deployed and normal results can be produced before the
        # row-scoped u.RECS windows.  Without this post-hoc metadata update,
        # run_meta.json misleadingly said energy_enabled=false even though energy
        # artifacts and merged u.RECS fields were later attached to the same run.
        try:
            _plrd = str((primary_out or {}).get("local_run_dir") or "").strip()
            if _plrd:
                _meta_p = Path(_plrd) / "run_meta.json"
                if _meta_p.exists():
                    _meta = json.loads(_meta_p.read_text(encoding="utf-8"))
                    _args = _meta.setdefault("args", {}) if isinstance(_meta, dict) else {}
                    if isinstance(_args, dict):
                        _args["energy_enabled"] = True
                        _args["dispatch_energy_enabled"] = True
                        _args["energy_measurement_scope"] = "command_energy"
                        _args["energy_setup_id"] = setup_id
                    _meta["objective"] = "remote benchmark with energy"
                    _meta.setdefault("objective_raw", _meta.get("objective"))
                    _meta["objective_source"] = "energy_dispatch_postprocess"
                    _meta["energy_measurement_scope"] = "command_energy"
                    _meta["energy_augmented_after_primary_run"] = True
                    _meta_p.write_text(json.dumps(_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as _exc:
            try:
                log(f"[energy][warn] could not update primary run_meta.json with energy metadata: {type(_exc).__name__}: {_exc}")
            except Exception:
                pass

        # v56p: energy windows should be long enough to overcome u.RECS pre/post
        # and idle dominance.  This mapping can scale the configured workload
        # for energy measurement without changing the ordinary benchmark settings.
        energy_phase_work_units: dict[str, int] = {}

        def _energy_target_variant_from_add_args(add_args: object) -> str:
            try:
                toks = shlex.split(str(add_args or ""))
                for i, tok in enumerate(toks):
                    if tok == "--energy-target-variant" and i + 1 < len(toks):
                        return str(toks[i + 1]).strip().lower()
                    if tok.startswith("--energy-target-variant="):
                        return tok.split("=", 1)[1].strip().lower()
            except Exception:
                return ""
            return ""

        def _energy_target_case_from_add_args(add_args: object) -> str:
            try:
                toks = shlex.split(str(add_args or ""))
                for i, tok in enumerate(toks):
                    if tok == "--energy-target-case" and i + 1 < len(toks):
                        return str(toks[i + 1]).strip()
                    if tok.startswith("--energy-target-case="):
                        return tok.split("=", 1)[1].strip()
            except Exception:
                return ""
            return ""

        def _clone_args_for_phase(phase: str) -> RemoteBenchmarkArgs:
            # All phase commands run a single benchmark repeat.  run_count in
            # run_fast_firmware_measurement controls the repeated energy windows.
            if phase == "latency":
                return RemoteBenchmarkArgs(
                    provider=str(getattr(args, "provider", "auto") or "auto"),
                    warmup=max(0, int(getattr(args, "warmup", 0) or 0)),
                    iters=max(1, int(energy_phase_work_units.get("latency") or getattr(args, "iters", 1) or 1)),
                    repeats=1,
                    timeout_s=getattr(args, "timeout_s", None),
                    throughput_frames=0,
                    throughput_warmup_frames=0,
                    throughput_queue_depth=max(1, int(getattr(args, "throughput_queue_depth", 1) or 1)),
                    validation_images=str(getattr(args, "validation_images", "") or ""),
                    validation_max_images=int(getattr(args, "validation_max_images", 0) or 0),
                    validation_reference_mode=str(getattr(args, "validation_reference_mode", "auto") or "auto"),
                    mini_coco_ap50=bool(getattr(args, "mini_coco_ap50", False)),
                    benchmark_task=str(getattr(args, "benchmark_task", "auto") or "auto"),
                    mini_classification_eval=bool(getattr(args, "mini_classification_eval", False)),
                    add_args=str(getattr(args, "add_args", "") or ""),
                    remote_venv=str(getattr(args, "remote_venv", "") or ""),
                    transfer_mode=str(getattr(args, "transfer_mode", "bundle") or "bundle"),
                    reuse_bundle=bool(getattr(args, "reuse_bundle", True)),
                    resume=False,
                )
            # Streaming phase.  Heterogeneous split rows can use the suite's
            # streaming/interleaving probe.  Full/same-backend rows do not have a
            # real two-stage streaming loop; for those, measure a long repeated
            # full-inference window instead.  Otherwise the old code measured
            # only one full inference and then divided by the analytical FPS,
            # which made full-run energy/frame look artificially small.
            target_variant = _energy_target_variant_from_add_args(getattr(args, "add_args", ""))
            streaming_units = max(0, int(energy_phase_work_units.get("streaming") or getattr(args, "throughput_frames", 0) or 0))
            if target_variant == "full":
                return RemoteBenchmarkArgs(
                    provider=str(getattr(args, "provider", "auto") or "auto"),
                    warmup=max(0, int(getattr(args, "throughput_warmup_frames", 0) or 0)),
                    iters=max(1, streaming_units or int(getattr(args, "iters", 1) or 1)),
                    repeats=1,
                    timeout_s=getattr(args, "timeout_s", None),
                    throughput_frames=0,
                    throughput_warmup_frames=0,
                    throughput_queue_depth=max(1, int(getattr(args, "throughput_queue_depth", 1) or 1)),
                    validation_images=str(getattr(args, "validation_images", "") or ""),
                    validation_max_images=int(getattr(args, "validation_max_images", 0) or 0),
                    validation_reference_mode=str(getattr(args, "validation_reference_mode", "auto") or "auto"),
                    mini_coco_ap50=bool(getattr(args, "mini_coco_ap50", False)),
                    benchmark_task=str(getattr(args, "benchmark_task", "auto") or "auto"),
                    mini_classification_eval=bool(getattr(args, "mini_classification_eval", False)),
                    add_args=str(getattr(args, "add_args", "") or ""),
                    remote_venv=str(getattr(args, "remote_venv", "") or ""),
                    transfer_mode=str(getattr(args, "transfer_mode", "bundle") or "bundle"),
                    reuse_bundle=bool(getattr(args, "reuse_bundle", True)),
                    resume=False,
                )
            return RemoteBenchmarkArgs(
                provider=str(getattr(args, "provider", "auto") or "auto"),
                warmup=0,
                iters=1,
                repeats=1,
                timeout_s=getattr(args, "timeout_s", None),
                throughput_frames=streaming_units,
                throughput_warmup_frames=max(0, int(getattr(args, "throughput_warmup_frames", 0) or 0)),
                throughput_queue_depth=max(1, int(getattr(args, "throughput_queue_depth", 1) or 1)),
                validation_images=str(getattr(args, "validation_images", "") or ""),
                validation_max_images=int(getattr(args, "validation_max_images", 0) or 0),
                validation_reference_mode=str(getattr(args, "validation_reference_mode", "auto") or "auto"),
                mini_coco_ap50=bool(getattr(args, "mini_coco_ap50", False)),
                benchmark_task=str(getattr(args, "benchmark_task", "auto") or "auto"),
                mini_classification_eval=bool(getattr(args, "mini_classification_eval", False)),
                add_args=str(getattr(args, "add_args", "") or ""),
                remote_venv=str(getattr(args, "remote_venv", "") or ""),
                transfer_mode=str(getattr(args, "transfer_mode", "bundle") or "bundle"),
                reuse_bundle=bool(getattr(args, "reuse_bundle", True)),
                resume=False,
            )

        phases: list[dict[str, Any]] = []
        if int(getattr(args, "iters", 0) or 0) > 0:
            phases.append({
                "phase": "latency",
                "label": f"latency_{int(getattr(args, 'iters', 0) or 0)}runs",
                "work_units_per_window": int(getattr(args, "iters", 0) or 0),
                "notes": "normal latency benchmark; streaming disabled",
            })
        if int(getattr(args, "throughput_frames", 0) or 0) > 0:
            phases.append({
                "phase": "streaming",
                "label": f"streaming_{int(getattr(args, 'throughput_frames', 0) or 0)}frames",
                "work_units_per_window": int(getattr(args, "throughput_frames", 0) or 0),
                "notes": "streaming/interleaving benchmark window; normal timing minimized to one sanity run by current suite CLI",
            })
        if not phases:
            phases.append({"phase": "latency", "label": "latency", "work_units_per_window": max(1, int(getattr(args, "iters", 1) or 1)), "notes": "fallback latency benchmark"})

        primary_remote_run_dir = str((primary_out or {}).get("remote_run_dir") or "").strip()
        primary_local_run_dir = str((primary_out or {}).get("local_run_dir") or "").strip()
        # v57q: for row-scoped recursive energy windows, keep exact metadata
        # from the selected benchmark result row.  This prevents full/canonical
        # baselines from being sized by sibling composed rows or from falling back
        # to the user-specified Runs count.
        target_meta: dict[str, Any] = energy_target_meta if isinstance(energy_target_meta, dict) else {}

        # v56s: row/variant-level energy windows.  A plan run such as
        # ort_tensorrt can contain multiple cases and multiple variants
        # (full/part1/part2/composed) in one benchmark_results row.  Measuring the
        # whole run-id would mix incomparable work.  Expand the dispatch into
        # case+variant targets unless this call is already scoped by
        # --energy-target-case/--energy-target-variant.
        add_args_existing = str(getattr(args, "add_args", "") or "")
        if primary_local_run_dir and "--energy-target-case" not in add_args_existing:
            try:
                targets = self._energy_collect_row_targets(primary_local_run_dir, str(run_id or ""))
            except Exception as exc:
                log(f"[energy][row-scope] warning: could not collect row targets: {type(exc).__name__}: {exc}")
                targets = []
            if targets:
                log(f"[energy][row-scope] measuring {len(targets)} case/variant target(s) for run_id={run_id or 'all'}")
                root_energy = group_dir / str(child_run_id) / "energy"
                root_energy.mkdir(parents=True, exist_ok=True)
                (root_energy / "energy_row_targets.json").write_text(json.dumps(targets, indent=2, ensure_ascii=False), encoding="utf-8")
                target_results: list[dict[str, Any]] = []
                for t in targets:
                    case_id = str(t.get("case_id") or "full")
                    variant = str(t.get("variant") or "composed")
                    safe_case = "".join((c if (c.isalnum() or c in "-_.") else "_") for c in case_id).strip("._-") or "case"
                    safe_var = "".join((c if (c.isalnum() or c in "-_.") else "_") for c in variant).strip("._-") or "variant"
                    extra = f" --energy-target-case {shlex.quote(case_id)} --energy-target-variant {shlex.quote(variant)}"
                    scoped_args = replace(args, add_args=(add_args_existing + extra).strip())
                    scoped_child = f"{child_run_id}_{safe_case}_{safe_var}"
                    log(f"[energy][row-scope] target case={case_id} variant={variant} -> {scoped_child}")
                    res = self._run_remote_dispatch_with_energy(
                        setup_id=setup_id,
                        run_id=run_id,
                        host=host,
                        benchmark_set_json=benchmark_set_json,
                        local_working_dir=local_working_dir,
                        child_run_id=scoped_child,
                        args=scoped_args,
                        results_group_id=results_group_id,
                        log=log,
                        primary_out=primary_out,
                        energy_target_meta=t,
                    )
                    if isinstance(res, dict):
                        res["energy_target_case"] = case_id
                        res["energy_target_variant"] = variant
                        res["energy_applies_to_all_cases"] = bool(t.get("applies_to_all_cases"))
                        res["canonical_full_baseline"] = bool(t.get("canonical_full_baseline"))
                        if t.get("canonical_full_target_case"):
                            res["canonical_full_target_case"] = t.get("canonical_full_target_case")
                        # v57f: keep a packaged copy of each per-row aggregate
                        # under the row-scope root.  Some lean bundles may omit
                        # nested child energy directories; target_results should
                        # never point at a missing artifact.
                        try:
                            src_agg_s = str(res.get("energy_aggregate") or "")
                            src_sum_s = str(res.get("energy_summary") or "")
                            target_pack_dir = root_energy / "targets" / scoped_child
                            target_pack_dir.mkdir(parents=True, exist_ok=True)
                            if src_agg_s and Path(src_agg_s).is_file():
                                dst = target_pack_dir / "energy_aggregate.json"
                                shutil.copy2(src_agg_s, dst)
                                res["energy_aggregate_original"] = src_agg_s
                                res["energy_aggregate"] = str(dst)
                                res["energy_aggregate_relpath"] = self._energy_artifact_relpath(dst)
                            if src_sum_s and Path(src_sum_s).is_file():
                                dsts = target_pack_dir / "energy_summary.json"
                                shutil.copy2(src_sum_s, dsts)
                                res["energy_summary_original"] = src_sum_s
                                res["energy_summary"] = str(dsts)
                                res["energy_summary_relpath"] = self._energy_artifact_relpath(dsts)
                        except Exception as exc:
                            res["energy_packaging_warning"] = f"{type(exc).__name__}: {exc}"
                        target_results.append(res)
                # Aggregate debug summary across targets while preserving phase separation.
                by_phase: dict[str, dict[str, Any]] = {}
                total_energy = 0.0
                total_valid = 0
                total_windows = 0
                for res in target_results:
                    agg_path = res.get("energy_aggregate") or res.get("energy_summary")
                    if not agg_path:
                        continue
                    try:
                        agg = json.loads(Path(agg_path).read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    for ph in agg.get("phases") or []:
                        if not isinstance(ph, dict):
                            continue
                        pn = str(ph.get("phase") or "unknown")
                        slot = by_phase.setdefault(pn, {"phase": pn, "target_count": 0, "energy_window_count": 0, "valid_energy_window_count": 0, "sum_energy_total_j": 0.0, "sum_window_duration_s": 0.0, "avg_power_weighted_num": 0.0, "avg_power_weighted_den": 0.0})
                        slot["target_count"] += 1
                        runs_l = ph.get("runs") or []
                        slot["energy_window_count"] += len(runs_l)
                        total_windows += len(runs_l)
                        for rr in runs_l:
                            if not isinstance(rr, dict):
                                continue
                            dur = rr.get("collector_duration_s")
                            if isinstance(dur, (int, float)):
                                slot["sum_window_duration_s"] += float(dur)
                            en = rr.get("energy_total_j")
                            if isinstance(en, (int, float)):
                                slot["valid_energy_window_count"] += 1
                                slot["sum_energy_total_j"] += float(en)
                                total_valid += 1
                                total_energy += float(en)
                            pw = rr.get("avg_power_w")
                            if isinstance(pw, (int, float)) and isinstance(dur, (int, float)) and float(dur) > 0:
                                slot["avg_power_weighted_num"] += float(pw) * float(dur)
                                slot["avg_power_weighted_den"] += float(dur)
                for slot in by_phase.values():
                    wc = int(slot.get("energy_window_count") or 0)
                    vc = int(slot.get("valid_energy_window_count") or 0)
                    slot["avg_window_duration_s"] = (float(slot.get("sum_window_duration_s") or 0.0) / wc) if wc else None
                    slot["avg_energy_total_j_per_window"] = (float(slot.get("sum_energy_total_j") or 0.0) / vc) if vc else None
                    den = float(slot.get("avg_power_weighted_den") or 0.0)
                    slot["avg_power_w_weighted"] = (float(slot.get("avg_power_weighted_num") or 0.0) / den) if den > 0 else None
                    slot.pop("avg_power_weighted_num", None)
                    slot.pop("avg_power_weighted_den", None)
                dispatch_avg_energy_j_per_window = (float(total_energy) / float(total_valid)) if total_valid else None
                summary = {
                    "schema": "onnx-splitpoint/energy-row-scope-dispatch",
                    "schema_version": 2,
                    "ok": any(bool(r.get("ok")) for r in target_results if isinstance(r, dict)),
                    "status": "ok" if target_results and all(str(r.get("status")) == "ok" for r in target_results if isinstance(r, dict)) else "partial",
                    "setup_id": setup_id,
                    "run_id": str(run_id or "all"),
                    "out_dir": str(root_energy),
                    "row_scope": True,
                    "target_count": len(targets),
                    "targets": targets,
                    "target_results": target_results,
                    "energy_window_count": total_windows,
                    "valid_energy_window_count": total_valid,
                    "sum_energy_total_j": total_energy if total_valid else None,
                    "dispatch_energy_total_j": total_energy if total_valid else None,
                    "dispatch_energy_window_count": total_windows,
                    "dispatch_valid_energy_window_count": total_valid,
                    "dispatch_avg_energy_j_per_window": dispatch_avg_energy_j_per_window,
                    "avg_energy_total_j_semantics": "deprecated_dispatch_sum_not_average; use dispatch_energy_total_j or dispatch_avg_energy_j_per_window",
                    "phases": list(by_phase.values()),
                    "energy_measurement_scope": "command_energy",
                    "energy_measurement_scope_note": "u.RECS windows measure the selected benchmark command phase. Init/warmup/logging can remain; steady-state loop-only energy is a future narrower mode.",
                    "debug_note": "Energy is measured per case+variant target. Latency and streaming windows are summarized separately; dispatch totals are diagnostics and are not per-row energy metrics.",
                }
                try:
                    summary["energy_aggregate_relpath"] = (root_energy / "energy_aggregate.json").relative_to(group_dir).as_posix()
                    summary["energy_summary_relpath"] = (root_energy / "energy_summary.json").relative_to(group_dir).as_posix()
                except Exception:
                    pass
                (root_energy / "energy_aggregate.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                (root_energy / "energy_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                if by_phase:
                    for ph in summary["phases"]:
                        log(f"[energy][row-scope] phase={ph.get('phase')} windows={ph.get('energy_window_count')} avg_window_s={ph.get('avg_window_duration_s')} total_J={ph.get('sum_energy_total_j')}")
                return {
                    "ok": bool(summary.get("ok")),
                    "status": summary.get("status"),
                    "setup_id": setup_id,
                    "run_id": str(run_id or "all"),
                    "energy": str(root_energy),
                    "energy_out_dir": str(root_energy),
                    "energy_aggregate": str(root_energy / "energy_aggregate.json"),
                    "energy_summary": str(root_energy / "energy_summary.json"),
                    "local_run_dir": str(group_dir),
                    "row_scope": True,
                    "target_count": len(targets),
                    "energy_window_count": total_windows,
                    "valid_energy_window_count": total_valid,
                    "dispatch_energy_total_j": summary.get("sum_energy_total_j"),
                    "dispatch_energy_window_count": total_windows,
                    "dispatch_valid_energy_window_count": total_valid,
                    "dispatch_avg_energy_j_per_window": (float(summary.get("sum_energy_total_j") or 0.0) / total_valid) if total_valid else None,
                    "phases": summary.get("phases"),
                }

        if primary_remote_run_dir:
            log(f"[energy] execution-only mode: using already deployed remote suite at {primary_remote_run_dir}/suite")
        else:
            log("[energy][warn] primary remote run did not expose remote_run_dir; falling back to benchmark-remote CLI wrapper for energy windows")

        def _extract_pipeline_fps_from_primary() -> float | None:
            """Best-effort: use downloaded result rows to annotate streaming FPS/W.

            v56q: do not use a raw backend `fps` field as pipeline FPS.  DX-RT's
            run_model can report an internal throughput number that is not the
            same as the steady-state application pipeline FPS.  Prefer explicit
            pipeline/streaming fields; for full rows fall back to 1000/full_mean_ms.
            """
            if not primary_local_run_dir:
                return None
            base = Path(primary_local_run_dir)
            if not base.exists():
                return None
            candidates: list[float] = []
            run_id_l = str(run_id or "").strip().lower()
            target_variant_l = _energy_target_variant_from_add_args(getattr(args, "add_args", ""))
            target_case_l = _energy_target_case_from_add_args(getattr(args, "add_args", "")).lower()
            try:
                f_meta = float(target_meta.get("energy_reference_fps") or target_meta.get("pipeline_fps_selected") or 0.0)
            except Exception:
                f_meta = 0.0
            if f_meta > 0:
                return f_meta
            def _rows_from(obj):
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict):
                    for k in ("rows", "results", "measurements", "records"):
                        v = obj.get(k)
                        if isinstance(v, list):
                            return v
                    return [obj]
                return []
            for jp in list(base.rglob("benchmark_results*.json"))[:100]:
                try:
                    obj = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                stem_l = jp.stem.lower()
                for row in _rows_from(obj):
                    if not isinstance(row, dict):
                        continue
                    tag = " ".join(str(row.get(k) or "") for k in ("run_id", "backend", "tag", "provider", "name")).lower()
                    if run_id_l and run_id_l not in stem_l and run_id_l not in tag:
                        continue
                    # v57e: if this energy window measures the full variant,
                    # target throughput must be derived from full latency.  The row
                    # can also contain same-backend composed pipeline diagnostics,
                    # which must not size full-energy streaming windows.
                    if target_variant_l == "full":
                        row_case_l = str(row.get("case_id") or row.get("case") or "").lower()
                        if target_case_l and row_case_l and row_case_l != target_case_l:
                            continue
                        for key in ("full_e2e_mean_ms", "full_mean_ms", "total_latency_ms"):
                            try:
                                ms = float(row.get(key))
                            except Exception:
                                continue
                            if ms > 0:
                                candidates.append(1000.0 / ms)
                                break
                        continue
                    # v57n: for row-scoped energy targets, only use the
                    # matching case/variant row.  Previously this function could
                    # pick the fastest pipeline FPS from a sibling case, which
                    # made target aggregates for b054 show b059 sizing metadata.
                    row_case_l = str(row.get("case_id") or row.get("case") or "").lower()
                    if target_case_l and row_case_l and row_case_l != target_case_l:
                        continue
                    if target_variant_l:
                        row_variant_l = str(row.get("variant") or "").lower()
                        row_primary_l = str(row.get("primary_variant") or "").lower()
                        # Most split-result rows have primary_variant=composed and
                        # no explicit variant field.  Treat primary_variant as the
                        # row variant for energy targeting.
                        if target_variant_l not in (row_variant_l, row_primary_l):
                            # Full rows are handled above.  For diagnostic rows
                            # without a variant marker keep compatibility only when
                            # no marker exists at all.
                            if row_variant_l or row_primary_l:
                                continue
                    # Explicit pipeline/streaming values only.
                    for key in ("pipeline_fps_selected", "pipeline_fps_measured", "throughput_fps_makespan", "pipeline_fps_with_transfer"):
                        try:
                            f = float(row.get(key))
                        except Exception:
                            continue
                        if 0.0 < f < 100000.0:
                            candidates.append(f)
                    # Conservative full-row fallback: full latency -> FPS.
                    variant_s = str(row.get("variant") or row.get("primary_variant") or "").lower()
                    if "full" in variant_s:
                        for key in ("full_e2e_mean_ms", "full_mean_ms", "total_latency_ms"):
                            try:
                                ms = float(row.get(key))
                            except Exception:
                                continue
                            if ms > 0:
                                candidates.append(1000.0 / ms)
                                break
            return max(candidates) if candidates else None

        primary_pipeline_fps = _extract_pipeline_fps_from_primary()

        def _extract_primary_energy_throughput_fps() -> float | None:
            """Best-effort throughput estimate for sizing u.RECS energy windows.

            This is deliberately separate from pipeline_fps_selected.  Some
            backends (notably DeepX run_model for full DXNNs) report an internal
            benchmark throughput that must *not* be used as application pipeline
            FPS, but it is useful for choosing how many iterations are needed to
            keep the measurement window active for min_active_s.  Without this,
            fast/full backends can get only a few seconds of real work in a
            nominal 30s energy window.
            """
            if not primary_local_run_dir:
                return None
            base = Path(primary_local_run_dir)
            if not base.exists():
                return None
            run_id_l = str(run_id or "").strip().lower()
            add_args_l = str(getattr(args, "add_args", "") or "").lower()
            # Only use backend-internal FPS for full-DXNN/full-backend targets.
            # For split rows, pipeline FPS remains the correct throughput basis.
            target_variant = _energy_target_variant_from_add_args(getattr(args, "add_args", ""))
            allow_internal = (
                "deepx" in run_id_l
                and (not target_variant or target_variant == "full" or "--energy-target-variant full" in add_args_l)
            )
            if not allow_internal:
                return None
            candidates: list[float] = []
            def _rows_from(obj):
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict):
                    for k in ("rows", "results", "measurements", "records"):
                        v = obj.get(k)
                        if isinstance(v, list):
                            return v
                    return [obj]
                return []
            for jp in list(base.rglob("benchmark_results*.json"))[:100]:
                try:
                    obj = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                stem_l = jp.stem.lower()
                for row in _rows_from(obj):
                    if not isinstance(row, dict):
                        continue
                    tag = " ".join(str(row.get(k) or "") for k in ("run_id", "backend", "tag", "provider", "name")).lower()
                    if run_id_l and run_id_l not in stem_l and run_id_l not in tag:
                        continue
                    variant_s = str(row.get("variant") or row.get("primary_variant") or "").lower()
                    if "full" not in variant_s:
                        continue
                    for key in ("dxrt_tool_fps", "backend_tool_fps", "run_model_fps", "fps"):
                        try:
                            f = float(row.get(key))
                        except Exception:
                            continue
                        if 0.0 < f < 100000.0:
                            candidates.append(f)
                            break
            return max(candidates) if candidates else None

        primary_energy_throughput_fps = _extract_primary_energy_throughput_fps()

        def _extract_primary_latency_ms() -> float | None:
            """Best-effort latency estimate for the *current energy target*.

            v57n: This must be target-specific.  A run_id can contain multiple
            cases and variants.  For example, deepx_m1_to_tensorrt/b054 must not
            inherit b059's composed latency, and TensorRT full must not size a
            full-energy window from the same-backend composed latency.
            """
            if not primary_local_run_dir:
                return None
            base = Path(primary_local_run_dir)
            if not base.exists():
                return None
            run_id_l = str(run_id or "").strip().lower()
            target_variant_l = _energy_target_variant_from_add_args(getattr(args, "add_args", ""))
            target_case_l = _energy_target_case_from_add_args(getattr(args, "add_args", "")).lower()
            candidates: list[float] = []
            try:
                if target_variant_l == "full":
                    f_meta = float(target_meta.get("full_mean_ms") or target_meta.get("full_e2e_mean_ms") or target_meta.get("total_latency_ms") or 0.0)
                elif target_variant_l == "part1":
                    f_meta = float(target_meta.get("part1_mean_ms") or 0.0)
                elif target_variant_l == "part2":
                    f_meta = float(target_meta.get("part2_mean_ms") or 0.0)
                else:
                    f_meta = float(target_meta.get("composed_mean_ms") or target_meta.get("split_latency_e2e_ms") or 0.0)
            except Exception:
                f_meta = 0.0
            if f_meta > 0:
                return f_meta

            def _rows_from(obj):
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict):
                    for k in ("rows", "results", "measurements", "records"):
                        v = obj.get(k)
                        if isinstance(v, list):
                            return v
                    return [obj]
                return []

            def _row_matches(row: dict) -> bool:
                tag = " ".join(str(row.get(k) or "") for k in ("run_id", "backend", "tag", "provider", "name")).lower()
                if run_id_l and run_id_l not in tag:
                    # The file stem check is done by caller context; keep tag
                    # compatibility but don't reject too eagerly here.
                    pass
                row_case_l = str(row.get("case_id") or row.get("case") or "").lower()
                if target_case_l and row_case_l and row_case_l != target_case_l:
                    return False
                if target_variant_l:
                    row_variant_l = str(row.get("variant") or "").lower()
                    row_primary_l = str(row.get("primary_variant") or "").lower()
                    if target_variant_l not in (row_variant_l, row_primary_l):
                        if row_variant_l or row_primary_l:
                            return False
                return True

            if target_variant_l == "full":
                keys = ("full_e2e_mean_ms", "full_mean_ms", "total_latency_ms", "latency_mean_ms", "mean_latency_ms")
            elif target_variant_l == "part1":
                keys = ("part1_mean_ms", "latency_mean_ms", "mean_latency_ms", "mean_ms")
            elif target_variant_l == "part2":
                keys = ("part2_mean_ms", "latency_mean_ms", "mean_latency_ms", "mean_ms")
            else:
                # Composed split rows: prefer end-to-end/composed latency.  Do
                # not allow full_mean_ms ahead of composed_mean_ms here.
                keys = ("composed_mean_ms", "split_latency_e2e_ms", "latency_mean_ms", "mean_latency_ms", "mean_ms", "total_latency_ms", "full_mean_ms")

            for jp in list(base.rglob("benchmark_results*.json"))[:100]:
                try:
                    obj = json.loads(jp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                stem_l = jp.stem.lower()
                if run_id_l and run_id_l not in stem_l:
                    # Avoid scanning unrelated result files when multiple run_ids
                    # live in the same local group.
                    continue
                for row in _rows_from(obj):
                    if not isinstance(row, dict) or not _row_matches(row):
                        continue
                    for key in keys:
                        try:
                            f = float(row.get(key))
                        except Exception:
                            continue
                        if 0.0 < f < 100000.0:
                            candidates.append(f)
                            break
            # Prefer the target row value; if duplicate candidate rows exist,
            # choose the median-ish lower-noise value rather than the global min.
            if not candidates:
                return None
            candidates = sorted(candidates)
            return candidates[len(candidates)//2]

        # v56p: scale energy-only workloads to a minimum active window.  The
        # default 30 s prevents very fast models (e.g. DeepX Full ResNet50) from
        # being measured as a mostly-idle 5-7 s window, which made energy per
        # inference look artificially low/high depending on normalization.
        try:
            min_active_s = max(0.0, float(getattr(defaults, "min_active_duration_s", 30.0) or 0.0))
        except Exception:
            min_active_s = 30.0
        latency_ms = _extract_primary_latency_ms()
        for pm in phases:
            ph = str(pm.get("phase") or "")
            original_wu = max(1, int(pm.get("work_units_per_window") or 1))
            scaled_wu = original_wu
            reason = ""
            # v57c: size energy windows from the *same benchmark path* that is
            # executed inside the u.RECS window.  Do not use DeepX run_model
            # dxrt_tool_fps as the primary sizing signal here: it is a backend
            # internal/buffered throughput diagnostic and can be much higher
            # than benchmark_suite.py's prepared-feed execution rate.  Using it
            # caused 30s target windows to become ~90s for DeepX Full.
            if min_active_s > 0 and ph == "streaming" and primary_pipeline_fps and primary_pipeline_fps > 0:
                import math
                scaled_wu = max(original_wu, int(math.ceil(min_active_s * float(primary_pipeline_fps))))
                reason = f"energy_sizing_pipeline_fps={primary_pipeline_fps:.3f}"
            elif min_active_s > 0 and latency_ms and latency_ms > 0:
                import math
                scaled_wu = max(original_wu, int(math.ceil((min_active_s * 1000.0) / float(latency_ms))))
                reason = f"energy_sizing_latency_mean_ms={latency_ms:.3f}"
            elif min_active_s > 0 and primary_energy_throughput_fps and primary_energy_throughput_fps > 0:
                import math
                scaled_wu = max(original_wu, int(math.ceil(min_active_s * float(primary_energy_throughput_fps))))
                reason = f"energy_sizing_backend_internal_throughput_fps_fallback={primary_energy_throughput_fps:.3f}"
            # Keep a practical hard cap to avoid accidental multi-hour energy windows.
            # Users can raise the configured benchmark frames/runs manually if needed.
            if scaled_wu > 50000:
                reason += f"; capped_from={scaled_wu}"
                scaled_wu = 50000
            energy_phase_work_units[ph] = scaled_wu
            pm["requested_work_units_per_window"] = original_wu
            pm["work_units_per_window"] = scaled_wu
            pm["min_active_duration_s"] = min_active_s
            pm["auto_scaled_for_energy"] = bool(scaled_wu != original_wu)
            if scaled_wu != original_wu:
                pm["notes"] = str(pm.get("notes") or "") + f"; auto-scaled for {min_active_s:.1f}s energy window ({original_wu}→{scaled_wu}; {reason})"
                log(f"[energy] phase={ph}: auto-scaled work units {original_wu} -> {scaled_wu} for min_active={min_active_s:.1f}s ({reason})")
            else:
                log(f"[energy] phase={ph}: work units {original_wu}; min_active={min_active_s:.1f}s no scaling ({reason or 'no estimate'})")

        def _primary_preflight_python_env() -> dict[str, str]:
            """Return the Python interpreter/sys.path that the successful primary remote run used.

            v59i: Energy remote-execution windows must use the same remote Python
            selection as the primary benchmark.  On Hailo hosts the configured
            runtime venv (~/hailo_py) provides hailo_platform, but it may not
            contain CUDA/TensorRT ORT providers or even onnx.  The primary
            benchmark solves this by running /usr/bin/python3 with
            SPLITPOINT_EXTRA_SITES pointing at hailo_py site-packages.  The
            previous energy-only command sourced the venv and then polluted PYTHONPATH
            with the Hailo venv site-packages, which shadowed the system
            GPU-enabled onnxruntime with a CPU-only wheel.  Reading preflight.json
            and exporting SPLITPOINT_EXTRA_SITES only keeps the energy windows
            consistent with the already verified primary run.
            """
            out: dict[str, str] = {}
            try:
                if not primary_local_run_dir:
                    return out
                pf = Path(primary_local_run_dir) / "results" / "preflight.json"
                if not pf.exists():
                    # Some older bundles extract preflight one level up.
                    alt = Path(primary_local_run_dir) / "preflight.json"
                    pf = alt if alt.exists() else pf
                if not pf.exists():
                    return out
                obj = json.loads(pf.read_text(encoding="utf-8"))
                py = obj.get("python") if isinstance(obj, dict) else {}
                if not isinstance(py, dict):
                    return out
                for key in ("run_py", "extra_sites", "hailo_site", "deepx_site", "env_site", "sys_py", "env_py"):
                    val = py.get(key)
                    if isinstance(val, str) and val.strip():
                        out[key] = val.strip()
            except Exception:
                return out
            return out

        def _remote_execution_only_command(phase_args: RemoteBenchmarkArgs, *, phase: str, stdout_log: Path, stderr_log: Path, exit_json: Path) -> tuple[str, str]:
            """Build a local command that measures only remote suite execution.

            v57h: write a small local SSH wrapper script instead of embedding a
            double-quoted ssh/bash command into the collector command.  The
            previous inline command could be broken by empty arguments (notably
            --validation-images "") and by add_args that already contained
            quotes.  A wrapper script keeps quoting stable and makes the exact
            command inspectable in the energy folder.
            """
            if not primary_remote_run_dir:
                cli_cmd = self._remote_benchmark_cli_command(
                    host=host,
                    benchmark_set_json=benchmark_set_json,
                    local_working_dir=local_working_dir,
                    run_id=f"{child_run_id}_{phase}",
                    args=phase_args,
                    results_group_id=results_group_id,
                )
                script_path = stdout_log.parent / "energy_remote_exec_fallback.sh"
                script = (
                    "#!/usr/bin/env bash\n"
                    "set +e\n"
                    + shlex.join(cli_cmd)
                    + " > " + shlex.quote(str(stdout_log))
                    + " 2> " + shlex.quote(str(stderr_log))
                    + "\nrc=$?\n"
                    + "printf '{\"rc\":%s}\\n' \"$rc\" > " + shlex.quote(str(exit_json)) + "\n"
                    + "exit \"$rc\"\n"
                )
                script_path.write_text(script, encoding="utf-8")
                try:
                    script_path.chmod(0o755)
                except Exception:
                    pass
                return shlex.join(["bash", str(script_path)]), "benchmark_remote_cli_fallback"

            remote_run_dir = primary_remote_run_dir.rstrip("/")
            remote_suite_dir = posixpath.join(remote_run_dir, "suite")
            remote_suite_root = posixpath.dirname(posixpath.dirname(remote_run_dir)) if "/" in remote_run_dir else remote_run_dir
            remote_trt_cache_root = posixpath.join(remote_suite_root, "_shared_trt_cache")
            provider = str(getattr(phase_args, "provider", "auto") or "auto").strip() or "auto"
            warmup = max(0, int(getattr(phase_args, "warmup", 0) or 0))
            effective_runs = max(1, int(getattr(phase_args, "repeats", 1) or 1) * int(getattr(phase_args, "iters", 1) or 1))
            throughput_frames = max(0, int(getattr(phase_args, "throughput_frames", 0) or 0))
            throughput_warmup_frames = max(0, int(getattr(phase_args, "throughput_warmup_frames", 0) or 0))
            throughput_queue_depth = max(1, int(getattr(phase_args, "throughput_queue_depth", 1) or 1))
            validation_images = str(getattr(phase_args, "validation_images", "") or "").strip()
            validation_max_images = max(0, int(getattr(phase_args, "validation_max_images", 0) or 0))
            validation_reference_mode = str(getattr(phase_args, "validation_reference_mode", "auto") or "auto")
            benchmark_task = str(getattr(phase_args, "benchmark_task", "auto") or "auto")
            add_args = str(getattr(phase_args, "add_args", "") or "").strip()

            bench_tokens = [
                "-u", "benchmark_suite.py",
                "--provider", provider,
                "--plan", "benchmark_plan.json",
                "--warmup", str(warmup),
                "--runs", str(effective_runs),
                "--trt-cache-root", remote_trt_cache_root,
                "--throughput-frames", str(throughput_frames),
                "--throughput-warmup-frames", str(throughput_warmup_frames),
                "--throughput-queue-depth", str(throughput_queue_depth),
                "--validation-max-images", str(validation_max_images),
                "--validation-reference-mode", validation_reference_mode,
                "--benchmark-task", benchmark_task,
                "--energy-measurement-only",
            ]
            # In energy-measurement-only mode validation images are not needed.
            # Passing an empty string caused nested-shell quoting breakage in v57g.
            if validation_images:
                bench_tokens.extend(["--validation-images", validation_images])
            if bool(getattr(phase_args, "mini_coco_ap50", False)):
                bench_tokens.append("--mini-coco-ap50")
            if bool(getattr(phase_args, "mini_classification_eval", False)):
                bench_tokens.append("--mini-classification-eval")
            if add_args:
                try:
                    bench_tokens.extend(shlex.split(add_args))
                except ValueError:
                    # Keep the old behavior as a last-resort fallback, but record the
                    # raw tokens in the generated script for debugging.
                    bench_tokens.extend(add_args.split())
            bench_cmd = '"$RUN_PY" ' + shlex.join(bench_tokens)

            remote_venv_cmd = str(getattr(phase_args, "remote_venv", "") or "").strip()
            env_line = ""
            if remote_venv_cmd:
                if any(ch.isspace() for ch in remote_venv_cmd):
                    env_line = remote_venv_cmd
                else:
                    if remote_venv_cmd.startswith("~/"):
                        remote_venv_cmd = "$HOME/" + remote_venv_cmd[2:]
                    env_line = f"source {remote_venv_cmd}"
            preflight_env = _primary_preflight_python_env()
            preflight_run_py = preflight_env.get("run_py") or ""
            preflight_extra = preflight_env.get("extra_sites") or ""
            remote_lines = ["set -e", f"cd {shlex.quote(remote_suite_dir)}", "mkdir -p logs"]
            # Source the configured venv for side effects, but use the interpreter
            # that the primary remote preflight already verified whenever possible.
            # This matters on Hailo hosts: the hailo_py venv provides
            # hailo_platform, while /usr/bin/python3 has CUDA/TensorRT ORT
            # providers.  The primary run uses /usr/bin/python3 plus
            # SPLITPOINT_EXTRA_SITES; energy windows must do the same.
            if env_line:
                remote_lines.append(env_line)
            if preflight_extra:
                remote_lines += [
                    f"export SPLITPOINT_EXTRA_SITES={shlex.quote(preflight_extra)}${{SPLITPOINT_EXTRA_SITES:+:$SPLITPOINT_EXTRA_SITES}}",
                    # Do NOT export PYTHONPATH here.  The runner extends sys.path with
                    # SPLITPOINT_EXTRA_SITES *after* default system paths via site.addsitedir.
                    # Prepending PYTHONPATH lets Hailo venv packages shadow the system
                    # GPU-enabled onnxruntime, which breaks CUDA/TensorRT energy replays.
                    'echo "[energy][remote-exec-only] extra_sites=$SPLITPOINT_EXTRA_SITES (not PYTHONPATH)" >&2',
                ]
            if preflight_run_py:
                remote_lines += [
                    f"RUN_PY={shlex.quote(preflight_run_py)}",
                    'if [ ! -x "$RUN_PY" ]; then echo "[energy][remote-exec-only] preflight RUN_PY not executable: $RUN_PY; falling back to python3" >&2; RUN_PY="$(command -v python3)"; fi',
                    'echo "[energy][remote-exec-only] using preflight RUN_PY=$RUN_PY" >&2',
                ]
            else:
                remote_lines += [
                    'RUN_PY="$(command -v python3)"',
                    'if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then RUN_PY="${VIRTUAL_ENV}/bin/python"; fi',
                    'echo "[energy][remote-exec-only] using fallback RUN_PY=$RUN_PY" >&2',
                ]
            remote_lines += [
                'echo "[energy][remote-exec-only] suite=$(pwd) RUN_PY=$RUN_PY" >&2',
                'echo "[energy][remote-exec-only] command: ' + bench_cmd.replace('"', '\"') + '" >&2',
                bench_cmd,
            ]
            remote_payload = stdout_log.parent / "energy_remote_payload.sh"
            remote_payload.write_text("\n".join(remote_lines) + "\n", encoding="utf-8")

            script_path = stdout_log.parent / "energy_remote_exec.sh"
            ssh_base = ["ssh"]
            try:
                if int(getattr(host, "port", 22) or 22) != 22:
                    ssh_base += ["-p", str(int(getattr(host, "port", 22) or 22))]
            except Exception:
                pass
            extra = str(getattr(host, "ssh_extra_args", "") or "").strip()
            if extra:
                try:
                    ssh_base += shlex.split(extra)
                except ValueError:
                    ssh_base += extra.split()
            user = str(getattr(host, "user", "") or "").strip()
            host_name = str(getattr(host, "host", "") or "").strip()
            target = f"{user}@{host_name}" if user else host_name
            ssh_cmd = ssh_base + [target, "bash", "-s"]
            script = (
                "#!/usr/bin/env bash\n"
                "set +e\n"
                + shlex.join(ssh_cmd)
                + " > " + shlex.quote(str(stdout_log))
                + " 2> " + shlex.quote(str(stderr_log))
                + " < " + shlex.quote(str(remote_payload))
                + "\nrc=$?\n"
                + "printf '{\"rc\":%s}\\n' \"$rc\" > " + shlex.quote(str(exit_json)) + "\n"
                + "exit \"$rc\"\n"
            )
            script_path.write_text(script, encoding="utf-8")
            try:
                script_path.chmod(0o755)
            except Exception:
                pass
            return shlex.join(["bash", str(script_path)]), "remote_execution_only"

        # v56m: TensorRT engine construction must not be part of the energy
        # window or its duration probe.  The primary remote run usually fills
        # the shared TensorRT cache already, but we explicitly prebuild/warm the
        # cache once more outside u.RECS measurement for any run_id involving
        # TensorRT.  This prevents ORT/TRT engine-build time from contaminating
        # the measured inference energy.
        if self._energy_run_uses_tensorrt(str(run_id or ""), args):
            prebuild_dir = energy_root / "trt_prebuild"
            prebuild_dir.mkdir(parents=True, exist_ok=True)
            pb_stdout = prebuild_dir / "stdout.log"
            pb_stderr = prebuild_dir / "stderr.log"
            pb_exit = prebuild_dir / "exit.json"
            try:
                pb_args = _clone_args_for_phase("latency")
                pb_args = replace(pb_args, warmup=0, iters=1, repeats=1, throughput_frames=0, throughput_warmup_frames=0, resume=False)
                pb_cmd, pb_mode = _remote_execution_only_command(pb_args, phase="trt_prebuild", stdout_log=pb_stdout, stderr_log=pb_stderr, exit_json=pb_exit)
                (prebuild_dir / "command.txt").write_text(pb_cmd + "\n", encoding="utf-8")
                log(f"[energy][trt-prebuild] run_id={run_id or 'all'} mode={pb_mode}; building/warming TensorRT cache outside u.RECS window")
                t0 = time.time()
                proc = subprocess.run(pb_cmd, shell=True, executable="/bin/bash", timeout=max(60, int(getattr(args, "timeout_s", 7200) or 7200)))
                dt = time.time() - t0
                status = {"ok": proc.returncode == 0, "rc": proc.returncode, "duration_s": dt, "mode": pb_mode, "stdout": str(pb_stdout), "stderr": str(pb_stderr)}
                (prebuild_dir / "trt_prebuild_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
                if proc.returncode == 0:
                    log(f"[energy][trt-prebuild] ok in {dt:.1f}s")
                else:
                    log(f"[energy][trt-prebuild] warning: rc={proc.returncode}; energy run continues but engine build may affect first measured window")
            except Exception as exc:
                try:
                    (prebuild_dir / "trt_prebuild_status.json").write_text(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                log(f"[energy][trt-prebuild] warning: {type(exc).__name__}: {exc}")

        log(f"[energy] plan setup={setup_id} run_id={run_id or 'all'} phase_repeats={phase_repeats} source={repeat_source} phases={[p['phase'] for p in phases]} uRECS={setup.urecs_address}")
        phase_results: list[dict[str, Any]] = []
        status_values: list[str] = []
        target_variant_for_host_normalization = str(
            target_meta.get("variant") or ""
        ).strip().lower()
        if not target_variant_for_host_normalization:
            target_variant_for_host_normalization = (
                _energy_target_variant_from_add_args(getattr(args, "add_args", ""))
            )
        host_normalization_role = select_host_normalization_role(
            run_id=run_id,
            target_variant=target_variant_for_host_normalization,
        )

        for phase_meta in phases:
            phase = str(phase_meta["phase"])
            phase_args = _clone_args_for_phase(phase)
            phase_child_run_id = f"{child_run_id}_{phase}"
            phase_dir = energy_root / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            stdout_log = phase_dir / "remote_benchmark_stdout.log"
            stderr_log = phase_dir / "remote_benchmark_stderr.log"
            exit_json = phase_dir / "remote_benchmark_exit.json"
            command, command_mode = _remote_execution_only_command(phase_args, phase=phase, stdout_log=stdout_log, stderr_log=stderr_log, exit_json=exit_json)
            try:
                (phase_dir / "energy_measured_command_mode.txt").write_text(str(command_mode) + "\n", encoding="utf-8")
            except Exception:
                pass
            log(f"[energy] phase={phase} repeats={phase_repeats} work_units/window={phase_meta.get('work_units_per_window')} out={phase_dir}")
            log(
                f"[energy] phase={phase}: starting duration probe and {phase_repeats} u.RECS window(s). "
                "Each u.RECS window measures the full configured phase, not single inferences; "
                "collector output is written to the energy/ folder and may be quiet until the window finishes."
            )
            result = run_fast_firmware_measurement(
                command,
                phase_dir,
                setup=setup,
                defaults=defaults,
                setup_id=setup_id,
                run_id=f"{run_id or 'all'}:{phase}",
                duration_s=None,
                run_count=phase_repeats,
                # Do not pass the outer remote timeout here.  Energy windows have
                # a known fixed collector duration after the probe; the collector
                # subprocess should be bounded by duration + pre/post + slack.
                # Passing the 7200s benchmark timeout made stalled collector runs
                # look like GUI hangs for hours.
                timeout_s=None,
                inference_count=int(phase_meta.get("work_units_per_window") or 0) or None,
                pipeline_fps_selected=(primary_pipeline_fps if phase == "streaming" else None),
                host_normalization_role=host_normalization_role,
                host_normalization_source_run_id=run_id,
                host_normalization_target_variant=target_variant_for_host_normalization,
            )
            remote_rc = None
            try:
                if exit_json.exists():
                    remote_rc = json.loads(exit_json.read_text(encoding="utf-8")).get("rc")
            except Exception:
                remote_rc = None
            result.update({
                "phase": phase,
                "phase_label": phase_meta.get("label"),
                "phase_notes": phase_meta.get("notes"),
                "energy_command_reference_fps": (float(primary_pipeline_fps) if (phase == "streaming" and primary_pipeline_fps) else None),
                "energy_command_reference_fps_source": ("window_sizing_hint_not_row_pipeline_fps" if (phase == "streaming" and primary_pipeline_fps) else None),
                "phase_repeat_count": phase_repeats,
                "repeat_source": repeat_source,
                "work_units_per_window": phase_meta.get("work_units_per_window"),
                "remote_benchmark_rc": remote_rc,
                "remote_benchmark_stdout": str(stdout_log),
                "remote_benchmark_stderr": str(stderr_log),
                "phase_out_dir": str(phase_dir),
                "energy_command_mode": command_mode,
                "primary_remote_run_dir": primary_remote_run_dir,
                # v57l: target aggregates should describe the u.RECS command
                # window explicitly.  Do not call these values
                # primary_pipeline_fps_selected, because row-level
                # pipeline_fps_selected belongs to the benchmark result row.
                "energy_command_work_units_per_window": phase_meta.get("work_units_per_window"),
                "energy_command_reference_pipeline_fps": primary_pipeline_fps if phase == "streaming" else None,
                "energy_command_reference_fps_source": "benchmark_row_pipeline_fps_selected_or_full_latency_fps" if phase == "streaming" and primary_pipeline_fps else None,
            })
            # v59h: The collector can successfully measure a command that
            # exited 0 even though the benchmark suite skipped the target because
            # the wrong remote Python environment was used.  Do not treat such
            # measurements as valid energy: they measured an error path, not
            # inference.  Typical markers are provider_unavailable, case failed,
            # or the suite saying that no results were produced.
            remote_invalid_reason = ""
            try:
                out_txt = stdout_log.read_text(encoding="utf-8", errors="replace") if stdout_log.exists() else ""
                err_txt = stderr_log.read_text(encoding="utf-8", errors="replace") if stderr_log.exists() else ""
                low = (out_txt + "\n" + err_txt).lower()
                if "provider unavailable" in low:
                    remote_invalid_reason = "provider_unavailable_in_energy_window"
                elif "[warn] case failed" in low or "case failed:" in low:
                    remote_invalid_reason = "case_failed_in_energy_window"
                elif "some runs produced no results" in low or "no results collected" in low:
                    remote_invalid_reason = "no_results_in_energy_window"
                elif "modulenotfounderror" in low:
                    remote_invalid_reason = "python_environment_missing_module_in_energy_window"
            except Exception:
                remote_invalid_reason = ""
            if remote_invalid_reason:
                result["remote_result_invalid"] = True
                result["remote_result_invalid_reason"] = remote_invalid_reason
                result["runs_ignored_due_to_remote_result_invalid"] = len(result.get("runs") or [])
                result["runs"] = []
                result["valid_postprocessed_runs"] = 0
                result["avg_energy_total_j"] = None
                result["avg_power_w"] = None
                result["avg_energy_dynamic_j"] = None
                result["avg_host_normalized_energy_est_j"] = None
                result[
                    "avg_host_normalized_energy_per_configured_work_unit_est_j"
                ] = None
                result["avg_energy_per_inference_j"] = None
                result["postprocess_status"] = "remote_result_invalid"

            try:
                wu = int(phase_meta.get("work_units_per_window") or 0)
                if wu > 0 and isinstance(result.get("avg_energy_total_j"), (int, float)):
                    result["avg_energy_per_configured_work_unit_j"] = float(result["avg_energy_total_j"]) / float(wu)
                if wu > 0 and isinstance(result.get("avg_energy_dynamic_j"), (int, float)):
                    result["avg_energy_dynamic_per_configured_work_unit_j"] = float(result["avg_energy_dynamic_j"]) / float(wu)
                if wu > 0 and isinstance(result.get("avg_host_normalized_energy_est_j"), (int, float)):
                    result["avg_host_normalized_energy_per_configured_work_unit_est_j"] = float(result["avg_host_normalized_energy_est_j"]) / float(wu)
                if wu > 0 and isinstance(result.get("avg_energy_total_j"), (int, float)) and float(result.get("avg_energy_total_j") or 0) > 0:
                    result["configured_work_units_per_j"] = float(wu) / float(result["avg_energy_total_j"])
                # v57l: explicit command-window throughput diagnostics.  These
                # describe what the u.RECS window actually ran and should not be
                # confused with benchmark-owned pipeline_fps_selected.
                if wu > 0:
                    dur_cmd = result.get("avg_workload_duration_s") or result.get("avg_active_duration_s") or result.get("avg_window_duration_s")
                    if isinstance(dur_cmd, (int, float)) and float(dur_cmd) > 0:
                        result["energy_command_work_units_per_s"] = float(wu) / float(dur_cmd)
                    dur_win = result.get("avg_window_duration_s")
                    if isinstance(dur_win, (int, float)) and float(dur_win) > 0:
                        result["energy_window_work_units_per_s"] = float(wu) / float(dur_win)
                if phase == "streaming" and primary_pipeline_fps and isinstance(result.get("avg_power_w"), (int, float)) and float(result.get("avg_power_w") or 0) > 0:
                    # v57l: these values describe the energy command/window, not
                    # the row-owned benchmark pipeline metric.  Do not write a
                    # bare pipeline_fps_selected into target aggregates; that led
                    # to confusing b059/b054 metadata.  Row-level selected-FPS
                    # diagnostics are recomputed after merge from each result row.
                    result["energy_command_streaming_fps"] = float(primary_pipeline_fps)
                    result["energy_command_streaming_fps_note"] = "phase-level reference used for this u.RECS command; row.pipeline_fps_selected remains the authoritative benchmark metric"
                    result["energy_command_fps_per_watt"] = float(primary_pipeline_fps) / float(result["avg_power_w"])
                    result["energy_command_j_per_frame"] = float(result["avg_power_w"]) / float(primary_pipeline_fps)
                    # Primary energy efficiency should be measured-window based:
                    # frames/J = configured work units / measured energy.  This is
                    # directly tied to the u.RECS window and avoids mixing a
                    # benchmark-level pipeline FPS with a different energy window.
                    if result.get("configured_work_units_per_j") is not None:
                        result["pipeline_fps_per_watt"] = result.get("configured_work_units_per_j")
                        result["energy_per_pipeline_frame_j"] = result.get("avg_energy_per_configured_work_unit_j")
                        result["energy_efficiency_source"] = "measured_energy_window_work_units_per_j"
                    else:
                        result["energy_streaming_fps_per_watt_from_command_fps"] = float(primary_pipeline_fps) / float(result["avg_power_w"])
                        result["energy_streaming_j_per_frame_from_command_fps"] = float(result["avg_power_w"]) / float(primary_pipeline_fps)
                        result["energy_efficiency_source"] = "command_reference_fps_over_avg_power"
            except Exception:
                pass
            phase_status = "ok" if bool(result.get("ok")) and (remote_rc in (0, None)) else ("partial" if bool(result.get("ok")) else "failed")
            if result.get("remote_result_invalid"):
                phase_status = "failed"
            if remote_rc not in (0, None):
                phase_status = "failed"
            result["status"] = phase_status
            status_values.append(phase_status)
            phase_results.append(result)
            try:
                (phase_dir / "energy_dispatch_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
            if result.get("avg_power_w") is not None:
                log(f"[energy] phase={phase} avg_power={float(result['avg_power_w']):.3f} W energy={result.get('avg_energy_total_j')} J")
            else:
                log(f"[energy] phase={phase} postprocess={result.get('status')} valid_runs={result.get('valid_postprocessed_runs')} raw measurement kept")
            if remote_rc not in (0, None):
                log(f"[energy] phase={phase} remote benchmark command rc={remote_rc}; see {stderr_log}")

        # Aggregate across phases for the dispatch while keeping phase-level
        # values explicit.  Do not collapse latency and streaming into a single
        # energy/inference claim without the phase labels.
        total_windows = 0
        valid_windows = 0
        total_energy = 0.0
        total_dynamic = 0.0
        valid_dynamic = 0
        total_host_norm = 0.0
        valid_host_norm = 0
        weighted_power_num = 0.0
        weighted_power_den = 0.0
        for pr in phase_results:
            for r in (pr.get("runs") or []):
                total_windows += 1
                if isinstance(r.get("energy_total_j"), (int, float)):
                    valid_windows += 1
                    total_energy += float(r.get("energy_total_j"))
                if isinstance(r.get("energy_dynamic_j"), (int, float)):
                    valid_dynamic += 1
                    total_dynamic += float(r.get("energy_dynamic_j"))
                if isinstance(r.get("host_normalized_energy_est_j"), (int, float)):
                    valid_host_norm += 1
                    total_host_norm += float(r.get("host_normalized_energy_est_j"))
                dur = r.get("collector_duration_s")
                if isinstance(r.get("avg_power_w"), (int, float)) and isinstance(dur, (int, float)) and float(dur) > 0:
                    weighted_power_num += float(r.get("avg_power_w")) * float(dur)
                    weighted_power_den += float(dur)
        aggregate = {
            "schema": "onnx-splitpoint/energy-dispatch-phases",
            "schema_version": 1,
            "ok": any(str(s) in {"ok", "partial"} for s in status_values),
            "status": "ok" if status_values and all(s == "ok" for s in status_values) else ("partial" if any(s in {"ok", "partial"} for s in status_values) else "failed"),
            "setup_id": setup_id,
            "run_id": str(run_id or "all"),
            "out_dir": str(energy_root),
            "urecs_address": setup.urecs_address,
            "benchmark_repeats": benchmark_repeats,
            "phase_repeat_count": phase_repeats,
            "repeat_source": repeat_source,
            "phase_count": len(phase_results),
            "energy_window_count": total_windows,
            "valid_energy_window_count": valid_windows,
            "sum_energy_total_j": total_energy if valid_windows else None,
            "dispatch_energy_total_j": total_energy if valid_windows else None,
            "dispatch_energy_window_count": total_windows,
            "dispatch_valid_energy_window_count": valid_windows,
            "dispatch_avg_energy_j_per_window": (float(total_energy) / float(valid_windows)) if valid_windows else None,
            "sum_energy_dynamic_j": total_dynamic if valid_dynamic else None,
            "sum_host_normalized_energy_est_j": total_host_norm if valid_host_norm else None,
            "avg_power_w_weighted": (weighted_power_num / weighted_power_den) if weighted_power_den > 0 else None,
            "energy_command_streaming_fps": primary_pipeline_fps,
            "energy_command_streaming_fps_source": "energy_command_phase_reference_not_row_pipeline_fps",
            "execution_measurement_mode": "remote_execution_only" if primary_remote_run_dir else "benchmark_remote_cli_fallback",
            "energy_measurement_scope": "command_energy",
            "energy_measurement_scope_note": "u.RECS windows measure the selected benchmark command phase; reports must distinguish this from future steady_state_energy loop-only measurements.",
            "phases": phase_results,
        }
        try:
            try:
                aggregate["energy_aggregate_relpath"] = (energy_root / "energy_aggregate.json").relative_to(group_dir).as_posix()
                aggregate["energy_summary_relpath"] = (energy_root / "energy_summary.json").relative_to(group_dir).as_posix()
            except Exception:
                pass
            (energy_root / "energy_aggregate.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
            (energy_root / "energy_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        if aggregate.get("avg_power_w_weighted") is not None:
            log(f"[energy] dispatch summary windows={total_windows} valid={valid_windows} weighted_avg_power={float(aggregate['avg_power_w_weighted']):.3f} W total_energy={aggregate.get('sum_energy_total_j')} J")
        return {
            "ok": bool(aggregate.get("ok")),
            "status": aggregate.get("status"),
            "setup_id": setup_id,
            "run_id": str(run_id or "all"),
            "energy": str(energy_root),
            "energy_out_dir": str(energy_root),
            "energy_aggregate": str(energy_root / "energy_aggregate.json"),
            "energy_summary": str(energy_root / "energy_summary.json"),
            "local_run_dir": str(group_dir),
            "phase_count": len(phase_results),
            "energy_window_count": total_windows,
            "valid_energy_window_count": valid_windows,
            "avg_power_w": aggregate.get("avg_power_w_weighted"),
            "dispatch_energy_total_j": aggregate.get("sum_energy_total_j"),
            "dispatch_energy_window_count": total_windows,
            "dispatch_valid_energy_window_count": valid_windows,
            "dispatch_avg_energy_j_per_window": (total_energy / valid_windows) if valid_windows else None,
            "energy_command_streaming_fps": primary_pipeline_fps,
            "energy_command_streaming_fps_source": "energy_command_phase_reference_not_row_pipeline_fps",
            "execution_measurement_mode": "remote_execution_only" if primary_remote_run_dir else "benchmark_remote_cli_fallback",
            "energy_measurement_scope": "command_energy",
            "energy_measurement_scope_note": "u.RECS windows measure the selected benchmark command phase; reports must distinguish this from future steady_state_energy loop-only measurements.",
            "phases": phase_results,
        }


    def _energy_run_uses_tensorrt(self, run_id: str, args: RemoteBenchmarkArgs) -> bool:
        """Return True when the measured run can trigger TensorRT engine build."""
        blob = " ".join([
            str(run_id or ""),
            str(getattr(args, "provider", "") or ""),
            str(getattr(args, "add_args", "") or ""),
        ]).lower()
        return any(tok in blob for tok in ("tensorrt", " trt", "_trt", "trt_", "ort_tensorrt"))

    def _energy_extract_phase(self, energy_aggregate: Mapping[str, Any], phase: str) -> dict[str, Any] | None:
        for p in energy_aggregate.get("phases") or []:
            if isinstance(p, dict) and str(p.get("phase") or "").lower() == phase.lower():
                return dict(p)
        return None

    def _energy_avg_from_phase(self, phase: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(phase, Mapping):
            return {}
        runs_l = phase.get("runs") or []
        durations = []
        try:
            for r in runs_l:
                if isinstance(r, Mapping) and isinstance(r.get("collector_duration_s"), (int, float)):
                    durations.append(float(r.get("collector_duration_s")))
        except Exception:
            durations = []
        avg_window_duration = (sum(durations) / len(durations)) if durations else phase.get("avg_window_duration_s")
        out = {
            "energy_total_j": phase.get("avg_energy_total_j"),
            "avg_power_w": phase.get("avg_power_w"),
            "energy_dynamic_j": phase.get("avg_energy_dynamic_j"),
            "host_normalized_energy_est_j": phase.get("avg_host_normalized_energy_est_j"),
            "host_normalized_energy_per_work_unit_est_j": phase.get("avg_host_normalized_energy_per_work_unit_est_j"),
            "host_normalized_energy_per_work_est_j_sample_stddev": phase.get("host_normalized_energy_per_work_unit_est_j_sample_stddev"),
            "host_normalized_energy_per_work_est_j_ci_low": phase.get("host_normalized_energy_per_work_unit_est_j_ci_low"),
            "host_normalized_energy_per_work_est_j_ci_high": phase.get("host_normalized_energy_per_work_unit_est_j_ci_high"),
            "host_normalized_average_power_est_w": phase.get("avg_host_normalized_average_power_est_w"),
            "host_normalization_role": phase.get("host_normalization_role"),
            "host_normalization_source_run_id": phase.get("host_normalization_source_run_id"),
            "host_normalization_target_variant": phase.get("host_normalization_target_variant"),
            "host_normalization_identity_verified": phase.get("host_normalization_identity_verified"),
            "accelerator_idle_correction_requested": phase.get("accelerator_idle_correction_requested"),
            "accelerator_idle_correction_applied": phase.get("accelerator_idle_correction_applied"),
            "accelerator_idle_correction_statuses": phase.get("accelerator_idle_correction_statuses"),
            "accelerator_idle_w_applied": phase.get("accelerator_idle_w_applied"),
            "accelerator_idle_calibration_verified": phase.get("accelerator_idle_calibration_verified"),
            "accelerator_idle_calibration_status": phase.get("accelerator_idle_calibration_status"),
            "accelerator_idle_calibration_binding_path": phase.get("accelerator_idle_calibration_binding_path"),
            "accelerator_idle_calibration_binding_sha256": phase.get("accelerator_idle_calibration_binding_sha256"),
            "accelerator_idle_calibration_evidence": phase.get("accelerator_idle_calibration_evidence"),
            "accelerator_idle_calibrated_at": phase.get("accelerator_idle_calibrated_at"),
            "energy_efficiency_claim_eligible": phase.get("energy_efficiency_claim_eligible"),
            "energy_per_inference_j": phase.get("avg_energy_per_configured_work_unit_j") or phase.get("avg_energy_per_inference_j"),
            "energy_per_configured_work_unit_j": phase.get("avg_energy_per_configured_work_unit_j"),
            "configured_work_units_per_j": phase.get("configured_work_units_per_j"),
            "energy_efficiency_source": phase.get("energy_efficiency_source"),
            "energy_command_reference_fps": phase.get("energy_command_reference_fps"),
            "energy_command_reference_fps_source": phase.get("energy_command_reference_fps_source"),
            "energy_streaming_fps_per_watt_from_command_fps": phase.get("energy_streaming_fps_per_watt_from_command_fps"),
            "energy_streaming_j_per_frame_from_command_fps": phase.get("energy_streaming_j_per_frame_from_command_fps"),
            # Legacy names may exist in old energy_aggregate files; reports should
            # prefer row-local recomputation after merge.
            "pipeline_fps_per_watt_from_selected_fps": phase.get("pipeline_fps_per_watt_from_selected_fps"),
            "energy_per_pipeline_frame_from_selected_fps_j": phase.get("energy_per_pipeline_frame_from_selected_fps_j"),
            "energy_command_streaming_fps": phase.get("energy_command_streaming_fps"),
            "energy_command_fps_per_watt": phase.get("energy_command_fps_per_watt"),
            "energy_command_j_per_frame": phase.get("energy_command_j_per_frame"),
            "energy_measured_work_units_per_s": phase.get("avg_energy_measured_work_units_per_s") or phase.get("energy_measured_work_units_per_s"),
            "energy_active_work_units_per_s": phase.get("avg_energy_active_work_units_per_s") or phase.get("energy_active_work_units_per_s"),
            "energy_work_units_per_j": phase.get("avg_energy_work_units_per_j") or phase.get("energy_work_units_per_j"),
            "phase_repeat_count": phase.get("phase_repeat_count") or phase.get("run_count"),
            "work_units_per_window": phase.get("work_units_per_window"),
            "valid_postprocessed_runs": phase.get("valid_postprocessed_runs"),
            "energy_window_count": phase.get("energy_window_count") or phase.get("run_count") or (len(runs_l) if isinstance(runs_l, list) else None),
            "avg_window_duration_s": avg_window_duration,
        }
        return {k: v for k, v in out.items() if v is not None}

    def _energy_artifact_relpath(self, path: Path | str) -> str | None:
        """Return a portable relative path for energy artifacts inside result bundles."""
        try:
            p = Path(path)
            parts = p.parts
            for i, part in enumerate(parts):
                if str(part).startswith("remote_"):
                    return str(Path(*parts[i:]))
            # Fallback: use the final Results/<benchmark>/<remote...> suffix if present.
            if "Results" in parts:
                i = parts.index("Results")
                return str(Path(*parts[i + 1:]))
            return str(p.name)
        except Exception:
            return None

    def _energy_relative_result_path(self, path: Path | str) -> str | None:
        """Return a portable path relative to the remote_* result bundle root.

        Absolute local paths are useful while running the GUI, but DebugPack/ZIP
        consumers need a stable in-bundle reference.  Prefer the path below the
        first remote_* component, e.g.
        remote_.../setup_run/energy/... -> setup_run/energy/... .
        """
        try:
            p = Path(path).expanduser()
            parts = list(p.parts)
            for i, part in enumerate(parts):
                if str(part).startswith("remote_") and i + 1 < len(parts):
                    return str(Path(*parts[i+1:]))
            # Fallback: if no remote_* component is present, keep the last few
            # components rather than returning a host-specific absolute path.
            return str(Path(*parts[-4:])) if len(parts) >= 4 else str(p)
        except Exception:
            return None

    def _energy_merge_payload(self, *, setup_id: str, run_id: str, energy_aggregate: Mapping[str, Any], aggregate_path: Path, result_root: Path | None = None) -> dict[str, Any]:
        latency = self._energy_extract_phase(energy_aggregate, "latency")
        streaming = self._energy_extract_phase(energy_aggregate, "streaming")
        lat = self._energy_avg_from_phase(latency)
        stream = self._energy_avg_from_phase(streaming)
        try:
            _energy_relpath = Path(aggregate_path).resolve().relative_to(Path(result_root).resolve()).as_posix() if result_root is not None else self._energy_artifact_relpath(aggregate_path)
        except Exception:
            _energy_relpath = self._energy_artifact_relpath(aggregate_path)
        payload: dict[str, Any] = {
            "energy_enabled": True,
            "energy_source": "urecs_fast_firmware",
            "energy_setup_id": setup_id,
            "energy_run_id": run_id,
            "energy_aggregate_path": str(aggregate_path),
            "energy_aggregate_relpath": _energy_relpath,
            "energy_window_count": energy_aggregate.get("energy_window_count"),
            "valid_energy_window_count": energy_aggregate.get("valid_energy_window_count"),
            "energy_measurement_mode": energy_aggregate.get("execution_measurement_mode"),
            "energy_repeat_source": energy_aggregate.get("repeat_source"),
            "energy_phase_repeat_count": energy_aggregate.get("phase_repeat_count"),
            "energy_measurement_scope": energy_aggregate.get("energy_measurement_scope") or "command_energy",
            "energy_measurement_scope_note": energy_aggregate.get("energy_measurement_scope_note") or "u.RECS window covers the selected remote benchmark command phase; benchmark-owned pipeline metrics are not overwritten by energy merge",
            "dispatch_energy_total_j": energy_aggregate.get("sum_energy_total_j"),
            "dispatch_energy_window_count": energy_aggregate.get("energy_window_count"),
            "dispatch_valid_energy_window_count": energy_aggregate.get("valid_energy_window_count"),
            "dispatch_avg_energy_j_per_window": (float(energy_aggregate.get("sum_energy_total_j") or 0.0) / float(energy_aggregate.get("valid_energy_window_count") or 0.0)) if energy_aggregate.get("valid_energy_window_count") else None,
            "energy_total_dispatch_j": energy_aggregate.get("sum_energy_total_j"),
            "dispatch_energy_total_j": energy_aggregate.get("sum_energy_total_j"),
            "dispatch_energy_window_count": energy_aggregate.get("energy_window_count"),
            "dispatch_valid_energy_window_count": energy_aggregate.get("valid_energy_window_count"),
            "dispatch_avg_energy_j_per_window": (float(energy_aggregate.get("sum_energy_total_j") or 0.0) / float(energy_aggregate.get("valid_energy_window_count") or 0)) if energy_aggregate.get("sum_energy_total_j") is not None and energy_aggregate.get("valid_energy_window_count") else None,
            "avg_power_dispatch_w": energy_aggregate.get("avg_power_w_weighted"),
        }
        # Generic fields use the latency window because they correspond to the
        # configured normal benchmark runs.  Streaming-specific energy fields are
        # kept separate below.
        if lat:
            payload.update({
                "energy_phase": "latency",
                "energy_total_j": lat.get("energy_total_j"),
                "avg_power_w": lat.get("avg_power_w"),
                "energy_dynamic_j": lat.get("energy_dynamic_j"),
                "host_normalized_energy_est_j": lat.get("host_normalized_energy_est_j"),
                "row_host_normalized_energy_latency_j_per_inference_est": lat.get("host_normalized_energy_per_work_unit_est_j"),
                "host_normalized_energy_per_work_est_j_sample_stddev": lat.get("host_normalized_energy_per_work_est_j_sample_stddev"),
                "host_normalized_energy_per_work_est_j_ci_low": lat.get("host_normalized_energy_per_work_est_j_ci_low"),
                "host_normalized_energy_per_work_est_j_ci_high": lat.get("host_normalized_energy_per_work_est_j_ci_high"),
                "host_normalized_average_power_est_w": lat.get("host_normalized_average_power_est_w"),
                "energy_per_inference_j": lat.get("energy_per_inference_j"),
                "energy_per_configured_work_unit_j": lat.get("energy_per_configured_work_unit_j"),
                "configured_work_units_per_j": lat.get("configured_work_units_per_j"),
                "energy_latency_total_j": lat.get("energy_total_j"),
                "energy_latency_avg_power_w": lat.get("avg_power_w"),
                "energy_latency_per_inference_j": lat.get("energy_per_inference_j"),
                "energy_latency_work_units_per_window": lat.get("work_units_per_window"),
                "energy_latency_avg_window_duration_s": lat.get("avg_window_duration_s"),
                "energy_latency_window_count": lat.get("energy_window_count"),
                "energy_latency_command_work_units_per_s": lat.get("energy_command_work_units_per_s") or lat.get("energy_measured_work_units_per_s"),
                "row_energy_latency_total_j": lat.get("energy_total_j"),
                "row_energy_latency_j_per_inference": lat.get("energy_per_inference_j"),
            })
        if stream:
            payload.update({
                "energy_streaming_total_j": stream.get("energy_total_j"),
                "energy_streaming_avg_power_w": stream.get("avg_power_w"),
                "host_normalized_streaming_avg_power_est_w": stream.get("host_normalized_average_power_est_w"),
                "energy_streaming_per_frame_j": stream.get("energy_per_inference_j"),
                "energy_command_streaming_fps": stream.get("energy_command_streaming_fps") or stream.get("energy_command_reference_pipeline_fps"),
                "energy_command_streaming_fps_source": stream.get("energy_command_reference_fps_source") or ("phase_energy_command_reference" if stream.get("energy_command_streaming_fps") is not None else None),
                "energy_command_fps_per_watt": stream.get("energy_command_fps_per_watt"),
                "energy_command_j_per_frame": stream.get("energy_command_j_per_frame"),
                "energy_streaming_work_units_per_window": stream.get("work_units_per_window"),
                "energy_streaming_avg_window_duration_s": stream.get("avg_window_duration_s"),
                "energy_streaming_window_count": stream.get("energy_window_count"),
                "energy_streaming_command_work_units_per_s": stream.get("energy_command_work_units_per_s") or stream.get("energy_measured_work_units_per_s"),
                "row_energy_streaming_total_j": stream.get("energy_total_j"),
                "row_energy_streaming_j_per_frame": stream.get("energy_per_inference_j"),
                "row_host_normalized_energy_streaming_j_per_frame_est": stream.get("host_normalized_energy_per_work_unit_est_j"),
                "host_normalized_energy_per_work_est_j_sample_stddev": stream.get("host_normalized_energy_per_work_est_j_sample_stddev"),
                "host_normalized_energy_per_work_est_j_ci_low": stream.get("host_normalized_energy_per_work_est_j_ci_low"),
                "host_normalized_energy_per_work_est_j_ci_high": stream.get("host_normalized_energy_per_work_est_j_ci_high"),
                "energy_efficiency_source": stream.get("energy_efficiency_source"),
                "energy_measured_work_units_per_s": stream.get("energy_measured_work_units_per_s"),
                "energy_active_work_units_per_s": stream.get("energy_active_work_units_per_s"),
                "energy_work_units_per_j": stream.get("energy_work_units_per_j"),
                "pipeline_fps_per_watt_from_selected_fps": stream.get("pipeline_fps_per_watt_from_selected_fps"),
                "energy_per_pipeline_frame_from_selected_fps_j": stream.get("energy_per_pipeline_frame_from_selected_fps_j"),
            })
        normalization = stream or lat
        for key in (
            "host_normalization_role", "host_normalization_source_run_id",
            "host_normalization_target_variant", "host_normalization_identity_verified",
            "accelerator_idle_correction_requested", "accelerator_idle_correction_applied",
            "accelerator_idle_correction_statuses", "accelerator_idle_w_applied",
            "accelerator_idle_calibration_verified", "accelerator_idle_calibration_status",
            "accelerator_idle_calibration_binding_path", "accelerator_idle_calibration_binding_sha256",
            "accelerator_idle_calibration_evidence", "accelerator_idle_calibrated_at",
            "energy_efficiency_claim_eligible",
        ):
            if normalization.get(key) is not None:
                payload[key] = normalization.get(key)
        # v57d: never overwrite benchmark/pipeline fields during the
        # energy merge.  pipeline_fps_selected and pipeline_cycle_selected_ms
        # are performance metrics owned by the benchmark runner.  Energy can
        # add FPS/W and J/frame diagnostics, but those must live under
        # energy_* names so case-specific pipeline values are not corrupted.
        if streaming:
            for src, dst in (
                ("energy_command_reference_fps", "energy_streaming_command_reference_fps"),
                ("energy_command_fps_per_watt_from_reference_fps", "energy_streaming_fps_per_watt_from_command_reference_fps"),
                ("energy_command_j_per_frame_from_reference_fps", "energy_streaming_j_per_frame_from_command_reference_fps"),
                ("energy_efficiency_source", "energy_efficiency_source"),
                ("pipeline_fps_selected", "energy_streaming_reference_pipeline_fps_deprecated"),
                ("energy_per_pipeline_frame_from_selected_fps_j", "energy_streaming_j_per_frame_from_selected_fps"),
                ("pipeline_fps_per_watt_from_selected_fps", "energy_streaming_fps_per_watt_from_selected_fps"),
            ):
                if streaming.get(src) is not None:
                    payload[dst] = streaming.get(src)
            measured_frames_per_j = streaming.get("energy_work_units_per_j") or streaming.get("energy_command_work_units_per_j")
            measured_work_units_per_s = streaming.get("energy_measured_work_units_per_s")
            if measured_frames_per_j is not None:
                payload["energy_streaming_frames_per_j"] = measured_frames_per_j
                # v57e/v57l clearer aliases: these are u.RECS-window work units
                # per joule, not necessarily the benchmark pipeline_fps_selected
                # divided by watt.  For full baselines they represent full
                # inferences/J; for split rows they represent composed frames/J.
                payload["energy_window_work_units_per_j"] = measured_frames_per_j
            if measured_work_units_per_s is not None:
                payload["energy_streaming_work_units_per_s"] = measured_work_units_per_s
                payload["energy_window_work_units_per_s"] = measured_work_units_per_s
        payload.update(resolve_energy_comparison(payload))
        return {k: v for k, v in payload.items() if v is not None}

    def _float_or_none(self, value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    def _energy_recompute_row_derived_fields(self, row: dict[str, Any]) -> None:
        """Recompute row-local derived energy diagnostics after merging.

        v57f: the measured u.RECS row must keep benchmark-owned pipeline
        metrics intact.  Derived "from selected FPS" energy diagnostics must be
        recomputed from *this row's* pipeline_fps_selected and this row's
        streaming power, never copied from the canonical energy target or another
        split case.
        """
        # v57o: choose the reference FPS according to throughput semantics.
        # Full-baseline energy rows must not use same-backend composed pipeline
        # diagnostics as their selected FPS.  Heterogeneous split rows use the
        # row-owned pipeline_fps_selected.
        fps = self._float_or_none(row.get("pipeline_fps_selected"))
        fps_source = "pipeline_fps_selected"
        target_variant = str(row.get("energy_target_variant") or row.get("variant") or row.get("primary_variant") or "").lower()
        if target_variant == "full" or row.get("energy_applies_to_all_cases") is True:
            full_fps = self._float_or_none(row.get("full_backend_throughput_fps"))
            full_ms_for_full = self._float_or_none(row.get("full_mean_ms") or row.get("full_e2e_mean_ms") or row.get("total_latency_ms"))
            if full_fps is None:
                if full_ms_for_full and full_ms_for_full > 0:
                    full_fps = 1000.0 / full_ms_for_full
            if full_fps and full_fps > 0:
                fps = full_fps
                fps_source = "full_backend_throughput_fps"
                # v57p: rows produced by same-backend diagnostic runs (for
                # example ort_tensorrt) can contain both full-model and composed
                # timings.  When the energy target is the canonical full
                # baseline, the row-level primary throughput must be the full
                # backend throughput.  Preserve the previous composed/pipeline
                # values as diagnostics instead of letting them masquerade as
                # the full baseline FPS.
                old_pipe_fps = self._float_or_none(row.get("pipeline_fps_selected"))
                old_pipe_cycle = self._float_or_none(row.get("pipeline_cycle_selected_ms"))
                if old_pipe_fps is not None and abs(old_pipe_fps - full_fps) > 1e-9:
                    row.setdefault("same_backend_composed_fps", old_pipe_fps)
                    row.setdefault("same_backend_composed_fps_source", "pre_energy_merge_pipeline_fps_selected")
                    row.setdefault("diagnostic_pipeline_fps_selected", old_pipe_fps)
                if old_pipe_cycle is not None and full_ms_for_full is not None and abs(old_pipe_cycle - full_ms_for_full) > 1e-9:
                    row.setdefault("same_backend_composed_cycle_ms", old_pipe_cycle)
                    row.setdefault("diagnostic_pipeline_cycle_selected_ms", old_pipe_cycle)
                row["full_backend_throughput_fps"] = full_fps
                row["throughput_kind"] = "full_backend"
                row["pipeline_applicable"] = False
                row["throughput_primary_fps"] = full_fps
                row["throughput_primary_metric"] = "full_backend_throughput_fps"
                row["throughput_primary_source"] = "full_latency_fps"
                row["throughput_primary_metric_note"] = "canonical full-backend throughput; same-backend composed diagnostics are kept in diagnostic_* fields"
                row["pipeline_fps_selected"] = full_fps
                if full_ms_for_full and full_ms_for_full > 0:
                    row["pipeline_cycle_selected_ms"] = full_ms_for_full
                row["pipeline_cycle_source"] = "full_latency_fps_compat"
                row["pipeline_note"] = "Full backend repeated-feed throughput; not a heterogeneous split pipeline."
        p_stream = self._float_or_none(row.get("energy_streaming_avg_power_w"))
        if fps is not None and fps > 0 and p_stream is not None and p_stream > 0:
            fps_per_w = fps / p_stream
            j_per_frame = p_stream / fps
            row["energy_streaming_reference_fps"] = fps
            row["energy_streaming_reference_fps_source"] = fps_source
            row["energy_streaming_fps_per_watt_from_reference_fps"] = fps_per_w
            row["energy_streaming_j_per_frame_from_reference_fps"] = j_per_frame
            # Backwards-compatible aliases.  They are now derived from the
            # semantically correct reference FPS for this row: full-backend FPS for
            # full energy rows, pipeline FPS for heterogeneous split rows.
            row["energy_streaming_reference_pipeline_fps"] = fps
            row["energy_streaming_fps_per_watt_from_selected_fps"] = fps_per_w
            row["energy_streaming_j_per_frame_from_selected_fps"] = j_per_frame
            row["pipeline_fps_per_watt_from_selected_fps"] = fps_per_w
            row["energy_per_pipeline_frame_from_selected_fps_j"] = j_per_frame
        # v57f: legacy model-estimated link-energy fields must not be confused
        # with measured u.RECS energy.  Preserve the old field for compatibility,
        # but add explicit aliases/semantics used by reports.
        if row.get("energy_total_mJ") is not None:
            row.setdefault("link_model_total_energy_mJ", row.get("energy_total_mJ"))
            row.setdefault("energy_total_mJ_semantics", "legacy_link_model_energy_mJ_not_urecs_measured_energy")
        if row.get("link_energy_mJ") is not None:
            row.setdefault("link_model_link_energy_mJ", row.get("link_energy_mJ"))
        for legacy_key, alias_key in (
            ("energy_left_mJ", "link_model_left_energy_mJ"),
            ("energy_right_mJ", "link_model_right_energy_mJ"),
        ):
            if row.get(legacy_key) is not None:
                row.setdefault(alias_key, row.get(legacy_key))

    def _write_benchmark_results_csv_from_json_obj(self, csv_path: Path, obj: Any) -> bool:
        """Rewrite benchmark_results_*.csv after JSON energy merge.

        Several downstream reports read CSVs rather than JSON.  Before v57f the
        JSON rows contained u.RECS fields, while the sibling CSV stayed stale.
        """
        try:
            if isinstance(obj, list):
                rows = [r for r in obj if isinstance(r, dict)]
            elif isinstance(obj, dict):
                rows = []
                for key in ("rows", "results", "measurements", "records"):
                    if isinstance(obj.get(key), list):
                        rows = [r for r in obj.get(key) if isinstance(r, dict)]
                        break
                if not rows:
                    rows = [obj]
            else:
                return False
            if not rows:
                return False
            keys: list[str] = []
            seen: set[str] = set()
            # Prefer stable identifying columns first, then append all energy and
            # remaining fields in discovery order.
            preferred = [
                "model_id", "run_id", "backend", "provider", "case_id", "boundary", "variant",
                "final_pass", "validation_ok", "runtime_ok", "full_mean_ms", "composed_mean_ms",
                "pipeline_fps_selected", "pipeline_cycle_selected_ms",
                "energy_target_case", "energy_target_variant", "energy_measurement_scope",
                "energy_total_j", "avg_power_w", "energy_per_inference_j",
                "row_energy_streaming_j_per_frame", "energy_streaming_avg_power_w",
                "energy_work_units_per_j", "energy_streaming_fps_per_watt_from_selected_fps",
            ]
            for k in preferred:
                for r in rows:
                    if k in r and k not in seen:
                        seen.add(k); keys.append(k); break
            for r in rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k); keys.append(k)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
            return True
        except Exception:
            return False

    def _normalize_result_row_semantics_for_energy(self, row: dict[str, Any]) -> bool:
        """Normalize post-merge result row semantics used by reports/CSVs.

        v57s: Keep u.RECS measured energy separate from legacy link-model
        energy, and provide a uniform semantic_validation_ok flag across full
        and split rows.  The function intentionally does not turn invalid rows
        into valid ones; it only mirrors already present pass/fail evidence into
        stable schema fields.
        """
        changed = False
        # Normalize semantic validation across runner generations.
        if row.get("semantic_validation_ok") is None:
            for key in (
                "semantic_validation_passed_all",
                "semantic_validation_passed",
                "semantic_pass",
                "semantic_validation_pass",
                "classification_semantic_pass",
                "detection_semantic_pass",
            ):
                if key in row and row.get(key) is not None:
                    row["semantic_validation_ok"] = bool(row.get(key))
                    changed = True
                    break
        if row.get("semantic_validation_passed") is None and row.get("semantic_validation_ok") is not None:
            row["semantic_validation_passed"] = bool(row.get("semantic_validation_ok")); changed = True
        if row.get("semantic_validation_passed_all") is None and row.get("semantic_validation_ok") is not None:
            row["semantic_validation_passed_all"] = bool(row.get("semantic_validation_ok")); changed = True
        # Clarify that old *_mJ fields are model/link estimates, not measured u.RECS energy.
        if row.get("energy_total_mJ") is not None:
            if row.get("link_model_total_energy_mJ") is None:
                row["link_model_total_energy_mJ"] = row.get("energy_total_mJ"); changed = True
            if row.get("energy_total_mJ_semantics") != "legacy_link_model_energy_mJ_not_urecs_measured_energy":
                row["energy_total_mJ_semantics"] = "legacy_link_model_energy_mJ_not_urecs_measured_energy"; changed = True
            if row.get("energy_total_mJ_deprecated") is not True:
                row["energy_total_mJ_deprecated"] = True; changed = True
        if row.get("link_energy_mJ") is not None and row.get("link_model_link_energy_mJ") is None:
            row["link_model_link_energy_mJ"] = row.get("link_energy_mJ"); changed = True
        for legacy_key, alias_key in (
            ("energy_left_mJ", "link_model_left_energy_mJ"),
            ("energy_right_mJ", "link_model_right_energy_mJ"),
        ):
            if row.get(legacy_key) is not None and row.get(alias_key) is None:
                row[alias_key] = row.get(legacy_key); changed = True
        return changed

    def _normalize_benchmark_results_file_after_energy(self, json_path: Path) -> int:
        """Normalize all rows in a benchmark_results*.json file and refresh CSV."""
        def _rows_container(obj: Any):
            if isinstance(obj, list):
                return obj, None
            if isinstance(obj, dict):
                for key in ("rows", "results", "measurements", "records"):
                    val = obj.get(key)
                    if isinstance(val, list):
                        return val, key
                return [obj], None
            return [], None
        try:
            obj = json.loads(Path(json_path).read_text(encoding="utf-8"))
        except Exception:
            return 0
        rows, _ = _rows_container(obj)
        changed = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if self._normalize_result_row_semantics_for_energy(row):
                changed += 1
            # Recompute selected-FPS-derived diagnostics after any row semantic cleanup.
            try:
                self._energy_recompute_row_derived_fields(row)
            except Exception:
                pass
        if changed:
            try:
                Path(json_path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        try:
            self._write_benchmark_results_csv_from_json_obj(Path(json_path).with_suffix(".csv"), obj)
        except Exception:
            pass
        return changed

    def _update_energy_run_metadata_files(self, *, child_dir: Path, run_id: str, setup_id: str) -> dict[str, Any]:
        """Update post-merge run_meta/run_results metadata to match energy-augmented results."""
        out: dict[str, Any] = {"updated": []}
        objective = "remote benchmark with energy"
        for name in ("run_meta.json", "run_results.json", "run_status.json"):
            fp = child_dir / name
            if not fp.exists():
                continue
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as exc:
                out.setdefault("errors", []).append({"file": str(fp), "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not isinstance(obj, dict):
                continue
            raw_objective = obj.get("objective")
            if raw_objective and str(raw_objective).lower() != objective:
                obj.setdefault("objective_raw", raw_objective)
            obj["objective"] = objective
            obj["dispatch_energy_enabled"] = True
            obj["energy_enabled"] = True
            obj.setdefault("energy_measurement_scope", "command_energy")
            obj["energy_setup_id"] = setup_id
            obj["energy_run_id"] = run_id
            if isinstance(obj.get("args"), dict):
                obj["args"]["energy_enabled"] = True
                obj["args"]["dispatch_energy_enabled"] = True
                obj["args"].setdefault("energy_measurement_scope", obj.get("energy_measurement_scope") or "command_energy")
                obj["args"].setdefault("energy_setup_id", setup_id)
            try:
                fp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                out["updated"].append(str(fp))
            except Exception as exc:
                out.setdefault("errors", []).append({"file": str(fp), "error": f"{type(exc).__name__}: {exc}"})
        return out

    def _repack_results_bundles_after_energy_merge(self, *, child_dir: Path, log: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Rebuild results_bundle*.tar.gz after energy merge.

        Remote runs originally package results before local u.RECS energy is
        merged.  Without repacking, results_bundle.tar.gz and the lean bundle
        contain stale pre-merge JSON/CSV rows.  v57s rebuilds both bundles from
        the post-merge local results directory.
        """
        out: dict[str, Any] = {"attempted": False, "rebuilt": []}
        results_dir = child_dir / "results"
        if not results_dir.exists():
            out["reason"] = "results_dir_missing"
            return out
        try:
            from ..benchmark.results_bundle import create_results_bundle_from_results_dir
        except Exception as exc:
            out["error"] = f"import_failed: {type(exc).__name__}: {exc}"
            return out
        out["attempted"] = True
        for fname, mode in (("results_bundle.tar.gz", "full"), ("results_bundle_lean.tar.gz", "lean")):
            target = child_dir / fname
            try:
                create_results_bundle_from_results_dir(results_dir, target, mode=mode)
                out["rebuilt"].append(str(target))
                if log:
                    log(f"[energy] rebuilt {fname} after energy merge")
            except Exception as exc:
                out.setdefault("errors", []).append({"bundle": str(target), "mode": mode, "error": f"{type(exc).__name__}: {exc}"})
                if log:
                    log(f"[energy][warn] failed to rebuild {fname}: {type(exc).__name__}: {exc}")
        return out

    def _post_energy_merge_artifact_consistency(self, *, group_dir: Path, child_run_id: str, run_id: str, setup_id: str, merged_files: list[str] | None = None, log: Callable[[str], None] | None = None) -> dict[str, Any]:
        """Make local result artifacts consistent after energy merge.

        v57s fixes artifact consistency: post-merge JSON/CSV rows, run metadata,
        and results_bundle*.tar.gz all describe the same energy-augmented data.
        """
        child_dir = Path(group_dir) / str(child_run_id)
        out: dict[str, Any] = {
            "schema": "onnx-splitpoint/post-energy-merge-consistency",
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "child_run_id": child_run_id,
            "run_id": run_id,
            "setup_id": setup_id,
            "child_dir": str(child_dir),
        }
        # Normalize every benchmark_results JSON in this child after merge. This
        # also refreshes sibling CSVs, including rows not directly touched by a
        # canonical-full energy target.
        normalized: list[str] = []
        for fp in sorted(child_dir.rglob("benchmark_results*.json")):
            changed = self._normalize_benchmark_results_file_after_energy(fp)
            normalized.append(str(fp))
        out["normalized_result_files"] = normalized
        out["metadata"] = self._update_energy_run_metadata_files(child_dir=child_dir, run_id=run_id, setup_id=setup_id)
        out["bundle_repack"] = self._repack_results_bundles_after_energy_merge(child_dir=child_dir, log=log)
        try:
            diag = child_dir / "diagnostics"
            diag.mkdir(parents=True, exist_ok=True)
            (diag / "post_energy_merge_consistency.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return out

    def _merge_energy_payload_into_json(self, json_path: Path, payload: Mapping[str, Any], run_id: str, target_case: str = "", target_variant: str = "", apply_to_all_cases: bool = False) -> int:
        def _rows_container(obj: Any):
            if isinstance(obj, list):
                return obj, None
            if isinstance(obj, dict):
                for key in ("rows", "results", "measurements", "records"):
                    val = obj.get(key)
                    if isinstance(val, list):
                        return val, key
                return [obj], None
            return [], None

        try:
            obj = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        rows, key = _rows_container(obj)
        if not rows:
            return 0
        n = 0
        run_id_l = str(run_id or "").lower()
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_text = " ".join(str(row.get(k) or "") for k in ("run_id", "backend", "tag", "provider", "name")).lower()
            # If no explicit run_id can be found in the row, trust the file name.
            if run_id_l and run_id_l not in row_text and run_id_l not in json_path.stem.lower():
                continue
            if target_case and not apply_to_all_cases:
                wanted = str(target_case).strip().lower()
                labels = set()
                for k in ("case_id", "case", "folder", "case_dir"):
                    if row.get(k) is not None:
                        labels.add(str(row.get(k)).strip().lower())
                b = row.get("boundary")
                if b is not None:
                    try:
                        labels.add(f"b{int(b):03d}")
                        labels.add(str(int(b)))
                    except Exception:
                        labels.add(str(b).strip().lower())
                if wanted not in labels and not (wanted == "full" and (str(row.get("variant") or row.get("primary_variant") or "").lower() == "full" or str(row.get("case_id") or "").lower() == "full")):
                    continue
            if apply_to_all_cases and target_variant == "full":
                # Only apply canonical full energy to rows that actually contain
                # a full-baseline measurement.  Do not stamp it onto unrelated
                # split-only result files.
                if row.get("full_mean_ms") is None and str(row.get("variant") or row.get("primary_variant") or "").lower() != "full":
                    continue
            # Energy merge must not corrupt benchmark-owned pipeline metrics.
            protected_pipeline_keys = {
                "pipeline_fps_selected", "pipeline_cycle_selected_ms", "pipeline_cycle_source",
                "pipeline_stage1_lane_ms", "pipeline_stage2_lane_ms", "pipeline_transfer_est_ms",
            }
            safe_payload = {k: v for k, v in dict(payload).items() if k not in protected_pipeline_keys}
            # v57l: keep portable, zip-relative references next to absolute paths.
            # The absolute path is convenient on the generating workstation, while
            # the relpath lets debug-pack/zip consumers find the same artifact after
            # extraction without depending on /home/kmika/... paths.
            try:
                agg_abs = safe_payload.get("energy_aggregate_path")
                if agg_abs:
                    safe_payload["energy_aggregate_relpath"] = os.path.relpath(str(agg_abs), start=str(json_path.parent))
            except Exception:
                pass
            try:
                summ_abs = safe_payload.get("energy_summary_path") or safe_payload.get("energy_summary")
                if summ_abs:
                    safe_payload["energy_summary_relpath"] = os.path.relpath(str(summ_abs), start=str(json_path.parent))
            except Exception:
                pass
            row.update(safe_payload)
            self._normalize_result_row_semantics_for_energy(row)
            self._energy_recompute_row_derived_fields(row)
            if target_case:
                row["energy_target_case"] = str(target_case)
            if target_variant:
                row["energy_target_variant"] = str(target_variant)
            if row.get("energy_aggregate_path") and not row.get("energy_aggregate_relpath"):
                row["energy_aggregate_relpath"] = self._energy_relative_result_path(row.get("energy_aggregate_path"))
            n += 1
        if n:
            try:
                json_path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
                self._write_benchmark_results_csv_from_json_obj(json_path.with_suffix(".csv"), obj)
            except Exception:
                return 0
        return n

    def _merge_energy_into_benchmark_results(self, *, group_dir: Path, child_run_id: str, run_id: str, setup_id: str, energy_aggregate_path: Path, log: Callable[[str], None] | None = None) -> dict[str, Any]:
        try:
            energy_aggregate = json.loads(Path(energy_aggregate_path).read_text(encoding="utf-8"))
        except Exception as exc:
            return {"ok": False, "error": f"cannot read energy aggregate: {type(exc).__name__}: {exc}"}
        target_dirs = [group_dir / child_run_id, group_dir / f"{child_run_id}_latency", group_dir / f"{child_run_id}_streaming"]
        files: list[Path] = []
        for td in target_dirs:
            if td.exists():
                files.extend(td.rglob("benchmark_results*.json"))
        # Some versions package results under child_run_id/results only; keep a broader fallback.
        if not files and group_dir.exists():
            files = list(group_dir.rglob(f"benchmark_results*{run_id}*.json"))
        touched = []
        rows = 0
        payload_keys_all: set[str] = set()

        if bool(energy_aggregate.get("row_scope")):
            # v56s: row-level dispatch summary.  Merge each target window into
            # only the matching case row and annotate whether it measured full
            # or composed.
            def _synth_target_aggregate(tres_obj: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
                # v57f: tolerate partial targets where latency/streaming phase
                # aggregates exist but the root energy_aggregate.json was not
                # written or was not packaged.  Synthesize the root aggregate so
                # result rows never point to a missing artifact.
                root_s = str(tres_obj.get("energy") or tres_obj.get("energy_out_dir") or "").strip()
                if not root_s:
                    agg_s = str(tres_obj.get("energy_aggregate") or tres_obj.get("energy_summary") or "").strip()
                    if agg_s:
                        root_s = str(Path(agg_s).parent)
                if not root_s:
                    return None, None
                root = Path(root_s)
                phases: list[dict[str, Any]] = []
                for pname in ("latency", "streaming"):
                    fp = root / pname / "energy_aggregate.json"
                    if fp.exists():
                        try:
                            ph_obj = json.loads(fp.read_text(encoding="utf-8"))
                            if isinstance(ph_obj, dict):
                                ph_obj.setdefault("phase", pname)
                                phases.append(ph_obj)
                        except Exception:
                            pass
                if not phases:
                    return None, None
                total_windows = 0
                valid_windows = 0
                total_energy = 0.0
                power_num = 0.0
                power_den = 0.0
                for ph in phases:
                    for rr in ph.get("runs") or []:
                        if not isinstance(rr, dict):
                            continue
                        total_windows += 1
                        en = rr.get("energy_total_j")
                        if isinstance(en, (int, float)):
                            valid_windows += 1
                            total_energy += float(en)
                        dur = rr.get("collector_duration_s")
                        pw = rr.get("avg_power_w")
                        if isinstance(pw, (int, float)) and isinstance(dur, (int, float)) and float(dur) > 0:
                            power_num += float(pw) * float(dur)
                            power_den += float(dur)
                out = {
                    "schema": "onnx-splitpoint/energy-target-aggregate",
                    "schema_version": 1,
                    "synthesized_from_phase_aggregates": True,
                    "ok": valid_windows > 0,
                    "status": "ok" if total_windows and valid_windows == total_windows else ("partial" if valid_windows else "failed"),
                    "setup_id": setup_id,
                    "run_id": run_id,
                    "out_dir": str(root),
                    "energy_window_count": total_windows,
                    "valid_energy_window_count": valid_windows,
                    "sum_energy_total_j": total_energy if valid_windows else None,
                    "dispatch_energy_total_j": total_energy if valid_windows else None,
                    "dispatch_energy_window_count": total_windows,
                    "dispatch_valid_energy_window_count": valid_windows,
                    "dispatch_avg_energy_j_per_window": (float(total_energy) / float(valid_windows)) if valid_windows else None,
                    "avg_power_w_weighted": (power_num / power_den) if power_den > 0 else None,
                    "execution_measurement_mode": energy_aggregate.get("execution_measurement_mode"),
                    "energy_measurement_scope": energy_aggregate.get("energy_measurement_scope") or "command_energy",
                    "phases": phases,
                }
                try:
                    root.mkdir(parents=True, exist_ok=True)
                    try:
                        out["energy_aggregate_relpath"] = (root / "energy_aggregate.json").relative_to(group_dir).as_posix()
                        out["energy_summary_relpath"] = (root / "energy_summary.json").relative_to(group_dir).as_posix()
                    except Exception:
                        pass
                    (root / "energy_aggregate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
                    (root / "energy_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                return out, root / "energy_aggregate.json"

            for tres in energy_aggregate.get("target_results") or []:
                if not isinstance(tres, dict):
                    continue
                agg_path_s = tres.get("energy_aggregate") or tres.get("energy_summary")
                subagg = None
                agg_path_obj = Path(str(agg_path_s)) if agg_path_s else None
                if agg_path_obj and agg_path_obj.exists():
                    try:
                        subagg = json.loads(agg_path_obj.read_text(encoding="utf-8"))
                    except Exception:
                        subagg = None
                if not isinstance(subagg, dict):
                    subagg, agg_path_obj = _synth_target_aggregate(tres)
                if not isinstance(subagg, dict) or agg_path_obj is None:
                    continue
                target_case = str(tres.get("energy_target_case") or "")
                target_variant = str(tres.get("energy_target_variant") or "")
                apply_all = bool(tres.get("energy_applies_to_all_cases") or tres.get("applies_to_all_cases"))
                payload = self._energy_merge_payload(setup_id=setup_id, run_id=run_id, energy_aggregate=subagg, aggregate_path=agg_path_obj, result_root=group_dir)
                payload["energy_row_scope"] = True
                payload["energy_target_case"] = target_case
                payload["energy_target_variant"] = target_variant
                payload["energy_applies_to_all_cases"] = apply_all
                if tres.get("canonical_full_target_case"):
                    payload["energy_canonical_full_target_case"] = tres.get("canonical_full_target_case")
                payload_keys_all.update(payload.keys())
                for fp in sorted(set(files)):
                    n = self._merge_energy_payload_into_json(fp, payload, run_id, target_case=target_case, target_variant=target_variant, apply_to_all_cases=apply_all)
                    if n:
                        touched.append(str(fp))
                        rows += n
        else:
            payload = self._energy_merge_payload(setup_id=setup_id, run_id=run_id, energy_aggregate=energy_aggregate, aggregate_path=Path(energy_aggregate_path), result_root=group_dir)
            payload_keys_all.update(payload.keys())
            for fp in sorted(set(files)):
                n = self._merge_energy_payload_into_json(fp, payload, run_id)
                if n:
                    touched.append(str(fp))
                    rows += n
        # v57g: if the normal child-run result file search found files but row-level
        # target matching did not touch anything, do one broader pass over the result
        # group.  This covers result layouts where energy target directories are
        # siblings of the primary dispatch directory or where a refreshed bundle
        # placed benchmark_results under a slightly different child path.
        if rows == 0 and group_dir.exists():
            broad_files = sorted(set(group_dir.rglob(f"benchmark_results*{run_id}*.json")))
            if not broad_files:
                broad_files = sorted(set(group_dir.rglob("benchmark_results*.json")))
            if broad_files and bool(energy_aggregate.get("row_scope")):
                for tres in energy_aggregate.get("target_results") or []:
                    if not isinstance(tres, dict):
                        continue
                    agg_path_s = tres.get("energy_aggregate") or tres.get("energy_summary")
                    subagg = None
                    agg_path_obj = Path(str(agg_path_s)) if agg_path_s else None
                    if agg_path_obj and agg_path_obj.exists():
                        try:
                            subagg = json.loads(agg_path_obj.read_text(encoding="utf-8"))
                        except Exception:
                            subagg = None
                    if not isinstance(subagg, dict):
                        subagg, agg_path_obj = _synth_target_aggregate(tres)
                    if not isinstance(subagg, dict) or agg_path_obj is None:
                        continue
                    target_case = str(tres.get("energy_target_case") or "")
                    target_variant = str(tres.get("energy_target_variant") or "")
                    apply_all = bool(tres.get("energy_applies_to_all_cases") or tres.get("applies_to_all_cases"))
                    payload = self._energy_merge_payload(setup_id=setup_id, run_id=run_id, energy_aggregate=subagg, aggregate_path=agg_path_obj, result_root=group_dir)
                    payload["energy_row_scope"] = True
                    payload["energy_target_case"] = target_case
                    payload["energy_target_variant"] = target_variant
                    payload["energy_applies_to_all_cases"] = apply_all
                    if tres.get("canonical_full_target_case"):
                        payload["energy_canonical_full_target_case"] = tres.get("canonical_full_target_case")
                    payload_keys_all.update(payload.keys())
                    for fp in broad_files:
                        n = self._merge_energy_payload_into_json(fp, payload, run_id, target_case=target_case, target_variant=target_variant, apply_to_all_cases=apply_all)
                        if n:
                            touched.append(str(fp))
                            rows += n
            elif broad_files:
                payload = self._energy_merge_payload(setup_id=setup_id, run_id=run_id, energy_aggregate=energy_aggregate, aggregate_path=Path(energy_aggregate_path), result_root=group_dir)
                payload_keys_all.update(payload.keys())
                for fp in broad_files:
                    n = self._merge_energy_payload_into_json(fp, payload, run_id)
                    if n:
                        touched.append(str(fp))
                        rows += n

        touched = sorted(set(touched))
        manifest = {
            "schema": "onnx-splitpoint/energy-result-merge",
            "schema_version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "setup_id": setup_id,
            "run_id": run_id,
            "energy_aggregate": str(energy_aggregate_path),
            "energy_aggregate_relpath": (os.path.relpath(str(energy_aggregate_path), start=str(group_dir)) if Path(energy_aggregate_path).exists() else None),
            "merged_files": touched,
            "merged_rows": rows,
            "row_scope": bool(energy_aggregate.get("row_scope")),
            "payload_keys": sorted(payload_keys_all),
        }
        try:
            consistency = self._post_energy_merge_artifact_consistency(group_dir=group_dir, child_run_id=child_run_id, run_id=run_id, setup_id=setup_id, merged_files=touched, log=log)
            manifest["post_merge_consistency"] = consistency
        except Exception as exc:
            manifest["post_merge_consistency_error"] = f"{type(exc).__name__}: {exc}"
        try:
            outp = group_dir / child_run_id / "energy_merge_manifest.json"
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        if log:
            if rows:
                log(f"[energy] merged energy metrics into {rows} benchmark result row(s) across {len(touched)} file(s)")
            else:
                log("[energy] warning: no benchmark_results*.json rows matched for energy merge")
        return {"ok": rows > 0, "merged_rows": rows, "merged_files": touched, "payload": payload}

    def _start_remote_benchmark_matrix_job(self, *, bench_path: Path, local_working_dir: Path, suite_name: str, run_id: str, dispatches: list[dict], base_args: RemoteBenchmarkArgs, completion_callback=None, show_result_dialogs: bool = True, job_name_override: str = "") -> Optional[str]:
        results_group_id = self._remote_result_group_name(run_id)
        ready = [d for d in dispatches if d.get("status") != "missing_host"]
        missing = [d for d in dispatches if d.get("status") == "missing_host"]
        if not ready:
            messagebox.showwarning("Remote benchmark", "No configured hardware setup matched the benchmark run plan. Configure Host/User/Venv in Tool Config.")
            return None
        job_id = f"remote-matrix-{run_id}"
        cancel_event = threading.Event()
        lines = [
            f"Suite: {bench_path.parent}",
            "Remote dispatch is automatic from benchmark_plan.json.",
            "Each run id is sent to the matching Tool Config hardware setup.",
        ]
        for d in ready:
            lines.append(f"- {d.get('run_id') or '(all)'} -> {d.get('setup_id')}")
        for d in missing:
            lines.append(f"- {d.get('run_id')} -> {d.get('setup_id')} (missing host; skipped)")
        self._jobs_register(
            job_id=job_id,
            kind="remote_run",
            type_label="Remote run",
            title=f"Remote benchmark matrix — {suite_name}",
            name=str(suite_name),
            output_dir=str(local_working_dir),
            initial_status="Starting…",
            initial_lines=lines,
            progress_maximum=1.0,
            cancel_callback=lambda: cancel_event.set(),
            can_cancel=True,
        )

        def _append(msg: str) -> None:
            self.root.after(0, lambda _s=str(msg): self._jobs_append_log(job_id, _s))

        def _worker() -> None:
            self._set_background_job_active("remote_run", True)
            records: list[dict] = []
            ok_count = 0
            partial_count = 0
            failed_count = 0

            def _env_bool(name: str, default: bool = True) -> bool:
                val = os.environ.get(name, "")
                if val == "":
                    return default
                return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

            def _env_int(name: str, default: int, minimum: int = 0) -> int:
                try:
                    val = int(float(str(os.environ.get(name, "") or default)))
                except Exception:
                    val = int(default)
                return max(minimum, val)

            def _run_one_dispatch(idx: int, d: dict, n: int) -> dict:
                sid = str(d.get("setup_id") or "")
                rid = str(d.get("run_id") or "")
                payload = self._hardware_setup_remote_payload(sid)
                host = self._hardware_setup_host_config(sid)
                if host is None:
                    return {"setup_id": sid, "run_id": rid, "status": "missing_host"}
                service = RemoteBenchmarkService()
                extra = str(base_args.add_args or "").strip()
                if rid:
                    extra = (extra + " " + f"--run-id {rid}").strip()
                args_i = replace(
                    base_args,
                    provider="auto" if rid else str(payload.get("provider") or base_args.provider or "auto"),
                    remote_venv=str(payload.get("remote_venv") or base_args.remote_venv or ""),
                    add_args=extra,
                    energy_setup_id=sid,
                    energy_enabled=bool(getattr(base_args, "energy_enabled", False)),
                    energy_run_count=int(getattr(base_args, "energy_run_count", 0) or 0),
                    resume=(False if bool(getattr(base_args, "energy_enabled", False)) else bool(getattr(base_args, "resume", True))),
                )
                child_run_id = "_".join(x for x in [sid, rid or "all"] if x).replace("/", "_").replace(" ", "_")
                _append("")
                _append("=" * 100)
                _append(f"[dispatch {idx}/{n}] run_id={rid or '(all)'} setup={sid} host={host.user_host_pretty} provider={args_i.provider} venv={args_i.remote_venv or '(none)'}")
                try:
                    # Use the immutable arguments captured for this dispatch,
                    # not the live checkbox state.  A UI change after launch
                    # must not route an energy-enabled job around admission.
                    if getattr(args_i, "energy_enabled", None) is True:
                        preflight_remote_energy_dispatch(
                            host=host,
                            args=args_i,
                            registry_path=self._hardware_setups_path(),
                        )
                        _append(f"[{sid}/{rid or 'all'}] [energy] primary benchmark run before energy phase measurements")
                        primary_out = service.run(
                            host=host,
                            benchmark_set_json=bench_path,
                            local_working_dir=local_working_dir,
                            run_id=child_run_id,
                            args=replace(args_i, energy_enabled=False, cleanup_remote_after_download=False),
                            log=lambda msg, _sid=sid, _rid=rid: _append(f"[{_sid}/{_rid or 'all'}] {msg}"),
                            progress=lambda pct, msg, _idx=idx, _n=n: self.root.after(0, lambda _pct=pct, _msg=msg, _idx=_idx, _n=_n: self._jobs_set_progress(job_id, value=((_idx - 1) + max(0.0, min(1.0, float(_pct)))) / _n, label=str(_msg or ""), display=f"{int(round(((_idx - 1) + max(0.0, min(1.0, float(_pct)))) / _n * 100.0))}%", progress_maximum=1.0)),
                            cancel_event=cancel_event,
                            results_group_id=results_group_id,
                        )
                        if str((primary_out or {}).get("status") or "").lower() not in {"ok", "partial"}:
                            out = dict(primary_out or {})
                            out.setdefault("status", "failed")
                            out.setdefault("error", f"primary remote run status={out.get('status')}; energy windows skipped")
                        else:
                            energy_out = self._run_remote_dispatch_with_energy(
                                setup_id=sid,
                                run_id=rid,
                                host=host,
                                benchmark_set_json=bench_path,
                                local_working_dir=local_working_dir,
                                child_run_id=child_run_id,
                                args=args_i,
                                results_group_id=results_group_id,
                                log=lambda msg, _sid=sid, _rid=rid: _append(f"[{_sid}/{_rid or 'all'}] {msg}"),
                                primary_out=primary_out,
                            )
                            try:
                                group_dir_for_merge = Path(local_working_dir).expanduser().resolve() / "Results" / bench_path.parent.name / results_group_id
                                agg_path = Path(str(energy_out.get("energy_aggregate") or energy_out.get("energy_summary") or ""))
                                if agg_path.exists():
                                    merge_out = self._merge_energy_into_benchmark_results(
                                        group_dir=group_dir_for_merge,
                                        child_run_id=child_run_id,
                                        run_id=rid,
                                        setup_id=sid,
                                        energy_aggregate_path=agg_path,
                                        log=lambda msg, _sid=sid, _rid=rid: _append(f"[{_sid}/{_rid or 'all'}] {msg}"),
                                    )
                                    energy_out["energy_merge"] = merge_out
                            except Exception as exc:
                                _append(f"[{sid}/{rid or 'all'}] [energy] warning: failed to merge energy metrics into benchmark results: {type(exc).__name__}: {exc}")
                            try:
                                cleanup_payload = self._cleanup_primary_remote_run_dir_after_energy(host=host, primary_out=primary_out, log=lambda msg, _sid=sid, _rid=rid: _append(f"[{_sid}/{_rid or 'all'}] {msg}"))
                                energy_out["primary_remote_cleanup"] = cleanup_payload
                            except Exception as exc:
                                _append(f"[{sid}/{rid or 'all'}] [cleanup][warn] primary remote cleanup bookkeeping failed: {type(exc).__name__}: {exc}")
                            out = dict(primary_out or {})
                            out["energy"] = energy_out
                            out["energy_summary"] = energy_out.get("energy_summary") or energy_out.get("energy_out_dir")
                            out["energy_aggregate"] = energy_out.get("energy_aggregate") or energy_out.get("energy_summary") or energy_out.get("energy_out_dir")
                            if str(out.get("status") or "").lower() == "ok" and str(energy_out.get("status") or "").lower() not in {"ok", ""}:
                                out["status"] = "partial"
                    else:
                        out = service.run(
                            host=host,
                            benchmark_set_json=bench_path,
                            local_working_dir=local_working_dir,
                            run_id=child_run_id,
                            args=args_i,
                            log=lambda msg, _sid=sid, _rid=rid: _append(f"[{_sid}/{_rid or 'all'}] {msg}"),
                            progress=lambda pct, msg, _idx=idx, _n=n: self.root.after(0, lambda _pct=pct, _msg=msg, _idx=_idx, _n=_n: self._jobs_set_progress(job_id, value=((_idx - 1) + max(0.0, min(1.0, float(_pct)))) / _n, label=str(_msg or ""), display=f"{int(round(((_idx - 1) + max(0.0, min(1.0, float(_pct)))) / _n * 100.0))}%", progress_maximum=1.0)),
                            cancel_event=cancel_event,
                            results_group_id=results_group_id,
                        )
                    status = str(out.get("status") or ("ok" if out.get("ok") else "failed")).strip().lower()
                    return {
                        "setup_id": sid,
                        "run_id": rid,
                        "status": status,
                        "local_run_dir": out.get("local_run_dir"),
                        "error": out.get("error"),
                        "energy_summary": out.get("energy_summary"),
                        "energy_aggregate": out.get("energy_aggregate"),
                        "dispatch_energy_total_j": (out.get("energy") or {}).get("dispatch_energy_total_j") if isinstance(out.get("energy"), dict) else None,
                        "dispatch_energy_window_count": (out.get("energy") or {}).get("dispatch_energy_window_count") if isinstance(out.get("energy"), dict) else None,
                        "dispatch_avg_energy_j_per_window": (out.get("energy") or {}).get("dispatch_avg_energy_j_per_window") if isinstance(out.get("energy"), dict) else None,
                    }
                except Exception as exc:
                    _append(f"[{sid}/{rid or 'all'}] failed: {type(exc).__name__}: {exc}")
                    return {"setup_id": sid, "run_id": rid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}

            try:
                n = max(1, len(ready))
                groups: dict[str, list[tuple[int, dict]]] = {}
                for idx, d in enumerate(ready, start=1):
                    sid = str(d.get("setup_id") or "") or f"setup_{idx}"
                    groups.setdefault(sid, []).append((idx, d))
                parallel_enabled = _env_bool("ONNX_SPLITPOINT_PARALLEL_REMOTE_SETUPS", True) and len(groups) > 1
                max_workers = min(len(groups), _env_int("ONNX_SPLITPOINT_MAX_PARALLEL_SETUPS", 3, 1)) if parallel_enabled else 1
                # Propagate upload/powercalc limits to nested helpers; defaults are conservative.
                os.environ.setdefault("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS", "1")
                os.environ.setdefault("ONNX_SPLITPOINT_POWER_CALC_WORKERS", "1")
                _append(f"[parallel] setup workers={'enabled' if parallel_enabled else 'disabled'} groups={len(groups)} max_workers={max_workers} max_parallel_uploads={os.environ.get('ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS')} powercalc_workers={os.environ.get('ONNX_SPLITPOINT_POWER_CALC_WORKERS')}")

                def _run_group(items: list[tuple[int, dict]]) -> list[dict]:
                    out_records: list[dict] = []
                    for idx, d in items:
                        if cancel_event.is_set():
                            out_records.append({"setup_id": d.get("setup_id"), "run_id": d.get("run_id"), "status": "cancelled"})
                            break
                        out_records.append(_run_one_dispatch(idx, d, n))
                    return out_records

                if parallel_enabled:
                    with __import__('concurrent.futures').futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="osp-manual-remote-setup") as pool:
                        futures = [pool.submit(_run_group, items) for items in groups.values()]
                        for fut in __import__('concurrent.futures').futures.as_completed(futures):
                            records.extend(fut.result())
                else:
                    for items in groups.values():
                        records.extend(_run_group(items))

                for rec in records:
                    st = str(rec.get("status") or "failed").lower()
                    if st == "ok":
                        ok_count += 1
                    elif st == "partial":
                        partial_count += 1
                    elif st == "cancelled":
                        partial_count += 1
                    elif st != "missing_host":
                        failed_count += 1

                group_dir = Path(local_working_dir).expanduser().resolve() / "Results" / bench_path.parent.name / results_group_id
                matrix_dir = group_dir / "_summary"
                matrix_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "schema": "onnx-splitpoint/manual-remote-hardware-matrix",
                    "schema_version": 2,
                    "suite": str(bench_path.parent),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "parallel_setup_dispatch": bool(parallel_enabled),
                    "max_parallel_setups": int(max_workers),
                    "max_parallel_uploads": int(os.environ.get("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS", "1") or 1),
                    "powercalc_workers": int(os.environ.get("ONNX_SPLITPOINT_POWER_CALC_WORKERS", "1") or 1),
                    "dispatches": records,
                    "ok_count": ok_count,
                    "partial_count": partial_count,
                    "failed_count": failed_count,
                    "missing_host": missing,
                }
                (matrix_dir / "remote_hardware_matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                # v59u: also emit the canonical manual-remote run manifest files
                # used by the non-parallel GUI path, so downstream report/debug
                # tooling sees the same artifact structure independent of whether
                # setup-level parallelism was used.
                try:
                    dispatch_summary = {
                        "schema": "onnx-splitpoint/gui-hardware-dispatch-summary",
                        "schema_version": 3,
                        "matrix_session_id": str(run_id),
                        "results_group_id": results_group_id,
                        "benchmark_set_json": str(bench_path),
                        "results_root": str(group_dir),
                        "summary_dir": str(matrix_dir),
                        "parallel_setup_dispatch": bool(parallel_enabled),
                        "setup_group_count": len(groups),
                        "max_parallel_setups": int(max_workers),
                        "max_parallel_uploads": int(os.environ.get("ONNX_SPLITPOINT_MAX_PARALLEL_UPLOADS", "1") or 1),
                        "powercalc_workers": int(os.environ.get("ONNX_SPLITPOINT_POWER_CALC_WORKERS", "1") or 1),
                        "dispatch_count": len(records),
                        "dispatches": records,
                    }
                    (matrix_dir / "results").mkdir(parents=True, exist_ok=True)
                    (matrix_dir / "logs").mkdir(parents=True, exist_ok=True)
                    (matrix_dir / "results" / "hardware_dispatch_summary.json").write_text(json.dumps(dispatch_summary, indent=2, ensure_ascii=False), encoding="utf-8")
                    (matrix_dir / "logs" / "runner.log").write_text(json.dumps(dispatch_summary, indent=2, ensure_ascii=False), encoding="utf-8")
                    (group_dir / "run_manifest.json").write_text(json.dumps(dispatch_summary, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    logger.debug("failed to write canonical parallel dispatch manifest", exc_info=True)
                final_kind = "ok" if ok_count and not partial_count and not failed_count else ("partial" if ok_count or partial_count else "failed")
                message = f"{ok_count} ok, {partial_count} partial, {failed_count} failed. Summary: {matrix_dir}"
                self.root.after(0, lambda: self._jobs_update_paths(job_id, output_dir=str(group_dir), log_path=""))
                self.root.after(0, lambda: self._jobs_finish(job_id, status={"ok":"success","partial":"warning","failed":"error"}.get(final_kind,"error"), message=message, output_dir=str(matrix_dir)))
                if callable(completion_callback):
                    self.root.after(0, lambda: completion_callback(final_kind, {"status": final_kind, "local_run_dir": str(group_dir), "summary_dir": str(matrix_dir), "dispatches": records, "error": message}))
                if show_result_dialogs:
                    def _show():
                        if final_kind == "ok":
                            messagebox.showinfo("Remote benchmark", f"Done. Hardware matrix results saved to:\n\n{group_dir}")
                        elif final_kind == "partial":
                            messagebox.showwarning("Remote benchmark", f"Partial hardware matrix run.\n\n{message}")
                        else:
                            messagebox.showerror("Remote benchmark", f"Hardware matrix run failed.\n\n{message}")
                    self.root.after(0, _show)
            except Exception as exc:
                self.root.after(0, lambda: self._jobs_finish(job_id, status="error", message=f"{type(exc).__name__}: {exc}"))
                if show_result_dialogs:
                    self.root.after(0, lambda: messagebox.showerror("Remote benchmark", f"Hardware matrix run failed:\n\n{type(exc).__name__}: {exc}"))
            finally:
                self._set_background_job_active("remote_run", False)

        self._remote_benchmark_thread = threading.Thread(target=_worker, name=f"remote-matrix-{run_id}", daemon=True)
        self._remote_benchmark_thread.start()
        return job_id

    def _remote_run_benchmark(
        self,
        *,
        benchmark_set_json_override: str | Path | None = None,
        local_working_dir_override: str | Path | None = None,
        completion_callback=None,
        show_result_dialogs: bool = True,
        job_name_override: str = "",
    ) -> Optional[str]:
        if bool(getattr(self, "_remote_benchmark_active", False)):
            messagebox.showinfo(
                "Remote benchmark already running",
                "A remote benchmark is already running in the background.\n\n"
                "You can keep one benchmark-set generation running in parallel, but only one remote benchmark at a time.",
            )
            return

        bench_json = str(benchmark_set_json_override or (self.var_remote_benchmark_set.get().strip() if hasattr(self, "var_remote_benchmark_set") else "")).strip()
        if not bench_json:
            messagebox.showwarning("Remote benchmark", "Please select a benchmark_set.json first.")
            return
        bench_path = Path(bench_json).expanduser()
        if bench_path.is_dir():
            bench_path = bench_path / "benchmark_set.json"
        if not bench_path.exists():
            messagebox.showerror("Remote benchmark", f"benchmark_set.json not found: {bench_path}")
            return

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

        # v52o: The benchmark/evaluation plan is now authoritative.  The user
        # no longer chooses a single remote host.  We inspect benchmark_plan.json
        # and dispatch each run-id to the central hardware setup that owns the
        # accelerator involved in that row.  Mixed runs such as hailo8_to_trt run
        # on the Hailo-8 NX; deepx_m1_full and deepx_m1_to_tensorrt run on the
        # DeepX NX; pure TensorRT references run on a configured reference NX.
        dispatches = self._auto_remote_dispatch_plan(bench_path)
        if not dispatches:
            messagebox.showwarning(
                "Remote benchmark",
                "No configured hardware setup matches this benchmark plan.\n\n"
                "Open Tool Config and fill Host/User/Venv for Hailo-8, Hailo-10 and/or DeepX.",
            )
            return

        run_id = time.strftime("%Y%m%d_%H%M%S")
        results_group_id = self._remote_result_group_name(run_id)
        local_working_dir = Path(local_working_dir_override).expanduser() if local_working_dir_override else Path(getattr(self, "default_output_dir", "."))
        suite_name = str(job_name_override or (bench_path.parent.name or bench_path.stem or f"remote_run_{run_id}"))

        base_args = RemoteBenchmarkArgs(
            provider="auto",
            warmup=warmup,
            iters=iters,
            repeats=repeats,
            timeout_s=timeout_s,
            throughput_frames=throughput_frames,
            throughput_warmup_frames=throughput_warmup_frames,
            throughput_queue_depth=throughput_queue_depth,
            validation_images=(self.var_bench_validation_images.get() if hasattr(self, "var_bench_validation_images") else ""),
            validation_max_images=int(((self.var_bench_validation_max_images.get() if hasattr(self, "var_bench_validation_max_images") else "50") or "50")),
            validation_reference_mode=(self.var_bench_validation_reference_mode.get() if hasattr(self, "var_bench_validation_reference_mode") else "auto"),
            mini_coco_ap50=(bool(self.var_bench_mini_coco_ap50.get()) if hasattr(self, "var_bench_mini_coco_ap50") else False),
            benchmark_task=(self.var_bench_task.get() if hasattr(self, "var_bench_task") else "auto"),
            mini_classification_eval=(bool(self.var_bench_mini_classification_eval.get()) if hasattr(self, "var_bench_mini_classification_eval") else False),
            add_args=self.var_remote_add_args.get() if hasattr(self, "var_remote_add_args") else "",
            remote_venv="",
            transfer_mode=self.var_remote_transfer_mode.get() if hasattr(self, "var_remote_transfer_mode") else "bundle",
            reuse_bundle=bool(self.var_remote_reuse_bundle.get()) if hasattr(self, "var_remote_reuse_bundle") else True,
            energy_enabled=bool(self.var_remote_measure_energy.get()) if hasattr(self, "var_remote_measure_energy") else False,
            energy_registry_path=str(self._hardware_setups_path()),
            energy_run_count=self._parse_remote_int(self.var_remote_energy_runs.get() if hasattr(self, "var_remote_energy_runs") and str(self.var_remote_energy_runs.get()).strip() else "0", default=0, label="Energy runs", minimum=0),
        )

        # v59u: The normal Benchmark-tab remote run used to dispatch the
        # hardware/setup matrix strictly sequentially, even though the Evaluation
        # workflow already supports parallel independent setup groups.  Flatten
        # the auto-dispatch plan into setup/run rows and hand it to the matrix
        # dispatcher, which executes different physical setups concurrently and
        # keeps run_ids within the same setup serial for clean per-device energy.
        flat_dispatches = []
        for _disp in dispatches:
            _run_ids = list((_disp.get("run_ids") or [])) or [""]
            for _rid in _run_ids:
                _row = dict(_disp)
                _row["run_id"] = str(_rid or "")
                flat_dispatches.append(_row)
        if flat_dispatches:
            return self._start_remote_benchmark_matrix_job(
                bench_path=bench_path,
                local_working_dir=local_working_dir,
                suite_name=suite_name,
                run_id=run_id,
                dispatches=flat_dispatches,
                base_args=base_args,
                completion_callback=completion_callback,
                show_result_dialogs=show_result_dialogs,
                job_name_override=job_name_override,
            )

        def worker():
            self._set_background_job_active("remote_run", True)
            results: list[dict] = []
            statuses: list[str] = []
            total = max(1, sum(max(1, len(d.get("run_ids") or [])) for d in dispatches))
            seq = 0
            try:
                for disp in dispatches:
                    if cancel_event.is_set():
                        break
                    payload = dict(disp.get("payload") or {})
                    setup_id = str(disp.get("setup_id") or payload.get("id") or "hardware")
                    host = self._host_config_from_hardware_payload(payload)
                    run_ids = list(disp.get("run_ids") or []) or [""]
                    for rid in run_ids:
                        if cancel_event.is_set():
                            break
                        seq += 1
                        child_name = f"{setup_id}" + (f"/{rid}" if rid else "")
                        self.root.after(0, lambda _s=child_name: self._jobs_append_log(job_id, f"\n=== Dispatch {_s} ==="))
                        add_args = str(getattr(base_args, "add_args", "") or "").strip()
                        if rid:
                            add_args = (add_args + f" --run-id {rid}").strip()
                        args = RemoteBenchmarkArgs(
                            provider="auto",
                            warmup=base_args.warmup,
                            iters=base_args.iters,
                            repeats=base_args.repeats,
                            timeout_s=base_args.timeout_s,
                            throughput_frames=base_args.throughput_frames,
                            throughput_warmup_frames=base_args.throughput_warmup_frames,
                            throughput_queue_depth=base_args.throughput_queue_depth,
                            validation_images=base_args.validation_images,
                            validation_max_images=base_args.validation_max_images,
                            validation_reference_mode=base_args.validation_reference_mode,
                            mini_coco_ap50=base_args.mini_coco_ap50,
                            benchmark_task=base_args.benchmark_task,
                            mini_classification_eval=base_args.mini_classification_eval,
                            add_args=add_args,
                            remote_venv=str(payload.get("remote_venv") or ""),
                            transfer_mode=base_args.transfer_mode,
                            reuse_bundle=base_args.reuse_bundle,
                            energy_enabled=bool(getattr(base_args, "energy_enabled", False)),
                            energy_setup_id=str(setup_id),
                            energy_registry_path=str(
                                getattr(base_args, "energy_registry_path", "")
                                or self._hardware_setups_path()
                            ),
                            energy_run_count=int(getattr(base_args, "energy_run_count", 0) or 0),
                            energy_output_root=str(Path(local_working_dir).expanduser().resolve() / "EnergyMeasurements" / "Benchmarks" / bench_path.parent.name),
                            resume=(False if bool(getattr(base_args, "energy_enabled", False)) else bool(getattr(base_args, "resume", True))),
                        )

                        def _log(s, _child=child_name):
                            self.root.after(0, lambda _s=s, _c=_child: self._jobs_append_log(job_id, f"[{_c}] {_s}"))

                        def _progress(p, lbl, _seq=seq, _child=child_name):
                            val = (float(_seq - 1) + max(0.0, min(1.0, float(p)))) / float(total)
                            self.root.after(0, lambda _v=val, _lbl=lbl, _c=_child: self._jobs_set_progress(
                                job_id,
                                value=max(0.0, min(1.0, _v)),
                                label=f"{_c}: {_lbl}",
                                display=f"{int(round(max(0.0, min(1.0, _v)) * 100.0))}%",
                                progress_maximum=1.0,
                            ))

                        child_results: list[tuple[str, dict]] = []
                        callbacks = RemoteBenchmarkCallbacks(
                            log=_log,
                            progress=_progress,
                            finish=lambda status, _child=child_name: self.root.after(0, lambda _status=status, _c=_child: self._jobs_set_progress(job_id, value=min(1.0, float(seq) / float(total)), label=f"{_c}: {_status}", progress_maximum=1.0)),
                            result=lambda kind, out, _child=child_name: child_results.append((str(kind), dict(out))),
                        )
                        child_run_id = f"{setup_id}_{rid or 'all'}"
                        # The dispatch contract is the captured args object;
                        # consulting mutable GUI state here creates a TOCTOU
                        # bypass between launch and the primary workload.
                        if getattr(args, "energy_enabled", None) is True:
                            preflight_remote_energy_dispatch(
                                host=host,
                                args=args,
                                registry_path=self._hardware_setups_path(),
                            )
                            # v56n: energy windows must measure an already deployed suite, not the
                            # full benchmark-remote packaging/scp/orchestration path.  Run the normal
                            # remote benchmark once first; then hand its remote_run_dir to the energy
                            # phase runner so it can SSH directly into <remote_run_dir>/suite.
                            _log("[energy] primary benchmark run before energy phase measurements")
                            # v57i: Energy row-level windows depend on the current benchmark_suite.py
                            # supporting --energy-target-case/--energy-target-variant and
                            # --energy-measurement-only.  Existing benchmarksets may still
                            # have an old cached dist/suite_bundle.tar.gz.  Rebuild the
                            # remote bundle for the primary run whenever energy is enabled,
                            # otherwise the later execution-only windows can fail with
                            # "unrecognized arguments" or silently execute stale harness code.
                            primary_out = self._remote_controller.run(
                                host=host,
                                benchmark_set_json=bench_path,
                                local_working_dir=local_working_dir,
                                run_id=child_run_id,
                                args=replace(args, energy_enabled=False, resume=False, reuse_bundle=False, cleanup_remote_after_download=False),
                                cancel_event=cancel_event,
                                callbacks=callbacks,
                                results_group_id=results_group_id,
                            )
                            primary_status = str((primary_out or {}).get("status") or ("ok" if (primary_out or {}).get("ok") else "")).lower()
                            primary_rc = (primary_out or {}).get("rc")
                            if not str((primary_out or {}).get("remote_run_dir") or "").strip():
                                _log("[energy][warn] primary run did not expose remote_run_dir; energy will fall back to benchmark-remote CLI wrapper")
                            else:
                                _log(f"[energy] primary remote_run_dir={(primary_out or {}).get('remote_run_dir')}")
                            if primary_status in {"failed", "partial"} or (primary_rc not in (None, 0)):
                                # Do not start u.RECS windows against a failed primary run.
                                # It only produces raw/partial energy folders and misleading
                                # merge warnings.  Surface the real primary failure first.
                                _log(f"[energy][abort] primary remote run status={primary_status or '-'} rc={primary_rc}; skipping energy windows")
                                out = {
                                    "ok": False,
                                    "status": "primary_remote_failed",
                                    "setup_id": setup_id,
                                    "run_id": str(rid or ""),
                                    "energy_skipped": True,
                                    "energy_skip_reason": "primary_remote_failed_before_energy_windows",
                                    "primary_remote_run_dir": (primary_out or {}).get("remote_run_dir"),
                                    "primary_local_run_dir": (primary_out or {}).get("local_run_dir"),
                                    "primary_status": (primary_out or {}).get("status"),
                                    "primary_rc": primary_rc,
                                }
                                kind = str(out.get("status"))
                                child_results.append((kind, dict(out)))
                                statuses.append(kind)
                                results.append({"setup_id": setup_id, "run_id": rid, "host": host.to_dict(), "status": kind, "result": dict(out), "energy": out.get("energy") or out.get("out_dir")})
                                continue
                            out = self._run_remote_dispatch_with_energy(
                                setup_id=setup_id,
                                run_id=str(rid or ""),
                                host=host,
                                benchmark_set_json=bench_path,
                                local_working_dir=local_working_dir,
                                child_run_id=child_run_id,
                                args=args,
                                results_group_id=results_group_id,
                                log=_log,
                                primary_out=primary_out,
                            )
                            try:
                                group_dir_for_merge = Path(local_working_dir).expanduser().resolve() / "Results" / bench_path.parent.name / results_group_id
                                agg_path = Path(str(out.get("energy_aggregate") or out.get("energy_summary") or ""))
                                if agg_path.exists():
                                    merge_out = self._merge_energy_into_benchmark_results(
                                        group_dir=group_dir_for_merge,
                                        child_run_id=child_run_id,
                                        run_id=str(rid or ""),
                                        setup_id=setup_id,
                                        energy_aggregate_path=agg_path,
                                        log=_log,
                                    )
                                    out["energy_merge"] = merge_out
                            except Exception as exc:
                                _log(f"[energy] warning: failed to merge energy metrics into benchmark results: {type(exc).__name__}: {exc}")
                            if isinstance(primary_out, dict):
                                out.setdefault("primary_remote_run_dir", primary_out.get("remote_run_dir"))
                                out.setdefault("primary_local_run_dir", primary_out.get("local_run_dir"))
                                out.setdefault("primary_status", primary_out.get("status"))
                            try:
                                cleanup_payload = self._cleanup_primary_remote_run_dir_after_energy(host=host, primary_out=primary_out, log=_log)
                                out["primary_remote_cleanup"] = cleanup_payload
                            except Exception as exc:
                                _log(f"[cleanup][warn] primary remote cleanup bookkeeping failed: {type(exc).__name__}: {exc}")
                            kind = str(out.get("status") or ("ok" if out.get("ok") else "failed"))
                            child_results.append((kind, dict(out)))
                        else:
                            out = self._remote_controller.run(
                                host=host,
                                benchmark_set_json=bench_path,
                                local_working_dir=local_working_dir,
                                run_id=child_run_id,
                                args=args,
                                cancel_event=cancel_event,
                                callbacks=callbacks,
                                results_group_id=results_group_id,
                            )
                            kind = str(child_results[-1][0] if child_results else out.get("status") or ("ok" if out.get("ok") else "failed"))
                        statuses.append(kind)
                        results.append({"setup_id": setup_id, "run_id": rid, "host": host.to_dict(), "status": kind, "result": dict(out), "energy": out.get("energy") or out.get("out_dir")})
                # Write a single local summary folder inside the per-button-press run group.
                group_dir = Path(local_working_dir).expanduser().resolve() / "Results" / bench_path.parent.name / results_group_id
                summary_dir = group_dir / "_summary"
                (summary_dir / "logs").mkdir(parents=True, exist_ok=True)
                (summary_dir / "results").mkdir(parents=True, exist_ok=True)
                summary = {
                    "schema": "onnx-splitpoint/gui-hardware-dispatch-summary",
                    "schema_version": 2,
                    "matrix_session_id": str(run_id),
                    "results_group_id": results_group_id,
                    "benchmark_set_json": str(bench_path),
                    "results_root": str(group_dir),
                    "summary_dir": str(summary_dir),
                    "dispatch_count": len(results),
                    "dispatches": results,
                }
                (summary_dir / "results" / "hardware_dispatch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                (summary_dir / "logs" / "runner.log").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                (group_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                def _dispatch_error_summary() -> str:
                    lines = []
                    for item in results[-8:]:
                        try:
                            setup = str(item.get("setup_id") or "?")
                            rid = str(item.get("run_id") or "?")
                            status = str(item.get("status") or "?")
                            res = item.get("result") or {}
                            err = str(res.get("error") or res.get("primary_status") or "").strip()
                            local = str(res.get("local_run_dir") or res.get("primary_local_run_dir") or "").strip()
                            if len(err) > 500:
                                err = err[:500] + "..."
                            line = f"- {setup}/{rid}: {status}"
                            if err:
                                line += f" — {err}"
                            if local:
                                line += f"\n  local: {local}"
                            lines.append(line)
                        except Exception:
                            pass
                    return "\n".join(lines)

                if cancel_event.is_set():
                    final_kind = "cancelled"
                    error = "Remote hardware dispatch cancelled."
                elif statuses and all(s == "ok" for s in statuses):
                    final_kind = "ok"
                    error = ""
                elif statuses and any(s in {"ok", "partial"} for s in statuses):
                    final_kind = "partial"
                    extra = _dispatch_error_summary()
                    error = "Some hardware dispatches were partial or failed." + (("\n" + extra) if extra else "")
                else:
                    final_kind = "failed"
                    extra = _dispatch_error_summary()
                    error = "All hardware dispatches failed." + (("\n" + extra) if extra else "")
                out = {"status": final_kind, "local_run_dir": str(group_dir), "summary_dir": str(summary_dir), "error": error, "dispatches": results}
                self.root.after(0, lambda _kind=final_kind, _out=out: self._finalize_remote_benchmark_job(job_id, _kind, _out, show_result_dialogs=show_result_dialogs, completion_callback=completion_callback))
            except Exception as exc:
                logger.exception("Auto remote hardware dispatch failed")
                out = {"status": "error", "error": f"{type(exc).__name__}: {exc}", "dispatches": results}
                self.root.after(0, lambda _out=out: self._finalize_remote_benchmark_job(job_id, "error", _out, show_result_dialogs=show_result_dialogs, completion_callback=completion_callback))

        try:
            self._remote_benchmark_thread = threading.Thread(target=worker, daemon=True)
            self._remote_benchmark_thread.start()
            return job_id
        except Exception as exc:
            self._set_background_job_active("remote_run", False)
            self._jobs_finish(job_id, status="error", message="Failed to start remote benchmark thread")
            if callable(completion_callback):
                try:
                    completion_callback('error', {'error': f'{type(exc).__name__}: {exc}'})
                except Exception:
                    logger.debug('Remote benchmark completion callback failed during startup', exc_info=True)
            raise

    def _finalize_remote_benchmark_job(self, job_id: str, final_kind: str, out: Dict[str, Any], *, show_result_dialogs: bool = True, completion_callback=None) -> None:
        self._set_background_job_active("remote_run", False)
        local_run_dir = str(out.get("local_run_dir") or "").strip()
        log_path = ""
        if local_run_dir:
            candidates = [
                Path(local_run_dir) / "logs" / "runner.log",
                Path(local_run_dir) / "logs" / "stdout.txt",
            ]
            summary_dir = str(out.get("summary_dir") or "").strip()
            if summary_dir:
                candidates.extend([
                    Path(summary_dir) / "logs" / "runner.log",
                    Path(summary_dir) / "logs" / "stdout.txt",
                ])
            for cand in candidates:
                if cand.exists():
                    log_path = str(cand)
                    break
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
        if callable(completion_callback):
            try:
                completion_callback(final_kind, out)
            except Exception:
                logger.debug('Remote benchmark completion callback failed', exc_info=True)
        if show_result_dialogs:
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



    def _apply_prepared_model_selection(self, selected_model_path: str, *, message: str = '') -> None:
        path = str(selected_model_path or '').strip()
        if not path:
            return
        try:
            clear = getattr(self, '_clear_results', None)
            if callable(clear):
                clear()
        except Exception:
            logger.debug('Failed to clear analysis state before applying prepared model', exc_info=True)
        self.model_path = path
        self.gui_state.current_model_path = path
        self.gui_state.model_type = 'onnx'
        try:
            if hasattr(self, 'lbl_model'):
                self.lbl_model.configure(text=os.path.basename(path))
        except Exception:
            pass
        try:
            self.events.emit_model_loaded({'path': path, 'model_type': 'onnx'})
        except Exception:
            pass
        try:
            if hasattr(self, 'var_bench_model_preparation_info'):
                info = f"Prepared model selected: {os.path.basename(path)}"
                if message:
                    info += f" · {message}"
                self.var_bench_model_preparation_info.set(info)
        except Exception:
            pass

    def _queue_prepare_current_model(self) -> Optional[str]:
        model_path = str(getattr(getattr(self, 'gui_state', None), 'current_model_path', None) or getattr(self, 'model_path', None) or '').strip()
        if not model_path:
            messagebox.showwarning('Model preparation', 'Please load a model first.')
            return None
        prep_mode_raw = str(getattr(self, 'var_bench_model_preparation_mode', tk.StringVar(value='Use current ONNX')).get() or '').strip()
        prep_mode = normalize_model_preparation_mode(prep_mode_raw)
        if prep_mode == 'current':
            messagebox.showinfo('Model preparation', 'Preparation mode is set to "Use current ONNX". Nothing to prepare.')
            return None
        export_meta = load_export_metadata_for_model(model_path)
        try:
            prep_root = ensure_workdir(Path(getattr(self, 'default_output_dir', '.') or '.')).benchmark_sets / '_prepared_models'
        except Exception:
            prep_root = Path(getattr(self, 'default_output_dir', '.') or '.').expanduser() / '_prepared_models'
        run_id = time.strftime('%Y%m%d_%H%M%S')
        job_id = f'prepare-model-{run_id}'
        cancel_event = threading.Event()
        self._jobs_register(
            job_id=job_id,
            kind='prepare_model',
            type_label='Model preparation',
            title=f'Model preparation — {os.path.basename(model_path)}',
            name=os.path.basename(model_path),
            output_dir=str(prep_root),
            initial_status='Preparing model…',
            initial_lines=[
                f'Model: {model_path}',
                f'Mode: {prep_mode}',
                'The tool will probe the current ONNX as full Hailo and, for supported YOLO detection models, export a small set of fallback variants until one succeeds.',
            ],
            progress_maximum=1.0,
            cancel_callback=lambda: cancel_event.set(),
            can_cancel=True,
            geometry='900x460',
        )

        def _log(line: str) -> None:
            self.root.after(0, lambda _s=str(line): self._jobs_append_log(job_id, _s))
            self.root.after(0, lambda _s=str(line): self._eval_workflow_text_append(_s + "\n"))

        def worker() -> None:
            try:
                runtime = snapshot_preparation_runtime(self)
                out = prepare_model_for_benchmark(
                    model_path,
                    export_metadata=export_meta,
                    requested_mode=prep_mode,
                    output_root=prep_root,
                    runtime=runtime,
                    log=_log,
                    cancel_event=cancel_event,
                )
                status = 'warning' if (out.skipped or getattr(out, 'tier2_success', False)) else ('success' if out.success else 'error')
                msg = str(out.message or '')
                self.root.after(0, lambda _out=out, _status=status, _msg=msg: self._jobs_finish(job_id, status=_status, message=_msg, output_dir=str(_out.screening_dir or prep_root)))
                if out.skipped:
                    self.root.after(0, lambda _out=out: messagebox.showwarning('Model preparation', str(_out.message or 'Preparation was skipped and the current ONNX remains selected.')))
                elif out.success:
                    self.root.after(0, lambda _out=out: self._apply_prepared_model_selection(_out.selected_model_path, message=_out.message))
                    self.root.after(0, lambda _out=out: messagebox.showinfo('Model preparation', f"Preparation finished.\n\nSelected model:\n{_out.selected_model_path}\n\nPlease run Analyse again for this prepared ONNX."))
                else:
                    self.root.after(0, lambda _out=out: messagebox.showwarning('Model preparation', f"No suitable full-Hailo export variant was found.\n\n{_out.message}"))
            except Exception as exc:
                logger.exception('Model preparation failed')
                msg = f'{type(exc).__name__}: {exc}'
                status = 'cancelled' if cancel_event.is_set() else 'error'
                self.root.after(0, lambda _msg=msg, _status=status: self._jobs_finish(job_id, status=_status, message=_msg, output_dir=str(prep_root)))
                if cancel_event.is_set():
                    self.root.after(0, lambda _msg=msg: messagebox.showwarning('Model preparation', _msg))
                else:
                    self.root.after(0, lambda _msg=msg: messagebox.showerror('Model preparation', _msg))

        threading.Thread(target=worker, daemon=True).start()
        return job_id

    # ------------------------------------------------------------------
    # Formal Evaluation Workflow tab (v49c)
    # ------------------------------------------------------------------

    def _eval_workflow_text_set(self, text: str) -> None:
        widget = getattr(self, "eval_workflow_summary_text", None)
        if widget is None:
            return
        try:
            widget.configure(state="normal")
            widget.delete("1.0", tk.END)
            widget.insert("1.0", str(text or ""))
            widget.see(tk.END)
            widget.configure(state="disabled")
        except Exception:
            logger.debug("Failed to update Evaluation Workflow summary text", exc_info=True)

    def _eval_workflow_text_append(self, text: str) -> None:
        widget = getattr(self, "eval_workflow_summary_text", None)
        if widget is None:
            return
        try:
            widget.configure(state="normal")
            widget.insert(tk.END, str(text or ""))
            widget.see(tk.END)
            widget.configure(state="disabled")
        except Exception:
            logger.debug("Failed to append Evaluation Workflow summary text", exc_info=True)

    def _eval_workflow_default_out_root(self) -> Path:
        root = Path(getattr(self, "default_output_dir", ".") or ".").expanduser()
        try:
            return ensure_workdir(root).root / "EvaluationRuns"
        except Exception:
            return root / "EvaluationRuns"

    def _eval_workflow_bool(self, value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        text = str(value).strip().lower()
        if text in {"", "none", "null"}:
            return bool(default)
        return text in {"1", "true", "yes", "y", "on"}

    def _eval_workflow_load_profile_payload(self, profile: str) -> Dict[str, Any]:
        try:
            from ..workflow.profile_options import (
                load_runtime_profile_snapshot,
            )

            runtime_payload, start_snapshot = load_runtime_profile_snapshot(
                profile
            )
            self._evaluation_workflow_resolved_start_snapshot = start_snapshot
            return runtime_payload
        except Exception as exc:
            logger.warning("Could not load evaluation profile defaults from %s: %s", profile, exc)
        self._evaluation_workflow_resolved_start_snapshot = {}
        return {}

    def _eval_workflow_remote_host_payload_from_profile(self, remote: Mapping[str, Any]) -> Dict[str, Any]:
        """Resolve the profile's remote host through the existing Benchmark-tab host list.

        The Workflow tab no longer has a second Remote Host form.  The YAML may
        store only a host_id, a safe inline host copy, or direct host fields.  We
        prefer the central in-tool host list when a matching host_id exists.
        """
        host_id = str(remote.get("host_id") or remote.get("id") or "").split("—", 1)[0].strip()
        if host_id:
            try:
                selected = self._remote_service.get_selected_host(getattr(self, "remote_hosts", []) or [], host_id)
                if selected is not None:
                    return selected.to_dict()
            except Exception:
                pass
        inline = remote.get("hosts") or remote.get("remote_hosts") or []
        if isinstance(inline, list):
            for item in inline:
                if not isinstance(item, Mapping):
                    continue
                hid = str(item.get("id") or item.get("label") or "").strip()
                if not host_id or hid == host_id:
                    return dict(item)
        raw_json = str(remote.get("host_json") or "").strip()
        if raw_json:
            try:
                payload = json.loads(raw_json)
                if isinstance(payload, Mapping):
                    return dict(payload)
            except Exception:
                pass
        host = str(remote.get("host") or "").strip()
        if host:
            user = str(remote.get("user") or "").strip()
            if "@" in host and not user:
                user, host = host.split("@", 1)
            return {
                "id": host_id or str(remote.get("label") or host or "workflow_remote"),
                "label": str(remote.get("label") or host_id or host),
                "host": host,
                "user": user,
                "port": int(remote.get("port") or 22),
                "remote_base_dir": str(remote.get("remote_base_dir") or "~/splitpoint_runs"),
                "ssh_extra_args": str(remote.get("ssh_extra_args") or ""),
            }
        return {}

    def _eval_workflow_snapshot_options(self, *, resume_override: Optional[bool] = None) -> WorkflowOptions:
        profile = str(getattr(getattr(self, "var_eval_workflow_profile", None), "get", lambda: "")() or "").strip()
        if not profile:
            raise ValueError("Please select an Evaluation Profile or YAML file.")
        out_root = str(getattr(getattr(self, "var_eval_workflow_out_root", None), "get", lambda: "")() or "").strip()
        if not out_root:
            out_root = str(self._eval_workflow_default_out_root())
            try:
                self.var_eval_workflow_out_root.set(out_root)
            except Exception:
                pass
        else:
            try:
                _p_out = Path(out_root).expanduser()
                if not _p_out.is_absolute():
                    _root = Path(getattr(self, "default_output_dir", ".") or ".").expanduser()
                    try:
                        _root = ensure_workdir(_root).root
                    except Exception:
                        pass
                    out_root = str((_root / _p_out).resolve())
                    try:
                        self.var_eval_workflow_out_root.set(out_root)
                    except Exception:
                        pass
            except Exception:
                pass

        payload = self._eval_workflow_load_profile_payload(profile)
        start_snapshot = dict(
            getattr(self, "_evaluation_workflow_resolved_start_snapshot", {}) or {}
        )
        if not payload or not start_snapshot:
            raise ValueError(
                "Evaluation Profile could not be resolved into a verified start snapshot. "
                "Reload the profile summary and try again."
            )
        from ..workflow.start_snapshot import start_snapshot_matches_preview, validate_profile_start_snapshot

        start_snapshot = validate_profile_start_snapshot(start_snapshot)
        visible_snapshot = dict(
            getattr(self, "_evaluation_workflow_visible_start_snapshot", {}) or {}
        )
        visible_matches = start_snapshot_matches_preview(
            visible_snapshot,
            start_snapshot,
            profile_request=profile,
        )
        if not visible_matches:
            try:
                refresh = getattr(self, "_evaluation_workflow_refresh_profile_summary", None)
                if callable(refresh):
                    refresh()
            except Exception:
                pass
            raise ValueError(
                "Evaluation start blocked: the profile or central run-mode configuration changed "
                "after the visible summary was generated. The summary has been refreshed; review it "
                "and press Start again."
            )
        remote = dict(
            payload.get("remote_execution") or payload.get("remote") or {}
        )
        hardware_cfg = dict(payload.get("hardware") or {})
        selected_setups = hardware_cfg.get("selected_setups")
        if isinstance(selected_setups, (list, tuple)):
            selected_setup_ids = [
                str(item).strip()
                for item in selected_setups
                if str(item).strip()
            ]
        else:
            selected_setup_ids = [
                item.strip()
                for item in str(selected_setups or "").split(",")
                if item.strip()
            ]
        remote_enabled = self._eval_workflow_bool(
            remote.get("enabled"), False
        )
        remote_host_payload = (
            self._eval_workflow_remote_host_payload_from_profile(remote)
            if remote_enabled and not selected_setup_ids
            else {}
        )
        models_root = str(
            getattr(
                getattr(self, "var_eval_workflow_models_root", None),
                "get",
                lambda: "",
            )()
            or payload.get("models_root_hint")
            or ""
        ).strip()

        from ..workflow.profile_options import (
            workflow_options_from_profile_snapshot,
        )

        return workflow_options_from_profile_snapshot(
            profile_request=profile,
            out_root=out_root,
            start_snapshot=start_snapshot,
            models_root=models_root,
            resume=(
                bool(resume_override)
                if resume_override is not None
                else False
            ),
            remote_host_payload=remote_host_payload,
            remote_working_dir=str(
                self._eval_workflow_default_out_root()
                / "RemoteBenchmarkRuns"
            ),
        )

    def _eval_workflow_command_preview(self, opts: WorkflowOptions) -> str:
        parts = [
            "python -m onnx_splitpoint_tool.workflow.run_evaluation",
            f"--profile {opts.profile}",
            f"--out {opts.out}",
        ]
        if opts.models_root:
            parts.append(f"--models-root {opts.models_root}")
        if opts.resume:
            parts.append("--resume")
        if opts.run_id:
            parts.append(f"--run-id {opts.run_id}")
        if opts.only_model:
            parts.append(f"--only-model {opts.only_model}")
        if opts.max_models is not None:
            parts.append(f"--max-models {opts.max_models}")
        if opts.include_reserve:
            parts.append("--include-reserve")
        if getattr(opts, "execution_mode", ""):
            parts.append(f"--execution-mode {opts.execution_mode}")
        if opts.dry_run:
            parts.append("--dry-run")
        if opts.skip_benchmarks:
            parts.append("--skip-benchmarks")
        if opts.no_remote:
            parts.append("--no-remote")
        if opts.no_model_hash:
            parts.append("--no-model-hash")
        if opts.stop_after:
            parts.append(f"--stop-after {opts.stop_after}")
        for _st in list(getattr(opts, "force_stage", []) or []):
            parts.append(f"--force-stage {_st}")
        for backend in tuple(getattr(opts, "force_build_confirmed_backends", ()) or ()):
            parts.append(f"--confirm-force-build {backend}")
        for src in list(getattr(opts, "result_sources", []) or []):
            parts.append(f"--result-source {src}")
        if getattr(opts, "benchmark_provider", ""):
            parts.append(f"--benchmark-provider {opts.benchmark_provider}")
        if getattr(opts, "benchmark_warmup", 1) != 1:
            parts.append(f"--benchmark-warmup {opts.benchmark_warmup}")
        if getattr(opts, "benchmark_runs", 3) != 3:
            parts.append(f"--benchmark-runs {opts.benchmark_runs}")
        if getattr(opts, "benchmark_timeout_s", 0):
            parts.append(f"--benchmark-timeout-s {opts.benchmark_timeout_s}")
        if getattr(opts, "benchmark_execution_backend", "auto") != "auto":
            parts.append(f"--benchmark-execution-backend {opts.benchmark_execution_backend}")
        if getattr(opts, "hardware_setups_file", ""):
            parts.append(f"--hardware-setups-file {opts.hardware_setups_file}")
        for sid in list(getattr(opts, "hardware_setup_ids", []) or []):
            parts.append(f"--hardware-setup {sid}")
        for gid in list(getattr(opts, "hardware_group_ids", []) or []):
            parts.append(f"--hardware-group {gid}")
        if getattr(opts, "hailo_build_mode", "reuse_only") != "reuse_only":
            parts.append(f"--hailo-build-mode {opts.hailo_build_mode}")
        if getattr(opts, "hailo_hw_arch", "hailo8") != "hailo8":
            parts.append(f"--hailo-hw-arch {opts.hailo_hw_arch}")
        if getattr(opts, "hailo_build_targets", None):
            parts.append(f"--hailo-targets {','.join(str(x) for x in opts.hailo_build_targets)}")
        if getattr(opts, "hailo_build_timeout_s", 3600) != 3600:
            parts.append(f"--hailo-build-timeout-s {opts.hailo_build_timeout_s}")
        if not getattr(opts, "hailo_build_full", True):
            parts.append("--no-hailo-build-full")
        if not getattr(opts, "hailo_build_part1", True):
            parts.append("--no-hailo-build-part1")
        if not getattr(opts, "hailo_build_part2", True):
            parts.append("--no-hailo-build-part2")
        if getattr(opts, "hailo_preset", "quick") != "quick":
            parts.append(f"--hailo-preset {opts.hailo_preset}")
        if getattr(opts, "hailo_optimization_level", 0):
            parts.append(f"--hailo-opt-level {opts.hailo_optimization_level}")
        if getattr(opts, "hailo_calib_dir", ""):
            parts.append(f"--hailo-calib-dir {opts.hailo_calib_dir}")
        if getattr(opts, "hailo_calib_count", 16) != 16:
            parts.append(f"--hailo-calib-count {opts.hailo_calib_count}")
        if getattr(opts, "hailo_calib_batch_size", 8) != 8:
            parts.append(f"--hailo-calib-batch-size {opts.hailo_calib_batch_size}")
        if getattr(opts, "hailo_force_build", False):
            parts.append("--hailo-force-build")
        if getattr(opts, "hailo_keep_artifacts", False):
            parts.append("--hailo-keep-artifacts")
        if getattr(opts, "validation_mode", "summary_only") != "summary_only":
            parts.append(f"--validation-mode {opts.validation_mode}")
        if getattr(opts, "hardware_smoke_mode", "summary_only") != "summary_only":
            parts.append(f"--hardware-smoke-mode {opts.hardware_smoke_mode}")
        if not getattr(opts, "no_remote", True):
            if getattr(opts, "remote_host", ""):
                parts.append(f"--remote-host {opts.remote_host}")
            if getattr(opts, "remote_user", ""):
                parts.append(f"--remote-user {opts.remote_user}")
            if getattr(opts, "remote_port", 22) != 22:
                parts.append(f"--remote-port {opts.remote_port}")
            if getattr(opts, "remote_host_id", ""):
                parts.append(f"--remote-host-id {opts.remote_host_id}")
            if getattr(opts, "remote_host_json", ""):
                parts.append("--remote-host-json <selected-gui-host>")
            if getattr(opts, "remote_base_dir", "") and getattr(opts, "remote_base_dir", "") != "~/splitpoint_runs":
                parts.append(f"--remote-base-dir {opts.remote_base_dir}")
            if getattr(opts, "remote_ssh_extra_args", ""):
                parts.append(f"--remote-ssh-extra-args {opts.remote_ssh_extra_args}")
            if getattr(opts, "remote_venv", ""):
                parts.append(f"--remote-venv {opts.remote_venv}")
            if getattr(opts, "remote_provider", "auto") and getattr(opts, "remote_provider", "auto") != "auto":
                parts.append(f"--remote-provider {opts.remote_provider}")
            if getattr(opts, "remote_warmup", 10) != 10:
                parts.append(f"--remote-warmup {opts.remote_warmup}")
            if getattr(opts, "remote_timeout_s", 0):
                parts.append(f"--remote-timeout-s {opts.remote_timeout_s}")
            if getattr(opts, "remote_transfer_mode", "bundle") != "bundle":
                parts.append(f"--remote-transfer-mode {opts.remote_transfer_mode}")
            if not getattr(opts, "remote_reuse_bundle", True):
                parts.append("--no-remote-reuse-bundle")
            if not getattr(opts, "remote_resume", True):
                parts.append("--no-remote-resume")
            if getattr(opts, "remote_repeats", 1) != 1:
                parts.append(f"--remote-repeats {opts.remote_repeats}")
            if getattr(opts, "remote_iters", 0):
                parts.append(f"--remote-iters {opts.remote_iters}")
            if getattr(opts, "remote_add_args", ""):
                parts.append(f"--remote-add-arg {opts.remote_add_args}")
        return " \
  ".join(parts)

    def _eval_workflow_render_result(self, payload: Mapping[str, Any]) -> None:
        run_dir = str(payload.get("run_dir") or "").strip()
        manifest_path = str(payload.get("manifest_path") or "").strip()
        reports = list(payload.get("report_paths") or [])
        stages = list(payload.get("stage_results") or [])
        counts: Dict[str, int] = {}
        for st in stages:
            if isinstance(st, Mapping):
                key = str(st.get("status") or "unknown")
                counts[key] = counts.get(key, 0) + 1
        lines = [
            f"Status: {payload.get('status', '')}",
            f"Run name: {payload.get('run_id', '') or '(auto)'}",
            f"Results bundle: {run_dir}",
            f"Manifest: {manifest_path}",
            "",
            "Stage status counts:",
        ]
        if counts:
            for key in sorted(counts):
                lines.append(f"  - {key}: {counts[key]}")
        else:
            lines.append("  - no stage results recorded")
        energy_messages = _evaluation_energy_not_started_messages(payload)
        if energy_messages:
            lines.extend(["", "Energy:"])
            lines.extend(f"  - {message}" for message in energy_messages)
        lines.append("")
        lines.append("Reports:")
        for pth in reports:
            lines.append(f"  - {pth}")
        dashboard_path = Path(run_dir) / "reports" / "result_dashboard.json" if run_dir else Path()
        try:
            if dashboard_path.is_file():
                dash = json.loads(dashboard_path.read_text(encoding="utf-8"))
                lines.append("")
                lines.append("Result dashboard:")
                lines.append(f"  - {dashboard_path}")
                lines.append(f"  - {dashboard_summary_line(dash)}")
                cards = dash.get("models") if isinstance(dash.get("models"), list) else []
                for card in cards[:8]:
                    if not isinstance(card, Mapping):
                        continue
                    best = card.get("best_split") if isinstance(card.get("best_split"), Mapping) else {}
                    speed = card.get("speedup_vs_cpu")
                    speed_txt = "" if speed in (None, "") else f", speedup_vs_cpu={speed}"
                    lines.append(
                        "  - {model}: health={health}, accepted={accepted}, complete_splits={splits}, best={best_case}/{best_backend} {latency} ms, validation={validation}, hailo_runtime={hailo}{speed}".format(
                            model=card.get("model_id", ""),
                            health=card.get("health", ""),
                            accepted=card.get("accepted_cases", 0),
                            splits=card.get("complete_split_result_count", 0),
                            best_case=best.get("case_id", "-"),
                            best_backend=best.get("backend", "-"),
                            latency=best.get("total_latency_ms", "-"),
                            validation=card.get("validation_status", ""),
                            hailo=card.get("hailo_runtime_verified", False),
                            speed=speed_txt,
                        )
                    )
        except Exception as exc:
            lines.append("")
            lines.append(f"Result dashboard: could not read {dashboard_path}: {type(exc).__name__}: {exc}")

        suite_lines = []
        try:
            models_dir = Path(run_dir) / "models"
            if models_dir.is_dir():
                for mdir in sorted(models_dir.iterdir()):
                    # v49p: the normal Evaluation Workflow uses the existing
                    # BenchmarkSet/legacy_suite path as source of truth.  Do not
                    # advertise stale v49c-v49m generated_suite folders here.
                    for suite_name in ("legacy_suite", "suite"):
                        suite = mdir / "benchmark_set" / suite_name
                        if suite.is_dir():
                            suite_lines.append(str(suite))
        except Exception:
            suite_lines = []
        if suite_lines:
            lines.append("")
            lines.append("Authoritative benchmark suites:")
            for pth in suite_lines[:20]:
                lines.append(f"  - {pth}")
        smoke_lines = []
        validation_lines = []
        readiness_lines = []
        try:
            models_dir = Path(run_dir) / "models"
            if models_dir.is_dir():
                for mdir in sorted(models_dir.iterdir()):
                    vpath = mdir / "validation" / "validation_summary.json"
                    hpath = mdir / "hardware" / "hardware_smoke_status.json"
                    if not hpath.is_file():
                        hpath = mdir / "benchmark_results" / "hardware_smoke_report.json"
                    hailo_path = mdir / "benchmark_set" / "hailo_artifact_status.json"
                    build_status_path = mdir / "benchmark_set" / "hailo_build_service_status.json"
                    remote_path = mdir / "benchmark_results" / "remote_benchmark_status.json"
                    norm_path = mdir / "benchmark_results" / "normalized_results.json"
                    vp = {}
                    hp = {}
                    hailo = {}
                    build = {}
                    remote = {}
                    norm = {}
                    if vpath.is_file():
                        try:
                            vp = json.loads(vpath.read_text(encoding="utf-8"))
                            validation_lines.append(f"  - {mdir.name}: {vp.get('status', '')}, validated={vp.get('validated_result_count', 0)}, validation_ok={vp.get('validation_ok', None)}")
                        except Exception:
                            vp = {}
                    if hpath.is_file():
                        try:
                            hp = json.loads(hpath.read_text(encoding="utf-8"))
                            smoke_count = hp.get('measured_hardware_result_count', hp.get('normalized_result_count', hp.get('result_count', 0)))
                            smoke_lines.append(f"  - {mdir.name}: {hp.get('status', '')}, hardware_results={smoke_count}, hardware_verified={hp.get('hardware_verified', False)}, hailo_runtime_verified={hp.get('hailo_runtime_verified', False)}")
                        except Exception:
                            hp = {}
                    for _path, _target in ((hailo_path, "hailo"), (build_status_path, "build"), (remote_path, "remote"), (norm_path, "norm")):
                        if _path.is_file():
                            try:
                                value = json.loads(_path.read_text(encoding="utf-8"))
                            except Exception:
                                value = {}
                            if _target == "hailo":
                                hailo = value
                            elif _target == "build":
                                build = value
                            elif _target == "remote":
                                remote = value
                            else:
                                norm = value
                    hailo_status = hailo.get("status") or hailo.get("artifact_status") or "n/a"
                    build_status = build.get("status") or build.get("service_status") or "n/a"
                    remote_status = remote.get("status") or remote.get("service_status") or "n/a"
                    remote_reason = remote.get("reason") or remote.get("message") or ""
                    validation_status = vp.get("status") or "n/a"
                    hardware_status = hp.get("status") or "n/a"
                    results_count = norm.get("result_count") or norm.get("normalized_result_count") or hp.get("measured_hardware_result_count") or hp.get("normalized_result_count") or 0
                    split_rows = []
                    try:
                        split_rows = [r for r in list(norm.get("results") or []) if isinstance(r, Mapping) and str(r.get("variant") or "").lower() == "split"]
                    except Exception:
                        split_rows = []
                    complete_split_rows = [r for r in split_rows if r.get("total_latency_ms") not in (None, "") or (r.get("part1_latency_ms") not in (None, "") and r.get("part2_latency_ms") not in (None, ""))]
                    split_status = "measured" if complete_split_rows else ("component_only" if split_rows else "none")
                    extra = f", remote_reason={remote_reason}" if remote_reason else ""
                    readiness_lines.append(
                        f"  - {mdir.name}: validation={validation_status}, hardware={hardware_status}, "
                        f"hailo={hailo_status}, hailo_runtime={hp.get('hailo_runtime_verified', False)}, "
                        f"hailo_build={build_status}, remote={remote_status}, results={results_count}, split_total={split_status}{extra}"
                    )
        except Exception:
            pass
        if readiness_lines:
            lines.append("")
            lines.append("Model readiness summary:")
            lines.extend(readiness_lines[:20])
        if smoke_lines:
            lines.append("")
            lines.append("Hardware smoke:")
            lines.extend(smoke_lines[:20])
        if validation_lines:
            lines.append("")
            lines.append("Validation summary:")
            lines.extend(validation_lines[:20])
        try:
            status_summary_path = Path(run_dir) / "reports" / "run_status_summary.json" if run_dir else Path()
            if status_summary_path.is_file():
                status_summary = json.loads(status_summary_path.read_text(encoding="utf-8"))
                lines.append("")
                summary_status = str(status_summary.get("status") or payload.get("status") or "")
                if isinstance(status_summary.get("completion"), Mapping):
                    lines.append(str(status_summary["completion"].get("message") or ""))
                native_evidence = (
                    status_summary.get("native_evidence_status")
                    if isinstance(
                        status_summary.get("native_evidence_status"), Mapping,
                    )
                    else {}
                )
                if native_evidence:
                    lines.append("Native evidence axes:")
                    for label, key in (
                        ("Runtime", "runtime"),
                        ("Semantics", "semantics"),
                        ("Claim", "claim"),
                        ("Energy", "energy"),
                    ):
                        axis = (
                            native_evidence.get(key)
                            if isinstance(native_evidence.get(key), Mapping)
                            else {}
                        )
                        lines.append(
                            f"  - {label}: " + (
                                energy_axis_description(native_evidence)
                                if key == "energy" else str(axis.get('status', 'unavailable'))
                            )
                        )
                    lines.append(
                        "  - Scientific ready: "
                        + str(bool(native_evidence.get("scientific_ready")))
                    )
                    lines.append("")
                blocking = blocking_reasons_for_display(status_summary)
                non_blocking = list(status_summary.get("non_blocking_reasons") or [])
                if summary_status == "ok" and non_blocking and not blocking:
                    lines.append("Workflow completion notes:")
                    lines.append("  - Run is complete; remaining items below are non-blocking warnings.")
                else:
                    lines.append("Blocking partial / failure reasons:")
                if blocking:
                    for item in blocking[:15]:
                        if isinstance(item, Mapping):
                            model = str(item.get("model_id") or "workflow")
                            stage = str(item.get("stage") or item.get("kind") or "")
                            reason = str(item.get("reason") or item.get("message") or item.get("status") or item)
                            lines.append(f"  - {model}/{stage}: {reason}")
                        else:
                            lines.append(f"  - {item}")
                else:
                    lines.append("  - none recorded")
                if non_blocking:
                    lines.append("")
                    lines.append("Non-blocking warnings:")
                    for item in non_blocking[:15]:
                        if isinstance(item, Mapping):
                            model = str(item.get("model_id") or "workflow")
                            stage = str(item.get("stage") or item.get("kind") or "")
                            reason = str(item.get("reason") or item.get("message") or item.get("status") or item)
                            lines.append(f"  - {model}/{stage}: {reason}")
                        else:
                            lines.append(f"  - {item}")
                dbg = status_summary.get("debug_upload_hint") or {}
                if isinstance(dbg, Mapping) and dbg.get("primary_log"):
                    lines.append(f"Debug log: {Path(run_dir) / str(dbg.get('primary_log'))}")
        except Exception:
            pass
        warnings = list(payload.get("warnings") or [])
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in warnings[:20]:
                lines.append(f"  - {w}")
        self._eval_workflow_text_set("\n".join(lines) + "\n")
        try:
            self.var_eval_workflow_status.set(str(payload.get("status") or "finished"))
        except Exception:
            pass
        if run_dir:
            try:
                self.var_eval_workflow_last_run_dir.set(run_dir)
            except Exception:
                pass
            self._last_evaluation_workflow_run_dir = run_dir

    def _evaluation_workflow_open_profile_editor(self) -> None:
        """Open the GUI profile builder for Evaluation Profile YAML files."""
        try:
            open_evaluation_profile_editor(self)
        except Exception as exc:
            logger.exception("Failed to open Evaluation Profile editor")
            messagebox.showerror("Evaluation Profile", f"Profile editor could not be opened:\n\n{type(exc).__name__}: {exc}")

    def _queue_evaluation_workflow(self, *, resume: Optional[bool] = None, force_stages: Optional[list[str]] = None, refresh_remote_bundles: bool = False, rerun_generated: bool = False) -> Optional[str]:
        audit_start_summary: Dict[str, Any] = {}
        try:
            opts = getattr(self, "_eval_finalize_partial_options_override", None) or self._eval_workflow_snapshot_options(resume_override=resume)
            if force_stages:
                existing = list(getattr(opts, "force_stage", []) or [])
                for _st in force_stages:
                    if _st not in existing:
                        existing.append(_st)
                opts.force_stage = existing
            if rerun_generated:
                # Re-use already generated benchmark suites/artifacts, but run
                # current benchmark/validation/report code.  This is the fast
                # path after updating runner templates, report logic or remote
                # dispatch code.
                opts.resume = True
                opts.execution_mode = "generate_and_run"
                opts.skip_benchmarks = False
                # Important: do not rebuild BenchmarkSets/HEFs/DXNNs when the
                # purpose is only to refresh runner templates, remote bundles,
                # validation, visual verification or reports.
                opts.rerun_generated_only = True
            if refresh_remote_bundles:
                opts.remote_reuse_bundle = False
                opts.remote_no_reuse_bundle = True
                opts.remote_resume = False
                opts.remote_no_resume = True
            if not bool(getattr(opts, "resume", False)) and not rerun_generated:
                resolved_profile = dict(
                    (
                        getattr(opts, "profile_start_snapshot", {}) or {}
                    ).get("resolved_profile")
                    or {}
                )
                audit_start_summary = (
                    panel_evaluation_workflow.score_independent_audit_start_summary(
                        build_effective_execution_plan(resolved_profile)
                    )
                    if resolved_profile
                    else {}
                )
                if audit_start_summary and not messagebox.askyesno(
                    "Ranking-Audit starten?",
                    str(audit_start_summary.get("confirmation_text") or ""),
                    parent=self,
                    default=messagebox.NO,
                    icon=messagebox.WARNING,
                ):
                    try:
                        self.var_eval_workflow_status.set(
                            "Start abgebrochen: Ranking-Audit nicht bestätigt."
                        )
                    except Exception:
                        pass
                    return None
            if not panel_evaluation_workflow.validate_productive_build_start(opts, parent=self):
                self.var_eval_workflow_status.set("Start blockiert: Force muss AUS sein.")
                return None
            if not panel_evaluation_workflow.confirm_force_build_start(opts, parent=self):
                try:
                    self.var_eval_workflow_status.set(
                        "Start abgebrochen: Force-Neubau nicht bestätigt."
                    )
                except Exception:
                    pass
                return None
        except Exception as exc:
            messagebox.showwarning("Evaluation Workflow", str(exc))
            return None
        out_root = Path(opts.out).expanduser()
        out_root.mkdir(parents=True, exist_ok=True)
        latest_log = out_root / "_latest_evaluation_workflow.log"
        run_label = str(opts.run_id or time.strftime("%Y%m%d_%H%M%S"))
        job_id = _new_evaluation_workflow_job_id()
        workflow_cancel = threading.Event()
        workflow_runner: Dict[str, Any] = {}

        def _cancel_workflow() -> bool:
            runner = workflow_runner.get("runner")
            if runner is not None:
                if runner.request_cancel("gui_user_requested") is False:
                    return False
            workflow_cancel.set()
            return True
        _start_snapshot = dict(getattr(opts, "profile_start_snapshot", {}) or {})
        _resolved_selection = dict(_start_snapshot.get("resolved_selection") or {})
        initial = [
            f"Profile request: {opts.profile}",
            f"Resolved profile: {_resolved_selection.get('profile_name') or _start_snapshot.get('profile_id') or opts.profile}",
            (
                f"Resolved start selection: mode={_resolved_selection.get('run_mode') or '?'} "
                f"native={'on' if _resolved_selection.get('native_enabled') else 'off'} "
                f"native_energy={'on' if _resolved_selection.get('energy_enabled') else 'off'} "
                f"models={','.join(_resolved_selection.get('models') or []) or 'none'}"
            ),
            f"Start snapshot: {_start_snapshot.get('snapshot_sha256') or '(unavailable)'}",
            f"Models root: {opts.models_root or '(from profile / unresolved placeholders)'}",
            f"Results parent folder: {opts.out}",
            f"Resume: {opts.resume}",
            f"Execution mode: {getattr(opts, 'execution_mode', 'contracts_only')}",
            f"Force stages: {', '.join(getattr(opts, 'force_stage', []) or []) or '(none)'}",
            "Force build start confirmation: " + (
                ", ".join(getattr(opts, "force_build_confirmed_backends", ()) or ())
                or "not required"
            ),
            f"Rerun generated suites/no rebuild: {bool(rerun_generated)}",
            f"Refresh remote bundles: {bool(refresh_remote_bundles)}",
            (
                "Ranking-audit start confirmation: accepted · "
                + str(audit_start_summary.get("concise") or "")
                if audit_start_summary
                else "Ranking-audit start confirmation: not required"
            ),
            f"Skip runtime benchmarks: {opts.skip_benchmarks}",
            f"No remote: {opts.no_remote}",
            f"Remote host: {getattr(opts, 'remote_user', '') + '@' if getattr(opts, 'remote_user', '') else ''}{getattr(opts, 'remote_host', '') or '(none selected)'}",
            f"Benchmark provider: {getattr(opts, 'benchmark_provider', '') or '(plan/default)'}",
            f"Benchmark warmup/runs/timeout: {getattr(opts, 'benchmark_warmup', 1)}/{getattr(opts, 'benchmark_runs', 3)}/{getattr(opts, 'benchmark_timeout_s', 0)}",
            f"Remote provider/warmup/iters/timeout: {getattr(opts, 'remote_provider', 'auto')}/{getattr(opts, 'remote_warmup', 10)}/{getattr(opts, 'remote_iters', 50)}/{getattr(opts, 'remote_timeout_s', 0)}",
            f"Hailo build policy: {getattr(opts, 'hailo_build_mode', 'reuse_only')} targets={','.join(str(x) for x in getattr(opts, 'hailo_build_targets', []) or []) or getattr(opts, 'hailo_hw_arch', 'hailo8')} build full/part1/part2={getattr(opts, 'hailo_build_full', True)}/{getattr(opts, 'hailo_build_part1', True)}/{getattr(opts, 'hailo_build_part2', True)} preset={getattr(opts, 'hailo_preset', 'quick')}",
            f"Validation mode: {getattr(opts, 'validation_mode', 'summary_only')}",
            f"Hardware smoke mode: {getattr(opts, 'hardware_smoke_mode', 'summary_only')}",
            f"Debug log: {latest_log}",
            "",
            "CLI equivalent:",
            self._eval_workflow_command_preview(opts),
        ]
        self._jobs_register(
            job_id=job_id,
            kind="evaluation_workflow",
            type_label="Evaluation workflow",
            title=f"Evaluation Workflow — {opts.profile}",
            name=str(opts.run_id or opts.profile),
            output_dir=str(out_root),
            log_path=str(latest_log),
            initial_status="Starting formal evaluation workflow…",
            initial_lines=initial,
            progress_maximum=1.0,
            cancel_callback=_cancel_workflow,
            can_cancel=True,
            geometry="1040x560",
        )
        try:
            self.var_eval_workflow_status.set("running")
        except Exception:
            pass
        self._eval_workflow_text_set("Workflow started. Live logs will appear below, in the Jobs tab, and in the debug log file.\n\n" + "\n".join(initial) + "\n")

        observer_lock = threading.Lock()
        pending_progress: Dict[str, Any] = {}
        pending_terminal_log: Dict[str, Any] = {}

        def _publish_log(line: str) -> None:
            self._jobs_append_log(job_id, line)
            self._eval_workflow_text_append(line + "\n")

        def _flush_terminal_log() -> None:
            with observer_lock:
                line = str(pending_terminal_log.pop("line", ""))
                pending_terminal_log.clear()
            if line:
                _publish_log(line)

        def _log(line: str) -> None:
            # A stalled/closed observer cannot build an unbounded queue of
            # hash heartbeats.  Detailed durable progress uses the parent log.
            line = str(line)
            if "[workflow] finalize_artifacts /" in line:
                with observer_lock:
                    pending_terminal_log["line"] = line
                    if pending_terminal_log.get("scheduled"):
                        return
                    pending_terminal_log["scheduled"] = True
                self.root.after(0, _flush_terminal_log)
            else:
                self.root.after(0, lambda _s=line: _publish_log(_s))

        def _flush_progress() -> None:
            with observer_lock:
                state = pending_progress.pop("state", None)
                pending_progress.clear()
            if state is None:
                return
            record = self._background_jobs.get(job_id)
            if record is None or str(record.status) not in {"running", "queued", "cancelling"}:
                return
            done, total, label = state
            self._jobs_set_progress(
                job_id, value=float(done) / float(total), label=label,
                display=f"{done}/{total}", progress_maximum=1.0,
            )
            self.var_eval_workflow_status.set(f"running: {done}/{total} {label}")

        def _progress(done: int, total: int, label: str) -> None:
            total = max(1, int(total or 1))
            done = max(0, min(total, int(done or 0)))
            with observer_lock:
                pending_progress["state"] = (done, total, str(label))
                if pending_progress.get("scheduled"):
                    return
                pending_progress["scheduled"] = True
            self.root.after(0, _flush_progress)

        def _job_event(event: Mapping[str, Any]) -> None:
            self.root.after(
                0,
                lambda _event=dict(event or {}), _scope=job_id: (
                    self._jobs_handle_workflow_job_event(
                        _event,
                        output_dir=str(out_root),
                        log_path=str(latest_log),
                        workflow_scope=_scope,
                    )
                ),
            )

        def worker() -> None:
            try:
                if workflow_cancel.is_set():
                    if not bool(getattr(self, "_gui_closing", False)):
                        self.root.after(
                            0,
                            lambda: self._jobs_finish(
                                job_id,
                                status="cancelled",
                                message="Evaluation workflow cancelled before start.",
                                output_dir=str(out_root),
                                log_path=str(latest_log),
                            ),
                        )
                    return
                _log("[gui] Worker thread started; loading profile and creating Results Bundle...")
                runner = EvaluationWorkflowRunner(opts, log=_log, progress=_progress, job_event=_job_event)
                workflow_runner["runner"] = runner
                if workflow_cancel.is_set():
                    runner.request_cancel("gui_user_requested_before_start")
                result = runner.run()
                payload = result.to_dict()
                from ..workflow.evidence_status import workflow_completion_projection
                summary_file = Path(str(payload.get("run_dir") or "")) / "reports" / "run_status_summary.json"
                try:
                    saved_summary = json.loads(summary_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    saved_summary = {}
                completion = saved_summary.get("completion") or workflow_completion_projection(result.status)
                payload["completion"] = completion
                energy_messages = _evaluation_energy_not_started_messages(
                    payload
                )
                energy_message = "\n".join(energy_messages)
                status = (
                    str(completion.get("severity") or "error")
                )
                if bool(getattr(self, "_gui_closing", False)):
                    return
                self.root.after(0, lambda _p=payload: self._eval_workflow_render_result(_p))
                self.root.after(
                    0,
                    lambda _p=payload, _status=status, _energy=energy_message: self._jobs_finish(
                        job_id,
                        status=_status,
                        message=(
                            f"{(_p.get('completion') or {}).get('message')}: {_p.get('run_dir')}"
                            + (f" — {_energy}" if _energy else "")
                        ),
                        output_dir=str(_p.get("run_dir") or out_root),
                        log_path=str(latest_log),
                    ),
                )
                self.root.after(0, lambda _p=payload: self.var_eval_workflow_status.set(str((_p.get('completion') or {}).get('label') or _p.get('status'))))
                self.root.after(
                    0,
                    lambda _p=payload, _energy=energy_message: None
                    if bool(getattr(self, "_gui_closing", False))
                    else (
                        messagebox.showinfo
                        if (_p.get("completion") or {}).get("severity") == "success"
                        else messagebox.showwarning
                    )(
                        "Evaluation Workflow",
                        f"{(_p.get('completion') or {}).get('message')}"
                        f"\n\nResults bundle:\n{_p.get('run_dir')}"
                        f"\n\nDebug log:\n"
                        f"{Path(str(_p.get('run_dir') or out_root)) / 'evaluation_workflow.log'}"
                        + (f"\n\n{_energy}" if _energy else ""),
                    ),
                )
            except Exception as exc:
                logger.exception("Evaluation workflow failed")
                msg = f"{type(exc).__name__}: {exc}"
                owner = workflow_runner.get("runner")
                measurement_status = str(getattr(owner, "_measurement_phase_status", "") or "")
                if measurement_status:
                    msg = (
                        f"Messphase beendet: {measurement_status}\n"
                        f"Abschluss fehlgeschlagen:\n{msg}"
                    )
                if bool(getattr(self, "_gui_closing", False)):
                    return
                status = "cancelled" if isinstance(exc, WorkflowRunCancelledError) else "error"
                self.root.after(0, lambda _msg=msg, _status=status: self._jobs_finish(job_id, status=_status, message=_msg, output_dir=str(out_root), log_path=str(latest_log)))
                self.root.after(0, lambda _msg=msg: self.var_eval_workflow_status.set("failed"))
                self.root.after(0, lambda _msg=msg: self._eval_workflow_text_set("Workflow failed.\n\n" + _msg + "\n"))
                self.root.after(
                    0,
                    lambda _msg=msg: None
                    if bool(getattr(self, "_gui_closing", False))
                    else messagebox.showerror("Evaluation Workflow", _msg),
                )

        workflow_thread = threading.Thread(
            target=worker,
            name=f"evaluation-workflow-{job_id}",
            daemon=False,
        )
        workflow_record = self._background_jobs.get(job_id)
        if workflow_record is not None:
            workflow_record.worker_thread = workflow_thread
        workflow_thread.start()
        return job_id

    def _queue_evaluation_workflow_rerun_generated(self) -> Optional[str]:
        """Rerun benchmark/validation/report stages for the latest generated EvaluationRun.

        This keeps resolved models, analysis, selected candidates, generated
        BenchmarkSet suites and backend artifacts, but refreshes runner templates,
        remote bundles, result ingestion, validation summaries and reports.  It is
        intended for the common development loop: fix the runner/reporting code,
        then re-measure the already-generated suite without paying the generation
        and compile cost again.
        """
        stages = ["run_benchmarks", "validate_outputs", "hardware_smoke", "aggregate_results", "generate_report"]
        return self._queue_evaluation_workflow(resume=True, force_stages=stages, refresh_remote_bundles=True, rerun_generated=True)


    def _queue_evaluation_workflow_finalize_partial(self) -> Optional[str]:
        """Finalize reports from completed model artifacts without running heavy stages."""
        stages = ["validate_outputs", "hardware_smoke", "aggregate_results", "generate_report"]
        try:
            opts = self._eval_workflow_snapshot_options(resume_override=True)
            opts.resume = True
            opts.skip_benchmarks = True
            opts.no_remote = True
            opts.energy_enabled = False
            existing = list(getattr(opts, "force_stage", []) or [])
            for _st in stages:
                if _st not in existing:
                    existing.append(_st)
            opts.force_stage = existing
            # Stash a one-shot override consumed by _queue_evaluation_workflow below.
            self._eval_finalize_partial_options_override = opts
        except Exception as exc:
            messagebox.showwarning("Evaluation Workflow", str(exc))
            return None
        try:
            return self._queue_evaluation_workflow(resume=True, force_stages=stages)
        finally:
            try:
                self._eval_finalize_partial_options_override = None
            except Exception:
                pass

    def _evaluation_workflow_last_run_dir(
        self,
        *,
        prefer_latest: bool = False,
        purpose: str = "debug",
    ) -> Path:
        """Return a valid EvaluationRun, recovering from stale persisted GUI state.

        Long EvaluationRuns often finish after the GUI was restarted.  Earlier
        versions trusted ``var_eval_workflow_last_run_dir`` unconditionally,
        which made Analysis/Debug Pack creation point at an older run.  v60d
        validates the explicit path and otherwise discovers the newest run from
        manifests and ``_latest_evaluation_workflow.log``.  Pack creation passes
        ``prefer_latest=True`` so an old-but-valid GUI selection cannot package a
        previous run after a newer workflow completed.
        """
        raw_gui = str(getattr(getattr(self, "var_eval_workflow_last_run_dir", None), "get", lambda: "")() or "").strip()
        raw_memory = str(getattr(self, "_last_evaluation_workflow_run_dir", "") or "").strip()
        out_root_raw = str(
            getattr(getattr(self, "var_eval_workflow_out_root", None), "get", lambda: "")()
            or self._eval_workflow_default_out_root()
        ).strip()
        out_root = Path(out_root_raw).expanduser()
        latest_logs = [out_root / "_latest_evaluation_workflow.log"]
        for raw in (raw_gui, raw_memory):
            if raw:
                try:
                    latest_logs.append(Path(raw).expanduser().parent / "_latest_evaluation_workflow.log")
                except Exception:
                    pass
        result = discover_evaluation_run(
            preferred=[raw_gui, raw_memory],
            output_roots=[out_root],
            latest_logs=latest_logs,
            prefer_valid_explicit=not prefer_latest,
            purpose=purpose,
        )
        self._evaluation_workflow_run_discovery = result.as_dict()
        selected = Path(result.selected).expanduser()
        if result.status != "not_found" and is_evaluation_run_dir(
            selected, purpose=purpose
        ):
            try:
                self.var_eval_workflow_last_run_dir.set(str(selected))
            except Exception:
                pass
            self._last_evaluation_workflow_run_dir = str(selected)
            return selected
        # Keep the old fallback for a not-yet-started workflow, but callers that
        # need an existing run validate it before opening/packing.
        return selected if str(selected) else out_root

    def _evaluation_workflow_pack_output_dir(self) -> Path:
        """Return a writable export directory without touching the source run."""

        configured = str(
            os.environ.get("ONNX_SPLITPOINT_EXPORT_DIR", "") or ""
        ).strip()
        export_dir = (
            Path(configured).expanduser()
            if configured
            else Path.home() / "Downloads"
        )
        require_write_target(
            export_dir,
            operation="Evaluation pack export",
            minimum_free_bytes=16 * 1024 * 1024,
            minimum_free_inodes=16,
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir.resolve()

    def _evaluation_workflow_no_run_message(self, requested: Path) -> str:
        diag = getattr(self, "_evaluation_workflow_run_discovery", {}) or {}
        roots = "\n".join(f"  - {p}" for p in list(diag.get("searched_roots") or [])) or "  - (none)"
        logs = "\n".join(f"  - {p}" for p in list(diag.get("log_candidates") or [])) or "  - (none)"
        return (
            "No Results Bundle found yet.\n\n"
            f"Selected/requested path:\n  {requested}\n\n"
            f"Searched EvaluationRuns roots:\n{roots}\n\n"
            f"Latest-workflow logs:\n{logs}"
        )

    def _evaluation_workflow_open_results_folder(self) -> None:
        self._open_path(str(self._evaluation_workflow_last_run_dir()))

    def _evaluation_workflow_open_manifest(self) -> None:
        run_dir = self._evaluation_workflow_last_run_dir()
        manifest = run_dir / "run_manifest.json"
        if manifest.is_file():
            self._open_path(str(manifest))
        else:
            messagebox.showinfo("Evaluation Workflow", f"No run_manifest.json found yet:\n\n{manifest}")

    def _evaluation_workflow_open_reports(self) -> None:
        run_dir = self._evaluation_workflow_last_run_dir()
        reports = run_dir / "reports"
        if reports.is_dir():
            self._open_path(str(reports))
        else:
            messagebox.showinfo("Evaluation Workflow", f"No reports folder found yet:\n\n{reports}")

    def _evaluation_workflow_open_dashboard(self) -> None:
        run_dir = self._evaluation_workflow_last_run_dir()
        candidates = [
            run_dir / "reports" / "result_dashboard.md",
            run_dir / "reports" / "result_dashboard.json",
            run_dir / "reports",
        ]
        for path in candidates:
            if path.is_file() or path.is_dir():
                self._open_path(str(path))
                return
        messagebox.showinfo("Evaluation Workflow", f"No v50 result dashboard found yet:\n\n{run_dir / 'reports' / 'result_dashboard.md'}")

    def _evaluation_workflow_open_benchmark_suite(self) -> None:
        run_dir = self._evaluation_workflow_last_run_dir()
        candidates = []
        try:
            models_dir = run_dir / "models"
            if models_dir.is_dir():
                for mdir in sorted(models_dir.iterdir()):
                    # Prefer the existing BenchmarkSet generator output.  The
                    # reduced generated_suite path is retained only for explicit
                    # diagnostics/old-run compatibility and is no longer opened
                    # from the standard Workflow button.
                    for suite_name in ("legacy_suite", "suite"):
                        suite = mdir / "benchmark_set" / suite_name
                        if suite.is_dir():
                            candidates.append(suite)
        except Exception:
            candidates = []
        if candidates:
            self._open_path(str(candidates[0]))
        else:
            messagebox.showinfo("Evaluation Workflow", f"No BenchmarkSet suite found yet under:\n\n{run_dir / 'models'}")

    def _evaluation_workflow_open_debug_log(self) -> None:
        run_dir = self._evaluation_workflow_last_run_dir()
        candidates = [run_dir / "evaluation_workflow.log"]
        try:
            candidates.append(run_dir.parent / "_latest_evaluation_workflow.log")
        except Exception:
            pass
        for p in candidates:
            if p.is_file():
                self._open_path(str(p))
                return
        messagebox.showinfo("Evaluation Workflow", f"No evaluation workflow log found yet. Looked under:\n\n{run_dir}")

    def _evaluation_workflow_prepare_debug_pack(self) -> None:
        """Create the canonical compact, verified Debug Pack."""

        run_dir = self._evaluation_workflow_last_run_dir(
            prefer_latest=True,
            purpose="debug",
        )
        if not run_dir.is_dir():
            messagebox.showinfo(
                "Evaluation Workflow",
                self._evaluation_workflow_no_run_message(run_dir),
            )
            return
        try:
            from ..workflow.debug_pack import create_evaluation_debug_pack

            run_id = run_dir.name or time.strftime("%Y%m%d_%H%M%S")
            out_zip = (
                self._evaluation_workflow_pack_output_dir()
                / f"{run_id}_debug_pack.zip"
            )
            result = create_evaluation_debug_pack(
                run_dir,
                out_zip,
                source_selection_policy=(
                    "newest_identified_evaluation_run_for_debug"
                ),
            )
            self._open_path(str(out_zip))
            messagebox.showinfo(
                "Evaluation Workflow",
                "Kompaktes Debug Pack erstellt und vollständig geprüft:\n\n"
                f"{out_zip}\n\n"
                f"SHA-256: {result.get('archive_sha256')}\n"
                f"Größe: {result.get('archive_size_bytes')} Bytes\n\n"
                "Das vollständige Workflow-Log und vorhandene Audit-Evidenz "
                "sind enthalten. Binärdumps, Abbildungen und Replay-Payloads "
                "wurden bewusst ausgeschlossen.",
            )
        except Exception as exc:
            messagebox.showerror(
                "Evaluation Workflow",
                "Debug Pack konnte nicht erstellt werden:\n\n"
                f"{type(exc).__name__}: {exc}",
            )

    def _evaluation_workflow_prepare_analysis_pack(self) -> None:
        """Create the current versioned thesis/paper analysis ZIP.

        The pack consumes ``reports/scientific``.  When that directory is
        absent (for example in a copied/older Results Bundle), the canonical
        reporter is run first.  Removed v59 claim tables are never required.
        """
        run_dir = self._evaluation_workflow_last_run_dir(
            prefer_latest=True,
            purpose="analysis",
        )
        if not is_evaluation_run_dir(run_dir, purpose="analysis"):
            messagebox.showinfo("Evaluation Workflow", self._evaluation_workflow_no_run_message(run_dir))
            return
        try:
            from ..workflow.analysis_pack import create_analysis_pack
            run_id = run_dir.name or time.strftime("%Y%m%d_%H%M%S")
            out_zip = self._evaluation_workflow_pack_output_dir() / f"{run_id}_analysis_pack.zip"
            result = create_analysis_pack(
                run_dir,
                out_zip,
                tool_version=TOOL_VERSION,
                materialize_missing_report=False,
            )
            try:
                self.var_eval_workflow_status.set(f"analysis pack ready: {run_id}")
            except Exception:
                pass
            self._open_path(str(out_zip))
            messagebox.showinfo(
                "Analyse Pack",
                "Analyse Pack erstellt:\n\n"
                f"{out_zip}\n\n"
                f"Modelle: {result.get('model_count', 0)} · "
                f"kanonische Rows: {result.get('row_count', 0)} · "
                f"Ranking-Gruppen: {result.get('ranking_method_group_count', 0)} · "
                f"Dateien: {result.get('file_count', 0)}",
            )
        except Exception as exc:
            messagebox.showerror(
                "Analyse Pack",
                "Analyse Pack konnte nicht erstellt werden:\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                f"Run: {run_dir}",
            )

    def _evaluation_workflow_check_validation_assets(self) -> None:
        try:
            from ..benchmark.validation_assets import validation_assets_status
            data = validation_assets_status().as_dict()
            text = json.dumps(data, indent=2, ensure_ascii=False)
            self._eval_workflow_text_set("Validation assets status:\n\n" + text + "\n")
            try:
                self.var_eval_workflow_status.set("validation assets checked")
            except Exception:
                pass
            try:
                self.var_tool_config_validation_status.set(
                    f"root={data.get('root')} · COCO-50={data.get('coco50_images', 0)} · COCO-200={data.get('coco200_images', 0)} · "
                    f"Imagenette200={data.get('imagenette200_images', 0)} · Imagenette500={data.get('imagenette500_images', 0)}"
                )
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Validation assets", f"Could not check validation assets:\n\n{type(exc).__name__}: {exc}")

    def _queue_evaluation_validation_assets(self) -> Optional[str]:
        job_id = f"validation-assets-{time.strftime('%Y%m%d_%H%M%S')}"
        self._jobs_register(
            job_id=job_id,
            kind="validation_assets",
            type_label="Validation assets",
            title="Prepare validation assets",
            name="COCO-50/200 / classification mini / test images",
            initial_status="Preparing validation assets…",
            initial_lines=["Preparing validation/calibration assets used by Evaluation Workflow profiles."],
            progress_maximum=1.0,
            can_cancel=False,
            geometry="900x460",
        )
        def _log(line: str) -> None:
            self.root.after(0, lambda _s=str(line): self._jobs_append_log(job_id, _s))
            self.root.after(0, lambda _s=str(line): self._eval_workflow_text_append(_s + "\n"))
        def worker() -> None:
            try:
                from ..benchmark.validation_assets import prepare_all_validation_assets
                out = prepare_all_validation_assets(include_coco50=True, include_coco200=True, include_imagenette200=True, include_imagenette500=True, include_test_images=True, log=_log)
                text = json.dumps(out, indent=2, ensure_ascii=False, default=str)
                self.root.after(0, lambda: self._jobs_set_progress(job_id, value=1.0, label="Validation assets prepared", display="1/1", progress_maximum=1.0))
                self.root.after(0, lambda _text=text: self._eval_workflow_text_set("Validation assets prepared:\n\n" + _text + "\n"))
                def _set_val_status() -> None:
                    try:
                        data = out.get('status') if isinstance(out, dict) and isinstance(out.get('status'), dict) else out
                        self.var_tool_config_validation_status.set(
                            f"prepared · root={data.get('root', '')} · COCO-50={data.get('coco50_images', data.get('coco50_ready', ''))} · "
                            f"COCO-200={data.get('coco200_images', data.get('coco200_ready', ''))} · "
                            f"Imagenette200={data.get('imagenette200_images', data.get('imagenette200_ready', ''))} · Imagenette500={data.get('imagenette500_images', data.get('imagenette500_ready', ''))}"
                        )
                    except Exception:
                        pass
                self.root.after(0, _set_val_status)
                self.root.after(0, lambda: self._jobs_finish(job_id, status="success", message="Validation assets prepared."))
            except Exception as exc:
                logger.exception("Validation asset preparation failed")
                msg = f"{type(exc).__name__}: {exc}"
                self.root.after(0, lambda _msg=msg: self._jobs_finish(job_id, status="error", message=_msg))
                self.root.after(0, lambda _msg=msg: messagebox.showerror("Validation assets", _msg))
        threading.Thread(target=worker, daemon=True).start()
        return job_id

    def _queue_profile_campaign(self, *, resume: bool = False) -> Optional[str]:
        profile_request = str(getattr(self, 'var_bench_evaluation_profile', tk.StringVar(value='')).get() or '').strip()
        if not profile_request:
            messagebox.showwarning('Evaluation profile', 'Please select an evaluation profile first.')
            return None
        models_root_raw = str(getattr(self, 'var_bench_profile_models_root', tk.StringVar(value='')).get() or '').strip()
        include_reserve = bool(getattr(self, 'var_bench_profile_include_reserve', tk.BooleanVar(value=False)).get())
        auto_remote_run = bool(getattr(self, 'var_bench_profile_auto_remote', tk.BooleanVar(value=False)).get())
        auto_export_analysis = bool(getattr(self, 'var_bench_profile_auto_analysis', tk.BooleanVar(value=False)).get())
        model_preparation_mode = normalize_model_preparation_mode(str(getattr(self, 'var_bench_model_preparation_mode', tk.StringVar(value='Use current ONNX')).get() or '').strip())
        work_root = Path(getattr(self, 'default_output_dir', '.') or '.').expanduser()
        if not models_root_raw:
            # Model root is optional when an external Evaluation Profile contains
            # explicit ONNX paths.  Otherwise the campaign will scan this
            # fallback root and report missing models in a structured way.
            try:
                from ..benchmark.evaluation_profiles import resolve_evaluation_profile_source
                src = resolve_evaluation_profile_source(profile_request)
                if src is not None and Path(src).is_file():
                    models_root_raw = str(Path(src).parent)
            except Exception:
                models_root_raw = ''
        if not models_root_raw:
            models_root_raw = str(work_root)
        benchmark_parent_dir = work_root
        evaluation_run_root = work_root / 'EvaluationRuns'
        try:
            layout = ensure_workdir(work_root)
            benchmark_parent_dir = layout.benchmark_sets
            evaluation_run_root = layout.root / 'EvaluationRuns'
        except Exception:
            benchmark_parent_dir = benchmark_parent_dir
            evaluation_run_root = benchmark_parent_dir / 'EvaluationRuns'
        try:
            eval_out = str(getattr(getattr(self, 'var_eval_workflow_out_root', None), 'get', lambda: '')() or '').strip()
            if eval_out:
                _eval_path = Path(eval_out).expanduser()
                if not _eval_path.is_absolute():
                    _eval_path = (Path(getattr(self, 'default_output_dir', '.') or '.').expanduser() / _eval_path)
                evaluation_run_root = _eval_path
        except Exception:
            pass
        eval_out_raw = str(getattr(self, 'var_eval_workflow_out', tk.StringVar(value='')).get() or '').strip()
        if eval_out_raw:
            _eval_path = Path(eval_out_raw).expanduser()
            if not _eval_path.is_absolute():
                _eval_path = (Path(getattr(self, 'default_output_dir', '.') or '.').expanduser() / _eval_path)
            evaluation_run_root = _eval_path
        workflow_run_id = str(getattr(self, 'var_eval_workflow_run_id', tk.StringVar(value='')).get() or '').strip() or None

        remote_defaults = None
        if auto_remote_run:
            try:
                remote_defaults = snapshot_remote_defaults(self)
            except Exception as exc:
                messagebox.showerror('Profile campaign', f'Invalid remote benchmark settings:\n\n{exc}')
                return None
            if remote_defaults.get('host') is None:
                messagebox.showwarning('Profile campaign', 'Auto remote run is enabled, but no remote host is selected.')
                return None

        options = ProfileCampaignOptions(
            profile_request=profile_request,
            models_root=Path(models_root_raw).expanduser(),
            benchmark_parent_dir=Path(benchmark_parent_dir),
            include_reserve=include_reserve,
            auto_remote_run=auto_remote_run,
            auto_export_analysis=auto_export_analysis,
            remote_host=(remote_defaults or {}).get('host'),
            remote_defaults=remote_defaults,
            model_preparation_mode=model_preparation_mode,
            preparation_output_root=Path(benchmark_parent_dir) / '_prepared_models',
            preparation_runtime=snapshot_preparation_runtime(self),
            evaluation_run_root=Path(evaluation_run_root),
            workflow_run_id=workflow_run_id,
            resume_workflow=bool(resume),
        )

        run_id = time.strftime('%Y%m%d_%H%M%S')
        job_id = f'profile-campaign-{run_id}'
        cancel_event = threading.Event()
        self._jobs_register(
            job_id=job_id,
            kind='profile_campaign',
            type_label='Evaluation workflow',
            title=f'Evaluation workflow — {profile_request}',
            name=str(Path(models_root_raw).name or profile_request),
            output_dir=str(evaluation_run_root),
            initial_status='Resuming evaluation workflow…' if bool(resume) else 'Preparing evaluation workflow…',
            initial_lines=[
                f'Profile: {profile_request}',
                f'Model root: {models_root_raw}',
                f'Resume: {bool(resume)}',
                f'Run ID: {workflow_run_id or "(auto)"}',
                f'Include reserve models: {include_reserve}',
                f'Auto remote run: {auto_remote_run}',
                f'Auto export analysis: {auto_export_analysis}',
                f'Model preparation: {model_preparation_mode}',
                f'EvaluationRuns root: {evaluation_run_root}',
                'The workflow runs per model as: optional preparation → analysis/prediction artifacts → benchmark set → optional remote run → optional benchmark analysis → reports.',
            ],
            progress_maximum=1.0,
            cancel_callback=lambda: cancel_event.set(),
            can_cancel=True,
            geometry='980x480',
        )

        def _log(line: str) -> None:
            self.root.after(0, lambda _s=str(line): self._jobs_append_log(job_id, _s))
            self.root.after(0, lambda _s=str(line): self._eval_workflow_text_append(_s + "\n"))

        def _progress(pct: float, label: str) -> None:
            self.root.after(
                0,
                lambda _p=float(pct), _lbl=str(label): self._jobs_set_progress(
                    job_id,
                    value=max(0.0, min(1.0, _p)),
                    label=_lbl,
                    display=f"{int(round(max(0.0, min(1.0, _p)) * 100.0))}%",
                    progress_maximum=1.0,
                ),
            )

        def worker() -> None:
            try:
                out = run_profile_campaign(self, options, job_id=job_id, log=_log, progress=_progress, cancel_event=cancel_event)
                run_dir = str(out.get('workflow_run_dir') or evaluation_run_root)
                self.root.after(0, lambda _run_dir=run_dir: setattr(self, '_last_evaluation_workflow_dir', _run_dir))
                self.root.after(0, lambda _run_dir=run_dir: getattr(self, 'var_eval_workflow_latest_run', tk.StringVar(value='')).set(_run_dir))
                self.root.after(0, lambda _run_dir=run_dir: getattr(self, 'var_eval_workflow_status', tk.StringVar(value='')).set(f'Workflow finished: {_run_dir}'))
                self.root.after(0, lambda _out=out, _run_dir=run_dir: self._jobs_finish(job_id, status='success', message=f"Processed {int(_out.get('models_processed') or 0)} model(s).", output_dir=_run_dir))
                self.root.after(0, lambda _out=out, _run_dir=run_dir: messagebox.showinfo('Evaluation workflow', f"Workflow finished.\n\nProcessed models: {int(_out.get('models_processed') or 0)}\nResults bundle root:\n{_run_dir}"))
            except Exception as exc:
                logger.exception('Profile campaign failed')
                msg = f'{type(exc).__name__}: {exc}'
                status = 'cancelled' if cancel_event.is_set() else 'error'
                self.root.after(0, lambda _msg=msg, _status=status: self._jobs_finish(job_id, status=_status, message=_msg, output_dir=str(evaluation_run_root)))
                if cancel_event.is_set():
                    self.root.after(0, lambda _msg=msg: messagebox.showwarning('Profile campaign', _msg))
                else:
                    self.root.after(0, lambda _msg=msg: messagebox.showerror('Evaluation workflow', _msg))

        threading.Thread(target=worker, daemon=True).start()
        return job_id



    def _evaluation_run_root(self) -> Path:
        work_root = Path(getattr(self, 'default_output_dir', '.') or '.').expanduser()
        try:
            layout = ensure_workdir(work_root)
            return Path(layout.root) / 'EvaluationRuns'
        except Exception:
            return work_root / 'EvaluationRuns'

    def _open_evaluation_workflow_results(self) -> None:
        path = str(getattr(self, '_last_evaluation_workflow_dir', '') or '').strip()
        if not path:
            rec = self._jobs_latest_record('profile_campaign') if hasattr(self, '_jobs_latest_record') else None
            path = str(getattr(rec, 'output_dir', '') or '').strip() if rec is not None else ''
        if not path:
            path = str(self._evaluation_run_root())
        self._open_path(path)

    def _open_validation_asset_root(self) -> None:
        try:
            from ..benchmark.validation_assets import validation_dataset_default_root
            self._open_path(str(validation_dataset_default_root()))
        except Exception as exc:
            messagebox.showerror('Validation assets', f'Could not open validation asset root:\n\n{exc}')

    def _open_jobs_tab(self) -> None:
        self._select_main_tab('jobs')

    def _select_main_tab(self, key: str) -> None:
        try:
            key = str(key)
            if key not in getattr(self, '_panel_built', set()) and hasattr(self, '_build_notebook_panel'):
                self._build_notebook_panel(key, initial=False)
            frame = getattr(self, 'panel_frames', {}).get(key)
            if frame is not None:
                self.main_notebook.select(frame)
        except Exception:
            logger.debug('Failed to select main tab %s', key, exc_info=True)

    def _queue_prepare_validation_assets(self) -> Optional[str]:
        run_id = time.strftime('%Y%m%d_%H%M%S')
        job_id = f'validation-assets-{run_id}'
        cancel_event = threading.Event()
        self._jobs_register(
            job_id=job_id,
            kind='validation_assets',
            type_label='Validation assets',
            title='Prepare validation assets',
            name='COCO-50/200 / Imagenette / test images',
            output_dir='',
            initial_status='Preparing validation assets…',
            initial_lines=[
                'Preparing validation assets used by the Evaluation Workflow.',
                'Includes COCO-50 validation, COCO-200 calibration, Imagenette mini 200/500 and runner test images unless already present.',
            ],
            progress_maximum=1.0,
            cancel_callback=lambda: cancel_event.set(),
            can_cancel=True,
            geometry='900x460',
        )

        def _log(line: str) -> None:
            self.root.after(0, lambda _s=str(line): self._jobs_append_log(job_id, _s))
            self.root.after(0, lambda _s=str(line): self._eval_workflow_text_append(_s + "\n"))

        def worker() -> None:
            try:
                from ..benchmark.validation_assets import prepare_all_validation_assets, validation_dataset_default_root
                out = prepare_all_validation_assets(
                    include_coco50=True,
                    include_coco200=True,
                    include_imagenette200=True,
                    include_imagenette500=True,
                    include_test_images=True,
                    overwrite=False,
                    log=lambda s: _log(s),
                )
                root = str(out.get('root') or validation_dataset_default_root())
                self.root.after(0, lambda: self._jobs_finish(job_id, status='success', message='Validation assets ready.', output_dir=root))
                self.root.after(0, lambda: messagebox.showinfo('Validation assets', f'Validation assets ready.\n\nRoot:\n{root}'))
            except Exception as exc:
                logger.exception('Validation asset preparation failed')
                status = 'cancelled' if cancel_event.is_set() else 'error'
                msg = f'{type(exc).__name__}: {exc}'
                self.root.after(0, lambda _msg=msg, _status=status: self._jobs_finish(job_id, status=_status, message=_msg))
                self.root.after(0, lambda _msg=msg: messagebox.showerror('Validation assets', _msg))

        threading.Thread(target=worker, daemon=True).start()
        return job_id

    def _init_central_notebook(self) -> None:
        logger.info("Initializing central notebook UI")
        root_children = list(self.winfo_children())

        self.main_notebook = ttk.Notebook(self)
        self.main_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        panel_builders = (
            ("analysis", panel_analysis.build_panel),
            ("export", panel_split_export.build_panel),
            ("evaluation_workflow", panel_evaluation_workflow.build_panel),
            ("validate", panel_validation.build_panel),
            ("bench_analysis", panel_benchmark_analysis.build_panel),
            ("jobs", panel_jobs.build_panel),
            ("hardware", panel_hardware.build_panel),
            ("logs", panel_logs.build_panel),
        )
        self._panel_builders = dict(panel_builders)
        self._panel_order = [key for key, _builder in panel_builders]
        self._panel_labels = {key: label for key, label in self.TAB_LABELS}
        self._panel_built = set()
        self.panel_frames: Dict[str, ttk.Frame] = {}

        lazy_tabs = str(os.environ.get("ONNX_SPLITPOINT_LAZY_TABS", "1") or "1").strip().lower() not in {"0", "false", "no", "off"}
        logger.info("Notebook lazy tab initialization %s", "enabled" if lazy_tabs else "disabled")

        def _make_placeholder(key: str) -> ttk.Frame:
            frame = ttk.Frame(self.main_notebook)
            ttk.Label(
                frame,
                text=f"{self._panel_labels.get(key, key)} wird beim ersten Öffnen geladen…",
                foreground="#555555",
            ).pack(padx=24, pady=24, anchor="nw")
            return frame

        # Build the Analyse tab immediately so the first visible page is useful.
        # Other heavy tabs are built on first selection; this avoids forcing Tk to
        # compute geometry for thousands of widgets before the first paint.
        initial_build = {"analysis"} if lazy_tabs else set(self._panel_order)
        for key in self._panel_order:
            if key in initial_build:
                self._build_notebook_panel(key, initial=True)
            else:
                self.panel_frames[key] = _make_placeholder(key)

        for key, label in self.TAB_LABELS:
            self.main_notebook.add(self.panel_frames[key], text=label)

        panel_analysis.hide_legacy_widgets(root_children, self)

        def _on_tab_changed(_event=None) -> None:
            try:
                selected = self.main_notebook.select()
                for key, frame in list(getattr(self, 'panel_frames', {}).items()):
                    if str(frame) == str(selected):
                        if key not in getattr(self, '_panel_built', set()):
                            self._build_notebook_panel(key, initial=False)
                        break
            except Exception:
                logger.debug("Lazy notebook tab build failed", exc_info=True)

        if lazy_tabs:
            self.main_notebook.bind("<<NotebookTabChanged>>", _on_tab_changed, add="+")

        self.jobs_status_bar = ttk.Frame(self)
        self.jobs_status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 8))
        ttk.Label(self.jobs_status_bar, text="Background jobs:").pack(side=tk.LEFT, padx=(0, 8))
        self.job_bar_generate = tk.Label(self.jobs_status_bar, text="Generate: idle", padx=8, pady=2, bg=self._jobs_status_color("idle"), fg="white")
        self.job_bar_generate.pack(side=tk.LEFT, padx=(0, 8))
        self.job_bar_remote = tk.Label(self.jobs_status_bar, text="Remote run: idle", padx=8, pady=2, bg=self._jobs_status_color("idle"), fg="white")
        self.job_bar_remote.pack(side=tk.LEFT, padx=(0, 8))
        self.job_bar_workflow = tk.Label(self.jobs_status_bar, text="Workflow: idle", padx=8, pady=2, bg=self._jobs_status_color("idle"), fg="white")
        self.job_bar_workflow.pack(side=tk.LEFT, padx=(0, 8))
        self.job_bar_validation_assets = tk.Label(self.jobs_status_bar, text="Validation assets: idle", padx=8, pady=2, bg=self._jobs_status_color("idle"), fg="white")
        self.job_bar_validation_assets.pack(side=tk.LEFT)

        try:
            self._set_ui_state(self._infer_ui_state())
        except Exception:
            logger.exception("Failed to refresh notebook action-button state")

        if not lazy_tabs:
            try:
                self._jobs_refresh_views()
            except Exception:
                logger.exception("Failed to refresh Jobs UI state")

        try:
            import os as _os, time as _time
            _shell_ts = _os.environ.get("ONNX_SPLITPOINT_STARTUP_SHELL_TS")
            if _shell_ts:
                logger.info("Startup elapsed from start_gui.sh to central notebook: %.1fs", max(0.0, _time.time() - float(_shell_ts)))
        except Exception:
            pass
        logger.info("Central notebook initialized")

    def _build_notebook_panel(self, key: str, *, initial: bool = False) -> Optional[ttk.Frame]:
        """Build one lazy panel without ever removing its visible tab on failure.

        A panel builder can fail after creating only part of its widget tree.  The
        former implementation removed the placeholder first, so an exception made
        the complete top-level tab disappear.  Keep the placeholder managed until
        the replacement exists; on failure, render a retryable error view in that
        same tab and remove only orphan widgets created by the failed builder.
        """
        if key in getattr(self, "_panel_built", set()):
            return self.panel_frames.get(key)
        builder = getattr(self, "_panel_builders", {}).get(key)
        if builder is None:
            return self.panel_frames.get(key)

        old_frame = self.panel_frames.get(key)
        label = self._panel_labels.get(key, key)
        notebook_tabs = tuple(self.main_notebook.tabs())
        old_managed = old_frame is not None and str(old_frame) in notebook_tabs
        try:
            tab_index = self.main_notebook.index(old_frame) if old_managed else self._panel_order.index(key)
        except Exception:
            tab_index = "end"

        before_children = set(self.main_notebook.winfo_children())
        t0 = time.perf_counter()
        logger.info("Initializing panel%s: %s", " (lazy)" if not initial else "", key)
        try:
            frame = builder(self.main_notebook, app=self)
            if frame is None:
                raise RuntimeError(f"Panel builder for {key!r} returned no frame")
        except Exception as exc:
            logger.exception("Failed to initialize notebook panel %s", key)

            # A failed builder may leave unmanaged child frames behind.  Destroy
            # those while deliberately preserving the still-managed placeholder.
            try:
                after_children = set(self.main_notebook.winfo_children())
                managed = set(self.main_notebook.tabs())
                for child in after_children - before_children:
                    if child is old_frame or str(child) in managed:
                        continue
                    try:
                        child.destroy()
                    except Exception:
                        pass
            except Exception:
                logger.debug("Failed to clean partial widgets for panel %s", key, exc_info=True)

            if old_frame is None:
                old_frame = ttk.Frame(self.main_notebook)
                self.panel_frames[key] = old_frame
            try:
                if str(old_frame) not in self.main_notebook.tabs():
                    self.main_notebook.insert(tab_index, old_frame, text=label)
            except Exception:
                try:
                    self.main_notebook.add(old_frame, text=label)
                except Exception:
                    pass

            try:
                for child in old_frame.winfo_children():
                    child.destroy()
                old_frame.columnconfigure(0, weight=1)
                old_frame.rowconfigure(0, weight=1)
                error_box = ttk.Frame(old_frame, padding=20)
                error_box.grid(row=0, column=0, sticky="nsew")
                ttk.Label(
                    error_box,
                    text=f"{label} konnte nicht geladen werden.",
                    font=("TkDefaultFont", 11, "bold"),
                    foreground="#a00000",
                ).grid(row=0, column=0, sticky="w")
                ttk.Label(
                    error_box,
                    text=f"{type(exc).__name__}: {exc}",
                    wraplength=900,
                    justify="left",
                ).grid(row=1, column=0, sticky="w", pady=(8, 12))
                ttk.Label(
                    error_box,
                    text="Der Tab bleibt verfügbar. Die Initialisierung kann nach einer Korrektur erneut versucht werden.",
                    wraplength=900,
                    justify="left",
                ).grid(row=2, column=0, sticky="w", pady=(0, 12))
                ttk.Button(
                    error_box,
                    text="Erneut laden",
                    command=lambda panel_key=key: self._build_notebook_panel(panel_key, initial=False),
                ).grid(row=3, column=0, sticky="w")
            except Exception:
                logger.debug("Failed to render panel error placeholder for %s", key, exc_info=True)

            errors = getattr(self, "_panel_build_errors", None)
            if not isinstance(errors, dict):
                errors = {}
                self._panel_build_errors = errors
            errors[key] = f"{type(exc).__name__}: {exc}"
            if not initial:
                try:
                    self.main_notebook.select(old_frame)
                except Exception:
                    pass
            return old_frame

        # Only now replace the placeholder; the panel is known to be complete.
        try:
            if old_managed:
                self.main_notebook.forget(old_frame)
            self.main_notebook.insert(tab_index, frame, text=label)
            if not initial:
                self.main_notebook.select(frame)
        except Exception:
            self.main_notebook.add(frame, text=label)
            if not initial:
                try:
                    self.main_notebook.select(frame)
                except Exception:
                    pass

        self.panel_frames[key] = frame
        self._panel_built.add(key)
        try:
            getattr(self, "_panel_build_errors", {}).pop(key, None)
        except Exception:
            pass
        try:
            if old_frame is not None and old_frame is not frame:
                old_frame.destroy()
        except Exception:
            pass

        logger.info(
            "Panel initialized%s: %s in %.3fs",
            " (lazy)" if not initial else "",
            key,
            time.perf_counter() - t0,
        )
        try:
            self._apply_persistent_settings(getattr(self, "_persisted_settings_cache", {}) or {})
        except Exception:
            logger.debug("Failed to re-apply settings after lazy panel build", exc_info=True)
        if key == "hardware":
            # Reload all read-only platform-power cards from the canonical
            # registry after generic settings have been restored.  The cards do
            # not keep a second editable setup/configuration snapshot.
            try:
                reload_platform_cards = getattr(
                    self, "_platform_power_reload_cards_callback", None
                )
                if callable(reload_platform_cards):
                    reload_platform_cards()
            except Exception:
                logger.debug(
                    "Failed to reload canonical platform-power cards after lazy build",
                    exc_info=True,
                )
        try:
            self._set_ui_state(self._infer_ui_state())
        except Exception:
            logger.debug("Failed to refresh UI state after lazy panel build", exc_info=True)
        if key == "jobs":
            try:
                self._jobs_refresh_views()
            except Exception:
                logger.debug("Failed to refresh jobs after lazy build", exc_info=True)
        return frame

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
            if not hasattr(self, "main_notebook"):
                continue
            frame = getattr(self, "panel_frames", {}).get(tab_key)
            try:
                # A failed lazy build keeps/recreates a placeholder.  Only ask
                # Tk to update a tab that is currently managed by this notebook.
                if frame is not None and str(frame) in self.main_notebook.tabs():
                    self.main_notebook.tab(frame, state="normal")
            except Exception:
                logger.debug("Failed to set notebook tab state for '%s'", tab_key, exc_info=True)


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

            # Use the Tool Config tab selection as the source of truth (fallback to
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
    try:
        from .widgets.messagebox_patch import install_messagebox_diagnostics
        install_messagebox_diagnostics()
    except Exception:
        pass
    log_path = _setup_gui_logging()
    try:
        print(f"[GUI] {os.path.abspath(__file__)} (v{__version__})")
        print(f"[CORE] {os.path.abspath(getattr(asc, '__file__', ''))} (v{getattr(asc, '__release__', getattr(asc, '__version__', '?'))})")
        if log_path:
            print(f"[LOG] {log_path}")
    except Exception:
        logger.exception("Failed to print GUI startup metadata")
    app = SplitPointAnalyserGUI()
    try:
        app._force_initial_window_paint()
    except Exception:
        logger.debug("Startup initial window paint failed", exc_info=True)
    app.mainloop()

# v60m: the top-level energy switch is authoritative for all native energy paths.
from onnx_splitpoint_tool.v60m_policy import install_energy_object_guards as _v60m_install_energy_guards
_v60m_install_energy_guards(globals())
