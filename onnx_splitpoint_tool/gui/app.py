"""GUI application entrypoint and incremental notebook shell."""

from __future__ import annotations

import os
import logging
import time
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from tkinter import messagebox
from tkinter import scrolledtext
from pathlib import Path
from typing import Dict

from .. import __version__ as TOOL_VERSION
from .. import api as asc
from ..settings import SettingsStore
from ..remote import HostConfig, SSHTransport
from ..benchmark.remote_run import RemoteBenchmarkArgs, run_remote_benchmark
from ..workdir import ensure_workdir
from ..gui_app import SplitPointAnalyserGUI as LegacySplitPointAnalyserGUI
from ..gui_app import _setup_gui_logging
from ..log_utils import sanitize_log
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
        ("validate", "Benchmark"),
        ("hardware", "Hardware"),
        ("logs", "Logs"),
    )

    def __init__(self):
        super().__init__()

        # The legacy GUI uses `self` as the Tk root. Some newer modules expect
        # a `.root` attribute as well, so keep an alias for compatibility.
        self.root = self

        # ------------------------------------------------------------------
        # Persistent settings
        # ------------------------------------------------------------------
        self._settings_store = SettingsStore()

        # Remote benchmarking vars (must exist before applying persisted tk_vars)
        self.remote_hosts = []
        self.var_remote_host_id = tk.StringVar(value="")
        self.var_remote_benchmark_set = tk.StringVar(value="")
        self.var_remote_provider = tk.StringVar(value="auto")
        self.var_remote_warmup = tk.IntVar(value=10)
        self.var_remote_iters = tk.IntVar(value=100)
        self.var_remote_repeats = tk.IntVar(value=1)
        self.var_remote_add_args = tk.StringVar(value="")
        self.var_remote_transfer_mode = tk.StringVar(value="bundle")
        self.var_remote_reuse_bundle = tk.BooleanVar(value=True)

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
    # Remote benchmark helpers (Benchmark tab)
    # ------------------------------------------------------------------

    def _remote_host_configs(self):
        out = []
        for h in getattr(self, "remote_hosts", []) or []:
            if not isinstance(h, dict):
                continue
            try:
                out.append(
                    HostConfig(
                        id=str(h.get("id") or h.get("label") or "host"),
                        label=str(h.get("label") or h.get("id") or "host"),
                        user=str(h.get("user") or ""),
                        host=str(h.get("host") or ""),
                        port=int(h.get("port") or 22),
                        remote_base_dir=str(h.get("remote_base_dir") or "~/splitpoint_runs"),
                        ssh_extra_args=str(h.get("ssh_extra_args") or ""),
                    )
                )
            except Exception:
                continue
        return out

    def _remote_get_selected_host(self) -> HostConfig | None:
        sel = ""
        try:
            sel = self.var_remote_host_id.get()
        except Exception:
            sel = ""
        if not sel:
            return None
        for h in self._remote_host_configs():
            if h.id == sel:
                return h
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

    def _remote_test_connection(self):
        host = self._remote_get_selected_host()
        if host is None:
            messagebox.showwarning("Remote benchmark", "Please select a remote host first.")
            return
        ok, msg = SSHTransport(host).test_connection(timeout_s=10)

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

    def _remote_run_benchmark(self):
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

        # Run id
        run_id = time.strftime("%Y%m%d_%H%M%S")

        # Progress window (reuse style of split/export)
        dlg = tk.Toplevel(self.root)
        dlg.title("Remote benchmark")
        dlg.geometry("760x460")
        dlg.transient(self.root)

        status_var = tk.StringVar(value="Starting…")
        ttk.Label(dlg, textvariable=status_var).pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))
        pb = ttk.Progressbar(dlg, orient="horizontal", length=400, mode="determinate")
        pb.pack(side=tk.TOP, fill=tk.X, padx=10, pady=6)

        txt = tk.Text(dlg, height=18, wrap="word")
        txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=6)
        txt.insert("end", f"Remote host: {host.user_host}\n")
        txt.insert("end", f"Suite: {bench_path.parent}\n")
        txt.see("end")

        btn_row = ttk.Frame(dlg)
        btn_row.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        cancel_event = threading.Event()
        alive = {"ok": True}

        def _on_close():
            # Best-effort cancel + close without crashing even if the worker still emits logs.
            alive["ok"] = False
            cancel_event.set()
            try:
                dlg.destroy()
            except Exception:
                pass

        # Handle window close (X) consistently
        try:
            dlg.protocol("WM_DELETE_WINDOW", _on_close)
        except Exception:
            pass

        def _append(line: str):
            if not alive.get("ok", False):
                return
            try:
                txt.insert("end", line + "\n")
                txt.see("end")
            except tk.TclError:
                alive["ok"] = False
            except Exception:
                # Never crash the UI on logging issues
                pass

        def _set_progress(p: float, label: str):
            if not alive.get("ok", False):
                return
            try:
                status_var.set(label)
            except Exception:
                pass
            try:
                pb["value"] = max(0.0, min(100.0, p * 100.0))
            except tk.TclError:
                alive["ok"] = False
            except Exception:
                pass

        def _on_cancel():
            cancel_event.set()
            _append("[ui] Cancel requested…")

        def _copy_log():
            """Copy current log window content to the clipboard."""
            try:
                # Ensure the copied log is readable even if the captured output
                # contained carriage returns (progress-style output) or ANSI
                # color escape sequences.
                text = sanitize_log(txt.get("1.0", "end-1c"))
                dlg.clipboard_clear()
                dlg.clipboard_append(text)
                dlg.update_idletasks()
                _append("[ui] Copied log to clipboard")
            except Exception as e:
                _append(f"[ui] Copy log failed: {e}")

        ttk.Button(btn_row, text="Cancel", command=_on_cancel).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Copy log", command=_copy_log).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(btn_row, text="Close", command=_on_close).pack(side=tk.RIGHT)

        def _worker():
            try:
                args = RemoteBenchmarkArgs(
                    provider=self.var_remote_provider.get() if hasattr(self, "var_remote_provider") else "auto",
                    warmup=int(self.var_remote_warmup.get()) if hasattr(self, "var_remote_warmup") else 10,
                    iters=int(self.var_remote_iters.get()) if hasattr(self, "var_remote_iters") else 100,
                    repeats=int(self.var_remote_repeats.get()) if hasattr(self, "var_remote_repeats") else 1,
                    add_args=self.var_remote_add_args.get() if hasattr(self, "var_remote_add_args") else "",
                    transfer_mode=self.var_remote_transfer_mode.get() if hasattr(self, "var_remote_transfer_mode") else "bundle",
                    reuse_bundle=bool(self.var_remote_reuse_bundle.get()) if hasattr(self, "var_remote_reuse_bundle") else True,
                )
                out = run_remote_benchmark(
                    host=host,
                    benchmark_set_json=bench_path,
                    repeats_idx="1",
                    local_working_dir=Path(getattr(self, "default_output_dir", ".")),
                    run_id=run_id,
                    args=args,
                    log=lambda s: self.root.after(0, _append, s),
                    progress=lambda p, lbl: self.root.after(0, _set_progress, p, lbl),
                    cancel_event=cancel_event,
                )
                if out.get("ok"):
                    self.root.after(0, _append, f"[ui] DONE: results saved to {out.get('local_run_dir')}")
                    self.root.after(0, _set_progress, 1.0, "Done")
                else:
                    self.root.after(0, _append, f"[ui] FAILED: {out.get('error')}")
                    self.root.after(0, _set_progress, 1.0, "Failed")
            except Exception as e:
                self.root.after(0, _append, f"[ui] ERROR: {e}")
                self.root.after(0, _set_progress, 1.0, "Error")

        threading.Thread(target=_worker, daemon=True).start()


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
