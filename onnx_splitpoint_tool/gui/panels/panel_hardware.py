"""Tool configuration panel for hardware, environments and validation assets."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from ... import api as asc
from ...hailo.backend_mode import backend_display_values
from ...benchmark.classification_validation_presets import list_available_presets
from ...energy.config import HardwareRegistryConflictError
from ..widgets.tooltip import attach_tooltip
from ..widgets.status_badge import StatusBadge
from ..dataset_dialogs import bind_registry_to_app, open_dataset_provisioning_dialog
from ...dataset_provisioning import default_dataset_root, default_registry_path, registry_status
from ...validation.official_coco import pycocotools_status
from ..run_mode_editor import build_run_modes_panel
from ..artifact_library_panel import ArtifactLibraryPanel


_HARDWARE_TOOLTIPS = {
    "left_accel": "Primäre Compute-Hardware der linken Modellhälfte.",
    "right_accel": "Compute-Hardware der rechten Modellhälfte.",
    "interface": "Verbindungsprofil zwischen beiden Seiten (Bandbreite/Overhead-Default).",
    "bw": "Link-Bandbreite für das Latenzmodell (z. B. MB/s, GB/s).",
    "bw_unit": "Einheit der eingetragenen Link-Bandbreite.",
    "gops_left": "Rechenleistung links in GOPS für die Latenzschätzung.",
    "gops_right": "Rechenleistung rechts in GOPS für die Latenzschätzung.",
    "overhead": "Fixer Link-Overhead pro Split in Millisekunden.",
    "link_model": "ideal: bytes/bandwidth + overhead; packetized: inkl. Paket-Overheads.",
    "link_energy": "Energie pro übertragenem Byte (pJ/B) für Energieabschätzung.",
    "mtu": "Nutzdaten pro Paket (MTU payload) bei packetized-Linkmodell.",
    "pkt_ovh_ms": "Zeit-Overhead pro Paket in Millisekunden.",
    "pkt_ovh_bytes": "Zusätzliche Protokollbytes pro Paket.",
    "link_max_ms": "Optionales Link-Latenzbudget; Kandidaten darüber werden markiert/gefiltert.",
    "link_max_mj": "Optionales Link-Energiebudget in mJ.",
    "link_max_bytes": "Optionales maximales Übertragungsbudget in Bytes.",
    "energy_left": "Energie pro FLOP der linken Seite (pJ/FLOP).",
    "energy_right": "Energie pro FLOP der rechten Seite (pJ/FLOP).",
    "peak_left": "Optionales Peak-Aktivierungslimit für die linke Seite.",
    "peak_left_unit": "Einheit des linken Peak-Aktivierungslimits.",
    "peak_right": "Optionales Peak-Aktivierungslimit für die rechte Seite.",
    "peak_right_unit": "Einheit des rechten Peak-Aktivierungslimits.",
    "hailo_enable": "Aktiviert Parse-Checks für Top-Kandidaten gegen den Hailo-Compiler.",
    "hailo_hw": "Ziel-Hailo-Architektur für den Parse-Check.",
    "hailo_max": "Maximale Anzahl Kandidaten, die zusätzlich mit Hailo geprüft werden. Danach läuft die Top-K-Auswahl ohne weitere Hailo-Checks weiter. Leer/auto = folgt Top-k.",
    "hailo_fixup": "Wendet ONNX-Fixups vor dem Hailo-Parse an.",
    "hailo_keep": "Behält temporäre Hailo-Artefakte für Debugging.",
    "hailo_target": "Welche Split-Seite mit Hailo geprüft wird.",
    "hailo_backend": "Backend-Auswahl: auto/subprocess/local/venv/wsl. subprocess erzwingt immer den Subprozess-Backend (Linux: verwaltetes DFC-Venv, Windows: WSL-Bridge).",
    "hailo_wsl_distro": "Optionaler WSL-Distro-Name für den Windows/WSL-Hailo-Backend. Unter Linux wird dieses Feld ignoriert. Leer lassen = Default-Distro. Tippfehler wie 'Ubuntu_22.04' werden nach Möglichkeit automatisch korrigiert.",
    "hailo_wsl_venv": "Optionaler Aktivierungsbefehl/Override für das Hailo-DFC-Venv. Tipp: 'auto' nutzt die verwalteten DFC-Profile (Hailo-8 vs Hailo-10) aus resources/hailo/profiles.json. Unter Linux ist das der bevorzugte Pfad, unter Windows wird es in WSL verwendet.",
    "hailo_status": "Zeigt, ob der Hailo DFC für Hailo-8/Hailo-10 erreichbar ist (automatisch beim Start + Refresh).",
    "hailo_refresh": "Aktualisiert die Hailo-Statusanzeige (Probe).",
    "hailo_clear": "Leert den lokalen Hailo-Parse-Cache.",
    "hailo_provision": (
        "Installiert/Repariert die verwalteten Hailo-DFC-Compiler-Umgebungen aus den jeweiligen Wheels. "
        "Ausgabe/Fehler landen zentral unter logs/provisioning/."
    ),
    "hailo8_provision": "Installiert/Repariert nur die verwaltete Hailo-8 DFC-venv aus resources/hailo/hailo8/*.whl.",
    "hailo10_provision": "Installiert/Repariert nur die verwaltete Hailo-10 DFC-venv aus resources/hailo/hailo10/*.whl.",
    "provisioning_logs": "Öffnet den zentralen Provisioning-Logordner logs/provisioning/.",
    "hailo_env_status": "Zeigt, ob Hailo-8/Hailo-10 Wheels vorhanden sind und ob die verwalteten DFC-venvs importbereit sind.",
    "hailo_wheel_folder": "Öffnet onnx_splitpoint_tool/resources/hailo/. Dort gehören die Hailo-DFC-Wheels in hailo8/ und hailo10/.",
    "build_env_status": "Zeigt den Status der zentralen Build-Umgebungen für Hailo DFC und DeepX DX-COM.",
    "build_env_config": "Öffnet/erstellt ~/.onnx_splitpoint_tool/build_environments.yaml.",
    "deepx_status": "Prüft die konfigurierte DeepX DX-COM/DX-RT Umgebung. v51d ist Status-/Config-MVP; DeepX-Benchmarking folgt separat.",
    "deepx_root": "dx-all-suite Root. 'auto' nutzt DEEPX_DX_ALL_SUITE_ROOT, DX_ALL_SUITE_ROOT oder ~/dx-all-suite. Install/Repair kann den Ordner per git checkout anlegen.",
    "deepx_open_root": "Öffnet den erkannten oder eingetragenen dx-all-suite Ordner.",
    "deepx_provision": "Checkt dx-all-suite bei Bedarf aus und erstellt/repariert DeepX Compiler-/Runtime-venvs. Treiber/Firmware werden nicht automatisch installiert.",
    "deepx_compiler_install": "Führt zusätzlich dx-compiler/install.sh aus, falls der DX-COM Compiler fehlt. Kann länger dauern und je nach Suite sudo/system dependencies benötigen.",
}


def _tt(key: str) -> str:
    return str(_HARDWARE_TOOLTIPS.get(key, ""))


def _ensure_var(app, name: str, var_type: type[tk.Variable], default):
    """Return app-bound Tk variable, creating and attaching it when missing."""
    if app is None:
        return var_type(value=default)
    existing = getattr(app, name, None)
    if existing is not None:
        return existing
    created = var_type(value=default)
    setattr(app, name, created)
    return created


def _str_var(app, name: str, default: str = "") -> tk.StringVar:
    return _ensure_var(app, name, tk.StringVar, default)


def _bool_var(app, name: str, default: bool = False) -> tk.BooleanVar:
    return _ensure_var(app, name, tk.BooleanVar, default)


def _deferred_platform_power_error_callback(
    exc: BaseException,
    receiver: Callable[[str], None],
) -> Callable[[], None]:
    """Materialize an exception before crossing the worker/Tk boundary.

    Python clears an ``except ... as exc`` binding when the block ends.  A Tk
    callback must therefore capture an ordinary string, not the exception
    binding itself.
    """

    message = f"Status refresh failed: {type(exc).__name__}: {exc}"

    def callback(stable_message: str = message) -> None:
        receiver(stable_message)

    return callback


def _schedule_initial_platform_power_refresh(
    app,
    refresh: Callable[..., None],
    *,
    delay_ms: int = 300,
) -> bool:
    """Schedule exactly one refresh for the first Tool Config construction."""

    if bool(getattr(app, "_platform_power_initial_refresh_started", False)):
        return False
    app._platform_power_initial_refresh_started = True
    app.root.after(int(delay_ms), lambda: refresh(quiet=True))
    return True


def _run_gui_m2_idle_calibration(
    setup_id: str,
    *,
    registry_path: str | Path | None,
    callback: Callable[[str], None] | None,
):
    """Run the direct full-system off/on calibration selected by the GUI."""
    from ...platform_power import calibrate_m2_accelerator_idle_power

    return calibrate_m2_accelerator_idle_power(
        setup_id,
        registry_path=registry_path,
        callback=callback,
    )


def _run_gui_full_system_input_calibration(
    setup_id: str,
    *,
    registry_path: str | Path | None,
    callback: Callable[[str], None] | None,
    operator_prompt: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
    confirm_jetson_already_off: bool = False,
):
    """Run the guided two-point full-system input calibration."""
    from ...platform_power import calibrate_full_system_input_scale

    return calibrate_full_system_input_scale(
        setup_id,
        registry_path=registry_path,
        callback=callback,
        operator_prompt=operator_prompt,
        confirm_jetson_already_off=confirm_jetson_already_off,
    )


def _energy_default_values() -> dict:
    try:
        from ...energy.config import DEFAULT_ENERGY_DEFAULTS
        return dict(DEFAULT_ENERGY_DEFAULTS)
    except Exception:
        return {
            "enabled": False,
            "collector_binary": "urecs-data-collector",
            "power_calculations_binary": "power_calculations",
            "mode": "fast_firmware",
            "data_port": 3000,
            "channel": 0,
            "sample_rate": 2000,
            "environment": "Jetson",
            "pre_duration_s": 5,
            "post_duration_s": 5,
            "duration_margin_s": 1.0,
            "power_estimated_duration_margin_s": 2.0,
            "run_count": 1,
            "physical_scope": "FS",
            "window_label": "command",
            "keep_raw_parquet": True,
            "postprocess_with_power_calculations": True,
            "include_raw_parquet_in_debug_pack": False,
        }


def _fs_energy_method_badge_state(
    setup_id: str,
    energy: Mapping[str, Any] | None,
    *,
    registry_path: str | Path | None = None,
    verifier: object = None,
) -> dict[str, str]:
    """Describe the fixed physical scope used by platform calibration."""

    return {
        "status": "full_system",
        "text": "Energy measurement: full system",
        "level": "ok",
        "detail": "u.RECS input power is measured for the complete system.",
        "path": "",
    }


def _platform_power_operation_error_text(exc: BaseException) -> str:
    """Format a GUI error without discarding final-gate evidence.

    Platform-power exceptions historically exposed only their message.  Newer
    backends may additionally attach a result/details mapping.  Accept both so
    the dialog remains useful across an in-place update and in isolated tests.
    """

    raw_message = str(exc).strip() or "No error details were provided."
    lines = [f"{type(exc).__name__}: {raw_message}"]
    payloads: list[Mapping[str, Any]] = []
    for attribute in ("result", "details", "payload", "summary", "evidence"):
        candidate = getattr(exc, attribute, None)
        if isinstance(candidate, Mapping):
            payloads.append(candidate)
    for argument in getattr(exc, "args", ()):
        if isinstance(argument, Mapping):
            payloads.append(argument)

    reasons: list[str] = []
    evidence: list[tuple[str, str]] = []
    visited: set[int] = set()
    for attribute in (
        "evidence_path",
        "evidence_dir",
        "calibration_evidence",
        "output_dir",
        "calibration_root",
        "summary_path",
        "aggregate_path",
    ):
        path_text = str(getattr(exc, attribute, None) or "").strip()
        if path_text:
            evidence.append((attribute, path_text))

    def add_reason(value: object) -> None:
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_reason(item)
            return
        text = str(value or "").strip().strip("[]'\"")
        if text and text not in reasons:
            reasons.append(text)

    def inspect(value: object, depth: int = 0) -> None:
        if depth > 4 or id(value) in visited:
            return
        if isinstance(value, Mapping):
            visited.add(id(value))
            for key, item in value.items():
                key_text = str(key)
                if key_text in {"final_energy_gate_reasons", "reasons"}:
                    add_reason(item)
                elif key_text == "final_energy_gate_failures":
                    inspect(item, depth + 1)
                elif key_text in {
                    "evidence_path",
                    "evidence_dir",
                    "calibration_evidence",
                    "output_dir",
                    "calibration_root",
                    "summary_path",
                    "aggregate_path",
                }:
                    path_text = str(item or "").strip()
                    pair = (key_text, path_text)
                    if path_text and pair not in evidence:
                        evidence.append(pair)
                elif isinstance(item, (Mapping, list, tuple)):
                    inspect(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            visited.add(id(value))
            for item in value:
                inspect(item, depth + 1)

    for payload in payloads:
        inspect(payload)

    for match in re.finditer(
        r"final_energy_gate_reasons\s*=\s*([^;\n]+)", raw_message
    ):
        for token in re.split(r"\s*,\s*", match.group(1)):
            add_reason(token)
    for match in re.finditer(
        r"\b(evidence_path|evidence_dir|calibration_evidence|output_dir|calibration_root|summary_path|aggregate_path)"
        r"\s*=\s*([^;\n]+)",
        raw_message,
    ):
        pair = (match.group(1), match.group(2).strip())
        if pair[1] and pair not in evidence:
            evidence.append(pair)

    if reasons:
        lines.append("Final energy gate reasons: " + ", ".join(reasons))
    if evidence:
        lines.append("Evidence:")
        lines.extend(f"  {label}: {path}" for label, path in evidence)
    return "\n".join(lines)



class _FullSystemInputCalibrationDialog:
    """Single modal Tk window that bridges operator steps to the backend."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        setup_id: str,
        registry_path: str | Path | None,
        on_progress: Callable[[str], None] | None = None,
        on_finished: Callable[[object | None, BaseException | None], None] | None = None,
        confirm_jetson_already_off: bool = False,
        app=None,
    ) -> None:
        self.setup_id = str(setup_id)
        self.registry_path = registry_path
        self.on_progress = on_progress
        self.on_finished = on_finished
        self.confirm_jetson_already_off = bool(confirm_jetson_already_off)
        self.app = app
        self._pending: tuple[dict[str, Any], threading.Event] | None = None
        self._pending_kind = ""
        self._cancel_requested = threading.Event()
        self._done = False
        self._result: object | None = None
        self._error: BaseException | None = None

        window = tk.Toplevel(parent)
        self.window = window
        window.title(f"Full-system calibration — {self.setup_id}")
        window.transient(parent.winfo_toplevel())
        window.resizable(True, True)
        window.minsize(720, 620)
        window.protocol("WM_DELETE_WINDOW", self._request_cancel)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(5, weight=1)

        ttk.Label(
            window,
            text=f"u.RECS full-system input calibration · {self.setup_id}",
            font=("", 13, "bold"),
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))

        warning = ttk.LabelFrame(window, text="Connection")
        warning.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 8))
        warning.columnconfigure(0, weight=1)
        ttk.Label(
            warning,
            text=(
                "Electronic load only as a sink from 9V_20V_IN on the load side "
                "of R16 to GND. Do not use the 5 V header for this routine. "
                "The actual current and voltage shown by the load are entered "
                "for the 0.5 A and 1.0 A points."
            ),
            wraplength=680,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=8)

        progress_frame = ttk.Frame(window)
        progress_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        progress_frame.columnconfigure(1, weight=1)
        self.step_var = tk.StringVar(value="Starting …")
        ttk.Label(progress_frame, textvariable=self.step_var, width=18).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.progress = ttk.Progressbar(
            progress_frame, mode="determinate", maximum=7, value=0
        )
        self.progress.grid(row=0, column=1, sticky="ew")

        prompt_frame = ttk.LabelFrame(window, text="Current step")
        prompt_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 8))
        prompt_frame.columnconfigure(0, weight=1)
        self.title_var = tk.StringVar(value="Preparing calibration backend …")
        self.message_var = tk.StringVar(
            value="The tool is checking the selected setup and acquiring the platform lock."
        )
        ttk.Label(
            prompt_frame, textvariable=self.title_var, font=("", 11, "bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Label(
            prompt_frame,
            textvariable=self.message_var,
            wraplength=680,
            justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        self.values_frame = ttk.Frame(prompt_frame)
        self.values_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.values_frame.columnconfigure(1, weight=1)
        self.target_current_var = tk.StringVar(value="0.5000")
        self.current_var = tk.StringVar(value="0.5000")
        self.voltage_var = tk.StringVar(value="19.0000")
        ttk.Label(self.values_frame, text="Sollpunkt [A]").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=2
        )
        ttk.Label(
            self.values_frame,
            textvariable=self.target_current_var,
        ).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(
            self.values_frame,
            text="Tatsächlich angezeigter Strom [A]",
        ).grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.current_entry = ttk.Entry(
            self.values_frame, textvariable=self.current_var, width=12
        )
        self.current_entry.grid(row=1, column=1, sticky="w", pady=2)
        ttk.Label(self.values_frame, text="Actual rail voltage [V]").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self.voltage_entry = ttk.Entry(
            self.values_frame, textvariable=self.voltage_var, width=12
        )
        self.voltage_entry.grid(row=2, column=1, sticky="w", pady=2)
        self.values_frame.grid_remove()

        self.result_text = tk.Text(
            prompt_frame,
            height=8,
            wrap="word",
            state="disabled",
        )
        self.result_text.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.result_text.grid_remove()

        button_frame = ttk.Frame(prompt_frame)
        button_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        self.primary_button = ttk.Button(
            button_frame, text="Continue", state=tk.DISABLED
        )
        self.primary_button.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.secondary_button = ttk.Button(
            button_frame, text="Finish without saving", state=tk.DISABLED
        )
        self.secondary_button.grid(row=0, column=1, sticky="ew", padx=3)
        self.secondary_button.grid_remove()
        self.cancel_button = ttk.Button(
            button_frame, text="Cancel", command=self._request_cancel
        )
        self.cancel_button.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        ttk.Label(window, text="Progress log", font=("", 9, "bold")).grid(
            row=4, column=0, sticky="w", padx=16, pady=(0, 3)
        )
        log_frame = ttk.Frame(window)
        log_frame.grid(row=5, column=0, sticky="nsew", padx=16, pady=(0, 14))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

        try:
            window.grab_set()
        except tk.TclError:
            pass
        window.after(50, self._start_worker)

    def _post(self, callback: Callable[[], None]) -> None:
        if bool(getattr(self.app, "_gui_closing", False)):
            self._cancel_requested.set()
            pending = self._pending
            if pending is not None:
                pending[0]["response"] = {"cancelled": True}
                pending[1].set()
            return
        try:
            self.window.after(0, callback)
        except tk.TclError:
            self._cancel_requested.set()
            pending = self._pending
            if pending is not None:
                pending[0]["response"] = {"cancelled": True}
                pending[1].set()

    def _append_log(self, text: str) -> None:
        try:
            self.log.configure(state="normal")
            stamp = time.strftime("%H:%M:%S")
            self.log.insert("end", f"[{stamp}] {text}\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        except tk.TclError:
            pass

    def _progress_callback(self, text: str) -> None:
        value = str(text)
        if self.on_progress is not None:
            try:
                self.on_progress(value)
            except Exception:
                pass
        self._post(lambda stable=value: self._append_log(stable))

    def _start_worker(self) -> None:
        self._append_log("Starting guided calibration.")

        def worker() -> None:
            try:
                result = _run_gui_full_system_input_calibration(
                    self.setup_id,
                    registry_path=self.registry_path,
                    callback=self._progress_callback,
                    operator_prompt=self._operator_prompt,
                    confirm_jetson_already_off=self.confirm_jetson_already_off,
                )
                error: BaseException | None = None
            except BaseException as exc:
                result = None
                error = exc
            self._post(lambda: self._finish(result, error))

        threading.Thread(
            target=worker,
            daemon=False,
            name=f"full-system-calibration-{self.setup_id}",
        ).start()

    def _operator_prompt(self, request: Mapping[str, Any]) -> Mapping[str, Any] | None:
        kind = str(request.get("kind") or "confirm")
        if self._cancel_requested.is_set() and kind != "recovery_zero_load":
            return {"cancelled": True}
        event = threading.Event()
        holder: dict[str, Any] = {}
        self._post(lambda: self._show_prompt(dict(request), holder, event))
        while not event.wait(0.25):
            if bool(getattr(self.app, "_gui_closing", False)):
                self._cancel_requested.set()
                return {"cancelled": True}
        response = holder.get("response")
        return dict(response) if isinstance(response, Mapping) else None

    def _show_prompt(
        self,
        request: dict[str, Any],
        holder: dict[str, Any],
        event: threading.Event,
    ) -> None:
        self._pending = (holder, event)
        kind = str(request.get("kind") or "confirm")
        self._pending_kind = kind
        step_index = int(request.get("step_index") or 0)
        step_count = int(request.get("step_count") or 7)
        self.progress.configure(maximum=max(1, step_count), value=step_index)
        self.step_var.set(f"Step {step_index}/{step_count}")
        self.title_var.set(str(request.get("title") or "Calibration step"))
        self.message_var.set(str(request.get("message") or ""))
        self.primary_button.configure(state=tk.NORMAL)
        self.cancel_button.configure(state=tk.NORMAL)
        self.secondary_button.grid_remove()
        self.result_text.grid_remove()
        self.values_frame.grid_remove()

        if kind == "load_step":
            self.values_frame.grid()
            self.target_current_var.set(
                f"{float(request.get('target_current_a') or 0.0):.4f}"
            )
            self.current_var.set(
                f"{float(request.get('default_current_a') or 0.0):.4f}"
            )
            self.voltage_var.set(
                f"{float(request.get('default_voltage_v') or 0.0):.4f}"
            )
            self.primary_button.configure(
                text=str(request.get("confirm_label") or "Measure"),
                command=lambda: self._respond_load(holder, event),
            )
            self.current_entry.focus_set()
        elif kind == "review":
            self.result_text.grid()
            factor = float(request.get("scale_factor") or 0.0)
            spread = float(request.get("point_spread_pct") or 0.0)
            residual = float(request.get("max_fit_residual_pct") or 0.0)
            idle_drift = float(request.get("idle_drift_w") or 0.0)
            quality_passed = request.get("quality_passed") is True
            quality_reasons = [
                str(value)
                for value in list(request.get("quality_reasons") or [])
                if str(value)
            ]
            lines = [
                f"Correction factor: {factor:.8f}",
                f"Correction relative to identity: {(factor - 1.0) * 100.0:+.4f}%",
                f"Point-factor spread: {spread:.4f}%",
                f"Maximum fit residual: {residual:.4f}%",
                f"Idle-window drift: {idle_drift:.6g} W",
                f"Technical validity: {'PASS' if quality_passed else 'FAIL — saving blocked'}",
                "",
            ]
            if quality_reasons:
                lines.extend(["Blocking technical reasons:"] + [f"• {v}" for v in quality_reasons] + [""])
            for point in list(request.get("point_results") or []):
                if not isinstance(point, Mapping):
                    continue
                lines.append(
                    f"Sollpunkt {float(point.get('target_current_a') or 0.0):.4f} A: "
                    f"Ist-Strom={float(point.get('reference_current_a') or 0.0):.4f} A, "
                    f"V={float(point.get('reference_voltage_v') or 0.0):.6g} V, "
                    f"measured increment={float(point.get('measured_increment_w') or 0.0):.6g} W, "
                    f"factor={float(point.get('point_scale_factor') or 0.0):.8f}"
                )
            warnings = [str(v) for v in list(request.get("warnings") or []) if str(v)]
            if warnings:
                lines.extend(["", "Plausibility warnings (saving remains allowed):"] + [f"• {v}" for v in warnings])
            else:
                lines.extend(["", "Plausibility check: OK"])
            self.result_text.configure(state="normal")
            self.result_text.delete("1.0", "end")
            self.result_text.insert("1.0", "\n".join(lines))
            self.result_text.configure(state="disabled")
            self.primary_button.configure(
                text=(
                    str(request.get("confirm_label") or "Save calibration")
                    if quality_passed
                    else "Save blocked: measurement technically invalid"
                ),
                state=(tk.NORMAL if quality_passed else tk.DISABLED),
                command=(
                    (lambda: self._respond(holder, event, {"save": True}))
                    if quality_passed
                    else None
                ),
            )
            self.secondary_button.configure(
                text=str(request.get("decline_label") or "Finish without saving"),
                state=tk.NORMAL,
                command=lambda: self._respond(holder, event, {"save": False}),
            )
            self.secondary_button.grid()
            self.cancel_button.configure(state=tk.DISABLED)
        else:
            self.primary_button.configure(
                text=str(request.get("confirm_label") or "Continue"),
                command=lambda: self._respond(holder, event, {"confirmed": True}),
            )
            if kind == "recovery_zero_load":
                self.cancel_button.configure(state=tk.DISABLED)
        self._append_log(self.title_var.get())

    def _respond_load(self, holder: dict[str, Any], event: threading.Event) -> None:
        try:
            current = float(self.current_var.get().replace(",", "."))
            voltage = float(self.voltage_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror(
                "Invalid reference values",
                "Current and voltage must be numeric values.",
                parent=self.window,
            )
            return
        if not (0.05 <= current <= 5.0 and 8.0 <= voltage <= 21.0):
            messagebox.showerror(
                "Invalid reference values",
                "Current must be 0.05–5 A and rail voltage 8–21 V.",
                parent=self.window,
            )
            return
        self._respond(
            holder,
            event,
            {"confirmed": True, "current_a": current, "voltage_v": voltage},
        )

    def _respond(
        self,
        holder: dict[str, Any],
        event: threading.Event,
        response: Mapping[str, Any],
    ) -> None:
        holder["response"] = dict(response)
        self._pending = None
        self._pending_kind = ""
        self.primary_button.configure(state=tk.DISABLED)
        self.secondary_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        event.set()

    def _request_cancel(self) -> None:
        if self._done:
            try:
                self.window.destroy()
            except tk.TclError:
                pass
            return
        if self._pending_kind == "recovery_zero_load":
            messagebox.showwarning(
                "Recovery confirmation required",
                "The platform may currently be in a modified power state. Set the "
                "electronic load to 0 A and confirm the recovery step before closing "
                "this window.",
                parent=self.window,
            )
            return
        if not messagebox.askyesno(
            "Cancel calibration",
            "Cancel the guided calibration? The tool will ask for 0 A and then "
            "attempt to restore the recorded initial Jetson state before it exits. "
            "The M.2 state is not changed by this calibration.",
            parent=self.window,
        ):
            return
        self._cancel_requested.set()
        self.title_var.set("Cancellation requested")
        self.message_var.set(
            "Do not disconnect the u.RECS. Set the electronic load to 0 A when "
            "the recovery step appears; restoration of the recorded initial Jetson "
            "state will then run."
        )
        pending = self._pending
        if pending is not None:
            pending[0]["response"] = {"cancelled": True}
            pending[1].set()
            self._pending = None
            self._pending_kind = ""
        self.primary_button.configure(state=tk.DISABLED)
        self.secondary_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.DISABLED)
        self._append_log("Cancellation requested by operator.")

    def _finish(self, result: object | None, error: BaseException | None) -> None:
        self._done = True
        self._result = result
        self._error = error
        try:
            self.window.grab_release()
        except tk.TclError:
            pass
        self.values_frame.grid_remove()
        self.secondary_button.grid_remove()
        self.result_text.grid()
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        if error is None:
            data = result.to_dict() if hasattr(result, "to_dict") else dict(result or {})
            factor = float(data.get("scale_factor") or 0.0)
            saved = bool(data.get("saved"))
            restoration_verified = data.get("initial_state_restored")
            restoration_text = (
                "yes"
                if restoration_verified is True
                else "no"
                if restoration_verified is False
                else "see evidence"
            )
            self.progress.configure(value=7)
            self.step_var.set("Completed")
            self.title_var.set("Calibration completed")
            quality_passed = bool(data.get("quality_passed"))
            invalidated = bool(data.get("invalidated_idle_baselines"))
            self.message_var.set(
                (
                    "The verified factor was saved and will be applied to future "
                    "full-system measurements. Re-run the setup's idle calibrations "
                    "before the next claim-bearing measurement."
                    if invalidated
                    else "The verified factor was saved and will be applied to future "
                    "full-system measurements."
                )
                if saved
                else (
                    "The measurements completed and were technically valid, but the "
                    "factor was not saved."
                    if quality_passed
                    else "The measurements completed, but technical validity failed; "
                    "saving was blocked."
                )
            )
            text = (
                f"Factor: {factor:.8f}\n"
                f"Technical validity: {'PASS' if quality_passed else 'FAIL'}\n"
                f"Saved: {'yes' if saved else 'no'}\n"
                f"Initial Jetson state restored: {restoration_text}\n"
                "M.2 state changed by calibration: no\n"
                f"Old idle baselines cleared: {'yes' if invalidated else 'no'}\n"
                f"Evidence: {data.get('evidence_path') or data.get('output_dir') or ''}\n"
                f"Evidence SHA-256: {data.get('evidence_sha256') or ''}"
            )
            completion_warnings = [
                str(value)
                for value in list(data.get("warnings") or [])
                if str(value)
            ]
            if completion_warnings:
                text += (
                    "\n\nPlausibility warnings (calibration was still saved):\n"
                    + "\n".join(f"• {value}" for value in completion_warnings)
                )
            self._append_log("Calibration completed successfully.")
        else:
            self.step_var.set("Stopped")
            self.title_var.set(
                "Calibration cancelled"
                if type(error).__name__ == "PlatformCalibrationCancelled"
                else "Calibration failed"
            )
            self.message_var.set(
                "The registry value was not changed. Check the evidence and the "
                "restoration status below."
            )
            text = _platform_power_operation_error_text(error)
            self._append_log(text)
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")
        self.primary_button.configure(
            text="Close", state=tk.NORMAL, command=self.window.destroy
        )
        self.cancel_button.grid_remove()
        if self.on_finished is not None:
            try:
                self.on_finished(result, error)
            except Exception:
                pass



_PLATFORM_POWER_SETUP_CARDS: tuple[tuple[str, str], ...] = (
    ("orin_nx_hailo8_01", "Orin NX + Hailo-8"),
    ("orin_nx_hailo10_01", "Orin NX + Hailo-10"),
    ("orin_nx_deepx_m1_01", "Orin NX + DeepX DX-M1"),
)


def _merge_accelerator_env_card_fields(
    setup: MutableMapping[str, Any],
    *,
    accelerator: str,
    host_address: str,
    host_user: str,
    host_port: int,
    remote_base_dir: str,
    remote_venv: str,
    provider: str,
    energy_enabled: bool,
    urecs_address: str,
    idle_baseline_w: float | None,
    accelerator_idle_w: float | None,
) -> MutableMapping[str, Any]:
    """Merge only fields exposed by one Accelerator env card.

    Hardware registries may carry SSH jump options, target identities and
    provider-specific settings that this compact editor does not expose.  A
    card save must preserve those fields byte-for-value instead of replacing
    the containing ``host``, ``runtime`` or legacy ``remote`` mappings.
    """

    setup["accelerator"] = str(accelerator)
    setup.setdefault("label", str(setup.get("id") or ""))

    host_saved = (
        dict(setup.get("host") or {})
        if isinstance(setup.get("host"), Mapping)
        else {}
    )
    host_saved.update(
        {
            "address": str(host_address),
            "user": str(host_user),
            "port": int(host_port),
            "base_dir": str(remote_base_dir),
        }
    )
    # A registry can retain both the current ``address`` key and a historical
    # ``host`` alias.  Once the visible address is edited, leaving that alias
    # stale would make the save itself create an ambiguous SSH target.
    if "host" in host_saved:
        host_saved["host"] = str(host_address)
    setup["host"] = host_saved

    runtime_saved = (
        dict(setup.get("runtime") or {})
        if isinstance(setup.get("runtime"), Mapping)
        else {}
    )
    runtime_saved.update(
        {
            "kind": "dxrt" if accelerator == "deepx_m1" else "hailort",
            "provider": str(provider),
            "activate": str(remote_venv),
            "venv": str(remote_venv),
        }
    )
    setup["runtime"] = runtime_saved

    raw_remote = setup.get("remote")
    if isinstance(raw_remote, Mapping):
        remote_saved = dict(raw_remote)
    elif isinstance(setup.get("remote_execution"), Mapping):
        # Import the legacy effective mapping before publishing the modern
        # alias; otherwise creating ``remote`` would shadow and silently drop
        # legacy ssh_extra_args/custom target selectors.
        remote_saved = dict(setup.get("remote_execution") or {})
    else:
        remote_saved = {}
    remote_saved.update(
        {
            "remote_base_dir": str(remote_base_dir),
            "remote_venv": str(remote_venv),
            "provider": str(provider),
        }
    )
    # Synchronize only endpoint aliases that already exist.  Unknown target
    # selectors and SSH options remain byte-for-value, while strict consumers
    # cannot observe an old legacy target after a visible host edit.
    for key, value in (
        ("host", str(host_address)),
        ("address", str(host_address)),
        ("user", str(host_user)),
        ("port", int(host_port)),
    ):
        if key in remote_saved:
            remote_saved[key] = value
    setup["remote"] = remote_saved

    for legacy_key in ("remote_execution",):
        legacy_raw = setup.get(legacy_key)
        if not isinstance(legacy_raw, Mapping):
            continue
        legacy_saved = dict(legacy_raw)
        for key, value in (
            ("host", str(host_address)),
            ("address", str(host_address)),
            ("user", str(host_user)),
            ("port", int(host_port)),
        ):
            if key in legacy_saved:
                legacy_saved[key] = value
        setup[legacy_key] = legacy_saved

    # Some oldest registries duplicated endpoint fields under runtime.  Keep
    # those aliases coherent without replacing provider-specific runtime data.
    for key, value in (
        ("host", str(host_address)),
        ("address", str(host_address)),
        ("user", str(host_user)),
        ("port", int(host_port)),
    ):
        if key in runtime_saved:
            runtime_saved[key] = value

    energy_saved = (
        dict(setup.get("energy") or {})
        if isinstance(setup.get("energy"), Mapping)
        else {}
    )
    energy_saved.update(
        {
            "enabled": bool(energy_enabled),
            "urecs_address": str(urecs_address),
            "idle_baseline_w": idle_baseline_w,
            "accelerator_idle_w": accelerator_idle_w,
        }
    )
    setup["energy"] = energy_saved
    return setup


def _build_platform_power_ui(detail_parent: ttk.Frame, app=None) -> None:
    """Build three registry-backed platform-power cards.

    The cards deliberately do not maintain a second editable copy of host,
    u.RECS or accelerator-idle configuration.  Those values come from the same
    hardware registry used by the three Accelerator envs cards.  Opening Tool
    Config schedules one read-only refresh for all cards; no polling loop is
    installed.
    """

    try:
        from ...platform_power import registry_setup_choices

        registry_labels = dict(registry_setup_choices())
    except Exception:
        registry_labels = {}

    detail_parent.rowconfigure(1, weight=1)
    for column in range(len(_PLATFORM_POWER_SETUP_CARDS)):
        detail_parent.columnconfigure(column, weight=1, uniform="platform-power-card")

    ttk.Label(
        detail_parent,
        text=(
            "Each setup uses the host, u.RECS address and calibration value from "
            "Accelerator envs / hardware_setups.yaml. Status is checked once when "
            "Tool Config is first opened and thereafter only on request."
        ),
        wraplength=1200,
        justify="left",
    ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 2))

    shared: dict[str, object] = {"mutating": False, "cards": []}

    def _gui_job_conflict() -> str:
        if app is None:
            return ""
        for record in list((getattr(app, "_background_jobs", {}) or {}).values()):
            status = str(getattr(record, "status", "") or "").strip().lower()
            if status in {"queued", "running", "cancelling", "cancel_requested"}:
                return f"{getattr(record, 'title', None) or getattr(record, 'kind', None) or getattr(record, 'job_id', 'job')} ({status})"
        return ""

    def _post_to_tk(callback: Callable[[], None]) -> None:
        if app is None:
            callback()
            return
        if bool(getattr(app, "_gui_closing", False)):
            return
        try:
            app.root.after(0, callback)
        except Exception:
            pass

    def _sync_button_states() -> None:
        global_busy = bool(shared["mutating"])
        for card_state in list(shared["cards"]):
            card_busy = bool(card_state.get("busy"))
            for button in list(card_state.get("buttons") or []):
                try:
                    button.configure(
                        state=(tk.DISABLED if global_busy or card_busy else tk.NORMAL)
                    )
                except Exception:
                    pass

    def _set_global_mutating(busy: bool) -> None:
        shared["mutating"] = bool(busy)
        _sync_button_states()

    def _build_setup_card(
        setup_id: str,
        fallback_label: str,
        column: int,
    ) -> dict[str, object]:
        label = str(registry_labels.get(setup_id) or fallback_label)
        card = ttk.LabelFrame(detail_parent, text=label)
        card.grid(
            row=1,
            column=column,
            sticky="nsew",
            padx=(8 if column == 0 else 4, 8 if column == 2 else 4),
            pady=(4, 8),
        )
        card.columnconfigure(0, weight=1)

        summary_var = _str_var(app, f"var_platform_power_{setup_id}_summary", "")
        detail_status_var = _str_var(
            app, f"var_platform_power_{setup_id}_status_detail", "Not checked yet."
        )
        operation_var = _str_var(
            app, f"var_platform_power_{setup_id}_operation", "Idle"
        )
        accelerator_idle_var = _str_var(
            app, f"var_hwsetup_{setup_id}_accel_idle_w", ""
        )
        full_system_scale_var = _str_var(
            app, f"var_platform_power_{setup_id}_fs_scale", ""
        )

        ttk.Label(
            card,
            textvariable=summary_var,
            wraplength=380,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))

        badges = ttk.Frame(card)
        badges.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        badge_urecs = StatusBadge(badges, text="u.RECS host: not checked", level="idle")
        badge_urecs.pack(side=tk.TOP, anchor="w", pady=(0, 3))
        badge_jetson = StatusBadge(badges, text="Jetson SSH: not checked", level="idle")
        badge_jetson.pack(side=tk.TOP, anchor="w", pady=(0, 3))
        badge_m2 = StatusBadge(badges, text="M.2 accelerator: not checked", level="idle")
        badge_m2.pack(side=tk.TOP, anchor="w", pady=(0, 3))
        badge_energy_method = StatusBadge(
            badges, text="Energy measurement: full system", level="ok"
        )
        badge_energy_method.pack(side=tk.TOP, anchor="w")

        ttk.Separator(card).grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 5))
        ttk.Label(card, text="Last status:", font=("", 9, "bold")).grid(
            row=3, column=0, sticky="w", padx=8
        )
        ttk.Label(
            card,
            textvariable=detail_status_var,
            wraplength=380,
            justify="left",
        ).grid(row=4, column=0, sticky="new", padx=8, pady=(2, 6))
        ttk.Label(card, text="Operation:", font=("", 9, "bold")).grid(
            row=5, column=0, sticky="w", padx=8
        )
        ttk.Label(
            card,
            textvariable=operation_var,
            wraplength=380,
            justify="left",
        ).grid(row=6, column=0, sticky="new", padx=8, pady=(2, 6))

        actions = ttk.Frame(card)
        actions.grid(row=7, column=0, sticky="ew", padx=8, pady=(2, 8))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        state: dict[str, object] = {
            "setup_id": setup_id,
            "last_status": None,
            "busy": False,
            "buttons": [],
            "power": {},
            "energy_method": {},
            "registry_path": None,
        }
        shared["cards"].append(state)

        def _load_config() -> None:
            try:
                from ...platform_power import host_config_from_setup, resolve_setup

                registry_path = None
                if app is not None:
                    path_resolver = getattr(app, "_hardware_setups_path", None)
                    if callable(path_resolver):
                        registry_path = path_resolver()
                _registry, setup, cfg = resolve_setup(
                    setup_id, registry_path=registry_path
                )
                state["registry_path"] = registry_path
                energy = dict(cfg.get("energy") or {})
                power = dict(cfg.get("power_control") or {})
                state["power"] = power
                method_state = _fs_energy_method_badge_state(
                    setup_id, energy, registry_path=registry_path
                )
                state["energy_method"] = method_state
                badge_energy_method.set(
                    text=method_state["text"], level=method_state["level"]
                )
                try:
                    host_text = host_config_from_setup(setup).user_host_pretty
                except Exception:
                    host_text = "not configured"
                accelerator_idle = energy.get("accelerator_idle_w")
                accelerator_idle_var.set(
                    "" if accelerator_idle is None else f"{float(accelerator_idle):.6g}"
                )
                full_system_scale = energy.get(
                    "full_system_current_scale_factor"
                )
                full_system_scale_var.set(
                    ""
                    if full_system_scale is None
                    else f"{float(full_system_scale):.8f}"
                )
                stabilize = float(
                    30.0
                    if power.get("calibration_stabilize_s") is None
                    else power.get("calibration_stabilize_s")
                )
                measure = float(
                    30.0
                    if power.get("calibration_measure_s") is None
                    else power.get("calibration_measure_s")
                )
                summary_var.set(
                    f"Setup: {setup_id}\n"
                    f"Jetson: {host_text}\n"
                    f"u.RECS: {str(energy.get('urecs_address') or 'not configured')}\n"
                    f"Accelerator idle: {accelerator_idle_var.get() or 'not calibrated'} W\n"
                    f"FS input scale: {full_system_scale_var.get() or 'identity / not calibrated'}\n"
                    f"Calibration: {stabilize:g}s stabilize + {measure:g}s measure; "
                    f"control {'enabled' if bool(power.get('enabled', True)) else 'disabled'}"
                )
            except Exception as exc:
                state["energy_method"] = {
                    "status": "invalid",
                    "detail": f"Configuration unavailable: {type(exc).__name__}: {exc}",
                }
                badge_energy_method.set(
                    text="Energy measurement: unavailable", level="error"
                )
                summary_var.set(
                    f"Setup: {setup_id}\nConfiguration unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )

        def _set_card_busy(busy: bool, text: str = "") -> None:
            state["busy"] = bool(busy)
            if text:
                operation_var.set(text)
            _sync_button_states()

        def _apply_status(status) -> None:
            state["last_status"] = status
            u = status.urecs_reachable
            j = status.jetson_ssh_ready
            m = status.m2_present
            jetson_button.configure(
                text=(
                    "Power off Jetson"
                    if j is True
                    else "Power on Jetson"
                    if j is False
                    else "Set Jetson state"
                )
            )
            badge_urecs.set(
                text=(
                    "u.RECS host: reachable"
                    if u is True
                    else "u.RECS host: unreachable"
                    if u is False
                    else "u.RECS host: unconfigured"
                ),
                level="ok" if u is True else "error" if u is False else "idle",
            )
            badge_jetson.set(
                text=(
                    "Jetson SSH: ready"
                    if j is True
                    else "Jetson SSH: not ready"
                    if j is False
                    else "Jetson SSH: unconfigured"
                ),
                level="ok" if j is True else "warn" if j is False else "idle",
            )
            badge_m2.set(
                text=(
                    "M.2 accelerator: detected"
                    if m is True
                    else "M.2 accelerator: not detected"
                    if m is False
                    else "M.2 accelerator: unknown"
                ),
                level="ok" if m is True else "warn" if m is False else "idle",
            )
            detail_status_var.set(
                f"Checked {status.checked_at}\n"
                f"u.RECS {status.urecs_address or '(not configured)'}: {status.urecs_detail}\n"
                f"Jetson {status.jetson_host or '(not configured)'}: {status.jetson_detail}\n"
                f"Accelerator {status.accelerator or '(unknown)'}: {status.m2_detail}\n"
                f"Energy measurement: "
                f"{dict(state.get('energy_method') or {}).get('detail') or 'not checked'}"
            )

        def _refresh_status(*_args, quiet: bool = False) -> None:
            if bool(shared["mutating"]) or bool(state["busy"]):
                return
            _load_config()
            _set_card_busy(True, "Refreshing saved setup status once …")

            def worker() -> None:
                try:
                    from ...platform_power import probe_platform_status

                    status = probe_platform_status(
                        setup_id,
                        registry_path=state.get("registry_path"),
                    )
                except Exception as exc:
                    def fail(error_text: str) -> None:
                        badge_urecs.set(text="u.RECS host: check error", level="error")
                        badge_jetson.set(text="Jetson SSH: error", level="error")
                        badge_m2.set(text="M.2 accelerator: unknown", level="idle")
                        operation_var.set(error_text)
                        _set_card_busy(False)
                        if not quiet:
                            messagebox.showwarning(
                                f"Platform status — {setup_id}", operation_var.get()
                            )

                    _post_to_tk(
                        _deferred_platform_power_error_callback(exc, fail)
                    )
                    return

                def done() -> None:
                    _apply_status(status)
                    operation_var.set("Idle")
                    _set_card_busy(False)

                _post_to_tk(done)

            threading.Thread(
                target=worker,
                daemon=True,
                name=f"platform-status-{setup_id}",
            ).start()

        def _progress(text: str) -> None:
            _post_to_tk(lambda value=str(text): operation_var.set(value))

        def _run_operation(label_text: str, call) -> None:
            if bool(shared["mutating"]) or bool(state["busy"]):
                return
            conflict = _gui_job_conflict()
            if conflict:
                messagebox.showwarning(
                    "Platform power blocked",
                    f"A Tool background job is active:\n{conflict}\n\nPower switching is blocked.",
                )
                return

            # Deliberately do not save GUI fields here.  The operation reloads
            # the canonical setup itself; this prevents a stale Power card from
            # overwriting u.RECS or calibration data edited elsewhere.
            _set_global_mutating(True)
            operation_var.set(label_text)

            def worker() -> None:
                try:
                    result = call()
                    error = None
                except Exception as exc:
                    result = None
                    error = exc

                def done() -> None:
                    if error is not None:
                        error_text = _platform_power_operation_error_text(error)
                        operation_var.set(error_text)
                        messagebox.showerror(
                            f"Platform power — {setup_id}", error_text
                        )
                    else:
                        operation_var.set(f"{label_text} completed")
                        try:
                            data = (
                                result.to_dict()
                                if hasattr(result, "to_dict")
                                else result
                            )
                            result_text = json.dumps(data, indent=2, sort_keys=True)
                        except Exception:
                            result_text = str(result)
                        _load_config()
                        if app is not None and hasattr(app, "_popup_text"):
                            app._popup_text(
                                f"Platform power result — {setup_id}",
                                result_text,
                                width=100,
                                height=28,
                            )
                        else:
                            messagebox.showinfo(
                                f"Platform power result — {setup_id}",
                                result_text[:10000],
                            )
                    _set_global_mutating(False)
                    _refresh_status(quiet=True)

                _post_to_tk(done)

            # State-changing sequences must finish their verification/recovery
            # even if the Tk window is closed.  Keep this thread non-daemon.
            threading.Thread(
                target=worker,
                daemon=False,
                name="platform-power-operation",
            ).start()

        def _toggle_jetson() -> None:
            status = state.get("last_status")
            if status is None or status.jetson_ssh_ready not in (True, False):
                messagebox.showinfo(
                    "Set Jetson state",
                    "A known SSH observation is required. Refresh this setup first.",
                )
                return
            observed_ssh_ready = bool(status.jetson_ssh_ready)
            desired_up = not observed_ssh_ready
            if desired_up is False:
                confirmed = messagebox.askyesno(
                    "Power off Jetson",
                    f"Setup: {setup_id}\n\nThe backend will re-check the exact SSH-ready "
                    "observation, shut down cleanly, send one unacknowledged "
                    "u.RECS toggle and require SSH not-ready. Continue?",
                )
            else:
                confirmed = messagebox.askyesno(
                    "Power on Jetson",
                    f"Setup: {setup_id}\n\nSSH not-ready may mean powered off, booting, "
                    "or a network fault. The backend will re-check that exact "
                    "observation, send one unacknowledged u.RECS toggle and "
                    "require authenticated SSH readiness. Continue?",
                )
            if not confirmed:
                return
            _run_operation(
                "Power on Jetson" if desired_up else "Power off Jetson",
                lambda: __import__(
                    "onnx_splitpoint_tool.platform_power",
                    fromlist=["set_jetson_state"],
                ).set_jetson_state(
                    setup_id,
                    desired_up,
                    registry_path=state.get("registry_path"),
                    expected_ssh_ready=observed_ssh_ready,
                    callback=_progress,
                ),
            )

        def _toggle_m2() -> None:
            status = state.get("last_status")
            if (
                status is None
                or status.jetson_ssh_ready is not True
                or status.m2_present is None
            ):
                messagebox.showinfo(
                    "M.2 toggle",
                    "A verified SSH-ready Jetson and known accelerator-presence "
                    "observation are required. Refresh this setup first.",
                )
                return
            desired_present = not bool(status.m2_present)
            target_text = "present" if desired_present else "absent"
            if not messagebox.askyesno(
                "Set M.2 accelerator state",
                f"Setup: {setup_id}\n\nThe Jetson will be shut down, both rail "
                f"toggles will be sequenced, and accelerator={target_text} will "
                "be verified after reboot. Continue?",
            ):
                return
            _run_operation(
                f"M.2 accelerator target: {target_text}",
                lambda: __import__(
                    "onnx_splitpoint_tool.platform_power",
                    fromlist=["set_m2_state"],
                ).set_m2_state(
                    setup_id,
                    desired_present,
                    registry_path=state.get("registry_path"),
                    callback=_progress,
                ),
            )

        def _calibrate() -> None:
            status = state.get("last_status")
            if (
                status is None
                or status.jetson_ssh_ready is not True
                or status.m2_present is not True
            ):
                messagebox.showinfo(
                    "M.2 idle calibration",
                    "Calibration requires an SSH-ready Jetson with this setup's "
                    "accelerator detected. Refresh this setup first.",
                )
                return
            power = dict(state.get("power") or {})
            stabilize = float(
                30.0
                if power.get("calibration_stabilize_s") is None
                else power.get("calibration_stabilize_s")
            )
            measure = float(
                30.0
                if power.get("calibration_measure_s") is None
                else power.get("calibration_measure_s")
            )
            if not messagebox.askyesno(
                "Run M.2 idle-power calibration",
                f"Setup: {setup_id}\n\nThis performs two controlled shutdown/boot "
                "cycles and full-system measurements with the accelerator absent and "
                f"present. Windows per state: {stabilize:g}s stabilize + "
                f"{measure:g}s measure.\n\nThe existing value changes only after "
                "both measurements and final restoration succeed.\n\nContinue?",
            ):
                return
            _run_operation(
                "M.2 idle calibration",
                lambda: _run_gui_m2_idle_calibration(
                    setup_id,
                    registry_path=state.get("registry_path"),
                    callback=_progress,
                ),
            )

        def _calibrate_full_system() -> None:
            if bool(shared["mutating"]) or bool(state["busy"]):
                return
            status = state.get("last_status")
            if status is None or status.urecs_reachable is not True:
                messagebox.showinfo(
                    "Full-system calibration",
                    "The wizard requires a current status with a reachable u.RECS. "
                    "Refresh this setup first.",
                )
                return
            confirm_jetson_already_off = False
            if status.jetson_ssh_ready is False:
                confirm_jetson_already_off = messagebox.askyesno(
                    "Confirm Jetson is intentionally off",
                    f"Setup: {setup_id}\n\nJetson SSH is not reachable. SSH being "
                    "unreachable alone does not prove that the Jetson power rail is "
                    "off.\n\nIs this Jetson deliberately powered off and stable now?\n\n"
                    "Choose No if it may still be booting, has a network problem, or "
                    "its power state is uncertain.",
                )
                if not confirm_jetson_already_off:
                    operation_var.set(
                        "Full-system calibration not started: Jetson-off state not confirmed"
                    )
                    return
            elif status.jetson_ssh_ready is not True:
                messagebox.showinfo(
                    "Full-system calibration",
                    "The Jetson power state is unknown. Refresh this setup. Start only "
                    "when SSH is ready, or when SSH is explicitly reported as not ready "
                    "and you can confirm that the Jetson is intentionally powered off.",
                )
                return
            conflict = _gui_job_conflict()
            if conflict:
                messagebox.showwarning(
                    "Platform power blocked",
                    f"A Tool background job is active:\n{conflict}\n\n"
                    "Full-system calibration is blocked.",
                )
                return
            _set_global_mutating(True)
            operation_var.set("Full-system calibration wizard running …")

            def finished(result: object | None, error: BaseException | None) -> None:
                if error is None:
                    operation_var.set(
                        "Full-system calibration completed"
                        if bool(getattr(result, "saved", False))
                        else "Full-system calibration completed without saving"
                    )
                else:
                    operation_var.set(_platform_power_operation_error_text(error))
                _load_config()
                _set_global_mutating(False)
                _refresh_status(quiet=True)

            try:
                parent = app.root if app is not None else card.winfo_toplevel()
                state["full_system_calibration_dialog"] = (
                    _FullSystemInputCalibrationDialog(
                        parent,
                        setup_id=setup_id,
                        registry_path=state.get("registry_path"),
                        on_progress=_progress,
                        on_finished=finished,
                        confirm_jetson_already_off=confirm_jetson_already_off,
                        app=app,
                    )
                )
            except BaseException as exc:
                _set_global_mutating(False)
                operation_var.set(_platform_power_operation_error_text(exc))
                messagebox.showerror(
                    f"Full-system calibration — {setup_id}",
                    _platform_power_operation_error_text(exc),
                )

        refresh_button = ttk.Button(actions, text="Refresh status", command=_refresh_status)
        refresh_button.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=(0, 4))
        jetson_button = ttk.Button(actions, text="Set Jetson state", command=_toggle_jetson)
        jetson_button.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=(0, 4))
        m2_button = ttk.Button(actions, text="Toggle M.2", command=_toggle_m2)
        m2_button.grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=(0, 4))
        calibrate_button = ttk.Button(
            actions,
            text="Calibrate M.2 idle power",
            command=_calibrate,
        )
        calibrate_button.grid(
            row=1, column=1, sticky="ew", padx=(2, 0), pady=(0, 4)
        )
        full_system_calibrate_button = ttk.Button(
            actions,
            text="Calibrate full-system input",
            command=_calibrate_full_system,
        )
        full_system_calibrate_button.grid(
            row=2, column=0, columnspan=2, sticky="ew"
        )
        state["buttons"] = [
            refresh_button,
            jetson_button,
            m2_button,
            calibrate_button,
            full_system_calibrate_button,
        ]

        attach_tooltip(
            refresh_button,
            "One bounded read-only refresh from the central setup registry; no polling is started.",
        )
        attach_tooltip(
            jetson_button,
            "Uses an explicit target bound to the last authenticated-SSH observation; no blind Jetson inversion is used.",
        )
        attach_tooltip(
            m2_button,
            "Gracefully shuts down the Jetson, sequences the M.2 toggle, reboots and verifies accelerator presence.",
        )
        attach_tooltip(
            calibrate_button,
            "Measures full-system idle power with M.2 absent and present, restores M.2 on, and atomically saves the difference.",
        )
        attach_tooltip(
            full_system_calibrate_button,
            "Guided idle-before, 0.5 A, middle-idle, 1.0 A and idle-after "
            "calibration on 9V_20V_IN after R16. An already-off Jetson can be "
            "explicitly confirmed; otherwise it is shut down once and its initial "
            "state is restored. M.2 is not switched. Saving is allowed when the "
            "measurement is technically valid; plausibility warnings remain visible "
            "but do not block saving.",
        )

        _load_config()
        return {
            "setup_id": setup_id,
            "reload": _load_config,
            "refresh": _refresh_status,
            "state": state,
        }

    controllers = [
        _build_setup_card(setup_id, label, column)
        for column, (setup_id, label) in enumerate(_PLATFORM_POWER_SETUP_CARDS)
    ]
    _sync_button_states()

    def _reload_all_cards() -> None:
        for controller in controllers:
            controller["reload"]()

    def _refresh_all(*_args, quiet: bool = True) -> None:
        for controller in controllers:
            controller["refresh"](quiet=quiet)

    if app is not None:
        try:
            app._platform_power_refresh_callback = lambda: _refresh_all(quiet=True)
            app._platform_power_reload_cards_callback = _reload_all_cards
        except Exception:
            pass
        # Tool Config is lazy; this runs once on its first real construction.
        _schedule_initial_platform_power_refresh(app, _refresh_all)

def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    # Keep Tool Config compact by grouping the previously long single page
    # into clear categories.  The variables remain app-bound, so existing
    # callers/settings continue to work.
    categories = ttk.Notebook(frame)
    categories.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

    tab_run_modes = ttk.Frame(categories)
    tab_model = ttk.Frame(categories)
    tab_envs = ttk.Frame(categories)
    tab_platform = ttk.Frame(categories)
    tab_energy = ttk.Frame(categories)
    tab_validation = ttk.Frame(categories)
    tab_final_data = ttk.Frame(categories)
    for _tab in (tab_run_modes, tab_model, tab_envs, tab_platform, tab_energy, tab_validation, tab_final_data):
        _tab.columnconfigure(0, weight=1)
    tab_run_modes.rowconfigure(0, weight=1)
    categories.add(tab_run_modes, text="Run modes")
    artifact_library_tab = ArtifactLibraryPanel(categories)
    categories.add(artifact_library_tab, text="Artifact Library")
    categories.add(tab_model, text="Split model")
    categories.add(tab_envs, text="Accelerator envs")
    categories.add(tab_platform, text="Platform power")
    categories.add(tab_energy, text="Energy measurement")
    categories.add(tab_validation, text="Screening datasets")
    categories.add(tab_final_data, text="Final campaign data")

    _build_platform_power_ui(tab_platform, app=app)

    run_modes_panel = build_run_modes_panel(tab_run_modes, app=app)
    try:
        if app is not None:
            app._tool_config_categories = categories
            app._tool_config_run_modes_tab = tab_run_modes
            app._artifact_library_tab = artifact_library_tab
            app._run_modes_panel = run_modes_panel
    except Exception:
        pass

    accel_names = []
    iface_names = []
    iface_by_name = {}
    if app is not None:
        specs = getattr(app, "accel_specs", {}) or {}
        accelerators = specs.get("accelerators") or []
        interfaces = specs.get("interfaces") or []
        accel_names = [str(x.get("name")) for x in accelerators]
        iface_names = [str(x.get("name")) for x in interfaces]
        iface_by_name = {str(x.get("name")): x for x in interfaces}
        accel_by_name = {str(x.get("name")): x for x in accelerators}

    accel = ttk.LabelFrame(tab_model, text="Accelerators")
    accel.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 8))

    left_var = _str_var(app, "var_memf_left_accel", accel_names[0] if accel_names else "")
    right_var = _str_var(app, "var_memf_right_accel", accel_names[1] if len(accel_names) > 1 else (accel_names[0] if accel_names else ""))
    iface_var = _str_var(app, "var_memf_interface", iface_names[0] if iface_names else "")

    ttk.Label(accel, text="Left Accelerator:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
    cb_left = ttk.Combobox(accel, textvariable=left_var, values=accel_names, state="readonly", width=30)
    cb_left.grid(row=0, column=1, sticky="w", pady=8)
    attach_tooltip(cb_left, _tt("left_accel"))

    ttk.Label(accel, text="Right Accelerator:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    cb_right = ttk.Combobox(accel, textvariable=right_var, values=accel_names, state="readonly", width=30)
    cb_right.grid(row=1, column=1, sticky="w", pady=(0, 8))
    attach_tooltip(cb_right, _tt("right_accel"))

    ttk.Label(accel, text="Link/Interface:").grid(row=2, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    cb_iface = ttk.Combobox(accel, textvariable=iface_var, values=iface_names, state="readonly", width=30)
    cb_iface.grid(row=2, column=1, sticky="w", pady=(0, 8))
    attach_tooltip(cb_iface, _tt("interface"))

    # Refresh the analysis-side Memory Fit widget when the user changes accelerators.
    # (The Memory Fit widget lives in the Analyse tab's Candidate Inspector.)
    _sync_latency_defaults = None  # assigned further down

    def _on_accel_change(*_args):
        # Update the legacy RAM Fit bars (Hardware tab), if present.
        if hasattr(app, "_refresh_memory_forecast"):
            try:
                app._refresh_memory_forecast()
            except Exception:
                pass
        # Update the Analyse-tab Memory Fit widget, if present.
        if hasattr(app, "_refresh_memory_fit_inspector"):
            try:
                app._refresh_memory_fit_inspector()
            except Exception:
                pass

        # Auto-fill latency model defaults from accelerator DB (GOPS + peak memory)
        if callable(_sync_latency_defaults):
            try:
                _sync_latency_defaults()
            except Exception:
                pass

    # Trace variable writes (covers programmatic changes)
    try:
        left_var.trace_add("write", _on_accel_change)   # Tk >= 8.5 / Py >= 3.6
        right_var.trace_add("write", _on_accel_change)
    except Exception:
        # Fallback for older Tk builds
        try:
            left_var.trace("w", _on_accel_change)
            right_var.trace("w", _on_accel_change)
        except Exception:
            pass

    # Also bind explicit combobox selection events (covers some Windows/theme edge cases)
    try:
        cb_left.bind("<<ComboboxSelected>>", lambda _e: _on_accel_change())
        cb_right.bind("<<ComboboxSelected>>", lambda _e: _on_accel_change())
    except Exception:
        pass

    latency = ttk.LabelFrame(tab_model, text="Latency model")
    latency.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

    bw_var = _str_var(app, "var_bw", "")
    bw_unit_var = _str_var(app, "var_bw_unit", "MB/s")
    gops_l_var = _str_var(app, "var_gops_l", "")
    gops_r_var = _str_var(app, "var_gops_r", "")
    overhead_var = _str_var(app, "var_overhead", "0")
    link_model_var = _str_var(app, "var_link_model", "ideal")
    link_energy_var = _str_var(app, "var_link_energy", "")
    mtu_var = _str_var(app, "var_link_mtu", "")
    pkt_ovh_ms_var = _str_var(app, "var_link_pkt_ovh_ms", "")
    pkt_ovh_bytes_var = _str_var(app, "var_link_pkt_ovh_bytes", "")
    link_max_ms_var = _str_var(app, "var_link_max_ms", "")
    link_max_mj_var = _str_var(app, "var_link_max_mJ", "")
    link_max_bytes_var = _str_var(app, "var_link_max_bytes", "")
    energy_l_var = _str_var(app, "var_energy_left", "")
    energy_r_var = _str_var(app, "var_energy_right", "")
    mem_l_var = _str_var(app, "var_mem_left", "")
    mem_l_unit_var = _str_var(app, "var_mem_left_unit", "MiB")
    mem_r_var = _str_var(app, "var_mem_right", "")
    mem_r_unit_var = _str_var(app, "var_mem_right_unit", "MiB")

    # ---------------------------------------------------------------------
    # Auto-fill (non-destructive) latency model defaults from accelerator DB
    # ---------------------------------------------------------------------
    def _autofill_cache() -> dict:
        d = getattr(app, "_autofill_defaults", None)
        if not isinstance(d, dict):
            d = {}
            setattr(app, "_autofill_defaults", d)
        return d

    def _autofill_set(var: tk.Variable, key: str, value: str) -> None:
        """Set var only if it is empty or still equals the last auto-filled value."""
        d = _autofill_cache()
        cur = str(var.get() or "").strip()
        prev = str(d.get(key, "") or "").strip()
        if cur == "" or cur == prev:
            var.set(value)
            d[key] = value

    def _to_float(x):
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _fmt_num(x: float) -> str:
        try:
            xf = float(x)
        except Exception:
            return ""
        # Keep the UI readable.
        if xf >= 100:
            return f"{xf:.0f}"
        if xf >= 10:
            return f"{xf:.1f}"
        return f"{xf:.2f}"

    def _default_gops_for(accel_spec):
        if not isinstance(accel_spec, dict):
            return None
        perf = accel_spec.get("perf") or {}
        prec = accel_spec.get("precision") or {}
        pref = str(prec.get("preferred_default") or "").strip().upper()
        eff = _to_float(perf.get("efficiency_factor"))
        if eff is None or eff <= 0:
            eff = 1.0

        raw = None
        if pref == "FP16":
            raw = _to_float(perf.get("gflops_fp16"))
            if not raw:
                raw = _to_float(perf.get("gflops_fp32"))
        elif pref == "FP32":
            raw = _to_float(perf.get("gflops_fp32"))
            if not raw:
                raw = _to_float(perf.get("gflops_fp16"))
        elif pref == "INT4":
            t = _to_float(perf.get("tops_int4"))
            if t:
                raw = t * 1000.0
        elif pref == "INT8":
            t = _to_float(perf.get("tops_int8"))
            if t:
                raw = t * 1000.0

        # Fallbacks if preferred_default is missing or the metric is absent.
        if not raw:
            raw = _to_float(perf.get("gflops_fp16")) or _to_float(perf.get("gflops_fp32"))
        if not raw:
            t = _to_float(perf.get("tops_int8"))
            if t:
                raw = t * 1000.0
        if not raw:
            t = _to_float(perf.get("tops_int4"))
            if t:
                raw = t * 1000.0

        if not raw:
            return None
        return raw * eff

    def _available_ram_mb(accel_spec):
        if not isinstance(accel_spec, dict):
            return None
        ram = _to_float(accel_spec.get("ram_limit_mb"))
        if ram is None:
            mem = accel_spec.get("memory") or {}
            ram_gb = _to_float(mem.get("ram_gb"))
            if ram_gb is not None:
                ram = ram_gb * 1024.0
        if ram is None:
            return None
        ov = _to_float(accel_spec.get("runtime_overhead_mb")) or 0.0
        return max(0.0, ram - ov)


    def _default_power_w(accel_spec):
        # Best-effort power estimate (W) for energy estimation.
        if not isinstance(accel_spec, dict):
            return None
        p = accel_spec.get("power") or {}
        if not isinstance(p, dict):
            p = {}

        # Prefer explicit typical/max if provided
        for k in ("typical_w", "max_w", "tdp_w", "tdp"):
            v = p.get(k)
            if v is None:
                continue
            try:
                return float(v)
            except Exception:
                pass

        # Otherwise, use the highest listed power mode
        modes = p.get("modes_w")
        if isinstance(modes, list) and modes:
            vals = []
            for v in modes:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            if vals:
                return max(vals)

        return None

    def _default_energy_pj_per_flop(accel_spec, gops):
        # Compute energy estimate in pJ/F from (power W, throughput GOPS).
        # pJ/F = (W * 1000) / GOPS
        p_w = _default_power_w(accel_spec)
        try:
            g = float(gops) if gops is not None else None
        except Exception:
            g = None
        if p_w is None or g is None or g <= 0:
            return None
        return (p_w * 1000.0) / g

    def _sync_latency_defaults():
        left_name = str(left_var.get() or "")
        right_name = str(right_var.get() or "")
        left_spec = accel_by_name.get(left_name)
        right_spec = accel_by_name.get(right_name)

        g_l = _default_gops_for(left_spec)
        g_r = _default_gops_for(right_spec)
        if g_l is not None:
            _autofill_set(gops_l_var, "lat_gops_left", _fmt_num(g_l))
        if g_r is not None:
            _autofill_set(gops_r_var, "lat_gops_right", _fmt_num(g_r))

        m_l = _available_ram_mb(left_spec)
        m_r = _available_ram_mb(right_spec)
        if m_l is not None:
            _autofill_set(mem_l_unit_var, "lat_peak_left_unit", "MiB")
            _autofill_set(mem_l_var, "lat_peak_left", f"{int(round(m_l))}")
        if m_r is not None:
            _autofill_set(mem_r_unit_var, "lat_peak_right_unit", "MiB")
            _autofill_set(mem_r_var, "lat_peak_right", f"{int(round(m_r))}")
        # Derived compute energy (pJ/F) from power + throughput
        e_l = _default_energy_pj_per_flop(left_spec, g_l)
        e_r = _default_energy_pj_per_flop(right_spec, g_r)
        if e_l is not None:
            _autofill_set(energy_l_var, "lat_e_left_pjpf", _fmt_num(e_l))
        if e_r is not None:
            _autofill_set(energy_r_var, "lat_e_right_pjpf", _fmt_num(e_r))


    ttk.Label(latency, text="Link bandwidth:").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=8)
    ent_bw = ttk.Entry(latency, textvariable=bw_var, width=10)
    ent_bw.grid(row=0, column=1, sticky="w", pady=8)
    attach_tooltip(ent_bw, _tt("bw"))

    cb_bw_unit = ttk.Combobox(latency, textvariable=bw_unit_var, values=sorted(asc.BANDWIDTH_MULT.keys()), width=8, state="readonly")
    cb_bw_unit.grid(row=0, column=2, sticky="w", padx=(6, 12), pady=8)
    attach_tooltip(cb_bw_unit, _tt("bw_unit"))

    ttk.Label(latency, text="GOPS left:").grid(row=0, column=3, sticky="w", padx=(0, 4), pady=8)
    ent_gops_l = ttk.Entry(latency, textvariable=gops_l_var, width=10)
    ent_gops_l.grid(row=0, column=4, sticky="w", pady=8)
    attach_tooltip(ent_gops_l, _tt("gops_left"))

    ttk.Label(latency, text="GOPS right:").grid(row=0, column=5, sticky="w", padx=(12, 4), pady=8)
    ent_gops_r = ttk.Entry(latency, textvariable=gops_r_var, width=10)
    ent_gops_r.grid(row=0, column=6, sticky="w", pady=8)
    attach_tooltip(ent_gops_r, _tt("gops_right"))

    ttk.Label(latency, text="Overhead (ms):").grid(row=0, column=7, sticky="w", padx=(12, 4), pady=8)
    ent_overhead = ttk.Entry(latency, textvariable=overhead_var, width=8)
    ent_overhead.grid(row=0, column=8, sticky="w", pady=8)
    attach_tooltip(ent_overhead, _tt("overhead"))

    ttk.Label(latency, text="Link model:").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
    cb_link_model = ttk.Combobox(latency, textvariable=link_model_var, values=["ideal", "packetized"], width=10, state="readonly")
    cb_link_model.grid(row=1, column=1, sticky="w", pady=(0, 8))
    attach_tooltip(cb_link_model, _tt("link_model"))

    ttk.Label(latency, text="E_link (pJ/B):").grid(row=1, column=2, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_link_energy = ttk.Entry(latency, textvariable=link_energy_var, width=10)
    ent_link_energy.grid(row=1, column=3, sticky="w", pady=(0, 8))
    attach_tooltip(ent_link_energy, _tt("link_energy"))

    ttk.Label(latency, text="MTU payload (B):").grid(row=1, column=4, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_mtu = ttk.Entry(latency, textvariable=mtu_var, width=10)
    ent_mtu.grid(row=1, column=5, sticky="w", pady=(0, 8))
    attach_tooltip(ent_mtu, _tt("mtu"))

    ttk.Label(latency, text="pkt ovh (ms):").grid(row=1, column=6, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_pkt_ms = ttk.Entry(latency, textvariable=pkt_ovh_ms_var, width=8)
    ent_pkt_ms.grid(row=1, column=7, sticky="w", pady=(0, 8))
    attach_tooltip(ent_pkt_ms, _tt("pkt_ovh_ms"))

    ttk.Label(latency, text="pkt ovh (B):").grid(row=1, column=8, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_pkt_b = ttk.Entry(latency, textvariable=pkt_ovh_bytes_var, width=8)
    ent_pkt_b.grid(row=1, column=9, sticky="w", pady=(0, 8))
    attach_tooltip(ent_pkt_b, _tt("pkt_ovh_bytes"))

    ttk.Label(latency, text="Link max ms:").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
    ent_link_max_ms = ttk.Entry(latency, textvariable=link_max_ms_var, width=10)
    ent_link_max_ms.grid(row=2, column=1, sticky="w", pady=(0, 8))
    attach_tooltip(ent_link_max_ms, _tt("link_max_ms"))

    ttk.Label(latency, text="Link max mJ:").grid(row=2, column=2, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_link_max_mj = ttk.Entry(latency, textvariable=link_max_mj_var, width=10)
    ent_link_max_mj.grid(row=2, column=3, sticky="w", pady=(0, 8))
    attach_tooltip(ent_link_max_mj, _tt("link_max_mj"))

    ttk.Label(latency, text="Link max bytes:").grid(row=2, column=4, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_link_max_bytes = ttk.Entry(latency, textvariable=link_max_bytes_var, width=10)
    ent_link_max_bytes.grid(row=2, column=5, sticky="w", pady=(0, 8))
    attach_tooltip(ent_link_max_bytes, _tt("link_max_bytes"))

    ttk.Label(latency, text="E_left (pJ/F):").grid(row=2, column=6, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_energy_left = ttk.Entry(latency, textvariable=energy_l_var, width=10)
    ent_energy_left.grid(row=2, column=7, sticky="w", pady=(0, 8))
    attach_tooltip(ent_energy_left, _tt("energy_left"))

    ttk.Label(latency, text="E_right (pJ/F):").grid(row=2, column=8, sticky="w", padx=(12, 4), pady=(0, 8))
    ent_energy_right = ttk.Entry(latency, textvariable=energy_r_var, width=10)
    ent_energy_right.grid(row=2, column=9, sticky="w", pady=(0, 8))
    attach_tooltip(ent_energy_right, _tt("energy_right"))

    ttk.Label(latency, text="Peak left ≤").grid(row=3, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
    ent_peak_left = ttk.Entry(latency, textvariable=mem_l_var, width=10)
    ent_peak_left.grid(row=3, column=1, sticky="w", pady=(0, 8))
    attach_tooltip(ent_peak_left, _tt("peak_left"))

    cb_peak_left_unit = ttk.Combobox(latency, textvariable=mem_l_unit_var, values=sorted(asc.UNIT_MULT.keys()), width=7, state="readonly")
    cb_peak_left_unit.grid(row=3, column=2, sticky="w", padx=(6, 12), pady=(0, 8))
    attach_tooltip(cb_peak_left_unit, _tt("peak_left_unit"))

    ttk.Label(latency, text="Peak right ≤").grid(row=3, column=3, sticky="w", padx=(0, 4), pady=(0, 8))
    ent_peak_right = ttk.Entry(latency, textvariable=mem_r_var, width=10)
    ent_peak_right.grid(row=3, column=4, sticky="w", pady=(0, 8))
    attach_tooltip(ent_peak_right, _tt("peak_right"))

    cb_peak_right_unit = ttk.Combobox(latency, textvariable=mem_r_unit_var, values=sorted(asc.UNIT_MULT.keys()), width=7, state="readonly")
    cb_peak_right_unit.grid(row=3, column=5, sticky="w", padx=(6, 12), pady=(0, 8))
    attach_tooltip(cb_peak_right_unit, _tt("peak_right_unit"))

    hailo = ttk.LabelFrame(tab_envs, text="Accelerator diagnostics / build environments")
    hailo.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 8))
    try:
        hailo.columnconfigure(0, weight=1)
    except Exception:
        pass

    hailo_check_var = _bool_var(app, "var_hailo_check", False)
    hailo_hw_var = _str_var(app, "var_hailo_hw_arch", "hailo8")
    hailo_max_var = _str_var(app, "var_hailo_max_checks", "auto")
    hailo_fixup_var = _bool_var(app, "var_hailo_fixup", True)
    hailo_keep_var = _bool_var(app, "var_hailo_keep", False)
    hailo_target_var = _str_var(app, "var_hailo_target", "part2")
    hailo_backend_var = _str_var(app, "var_hailo_backend", "auto")
    hailo_wsl_distro_var = _str_var(app, "var_hailo_wsl_distro", "")
    hailo_wsl_venv_var = _str_var(app, "var_hailo_wsl_venv", "auto")

    # Backwards-compat: normalize deprecated "hailo10" -> "hailo10h".
    try:
        if (hailo_hw_var.get() or "").strip() == "hailo10":
            hailo_hw_var.set("hailo10h")
    except Exception:
        pass

    parse = ttk.LabelFrame(hailo, text="Hailo parse-check for Analyze & Split ranking")
    parse.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
    chk_hailo = ttk.Checkbutton(parse, text="Enable parse-check in ranking", variable=hailo_check_var)
    chk_hailo.grid(row=0, column=0, sticky="w", padx=(8, 12), pady=8)
    attach_tooltip(chk_hailo, _tt("hailo_enable"))

    ttk.Label(parse, text="HW arch:").grid(row=0, column=1, sticky="e", pady=8)
    cb_hailo_hw = ttk.Combobox(parse, textvariable=hailo_hw_var, values=["hailo8", "hailo8l", "hailo8r", "hailo10h", "hailo10p"], width=10, state="readonly")
    cb_hailo_hw.grid(row=0, column=2, sticky="w", padx=(4, 12), pady=8)
    attach_tooltip(cb_hailo_hw, _tt("hailo_hw"))

    ttk.Label(parse, text="Target:").grid(row=0, column=3, sticky="e", pady=8)
    cb_hailo_target = ttk.Combobox(parse, textvariable=hailo_target_var, values=["either", "part2", "part1"], width=10, state="readonly")
    cb_hailo_target.grid(row=0, column=4, sticky="w", padx=(4, 12), pady=8)
    attach_tooltip(cb_hailo_target, _tt("hailo_target"))

    ttk.Label(parse, text="Max checks:").grid(row=0, column=5, sticky="e", pady=8)
    ent_hailo_max = ttk.Entry(parse, textvariable=hailo_max_var, width=8)
    ent_hailo_max.grid(row=0, column=6, sticky="w", padx=(4, 12), pady=8)
    attach_tooltip(ent_hailo_max, _tt("hailo_max"))

    chk_fixup = ttk.Checkbutton(parse, text="ONNX fixup", variable=hailo_fixup_var)
    chk_fixup.grid(row=1, column=0, sticky="w", padx=(8, 12), pady=(0, 8))
    attach_tooltip(chk_fixup, _tt("hailo_fixup"))

    chk_keep = ttk.Checkbutton(parse, text="Keep artifacts", variable=hailo_keep_var)
    chk_keep.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=(0, 8))
    attach_tooltip(chk_keep, _tt("hailo_keep"))

    ttk.Label(parse, text="Backend:").grid(row=1, column=2, sticky="e", pady=(0, 8))
    cb_hailo_backend = ttk.Combobox(parse, textvariable=hailo_backend_var, values=list(backend_display_values()), width=12, state="readonly")
    cb_hailo_backend.grid(row=1, column=3, sticky="w", padx=(4, 12), pady=(0, 8))
    attach_tooltip(cb_hailo_backend, _tt("hailo_backend"))

    ttk.Label(parse, text="WSL distro:").grid(row=1, column=4, sticky="e", pady=(0, 8))
    distro_values = [""]
    try:
        from ...hailo_backend import wsl_list_distros
        distro_values += wsl_list_distros()
    except Exception:
        pass
    cb_hailo_distro = ttk.Combobox(parse, textvariable=hailo_wsl_distro_var, values=distro_values, width=18, state="normal")
    cb_hailo_distro.grid(row=1, column=5, sticky="w", padx=(4, 12), pady=(0, 8))
    attach_tooltip(cb_hailo_distro, _tt("hailo_wsl_distro"))

    ttk.Label(parse, text="Venv override:").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
    ent_hailo_venv = ttk.Entry(parse, textvariable=hailo_wsl_venv_var, width=62)
    ent_hailo_venv.grid(row=2, column=1, columnspan=5, sticky="ew", pady=(0, 8))
    attach_tooltip(ent_hailo_venv, _tt("hailo_wsl_venv"))
    if app is not None and hasattr(app, "_hailo_refresh_status"):
        btn_ref = ttk.Button(parse, text="Refresh parse status", command=app._hailo_refresh_status)
        btn_ref.grid(row=2, column=6, sticky="e", padx=(0, 8), pady=(0, 8))
        attach_tooltip(btn_ref, _tt("hailo_refresh"))

    build_defaults = ttk.LabelFrame(hailo, text="Hailo build defaults for Benchmark/Evaluation")
    build_defaults.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
    build_defaults.columnconfigure(9, weight=1)
    var_hef_preset = _str_var(app, "var_hailo_hef_preset", "Standard")
    var_hef_opt = _str_var(app, "var_hailo_hef_opt_level", "1")
    var_hef_calib_count = _str_var(app, "var_hailo_hef_calib_count", "64")
    var_hef_calib_bs = _str_var(app, "var_hailo_hef_calib_batch_size", "8")
    var_hef_calib_dir = _str_var(app, "var_hailo_hef_calib_dir", "")
    var_hef_force = _bool_var(app, "var_hailo_hef_force", False)
    var_hef_keep = _bool_var(app, "var_hailo_hef_keep_artifacts", False)
    ttk.Label(build_defaults, text="Preset:").grid(row=0, column=0, sticky="w", padx=(8, 4), pady=6)
    cb_build_preset = ttk.Combobox(build_defaults, textvariable=var_hef_preset, values=["Quick", "Standard", "Accurate"], width=12, state="readonly")
    cb_build_preset.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=6)
    attach_tooltip(cb_build_preset, "Hailo build preset used by generated benchmark sets and evaluation workflows.")
    ttk.Label(build_defaults, text="Opt:").grid(row=0, column=2, sticky="e", padx=(0, 4), pady=6)
    ttk.Entry(build_defaults, textvariable=var_hef_opt, width=5).grid(row=0, column=3, sticky="w", padx=(0, 10), pady=6)
    ttk.Label(build_defaults, text="Calib count/batch:").grid(row=0, column=4, sticky="e", padx=(0, 4), pady=6)
    ttk.Entry(build_defaults, textvariable=var_hef_calib_count, width=7).grid(row=0, column=5, sticky="w", padx=(0, 4), pady=6)
    ttk.Label(build_defaults, text="/").grid(row=0, column=6, sticky="w", pady=6)
    ttk.Entry(build_defaults, textvariable=var_hef_calib_bs, width=7).grid(row=0, column=7, sticky="w", padx=(4, 10), pady=6)
    ttk.Checkbutton(build_defaults, text="Force rebuild", variable=var_hef_force).grid(row=0, column=8, sticky="w", padx=(0, 8), pady=6)
    ttk.Checkbutton(build_defaults, text="Keep HARs", variable=var_hef_keep).grid(row=0, column=9, sticky="w", padx=(0, 8), pady=6)
    ttk.Label(build_defaults, text="Calib dir:").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
    ent_hailo_calib = ttk.Entry(build_defaults, textvariable=var_hef_calib_dir, width=72)
    ent_hailo_calib.grid(row=1, column=1, columnspan=8, sticky="ew", padx=(0, 8), pady=(0, 8))
    attach_tooltip(ent_hailo_calib, "Optional explicit calibration image directory for Hailo HEF builds. Leave empty to use the Tool-wide Calibration / validation split below (Imagenette-500 for classification, COCO-200 for detection by default).")
    def _browse_hailo_calib() -> None:
        start = str(var_hef_calib_dir.get() or "").strip() or None
        p = filedialog.askdirectory(title="Select Hailo calibration directory", initialdir=start)
        if p:
            var_hef_calib_dir.set(p)
    ttk.Button(build_defaults, text="Browse…", command=_browse_hailo_calib).grid(row=1, column=9, sticky="w", padx=(0, 8), pady=(0, 8))

    run_defaults = ttk.LabelFrame(hailo, text="Hailo benchmark/run defaults")
    run_defaults.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))
    for _c in range(10):
        try:
            run_defaults.columnconfigure(_c, weight=1 if _c in (1, 5) else 0)
        except Exception:
            pass
    var_hailo_bench_preset = _str_var(app, "var_hailo_bench_preset", "End-to-end compare")
    var_hailo_custom_full = _bool_var(app, "var_hailo_bench_custom_full", True)
    var_hailo_custom_composed = _bool_var(app, "var_hailo_bench_custom_composed", True)
    var_hailo_custom_part1 = _bool_var(app, "var_hailo_bench_custom_part1", False)
    var_hailo_custom_part2 = _bool_var(app, "var_hailo_bench_custom_part2", False)
    var_hailo_full_hef_order = _str_var(app, "var_hailo_full_hef_order", "Build at end (recommended)")
    var_hailo_full_model_preflight = _str_var(app, "var_hailo_full_model_preflight", "Enabled (plan-aware)")
    ttk.Label(run_defaults, text="Mode:").grid(row=0, column=0, sticky="e", padx=(8, 4), pady=(8, 4))
    ttk.Combobox(run_defaults, textvariable=var_hailo_bench_preset, values=["End-to-end compare", "Split diagnostics", "Everything", "Custom"], width=20, state="readonly").grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(8, 4))
    ttk.Label(run_defaults, text="Full HEF:").grid(row=0, column=2, sticky="e", padx=(8, 4), pady=(8, 4))
    ttk.Combobox(run_defaults, textvariable=var_hailo_full_hef_order, values=["Build at end (recommended)", "Build at start", "Skip full-model HEF"], width=24, state="readonly").grid(row=0, column=3, sticky="w", padx=(0, 12), pady=(8, 4))
    ttk.Label(run_defaults, text="Parser preflight:").grid(row=0, column=4, sticky="e", padx=(8, 4), pady=(8, 4))
    ttk.Combobox(run_defaults, textvariable=var_hailo_full_model_preflight, values=["Enabled (plan-aware)", "Disabled (always try full HEF)"], width=28, state="readonly").grid(row=0, column=5, sticky="w", padx=(0, 12), pady=(8, 4))
    custom = ttk.Frame(run_defaults)
    custom.grid(row=1, column=1, columnspan=8, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Label(custom, text="Custom variants:").pack(side=tk.LEFT, padx=(0, 6))
    ttk.Checkbutton(custom, text="full", variable=var_hailo_custom_full).pack(side=tk.LEFT)
    ttk.Checkbutton(custom, text="composed", variable=var_hailo_custom_composed).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Checkbutton(custom, text="part1", variable=var_hailo_custom_part1).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Checkbutton(custom, text="part2", variable=var_hailo_custom_part2).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Label(run_defaults, text="Used by generated benchmark/evaluation run plans; not configured per run in the Benchmark tab anymore.", foreground="#666", wraplength=1050).grid(row=2, column=0, columnspan=10, sticky="ew", padx=8, pady=(0, 8))

    envs = ttk.LabelFrame(hailo, text="Managed accelerator build environments")
    envs.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 6))
    try:
        for _c in range(3):
            envs.columnconfigure(_c, weight=1)
    except Exception:
        pass

    h8_box = ttk.LabelFrame(envs, text="Hailo-8 DFC")
    h8_box.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
    h10_box = ttk.LabelFrame(envs, text="Hailo-10 DFC")
    h10_box.grid(row=0, column=1, sticky="nsew", padx=4, pady=8)
    deepx_box = ttk.LabelFrame(envs, text="DeepX DX-M1 / DX-COM")
    deepx_box.grid(row=0, column=2, sticky="nsew", padx=(4, 8), pady=8)

    def _pack_hailo_box(box, profile_id: str, label: str, badge_attr: str, profile_arg: str, wheel_subdir: str) -> StatusBadge:
        badge = StatusBadge(box, text=f"{label} …", level="idle")
        badge.pack(side=tk.TOP, anchor="w", padx=8, pady=(8, 4))
        ttk.Label(box, text=f"Wheels: resources/hailo/{wheel_subdir}", wraplength=250).pack(side=tk.TOP, anchor="w", padx=8)
        actions = ttk.Frame(box)
        actions.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 8))
        if app is not None and hasattr(app, "_hailo_show_probe_details"):
            try:
                badge.bind("<Button-1>", lambda _e, _p=profile_id: app._hailo_show_probe_details(_p))
            except Exception:
                pass
        if app is not None and hasattr(app, "_hailo_show_dfc_env_status"):
            btn = ttk.Button(actions, text="Status", command=app._hailo_show_dfc_env_status)
            btn.pack(side=tk.LEFT, padx=(0, 4), pady=(0, 4))
            attach_tooltip(btn, _tt("hailo_env_status"))
        if app is not None and hasattr(app, "_hailo_open_dfc_wheel_folder"):
            btn = ttk.Button(actions, text="Open wheels", command=lambda _sub=wheel_subdir: app._hailo_open_dfc_wheel_folder(_sub))
            btn.pack(side=tk.LEFT, padx=(0, 4), pady=(0, 4))
            attach_tooltip(btn, _tt("hailo_wheel_folder"))
        if app is not None and hasattr(app, "_hailo_provision_dfcs"):
            btn = ttk.Button(actions, text="Install/Repair", command=lambda _pid=profile_arg: app._hailo_provision_dfcs([_pid]))
            btn.pack(side=tk.LEFT, padx=(0, 4), pady=(0, 4))
            attach_tooltip(btn, _tt(f"{profile_arg}_provision") or _tt("hailo_provision"))
            try:
                existing = list(getattr(app, "_hailo_provision_buttons", []) or [])
                existing.append(btn)
                setattr(app, "_hailo_provision_buttons", existing)
            except Exception:
                pass
        try:
            setattr(app, badge_attr, badge)
        except Exception:
            pass
        return badge

    badge_h8 = _pack_hailo_box(h8_box, "hailo8", "Hailo-8", "hailo_badge_h8", "hailo8", "hailo8")
    badge_h10 = _pack_hailo_box(h10_box, "hailo10", "Hailo-10", "hailo_badge_h10", "hailo10", "hailo10")

    def _remote_setup_editor(parent_box: ttk.LabelFrame, *, setup_id: str, accelerator: str, default_venv: str, default_provider: str) -> None:
        """Small central remote setup editor inside each accelerator card.

        The values are persisted to ~/.onnx_splitpoint_tool/hardware_setups.yaml.
        This is now the single source of truth for evaluation and manual remote
        benchmark dispatch; the legacy host/venv fields are filled from here.
        """
        payload = {}
        if app is not None and hasattr(app, "_hardware_setup_remote_payload"):
            try:
                payload = app._hardware_setup_remote_payload(setup_id) or {}
            except Exception:
                payload = {}
        host_var = _str_var(app, f"var_hwsetup_{setup_id}_host", str(payload.get("host") or ""))
        user_var = _str_var(app, f"var_hwsetup_{setup_id}_user", str(payload.get("user") or "nx"))
        port_var = _str_var(app, f"var_hwsetup_{setup_id}_port", str(payload.get("port") or "22"))
        base_var = _str_var(app, f"var_hwsetup_{setup_id}_base", str(payload.get("remote_base_dir") or "~/splitpoint_runs"))
        venv_var = _str_var(app, f"var_hwsetup_{setup_id}_venv", str(payload.get("remote_venv") or default_venv))
        provider_var = _str_var(app, f"var_hwsetup_{setup_id}_provider", str(payload.get("provider") or default_provider))
        # Per-setup u.RECS energy configuration. Shared collector defaults
        # live in the global Energy defaults section below.
        energy_payload = {}
        try:
            if app is not None and hasattr(app, "_hardware_registry_load"):
                for _raw in (app._hardware_registry_load().get("hardware_setups") or []):
                    if isinstance(_raw, dict) and str(_raw.get("id") or "") == setup_id:
                        energy_payload = dict(_raw.get("energy") or {}) if isinstance(_raw.get("energy"), dict) else {}
                        break
        except Exception:
            energy_payload = {}
        energy_enabled_var = _bool_var(app, f"var_hwsetup_{setup_id}_energy_enabled", bool(energy_payload.get("enabled") or False))
        urecs_addr_var = _str_var(app, f"var_hwsetup_{setup_id}_urecs_addr", str(energy_payload.get("urecs_address") or ""))
        idle_w_var = _str_var(app, f"var_hwsetup_{setup_id}_idle_w", "" if energy_payload.get("idle_baseline_w") is None else str(energy_payload.get("idle_baseline_w")))
        accel_idle_w_var = _str_var(app, f"var_hwsetup_{setup_id}_accel_idle_w", "" if energy_payload.get("accelerator_idle_w") is None else str(energy_payload.get("accelerator_idle_w")))
        status_var = _str_var(app, f"var_hwsetup_{setup_id}_status", "")

        # v59b: These card variables are backed by the central hardware registry,
        # not by generic GUI settings.  Force-refresh them from the registry when
        # the hardware tab is built so stale var_hwsetup_* values from old
        # settings files cannot make the cards appear to reset after a tool update.
        try:
            host_var.set(str(payload.get("host") or ""))
            user_var.set(str(payload.get("user") or "nx"))
            port_var.set(str(payload.get("port") or "22"))
            base_var.set(str(payload.get("remote_base_dir") or "~/splitpoint_runs"))
            venv_var.set(str(payload.get("remote_venv") or default_venv))
            provider_var.set(str(payload.get("provider") or default_provider))
            energy_enabled_var.set(bool(energy_payload.get("enabled") or False))
            urecs_addr_var.set(str(energy_payload.get("urecs_address") or ""))
            idle_w_var.set("" if energy_payload.get("idle_baseline_w") is None else str(energy_payload.get("idle_baseline_w")))
            accel_idle_w_var.set("" if energy_payload.get("accelerator_idle_w") is None else str(energy_payload.get("accelerator_idle_w")))
        except Exception:
            pass

        sep = ttk.Separator(parent_box)
        sep.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 6))
        title = ttk.Label(parent_box, text="Remote NX setup", font=("", 9, "bold"))
        title.pack(side=tk.TOP, anchor="w", padx=8)

        grid = ttk.Frame(parent_box)
        grid.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(4, 2))
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=0)
        ttk.Label(grid, text="Host:").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(grid, textvariable=host_var, width=18).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=2)
        ttk.Label(grid, text="User:").grid(row=0, column=2, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(grid, textvariable=user_var, width=8).grid(row=0, column=3, sticky="w", padx=(0, 0), pady=2)

        ttk.Label(grid, text="Base:").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(grid, textvariable=base_var, width=18).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=2)
        ttk.Label(grid, text="Port:").grid(row=1, column=2, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(grid, textvariable=port_var, width=8).grid(row=1, column=3, sticky="w", pady=2)

        ttk.Label(grid, text="Venv:").grid(row=2, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(grid, textvariable=venv_var, width=26).grid(row=2, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Label(grid, text="Provider:").grid(row=3, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(grid, textvariable=provider_var, width=14).grid(row=3, column=1, sticky="w", padx=(0, 6), pady=2)
        cb_energy = ttk.Checkbutton(grid, text="Energy", variable=energy_enabled_var)
        cb_energy.grid(row=3, column=2, sticky="w", padx=(4, 4), pady=2)
        attach_tooltip(cb_energy, "Enables u.RECS energy measurement for this hardware setup. It does not start a measurement by itself; Benchmark/Eval energy measurement is enabled separately and uses the global Energy defaults.")

        ttk.Label(grid, text="u.RECS:").grid(row=4, column=0, sticky="w", padx=(0, 4), pady=(6, 2))
        ttk.Entry(grid, textvariable=urecs_addr_var, width=16).grid(row=4, column=1, sticky="ew", padx=(0, 6), pady=(6, 2))
        ttk.Label(grid, text="Idle W:").grid(row=4, column=2, sticky="w", padx=(0, 4), pady=(6, 2))
        ttk.Entry(grid, textvariable=idle_w_var, width=8).grid(row=4, column=3, sticky="w", pady=(6, 2))
        ttk.Label(grid, text="Calibrated M.2 idle W:").grid(row=5, column=0, sticky="w", padx=(0, 4), pady=2)
        ttk.Entry(
            grid,
            textvariable=accel_idle_w_var,
            width=8,
            state="readonly",
        ).grid(row=5, column=1, sticky="w", padx=(0, 6), pady=2)

        actions = ttk.Frame(parent_box)
        actions.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 8))
        ttk.Label(actions, textvariable=status_var, foreground="#555").pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _save() -> None:
            if app is None or not hasattr(app, "_hardware_registry_load"):
                status_var.set("No app registry API")
                return
            try:
                reg = app._hardware_registry_load()
                setups = list(reg.get("hardware_setups") or [])
                found = None
                for raw in setups:
                    if isinstance(raw, dict) and str(raw.get("id") or "") == setup_id:
                        found = raw
                        break
                if found is None:
                    found = {"id": setup_id, "label": setup_id, "accelerator": accelerator}
                    setups.append(found)
                def _float_or_none(_v):
                    try:
                        _s = str(_v or "").strip()
                        return None if not _s else float(_s)
                    except Exception:
                        return None
                _merge_accelerator_env_card_fields(
                    found,
                    accelerator=accelerator,
                    host_address=str(host_var.get() or "").strip(),
                    host_user=str(user_var.get() or "nx").strip() or "nx",
                    host_port=int(str(port_var.get() or "22").strip() or 22),
                    remote_base_dir=(
                        str(base_var.get() or "~/splitpoint_runs").strip()
                        or "~/splitpoint_runs"
                    ),
                    remote_venv=(
                        str(venv_var.get() or default_venv).strip()
                        or default_venv
                    ),
                    provider=(
                        str(provider_var.get() or default_provider).strip()
                        or default_provider
                    ),
                    energy_enabled=bool(energy_enabled_var.get()),
                    urecs_address=str(urecs_addr_var.get() or "").strip(),
                    idle_baseline_w=_float_or_none(idle_w_var.get()),
                    accelerator_idle_w=_float_or_none(accel_idle_w_var.get()),
                )
                reg["hardware_setups"] = setups
                app._hardware_registry_save(reg)
                if hasattr(app, "_sync_remote_hosts_from_hardware_setups"):
                    app._sync_remote_hosts_from_hardware_setups()
                status_var.set("saved")
            except Exception as exc:
                # A platform calibration may have committed between this
                # card's full-registry load and save.  The central CAS writer
                # rejects that stale payload.  Refresh every displayed field
                # from the winning registry revision instead of inviting a
                # second click with the same stale values.
                if isinstance(exc, HardwareRegistryConflictError):
                    try:
                        latest = app._hardware_registry_load()
                        latest_row = next(
                            (
                                raw
                                for raw in (latest.get("hardware_setups") or [])
                                if isinstance(raw, dict)
                                and str(raw.get("id") or "") == setup_id
                            ),
                            {},
                        )
                        latest_host = dict(latest_row.get("host") or {})
                        latest_runtime = dict(latest_row.get("runtime") or {})
                        latest_remote = dict(latest_row.get("remote") or {})
                        latest_energy = dict(latest_row.get("energy") or {})
                        host_var.set(str(latest_host.get("address") or ""))
                        user_var.set(str(latest_host.get("user") or "nx"))
                        port_var.set(str(latest_host.get("port") or "22"))
                        base_var.set(
                            str(
                                latest_host.get("base_dir")
                                or latest_remote.get("remote_base_dir")
                                or "~/splitpoint_runs"
                            )
                        )
                        venv_var.set(
                            str(
                                latest_runtime.get("activate")
                                or latest_runtime.get("venv")
                                or latest_remote.get("remote_venv")
                                or default_venv
                            )
                        )
                        provider_var.set(
                            str(
                                latest_runtime.get("provider")
                                or latest_remote.get("provider")
                                or default_provider
                            )
                        )
                        energy_enabled_var.set(bool(latest_energy.get("enabled") or False))
                        urecs_addr_var.set(str(latest_energy.get("urecs_address") or ""))
                        idle_w_var.set(
                            ""
                            if latest_energy.get("idle_baseline_w") is None
                            else str(latest_energy.get("idle_baseline_w"))
                        )
                        accel_idle_w_var.set(
                            ""
                            if latest_energy.get("accelerator_idle_w") is None
                            else str(latest_energy.get("accelerator_idle_w"))
                        )
                    except Exception:
                        pass
                    status_var.set("save blocked: registry changed; fields reloaded")
                    try:
                        messagebox.showwarning(
                            "Hardware setup changed",
                            "The hardware registry changed while this card was open. "
                            "The stale save was rejected and the current values were reloaded.",
                        )
                    except Exception:
                        pass
                    return
                status_var.set(f"save failed: {type(exc).__name__}")

        ttk.Button(actions, text="Save", command=_save).pack(side=tk.RIGHT, padx=(4, 0))
        def _test() -> None:
            _save()
            if app is not None and hasattr(app, "_test_hardware_setup_async"):
                try:
                    status_var.set("testing…")
                    app._test_hardware_setup_async(setup_id, status_var=status_var)
                except Exception as exc:
                    status_var.set(f"test failed: {type(exc).__name__}")
            else:
                status_var.set("No test API")
        btn_test_setup = ttk.Button(actions, text="Test", command=_test)
        btn_test_setup.pack(side=tk.RIGHT, padx=(4, 0))
        def _test_energy() -> None:
            _save()
            try:
                script = Path.cwd() / "scripts" / "check_energy_setup.py"
                try:
                    from ...energy.config import energy_measurements_root
                    out_dir = energy_measurements_root(getattr(app, "default_output_dir", None)) / "Tests" / setup_id / time.strftime("%Y%m%d_%H%M%S")
                except Exception:
                    out_dir = Path.cwd() / "EnergyMeasurements" / "Tests" / setup_id / time.strftime("%Y%m%d_%H%M%S")
                cmd = [sys.executable, str(script), "--setup-id", setup_id, "--test", "sleep", "--sleep", "2", "--output-dir", str(out_dir)]
                _env = os.environ.copy()
                _extras = [str(Path.home() / ".cargo" / "bin"), str(Path.home() / ".local" / "bin"), "/usr/local/bin"]
                _parts = [p for p in str(_env.get("PATH", "")).split(os.pathsep) if p]
                for _extra in _extras:
                    if _extra not in _parts:
                        _parts.insert(0, _extra)
                _env["PATH"] = os.pathsep.join(_parts)
                proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(Path.cwd()), env=_env)
                raw = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
                ok = proc.returncode == 0
                text = raw
                try:
                    data = json.loads((proc.stdout or "").strip().splitlines()[-1])
                    status_line = "SUCCESS" if data.get("ok") else "FAIL"
                    reason = data.get("error") or (data.get("measurement") or {}).get("reason") or (data.get("measurement") or {}).get("status") or ""
                    text = f"{status_line}: {setup_id} energy sleep measurement\nuRECS={urecs_addr_var.get()}\nreason={reason}\n\n--- raw ---\n" + json.dumps(data, indent=2)
                    ok = bool(data.get("ok"))
                except Exception:
                    pass
                (messagebox.showinfo if ok else messagebox.showwarning)("Energy setup test", text[:12000])
                status_var.set("energy OK" if ok else "energy failed")
            except Exception as exc:
                status_var.set(f"energy test failed: {type(exc).__name__}")
                messagebox.showerror("Energy setup test", f"Failed to run energy test:\n{type(exc).__name__}: {exc}")
        btn_energy_setup = ttk.Button(actions, text="Test energy", command=_test_energy)
        btn_energy_setup.pack(side=tk.RIGHT, padx=(4, 0))
        attach_tooltip(btn_test_setup, f"SSH + runtime preflight for {setup_id}. Saves this setup first, then checks the remote venv/provider.")
        attach_tooltip(btn_energy_setup, "Runs a short local u.RECS fast-firmware sleep measurement for this setup. Requires urecs-data-collector and u.RECS address.")
        attach_tooltip(grid, f"Central remote setup for {setup_id}. Benchmark/Evaluation dispatch maps run-plan rows to this setup automatically.")

    _remote_setup_editor(h8_box, setup_id="orin_nx_hailo8_01", accelerator="hailo8", default_venv="source ~/hailo_py/bin/activate", default_provider="hailo8")
    _remote_setup_editor(h10_box, setup_id="orin_nx_hailo10_01", accelerator="hailo10", default_venv="source ~/hailo_py/bin/activate", default_provider="hailo10h")

    # Details line (compute summary + probe reasons)
    status_details_var = _str_var(app, "var_hailo_status_details", "")
    status_line = ttk.Label(envs, textvariable=status_details_var, wraplength=1080)
    status_line.grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8))
    attach_tooltip(status_line, _tt("hailo_status"))

    # DeepX box
    deepx_root_var = _str_var(app, "var_deepx_root", "auto")
    badge_dx = StatusBadge(deepx_box, text="DeepX … (venv)", level="idle")
    badge_dx.pack(side=tk.TOP, anchor="w", padx=8, pady=(8, 4))
    root_row = ttk.Frame(deepx_box)
    root_row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
    ttk.Label(root_row, text="dx-all-suite:").pack(side=tk.LEFT)
    ent_deepx_root = ttk.Entry(root_row, textvariable=deepx_root_var, width=32)
    ent_deepx_root.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
    attach_tooltip(ent_deepx_root, _tt("deepx_root"))
    dx_actions = ttk.Frame(deepx_box)
    dx_actions.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
    if app is not None:
        try:
            setattr(app, "deepx_badge_env", badge_dx)
            setattr(app, "deepx_badge", badge_dx)
        except Exception:
            pass
    if app is not None and hasattr(app, "_deepx_show_env_status"):
        btn_dx_status = ttk.Button(dx_actions, text="Status", command=app._deepx_show_env_status)
        btn_dx_status.pack(side=tk.LEFT, padx=(0, 4), pady=(0, 4))
        attach_tooltip(btn_dx_status, _tt("deepx_status"))
    if app is not None and hasattr(app, "_deepx_open_root"):
        btn_dx_open = ttk.Button(dx_actions, text="Open dx-all-suite", command=app._deepx_open_root)
        btn_dx_open.pack(side=tk.LEFT, padx=(0, 4), pady=(0, 4))
        attach_tooltip(btn_dx_open, _tt("deepx_open_root"))
    if app is not None and hasattr(app, "_deepx_provision_runtime"):
        btn_dx_prov = ttk.Button(dx_actions, text="Install/Repair runtime", command=app._deepx_provision_runtime)
        btn_dx_prov.pack(side=tk.LEFT, padx=(0, 4), pady=(0, 4))
        attach_tooltip(btn_dx_prov, _tt("deepx_provision"))
        btn_dx_comp = ttk.Button(dx_actions, text="Install/Repair compiler", command=lambda: app._deepx_provision_runtime(run_compiler_install=True))
        btn_dx_comp.pack(side=tk.LEFT, padx=(0, 4), pady=(0, 4))
        attach_tooltip(btn_dx_comp, _tt("deepx_compiler_install"))
        try:
            setattr(app, "_deepx_btn_provision", btn_dx_prov)
            setattr(app, "_deepx_btn_provision_compiler", btn_dx_comp)
        except Exception:
            pass

    try:
        _remote_setup_editor(deepx_box, setup_id="orin_nx_deepx_m1_01", accelerator="deepx_m1", default_venv="source ~/venvs/deepx-runtime/bin/activate", default_provider="deepx_m1")
    except Exception:
        pass

    common_actions = ttk.Frame(envs)
    common_actions.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
    if app is not None and hasattr(app, "_show_build_environment_status"):
        btn_build_env = ttk.Button(common_actions, text="All build env status", command=app._show_build_environment_status)
        btn_build_env.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_build_env, _tt("build_env_status"))
    if app is not None and hasattr(app, "_open_build_environment_config"):
        btn_env_cfg = ttk.Button(common_actions, text="Open build env YAML", command=app._open_build_environment_config)
        btn_env_cfg.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_env_cfg, _tt("build_env_config"))
    if app is not None and hasattr(app, "_test_all_hardware_setups_remote"):
        btn_test_all = ttk.Button(common_actions, text="Test all remote setups", command=app._test_all_hardware_setups_remote)
        btn_test_all.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_test_all, "Runs SSH/runtime smoke tests for every configured Hailo/DeepX remote setup.")
    if app is not None and hasattr(app, "_open_hardware_setups_config"):
        btn_hw_cfg = ttk.Button(common_actions, text="Open hardware setups YAML", command=app._open_hardware_setups_config)
        btn_hw_cfg.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_hw_cfg, "Opens ~/.onnx_splitpoint_tool/hardware_setups.yaml, the central remote setup registry.")
    if app is not None and hasattr(app, "_test_all_hardware_setups"):
        btn_test_all = ttk.Button(common_actions, text="Test all remote setups", command=app._test_all_hardware_setups)
        btn_test_all.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_test_all, "Runs SSH/runtime preflight for all configured Hailo-8, Hailo-10 and DeepX remote setups.")
    if app is not None and hasattr(app, "_open_provisioning_logs_folder"):
        btn_prov_logs = ttk.Button(common_actions, text="Open provision logs", command=app._open_provisioning_logs_folder)
        btn_prov_logs.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_prov_logs, _tt("provisioning_logs"))
    if app is not None and hasattr(app, "_hailo_provision_dfcs"):
        btn_all_hailo = ttk.Button(common_actions, text="Install/Repair both Hailo DFCs", command=lambda: app._hailo_provision_dfcs())
        btn_all_hailo.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_all_hailo, _tt("hailo_provision"))
        try:
            setattr(app, "_hailo_btn_provision", btn_all_hailo)
            existing = list(getattr(app, "_hailo_provision_buttons", []) or [])
            existing.append(btn_all_hailo)
            setattr(app, "_hailo_provision_buttons", existing)
        except Exception:
            pass
    if app is not None and hasattr(app, "_hailo_clear_cache"):
        btn_clear = ttk.Button(common_actions, text="Clear parse cache", command=app._hailo_clear_cache)
        btn_clear.pack(side=tk.LEFT, padx=(0, 4))
        attach_tooltip(btn_clear, _tt("hailo_clear"))


    # ------------------------------------------------------------------
    # Energy measurement defaults (u.RECS fast-firmware)
    # ------------------------------------------------------------------
    energy_defaults_box = ttk.LabelFrame(tab_energy, text="Global energy measurement defaults (u.RECS fast-firmware)")
    energy_defaults_box.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 8))
    for _c in range(12):
        try:
            energy_defaults_box.columnconfigure(_c, weight=1 if _c in (2, 4) else 0)
        except Exception:
            pass
    try:
        from ...energy.config import load_energy_defaults as _load_energy_defaults
        _ed = _load_energy_defaults()
    except Exception:
        _ed = None
    e_enabled = _bool_var(app, "var_energy_enabled", bool(getattr(_ed, "enabled", False)))
    e_collector = _str_var(app, "var_energy_collector_binary", str(getattr(_ed, "collector_binary", "urecs-data-collector")))
    e_power = _str_var(app, "var_energy_power_calculations_binary", str(getattr(_ed, "power_calculations_binary", "power_calculations")))
    e_port = _str_var(app, "var_energy_data_port", str(getattr(_ed, "data_port", 3000)))
    e_channel = _str_var(app, "var_energy_channel", str(getattr(_ed, "channel", 0)))
    e_samplerate = _str_var(app, "var_energy_sample_rate", str(getattr(_ed, "sample_rate", 2000)))
    e_env = _str_var(app, "var_energy_environment", str(getattr(_ed, "environment", "Jetson")))
    e_pre = _str_var(app, "var_energy_pre_duration_s", str(getattr(_ed, "pre_duration_s", 5)))
    e_post = _str_var(app, "var_energy_post_duration_s", str(getattr(_ed, "post_duration_s", 5)))
    e_margin = _str_var(app, "var_energy_duration_margin_s", str(getattr(_ed, "duration_margin_s", 1)))
    e_min_active = _str_var(app, "var_energy_min_active_duration_s", str(getattr(_ed, "min_active_duration_s", 30)))
    e_native_duration = _str_var(app, "var_energy_native_duration_s", str(getattr(_ed, "native_energy_duration_s", 60)))
    e_generic_duration = _str_var(app, "var_energy_generic_duration_s", str(getattr(_ed, "generic_energy_duration_s", 60)))
    e_runs = _str_var(app, "var_energy_run_count", str(getattr(_ed, "run_count", 3)))
    e_keep_raw = _bool_var(app, "var_energy_keep_raw_parquet", bool(getattr(_ed, "keep_raw_parquet", True)))
    e_include_raw = _bool_var(app, "var_energy_include_raw_parquet_debug", bool(getattr(_ed, "include_raw_parquet_in_debug_pack", False)))
    e_compare_legacy = _bool_var(app, "var_energy_compare_legacy_window", bool(getattr(_ed, "compare_legacy_window", True)))
    e_probe_enabled = _bool_var(app, "var_energy_window_probe_enabled", bool(getattr(_ed, "window_method_validation_probe_enabled", True)))
    e_probe_repeats = _str_var(app, "var_energy_window_probe_repeats", str(getattr(_ed, "window_method_validation_probe_repeats", 3)))
    e_probe_raw = _bool_var(app, "var_energy_window_probe_include_raw", bool(getattr(_ed, "window_method_validation_probe_include_raw_parquet", True)))
    e_probe_strict = _bool_var(app, "var_energy_window_probe_strict", bool(getattr(_ed, "window_method_validation_probe_strict", True)))
    ttk.Checkbutton(energy_defaults_box, text="Enable energy by default", variable=e_enabled).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
    ttk.Label(energy_defaults_box, text="Collector:").grid(row=0, column=1, sticky="e", padx=(8, 4), pady=(8, 4))
    ttk.Entry(energy_defaults_box, textvariable=e_collector, width=24).grid(row=0, column=2, sticky="ew", padx=(0, 8), pady=(8, 4))
    ttk.Label(energy_defaults_box, text="Power calc:").grid(row=0, column=3, sticky="e", padx=(8, 4), pady=(8, 4))
    ttk.Entry(energy_defaults_box, textvariable=e_power, width=24).grid(row=0, column=4, sticky="ew", padx=(0, 8), pady=(8, 4))
    ttk.Label(energy_defaults_box, text="Port / channel / rate:").grid(row=1, column=0, sticky="e", padx=(8, 4), pady=4)
    ttk.Entry(energy_defaults_box, textvariable=e_port, width=7).grid(row=1, column=1, sticky="w", padx=(0, 2), pady=4)
    ttk.Entry(energy_defaults_box, textvariable=e_channel, width=5).grid(row=1, column=2, sticky="w", padx=(0, 2), pady=4)
    ttk.Entry(energy_defaults_box, textvariable=e_samplerate, width=8).grid(row=1, column=3, sticky="w", padx=(0, 8), pady=4)
    ttk.Label(energy_defaults_box, text="Environment:").grid(row=1, column=4, sticky="e", padx=(8, 4), pady=4)
    ttk.Combobox(energy_defaults_box, textvariable=e_env, values=["Jetson", "Static"], state="readonly", width=10).grid(row=1, column=5, sticky="w", padx=(0, 8), pady=4)
    ttk.Label(energy_defaults_box, text="pre / post / margin:").grid(row=2, column=0, sticky="e", padx=(8, 4), pady=(4, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_pre, width=6).grid(row=2, column=1, sticky="w", padx=(0, 2), pady=(4, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_post, width=6).grid(row=2, column=2, sticky="w", padx=(0, 2), pady=(4, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_margin, width=6).grid(row=2, column=3, sticky="w", padx=(0, 8), pady=(4, 8))
    ttk.Label(energy_defaults_box, text="Min active s:").grid(row=2, column=4, sticky="e", padx=(8, 4), pady=(4, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_min_active, width=6).grid(row=2, column=5, sticky="w", padx=(0, 8), pady=(4, 8))
    ttk.Label(energy_defaults_box, text="Native/generic energy duration s:").grid(row=3, column=0, sticky="e", padx=(8, 4), pady=(4, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_native_duration, width=7).grid(row=3, column=1, sticky="w", padx=(0, 2), pady=(4, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_generic_duration, width=7).grid(row=3, column=2, sticky="w", padx=(0, 8), pady=(4, 8))
    ttk.Label(energy_defaults_box, text="Runs:").grid(row=3, column=3, sticky="e", padx=(8, 4), pady=(4, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_runs, width=6).grid(row=3, column=4, sticky="w", padx=(0, 8), pady=(4, 8))
    ttk.Label(
        energy_defaults_box,
        text="Physical scope: FS (full system)",
    ).grid(row=3, column=5, columnspan=4, sticky="w", padx=(8, 4), pady=(4, 8))
    ttk.Checkbutton(energy_defaults_box, text="Keep raw parquet", variable=e_keep_raw).grid(row=4, column=0, columnspan=2, sticky="w", padx=(8, 4), pady=(0, 8))
    ttk.Checkbutton(energy_defaults_box, text="Include parquet in debug pack", variable=e_include_raw).grid(row=4, column=2, columnspan=3, sticky="w", padx=(8, 4), pady=(0, 8))
    ttk.Checkbutton(energy_defaults_box, text="Compare legacy flank window (diagnostic)", variable=e_compare_legacy).grid(row=4, column=5, columnspan=5, sticky="w", padx=(8, 4), pady=(0, 8))
    ttk.Checkbutton(energy_defaults_box, text="Native window A/B probe", variable=e_probe_enabled).grid(row=5, column=0, columnspan=2, sticky="w", padx=(8, 4), pady=(0, 8))
    ttk.Label(energy_defaults_box, text="Probe repeats:").grid(row=5, column=2, sticky="e", padx=(8, 4), pady=(0, 8))
    ttk.Entry(energy_defaults_box, textvariable=e_probe_repeats, width=6).grid(row=5, column=3, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Checkbutton(energy_defaults_box, text="Probe Parquet in debug", variable=e_probe_raw).grid(row=5, column=4, columnspan=3, sticky="w", padx=(8, 4), pady=(0, 8))
    ttk.Checkbutton(energy_defaults_box, text="Probe strict", variable=e_probe_strict).grid(row=5, column=7, columnspan=2, sticky="w", padx=(8, 4), pady=(0, 8))

    def _save_energy_defaults() -> None:
        try:
            from ...energy.config import EnergyDefaults, save_energy_defaults
            def _f(v, d):
                try: return float(str(v.get() or d))
                except Exception: return float(d)
            def _i(v, d):
                try: return int(float(str(v.get() or d)))
                except Exception: return int(d)
            defaults = EnergyDefaults(
                enabled=bool(e_enabled.get()),
                collector_binary=str(e_collector.get() or "urecs-data-collector"),
                power_calculations_binary=str(e_power.get() or "power_calculations"),
                mode="fast_firmware",
                data_port=_i(e_port, 3000),
                channel=_i(e_channel, 0),
                sample_rate=_i(e_samplerate, 2000),
                environment=str(e_env.get() or "Jetson"),
                pre_duration_s=_f(e_pre, 5),
                post_duration_s=_f(e_post, 5),
                duration_margin_s=_f(e_margin, 1),
                min_active_duration_s=_f(e_min_active, 30),
                native_energy_duration_s=max(1.0, _f(e_native_duration, 60)),
                generic_energy_duration_s=max(1.0, _f(e_generic_duration, 60)),
                run_count=max(1, _i(e_runs, 3)),
                physical_scope="FS",
                window_label="command",
                keep_raw_parquet=bool(e_keep_raw.get()),
                compare_legacy_window=bool(e_compare_legacy.get()),
                window_method_validation_probe_enabled=bool(e_probe_enabled.get()),
                window_method_validation_probe_repeats=max(1, _i(e_probe_repeats, 3)),
                window_method_validation_probe_include_raw_parquet=bool(e_probe_raw.get()),
                window_method_validation_probe_strict=bool(e_probe_strict.get()),
                include_raw_parquet_in_debug_pack=bool(e_include_raw.get()),
            )
            save_energy_defaults(defaults)
            if app is not None and hasattr(app, "_persist_settings"):
                try: app._persist_settings()
                except Exception: pass
            messagebox.showinfo("Energy defaults", "Saved energy defaults.")
        except Exception as exc:
            messagebox.showerror("Energy defaults", f"Failed to save energy defaults:\n{type(exc).__name__}: {exc}")

    def _check_energy_defaults() -> None:
        _save_energy_defaults()
        script = Path.cwd() / "scripts" / "check_energy_setup.py"
        cmd = [sys.executable, str(script)]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(Path.cwd()))
            raw = (proc.stdout or "") + (("\n\n[stderr]\n" + proc.stderr) if proc.stderr else "")
            ok = proc.returncode == 0
            dlg = tk.Toplevel(energy_defaults_box)
            dlg.title(("SUCCESS — " if ok else "FAIL — ") + "Energy collector tools")
            dlg.geometry("850x540"); dlg.columnconfigure(0, weight=1); dlg.rowconfigure(1, weight=1)
            ttk.Label(dlg, text="SUCCESS" if ok else "FAIL", foreground="#0a7d24" if ok else "#b00020", font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))
            txt = tk.Text(dlg, wrap="none"); txt.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8)); txt.insert("1.0", raw[:20000]); txt.configure(state="disabled")
            ttk.Button(dlg, text="OK", command=dlg.destroy).grid(row=2, column=0, sticky="e", padx=12, pady=(0, 12))
        except Exception as exc:
            messagebox.showerror("Energy defaults", f"Failed to run energy check:\n{type(exc).__name__}: {exc}")
    ttk.Button(energy_defaults_box, text="Save energy defaults", command=_save_energy_defaults).grid(row=3, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
    ttk.Button(energy_defaults_box, text="Test collector tools", command=_check_energy_defaults).grid(row=3, column=2, columnspan=2, sticky="w", padx=8, pady=(0, 8))
    ttk.Label(energy_defaults_box, text="fast-firmware only. These defaults are shared by all hardware setups. Native/generic energy durations are configured here once and reused by EvalRuns; frame counts are derived internally from measured FPS. Per-setup u.RECS IP and idle baselines are configured in the Accelerator envs cards above.", foreground="#666", wraplength=1100).grid(row=5, column=0, columnspan=12, sticky="ew", padx=8, pady=(0, 8))


    # ------------------------------------------------------------------
    # Tool-wide validation assets
    # ------------------------------------------------------------------
    assets = ttk.LabelFrame(tab_validation, text="Tool-wide validation datasets")
    assets.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 8))
    try:
        assets.columnconfigure(1, weight=1)
    except Exception:
        pass
    val_status_var = _str_var(app, "var_tool_config_validation_status", "COCO-50 / COCO-200 / Imagenette validation assets: not checked in this session")
    ttk.Label(
        assets,
        text="Diese Datasets sind globale Tool-Konfiguration, nicht Teil eines einzelnen Evaluation-Runs.",
        foreground="#666",
        wraplength=1050,
        justify="left",
    ).grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 4))
    ttk.Label(assets, textvariable=val_status_var, wraplength=980).grid(row=1, column=0, columnspan=4, sticky="ew", padx=8, pady=(0, 6))
    if app is not None and hasattr(app, "_evaluation_workflow_check_validation_assets"):
        btn_val_status = ttk.Button(assets, text="Validation assets status", command=app._evaluation_workflow_check_validation_assets)
        btn_val_status.grid(row=2, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
        attach_tooltip(btn_val_status, "Prüft COCO-50, COCO-200, Imagenette-mini und Testbilder unter ~/.onnx_splitpoint_tool/validation_datasets.")
    if app is not None and hasattr(app, "_queue_evaluation_validation_assets"):
        btn_val_prepare = ttk.Button(assets, text="Prepare validation assets…", command=app._queue_evaluation_validation_assets)
        btn_val_prepare.grid(row=2, column=1, sticky="w", padx=(4, 8), pady=(0, 8))
        attach_tooltip(btn_val_prepare, "Bereitet COCO-50 Detection-Validation, COCO-200 Detection-Calibration, Imagenette-mini Classification und Testbilder außerhalb der Tool-ZIP vor.")

    # ------------------------------------------------------------------
    # Tool-wide calibration/validation policy
    # ------------------------------------------------------------------
    policy = ttk.LabelFrame(tab_validation, text="Calibration / validation split")
    policy.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
    for _c in range(8):
        try:
            policy.columnconfigure(_c, weight=1 if _c in (1, 3, 5, 7) else 0)
        except Exception:
            pass

    preset_values = list_available_presets()
    cls_calib_var = _str_var(app, "var_tool_cls_calib_preset", "imagenette_val_mini_500")
    cls_val_var = _str_var(app, "var_tool_cls_validation_preset", "imagenette_val_mini_200")
    det_calib_var = _str_var(app, "var_tool_det_calib_preset", "coco_200")
    det_val_var = _str_var(app, "var_tool_det_validation_preset", "coco_50")
    val_max_cls_var = _str_var(app, "var_tool_cls_validation_max", "200")
    val_max_det_var = _str_var(app, "var_tool_det_validation_max", "50")
    calib_count_var = _str_var(app, "var_tool_calibration_count", "200")

    ttk.Label(policy, text="Classification calib:").grid(row=0, column=0, sticky="e", padx=(8, 6), pady=(8, 4))
    ttk.Combobox(policy, textvariable=cls_calib_var, values=preset_values, state="readonly", width=28).grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=(8, 4))
    ttk.Label(policy, text="Classification validation:").grid(row=0, column=2, sticky="e", padx=(8, 6), pady=(8, 4))
    ttk.Combobox(policy, textvariable=cls_val_var, values=preset_values, state="readonly", width=28).grid(row=0, column=3, sticky="ew", padx=(0, 12), pady=(8, 4))
    ttk.Label(policy, text="Max val imgs:").grid(row=0, column=4, sticky="e", padx=(8, 6), pady=(8, 4))
    ttk.Entry(policy, textvariable=val_max_cls_var, width=7).grid(row=0, column=5, sticky="w", padx=(0, 12), pady=(8, 4))

    ttk.Label(policy, text="Detection calib:").grid(row=1, column=0, sticky="e", padx=(8, 6), pady=4)
    ttk.Combobox(policy, textvariable=det_calib_var, values=["coco_200", "coco_50"], state="readonly", width=28).grid(row=1, column=1, sticky="ew", padx=(0, 12), pady=4)
    ttk.Label(policy, text="Detection validation:").grid(row=1, column=2, sticky="e", padx=(8, 6), pady=4)
    ttk.Combobox(policy, textvariable=det_val_var, values=["coco_50", "coco_200"], state="readonly", width=28).grid(row=1, column=3, sticky="ew", padx=(0, 12), pady=4)
    ttk.Label(policy, text="Max val imgs:").grid(row=1, column=4, sticky="e", padx=(8, 6), pady=4)
    ttk.Entry(policy, textvariable=val_max_det_var, width=7).grid(row=1, column=5, sticky="w", padx=(0, 12), pady=4)
    ttk.Label(policy, text="Calib imgs:").grid(row=1, column=6, sticky="e", padx=(8, 6), pady=4)
    ttk.Entry(policy, textvariable=calib_count_var, width=8).grid(row=1, column=7, sticky="w", padx=(0, 8), pady=4)

    ttk.Label(policy, text="Calibration is used for HEF/DXNN generation (defaults: Imagenette-500 / COCO-200). Validation is used for final semantic metrics (defaults: Imagenette-200 / COCO-50). Detection defaults: COCO-200 calibration images, COCO-50 annotated validation.", foreground="#666", wraplength=1100).grid(row=2, column=0, columnspan=8, sticky="ew", padx=8, pady=(4, 8))

    # ------------------------------------------------------------------
    # Final campaign datasets, manifests and official COCO evaluation
    # ------------------------------------------------------------------
    try:
        bind_registry_to_app(app)
    except Exception:
        pass

    final_box = ttk.LabelFrame(tab_final_data, text="Final ImageNet / COCO registry")
    final_box.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 8))
    final_box.columnconfigure(1, weight=1)
    final_box.columnconfigure(3, weight=1)
    final_root_var = _str_var(app, "var_final_dataset_root", str(default_dataset_root()))
    final_registry_var = _str_var(app, "var_final_dataset_registry", str(default_registry_path()))
    final_status_var = _str_var(app, "var_final_dataset_status", "Final dataset registry: not checked")
    cls_cal_manifest_var = _str_var(app, "var_manifest_cls_calibration", "")
    cls_val_manifest_var = _str_var(app, "var_manifest_cls_validation", "")
    det_cal_manifest_var = _str_var(app, "var_manifest_det_calibration", "")
    det_val_manifest_var = _str_var(app, "var_manifest_det_validation", "")
    imagenet_train_var = _str_var(app, "var_final_imagenet_train", "")
    imagenet_val_var = _str_var(app, "var_final_imagenet_val", "")
    imagenet_labels_var = _str_var(app, "var_final_imagenet_labels", "")
    coco_train_var = _str_var(app, "var_final_coco_train", "")
    coco_val_var = _str_var(app, "var_final_coco_val", "")
    coco_ann_var = _str_var(app, "var_final_coco_annotations", "")

    ttk.Label(final_box, text=(
        "Finale Datensätze werden außerhalb des Tool-ZIPs installiert und über eine content-addressed Registry gebunden. "
        "COCO kann automatisch provisioniert werden; ImageNet verwendet einen bereits autorisierten lokalen Download oder die authentifizierte Kaggle CLI."
    ), foreground="#555", wraplength=1100, justify="left").grid(row=0, column=0, columnspan=4, sticky="ew", padx=8, pady=(8, 4))
    ttk.Label(final_box, text="Dataset root:").grid(row=1, column=0, sticky="e", padx=(8, 6), pady=4)
    ttk.Entry(final_box, textvariable=final_root_var).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(final_box, text="Ordner…", command=lambda: final_root_var.set(filedialog.askdirectory(parent=frame, initialdir=final_root_var.get() or str(Path.home())) or final_root_var.get())).grid(row=1, column=2, padx=6, pady=4)
    ttk.Label(final_box, text="Registry JSON:").grid(row=1, column=3, sticky="e", padx=(8, 6), pady=4)
    ttk.Entry(final_box, textvariable=final_registry_var).grid(row=1, column=4, sticky="ew", padx=(0, 8), pady=4)

    def _refresh_final_dataset_status() -> None:
        try:
            bind_registry_to_app(app, final_registry_var.get() or None)
            payload = registry_status(final_registry_var.get() or None, verify_manifests=False)
            missing = list(payload.get("missing_required_manifests") or [])
            coco_status = pycocotools_status()
            tasks = dict(payload.get("task_readiness") or {})
            cls_ready = bool((tasks.get("classification") or {}).get("ready"))
            det_ready = bool((tasks.get("detection") or {}).get("ready"))
            text = ("READY" if payload.get("ready_for_final_profile") else "INCOMPLETE")
            text += f" — ImageNet: {'ready' if cls_ready else 'incomplete'}"
            text += f" — COCO: {'ready' if det_ready else 'incomplete'}"
            text += f" — missing manifests: {', '.join(missing) if missing else 'none'}"
            text += f" — pycocotools: {'available' if coco_status.get('available') else 'missing'}"
            final_status_var.set(text)
        except Exception as exc:
            final_status_var.set(f"ERROR: {type(exc).__name__}: {exc}")

    def _open_final_dataset_dialog() -> None:
        open_dataset_provisioning_dialog(frame, app=app, on_updated=_refresh_final_dataset_status)

    ttk.Button(final_box, text="Provision / import / register…", command=_open_final_dataset_dialog).grid(row=2, column=0, sticky="w", padx=8, pady=6)
    ttk.Button(final_box, text="Registry & manifest status", command=_refresh_final_dataset_status).grid(row=2, column=1, sticky="w", padx=6, pady=6)
    ttk.Label(final_box, textvariable=final_status_var, wraplength=1050).grid(row=3, column=0, columnspan=5, sticky="ew", padx=8, pady=(0, 8))

    paths = ttk.LabelFrame(tab_final_data, text="Registered paths used by final profiles")
    paths.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
    paths.columnconfigure(1, weight=1); paths.columnconfigure(3, weight=1)
    rows = [
        ("ImageNet train/calib:", imagenet_train_var, "ImageNet val:", imagenet_val_var),
        ("ImageNet labels:", imagenet_labels_var, "COCO train/calib:", coco_train_var),
        ("COCO val:", coco_val_var, "COCO annotations:", coco_ann_var),
        ("CLS calibration manifest:", cls_cal_manifest_var, "CLS validation manifest:", cls_val_manifest_var),
        ("DET calibration manifest:", det_cal_manifest_var, "DET validation manifest:", det_val_manifest_var),
    ]
    for r, (l1,v1,l2,v2) in enumerate(rows):
        ttk.Label(paths, text=l1).grid(row=r, column=0, sticky="e", padx=(8,6), pady=3)
        ttk.Entry(paths, textvariable=v1).grid(row=r, column=1, sticky="ew", pady=3)
        ttk.Label(paths, text=l2).grid(row=r, column=2, sticky="e", padx=(12,6), pady=3)
        ttk.Entry(paths, textvariable=v2).grid(row=r, column=3, sticky="ew", padx=(0,8), pady=3)

    coco_eval = ttk.LabelFrame(tab_final_data, text="Official COCO verification (pycocotools)")
    coco_eval.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
    coco_eval.columnconfigure(1, weight=1); coco_eval.columnconfigure(3, weight=1)
    official_enabled_var = _bool_var(app, "var_official_coco_enabled", True)
    official_required_var = _bool_var(app, "var_official_coco_required", False)
    official_archive_var = _bool_var(app, "var_official_coco_archive_tensors", True)
    official_ann_var = _str_var(app, "var_official_coco_annotations", coco_ann_var.get())
    official_remote_ann_var = _str_var(app, "var_official_coco_remote_annotations", "/datasets/coco2017/annotations/instances_val2017.json")
    official_status_var = _str_var(app, "var_official_coco_status", "pycocotools: not checked")
    ttk.Checkbutton(coco_eval, text="Official COCO artifact erzeugen", variable=official_enabled_var).grid(row=0, column=0, sticky="w", padx=8, pady=6)
    ttk.Checkbutton(coco_eval, text="Für finalen Run zwingend", variable=official_required_var).grid(row=0, column=1, sticky="w", padx=6, pady=6)
    ttk.Checkbutton(coco_eval, text="Precision/Recall-Tensoren archivieren", variable=official_archive_var).grid(row=0, column=2, columnspan=2, sticky="w", padx=6, pady=6)
    ttk.Label(coco_eval, text="Local annotations:").grid(row=1, column=0, sticky="e", padx=(8,6), pady=4)
    ttk.Entry(coco_eval, textvariable=official_ann_var).grid(row=1, column=1, sticky="ew", pady=4)
    ttk.Button(coco_eval, text="JSON…", command=lambda: official_ann_var.set(filedialog.askopenfilename(parent=frame, filetypes=[("COCO JSON", "*.json")]) or official_ann_var.get())).grid(row=1, column=2, padx=6, pady=4)
    ttk.Label(coco_eval, text="Remote annotations:").grid(row=2, column=0, sticky="e", padx=(8,6), pady=4)
    ttk.Entry(coco_eval, textvariable=official_remote_ann_var).grid(row=2, column=1, columnspan=3, sticky="ew", padx=(0,8), pady=4)
    def _check_pycoco() -> None:
        st = pycocotools_status()
        official_status_var.set("pycocotools " + (str(st.get("version") or "available") if st.get("available") else "MISSING — install optional dataset dependency on host and benchmark target"))
    ttk.Button(coco_eval, text="Check pycocotools", command=_check_pycoco).grid(row=3, column=0, sticky="w", padx=8, pady=(4,8))
    ttk.Label(coco_eval, textvariable=official_status_var, foreground="#555", wraplength=900).grid(row=3, column=1, columnspan=3, sticky="ew", padx=6, pady=(4,8))
    ttk.Label(coco_eval, text="Die interne gepaarte Bootstrap-Metrik bleibt das Non-Inferiority-Gate. pycocotools erzeugt zusätzlich das offizielle COCOeval-Artefakt (Predictions, Parameter, Summary, Arrays, Hash-Manifest).", foreground="#666", wraplength=1100).grid(row=4, column=0, columnspan=4, sticky="ew", padx=8, pady=(0,8))
    _refresh_final_dataset_status()
    _check_pycoco()

    # ------------------------------------------------------------------
    # Activation-proxy and experimental DeepX split settings
    # ------------------------------------------------------------------
    proxy = ttk.LabelFrame(tab_validation, text="Activation proxy / Stage2 accelerator experiments")
    proxy.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
    for _c in range(8):
        try:
            proxy.columnconfigure(_c, weight=1 if _c in (1, 3, 5, 7) else 0)
        except Exception:
            pass

    proxy_backend_var = _str_var(app, "var_activation_proxy_backend", "cuda_ort")
    proxy_store_var = _str_var(app, "var_activation_proxy_store_samples", "0")
    _strict_default = str(os.environ.get("ONNX_SPLITPOINT_ACTIVATION_PROXY_STRICT") or os.environ.get("SPLITPOINT_ACTIVATION_PROXY_STRICT") or "").strip().lower() in {"1", "true", "yes", "on", "strict"}
    proxy_strict_var = _bool_var(app, "var_activation_proxy_strict", _strict_default)
    enable_deepx_split_var = _bool_var(app, "var_enable_deepx_split_plan", True)
    enable_deepx_part2_build_var = _bool_var(app, "var_enable_deepx_part2_experimental_build", False)

    ttk.Label(proxy, text="Proxy producer:").grid(row=0, column=0, sticky="e", padx=(8, 6), pady=(8, 4))
    ttk.Combobox(proxy, textvariable=proxy_backend_var, values=["cuda_ort", "ort_cpu", "tensorrt_ort", "remote_deepx_tensorrt", "remote_deepx_cuda"], state="readonly", width=18).grid(row=0, column=1, sticky="w", padx=(0, 12), pady=(8, 4))
    ttk.Label(proxy, text="Store NPZ samples:").grid(row=0, column=2, sticky="e", padx=(8, 6), pady=(8, 4))
    ttk.Entry(proxy, textvariable=proxy_store_var, width=8).grid(row=0, column=3, sticky="w", padx=(0, 12), pady=(8, 4))
    ttk.Checkbutton(proxy, text="Plan DeepX split rows", variable=enable_deepx_split_var).grid(row=0, column=4, columnspan=2, sticky="w", padx=(8, 12), pady=(8, 4))
    ttk.Checkbutton(proxy, text="Experimental DeepX Part2 DX-COM build", variable=enable_deepx_part2_build_var).grid(row=0, column=6, columnspan=2, sticky="w", padx=(8, 12), pady=(8, 4))
    ttk.Checkbutton(proxy, text="Strict proxy: fail instead of fallback", variable=proxy_strict_var).grid(row=1, column=6, columnspan=2, sticky="w", padx=(8, 12), pady=(4, 4))

    def _activation_proxy_selected_backend() -> str:
        raw = str(proxy_backend_var.get() or "cuda_ort").strip()
        key = raw.lower().replace(" ", "_").replace("-", "_")
        aliases = {
            "cpu": "ort_cpu", "cpu_ort": "ort_cpu", "ort_cpu": "ort_cpu",
            "cuda": "cuda_ort", "ort_cuda": "cuda_ort", "cuda_ort": "cuda_ort",
            "trt": "tensorrt_ort", "tensorrt": "tensorrt_ort", "ort_tensorrt": "tensorrt_ort", "tensorrt_ort": "tensorrt_ort",
            "remote": "remote_deepx_tensorrt", "remote_trt": "remote_deepx_tensorrt", "remote_tensorrt": "remote_deepx_tensorrt", "remote_deepx": "remote_deepx_tensorrt", "remote_deepx_trt": "remote_deepx_tensorrt", "remote_deepx_tensorrt": "remote_deepx_tensorrt",
            "remote_cuda": "remote_deepx_cuda", "remote_deepx_cuda": "remote_deepx_cuda",
        }
        return aliases.get(key, raw or "cuda_ort")

    def _run_activation_proxy_tool(*, install: bool = False, force_backend: str | None = None) -> None:
        backend = force_backend or _activation_proxy_selected_backend()
        try:
            proxy_backend_var.set(backend)
        except Exception:
            pass
        script = Path.cwd() / "scripts" / "check_activation_proxy_backend.py"
        if not script.exists():
            messagebox.showerror("Activation proxy", f"Script not found:\n{script}")
            return
        cmd = [sys.executable, str(script), "--backend", backend]
        if install:
            cmd.extend(["--install", "--replace"])
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, cwd=str(Path.cwd()))
        except Exception as exc:
            messagebox.showerror("Activation proxy", f"Failed to run check:\n{type(exc).__name__}: {exc}")
            return
        raw_text = (proc.stdout or "")
        if proc.stderr:
            raw_text += "\n\n[stderr]\n" + proc.stderr
        title = "Activation proxy install/check" if install else "Activation proxy check"
        text = raw_text
        try:
            import json as _json
            data = _json.loads((proc.stdout or "").strip().splitlines()[-1])
            providers = ", ".join(data.get("available_providers") or []) or "none"
            requested = data.get("requested_backend", backend)
            req = data.get("required_provider") or "CPUExecutionProvider"
            status = "OK" if data.get("ok") else "FAILED"
            lines = [
                f"{status}: {requested}",
                f"selected in GUI: {backend}",
                f"python: {data.get('python')}",
                f"venv: {data.get('actual_venv') or data.get('venv')}",
                f"required: {req}",
                f"providers: {providers}",
            ]
            smoke = data.get("session_smoke") or {}
            if isinstance(smoke, dict) and smoke:
                lines.append(f"session smoke: {smoke.get('ok')} providers={smoke.get('session_providers')}")
                if smoke.get("error"):
                    lines.append("session error: " + str(smoke.get("error")))
            if data.get("venv_warning"):
                lines.append("warning: " + str(data.get("venv_warning")))
            if data.get("install_note"):
                lines.append("note: " + str(data.get("install_note")))
            if data.get("reason"):
                lines.append("reason: " + str(data.get("reason")))
            lines.append("\n--- raw JSON ---")
            text = "\n".join(lines) + "\n" + _json.dumps(data, indent=2)
        except Exception:
            text = raw_text
        # Render a small custom dialog instead of a native messagebox.  Native
        # Tk messageboxes hide long text behind a generic "See details below"
        # section on some platforms, which made pass/fail checks hard to read.
        ok_result = proc.returncode == 0
        if len(text) > 20000:
            text = text[:20000] + "\n... output truncated ..."
        summary = (text or "OK").split("\n--- raw JSON ---", 1)[0].strip()
        try:
            dlg = tk.Toplevel(proxy)
            dlg.title(("SUCCESS — " if ok_result else "FAIL — ") + title)
            dlg.transient(proxy.winfo_toplevel())
            dlg.geometry("900x620")
            dlg.columnconfigure(0, weight=1)
            dlg.rowconfigure(2, weight=1)
            status_txt = "SUCCESS" if ok_result else "FAIL"
            status_fg = "#0a7d24" if ok_result else "#b00020"
            ttk.Label(dlg, text=status_txt, foreground=status_fg, font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))
            ttk.Label(dlg, text=summary or ("OK" if ok_result else f"Failed with rc={proc.returncode}"), wraplength=850, justify="left").grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
            nb = ttk.Notebook(dlg)
            nb.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 8))
            frm_details = ttk.Frame(nb)
            frm_details.columnconfigure(0, weight=1)
            frm_details.rowconfigure(0, weight=1)
            nb.add(frm_details, text="Details")
            txt = tk.Text(frm_details, wrap="none")
            txt.grid(row=0, column=0, sticky="nsew")
            ysb = ttk.Scrollbar(frm_details, orient="vertical", command=txt.yview)
            xsb = ttk.Scrollbar(frm_details, orient="horizontal", command=txt.xview)
            txt.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
            ysb.grid(row=0, column=1, sticky="ns")
            xsb.grid(row=1, column=0, sticky="ew")
            txt.insert("1.0", text or "")
            txt.configure(state="disabled")
            btns = ttk.Frame(dlg)
            btns.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
            def _copy_details() -> None:
                try:
                    dlg.clipboard_clear(); dlg.clipboard_append(text or "")
                except Exception:
                    pass
            ttk.Button(btns, text="Copy details", command=_copy_details).pack(side="left")
            ttk.Button(btns, text="OK", command=dlg.destroy).pack(side="right")
            try:
                dlg.grab_set()
            except Exception:
                pass
        except Exception:
            if ok_result:
                messagebox.showinfo("SUCCESS — " + title, summary or "OK")
            else:
                messagebox.showwarning("FAIL — " + title, summary or f"Failed with rc={proc.returncode}")

    btn_proxy_check = ttk.Button(proxy, text="Check selected proxy", command=lambda: _run_activation_proxy_tool(install=False))
    btn_proxy_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=(8, 8), pady=(4, 4))
    btn_proxy_install = ttk.Button(proxy, text="Install/repair selected", command=lambda: _run_activation_proxy_tool(install=True))
    btn_proxy_install.grid(row=1, column=2, columnspan=2, sticky="w", padx=(8, 8), pady=(4, 4))
    btn_proxy_check_cuda = ttk.Button(proxy, text="Check CUDA", command=lambda: _run_activation_proxy_tool(install=False, force_backend="cuda_ort"))
    btn_proxy_check_cuda.grid(row=1, column=4, sticky="w", padx=(8, 8), pady=(4, 4))
    btn_proxy_check_trt = ttk.Button(proxy, text="Check TensorRT", command=lambda: _run_activation_proxy_tool(install=False, force_backend="tensorrt_ort"))
    btn_proxy_check_trt.grid(row=1, column=5, sticky="w", padx=(8, 8), pady=(4, 4))
    ttk.Label(
        proxy,
        text=(
            "Default proxy=cuda_ort with automatic fallback to ORT CPU. Remote options run Part1 activation generation on the configured DeepX NX and copy NPZ tensors back automatically. Enable Strict proxy to abort if the requested proxy falls back to CPU. Use 'Check proxy backend' to verify the selected provider in this tool venv. "
            "Use 'Install/repair selected' to install onnxruntime-gpu for cuda_ort/tensorrt_ort or onnxruntime for ort_cpu. The check now performs a provider session smoke test, not only provider listing. "
            "The DeepX Part2 build uses NPZ activation proxy samples and is experimental; failed builds are recorded but do not break the full suite."
        ),
        foreground="#666",
        wraplength=1100,
        justify="left",
    ).grid(row=2, column=0, columnspan=8, sticky="ew", padx=8, pady=(4, 8))


    # ------------------------------------------------------------------
    if app is not None and hasattr(app, "_update_deepx_env_badge"):
        try:
            if str(os.environ.get("ONNX_SPLITPOINT_STARTUP_DEEPX_BADGE", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                app.after(8000, app._update_deepx_env_badge)
            else:
                try:
                    badge_dx.set(text="DeepX … (not checked)", level="idle")
                except Exception:
                    pass
                import logging as _logging
                _logging.getLogger(__name__).info("Startup DeepX env badge check disabled")
        except Exception:
            pass
    # Auto-refresh status when backend settings change.
    #
    # v58ag: keep this *manual by default*.  Opening the Tool Config / Hardware
    # tab should not silently start heavy DFC/ORT/CUDA probes.  On this project
    # machine these probes can emit long cuDNN errors or spend tens of seconds in
    # subprocesses, which looks like a GUI/workflow hang.  Users can still press
    # the Status/Refresh buttons explicitly, or opt into automatic refreshes.
    if app is not None and hasattr(app, "_hailo_schedule_status_refresh"):
        def _auto_status_enabled() -> bool:
            try:
                import os as _os
                return str(_os.environ.get("ONNX_SPLITPOINT_HARDWARE_AUTOPROBE", "0")).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                return False

        def _on_status_change(*_args) -> None:
            try:
                if bool(getattr(app, "_startup_status_refresh_suppressed", False)):
                    return
                if not _auto_status_enabled():
                    return
                app._hailo_schedule_status_refresh()
            except Exception:
                pass

        for v in (hailo_backend_var, hailo_wsl_distro_var, hailo_wsl_venv_var):
            try:
                v.trace_add("write", _on_status_change)
            except Exception:
                pass
        try:
            import logging as _logging
            _logging.getLogger(__name__).info(
                "Hardware tab Hailo auto-probe-on-setting-change is disabled "
                "(set ONNX_SPLITPOINT_HARDWARE_AUTOPROBE=1 to enable)."
            )
        except Exception:
            pass

    def _sync_link_from_interface(*_args) -> None:
        iface = iface_by_name.get(str(iface_var.get()))
        if not iface:
            return
        bw = iface.get("bandwidth_mb_s")
        ovh = iface.get("latency_overhead_ms")
        if bw is not None:
            _autofill_set(bw_unit_var, "lat_bw_unit", "MB/s")
            _autofill_set(bw_var, "lat_bw", str(bw))
        if ovh is not None:
            _autofill_set(overhead_var, "lat_overhead_ms", str(ovh))

        # Advanced link model defaults (optional)
        lm = iface.get("link_model") or {}
        if isinstance(lm, dict):
            lm_type = str(lm.get("type") or "").strip()
            if lm_type:
                _autofill_set(link_model_var, "lat_link_model", lm_type)

            e_pj_per_b = _to_float(lm.get("energy_pj_per_byte"))
            if e_pj_per_b is not None:
                _autofill_set(link_energy_var, "lat_E_link", _fmt_num(e_pj_per_b))

            mtu = lm.get("mtu_payload_bytes")
            if mtu is not None:
                try:
                    _autofill_set(mtu_var, "lat_mtu", str(int(mtu)))
                except Exception:
                    pass

            pkt_ms = _to_float(lm.get("per_packet_overhead_ms"))
            if pkt_ms is not None:
                _autofill_set(pkt_ovh_ms_var, "lat_pkt_ms", _fmt_num(pkt_ms))

            pkt_b = lm.get("per_packet_overhead_bytes")
            if pkt_b is not None:
                try:
                    _autofill_set(pkt_ovh_bytes_var, "lat_pkt_b", str(int(pkt_b)))
                except Exception:
                    pass

            cons = lm.get("constraints") or {}
            if isinstance(cons, dict):
                max_ms = _to_float(cons.get("max_latency_ms"))
                if max_ms is not None:
                    _autofill_set(link_max_ms_var, "lat_max_ms", _fmt_num(max_ms))

                max_mj = _to_float(cons.get("max_energy_mJ"))
                if max_mj is not None:
                    _autofill_set(link_max_mj_var, "lat_max_mj", _fmt_num(max_mj))

                max_bytes = cons.get("max_bytes")
                if max_bytes is not None:
                    try:
                        _autofill_set(link_max_bytes_var, "lat_max_bytes", str(int(max_bytes)))
                    except Exception:
                        pass


    iface_var.trace_add("write", _sync_link_from_interface)
    _sync_link_from_interface()

    # Fill GOPS / peak mem defaults once on startup (without clobbering user edits).
    try:
        _sync_latency_defaults()
    except Exception:
        pass


    # Latency plot should update when HW / link settings change.
    def _request_latency_recompute(*_args: object) -> None:
        try:
            fn = getattr(app, "_schedule_latency_recompute", None)
            if callable(fn):
                fn("hw_change")
        except Exception:
            pass

    for _v in (
        left_var,
        right_var,
        iface_var,
        bw_var,
        bw_unit_var,
        overhead_var,
        gops_l_var,
        gops_r_var,
        link_model_var,
        link_energy_var,
        mtu_var,
        pkt_ovh_ms_var,
        pkt_ovh_bytes_var,
        link_max_ms_var,
        link_max_mj_var,
        link_max_bytes_var,
    ):
        try:
            _v.trace_add("write", _request_latency_recompute)
        except Exception:
            pass

    return frame

# v60m: development profiles keep screening validation unless final validation
# was explicitly selected; calibration manifests may still auto-bind.
from onnx_splitpoint_tool.v60m_policy import install_dataset_binding_guards as _v60m_install_dataset_binding_guards
_v60m_install_dataset_binding_guards(globals())

# v60m: the top-level energy switch is authoritative for all native energy paths.
from onnx_splitpoint_tool.v60m_policy import install_energy_object_guards as _v60m_install_energy_guards
_v60m_install_energy_guards(globals())
