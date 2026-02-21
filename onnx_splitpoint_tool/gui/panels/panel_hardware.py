"""Hardware panel for accelerator, latency-model and Hailo configuration."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ... import api as asc
from ..widgets.tooltip import attach_tooltip
from ..widgets.status_badge import StatusBadge


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
    "hailo_max": "Maximale Anzahl Kandidaten, die mit Hailo geprüft werden.",
    "hailo_fixup": "Wendet ONNX-Fixups vor dem Hailo-Parse an.",
    "hailo_keep": "Behält temporäre Hailo-Artefakte für Debugging.",
    "hailo_target": "Welche Split-Seite mit Hailo geprüft wird.",
    "hailo_backend": "Backend-Auswahl: auto/local/wsl.",
    "hailo_wsl_distro": "Optionaler WSL-Distro-Name für Hailo-Backend.",
    "hailo_wsl_venv": "Aktivierungsbefehl für das WSL-Python-Venv mit Hailo SDK. Tipp: 'auto' nutzt die DFC-Profile (Hailo-8 vs Hailo-10) aus resources/hailo/profiles.json.",
    "hailo_status": "Zeigt, ob der Hailo DFC für Hailo-8/Hailo-10 erreichbar ist (automatisch beim Start + Refresh).",
    "hailo_refresh": "Aktualisiert die Hailo-Statusanzeige (Probe).",
    "hailo_clear": "Leert den lokalen Hailo-Parse-Cache.",
    "hailo_provision": (
        "Installiert/Repariert die verwalteten Hailo-DFC-Umgebungen (Hailo-8/Hailo-10). "
        "Erstellt bei Bedarf die venvs unter ~/.onnx_splitpoint_tool/hailo/ neu und pinnt bekannte Paketversionen. "
        "Ausgabe/Fehler landen in gui.log (Logs-Tab)."
    ),
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


def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)

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

    accel = ttk.LabelFrame(frame, text="Accelerators")
    accel.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))

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

    latency = ttk.LabelFrame(frame, text="Latency model")
    latency.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

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

    hailo = ttk.LabelFrame(frame, text="Hailo feasibility check")
    hailo.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 12))

    hailo_check_var = _bool_var(app, "var_hailo_check", False)
    hailo_hw_var = _str_var(app, "var_hailo_hw_arch", "hailo8")
    hailo_max_var = _str_var(app, "var_hailo_max_checks", "25")
    hailo_fixup_var = _bool_var(app, "var_hailo_fixup", True)
    hailo_keep_var = _bool_var(app, "var_hailo_keep", False)
    hailo_target_var = _str_var(app, "var_hailo_target", "either")
    hailo_backend_var = _str_var(app, "var_hailo_backend", "auto")
    hailo_wsl_distro_var = _str_var(app, "var_hailo_wsl_distro", "")
    hailo_wsl_venv_var = _str_var(app, "var_hailo_wsl_venv", "auto")

    chk_hailo = ttk.Checkbutton(hailo, text="Enable parse-check in ranking", variable=hailo_check_var)
    chk_hailo.grid(row=0, column=0, sticky="w", padx=(8, 8), pady=8)
    attach_tooltip(chk_hailo, _tt("hailo_enable"))

    ttk.Label(hailo, text="HW arch:").grid(row=0, column=1, sticky="e", pady=8)
    cb_hailo_hw = ttk.Combobox(
        hailo,
        textvariable=hailo_hw_var,
        values=["hailo8", "hailo8l", "hailo8r", "hailo10", "hailo10h"],
        width=10,
        state="normal",
    )
    cb_hailo_hw.grid(row=0, column=2, sticky="w", padx=(4, 12), pady=8)
    attach_tooltip(cb_hailo_hw, _tt("hailo_hw"))

    ttk.Label(hailo, text="Max checks:").grid(row=0, column=3, sticky="e", pady=8)
    ent_hailo_max = ttk.Entry(hailo, textvariable=hailo_max_var, width=8)
    ent_hailo_max.grid(row=0, column=4, sticky="w", padx=(4, 12), pady=8)
    attach_tooltip(ent_hailo_max, _tt("hailo_max"))

    chk_fixup = ttk.Checkbutton(hailo, text="ONNX fixup", variable=hailo_fixup_var)
    chk_fixup.grid(row=0, column=5, sticky="w", pady=8)
    attach_tooltip(chk_fixup, _tt("hailo_fixup"))

    chk_keep = ttk.Checkbutton(hailo, text="Keep artifacts", variable=hailo_keep_var)
    chk_keep.grid(row=0, column=6, sticky="w", padx=(10, 0), pady=8)
    attach_tooltip(chk_keep, _tt("hailo_keep"))

    ttk.Label(hailo, text="Target:").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
    cb_hailo_target = ttk.Combobox(hailo, textvariable=hailo_target_var, values=["either", "part2", "part1"], width=10, state="readonly")
    cb_hailo_target.grid(row=1, column=1, sticky="w", pady=(0, 8))
    attach_tooltip(cb_hailo_target, _tt("hailo_target"))

    ttk.Label(hailo, text="Backend:").grid(row=1, column=2, sticky="e", pady=(0, 8))
    cb_hailo_backend = ttk.Combobox(hailo, textvariable=hailo_backend_var, values=["auto", "local", "wsl"], width=10, state="readonly")
    cb_hailo_backend.grid(row=1, column=3, sticky="w", padx=(4, 12), pady=(0, 8))
    attach_tooltip(cb_hailo_backend, _tt("hailo_backend"))

    ttk.Label(hailo, text="WSL distro:").grid(row=1, column=4, sticky="e", pady=(0, 8))
    ent_hailo_distro = ttk.Entry(hailo, textvariable=hailo_wsl_distro_var, width=18)
    ent_hailo_distro.grid(row=1, column=5, sticky="w", pady=(0, 8))
    attach_tooltip(ent_hailo_distro, _tt("hailo_wsl_distro"))

    ttk.Label(hailo, text="WSL venv override:").grid(row=2, column=0, sticky="w", padx=(8, 4), pady=(0, 8))
    ent_hailo_venv = ttk.Entry(hailo, textvariable=hailo_wsl_venv_var, width=56)
    ent_hailo_venv.grid(row=2, column=1, columnspan=4, sticky="w", pady=(0, 8))
    attach_tooltip(ent_hailo_venv, _tt("hailo_wsl_venv"))

    # Status badges (auto-detected on startup).
    status_row = ttk.Frame(hailo)
    status_row.grid(row=3, column=0, columnspan=6, sticky="w", padx=(8, 8), pady=(0, 8))
    ttk.Label(status_row, text="DFC status:").pack(side=tk.LEFT)

    badge_h8 = StatusBadge(status_row, text="Hailo-8 …", level="idle")
    badge_h8.pack(side=tk.LEFT, padx=(6, 6))
    badge_h10 = StatusBadge(status_row, text="Hailo-10 …", level="idle")
    badge_h10.pack(side=tk.LEFT, padx=(0, 6))

    # Clicking a badge shows probe details (last output tail), which helps a lot
    # when Hailo-10 fails due to missing wheel or dependency drift.
    if app is not None and hasattr(app, "_hailo_show_probe_details"):
        try:
            badge_h8.bind("<Button-1>", lambda _e: app._hailo_show_probe_details("hailo8"))
            badge_h10.bind("<Button-1>", lambda _e: app._hailo_show_probe_details("hailo10"))
        except Exception:
            pass

    status_details_var = _str_var(app, "var_hailo_status_details", "")
    lbl_details = ttk.Label(status_row, textvariable=status_details_var)
    lbl_details.pack(side=tk.LEFT, padx=(8, 0))
    attach_tooltip(status_row, _tt("hailo_status"))

    if app is not None:
        # Expose widgets to the app so the probe worker can update them.
        setattr(app, "hailo_badge_h8", badge_h8)
        setattr(app, "hailo_badge_h10", badge_h10)

    btn_col = ttk.Frame(hailo)
    btn_col.grid(row=1, column=6, rowspan=2, sticky="ne", padx=(12, 8), pady=(0, 8))
    if app is not None and hasattr(app, "_hailo_refresh_status"):
        btn_ref = ttk.Button(btn_col, text="Refresh status", command=app._hailo_refresh_status)
        btn_ref.pack(side=tk.TOP, fill=tk.X)
        attach_tooltip(btn_ref, _tt("hailo_refresh"))
    if app is not None and hasattr(app, "_hailo_clear_cache"):
        btn_clear = ttk.Button(btn_col, text="Clear cache", command=app._hailo_clear_cache)
        btn_clear.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        attach_tooltip(btn_clear, _tt("hailo_clear"))

    if app is not None and hasattr(app, "_hailo_provision_dfcs"):
        btn_prov = ttk.Button(btn_col, text="Provision DFC", command=app._hailo_provision_dfcs)
        btn_prov.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        attach_tooltip(btn_prov, _tt("hailo_provision"))
        try:
            setattr(app, "_hailo_btn_provision", btn_prov)
        except Exception:
            pass

    # Auto-refresh status when backend settings change.
    if app is not None and hasattr(app, "_hailo_schedule_status_refresh"):
        def _on_status_change(*_args) -> None:
            try:
                app._hailo_schedule_status_refresh()
            except Exception:
                pass

        for v in (hailo_backend_var, hailo_wsl_distro_var, hailo_wsl_venv_var):
            try:
                v.trace_add("write", _on_status_change)
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
