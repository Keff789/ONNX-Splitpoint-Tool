"""Evaluation Workflow tab.

v60n keeps the run surface intentionally small.  An Evaluation Profile stores
models, candidate selection, logical hardware run profiles and the Native/Energy
switches.  The selected central run mode (Smoke, Standard or Final) resolves the
detailed build, validation, reporting, hold-out and reproducibility policy.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Mapping

from ...benchmark.evaluation_profiles import list_available_evaluation_profiles, load_evaluation_profile
from ...run_modes import infer_run_mode, profile_build_summary, run_mode_profile_brief
from ...execution_plan import build_effective_execution_plan, execution_plan_text
from ...measurement_configuration import measurement_configuration, measurement_configuration_lines
from ...workflow.start_snapshot import resolve_runtime_profile_start_snapshot
from ..profile_editor import open_evaluation_profile_editor
from ..widgets.tooltip import attach_tooltip


def _str_var(app: Any | None, name: str, default: str) -> tk.StringVar:
    if app is None:
        return tk.StringVar(value=default)
    existing = getattr(app, name, None)
    if isinstance(existing, tk.StringVar):
        return existing
    created = tk.StringVar(value=default)
    setattr(app, name, created)
    return created


def _available_profiles() -> list[str]:
    try:
        return list(list_available_evaluation_profiles())
    except Exception:
        return []


def _browse_profile(var: tk.StringVar) -> None:
    path = filedialog.askopenfilename(
        title="Evaluation Profile YAML wählen",
        filetypes=[("YAML", "*.yaml *.yml"), ("All files", "*.*")],
    )
    if path:
        var.set(path)


def _browse_dir(var: tk.StringVar, title: str) -> None:
    path = filedialog.askdirectory(title=title)
    if path:
        var.set(path)


def _tip(widget: tk.Widget, text: str) -> tk.Widget:
    try:
        attach_tooltip(widget, text, delay_ms=350, wraplength=480)
    except Exception:
        pass
    return widget


def _label(parent: tk.Misc, text: str, tooltip: str = "") -> ttk.Label:
    lbl = ttk.Label(parent, text=text)
    if tooltip:
        _tip(lbl, tooltip)
    return lbl


def _button(parent: tk.Misc, text: str, command: Callable[[], Any] | None, tooltip: str = "") -> ttk.Button:
    btn = ttk.Button(parent, text=text, command=command if callable(command) else (lambda: None))
    if tooltip:
        _tip(btn, tooltip)
    return btn


def _entry(parent: tk.Misc, variable: tk.StringVar, tooltip: str = "", **kw: Any) -> ttk.Entry:
    ent = ttk.Entry(parent, textvariable=variable, **kw)
    if tooltip:
        _tip(ent, tooltip)
    return ent


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return default
    return text in {"1", "true", "yes", "y", "on"}


def score_independent_audit_start_summary(
    plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the compact, testable GUI warning contract for an audit run.

    ``Cases/model`` is only the deployment shortlist.  A score-independent
    audit expands independently and can therefore turn an apparently small
    profile into tens of candidates.  Keep that distinction in one helper so
    the visible summary and the start confirmation cannot drift apart.
    """

    value = dict(plan or {}) if isinstance(plan, Mapping) else {}
    counts = (
        dict(value.get("score_independent_audit_counts") or {})
        if isinstance(value.get("score_independent_audit_counts"), Mapping)
        else {}
    )
    if not counts:
        return {}

    normalized_counts = {
        str(model_id): max(0, int(count or 0))
        for model_id, count in counts.items()
        if str(model_id).strip()
    }
    if not normalized_counts:
        return {}

    candidate_min = max(
        0, int(value.get("execution_union_candidate_count_min_total") or 0)
    )
    candidate_max = max(
        candidate_min,
        int(value.get("execution_union_candidate_count_upper_bound_total") or 0),
    )
    rows_min = max(
        0, int(value.get("expected_generic_result_rows_min_total") or 0)
    )
    rows_max = max(
        rows_min,
        int(
            value.get("expected_generic_result_rows_total")
            or value.get("generic_rows_total")
            or 0
        ),
    )
    counts_text = ", ".join(
        f"{model_id}={count}"
        for model_id, count in normalized_counts.items()
    )
    candidates_text = (
        str(candidate_max)
        if candidate_min == candidate_max
        else f"{candidate_min}–{candidate_max}"
    )
    rows_text = str(rows_max) if rows_min == rows_max else f"{rows_min}–{rows_max}"
    native_enabled = bool(value.get("native_enabled"))
    native_energy_enabled = bool(value.get("native_energy_enabled"))
    concise = (
        "Score-independent Ranking-Audit aktiv: "
        f"{counts_text} Kandidaten/Modell; Ausführungsunion "
        f"{candidates_text} Kandidaten gesamt; Generic-Zeilen "
        f"{rows_text}; Native={'an' if native_enabled else 'aus'}; "
        f"Native Energy={'an' if native_energy_enabled else 'aus'}."
    )
    return {
        "schema": "onnx-splitpoint/gui-audit-start-summary",
        "schema_version": 1,
        "confirmation_required": True,
        "audit_counts": normalized_counts,
        "execution_union_candidate_count_min_total": candidate_min,
        "execution_union_candidate_count_upper_bound_total": candidate_max,
        "expected_generic_result_rows_min_total": rows_min,
        "expected_generic_result_rows_total": rows_max,
        "native_enabled": native_enabled,
        "native_energy_enabled": native_energy_enabled,
        "concise": concise,
        "confirmation_text": (
            concise
            + "\n\nDer Audit wird unabhängig von „Cases/model“ ausgeführt. "
            "Das ist kein kleiner Standardlauf. Soll dieser neue Run wirklich "
            "gestartet werden?"
        ),
    }


def validate_productive_build_start(
    options: Any, *, parent: Any,
    preview: Callable[[Any], Mapping[str, Any]] | None = None,
) -> bool:
    """Reject legacy Force before offering any obsolete confirmation UI."""
    from ...build_dispatch_policy import require_productive_force_off
    options.force_build_confirmed_backends = ()
    options.force_build_confirmation_source = ""
    try:
        if bool(getattr(options, "resume", False)):
            if preview is None:
                from ...workflow.runner import EvaluationWorkflowRunner
                preview = lambda opts: EvaluationWorkflowRunner(opts).preview_force_build_start()
            shown = dict(preview(options))
            profile = shown.get("resolved_profile") or {}
            if not profile and shown.get("backends"):
                raise ValueError("productive_force_build_disabled: Force im archivierten Resume-Profil; neues Profil mit Force AUS verwenden.")
        else:
            profile = (getattr(options, "profile_start_snapshot", {}) or {}).get("resolved_profile") or {}
        require_productive_force_off(profile, hailo_force_build=getattr(options, "hailo_force_build", False))
    except ValueError as exc:
        messagebox.showerror("Buildpolitik: Force AUS", str(exc), parent=parent)
        return False
    return True


def confirm_force_build_start(
    options: Any,
    *,
    parent: Any,
    preview: Callable[[Any], Mapping[str, Any]] | None = None,
) -> bool:
    """Ask once per start, including an archived resume; never persist consent.

    The runner repeats admission against its verified effective profile. This
    UI decision cannot admit a backend added after this preview.
    """
    from ...force_build_admission import force_build_backends

    options.force_build_confirmed_backends = ()
    options.force_build_confirmation_source = ""
    resume = bool(getattr(options, "resume", False))
    mismatch_notice = ""
    if resume:
        if preview is None:
            from ...workflow.runner import EvaluationWorkflowRunner

            preview = lambda opts: EvaluationWorkflowRunner(opts).preview_force_build_start()
        resolved = dict(preview(options))
        backends = tuple(resolved.get("backends") or ())
        source = str(resolved.get("profile_source") or "gebundener Profilsnapshot")
        path = str(resolved.get("profile_path") or "")
        if path:
            source += " / " + path
        if resolved.get("profile_mismatch"):
            mismatch_notice = (
                "\n\nDas aktuelle Profil weicht vom archivierten Vertrag ab. "
                "Die Resume-Prüfung bleibt aktiv; diese Zustimmung ändert keinen Vertrag. "
                "Für geänderte Einstellungen einen neuen Run verwenden."
            )
    else:
        profile = dict(
            (getattr(options, "profile_start_snapshot", {}) or {}).get("resolved_profile")
            or {}
        )
        backends = force_build_backends(
            profile,
            hailo_force_build=getattr(options, "hailo_force_build", False),
        )
        source = str(profile_build_summary(profile)["values_source"])
    if not backends:
        return True
    labels = {"hailo": "Hailo", "deepx": "DeepX"}
    names = ", ".join(labels.get(name, name) for name in backends)
    action = "Resume mit dem gebundenen Profilsnapshot" if resume else "Neuer Workflow-Start"
    accepted = messagebox.askyesno(
        "Force-Neubau bewusst starten?",
        f"{action}\n\nForce AN: {names}\nWertequelle: {source}\n\n"
        "Kompatible Cachetreffer dieser Backends werden bewusst übergangen. "
        "Dies kann vorhandene Artefakte erneut kompilieren."
        + mismatch_notice
        + "\n\nForce nur für diesen Start bestätigen? Das archivierte Profil bleibt unverändert.",
        parent=parent,
        default=messagebox.NO,
        icon=messagebox.WARNING,
    )
    if not accepted:
        return False
    options.force_build_confirmed_backends = tuple(backends)
    options.force_build_confirmation_source = "gui_confirmation"
    return True


def _profile_summary_payload(
    request: str,
) -> tuple[list[str], dict[str, str]]:
    request = str(request or "").strip()
    if not request:
        return ["Kein Evaluation Profile ausgewählt."], {}
    try:
        loaded = load_evaluation_profile(request, validate=True)
        if loaded is None or isinstance(loaded, tuple):
            raise ValueError(f"Profile not found: {request}")
        raw = dict(getattr(loaded, "raw_profile", {}) or {})
        profile_id = str(getattr(loaded, "profile_id", "") or raw.get("name") or request)
        profile_path = str(getattr(loaded, "profile_path", "") or request)
        raw, start_snapshot = resolve_runtime_profile_start_snapshot(
            profile_request=request,
            source_profile=dict(getattr(loaded, "source_profile", {}) or raw),
            resolved_profile=raw,
            profile_id=profile_id,
            profile_path=profile_path,
            profile_source=str(getattr(loaded, "source", "") or "file"),
        )
        build_summary = profile_build_summary(raw)
        visible_snapshot = {
            "profile_request": request,
            "selection_fingerprint": str(
                start_snapshot.get("selection_fingerprint") or ""
            ),
            "source_profile_sha256": str(
                start_snapshot.get("source_profile_sha256") or ""
            ),
            "resolved_execution_sha256": str(
                start_snapshot.get("resolved_execution_sha256") or ""
            ),
            "snapshot_sha256": str(
                start_snapshot.get("snapshot_sha256") or ""
            ),
        }
    except Exception as exc:
        return [
            "Profil konnte nicht geladen/validiert werden: "
            f"{type(exc).__name__}: {exc}"
        ], {}

    suite = dict(raw.get("model_suite") or {})
    primary = [
        x for x in list(suite.get("primary") or [])
        if isinstance(x, Mapping) and _as_bool(x.get("enabled"), True)
    ]
    run_profiles = [
        x for x in list(raw.get("run_profiles") or [])
        if isinstance(x, Mapping) and _as_bool(x.get("enabled"), True)
    ]
    selection = dict(raw.get("selection_policy") or {})
    preset = dict(raw.get("execution_preset") or {})
    snapshot = dict(preset.get("snapshot") or {})
    overrides = dict(preset.get("overrides") or {})
    mode_id = str(preset.get("id") or infer_run_mode(raw))

    model_names = ", ".join(
        f"{str(x.get('id') or '?')} ({str(x.get('task') or 'auto')}, {str(x.get('evaluation_role') or 'development')})"
        for x in primary[:8]
    )
    if len(primary) > 8:
        model_names += f", … (+{len(primary) - 8})"
    target_names = ", ".join(str(x.get("id") or x.get("full") or "?") for x in run_profiles[:12])
    if len(run_profiles) > 12:
        target_names += f", … (+{len(run_profiles) - 12})"

    data_cfg = dict(snapshot.get("data") or {})
    calib_cfg = dict(data_cfg.get("calibration_items") or {})
    validation_cfg = dict(data_cfg.get("validation_items") or {})
    quality_cfg = dict(snapshot.get("quality") or {})
    hailo_cfg = dict((snapshot.get("build") or {}).get("hailo") or {}) if isinstance(snapshot.get("build"), Mapping) else {}
    runtime_cfg = dict(snapshot.get("runtime") or {})
    benchmark_cfg = dict(runtime_cfg.get("benchmark") or {})
    native_cfg = dict(runtime_cfg.get("native") or {})
    repro_cfg = dict(snapshot.get("reproducibility") or {})
    final_campaign = str((snapshot.get("campaign") or {}).get("mode") or "").lower() == "final" if isinstance(snapshot.get("campaign"), Mapping) else False

    def _items_text(value: Any) -> str:
        try:
            number = int(value)
        except Exception:
            return "?"
        return "full" if number == 0 else str(number)

    _effective_plan: dict[str, Any] = {}
    try:
        _effective_plan = build_effective_execution_plan(raw)
        _effective_plan_text = execution_plan_text(_effective_plan)
    except Exception as _plan_exc:
        _effective_plan_text = f"Execution plan unavailable: {type(_plan_exc).__name__}: {_plan_exc}"

    lines = [
        f"Profile: {profile_id}",
        f"Run mode: {run_mode_profile_brief(raw).splitlines()[0]}",
        f"Buildpolitik: {build_summary['build_policy']}",
        f"Hailo Force: {build_summary['hailo_force_text']}",
        f"DeepX Force: {build_summary['deepx_force_text']}",
        f"Wertequelle: {build_summary['values_source']}",
        f"Hailo-Integrität: {build_summary['hailo_cache_integrity']}",
        f"DeepX Classification: {build_summary['deepx_classification_preprocessing']} (Quelle: {build_summary['deepx_classification_source']})",
        build_summary["hailo_compute_text"],
        f"Native Force: {build_summary['native_force_text']} · Hailo-Cache: {'AN' if build_summary['hailo_cache_enabled'] else 'AUS'} · Artifact Store: {'AN' if build_summary['artifact_store_enabled'] else 'AUS'}",
        f"Models: {len(primary)}" + (f" — {model_names}" if model_names else ""),
        f"Cases / selection: {selection.get('max_accepted_cases_per_model', 1)} per model · shortlist={selection.get('preferred_shortlist', 1)} · strategy={selection.get('selection_strategy', 'stratified_windows')} · min gap={selection.get('min_gap', 1)} · pool={selection.get('candidate_search_pool', 'auto')} · Part-2 inputs=1={'on' if selection.get('require_single_part2_input', False) else 'off'}",
        f"Logical hardware profiles: {len(run_profiles)}" + (f" — {target_names}" if target_names else ""),
        f"Effective effort: calibration CLS/DET={_items_text(calib_cfg.get('classification'))}/{_items_text(calib_cfg.get('detection'))} · validation CLS/DET={_items_text(validation_cfg.get('classification'))}/{_items_text(validation_cfg.get('detection'))} · bootstrap={quality_cfg.get('bootstrap_repetitions', '?')}",
        f"Build / timing: Hailo preset={hailo_cfg.get('preset', mode_id)} opt={hailo_cfg.get('optimization_level', '?')} · benchmark warmup/runs={benchmark_cfg.get('warmup', '?')}/{benchmark_cfg.get('runs', '?')} · native frames/warmup={native_cfg.get('frames', '?')}/{native_cfg.get('warmup', '?')}",
        f"Reproducibility: {repro_cfg.get('level', 'relaxed')} · Campaign: {'final' if final_campaign else 'development'} (effektive Policy; kein impliziter Wechsel durch den Modusnamen)",
        f"Hardware resolution: logical profiles are mapped automatically through Tool Config → Hardware run profiles.",
        f"Profile YAML: {profile_path}",
    ]

    lines.extend(measurement_configuration_lines(measurement_configuration(raw, _effective_plan)))
    audit_summary = score_independent_audit_start_summary(_effective_plan)
    if audit_summary:
        lines.extend(["", "⚠ " + str(audit_summary.get("concise") or "")])
    lines.extend(["", "Effective execution plan:", _effective_plan_text])

    warnings: list[str] = []
    if not primary:
        warnings.append("⚠ Kein Modell im Profil.")
    if not run_profiles:
        warnings.append("⚠ Kein logisches Hardware-Run-Profil ausgewählt.")
    if bool(overrides.get("energy_enabled")) and not bool(overrides.get("native_enabled")):
        warnings.append("Hinweis: Energy ist aktiv; Native Energy bleibt für deaktivierte Native Runner automatisch aus.")
    if not bool(overrides.get("energy_enabled")):
        warnings.append("✓ Native Energy ist aus; Generic-Runner-Energie ist in EvaluationRuns grundsätzlich deaktiviert.")
    if warnings:
        lines.append("")
        lines.extend(warnings)
    lines.append("")
    lines.append(
        "Die Wertequelle oben benennt die tatsächlich verwendete Moduskonfiguration. Explizite Profilwerte wie "
        "DeepX Classification bleiben erhalten; jeder Run archiviert sein effektives Profil als Snapshot."
    )
    return lines, visible_snapshot


def _profile_summary_lines(app: Any | None, request: str) -> list[str]:
    """Compatibility wrapper without mutating the start-guard snapshot."""

    lines, _snapshot = _profile_summary_payload(request)
    return lines


def _render_profile_summary(
    widget: Any,
    text: str,
) -> bool:
    """Render one summary atomically from the start guard's perspective."""

    try:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        return True
    except Exception:
        try:
            widget.configure(state="disabled")
        except Exception:
            pass
        return False


def _commit_profile_summary(
    app: Any | None,
    widget: Any,
    text: str,
    visible_snapshot: Mapping[str, Any],
) -> bool:
    """Bind the start-guard hash only after the matching text is visible."""

    rendered = _render_profile_summary(widget, text)
    if app is not None:
        app._evaluation_workflow_visible_start_snapshot = (
            dict(visible_snapshot) if rendered else {}
        )
    return rendered


def build_panel(parent: tk.Misc, app: Any | None = None) -> ttk.Frame:
    outer = ttk.Frame(parent)
    outer.columnconfigure(0, weight=1)
    # Keep only the compact log/status box expandable.  Older builds expanded
    # the status row inside the action frame, leaving a large empty area.
    outer.rowconfigure(3, weight=1)

    try:
        default_out = str(Path(getattr(app, "default_output_dir")).expanduser() / "EvaluationRuns") if app is not None and getattr(app, "default_output_dir", None) else "EvaluationRuns"
    except Exception:
        default_out = "EvaluationRuns"
    profiles = _available_profiles()
    # v59dx: make the current thesis native smoke profile the first-click GUI default
    # when it is available.  Users can still choose smoke_regression_v1/final/etc.
    # explicitly from the combobox.
    if "native_resnet_yolo26s_hailo8_smoke_v1" in profiles:
        default_profile = "native_resnet_yolo26s_hailo8_smoke_v1"
    elif "smoke_regression_v1" in profiles:
        default_profile = "smoke_regression_v1"
    else:
        default_profile = profiles[0] if profiles else ""

    var_profile = _str_var(app, "var_eval_workflow_profile", default_profile)
    var_out_root = _str_var(app, "var_eval_workflow_out_root", default_out)
    var_status = _str_var(app, "var_eval_workflow_status", "Bereit. Profil wählen oder erstellen, dann Workflow starten.")
    var_last_run = _str_var(app, "var_eval_workflow_last_run_dir", "")

    intro = ttk.Label(
        outer,
        text=(
            "Geführter Evaluations-/Dissertationsworkflow: Run-Modus wählen, Modelle/Kandidaten/Hardwareprofile festlegen, "
            "Native und Energy bei Bedarf aktivieren und starten. Der Modus liefert Build- und Prüfdefaults; "
            "explizite Profilwerte wie DeepX Classification bleiben maßgeblich."
        ),
        wraplength=1120,
        justify="left",
    )
    intro.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
    _tip(intro, "Das ist der Hauptpfad für komplette Evaluationsläufe: Analyse → Prediction → Benchmarkset → Benchmarks → Validation → Reports.")

    profile_box = ttk.LabelFrame(outer, text="1. Evaluation Profile")
    profile_box.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
    profile_box.columnconfigure(1, weight=1)

    profile_tip = "Evaluation Profile: Modelle, Fälle/Selection, Hardwareprofile, Native/Energy und explizite DeepX Classification. Die Summary zeigt Force und die tatsächlich maßgebliche Modusquelle."
    _label(profile_box, "Profile YAML:", profile_tip).grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
    cbo = ttk.Combobox(profile_box, textvariable=var_profile, values=profiles, state="normal")
    cbo.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=8)
    _tip(cbo, profile_tip)

    summary_box = ttk.LabelFrame(outer, text="2. Effective run summary")
    summary_box.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
    summary_box.columnconfigure(0, weight=1)
    profile_summary = tk.Text(summary_box, height=12, wrap="word")
    profile_summary.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
    profile_summary.configure(state="disabled")
    _tip(profile_summary, "Kontrolliere hier Run-Modus, Modelle, Kandidaten, logische Hardwareprofile sowie Native/Energy. Tiefe Details werden zentral verwaltet.")

    def refresh_summary(*_args: Any) -> None:
        lines, visible_snapshot = _profile_summary_payload(var_profile.get())
        _commit_profile_summary(
            app,
            profile_summary,
            "\n".join(lines) + "\n",
            visible_snapshot,
        )

    def _refresh_profile_values(extra: str = "") -> None:
        vals = _available_profiles()
        if extra and extra not in vals:
            vals.insert(0, extra)
        try:
            cbo.configure(values=vals)
        except Exception:
            pass
        refresh_summary()

    def _open_editor() -> None:
        open_evaluation_profile_editor(
            outer,
            app=app,
            profile_var=var_profile,
            profile_combo=cbo,
            models_root_var=None,
        )
        _refresh_profile_values(str(var_profile.get() or ""))
        try:
            var_status.set("profile editor opened")
        except Exception:
            pass

    _button(profile_box, "YAML…", lambda: (_browse_profile(var_profile), refresh_summary()), "Vorhandene Evaluation-Profile-YAML auswählen.").grid(row=0, column=2, sticky="w", padx=(0, 8), pady=8)
    _button(profile_box, "Profil erstellen/bearbeiten…", _open_editor, "Öffnet den vereinfachten Profile Builder. Run-Modus-Details werden zentral in Tool Config gepflegt.").grid(row=0, column=3, sticky="w", padx=(0, 8), pady=8)
    _button(profile_box, "Summary aktualisieren", refresh_summary, "YAML erneut laden und die Zusammenfassung aktualisieren.").grid(row=0, column=4, sticky="w", padx=(0, 8), pady=8)

    out_tip = "Übergeordneter Ordner. Darunter erzeugt der Workflow pro Lauf automatisch ein Results Bundle <profile>_<timestamp>."
    _label(profile_box, "Results parent folder:", out_tip).grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    _entry(profile_box, var_out_root, out_tip).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(0, 8), pady=(0, 8))
    _button(profile_box, "Folder…", lambda: _browse_dir(var_out_root, "EvaluationRuns-Ausgabeordner wählen"), "Übergeordneten Ordner für EvaluationRuns-Bundles auswählen.").grid(row=1, column=4, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Label(profile_box, text="Run ID wird automatisch erzeugt. Tiefe Build-, Quality-, Hold-out-, Reporting- und Reproducibility-Einstellungen liegen zentral im gewählten Run-Modus.", foreground="#666", wraplength=980, justify="left").grid(row=2, column=1, columnspan=4, sticky="ew", padx=(0, 8), pady=(0, 8))

    action_box = ttk.LabelFrame(outer, text="3. Start / Results Bundle")
    action_box.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 8))
    action_box.columnconfigure(0, weight=1)
    action_box.rowconfigure(2, weight=0)
    action_box.rowconfigure(3, weight=1)
    buttons = ttk.Frame(action_box)
    buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

    _button(buttons, "Start Evaluation Workflow", lambda: getattr(app, "_queue_evaluation_workflow", lambda **_: None)(resume=False), "Startet einen neuen Run. Der Run-Ordner wird automatisch benannt.").pack(side=tk.LEFT)
    _button(buttons, "Resume latest", lambda: getattr(app, "_queue_evaluation_workflow", lambda **_: None)(resume=True), "Setzt den neuesten passenden Run dieses Profils fort.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Rerun generated", getattr(app, "_queue_evaluation_workflow_rerun_generated", None), "Führt vorhandene Benchmarksets mit aktualisierten Runnern, Remote-Bundles, Validation und Reports erneut aus; Generate/Build-Stufen werden hart wiederverwendet und nicht neu gebaut.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Finalize partial", getattr(app, "_queue_evaluation_workflow_finalize_partial", None), "Erzeugt Validation/Aggregation/Reports aus bereits abgeschlossenen Modellordnern, ohne neue Benchmarks/Energy zu starten. Nützlich nach Abbruch eines späten YOLO/Energy-Jobs.").pack(side=tk.LEFT, padx=(8, 18))
    _button(buttons, "Open Results Bundle", getattr(app, "_evaluation_workflow_open_results_folder", None), "Öffnet den Ordner des letzten Results Bundles. Dort liegen Manifest, Stufenartefakte und Reports.").pack(side=tk.LEFT)
    _button(buttons, "Open Manifest", getattr(app, "_evaluation_workflow_open_manifest", None), "Öffnet run_manifest.json des letzten Results Bundles.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Open Reports", getattr(app, "_evaluation_workflow_open_reports", None), "Öffnet den reports/-Ordner des letzten Results Bundles.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Open Dashboard", getattr(app, "_evaluation_workflow_open_dashboard", None), "Öffnet reports/result_dashboard.md mit Modellstatus, gemessenen Splits, Hailo-Evidenz und Thesis-Metriken.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Open Benchmark Suite", getattr(app, "_evaluation_workflow_open_benchmark_suite", None), "Öffnet die legacy_suite/ des vorhandenen BenchmarkSet-Generators im letzten Results Bundle. Alte generated_suite/-Ordner werden im normalen Workflow nicht mehr verwendet.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Open Debug Log", getattr(app, "_evaluation_workflow_open_debug_log", None), "Öffnet evaluation_workflow.log aus dem letzten Results Bundle bzw. _latest_evaluation_workflow.log aus dem Parent-Ordner.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Debug Pack", getattr(app, "_evaluation_workflow_prepare_debug_pack", None), "Erzeugt ein kleines ZIP mit Log, Manifest und Summary-Reports zum Hochladen/Debuggen.").pack(side=tk.LEFT, padx=(8, 0))
    _button(buttons, "Analyse Pack", getattr(app, "_evaluation_workflow_prepare_analysis_pack", None), "Erzeugt ein strukturiertes ZIP für Thesis/Paper-Dokumentation: Claim-Tabellen, Figuren, LaTeX und per-Modell/per-Split-Notizen.").pack(side=tk.LEFT, padx=(8, 0))

    assets_hint = ttk.Label(
        action_box,
        text="Finale und Screening-Datensätze sowie Hardware-Run-Profile werden zentral in Tool Config gepflegt. Der Run-Modus entscheidet automatisch über Umfang und Strenge.",
        foreground="#666",
        wraplength=1100,
        justify="left",
    )
    assets_hint.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
    _tip(assets_hint, "Smoke nutzt kleine Screeningdaten; Standard und Final binden die registrierten Calibration-/Validation-Manifeste gemäß zentralem Run-Modus.")

    status_row = ttk.Frame(action_box)
    status_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 6))
    _label(status_row, "Status:", "Aktueller GUI-Status des Workflow-Jobs.").pack(side=tk.LEFT)
    lbl_status = ttk.Label(status_row, textvariable=var_status)
    lbl_status.pack(side=tk.LEFT, padx=(6, 18))
    _tip(lbl_status, "Live-Logs und Fortschritt siehst du im Jobs-Tab oder im automatisch geöffneten Fortschrittsfenster.")
    _label(status_row, "Letztes Results Bundle:", "Ordner des letzten Evaluation-Runs in dieser GUI-Session.").pack(side=tk.LEFT)
    lbl_last = ttk.Label(status_row, textvariable=var_last_run)
    lbl_last.pack(side=tk.LEFT, padx=(6, 0))
    _tip(lbl_last, "Dieses Bundle enthält profile.yaml, run_manifest.json, models/<id>/..., reports/... und alle Stufenartefakte.")

    txt = tk.Text(action_box, height=8, wrap="word")
    txt.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
    yscroll = ttk.Scrollbar(action_box, orient="vertical", command=txt.yview)
    yscroll.grid(row=3, column=1, sticky="ns", pady=(0, 8))
    txt.configure(yscrollcommand=yscroll.set)
    txt.insert(
        "1.0",
        (
            "Noch kein Workflow-Lauf in dieser GUI-Session.\n\n"
            "Bedienung:\n"
            "1) YAML-Profil erstellen/auswählen.\n"
            "2) Profile Summary prüfen.\n"
            "3) Start Evaluation Workflow drücken.\n"
            "4) Jobs beobachten und anschließend Results Bundle öffnen.\n\n"
            "Remote/Hailo/Runtime werden aus dem YAML übernommen; es gibt hier keine zweite Advanced-Konfiguration mehr.\n"
        ),
    )
    txt.configure(state="disabled")
    _tip(txt, "Nach dem Lauf zeigt diese Zusammenfassung Run-Verzeichnis, Manifest, Reports, Validation und Hardware-Smoke-Status.")

    try:
        var_profile.trace_add("write", lambda *_: refresh_summary())
    except Exception:
        try:
            var_profile.trace("w", lambda *_: refresh_summary())
        except Exception:
            pass
    refresh_summary()

    if app is not None:
        app.eval_workflow_summary_text = txt
        app.eval_workflow_profile_summary_text = profile_summary
        app.eval_workflow_profile_combo = cbo
        app._evaluation_workflow_refresh_profile_summary = refresh_summary

    return outer
