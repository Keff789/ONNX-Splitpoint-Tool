"""Benchmark-analysis tab.

Loads benchmark results (folder or archive), compares measured results with the
benchmark-set predictions and renders concise summaries, tables and plots.
"""

from __future__ import annotations

import csv
import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Dict, Optional

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from ...benchmark.analysis import (
    BenchmarkAnalysisReport,
    BenchmarkComparisonReport,
    build_benchmark_analysis_figures,
    build_benchmark_comparison_figures,
    candidate_summary_rows,
    comparison_candidate_rows,
    comparison_hailo_rows,
    comparison_provider_rows,
    export_benchmark_analysis,
    export_benchmark_comparison,
    hailo_context_rows,
    hailo_outlook_rows,
    hailo_outlook_summary,
    hailo_part2_fallback_rows,
    hailo_part2_fallback_summary,
    load_benchmark_analysis,
    load_benchmark_analysis_comparison,
    provider_summary_rows,
)
from ...benchmark.interleaving_analysis import (
    build_metric_audit_fps_figure,
    build_metric_audit_rank_figure,
    build_comparison_metric_audit_rank_error_figure,
    build_comparison_interleaving_fps_delta_figure,
    build_interleaving_gain_figure,
    build_interleaving_tradeoff_figure,
    build_interleaving_residual_overhead_figure,
    compare_interleaving_reports,
    comparison_interleaving_rows,
    compute_interleaving_analysis,
    metric_audit_comparison_rows,
    metric_audit_comparison_summary,
    export_interleaving_analysis,
    export_interleaving_comparison,
    export_metric_audit_comparison,
    interleaving_candidate_rows,
    interleaving_provider_rows,
    metric_audit_rows,
    metric_audit_summary,
    research_best_full_vs_split_rows,
    research_prediction_audit_rows,
    research_stage_breakdown_rows,
    research_summary_cards,
    research_comparison_rows,
    export_publication_analysis,
    export_publication_comparison,
)
from ...benchmark.services import BenchmarkAnalysisService, LoadedBenchmarkAnalysis, LoadedBenchmarkComparison
from ...objective_scoring import set_throughput_calibration_enabled
from ...workdir import ensure_workdir
from ..widgets.tooltip import attach_tooltip
from ..widgets.status_badge import StatusBadge

logger = logging.getLogger(__name__)


def _bool_var(app, name: str, default: bool) -> tk.BooleanVar:
    if app is None:
        return tk.BooleanVar(value=default)
    existing = getattr(app, name, None)
    if existing is not None:
        return existing
    created = tk.BooleanVar(value=default)
    setattr(app, name, created)
    return created


def _objective_badge_level(objective_name: str) -> str:
    slug = str(objective_name or '').strip().lower()
    if slug.startswith('through'):
        return 'ok'
    if slug.startswith('hailo'):
        return 'warn'
    if slug.startswith('lat'):
        return 'error'
    return 'idle'


def _str_var(app: Any, name: str, default: str) -> tk.StringVar:
    if app is None:
        return tk.StringVar(value=default)
    existing = getattr(app, name, None)
    if existing is not None:
        return existing
    created = tk.StringVar(value=default)
    setattr(app, name, created)
    return created



def _analysis_cache_base(app: Any | None) -> Path:
    """Return a stable cache directory for extracted benchmark-analysis archives.

    Prefer the configured workdir under the GUI output root; fall back to the
    current working directory when no output root is configured.
    """

    root = getattr(app, "default_output_dir", None) if app is not None else None
    if root:
        try:
            return ensure_workdir(Path(root)).results / "benchmark_analysis_cache"
        except Exception:
            try:
                return Path(root).expanduser().resolve() / "benchmark_analysis_cache"
            except Exception:
                pass
    return Path.cwd().resolve() / "benchmark_analysis_cache"



def _placeholder_figure(title: str, message: str) -> Figure:
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig



def build_panel(parent, app=None) -> ttk.Frame:
    frame = ttk.Frame(parent)
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(2, weight=1)

    var_source = _str_var(app, "var_benchmark_analysis_source", "")
    var_compare_source = _str_var(app, "var_benchmark_analysis_compare_source", "")
    status_var = _str_var(app, "var_benchmark_analysis_status", "Noch keine Analyse geladen.")
    var_use_cal = _bool_var(app, 'var_use_throughput_calibration', True)
    try:
        set_throughput_calibration_enabled(bool(var_use_cal.get()))
    except Exception:
        pass

    ctrl = ttk.LabelFrame(frame, text="Benchmark-Ergebnisse analysieren")
    ctrl.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 8))
    ctrl.columnconfigure(1, weight=1)

    ttk.Label(ctrl, text="Quelle A:").grid(row=0, column=0, sticky="w", padx=(8, 6), pady=8)
    ent_source = ttk.Entry(ctrl, textvariable=var_source)
    ent_source.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=8)
    attach_tooltip(
        ent_source,
        "Pfad zu einer Benchmark-Suite, einem results/-Ordner, benchmark_set.json oder einem Archiv (tar.gz/zip).",
    )

    ttk.Label(ctrl, text="Vergleich B:").grid(row=1, column=0, sticky="w", padx=(8, 6), pady=(0, 8))
    ent_compare = ttk.Entry(ctrl, textvariable=var_compare_source)
    ent_compare.grid(row=1, column=1, sticky="ew", padx=(0, 8), pady=(0, 8))
    attach_tooltip(
        ent_compare,
        "Optional: zweite Benchmark-Quelle zum direkten Vergleich. Wenn leer, wird nur Quelle A analysiert.",
    )

    def _browse_file(target: tk.StringVar, title: str) -> None:
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[
                ("Results / archive", "*.tar.gz *.tgz *.tar *.zip *.json *.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            target.set(path)

    def _browse_dir(target: tk.StringVar, title: str) -> None:
        path = filedialog.askdirectory(title=title)
        if path:
            target.set(path)

    btn_file = ttk.Button(ctrl, text="Datei…", command=lambda: _browse_file(var_source, "Benchmark-Ergebnisse oder Archiv für Quelle A wählen"))
    btn_file.grid(row=0, column=2, sticky="w", padx=(0, 6), pady=8)
    btn_dir = ttk.Button(ctrl, text="Ordner…", command=lambda: _browse_dir(var_source, "Benchmark-Suite / results-Ordner für Quelle A wählen"))
    btn_dir.grid(row=0, column=3, sticky="w", padx=(0, 12), pady=8)

    btn_compare_file = ttk.Button(ctrl, text="Datei…", command=lambda: _browse_file(var_compare_source, "Benchmark-Ergebnisse oder Archiv für Vergleich B wählen"))
    btn_compare_file.grid(row=1, column=2, sticky="w", padx=(0, 6), pady=(0, 8))
    btn_compare_dir = ttk.Button(ctrl, text="Ordner…", command=lambda: _browse_dir(var_compare_source, "Benchmark-Suite / results-Ordner für Vergleich B wählen"))
    btn_compare_dir.grid(row=1, column=3, sticky="w", padx=(0, 12), pady=(0, 8))

    btn_clear_compare = ttk.Button(ctrl, text="B leeren", command=lambda: var_compare_source.set(""))
    btn_clear_compare.grid(row=1, column=4, sticky="w", padx=(0, 8), pady=(0, 8))

    btn_analyze = ttk.Button(ctrl, text="Analyse starten")
    btn_analyze.grid(row=0, column=5, rowspan=1, sticky="e", padx=(0, 8), pady=8)
    btn_export = ttk.Button(ctrl, text="Report exportieren…")
    btn_export.grid(row=1, column=5, sticky="e", padx=(0, 8), pady=(0, 8))
    btn_pub = ttk.Button(ctrl, text="Publication export…")
    btn_pub.grid(row=1, column=6, sticky="e", padx=(0, 8), pady=(0, 8))

    chk_use_cal = ttk.Checkbutton(ctrl, text='Use TH calibration', variable=var_use_cal)
    chk_use_cal.grid(row=2, column=4, sticky='e', padx=(0, 8), pady=(0, 8))
    attach_tooltip(chk_use_cal, 'Use the benchmark-derived throughput calibration in Metric Audit and research summaries.\nDisable this to inspect the raw throughput heuristic.')
    attach_tooltip(btn_analyze, "Lädt benchmark_set.json und benchmark_results_*.csv, erzeugt Zusammenfassungen und Grafiken. Mit Quelle B zusätzlich als direkter Run-Vergleich.")
    attach_tooltip(btn_export, "Exportiert die aktuelle Analyse oder den Vergleich als Markdown, CSV sowie PNG/PDF/SVG-Grafiken (paper-tauglich).")
    attach_tooltip(btn_pub, "Exportiert eine kompakte publication-ready Mappe mit CSV-, TeX- und Markdown-Dateien für die wichtigsten Forschungsresultate.")

    lbl_status = ttk.Label(ctrl, textvariable=status_var)
    lbl_status.grid(row=2, column=0, columnspan=4, sticky="w", padx=(8, 8), pady=(0, 8))
    objective_badge = StatusBadge(ctrl, text='Balanced', level='idle')
    objective_badge.grid(row=2, column=5, sticky='e', padx=(0, 8), pady=(0, 8))
    calibration_badge = StatusBadge(ctrl, text='CAL', level='ok')
    calibration_badge.grid(row=2, column=6, sticky='e', padx=(0, 8), pady=(0, 8))

    cards = ttk.Frame(frame)
    cards.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
    for _c in range(4):
        cards.columnconfigure(_c, weight=1)

    card_vars = {
        'best_full_title': tk.StringVar(value='Best full deployment'),
        'best_full_value': tk.StringVar(value='-'),
        'best_full_detail': tk.StringVar(value='Noch keine Analyse geladen.'),
        'best_split_title': tk.StringVar(value='Best split deployment'),
        'best_split_value': tk.StringVar(value='-'),
        'best_split_detail': tk.StringVar(value='Noch keine Analyse geladen.'),
        'delta_title': tk.StringVar(value='Δ throughput vs best full'),
        'delta_value': tk.StringVar(value='-'),
        'delta_detail': tk.StringVar(value='Noch keine Analyse geladen.'),
        'quality_title': tk.StringVar(value='Prediction quality'),
        'quality_value': tk.StringVar(value='-'),
        'quality_detail': tk.StringVar(value='Noch keine Analyse geladen.'),
    }

    def _make_card(parent_frame: ttk.Frame, col: int, title_var: tk.StringVar, value_var: tk.StringVar, detail_var: tk.StringVar) -> None:
        box = ttk.LabelFrame(parent_frame, text=str(title_var.get()))
        box.grid(row=0, column=col, sticky='nsew', padx=(0 if col == 0 else 6, 0))
        box.columnconfigure(0, weight=1)
        try:
            title_var.trace_add('write', lambda *_args, _box=box, _var=title_var: _box.configure(text=str(_var.get())))
        except Exception:
            pass
        ttk.Label(box, textvariable=value_var, font=("TkDefaultFont", 14, "bold")).grid(row=0, column=0, sticky='w', padx=8, pady=(6, 2))
        ttk.Label(box, textvariable=detail_var, wraplength=250, justify='left').grid(row=1, column=0, sticky='w', padx=8, pady=(0, 8))

    _make_card(cards, 0, card_vars['best_full_title'], card_vars['best_full_value'], card_vars['best_full_detail'])
    _make_card(cards, 1, card_vars['best_split_title'], card_vars['best_split_value'], card_vars['best_split_detail'])
    _make_card(cards, 2, card_vars['delta_title'], card_vars['delta_value'], card_vars['delta_detail'])
    _make_card(cards, 3, card_vars['quality_title'], card_vars['quality_value'], card_vars['quality_detail'])

    body = ttk.Notebook(frame)
    body.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))

    summary_tab = ttk.Frame(body)
    summary_tab.columnconfigure(0, weight=1)
    summary_tab.rowconfigure(0, weight=1)
    txt_summary = scrolledtext.ScrolledText(summary_tab, wrap="word", height=18)
    txt_summary.grid(row=0, column=0, sticky="nsew")
    txt_summary.configure(state="disabled")
    body.add(summary_tab, text="Zusammenfassung")

    research_tab = ttk.Frame(body)
    research_tab.columnconfigure(0, weight=1)
    research_tab.rowconfigure(0, weight=1)
    research_nb = ttk.Notebook(research_tab)
    research_nb.grid(row=0, column=0, sticky='nsew')

    best_tab = ttk.Frame(research_nb)
    best_tab.columnconfigure(0, weight=1)
    best_tab.rowconfigure(0, weight=1)
    best_cols = ('role', 'provider_or_pipeline', 'boundary', 'latency_ms', 'fps_equiv', 'delta_vs_best_full_pct', 'predicted_rank', 'cut_mib')
    tree_best = ttk.Treeview(best_tab, columns=best_cols, show='headings', height=6)
    best_headings = {
        'role': 'Role',
        'provider_or_pipeline': 'Provider / Pipeline',
        'boundary': 'Boundary',
        'latency_ms': 'Latency [ms]',
        'fps_equiv': 'FPS',
        'delta_vs_best_full_pct': 'Δ vs best full [%]',
        'predicted_rank': 'Pred rank',
        'cut_mib': 'Cut [MiB]',
    }
    for col in best_cols:
        tree_best.heading(col, text=best_headings[col])
        tree_best.column(col, width=(130 if col == 'provider_or_pipeline' else 84), stretch=(col == 'provider_or_pipeline'), anchor='center')
    vsb_best = ttk.Scrollbar(best_tab, orient='vertical', command=tree_best.yview)
    tree_best.configure(yscrollcommand=vsb_best.set)
    tree_best.grid(row=0, column=0, sticky='nsew')
    vsb_best.grid(row=0, column=1, sticky='ns')
    research_nb.add(best_tab, text='Best full vs split')

    audit_tab = ttk.Frame(research_nb)
    audit_tab.columnconfigure(0, weight=1)
    audit_tab.rowconfigure(0, weight=1)
    audit_cols = ('boundary', 'provider', 'predicted_rank', 'predicted_score', 'actual_rank', 'measured_streaming_fps', 'streaming_efficiency_pct', 'residual_overhead_ms', 'cut_mib', 'compile_risk')
    tree_audit = ttk.Treeview(audit_tab, columns=audit_cols, show='headings', height=12)
    audit_headings = {
        'boundary': 'Boundary', 'provider': 'Pipeline', 'predicted_rank': 'Pred rank', 'predicted_score': 'Pred score', 'actual_rank': 'Actual rank',
        'measured_streaming_fps': 'Measured FPS', 'streaming_efficiency_pct': 'Eff. [%]', 'residual_overhead_ms': 'Residual [ms]', 'cut_mib': 'Cut [MiB]', 'compile_risk': 'Compile risk'
    }
    for col in audit_cols:
        tree_audit.heading(col, text=audit_headings[col])
        tree_audit.column(col, width=(112 if col == 'provider' else 82), stretch=(col == 'provider'), anchor='center')
    vsb_audit = ttk.Scrollbar(audit_tab, orient='vertical', command=tree_audit.yview)
    tree_audit.configure(yscrollcommand=vsb_audit.set)
    tree_audit.grid(row=0, column=0, sticky='nsew')
    vsb_audit.grid(row=0, column=1, sticky='ns')
    research_nb.add(audit_tab, text='Prediction audit')

    stage_tab = ttk.Frame(research_nb)
    stage_tab.columnconfigure(0, weight=1)
    stage_tab.rowconfigure(0, weight=1)
    stage_cols = ('boundary', 'provider', 'stage1_provider', 'stage2_provider', 'part1_mean_ms', 'part2_mean_ms', 'ideal_bottleneck_fps', 'measured_streaming_fps', 'streaming_efficiency_pct', 'residual_overhead_ms', 'cut_mib')
    tree_stage = ttk.Treeview(stage_tab, columns=stage_cols, show='headings', height=12)
    stage_headings = {
        'boundary': 'Boundary', 'provider': 'Pipeline', 'stage1_provider': 'Stage1', 'stage2_provider': 'Stage2', 'part1_mean_ms': 'Stage1 [ms]', 'part2_mean_ms': 'Stage2 [ms]',
        'ideal_bottleneck_fps': 'Ideal FPS', 'measured_streaming_fps': 'Measured FPS', 'streaming_efficiency_pct': 'Eff. [%]', 'residual_overhead_ms': 'Residual [ms]', 'cut_mib': 'Cut [MiB]'
    }
    for col in stage_cols:
        tree_stage.heading(col, text=stage_headings[col])
        tree_stage.column(col, width=(96 if col in {'provider','stage1_provider','stage2_provider'} else 82), stretch=(col in {'provider','stage1_provider','stage2_provider'}), anchor='center')
    vsb_stage = ttk.Scrollbar(stage_tab, orient='vertical', command=tree_stage.yview)
    tree_stage.configure(yscrollcommand=vsb_stage.set)
    tree_stage.grid(row=0, column=0, sticky='nsew')
    vsb_stage.grid(row=0, column=1, sticky='ns')
    research_nb.add(stage_tab, text='Stage breakdown')

    metric_tab = ttk.Frame(research_nb)
    metric_tab.columnconfigure(0, weight=1)
    metric_tab.rowconfigure(1, weight=1)
    metric_summary_var = tk.StringVar(value='Noch keine Metric-Audit-Daten geladen.')
    metric_detail_var = tk.StringVar(value='Die throughput-orientierte Metrik erweitert den bisherigen Score um eine explizite Handover-Penalty.')
    ttk.Label(metric_tab, textvariable=metric_summary_var, font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=(6, 6), pady=(6, 2))
    ttk.Label(metric_tab, textvariable=metric_detail_var, wraplength=980, justify='left').grid(row=0, column=1, sticky='w', padx=(0, 6), pady=(6, 2))
    metric_cols = (
        'boundary', 'provider', 'predicted_rank_old', 'predicted_rank_uncal', 'predicted_rank_cal', 'actual_rank',
        'predicted_score_old', 'hailo_feasibility_risk', 'hailo_interface_penalty',
        'predicted_handover_ms_uncal', 'predicted_handover_ms_cal',
        'predicted_stream_fps_uncal', 'predicted_stream_fps_cal', 'measured_streaming_fps',
        'rank_error_old_abs', 'rank_error_uncal_abs', 'rank_error_cal_abs'
    )
    tree_metric = ttk.Treeview(metric_tab, columns=metric_cols, show='headings', height=12)
    metric_headings = {
        'boundary': 'Boundary', 'provider': 'Pipeline', 'predicted_rank_old': 'Old rank', 'predicted_rank_uncal': 'TH rank raw', 'predicted_rank_cal': 'TH rank cal', 'actual_rank': 'Actual rank',
        'predicted_score_old': 'Old score', 'hailo_feasibility_risk': 'H feas', 'hailo_interface_penalty': 'H iface',
        'predicted_handover_ms_uncal': 'HO raw [ms]', 'predicted_handover_ms_cal': 'HO cal [ms]',
        'predicted_stream_fps_uncal': 'TH raw FPS', 'predicted_stream_fps_cal': 'TH cal FPS', 'measured_streaming_fps': 'Measured FPS',
        'rank_error_old_abs': '|Δ old|', 'rank_error_uncal_abs': '|Δ raw|', 'rank_error_cal_abs': '|Δ cal|',
    }
    for col in metric_cols:
        tree_metric.heading(col, text=metric_headings[col])
        tree_metric.column(col, width=(96 if col == 'provider' else 72), stretch=(col == 'provider'), anchor='center')
    vsb_metric = ttk.Scrollbar(metric_tab, orient='vertical', command=tree_metric.yview)
    tree_metric.configure(yscrollcommand=vsb_metric.set)
    tree_metric.grid(row=1, column=0, sticky='nsew')
    vsb_metric.grid(row=1, column=1, sticky='ns')
    tree_metric.tag_configure('metric_better', background='#eef8f0')
    tree_metric.tag_configure('metric_worse', background='#fdeeee')
    research_nb.add(metric_tab, text='Metric Audit')

    body.add(research_tab, text='Research Summary')

    provider_tab = ttk.Frame(body)
    provider_tab.columnconfigure(0, weight=1)
    provider_tab.rowconfigure(0, weight=1)
    provider_cols = (
        "provider",
        "rows",
        "full_baseline_ms",
        "best_boundary",
        "best_composed_ms",
        "delta_vs_full_pct",
        "score_spearman",
        "latency_spearman",
        "top5_overlap",
    )
    tree_provider = ttk.Treeview(provider_tab, columns=provider_cols, show="headings", height=10)
    provider_headings = {
        "provider": "Provider",
        "rows": "Rows",
        "full_baseline_ms": "Full [ms]",
        "best_boundary": "Best b",
        "best_composed_ms": "Best split [ms]",
        "delta_vs_full_pct": "Δ vs full [%]",
        "score_spearman": "ρ(score)",
        "latency_spearman": "ρ(latency)",
        "top5_overlap": "Top-5 overlap",
    }
    for col in provider_cols:
        tree_provider.heading(col, text=provider_headings[col])
        tree_provider.column(col, width=92 if col != 'provider' else 132, stretch=(col == 'provider'), anchor='center')
    vsb_provider = ttk.Scrollbar(provider_tab, orient="vertical", command=tree_provider.yview)
    tree_provider.configure(yscrollcommand=vsb_provider.set)
    tree_provider.grid(row=0, column=0, sticky="nsew")
    vsb_provider.grid(row=0, column=1, sticky="ns")
    body.add(provider_tab, text="Provider A")

    cand_tab = ttk.Frame(body)
    cand_tab.columnconfigure(0, weight=1)
    cand_tab.rowconfigure(0, weight=1)
    cand_cols = (
        "boundary",
        "hailo_part2_marker",
        "avg_rank",
        "top3_hits",
        "providers_present",
        "best_provider",
        "best_rank",
        "score_pred",
        "cut_mib",
    )
    tree_cand = ttk.Treeview(cand_tab, columns=cand_cols, show="headings", height=12)
    cand_headings = {
        "boundary": "Boundary",
        "hailo_part2_marker": "P2 alt",
        "avg_rank": "Ø Rang",
        "top3_hits": "Top-3 Hits",
        "providers_present": "Provider",
        "best_provider": "Bester Provider",
        "best_rank": "Best Rang",
        "score_pred": "Score pred",
        "cut_mib": "Cut [MiB]",
    }
    for col in cand_cols:
        tree_cand.heading(col, text=cand_headings[col])
        width = 68 if col == "hailo_part2_marker" else (105 if col != "best_provider" else 130)
        tree_cand.column(col, width=min(width, 108) if col not in {'best_provider','fallback_detail'} else width, stretch=(col in {'best_provider','fallback_detail'}), anchor='center')
    vsb_cand = ttk.Scrollbar(cand_tab, orient="vertical", command=tree_cand.yview)
    tree_cand.configure(yscrollcommand=vsb_cand.set)
    tree_cand.grid(row=0, column=0, sticky="nsew")
    vsb_cand.grid(row=0, column=1, sticky="ns")
    body.add(cand_tab, text="Kandidaten A")

    hailo_outlook_tab = ttk.Frame(body)
    hailo_outlook_tab.columnconfigure(0, weight=1)
    hailo_outlook_tab.rowconfigure(1, weight=1)
    hailo_outlook_summary_var = tk.StringVar(value="Noch keine Hailo-Outlook-Daten geladen.")
    hailo_outlook_detail_var = tk.StringVar(value="Die Outlook-Tabelle verdichtet Compile-Risiko und 1-Kontext-Wahrscheinlichkeit aus dem Benchmark-Set.")
    ttk.Label(hailo_outlook_tab, textvariable=hailo_outlook_summary_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 2))
    ttk.Label(hailo_outlook_tab, textvariable=hailo_outlook_detail_var, wraplength=980, justify="left").grid(row=0, column=1, sticky="w", padx=(0, 6), pady=(6, 2))
    outlook_table = ttk.Frame(hailo_outlook_tab)
    outlook_table.grid(row=1, column=0, columnspan=2, sticky="nsew")
    outlook_table.columnconfigure(0, weight=1)
    outlook_cols = (
        "rank",
        "boundary",
        "hailo_part2_marker",
        "compile_risk_score",
        "risk_band",
        "single_context_probability",
        "cut_mib",
        "score_pred",
        "avg_rank",
        "providers_present",
        "recommendation",
    )
    tree_hailo_outlook = ttk.Treeview(hailo_outlook_tab, columns=outlook_cols, show="headings", height=12)
    outlook_headings = {
        "rank": "#",
        "boundary": "Boundary",
        "hailo_part2_marker": "P2 alt",
        "compile_risk_score": "Compile risk",
        "risk_band": "Band",
        "single_context_probability": "1-context %",
        "cut_mib": "Cut [MiB]",
        "score_pred": "Score pred",
        "avg_rank": "Ø Rang",
        "providers_present": "Provider",
        "recommendation": "Recommendation",
    }
    outlook_widths = {
        "rank": 44,
        "boundary": 80,
        "hailo_part2_marker": 68,
        "compile_risk_score": 96,
        "risk_band": 72,
        "single_context_probability": 96,
        "cut_mib": 86,
        "score_pred": 86,
        "avg_rank": 78,
        "providers_present": 72,
        "recommendation": 280,
    }
    for col in outlook_cols:
        tree_hailo_outlook.heading(col, text=outlook_headings[col])
        tree_hailo_outlook.column(col, width=min(outlook_widths.get(col, 100), 108) if col != 'recommendation' else 180, stretch=(col == 'recommendation'), anchor='center')
    vsb_hailo_outlook = ttk.Scrollbar(hailo_outlook_tab, orient="vertical", command=tree_hailo_outlook.yview)
    tree_hailo_outlook.configure(yscrollcommand=vsb_hailo_outlook.set)
    tree_hailo_outlook.grid(row=1, column=0, sticky="nsew", padx=(0, 0), pady=(6, 0))
    vsb_hailo_outlook.grid(row=1, column=1, sticky="ns", pady=(6, 0))
    attach_tooltip(tree_hailo_outlook, "Kompakter Hailo-Outlook für die wichtigsten Kandidaten: geringes Compile-Risiko und hohe 1-Kontext-Wahrscheinlichkeit sind besser.")
    try:
        tree_hailo_outlook.tag_configure("risk_low", background="#eef8f0")
        tree_hailo_outlook.tag_configure("risk_medium", background="#fff6e6")
        tree_hailo_outlook.tag_configure("risk_high", background="#fdeeee")
    except Exception:
        pass
    body.add(hailo_outlook_tab, text="Hailo Outlook A")

    hailo_fallback_tab = ttk.Frame(body)
    hailo_fallback_tab.columnconfigure(0, weight=1)
    hailo_fallback_tab.rowconfigure(1, weight=1)
    hailo_fallback_summary_var = tk.StringVar(value="Noch keine Hailo-Part2-Fallback-Daten geladen.")
    hailo_fallback_detail_var = tk.StringVar(value="Diese Sicht zeigt Kandidaten, die beim Hailo-Part2-Build eine alternative suggested end-node Strategie verwenden.")
    ttk.Label(hailo_fallback_tab, textvariable=hailo_fallback_summary_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 2))
    ttk.Label(hailo_fallback_tab, textvariable=hailo_fallback_detail_var, wraplength=980, justify="left").grid(row=0, column=1, sticky="w", padx=(0, 6), pady=(6, 2))
    fallback_cols = (
        "boundary",
        "marker",
        "strategy_label",
        "effective_outputs_text",
        "avg_rank",
        "best_provider",
        "best_rank",
        "single_context_probability",
        "compile_risk_score",
        "recommendation",
    )
    tree_hailo_fallback = ttk.Treeview(hailo_fallback_tab, columns=fallback_cols, show="headings", height=12)
    fallback_headings = {
        "boundary": "Boundary",
        "marker": "P2 alt",
        "strategy_label": "Strategie",
        "effective_outputs_text": "Eff. outputs",
        "avg_rank": "Ø Rang",
        "best_provider": "Bester Provider",
        "best_rank": "Best Rang",
        "single_context_probability": "1-context %",
        "compile_risk_score": "Compile risk",
        "recommendation": "Recommendation",
    }
    fallback_widths = {
        "boundary": 80,
        "marker": 68,
        "strategy_label": 96,
        "effective_outputs_text": 220,
        "avg_rank": 78,
        "best_provider": 120,
        "best_rank": 72,
        "single_context_probability": 96,
        "compile_risk_score": 96,
        "recommendation": 260,
    }
    for col in fallback_cols:
        tree_hailo_fallback.heading(col, text=fallback_headings[col])
        tree_hailo_fallback.column(col, width=150 if col in {'effective_outputs_text','recommendation'} else min(fallback_widths.get(col,100), 108), stretch=(col in {'effective_outputs_text','recommendation'}), anchor='center')
    vsb_hailo_fallback = ttk.Scrollbar(hailo_fallback_tab, orient="vertical", command=tree_hailo_fallback.yview)
    tree_hailo_fallback.configure(yscrollcommand=vsb_hailo_fallback.set)
    tree_hailo_fallback.grid(row=1, column=0, sticky="nsew", padx=(0, 0), pady=(6, 0))
    vsb_hailo_fallback.grid(row=1, column=1, sticky="ns", pady=(6, 0))
    attach_tooltip(tree_hailo_fallback, "Kandidaten, die beim Hailo-Part2-Build eine alternative suggested end-node Strategie verwendet haben.")
    body.add(hailo_fallback_tab, text="Hailo Fallback A")

    hailo_tab = ttk.Frame(body)
    hailo_tab.columnconfigure(0, weight=1)
    hailo_tab.rowconfigure(0, weight=1)
    hailo_cols = (
        "boundary",
        "hw_arch",
        "part1_context_count",
        "part1_context_mode",
        "part2_context_count",
        "part2_context_mode",
        "part2_single_context",
        "both_parts_single_context",
        "part2_partition_time_s",
        "part2_allocation_time_s",
        "part2_compilation_time_s",
        "part2_calib_source",
        "direct_hailo_composed_ms",
        "cut_mib",
        "score_pred",
    )
    tree_hailo = ttk.Treeview(hailo_tab, columns=hailo_cols, show="headings", height=12)
    hailo_headings = {
        "boundary": "Boundary",
        "hw_arch": "HW",
        "part1_context_count": "p1 ctx",
        "part1_context_mode": "p1 mode",
        "part2_context_count": "p2 ctx",
        "part2_context_mode": "p2 mode",
        "part2_single_context": "p2 single",
        "both_parts_single_context": "both single",
        "part2_partition_time_s": "p2 partition [s]",
        "part2_allocation_time_s": "p2 alloc [s]",
        "part2_compilation_time_s": "p2 compile [s]",
        "part2_calib_source": "p2 calib",
        "direct_hailo_composed_ms": "direct Hailo [ms]",
        "cut_mib": "Cut [MiB]",
        "score_pred": "Score pred",
    }
    for col in hailo_cols:
        tree_hailo.heading(col, text=hailo_headings[col])
        tree_hailo.column(col, width=96, stretch=True, anchor='center')
    vsb_hailo = ttk.Scrollbar(hailo_tab, orient="vertical", command=tree_hailo.yview)
    tree_hailo.configure(yscrollcommand=vsb_hailo.set)
    tree_hailo.grid(row=0, column=0, sticky="nsew")
    vsb_hailo.grid(row=0, column=1, sticky="ns")
    body.add(hailo_tab, text="Hailo Fit A")

    inter_tab = ttk.Frame(body)
    inter_tab.columnconfigure(0, weight=1)
    inter_tab.rowconfigure(0, weight=1)
    inter_cols = (
        "provider",
        "stage1_provider",
        "stage2_provider",
        "candidate_count",
        "best_boundary",
        "best_cycle_ms_cons",
        "best_pipeline_fps_cons",
        "gain_vs_stage1_full_pct",
        "gain_vs_stage2_full_pct",
        "gain_vs_best_single_full_pct",
        "latency_delta_vs_best_single_full_ms",
        "stage_balance",
    )
    tree_inter = ttk.Treeview(inter_tab, columns=inter_cols, show="headings", height=10)
    inter_headings = {
        "provider": "Pipeline",
        "stage1_provider": "Stage1",
        "stage2_provider": "Stage2",
        "candidate_count": "Cases",
        "best_boundary": "Best b",
        "best_cycle_ms_cons": "Cycle cons [ms]",
        "best_pipeline_fps_cons": "FPS cons",
        "gain_vs_stage1_full_pct": "vs s1 full [%]",
        "gain_vs_stage2_full_pct": "vs s2 full [%]",
        "gain_vs_best_single_full_pct": "vs best single [%]",
        "latency_delta_vs_best_single_full_ms": "Δ Latenz [ms]",
        "stage_balance": "Balance",
    }
    for col in inter_cols:
        tree_inter.heading(col, text=inter_headings[col])
        tree_inter.column(col, width=92 if col != 'provider' else 126, stretch=(col == 'provider'), anchor='center')
    vsb_inter = ttk.Scrollbar(inter_tab, orient="vertical", command=tree_inter.yview)
    tree_inter.configure(yscrollcommand=vsb_inter.set)
    tree_inter.grid(row=0, column=0, sticky="nsew")
    vsb_inter.grid(row=0, column=1, sticky="ns")
    body.add(inter_tab, text="Durchsatz / FPS A")

    inter_cand_tab = ttk.Frame(body)
    inter_cand_tab.columnconfigure(0, weight=1)
    inter_cand_tab.rowconfigure(0, weight=1)
    inter_cand_cols = (
        "provider",
        "boundary",
        "cycle_ms_cons",
        "pipeline_fps_cons",
        "composed_mean_ms",
        "gain_vs_best_single_full_pct",
        "latency_delta_vs_best_single_full_ms",
        "stage_balance",
        "cut_mib",
        "score_pred",
    )
    tree_inter_cand = ttk.Treeview(inter_cand_tab, columns=inter_cand_cols, show="headings", height=12)
    inter_cand_headings = {
        "provider": "Pipeline",
        "boundary": "Boundary",
        "cycle_ms_cons": "Cycle cons [ms]",
        "pipeline_fps_cons": "FPS cons",
        "composed_mean_ms": "Latenz [ms]",
        "gain_vs_best_single_full_pct": "vs best single [%]",
        "latency_delta_vs_best_single_full_ms": "Δ Latenz [ms]",
        "stage_balance": "Balance",
        "cut_mib": "Cut [MiB]",
        "score_pred": "Score pred",
    }
    for col in inter_cand_cols:
        tree_inter_cand.heading(col, text=inter_cand_headings[col])
        tree_inter_cand.column(col, width=92 if col != 'provider' else 126, stretch=(col == 'provider'), anchor='center')
    vsb_inter_cand = ttk.Scrollbar(inter_cand_tab, orient="vertical", command=tree_inter_cand.yview)
    tree_inter_cand.configure(yscrollcommand=vsb_inter_cand.set)
    tree_inter_cand.grid(row=0, column=0, sticky="nsew")
    vsb_inter_cand.grid(row=0, column=1, sticky="ns")
    body.add(inter_cand_tab, text="FPS-Kandidaten A")

    research_cmp_tab = ttk.Frame(body)
    research_cmp_tab.columnconfigure(0, weight=1)
    research_cmp_tab.rowconfigure(1, weight=1)
    research_cmp_summary_var = tk.StringVar(value="Noch kein Research-Vergleich geladen.")
    research_cmp_detail_var = tk.StringVar(value="Diese Sicht verdichtet die für eine wissenschaftliche Darstellung wichtigsten A↔B-Deltas kompakt in einer Tabelle.")
    ttk.Label(research_cmp_tab, textvariable=research_cmp_summary_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 2))
    ttk.Label(research_cmp_tab, textvariable=research_cmp_detail_var, wraplength=980, justify="left").grid(row=0, column=1, sticky="w", padx=(0, 6), pady=(6, 2))
    research_cmp_cols = ("metric", "left", "right", "delta")
    tree_research_cmp = ttk.Treeview(research_cmp_tab, columns=research_cmp_cols, show="headings", height=12)
    research_cmp_headings = {"metric": "Metric", "left": "Run A", "right": "Run B", "delta": "Δ"}
    research_cmp_widths = {"metric": 220, "left": 110, "right": 110, "delta": 96}
    for col in research_cmp_cols:
        tree_research_cmp.heading(col, text=research_cmp_headings[col])
        tree_research_cmp.column(col, width=research_cmp_widths[col], stretch=(col == 'metric'), anchor='center')
    vsb_research_cmp = ttk.Scrollbar(research_cmp_tab, orient="vertical", command=tree_research_cmp.yview)
    tree_research_cmp.configure(yscrollcommand=vsb_research_cmp.set)
    tree_research_cmp.grid(row=1, column=0, sticky="nsew")
    vsb_research_cmp.grid(row=1, column=1, sticky="ns")
    body.add(research_cmp_tab, text="Research-Vergleich")

    provider_cmp_tab = ttk.Frame(body)
    provider_cmp_tab.columnconfigure(0, weight=1)
    provider_cmp_tab.rowconfigure(0, weight=1)
    provider_cmp_cols = (
        "provider",
        "left_full_baseline_ms",
        "right_full_baseline_ms",
        "full_delta_pct",
        "left_best_boundary",
        "right_best_boundary",
        "left_best_composed_ms",
        "right_best_composed_ms",
        "best_delta_pct",
        "score_spearman_delta",
        "latency_spearman_delta",
        "best_boundary_changed",
    )
    tree_provider_cmp = ttk.Treeview(provider_cmp_tab, columns=provider_cmp_cols, show="headings", height=10)
    provider_cmp_headings = {
        "provider": "Provider",
        "left_full_baseline_ms": "A Full [ms]",
        "right_full_baseline_ms": "B Full [ms]",
        "full_delta_pct": "Full Δ [%]",
        "left_best_boundary": "A best b",
        "right_best_boundary": "B best b",
        "left_best_composed_ms": "A best [ms]",
        "right_best_composed_ms": "B best [ms]",
        "best_delta_pct": "Best Δ [%]",
        "score_spearman_delta": "Δ ρ(score)",
        "latency_spearman_delta": "Δ ρ(lat)",
        "best_boundary_changed": "best b geändert",
    }
    for col in provider_cmp_cols:
        tree_provider_cmp.heading(col, text=provider_cmp_headings[col])
        tree_provider_cmp.column(col, width=92 if col != 'provider' else 132, stretch=(col == 'provider'), anchor='center')
    vsb_provider_cmp = ttk.Scrollbar(provider_cmp_tab, orient="vertical", command=tree_provider_cmp.yview)
    tree_provider_cmp.configure(yscrollcommand=vsb_provider_cmp.set)
    tree_provider_cmp.grid(row=0, column=0, sticky="nsew")
    vsb_provider_cmp.grid(row=0, column=1, sticky="ns")
    body.add(provider_cmp_tab, text="Provider-Vergleich")

    cand_cmp_tab = ttk.Frame(body)
    cand_cmp_tab.columnconfigure(0, weight=1)
    cand_cmp_tab.rowconfigure(0, weight=1)
    cand_cmp_cols = (
        "boundary",
        "left_avg_rank",
        "right_avg_rank",
        "avg_rank_delta",
        "left_top3_hits",
        "right_top3_hits",
        "top3_delta",
        "left_best_provider",
        "right_best_provider",
    )
    tree_cand_cmp = ttk.Treeview(cand_cmp_tab, columns=cand_cmp_cols, show="headings", height=12)
    cand_cmp_headings = {
        "boundary": "Boundary",
        "left_avg_rank": "A Ø-Rang",
        "right_avg_rank": "B Ø-Rang",
        "avg_rank_delta": "Δ Ø-Rang",
        "left_top3_hits": "A Top-3",
        "right_top3_hits": "B Top-3",
        "top3_delta": "Δ Top-3",
        "left_best_provider": "A best Provider",
        "right_best_provider": "B best Provider",
    }
    for col in cand_cmp_cols:
        tree_cand_cmp.heading(col, text=cand_cmp_headings[col])
        tree_cand_cmp.column(col, width=90 if col not in {'left_best_provider','right_best_provider'} else 124, stretch=(col in {'left_best_provider','right_best_provider'}), anchor='center')
    vsb_cand_cmp = ttk.Scrollbar(cand_cmp_tab, orient="vertical", command=tree_cand_cmp.yview)
    tree_cand_cmp.configure(yscrollcommand=vsb_cand_cmp.set)
    tree_cand_cmp.grid(row=0, column=0, sticky="nsew")
    vsb_cand_cmp.grid(row=0, column=1, sticky="ns")
    body.add(cand_cmp_tab, text="Kandidaten-Vergleich")

    hailo_cmp_tab = ttk.Frame(body)
    hailo_cmp_tab.columnconfigure(0, weight=1)
    hailo_cmp_tab.rowconfigure(0, weight=1)
    hailo_cmp_cols = (
        "boundary",
        "hw_arch",
        "left_part2_context_count",
        "right_part2_context_count",
        "context_delta",
        "left_part2_context_mode",
        "right_part2_context_mode",
        "left_part2_single_context",
        "right_part2_single_context",
        "latency_delta_ms",
    )
    tree_hailo_cmp = ttk.Treeview(hailo_cmp_tab, columns=hailo_cmp_cols, show="headings", height=12)
    hailo_cmp_headings = {
        "boundary": "Boundary",
        "hw_arch": "HW",
        "left_part2_context_count": "A p2 ctx",
        "right_part2_context_count": "B p2 ctx",
        "context_delta": "Δ ctx",
        "left_part2_context_mode": "A p2 mode",
        "right_part2_context_mode": "B p2 mode",
        "left_part2_single_context": "A single",
        "right_part2_single_context": "B single",
        "latency_delta_ms": "Δ Hailo [ms]",
    }
    for col in hailo_cmp_cols:
        tree_hailo_cmp.heading(col, text=hailo_cmp_headings[col])
        tree_hailo_cmp.column(col, width=94, stretch=True, anchor='center')
    vsb_hailo_cmp = ttk.Scrollbar(hailo_cmp_tab, orient="vertical", command=tree_hailo_cmp.yview)
    tree_hailo_cmp.configure(yscrollcommand=vsb_hailo_cmp.set)
    tree_hailo_cmp.grid(row=0, column=0, sticky="nsew")
    vsb_hailo_cmp.grid(row=0, column=1, sticky="ns")
    body.add(hailo_cmp_tab, text="Hailo-Vergleich")

    inter_cmp_tab = ttk.Frame(body)
    inter_cmp_tab.columnconfigure(0, weight=1)
    inter_cmp_tab.rowconfigure(0, weight=1)
    inter_cmp_cols = (
        "provider",
        "stage1_provider",
        "stage2_provider",
        "left_best_boundary",
        "right_best_boundary",
        "left_best_pipeline_fps_cons",
        "right_best_pipeline_fps_cons",
        "fps_delta_pct",
        "left_gain_vs_best_single_full_pct",
        "right_gain_vs_best_single_full_pct",
        "gain_delta_pct",
        "latency_delta_ms",
    )
    tree_inter_cmp = ttk.Treeview(inter_cmp_tab, columns=inter_cmp_cols, show="headings", height=10)
    inter_cmp_headings = {
        "provider": "Pipeline",
        "stage1_provider": "Stage1",
        "stage2_provider": "Stage2",
        "left_best_boundary": "A best b",
        "right_best_boundary": "B best b",
        "left_best_pipeline_fps_cons": "A FPS cons",
        "right_best_pipeline_fps_cons": "B FPS cons",
        "fps_delta_pct": "Δ FPS [%]",
        "left_gain_vs_best_single_full_pct": "A vs best [%]",
        "right_gain_vs_best_single_full_pct": "B vs best [%]",
        "gain_delta_pct": "Δ gain [pct-pt]",
        "latency_delta_ms": "Δ Latenz [ms]",
    }
    for col in inter_cmp_cols:
        tree_inter_cmp.heading(col, text=inter_cmp_headings[col])
        tree_inter_cmp.column(col, width=92 if col != 'provider' else 126, stretch=(col == 'provider'), anchor='center')
    vsb_inter_cmp = ttk.Scrollbar(inter_cmp_tab, orient="vertical", command=tree_inter_cmp.yview)
    tree_inter_cmp.configure(yscrollcommand=vsb_inter_cmp.set)
    tree_inter_cmp.grid(row=0, column=0, sticky="nsew")
    vsb_inter_cmp.grid(row=0, column=1, sticky="ns")
    body.add(inter_cmp_tab, text="Durchsatz-Vergleich")

    metric_cmp_tab = ttk.Frame(body)
    metric_cmp_tab.columnconfigure(0, weight=1)
    metric_cmp_tab.rowconfigure(1, weight=1)
    metric_cmp_summary_var = tk.StringVar(value="Noch kein Metric-Audit-Vergleich geladen.")
    metric_cmp_detail_var = tk.StringVar(value="Der Vergleich zeigt, ob die throughput-orientierte Metrik gegenüber dem alten Ranking in Lauf B relativ zu Lauf A gewinnt oder verliert.")
    ttk.Label(metric_cmp_tab, textvariable=metric_cmp_summary_var, font=("TkDefaultFont", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(6, 6), pady=(6, 2))
    ttk.Label(metric_cmp_tab, textvariable=metric_cmp_detail_var, wraplength=980, justify="left").grid(row=0, column=1, sticky="w", padx=(0, 6), pady=(6, 2))
    metric_cmp_cols = (
        "boundary",
        "left_actual_rank",
        "right_actual_rank",
        "left_predicted_rank_old",
        "right_predicted_rank_old",
        "left_predicted_rank_uncal",
        "right_predicted_rank_uncal",
        "left_predicted_rank_cal",
        "right_predicted_rank_cal",
        "left_measured_streaming_fps",
        "right_measured_streaming_fps",
        "old_rank_error_abs_delta",
        "uncal_rank_error_abs_delta",
        "cal_rank_error_abs_delta",
        "fps_delta_pct",
    )
    tree_metric_cmp = ttk.Treeview(metric_cmp_tab, columns=metric_cmp_cols, show="headings", height=12)
    metric_cmp_headings = {
        "boundary": "Boundary",
        "left_actual_rank": "A actual",
        "right_actual_rank": "B actual",
        "left_predicted_rank_old": "A old rank",
        "right_predicted_rank_old": "B old rank",
        "left_predicted_rank_uncal": "A raw rank",
        "right_predicted_rank_uncal": "B raw rank",
        "left_predicted_rank_cal": "A cal rank",
        "right_predicted_rank_cal": "B cal rank",
        "left_measured_streaming_fps": "A FPS",
        "right_measured_streaming_fps": "B FPS",
        "old_rank_error_abs_delta": "Δ |old error|",
        "uncal_rank_error_abs_delta": "Δ |raw error|",
        "cal_rank_error_abs_delta": "Δ |cal error|",
        "fps_delta_pct": "Δ FPS [%]",
    }
    for col in metric_cmp_cols:
        tree_metric_cmp.heading(col, text=metric_cmp_headings[col])
        tree_metric_cmp.column(col, width=72, stretch=True, anchor='center')
    vsb_metric_cmp = ttk.Scrollbar(metric_cmp_tab, orient="vertical", command=tree_metric_cmp.yview)
    tree_metric_cmp.configure(yscrollcommand=vsb_metric_cmp.set)
    tree_metric_cmp.grid(row=1, column=0, sticky="nsew")
    vsb_metric_cmp.grid(row=1, column=1, sticky="ns")
    tree_metric_cmp.tag_configure('metric_cmp_better', background='#eef8f0')
    tree_metric_cmp.tag_configure('metric_cmp_worse', background='#fdeeee')
    body.add(metric_cmp_tab, text='Metric Audit-Vergleich')

    plots_tab = ttk.Notebook(body)
    body.add(plots_tab, text="Grafiken")
    plot_hosts: Dict[str, ttk.Frame] = {}
    plot_labels = {
        "best_vs_full": "A: Best split vs full",
        "predictor_quality": "A: Prognosegüte",
        "candidate_stability": "A: Kandidaten-Stabilität",
        "hailo_context_fit": "A: Hailo Context Fit",
        "hailo_latency_vs_context": "A: Hailo Latenz vs Kontext",
        "hailo_cut_vs_context": "A: Cut vs Kontext",
        "hailo_outlook_risk": "A: Hailo Risiko",
        "hailo_outlook_scatter": "A: Hailo Risiko vs 1ctx",
        "hailo_part2_fallback": "A: Hailo Part2 Fallback",
        "interleaving_gain": "A: Interleaving FPS",
        "interleaving_tradeoff": "A: FPS vs Latenz",
        "interleaving_residual_overhead": "A: Cut vs residual",
        "metric_audit_rank": "A: Metric Audit rank",
        "metric_audit_fps": "A: Metric Audit throughput",
        "comparison_provider_latency": "A↔B: Provider-Latenz",
        "comparison_predictor_delta": "A↔B: Prognosegüte Δ",
        "comparison_candidate_rank_shift": "A↔B: Kandidaten-Rang Δ",
        "comparison_hailo_context_delta": "A↔B: Hailo Context Δ",
        "comparison_hailo_latency_delta": "A↔B: Hailo Latenz Δ",
        "comparison_interleaving_fps_delta": "A↔B: Interleaving FPS Δ",
        "comparison_metric_audit_rank_error": "A↔B: Metric Audit Δ",
    }
    for key, label in plot_labels.items():
        host = ttk.Frame(plots_tab)
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        plots_tab.add(host, text=label)
        plot_hosts[key] = host

    state: Dict[str, Any] = {
        "report": None,
        "comparison": None,
        "interleaving": None,
        "interleaving_comparison": None,
        "figures": {},
        "canvases": {},
        "toolbars": {},
    }

    def _use_throughput_calibration() -> bool:
        try:
            return bool(var_use_cal.get())
        except Exception:
            return True

    def _apply_throughput_calibration_setting() -> None:
        try:
            set_throughput_calibration_enabled(_use_throughput_calibration())
        except Exception:
            logger.debug('Could not apply throughput calibration toggle', exc_info=True)

    def _rerender_with_current_calibration() -> None:
        _apply_throughput_calibration_setting()
        try:
            comparison = state.get('comparison')
            if comparison is not None:
                _render_comparison(comparison, inter_cmp=state.get('interleaving_comparison'))
                return
            report = state.get('report')
            if report is not None:
                _render_report(report, inter_analysis=state.get('interleaving'))
        except Exception:
            logger.exception('Failed to rerender benchmark analysis after calibration toggle')

    try:
        var_use_cal.trace_add('write', lambda *_args: frame.after_idle(_rerender_with_current_calibration))
    except Exception:
        pass

    frame.var_benchmark_analysis_source = var_source  # type: ignore[attr-defined]
    frame.var_benchmark_analysis_compare_source = var_compare_source  # type: ignore[attr-defined]
    frame.var_benchmark_analysis_status = status_var  # type: ignore[attr-defined]
    frame.benchmark_analysis_state = state  # type: ignore[attr-defined]

    def _set_summary_text(text: str) -> None:
        txt_summary.configure(state="normal")
        txt_summary.delete("1.0", tk.END)
        txt_summary.insert("1.0", text)
        txt_summary.configure(state="disabled")

    def _clear_tree(tree: ttk.Treeview) -> None:
        for item in tree.get_children(""):
            tree.delete(item)

    def _fmt_cell(key: str, value: Any) -> str:
        if value is None:
            return "-"
        numeric_keys = {
            "delta_vs_full_pct",
            "full_baseline_ms",
            "best_composed_ms",
            "score_spearman",
            "latency_spearman",
            "avg_rank",
            "score_pred",
            "cut_mib",
            "part1_partition_time_s",
            "part1_allocation_time_s",
            "part1_compilation_time_s",
            "part1_elapsed_s",
            "part2_partition_time_s",
            "part2_allocation_time_s",
            "part2_compilation_time_s",
            "part2_elapsed_s",
            "direct_hailo_composed_ms",
            "left_full_baseline_ms",
            "right_full_baseline_ms",
            "full_delta_ms",
            "full_delta_pct",
            "left_best_composed_ms",
            "right_best_composed_ms",
            "best_delta_ms",
            "best_delta_pct",
            "left_score_spearman",
            "right_score_spearman",
            "score_spearman_delta",
            "left_latency_spearman",
            "right_latency_spearman",
            "latency_spearman_delta",
            "left_avg_rank",
            "right_avg_rank",
            "avg_rank_delta",
            "left_score_pred",
            "right_score_pred",
            "left_cut_mib",
            "right_cut_mib",
            "context_delta",
            "left_direct_hailo_composed_ms",
            "right_direct_hailo_composed_ms",
            "latency_delta_ms",
            "candidate_count",
            "best_cycle_ms_opt",
            "best_cycle_ms_cons",
            "best_pipeline_fps_opt",
            "best_pipeline_fps_cons",
            "gain_vs_stage1_full_pct",
            "gain_vs_stage2_full_pct",
            "gain_vs_best_single_full_pct",
            "latency_delta_vs_best_single_full_ms",
            "latency_delta_vs_best_single_full_pct",
            "stage_balance",
            "cycle_ms_cons",
            "cycle_ms_opt",
            "pipeline_fps_cons",
            "pipeline_fps_opt",
            "composed_mean_ms",
            "part1_mean_ms",
            "part2_mean_ms",
            "overhead_ms",
            "full_stage1_ms",
            "full_stage2_ms",
            "full_best_single_ms",
            "left_best_cycle_ms_cons",
            "right_best_cycle_ms_cons",
            "left_best_pipeline_fps_cons",
            "right_best_pipeline_fps_cons",
            "fps_delta_abs",
            "fps_delta_pct",
            "left_gain_vs_best_single_full_pct",
            "right_gain_vs_best_single_full_pct",
            "gain_delta_pct",
            "left_latency_delta_vs_best_single_full_ms",
            "right_latency_delta_vs_best_single_full_ms",
            "latency_ms",
            "fps_equiv",
            "predicted_score",
            "actual_rank",
            "measured_streaming_fps",
            "streaming_efficiency_pct",
            "residual_overhead_ms",
            "ideal_bottleneck_fps",
            "stage1_fps_equiv",
            "stage2_fps_equiv",
            "predicted_rank_old",
            "predicted_rank_uncal",
            "predicted_rank_cal",
            "predicted_rank_new",
            "predicted_score_old",
            "hailo_feasibility_risk",
            "hailo_interface_penalty",
            "predicted_handover_ms_uncal",
            "predicted_handover_ms_cal",
            "predicted_handover_ms",
            "predicted_stream_fps_uncal",
            "predicted_stream_fps_cal",
            "predicted_stream_fps",
            "rank_error_old_abs",
            "rank_error_uncal_abs",
            "rank_error_cal_abs",
            "rank_error_new_abs",
        }
        if key in numeric_keys:
            try:
                return f"{float(value):.2f}"
            except Exception:
                return str(value)
        if key in {
            "part2_single_context",
            "both_parts_single_context",
            "best_boundary_changed",
            "left_part2_single_context",
            "right_part2_single_context",
            "single_context_changed",
        }:
            return "yes" if bool(value) else "no"
        return str(value)

    def _populate_research_cards(report: BenchmarkAnalysisReport, inter_analysis) -> None:
        cards_data = research_summary_cards(report, inter_analysis)
        metric = metric_audit_summary(report, inter_analysis, use_calibration=_use_throughput_calibration())
        objective_name = str(report.summary.get('objective') or 'throughput').strip()
        try:
            objective_badge.set(text=objective_name, level=_objective_badge_level(objective_name))
            calibration_badge.set(text=('CAL' if _use_throughput_calibration() else 'RAW'), level=('ok' if _use_throughput_calibration() else 'idle'))
        except Exception:
            pass
        objective_slug = objective_name.lower()
        if objective_slug.startswith('lat'):
            card_vars['best_split_title'].set('Best latency split')
            card_vars['delta_title'].set('Δ latency vs best full')
        elif objective_slug.startswith('hailo'):
            card_vars['best_split_title'].set('Best Hailo-feasible split')
            card_vars['delta_title'].set('Δ throughput vs best full')
        elif objective_slug.startswith('bal'):
            card_vars['best_split_title'].set('Best balanced split')
            card_vars['delta_title'].set('Δ throughput vs best full')
        else:
            card_vars['best_split_title'].set('Best throughput split')
            card_vars['delta_title'].set('Δ throughput vs best full')
        card_vars['quality_title'].set(f'Prediction quality ({objective_name})')

        card_vars['best_full_value'].set(
            '-' if cards_data.get('best_full_fps') is None else f"{cards_data.get('best_full_fps'):.2f} FPS"
        )
        card_vars['best_full_detail'].set(str(cards_data.get('best_full_label') or '-'))
        card_vars['best_split_value'].set(
            '-' if cards_data.get('best_split_fps') is None else f"{cards_data.get('best_split_fps'):.2f} FPS"
        )
        card_vars['best_split_detail'].set(f"{cards_data.get('best_split_label') or '-'} · objective: {objective_name}")
        delta = cards_data.get('delta_vs_best_full_pct')
        card_vars['delta_value'].set('-' if delta is None else f"{delta:+.1f}%")
        card_vars['delta_detail'].set(f'Best split under objective {objective_name} relative to best full deployment')
        rho = cards_data.get('spearman_rank')
        tau = cards_data.get('kendall_rank')
        rank_txt = []
        if rho is not None:
            rank_txt.append(f"old ρ {rho:.3f}")
        if metric.get('uncal_spearman_rank') is not None:
            rank_txt.append(f"raw ρ {metric.get('uncal_spearman_rank'):.3f}")
        if metric.get('cal_spearman_rank') is not None:
            rank_txt.append(f"cal ρ {metric.get('cal_spearman_rank'):.3f}")
        card_vars['quality_value'].set(' · '.join(rank_txt) if rank_txt else '-')
        active_label = 'cal' if _use_throughput_calibration() else 'raw'
        detail = (
            f"best split old/raw/cal rank: {cards_data.get('best_split_predicted_rank') or '-'} / {metric.get('best_uncal_rank') or '-'} / {metric.get('best_cal_rank') or '-'}"
        )
        if metric.get('calibration_profile'):
            detail += f" · profile {metric.get('calibration_profile')}"
        detail += f" · active {active_label}"
        if (_use_throughput_calibration() and metric.get('cal_top3_hit')) or ((not _use_throughput_calibration()) and metric.get('uncal_top3_hit')):
            detail += ' · throughput Top-3 hit'
        card_vars['quality_detail'].set(detail)

    def _populate_research_trees(report: BenchmarkAnalysisReport, inter_analysis) -> None:
        _clear_tree(tree_best)
        for row in research_best_full_vs_split_rows(report, inter_analysis):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in best_cols)
            tree_best.insert('', 'end', values=vals)
        _clear_tree(tree_audit)
        for row in research_prediction_audit_rows(report, inter_analysis, limit=25):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in audit_cols)
            tree_audit.insert('', 'end', values=vals)
        _clear_tree(tree_stage)
        for row in research_stage_breakdown_rows(inter_analysis, limit=25):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in stage_cols)
            tree_stage.insert('', 'end', values=vals)

    def _populate_metric_audit_tree(report: BenchmarkAnalysisReport, inter_analysis) -> None:
        _clear_tree(tree_metric)
        summary = metric_audit_summary(report, inter_analysis, use_calibration=_use_throughput_calibration())
        metric_summary_var.set(
            f"old ρ/τ: {_fmt_cell('score_spearman', summary.get('old_spearman_rank'))} / {_fmt_cell('score_spearman', summary.get('old_kendall_rank'))} · "
            f"raw ρ/τ: {_fmt_cell('score_spearman', summary.get('uncal_spearman_rank'))} / {_fmt_cell('score_spearman', summary.get('uncal_kendall_rank'))} · "
            f"cal ρ/τ: {_fmt_cell('score_spearman', summary.get('cal_spearman_rank'))} / {_fmt_cell('score_spearman', summary.get('cal_kendall_rank'))}"
            + (f" · profile {summary.get('calibration_profile')}" if summary.get('calibration_profile') else "")
            + (" · active cal" if _use_throughput_calibration() else " · active raw")
        )
        metric_detail_var.set(
            f"Top-1 old/raw/cal: {'yes' if summary.get('old_top1_hit') else 'no'} / {'yes' if summary.get('uncal_top1_hit') else 'no'} / {'yes' if summary.get('cal_top1_hit') else 'no'} · "
            f"Top-3 old/raw/cal: {'yes' if summary.get('old_top3_hit') else 'no'} / {'yes' if summary.get('uncal_top3_hit') else 'no'} / {'yes' if summary.get('cal_top3_hit') else 'no'} · "
            f"best b{summary.get('best_boundary') or '-'}"
        )
        for row in metric_audit_rows(report, inter_analysis, limit=40, use_calibration=_use_throughput_calibration()):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in metric_cols)
            old_err = row.get('rank_error_old_abs')
            new_err = row.get('rank_error_cal_abs') if _use_throughput_calibration() else row.get('rank_error_uncal_abs')
            tag = ''
            try:
                if old_err is not None and new_err is not None and float(new_err) < float(old_err):
                    tag = 'metric_better'
                elif old_err is not None and new_err is not None and float(new_err) > float(old_err):
                    tag = 'metric_worse'
            except Exception:
                tag = ''
            tree_metric.insert('', 'end', values=vals, tags=((tag,) if tag else ()))

    def _export_research_summary(report: BenchmarkAnalysisReport, inter_analysis, output_dir: Path) -> Dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, Path] = {}
        tables = {
            'benchmark_analysis_research_best_vs_split.csv': research_best_full_vs_split_rows(report, inter_analysis),
            'benchmark_analysis_research_prediction_audit.csv': research_prediction_audit_rows(report, inter_analysis, limit=None),
            'benchmark_analysis_research_stage_breakdown.csv': research_stage_breakdown_rows(inter_analysis, limit=None),
            'benchmark_analysis_metric_audit.csv': metric_audit_rows(report, inter_analysis, limit=None, use_calibration=_use_throughput_calibration()),
        }
        for filename, rows in tables.items():
            out_path = output_dir / filename
            if rows:
                with out_path.open('w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
            else:
                out_path.write_text('', encoding='utf-8')
            paths[filename] = out_path
        fig = build_interleaving_residual_overhead_figure(inter_analysis)
        for ext in ('png', 'pdf', 'svg'):
            out_path = output_dir / f'benchmark_analysis_interleaving_residual_overhead.{ext}'
            fig.savefig(out_path, bbox_inches='tight')
            paths[f'interleaving_residual_overhead_{ext}'] = out_path
        for stem, fig in {
            'benchmark_analysis_metric_audit_rank': build_metric_audit_rank_figure(report, inter_analysis, use_calibration=_use_throughput_calibration()),
            'benchmark_analysis_metric_audit_fps': build_metric_audit_fps_figure(report, inter_analysis, use_calibration=_use_throughput_calibration()),
        }.items():
            for ext in ('png', 'pdf', 'svg'):
                out_path = output_dir / f'{stem}.{ext}'
                fig.savefig(out_path, bbox_inches='tight')
                paths[f'{stem}_{ext}'] = out_path
        return paths

    def _populate_provider_tree(report: BenchmarkAnalysisReport) -> None:
        _clear_tree(tree_provider)
        for row in provider_summary_rows(report):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in provider_cols)
            tree_provider.insert("", "end", values=vals)

    def _populate_candidate_tree(report: BenchmarkAnalysisReport) -> None:
        _clear_tree(tree_cand)
        for row in candidate_summary_rows(report, limit=20):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in cand_cols)
            tree_cand.insert("", "end", values=vals)

    def _populate_hailo_outlook_tree(report: BenchmarkAnalysisReport) -> None:
        _clear_tree(tree_hailo_outlook)
        summary = hailo_outlook_summary(report)
        rows = hailo_outlook_rows(report, limit=20)
        if not rows:
            hailo_outlook_summary_var.set("Keine Hailo-Outlook-Daten verfügbar.")
            hailo_outlook_detail_var.set("Dieses Benchmark-Set enthält noch keine Hailo-Risikometriken. Ein neu erzeugtes Benchmark-Set mit dem aktuellen Tool exportiert sie direkt.")
            return
        top_boundary = summary.get("top_boundary")
        likely = int(summary.get("likely_single_context_count") or 0)
        total = int(summary.get("candidate_count") or 0)
        avg_risk = summary.get("avg_risk_score")
        hailo_outlook_summary_var.set(
            f"Top Hailo outlook: b{int(top_boundary)} · likely 1-context {likely}/{total} · avg risk {avg_risk:.2f}"
            if top_boundary is not None and avg_risk is not None
            else "Hailo outlook loaded."
        )
        hailo_outlook_detail_var.set(
            f"Risk bands low/medium/high: {int(summary.get('low_risk_count') or 0)}/{int(summary.get('medium_risk_count') or 0)}/{int(summary.get('high_risk_count') or 0)}. "
            "Nutze diese Sicht, um compile-freundliche und wahrscheinlich single-context-fähige Boundaries schnell zu erkennen."
        )
        for row in rows:
            vals = (
                row.get("rank"),
                f"b{int(row.get('boundary'))}" if row.get("boundary") is not None else "",
                row.get("hailo_part2_marker") or '-',
                _fmt_cell("compile_risk_score", row.get("compile_risk_score")),
                row.get("risk_band"),
                (f"{100.0 * float(row.get('single_context_probability')):.0f}%" if row.get("single_context_probability") is not None else ""),
                _fmt_cell("cut_mib", row.get("cut_mib")),
                _fmt_cell("score_pred", row.get("score_pred")),
                _fmt_cell("avg_rank", row.get("avg_rank")),
                _fmt_cell("providers_present", row.get("providers_present")),
                row.get("recommendation") or "",
            )
            risk_band = str(row.get("risk_band") or "").strip().lower()
            tag = f"risk_{risk_band}" if risk_band in {"low", "medium", "high"} else ""
            tree_hailo_outlook.insert("", "end", values=vals, tags=((tag,) if tag else ()))

    def _populate_hailo_fallback_tree(report: BenchmarkAnalysisReport) -> None:
        _clear_tree(tree_hailo_fallback)
        summary = hailo_part2_fallback_summary(report)
        rows = hailo_part2_fallback_rows(report, limit=25)
        if not rows:
            hailo_fallback_summary_var.set("Keine Hailo-Part2-Fallback-Kandidaten vorhanden.")
            hailo_fallback_detail_var.set("Dieses Benchmark-Set verwendet keine suggested end-node Strategie oder exportiert die Metadaten noch nicht.")
            return
        hailo_fallback_summary_var.set(
            f"Fallback-Kandidaten: {int(summary.get('fallback_count') or 0)}/{int(summary.get('candidate_count') or 0)} · "
            f"Top b{int(summary.get('top_boundary')) if summary.get('top_boundary') is not None else '?'} · "
            f"avg rank {float(summary.get('avg_rank')):.2f}"
            if summary.get('avg_rank') is not None else
            f"Fallback-Kandidaten: {int(summary.get('fallback_count') or 0)}/{int(summary.get('candidate_count') or 0)}"
        )
        hailo_fallback_detail_var.set(
            f"Likely 1-context trotz Fallback: {int(summary.get('single_context_likely_count') or 0)}. "
            "Die Eff. outputs-Spalte zeigt, auf welche intermediate Part2-Ausgänge der Build gekürzt wurde."
        )
        for row in rows:
            vals = (
                f"b{int(row.get('boundary'))}" if row.get('boundary') is not None else '',
                row.get('marker') or '-',
                row.get('strategy_label') or row.get('strategy') or '',
                row.get('effective_outputs_text') or '',
                _fmt_cell('avg_rank', row.get('avg_rank')),
                row.get('best_provider') or '',
                _fmt_cell('best_rank', row.get('best_rank')),
                (f"{100.0 * float(row.get('single_context_probability')):.0f}%" if row.get('single_context_probability') is not None else ''),
                _fmt_cell('compile_risk_score', row.get('compile_risk_score')),
                row.get('recommendation') or '',
            )
            tree_hailo_fallback.insert('', 'end', values=vals)

    def _populate_hailo_tree(report: BenchmarkAnalysisReport) -> None:
        _clear_tree(tree_hailo)
        for row in hailo_context_rows(report):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in hailo_cols)
            tree_hailo.insert("", "end", values=vals)

    def _populate_inter_tree(rows: list[dict[str, Any]]) -> None:
        _clear_tree(tree_inter)
        for row in rows:
            vals = tuple(_fmt_cell(col, row.get(col)) for col in inter_cols)
            tree_inter.insert("", "end", values=vals)

    def _populate_inter_candidate_tree(rows: list[dict[str, Any]]) -> None:
        _clear_tree(tree_inter_cand)
        for row in rows:
            vals = tuple(_fmt_cell(col, row.get(col)) for col in inter_cand_cols)
            tree_inter_cand.insert("", "end", values=vals)

    def _populate_provider_cmp_tree(comparison: BenchmarkComparisonReport) -> None:
        _clear_tree(tree_research_cmp)
        _clear_tree(tree_provider_cmp)
        for row in comparison_provider_rows(comparison):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in provider_cmp_cols)
            tree_provider_cmp.insert("", "end", values=vals)

    def _populate_candidate_cmp_tree(comparison: BenchmarkComparisonReport) -> None:
        _clear_tree(tree_cand_cmp)
        for row in comparison_candidate_rows(comparison, limit=25):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in cand_cmp_cols)
            tree_cand_cmp.insert("", "end", values=vals)

    def _populate_hailo_cmp_tree(comparison: BenchmarkComparisonReport) -> None:
        _clear_tree(tree_hailo_cmp)
        for row in comparison_hailo_rows(comparison):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in hailo_cmp_cols)
            tree_hailo_cmp.insert("", "end", values=vals)

    def _populate_inter_cmp_tree(rows: list[dict[str, Any]]) -> None:
        _clear_tree(tree_inter_cmp)
        for row in rows:
            vals = tuple(_fmt_cell(col, row.get(col)) for col in inter_cmp_cols)
            tree_inter_cmp.insert("", "end", values=vals)

    def _populate_research_cmp_tree(
        left_report: BenchmarkAnalysisReport,
        left_inter,
        right_report: BenchmarkAnalysisReport,
        right_inter,
    ) -> None:
        _clear_tree(tree_research_cmp)
        rows = research_comparison_rows(left_report, left_inter, right_report, right_inter, use_calibration=_use_throughput_calibration())
        summary = metric_audit_comparison_summary(left_report, left_inter, right_report, right_inter, use_calibration=_use_throughput_calibration())
        mode = 'calibrated' if _use_throughput_calibration() else 'raw'
        research_cmp_summary_var.set(
            f"Research-Vergleich · active {mode}" + (f" · profile {summary.get('calibration_profile')}" if summary.get('calibration_profile') else '')
        )
        research_cmp_detail_var.set(
            f"Δ old/raw/cal Spearman: {_fmt_cell('score_spearman', summary.get('old_spearman_rank_delta'))} / {_fmt_cell('score_spearman', summary.get('uncal_spearman_rank_delta'))} / {_fmt_cell('score_spearman', summary.get('cal_spearman_rank_delta'))} · common cases: {int(summary.get('common_case_count') or 0)}"
        )
        for row in rows:
            vals = tuple(_fmt_cell(col, row.get(col)) for col in research_cmp_cols)
            tree_research_cmp.insert('', 'end', values=vals)

    def _populate_metric_cmp_tree(
        left_report: BenchmarkAnalysisReport,
        left_inter,
        right_report: BenchmarkAnalysisReport,
        right_inter,
    ) -> None:
        _clear_tree(tree_metric_cmp)
        summary = metric_audit_comparison_summary(left_report, left_inter, right_report, right_inter, use_calibration=_use_throughput_calibration())
        metric_cmp_summary_var.set(
            f"old Δρ/τ: {_fmt_cell('score_spearman', summary.get('old_spearman_rank_delta'))} / {_fmt_cell('score_spearman', summary.get('old_kendall_rank_delta'))} · "
            f"raw Δρ/τ: {_fmt_cell('score_spearman', summary.get('uncal_spearman_rank_delta'))} / {_fmt_cell('score_spearman', summary.get('uncal_kendall_rank_delta'))} · "
            f"cal Δρ/τ: {_fmt_cell('score_spearman', summary.get('cal_spearman_rank_delta'))} / {_fmt_cell('score_spearman', summary.get('cal_kendall_rank_delta'))}"
            + (f" · profile {summary.get('calibration_profile')}" if summary.get('calibration_profile') else "")
            + (" · active cal" if _use_throughput_calibration() else " · active raw")
        )
        metric_cmp_detail_var.set(
            f"Top-1 old/raw/cal A→B: {'yes' if summary.get('left_old_top1_hit') else 'no'}→{'yes' if summary.get('right_old_top1_hit') else 'no'} / "
            f"{'yes' if summary.get('left_uncal_top1_hit') else 'no'}→{'yes' if summary.get('right_uncal_top1_hit') else 'no'} / "
            f"{'yes' if summary.get('left_cal_top1_hit') else 'no'}→{'yes' if summary.get('right_cal_top1_hit') else 'no'} · "
            f"mean |error| old/raw/cal Δ: {_fmt_cell('score_spearman', summary.get('old_mean_abs_rank_error_delta'))} / {_fmt_cell('score_spearman', summary.get('uncal_mean_abs_rank_error_delta'))} / {_fmt_cell('score_spearman', summary.get('cal_mean_abs_rank_error_delta'))} · "
            f"gemeinsame Fälle: {int(summary.get('common_case_count') or 0)}"
        )
        for row in metric_audit_comparison_rows(left_report, left_inter, right_report, right_inter, limit=40, use_calibration=_use_throughput_calibration()):
            vals = tuple(_fmt_cell(col, row.get(col)) for col in metric_cmp_cols)
            tag = ''
            try:
                old_delta = row.get('old_rank_error_abs_delta')
                new_delta = row.get('cal_rank_error_abs_delta') if _use_throughput_calibration() else row.get('uncal_rank_error_abs_delta')
                if old_delta is not None and new_delta is not None and float(new_delta) < float(old_delta):
                    tag = 'metric_cmp_better'
                elif old_delta is not None and new_delta is not None and float(new_delta) > float(old_delta):
                    tag = 'metric_cmp_worse'
            except Exception:
                tag = ''
            tree_metric_cmp.insert('', 'end', values=vals, tags=((tag,) if tag else ()))

    def _render_plot(host: ttk.Frame, key: str, figure: Figure) -> None:
        old_canvas = state["canvases"].get(key)
        if old_canvas is not None:
            try:
                old_canvas.get_tk_widget().destroy()
            except Exception:
                pass
        old_toolbar = state["toolbars"].get(key)
        if old_toolbar is not None:
            try:
                old_toolbar.destroy()
            except Exception:
                pass
        canvas = FigureCanvasTkAgg(figure, master=host)
        widget = canvas.get_tk_widget()
        widget.grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(canvas, host, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")
        canvas.draw_idle()
        state["figures"][key] = figure
        state["canvases"][key] = canvas
        state["toolbars"][key] = toolbar

    def _clear_comparison_views(message: str = "Noch kein Vergleich geladen.") -> None:
        state["comparison"] = None
        state["interleaving_comparison"] = None
        metric_cmp_summary_var.set("Noch kein Metric-Audit-Vergleich geladen.")
        metric_cmp_detail_var.set("Der Vergleich zeigt, ob sich die throughput-orientierte Metrik zwischen Lauf A und Lauf B verbessert oder verschlechtert.")
        _clear_tree(tree_provider_cmp)
        _clear_tree(tree_cand_cmp)
        _clear_tree(tree_hailo_cmp)
        _clear_tree(tree_inter_cmp)
        _clear_tree(tree_metric_cmp)
        research_cmp_summary_var.set(message)
        research_cmp_detail_var.set('Kein A↔B-Research-Vergleich geladen.')
        metric_cmp_summary_var.set(message)
        metric_cmp_detail_var.set('Kein A↔B-Metric-Audit geladen.')
        placeholders = {
            "comparison_provider_latency": "Provider-Latenzvergleich",
            "comparison_predictor_delta": "Prognosegüte-Vergleich",
            "comparison_candidate_rank_shift": "Kandidaten-Verschiebung",
            "comparison_hailo_context_delta": "Hailo Context-Vergleich",
            "comparison_hailo_latency_delta": "Hailo-Latenzvergleich",
            "comparison_interleaving_fps_delta": "Interleaving FPS-Vergleich",
            "comparison_metric_audit_rank_error": "Metric Audit-Vergleich",
            "interleaving_residual_overhead": "Cut vs residual overhead",
            "metric_audit_rank": "Metric audit: rank",
            "metric_audit_fps": "Metric audit: throughput",
        }
        try:
            calibration_badge.set(text=('CAL' if _use_throughput_calibration() else 'RAW'), level=('ok' if _use_throughput_calibration() else 'idle'))
        except Exception:
            pass
        for key, title in placeholders.items():
            _render_plot(plot_hosts[key], key, _placeholder_figure(title, message))

    def _render_report(report: BenchmarkAnalysisReport, summary_text: Optional[str] = None, status_text: Optional[str] = None, inter_analysis=None) -> None:
        state["report"] = report
        if inter_analysis is None:
            inter_analysis = compute_interleaving_analysis(report)
        state["interleaving"] = inter_analysis
        if status_text is None:
            status_text = (
                f"Analyse geladen: {report.source.display_name} | Cases: {report.summary.get('generated_cases', report.summary.get('case_count', '-'))} | Provider: {len(report.providers)} | Objective: {report.summary.get('objective') or '-'}"
            )
        status_var.set(status_text)
        combined_summary = summary_text or report.summary_markdown
        if summary_text is None and inter_analysis.summary_markdown:
            combined_summary = combined_summary.rstrip() + "\n\n" + inter_analysis.summary_markdown
        _set_summary_text(combined_summary)
        _populate_research_cards(report, inter_analysis)
        _populate_research_trees(report, inter_analysis)
        _populate_metric_audit_tree(report, inter_analysis)
        _populate_provider_tree(report)
        _populate_candidate_tree(report)
        _populate_hailo_outlook_tree(report)
        _populate_hailo_fallback_tree(report)
        _populate_hailo_tree(report)
        _populate_inter_tree(interleaving_provider_rows(inter_analysis))
        _populate_inter_candidate_tree(interleaving_candidate_rows(inter_analysis, limit=40))
        for key, figure in build_benchmark_analysis_figures(report).items():
            if key in plot_hosts:
                _render_plot(plot_hosts[key], key, figure)
        _render_plot(plot_hosts["interleaving_gain"], "interleaving_gain", build_interleaving_gain_figure(inter_analysis))
        _render_plot(plot_hosts["interleaving_tradeoff"], "interleaving_tradeoff", build_interleaving_tradeoff_figure(inter_analysis))
        _render_plot(plot_hosts["interleaving_residual_overhead"], "interleaving_residual_overhead", build_interleaving_residual_overhead_figure(inter_analysis))
        _render_plot(plot_hosts["metric_audit_rank"], "metric_audit_rank", build_metric_audit_rank_figure(report, inter_analysis, use_calibration=_use_throughput_calibration()))
        _render_plot(plot_hosts["metric_audit_fps"], "metric_audit_fps", build_metric_audit_fps_figure(report, inter_analysis, use_calibration=_use_throughput_calibration()))
        body.select(research_tab)

    def _render_comparison(comparison: BenchmarkComparisonReport, inter_cmp=None) -> None:
        state["comparison"] = comparison
        if inter_cmp is None:
            inter_cmp = compare_interleaving_reports(comparison.left, comparison.right)
        state["interleaving_comparison"] = inter_cmp
        left_inter = compute_interleaving_analysis(comparison.left)
        right_inter = compute_interleaving_analysis(comparison.right)
        combined_summary = comparison.summary_markdown
        if inter_cmp.summary_markdown:
            combined_summary = combined_summary.rstrip() + "\n\n" + inter_cmp.summary_markdown
        _render_report(
            comparison.left,
            summary_text=combined_summary,
            status_text=(
                f"Vergleich geladen: A={comparison.left.source.display_name} | B={comparison.right.source.display_name} | "
                f"gemeinsame Provider: {len(set(comparison.left.provider_tags) & set(comparison.right.provider_tags))}"
            ),
        )
        _populate_research_cmp_tree(comparison.left, left_inter, comparison.right, right_inter)
        _populate_provider_cmp_tree(comparison)
        _populate_candidate_cmp_tree(comparison)
        _populate_hailo_cmp_tree(comparison)
        _populate_inter_cmp_tree(comparison_interleaving_rows(inter_cmp))
        _populate_metric_cmp_tree(comparison.left, left_inter, comparison.right, right_inter)
        for key, figure in build_benchmark_comparison_figures(comparison).items():
            if key in plot_hosts:
                _render_plot(plot_hosts[key], key, figure)
        _render_plot(plot_hosts["comparison_interleaving_fps_delta"], "comparison_interleaving_fps_delta", build_comparison_interleaving_fps_delta_figure(inter_cmp))
        _render_plot(plot_hosts["comparison_metric_audit_rank_error"], "comparison_metric_audit_rank_error", build_comparison_metric_audit_rank_error_figure(comparison.left, left_inter, comparison.right, right_inter, use_calibration=_use_throughput_calibration()))
        body.select(summary_tab)

    def _rerender_for_calibration_toggle(*_args) -> None:
        try:
            if state.get('comparison') is not None:
                _render_comparison(state['comparison'], state.get('interleaving_comparison'))
            elif state.get('report') is not None:
                _render_report(state['report'], inter_analysis=state.get('interleaving'))
        except Exception:
            logger.exception('Failed to rerender benchmark analysis after calibration toggle')

    try:
        var_use_cal.trace_add('write', _rerender_for_calibration_toggle)
    except Exception:
        pass

    def _run_analysis() -> None:
        _apply_throughput_calibration_setting()
        source = var_source.get().strip()
        compare_source = var_compare_source.get().strip()
        if not source:
            messagebox.showerror("Benchmark-Analyse", "Bitte zuerst Quelle A auswählen.")
            return
        cache_base = _analysis_cache_base(app)
        cache_base.mkdir(parents=True, exist_ok=True)
        btn_analyze.configure(state="disabled")
        status_var.set("Analysiere Benchmark-Ergebnisse…")

        def _worker() -> None:
            try:
                service = BenchmarkAnalysisService(cache_base)
                if compare_source:
                    loaded_cmp = service.load_comparison(source, compare_source)
                    frame.after(0, lambda: _render_comparison(loaded_cmp.comparison, loaded_cmp.interleaving_comparison))
                else:
                    loaded = service.load_single(source)
                    frame.after(0, lambda: _render_report(loaded.report, inter_analysis=loaded.interleaving))
                    frame.after(0, lambda: _clear_comparison_views())
            except Exception as exc:
                logger.exception("Benchmark analysis failed")
                frame.after(0, lambda: messagebox.showerror("Benchmark-Analyse", str(exc)))
                frame.after(0, lambda: status_var.set(f"Analyse fehlgeschlagen: {exc}"))
            finally:
                frame.after(0, lambda: btn_analyze.configure(state="normal"))

        threading.Thread(target=_worker, daemon=True).start()

    def _export_report() -> None:
        _apply_throughput_calibration_setting()
        comparison: Optional[BenchmarkComparisonReport] = state.get("comparison")
        report: Optional[BenchmarkAnalysisReport] = state.get("report")
        if comparison is None and report is None:
            messagebox.showerror("Benchmark-Analyse", "Es ist noch keine Analyse geladen.")
            return
        base_root = comparison.left.source.results_root if comparison is not None else report.source.results_root  # type: ignore[union-attr]
        out_dir = filedialog.askdirectory(title="Analyse-Report exportieren", initialdir=str(base_root))
        if not out_dir:
            return
        try:
            paths = {}
            service = BenchmarkAnalysisService(_analysis_cache_base(app))
            if comparison is not None:
                inter_cmp = state.get("interleaving_comparison")
                loaded_cmp = LoadedBenchmarkComparison(comparison=comparison, interleaving_comparison=inter_cmp)
                out_dir_path = Path(out_dir)
                paths.update({f"analysis_{k}": v for k, v in service.export_comparison(loaded_cmp, out_dir_path).items()})
                left_inter = compute_interleaving_analysis(comparison.left)
                right_inter = compute_interleaving_analysis(comparison.right)
                paths.update({f"metric_cmp_{k}": v for k, v in export_metric_audit_comparison(comparison.left, left_inter, comparison.right, right_inter, out_dir_path, use_calibration=_use_throughput_calibration()).items()})
            else:
                inter_analysis = state.get("interleaving")
                loaded = LoadedBenchmarkAnalysis(report=report, interleaving=inter_analysis)  # type: ignore[arg-type]
                out_dir_path = Path(out_dir)
                paths.update({f"analysis_{k}": v for k, v in service.export_single(loaded, out_dir_path).items()})
                if inter_analysis is not None:
                    paths.update({f"research_{k}": v for k, v in _export_research_summary(report, inter_analysis, out_dir_path).items()})
        except Exception as exc:
            logger.exception("Failed to export benchmark analysis")
            messagebox.showerror("Benchmark-Analyse", f"Export fehlgeschlagen: {exc}")
            return
        messagebox.showinfo(
            "Benchmark-Analyse",
            "Analyse exportiert nach:\n\n" + "\n".join(str(p) for p in paths.values()),
        )

    def _export_publication_report() -> None:
        _apply_throughput_calibration_setting()
        comparison: Optional[BenchmarkComparisonReport] = state.get("comparison")
        report: Optional[BenchmarkAnalysisReport] = state.get("report")
        if comparison is None and report is None:
            messagebox.showerror("Benchmark-Analyse", "Es ist noch keine Analyse geladen.")
            return
        base_root = comparison.left.source.results_root if comparison is not None else report.source.results_root  # type: ignore[union-attr]
        out_dir = filedialog.askdirectory(title="Publication-Export erzeugen", initialdir=str(base_root))
        if not out_dir:
            return
        try:
            service = BenchmarkAnalysisService(_analysis_cache_base(app))
            out_dir_path = Path(out_dir)
            paths = {}
            if comparison is not None:
                inter_cmp = state.get('interleaving_comparison')
                loaded_cmp = LoadedBenchmarkComparison(comparison=comparison, interleaving_comparison=inter_cmp)
                paths.update({f'publication_{k}': v for k, v in service.export_publication_comparison(loaded_cmp, out_dir_path, use_calibration=_use_throughput_calibration()).items()})
            else:
                inter_analysis = state.get('interleaving')
                loaded = LoadedBenchmarkAnalysis(report=report, interleaving=inter_analysis)  # type: ignore[arg-type]
                paths.update({f'publication_{k}': v for k, v in service.export_publication_single(loaded, out_dir_path, use_calibration=_use_throughput_calibration()).items()})
        except Exception as exc:
            logger.exception("Failed to export publication bundle")
            messagebox.showerror("Benchmark-Analyse", f"Publication-Export fehlgeschlagen: {exc}")
            return
        messagebox.showinfo("Benchmark-Analyse", "Publication-Export geschrieben nach:\n\n" + "\n".join(str(p) for p in paths.values()))

    btn_analyze.configure(command=_run_analysis)
    btn_export.configure(command=_export_report)
    btn_pub.configure(command=_export_publication_report)
    _clear_comparison_views()

    return frame
