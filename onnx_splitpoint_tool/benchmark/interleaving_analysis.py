"""Interleaving / throughput analysis for heterogeneous split benchmarks.

This module estimates steady-state pipeline throughput for split runs that use
*different* stage1/stage2 accelerators. The goal is to capture the actual
benefit of split execution for papers and engineering decisions: a split may
increase end-to-end latency while still increasing steady-state FPS once stage1
and stage2 are interleaved.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from matplotlib.figure import Figure

from .analysis import BenchmarkAnalysisReport, BenchmarkComparisonReport, candidate_summary_rows
from ..objective_scoring import active_throughput_calibration_profile as _active_calibration_profile, as_float as _score_as_float, as_int as _score_as_int, feature_count as _score_feature_count, hailo_feasibility_risk as _hailo_feasibility_risk, hailo_interface_penalty as _hailo_interface_penalty, predicted_handover_ms as _predicted_handover_ms_shared, predicted_stream_cycle_ms as _predicted_stream_cycle_ms_shared, predicted_stream_fps as _predicted_stream_fps_shared


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _as_int(v: Any) -> Optional[int]:
    x = _as_float(v)
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _slug(text: Any) -> str:
    s = str(text or "").strip().lower()
    out = []
    last_sep = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_")


def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "-"
    if not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class InterleavingCandidateSummary:
    provider_tag: str
    stage1_provider: str
    stage2_provider: str
    boundary: int
    part1_mean_ms: float
    part2_mean_ms: float
    composed_mean_ms: float
    overhead_ms: float
    cycle_ms_opt: float
    cycle_ms_cons: float
    pipeline_fps_opt: float
    pipeline_fps_cons: float
    stage_balance: float
    full_stage1_ms: Optional[float]
    full_stage2_ms: Optional[float]
    full_best_single_ms: Optional[float]
    gain_vs_stage1_full_pct: Optional[float]
    gain_vs_stage2_full_pct: Optional[float]
    gain_vs_best_single_full_pct: Optional[float]
    gain_vs_stage1_full_fps: Optional[float]
    gain_vs_stage2_full_fps: Optional[float]
    gain_vs_best_single_full_fps: Optional[float]
    latency_delta_vs_stage1_full_ms: Optional[float]
    latency_delta_vs_stage2_full_ms: Optional[float]
    latency_delta_vs_best_single_full_ms: Optional[float]
    latency_delta_vs_best_single_full_pct: Optional[float]
    score_pred: Optional[float]
    cut_mib: Optional[float]
    measured_streaming_fps: Optional[float] = None
    measured_streaming_cycle_ms: Optional[float] = None
    measured_latency_mean_ms: Optional[float] = None
    measured_latency_p50_ms: Optional[float] = None
    measured_latency_p95_ms: Optional[float] = None
    throughput_mode: Optional[str] = None


@dataclass
class InterleavingProviderSummary:
    provider_tag: str
    stage1_provider: str
    stage2_provider: str
    candidate_count: int
    best_boundary: int
    best_cycle_ms_opt: float
    best_cycle_ms_cons: float
    best_pipeline_fps_opt: float
    best_pipeline_fps_cons: float
    best_stage_balance: float
    full_stage1_ms: Optional[float]
    full_stage2_ms: Optional[float]
    full_best_single_ms: Optional[float]
    gain_vs_stage1_full_pct: Optional[float]
    gain_vs_stage2_full_pct: Optional[float]
    gain_vs_best_single_full_pct: Optional[float]
    latency_delta_vs_best_single_full_ms: Optional[float]
    latency_delta_vs_best_single_full_pct: Optional[float]
    score_pred: Optional[float]
    cut_mib: Optional[float]
    measured_streaming_fps: Optional[float] = None
    measured_streaming_cycle_ms: Optional[float] = None
    measured_latency_p50_ms: Optional[float] = None
    throughput_mode: Optional[str] = None


@dataclass
class InterleavingAnalysisReport:
    source_name: str
    provider_summaries: List[InterleavingProviderSummary]
    candidate_summaries: List[InterleavingCandidateSummary]
    summary_markdown: str
    issues: List[str]


@dataclass
class ComparisonInterleavingSummary:
    provider_tag: str
    stage1_provider: str
    stage2_provider: str
    left_candidate_count: int
    right_candidate_count: int
    left_best_boundary: Optional[int]
    right_best_boundary: Optional[int]
    left_best_cycle_ms_cons: Optional[float]
    right_best_cycle_ms_cons: Optional[float]
    left_best_pipeline_fps_cons: Optional[float]
    right_best_pipeline_fps_cons: Optional[float]
    fps_delta_abs: Optional[float]
    fps_delta_pct: Optional[float]
    left_gain_vs_best_single_full_pct: Optional[float]
    right_gain_vs_best_single_full_pct: Optional[float]
    gain_delta_pct: Optional[float]
    left_latency_delta_vs_best_single_full_ms: Optional[float]
    right_latency_delta_vs_best_single_full_ms: Optional[float]
    latency_delta_ms: Optional[float]
    best_boundary_changed: bool


@dataclass
class InterleavingComparisonReport:
    left_name: str
    right_name: str
    provider_summaries: List[ComparisonInterleavingSummary]
    summary_markdown: str
    issues: List[str]


_SUPPORTED_HAILO_TOKENS = ("hailo8", "hailo10", "hailo10h")


def _is_heterogeneous(stage1: str, stage2: str) -> bool:
    return bool(stage1) and bool(stage2) and _slug(stage1) != _slug(stage2)


def _base_fps(full_ms: Optional[float]) -> Optional[float]:
    if full_ms is None or full_ms <= 0:
        return None
    return 1000.0 / float(full_ms)


def _gain_pct(cycle_ms: float, full_ms: Optional[float]) -> Optional[float]:
    if full_ms is None or full_ms <= 0 or cycle_ms <= 0:
        return None
    return ((float(full_ms) / float(cycle_ms)) - 1.0) * 100.0


def _gain_fps(cycle_ms: float, full_ms: Optional[float]) -> Optional[float]:
    if cycle_ms <= 0:
        return None
    pipe_fps = 1000.0 / float(cycle_ms)
    base = _base_fps(full_ms)
    if base is None:
        return None
    return pipe_fps - base


def _delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return float(new) - float(old)


def _delta_pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or abs(float(old)) <= 1e-12:
        return None
    return ((float(new) - float(old)) / float(old)) * 100.0


def _placeholder_figure(title: str, message: str) -> Figure:
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


def _direct_stage_provider_baselines(report: BenchmarkAnalysisReport) -> Dict[str, Dict[str, Any]]:
    provider_summary = {ps.tag: ps for ps in report.providers}
    out: Dict[str, Dict[str, Any]] = {}
    for tag, rows in report.provider_rows.items():
        ps = provider_summary.get(tag)
        if ps is None or ps.full_baseline_ms is None:
            continue
        token: Optional[str] = None
        for row in rows:
            s1 = str(row.get("stage1_provider") or "").strip()
            s2 = str(row.get("stage2_provider") or "").strip()
            if s1 and s1 == s2:
                token = s1
                break
        if not token:
            continue
        cur = out.get(token)
        if cur is None or float(ps.full_baseline_ms) < float(cur["full_ms"]):
            out[token] = {"provider_tag": tag, "full_ms": float(ps.full_baseline_ms)}
    return out


def _valid_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = [
        r
        for r in rows
        if (_as_bool(r.get("final_pass_all")) is not False)
        and _as_int(r.get("boundary")) is not None
        and _as_float(r.get("part1_mean_ms")) is not None
        and _as_float(r.get("part2_mean_ms")) is not None
        and _as_float(r.get("composed_mean_ms")) is not None
    ]
    return selected if selected else [
        r
        for r in rows
        if _as_int(r.get("boundary")) is not None
        and _as_float(r.get("part1_mean_ms")) is not None
        and _as_float(r.get("part2_mean_ms")) is not None
        and _as_float(r.get("composed_mean_ms")) is not None
    ]


def _label_for_provider(stage1: str, stage2: str, tag: str) -> str:
    return f"{stage1}→{stage2}" if _is_heterogeneous(stage1, stage2) else tag


def _effective_streaming_metrics(row: Dict[str, Any], *, cycle_ms_cons: float, composed_ms: float) -> Dict[str, Any]:
    measured_fps = _as_float(row.get("throughput_fps_makespan"))
    measured_cycle = _as_float(row.get("throughput_cycle_est_ms"))
    if (measured_cycle is None or measured_cycle <= 0.0) and measured_fps is not None and measured_fps > 0.0:
        measured_cycle = 1000.0 / float(measured_fps)
    measured_latency_mean = _as_float(row.get("throughput_latency_mean_ms"))
    measured_latency_p50 = _as_float(row.get("throughput_latency_p50_ms"))
    measured_latency_p95 = _as_float(row.get("throughput_latency_p95_ms"))
    mode = str(row.get("throughput_mode") or "").strip() or None
    if measured_cycle is not None and measured_cycle > 0.0:
        effective_cycle = float(measured_cycle)
        effective_fps = float(measured_fps) if measured_fps is not None and measured_fps > 0.0 else (1000.0 / effective_cycle)
        throughput_mode = mode or "measured"
    else:
        effective_cycle = float(cycle_ms_cons)
        effective_fps = 1000.0 / effective_cycle if effective_cycle > 0.0 else None
        throughput_mode = mode or "estimated"
    latency_ref = measured_latency_p50
    if latency_ref is None:
        latency_ref = measured_latency_mean if measured_latency_mean is not None else composed_ms
    return {
        "effective_cycle_ms": effective_cycle,
        "effective_fps": effective_fps,
        "measured_streaming_fps": measured_fps,
        "measured_streaming_cycle_ms": measured_cycle,
        "measured_latency_mean_ms": measured_latency_mean,
        "measured_latency_p50_ms": measured_latency_p50,
        "measured_latency_p95_ms": measured_latency_p95,
        "latency_reference_ms": latency_ref,
        "throughput_mode": throughput_mode,
    }


def compute_interleaving_analysis(report: BenchmarkAnalysisReport) -> InterleavingAnalysisReport:
    baselines = _direct_stage_provider_baselines(report)
    candidates: List[InterleavingCandidateSummary] = []
    issues: List[str] = []
    for tag, rows in report.provider_rows.items():
        if not rows:
            continue
        stage1 = str(rows[0].get("stage1_provider") or "").strip()
        stage2 = str(rows[0].get("stage2_provider") or "").strip()
        if not _is_heterogeneous(stage1, stage2):
            continue
        valid = _valid_rows(rows)
        if not valid:
            issues.append(f"{tag}: keine gültigen part1/part2/composed-Latenzen für Interleaving-Analyse gefunden.")
            continue
        full_stage1_ms = _as_float((baselines.get(stage1) or {}).get("full_ms"))
        full_stage2_ms = _as_float((baselines.get(stage2) or {}).get("full_ms"))
        full_best_single_ms = None
        available = [v for v in (full_stage1_ms, full_stage2_ms) if v is not None and v > 0]
        if available:
            full_best_single_ms = min(available)
        for row in valid:
            boundary = int(_as_int(row.get("boundary")) or 0)
            part1 = float(_as_float(row.get("part1_mean_ms")) or 0.0)
            part2 = float(_as_float(row.get("part2_mean_ms")) or 0.0)
            composed = float(_as_float(row.get("composed_mean_ms")) or 0.0)
            overhead = _as_float(row.get("overhead_ms"))
            if overhead is None:
                overhead = composed - (part1 + part2)
            overhead = float(overhead)
            cycle_opt = max(part1, part2)
            cycle_cons = max(part1 + max(overhead, 0.0), part2)
            if cycle_opt <= 0 or cycle_cons <= 0:
                continue
            fps_opt = 1000.0 / cycle_opt
            stage_balance = min(part1, part2) / max(part1, part2) if max(part1, part2) > 0 else 0.0
            score_pred = _as_float(row.get("score_pred") or row.get("score"))
            cut_mib = _as_float(row.get("cut_mib"))
            eff = _effective_streaming_metrics(row, cycle_ms_cons=cycle_cons, composed_ms=composed)
            eff_cycle = float(eff["effective_cycle_ms"])
            eff_fps = _as_float(eff.get("effective_fps"))
            latency_ref = _as_float(eff.get("latency_reference_ms"))
            candidates.append(
                InterleavingCandidateSummary(
                    provider_tag=tag,
                    stage1_provider=stage1,
                    stage2_provider=stage2,
                    boundary=boundary,
                    part1_mean_ms=part1,
                    part2_mean_ms=part2,
                    composed_mean_ms=composed,
                    overhead_ms=overhead,
                    cycle_ms_opt=cycle_opt,
                    cycle_ms_cons=eff_cycle,
                    pipeline_fps_opt=fps_opt,
                    pipeline_fps_cons=(float(eff_fps) if eff_fps is not None and eff_fps > 0.0 else (1000.0 / eff_cycle)),
                    stage_balance=stage_balance,
                    full_stage1_ms=full_stage1_ms,
                    full_stage2_ms=full_stage2_ms,
                    full_best_single_ms=full_best_single_ms,
                    gain_vs_stage1_full_pct=_gain_pct(eff_cycle, full_stage1_ms),
                    gain_vs_stage2_full_pct=_gain_pct(eff_cycle, full_stage2_ms),
                    gain_vs_best_single_full_pct=_gain_pct(eff_cycle, full_best_single_ms),
                    gain_vs_stage1_full_fps=_gain_fps(eff_cycle, full_stage1_ms),
                    gain_vs_stage2_full_fps=_gain_fps(eff_cycle, full_stage2_ms),
                    gain_vs_best_single_full_fps=_gain_fps(eff_cycle, full_best_single_ms),
                    latency_delta_vs_stage1_full_ms=(latency_ref - full_stage1_ms) if latency_ref is not None and full_stage1_ms is not None else None,
                    latency_delta_vs_stage2_full_ms=(latency_ref - full_stage2_ms) if latency_ref is not None and full_stage2_ms is not None else None,
                    latency_delta_vs_best_single_full_ms=(latency_ref - full_best_single_ms) if latency_ref is not None and full_best_single_ms is not None else None,
                    latency_delta_vs_best_single_full_pct=_delta_pct(latency_ref, full_best_single_ms),
                    score_pred=score_pred,
                    cut_mib=cut_mib,
                    measured_streaming_fps=_as_float(eff.get("measured_streaming_fps")),
                    measured_streaming_cycle_ms=_as_float(eff.get("measured_streaming_cycle_ms")),
                    measured_latency_mean_ms=_as_float(eff.get("measured_latency_mean_ms")),
                    measured_latency_p50_ms=_as_float(eff.get("measured_latency_p50_ms")),
                    measured_latency_p95_ms=_as_float(eff.get("measured_latency_p95_ms")),
                    throughput_mode=(str(eff.get("throughput_mode") or "") or None),
                )
            )
    candidates.sort(key=lambda row: (row.cycle_ms_cons, -(row.gain_vs_best_single_full_pct or float("-inf")), row.boundary))

    provider_groups: Dict[str, List[InterleavingCandidateSummary]] = {}
    for cand in candidates:
        provider_groups.setdefault(cand.provider_tag, []).append(cand)

    provider_summaries: List[InterleavingProviderSummary] = []
    for tag, rows in sorted(provider_groups.items()):
        best = min(rows, key=lambda row: (row.cycle_ms_cons, row.composed_mean_ms, row.boundary))
        provider_summaries.append(
            InterleavingProviderSummary(
                provider_tag=tag,
                stage1_provider=best.stage1_provider,
                stage2_provider=best.stage2_provider,
                candidate_count=len(rows),
                best_boundary=best.boundary,
                best_cycle_ms_opt=best.cycle_ms_opt,
                best_cycle_ms_cons=best.cycle_ms_cons,
                best_pipeline_fps_opt=best.pipeline_fps_opt,
                best_pipeline_fps_cons=best.pipeline_fps_cons,
                best_stage_balance=best.stage_balance,
                full_stage1_ms=best.full_stage1_ms,
                full_stage2_ms=best.full_stage2_ms,
                full_best_single_ms=best.full_best_single_ms,
                gain_vs_stage1_full_pct=best.gain_vs_stage1_full_pct,
                gain_vs_stage2_full_pct=best.gain_vs_stage2_full_pct,
                gain_vs_best_single_full_pct=best.gain_vs_best_single_full_pct,
                latency_delta_vs_best_single_full_ms=best.latency_delta_vs_best_single_full_ms,
                latency_delta_vs_best_single_full_pct=best.latency_delta_vs_best_single_full_pct,
                score_pred=best.score_pred,
                cut_mib=best.cut_mib,
                measured_streaming_fps=best.measured_streaming_fps,
                measured_streaming_cycle_ms=best.measured_streaming_cycle_ms,
                measured_latency_p50_ms=best.measured_latency_p50_ms,
                throughput_mode=best.throughput_mode,
            )
        )
    provider_summaries.sort(key=lambda row: row.best_cycle_ms_cons)

    lines: List[str] = []
    lines.append("## Interleaving / Durchsatz\n")
    if not provider_summaries:
        lines.append("- Keine heterogenen Stage1→Stage2-Pipelines im Benchmark gefunden.\n")
    else:
        lines.append(
            "- Wenn vorhanden, werden **gemessene Streaming-/Interleaving-Metriken** aus dem Runner bevorzugt. Ohne Messwerte fällt die Analyse auf die konservative Zykluszeit `max(part1 + max(overhead, 0), part2)` zurück.\n"
        )
        lines.append(
            "- Positive Werte bei `Gewinn vs best single` bedeuten: **mehr steady-state FPS** als die beste Single-Accelerator-Full-Baseline, auch wenn die End-to-End-Latenz steigen kann.\n"
        )
        for row in provider_summaries:
            label = _label_for_provider(row.stage1_provider, row.stage2_provider, row.provider_tag)
            metric_label = "gemessene Pipeline-FPS" if str(row.throughput_mode or "").lower() != "estimated" else "konservative Pipeline-FPS"
            cycle_label = "gemessene Zykluszeit" if str(row.throughput_mode or "").lower() != "estimated" else "konservative Zykluszeit"
            lines.append(
                f"- **{label}**: bester Boundary **b{row.best_boundary}**, {cycle_label} **{_fmt(row.best_cycle_ms_cons)} ms**, "
                f"{metric_label} **{_fmt(row.best_pipeline_fps_cons)}**, Gewinn vs stage1 full **{_fmt(row.gain_vs_stage1_full_pct,1)} %**, "
                f"vs stage2 full **{_fmt(row.gain_vs_stage2_full_pct,1)} %**, vs best single **{_fmt(row.gain_vs_best_single_full_pct,1)} %**. "
                f"Latenzaufschlag vs best single: **{_fmt(row.latency_delta_vs_best_single_full_ms)} ms** "
                f"({_fmt(row.latency_delta_vs_best_single_full_pct,1)} %).\n"
            )
    return InterleavingAnalysisReport(
        source_name=report.source.display_name,
        provider_summaries=provider_summaries,
        candidate_summaries=candidates,
        summary_markdown="".join(lines),
        issues=issues,
    )


def interleaving_provider_rows(analysis: InterleavingAnalysisReport) -> List[Dict[str, Any]]:
    return [
        {
            "provider": row.provider_tag,
            "stage1_provider": row.stage1_provider,
            "stage2_provider": row.stage2_provider,
            "candidate_count": row.candidate_count,
            "best_boundary": row.best_boundary,
            "best_cycle_ms_opt": row.best_cycle_ms_opt,
            "best_cycle_ms_cons": row.best_cycle_ms_cons,
            "best_pipeline_fps_opt": row.best_pipeline_fps_opt,
            "best_pipeline_fps_cons": row.best_pipeline_fps_cons,
            "full_stage1_ms": row.full_stage1_ms,
            "full_stage2_ms": row.full_stage2_ms,
            "full_best_single_ms": row.full_best_single_ms,
            "gain_vs_stage1_full_pct": row.gain_vs_stage1_full_pct,
            "gain_vs_stage2_full_pct": row.gain_vs_stage2_full_pct,
            "gain_vs_best_single_full_pct": row.gain_vs_best_single_full_pct,
            "latency_delta_vs_best_single_full_ms": row.latency_delta_vs_best_single_full_ms,
            "latency_delta_vs_best_single_full_pct": row.latency_delta_vs_best_single_full_pct,
            "stage_balance": row.best_stage_balance,
            "score_pred": row.score_pred,
            "cut_mib": row.cut_mib,
            "measured_streaming_fps": row.measured_streaming_fps,
            "measured_streaming_cycle_ms": row.measured_streaming_cycle_ms,
            "measured_latency_p50_ms": row.measured_latency_p50_ms,
            "throughput_mode": row.throughput_mode,
        }
        for row in analysis.provider_summaries
    ]


def interleaving_candidate_rows(analysis: InterleavingAnalysisReport, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = analysis.candidate_summaries
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return [
        {
            "provider": row.provider_tag,
            "stage1_provider": row.stage1_provider,
            "stage2_provider": row.stage2_provider,
            "boundary": row.boundary,
            "part1_mean_ms": row.part1_mean_ms,
            "part2_mean_ms": row.part2_mean_ms,
            "composed_mean_ms": row.composed_mean_ms,
            "overhead_ms": row.overhead_ms,
            "cycle_ms_opt": row.cycle_ms_opt,
            "cycle_ms_cons": row.cycle_ms_cons,
            "pipeline_fps_opt": row.pipeline_fps_opt,
            "pipeline_fps_cons": row.pipeline_fps_cons,
            "full_stage1_ms": row.full_stage1_ms,
            "full_stage2_ms": row.full_stage2_ms,
            "full_best_single_ms": row.full_best_single_ms,
            "gain_vs_stage1_full_pct": row.gain_vs_stage1_full_pct,
            "gain_vs_stage2_full_pct": row.gain_vs_stage2_full_pct,
            "gain_vs_best_single_full_pct": row.gain_vs_best_single_full_pct,
            "latency_delta_vs_best_single_full_ms": row.latency_delta_vs_best_single_full_ms,
            "latency_delta_vs_best_single_full_pct": row.latency_delta_vs_best_single_full_pct,
            "stage_balance": row.stage_balance,
            "score_pred": row.score_pred,
            "cut_mib": row.cut_mib,
            "measured_streaming_fps": row.measured_streaming_fps,
            "measured_streaming_cycle_ms": row.measured_streaming_cycle_ms,
            "measured_latency_mean_ms": row.measured_latency_mean_ms,
            "measured_latency_p50_ms": row.measured_latency_p50_ms,
            "measured_latency_p95_ms": row.measured_latency_p95_ms,
            "throughput_mode": row.throughput_mode,
        }
        for row in rows
    ]


def _best_streaming_fps(row: "InterleavingCandidateSummary") -> float:
    val = row.measured_streaming_fps if row.measured_streaming_fps is not None else row.pipeline_fps_cons
    return float(val or 0.0)


def _best_streaming_cycle_ms(row: "InterleavingCandidateSummary") -> float:
    val = row.measured_streaming_cycle_ms if row.measured_streaming_cycle_ms is not None else row.cycle_ms_cons
    return float(val or 0.0)


def _best_streaming_latency_ms(row: "InterleavingCandidateSummary") -> float:
    val = row.measured_latency_mean_ms if row.measured_latency_mean_ms is not None else row.composed_mean_ms
    return float(val or 0.0)


def _ideal_bottleneck_fps(row: "InterleavingCandidateSummary") -> Optional[float]:
    bottleneck_ms = max(float(row.part1_mean_ms or 0.0), float(row.part2_mean_ms or 0.0))
    if bottleneck_ms <= 0.0:
        return None
    return 1000.0 / bottleneck_ms


def _residual_overhead_ms(row: "InterleavingCandidateSummary") -> Optional[float]:
    bottleneck_ms = max(float(row.part1_mean_ms or 0.0), float(row.part2_mean_ms or 0.0))
    cycle_ms = _best_streaming_cycle_ms(row)
    if bottleneck_ms <= 0.0 or cycle_ms <= 0.0:
        return None
    return max(0.0, cycle_ms - bottleneck_ms)


def _collapse_best_by_boundary(analysis: InterleavingAnalysisReport) -> List[InterleavingCandidateSummary]:
    best: Dict[int, InterleavingCandidateSummary] = {}
    for row in analysis.candidate_summaries:
        current = best.get(int(row.boundary))
        if current is None:
            best[int(row.boundary)] = row
            continue
        cur_fps = _best_streaming_fps(current)
        row_fps = _best_streaming_fps(row)
        if row_fps > cur_fps + 1e-9:
            best[int(row.boundary)] = row
            continue
        if abs(row_fps - cur_fps) <= 1e-9 and _best_streaming_latency_ms(row) < _best_streaming_latency_ms(current):
            best[int(row.boundary)] = row
    return sorted(best.values(), key=lambda r: (-_best_streaming_fps(r), _best_streaming_latency_ms(r), int(r.boundary)))


def _spearman_from_ranks(xs: List[int], ys: List[int]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = len(xs)
    diffsq = 0.0
    for x, y in zip(xs, ys):
        d = float(x - y)
        diffsq += d * d
    denom = float(n) * (float(n) ** 2 - 1.0)
    if denom == 0.0:
        return None
    return 1.0 - (6.0 * diffsq) / denom


def _kendall_from_ranks(xs: List[int], ys: List[int]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    concordant = 0
    discordant = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return None
    return float(concordant - discordant) / float(total)


def research_best_full_vs_split_rows(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport) -> List[Dict[str, Any]]:
    provider_rows = [row for row in report.providers if row.full_baseline_ms is not None and row.full_baseline_ms > 0]
    if not provider_rows:
        return []
    best_full = min(provider_rows, key=lambda row: float(row.full_baseline_ms or float('inf')))
    full_fps = _base_fps(best_full.full_baseline_ms)
    rows: List[Dict[str, Any]] = [
        {
            'role': 'best_full',
            'provider_or_pipeline': best_full.tag,
            'boundary': None,
            'latency_ms': best_full.full_baseline_ms,
            'fps_equiv': full_fps,
            'delta_vs_best_full_pct': 0.0,
            'predicted_rank': None,
            'cut_mib': None,
        }
    ]
    best_streaming = _collapse_best_by_boundary(analysis)
    if not best_streaming:
        return rows
    split = best_streaming[0]
    cand_rank_map = {int(row.boundary): idx + 1 for idx, row in enumerate(report.candidate_summaries)}
    split_fps = _best_streaming_fps(split)
    delta = None
    if full_fps is not None and full_fps > 0:
        delta = (split_fps - full_fps) / full_fps * 100.0
    rows.append(
        {
            'role': 'best_split',
            'provider_or_pipeline': split.provider_tag,
            'boundary': int(split.boundary),
            'latency_ms': _best_streaming_latency_ms(split),
            'fps_equiv': split_fps,
            'delta_vs_best_full_pct': delta,
            'predicted_rank': cand_rank_map.get(int(split.boundary)),
            'cut_mib': split.cut_mib,
        }
    )
    return rows


def research_prediction_audit_rows(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    predicted_rows = {int(row['boundary']): row for row in candidate_summary_rows(report, limit=None)}
    collapsed = _collapse_best_by_boundary(analysis)
    rows: List[Dict[str, Any]] = []
    for actual_rank, row in enumerate(collapsed, start=1):
        pred = predicted_rows.get(int(row.boundary), {})
        ideal_fps = _ideal_bottleneck_fps(row)
        residual = _residual_overhead_ms(row)
        measured_fps = _best_streaming_fps(row)
        eff = None
        if ideal_fps is not None and ideal_fps > 0:
            eff = measured_fps / ideal_fps * 100.0
        predicted_rank = pred.get('avg_rank')
        rows.append(
            {
                'boundary': int(row.boundary),
                'provider': row.provider_tag,
                'predicted_rank': int(round(float(predicted_rank))) if predicted_rank is not None else None,
                'predicted_score': pred.get('score_pred'),
                'actual_rank': int(actual_rank),
                'measured_streaming_fps': measured_fps,
                'measured_latency_ms': _best_streaming_latency_ms(row),
                'ideal_bottleneck_fps': ideal_fps,
                'streaming_efficiency_pct': eff,
                'residual_overhead_ms': residual,
                'cut_mib': row.cut_mib,
                'compile_risk': pred.get('hailo_compile_risk_score'),
                'single_context_probability': pred.get('hailo_single_context_probability'),
                'rank_error': (None if predicted_rank is None else int(round(float(predicted_rank))) - int(actual_rank)),
            }
        )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def research_stage_breakdown_rows(analysis: InterleavingAnalysisReport, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in _collapse_best_by_boundary(analysis):
        ideal_fps = _ideal_bottleneck_fps(row)
        measured_fps = _best_streaming_fps(row)
        eff = None
        if ideal_fps is not None and ideal_fps > 0:
            eff = measured_fps / ideal_fps * 100.0
        rows.append(
            {
                'boundary': int(row.boundary),
                'provider': row.provider_tag,
                'stage1_provider': row.stage1_provider,
                'stage2_provider': row.stage2_provider,
                'part1_mean_ms': row.part1_mean_ms,
                'part2_mean_ms': row.part2_mean_ms,
                'stage1_fps_equiv': (1000.0 / row.part1_mean_ms if row.part1_mean_ms > 0 else None),
                'stage2_fps_equiv': (1000.0 / row.part2_mean_ms if row.part2_mean_ms > 0 else None),
                'ideal_bottleneck_fps': ideal_fps,
                'measured_streaming_fps': measured_fps,
                'measured_latency_ms': _best_streaming_latency_ms(row),
                'streaming_efficiency_pct': eff,
                'residual_overhead_ms': _residual_overhead_ms(row),
                'cut_mib': row.cut_mib,
            }
        )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def research_summary_cards(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport) -> Dict[str, Any]:
    best_rows = research_best_full_vs_split_rows(report, analysis)
    audit_rows = research_prediction_audit_rows(report, analysis, limit=None)
    best_full = best_rows[0] if best_rows else {}
    best_split = best_rows[1] if len(best_rows) > 1 else {}
    xs: List[int] = []
    ys: List[int] = []
    best_split_predicted_rank = None
    top3_hit = False
    for row in audit_rows:
        pr = row.get('predicted_rank')
        ar = row.get('actual_rank')
        if pr is not None and ar is not None:
            xs.append(int(pr))
            ys.append(int(ar))
        if int(row.get('actual_rank') or 0) == 1:
            best_split_predicted_rank = row.get('predicted_rank')
            top3_hit = (best_split_predicted_rank is not None and int(best_split_predicted_rank) <= 3)
    return {
        'best_full_label': str(best_full.get('provider_or_pipeline') or '-'),
        'best_full_fps': best_full.get('fps_equiv'),
        'best_split_label': (f"{best_split.get('provider_or_pipeline')} / b{int(best_split.get('boundary'))}" if best_split.get('boundary') is not None else '-'),
        'best_split_fps': best_split.get('fps_equiv'),
        'delta_vs_best_full_pct': best_split.get('delta_vs_best_full_pct'),
        'spearman_rank': _spearman_from_ranks(xs, ys),
        'kendall_rank': _kendall_from_ranks(xs, ys),
        'best_split_predicted_rank': best_split_predicted_rank,
        'top3_hit': top3_hit,
        'audit_count': len(audit_rows),
    }


# ---------------------------------------------------------------------------
# Throughput-oriented metric audit
# ---------------------------------------------------------------------------

def _metric_meta_by_boundary(report: BenchmarkAnalysisReport) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for rows in report.provider_rows.values():
        for row in rows:
            b = _as_int(row.get('boundary'))
            if b is None:
                continue
            tgt = lookup.setdefault(int(b), {})
            for key, value in row.items():
                if key not in tgt or tgt.get(key) in (None, '', []):
                    tgt[key] = value
    return lookup


def _feature_count(v: Any) -> Optional[int]:
    return _score_feature_count(v)


def _predicted_bottleneck_ms(meta: Dict[str, Any]) -> Optional[float]:
    total_ms = _as_float(meta.get('latency_total_ms'))
    if total_ms is None or total_ms <= 0.0:
        return None
    left = _as_float(meta.get('flops_left'))
    right = _as_float(meta.get('flops_right'))
    if left is not None and right is not None and (left + right) > 0.0:
        share = max(float(left), float(right)) / (float(left) + float(right))
        return float(total_ms) * share
    imb = _as_float(meta.get('imbalance_pred') or meta.get('imbalance'))
    if imb is not None:
        imb = max(0.0, min(1.0, float(imb)))
        return float(total_ms) * (0.5 * (1.0 + imb))
    return float(total_ms) * 0.5


def _predicted_handover_ms(meta: Dict[str, Any], stage1: str, stage2: str, *, use_calibration: bool = True) -> Optional[float]:
    return _predicted_handover_ms_shared(
        cut_mib=meta.get('cut_mib'),
        n_cut_tensors=meta.get('n_cut_tensors') or meta.get('crossing_tensors_all'),
        unknown_crossing_tensors=meta.get('unknown_crossing_tensors'),
        peak_act_right_mib=meta.get('peak_act_right_mib'),
        compile_risk_score=meta.get('hailo_compile_risk_score'),
        single_context_probability=meta.get('hailo_single_context_probability'),
        fallback_used=bool(meta.get('hailo_part2_fallback_used')),
        stage1=stage1,
        stage2=stage2,
        imbalance=meta.get('imbalance_pred') or meta.get('imbalance'),
        use_calibration=use_calibration,
    )


def _predicted_stream_cycle_ms(meta: Dict[str, Any], stage1: str, stage2: str, *, use_calibration: bool = True) -> Optional[float]:
    return _predicted_stream_cycle_ms_shared(
        bottleneck_ms=_predicted_bottleneck_ms(meta),
        handover_ms=_predicted_handover_ms(meta, stage1, stage2, use_calibration=use_calibration),
    )


def _predicted_stream_fps(meta: Dict[str, Any], stage1: str, stage2: str, *, use_calibration: bool = True) -> Optional[float]:
    return _predicted_stream_fps_shared(
        bottleneck_ms=_predicted_bottleneck_ms(meta),
        handover_ms=_predicted_handover_ms(meta, stage1, stage2, use_calibration=use_calibration),
    )


def _predicted_stream_rank_map(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport, *, use_calibration: bool = True) -> Dict[int, int]:
    meta_lookup = _metric_meta_by_boundary(report)
    scored: List[tuple[int, float]] = []
    for row in _collapse_best_by_boundary(analysis):
        meta = meta_lookup.get(int(row.boundary), {})
        pred_fps = _predicted_stream_fps(meta, row.stage1_provider, row.stage2_provider, use_calibration=use_calibration)
        if pred_fps is not None:
            scored.append((int(row.boundary), float(pred_fps)))
    scored.sort(key=lambda kv: (-kv[1], kv[0]))
    return {b: idx + 1 for idx, (b, _fps) in enumerate(scored)}


def metric_audit_rows(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport, limit: Optional[int] = None, *, use_calibration: bool = True) -> List[Dict[str, Any]]:
    old_rows = {int(row['boundary']): row for row in candidate_summary_rows(report, limit=None)}
    meta_lookup = _metric_meta_by_boundary(report)
    pred_stream_rank_uncal = _predicted_stream_rank_map(report, analysis, use_calibration=False)
    pred_stream_rank_cal = _predicted_stream_rank_map(report, analysis, use_calibration=True)
    rows: List[Dict[str, Any]] = []
    for actual_rank, row in enumerate(_collapse_best_by_boundary(analysis), start=1):
        meta = meta_lookup.get(int(row.boundary), {})
        old = old_rows.get(int(row.boundary), {})
        old_rank = old.get('avg_rank')
        pred_handover_uncal = _predicted_handover_ms(meta, row.stage1_provider, row.stage2_provider, use_calibration=False)
        pred_handover_cal = _predicted_handover_ms(meta, row.stage1_provider, row.stage2_provider, use_calibration=True)
        pred_cycle_uncal = _predicted_stream_cycle_ms(meta, row.stage1_provider, row.stage2_provider, use_calibration=False)
        pred_cycle_cal = _predicted_stream_cycle_ms(meta, row.stage1_provider, row.stage2_provider, use_calibration=True)
        pred_fps_uncal = _predicted_stream_fps(meta, row.stage1_provider, row.stage2_provider, use_calibration=False)
        pred_fps_cal = _predicted_stream_fps(meta, row.stage1_provider, row.stage2_provider, use_calibration=True)
        new_rank_uncal = pred_stream_rank_uncal.get(int(row.boundary))
        new_rank_cal = pred_stream_rank_cal.get(int(row.boundary))
        active_rank = new_rank_cal if bool(use_calibration) else new_rank_uncal
        active_handover = pred_handover_cal if bool(use_calibration) else pred_handover_uncal
        active_cycle = pred_cycle_cal if bool(use_calibration) else pred_cycle_uncal
        active_fps = pred_fps_cal if bool(use_calibration) else pred_fps_uncal
        rows.append(
            {
                'boundary': int(row.boundary),
                'provider': row.provider_tag,
                'predicted_rank_old': (int(round(float(old_rank))) if old_rank is not None else None),
                'predicted_rank_uncal': new_rank_uncal,
                'predicted_rank_cal': new_rank_cal,
                'predicted_rank_new': active_rank,
                'predicted_score_old': old.get('score_pred'),
                'predicted_handover_ms_uncal': pred_handover_uncal,
                'predicted_handover_ms_cal': pred_handover_cal,
                'predicted_handover_ms': active_handover,
                'predicted_stream_cycle_ms_uncal': pred_cycle_uncal,
                'predicted_stream_cycle_ms_cal': pred_cycle_cal,
                'predicted_stream_cycle_ms': active_cycle,
                'predicted_stream_fps_uncal': pred_fps_uncal,
                'predicted_stream_fps_cal': pred_fps_cal,
                'predicted_stream_fps': active_fps,
                'calibration_profile': _active_calibration_profile(enabled=use_calibration),
                'calibration_enabled': bool(use_calibration),
                'hailo_feasibility_risk': _hailo_feasibility_risk(
                    compile_risk_score=meta.get('hailo_compile_risk_score'),
                    single_context_probability=meta.get('hailo_single_context_probability'),
                    fallback_used=bool(meta.get('hailo_part2_fallback_used')),
                ),
                'hailo_interface_penalty': _hailo_interface_penalty(
                    cut_mib=meta.get('cut_mib'),
                    n_cut_tensors=meta.get('n_cut_tensors') or meta.get('crossing_tensors_all'),
                    unknown_crossing_tensors=meta.get('unknown_crossing_tensors'),
                    peak_act_right_mib=meta.get('peak_act_right_mib'),
                    stage1=row.stage1_provider,
                    stage2=row.stage2_provider,
                ),
                'actual_rank': int(actual_rank),
                'measured_streaming_fps': _best_streaming_fps(row),
                'measured_latency_ms': _best_streaming_latency_ms(row),
                'residual_overhead_ms': _residual_overhead_ms(row),
                'cut_mib': row.cut_mib,
                'compile_risk': old.get('hailo_compile_risk_score'),
                'single_context_probability': old.get('hailo_single_context_probability'),
                'rank_error_old': (None if old_rank is None else int(round(float(old_rank))) - int(actual_rank)),
                'rank_error_uncal': (None if new_rank_uncal is None else int(new_rank_uncal) - int(actual_rank)),
                'rank_error_cal': (None if new_rank_cal is None else int(new_rank_cal) - int(actual_rank)),
                'rank_error_new': (None if active_rank is None else int(active_rank) - int(actual_rank)),
                'rank_error_old_abs': (None if old_rank is None else abs(int(round(float(old_rank))) - int(actual_rank))),
                'rank_error_uncal_abs': (None if new_rank_uncal is None else abs(int(new_rank_uncal) - int(actual_rank))),
                'rank_error_cal_abs': (None if new_rank_cal is None else abs(int(new_rank_cal) - int(actual_rank))),
                'rank_error_new_abs': (None if active_rank is None else abs(int(active_rank) - int(actual_rank))),
                'n_cut_tensors': (_feature_count(meta.get('n_cut_tensors') or meta.get('crossing_tensors_all'))),
                'unknown_crossing_tensors': (_feature_count(meta.get('unknown_crossing_tensors'))),
                'peak_act_right_mib': _as_float(meta.get('peak_act_right_mib')),
                'hailo_part2_fallback_used': bool(meta.get('hailo_part2_fallback_used')),
            }
        )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def metric_audit_summary(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport, *, use_calibration: bool = True) -> Dict[str, Any]:
    rows = metric_audit_rows(report, analysis, limit=None, use_calibration=use_calibration)
    xs_old = [int(r['predicted_rank_old']) for r in rows if r.get('predicted_rank_old') is not None and r.get('actual_rank') is not None]
    ys_old = [int(r['actual_rank']) for r in rows if r.get('predicted_rank_old') is not None and r.get('actual_rank') is not None]
    xs_uncal = [int(r['predicted_rank_uncal']) for r in rows if r.get('predicted_rank_uncal') is not None and r.get('actual_rank') is not None]
    ys_uncal = [int(r['actual_rank']) for r in rows if r.get('predicted_rank_uncal') is not None and r.get('actual_rank') is not None]
    xs_cal = [int(r['predicted_rank_cal']) for r in rows if r.get('predicted_rank_cal') is not None and r.get('actual_rank') is not None]
    ys_cal = [int(r['actual_rank']) for r in rows if r.get('predicted_rank_cal') is not None and r.get('actual_rank') is not None]
    xs_new = [int(r['predicted_rank_new']) for r in rows if r.get('predicted_rank_new') is not None and r.get('actual_rank') is not None]
    ys_new = [int(r['actual_rank']) for r in rows if r.get('predicted_rank_new') is not None and r.get('actual_rank') is not None]
    best = next((r for r in rows if int(r.get('actual_rank') or 0) == 1), None)

    def _hit(rank: Any, k: int) -> bool:
        try:
            return int(rank) <= int(k)
        except Exception:
            return False

    return {
        'count': len(rows),
        'old_spearman_rank': _spearman_from_ranks(xs_old, ys_old),
        'old_kendall_rank': _kendall_from_ranks(xs_old, ys_old),
        'raw_spearman_rank': _spearman_from_ranks(xs_uncal, ys_uncal),
        'raw_kendall_rank': _kendall_from_ranks(xs_uncal, ys_uncal),
        'uncal_spearman_rank': _spearman_from_ranks(xs_uncal, ys_uncal),
        'uncal_kendall_rank': _kendall_from_ranks(xs_uncal, ys_uncal),
        'cal_spearman_rank': _spearman_from_ranks(xs_cal, ys_cal),
        'cal_kendall_rank': _kendall_from_ranks(xs_cal, ys_cal),
        'new_spearman_rank': _spearman_from_ranks(xs_new, ys_new),
        'new_kendall_rank': _kendall_from_ranks(xs_new, ys_new),
        'old_top1_hit': bool(best and _hit(best.get('predicted_rank_old'), 1)),
        'old_top3_hit': bool(best and _hit(best.get('predicted_rank_old'), 3)),
        'old_top5_hit': bool(best and _hit(best.get('predicted_rank_old'), 5)),
        'raw_top1_hit': bool(best and _hit(best.get('predicted_rank_uncal'), 1)),
        'raw_top3_hit': bool(best and _hit(best.get('predicted_rank_uncal'), 3)),
        'raw_top5_hit': bool(best and _hit(best.get('predicted_rank_uncal'), 5)),
        'uncal_top1_hit': bool(best and _hit(best.get('predicted_rank_uncal'), 1)),
        'uncal_top3_hit': bool(best and _hit(best.get('predicted_rank_uncal'), 3)),
        'uncal_top5_hit': bool(best and _hit(best.get('predicted_rank_uncal'), 5)),
        'cal_top1_hit': bool(best and _hit(best.get('predicted_rank_cal'), 1)),
        'cal_top3_hit': bool(best and _hit(best.get('predicted_rank_cal'), 3)),
        'cal_top5_hit': bool(best and _hit(best.get('predicted_rank_cal'), 5)),
        'new_top1_hit': bool(best and _hit(best.get('predicted_rank_new'), 1)),
        'new_top3_hit': bool(best and _hit(best.get('predicted_rank_new'), 3)),
        'new_top5_hit': bool(best and _hit(best.get('predicted_rank_new'), 5)),
        'best_boundary': (best.get('boundary') if best else None),
        'best_old_rank': (best.get('predicted_rank_old') if best else None),
        'best_raw_rank': (best.get('predicted_rank_uncal') if best else None),
        'best_uncal_rank': (best.get('predicted_rank_uncal') if best else None),
        'best_cal_rank': (best.get('predicted_rank_cal') if best else None),
        'best_new_rank': (best.get('predicted_rank_new') if best else None),
        'calibration_profile': _active_calibration_profile(enabled=use_calibration),
        'calibration_enabled': bool(use_calibration),
    }


def metric_audit_comparison_rows(
    left_report: BenchmarkAnalysisReport,
    left_analysis: InterleavingAnalysisReport,
    right_report: BenchmarkAnalysisReport,
    right_analysis: InterleavingAnalysisReport,
    limit: Optional[int] = None,
    *,
    use_calibration: bool = True,
) -> List[Dict[str, Any]]:
    left_rows = {int(r['boundary']): r for r in metric_audit_rows(left_report, left_analysis, limit=None, use_calibration=use_calibration)}
    right_rows = {int(r['boundary']): r for r in metric_audit_rows(right_report, right_analysis, limit=None, use_calibration=use_calibration)}
    boundaries = sorted(set(left_rows) & set(right_rows))
    rows: List[Dict[str, Any]] = []
    for boundary in boundaries:
        l = left_rows[boundary]
        r = right_rows[boundary]
        rows.append(
            {
                'boundary': int(boundary),
                'left_provider': l.get('provider'),
                'right_provider': r.get('provider'),
                'left_actual_rank': l.get('actual_rank'),
                'right_actual_rank': r.get('actual_rank'),
                'left_predicted_rank_old': l.get('predicted_rank_old'),
                'right_predicted_rank_old': r.get('predicted_rank_old'),
                'left_predicted_rank_raw': l.get('predicted_rank_uncal'),
                'right_predicted_rank_raw': r.get('predicted_rank_uncal'),
                'left_predicted_rank_uncal': l.get('predicted_rank_uncal'),
                'right_predicted_rank_uncal': r.get('predicted_rank_uncal'),
                'left_predicted_rank_new': l.get('predicted_rank_new'),
                'right_predicted_rank_new': r.get('predicted_rank_new'),
                'left_measured_streaming_fps': l.get('measured_streaming_fps'),
                'right_measured_streaming_fps': r.get('measured_streaming_fps'),
                'fps_delta_abs': _delta(r.get('measured_streaming_fps'), l.get('measured_streaming_fps')),
                'fps_delta_pct': _delta_pct(r.get('measured_streaming_fps'), l.get('measured_streaming_fps')),
                'left_rank_error_old_abs': l.get('rank_error_old_abs'),
                'right_rank_error_old_abs': r.get('rank_error_old_abs'),
                'old_rank_error_abs_delta': _delta(r.get('rank_error_old_abs'), l.get('rank_error_old_abs')),
                'left_rank_error_raw_abs': l.get('rank_error_uncal_abs'),
                'right_rank_error_raw_abs': r.get('rank_error_uncal_abs'),
                'raw_rank_error_abs_delta': _delta(r.get('rank_error_uncal_abs'), l.get('rank_error_uncal_abs')),
                'uncal_rank_error_abs_delta': _delta(r.get('rank_error_uncal_abs'), l.get('rank_error_uncal_abs')),
                'left_rank_error_new_abs': l.get('rank_error_new_abs'),
                'right_rank_error_new_abs': r.get('rank_error_new_abs'),
                'new_rank_error_abs_delta': _delta(r.get('rank_error_new_abs'), l.get('rank_error_new_abs')),
                'left_predicted_stream_fps_uncal': l.get('predicted_stream_fps_uncal'),
                'right_predicted_stream_fps_uncal': r.get('predicted_stream_fps_uncal'),
                'predicted_stream_fps_uncal_delta': _delta(r.get('predicted_stream_fps_uncal'), l.get('predicted_stream_fps_uncal')),
                'left_predicted_stream_fps_cal': l.get('predicted_stream_fps_cal'),
                'right_predicted_stream_fps_cal': r.get('predicted_stream_fps_cal'),
                'predicted_stream_fps_cal_delta': _delta(r.get('predicted_stream_fps_cal'), l.get('predicted_stream_fps_cal')),
                'left_predicted_stream_fps': l.get('predicted_stream_fps'),
                'right_predicted_stream_fps': r.get('predicted_stream_fps'),
                'predicted_stream_fps_delta': _delta(r.get('predicted_stream_fps'), l.get('predicted_stream_fps')),
            }
        )
    rows.sort(key=lambda row: (abs(float(row.get('new_rank_error_abs_delta') or 0.0)), abs(float(row.get('old_rank_error_abs_delta') or 0.0))), reverse=True)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def metric_audit_comparison_summary(
    left_report: BenchmarkAnalysisReport,
    left_analysis: InterleavingAnalysisReport,
    right_report: BenchmarkAnalysisReport,
    right_analysis: InterleavingAnalysisReport,
    *,
    use_calibration: bool = True,
) -> Dict[str, Any]:
    left = metric_audit_summary(left_report, left_analysis, use_calibration=use_calibration)
    right = metric_audit_summary(right_report, right_analysis, use_calibration=use_calibration)
    rows = metric_audit_comparison_rows(left_report, left_analysis, right_report, right_analysis, limit=None, use_calibration=use_calibration)

    def _mean(key: str) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    return {
        'common_case_count': len(rows),
        'left_old_spearman_rank': left.get('old_spearman_rank'),
        'right_old_spearman_rank': right.get('old_spearman_rank'),
        'old_spearman_rank_delta': _delta(right.get('old_spearman_rank'), left.get('old_spearman_rank')),
        'left_raw_spearman_rank': left.get('raw_spearman_rank'),
        'right_raw_spearman_rank': right.get('raw_spearman_rank'),
        'raw_spearman_rank_delta': _delta(right.get('raw_spearman_rank'), left.get('raw_spearman_rank')),
        'left_uncal_spearman_rank': left.get('uncal_spearman_rank'),
        'right_uncal_spearman_rank': right.get('uncal_spearman_rank'),
        'uncal_spearman_rank_delta': _delta(right.get('uncal_spearman_rank'), left.get('uncal_spearman_rank')),
        'left_cal_spearman_rank': left.get('cal_spearman_rank'),
        'right_cal_spearman_rank': right.get('cal_spearman_rank'),
        'cal_spearman_rank_delta': _delta(right.get('cal_spearman_rank'), left.get('cal_spearman_rank')),
        'left_new_spearman_rank': left.get('new_spearman_rank'),
        'right_new_spearman_rank': right.get('new_spearman_rank'),
        'new_spearman_rank_delta': _delta(right.get('new_spearman_rank'), left.get('new_spearman_rank')),
        'left_old_kendall_rank': left.get('old_kendall_rank'),
        'right_old_kendall_rank': right.get('old_kendall_rank'),
        'old_kendall_rank_delta': _delta(right.get('old_kendall_rank'), left.get('old_kendall_rank')),
        'left_raw_kendall_rank': left.get('raw_kendall_rank'),
        'right_raw_kendall_rank': right.get('raw_kendall_rank'),
        'raw_kendall_rank_delta': _delta(right.get('raw_kendall_rank'), left.get('raw_kendall_rank')),
        'left_uncal_kendall_rank': left.get('uncal_kendall_rank'),
        'right_uncal_kendall_rank': right.get('uncal_kendall_rank'),
        'uncal_kendall_rank_delta': _delta(right.get('uncal_kendall_rank'), left.get('uncal_kendall_rank')),
        'left_cal_kendall_rank': left.get('cal_kendall_rank'),
        'right_cal_kendall_rank': right.get('cal_kendall_rank'),
        'cal_kendall_rank_delta': _delta(right.get('cal_kendall_rank'), left.get('cal_kendall_rank')),
        'left_new_kendall_rank': left.get('new_kendall_rank'),
        'right_new_kendall_rank': right.get('new_kendall_rank'),
        'new_kendall_rank_delta': _delta(right.get('new_kendall_rank'), left.get('new_kendall_rank')),
        'left_old_top1_hit': left.get('old_top1_hit'),
        'right_old_top1_hit': right.get('old_top1_hit'),
        'left_raw_top1_hit': left.get('raw_top1_hit'),
        'right_raw_top1_hit': right.get('raw_top1_hit'),
        'left_uncal_top1_hit': left.get('uncal_top1_hit'),
        'right_uncal_top1_hit': right.get('uncal_top1_hit'),
        'left_cal_top1_hit': left.get('cal_top1_hit'),
        'right_cal_top1_hit': right.get('cal_top1_hit'),
        'left_new_top1_hit': left.get('new_top1_hit'),
        'right_new_top1_hit': right.get('new_top1_hit'),
        'left_old_top3_hit': left.get('old_top3_hit'),
        'right_old_top3_hit': right.get('old_top3_hit'),
        'left_raw_top3_hit': left.get('raw_top3_hit'),
        'right_raw_top3_hit': right.get('raw_top3_hit'),
        'left_uncal_top3_hit': left.get('uncal_top3_hit'),
        'right_uncal_top3_hit': right.get('uncal_top3_hit'),
        'left_cal_top3_hit': left.get('cal_top3_hit'),
        'right_cal_top3_hit': right.get('cal_top3_hit'),
        'left_new_top3_hit': left.get('new_top3_hit'),
        'right_new_top3_hit': right.get('new_top3_hit'),
        'left_old_top5_hit': left.get('old_top5_hit'),
        'right_old_top5_hit': right.get('old_top5_hit'),
        'left_raw_top5_hit': left.get('raw_top5_hit'),
        'right_raw_top5_hit': right.get('raw_top5_hit'),
        'left_uncal_top5_hit': left.get('uncal_top5_hit'),
        'right_uncal_top5_hit': right.get('uncal_top5_hit'),
        'left_cal_top5_hit': left.get('cal_top5_hit'),
        'right_cal_top5_hit': right.get('cal_top5_hit'),
        'left_new_top5_hit': left.get('new_top5_hit'),
        'right_new_top5_hit': right.get('new_top5_hit'),
        'left_old_mean_abs_rank_error': _mean('left_rank_error_old_abs'),
        'right_old_mean_abs_rank_error': _mean('right_rank_error_old_abs'),
        'old_mean_abs_rank_error_delta': _delta(_mean('right_rank_error_old_abs'), _mean('left_rank_error_old_abs')),
        'left_raw_mean_abs_rank_error': _mean('left_rank_error_raw_abs'),
        'right_raw_mean_abs_rank_error': _mean('right_rank_error_raw_abs'),
        'raw_mean_abs_rank_error_delta': _delta(_mean('right_rank_error_raw_abs'), _mean('left_rank_error_raw_abs')),
        'left_uncal_mean_abs_rank_error': _mean('left_rank_error_raw_abs'),
        'right_uncal_mean_abs_rank_error': _mean('right_rank_error_raw_abs'),
        'uncal_mean_abs_rank_error_delta': _delta(_mean('right_rank_error_raw_abs'), _mean('left_rank_error_raw_abs')),
        'left_cal_mean_abs_rank_error': _mean('left_rank_error_cal_abs'),
        'right_cal_mean_abs_rank_error': _mean('right_rank_error_cal_abs'),
        'cal_mean_abs_rank_error_delta': _delta(_mean('right_rank_error_cal_abs'), _mean('left_rank_error_cal_abs')),
        'left_new_mean_abs_rank_error': _mean('left_rank_error_new_abs'),
        'right_new_mean_abs_rank_error': _mean('right_rank_error_new_abs'),
        'new_mean_abs_rank_error_delta': _delta(_mean('right_rank_error_new_abs'), _mean('left_rank_error_new_abs')),
        'cal_mean_abs_rank_error_delta': _delta(_mean('right_rank_error_new_abs'), _mean('left_rank_error_new_abs')),
        'mean_fps_delta_pct': _mean('fps_delta_pct'),
        'mean_predicted_stream_fps_uncal_delta': _mean('predicted_stream_fps_uncal_delta'),
        'mean_predicted_stream_fps_cal_delta': _mean('predicted_stream_fps_cal_delta'),
        'calibration_enabled': bool(use_calibration),
        'calibration_profile': _active_calibration_profile(enabled=use_calibration),
    }


def build_metric_audit_rank_figure(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport, *, use_calibration: bool = True) -> Figure:
    rows = metric_audit_rows(report, analysis, limit=None, use_calibration=use_calibration)
    pts = [r for r in rows if r.get('actual_rank') is not None and (r.get('predicted_rank_old') is not None or r.get('predicted_rank_new') is not None)]
    if not pts:
        return _placeholder_figure('Metric audit: rank', 'Keine Audit-Daten verfügbar.')
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    max_rank = max(int(r.get('actual_rank') or 0) for r in pts)
    ax.plot([1, max_rank], [1, max_rank], linewidth=1.0)
    xs_old = [int(r['predicted_rank_old']) for r in pts if r.get('predicted_rank_old') is not None]
    ys_old = [int(r['actual_rank']) for r in pts if r.get('predicted_rank_old') is not None]
    xs_new = [int(r['predicted_rank_new']) for r in pts if r.get('predicted_rank_new') is not None]
    ys_new = [int(r['actual_rank']) for r in pts if r.get('predicted_rank_new') is not None]
    xs_uncal = [int(r['predicted_rank_uncal']) for r in pts if r.get('predicted_rank_uncal') is not None]
    ys_uncal = [int(r['actual_rank']) for r in pts if r.get('predicted_rank_uncal') is not None]
    xs_cal = [int(r['predicted_rank_cal']) for r in pts if r.get('predicted_rank_cal') is not None]
    ys_cal = [int(r['actual_rank']) for r in pts if r.get('predicted_rank_cal') is not None]
    if xs_old:
        ax.scatter(xs_old, ys_old, alpha=0.8, label='old score')
    if xs_uncal:
        ax.scatter(xs_uncal, ys_uncal, alpha=0.8, marker='x', label='uncal throughput')
    if xs_cal:
        ax.scatter(xs_cal, ys_cal, alpha=0.8, marker='+', label='cal throughput')
    for r in pts:
        target_rank = r.get('predicted_rank_new') if bool(use_calibration) else r.get('predicted_rank_uncal')
        if target_rank is not None and r.get('actual_rank') is not None:
            ax.annotate(f"b{int(r['boundary'])}", (int(target_rank), int(r['actual_rank'])), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Predicted rank')
    ax.set_ylabel('Measured rank')
    ax.set_title('Metric audit: predicted rank vs measured rank')
    ax.legend()
    fig.tight_layout()
    return fig


def build_metric_audit_fps_figure(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport, *, use_calibration: bool = True) -> Figure:
    rows = [r for r in metric_audit_rows(report, analysis, limit=None, use_calibration=use_calibration) if (r.get('predicted_stream_fps_uncal') is not None or r.get('predicted_stream_fps_cal') is not None) and r.get('measured_streaming_fps') is not None]
    if not rows:
        return _placeholder_figure('Metric audit: throughput', 'Keine Audit-Daten verfügbar.')
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    xs_uncal = [float(r['predicted_stream_fps_uncal']) for r in rows if r.get('predicted_stream_fps_uncal') is not None]
    ys_uncal = [float(r['measured_streaming_fps']) for r in rows if r.get('predicted_stream_fps_uncal') is not None]
    xs_cal = [float(r['predicted_stream_fps_cal']) for r in rows if r.get('predicted_stream_fps_cal') is not None]
    ys_cal = [float(r['measured_streaming_fps']) for r in rows if r.get('predicted_stream_fps_cal') is not None]
    if xs_uncal:
        ax.scatter(xs_uncal, ys_uncal, label='uncal throughput', alpha=0.8, marker='x')
    if xs_cal:
        ax.scatter(xs_cal, ys_cal, label='cal throughput', alpha=0.8, marker='+')
    allx = xs_uncal + xs_cal
    ally = ys_uncal + ys_cal
    lim = max(max(allx), max(ally)) if allx and ally else 1.0
    ax.plot([0, lim], [0, lim], linewidth=1.0)
    for r in rows:
        x = r.get('predicted_stream_fps_cal') if bool(use_calibration) else r.get('predicted_stream_fps_uncal')
        y = r.get('measured_streaming_fps')
        if x is not None and y is not None:
            ax.annotate(f"b{int(r['boundary'])}", (float(x), float(y)), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Predicted throughput FPS (uncal / cal)')
    ax.set_ylabel('Measured streaming FPS')
    ax.set_title('Metric audit: predicted vs measured throughput')
    ax.legend()
    fig.tight_layout()
    return fig


def build_comparison_metric_audit_rank_error_figure(
    left_report: BenchmarkAnalysisReport,
    left_analysis: InterleavingAnalysisReport,
    right_report: BenchmarkAnalysisReport,
    right_analysis: InterleavingAnalysisReport,
    *,
    use_calibration: bool = True,
) -> Figure:
    rows = metric_audit_comparison_rows(left_report, left_analysis, right_report, right_analysis, limit=None, use_calibration=use_calibration)
    if not rows:
        return _placeholder_figure('Metric audit Δ', 'Keine gemeinsamen Metric-Audit-Fälle verfügbar.')
    rows = rows[: min(20, len(rows))]
    labels = [f"b{int(r['boundary'])}" for r in rows]
    x = np.arange(len(labels), dtype=float)
    w = 0.35
    old_vals = [float(r.get('old_rank_error_abs_delta') or 0.0) for r in rows]
    raw_vals = [float(r.get('raw_rank_error_abs_delta') or 0.0) for r in rows]
    new_vals = [float(r.get('new_rank_error_abs_delta') or 0.0) for r in rows]
    fig = Figure(figsize=(7.8, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(x - w, old_vals, width=w, label='Δ |error| old')
    ax.bar(x, raw_vals, width=w, label='Δ |error| TH raw')
    ax.bar(x + w, new_vals, width=w, label='Δ |error| TH cal')
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Run B - Run A')
    ax.set_title('Metric audit comparison: rank error change')
    ax.legend()
    fig.tight_layout()
    return fig


def build_interleaving_gain_figure(analysis: InterleavingAnalysisReport) -> Figure:
    rows = analysis.provider_summaries
    if not rows:
        return _placeholder_figure("Interleaving FPS", "Keine heterogenen Pipelines verfügbar.")
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    labels = [_label_for_provider(r.stage1_provider, r.stage2_provider, r.provider_tag) for r in rows]
    x = np.arange(len(labels), dtype=float)
    w = 0.25
    vals1 = [r.gain_vs_stage1_full_pct if r.gain_vs_stage1_full_pct is not None else float("nan") for r in rows]
    vals2 = [r.gain_vs_stage2_full_pct if r.gain_vs_stage2_full_pct is not None else float("nan") for r in rows]
    vals3 = [r.gain_vs_best_single_full_pct if r.gain_vs_best_single_full_pct is not None else float("nan") for r in rows]
    ax.bar(x - w, vals1, width=w, label="vs stage1 full")
    ax.bar(x, vals2, width=w, label="vs stage2 full")
    ax.bar(x + w, vals3, width=w, label="vs best single")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("FPS gain [%]")
    ax.set_title("Interleaving / steady-state Throughput")
    ax.legend()
    fig.tight_layout()
    return fig


def build_interleaving_tradeoff_figure(analysis: InterleavingAnalysisReport) -> Figure:
    rows = analysis.provider_summaries
    if not rows:
        return _placeholder_figure("FPS vs Latenz", "Keine heterogenen Pipelines verfügbar.")
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    xs = [float(r.latency_delta_vs_best_single_full_ms or float("nan")) for r in rows]
    ys = [float(r.gain_vs_best_single_full_pct or float("nan")) for r in rows]
    ax.scatter(xs, ys)
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)
    for row, x, y in zip(rows, xs, ys):
        if math.isfinite(x) and math.isfinite(y):
            label = f"{_label_for_provider(row.stage1_provider, row.stage2_provider, row.provider_tag)}\nb{row.best_boundary}"
            ax.text(x, y, label, fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("Latenzaufschlag vs best single [ms]")
    ax.set_ylabel("FPS gain vs best single [%]")
    ax.set_title("Interleaving Trade-off: FPS vs Latenz")
    fig.tight_layout()
    return fig


def build_interleaving_residual_overhead_figure(analysis: InterleavingAnalysisReport) -> Figure:
    rows = [row for row in _collapse_best_by_boundary(analysis) if row.cut_mib is not None and _residual_overhead_ms(row) is not None]
    if not rows:
        return _placeholder_figure("Cut vs residual overhead", "Keine residual-overhead Daten verfügbar.")
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    xs = [float(row.cut_mib or 0.0) for row in rows]
    ys = [float(_residual_overhead_ms(row) or 0.0) for row in rows]
    ax.scatter(xs, ys)
    for row, x, y in zip(rows, xs, ys):
        ax.annotate(f"b{int(row.boundary)}", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Cut [MiB]")
    ax.set_ylabel("Residual overhead [ms]")
    ax.set_title("Cut-Größe vs residualer Streaming-Overhead")
    fig.tight_layout()
    return fig


def _build_caption_lines(analysis: InterleavingAnalysisReport) -> List[str]:
    return [
        "# Interleaving / Throughput – Figure Captions\n",
        "- `benchmark_analysis_interleaving_gain.*`: steady-state FPS-Gewinn (gemessen wenn verfügbar, sonst konservativ geschätzt) der besten heterogenen Pipeline je Provider gegenüber stage1 full, stage2 full und der besten Single-Accelerator-Full-Baseline.\n",
        "- `benchmark_analysis_interleaving_tradeoff.*`: Trade-off aus FPS-Gewinn (gemessen wenn verfügbar, sonst konservativ geschätzt) vs Latenzaufschlag gegenüber der besten Single-Accelerator-Full-Baseline.\n",
    ]


def export_interleaving_analysis(analysis: InterleavingAnalysisReport, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    summary_md = output_dir / "benchmark_analysis_interleaving_summary.md"
    summary_md.write_text(analysis.summary_markdown, encoding="utf-8")
    paths["summary_md"] = summary_md
    provider_csv = output_dir / "benchmark_analysis_interleaving_provider_summary.csv"
    _write_csv(provider_csv, interleaving_provider_rows(analysis))
    paths["provider_csv"] = provider_csv
    cand_csv = output_dir / "benchmark_analysis_interleaving_candidates.csv"
    _write_csv(cand_csv, interleaving_candidate_rows(analysis))
    paths["candidate_csv"] = cand_csv
    captions_md = output_dir / "benchmark_analysis_interleaving_captions.md"
    captions_md.write_text("".join(_build_caption_lines(analysis)), encoding="utf-8")
    paths["captions_md"] = captions_md
    plots = {
        "interleaving_gain": build_interleaving_gain_figure(analysis),
        "interleaving_tradeoff": build_interleaving_tradeoff_figure(analysis),
        "interleaving_residual_overhead": build_interleaving_residual_overhead_figure(analysis),
    }
    for name, fig in plots.items():
        for ext in ("png", "pdf", "svg"):
            out_path = output_dir / f"benchmark_analysis_{name}.{ext}"
            fig.savefig(out_path, bbox_inches="tight")
            paths[f"plot_{name}_{ext}"] = out_path
    return paths


def compare_interleaving_reports(left_report: BenchmarkAnalysisReport, right_report: BenchmarkAnalysisReport) -> InterleavingComparisonReport:
    left = compute_interleaving_analysis(left_report)
    right = compute_interleaving_analysis(right_report)
    left_map = {row.provider_tag: row for row in left.provider_summaries}
    right_map = {row.provider_tag: row for row in right.provider_summaries}
    tags = sorted(set(left_map) | set(right_map))
    rows: List[ComparisonInterleavingSummary] = []
    issues: List[str] = []
    for tag in tags:
        l = left_map.get(tag)
        r = right_map.get(tag)
        stage1 = (r.stage1_provider if r is not None else (l.stage1_provider if l is not None else ""))
        stage2 = (r.stage2_provider if r is not None else (l.stage2_provider if l is not None else ""))
        fps_delta_abs = _delta(r.best_pipeline_fps_cons if r is not None else None, l.best_pipeline_fps_cons if l is not None else None)
        fps_delta_pct = _delta_pct(r.best_pipeline_fps_cons if r is not None else None, l.best_pipeline_fps_cons if l is not None else None)
        rows.append(
            ComparisonInterleavingSummary(
                provider_tag=tag,
                stage1_provider=stage1,
                stage2_provider=stage2,
                left_candidate_count=l.candidate_count if l is not None else 0,
                right_candidate_count=r.candidate_count if r is not None else 0,
                left_best_boundary=l.best_boundary if l is not None else None,
                right_best_boundary=r.best_boundary if r is not None else None,
                left_best_cycle_ms_cons=l.best_cycle_ms_cons if l is not None else None,
                right_best_cycle_ms_cons=r.best_cycle_ms_cons if r is not None else None,
                left_best_pipeline_fps_cons=l.best_pipeline_fps_cons if l is not None else None,
                right_best_pipeline_fps_cons=r.best_pipeline_fps_cons if r is not None else None,
                fps_delta_abs=fps_delta_abs,
                fps_delta_pct=fps_delta_pct,
                left_gain_vs_best_single_full_pct=l.gain_vs_best_single_full_pct if l is not None else None,
                right_gain_vs_best_single_full_pct=r.gain_vs_best_single_full_pct if r is not None else None,
                gain_delta_pct=_delta(r.gain_vs_best_single_full_pct if r is not None else None, l.gain_vs_best_single_full_pct if l is not None else None),
                left_latency_delta_vs_best_single_full_ms=l.latency_delta_vs_best_single_full_ms if l is not None else None,
                right_latency_delta_vs_best_single_full_ms=r.latency_delta_vs_best_single_full_ms if r is not None else None,
                latency_delta_ms=_delta(r.latency_delta_vs_best_single_full_ms if r is not None else None, l.latency_delta_vs_best_single_full_ms if l is not None else None),
                best_boundary_changed=(l is not None and r is not None and l.best_boundary != r.best_boundary),
            )
        )
    left_only = sorted(set(left_map) - set(right_map))
    right_only = sorted(set(right_map) - set(left_map))
    if left_only:
        issues.append(f"Nur in Lauf A heterogene Pipelines: {', '.join(left_only)}")
    if right_only:
        issues.append(f"Nur in Lauf B heterogene Pipelines: {', '.join(right_only)}")

    lines: List[str] = []
    lines.append("## Interleaving / Durchsatz – Änderungen\n")
    if not rows:
        lines.append("- Keine heterogenen Pipelines in beiden Läufen vergleichbar.\n")
    else:
        for row in rows:
            label = _label_for_provider(row.stage1_provider, row.stage2_provider, row.provider_tag)
            lines.append(
                f"- **{label}**: konservative Pipeline-FPS **{_fmt(row.left_best_pipeline_fps_cons)} → {_fmt(row.right_best_pipeline_fps_cons)}** "
                f"(Δ **{_fmt(row.fps_delta_pct,1)} %**), Gewinn vs best single **{_fmt(row.left_gain_vs_best_single_full_pct,1)} → {_fmt(row.right_gain_vs_best_single_full_pct,1)} %**. "
                f"Latenzaufschlag vs best single **{_fmt(row.left_latency_delta_vs_best_single_full_ms)} → {_fmt(row.right_latency_delta_vs_best_single_full_ms)} ms**."
            )
            if row.best_boundary_changed:
                lines.append(f" Best Boundary: **b{row.left_best_boundary} → b{row.right_best_boundary}**.\n")
            else:
                lines.append("\n")
    if issues:
        lines.append("\n")
        for issue in issues:
            lines.append(f"- {issue}\n")
    return InterleavingComparisonReport(
        left_name=left_report.source.display_name,
        right_name=right_report.source.display_name,
        provider_summaries=rows,
        summary_markdown="".join(lines),
        issues=issues,
    )


def comparison_interleaving_rows(report: InterleavingComparisonReport) -> List[Dict[str, Any]]:
    return [
        {
            "provider": row.provider_tag,
            "stage1_provider": row.stage1_provider,
            "stage2_provider": row.stage2_provider,
            "left_candidate_count": row.left_candidate_count,
            "right_candidate_count": row.right_candidate_count,
            "left_best_boundary": row.left_best_boundary,
            "right_best_boundary": row.right_best_boundary,
            "left_best_cycle_ms_cons": row.left_best_cycle_ms_cons,
            "right_best_cycle_ms_cons": row.right_best_cycle_ms_cons,
            "left_best_pipeline_fps_cons": row.left_best_pipeline_fps_cons,
            "right_best_pipeline_fps_cons": row.right_best_pipeline_fps_cons,
            "fps_delta_abs": row.fps_delta_abs,
            "fps_delta_pct": row.fps_delta_pct,
            "left_gain_vs_best_single_full_pct": row.left_gain_vs_best_single_full_pct,
            "right_gain_vs_best_single_full_pct": row.right_gain_vs_best_single_full_pct,
            "gain_delta_pct": row.gain_delta_pct,
            "left_latency_delta_vs_best_single_full_ms": row.left_latency_delta_vs_best_single_full_ms,
            "right_latency_delta_vs_best_single_full_ms": row.right_latency_delta_vs_best_single_full_ms,
            "latency_delta_ms": row.latency_delta_ms,
            "best_boundary_changed": row.best_boundary_changed,
        }
        for row in report.provider_summaries
    ]


def build_comparison_interleaving_fps_delta_figure(report: InterleavingComparisonReport) -> Figure:
    rows = report.provider_summaries
    if not rows:
        return _placeholder_figure("Interleaving FPS Δ", "Keine heterogenen Pipelines im Vergleich verfügbar.")
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    labels = [_label_for_provider(r.stage1_provider, r.stage2_provider, r.provider_tag) for r in rows]
    x = np.arange(len(labels), dtype=float)
    w = 0.35
    fps_vals = [r.fps_delta_pct if r.fps_delta_pct is not None else float("nan") for r in rows]
    gain_vals = [r.gain_delta_pct if r.gain_delta_pct is not None else float("nan") for r in rows]
    ax.bar(x - w / 2.0, fps_vals, width=w, label="Δ pipeline FPS [%]")
    ax.bar(x + w / 2.0, gain_vals, width=w, label="Δ gain vs best single [pct-pt]")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Lauf B - Lauf A")
    ax.set_title("Interleaving / Throughput Vergleich")
    ax.legend()
    fig.tight_layout()
    return fig


def _build_comparison_caption_lines(report: InterleavingComparisonReport) -> List[str]:
    return [
        "# Interleaving / Throughput Vergleich – Figure Captions\n",
        "- `benchmark_comparison_interleaving_fps_delta.*`: Änderung der konservativen Pipeline-FPS und des FPS-Gewinns vs best single zwischen Lauf A und Lauf B.\n",
    ]


def export_interleaving_comparison(report: InterleavingComparisonReport, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    summary_md = output_dir / "benchmark_comparison_interleaving_summary.md"
    summary_md.write_text(report.summary_markdown, encoding="utf-8")
    paths["summary_md"] = summary_md
    provider_csv = output_dir / "benchmark_comparison_interleaving_provider_summary.csv"
    _write_csv(provider_csv, comparison_interleaving_rows(report))
    paths["provider_csv"] = provider_csv
    captions_md = output_dir / "benchmark_comparison_interleaving_captions.md"
    captions_md.write_text("".join(_build_comparison_caption_lines(report)), encoding="utf-8")
    paths["captions_md"] = captions_md
    plots = {"interleaving_fps_delta": build_comparison_interleaving_fps_delta_figure(report)}
    for name, fig in plots.items():
        for ext in ("png", "pdf", "svg"):
            out_path = output_dir / f"benchmark_comparison_{name}.{ext}"
            fig.savefig(out_path, bbox_inches="tight")
            paths[f"plot_{name}_{ext}"] = out_path
    return paths


def export_metric_audit_comparison(
    left_report: BenchmarkAnalysisReport,
    left_analysis: InterleavingAnalysisReport,
    right_report: BenchmarkAnalysisReport,
    right_analysis: InterleavingAnalysisReport,
    output_dir: Path,
    *,
    use_calibration: bool = True,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    rows = metric_audit_comparison_rows(left_report, left_analysis, right_report, right_analysis, limit=None, use_calibration=use_calibration)
    csv_path = output_dir / 'benchmark_comparison_metric_audit.csv'
    _write_csv(csv_path, rows)
    paths['metric_audit_csv'] = csv_path
    fig = build_comparison_metric_audit_rank_error_figure(left_report, left_analysis, right_report, right_analysis, use_calibration=use_calibration)
    for ext in ('png', 'pdf', 'svg'):
        out_path = output_dir / f'benchmark_comparison_metric_audit_rank_error.{ext}'
        fig.savefig(out_path, bbox_inches='tight')
        paths[f'metric_audit_rank_error_{ext}'] = out_path
    return paths


# ---------------------------------------------------------------------------
# Publication-oriented exports
# ---------------------------------------------------------------------------

def _latex_escape(value: Any) -> str:
    s = str(value if value is not None else '-')
    repl = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
    }
    return ''.join(repl.get(ch, ch) for ch in s)


def _write_tex_table(path: Path, rows: List[Dict[str, Any]], columns: List[tuple[str, str]], *, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('% empty table\n', encoding='utf-8')
        return
    align = 'l' + 'c' * max(0, len(columns) - 1)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        rf'\caption{{{_latex_escape(caption)}}}',
        rf'\label{{{_latex_escape(label)}}}',
        rf'\begin{{tabular}}{{{align}}}',
        r'\toprule',
        ' & '.join(_latex_escape(title) for _, title in columns) + r' \\',
        r'\midrule',
    ]
    for row in rows:
        vals = []
        for key, _title in columns:
            val = row.get(key)
            if isinstance(val, float):
                vals.append(_latex_escape(f"{val:.2f}"))
            else:
                vals.append(_latex_escape(val))
        lines.append(' & '.join(vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def research_comparison_rows(
    left_report: BenchmarkAnalysisReport,
    left_analysis: InterleavingAnalysisReport,
    right_report: BenchmarkAnalysisReport,
    right_analysis: InterleavingAnalysisReport,
    *,
    use_calibration: bool = True,
) -> List[Dict[str, Any]]:
    left_cards = research_summary_cards(left_report, left_analysis)
    right_cards = research_summary_cards(right_report, right_analysis)
    metric_cmp = metric_audit_comparison_summary(left_report, left_analysis, right_report, right_analysis, use_calibration=use_calibration)
    return [
        {'metric': 'Best full FPS', 'left': left_cards.get('best_full_fps'), 'right': right_cards.get('best_full_fps'), 'delta': _delta(right_cards.get('best_full_fps'), left_cards.get('best_full_fps'))},
        {'metric': 'Best split FPS', 'left': left_cards.get('best_split_fps'), 'right': right_cards.get('best_split_fps'), 'delta': _delta(right_cards.get('best_split_fps'), left_cards.get('best_split_fps'))},
        {'metric': 'Δ vs best full [%]', 'left': left_cards.get('delta_vs_best_full_pct'), 'right': right_cards.get('delta_vs_best_full_pct'), 'delta': _delta(right_cards.get('delta_vs_best_full_pct'), left_cards.get('delta_vs_best_full_pct'))},
        {'metric': 'Old Spearman ρ', 'left': metric_cmp.get('left_old_spearman_rank'), 'right': metric_cmp.get('right_old_spearman_rank'), 'delta': metric_cmp.get('old_spearman_rank_delta')},
        {'metric': 'Raw Spearman ρ', 'left': metric_cmp.get('left_uncal_spearman_rank'), 'right': metric_cmp.get('right_uncal_spearman_rank'), 'delta': metric_cmp.get('uncal_spearman_rank_delta')},
        {'metric': 'Cal Spearman ρ', 'left': metric_cmp.get('left_cal_spearman_rank'), 'right': metric_cmp.get('right_cal_spearman_rank'), 'delta': metric_cmp.get('cal_spearman_rank_delta')},
        {'metric': 'Old Kendall τ', 'left': metric_cmp.get('left_old_kendall_rank'), 'right': metric_cmp.get('right_old_kendall_rank'), 'delta': metric_cmp.get('old_kendall_rank_delta')},
        {'metric': 'Raw Kendall τ', 'left': metric_cmp.get('left_uncal_kendall_rank'), 'right': metric_cmp.get('right_uncal_kendall_rank'), 'delta': metric_cmp.get('uncal_kendall_rank_delta')},
        {'metric': 'Cal Kendall τ', 'left': metric_cmp.get('left_cal_kendall_rank'), 'right': metric_cmp.get('right_cal_kendall_rank'), 'delta': metric_cmp.get('cal_kendall_rank_delta')},
        {'metric': 'Top-1 hit (old)', 'left': 'yes' if metric_cmp.get('left_old_top1_hit') else 'no', 'right': 'yes' if metric_cmp.get('right_old_top1_hit') else 'no', 'delta': ''},
        {'metric': 'Top-1 hit (raw)', 'left': 'yes' if metric_cmp.get('left_uncal_top1_hit') else 'no', 'right': 'yes' if metric_cmp.get('right_uncal_top1_hit') else 'no', 'delta': ''},
        {'metric': 'Top-1 hit (cal)', 'left': 'yes' if metric_cmp.get('left_cal_top1_hit') else 'no', 'right': 'yes' if metric_cmp.get('right_cal_top1_hit') else 'no', 'delta': ''},
        {'metric': 'Top-3 hit (old)', 'left': 'yes' if metric_cmp.get('left_old_top3_hit') else 'no', 'right': 'yes' if metric_cmp.get('right_old_top3_hit') else 'no', 'delta': ''},
        {'metric': 'Top-3 hit (raw)', 'left': 'yes' if metric_cmp.get('left_uncal_top3_hit') else 'no', 'right': 'yes' if metric_cmp.get('right_uncal_top3_hit') else 'no', 'delta': ''},
        {'metric': 'Top-3 hit (cal)', 'left': 'yes' if metric_cmp.get('left_cal_top3_hit') else 'no', 'right': 'yes' if metric_cmp.get('right_cal_top3_hit') else 'no', 'delta': ''},
    ]


def export_publication_analysis(report: BenchmarkAnalysisReport, analysis: InterleavingAnalysisReport, output_dir: Path, *, use_calibration: bool = True) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    objective = str(report.summary.get('objective') or 'throughput')
    summary = metric_audit_summary(report, analysis, use_calibration=use_calibration)
    summary_md = output_dir / 'publication_summary.md'
    summary_md.write_text(
        f"# Publication-ready benchmark summary\n\n"
        f"Objective: **{objective}**\n\n"
        f"Calibration: **{'calibrated' if use_calibration else 'raw'}**"
        + (f" (profile `{summary.get('calibration_profile')}`)" if summary.get('calibration_profile') else '')
        + "\n\n"
        f"Old Spearman/Kendall: {summary.get('old_spearman_rank')} / {summary.get('old_kendall_rank')}\n\n"
        f"Raw Spearman/Kendall: {summary.get('uncal_spearman_rank')} / {summary.get('uncal_kendall_rank')}\n\n"
        f"Cal Spearman/Kendall: {summary.get('cal_spearman_rank')} / {summary.get('cal_kendall_rank')}\n",
        encoding='utf-8',
    )
    paths['publication_summary_md'] = summary_md

    best_rows = research_best_full_vs_split_rows(report, analysis)
    best_csv = output_dir / 'publication_best_full_vs_split.csv'
    _write_csv(best_csv, best_rows)
    paths['publication_best_csv'] = best_csv
    best_tex = output_dir / 'publication_best_full_vs_split.tex'
    _write_tex_table(best_tex, best_rows, [
        ('role', 'Role'), ('provider_or_pipeline', 'Provider / Pipeline'), ('boundary', 'Boundary'), ('latency_ms', 'Latency [ms]'), ('fps_equiv', 'FPS'), ('delta_vs_best_full_pct', 'Δ vs best full [%]'), ('predicted_rank', 'Pred rank'), ('cut_mib', 'Cut [MiB]')
    ], caption='Best full deployment versus best split deployment.', label='tab:pub_best_full_vs_split')
    paths['publication_best_tex'] = best_tex

    audit_rows = research_prediction_audit_rows(report, analysis, limit=None)
    audit_csv = output_dir / 'publication_prediction_audit.csv'
    _write_csv(audit_csv, audit_rows)
    paths['publication_audit_csv'] = audit_csv
    audit_tex = output_dir / 'publication_prediction_audit.tex'
    _write_tex_table(audit_tex, audit_rows[:15], [
        ('boundary', 'Boundary'), ('provider', 'Pipeline'), ('predicted_rank', 'Pred rank'), ('actual_rank', 'Actual rank'), ('measured_streaming_fps', 'Measured FPS'), ('streaming_efficiency_pct', 'Eff. [%]'), ('residual_overhead_ms', 'Residual [ms]'), ('cut_mib', 'Cut [MiB]'), ('compile_risk', 'Compile risk')
    ], caption='Prediction audit for the strongest benchmarked split candidates.', label='tab:pub_prediction_audit')
    paths['publication_audit_tex'] = audit_tex

    stage_rows = research_stage_breakdown_rows(analysis, limit=None)
    stage_csv = output_dir / 'publication_stage_breakdown.csv'
    _write_csv(stage_csv, stage_rows)
    paths['publication_stage_csv'] = stage_csv
    stage_tex = output_dir / 'publication_stage_breakdown.tex'
    _write_tex_table(stage_tex, stage_rows[:15], [
        ('boundary', 'Boundary'), ('provider', 'Pipeline'), ('part1_mean_ms', 'Stage1 [ms]'), ('part2_mean_ms', 'Stage2 [ms]'), ('ideal_bottleneck_fps', 'Ideal FPS'), ('measured_streaming_fps', 'Measured FPS'), ('streaming_efficiency_pct', 'Eff. [%]'), ('residual_overhead_ms', 'Residual [ms]')
    ], caption='Stage-level breakdown for heterogeneous streaming candidates.', label='tab:pub_stage_breakdown')
    paths['publication_stage_tex'] = stage_tex

    raw_rows = metric_audit_rows(report, analysis, limit=None, use_calibration=False)
    raw_csv = output_dir / 'publication_metric_audit_raw.csv'
    _write_csv(raw_csv, raw_rows)
    paths['publication_metric_raw_csv'] = raw_csv
    raw_tex = output_dir / 'publication_metric_audit_raw.tex'
    _write_tex_table(raw_tex, raw_rows[:15], [
        ('boundary', 'Boundary'), ('provider', 'Pipeline'), ('predicted_rank_old', 'Old rank'), ('predicted_rank_uncal', 'Raw TH rank'), ('actual_rank', 'Actual rank'), ('predicted_handover_ms_uncal', 'HO raw [ms]'), ('predicted_stream_fps_uncal', 'Raw TH FPS'), ('measured_streaming_fps', 'Measured FPS'), ('rank_error_old_abs', '|Δ old|'), ('rank_error_uncal_abs', '|Δ raw|')
    ], caption='Metric audit using the uncalibrated throughput heuristic.', label='tab:pub_metric_audit_raw')
    paths['publication_metric_raw_tex'] = raw_tex

    cal_rows = metric_audit_rows(report, analysis, limit=None, use_calibration=True)
    cal_csv = output_dir / 'publication_metric_audit_cal.csv'
    _write_csv(cal_csv, cal_rows)
    paths['publication_metric_cal_csv'] = cal_csv
    cal_tex = output_dir / 'publication_metric_audit_cal.tex'
    _write_tex_table(cal_tex, cal_rows[:15], [
        ('boundary', 'Boundary'), ('provider', 'Pipeline'), ('predicted_rank_old', 'Old rank'), ('predicted_rank_cal', 'Cal TH rank'), ('actual_rank', 'Actual rank'), ('predicted_handover_ms_cal', 'HO cal [ms]'), ('predicted_stream_fps_cal', 'Cal TH FPS'), ('measured_streaming_fps', 'Measured FPS'), ('rank_error_old_abs', '|Δ old|'), ('rank_error_cal_abs', '|Δ cal|')
    ], caption='Metric audit using the calibrated throughput heuristic.', label='tab:pub_metric_audit_cal')
    paths['publication_metric_cal_tex'] = cal_tex
    return paths


def export_publication_comparison(
    left_report: BenchmarkAnalysisReport,
    left_analysis: InterleavingAnalysisReport,
    right_report: BenchmarkAnalysisReport,
    right_analysis: InterleavingAnalysisReport,
    output_dir: Path,
    *,
    use_calibration: bool = True,
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    cmp_rows = research_comparison_rows(left_report, left_analysis, right_report, right_analysis, use_calibration=use_calibration)
    csv_path = output_dir / 'publication_comparison_research_summary.csv'
    _write_csv(csv_path, cmp_rows)
    paths['publication_comparison_csv'] = csv_path
    cmp_tex = output_dir / 'publication_comparison_research_summary.tex'
    _write_tex_table(cmp_tex, cmp_rows, [('metric', 'Metric'), ('left', 'Run A'), ('right', 'Run B'), ('delta', 'Δ')], caption='Compact A/B research summary for publication use.', label='tab:pub_comparison_research_summary')
    paths['publication_comparison_tex'] = cmp_tex

    metric_rows = metric_audit_comparison_rows(left_report, left_analysis, right_report, right_analysis, limit=None, use_calibration=use_calibration)
    metric_csv = output_dir / 'publication_comparison_metric_audit.csv'
    _write_csv(metric_csv, metric_rows)
    paths['publication_comparison_metric_csv'] = metric_csv
    metric_tex = output_dir / 'publication_comparison_metric_audit.tex'
    _write_tex_table(metric_tex, metric_rows[:15], [
        ('boundary', 'Boundary'), ('left_actual_rank', 'A actual'), ('right_actual_rank', 'B actual'), ('left_predicted_rank_old', 'A old'), ('right_predicted_rank_old', 'B old'), ('left_predicted_rank_cal', 'A cal'), ('right_predicted_rank_cal', 'B cal'), ('old_rank_error_abs_delta', 'Δ |old error|'), ('cal_rank_error_abs_delta', 'Δ |cal error|'), ('fps_delta_pct', 'Δ FPS [%]')
    ], caption='A/B comparison of metric-audit behaviour.', label='tab:pub_comparison_metric_audit')
    paths['publication_comparison_metric_tex'] = metric_tex

    summary_md = output_dir / 'publication_comparison_summary.md'
    summary = metric_audit_comparison_summary(left_report, left_analysis, right_report, right_analysis, use_calibration=use_calibration)
    summary_md.write_text(
        f"# Publication-ready comparison summary\n\n"
        f"Calibration mode: **{'calibrated' if use_calibration else 'raw'}**"
        + (f" (profile `{summary.get('calibration_profile')}`)" if summary.get('calibration_profile') else '')
        + "\n\n"
        f"Δ old Spearman/Kendall: {summary.get('old_spearman_rank_delta')} / {summary.get('old_kendall_rank_delta')}\n\n"
        f"Δ raw Spearman/Kendall: {summary.get('uncal_spearman_rank_delta')} / {summary.get('uncal_kendall_rank_delta')}\n\n"
        f"Δ cal Spearman/Kendall: {summary.get('cal_spearman_rank_delta')} / {summary.get('cal_kendall_rank_delta')}\n",
        encoding='utf-8',
    )
    paths['publication_comparison_summary_md'] = summary_md
    return paths
