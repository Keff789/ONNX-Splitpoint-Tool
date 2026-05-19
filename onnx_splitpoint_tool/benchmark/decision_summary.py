"""Compact decision-oriented benchmark analysis.

The classic benchmark-analysis module intentionally exposes many engineering
metrics. This module distills the same raw data into the few numbers that are
usually needed to decide whether a splitpoint is useful:

* best single/full baseline
* best streaming split candidate
* latency tradeoff
* task-aware quality metric (Mini-COCO AP50 or Top-1) / backend drift indicators
* a short recommendation per provider/pipeline
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from matplotlib.figure import Figure

from .analysis import BenchmarkAnalysisReport
from .interleaving_analysis import (
    InterleavingAnalysisReport,
    interleaving_provider_rows,
    research_prediction_audit_rows,
    research_stage_breakdown_rows,
)


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


def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
    except Exception:
        return "-"
    if not math.isfinite(xf):
        return "-"
    return f"{xf:.{nd}f}"



def _fmt_boundary(v: Any) -> str:
    b = _as_int(v)
    if b is None:
        return "-"
    if 0 <= b < 1000:
        return f"b{b:03d}"
    return f"b{b}"

def _fps_from_ms(ms: Optional[float]) -> Optional[float]:
    if ms is None or ms <= 0:
        return None
    return 1000.0 / float(ms)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _placeholder_figure(title: str, message: str) -> Figure:
    fig = Figure(figsize=(7.2, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig


def _provider_best_row(report: BenchmarkAnalysisReport, tag: str, boundary: Optional[int]) -> Optional[Dict[str, Any]]:
    rows = report.provider_rows.get(str(tag), [])
    if not rows:
        return None
    if boundary is not None:
        for row in rows:
            if _as_int(row.get("boundary")) == int(boundary):
                return row
    best = None
    best_ms = float("inf")
    for row in rows:
        ms = _as_float(row.get("composed_mean_ms"))
        if ms is None:
            continue
        if ms < best_ms:
            best_ms = ms
            best = row
    return best


def _first_number(row: Optional[Dict[str, Any]], keys: List[str]) -> Optional[float]:
    if not row:
        return None
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _ap50_primary(row: Optional[Dict[str, Any]]) -> Optional[float]:
    return _first_number(
        row,
        [
            "mini_coco_ap50_primary",
            "mini_coco_ap50_composed",
            "mini_coco_ap50_full",
            "mini_coco_ap50_part2",
            "mini_coco_ap50_part1",
        ],
    )


def _ap50_full(row: Optional[Dict[str, Any]]) -> Optional[float]:
    return _first_number(row, ["mini_coco_ap50_full"])


def _ap50_delta(row: Optional[Dict[str, Any]]) -> Optional[float]:
    direct = _first_number(row, ["mini_coco_ap50_delta_primary_minus_full"])
    if direct is not None:
        return direct
    primary = _ap50_primary(row)
    full = _ap50_full(row)
    if primary is not None and full is not None:
        return primary - full
    return None


def _backend_match(row: Optional[Dict[str, Any]]) -> Optional[float]:
    return _first_number(row, ["backend_drift_dataset_global_match_ratio", "backend_drift_single_match_ratio"])


def _backend_iou(row: Optional[Dict[str, Any]]) -> Optional[float]:
    return _first_number(row, ["backend_drift_dataset_global_mean_iou", "backend_drift_single_mean_iou"])


def _row_task(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "detection"
    raw = str(row.get("benchmark_task") or row.get("benchmark_task_used") or row.get("benchmark_task_requested") or row.get("validation_dataset_task") or "").strip().lower()
    if raw == "classification":
        return "classification"
    if raw == "detection":
        return "detection"
    if any(row.get(k) is not None for k in (
        "mini_classification_eval_primary_top1",
        "mini_classification_eval_full_top1",
        "validation_dataset_primary_top1_accuracy",
        "backend_drift_dataset_top1_agreement",
    )):
        return "classification"
    return "detection"


def _quality_label(task: str) -> str:
    return "Top-1" if str(task or "detection") == "classification" else "AP50"


def _quality_primary(row: Optional[Dict[str, Any]]) -> Optional[float]:
    if _row_task(row) == "classification":
        return _first_number(
            row,
            [
                "mini_classification_eval_primary_top1",
                "mini_classification_eval_composed_top1",
                "mini_classification_eval_full_top1",
                "mini_classification_eval_part2_top1",
                "mini_classification_eval_part1_top1",
                "validation_dataset_primary_top1_accuracy",
                "validation_dataset_primary_top1_agreement",
            ],
        )
    return _ap50_primary(row)


def _quality_full(row: Optional[Dict[str, Any]]) -> Optional[float]:
    if _row_task(row) == "classification":
        return _first_number(row, ["mini_classification_eval_full_top1"])
    return _ap50_full(row)


def _quality_delta(row: Optional[Dict[str, Any]]) -> Optional[float]:
    if _row_task(row) == "classification":
        direct = _first_number(row, ["mini_classification_eval_delta_primary_minus_full_top1"])
        if direct is not None:
            return direct
        primary = _quality_primary(row)
        full = _quality_full(row)
        if primary is not None and full is not None:
            return primary - full
        return None
    return _ap50_delta(row)


def _backend_match_generic(row: Optional[Dict[str, Any]]) -> Optional[float]:
    if _row_task(row) == "classification":
        return _first_number(
            row,
            [
                "backend_drift_dataset_top1_agreement",
                "backend_drift_single_top1_match",
                "backend_drift_dataset_top5_agreement",
                "backend_drift_single_top5_agreement",
                "backend_drift_dataset_mean_cosine_similarity",
                "backend_drift_single_cosine_similarity",
            ],
        )
    return _backend_match(row)


def _backend_metric_label(task: str) -> str:
    return "Top-1 agree vs CPU" if str(task or "detection") == "classification" else "Match vs CPU"


def _status_text(row: Optional[Dict[str, Any]]) -> str:
    if not row:
        return "unknown"
    passed = _as_bool(row.get("final_pass_all"))
    task = _row_task(row)
    quality_status = str(
        row.get("mini_classification_eval_status") if task == "classification" else row.get("mini_coco_ap50_status") or ""
    ).strip().lower()
    drift_status = str(row.get("backend_drift_status") or "").strip().lower()
    if passed is False:
        return "fail"
    if quality_status and quality_status not in {"ok", "disabled", "n/a", "na"}:
        prefix = "top1" if task == "classification" else "ap50"
        return f"{prefix}:{quality_status}"
    if drift_status and drift_status not in {"ok", "disabled", "n/a", "na"}:
        return f"drift:{drift_status}"
    if passed is True:
        return "ok"
    return "n/a"


def _slug(text: Any) -> str:
    return str(text or "").strip().lower().replace("-", "_")


def _is_hailo_token(text: Any) -> bool:
    return _slug(text).startswith("hailo")


def _context_is_multi(mode: Any, count: Any) -> bool:
    m = _slug(mode)
    c = _as_int(count)
    if "failed_to_multi" in m or m == "multi_context_used" or m.startswith("multi"):
        return True
    return c is not None and c > 1


def _context_label(mode: Any, count: Any) -> str:
    m = _slug(mode)
    c = _as_int(count)
    if not m and c is None:
        return "-"
    if m == "single_context_failed_to_multi":
        base = "single→multi"
    elif m == "multi_context_used":
        base = "multi"
    elif m == "single_context_used":
        base = "single"
    elif m in {"unknown", "skipped", "failed"}:
        base = m
    elif m:
        base = m.replace("_context", "").replace("_", " ")
    else:
        base = "context"
    return f"{base}({c})" if c is not None else base


def _empty_hailo_context() -> Dict[str, Any]:
    return {
        "hailo_context_role": "-",
        "hailo_context_label": "-",
        "hailo_context_mode": None,
        "hailo_context_count": None,
        "hailo_multi_context": False,
    }


def _hailo_arch_matches(row_arch: Any, provider_token: Any) -> bool:
    arch = _slug(row_arch)
    token = _slug(provider_token).replace("_auto", "")
    if not arch or not token:
        return True
    return token.startswith(arch) or arch.startswith(token)


def _hailo_context_for(
    report: BenchmarkAnalysisReport,
    boundary: Any,
    stage1: Any = None,
    stage2: Any = None,
    setup: Any = None,
) -> Dict[str, Any]:
    """Return Hailo context metadata for a decision row.

    Multi-context is intentionally not treated as invalid here. It is a
    visibility/risk label; the measured FPS/AP50 numbers decide whether a
    candidate is useful.
    """
    b = _as_int(boundary)
    if b is None:
        return _empty_hailo_context()
    rows = [r for r in getattr(report, "hailo_context_summaries", []) if _as_int(getattr(r, "boundary", None)) == int(b)]
    if not rows:
        return _empty_hailo_context()

    s1 = _slug(stage1)
    s2 = _slug(stage2)
    st = _slug(setup)

    def pick_for(token: str):
        for r in rows:
            if _hailo_arch_matches(getattr(r, "hw_arch", ""), token):
                return r
        return rows[0]

    if _is_hailo_token(s1) and _is_hailo_token(s2):
        r = pick_for(s1 or s2 or st)
        p1_label = _context_label(getattr(r, "part1_context_mode", None), getattr(r, "part1_context_count", None))
        p2_label = _context_label(getattr(r, "part2_context_mode", None), getattr(r, "part2_context_count", None))
        p1_multi = _context_is_multi(getattr(r, "part1_context_mode", None), getattr(r, "part1_context_count", None))
        p2_multi = _context_is_multi(getattr(r, "part2_context_mode", None), getattr(r, "part2_context_count", None))
        counts = [c for c in (_as_int(getattr(r, "part1_context_count", None)), _as_int(getattr(r, "part2_context_count", None))) if c is not None]
        return {
            "hailo_context_role": "part1+part2",
            "hailo_context_label": f"p1:{p1_label}; p2:{p2_label}",
            "hailo_context_mode": f"part1={getattr(r, 'part1_context_mode', None)};part2={getattr(r, 'part2_context_mode', None)}",
            "hailo_context_count": max(counts) if counts else None,
            "hailo_multi_context": bool(p1_multi or p2_multi),
        }

    if _is_hailo_token(s1):
        r = pick_for(s1)
        mode = getattr(r, "part1_context_mode", None)
        count = getattr(r, "part1_context_count", None)
        return {
            "hailo_context_role": "part1",
            "hailo_context_label": _context_label(mode, count),
            "hailo_context_mode": mode,
            "hailo_context_count": _as_int(count),
            "hailo_multi_context": _context_is_multi(mode, count),
        }

    if _is_hailo_token(s2):
        r = pick_for(s2)
        mode = getattr(r, "part2_context_mode", None)
        count = getattr(r, "part2_context_count", None)
        return {
            "hailo_context_role": "part2",
            "hailo_context_label": _context_label(mode, count),
            "hailo_context_mode": mode,
            "hailo_context_count": _as_int(count),
            "hailo_multi_context": _context_is_multi(mode, count),
        }

    if _is_hailo_token(st):
        r = pick_for(st)
        mode = getattr(r, "part2_context_mode", None)
        count = getattr(r, "part2_context_count", None)
        return {
            "hailo_context_role": "part2",
            "hailo_context_label": _context_label(mode, count),
            "hailo_context_mode": mode,
            "hailo_context_count": _as_int(count),
            "hailo_multi_context": _context_is_multi(mode, count),
        }

    return _empty_hailo_context()


def _best_full_provider(report: BenchmarkAnalysisReport) -> Optional[Any]:
    candidates = [p for p in report.providers if p.full_baseline_ms is not None and p.full_baseline_ms > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda p: float(p.full_baseline_ms or float("inf")))


def _best_full_row(report: BenchmarkAnalysisReport, tag: str) -> Optional[Dict[str, Any]]:
    rows = report.provider_rows.get(str(tag), [])
    if not rows:
        return None
    # Any row from this provider contains the repeated full AP50/drift fields.
    return rows[0]


def _best_interleaving_provider_row(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> Optional[Dict[str, Any]]:
    rows = interleaving_provider_rows(inter)
    if not rows:
        return None
    usable = [r for r in rows if _as_float(r.get("measured_streaming_fps") or r.get("best_pipeline_fps_cons")) is not None]
    if not usable:
        return None
    return max(usable, key=lambda r: float(_as_float(r.get("measured_streaming_fps") or r.get("best_pipeline_fps_cons")) or -1.0))


def _row_recommendation(
    *,
    row_type: str,
    gain_pct: Optional[float],
    quality_delta: Optional[float],
    backend_match: Optional[float],
    status: str,
    hailo_multi_context: bool = False,
) -> str:
    def with_context_note(text: str) -> str:
        if hailo_multi_context and status != "fail":
            return text + " Multi-Context: gültig, aber längeren Stream-Test prüfen."
        return text

    if status == "fail":
        return "Nicht verwenden: Gate fehlgeschlagen."
    if row_type == "full":
        return with_context_note("Baseline: gut für niedrigste Einzelbild-Latenz.")
    if row_type == "sequential_split":
        if gain_pct is not None and gain_pct > 3.0:
            return with_context_note("Sequenziell schneller als Full; als Latenz-Kandidat prüfen.")
        return with_context_note("Sequenziell nicht attraktiv; nur aus Architekturgründen splitten.")
    acc_warn = quality_delta is not None and quality_delta < -0.03
    drift_warn = backend_match is not None and backend_match < 0.85
    if gain_pct is not None and gain_pct >= 15.0 and not acc_warn:
        if drift_warn:
            return with_context_note("Durchsatz stark, aber Backend-Drift sichtbar; Accuracy prüfen.")
        return with_context_note("Top-Kandidat für Streaming-FPS.")
    if gain_pct is not None and gain_pct >= 5.0:
        return with_context_note("Möglicher Streaming-Kandidat; Tradeoff prüfen.")
    return with_context_note("Kein klarer Streaming-Vorteil.")


def decision_provider_rows(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> List[Dict[str, Any]]:
    """Return one compact decision row per full provider and streaming pipeline."""

    rows: List[Dict[str, Any]] = []
    best_full = _best_full_provider(report)
    best_full_ms = float(best_full.full_baseline_ms) if best_full and best_full.full_baseline_ms else None
    best_full_fps = _fps_from_ms(best_full_ms)

    for ps in report.providers:
        full_ms = ps.full_baseline_ms
        full_fps = _fps_from_ms(full_ms)
        full_row = _best_full_row(report, ps.tag)
        if full_ms is not None and full_ms > 0:
            ctx = _empty_hailo_context()
            rows.append({
                "kind": "full", "setup": ps.tag, "boundary": None,
                "latency_ms": full_ms, "streaming_fps": full_fps,
                "gain_vs_best_full_pct": (None if best_full_fps in (None, 0) or full_fps is None else (full_fps - best_full_fps) / best_full_fps * 100.0),
                "task": _row_task(full_row), "quality_label": _quality_label(_row_task(full_row)), "backend_metric_label": _backend_metric_label(_row_task(full_row)),
                "quality": _quality_full(full_row) or _quality_primary(full_row),
                "ap50": _quality_full(full_row) or _quality_primary(full_row),
                "quality_delta_vs_full": 0.0 if (_quality_full(full_row) is not None or _quality_primary(full_row) is not None) else None,
                "ap50_delta_vs_full": 0.0 if (_quality_full(full_row) is not None or _quality_primary(full_row) is not None) else None,
                "backend_match_vs_cpu": _backend_match_generic(full_row),
                "backend_iou_vs_cpu": _backend_iou(full_row),
                "hailo_context_role": ctx.get("hailo_context_role"),
                "hailo_context_label": ctx.get("hailo_context_label"),
                "hailo_context_mode": ctx.get("hailo_context_mode"),
                "hailo_context_count": ctx.get("hailo_context_count"),
                "hailo_multi_context": ctx.get("hailo_multi_context"),
                "status": _status_text(full_row),
                "recommendation": _row_recommendation(row_type="full", gain_pct=0.0, quality_delta=0.0, backend_match=_backend_match_generic(full_row), status=_status_text(full_row), hailo_multi_context=False),
            })

        if ps.best_boundary is not None and ps.best_composed_ms is not None:
            best_row = _provider_best_row(report, ps.tag, ps.best_boundary)
            split_fps = _fps_from_ms(ps.best_composed_ms)
            gain_vs_full = None
            if full_fps not in (None, 0) and split_fps is not None:
                gain_vs_full = (split_fps - full_fps) / full_fps * 100.0
            ctx = _hailo_context_for(report, ps.best_boundary, stage1=(best_row or {}).get("stage1_provider"), stage2=(best_row or {}).get("stage2_provider"), setup=ps.tag)
            rows.append({
                "kind": "sequential_split", "setup": ps.tag, "boundary": ps.best_boundary,
                "latency_ms": ps.best_composed_ms, "streaming_fps": split_fps,
                "gain_vs_best_full_pct": (None if best_full_fps in (None, 0) or split_fps is None else (split_fps - best_full_fps) / best_full_fps * 100.0),
                "gain_vs_own_full_pct": gain_vs_full,
                "task": _row_task(best_row), "quality_label": _quality_label(_row_task(best_row)), "backend_metric_label": _backend_metric_label(_row_task(best_row)),
                "quality": _quality_primary(best_row), "ap50": _quality_primary(best_row), "quality_delta_vs_full": _quality_delta(best_row), "ap50_delta_vs_full": _quality_delta(best_row),
                "backend_match_vs_cpu": _backend_match_generic(best_row), "backend_iou_vs_cpu": _backend_iou(best_row),
                "hailo_context_role": ctx.get("hailo_context_role"),
                "hailo_context_label": ctx.get("hailo_context_label"),
                "hailo_context_mode": ctx.get("hailo_context_mode"),
                "hailo_context_count": ctx.get("hailo_context_count"),
                "hailo_multi_context": ctx.get("hailo_multi_context"),
                "status": _status_text(best_row),
                "recommendation": _row_recommendation(row_type="sequential_split", gain_pct=gain_vs_full, quality_delta=_quality_delta(best_row), backend_match=_backend_match_generic(best_row), status=_status_text(best_row), hailo_multi_context=bool(ctx.get("hailo_multi_context"))),
            })

    for ir in interleaving_provider_rows(inter):
        boundary = _as_int(ir.get("best_boundary"))
        tag = str(ir.get("provider") or "")
        src_row = _provider_best_row(report, tag, boundary)
        fps = _as_float(ir.get("measured_streaming_fps") or ir.get("best_pipeline_fps_cons"))
        latency = _as_float(ir.get("measured_latency_p50_ms"))
        if latency is None:
            latency = _as_float(src_row.get("throughput_latency_mean_ms") if src_row else None) or _as_float(src_row.get("composed_mean_ms") if src_row else None)
        stage_gain = _as_float(ir.get("gain_vs_best_single_full_pct"))
        global_gain = None
        if fps is not None and best_full_fps not in (None, 0):
            global_gain = (fps - best_full_fps) / best_full_fps * 100.0
        ctx = _hailo_context_for(report, boundary, stage1=ir.get("stage1_provider"), stage2=ir.get("stage2_provider"), setup=tag)
        rows.append({
            "kind": "streaming_split", "setup": tag, "stage1": ir.get("stage1_provider"), "stage2": ir.get("stage2_provider"), "boundary": boundary,
            "latency_ms": latency, "streaming_fps": fps,
            "gain_vs_best_full_pct": global_gain, "gain_vs_stage_best_full_pct": stage_gain,
            "stage_balance": _as_float(ir.get("stage_balance")),
            "task": _row_task(src_row), "quality_label": _quality_label(_row_task(src_row)), "backend_metric_label": _backend_metric_label(_row_task(src_row)),
            "quality": _quality_primary(src_row), "ap50": _quality_primary(src_row), "quality_delta_vs_full": _quality_delta(src_row), "ap50_delta_vs_full": _quality_delta(src_row),
            "backend_match_vs_cpu": _backend_match_generic(src_row), "backend_iou_vs_cpu": _backend_iou(src_row),
            "hailo_context_role": ctx.get("hailo_context_role"),
            "hailo_context_label": ctx.get("hailo_context_label"),
            "hailo_context_mode": ctx.get("hailo_context_mode"),
            "hailo_context_count": ctx.get("hailo_context_count"),
            "hailo_multi_context": ctx.get("hailo_multi_context"),
            "status": _status_text(src_row),
            "recommendation": _row_recommendation(row_type="streaming_split", gain_pct=global_gain, quality_delta=_quality_delta(src_row), backend_match=_backend_match_generic(src_row), status=_status_text(src_row), hailo_multi_context=bool(ctx.get("hailo_multi_context"))),
        })

    def _sort_key(row: Dict[str, Any]) -> tuple:
        kind_order = {"streaming_split": 0, "full": 1, "sequential_split": 2}
        fps = _as_float(row.get("streaming_fps")) or -1.0
        return (kind_order.get(str(row.get("kind")), 9), -fps, str(row.get("setup")))
    return sorted(rows, key=_sort_key)


def decision_candidate_rows(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport, limit: Optional[int] = 8) -> List[Dict[str, Any]]:
    """Return compact top streaming candidates, one row per boundary."""

    rows: List[Dict[str, Any]] = []
    audit_rows = research_prediction_audit_rows(report, inter, limit=None)
    stage_by_key: Dict[tuple, Dict[str, Any]] = {}
    for sr in research_stage_breakdown_rows(inter, limit=None):
        stage_by_key[(int(sr.get("boundary") or -1), str(sr.get("provider") or ""))] = sr
    for item in audit_rows:
        boundary = _as_int(item.get("boundary"))
        provider = str(item.get("provider") or "")
        src_row = _provider_best_row(report, provider, boundary)
        stage = stage_by_key.get((int(boundary or -1), provider), {})
        quality = _quality_primary(src_row)
        task = _row_task(src_row)
        gain = None
        best_full = _best_full_provider(report)
        best_full_fps = _fps_from_ms(float(best_full.full_baseline_ms)) if best_full and best_full.full_baseline_ms else None
        fps = _as_float(item.get("measured_streaming_fps"))
        if fps is not None and best_full_fps not in (None, 0):
            gain = (fps - best_full_fps) / best_full_fps * 100.0
        ctx = _hailo_context_for(report, boundary, stage1=stage.get("stage1_provider"), stage2=stage.get("stage2_provider"), setup=provider)
        rows.append({
            "rank": item.get("actual_rank"), "boundary": boundary, "pipeline": provider,
            "stage1": stage.get("stage1_provider"), "stage2": stage.get("stage2_provider"),
            "streaming_fps": fps, "latency_ms": item.get("measured_latency_ms"), "gain_vs_best_full_pct": gain,
            "task": task, "quality_label": _quality_label(task), "backend_metric_label": _backend_metric_label(task),
            "quality": _quality_primary(src_row), "ap50": _quality_primary(src_row), "quality_delta_vs_full": _quality_delta(src_row), "ap50_delta_vs_full": _quality_delta(src_row),
            "backend_match_vs_cpu": _backend_match_generic(src_row), "backend_iou_vs_cpu": _backend_iou(src_row),
            "hailo_context_role": ctx.get("hailo_context_role"),
            "hailo_context_label": ctx.get("hailo_context_label"),
            "hailo_context_mode": ctx.get("hailo_context_mode"),
            "hailo_context_count": ctx.get("hailo_context_count"),
            "hailo_multi_context": ctx.get("hailo_multi_context"),
            "stage_balance": None if stage.get("part1_mean_ms") is None or stage.get("part2_mean_ms") is None else min(float(stage.get("part1_mean_ms") or 0.0), float(stage.get("part2_mean_ms") or 0.0)) / max(float(stage.get("part1_mean_ms") or 0.0), float(stage.get("part2_mean_ms") or 1.0)),
            "residual_overhead_ms": item.get("residual_overhead_ms"),
            "status": _status_text(src_row),
            "recommendation": _row_recommendation(row_type="streaming_split", gain_pct=gain, quality_delta=_quality_delta(src_row), backend_match=_backend_match_generic(src_row), status=_status_text(src_row), hailo_multi_context=bool(ctx.get("hailo_multi_context"))),
        })
    rows.sort(key=lambda r: (int(r.get("rank") or 9999), -float(_as_float(r.get("streaming_fps")) or -1.0)))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows


def decision_overview(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> Dict[str, Any]:
    best_full = _best_full_provider(report)
    best_stream = _best_interleaving_provider_row(report, inter)
    total_rows = sum(len(rows) for rows in report.provider_rows.values())
    pass_rows = 0
    for rows in report.provider_rows.values():
        for row in rows:
            if _as_bool(row.get("final_pass_all")) is True:
                pass_rows += 1
    best_stream_src = None
    if best_stream:
        best_stream_src = _provider_best_row(report, str(best_stream.get("provider") or ""), _as_int(best_stream.get("best_boundary")))
    best_full_row = _best_full_row(report, best_full.tag) if best_full else None
    best_quality_setup = None
    best_quality_value = None
    for ps in report.providers:
        if ps.full_baseline_ms is None or ps.full_baseline_ms <= 0:
            continue
        row = _best_full_row(report, ps.tag)
        ap = _quality_full(row) or _quality_primary(row)
        if ap is None:
            continue
        if best_quality_value is None or float(ap) > float(best_quality_value):
            best_quality_value = float(ap)
            best_quality_setup = ps.tag
    full_ms = float(best_full.full_baseline_ms) if best_full and best_full.full_baseline_ms else None
    full_fps = _fps_from_ms(full_ms)
    stream_fps = _as_float(best_stream.get("measured_streaming_fps") or best_stream.get("best_pipeline_fps_cons")) if best_stream else None
    stream_gain = None
    if full_fps not in (None, 0) and stream_fps is not None:
        stream_gain = (stream_fps - full_fps) / full_fps * 100.0
    stream_latency = None
    if best_stream:
        stream_latency = _as_float(best_stream.get("measured_latency_p50_ms"))
        if stream_latency is None and best_stream_src:
            stream_latency = _as_float(best_stream_src.get("throughput_latency_mean_ms")) or _as_float(best_stream_src.get("composed_mean_ms"))
    return {
        "case_count": report.summary.get("generated_cases", report.summary.get("case_count")),
        "provider_count": len(report.providers),
        "pass_rows": pass_rows,
        "total_rows": total_rows,
        "best_full_setup": getattr(best_full, "tag", None),
        "best_full_latency_ms": full_ms,
        "best_full_fps": full_fps,
        "best_full_quality": _quality_full(best_full_row) or _quality_primary(best_full_row),
        "best_full_quality_setup": best_quality_setup,
        "best_full_quality_value": best_quality_value,
        "best_full_ap50_setup": best_quality_setup,
        "best_full_ap50_value": best_quality_value,
        "task": _row_task(best_full_row),
        "quality_label": _quality_label(_row_task(best_full_row)),
        "backend_metric_label": _backend_metric_label(_row_task(best_full_row)),
        "best_streaming_setup": best_stream.get("provider") if best_stream else None,
        "best_streaming_boundary": _as_int(best_stream.get("best_boundary")) if best_stream else None,
        "best_streaming_fps": stream_fps,
        "best_streaming_latency_ms": stream_latency,
        "best_streaming_gain_pct": stream_gain,
        "best_streaming_quality": _quality_primary(best_stream_src),
        "best_streaming_backend_match_vs_cpu": _backend_match_generic(best_stream_src),
    }


def build_decision_summary_markdown(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> str:
    ov = decision_overview(report, inter)
    rows = decision_provider_rows(report, inter)
    cand_rows = decision_candidate_rows(report, inter, limit=5)
    lines: List[str] = []
    lines.append(f"# Entscheidungs-Zusammenfassung: {report.source.display_name}\n\n")
    tool_meta = report.summary.get("_tool_meta") if isinstance(report.summary.get("_tool_meta"), dict) else {}
    if tool_meta:
        lines.append(f"Tool: GUI `{tool_meta.get('gui', '-')}`, Core `{tool_meta.get('core', '-')}` · ")
    lines.append(
        f"Cases: **{ov.get('case_count', '-')}** · Provider-Gruppen: **{ov.get('provider_count', '-')}** · "
        f"Passed rows: **{ov.get('pass_rows', 0)}/{ov.get('total_rows', 0)}**\n\n"
    )

    quality_label = str(ov.get("quality_label") or "Qualität")
    backend_label = str(ov.get("backend_metric_label") or "Drift")
    task = str(ov.get("task") or "detection")
    gain = _as_float(ov.get("best_streaming_gain_pct"))
    if ov.get("best_streaming_setup") and gain is not None and gain > 5.0:
        lines.append(
            f"**Kurzurteil:** Der Split lohnt sich vor allem als **Streaming-Pipeline**, nicht als sequenzieller Einzelbild-Lauf. "
            f"Bester Kandidat ist **{ov.get('best_streaming_setup')} {_fmt_boundary(ov.get('best_streaming_boundary'))}** mit "
            f"**{_fmt(ov.get('best_streaming_fps'), 1)} FPS**, also **{_fmt(gain, 1)} %** gegenüber der besten Full-Baseline "
            f"(**{ov.get('best_full_setup')}**, {_fmt(ov.get('best_full_fps'), 1)} FPS).\n\n"
        )
    elif ov.get("best_streaming_setup"):
        lines.append(
            f"**Kurzurteil:** Es gibt keinen klaren Streaming-Durchsatzgewinn. Bester Streaming-Kandidat ist "
            f"**{ov.get('best_streaming_setup')} {_fmt_boundary(ov.get('best_streaming_boundary'))}** mit **{_fmt(ov.get('best_streaming_fps'), 1)} FPS**.\n\n"
        )
    else:
        lines.append(
            "**Kurzurteil:** Keine heterogene Streaming-Pipeline im Ergebnis erkannt. "
            "Bewertung basiert auf Full- und sequenziellen Split-Latenzen.\n\n"
        )

    if ov.get("best_full_setup") or ov.get("best_full_quality_setup"):
        lines.append(
            f"Beste Full-Latenz: **{ov.get('best_full_setup') or '-'}** mit **{_fmt(ov.get('best_full_latency_ms'), 2)} ms** "
            f"({_fmt(ov.get('best_full_fps'), 1)} FPS). "
            f"Beste Full-{quality_label}: **{ov.get('best_full_quality_setup') or '-'}** mit **{_fmt(ov.get('best_full_quality_value'), 3)}**.\n\n"
        )

    lines.append("## Kerndaten\n\n")
    lines.append(f"| Rolle | Setup | b | FPS | Latenz ms | {quality_label} | {backend_label} | Hailo ctx | Empfehlung |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for row in rows[:8]:
        role = {"full": "Full", "sequential_split": "Seq. Split", "streaming_split": "Streaming"}.get(
            str(row.get("kind")), str(row.get("kind"))
        )
        boundary = _fmt_boundary(row.get("boundary"))
        lines.append(
            f"| {role} | {row.get('setup', '-')} | {boundary} | {_fmt(row.get('streaming_fps'), 1)} | {_fmt(row.get('latency_ms'), 2)} | "
            f"{_fmt(row.get('quality'), 3)} | {_fmt(row.get('backend_match_vs_cpu'), 3)} | {row.get('hailo_context_label') or '-'} | "
            f"{row.get('recommendation', '-')} |\n"
        )

    if cand_rows:
        lines.append("\n## Top Streaming-Kandidaten\n\n")
        lines.append(f"| Rang | Pipeline | b | FPS | Latenz ms | Gain vs best full | {quality_label} | {backend_label} | Hailo ctx |\n")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in cand_rows:
            lines.append(
                f"| {row.get('rank', '-')} | {row.get('pipeline', '-')} | {_fmt_boundary(row.get('boundary'))} | {_fmt(row.get('streaming_fps'), 1)} | "
                f"{_fmt(row.get('latency_ms'), 2)} | {_fmt(row.get('gain_vs_best_full_pct'), 1)} % | {_fmt(row.get('quality'), 3)} | "
                f"{_fmt(row.get('backend_match_vs_cpu'), 3)} | {row.get('hailo_context_label') or '-'} |\n"
            )

    lines.append("\n## Einordnung\n")
    lines.append("- `Full` bleibt die Referenz für niedrigste Einzelbild-Latenz.\n")
    lines.append("- `Streaming` bewertet steady-state FPS mit paralleler Stage-Ausführung; höhere End-to-End-Latenz ist hier normal.\n")
    if task == "classification":
        lines.append(
            "- `Top-1` ist der leichte Klassifikations-Zusatzcheck auf dem konfigurierten Label-Datensatz; "
            "ohne Label-Datensatz bleibt nur die Split-Fidelity gegen full auf Logit-/Top-k-Ebene.\n"
        )
    else:
        lines.append(
            "- `AP50` ist der optionale Mini-COCO-AP50-Zusatzcheck; `Drift/Match` zeigt den Backend-Drift gegen CPU-full, "
            "nicht automatisch Split-Schaden.\n"
        )
    lines.append(
        f"- Hailo Multi-Context oder `single→multi` wird nicht verworfen. Es ist ein Risiko-/Stabilitätslabel; "
        f"die Empfehlung basiert weiter auf gemessenen FPS, {quality_label} und {backend_label}.\n"
    )
    return "".join(lines)


def build_decision_fps_latency_figure(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> Figure:
    rows = decision_provider_rows(report, inter)
    plot_rows = [r for r in rows if _as_float(r.get("streaming_fps")) is not None and _as_float(r.get("latency_ms")) is not None]
    if not plot_rows:
        return _placeholder_figure("Decision: FPS vs Latenz", "Keine FPS/Latenz-Daten verfügbar.")
    fig = Figure(figsize=(7.4, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    xs = [float(_as_float(r.get("latency_ms")) or 0.0) for r in plot_rows]
    ys = [float(_as_float(r.get("streaming_fps")) or 0.0) for r in plot_rows]
    ax.scatter(xs, ys)
    for r, x, y in zip(plot_rows, xs, ys):
        kind = str(r.get("kind") or "")
        label = str(r.get("setup") or "-")
        if r.get("boundary") is not None:
            label = f"{label} {_fmt_boundary(r.get('boundary'))}"
        if kind == "sequential_split":
            # keep the chart focused on decision-grade points
            continue
        ax.annotate(label.replace("_auto", ""), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("End-to-end latency [ms]")
    ax.set_ylabel("Steady-state FPS")
    ax.set_title("Decision: Streaming-FPS vs Latenz")
    fig.tight_layout()
    return fig


def build_decision_ap50_figure(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> Figure:
    all_rows = decision_provider_rows(report, inter)
    task = decision_overview(report, inter).get("task") if all_rows else "detection"
    quality_label = _quality_label(str(task or "detection"))
    rows = [r for r in all_rows if _as_float(r.get("quality")) is not None]
    # Keep full baselines and streaming splits; sequential splits add little signal here.
    rows = [r for r in rows if str(r.get("kind")) in {"full", "streaming_split"}]
    if not rows:
        return _placeholder_figure(f"Decision: {quality_label}", f"Keine {quality_label}-Daten verfügbar.")
    rows = sorted(rows, key=lambda r: (str(r.get("kind")) != "streaming_split", -float(_as_float(r.get("quality")) or 0.0)))[:10]
    fig = Figure(figsize=(7.4, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    labels = []
    vals = []
    for r in rows:
        label = str(r.get("setup") or "-").replace("_auto", "")
        if r.get("boundary") is not None:
            label = f"{label}\n{_fmt_boundary(r.get('boundary'))}"
        labels.append(label)
        vals.append(float(_as_float(r.get("quality")) or 0.0))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(quality_label)
    ax.set_title(f"Decision: {quality_label} nach Setup")
    fig.tight_layout()
    return fig


def build_decision_backend_match_figure(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> Figure:
    all_rows = decision_provider_rows(report, inter)
    task = decision_overview(report, inter).get("task") if all_rows else "detection"
    backend_label = _backend_metric_label(str(task or "detection"))
    rows = [r for r in all_rows if _as_float(r.get("backend_match_vs_cpu")) is not None]
    rows = [r for r in rows if str(r.get("kind")) in {"full", "streaming_split"}]
    if not rows:
        return _placeholder_figure("Decision: Backend-Drift", f"Keine {backend_label}-Daten verfügbar.")
    rows = sorted(rows, key=lambda r: (str(r.get("kind")) != "streaming_split", -float(_as_float(r.get("backend_match_vs_cpu")) or 0.0)))[:10]
    fig = Figure(figsize=(7.4, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    labels = []
    vals = []
    for r in rows:
        label = str(r.get("setup") or "-").replace("_auto", "")
        if r.get("boundary") is not None:
            label = f"{label}\n{_fmt_boundary(r.get('boundary'))}"
        labels.append(label)
        vals.append(float(_as_float(r.get("backend_match_vs_cpu")) or 0.0))
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(backend_label)
    ax.set_title("Decision: Backend-Drift")
    fig.tight_layout()
    return fig


def build_decision_summary_figures(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport) -> Dict[str, Figure]:
    return {
        "decision_fps_latency": build_decision_fps_latency_figure(report, inter),
        "decision_ap50": build_decision_ap50_figure(report, inter),
        "decision_backend_match": build_decision_backend_match_figure(report, inter),
    }


def export_decision_summary(report: BenchmarkAnalysisReport, inter: InterleavingAnalysisReport, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    summary_md = output_dir / "benchmark_decision_summary.md"
    summary_md.write_text(build_decision_summary_markdown(report, inter), encoding="utf-8")
    paths["decision_summary_md"] = summary_md

    provider_csv = output_dir / "benchmark_decision_provider_table.csv"
    _write_csv(provider_csv, decision_provider_rows(report, inter))
    paths["decision_provider_csv"] = provider_csv

    candidate_csv = output_dir / "benchmark_decision_candidate_table.csv"
    _write_csv(candidate_csv, decision_candidate_rows(report, inter, limit=20))
    paths["decision_candidate_csv"] = candidate_csv

    for name, fig in build_decision_summary_figures(report, inter).items():
        # Keep this export intentionally lightweight. The older technical export
        # still writes PNG/PDF/SVG for paper figures; the decision summary is meant
        # to be fast and immediately useful after a run.
        out_path = output_dir / f"benchmark_{name}.png"
        fig.savefig(out_path, bbox_inches="tight")
        paths[f"plot_{name}_png"] = out_path
        try:
            fig.clear()
        except Exception:
            pass
    return paths
