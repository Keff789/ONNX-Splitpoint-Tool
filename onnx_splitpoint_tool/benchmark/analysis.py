"""Benchmark-results analysis helpers.

This module powers the GUI's *Benchmark Analysis* tab. It loads benchmark-set
metadata plus measured CSV exports, compares predicted heuristics to actual
results and produces compact summaries/plots that are useful directly in the
GUI or for report export.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from matplotlib.figure import Figure

from .schema import migrate_benchmark_set_payload


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, int):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _as_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if math.isfinite(v) else None
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(float(s))
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
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s



def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    xm = sum(x) / len(x)
    ym = sum(y) / len(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - xm) ** 2 for a in x))
    deny = math.sqrt(sum((b - ym) ** 2 for b in y))
    den = denx * deny
    return (num / den) if den > 0 else float("nan")



def _spearman(x: Iterable[Optional[float]], y: Iterable[Optional[float]]) -> Optional[float]:
    xv = []
    yv = []
    for a, b in zip(list(x), list(y)):
        if a is None or b is None:
            continue
        if not math.isfinite(float(a)) or not math.isfinite(float(b)):
            continue
        xv.append(float(a))
        yv.append(float(b))
    if len(xv) < 2:
        return None

    def _ranks(vals: List[float]) -> List[float]:
        n = len(vals)
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _ranks(xv)
    ry = _ranks(yv)
    out = _pearson(rx, ry)
    return out if math.isfinite(out) else None



def _fmt(x: Optional[float], nd: int = 2) -> str:
    if x is None:
        return "-"
    if not math.isfinite(float(x)):
        return "-"
    return f"{float(x):.{nd}f}"


@dataclass
class BenchmarkAnalysisSource:
    source_path: Path
    results_root: Path
    display_name: str
    extracted: bool = False
    extraction_root: Optional[Path] = None


@dataclass
class ProviderSummary:
    tag: str
    row_count: int
    full_baseline_ms: Optional[float]
    best_boundary: Optional[int]
    best_composed_ms: Optional[float]
    delta_vs_full_ms: Optional[float]
    delta_vs_full_pct: Optional[float]
    speedup_full_over_best: Optional[float]
    score_spearman: Optional[float]
    latency_spearman: Optional[float]
    top5_overlap: Optional[int]
    best_actual_predicted_rank: Optional[int]
    best_predicted_actual_rank: Optional[int]


@dataclass
class CandidateSummary:
    boundary: int
    avg_rank: float
    top3_hits: int
    providers_present: int
    best_provider: str
    best_rank: int
    score_pred: Optional[float]
    cut_mib: Optional[float]
    hailo_compile_risk_score: Optional[float]
    hailo_single_context_probability: Optional[float]


@dataclass
class HailoContextSummary:
    boundary: int
    hw_arch: str
    part1_context_count: Optional[int]
    part1_context_mode: str
    part2_context_count: Optional[int]
    part2_context_mode: str
    part1_partition_time_s: Optional[float]
    part1_allocation_time_s: Optional[float]
    part1_compilation_time_s: Optional[float]
    part1_elapsed_s: Optional[float]
    part2_partition_time_s: Optional[float]
    part2_allocation_time_s: Optional[float]
    part2_compilation_time_s: Optional[float]
    part2_elapsed_s: Optional[float]
    part2_calib_source: Optional[str]
    part2_single_context: bool
    both_parts_single_context: bool
    score_pred: Optional[float]
    cut_mib: Optional[float]
    direct_hailo_composed_ms: Optional[float]
    direct_hailo_provider: Optional[str]


@dataclass
class BenchmarkAnalysisReport:
    source: BenchmarkAnalysisSource
    benchmark_set_path: Path
    summary: Dict[str, Any]
    providers: List[ProviderSummary]
    provider_rows: Dict[str, List[Dict[str, Any]]]
    candidate_summaries: List[CandidateSummary]
    hailo_context_summaries: List[HailoContextSummary]
    provider_tags: List[str]
    bench_cases: List[Dict[str, Any]]
    summary_markdown: str
    issues: List[str] = field(default_factory=list)


@dataclass
class ComparisonProviderSummary:
    provider: str
    left_rows: int
    right_rows: int
    left_full_baseline_ms: Optional[float]
    right_full_baseline_ms: Optional[float]
    full_delta_ms: Optional[float]
    full_delta_pct: Optional[float]
    left_best_boundary: Optional[int]
    right_best_boundary: Optional[int]
    left_best_composed_ms: Optional[float]
    right_best_composed_ms: Optional[float]
    best_delta_ms: Optional[float]
    best_delta_pct: Optional[float]
    left_score_spearman: Optional[float]
    right_score_spearman: Optional[float]
    score_spearman_delta: Optional[float]
    left_latency_spearman: Optional[float]
    right_latency_spearman: Optional[float]
    latency_spearman_delta: Optional[float]
    best_boundary_changed: bool


@dataclass
class ComparisonCandidateSummary:
    boundary: int
    left_avg_rank: Optional[float]
    right_avg_rank: Optional[float]
    avg_rank_delta: Optional[float]
    left_top3_hits: int
    right_top3_hits: int
    top3_delta: int
    left_best_provider: Optional[str]
    right_best_provider: Optional[str]
    left_best_rank: Optional[int]
    right_best_rank: Optional[int]
    left_score_pred: Optional[float]
    right_score_pred: Optional[float]
    left_cut_mib: Optional[float]
    right_cut_mib: Optional[float]


@dataclass
class ComparisonHailoContextSummary:
    boundary: int
    hw_arch: str
    left_part2_context_count: Optional[int]
    right_part2_context_count: Optional[int]
    context_delta: Optional[float]
    left_part2_context_mode: str
    right_part2_context_mode: str
    left_part2_single_context: bool
    right_part2_single_context: bool
    single_context_changed: bool
    left_direct_hailo_composed_ms: Optional[float]
    right_direct_hailo_composed_ms: Optional[float]
    latency_delta_ms: Optional[float]
    left_score_pred: Optional[float]
    right_score_pred: Optional[float]
    left_cut_mib: Optional[float]
    right_cut_mib: Optional[float]


@dataclass
class BenchmarkComparisonReport:
    left: BenchmarkAnalysisReport
    right: BenchmarkAnalysisReport
    provider_summaries: List[ComparisonProviderSummary]
    candidate_summaries: List[ComparisonCandidateSummary]
    hailo_context_summaries: List[ComparisonHailoContextSummary]
    summary_markdown: str
    issues: List[str] = field(default_factory=list)


SUPPORTED_ARCHIVE_SUFFIXES = (
    ".tar.gz",
    ".tgz",
    ".tar",
    ".zip",
)



def _strip_archive_suffix(name: str) -> str:
    lower = name.lower()
    for sfx in sorted(SUPPORTED_ARCHIVE_SUFFIXES, key=len, reverse=True):
        if lower.endswith(sfx):
            return name[: -len(sfx)]
    return Path(name).stem



def _display_name_from_source(src: Path, results_root: Path) -> str:
    if src.is_file() and _is_supported_archive(src):
        base = _strip_archive_suffix(src.name).strip()
        if base:
            return base
    if results_root.name == "results" and results_root.parent.name:
        return results_root.parent.name
    if results_root.name:
        return results_root.name
    return src.stem or src.name



def _is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(sfx) for sfx in SUPPORTED_ARCHIVE_SUFFIXES)



def _stable_extract_dir(cache_base: Path, source_file: Path) -> Path:
    st = source_file.stat()
    h = hashlib.sha1()
    h.update(str(source_file.resolve()).encode("utf-8", "ignore"))
    h.update(str(st.st_size).encode("utf-8"))
    h.update(str(getattr(st, "st_mtime_ns", int(st.st_mtime))).encode("utf-8"))
    slug = source_file.name.replace(".", "_")
    return cache_base / f"{slug}_{h.hexdigest()[:12]}"



def _extract_archive(source_file: Path, cache_base: Path) -> Path:
    dst = _stable_extract_dir(cache_base, source_file)
    sentinel = dst / ".extract_ok"
    if sentinel.exists():
        return dst
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)
    name = source_file.name.lower()
    if name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(source_file, "r:*") as tf:
            tf.extractall(dst)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(source_file, "r") as zf:
            zf.extractall(dst)
    else:
        raise ValueError(f"Unsupported archive format: {source_file}")
    sentinel.write_text("ok", encoding="utf-8")
    return dst



def _find_results_root(base: Path) -> Optional[Path]:
    candidates: List[Path] = []
    search_roots = [base]
    if base.is_dir():
        try:
            for p in base.rglob("benchmark_set.json"):
                search_roots.append(p.parent)
        except Exception:
            pass
    seen: set[str] = set()
    ordered: List[Path] = []
    for root in search_roots:
        root = Path(root)
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(root)
    for root in ordered:
        for cand in (root, root / "results"):
            if not cand.exists() or not cand.is_dir():
                continue
            if (cand / "benchmark_set.json").exists() and list(cand.glob("benchmark_results_*.csv")):
                candidates.append(cand)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]



def discover_benchmark_analysis_source(source_path: str | Path, cache_base: Path) -> BenchmarkAnalysisSource:
    src = Path(str(source_path)).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Benchmark analysis source not found: {src}")

    extracted = False
    extraction_root: Optional[Path] = None
    search_base = src
    if src.is_file() and _is_supported_archive(src):
        extraction_root = _extract_archive(src, cache_base)
        search_base = extraction_root
        extracted = True
    elif src.is_file():
        search_base = src.parent

    results_root = _find_results_root(search_base)
    if results_root is None:
        raise FileNotFoundError(
            "Could not locate a benchmark results directory containing benchmark_set.json and benchmark_results_*.csv"
        )
    display_name = _display_name_from_source(src, results_root)
    return BenchmarkAnalysisSource(
        source_path=src,
        results_root=results_root,
        display_name=display_name,
        extracted=extracted,
        extraction_root=extraction_root,
    )



def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))



def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows



def _provider_tag_from_csv(path: Path) -> str:
    name = path.name
    if name.startswith("benchmark_results_") and name.endswith(".csv"):
        return name[len("benchmark_results_") : -len(".csv")]
    return path.stem



def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    n = len(vals)
    m = n // 2
    if n % 2 == 1:
        return vals[m]
    return (vals[m - 1] + vals[m]) / 2.0



def _actual_rank_map(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    ranked = sorted(
        [r for r in rows if _as_int(r.get("boundary")) is not None and _as_float(r.get("composed_mean_ms")) is not None],
        key=lambda r: float(_as_float(r.get("composed_mean_ms")) or float("inf")),
    )
    out: Dict[int, int] = {}
    for i, row in enumerate(ranked, start=1):
        b = int(_as_int(row.get("boundary")) or 0)
        out[b] = i
    return out



def _predicted_score_rank_map(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    ranked = sorted(
        [r for r in rows if _as_int(r.get("boundary")) is not None and _as_float(r.get("score_pred") or r.get("score")) is not None],
        key=lambda r: -float(_as_float(r.get("score_pred") or r.get("score")) or -float("inf")),
    )
    out: Dict[int, int] = {}
    for i, row in enumerate(ranked, start=1):
        b = int(_as_int(row.get("boundary")) or 0)
        out[b] = i
    return out



def _latency_rank_map(rows: List[Dict[str, Any]]) -> Dict[int, int]:
    ranked = sorted(
        [r for r in rows if _as_int(r.get("boundary")) is not None and _as_float(r.get("latency_total_ms")) is not None],
        key=lambda r: float(_as_float(r.get("latency_total_ms")) or float("inf")),
    )
    out: Dict[int, int] = {}
    for i, row in enumerate(ranked, start=1):
        b = int(_as_int(row.get("boundary")) or 0)
        out[b] = i
    return out



def _provider_summary(tag: str, rows: List[Dict[str, Any]]) -> ProviderSummary:
    full_baseline = _median(_as_float(r.get("full_mean_ms")) for r in rows)
    valid_rows = [r for r in rows if str(r.get("final_pass_all", "")).strip().lower() in {"true", "1", "yes"}]
    if not valid_rows:
        valid_rows = rows
    best_row = None
    best_obj = float("inf")
    for row in valid_rows:
        obj = _as_float(row.get("composed_mean_ms"))
        if obj is None:
            continue
        if obj < best_obj:
            best_obj = obj
            best_row = row
    best_boundary = _as_int(best_row.get("boundary")) if best_row else None
    best_comp = _as_float(best_row.get("composed_mean_ms")) if best_row else None
    delta_ms = (best_comp - full_baseline) if (best_comp is not None and full_baseline is not None) else None
    delta_pct = ((delta_ms / full_baseline) * 100.0) if (delta_ms is not None and full_baseline and full_baseline > 0) else None
    speedup = ((full_baseline / best_comp) if (best_comp and full_baseline and best_comp > 0) else None)

    score_spearman = _spearman(
        [_as_float(r.get("score_pred") or r.get("score")) for r in valid_rows],
        [(-float(v) if v is not None else None) for v in [_as_float(r.get("composed_mean_ms")) for r in valid_rows]],
    )
    latency_spearman = _spearman(
        [_as_float(r.get("latency_total_ms")) for r in valid_rows],
        [_as_float(r.get("composed_mean_ms")) for r in valid_rows],
    )

    actual_rank = _actual_rank_map(valid_rows)
    pred_rank = _predicted_score_rank_map(valid_rows)
    actual_top5 = {b for b, rk in actual_rank.items() if rk <= 5}
    pred_top5 = {b for b, rk in pred_rank.items() if rk <= 5}
    overlap = len(actual_top5 & pred_top5) if actual_top5 or pred_top5 else None

    best_actual_pred_rank = pred_rank.get(int(best_boundary)) if best_boundary is not None else None
    if pred_rank:
        pred_best_boundary = min(pred_rank.items(), key=lambda kv: kv[1])[0]
        pred_best_actual_rank = actual_rank.get(pred_best_boundary)
    else:
        pred_best_actual_rank = None

    return ProviderSummary(
        tag=tag,
        row_count=len(rows),
        full_baseline_ms=full_baseline,
        best_boundary=best_boundary,
        best_composed_ms=best_comp,
        delta_vs_full_ms=delta_ms,
        delta_vs_full_pct=delta_pct,
        speedup_full_over_best=speedup,
        score_spearman=score_spearman,
        latency_spearman=latency_spearman,
        top5_overlap=overlap,
        best_actual_predicted_rank=best_actual_pred_rank,
        best_predicted_actual_rank=pred_best_actual_rank,
    )



def _candidate_summaries(provider_rows: Dict[str, List[Dict[str, Any]]]) -> List[CandidateSummary]:
    rank_maps = {tag: _actual_rank_map(rows) for tag, rows in provider_rows.items()}
    predicted_lookup: Dict[int, Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = {}
    for rows in provider_rows.values():
        for row in rows:
            b = _as_int(row.get("boundary"))
            if b is None or b in predicted_lookup:
                continue
            predicted_lookup[b] = (
                _as_float(row.get("score_pred") or row.get("score")),
                _as_float(row.get("cut_mib")),
                _as_float(row.get("hailo_compile_risk_score")),
                _as_float(row.get("hailo_single_context_probability")),
            )

    boundaries = sorted({b for rk in rank_maps.values() for b in rk})
    out: List[CandidateSummary] = []
    for b in boundaries:
        ranks: List[Tuple[str, int]] = []
        for tag, rk in rank_maps.items():
            if b in rk:
                ranks.append((tag, rk[b]))
        if not ranks:
            continue
        avg_rank = sum(r for _, r in ranks) / len(ranks)
        best_provider, best_rank = min(ranks, key=lambda kv: kv[1])
        top3_hits = sum(1 for _, r in ranks if r <= 3)
        score_pred, cut_mib, hailo_compile_risk_score, hailo_single_context_probability = predicted_lookup.get(b, (None, None, None, None))
        out.append(
            CandidateSummary(
                boundary=int(b),
                avg_rank=float(avg_rank),
                top3_hits=int(top3_hits),
                providers_present=len(ranks),
                best_provider=best_provider,
                best_rank=int(best_rank),
                score_pred=score_pred,
                cut_mib=cut_mib,
                hailo_compile_risk_score=hailo_compile_risk_score,
                hailo_single_context_probability=hailo_single_context_probability,
            )
        )
    out.sort(key=lambda c: (c.avg_rank, -c.top3_hits, c.boundary))
    return out



def _row_lookup_by_boundary(provider_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
    lookup: Dict[int, Dict[str, Any]] = {}
    for rows in provider_rows.values():
        for row in rows:
            b = _as_int(row.get("boundary"))
            if b is None or b in lookup:
                continue
            lookup[b] = row
    return lookup



def _hailo_part2_fallback_meta_map(bench_cases: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for case in list(bench_cases or []):
        if not isinstance(case, dict):
            continue
        b = _as_int(case.get("boundary"))
        if b is None:
            continue
        outputs: List[str] = []
        for raw in list(case.get("hailo_part2_effective_outputs") or []):
            name = str(raw or "").strip()
            if name:
                outputs.append(name)
        strategy = str(case.get("hailo_part2_output_strategy") or "original").strip() or "original"
        used = bool(strategy != "original" or outputs)
        strategy_label = "orig"
        if used:
            if strategy == "hailo_parser_suggested_end_nodes":
                strategy_label = "suggested"
            else:
                strategy_label = str(strategy or "alt")
        outputs_text = ", ".join(outputs[:3])
        if len(outputs) > 3:
            outputs_text = (outputs_text + ", …") if outputs_text else "…"
        out[int(b)] = {
            "hailo_part2_fallback_used": used,
            "hailo_part2_marker": "↩" if used else "-",
            "hailo_part2_output_strategy": strategy,
            "hailo_part2_strategy_label": strategy_label,
            "hailo_part2_effective_outputs": outputs,
            "hailo_part2_effective_outputs_text": outputs_text,
        }
    return out



def _direct_hailo_provider_tag(provider_rows: Dict[str, List[Dict[str, Any]]], hw_arch: str) -> Optional[str]:
    slug = _slug(hw_arch)
    if not slug:
        return None
    direct_tags = [tag for tag in provider_rows.keys() if "_to_" not in str(tag)]
    if hw_arch in provider_rows and "_to_" not in hw_arch:
        return hw_arch
    for tag in direct_tags:
        t = _slug(tag)
        if t == slug:
            return tag
    if slug.startswith("hailo10"):
        for tag in direct_tags:
            if _slug(tag).startswith("hailo10"):
                return tag
    if slug.startswith("hailo8"):
        for tag in direct_tags:
            if _slug(tag).startswith("hailo8"):
                return tag
    return None



def _context_mode_from_stage_meta(meta: Dict[str, Any]) -> str:
    mode = str(meta.get("context_mode") or "").strip()
    if mode:
        return mode
    if bool(meta.get("single_context_failed")) and bool(meta.get("multi_context_used")):
        return "single_context_failed_to_multi"
    count = _as_int(meta.get("context_count"))
    if count == 1:
        return "single_context_used"
    if count is not None and count > 1:
        return "multi_context_used"
    if _as_bool(meta.get("skipped")) is True:
        return "skipped"
    if _as_bool(meta.get("ok")) is False:
        return "failed"
    return "unknown"



def _normalize_stage_meta(meta: Any) -> Dict[str, Any]:
    raw = meta if isinstance(meta, dict) else {}
    out: Dict[str, Any] = {}
    for key in (
        "ok",
        "skipped",
        "timed_out",
        "error",
        "failure_kind",
        "unsupported_reason",
        "elapsed_s",
        "context_count",
        "partition_iterations",
        "partition_time_s",
        "allocation_time_s",
        "compilation_time_s",
        "calib_source",
        "single_context_failed",
        "single_context_used",
        "multi_context_used",
        "mapping_failed",
        "watchdog_expired",
        "context_mode",
        "hef_path",
    ):
        if key in raw:
            out[key] = raw.get(key)
    out["context_mode"] = _context_mode_from_stage_meta(out)
    return out



def _extract_hailo_compile_from_case(case: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    payload = case.get("hailo_compile")
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return out
    for hw_arch, hw_meta in payload.items():
        if not isinstance(hw_meta, dict):
            continue
        arch_entry: Dict[str, Dict[str, Any]] = {}
        for stage in ("part1", "part2"):
            stage_meta = hw_meta.get(stage)
            if isinstance(stage_meta, dict):
                arch_entry[stage] = _normalize_stage_meta(stage_meta)
        if arch_entry:
            out[str(hw_arch)] = arch_entry
    return out



def _extract_hailo_compile_from_rows(provider_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Dict[str, Dict[str, Any]]]]:
    out: Dict[int, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    pat = re.compile(r"^hailo_(?P<arch>[a-z0-9_]+)_(?P<stage>part1|part2)_(?P<field>.+)$")
    for rows in provider_rows.values():
        for row in rows:
            boundary = _as_int(row.get("boundary"))
            if boundary is None:
                continue
            row_out = out.setdefault(int(boundary), {})
            for key, value in row.items():
                m = pat.match(str(key))
                if m is None:
                    continue
                arch = str(m.group("arch"))
                stage = str(m.group("stage"))
                field = str(m.group("field"))
                arch_entry = row_out.setdefault(arch, {})
                stage_entry = arch_entry.setdefault(stage, {})
                stage_entry[field] = value
    # normalize inferred payloads
    for boundary, hw_map in list(out.items()):
        for hw_arch, hw_meta in list(hw_map.items()):
            for stage, stage_meta in list(hw_meta.items()):
                hw_meta[stage] = _normalize_stage_meta(stage_meta)
    return out



def _extract_hailo_context_summaries(bench_cases: List[Dict[str, Any]], provider_rows: Dict[str, List[Dict[str, Any]]]) -> List[HailoContextSummary]:
    row_lookup = _row_lookup_by_boundary(provider_rows)
    inferred_from_rows = _extract_hailo_compile_from_rows(provider_rows)

    direct_hailo_latency: Dict[str, Dict[int, float]] = {}
    for hw_arch in {hw for case in bench_cases for hw in _extract_hailo_compile_from_case(case).keys()} | {hw for meta in inferred_from_rows.values() for hw in meta.keys()}:
        tag = _direct_hailo_provider_tag(provider_rows, hw_arch)
        if tag and tag in provider_rows:
            direct_hailo_latency[hw_arch] = {
                int(_as_int(row.get("boundary")) or 0): float(_as_float(row.get("composed_mean_ms")) or float("nan"))
                for row in provider_rows[tag]
                if _as_int(row.get("boundary")) is not None and _as_float(row.get("composed_mean_ms")) is not None
            }

    out: List[HailoContextSummary] = []
    seen: set[Tuple[int, str]] = set()
    candidate_cases = list(bench_cases)
    if not candidate_cases:
        # fall back to synthetic cases based on rows when benchmark_set has no case entries
        for b in sorted(inferred_from_rows):
            row = row_lookup.get(int(b), {})
            candidate_cases.append({
                "boundary": int(b),
                "predicted": {
                    "score": row.get("score_pred") or row.get("score"),
                    "cut_mib": row.get("cut_mib"),
                },
            })

    for case in candidate_cases:
        boundary = _as_int(case.get("boundary"))
        if boundary is None:
            continue
        boundary = int(boundary)
        pred = case.get("predicted") if isinstance(case.get("predicted"), dict) else {}
        score_pred = _as_float(pred.get("score") or pred.get("score_pred"))
        cut_mib = _as_float(pred.get("cut_mib"))
        row = row_lookup.get(boundary, {})
        if score_pred is None:
            score_pred = _as_float(row.get("score_pred") or row.get("score"))
        if cut_mib is None:
            cut_mib = _as_float(row.get("cut_mib"))

        compile_meta = _extract_hailo_compile_from_case(case)
        if not compile_meta:
            compile_meta = inferred_from_rows.get(boundary, {})
        for hw_arch, hw_meta in compile_meta.items():
            key = (boundary, str(hw_arch))
            if key in seen:
                continue
            seen.add(key)
            part1 = _normalize_stage_meta(hw_meta.get("part1"))
            part2 = _normalize_stage_meta(hw_meta.get("part2"))
            part1_mode = _context_mode_from_stage_meta(part1)
            part2_mode = _context_mode_from_stage_meta(part2)
            part1_count = _as_int(part1.get("context_count"))
            part2_count = _as_int(part2.get("context_count"))
            part2_single = bool(part2_mode == "single_context_used" or part2_count == 1)
            part1_single = bool(part1_mode == "single_context_used" or part1_count == 1)
            direct_tag = _direct_hailo_provider_tag(provider_rows, hw_arch)
            latency_map = direct_hailo_latency.get(str(hw_arch), {})
            direct_latency = latency_map.get(boundary)
            if direct_latency is not None and not math.isfinite(float(direct_latency)):
                direct_latency = None
            out.append(
                HailoContextSummary(
                    boundary=boundary,
                    hw_arch=str(hw_arch),
                    part1_context_count=part1_count,
                    part1_context_mode=part1_mode,
                    part2_context_count=part2_count,
                    part2_context_mode=part2_mode,
                    part1_partition_time_s=_as_float(part1.get("partition_time_s")),
                    part1_allocation_time_s=_as_float(part1.get("allocation_time_s")),
                    part1_compilation_time_s=_as_float(part1.get("compilation_time_s")),
                    part1_elapsed_s=_as_float(part1.get("elapsed_s")),
                    part2_partition_time_s=_as_float(part2.get("partition_time_s")),
                    part2_allocation_time_s=_as_float(part2.get("allocation_time_s")),
                    part2_compilation_time_s=_as_float(part2.get("compilation_time_s")),
                    part2_elapsed_s=_as_float(part2.get("elapsed_s")),
                    part2_calib_source=(str(part2.get("calib_source")) if part2.get("calib_source") not in (None, "") else None),
                    part2_single_context=part2_single,
                    both_parts_single_context=bool(part1_single and part2_single),
                    score_pred=score_pred,
                    cut_mib=cut_mib,
                    direct_hailo_composed_ms=direct_latency,
                    direct_hailo_provider=direct_tag,
                )
            )
    out.sort(key=lambda r: (_slug(r.hw_arch), r.boundary))
    return out



def _hailo_target_stats(report: BenchmarkAnalysisReport, hw_arch: str) -> Dict[str, Any]:
    rows = [row for row in report.hailo_context_summaries if _slug(row.hw_arch) == _slug(hw_arch)]
    total = len(rows)
    single = sum(1 for row in rows if row.part2_single_context)
    both_single = sum(1 for row in rows if row.both_parts_single_context)
    fallback = sum(1 for row in rows if row.part2_context_mode == "single_context_failed_to_multi")
    multi = sum(1 for row in rows if row.part2_context_mode == "multi_context_used")
    best_single = None
    candidates = [row for row in rows if row.part2_single_context and row.direct_hailo_composed_ms is not None]
    if candidates:
        best_single = min(candidates, key=lambda row: float(row.direct_hailo_composed_ms or float("inf")))
    return {
        "rows": rows,
        "total": total,
        "part2_single": single,
        "both_single": both_single,
        "fallback_multi": fallback,
        "multi": multi,
        "best_single": best_single,
    }



def _preferred_hailo_target(report: BenchmarkAnalysisReport) -> Optional[str]:
    archs = sorted({row.hw_arch for row in report.hailo_context_summaries}, key=lambda s: (_slug(s) != "hailo8", _slug(s)))
    return archs[0] if archs else None



def build_summary_markdown(report: BenchmarkAnalysisReport) -> str:
    summary = report.summary
    lines: List[str] = []
    lines.append(f"# Benchmark-Analyse: {report.source.display_name}\n")
    try:
        schema_v = int(report.summary.get("schema_version") or 0)
    except Exception:
        schema_v = 0
    tool_meta = report.summary.get("_tool_meta") if isinstance(report.summary.get("_tool_meta"), dict) else {}
    if schema_v or tool_meta:
        gui_v = str(tool_meta.get("gui") or "-").strip()
        core_v = str(tool_meta.get("core") or "-").strip()
        lines.append(f"- Artefakt-Schema: **v{schema_v or '-'}** · Tool: GUI **{gui_v}**, Core **{core_v}**.\n")
    lines.append(f"- Quelle: `{report.source.source_path}`\n")
    lines.append(f"- Ergebnisse: `{report.source.results_root}`\n")
    if report.source.extracted and report.source.extraction_root is not None:
        lines.append(f"- Archiv entpackt nach: `{report.source.extraction_root}`\n")
    lines.append("\n## Suite-Überblick\n")
    lines.append(f"- Akzeptierte Cases: **{summary.get('generated_cases', summary.get('case_count', '-'))}**")
    if summary.get("requested_cases") is not None:
        lines.append(f" (angefordert: {summary.get('requested_cases')})")
    lines.append("\n")
    if summary.get("preferred_shortlist_cases") is not None:
        lines.append(f"- Preferred shortlist: **{summary.get('preferred_shortlist_cases')}**\n")
    if summary.get("preferred_shortlist_after_prefilter") is not None:
        lines.append(f"- Preferred shortlist nach Prefilter: **{summary.get('preferred_shortlist_after_prefilter')}**\n")
    if summary.get("preferred_shortlist_filtered_candidates") is not None:
        lines.append(f"- Durch Prefilter übersprungen: **{summary.get('preferred_shortlist_filtered_candidates')}**\n")
    if summary.get("candidate_search_pool") is not None:
        lines.append(f"- Candidate search pool: **{summary.get('candidate_search_pool')}**\n")
    if summary.get("discarded_cases") is not None:
        lines.append(f"- Bei der Generierung verworfen: **{summary.get('discarded_cases')}**\n")

    lines.append("\n## Wichtigste Erkenntnisse\n")
    wins = []
    no_wins = []
    for ps in report.providers:
        if ps.best_boundary is None or ps.best_composed_ms is None or ps.full_baseline_ms is None:
            continue
        if ps.delta_vs_full_ms is not None and ps.delta_vs_full_ms < 0:
            wins.append(ps)
        else:
            no_wins.append(ps)
    if wins:
        for ps in wins:
            lines.append(
                f"- **{ps.tag}**: bester Split **b{ps.best_boundary}** mit **{_fmt(ps.best_composed_ms)} ms**, "
                f"also **{_fmt(abs(ps.delta_vs_full_pct),1)} % schneller** als Full.\n"
            )
    if no_wins:
        for ps in no_wins:
            lines.append(
                f"- **{ps.tag}**: bester Split **b{ps.best_boundary}** mit **{_fmt(ps.best_composed_ms)} ms**, "
                f"damit **{_fmt(ps.delta_vs_full_pct,1)} % langsamer** als Full.\n"
            )

    top_candidates = report.candidate_summaries[:5]
    if top_candidates:
        lines.append("\n## Robuste Kandidaten über mehrere Provider\n")
        for cand in top_candidates:
            lines.append(
                f"- **b{cand.boundary}**: mittlerer Rang **{_fmt(cand.avg_rank,2)}**, Top-3 in **{cand.top3_hits}** Provider-Läufen, "
                f"bester Provider: **{cand.best_provider}** (Rang {cand.best_rank}).\n"
            )

    if report.hailo_context_summaries:
        lines.append("\n## Hailo Context Fit\n")
        for hw_arch in sorted({row.hw_arch for row in report.hailo_context_summaries}, key=lambda s: (_slug(s) != "hailo8", _slug(s))):
            stats = _hailo_target_stats(report, hw_arch)
            lines.append(
                f"- **{hw_arch}**: part2 läuft bei **{stats['part2_single']} / {stats['total']}** Splits in **einem Kontext**, "
                f"bei **{stats['both_single']} / {stats['total']}** sogar **part1 und part2** jeweils in einem Kontext. "
                f"Fallback **single→multi**: **{stats['fallback_multi']}**, direkt **multi-context**: **{stats['multi']}**.\n"
            )
            best_single = stats.get("best_single")
            if isinstance(best_single, HailoContextSummary):
                lines.append(
                    f"  - Schnellster direkte Hailo-Ein-Kontext-Split: **b{best_single.boundary}** "
                    f"mit **{_fmt(best_single.direct_hailo_composed_ms)} ms** "
                    f"(cut={_fmt(best_single.cut_mib,2)} MiB, score={_fmt(best_single.score_pred,2)}).\n"
                )
    else:
        lines.append("\n## Hailo Context Fit\n")
        lines.append("- Keine Hailo-Kontext-Metadaten im Benchmark-Set / CSV gefunden. Für diese Ansicht muss das Benchmark-Set mit dem gepatchten Tool neu erzeugt werden.\n")

    fallback_summary = hailo_part2_fallback_summary(report)
    if int(fallback_summary.get("fallback_count") or 0) > 0:
        lines.append("\n## Hailo Part2 Suggested-End Fallback\n")
        lines.append(
            f"- **{int(fallback_summary.get('fallback_count') or 0)} / {int(fallback_summary.get('candidate_count') or 0)}** Kandidaten verwenden beim Benchmark-Set-Bau die alternative `suggested end-node`-Strategie.\n"
        )
        if fallback_summary.get("top_boundary") is not None:
            lines.append(
                f"- Bester Fallback-Kandidat: **b{int(fallback_summary.get('top_boundary'))}** mit mittlerem Rang **{_fmt(fallback_summary.get('avg_rank'), 2)}**.\n"
            )
        lines.append(
            f"- Davon wirken **{int(fallback_summary.get('single_context_likely_count') or 0)}** Kandidaten laut Outlook trotzdem wahrscheinlich 1-Kontext-fähig.\n"
        )

    lines.append("\n## Prognosegüte\n")
    for ps in report.providers:
        lines.append(
            f"- **{ps.tag}**: Spearman(score→actual) = **{_fmt(ps.score_spearman,2)}**, "
            f"Spearman(latency_model→actual) = **{_fmt(ps.latency_spearman,2)}**, "
            f"Top-5-Overlap = **{ps.top5_overlap if ps.top5_overlap is not None else '-'}**.\n"
        )

    if report.issues:
        lines.append("\n## Hinweise\n")
        for item in report.issues:
            lines.append(f"- {item}\n")

    return "".join(lines)



def load_benchmark_analysis(source: str | Path, cache_base: Path) -> BenchmarkAnalysisReport:
    src = discover_benchmark_analysis_source(source, cache_base=cache_base)
    benchmark_set_path = src.results_root / "benchmark_set.json"
    bench = migrate_benchmark_set_payload(_read_json(benchmark_set_path))
    summary = dict(bench.get("summary") or {})
    summary.setdefault("schema_version", bench.get("schema_version"))
    summary.setdefault("objective", (bench.get("plan") or {}).get("objective") if isinstance(bench.get("plan"), dict) else bench.get("objective"))
    if isinstance(bench.get("tool"), dict):
        summary["_tool_meta"] = dict(bench.get("tool") or {})
    cases = list(bench.get("cases") or [])
    summary.setdefault("case_count", len(cases))

    provider_rows: Dict[str, List[Dict[str, Any]]] = {}
    issues: List[str] = []
    for csv_path in sorted(src.results_root.glob("benchmark_results_*.csv")):
        tag = _provider_tag_from_csv(csv_path)
        rows = _read_csv_rows(csv_path)
        provider_rows[tag] = rows
    if not provider_rows:
        raise FileNotFoundError(f"No benchmark_results_*.csv files found in {src.results_root}")

    case_by_boundary: Dict[int, Dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        b = _as_int(case.get("boundary"))
        if b is None or b in case_by_boundary:
            continue
        case_by_boundary[int(b)] = case

    for rows in provider_rows.values():
        for row in rows:
            b = _as_int(row.get("boundary"))
            if b is None:
                continue
            case = case_by_boundary.get(int(b))
            if not isinstance(case, dict):
                continue
            pred = case.get("predicted")
            if isinstance(pred, dict):
                for kk, vv in pred.items():
                    row.setdefault(str(kk), vv)
            strategy = str(case.get('hailo_part2_output_strategy') or 'original').strip() or 'original'
            eff_outputs = [str(x).strip() for x in list(case.get('hailo_part2_effective_outputs') or []) if str(x).strip()]
            row.setdefault('hailo_part2_output_strategy', strategy)
            row.setdefault('hailo_part2_effective_outputs', list(eff_outputs))
            row.setdefault('hailo_part2_fallback_used', bool(strategy != 'original' or eff_outputs))

    providers = [_provider_summary(tag, rows) for tag, rows in provider_rows.items()]
    providers.sort(key=lambda ps: ps.tag)

    expected_cases = _as_int(summary.get("generated_cases") or summary.get("requested_cases"))
    if expected_cases is not None:
        for ps in providers:
            if ps.row_count != expected_cases:
                issues.append(
                    f"{ps.tag}: erwartet {expected_cases} Cases laut benchmark_set, gefunden {ps.row_count} im CSV."
                )

    for tag, rows in provider_rows.items():
        pass_flags = [str(r.get("final_pass_all", "")).strip().lower() in {"true", "1", "yes"} for r in rows]
        if pass_flags and not all(pass_flags):
            issues.append(f"{tag}: nicht alle Cases haben final_pass_all=True.")

    candidate_summaries = _candidate_summaries(provider_rows)
    hailo_context_summaries = _extract_hailo_context_summaries(cases, provider_rows)
    if not hailo_context_summaries and any("hailo" in str(tag).lower() for tag in provider_rows):
        issues.append(
            "Hailo-Läufe vorhanden, aber keine Hailo-Kontext-Metadaten im Benchmark-Set/CSV. "
            "Ein neu generiertes Benchmark-Set mit dem aktuellen Tool exportiert diese Informationen direkt."
        )

    report = BenchmarkAnalysisReport(
        source=src,
        benchmark_set_path=benchmark_set_path,
        summary=summary,
        providers=providers,
        provider_rows=provider_rows,
        candidate_summaries=candidate_summaries,
        hailo_context_summaries=hailo_context_summaries,
        provider_tags=[ps.tag for ps in providers],
        bench_cases=cases,
        summary_markdown="",
        issues=issues,
    )
    report.summary_markdown = build_summary_markdown(report)
    return report



def provider_summary_rows(report: BenchmarkAnalysisReport) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ps in report.providers:
        rows.append(
            {
                "provider": ps.tag,
                "rows": ps.row_count,
                "full_baseline_ms": ps.full_baseline_ms,
                "best_boundary": ps.best_boundary,
                "best_composed_ms": ps.best_composed_ms,
                "delta_vs_full_ms": ps.delta_vs_full_ms,
                "delta_vs_full_pct": ps.delta_vs_full_pct,
                "score_spearman": ps.score_spearman,
                "latency_spearman": ps.latency_spearman,
                "top5_overlap": ps.top5_overlap,
                "best_actual_predicted_rank": ps.best_actual_predicted_rank,
                "best_predicted_actual_rank": ps.best_predicted_actual_rank,
            }
        )
    return rows



def candidate_summary_rows(report: BenchmarkAnalysisReport, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fallback_meta = _hailo_part2_fallback_meta_map(report.bench_cases)
    source = report.candidate_summaries if limit is None else report.candidate_summaries[: max(0, int(limit))]
    for cand in source:
        meta = fallback_meta.get(int(cand.boundary), {})
        rows.append(
            {
                "boundary": cand.boundary,
                "hailo_part2_marker": meta.get("hailo_part2_marker", "-"),
                "hailo_part2_fallback_used": bool(meta.get("hailo_part2_fallback_used", False)),
                "hailo_part2_output_strategy": meta.get("hailo_part2_output_strategy", "original"),
                "hailo_part2_strategy_label": meta.get("hailo_part2_strategy_label", "orig"),
                "hailo_part2_effective_outputs": list(meta.get("hailo_part2_effective_outputs") or []),
                "hailo_part2_effective_outputs_text": meta.get("hailo_part2_effective_outputs_text", ""),
                "avg_rank": cand.avg_rank,
                "top3_hits": cand.top3_hits,
                "providers_present": cand.providers_present,
                "best_provider": cand.best_provider,
                "best_rank": cand.best_rank,
                "score_pred": cand.score_pred,
                "cut_mib": cand.cut_mib,
                "hailo_compile_risk_score": cand.hailo_compile_risk_score,
                "hailo_single_context_probability": cand.hailo_single_context_probability,
            }
        )
    return rows



def _hailo_risk_band(score: Optional[float]) -> str:
    if score is None or not math.isfinite(float(score)):
        return "n/a"
    s = float(score)
    if s <= 1.7:
        return "low"
    if s <= 2.5:
        return "medium"
    return "high"



def _hailo_recommendation(single_prob: Optional[float], risk: Optional[float]) -> str:
    if single_prob is None or risk is None:
        return "insufficient data"
    if float(single_prob) >= 0.80 and float(risk) <= 1.9:
        return "Very likely 1-context"
    if float(single_prob) >= 0.65:
        return "Likely 1-context"
    if float(single_prob) >= 0.45:
        return "Borderline / measure"
    return "Compile-risky / likely multi-context"



def hailo_outlook_rows(report: BenchmarkAnalysisReport, limit: Optional[int] = 12) -> List[Dict[str, Any]]:
    cands = [c for c in report.candidate_summaries if c.hailo_compile_risk_score is not None or c.hailo_single_context_probability is not None]
    fallback_meta = _hailo_part2_fallback_meta_map(report.bench_cases)
    rows: List[Dict[str, Any]] = []
    if cands:
        cands.sort(
            key=lambda c: (
                -(float(c.hailo_single_context_probability) if c.hailo_single_context_probability is not None else -1.0),
                float(c.hailo_compile_risk_score) if c.hailo_compile_risk_score is not None else 1e9,
                float(c.avg_rank),
                int(c.boundary),
            )
        )
        if limit is not None:
            cands = cands[: max(0, int(limit))]
        for idx, cand in enumerate(cands, start=1):
            meta = fallback_meta.get(int(cand.boundary), {})
            rows.append({
                "rank": idx,
                "boundary": cand.boundary,
                "hailo_part2_marker": meta.get("hailo_part2_marker", "-"),
                "hailo_part2_fallback_used": bool(meta.get("hailo_part2_fallback_used", False)),
                "hailo_part2_output_strategy": meta.get("hailo_part2_output_strategy", "original"),
                "hailo_part2_strategy_label": meta.get("hailo_part2_strategy_label", "orig"),
                "hailo_part2_effective_outputs": list(meta.get("hailo_part2_effective_outputs") or []),
                "hailo_part2_effective_outputs_text": meta.get("hailo_part2_effective_outputs_text", ""),
                "compile_risk_score": cand.hailo_compile_risk_score,
                "risk_band": _hailo_risk_band(cand.hailo_compile_risk_score),
                "single_context_probability": cand.hailo_single_context_probability,
                "cut_mib": cand.cut_mib,
                "score_pred": cand.score_pred,
                "avg_rank": cand.avg_rank,
                "providers_present": cand.providers_present,
                "recommendation": _hailo_recommendation(cand.hailo_single_context_probability, cand.hailo_compile_risk_score),
            })
        return rows

    # Fallback for older benchmark sets: derive a coarse Hailo outlook from measured
    # context-fit metadata so the analysis tab still shows something useful.
    grouped: Dict[int, Dict[str, Any]] = {}
    for item in report.hailo_context_summaries:
        rec = grouped.setdefault(int(item.boundary), {
            "boundary": int(item.boundary),
            "cut_mib": item.cut_mib,
            "score_pred": item.score_pred,
            "avg_rank": None,
            "providers_present": 0,
            "single_hits": 0,
            "context_counts": [],
            "fallback_hits": 0,
        })
        rec["providers_present"] = int(rec.get("providers_present", 0)) + 1
        if item.part2_context_count is not None:
            rec.setdefault("context_counts", []).append(int(item.part2_context_count))
        if item.part2_single_context:
            rec["single_hits"] = int(rec.get("single_hits", 0)) + 1
        if str(item.part2_context_mode or "") == "single_context_failed_to_multi":
            rec["fallback_hits"] = int(rec.get("fallback_hits", 0)) + 1
    if not grouped:
        return rows
    avg_rank_map = {c.boundary: c.avg_rank for c in report.candidate_summaries}
    for rec in grouped.values():
        boundary = int(rec["boundary"])
        ctxs = [int(x) for x in rec.get("context_counts") or []]
        avg_ctx = (sum(ctxs) / len(ctxs)) if ctxs else None
        single_prob = float(rec.get("single_hits", 0)) / float(rec.get("providers_present") or 1)
        if rec.get("fallback_hits", 0):
            risk = 2.8
        elif avg_ctx is None:
            risk = None
        elif avg_ctx <= 1.0:
            risk = 1.2
        elif avg_ctx <= 2.0:
            risk = 2.1
        else:
            risk = 3.0
        meta = fallback_meta.get(int(boundary), {})
        rows.append({
            "rank": None,
            "boundary": boundary,
            "hailo_part2_marker": meta.get("hailo_part2_marker", "-"),
            "hailo_part2_fallback_used": bool(meta.get("hailo_part2_fallback_used", False)),
            "hailo_part2_output_strategy": meta.get("hailo_part2_output_strategy", "original"),
            "hailo_part2_strategy_label": meta.get("hailo_part2_strategy_label", "orig"),
            "hailo_part2_effective_outputs": list(meta.get("hailo_part2_effective_outputs") or []),
            "hailo_part2_effective_outputs_text": meta.get("hailo_part2_effective_outputs_text", ""),
            "compile_risk_score": risk,
            "risk_band": _hailo_risk_band(risk),
            "single_context_probability": single_prob,
            "cut_mib": rec.get("cut_mib"),
            "score_pred": rec.get("score_pred"),
            "avg_rank": avg_rank_map.get(boundary),
            "providers_present": rec.get("providers_present"),
            "recommendation": _hailo_recommendation(single_prob, risk),
        })
    rows.sort(
        key=lambda r: (
            -(float(r.get("single_context_probability")) if r.get("single_context_probability") is not None else -1.0),
            float(r.get("compile_risk_score")) if r.get("compile_risk_score") is not None else 1e9,
            float(r.get("avg_rank")) if r.get("avg_rank") is not None else 1e9,
            int(r.get("boundary") or 0),
        )
    )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows



def hailo_outlook_summary(report: BenchmarkAnalysisReport) -> Dict[str, Any]:
    rows = hailo_outlook_rows(report, limit=None)
    if not rows:
        return {
            "candidate_count": 0,
            "avg_risk_score": None,
            "likely_single_context_count": 0,
            "low_risk_count": 0,
            "medium_risk_count": 0,
            "high_risk_count": 0,
            "top_boundary": None,
            "top_single_context_probability": None,
        }
    risks = [float(r["compile_risk_score"]) for r in rows if r.get("compile_risk_score") is not None]
    return {
        "candidate_count": len(rows),
        "avg_risk_score": (sum(risks) / len(risks) if risks else None),
        "likely_single_context_count": sum(1 for r in rows if (r.get("single_context_probability") or 0.0) >= 0.65),
        "low_risk_count": sum(1 for r in rows if r.get("risk_band") == "low"),
        "medium_risk_count": sum(1 for r in rows if r.get("risk_band") == "medium"),
        "high_risk_count": sum(1 for r in rows if r.get("risk_band") == "high"),
        "top_boundary": rows[0].get("boundary"),
        "top_single_context_probability": rows[0].get("single_context_probability"),
    }



def hailo_context_rows(report: BenchmarkAnalysisReport, hw_arch: Optional[str] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    src = report.hailo_context_summaries
    if hw_arch:
        src = [row for row in src if _slug(row.hw_arch) == _slug(hw_arch)]
    for row in src:
        rows.append(
            {
                "boundary": row.boundary,
                "hw_arch": row.hw_arch,
                "part1_context_count": row.part1_context_count,
                "part1_context_mode": row.part1_context_mode,
                "part2_context_count": row.part2_context_count,
                "part2_context_mode": row.part2_context_mode,
                "part2_single_context": row.part2_single_context,
                "both_parts_single_context": row.both_parts_single_context,
                "part1_partition_time_s": row.part1_partition_time_s,
                "part1_allocation_time_s": row.part1_allocation_time_s,
                "part1_compilation_time_s": row.part1_compilation_time_s,
                "part1_elapsed_s": row.part1_elapsed_s,
                "part2_partition_time_s": row.part2_partition_time_s,
                "part2_allocation_time_s": row.part2_allocation_time_s,
                "part2_compilation_time_s": row.part2_compilation_time_s,
                "part2_elapsed_s": row.part2_elapsed_s,
                "part2_calib_source": row.part2_calib_source,
                "score_pred": row.score_pred,
                "cut_mib": row.cut_mib,
                "direct_hailo_provider": row.direct_hailo_provider,
                "direct_hailo_composed_ms": row.direct_hailo_composed_ms,
            }
        )
    return rows



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



def _placeholder_figure(title: str, message: str) -> Figure:
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    return fig



def build_best_vs_full_figure(report: BenchmarkAnalysisReport) -> Figure:
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    labels = [ps.tag for ps in report.providers]
    deltas = [ps.delta_vs_full_pct if ps.delta_vs_full_pct is not None else float("nan") for ps in report.providers]
    ax.bar(labels, deltas)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("Best split vs full [%]")
    ax.set_title("Bester Split relativ zu Full")
    ax.tick_params(axis="x", rotation=25)
    for idx, ps in enumerate(report.providers):
        val = deltas[idx]
        if math.isfinite(val):
            ax.text(idx, val, f"b{ps.best_boundary}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    fig.tight_layout()
    return fig



def build_predictor_quality_figure(report: BenchmarkAnalysisReport) -> Figure:
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    labels = [ps.tag for ps in report.providers]
    x = np.arange(len(labels), dtype=float)
    w = 0.35
    score_vals = [ps.score_spearman if ps.score_spearman is not None else float("nan") for ps in report.providers]
    lat_vals = [ps.latency_spearman if ps.latency_spearman is not None else float("nan") for ps in report.providers]
    ax.bar(x - w / 2.0, score_vals, width=w, label="score vs actual")
    ax.bar(x + w / 2.0, lat_vals, width=w, label="latency model vs actual")
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Prognosegüte nach Provider")
    ax.legend()
    fig.tight_layout()
    return fig



def build_candidate_stability_figure(report: BenchmarkAnalysisReport, topn: int = 8) -> Figure:
    top = report.candidate_summaries[: max(1, int(topn))]
    if not top:
        return _placeholder_figure("Kandidaten-Stabilität", "Keine Kandidaten-Daten verfügbar.")
    fig = Figure(figsize=(7.0, max(2.8, 0.45 * len(top) + 1.8)), dpi=100)
    ax = fig.add_subplot(111)
    providers = report.provider_tags
    matrix = np.full((len(top), len(providers)), np.nan, dtype=float)
    rank_maps = {tag: _actual_rank_map(rows) for tag, rows in report.provider_rows.items()}
    for i, cand in enumerate(top):
        for j, tag in enumerate(providers):
            rk = rank_maps.get(tag, {}).get(cand.boundary)
            if rk is not None:
                matrix[i, j] = float(rk)
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(providers)))
    ax.set_xticklabels(providers, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([f"b{cand.boundary}" for cand in top])
    ax.set_title("Kandidaten-Stabilität (Rang je Provider)")
    ax.set_xlabel("Provider")
    ax.set_ylabel("Boundary")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if math.isfinite(matrix[i, j]):
                ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.9, label="Rang")
    fig.tight_layout()
    return fig



def build_hailo_context_count_figure(report: BenchmarkAnalysisReport, hw_arch: Optional[str] = None) -> Figure:
    target = hw_arch or _preferred_hailo_target(report)
    if not target:
        return _placeholder_figure("Hailo Context Fit", "Keine Hailo-Kontext-Metadaten verfügbar.")
    rows = [row for row in report.hailo_context_summaries if _slug(row.hw_arch) == _slug(target)]
    if not rows:
        return _placeholder_figure("Hailo Context Fit", f"Keine Hailo-Kontext-Daten für {target} verfügbar.")
    rows.sort(key=lambda row: row.boundary)
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    labels = [f"b{row.boundary}" for row in rows]
    counts = [float(row.part2_context_count) if row.part2_context_count is not None else float("nan") for row in rows]
    ax.bar(labels, counts)
    ax.set_ylabel("part2 context count")
    ax.set_title(f"Hailo Context Fit ({target})")
    ax.tick_params(axis="x", rotation=70)
    for idx, row in enumerate(rows):
        count = counts[idx]
        if math.isfinite(count):
            mode = row.part2_context_mode.replace("single_context_failed_to_multi", "fallback")
            mode = mode.replace("single_context_used", "single")
            mode = mode.replace("multi_context_used", "multi")
            ax.text(idx, count, mode, ha="center", va="bottom", fontsize=7, rotation=90)
    fig.tight_layout()
    return fig



def build_hailo_latency_vs_context_figure(report: BenchmarkAnalysisReport, hw_arch: Optional[str] = None) -> Figure:
    target = hw_arch or _preferred_hailo_target(report)
    if not target:
        return _placeholder_figure("Hailo Latenz vs Kontext", "Keine Hailo-Kontext-Metadaten verfügbar.")
    rows = [row for row in report.hailo_context_summaries if _slug(row.hw_arch) == _slug(target) and row.direct_hailo_composed_ms is not None and row.part2_context_count is not None]
    if not rows:
        return _placeholder_figure("Hailo Latenz vs Kontext", f"Keine direkten Hailo-Latenzen für {target} verfügbar.")
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    xs = [float(row.part2_context_count or 0) for row in rows]
    ys = [float(row.direct_hailo_composed_ms or 0.0) for row in rows]
    ax.scatter(xs, ys)
    for row, x, y in zip(rows, xs, ys):
        ax.text(x, y, f"b{row.boundary}", fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("part2 context count")
    ax.set_ylabel("direct Hailo composed latency [ms]")
    ax.set_title(f"Hailo-Latenz vs Kontextzahl ({target})")
    fig.tight_layout()
    return fig



def build_hailo_cut_vs_context_figure(report: BenchmarkAnalysisReport, hw_arch: Optional[str] = None) -> Figure:
    target = hw_arch or _preferred_hailo_target(report)
    if not target:
        return _placeholder_figure("Cut-Größe vs Kontext", "Keine Hailo-Kontext-Metadaten verfügbar.")
    rows = [row for row in report.hailo_context_summaries if _slug(row.hw_arch) == _slug(target) and row.cut_mib is not None and row.part2_context_count is not None]
    if not rows:
        return _placeholder_figure("Cut-Größe vs Kontext", f"Keine Cut/Context-Daten für {target} verfügbar.")
    fig = Figure(figsize=(7.0, 3.8), dpi=100)
    ax = fig.add_subplot(111)
    xs = [float(row.cut_mib or 0.0) for row in rows]
    ys = [float(row.part2_context_count or 0) for row in rows]
    ax.scatter(xs, ys)
    for row, x, y in zip(rows, xs, ys):
        ax.text(x, y, f"b{row.boundary}", fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("predicted cut size [MiB]")
    ax.set_ylabel("part2 context count")
    ax.set_title(f"Cut-Größe vs Kontextzahl ({target})")
    fig.tight_layout()
    return fig



def build_hailo_outlook_risk_figure(report: BenchmarkAnalysisReport) -> Figure:
    rows = hailo_outlook_rows(report, limit=12)
    if not rows:
        return _placeholder_figure("Hailo Compile-Risiko", "Keine Hailo-Outlook-Daten vorhanden.")
    fig = Figure(figsize=(7.4, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    color_map = {"low": "#2e8b57", "medium": "#d18f00", "high": "#c0392b"}
    labels = [f"b{int(r['boundary'])}" for r in rows]
    risks = [float(r.get('compile_risk_score') or 0.0) for r in rows]
    probs = [float(r.get('single_context_probability') or 0.0) for r in rows]
    colors = [color_map.get(str(r.get('risk_band') or '').lower(), '#666666') for r in rows]
    ypos = list(range(len(rows)))[::-1]
    ax.barh(ypos, risks, color=colors, alpha=0.85)
    for y, risk, prob in zip(ypos, risks, probs):
        ax.text(risk + 0.05, y, f"{prob*100:.0f}% 1ctx", va="center", fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Compile risk score (niedriger ist besser)")
    ax.set_title("Hailo Compile-Risiko und 1-Kontext-Wahrscheinlichkeit")
    fig.tight_layout()
    return fig


def build_hailo_outlook_scatter_figure(report: BenchmarkAnalysisReport) -> Figure:
    rows = hailo_outlook_rows(report, limit=20)
    if not rows:
        return _placeholder_figure("Hailo Outlook Scatter", "Keine Hailo-Outlook-Daten vorhanden.")
    fig = Figure(figsize=(7.4, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    color_map = {"low": "#2e8b57", "medium": "#d18f00", "high": "#c0392b"}
    xs = [float(r.get('compile_risk_score') or 0.0) for r in rows]
    ys = [float(r.get('single_context_probability') or 0.0) * 100.0 for r in rows]
    sizes = [max(40.0, float(r.get('cut_mib') or 0.0) * 30.0) for r in rows]
    colors = [color_map.get(str(r.get('risk_band') or '').lower(), '#666666') for r in rows]
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.8)
    for row, x, y in zip(rows, xs, ys):
        ax.annotate(f"b{int(row['boundary'])}", (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Compile risk score")
    ax.set_ylabel("1-Kontext-Wahrscheinlichkeit [%]")
    ax.set_ylim(0, 100)
    ax.set_title("Hailo Outlook: Risiko vs 1-Kontext-Wahrscheinlichkeit")
    fig.tight_layout()
    return fig


def build_hailo_part2_fallback_figure(report: BenchmarkAnalysisReport) -> Figure:
    rows = hailo_part2_fallback_rows(report, limit=12)
    if not rows:
        return _placeholder_figure("Hailo Part2 Fallback", "Keine Kandidaten mit suggested end-node fallback vorhanden.")
    fig = Figure(figsize=(7.4, 4.2), dpi=100)
    ax = fig.add_subplot(111)
    labels = [f"b{int(row['boundary'])}" for row in rows]
    avg_ranks = [float(row.get('avg_rank') or 0.0) for row in rows]
    ypos = list(range(len(rows)))[::-1]
    ax.barh(ypos, avg_ranks, color='#4c78a8', alpha=0.85)
    for y, row, rank in zip(ypos, rows, avg_ranks):
        prob = row.get('single_context_probability')
        prob_txt = f"{float(prob) * 100.0:.0f}% 1ctx" if prob is not None else "? 1ctx"
        strat = str(row.get('strategy_label') or row.get('strategy') or 'alt')
        ax.text(rank + 0.08, y, f"{strat} · {prob_txt}", va='center', fontsize=8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ø Rang (niedriger ist besser)")
    ax.set_title("Hailo Part2 suggested-end fallback-Kandidaten")
    fig.tight_layout()
    return fig


def build_benchmark_analysis_figures(report: BenchmarkAnalysisReport) -> Dict[str, Figure]:
    target = _preferred_hailo_target(report)
    return {
        "best_vs_full": build_best_vs_full_figure(report),
        "predictor_quality": build_predictor_quality_figure(report),
        "candidate_stability": build_candidate_stability_figure(report),
        "hailo_context_fit": build_hailo_context_count_figure(report, hw_arch=target),
        "hailo_latency_vs_context": build_hailo_latency_vs_context_figure(report, hw_arch=target),
        "hailo_cut_vs_context": build_hailo_cut_vs_context_figure(report, hw_arch=target),
        "hailo_outlook_risk": build_hailo_outlook_risk_figure(report),
        "hailo_outlook_scatter": build_hailo_outlook_scatter_figure(report),
        "hailo_part2_fallback": build_hailo_part2_fallback_figure(report),
    }



def _build_caption_lines(report: BenchmarkAnalysisReport) -> List[str]:
    target = _preferred_hailo_target(report) or "hailo"
    return [
        "# Benchmark-Analyse – Figure Captions\n",
        f"- `benchmark_analysis_best_vs_full.*`: Bester gemessener Split relativ zur Full-Baseline pro Provider für `{report.source.display_name}`.\n",
        f"- `benchmark_analysis_predictor_quality.*`: Spearman-Korrelation der Benchmark-Prognosen gegenüber den real gemessenen Latenzen.\n",
        f"- `benchmark_analysis_candidate_stability.*`: Rang-Stabilität der Top-Kandidaten über alle gemessenen Provider hinweg.\n",
        f"- `benchmark_analysis_hailo_outlook.*`: Compile-/1-Kontext-Outlook für Hailo-freundliche Kandidaten inklusive Risikoband.\n",
        f"- `benchmark_analysis_hailo_context_fit.*`: Kontextzahl von `part2` pro Boundary für `{target}` inklusive Single-/Multi-/Fallback-Modus.\n",
        f"- `benchmark_analysis_hailo_latency_vs_context.*`: Direkte Hailo-Latenz gegen `part2`-Kontextzahl für `{target}`.\n",
        f"- `benchmark_analysis_hailo_cut_vs_context.*`: Prognostizierte Cut-Größe gegen `part2`-Kontextzahl für `{target}`.\n",
        f"- `benchmark_analysis_hailo_outlook_risk.*`: Top-Hailo-Kandidaten als Compile-Risiko-Balken inklusive 1-Kontext-Wahrscheinlichkeit.\n",
        f"- `benchmark_analysis_hailo_outlook_scatter.*`: Risiko-vs-1-Kontext-Scatter für die wichtigsten Hailo-Kandidaten.\n",
        f"- `benchmark_analysis_hailo_part2_fallback.*`: Kandidaten, die beim Hailo-Part2-Build eine `suggested end-node`-Fallback-Strategie verwenden.\n",
    ]



def export_benchmark_analysis(report: BenchmarkAnalysisReport, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    summary_md = output_dir / "benchmark_analysis_summary.md"
    summary_md.write_text(report.summary_markdown, encoding="utf-8")
    paths["summary_md"] = summary_md

    provider_csv = output_dir / "benchmark_analysis_provider_summary.csv"
    _write_csv(provider_csv, provider_summary_rows(report))
    paths["provider_csv"] = provider_csv

    cand_csv = output_dir / "benchmark_analysis_candidate_summary.csv"
    _write_csv(cand_csv, candidate_summary_rows(report, limit=20))
    paths["candidate_csv"] = cand_csv

    hailo_csv = output_dir / "benchmark_analysis_hailo_context_fit.csv"
    _write_csv(hailo_csv, hailo_context_rows(report))
    paths["hailo_context_csv"] = hailo_csv

    hailo_outlook_csv = output_dir / "benchmark_analysis_hailo_outlook.csv"
    _write_csv(hailo_outlook_csv, hailo_outlook_rows(report, limit=20))
    paths["hailo_outlook_csv"] = hailo_outlook_csv

    hailo_fallback_csv = output_dir / "benchmark_analysis_hailo_part2_fallback.csv"
    _write_csv(hailo_fallback_csv, hailo_part2_fallback_rows(report, limit=20))
    paths["hailo_fallback_csv"] = hailo_fallback_csv

    captions_md = output_dir / "benchmark_analysis_captions.md"
    captions_md.write_text("".join(_build_caption_lines(report)), encoding="utf-8")
    paths["captions_md"] = captions_md

    plots = build_benchmark_analysis_figures(report)
    for name, fig in plots.items():
        for ext in ("png", "pdf", "svg"):
            out_path = output_dir / f"benchmark_analysis_{name}.{ext}"
            fig.savefig(out_path, bbox_inches="tight")
            paths[f"plot_{name}_{ext}"] = out_path
    return paths



def _delta(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    return float(new) - float(old)



def _delta_pct(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None:
        return None
    if abs(float(old)) <= 1e-12:
        return None
    return ((float(new) - float(old)) / float(old)) * 100.0



def _comparison_provider_summaries(left: BenchmarkAnalysisReport, right: BenchmarkAnalysisReport) -> List[ComparisonProviderSummary]:
    left_map = {ps.tag: ps for ps in left.providers}
    right_map = {ps.tag: ps for ps in right.providers}
    tags = sorted(set(left_map) | set(right_map))
    out: List[ComparisonProviderSummary] = []
    for tag in tags:
        l = left_map.get(tag)
        r = right_map.get(tag)
        left_best_boundary = l.best_boundary if l is not None else None
        right_best_boundary = r.best_boundary if r is not None else None
        out.append(
            ComparisonProviderSummary(
                provider=tag,
                left_rows=l.row_count if l is not None else 0,
                right_rows=r.row_count if r is not None else 0,
                left_full_baseline_ms=l.full_baseline_ms if l is not None else None,
                right_full_baseline_ms=r.full_baseline_ms if r is not None else None,
                full_delta_ms=_delta(r.full_baseline_ms if r is not None else None, l.full_baseline_ms if l is not None else None),
                full_delta_pct=_delta_pct(r.full_baseline_ms if r is not None else None, l.full_baseline_ms if l is not None else None),
                left_best_boundary=left_best_boundary,
                right_best_boundary=right_best_boundary,
                left_best_composed_ms=l.best_composed_ms if l is not None else None,
                right_best_composed_ms=r.best_composed_ms if r is not None else None,
                best_delta_ms=_delta(r.best_composed_ms if r is not None else None, l.best_composed_ms if l is not None else None),
                best_delta_pct=_delta_pct(r.best_composed_ms if r is not None else None, l.best_composed_ms if l is not None else None),
                left_score_spearman=l.score_spearman if l is not None else None,
                right_score_spearman=r.score_spearman if r is not None else None,
                score_spearman_delta=_delta(r.score_spearman if r is not None else None, l.score_spearman if l is not None else None),
                left_latency_spearman=l.latency_spearman if l is not None else None,
                right_latency_spearman=r.latency_spearman if r is not None else None,
                latency_spearman_delta=_delta(r.latency_spearman if r is not None else None, l.latency_spearman if l is not None else None),
                best_boundary_changed=(left_best_boundary is not None and right_best_boundary is not None and left_best_boundary != right_best_boundary),
            )
        )
    return out



def _candidate_map(report: BenchmarkAnalysisReport) -> Dict[int, CandidateSummary]:
    return {int(c.boundary): c for c in report.candidate_summaries}



def _comparison_candidate_summaries(left: BenchmarkAnalysisReport, right: BenchmarkAnalysisReport) -> List[ComparisonCandidateSummary]:
    left_map = _candidate_map(left)
    right_map = _candidate_map(right)
    boundaries = sorted(set(left_map) | set(right_map))
    out: List[ComparisonCandidateSummary] = []
    for boundary in boundaries:
        l = left_map.get(boundary)
        r = right_map.get(boundary)
        out.append(
            ComparisonCandidateSummary(
                boundary=int(boundary),
                left_avg_rank=l.avg_rank if l is not None else None,
                right_avg_rank=r.avg_rank if r is not None else None,
                avg_rank_delta=_delta(r.avg_rank if r is not None else None, l.avg_rank if l is not None else None),
                left_top3_hits=l.top3_hits if l is not None else 0,
                right_top3_hits=r.top3_hits if r is not None else 0,
                top3_delta=(r.top3_hits if r is not None else 0) - (l.top3_hits if l is not None else 0),
                left_best_provider=l.best_provider if l is not None else None,
                right_best_provider=r.best_provider if r is not None else None,
                left_best_rank=l.best_rank if l is not None else None,
                right_best_rank=r.best_rank if r is not None else None,
                left_score_pred=l.score_pred if l is not None else None,
                right_score_pred=r.score_pred if r is not None else None,
                left_cut_mib=l.cut_mib if l is not None else None,
                right_cut_mib=r.cut_mib if r is not None else None,
            )
        )
    out.sort(key=lambda row: (row.avg_rank_delta is None, -(abs(float(row.avg_rank_delta)) if row.avg_rank_delta is not None else -1.0), row.boundary))
    return out



def _hailo_context_map(report: BenchmarkAnalysisReport) -> Dict[Tuple[int, str], HailoContextSummary]:
    return {(int(row.boundary), _slug(row.hw_arch)): row for row in report.hailo_context_summaries}



def _comparison_hailo_context_summaries(left: BenchmarkAnalysisReport, right: BenchmarkAnalysisReport) -> List[ComparisonHailoContextSummary]:
    left_map = _hailo_context_map(left)
    right_map = _hailo_context_map(right)
    keys = sorted(set(left_map) | set(right_map), key=lambda item: (item[1], item[0]))
    out: List[ComparisonHailoContextSummary] = []
    for boundary, arch_slug in keys:
        l = left_map.get((boundary, arch_slug))
        r = right_map.get((boundary, arch_slug))
        hw_arch = r.hw_arch if r is not None else (l.hw_arch if l is not None else arch_slug)
        out.append(
            ComparisonHailoContextSummary(
                boundary=int(boundary),
                hw_arch=str(hw_arch),
                left_part2_context_count=l.part2_context_count if l is not None else None,
                right_part2_context_count=r.part2_context_count if r is not None else None,
                context_delta=_delta(float(r.part2_context_count) if (r is not None and r.part2_context_count is not None) else None, float(l.part2_context_count) if (l is not None and l.part2_context_count is not None) else None),
                left_part2_context_mode=l.part2_context_mode if l is not None else '-',
                right_part2_context_mode=r.part2_context_mode if r is not None else '-',
                left_part2_single_context=bool(l.part2_single_context) if l is not None else False,
                right_part2_single_context=bool(r.part2_single_context) if r is not None else False,
                single_context_changed=((bool(l.part2_single_context) if l is not None else False) != (bool(r.part2_single_context) if r is not None else False)),
                left_direct_hailo_composed_ms=l.direct_hailo_composed_ms if l is not None else None,
                right_direct_hailo_composed_ms=r.direct_hailo_composed_ms if r is not None else None,
                latency_delta_ms=_delta(r.direct_hailo_composed_ms if r is not None else None, l.direct_hailo_composed_ms if l is not None else None),
                left_score_pred=l.score_pred if l is not None else None,
                right_score_pred=r.score_pred if r is not None else None,
                left_cut_mib=l.cut_mib if l is not None else None,
                right_cut_mib=r.cut_mib if r is not None else None,
            )
        )
    out.sort(key=lambda row: (_slug(row.hw_arch), row.boundary))
    return out



def build_comparison_summary_markdown(comparison: BenchmarkComparisonReport) -> str:
    left = comparison.left
    right = comparison.right
    lines: List[str] = []
    lines.append(f"# Benchmark-Vergleich: {left.source.display_name} → {right.source.display_name}\n")
    lines.append("\n## Quellen\n")
    lines.append(f"- Lauf A: `{left.source.source_path}`\n")
    lines.append(f"- Lauf B: `{right.source.source_path}`\n")
    lines.append(f"- Cases A/B: **{left.summary.get('generated_cases', left.summary.get('case_count', '-'))} / {right.summary.get('generated_cases', right.summary.get('case_count', '-'))}**\n")

    left_tags = set(left.provider_tags)
    right_tags = set(right.provider_tags)
    common_tags = sorted(left_tags & right_tags)
    added = sorted(right_tags - left_tags)
    removed = sorted(left_tags - right_tags)
    lines.append("\n## Provider-Abdeckung\n")
    lines.append(f"- Gemeinsam: **{', '.join(common_tags) if common_tags else '-'}**\n")
    if added:
        lines.append(f"- Nur in Lauf B: **{', '.join(added)}**\n")
    if removed:
        lines.append(f"- Nur in Lauf A: **{', '.join(removed)}**\n")

    lines.append("\n## Latenz-Änderungen je Provider\n")
    comparable = [row for row in comparison.provider_summaries if row.left_full_baseline_ms is not None and row.right_full_baseline_ms is not None]
    if comparable:
        for row in comparable:
            full_phrase = f"Full { _fmt(row.right_full_baseline_ms)} ms ({_fmt(row.full_delta_pct,1)} % vs A)" if row.right_full_baseline_ms is not None else "Full -"
            best_phrase = f"best split b{row.right_best_boundary} @ {_fmt(row.right_best_composed_ms)} ms ({_fmt(row.best_delta_pct,1)} % vs A)" if row.right_best_boundary is not None else "best split -"
            if row.best_boundary_changed:
                best_phrase += f", vorher b{row.left_best_boundary}"
            lines.append(f"- **{row.provider}**: {full_phrase}; {best_phrase}.\n")
    else:
        lines.append("- Keine gemeinsamen Provider mit vergleichbaren Full-/Split-Latenzen.\n")

    improved = [row for row in comparison.candidate_summaries if row.avg_rank_delta is not None and row.avg_rank_delta < 0]
    improved.sort(key=lambda row: row.avg_rank_delta or 0.0)
    regressed = [row for row in comparison.candidate_summaries if row.avg_rank_delta is not None and row.avg_rank_delta > 0]
    regressed.sort(key=lambda row: -(row.avg_rank_delta or 0.0))
    lines.append("\n## Größte Rangverschiebungen\n")
    if improved:
        lines.append("- Verbesserte Kandidaten in Lauf B:\n")
        for row in improved[:5]:
            lines.append(
                f"  - **b{row.boundary}**: Ø-Rang **{_fmt(row.left_avg_rank,2)} → {_fmt(row.right_avg_rank,2)}** "
                f"(Δ **{_fmt(row.avg_rank_delta,2)}**), Top-3 Hits **{row.left_top3_hits} → {row.right_top3_hits}**.\n"
            )
    if regressed:
        lines.append("- Schlechtere Kandidaten in Lauf B:\n")
        for row in regressed[:5]:
            lines.append(
                f"  - **b{row.boundary}**: Ø-Rang **{_fmt(row.left_avg_rank,2)} → {_fmt(row.right_avg_rank,2)}** "
                f"(Δ **{_fmt(row.avg_rank_delta,2)}**), Top-3 Hits **{row.left_top3_hits} → {row.right_top3_hits}**.\n"
            )
    if not improved and not regressed:
        lines.append("- Keine vergleichbaren Kandidaten mit Rangverschiebungen.\n")

    lines.append("\n## Hailo Context Fit – Änderungen\n")
    hailo_rows = comparison.hailo_context_summaries
    if hailo_rows:
        archs = sorted({row.hw_arch for row in hailo_rows}, key=lambda s: (_slug(s) != 'hailo8', _slug(s)))
        for arch in archs:
            rows = [row for row in hailo_rows if _slug(row.hw_arch) == _slug(arch)]
            left_single = sum(1 for row in rows if row.left_part2_single_context)
            right_single = sum(1 for row in rows if row.right_part2_single_context)
            gained = [row.boundary for row in rows if (not row.left_part2_single_context and row.right_part2_single_context)]
            lost = [row.boundary for row in rows if (row.left_part2_single_context and not row.right_part2_single_context)]
            lines.append(
                f"- **{arch}**: part2 Single-Context **{left_single} → {right_single}** über **{len(rows)}** vergleichbare Splits.\n"
            )
            if gained:
                lines.append(f"  - Neu Single-Context in Lauf B: {', '.join(f'b{b}' for b in gained[:8])}{' …' if len(gained) > 8 else ''}\n")
            if lost:
                lines.append(f"  - Nicht mehr Single-Context in Lauf B: {', '.join(f'b{b}' for b in lost[:8])}{' …' if len(lost) > 8 else ''}\n")
    else:
        lines.append("- Keine vergleichbaren Hailo-Context-Metadaten vorhanden.\n")

    lines.append("\n## Prognosegüte – Änderungen\n")
    for row in comparison.provider_summaries:
        if row.left_score_spearman is None and row.right_score_spearman is None:
            continue
        lines.append(
            f"- **{row.provider}**: score-Spearman **{_fmt(row.left_score_spearman,2)} → {_fmt(row.right_score_spearman,2)}** "
            f"(Δ {_fmt(row.score_spearman_delta,2)}), latency-Spearman **{_fmt(row.left_latency_spearman,2)} → {_fmt(row.right_latency_spearman,2)}** "
            f"(Δ {_fmt(row.latency_spearman_delta,2)}).\n"
        )

    if comparison.issues:
        lines.append("\n## Hinweise\n")
        for issue in comparison.issues:
            lines.append(f"- {issue}\n")

    return ''.join(lines)



def compare_benchmark_reports(left: BenchmarkAnalysisReport, right: BenchmarkAnalysisReport) -> BenchmarkComparisonReport:
    issues: List[str] = []
    left_tags = set(left.provider_tags)
    right_tags = set(right.provider_tags)
    if left_tags != right_tags:
        added = sorted(right_tags - left_tags)
        removed = sorted(left_tags - right_tags)
        if added:
            issues.append(f"Nur in Lauf B vorhandene Provider: {', '.join(added)}")
        if removed:
            issues.append(f"Nur in Lauf A vorhandene Provider: {', '.join(removed)}")
    provider_summaries = _comparison_provider_summaries(left, right)
    candidate_summaries = _comparison_candidate_summaries(left, right)
    hailo_context_summaries = _comparison_hailo_context_summaries(left, right)
    comp = BenchmarkComparisonReport(
        left=left,
        right=right,
        provider_summaries=provider_summaries,
        candidate_summaries=candidate_summaries,
        hailo_context_summaries=hailo_context_summaries,
        summary_markdown='',
        issues=issues,
    )
    comp.summary_markdown = build_comparison_summary_markdown(comp)
    return comp



def load_benchmark_analysis_comparison(left_source: str | Path, right_source: str | Path, cache_base: Path) -> BenchmarkComparisonReport:
    left = load_benchmark_analysis(left_source, cache_base=cache_base)
    right = load_benchmark_analysis(right_source, cache_base=cache_base)
    return compare_benchmark_reports(left, right)



def comparison_provider_rows(comparison: BenchmarkComparisonReport) -> List[Dict[str, Any]]:
    return [
        {
            'provider': row.provider,
            'left_rows': row.left_rows,
            'right_rows': row.right_rows,
            'left_full_baseline_ms': row.left_full_baseline_ms,
            'right_full_baseline_ms': row.right_full_baseline_ms,
            'full_delta_ms': row.full_delta_ms,
            'full_delta_pct': row.full_delta_pct,
            'left_best_boundary': row.left_best_boundary,
            'right_best_boundary': row.right_best_boundary,
            'left_best_composed_ms': row.left_best_composed_ms,
            'right_best_composed_ms': row.right_best_composed_ms,
            'best_delta_ms': row.best_delta_ms,
            'best_delta_pct': row.best_delta_pct,
            'left_score_spearman': row.left_score_spearman,
            'right_score_spearman': row.right_score_spearman,
            'score_spearman_delta': row.score_spearman_delta,
            'left_latency_spearman': row.left_latency_spearman,
            'right_latency_spearman': row.right_latency_spearman,
            'latency_spearman_delta': row.latency_spearman_delta,
            'best_boundary_changed': row.best_boundary_changed,
        }
        for row in comparison.provider_summaries
    ]



def comparison_candidate_rows(comparison: BenchmarkComparisonReport, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    src = comparison.candidate_summaries if limit is None else comparison.candidate_summaries[: max(0, int(limit))]
    return [
        {
            'boundary': row.boundary,
            'left_avg_rank': row.left_avg_rank,
            'right_avg_rank': row.right_avg_rank,
            'avg_rank_delta': row.avg_rank_delta,
            'left_top3_hits': row.left_top3_hits,
            'right_top3_hits': row.right_top3_hits,
            'top3_delta': row.top3_delta,
            'left_best_provider': row.left_best_provider,
            'right_best_provider': row.right_best_provider,
            'left_best_rank': row.left_best_rank,
            'right_best_rank': row.right_best_rank,
            'left_score_pred': row.left_score_pred,
            'right_score_pred': row.right_score_pred,
            'left_cut_mib': row.left_cut_mib,
            'right_cut_mib': row.right_cut_mib,
        }
        for row in src
    ]



def comparison_hailo_rows(comparison: BenchmarkComparisonReport, hw_arch: Optional[str] = None) -> List[Dict[str, Any]]:
    src = comparison.hailo_context_summaries
    if hw_arch:
        src = [row for row in src if _slug(row.hw_arch) == _slug(hw_arch)]
    return [
        {
            'boundary': row.boundary,
            'hw_arch': row.hw_arch,
            'left_part2_context_count': row.left_part2_context_count,
            'right_part2_context_count': row.right_part2_context_count,
            'context_delta': row.context_delta,
            'left_part2_context_mode': row.left_part2_context_mode,
            'right_part2_context_mode': row.right_part2_context_mode,
            'left_part2_single_context': row.left_part2_single_context,
            'right_part2_single_context': row.right_part2_single_context,
            'single_context_changed': row.single_context_changed,
            'left_direct_hailo_composed_ms': row.left_direct_hailo_composed_ms,
            'right_direct_hailo_composed_ms': row.right_direct_hailo_composed_ms,
            'latency_delta_ms': row.latency_delta_ms,
            'left_score_pred': row.left_score_pred,
            'right_score_pred': row.right_score_pred,
            'left_cut_mib': row.left_cut_mib,
            'right_cut_mib': row.right_cut_mib,
        }
        for row in src
    ]



def hailo_part2_fallback_rows(report: BenchmarkAnalysisReport, limit: Optional[int] = 20) -> List[Dict[str, Any]]:
    meta_map = _hailo_part2_fallback_meta_map(report.bench_cases)
    cand_map = {int(row.get("boundary")): row for row in candidate_summary_rows(report, limit=None) if row.get("boundary") is not None}
    outlook_map = {int(row.get("boundary")): row for row in hailo_outlook_rows(report, limit=None) if row.get("boundary") is not None}
    rows: List[Dict[str, Any]] = []
    for boundary, meta in sorted(meta_map.items()):
        if not bool(meta.get("hailo_part2_fallback_used", False)):
            continue
        cand = cand_map.get(int(boundary), {})
        outlook = outlook_map.get(int(boundary), {})
        rows.append({
            "boundary": int(boundary),
            "marker": meta.get("hailo_part2_marker", "↩"),
            "strategy": meta.get("hailo_part2_output_strategy", "original"),
            "strategy_label": meta.get("hailo_part2_strategy_label", "alt"),
            "effective_outputs": list(meta.get("hailo_part2_effective_outputs") or []),
            "effective_outputs_text": meta.get("hailo_part2_effective_outputs_text", ""),
            "avg_rank": cand.get("avg_rank"),
            "top3_hits": cand.get("top3_hits"),
            "providers_present": cand.get("providers_present"),
            "best_provider": cand.get("best_provider"),
            "best_rank": cand.get("best_rank"),
            "score_pred": cand.get("score_pred"),
            "cut_mib": cand.get("cut_mib"),
            "compile_risk_score": outlook.get("compile_risk_score"),
            "single_context_probability": outlook.get("single_context_probability"),
            "recommendation": outlook.get("recommendation") or "Uses suggested end-node fallback",
        })
    rows.sort(key=lambda r: (float(r.get("avg_rank")) if r.get("avg_rank") is not None else 1e9, int(r.get("best_rank") or 1e9), int(r.get("boundary") or 0)))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows



def hailo_part2_fallback_summary(report: BenchmarkAnalysisReport) -> Dict[str, Any]:
    rows = hailo_part2_fallback_rows(report, limit=None)
    if not rows:
        return {
            "fallback_count": 0,
            "candidate_count": len(report.candidate_summaries),
            "top_boundary": None,
            "avg_rank": None,
            "single_context_likely_count": 0,
        }
    likely = sum(1 for row in rows if row.get("single_context_probability") is not None and float(row.get("single_context_probability") or 0.0) >= 0.65)
    avg_rank = [float(row.get("avg_rank")) for row in rows if row.get("avg_rank") is not None]
    return {
        "fallback_count": len(rows),
        "candidate_count": len(report.candidate_summaries),
        "top_boundary": rows[0].get("boundary"),
        "avg_rank": (sum(avg_rank) / len(avg_rank)) if avg_rank else None,
        "single_context_likely_count": likely,
    }


def _preferred_hailo_target_comparison(comparison: BenchmarkComparisonReport) -> Optional[str]:
    left_archs = {row.hw_arch for row in comparison.left.hailo_context_summaries}
    right_archs = {row.hw_arch for row in comparison.right.hailo_context_summaries}
    common = sorted(left_archs & right_archs, key=lambda s: (_slug(s) != 'hailo8', _slug(s)))
    if common:
        return common[0]
    all_archs = sorted(left_archs | right_archs, key=lambda s: (_slug(s) != 'hailo8', _slug(s)))
    return all_archs[0] if all_archs else None



def build_comparison_provider_latency_figure(comparison: BenchmarkComparisonReport) -> Figure:
    rows = comparison.provider_summaries
    if not rows:
        return _placeholder_figure('Provider-Vergleich', 'Keine Provider-Daten für einen Vergleich verfügbar.')
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    labels = [row.provider for row in rows]
    x = np.arange(len(labels), dtype=float)
    w = 0.35
    full_vals = [row.full_delta_pct if row.full_delta_pct is not None else float('nan') for row in rows]
    best_vals = [row.best_delta_pct if row.best_delta_pct is not None else float('nan') for row in rows]
    ax.bar(x - w / 2.0, full_vals, width=w, label='Full Δ [%]')
    ax.bar(x + w / 2.0, best_vals, width=w, label='Best split Δ [%]')
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Lauf B relativ zu A [%]')
    ax.set_title('Provider-Latenzvergleich')
    ax.legend()
    fig.tight_layout()
    return fig



def build_comparison_predictor_delta_figure(comparison: BenchmarkComparisonReport) -> Figure:
    rows = comparison.provider_summaries
    if not rows:
        return _placeholder_figure('Prognosegüte-Vergleich', 'Keine Provider-Daten für einen Vergleich verfügbar.')
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    labels = [row.provider for row in rows]
    x = np.arange(len(labels), dtype=float)
    w = 0.35
    score_vals = [row.score_spearman_delta if row.score_spearman_delta is not None else float('nan') for row in rows]
    lat_vals = [row.latency_spearman_delta if row.latency_spearman_delta is not None else float('nan') for row in rows]
    ax.bar(x - w / 2.0, score_vals, width=w, label='Δ score Spearman')
    ax.bar(x + w / 2.0, lat_vals, width=w, label='Δ latency Spearman')
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylabel('Lauf B - Lauf A')
    ax.set_title('Prognosegüte-Vergleich')
    ax.legend()
    fig.tight_layout()
    return fig



def build_comparison_candidate_rank_shift_figure(comparison: BenchmarkComparisonReport, topn: int = 12) -> Figure:
    rows = [row for row in comparison.candidate_summaries if row.avg_rank_delta is not None]
    if not rows:
        return _placeholder_figure('Kandidaten-Verschiebung', 'Keine gemeinsamen Kandidaten mit vergleichbaren Rängen verfügbar.')
    rows.sort(key=lambda row: abs(float(row.avg_rank_delta or 0.0)), reverse=True)
    rows = rows[: max(1, int(topn))]
    fig = Figure(figsize=(7.4, max(3.2, 0.34 * len(rows) + 1.6)), dpi=100)
    ax = fig.add_subplot(111)
    labels = [f'b{row.boundary}' for row in rows]
    vals = [float(row.avg_rank_delta or 0.0) for row in rows]
    ypos = np.arange(len(rows), dtype=float)
    ax.barh(ypos, vals)
    ax.axvline(0.0, linewidth=1.0)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Δ Ø-Rang (B - A, negativ = besser)')
    ax.set_title('Größte Kandidaten-Verschiebungen')
    fig.tight_layout()
    return fig



def build_comparison_hailo_context_delta_figure(comparison: BenchmarkComparisonReport, hw_arch: Optional[str] = None) -> Figure:
    target = hw_arch or _preferred_hailo_target_comparison(comparison)
    if not target:
        return _placeholder_figure('Hailo Context Δ', 'Keine vergleichbaren Hailo-Daten verfügbar.')
    rows = [row for row in comparison.hailo_context_summaries if _slug(row.hw_arch) == _slug(target) and row.context_delta is not None]
    if not rows:
        return _placeholder_figure('Hailo Context Δ', f'Keine Kontext-Deltas für {target} verfügbar.')
    rows.sort(key=lambda row: row.boundary)
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    labels = [f'b{row.boundary}' for row in rows]
    vals = [float(row.context_delta or 0.0) for row in rows]
    ax.bar(labels, vals)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel('Δ part2 context count (B - A)')
    ax.set_title(f'Hailo Context-Fit Vergleich ({target})')
    ax.tick_params(axis='x', rotation=70)
    fig.tight_layout()
    return fig



def build_comparison_hailo_latency_delta_figure(comparison: BenchmarkComparisonReport, hw_arch: Optional[str] = None) -> Figure:
    target = hw_arch or _preferred_hailo_target_comparison(comparison)
    if not target:
        return _placeholder_figure('Hailo Latenz Δ', 'Keine vergleichbaren Hailo-Daten verfügbar.')
    rows = [row for row in comparison.hailo_context_summaries if _slug(row.hw_arch) == _slug(target) and row.latency_delta_ms is not None]
    if not rows:
        return _placeholder_figure('Hailo Latenz Δ', f'Keine direkten Hailo-Latenz-Deltas für {target} verfügbar.')
    rows.sort(key=lambda row: row.boundary)
    fig = Figure(figsize=(7.4, 4.0), dpi=100)
    ax = fig.add_subplot(111)
    labels = [f'b{row.boundary}' for row in rows]
    vals = [float(row.latency_delta_ms or 0.0) for row in rows]
    ax.bar(labels, vals)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel('Δ direct Hailo latency [ms] (B - A)')
    ax.set_title(f'Hailo-Latenzvergleich ({target})')
    ax.tick_params(axis='x', rotation=70)
    fig.tight_layout()
    return fig



def build_benchmark_comparison_figures(comparison: BenchmarkComparisonReport) -> Dict[str, Figure]:
    target = _preferred_hailo_target_comparison(comparison)
    return {
        'comparison_provider_latency': build_comparison_provider_latency_figure(comparison),
        'comparison_predictor_delta': build_comparison_predictor_delta_figure(comparison),
        'comparison_candidate_rank_shift': build_comparison_candidate_rank_shift_figure(comparison),
        'comparison_hailo_context_delta': build_comparison_hailo_context_delta_figure(comparison, hw_arch=target),
        'comparison_hailo_latency_delta': build_comparison_hailo_latency_delta_figure(comparison, hw_arch=target),
    }



def _build_comparison_caption_lines(comparison: BenchmarkComparisonReport) -> List[str]:
    target = _preferred_hailo_target_comparison(comparison) or 'hailo'
    return [
        '# Benchmark-Vergleich – Figure Captions\n',
        '- `benchmark_comparison_provider_latency.*`: Full- und Best-Split-Latenzen von Lauf B relativ zu Lauf A je Provider.\n',
        '- `benchmark_comparison_predictor_delta.*`: Änderung der Prognosegüte (Spearman) zwischen Lauf A und Lauf B.\n',
        '- `benchmark_comparison_candidate_rank_shift.*`: größte Verschiebungen des mittleren Kandidaten-Rangs zwischen Lauf A und Lauf B.\n',
        f'- `benchmark_comparison_hailo_context_delta.*`: Änderung der `part2`-Kontextzahl zwischen Lauf A und Lauf B für `{target}`.\n',
        f'- `benchmark_comparison_hailo_latency_delta.*`: Änderung der direkten Hailo-Latenz zwischen Lauf A und Lauf B für `{target}`.\n',
    ]



def export_benchmark_comparison(comparison: BenchmarkComparisonReport, output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    summary_md = output_dir / 'benchmark_comparison_summary.md'
    summary_md.write_text(comparison.summary_markdown, encoding='utf-8')
    paths['summary_md'] = summary_md

    provider_csv = output_dir / 'benchmark_comparison_provider_summary.csv'
    _write_csv(provider_csv, comparison_provider_rows(comparison))
    paths['provider_csv'] = provider_csv

    cand_csv = output_dir / 'benchmark_comparison_candidate_summary.csv'
    _write_csv(cand_csv, comparison_candidate_rows(comparison))
    paths['candidate_csv'] = cand_csv

    hailo_csv = output_dir / 'benchmark_comparison_hailo_context_summary.csv'
    _write_csv(hailo_csv, comparison_hailo_rows(comparison))
    paths['hailo_csv'] = hailo_csv

    captions_md = output_dir / 'benchmark_comparison_captions.md'
    captions_md.write_text(''.join(_build_comparison_caption_lines(comparison)), encoding='utf-8')
    paths['captions_md'] = captions_md

    plots = build_benchmark_comparison_figures(comparison)
    for name, fig in plots.items():
        for ext in ('png', 'pdf', 'svg'):
            out_path = output_dir / f'benchmark_{name}.{ext}'
            fig.savefig(out_path, bbox_inches='tight')
            paths[f'plot_{name}_{ext}'] = out_path
    return paths
