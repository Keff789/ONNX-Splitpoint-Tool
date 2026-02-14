"""Context export helpers for CV-style analysis outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


def _objective_value(row: Dict[str, Any], objective: str) -> Optional[float]:
    full = row.get("full_mean_ms")
    p1 = row.get("part1_mean_ms")
    p2 = row.get("part2_mean_ms")
    comp = row.get("composed_mean_ms")
    if objective == "full":
        return float(full) if full is not None else None
    if objective == "composed":
        return float(comp) if comp is not None else None
    if objective == "sum_parts":
        if p1 is None or p2 is None:
            return None
        return float(p1) + float(p2)
    if objective == "max_parts":
        if p1 is None or p2 is None:
            return None
        return max(float(p1), float(p2))
    return None


def _write_results_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _write_table_tex(path: Path, rows: List[Dict[str, Any]], tag: str, objective: str, topk: int = 10) -> None:
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: float("inf") if _objective_value(r, objective) is None else float(_objective_value(r, objective)))
    rows_sorted = rows_sorted[: max(1, min(topk, len(rows_sorted)))]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Boundary & Cut (MiB) & \#T & $F_L$ (GFLOPs) & $F_R$ (GFLOPs) & Pass \\",
        r"\midrule",
    ]
    for r in rows_sorted:
        b = r.get("boundary", "-")
        cut = r.get("cut_mib", r.get("cut_mb", "-"))
        nt = r.get("n_cut_tensors", r.get("num_tensors", "-"))
        fl = r.get("flops_left", "-")
        fr = r.get("flops_right", "-")
        ok = r.get("eps_pass", r.get("ok", "-"))
        lines.append(f"{b} & {cut} & {nt} & {fl} & {fr} & {ok} \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Top split candidates for \\texttt{{{tag}}} ({objective}).}}",
        r"\end{table}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary_md(path: Path, rows: List[Dict[str, Any]], tag: str, objective: str, topk: int = 10) -> None:
    if not rows:
        path.write_text("# Benchmark summary\n\nNo results.\n", encoding="utf-8")
        return
    rows_sorted = sorted(rows, key=lambda r: float("inf") if _objective_value(r, objective) is None else float(_objective_value(r, objective)))
    best = rows_sorted[: max(1, min(topk, len(rows_sorted)))]
    lines = [
        "# Benchmark summary",
        "",
        f"- tag: `{tag}`",
        f"- objective: `{objective}`",
        f"- rows: {len(rows)}",
        "",
        "## Top candidates",
    ]
    for r in best:
        lines.append(f"- boundary {r.get('boundary', '-')}: objective={_objective_value(r, objective)}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_context(path_base: Path, rows: List[Dict[str, Any]], tag: str, objective: str, topk: int = 10) -> None:
    _write_results_csv(path_base.with_suffix('.csv'), rows)
    _write_table_tex(path_base.with_suffix('.tex'), rows, tag, objective, topk=topk)
    _write_summary_md(path_base.with_suffix('.md'), rows, tag, objective, topk=topk)
