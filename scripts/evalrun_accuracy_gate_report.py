#!/usr/bin/env python3
"""Create one thesis-facing accuracy-gate report for an EvaluationRun.

The report merges rows from the reports that are commonly produced by the tool:

* reports/native_validation/native_producer_validation_summary.json
* reports/native_producer_combined_summary.json
* any user-supplied --summary JSON files

The output separates execution/contract evidence from task/accuracy eligibility,
so fast rows without a declared task gate are not silently ranked.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from onnx_splitpoint_tool.accuracy_gates import load_policy, apply_accuracy_gates_to_rows  # type: ignore


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _source_rows(path: Path, kind: str) -> list[dict[str, Any]]:
    data = _load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return []
    out = []
    for r in data.get("rows") or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        rr.setdefault("source_summary_kind", kind)
        rr.setdefault("source_summary", str(path))
        out.append(rr)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "source_summary_kind","backend","model","case","precision","task","status",
        "buildable","runtime_executable","contract_consistent","task_valid","accuracy_gate_pass","eligible_for_ranking",
        "gate_status","accuracy_gate_reason","accuracy_reference_source","semantic_reference_source",
        "claim_ok","semantic_ok","ap50_proxy","top1_match","top5_overlap","decode_mode","fps_makespan","handoff_ms","source_summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _write_md(path: Path, rows: list[dict[str, Any]], policy: dict[str, Any], run_dir: Path) -> None:
    md = [
        "# EvaluationRun accuracy/ranking gate report",
        "",
        f"Run: `{run_dir}`",
        "",
        f"Policy: `{policy}`",
        "",
        "The `eligible` column is the only safe ranking gate for speed/energy tables. Native self-reference rows can be contract-consistent but remain ineligible unless a separate task/accuracy gate is available.",
        "",
        "## Summary",
        "",
        f"- rows: `{len(rows)}`",
        f"- buildable: `{sum(1 for r in rows if r.get('buildable') is True)}`",
        f"- runtime executable: `{sum(1 for r in rows if r.get('runtime_executable') is True)}`",
        f"- contract consistent: `{sum(1 for r in rows if r.get('contract_consistent') is True)}`",
        f"- task valid: `{sum(1 for r in rows if r.get('task_valid') is True)}`",
        f"- accuracy gate pass: `{sum(1 for r in rows if r.get('accuracy_gate_pass') is True)}`",
        f"- eligible for ranking: `{sum(1 for r in rows if r.get('eligible_for_ranking') is True)}`",
        "",
        "## Rows",
        "",
        "| source | backend | model | case | precision | status | contract | task | acc | eligible | reason |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r.get('source_summary_kind','')} | {r.get('backend','')} | {r.get('model','')} | {r.get('case','')} | {r.get('precision','')} | "
            f"{r.get('status','')} | {r.get('contract_consistent','')} | {r.get('task_valid','')} | {r.get('accuracy_gate_pass','')} | "
            f"{r.get('eligible_for_ranking','')} | {r.get('accuracy_gate_reason') or r.get('gate_status','')} |"
        )
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--summary", action="append", default=[], help="Additional summary JSON with rows[]")
    ap.add_argument("--policy", default="")
    ap.add_argument("--allow-self-reference-as-task-gate", action="store_true", default=False)
    ns = ap.parse_args()
    run = Path(ns.run_dir).expanduser().resolve()
    reports = run / "reports"
    candidates = [
        (reports / "native_validation" / "native_producer_validation_summary.json", "native_validation"),
        (reports / "native_producer_combined_summary.json", "native_execution"),
    ]
    for s in ns.summary:
        candidates.append((Path(s).expanduser().resolve(), "user_summary"))
    rows: list[dict[str, Any]] = []
    seen = set()
    for p, kind in candidates:
        if not p.is_file():
            continue
        for r in _source_rows(p, kind):
            key = (kind, r.get("backend"), r.get("model"), r.get("case"), r.get("precision"), r.get("source_summary"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    policy = load_policy(ns.policy)
    if ns.allow_self_reference_as_task_gate:
        cfg = policy.to_dict(); cfg["native_self_reference_is_task_gate"] = True
        policy = load_policy(cfg)
    gated = apply_accuracy_gates_to_rows(rows, policy)
    outdir = Path(ns.out_dir).expanduser().resolve() if ns.out_dir else reports / "accuracy_gates"
    outdir.mkdir(parents=True, exist_ok=True)
    jsonp = outdir / "accuracy_gate_summary.json"
    csvp = outdir / "accuracy_gate_summary.csv"
    mdp = outdir / "accuracy_gate_summary.md"
    payload = {
        "schema": "onnx-splitpoint/evalrun-accuracy-gate-summary",
        "schema_version": 1,
        "run_dir": str(run),
        "accuracy_gate_policy": policy.to_dict(),
        "rows": gated,
        "row_count": len(gated),
        "buildable_count": sum(1 for r in gated if r.get("buildable") is True),
        "runtime_executable_count": sum(1 for r in gated if r.get("runtime_executable") is True),
        "contract_consistent_count": sum(1 for r in gated if r.get("contract_consistent") is True),
        "task_valid_count": sum(1 for r in gated if r.get("task_valid") is True),
        "accuracy_gate_pass_count": sum(1 for r in gated if r.get("accuracy_gate_pass") is True),
        "eligible_for_ranking_count": sum(1 for r in gated if r.get("eligible_for_ranking") is True),
    }
    jsonp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(csvp, gated)
    _write_md(mdp, gated, policy.to_dict(), run)
    print(json.dumps({"ok": True, "rows": len(gated), "eligible_for_ranking_count": payload["eligible_for_ranking_count"], "json": str(jsonp), "csv": str(csvp), "md": str(mdp)}, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
