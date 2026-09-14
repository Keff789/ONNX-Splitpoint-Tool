#!/usr/bin/env python3
"""Apply the consistent accuracy/eligibility gate to any result-summary JSON.

The script is intentionally generic: it accepts summaries produced by native
validation, generic benchmark analysis, full-baseline validators, or custom CSV/JSON
conversions as long as rows are present under ``rows``.  It adds these fields to
each row:

  buildable, runtime_executable, contract_consistent, task_valid,
  accuracy_gate_pass, eligible_for_ranking, gate_status, accuracy_gate_reason

By default, Full-ONNX/native self-reference rows are contract-consistent but not
eligible for ranking unless dataset-level accuracy is available and within the
thresholds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy, apply_accuracy_gates_to_payload  # noqa:E402


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [r for r in payload.get("rows", []) if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input JSON summary containing rows")
    ap.add_argument("--out-json", default="", help="Output JSON path. Default: <input stem>_gated.json")
    ap.add_argument("--out-csv", default="", help="Optional CSV path. Default: alongside JSON")
    ap.add_argument("--out-md", default="", help="Optional Markdown path. Default: alongside JSON")
    ap.add_argument("--classification-max-top1-drop", type=float, default=0.01)
    ap.add_argument("--classification-max-top5-drop", type=float, default=0.01)
    ap.add_argument("--detection-max-ap50-drop", type=float, default=0.01)
    ap.add_argument("--detection-max-ap-drop", type=float, default=0.01)
    ap.add_argument("--allow-contract-only-ranking", action="store_true")
    ns = ap.parse_args()

    inp = Path(ns.input).expanduser().resolve()
    payload = _load_json(inp)
    if isinstance(payload, list):
        payload = {"schema": "onnx-splitpoint/gated-row-list", "rows": payload}
    rows = _rows_from_payload(payload)
    policy = AccuracyGatePolicy(
        classification_max_top1_drop=float(ns.classification_max_top1_drop),
        classification_max_top5_drop=float(ns.classification_max_top5_drop),
        detection_max_ap50_drop=float(ns.detection_max_ap50_drop),
        detection_max_ap_drop=float(ns.detection_max_ap_drop),
        contract_only_eligible_for_ranking=bool(ns.allow_contract_only_ranking),
    )
    apply_accuracy_gates_to_payload(payload, policy)
    payload["schema_version"] = max(int(payload.get("schema_version") or 0), 1)
    payload["accuracy_gate_note"] = (
        "eligible_for_ranking requires buildable, runtime_executable, contract_consistent, "
        "task_valid and accuracy_gate_pass. Full-ONNX/native self-reference is a contract gate, "
        "not a dataset accuracy gate, unless --allow-contract-only-ranking is explicitly used."
    )

    out_json = Path(ns.out_json).expanduser().resolve() if ns.out_json else inp.with_name(inp.stem + "_gated.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fields: list[str] = []
    for preferred in [
        "backend", "provider", "setup", "model", "case", "precision", "task",
        "buildable", "runtime_executable", "contract_consistent", "task_valid",
        "accuracy_gate_pass", "eligible_for_ranking", "gate_status", "accuracy_gate_reason",
        "quality", "ap50", "ap50_proxy", "top1", "top1_match", "top5_overlap",
        "accuracy_primary", "accuracy_reference", "accuracy_delta_vs_reference",
        "status", "semantic_reference_source", "self_reference_diagnosis",
    ]:
        if any(preferred in r for r in rows):
            fields.append(preferred)
    for r in rows:
        for k in r.keys():
            if k not in fields and isinstance(k, str):
                fields.append(k)

    out_csv = Path(ns.out_csv).expanduser().resolve() if ns.out_csv else out_json.with_suffix(".csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    out_md = Path(ns.out_md).expanduser().resolve() if ns.out_md else out_json.with_suffix(".md")
    lines = [
        "# Accuracy-gated result summary", "",
        f"Source: `{inp}`", "",
        f"Policy: `{json.dumps(policy.to_json(), sort_keys=True)}`", "",
        f"Rows: `{len(rows)}` · eligible_for_ranking: `{payload.get('eligible_for_ranking_count')}` · accuracy_gate_pass: `{payload.get('accuracy_gate_pass_count')}`", "",
        "| backend/setup | model | case | precision | task | buildable | runtime | contract | task_valid | acc_gate | eligible | reason |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        setup = r.get("backend") or r.get("provider") or r.get("setup") or ""
        lines.append(
            f"| {setup} | {r.get('model','')} | {r.get('case','')} | {r.get('precision','')} | {r.get('task','')} | "
            f"{r.get('buildable','')} | {r.get('runtime_executable','')} | {r.get('contract_consistent','')} | "
            f"{r.get('task_valid','')} | {r.get('accuracy_gate_pass','')} | {r.get('eligible_for_ranking','')} | {r.get('accuracy_gate_reason','')} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": True,
        "input": str(inp),
        "json": str(out_json),
        "csv": str(out_csv),
        "md": str(out_md),
        "row_count": len(rows),
        "accuracy_gate_pass_count": payload.get("accuracy_gate_pass_count"),
        "eligible_for_ranking_count": payload.get("eligible_for_ranking_count"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
