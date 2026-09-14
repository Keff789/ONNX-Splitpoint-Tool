#!/usr/bin/env python3
"""Apply the central task/accuracy gate policy to an existing JSON summary.

Works with native producer validation summaries, native producer combined
summaries, generic benchmark row summaries and decision-table-like JSON files
that contain a top-level ``rows`` list.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy, apply_accuracy_gates_to_payload, gate_counts


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields=[]; seen=set()
    preferred=[
        'backend','setup','pipeline','model','case','boundary','precision','task','status',
        'buildable','runtime_executable','contract_consistent','task_valid','accuracy_gate_pass','eligible_for_ranking',
        'gate_status','validation_verdict','ranking_exclusion_reason','accuracy_gate_reason',
        'quality','ap50','ap50_proxy','quality_delta_vs_full','ap50_delta_vs_full','top1_match','top5_overlap','streaming_fps','FPS'
    ]
    for f in preferred:
        if any(f in r for r in rows) and f not in seen:
            seen.add(f); fields.append(f)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open('w', newline='', encoding='utf-8') as fh:
        w=csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def _write_md(path: Path, payload: dict) -> None:
    rows=payload.get('rows') if isinstance(payload.get('rows'), list) else []
    lines=['# Accuracy gate summary','']
    lines.append('Rows are eligible for ranking only when buildable, runtime_executable, contract_consistent, task_valid and accuracy_gate_pass are all true. Native Full-ONNX self-reference is a contract gate, not a dataset accuracy gate by default.\n')
    lines.append('## Counts\n')
    for k in ['row_count','buildable_count','runtime_executable_count','contract_consistent_count','task_valid_count','accuracy_gate_pass_count','eligible_for_ranking_count']:
        lines.append(f'- `{k}`: `{payload.get(k, len(rows) if k=="row_count" else 0)}`')
    lines.append('\n## Rows\n')
    lines.append('| backend/setup | model | case/boundary | precision | task | build | run | contract | task_valid | acc_gate | eligible | reason | metric |')
    lines.append('|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|')
    for r in rows:
        setup=r.get('backend') or r.get('setup') or r.get('pipeline') or ''
        case=r.get('case') or r.get('boundary') or ''
        metric=r.get('accuracy_gate_delta')
        if isinstance(r.get('accuracy_gate_metrics'), dict):
            metric=r.get('accuracy_gate_metrics')
        lines.append(f"| {setup} | {r.get('model','')} | {case} | {r.get('precision','')} | {r.get('task','')} | {r.get('buildable','')} | {r.get('runtime_executable','')} | {r.get('contract_consistent','')} | {r.get('task_valid','')} | {r.get('accuracy_gate_pass','')} | {r.get('eligible_for_ranking','')} | {r.get('ranking_exclusion_reason') or r.get('accuracy_gate_reason','')} | `{metric}` |")
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--summary', required=True, help='Input JSON file with a top-level rows list')
    ap.add_argument('--out', default='', help='Output JSON. Default: <summary stem>_gated.json')
    ap.add_argument('--csv', default='', help='Optional output CSV. Default: next to JSON')
    ap.add_argument('--md', default='', help='Optional output markdown. Default: next to JSON')
    ap.add_argument('--classification-max-top1-drop', type=float, default=0.01)
    ap.add_argument('--classification-max-top5-drop', type=float, default=0.01)
    ap.add_argument('--detection-max-ap50-drop', type=float, default=0.01)
    ap.add_argument('--detection-max-ap-drop', type=float, default=0.01)
    ap.add_argument('--allow-contract-only-ranking', action='store_true')
    ns=ap.parse_args()
    inp=Path(ns.summary).expanduser().resolve()
    payload=json.loads(inp.read_text(encoding='utf-8'))
    policy=AccuracyGatePolicy(
        classification_max_top1_drop=ns.classification_max_top1_drop,
        classification_max_top5_drop=ns.classification_max_top5_drop,
        detection_max_ap50_drop=ns.detection_max_ap50_drop,
        detection_max_ap_drop=ns.detection_max_ap_drop,
        contract_only_eligible_for_ranking=bool(ns.allow_contract_only_ranking),
    )
    apply_accuracy_gates_to_payload(payload, policy)
    rows=payload.get('rows') if isinstance(payload.get('rows'), list) else []
    payload['row_count']=len(rows)
    payload.update(gate_counts(rows))
    out=Path(ns.out).expanduser().resolve() if ns.out else inp.with_name(inp.stem+'_gated.json')
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    csvp=Path(ns.csv).expanduser().resolve() if ns.csv else out.with_suffix('.csv')
    mdp=Path(ns.md).expanduser().resolve() if ns.md else out.with_suffix('.md')
    _write_csv(csvp, rows)
    _write_md(mdp, payload)
    print(json.dumps({'ok': True, 'json': str(out), 'csv': str(csvp), 'md': str(mdp), 'row_count': len(rows), 'eligible_for_ranking_count': payload.get('eligible_for_ranking_count')}, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
