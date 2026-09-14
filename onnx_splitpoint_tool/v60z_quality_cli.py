from __future__ import annotations
from pathlib import Path
import argparse, json, yaml
from .v60z_integration import augment_scientific_report


def _rows(report_dir: Path):
    p=report_dir/'scientific_report.json'
    if not p.exists(): return []
    data=json.loads(p.read_text(encoding='utf-8'))
    if isinstance(data,list): return data
    for key in ('rows','results','canonical_rows','row_eligibility'):
        if isinstance(data.get(key),list): return data[key]
    return []


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--profile')
    ap.add_argument('--report-dir')
    ns=ap.parse_args(argv)
    run=Path(ns.run_dir).expanduser().resolve()
    report=Path(ns.report_dir).expanduser().resolve() if ns.report_dir else run/'reports'/'scientific'
    profile={}
    candidates=[Path(ns.profile)] if ns.profile else []
    candidates += [run/'evaluation_profile_resolved.yaml',run/'profile_resolved.yaml',run/'run_profile.yaml']
    for p in candidates:
        if p and p.exists():
            profile=yaml.safe_load(p.read_text(encoding='utf-8')) or {}; break
    result=augment_scientific_report(run_dir=run,report_dir=report,profile=profile,rows=_rows(report))
    print(json.dumps(result,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
