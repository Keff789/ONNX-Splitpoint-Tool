#!/usr/bin/env python3
"""Inspect the installed DeepX runtime Python API and case artifacts.

This is a safe introspection helper for implementing the DeepX native FIFO producer.
It does not run inference; it lists importable modules, public attributes, likely
classes/functions, and case artifact paths.
"""
from __future__ import annotations
import argparse, importlib, inspect, json, os, sys
from pathlib import Path
from typing import Any

CANDIDATE_MODULES=['dx_engine','dxrt','deepx','dx_runtime','dx_com','dx_engine.inference','dx_engine.runtime']

def _case_id(x:str)->str:
    return x if str(x).startswith('b') else f'b{int(x):03d}'

def _mod_info(name:str)->dict[str,Any]:
    info={'module':name,'importable':False,'error':'','public_attrs':[],'classes':[],'functions':[]}
    try:
        m=importlib.import_module(name)
        info['importable']=True
        attrs=[a for a in dir(m) if not a.startswith('_')]
        info['public_attrs']=attrs[:200]
        for a in attrs:
            try:
                obj=getattr(m,a)
                if inspect.isclass(obj):
                    methods=[x for x in dir(obj) if not x.startswith('_')][:80]
                    info['classes'].append({'name':a,'module':getattr(obj,'__module__',''),'methods':methods})
                elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                    try: sig=str(inspect.signature(obj))
                    except Exception: sig='?'
                    info['functions'].append({'name':a,'signature':sig})
            except Exception:
                pass
    except Exception as e:
        info['error']=repr(e)
    return info

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark-set', default='')
    ap.add_argument('--case', default='')
    ap.add_argument('--out', default='')
    ns=ap.parse_args()
    report={'python':sys.executable,'cwd':os.getcwd(),'modules':[_mod_info(m) for m in CANDIDATE_MODULES]}
    if ns.benchmark_set and ns.case:
        bs=Path(ns.benchmark_set).expanduser().resolve(); case=_case_id(ns.case)
        cdir=bs/case
        arts=[]
        for p in sorted(cdir.rglob('*')):
            if any(s in str(p).lower() for s in ['deepx','dxnn','output_contract','runtime_manifest','config_deepx']):
                arts.append(str(p))
        report.update({'benchmark_set':str(bs),'case':case,'deepx_artifacts':arts[:500]})
        for rel in [f'{case}/deepx/deepx_m1/part1/output_contract.json', f'{case}/deepx/deepx_m1/part1/config_deepx.json', f'{case}/deepx/deepx_m1/part1/deepx_part1_artifact_status.json']:
            p=bs/rel
            if p.is_file():
                try: report[p.name]=json.loads(p.read_text(encoding='utf-8'))
                except Exception as e: report[p.name]={'error':repr(e),'path':str(p)}
    out=Path(ns.out).expanduser().resolve() if ns.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report if not out else {'ok':True,'out':str(out)}, indent=2)[:20000])
    return 0
if __name__=='__main__': raise SystemExit(main())
