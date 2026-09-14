#!/usr/bin/env python3
"""H1b: synthetic stages through the installed run/cleanup/closure lifecycle.

Only local scratch files are created. The terminal hashes and both verifiers
are production implementations; cache calls/read bytes are observed, not faked.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _worker(mode: str, destination: Path) -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(1, str(ROOT / 'tests'))
    from _v27930_terminal_lifecycle_fixture import TerminalLifecycleRunner, options_for, snapshot
    from onnx_splitpoint_tool import __version__, v60m_policy as policy
    from onnx_splitpoint_tool.workflow import artifacts, runner as runner_module
    assert Path(runner_module.__file__).resolve() == ROOT / 'onnx_splitpoint_tool/workflow/runner.py'
    os.environ['ONNX_SPLITPOINT_INTEGRITY_MODE'] = mode
    destination.mkdir(parents=True, exist_ok=True)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='onnx-terminal-synthetic-') as temp:
        base=Path(temp)
        cache=base/'synthetic_hash_cache.json'
        cache.write_text(json.dumps({'synthetic_padding': 'x' * 8192}))
        os.environ['ONNX_SPLITPOINT_HASH_CACHE']=str(cache)
        for damage in (False, True):
            events=[]; phases=[]; captured={}; terminal_calls={name:0 for name in ('_load_cache','_atomic_json','cached_sha256')}
            counts={'index_writes':0,'closure_report_writes':0,'hash_calls_by_phase':{},'bytes_read_by_phase':{}}
            r=TerminalLifecycleRunner(options_for(base/('negative' if damage else 'success')),log=events.append,progress=lambda d,t,l:phases.append(l))
            real_finalize=r._finalize_artifact_index
            real_verify=r._verify_terminal_artifact_index
            real_hash=runner_module.sha256_file_uncached
            real_write=runner_module.atomic_write_json
            verifier_phases=[]
            def verify(**kwargs):
                verifier_phases.append(kwargs['expected_closure_status'])
                if damage and kwargs['expected_closure_status']=='pass':
                    report=r.run_dir/'reports/synthetic_measurements.json'
                    report.write_bytes(report.read_bytes()+b' ')
                return real_verify(**kwargs)
            r._verify_terminal_artifact_index=verify
            def terminal(*,status):
                stat=cache.stat(); before=hashlib.sha256(cache.read_bytes()).hexdigest(); environment=dict(os.environ)
                def observed_hash(path,*args,on_chunk=None,**kwargs):
                    phase=getattr(r,'_terminal_hash_state',{}).get('phase','alias_receipt')
                    counts['hash_calls_by_phase'][phase]=counts['hash_calls_by_phase'].get(phase,0)+1
                    def chunk(n):
                        counts['bytes_read_by_phase'][phase]=counts['bytes_read_by_phase'].get(phase,0)+n
                        if on_chunk is not None:on_chunk(n)
                    return real_hash(path,*args,on_chunk=chunk,**kwargs)
                def write(path,payload):
                    if Path(path)==r.artifact_index_path:counts['index_writes']+=1
                    if Path(path).name=='artifact_index_closure.json':counts['closure_report_writes']+=1
                    return real_write(path,payload)
                with ExitStack() as stack:
                    for name in terminal_calls:
                        original=getattr(policy,name)
                        def observe(*args,__name=name,__original=original,**kwargs):
                            terminal_calls[__name]+=1
                            return __original(*args,**kwargs)
                        stack.enter_context(patch.object(policy,name,observe))
                    stack.enter_context(patch.object(runner_module,'sha256_file_uncached',observed_hash))
                    stack.enter_context(patch.object(runner_module,'atomic_write_json',write))
                    try:real_finalize(status=status)
                    finally:
                        captured['cache_unchanged']=(before==hashlib.sha256(cache.read_bytes()).hexdigest() and (stat.st_size,stat.st_mtime_ns,stat.st_ino)==(cache.stat().st_size,cache.stat().st_mtime_ns,cache.stat().st_ino))
                        captured['environment_unchanged']=environment==dict(os.environ)
            r._finalize_artifact_index=terminal
            before_threads={t.ident for t in threading.enumerate()}
            error='';start=time.perf_counter()
            try:
                result=r.run()
                assert not damage
                assert result.status=='failed'
            except RuntimeError as exc:
                if not damage:raise
                error=str(exc)
                assert 'artifact_index_terminal_commit_verify_failed' in error
            assert r._run_lock is None
            assert verifier_phases==['pending','pass']
            assert all(value==0 for value in terminal_calls.values())
            assert captured=={'cache_unchanged':True,'environment_unchanged':True}
            assert not list(base.glob('synthetic_hash_cache.json.tmp-*'))
            closure=json.loads((r.run_dir/'reports/artifact_index_closure.json').read_text())
            index=json.loads(r.artifact_index_path.read_text())
            assert closure['workflow_status']=='failed'
            assert closure['status']==index['terminal_closure']['status']==('fail' if damage else 'pass')
            assert len(index['artifacts'])==len({x['path'] for x in index['artifacts']})
            finished=[e for e in events if '[workflow] finished ' in e]
            assert len(finished)==(0 if damage else 1)
            assert any('verify_pending_index' in p for p in phases) and any('verify_pass_index' in p for p in phases)
            sealed=snapshot(r.run_dir)
            r._emit_log('[workflow] observer after returned runner')
            r.request_cancel();r.request_detach()
            assert snapshot(r.run_dir)==sealed
            assert not [t.name for t in threading.enumerate() if t.ident not in before_threads]
            if not damage:assert (counts['index_writes'],counts['closure_report_writes'])==(2,3)
            cases.append({'synthetic':True,'case':'pass_mutation_negative' if damage else 'failed_measurement_valid_closure',
                'closure_status':closure['status'],'workflow_status':closure['workflow_status'],'terminal_cache_calls':terminal_calls,
                'verifier_phases':verifier_phases,'finished_events':len(finished),'run_lock_released':True,'no_late_run_writes':True,
                'no_surviving_worker':True,'error':error,**captured,**counts,'elapsed_s':time.perf_counter()-start})
    report={'ok':True,'synthetic':True,'version':__version__,'source_root':str(ROOT),'mode':mode,'cases':cases}
    (destination/f'terminal_closure_smoke_{mode}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'ok':True,'mode':mode,'case_count':len(cases)}))
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path.home()/'Downloads')
    parser.add_argument('--worker',choices=('fast','strict'),help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:return _worker(args.worker,args.output_dir.resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    destination=Path(tempfile.mkdtemp(prefix='terminal_closure_v27930_',dir=args.output_dir))
    modes=[]
    for mode in ('fast','strict'):
        command=[sys.executable,'-I','-B',str(Path(__file__).resolve()),'--worker',mode,'--output-dir',str(destination)]
        result=subprocess.run(command,capture_output=True,text=True,timeout=180)
        (destination/f'{mode}_console.log').write_text(result.stdout+result.stderr)
        modes.append({'mode':mode,'returncode':result.returncode})
        print(result.stdout,end='')
        if result.returncode:
            print(result.stderr,file=sys.stderr)
            print('TERMINAL_SMOKE_REPORT='+str(destination))
            return result.returncode
    (destination/'summary.json').write_text(json.dumps({'ok':True,'synthetic':True,'modes':modes},indent=2)+'\n')
    print('TERMINAL_SMOKE_STATUS=PASS')
    print('TERMINAL_SMOKE_REPORT='+str(destination))
    return 0


if __name__=='__main__':raise SystemExit(main())
