#!/usr/bin/env python3
"""Use the normal Full-workflow supervisor for this bounded diagnostic tree."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from smoke_common import read_json,write_json,clean_env
from deepx_full_workflow_smoke_worker_v27930 import supervised_run,_processes


def stop(stage):
    completed=stage/'results/execution.json'
    if completed.is_file():
        row=read_json(completed)
        if row.get('cleanup_complete') is True:return {'cleanup_confirmed':True,'reason':'owned_tree_already_reaped'}
    owner=stage/'supervisor_owner.json'
    if not owner.is_file():return {'cleanup_confirmed':False,'reason':'supervisor_owner_missing'}
    row=read_json(owner); pid=int(row['pid']); current=_processes()
    if current.get(pid,{}).get('start')!=row['start_time']:
        return {'cleanup_confirmed':False,'reason':'supervisor_identity_unverifiable_no_signal'}
    os.kill(pid,signal.SIGTERM)
    until=time.monotonic()+15
    while time.monotonic()<until:
        if completed.is_file() and read_json(completed).get('cleanup_complete') is True:
            return {'cleanup_confirmed':True,'reason':'owned_tree_reaped_after_stop'}
        time.sleep(.1)
    return {'cleanup_confirmed':False,'reason':'supervisor_cleanup_not_confirmed'}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--stop',action='store_true');p.add_argument('--timeout',type=float,default=180)
    a=p.parse_args(argv);stage=a.stage.resolve();(stage/'results').mkdir(exist_ok=True)
    if a.stop:
        result=stop(stage);write_json(stage/'stop_result.json',result);print(result)
        return 0 if result['cleanup_confirmed'] else 3
    if not 0<a.timeout<=180:raise ValueError('execution_budget_must_be_1_to_180_seconds')
    own=_processes()[os.getpid()]
    write_json(stage/'supervisor_owner.json',{'pid':os.getpid(),'start_time':own['start'],'stage':str(stage)})
    os.environ.update(clean_env(stage))
    result=supervised_run([sys.executable,'-I','-B',str(stage/'lib/remote_worker.py'),'--request',str(stage/'request.json')],cwd=stage,log_path=stage/'results/worker.log',timeout=a.timeout,grace=5)
    write_json(stage/'results/execution.json',result)
    return 124 if result['timed_out'] else (result['returncode'] or (0 if result['cleanup_complete'] else 3))

if __name__=='__main__':raise SystemExit(main())
