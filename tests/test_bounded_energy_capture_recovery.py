"""Recover real temporary locks using the external energy parent's old layout."""
from datetime import datetime, timezone
import hashlib
import json
import shutil

import pytest

from test_physical_resource_recovery import recovery_case, _write, _fences, _retry_evidence
from onnx_splitpoint_tool.remote import resource_recovery


@pytest.fixture
def bounded_case(recovery_case):
    case = recovery_case
    journal = case.run_dir / 'remote-process-leases'
    shutil.move(str(case.journal), journal)
    case.journal = journal
    case.reply_path = journal / case.reply_path.name
    case.request_path = journal / case.request_path.name
    case.config_path = journal / case.config_path.name
    stamp = datetime.now(timezone.utc).isoformat()
    plan = case.run_dir / 'energy_parent_plan.json'
    _write(plan, {'schema':'yolov7-full-fix/bounded-energy-parent-v1','run_dir':str(case.run_dir)})
    _write(case.run_dir/'PREPARED.json', {'ok':True,'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()})
    _write(case.run_dir/'production-start.json', {'started_at':stamp,'run_id':case.run_dir.name,
        'session_id':case.kwargs['session_id'],'parent_pid':99999993})
    _write(case.run_dir/'production-result.json', {'finished_at':stamp,'returncode':130})
    _write(case.run_dir/'cleanup-status.json', {'at':stamp,'owned_remote_active':0,
        'owned_local_active':0,'foreign_processes_stopped':False,'cleanup_proven':False})
    case.kwargs.pop('operator_evidence')
    case.kwargs.update(bounded_energy_attempt_dir=case.run_dir,retry_authorization=_retry_evidence(case))
    # The external parent never had a workflow manifest. Do not invent one.
    (case.run_dir/'run_manifest.json').unlink()
    return case


def test_bounded_parent_recovery_preserves_terminal_evidence_and_budget(bounded_case):
    case = bounded_case
    paths = [case.run_dir/name for name in ('production-start.json','production-result.json',
        'cleanup-status.json','energy_task_budget.json','energy_parent_plan.json','PREPARED.json')]
    before = {p:p.read_bytes() for p in paths}
    assert resource_recovery.recover_capture_resources(**case.kwargs)['status']=='released'
    assert all(p.read_bytes()==data for p,data in before.items())
    assert json.loads(case.reply_path.read_text())['state']=='STOP'
    assert all(not lock.quarantine_path.exists() for lock in case.locks.values())
    assert resource_recovery.recover_capture_resources(**case.kwargs)['status']=='already_released'


@pytest.mark.parametrize('fault',['missing_result','live_cleanup','foreign_cleanup','session','pid','plan','time','symlink'])
def test_bounded_parent_refuses_unbound_recovery(bounded_case,fault):
    case = bounded_case
    if fault=='missing_result': (case.run_dir/'production-result.json').unlink()
    elif fault=='plan':
        _write(case.run_dir/'energy_parent_plan.json',{'run_dir':'/foreign'})
    elif fault=='symlink':
        p=case.run_dir/'production-result.json'; q=case.run_dir/'result-copy.json'
        p.rename(q);p.symlink_to(q)
    else:
        name='cleanup-status.json' if fault in ('live_cleanup','foreign_cleanup') else 'production-start.json'
        p=case.run_dir/name; data=json.loads(p.read_text())
        if fault=='live_cleanup':data['owned_remote_active']=1
        elif fault=='foreign_cleanup':data['foreign_processes_stopped']=True
        elif fault=='session':data['session_id']='foreign'
        elif fault=='pid':data['parent_pid']=99999994
        elif fault=='time':data['started_at']='2099-01-01T00:00:00+00:00'
        _write(p,data)
    before=_fences(case)
    with pytest.raises(Exception):resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case)==before
