"""Clean parent handoff preserves the old authorization and measurement budget."""
from datetime import datetime, timezone
import copy
import hashlib
import json
import os

import pytest

from test_yolo_authorized_energy_budget import case, install, write, SOURCE, ROW, LIFECYCLE
from onnx_splitpoint_tool.energy.task_budget import continue_authorized_source_reentry


@pytest.fixture
def clean(case):
    install(case)
    old = dict(case.parent, pid=99999990, start_ticks='104')
    state=json.loads(case.path.read_text())
    state['sources'][SOURCE]['authorized_reentry']['new_parent']=old
    # Existing completed acquisitions require their actual protocol-end logs.
    for task in state['tasks'].values():
        for chain in task['chains']:
            if chain.get('finished'):
                from pathlib import Path
                root=Path(chain['run_directory']);root.mkdir(parents=True,exist_ok=True)
                (root/'collector_stdout.log').write_text(LIFECYCLE)
    write(case.path,state)
    previous=case.run/'attempts'/old['session_id'];previous.mkdir(parents=True)
    stamp=datetime.now(timezone.utc).isoformat()
    plan=case.run/'energy_parent_plan.json'
    write(plan,{'schema':'yolov7-full-fix/bounded-energy-parent-v1','run_dir':str(case.run)})
    write(case.run/'PREPARED.json',{'ok':True,'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest()})
    write(previous/'production-start.json',{'run_id':case.run.name,'session_id':old['session_id'],
        'parent_pid':old['pid'],'started_at':stamp})
    write(previous/'production-result.json',{'returncode':1,'finished_at':stamp})
    write(previous/'cleanup-status.json',{'at':stamp,'cleanup_proven':True,'errors':[],
        'owned_remote_active':0,'owned_local_active':0,'foreign_processes_stopped':False})
    directory=previous/'remote-process-leases'
    write(directory/'controller.resources.json',{'run_id':case.run.name,'run_dir':str(case.run),
        'session_id':old['session_id'],'parent_pid':old['pid'],'parent_start_ticks':old['start_ticks']})
    request=dict(case.request,session_id=old['session_id'])
    write(directory/'capture-a.resource-request.json',request)
    write(directory/'capture-a.resource-reply.json',{**request,'state':'RELEASED'})
    parent=dict(case.parent,session_id='clean-next')
    return case,previous,parent


def test_clean_terminal_parent_handoff_preserves_all_measurements_and_limits(clean):
    case,previous,parent=clean
    before=json.loads(case.path.read_text())
    grant=continue_authorized_source_reentry(case.path,previous_attempt_dir=previous,new_parent=parent)
    after=json.loads(case.path.read_text())
    assert grant['new_parent']==parent
    assert grant['parent_history'][0]['parent']==before['sources'][SOURCE]['authorized_reentry']['new_parent']
    assert after['tasks']==before['tasks'] and after['max_transport_failures']==before['max_transport_failures']
    for k in before['sources'][SOURCE]:
        if k!='authorized_reentry':assert after['sources'][SOURCE][k]==before['sources'][SOURCE][k]
    old=copy.deepcopy(before['sources'][SOURCE]['authorized_reentry'])
    new=copy.deepcopy(after['sources'][SOURCE]['authorized_reentry'])
    old.pop('new_parent');new.pop('new_parent');new.pop('parent_history')
    assert old==new
    with pytest.raises(ValueError,match='previous continuation grant'):
        continue_authorized_source_reentry(case.path,previous_attempt_dir=previous,new_parent=parent)


@pytest.mark.parametrize('fault',['unclean','foreign_pid','active','unresolved','source_stop','task_stop','budget','foreign_session','release'])
def test_clean_handoff_refuses_unproven_or_changed_state(clean,fault):
    case,previous,parent=clean
    root=json.loads(case.path.read_text());grant=root['sources'][SOURCE]['authorized_reentry']
    if fault=='unclean':
        p=previous/'cleanup-status.json';d=json.loads(p.read_text());d['cleanup_proven']=False;write(p,d)
    elif fault=='foreign_pid':parent['pid']=99999989
    elif fault=='active':
        p=previous/'production-start.json';d=json.loads(p.read_text());d['parent_pid']=os.getpid();write(p,d)
        p=previous/'remote-process-leases/controller.resources.json';d=json.loads(p.read_text());d.update(parent_pid=os.getpid(),parent_start_ticks=case.parent['start_ticks']);write(p,d)
    elif fault=='unresolved':root['tasks'][ROW]['chains'][0]['source_completion_verified']=False
    elif fault=='source_stop':grant['source_stop_reason']='campaign_source_completion_unresolved'
    elif fault=='task_stop':grant['task_stop_reasons'][ROW]='task_chain_limit'
    elif fault=='budget':root['tasks'][ROW]['limits']['max_chains']=10
    elif fault=='foreign_session':grant['new_parent']['session_id']='foreign'
    elif fault=='release':
        p=previous/'remote-process-leases/capture-a.resource-reply.json';d=json.loads(p.read_text());d['state']='STOP';write(p,d)
    write(case.path,root);before=case.path.read_bytes()
    with pytest.raises((ValueError,RuntimeError)):
        continue_authorized_source_reentry(case.path,previous_attempt_dir=previous,new_parent=parent)
    assert case.path.read_bytes()==before


@pytest.mark.parametrize('receipt_ok',[True,False])
def test_clean_parent_before_first_operation_needs_unchanged_handoff(clean,receipt_ok):
    case,previous,parent=clean
    for path in (previous/'remote-process-leases').glob('*.resource-*.json'):path.unlink()
    before=json.loads(case.path.read_text())
    grant=before['sources'][SOURCE]['authorized_reentry']
    if receipt_ok:write(previous/'source-budget-handoff.json',grant)
    if not receipt_ok:
        with pytest.raises(ValueError,match='handoff receipt'):
            continue_authorized_source_reentry(case.path,previous_attempt_dir=previous,new_parent=parent)
        assert json.loads(case.path.read_text())==before
    else:
        result=continue_authorized_source_reentry(case.path,previous_attempt_dir=previous,new_parent=parent)
        assert result['new_parent']==parent
        assert json.loads(case.path.read_text())['tasks']==before['tasks']
