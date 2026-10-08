"""Negative capture accounting, exact resume recovery and retained history."""
import copy
import json
from pathlib import Path
import shutil
import threading

import pytest

from onnx_splitpoint_tool.energy.recovery_accounting import reconcile_stopped_capture
from onnx_splitpoint_tool.energy.task_budget import EnergyTaskBudget, install_authorized_source_reentry
from onnx_splitpoint_tool.remote import process_lease, resource_recovery
from test_v283_campaign_energy_budget import campaign_rig, measure
from test_physical_resource_recovery import recovery_case, _write, _fences
from test_bounded_energy_capture_recovery import bounded_case
from test_yolo_authorized_energy_budget import case, install, budget, LIMITS


def test_capture_transport_stop_books_once_before_exception(tmp_path, monkeypatch):
    rig = campaign_rig(tmp_path, monkeypatch, ['closed'])
    from onnx_splitpoint_tool.workflow import run_control
    from onnx_splitpoint_tool.quality_service import ResourcePauseGate
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path', lambda:tmp_path/'locks/interlock')
    journal = process_lease.RemoteProcessLeaseJournal(
        scope=process_lease.RemoteProcessLeaseScope(tmp_path.name,'failed-resume'), directory=tmp_path/'journal')
    resources=['controller:capture','controller:nic','dut:fixture','source:fake_no_socket']
    broker = process_lease.ControllerResourceBroker(journal,ResourcePauseGate(),run_dir=tmp_path,setups={'fake':resources})
    rig.env.update(journal.environment(),ONNX_SPLITPOINT_CONTROLLER_RESOURCES='required')
    done=threading.Event(); errors=[]
    def pump():
        while not done.wait(.01):
            try: broker.tick()
            except BaseException as exc: errors.append(exc);return
    thread=threading.Thread(target=pump,daemon=True);thread.start()
    try:
        with pytest.raises(process_lease.RemoteProcessLeaseJournalError, match='completion_unresolved'):
            measure(rig, 'failed', run_count=1, exact_run_count=True, invalid_repeat_max_retries=0)
    finally:
        done.set();thread.join(2);broker.close()
    assert not errors and broker.stopped_sources==set(resources)
    assert all(run_control.EvaluationRunLock.for_resource(r,owner={}).quarantine_path.exists() for r in resources)
    p = tmp_path/'campaign.json'; root = json.loads(p.read_text())
    chain = root['tasks']['row-a']['chains'][0]
    assert chain['finished'] and not chain['valid'] and chain['transport_reasons']
    assert root['sources']['fake_no_socket']['transport_failures'] == 1
    assert root['sources']['fake_no_socket']['stop_reason'] == 'campaign_source_completion_unresolved'
    run = Path(chain['run_directory']); entry = json.loads((run/'energy_summary.json').read_text())
    assert entry['final_energy_gate_status'] == 'fail'
    with p.open('r+') as handle:
        b = EnergyTaskBudget(handle, root['tasks']['row-a']['limits'], campaign_row='row-a', source_id='fake_no_socket')
        b.finish(chain, entry, run)
        assert b.counts()['transport_failures'] == 1
        assert not b.allowed('repeat:1')
    assert json.loads(p.read_text())['tasks']['row-a']['chains'] == [chain]
    assert rig.counts()['collector'] == 1


@pytest.fixture
def resume_case(bounded_case):
    c = bounded_case
    parent = c.run_dir/'attempts'/c.request['session_id']; parent.mkdir(parents=True)
    for name in ('remote-process-leases','production-start.json','production-result.json','cleanup-status.json','reports'):
        shutil.move(str(c.run_dir/name), parent/name)
    c.journal = parent/'remote-process-leases'
    c.request_path = c.journal/c.request_path.name; c.reply_path = c.journal/c.reply_path.name
    c.config_path = c.journal/c.config_path.name
    c.request['attempt_dir'] = str(parent/'reports/energy/run_000')
    _write(c.request_path, c.request)
    for lock in c.locks.values():
        fence = json.loads(lock.quarantine_path.read_text())
        fence['owner']['resource_request'] = c.request
        _write(lock.quarantine_path, fence)
    capture = Path(c.request['attempt_dir'])
    (capture/'collector_stderr.log').write_text('receive socket error: silence timeout')
    _write(capture/'energy_summary.json', {'status':'collector_running', 'collector_started':True,
                                         'collector_pid':c.request['collector_pid']})
    row = json.dumps(['native_full_tensorrt','yolov7_paper','full','fixture-h10'],separators=(',',':'))
    chain = {k:c.request[k] for k in ('owner_pid','owner_start_ticks')}
    chain.update(logical_repeat='repeat:2', run_directory=str(capture), preflight_requested=True,
                 collector_started=True, finished=False)
    valid = dict(chain, logical_repeat='repeat:0', run_directory=str(parent/'valid'), finished=True,
                 valid=True, source_completion_verified=True, transport_reasons=[])
    root = dict(campaign=True,max_transport_failures=20,
        tasks={row:dict(limits=LIMITS.copy(),source_id='192.0.2.176',stop_reason='task_execution_interrupted',chains=[valid,chain])},
        sources={'192.0.2.176':dict(stop_reason='',transport_failures=0)})
    _write(c.run_dir/'energy_task_budget.json',root)
    c.kwargs['bounded_energy_attempt_dir'] = parent
    c.parent = parent; c.row = row; c.root = root
    return c


def reconcile(c):
    return reconcile_stopped_capture(c.run_dir/'energy_task_budget.json',parent_attempt=c.parent,
                                    operation_id=c.request['operation_id'])


def test_failed_resume_reconciliation_and_regular_recovery_are_idempotent(resume_case):
    c = resume_case
    originals = {p:p.read_bytes() for p in c.parent.rglob('*') if p.is_file()}
    fences = _fences(c)
    receipt = reconcile(c)
    assert reconcile(c) == receipt
    assert _fences(c) == fences
    assert all(p.read_bytes() == data for p,data in originals.items())
    value = json.loads((c.run_dir/'energy_task_budget.json').read_text())
    assert value['sources']['192.0.2.176']['transport_failures'] == 1
    assert len(value['tasks'][c.row]['chains']) == 2
    assert value['tasks'][c.row]['chains'][0] == c.root['tasks'][c.row]['chains'][0]
    assert resource_recovery.recover_capture_resources(**c.kwargs)['status'] == 'released'
    assert resource_recovery.recover_capture_resources(**c.kwargs)['status'] == 'already_released'
    assert reconcile(c) == receipt
    assert json.loads(c.reply_path.read_text())['state'] == 'STOP'


@pytest.mark.parametrize('fault',['parent','operation','owner','log','valid','config_session'])
def test_reconciliation_rejects_wrong_or_changed_evidence(resume_case,fault):
    c = resume_case; p = c.run_dir/'energy_task_budget.json'
    if fault=='parent': c.parent=c.run_dir
    elif fault=='operation': c.request['operation_id']='capture-'+'f'*32
    elif fault=='owner':
        v=json.loads(p.read_text());v['tasks'][c.row]['chains'][-1]['owner_pid']=99999990;_write(p,v)
    elif fault=='log': (Path(c.request['attempt_dir'])/'collector_stderr.log').write_text('no transport error')
    elif fault=='valid':
        v=json.loads(p.read_text());v['tasks'][c.row]['chains'][-1].update(finished=True,valid=True);_write(p,v)
    elif fault=='config_session':
        v=json.loads(c.config_path.read_text());v['session_id']='wrong';_write(c.config_path,v)
    before=p.read_bytes();fences=_fences(c)
    with pytest.raises(Exception):reconcile(c)
    assert p.read_bytes()==before and _fences(c)==fences


def test_new_stop_requires_new_exact_recovery_and_keeps_previous_grant(case):
    install(case)
    root=json.loads(case.path.read_text());previous=root['sources'][case.source]['authorized_reentry']
    previous['new_parent'].update(pid=99999994,start_ticks='104')
    previous['source_stop_reason']='campaign_source_completion_unresolved'
    previous['task_stop_reasons'][case.row]='campaign_source_completion_unresolved'
    request={**case.request,'session_id':previous['new_parent']['session_id'],
             'operation_id':'capture-'+'c'*32,'attempt_dir':str(case.run/'new-failed/run_000')}
    chain={**root['tasks'][case.row]['chains'][1], 'run_directory':request['attempt_dir'],
           'finished':True,'valid':False,'source_completion_verified':False,'transport_reasons':['transport_log_error']}
    root['tasks'][case.row]['chains'].append(chain)
    root['sources'][case.source]['transport_failures']=4
    _write(case.path,root)
    reply=copy.deepcopy(case.reply)
    for k in ('session_id','operation_id'): reply[k]=request[k];reply['recovery'][k]=request[k]
    for item in reply['recovery']['original_resources'].values(): item['lock_owner']={'pid':99999994}
    parent={**case.parent,'session_id':'second-authorized-parent'}
    before=case.path.read_bytes()
    with pytest.raises(ValueError):install_authorized_source_reentry(case.path,capture_request=case.request,recovery_reply=case.reply,new_parent=parent)
    assert case.path.read_bytes()==before
    grant=install_authorized_source_reentry(case.path,capture_request=request,recovery_reply=reply,new_parent=parent)
    assert grant['previous_grants']==[previous] and not grant['source_stop_reason']
    assert install_authorized_source_reentry(case.path,capture_request=request,recovery_reply=reply,new_parent=parent)==grant
    after=json.loads(case.path.read_text())
    assert after['tasks']==root['tasks'] and after['sources'][case.source]['transport_failures']==4
