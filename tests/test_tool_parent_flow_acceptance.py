"""Normal Native/Generic completion-loop fixtures; no device dispatch.

Transferred unchanged from prior successful acceptance, using the installed
product imports and this checkout's controller root.
"""
from pathlib import Path
from types import SimpleNamespace
import pytest
from onnx_splitpoint_tool.workflow import energy_parent_flow as flow, energy_resume, generic_energy_parent
W = Path(__file__).resolve().parents[1]

def test_same_product_loop_recovers_repeated_timeouts_and_imports_once_no_loss_hook(tmp_path,monkeypatch):
    run=tmp_path/'parent';run.mkdir();previous=run/'resume_attempts/old'
    item={'run':run,'previous':previous,'state':'RESUME_REQUIRED','config':{'schema':flow.GENERIC_SCHEMA},
          'context':{'report_root':tmp_path,'controller_root':W},'importer':['bound_import'],'group':{'regime':'generic_completed','setup':'setup',
              'resume_command':['normal_parent','resume','--run-dir','{parent}','--slot-handoff','{handoff}']}}
    monkeypatch.setattr(flow,'describe_parent',lambda r:item)
    commands=[];results=iter([1,1,0]);events=[]
    monkeypatch.setattr(flow,'_run_parent_command',lambda i,c:(commands.append(c),SimpleNamespace(returncode=next(results)))[1])
    def timeout(i,**kw):
        number=len(events)+1;failed=run/'resume_attempts'/('failed'+str(number));events.append(('timeout',failed));return failed,tmp_path/('auth'+str(number))
    monkeypatch.setattr(flow,'_timeout_recovery_authorization',timeout)
    def recover(i,auth,expected_previous=None):events.append(('recover',expected_previous));item['previous']=expected_previous;return item
    monkeypatch.setattr(flow,'_recover_parent',recover)
    monkeypatch.setattr(generic_energy_parent,'configure',lambda *a:None)
    monkeypatch.setattr(generic_energy_parent,'_sample_loss_retry_attempt',lambda *a,**kw:pytest.fail('timeout entered source-closed loss path'))
    from onnx_splitpoint_tool.energy import collector
    monkeypatch.setattr(collector,'_collector_reconnect_backoff',lambda seconds:events.append(('backoff',seconds))or{})
    imports=[];monkeypatch.setattr(flow,'_import_saved',lambda i:imports.append(i)or{'results':[{'normal_energy_eligible':True}]})
    assert flow.run_parent(run,resume=True)==0
    assert len(commands)==3 and len(imports)==1
    assert [x[0]for x in events]==['timeout','recover','backoff','timeout','recover','backoff']
    assert flow.read(tmp_path/'operations/token_saving_batch_20261009/parent_handoff.json')['decision']==(
        'exact standing-authorized receive-timeout recovery; old source completion remains unverified; normal resume')

def test_import_failure_never_restarts_capture_or_recovers(tmp_path,monkeypatch):
    item={'run':tmp_path,'previous':tmp_path,'state':'IMPORT_PENDING','config':{'schema':flow.GENERIC_SCHEMA},'context':{}}
    monkeypatch.setattr(flow,'describe_parent',lambda run:item)
    monkeypatch.setattr(flow,'_import_saved',lambda i:(_ for _ in ()).throw(RuntimeError('normal import failed')))
    monkeypatch.setattr(flow,'_run_parent_command',lambda *a:pytest.fail('capture after import failure'))
    monkeypatch.setattr(flow,'_timeout_recovery_authorization',lambda *a,**kw:pytest.fail('recovery after import failure'))
    with pytest.raises(RuntimeError,match='normal import failed'):flow.run_parent(tmp_path,import_only=True)

def test_cooperative_cancel_never_enters_recovery_hook(tmp_path,monkeypatch):
    item={'run':tmp_path,'previous':tmp_path,'state':'RESUME_REQUIRED','config':{'schema':flow.GENERIC_SCHEMA},
          'context':{'report_root':tmp_path},'importer':['bound_import'],'group':{'regime':'generic_completed','setup':'setup',
              'resume_command':['normal_parent','resume','--run-dir','{parent}','--slot-handoff','{handoff}']}}
    monkeypatch.setattr(flow,'describe_parent',lambda run:item)
    monkeypatch.setattr(flow,'_run_parent_command',lambda *a:(_ for _ in ()).throw(RuntimeError('normal energy parent cancelled')))
    monkeypatch.setattr(flow,'_timeout_recovery_authorization',lambda *a,**kw:pytest.fail('cancel triggered recovery'))
    with pytest.raises(RuntimeError,match='cancelled'):flow.run_parent(tmp_path,resume=True)

def test_native_product_nonzero_uses_same_recovery_resume_before_old_terminal_stop(tmp_path,monkeypatch):
    run=tmp_path/'parent';run.mkdir()
    item={'run':run,'previous':run,'state':'RESUME_REQUIRED','config':{'schema':flow.NATIVE_SCHEMA},'importer':['bound_import'],
        'context':{'report_root':tmp_path,'controller_root':W},'group':{'regime':'native','setup':'setup',
            'resume_command':['bound_native','resume','--run-dir','{parent}','--previous-attempt','{previous_attempt}',
                              '--slot-handoff','{handoff}']}}
    monkeypatch.setattr(flow,'describe_parent',lambda run:item)
    results=iter([1,0]);commands=[];events=[]
    monkeypatch.setattr(flow,'_run_parent_command',lambda i,c:(commands.append(c),SimpleNamespace(returncode=next(results)))[1])
    monkeypatch.setattr(flow,'_timeout_recovery_authorization',lambda i,**kw:(run,tmp_path/'AUTHORIZATION.json'))
    monkeypatch.setattr(flow,'_recover_parent',lambda i,a,**kw:events.append('recover')or item)
    from onnx_splitpoint_tool.energy import collector
    monkeypatch.setattr(collector,'_collector_reconnect_backoff',lambda seconds:events.append(('backoff',seconds))or{})
    imports=[];monkeypatch.setattr(flow,'_import_saved',lambda i:imports.append(i)or{'results':[{}]*5})
    assert flow.run_parent(run,resume=True)==0
    assert len(commands)==2 and len(imports)==1 and events==['recover',('backoff',5.0)]
    assert flow.read(tmp_path/'operations/token_saving_batch_20261009/parent_handoff.json')['group']['regime']=='native'
    assert flow.read(tmp_path/'operations/token_saving_batch_20261009/parent_handoff.json')['decision']==(
        'exact standing-authorized receive-timeout recovery; old source completion remains unverified; normal resume')

def test_native_non_timeout_still_stops_without_generic_loss_hook(tmp_path,monkeypatch):
    run=tmp_path/'parent';run.mkdir();item={'run':run,'previous':run,'state':'RESUME_REQUIRED','config':{'schema':flow.NATIVE_SCHEMA},
        'importer':['bound_import'],'context':{'report_root':tmp_path},'group':{'regime':'native','setup':'setup',
            'resume_command':['bound_native','resume','--run-dir','{parent}','--slot-handoff','{handoff}']}}
    monkeypatch.setattr(flow,'describe_parent',lambda r:item)
    monkeypatch.setattr(flow,'_run_parent_command',lambda *a:SimpleNamespace(returncode=1))
    monkeypatch.setattr(flow,'_timeout_recovery_authorization',lambda *a,**kw:None)
    monkeypatch.setattr(flow,'_recover_parent',lambda *a,**kw:pytest.fail('untyped Native recovery'))
    with pytest.raises(RuntimeError,match='normal energy parent stopped'):flow.run_parent(run,resume=True)

def test_source_closed_sample_loss_resumes_normally_and_imports_once(tmp_path, monkeypatch):
    run = tmp_path / 'parent'
    run.mkdir()
    item = {'run': run, 'previous': run / 'resume_attempts/old', 'state': 'RESUME_REQUIRED',
            'config': {'schema': flow.GENERIC_SCHEMA}, 'importer': ['bound_import'],
            'context': {'report_root': tmp_path, 'controller_root': W},
            'group': {'regime': 'generic_completed', 'setup': 'setup',
                      'resume_command': ['normal_parent', 'resume', '--run-dir', '{parent}',
                                         '--previous-attempt', '{previous_attempt}',
                                         '--slot-handoff', '{handoff}']}}
    monkeypatch.setattr(flow, 'describe_parent', lambda r: item)
    commands, events, imports = [], [], []
    results = iter([1, 0])
    monkeypatch.setattr(flow, '_run_parent_command',
                        lambda i, c: (commands.append(c), SimpleNamespace(returncode=next(results)))[1])
    monkeypatch.setattr(flow, '_timeout_recovery_authorization', lambda *a, **kw: None)
    monkeypatch.setattr(flow, '_recover_parent', lambda *a, **kw: pytest.fail('closed loss entered physical recovery'))
    monkeypatch.setattr(generic_energy_parent, 'configure', lambda *a: None)
    closed = run / 'resume_attempts/source_closed_failed'
    monkeypatch.setattr(generic_energy_parent, '_sample_loss_retry_attempt',
                        lambda *a, **kw: events.append('source_closed_loss_admission') or closed)
    from onnx_splitpoint_tool.energy import collector
    monkeypatch.setattr(collector, '_collector_reconnect_backoff',
                        lambda seconds: events.append(('backoff', seconds)) or {})
    monkeypatch.setattr(flow, '_import_saved', lambda i: imports.append(i) or {'results': [{}]})
    assert flow.run_parent(run, resume=True) == 0
    assert len(commands) == 2 and len(imports) == 1
    assert commands[1][commands[1].index('--previous-attempt') + 1] == str(closed)
    assert events == ['source_closed_loss_admission', ('backoff', 5.0)]
    assert flow.read(tmp_path / 'operations/token_saving_batch_20261009/parent_handoff.json')['decision'] == (
        'existing source-closed pure-loss authority; normal finite-series resume')

