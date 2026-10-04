"""Explicit three-row H8 budget extension; no physical recovery or acquisition."""
import copy
import json
from pathlib import Path

import pytest

from test_yolo_authorized_energy_budget import case, budget, write, LIFECYCLE
from onnx_splitpoint_tool.energy.task_budget import (
    authorized_campaign_row_policy, campaign_budget_stop_reason,
    install_authorized_source_reentry,
)
from onnx_splitpoint_tool.remote.process_lease import ControllerResourceBroker, RemoteProcessLeaseJournal

OLD = {'max_retries': 2, 'max_chains': 9, 'max_transport_failures': 20}
NEW = {'max_retries': 5, 'max_chains': 18, 'max_transport_failures': 20}
ROWS = [json.dumps(['hailo8_to_trt',model,boundary,'orin_nx_hailo8_01'],separators=(',',':'))
        for model,boundary in [('yolo26s','b038'),('yolov7_paper','b044'),('yolov7_paper','b066')]]


@pytest.fixture
def h8(case):
    c=case
    oldsource=c.source; oldrow=c.row
    c.source='192.0.2.197'; c.row=ROWS[0]  # RFC 5737 software fixture.
    source=c.root['sources'].pop(oldsource)
    source.update(stop_reason='campaign_source_interrupted_chain_unresolved',transport_failures=0)
    c.root['sources'][c.source]=source
    task=c.root['tasks'].pop(oldrow); task['source_id']=c.source
    task['chains'][0].update(valid=True,transport_reasons=[])
    c.root['tasks']={c.row:task, ROWS[1]:{'source_id':c.source,'limits':OLD.copy(),
        'chains':[],'stop_reason':'campaign_source_interrupted_chain_unresolved'}}
    # A different, dead parent grant remains historical, even for read-only
    # policy lookup. It must never be relabelled to this new H8 session.
    c.root['sources'][oldsource]={'stop_reason':'','transport_failures':3,
        'authorized_reentry':{'new_parent':{'run_id':'old','session_id':'old','pid':99999997,'start_ticks':'1'},
                              'old_unfinished_capture':'unchanged'}}
    c.request['setup_id']='orin_nx_hailo8_01'
    recovery=c.reply['recovery']
    resources=['controller:capture','controller:nic','dut:192.0.2.104','source:'+c.source]
    original=next(iter(recovery['original_resources'].values()))
    recovery.update(resources=resources,original_resources={r:copy.deepcopy(original) for r in resources})
    for resource, item in recovery['original_resources'].items():
        item['lock_owner'] = {key: c.request[key] for key in ('run_id', 'session_id', 'operation_id')}
        item['lock_owner']['physical_resource'] = resource
        item['quarantine']['owner'] = {'resource_request': copy.deepcopy(c.request)}
    recovery['retry_authorization']['source']='source:'+c.source
    c.journal=RemoteProcessLeaseJournal(scope=c.journal.scope,directory=str(c.run/'new-h8-journal'))
    ControllerResourceBroker(c.journal,None,run_dir=c.run,setups={c.request['setup_id']:resources})
    c.env={**c.journal.environment(),'ONNX_SPLITPOINT_CONTROLLER_RESOURCES':'required'}
    write(c.path,c.root)
    return c


def install(c, changes=None):
    return install_authorized_source_reentry(c.path,capture_request=c.request,recovery_reply=c.reply,
        new_parent=c.parent,authorized_limits=changes if changes is not None else {r:NEW.copy() for r in ROWS})


def history(c):
    value=json.loads(c.path.read_text())
    for row,old in c.root['tasks'].items():
        assert value['tasks'][row]['chains'][:len(old['chains'])] == old['chains']
        assert value['tasks'][row]['stop_reason'] == old['stop_reason']
    assert value['sources']['192.0.2.176'] == c.root['sources']['192.0.2.176']
    assert value['sources'][c.source]['stop_reason'] == c.root['sources'][c.source]['stop_reason']
    assert value['sources'][c.source]['transport_failures'] == 0
    return value


def test_exact_three_limits_historized_and_b066_waits_for_normal_initializer(h8):
    grant=install(h8)
    assert install(h8)==grant
    value=history(h8)
    assert ROWS[2] not in value['tasks']
    assert grant['limit_changes'][ROWS[0]]['before']==OLD
    assert grant['limit_changes'][ROWS[1]]['before']==OLD
    assert grant['limit_changes'][ROWS[2]]['before'] is None
    for row in ROWS:
        policy=authorized_campaign_row_policy(h8.path,row,h8.env)
        assert policy['after']==NEW and policy['reconnect_backoff_s']==10
        with budget(h8,row=row,limits=NEW) as current:
            assert current.authorized_retry_limit()==5
            assert not current.is_set()
            if row==ROWS[0]:
                assert current.counts()['begun_chains']==2
                assert current.counts()['valid_logical_repeats']==1
                assert not current.allowed('repeat:0')
                assert current.busy_reason=='energy_repeat_already_valid:repeat:0'
                assert current.allowed('repeat:1')
            else:
                assert current.allowed('repeat:0')
    history(h8)


def test_global_policy_old_source_and_completed_h8_row_receive_no_override(h8):
    install(h8)
    for row in ['other',json.dumps(['hailo10h_to_trt','yolov7_paper','b044','orin_nx_hailo10_01'],separators=(',',':')),
                json.dumps(['hailo8_to_trt','yolo11l','b120','orin_nx_hailo8_01'],separators=(',',':'))]:
        assert authorized_campaign_row_policy(h8.path,row,{'ONNX_SPLITPOINT_CONTROLLER_RESOURCES':'not-required'}) is None
    history(h8)


@pytest.mark.parametrize('mutation', ['setup', 'capture', 'source', 'dut', 'owner_session', 'missing_original'])
def test_original_capture_identity_cannot_be_relabelled(h8, mutation):
    original = h8.reply['recovery']['original_resources']
    resource = next(key for key in original if key.startswith('source:'))
    item = original[resource]
    if mutation == 'setup': h8.request['setup_id'] = 'different-setup'
    elif mutation == 'capture': item['quarantine']['owner']['resource_request']['attempt_dir'] += '-other'
    elif mutation == 'source': item['lock_owner']['physical_resource'] = 'source:192.0.2.99'
    elif mutation == 'dut':
        resource = next(key for key in original if key.startswith('dut:'))
        original[resource]['lock_owner']['physical_resource'] = 'dut:192.0.2.99'
    elif mutation == 'owner_session': item['lock_owner']['session_id'] = 'foreign-session'
    else: item['quarantine'].pop('owner')
    before = h8.path.read_bytes()
    with pytest.raises(ValueError): install(h8)
    assert h8.path.read_bytes() == before


def test_six_total_attempts_per_repeat_include_old_unfinished_capture(h8):
    install(h8)
    with budget(h8,limits=NEW) as current:
        for i in range(5):
            run=h8.run/'new'/str(i);run.mkdir(parents=True)
            (run/'collector_stdout.log').write_text(LIFECYCLE)
            chain=current.reserve('repeat:1',run,True); assert chain
            current.collector_started(chain)
            current.finish(chain,{'ok':False},run)
        assert not current.allowed('repeat:1')
        assert len([c for c in current.data['chains'] if c['logical_repeat']=='repeat:1'])==6
        assert current.counts()['begun_chains']==7 # one old valid repeat0 plus six repeat1 attempts
        assert current.report({'ok':False},h8.run/'report')['error']=='task_logical_retry_limit'
    history(h8)


def test_eighteen_chain_limit_keeps_original_costs(h8):
    install(h8)
    with budget(h8,limits=NEW) as current:
        for i in range(16):
            run=h8.run/'new'/str(i); run.mkdir(parents=True)
            chain=current.reserve('preflight:'+str(i),run,True);assert chain
            current.finish(chain,{'ok':False},run)
        assert current.counts()['begun_chains']==18
        assert not current.allowed('repeat:2')
        assert current.report({'ok':False},h8.run/'report')['error']=='task_chain_limit'
    history(h8)


def test_new_unknown_end_blocks_all_three_and_cannot_be_cleared(h8):
    install(h8)
    with budget(h8,limits=NEW) as current:
        run=h8.run/'new'
        chain=current.reserve('repeat:1',run,True);current.collector_started(chain)
        current.finish(chain,{'ok':False},run)
        assert current.is_set()
    install(h8)
    for row in ROWS:
        with budget(h8,row=row,limits=NEW) as current:
            assert current.is_set() and not current.allowed('repeat:0')
            assert current.report({'ok':False},h8.run/('report'+str(ROWS.index(row))))['error']=='campaign_source_completion_unresolved'
    history(h8)


@pytest.mark.parametrize('mutation',['extra_row','missing_row','retry6','chains19','failures21','wrong_source',
    'b044_other_stop','b044_prior_attempt','changed_old_limit','b066_existing_work'])
def test_scoped_migration_rejects_unrelated_changes(h8,mutation):
    changes={r:NEW.copy() for r in ROWS}
    value=json.loads(h8.path.read_text())
    if mutation=='extra_row': changes['outside']=NEW.copy()
    elif mutation=='missing_row': changes.pop(ROWS[2])
    elif mutation=='retry6': changes[ROWS[0]]['max_retries']=6
    elif mutation=='chains19': changes[ROWS[0]]['max_chains']=19
    elif mutation=='failures21': changes[ROWS[0]]['max_transport_failures']=21
    elif mutation=='wrong_source': h8.reply['recovery']['retry_authorization']['source']='source:192.0.2.176'
    elif mutation=='b044_other_stop': value['tasks'][ROWS[1]]['stop_reason']='task_chain_limit'
    elif mutation=='b044_prior_attempt': value['tasks'][ROWS[1]]['chains']=[dict(value['tasks'][ROWS[0]]['chains'][0],run_directory='/another')]
    elif mutation=='changed_old_limit': value['tasks'][ROWS[0]]['limits']['max_retries']=3
    elif mutation=='b066_existing_work': value['tasks'][ROWS[2]]=copy.deepcopy(value['tasks'][ROWS[0]])
    write(h8.path,value);before=h8.path.read_bytes()
    with pytest.raises((ValueError,RuntimeError)): install(h8,changes)
    assert h8.path.read_bytes()==before


@pytest.mark.parametrize('mutation',['parent','limits','historical_stop'])
def test_read_policy_requires_exact_live_parent_and_grant(h8,mutation):
    install(h8)
    if mutation=='parent': h8.env['ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID']='different'
    else:
        value=json.loads(h8.path.read_text())
        if mutation=='limits': value['tasks'][ROWS[1]]['limits']['max_retries']=6
        else:value['tasks'][ROWS[1]]['stop_reason']=''
        write(h8.path,value)
    before=h8.path.read_bytes()
    with pytest.raises(ValueError): authorized_campaign_row_policy(h8.path,ROWS[0],h8.env)
    assert h8.path.read_bytes()==before


def test_new_task_stop_not_hidden_by_old_b044_propagation(h8):
    install(h8)
    with budget(h8,row=ROWS[1],limits=NEW) as current:
        current.stop('task_execution_interrupted')
        result=current.report({'ok':False},h8.run/'report')
        assert result['error']=='task_execution_interrupted'
        assert result['task_budget']['stop_reason']=='campaign_source_interrupted_chain_unresolved'
    value=history(h8)
    assert campaign_budget_stop_reason(value,ROWS[1],h8.env)=='task_execution_interrupted'
