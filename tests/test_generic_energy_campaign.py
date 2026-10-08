"""Shared budget/dispatcher provenance regressions, exclusively local fixtures."""
import copy
import json
from pathlib import Path
import sys
import pytest
from onnx_splitpoint_tool.workflow import generic_completed_task as workflow
from onnx_splitpoint_tool.runners import generic_energy as energy
from onnx_splitpoint_tool.generic_energy_reporting import collect_generic_energy_case
from tests.test_generic_energy_generated_processes import generated_case
from tests.test_generic_energy_contract import configured


def test_three_stable_logical_repeats_use_shared_case_budget(generated_case):
    plan,_,case=generated_case
    keys=[];logical=[]
    for call in plan['calls']:
        a=call['argv'];get=lambda k:a[a.index(k)+1]
        keys.append(get('--campaign-row-id'));logical.append(get('--task-logical-repeat'))
        assert get('--campaign-budget-file')==plan['campaign_budget_file']
        assert get('--campaign-repeats')=='3' and get('--campaign-max-retries')=='2'
        assert get('--campaign-max-transport-failures')=='20'
        assert get('--invalid-repeat-max-retries')=='0' and '--task-budget-file' not in a
    assert logical==['repeat:0','repeat:1','repeat:2'] and len(set(keys))==1
    assert json.loads(keys[0])==['generic_completed:deepx_to_trt','fixture','b001','orin_nx_deepx_m1_01']
    assert workflow.check_energy_campaign(plan)['begun_chains']==3


@pytest.mark.parametrize('change',['source_stop','task_stop','limits','source','global_limits','chain_cap','already_valid'])
def test_shared_source_stop_and_old_chains_block_before_transport(generated_case,tmp_path,change):
    plan,_,_=copy.deepcopy(generated_case)
    p=tmp_path/'budget.json';state=json.loads(Path(plan['campaign_budget_file']).read_text())
    task=state['tasks'][plan['campaign_row_id']]
    if change=='source_stop':state['sources']['fixture-source']['stop_reason']='campaign_source_completion_unresolved'
    if change=='task_stop':task['stop_reason']='task_transport_failure_limit'
    if change=='limits':task['limits']['max_chains']=99
    if change=='source':task['source_id']='another-source'
    if change=='global_limits':state['max_transport_failures']=999
    if change=='chain_cap':task['chains']*=3
    p.write_text(json.dumps(state));plan['campaign_budget_file']=str(p);before=p.read_bytes()
    with pytest.raises((ValueError,RuntimeError)):
        workflow.check_energy_campaign(plan,logical_repeat='repeat:0' if change=='already_valid' else None)
    assert p.read_bytes()==before


@pytest.mark.parametrize('change',['python','threads','gpu'])
def test_prepare_rejects_changed_dispatcher_vs_observed_performance(generated_case,tmp_path,change):
    plan,_,case=copy.deepcopy(generated_case)
    if change=='python':case['remote_python_executable']='/unmeasured/python'
    if change=='threads':case['remote_environment']['OMP_NUM_THREADS']='97'
    if change=='gpu':case['remote_environment']['CUDA_VISIBLE_DEVICES']='5'
    with pytest.raises(ValueError,match='Python/environment'):
        workflow.prepare_energy_case(case,performance_result=Path(plan['performance_result']),output=tmp_path/'new',
            collector_python=sys.executable,collector_script=Path(plan['collector_script']),
            hardware_setups_file=Path(plan['hardware_setups_file']),remote_command_prefix=[],
            campaign_budget_file=Path(plan['campaign_budget_file']))


@pytest.mark.parametrize('change',['threads','gpu','loader','hailo_sites','affinity','python','venv'])
def test_preflight_compares_actual_execution_to_performance(tmp_path,monkeypatch,change):
    _,config,_,_,_=configured(tmp_path,monkeypatch)
    if change=='threads':monkeypatch.setenv('OMP_NUM_THREADS','97')
    if change=='gpu':monkeypatch.setenv('CUDA_VISIBLE_DEVICES','5')
    if change=='loader':monkeypatch.setenv('LD_LIBRARY_PATH','/another/library')
    if change=='hailo_sites':monkeypatch.setenv('SPLITPOINT_EXTRA_SITES','/unmeasured/hailo')
    if change=='affinity':monkeypatch.setattr(energy.os,'sched_getaffinity',lambda pid:{9999})
    if change=='python':monkeypatch.setattr(energy.sys,'version','unmeasured interpreter')
    if change=='venv':monkeypatch.setattr(energy.sys,'prefix','/another/venv')
    with pytest.raises(ValueError,match='Python/environment'):
        energy.validate_config(config,check_files=True)


def test_unrelated_secret_environment_not_exported(tmp_path,monkeypatch):
    monkeypatch.setenv('UNRELATED_ACCESS_TOKEN','must-never-be-published')
    monkeypatch.setenv('DX_API_TOKEN','must-never-be-published')
    identity=energy.runtime_execution_identity({})
    assert 'UNRELATED_ACCESS_TOKEN' not in identity['environment']
    assert 'DX_API_TOKEN' not in identity['environment']


def test_descriptor_cannot_switch_to_unplanned_capture_command(generated_case):
    plan,ds,case=copy.deepcopy(generated_case)
    ds[0]['command'].append('--another-command')
    ds[0]['command_sha256']=energy.canonical_hash(ds[0]['command'])
    ds[0]['process_receipt'].update(argv=ds[0]['command'],command_sha256=ds[0]['command_sha256'])
    with pytest.raises(ValueError,match='prepared command'):collect_generic_energy_case(plan,ds,case)


@pytest.mark.parametrize('change',['unfinished','invalid','unclosed','directory','logical','source'])
def test_normal_report_requires_actual_accepted_shared_chain(generated_case,tmp_path,change):
    plan,ds,case=copy.deepcopy(generated_case)
    p=tmp_path/'budget.json';state=json.loads(Path(plan['campaign_budget_file']).read_text());task=state['tasks'][plan['campaign_row_id']];chain=task['chains'][0]
    if change=='unfinished':chain['finished']=False
    if change=='invalid':chain['valid']=False
    if change=='unclosed':chain['source_completion_verified']=False
    if change=='directory':chain['run_directory']+='/other'
    if change=='logical':chain['logical_repeat']='repeat:other'
    if change=='source':task['source_id']='other-source'
    p.write_text(json.dumps(state));plan['campaign_budget_file']=str(p)
    for d,c in zip(ds,plan['calls']):
        for a in [d['command'],c['argv']]:a[a.index('--campaign-budget-file')+1]=str(p)
        c['command_sha256']=energy.canonical_hash(c['argv']);d['command_sha256']=energy.canonical_hash(d['command']);d['process_receipt'].update(argv=d['command'],command_sha256=d['command_sha256'])
    with pytest.raises(ValueError,match='campaign|chain'):collect_generic_energy_case(plan,ds,case)
