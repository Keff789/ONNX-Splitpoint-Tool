import copy
import json
import time
from pathlib import Path
import pytest
from onnx_splitpoint_tool import generic_energy_reporting as reporting
from onnx_splitpoint_tool.runners import generic_energy as energy, task_completion
from tests.test_generic_energy_contract import configured
from tests.test_v267_energy_aggregate_import import _aggregate


def capture(tmp_path,monkeypatch,capsys):
    module,c,logits,att,path=configured(tmp_path,monkeypatch)
    real=task_completion.measure_split_completion
    monkeypatch.setattr(task_completion,'measure_split_completion',lambda *a,**kw:real(*a,**kw,clock=lambda:time.perf_counter_ns()*3000))
    result_dir=tmp_path/'runtime';result_dir.mkdir()
    artifacts={k:Path(v['path']) for k,v in c['expected_runtime_artifacts'].items()}
    module._run_generic_split_completion(c,result_dir,lambda:{'cut':logits},lambda x:x,
        lambda x:{'logits':x['cut']},artifacts,clone_payload=None,repetition_index=1)
    stdout=capsys.readouterr().out
    result=result_dir/'generic_energy_results.json';row=json.loads(result.read_text())['repetitions'][0]
    output=tmp_path/'capture';directory=output/'run_000';directory.mkdir(parents=True)
    log=directory/'workload_stdout.log';log.write_text(stdout)
    start=time.time_ns()-1_000_000
    key=json.dumps(['generic_completed:deepx_to_trt','fixture','b001','fixture'],separators=(',',':'));budget=tmp_path/'campaign.json'
    budget.write_text(json.dumps({'campaign':True,'max_transport_failures':20,'sources':{'fixture-source':{'stop_reason':''}},'tasks':{key:{'source_id':'fixture-source','limits':{'max_chains':9,'max_retries':2,'max_transport_failures':20},'chains':[{'logical_repeat':'repeat:0','finished':True,'valid':True,'source_completion_verified':True,'run_directory':str(directory)}]}}}))
    command=['python','energy_measurement_cli.py','measure','--setup-id','fixture','--run-id','energy-fixture','--runs','1','--exact-run-count','--out',str(output),
        '--preflight-expected-command-contract-sha256',att['command_contract_sha256'], '--campaign-budget-file',str(budget),'--campaign-row-id',key,'--campaign-repeats','3','--campaign-max-retries','2','--campaign-max-transport-failures','20','--task-logical-repeat','repeat:0']
    record={'run_index':0,'postprocess_status':'ok','final_energy_gate_status':'pass',
        'scientific_primary_energy_status':'available','scientific_primary_energy_j':1200.,'scientific_primary_active_duration_s':61.,
        'energy_configured_workload_duration_s':60.,'energy_work_units_used':row['completed_work_units'],
        'runtime_work_unit_evidence':{'exact':True,'count':row['completed_work_units']},
        'workload_timing':{'status':'ok','rc':0,'start_ns':start,'end_ns':start+61_000_000_000},
        'preflight_evidence':{'ok':True,'nonce':att['nonce']},
        'collector_and_workload_log_diagnostics':{'workload_stdout':{'available':True,'bytes':log.stat().st_size,'sha256':energy.file_hash(log)}},
        'energy_efficiency_claim_eligible':True}
    aggregate=_aggregate(output,requested_runs=1,setup_id='fixture',run_id='energy-fixture')
    aggregate.update(runs=[record],preflight_requested=True,preflight_expected_command_contract_sha256=att['command_contract_sha256'],preflight_verified_run_count=1,energy_efficiency_claim_eligible=True)
    raw=output/'energy_aggregate.json';raw.write_text(json.dumps(aggregate))
    descriptor={'repetition_index':1,'expected_campaign':{'campaign_budget_file':str(budget),'campaign_row_id':key,'campaign_source_id':'fixture-source'},'command':command,'command_sha256':energy.canonical_hash(command),'output_dir':str(output),'raw_aggregate_path':str(raw),'raw_aggregate_sha256':energy.file_hash(raw),
        'process_receipt':{'argv':command,'command_sha256':energy.canonical_hash(command),'returncode':0,'started_ns':start,'finished_ns':time.time_ns(),'aggregate_absent_before_execution':True},
        'runtime_result_path':str(result),'runtime_remote_result_path':str(result)}
    return descriptor,c,record,aggregate,raw


def test_real_collector_import_and_generic_marker_join(tmp_path,monkeypatch,capsys):
    d,c,record,a,raw=capture(tmp_path,monkeypatch,capsys);before=raw.read_bytes()
    r=reporting.import_generic_energy_capture(d,c,expected_setup_id='fixture',expected_campaign=d['expected_campaign'])
    assert r['normal_import']['energy_aggregate_verified'] is True
    assert r['completed_tasks']==record['energy_work_units_used']
    assert r['energy_per_completed_task_j']==1200/r['completed_tasks']
    assert r['completed_tasks_per_j']==r['completed_tasks']/1200
    assert r['raw_run_index']==0 and r['primary_energy_claim_gate'] is True
    assert raw.read_bytes()==before


@pytest.mark.parametrize('change',['nonce','count','window','command','hash','diagnostic','shadow_diagnostic','source','mode'])
def test_capture_admission_negative_and_shadow_role(tmp_path,monkeypatch,capsys,change):
    d,c,r,a,raw=capture(tmp_path,monkeypatch,capsys)
    if change=='nonce':r['preflight_evidence']['nonce']='another-real-nonce-000000'
    if change=='count':r['energy_work_units_used']+=1
    if change=='window':r['scientific_primary_active_duration_s']=59
    if change=='command':d['process_receipt']['command_sha256']='a'*64
    if change=='hash':d['raw_aggregate_sha256']='a'*64
    if change=='diagnostic':a['diagnostic_only']=True
    if change=='shadow_diagnostic':a['energy_window_method_ab']['diagnostic_only']=True
    if change=='source':c['energy_capture']['contract']['source_files'][0]['sha256']='a'*64
    if change=='mode':c['energy_capture']['contract']['completion_runtime_mode']='strict'
    if change not in {'hash','command','source','mode'}:
        raw.write_text(json.dumps(a));d['raw_aggregate_sha256']=energy.file_hash(raw)
    if change in {'diagnostic','shadow_diagnostic'}:
        got=reporting.import_generic_energy_capture(d,c,expected_setup_id='fixture',expected_campaign=d['expected_campaign'])
        assert got['primary_energy_claim_gate'] is (change=='shadow_diagnostic')
    else:
        with pytest.raises(ValueError):reporting.import_generic_energy_capture(d,c,expected_setup_id='fixture',expected_campaign=d['expected_campaign'])
