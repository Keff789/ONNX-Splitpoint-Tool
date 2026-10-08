"""CPU fixture: the generated bundle dispatches real independent processes.

Inference is a deterministic callback and energy captures are synthetic fixtures;
these test outputs are never admitted to the Revision3 current-results corpus.
"""
import copy
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import pytest
from onnx_splitpoint_tool.gui.controller import _copy_runner_lib
from onnx_splitpoint_tool.split_export_runners import write_runner_onnxruntime
from onnx_splitpoint_tool.workflow.generic_completed_task import bind_runtime_source_closure, prepare_energy_case
from onnx_splitpoint_tool.runners.generic_energy import canonical_hash,file_hash
from onnx_splitpoint_tool.generic_energy_reporting import collect_generic_energy_case
from tests.test_v2912_generic_split_completion import _fixture
from tests.test_v267_energy_aggregate_import import _aggregate

WORKER='''import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
runner,config_path,output,index=sys.argv[1:]
spec=importlib.util.spec_from_file_location('actual_generated_generic',runner)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
c=json.loads(Path(config_path).read_text());out=Path(output);out.mkdir(parents=True,exist_ok=True)
if c.get('energy_capture'):
 from splitpoint_runners import generic_energy as energy,task_completion
 nonce='cpu-fixture-'+str(index)+'-0000000000000000'
 att=energy.build_preflight(c,nonce=nonce,expected_sha256=c['energy_capture']['contract']['command_contract_sha256'])
 attpath=out/'preflight.json';attpath.write_text(json.dumps(att))
 energy.verify_runtime_preflight(c,nonce=nonce,attestation_path=attpath,expected_sha256=att['command_contract_sha256'])
 c['_runtime_energy']={'energy_preflight_nonce':nonce,'energy_preflight_attestation':str(attpath),'energy_command_contract_sha256':att['command_contract_sha256']}
 real=task_completion.measure_split_completion
 task_completion.measure_split_completion=lambda *a,**kw:real(*a,**kw,clock=lambda:time.perf_counter_ns()*3000)
logits=np.load(c['functional_reference']['outputs']['logits']['path'])
artifacts={k:Path(v['path']) for k,v in c['expected_runtime_artifacts'].items()}
raise SystemExit(m._run_generic_split_completion(c,out,lambda:{'cut':logits},lambda x:x,lambda x:{'logits':x['cut']},artifacts,clone_payload=None,repetition_index=int(index)))
'''

@pytest.fixture(scope='module')
def generated_case(tmp_path_factory):
    tmp=tmp_path_factory.mktemp('real_process_fixture');config,logits=_fixture(tmp)
    config['identity'].update(direction='deepx_m1_to_tensorrt',comparison_backend='deepx_to_trt',output_endpoint_id='classification_top1_top5',setup_id='orin_nx_deepx_m1_01')
    bundle=tmp/'suite';bundle.mkdir();_copy_runner_lib(bundle)
    runner=Path(write_runner_onnxruntime(str(bundle/'runner')))
    case={'runtime_config':config,'remote_root':str(bundle),'original_run':str(tmp/'closed-original'),
          'remote_python_executable':sys.executable,'remote_environment':{'PATH':os.environ['PATH']},'identity':config['identity'],
          'setup_id':config['identity']['setup_id']}
    bind_runtime_source_closure(case,bundle)
    worker=tmp/'worker.py';worker.write_text(WORKER)
    cfg=tmp/'performance-config.json';cfg.write_text(json.dumps(config));perf=tmp/'performance'
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
    for rep in (1,2,3):
        subprocess.run([sys.executable,'-B',str(worker),str(runner),str(cfg),str(perf),str(rep)],env=env,check=True,capture_output=True,text=True)
    performance=perf/'generic_completion_results.json';actual=json.loads(performance.read_text())
    assert len({(r['pid'],r['process_start_ticks'],r['boot_id']) for r in actual['repetitions']})==3
    hardware=tmp/'hardware.json';hardware.write_text(json.dumps({'hardware_setups':[{'id':case['setup_id'],'energy':{'urecs_address':'fixture-source'}}]}))
    budget=tmp/'campaign.json';budget.write_text(json.dumps({'campaign':True,'max_transport_failures':20,'sources':{'fixture-source':{'stop_reason':''}},'tasks':{}}))
    plan=prepare_energy_case(case,performance_result=performance,output=tmp/'energy-plan',collector_python=sys.executable,
        collector_script=Path(__file__).resolve().parents[1]/'scripts/energy_measurement_cli.py',hardware_setups_file=hardware,remote_command_prefix=[],campaign_budget_file=budget)
    descriptors=[]
    for index in (1,2,3):
        resultdir=tmp/f'runtime-energy-{index}';start=time.time_ns()
        completed=subprocess.run([sys.executable,'-B',str(worker),str(runner),plan['config_path'],str(resultdir),str(index)],env=env,check=True,capture_output=True,text=True)
        finished=time.time_ns();result=resultdir/'generic_energy_results.json';row=json.loads(result.read_text())['repetitions'][0]
        command=plan['calls'][index-1]['argv'];output=Path(command[command.index('--out')+1]);directory=output/'run_000';directory.mkdir(parents=True);log=directory/'workload_stdout.log';log.write_text(completed.stdout)
        runid=command[command.index('--run-id')+1]
        energy=1200.*index**2
        record={'run_index':0,'postprocess_status':'ok','final_energy_gate_status':'pass','scientific_primary_energy_status':'available',
            'scientific_primary_energy_j':energy,'scientific_primary_active_duration_s':61.,'energy_configured_workload_duration_s':60.,
            'energy_work_units_used':row['completed_work_units'],'runtime_work_unit_evidence':{'exact':True,'count':row['completed_work_units']},
            'workload_timing':{'status':'ok','rc':0,'start_ns':start,'end_ns':start+61_000_000_000},
            'preflight_evidence':{'ok':True,'nonce':row['preflight_nonce']},
            'collector_and_workload_log_diagnostics':{'workload_stdout':{'available':True,'bytes':log.stat().st_size,'sha256':file_hash(log)}},
            'energy_efficiency_claim_eligible':True}
        aggregate=_aggregate(output,requested_runs=1,setup_id=case['setup_id'],run_id=runid)
        aggregate.update(runs=[record],preflight_requested=True,preflight_expected_command_contract_sha256=plan['config']['energy_capture']['contract']['command_contract_sha256'],preflight_verified_run_count=1,energy_efficiency_claim_eligible=True)
        raw=output/'energy_aggregate.json';raw.write_text(json.dumps(aggregate))
        descriptors.append({'repetition_index':index,'command':command,'command_sha256':canonical_hash(command),'output_dir':str(output),'raw_aggregate_path':str(raw),'raw_aggregate_sha256':file_hash(raw),
            'process_receipt':{'argv':command,'command_sha256':canonical_hash(command),'returncode':0,'started_ns':start,'finished_ns':finished,'aggregate_absent_before_execution':True},
            'runtime_result_path':str(result),'runtime_remote_result_path':str(result)})
    state=json.loads(budget.read_text());state['tasks'][plan['campaign_row_id']]={'source_id':'fixture-source','limits':plan['campaign_limits'],'stop_reason':'','chains':[{'logical_repeat':f'repeat:{i-1}','finished':True,'valid':True,'source_completion_verified':True,'run_directory':str(Path(d['output_dir'])/'run_000')} for i,d in enumerate(descriptors,1)]};budget.write_text(json.dumps(state))
    return plan,descriptors,case


def test_generated_bundle_six_real_processes_and_normal_three_energy_import(generated_case):
    plan,descriptors,case=generated_case
    before={d['raw_aggregate_path']:Path(d['raw_aggregate_path']).read_bytes() for d in descriptors}
    result=collect_generic_energy_case(plan,descriptors,case)
    assert result['normal_report_verified'] is True and result['valid_repetitions']==3
    assert result['raw_run_indices']==[0,0,0]
    assert len({(r['pid'],r['process_start_ticks'],r['boot_id']) for r in result['observations']})==3
    assert result['energy_efficiency_claim_eligible'] is False  # No dataset Quality fixture.
    assert result['completed_tasks_per_j']==pytest.approx(statistics.mean(r['completed_tasks_per_j'] for r in result['observations']))
    assert result['completed_tasks_per_j']!=pytest.approx(result['inverse_mean_energy_per_completed_task_j'])
    assert all(Path(p).read_bytes()==b for p,b in before.items())
    assert all('--inference-count' not in c['argv'] and '--pipeline-fps' not in c['argv'] for c in plan['calls'])

@pytest.mark.parametrize('mutation',['duplicate','source','performance_hash','mode','quality'])
def test_three_capture_source_and_pairing_guards(generated_case,mutation):
    plan,descriptors,case=copy.deepcopy(generated_case)
    if mutation=='duplicate':descriptors[2]=copy.deepcopy(descriptors[0]);descriptors[2]['repetition_index']=3
    if mutation=='source':plan['config']['energy_capture']['contract']['source_files'][0]['sha256']='a'*64
    if mutation=='performance_hash':plan['performance_result_sha256']='a'*64
    if mutation=='mode':plan['config']['energy_capture']['contract']['completion_runtime_mode']='strict'
    if mutation=='quality':case['runtime_config']['completion_contract']['label_offset']=1
    with pytest.raises(ValueError):collect_generic_energy_case(plan,descriptors,case)


def test_planned_preflight_runs_through_normal_collector_without_acquisition(generated_case,tmp_path):
    from onnx_splitpoint_tool.energy.collector import _run_energy_preflight
    plan,descriptors,case=generated_case
    # The test's remote_root is the local fixture suite, never a DUT.
    config_path=Path(plan['remote_config_path']);config_path.parent.mkdir(parents=True,exist_ok=True)
    config_path.write_bytes(Path(plan['config_path']).read_bytes())
    call=plan['calls'][0];argv=call['argv']
    template=Path(call['preflight_command_file']).read_text()
    workload=Path(call['workload_command_file']).read_text()
    evidence,rendered=_run_energy_preflight(template,workload_command_template=workload,evidence_dir=tmp_path/'preflight',repeat_index=0,
        timeout_s=10,max_age_s=60,expected_command_contract_sha256=plan['config']['energy_capture']['contract']['command_contract_sha256'],
        runtime_attestation_path_template=argv[argv.index('--preflight-runtime-attestation-path')+1])
    assert evidence['ok'] is True,evidence
    assert '__ONNX_SPLITPOINT_PREFLIGHT_NONCE__' not in rendered
    assert evidence['nonce'] in rendered
    assert not (tmp_path/'preflight'/'energy_aggregate.json').exists()
