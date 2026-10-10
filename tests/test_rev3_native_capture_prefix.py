"""Normal resume integration with existing bounded local fixtures; no DUT."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import test_v2712_energy_resume as fixture
import test_v2912_energy_previous_results as previous_fixture

IDENTITIES=[('hailo10h_to_trt','mobilenet_v3_large','b001','orin_nx_hailo10_01'),
            ('native_full_tensorrt','mobilenet_v3_large','full','orin_nx_hailo10_01'),
            ('native_full_hailo10h','mobilenet_v3_large','full','orin_nx_hailo10_01')]

@pytest.fixture(params=['scripts/run_native_producer_energy_from_summary.py',
                        'onnx_splitpoint_tool/resources/remote_scripts/run_native_producer_energy_from_summary.py'])
def runner_namespace(request):
    return runpy.run_path(str(ROOT/request.param))


@pytest.mark.parametrize('reused,limit,negative',[
    (0,2,None),(1,2,None),(2,2,None),(0,0,None),
    (0,2,'preflight'),(0,2,'context_command'),(0,2,'context_science'),
])
def test_normal_main_preflights_and_captures_same_immutable_prefix(tmp_path,monkeypatch,reused,limit,negative,runner_namespace):
    ns=runner_namespace
    rows=[]
    for index,identity in enumerate(IDENTITIES):
        row=previous_fixture._row(tmp_path/f'previous/measurements/{index}',attempt='old',identity=identity)
        row['measure_command']+=previous_fixture._campaign_args(tmp_path)
        rows.append(row)
    plan=previous_fixture._plan(rows,attempt='old')
    results=[]
    for index,row in enumerate(rows):
        if index<reused:
            item=fixture._complete_result(row,marker='immutable-original-'+str(index))
            aggregate=Path(row['measurement_output_dir'])/'energy_aggregate.json'
            aggregate.parent.mkdir(parents=True,exist_ok=True)
            aggregate.write_text(json.dumps(item['run']['energy_aggregate']))
            item['run'].update(energy_aggregate_path=str(aggregate),energy_aggregate_sha256=hashlib.sha256(aggregate.read_bytes()).hexdigest())
        else:item=fixture._failed_result(row)
        results.append(item)
    previous={'plan_payload':plan,'rows':results,'status':'partial','ok':False,'complete':False,
              'requested':True,'dry_run':False,**{k:reused if k=='started_measurement_count' else 0 for k in previous_fixture.COUNTERS}}
    previous['plan']={'cmd':['python','energy_plan.py']}
    for field,value in fixture._EXECUTION_CONTEXT.items():previous['plan']['cmd']+=['--'+field.replace('_','-'),value]
    source=tmp_path/'previous/native_producer_energy_results.json'
    source.write_text(json.dumps(previous))
    summary=tmp_path/'summary.json';summary.write_text('{"rows":[]}')
    out=tmp_path/'new'
    original_plan=fixture._plan
    monkeypatch.setattr(fixture,'_plan',lambda *args,**kwargs:{**original_plan(*args,**kwargs),
        'energy_profile_requested_runs_per_row':3,'energy_effective_runs_per_row':3,
        'energy_repeat_expansion_applied':False,'energy_repeat_expansion_reason':'none'})
    def transform(row,index,_root):
        row['measure_command']+=previous_fixture._campaign_args(tmp_path)
        row.update(measurement_profile_requested_repeats=3,measurement_effective_repeats=3,
                   measurement_repeat_expansion_applied=False,measurement_repeat_expansion_reason='none')
        if index==2 and negative=='context_command':Path(row['preflight_command_file']).write_text('#!/bin/sh\nexit 0\n')
        if index==2 and negative=='context_science':row['successful_command_contract_sha256']='b'*64
        return row
    calls=fixture._install_fake_runner(ns,out=out,old_plan=plan,row_transform=transform)
    observed=[]
    def preflight(selected,**kwargs):
        observed.append(copy.deepcopy(selected))
        return {'ok':negative!='preflight','measurement_wrapper_allowed':negative!='preflight'}
    ns['main'].__globals__['_prepare_resume_measurement_cohort']=preflight
    argv=['runner','--summary',str(summary),'--out-dir',str(out),'--previous-results',str(source),'--limit',str(limit)]
    for field,value in fixture._EXECUTION_CONTEXT.items():argv+=['--'+field.replace('_','-'),value]
    monkeypatch.setattr(sys,'argv',argv)
    before=previous_fixture._files(source.parent)
    if negative in ('context_science','context_command'):
        with pytest.raises(ValueError,match='contract drift|remote command drift'):
            ns['main']()
        assert not calls and not observed
    else:
        rc=ns['main']()
        if negative in ('context_command','preflight'):
            assert rc!=0 and not calls
            if negative=='context_command':assert not observed
        else:
            expected=IDENTITIES[reused:limit or len(IDENTITIES)]
            assert calls==['|'.join(x)for x in expected]
            assert [[tuple(r[k]for k in ('backend','model','case','setup_id'))for r in selected]for selected in observed]==([expected]if expected else [])
            report=json.loads((out/'native_producer_energy_results.json').read_text())
            assert len(report['rows'])==3
            assert report['supplement_reused_row_count']==reused
            assert report['rows'][2].get('skipped')==('limit'if limit else None)
            # Fake aggregate leaves contain no physical start journal. They
            # verify dispatch scope, never claim a scientific capture PASS.
            assert rc in (0,2,3)
    assert previous_fixture._files(source.parent)==before

