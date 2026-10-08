import copy
import json
import sys
import time
from pathlib import Path
import pytest
from onnx_splitpoint_tool.runners import generic_energy as energy
from onnx_splitpoint_tool.runners import task_completion
from onnx_splitpoint_tool.energy.collector import _validate_energy_preflight_attestation, _extract_work_units_from_text_v60i
from tests import test_v2912_generic_split_completion as fixtures


def configured(tmp_path, monkeypatch):
    module=fixtures._module(monkeypatch)
    monkeypatch.setitem(sys.modules,'splitpoint_runners.generic_energy',energy)
    config,logits=fixtures._fixture(tmp_path)
    config['identity']['comparison_backend']='deepx_to_trt'
    sources=sorted([fixtures._descriptor(Path(module.__file__)),fixtures._descriptor(Path(energy.__file__)),fixtures._descriptor(Path(task_completion.__file__))],key=lambda x:x['path'])
    config['runtime_source_files']=sources
    c=energy.build_contract(config,source_files=sources,runner_path=str(Path(module.__file__)),completion_runtime_mode='classification_top1_top5')
    config['energy_capture']={'contract':c}
    nonce='test-fresh-nonce-0000000001';path=tmp_path/'preflight.json'
    att=energy.build_preflight(config,nonce=nonce,expected_sha256=c['command_contract_sha256'])
    path.write_text(json.dumps(att))
    config['_runtime_energy']={'energy_preflight_nonce':nonce,'energy_preflight_attestation':str(path),'energy_command_contract_sha256':c['command_contract_sha256']}
    return module,config,logits,att,path


def test_generic_preflight_is_accepted_by_normal_collector(tmp_path,monkeypatch):
    module,c,logits,att,path=configured(tmp_path,monkeypatch)
    now=time.time_ns()
    result=_validate_energy_preflight_attestation(att,nonce=c['_runtime_energy']['energy_preflight_nonce'],started_at_unix_ns=now-1_000_000_000,ended_at_unix_ns=now,max_age_s=60,expected_command_contract_sha256=c['energy_capture']['contract']['command_contract_sha256'])
    assert result['ok'] is True
    assert energy.verify_runtime_preflight(c,nonce=att['nonce'],attestation_path=path,expected_sha256=att['command_contract_sha256'])==att


@pytest.mark.parametrize('mutation',['nonce','seal','expired','source','input','mode','queue'])
def test_preflight_and_contract_fail_closed(tmp_path,monkeypatch,mutation):
    module,c,logits,att,path=configured(tmp_path,monkeypatch)
    if mutation=='nonce':att['nonce']='different-000000000000'
    if mutation=='seal':att['attestation_sha256']='a'*64
    if mutation=='expired':att['expires_at_unix_ns']=1;att['attestation_sha256']=energy.canonical_hash({k:v for k,v in att.items() if k!='attestation_sha256'})
    if mutation=='source':c['energy_capture']['contract']['source_files'][0]['sha256']='a'*64
    if mutation=='input':c['input_binding']['prepared_input']['sha256']='a'*64
    if mutation=='mode':c['energy_capture']['contract']['completion_runtime_mode']='strict'
    if mutation=='queue':c['queue_depth']=3
    path.write_text(json.dumps(att))
    with pytest.raises(ValueError):energy.verify_runtime_preflight(c,nonce=c['_runtime_energy']['energy_preflight_nonce'],attestation_path=path,expected_sha256=c['_runtime_energy']['energy_command_contract_sha256'])


def test_real_template_energy_duration_marker_samples_and_no_reuse(tmp_path,monkeypatch,capsys):
    module,c,logits,att,path=configured(tmp_path,monkeypatch)
    # Synthetic clock acceleration is confined to this CPU fixture; no capture or scientific output.
    real=task_completion.measure_split_completion
    monkeypatch.setattr(task_completion,'measure_split_completion',lambda *a,**kw:real(*a,**kw,clock=lambda:time.perf_counter_ns()*3000))
    output=tmp_path/'energy';output.mkdir()
    artifacts={k:Path(v['path']) for k,v in c['expected_runtime_artifacts'].items()}
    assert module._run_generic_split_completion(c,output,lambda:{'cut':logits},lambda x:x,
        lambda x:{'logits':x['cut']},artifacts,clone_payload=None,repetition_index=1)==0
    result=json.loads((output/'generic_energy_results.json').read_text())
    row=energy.validate_energy_result(result,c,result_root=output)
    stdout=capsys.readouterr().out
    count,source=_extract_work_units_from_text_v60i(stdout)
    assert count==row['completed_task_count']>0
    assert 'completed_work_units' in source
    assert row['warmup_completed_count']==0 and row['accepted_work_drained'] is True
    markers=[json.loads(x[len(energy.COMPLETION_MARKER):]) for x in stdout.splitlines() if x.startswith(energy.COMPLETION_MARKER)]
    assert len(markers)==1 and markers[0]['report_sha256']==energy.file_hash(output/'generic_energy_results.json')
    assert not (output/'generic_completion_results.json').exists()
    with pytest.raises(ValueError,match='already exists'):
        module._run_generic_split_completion(c,output,lambda:{'cut':logits},lambda x:x,
            lambda x:{'logits':x['cut']},artifacts,clone_payload=None,repetition_index=1)


@pytest.mark.parametrize('family',['yolo26','yolo11','yolov7'])
def test_duration_uses_actual_detection_factory_and_strict_postflight(tmp_path,monkeypatch,capsys,family):
    import numpy as np
    from onnx_splitpoint_tool.native_detection_postprocess import build_detection_completion_runtime
    from onnx_splitpoint_tool import native_three_stage
    module,c,_,att,path=configured(tmp_path,monkeypatch)
    monkeypatch.setitem(sys.modules,'splitpoint_runners.native_three_stage',native_three_stage)
    if family=='yolo26':
        from tests.test_v272_detection_completion_runtime import _decoded_outputs,_runtime
        outputs=_decoded_outputs();reference=_runtime(outputs)
    elif family=='yolo11':
        from tests.test_v27923_decoded_completion import _outputs,_runtime
        outputs=_outputs();reference=_runtime(tmp_path,outputs)
    else:
        from tests.test_v272_yolov7_head_mapping import _heads_640,_raw_source
        outputs=_heads_640();reference=build_detection_completion_runtime(model_id='yolov7_paper',outputs=outputs,input_hw=[640,640],original_wh=[640,480],source_endpoint_contract=_raw_source(outputs,endpoint_hash='a'*64))
    c['task']='detection';c['completion_contract']=reference.execution_contract
    c['functional_reference']['outputs']={}
    for i,(name,value) in enumerate(outputs.items()):
        p=tmp_path/f'reference-output-{i}.npy';np.save(p,value);c['functional_reference']['outputs'][name]=fixtures._descriptor(p,value)
    c['functional_reference']['tolerances']={'match_iou_threshold':.5,'min_match_ratio':.8,'min_mean_iou':.85}
    contract=energy.build_contract(c,source_files=c['runtime_source_files'],runner_path=str(Path(module.__file__)),completion_runtime_mode='fast_oracle_outside_timing')
    c['energy_capture']={'contract':contract};c['_runtime_energy']['energy_command_contract_sha256']=contract['command_contract_sha256']
    att=energy.build_preflight(c,nonce=c['_runtime_energy']['energy_preflight_nonce'],expected_sha256=contract['command_contract_sha256'])
    path.write_text(json.dumps(att))
    real=task_completion.measure_split_completion
    monkeypatch.setattr(task_completion,'measure_split_completion',lambda *a,**kw:real(*a,**kw,clock=lambda:time.perf_counter_ns()*6000))
    output=tmp_path/'energy-detection';output.mkdir();artifacts={k:Path(v['path']) for k,v in c['expected_runtime_artifacts'].items()}
    assert module._run_generic_split_completion(c,output,lambda:outputs,lambda x:x,lambda x:x,artifacts,clone_payload=None,repetition_index=1)==0
    result=json.loads((output/'generic_energy_results.json').read_text());row=energy.validate_energy_result(result,c,result_root=output)
    assert row['completion_runtime_mode']=='fast_oracle_outside_timing'
    assert row['completion_execution_attestation']['completed_work_units']==row['completed_work_units']
    assert row['functional_control']['status']=='passed'
    bad=copy.deepcopy(result);bad['repetitions'][0]['completion_execution_attestation']['completed_work_units']+=1
    with pytest.raises(ValueError):energy.validate_energy_result(bad,c,result_root=output)
