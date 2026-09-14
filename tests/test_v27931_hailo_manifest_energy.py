"""AP3/AP4: unchanged original manifest replay and explicitly synthetic integration.

Original reduced fixtures lack tensor payloads and cannot certify full evidence.
Synthetic fixtures execute real command, consumer, oracle and energy verifiers.
"""
from __future__ import annotations
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256, seal_native_command_contract
from onnx_splitpoint_tool.native_three_stage import FastDetectionCompletionRuntime
from scripts import native_producer_final_report as final
from scripts import native_producer_energy_plan as energy
from scripts import run_native_producer_energy_from_summary as runner
from scripts.native_hailo_trt_fifo_from_benchmarkset import _hailo8_completion_fields
from scripts.native_producer_validate_visualize import _verified_completed_v2_contract, _completed_v2_self_reference_detection
from onnx_splitpoint_tool.native_detection_postprocess import persist_detection_completion_execution_artifacts
from test_v269f_native_split_final_energy_integrity import _quality_first_row
from test_v279_native_three_stage import _decoded_completion_contract
from test_v272_hailo8_completed_detection_hotloop import _sealed_hailo8_command_contract

FIXTURE = Path(__file__).parent / 'fixtures/v27931_complete_set/hailo8'
CASES = [('yolo11l','b062','584573610c7409471264c78c0b951cf957b68ef1da8b3bf0c3ad6b8e900288a6'),
         ('yolov7_paper','b044','54046c8ce09128e95995fae55b6259fc47637a9977e9101ca9fbe239bffa09d2')]
MODE = 'native_three_stage_fast_oracle_outside_timing'

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _original(tmp_path, model, case):
    root = tmp_path/model/'benchmark_set/native_pipeline'/case/'hailo_to_trt/uint8_dequant_fp16'
    shutil.copytree(FIXTURE/model/case, root)
    path=root/'native_fifo_results.json';row=json.loads(path.read_text())
    command=row['native_command_contract']
    row.update({key:command[key] for key in ('backend','model','case','precision','setup_id','comparison_backend')})
    return row,path

def _reseal(row):
    row['native_command_contract']=seal_native_command_contract(row['native_command_contract'])
    row['native_command_contract_sha256']=row['native_command_contract']['contract_sha256']
    att=row['native_split_quality_consumer_attestation'];att.pop('attestation_sha256',None)
    att['command_contract_sha256']=row['native_command_contract_sha256']
    for kind in ('output','boundary'):
        att['semantic_'+kind+'_manifest_sha256']=row['native_command_contract']['artifacts']['semantic_'+kind+'_manifest']['sha256']
    att['attestation_sha256']=canonical_json_sha256(att)

def _synthetic(tmp_path, *, nested=True):
    row, old_manifest, old_payload = _quality_first_row(tmp_path/'origin')
    job=tmp_path/'collection/yolo26s/benchmark_set/native_pipeline/b038/hailo_to_trt/uint8_dequant_fp16'
    job.mkdir(parents=True);result_path=job/'native_fifo_results.json';result_path.write_text('{}')
    outputs,contract=_decoded_completion_contract();runtime=FastDetectionCompletionRuntime(contract)
    runtime.process(outputs);runtime.process(outputs)
    row.update(_hailo8_completion_fields(runtime,completed_work_units=2))
    row.update(stage='decoded_nms',contract_family='decoded_nms',output_format='decoded_nms',endpoint_contract_complete=True,
               measurement_endpoint='completed_task',measurement_boundary='workers_ready_to_last_completed_task_frame',
               frames=2,warmup=10,three_stage_concurrency_directly_measured=False)
    command=row['native_command_contract'];command['measurement_endpoint']='completed_task'
    mixed=_sealed_hailo8_command_contract('hailo8_python_vstreams_fifo')
    command['mixed_runtime_contract']=mixed['mixed_runtime_contract']
    command['interpreter_identity']=mixed['interpreter_identity']
    command['python_executable']=mixed['python_executable']
    command['runtime_options'].update(mixed['runtime_options'])
    for key in ('python_executable','native_executable','native_trt_consumer_source'):
        command['artifacts'][key]=mixed['artifacts'][key]

    command['runtime_options'].update(measurement_endpoint='completed_task',frames=2,duration_s=0,
        completion_runtime_mode='fast_oracle_outside_timing',completion_execution_contract=contract)
    for kind in ('output','boundary'):
        folder = job/('endpoints/completed_task' if nested else '')/('native_fifo_outputs' if kind=='output' else 'native_fifo_boundary')
        folder.mkdir(parents=True)
        payload=folder/(kind+'.bin')
        values=next(iter(outputs.values())) if kind=='output' else np.arange(4,dtype=np.uint8)
        payload.write_bytes(values.tobytes())
        records=[{'path':str(payload),'size_bytes':payload.stat().st_size,'sha256':_sha(payload)}]
        manifest={'task':'detection','stage':'decoded_nms','output_format':'decoded_nms','contract_family':'decoded_nms',
            'endpoint_contract_complete':True,'endpoint_contract_hash':contract['source_endpoint']['endpoint_contract_hash'],
            'payload_artifacts':records,'payload_artifacts_sha256':canonical_json_sha256(records),
            'outputs':[{'name':next(iter(outputs)) if kind=='output' else 'boundary','file':payload.name,'shape':list(values.shape),'dtype':str(values.dtype)}]}
        target=folder/('native_fifo_output_manifest.json' if kind=='output' else 'native_fifo_boundary_manifest.json')
        target.write_text(json.dumps(manifest,indent=2))
        command['artifacts']['semantic_'+kind+'_manifest']={'path':str(target),'sha256':_sha(target),'size_bytes':target.stat().st_size}
        row['native_fifo_'+kind+'_manifest']=str(target)
    _reseal(row)
    return row,result_path,outputs

def _collect(row,path):
    row.update(final._split_output_contract(path,row,fallback_relpaths=('native_fifo_outputs/native_fifo_output_manifest.json',)))
    return row

@pytest.mark.parametrize('model,case,expected',CASES)
def test_t03_1_2_original_role_manifest_bytes_found_but_payloads_remain_missing(tmp_path,model,case,expected):
    row,path=_original(tmp_path,model,case);before=copy.deepcopy(row['native_command_contract'])
    fields=final._split_output_contract(path,row)
    assert fields['manifest_resolution_status']=='manifest_verified'
    assert fields['output_manifest_sha256']==expected
    assert fields['native_split_semantic_binding_valid'] is False
    assert fields['native_split_semantic_binding_status']=='payload_artifact_0_missing'
    assert row['native_command_contract']==before
    assert fields['stage']==('decoded_pre_nms' if model=='yolo11l' else 'raw_head')

@pytest.mark.parametrize('model,case,expected',CASES)
def test_t03_3_bound_role_wins_with_raw_and_completed_directories(tmp_path,model,case,expected):
    row,path=_original(tmp_path,model,case)
    fields=final._split_output_contract(path,row)
    assert '/endpoints/completed_task/' in fields['resolved_local_path']
    assert fields['output_manifest_sha256']==expected

@pytest.mark.parametrize('mutation',['setup','case','run','traversal','symlink','conflicting_sources','role'])
def test_t03_4_foreign_context_paths_and_ambiguous_sources_fail_closed(tmp_path,mutation):
    row,path,outputs=_synthetic(tmp_path)
    manifest=Path(row['native_fifo_output_manifest'])
    if mutation=='setup':row['setup_id']='another_setup'
    elif mutation=='case':row['case']='b999'
    elif mutation=='run':row['eval_run_id']='foreign-run'
    elif mutation=='traversal':row['native_fifo_output_manifest']='../foreign.json'
    elif mutation=='symlink':
        external=tmp_path/'outside.json';shutil.copyfile(manifest,external);manifest.unlink();manifest.symlink_to(external)
    elif mutation=='role':row['measurement_endpoint']='raw_model_outputs'
    else:
        another=path.parent/'native_fifo_outputs/native_fifo_output_manifest.json';another.parent.mkdir();shutil.copyfile(manifest,another)
    fields=final._split_output_contract(path,row,fallback_relpaths=('native_fifo_outputs/native_fifo_output_manifest.json',))
    assert fields['native_split_semantic_binding_valid'] is False
    assert fields['manifest_resolution_status'] not in ('manifest_verified','legacy_manifest_found')

@pytest.mark.parametrize('mutation',['manifest_bytes','consumer_hash'])
def test_t03_5_found_content_conflict_is_not_missing(tmp_path,mutation):
    row,path,outputs=_synthetic(tmp_path)
    if mutation=='manifest_bytes':Path(row['native_fifo_output_manifest']).write_text('{}')
    else:row['native_split_quality_consumer_attestation']['semantic_output_manifest_sha256']='f'*64
    fields=final._split_output_contract(path,row)
    assert fields['output_contract_manifest_status']=='content_invalid'
    assert fields['manifest_resolution_status']!='missing'

@pytest.mark.parametrize('kind',['output','boundary'])
@pytest.mark.parametrize('mutation',['missing','size','bytes','duplicate_keys'])
def test_t03_6_manifest_presence_never_substitutes_for_payload_closure(tmp_path,kind,mutation):
    row,path,outputs=_synthetic(tmp_path);manifest=Path(row['native_fifo_'+kind+'_manifest']);data=json.loads(manifest.read_text())
    payload=Path(data['payload_artifacts'][0]['path'])
    if mutation=='missing':payload.unlink()
    elif mutation=='size':payload.write_bytes(b'x')
    elif mutation=='bytes':payload.write_bytes(b'x'*payload.stat().st_size)
    else:manifest.write_text('{"payload_artifacts":[],"payload_artifacts":[]}')
    fields=final._split_output_contract(path,row)
    assert fields['native_split_semantic_binding_valid'] is False

@pytest.mark.parametrize('nested',[False,True])
def test_t03_7_flat_and_nested_use_identical_real_portable_binder(tmp_path,nested):
    row,path,outputs=_synthetic(tmp_path,nested=nested);_collect(row,path)
    assert row['native_split_semantic_binding_valid'] is True
    assert row['native_split_final_portable_binding_valid'] is True
    evidence,status=energy._split_quality_energy_evidence(row,row['native_command_contract'])
    assert evidence and status=='portable_join_and_semantic_payload_bytes_rehashed'


def test_t04_1_real_fast_verifier_through_collection_report_validator_and_energy(tmp_path):
    row,path,outputs=_synthetic(tmp_path);_collect(row,path)
    persist_detection_completion_execution_artifacts(row,output_path=path.parent/'sentinel.json')
    assert _verified_completed_v2_contract(row)[0]==MODE
    assert _completed_v2_self_reference_detection(outputs,outputs,row)['available'] is True
    assert final._explicit_completed_task_comparison_endpoint(row)==row['completed_task_comparison_output_endpoint_id']
    identity=energy._energy_endpoint_identity(row,None,row['native_command_contract'])
    assert identity['completion_pairing_eligible'] is True,identity
    evidence,status=energy._split_quality_energy_evidence(row,row['native_command_contract'])
    assert evidence and evidence['native_split_energy_binding_valid'] is True
    assert row['completed_task_endpoint_attestation']['observation_relation']=='postflight_oracle_sentinel'
    assert row['three_stage_concurrency_directly_measured'] is False

@pytest.mark.parametrize('mutation',['oracle_missing','artifact','location','contract','sentinel'])
def test_t04_2_inner_resealed_oracle_and_dumped_sentinel_conflicts_rejected(tmp_path,mutation):
    row,path,outputs=_synthetic(tmp_path);_collect(row,path)
    att=row['completed_task_endpoint_attestation']
    if mutation=='oracle_missing':att.pop('artifact')
    elif mutation=='artifact':att['artifact']['source_content_sha256']='f'*64
    elif mutation=='location':att['quality_oracle_location']='inside_timing'
    elif mutation=='contract':att['execution_contract_sha256']='f'*64
    else:
        outputs={k:v.copy() for k,v in outputs.items()};next(iter(outputs.values())).flat[0]+=1
    if mutation!='sentinel':att.pop('attestation_sha256',None);att['attestation_sha256']=canonical_json_sha256(att)
    with pytest.raises(Exception):energy._verified_fast_energy_completion(row,None,row['native_command_contract'],outputs=outputs)

@pytest.mark.parametrize('field,value',[('completed_work_units',0),('completed_frames',True),('completed_work_units',12),('postprocess_completed_frames',3)])
def test_t04_3_exact_counts_exclude_warmup_and_sentinel(tmp_path,field,value):
    row,path,outputs=_synthetic(tmp_path);_collect(row,path);row[field]=value
    assert energy._energy_endpoint_identity(row,None,row['native_command_contract'])['completion_pairing_eligible'] is False

@pytest.mark.parametrize('field,value',[('input_image_sha256','0'*64),('hef_sha256','f'*64),('case','b039'),('precision','fp16'),('comparison_backend','hailo10h')])
def test_t04_4_same_name_does_not_allow_other_input_artifact_or_job(tmp_path,field,value):
    row,path,outputs=_synthetic(tmp_path);_collect(row,path)
    if field=='hef_sha256':row['native_command_contract']['hef_sha256']='e'*64;_reseal(row)
    row[field]=value
    assert energy._energy_endpoint_identity(row,None,row['native_command_contract'])['completion_pairing_eligible'] is False

@pytest.mark.parametrize('field,value',[('measurement_endpoint','raw_model_outputs'),('measurement_boundary','workers_ready_to_last_completed_trt_frame')])
def test_t04_5_completed_fps_requires_completed_execution_window(tmp_path,field,value):
    row,path,outputs=_synthetic(tmp_path);_collect(row,path);row[field]=value
    assert energy._energy_endpoint_identity(row,None,row['native_command_contract'])['completion_pairing_eligible'] is False


def _fresh(tmp_path):
    row,path,outputs=_synthetic(tmp_path);_collect(row,path)
    runtime=FastDetectionCompletionRuntime(row['completion_execution_contract'])
    for _ in range(3):runtime.process(outputs)
    fresh={**{key:row[key] for key in ('model','case','setup_id','comparison_backend','precision')},
           **_hailo8_completion_fields(runtime,completed_work_units=3),'warmup':0,'measurement_endpoint':'completed_task',
           'energy_preflight_nonce':'fresh-nonce','source_contract_sha256':row['native_command_contract_sha256'],
           'energy_completion_observation_scope':'fresh_energy_invocation','energy_completion_window_source':'collector_command_marker_window'}
    run={'preflight_evidence':{'ok':True,'nonce':'fresh-nonce'},'runtime_work_unit_evidence':{'exact':True,'count':3},
         'workload_timing':{'status':'ok','rc':0,'start_ns':10,'end_ns':20}}
    directory=tmp_path/'energy/run_000';directory.mkdir(parents=True)
    return row,fresh,run,directory

@pytest.mark.parametrize('mutation',['none','copied_performance','nonce','count','warmup','window','duplicate'])
def test_t04_6_only_fresh_energy_completion_is_accepted(tmp_path,mutation):
    row,fresh,run,directory=_fresh(tmp_path)
    if mutation=='copied_performance':fresh=copy.deepcopy(row)
    elif mutation=='nonce':fresh['energy_preflight_nonce']='old-nonce'
    elif mutation=='count':run['runtime_work_unit_evidence'].pop('count')
    elif mutation=='warmup':fresh['warmup']=1
    elif mutation=='window':run['workload_timing']['end_ns']=9
    line='__SPLITPOINT_ENERGY_COMPLETION__='+json.dumps(fresh)+'\n'
    (directory/'workload_stdout.log').write_text(line*(2 if mutation=='duplicate' else 1))
    run['collector_and_workload_log_diagnostics']={'workload_stdout':{'available':True,'bytes':(directory/'workload_stdout.log').stat().st_size,'sha256':_sha(directory/'workload_stdout.log')}}
    evidence,reason=runner._verify_fresh_fast_energy_completion(run,row,run_dir=directory)
    assert (evidence is not None)==(mutation=='none'),reason
    if evidence:assert evidence['completed_work_units']==3 and evidence['observation_relation']=='postflight_oracle_sentinel'

@pytest.mark.parametrize('model,case,expected',CASES)
def test_t04_7_both_originals_cross_real_consumers_without_fake_payload_qualification(tmp_path,model,case,expected):
    row,path=_original(tmp_path,model,case);_collect(row,path)
    assert _verified_completed_v2_contract(row)[0]==MODE
    identity=energy._energy_endpoint_identity(row,None,row['native_command_contract'])
    assert identity['completion_pairing_status']=='strict_fast_postflight_completion_verified',identity
    assert final._explicit_completed_task_comparison_endpoint(row)==row['completed_task_comparison_output_endpoint_id']
    evidence,status=energy._split_quality_energy_evidence(row,row['native_command_contract'])
    assert evidence is None
    assert row['native_split_semantic_binding_valid'] is False
    assert row['three_stage_concurrency_directly_measured'] is False


def test_t04_6_actual_energy_entrypoint_runs_fast_fifo_and_emits_own_nonce(tmp_path,monkeypatch,capsys):
    """Only device interfaces are simulated; preflight seal, FIFO and oracle are real."""
    import time
    from types import SimpleNamespace
    from onnx_splitpoint_tool.native_command_contract import seal_split_energy_preflight_attestation
    from scripts import native_hailo_trt_fifo_from_benchmarkset as native
    row,result_path,outputs=_synthetic(tmp_path)
    command=row['native_command_contract']
    prepared=tmp_path/'prepared.bin';prepared.write_bytes(np.zeros((1,4),dtype=np.float32).tobytes())
    artifact_paths={}
    for key in ('hef','engine','native_executable'):
        path=tmp_path/key;path.write_bytes(b'synthetic-device-artifact');artifact_paths[key]={'path':str(path)}
    artifact_paths['prepared_input']={'path':str(prepared)}
    source_image=tmp_path/'input.jpg';source_image.write_bytes(b'synthetic-prepared-source')
    options=dict(command['runtime_options']);options.update(warmup=0,build=False,dump_outputs=False,dump_boundary=False,
        task='detection',preprocess_mode_requested='auto',preprocess_mode_effective='letterbox',
        letterbox_pad_value_requested=0,letterbox_pad_value_effective=0,letterbox_pad_value=0,
        queue_depth=2,hailo_format='float32',copy_outputs=True,producer_impl='hailo8_python_vstreams_fifo')
    prep={'name':'images','shape':[1,4],'dtype':'float32','task':'detection',
          'preprocess_mode_requested':'auto','preprocess_mode_effective':'letterbox',
          'letterbox_pad_value_requested':0,'letterbox_pad_value_effective':0,'letterbox_pad_value':0,'pad_value_effective':0}
    stat=prepared.stat()
    binding={'schema':'onnx-splitpoint/split-energy-workload-binding','schema_version':1,'workload_supported':True,
        'command_contract_sha256':command['contract_sha256'],'backend':'hailo8_to_trt',
        **{key:command[key] for key in ('model','case','setup_id','comparison_backend','precision')},
        'benchmark_set':str(tmp_path),'hw_arch':'hailo8','input_image':str(source_image),
        'runtime_options':options,'prepared_input_contract':prep,'artifacts':artifact_paths,
        'runtime_boundary_evidence':{'status':'exact_runtime_boundary_verified','output_count':1,'output_name':'boundary','output_shape':[1,4],'output_dtype':'float32'},
        'verified_files':[{'path':str(prepared),'size_bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'device':stat.st_dev,'inode':stat.st_ino}]}
    now=time.time_ns();seal=seal_split_energy_preflight_attestation({'ok':True,'artifact_verification_status':'pass',
        'nonce':'actual-energy-nonce','command_contract_sha256':command['contract_sha256'],
        'created_at_unix_ns':now,'expires_at_unix_ns':now+60_000_000_000,'workload_binding':binding})
    seal_path=tmp_path/'preflight.json';seal_path.write_text(json.dumps(seal))
    args=SimpleNamespace(energy_preflight_attestation=str(seal_path),energy_preflight_nonce='actual-energy-nonce',
        source_contract_sha256=command['contract_sha256'],energy_preflight_max_age_s=300.,warmup=0,build=False,dump_outputs=False,dump_boundary=False,
        duration_s=.05,benchmark_set=str(tmp_path),case=command['case'],precision=command['precision'],image=str(source_image),
        hw_arch='hailo8',queue_depth=2,hailo_format='float32',task='detection',preprocess_mode='auto',copy_outputs=True,
        letterbox_pad_value=0,result_json=str(tmp_path/'fresh.json'),frames=1,device_id='')
    calls=[]
    class Device:
        def run(self,prepared,inputs):
            calls.append('p1');return SimpleNamespace(outputs={'boundary':np.zeros((1,4),dtype=np.float32)})
    class TensorRT:
        inputs=['part2'];shapes={'part2':(1,4)};dtypes={'part2':np.float32}
        def run(self,inputs):calls.append('p2');return outputs
    monkeypatch.setattr(native,'_open_hailo8_python_runtime',lambda **kw:(Device(),SimpleNamespace(input_names=['images']),TensorRT()))
    monkeypatch.setattr(native,'_close_hailo8_python_runtime',lambda *a:None)
    assert native._energy_workload_only(args)==0
    fresh=json.loads((tmp_path/'fresh.json').read_text())
    assert fresh['completed_task_completion_mode']==MODE
    assert fresh['completed_work_units']==calls.count('p2')==calls.count('p1')>0
    assert fresh['warmup']==0
    directory=tmp_path/'captured';directory.mkdir();(directory/'workload_stdout.log').write_text(capsys.readouterr().out)
    run={'preflight_evidence':{'ok':True,'nonce':'actual-energy-nonce'},
        'runtime_work_unit_evidence':{'exact':True,'count':fresh['completed_work_units']},
        'workload_timing':{'status':'ok','rc':0,'start_ns':10,'end_ns':20}}
    run['collector_and_workload_log_diagnostics']={'workload_stdout':{'available':True,'bytes':(directory/'workload_stdout.log').stat().st_size,'sha256':_sha(directory/'workload_stdout.log')}}
    evidence,reason=runner._verify_fresh_fast_energy_completion(run,row,run_dir=directory)
    assert evidence is not None,reason
    assert evidence['completed_work_units']==fresh['completed_work_units']


def test_t01_6_reporter_groups_planned_repetitions_once_and_retains_all_errors():
    from onnx_splitpoint_tool.native_job_identity import planned_native_identity
    job=planned_native_identity({'backend':'hailo10','model':'yolo26s','case':'b364','setup_id':'setup-two','comparison_backend':'hailo10','precision':'uint8_dequant_fp16'})
    rows=[]
    for index,fps in enumerate((20.,10.,30.),1):
        rows.append({**job,'planned_native_identity':job,'producer_impl':'device' if index!=2 else 'scaffold',
            'task':'detection' if index!=2 else '', 'ok':True,'runtime_success':True,'fps_makespan':fps,
            'repetition_index':index,'repetition_id':f'rep-{index}','runtime_instance_id':f'process-{index}',
            'report':f'/isolated/report-{index}.json','repetition_count_requested':1,'repetition_count_attempted':1})
    # Scientific claim identity deliberately stays incomplete in this small
    # count-only fixture; median aggregation must still preserve the samples.
    grouped=final._aggregate_repetitions(rows)
    assert len(grouped)==1 and grouped[0]['fps_makespan']==20.
    assert grouped[0]['repetition_count_attempted']==3
    assert grouped[0]['fps_repetition_samples']==[20.,10.,30.]
    assert len(grouped[0]['repetition_records'])==3
    failed=copy.deepcopy(rows);failed[1].update(ok=False,runtime_success=False,fps_makespan=None,
        failure_stage='runtime',failure_reason='device_failure',error='exact-device-error')
    grouped=final._aggregate_repetitions(failed)
    assert len(grouped)==1 and grouped[0]['ok'] is False
    assert any(item.get('failure_reason')=='device_failure' for item in grouped[0]['native_job_observations'])
    assert grouped[0]['repetition_count_attempted']==3


def test_t01_8_report_and_energy_aliases_preserve_full_runtime_precision():
    from onnx_splitpoint_tool.native_job_identity import planned_native_identity, native_identity_key
    raw={'backend':'native_full_hailo10','model':'yolo26s','case':'full','setup_id':'a','comparison_backend':'hailo10','precision':'uint8','execution_precision':'uint8'}
    normalized=planned_native_identity(raw)
    assert runner._canonical_energy_identity(raw)==native_identity_key(raw)
    assert runner._canonical_energy_identity(raw)[0]=='native_full_hailo10h'
    assert runner._canonical_energy_identity(raw)[4]=='hailo10h'
    assert runner._canonical_energy_identity(raw)[5]==''
    assert energy._validation_identity(raw)==('native_full_hailo10h','yolo26s','full','uint8','a','hailo10h')
    fields=final._native_identity_fields({**raw,'planned_native_identity':normalized})
    assert fields['planned_native_identity']['precision']=='uint8'
    assert fields['comparison_backend']=='hailo10h'
    assert raw['precision']==raw['execution_precision']=='uint8'
    ambiguous={'backend':'native_full_tensorrt','model':'x','case':'full','setup_id':'deepx_named_setup','precision':'fp16'}
    with pytest.raises(ValueError,match='incomplete'):runner._canonical_energy_identity(ambiguous)


@pytest.mark.parametrize('model,case,expected',CASES)
@pytest.mark.parametrize('new_command_schema',[False,True])
def test_t04_7_real_plan_preserves_fast_contract_and_blocks_unbound_legacy_mode(tmp_path,monkeypatch,model,case,expected,new_command_schema):
    """The mode-added arm is synthetic; original historical evidence is not repaired."""
    import sys
    row,path=_original(tmp_path,model,case);_collect(row,path);row.update(runtime_success=True,ok=True)
    if new_command_schema:
        row['native_command_contract']['runtime_options']['completion_runtime_mode']='fast_oracle_outside_timing'
        _reseal(row)
    summary=tmp_path/'native_producer_summary.json';summary.write_text(json.dumps({'rows':[row]}))
    out=tmp_path/'plan'
    monkeypatch.setattr(sys,'argv',['energy','--summary',str(summary),'--out-dir',str(out),
        '--hailo8-ssh','diagnostic-no-connection','--duration-s','1','--screening-energy',
        '--measure-all-runtime-successful','--allow-unpaired'])
    rc=energy.main()
    plan=json.loads((out/'native_producer_energy_plan.json').read_text())
    if not new_command_schema:
        assert plan['rows']==[]
        assert plan['excluded_rows'][0]['reason']=='fast_energy_command_runtime_mode_unbound'
        assert plan['excluded_rows'][0]['repetition_count_attempted']==0
    else:
        assert rc==0 and len(plan['rows'])==1
        entry=plan['rows'][0]
        assert entry['completed_task_completion_mode']==MODE
        assert entry['native_command_contract']==row['native_command_contract']
        assert entry['energy_completion_requires_fresh_observation'] is True
        assert entry['energy_claim_eligible'] is False
        prepared=runner._prepare_measurement_execution(entry,plan,allowed_root=tmp_path,validate_only=True)
        assert prepared['validated'] is True
        prepared=runner._prepare_measurement_execution(entry,plan,allowed_root=tmp_path)
        assert prepared['row']['completed_task_completion_mode']==MODE
        assert prepared['row']['native_command_contract']==row['native_command_contract']


@pytest.mark.parametrize('valid_marker',[True,False])
def test_t04_6_aggregate_revalidates_fresh_energy_without_mutating_original(tmp_path,valid_marker):
    from test_v267_energy_aggregate_import import _aggregate,_command,_write_aggregate
    row,fresh,run,directory=_fresh(tmp_path)
    line='__SPLITPOINT_ENERGY_COMPLETION__='+json.dumps(fresh)+'\n' if valid_marker else '__SPLITPOINT_WORK_UNITS__=3\n'
    stdout=directory/'workload_stdout.log';stdout.write_text(line)
    run['collector_and_workload_log_diagnostics']={'workload_stdout':{'available':True,'bytes':stdout.stat().st_size,'sha256':_sha(stdout)}}
    aggregate=_aggregate(directory.parent,requested_runs=1,setup_id=row['setup_id'])
    aggregate.update(preflight_requested=True,preflight_expected_command_contract_sha256=row['native_command_contract_sha256'],preflight_verified_run_count=1)
    aggregate['runs']=[{'run_index':0,**run}]
    path,started=_write_aggregate(directory.parent,aggregate);before=path.read_bytes()
    result=runner._attach_energy_aggregate({'rc':0},_command(directory.parent,requested_runs=1,setup_id=row['setup_id']),directory.parent,
        expected_runs=1,expected_setup_id=row['setup_id'],expected_command_contract_sha256=row['native_command_contract_sha256'],
        execution_started_ns=started,aggregate_absent_before_execution=True,expected_native_row=row)
    assert result['energy_aggregate_verified'] is valid_marker
    assert result['energy_aggregate']==aggregate
    assert path.read_bytes()==before
    if valid_marker:assert result['fresh_energy_completion_evidence'][0]['completed_work_units']==3
    else:assert any('fresh_marker_missing' in reason for reason in result['energy_aggregate_completion_errors'])
