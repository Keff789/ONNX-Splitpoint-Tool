"""R9H: task completion, replicate arithmetic and faithful report projection."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from onnx_splitpoint_tool.energy.comparison import resolve_energy_comparison
from onnx_splitpoint_tool.native_split_quality_authority import apply_native_split_quality_authority
from onnx_splitpoint_tool.workflow.native_transfer import build_native_validation_image_map


def normalized_row(tmp_path):
    from test_v27914_simple_idle_calibration_comparison import _simple_verified_trt_row
    row = _simple_verified_trt_row(tmp_path / 'calibration.json')
    records = []
    for i, (energy, duration, units) in enumerate([(100., 8., 200), (150., 15., 100), (70., 6., 180)]):
        normalized = energy - 2 * duration
        records.append(dict(run_index=i, energy_total_j=energy, active_duration_s=duration,
            energy_work_units_used=units, energy_per_work_unit_j=energy/units, avg_power_w=energy/duration,
            host_normalized_energy_est_j=normalized, host_normalized_energy_per_work_unit_est_j=normalized/units,
            host_normalized_average_power_est_w=normalized/duration, accelerator_idle_w_applied=2.,
            postprocess_status='ok', accelerator_idle_correction_applied=True))
    aliases = {'energy_total_j':'energy_total_j','host_normalized_energy_est_j':'host_normalized_energy_est_j',
        'energy_per_work_unit_j':'energy_per_work_j','host_normalized_energy_per_work_unit_est_j':'host_normalized_energy_per_work_est_j',
        'avg_power_w':'average_power_w','host_normalized_average_power_est_w':'host_normalized_average_power_est_w',
        'active_duration_s':'active_duration_s','energy_work_units_used':'work_units'}
    for source, target in aliases.items(): row[target] = sum(r[source] for r in records)/len(records)
    row.update(energy_normalization_repeats=records, energy_repeat_valid_n=3)
    return row


def test_normalization_averages_ratios_on_the_same_replicates(tmp_path):
    row = normalized_row(tmp_path)
    assert row['energy_per_work_j'] != pytest.approx(row['energy_total_j']/row['work_units'])
    result = resolve_energy_comparison(row)
    assert result['energy_comparison_status'] == 'host_normalized_verified'
    assert result['comparison_energy_per_work_j'] == row['host_normalized_energy_per_work_est_j']
    assert row['energy_total_j'] > row['host_normalized_energy_est_j']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'ratio', 'aggregate', 'idle', 'count'])
def test_normalization_missing_or_conflicting_replicates_stay_invalid(tmp_path, damage):
    row = normalized_row(tmp_path)
    if damage == 'missing': row['energy_normalization_repeats'].pop()
    elif damage == 'duplicate': row['energy_normalization_repeats'][1]['run_index'] = 0
    elif damage == 'ratio': row['energy_normalization_repeats'][0]['host_normalized_energy_per_work_unit_est_j'] *= 1.01
    elif damage == 'aggregate': row['host_normalized_energy_per_work_est_j'] *= 1.01
    elif damage == 'idle': row['energy_normalization_repeats'][0]['accelerator_idle_w_applied'] = 1
    elif damage == 'count': row['energy_repeat_valid_n'] = 2
    assert resolve_energy_comparison(row)['energy_comparison_status'] == 'required_tensorrt_full_normalized_metrics_invalid'


@pytest.mark.parametrize('authority, errors', [(None,['native_split_quality_authority_missing']),
    ({'valid':True,'mode':'required','errors':[]},[]),
    ({'valid':False,'mode':'invalid','errors':['binding_conflict']},['binding_conflict'])])
def test_current_authority_does_not_invent_missing_error(authority, errors):
    row={'backend':'deepx_to_trt','native_split_quality_authority_errors':['old_error']}
    apply_native_split_quality_authority(row,authority)
    assert row['native_split_quality_authority_errors']==errors
    assert row['native_split_quality_authority_previous_errors']==['old_error']


def test_model_reference_is_shared_before_full_and_three_split_fanout(tmp_path):
    from scripts.native_full_baseline_eval_runner import _resolve_image
    model='renamed_classifier'; cases=['boundary_z','boundary_b','boundary_c']; bs=tmp_path/'suite';bs.mkdir()
    for i,case in enumerate(cases):
        image=bs/f'input_{i}.jpg';image.write_bytes(bytes([i]))
        report=tmp_path/'models'/model/'benchmark_results/remote_diagnostics/case_reports/results'/case/'results_ort_cpu/validation_report.json'
        report.parent.mkdir(parents=True);report.write_text(json.dumps({'run_cfg':{'image':str(image)}}))
    images,sources=build_native_validation_image_map(tmp_path,[model],{model:cases},{model:bs})
    assert set(images[model])==set(cases+['full'])
    assert len(set(images[model].values()))==1
    assert len(set(sources[model].values()))==1
    assert _resolve_image(bs,model,'unselected_container',images)[0]==bs/'input_0.jpg'


def trt_case(tmp_path, monkeypatch):
    from test_v269d_trt_quality_first_runtime import _write_quality_producer_set
    from test_v2711_direct_bn6_producer_binding import _ns
    from test_v270i_p0_completed_endpoint import _write_runtime_input
    from scripts import native_full_baseline_eval_runner as full, native_trt_full_completed_hotloop as hotloop
    bs, _, producer = _write_quality_producer_set(tmp_path,model='renamed_classifier')
    ns=_ns(setup_id='renamed_setup',comparison_backend='ort_tensorrt');ns.timeout=30
    output=full._native_full_dump_dir(bs,'renamed_classifier','native_full_tensorrt',ns);output.mkdir(parents=True)
    manifest, tensor, _ = _write_runtime_input(output)
    image=bs/'image.jpg';image.write_bytes(b'input')
    meta=output/'native_trt_meta.json';meta.write_text(json.dumps({
        'onnx':producer['build_onnx']['path'],'engine_build_receipt_path':producer['engine_build_receipt']['path'],
        'engine_build_receipt':producer['engine_build_receipt']['receipt'],
        'run_smoke':{'returncode':0,'cmd':[producer['trtexec']['path'],'--loadEngine='+producer['engine']['path'],'--iterations=3','--warmUp=0']}}))
    instances=[]; commands=[]
    class TRT:
        inputs=['images']; shapes={'images':(1,3,4,4)}; dtypes={'images':np.dtype('float32')}
        def __init__(self,path): self.n=0;self.closed=False;instances.append(self)
        def prepare_inputs(self,feeds): assert feeds['images'].shape==(1,3,4,4)
        def run_prepared(self):
            self.n+=1;time.sleep(.002)
            return {'logits':np.array([[0.,float(self.n),.5,1.,2.,3.]])}
        def close(self): self.closed=True
    monkeypatch.setattr(hotloop,'NativeTRT',TRT)
    def run(command,**kwargs):
        commands.append(command);monkeypatch.setattr(sys,'argv',command[1:])
        return {'rc':hotloop.main(),'timed_out':False}
    monkeypatch.setattr(full,'_run',run)
    row=dict(ok=True,status='ok',task='classification',contract_family='classification_logits',
        backend='native_full_tensorrt',model='renamed_classifier',setup_id='renamed_setup',comparison_backend='ort_tensorrt',
        input_case='full',input_image=str(image),input_image_sha256=full._sha256_file(image),input_manifest=str(manifest),
        report=str(meta),frames=3,quality_first_producer_identity=producer,
        quality_first_producer_identity_sha256=producer['producer_identity_sha256'])
    row=full._attach_trt_completed_task_hotloop(row,bs,'renamed_classifier',ns)
    assert row['ok'],row.get('error')
    contract=full._full_command_contract(row=row,root=bs.parent.parent,benchmark_set=bs,
        model='renamed_classifier',backend_arg='tensorrt',ns=ns)
    return full,hotloop,bs,ns,row,contract,commands,instances


@pytest.mark.parametrize('damage',[None,'missing_postprocess','bad_topk'])
def test_real_trt_performance_contract_preflight_and_duration_energy_dispatch(tmp_path,monkeypatch,damage,capsys):
    from scripts import native_producer_energy_plan as plan
    full,hotloop,bs,ns,row,contract,commands,instances=trt_case(tmp_path,monkeypatch)
    assert contract['complete'],contract['energy_workload']
    assert contract['energy_workload']['kind']=='tensorrt_full_completed_task_hotloop'
    verified,status=plan._verify_full_command_contract(contract,expected_identity={'backend':'native_full_tensorrt','model':'renamed_classifier'})
    assert verified is not None,status
    ns.root=str(bs.parent.parent);ns.energy_command_contract_file='';ns.energy_command_contract_json=json.dumps(contract)
    ns.preflight_nonce='local-test';ns.preflight_attestation_max_age_s=300
    ns.preflight_attestation_out=str(tmp_path/'preflight.json');ns.preflight_attestation=ns.preflight_attestation_out
    assert full._energy_preflight_only(ns)==0
    ns.out_dir=str(tmp_path/'energy');ns.duration_s=.015
    if damage=='missing_postprocess':
        real=hotloop.ClassificationCompletion.report
        def incomplete(self,completed=None):
            result=real(self,completed);result['postprocess_completed_frames']=0;return result
        monkeypatch.setattr(hotloop.ClassificationCompletion,'report',incomplete)
    elif damage=='bad_topk':
        import onnx_splitpoint_tool.runners.harness.classification as c
        monkeypatch.setattr(c,'classification_topk',lambda *a,**k: (_ for _ in ()).throw(ValueError('invalid live logits')))
    capsys.readouterr()
    result=full._energy_workload_only(ns);stdout=capsys.readouterr().out
    assert result==(0 if damage is None else 5)
    assert all('--classification' in c for c in commands)
    assert len({Path(c[1]).name for c in commands})==1
    assert all(i.closed for i in instances)
    if damage is None:
        payload=json.loads((Path(ns.out_dir)/'native_full_energy_hotloop.json').read_text())
        assert payload['completed_work_units']==payload['postprocess_completed_frames']>=3
        assert payload['minimum_duration_satisfied'] is True
        assert payload['classification_topk']['top1']==[1]
        assert '__SPLITPOINT_WORK_UNITS_EXACT__=1' in stdout
    else: assert '__SPLITPOINT_WORK_UNITS_EXACT__=1' not in stdout


def test_generic_energy_off_is_not_missing_and_never_borrows_native(tmp_path):
    from onnx_splitpoint_tool.workflow.results import _augment_results_with_target_energy_v59j
    (tmp_path/'profile.yaml').write_text('energy:\n  enabled: false\n  generic_enabled: false\n  measurement_path: native_only\nnative_producers:\n  energy:\n    enabled: true\n')
    rows,summary=_augment_results_with_target_energy_v59j([{'backend':'example','variant':'split','case_id':'b009'}],model_id='renamed',run_root=tmp_path)
    assert rows[0]['energy_coverage_status']=='not_requested'
    assert rows[0]['energy_enabled'] is False
    assert summary['missing_split_energy_count']==0
    assert 'energy_total_j' not in rows[0]


@pytest.mark.parametrize('damage',[None,'missing','source','tamper'])
def test_detection_completion_projection_verifies_actual_execution_attestation(damage):
    from test_v272_detection_completion_runtime import _decoded_outputs,_runtime
    from onnx_splitpoint_tool.native_energy_reporting import _completed_detection_energy_endpoint
    outputs=_decoded_outputs();runtime=_runtime(outputs);runtime.process(outputs);a=runtime.attestation()
    contract=a['completed_task_comparison_endpoint_contract'];source=runtime.execution_contract['source_endpoint']
    plan=dict(completion_pairing_eligible=True,endpoint_contract_complete=True,output_endpoint_match=True,
        completion_pairing_status='strict_completed_detection_endpoint_verified',comparison_output_endpoint_id=contract['output_endpoint_id'],
        comparison_endpoint_contract_hash=contract['endpoint_contract_hash'],comparison_endpoint_stage='decoded_nms',
        output_endpoint_id=contract['output_endpoint_id'],endpoint_contract_hash=contract['endpoint_contract_hash'],endpoint_stage='decoded_nms',
        physical_endpoint_contract_hash=source['endpoint_contract_hash'],physical_output_endpoint_id=source['output_endpoint_id'])
    validation=dict(completed_task_stage='decoded_nms',completed_task_contract_family='decoded_nms',
        completed_task_endpoint_attested=True,completed_task_endpoint_attestation_status='passed',completed_task_endpoint_attestation=a,
        completed_task_comparison_endpoint_contract_hash=contract['endpoint_contract_hash'],completed_task_comparison_output_endpoint_id=contract['output_endpoint_id'],
        completed_task_comparison_endpoint_contract=contract,completed_task_completion_mode='detection_completion_execution_v1',
        completion_execution_contract=runtime.execution_contract)
    if damage=='missing':validation.pop('completion_execution_contract')
    elif damage=='source':plan['physical_endpoint_contract_hash']='f'*64
    elif damage=='tamper':validation['completed_task_endpoint_attestation']['completed_work_units']=99
    endpoint,_,_,status=_completed_detection_energy_endpoint(plan,validation,physical_match=True)
    assert bool(endpoint)==(damage is None),status


@pytest.mark.parametrize('backend',['deepx','h10','h8_python'])
@pytest.mark.parametrize('duration',[0,.015])
def test_classification_fifo_duration_drains_topk_before_counting(backend,duration):
    from test_v272_detection_hotloop_integration import _FakeTRT,_FakeDeepX,_FakeHailo10Session
    from scripts import native_hailo10_trt_e2e_from_benchmarkset as h10,native_deepx_trt_e2e_from_benchmarkset as dx,native_hailo_trt_fifo_from_benchmarkset as h8
    from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
    class TRT(_FakeTRT):
        def run_prepared(self):time.sleep(.001);return {'logits':np.array([[0.,4.,2.]])}
    opts=dict(frames=3,warmup=2,queue_depth=1,duration_s=duration)
    if backend=='deepx':r=dx._deepx_fifo_run(_FakeDeepX(),np.zeros((1,4)),TRT(),**opts)
    elif backend=='h10':r=h10._hailo10_async_fifo_run(_FakeHailo10Session(),{'input':np.zeros((1,4))},TRT(),inflight=3,**opts)
    else:
        class Hailo:
            def run(self,*args):return SimpleNamespace(outputs={'boundary':np.zeros((1,4),np.float32)})
        r=h8._hailo8_python_fifo_run(Hailo(),None,{'input':np.zeros((1,4))},TRT(),
            completion_runtime=ClassificationCompletion(),warmup_completion_runtime=ClassificationCompletion(),**opts)
    assert r['completed_frames']==r['postprocess_completed_frames']>=3
    assert r['classification_topk']['top1']==[1]


def test_three_backend_cases_real_generator_and_full_deduplication(tmp_path):
    from test_v283_r9a_backfill import (onnx,helper,TensorProto,analyze_model,BenchmarkGenerationRuntime,
        BenchmarkGenerationExecutionConfig,BenchmarkGenerationExecutionCallbacks,BenchmarkGenerationExecutionService,
        DEFAULT_BACKFILL,bind_plan_cases,expected_profile_measurements_v60r,negative)
    source=tmp_path/'renamed.onnx'
    nodes=[helper.make_node('Identity',['x' if i==0 else f'v{i}'],[f'v{i+1}']) for i in range(8)]
    graph=helper.make_graph(nodes,'chain',[helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,3,4,4])],
        [helper.make_tensor_value_info('v8',TensorProto.FLOAT,[1,3,4,4])])
    onnx.save(helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]),source)
    analysis=analyze_model(str(source));suite=tmp_path/'suite';suite.mkdir()
    runs=[{'id':f'{b}_to_tensorrt','stage1':{'provider':b},'stage2':{'provider':'tensorrt'}} for b in ('hailo8','hailo10','deepx_m1')]
    runtime=BenchmarkGenerationRuntime(suite,suite/'log',suite/'generation_state.json',3,[2,3,4],[2,3,4,5,6],'renamed',str(source),'end')
    def leaf(path,**kw):
        b=kw['build_evidence_context']['boundary'];arch=kw['hw_arch']
        if b==2 and arch=='hailo8':return SimpleNamespace(ok=False,skipped=True,timed_out=False,hef_path=None,
            failure_kind='known_negative_build_evidence',unsupported_reason='COMPILE_INFEASIBLE',error='exact negative',elapsed_s=0,
            details={'build_evidence':negative()},calib_info={})
        artifact=Path(kw['outdir'])/'part1.hef';artifact.parent.mkdir(parents=True,exist_ok=True);artifact.write_bytes(b'controlled cached artifact')
        return SimpleNamespace(ok=True,skipped=False,timed_out=False,hef_path=artifact,failure_kind='',error='',elapsed_s=0,
            details={'cache_hit':True},calib_info={})
    cfg=BenchmarkGenerationExecutionConfig(runtime=runtime,target_cases=3,gap=0,ranked_candidates=[2,3,4],candidate_search_pool=[2,3,4,5,6],
        out_dir=suite,base='renamed',pad=3,strict_boundary=False,model=analysis['model'],nodes=analysis['nodes'],order=analysis['order'],
        analysis_payload=analysis,full_model_src=str(source),require_single_part2_input=True,hef_targets=['hailo8','hailo10h'],hef_part1=True,
        bench_plan_runs=runs,hailo_build_hef_fn=leaf,backend_backfill_policy={**DEFAULT_BACKFILL,'max_candidates_per_backend':12})
    cb=BenchmarkGenerationExecutionCallbacks(log=lambda *a,**k:None,queue_put=lambda *a:None,persist_state=runtime.persist,
        publish_hailo_diagnostics=lambda *a,**k:None,predicted_metrics_for_boundary=lambda *a:{},hailo_parse_entry_for_boundary=lambda *a:None,
        hailo_parse_scalar_fields=lambda *a:{})
    BenchmarkGenerationExecutionService().execute_case_build_loop(cfg,cb)
    state=json.loads(runtime.state_path.read_text())['backend_backfill']
    selections={c['backend']:c['selected_case_ids'] for c in state['contracts']}
    assert selections=={'hailo8':['b003','b004','b005'],'hailo10h':['b002','b003','b004'],'deepx':['b002','b003','b004']}
    plan={'runs':runs+[{'id':'hailo8_full'},{'id':'ort_tensorrt','variants':['full']}]};bind_plan_cases(plan,state)
    matrix=expected_profile_measurements_v60r(model_id='renamed',benchmark_plan=plan,benchmark_set_contract={'cases':runtime.cases})
    assert len([r for r in matrix if r['variant']=='split'])==9
    assert len([r for r in matrix if r['variant']=='full'])==2
    assert state['cold_builds_started']==0


@pytest.mark.parametrize('bad',[False,True])
def test_deepx_full_energy_cli_uses_real_topk_with_prepared_feed(tmp_path,monkeypatch,bad):
    import socket,hashlib,types
    from test_v275_deepx_sealed_runtime_input import _sealed_input_fixture
    from scripts import native_deepx_full_energy_hotloop as hotloop
    manifest,_,_,feed=_sealed_input_fixture(tmp_path);m=json.loads(manifest.read_text())
    tensor=Path(m['runtime_input_file']);dxnn=tmp_path/'existing.dxnn';dxnn.write_bytes(b'existing artifact')
    runner=Path(hotloop.__file__).resolve();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    now=time.time_ns();preflight=tmp_path/'preflight.json'
    proof=dict(schema=hotloop.PREFLIGHT_SCHEMA,schema_version=1,ok=True,nonce='r9h',command_contract_sha256='a'*64,
        artifact_verification_status='pass',host=socket.gethostname(),created_at_unix_ns=now,expires_at_unix_ns=now+60_000_000_000,
        verified_artifact_sha256=dict(hotloop_runner=sha(runner),dxnn=sha(dxnn),runtime_input_tensor=sha(tensor)))
    proof['attestation_sha256']=hotloop._canonical_json_sha256(proof);preflight.write_text(json.dumps(proof))
    calls=[]
    class Engine:
        def __init__(self,path):assert path==str(dxnn)
        def run(self,feeds):
            np.testing.assert_array_equal(feeds[0],feed);calls.append(1);time.sleep(.001)
            return [np.array([[0.,float('nan') if bad else 7.,2.,3.,4.,1.]])]
    module=types.ModuleType('dx_engine');module.InferenceEngine=Engine;monkeypatch.setitem(sys.modules,'dx_engine',module)
    opts={'dxnn':dxnn,'prepared-input-file':tensor,'prepared-feed-contract-version':hotloop.CONTRACT_VERSION,
        'expected-prepared-input-path':tensor,'expected-prepared-input-root':tmp_path,
        'expected-prepared-input-sha256':sha(tensor),'expected-prepared-input-bytes':m['runtime_input_bytes'],
        'expected-prepared-input-name':m['runtime_input_name'],'expected-prepared-input-shape-json':json.dumps(m['runtime_input_shape']),
        'expected-prepared-input-dtype':m['runtime_input_dtype'],'expected-prepared-input-layout':m['runtime_input_layout'],
        'runtime-preprocessing-identity-json':json.dumps(m['runtime_preprocessing_identity']),
        'expected-runtime-preprocessing-sha256':m['runtime_preprocessing_sha256'],
        'runtime-numeric-input-identity-json':json.dumps(m['runtime_numeric_input_identity']),
        'expected-runtime-numeric-input-sha256':m['runtime_numeric_input_sha256'],'original-image-wh-json':'[2,2]',
        'task':'classification','frames':3,'warmup':2,'duration-s':.01,'json-out':tmp_path/'report.json',
        'expected-runner-sha256':sha(runner),'expected-runner-path':runner,'expected-runner-root':runner.parent,
        'expected-dxnn-sha256':sha(dxnn),'expected-dxnn-path':dxnn,'expected-dxnn-root':tmp_path,
        'source-contract-sha256':'a'*64,'preflight-attestation':preflight,'preflight-nonce':'r9h'}
    monkeypatch.setattr(sys,'argv',[str(runner)]+[part for k,v in opts.items() for part in ['--'+k,str(v)]])
    rc=hotloop.main();payload=json.loads((tmp_path/'report.json').read_text())
    assert rc==(5 if bad else 0),payload
    if bad:assert not payload['ok'] and len(calls)==1
    else:
        assert payload['classification_topk']['top1']==[1]
        assert payload['completed_work_units']==payload['postprocess_completed_frames']==len(calls)-2
        assert payload['minimum_duration_satisfied'] and payload['task_complete']


def test_generic_normalization_preserves_declared_producer_without_claim_promotion(tmp_path):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
    producer={'schema':'producer','endpoint':{'identity':{'task':'classification','stage':'classification_logits'},'sha256':'a'*64}}
    row=dict(variant='full',primary_variant='full',backend='deepx_m1',candidate_execution_contract=producer,candidate_execution_contract_sha256='b'*64)
    normalized=normalize_benchmark_row(row,model_id='renamed',source_path=tmp_path/'benchmark_results.json')
    assert normalized['candidate_execution_contract']==producer
    assert normalized['candidate_execution_contract_sha256']=='b'*64
    assert normalized['endpoint_contract_complete'] is False


def test_native_unrequested_energy_has_explicit_display():
    from onnx_splitpoint_tool.workflow.status_reporting import energy_axis_description
    assert energy_axis_description({'energy':{'requested':False}})=='not_requested'


def load_postcheck():
    import os
    operator=Path(os.environ.get('R9H_OPERATOR_DIR','/home/kmika/.local/share/onnx-splitpoint-codex/v283_R9H_20260918_233209_748qhzjo/operator'))
    if not (operator/'postcheck.py').is_file():pytest.skip('R9H host operator is an external task artifact')
    spec=importlib.util.spec_from_file_location('r9h_postcheck',operator/'postcheck.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('damage',[None,'missing','trtexec','counter','postprocess','source'])
def test_postcheck_distinguishes_measured_defects_from_unknown(damage):
    checker=load_postcheck();repeat={'energy_work_units_used':17,'energy_observed_work_units_exact':True}
    proof=dict(task='classification',completed_task_stage='classification_top1_top5',completed_work_units=17,
        postprocess_completed_frames=17,postprocess_completion_verified=True,source_contract_sha256='a'*64)
    if damage=='trtexec':repeat['energy_observed_work_units_source']='verified_full_energy_hotloop:tensorrt_full_hotloop'
    if damage=='counter':proof['completed_work_units']=18
    if damage=='postprocess':proof['postprocess_completed_frames']=16
    if damage=='source':proof['source_contract_sha256']='b'*64
    text='__SPLITPOINT_WORK_UNITS__=17\n__SPLITPOINT_WORK_UNITS_EXACT__=1\n'+json.dumps(proof)
    if damage=='missing':text=''
    defects,unknown=checker.check_task_evidence('classification',repeat,text,{'contract_sha256':'a'*64})
    assert bool(defects)==(damage not in {None,'missing'})
    assert bool(unknown)==(damage=='missing')


def test_postcheck_cli_writes_unknown_without_inventing_defects(tmp_path):
    import subprocess,os
    checker=load_postcheck();run=tmp_path/'empty_run';run.mkdir();out=tmp_path/'check'
    result=subprocess.run([sys.executable,checker.__file__,'--run',str(run),'--out',str(out)],capture_output=True,text=True,
        timeout=30,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==1
    payload=json.loads((out/'POSTCHECK.json').read_text())
    assert payload['pass'] is False and payload['defect_ids']==[]
    assert payload['details']['status']=='inconclusive'


@pytest.mark.parametrize('phase',['smoke','eval'])
def test_host_profile_roundtrip_scope_and_snapshot(phase):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    checker=load_postcheck();profile=load_evaluation_profile(Path(checker.__file__).parent.parent/(phase.upper()+'.yaml'))
    checker.validate_scope(profile.raw_profile,phase)
    assert profile.start_snapshot['resolved_selection']['selection_policy']['max_accepted_cases_per_model']==(3 if phase=='eval' else 1)


@pytest.mark.parametrize('damage',[None,'missing_repeat','count','tamper','source'])
def test_fast_detection_projection_requires_each_fresh_bound_energy_repeat(damage):
    from test_v27925_fast_oracle_validation import _archived
    from onnx_splitpoint_tool.native_energy_reporting import _completed_detection_energy_endpoint
    validation=_archived();execution=validation['completion_execution_contract'];source=execution['source_endpoint'];comp=execution['comparison_endpoint_contract']
    plan=dict(completion_pairing_eligible=True,endpoint_contract_complete=True,output_endpoint_match=True,
        completion_pairing_status='strict_fast_postflight_completion_verified',comparison_output_endpoint_id=comp['output_endpoint_id'],
        comparison_endpoint_contract_hash=comp['endpoint_contract_hash'],comparison_endpoint_stage='decoded_nms',
        output_endpoint_id=comp['output_endpoint_id'],endpoint_contract_hash=comp['endpoint_contract_hash'],endpoint_stage='decoded_nms',
        physical_endpoint_contract_hash=source['endpoint_contract_hash'],physical_output_endpoint_id=source['output_endpoint_id'])
    repeats=[{'logical_repeat_index':i,'energy_work_units_used':3000} for i in range(3)]
    proofs=[dict(logical_repeat_index=i,status='fresh_energy_completion_nonce_count_and_window_verified',preflight_nonce=str(i),
        stdout_sha256='a'*64,comparison_output_endpoint_id=comp['output_endpoint_id'],completed_work_units=3000,
        completion_execution_attestation=deepcopy(validation['completed_task_endpoint_attestation']),
        workload_timing=dict(status='ok',rc=0,start_ns=1,end_ns=2)) for i in range(3)]
    if damage=='missing_repeat':proofs.pop()
    elif damage=='count':proofs[0]['completed_work_units']=1
    elif damage=='tamper':proofs[0]['completion_execution_attestation']['completion_count']=1
    elif damage=='source':plan['physical_endpoint_contract_hash']='b'*64
    endpoint,_,_,status=_completed_detection_energy_endpoint(plan,validation,physical_match=True,fresh_completions=proofs,energy_repeats=repeats)
    assert bool(endpoint)==(damage is None),status


def test_output_contract_and_quality_display_are_separate():
    from onnx_splitpoint_tool.native_rate_endpoints import format_rate_endpoints
    text=format_rate_endpoints({'semantic_status':'claim_ok','task_quality_status':'fail'})
    assert 'Output: Ausgabevertrag erfüllt' in text and 'Taskqualität: fail' in text
    assert 'Output/Qualität: claim_ok' not in text


@pytest.mark.parametrize('task', ['classification', 'detection'])
@pytest.mark.parametrize('changed_image', [False, True])
def test_deepx_full_dispatch_prepares_model_reference_before_all_repetitions(
    tmp_path, monkeypatch, task, changed_image,
):
    from PIL import Image
    from test_v27930_native_full_semantic_merge import prepare_runner_case, runner
    output = np.array([[1., 2., 3., 7., 0., 4.]], np.float32) if task == 'classification' else None
    case = prepare_runner_case(tmp_path, monkeypatch, model='renamed_model', task=task, output=output)
    (case.root/'benchmark_plan.json').write_text(json.dumps({'runs':[{'id':'deepx_m1_full','task':task}]}))
    (case.root/'b003').rename(case.root/'boundary_renamed')
    old_manifest = case.root/'results/deepx_m1_full/prepared_input/native_full_input_manifest.json'
    old_bytes = old_manifest.read_bytes()
    if changed_image:
        case.image = case.root/'comparison.png'
        Image.new('RGB', (640, 480), color=(77, 43, 21)).save(case.image)
    case.ns.image_map_data = {case.model:{'full':str(case.image), 'boundary_renamed':str(case.image)}}
    actual_run = runner._run
    def child(command, **kwargs):
        if 'native_full_semantic_dump.py' in str(command[1]):
            assert Path(command[command.index('--image')+1]) == case.image
        return actual_run(command, **kwargs)
    monkeypatch.setattr(runner, '_run', child)
    semantic, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is None, blocked
    case.ns.deepx_full_precomputed_semantic = semantic
    rows = [runner._row_for_backend(case.root, case.model, 'deepx', case.ns) for _ in range(3)]
    assert all(row['ok'] for row in rows), rows
    assert case.processes == ['semantic', 'performance', 'performance', 'performance']
    assert old_manifest.read_bytes() == old_bytes
    assert all(row['input_image_sha256'] == runner._sha256_file(case.image) for row in rows)
    from onnx_splitpoint_tool.runners.native_full_input import load_sealed_deepx_native_full_input
    prepared = load_sealed_deepx_native_full_input(
        Path(semantic['input_manifest']), image_path=case.image,
        input_contract=case.contract, task=task, expected_model=case.model,
        expected_setup_id=case.setup_id, expected_comparison_backend='deepx',
    )
    assert len(case.calls) > 3
    assert all(np.array_equal(feed, prepared['runtime_input']) for feed in case.calls)


@pytest.mark.parametrize('runtime, technical, wanted', [
    (True, 'ok', ('terminal_completion', 'pass')),
    (True, 'partial', ('terminal_completion', 'pass')),
    (False, 'partial', ('terminal_evidence_missing', 'unknown')),
    (None, 'partial', ('terminal_evidence_missing', 'unknown')),
])
def test_postcheck_terminal_is_independent_of_quality_or_technical_negatives(runtime, technical, wanted):
    checker = load_postcheck()
    assert checker.terminal_check({'runtime_complete':runtime, 'technical_status':technical,
        'completion':{'status':'partial','quality_warning':True}}) == wanted


@pytest.mark.parametrize('damage', [None, 'unresolved', 'duplicate', 'imported', 'unexplained', 'invalid_index'])
def test_postcheck_rejected_energy_preserves_actual_failure_and_repeat_accounting(damage):
    checker = load_postcheck()
    execution = {'energy_aggregate_import_status':'rejected_fail_closed'}
    aggregate = {'terminal':True, 'execution_status':'INCOMPLETE', 'runs':[{'logical_repeat_index':0}],
        'error':'recorded_transport_error', 'task_budget':{'chains':[{'collector_started':True,'source_completion_verified':True}]}}
    if damage == 'unresolved':aggregate['task_budget']['chains'][0]['source_completion_verified'] = False
    elif damage == 'duplicate':aggregate['runs'].append({'logical_repeat_index':0})
    elif damage == 'imported':execution['eligible_for_energy_results_import'] = True
    elif damage == 'unexplained':aggregate.pop('terminal')
    elif damage == 'invalid_index':aggregate['runs'][0]['logical_repeat_index'] = None
    failed, checks = checker.energy_attempt_check(execution, aggregate, 3)
    assert failed
    defects = {code for code, status, _ in checks if status == 'defect'}
    expected = {None:set(), 'unresolved':{'energy_source_completion_unresolved'},
        'duplicate':{'energy_repetitions_incomplete_or_duplicate'}, 'imported':{'failed_energy_attempt_imported'},
        'unexplained':{'energy_repetitions_incomplete_or_duplicate'}, 'invalid_index':{'energy_repetitions_incomplete_or_duplicate'}}
    assert defects == expected[damage]
    proof = {'completed_task_stage':'classification_top1_top5','completed_work_units':760,
        'postprocess_completed_frames':760,'postprocess_completion_verified':True,'source_contract_sha256':'a'*64}
    stdout = '__SPLITPOINT_WORK_UNITS__=760\n__SPLITPOINT_WORK_UNITS_EXACT__=1\n'+json.dumps(proof)
    assert checker.check_task_evidence('classification', {'energy_observed_work_units':760,
        'energy_observed_work_units_exact':True}, stdout, {'contract_sha256':'a'*64}) == ([], [])


@pytest.mark.parametrize('damage', ['missing_image', 'invalid_contract', 'symlink_output'])
def test_deepx_model_reference_failure_never_reuses_old_generic_feed(tmp_path, monkeypatch, damage):
    from test_v27930_native_full_semantic_merge import prepare_runner_case, runner
    case = prepare_runner_case(tmp_path, monkeypatch)
    case.ns.image_map_data = {case.model:{'full':str(case.image)}}
    old_manifest = case.root/'results/deepx_m1_full/prepared_input/native_full_input_manifest.json'
    old_bytes = old_manifest.read_bytes()
    if damage == 'missing_image':
        case.ns.image_map_data[case.model]['full'] = 'missing.png'
        (case.root/'test_image.png').write_bytes(case.image.read_bytes())
    elif damage == 'invalid_contract':(case.full/'output_contract.json').write_text('{}')
    else:
        output = runner._native_full_dump_dir(case.root, case.model, 'native_full_deepx', case.ns)
        output.parent.mkdir(parents=True, exist_ok=True)
        outside = tmp_path/'outside';outside.mkdir();output.symlink_to(outside)
    monkeypatch.setattr(runner, '_run', lambda *_a, **_k: pytest.fail('no runtime on invalid preparation'))
    _, blocked = runner._deepx_full_series_preflight(case.root, case.model, case.ns)
    assert blocked is not None
    assert blocked['repetition_count_attempted'] == 0
    assert 'deepx_comparison_input' in blocked['failure_reason']
    assert old_manifest.read_bytes() == old_bytes


@pytest.mark.parametrize('damage', [None, 'command', 'template', 'attestation', 'source', 'missing'])
def test_postcheck_binds_actual_workload_and_artifact_source(tmp_path, damage):
    checker = load_postcheck()
    from onnx_splitpoint_tool.energy.collector import _render_energy_preflight_template, write_command_script
    template = 'runner --preflight-nonce __ONNX_SPLITPOINT_PREFLIGHT_NONCE__ --preflight-attestation __ONNX_SPLITPOINT_PREFLIGHT_ATTESTATION__'
    plan_file = tmp_path/'planned.sh';plan_file.write_text(template+'\n')
    attestation = tmp_path/'preflight.json';attestation.write_text('{}')
    rendered = _render_energy_preflight_template(template, nonce='nonce', repeat_index=0, attestation_path='/remote/proof.json')
    workload = write_command_script(tmp_path/'workload_command.sh', rendered)
    (tmp_path/'energy_command.sh').write_text(str(workload)+' > workload_stdout.log')
    import hashlib
    preflight = dict(nonce='nonce',repeat_index=0,runtime_attestation_path='/remote/proof.json',
        workload_template_sha256=hashlib.sha256(template.encode()).hexdigest(),
        rendered_workload_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
        expected_command_contract_sha256='a'*64,attestation_path=str(attestation),
        attestation_file_sha256=checker.digest(attestation))
    plan = dict(command_file=str(plan_file),successful_command_contract_sha256='a'*64)
    if damage == 'command':workload.write_text(workload.read_text()+'wrong_runner\n')
    elif damage == 'template':plan_file.write_text(template+' --foreign-artifact')
    elif damage == 'attestation':attestation.write_text('{"tampered":true}')
    elif damage == 'source':plan['successful_command_contract_sha256'] = 'b'*64
    elif damage == 'missing':workload.unlink()
    defects, unknown = checker.check_command_evidence(plan, {'preflight_evidence':preflight}, tmp_path)
    assert bool(defects) == (damage not in {None, 'missing'})
    assert bool(unknown) == (damage == 'missing')


@pytest.mark.parametrize('state', ['clean', 'quality_fail', 'source_unresolved', 'quarantine', 'unreadable'])
def test_gui_cleanup_does_not_confuse_terminal_workers_with_energy_source_completion(tmp_path, state):
    checker = load_postcheck()
    spec = importlib.util.spec_from_file_location('r9h_gui_operator', Path(checker.__file__).with_name('gui_eval.py'))
    gui = importlib.util.module_from_spec(spec);spec.loader.exec_module(gui)
    budget = {'sources':{'source_a':{'stop_reason':''}}}
    if state == 'source_unresolved':budget['sources']['source_a']['stop_reason'] = 'campaign_source_completion_unresolved'
    path = tmp_path/'energy_task_budget.json';path.write_text(json.dumps(budget))
    if state == 'unreadable':path.write_text('{')
    if state == 'quarantine':
        (tmp_path/'jobs').mkdir();(tmp_path/'jobs/test_unresolved_cleanup_quarantine.json').write_text('{}')
    if state == 'quality_fail':
        (tmp_path/'reports').mkdir();(tmp_path/'reports/run_status_summary.json').write_text('{"quality_decision":"fail"}')
    assert bool(gui.unresolved_cleanup_reasons(tmp_path)) == (state not in {'clean', 'quality_fail'})
    checked = checker.check_run(tmp_path)
    assert ('energy_source_completion_unresolved' in checked['defect_ids']) == (state == 'source_unresolved')
    assert ('energy_source_cleanup_evidence_unreadable' in checked['details']['unknown_ids']) == (state == 'unreadable')


def split_budget_state(remaining=12, maximum=12):
    from onnx_splitpoint_tool.backend_backfill import DEFAULT_BACKFILL
    cases = ['boundary_z', 'boundary_a', 'boundary_m']
    contracts = [dict(stage='part1', setup_id=setup, selected_case_ids=cases,
                      run_id=run, backend=backend)
                 for setup, run, backend in [('device_c', 'ort_tensorrt', 'tensorrt'),
                     ('device_c', 'deepx_m1_to_tensorrt', 'deepx'),
                     ('device_b', 'hailo8_to_trt', 'hailo8'),
                     ('device_a', 'hailo10_to_tensorrt', 'hailo10h')]]
    return dict(enabled=True, policy={**DEFAULT_BACKFILL, 'max_cold_builds':24,
                'max_trt_part2_builds':maximum}, cold_builds_started=24-remaining, contracts=contracts)


@pytest.mark.parametrize('remaining,maximum,expected', [
    (23, 12, [4, 4, 4]), (8, 12, [3, 3, 2]), (1, 12, [1, 0, 0]),
    (0, 12, [0, 0, 0]), (23, 2, [1, 1, 0]),
])
def test_eval_budget_is_shared_across_distinct_setups_without_starvation(remaining, maximum, expected):
    from onnx_splitpoint_tool.backend_backfill import bind_plan_cases
    state = split_budget_state(remaining, maximum);before = deepcopy(state);plan = {}
    bind_plan_cases(plan, state)
    assert plan['backend_backfill_build_budget']['trt_starts_by_setup'] == dict(zip(['device_c','device_b','device_a'], expected))
    assert state == before
    assert sum(expected) <= min(remaining, maximum)


def test_eval_three_split_budget_reaches_real_start_reservations_and_stays_bounded(tmp_path):
    from onnx_splitpoint_tool.backend_backfill import bind_plan_cases
    from scripts.native_trt_from_benchmarkset import _run
    plan = {};state = split_budget_state();bind_plan_cases(plan, state)
    started = 0
    for setup, maximum in plan['backend_backfill_build_budget']['trt_starts_by_setup'].items():
        folder = tmp_path/setup;folder.mkdir();checkpoint = folder/'native_trt_build_state.json'
        kwargs = dict(cwd=folder, log_path=folder/'child.log', dry_run=False,
            build_state=str(checkpoint), max_build_starts=maximum, timeout_s=2,
            artifact_role='part2', case_id='')
        for case_id in ['boundary_z','boundary_a','boundary_m','reserve_case']:
            result = _run([sys.executable, '-c', 'print("controlled local child")'], **{**kwargs,'case_id':case_id})
            assert result['returncode'] == 0, result
            started += 1
        blocked = _run([sys.executable, '-c', 'raise SystemExit(99)'], **kwargs)
        assert blocked['status'] == 'build_budget_exhausted' and blocked['compiler_dispatched'] is False
        assert len(json.loads(checkpoint.read_text())['starts']) == 4
    assert started == 12


def prepared_quality_binding(case, row):
    """Seal a controlled quality producer using the actual prepared runtime feed."""
    from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256 as seal
    join = dict(schema='onnx-splitpoint/deepx-performance-quality-input-binding', schema_version=1,
        binding_verified=True, source_image_id=row['prepared_input_source_image_id'],
        source_image_sha256=row['prepared_input_source_image_sha256'],
        runtime_preprocessing_sha256=row['runtime_preprocessing_sha256'],
        runtime_numeric_input_sha256=row['runtime_numeric_input_sha256'])
    for key in ('prepared_input_sha256','prepared_input_bytes','prepared_input_name','prepared_input_shape',
                'prepared_input_dtype','prepared_input_layout'):
        join[key] = row[key]
    binding = dict(schema='onnx-splitpoint/native-full-quality-request-binding', schema_version=1,
        model_id=case.model, backend='native_full_deepx', task=case.task, variant='full', setup_id=case.setup_id,
        comparison_backend='deepx', source_run_id='deepx_m1_full', source_case_id='full',
        prepared_input_join_binding=join, prepared_input_join_binding_sha256=seal(join),
        preprocessing_contract={'identity':row['runtime_preprocessing_identity'],'sha256':row['runtime_preprocessing_sha256']},
        preprocessing_contract_sha256=row['runtime_preprocessing_sha256'],
        runtime_precision_identity='deepx_dxnn_sha256:'+case.runner._sha256_file(case.full/'model.dxnn'),
        endpoint_contract_hash=row['endpoint_contract_hash'], model_sha256=case.contract['source_onnx_sha256'])
    for field in ('source_request_sha256','validation_dataset_sha256','validation_dataset_image_ids_sha256',
                  'validation_dataset_ground_truth_sha256','task_quality_policy_sha256','quality_contract_sha256',
                  'quality_record_endpoint_contract_sha256','central_quality_result_sha256',
                  'prepared_input_evidence_sha256','decoder_contract_sha256','nms_contract_sha256'):
        binding[field] = seal({'controlled_quality_field':field})
    binding['binding_sha256'] = seal(binding)
    return binding


@pytest.mark.parametrize('task', ['classification','detection'])
@pytest.mark.parametrize('damage', [None,'missing','changed_bytes','symlink','conflict','unsealed','tensor_conflict'])
def test_eval_bound_reference_fanout_preserves_exact_deepx_quality_join(tmp_path, monkeypatch, task, damage):
    from PIL import Image
    from test_v27930_native_full_semantic_merge import prepare_runner_case, runner
    from test_v2711_direct_bn6_producer_binding import _ns
    from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256 as seal
    values = np.array([[1.,2.,3.,7.,0.,4.]], np.float32) if task == 'classification' else None
    case = prepare_runner_case(tmp_path, monkeypatch, model='renamed_classifier' if task=='classification' else 'renamed_yolo11', task=task, output=values)
    case.runner = runner
    declarations_path = case.root/'output_contracts.json'
    declarations = json.loads(declarations_path.read_text())
    declarations.update(model_id=case.model, task=task)
    for declaration in declarations['contracts']:
        declaration.update(model_id=case.model, task=task)
        if task == 'classification':
            declaration.update(stage='classification_logits', contract_family='classification_logits',
                               endpoint_mode='classification_logits', output_format='classification_logits',
                               host_tail_required=False, postprocessing_required=False,
                               requires_external_postprocess=False)
    declarations_path.write_text(json.dumps(declarations))
    if task == 'classification':
        from onnx_splitpoint_tool.runners.native_full_input import prepare_and_seal_deepx_native_full_input
        case.contract['input'].update(preprocess_mode='resize', letterbox_pad_value=0)
        (case.full/'output_contract.json').write_text(json.dumps(case.contract))
        prepare_and_seal_deepx_native_full_input(image_path=case.image,input_contract=case.contract,task=task,
            out_dir=case.root/'results/deepx_m1_full/prepared_input',model=case.model,setup_id=case.setup_id,comparison_backend='deepx')
    (case.root/'benchmark_plan.json').write_text(json.dumps({'runs':[{'id':'deepx_m1_full','task':task}]}))
    original = runner._row_for_backend(case.root, case.model, 'deepx', case.ns)
    assert original['ok'], original
    binding = prepared_quality_binding(case, original);bindings = [binding]
    visual = case.root/'different_visual.png';Image.new('RGB',(640,480),(99,12,87)).save(visual)
    cases = ['boundary_z','boundary_a','boundary_m']
    report = tmp_path/'models'/case.model/'benchmark_results/remote_diagnostics/case_reports/results'/cases[0]/'results_ort_cpu/validation_report.json'
    report.parent.mkdir(parents=True);report.write_text(json.dumps({'run_cfg':{'image':str(visual)}}))
    before = deepcopy(binding)
    if damage == 'missing':case.image.unlink()
    elif damage == 'changed_bytes':case.image.write_bytes(b'changed')
    elif damage == 'symlink':
        saved = tmp_path/'outside.png';saved.write_bytes(case.image.read_bytes());case.image.unlink();case.image.symlink_to(saved)
    elif damage == 'conflict':
        extra = deepcopy(binding);extra['prepared_input_join_binding']['source_image_sha256'] = 'b'*64
        extra['prepared_input_join_binding_sha256'] = seal(extra['prepared_input_join_binding'])
        extra['binding_sha256'] = seal({k:v for k,v in extra.items() if k!='binding_sha256'});bindings.append(extra)
    elif damage == 'unsealed':binding['binding_sha256'] = 'a'*64
    elif damage == 'tensor_conflict':
        binding['prepared_input_join_binding']['prepared_input_sha256'] = 'b'*64
        binding['prepared_input_join_binding_sha256'] = seal(binding['prepared_input_join_binding'])
        binding['binding_sha256'] = seal({k:v for k,v in binding.items() if k!='binding_sha256'})
    if damage in {'missing','changed_bytes','symlink','conflict','unsealed'}:
        count = len(case.processes)
        with pytest.raises(ValueError, match='native_reference_prepared_'):
            build_native_validation_image_map(tmp_path,[case.model],{case.model:cases},{case.model:case.root},prepared_input_bindings=bindings)
        assert len(case.processes) == count
        return
    images, sources = build_native_validation_image_map(tmp_path,[case.model],{case.model:cases},{case.model:case.root},prepared_input_bindings=bindings)
    assert set(images[case.model]) == set(cases+['full'])
    for case_id in cases+['full']:
        assert runner._resolve_image(case.root,case.model,case_id,images)[0] == case.image
    assert len(set(sources[case.model].values())) == 1
    case.ns.image_map_data = images
    semantic, blocked = runner._deepx_full_series_preflight(case.root,case.model,case.ns)
    assert blocked is None
    case.ns.deepx_full_precomputed_semantic = semantic
    row = runner._row_for_backend(case.root,case.model,'deepx',case.ns)
    ns = _ns();vars(ns).update(vars(case.ns))
    ns.quality_request_binding_set_data = {'bindings_by_backend_model':{'native_full_deepx|'+case.model:binding},'binding_set_sha256':'c'*64}
    contract = runner._full_command_contract(row=row,root=case.root.parent,benchmark_set=case.root,model=case.model,backend_arg='deepx',ns=ns)
    row, status = runner._attach_full_quality_request_binding(row,contract,model=case.model,ns=ns)
    if damage == 'tensor_conflict':
        assert status == 'quality_request_binding_failed'
        assert 'quality_performance_prepared_input_sha256_mismatch' in row['quality_request_binding_errors']
    else:
        assert status == 'quality_request_binding_verified_exact', row.get('quality_request_binding_errors')
        assert row['quality_request_binding'] == before
        assert row['prepared_input_sha256'] == original['prepared_input_sha256']
        assert binding == before


@pytest.mark.parametrize('damage,wanted', [(None,('terminal_completion','pass')),
    ('active',('run_not_terminal','defect')), ('running',('run_not_terminal','defect')),
    ('foreign',('terminal_run_identity_conflict','defect')), ('missing',('terminal_evidence_missing','unknown'))])
def test_eval_terminal_control_is_independent_of_failed_coverage(damage, wanted):
    checker = load_postcheck()
    control = {'state':'finished','run_id':'renamed_run'}
    jobs = {'active_jobs':[],'running_job_ids':[],'run_id':'renamed_run'}
    if damage == 'active':jobs['active_jobs'] = [{'id':'pending'}]
    elif damage == 'running':control['state'] = 'running'
    elif damage == 'foreign':jobs['run_id'] = 'different_run'
    elif damage == 'missing':jobs = {}
    assert checker.terminal_check({'run_id':'renamed_run','runtime_complete':False,
        'completion':{'status':'partial','quality_warning':True}}, control, jobs) == wanted


@pytest.mark.parametrize('kind', ['quality_fail','screening','binding_error','native_failed','native_missing'])
def test_eval_postcheck_separates_technical_binding_failures_from_scientific_negatives(tmp_path, kind):
    checker = load_postcheck();reports = tmp_path/'reports';reports.mkdir()
    validation = reports/'native_validation';validation.mkdir()
    row = {'model':'renamed_model','backend':'native_full_deepx','case':'full','ok':True,
           'quality_first_binding_errors':[],'technical_quality_error':False,
           'task_quality_status':'fail' if kind=='quality_fail' else 'screening_only'}
    if kind == 'binding_error':row.update(technical_quality_error=True,quality_first_binding_errors=['quality_performance_source_image_sha256_mismatch'])
    (validation/'native_producer_validation_summary.json').write_text(json.dumps({'rows':[row]}))
    (reports/'native_expected_matrix.json').write_text(json.dumps({'expected_row_count':1,
        'execution_terminal_complete':False,'failed_expected_row_count':int(kind=='native_failed'),
        'missing_expected_row_count':int(kind=='native_missing')}))
    result = checker.check_run(tmp_path)
    assert result['defect_ids'] == {'quality_fail':[],'screening':[],
        'binding_error':['native_quality_binding_failed'],'native_failed':['native_execution_failed'],
        'native_missing':['native_plan_rows_missing']}[kind]
