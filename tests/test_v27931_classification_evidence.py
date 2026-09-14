from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from onnx_splitpoint_tool.deepx.preprocessing_probe import (
    classification_input_tensor, compare_logical_input, compare_prediction_errors,
    load_probe_samples, resolve_source_build_semantics, run_paired_probe,
    select_classification_logits, topk,
)
from onnx_splitpoint_tool.runners.harness.classification import ClassificationHarness
from onnx_splitpoint_tool.quality_metrics import classification_quality_evaluator

ROOT=Path(__file__).resolve().parents[1]
FIX=ROOT/'tests/fixtures/v27931_classification'
EVIDENCE=json.loads((FIX/'original_evidence.json').read_text())
MODELS=tuple(EVIDENCE['models'])


def evidence(model):return EVIDENCE['models'][model]


@pytest.mark.parametrize('model',MODELS)
def test_t07_6_original_contract_graph_and_numeric_conflict(model):
    m=evidence(model);r=resolve_source_build_semantics(model_id=model,graph_observation=m['graph'],output_contract=m['source_contract'],profile_mode='current_scale_only',export_metadata=m['export_metadata'])
    assert r['source_numeric_mode']=='imagenet_mean_std'
    assert r['reasons']==['deepx_legacy_classification_preprocessing']
    assert r['spatial_hash_is_numeric_proof'] is False
    assert r['logical_input_status']=='reconstructed' and not r['hardware_executed']
    assert r['source_graph_embedded_normalization'] is False
    if model=='resnet50':assert m['export_metadata']=={} and m['graph']['metadata']['mean']=='[0.485, 0.456, 0.406]'


@pytest.mark.parametrize('problem',['metadata_missing','graph_unknown','wrong_model','wrong_source','profile_conflict','double_norm','already_embedded'])
def test_t07_7_explicit_graph_constants_not_names(problem):
    m=copy.deepcopy(evidence(MODELS[0]));model=MODELS[0];mode='current_scale_only'
    if problem=='metadata_missing':m['graph']['metadata']={};m['export_metadata']={}
    if problem=='graph_unknown':m['graph']['input_dataflow']=[{'depth':0,'op_type':'CustomFused','name':'normalize'}]
    if problem=='wrong_model':model='resnet50'
    if problem=='wrong_source':m['source_contract']['source_onnx_sha256']=''
    if problem=='profile_conflict':mode='imagenet_mean_std'
    if problem in {'double_norm','already_embedded'}:
        m['graph']['input_dataflow']=[{'depth':0,'op_type':'Sub','inputs':['input','m'],'outputs':['sub'],'small_initializer_values':{'m':[.485,.456,.406]}},{'depth':1,'op_type':'Div','inputs':['sub','s'],'outputs':['norm'],'small_initializer_values':{'s':[.229,.224,.225]}}]
        if problem=='double_norm':mode='imagenet_mean_std';m['source_contract']['classification_preprocessing']=mode
    if problem in {'wrong_model','wrong_source'}:
        with pytest.raises(ValueError):resolve_source_build_semantics(model_id=model,graph_observation=m['graph'],output_contract=m['source_contract'],profile_mode=mode,export_metadata=m['export_metadata'])
    else:
        r=resolve_source_build_semantics(model_id=model,graph_observation=m['graph'],output_contract=m['source_contract'],profile_mode=mode,export_metadata=m['export_metadata'])
        if problem=='already_embedded':assert r['source_numeric_mode']=='scale_only' and not r['reasons']
        else:assert r['status']=='mismatch'


@pytest.mark.parametrize('model',MODELS)
def test_t07_17_18_36_r1_actual_raw_topk_records_remain_exact(model):
    m=evidence(model);harness=ClassificationHarness(labels=[str(i) for i in range(1000)])
    refs=[];cands=[]
    with np.load(FIX/'original_R1_R2_logits.npz',allow_pickle=False) as payload:
        for i,row in enumerate(m['quality_rows']):
            raw=payload[f'{model}_{i:03d}_native'];quality=payload[f'{model}_{i:03d}_quality'];before=raw.copy()
            assert np.array_equal(raw,quality)
            assert row['native_feed_sha256']==row['quality_feed_sha256']
            label=row['independent']['label_id'];bound=row['independent']['output_name']
            logits=select_classification_logits({bound:raw},output_name=bound,class_count=1000)
            independent=topk(logits);product=harness.postprocess({bound:raw},{})
            ids=[int(x['id']) for x in product.json['topk']]
            old=row['product_quality_record']
            assert independent[0]==ids[0]==old['top1']
            assert set(independent)==set(ids)==set(old['top5'])
            assert np.array_equal(raw,before)
            refs.append({'image_id':row['image_id'],'label_id':label,'reference':{'top1_hit':independent[0]==label,'top5_hit':label in independent}})
            cands.append({'image_id':row['image_id'],'label_id':label,'candidate':{'top1_hit':old['top1_correct'],'top5_hit':old['top5_correct']}})
    ev=classification_quality_evaluator(refs,cands,[],{'metric_gate_config':{'non_inferiority_margin':.01,'guardrails':{'top5_accuracy_margin':.01}}})
    result=ev.evaluate(np.ones(16))
    assert result['primary']['delta']==0 and result['guardrails']['top5_accuracy']['delta']==0
    assert result['primary']['margin']==.01


@pytest.mark.parametrize('model',MODELS[:2])
def test_t07_37_original_r2_changes_numeric_arm_without_b500_claim(model):
    m=evidence(model);r=m['r2'];hit_a=hit_b=0
    with np.load(FIX/'original_R1_R2_logits.npz',allow_pickle=False) as p:
        def cos(a,b):return np.dot(a.ravel().astype(np.float64),b.ravel())/(np.linalg.norm(a.astype(np.float64))*np.linalg.norm(b.astype(np.float64)))
        for i,row in enumerate(m['quality_rows']):
            a=p[f'{model}_{i:03d}_r2_a'];b=p[f'{model}_{i:03d}_r2_b'];scale=p[f'{model}_{i:03d}_cpu_scale'];mean=p[f'{model}_{i:03d}_cpu_mean'];label=row['independent']['label_id']
            assert np.array_equal(a,p[f'{model}_{i:03d}_native'])
            assert cos(a,scale)>cos(a,mean) and cos(b,mean)>cos(b,scale)
            hit_a+=int(np.argmax(a)==label);hit_b+=int(np.argmax(b)==label)
    assert (hit_a,hit_b)==((11,12) if model=='mobilenet_v3_large' else (12,15))
    assert r['adapter_receipt']['source_onnx_sha256']!=r['adapter_receipt']['build_onnx_sha256']


@pytest.mark.parametrize('corruption',['bgr','double_scale','double_norm','batch','strides','negative_uint8'])
def test_t07_2_9_12_physical_to_logical_mapping_detects_corruption(corruption):
    u=np.ascontiguousarray(np.arange(5*7*3).reshape(5,7,3),dtype=np.uint8)
    f=classification_input_tensor(u,mode='imagenet_mean_std',layout='NCHW')
    assert compare_logical_input(u,f,mode='imagenet_mean_std')['consistent']
    if corruption=='bgr':f=classification_input_tensor(np.ascontiguousarray(u[...,::-1]),mode='imagenet_mean_std',layout='NCHW')
    elif corruption=='double_scale':f=f/np.float32(255)
    elif corruption=='double_norm':f=(f-np.array([.485,.456,.406],np.float32)[None,:,None,None])/np.array([.229,.224,.225],np.float32)[None,:,None,None]
    elif corruption=='batch':f=np.repeat(f,2,axis=0)
    elif corruption=='strides':f=f[:,:,:,::-1]
    elif corruption=='negative_uint8':
        with pytest.raises(ValueError,match='physical image'):classification_input_tensor(f,mode='imagenet_mean_std',layout='NCHW')
        return
    if corruption in {'batch','strides'}:
        with pytest.raises(ValueError):compare_logical_input(u,f,mode='imagenet_mean_std')
    else:assert not compare_logical_input(u,f,mode='imagenet_mean_std')['consistent']


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_t07_15_16_17_named_outputs_finite_class_axis(bad):
    good=np.arange(1000,dtype=np.float32)[None];aux=-good
    assert topk(select_classification_logits([aux,good],output_names=['aux','logits'],output_name='logits',class_count=1000))[0]==999
    with pytest.raises(ValueError,match='unambiguous'):select_classification_logits([aux,good])
    with pytest.raises(ValueError,match='class count'):select_classification_logits({'logits':np.zeros((1,1001),np.float32)},output_name='logits',class_count=1000)
    with pytest.raises(ValueError):select_classification_logits({'logits':np.zeros((2,1000),np.float32)},output_name='logits')
    good[0,3]=bad
    with pytest.raises(ValueError,match='nonfinite'):select_classification_logits({'logits':good},output_name='logits')
    with pytest.raises(ValueError,match='nonfinite'):topk(good)


def test_t07_17_ties_are_explicit_diagnostic_oracle_policy():
    assert topk(np.ones(1000,np.float32))==[0,1,2,3,4]
    # Existing product tie ordering is deliberately not changed for diagnosis.
    assert 'Stable ordering' in __import__('inspect').getsource(topk)


@pytest.mark.parametrize('model',MODELS)
def test_t07_13_14_actual_cpu_ort_ordered32_and_production_feed(tmp_path,model):
    import onnx
    from onnx import helper,TensorProto,numpy_helper
    graph=helper.make_graph([helper.make_node('Flatten',['input'],['flat']),helper.make_node('MatMul',['flat','w'],['logits'])],'explicitly_synthetic_CPU_diagnostic',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,3,2,2])],[helper.make_tensor_value_info('logits',TensorProto.FLOAT,[1,6])],[numpy_helper.from_array(np.arange(72,dtype=np.float32).reshape(12,6)/np.float32(100),'w')])
    m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]);m.ir_version=10;p=tmp_path/'synthetic.onnx';onnx.save(m,p)
    rows=[]
    for i in range(40):
        im=tmp_path/f'{i:03d}.png';Image.new('RGB',(7,9),(i,200-i,i*3)).save(im);rows.append({'image':im.name,'sample_id':im.name,'label_id':i%6})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'samples':rows}))
    result=run_paired_probe(model_path=p,manifest_path=manifest,expected_images=40,limit=32,model_id=model,output_name='logits',class_count=6,source_numeric_mode='imagenet_mean_std',raw_logits_path=tmp_path/'logits.npz')
    assert result['dataset']['sample_count']==32 and result['dataset']['source_sample_count']==40
    assert result['dataset']['selected_image_ids']==[r['image'] for r in rows[:32]]
    assert result['providers_effective']==['CPUExecutionProvider'] and result['model_acceptance']=='NOT_EVALUATED'
    assert result['counts_as_benchmark'] is False and result['hardware_executed'] is False
    assert all(r['production_reference']['top5']==r['modes']['imagenet_mean_std']['top5'] for r in result['records'])
    with np.load(tmp_path/'logits.npz',allow_pickle=False) as raw:assert len(raw.files)==96
    assert not list(tmp_path.glob('*.dxnn'))


@pytest.mark.parametrize('corruption',['duplicate','label','order','missing'])
def test_t07_14_ordered_manifest_never_partial_or_label_guess(tmp_path,corruption):
    for i in range(3):(tmp_path/f'{i}.jpg').write_bytes(b'identity-only')
    rows=[{'image':f'{i}.jpg','label_id':i} for i in range(3)];expected=[r['image'] for r in rows]
    if corruption=='duplicate':rows[1]['image']='0.jpg'
    if corruption=='label':rows[1]['label_id']=True
    if corruption=='order':rows.reverse()
    if corruption=='missing':(tmp_path/'1.jpg').unlink()
    p=tmp_path/'manifest.json';p.write_text(json.dumps({'samples':rows}))
    with pytest.raises((ValueError,FileNotFoundError)):load_probe_samples(p,limit=2,expected_image_ids=expected)


def test_t07_20_similar_accuracy_different_errors_is_no_cause():
    full=[{'image_id':'a','label_id':1,'top1':0},{'image_id':'b','label_id':1,'top1':1}]
    split=[{'image_id':'a','label_id':1,'top1':1},{'image_id':'b','label_id':1,'top1':0}]
    r=compare_prediction_errors(full,split)
    assert r['shared_error_ids']==[] and r['common_cause_proven'] is False
    assert compare_prediction_errors(full,None)['full_available'] is True


def test_t07_21_23_24_calibration_evidence_does_not_become_hardware_truth():
    m=evidence(MODELS[0]);r=resolve_source_build_semantics(model_id=MODELS[0],graph_observation=m['graph'],output_contract=m['source_contract'],profile_mode='current_scale_only')
    assert r['logical_input_status']=='reconstructed' and r['vendor_internal_input_status']=='unresolved'
    # A successful compiler or a cache identity cannot erase a different numerical model.
    c=copy.deepcopy(m['source_contract']);c.update(compiler_cuda_probe='PASS',cache_hit=True)
    assert resolve_source_build_semantics(model_id=MODELS[0],graph_observation=m['graph'],output_contract=c,profile_mode='current_scale_only')['reasons']==r['reasons']


def test_t07_4_18_product_postprocess_export_central_loader_evaluator(tmp_path):
    from tests.test_deepx_full_only_quality_dispatch import _classification_semantic
    from tests.test_v269a_deepx_full_central_quality import _base_tree,_suite_module,_run,_runner_functions,RUNNER
    from onnx_splitpoint_tool.quality_service import quality_request_from_manifest,prepare_evaluation,_evaluate_payload,QualityArtifactIntegrityError
    model='resnet50'; row=evidence(model)['quality_rows'][0];label=row['independent']['label_id']
    suite,source,dxnn=_base_tree(tmp_path,task='classification',samples=[{'image':'a.jpg','label_id':label}])
    semantic,prepared=_classification_semantic(suite)
    with np.load(FIX/'original_R1_R2_logits.npz',allow_pickle=False) as p:raw=p[f'{model}_000_native'].copy()
    logits=select_classification_logits({'logits':raw},output_name='logits',class_count=1000)
    product=ClassificationHarness(labels=[str(i) for i in range(1000)]).postprocess({'logits':raw},{})
    top=[int(v['id']) for v in product.json['topk']];oracle=topk(logits)
    assert top[0]==oracle[0] and set(top)==set(oracle)
    record_path=suite/semantic['classification_topk_json'];record=json.loads(record_path.read_text());r=record['images'][0]
    r.update(label_id=label,label_name=str(label),top1=top[0],top5=top,scores=[float(logits[i]) for i in top],top1_correct=top[0]==label,top5_correct=label in top)
    record_path.write_text(json.dumps(record))
    exported=_suite_module()._deepx_export_central_quality_request(suite,dxnn,_run('classification'),semantic,suite/'results/deepx_m1_full',completed_task_evidence=prepared)
    request_path=Path(exported['request']['path']);candidate_path=request_path.parent/exported['candidate']['path'];candidate=json.loads(candidate_path.read_text())
    namespace=_runner_functions(['_quality_json_safe','_quality_file_sha256','_quality_contract_sha256','_portable_dataset_manifest_sha256','_build_classification_quality_contract'])
    contract=namespace['_build_classification_quality_contract'](model_path=source,validation_source=suite/'validation',rows=[{'image_id':'a.jpg','label_id':label}],image_scale='imagenet',letterbox=False,input_hw=(224,224),input_dtype=np.float32,runner_path=RUNNER,endpoint_attestor_sha256='a'*64)
    reference={'schema':'onnx-splitpoint/task-quality-reference-input','schema_version':1,'task':'classification','pairing_key':'image_id','reference_role':'canonical_cpu_ort','semantic_reference_only':True,'provenance_required':True,'quality_contract':contract,'quality_contract_sha256':contract['quality_contract_sha256'],'records':[{'image_id':'a.jpg','label_id':label,'reference':{'top1':oracle[0],'top5':oracle,'top1_hit':oracle[0]==label,'top5_hit':label in oracle}}]}
    reference_path=tmp_path/'diagnostic_reference.json';reference_path.write_text(json.dumps(reference))
    loaded=quality_request_from_manifest(request_path,reference_artifact=reference_path)
    assert loaded.candidate_records[0]['candidate']['top1']==oracle[0]
    _,payload=prepare_evaluation(loaded);result=_evaluate_payload(payload)
    assert result['primary']['delta']==0
    # This executes ordinary producer/loader/evaluator bytes on a one-sample
    # diagnostic; the fixture is deliberately not a new B500 reference.
    assert len(loaded.candidate_records)==1
    candidate['records'][0]['label_id']=(label+1)%1000;candidate_path.write_text(json.dumps(candidate))
    with pytest.raises(QualityArtifactIntegrityError):quality_request_from_manifest(request_path,reference_artifact=reference_path)


def test_t07_11_actual_deepx_session_reports_selected_candidate_without_retry():
    import ast,typing
    p=ROOT/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'
    tree=ast.parse(p.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepXSession')
    env={'np':np,'Path':Path,'__name__':'synthetic_actual_call_fixture','_try_adapt_tensor':lambda a,s:(True,a),**vars(typing)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[])),str(p),'exec'),env)
    session=object.__new__(env['DeepXSession']);u=np.arange(27,dtype=np.uint8).reshape(3,3,3)
    session._input_names=['input'];session._output_names=['logits'];session.input_order_override=[];session.output_shapes={'logits':(1,1000)};session.dxnn_path=Path('/explicit_synthetic_engine.dxnn')
    session._source_for_input=lambda inputs,name,used:inputs[name];session._is_image_runtime_input=lambda name:True
    session.runtime_input_shapes={'input':(1,3,3,3)}
    session._should_use_feature_subprocess=lambda:False
    calls=[];session._engine=SimpleNamespace(run=lambda feeds:(calls.append(feeds[0].copy()) or [np.zeros((1,1000),np.float32)]))
    observations=[];session._diagnostic_observer=observations.append
    normalized=classification_input_tensor(u,mode='imagenet_mean_std',layout='NCHW')
    session.infer({'input':normalized});assert len(calls)==1 and observations[0]['candidate_index']==0
    expected=np.ascontiguousarray(np.clip((normalized[0]*np.array([.229,.224,.225],np.float32)[:,None,None]+np.array([.485,.456,.406],np.float32)[:,None,None])*255,0,255).astype(np.uint8).transpose(1,2,0))
    assert np.array_equal(observations[0]['feeds'][0],expected)
    assert not np.array_equal(expected,u)  # Observed roundtrip loss; never hidden as an exact feed.
    def broken_observer(row):raise ValueError('diagnostic observation failed')
    session._diagnostic_observer=broken_observer
    with pytest.raises(ValueError):session.infer({'input':normalized})
    assert len(calls)==2


def test_t07_26_existing_adapter_actual_ort_parity_and_original_unchanged(tmp_path):
    import onnx,onnxruntime as ort
    from onnx import helper,TensorProto
    from onnx_splitpoint_tool.deepx.config import materialize_imagenet_normalized_build_onnx
    graph=helper.make_graph([helper.make_node('Identity',['input'],['output'])],'synthetic_identity_norm_oracle',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,3,4,5])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,3,4,5])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',18)]);model.ir_version=10
    source=tmp_path/'source.onnx';onnx.save(model,source);before=source.read_bytes()
    legacy=tmp_path/'old.dxnn';legacy.write_bytes(b'explicit synthetic legacy artifact; preserve')
    build,receipt=materialize_imagenet_normalized_build_onnx(source_onnx=source,output_dir=tmp_path/'new_adapter')
    assert source.read_bytes()==before and legacy.read_bytes()==b'explicit synthetic legacy artifact; preserve'
    u=np.arange(60,dtype=np.uint8).reshape(4,5,3)
    scale=classification_input_tensor(u,mode='current_scale_only',layout='NCHW');norm=classification_input_tensor(u,mode='imagenet_mean_std',layout='NCHW')
    a=ort.InferenceSession(str(source),providers=['CPUExecutionProvider']).run(None,{'input':norm})[0]
    b=ort.InferenceSession(str(build),providers=['CPUExecutionProvider']).run(None,{'input':scale})[0]
    np.testing.assert_array_equal(a,b)
    assert receipt['source_onnx_sha256']!=receipt['build_onnx_sha256']


def test_t07_5_historical_b500_loss_and_ap75_guardrail_remain_fail():
    p=ROOT/'tests/fixtures/v27931_quality/legacy_complete_set_quality.json';before=p.read_bytes();original=json.loads(before)
    mobile=original['mobilenet_full_point_fail'];yolo=original['yolo26s_guardrail_fail']
    assert mobile['decision']=='fail' and mobile['primary']['delta']==pytest.approx(-.066)
    assert mobile['primary']['reference']==pytest.approx(.736) and mobile['primary']['candidate']==pytest.approx(.67)
    assert yolo['decision']=='fail'
    assert yolo['guardrails']['ap75']['decision']=='fail'
    assert p.read_bytes()==before
