from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types
import venv
import zipfile
import pytest
import numpy as np
from PIL import Image
SOURCE=Path(__file__).resolve().parents[1]
ROOT=SOURCE/'scripts/classification_probe_v27931'
sys.path.insert(0,str(ROOT/'lib'))
import smoke_common as c
import collect_smokes as collect
import cpu_worker
import remote_worker
import analyze_evidence


def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return path


def fixture_run(tmp_path,count=3):
    run=tmp_path/'run';model='mobilenet_v3_large';suite=run/f'models/{model}/benchmark_set/legacy_suite';suite.mkdir(parents=True)
    collect.copy_source_snapshot(suite,SOURCE,suite)
    onnx=tmp_path/'model.onnx';onnx.write_text('{"synthetic_graph":true}')
    dx=suite/'deepx/deepx_m1/full/model.dxnn';dx.parent.mkdir(parents=True);dx.write_bytes(b'synthetic DXNN fixture; NEVER a real compiled model')
    authority={'model_id':model,'backend':'deepx_m1','variant':'full','contract_status':'recorded','endpoint_mode':'decoded',
               'host_tail_required':False,'postprocessing_required':False,'task':'classification'}
    put(suite/'output_contracts.json',{'model_id':model,'contracts':[authority]})
    contract={'model_id':model,'backend':'deepx_m1','variant':'full','endpoint_mode':'decoded','host_tail_required':False,'postprocessing_required':False,
              'classification_preprocessing':'current_scale_only','source_onnx_sha256':c.digest(onnx),'build_onnx_sha256':c.digest(onnx),'artifact_sha256':c.digest(dx),'suite_artifact_sha256':c.digest(dx),
              'source_model_input':{'name':'input','layout':'NCHW','shape':[1,3,224,224],'dtype':'float32'},
              'outputs':[{'name':'logits','dtype':'float32','shape':None}],
              'input':{'name':'input','shape':[224,224,3],'dtype':'uint8','layout':'HWC','color_space':'RGB',
                       'normalization':'embedded_dxcom_preprocessing','preprocess_mode':'resize','letterbox_pad_value':0}}
    put(dx.with_name('output_contract.json'),contract)
    samples=[]
    for i in range(count):
        a=np.zeros((91+i*7,127+i*9,3),dtype=np.uint8)
        yy,xx=np.indices(a.shape[:2]);a[...,0]=(xx*3+i)%256;a[...,1]=(yy*7+19)%256;a[...,2]=(xx+yy*2)%256
        image=suite/f'data/images/{i:06d}.png';image.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(a).save(image)
        samples.append({'image':'images/'+image.name,'label_id':i,'sha256':c.digest(image)})
    put(suite/'data/manifest.json',{'samples':samples})
    row={'id':'deepx_m1_full','setup_id':'fixture_deepx','benchmark_task':'classification','validation_images':'data','validation_max_images':500,
         'contract_path':'deepx/deepx_m1/full/output_contract.json'}
    put(suite/'benchmark_plan.json',{'runs':[row]})
    put(run/f'models/{model}/model_manifest.json',{'resolved_path':str(onnx),'profile_entry':{}})
    put(run/f'models/{model}/benchmark_results/quality_inputs/fixture_deepx/results/deepx_m1_full/task_quality_inputs/full_request.json',{'expected_image_ids':[Path(r['image']).name for r in samples]})
    target={'id':'fixture_deepx','accelerator':'deepx_m1','enabled':True,'runtime':{'host':'fixturehost','user':'nx','port':22},
            'build_environment':{'runtime_venv':str(tmp_path/'venv'),'compiler_venv':str(tmp_path/'absent_compiler')}}
    put(run/'hardware_matrix.json',{'hardware_targets':[target]})
    args=types.SimpleNamespace(cpu_samples=count,hardware_samples=min(2,count),tool_dir=SOURCE)
    out=tmp_path/'model_evidence';out.mkdir()
    stage,request=collect.prepare_model(run,model,target,{'runtime_venv':str(tmp_path/'venv')},args,out)
    return types.SimpleNamespace(run=run,model=model,suite=suite,stage=stage,request=request,out=out,target=target,args=args,onnx=onnx,dx=dx)


def fake_logits(data):
    x=np.asarray(data,dtype=np.float32);v=float(x.mean())
    a=np.arange(1000,dtype=np.float32);return (np.sin(a*.173+v*3)+a*.0001)[None].astype(np.float32)


def fake_dx_module():
    m=types.ModuleType('dx_engine');m.__file__='SYNTHETIC_TEST_ENGINE_ONLY'
    class Engine:
        def __init__(self,path):self.path=path
        def run(self,feeds):
            assert len(feeds)==1 and feeds[0].shape==(224,224,3) and feeds[0].dtype==np.uint8
            return [fake_logits(feeds[0].astype(np.float32)/np.float32(255))]
    m.InferenceEngine=Engine;return m


def fake_ort_module():
    m=types.ModuleType('onnxruntime');m.__version__='SYNTHETIC_TEST_ENGINE_ONLY'
    class Options:pass
    class Session:
        def __init__(self,*args,**kw):assert kw['providers']==['CPUExecutionProvider']
        def get_inputs(self):return [types.SimpleNamespace(name='input',shape=[1,3,224,224],type='tensor(float)')]
        def get_outputs(self):return [types.SimpleNamespace(name='logits',shape=[1,1000])]
        def get_providers(self):return ['CPUExecutionProvider']
        def run(self,names,feeds):return [fake_logits(next(iter(feeds.values())))]
    m.SessionOptions=Options;m.InferenceSession=Session;return m


@pytest.fixture(autouse=True)
def restore_isolated_imports():
    original={k:v for k,v in sys.modules.items() if k=='onnx_splitpoint_tool' or k.startswith('onnx_splitpoint_tool.') or k=='splitpoint_runners' or k.startswith('splitpoint_runners.')}
    before=list(sys.meta_path); paths=list(sys.path)
    yield
    for k in list(sys.modules):
        if k=='onnx_splitpoint_tool' or k.startswith('onnx_splitpoint_tool.') or k=='splitpoint_runners' or k.startswith('splitpoint_runners.'):
            sys.modules.pop(k,None)
    sys.modules.update(original);sys.meta_path[:]=before;sys.path[:]=paths


def test_t07_27_source_staging_uses_current_product(tmp_path):
    f=fixture_run(tmp_path)
    assert all(r['origin']=='installed_tool' for r in c.read_json(f.out/'resolution.json')['source_provenance'])
    assert c.digest(f.stage/'suite/run_split_onnxruntime.py')==c.digest(SOURCE/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt')


@pytest.mark.parametrize('value',['','bad','a'*63,'g'*64,'sha256:123'])
def test_bad_hash_rejected(value):
    with pytest.raises(ValueError):c.sha_token(value)


def test_exact_hash_fallback(tmp_path):
    a=tmp_path/'a';b=tmp_path/'b';a.write_bytes(b'wrong');b.write_bytes(b'correct')
    p,rows=c.exact_artifact([a,b],c.digest(b));assert p==b and len(rows)==2


def test_artifact_mismatch_no_arbitrary_fallback(tmp_path):
    a=tmp_path/'a';a.write_bytes(b'wrong')
    with pytest.raises(ValueError):c.exact_artifact([a],'1'*64)


def test_selection_original_order_and_exact_bytes(tmp_path):
    f=fixture_run(tmp_path)
    assert [x['image_id'] for x in f.request['samples']]==['000000.png','000001.png','000002.png']
    assert all(s['original_content_hash_verified'] for s in f.request['samples'])
    assert c.digest(f.stage/'suite/deepx/deepx_m1/full/model.dxnn')==c.digest(f.dx)
    assert c.digest(f.onnx)==f.request['expected_onnx_sha256']


def test_hash_mismatch_sample_is_not_skipped(tmp_path):
    p=tmp_path/'i.png';Image.new('RGB',(3,3)).save(p)
    man=put(tmp_path/'manifest.json',{'samples':[{'image':p.name,'label_id':0,'sha256':'1'*64}]})
    with pytest.raises(ValueError,match='digest_mismatch'):collect.select_samples([man],[p.name])


def test_missing_label_not_guessed(tmp_path):
    p=tmp_path/'i.png';Image.new('RGB',(3,3)).save(p)
    man=put(tmp_path/'manifest.json',{'samples':[{'image':p.name}]})
    with pytest.raises(ValueError,match='label_missing'):collect.select_samples([man],[p.name])


def test_duplicate_sample_rejected(tmp_path):
    man=put(tmp_path/'manifest.json',{'samples':[{'image':'a/i.png','label_id':1},{'image':'b/i.png','label_id':2}]})
    with pytest.raises(ValueError,match='duplicate_image'):collect.sample_rows(man)


@pytest.mark.parametrize('value',[-1,1000,True,'1'])
def test_bad_label_rejected(tmp_path,value):
    p=tmp_path/'i.png';Image.new('RGB',(3,3)).save(p)
    man=put(tmp_path/'manifest.json',{'samples':[{'image':p.name,'label_id':value}]})
    with pytest.raises(ValueError,match='zero_based'):collect.select_samples([man],[p.name])


def test_lock_refuses_active_workflow(tmp_path):
    p=tmp_path/'lock'
    with collect.workflow_gate(p):
        with pytest.raises(RuntimeError,match='active'):
            with collect.workflow_gate(p):pass
    assert p.exists()


def test_lock_preserves_existing_metadata(tmp_path):
    p=tmp_path/'lock';p.write_text('preserve me');st=p.stat()
    with collect.workflow_gate(p):pass
    assert p.read_text()=='preserve me' and p.stat().st_ino==st.st_ino and p.stat().st_mtime_ns==st.st_mtime_ns


@pytest.mark.parametrize('p',['/tmp/other-1234567890','/tmp/onnx-deepx-classification-v27931-123/../../home','/tmp/onnx-deepx-classification-v27931-abc','/home/nx'])
def test_no_cleanup_outside_allocated_prefix(p):assert not collect.remote_path_ok(p)


@pytest.mark.parametrize('shape',[(2,1000),(1000,2),(1,1001),(1,84,8400)])
def test_classification_ambiguous_shape_rejected(shape):
    with pytest.raises(ValueError):c.logits_record([np.zeros(shape)],['logits'],0)


def test_nonfinite_logits_rejected():
    a=np.zeros((1,1000));a[0,30]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):c.logits_record([a],['logits'],0)


def test_no_extra_sigmoid_or_score_clipping():
    a=np.arange(-500,500,dtype=np.float32)[None]
    before=a.copy();r=c.logits_record([a],['logits'],999)
    assert r['top1']==999 and r['top5_raw_values'][0]==499 and np.array_equal(a,before)


def test_real_native_and_quality_functions_with_synthetic_engine(tmp_path,monkeypatch):
    f=fixture_run(tmp_path);monkeypatch.setitem(sys.modules,'dx_engine',fake_dx_module())
    before={str(x):c.digest(x) for x in [f.onnx,f.dx]}
    result=remote_worker.run(f.request,f.stage,f.out/'remote')
    assert result['status']=='complete',result['errors']
    assert len(result['native_calls'])==len(result['quality_calls'])==2
    for r in result['native_calls']:
        assert r['sealer_roundtrip']['exact_equal']
        assert r['native_vs_quality_feed']['exact_equal']
        assert r['native_vs_quality_output']['exact_equal']
        assert r['prediction']['harness_top1_match']
    for r in result['quality_calls']:
        assert r['export_image_matches'] and r['export_label_matches'] and r['export_top1_matches_independent']
    assert before=={str(x):c.digest(x) for x in [f.onnx,f.dx]}
    assert result['compiler_invoked'] is False and result['counts_as_benchmark'] is False
    assert len(result['sentinel_repeats'])==3
    assert result['pipeline_consistency']=='consistent_observed_scope'


def test_cpu_real_preprocessing_with_synthetic_ORT(tmp_path,monkeypatch):
    f=fixture_run(tmp_path)
    monkeypatch.setitem(sys.modules,'onnxruntime',fake_ort_module())
    monkeypatch.setattr(cpu_worker,'inspect_graph',lambda _: {'status':'observed','source':'SYNTHETIC_TEST_GRAPH_ONLY'})
    result=cpu_worker.run(f.request,f.stage,f.out/'cpu')
    assert result['status']=='complete',result['errors']
    for r in result['rows']:
        assert r['input_comparisons']['native_vs_semantic']['exact_equal']
        assert r['input_comparisons']['mean_std_vs_canonical_reference']['exact_equal']
        assert not r['input_comparisons']['canonical_direct_vs_harness_crop256']['exact_equal']
    assert (f.out/'cpu/cpu_000.npz').is_file()
    assert len(result['synthetic_input_sentinels'])==3
    assert all(r['accuracy_evaluated'] is False and r['native_vs_semantic']['exact_equal'] for r in result['synthetic_input_sentinels'])


def test_pairing_and_report(tmp_path,monkeypatch):
    f=fixture_run(tmp_path);monkeypatch.setitem(sys.modules,'dx_engine',fake_dx_module());monkeypatch.setitem(sys.modules,'onnxruntime',fake_ort_module())
    monkeypatch.setattr(cpu_worker,'inspect_graph',lambda _: {'status':'observed','source':'SYNTHETIC_TEST_GRAPH_ONLY'})
    cpu_worker.run(f.request,f.stage,f.out/'cpu');remote_worker.run(f.request,f.stage,f.out/'remote')
    result=analyze_evidence.run(f.out)
    assert len(result['paired_hardware'])==2
    assert result['cpu_controls']['scale_only']['sample_count']==3
    assert result['hardware_native']['sample_count']==2
    assert result['model_acceptance']=='NOT_EVALUATED'
    assert (f.out/'REPORT.md').is_file()


def test_remote_refuses_wrong_staged_model(tmp_path,monkeypatch):
    f=fixture_run(tmp_path);monkeypatch.setitem(sys.modules,'dx_engine',fake_dx_module())
    (f.stage/'suite/deepx/deepx_m1/full/model.dxnn').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='digest_mismatch'):remote_worker.run(f.request,f.stage,f.out/'remote')


def test_remote_refuses_wrong_staged_image(tmp_path,monkeypatch):
    f=fixture_run(tmp_path);monkeypatch.setitem(sys.modules,'dx_engine',fake_dx_module())
    (f.stage/'suite'/f.request['samples'][0]['image']).write_bytes(b'wrong')
    with pytest.raises(ValueError,match='digest_mismatch'):remote_worker.run(f.request,f.stage,f.out/'remote')


def test_cpu_source_changed_refused(tmp_path):
    f=fixture_run(tmp_path);f.onnx.write_text('changed')
    with pytest.raises(ValueError,match='changed_before'):cpu_worker.run(f.request,f.stage,f.out/'cpu')


def test_missing_cv2_is_explicit_but_native_still_captured(tmp_path,monkeypatch):
    f=fixture_run(tmp_path);monkeypatch.setitem(sys.modules,'dx_engine',fake_dx_module());monkeypatch.setitem(sys.modules,'cv2',None)
    r=remote_worker.run(f.request,f.stage,f.out/'remote')
    assert r['status']=='complete' and len(r['native_calls'])==2
    assert r['product_quality_result']['status']=='ok'


def test_timeout_records_failure(tmp_path):
    r=c.run_bounded([sys.executable,'-I','-S','-c','import time;time.sleep(20)'],tmp_path/'log',.3)
    assert r['timed_out'] and r['returncode']!=0


def test_package_does_not_archive_models_or_staging(tmp_path):
    out=tmp_path/'result';out.mkdir();put(out/'status.json',{'status':'test'})
    stage=out/'_stage';stage.mkdir();(stage/'model.dxnn').write_bytes(b'not archive');(stage/'secret').write_text('not archive')
    z=collect.archive_output(out)
    with zipfile.ZipFile(z) as a:assert a.namelist()==['status.json']


def test_feature_vector_singletons_and_layouts():
    a=np.arange(1000,dtype=np.float32).reshape(1,1,1000)
    b,m=analyze_evidence.feature_for_input(a,[1,1000]);assert b.shape==(1,1000) and m=='singleton_feature_vector_reshape'
    with pytest.raises(ValueError):analyze_evidence.feature_for_input(np.zeros((2,3)),[1,6])


def test_recorded_export_metadata_arm_is_explicit(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=1)
    f.request['export_preprocess']={'resize_size':[232],'crop_size':[224],'mean':[.485,.456,.406],'std':[.229,.224,.225],'interpolation':'BILINEAR'}
    monkeypatch.setitem(sys.modules,'onnxruntime',fake_ort_module())
    monkeypatch.setattr(cpu_worker,'inspect_graph',lambda _: {'status':'observed','source':'SYNTHETIC_TEST_GRAPH_ONLY'})
    result=cpu_worker.run(f.request,f.stage,f.out/'cpu')
    assert result['status']=='complete',result['errors']
    row=result['rows'][0]
    assert row['controls']['export_metadata_geometry']['status']=='inferred'
    assert 'not_a_new_reference' in row['metadata_geometry_scope']


def test_optional_hardware_part1_and_existing_CPU_tail_control(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=2)
    p1=tmp_path/'part1.onnx';p1.write_text('synthetic part1')
    p2=tmp_path/'part2.onnx';p2.write_text('synthetic part2')
    dx=f.stage/'suite/split/model.dxnn';dx.parent.mkdir();dx.write_bytes(b'synthetic part1')
    contract=c.read_json(f.stage/'suite/deepx/deepx_m1/full/output_contract.json')
    contract['outputs']=[{'name':'feature'}]
    put(dx.with_name('output_contract.json'),contract)
    f.request['split']={'status':'ready','local_part1':str(p1),'local_part2':str(p2),
                        'source_onnx_sha256':c.digest(p1),'part2_sha256_observed':c.digest(p2),
                        'sha256':c.digest(dx),'output_names':['feature']}
    c.write_json(f.out/'evidence_request.json',f.request)
    def feature(x):return np.full((1,960),float(np.asarray(x).mean()),dtype=np.float32)
    dxmod=fake_dx_module();real=dxmod.InferenceEngine
    class Engine(real):
        def run(self,feeds):
            if '/split/' in self.path:return [feature(feeds[0].astype(np.float32)/np.float32(255)).reshape(1,1,960)]
            return super().run(feeds)
    dxmod.InferenceEngine=Engine;monkeypatch.setitem(sys.modules,'dx_engine',dxmod)
    ort=fake_ort_module();BaseSession=ort.InferenceSession
    class Session(BaseSession):
        def __init__(self,path,*a,**kw):super().__init__(path,*a,**kw);self.path=Path(path).name
        def get_inputs(self):
            return [types.SimpleNamespace(name='feature',shape=[1,960],type='tensor(float)')] if self.path=='part2.onnx' else super().get_inputs()
        def get_outputs(self):
            return [types.SimpleNamespace(name='feature',shape=[1,960])] if self.path=='part1.onnx' else super().get_outputs()
        def run(self,names,feeds):
            a=next(iter(feeds.values()))
            return [feature(a)] if self.path=='part1.onnx' else [fake_logits(a)]
    ort.InferenceSession=Session;monkeypatch.setitem(sys.modules,'onnxruntime',ort)
    monkeypatch.setattr(cpu_worker,'inspect_graph',lambda _: {'status':'observed','source':'SYNTHETIC_TEST_GRAPH_ONLY'})
    cpu_worker.run(f.request,f.stage,f.out/'cpu')
    remote=remote_worker.run(f.request,f.stage,f.out/'remote')
    assert remote['split']['status']=='observed',remote['split']
    result=analyze_evidence.split_control(f.request,f.out)
    assert result['status']=='observed',result
    assert len(result['rows'])==2
    assert result['rows'][0]['layout_mapping']=='singleton_feature_vector_reshape'
    assert 'NOT TensorRT' in result['scope']


def test_t07_25_cache_verify_only_missing_artifact_never_builds(tmp_path):
    f=fixture_run(tmp_path);f.dx.unlink();(f.stage/'suite/deepx/deepx_m1/full/model.dxnn').unlink()
    second=f.out/'second';second.mkdir()
    with pytest.raises(ValueError,match='exact_artifact_unavailable'):
        collect.prepare_model(f.run,f.model,f.target,{'runtime_venv':'/not-used'},f.args,second)
    assert not list(second.rglob('*.dxnn'))
    assert c.clean_env(second)['ONNX_SPLITPOINT_ARTIFACT_POLICY']=='cache_verify_only'


def test_t07_27_payload_budget_checked_before_write(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'MAX_MODEL_PAYLOAD_BYTES',16)
    with pytest.raises(ValueError,match='budget'):c.save_tensors(tmp_path/'large.npz',x=np.zeros(100,np.float32))
    assert not (tmp_path/'large.npz').exists()
    with pytest.raises(ValueError,match='pickle'):c.save_tensors(tmp_path/'obj.npz',x=np.array([{}],dtype=object))


def test_t07_19_owned_outputs_survive_next_engine_call(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=2);dx=fake_dx_module();Base=dx.InferenceEngine
    class BufferReuse(Base):
        def __init__(self,path):super().__init__(path);self.buffer=np.empty((1,1000),np.float32)
        def run(self,feeds):self.buffer[:]=fake_logits(feeds[0].astype(np.float32)/np.float32(255));return [self.buffer]
    dx.InferenceEngine=BufferReuse;monkeypatch.setitem(sys.modules,'dx_engine',dx)
    result=remote_worker.run(f.request,f.stage,f.out/'remote')
    assert result['status']=='complete'
    with np.load(f.out/'remote/native_000.npz',allow_pickle=False) as first,np.load(f.out/'remote/native_001.npz',allow_pickle=False) as second:
        assert not np.array_equal(first['output_00'],second['output_00'])
    assert len(result['sentinel_repeats'])==3


def test_t07_27_collected_mismatch_is_failure(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=2);dx=fake_dx_module();Base=dx.InferenceEngine
    class WrongNative(Base):
        instances=0
        def __init__(self,path):super().__init__(path);self.ordinal=WrongNative.instances;WrongNative.instances+=1
        def run(self,feeds):return [super().run(feeds)[0]+np.float32(self.ordinal)]
    dx.InferenceEngine=WrongNative;monkeypatch.setitem(sys.modules,'dx_engine',dx)
    result=remote_worker.run(f.request,f.stage,f.out/'remote')
    assert result['collection_status']=='complete' and result['status']=='partial'
    assert result['pipeline_consistency']=='failed_or_incomplete'
    assert collect.archive_output(f.out).is_file()
    assert result['counts_as_benchmark'] is False


def test_t07_22_calibration_identity_and_recipe_not_guessed(tmp_path):
    f=fixture_run(tmp_path);f.request['calibration_samples']=[f.request['samples'][0]]
    f.request['calibration_config']={'default_loader':{'preprocessings':[{'wrong_norm':{}}]}}
    out=f.out/'calibration';out.mkdir();bench=c.isolated_product(f.stage/'suite')
    r=remote_worker.calibration_replay(f.stage/'suite',f.request,out,bench)
    assert r['reason']=='loader_recipe_not_exactly_supported' and r['historical_calibration_evidence_missing']
    f.request['calibration_samples']*=2
    with pytest.raises(ValueError,match='duplicate_calibration'):remote_worker.calibration_replay(f.stage/'suite',f.request,out,bench)


def test_t07_23_missing_vendor_loader_dependency_remains_missing(tmp_path,monkeypatch):
    f=fixture_run(tmp_path);f.request['calibration_samples']=[f.request['samples'][0]]
    f.request['calibration_config']={'default_loader':{'preprocessings':[{'resize':{'width':224,'height':224}},{'div':{'x':255.0}},{'convertColor':{'form':'BGR2RGB'}},{'transpose':{'axis':[2,0,1]}},{'expandDim':{'axis':0}}]}}
    monkeypatch.setitem(sys.modules,'cv2',None)
    r=remote_worker.calibration_replay(f.stage/'suite',f.request,f.out,None)
    assert r['status']=='not_evaluated' and r['vendor_loader_executed'] is False
    assert r['historical_calibration_evidence_missing'] is True


def test_t07_27_thin_remote_supervisor_runs_existing_process_owner(tmp_path):
    stage=tmp_path/'owned-stage';lib=stage/'lib';lib.mkdir(parents=True)
    for name in ('remote_supervisor.py','smoke_common.py'):
        shutil.copyfile(ROOT/'lib'/name,lib/name)
    for name in ('deepx_full_workflow_smoke_worker_v27930.py','deepx_full_output_probe_worker_v27930.py'):
        shutil.copyfile(SOURCE/'scripts'/name,lib/name)
    (lib/'remote_worker.py').write_text("from pathlib import Path\nimport json\np=Path(__file__).resolve().parents[1]/'results/remote_result.json'\np.write_text(json.dumps({'status':'fixture_worker_complete','hardware_executed':False}))\n")
    put(stage/'request.json',{'diagnostic_only':True})
    process=subprocess.run([sys.executable,'-I','-B',str(lib/'remote_supervisor.py'),'--stage',str(stage),'--timeout','5'],capture_output=True,text=True,timeout=20,env=c.clean_env(stage))
    assert process.returncode==0,(process.stdout,process.stderr)
    report=c.read_json(stage/'results/execution.json')
    assert report['cleanup_complete'] and not report['owned_survivors']
    stopped=subprocess.run([sys.executable,'-I','-B',str(lib/'remote_supervisor.py'),'--stage',str(stage),'--stop'],capture_output=True,text=True,timeout=10,env=c.clean_env(stage))
    assert stopped.returncode==0
    assert c.read_json(stage/'results/remote_result.json')['hardware_executed'] is False


def test_t07_14_launcher_advertises_explicit_stages():
    result=subprocess.run(['bash',str(SOURCE/'scripts/run_classification_input_probe_v27931.sh'),'--help'],capture_output=True,text=True,timeout=10,env={**os.environ,'ONNX_SPLITPOINT_PYTHON':sys.executable,'ORT_DISABLE_TELEMETRY':'1'})
    assert result.returncode==0,result.stderr
    assert all(s in result.stdout for s in ('contracts','inputs','cpu-paired','hardware-paired','calibration-audit'))


def test_t07_8_cpu_actual_input_name_must_match_source_contract(tmp_path,monkeypatch):
    f=fixture_run(tmp_path,count=1);ort=fake_ort_module();Base=ort.InferenceSession
    class WrongName(Base):
        def get_inputs(self):return [types.SimpleNamespace(name='unbound_input',shape=[1,3,224,224],type='tensor(float)')]
        def run(self,*a,**kw):raise AssertionError('unbound input executed')
    ort.InferenceSession=WrongName;monkeypatch.setitem(sys.modules,'onnxruntime',ort)
    monkeypatch.setattr(cpu_worker,'inspect_graph',lambda _: {'status':'observed','source':'SYNTHETIC_TEST_GRAPH_ONLY'})
    result=cpu_worker.run(f.request,f.stage,f.out/'cpu')
    assert result['status']=='partial'
    assert any('input_name_disagrees' in r['error'] for r in result['errors'])
    assert all(v['status']=='input_observed' for r in result['rows'] for v in r['controls'].values())


@pytest.mark.parametrize('dtype',[np.uint8,np.int8,np.int64])
def test_t07_12_remote_quantized_logits_require_dequantization_contract(tmp_path,monkeypatch,dtype):
    f=fixture_run(tmp_path,count=1);mod=fake_dx_module();Base=mod.InferenceEngine
    class IntegerOutput(Base):
        def run(self,feeds):return [np.arange(1000).astype(dtype)[None]]
    mod.InferenceEngine=IntegerOutput;monkeypatch.setitem(sys.modules,'dx_engine',mod)
    result=remote_worker.run(f.request,f.stage,f.out/'remote')
    assert result['status']=='partial' and result['pipeline_consistency']=='failed_or_incomplete'
    assert any('dequantization_contract' in r['error'] for r in result['errors'])
    assert not result['sentinel_repeats']


@pytest.mark.parametrize('mode',['extra_output','integer','nonfinite','empty_name','duplicate_names'])
def test_t07_19_part1_capture_rejects_unbound_or_invalid_boundary(tmp_path,monkeypatch,mode):
    f=fixture_run(tmp_path,count=1);dx=f.stage/'suite/split/model.dxnn';dx.parent.mkdir();dx.write_bytes(b'synthetic part1')
    names={'empty_name':[''],'duplicate_names':['feature','feature']}.get(mode,['feature'])
    contract=c.read_json(f.stage/'suite/deepx/deepx_m1/full/output_contract.json');contract['outputs']=[{'name':v} for v in names]
    put(dx.with_name('output_contract.json'),contract)
    f.request['split']={'status':'ready','sha256':c.digest(dx),'output_names':names}
    mod=fake_dx_module();Base=mod.InferenceEngine
    class BadPart1(Base):
        def run(self,feeds):
            if '/split/' not in self.path:return super().run(feeds)
            a=np.zeros((1,960),dtype=np.int8 if mode=='integer' else np.float32)
            if mode=='nonfinite':a[0,0]=np.nan
            return [a,a] if mode=='extra_output' else [a]
    mod.InferenceEngine=BadPart1;monkeypatch.setitem(sys.modules,'dx_engine',mod)
    result=remote_worker.run(f.request,f.stage,f.out/'remote')
    assert result['split']['status']=='failed'
    assert 'part1_' in result['split']['error']
    assert not list((f.out/'remote').glob('part1_*.npz'))


@pytest.mark.parametrize('scopes,expected',[
    (['not_evaluated'],'not_evaluated'),
    (['consistent_observed_scope','not_evaluated'],'not_evaluated'),
    (['consistent_observed_scope']*3,'consistent_observed_scope'),
    (['reconstructed_scope_only'],'reconstructed_scope_only'),
    (['failed_source_build_semantics','not_evaluated'],'failed_source_build_semantics'),
    (['failed_or_incomplete'],'failed_or_incomplete'),
])
def test_t07_27_collection_cannot_upgrade_unobserved_pipeline(scopes,expected):
    assert collect.aggregate_pipeline_consistency(scopes)==expected
    assert collect.aggregate_pipeline_consistency(scopes,plan_only=True)=='not_evaluated'
