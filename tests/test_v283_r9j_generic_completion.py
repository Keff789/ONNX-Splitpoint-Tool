"""Generic producer completion, phase clocks and archived invalid-image evidence."""
from copy import deepcopy
from pathlib import Path
import ast
import json
import sys
import types
import time
import numpy as np
import pytest
from onnx_splitpoint_tool.runners.task_completion import TimedTaskCompletion, validate_completion, completion_projection
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
ROOT=Path(__file__).resolve().parents[1]
TASK=Path('/home/kmika/.local/share/onnx-splitpoint-codex/v283_R9J_20260919_211316_8daof7vt')

def evidence():
    p=ClassificationCompletion();timer=TimedTaskCompletion('generic_deepx_full','classification',lambda:p.completed_count)
    for _ in range(3):timer.run(lambda:{'logits':np.array([[1,3,2]])},p.process)
    return timer.report(1,2)

@pytest.mark.parametrize('damage',['producer','task','late','missing','counter','double_nms'])
def test_negative_execution_evidence(damage):
    e=evidence()
    if damage=='producer':e['producer']='native_full_deepx'
    if damage=='task':e['task']='detection'
    if damage=='late':e['frames'][0]['tail_end_ns']=e['frames'][0]['timer_end_ns']+1
    if damage=='missing':e['frames'].pop()
    if damage=='counter':e['frames'][0]['completions']=2
    if damage=='double_nms':e['postprocess_contract']={'source_nms_attested':True,'host_nms_applied':True}
    with pytest.raises(ValueError):validate_completion(e,task='classification',producer='generic_deepx_full')

def test_real_classification_completion_and_projection():
    e=evidence();p=completion_projection(e,task='classification',producer='generic_deepx_full')
    assert p['measurement_endpoint']=='completed_classification' and p['postprocess_completed_frames']==2
    assert p['completed_task_mean_ms']>=p['raw_stage_mean_ms']+p['host_tail_mean_ms']

def test_generated_hailo_callback_runs_real_shared_decode_inside_timer(tmp_path):
    from test_v27923_decoded_completion import _runtime,_outputs
    from onnx_splitpoint_tool.native_detection_postprocess import FrozenDetectionPostprocessor, build_frozen_postprocess_contract
    outputs=_outputs();contract=build_frozen_postprocess_contract(model_id='yolo11l',outputs=outputs,input_hw=[640,640],original_wh=[640,480],source_contract_family='decoded_pre_nms')
    processor=FrozenDetectionPostprocessor(contract)
    timer=TimedTaskCompletion('generic_hailo_full','detection',lambda:processor.completed_count,contract=contract)
    template=ROOT/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'
    tree=ast.parse(template.read_text())
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_run_full_frozen_detection_hotloop')
    class SDK:
        def infer(self,inputs):return outputs
    ns={'Dict':dict,'np':np,'hailo_full':SDK(),'full_inputs_hailo':{'input':np.zeros((1,3,640,640))},
        'generic_full_frozen_postprocessor':processor,'generic_full_frozen_original_wh':[640,480],'generic_full_completion_timer':timer}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(template),'exec'),ns)
    ns[fn.name]();e=timer.report(0,1)
    assert processor.last_result['detection_count']==2
    assert processor.contract['decoder_id']=='ultralytics_decoded_classaware_nms_v1'
    assert validate_completion(e,task='detection',producer='generic_hailo_full')

def test_real_generic_deepx_prepared_loop(case,monkeypatch):
    from test_v27926_deepx_full_decoded_pre_nms import _suite,_install_runtime,_decoded_outputs
    from PIL import Image
    suite=_suite();count=_install_runtime(monkeypatch,_decoded_outputs())
    import onnx_splitpoint_tool.runners.task_completion as module
    monkeypatch.setitem(sys.modules,'splitpoint_runners.task_completion',module)
    image=case.root/'image.jpg';Image.new('RGB',(640,480)).save(image)
    monkeypatch.setattr(suite,'_deepx_find_prepared_feed_image',lambda *a:(image,'fixture'))
    result=suite._run_deepx_prepared_feed_benchmark(case.root,case.cached,
        {'benchmark_task':'detection','model_id':'yolo11l'},types.SimpleNamespace(warmup=1,runs=3,benchmark_task='detection'),case.root)
    assert result['status']=='ok',result
    assert validate_completion(result['generic_completion_evidence'],task='detection',producer='generic_deepx_full')
    assert result['completed_frames']==3 and result['postprocess_completed_frames']==3

from test_v27926_deepx_full_decoded_pre_nms import case

@pytest.mark.parametrize('image_id,bad',[('000000052891',True),('000000395801',True),('000000000139',False),('000000000285',False)])
def test_archived_sdk_outputs_preserve_invalid_final_box_contract(image_id,bad):
    from test_v275_deepx_full_quality_contract import _functions
    archive=Path('/home/kmika/.local/share/onnx-splitpoint-codex/v283_r3_lokal_20260914_192352_8vql14m7/archivierte_codex_runs/v283-nightfix-20260914_161840-ZUNmFW/fortsetzung_r2_20260914_173733_k7C9wO')
    raw=archive/'deepx_remote/results'/f'{image_id}_raw.npz'
    assert raw.is_file(),raw
    arrays=np.load(raw);arr=arrays[arrays.files[0]];before=arr.copy()
    rows=arr.reshape(-1,6)
    assert np.isfinite(rows).all()
    assert bool(np.any(rows[:,2]<rows[:,0])) is bad
    # Exercise the actual strict runtime attestation, before confidence filtering.
    from test_v27926_deepx_full_decoded_pre_nms import _suite
    suite=_suite()
    contract=json.loads((archive/'tmp/deepx_stage/suite/deepx/deepx_m1/full/output_contract.json').read_text())
    bound=suite._deepx_bind_authoritative_endpoint_contract(archive/'tmp/deepx_stage/suite',{'model_id':'yolo26s','benchmark_task':'detection'},contract)
    result=suite._deepx_bn6_runtime_semantic_attestation(arr,bound)
    assert result['pass'] is (not bad)
    np.testing.assert_array_equal(arr,before)

def test_profiles_normal_loader_frozen_scope_and_diagnostic_denominator(monkeypatch):
    from onnx_splitpoint_tool.benchmark.evaluation_profiles import load_evaluation_profile
    spec=__import__('importlib.util',fromlist=['spec_from_file_location']).spec_from_file_location('r9j_scope',TASK/'operator/scope_contract.py')
    scope=__import__('importlib.util',fromlist=['module_from_spec']).module_from_spec(spec);spec.loader.exec_module(scope)
    for cell in ('A','B','EVAL'):
        monkeypatch.setenv('R9J_CELL',cell)
        p=load_evaluation_profile(TASK/'profiles'/f'{cell}.yaml').raw_profile
        scope.validate_scope(p,'eval' if cell=='EVAL' else 'smoke')
    p=load_evaluation_profile(TASK/'profiles/A.yaml').raw_profile
    manifest=json.loads(Path(p['campaign']['dataset_manifests']['detection']['validation']).read_text())
    assert len(manifest['items'])==32
    assert {52891,395801}<={r['image_id'] for r in manifest['items']}

def test_sentinel_proof_uses_completed_observations_not_manifest():
    from onnx_splitpoint_tool.accuracy_reporting import observed_coverage
    row={'task':'detection','technical_status':'completed','accuracy_assessment':{'accuracy_class':'accuracy_loss'},'observed_image_ids':['000000052891.jpg','000000395801.jpg'],'evaluated_images':2}
    p=observed_coverage([row],run_id='run')
    assert p['evaluated_images']==2 and set(p['observed_image_ids'])=={'52891','395801'}
    assert observed_coverage([{**row,'technical_status':'failed'}],run_id='run')['evaluated_images']==0


def test_measured_completion_survives_actual_normalization_and_foreign_report_is_rejected(tmp_path):
    from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
    raw={'backend':'deepx_m1','variant':'full','case_id':'full','task':'classification','runtime_ok':True,'full_mean_ms':4,'generic_completion_evidence':evidence()}
    row=normalize_benchmark_row(raw,model_id='resnet50',source_path=tmp_path/'generic.json')
    assert row['endpoint_contract_complete'] is True
    assert row['measurement_endpoint']=='completed_classification'
    assert row['generic_completion_evidence']==raw['generic_completion_evidence']
    raw['generic_completion_evidence']['producer']='native_full_deepx'
    with pytest.raises(ValueError):normalize_benchmark_row(raw,model_id='resnet50',source_path=tmp_path/'generic.json')


def test_generic_other_full_callback_times_actual_topk():
    template=ROOT/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'
    fn=next(n for n in ast.walk(ast.parse(template.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_run_other_full_completed')
    processor=ClassificationCompletion()
    timer=TimedTaskCompletion('generic_hailo_full','classification',lambda:processor.completed_count)
    ns={'generic_other_full_timer':timer,'generic_other_full_processor':processor,
        '_run_full_variant_outputs_map':lambda **kw:({'logits':np.array([[1,4,2]])},{})}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(template),'exec'),ns)
    ns[fn.name]()
    assert processor.completed_count==1
    assert validate_completion(timer.report(0,1),task='classification',producer='generic_hailo_full')


def _initialize_generated_full(tmp_path, monkeypatch, outputs, *, model='yolo26s', backend='tensorrt'):
    """Execute the generated initialization AND callback, stubbing only inference."""
    from PIL import Image
    from test_v275_preprocessing_contract import _runner_module
    import onnx_splitpoint_tool.native_detection_postprocess as pp
    import onnx_splitpoint_tool.native_output_endpoint as endpoint
    monkeypatch.setitem(sys.modules, 'splitpoint_runners.native_detection_postprocess', pp)
    monkeypatch.setitem(sys.modules, 'splitpoint_runners.native_output_endpoint', endpoint)
    module = _runner_module()
    template = ROOT/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'
    tree = ast.parse(template.read_text())
    block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                 and 'not hailo_full_raw_detection_head' in ast.unparse(n.test))
    callback = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == '_run_other_full_completed')
    image = tmp_path/'input.jpg'; Image.new('RGB', (640,480)).save(image)
    run = Path('/home/kmika/Models/EvaluationRuns/v283_R9J_20260919_211316_8daof7vt/a_01_rgxiu54f/r9j_a_20260919_220725')
    declaration = json.loads((run/'models/yolo26s/benchmark_set/legacy_suite/output_contracts.json').read_text())
    artifact=tmp_path/'fake.hef';artifact.write_bytes(b'fake SDK boundary; never executed')
    for row in declaration['contracts']:
        if row['backend'] == 'cuda_ort':
            row['backend'] = backend
            if backend.startswith('hailo'):
                row.update(recorded_artifact_path=str(artifact),
                           recorded_artifact_size_bytes=artifact.stat().st_size,
                           recorded_artifact_sha256=__import__('hashlib').sha256(artifact.read_bytes()).hexdigest())
    (tmp_path/'output_contracts.json').write_text(json.dumps(declaration))
    ns = dict(module.__dict__)
    ns.update(variants=['full'], hailo_full_raw_detection_head=False, variant_errors={},
              args=types.SimpleNamespace(benchmark_task='detection',model_id=model,quality_evidence_model_id=''),
              manifest={'task':'detection'}, plan={}, img_path=image, img_hw=(640,640),
              base_dir=tmp_path/'case', full_path=tmp_path/'full.onnx', full_tok=backend,
              hailo_full=object() if backend.startswith('hailo') else None,
              generic_other_full_processor=None, generic_other_full_timer=None,
              _generic_task='detection', _decoder_model_sha256='', TimedTaskCompletion=TimedTaskCompletion,
              _run_full_variant_outputs_map=lambda **kw:(outputs, {}))
    exec(compile(ast.Module(body=[block,callback],type_ignores=[]), str(template), 'exec'), ns)
    return ns


@pytest.mark.parametrize('backend', ['tensorrt', 'hailo8', 'hailo10h'])
def test_generated_bn6_initialization_and_completed_callback(tmp_path, monkeypatch, backend):
    # Two overlapping final detections must BOTH survive; source already completed selection.
    outputs={'output0':np.array([[[100,100,200,200,.9,0],[101,101,201,201,.8,0]]],dtype=np.float32)}
    ns=_initialize_generated_full(tmp_path,monkeypatch,outputs,backend=backend)
    assert not ns['variant_errors'],ns['variant_errors']
    ns['_run_other_full_completed']()
    processor=ns['generic_other_full_processor']
    assert processor.completed_count==1
    assert len(processor.last_detections)==2
    assert processor.contract['source_nms_attested'] is True
    assert processor.contract['host_nms_applied'] is False
    assert validate_completion(ns['generic_other_full_timer'].report(0,1),task='detection')


@pytest.mark.parametrize('bad_value', ['inverted_low_score','nan','class','score'])
def test_generated_bn6_rejects_invalid_final_values_before_count(tmp_path,monkeypatch,bad_value):
    outputs={'output0':np.array([[[100,100,200,200,.9,0],[0,0,0,0,0,0]]],dtype=np.float32)}
    ns=_initialize_generated_full(tmp_path,monkeypatch,outputs)
    assert not ns['variant_errors'],ns['variant_errors']
    row=outputs['output0'][0,1]
    if bad_value=='inverted_low_score':row[:4]=[5,0,1,2]
    elif bad_value=='nan':row[0]=np.nan
    elif bad_value=='class':row[5]=.5
    else:row[4]=1.1
    with pytest.raises(RuntimeError):ns['_run_other_full_completed']()
    assert ns['generic_other_full_processor'].completed_count==0
    assert ns['generic_other_full_timer'].frames==[]


def _saved_deepx_suite(tmp_path):
    import shutil
    run=Path('/home/kmika/Models/EvaluationRuns/v283_R9J_20260919_211316_8daof7vt/a_01_rgxiu54f/r9j_a_20260919_220725')
    source=run/'models/yolo26s/benchmark_set/legacy_suite'
    for relative in ('output_contracts.json','deepx/deepx_m1/full/output_contract.json'):
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/relative,target)
    return tmp_path


def test_archived_bad_images_pass_through_real_32_image_consumer_without_denominator_loss(tmp_path,monkeypatch):
    """CPU replay: two real saved bad outputs + a saved control; no device inference."""
    from test_v27926_deepx_full_decoded_pre_nms import _suite,_install_runtime
    suite=_suite()
    root=_saved_deepx_suite(tmp_path)
    diagnosis=json.loads((TASK/'ARCHIVED_IMAGE_DIAGNOSIS.json').read_text())
    saved={row['image_id']:row['source'] for row in diagnosis['rows']}
    images=TASK/'profiles/diagnostic32/images'
    samples=suite._deepx_collect_validation_samples(str(images),root,32)
    assert len(samples)==32
    _install_runtime(monkeypatch,np.zeros((1,300,6),dtype=np.float32))
    observed=[]
    class SavedSDK:
        def __init__(self,path):pass
        def run(self,feeds):
            sample=samples[len(observed)];name=Path(sample['path']).stem
            observed.append(name)
            source=saved.get(name,saved['000000000139'])
            with np.load(source) as data:return [data[data.files[0]].copy()]
    monkeypatch.setattr(sys.modules['dx_engine'],'InferenceEngine',SavedSDK)
    result=suite._run_deepx_semantic_validation(root,tmp_path/'unused.dxnn',
        {'model_id':'yolo26s','benchmark_task':'detection','validation_images':str(images),'validation_max_images':32},
        types.SimpleNamespace(),tmp_path)
    assert result['image_count']==32,result
    assert len(observed)==32 and len(set(observed))==32
    assert result['status']=='semantic_runtime_incomplete'
    assert result['validated_image_count']==30 and result['error_count']==2,result.get('errors')
    assert {Path(e['image']).stem for e in result['errors']}=={'000000052891','000000395801'}
    assert result['decoder_postprocess_contract']['pass'] is False
    assert len(json.loads((tmp_path/'detections.json').read_text())['images'])==30
    assert len(list(images.glob('*.jpg')))==32


def test_generic_deepx_bn6_prepared_loop_materializes_without_second_nms(tmp_path,monkeypatch):
    from test_v27926_deepx_full_decoded_pre_nms import _suite,_install_runtime
    suite=_suite()
    root=_saved_deepx_suite(tmp_path)
    raw=np.zeros((1,300,6),dtype=np.float32)
    raw[0,:2]=[[100,100,200,200,.9,0],[101,101,201,201,.8,0]]
    _install_runtime(monkeypatch,raw)
    import onnx_splitpoint_tool.runners.task_completion as completion
    monkeypatch.setitem(sys.modules,'splitpoint_runners.task_completion',completion)
    image=(TASK/'profiles/diagnostic32/images/000000052891.jpg').resolve()
    monkeypatch.setattr(suite,'_deepx_find_prepared_feed_image',lambda *a:(image,'fixture'))
    result=suite._run_deepx_prepared_feed_benchmark(root,Path(json.loads((TASK/'ARCHIVED_IMAGE_DIAGNOSIS.json').read_text())['final_output_contract']['artifact_path']),
        {'model_id':'yolo26s','benchmark_task':'detection'},types.SimpleNamespace(warmup=1,runs=2,benchmark_task='detection'),tmp_path)
    assert result['status']=='ok',result
    assert result['frozen_decoded_nms_normalization_result']['detection_count']==2
    assert result['frozen_decoded_nms_normalization_contract']['host_nms_applied'] is False
    assert result['completed_task_result_artifact_verification_status']=='verified_exact'
    assert validate_completion(result['generic_completion_evidence'],task='detection',producer='generic_deepx_full')


@pytest.mark.parametrize('damage',['nested_model','source_hash','double_nms'])
def test_completed_bn6_contract_rejects_resigned_wrong_binding(tmp_path,monkeypatch,damage):
    import onnx_splitpoint_tool.native_detection_postprocess as pp
    outputs={'output0':np.array([[[100,100,200,200,.9,0]]],dtype=np.float32)}
    ns=_initialize_generated_full(tmp_path,monkeypatch,outputs)
    contract=deepcopy(ns['generic_other_full_processor'].contract)
    if damage=='nested_model':contract['model_id']='yolo26m'
    elif damage=='source_hash':contract['source_endpoint_contract_hash']='0'*64
    else:contract['host_nms_applied']=True
    contract.pop('contract_sha256');contract['contract_sha256']=pp.canonical_json_sha256(contract)
    with pytest.raises(pp.FrozenPostprocessError):pp.FrozenDecodedNmsPostprocessor(contract)


def test_generated_hailo_rejects_changed_artifact_before_timing(tmp_path,monkeypatch):
    outputs={'output0':np.array([[[100,100,200,200,.9,0]]],dtype=np.float32)}
    ns=_initialize_generated_full(tmp_path,monkeypatch,outputs,backend='hailo8')
    # Replay the same generated initialization with the recorded HEF replaced.
    (tmp_path/'fake.hef').write_bytes(b'different artifact')
    template=ROOT/'onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt'
    block=next(n for n in ast.walk(ast.parse(template.read_text())) if isinstance(n,ast.If)
               and 'not hailo_full_raw_detection_head' in ast.unparse(n.test))
    exec(compile(ast.Module(body=[block],type_ignores=[]),str(template),'exec'),ns)
    assert 'Full task completion unavailable' in ns['variant_errors']['full']
    assert ns['generic_other_full_timer'].frames==[]
