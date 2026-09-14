"""AP7 productive routing. Vendor compilation simulated; ONNX/ORT are real."""
import copy
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from onnx_splitpoint_tool.campaign import create_dataset_manifest
from onnx_splitpoint_tool.deepx.config import classification_profile_admission, resolve_profile_classification_preprocessing
from onnx_splitpoint_tool.deepx import env_status
from onnx_splitpoint_tool.gui.benchmark_workflow import _materialize_manual_deepx_part1_artifacts
from onnx_splitpoint_tool.workflow.deepx_build_binding import materialize_deepx_build_binding, _deepx_compiler_identity
from tests.test_v27925_deepx_compiler_overlay import compiler_fixture
from tests.test_v27931_compiler_routing import _save_config


def _model(path):
    graph=helper.make_graph([helper.make_node('Identity',['images'],['features'])], 'recorded-test-graph',
        [helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,2,2])],
        [helper.make_tensor_value_info('features',TensorProto.FLOAT,[1,3,2,2])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)]);model.ir_version=9
    onnx.save_model(model,path)
    return path


def test_T07_30_regular_profile_mode_resolution():
    assert resolve_profile_classification_preprocessing({})=='imagenet_mean_std'
    assert classification_profile_admission({}, {'task':'classification'})['allowed']


@pytest.mark.parametrize('run_mode',[{'id':'final'},{'mode':'final'},'final'])
def test_T07_31_legacy_final_is_blocked_even_diagnostic(run_mode):
    profile={'run_mode':run_mode,'purpose':'deepx_preprocessing_ab','deepx_build':{'classification_preprocessing':'current_scale_only','diagnostic_only':True}}
    before=copy.deepcopy(profile)
    status=classification_profile_admission(profile,{'task':'classification','preprocessing':{'mean':[.485,.456,.406],'std':[.229,.224,.225]}})
    assert not status['allowed'] and not status['claim_eligible']
    assert status['scientific_claim_exclusion_reason']=='deepx_legacy_classification_preprocessing'
    assert profile==before


def test_T07_32_explicit_development_ab_no_claim_elevation():
    profile={'run_mode':{'id':'development'},'purpose':'deepx_preprocessing_ab','deepx_build':{'classification_preprocessing':'current_scale_only'}}
    result=classification_profile_admission(profile,{'task':'classification'})
    assert result['allowed'] and result['diagnostic_only']
    assert not result['claim_eligible'] and not result['counts_as_benchmark']
    profile.pop('purpose')
    assert not classification_profile_admission(profile,{'task':'classification'})['allowed']
    assert classification_profile_admission(profile,{'task':'detection'})['allowed']


def _production_run(tmp_path, f, cfg, mode, name, *, environment_only=False):
    source=_model(tmp_path/'classification.onnx')
    calibration=tmp_path/'calibration';calibration.mkdir(exist_ok=True)
    (calibration/'label').mkdir(exist_ok=True);(calibration/'label/img.jpg').write_bytes(b'calibration fixture only, not inference input')
    manifest=tmp_path/'calibration.json'
    if not manifest.is_file():
        create_dataset_manifest(task='classification',role='calibration',dataset_id='fixture',split='train',root=calibration,output=manifest,hash_mode='content')
    build={**cfg,'mode':'reuse_and_build_missing','calibration_dir':str(calibration),'calib_count':1,
           'classification_preprocessing':mode,'diagnostic_only':mode=='current_scale_only'}
    profile={'deepx_build':build,'campaign':{'dataset_manifests':{'classification':{'calibration':str(manifest)}}}}
    if environment_only:
        environment = {'kind':'deepx_dxcom', **{key:build.pop(key) for key in ('dx_all_suite_root','compiler_venv','compiler_overlay','cache_dir')}}
        profile['build_environments'] = [environment]

    run=tmp_path/name;suite=run/'suite';suite.mkdir(parents=True)
    result=materialize_deepx_build_binding(run_dir=run,model_id='classification',model_path=str(source),row={'task':'classification','input_shape':[1,3,2,2]},profile_payload=profile,targets=['deepx_m1'],benchmark_set_contract={'suite_dir':str(suite)})
    status=json.loads(Path(result['artifacts']['deepx_artifact_status_json']).read_text())
    assert result['status']=='ok',json.dumps(status,indent=2)
    case=suite/'b135';case.mkdir();_model(case/'part1.onnx');(case/'split_manifest.json').write_text(json.dumps({'part1_model':'part1.onnx'}))
    part1=_materialize_manual_deepx_part1_artifacts(out_dir=suite,bench_plan_runs=[{'type':'matrix','stage1':'deepx_m1','stage2':'tensorrt'}],validation_images='',fallback_calib_dir=str(calibration),calibration_num=1,task_hint='classification',classification_preprocessing=mode,build_config=build,profile_payload=profile,calibration_manifest=str(manifest))
    assert part1['status']=='ok',json.dumps(part1,indent=2)
    return result,status,part1,source


def test_T07_33_T07_34_full_part1_productive_adapter_and_ort(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture;cfg=_save_config(tmp_path,monkeypatch,f)
    result,status,part1,source=_production_run(tmp_path,f,cfg,'imagenet_mean_std','normal')
    receipt=json.loads(Path(result['artifacts']['deepx_build_onnx_adapter_receipt_json']).read_text())
    p1=part1['cases'][0]
    assert receipt['source_onnx_sha256']!=receipt['build_onnx_sha256']
    assert p1['source_onnx_sha256']!=p1['build_onnx_sha256']
    full_adapter=Path(result['artifacts']['deepx_build_onnx_adapter'])
    part1_adapter=(tmp_path/'normal/suite/b135'/p1['build_onnx']).resolve()
    numeric=np.random.default_rng(31).random((1,3,2,2),dtype=np.float32)
    mean=np.asarray([.485,.456,.406],dtype=np.float32).reshape(1,3,1,1)
    std=np.asarray([.229,.224,.225],dtype=np.float32).reshape(1,3,1,1)
    expected=ort.InferenceSession(str(source),providers=['CPUExecutionProvider']).run(None,{'images':(numeric-mean)/std})[0]
    for adapter in [full_adapter,part1_adapter]:
        graph=onnx.load(adapter)
        assert [node.op_type for node in graph.graph.node[:2]]==['Sub','Div']
        observed=ort.InferenceSession(str(adapter),providers=['CPUExecutionProvider']).run(None,{'images':numeric})[0]
        np.testing.assert_array_equal(observed,expected)
    assert receipt['preprocessing']['build_onnx_adapter']['mean']==[.485,.456,.406]
    assert receipt['preprocessing']['build_onnx_adapter']['std']==[.229,.224,.225]
    assert part1['classification_preprocessing']==status['classification_preprocessing']=='imagenet_mean_std'


def test_T05_4_T07_35_cache_separates_modes_and_hit_does_not_probe(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture;cfg=_save_config(tmp_path,monkeypatch,f)
    old,oldstatus,oldp1,_=_production_run(tmp_path,f,cfg,'current_scale_only','old')
    legacyfiles={str(p):p.read_bytes() for p in Path(cfg['cache_dir']).rglob('*') if p.is_file()}
    new,newstatus,newp1,_=_production_run(tmp_path,f,cfg,'imagenet_mean_std','new')
    assert oldstatus['cache_lookup']['identity']!=newstatus['cache_lookup']['identity']
    assert oldp1['cases'][0]['cache_key']!=newp1['cases'][0]['cache_key']
    assert newstatus['cache_lookup']['outcome']=='MISS'
    assert all(Path(path).read_bytes()==data for path,data in legacyfiles.items())
    monkeypatch.setattr(env_status,'_run_owned_probe',lambda *a,**k:pytest.fail('cache HIT started compiler probe'))
    _,again,p1,_=_production_run(tmp_path,f,cfg,'imagenet_mean_std','again')
    assert again['cache_lookup']['outcome']=='HIT'
    assert p1['cases'][0]['cache_lookup']['outcome']=='HIT'


def test_T05_4_static_and_import_identity_are_identical(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture;cfg=_save_config(tmp_path,monkeypatch,f)
    site=f['venv']/'lib/python3.12/site-packages';site.mkdir(parents=True)
    module=site/'dx_com.py';module.write_text('__version__="2.3.0"\n')
    metadata=site/'dx_com-2.3.0.dist-info';metadata.mkdir();(metadata/'METADATA').write_text('Name: dx_com\nVersion: 2.3.0\n');(metadata/'top_level.txt').write_text('dx_com\n')
    (f['venv']/'pyvenv.cfg').write_text('version = 3.12.0\n')
    monkeypatch.setenv('PYTHONPATH',str(site))
    static=env_status.inspect_deepx_environment(config=cfg,path_only=True)
    active=env_status.inspect_deepx_environment(config=cfg,probe_import=True)
    assert _deepx_compiler_identity(cfg=cfg,environment_status=static)==_deepx_compiler_identity(cfg=cfg,environment_status=active)


def test_T07_30_T07_31_real_run_mode_keeps_explicit_legacy_and_compiler_selection():
    from onnx_splitpoint_tool.run_modes import apply_run_mode,default_run_modes_config
    config=default_run_modes_config()
    profile={'name':'v31 fixture','purpose':'deepx_preprocessing_ab','model_suite':{'primary':[{'id':'resnet50','task':'classification'}]},'run_profiles':[],
        'execution_preset':{'id':'final','follow_tool_config':False,'snapshot':config['modes']['final']},
        'deepx_build':{'classification_preprocessing':'current_scale_only','compiler_overlay':'/configured-overlay','compiler_venv':'/configured-venv','diagnostic_only':True}}
    resolved,_=apply_run_mode(profile,config=config)
    assert resolved['deepx_build']['compiler_overlay']=='/configured-overlay'
    assert resolved['deepx_build']['compiler_venv']=='/configured-venv'
    assert resolved['deepx_build']['classification_preprocessing']=='current_scale_only'
    assert not classification_profile_admission(resolved,{'task':'classification'})['allowed']
    del profile['deepx_build']['classification_preprocessing']
    resolved,_=apply_run_mode(profile,config=config)
    assert resolved['deepx_build']['classification_preprocessing']=='imagenet_mean_std'


def test_T07_31_actual_part1_final_blocks_before_cache_probe_or_compiler(tmp_path,monkeypatch):
    monkeypatch.setattr(env_status,'inspect_deepx_environment',lambda *a,**k:pytest.fail('legacy Final attempted compiler/cache environment lookup'))
    profile={'execution_preset':{'id':'final'},'deepx_build':{'classification_preprocessing':'current_scale_only','diagnostic_only':True}}
    result=_materialize_manual_deepx_part1_artifacts(out_dir=tmp_path/'suite',bench_plan_runs=[{'type':'matrix','stage1':'deepx_m1','stage2':'tensorrt'}],validation_images='',fallback_calib_dir='',task_hint='classification',build_config=profile['deepx_build'],profile_payload=profile)
    assert result['status']=='deepx_legacy_classification_preprocessing'
    assert result['compiler_dispatched'] is False
    assert not list(tmp_path.rglob('*.dxnn'))


def test_T07_31_deferred_part1_preserves_final_guard(tmp_path,monkeypatch):
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import defer_deepx_part1_build,_finalize_deepx
    suite=tmp_path/'suite';suite.mkdir();(suite/'benchmark_set.json').write_text(json.dumps({'cases':[]}))
    profile={'execution_preset':{'id':'final'},'deepx_build':{'classification_preprocessing':'current_scale_only','diagnostic_only':True}}
    deferred=defer_deepx_part1_build(out_dir=suite,bench_plan_runs=[{'type':'matrix','stage1':'deepx_m1','stage2':'tensorrt'}],validation_images='',fallback_calib_dir='',task_hint='classification',build_config=profile['deepx_build'],profile_payload=profile)
    assert deferred['compiler_dispatched'] is False
    monkeypatch.setattr(env_status,'inspect_deepx_environment',lambda *a,**k:pytest.fail('deferred legacy Final started environment/probe'))
    jobs=_finalize_deepx(suite,set(),None,None)
    assert jobs[0]['status']=='failed'
    result=json.loads((suite.parent/'deepx_part1_artifact_status.json').read_text())
    assert result['status']=='deepx_legacy_classification_preprocessing'


def test_T05_4_part1_multicase_cache_identity_does_not_depend_on_prior_build(compiler_fixture,tmp_path,monkeypatch):
    f=compiler_fixture;cfg=_save_config(tmp_path,monkeypatch,f)
    suite=tmp_path/'two_cases'
    for case_name in ['b119','b135']:
        case=suite/case_name;case.mkdir(parents=True);_model(case/'part1.onnx')
        (case/'split_manifest.json').write_text(json.dumps({'part1_model':'part1.onnx'}))
    calibration=tmp_path/'calibration';calibration.mkdir();(calibration/'image.jpg').write_bytes(b'fixture')
    kwargs=dict(out_dir=suite,bench_plan_runs=[{'type':'matrix','stage1':'deepx_m1','stage2':'tensorrt'}],validation_images='',fallback_calib_dir=str(calibration),calibration_num=1,task_hint='classification',build_config=cfg)
    first=_materialize_manual_deepx_part1_artifacts(**kwargs)
    assert first['ok_count']==2,first
    monkeypatch.setattr(env_status,'_run_owned_probe',lambda *a,**k:pytest.fail('multicase cache HIT changed identity after prior cold build'))
    second=_materialize_manual_deepx_part1_artifacts(**kwargs)
    assert second['ok_count']==2,second
    assert [row['cache_key'] for row in first['cases']]==[row['cache_key'] for row in second['cases']]
    assert all(row['cache_lookup']['outcome']=='HIT' for row in second['cases'])


def test_T05_1_profile_build_environment_reaches_actual_full_and_part1(compiler_fixture,tmp_path,monkeypatch):
    from onnx_splitpoint_tool import backend_build_environments
    f=compiler_fixture
    cfg=dict(dx_all_suite_root=str(f['venv'].parent),compiler_venv=str(f['venv']),compiler_overlay=str(f['overlay']),cache_dir=str(tmp_path/'profile-cache'))
    monkeypatch.delenv('ONNX_SPLITPOINT_DEEPX_COMPILER_OVERLAY')
    monkeypatch.setattr(backend_build_environments,'CONFIG_PATH',tmp_path/'missing-global-config.yaml')
    result,status,part1,_=_production_run(tmp_path,f,cfg,'imagenet_mean_std','profile-only',environment_only=True)
    assert result['status']==part1['status']=='ok'
    assert status['environment_status']['compiler_overlay']==str(f['overlay'])
    assert part1['environment_status']['compiler_overlay']==str(f['overlay'])
    assert not (tmp_path/'missing-global-config.yaml').exists()


@pytest.mark.parametrize('contract,output,plan_task,reason',[
    ({'model':{'task':'classification'}},{'classification_preprocessing':'current_scale_only'},'', 'deepx_legacy_classification_preprocessing'),
    ({},{'classification_preprocessing':'current_scale_only','input':{'task':'classification'}},'', 'deepx_legacy_classification_preprocessing'),
    ({},{'classification_preprocessing':'current_scale_only'},'classification', 'deepx_legacy_classification_preprocessing'),
    ({},{'classification_preprocessing':'current_scale_only'},'', 'deepx_task_contract_missing'),
    ({'task':'detection'},{'classification_preprocessing':'current_scale_only','input':{'task':'classification'}},'', 'deepx_task_contract_conflict'),
])
def test_T07_31_actual_full_missing_nested_conflicting_task_cannot_bypass_admission(tmp_path,monkeypatch,contract,output,plan_task,reason):
    import sys
    from scripts import native_full_baseline_eval_runner as full
    from tests.test_v27931_native_identity_prerequisites import stage_case
    root,suite=stage_case(tmp_path)
    (suite/'benchmark_set.json').write_text(json.dumps(contract))
    artifact=suite/'deepx/deepx_m1/full/output_contract.json';artifact.parent.mkdir(parents=True);artifact.write_text(json.dumps(output))
    if plan_task:
        (suite/'benchmark_plan.json').write_text(json.dumps({'runs':[{'id':'deepx_m1_full','task':plan_task}]}))
    profile={'execution_preset':{'id':'final'},'deepx_build':{'classification_preprocessing':'current_scale_only','diagnostic_only':True}}
    monkeypatch.setattr(full,'_select_engine_python',lambda *a: ('',{}))
    monkeypatch.setattr(full,'_deepx_full_series_preflight',lambda *a,**k:pytest.fail('unadmitted Full attempted semantic inference'))
    monkeypatch.setattr(full,'_row_for_backend',lambda *a,**k:pytest.fail('unadmitted Full attempted repetition'))
    monkeypatch.setattr(sys,'argv',['full','--root',str(root),'--models','yolo26m','--backends','deepx','--repetitions','3','--deepx-classification-profile-json',json.dumps(profile)])
    assert full.main()!=0
    row=json.loads((root/'analysis_tables/native_full_baseline_eval.json').read_text())['rows'][0]
    assert reason in row['failure_reason'],row
    assert row['repetition_count_attempted']==0


def test_T07_31_actual_split_missing_task_cannot_bypass_admission(tmp_path,monkeypatch):
    from tests.test_v27931_native_identity_prerequisites import run_coordinator
    def prepare(root,suite):
        (suite/'benchmark_set.json').write_text('{}')
        artifact=suite/'b398/deepx/deepx_m1/part1/output_contract.json';artifact.parent.mkdir(parents=True);artifact.write_text(json.dumps({'classification_preprocessing':'current_scale_only'}))
    profile={'deepx_build':{'classification_preprocessing':'current_scale_only','diagnostic_only':True}}
    rc,payload,calls=run_coordinator(tmp_path,monkeypatch,extra=['--deepx-classification-profile-json',json.dumps(profile)],prepare=prepare)
    assert rc!=0 and not calls
    assert 'deepx_task_contract_missing' in payload['rows'][0]['failure_reason']
