from __future__ import annotations
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import pytest
from onnx_splitpoint_tool.native_job_identity import (
    planned_native_identity, native_identity_key, failed_native_result,
    attach_identity_without_conflicts, complete_historical_identity,
    native_job_prerequisite,
)
from onnx_splitpoint_tool.workflow.runner import _native_expected_matrix_status_v60y
from onnx_splitpoint_tool.workflow.evidence_status import _ledger_identity
from scripts import native_producer_e2e_eval_runner as coordinator
from scripts.run_complete_set_replay_v27931 import replay

FIXTURE=Path(__file__).parent/'fixtures/v27931_complete_set'

def job(**changes):
    return dict(backend='deepx_to_trt',model='yolo26m',case='b398',setup_id='device_deepx_A',comparison_backend='deepx',precision='float32_layout_fp16',**changes)

def stage_case(tmp_path, backend='deepx', case='b398'):
    root=tmp_path/'run'; suite=root/'yolo26m/benchmark_set'
    (suite/case).mkdir(parents=True)
    (suite/'benchmark_set.json').write_text(json.dumps({'task':'classification'}))
    return root,suite

def run_coordinator(tmp_path,monkeypatch,extra=(),backend='deepx',prepare=None,child=None):
    root,suite=stage_case(tmp_path,backend)
    if prepare: prepare(root,suite)
    calls=[]
    def physical_child(cmd,**kwargs):
        calls.append(cmd)
        if child: return child(cmd,kwargs,suite)
        raise AssertionError('blocked job must not launch a physical child')
    monkeypatch.setattr(coordinator.subprocess,'run',physical_child)
    monkeypatch.setattr(sys,'argv',['coordinator','--root',str(root),'--backend',backend,'--models','yolo26m','--case-map','{"yolo26m":["b398"]}','--setup-id','device_deepx_A','--precision','float32_layout_fp16','--repetitions','3',*extra])
    rc=coordinator.main()
    summary=json.loads((root/f'analysis_tables/native_{backend}_producer_e2e_eval.json').read_text())
    return rc,summary,calls

def test_t011_new_missing_deepx_has_planned_identity_zero_attempts(tmp_path,monkeypatch):
    rc,summary,calls=run_coordinator(tmp_path,monkeypatch)
    row=summary['rows'][0]
    assert rc==3 and len(summary['rows'])==1 and calls==[]
    assert native_identity_key(row)==native_identity_key(job())
    assert row['planned_native_identity']==planned_native_identity(job())
    assert row['repetition_count_attempted']==0 and row['repetition_count_valid']==0
    assert row['failure_reason'].startswith('missing_deepx_part1_artifact')
    assert _native_expected_matrix_status_v60y([job()],[row],[])['failed_expected_row_count']==1

def test_t012_hailo_binding_error_stays_present(tmp_path,monkeypatch):
    binding=tmp_path/'invalid-binding.json';binding.write_text('{}')
    rc,summary,calls=run_coordinator(tmp_path,monkeypatch,['--native-split-quality-binding-set',str(binding)],backend='hailo10h')
    row=summary['rows'][0]; expected={**job(),'backend':'hailo10h_to_trt','comparison_backend':'hailo10h'}
    assert calls==[] and row['failure_reason']=='native_split_quality_binding_set_invalid'
    matrix=_native_expected_matrix_status_v60y([expected],[row],[])
    assert matrix['present_expected_row_count']==1 and matrix['missing_expected_row_count']==0

def test_t013_original_complete_set_replay(tmp_path):
    result=replay(None,tmp_path/'replay')
    assert [result[key] for key in ('expected_row_count','present_expected_row_count','successful_expected_row_count','failed_expected_row_count','missing_expected_row_count','concise_logical_row_count')]==[63,63,55,8,0,63]
    assert result['row_presence_complete'] and not result['execution_success_complete'] and not result['matrix_complete']
    assert result['original_inputs_byte_unchanged'] and not result['historical_measurements_modified']

@pytest.mark.parametrize('second_setup',['device_deepx_B','device_deepx_C'])
def test_t014_two_vendor_setups_remain_ambiguous(second_setup):
    planned=[job(),{**job(),'setup_id':second_setup}]
    row={**job(),'setup_id':'','ok':False}
    assert complete_historical_identity(row,planned)['identity_status']=='identity_ambiguous'
    assert _native_expected_matrix_status_v60y(planned,[row],[])['missing_expected_row_count']==2

@pytest.mark.parametrize('field,value',[('setup_id','wrong_device'),('precision','uint8_cast_fp16'),('model','wrong_model')])
def test_t015_nonempty_conflicts_are_not_repaired(field,value):
    child={**job(),field:value,'ok':True,'fps_makespan':100}
    result=attach_identity_without_conflicts(job(),child)
    assert not result['ok'] and field in result['identity_conflicts']
    assert result['child_observation'][field]==value
    assert _native_expected_matrix_status_v60y([job()],[child],[])['missing_expected_row_count']==1

def test_t016_repeated_presentations_are_one_logical_row(tmp_path):
    fixture=json.loads((FIXTURE/'original_identity_projection.json').read_text())
    row=next(row for row in fixture['rows'] if row['ok'])
    result=_native_expected_matrix_status_v60y([row],[row,copy.deepcopy(row),copy.deepcopy(row)],[])
    assert result['expected_row_count']==result['present_expected_row_count']==1
    assert len(row['fps_repetition_samples'])==3
    original=copy.deepcopy(row)
    complete_historical_identity(row,[row])
    assert row==original

def test_t017_historical_completion_is_explicit_and_read_only(tmp_path):
    original={**job(),'setup_id':'','comparison_backend':'','ok':False,'fps_makespan':None}
    path=tmp_path/'original.json';path.write_text(json.dumps(original));before=path.read_bytes()
    result=complete_historical_identity(original,[job()])
    assert result['setup_id']==job()['setup_id']
    assert result['identity_completion']['source']=='documented_dispatcher_plan'
    assert result['ok'] is False and path.read_bytes()==before and original['setup_id']==''
    assert complete_historical_identity(original,[])['identity_status']=='identity_unresolved'

@pytest.mark.parametrize('backend,alias,comparison',[('hailo10h_to_trt','hailo10_to_trt','hailo10h'),('native_full_hailo10h','native_full_hailo10','hailo10h'),('native_full_tensorrt','native_full_tensorrt','deepx')])
def test_t018_full_precision_rule_and_hailo_aliases(backend,alias,comparison):
    planned={**job(),'backend':backend,'case':'full' if backend.startswith('native_full_') else 'b398','comparison_backend':comparison}
    actual={**planned,'backend':alias,'ok':True}
    if backend.startswith('native_full_'): actual['precision']='fp16'
    assert native_identity_key(planned)==native_identity_key(actual)
    assert _ledger_identity(planned)==_ledger_identity(actual)
    assert _native_expected_matrix_status_v60y([planned],[actual],[])['successful_expected_row_count']==1

def test_t021_original_compiler_error_precedes_missing_dxnn(tmp_path,monkeypatch):
    model=tmp_path/'model';suite=model/'benchmark_set/legacy_suite';path=suite/'deepx/deepx_m1/part1/deepx_part1_artifact_status.json';path.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE/'original_yolo26m_deepx_part1_status.json',path)
    result=native_job_prerequisite(job(),suite_dir=suite,model_dir=model,quality_error='binding missing')
    assert 'deepx_compiler_cuda_architecture_unsupported' in result['primary_failure_reason']
    prereq=tmp_path/'prerequisites.json';prereq.write_text(json.dumps({'rows':[result]}))
    rc,summary,calls=run_coordinator(tmp_path,monkeypatch,['--native-job-prerequisites',str(prereq)])
    assert calls==[] and summary['rows'][0]['primary_failure_reason']==result['primary_failure_reason']
    assert summary['rows'][0]['repetition_count_attempted']==0

def test_t022_invalid_endpoint_primary_reason_and_no_unbound_fallback(tmp_path,monkeypatch):
    model=tmp_path/'model';results=model/'benchmark_results';results.mkdir(parents=True)
    reason='score_column_not_probability_like;coordinates_not_ordered_xyxy'
    selected={**job(),'backend':'hailo10h_to_trt','comparison_backend':'hailo10h'}
    (results/'normalized_results.json').write_text(json.dumps({'results':[{**selected,'case_id':'b398','endpoint':{'endpoint_attestation_reason':reason}}]}))
    blocked=native_job_prerequisite(selected,suite_dir=model/'suite',model_dir=model,quality_error='binding missing')
    prereq=tmp_path/'prerequisites.json';prereq.write_text(json.dumps({'rows':[blocked]}))
    _,summary,calls=run_coordinator(tmp_path,monkeypatch,['--native-job-prerequisites',str(prereq)],backend='hailo10h')
    assert calls==[] and summary['rows'][0]['primary_failure_reason']==reason

@pytest.mark.parametrize('model,case',[('yolo26m','b398'),('yolo26s','b364')])
def test_t023_original_exact_negative_case_remains_blocked(model,case,tmp_path):
    model_dir=tmp_path/model;(model_dir/'benchmark_set').mkdir(parents=True)
    shutil.copy2(FIXTURE/f'original_{model}_backend_artifact_decisions.json',model_dir/'benchmark_set/backend_artifact_decisions.json')
    selected={**job(),'backend':'hailo8_to_trt','model':model,'case':case,'comparison_backend':'hailo8'}
    result=native_job_prerequisite(selected,suite_dir=model_dir/'suite',model_dir=model_dir)
    assert result['prerequisite_status']=='blocked'
    assert 'exact_deterministic_outcome' in result['primary_failure_reason']
    assert result['repetition_count_attempted']==0
    matrix=_native_expected_matrix_status_v60y([selected],[result],[])
    assert matrix['present_expected_row_count']==1 and matrix['failed_expected_row_count']==1

def test_t024_other_case_target_or_timeout_not_blacklisted(tmp_path):
    model_dir=tmp_path/'model';(model_dir/'benchmark_set').mkdir(parents=True)
    decisions=json.loads((FIXTURE/'original_yolo26m_backend_artifact_decisions.json').read_text())
    target=model_dir/'benchmark_set/backend_artifact_decisions.json';target.write_text(json.dumps(decisions))
    for changes in ({'backend':'deepx_to_trt'},{'case':'b003'}):
        selected={**job(),'backend':'hailo8_to_trt',**changes}
        assert native_job_prerequisite(selected,suite_dir=model_dir/'suite',model_dir=model_dir)['prerequisite_status']=='ready'
    for row in decisions['case_build_requests']:
        if row.get('backend')=='hailo8': row['hailo_case_variant_availability']['part1_error']='compiler timeout'
    target.write_text(json.dumps(decisions))
    assert native_job_prerequisite({**job(),'backend':'hailo8_to_trt'},suite_dir=model_dir/'suite',model_dir=model_dir)['prerequisite_status']=='ready'

def test_t025_mixed_plan_preserves_independent_full(tmp_path):
    blocked=failed_native_result(job(),failure_stage='build',failure_reason='compiler_not_ready')
    full={**job(),'backend':'native_full_tensorrt','case':'full','ok':True}
    matrix=_native_expected_matrix_status_v60y([job(),full],[blocked,full],[])
    assert matrix['present_expected_row_count']==2 and matrix['successful_expected_row_count']==1 and matrix['failed_expected_row_count']==1

def test_t026_partial_build_status_reaches_real_stage_and_log(tmp_path):
    from onnx_splitpoint_tool.native_job_identity import native_build_summary
    from test_v27921_final_selection_cache_preflight import _real_stage_runner
    workflow=_real_stage_runner(tmp_path)
    logs=[]; workflow.log=logs.append
    summary=native_build_summary({'artifacts':'ok','hailo':'partial','deepx':'ok'})
    result=workflow._run_stage('yolo26m','build_backend_artifacts',lambda: ({},summary,'one backend blocked',summary['backend_build_status']))
    assert result.status=='partial' and result.details['ready_backend_count']==2 and result.details['blocked_backend_count']==1
    assert any('[workflow] partial ' in line for line in logs)
    persisted=json.loads((tmp_path/'models/yolo26m/stages/build_backend_artifacts/stage_result.json').read_text())
    assert persisted['status']=='partial'

@pytest.mark.parametrize('profile,artifact_mode,reason',[
    ({'deepx_build':{'classification_preprocessing':'current_scale_only'}},'current_scale_only','deepx_legacy_classification_preprocessing'),
    ({'deepx_build':{'classification_preprocessing':'current_scale_only','diagnostic_only':True},'run_mode':'final'},'current_scale_only','deepx_legacy_classification_preprocessing'),
    ({},'current_scale_only','deepx_classification_preprocessing_artifact_mismatch'),
])
def test_managed_full_legacy_classification_blocked_before_any_inference(tmp_path,monkeypatch,profile,artifact_mode,reason):
    from scripts import native_full_baseline_eval_runner as full
    root,suite=stage_case(tmp_path)
    artifact=suite/'deepx/deepx_m1/full/output_contract.json';artifact.parent.mkdir(parents=True)
    artifact.write_text(json.dumps({'classification_preprocessing':artifact_mode}))
    # Python selection is environment discovery. No runtime/semantic method may
    # execute once the actual profile/recorded model contract rejects this job.
    monkeypatch.setattr(full,'_select_engine_python',lambda *a: ('',{}))
    monkeypatch.setattr(full,'_deepx_full_series_preflight',lambda *a,**k: pytest.fail('legacy job must not infer'))
    monkeypatch.setattr(full,'_row_for_backend',lambda *a,**k: pytest.fail('legacy job must not run a repetition'))
    monkeypatch.setattr(sys,'argv',['full','--root',str(root),'--models','yolo26m','--backends','deepx','--setup-id','device_deepx_A','--comparison-backend','deepx','--comparison-precision','float32_layout_fp16','--repetitions','3','--deepx-classification-profile-json',json.dumps(profile)])
    assert full.main()!=0
    payload=json.loads((root/'analysis_tables/native_full_baseline_eval.json').read_text())
    row=payload['rows'][0]
    assert row['failure_reason'].startswith(reason) and row['repetition_count_attempted']==0
    assert row['setup_id']=='device_deepx_A' and row['case']=='full' and row['ok'] is False

def test_t001_original_fixture_provenance_required_no_missing_skip(tmp_path,monkeypatch):
    from scripts import run_complete_set_replay_v27931 as replay_module
    synthetic_root=tmp_path/'isolated_source'
    folder=synthetic_root/'tests/fixtures/v27931_complete_set';folder.mkdir(parents=True)
    shutil.copy2(FIXTURE/'original_identity_projection.json',folder/'original_identity_projection.json')
    monkeypatch.setattr(replay_module,'ROOT',synthetic_root)
    with pytest.raises(FileNotFoundError): replay_module.replay(None,tmp_path/'out')
    shutil.copy2(FIXTURE/'PROVENANCE.json',folder/'PROVENANCE.json')
    projection=folder/'original_identity_projection.json';projection.write_bytes(projection.read_bytes()+b' ')
    with pytest.raises(ValueError,match='fixture_provenance_invalid'): replay_module.replay(None,tmp_path/'out')

@pytest.mark.parametrize('child_mode',['bad_json','wrong_identity','timeout'])
def test_actual_coordinator_preserves_identity_on_child_failures(tmp_path,monkeypatch,child_mode):
    def prepare(root,suite):
        artifact=suite/'b398/deepx/deepx_m1/part1/model.dxnn';artifact.parent.mkdir(parents=True);artifact.write_bytes(b'synthetic engine existence fixture')
        engine=suite/'native_trt/b398/part2/float32_layout_fp16/part2_float32_layout_fp16.engine';engine.parent.mkdir(parents=True);engine.write_bytes(b'synthetic TRT fixture')
    def child(cmd,kwargs,suite):
        if child_mode=='timeout': raise subprocess.TimeoutExpired(cmd,1)
        result=suite/'native_pipeline/b398/deepx_to_trt/float32_layout_fp16/deepx_native_fifo_e2e_results.json';result.parent.mkdir(parents=True)
        result.write_text('{broken' if child_mode=='bad_json' else json.dumps({'ok':True,'setup_id':'unrelated_device','fps_makespan':99}))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    rc,summary,calls=run_coordinator(tmp_path,monkeypatch,prepare=prepare,child=child)
    row=summary['rows'][0]
    assert rc==3 and len(calls)==1 and len(summary['rows'])==1
    assert native_identity_key(row)==native_identity_key(job())
    assert row['ok'] is False
    assert row['failure_reason'].startswith({'bad_json':'native_result_invalid_json','wrong_identity':'native_job_identity_conflict','timeout':'native_producer_timeout'}[child_mode])

def test_t014_current_target_resolver_rejects_two_setups_without_job_choice():
    from onnx_splitpoint_tool.native_job_identity import select_native_target
    targets=[{'id':'A','accelerator':'deepx_m1','enabled':True},{'id':'B','accelerator':'deepx_m1','enabled':True}]
    with pytest.raises(ValueError,match='native_setup_identity_ambiguous:A,B'): select_native_target(targets,'deepx')
    assert select_native_target([targets[1]],'deepx')['id']=='B'

def test_explicit_legacy_split_diagnostic_cannot_emit_regular_claims(tmp_path,monkeypatch):
    def prepare(root,suite):
        artifact=suite/'b398/deepx/deepx_m1/part1/model.dxnn';artifact.parent.mkdir(parents=True);artifact.write_bytes(b'synthetic engine fixture')
        (artifact.parent/'output_contract.json').write_text(json.dumps({'classification_preprocessing':'current_scale_only'}))
        engine=suite/'native_trt/b398/part2/float32_layout_fp16/part2_float32_layout_fp16.engine';engine.parent.mkdir(parents=True);engine.write_bytes(b'synthetic TRT fixture')
    def child(cmd,kwargs,suite):
        result=suite/'native_pipeline/b398/deepx_to_trt/float32_layout_fp16/deepx_native_fifo_e2e_results.json';result.parent.mkdir(parents=True)
        result.write_text(json.dumps({'ok':True,'setup_id':'device_deepx_A','fps_makespan':99,'repetition_count_attempted':3,'repetition_count_valid':3}))
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    profile={'deepx_build':{'classification_preprocessing':'current_scale_only','diagnostic_only':True}}
    rc,summary,calls=run_coordinator(tmp_path,monkeypatch,['--deepx-classification-profile-json',json.dumps(profile)],prepare=prepare,child=child)
    row=summary['rows'][0]
    assert rc==0 and len(calls)==1 and row['ok'] is True
    assert row['diagnostic_only'] and row['counts_as_benchmark'] is False
    assert row['scientific_claim_exclusion_reason']=='deepx_legacy_classification_preprocessing'
    assert row['performance_claims_emitted'] is False and row['energy_claim_eligible'] is False
