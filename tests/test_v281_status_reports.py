from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from onnx_splitpoint_tool.native_job_identity import (
    apply_known_build_disposition, known_build_exclusion, native_identity_key,
    project_known_build_exclusions, required_profile_build_exclusions,
)
from onnx_splitpoint_tool.workflow.results import _apply_quality_source_identity_v265
from onnx_splitpoint_tool.workflow.evidence_status import derive_native_evidence_status, blocking_status, project_native_evidence_status
from onnx_splitpoint_tool.workflow.runner import _native_expected_matrix_status_v60y, _native_evidence_status_v60o

FIXTURE = Path(__file__).parent / 'fixtures/v281_status'
def read(name): return json.loads((FIXTURE / name).read_text())
def blocked(): return read('night_blocked_native_rows.json')
def actual_matrix():
    old = read('night_native_matrix.json')
    expected = old['present_expected_rows']
    by_key = {native_identity_key(row): row for row in blocked()}
    observations = [by_key.get(native_identity_key(row), {**row, 'ok': row['actual_ok'], 'status':row['actual_status']}) for row in expected]
    return _native_expected_matrix_status_v60y(expected, observations, [])

def test_actual_59_request_aliases_resolve_without_mutating_source():
    rows=read('night_quality_identity_rows.json'); before=copy.deepcopy(rows)
    assert len(rows)==59
    original_errors = [error for row in rows for error in row['quality_identity_errors']]
    assert sum('source_run_id' in error for error in original_errors)==52
    assert sum('runtime_precision_identity' in error for error in original_errors)==7
    for original in rows:
        row=copy.deepcopy(original);_apply_quality_source_identity_v265(row)
        assert row['quality_request_identities_by_variant']
        assert all(identity['identity_valid'] and not identity['identity_errors'] for identity in row['quality_request_identities_by_variant'].values())
    assert rows==before

@pytest.mark.parametrize('field,value', [('source_run_id','hailo8_to_trt'),('setup_id','another_setup')])
def test_actual_alias_projection_rejects_real_family_or_setup_drift(field,value):
    row=next(r for r in read('night_quality_identity_rows.json') if r['backend']=='hailo10_to_tensorrt')
    request=row['task_quality_gate']['quality_input_request'];request[field]=value
    _apply_quality_source_identity_v265(row)
    assert any(not i['identity_valid'] for i in row['quality_request_identities_by_variant'].values())

@pytest.mark.parametrize('kind',['different_hash','malformed_hash','wrong_artifact_kind'])
def test_deepx_precision_rejects_real_contract_changes(kind):
    row=next(r for r in read('night_quality_identity_rows.json') if r['case_id']=='full')
    gate=row['task_quality_gate']['quality_input_request']
    pending=[gate]; changed=False
    while pending:
        item=pending.pop()
        if isinstance(item,dict):
            if item.get('schema')=='onnx-splitpoint/deepx-runtime-precision-contract':
                item['artifact_kind' if kind=='wrong_artifact_kind' else 'artifact_sha256'] = ('hef' if kind=='wrong_artifact_kind' else 'f'*64 if kind=='different_hash' else 'invalid')
                changed=True
            pending.extend(item.values())
    assert changed
    _apply_quality_source_identity_v265(row)
    assert any(not i['identity_valid'] for i in row['quality_request_identities_by_variant'].values())

def test_actual_126_rows_keep_six_exclusions_and_seven_real_failures():
    matrix=actual_matrix()
    assert matrix['expected_row_count']==126 and matrix['present_expected_row_count']==126
    assert matrix['successful_expected_row_count']==113
    assert matrix['excluded_expected_row_count']==6
    assert matrix['failed_expected_row_count']==7
    assert matrix['missing_expected_row_count']==0
    assert not matrix['technical_execution_complete'] and not matrix['execution_success_complete']
    assert len(matrix['failed_expected_rows'])==7
    assert all(known_build_exclusion(row) and row['actual_ok'] is False and row['repetition_count_attempted']==0 for row in matrix['excluded_expected_rows'])
    assert project_known_build_exclusions(matrix)==matrix

@pytest.mark.parametrize('problem', ['record_hash','key_hash','category','family','model','boundary','source_sha','context_model','dispatched','attempted','success','conflict','missing_record'])
def test_exclusion_rejects_malformed_or_contradictory_evidence(problem):
    row=next(r for r in blocked() if known_build_exclusion(r)); obs=row['upstream_build_observation']; decision=obs['build_evidence']
    if problem=='record_hash':decision['record']['record_sha256']='0'*64
    elif problem=='key_hash':decision['key_sha256']='0'*64
    elif problem=='category':decision['state']='TRANSIENT_INFRASTRUCTURE'
    elif problem=='family':obs['backend']='hailo10h'
    elif problem=='model':obs['model_id']='another_model'
    elif problem=='boundary':obs['boundary']='b040'
    elif problem=='source_sha':decision['context']['full_source_onnx_sha256']='0'*64
    elif problem=='context_model':decision['context']['model_id']='another_model'
    elif problem=='dispatched':obs['compiler_dispatched']=True
    elif problem=='attempted':row['repetition_count_attempted']=1
    elif problem=='success':row['ok']=True
    elif problem=='conflict':row['identity_conflicts']=['setup_id']
    elif problem=='missing_record':decision.pop('record')
    assert not known_build_exclusion(row)
    assert apply_known_build_disposition(row).get('disposition')!='excluded_known_build'

@pytest.mark.parametrize('edit_manifest', [False, True])
def test_existing_negative_cannot_be_relabelled_to_a_different_split(edit_manifest):
    row=next(r for r in blocked() if r.get('case')=='b398' and known_build_exclusion(r))
    row['case']='b040'
    row['planned_native_identity']['case']='b040'
    observation=row['upstream_build_observation']
    observation['boundary']='b040'
    context=observation['build_evidence']['context']
    context['boundary']=40
    if edit_manifest:
        context['split_manifest']['boundary']=40
        context['split_manifest']['boundary_index']=40
    assert not known_build_exclusion(row)
    assert apply_known_build_disposition(row).get('disposition')!='excluded_known_build'

def test_ten_missing_profile_outcomes_resolve_six_only():
    required=read('night_required_missing_rows.json')
    readiness={'blocked_jobs':[r['upstream_build_observation'] for r in blocked() if r.get('upstream_build_observation')]}
    excluded=required_profile_build_exclusions(required,readiness)
    assert len(required)==10 and len(excluded)==6
    assert all(row['measurement_values_synthesized'] is False and row['runtime_executable'] is False for row in excluded)
    strict=[dict(row,success_required=True) for row in required]
    assert required_profile_build_exclusions(strict,readiness)==[]
    contradictory=[dict(row,runtime_ok=True) for row in excluded]
    assert required_profile_build_exclusions(required,readiness,contradictory)==[]

def resolved_matrix_and_validation():
    matrix=actual_matrix()
    # Counterfactual unit-test observations only: successful actual completion
    # of the seven real failures must be supplied before technical release.
    present=[]
    for row in matrix['present_expected_rows']:
        item=copy.deepcopy(row)
        if not known_build_exclusion(item):item.update(ok=True,status='ok',actual_ok=True,actual_status='ok')
        present.append(item)
    fixed=_native_expected_matrix_status_v60y(present,present,[])
    validation={'technical_chain_complete':True,'technical_error_count':0,'rows':[
        dict(row,semantic_available=True,semantic_ok=True,task_quality_status='fail',claim_ok=False)
        for row in fixed['successful_expected_rows']]}
    return fixed,validation

@pytest.mark.parametrize('mode',['smoke','standard','final'])
def test_terminal_exclusions_and_negative_quality_complete_technical_axis_only(mode):
    matrix,validation=resolved_matrix_and_validation()
    assert matrix['successful_expected_row_count']==120 and matrix['excluded_expected_row_count']==6
    assert matrix['technical_execution_complete'] is True
    assert matrix['execution_success_complete'] is False
    evidence=derive_native_evidence_status(run_mode=mode,expected_matrix=matrix,validation_payload=validation,validation_requested=True,energy_requested=False)
    assert evidence['technical_status']=='complete'
    assert evidence['runtime']['status']=='complete_with_exclusions'
    assert evidence['task_quality']['fail_count']==120
    assert not evidence['positive_performance_claim_available']
    assert not evidence['scientific_ready']
    assert blocking_status(evidence,run_mode=mode)==''
    assert project_native_evidence_status(evidence)['technical_complete'] is True
    assert _native_evidence_status_v60o({'rows':[dict(r,ok=r['actual_ok']) for r in matrix['present_expected_rows']]})['evidence_status']=='complete'

def test_new_real_error_is_not_hidden_by_known_exclusions():
    matrix,validation=resolved_matrix_and_validation()
    validation['technical_error_count']=1
    evidence=derive_native_evidence_status(run_mode='standard',expected_matrix=matrix,validation_payload=validation,validation_requested=True,energy_requested=False)
    assert evidence['technical_status']=='partial'
    assert evidence['technical_quality_failure'] is True
    assert blocking_status(evidence,run_mode='standard')=='partial'

def report_module():
    path=Path(__file__).parents[1]/'scripts/native_producer_final_report.py'
    spec=importlib.util.spec_from_file_location('v281_final_report',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_final_report_retains_exclusions_through_diagnostic_and_aggregate_projection():
    module=report_module()
    row=apply_known_build_disposition(next(r for r in blocked() if known_build_exclusion(r)))
    diag=module._diagnostic_fields(row,row)
    merged={**row,**diag}
    assert known_build_exclusion(merged)
    output=module._aggregate_repetitions([merged])
    assert len(output)==1 and known_build_exclusion(output[0])
    assert output[0]['repetition_status']=='not_started'
    assert output[0]['ok'] is False and output[0]['fps_makespan'] is None
    matrix,validation=resolved_matrix_and_validation()
    rows=[dict(row,ok=row['actual_ok']) for row in matrix['present_expected_rows']]
    result=module._evidence_summary(rows,matrix)
    assert result['matrix_counts_consistent'] is True
    assert result['excluded_expected_row_count']==6 and result['failed_expected_row_count']==0
    assert result['technical_execution_complete'] is True and result['matrix_complete'] is False
    assert result['evidence_status']=='complete_with_exclusions'

@pytest.mark.parametrize('mode',['standard','final'])
def test_final_workflow_status_uses_terminal_axis_without_promoting_quality(tmp_path,mode):
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
    matrix,validation=resolved_matrix_and_validation()
    evidence=derive_native_evidence_status(run_mode=mode,expected_matrix=matrix,validation_payload=validation,validation_requested=True,energy_requested=False)
    runner=EvaluationWorkflowRunner(WorkflowOptions(profile='',out=str(tmp_path)))
    runner.run_dir=tmp_path/'run';runner.profile_payload={'execution_preset':{'id':mode}}
    runner.stage_results=[{'stage':'run_native_producers','status':'ok','model_id':None,'details':{'native_evidence_status':evidence}}]
    status,decision=runner._derive_final_status()
    assert status=='ok' and decision['blocking_reason_count']==0
    assert evidence['task_quality']['fail_count']==120 and not evidence['scientific_ready']

def test_actual_six_exclusions_survive_persist_read_aggregate_matrix(tmp_path):
    from onnx_splitpoint_tool.workflow.runner import _persist_native_blocked_rows
    module=report_module()
    source=[dict(apply_known_build_disposition(row),execution_mode='native_split') for row in blocked() if known_build_exclusion(row)]
    expected=[{**row['planned_native_identity'],'execution_mode':'native_split'} for row in source]
    artifacts,roots={},[]
    _persist_native_blocked_rows(tmp_path,source,expected_native_rows=expected,
        artifact_paths=artifacts,collected_roots=roots,repetition_count_requested=3)
    reread=module._aggregate_repetitions([*module._rows_from_native_fifo_runner(roots[0]),*module._rows_from_hailo10(roots[0])])
    assert len(reread)==6 and all(known_build_exclusion(row) for row in reread)
    assert all(row['repetition_count_attempted']==0 and row['fps_makespan'] is None for row in reread)
    matrix=_native_expected_matrix_status_v60y(expected,reread,[])
    assert matrix['excluded_expected_row_count']==6 and matrix['failed_expected_row_count']==0
    assert matrix['technical_execution_complete'] is True
    assert matrix['successful_expected_row_count']==0 and matrix['execution_success_complete'] is False
