from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import pytest
from scripts import native_full_baseline_eval_runner as full
from scripts import native_producer_validate_visualize as validator
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner, _vendor_full_completed_quality_fields_v282
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.evidence_status import workflow_completion_projection, project_historical_workflow_status, blocking_status

FIXTURES=Path(__file__).parent/'fixtures/v282_full_quality'
def original(): return json.loads((FIXTURES/'original_yolo26m.json').read_text())
def prepare(backend):
    fixture=original(); evidence=fixture['backends'][backend]
    row=copy.deepcopy(evidence['original_full_row']); binding_set=copy.deepcopy(evidence['original_binding_set'])
    binding=binding_set['bindings_by_backend_model'][row['backend']+'|yolo26m']
    result=next(r for r in fixture['central_quality_results'] if validator._canonical_json_sha256(r)==binding['central_quality_result_sha256'])
    binding.update(_vendor_full_completed_quality_fields_v282(result,result['request_identity']))
    binding.pop('binding_sha256');binding['binding_sha256']=validator._canonical_json_sha256(binding)
    binding_set.pop('binding_set_sha256');binding_set['binding_set_sha256']=validator._canonical_json_sha256(binding_set)
    ns=argparse.Namespace(quality_request_binding_set_data=binding_set,setup_id=row['setup_id'],comparison_backend=row['comparison_backend'])
    return fixture,row,binding,result,ns

@pytest.mark.parametrize('backend',['hailo8','hailo10h'])
def test_actual_original_endpoint_reject_reproduced_and_exact_completed_join_repaired(backend):
    fixture,row,binding,result,ns=prepare(backend)
    before=copy.deepcopy(fixture)
    old_ns=argparse.Namespace(**vars(ns));old_ns.quality_request_binding_set_data=fixture['backends'][backend]['original_binding_set']
    old,status=full._attach_full_quality_request_binding(copy.deepcopy(row),row['full_command_contract'],model='yolo26m',ns=old_ns)
    assert status=='quality_request_binding_failed' and 'endpoint_contract_hash_mismatch' in old['quality_request_binding_errors']
    assert binding['variant']=='full' and binding['source_case_id']=='b038'
    raw_endpoint=row['endpoint_contract_hash']
    row,status=full._attach_full_quality_request_binding(row,row['full_command_contract'],model='yolo26m',ns=ns)
    assert status=='quality_request_binding_verified_exact'
    assert row['endpoint_contract_hash']==raw_endpoint!=binding['endpoint_contract_hash']
    assert result==next(r for r in before['central_quality_results'] if r==result)
    contract=row['full_command_contract'];contract.update({k:row[k] for k in ('quality_request_binding','quality_request_binding_sha256','quality_request_binding_set_sha256')})
    contract.pop('contract_sha256');contract['contract_sha256']=validator._canonical_json_sha256(contract)
    rec=copy.deepcopy(fixture['backends'][backend]['original_validation_row'])
    for key in list(rec):
        if key.startswith(('quality_','central_','precision_quality','vendor_full_')): rec.pop(key,None)
    rec['full_command_contract']=contract
    validator._merge_vendor_full_quality_evidence(rec,row)
    validator._bind_central_quality_evidence(rec,[result],AccuracyGatePolicy.from_mapping(fixture['policy']))
    assert rec['central_quality_evidence_verified'] is True,rec.get('quality_first_binding_errors')
    assert rec['central_quality_binding_status']=='exact_identity_match'
    assert fixture==before

@pytest.mark.parametrize('field,value',[('variant','composed'),('setup_id','other'),('model_id','yolo26s'),('runtime_precision_identity','hailo_hef_sha256:'+'0'*64),('completed_task_endpoint_contract_hash','0'*64)])
def test_actual_completed_join_rejects_identity_and_endpoint_drift(field,value):
    _,row,binding,_,ns=prepare('hailo8');binding[field]=value
    _,status=full._attach_full_quality_request_binding(row,row['full_command_contract'],model='yolo26m',ns=ns)
    assert status=='quality_request_binding_failed'

@pytest.mark.parametrize('field',['iou_threshold','decoder_id'])
def test_actual_completed_join_rejects_changed_runtime_decoder(field):
    _,row,_,_,ns=prepare('hailo8');row['frozen_host_postprocess_contract'][field]='changed'
    failed,status=full._attach_full_quality_request_binding(row,row['full_command_contract'],model='yolo26m',ns=ns)
    assert status=='quality_request_binding_failed'
    assert 'endpoint_contract_hash_mismatch' in failed['quality_request_binding_errors']

def test_quality_endpoint_mirror_conflict_is_not_rewritten():
    _,_,_,result,_=prepare('hailo8');identity=copy.deepcopy(result['request_identity']);identity['completed_task_endpoint_contract_hash']='0'*64
    with pytest.raises(ValueError,match='alias_mismatch'):
        _vendor_full_completed_quality_fields_v282(result,identity)

@pytest.mark.parametrize('status,label,severity',[
 ('ok','Abgeschlossen mit Qualitätswarnungen','warning'),
 ('partial','Abgeschlossen mit technischen Teilausfällen','warning'),
 ('failed','Lauf fehlgeschlagen','error'),('cancelled','Lauf abgebrochen','cancelled')])
def test_one_completion_projection_preserves_negative_quality(status,label,severity):
    quality={'completed_count':150,'request_count':150,'decision_counts':{'pass':81,'fail':46,'inconclusive':23},'campaign_quality_missing_count':11}
    evidence={'known_build_excluded_count':6,'validation_technical_error_count':6,'runtime':{'expected_count':126,'present_count':126,'successful_count':115,'failed_count':5,'missing_count':0},'energy':{'requested':True,'measurement_started_count':115,'measurement_success_count':113,'measurement_failed_count':2,'claim_eligible_count':0}}
    before=copy.deepcopy((quality,evidence))
    projection=workflow_completion_projection(status,native_evidence=evidence,central_quality=quality)
    assert projection['label']==label and projection['severity']==severity
    counts=projection['counts'];assert counts['native_selected']==counts['native_measured']+counts['native_excluded']+counts['native_failed_or_blocked']==126
    assert counts['energy_verified']==113 and counts['energy_started']==115
    assert (quality,evidence)==before

@pytest.mark.parametrize('stage,model,status,expected',[
 ('run_benchmarks','yolo26m','failed','partial'),('generate_report',None,'failed','failed'),
 ('artifact_index_closure',None,'failed','failed'),('run_benchmarks','yolo26m','cancelled','cancelled')])
def test_workflow_local_partial_global_failed_cancelled(tmp_path,stage,model,status,expected):
    runner=EvaluationWorkflowRunner(WorkflowOptions(profile='',out=str(tmp_path)))
    runner.run_dir=tmp_path;runner.profile_payload={'execution_preset':{'id':'final'}}
    runner.stage_results=[{'stage':stage,'model_id':model,'status':status}]
    actual,_=runner._derive_final_status();assert actual==expected

@pytest.mark.parametrize('mode',['smoke','standard','final'])
def test_native_gap_is_partial_and_global_integrity_remains_failed(mode):
    assert blocking_status({'technical_quality_failure':True},run_mode=mode)=='partial'
    assert blocking_status({'global_integrity_failure':True},run_mode=mode)=='failed'

def test_historical_projection_requires_proven_local_blockers_and_never_mutates():
    source={'status':'failed','blocking_reasons':[{'kind':'native_evidence','stage':'run_native_producers'}]}
    evidence={'runtime':{'successful_count':115}};before=copy.deepcopy(source)
    assert project_historical_workflow_status(source,evidence)['projected_technical_status']=='partial'
    assert source==before
    source['blocking_reasons'].append({'stage':'generate_report','status':'failed'})
    assert project_historical_workflow_status(source,evidence)['projected_technical_status']=='failed'

def test_actual_126_runtime_115_energy_6_validation_gaps_are_distinct():
    actual=json.loads((FIXTURES/'original_status_evidence.json').read_text())
    evidence=actual['original_evidence_status'];runtime=evidence['runtime'];energy=evidence['energy']
    assert runtime['expected_count']==126 and runtime['successful_count']==115 and runtime['excluded_count']==6 and runtime['failed_count']==5
    assert energy['measurement_started_count']==115 and energy['measurement_success_count']==113
    conflicts=actual['original_dump_conflict_validation_rows']
    assert len(conflicts)==4 and all(validator._technical_quality_error(row) for row in conflicts)
    from onnx_splitpoint_tool.native_job_identity import known_build_exclusion
    excluded=[row for row in actual['original_nonexecuted_runtime_rows'] if known_build_exclusion(row)]
    assert len(excluded)==6
    assert all(not validator._technical_quality_error(row) for row in excluded)
    binding_rows=[row['original_validation_row'] for row in original()['backends'].values()]
    assert all(validator._technical_quality_error(row) for row in binding_rows)
    assert len(conflicts)+len(binding_rows)==6
    historical=project_historical_workflow_status({'status':actual['original_status'],'blocking_reasons':actual['original_blocking_reasons']},evidence)
    assert historical['applied'] and historical['projected_technical_status']=='partial'

@pytest.mark.parametrize('decision',['fail','inconclusive'])
def test_valid_negative_quality_and_numeric_similarity_are_not_technical_errors(decision):
    row={'backend':'cpu','task_quality_status':decision,'semantic_available':True,'semantic_ok':False,'numerical_similarity_pass':False,'tensor_ok':True,'structural_contract_pass':True,'runtime_executable':True,'buildable':True}
    assert not validator._technical_quality_error(row)

@pytest.mark.parametrize('decision',['pass','fail','inconclusive'])
def test_completed_quality_gate_status_is_not_a_conflicting_decision(decision):
    from onnx_splitpoint_tool.workflow.evidence_status import _task_quality_decision
    assert _task_quality_decision({'task_quality_status':decision,'task_quality_gate':{'status':'completed','decision':decision}})[0]==decision
    assert _task_quality_decision({'task_quality_gate':{'status':'completed'}})[0]=='unavailable'

def test_original_exclusions_survive_smoke_projection_as_negative_outcomes():
    from onnx_splitpoint_tool.native_job_identity import known_build_exclusion
    actual=json.loads((FIXTURES/'original_status_evidence.json').read_text())
    rows=[r for r in actual['original_nonexecuted_runtime_rows'] if known_build_exclusion(r)]
    validator._apply_smoke_diagnostic_policy(rows)
    assert len(rows)==6 and all(r['status']=='excluded_known_build' for r in rows)
    assert all(not validator._technical_quality_error(r) for r in rows)

def test_partial_runtime_keeps_quality_fail_and_claim_release_closed():
    from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_decision_axes
    result=_scientific_decision_axes({'rows':[]},technical_execution_status='partial',central_quality_reporting={'technical_status':'ok','quality_decision':'fail'})
    assert result['scientific_status']=='fail' and result['scientific_pass'] is False
    assert result['technical_execution_status']=='partial'

@pytest.mark.parametrize('stage,status,expected',[('run_native_producers','cancelled','cancelled'),('generate_report','failed','failed')])
def test_cancellation_and_publication_failure_cannot_be_demoted_by_partial_native_evidence(tmp_path,stage,status,expected):
    runner=EvaluationWorkflowRunner(WorkflowOptions(profile='',out=str(tmp_path)))
    runner.run_dir=tmp_path;runner.profile_payload={}
    runner.stage_results=[{'stage':stage,'status':status,'details':{'native_evidence_status':{'technical_quality_failure':True}}}]
    assert runner._derive_final_status()[0]==expected
