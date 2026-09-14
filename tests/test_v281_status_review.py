"""Independent review regression: exclusions cannot conceal conflicting rows."""
import copy
import pytest
from onnx_splitpoint_tool.native_job_identity import known_build_exclusion, project_known_build_exclusions
from onnx_splitpoint_tool.workflow.evidence_status import derive_native_evidence_status
from onnx_splitpoint_tool.workflow.runner import _native_expected_matrix_status_v60y
from test_v281_status_reports import blocked, resolved_matrix_and_validation


@pytest.mark.parametrize('flag',['runtime_ok','measurement_started','runtime_started','compiler_dispatched'])
def test_exclusion_contradiction_survives_runner_projection(flag):
    original=next(row for row in blocked() if known_build_exclusion(row))
    changed=copy.deepcopy(original);changed[flag]=True
    assert not known_build_exclusion(changed)
    matrix=_native_expected_matrix_status_v60y([original],[changed],[])
    assert matrix['excluded_expected_row_count']==0
    assert matrix['failed_expected_row_count']==1
    assert not matrix['technical_execution_complete']


@pytest.mark.parametrize('problem',['duplicate_present','missing_present','missing_success','success_exclusion_overlap',
    'contradictory_failed','duplicate_excluded','false_counts','missing_row_list','unresolved_identity'])
def test_exclusion_completion_requires_distinct_complete_noncontradictory_rows(problem):
    matrix,validation=resolved_matrix_and_validation();matrix=copy.deepcopy(matrix)
    if problem=='duplicate_present':matrix['present_expected_rows'][1]=copy.deepcopy(matrix['present_expected_rows'][0])
    elif problem=='missing_present':matrix['present_expected_rows'].pop()
    elif problem=='missing_success':matrix['successful_expected_rows'].pop()
    elif problem=='success_exclusion_overlap':matrix['successful_expected_rows'][0]=copy.deepcopy(matrix['excluded_expected_rows'][0])
    elif problem=='contradictory_failed':
        row=copy.deepcopy(matrix['excluded_expected_rows'][0]);row['upstream_build_observation']['compiler_dispatched']=True
        matrix['failed_expected_rows']=[row];matrix['failed_expected_row_count']=1
    elif problem=='duplicate_excluded':
        # A forged extra expected/present count must not be satisfied by copies.
        matrix['present_expected_rows'].append(copy.deepcopy(matrix['excluded_expected_rows'][0]))
        matrix['expected_row_count']+=1;matrix['present_expected_row_count']+=1
    elif problem=='false_counts':matrix['successful_expected_row_count']+=1;matrix['expected_row_count']+=1;matrix['present_expected_row_count']+=1
    elif problem=='missing_row_list':matrix['missing_expected_rows']=[copy.deepcopy(matrix['present_expected_rows'][0])]
    elif problem=='unresolved_identity':matrix['identity_unresolved_rows']=[{'identity_status':'identity_conflict'}]
    projected=project_known_build_exclusions(matrix)
    assert not projected['technical_execution_complete']
    if problem=='contradictory_failed':
        assert projected['failed_expected_row_count']==1
        assert projected['excluded_expected_row_count']==5
    evidence=derive_native_evidence_status(run_mode='standard',expected_matrix=matrix,validation_payload=validation,
        validation_requested=True,energy_requested=False)
    assert evidence['technical_status']!='complete'


def test_exact_six_exclusions_project_idempotently_and_do_not_change_quality():
    matrix,validation=resolved_matrix_and_validation()
    projected=project_known_build_exclusions(matrix)
    assert projected==matrix
    assert projected['exclusion_accounting_valid']
    assert not projected['exclusion_projection_errors']
    assert projected['excluded_expected_row_count']==6 and projected['successful_expected_row_count']==120
    assert projected['technical_execution_complete']
    evidence=derive_native_evidence_status(run_mode='standard',expected_matrix=matrix,validation_payload=validation,
        validation_requested=True,energy_requested=False)
    assert evidence['task_quality']['fail_count']==120
    assert evidence['scientific_ready'] is False


def test_legacy_numeric_only_without_exclusion_preserves_terminal_counts():
    matrix={'expected_row_count':2,'present_expected_row_count':2,'successful_expected_row_count':2,
        'failed_expected_row_count':0,'missing_expected_row_count':0}
    result=project_known_build_exclusions(matrix)
    assert result['technical_execution_complete']
    assert all(field not in result for field in ('present_expected_rows','failed_expected_rows','excluded_expected_rows'))
