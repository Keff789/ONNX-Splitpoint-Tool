from __future__ import annotations

import copy
from pathlib import Path

import yaml

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan, execution_plan_text, execution_plan_markdown
from onnx_splitpoint_tool.workflow.scientific_reporting import _ranking_method_comparison, _ranking_observed_scope
from tests.test_v27530_score_independent_native_ranking_audit import _audit_contract_for_cases, _native_ranking_row


FIXTURE=Path(__file__).parent/'fixtures/v27931_complete_set/original_resolved_profile.yaml'


def test_T10_1_original_profile_and_shortfall_are_unchanged():
    before=FIXTURE.read_bytes();profile=yaml.safe_load(before)
    plan=build_effective_execution_plan(profile)
    assert plan['cases_per_model']==1
    assert plan['ranking_enabled'] is True
    assert {warning['id'] for warning in plan['warnings']} >= {'ranking_candidate_shortfall'}
    assert plan['ranking_candidate_shortfall']==2
    assert FIXTURE.read_bytes()==before


def test_T10_2_coverage_has_explicit_scope_without_expansion_or_ranking():
    profile=yaml.safe_load(FIXTURE.read_text());baseline=build_effective_execution_plan(profile)
    profile['purpose']='coverage_integration';before=copy.deepcopy(profile)
    plan=build_effective_execution_plan(profile)
    assert profile==before
    assert plan['ranking_enabled'] is False and plan['ranking_requested'] is True
    assert plan['candidate_counts_by_model']==baseline['candidate_counts_by_model']
    assert plan['experiment_scope']['observed_comparable_valid_candidates'] is None
    assert plan['experiment_scope']['ranking_admission_status']=='not_requested_for_integration'
    for text in (execution_plan_text(plan),execution_plan_markdown(plan)):
        assert 'coverage_integration' in text and 'separate' in text
    predictions,_,policy=_audit_contract_for_cases(['b001','b002','b003'])
    rows=[_native_ranking_row(case,10.0+i) for i,case in enumerate(['b001','b002','b003'])]
    details,macro,summary=_ranking_method_comparison(rows,predictions,profile,policy)
    assert details==macro==[]
    assert summary['status']=='not_requested_for_integration'


def test_T10_3_replicates_and_incompatible_strata_do_not_add_candidates():
    _,profile,policy=_audit_contract_for_cases(['b001','b002','b003'])
    policy['minimum_candidates_for_correlation']=3
    rows=[_native_ranking_row('b001',10.0),_native_ranking_row('b002',12.0)]
    rows += [copy.deepcopy(rows[0]),copy.deepcopy(rows[0])]
    other=copy.deepcopy(rows[0]);other['setup_id']='another_physical_setup';rows.append(other)
    precision=copy.deepcopy(rows[0]);precision['precision']='int8';precision['runtime_precision_identity']='int8';rows.append(precision)
    scope=_ranking_observed_scope(rows,profile,policy)
    assert max(group['distinct_candidate_count'] for group in scope['groups'])==2
    assert len(scope['groups'])==3
    assert all(group['status']=='insufficient_candidates' for group in scope['groups'])


def test_T10_4_insufficient_or_invalid_candidates_have_no_claim_metrics():
    predictions,profile,policy=_audit_contract_for_cases(['b001','b002','b003'])
    profile['purpose']='ranking_experiment'
    policy['minimum_candidates_for_correlation']=3
    rows=[_native_ranking_row('b001',10.0),_native_ranking_row('b002',12.0)]
    failed=_native_ranking_row('b003',8.0)
    failed.update(ranking_eligible=False,task_quality_status='fail',accuracy_gate_pass=False,
                  failure_reason='KNOWN_INFEASIBLE',runtime_success=False)
    rows.append(failed)
    details,macro,summary=_ranking_method_comparison(rows,predictions,profile,policy)
    assert details
    assert all(row.get('spearman_rho') is None and row.get('kendall_tau_b') is None for row in details)
    assert all(row.get('hit_at_1') is None for row in details)
    scope=summary['ranking_scope']
    assert all(group['status']=='insufficient_candidates' for group in scope['groups'])
    assert any('KNOWN_INFEASIBLE' in exclusion['observed_reasons'] for group in scope['groups'] for exclusion in group['excluded_candidates'])
