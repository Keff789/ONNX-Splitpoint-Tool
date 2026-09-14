from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_energy_quality_admission import energy_quality_reason_projection
from onnx_splitpoint_tool.workflow.evidence_status import derive_native_evidence_status, project_native_evidence_status
from scripts.run_complete_set_replay_v27931 import replay_energy
from scripts.native_producer_energy_plan import _energy_quality_result_fields
from scripts.run_native_producer_energy_from_summary import _energy_quality_result_projection

ROOT = Path(__file__).resolve().parents[1]


def job(case='b001'):
    return dict(backend='deepx_to_trt', model='model', case=case,
                setup_id='setup', comparison_backend='deepx', precision='fp16')


def derive(rows, excluded=(), expected=None):
    expected = expected or [job()]
    return derive_native_evidence_status(run_mode='standard',
        expected_matrix={'expected_row_count':len(expected), 'present_expected_row_count':len(expected),
                         'successful_expected_row_count':len(expected), 'present_expected_rows':expected},
        validation_payload=None, validation_requested=False, energy_requested=True,
        energy_plan_payload={'rows':[r.get('row',r) for r in rows], 'excluded_rows':list(excluded)},
        energy_results_payload={'rows':rows})


def test_T09_1_disjoint_sets_duplicates_and_conflicts():
    rows = [dict(job(),ok=True)]
    result = derive(rows,[job('b002')],[job(),job('b002'),job('b003')])
    accounting = result['energy']['accounting']
    assert accounting['counts']=={'measured':1,'excluded':1,'missing':1,'unexpected':0}
    sets = [set(map(tuple,x)) for x in accounting['sets'].values()]
    assert all(not a & b for i,a in enumerate(sets) for b in sets[i+1:])
    for duplicate_rows, exclusions, reason in [
        (rows+rows, [], 'results_duplicate_identity'),
        (rows, [job(),job()], 'excluded_duplicate_identity'),
        (rows, [job()], 'measured_excluded_identity_overlap'),
        (rows+[dict(job('other'),ok=True)], [], 'unexpected_identity')]:
        bad=derive(duplicate_rows,exclusions)['energy']['accounting']
        assert bad['status']=='invalid' and reason in bad['identity_conflicts']


def test_T09_2_T09_3_original_55_rows_165_repeats_preserved(tmp_path):
    fixture=ROOT/'tests/fixtures/v27931_complete_set/original_energy_projection.json'
    before=fixture.read_bytes()
    result=replay_energy(None,tmp_path)
    assert fixture.read_bytes()==before
    assert result['measurement_rows']==55 and result['valid_repetitions']==165
    evidence=result['energy_evidence']; axis=evidence['energy']
    assert axis['excluded_identity_contract_valid'] is True
    assert axis['accounting']['counts']=={'measured':55,'excluded':8,'missing':0,'unexpected':0}
    assert axis['accounting']['status']=='consistent'
    assert axis['matrix_expected_count']==63 and axis['measurement_success_count']==55
    assert axis['matrix_measurements_complete'] is False
    assert evidence['scientific_ready'] is False
    assert axis['accounting']['campaign_comparison_released'] is False
    original=json.loads(before)
    metrics=[row['run']['energy_aggregate'] for row in original['rows']]
    assert result['metric_payload_sha256']==hashlib.sha256(json.dumps(metrics,sort_keys=True).encode()).hexdigest()
    assert all(len(row['runs'])==3 for row in metrics)
    assert '55/63' in result['projection']['energy_accounting_summary']


def admission(decision):
    value={key:True for key in ('central_quality_evidence_verified','precision_quality_binding_verified',
        'task_quality_observation_valid','quality_provenance_complete')}
    value.update(local_task_quality_decision=decision, accuracy_gate_pass=decision=='pass',
        admission_scope='native_runtime_observation',diagnostic_only=True,
        claim_comparable=False,energy_claim_eligible=False,quality_claim_result_verified=decision=='pass',
        runtime_observation_reason='incomplete_expected_native_matrix')
    return value


@pytest.mark.parametrize('decision',['pass','fail','inconclusive'])
def test_T09_4_local_quality_separate_from_campaign_in_both_consumers(decision):
    source=admission(decision); before=copy.deepcopy(source)
    planner=_energy_quality_result_fields(source)
    result=_energy_quality_result_projection({'energy_quality_admission':source},
        measurement_started=True,raw_energy_collected=True)
    for projection in (planner,result):
        assert projection['local_task_quality_decision']==decision
        assert projection['energy_quality_qualified'] is False
        reasons=projection['energy_quality_exclusion_reasons']
        assert reasons['campaign']==['incomplete_expected_native_matrix']
        assert reasons['local_quality']==([] if decision=='pass' else ['task_quality_'+decision])
    assert source==before
    legacy=admission('fail');legacy.pop('local_task_quality_decision')
    assert energy_quality_reason_projection(legacy)['local_task_quality_decision']=='not_pass'


def test_T09_5_not_started_partial_and_real_zero_are_distinct():
    source={'energy_quality_admission':admission('pass')}
    not_started=_energy_quality_result_projection(source,measurement_started=False,raw_energy_collected=False)
    partial=_energy_quality_result_projection(source,measurement_started=True,raw_energy_collected=False)
    measured=_energy_quality_result_projection(source,measurement_started=True,raw_energy_collected=True)
    assert len({item['energy_quality_status'] for item in (not_started,partial,measured)})==3
    row=dict(job(),ok=False,measurement_started=False,energy_aggregate_valid_repeat_count=0)
    axis=derive([row])['energy']
    assert axis['measurement_started_count']==0
    assert axis['accounting']['counts']['measured']==0
    assert axis['accounting']['counts']['excluded']==1
    assert axis['accounting']['valid_repetition_count']==0
    zero=dict(job(),ok=True,energy_total_j=0.0,energy_aggregate_valid_repeat_count=3)
    before=copy.deepcopy(zero);evidence=derive([zero]);project_native_evidence_status(evidence)
    assert zero==before and evidence['energy']['measurement_success_count']==1


def test_legacy_classification_exclusion_survives_both_energy_consumers():
    source=admission('pass')
    source['scientific_claim_exclusion_reason']='deepx_legacy_classification_preprocessing'
    for result in (_energy_quality_result_fields(source),
                   _energy_quality_result_projection({'energy_quality_admission':source},
                       measurement_started=True,raw_energy_collected=True)):
        assert result['local_task_quality_decision']=='pass'
        assert result['energy_quality_exclusion_reasons']['local_quality']==[
            'deepx_legacy_classification_preprocessing']
        assert result['campaign_comparison_released'] is False


def test_full_comparison_identity_is_not_inferred_from_setup_spelling():
    from onnx_splitpoint_tool.workflow.evidence_status import _ledger_identity
    from onnx_splitpoint_tool.native_job_identity import native_identity_key
    row=dict(backend='native_full_tensorrt',model='model',case='full',
             setup_id='orin_nx_deepx_m1_01',precision='fp16')
    assert _ledger_identity(row) is None
    assert native_identity_key(row)[4] == ''
    row['comparison_backend']='deepx'
    assert _ledger_identity(row) is not None
