from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_output_endpoint import load_authoritative_output_contract
from onnx_splitpoint_tool.workflow.scientific_reporting import project_central_quality_status


IDENTITY_FIELDS = [
    'model_id', 'source_run_id', 'setup_id', 'backend', 'variant',
    'execution_role', 'performance_claims_emitted',
]


def _summary(*, full_only: bool = False, split_decision: str = 'fail') -> dict:
    identity = dict(model_id='resnet50', source_run_id='native_full_tensorrt',
                    setup_id='orin_nx_hailo8_01', backend='tensorrt', variant='full',
                    execution_role='full_quality_only', performance_claims_emitted=False)
    def result(fields: dict, decision: str) -> dict:
        return {**fields, 'run_id': fields['source_run_id'], 'case_id': 'full',
                'task': 'classification', 'status': 'completed',
                'technical_status': 'completed', 'decision': decision,
                'scientific_status': decision, 'n': 500,
                'primary': {'metric': 'top1_accuracy', 'candidate': .8,
                            'reference': .8, 'delta': 0., 'margin': .01,
                            'ci_low': 0., 'ci_high': 0., 'decision': decision},
                'guardrails': {'top5_accuracy': {'decision': 'pass'}}}
    split = {**identity, 'source_run_id': 'hailo8_to_trt', 'backend': 'hailo8_to_trt',
             'variant': 'composed', 'execution_role': ''}
    return {
        'status': 'ok', 'request_count': 2, 'failed_count': 0,
        'quality_applicable': 1, 'quality_completed': 1,
        'merge': {'unmatched_result_count': 0},
        'quality_acceptance_identity_contract': {
            'schema': 'onnx-splitpoint/' + (
                'full-only-quality-acceptance-identity-contract' if full_only else
                'standard-setup-local-tensorrt-quality-acceptance-identity-contract'),
            'schema_version': 1, 'execution_scope': 'full_only' if full_only else
                'standard_quality_setup_local_tensorrt',
            'identity_key_fields': IDENTITY_FIELDS,
            'model_ids': ['resnet50'], 'expected_identities': [identity],
        },
        'results': [result(identity, 'pass'), result(split, split_decision)],
    }


def test_standard_split_failure_does_not_hide_behind_full_reference() -> None:
    result = project_central_quality_status(_summary())
    assert result['technical_status'] == 'ok'
    assert result['quality_decision'] == 'fail'
    assert result['scientific_pass'] is False
    assert result['full_reference_quality_decision'] == 'pass'
    assert result['full_reference_scientific_pass'] is True
    assert result['campaign_quality_decision'] == 'fail'
    assert result['aggregate_full_result_count'] == 1


@pytest.mark.parametrize('missing_source', ['quality_missing', 'evidence', 'coverage'])
def test_missing_required_quality_prevents_campaign_pass(missing_source: str) -> None:
    summary = _summary(split_decision='pass')
    if missing_source == 'quality_missing':
        summary['quality_missing'] = 1
    elif missing_source == 'evidence':
        summary['evidence_state_summary'] = {'quality_missing': 1}
    else:
        summary['quality_applicable'] = 2
    result = project_central_quality_status(summary)
    assert result['quality_decision'] == 'not_evaluated'
    assert result['scientific_pass'] is False
    assert result['results_complete'] is False
    assert result['full_reference_quality_decision'] == 'pass'
    assert result['campaign_quality_missing_count'] == 1


def test_complete_standard_pass_and_inconclusive_are_distinct() -> None:
    for decision in ['pass', 'inconclusive']:
        result = project_central_quality_status(_summary(split_decision=decision))
        assert result['quality_decision'] == decision
        assert result['scientific_pass'] is (decision == 'pass')


def test_explicit_full_only_canary_scope_remains_exact() -> None:
    summary = _summary(full_only=True)
    summary['quality_missing'] = 19
    result = project_central_quality_status(summary)
    assert result['quality_decision'] == 'pass'
    assert result['campaign_quality_decision'] == 'not_applicable'
    summary['results'][0]['decision'] = 'fail'
    result = project_central_quality_status(summary)
    assert result['quality_decision'] == 'fail'


def test_standard_missing_companion_contract_still_blocks_all_passes() -> None:
    summary = _summary(split_decision='pass')
    second = copy.deepcopy(summary['quality_acceptance_identity_contract']['expected_identities'][0])
    second['setup_id'] = 'orin_nx_hailo10_01'
    summary['quality_acceptance_identity_contract']['expected_identities'].append(second)
    result = project_central_quality_status(summary)
    assert result['quality_decision'] == 'not_evaluated'
    assert result['aggregate_missing_identity_count'] == 1
    assert result['full_reference_quality_decision'] == 'not_evaluated'


def test_single_pending_hailo_full_contract_is_invalid_not_duplicate(tmp_path: Path) -> None:
    payload = {'model_id': 'resnet50', 'task': 'classification', 'contracts': [{
        'model_id': 'resnet50', 'backend': 'hailo8', 'variant': 'full',
        'task': 'classification', 'contract_status': 'pending_build_or_prepare',
        'artifact_binding_status': 'pending_receipt_validation',
        'artifact_binding_error': 'hailo_full_receipt_not_verified',
        'endpoint_mode': 'decoded', 'host_tail_required': False,
        'postprocessing_required': False,
    }]}
    (tmp_path / 'output_contracts.json').write_text(json.dumps(payload))
    result = load_authoritative_output_contract(tmp_path, backend='hailo8',
        model_id='resnet50', variant='full', task='classification')
    assert result['contract_resolution_status'] == 'conflict'
    assert result['contract_resolution_reason'] == 'suite_output_contract_exact_match_invalid'
    assert 'endpoint_contract_not_recorded' in result['contract_resolution_errors']
    assert 'stage' not in result
