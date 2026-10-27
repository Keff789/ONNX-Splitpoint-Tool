"""Standalone energy retains normal Generic-only authority and quality outcome."""
import copy
import json

import pytest

from onnx_splitpoint_tool import generic_energy_reporting as reporting
from onnx_splitpoint_tool import generic_only_reporting
from onnx_splitpoint_tool.runners.generic_energy import file_hash
from onnx_splitpoint_tool.workflow import generic_energy_parent as parent


def admitted_case(tmp_path, monkeypatch, *, quality=True):
    case = {'case_key': 'generic-only-fixture', 'admission_mode': 'generic_only'}
    plan = {'performance_result': str(tmp_path / 'actual-performance.json'),
            'performance_result_sha256': 'a' * 64}
    own = {'admission_mode': 'generic_only', 'case_key': case['case_key'],
           'eligible_for_generic_only_execution': True, 'quality_eligible': quality,
           'quality_status': 'reference_close' if quality else 'accuracy_loss',
           'native_performance_claimed': False, 'reasons': []}
    monkeypatch.setattr(generic_only_reporting, 'generic_only_admission', lambda *a, **kw: copy.deepcopy(own))
    report = tmp_path / 'normal_generic_only_admissions.json'
    report.write_text(json.dumps({'rows': [own]}))
    admission = {'admission_mode': 'generic_only', 'valid_repetitions': 3,
                 'performance_result': {'path': plan['performance_result'],
                                        'sha256': plan['performance_result_sha256']},
                 'normal_generic_only_report': {'path': str(report), 'sha256': file_hash(report)},
                 'normal_generic_only_admission': own, 'native_performance_claimed': False}
    return plan, case, admission


@pytest.mark.parametrize('quality', [True, False])
def test_own_quality_outcome_is_preserved_without_native_pair(tmp_path, monkeypatch, quality):
    plan, case, admission = admitted_case(tmp_path, monkeypatch, quality=quality)
    own = reporting._admitted_generic_only_performance(plan, case, {}, admission)
    assert own['quality_eligible'] is quality
    assert own['native_performance_claimed'] is False
    assert 'normal_pair' not in admission


@pytest.mark.parametrize('fault', ['missing', 'different_result', 'changed_report', 'native_claim'])
def test_generic_only_energy_rejects_unbound_admission(tmp_path, monkeypatch, fault):
    plan, case, admission = admitted_case(tmp_path, monkeypatch)
    if fault == 'missing':
        admission = None
    elif fault == 'different_result':
        admission['performance_result']['sha256'] = '0' * 64
    elif fault == 'changed_report':
        admission['normal_generic_only_report']['sha256'] = '0' * 64
    else:
        admission['native_performance_claimed'] = True
    with pytest.raises(ValueError):
        reporting._admitted_generic_only_performance(plan, case, {}, admission)


@pytest.mark.parametrize('fault', [None, 'build', 'identity', 'scope'])
def test_parent_selects_only_unchanged_completed_generic_only_source(tmp_path, monkeypatch, fault):
    scope = tmp_path / 'scope.csv'
    scope.write_text('fixed original scope\n')
    identity = {'setup_id': 'orin_nx_deepx_m1_01', 'case_id': 'b511'}
    config = {'identity': identity, 'build_allowed': False, 'fallback_allowed': False, 'automatic_retry': False}
    row = {'case_id': 'b511', 'case_key': 'fixed-key', 'setup_id': identity['setup_id'],
           'identity': identity, 'runtime_config': copy.deepcopy(config),
           'result_path': str(tmp_path / 'results/b511/generic_completion_results.json')}
    plan = {'scope_mode': 'generic_only_completed_task', 'admission_mode': 'generic_only',
            'build_allowed': False, 'fallback_allowed': False, 'automatic_retry': False,
            'scope_source': str(scope), 'scope_source_sha256': file_hash(scope),
            'output_root': str(tmp_path), 'rows': [row]}
    if fault == 'build':
        row['runtime_config']['build_allowed'] = True
    elif fault == 'identity':
        row['runtime_config']['identity']['case_id'] = 'another-cut'
    elif fault == 'scope':
        scope.write_text('changed scope\n')
    source = tmp_path / 'execution-plan.json'
    source.write_text(json.dumps(plan))
    monkeypatch.setattr(parent, 'R', tmp_path)
    if fault:
        with pytest.raises(ValueError):
            parent.select_case(source, identity['setup_id'], 'b511')
    else:
        actual, selected = parent.select_case(source, identity['setup_id'], 'b511')
        assert actual == plan and selected == row
