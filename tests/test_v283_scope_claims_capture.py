"""Exact original exclusions, separate claim axes, and acquisition budget."""
import copy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.evidence_state_model import summarize_run_scope
from onnx_splitpoint_tool.workflow.scientific_reporting import (
    _scientific_row, _performance_claim_eligible, _performance_cohort_projection,
    _write_reports,
)
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
from onnx_splitpoint_tool.energy.collector import _command_capture_budget
from onnx_splitpoint_tool.energy.config import EnergyDefaults

FIXTURE = Path(__file__).parent / 'fixtures/v283_scope_claims.json'


def _scope(tmp_path, mutate=None):
    fixture = json.loads(FIXTURE.read_text())
    for model, data in fixture['models'].items():
        if mutate:
            mutate(data)
        root = tmp_path / 'models' / model
        for relative, value in (
            ('benchmark_set/required_run_scope.json', data['scope']),
            ('benchmark_results/normalized_results.json', {'results': data['rows']}),
            ('benchmark_results/required_profile_matrix.json', data['required_matrix']),
            ('stages/build_backend_artifacts/stage_result.json', data['stage']),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
    return summarize_run_scope(tmp_path)


def test_original_composed_exclusions_leave_part2_and_planned_count_intact(tmp_path):
    result = _scope(tmp_path)
    assert result['matrix_required'] == 2
    assert result['quality_not_applicable'] == 2 and result['quality_missing'] == 0
    assert result['quality_completed'] == 0
    assert result['build_runtime_counts'] == {'build_failed': 2}
    assert all(row['build_exclusion']['source'] == 'existing_exact_build_evidence' for row in result['states'])
    assert all(row['setup_id'] == 'orin_nx_hailo8_01' for row in result['states'])


@pytest.mark.parametrize('mutation', ['model', 'case', 'setup', 'recipe', 'role'])
def test_exclusion_never_transfers_to_foreign_scope_or_recipe(tmp_path, mutation):
    def change(data):
        entry = data['scope']['identities'][0]
        if mutation == 'recipe':
            jobs = data['stage']['details']['deferred_build_readiness']['blocked_jobs']
            for job in jobs:
                job['build_evidence']['key']['recipe']['net_name'] = 'changed'
        else:
            field = {'model': 'model_id', 'case': 'case_id', 'setup': 'expected_setup_id', 'role': 'variant'}[mutation]
            entry[field] = 'foreign'
    result = _scope(tmp_path, change)
    assert result['quality_not_applicable'] == 0


def test_actual_completed_composed_result_is_not_replaced_by_exclusion(tmp_path):
    def change(data):
        row = data['rows'][0]
        row.update(primary_variant='composed', measured_variants=['composed'],
                   quality_source_variant='composed',
                   runtime_ok=True, runtime_executable=True, quality_applicability='applicable',
                   central_quality_decision='fail', variant_status={'composed': 'completed'})
    result = _scope(tmp_path, change)
    assert result['quality_completed'] == 2
    assert result['quality_not_applicable'] == 0
    assert result['decision_counts'] == {'fail': 2}


@pytest.mark.parametrize('explicit,expected', [('absent', True), (None, True), (False, False), (True, True), ('invalid', False)])
def test_real_scientific_projection_resolves_null_without_overriding_false(explicit, expected):
    row = {'model_id': 'resnet50', 'task': 'classification', 'backend': 'tensorrt', 'performance_eligible': True}
    if explicit != 'absent':
        row['performance_claim_eligible'] = explicit
    projected = _scientific_row(row)
    assert _performance_claim_eligible(projected) is expected


def test_conflicting_eligibility_assertions_remain_blocked_after_projection():
    row = {'model_id': 'resnet50', 'task': 'classification', 'backend': 'tensorrt',
           'performance_claim_eligible': True, 'performance_eligible': False}
    assert not _performance_claim_eligible(row)
    assert not _performance_claim_eligible(_scientific_row(row))


@pytest.mark.parametrize('veto', [
    {'claim_eligible': False}, {'diagnostic_only': True},
    {'performance_claim_exclusion_reasons': ['screening_only']},
    {'identity_conflicts': ['setup']}, {'runtime_executable': False},
])
def test_null_fallback_honors_existing_vetoes(veto):
    assert not _performance_claim_eligible({'performance_eligible': True, 'performance_claim_eligible': None, **veto})
    assert not _performance_claim_eligible(_scientific_row({
        'model_id': 'resnet50', 'task': 'classification', 'backend': 'tensorrt',
        'performance_eligible': True, 'performance_claim_eligible': None, **veto,
    }))


def test_original_54_generic_rows_only_recover_six_legacy_claims():
    rows = json.loads(FIXTURE.read_text())['performance_observations']
    before = copy.deepcopy(rows)
    assert len(rows) == 54
    assert sum(_performance_claim_eligible(row) for row in rows) == 6
    assert rows == before


def test_actual_export_keeps_native_screening_and_generic_claim_cohorts_separate(tmp_path):
    fixture = json.loads(FIXTURE.read_text())
    generic = fixture['performance_observations']
    native = fixture['native_performance_observations']
    assert len(native) == 63
    assert all(row['claim_eligible'] is False for row in native)
    payload = {'rows': generic, 'summary': {}, 'native_performance_matrix': {'observations': native}}
    _write_reports(tmp_path, payload)
    exported = json.loads((tmp_path / 'performance_results.json').read_text())
    assert len(exported) == 6
    assert all(_performance_claim_eligible(row) for row in exported)
    native_export = json.loads((tmp_path / 'native_performance_observations.json').read_text())
    assert len(native_export) == 63
    assert all(row['claim_eligible'] is False for row in native_export)


def test_five_original_technical_cases_are_counted_once():
    from scripts.native_producer_validate_visualize import _technical_quality_error
    rows = json.loads(FIXTURE.read_text())['original_technical_rows']
    assert len(rows) == len({(r['backend'], r['model'], r['case']) for r in rows}) == 5
    assert sum(_technical_quality_error(row) for row in rows) == 5


@pytest.mark.parametrize('model', ['yolo26m', 'yolo26s'])
def test_actual_h10_reader_keeps_structural_pass_and_numeric_fail(model):
    path = Path(__file__).parent / f'fixtures/v282_h10_generic/models/{model}/benchmark_results/benchmark_results_hailo10_to_tensorrt_auto.json'
    row = json.loads(path.read_text())[0]
    result = normalize_benchmark_row(row, model_id=model, source_path=path)
    assert result['interface_contract_pass'] is True
    assert result['strict_boundary_numeric_pass'] is False
    assert result['interface_stage1_mapping_pass'] is True
    assert result['interface_stage1_shape_pass'] is True
    assert result.get('semantic_e2e_pass') is not True


def test_capture_covers_original_late_command_end_without_changing_inner_duration():
    budget = _command_capture_budget(1., EnergyDefaults())
    assert budget['collector_duration_s'] == 17
    assert budget['configured_inner_workload_duration_s'] == 1.
    assert budget['collector_duration_s'] + 5 + 5 > 15
    retry = _command_capture_budget(1., EnergyDefaults(), 20.2)
    assert retry['collector_duration_s'] == 22
    assert retry['capture_budget_reason'] == 'observed_full_command_exceeds_startup_budget'


@pytest.mark.parametrize('observed', [float('nan'), float('inf'), -1, 122])
def test_capture_budget_is_bounded(observed):
    with pytest.raises(ValueError, match='out_of_bounds'):
        _command_capture_budget(1., EnergyDefaults(), observed)


@pytest.mark.parametrize('selected_count,reason', [
    (0, 'energy_repeat_selection_no_valid_attempt'),
    (2, 'energy_repeat_selection_ambiguous'),
])
def test_no_valid_repeat_is_distinct_from_ambiguous_selection(tmp_path, selected_count, reason):
    from scripts.run_native_producer_energy_from_summary import _selected_energy_repeat_run_dir
    from test_v282_energy_selected_attempt import _case
    root, aggregate, *_ = _case(tmp_path)
    row = aggregate['runs'][0]
    for attempt in row['repeat_attempt_history']:
        attempt['selected'] = selected_count == 2
    with pytest.raises(ValueError, match=reason):
        _selected_energy_repeat_run_dir(row, measurement_root=root, logical_index=0)
