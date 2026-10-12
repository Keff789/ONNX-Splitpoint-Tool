"""Actual closed cohorts through the existing reader; no hardware or Current writes."""
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.cross_runner_reporting import (
    _bound_native_cohort_rates, compute_cross_runner_report,
)

ARCHIVE = os.environ.get('ONNX_SPLITPOINT_REV3_ARCHIVE', '')
R = Path(ARCHIVE).expanduser()
pytestmark = pytest.mark.skipif(not ARCHIVE, reason='Optional original-data replay requires ONNX_SPLITPOINT_REV3_ARCHIVE')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def actual12():
    helper = load_module('existing_claim_tests', R / 'operations/finalisation_generic_20261008/test_normal_generic_claim_projection.py')
    exporter = helper.exporter
    current = exporter.read(R / 'current_results/generic_current.json')['rows']
    fixtures = []
    for row in current:
        if row['model_id'] not in ('yolo26m', 'yolo26s') or row['comparison_backend'] not in ('hailo8', 'hailo10h'):
            continue
        plan = exporter.read(row['execution_plan'])
        case = next(c for c in plan['rows'] if c['case_key'] == row['case_key'])
        result = exporter.read(case['result_path'])
        observed = exporter.validate_completion_case(result, case, result_root=Path(case['result_path']).parent)
        assert observed['status'] == 'completed'
        native, quality = exporter.native_projection(case, exporter.read(row['native_cohort_proof']))
        generic, native, validation = helper.projected_case((case, result, observed, native, quality, row['generic_quality_transfer']))
        validation = exporter.native_validation_projection(quality, generic, native, row['generic_quality_transfer_status'])
        # Existing relative report binding becomes absolute for an isolated test root.
        native['normal_quality_provenance']['normal_report'] = str(R / native['normal_quality_provenance']['normal_report'])
        fixtures.append((generic, native, validation))
    assert len(fixtures) == 12
    return fixtures


@pytest.mark.parametrize('index', range(12))
def test_actual12_normal_consumer(actual12, tmp_path, index):
    generic, native, validation = actual12[index]
    reports = tmp_path / 'reports'
    (reports / 'native_validation').mkdir(parents=True)
    (reports / 'native_producer_combined_summary.json').write_text(json.dumps({'rows': [native]}))
    (reports / 'native_validation/native_producer_validation_summary.json').write_text(json.dumps({'rows': [validation]}))
    (reports / 'native_expected_matrix.json').write_text(json.dumps({'expected_rows': [{**generic, 'execution_mode': 'native_split'}]}))
    report = compute_cross_runner_report(tmp_path, [generic])
    assert report['pair_count'] == 1, report['identity_exclusions']
    pair = report['pairs'][0]
    assert pair['eligible_for_technical_transfer'] is True, pair['technical_transfer_eligibility_checks']
    assert pair['eligible_for_quality_transfer'] is True
    assert pair['eligible_for_claim_transfer'] is True, pair['transfer_exclusion_reasons']
    assert pair['native_measured_throughput_fps'] == native['fps_median']
    assert pair['generic_measured_throughput_fps'] == generic['completed_task_fps']


@pytest.mark.parametrize('damage', ['missing_rate', 'nan', 'zero', 'negative', 'regime', 'setup', 'endpoint', 'original_binding', 'report_binding'])
def test_bound_cohort_rate_negatives(actual12, tmp_path, damage):
    native = copy.deepcopy(actual12[0][1])
    if damage in ('missing_rate', 'nan', 'zero', 'negative'):
        native['fps_median'] = {'missing_rate': None, 'nan': float('nan'), 'zero': 0, 'negative': -1}[damage]
    elif damage == 'regime':
        native['projection_role'] = 'generic'
    elif damage == 'setup':
        native['setup_id'] = 'orin_nx_deepx_m1_01'
    elif damage == 'endpoint':
        native['physical_source_output_endpoint_id'] = 'unmeasured'
    elif damage == 'original_binding':
        native['source_descriptors'][0]['normalized_report_sha256'] = '0' * 64
    else:
        native['normal_quality_provenance']['normal_report_sha256'] = '0' * 64
    assert _bound_native_cohort_rates(native, tmp_path) == {}


@pytest.mark.parametrize('damage', ['attestation', 'original_binding', 'comparison', 'operating_point'])
def test_validation_completion_binding_negatives(actual12, damage):
    from onnx_splitpoint_tool.workflow.cross_runner_reporting import _completed_endpoint_attested
    validation = copy.deepcopy(actual12[0][2])
    if damage == 'attestation':
        validation['output_endpoint_attestation']['status'] = 'failed'
    elif damage == 'original_binding':
        validation['source_descriptors'][0]['normalized_report_sha256'] = '0' * 64
    elif damage == 'comparison':
        validation['completed_task_comparison_endpoint_contract_hash'] = '0' * 64
    else:
        validation['repetition_operating_point_evidence'][0]['operating_point_sha256'] = '0' * 64
    assert _completed_endpoint_attested(validation, validation['endpoint_contract_hash']) is False


def test_plan_progress_keeps_regimes_separate(tmp_path, monkeypatch):
    import csv
    updater = load_module('existing_matrix_updater', R / 'current_results/build_fairness_matrix.py')
    monkeypatch.setattr(updater, 'R', tmp_path)
    def read_with_generic_stop(path):
        path = Path(path)
        source = R / path.relative_to(tmp_path) if path.is_relative_to(tmp_path) else path
        value = json.loads(source.read_text())
        if path.name == 'REMEASUREMENT_PLAN_SUMMARY.json':
            value['plan_file'] = str(tmp_path / 'REMEASUREMENT_PLAN.csv')
        if Path(path).name == 'energy_task_budget.json':
            key = json.dumps(['generic_completed:deepx', 'yolov7_paper', 'b044', 'orin_nx_deepx_m1_01'], separators=(',', ':'))
            source_id = value['tasks'][key]['source_id']
            # This is a reporting fixture, independent of any live campaign
            # permission. A later session may intentionally supersede that
            # permission without invalidating the archived scientific data.
            value = {'campaign': True, 'max_transport_failures': 20,
                     'sources': {source_id: {'stop_reason': '', 'transport_failures': 0}},
                     'tasks': {key: {'source_id': source_id, 'stop_reason': 'generic_fixture_stop',
                                     'chains': [], 'limits': {'max_chains': 9, 'max_retries': 2,
                                                             'max_transport_failures': 20}}}}
        return value
    monkeypatch.setattr(updater, 'read', read_with_generic_stop)
    plan = list(csv.DictReader((R / 'REMEASUREMENT_PLAN.csv').open()))
    selected = [r for r in plan if r['model'] == 'yolov7_paper' and r['setup'] == 'orin_nx_deepx_m1_01' and r['case'] in ('b044', 'b066')]
    for name in ('REMEASUREMENT_PLAN.csv', 'NATIVE_REMEASUREMENT_PLAN.csv', 'GENERIC_REMEASUREMENT_PLAN.csv'):
        rows = selected if name == 'REMEASUREMENT_PLAN.csv' else [r for r in selected if r['regime'] == ('native' if name.startswith('NATIVE') else 'generic_completed')]
        with (tmp_path / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(selected[0]));w.writeheader();w.writerows(rows)
    native = json.loads((R / 'current_results/native_current.json').read_text())
    generic = json.loads((R / 'current_results/generic_current.json').read_text())
    for g in generic['rows']:
        if g['model_id'] == 'yolov7_paper' and g['case_id'] in ('b044', 'b066') and g['comparison_backend'] == 'deepx':
            g['normal_claim_eligible'] = False
            # The original archive may have gained valid Energy since this
            # fixture was written. Keep the intended missing-Generic scenario
            # explicit in this copied fixture, without changing original data.
            g['normal_energy_eligible'] = False
            g['energy_provenance']['normal_report_verified'] = False
    updater.update_plan_progress(native, generic)
    result = list(csv.DictReader((tmp_path / 'REMEASUREMENT_PLAN.csv').open()))
    for row in result:
        if row['regime'] == 'native':
            assert 'normal_energy_verified' in row['execution_status']
            assert 'normal_claim_verified' in row['execution_status']
        else:
            assert 'normal_energy_verified' not in row['execution_status']
            assert 'normal_claim_not_eligible' in row['execution_status']
            if row['case'] == 'b044':
                assert row['energy_completion_category'] == 'blocked_stop'
                assert 'generic_fixture_stop' in row['energy_completion_reason']
        original = next(r for r in selected if r['case_identity'] == row['case_identity'])
        assert all(row[k] == original[k] for k in original if k not in ('execution_status', 'final_source_binding') and not k.startswith('energy_completion_'))
