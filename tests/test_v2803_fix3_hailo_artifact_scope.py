"""Synthetic reconstruction of the FIX2 fixed16 preflight scope failure.

The uploaded debug pack contains the runtime plans and eight HIT/two UNKNOWN
observations, but omits the source service plan and benchmark_set contract.
These fixtures reconstruct that shape; they are not hardware evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    collect_model_artifact_cache_probes, build_artifact_cache_preflight,
    resolve_artifact_cache_preflight_policy,
)
from onnx_splitpoint_tool.workflow.benchmark_binding import materialize_backend_artifact_decisions
from onnx_splitpoint_tool.workflow.hailo_remote_binding import materialize_hailo_artifact_service_plan


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _forward(**extra):
    return dict(id='hailo10_to_tensorrt', type='matrix',
                stage1={'type': 'hailo', 'hw_arch': 'hailo10'},
                stage2={'type': 'onnxruntime', 'provider': 'tensorrt'},
                variants=['part1', 'part2', 'composed'], **extra)


def _suite(root, runs, *, model='model', cases=('b056',), key='planned_runs'):
    suite = root / 'models' / model / 'benchmark_set/legacy_suite'
    contract = {'cases': [{'case_id': case, 'boundary': int(case[1:])} for case in cases],
                'materialized': True, 'legacy_suite_dir': str(suite)}
    _write(suite / 'benchmark_set.json', contract)
    _write(suite / 'benchmark_plan.json', {key: runs})
    return suite, contract


def _producer(root, contract, *, model='model'):
    result = materialize_backend_artifact_decisions(
        run_dir=root, model_id=model, targets=['hailo10'],
        full_baseline_plan={}, output_contracts={}, benchmark_set_contract=contract)
    return json.loads(result['artifacts']['backend_artifact_decisions_json'].read_text())


def _collect(root, *, model='model', remote=()):
    return collect_model_artifact_cache_probes(
        run_dir=root, model_id=model, targets=['hailo10', 'tensorrt'],
        policy=resolve_artifact_cache_preflight_policy({'artifact_cache_preflight': {'require_warm_cache': True}}),
        remote_trt_observations=remote)


def _stale_service(suite, *, cases=('b056',), current_build=False):
    _write(suite.parent / 'hailo_artifact_service_plan.json', {
        'case_hef_requests': [dict(backend='hailo10', case_id=case, stage=stage,
            status='pending_dfc_build', build_attempt={'ok': True, 'skipped': False} if current_build and stage == 'part2' else {})
            for case in cases for stage in ('part1', 'part2')]})


@pytest.mark.parametrize('key', ['runs', 'planned_runs'])
def test_producer_does_not_request_accelerator_tail_for_forward_pipeline(tmp_path, key):
    suite, contract = _suite(tmp_path, [_forward()], key=key)
    first = _producer(tmp_path, contract)
    assert [(r['case_id'], r['build_part1'], r['build_part2']) for r in first['case_build_requests']] == [('b056', True, False)]
    assert _producer(tmp_path, contract)['case_build_requests'] == first['case_build_requests']
    result = materialize_hailo_artifact_service_plan(
        run_dir=tmp_path, model_id='model', targets=['hailo10'], full_baseline_plan={},
        output_contracts={}, benchmark_set_contract=contract,
        backend_artifact_decisions=first, build_backend_artifacts_request={},
        no_remote=False, execution_mode='generate_and_run')
    service = json.loads(result['artifacts']['hailo_artifact_service_plan_json'].read_text())
    assert [r['stage'] for r in service['case_hef_requests']] == ['part1']
    assert service['missing_case_hef_count'] == 1


def test_collector_filters_stale_forward_part2_but_keeps_missing_required_part1(tmp_path):
    suite, _ = _suite(tmp_path, [_forward()])
    _stale_service(suite)
    rows, _ = _collect(tmp_path)
    assert [(r.item_id, r.status) for r in rows if r.role == 'hailo10_hef'] == [('b056:part1', 'UNKNOWN')]
    (suite.parent / 'hailo_artifact_service_plan.json').unlink()
    rows, roles = _collect(tmp_path)
    assert [(r.item_id, r.reason) for r in rows if r.role == 'hailo10_hef'] == [('b056:part1', 'required_artifact_cache_probe_missing')]
    assert build_artifact_cache_preflight(model_ids=['model'], observations=rows,
        applicable_roles={'model': roles}, block_on_unexpected_cold_builds=True)['runtime_dispatch_allowed'] is False


@pytest.mark.parametrize('run', [
    {'id': 'tensorrt_to_hailo10h', 'type': 'matrix', 'stage1': 'tensorrt', 'stage2': 'hailo10h', 'variants': ['composed']},
    {'id': 'hailo10', 'type': 'hailo', 'hw_arch': 'hailo10', 'stage1': {'hw_arch': 'hailo10'}, 'stage2': {'hw_arch': 'hailo10'}, 'variants': ['part1', 'part2', 'composed'], 'same_backend_split_diagnostics_enabled': True},
])
def test_required_accelerator_part2_remains_blocking_even_without_service_request(tmp_path, run):
    suite, contract = _suite(tmp_path, [run])
    assert _producer(tmp_path, contract)['case_build_requests'][0]['build_part2'] is True
    rows, roles = _collect(tmp_path)
    p2 = [r for r in rows if r.role == 'hailo10_hef' and r.item_id == 'b056:part2']
    assert len(p2) == 1 and p2[0].status == 'UNKNOWN'
    assert build_artifact_cache_preflight(model_ids=['model'], observations=rows,
        applicable_roles={'model': roles}, block_on_unexpected_cold_builds=True)['runtime_dispatch_allowed'] is False


def test_full_only_hailo_ignores_descriptive_stage_fields(tmp_path):
    suite, contract = _suite(tmp_path, [{'id': 'hailo10', 'type': 'hailo', 'hw_arch': 'hailo10',
        'stage1': {'hw_arch': 'hailo10'}, 'stage2': {'hw_arch': 'hailo10'}, 'variants': ['full']}])
    requests = _producer(tmp_path, contract)['case_build_requests']
    assert not any(r['build_part1'] or r['build_part2'] for r in requests)
    _stale_service(suite)
    rows, _ = _collect(tmp_path)
    assert not [r for r in rows if r.role == 'hailo10_hef' and ':part' in r.item_id]
    assert any(r.item_id == 'full' and r.status == 'UNKNOWN' for r in rows)


def test_selected_cases_and_disabled_directions_do_not_expand_requests(tmp_path):
    suite, contract = _suite(tmp_path, [_forward(case_ids=[56]),
        {'id': 'tensorrt_to_hailo10', 'type': 'matrix', 'stage1': 'tensorrt', 'stage2': 'hailo10', 'enabled': False}],
        cases=('b056', 'b062'))
    requests = _producer(tmp_path, contract)['case_build_requests']
    assert {(r['case_id'], s) for r in requests for s in ('part1', 'part2') if r['build_' + s]} == {('b056', 'part1')}
    _stale_service(suite, cases=('b056', 'b062'))
    rows, _ = _collect(tmp_path)
    assert {(r.item_id, r.status) for r in rows if r.role == 'hailo10_hef'} == {('b056:part1', 'UNKNOWN')}


@pytest.mark.parametrize('plan', [{}, {'runs': []}, {'runs': [{'id': 'custom_unresolved_provider'}]},
    {'runs': [{'id': 'trt_to_hailo', 'type': 'matrix', 'variants': ['composed']}]},
    {'runs': [{'type': 'matrix', 'stage1': 'tensorrt', 'stage2': {'type': 'hailo'}, 'variants': ['composed']}]},
    {'runs': [{'id': 'unknown_hailo', 'type': 'hailo', 'variants': ['full']}]}])
def test_legacy_empty_or_unresolved_scope_preserves_existing_requests(tmp_path, plan):
    suite, contract = _suite(tmp_path, [])
    _write(suite / 'benchmark_plan.json', plan)
    assert _producer(tmp_path, contract)['case_build_requests'][0]['build_part2'] is True
    _stale_service(suite)
    rows, _ = _collect(tmp_path)
    assert {r.item_id for r in rows if r.role == 'hailo10_hef'} == {'b056:part1', 'b056:part2'}


def test_missing_case_contract_cannot_hide_existing_hailo_probe(tmp_path):
    suite, _ = _suite(tmp_path, [_forward()])
    _stale_service(suite)
    _write(suite / 'benchmark_set.json', {})
    rows, _ = _collect(tmp_path)
    assert {r.item_id for r in rows if r.role == 'hailo10_hef'} == {'b056:part1', 'b056:part2'}


@pytest.mark.parametrize('case_selected', [True, False])
def test_actual_out_of_scope_build_remains_in_warm_failure_ledger(tmp_path, monkeypatch, case_selected):
    from onnx_splitpoint_tool import hailo_backend
    suite, _ = _suite(tmp_path, [_forward()])
    _stale_service(suite, current_build=True)
    if not case_selected:
        _write(suite / 'benchmark_set.json', {'cases': [{'case_id': 'b062'}]})
    path = suite / 'b056/hailo/hailo10/part2/compiled.hef'
    path.parent.mkdir(parents=True); path.write_bytes(b'synthetic compiled artifact')
    service = json.loads((suite.parent / 'hailo_artifact_service_plan.json').read_text())
    service['case_hef_requests'][1]['hef_path'] = str(path)
    _write(suite.parent / 'hailo_artifact_service_plan.json', service)
    monkeypatch.setattr(hailo_backend, '_load_valid_hailo_receipt', lambda _: {'hw_arch': 'hailo10', 'cache_key': 'synthetic'})
    rows, roles = _collect(tmp_path)
    p2 = next(r for r in rows if r.item_id == 'b056:part2')
    assert p2.evidence['current_build'] is True and p2.status == 'MISS'
    report = build_artifact_cache_preflight(model_ids=['model'], observations=rows,
        applicable_roles={'model': roles}, block_on_unexpected_cold_builds=True)
    assert report['runtime_dispatch_allowed'] is False
    assert report['completed_cold_build_count'] == 1


def test_synthetic_fixed16_matrix_has_eight_hits_and_no_phantom_tail(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import hailo_backend
    monkeypatch.setattr(hailo_backend, '_load_valid_hailo_receipt', lambda _: {'hw_arch': 'hailo10', 'cache_key': 'synthetic-existing'})
    all_rows, all_roles = [], {}
    for model, case in [('mobilenet_v3_large', 'b056'), ('yolo11l', 'b062')]:
        suite, _ = _suite(tmp_path, [{'id': 'hailo10', 'type': 'hailo', 'hw_arch': 'hailo10', 'variants': ['full']},
            {'id': 'ort_tensorrt', 'type': 'onnxruntime', 'provider': 'tensorrt', 'variants': ['full']}, _forward()],
            model=model, cases=(case,), key='runs')
        _stale_service(suite, cases=(case,))
        service = json.loads((suite.parent / 'hailo_artifact_service_plan.json').read_text())
        for stage in ('full', 'part1'):
            hef = suite / ('hailo' if stage == 'full' else case + '/hailo') / 'hailo10' / stage / 'compiled.hef'
            hef.parent.mkdir(parents=True); hef.write_bytes(b'synthetic existing artifact')
            row = {'backend': 'hailo10', 'stage': stage, 'status': 'ready_existing_hef', 'hef_path': str(hef)}
            if stage == 'full':
                service['full_baseline_requests'] = [dict(row, requested=True)]
            else:
                service['case_hef_requests'][0].update(row)
        _write(suite.parent / 'hailo_artifact_service_plan.json', service)
        remote = [{'model_id': model, 'role': role, 'item_id': 'orin/' + item, 'status': 'HIT', 'reason': 'synthetic-compatible-receipt'}
                  for role, item in [('trt_full', 'full'), ('trt_p2', case)]]
        rows, roles = _collect(tmp_path, model=model, remote=remote)
        all_rows.extend(rows); all_roles[model] = roles
    report = build_artifact_cache_preflight(model_ids=list(all_roles), observations=all_rows,
        applicable_roles=all_roles, block_on_unexpected_cold_builds=True)
    assert len(all_rows) == 8 and all(row.status == 'HIT' for row in all_rows)
    assert report['unknown_count'] == 0 and report['confirmed_miss_count'] == 0
    assert report['runtime_dispatch_allowed'] is True


def test_recorded_user_scope_projection_retains_all_eight_recorded_hits(tmp_path, monkeypatch):
    """Project recorded verifier decisions only; do not claim fresh validation."""
    from onnx_splitpoint_tool.workflow import artifact_cache_preflight as module
    fixture = json.loads((Path(__file__).parent / 'fixtures/v2803_fix3_recorded_preflight_scope.json').read_text())
    assert fixture['evidence_kind'] == 'recorded_preflight_scope_projection_not_new_artifact_validation'
    recorded = {model['model_id']: model for model in fixture['models']}
    def recorded_hailo_observations(*, model_id, **_):
        rows = [module.CacheProbeObservation.from_mapping(value)
                for value in recorded[model_id]['observations'] if value['role'] == 'hailo10_hef']
        return rows, {'hailo10_hef'}
    monkeypatch.setattr(module, '_hailo_observations', recorded_hailo_observations)
    rows, roles = [], {}
    for model, payload in recorded.items():
        suite, _ = _suite(tmp_path, payload['plan']['runs'], model=model, cases=payload['cases'], key='runs')
        _write(suite / 'benchmark_plan.json', payload['plan'])
        remote = [value for value in payload['observations'] if value['role'].startswith('trt_')]
        observed, applicable = _collect(tmp_path, model=model, remote=remote)
        rows.extend(observed); roles[model] = applicable
    original_hits = {(p['model_id'], row['role'], row['item_id'], row['identity'])
                     for p in fixture['models'] for row in p['observations'] if row['status'] == 'HIT'}
    assert {(r.model_id, r.role, r.item_id, r.identity) for r in rows if r.status == 'HIT'} == original_hits
    # FIX4 retains all recorded hits but exposes four previously unprobed
    # generic TRT split engines. The archived evidence stays byte-identical.
    assert len(rows) == 12
    assert sum(row.status == 'HIT' for row in rows) == 8
    assert sum(row.status == 'UNKNOWN' for row in rows) == 4
    report = build_artifact_cache_preflight(model_ids=list(roles), observations=rows,
        applicable_roles=roles, block_on_unexpected_cold_builds=True)
    assert report['runtime_dispatch_allowed'] is False
    assert report['unknown_count'] == 4 and report['confirmed_miss_count'] == 0
