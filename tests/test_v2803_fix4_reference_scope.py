from __future__ import annotations

import copy
import json
from pathlib import Path
import runpy
import sys

import pytest

from onnx_splitpoint_tool.workflow.required_run_scope import (
    RequiredRunScopeError, authoritative_run_descriptors,
    build_global_required_run_scope, build_model_required_run_scope,
    merge_authoritative_run_descriptors, seal_required_run_scope,
)
from onnx_splitpoint_tool.workflow.runner import expected_profile_measurements_v60r

ROOT = Path(__file__).resolve().parents[1]
SETUP = 'orin_nx_hailo10_01'


def _rows(variants=None):
    rows = [
        {'id': 'ort_tensorrt', 'type': 'onnxruntime', 'provider': 'tensorrt',
         'stage1': 'tensorrt', 'stage2': 'tensorrt'},
        {'id': 'hailo10', 'type': 'hailo', 'hw_arch': 'hailo10h',
         'stage1': 'hailo10h', 'stage2': 'hailo10h', 'variants': ['full']},
        {'id': 'hailo10_to_tensorrt', 'type': 'matrix',
         'stage1': 'hailo10h', 'stage2': 'tensorrt', 'variants': ['composed']},
    ]
    if variants is not None:
        rows[0]['variants'] = list(variants)
    return rows


def _global_scope(rows):
    run_ids = [row['id'] for row in rows]
    return build_global_required_run_scope(
        profile_id='fixed16', model_entries=[{'id': 'yolo11l', 'task': 'detection'}],
        effective_plan={'effective_generic_run_ids': run_ids,
                        'logical_run_profiles': run_ids,
                        'setup_groups': {'hailo10h_setup': run_ids}},
        hardware_targets=[{'id': SETUP, 'accelerator': 'hailo10h'}],
        run_profiles=rows, created_at='before-build',
    )


@pytest.mark.parametrize('variants, expected_trt', [
    (None, {'full', 'split'}),
    (['full', 'part1', 'part2', 'composed'], {'full', 'split'}),
    (['full'], {'full'}),
    (['composed'], {'split'}),
])
def test_declared_reference_scope_matches_model_measurements_without_losing_h10_split(
    variants, expected_trt,
):
    rows = _rows(variants)
    before = copy.deepcopy(rows)
    scope = _global_scope(rows)
    descriptors = authoritative_run_descriptors(scope, model_id='yolo11l')
    assert {row['variant'] for row in descriptors if row['run_id'] == 'ort_tensorrt'} == expected_trt
    plan = merge_authoritative_run_descriptors({'runs': rows}, descriptors)
    contract = {'cases': [{'case_id': 'b062', 'boundary': 62}]}
    measurements = expected_profile_measurements_v60r(
        model_id='yolo11l', benchmark_plan=plan, benchmark_set_contract=contract,
    )
    assert {row['variant'] for row in measurements if row['run_id'] == 'ort_tensorrt'} == expected_trt
    assert {(row['run_id'], row['variant']) for row in measurements} == (
        {('ort_tensorrt', variant) for variant in expected_trt}
        | {('hailo10', 'full'), ('hailo10_to_tensorrt', 'split')}
    )
    model_scope = build_model_required_run_scope(
        model_id='yolo11l', measurements=measurements, benchmark_plan=plan,
        benchmark_set_contract=contract, created_at='before-runtime',
    )
    assert model_scope['identity_count'] == len(expected_trt) + 2
    assert all(row['expected_setup_id'] == SETUP for row in measurements)
    assert rows == before


def test_scope_cannot_be_shrunk_after_it_was_sealed(tmp_path):
    receipt = tmp_path / 'required_run_scope.json'
    original = seal_required_run_scope(receipt, _global_scope(_rows()))
    with pytest.raises(RequiredRunScopeError, match='immutable_mismatch'):
        seal_required_run_scope(receipt, _global_scope(_rows(['full'])))
    assert json.loads(receipt.read_text()) == original


def test_ambiguous_reference_variant_contract_is_rejected():
    rows = _rows(['full'])
    rows.append(dict(rows[0], variants=['full', 'composed']))
    with pytest.raises(RequiredRunScopeError, match='reference_variants_ambiguous'):
        _global_scope(rows)


def test_real_generated_suite_dispatch_keeps_full_performance_companion_and_h10_split(
    tmp_path, monkeypatch,
):
    """Run real suite main and ownership/pruning; replace only hardware workers."""
    suite_path = tmp_path / 'benchmark_suite.py'
    suite_path.write_text((ROOT / 'onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt').read_text())
    case = {'case_id': 'b062', 'case_dir': 'b062', 'boundary': 62,
            'hailo_case_variant_availability': {'hailo10h': {'part1': True, 'full': True}}}
    (tmp_path / '__BENCH_JSON__').write_text(json.dumps({'model_id': 'yolo11l', 'cases': [case]}))
    (tmp_path / 'benchmark_plan.json').write_text(json.dumps({'model_id': 'yolo11l', 'runs': _rows(['full'])}))
    (tmp_path / 'b062').mkdir()
    suite = runpy.run_path(str(suite_path), run_name='fix4_real_dispatch_test')
    globals_ = suite['main'].__globals__
    performance = []
    companions = []
    bindings = []

    def fake_worker(case_dir, **kwargs):
        performance.append((kwargs['run_id'], tuple(kwargs['variants']), kwargs['native_trt_build']))
        return {'run_id': kwargs['run_id'], 'case_id': case_dir.name,
                'runtime_ok': True, 'requested_variants': kwargs['variants']}

    def fake_companion(**kwargs):
        assert kwargs['run']['_native_full_trt_quality_companion'] is True
        companions.append(kwargs['run']['id'])
        return {'record_count': 16, 'source_run_id': 'native_full_tensorrt',
                'setup_id': SETUP, 'variant': 'full'}

    def fake_binding(**kwargs):
        if kwargs['run_id'] == 'hailo10_to_tensorrt':
            bindings.append((kwargs['run_id'], tuple(kwargs['variants'])))
            return {'status': 'ready', 'binding_path': 'bound-native-split.json'}
        return {}

    globals_['_run_case'] = fake_worker
    globals_['_run_native_full_trt_quality_companion'] = fake_companion
    globals_['_prepare_native_split_quality_for_case'] = fake_binding
    globals_['_provider_unavailable_reason'] = lambda *_args, **_kwargs: None
    globals_['_write_v60_scientific_report'] = lambda *_args, **_kwargs: None
    # Any accidental direct child launch instead of the instrumented workers fails.
    def forbidden_child(*_args, **_kwargs):
        pytest.fail('unexpected compiler/process launch during dispatch smoke')
    monkeypatch.setattr(globals_['subprocess'], 'Popen', forbidden_child)
    monkeypatch.setattr(sys, 'argv', [str(suite_path), '--plan', 'benchmark_plan.json',
        '--run-ids', 'ort_tensorrt,hailo10,hailo10_to_tensorrt',
        '--quality-evidence-eval-id', 'fix4-dispatch',
        '--quality-evidence-setup-id', SETUP, '--quality-evidence-model-id', 'yolo11l',
        '--quality-evidence-endpoint-id', 'tensorrt_at_hailo10h_full',
        '--no-native-trt-build', '--no-plot', '--no-csv'])
    assert suite['main']() == 0
    assert companions == ['ort_tensorrt']
    assert performance == [
        ('ort_tensorrt', ('full',), False),
        ('hailo10', ('full',), False),
        ('hailo10_to_tensorrt', ('composed',), False),
    ]
    assert bindings == [('hailo10_to_tensorrt', ('composed',))]
    status = json.loads((tmp_path / 'benchmark_suite_status.json').read_text())
    assert status['failed_runs'] == []
    assert status['any_rows'] is True
    assert status['quality_evidence_report_count'] == 1
    assert status['performance_claims_emitted'] is True
