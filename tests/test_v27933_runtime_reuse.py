"""AP5: real cache/receipt paths with simulated vendor compilation only.

These tests create synthetic artifacts in pytest temporary directories. They
exercise the productive build boundary, not a mocked cache HIT or preflight.
They do not establish that a user's HEF/DXNN or GPU passed hardware acceptance.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
from onnx import TensorProto, helper

from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.build_evidence_store import BuildEvidenceStore
from onnx_splitpoint_tool.benchmark.services import BenchmarkGenerationOrchestrationService
from onnx_splitpoint_tool.campaign import create_dataset_manifest
from onnx_splitpoint_tool.deepx.compiler import DeepXBuildResult
from onnx_splitpoint_tool.run_modes import apply_run_mode, default_run_modes_config
from onnx_splitpoint_tool.workflow import deepx_build_binding as deepx_binding
from tests.test_v27922_negative_backend import harness, _build


def receipt_bytes(root: Path) -> dict[str, bytes]:
    """Existing artifact bytes and receipts; no new identity or global cache."""
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in root.rglob('*')
        if p.is_file() and (p.suffix in {'.hef', '.dxnn'} or
                           p.name in {'build_manifest.json', 'hailo_hef_build_receipt.json'})
    }


def test_T33_27_hailo_two_normal_requests_reuse_verified_artifact_without_dispatch(harness, monkeypatch):
    h = harness
    h.context['stage'] = 'full'
    h.kwargs.update(net_name='fixture_full', force=False)
    h.behavior['error'] = ''
    # Seed only the vendor result; cache publication and all lookups are real.
    seeded = _build(h, 'seed')
    assert seeded.ok and h.calls.count('compile') == 1
    before = receipt_bytes(h.root/'cache')
    assert before
    compiler_calls = []

    def forbidden(*args, **kwargs):
        compiler_calls.append(True)
        raise AssertionError('compatible artifact reached vendor dispatcher')

    monkeypatch.setattr(hailo_backend, 'hailo_build_hef_via_venv', forbidden)
    monkeypatch.setattr(hailo_backend, 'hailo_build_hef_via_wsl', forbidden)
    results = [hailo_backend.hailo_build_hef_auto(
        h.source, backend='venv', outdir=h.root/f'warm-{i}',
        build_evidence_context=h.context, **h.kwargs,
    ) for i in (1, 2)]
    assert all(r.ok and r.skipped for r in results)
    assert all(Path(r.hef_path).read_bytes() == Path(seeded.hef_path).read_bytes() for r in results)
    assert len({r.calib_info['cache_key'] for r in results}) == 1
    assert compiler_calls == [] and h.calls.count('compile') == 1
    assert receipt_bytes(h.root/'cache') == before


def test_T33_29_negative_full_request_skips_repeat_and_independent_target_builds(harness):
    h = harness
    h.context['stage'] = 'full'
    h.kwargs.update(net_name='fixture_full', force=False)
    first = _build(h, 'negative-seed')
    assert first.details['build_evidence']['recorded_state'] == 'COMPILE_INFEASIBLE'
    before = BuildEvidenceStore().index_path.read_bytes()
    for i in (1, 2):
        result = _build(h, f'negative-repeat-{i}')
        assert result.skipped and not result.ok
        assert result.failure_kind == 'known_negative_build_evidence'
        assert result.details['build_evidence']['state'] == 'COMPILE_INFEASIBLE'
    assert h.calls.count('compile') == 1
    assert BuildEvidenceStore().index_path.read_bytes() == before
    h.behavior['error'] = ''
    independent = _build(h, 'independent-hailo10', hw_arch='hailo10h')
    assert independent.ok and h.calls.count('compile') == 2
    assert independent.details['build_evidence']['key'] != first.details['build_evidence']['key']


def test_T33_27_normal_suite_generation_uses_real_cache_twice(harness, monkeypatch):
    h = harness
    h.behavior['error'] = ''
    graph = helper.make_graph([helper.make_node('Identity', ['images'], ['output'])],
        'SYNTHETIC_WORKFLOW_REUSE',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)]), h.source)
    cfg = SimpleNamespace(
        hef_targets=['hailo8'], hef_full=True, hef_part1=False, hef_part2=False,
        hailo_build_hef_fn=hailo_backend.hailo_build_hef_auto,
        hailo_build_unavailable=None, should_cancel=None,
        hailo_full_end_node_names=[], hailo_full_endpoint_mode='', hailo_full_output_contract=None,
        out_dir=h.root/'suite-seed', full_model_src=str(h.source), full_model_dst=str(h.source),
        base='synthetic_classifier', hailo_full_timeout_explicit=False, hailo_full_timeout_s=0,
        hef_timeout_s=60, hef_backend='local', hef_fixup=False, hef_opt_level=1,
        hef_calib_dir=h.kwargs['calib_dir'], hef_calib_count=1, hef_calib_bs=1,
        hef_force=False, hef_keep=True, hef_wsl_distro=None, hef_wsl_venv='',
        hailo_cache_only=False, hailo_full_cache_only=False, hailo_run_mode='standard',
        execution_cfg=SimpleNamespace(benchmark_task='classification',
                                     build_scheduler_config={}, hailo_run_mode='standard'),
        analysis_payload={}, analysis_params_payload={}, bench_log_path=str(h.root/'benchmark.log'))
    service = BenchmarkGenerationOrchestrationService()

    def request(name):
        cfg.out_dir = h.root/name
        evidence, errors, logs = {}, [], []
        service._build_suite_full_hefs(cfg, log=lambda message, **kwargs: logs.append(str(message)),
            queue_put=lambda event: None, errors=errors, suite_hailo_hefs=evidence,
            publish_hailo_diagnostics=lambda *args: None)
        assert errors == [], (errors, logs)
        assert evidence['hailo8']['full']
        return evidence['hailo8']['full_build']

    request('suite-seed')
    assert h.calls.count('compile') == 1
    before = receipt_bytes(h.root/'cache')
    assert before
    # The full generation service runs again, all actual cache checks remain.
    # Any new SDK instance/parse/compile would append to the instrumented calls.
    calls_before = list(h.calls)
    first = request('suite-warm-1')
    second = request('suite-warm-2')
    assert h.calls == calls_before
    assert first['skipped'] is second['skipped'] is True
    assert receipt_bytes(h.root/'cache') == before


def test_T33_27_28_deepx_mean_std_cold_contract_then_two_verified_runtime_reuses(tmp_path, monkeypatch):
    source = tmp_path/'classification.onnx'
    graph = helper.make_graph([helper.make_node('Identity', ['images'], ['output'])],
        'SYNTHETIC_REUSE_FIXTURE',
        [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 224, 224])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)]), source)
    source_before = source.read_bytes()
    calibration = tmp_path/'calibration'
    (calibration/'class_a').mkdir(parents=True)
    # The vendor is simulated, but the manifest/cardinality verification is real.
    (calibration/'class_a/sample.jpg').write_bytes(b'SYNTHETIC_CALIBRATION_FILE')
    manifest = tmp_path/'calibration.json'
    create_dataset_manifest(task='classification', role='calibration',
        dataset_id='SYNTHETIC_TEST_ONLY', split='train', root=calibration,
        output=manifest, hash_mode='content')
    cache = tmp_path/'cache'
    suite = tmp_path/'suite'
    suite.mkdir()
    monkeypatch.setattr(deepx_binding, 'inspect_deepx_environment', lambda **kwargs: {
        'compiler_ready': True, 'runtime_ready': True, 'cache_dir': str(cache),
        'compiler_imports': [{'module': 'dx_com', 'ok': True,
                             'package_version': 'SYNTHETIC_VENDOR_VERSION'}],
    })
    calls = []

    def simulated_vendor(**kwargs):
        calls.append(str(kwargs['onnx_path']))
        output = Path(kwargs['output_dir'])/'synthetic.dxnn'
        output.write_bytes(b'SYNTHETIC_DXNN_NO_HARDWARE_EXECUTION')
        return DeepXBuildResult(ok=True, status='ok', onnx_path=str(kwargs['onnx_path']),
            config_path=str(kwargs['config_path']), output_dir=str(kwargs['output_dir']),
            dxnn_path=str(output), message='simulated vendor only')

    monkeypatch.setattr(deepx_binding, 'compile_dxnn', simulated_vendor)
    profile = {'deepx_build': {'mode': 'reuse_and_build_missing', 'force_build': False,
        'cache_dir': str(cache), 'calibration_dir': str(calibration), 'calib_count': 1,
        'calibration_method': 'ema', 'opt_level': 0,
        'classification_preprocessing': 'current_scale_only', 'diagnostic_only': True},
        'hailo_build': {'mode': 'reuse_and_build_missing', 'force_build': False, 'cache_integrity': 'relaxed'},
        'campaign': {'dataset_manifests': {'classification': {'calibration': str(manifest)}}}}

    def request(name):
        return deepx_binding.materialize_deepx_build_binding(run_dir=tmp_path/name,
            model_id='synthetic_classification', model_path=str(source),
            row={'task': 'classification', 'input_shape': [1, 3, 224, 224]},
            profile_payload=profile, targets=['deepx_m1'],
            benchmark_set_contract={'suite_dir': str(suite)})

    old = request('legacy-diagnostic-seed')
    assert old['metrics']['deepx_build_status'] == 'ready_built', old
    profile['deepx_build'].update(classification_preprocessing='imagenet_mean_std', diagnostic_only=False)
    new = request('normal-mean-std-cold')
    assert new['metrics']['deepx_build_status'] == 'ready_built', new
    assert new['metrics']['deepx_cache_outcome'] == 'MISS'
    assert len(calls) == 2
    assert calls[0] != calls[1]  # Mean/Std adapter belongs to the changed contract.
    before = receipt_bytes(cache)

    def forbidden(**kwargs):
        raise AssertionError('DX-COM called for compatible Full cache')

    monkeypatch.setattr(deepx_binding, 'compile_dxnn', forbidden)
    warm = [request('normal-warm-1'), request('normal-warm-2')]
    for result in warm:
        assert result['metrics']['deepx_build_status'] == 'ready_reused'
        assert result['metrics']['deepx_cache_outcome'] == 'HIT'
        receipt = json.loads(Path(result['artifacts']['deepx_cache_receipt_json']).read_text())
        assert receipt['cache_contract']['classification_preprocessing'] == 'imagenet_mean_std'
    assert receipt_bytes(cache) == before
    assert source.read_bytes() == source_before
    assert profile['hailo_build']['cache_integrity'] == 'relaxed'
    assert profile['hailo_build']['force_build'] is profile['deepx_build']['force_build'] is False


def test_T33_28_preparation_integrity_stays_relaxed_after_mode_resolution():
    config = default_run_modes_config()
    assert config['modes']['final']['build']['hailo']['cache_integrity'] == 'relaxed'
    profile, _ = apply_run_mode({'execution_preset': {'id': 'final', 'follow_tool_config': True}},
                                config=config)
    assert profile['hailo_build']['cache_integrity'] == 'relaxed'
    # An explicitly chosen strict policy stays strict; no cache-hit migration.
    config['modes']['final']['build']['hailo']['cache_integrity'] = 'strict'
    strict_profile, _ = apply_run_mode({'execution_preset': {'id': 'final', 'follow_tool_config': True}},
                                       config=config)
    assert strict_profile['hailo_build']['cache_integrity'] == 'strict'


def test_T33_27_q3_retention_original_lines_do_not_claim_new_deletions(monkeypatch):
    from onnx_splitpoint_tool.benchmark.remote_run import _remote_trt_retention_limits
    fixture = Path(__file__).parent/'fixtures/v27933_reuse/q3_retention_original_fields.json'
    raw = fixture.read_bytes()
    records = json.loads(raw)['records']
    assert len(records) == 9
    assert all(r['retention']['managed_count'] == 7 for r in records)
    assert all(r['retention']['max_namespaces'] == 6 for r in records)
    assert all(r['retention']['retention_deferred'] for r in records)
    assert all(r['retention']['removed'] == [] for r in records)
    # Older eviction history exists and must remain separate from this run.
    assert sum(bool(r['retention']['prior_eviction']) for r in records) == 3
    monkeypatch.setenv('ONNX_SPLITPOINT_REMOTE_TRT_CACHE_MAX_NAMESPACES', '7')
    monkeypatch.delenv('ONNX_SPLITPOINT_REMOTE_TRT_CACHE_MAX_BYTES', raising=False)
    assert _remote_trt_retention_limits() == (7, 20*1024**3)
    assert fixture.read_bytes() == raw
