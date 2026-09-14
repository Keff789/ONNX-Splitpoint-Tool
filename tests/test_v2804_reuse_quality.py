"""AP4 recipe reuse and mathematical-quality/provenance separation.

Real normal cache publication and fresh processes use a synthetic DFC boundary;
none of these fixtures is reported as a hardware build or quality acceptance.
"""
from __future__ import annotations
import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import pytest
from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool.quality_service import (
    ManagementQualityService, QualityEvaluationRequest, prepare_evaluation,
    quality_request_from_manifest, QualityArtifactIntegrityError,
)
ROOT = Path(__file__).resolve().parents[1]


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / filename)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


_old = _load('v2804_hailo_publication_fixture', 'test_v280_artifact_reuse.py')
@pytest.fixture
def publication_case(tmp_path, monkeypatch):
    case = _old.publication_case.__wrapped__(tmp_path, monkeypatch)
    # Test-owned fresh DFC fixture: explicitly disable ORT telemetry in this
    # process too. The parent pytest configuration cannot propagate API state.
    sdk = next((tmp_path / 'selected_dfc/lib').glob('python*/site-packages/hailo_sdk_client.py'))
    sdk.write_text('import onnxruntime; onnxruntime.disable_telemetry_events()\n' + sdk.read_text())
    return case


def _controller(case, out, *, compute='cpu', guard=False):
    kwargs = {k: str(v) if isinstance(v, Path) else v for k, v in case.items()}
    kwargs.update(outdir=str(out), backend='venv', compute_device=compute)
    request, response = out.with_name(out.name + '_request.json'), out.with_name(out.name + '_result.json')
    request.parent.mkdir(parents=True, exist_ok=True)
    request.write_text(json.dumps(kwargs))
    program = """
import onnxruntime; onnxruntime.disable_telemetry_events()
import dataclasses, importlib.abc, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from onnx_splitpoint_tool import hailo_backend as backend
from onnx_splitpoint_tool import hailo_compiler_context as context
attempts = []
if sys.argv[4] == 'guard':
    class BlockSDK(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'hailo_sdk_client', 'tensorflow'}:
                attempts.append(fullname)
                raise AssertionError('SDK import on cache reuse: ' + fullname)
    sys.meta_path.insert(0, BlockSDK())
    def forbidden(*a, **kw):
        raise AssertionError('Compiler or GPU probe on cache reuse')
    backend._run_streamed_subprocess = forbidden
    context.resolve_hailo_compiler_context = forbidden
result = backend.hailo_build_hef_auto(**json.loads(Path(sys.argv[2]).read_text()))
payload = dataclasses.asdict(result)
payload.update(controller_pid=os.getpid(), sdk_import_attempts=attempts)
Path(sys.argv[3]).write_text(json.dumps(payload))
"""
    process = subprocess.run([sys.executable, '-I', '-B', '-c', program, str(ROOT), str(request), str(response),
                              'guard' if guard else 'build'], capture_output=True, text=True, timeout=40)
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(response.read_text())


def test_t0402_h8_two_fresh_gpu_hits_ignore_missing_saved_manifest(publication_case, tmp_path):
    case = dict(publication_case, hw_arch='hailo8')
    first = _controller(case, tmp_path / 'cpu')
    assert first['ok'], first.get('error')
    receipt = backend._load_valid_hailo_receipt(Path(first['hef_path']))
    assert receipt['hw_arch'] == 'hailo8'
    before_cache = _old._payload_files(tmp_path / 'production_cache')
    before_store = _old._payload_files(tmp_path / 'production_store')
    gpu = dict(case, compute_by_family={'hailo8': {'device': 'gpu', 'dependency_manifest': str(tmp_path / 'absent_manifest.json')}})
    results = [_controller(gpu, tmp_path / f'reuse_{i}', compute='gpu', guard=True) for i in (1, 2)]
    assert len({r['controller_pid'] for r in results}) == 2
    for result in results:
        assert result['ok'], result.get('error')
        assert result['calib_info']['compiler_dispatch_count'] == 0
        assert result['calib_info']['build_receipt'] == receipt
        assert result['sdk_import_attempts'] == []
        assert backend._load_valid_hailo_receipt(Path(result['hef_path'])) == receipt
    assert _old._payload_files(tmp_path / 'production_cache') == before_cache
    assert _old._payload_files(tmp_path / 'production_store') == before_store


def test_t0403_private_different_hef_cannot_replace_productive_cached_hef(publication_case, tmp_path):
    case = dict(publication_case, hw_arch='hailo8')
    normal = _controller(case, tmp_path / 'normal')
    assert normal['ok'], normal.get('error')
    before = _old._payload_files(tmp_path / 'production_cache')
    sdk = next((tmp_path / 'selected_dfc/lib').glob('python*/site-packages/hailo_sdk_client.py'))
    sdk.write_text(sdk.read_text().replace('synthetic SDK HEF payload', 'private different diagnostic HEF'))
    private = _controller(dict(case, force=True, publish_artifacts=False), tmp_path / 'private')
    assert private['ok'], private.get('error')
    assert Path(private['hef_path']).read_bytes() != Path(normal['hef_path']).read_bytes()
    assert backend._load_valid_hailo_receipt(Path(private['hef_path'])) is None
    actual = _controller(case, tmp_path / 'reused', compute='gpu', guard=True)
    assert actual['ok'], actual.get('error')
    assert Path(actual['hef_path']).read_bytes() == Path(normal['hef_path']).read_bytes()
    assert _old._payload_files(tmp_path / 'production_cache') == before


def _request():
    records = [{'image_id': 'i1', 'value': 1.0}, {'image_id': 'i2', 'value': 0.0}]
    return QualityEvaluationRequest(reference_records=records, candidate_records=copy.deepcopy(records),
        annotations=[{'image_id': 'i1', 'label': 1}, {'image_id': 'i2', 'label': 0}],
        metric_gate_config={'primary_metric': 'paired_mean', 'guardrails': {}}, repetitions=8,
        seed=20260710, confidence_level=0.95, non_inferiority_margin=0.01,
        request_id='cpu-hef', reference_identity='original-reference',
        candidate_execution_completion_contract_sha256='1' * 64)


def test_t0404_same_recipe_different_predictions_do_not_hit_quality_cache(tmp_path):
    first = _request()
    second = replace(first, request_id='different-hef', candidate_records=[{'image_id': 'i1', 'value': 0.0}, {'image_id': 'i2', 'value': 0.0}],
                     candidate_execution_completion_contract_sha256='2' * 64)
    assert prepare_evaluation(first)[0] != prepare_evaluation(second)[0]
    with ManagementQualityService(tmp_path / 'cache', workers=1) as service:
        a, b = service.evaluate(first, timeout=15), service.evaluate(second, timeout=15)
    assert a['cache_hit'] is False and b['cache_hit'] is False
    assert b['request_id'] == 'different-hef'
    assert b['candidate_execution_completion_contract_sha256'] == '2' * 64


def test_t0405_identical_predictions_reuse_math_with_current_validated_request_metadata(tmp_path):
    first = _request()
    second = replace(first, request_id='new-verified-artifact-request', reference_identity='new-reference-binding',
                     candidate_execution_completion_contract_sha256='2' * 64)
    assert prepare_evaluation(first)[0] == prepare_evaluation(second)[0]
    with ManagementQualityService(tmp_path / 'cache', workers=1) as service:
        a, b = service.evaluate(first, timeout=15), service.evaluate(second, timeout=15)
    assert a['cache_hit'] is False and b['cache_hit'] is True
    assert b['request_id'] == second.request_id
    assert b['reference_identity'] == second.reference_identity
    assert b['candidate_execution_completion_contract_sha256'] == '2' * 64


def test_t0405_conflicting_producer_artifact_rejected_before_mathematical_cache(tmp_path):
    helpers = _load('v2804_bound_producer_fixture', 'test_v269d_trt_central_quality_producer.py')
    request, _ = helpers._write_request_bundle(tmp_path, helpers._producer())
    payload = json.loads(request.read_text())
    payload['runtime_artifact_sha256'] = 'd' * 64
    request.write_text(json.dumps(payload))
    with pytest.raises(QualityArtifactIntegrityError, match='runtime_artifact_sha256 differs'):
        quality_request_from_manifest(request)


@pytest.mark.parametrize('field', ['subset', 'annotations', 'seed', 'bootstrap'])
def test_t0406_statistical_inputs_keep_existing_invalidation_contract(field):
    original = _request()
    if field == 'subset':
        modified = replace(original, reference_records=original.reference_records[:1], candidate_records=original.candidate_records[:1], annotations=original.annotations[:1])
    elif field == 'annotations':
        modified = replace(original, annotations=[{'image_id': 'i1', 'label': 0}, {'image_id': 'i2', 'label': 0}])
    elif field == 'seed':
        modified = replace(original, seed=original.seed + 1)
    else:
        modified = replace(original, repetitions=original.repetitions + 1)
    assert prepare_evaluation(original)[0] != prepare_evaluation(modified)[0]


def test_t0402_real_target_reuse_worker_uses_saved_profile_and_exact_deferred_receipt(publication_case, tmp_path):
    import os
    import yaml
    source = tmp_path / 'source_run'
    part = source / 'models/mobilenet_v3_large/benchmark_set/legacy_suite/b056'
    part.mkdir(parents=True)
    model = part / 'mobilenet_v3_large_part1_b56.onnx'
    model.write_bytes(Path(publication_case['onnx_path']).read_bytes())
    output = part / 'hailo/hailo8/part1'
    calibration = Path(publication_case['calib_dir'])
    sample = (calibration / 'sample.npy').read_bytes()
    for number in range(1, 500):
        (calibration / f'sample_{number:03d}.npy').write_bytes(sample)
    sdk = next((tmp_path / 'selected_dfc/lib').glob('python*/site-packages/hailo_sdk_client.py'))
    sdk.write_text(sdk.read_text().replace('== (1,8,8,3)', '== (500,8,8,3)'))
    case = dict(publication_case, onnx_path=model, hw_arch='hailo8', net_name='mobilenet_v3_large_part1_b56',
                calib_count=500, calib_batch_size=8, opt_level=1, force=False,
                build_evidence_context={'model_id': 'mobilenet_v3_large', 'boundary': 56, 'stage': 'part1'})
    publication = _controller(case, output)
    assert publication['ok'], publication.get('error')
    kwargs = {k: str(v) if isinstance(v, Path) else v for k, v in case.items() if k != 'onnx_path'}
    kwargs['outdir'] = str(output)
    request = {'schema': 'onnx-splitpoint/deferred-hailo-build/v1', 'source_onnx': str(model),
               'source_onnx_sha256': 'sha256:' + hashlib.sha256(model.read_bytes()).hexdigest(),
               'kwargs': kwargs, 'environment': {k: v for k, v in os.environ.items() if k.startswith('ONNX_SPLITPOINT_HAILO_')}}
    (output / 'deferred_hailo_build.json').write_text(json.dumps(request))
    saved = yaml.safe_load((ROOT / 'tests/fixtures/v2803_night_regression/original/profile.yaml').read_text())
    saved['execution_preset']['follow_tool_config'] = False
    saved['hailo_build']['compute_by_family']['hailo8'] = {'device': 'gpu', 'dependency_manifest': str(tmp_path / 'absent_overlay.json')}
    profile = tmp_path / 'saved_profile.yaml'; profile.write_text(yaml.safe_dump(saved))
    results = []
    for number in (1, 2):
        work = tmp_path / f'target_lookup_{number}'
        program = "import onnxruntime; onnxruntime.disable_telemetry_events(); import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
        command = [sys.executable, '-I', '-B', '-c', program, str(ROOT / 'scripts/hailo_reuse_gate_v2804.py'),
                   '--source-run', str(source), '--profile', str(profile), '--worker', str(work)]
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert (work / 'result.json').is_file(), process.stdout + process.stderr
        row = json.loads((work / 'result.json').read_text())
        assert process.returncode == 0, row
        assert row['status'] == 'cache_hit_pass'
        assert row['exact_receipt_match'] is True and row['compiler_dispatch_count'] == 0
        assert row['compute_selection'] == saved['hailo_build']['compute_by_family']['hailo8']
        assert row['observations'] == {'sdk_import_attempts': [], 'child_process_attempts': []}
        results.append(row)
    assert results[0]['controller_pid'] != results[1]['controller_pid']
    assert results[0]['hef_sha256'] == results[1]['hef_sha256']
