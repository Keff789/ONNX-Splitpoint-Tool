"""A5/A7/AP3: request-local readiness with original workspace observations."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import hailo_backend
from onnx_splitpoint_tool.filesystem_admission import FilesystemWriteInspection
from onnx_splitpoint_tool.workflow.artifact_cache_preflight import build_artifact_cache_preflight
from onnx_splitpoint_tool.workflow.hailo_workspace_preflight import preflight_cold_hailo_workspaces
from onnx_splitpoint_tool.workflow.selection_request_counts import requested_cases_for_model

FIXTURES = Path(__file__).parent / 'fixtures/v282_workspace_originals'


def _cold(target, *, model='yolo11l', case='b064', shape=(1, 3, 640, 640), count=500):
    return {'model_id': model, 'role': 'hailo8_hef', 'item_id': case + ':part1',
            'boundary': case, 'backend': 'hailo8', 'artifact_stage': 'part1',
            'status': 'MISS', 'reason': 'not_found', 'compiler_dispatch_allowed': True,
            'artifact_path': str(Path(target) / 'compiled.hef'), 'evidence': {
                'workspace_contract': {'out_dir': str(target), 'effective_calibration_count': count,
                    'calibration_identity_shapes': [list(shape)], 'calibration_storage': 'memmap',
                    'source': 'prepared_hailo_build_contract'}}}


@pytest.mark.parametrize('name', ['yolo11l_b064', 'yolo26m_b040', 'yolo26s_b023', 'regnet_x_1_6gf_b073'])
def test_original_four_workspace_reports_replayed_through_early_preflight(name, monkeypatch):
    original = json.loads((FIXTURES / f'workspace_{name}.json').read_text())
    w = dict(original['workspace']); w['requested_path'] = Path(w['requested_path']); w['probe_path'] = Path(w['probe_path'])
    monkeypatch.setattr(hailo_backend, 'inspect_write_target', lambda path: FilesystemWriteInspection(**w))
    for key in ('ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_PREFLIGHT', 'ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_MULTIPLIER',
                'ONNX_SPLITPOINT_HAILO_DFC_WORKSPACE_RESERVE_BYTES', 'ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_BYTES',
                'ONNX_SPLITPOINT_HAILO_DFC_MIN_FREE_INODES'):
        monkeypatch.delenv(key, raising=False)
    model, case = name.rsplit('_', 1)
    c = original['calculation']
    row = _cold(w['requested_path'], model=model, case=case, shape=c['input_shapes'][0], count=c['calibration_count'])
    report = preflight_cold_hailo_workspaces({'cold_build_rows': [row]}, {})
    job = report['jobs'][0]
    assert job['calculation'] == c
    assert job['status'] == original['status']
    assert job['workspace']['free_bytes'] == w['free_bytes']
    assert job['compiler_dispatch_count'] == 0
    assert report['reservation'] is False and report['repeat_at_dispatch'] is True
    if model.startswith('yolo'):
        assert c['required_free_bytes'] == 61_129_883_648
        assert job['reason'] == 'local_dfc_workspace_insufficient'
    else:
        assert report['status'] == 'passed'


@pytest.mark.parametrize('change', [
    {'status': 'HIT'}, {'status': 'KNOWN_INFEASIBLE'}, {'status': 'UNKNOWN'},
    {'compiler_dispatch_allowed': False}, {'runtime_artifact_available': True},
    {'evidence': {'compiler_cache_only': True}},
])
def test_non_cold_jobs_never_inspect_workspace(tmp_path, monkeypatch, change):
    def forbidden(*args, **kwargs):
        raise AssertionError('Warm/excluded/unknown/prohibited jobs must not probe DFC capacity')
    monkeypatch.setattr(hailo_backend, 'hailo_dfc_workspace_preflight', forbidden)
    row = dict(_cold(tmp_path), **change)
    assert preflight_cold_hailo_workspaces({'cold_build_rows': [row]}, {})['status'] == 'not_required'


@pytest.mark.parametrize('field,value', [('out_dir', ''), ('out_dir', '/different/workspace'),
    ('calibration_identity_shapes', []), ('calibration_identity_shapes', [[1, 3, -1, 640]]),
    ('effective_calibration_count', None)])
def test_missing_or_conflicting_contract_is_unknown_never_small_default(tmp_path, field, value):
    row = _cold(tmp_path)
    row['evidence']['workspace_contract'][field] = value
    report = preflight_cold_hailo_workspaces({'cold_build_rows': [row]}, {})
    assert report['status'] == 'unknown'
    job = report['jobs'][0]
    assert job['reason'] == 'local_dfc_workspace_unresolved'
    assert (job.get('calculation') or {}).get('required_free_bytes') is None


def test_effective_count_and_actual_output_target_win_over_profile_and_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv('TMPDIR', str(tmp_path / 'irrelevant_other_volume'))
    seen = []
    real = hailo_backend.inspect_write_target
    monkeypatch.setattr(hailo_backend, 'inspect_write_target', lambda path: (seen.append(path), real(path))[1])
    target = tmp_path / 'actual_runroot/model/b064/hailo8/part1'
    report = preflight_cold_hailo_workspaces({'cold_build_rows': [_cold(target, count=73)]},
                                            {'hailo_build': {'calib_count': 500}})
    assert seen == [target]
    job = report['jobs'][0]
    assert job['calculation']['calibration_count'] == 73
    assert job['workspace']['probe_path'] == str(tmp_path)
    assert not target.exists()  # read-only ancestor inspection


def test_missing_producer_dependency_is_deferred_only_for_exact_allowed_miss():
    dep = {'model_id': 'model', 'role': 'trt_p2', 'item_id': 'setup/b024', 'status': 'UNKNOWN',
           'reason': 'native_part1_identity_unavailable', 'expectation': 'warm',
           'evidence': {'dependency_backend': 'hailo8', 'dependency_boundary': 'b024', 'dependency_stage': 'part1'}}
    producer = {'model_id': 'model', 'role': 'hailo8_hef', 'item_id': 'b024:part1',
                'status': 'MISS', 'reason': 'not_found', 'expectation': 'cold'}
    def make(p):
        return build_artifact_cache_preflight(model_ids=['model'], observations=[dep, p],
            applicable_roles={'model': ['trt_p2', 'hailo8']}, block_on_unexpected_cold_builds=True)
    report = make(producer)
    assert report['runtime_dispatch_allowed'] is True
    assert report['unknown_count'] == 1 and report['strict_unknown_blocker_count'] == 0
    assert report['deferred_dependency_rows'][0]['status'] == 'UNKNOWN'
    assert report['deferred_dependency_rows'][0]['compiler_dispatch_allowed'] is False
    assert report['deferred_dependency_rows'][0]['evidence']['dependent_runtime_dispatch_allowed'] is False
    for changed in (dict(producer, status='UNKNOWN'), dict(producer, expectation='warm'),
                    dict(producer, item_id='b025:part1'), dict(producer, evidence={'compiler_cache_only': True})):
        assert make(changed)['runtime_dispatch_allowed'] is False


def test_selection_requested_cardinality_follows_each_explicit_model():
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.profile_payload = {'selection_policy': {'max_accepted_cases_per_model': 3,
        'forced_cases': {'regnet': ['b073'], 'yolo11l': ['b062', 'b064']}}}
    assert runner._selection_numbers('regnet')[0] == 1
    assert runner._selection_numbers('yolo11l')[0] == 2
    assert runner._selection_numbers('other')[0] == 3
    with pytest.raises(ValueError, match='duplicate'):
        requested_cases_for_model({'forced_cases': {'regnet': [73, 'b073']}}, 'regnet', default=3)


def test_lookup_without_compiler_is_not_failed_attempt_and_stage_placeholder_is_weak(tmp_path):
    from onnx_splitpoint_tool.workflow.hailo_remote_binding import _hailo_attempt_status, _immutable_attempt_row
    from onnx_splitpoint_tool.workflow.hailo_remote_binding import _attempt_stage_from_payload
    path = tmp_path / 'b024/hailo/hailo8/part1/hailo_attempt_receipts/attempt.json'
    path.parent.mkdir(parents=True); path.write_text('{}')
    payload = {'endpoint': 'full_or_unspecified', 'last_active_stage': 'build_evidence_lookup',
               'failure_kind': 'known_negative_build_evidence', 'details': {'build_evidence': {
                   'status': 'HIT', 'negative_evidence_hit': True, 'reusable': True, 'state': 'COMPILE_INFEASIBLE'}}}
    assert _hailo_attempt_status(payload) == 'known_infeasible'
    assert _attempt_stage_from_payload(payload, path) == ('split', 'part1')
    row = _immutable_attempt_row(path, payload, tmp_path)
    assert row['compiler_dispatch_count'] == 0 and row['stage'] == 'part1'
    conflict = _immutable_attempt_row(path, dict(payload, endpoint='decoded_full'), tmp_path)
    assert conflict['stage'] == 'unknown' and conflict['identity_conflicts'] == ['full', 'part1']
    assert _hailo_attempt_status({'last_stage': 'build_evidence_lookup'}) == 'not_started_evidence_unconfirmed'
    assert _hailo_attempt_status({'last_stage': 'local_dfc_workspace_preflight'}) == 'not_started_workspace_blocked'


def test_real_cache_probe_captures_effective_contract_without_workspace_probe(tmp_path, monkeypatch):
    from tests.test_v27934_hailo_backend import managed
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import cache_preflight_builder, REQUEST_NAME
    case = managed.__wrapped__(tmp_path, monkeypatch)
    from PIL import Image
    (case['calib_dir'] / 'sample.npy').unlink()
    Image.new('RGB', (8, 8)).save(case['calib_dir'] / 'sample.png')
    case['calib_count'] = 500  # the actual prepared dataset holds only one sample
    source = case.pop('onnx_path')
    output = tmp_path / 'output'
    def forbidden(*args, **kwargs):
        raise AssertionError('Cache-only lookup must not check resource admission')
    monkeypatch.setattr(hailo_backend, 'hailo_dfc_workspace_preflight', forbidden)
    result = cache_preflight_builder(hailo_backend._hailo_build_hef_legacy)(
        source, **case, outdir=output, cache_only=False, force=False,
        sdk_version_token='hailo_sdk_client:BOUNDARY_TEST')
    assert not result.ok
    request = json.loads((output / REQUEST_NAME).read_text())
    assert request['cache_probe_status'] == 'MISS'
    contract = request['cache_probe_details']['workspace_contract']
    assert contract['effective_calibration_count'] == 1
    assert contract['calibration_identity_shapes'] == [[1, 3, 8, 8]]
    assert contract['out_dir'] == str(output.resolve())
    assert request['kwargs']['calib_count'] == 500  # original dispatch policy preserved
    diagnostic = json.loads((output / 'hailo_cache_miss.json').read_text())
    assert diagnostic['workspace_contract'] == contract
    assert 'workspace_contract' not in diagnostic['cache_payload_v3']


def test_missing_native_part1_does_not_poison_verified_full_cache_hit(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.benchmark import remote_run
    from tests.test_v27920_remote_trt_cache_preflight import (
        _suite, _LocalReadOnlyTransport, _builder_abi, _builder_abi_sha256, _receipt, _owner, _sha)
    suite = _suite(tmp_path / 'input', [
        {'id': 'ort_tensorrt', 'type': 'onnxruntime', 'provider': 'tensorrt', 'variants': ['full']},
        {'id': 'hailo8_to_trt', 'type': 'matrix', 'case_id': 24, 'variants': ['composed'],
         'stage1': {'hw_arch': 'hailo8'}, 'stage2': {'provider': 'tensorrt'}}])
    monkeypatch.setattr(remote_run, '_trt_preflight_native_policy', lambda *a, **k:
                        {'precision': 'float32_layout_fp16', 'policy_sha256': '1' * 64})
    builder = tmp_path / 'trtexec'; builder.write_bytes(b'not-executed')
    abi = _builder_abi(builder); base = tmp_path / 'remote'; base.mkdir()
    key = remote_run._stable_trt_engine_cache_key(suite, builder_abi=abi)
    namespace = base / '_onnx_splitpoint_cache/tensorrt_managed_v27516' / key
    source = suite / 'models/model.onnx'
    leaf = namespace / remote_run._trt_persistent_engine_relative_dir(
        role='full', source_onnx_sha256=_sha(source), precision='fp16')
    _receipt(leaf=leaf, source_bytes=source.read_bytes(), builder=builder, role='full')
    _owner(namespace, builder_abi_sha256=_builder_abi_sha256(abi))
    transport = _LocalReadOnlyTransport(base)
    result = remote_run.probe_remote_trt_artifact_cache(transport=transport, suite_dir=suite,
        setup_id='setup', setup_accelerator='hailo8', builder_abi=abi, resolved_remote_base=str(base))
    by_role = {row['role']: row for row in result['observations']}
    assert by_role['trt_full']['status'] == 'HIT'
    assert by_role['trt_p2']['status'] == 'UNKNOWN'
    assert by_role['trt_p2']['reason'] == 'native_part1_identity_unavailable'
    assert by_role['trt_p2']['evidence']['dependency_backend'] == 'hailo8'
    assert len(transport.commands) == 1
    assert result['status'] == 'partial'


def test_normal_runner_writes_job_local_workspace_report_before_dispatch(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import hailo_compiler_preflight
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    from tests.test_v27921_final_selection_cache_preflight import _yolo_suite, _write
    suite = _yolo_suite(tmp_path)
    output = suite / 'b398/hailo/hailo10/part1'
    contract = _cold(output)['evidence']['workspace_contract']
    _write(output / 'hailo_cache_miss.json', {
        'net_name': 'yolo26m_part1_b398', 'hw_arch': 'hailo10h', 'probe_outcomes': {},
        'workspace_contract': contract})
    monkeypatch.setattr(hailo_compiler_preflight, 'preflight_cold_hailo_contexts',
        lambda *a, **k: {'status': 'pass', 'compiler_dispatch_allowed': True, 'contexts': []})
    original = json.loads((FIXTURES / 'workspace_yolo26m_b040.json').read_text())['workspace']
    def inspect(path):
        return FilesystemWriteInspection(**dict(original, requested_path=path, probe_path=tmp_path))
    monkeypatch.setattr(hailo_backend, 'inspect_write_target', inspect)
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path; runner.profile_payload = {}; runner.outputs = {}; runner.report_paths = []
    runner.options = SimpleNamespace(benchmark_execution_backend='local')
    runner._targets = lambda: ['hailo10h']
    runner._materialize_local_cache_preflight_inputs = lambda *_: []
    logs = []; runner._emit_log = logs.append
    artifacts, metrics, message, status = runner._stage_artifact_cache_preflight([{'id': 'yolo26m'}])
    assert status == 'ok' and metrics['runtime_dispatch_allowed'] is True
    assert metrics['hailo_workspace_blocked_jobs'] == 1
    report = json.loads(artifacts['hailo_workspace_preflight_json'].read_text())
    assert report['jobs'][0]['workspace']['requested_path'] == str(output)
    assert report['jobs'][0]['artifact_stage'] == 'part1'
    assert 'Compilerkontext bereit, Arbeitsraum unzureichend' in message
    assert 'local_dfc_workspace_insufficient' in artifacts['artifact_cache_preflight_md'].read_text()
    assert any('[hailo-workspace-preflight]' in line for line in logs)


def test_workspace_rechecked_at_actual_dispatch_and_job_stays_infrastructure(tmp_path, monkeypatch):
    from dataclasses import replace
    from tests.test_v27934_hailo_backend import managed
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import (
        cache_preflight_builder, REQUEST_NAME, _completed_job_observation, project_deferred_build_readiness)
    from onnx_splitpoint_tool.build_evidence import classify_build_outcome, TRANSIENT_INFRASTRUCTURE
    case = managed.__wrapped__(tmp_path, monkeypatch)
    source = case.pop('onnx_path'); output = tmp_path / 'build'; output.mkdir()
    builder = hailo_backend._hailo_build_hef_legacy
    result = cache_preflight_builder(builder)(source, **case, outdir=output, force=False,
        sdk_version_token='hailo_sdk_client:BOUNDARY_TEST')
    request = json.loads((output / REQUEST_NAME).read_text())
    contract = request['cache_probe_details']['workspace_contract']
    real_fs = hailo_backend.inspect_write_target(output)
    monkeypatch.setattr(hailo_backend, 'inspect_write_target', lambda path:
        replace(real_fs, requested_path=path, free_bytes=2**40, free_inodes=2**20))
    row = _cold(output); row['evidence']['workspace_contract'] = contract
    assert preflight_cold_hailo_workspaces({'cold_build_rows': [row]}, {})['status'] == 'passed'
    monkeypatch.setattr(hailo_backend, 'inspect_write_target', lambda path:
        replace(real_fs, requested_path=path, free_bytes=0, free_inodes=2**20))
    failed = builder(source, **case, outdir=output, force=False, sdk_version_token='hailo_sdk_client:BOUNDARY_TEST')
    assert failed.failure_kind == 'local_dfc_workspace_insufficient'
    assert failed.details['compiler_dispatch_count'] == 0
    assert not (output / 'compiled.hef').exists()
    observation = _completed_job_observation(request, failed, model_id='model')
    assert observation['primary_failure_reason'] == 'local_dfc_workspace_insufficient'
    assert classify_build_outcome({'ok': False, 'failure_kind': failed.failure_kind, 'details': failed.details}) == TRANSIENT_INFRASTRUCTURE


def test_known_exact_producer_exclusion_needs_no_dependent_engine_but_unconfirmed_does():
    producer = {'model_id': 'model', 'role': 'hailo8_hef', 'item_id': 'b024:part1',
                'status': 'KNOWN_INFEASIBLE', 'reason': 'COMPILE_INFEASIBLE', 'expectation': 'warm',
                'identity': 'existing-backend-key', 'evidence': {'build_evidence': {
                    'status': 'HIT', 'negative_evidence_hit': True, 'reusable': True,
                    'state': 'COMPILE_INFEASIBLE'}}}
    dependent = {'model_id': 'model', 'role': 'trt_p2', 'item_id': 'setup/b024', 'status': 'UNKNOWN',
                 'reason': 'native_part1_identity_unavailable', 'evidence': {
                     'dependency_backend': 'hailo8', 'dependency_boundary': 'b024', 'dependency_stage': 'part1'}}
    def project():
        return build_artifact_cache_preflight(model_ids=['model'], observations=[producer, dependent],
            applicable_roles={'model': ['hailo8', 'trt_p2']}, block_on_unexpected_cold_builds=True)
    report = project()
    assert report['runtime_dispatch_allowed'] is True
    assert report['cold_builds_required'] == 0 and report['known_infeasible_count'] == 1
    assert report['deferred_dependency_rows'][0]['evidence']['dependency_resolution'] == 'upstream_known_infeasible'
    producer['evidence']['build_evidence']['reusable'] = False
    assert project()['runtime_dispatch_allowed'] is False


def test_delivered_warm_gate_reuses_real_saved_request_in_fresh_process(tmp_path, monkeypatch):
    import hashlib
    import os
    import subprocess
    import sys
    from tests.test_v280_artifact_reuse import publication_case, _run_controller
    from onnx_splitpoint_tool.workflow.deferred_hailo_builds import cache_preflight_builder
    case = publication_case.__wrapped__(tmp_path, monkeypatch)
    case.update(hw_arch='hailo8', net_name='yolo11l_part1_b064')
    run = tmp_path / 'closed_run'
    output = run / 'models/yolo11l/benchmark_set/legacy_suite/b064/hailo/hailo8/part1'
    output.parent.mkdir(parents=True)
    built = _run_controller(case, output)
    assert built['ok'], built.get('error')
    assert built['details']['compiler_dispatch_count'] == 1
    args = {key: str(value) if isinstance(value, Path) else value for key, value in case.items()}
    source = args.pop('onnx_path')
    cache_preflight_builder(hailo_backend.hailo_build_hef_auto)(source, **args,
        outdir=str(output), backend='venv', force=False, compute_device='cpu',
        build_evidence_context={'model_id': 'yolo11l', 'boundary': 64, 'stage': 'part1'})
    def files(root):
        return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob('*') if path.is_file()}
    before = files(run)
    report_path = tmp_path / 'acceptance/warm_recheck.json'
    script = Path(__file__).resolve().parents[1] / 'scripts/reference_workflow_gate_v282.py'
    child = subprocess.run([sys.executable, '-I', '-B', str(script), '--recheck-run', str(run),
        '--recheck-scope', 'h8_yolo11l_b064', '--recheck-output', str(report_path)],
        capture_output=True, text=True, timeout=45)
    assert child.returncode == 0, child.stdout + child.stderr + (report_path.read_text() if report_path.exists() else '')
    report = json.loads(report_path.read_text())
    assert report['status'] == 'pass' and report['pid'] != os.getpid()
    assert report['jobs'][0]['status'] == 'HIT' and report['jobs'][0]['compiler_dispatch_count'] == 0
    assert before == files(run)
    # Corrupt a current saved-source binding; no repaired HIT may be fabricated.
    from scripts.reference_workflow_gate_v282 import recheck_warm_requests
    request_path = output / 'deferred_hailo_build.json'
    request = json.loads(request_path.read_text()); request['source_onnx_sha256'] = '0' * 64
    request_path.write_text(json.dumps(request))
    failed = recheck_warm_requests(run, 'h8_yolo11l_b064', tmp_path / 'rejected/recheck.json')
    assert failed['status'] == 'incomplete'
    assert failed['jobs'] == []
    assert 'saved_request_identity_or_source_changed' in failed['errors'][0]
