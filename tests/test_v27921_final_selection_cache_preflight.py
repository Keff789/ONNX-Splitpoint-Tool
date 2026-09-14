from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from onnx_splitpoint_tool.workflow.artifact_cache_preflight import (
    build_artifact_cache_preflight,
    collect_model_artifact_cache_probes,
    collect_selection_changes,
    render_artifact_cache_preflight_log_lines,
    resolve_artifact_cache_preflight_policy,
    write_artifact_cache_preflight,
)
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _yolo_suite(root):
    base = root / 'models/yolo26m/benchmark_set'
    suite = base / 'legacy_suite'
    _write(suite / 'benchmark_set.json', {'cases': [{'case_id': 'b398', 'boundary': 398}]})
    _write(suite / 'benchmark_plan.json', {'runs': []})
    _write(base / 'hailo_artifact_service_plan.json', {
        'full_baseline_requests': [],
        'case_hef_requests': [{
            'case_id': 'b398', 'stage': 'part1', 'backend': 'hailo10h',
            'status': 'pending_dfc_build', 'build_attempt': {},
        }],
    })
    return suite


def test_yolo26m_b398_exact_miss_is_visible_before_any_compiler_attempt(tmp_path):
    suite = _yolo_suite(tmp_path)
    _write(suite / 'b398/hailo/hailo10/part1/hailo_cache_miss.json', {
        'net_name': 'yolo26m_part1_b398', 'hw_arch': 'hailo10h',
        'cache_key_v3': 'existing-backend-key', 'compiler_dispatch_allowed': False,
        'probe_outcomes': {'exact_v3_hef_present': False, 'legacy_v2_hef_present': False},
    })
    observations, roles = collect_model_artifact_cache_probes(
        run_dir=tmp_path, model_id='yolo26m', targets=['hailo10h'],
        policy=resolve_artifact_cache_preflight_policy({}),
    )
    report = build_artifact_cache_preflight(
        model_ids=['yolo26m'], observations=observations,
        applicable_roles={'yolo26m': list(roles)},
    )
    item = report['cold_build_rows'][0]
    assert item['boundary'] == 'b398'
    assert item['backend'] == 'hailo10h'
    assert item['artifact_stage'] == 'part1'
    assert item['reason'] == 'exact_cache_artifact_missing'
    assert item['identity'] == 'existing-backend-key'
    assert item['evidence']['current_build'] is False
    text = '\n'.join(render_artifact_cache_preflight_log_lines(report))
    assert 'expected_cold_build: model=yolo26m boundary=b398 backend=hailo10h' in text
    paths = write_artifact_cache_preflight(report, output_dir=tmp_path / 'reports')
    with paths['artifact_cache_preflight_items_csv'].open() as handle:
        assert list(csv.DictReader(handle))[0]['boundary'] == 'b398'
    assert '| yolo26m | b398 | hailo10h | part1 | MISS |' in paths['artifact_cache_preflight_md'].read_text()


def test_legacy_unsealed_exact_probe_is_explained_and_lookup_errors_stay_unknown(tmp_path):
    suite = _yolo_suite(tmp_path)
    miss = suite / 'b398/hailo/hailo10/part1/hailo_cache_miss.json'
    payload = {'net_name': 'yolo26m_part1_b398', 'hw_arch': 'hailo10h',
               'probe_outcomes': {'legacy_v2_hef_present': True, 'legacy_v2_receipt_present': False}}
    _write(miss, payload)
    def probe():
        return collect_model_artifact_cache_probes(
            run_dir=tmp_path, model_id='yolo26m', targets=['hailo10h'],
            policy=resolve_artifact_cache_preflight_policy({}),
        )[0][0]
    assert probe().status == 'MISS'
    assert probe().reason == 'legacy_unsealed'
    payload['probe_outcomes']['cache_lookup_error'] = 'permission denied'
    _write(miss, payload)
    assert probe().status == 'UNKNOWN'
    assert probe().reason == 'cache_lookup_error'


def test_selection_changed_requires_valid_model_matched_artifact_bytes(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import hailo_backend
    suite = _yolo_suite(tmp_path)
    cache = tmp_path / 'cache'
    monkeypatch.setattr(hailo_backend, '_hailo_cache_root', lambda: cache)
    validated = []
    def verifier(path):
        validated.append(path)
        receipt = json.loads((path.parent / 'hailo_hef_build_receipt.json').read_text())
        return receipt if receipt.get('hef_sha256') == hashlib.sha256(path.read_bytes()).hexdigest() else None
    monkeypatch.setattr(hailo_backend, '_load_valid_hailo_receipt', verifier)
    for key, model, boundary, valid in [('good', 'yolo26m', 366, True), ('other', 'yolo26s', 100, True), ('corrupt', 'yolo26m', 399, False)]:
        path = cache / key / 'compiled.hef'
        path.parent.mkdir(parents=True)
        path.write_bytes(key.encode())
        _write(path.parent / 'hailo_hef_build_receipt.json', {
            'net_name': f'{model}_part1_b{boundary}',
            'hef_sha256': hashlib.sha256(path.read_bytes()).hexdigest() if valid else 'wrong',
        })
    orphan = cache / 'hef-only' / 'compiled.hef'
    orphan.parent.mkdir()
    orphan.write_bytes(b'not-valid-history')
    report = build_artifact_cache_preflight(model_ids=['yolo26m'], observations=[{
        'model_id': 'yolo26m', 'role': 'hailo10h', 'item_id': 'b398:part1',
        'status': 'MISS', 'reason': 'exact_cache_artifact_missing',
    }], applicable_roles={'yolo26m': ['hailo10h']})
    changes = collect_selection_changes(run_dir=tmp_path, model_ids=['yolo26m'], report=report)
    assert len(changes) == 1
    assert changes[0]['previously_built_boundaries'] == ['b366']
    assert changes[0]['currently_selected'] == 'b398'
    assert changes[0]['expected_cold_builds'] == ['hailo10h_part1']
    report['selection_changes'] = changes
    text = '\n'.join(render_artifact_cache_preflight_log_lines(report))
    assert 'selection_changed:\npreviously_built_boundaries=["b366"]\ncurrently_selected=b398' in text
    assert len(validated) == 2


def test_generation_fence_does_not_mutate_requested_hailo_build_policy():
    runner = object.__new__(EvaluationWorkflowRunner)
    runner._artifact_cache_preflight_pending = True
    runner.profile_payload = {'hailo_build': {'mode': 'reuse_and_build_missing', 'force_build': False}}
    profile = runner._profile_for_benchmark_generation()
    assert profile['hailo_build']['defer_until_cache_preflight'] is True
    assert profile['hailo_build']['mode'] == 'reuse_and_build_missing'
    assert 'defer_until_cache_preflight' not in runner.profile_payload['hailo_build']


def test_detailed_matrix_is_logged_before_first_backend_build(tmp_path, monkeypatch):
    # v2.81 validates a real pending compiler context at the barrier. Supply a
    # valid lexical CPU venv to keep this test about matrix-before-dispatch
    # ordering; no SDK or GPU is involved in this context.
    import sys
    from onnx_splitpoint_tool import hailo_backend
    venv = tmp_path / "fixture_venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    (venv / "bin/python").symlink_to(sys.executable)
    monkeypatch.setattr(hailo_backend, "_resolve_managed_venv_python", lambda **_kw:
                        ("fixture", venv / "bin/python", str(venv / "bin/activate")))
    runner = object.__new__(EvaluationWorkflowRunner)
    runner._stop_requested = False
    runner.jobs = None
    runner.run_dir = tmp_path
    runner.profile_payload = {}
    runner.options = SimpleNamespace(benchmark_execution_backend='local')
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: ['hailo10h']
    runner._materialize_local_cache_preflight_inputs = lambda *_: []
    _yolo_suite(tmp_path)
    _write(tmp_path / 'models/yolo26m/benchmark_set/legacy_suite/b398/hailo/hailo10/part1/hailo_cache_miss.json', {
        'net_name': 'yolo26m_part1_b398', 'hw_arch': 'hailo10h', 'probe_outcomes': {},
    })
    events = []
    runner._emit_log = events.append
    def stage(_model, _name, fn):
        _, details, _, status = fn()
        return SimpleNamespace(status=status, details=details)
    runner._run_stage = stage
    def model(_row, *, stage_names, **_kwargs):
        if 'build_backend_artifacts' in stage_names:
            events.append('COMPILER_DISPATCH')
    runner._run_model = model
    runner._run_model_pipeline_with_cache_preflight([{'id': 'yolo26m'}])
    cold_index = next(i for i, line in enumerate(events) if 'expected_cold_build: model=yolo26m boundary=b398' in line)
    assert cold_index < events.index('COMPILER_DISPATCH')


def _real_stage_runner(tmp_path):
    import threading
    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = tmp_path
    runner.jobs = None
    runner.options = SimpleNamespace(force_stage=[], resume=False, stop_after='', benchmark_execution_backend='local')
    runner._active_model_id = None
    runner._active_stage = ''
    runner._stop_requested = False
    runner._cancel_event = threading.Event()
    runner._progress_done = 0
    runner._progress_total = 1
    runner._stage_input_hash = lambda *_: 'test-input'
    runner._resume_reuse_decision = lambda **_: {'reusable': False}
    runner._missing_full_quality_resume_enabled = lambda: False
    runner._record_stage = lambda *_: None
    runner._register_artifacts = lambda *_, **__: None
    runner._save_manifest = lambda *_: None
    runner.log = lambda *_: None
    runner.progress = lambda *_: None
    return runner


def test_gate_a_generation_exception_stops_entire_campaign_and_records_dispatch_gate(tmp_path):
    runner = _real_stage_runner(tmp_path)
    def generate():
        raise ValueError('cache_preflight_selection_blocked: Gate-A cannot freeze final selection before compiler probes')
    result = runner._run_stage('yolo26m', 'generate_benchmark_set', generate)
    assert result.status == 'failed'
    assert result.error_class == 'cache_preflight_selection_blocked'
    assert result.details['compiler_dispatch_allowed'] is False
    assert result.details['runtime_dispatch_allowed'] is False
    assert runner._stop_requested is True
    persisted = json.loads((tmp_path / 'models/yolo26m/stages/generate_benchmark_set/stage_result.json').read_text())
    assert persisted['details']['stop_workflow'] is True
    dispatch = []
    runner._run_model = lambda *_, **__: dispatch.append('runtime')
    runner._run_model_pipeline_with_cache_preflight([{'id': 'yolo26m'}])
    assert dispatch == []


def test_matrix_disk_write_failure_stops_before_backend_dispatch(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import runner as runner_module
    runner = _real_stage_runner(tmp_path)
    runner.profile_payload = {}
    runner.outputs = {}
    runner.report_paths = []
    runner._targets = lambda: []
    runner._emit_log = lambda *_: None
    runner._materialize_local_cache_preflight_inputs = lambda *_: []
    runner._run_model = lambda _row, *, stage_names, **__: (
        (_ for _ in ()).throw(AssertionError('compiler dispatched after failed matrix write'))
        if 'build_backend_artifacts' in stage_names else None
    )
    monkeypatch.setattr(runner_module, 'write_artifact_cache_preflight',
                        lambda *_, **__: (_ for _ in ()).throw(OSError('no space left on device')))
    runner._run_model_pipeline_with_cache_preflight([{'id': 'yolo26m'}])
    assert runner._stop_requested is True
    stage = json.loads((tmp_path / 'stages/artifact_cache_preflight/stage_result.json').read_text())
    assert stage['status'] == 'failed'
    assert 'no space left on device' in stage['error_detail']


def test_selection_history_validates_embedded_store_receipt_without_materializing(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import hailo_backend as backend
    from onnx_splitpoint_tool.artifact_store import ArtifactStore
    from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract, preprocessing_contract_sha256
    _yolo_suite(tmp_path)
    monkeypatch.setenv('ONNX_SPLITPOINT_ARTIFACT_STORE_ROOT', str(tmp_path / 'store'))
    monkeypatch.setattr(backend, '_hailo_cache_root', lambda: tmp_path / 'empty-cache')
    monkeypatch.setattr(backend, '_hailo_sdk_version_token', lambda: 'hailo-dataflow-compiler:3.31.0')
    source = tmp_path / 'source.onnx'
    source.write_bytes(b'stable-source')
    contract = canonical_image_preprocessing_contract('classification', (224, 224))
    key, payload = backend._hailo_cache_key(
        model_path=source, activation_part1=None, hw_arch='hailo10h', opt_level=1,
        calib_dir=None, calib_count=1, calib_batch_size=1, extra_model_script='',
        start_nodes=None, end_nodes=None, preprocessing_contract=contract,
        effective_calib_count=1, calibration_storage='memory', calibration_memory_cap_bytes=64 * 1024 * 1024,
        net_name='yolo26m_part1_b366', net_input_shapes={'images': [1, 3, 224, 224]},
        disable_rt_metadata_extraction=True,
    )
    hef = tmp_path / 'historical/compiled.hef'
    hef.parent.mkdir()
    hef.write_bytes(b'previously-built-b366')
    receipt = backend._write_hailo_receipt(
        hef_path=hef, source_onnx=source, compiler_onnx=source, hw_arch='hailo10h', net_name='yolo26m_part1_b366',
        preprocessing_contract=contract, preprocessing_sha256=preprocessing_contract_sha256(contract),
        cache_key=key, cache_payload=payload, calibration_identity=payload['calibration_identity'], calibration_count=1,
    )
    store = ArtifactStore(tmp_path / 'store')
    record = store.register(source_path=hef, kind='hailo_hef', contract={'key': key}, metadata={'build_receipt': receipt})
    before = store.list(kind='hailo_hef')[0]
    report = build_artifact_cache_preflight(model_ids=['yolo26m'], observations=[{
        'model_id': 'yolo26m', 'role': 'hailo10h', 'item_id': 'b398:part1', 'status': 'MISS', 'reason': 'not_found',
    }], applicable_roles={'yolo26m': ['hailo10h']})
    changes = collect_selection_changes(run_dir=tmp_path, model_ids=['yolo26m'], report=report)
    assert changes[0]['previously_built_boundaries'] == ['b366']
    assert changes[0]['previous_build_evidence']['b366'] == [record.object_path]
    assert store.list(kind='hailo_hef')[0] == before
    assert not (tmp_path / 'empty-cache').exists()
    Path(record.object_path).write_bytes(b'corrupt-same-cache-record')
    assert collect_selection_changes(run_dir=tmp_path, model_ids=['yolo26m'], report=report) == []


def test_actual_generation_stage_preserves_gate_a_policy_exception_and_stops(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import runner as module
    from onnx_splitpoint_tool.cache_verify_policy import CacheVerifyPolicyError
    runner = _real_stage_runner(tmp_path)
    runner.options.dry_run = False
    runner.profile_payload = {'benchmark_generator': {'mode': 'legacy'}}
    runner.profile_id = 'test'
    runner.run_id = 'test-run'
    runner._artifact_cache_preflight_pending = True
    runner._targets = lambda: ['hailo8']
    runner._execution_mode = lambda: 'generate_and_run'
    runner._cleanup_formal_direct_suite_artifacts = lambda _: None
    monkeypatch.setattr(module, '_authoritative_benchmark_plan_v2792', lambda *_, **__: ({'runs': []}, {}))
    monkeypatch.setattr(module, '_model_row_suite_import_allowed_v27519', lambda _: False)
    calls = []
    def generator(**kwargs):
        assert kwargs['profile_payload']['hailo_build']['defer_until_cache_preflight'] is True
        calls.append('selection_probe')
        raise CacheVerifyPolicyError('cache_preflight_selection_blocked: unexpected selection-probe cold build')
    monkeypatch.setattr(module, 'materialize_legacy_benchmark_set', generator)
    result = runner._run_stage('yolo26m', 'generate_benchmark_set',
        lambda: runner._stage_generate_benchmark_set('yolo26m', {'resolved_path': str(tmp_path / 'source.onnx')}))
    assert calls == ['selection_probe']
    assert result.status == 'failed'
    assert result.details['failure_kind'] == 'cache_preflight_selection_blocked'
    assert result.details['stop_workflow'] is True
    assert result.details['runtime_dispatch_allowed'] is False
    assert runner._stop_requested is True
    diagnostic = json.loads((tmp_path / 'models/yolo26m/benchmark_set/cache_verify_generation_result.json').read_text())
    assert 'cache_preflight_selection_blocked' in diagnostic['error']
    assert result.error_detail == ''
