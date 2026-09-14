"""Cross-component v2.80.1 gates.

Real: management scheduling, refreshed Suite, ONNX Runtime CPU execution,
reference publication/consumer, paired quality worker, merge/report/export.
Synthetic: tiny input models/images and accelerator candidate records only.
No physical accelerator execution or performance/energy value is claimed.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pytest
import yaml

from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.artifacts import write_json
from onnx_splitpoint_tool.workflow.debug_pack import create_evaluation_debug_pack
from onnx_splitpoint_tool.workflow.scientific_reporting import build_scientific_reports
from onnx_splitpoint_tool.native_job_identity import native_job_prerequisite
from onnx_splitpoint_tool.cache_verify_policy import cache_verify_guard

ROOT = Path(__file__).resolve().parents[1]


def _fixture_module():
    name = 'v2801_integration_real_reference_fixture'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests/test_v2801_cpu_reference_dispatch.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _runner(root):
    run = root / 'evaluation_v2801_integration'
    run.mkdir()
    log = []
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile='', out=str(root)), log=log.append)
    runner.run_dir = run
    runner.run_id = run.name
    runner.profile_id = 'v2801_integration_synthetic_accelerator'
    runner.run_log_path = run / 'evaluation_workflow.log'
    runner.parent_log_path = root / 'latest.log'
    runner.profile_payload = {
        'id': runner.profile_id,
        'execution_preset': {'id': 'standard'},
        'quality_gate': {'statistics': {'execution_location': 'central_management', 'workers': 1}},
        'hailo_build': {'force_build': False, 'cache_enabled': True},
        'deepx_build': {'force_build': False},
        'models': [],
    }
    assert not cache_verify_guard(runner.profile_payload)
    return runner, log


def _schedule(runner, root, task, *, broken=False, model_id=None):
    suite = _fixture_module().make_real_reference_suite(root, task)
    model = model_id or ('tiny_' + task)
    if model_id:
        for name in ('benchmark_plan.json', 'benchmark_set.json', 'b001/split_manifest.json'):
            payload = json.loads((suite / name).read_text())
            payload['model_id'] = model_id
            (suite / name).write_text(json.dumps(payload))
    if broken:
        # Inject a real exception at the generated suite dispatch boundary.
        # Model/GT contracts remain identical; the source contract binds this
        # modified test harness. The real launcher/process/cleanup are used,
        # and no reference or successful runner result is manufactured.
        harness = suite / 'benchmark_suite.py'
        lines = harness.read_text().splitlines(keepends=True)
        function = next(node for node in ast.parse(''.join(lines)).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_run_case')
        lines.insert(function.body[0].lineno - 1,
                     '    raise RuntimeError("integration_injected_reference_failure")\n')
        harness.write_text(''.join(lines))
    model_dir = runner.run_dir / 'models' / model
    (model_dir / 'benchmark_set').mkdir(parents=True)
    for name in ('benchmark_plan.json', 'benchmark_set.json'):
        shutil.copy2(suite / name, model_dir / 'benchmark_set' / name)
    write_json(model_dir / 'model_manifest.json', {'model_id': model, 'task': task, 'resolved_path': str(suite / 'b001/model.onnx')})
    runner.profile_payload['models'].append({'id': model, 'task': task})
    runner.manifest.setdefault('models', {})[model] = {'model_id': model, 'task': task}
    runner._schedule_management_cpu_reference(model, suite)
    future = runner._management_reference_futures[model]
    runner._schedule_management_cpu_reference(model, suite)
    assert runner._management_reference_futures[model] is future
    status = future.result(timeout=90)
    return model, suite, status


def _candidate_request(runner, model, task, reference_payload, *, bad=False):
    """Synthetic accelerator input, with real reference provenance unchanged."""
    setup = 'synthetic_hailo8_setup' if task == 'classification' else 'synthetic_deepx_setup'
    source_run = 'hailo8_to_trt' if task == 'classification' else 'deepx_to_trt'
    request_dir = runner.run_dir / 'models' / model / 'benchmark_results' / 'quality_inputs' / setup / 'b001' / ('results_' + source_run) / 'task_quality_inputs'
    request_dir.mkdir(parents=True)
    records = []
    for original in reference_payload['records']:
        row = copy.deepcopy(original)
        value = row.pop('reference')
        if bad:
            value = {'top1_hit': False, 'top5_hit': False} if task == 'classification' else []
        row['candidate'] = value
        records.append(row)
    contract = copy.deepcopy(reference_payload['quality_contract'])
    contract_sha = reference_payload['quality_contract_sha256']
    candidate = {
        'schema': 'onnx-splitpoint/task-quality-candidate-input', 'schema_version': 1,
        'task': task, 'variant': 'full', 'pairing_key': 'image_id',
        'provenance_required': True, 'quality_contract': contract,
        'quality_contract_sha256': contract_sha, 'records': records,
        'test_evidence_scope': 'synthetic_accelerator_candidate_only',
    }
    candidate_path = request_dir / 'full_candidate.json'
    write_json(candidate_path, candidate)
    stage = 'classification_logits' if task == 'classification' else 'decoded_nms_detections'
    endpoint = {'stage': stage, 'contract_family': stage, 'endpoint_contract_complete': True,
                'test_evidence_scope': 'synthetic_accelerator_endpoint'}
    endpoint['endpoint_contract_hash'] = _digest(endpoint)
    gate = ({'primary_metric': 'top1_accuracy', 'guardrails': {'top5_accuracy_margin': .01}}
            if task == 'classification' else {'primary_metric': 'coco_ap_50_95', 'guardrails': {'ap50_margin': .01, 'ap75_margin': .01}})
    gate.update(task=task, non_inferiority_margin=.01)
    policy = {'task': task, 'metric_gate_config': gate}
    manifest = {
        'schema': 'onnx-splitpoint/central-quality-evaluation-request', 'schema_version': 1,
        'status': 'pending_central_evaluation', 'task': task, 'variant': 'full',
        'model_id': model, 'case_id': 'b001', 'eval_run_id': runner.run_id,
        'setup_id': setup, 'source_run_id': source_run, 'backend': source_run,
        'pairing_key': 'image_id', 'execution_location': 'central_management',
        'reference': {'source': 'management_cpu_reference', 'required': True, 'quality_contract_sha256': contract_sha},
        'candidate': {'path': candidate_path.name, 'size_bytes': candidate_path.stat().st_size,
                      'sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest()},
        'record_count': len(records), 'reference_record_count': len(records),
        'provenance_required': True, 'quality_contract': contract, 'quality_contract_sha256': contract_sha,
        'endpoint_contract': endpoint, 'endpoint_contract_hash': endpoint['endpoint_contract_hash'],
        'runtime_precision_identity': 'fp32', 'policy_sha256': _digest(policy),
        'metric_gate_config': gate,
        'statistics': {'method': 'paired_bootstrap', 'bootstrap_repetitions': 20,
                       'seed': 20260710, 'confidence_level': .95, 'decision': 'lower_one_sided_bound'},
        'test_evidence_scope': 'synthetic_accelerator_candidate_only',
    }
    path = request_dir / 'full_request.json'
    write_json(path, manifest)
    identity = runner._quality_request_identity(path, model_id=model)
    assert identity['identity_valid'], identity
    request_row = copy.deepcopy(manifest)
    request_row['request'] = {'path': str(path), 'sha256': identity['source_request_sha256']}
    pending = {'schema': 'onnx-splitpoint/task-quality-gate', 'schema_version': 2,
               'task': task, 'variant': 'full', 'status': 'pending_central_evaluation',
               'decision': 'pending_central_evaluation', 'quality_input_request': request_row}
    row = {
        'model_id': model, 'task': task, 'case_id': 'b001', 'run_id': source_run,
        'backend': source_run, 'setup_id': setup, 'source_tag': source_run,
        'quality_source_run_id': source_run, 'quality_source_setup_ids': [setup],
        'variant': 'full', 'primary_variant': 'full', 'quality_source_variant': 'full',
        'task_quality_gates_by_variant': {'full': pending}, 'task_quality_gate': pending,
        'quality_evaluation_pending': True, 'technical_status': 'completed',
        'endpoint_contract_hash': identity['endpoint_contract_hash'], 'runtime_precision_identity': 'fp32',
        'diagnostic_only': True, 'claim_eligible': False, 'counts_as_benchmark': False,
        'test_evidence_scope': 'synthetic_accelerator_candidate_only',
    }
    write_json(runner.run_dir / 'models' / model / 'benchmark_results/normalized_results.json', {'results': [row], 'result_count': 1})
    return path, row


def _reference_bytes(status):
    return json.loads(Path(status['reference_path']).read_text())


def _assert_debug_pack(runner, root, models):
    before = {p: p.read_bytes() for p in runner.run_dir.rglob('*') if p.is_file()}
    exported = create_evaluation_debug_pack(runner.run_dir, root / 'integration_debug.zip')
    with zipfile.ZipFile(root / 'integration_debug.zip') as archive:
        names = archive.namelist()
        for model in models:
            for filename in ('management_cpu_reference_status.json', 'management_cpu_reference_stdout.txt'):
                relative = f'quality_management/references/{model}/{filename}'
                match = [name for name in names if name.endswith(relative)]
                assert len(match) == 1, names
                assert archive.read(match[0]) == before[runner.run_dir / relative]
        # v2.80.4 includes existing declared decoded records with exact bytes.
        prediction_names = [name for name in names if name.endswith(('canonical_cpu_reference.json', 'full_candidate.json'))]
        for name in prediction_names:
            assert archive.read(name) == before[runner.run_dir / name]
        assert not any(name.endswith(('.onnx', '.npz', '.png')) for name in names)
    assert all(path.read_bytes() == encoded for path, encoded in before.items())
    return exported


@pytest.mark.parametrize('task,bad', [('classification', False), ('classification', True), ('detection', False), ('detection', True)])
def test_real_reference_central_quality_native_prerequisite_report_and_zip(tmp_path, task, bad):
    runner, logs = _runner(tmp_path)
    try:
        model, suite, status = _schedule(runner, tmp_path, task)
        assert status['status'] == 'completed', status
        records, consumed, encoded = runner._management_reference_records(model)
        assert len(records) == 3 and encoded == Path(status['reference_path']).read_bytes()
        assert not consumed.get('setup_id')
        for key in ('include_in_latency_fps_energy', 'include_in_ranking', 'include_in_pareto'):
            assert consumed[key] is False
        path, row = _candidate_request(runner, model, task, _reference_bytes(status), bad=bad)
        assert runner._queue_central_quality_requests(model) == 1
        artifacts, metrics, _, stage_status = runner._stage_evaluate_quality()
        summary = json.loads(artifacts['central_quality_summary_json'].read_text())
        assert summary['request_count'] == summary['completed_count'] == 1, summary
        assert summary['failed_count'] == 0 and summary['cpu_reference_benchmark_row_count'] == 0
        result = summary['results'][0]
        assert result['technical_status'] == 'completed', result
        assert result['decision'] == ('fail' if bad else 'pass'), result
        assert result['n'] == 3
        assert summary['merge']['matched_primary_result_count'] == 1
        assert summary['merge']['unmatched_result_count'] == 0
        assert result['source_request_sha256'].removeprefix('sha256:') == hashlib.sha256(path.read_bytes()).hexdigest()
        job = {'backend': row['backend'], 'model': model, 'case': 'b001', 'setup_id': row['setup_id'],
               'precision': 'fp32', 'comparison_backend': 'hailo8' if task == 'classification' else 'deepx'}
        # Technical quality completion permits subsequent admission checks even
        # if the candidate quality decision fails. This is not runtime approval.
        prerequisite = native_job_prerequisite(job, suite_dir=suite, model_dir=runner.run_dir / 'models' / model)
        assert prerequisite['prerequisite_status'] == 'ready'
        assert not any(key in prerequisite for key in ('fps', 'energy_j', 'latency_ms'))
        # A missing required execution binding remains blocked; synthetic
        # candidates are never promoted into a physical Native measurement.
        blocked = native_job_prerequisite(job, suite_dir=suite, model_dir=runner.run_dir / 'models' / model,
                                         quality_error='synthetic_candidate_has_no_physical_native_binding')
        assert blocked['prerequisite_status'] == 'blocked' and blocked['repetition_count_attempted'] == 0
        write_json(runner.run_dir / 'native_prerequisites.json', {'rows': [blocked], 'runtime_starts': 0})
        (runner.run_dir / 'profile.yaml').write_text(yaml.safe_dump(runner.profile_payload))
        report = build_scientific_reports(runner.run_dir, cleanup_legacy=False)
        scientific = json.loads(Path(report['artifacts']['scientific_report_json']).read_text())
        assert any(value.get('model_id') == model for value in scientific['model_facts']), scientific.keys()
        assert report['model_count'] == scientific['summary']['model_count'] == 1
        dashboard = json.loads(Path(report['artifacts']['result_dashboard_json']).read_text())
        assert dashboard['summary']['model_count'] == 1
        assert dashboard['quality_decision'] == ('fail' if bad else 'pass')
        assert scientific['summary']['energy_claim_eligible_row_count'] == 0
        assert scientific['summary']['performance_claim_eligible_row_count'] == 0
        _assert_debug_pack(runner, tmp_path, [model])
    finally:
        runner._shutdown_management_services()


def test_failed_reference_keeps_good_model_quality_summary_dashboard_and_debug(tmp_path):
    runner, logs = _runner(tmp_path)
    try:
        good_model, suite, good_status = _schedule(runner, tmp_path / 'good', 'classification')
        assert good_status['status'] == 'completed', good_status
        failed_model, failed_suite, failed_status = _schedule(
            runner, tmp_path / 'broken', 'classification', broken=True, model_id='broken_classification')
        assert failed_status['status'] == 'failed', failed_status
        assert failed_status['return_code'] != 0
        assert 'integration_injected_reference_failure' in failed_status['error']
        assert 'quality_reference_not_emitted' in failed_status['errors']
        historical = {p: p.read_bytes() for p in (runner.run_dir / 'quality_management/references' / failed_model).glob('*') if p.is_file()}
        good_request, _ = _candidate_request(runner, good_model, 'classification', _reference_bytes(good_status))
        failed_request, _ = _candidate_request(runner, failed_model, 'classification', _reference_bytes(good_status))
        assert runner._queue_central_quality_requests(good_model) == 1
        assert runner._queue_central_quality_requests(failed_model) == 1
        # Discovering the same request twice cannot produce a duplicate row.
        assert runner._queue_central_quality_requests(good_model) == 0
        artifacts, _, _, _ = runner._stage_evaluate_quality()
        summary = json.loads(artifacts['central_quality_summary_json'].read_text())
        assert summary['request_count'] == 2 and summary['completed_count'] == 1 and summary['failed_count'] == 1, summary
        by_model = {result['model_id']: result for result in summary['results']}
        assert set(by_model) == {good_model, failed_model}
        assert by_model[good_model]['technical_status'] == 'completed'
        assert by_model[good_model]['decision'] == 'pass'
        failure = by_model[failed_model]
        assert failure['technical_status'] == 'failed'
        assert failure['decision'] == failure['scientific_status'] == 'unavailable'
        assert failure['failure_stage'] == 'management_cpu_reference'
        assert failure['management_cpu_reference']['return_code'] == failed_status['return_code']
        assert failure['management_cpu_reference']['stdout_path'] == failed_status['stdout_path']
        assert failure['management_cpu_reference']['source_contract_sha256'] == failed_status['source_contract_sha256']
        assert failed_status['error'] in failure['error']
        assert failed_status['stdout_path'] in failure['error']
        assert 'primary' not in failure and 'n' not in failure
        assert any(failed_status['error'] in line for line in logs)
        # The failed dependency occurs before the candidate loader. Its
        # terminal central diagnostic retains identity but is not allowed to
        # become a validated row binding. The independent valid row joins.
        assert summary['merge']['matched_primary_result_count'] == 1
        assert summary['merge']['unmatched_result_count'] == 1
        assert summary['merge']['unmatched_results'][0]['model_id'] == failed_model
        for model, expected in ((good_model, 'pass'), (failed_model, 'pending_central_evaluation')):
            normalized = json.loads((runner.run_dir / 'models' / model / 'benchmark_results/normalized_results.json').read_text())
            assert len(normalized['results']) == 1
            assert normalized['results'][0]['task_quality_gate']['decision'] == expected
        (runner.run_dir / 'profile.yaml').write_text(yaml.safe_dump(runner.profile_payload))
        report = build_scientific_reports(runner.run_dir, cleanup_legacy=False)
        scientific = json.loads(Path(report['artifacts']['scientific_report_json']).read_text())
        assert {row['model_id'] for row in scientific['model_facts']} == {good_model, failed_model}
        assert report['model_count'] == 2
        dashboard = json.loads(Path(report['artifacts']['result_dashboard_json']).read_text())
        assert dashboard['summary']['model_count'] == 2
        assert {row['model_id'] for row in dashboard['central_quality_results']} == {good_model, failed_model}
        # Dashboard/report generation must preserve available model facts even
        # though this synthetic fixture has no physical benchmark population.
        _assert_debug_pack(runner, tmp_path, [good_model, failed_model])
        assert all(p.read_bytes() == value for p, value in historical.items())
    finally:
        runner._shutdown_management_services()


def test_historical_run_replay_keeps_negative_rows_and_original_files(tmp_path):
    from scripts.run_complete_set_replay_v27931 import replay
    fixture = ROOT / 'tests/fixtures/v27931_complete_set'
    before = {p: p.read_bytes() for p in fixture.rglob('*') if p.is_file()}
    result = replay(None, tmp_path / 'historical_replay')
    assert [result[k] for k in ('expected_row_count', 'present_expected_row_count',
        'successful_expected_row_count', 'failed_expected_row_count', 'missing_expected_row_count')] == [63, 63, 55, 8, 0]
    assert result['original_inputs_byte_unchanged'] is True
    assert result['historical_measurements_modified'] is False
    assert all(p.read_bytes() == data for p, data in before.items())


def test_real_reference_consumer_rejects_status_and_candidate_contract_tampering(tmp_path):
    import concurrent.futures
    runner, _ = _runner(tmp_path)
    try:
        model, suite, status = _schedule(runner, tmp_path, 'classification')
        assert status['status'] == 'completed', status
        reference_path = Path(status['reference_path'])
        original_reference = reference_path.read_bytes()
        future = runner._management_reference_futures[model]
        for field, replacement in (
            ('model_id', 'another_model'),
            ('source_contract_sha256', 'f' * 64),
            ('reference_sha256', 'f' * 64),
            ('image_ids_sha256', 'f' * 64),
            ('provider', 'onnxruntime_cuda'),
        ):
            mutated = copy.deepcopy(status)
            mutated[field] = replacement
            candidate_future = concurrent.futures.Future()
            candidate_future.set_result(mutated)
            runner._management_reference_futures[model] = candidate_future
            with pytest.raises(RuntimeError):
                runner._management_reference_records(model)
        runner._management_reference_futures[model] = future
        assert len(runner._management_reference_records(model)[0]) == 3
        path, _ = _candidate_request(runner, model, 'classification', _reference_bytes(status))
        manifest = json.loads(path.read_text())
        candidate_path = path.with_name('full_candidate.json')
        original_candidate = candidate_path.read_bytes()
        original_manifest = path.read_bytes()
        for mutation in ('image_id', 'label_id', 'model_contract', 'preprocessing_contract'):
            candidate = json.loads(original_candidate)
            if mutation == 'image_id':
                candidate['records'][0]['image_id'] = 'unbound-image.png'
            elif mutation == 'label_id':
                candidate['records'][0]['label_id'] = 999
            elif mutation == 'model_contract':
                candidate['quality_contract']['model']['sha256'] = 'f' * 64
            else:
                candidate['quality_contract']['preprocessing']['identity']['image_scale'] = 'unbound-scale'
            write_json(candidate_path, candidate)
            request = copy.deepcopy(manifest)
            request['candidate'].update(size_bytes=candidate_path.stat().st_size,
                sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest())
            write_json(path, request)
            result = runner._evaluate_central_quality_request(model, path)
            assert result['technical_status'] == 'failed' and result['decision'] == 'unavailable', result
            assert result['producer_binding_eligible'] is False
            assert 'primary' not in result
            assert reference_path.read_bytes() == original_reference
        candidate_path.write_bytes(original_candidate)
        path.write_bytes(original_manifest)
        result = runner._evaluate_central_quality_request(model, path)
        assert result['technical_status'] == 'completed' and result['decision'] == 'pass', result
        assert reference_path.read_bytes() == original_reference
    finally:
        runner._shutdown_management_services()
