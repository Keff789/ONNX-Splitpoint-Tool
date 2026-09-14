"""v2.80.2 real Generator→Suite→ORT→central Quality→Native blocker→Report.

Synthetic: tiny source ONNX/images and explicitly diagnostic accelerator
candidate inputs. Real: legacy materializer, reference subprocess/ORT, paired
quality evaluator, prerequisite/producer validation, persistence and reports.
No successful reference, Quality result or Native measurement is fabricated.
Historical fixtures remain immutable. Hardware acceptance is NOT_RUN.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from onnx_splitpoint_tool.gui.dashboard_projection import dashboard_summary_line
from onnx_splitpoint_tool.native_job_identity import native_job_prerequisite
from onnx_splitpoint_tool.trt_quality_chain import (
    TensorRTQualityChainError, load_producer_set_from_central_quality_summary,
)
from onnx_splitpoint_tool.workflow.artifacts import write_json
from onnx_splitpoint_tool.workflow.runner import (
    _native_tensorrt_full_quality_blocked_rows_v2802,
    _persist_native_blocked_rows,
)
from onnx_splitpoint_tool.workflow.scientific_reporting import build_scientific_reports

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/v2802_completsetdev_regression'


def _module(filename):
    name = 'v2802_integration_' + Path(filename).stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _runner(root):
    runner, logs = _module('test_v2801_workflow_integration.py')._runner(root)
    runner.profile_id = 'v2802_real_generator_synthetic_accelerator_candidate'
    runner.profile_payload['id'] = runner.profile_id
    return runner, logs


def _schedule_generated(runner, root, task, *, requested_model_id=None):
    suite = _module('test_v2802_cpu_reference_binding.py').make_generated_reference_suite(root, task)
    actual_model = 'tiny_' + task
    model = requested_model_id or actual_model
    model_dir = runner.run_dir / 'models' / model
    (model_dir / 'benchmark_set').mkdir(parents=True)
    for filename in ('benchmark_plan.json', 'benchmark_set.json'):
        shutil.copy2(suite / filename, model_dir / 'benchmark_set' / filename)
    write_json(model_dir / 'model_manifest.json', {'model_id': model, 'task': task,
        'resolved_path': str(suite / 'models' / (actual_model + '.onnx'))})
    runner.profile_payload['models'].append({'id': model, 'task': task})
    runner.manifest.setdefault('models', {})[model] = {'model_id': model, 'task': task}
    runner._schedule_management_cpu_reference(model, suite)
    future = runner._management_reference_futures[model]
    runner._schedule_management_cpu_reference(model, suite)
    assert runner._management_reference_futures[model] is future
    return model, suite, future.result(timeout=90)


def _candidate(runner, model, task, reference, *, bad=False):
    # Reuse only the pre-existing explicitly synthetic candidate-input writer.
    # All repaired producers and their resulting success/failure bytes stay real.
    return _module('test_v2801_workflow_integration.py')._candidate_request(
        runner, model, task, reference, bad=bad)


def _blocked_full_from_actual_summary(runner, model, setup, summary_path):
    planned = {'backend': 'native_full_tensorrt', 'model': model, 'case': 'full',
               'setup_id': setup, 'comparison_backend': 'hailo8', 'precision': '',
               'execution_mode': 'native_full_baseline'}
    # The synthetic candidate has no signed physical TRT producer. The actual
    # strict loader must reject it even after a successful quality computation.
    with pytest.raises(TensorRTQualityChainError) as raised:
        load_producer_set_from_central_quality_summary(summary_path,
            eval_run_id=runner.run_id, setup_id=setup, model_ids=[model])
    reason = type(raised.value).__name__ + ': ' + str(raised.value)
    quality_first = {'central_quality_summary': str(summary_path),
        'producer_sets_by_setup': {}, 'errors_by_setup_model': {setup: {model: reason}}}
    blocked = _native_tensorrt_full_quality_blocked_rows_v2802(
        [planned], quality_first, repetition_count_requested=3)
    assert len(blocked) == 1
    row = blocked[0]
    assert row['failure_reason'] == row['error'] == reason
    assert row['failure_class'] == 'upstream_quality_evidence'
    assert row['status'] == 'blocked_upstream_quality'
    assert row['repetition_count_attempted'] == row['repetition_count_valid'] == 0
    assert row['fps_makespan'] is None and row['runtime_success'] is False
    assert 'native_transfer_failed' not in json.dumps(row)
    artifacts, roots = {}, []
    _persist_native_blocked_rows(runner.run_dir, blocked, expected_native_rows=[planned],
        artifact_paths=artifacts, collected_roots=roots, repetition_count_requested=3)
    original_blocker = artifacts['native_full_blocked_jobs_json'].read_bytes()
    # Replay/persistence deduplicates this exact unstarted planned identity.
    _persist_native_blocked_rows(runner.run_dir, blocked, expected_native_rows=[planned],
        artifact_paths=artifacts, collected_roots=roots, repetition_count_requested=3)
    assert artifacts['native_full_blocked_jobs_json'].read_bytes() == original_blocker
    report = subprocess.run([sys.executable, str(ROOT / 'scripts/native_producer_final_report.py'),
        '--root', str(roots[0]), '--recursive', '--out-dir', str(runner.run_dir / 'reports')],
        capture_output=True, text=True, timeout=60)
    assert report.returncode == 0, report.stdout + report.stderr
    native = _read(runner.run_dir / 'reports/native_producer_combined_summary.json')
    assert len(native['rows']) == 1
    assert native['rows'][0]['failure_reason'] == reason
    assert native['rows'][0]['repetition_count_attempted'] == 0
    return row


@pytest.mark.parametrize('task,bad', [
    ('classification', False), ('classification', True),
    ('detection', False), ('detection', True),
])
def test_real_generated_reference_quality_native_blocker_and_report(tmp_path, task, bad):
    runner, _ = _runner(tmp_path)
    try:
        model, suite, status = _schedule_generated(runner, tmp_path / 'generated', task)
        assert status['status'] == 'completed', status
        assert status['return_code'] == 0
        reference_path = Path(status['reference_path'])
        original_reference = reference_path.read_bytes()
        records, consumed, encoded = runner._management_reference_records(model)
        assert len(records) == 3 and encoded == original_reference
        assert consumed['semantic_reference_only'] is True
        assert all(consumed[key] is False for key in (
            'include_in_latency_fps_energy', 'include_in_ranking', 'include_in_pareto'))
        generated = _read(suite / 'benchmark_set.json')
        assert generated['model_name'] == model
        assert generated['model'] == f'models/{model}.onnx'
        request_path, candidate_row = _candidate(runner, model, task, _read(reference_path), bad=bad)
        assert runner._queue_central_quality_requests(model) == 1
        assert runner._queue_central_quality_requests(model) == 0
        artifacts, _, _, _ = runner._stage_evaluate_quality()
        summary_path = artifacts['central_quality_summary_json']
        summary = _read(summary_path)
        assert summary['request_count'] == summary['completed_count'] == 1
        assert summary['failed_count'] == summary['cpu_reference_benchmark_row_count'] == 0
        result = summary['results'][0]
        assert result['technical_status'] == 'completed'
        assert result['decision'] == ('fail' if bad else 'pass')
        assert result['n'] == 3
        assert result['source_request_sha256'].removeprefix('sha256:') == _digest(request_path)
        assert summary['merge']['matched_primary_result_count'] == 1
        assert summary['merge']['unmatched_result_count'] == 0
        prerequisite = native_job_prerequisite({'backend': candidate_row['backend'],
            'model': model, 'case': 'b001', 'setup_id': candidate_row['setup_id'],
            'precision': 'fp32', 'comparison_backend': 'hailo8' if task == 'classification' else 'deepx'},
            suite_dir=suite, model_dir=runner.run_dir / 'models' / model)
        assert prerequisite['prerequisite_status'] == 'ready'
        _blocked_full_from_actual_summary(runner, model, candidate_row['setup_id'], summary_path)
        (runner.run_dir / 'profile.yaml').write_text(yaml.safe_dump(runner.profile_payload))
        report = build_scientific_reports(runner.run_dir, cleanup_legacy=False)
        dashboard = _read(report['artifacts']['result_dashboard_json'])
        scientific = _read(report['artifacts']['scientific_report_json'])
        assert dashboard['summary']['model_count'] == 1
        assert 'models=1' in dashboard_summary_line(dashboard)
        assert dashboard['quality_decision'] == ('fail' if bad else 'pass')
        assert scientific['summary']['performance_claim_eligible_row_count'] == 0
        assert scientific['summary']['energy_claim_eligible_row_count'] == 0
        native_rows = scientific['native_performance_matrix']['observations']
        assert len(native_rows) == 1
        assert native_rows[0]['repetition_count_attempted'] == 0
        assert native_rows[0]['failure_reason'] != 'native_transfer_failed'
        _module('test_v2801_workflow_integration.py')._assert_debug_pack(runner, tmp_path, [model])
        assert reference_path.read_bytes() == original_reference
    finally:
        runner._shutdown_management_services()


def test_real_wrong_model_reference_failure_remains_unavailable_and_preserves_good_model(tmp_path):
    runner, _ = _runner(tmp_path)
    try:
        good, suite, good_status = _schedule_generated(runner, tmp_path / 'good', 'classification')
        assert good_status['status'] == 'completed'
        wrong, _, failed_status = _schedule_generated(runner, tmp_path / 'wrong', 'classification',
            requested_model_id='wrong_model_identity')
        assert failed_status['status'] == 'failed'
        assert 'model' in failed_status['error'].lower(), failed_status
        history = {p: p.read_bytes() for p in (runner.run_dir / 'quality_management/references' / wrong).rglob('*') if p.is_file()}
        reference = _read(good_status['reference_path'])
        for model in (good, wrong):
            _candidate(runner, model, 'classification', reference)
            assert runner._queue_central_quality_requests(model) == 1
        artifacts, _, _, _ = runner._stage_evaluate_quality()
        summary = _read(artifacts['central_quality_summary_json'])
        assert (summary['request_count'], summary['completed_count'], summary['failed_count']) == (2, 1, 1)
        by_model = {row['model_id']: row for row in summary['results']}
        assert by_model[good]['decision'] == 'pass'
        failure = by_model[wrong]
        assert failure['technical_status'] == 'failed'
        assert failure['decision'] == failure['scientific_status'] == 'unavailable'
        assert failure['failure_stage'] == 'management_cpu_reference'
        assert 'primary' not in failure and 'n' not in failure
        assert failed_status['error'] in failure['error']
        _blocked_full_from_actual_summary(runner, wrong, 'synthetic_hailo8_setup',
                                         artifacts['central_quality_summary_json'])
        (runner.run_dir / 'profile.yaml').write_text(yaml.safe_dump(runner.profile_payload))
        report = build_scientific_reports(runner.run_dir, cleanup_legacy=False)
        dashboard = _read(report['artifacts']['result_dashboard_json'])
        assert dashboard['summary']['model_count'] == 2
        assert 'models=2' in dashboard_summary_line(dashboard)
        assert dashboard['quality_decision'] != 'fail'
        assert {r['model_id'] for r in dashboard['central_quality_results']} == {good, wrong}
        _module('test_v2801_workflow_integration.py')._assert_debug_pack(runner, tmp_path, [good, wrong])
        assert all(path.read_bytes() == data for path, data in history.items())
    finally:
        runner._shutdown_management_services()


def test_original_completsetdev_fixtures_provenance_and_historical_failure_axes():
    provenance = _read(FIXTURES / 'PROVENANCE.json')
    for row in provenance['original_members'] + provenance['derived_members']:
        path = FIXTURES / row['path']
        assert path.stat().st_size == row['size_bytes']
        assert _digest(path) == row['sha256']
    indexed = _read(FIXTURES / 'derived/original_index_records.json')['artifacts']
    assert len(indexed) == 14
    for row in indexed:
        assert _digest(FIXTURES / row['path']) == row['sha256'].removeprefix('sha256:')
        assert (FIXTURES / row['path']).stat().st_size == row['size_bytes']
    quality = _read(FIXTURES / 'derived/central_quality_summary.json')
    assert quality['request_count'] == quality['failed_count'] == 71
    assert quality['completed_count'] == 0 and quality['quality_decision'] == 'not_evaluated'
    assert len({r['model_id'] for r in quality['results']}) == 7
    assert all(r['decision'] == 'unavailable' and 'model_binding' in r['error'] for r in quality['results'])
    replicas = []
    for backend in ('hailo8', 'hailo10h', 'deepx'):
        payload = _read(FIXTURES / f'derived/native_full_baseline_{backend}.json')
        assert len(payload['rows']) == 7
        for row in payload['rows']:
            assert len(row['repetition_records']) == 3
            replicas.extend(row['repetition_records'])
    assert len(replicas) == 63
    assert sum(row['completed_frames'] for row in replicas) == 63000
    assert all(row['runtime_success'] is True and row['returncode'] == 0 and row['timed_out'] is False for row in replicas)
    old_matrix = _read(FIXTURES / 'reports/native_expected_matrix.json')
    assert old_matrix['expected_row_count'] == 63
    assert len(old_matrix['missing_expected_rows']) == 21
    assert all(row['failure_reason'] == 'native_transfer_failed' for row in old_matrix['missing_expected_rows'])
    dashboard = _read(FIXTURES / 'reports/result_dashboard.json')
    assert 'models=7' in dashboard_summary_line(dashboard)
    assert dashboard['quality_decision'] == 'not_evaluated'
