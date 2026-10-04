from __future__ import annotations
import json
from pathlib import Path

from onnx_splitpoint_tool.workflow.supplement import build_supplement_plan, write_plan
from onnx_splitpoint_tool.workflow.status_reporting import energy_attempt_failure
from tests.test_v2911_supplement_workflow import archive, write


def test_policy_is_separate_from_compile_and_never_new_generic_work(archive, tmp_path):
    source = archive / 'models/yolo26m/benchmark_set/legacy_suite/b002/split_manifest.json'
    payload = json.loads(source.read_text())
    payload['hailo'] = {'backend_exclusions': [dict(folder='b002', hw_arch='hailo8', stage='part1',
        status='rejected', reason='hailo_yolo26_static_part1_guard', detail='historical backfill text')]}
    write(source, payload)
    plan = build_supplement_plan(archive)
    assert plan['counts']['generic'] == dict(preserve=1, policy_excluded=1, negative_completed=1)
    excluded = next(r for r in plan['generic_rows'] if r['action'] == 'policy_excluded')
    assert excluded['source_manifest'].endswith('b002/split_manifest.json')
    assert excluded['reason'] == 'hailo_yolo26_static_part1_guard'
    out = tmp_path / 'report'; out.mkdir()
    _, md = write_plan(plan, out)
    assert 'Part2-only' in md.read_text()
    assert plan == build_supplement_plan(archive)


def test_failed_statistics_with_original_predictions_are_local_work(archive):
    source = archive / 'quality_management/central_quality_summary.json'
    payload = json.loads(source.read_text())
    payload['results'][0].update(status='failed', technical_status='failed', error='reference binding')
    write(source, payload)
    plan = build_supplement_plan(archive)
    assert plan['completed_quality_count'] == 0
    assert len(plan['pending_quality_rows']) == 1
    assert plan['pending_quality_rows'][0]['action'] == 'local_statistics'
    # Retained Generic remains independent of local statistics.
    assert plan['counts']['generic']['preserve'] == 1


def failure_run(root):
    write(root / 'run_manifest.json', {'execution_sessions': [{'session_id': 'old'}, {'session_id': 'new'}]})
    energy = root / 'reports/native_energy_measurements'
    old = write(energy / 'native_producer_energy_results.json', {'rows': [dict(old=True)], 'started_measurement_count': 0})
    attempt = energy / 'attempts/new'
    att = write(attempt / 'row/preflight_attestation.json', {'failure_reason': 'RuntimeError:preflight_artifact_missing:artifact:cmake:/original/CMakeLists.txt'})
    write(attempt / 'resume_failure.json', dict(status='resume_preparation_or_cohort_preflight_failed',
        started_measurement_count=0, collector_started_repeat_count=0, workload_started_repeat_count=0,
        measurement_wrapper_started_count=0, resume_preparation={'cohort_preflight': dict(
            preflight_attempted_count=222, preflight_passed_count=142, preflight_failed_count=80,
            rows=[dict(ok=False, reason='preflight_evidence_not_verified', identity={'model':'m'},
                       preflight={'attestation_path':str(att)})])}))
    return old, attempt


def test_failure_projection_uses_attempt_order_and_keeps_historical_results(tmp_path):
    old, attempt = failure_run(tmp_path)
    before = old.read_bytes()
    projection = energy_attempt_failure(tmp_path)
    assert projection['attempt_id'] == 'new'
    assert projection['preflight_attempted_count'] == 222
    assert projection['preflight_failed_count'] == 80
    assert projection['failure_counts'] == {'preflight_artifact_missing:artifact:cmake': 1}
    assert str(attempt / 'resume_failure.json') in projection['energy_not_started_reason']
    assert 'wrapper/collector/workload=0/0/0' in projection['energy_not_started_reason']
    assert old.read_bytes() == before
    # A later completed result is authoritative; do not resurrect an old abort.
    write(attempt / 'native_producer_energy_results.json', {'ok':True})
    assert energy_attempt_failure(tmp_path) == {}


def test_normal_dashboard_displays_attempt_error_without_mutating_results(tmp_path):
    from onnx_splitpoint_tool.workflow.dashboard import build_result_dashboard_reports
    old, attempt = failure_run(tmp_path)
    before = old.read_bytes()
    build_result_dashboard_reports(tmp_path, profile_id='p', tool_version='2.91.2', workflow_version='v')
    report = json.loads((tmp_path / 'reports/result_dashboard.json').read_text())
    assert report['overview']['energy_attempt_failure']['preflight_failed_count'] == 80
    md = (tmp_path / 'reports/result_dashboard.md').read_text()
    assert '142/222 PASS, 80 FAIL' in md and 'artifact:cmake' in md
    assert old.read_bytes() == before


from tests.test_v27924_terminal_hailo_aliases import published


def test_real_published_bundle_reentry_below_native_attempt(published, tmp_path):
    import shutil
    import pytest
    from onnx_splitpoint_tool.workflow.runner import WorkflowRunTargetError
    runner, alias, hef = published
    destination = runner.run_dir / 'native_producers/attempts/new/deepx/mobilenet_v3_large/benchmark_set/b001/hailo/hailo8/part1'
    destination.parent.mkdir(parents=True)
    shutil.copytree(alias.parent, destination, symlinks=True)
    runner._validate_resume_target_paths()
    # Attempt layout does not relax the bundle publication or output guards.
    pointer = destination / '.hailo-current'
    pointer.unlink(); pointer.symlink_to(tmp_path)
    with pytest.raises(WorkflowRunTargetError, match='symlink/non-regular'):
        runner._validate_resume_target_paths()
