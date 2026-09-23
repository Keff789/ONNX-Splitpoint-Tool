"""AP05: synthetic offline loader/service/writer/UI integration; no hardware starts."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest

from onnx_splitpoint_tool import quality_replay as replay
from onnx_splitpoint_tool import quality_service as service_module
from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY, assess_accuracy
from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile, save_evaluation_profile_yaml,
)
from onnx_splitpoint_tool.quality_cache import json_fingerprint
from test_quality_speed_settings import _custom, _profile
from test_v27522_quality_ap75_replay import _make_replay_run, _write_json


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _modern_run(tmp_path):
    run, _ = _make_replay_run(tmp_path)
    summary_path = run / 'quality_management/central_quality_summary.json'
    summary = json.loads(summary_path.read_text())
    reference_path = Path(summary['results'][0]['management_cpu_reference']['reference_path'])
    reference = json.loads(reference_path.read_text())
    reference['records'] = [dict(copy.deepcopy(reference['records'][0]), image_id=f'i{i}') for i in range(3)]
    _write_json(reference_path, reference)
    for index in (0, 1):
        row = summary['results'][index]
        path = run / row['source_request']
        manifest = json.loads(path.read_text())
        candidate_path = path.parent / manifest['candidate']['path']
        candidate = json.loads(candidate_path.read_text())
        candidate['records'] = [dict(copy.deepcopy(candidate['records'][0]), image_id=f'i{i}') for i in range(3)]
        # A real accuracy loss with all AP warnings, not identical predictions.
        for record in candidate['records']:
            record['candidate'][0]['x1'] = 20.0
            record['candidate'][0]['x2'] = 30.0
        manifest['candidate'] = _write_json(candidate_path, candidate)
        manifest['metric_gate_config']['reporting_policy'] = copy.deepcopy(DEFAULT_REPORTING_POLICY)
        manifest['statistics']['bootstrap_repetitions'] = 11
        manifest['record_count'] = 3
        manifest['expected_image_ids'] = ['i0', 'i1', 'i2']
        _write_json(path, manifest)
        row['source_request_sha256'] = _digest(path)
        row['management_cpu_reference']['reference_sha256'] = _digest(reference_path)
        row['technical_status'] = 'completed'
        request = service_module.quality_request_from_manifest(path, reference_artifact=reference_path)
        _, prepared = service_module.prepare_evaluation(request)
        for key in ('reference_predictions_sha256', 'candidate_predictions_sha256', 'annotations_sha256'):
            row[key] = prepared[key]
    _write_json(summary_path, summary)
    return run


def _producer_run(tmp_path):
    run = _modern_run(tmp_path)
    summary_path = run / 'quality_management/central_quality_summary.json'
    summary = json.loads(summary_path.read_text())
    row = summary['results'][0]
    path = run / row['source_request']
    manifest = json.loads(path.read_text())
    candidate_path = path.parent / manifest['candidate']['path']
    candidate = json.loads(candidate_path.read_text())
    completion = {'schema': 'onnx-splitpoint/hailo-full-host-tail-completion', 'schema_version': 1,
                  'completed_endpoint': 'decoded_xyxy_score_class_detection_records'}
    completion['contract_sha256'] = json_fingerprint(completion)
    for value in (manifest, candidate):
        value['candidate_execution_completion_contract'] = completion
        value['candidate_execution_completion_contract_sha256'] = completion['contract_sha256']
    manifest['candidate'] = _write_json(candidate_path, candidate)
    _write_json(path, manifest)
    old_reference = Path(row['management_cpu_reference']['reference_path'])
    immutable = old_reference.parent / 'by_source_contract' / ('a' * 64) / old_reference.name
    immutable.parent.mkdir(parents=True)
    immutable.write_bytes(old_reference.read_bytes())
    status_path = old_reference.parent / 'management_cpu_reference_status.json'
    status = dict(schema='onnx-splitpoint/management-cpu-reference-job', schema_version=1,
        model_id='yolov7_paper', status='completed', provider='onnxruntime_cpu',
        execution_location='central_management', semantic_reference_only=True,
        include_in_latency_fps_energy=False, include_in_ranking=False, include_in_pareto=False,
        reference_storage='immutable_source_contract', reference_immutable=True,
        source_contract_sha256='a' * 64, reference_path=str(immutable),
        reference_size_bytes=immutable.stat().st_size, reference_sha256=_digest(immutable))
    _write_json(status_path, status)
    row.update(technical_status='cancelled', decision='cancelled', source_request_sha256=_digest(path))
    for key in ('management_cpu_reference', 'reference_predictions_sha256',
                'candidate_predictions_sha256', 'annotations_sha256'):
        row.pop(key, None)
    _write_json(summary_path, summary)
    return run, status_path


@pytest.fixture
def forbidden_starts(monkeypatch):
    """Coordinator guards; spawned workers use only the CPU statistics entrypoint."""
    import onnxruntime
    from onnx_splitpoint_tool import management_reference, hailo_backend
    from onnx_splitpoint_tool.remote import process_lease
    from onnx_splitpoint_tool.remote.ssh_transport import SSHTransport
    from onnx_splitpoint_tool.benchmark import remote_run
    from onnx_splitpoint_tool.energy import collector
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
    from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('offline replay crossed a hardware/inference/workflow start boundary')
    for owner, names in (
        (onnxruntime, ['InferenceSession']),
        (management_reference, ['generate_management_cpu_reference']),
        (SSHTransport, ['run', 'run_read_only', 'run_streaming', 'scp_upload', 'scp_download']),
        (process_lease, ['run_journaled_ssh']), (remote_run, ['run_remote_benchmark']),
        (hailo_backend, ['hailo_build_hef', 'hailo_build_hef_auto']),
        (collector.EnergyMeasurementService, ['run_measurement']),
        (collector, ['run_fast_firmware_measurement', 'run_duration_probe']),
        (EvaluationWorkflowRunner, ['run', '_schedule_management_cpu_reference', '_stage_prepare_full_baselines',
            '_stage_build_backend_artifacts', '_stage_hardware_smoke', '_stage_run_benchmarks', '_stage_run_native_producers']),
        (SplitPointAnalyserGUI, ['_queue_evaluation_workflow', '_queue_evaluation_validation_assets']),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    yield calls
    assert calls == []


def _render(out_dir):
    from onnx_splitpoint_tool.gui.app import SplitPointAnalyserGUI
    rendered = []
    view = SimpleNamespace(_eval_workflow_text_set=rendered.append)
    SplitPointAnalyserGUI._eval_workflow_render_result(view, {'run_dir': str(out_dir), 'status': 'completed'})
    return rendered[0]


def test_normal_profile_spawn_writers_and_gui_preserve_saved_budget(tmp_path, monkeypatch, forbidden_starts):
    run = _modern_run(tmp_path)
    profile = tmp_path / 'normal_saved_profile.yaml'
    save_evaluation_profile_yaml(profile, _custom(_profile()))
    loaded = load_evaluation_profile(profile)
    assert loaded.start_snapshot['resolved_profile']['quality_gate']['statistics']['bootstrap_repetitions'] == 5000
    originals = {path: _digest(path) for path in run.rglob('*.json')}
    out_dir = tmp_path / 'derived'
    seen = []
    real_loader = replay.quality_request_from_manifest
    previous = []
    def observed_loader(*args, **kwargs):
        assert not previous or previous[-1]() is None, 'preceding heavy request retained at next load'
        request = real_loader(*args, **kwargs)
        seen.append((len(request.candidate_records), request.repetitions, request.seed,
                     request.confidence_level, copy.deepcopy(request.metric_gate_config)))
        return request
    monkeypatch.setattr(replay, 'quality_request_from_manifest', observed_loader)
    real_evaluate = replay.ManagementQualityService.evaluate
    def observed_evaluate(self, request, *args, **kwargs):
        previous.append(weakref.ref(request))
        return real_evaluate(self, request, *args, **kwargs)
    monkeypatch.setattr(replay.ManagementQualityService, 'evaluate', observed_evaluate)
    spawned = []
    real_pool = service_module.ProcessPoolExecutor
    def pool(*args, **kwargs):
        spawned.append((kwargs['max_workers'], kwargs['mp_context'].get_start_method()))
        return real_pool(*args, **kwargs)
    monkeypatch.setattr(service_module, 'ProcessPoolExecutor', pool)
    output = replay.replay_evaluation_run(run, out_dir=out_dir, profile=profile, row_indices=[1, 0])
    assert output['selected_source_row_indices'] == [1, 0]
    assert output['canonical_full_only_status'] == 'not_requested'
    assert spawned and spawned[0][1] == 'spawn'
    assert output['statistics_execution']['workers_requested'] == 2
    assert output['statistics_execution']['reference_threads_configured_but_not_executed'] == 6
    assert output['statistics_execution']['profile_start_snapshot'] == loaded.start_snapshot
    assert all(value[:4] == (3, 11, 20260710, .95) for value in seen)
    assert all(value[4]['reporting_policy'] == DEFAULT_REPORTING_POLICY for value in seen)
    assert output['decision_counts'] == {'accuracy_loss': 2}
    assert output['technical_status'] == 'ok' and output['scientific_pass'] is None
    row = output['results'][0]
    assert row['primary']['bootstrap_repetitions_requested'] == 11
    assert row['primary']['bootstrap_repetitions'] == 11
    assert row['accuracy_assessment']['relative_loss'] == 1.0
    assert row['accuracy_assessment']['relative_loss_ci'] == [1.0, 1.0]
    assert row['accuracy_warnings']
    observation = row['execution_observation']['statistics_observation']
    assert observation['engine'] == 'optimized_coco_v1'
    assert observation['shards']
    assert all(shard['worker_pid'] != os.getpid() for shard in observation['shards'])
    reports = output['scientific_report_paths']
    projected = json.loads(Path(reports['json']).read_text())
    assert projected[0]['accuracy_assessment'] == row['accuracy_assessment']
    assert projected[0]['technical_status'] == 'completed'
    assert projected[0]['performance_claims_emitted'] is False
    for csv_path in (Path(reports['csv']), out_dir / replay.REPLAY_CSV_NAME):
        with csv_path.open(newline='') as stream:
            saved = list(csv.DictReader(stream))[0]
        assert saved['accuracy_class'] == 'accuracy_loss'
        assert saved['accuracy_relative_loss_ci'] == '[1.0,1.0]' or saved['accuracy_relative_loss_ci'] == '[1.0, 1.0]'
        assert saved['accuracy_warnings']
    assert 'accuracy_loss' in Path(reports['markdown']).read_text()
    assert 'accuracy\\_loss' in Path(reports['latex']).read_text()
    rendered = _render(out_dir)
    assert 'Genauigkeitsverlust / statistisch gestützt' in rendered
    assert '100.00% relativ' in rendered and '95%-CI=[1.0, 1.0]' in rendered
    assert 'Warnung:' in rendered and 'Accuracyreport nicht lesbar' not in rendered
    assert {path: _digest(path) for path in originals} == originals
    # Pair-cache warmth must not change the scientific result identity.
    warm = replay.replay_evaluation_run(run, out_dir=out_dir, profile=profile, row_indices=[1, 0])
    assert [r['scientific_result_sha256'] for r in warm['results']] == [r['scientific_result_sha256'] for r in output['results']]
    legacy = replay.replay_evaluation_run(run, out_dir=tmp_path / 'legacy', profile=profile,
        row_indices=[1, 0], statistics={'engine': 'legacy'})
    assert [r['scientific_result_sha256'] for r in legacy['results']] == [r['scientific_result_sha256'] for r in output['results']]


@pytest.mark.parametrize('indices', [[], [True], [-1], [6], [0, 0], '0'])
def test_invalid_row_indices_fail_before_output(tmp_path, indices):
    run, _ = _make_replay_run(tmp_path)
    out = tmp_path / 'must_not_exist'
    with pytest.raises(replay.OfflineQualityReplayError, match='row_indices'):
        replay.replay_evaluation_run(run, out_dir=out, row_indices=indices)
    assert not out.exists()


@pytest.mark.parametrize('options', [dict(engine='typo'), dict(block_repetitions=0),
    dict(checkpoint_blocks=1), dict(bootstrap_repetitions=1), dict(max_active_requests=3)])
def test_execution_options_cannot_replace_scientific_budget(tmp_path, options):
    run, _ = _make_replay_run(tmp_path)
    out = tmp_path / 'must_not_exist'
    with pytest.raises(replay.OfflineQualityReplayError, match='invalid offline execution settings'):
        replay.replay_evaluation_run(run, out_dir=out, row_indices=[0], statistics=options)
    assert not out.exists()


def test_explicit_producer_only_keeps_cancelled_history(tmp_path, forbidden_starts):
    run, status = _producer_run(tmp_path)
    originals = {path: _digest(path) for path in run.rglob('*.json')}
    with pytest.raises(replay.OfflineQualityReplayError, match='requires explicit producer-only'):
        replay.replay_evaluation_run(run, out_dir=tmp_path / 'rejected', row_indices=[0])
    result = replay.replay_evaluation_run(run, out_dir=tmp_path / 'producer', row_indices=[0],
        workers=1, statistics={'engine': 'optimized_coco_v1', 'block_repetitions': 4},
        producer_reference_statuses={0: status})
    row = result['results'][0]
    assert row['admission_kind'] == 'complete_producer_records_only'
    assert row['historical_technical_status'] == 'cancelled'
    assert row['historical_ci_available'] is False
    assert row['historical_decision'] == 'cancelled'
    assert row['technical_status'] == 'completed' and row['primary']['bootstrap_repetitions'] == 11
    assert row['producer_reference_status_sha256'] == _digest(status)
    assert {path: _digest(path) for path in originals} == originals


@pytest.mark.parametrize(('field', 'value'), [('provider', 'cuda'), ('status', 'running'),
    ('reference_sha256', 'b' * 64), ('source_contract_sha256', 'b' * 64),
    ('semantic_reference_only', 1), ('reference_size_bytes', True)])
def test_invalid_producer_reference_status_fails_closed(tmp_path, field, value, forbidden_starts):
    run, status = _producer_run(tmp_path)
    payload = json.loads(status.read_text())
    payload[field] = value
    _write_json(status, payload)
    with pytest.raises(replay.OfflineQualityReplayError):
        replay.replay_evaluation_run(run, out_dir=tmp_path / 'rejected', row_indices=[0],
                                    producer_reference_statuses={0: status})
    assert not (tmp_path / 'rejected').exists()


@pytest.mark.parametrize('case', ['intermediate_endpoint', 'missing_population_binding'])
def test_producer_only_rejects_incomplete_task_contract(tmp_path, case, forbidden_starts):
    run, status = _producer_run(tmp_path)
    summary_path = run / 'quality_management/central_quality_summary.json'
    summary = json.loads(summary_path.read_text())
    row = summary['results'][0]
    request_path = run / row['source_request']
    manifest = json.loads(request_path.read_text())
    if case == 'intermediate_endpoint':
        candidate_path = request_path.parent / manifest['candidate']['path']
        candidate = json.loads(candidate_path.read_text())
        completion = dict(manifest['candidate_execution_completion_contract'])
        completion.pop('contract_sha256')
        completion['completed_endpoint'] = 'hailo_raw_detection_head'
        completion['contract_sha256'] = json_fingerprint(completion)
        for value in (manifest, candidate):
            value['candidate_execution_completion_contract'] = completion
            value['candidate_execution_completion_contract_sha256'] = completion['contract_sha256']
        manifest['candidate'] = _write_json(candidate_path, candidate)
    else:
        manifest.pop('record_count')
    _write_json(request_path, manifest)
    row['source_request_sha256'] = _digest(request_path)
    _write_json(summary_path, summary)
    out = tmp_path / 'rejected'
    with pytest.raises(replay.OfflineQualityReplayError, match='producer-only replay requires'):
        replay.replay_evaluation_run(run, out_dir=out, row_indices=[0],
            producer_reference_statuses={0: status}, workers=1)
    assert not (out / replay.REPLAY_OUTPUT_NAME).exists()
    assert not (out / 'reports').exists()


def test_selection_contract_and_source_output_are_rejected(tmp_path):
    run, _ = _make_replay_run(tmp_path)
    with pytest.raises(replay.OfflineQualityReplayError, match='mutually exclusive'):
        replay.replay_evaluation_run(run, out_dir=tmp_path / 'rejected', row_indices=[0], full_only=True)
    with pytest.raises(replay.OfflineQualityReplayError, match='outside original run and source tree'):
        replay.replay_evaluation_run(run, out_dir=Path(replay.__file__).resolve().parents[1] / 'forbidden_replay',
                                    row_indices=[0])


def test_missing_later_candidate_never_starts_service(tmp_path, monkeypatch, forbidden_starts):
    run = _modern_run(tmp_path)
    summary = json.loads((run / 'quality_management/central_quality_summary.json').read_text())
    path = run / summary['results'][1]['source_request']
    candidate = path.parent / json.loads(path.read_text())['candidate']['path']
    candidate.unlink()  # Synthetic fixture only.
    monkeypatch.setattr(replay, 'ManagementQualityService', lambda *a, **k: pytest.fail('service started before input preflight'))
    with pytest.raises(replay.OfflineQualityReplayError):
        replay.replay_evaluation_run(run, out_dir=tmp_path / 'rejected', row_indices=[0, 1])
    assert not (tmp_path / 'rejected').exists()


def test_scientific_hash_excludes_execution_observations_but_keeps_uncertainty(tmp_path):
    base = {'decision': 'accuracy_loss', 'primary': {'candidate': .5, 'reference': 1., 'ci_low': -.6,
        'ci_high': -.4, 'bootstrap_elapsed_s': 12., 'bootstrap_engine': 'legacy'},
        'guardrails': {}, 'statistics_observation': {'worker_pid': 123, 'checkpoint_hit': False},
        'bootstrap_workers_requested': 4, 'bootstrap_workers_effective': 4, 'cache_hit': False}
    varied = copy.deepcopy(base)
    varied.update(cache_hit=True, bootstrap_workers_requested=8, bootstrap_workers_effective=2,
                  statistics_observation={'worker_pid': 789, 'checkpoint_hit': True})
    varied['primary'].update(bootstrap_elapsed_s=.1, bootstrap_engine='optimized_coco_v1')
    assert json_fingerprint(replay._stable_result(base)) == json_fingerprint(replay._stable_result(varied))
    varied['primary']['ci_low'] = -.7
    assert json_fingerprint(replay._stable_result(base)) != json_fingerprint(replay._stable_result(varied))


@pytest.mark.parametrize(('reference', 'candidate', 'decision', 'label'), [
    (1., 1., 'reference_close', 'Referenznah'), (1., .8, 'accuracy_loss', 'Genauigkeitsverlust'),
    (0., 0., 'not_estimable', 'Nicht einstufbar')])
def test_modern_decisions_and_nullable_ratio_survive_normal_reporting(tmp_path, reference, candidate, decision, label):
    assessment = assess_accuracy(reference, candidate, None)
    row = {'model_id': 'synthetic', 'source_row_index': 0, 'decision': decision, 'technical_status': 'completed',
           'accuracy_assessment': assessment, 'accuracy_warnings': [], 'primary': {}, 'guardrails': {}}
    paths = replay._write_scientific_projection(tmp_path, [row])
    assert replay._decision_summary([row]) == ({decision: 1}, decision)
    saved = json.loads(Path(paths['json']).read_text())[0]
    assert saved['accuracy_assessment'] == assessment
    assert saved['accuracy_relative_loss_ci'] is None
    assert label in _render(tmp_path)
    assert 'Accuracyreport nicht lesbar' not in _render(tmp_path)


def test_cli_forwards_selection_and_execution_flags(monkeypatch, tmp_path, capsys):
    path = Path(replay.__file__).resolve().parents[1] / 'scripts/replay_central_quality.py'
    spec = importlib.util.spec_from_file_location('quality_replay_cli_test', path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []
    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(request_count=1, technical_status='ok', quality_decision='accuracy_loss',
            decision_counts={'accuracy_loss': 1}, canonical_full_only_quality_decision='not_evaluated',
            canonical_full_only_decision_counts={}, selection_scope='selected_rows', selected_source_row_indices=[70],
            scientific_report_paths={})
    monkeypatch.setattr(cli, 'replay_evaluation_run', fake)
    assert cli.main(['--eval-run-dir', str(tmp_path), '--row-index', '70', '--profile', 'saved.yaml',
        '--engine', 'optimized_coco_v1', '--block-repetitions', '128', '--checkpoint-blocks',
        '--prepared-cache-limit-mib', '64', '--producer-reference-status', '70=status.json']) == 0
    options = calls[0][1]
    assert options['workers'] is None and options['row_indices'] == [70]
    assert options['statistics'] == dict(engine='optimized_coco_v1', block_repetitions=128,
                                       checkpoint_blocks=True, prepared_cache_limit_mib=64)
    assert options['producer_reference_statuses'] == {70: 'status.json'}
    assert 'hardware_executed=false' in capsys.readouterr().out
