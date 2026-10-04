"""Explicit repair of a real terminal Quality nonstart; no hardware."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_yolo_augmentation_native_barrier import coordinator, _normal_main_fixture
from onnx_splitpoint_tool.workflow.checkpoints import canonical_json_sha256, load_stage_checkpoint, write_stage_checkpoint
from onnx_splitpoint_tool.workflow.supplement import preserve_metadata
from onnx_splitpoint_tool.workflow.yolo_augmentation_reentry import (
    COORDINATOR, PERFORMANCE, native_reentry_root, validate_augmentation_reentry,
)


@pytest.fixture
def quality_nonstart(coordinator, tmp_path, monkeypatch, capsys):
    run, cfg, config_path = _normal_main_fixture(tmp_path)
    plan_path = Path(cfg['augmentation_startcase_barrier']['plan_path'])
    plan = json.loads(plan_path.read_text())
    plan['augmentation_run'] = str(run)
    plan_path.write_text(json.dumps(plan))
    cfg['augmentation_startcase_barrier']['plan_sha256'] = canonical_json_sha256(plan)
    config_path.write_text(json.dumps(cfg))
    quality_path = Path(cfg['_workflow_context']['central_quality_summary'])
    quality = quality_path.read_bytes()
    empty_quality = json.loads(quality)
    empty_quality.update(complete=True, rows=[], results=[], status='skipped')
    quality_path.write_text(json.dumps(empty_quality))

    def no_child(*args, **kwargs):
        pytest.fail('no physical or report child may precede Quality release')

    monkeypatch.setattr(coordinator, '_run', no_child)
    args = ['coordinator', '--eval-run-dir', str(run), '--config', str(config_path)]
    monkeypatch.setattr(sys, 'argv', args)
    assert coordinator.main() == 2
    checkpoint = json.loads((run / COORDINATOR).read_text())
    assert checkpoint['state'] == 'failed' and checkpoint['complete'] is True
    assert 'no completed Native split quality binding' in checkpoint['error']
    assert checkpoint['error'] in capsys.readouterr().err
    return run, cfg, config_path, plan_path, quality_path, quality


def test_quality_nonstart_replays_error_without_starting_again(quality_nonstart, coordinator, monkeypatch, capsys):
    run, cfg, config_path, plan_path, _, _ = quality_nonstart
    before = (run / COORDINATOR).read_bytes()
    monkeypatch.setattr(sys, 'argv', ['coordinator', '--eval-run-dir', str(run), '--config', str(config_path), '--resume'])
    assert coordinator.main() == 2
    assert json.loads(before)['error'] in capsys.readouterr().err
    assert (run / COORDINATOR).read_bytes() == before
    assert validate_augmentation_reentry(run, plan_path)['predecessor_checkpoint_sha256'] == json.loads(before)['checkpoint_sha256']


def test_new_attempt_preserves_failed_checkpoint_and_completed_leaf_on_second_entry(
        quality_nonstart, coordinator, monkeypatch):
    run, cfg, _, plan_path, quality_path, quality = quality_nonstart
    original_files = {name: (run / name).read_bytes() for name in (
        COORDINATOR, PERFORMANCE, Path('reports/native_producer_stage.json'),
        Path('reports/native_producer_stage_config.json'))}
    validate_augmentation_reentry(run, plan_path)
    original = preserve_metadata(run, 'explicit-repair')
    descriptor = validate_augmentation_reentry(run, plan_path, original)
    cfg['augmentation_startcase_barrier']['reentry'] = descriptor
    attempt = native_reentry_root(run, cfg)
    attempt.mkdir()
    invocation = attempt / 'native_invocation_config.json'
    invocation.write_text(json.dumps(cfg))
    invocation_before = invocation.read_bytes()
    quality_path.write_bytes(quality)
    args = ['coordinator', '--eval-run-dir', str(run), '--config', str(invocation), '--resume']
    calls = []

    class StopAfterNative(BaseException):
        pass

    def first(command, *, label, **kwargs):
        calls.append(label)
        if label.startswith('variant:'):
            (run / 'native_producers/hailo8').mkdir(parents=True, exist_ok=True)
            return {'rc': 0, 'status': 'controlled_native_completion'}
        raise StopAfterNative(label)

    monkeypatch.setattr(coordinator, '_run', first)
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(StopAfterNative):
        coordinator.main()
    assert len([label for label in calls if label.startswith('variant:')]) == 1
    assert json.loads((attempt / 'native_coordinator/stage_result.json').read_text())['state'] == 'running'
    leaf = next((run / 'reports/native_producer_variants').glob('*/performance_stage_result.json'))
    leaf_before = leaf.read_bytes()
    reused_calls = []

    def second(command, *, label, **kwargs):
        reused_calls.append(label)
        assert not label.startswith('variant:'), 'completed Native leaf must not repeat'
        return {'rc': 2, 'stderr_tail': 'controlled missing measured outputs; do not assert a fake pair'}

    monkeypatch.setattr(coordinator, '_run', second)
    assert coordinator.main() == 2
    assert reused_calls == ['augmentation_startcase:final_report']
    assert leaf.read_bytes() == leaf_before and invocation.read_bytes() == invocation_before
    # Original failed attempt is preserved verbatim, not reset or upgraded.
    assert (run / COORDINATOR).read_bytes() == original_files[COORDINATOR]
    assert (run / PERFORMANCE).read_bytes() == original_files[PERFORMANCE]
    for name, value in original_files.items():
        assert (original / name).read_bytes() == value
    old_hash = json.loads(original_files[COORDINATOR])['input_hash']
    assert load_stage_checkpoint(original / COORDINATOR, stage='native_coordinator',
                                 input_hash=old_hash, run_root=original)[0] is not None
    # The current canonical stage alias now describes the new attempt. The
    # historical CP is validated only with its preserved original artifacts.
    assert load_stage_checkpoint(run / COORDINATOR, stage='native_coordinator',
                                 input_hash=old_hash, run_root=run)[0] is None
    new_terminal = json.loads((attempt / 'native_coordinator/stage_result.json').read_text())
    assert new_terminal['state'] == 'failed' and new_terminal['complete'] is True
    assert 'controlled missing measured outputs' in new_terminal['error']


@pytest.mark.parametrize('fault', ['checkpoint_digest', 'stage_artifact', 'config', 'plan', 'run',
    'native_leaf', 'native_raw', 'measured_summary', 'performance_digest', 'performance_complete', 'descriptor'])
def test_explicit_reentry_rejects_changed_history_or_started_native(quality_nonstart, fault):
    run, cfg, _, plan_path, _, _ = quality_nonstart
    if fault == 'checkpoint_digest':
        path = run / COORDINATOR; value = json.loads(path.read_text()); value['error'] += ' drift'; path.write_text(json.dumps(value))
    elif fault == 'stage_artifact':
        (run / 'reports/native_producer_stage.json').write_text('{}')
    elif fault == 'config':
        (run / 'reports/native_producer_stage_config.json').write_text('{}')
    elif fault == 'plan':
        value = json.loads(plan_path.read_text()); value['unplanned'] = True; plan_path.write_text(json.dumps(value))
    elif fault == 'run':
        value = json.loads((run / 'run_manifest.json').read_text()); value['run_id'] = 'other'; (run / 'run_manifest.json').write_text(json.dumps(value))
    elif fault == 'native_leaf':
        path = run / 'reports/native_producer_variants/leaf/performance_stage_result.json'; path.parent.mkdir(parents=True); path.write_text('{}')
    elif fault == 'native_raw':
        path = run / 'native_producers/leaf/yolo26s/benchmark_set/native_pipeline/b038/deepx_to_trt/fp16/deepx_native_fifo_e2e_results.json'
        path.parent.mkdir(parents=True); path.write_text('{"ok":false,"repetition_count_attempted":1}')
    elif fault == 'measured_summary':
        (run / 'reports/native_producer_summary.json').write_text('{"rows":[{"ok":false,"repetition_count_attempted":1}]}')
    elif fault == 'performance_digest':
        path = run / PERFORMANCE; value = json.loads(path.read_text()); value['details']['drift'] = True; path.write_text(json.dumps(value))
    elif fault == 'performance_complete':
        value = json.loads((run / PERFORMANCE).read_text())
        write_stage_checkpoint(run / PERFORMANCE, stage='native_performance', state='completed', complete=True,
                               input_hash=value['input_hash'], run_root=run)
    elif fault == 'descriptor':
        original = preserve_metadata(run, 'repair')
        descriptor = validate_augmentation_reentry(run, plan_path, original)
        descriptor['attempt_root'] = str(run / 'arbitrary-attempt')
        cfg['augmentation_startcase_barrier']['reentry'] = descriptor
        with pytest.raises(ValueError, match='descriptor_changed'):
            native_reentry_root(run, cfg)
        return
    with pytest.raises(ValueError, match='augmentation_reentry'):
        validate_augmentation_reentry(run, plan_path)


def test_metadata_only_native_directory_does_not_invent_started_measurements(quality_nonstart):
    run, _, _, plan_path, _, _ = quality_nonstart
    path = run / 'native_producers/metadata/run_profile.json'
    path.parent.mkdir(parents=True); path.write_text('{}')
    assert validate_augmentation_reentry(run, plan_path)['kind'] == 'augmentation_quality_release_nonstart'


def test_original_productive_failed_checkpoint_readonly():
    output = (Path(os.environ.get('THESIS20_FIXTURE_HOME', Path.home())) / 'Reports/thesis20-yolo-min3-augmentation-20261002_222904')
    run = output / 'runs/yolo_min3_20261002_222904'
    if not (run / COORDINATOR).is_file():
        pytest.skip('original productive local evidence unavailable')
    paths = [run / COORDINATOR, run / PERFORMANCE, run / 'reports/native_producer_stage.json',
             run / 'reports/native_producer_stage_config.json', output / 'plan/augmentation-plan.json']
    before = [p.read_bytes() for p in paths]
    descriptor = validate_augmentation_reentry(run, paths[-1])
    if descriptor['kind'] == 'augmentation_completed_native_first_generic':
        descriptor = descriptor['cancel_nonstart_ancestor']
    if descriptor['kind'] == 'augmentation_native_cancel_nonstart':
        descriptor = descriptor['quality_nonstart_ancestor']
    assert descriptor['predecessor_checkpoint_sha256'] == '07994705aaf772ee5733c9a896deccf526052e17c6965ab109676dfcc14996f4'
    assert "('yolo11l', 'b120', 'hailo8_to_trt')" in descriptor['predecessor_error']
    assert [p.read_bytes() for p in paths] == before


def test_normal_parent_handoff_starts_new_attempt_and_logs_actual_child_failure(
        quality_nonstart, coordinator, monkeypatch):
    from tests.test_p02_atomic_stage_energy_checkpoints import AtomicCheckpointTests
    from onnx_splitpoint_tool.workflow import runner as runner_module
    run, cfg, _, plan_path, quality_path, quality = quality_nonstart
    original = preserve_metadata(run, 'parent-repair')
    cfg, _ = coordinator._bind_evalrun_quality_gate_policy(run, cfg)
    cfg['enabled'] = True
    cfg['augmentation_startcase_barrier']['reentry'] = validate_augmentation_reentry(run, plan_path, original)
    quality_path.write_bytes(quality)
    (run / 'quality_management/case_quality_progress.json').write_bytes(quality)
    parent = AtomicCheckpointTests._outer_runner_fixture(run, cfg)
    parent.options.resume_missing = True
    parent.profile_payload = {'workflow_execution': {'native_release_mode': 'per_case'}}
    from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseJournal, RemoteProcessLeaseScope
    journal = RemoteProcessLeaseJournal(scope=RemoteProcessLeaseScope(run_id=run.name, session_id=parent.session_id),
                                       directory=run / 'jobs/fixture-remote-lease')
    parent._remote_process_registry.journal_environment = journal.environment
    logs = []
    parent.log = logs.append
    dispatched = []
    reason = 'controlled Native leaf diagnostic reached parent without hardware'

    def native_leaf(command, *, label, **kwargs):
        assert label.startswith('variant:')
        return {'rc': 2, 'stderr_tail': reason}

    def local_coordinator(command, **kwargs):
        dispatched.append(command)
        capture = io.StringIO()
        with monkeypatch.context() as local:
            local.setattr(sys, 'argv', [str(coordinator.__file__), *command[command.index('--eval-run-dir'):]])
            with contextlib.redirect_stderr(capture):
                rc = coordinator.main()
        stderr = capture.getvalue()
        for line in stderr.splitlines():
            kwargs['line_callback'](line)
        return SimpleNamespace(returncode=rc, stdout='', stderr=stderr, elapsed_s=0.0)

    monkeypatch.setattr(coordinator, '_run', native_leaf)
    monkeypatch.setattr(runner_module, 'run_streaming', local_coordinator)
    try:
        _, details, message, status = parent._stage_run_native_producers()
    finally:
        parent._shutdown_management_services()
    assert len(dispatched) == 1 and '--resume' in dispatched[0]
    new_cfg = Path(dispatched[0][dispatched[0].index('--config') + 1])
    assert new_cfg.parent == original.parent / 'native'
    assert details['coordinator_handoff_reused'] is False and status == 'failed'
    assert details['coordinator_handoff_reason'] == 'terminal'
    assert 'augmentation initial Native case failed' in details['failure_reason']
    assert reason in details['failure_reason']
    assert details['failure_reason'] in message
    assert (run / COORDINATOR).read_bytes() == (original / COORDINATOR).read_bytes()
    diagnostic = next((run / 'reports/diagnostics').glob('native_variant-coordinator_*.log'))
    assert details['failure_reason'] in diagnostic.read_text()
    assert any(details['failure_reason'] in line for line in logs)


def test_callback_attempt_output_is_an_explicit_argument(coordinator, tmp_path, monkeypatch):
    run, cfg, _ = _normal_main_fixture(tmp_path)
    for rel in ('reports/native_producer_combined_summary.json',
                'reports/native_validation/native_producer_validation_summary.json'):
        path = run / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
    output = tmp_path / 'augmentation/attempts/explicit-repair'
    cfg['augmentation_startcase_barrier']['execution_output'] = str(output)
    commands = []
    monkeypatch.setattr(coordinator, '_run', lambda command, **kwargs: commands.append(command) or {'rc': 0})
    coordinator._augmentation_completion_callback(run, cfg, case_key='yolo26s|b038|hailo8_setup')
    assert commands[0][-2:] == ['--attempt-dir', str(output)]


def test_three_real_quality_consumers_release_frozen_new_starters_after_historical_nonstart(
        coordinator, tmp_path, monkeypatch):
    """Normal N5000 writer/consumer -> all three release contracts -> dispatch.

    The outer Native process stops with a diagnostic. This test does not invent
    Native measurements, a successful output validator or a completed pair.
    """
    import hashlib
    from tests.test_yolo_augmentation_quality_release import make_runner, emit_request, STARTS, write
    from tests.test_v269f_variant_native_split_quality_first import _evalrun
    from onnx_splitpoint_tool.native_split_quality import known_native_split_policy

    runner = make_runner(tmp_path)
    run = runner.run_dir
    template = _evalrun(tmp_path / 'metadata-template')
    manifest = json.loads((template / 'run_manifest.json').read_text())
    manifest['run_id'] = run.name
    write(run / 'run_manifest.json', manifest)
    (run / 'profile.yaml').write_bytes((template / 'profile.yaml').read_bytes())
    write(run / 'reports/native_producer_stage.json', dict(
        json.loads((template / 'reports/native_producer_stage.json').read_text()), run_id=run.name))
    physical = {'hailo8_to_trt':'hailo8', 'hailo10h_to_trt':'hailo10h', 'deepx_to_trt':'deepx'}
    rows = []
    for model, case in [('yolo11l','b120'), ('yolo26s','b038'), ('yolov7_paper','b044'), ('yolov7_paper','b066')]:
        for _, _, setup, backend in STARTS:
            policy = known_native_split_policy(model_id=model, case_id=case, setup_id=setup, backend=backend)
            rows.append(dict(model_id=model, case_id=case, setup_id=setup, backend=physical[backend],
                             precision=policy['precision'], selection_role='primary'))
            write(run / 'models' / model / 'benchmark_set' / case / 'split_manifest.json', {})
            write(run / 'models' / model / 'benchmark_set/benchmark_set.json', {'benchmark_task':'detection'})
    starts = [next(row for row in rows if (row['model_id'],row['case_id'],row['setup_id']) == entry[:3])
              for entry in STARTS]
    plan = {'cohort_id':'yolo_min3_capability_augmentation', 'frozen':True,
            'augmentation_run':str(run), 'rows':rows}
    plan_path = tmp_path / 'frozen-plan.json'; write(plan_path, plan)
    reference = tmp_path / 'original-validation.json'; write(reference, {'rows':[]})
    progress = run / 'quality_management/case_quality_progress.json'
    cfg = {'native_release_mode':'per_case', 'backends':list(physical.values()),
        'split_backends':list(physical.values()), 'frames':1000, 'warmup':100, 'repetitions':3,
        'queue_depth':3, 'inflight':8, 'build_missing_engines':False,
        'energy':{'enabled':False}, 'validation':{'enabled':True}, 'full_baselines':{'enabled':False},
        'remotes':{physical[b]:{'ssh':'fixture@'+s, 'setup_id':s} for _,_,s,b in STARTS},
        'variants':[{'id':str(index), 'backends':[row['backend']], 'split_backends':[row['backend']],
            'precision':row['precision'], 'models':[row['model_id']], 'case_policy':'case_map_only',
            'case_map':{row['model_id']:[row['case_id']]}} for index,row in enumerate(rows)],
        '_workflow_context':{'central_quality_summary':str(progress),
            'workflow_execution':{'setup_queue_mode':'per_setup'}, 'execution_preset':{'id':'standard'}},
        'augmentation_startcase_barrier':{'plan_path':str(plan_path), 'plan_sha256':canonical_json_sha256(plan),
            'startcases':starts, 'original_reference_validation_summary':str(reference)}}
    config = run / 'reports/first-config.json'; write(config, cfg)
    write(progress, {'schema':'onnx-splitpoint/central-quality-summary', 'schema_version':1,
                     'results':[], 'complete':True, 'status':'skipped'})
    monkeypatch.setattr(coordinator, '_run', lambda *a, **k: pytest.fail('empty Quality released Native'))
    monkeypatch.setattr(sys, 'argv', ['coordinator','--eval-run-dir',str(run),'--config',str(config)])
    assert coordinator.main() == 2
    old_checkpoint = (run / COORDINATOR).read_bytes()
    assert json.loads(old_checkpoint)['complete'] is True
    original = preserve_metadata(run, 'three-start-repair')
    cfg['augmentation_startcase_barrier']['reentry'] = validate_augmentation_reentry(run, plan_path, original)
    requests, results, producers = {}, [], {}
    try:
        for row in runner.augmentation_plan['rows']:
            request, bound = emit_request(runner, row, tmp_path / 'new-producer')
            key = (row['model_id'],row['case_id'],row['setup_id'])
            requests[key], producers[key] = request, bound
            assert runner._queue_central_quality_requests(row['model_id']) == 1
            results.append(runner._central_quality_futures[str(request.resolve())].result(timeout=120))
        assert all(row['technical_status'] == 'completed' for row in results)
        summary_path = runner._publish_case_quality(results, complete=True)
        summary = json.loads(summary_path.read_text())
        leaves = coordinator._per_case_variants(run, cfg, cfg['variants'])
        order, selected_starts = coordinator._augmentation_dispatch_order(cfg, leaves, coordinator._augmentation_barrier_config(cfg))
        assert len(leaves) == 12 and len(selected_starts) == 3
        assert list(selected_starts.values()) == ['|'.join(entry[:3]) for entry in STARTS]
        for index in order[:3]:
            prepared, paths, _, _, _ = coordinator._prepare_case_release(run, cfg, leaves[index], summary)
            assert len(paths) == 1
            binding_path = next(iter(next(iter(paths.values())).values()))
            binding_set = json.loads(Path(binding_path).read_text())
            bound = next(iter(binding_set['bindings_by_model_case_backend'].values()))
            key = tuple(selected_starts[index].split('|'))
            assert bound['producer_binding_sha256'] == producers[key]['binding_sha256']
            assert bound['source_request_sha256'] == hashlib.sha256(requests[key].read_bytes()).hexdigest()
            command = coordinator._build_update_cmd(run, cfg, prepared, refresh_suites=False, timeout_s=7200)
            assert command[command.index('--frames')+1] == '1000'
            assert command[command.index('--warmup')+1] == '100'
            assert command[command.index('--repetitions')+1] == '3'
        invocation = native_reentry_root(run, cfg) / 'native_invocation_config.json'; write(invocation, cfg)
        calls = []
        def outer_native(command, *, label, **kwargs):
            assert label.startswith('variant:')
            calls.append(command)
            return {'rc':2, 'stderr_tail':'controlled outer Native stop; no measurements fabricated'}
        monkeypatch.setattr(coordinator, '_run', outer_native)
        monkeypatch.setattr(sys, 'argv', ['coordinator','--eval-run-dir',str(run),'--config',str(invocation),'--resume'])
        assert coordinator.main() == 2
        assert len(calls) == 1
        assert json.loads(calls[0][calls[0].index('--case-map')+1]) == {'yolo11l':['b120']}
        stage = json.loads((run / 'reports/native_producer_stage.json').read_text())
        assert len(stage['case_release_plan']['expected_rows']) == 12
        assert stage['native_energy']['status'] == 'not_started_startcase_barrier'
        assert (run / COORDINATOR).read_bytes() == old_checkpoint
        assert (original / COORDINATOR).read_bytes() == old_checkpoint
        assert json.loads(plan_path.read_text()) == plan
    finally:
        runner._shutdown_management_services()
