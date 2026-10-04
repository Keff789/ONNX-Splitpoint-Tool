"""Original-source reentry and admission guards; no model/device inference."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import json
import shutil

import pytest

from onnx_splitpoint_tool.management_reference import _source_contract
from onnx_splitpoint_tool.workflow.artifacts import write_json
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.supplement import completed_quality
from tests.test_v2911_supplement_workflow import archive
from tests.test_v2911_supplement_reentry_admission import _local_quality


def _runner(tmp_path):
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile='', out=str(tmp_path)))
    runner.run_dir = tmp_path
    runner.run_id = tmp_path.name
    runner._emit_log = lambda *_: None
    return runner


def _suite(path):
    path.mkdir(parents=True)
    write_json(path / 'benchmark_plan.json', {'model_id': 'model', 'runs': []})
    write_json(path / 'benchmark_set.json', {'model_id': 'model'})
    (path / 'benchmark_suite.py').write_text('original source\n')
    return path


def _contract(suite):
    return _source_contract(suite, json.loads((suite / 'benchmark_plan.json').read_text()),
                            json.loads((suite / 'benchmark_set.json').read_text()))


def test_admission_rejects_source_changed_since_schedule_before_generator(tmp_path):
    runner = _runner(tmp_path)
    suite = _suite(tmp_path / 'suite')
    runner._management_reference_source_contracts['model'] = _contract(suite).removeprefix('sha256:')
    runner._ensure_central_quality_service = lambda: SimpleNamespace(
        pause_gate=SimpleNamespace(activity=lambda *a, **kw: nullcontext()))
    generator = Mock()
    (suite / 'benchmark_suite.py').write_text('refreshed after schedule\n')
    with pytest.raises(RuntimeError, match='source changed before admission.*expected=.*observed='):
        runner._run_budgeted_management_reference(generator, suite_dir=suite, model_id='model', workers=4)
    generator.assert_not_called()


@pytest.mark.parametrize('damage', ['', 'source', 'model', 'status'])
def test_restore_requires_exact_original_source_without_rebinding(tmp_path, damage):
    runner = _runner(tmp_path)
    root = tmp_path / 'quality_management/references/model'
    original = _suite(root / 'workspaces/original/suite')
    expected = _contract(original)
    current = tmp_path / 'models/model/benchmark_set/legacy_suite'
    shutil.copytree(original, current)
    (current / 'benchmark_suite.py').write_text('later source\n')
    status = {'status': 'completed', 'model_id': 'model', 'source_contract_sha256': expected}
    status_path = write_json(root / 'management_cpu_reference_status.json', status)
    before = status_path.read_bytes()
    if damage == 'source':
        (original / 'benchmark_suite.py').write_text('not original\n')
    elif damage == 'model':
        status['model_id'] = 'other'
        write_json(status_path, status)
    elif damage == 'status':
        status['source_contract_sha256'] = 'f' * 64
        write_json(status_path, status)
    if damage:
        with pytest.raises(RuntimeError):
            runner._restore_management_cpu_reference('model')
        assert runner._management_reference_futures == {}
    else:
        runner._restore_management_cpu_reference('model')
        first = runner._management_reference_futures['model']
        runner._restore_management_cpu_reference('model')
        assert runner._management_reference_futures['model'] is first
        assert first.result() == status
        assert runner._management_reference_source_contracts['model'] == expected.removeprefix('sha256:')
        assert status_path.read_bytes() == before


def test_attempt_result_discovers_original_request_without_overwriting_failed_sidecar(archive):
    local, row = _local_quality(archive, include_summary=False)
    completed = local.read_bytes()
    failed = dict(row, status='failed', technical_status='failed', decision='unavailable')
    write_json(local, failed)
    relative = local.relative_to(archive)
    suffix = relative.parts[relative.parts.index('quality_inputs') + 1:]
    attempt = archive / 'models' / row['model_id'] / 'benchmark_results/attempts/new/quality_inputs'
    target = attempt.joinpath(*suffix)
    target.parent.mkdir(parents=True)
    target.write_bytes(completed)
    kept, requests = completed_quality(archive)
    assert kept == [row]
    assert len(requests) == 1
    assert json.loads(local.read_text()) == failed
    assert str(archive / row['source_request']) in requests


def test_schedule_binds_contract_before_executor_can_start(tmp_path, monkeypatch):
    from concurrent.futures import Future
    from onnx_splitpoint_tool import management_reference
    runner = _runner(tmp_path)
    runner.profile_payload = {'quality_gate': {'statistics': {'execution_location': 'central_management'}}}
    suite = _suite(tmp_path / 'suite')
    runner._ensure_central_quality_service = lambda: SimpleNamespace(
        pause_gate=SimpleNamespace(activity=lambda *a, **kw: nullcontext()))
    calls = []
    def no_inference(**kwargs):
        calls.append(kwargs)
        return {'diagnostic_only': True}
    monkeypatch.setattr(management_reference, 'generate_management_cpu_reference', no_inference)
    class ImmediateExecutor:
        def submit(self, function, **kwargs):
            future = Future()
            future.set_result(function(**kwargs))
            return future
    runner._management_reference_executor = ImmediateExecutor()
    runner._schedule_management_cpu_reference('model', suite)
    assert len(calls) == 1
    assert runner._management_reference_futures['model'].result() == {'diagnostic_only': True}


def test_restored_workspace_keeps_original_dataset_path_without_symlinks(tmp_path):
    original = _suite(tmp_path / 'original')
    write_json(original / 'validation/manifest.json', {'images': ['one']})
    plan = {'model_id': 'model', 'runs': [{'validation_images': 'validation'}]}
    write_json(original / 'benchmark_plan.json', plan)
    expected = _contract(original)
    restored = tmp_path / 'restored'
    shutil.copytree(original, restored)
    assert _contract(restored) != expected
    contract = json.loads((restored / 'benchmark_set.json').read_text())
    assert _source_contract(restored, plan, contract, validation_source=original) == expected
    write_json(original / 'validation/manifest.json', {'images': ['different']})
    assert _source_contract(restored, plan, contract, validation_source=original) != expected
    assert not any(path.is_symlink() for path in restored.rglob('*'))


@pytest.mark.parametrize('refreshed', [False, True])
def test_refresh_binding_reaches_normal_remote_consumer_arguments(tmp_path, monkeypatch, refreshed):
    from onnx_splitpoint_tool.workflow import execution_binding
    from tests.test_v27539_remote_dispatch_propagation import _tree, _service_output
    suite, benchmark_set, results = _tree(tmp_path)
    local_run = tmp_path / 'download'
    local_run.mkdir()
    seen = []
    class NoRemoteWork:
        def run(self, **kwargs):
            seen.append(kwargs['args'].runtime_suite_finalized)
            return _service_output(local_run, dispatched=False)
    monkeypatch.setattr('onnx_splitpoint_tool.benchmark.services.RemoteBenchmarkService', NoRemoteWork)
    execution_binding._run_remote_dispatch_once(
        run_root=tmp_path, model_id='model', options=SimpleNamespace(), profile_payload={},
        suite_dir=suite, benchmark_set_json=benchmark_set, result_dir=results,
        gates={'suite_harness_refreshed': refreshed}, log=None,
        runtime_override={'enabled': True, 'host': '192.0.2.10', 'user': 'fixture'},
        target_id='setup',
    )
    assert seen == [refreshed]


def test_generator_rejects_post_admission_source_change_before_any_inference(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import management_reference
    suite = _suite(tmp_path / 'suite')
    expected = _contract(suite)
    (suite / 'benchmark_suite.py').write_text('changed after admission\n')
    launched = Mock(side_effect=AssertionError('no child permitted'))
    monkeypatch.setattr(management_reference.subprocess, 'Popen', launched)
    with pytest.raises(RuntimeError, match='source changed before generation'):
        management_reference.generate_management_cpu_reference(
            suite_dir=suite, output_dir=tmp_path / 'references', model_id='model',
            expected_source_contract_sha256=expected,
        )
    launched.assert_not_called()
    assert not (tmp_path / 'references/management_cpu_reference_status.json').exists()
