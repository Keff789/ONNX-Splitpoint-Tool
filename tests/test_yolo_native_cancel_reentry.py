"""Normal owned Run entry must publish its own control state before Native.

No Native/Quality/SSH process runs. The unmodified nested Native cancellation
consumer is compiled directly from the actual coordinator source. The full
coordinator lifecycle is covered by the existing AP07 regression alongside it.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
from zipfile import ZipFile

import pytest

from onnx_splitpoint_tool.workflow import runner as runner_module
from onnx_splitpoint_tool.workflow import run_control
from tests import test_p01_runner_gui_guards as guards

ROOT = Path(__file__).resolve().parents[1]
REPAIR = (Path(os.environ.get('THESIS20_FIXTURE_HOME', Path.home())) / 'Reports/thesis20-yolo-min3-augmentation-20261002_222904/repairs/native-cancel-20261003_134048')


def _native_consumer(run):
    """Load the actual nested consumer without rewriting its body or globals."""
    from tests.test_v269f_variant_native_split_quality_first import _load_script
    module = _load_script('run_evalrun_native_producer_variants.py')
    path = ROOT / 'scripts/run_evalrun_native_producer_variants.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'release_cancelled']
    assert len(nodes) == 1
    namespace = dict(vars(module), run_dir=run)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['release_cancelled']


def _runner(tmp_path, monkeypatch, *, resume=False):
    monkeypatch.setattr(run_control, 'platform_workflow_interlock_path', lambda: tmp_path/'isolated-locks/platform.lock')
    runner = guards.RunnerOwnershipGuardTests._runner(tmp_path)
    runner.options.resume = resume
    # Same minimal exact-contract fixture as the existing R1 entry tests: no
    # model/profile snapshot exists; all ownership/option/admission checks run.
    runner._validate_resume_profile_snapshot = lambda: None
    return runner


def _prior_cancel(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    def work():
        guards.RunnerOwnershipGuardTests._write_current_resume_fixture(runner, runner.run_dir)
        manifest_path = runner.run_dir/'run_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest.update(status='cancelled', options=runner.options.to_dict(),
                        execution_sessions=[{'session_id':runner.session_id,'status':'cancelled'}])
        manifest_path.write_text(json.dumps(manifest))
        (runner.run_dir/'retained_quality.json').write_text('{"count":12,"inference_repeated":false}\n')
        assert runner.request_cancel('historical_fixture_cancel') is True
        return SimpleNamespace(status='cancelled')
    runner._run_locked = work
    assert runner.run().status == 'cancelled'
    marker = runner.run_dir/'jobs/workflow_control.json'
    assert json.loads(marker.read_text())['cancel_requested'] is True
    return runner, marker, marker.read_bytes()


def _before_fix_run(runner):
    archive = REPAIR/'source-before/source.zip'
    with ZipFile(archive) as source:
        text = source.read('onnx_splitpoint_tool/workflow/runner.py').decode()
    cls = next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name=='EvaluationWorkflowRunner')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name=='run')
    namespace = dict(vars(runner_module))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(archive)+'!workflow/runner.py', 'exec'), namespace)
    runner.run = MethodType(namespace['run'], runner)


def test_normal_resume_replaces_historical_cancel_before_native_reads(tmp_path, monkeypatch):
    previous, marker, prior_bytes = _prior_cancel(tmp_path, monkeypatch)
    quality_path = previous.run_dir/'retained_quality.json'
    retained = quality_path.read_bytes()
    continued = _runner(tmp_path, monkeypatch, resume=True)
    if os.environ.get('YOLO_NATIVE_CANCEL_BEFORE_FIX') == '1':
        _before_fix_run(continued)
    seen = {}
    def work():
        assert continued._run_lock is not None
        assert continued._run_control_write_enabled is True
        control = json.loads(marker.read_text())
        blocked = _native_consumer(continued.run_dir)({'run_id':continued.run_id,'complete':True,
            'status':'completed','cancel_requested':False})
        seen.update(control=control, native_cancelled=blocked, old_session=previous.session_id,
                    current_session=continued.session_id, quality_preserved=quality_path.read_bytes()==retained)
        (tmp_path/'cancel-entry-evidence.json').write_text(json.dumps(seen,indent=2)+'\n')
        assert control['session_id'] == continued.session_id, seen
        assert blocked is False, seen
        return SimpleNamespace(status='partial')
    continued._run_locked = work
    assert continued.run().status == 'partial'
    assert seen and seen['quality_preserved']
    assert quality_path.read_bytes() == retained
    assert json.loads(prior_bytes)['session_id'] != seen['control']['session_id']


def test_current_owned_cancel_still_blocks_native_after_new_start(tmp_path, monkeypatch):
    previous, marker, _ = _prior_cancel(tmp_path, monkeypatch)
    continued = _runner(tmp_path, monkeypatch, resume=True)
    def work():
        consumer = _native_consumer(continued.run_dir)
        summary = {'status':'completed','complete':True,'cancel_requested':False}
        assert not consumer(summary)
        assert continued.request_cancel('current_fixture_cancel') is True
        control = json.loads(marker.read_text())
        assert control['session_id'] == continued.session_id != previous.session_id
        assert control['cancel_requested'] is True
        assert consumer(summary) is True
        return SimpleNamespace(status='cancelled')
    continued._run_locked = work
    assert continued.run().status == 'cancelled'


def test_cancelled_quality_summary_still_blocks_native(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)
    def work():
        consumer = _native_consumer(runner.run_dir)
        assert consumer({'status':'cancelled'}) is True
        assert consumer({'cancel_requested':True}) is True
        return SimpleNamespace(status='partial')
    runner._run_locked = work
    assert runner.run().status == 'partial'


def test_prestart_cancel_never_overwrites_historical_marker(tmp_path, monkeypatch):
    _, marker, before = _prior_cancel(tmp_path, monkeypatch)
    runner = _runner(tmp_path, monkeypatch, resume=True)
    runner.request_cancel('before_ownership')
    runner._run_locked = lambda: pytest.fail('cancelled run entered stage')
    with pytest.raises(run_control.WorkflowRunCancelledError):
        runner.run()
    assert marker.read_bytes() == before


def test_other_owner_never_overwrites_historical_marker(tmp_path, monkeypatch):
    previous, marker, before = _prior_cancel(tmp_path, monkeypatch)
    runner = _runner(tmp_path, monkeypatch, resume=True)
    runner._run_locked = lambda: pytest.fail('unowned run entered stage')
    with run_control.EvaluationRunLock(out_root=tmp_path,run_dir=previous.run_dir,owner={'session_id':'other'}):
        with pytest.raises(run_control.WorkflowRunLockedError):
            runner.run()
    assert marker.read_bytes() == before


def test_control_publication_failure_prevents_stage_entry(tmp_path, monkeypatch):
    _, marker, before = _prior_cancel(tmp_path, monkeypatch)
    runner = _runner(tmp_path, monkeypatch, resume=True)
    runner._run_locked = lambda: pytest.fail('failed control publication entered stage')
    original_replace = runner_module.os.replace
    def fail_control(source, target):
        if Path(target) == marker:
            raise PermissionError('controlled workflow marker write failure')
        return original_replace(source,target)
    monkeypatch.setattr(runner_module.os, 'replace', fail_control)
    with pytest.raises(PermissionError, match='controlled workflow marker write failure'):
        runner.run()
    assert marker.read_bytes() == before
