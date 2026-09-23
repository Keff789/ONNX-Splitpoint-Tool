"""Real stage/job/resume plumbing and real Tk profile persistence.

Only leaf work is synthetic. No resolver, scheduling, resource, status or
profile persistence decision is substituted; no hardware work is performed.
"""
from __future__ import annotations

import contextvars
import json
from pathlib import Path
import subprocess
import sys
import threading
import tkinter as tk

import jsonschema
import pytest
import yaml

from onnx_splitpoint_tool.process_control import (
    ProcessTreeRegistry, bind_process_registry, current_process_registry,
)
from onnx_splitpoint_tool.workflow.artifacts import write_json
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.runner import (
    EvaluationWorkflowRunner, WorkflowRunCancelledError,
)


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("model_1", "model_2")


def _runner(tmp_path, cls=EvaluationWorkflowRunner):
    events = []
    runner = cls(
        WorkflowOptions(profile="", out=str(tmp_path), no_remote=True),
        job_event=lambda event: events.append((threading.get_ident(), dict(event))),
    )
    runner.run_id = "controlled-stage-lifecycle"
    runner.run_dir = tmp_path / runner.run_id
    runner.run_dir.mkdir()
    runner.profile_id = "controlled-local-leaves"
    runner.profile_payload = {"id": runner.profile_id, "models": [{"id": mid} for mid in MODELS]}
    runner.manifest_path = runner.run_dir / "run_manifest.json"
    runner.artifact_index_path = runner.run_dir / "artifact_index.json"
    runner.run_log_path = runner.run_dir / "evaluation_workflow.log"
    runner.parent_log_path = tmp_path / "latest.log"
    runner._init_manifest_and_index()
    runner._init_job_queue(runner.profile_payload["models"])
    assert runner.jobs is not None, runner.warnings
    for model_id in MODELS:
        runner.jobs.start_model(model_id)
    return runner, events


def _stage_path(runner, model_id):
    return runner.run_dir / "models" / model_id / "stages/run_benchmarks/stage_result.json"


def _leaf(runner, model_id, seen):
    owner = runner._active_model_id, runner._active_stage
    seen.append((model_id, "prepare", owner))
    value = yield lambda: model_id + "-external-result"
    seen.append((model_id, "resume", (runner._active_model_id, runner._active_stage)))
    assert owner == (model_id, "run_benchmarks")
    assert value == model_id + "-external-result"
    artifact = write_json(runner.run_dir / "models" / model_id / "controlled_result.json",
                          {"model_id": model_id, "value": value, "diagnostic_only": True})
    runner.log("controlled-parent-marker:" + model_id)
    return {"controlled_result": artifact}, {"controlled_leaf": True}, "local leaf completed", "ok"


def _finish(runner, steps, model_id, *, value=None, error=None):
    with pytest.raises(StopIteration) as finished:
        runner._advance_stage(steps, model_id, "run_benchmarks", value=value, error=error)
    return finished.value.value


def test_two_suspended_stages_keep_locals_lifecycle_and_parent_event_ownership(tmp_path):
    runner, events = _runner(tmp_path)
    runner._active_model_id, runner._active_stage = "outer", "outer-stage"
    seen, prepared = [], {}
    for model_id in MODELS:
        steps = runner._run_stage_steps(model_id, "run_benchmarks",
            lambda mid=model_id: _leaf(runner, mid, seen))
        prepared[model_id] = steps, runner._advance_stage(steps, model_id, "run_benchmarks")
        running = json.loads(_stage_path(runner, model_id).read_text())
        assert running["state"] == "running" and running["complete"] is False
        assert (runner._active_model_id, runner._active_stage) == ("outer", "outer-stage")
    for model_id in reversed(MODELS):
        steps, dispatch = prepared[model_id]
        result = _finish(runner, steps, model_id, value=dispatch())
        assert result.model_id == model_id and result.status == "ok"
        persisted = json.loads(_stage_path(runner, model_id).read_text())
        assert persisted["state"] == "completed" and persisted["complete"] is True
        assert (runner._active_model_id, runner._active_stage) == ("outer", "outer-stage")
    assert [row[:2] for row in seen] == [
        ("model_1", "prepare"), ("model_2", "prepare"),
        ("model_2", "resume"), ("model_1", "resume"),
    ]
    tagged = [event for _, event in events if event["message"].startswith("controlled-parent-marker:")]
    assert len(tagged) == 2
    assert all(event["model_id"] == event["message"].split(":")[-1] for event in tagged)
    assert all(owner == threading.get_ident() for owner, _ in events)


def test_nested_stage_and_process_context_restore_on_every_advance(tmp_path):
    runner, _ = _runner(tmp_path)
    outer_registry, nested_registry = ProcessTreeRegistry(), ProcessTreeRegistry()
    trace = contextvars.ContextVar("ap08_test_trace", default="unbound")
    seen = []

    def nested():
        assert (runner._active_model_id, runner._active_stage) == ("model_2", "validate_outputs")
        assert current_process_registry() is outer_registry
        with bind_process_registry(nested_registry):
            assert current_process_registry() is nested_registry
        return {}, {"controlled_leaf": True}, "nested local leaf", "ok"

    def leaf():
        runner._run_stage("model_2", "validate_outputs", nested)
        seen.append((runner._active_model_id, runner._active_stage, current_process_registry()))
        captured = contextvars.copy_context()
        value = yield lambda: captured.run(lambda: (current_process_registry(), trace.get()))
        assert value == (outer_registry, "workflow-owner")
        seen.append((runner._active_model_id, runner._active_stage, current_process_registry()))
        return {}, {"controlled_leaf": True}, "outer local leaf", "ok"

    with bind_process_registry(outer_registry):
        token = trace.set("workflow-owner")
        try:
            steps = runner._run_stage_steps("model_1", "run_benchmarks", leaf)
            dispatch = runner._advance_stage(steps, "model_1", "run_benchmarks")
            assert runner._active_model_id is None and runner._active_stage == ""
            assert current_process_registry() is outer_registry
            assert _finish(runner, steps, "model_1", value=dispatch()).status == "ok"
        finally:
            trace.reset(token)
    assert seen == [("model_1", "run_benchmarks", outer_registry)] * 2
    assert current_process_registry() is None and trace.get() == "unbound"


@pytest.mark.parametrize("cancel", [False, True])
def test_dispatch_exception_becomes_real_terminal_stage_without_leaking_active_owner(tmp_path, cancel):
    runner, _ = _runner(tmp_path)
    steps = runner._run_stage_steps("model_1", "run_benchmarks", lambda: _leaf(runner, "model_1", []))
    runner._advance_stage(steps, "model_1", "run_benchmarks")
    error = WorkflowRunCancelledError("controlled cancel") if cancel else ValueError("controlled dispatch error")
    result = _finish(runner, steps, "model_1", error=error)
    expected = "cancelled" if cancel else "failed"
    assert result.status == expected
    state = json.loads(_stage_path(runner, "model_1").read_text())
    assert state["state"] == expected and state["complete"] is (not cancel)
    assert str(error) in state["error_detail"]
    assert runner._cancel_event.is_set() is cancel
    assert runner._active_model_id is None and runner._active_stage == ""


def test_completed_stage_resume_reuses_real_indexed_artifact_without_entering_leaf(tmp_path):
    runner, _ = _runner(tmp_path)
    result = runner._run_stage("model_1", "run_benchmarks", lambda: _leaf(runner, "model_1", []))
    assert result.status == "ok"
    artifact = runner.run_dir / "models/model_1/controlled_result.json"
    before = artifact.read_bytes()
    runner.options.resume = True

    def forbidden():
        pytest.fail("completed stage entered leaf during valid Resume")

    steps = runner._run_stage_steps("model_1", "run_benchmarks", forbidden)
    reused = _finish(runner, steps, "model_1")
    assert reused.skip_reason == "resume_reused_existing_stage_result"
    assert reused.details["resume_decision"]["reusable"] is True
    assert artifact.read_bytes() == before


def test_interrupted_suspended_stage_is_not_reused_on_resume(tmp_path):
    runner, _ = _runner(tmp_path)
    original = runner._run_stage_steps("model_1", "run_benchmarks", lambda: _leaf(runner, "model_1", []))
    runner._advance_stage(original, "model_1", "run_benchmarks")
    original.close()  # Simulated interpreter interruption leaves the real incomplete record.
    runner.options.resume = True
    seen = []
    steps = runner._run_stage_steps("model_1", "run_benchmarks", lambda: _leaf(runner, "model_1", seen))
    dispatch = runner._advance_stage(steps, "model_1", "run_benchmarks")
    assert seen[0][:2] == ("model_1", "prepare")
    decision = json.loads((_stage_path(runner, "model_1").parent / "resume_decision.json").read_text())
    assert decision["reusable"] is False
    assert decision["reason"] == "previous_stage_not_complete:running"
    assert _finish(runner, steps, "model_1", value=dispatch()).status == "ok"


class _ControlledSetupLeaves(EvaluationWorkflowRunner):
    """Replace only external leaves; use real parent queue and stage machinery."""

    def _stage_run_benchmarks_steps(self, model_id, row, *, dispatch_manager, dispatch_log, **_kwargs):
        def external_leaf():
            assert current_process_registry() is self._process_registry
            dispatch_log("controlled-worker-marker:" + model_id)
            return model_id

        def dispatch():
            future = dispatch_manager.submit_group(model_id, "dut:controlled-local", external_leaf)
            dispatch_manager.finish_registration(model_id)
            return future.result(timeout=5)

        value = yield dispatch
        assert value == model_id
        artifact = write_json(self.run_dir / "models" / model_id / "controlled_result.json",
                              {"model_id": model_id, "diagnostic_only": True})
        return {"controlled_result": artifact}, {"controlled_leaf": True}, "local controlled leaf", "ok"

    def _stage_validate_outputs(self, model_id, row):
        return {}, {"controlled_leaf": True}, "local validation leaf", "ok"

    _stage_hardware_smoke = _stage_validate_outputs


def test_actual_parent_queue_delivers_worker_activity_only_from_parent_with_correct_owner(tmp_path):
    runner, events = _runner(tmp_path, _ControlledSetupLeaves)
    with bind_process_registry(runner._process_registry):
        completed = runner._run_prepared_setup_queues(runner.profile_payload["models"])
    assert completed == set(MODELS)
    markers = [event for _, event in events if event["message"].startswith("controlled-worker-marker:")]
    assert {event["message"] for event in markers} == {"controlled-worker-marker:" + mid for mid in MODELS}
    assert all(event["model_id"] == event["message"].split(":")[-1] for event in markers)
    assert all(event["stage"] == "run_benchmarks" for event in markers)
    assert all(owner == threading.get_ident() for owner, _ in events)
    assert all(json.loads(_stage_path(runner, mid).read_text())["complete"] for mid in MODELS)


def test_real_tk_editor_save_reload_preserves_parallel_modes_and_legacy_defaults(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.gui import profile_editor as module
    # A stale forwarded X socket can block inside Tk(), before Python can
    # raise TclError. Probe only the display in an owned, bounded child.
    try:
        probe = subprocess.run(
            [sys.executable, "-B", "-c", "import tkinter as tk; r=tk.Tk(); r.destroy()"],
            capture_output=True, text=True, timeout=6,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("real Tk display connection did not answer within 6 s")
    if probe.returncode:
        pytest.skip("real Tk display unavailable: " + probe.stderr.strip())
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"real Tk display unavailable: {exc}")
    root.withdraw()
    source = ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml"
    schema = json.loads((ROOT / "onnx_splitpoint_tool/resources/schemas/evaluation_profile.schema.json").read_text())
    errors = []
    monkeypatch.setattr(module.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
    saved = tmp_path / "parallel_modes.yaml"
    monkeypatch.setattr(module.filedialog, "asksaveasfilename", lambda **kwargs: str(saved))
    editor = None
    try:
        editor = module.EvaluationProfileEditor(root, profile_var=tk.StringVar(root, str(source)))
        editor.withdraw()
        assert editor._load_profile(str(source)), errors
        assert editor.var_native_release_mode.get() == "global_barrier"
        assert editor.var_setup_queue_mode.get() == "model_barrier"
        for native, setup in (("per_case", "per_setup"), ("global_barrier", "model_barrier")):
            editor.var_native_release_mode.set(native)
            editor.var_setup_queue_mode.set(setup)
            root.update()
            editor._save(use_after=False)
            assert saved.is_file() and not errors, errors
            payload = yaml.safe_load(saved.read_text())
            assert payload["workflow_execution"] == {"native_release_mode": native, "setup_queue_mode": setup}
            jsonschema.validate(payload, schema)
            assert editor._load_profile(str(saved)), errors
            assert editor.var_native_release_mode.get() == native
            assert editor.var_setup_queue_mode.get() == setup
    finally:
        if editor is not None:
            editor.destroy()
        root.destroy()


@pytest.mark.parametrize("value", [{"native_release_mode": "unknown"}, {"setup_queue_mode": "parallel_anywhere"}])
def test_workflow_execution_schema_rejects_unknown_modes(value):
    schema = json.loads((ROOT / "onnx_splitpoint_tool/resources/schemas/evaluation_profile.schema.json").read_text())
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, schema["properties"]["workflow_execution"])
