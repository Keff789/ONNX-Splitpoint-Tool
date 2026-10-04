"""Visible supplement progress and separate administrative timing."""
import json
from types import SimpleNamespace

from onnx_splitpoint_tool.workflow import jobs, runner, supplement


def test_cli_forwards_workflow_messages_immediately(tmp_path, monkeypatch, capsys):
    class LocalRunner:
        def __init__(self, options, log):
            self.log = log
            self._emit_log = log

        def _load_profile(self):
            self.log("[workflow] start archived checks")

        def _build_current_resume_contract(self):
            return {}

        def _validate_resume_target_paths(self):
            pass

        _validate_resume_profile_snapshot = _validate_resume_target_paths
        _validate_resume_contract = _validate_resume_target_paths

        def run(self):
            self.log("[workflow][resume] reused existing measurement")
            self.log("[workflow] failed local projection")
            return SimpleNamespace(status="failed", run_dir=str(tmp_path), completed=False)

    monkeypatch.setattr(runner, "EvaluationWorkflowRunner", LocalRunner)
    monkeypatch.setattr(supplement, "options_from_run", lambda path: object())
    monkeypatch.setattr(supplement, "build_supplement_plan", lambda path: {})
    assert supplement.main(["--run-dir", str(tmp_path)]) == 1
    output = capsys.readouterr().out
    assert "start archived checks" in output
    assert "reused existing measurement" in output
    assert "failed local projection" in output


def test_reused_job_retains_original_execution_interval(tmp_path, monkeypatch):
    recorder = jobs.WorkflowJobQueueRecorder(
        run_dir=tmp_path, run_id="run", profile_id="profile",
        workflow_version="v50", tool_version="2.91.1",
    )
    recorder.plan(model_rows=[{"id": "model"}], root_stages=[],
                  model_stages=["run_benchmarks"], final_stages=[])
    job_id = recorder.stage_job_id("model", "run_benchmarks")
    original = {"started_at": "2026-09-25T12:00:00+00:00",
                "finished_at": "2026-09-25T12:00:09+00:00", "duration_s": 9.0}
    recorder.jobs[job_id].update(original, status="ok")
    monkeypatch.setattr(jobs, "now_iso", lambda: "2026-09-29T12:00:00+00:00")
    recorder.start_stage("model", "run_benchmarks", resume_checked=True)
    monkeypatch.setattr(jobs, "now_iso", lambda: "2026-09-29T12:00:02+00:00")
    recorder.finish_stage("model", "run_benchmarks", {"status": "ok"}, reused=True)
    row = recorder.jobs[job_id]
    assert {key: row[key] for key in original} == original
    assert row["resume_attempt_duration_s"] == 2.0
    persisted = json.loads(recorder.plan_path.read_text())
    stored = next(row for row in persisted["jobs"] if row["job_id"] == job_id)
    assert stored["duration_s"] == 9.0
    assert stored["resume_attempt_duration_s"] == 2.0

    # Reentry repairs only the job display from preserved original evidence;
    # it must not retain a duration polluted by the earlier failed attempt.
    recorder.jobs[job_id].update(finished_at="2026-09-29T12:00:00+00:00", duration_s=345600)
    recorder.start_stage("model", "run_benchmarks", resume_checked=True)
    recorder.finish_stage("model", "run_benchmarks", {
        "status": "ok", **original, "details": {"duration_s": 9.0,
        "resume_decision": {"previous_measurement_provenance_preserved": True}},
    }, reused=True)
    assert {key: recorder.jobs[job_id][key] for key in original} == original


def test_reexecuted_job_uses_current_stage_start(tmp_path, monkeypatch):
    recorder = jobs.WorkflowJobQueueRecorder(
        run_dir=tmp_path, run_id="run", profile_id="profile",
        workflow_version="v50", tool_version="2.91.1",
    )
    recorder.plan(model_rows=[], root_stages=["artifact_cache_preflight"],
                  model_stages=[], final_stages=[])
    job_id = recorder.stage_job_id(None, "artifact_cache_preflight")
    recorder.jobs[job_id].update(started_at="2026-09-25T12:00:00+00:00", status="ok")
    monkeypatch.setattr(jobs, "now_iso", lambda: "2026-09-29T12:00:00+00:00")
    recorder.start_stage(None, "artifact_cache_preflight", resume_checked=True)
    monkeypatch.setattr(jobs, "now_iso", lambda: "2026-09-29T12:00:05+00:00")
    recorder.finish_stage(None, "artifact_cache_preflight", {
        "status": "ok", "started_at": "2026-09-29T12:00:01+00:00",
    })
    assert recorder.jobs[job_id]["duration_s"] == 4.0


def test_reused_stage_preserves_measurement_times_and_reports_check(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "package_build_snapshot", lambda: {})
    messages = []
    local = runner.EvaluationWorkflowRunner(runner.WorkflowOptions(
        profile="fixture", out=str(tmp_path), resume=True, resume_missing=True,
    ), log=messages.append)
    local.run_dir = tmp_path
    local.manifest = {}
    local.jobs = jobs.WorkflowJobQueueRecorder(
        run_dir=tmp_path, run_id="run", profile_id="profile",
        workflow_version="v50", tool_version="2.91.1",
    )
    local.jobs.plan(model_rows=[{"id": "model"}], root_stages=[],
                    model_stages=["run_benchmarks"], final_stages=[])
    registrations = []
    monkeypatch.setattr(local, "_register_artifacts", lambda *a, **kw: registrations.append(a))
    stage_path = tmp_path / "models/model/stages/run_benchmarks/stage_result.json"
    stage_path.parent.mkdir(parents=True)
    previous = {"status": "ok", "started_at": "2026-09-25T12:00:00+00:00",
                "finished_at": "2026-09-25T12:00:09+00:00",
                "artifacts": ["measurement.json"], "details": {"duration_s": 9.0}}
    stage_path.write_text(json.dumps(previous))
    monkeypatch.setattr(local, "_resume_reuse_decision", lambda **kw: {
        "reusable": True, "reason": "preserved_original",
    })
    result = local._run_stage("model", "run_benchmarks", lambda: (_ for _ in ()).throw(
        AssertionError("must not execute a reused measurement")))
    assert result.started_at == previous["started_at"]
    assert result.finished_at == previous["finished_at"]
    assert result.details["duration_s"] == 9.0
    assert result.details["reuse_check_duration_s"] >= 0
    assert result.details["reuse_check_started_at"] != previous["started_at"]
    assert any("[workflow][resume] reused model / run_benchmarks" in line for line in messages)
    assert json.loads(stage_path.read_text()) == previous
    assert registrations == []
    assert local.jobs.plan_path.is_file()
    assert local.jobs.events_path.is_file()
    monkeypatch.setattr(local, "_stage_run_benchmarks", lambda *a: (_ for _ in ()).throw(
        AssertionError("must not execute a reused measurement")))
    local._run_model({"id": "model"}, stage_names=["run_benchmarks"])
    assert registrations == []
