from __future__ import annotations

import json

from onnx_splitpoint_tool.workflow.jobs import WorkflowJobQueueRecorder


PREP = ("generate_benchmark_set",)
POST = ("build_backend_artifacts",)
EXEC = ("run_benchmarks", "validate_outputs", "hardware_smoke")
BARRIER = ("artifact_cache_preflight",)


def _recorder(tmp_path):
    recorder = WorkflowJobQueueRecorder(
        run_dir=tmp_path,
        run_id="run",
        profile_id="profile",
        workflow_version="v2.79.20-artifact-reuse-closure",
        tool_version="2.79.20",
    )
    recorder.plan(
        model_rows=[{"id": "model_a"}, {"id": "model_b"}],
        root_stages=["resolve_profile", "campaign_preflight"],
        model_stages=[*PREP, *POST, *EXEC],
        barrier_stages=BARRIER,
        model_preparation_stages=PREP,
        model_execution_stages=[*POST, *EXEC],
        final_stages=["evaluate_quality", "generate_report"],
    )
    return recorder


def test_job_plan_places_cache_preflight_between_prepare_and_runtime(tmp_path):
    recorder = _recorder(tmp_path)
    jobs = {
        row["job_id"]: row
        for row in json.loads(recorder.plan_path.read_text())["jobs"]
    }
    prep_orders = [
        jobs[recorder.stage_job_id(model, stage)]["queue_order"]
        for model in ("model_a", "model_b")
        for stage in PREP
    ]
    post_barrier_orders = [
        jobs[recorder.stage_job_id(model, stage)]["queue_order"]
        for model in ("model_a", "model_b")
        for stage in (*POST, *EXEC)
    ]
    barrier_order = jobs[
        recorder.stage_job_id(None, "artifact_cache_preflight")
    ]["queue_order"]
    assert max(prep_orders) < barrier_order < min(post_barrier_orders)
    for model in ("model_a", "model_b"):
        assert jobs[
            recorder.stage_job_id(model, "build_backend_artifacts")
        ]["queue_order"] < jobs[
            recorder.stage_job_id(model, "run_benchmarks")
        ]["queue_order"]
    assert jobs[
        recorder.stage_job_id(None, "artifact_cache_preflight")
    ]["job_type"] == "ArtifactCachePreflightJob"


def test_unstarted_jobs_become_terminal_without_touching_completed_work(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.mark(
        recorder.stage_job_id("model_a", "generate_benchmark_set"),
        "ok",
    )
    changed = recorder.finish_unstarted_jobs(
        status="skipped",
        reason="artifact_cache_preflight_blocked",
        stages=EXEC,
    )
    assert len(changed) == 6
    for model in ("model_a", "model_b"):
        for stage in EXEC:
            job = recorder.jobs[recorder.stage_job_id(model, stage)]
            assert job["status"] == "skipped"
            assert job["skip_reason"] == "artifact_cache_preflight_blocked"
    assert recorder.jobs[
        recorder.stage_job_id("model_a", "generate_benchmark_set")
    ]["status"] == "ok"


def test_terminal_workflow_has_no_queued_follow_on_jobs(tmp_path):
    recorder = _recorder(tmp_path)
    recorder.start_workflow()
    recorder.start_model("model_a")
    recorder.mark(
        recorder.stage_job_id("model_a", "generate_benchmark_set"), "ok",
    )

    changed = recorder.finish_unstarted_jobs(
        status="skipped",
        reason="artifact_cache_preflight_blocked",
        include_model_parent=True,
    )
    recorder.finish_model("model_a")
    recorder.finish_workflow("partial")

    assert changed
    assert not [
        job for job in recorder.jobs.values()
        if job["status"] == "queued"
    ]
    assert recorder.jobs[
        recorder.stage_job_id(None, "evaluate_quality")
    ]["skip_reason"] == "artifact_cache_preflight_blocked"
    assert recorder.jobs[recorder.model_job_id("model_a")]["status"] == (
        "partial"
    )
