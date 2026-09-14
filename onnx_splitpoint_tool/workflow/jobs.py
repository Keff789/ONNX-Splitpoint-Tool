
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .artifacts import now_iso, relpath, sha256_payload, write_json, write_text

JOB_SCHEMA_VERSION = 1


def _safe(value: Any, default: str = "item") -> str:
    text = str(value or "").strip()
    if not text:
        return default
    out = []
    for ch in text.lower():
        if ch.isalnum() or ch in {"_", "-", ".", ":"}:
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("_")
    return token or default


def _parse_iso_ts(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _duration_between(start: Any, end: Any) -> Optional[float]:
    a = _parse_iso_ts(start)
    b = _parse_iso_ts(end)
    if a is None or b is None:
        return None
    try:
        return round(max(0.0, (b - a).total_seconds()), 6)
    except Exception:
        return None


class WorkflowJobQueueRecorder:
    """Small formal job-tree recorder for EvaluationWorkflowRunner.

    It intentionally does not replace the existing GUI/thread/remote services.
    It adds the missing contract around them: a parent EvaluationWorkflowJob,
    ModelJobs, StageJobs, stable job ids, job events, and machine-readable
    artifacts that can be shown in the GUI and uploaded for debugging.
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str,
        profile_id: str,
        workflow_version: str,
        tool_version: str,
        emit: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = str(run_id or "")
        self.profile_id = str(profile_id or "")
        self.workflow_version = str(workflow_version or "")
        self.tool_version = str(tool_version or "")
        self.emit = emit or (lambda _evt: None)
        self.jobs_dir = self.run_dir / "jobs"
        self.plan_path = self.jobs_dir / "job_plan.json"
        self.summary_path = self.jobs_dir / "job_summary.json"
        self.events_path = self.jobs_dir / "job_events.jsonl"
        self.timeline_path = self.jobs_dir / "job_timeline.md"
        self.resume_summary_path = self.jobs_dir / "resume_summary.json"
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []
        self.root_job_id = f"workflow:{_safe(self.run_id, 'run')}"
        self._plan_written = False
        self._existing_jobs: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(
        self,
        *,
        model_rows: Sequence[Mapping[str, Any]],
        root_stages: Sequence[str],
        model_stages: Sequence[str],
        final_stages: Sequence[str],
        barrier_stages: Sequence[str] = (),
        model_preparation_stages: Sequence[str] = (),
        model_execution_stages: Sequence[str] = (),
        resume: bool = False,
    ) -> Dict[str, Any]:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._existing_jobs = {}
        if resume and self.plan_path.is_file():
            try:
                payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
                if isinstance(payload, Mapping):
                    self._existing_jobs = {str(j.get("job_id")): dict(j) for j in list(payload.get("jobs") or []) if isinstance(j, Mapping) and str(j.get("job_id") or "")}
            except Exception:
                self._existing_jobs = {}
        # Fresh event log for a new run; resume appends later stage events.
        if not resume:
            try:
                self.events_path.write_text("", encoding="utf-8")
            except Exception:
                pass
        self.jobs = {}
        self._add_job(
            self.root_job_id,
            parent_id="",
            job_type="EvaluationWorkflowJob",
            model_id="",
            stage="",
            title=f"Evaluation workflow {self.profile_id or self.run_id}",
            queue_order=0,
            status="queued",
            expected_artifacts=["run_manifest.json", "artifact_index.json", "reports/summary.csv", "reports/prediction_vs_benchmark.csv"],
        )
        order = 1
        for stage in root_stages:
            self._add_job(
                self.stage_job_id(None, stage),
                parent_id=self.root_job_id,
                job_type=self._stage_job_type(stage),
                model_id="",
                stage=stage,
                title=stage,
                queue_order=order,
                status="queued",
                expected_artifacts=[f"stages/{stage}/stage_result.json"],
            )
            order += 1
        for row in model_rows:
            mid = _safe(row.get("id") or row.get("model_id") or row.get("name") or "model", "model")
            model_job = self.model_job_id(mid)
            self._add_job(
                model_job,
                parent_id=self.root_job_id,
                job_type="ModelJob",
                model_id=mid,
                stage="",
                title=f"Model {mid}",
                queue_order=order,
                status="queued",
                expected_artifacts=[f"models/{mid}/model_manifest.json", f"models/{mid}/benchmark_results/normalized_results.json"],
            )
            order += 1
            for stage in model_stages:
                self._add_job(
                    self.stage_job_id(mid, stage),
                    parent_id=model_job,
                    job_type=self._stage_job_type(stage, model_id=mid),
                    model_id=mid,
                    stage=stage,
                    title=f"{mid} / {stage}",
                    queue_order=order,
                    status="queued",
                    expected_artifacts=[f"models/{mid}/stages/{stage}/stage_result.json"],
                )
                order += 1
        for stage in barrier_stages:
            if stage in final_stages:
                continue
            self._add_job(
                self.stage_job_id(None, stage),
                parent_id=self.root_job_id,
                job_type=self._stage_job_type(stage),
                model_id="",
                stage=stage,
                title=stage,
                queue_order=order,
                status="queued",
                expected_artifacts=[f"stages/{stage}/stage_result.json"],
            )
            order += 1
        for stage in final_stages:
            expected = [f"stages/{stage}/stage_result.json"]
            if stage == "aggregate_results":
                expected.extend(["reports/summary.csv", "reports/prediction_vs_benchmark.csv"])
            if stage == "generate_report":
                expected.extend(["reports/thesis_section.md", "reports/results_bundle_manifest.json"])
            self._add_job(
                self.stage_job_id(None, stage),
                parent_id=self.root_job_id,
                job_type=self._stage_job_type(stage),
                model_id="",
                stage=stage,
                title=stage,
                queue_order=order,
                status="queued",
                expected_artifacts=expected,
            )
            order += 1

        # v2.79.20 executes model work around one workflow-wide cache barrier:
        # every BenchmarkSet is generated first, then cache readiness is
        # reported, and only afterwards may backend build/compiler and
        # runtime/hardware jobs start.  Preserve the long-standing tree shape,
        # but make queue_order describe that real execution order instead of
        # the older model-at-a-time schedule.  Callers which do not provide
        # phase lists retain the historical ordering above.
        barrier_names = [
            str(value) for value in barrier_stages if str(value).strip()
        ]
        preparation_names = [
            str(value)
            for value in model_preparation_stages
            if str(value).strip()
        ]
        execution_names = [
            str(value)
            for value in model_execution_stages
            if str(value).strip()
        ]
        if barrier_names and preparation_names and execution_names:
            ordered_ids: List[str] = [self.root_job_id]
            ordered_ids.extend(
                self.stage_job_id(None, stage) for stage in root_stages
            )
            model_ids = [
                _safe(
                    row.get("id")
                    or row.get("model_id")
                    or row.get("name")
                    or "model",
                    "model",
                )
                for row in model_rows
            ]
            for model_id in model_ids:
                ordered_ids.append(self.model_job_id(model_id))
                ordered_ids.extend(
                    self.stage_job_id(model_id, stage)
                    for stage in preparation_names
                )
            ordered_ids.extend(
                self.stage_job_id(None, stage) for stage in barrier_names
            )
            for model_id in model_ids:
                ordered_ids.extend(
                    self.stage_job_id(model_id, stage)
                    for stage in execution_names
                )
            ordered_ids.extend(
                self.stage_job_id(None, stage)
                for stage in final_stages
                if str(stage) not in set(barrier_names)
            )
            # Defensive compatibility: append any custom job unknown to the
            # phase contract without dropping it from the plan.
            ordered_ids.extend(
                job_id for job_id in self.jobs if job_id not in ordered_ids
            )
            for queue_order, job_id in enumerate(ordered_ids):
                if job_id in self.jobs:
                    self.jobs[job_id]["queue_order"] = queue_order
        # derive children after all jobs exist
        for job in self.jobs.values():
            job["child_job_ids"] = []
        for job_id, job in self.jobs.items():
            parent = str(job.get("parent_job_id") or "")
            if parent and parent in self.jobs:
                self.jobs[parent].setdefault("child_job_ids", []).append(job_id)
        for job in self.jobs.values():
            job["child_job_ids"] = sorted(job.get("child_job_ids") or [], key=lambda jid: int(self.jobs.get(jid, {}).get("queue_order") or 0))
        self._write_plan()
        self._emit_event("planned", self.root_job_id, status="queued", message="formal evaluation workflow job tree planned")
        for job in sorted(self.jobs.values(), key=lambda j: int(j.get("queue_order") or 0)):
            if job["job_id"] == self.root_job_id:
                continue
            self._emit_event("planned", job["job_id"], status="queued", message=f"queued {job.get('title')}")
        self._write_summary()
        self._plan_written = True
        return self.plan_payload()

    def _add_job(
        self,
        job_id: str,
        *,
        parent_id: str,
        job_type: str,
        model_id: str,
        stage: str,
        title: str,
        queue_order: int,
        status: str,
        expected_artifacts: Sequence[str],
    ) -> None:
        prev = dict(getattr(self, "_existing_jobs", {}).get(str(job_id), {}) or {})
        item = {
            "schema": "onnx-splitpoint/evaluation-job",
            "schema_version": JOB_SCHEMA_VERSION,
            "job_id": str(job_id),
            "parent_job_id": str(parent_id or ""),
            "job_type": str(job_type or "StageJob"),
            "model_id": str(model_id or ""),
            "stage": str(stage or ""),
            "title": str(title or job_id),
            "status": str(status or "queued"),
            "queue_order": int(queue_order),
            "created_at": now_iso(),
            "started_at": "",
            "finished_at": "",
            "duration_s": None,
            "stage_result_path": "",
            "expected_artifacts": list(expected_artifacts or []),
            "artifacts": [],
            "input_hash": "",
            "output_hash": "",
            "skip_reason": "",
            "error_class": "",
            "error_detail": "",
            "message": "",
            "last_activity_at": "",
            "activity": "",
            "activity_details": {},
            "idempotence": {
                "resume_checked": False,
                "existing_stage_result_reused": False,
                "profile_version_match": None,
                "artifact_completeness": "unknown",
            },
            "child_job_ids": [],
        }
        if prev:
            for key in ("created_at", "started_at", "finished_at", "duration_s", "stage_result_path", "artifacts", "input_hash", "output_hash", "skip_reason", "error_class", "error_detail", "message", "last_activity_at", "activity", "activity_details", "idempotence"):
                if key in prev and prev.get(key) not in (None, "", [], {}):
                    item[key] = prev.get(key)
            item["previous_status"] = prev.get("status", "")
            item["status"] = str(prev.get("status") or item["status"])
        self.jobs[str(job_id)] = item

    def _stage_job_type(self, stage: str, *, model_id: Optional[str] = None) -> str:
        name = str(stage or "").strip()
        mapping = {
            "resolve_profile": "ProfileResolutionJob",
            "resolve_model": "ModelResolutionJob",
            "check_validation_assets": "ValidationAssetCheckJob",
            "prepare_model": "ModelPreparationJob",
            "analyze_model": "AnalysisJob",
            "select_split_candidates": "CandidateSelectionJob",
            "prepare_full_baselines": "FullBaselinePreparationJob",
            "generate_benchmark_set": "BenchmarkSetGenerationJob",
            "build_backend_artifacts": "BackendArtifactBuildJob",
            "run_benchmarks": "BenchmarkExecutionJob",
            "validate_outputs": "ValidationJob",
            "hardware_smoke": "HardwareSmokeJob",
            "artifact_cache_preflight": "ArtifactCachePreflightJob",
            "evaluate_quality": "CentralQualityEvaluationJob",
            "aggregate_results": "ReportAggregationJob",
            "generate_report": "FinalReportJob",
        }
        if name == "prepare_full_baselines" and model_id and "yolo" in str(model_id).lower():
            return "FullHailoRawHeadBaselineJob"
        return mapping.get(name, "StageJob")

    def model_job_id(self, model_id: str) -> str:
        return f"model:{_safe(model_id, 'model')}"

    def stage_job_id(self, model_id: Optional[str], stage: str) -> str:
        st = _safe(stage, "stage")
        if model_id:
            return f"stage:{_safe(model_id, 'model')}:{st}"
        return f"stage:workflow:{st}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_workflow(self) -> None:
        self.mark(self.root_job_id, "running", message="workflow started")

    def finish_workflow(self, status: str, message: str = "") -> None:
        self.mark(self.root_job_id, self._job_status(status), message=message or f"workflow finished: {status}")
        self._write_summary()
        self._write_timeline()

    def request_cancel(self, reason: str = "user_requested") -> None:
        """Record an explicit cooperative cancel without losing completed jobs."""
        now = now_iso()
        for job_id, job in self.jobs.items():
            state = str(job.get("status") or "")
            if state == "queued":
                job["status"] = "cancelled"
                job["started_at"] = job.get("started_at") or now
                job["finished_at"] = now
                job["message"] = f"not started: cancellation requested ({reason})"
                self._emit_event("cancelled", job_id, status="cancelled", message=job["message"])
            elif state == "running":
                job["cancel_requested"] = True
                job["cancel_reason"] = str(reason or "user_requested")
                self._emit_event(
                    "cancel_requested", job_id, status="running",
                    message=f"cooperative cancellation requested ({reason})",
                )
        self._write_plan()
        self._write_summary()

    def start_model(self, model_id: str) -> None:
        self.mark(self.model_job_id(model_id), "running", message=f"model job started: {model_id}")

    def finish_model(self, model_id: str) -> None:
        mid = _safe(model_id, "model")
        model_job = self.model_job_id(mid)
        children = [self.jobs[c] for c in self.jobs.get(model_job, {}).get("child_job_ids", []) if c in self.jobs]
        status = self._aggregate_children_status(children)
        self.mark(model_job, status, message=f"model job finished: {mid} ({status})")

    def finish_unstarted_jobs(
        self,
        *,
        status: str,
        reason: str,
        model_id: Optional[str] = None,
        stages: Optional[Sequence[str]] = None,
        include_model_parent: bool = False,
    ) -> List[str]:
        """Terminalize jobs that cannot start after a barrier/stop/cancel.

        A terminal workflow must not retain misleading ``queued`` children.
        This helper changes only jobs which never started; completed or
        running work is left untouched.  Model parents are normally finalized
        through :meth:`finish_model`, and can be included explicitly for
        models that were never entered at all.
        """

        terminal_status = self._job_status(status)
        if terminal_status not in {"skipped", "cancelled", "failed"}:
            raise ValueError(
                "unstarted job status must be skipped, cancelled or failed"
            )
        wanted_model = _safe(model_id, "model") if model_id else ""
        wanted_stages = (
            {str(value) for value in stages}
            if stages is not None else None
        )
        changed: List[str] = []
        now = now_iso()
        for job_id, job in self.jobs.items():
            if job_id == self.root_job_id:
                continue
            if str(job.get("status") or "") != "queued":
                continue
            job_model = _safe(job.get("model_id"), "model") if job.get("model_id") else ""
            if model_id is not None and job_model != wanted_model:
                continue
            job_stage = str(job.get("stage") or "")
            if wanted_stages is not None and job_stage not in wanted_stages:
                continue
            if str(job.get("job_type") or "") == "ModelJob" and not include_model_parent:
                continue
            job["status"] = terminal_status
            job["started_at"] = job.get("started_at") or now
            job["finished_at"] = now
            job["duration_s"] = 0.0
            job["skip_reason"] = str(reason or "not_started")
            job["message"] = f"not started: {reason or terminal_status}"
            changed.append(job_id)
            self._emit_event(
                terminal_status,
                job_id,
                status=terminal_status,
                message=job["message"],
            )
        if changed:
            self._write_plan()
            self._write_summary()
        return changed

    def start_stage(self, model_id: Optional[str], stage: str, *, resume_checked: bool = False, forced: bool = False) -> str:
        job_id = self.stage_job_id(model_id, stage)
        job = self.jobs.get(job_id)
        if job is not None:
            idem = dict(job.get("idempotence") or {})
            idem["resume_checked"] = bool(resume_checked)
            idem["forced"] = bool(forced)
            job["idempotence"] = idem
        self.mark(job_id, "running", message=f"stage started: {model_id + ' / ' if model_id else ''}{stage}")
        return job_id

    def finish_stage(self, model_id: Optional[str], stage: str, stage_result: Mapping[str, Any], *, reused: bool = False) -> str:
        job_id = self.stage_job_id(model_id, stage)
        status = self._job_status(stage_result.get("status"))
        job = self.jobs.get(job_id)
        if job is not None:
            job["stage_result_path"] = str(stage_result.get("stage_result_path") or "")
            job["artifacts"] = list(stage_result.get("artifacts") or [])
            job["input_hash"] = str(stage_result.get("input_hash") or "")
            job["output_hash"] = str(stage_result.get("output_hash") or "")
            job["skip_reason"] = str(stage_result.get("skip_reason") or "")
            job["error_class"] = str(stage_result.get("error_class") or "")
            job["error_detail"] = str(stage_result.get("error_detail") or "")
            notes = list(stage_result.get("notes") or [])
            job["message"] = str(notes[0] if notes else stage_result.get("status") or "")
            idem = dict(job.get("idempotence") or {})
            idem["existing_stage_result_reused"] = bool(reused or stage_result.get("skip_reason") == "resume_reused_existing_stage_result")
            idem["artifact_completeness"] = "complete" if list(stage_result.get("artifacts") or []) else "unknown"
            job["idempotence"] = idem
        self.mark(job_id, status, message=f"stage finished: {model_id + ' / ' if model_id else ''}{stage} ({stage_result.get('status')})")
        return job_id

    def update_stage_activity(self, model_id: Optional[str], stage: str, message: str, *, details: Optional[Mapping[str, Any]] = None) -> None:
        job_id = self.stage_job_id(model_id, stage)
        job = self.jobs.get(str(job_id))
        if job is None:
            return
        if str(job.get("status") or "") not in {"running", "queued"}:
            return
        now = now_iso()
        if str(job.get("status") or "") == "queued":
            job["status"] = "running"
            job["started_at"] = job.get("started_at") or now
        job["last_activity_at"] = now
        job["activity"] = str(message or "")
        if message:
            job["message"] = str(message)
        detail_payload = dict(details or {})
        if detail_payload:
            job["activity_details"] = detail_payload
            if detail_payload.get("subtask"):
                job["progress_hint"] = str(detail_payload.get("subtask"))
        self._emit_event("log", job_id, status="running", message=str(message or ""), details=detail_payload)
        self._write_plan()
        self._write_summary()

    def mark(self, job_id: str, status: str, *, message: str = "") -> None:
        job = self.jobs.get(str(job_id))
        if job is None:
            return
        now = now_iso()
        old = str(job.get("status") or "queued")
        status = self._job_status(status)
        if old == "queued" and status == "running":
            job["started_at"] = now
        if status not in {"queued", "running"}:
            if not job.get("started_at"):
                job["started_at"] = now
            job["finished_at"] = now
            dur = _duration_between(job.get("started_at"), job.get("finished_at"))
            if dur is not None:
                job["duration_s"] = dur
        job["status"] = status
        if message:
            job["message"] = str(message)
        self._emit_event("status", str(job_id), status=status, message=message)
        self._write_plan()

    def _job_status(self, status: Any) -> str:
        s = str(status or "queued").strip().lower()
        if s in {"ok", "success", "measured", "validated", "hardware_verified"}:
            return "ok"
        if s in {"partial", "warn", "warning", "queued_missing_builds_for_configured_service"}:
            return "partial" if s == "partial" else "warn"
        if s in {"failed", "error"}:
            return "failed"
        if s in {"cancelled", "canceled"}:
            return "cancelled"
        if s in {"skipped", "resume_reused_existing_stage_result"}:
            return "skipped"
        if s in {"running"}:
            return "running"
        return s if s in {"queued", "planned"} else "warn"

    def _aggregate_children_status(self, children: Sequence[Mapping[str, Any]]) -> str:
        statuses = [self._job_status(c.get("status")) for c in children]
        if any(s == "failed" for s in statuses):
            return "failed"
        if any(s == "cancelled" for s in statuses):
            return "cancelled"
        if any(s == "partial" for s in statuses):
            return "partial"
        if any(s == "running" for s in statuses):
            return "running"
        if any(s == "warn" for s in statuses):
            return "warn"
        if statuses and all(s in {"ok", "skipped"} for s in statuses):
            control_skip_reasons = {
                "artifact_cache_preflight_blocked",
                "workflow_stop_before_job_start",
                "workflow_completed_before_job_start",
            }
            control_skips = [
                child for child in children
                if self._job_status(child.get("status")) == "skipped"
                and str(child.get("skip_reason") or "")
                in control_skip_reasons
            ]
            if control_skips:
                return (
                    "skipped"
                    if all(status == "skipped" for status in statuses)
                    else "partial"
                )
            return "ok"
        return "queued"

    # ------------------------------------------------------------------
    # Persistence / events
    # ------------------------------------------------------------------

    def plan_payload(self) -> Dict[str, Any]:
        jobs = sorted(self.jobs.values(), key=lambda j: int(j.get("queue_order") or 0))
        return {
            "schema": "onnx-splitpoint/evaluation-job-plan",
            "schema_version": JOB_SCHEMA_VERSION,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "tool_version": self.tool_version,
            "workflow_version": self.workflow_version,
            "created_at": now_iso(),
            "root_job_id": self.root_job_id,
            "job_count": len(jobs),
            "jobs": jobs,
        }

    def _write_plan(self) -> None:
        try:
            write_json(self.plan_path, self.plan_payload())
        except Exception:
            pass

    def _write_summary(self) -> None:
        counts: Dict[str, int] = {}
        for job in self.jobs.values():
            st = str(job.get("status") or "unknown")
            counts[st] = counts.get(st, 0) + 1
        running = [j for j in self.jobs.values() if str(j.get("status") or "") == "running"]
        blocking = [j for j in self.jobs.values() if str(j.get("status") or "") in {"failed", "partial"}]
        warnings = [j for j in self.jobs.values() if str(j.get("status") or "") == "warn"]
        by_type: Dict[str, int] = {}
        for job in self.jobs.values():
            typ = str(job.get("job_type") or "unknown")
            by_type[typ] = by_type.get(typ, 0) + 1
        payload = {
            "schema": "onnx-splitpoint/evaluation-job-summary",
            "schema_version": JOB_SCHEMA_VERSION,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "tool_version": self.tool_version,
            "workflow_version": self.workflow_version,
            "created_at": now_iso(),
            "job_count": len(self.jobs),
            "status_counts": counts,
            "jobs_by_type": by_type,
            "root_job_id": self.root_job_id,
            "running_job_ids": [j.get("job_id") for j in running],
            "active_jobs": running[:50],
            "blocking_job_ids": [j.get("job_id") for j in blocking[:100]],
            "blocking_jobs": blocking[:50],
            "warning_job_ids": [j.get("job_id") for j in warnings[:100]],
            "warning_jobs": warnings[:50],
        }
        try:
            write_json(self.summary_path, payload)
        except Exception:
            pass

    def _write_timeline(self) -> None:
        lines = [f"# Evaluation Job Timeline — {self.run_id}", "", f"Workflow: `{self.workflow_version}`", ""]
        lines.append("| Order | Status | Job | Type | Parent | Last activity | Stage result |")
        lines.append("|---:|---|---|---|---|---|---|")
        for job in sorted(self.jobs.values(), key=lambda j: int(j.get("queue_order") or 0)):
            lines.append(
                f"| {job.get('queue_order')} | `{job.get('status')}` | `{job.get('job_id')}` | {job.get('job_type')} | `{job.get('parent_job_id') or ''}` | {str(job.get('activity') or job.get('message') or '').replace('|', '/')[0:120]} | {job.get('stage_result_path') or ''} |"
            )
        try:
            write_text(self.timeline_path, "\n".join(lines) + "\n")
        except Exception:
            pass

    def _emit_event(self, event_type: str, job_id: str, *, status: str, message: str = "", details: Optional[Mapping[str, Any]] = None) -> None:
        job = dict(self.jobs.get(str(job_id)) or {})
        evt = {
            "schema": "onnx-splitpoint/evaluation-job-event",
            "schema_version": JOB_SCHEMA_VERSION,
            "time": now_iso(),
            "event": str(event_type),
            "job_id": str(job_id),
            "parent_job_id": str(job.get("parent_job_id") or ""),
            "job_type": str(job.get("job_type") or ""),
            "model_id": str(job.get("model_id") or ""),
            "stage": str(job.get("stage") or ""),
            "title": str(job.get("title") or job_id),
            "status": str(status or job.get("status") or ""),
            "queue_order": int(job.get("queue_order") or 0),
            "created_at": str(job.get("created_at") or ""),
            "started_at": str(job.get("started_at") or ""),
            "finished_at": str(job.get("finished_at") or ""),
            "duration_s": job.get("duration_s"),
            "last_activity_at": str(job.get("last_activity_at") or ""),
            "activity": str(job.get("activity") or ""),
            "message": str(message or ""),
            "run_dir": str(self.run_dir),
            "job_plan_path": str(self.plan_path),
            "job_events_path": str(self.events_path),
            "details": dict(details or {}),
        }
        self.events.append(evt)
        try:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(evt, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception:
            pass
        try:
            self.emit(evt)
        except Exception:
            pass

    def record_resume_decision(self, decision: Mapping[str, Any]) -> None:
        payload = dict(decision or {})
        job_id = str(payload.get("job_id") or "")
        job = self.jobs.get(job_id)
        if job is not None:
            idem = dict(job.get("idempotence") or {})
            idem.update({
                "resume_checked": True,
                "existing_stage_result_reused": bool(payload.get("reusable")),
                "artifact_completeness": "complete" if bool(payload.get("reusable")) else str(payload.get("reason") or "not_reused"),
                "resume_reason": str(payload.get("reason") or ""),
                "missing_artifacts": list(payload.get("missing_artifacts") or []),
                "hash_match": bool(payload.get("hash_match")),
            })
            job["idempotence"] = idem
        self._emit_event("resume_decision", job_id, status=("skipped" if bool(payload.get("reusable")) else "running"), message=f"resume: {payload.get('reason')}")
        self._write_summary()
        self._write_timeline()

    def write_resume_summary(self, decisions: Iterable[Mapping[str, Any]]) -> Path:
        items = [dict(x or {}) for x in list(decisions or [])]
        counts: Dict[str, int] = {}
        for item in items:
            key = "reused" if bool(item.get("reusable")) else str(item.get("reason") or "not_reused")
            counts[key] = counts.get(key, 0) + 1
        payload = {
            "schema": "onnx-splitpoint/evaluation-resume-summary",
            "schema_version": JOB_SCHEMA_VERSION,
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "tool_version": self.tool_version,
            "workflow_version": self.workflow_version,
            "created_at": now_iso(),
            "decision_count": len(items),
            "counts": counts,
            "decisions": items,
        }
        return write_json(self.resume_summary_path, payload)

    def artifacts(self) -> List[Path]:
        return [p for p in (self.plan_path, self.summary_path, self.events_path, self.timeline_path, self.resume_summary_path) if p.is_file()]
