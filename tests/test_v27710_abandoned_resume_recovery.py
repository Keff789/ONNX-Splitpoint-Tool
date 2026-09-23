from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool.workflow.runner import (
    _ABANDONED_V2779_RESUME_WORKFLOW,
    EvaluationWorkflowRunner,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runner_with_abandoned_v2779(tmp_path: Path) -> EvaluationWorkflowRunner:
    run_dir = tmp_path / "phase5-run"
    run_dir.mkdir()
    abandoned_id = "a" * 32
    build = {"build_id": _ABANDONED_V2779_RESUME_WORKFLOW}
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run_dir.name,
        "profile_id": "phase5",
        "status": "running",
        "current_tool_version": "2.77.9",
        "current_workflow_version": _ABANDONED_V2779_RESUME_WORKFLOW,
        "current_tool_build": build,
        "current_session_id": abandoned_id,
        "resume_contract": {},
        "execution_sessions": [
            {
                "session_id": "b" * 32,
                "status": "failed",
                "workflow_version": "v2.77.7-old",
                "tool_version": "2.77.7",
            },
            {
                "session_id": abandoned_id,
                "status": "running",
                "resume_requested": True,
                "resumed_existing_manifest": True,
                "workflow_version": _ABANDONED_V2779_RESUME_WORKFLOW,
                "tool_version": "2.77.9",
                "tool_build": build,
            },
        ],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "artifact_index.json", {
        "schema": "onnx-splitpoint/artifact-index",
        "schema_version": 1,
        "run_id": run_dir.name,
        "profile_id": "phase5",
        "artifacts": [],
    })
    stage_dir = (
        run_dir / "models" / "mobilenet_v3_large" / "stages"
        / "prepare_full_baselines"
    )
    _write_json(stage_dir / "stage_result.json", {
        "status": "ok",
        "state": "completed",
        "complete": True,
        "details": {"resume_key_hash": "old"},
        "artifacts": ["placeholder"],
    })
    _write_json(stage_dir / "resume_decision.json", {
        "reason": "missing_full_quality_required_reuse_unavailable:status=ok",
        "previous_status": "ok",
        "artifacts_complete": False,
        "missing_artifacts": [
            "unbound_or_tampered:models/mobilenet_v3_large/"
            "full_baselines/output_contracts.json"
        ],
    })
    _write_json(run_dir / "reports" / "run_status_summary.json", {
        "schema": "onnx-splitpoint/run-status-summary",
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "failed",
    })
    run_hash = hashlib.sha256(run_dir.name.encode()).hexdigest()
    session_hash = hashlib.sha256(abandoned_id.encode()).hexdigest()
    _write_json(
        run_dir / "jobs" / "remote_process_leases" / abandoned_id
        / "session.cancelled.json",
        {
            "schema": "onnx-splitpoint/remote-process-lease-journal",
            "schema_version": 1,
            "run_sha256": run_hash,
            "session_sha256": session_hash,
            "cancelled": True,
        },
    )

    runner = object.__new__(EvaluationWorkflowRunner)
    runner.run_dir = run_dir
    runner.run_id = run_dir.name
    runner.manifest_path = run_dir / "run_manifest.json"
    runner.artifact_index_path = run_dir / "artifact_index.json"
    runner.options = SimpleNamespace(
        resume=True,
        resume_missing_full_quality_only=True,
    )
    runner._abandoned_v2779_resume_recovery = {}
    return runner


def test_exact_v2779_abandoned_session_is_recognised_read_only(
    tmp_path: Path,
) -> None:
    runner = _runner_with_abandoned_v2779(tmp_path)
    before = runner.manifest_path.read_bytes()

    result = runner._abandoned_v2779_resume_recovery_candidate()

    assert result["abandoned_session_id"] == "a" * 32
    assert result["recovered_status"] == "failed"
    assert runner.manifest_path.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_tool",
        "wrong_build",
        "wrong_decision",
        "missing_tombstone",
        "running_stage",
    ],
)
def test_near_match_abandoned_session_is_rejected(
    tmp_path: Path, mutation: str,
) -> None:
    runner = _runner_with_abandoned_v2779(tmp_path)
    manifest = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    stage_dir = (
        runner.run_dir / "models" / "mobilenet_v3_large" / "stages"
        / "prepare_full_baselines"
    )
    if mutation == "wrong_tool":
        manifest["current_tool_version"] = "2.77.8"
        _write_json(runner.manifest_path, manifest)
    elif mutation == "wrong_build":
        manifest["current_tool_build"]["build_id"] = "forged"
        _write_json(runner.manifest_path, manifest)
    elif mutation == "wrong_decision":
        decision = json.loads(
            (stage_dir / "resume_decision.json").read_text(encoding="utf-8")
        )
        decision["missing_artifacts"] = ["something_else"]
        _write_json(stage_dir / "resume_decision.json", decision)
    elif mutation == "missing_tombstone":
        (
            runner.run_dir / "jobs" / "remote_process_leases" / ("a" * 32)
            / "session.cancelled.json"
        ).unlink()
    else:
        stage = json.loads(
            (stage_dir / "stage_result.json").read_text(encoding="utf-8")
        )
        stage.update({"state": "running", "complete": False})
        _write_json(stage_dir / "stage_result.json", stage)

    assert runner._abandoned_v2779_resume_recovery_candidate() == {}


def test_manifest_initialisation_terminalises_only_the_attested_session(
    tmp_path: Path,
) -> None:
    runner = _runner_with_abandoned_v2779(tmp_path)
    recovery = runner._abandoned_v2779_resume_recovery_candidate()
    original = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    runner._abandoned_v2779_resume_recovery = recovery
    runner.session_id = "c" * 32
    runner.profile_payload = {}
    runner.profile_id = "phase5"
    runner.profile_path = "phase5.yaml"
    runner.profile_source = "profile"
    runner.profile_start_snapshot = {}
    runner._resume_contract = {}
    runner._tool_build_snapshot = {"build_id": "current"}
    runner._force_build_start_provenance = {}
    runner.manifest = {}
    runner.artifact_index = {}
    runner.outputs = {}
    runner.warnings = []
    runner.run_log_path = runner.run_dir / "evaluation_workflow.log"
    runner.parent_log_path = Path()
    runner._execution_session_index = None

    runner._init_manifest_and_index()

    updated = json.loads(runner.manifest_path.read_text(encoding="utf-8"))
    assert original["execution_sessions"][0]["status"] == "failed"
    assert updated["execution_sessions"][0]["status"] == "failed"
    assert updated["execution_sessions"][1]["status"] == "failed"
    assert updated["execution_sessions"][1]["abandoned_session_recovered"] is True
    assert updated["execution_sessions"][2]["session_id"] == "c" * 32
    assert len(updated["abandoned_execution_session_recoveries"]) == 1
