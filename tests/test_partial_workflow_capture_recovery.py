"""Recover a finished ordinary GUI acceptance without admitting live partials."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from onnx_splitpoint_tool.energy.task_budget import source_completion
from onnx_splitpoint_tool.remote import resource_recovery
from test_physical_resource_recovery import recovery_case, _write, _fences, _retry_evidence


@pytest.fixture
def partial_case(recovery_case):
    case = recovery_case
    now = datetime.now(timezone.utc)
    started = (now - timedelta(minutes=2)).isoformat()
    finished = (now - timedelta(seconds=1)).isoformat()
    schema = "onnx-splitpoint/hardware-acceptance/v1"
    manifest = {"schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": case.run_dir.name, "run_dir": str(case.run_dir), "status": "partial",
        "current_session_id": case.request["session_id"], "workflow_stages": ["hardware_acceptance"],
        "execution_sessions": [{"session_id": case.request["session_id"], "status": "partial",
            "started_at": started, "finished_at": finished}],
        "hardware_acceptance": {"schema": schema, "cell_count": 1, "runtime_path_count": 1}}
    _write(case.run_dir / "run_manifest.json", manifest)
    control = {"schema": "onnx-splitpoint/workflow-control", "run_id": case.run_dir.name,
        "session_id": case.request["session_id"], "state": "finished", "updated_at": now.isoformat()}
    _write(case.run_dir / "jobs/workflow_control.json", control)
    key = json.dumps(["generic_completed:hailo10h", "fixture-model", "b001", case.request["setup_id"]], separators=(",", ":"))
    result = {"schema": schema, "run_id": case.run_dir.name, "status": "partial", "complete": False,
        "matrix_cells": 1, "runtime_paths": 1, "source_campaign_resumed": False,
        "new_dataset_quality_started": False, "rows": [{"model_id": "fixture-model", "case_id": "b001",
            "setup_id": case.request["setup_id"], "runtime_variant": "generic_completed", "status": "failed",
            "fresh_hardware_executed": True, "new_capture_count": 1, "finished_at": finished}]}
    _write(case.run_dir / "reports/hardware_acceptance_results.json", result)
    budget = {"tasks": {key: {"source_id": "192.0.2.176", "stop_reason": case.reply["reason"],
        "chains": [{"run_directory": case.request["attempt_dir"], "owner_pid": case.request["owner_pid"],
            "owner_start_ticks": case.request["owner_start_ticks"], "collector_started": True,
            "finished": True, "valid": False, "source_completion_verified": False,
            "source_completion": source_completion(case.run_dir / "reports/energy/run_000")}] }},
        "sources": {"192.0.2.176": {"stop_reason": case.reply["reason"]}}}
    _write(case.run_dir / "energy/energy_task_budget.json", budget)
    _write(case.run_dir / "reports/energy/run_000/energy_summary.json", {"campaign_row": key,
        "campaign_run_id": case.run_dir.name, "collector_started": True,
        "collector_pid": case.request["collector_pid"], "status": "collector_failed",
        "final_energy_gate_reasons": ["capture_release_failed"],
        "capture_release_error": {"detail": case.reply["reason"]}})
    case.kwargs.pop("operator_evidence")
    case.kwargs["retry_authorization"] = _retry_evidence(case)
    for path in [case.run_dir / "run_manifest.json", case.run_dir / "jobs/workflow_control.json",
                 case.run_dir / "reports/hardware_acceptance_results.json",
                 case.run_dir / "energy/energy_task_budget.json",
                 case.run_dir / "reports/energy/run_000/energy_summary.json"]:
        case.preserved[path] = path.read_bytes()
    return case


def test_terminal_partial_recovery_preserves_old_results_and_source_stop(partial_case):
    case = partial_case
    result = resource_recovery.recover_capture_resources(**case.kwargs)
    assert result["status"] == "released"
    assert result["measurements_started"] == 0
    assert all(path.read_bytes() == value for path, value in case.preserved.items())
    reply = json.loads(case.reply_path.read_text())
    assert reply["state"] == "STOP" and reply["reason"] == case.reply["reason"]
    assert reply["recovery"]["old_attempt_validated"] is False
    assert reply["recovery"]["terminal_workflow"]["status"] == "partial"
    assert all(not lock.quarantine_path.exists() for lock in case.locks.values())
    before = case.reply_path.read_bytes()
    assert resource_recovery.recover_capture_resources(**case.kwargs)["status"] == "already_released"
    assert case.reply_path.read_bytes() == before


@pytest.mark.parametrize("change", ["session_foreign", "session_running", "session_unfinished", "control_running",
    "control_foreign", "result_unfinished", "result_foreign", "capture_identity", "capture_not_started",
    "budget_foreign_source", "budget_valid", "source_completion_changed"])
def test_partial_or_foreign_workflow_cannot_release(partial_case, change):
    case = partial_case
    if change.startswith("session_"):
        path = case.run_dir / "run_manifest.json"; value = json.loads(path.read_text())
        if change == "session_foreign": value["current_session_id"] = "foreign"
        elif change == "session_running": value["execution_sessions"][0]["status"] = "running"
        else: value["execution_sessions"][0].pop("finished_at")
    elif change.startswith("control_"):
        path = case.run_dir / "jobs/workflow_control.json"; value = json.loads(path.read_text())
        value["state" if change == "control_running" else "session_id"] = "running" if change == "control_running" else "foreign"
    elif change.startswith(("result_", "capture_")):
        path = case.run_dir / "reports/hardware_acceptance_results.json"; value = json.loads(path.read_text())
        if change == "result_unfinished": value.pop("status")
        elif change == "result_foreign": value["run_id"] = "foreign"
        elif change == "capture_identity": value["rows"][0]["case_id"] = "b002"
        else: value["rows"][0]["fresh_hardware_executed"] = False
    else:
        path = case.run_dir / "energy/energy_task_budget.json"; value = json.loads(path.read_text())
        task = next(iter(value["tasks"].values()))
        if change == "budget_foreign_source": task["source_id"] = "192.0.2.199"
        elif change == "budget_valid": task["chains"][0]["valid"] = True
        else: task["chains"][0]["source_completion"]["verified"] = True
    _write(path, value)
    fences = _fences(case)
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == fences
    assert json.loads(case.reply_path.read_text()) == case.reply


def test_partial_terminal_evidence_must_remain_unchanged_during_observation(partial_case, monkeypatch):
    case = partial_case
    def observe(*args, **kwargs):
        path = case.run_dir / "jobs/workflow_control.json"
        value = json.loads(path.read_text()); value["state"] = "running"; _write(path, value)
        return {"ok": True}
    monkeypatch.setattr(resource_recovery, "observe_resource_owners", observe)
    fences = _fences(case)
    with pytest.raises(Exception, match="ownership changed"):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == fences


def test_partial_source_completion_log_must_remain_bound_during_observation(partial_case, monkeypatch):
    case = partial_case
    def observe(*args, **kwargs):
        path = case.run_dir / "reports/energy/run_000/collector_stdout.log"
        path.write_text(path.read_text().replace("verified=false", "verified=true"))
        return {"ok": True}
    monkeypatch.setattr(resource_recovery, "observe_resource_owners", observe)
    fences = _fences(case)
    with pytest.raises(Exception, match="source completion changed"):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert _fences(case) == fences
    assert json.loads(case.reply_path.read_text()) == case.reply


@pytest.mark.parametrize("change", ["source_log", "control", "recorded_binding"])
def test_partial_recovery_replay_requires_same_terminal_proof(partial_case, change):
    case = partial_case
    resource_recovery.recover_capture_resources(**case.kwargs)
    if change == "source_log":
        path = case.run_dir / "reports/energy/run_000/collector_stdout.log"
        path.write_text(path.read_text().replace("verified=false", "verified=true"))
    elif change == "control":
        path = case.run_dir / "jobs/workflow_control.json"
        value = json.loads(path.read_text()); value["state"] = "running"; _write(path, value)
    else:
        value = json.loads(case.reply_path.read_text())
        value["recovery"]["terminal_workflow"]["session_id"] = "foreign"
        _write(case.reply_path, value)
    before = case.reply_path.read_bytes()
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**case.kwargs)
    assert case.reply_path.read_bytes() == before
