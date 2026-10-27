"""A real stopped finite action can recover without fake legacy phase files."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import copy

import pytest
import yaml

from test_physical_resource_recovery import recovery_case, _write, _fences, _retry_evidence
from onnx_splitpoint_tool.native_command_contract import canonical_json_sha256
from onnx_splitpoint_tool.remote import resource_recovery
from onnx_splitpoint_tool.workflow import finite_native_energy_recovery as finite
from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock


def descriptor(path):
    return {"path": str(path), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def finite_case(recovery_case):
    c = recovery_case
    journal = c.run_dir / "remote-process-leases"
    shutil.move(str(c.journal), journal)
    c.journal = journal
    for name in ("reply_path", "request_path", "config_path"):
        setattr(c, name, journal / getattr(c, name).name)
    (c.run_dir / "run_manifest.json").unlink()
    stamp = datetime.now(timezone.utc).isoformat()
    identity = ["hailo8_to_trt", "fixture_model", "b009", c.kwargs["setup_id"]]
    c.row_id = json.dumps(identity, separators=(",", ":"))
    fields = dict(zip(("backend", "model", "case", "setup_id"), identity))
    contract = {**fields, "schema": "onnx-splitpoint/native-command-contract"}
    digest = canonical_json_sha256(contract); contract["contract_sha256"] = digest
    output = c.run_dir / "reports/energy"
    contract_path = output / "plan/task.command_contract.json"; _write(contract_path, contract)
    row = {**fields, "successful_command_contract_sha256": digest,
           "measurement_output_base_dir": str(output), "command_contract_file": str(contract_path),
           "command_contract_file_sha256": descriptor(contract_path)["sha256"],
           "energy_quality_admission": {"admission_scope": "native_dataset_energy", "diagnostic_only": False}}
    plan = output / "plan/native_producer_energy_plan.json"; _write(plan, {"rows": [row]})
    offline = c.run_dir / "offline_plan/native_producer_energy_plan.json"; _write(offline, {"rows": [row]})
    source = c.run_dir / "source_export"; source.mkdir()
    freeze = c.run_dir / "source_freeze.json"; _write(freeze, {"source_export_root": str(source),
        "commit": "a" * 40, "remote_branch_head": "a" * 40, "pushed": True})
    argv = ["fixture-python", "-B", str(source / "scripts/run_native_producer_energy_from_summary.py"),
        "--duration-s", "60", "--runs", "3", "--frames", "0", "--warmup", "0",
        "--require-environment-observation", "--environment-run-id", c.run_dir.name,
        "--require-runtime-work-units", "--require-command-window-alignment",
        "--campaign-budget-file", str(c.run_dir / "energy_task_budget.json"),
        "--campaign-max-retries", "2", "--campaign-max-transport-failures", "20",
        "--out-dir", str(output)]
    _write(c.run_dir / "ACTION.json", {"schema": "six-pack-normal-native-energy-action-v1",
        "engine_build_allowed": False, "automatic_controller_restart": False, "runs": 3,
        "duration_s": 60, "output_root": str(c.run_dir), "source_export": str(source),
        "source_freeze": descriptor(freeze), "input_descriptors": [], "normal_offline_plan": descriptor(offline),
        "target": {"id": identity[3], "energy": {"urecs_address": "192.0.2.176"}},
        "normal_quality_admission": row["energy_quality_admission"], "resource_keys": list(c.locks), "argv": argv})
    _write(c.run_dir / "production-start.json", {"at": stamp, "parent_pid": 99999993,
        "session_id": c.kwargs["session_id"], "argv": argv, "engine_build_allowed": False,
        "actual_source_commit": "a" * 40, "common_resources": list(c.locks)})
    _write(c.run_dir / "production-result.json", {"finished_at": stamp, "returncode": 1})
    errors = ["stopped physical resources: " + repr(sorted(c.locks))]
    _write(c.run_dir / "CLEANUP_STATUS.json", {"finished_at": stamp, "cleanup_proven": False,
        "active_owned_remote_operations": 0, "owned_local_active": 0, "foreign_processes_stopped": False,
        "session_id": c.kwargs["session_id"], "errors": errors})
    c.capture = Path(c.request["attempt_dir"])
    attestation = c.capture / "preflight/preflight_attestation.json"
    _write(attestation, {"ok": True, "nonce": "real-test-nonce", "command_contract_sha256": digest})
    _write(c.capture / "preflight/preflight_evidence.json", {"ok": True, "rc": 0,
        "nonce": "real-test-nonce", "expected_command_contract_sha256": digest,
        "attestation_file_sha256": descriptor(attestation)["sha256"]})
    chain = {key: c.request[key] for key in ("owner_pid", "owner_start_ticks")}
    chain.update(run_directory=str(c.capture), logical_repeat="repeat:0", finished=True,
        collector_started=True, workload_started=True, valid=False, source_completion_verified=False,
        transport_reasons=["transport_log_error"])
    _write(c.run_dir / "energy_task_budget.json", {"campaign": True, "max_transport_failures": 20,
        "tasks": {c.row_id: {"source_id": "192.0.2.176", "limits": {
            "max_chains": 9, "max_transport_failures": 20, "max_retries": 2}, "chains": [chain],
            "stop_reason": "campaign_source_completion_unresolved"}},
        "sources": {"192.0.2.176": {"stop_reason": "campaign_source_completion_unresolved", "transport_failures": 1}}})
    c.kwargs.pop("operator_evidence")
    c.kwargs.update(bounded_energy_attempt_dir=c.run_dir, retry_authorization=_retry_evidence(c))
    c.replacement = c.run_dir / "recovery_attempts" / ("f" * 32)
    new_action = copy.deepcopy(_read(c.run_dir / "ACTION.json"))
    new_action.update(attempt_dir=str(c.replacement), failed_attempt=str(c.run_dir))
    new_action["argv"][new_action["argv"].index("--out-dir") + 1] = str(c.replacement / "reports/energy")
    _write(c.replacement / "ACTION.json", new_action)
    profile = Path(c.kwargs["retry_authorization"]["new_workflow_profile"])
    profile.write_text(yaml.safe_dump({"schema": "six-pack-native-energy-retry-v1",
        "run_dir": str(c.run_dir), "failed_attempt": str(c.run_dir), "historical_operation": c.request["operation_id"],
        "source": "192.0.2.176", "setup_id": c.kwargs["setup_id"],
        "same_original_budget": str(c.run_dir / "energy_task_budget.json"),
        "unchanged_limits": {"max_chains": 9, "max_transport_failures": 20, "max_retries": 2},
        "source_completion_remains_unproven": True, "action_file": str(c.replacement / "ACTION.json")}))
    c.run_lock = EvaluationRunLock(out_root=c.run_dir.parent, run_dir=c.run_dir,
        owner={"purpose": "six_pack_native_energy", "pid": 99999993, "acquired_at": stamp})
    c.run_lock.acquire()
    c.run_lock.commit_quarantine_fence({"reason": "six_pack_native_energy_cleanup_unproven", "pending": 0, "errors": errors})
    c.run_lock.release()
    return c


def test_exact_finite_recovery_preserves_raw_stop_and_consumed_budget(finite_case):
    c = finite_case
    originals = {p: p.read_bytes() for p in c.run_dir.rglob("*") if p.is_file() and p != c.reply_path}
    assert resource_recovery.recover_capture_resources(**c.kwargs)["status"] == "released"
    assert all(p.read_bytes() == raw for p, raw in originals.items())
    assert _read(c.reply_path)["state"] == "STOP"
    assert c.run_lock.quarantine_path.exists()  # Physical recovery never releases the Run.
    assert resource_recovery.recover_capture_resources(**c.kwargs)["status"] == "already_released"


def _read(path):
    return json.loads(path.read_text())


def test_live_parent_handoff_uses_original_budget_and_keeps_failed_chain(finite_case):
    c = finite_case
    before = _read(c.run_dir / "energy_task_budget.json")
    resource_recovery.recover_capture_resources(**c.kwargs)
    output = c.replacement
    lock = EvaluationRunLock(out_root=c.run_dir.parent, run_dir=c.run_dir,
        owner={"purpose": "six_pack_native_energy"})
    lock.acquire(recover_quarantine=lambda fence: finite.recover_finite_run_fence(
        c.run_dir, output, "new-explicit-session", lock, fence, c.run_dir))
    try:
        assert not lock.quarantine_path.exists()
        after = _read(c.run_dir / "energy_task_budget.json")
        assert after["tasks"] == before["tasks"]
        assert after["max_transport_failures"] == before["max_transport_failures"]
        source = after["sources"]["192.0.2.176"]
        assert source["stop_reason"] == before["sources"]["192.0.2.176"]["stop_reason"]
        assert source["transport_failures"] == 1
        grant = source["authorized_reentry"]
        assert grant["new_parent"]["session_id"] == "new-explicit-session"
        assert grant["chain"] == before["tasks"][c.row_id]["chains"][0]
        assert _read(output / "source-budget-handoff.json") == grant
        assert _read(c.reply_path)["state"] == "STOP"
    finally:
        lock.release()


@pytest.mark.parametrize("fault", ["active_owner", "unknown_operation", "wrong_identity", "live_cleanup", "wrong_source", "changed_contract"])
def test_finite_recovery_denies_unresolved_or_foreign_scope(finite_case, monkeypatch, fault):
    c = finite_case
    if fault == "active_owner":
        monkeypatch.setattr(resource_recovery, "observe_resource_owners", lambda *a, **k: {"ok": False})
    elif fault == "unknown_operation":
        _write(c.journal / "unknown.remote-lease.json", {"foreign": True})
    elif fault == "wrong_identity":
        value = _read(c.request_path); value["setup_id"] = "foreign"; _write(c.request_path, value)
    elif fault == "live_cleanup":
        path = c.run_dir / "CLEANUP_STATUS.json"; value = _read(path); value["owned_local_active"] = 1; _write(path, value)
    elif fault == "wrong_source":
        path = c.config_path; value = _read(path); value["setups"][c.kwargs["setup_id"]][-1] = "source:192.0.2.99"; _write(path, value)
    else:
        path = c.capture.parent / "plan/task.command_contract.json"; value = _read(path); value["case"] = "wrong"; _write(path, value)
    before = _fences(c)
    with pytest.raises(Exception):
        resource_recovery.recover_capture_resources(**c.kwargs)
    assert _fences(c) == before


def test_finite_run_handoff_rejects_foreign_run_fence(finite_case):
    c = finite_case
    resource_recovery.recover_capture_resources(**c.kwargs)
    fence = _read(c.run_lock.quarantine_path); fence["owner"]["reason"] = "foreign"
    _write(c.run_lock.quarantine_path, fence)
    before = (c.run_dir / "energy_task_budget.json").read_bytes()
    lock = EvaluationRunLock(out_root=c.run_dir.parent, run_dir=c.run_dir, owner={"purpose": "six_pack_native_energy"})
    with pytest.raises(ValueError, match="run fence"):
        lock.acquire(recover_quarantine=lambda current: finite.recover_finite_run_fence(
            c.run_dir, c.run_dir / "recovery_attempts" / ("e" * 32), "new-session", lock, current, c.run_dir))
    assert c.run_lock.quarantine_path.exists()
    assert (c.run_dir / "energy_task_budget.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["session", "argv", "profile", "active_parent", "unknown_resource"])
def test_live_handoff_denies_foreign_session_or_replacement(finite_case, monkeypatch, fault):
    c = finite_case
    resource_recovery.recover_capture_resources(**c.kwargs)
    if fault == "session":
        value = _read(c.run_lock.lock_path); value["session_id"] = "foreign-session"; _write(c.run_lock.lock_path, value)
    elif fault == "argv":
        path = c.replacement / "ACTION.json"; value = _read(path)
        value["argv"][value["argv"].index("--duration-s") + 1] = "1"; _write(path, value)
    elif fault == "profile":
        path = Path(c.kwargs["retry_authorization"]["new_workflow_profile"])
        value = yaml.safe_load(path.read_text()); value["same_original_budget"] = "/new/budget.json"
        path.write_text(yaml.safe_dump(value))
    elif fault == "active_parent":
        from onnx_splitpoint_tool import process_control
        original = process_control._proc_start_time
        monkeypatch.setattr(process_control, "_proc_start_time", lambda pid: "103" if pid == 99999993 else original(pid))
    else:
        lease = next(iter(c.locks.values())); value = _read(lease.lock_path)
        value["state"] = "running"; _write(lease.lock_path, value)
    before = (c.run_dir / "energy_task_budget.json").read_bytes()
    lock = EvaluationRunLock(out_root=c.run_dir.parent, run_dir=c.run_dir,
        owner={"purpose": "six_pack_native_energy", "session_id": "new-explicit-session"})
    with pytest.raises(ValueError):
        lock.acquire(recover_quarantine=lambda fence: finite.recover_finite_run_fence(
            c.run_dir, c.replacement, "new-explicit-session", lock, fence, c.run_dir))
    assert c.run_lock.quarantine_path.exists()
    assert (c.run_dir / "energy_task_budget.json").read_bytes() == before
