"""Exact pre-dispatch Generic recovery with temporary locks and no hardware.

The normal product recovery, observation and durable commit paths are exercised.
The remote transport is doubled and unrelated host product names are isolated
at the shared fixture observation seam. Recorded PID/start checks and the full
original run/PGID/descendant scan use the bounded real logger/controller fixture.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.remote import process_lease, resource_recovery
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot, public_start_snapshot_metadata,
)
from tests.test_generic_completed_task_resources import workflow
from tests.test_generic_orphan_recovery import (
    _descriptor,
    _held_group,
    _normal_lock,
    _refresh_evidence_descriptors,
    _write,
    group_case,
    original_process_group,
    orphan_case,
)


SCHEMA = "onnx-splitpoint/generic-predispatch-run-only-recovery"
MARKER = "generic_predispatch_run_only_recovery"


@pytest.fixture
def predispatch_case(orphan_case):
    case = orphan_case
    physical = case.physical_locks[case.resource]
    # Replace only temporary fixture state through the normal acquire/release
    # lifecycle. The physical predecessor is another completed Generic run.
    physical.quarantine_path.unlink()
    physical.lock_path.unlink()
    predecessor_run = case.run.parent / "previous-normal-generic-quality"
    prior_session = "c" * 32
    prior_process = case.processes.identities["logger"]
    contract = json.loads(case.manifest_path.read_text())["resume_contract"]
    prior_owner = {"run_id": predecessor_run.name, "session_id": prior_session,
        "pid": prior_process["pid"], "ppid": prior_process["ppid"],
        "pgid": prior_process["pgrp"],
        "profile_sha256": contract["profile_sha256"], "plan_sha256": contract["plan_sha256"]}
    predecessor_run_lock = run_control.EvaluationRunLock(
        out_root=predecessor_run.parent, run_dir=predecessor_run, owner=prior_owner)
    predecessor_run_lock.acquire()
    prior_manifest_path = predecessor_run / "run_manifest.json"
    _write(prior_manifest_path, {
        "schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": predecessor_run.name, "run_dir": str(predecessor_run),
        "status": "completed", "technical_status": "completed", "current_session_id": prior_session,
        "resume_contract": contract,
        "execution_sessions": [{"session_id": prior_session, "status": "completed",
            "start_origin": {"pid": prior_process["pid"]},
            **{key: contract[key] for key in ("profile_sha256", "plan_sha256", "resume_contract_sha256")}}],
    })
    predecessor = run_control.EvaluationRunLock.for_resource(case.resource, owner={
        **{key: value for key, value in prior_owner.items() if key != "session_id"},
        "purpose": "generic"})
    predecessor.acquire()
    predecessor.release()
    predecessor_run_lock.release()
    assert json.loads(predecessor.lock_path.read_text())["state"] == "released"
    case.physical_locks = {case.resource: predecessor}
    case.physical_before = predecessor.lock_path.read_bytes()
    case.predecessor_run_lock = predecessor_run_lock
    case.predecessor_manifest_path = prior_manifest_path
    dispatch = (case.run / "models/fixture/benchmark_results"
                / "remote_benchmark_dispatch_fixture-deepx.json")
    dispatch.unlink()
    # No dispatch or remote lease has ever been created for this session.
    evidence = json.loads(case.evidence_path.read_text())
    evidence["schema"] = "onnx-splitpoint/generic-predispatch-run-only-recovery-input"
    evidence["released_physical_predecessor"] = {
        "owner": _descriptor(predecessor.lock_path),
        "run_manifest": _descriptor(prior_manifest_path),
        "run_lock": _descriptor(predecessor_run_lock.lock_path),
    }
    evidence["registry"] = _descriptor(case.registry_path)
    _write(case.evidence_path, evidence)
    authorization = json.loads(case.authorization_path.read_text())
    authorization.update(
        schema="onnx-splitpoint/generic-predispatch-run-only-recovery-authorization",
        action="recover_generic_predispatch_run_only",
        evidence=_descriptor(case.evidence_path),
    )
    _write(case.authorization_path, authorization)
    # Preserve explicit creation, quality, stage and prior-session facts.
    manifest = json.loads(case.manifest_path.read_text())
    manifest["created_at"] = "2026-10-20T00:00:00+00:00"
    manifest["models"] = {"fixture": {"stages": {"analyze_model": {"state": "running"}}}}
    # The original ordinary-recovery fixture has post-dispatch candidate bytes;
    # this run instead has only a central analysis diagnostic from preparation.
    original_result = case.result
    case.result = case.run / "models/fixture/analysis/diagnostic.json"
    case.result.parent.mkdir(parents=True)
    original_result.rename(case.result)
    original_result.parent.rmdir()
    original_result.parent.parent.rmdir()
    case.result.write_bytes(b'{"diagnostic":"original central analysis","state":"incomplete"}\n')
    case.result_bytes = case.result.read_bytes()
    case.preserved.pop(original_result)
    case.preserved[case.result] = case.result_bytes
    profile = yaml.safe_load((case.run / "profile.yaml").read_text())
    snapshot = public_start_snapshot_metadata(build_profile_start_snapshot(
        profile_request=str(case.processes.profile), profile_id="fixture-source-only",
        profile_path=str(case.processes.profile), profile_source="file",
        source_profile=profile, resolved_profile=profile,
        runtime_bindings={"hardware_targets": manifest["hardware_targets"]},
    ))
    _write(case.run / "profile_start_snapshot.json", snapshot)
    manifest["profile_start_snapshot"] = snapshot
    manifest["execution_sessions"].insert(0, {
        "session_id": "d" * 32, "status": "cancelled",
        "started_at": "2026-10-20T00:00:00+00:00",
        "finished_at": "2026-10-20T00:00:01+00:00",
    })
    (case.run / "jobs/remote_process_leases" / ("d" * 32)).mkdir()
    _write(case.manifest_path, manifest)
    case.original_manifest = case.manifest_path.read_bytes()
    case.manifest_before = case.original_manifest
    case.manifest_after = copy.deepcopy(manifest)
    case.manifest_after.update(status="cancelled", technical_status="cancelled")
    case.manifest_after["execution_sessions"][-1]["status"] = "cancelled"
    case.receipt_data.update(schema=SCHEMA, recovery_scope="predispatch_run_only")
    case.preserved.update({
        case.run / "jobs/workflow_control.json": (case.run / "jobs/workflow_control.json").read_bytes(),
        case.run / "profile.yaml": (case.run / "profile.yaml").read_bytes(),
        case.run / "profile_source.yaml": (case.run / "profile_source.yaml").read_bytes(),
        case.run / "effective_execution_plan.json": (case.run / "effective_execution_plan.json").read_bytes(),
        case.run / "profile_start_snapshot.json": (case.run / "profile_start_snapshot.json").read_bytes(),
        prior_manifest_path: prior_manifest_path.read_bytes(),
        predecessor_run_lock.lock_path: predecessor_run_lock.lock_path.read_bytes(),
    })
    return case


def _recover(case):
    return resource_recovery.recover_generic_predispatch_run_only(**case.kwargs)


def _commit(case, *, physical_locks=None):
    with _held_group(case):
        return run_control.commit_generic_predispatch_run_only_recovery(
            run_lock=case.run_lock,
            physical_locks=(case.physical_locks if physical_locks is None else physical_locks),
            receipt_path=case.receipt_path, receipt_data=case.receipt_data,
            manifest_path=case.manifest_path, manifest_before=case.manifest_before,
            manifest_after=case.manifest_after,
        )


def _assert_rejected_without_release(case):
    paths = [case.manifest_path, case.run_lock.lock_path,
             *(lock.lock_path for lock in case.physical_locks.values())]
    before = {path: path.read_bytes() for path in paths}
    fences = {lock.quarantine_path: lock.quarantine_path.read_bytes()
              for lock in [case.run_lock, *case.physical_locks.values()]
              if lock.quarantine_path.exists()}
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _recover(case)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert all(path.read_bytes() == data for path, data in fences.items())
    assert case.result.read_bytes() == case.result_bytes
    assert not case.receipt_path.exists()


def test_exact_ended_predispatch_run_only_recovery_preserves_other_physical_owner(predispatch_case):
    case = predispatch_case
    result = _recover(case)
    assert result["status"] == "committed"
    assert result["measurements_started"] == 0
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.physical_locks[case.resource].quarantine_path.exists()
    assert all(path.read_bytes() == data for path, data in case.preserved.items())
    before = json.loads(case.original_manifest)
    after = json.loads(case.manifest_path.read_text())
    assert after["status"] == after["technical_status"] == "cancelled"
    assert after["current_session_id"] == case.session
    assert after["execution_sessions"][-1]["status"] == "cancelled"
    assert after["execution_sessions"][:-1] == before["execution_sessions"][:-1]
    assert after["created_at"] == before["created_at"]
    assert after["models"] == before["models"]
    assert after["quality_decision"] == before["quality_decision"] == "not_evaluated"
    receipt = json.loads(case.receipt_path.read_text())
    assert receipt["schema"] == SCHEMA
    assert receipt["state"] == "committed"
    assert receipt["session_id"] == case.session
    assert receipt["original_completion_proven"] is False
    assert receipt["measurements_started"] == 0
    assert after[MARKER]["receipt_path"] == str(case.receipt_path)
    assert MARKER not in json.loads(case.physical_before)
    # The new durable run marker is also accepted by ordinary run admission.
    normal = _normal_lock(case)
    normal.acquire()
    normal.release(write_released_state=False)


def test_exact_committed_predispatch_replay_is_byte_stable_without_new_observations(predispatch_case, monkeypatch):
    case = predispatch_case
    _recover(case)
    paths = [case.receipt_path, case.manifest_path, case.run_lock.lock_path,
             case.physical_locks[case.resource].lock_path, *case.preserved]
    before = {path: path.read_bytes() for path in paths}

    def unexpected(*_args, **_kwargs):
        pytest.fail("a committed exact replay must not repeat remote or local observations")

    monkeypatch.setattr(resource_recovery, "_generic_orphan_local_observation", unexpected)
    monkeypatch.setattr(resource_recovery, "observe_resource_owners", unexpected)
    result = _recover(case)
    assert result["status"] == "already_committed"
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("group_case", ["active"], indirect=True)
def test_live_original_predispatch_controller_cannot_be_recovered(predispatch_case):
    _assert_rejected_without_release(predispatch_case)


@pytest.mark.parametrize("fault", ["request_session", "input_run", "manifest_session", "start_pid", "lineage", "authorization"])
def test_wrong_or_incomplete_original_predispatch_identity_blocks_release(predispatch_case, fault):
    case = predispatch_case
    if fault == "request_session":
        case.kwargs["session_id"] = "e" * 32
    elif fault == "input_run":
        value = json.loads(case.evidence_path.read_text())
        value["run_id"] = "another-run"
        _write(case.evidence_path, value)
        _refresh_evidence_descriptors(case)
    elif fault in {"manifest_session", "start_pid"}:
        value = json.loads(case.manifest_path.read_text())
        if fault == "manifest_session":
            value["current_session_id"] = "e" * 32
        else:
            value["execution_sessions"][-1]["start_origin"]["pid"] += 1
        _write(case.manifest_path, value)
    elif fault == "lineage":
        value = json.loads(case.lineage_path.read_text())
        value["workflow_owner"]["process_start_ticks"] += 1
        _write(case.lineage_path, value)
    else:
        value = json.loads(case.authorization_path.read_text())
        value["authorized"] = False
        _write(case.authorization_path, value)
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("fault", ["active", "unclear", "same_run", "quarantine", "recovery_marker"])
def test_physical_predecessor_must_be_other_normal_released_unquarantined_owner(predispatch_case, fault):
    case = predispatch_case
    lock = case.physical_locks[case.resource]
    owner = json.loads(lock.lock_path.read_text())
    if fault == "active":
        owner["state"] = "running"
    elif fault == "unclear":
        owner.pop("released_at")
    elif fault == "same_run":
        owner["run_id"] = case.run.name
    elif fault == "recovery_marker":
        owner["generic_orphan_recovery"] = {"receipt_path": "/missing/receipt.json"}
    else:
        _write(lock.quarantine_path, {"schema": "onnx-splitpoint/evaluation-run-lock-quarantine",
            "state": "quarantined", "owner": {"reason": "unresolved"}})
    _write(lock.lock_path, owner)
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("fault", ["dispatch", "foreign_dispatch", "lease", "foreign_journal", "control_evidence", "handoff_lease"])
def test_any_dispatch_or_remote_journal_evidence_blocks_predispatch_recovery(predispatch_case, fault):
    case = predispatch_case
    if fault in {"dispatch", "foreign_dispatch"}:
        model = "fixture" if fault == "dispatch" else "another-model"
        _write(case.run / "models" / model / "benchmark_results/remote_benchmark_dispatch_fixture-deepx.json",
               {"schema": "onnx-splitpoint/remote-benchmark-dispatch"})
    elif fault == "lease":
        _write(case.run / "jobs/remote_process_leases" / case.session / "started.remote-lease.json", {})
    elif fault == "foreign_journal":
        _write(case.run / "jobs/remote_process_leases" / ("e" * 32) / "started.remote-lease.json", {})
    elif fault == "control_evidence":
        _write(case.run / "jobs/remote_process_leases" / case.session / "operation.resource-request.json", {})
    else:
        value = json.loads(case.handoff_path.read_text())
        value["actions"][0]["lease_records"] = [{"operation_id": "earlier-operation", "value": {}}]
        _write(case.handoff_path, value)
        _refresh_evidence_descriptors(case)
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("fault", ["lease", "process", "visibility", "unknown_status"])
def test_remote_quiescence_must_be_complete_and_read_only(predispatch_case, fault):
    case = predispatch_case
    if fault == "lease":
        case.remote_observation.update(ok=False, status="busy", live_product_leases=[{"pid": 123}])
    elif fault == "process":
        case.remote_observation.update(ok=False, status="busy", processes=[{"pid": 123}])
    elif fault == "visibility":
        case.remote_observation.update(ok=False, visibility_errors=["process visibility denied"])
    else:
        case.remote_observation.update(ok=False, status="unobservable")
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("fault", ["source_profile", "resolved_snapshot", "archived_source", "registry_target", "plan"])
def test_original_source_snapshot_or_registry_mismatch_cannot_release(predispatch_case, fault):
    case = predispatch_case
    if fault == "source_profile":
        value = json.loads(case.processes.profile.read_text())
        value["name"] = "changed-source-profile"
        _write(case.processes.profile, value)
    elif fault in {"resolved_snapshot", "archived_source"}:
        path = case.run / ("profile.yaml" if fault == "resolved_snapshot" else "profile_source.yaml")
        value = yaml.safe_load(path.read_text())
        value["name"] = "changed-snapshot"
        path.write_text(yaml.safe_dump(value))
    elif fault == "registry_target":
        value = yaml.safe_load(case.registry_path.read_text())
        value["hardware_setups"][0]["host"]["address"] = "192.0.2.200"
        case.registry_path.write_text(yaml.safe_dump(value))
    else:
        _write(case.run / "effective_execution_plan.json", {"hardware_targets": ["foreign-setup"]})
    _assert_rejected_without_release(case)


def test_ordinary_generic_orphan_recovery_still_rejects_predispatch_other_owner(predispatch_case):
    case = predispatch_case
    # Give ordinary recovery its own valid authorization schema. Its genuine
    # dispatch and same-original-run physical-ownership rules must still apply.
    evidence = json.loads(case.evidence_path.read_text())
    evidence["schema"] = "onnx-splitpoint/generic-orphan-recovery-input"
    _write(case.evidence_path, evidence)
    auth = json.loads(case.authorization_path.read_text())
    auth.update(schema="onnx-splitpoint/generic-orphan-recovery-authorization",
                action="recover_generic_orphan_run", evidence=_descriptor(case.evidence_path))
    _write(case.authorization_path, auth)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        resource_recovery.recover_generic_orphan_run(**case.kwargs)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.receipt_path.exists()


def test_predispatch_does_not_invoke_remote_cleanup_or_create_journal_entries(predispatch_case, monkeypatch):
    case = predispatch_case

    def forbidden(*_args, **_kwargs):
        pytest.fail("pre-dispatch recovery must not invoke remote cleanup or journal configuration")

    monkeypatch.setattr(process_lease.RemoteProcessLeaseRegistry, "configure_journal", forbidden)
    monkeypatch.setattr(process_lease.RemoteProcessLeaseRegistry, "cancel_all", forbidden)
    before = sorted(str(path.relative_to(case.run)) for path in case.run.rglob("*"))
    _recover(case)
    assert sorted(str(path.relative_to(case.run)) for path in case.run.rglob("*")) == before


def test_typed_predispatch_commit_keeps_physical_outside_mutable_group(predispatch_case):
    case = predispatch_case
    receipt = _commit(case)
    assert receipt["schema"] == SCHEMA
    assert receipt["recovery_scope"] == "predispatch_run_only"
    assert [member["lock_path"] for member in receipt["group"]] == [str(case.run_lock.lock_path)]
    assert len(receipt["inspected_resources"]) == 1
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.physical_locks[case.resource].quarantine_path.exists()
    assert case.result.read_bytes() == case.result_bytes
    before = {path: path.read_bytes() for path in (case.receipt_path, case.manifest_path,
        case.run_lock.lock_path, case.physical_locks[case.resource].lock_path)}
    _commit(case)
    assert all(path.read_bytes() == data for path, data in before.items())


def test_neither_typed_commit_accepts_empty_physical_proof(predispatch_case):
    case = predispatch_case
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case, physical_locks={})
    case.receipt_data["schema"] = "onnx-splitpoint/generic-orphan-recovery"
    with _held_group(case):
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            run_control.commit_generic_orphan_recovery_group(
                run_lock=case.run_lock, physical_locks={}, receipt_path=case.receipt_path,
                receipt_data=case.receipt_data, manifest_path=case.manifest_path,
                manifest_before=case.manifest_before, manifest_after=case.manifest_after,
            )
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.receipt_path.exists()


@pytest.mark.parametrize("promotion", ["technical_success", "quality_success", "stage_success", "creation", "prior_session"])
def test_typed_commit_cannot_promote_science_stages_creation_or_other_sessions(predispatch_case, promotion):
    case = predispatch_case
    if promotion == "technical_success":
        case.manifest_after["technical_status"] = "completed"
    elif promotion == "quality_success":
        case.manifest_after["quality_decision"] = "PASS"
    elif promotion == "stage_success":
        case.manifest_after["models"]["fixture"]["stages"]["analyze_model"]["state"] = "completed"
    elif promotion == "creation":
        case.manifest_after["created_at"] = "2026-10-23T00:00:00+00:00"
    else:
        case.manifest_after["execution_sessions"][0]["status"] = "completed"
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.receipt_path.exists()


@pytest.mark.parametrize("fault", ["owner", "manifest", "final_receipt"])
def test_typed_commit_interruption_rolls_back_run_and_never_writes_physical(predispatch_case, monkeypatch, fault):
    case = predispatch_case
    original_owner = case.run_lock.lock_path.read_bytes()
    original_write = run_control._generic_orphan_atomic_write
    original_write_owner = run_control.EvaluationRunLock.write_recovery_owner
    failed = []

    def fail_write(path, data):
        should_fail = ((fault == "manifest" and path == case.manifest_path)
                       or (fault == "final_receipt" and path == case.receipt_path
                           and json.loads(data).get("state") == "committed"))
        if should_fail and not failed:
            failed.append(True)
            raise OSError("controlled interrupted pre-dispatch commit")
        return original_write(path, data)

    def fail_owner(lock, owner):
        assert lock.lock_path != case.physical_locks[case.resource].lock_path
        if fault == "owner" and owner.get("state") == "released" and not failed:
            failed.append(True)
            raise OSError("controlled interrupted pre-dispatch commit")
        return original_write_owner(lock, owner)

    monkeypatch.setattr(run_control, "_generic_orphan_atomic_write", fail_write)
    monkeypatch.setattr(run_control.EvaluationRunLock, "write_recovery_owner", fail_owner)
    with pytest.raises(OSError, match="controlled interrupted"):
        _commit(case)
    assert failed
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.run_lock.lock_path.read_bytes() == original_owner
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.physical_locks[case.resource].quarantine_path.exists()
    assert case.result.read_bytes() == case.result_bytes
    assert json.loads(case.receipt_path.read_text())["state"] != "committed"
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _normal_lock(case).acquire()


@pytest.mark.parametrize("fault", ["missing", "uncommitted", "blocked", "hash_mismatch", "manifest_changed", "malformed_marker", "corrupt_owner"])
def test_run_marker_keeps_normal_admission_closed_without_valid_committed_receipt(predispatch_case, fault):
    case = predispatch_case
    _commit(case)
    assert not case.run_lock.quarantine_path.exists()
    if fault == "missing":
        case.receipt_path.unlink()
    elif fault in {"uncommitted", "blocked", "hash_mismatch"}:
        value = json.loads(case.receipt_path.read_text())
        if fault == "hash_mismatch":
            value["operator"] = "changed-receipt-operator"
        else:
            value["state"] = "committing" if fault == "uncommitted" else "blocked"
        _write(case.receipt_path, value)
    elif fault == "manifest_changed":
        value = json.loads(case.manifest_path.read_text())
        value["quality_decision"] = "PASS"
        _write(case.manifest_path, value)
    elif fault == "malformed_marker":
        value = json.loads(case.run_lock.lock_path.read_text())
        value[MARKER] = "not-a-marker-object"
        _write(case.run_lock.lock_path, value)
    else:
        case.run_lock.lock_path.write_bytes(b'{"generic_predispatch_run_only_recovery":')
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _normal_lock(case).acquire()
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before


def test_unobservable_original_pid_scope_cannot_be_interpreted_as_dead(predispatch_case, monkeypatch):
    monkeypatch.setattr(resource_recovery, "_generic_orphan_local_observation", lambda *_args, **_kwargs: {
        "ok": False, "read_only": True,
        "recorded_scope": {"ok": False, "recorded_owners": [{"state": "unknown"}]},
        "visibility_errors": ["process_identity_unreadable"],
    })
    _assert_rejected_without_release(predispatch_case)


@pytest.mark.parametrize("fault", ["registry", "source_snapshot", "dispatch_appeared", "physical_owner"])
def test_bound_inputs_are_rechecked_after_fresh_remote_observation(predispatch_case, monkeypatch, fault):
    case = predispatch_case
    original_observe = resource_recovery.observe_resource_owners

    def change_during_observation(*args, **kwargs):
        result = original_observe(*args, **kwargs)
        if fault == "registry":
            case.registry_path.write_bytes(case.registry_path.read_bytes() + b"\n# changed during proof\n")
        elif fault == "source_snapshot":
            path = case.run / "profile.yaml"
            path.write_bytes(path.read_bytes() + b"\n# changed during proof\n")
        elif fault == "dispatch_appeared":
            _write(case.run / "models/fixture/benchmark_results/remote_benchmark_dispatch_fixture-deepx.json", {})
        else:
            path = case.physical_locks[case.resource].lock_path
            value = json.loads(path.read_text())
            value["run_id"] = "concurrently-replaced-owner"
            _write(path, value)
        return result

    monkeypatch.setattr(resource_recovery, "observe_resource_owners", change_during_observation)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _recover(case)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.result.read_bytes() == case.result_bytes
    assert not case.receipt_path.exists()


@pytest.mark.parametrize("fault", ["quarantine", "recovery_marker"])
def test_typed_commit_rejects_physical_fence_or_recovery_marker_without_mutation(predispatch_case, fault):
    case = predispatch_case
    physical = case.physical_locks[case.resource]
    if fault == "quarantine":
        _write(physical.quarantine_path, {"schema": "onnx-splitpoint/evaluation-run-lock-quarantine",
            "state": "quarantined", "owner": {"reason": "unknown prior release"}})
    else:
        owner = json.loads(physical.lock_path.read_text())
        owner[MARKER] = {"receipt_path": "/not-a-physical-recovery.json"}
        _write(physical.lock_path, owner)
    paths = [case.run_lock.lock_path, case.manifest_path, physical.lock_path]
    if physical.quarantine_path.exists():
        paths.append(physical.quarantine_path)
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not case.receipt_path.exists()


def test_typed_commit_cannot_absorb_a_physical_change_during_durable_commit(predispatch_case, monkeypatch):
    case = predispatch_case
    physical = case.physical_locks[case.resource]
    original_write = run_control._generic_orphan_atomic_write
    injected = []

    def external_change(path, data):
        result = original_write(path, data)
        if path == case.receipt_path and json.loads(data).get("state") == "committing" and not injected:
            owner = json.loads(physical.lock_path.read_text())
            owner["run_id"] = "externally-written-owner"
            _write(physical.lock_path, owner)
            injected.append(physical.lock_path.read_bytes())
        return result

    monkeypatch.setattr(run_control, "_generic_orphan_atomic_write", external_change)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _commit(case)
    assert injected
    assert physical.lock_path.read_bytes() == injected[0]
    assert not physical.quarantine_path.exists()
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert json.loads(case.receipt_path.read_text())["state"] != "committed"
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _normal_lock(case).acquire()


@pytest.mark.parametrize("journal_state", ["absent", "empty", "cancelled"])
def test_unstarted_normal_journal_directory_or_bound_cancellation_tombstone_is_safe(predispatch_case, journal_state):
    case = predispatch_case
    root = case.run / "jobs/remote_process_leases"
    if journal_state == "absent":
        for path in root.iterdir():
            path.rmdir()
        root.rmdir()
    elif journal_state == "cancelled":
        scope = process_lease.RemoteProcessLeaseScope(run_id=case.run.name, session_id=case.session)
        journal = process_lease.RemoteProcessLeaseJournal(scope=scope, directory=root / case.session)
        journal.mark_cancelled()
        tombstone = journal.directory / journal._CANCEL_FILE
        case.preserved[tombstone] = tombstone.read_bytes()
    _recover(case)
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert all(path.read_bytes() == data for path, data in case.preserved.items())


@pytest.mark.parametrize("fault", ["run", "session", "cancelled_false", "unknown_entry", "foreign_empty_directory"])
def test_wrong_tombstone_or_unknown_journal_entry_cannot_prove_predispatch(predispatch_case, fault):
    case = predispatch_case
    root = case.run / "jobs/remote_process_leases"
    if fault == "unknown_entry":
        _write(root / case.session / "unknown-operation.json", {})
    elif fault == "foreign_empty_directory":
        (root / ("e" * 32)).mkdir()
    else:
        scope = process_lease.RemoteProcessLeaseScope(run_id=case.run.name, session_id=case.session)
        journal = process_lease.RemoteProcessLeaseJournal(scope=scope, directory=root / case.session)
        journal.mark_cancelled()
        path = journal.directory / journal._CANCEL_FILE
        value = json.loads(path.read_text())
        if fault == "cancelled_false":
            value["cancelled"] = False
        else:
            value[fault + "_sha256"] = "f" * 64
        _write(path, value)
    _assert_rejected_without_release(case)


def test_typed_commit_rejects_changed_acquisition_bytes_even_for_equal_json_owner(predispatch_case):
    case = predispatch_case
    physical = case.physical_locks[case.resource]
    with _held_group(case):
        value = json.loads(physical.lock_path.read_text())
        physical.lock_path.write_text(json.dumps(value, separators=(",", ":")))
        external_bytes = physical.lock_path.read_bytes()
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            run_control.commit_generic_predispatch_run_only_recovery(
                run_lock=case.run_lock, physical_locks=case.physical_locks,
                receipt_path=case.receipt_path, receipt_data=case.receipt_data,
                manifest_path=case.manifest_path, manifest_before=case.manifest_before,
                manifest_after=case.manifest_after,
            )
    assert physical.lock_path.read_bytes() == external_bytes
    assert not physical.quarantine_path.exists()
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert not case.receipt_path.exists()


def _refresh_predispatch_predecessor_descriptors(case):
    evidence = json.loads(case.evidence_path.read_text())
    for key, source in evidence["released_physical_predecessor"].items():
        evidence["released_physical_predecessor"][key] = _descriptor(Path(source["path"]))
    evidence["registry"] = _descriptor(case.registry_path)
    _write(case.evidence_path, evidence)
    auth = json.loads(case.authorization_path.read_text())
    auth["evidence"] = _descriptor(case.evidence_path)
    _write(case.authorization_path, auth)


@pytest.mark.parametrize("fault", ["running_manifest", "wrong_manifest_session", "wrong_run_lock_pid", "wrong_contract", "noncanonical_run_lock", "run_quarantine"])
def test_exact_prior_normal_release_requires_genuine_terminal_predecessor(predispatch_case, fault):
    case = predispatch_case
    if fault in {"running_manifest", "wrong_manifest_session", "wrong_contract"}:
        path = case.predecessor_manifest_path
        value = json.loads(path.read_text())
        if fault == "running_manifest":
            value["status"] = value["execution_sessions"][-1]["status"] = "running"
        elif fault == "wrong_manifest_session":
            value["current_session_id"] = "e" * 32
        else:
            value["resume_contract"]["profile_sha256"] = "f" * 64
        _write(path, value)
    elif fault == "wrong_run_lock_pid":
        path = case.predecessor_run_lock.lock_path
        value = json.loads(path.read_text())
        value["pid"] += 1
        _write(path, value)
    elif fault == "noncanonical_run_lock":
        path = case.run.parent.parent / "recovery/copied-prior-run.lock"
        path.write_bytes(case.predecessor_run_lock.lock_path.read_bytes())
        evidence = json.loads(case.evidence_path.read_text())
        evidence["released_physical_predecessor"]["run_lock"] = _descriptor(path)
        _write(case.evidence_path, evidence)
    else:
        _write(case.predecessor_run_lock.quarantine_path, {"state": "quarantined"})
    _refresh_predispatch_predecessor_descriptors(case)
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("stage", ["run_benchmarks", "evaluate_quality", "capture",
                                   "unknown_device_work", "build_backend_artifacts"])
def test_hardware_or_quality_work_in_stage_journal_disqualifies_predispatch_recovery(predispatch_case, stage):
    case = predispatch_case
    value = json.loads(case.manifest_path.read_text())
    value["models"]["fixture"]["stages"][stage] = {"state": "running"}
    _write(case.manifest_path, value)
    _assert_rejected_without_release(case)


def test_creation_snapshot_must_equal_manifest_and_rebuilt_original_profile(predispatch_case):
    case = predispatch_case
    path = case.run / "profile_start_snapshot.json"
    value = json.loads(path.read_text())
    value["snapshot_sha256"] = "sha256:" + "f" * 64
    _write(path, value)
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("phase", ["committing", "blocked"])
def test_typed_run_fence_cannot_be_bypassed_by_generic_quarantine_callback(predispatch_case, monkeypatch, phase):
    case = predispatch_case
    assert not case.run_lock.quarantine_path.exists()
    if phase == "committing":
        _write(case.run_lock.quarantine_path, {
            "schema": "onnx-splitpoint/evaluation-run-lock-quarantine", "schema_version": 1,
            "state": "quarantined", "run_id": case.run.name, "run_dir": str(case.run),
            "owner": {"reason": "generic_predispatch_run_only_recovery_committing",
                MARKER: {"receipt_path": str(case.receipt_path)}}})
    else:
        original_write = run_control._generic_orphan_atomic_write
        failed = []

        def fail_manifest(path, data):
            if path == case.manifest_path and not failed:
                failed.append(True)
                raise OSError("controlled typed rollback")
            return original_write(path, data)

        monkeypatch.setattr(run_control, "_generic_orphan_atomic_write", fail_manifest)
        with pytest.raises(OSError, match="controlled typed rollback"):
            _commit(case)
    paths = [case.run_lock.quarantine_path, case.run_lock.lock_path, case.manifest_path,
        case.physical_locks[case.resource].lock_path]
    before = {path: path.read_bytes() for path in paths}
    callbacks = []
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _normal_lock(case).acquire(recover_quarantine=lambda owner: callbacks.append(owner) or True)
    assert callbacks == []
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("scope_flag", ["Native", "energy", "FPS"])
def test_even_hash_bound_broader_original_scope_is_never_predispatch_authority(predispatch_case, scope_flag):
    case = predispatch_case
    lineage = json.loads(case.lineage_path.read_text())
    lineage["scope"][scope_flag] = True
    action = json.loads(case.processes.action.read_text())
    action["scope"] = lineage["scope"]
    _write(case.processes.action, action)
    lineage["action"] = _descriptor(case.processes.action)
    _write(case.lineage_path, lineage)
    handoff = json.loads(case.handoff_path.read_text())
    original = handoff["actions"][0]
    original["action"] = lineage["action"]
    original["initial_lineage"] = _descriptor(case.lineage_path)
    _write(case.handoff_path, handoff)
    _refresh_evidence_descriptors(case)
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("fault", ["capture_report", "unknown_previous_session", "prior_duplicate_session", "prior_missing_session", "prior_lock_schema", "snapshot_metadata", "manifest_snapshot_metadata"])
def test_ambiguous_or_internally_inconsistent_original_evidence_never_authorizes_recovery(predispatch_case, fault):
    case = predispatch_case
    if fault == "capture_report":
        _write(case.run / "reports/energy/run_000/collector_stdout.log.cleanup.json", {
            "process_started": True, "owned_tree_quiescent": True})
    elif fault == "unknown_previous_session":
        value = json.loads(case.manifest_path.read_text())
        value["execution_sessions"][0]["status"] = "unknown"
        _write(case.manifest_path, value)
    elif fault in {"prior_duplicate_session", "prior_missing_session"}:
        value = json.loads(case.predecessor_manifest_path.read_text())
        if fault == "prior_duplicate_session":
            value["execution_sessions"].insert(0, copy.deepcopy(value["execution_sessions"][0]))
        else:
            value.pop("current_session_id")
            value["execution_sessions"][-1].pop("session_id")
            prior_lock = json.loads(case.predecessor_run_lock.lock_path.read_text())
            prior_lock.pop("session_id")
            _write(case.predecessor_run_lock.lock_path, prior_lock)
        _write(case.predecessor_manifest_path, value)
        _refresh_predispatch_predecessor_descriptors(case)
    elif fault == "prior_lock_schema":
        value = json.loads(case.predecessor_run_lock.lock_path.read_text())
        value["schema_version"] = 2
        _write(case.predecessor_run_lock.lock_path, value)
        _refresh_predispatch_predecessor_descriptors(case)
    elif fault == "snapshot_metadata":
        path = case.run / "profile_start_snapshot.json"
        value = json.loads(path.read_text())
        value["requested_selection"]["profile_name"] = "forged-selection-claim"
        _write(path, value)
    else:
        value = json.loads(case.manifest_path.read_text())
        value["profile_start_snapshot"]["profile_request"] = "forged-manifest-creation-claim"
        _write(case.manifest_path, value)
    _assert_rejected_without_release(case)


def test_valid_cancellation_tombstone_bytes_cannot_change_during_locked_observation(predispatch_case, monkeypatch):
    case = predispatch_case
    scope = process_lease.RemoteProcessLeaseScope(run_id=case.run.name, session_id=case.session)
    journal = process_lease.RemoteProcessLeaseJournal(
        scope=scope, directory=case.run / "jobs/remote_process_leases" / case.session)
    journal.mark_cancelled()
    tombstone = journal.directory / journal._CANCEL_FILE
    original_observe = resource_recovery.observe_resource_owners

    def change_tombstone(*args, **kwargs):
        result = original_observe(*args, **kwargs)
        # Equal JSON semantics do not permit replacing original control bytes.
        tombstone.write_bytes(tombstone.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(resource_recovery, "observe_resource_owners", change_tombstone)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _recover(case)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.receipt_path.exists()


def test_prior_physical_release_must_be_within_its_original_run_lease(predispatch_case):
    case = predispatch_case
    physical = json.loads(case.physical_locks[case.resource].lock_path.read_text())
    prior_lock = json.loads(case.predecessor_run_lock.lock_path.read_text())
    # Both intervals are individually valid and in the past, but the physical
    # release now follows the claimed normal Run-lock release.
    prior_lock["released_at"] = physical["acquired_at"]
    _write(case.predecessor_run_lock.lock_path, prior_lock)
    _refresh_predispatch_predecessor_descriptors(case)
    _assert_rejected_without_release(case)


def test_equal_source_bytes_do_not_allow_parent_symlink_race(predispatch_case, monkeypatch):
    case = predispatch_case
    original_observe = resource_recovery.observe_resource_owners

    def replace_source_parent(*args, **kwargs):
        result = original_observe(*args, **kwargs)
        original = case.predecessor_manifest_path.parent
        replacement = original.with_name(original.name + "-moved")
        original.rename(replacement)
        original.symlink_to(replacement, target_is_directory=True)
        return result

    monkeypatch.setattr(resource_recovery, "observe_resource_owners", replace_source_parent)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _recover(case)
    assert case.manifest_path.read_bytes() == case.original_manifest
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert not case.receipt_path.exists()


def test_diagnostics_child_symlink_cannot_write_into_original_run(predispatch_case, monkeypatch):
    case = predispatch_case
    (case.receipt_path.parent / "diagnostics").symlink_to(case.run, target_is_directory=True)

    def unexpected(*_args, **_kwargs):
        pytest.fail("unsafe diagnostic destinations must be rejected before remote observation")

    monkeypatch.setattr(resource_recovery, "observe_resource_owners", unexpected)
    _assert_rejected_without_release(case)


@pytest.mark.parametrize("group_case", ["diagnostics"], indirect=True)
def test_diagnostics_directory_cannot_collide_with_original_run(predispatch_case, monkeypatch):
    case = predispatch_case
    assert case.run.name == "diagnostics"
    case.receipt_path = case.run.parent / "receipt.json"
    case.kwargs["evidence_output"] = case.receipt_path

    def unexpected(*_args, **_kwargs):
        pytest.fail("a diagnostic directory equal to the original run must be rejected before observation")

    monkeypatch.setattr(resource_recovery, "observe_resource_owners", unexpected)
    _assert_rejected_without_release(case)


@pytest.fixture
def task192_case(predispatch_case, request):
    case = predispatch_case
    terminal_status = getattr(request, "param", "complete")
    task_output = case.run.parent.parent / "previous-completed-task-output"
    task_output.mkdir()
    worker = case.processes.identities["logger"]
    task_session = "e" * 32
    task_owner = {"run_id": case.predecessor_manifest_path.parent.name,
        "session_id": task_session, "purpose": "generic_completed_task_192",
        "pid": worker["pid"], "ppid": worker["ppid"], "pgid": worker["pgrp"]}
    task_lock = run_control.EvaluationRunLock(
        out_root=task_output.parent, run_dir=task_output, owner=task_owner)
    task_lock.acquire()
    physical = run_control.EvaluationRunLock.for_resource(case.resource, owner=task_owner)
    physical.acquire()
    branch_scope = process_lease.RemoteProcessLeaseScope(
        run_id=task_owner["run_id"], session_id=task_session + ":fixture-deepx")
    journal = task_output / "remote-process-leases" / task_session / "fixture-deepx"
    journal.mkdir(parents=True)
    cleanup_path = task_output / "cleanup-status.json"
    terminal_path = task_output / "terminal-status.json"
    _write(cleanup_path, {"at": datetime.now(timezone.utc).isoformat(),
        "active_owned_remote_operations": 0, "registry_cancelled_or_poisoned": False,
        "foreign_processes_stopped": False, "per_backend_journals": {"fixture-deepx": {
            "active_owned_remote_operations": 0, "registry_cancelled_or_poisoned": False,
            "journal": str(journal), "journal_pending_count": 0,
            "journal_cancelled": False, "journal_empty": True,
            "ownership_scope": {"run_sha256": branch_scope.run_sha256,
                "session_sha256": branch_scope.session_sha256}}}})
    _write(terminal_path, {"finished_at": datetime.now(timezone.utc).isoformat(),
        "status": terminal_status, "exitcode": 0 if terminal_status == "complete" else 1,
        "summary": {}, "error": "", "automatic_restart": False})
    physical.release()
    task_lock.release()
    case.physical_locks = {case.resource: physical}
    case.physical_before = physical.lock_path.read_bytes()
    case.task_run_lock, case.task_output = task_lock, task_output
    case.task_terminal_path, case.task_cleanup_path = terminal_path, cleanup_path
    evidence = json.loads(case.evidence_path.read_text())
    evidence["released_physical_predecessor"].update(
        owner=_descriptor(physical.lock_path), run_lock=_descriptor(task_lock.lock_path),
        task_terminal=_descriptor(terminal_path), task_cleanup=_descriptor(cleanup_path))
    _write(case.evidence_path, evidence)
    auth = json.loads(case.authorization_path.read_text())
    auth["evidence"] = _descriptor(case.evidence_path)
    _write(case.authorization_path, auth)
    case.preserved.update({path: path.read_bytes()
                          for path in (task_lock.lock_path, terminal_path, cleanup_path)})
    return case


@pytest.mark.parametrize("task192_case", ["complete", "complete_with_cell_failures"], indirect=True)
def test_normally_released_completed_task192_predecessor_is_preserved_byte_for_byte(task192_case):
    case = task192_case
    result = _recover(case)
    assert result["status"] == "committed"
    assert result["measurements_started"] == 0
    assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
    assert all(path.read_bytes() == data for path, data in case.preserved.items())
    before = json.loads(case.original_manifest)
    after = json.loads(case.manifest_path.read_text())
    assert after["status"] == after["technical_status"] == "cancelled"
    assert after["models"] == before["models"]
    assert after["quality_decision"] == before["quality_decision"]


@pytest.mark.parametrize("fault", ["missing_terminal", "missing_cleanup", "terminal_running",
    "cleanup_active", "cleanup_cancelled", "cleanup_foreign", "branch_pending", "arbitrary_purpose"])
def test_completed_task192_requires_exact_genuine_normal_terminal_and_cleanup(task192_case, fault):
    case = task192_case
    if fault in {"missing_terminal", "missing_cleanup"}:
        evidence = json.loads(case.evidence_path.read_text())
        evidence["released_physical_predecessor"].pop("task_terminal" if fault == "missing_terminal" else "task_cleanup")
        _write(case.evidence_path, evidence)
    elif fault == "terminal_running":
        value = json.loads(case.task_terminal_path.read_text())
        value["status"] = "running"
        _write(case.task_terminal_path, value)
    elif fault == "arbitrary_purpose":
        for path in (case.physical_locks[case.resource].lock_path, case.task_run_lock.lock_path):
            value = json.loads(path.read_text())
            value["purpose"] = "generic_completed_task_193"
            _write(path, value)
    else:
        value = json.loads(case.task_cleanup_path.read_text())
        if fault == "cleanup_active":
            value["active_owned_remote_operations"] = 1
        elif fault == "cleanup_cancelled":
            value["registry_cancelled_or_poisoned"] = True
        elif fault == "cleanup_foreign":
            value["foreign_processes_stopped"] = True
        else:
            value["per_backend_journals"]["fixture-deepx"]["journal_pending_count"] = 1
        _write(case.task_cleanup_path, value)
    _refresh_predispatch_predecessor_descriptors(case)
    originals = {path: path.read_bytes() for path in (
        case.task_run_lock.lock_path, case.task_terminal_path, case.task_cleanup_path)}
    _assert_rejected_without_release(case)
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize("host,central,allowed", [
    ("192.0.2.10", False, True), ("192.0.2.102", True, False)])
def test_public_predispatch_recovery_keeps_independent_live_normal_work_scoped(
        predispatch_case, workflow, monkeypatch, host, central, allowed):
    from onnx_splitpoint_tool.remote import resource_recovery_observation
    case = predispatch_case
    foreign_run, foreign = workflow(host=host, central=central)
    foreign_lock = run_control.EvaluationRunLock(
        out_root=foreign_run.parent, run_dir=foreign_run, owner={}).lock_path
    foreign_originals = {path: path.read_bytes() for path in [
        *(path for path in foreign_run.rglob("*") if path.is_file()), foreign_lock]}
    fixture_observation = resource_recovery_observation._observe_processes

    def include_known_live_authority(owners, **kwargs):
        observed = fixture_observation(owners, **kwargs)
        observed["processes"] += copy.deepcopy(foreign["processes"])
        observed.update(ok=False, status="busy")
        return observed

    monkeypatch.setattr(resource_recovery_observation, "_observe_processes", include_known_live_authority)
    if allowed:
        assert _recover(case)["status"] == "committed"
        assert case.physical_locks[case.resource].lock_path.read_bytes() == case.physical_before
        assert all(path.read_bytes() == data for path, data in case.preserved.items())
    else:
        _assert_rejected_without_release(case)
    assert all(path.read_bytes() == data for path, data in foreign_originals.items())
