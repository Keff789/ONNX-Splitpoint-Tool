"""Explicit recovery of one stopped capture's existing physical resource locks.

No measurement or device command is sent. Operator evidence describes an
actually completed intervention, not an authorization to invent one. The old
STOP, collector result and energy budgets remain unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import copy
import os
import re
import socket

from .process_lease import RemoteProcessLeaseJournal
from .resource_recovery_observation import observe_resource_owners
from ..workflow.run_control import EvaluationRunLock, WorkflowRunCleanupQuarantineError


def _require(condition, reason):
    if not condition:
        raise WorkflowRunCleanupQuarantineError("physical_resource_recovery: " + reason)


def _time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "operator timestamp requires timezone")
    return parsed


def _bound(row, expected, fields):
    return isinstance(row, Mapping) and all(row.get(k) == expected.get(k) for k in fields)


def _operator_evidence(value, source, stopped_at):
    fields = ("source", "operator", "action", "performed_at", "supply_effect", "ready_observation")
    _require(isinstance(value, Mapping) and value.get("action_performed") is True,
             "actual completed operator action is required")
    _require(all(isinstance(value.get(k), str) and value[k].strip() for k in fields),
             "operator action, time, supply effect and readiness observation are required")
    _require(value["source"] == source, "operator source does not match capture")
    performed = _time(value["performed_at"])
    _require(_time(stopped_at) <= performed <= datetime.now(timezone.utc),
             "operator action must follow STOP and cannot be in the future")
    return {**{k: value[k] for k in fields}, "action_performed": True}


def _write(path, value):
    # Use the existing atomic metadata writer without configuring/mutating a
    # historical RemoteProcessLeaseJournal session or creating another journal.
    writer = object.__new__(RemoteProcessLeaseJournal)
    writer.directory = path.parent
    writer._write_json_atomic(path, value, strict_directory_sync=True)


def recover_capture_resources(*, run_dir, session_id, operation_id, setup_id,
                              operator_evidence, registry_path=None):
    """Recover exactly one four-resource capture STOP, under all four flocks.

    Evidence and original fences are retained in the existing resource reply.
    A repeated identical call is a no-op; a new or foreign quarantine blocks.
    Normal acquire never calls this function and still rejects quarantines.
    """
    supplied = Path(run_dir).expanduser()
    _require(not supplied.is_symlink() and supplied.is_dir(), "invalid run directory")
    run = supplied.resolve()
    for value in (session_id, operation_id):
        _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value)
                 and value not in {".", ".."}, "invalid session/operation")
    directory = run / "jobs" / "remote_process_leases" / session_id
    _require(directory.is_dir() and not directory.is_symlink()
             and directory.resolve().is_relative_to(run), "invalid existing session directory")
    read = RemoteProcessLeaseJournal._read_json_exact
    config_path = directory / "controller.resources.json"
    request_path = directory / (operation_id + ".resource-request.json")
    reply_path = directory / (operation_id + ".resource-reply.json")
    config, request, reply = read(config_path), read(request_path), read(reply_path)
    identity = {"run_id": run.name, "session_id": session_id, "operation_id": operation_id}
    _require(_bound(config, identity, ("run_id", "session_id"))
             and config.get("run_dir") == str(run), "controller session binding mismatch")
    _require(_bound(request, identity, identity) and request.get("setup_id") == setup_id
             and request.get("purpose", "capture") == "capture"
             and request.get("phase") == "FINALIZING" and request.get("process_started") is True
             and re.fullmatch(r"[0-9a-f]{32}", str(request.get("token") or "")),
             "capture operation binding mismatch")
    _require(_bound(reply, request, (*identity, "token")) and reply.get("state") == "STOP"
             and reply.get("reason") == "campaign_source_completion_unresolved",
             "only the exact source-completion STOP is recoverable")
    attempt = Path(str(request.get("attempt_dir") or ""))
    _require(attempt.is_absolute() and not attempt.is_symlink()
             and attempt.resolve().is_relative_to(run), "capture attempt outside run")
    resources = sorted(r for r in config.get("setups", {}).get(setup_id, [])
                       if r.startswith(("source:", "dut:")) or r in {"controller:capture", "controller:nic"})
    _require(len(resources) == len(set(resources)) == 4
             and sum(r.startswith("source:") for r in resources) == 1
             and sum(r.startswith("dut:") for r in resources) == 1
             and {"controller:capture", "controller:nic"}.issubset(resources),
             "exact four-resource capture set required")
    source = next(r for r in resources if r.startswith("source:"))

    from ..energy.config import default_registry_path, _read_hardware_registry_unlocked
    from ..platform_power import host_config_from_setup
    from ..workflow.execution_binding import physical_dut_key
    registry_file = Path(registry_path or default_registry_path()).expanduser()
    _require(registry_file.is_file() and not registry_file.is_symlink(), "registry unavailable")
    registry_before = registry_file.read_bytes()
    registry, _ = _read_hardware_registry_unlocked(registry_file)
    setups = [s for s in registry.get("hardware_setups", []) if s.get("id") == setup_id]
    _require(len(setups) == 1, "setup missing or ambiguous in normal registry")
    setup = setups[0]
    host = host_config_from_setup(setup)
    host_fields = setup.get("host") if isinstance(setup.get("host"), Mapping) else {}
    dut = physical_dut_key({"remote": {"host": host.host,
        "physical_host_id": setup.get("physical_host_id") or host_fields.get("physical_host_id")}})
    _require(dut in resources and "source:" + str((setup.get("energy") or {}).get("urecs_address") or "").strip().lower() == source,
             "normal registry differs from stopped physical resources")

    leases = []
    originals = {}
    mutation_started = False
    recovery = None
    try:
        for resource in resources:
            lease = EvaluationRunLock.for_resource(resource, owner={**identity, "purpose": "explicit_capture_recovery"})
            lease.acquire_for_recovery()
            leases.append((resource, lease))
        # Re-read every input once the complete exclusive group is held.
        _require(read(config_path) == config and read(request_path) == request and read(reply_path) == reply,
                 "capture metadata changed during lock acquisition")
        previous_recovery = reply.get("recovery") or {}
        _require(not previous_recovery or (
            _bound(previous_recovery, identity, identity)
            and previous_recovery.get("resources") == resources), "foreign recovery metadata")
        for resource, lease in leases:
            owner = lease.previous_owner
            _require(bool(owner), "resource owner unreadable")
            if previous_recovery:
                original = (previous_recovery.get("original_resources") or {}).get(resource)
                _require(isinstance(original, Mapping), "recovery original evidence missing")
            else:
                original = {"lock_owner": owner, "quarantine": read(lease.quarantine_path)}
            original_owner, fence = original["lock_owner"], original["quarantine"]
            _require(_bound(original_owner, identity, identity)
                     and original_owner.get("physical_resource") == resource
                     and original_owner.get("hostname") == socket.gethostname()
                     and original_owner.get("pid") == config.get("parent_pid")
                     and original_owner.get("owner_pid") == request.get("owner_pid"),
                     "foreign resource ownership")
            payload = fence.get("owner") or {}
            _require(fence.get("state") == "quarantined"
                     and fence.get("schema") == "onnx-splitpoint/evaluation-run-lock-quarantine"
                     and fence.get("run_dir") == str(lease.run_dir)
                     and fence.get("run_id") == lease.run_dir.name
                     and payload.get("reason") == "campaign_source_completion_unresolved"
                     and payload.get("source_completion_unproven") is True
                     and payload.get("resource_request") == request,
                     "resource quarantine does not belong to exact stopped capture")
            own_recovery = owner.get("resource_recovery") or {}
            owner_is_recovery = (_bound(own_recovery, identity, identity)
                                 and own_recovery.get("resources") == resources)
            if os.path.lexists(lease.quarantine_path):
                current_fence = read(lease.quarantine_path)
                # Only our interrupted owner may have acquired a conservative
                # stale-writer fence; never admit another capture's quarantine.
                stale = current_fence.get("owner") or {}
                stale_owner = stale.get("previous_owner") or {}
                stale_recovery = stale_owner.get("resource_recovery") or {}
                same_stale = (stale.get("reason") == "previous_writer_terminated_uncleanly"
                              and _bound(stale_recovery, identity, identity)
                              and stale_recovery.get("resources") == resources)
                _require(current_fence == fence or (previous_recovery and same_stale),
                         "resource quarantine changed or belongs to foreign operation")
            else:
                _require(previous_recovery and (owner_is_recovery or
                         (previous_recovery.get("status") == "released" and owner.get("state") == "released")),
                         "missing unexplained resource quarantine")
            _require(owner == original_owner or owner_is_recovery or (
                previous_recovery.get("status") == "released" and owner.get("state") == "released"
                and not lease.quarantine_path.exists()), "foreign or unexplained current owner")
            originals[resource] = copy.deepcopy(original)

        stopped_at = max((item["quarantine"]["quarantined_at"] for item in originals.values()), key=_time)
        evidence = _operator_evidence(operator_evidence, source, stopped_at)
        _require(not previous_recovery or previous_recovery.get("operator_evidence") == evidence,
                 "repeat must refer to the same recorded operator action")
        if previous_recovery.get("status") == "released" and all(
                not lease.quarantine_path.exists() and lease.previous_owner.get("state") == "released"
                for _, lease in leases):
            return {"status": "already_released", **identity, "resources": resources,
                    "measurements_started": 0}

        cleanup = read(attempt / "collector_stdout.log.cleanup.json")
        _require(cleanup.get("owned_tree_quiescent") is True and cleanup.get("process_started") is True
                 and _bound(cleanup, request, ("collector_pid", "collector_start_ticks")),
                 "exact durable collector cleanup is missing")
        manifest = read(run / "run_manifest.json")
        _require(manifest.get("run_id") == run.name and manifest.get("status") in
                 {"failed", "cancelled", "completed", "success", "done"}, "parent workflow is not terminal")
        for path in directory.glob("*.remote-lease.json"):
            # Active or corrupt predecessor descriptors require their existing
            # exact cleanup path first. Recovery never kills an unknown owner.
            read(path)
            _require(False, "unresolved remote predecessor descriptor: " + path.name)
        owners = [{"pid": config.get("parent_pid"), "start_time_ticks": config.get("parent_start_ticks")},
                  {"pid": request.get("owner_pid"), "start_time_ticks": request.get("owner_start_ticks")},
                  {"pid": request.get("collector_pid"), "start_time_ticks": request.get("collector_start_ticks")}]
        _require(all(type(x["pid"]) is int and x["pid"] > 0 and str(x["start_time_ticks"] or "").isdigit()
                     for x in owners), "exact predecessor identities incomplete")
        observed = observe_resource_owners(setup, local_owners=owners,
                                           diagnostics_dir=directory)
        _require(observed.get("ok") is True, "active or unobservable local/remote predecessor")
        _require(registry_file.read_bytes() == registry_before and read(request_path) == request
                 and read(reply_path) == reply and read(config_path) == config,
                 "registry or ownership changed during observation")
        recovery = {**identity, "resources": resources, "operator_evidence": evidence,
                    "original_resources": originals, "ownership_observation": observed,
                    "status": "releasing", "started_at": datetime.now(timezone.utc).isoformat(),
                    "measurements_started": 0, "old_attempt_validated": False}
        # Commit original evidence and readiness to the existing STOP reply;
        # never rewrite its state/reason, old attempt, result or budget.
        mutation_started = True
        _write(reply_path, {**reply, "recovery": recovery})
        binding = {**identity, "resources": resources}
        recovery["status"] = "released"
        recovery["released_at"] = datetime.now(timezone.utc).isoformat()
        for resource, lease in leases:
            # Keep every original sidecar while rewriting lock inodes: a
            # process crash during truncate/write must not expose an empty
            # inode as an unfenced resource. Strict writes precede all unlinks.
            lease.write_recovery_owner({**lease.owner, "state": "released", "resource_recovery": binding,
                                        "released_at": recovery["released_at"]})
        _write(reply_path, {**reply, "recovery": recovery})
        # Every capture needs this same controller fence. Remove it last so a
        # crash in an earlier unlink cannot admit a partially released capture.
        for resource, lease in sorted(leases, key=lambda item: item[0] == "controller:capture"):
            lease.quarantine_path.unlink(missing_ok=True)
            RemoteProcessLeaseJournal._fsync_directory(lease.lock_path.parent, strict=True)
        return {"status": "released", **identity, "resources": resources,
                "measurements_started": 0, "evidence_path": str(reply_path)}
    except BaseException:
        if mutation_started:
            # Restore all original fences before dropping any resource lock.
            # The existing quarantine writer also fences its held inode when
            # the sidecar cannot be persisted.
            rollback_errors = []
            for resource, lease in leases:
                try:
                    lease.commit_quarantine_fence(originals[resource]["quarantine"]["owner"])
                    _write(lease.quarantine_path, originals[resource]["quarantine"])
                except BaseException as exc:
                    rollback_errors.append(type(exc).__name__ + ": " + str(exc))
            if recovery is not None:
                recovery.update(status="blocked", rollback_errors=rollback_errors)
                try:
                    _write(reply_path, {**reply, "recovery": recovery})
                except BaseException:
                    pass
            if rollback_errors:
                raise WorkflowRunCleanupQuarantineError(
                    "physical_resource_recovery rollback incomplete; keep resources blocked: "
                    + "; ".join(rollback_errors))
        raise
    finally:
        for _, lease in reversed(leases):
            lease.release(write_released_state=False)
