"""Explicit, evidence-bound recovery of existing physical resource locks.

Fresh owner observations are read-only; no measurement or device reset is sent.
Native numerical completion, the archived pre-inference case and capture
recovery have separate release grounds. None changes old STOPs or results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
import copy
import ast
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import tarfile
import zipfile

from .process_lease import RemoteProcessLeaseJournal
from .resource_recovery_observation import observe_resource_owners
from ..workflow.run_control import EvaluationRunLock, WorkflowRunCleanupQuarantineError


def _generic_orphan_read(path, snapshots, descriptor=None):
    """Read a regular, explicitly bound control file; retain its exact bytes."""
    _require(isinstance(path, (str, os.PathLike)), "missing Generic orphan evidence path")
    path = Path(path).expanduser()
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)),
             "unsafe Generic orphan evidence path")
    _require(path.is_file() and stat.S_ISREG(path.stat().st_mode),
             "missing or nonregular Generic orphan evidence: " + str(path))
    data = path.read_bytes()
    if descriptor is not None:
        _require(isinstance(descriptor, Mapping) and descriptor.get("path") == str(path)
                 and descriptor.get("size_bytes") == len(data)
                 and descriptor.get("sha256") == hashlib.sha256(data).hexdigest(),
                 "Generic orphan evidence descriptor differs: " + str(path))
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate Generic orphan evidence key")
            value[key] = item
        return value
    try:
        value = json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise WorkflowRunCleanupQuarantineError("invalid Generic orphan evidence: " + str(path)) from exc
    _require(isinstance(value, dict), "Generic orphan evidence must be an object")
    snapshots[path] = data
    return value


def _generic_orphan_local_observation(owners, *, run_dir, profile_path,
                                      process_groups, ssh_host, known_operations,
                                      resource_setup=None):
    """Close the recorded Generic scope, including detached transfer/helpers.

    Missing PIDs alone are insufficient: inspect the complete visible process
    namespace, old process groups/sessions and run/transfer identities too.
    No process is signalled. Only this caller's exact live ancestry is exempt
    from argument matching, never from the recorded-predecessor check.
    """
    from .resource_recovery_observation import _observe_processes
    observed = _observe_processes(owners)
    if resource_setup is not None:
        from .resource_recovery_observation import scope_local_resource_owners
        observed = scope_local_resource_owners(
            resource_setup, local_owners=owners, raw_observation=observed)
    if not observed.get("ok"):
        return {"ok": False, "recorded_scope": observed}
    proc = Path("/proc")
    def identity(pid):
        tail = (proc / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": pid, "state": tail[0], "ppid": int(tail[1]),
                "pgid": int(tail[2]), "session": int(tail[3]), "start_ticks": tail[19]}
    ancestors, errors, matches = {}, [], []
    try:
        current = os.getpid()
        while current > 0 and current not in ancestors:
            row = identity(current)
            ancestors[current] = row["start_ticks"]
            current = row["ppid"]
        entries = list(proc.iterdir())
    except (OSError, ValueError, IndexError):
        return {"ok": False, "recorded_scope": observed,
                "visibility_errors": ["caller_ancestry_or_process_visibility_unproven"]}
    tokens = [str(run_dir), Path(run_dir).name, str(profile_path), *known_operations]
    known_pids = {row["pid"] for row in owners}
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            row = identity(int(entry.name))
            if row["state"] == "Z":
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            args = [arg.decode("utf-8", "replace") for arg in argv if arg]
            own_ancestor = ancestors.get(row["pid"]) == row["start_ticks"]
            reasons = []
            if row["pgid"] in process_groups or row["session"] in process_groups:
                reasons.append("original_process_group_or_session")
            if row["ppid"] in known_pids:
                reasons.append("recorded_parent_descendant")
            if not own_ancestor:
                if any(token and token in arg for token in tokens for arg in args):
                    reasons.append("original_run_or_operation_argument")
                names = {Path(arg).name for arg in args if " " not in arg and "\n" not in arg}
                if names & {"ssh", "scp", "sftp", "rsync"} and any(ssh_host in arg for arg in args):
                    reasons.append("same_target_transfer_not_proven_foreign")
                if row["ppid"] == 1 and any("multiprocessing." in arg for arg in args):
                    reasons.append("unattributed_detached_multiprocessing_helper")
            if reasons:
                matches.append({**row, "reasons": reasons})
        except FileNotFoundError:
            if entry.exists():
                errors.append("process_identity_incomplete:" + entry.name)
        except (OSError, ValueError, IndexError):
            errors.append("process_visibility_unproven:" + entry.name)
    return {"ok": not matches and not errors, "recorded_scope": observed,
            "scope_processes": matches, "visibility_errors": errors,
            "caller_ancestry": ancestors, "read_only": True}


def recover_generic_orphan_run(*, run_dir, session_id, setup_id, evidence_path,
                               authorization_path, evidence_output, registry_path=None):
    return _recover_generic_original_run(
        run_dir=run_dir, session_id=session_id, setup_id=setup_id,
        evidence_path=evidence_path, authorization_path=authorization_path,
        evidence_output=evidence_output, registry_path=registry_path,
        predispatch_run_only=False)


def recover_generic_predispatch_run_only(*, run_dir, session_id, setup_id,
                                         evidence_path, authorization_path,
                                         evidence_output, registry_path=None):
    """Close an ended, never-dispatched Generic Quality run, without releasing a DUT.

    Only genuine original control records and a normally released physical
    predecessor are accepted. This starts no work and never resumes the run.
    """
    return _recover_generic_original_run(
        run_dir=run_dir, session_id=session_id, setup_id=setup_id,
        evidence_path=evidence_path, authorization_path=authorization_path,
        evidence_output=evidence_output, registry_path=registry_path,
        predispatch_run_only=True)


def _recover_generic_original_run(*, run_dir, session_id, setup_id, evidence_path,
                                   authorization_path, evidence_output,
                                   registry_path, predispatch_run_only):
    """Explicitly close one proven quiescent orphaned Generic Quality session.

    Input documents point to genuine original launcher, handoff and interruption
    records. Caller-created cleanup booleans are never admission evidence. The
    old run is cancelled, never promoted to a completed measurement. No build,
    SDK, inference, source/collector release or automatic resume occurs here.
    """
    from ..workflow.execution_binding import physical_dut_key
    from ..platform_power import host_config_from_setup
    from ..energy.config import default_registry_path, _read_hardware_registry_unlocked
    import uuid

    supplied = Path(run_dir).expanduser()
    _require(supplied.is_absolute() and supplied.is_dir()
             and not any(p.is_symlink() for p in (supplied, *supplied.parents)),
             "Generic orphan run must be an existing regular directory")
    run = supplied.resolve()
    _require(isinstance(session_id, str) and re.fullmatch(r"[0-9a-f]{32}", session_id),
             "exact original Generic session_id is required")
    _require(isinstance(setup_id, str) and bool(setup_id) and Path(setup_id).name == setup_id
             and setup_id not in {".", ".."}, "exact original Generic setup_id is required")
    identity = {"run_id": run.name, "run_dir": str(run), "session_id": session_id, "setup_id": setup_id}
    snapshots = {}
    evidence_file, auth_file = Path(evidence_path).expanduser(), Path(authorization_path).expanduser()
    evidence = _generic_orphan_read(evidence_file, snapshots)
    auth = _generic_orphan_read(auth_file, snapshots)
    schema = ("onnx-splitpoint/generic-predispatch-run-only-recovery"
              if predispatch_run_only else "onnx-splitpoint/generic-orphan-recovery")
    _require(evidence.get("schema") == schema + "-input"
             and type(evidence.get("schema_version")) is int and evidence["schema_version"] == 1
             and _bound(evidence, identity, identity), "Generic orphan input scope differs")
    _require(auth.get("schema") == schema + "-authorization"
             and type(auth.get("schema_version")) is int and auth["schema_version"] == 1
             and auth.get("authorized") is True and auth.get("action") == (
                 "recover_generic_predispatch_run_only" if predispatch_run_only
                 else "recover_generic_orphan_run")
             and _bound(auth, identity, identity)
             and all(isinstance(auth.get(k), str) and auth[k].strip() for k in ("operator", "source", "authorized_at")),
             "explicit Generic orphan operator authorization differs")
    _require(_time(auth["authorized_at"]) <= datetime.now(timezone.utc), "Generic authorization is in the future")
    _require(isinstance(auth.get("evidence"), Mapping), "authorization lacks its exact evidence descriptor")
    _generic_orphan_read(evidence_file, snapshots, auth["evidence"])
    if predispatch_run_only and "live_binding" in evidence:
        _require(not any(key in evidence for key in ("initial_lineage", "handoff", "interruption")),
                 "pre-dispatch binding forms must not be mixed")
        return _recover_generic_predispatch_live(
            run=run, identity=identity, evidence=evidence, auth=auth,
            evidence_file=evidence_file, auth_file=auth_file, output=Path(evidence_output).expanduser(),
            snapshots=snapshots, registry_path=registry_path)
    def referenced(name, source=evidence):
        descriptor = source.get(name)
        _require(isinstance(descriptor, Mapping), "Generic orphan lacks original " + name)
        return _generic_orphan_read(descriptor.get("path", ""), snapshots, descriptor)
    lineage, handoff, interruption = referenced("initial_lineage"), referenced("handoff"), referenced("interruption")
    _require(lineage.get("schema") == "generic420-normal-quality-group-initial-local-lineage-v1"
             and lineage.get("initial_lineage_status") == "bound" and not lineage.get("observation_error"),
             "original Generic launcher lineage is not bound")
    _require(isinstance(handoff.get("actions"), list) and isinstance(lineage.get("target"), Mapping)
             and isinstance(lineage.get("scope"), Mapping), "original Generic scope inventory is incomplete")
    rows = [row for row in handoff.get("actions", []) if isinstance(row, Mapping)
            and row.get("normal_run_dir") == str(run) and row.get("normal_run_id") == run.name]
    _require(len(rows) == 1, "original Generic handoff is missing or ambiguous")
    original = rows[0]
    _require(original.get("session_id") == session_id and original.get("initial_lineage") == evidence["initial_lineage"]
             and original.get("target") == lineage.get("target")
             and original.get("target", {}).get("setup_id") == setup_id,
             "original handoff/lineage/session/setup binding differs")
    for name in ("action", "profile", "started"):
        _require(original.get(name) == lineage.get(name), "original Generic " + name + " differs")
    action, source_profile, started = (referenced(name, lineage) for name in ("action", "profile", "started"))
    _require(action.get("profile") == lineage["profile"] and action.get("target") == lineage["target"]
             and action.get("scope") == lineage.get("scope")
             and started.get("action_spec") == lineage["action"]["path"]
             and started.get("argv") == action.get("argv"), "original Generic start/action contract differs")
    scope = lineage.get("scope") or {}
    _require(scope.get("Native") is False and scope.get("energy") is False
             and scope.get("FPS") is False and isinstance(scope.get("model_id"), str)
             and isinstance(scope.get("case_ids"), list) and bool(scope["case_ids"]),
             "only original source-only Generic Quality scope is recoverable")
    if predispatch_run_only:
        _require(scope.get("Full", False) is False
                 and len(set(scope["case_ids"])) == len(scope["case_ids"])
                 and all(isinstance(case, str) and re.fullmatch(r"b[0-9]+", case)
                         for case in scope["case_ids"]),
                 "pre-dispatch recovery requires exact split-only Quality cases")
    model_id = scope["model_id"]
    _require(Path(model_id).name == model_id and model_id not in {".", ".."}, "invalid original model scope")
    owner_rows = []
    for role in ("workflow_owner", "logger_owner"):
        owner = lineage.get(role) or {}
        _require(isinstance(owner, Mapping) and type(owner.get("pid")) is int and owner["pid"] > 0
                 and str(owner.get("process_start_ticks") or "").isdecimal()
                 and all(type(owner.get(k)) is int and owner[k] > 0 for k in ("ppid", "pgrp", "session_id"))
                 and isinstance(owner.get("argv"), list) and owner["argv"]
                 and all(isinstance(arg, str) for arg in owner["argv"]), "incomplete original " + role)
        owner_rows.append({"pid": owner["pid"], "start_time_ticks": str(owner["process_start_ticks"])})
    workflow, logger = lineage["workflow_owner"], lineage["logger_owner"]
    _require(workflow["ppid"] == logger["pid"] and workflow["argv"] == action.get("argv")
             and original.get("workflow_pid") == workflow["pid"]
             and original.get("workflow_process_start_ticks") == workflow["process_start_ticks"]
             and original.get("logger_pid") == logger["pid"]
             and original.get("logger_process_start_ticks") == logger["process_start_ticks"],
             "original controller/logger lineage differs")
    _require("run_evaluation_workflow.py" in {Path(arg).name for arg in workflow["argv"]}
             and "--profile-driven" in workflow["argv"], "original normal CLI is not proven")
    for flag, expected in (("--profile", lineage["profile"]["path"]), ("--out", str(run.parent))):
        argv = workflow["argv"]
        _require(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                 and argv[argv.index(flag) + 1] == expected, "original Generic CLI target differs")
    histories = original.get("current_local_processes")
    _require(isinstance(histories, list) and bool(histories), "original local process inventory missing")
    groups = {workflow["pgrp"], workflow["session_id"], logger["pgrp"], logger["session_id"]}
    for row in histories:
        _require(isinstance(row, Mapping) and type(row.get("pid")) is int
                 and str(row.get("start_ticks") or "").isdecimal(), "corrupt original local process inventory")
        owner_rows.append({"pid": row["pid"], "start_time_ticks": str(row["start_ticks"])})
        groups.update(row[k] for k in ("pgrp", "session_id") if type(row.get(k)) is int and row[k] > 0)
    _require(interruption.get("schema") == "generic420-continuation-v3-lifecycle-interruption-v1"
             and interruption.get("source_handoff") == evidence["handoff"]["path"]
             and isinstance(interruption.get("workflows"), list),
             "genuine interruption/handoff source differs")
    interrupted = [row for row in interruption.get("workflows", []) if isinstance(row, Mapping)
                   and row.get("normal_run_dir") == str(run) and row.get("session_id") == session_id]
    _require(len(interrupted) == 1 and interrupted[0].get("setup_id") == setup_id
             and interrupted[0].get("original_action_path") == lineage["action"]["path"],
             "interruption does not bind the exact original workflow")
    last = interrupted[0].get("last_genuine_local_owner_snapshot") or {}
    _require(last.get("pid") == workflow["pid"] and last.get("ppid") == workflow["ppid"]
             and last.get("start_ticks") == workflow["process_start_ticks"]
             and last.get("pgrp") == workflow["pgrp"], "last genuine controller identity differs")
    phase = interrupted[0].get("phase") or {}
    if phase.get("orphan_scp_pid") is not None:
        _require(type(phase["orphan_scp_pid"]) is int and str(phase.get("orphan_scp_start_ticks") or "").isdecimal(),
                 "original orphan SCP identity is incomplete")
        owner_rows.append({"pid": phase["orphan_scp_pid"], "start_time_ticks": str(phase["orphan_scp_start_ticks"])})
    target = lineage["target"]
    _require(isinstance(original.get("lease_records"), list)
             and all(isinstance(row, Mapping) and isinstance(row.get("value", {}), Mapping)
                     for row in original["lease_records"]), "original lease inventory is incomplete")
    orphan = (interruption.get("local_orphan_scope_latest") or {}).get("ssh_orphan") or {}
    operations = [str(row.get("operation_id")) for row in original.get("lease_records", [])
                  if row.get("operation_id")]
    operations += [str(row.get("value", {}).get("operation_id")) for row in original.get("lease_records", [])
                   if row.get("value", {}).get("operation_id")]
    if predispatch_run_only:
        _require(not original["lease_records"] and not operations
                 and not phase.get("orphan_scp_pid") and not orphan,
                 "pre-dispatch recovery cannot contain prior lease/transfer work")
    if orphan.get("target") == target.get("user", "") + "@" + target.get("host", ""):
        _require(orphan.get("exact_original_remote_operation") in operations
                 and type(orphan.get("pid")) is int and str(orphan.get("start_ticks") or "").isdecimal(),
                 "detached original SSH binding is incomplete")
        owner_rows.append({"pid": orphan["pid"], "start_time_ticks": str(orphan["start_ticks"])})
        groups.update(orphan[k] for k in ("pgrp", "session") if type(orphan.get(k)) is int and orphan[k] > 0)
    owner_map = {}
    for row in owner_rows:
        _require(row["pid"] not in owner_map or owner_map[row["pid"]] == row["start_time_ticks"],
                 "conflicting original process identities")
        owner_map[row["pid"]] = row["start_time_ticks"]
    owners = [{"pid": pid, "start_time_ticks": ticks} for pid, ticks in sorted(owner_map.items())]

    registry_file = Path(registry_path or default_registry_path()).expanduser()
    _require(registry_file.is_absolute() and registry_file.is_file()
             and not any(p.is_symlink() for p in (registry_file, *registry_file.parents)),
             "normal hardware registry unavailable")
    registry_before = registry_file.read_bytes()
    if predispatch_run_only:
        descriptor = evidence.get("registry") or {}
        _require(descriptor == {"path": str(registry_file),
                 "sha256": hashlib.sha256(registry_before).hexdigest(),
                 "size_bytes": len(registry_before)},
                 "pre-dispatch registry descriptor differs")
    registry, _ = _read_hardware_registry_unlocked(registry_file)
    setups = [row for row in registry.get("hardware_setups", []) if row.get("id") == setup_id]
    _require(len(setups) == 1, "registered Generic setup missing or ambiguous")
    setup, host = setups[0], host_config_from_setup(setups[0])
    _require(host.host == target.get("host") and host.user == target.get("user")
             and int(host.port) == int(target.get("port", 22)), "registered Generic SSH target differs")
    fields = setup.get("host") if isinstance(setup.get("host"), Mapping) else {}
    resource = physical_dut_key({"remote": {"host": host.host,
        "physical_host_id": setup.get("physical_host_id") or fields.get("physical_host_id")}})
    _require(interrupted[0].get("physical_resource") == resource, "interrupted physical DUT differs")
    output = Path(evidence_output).expanduser()
    _require(output.is_absolute() and not any(p.is_symlink() for p in (output, *output.parents))
             and not output.resolve().is_relative_to(run)
             and output not in snapshots and output != registry_file,
             "Generic recovery receipt must be separate from original run and evidence")
    recovery_id = uuid.uuid4().hex
    recovery_owner = {"run_id": run.name, "session_id": session_id, "purpose": "explicit_generic_orphan_recovery"}
    run_lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner=recovery_owner)
    physical = EvaluationRunLock.for_resource(resource, owner=recovery_owner)
    run_lock.acquire_for_recovery()
    try:
        physical.acquire_for_recovery()
        try:
            if predispatch_run_only:
                return _recover_generic_predispatch_locked(
                    run=run, identity=identity, lineage=lineage, original=original,
                    source_profile=source_profile, evidence=evidence, auth=auth,
                    evidence_file=evidence_file, auth_file=auth_file, output=output,
                    snapshots=snapshots, owners=owners, groups=groups, operations=operations,
                    host=host, setup=setup, registry_file=registry_file,
                    registry_before=registry_before, run_lock=run_lock,
                    physical=physical, resource=resource, recovery_id=recovery_id)
            return _recover_generic_orphan_locked(
                run=run, identity=identity, lineage=lineage, original=original, source_profile=source_profile,
                auth=auth, evidence_file=evidence_file, auth_file=auth_file, output=output,
                snapshots=snapshots, owners=owners, groups=groups, operations=operations,
                host=host, setup=setup, registry_file=registry_file, registry_before=registry_before,
                run_lock=run_lock, physical=physical, resource=resource, recovery_id=recovery_id)
        finally:
            physical.release(write_released_state=False)
    finally:
        run_lock.release(write_released_state=False)


def _recover_generic_predispatch_live(*, run, identity, evidence, auth, evidence_file,
        auth_file, output, snapshots, registry_path):
    """Use a real live Action/Creation binding, never synthesize launcher history."""
    from .predispatch_binding import load_generic_predispatch_live_binding
    from ..workflow.execution_binding import physical_dut_key
    from ..platform_power import host_config_from_setup
    from ..energy.config import default_registry_path, _read_hardware_registry_unlocked
    import uuid
    registry_file = Path(registry_path or default_registry_path()).expanduser()
    _require(registry_file.is_absolute() and registry_file.is_file()
             and not any(path.is_symlink() for path in (registry_file, *registry_file.parents)),
             "normal pre-dispatch registry is unavailable")
    registry_before = registry_file.read_bytes()
    registry, _ = _read_hardware_registry_unlocked(registry_file)
    selected = [row for row in registry.get("hardware_setups", []) if row.get("id") == identity["setup_id"]]
    _require(len(selected) == 1, "exact pre-dispatch setup is missing or ambiguous")
    setup = selected[0]
    host = host_config_from_setup(setup)
    fields = setup.get("host") if isinstance(setup.get("host"), Mapping) else {}
    resource = physical_dut_key({"remote": {"host": host.host,
        "physical_host_id": setup.get("physical_host_id") or fields.get("physical_host_id")}})
    _require(output.is_absolute() and not any(path.is_symlink() for path in (output, *output.parents))
             and not output.resolve().is_relative_to(run)
             and output not in snapshots and output != registry_file,
             "pre-dispatch receipt must be separate from original evidence")
    recovery_owner = {"run_id": run.name, "session_id": identity["session_id"],
                      "purpose": "explicit_generic_predispatch_run_only_recovery"}
    run_lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner=recovery_owner)
    physical = EvaluationRunLock.for_resource(resource, owner=recovery_owner)
    run_lock.acquire_for_recovery()
    try:
        physical.acquire_for_recovery()
        try:
            context = load_generic_predispatch_live_binding(
                run_dir=run, session_id=identity["session_id"], setup_id=identity["setup_id"],
                evidence=evidence, snapshots={}, run_lock=run_lock, registry_path=registry_file)
            frozen = dict(context["source_snapshots"])
            snapshots.update(frozen)
            end_descriptor = evidence.get("lifecycle_end") or {}
            end = _generic_orphan_read(end_descriptor.get("path", ""), snapshots, end_descriptor)
            process = context["process"]
            report = end.get("process_tree_report") or {}
            ended_identity = end.get("original_process_identity") or {}
            _require(end.get("schema") == "onnx-splitpoint/generic-predispatch-lifecycle-end-v1"
                     and type(end.get("schema_version")) is int and end["schema_version"] == 1
                     and _bound(end, identity, identity)
                     and end.get("live_binding") == evidence["live_binding"]
                     and isinstance(ended_identity, Mapping)
                     and set(ended_identity) == {"pid", "start_ticks"}
                     and type(ended_identity.get("pid")) is int and ended_identity["pid"] == process["pid"]
                     and not isinstance(ended_identity.get("start_ticks"), bool)
                     and str(ended_identity.get("start_ticks", "")).isdecimal()
                     and str(ended_identity["start_ticks"]) == str(process["start_ticks"])
                     and isinstance(report, Mapping) and report.get("root_pid") == process["pid"]
                     and type(report.get("captured_process_count")) is int and report["captured_process_count"] >= 1
                     and type(report.get("remaining_process_count")) is int and report["remaining_process_count"] == 0
                     and report.get("nested_session_cancellation_safe") is True
                     and _time(context["report"]["observed_at"]) <= _time(end.get("observed_at"))
                         <= datetime.now(timezone.utc),
                     "genuine bound pre-dispatch lifecycle end evidence is missing or unresolved")
            owner = context["run_owner"]
            argv = context["action"]["argv"]
            profile_path = Path(argv[argv.index("--profile") + 1]).expanduser()
            profile_bytes = frozen[profile_path]
            binding = {"workflow_owner": {"pid": process["pid"], "ppid": owner["ppid"],
                "pgrp": owner["pgid"], "process_start_ticks": process["start_ticks"]},
                "scope": context["scope"], "profile": {"path": str(profile_path),
                    "sha256": hashlib.sha256(profile_bytes).hexdigest(), "size_bytes": len(profile_bytes)}}
            original = {"resume_contract": context["manifest"]["resume_contract"]}
            return _recover_generic_predispatch_locked(
                run=run, identity=identity, lineage=binding, original=original,
                source_profile=context["source_profile"], evidence=evidence, auth=auth,
                evidence_file=evidence_file, auth_file=auth_file, output=output,
                snapshots=snapshots, owners=[{"pid": process["pid"],
                    "start_time_ticks": str(process["start_ticks"])}],
                groups={process["pgid"], process["session"]}, operations=[], host=host, setup=setup,
                registry_file=registry_file, registry_before=registry_before, run_lock=run_lock,
                physical=physical, resource=resource, recovery_id=uuid.uuid4().hex,
                live_context={**context, "frozen_sources": frozen})
        finally:
            physical.release(write_released_state=False)
    finally:
        run_lock.release(write_released_state=False)


def _generic_predispatch_no_work(run, model_id, manifest, snapshots):
    """Inspect only normal control/output boundaries, never suite payload trees."""
    from .process_lease import (RemoteProcessLeaseScope, REMOTE_LEASE_JOURNAL_SCHEMA,
                               REMOTE_LEASE_JOURNAL_SCHEMA_VERSION)
    preparation = {"resolve_profile", "campaign_preflight", "artifact_cache_preflight",
        "resolve_model", "check_validation_assets", "prepare_model", "analyze_model",
        "select_split_candidates", "prepare_full_baselines", "generate_benchmark_set"}
    forbidden = {"build_backend_artifacts", "run_benchmarks", "evaluate_quality", "hardware_smoke",
                 "validate_outputs", "run_native_producers", "capture", "energy", "measure_energy",
                 "aggregate_results", "generate_report"}
    models = manifest.get("models") or {}
    _require(isinstance(models, Mapping) and set(models) <= {model_id},
             "pre-dispatch model scope is ambiguous")
    model_root = run / "models"
    _require(model_root.is_dir() and not model_root.is_symlink()
             and all(path.is_dir() and not path.is_symlink() and path.name == model_id
                     for path in model_root.iterdir()),
             "pre-dispatch filesystem contains a foreign or unsafe model")
    model_row = models.get(model_id) or {}
    _require(isinstance(model_row, Mapping), "pre-dispatch model control is invalid")
    for stages in [manifest.get("root_stages") or {},
                   model_row.get("stages") or {}]:
        _require(isinstance(stages, Mapping), "pre-dispatch stage inventory is incomplete")
        for name, row in stages.items():
            _require(name in preparation | forbidden, "unknown pre-dispatch stage evidence")
            if name not in forbidden:
                continue
            _require(isinstance(row, Mapping) and not row.get("started_at")
                     and not row.get("finished_at") and not row.get("complete")
                     and row.get("state", row.get("status", "planned"))
                         in {"planned", "queued", "pending", "skipped", "not_started"},
                     "pre-dispatch run already entered a hardware/quality stage")
    model = run / "models" / model_id
    for path in (model / "benchmark_results", model / "hardware", model / "results",
                 model / "quality", run / "quality_management", run / "native_producers",
                 run / "energy"):
        if os.path.lexists(path):
            _require(path.is_dir() and not path.is_symlink() and not list(path.iterdir()),
                     "pre-dispatch run contains dispatch or execution evidence: " + str(path))
    reports = run / "reports"
    if os.path.lexists(reports):
        # ensure_run_layout creates these two directories before any stage.
        # Their empty presence is not a generated table, figure or report.
        _require(not reports.is_symlink() and reports.is_dir(),
                 "unsafe pre-dispatch reports root")
        for path in reports.iterdir():
            if path.name == "remote_trt_cache_preflight.json":
                value = _generic_orphan_read(path, snapshots)
                # The normal running StageResult is written before fn(), but
                # root_stages receives it only after fn() returns. A blocked
                # preflight therefore has a genuine file and no manifest row.
                stage_path = run / "stages" / "artifact_cache_preflight" / "stage_result.json"
                stage = _generic_orphan_read(stage_path, snapshots)
                manifest_stage = (manifest.get("root_stages") or {}).get("artifact_cache_preflight")
                _require(stage.get("schema_version") == 1
                         and stage.get("stage") == "artifact_cache_preflight"
                         and stage.get("model_id") is None
                         and stage.get("state") in {"running", "completed", "failed", "cancelled"}
                         and isinstance(stage.get("started_at"), str) and bool(stage["started_at"])
                         and (manifest_stage is None or isinstance(manifest_stage, Mapping)
                              and {key: item for key, item in manifest_stage.items()
                                   if key != "stage_result_path"} == stage),
                         "cache-preflight normal stage_result identity differs")
                probes = value.get("probes")
                targets = {row.get("id") or row.get("setup_id")
                           for row in manifest.get("hardware_targets") or [] if isinstance(row, Mapping)}
                _require(set(value) <= {"schema", "schema_version", "created_at", "read_only",
                         "hardware_action_performed", "probe_count", "probes", "status", "reason"}
                         and value.get("schema") == "onnx-splitpoint/remote-trt-cache-preflight-set"
                         and type(value.get("schema_version")) is int and value["schema_version"] == 1
                         and value.get("read_only") is True
                         and value.get("hardware_action_performed") is False
                         and _time(stage["started_at"]) <= _time(value.get("created_at"))
                         and (not stage.get("finished_at")
                              or _time(value["created_at"]) <= _time(stage["finished_at"]))
                         and isinstance(probes, list) and type(value.get("probe_count")) is int
                         and value["probe_count"] == len(probes),
                         "unbound or non-read-only pre-dispatch cache-preflight set")
                allowed_probe_fields = {"schema", "schema_version", "status", "setup_id",
                    "setup_accelerator", "model_id", "suite_dir", "active_run_ids", "runtime_contract",
                    "requirement_plan", "requirements", "native_quality_validator", "local_errors",
                    "observations", "read_only", "hardware_action_performed", "reason", "error",
                    "stable_key", "remote_base", "current_namespace", "namespace_count", "receipt_count",
                    "builder_abi", "builder_cache_abi", "builder_abi_sha256", "probe_batches"}
                for probe in probes:
                    _require(isinstance(probe, Mapping) and set(probe) <= allowed_probe_fields
                             and probe.get("schema") == "onnx-splitpoint/remote-trt-cache-preflight"
                             and type(probe.get("schema_version")) is int and probe["schema_version"] == 1
                             and probe.get("read_only") is True
                             and probe.get("hardware_action_performed") is False
                             and probe.get("model_id") == model_id and probe.get("setup_id") in targets
                             and probe.get("suite_dir") in {str(model / "benchmark_set"),
                                                           str(model / "benchmark_set" / "legacy_suite")},
                             "unbound or non-read-only pre-dispatch cache probe")
                continue
            _require(path.name in {"tables", "figures"} and not path.is_symlink()
                     and path.is_dir() and not any(path.iterdir()),
                     "pre-dispatch run contains report output or unknown entry: " + str(path))
    sessions = {row["session_id"] for row in manifest["execution_sessions"]}
    root = run / "jobs" / "remote_process_leases"
    inventory = []
    if os.path.lexists(root):
        _require(root.is_dir() and not root.is_symlink(), "unsafe pre-dispatch journal root")
        for directory in sorted(root.iterdir()):
            _require(directory.is_dir() and not directory.is_symlink()
                     and directory.name in sessions, "unknown pre-dispatch journal session")
            entries = sorted(directory.iterdir())
            _require(all(path.name == "session.cancelled.json" for path in entries),
                     "pre-dispatch run has a remote lease or unknown journal record")
            for path in entries:
                value = _generic_orphan_read(path, snapshots)
                scope = RemoteProcessLeaseScope(run_id=run.name, session_id=directory.name)
                _require(value == {"schema": REMOTE_LEASE_JOURNAL_SCHEMA,
                         "schema_version": REMOTE_LEASE_JOURNAL_SCHEMA_VERSION,
                         "run_sha256": scope.run_sha256, "session_sha256": scope.session_sha256,
                         "cancelled": True}, "foreign pre-dispatch cancellation tombstone")
            inventory.append((directory.name, tuple(path.name for path in entries)))
    return inventory


def _generic_predispatch_predecessor(evidence, snapshots, physical, run, resource):
    """Bind a different, normally terminal predecessor without rewriting it."""
    proof = evidence.get("released_physical_predecessor") or {}
    purpose = physical.previous_owner.get("purpose")
    task = purpose == "generic_completed_task_192"
    fields = {"owner", "run_manifest", "run_lock"} | ({"task_terminal", "task_cleanup"} if task else set())
    _require(isinstance(proof, Mapping) and set(proof) == fields,
             "missing genuine released physical predecessor proof")
    def read(name):
        descriptor = proof.get(name) or {}
        return _generic_orphan_read(descriptor.get("path", ""), snapshots, descriptor)
    owner, prior, prior_lock = read("owner"), read("run_manifest"), read("run_lock")
    _require(proof["owner"]["path"] == str(physical.lock_path)
             and owner == physical.previous_owner and owner.get("state") == "released"
             and purpose in {"generic", "generic_completed_task_192"}
             and owner.get("purpose") == purpose and owner.get("physical_resource") == resource
             and owner.get("run_id") != run.name and not os.path.lexists(physical.quarantine_path)
             and not any("recovery" in str(key) for key in owner),
             "physical predecessor is not an ordinary released foreign Generic owner")
    prior_dir = Path(proof["run_manifest"]["path"]).parent
    terminal = {"ok", "partial", "failed", "cancelled", "completed", "skipped"}
    _require(Path(proof["run_manifest"]["path"]) == prior_dir / "run_manifest.json"
             and prior.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
             and type(prior.get("schema_version")) is int and prior["schema_version"] == 1
             and prior.get("run_id") == prior_dir.name == owner.get("run_id")
             and prior.get("run_dir") == str(prior_dir) and prior.get("status") in terminal
             and prior.get("technical_status") in terminal,
             "released physical predecessor lacks its normal terminal Run")
    sessions = prior.get("execution_sessions") or []
    _require(isinstance(sessions, list) and bool(sessions)
             and all(isinstance(row, Mapping) and row.get("status") in terminal
                     and isinstance(row.get("session_id"), str)
                     and re.fullmatch(r"[0-9a-f]{32}", row["session_id"]) for row in sessions)
             and len({row["session_id"] for row in sessions}) == len(sessions)
             and sessions[-1].get("session_id") == prior.get("current_session_id")
             and (task or sessions[-1].get("start_origin", {}).get("pid") == owner.get("pid")),
             "released physical predecessor session identity differs")
    contract = prior.get("resume_contract") or {}
    lock_run = Path(str(prior_lock.get("run_dir") or "")) if task else prior_dir
    _require(lock_run.is_absolute() and not any(path.is_symlink() for path in (lock_run, *lock_run.parents)),
             "unsafe released predecessor task directory")
    expected = EvaluationRunLock(out_root=lock_run.parent, run_dir=lock_run, owner={})
    _require(proof["run_lock"]["path"] == str(expected.lock_path)
             and not os.path.lexists(expected.quarantine_path)
             and prior_lock.get("schema") == "onnx-splitpoint/evaluation-run-lock"
             and type(prior_lock.get("schema_version")) is int and prior_lock["schema_version"] == 1
             and prior_lock.get("state") == "released"
             and prior_lock.get("run_id") == prior_dir.name
             and prior_lock.get("run_dir") == str(lock_run)
             and (prior_lock.get("session_id") == owner.get("session_id")
                  and isinstance(owner.get("session_id"), str) and bool(owner["session_id"])
                  and prior_lock.get("purpose") == purpose if task
                  else prior_lock.get("session_id") == prior.get("current_session_id"))
             and not any("recovery" in str(key) for key in prior_lock)
             and all(prior_lock.get(key) == owner.get(key) for key in ("pid", "ppid", "pgid", "hostname"))
             and prior_lock.get("hostname") == socket.gethostname()
             and (task or all(prior_lock.get(key) == contract.get(key) == sessions[-1].get(key)
                     and isinstance(contract.get(key), str) and contract[key]
                     for key in ("profile_sha256", "plan_sha256"))),
             "released physical predecessor lacks its matching normal released Run lock")
    for value in (owner, prior_lock):
        _require(_time(value.get("acquired_at")) <= _time(value.get("released_at"))
                 <= datetime.now(timezone.utc), "released predecessor chronology is invalid")
    _require(_time(prior_lock["acquired_at"]) <= _time(owner["acquired_at"])
             <= _time(owner["released_at"]) <= _time(prior_lock["released_at"]),
             "physical predecessor release is outside its normal Run-lock lifetime")
    if task:
        terminal_record, cleanup = read("task_terminal"), read("task_cleanup")
        _require(Path(proof["task_terminal"]["path"]) == lock_run / "terminal-status.json"
                 and Path(proof["task_cleanup"]["path"]) == lock_run / "cleanup-status.json"
                 and terminal_record.get("status") in {"complete", "complete_with_cell_failures"}
                 and type(terminal_record.get("exitcode")) is int
                 and terminal_record["exitcode"] == (0 if terminal_record["status"] == "complete" else 1)
                 and type(cleanup.get("active_owned_remote_operations")) is int
                 and cleanup["active_owned_remote_operations"] == 0
                 and cleanup.get("registry_cancelled_or_poisoned") is False
                 and cleanup.get("foreign_processes_stopped") is False
                 and _time(owner["acquired_at"]) <= _time(cleanup.get("at"))
                 <= _time(terminal_record.get("finished_at")) <= _time(owner["released_at"]),
                 "released Generic FPS predecessor lacks normal terminal/cleanup evidence")
        branches = cleanup.get("per_backend_journals", {})
        _require(isinstance(branches, Mapping) and all(
            isinstance(branch, Mapping)
            and type(branch.get("active_owned_remote_operations")) is int
            and branch["active_owned_remote_operations"] == 0
            and branch.get("registry_cancelled_or_poisoned") is False
            and type(branch.get("journal_pending_count")) is int and branch["journal_pending_count"] == 0
            and branch.get("journal_cancelled") is False and branch.get("journal_empty") is True
            for branch in branches.values()), "released Generic FPS predecessor journal cleanup is unresolved")
    return proof


def _recover_generic_predispatch_locked(*, run, identity, lineage, original, source_profile,
        evidence, auth, evidence_file, auth_file, output, snapshots, owners, groups, operations,
        host, setup, registry_file, registry_before, run_lock, physical, resource, recovery_id,
        live_context=None):
    from ..workflow.run_control import (build_resume_contract,
        commit_generic_predispatch_run_only_recovery,
        validate_generic_predispatch_run_only_recovery_owner)
    from ..workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
    import yaml
    def descriptor(path, data):
        return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    inputs = {"evidence": descriptor(evidence_file, snapshots[evidence_file]),
              "authorization": descriptor(auth_file, snapshots[auth_file])}
    if output.exists():
        prior = _generic_orphan_read(output, {})
        _require(prior.get("schema") == "onnx-splitpoint/generic-predispatch-run-only-recovery"
                 and prior.get("state") == "committed" and _bound(prior, identity, identity)
                 and prior.get("inputs") == inputs
                 and validate_generic_predispatch_run_only_recovery_owner(run_lock) == prior
                 and prior.get("registry") == descriptor(registry_file, registry_before),
                 "existing pre-dispatch Run-only receipt is incomplete, changed or foreign")
        mutable = {str(run / "run_manifest.json"), str(run_lock.lock_path), str(run_lock.quarantine_path),
                   str(physical.lock_path)}
        sources = prior.get("source_evidence")
        _require(isinstance(sources, list) and bool(sources), "committed pre-dispatch source inventory missing")
        for source in sources:
            _require(isinstance(source, Mapping), "corrupt committed pre-dispatch source descriptor")
            if source.get("path") not in mutable:
                path = Path(source["path"])
                _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                         and descriptor(path, path.read_bytes()) == source,
                         "committed pre-dispatch original source changed")
        return {"status": "already_committed", **identity, "resources": [resource],
                "evidence_path": str(output), "measurements_started": 0}

    frozen_sources = (live_context or {}).get("frozen_sources", {})
    _require(all(path.is_file() and not any(parent.is_symlink() for parent in (path, *path.parents))
                 and path.read_bytes() == raw for path, raw in frozen_sources.items()),
             "captured live pre-dispatch sources changed")

    manifest_path = run / "run_manifest.json"
    manifest = _generic_orphan_read(manifest_path, snapshots)
    control = _generic_orphan_read(run / "jobs/workflow_control.json", snapshots)
    _require(manifest.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
             and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
             and manifest.get("run_id") == run.name and manifest.get("run_dir") == str(run)
             and manifest.get("status") == "running" and manifest.get("current_session_id") == identity["session_id"]
             and control.get("schema") == "onnx-splitpoint/workflow-control"
             and _bound(control, identity, ("run_id", "session_id")), "original running workflow/session differs")
    sessions = manifest.get("execution_sessions")
    terminal = {"ok", "partial", "failed", "cancelled", "completed", "skipped"}
    _require(isinstance(sessions, list) and bool(sessions)
             and all(isinstance(row, Mapping) and isinstance(row.get("session_id"), str)
                     and re.fullmatch(r"[0-9a-f]{32}", row["session_id"]) for row in sessions)
             and len({row["session_id"] for row in sessions}) == len(sessions)
             and sessions[-1].get("session_id") == identity["session_id"]
             and sessions[-1].get("status") == "running"
             and all(row.get("status") in terminal for row in sessions[:-1])
             and sessions[-1].get("start_origin", {}).get("pid") == lineage["workflow_owner"]["pid"],
             "original pre-dispatch session inventory differs")
    def yaml_file(name):
        path = run / name
        _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                 "unsafe original pre-dispatch profile")
        data = path.read_bytes(); snapshots[path] = data
        value = yaml.safe_load(data)
        _require(isinstance(value, Mapping), "original pre-dispatch profile is not a mapping")
        return dict(value)
    resolved, archived_source = yaml_file("profile.yaml"), yaml_file("profile_source.yaml")
    _require(archived_source == source_profile, "original pre-dispatch profile source changed")
    options = manifest.get("options")
    plan = _generic_orphan_read(run / "effective_execution_plan.json", snapshots)
    _require(isinstance(options, Mapping), "original archived pre-dispatch options missing")
    contract = build_resume_contract(profile_payload=resolved, effective_execution_plan=plan, options=options)
    _require(contract == manifest.get("resume_contract") == original.get("resume_contract")
             and all(sessions[-1].get(key) == contract[key] for key in
                     ("profile_sha256", "plan_sha256", "resume_contract_sha256")),
             "original pre-dispatch start/resume contract changed")
    _require(options.get("energy_enabled") is False and options.get("native_producer_enabled") is False
             and (resolved.get("energy") or {}).get("enabled") is False
             and (resolved.get("native_producers") or {}).get("enabled") is False
             and (resolved.get("hardware") or {}).get("selected_setups") == [identity["setup_id"]],
             "pre-dispatch recovery cannot close Native/Full/Energy or broader hardware")
    runs = [row for row in resolved.get("run_profiles", []) if row.get("enabled", True)]
    _require(len(runs) == 1 and runs[0].get("quality_source_only") is True
             and runs[0].get("hardware_setup_id") == identity["setup_id"]
             and runs[0].get("case_ids") == lineage["scope"]["case_ids"],
             "original pre-dispatch split-only Quality scope changed")
    archived = _generic_orphan_read(run / "profile_start_snapshot.json", snapshots)
    _require(type(archived.get("schema_version")) is int and archived["schema_version"] in {1, 2}
             and archived.get("profile_path") == lineage["profile"]["path"]
             and archived.get("profile_request") == lineage["profile"]["path"],
             "original pre-dispatch snapshot source identity differs")
    rebuilt = build_profile_start_snapshot(
        **{key: str(archived.get(key) or "") for key in
           ("profile_request", "profile_id", "profile_path", "profile_source")},
        source_profile=archived_source, resolved_profile=resolved,
        runtime_bindings=archived.get("runtime_bindings") or {},
        schema_version=int(archived.get("schema_version") or 1))
    _require(archived == public_start_snapshot_metadata(rebuilt)
             == manifest.get("profile_start_snapshot"),
             "original pre-dispatch start snapshot changed")
    workflow = lineage["workflow_owner"]
    owner = run_lock.previous_owner
    _require(owner.get("schema") == "onnx-splitpoint/evaluation-run-lock"
             and owner.get("state") == "running" and owner.get("run_id") == run.name
             and owner.get("run_dir") == str(run) and owner.get("session_id") == identity["session_id"]
             and owner.get("pid") == workflow["pid"] and owner.get("ppid") == workflow["ppid"]
             and owner.get("pgid") == workflow["pgrp"] and owner.get("hostname") == socket.gethostname()
             and all(owner.get(key) == contract[key] for key in ("profile_sha256", "plan_sha256")),
             "original pre-dispatch Run-lock identity differs")
    if os.path.lexists(run_lock.quarantine_path):
        fence = _generic_orphan_read(run_lock.quarantine_path, snapshots)
        _require(fence.get("schema") == "onnx-splitpoint/evaluation-run-lock-quarantine"
                 and fence.get("state") == "quarantined"
                 and fence.get("run_dir") == str(run)
                 and fence.get("owner") == {"reason": "previous_writer_terminated_uncleanly", "previous_owner": owner},
                 "foreign pre-dispatch quarantine must remain closed")
    predecessor = _generic_predispatch_predecessor(evidence, snapshots, physical, run, resource)
    prior_roots = [Path(predecessor["run_manifest"]["path"]).parent]
    if physical.previous_owner.get("purpose") == "generic_completed_task_192":
        prior_roots.append(Path(json.loads(snapshots[Path(predecessor["run_lock"]["path"])] )["run_dir"]))
    diagnostics = output.parent / "diagnostics"
    _require(not any(path.is_symlink() for path in (diagnostics, *diagnostics.parents))
             and (not os.path.lexists(diagnostics) or diagnostics.is_dir())
             and all(not diagnostics.resolve().is_relative_to(root.resolve())
                     and not output.resolve().is_relative_to(root.resolve()) for root in [run, *prior_roots])
             and output not in snapshots
             and not any(path.resolve().is_relative_to(diagnostics.resolve()) for path in snapshots),
             "recovery diagnostics/receipt would modify protected original evidence")
    inventory = _generic_predispatch_no_work(run, lineage["scope"]["model_id"], manifest, snapshots)
    before = _generic_orphan_local_observation(owners, run_dir=run, profile_path=lineage["profile"]["path"],
        process_groups=groups, ssh_host=host.host, known_operations=operations, resource_setup=setup)
    _require(before.get("ok") is True, "active, unknown or unbound original pre-dispatch work")
    observed = observe_resource_owners(setup, local_owners=owners, diagnostics_dir=diagnostics, resource_scoped=True)
    after_local = _generic_orphan_local_observation(owners, run_dir=run, profile_path=lineage["profile"]["path"],
        process_groups=groups, ssh_host=host.host, known_operations=operations, resource_setup=setup)
    _require(observed.get("ok") is True and after_local.get("ok") is True,
             "fresh pre-dispatch local/remote quiescence is active or unobservable")
    final_snapshots = {}
    _require(registry_file.is_file()
             and not any(p.is_symlink() for p in (registry_file, *registry_file.parents))
             and registry_file.read_bytes() == registry_before
             and _generic_predispatch_no_work(run, lineage["scope"]["model_id"], manifest, final_snapshots) == inventory
             and all(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                     and path.read_bytes() == data for path, data in snapshots.items())
             and all(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                     and path.read_bytes() == raw for path, raw in frozen_sources.items())
             and run_lock._read_owner() == owner and physical._read_owner() == physical.previous_owner,
             "pre-dispatch ownership, source, registry or absence proof changed")
    now = datetime.now(timezone.utc).isoformat()
    after = copy.deepcopy(manifest)
    after.update(status="cancelled", technical_status="cancelled", updated_at=now)
    marker = {"recovery_id": recovery_id, "session_id": identity["session_id"], "receipt_path": str(output)}
    after["execution_sessions"][-1].update(status="cancelled", finished_at=now,
        interrupted_at=now, generic_predispatch_run_only_recovery=marker)
    receipt = {"schema": "onnx-splitpoint/generic-predispatch-run-only-recovery", "schema_version": 1,
        "recovery_scope": "predispatch_run_only", **identity, "recovery_id": recovery_id,
        "inputs": inputs, "operator": auth["operator"], "authorized_at": auth["authorized_at"],
        "source_evidence": [descriptor(path, data) for path, data in snapshots.items()],
        "registry": descriptor(registry_file, registry_before), "original_local_identities": owners,
        "local_observation_before": before, "local_observation_after": after_local,
        "ownership_observation": observed, "predispatch_journal_inventory": inventory,
        "original_completion_proven": False, "measurements_started": 0}
    if live_context is not None:
        receipt["original_binding_kind"] = "stable_live_action_creation_binding"
        receipt["live_binding"] = evidence["live_binding"]
        receipt["lifecycle_end"] = evidence["lifecycle_end"]
    committed = commit_generic_predispatch_run_only_recovery(
        run_lock=run_lock, physical_locks={resource: physical}, receipt_path=output,
        receipt_data=receipt, manifest_path=manifest_path,
        manifest_before=snapshots[manifest_path], manifest_after=after)
    return {"status": "committed", **identity, "resources": [resource], "recovery_id": recovery_id,
            "evidence_path": str(output), "measurements_started": 0,
            "manifest_after_sha256": committed["manifest_after_sha256"]}


def _recover_generic_orphan_locked(*, run, identity, lineage, original, source_profile,
                                    auth, evidence_file, auth_file, output, snapshots,
                                    owners, groups, operations, host, setup, registry_file,
                                    registry_before, run_lock, physical, resource, recovery_id):
    from .process_lease import RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
    from ..workflow.run_control import (build_resume_contract, commit_generic_orphan_recovery_group,
                                       validate_generic_orphan_recovery_owner)
    import yaml

    def descriptor(path, data):
        return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
    inputs = {"evidence": descriptor(evidence_file, snapshots[evidence_file]),
              "authorization": descriptor(auth_file, snapshots[auth_file])}
    if output.exists():
        prior = _generic_orphan_read(output, {})
        _require(prior.get("schema") == "onnx-splitpoint/generic-orphan-recovery"
                 and prior.get("state") == "committed" and _bound(prior, identity, identity)
                 and prior.get("inputs") == inputs and prior.get("resources") == [resource],
                 "existing Generic recovery receipt is incomplete, changed or foreign")
        _require(validate_generic_orphan_recovery_owner(run_lock) == prior
                 and validate_generic_orphan_recovery_owner(physical) == prior
                 and hashlib.sha256((run / "run_manifest.json").read_bytes()).hexdigest()
                 == prior.get("manifest_after_sha256"), "committed Generic group changed")
        _require(prior.get("registry") == descriptor(registry_file, registry_before),
                 "committed Generic registry binding changed")
        mutable = {str(run / "run_manifest.json"), str(run_lock.quarantine_path), str(physical.quarantine_path)}
        sources = prior.get("source_evidence")
        _require(isinstance(sources, list), "committed Generic source inventory missing")
        for source in sources:
            _require(isinstance(source, Mapping), "committed Generic source descriptor is corrupt")
            if source.get("path") not in mutable:
                path = Path(source.get("path", ""))
                _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
                         and descriptor(path, path.read_bytes()) == source,
                         "committed Generic original source changed")
        return {"status": "already_committed", **identity, "resources": [resource],
                "evidence_path": str(output), "measurements_started": 0}

    manifest_path = run / "run_manifest.json"
    manifest = _generic_orphan_read(manifest_path, snapshots)
    control = _generic_orphan_read(run / "jobs/workflow_control.json", snapshots)
    _require(manifest.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
             and manifest.get("run_id") == run.name and manifest.get("run_dir") == str(run)
             and manifest.get("status") == "running" and manifest.get("current_session_id") == identity["session_id"]
             and control.get("schema") == "onnx-splitpoint/workflow-control"
             and _bound(control, identity, ("run_id", "session_id")), "original running workflow/session differs")
    sessions = manifest.get("execution_sessions")
    _require(isinstance(sessions, list) and bool(sessions)
             and all(isinstance(row, Mapping) and isinstance(row.get("session_id"), str) for row in sessions)
             and len({row["session_id"] for row in sessions}) == len(sessions), "original session inventory incomplete")
    selected = [row for row in sessions if row["session_id"] == identity["session_id"]]
    _require(len(selected) == 1 and selected[0] is sessions[-1] and selected[0].get("status") == "running"
             and selected[0].get("start_origin", {}).get("pid") == lineage["workflow_owner"]["pid"]
             and all(row.get("status") != "running" for row in sessions[:-1]), "original active session differs")
    def yaml_file(name):
        path = run / name
        _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "unsafe original profile")
        data = path.read_bytes()
        value = yaml.safe_load(data)
        _require(isinstance(value, Mapping), "original profile is not a mapping")
        snapshots[path] = data
        return dict(value)
    resolved_profile, archived_source = yaml_file("profile.yaml"), yaml_file("profile_source.yaml")
    _require(archived_source == source_profile, "original profile-source bytes/meaning changed")
    options = manifest.get("options")
    plan = _generic_orphan_read(run / "effective_execution_plan.json", snapshots)
    _require(isinstance(options, Mapping), "original archived options missing")
    contract = build_resume_contract(profile_payload=resolved_profile,
                                     effective_execution_plan=plan, options=options)
    _require(contract == manifest.get("resume_contract") == original.get("resume_contract")
             and all(selected[0].get(k) == contract[k] for k in
                     ("profile_sha256", "plan_sha256", "resume_contract_sha256")),
             "original start/resume contract changed")
    _require(options.get("energy_enabled") is False and options.get("native_producer_enabled") is False
             and (resolved_profile.get("energy") or {}).get("enabled") is False
             and (resolved_profile.get("native_producers") or {}).get("enabled") is False
             and (resolved_profile.get("hardware") or {}).get("selected_setups") == [identity["setup_id"]],
             "Generic recovery cannot release a broader Native/Energy/hardware scope")
    runs = [row for row in resolved_profile.get("run_profiles", []) if row.get("enabled", True)]
    _require(len(runs) == 1 and runs[0].get("quality_source_only") is True
             and runs[0].get("hardware_setup_id") == identity["setup_id"]
             and runs[0].get("case_ids") == lineage["scope"]["case_ids"],
             "original source-only Generic run scope changed")
    dispatch = _generic_orphan_read(run / "models" / lineage["scope"]["model_id"] / "benchmark_results"
                                   / ("remote_benchmark_dispatch_" + identity["setup_id"] + ".json"), snapshots)
    _require(dispatch.get("schema") == "onnx-splitpoint/remote-benchmark-dispatch"
             and dispatch.get("hardware_target_id") == identity["setup_id"]
             and dispatch.get("model_id") == lineage["scope"]["model_id"]
             and dispatch.get("host", {}).get("host") == host.host
             and dispatch.get("host", {}).get("user") == host.user
             and int(dispatch.get("host", {}).get("port", 22)) == int(host.port)
             and dispatch.get("args", {}).get("quality_evidence_eval_id") == run.name
             and dispatch.get("args", {}).get("quality_evidence_setup_id") == identity["setup_id"],
             "genuine original remote dispatch differs")
    workflow = lineage["workflow_owner"]
    for lock, is_run in ((run_lock, True), (physical, False)):
        owner = lock.previous_owner
        _require(owner.get("schema") == "onnx-splitpoint/evaluation-run-lock"
                 and owner.get("state") == "running" and owner.get("run_id") == run.name
                 and owner.get("pid") == workflow["pid"] and owner.get("ppid") == workflow["ppid"]
                 and owner.get("pgid") == workflow["pgrp"] and owner.get("hostname") == socket.gethostname(),
                 "original Generic lock owner differs")
        if is_run:
            _require(owner.get("run_dir") == str(run) and owner.get("session_id") == identity["session_id"]
                     and owner.get("profile_sha256") == contract["profile_sha256"]
                     and owner.get("plan_sha256") == contract["plan_sha256"], "original Run-lock contract differs")
        else:
            _require(owner.get("purpose") == "generic" and owner.get("physical_resource") == resource
                     and ("session_id" not in owner or owner["session_id"] == identity["session_id"]),
                     "physical owner is not this original Generic workflow")
        if os.path.lexists(lock.quarantine_path):
            fence = _generic_orphan_read(lock.quarantine_path, snapshots)
            _require(fence.get("schema") == "onnx-splitpoint/evaluation-run-lock-quarantine"
                     and fence.get("state") == "quarantined"
                     and fence.get("run_dir") == str(lock.run_dir)
                     and fence.get("owner") == {"reason": "previous_writer_terminated_uncleanly", "previous_owner": owner},
                     "foreign Generic quarantine must remain closed")
    original_local = _generic_orphan_local_observation(
        owners, run_dir=run, profile_path=lineage["profile"]["path"],
        process_groups=groups, ssh_host=host.host, known_operations=operations)
    _require(original_local.get("ok") is True, "active, unknown or unbound original local descendants")

    journal_root = run / "jobs/remote_process_leases"
    _require(journal_root.is_dir() and not journal_root.is_symlink(), "original Generic lease journal missing")
    directories = sorted(journal_root.iterdir())
    _require(all(path.is_dir() and not path.is_symlink() for path in directories)
             and {path.name for path in directories} == {row["session_id"] for row in sessions},
             "original session/journal inventory changed or is incomplete")
    cleanup = []
    for directory in directories:
        scope = RemoteProcessLeaseScope(run_id=run.name, session_id=directory.name)
        for path in directory.glob("*.remote-lease.json"):
            value = RemoteProcessLeaseJournal._read_json_exact(path)
            prefix = value.get("control_argv_prefix") or []
            _require(value.get("run_sha256") == scope.run_sha256
                     and value.get("session_sha256") == scope.session_sha256
                     and isinstance(prefix, list) and all(isinstance(arg, str) for arg in prefix)
                     and prefix and prefix[-1] == host.user_host and prefix.count("-p") == 1
                     and prefix.index("-p") + 1 < len(prefix)
                     and prefix[prefix.index("-p") + 1] == str(int(host.port)),
                     "original remote lease target/scope differs")
        registry = RemoteProcessLeaseRegistry()
        journal = registry.configure_journal(scope=scope, journal_dir=directory)
        reports = registry.cancel_all(grace_s=3.0)
        active, pending = registry.active_count(), journal.pending_count()
        _require(not active and not pending and all(row.get("ok") is True for row in reports),
                 "exact original remote cleanup is unresolved")
        cleanup.append({"session_id": directory.name, "journal_dir": str(directory),
                        "reports": reports, "active_count": active, "pending_count": pending})
    observed = observe_resource_owners(setup, local_owners=owners, diagnostics_dir=output.parent / "diagnostics")
    final_local = _generic_orphan_local_observation(
        owners, run_dir=run, profile_path=lineage["profile"]["path"],
        process_groups=groups, ssh_host=host.host, known_operations=operations)
    _require(observed.get("ok") is True and final_local.get("ok") is True,
             "fresh locked local/remote ownership is active or unobservable")
    _require(registry_file.read_bytes() == registry_before
             and sorted(journal_root.iterdir()) == directories
             and not list(journal_root.rglob("*.remote-lease.json"))
             and all(not path.is_symlink() and path.read_bytes() == data for path, data in snapshots.items())
             and run_lock._read_owner() == run_lock.previous_owner and physical._read_owner() == physical.previous_owner,
             "Generic ownership, source or registry changed during recovery")
    now = datetime.now(timezone.utc).isoformat()
    after = copy.deepcopy(manifest)
    after.update(status="cancelled", technical_status="cancelled", updated_at=now)
    closed = after["execution_sessions"][-1]
    closed.update(status="cancelled", finished_at=now, interrupted_at=now,
                  generic_orphan_recovery={"recovery_id": recovery_id, "session_id": identity["session_id"],
                                           "receipt_path": str(output)})
    receipt = {"schema": "onnx-splitpoint/generic-orphan-recovery", "schema_version": 1,
               **identity, "recovery_id": recovery_id, "resources": [resource], "inputs": inputs,
               "operator": auth["operator"], "authorized_at": auth["authorized_at"],
               "source_evidence": [descriptor(path, data) for path, data in snapshots.items()],
               "registry": descriptor(registry_file, registry_before), "original_local_identities": owners,
               "local_observation_before": original_local, "local_observation_after": final_local,
               "ownership_observation": observed, "scoped_remote_cleanup": cleanup,
               "original_completion_proven": False, "measurements_started": 0,
               "recovery_scope": "original_run_and_bound_physical_dut_only"}
    committed = commit_generic_orphan_recovery_group(
        run_lock=run_lock, physical_locks={resource: physical}, receipt_path=output,
        receipt_data=receipt, manifest_path=manifest_path,
        manifest_before=snapshots[manifest_path], manifest_after=after)
    return {"status": "committed", **identity, "resources": [resource], "recovery_id": recovery_id,
            "evidence_path": str(output), "measurements_started": 0,
            "manifest_after_sha256": committed["manifest_after_sha256"]}


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


def _retry_authorization(value, source, stopped_at):
    fields = ("source", "operator", "authorized_at", "reason", "new_workflow_profile")
    _require(isinstance(value, Mapping) and value.get("authorize_new_attempt") is True
             and value.get("scope") == "one_new_workflow"
             and value.get("source_completion_unproven") is True,
             "explicit authorization of one new workflow with unproven source end is required")
    _require(set(value) == set(fields) | {
        "authorize_new_attempt", "scope", "source_completion_unproven"},
        "retry authorization cannot include operator action or protocol confirmations")
    _require(all(isinstance(value.get(k), str) and value[k].strip() for k in fields),
             "retry source, operator, time, reason and new workflow profile are required")
    _require(value["source"] == source, "authorized source does not match capture")
    authorized = _time(value["authorized_at"])
    _require(_time(stopped_at) <= authorized <= datetime.now(timezone.utc),
             "retry authorization must follow STOP and cannot be in the future")
    profile = Path(value["new_workflow_profile"])
    _require(profile.is_absolute() and not profile.is_symlink() and profile.is_file()
             and profile.suffix.lower() in {".yaml", ".yml"},
             "new workflow profile must be an existing absolute YAML file")
    import yaml
    try:
        payload = yaml.safe_load(profile.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        payload = None
    _require(isinstance(payload, Mapping) and bool(payload), "new workflow profile is not a YAML mapping")
    return {**{k: value[k] for k in fields}, "authorize_new_attempt": True,
            "scope": "one_new_workflow", "source_completion_unproven": True}


def _write(path, value):
    # Use the existing atomic metadata writer without configuring/mutating a
    # historical RemoteProcessLeaseJournal session or creating another journal.
    writer = object.__new__(RemoteProcessLeaseJournal)
    writer.directory = path.parent
    writer._write_json_atomic(path, value, strict_directory_sync=True)


# This is the one archived guard failure admitted by the explicit 2.95 recovery
# policy. These are existing source/report identities, not a new seal registry.
_PRE_INFERENCE_RUN = "deepx_native_a4_295_acceptance_mobilenet_v3_large_numerical_20261015_101036"
_PRE_INFERENCE_WORKER = "18443a04008bdedee8ce50265532e54def27fc93bcd39b4b8393237fdfac7eb8"
_PRE_INFERENCE_ARCHIVE = "e6772b2a4ce11922cc58ef420b088826918765b98c63cfe92b08c20599eadc60"
_PRE_INFERENCE_RAW = "c5f8db6ba66823b78904cd1b3de5d8e7ba3d56c5fb5b232dbe1f99e47188efae"
_NATIVE_NUMERICAL_REASON = "hardware_acceptance_native_numerical_completion_unproven"
_PRE_INFERENCE_REASON = _NATIVE_NUMERICAL_REASON


def _pre_inference_case(run_dir, source_proof, setup_id):
    """Read and independently join the original failure; execute no old code."""
    run = Path(run_dir).expanduser()
    _require(run.is_absolute() and not run.is_symlink() and run.is_dir()
             and run.name == _PRE_INFERENCE_RUN, "not the admitted original failed run")
    run = run.resolve()
    snapshots = {}
    def read(path):
        path = Path(path)
        _require(path.is_absolute() and not path.is_symlink() and path.is_file()
                 and all(not parent.is_symlink() for parent in path.parents), "unsafe original evidence: " + str(path))
        data = path.read_bytes()
        snapshots[path] = data
        value = json.loads(data)
        _require(isinstance(value, dict), "original evidence must be an object")
        return value
    proof_path = Path(source_proof).expanduser()
    proof = read(proof_path)
    archive = Path(str(proof.get("archive") or ""))
    _require(archive.is_absolute() and archive.is_file() and not archive.is_symlink(), "original source archive missing")
    archive_bytes = archive.read_bytes()
    _require(hashlib.sha256(archive_bytes).hexdigest() == _PRE_INFERENCE_ARCHIVE
             and proof.get("archive_sha256") == _PRE_INFERENCE_ARCHIVE,
             "original source archive differs")
    snapshots[archive] = archive_bytes
    member = "ONNX-Splitpoint-Tool_v2.95.0/scripts/native_deepx_trt_e2e_from_benchmarkset.py"
    imported = {
        "scripts/native_hailo10_trt_e2e_from_benchmarkset.py": "verify_remote_script_native_hailo10_trt_e2e_from_benchmarkset",
        "onnx_splitpoint_tool/native_deepx_async.py": "verify_remote_asset_onnx_splitpoint_tool_native_deepx_async.py",
    }
    with zipfile.ZipFile(archive) as bundle:
        worker = bundle.read(member)
        dependencies = {relative: hashlib.sha256(bundle.read("ONNX-Splitpoint-Tool_v2.95.0/" + relative)).hexdigest()
                        for relative in imported}
    _require(hashlib.sha256(worker).hexdigest() == _PRE_INFERENCE_WORKER
             and proof.get("worker_member") == member and proof.get("worker_sha256") == _PRE_INFERENCE_WORKER
             and proof.get("observed_failure_worker_sha256") == _PRE_INFERENCE_WORKER,
             "archived worker differs from the independently observed deployment")
    tree = ast.parse(worker)
    main = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "main"), None)
    _require(main is not None, "archived worker main missing")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    guards = [node for node in calls if isinstance(node.func, ast.Name)
              and node.func.id == "_assert_expected_sha256" and node.args
              and isinstance(node.args[0], ast.Constant) and node.args[0].value == "runner"]
    constructors = [node for node in calls if isinstance(node.func, ast.Name)
                    and node.func.id in {"InferenceEngine", "NativeTRT"}]
    _require(len(guards) == 1 and constructors and guards[0].lineno < min(node.lineno for node in constructors),
             "archived guard does not precede all SDK/TRT constructors")
    identity = {"model": "mobilenet_v3_large", "case": "b027", "backend": "deepx_to_trt", "setup_id": setup_id}
    base = run / "native" / (identity["model"] + "__" + setup_id + "__b027")
    directory = base / "numerical"
    prepared, result = read(base / "PREPARED.json"), read(base / "NUMERICAL_RESULT.json")
    command, dispatch = read(directory / "COMMAND.json"), read(directory / "DISPATCH.json")
    raw_path = directory / "download/outputs/deepx_native_fifo_e2e_results.json"
    raw = read(raw_path)
    _require(hashlib.sha256(snapshots[raw_path]).hexdigest() == _PRE_INFERENCE_RAW
             and result.get("worker_report") == {"path": str(raw_path), "sha256": _PRE_INFERENCE_RAW},
             "original worker report binding differs")
    argv = command.get("argv")
    _require(isinstance(argv, list) and all(isinstance(value, str) for value in argv)
             and dispatch.get("argv") == argv and type(dispatch.get("returncode")) is int
             and dispatch["returncode"] == 4, "original command/dispatch did not end in the admitted guard failure")
    def option(name):
        _require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "missing or duplicate original option: " + name)
        return argv[argv.index(name) + 1]
    match = re.fullmatch(r"RuntimeError: replay artifact SHA-256 mismatch for runner: expected=([a-f0-9]{64}) actual=([a-f0-9]{64}) path=(.+)", str(raw.get("error") or ""))
    _require(match is not None and match[1] == option("--expected-runner-sha256")
             and match[1] != match[2] and match[2] == _PRE_INFERENCE_WORKER and match[3] in argv[:3],
             "failure is not the exact pre-inference runner guard")
    _require(not any(flag in argv for flag in ("--build-missing-engine", "--force-rebuild-engine", "--energy-workload-only"))
             and option("--deepx-runtime-mode") == "A4" and "--deepx-numerical-only" in argv
             and option("--frames") == "64" and option("--warmup") == "0"
             and option("--repetitions") == "1" and float(option("--duration-s")) == 0
             and raw.get("ok") is False and raw.get("energy_workload_only") is False,
             "original dispatch has a different execution scope")
    _require(prepared.get("identity") == identity and result.get("identity") == identity
             and result.get("status") == "failed" and result.get("new_capture_count") == 0
             and result.get("new_performance_processes") == 0, "original cell/result identity differs")
    deployment = read(base / "RUNTIME_DEPLOYMENT.json")
    verified = [step for step in deployment.get("steps", [])
                if step.get("name") == "verify_remote_script_native_deepx_trt_e2e_from_benchmarkset"]
    _require(deployment.get("status") == "verified" and len(verified) == 1
             and verified[0].get("rc") == 0 and verified[0].get("sha256_ok") is True
             and verified[0].get("expected_sha256") == verified[0].get("remote_sha256") == _PRE_INFERENCE_WORKER,
             "original deployed worker bytes are not proven")
    for relative, name in imported.items():
        steps = [step for step in deployment.get("steps", []) if step.get("name") == name]
        _require(len(steps) == 1 and steps[0].get("rc") == 0 and steps[0].get("sha256_ok") is True
                 and steps[0].get("expected_sha256") == steps[0].get("remote_sha256") == dependencies[relative],
                 "original import-before-guard dependency differs: " + relative)
    manifest, control = read(run / "run_manifest.json"), read(run / "jobs/workflow_control.json")
    session = control.get("session_id")
    closure = read(run / "reports/artifact_index_closure.json")
    _require(manifest.get("run_id") == control.get("run_id") == run.name
             and manifest.get("run_dir") == str(run) and manifest.get("current_session_id") == session
             and manifest.get("status") == "partial" and control.get("state") == "finished"
             and closure.get("run_id") == run.name and closure.get("session_id") == session
             and closure.get("status") == "pass", "original workflow is not durably terminal")
    _require(isinstance(session, str) and re.fullmatch(r"[a-f0-9]{32}", session), "original session identity invalid")
    journal = run / "jobs/remote_process_leases" / session
    config = read(journal / "controller.resources.json")
    _require(config.get("run_id") == run.name and config.get("run_dir") == str(run)
             and config.get("session_id") == session and type(config.get("parent_pid")) is int
             and str(config.get("parent_start_ticks") or "").isdigit(), "original controller identity differs")
    _require(not list((run / "jobs/remote_process_leases").rglob("*.remote-lease.json")), "unresolved remote lease descriptor")
    requests = list(journal.glob("*.resource-request.json"))
    _require(bool(requests), "original transfer journal missing")
    for path in requests:
        request = read(path)
        reply = read(path.with_name(path.name.replace(".resource-request.json", ".resource-reply.json")))
        _require(request.get("purpose") == "transfer" and request.get("phase") == "FINALIZING"
                 and request.get("process_started") is False and reply.get("state") == "RELEASED"
                 and request.get("run_id") == run.name and request.get("session_id") == session
                 and request.get("setup_id") == setup_id
                 and _bound(reply, request, ("run_id", "session_id", "operation_id", "token")),
                 "original journal has a capture, active or unresolved operation")
    report_root = proof_path.parent.parent
    original = read(report_root / "evidence/PHYSICAL_DUT_QUARANTINE_ORIGINAL.json")
    binding = read(report_root / "evidence/PHYSICAL_DUT_QUARANTINE_BINDING.json")
    return {"run": run, "directory": directory, "identity": identity, "prepared": prepared,
            "session": session, "config": config, "snapshots": snapshots,
            "original_fence": original, "fence_binding": binding,
            "imported_dependency_sha256": dependencies,
            "guard_line": guards[0].lineno, "first_constructor_line": min(node.lineno for node in constructors)}


def recover_pre_inference_resource(*, run_dir, setup_id, source_proof, operator,
                                   evidence_output, registry_path=None):
    """Release only the exact original 2.95 pre-constructor numerical failure."""
    _require(isinstance(operator, str) and bool(operator.strip()), "operator is required")
    case = _pre_inference_case(run_dir, source_proof, setup_id)
    run, config, snapshots = case["run"], case["config"], case["snapshots"]
    output = Path(evidence_output).expanduser()
    _require(output.is_absolute() and not output.is_symlink() and not output.resolve().is_relative_to(run),
             "recovery evidence must be outside the sealed failed run")
    from ..energy.config import default_registry_path, _read_hardware_registry_unlocked
    from ..platform_power import host_config_from_setup
    from ..workflow.execution_binding import physical_dut_key
    registry = Path(registry_path or default_registry_path()).expanduser()
    _require(registry.is_file() and not registry.is_symlink(), "normal hardware registry unavailable")
    before_registry = registry.read_bytes()
    payload, _ = _read_hardware_registry_unlocked(registry)
    setups = [value for value in payload.get("hardware_setups", []) if value.get("id") == setup_id]
    _require(len(setups) == 1, "registered setup missing or ambiguous")
    setup, prepared = setups[0], case["prepared"]
    host = host_config_from_setup(setup)
    fields = setup.get("host") if isinstance(setup.get("host"), Mapping) else {}
    resource = physical_dut_key({"remote": {"host": host.host,
        "physical_host_id": setup.get("physical_host_id") or fields.get("physical_host_id")}})
    _require(resource == physical_dut_key(prepared["target"])
             and resource == case["fence_binding"].get("resource")
             and resource in config.get("setups", {}).get(setup_id, []), "registered physical DUT differs from original")
    owner = {"run_id": run.name, "session_id": case["session"], "purpose": "explicit_pre_inference_recovery"}
    run_lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    lease = EvaluationRunLock.for_resource(resource, owner=owner)
    run_lock.acquire_for_recovery()
    try:
        lease.acquire_for_recovery()
        try:
            if output.is_file() and not lease.quarantine_path.exists():
                prior = RemoteProcessLeaseJournal._read_json_exact(output)
                recovery = lease.previous_owner.get("resource_recovery") or {}
                _require(prior.get("status") == "released" and prior.get("run_dir") == str(run)
                         and prior.get("resource") == resource and prior.get("operator") == operator
                         and recovery.get("evidence_path") == str(output)
                         and recovery.get("run_id") == run.name and lease.previous_owner.get("state") == "released",
                         "missing fence or foreign recovery receipt")
                return {"status": "already_released", "resource": resource, "evidence_path": str(output), "measurements_started": 0}
            _require(not output.exists(), "recovery evidence output already exists")
            original_owner = lease.previous_owner
            fence = lease.quarantine_path.read_bytes()
            _require(not lease.quarantine_path.is_symlink()
                     and str(lease.quarantine_path) == case["fence_binding"].get("original_path")
                     and hashlib.sha256(fence).hexdigest() == case["fence_binding"].get("sha256")
                     and json.loads(fence) == case["original_fence"], "original numerical fence changed")
            payload = case["original_fence"]
            _require(payload.get("state") == "quarantined"
                     and payload.get("schema") == "onnx-splitpoint/evaluation-run-lock-quarantine"
                     and payload.get("owner") == {"reason": _PRE_INFERENCE_REASON, "evidence": str(case["directory"])}
                     and original_owner.get("physical_resource") == resource
                     and original_owner.get("run_id") == run.name and original_owner.get("purpose") == "hardware_acceptance_native_numerical"
                     and original_owner.get("pid") == config["parent_pid"] and original_owner.get("hostname") == socket.gethostname()
                     and original_owner.get("state") == "running", "foreign numerical fence or predecessor")
            previous_run = run_lock.previous_owner
            _require(not os.path.lexists(run_lock.quarantine_path)
                     and previous_run.get("run_id") == run.name and previous_run.get("session_id") == case["session"]
                     and previous_run.get("pid") == config["parent_pid"] and previous_run.get("state") == "released",
                     "original run owner is not terminal")
            observed = observe_resource_owners(setup, local_owners=[{"pid": config["parent_pid"],
                "start_time_ticks": config["parent_start_ticks"]}], diagnostics_dir=output.parent / "diagnostics")
            _require(observed.get("ok") is True, "active or unobservable original/local/remote owner")
            _require(registry.read_bytes() == before_registry
                     and all(path.read_bytes() == data for path, data in snapshots.items())
                     and not list((run / "jobs/remote_process_leases").rglob("*.remote-lease.json"))
                     and lease._read_owner() == original_owner and lease.quarantine_path.read_bytes() == fence,
                     "original evidence, registry, lease or fence changed during observation")
            receipt = {"schema": "onnx-splitpoint/physical-resource-recovery", "status": "verified_pre_inference",
                "kind": "legacy_native_numerical_runner_guard", "run_id": run.name, "run_dir": str(run),
                "session_id": case["session"], "resource": resource, "setup_id": setup_id, "operator": operator,
                "source_proof": str(Path(source_proof)), "evidence_path": str(output),
                "original_owner": original_owner, "original_quarantine": payload,
                "original_quarantine_bytes": fence.decode("utf-8"), "ownership_observation": observed,
                "original_evidence": [{"path": str(path), "sha256": hashlib.sha256(data).hexdigest()} for path, data in snapshots.items()],
                "pre_inference_proof": {"worker_sha256": _PRE_INFERENCE_WORKER,
                    "imported_dependency_sha256": case["imported_dependency_sha256"],
                    "guard_line": case["guard_line"], "first_constructor_line": case["first_constructor_line"],
                    "derived_inference_requests": 0, "raw_counters_rewritten": False},
                "measurements_started": 0, "old_attempt_validated": False,
                "verified_at": datetime.now(timezone.utc).isoformat()}
            output.parent.mkdir(parents=True, exist_ok=True)
            _write(output, receipt)
            receipt.update(status="released", released_at=datetime.now(timezone.utc).isoformat())
            try:
                lease.resolve_quarantine(expected_owner=original_owner, expected_quarantine=fence,
                    recovery={"status": "verified_pre_inference", "resource": resource, "run_id": run.name,
                              "evidence_path": str(output)}, commit_evidence=lambda: _write(output, receipt))
            except BaseException:
                receipt.update(status="blocked")
                try:
                    _write(output, receipt)
                except BaseException:
                    pass
                raise
            return {"status": "released", "resource": resource, "evidence_path": str(output), "measurements_started": 0}
        finally:
            lease.release(write_released_state=False)
    finally:
        run_lock.release(write_released_state=False)


def _full_ort_pre_runtime_proof(raw, prepared, directory, source_archive, read, snapshots):
    """Prove a closed ORT fallback stopped before Native accelerator entry.

    This is an independent release ground, not a replacement worker report or
    an interpretation of unknown counters as zero. Only the complete observed
    missing-plan/CPU-session-failure route is supported.
    """
    from ..workflow.hardware_acceptance_numerical import device_completion_safety
    identity = prepared["identity"]
    rows = raw.get("rows") or []
    _require(identity["backend"] == "native_full_deepx" and len(rows) == 1, "source release requires one Native Full row")
    row = rows[0]
    _require(row.get("backend") == "native_full_deepx" and row.get("deepx_runtime_mode") == "A4"
             and row.get("unsafe_runtime_state") is True and row.get("unsafe_runtime_reasons") == []
             and all(row.get(key) == {} for key in ("deepx_prepared_feed_benchmark", "deepx_async_observation", "numerical_validation", "failed_phase_counts"))
             and all(row.get(key) is None for key in ("pending", "unreleased", "submitted_count", "completed_frames", "completed_work_units"))
             and row.get("repetition_count_attempted") == row.get("repetition_count_requested") == 1
             and row.get("repetition_count_valid") == 0, "not the closed missing measured-child report shape")
    def contradictions(value, semantic_child=False):
        if isinstance(value, Mapping):
            semantic_child = semantic_child or value is row.get("preparation_evidence")
            _require("deepx_execution_lifecycle" not in value or semantic_child, "unexpected additional SDK lifecycle")
            if not semantic_child:
                for key, counter in value.items():
                    if key in {"submitted", "submitted_count", "submit_attempts", "wait_completed", "accepted_work_units", "mixed_submitted_requests"} or key.endswith("_calls_started"):
                        _require(counter is None or (type(counter) is int and counter == 0), "unexpected measured SDK entry")
            for key in ("unsafe_runtime_state", "deepx_unsafe_runtime_state", "unsafe_device_state", "serial_completion_unresolved"):
                if key in value and not (value is row and key == "unsafe_runtime_state"):
                    _require(value[key] is False, "additional explicit unsafe SDK evidence")
            for key in ("pending", "unreleased", "pending_count", "unresolved_jobs"):
                if key in value and not (value is row and key in {"pending", "unreleased"} and value[key] is None):
                    _require(type(value[key]) is int and value[key] == 0, "additional pending SDK work")
            for key in tuple(value):
                if key.endswith("_calls_started") and value[key] is not None:
                    returned = key.removesuffix("_started") + "_completed"
                    _require(type(value[key]) is int and type(value.get(returned)) is int and value[key] == value[returned], "additional incomplete serial SDK call")
            for key, child in value.items():
                if key != "full_command_contract": contradictions(child, semantic_child)
        elif isinstance(value, list):
            for child in value: contradictions(child, semantic_child)
    contradictions(raw)
    steps = row.get("steps") or []
    _require(len(steps) == 1 and isinstance(steps[0], Mapping), "unexpected measured child sequence")
    step, benchmark = steps[0], None
    cmd, output = step.get("cmd"), step.get("stdout")
    _require(isinstance(cmd, list) and all(isinstance(value, str) for value in cmd)
             and Path(cmd[1]).name == "run_benchmark_suite_from_set.py" and isinstance(output, str)
             and step.get("rc") == step.get("returncode") == 0 and not step.get("timed_out")
             and cmd.count("--benchmark-set") == 1 and cmd.count("--") == 1, "measured wrapper did not close normally")
    benchmark = cmd[cmd.index("--benchmark-set") + 1]
    forwarded = cmd[cmd.index("--") + 1:]
    _require("--plan" not in forwarded and "--provider" not in forwarded and "--deepx-numerical-only" in forwarded
             and "--deepx-runtime-mode" in forwarded and forwarded[forwarded.index("--deepx-runtime-mode") + 1] == "A4",
             "unexpected measured suite route")
    traces = [line.removeprefix("[suite-wrapper] cmd: ") for line in output.splitlines() if line.startswith("[suite-wrapper] cmd: ")]
    _require(len(traces) == 1 and shlex.split(traces[0]) == [cmd[0], benchmark + "/benchmark_suite.py", *forwarded],
             "observed suite command differs from dispatched wrapper")
    def binary(path):
        path = Path(path)
        _require(path.is_absolute() and path.is_file() and not path.is_symlink()
                 and all(not parent.is_symlink() for parent in path.parents), "unsafe original source evidence")
        data = path.read_bytes()
        _require(path not in snapshots or snapshots[path] == data, "original source changed while joining")
        snapshots[path] = data
        return data
    archive = Path(source_archive).expanduser()
    archive_bytes = binary(archive)
    with zipfile.ZipFile(archive) as bundle:
        def member(suffix):
            names = [name for name in bundle.namelist() if name.endswith("/" + suffix)]
            _require(len(names) == 1, "original source archive member missing or ambiguous: " + suffix)
            _require(not Path(names[0]).is_absolute() and ".." not in Path(names[0]).parts
                     and not stat.S_ISLNK(bundle.getinfo(names[0]).external_attr >> 16), "unsafe original source archive member")
            return bundle.read(names[0])
        outer = member("scripts/native_full_baseline_eval_runner.py")
        suite_template = member("onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt")
        case_template = member("onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt")
        package_adapter = member("onnx_splitpoint_tool/runners/backends/hailo_backend.py")
        release_identity = member("onnx_splitpoint_tool/release_identity.py")
        endpoint = member("onnx_splitpoint_tool/native_output_endpoint.py")
    _require(hashlib.sha256(outer).hexdigest() == (row.get("full_command_contract") or {}).get("runner_sha256"),
             "source archive differs from the actually executed Full runner")
    outer_tree = ast.parse(outer)
    guards = [node for node in ast.walk(outer_tree) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "step.get('timed_out') or not prepared_feed or (not changed)"
              and any(isinstance(child, ast.Assign) and ast.unparse(child) == "result['unsafe_runtime_state'] = True" for child in node.body)]
    _require(len(guards) == 1, "original missing-child unsafe projection is not established")
    staging = read(directory / "software_staging.json")
    tar_path = directory / "full_software.tar.gz"
    tar_bytes = binary(tar_path)
    _require(staging.get("returncode") == 0 and staging.get("bundle") == str(tar_path)
             and staging.get("bundle_sha256") == hashlib.sha256(tar_bytes).hexdigest(), "original software staging bundle differs")
    run = directory.parents[2]
    log = binary(run / "evaluation_workflow.log").decode()
    remote_tar = benchmark + "/full_software.tar.gz"
    digest = hashlib.sha256(tar_bytes).hexdigest()
    extraction = ("test " + digest + ' = "$(sha256sum -- ' + shlex.quote(remote_tar)
                  + " | cut -d ' ' -f 1)\" && tar -xzf " + shlex.quote(remote_tar) + " -C " + shlex.quote(benchmark))
    diagnostics = []
    for path in directory.parent.joinpath("diagnostics").glob("ssh_*.json"):
        value = json.loads(path.read_text())
        if value.get("command") == extraction:
            diagnostics.append((path, read(path)))
    _require(len(diagnostics) == 1 and diagnostics[0][1].get("returncode") == 0
             and diagnostics[0][1].get("cleanup_proven") is True and diagnostics[0][1].get("target_id") == identity["setup_id"]
             and any("[scp-upload] " + str(tar_path) + " -> " in line and line.endswith(":" + remote_tar) for line in log.splitlines())
             and any("[ssh-result] setup=" + identity["setup_id"] + " rc=0 " in line and str(diagnostics[0][0]) in line for line in log.splitlines()),
             "normal indexed staging trace does not prove SHA-checked software extraction")
    with tarfile.open(tar_path, "r:gz") as bundle:
        members = bundle.getmembers()
        _require(len({item.name for item in members}) == len(members)
                 and all(item.isfile() and not Path(item.name).is_absolute() and ".." not in Path(item.name).parts for item in members), "unsafe staged source bundle")
        files = {item.name: bundle.extractfile(item).read() for item in members}
    _require("benchmark_suite.py" in files and not any(Path(name).name == "benchmark_plan.json" for name in files), "source bundle has a separate plan")
    with zipfile.ZipFile(archive) as bundle:
        for name, data in files.items():
            if not name.startswith("splitpoint_runners/") or not name.endswith(".py"):
                continue
            relative = name.removeprefix("splitpoint_runners/")
            if relative == "runners/__init__.py":
                expected_namespace = ('"""Canonical Native helper namespace in generated suites."""\nimport sys as _sys\n'
                    'from .. import harness as _harness\nfrom ..harness import yolo as _yolo\n'
                    "_sys.modules[__name__ + '.harness'] = _harness\n_sys.modules[__name__ + '.harness.yolo'] = _yolo\n").encode()
                _require(data == expected_namespace, "unexpected generated helper namespace")
            else:
                candidates = [item for item in bundle.namelist() if any(item.endswith("/" + prefix + relative)
                              for prefix in ("onnx_splitpoint_tool/", "onnx_splitpoint_tool/runners/", "scripts/"))]
                _require(any(bundle.read(item) == data for item in candidates), "staged import dependency differs from original archive: " + name)
            _require(binary(directory / "software" / name) == data, "retained import dependency differs from transmitted bytes")
    suitesource = files["benchmark_suite.py"]
    _require(binary(directory / "software/benchmark_suite.py") == suitesource, "staged suite differs from transmitted bytes")
    def function(source, name):
        return next((node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name), None)
    suite_main = function(suitesource, "main")
    _require(suite_main is not None and suitesource == suite_template.replace(b"__BENCH_JSON__", b"benchmark_set.json"),
             "staged suite control flow differs from original normal source")
    text = ast.unparse(suite_main)
    _require("default_plan = root / 'benchmark_plan.json'" in text
             and "runs = [{'id': f'ort_{args.provider}', 'type': 'onnxruntime', 'provider': args.provider, 'image_scale': args.image_scale}]" in text,
             "original suite does not have the observed no-plan ORT fallback")
    bench = read(directory / "software/benchmark_set.json")
    _require(forwarded.count("--run-id") == 1 and any(item.get("id") == forwarded[forwarded.index("--run-id") + 1]
             and item.get("type", item.get("kind")) in {"deepx", "deepx_m1", "dx_m1"}
             and item.get("variants") == ["full"] for item in (bench.get("plan") or {}).get("runs") or []),
             "original embedded plan does not declare the requested DeepX Full run")
    copy_receipt = read(directory / "benchmark_staging.json")
    copy_lines = [line.removeprefix("FULL_STAGING=") for line in str(copy_receipt.get("output") or "").splitlines() if line.startswith("FULL_STAGING=")]
    _require(copy_receipt.get("returncode") == 0 and len(copy_lines) == 1, "original artifact staging is not closed")
    copied = json.loads(copy_lines[0]); copied_files = {item["path"]: item for item in copied.get("files") or [] if "path" in item}
    _require(copied.get("destination") == benchmark and copied.get("model_builds") == 0 and copied.get("historical_results_copied") is False
             and "benchmark_plan.json" not in copied_files
             and copied_files.get("benchmark_set.json", {}).get("sha256") == hashlib.sha256(snapshots[directory / "software/benchmark_set.json"]).hexdigest(),
             "original fresh staging does not establish the absent separate plan")
    cases = [str(item.get("case_dir") or item.get("case_id") or item.get("case") or item.get("id") or item.get("name") or "") for item in bench.get("cases") or []]
    observed = re.findall(r"^\[ort_auto\] \[(\d+)/(\d+)\] Running (b\d+) \(provider=auto\)$", output, re.M)
    _require(cases and observed == [(str(i + 1), str(len(cases)), case) for i, case in enumerate(cases)]
             and output.rstrip().endswith("Done.") and output.count("Traceback (most recent call last):") == 1
             and "Using mixed backends: default=auto, full=auto, stage1=auto, stage2=auto (ORT baseline: ['CPUExecutionProvider'])" in output
             and "providers in use: ['CPUExecutionProvider']" in output, "complete observed CPU-only case route is not established")
    runners = [name for name in files if name.endswith("/run_split_onnxruntime.py")]
    _require(runners == [cases[0] + "/run_split_onnxruntime.py"], "unexpected additional runnable cases in transmitted source")
    _require(all(name in {"benchmark_suite.py", "scientific_reporter_v60.py", runners[0]}
                 or (name.startswith("splitpoint_runners/") and name.endswith(".py")) for name in files), "unexpected executable in transmitted source")
    if "scientific_reporter_v60.py" in files:
        with zipfile.ZipFile(archive) as bundle:
            names = [name for name in bundle.namelist() if name.endswith("/onnx_splitpoint_tool/resources/templates/scientific_reporter_v60.py.txt")]
            _require(len(names) == 1 and bundle.read(names[0]) == files["scientific_reporter_v60.py"], "unexpected companion reporter bytes")
    case_source = files[runners[0]]
    _require(binary(directory / "software" / runners[0]) == case_source, "case bytes differ from original transmitted source")
    case_main = function(case_source, "main")
    template_text, adapter_text = case_template.decode(), package_adapter.decode()
    def adapter(source):
        return ast.get_source_segment(source, function(source, "_adapt_tensor"))
    generated = template_text.replace(adapter(template_text), adapter(adapter_text), 1)
    provenance = {"template": {"path": "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt", "source_sha256": hashlib.sha256(case_template).hexdigest()},
                  "package_adapter": {"path": "onnx_splitpoint_tool/runners/backends/hailo_backend.py", "source_sha256": hashlib.sha256(package_adapter).hexdigest()}}
    release = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(release_identity).body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"VERSION", "BUILD_ID"}}
    generated = generated.replace('"__GENERATED_SOURCE_FILES__"', repr(provenance)).replace('"__GENERATED_RELEASE_IDENTITY__"', repr({"version": release["VERSION"], "build_id": release["BUILD_ID"]}))
    generated = generated.replace("__MANIFEST_FILENAME__", "split_manifest.json").replace("__VENDORED_ENDPOINT_ATTESTOR_SHA256__", hashlib.sha256(endpoint).hexdigest()).replace("__DEFAULT_PROVIDER__", "auto")
    _require(case_main is not None and generated.encode() == case_source, "case control flow differs from original normal source")
    sessions = [node for node in ast.walk(case_main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_create_session" and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "part1"]
    constructors = [node for node in ast.walk(case_main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in {"InferenceEngine", "DeepXSession", "NativeTRTSession", "NativeTRT", "HailoSession"}]
    _require(len(sessions) == 1 and constructors and sessions[0].lineno < min(node.lineno for node in constructors)
             and "providers = _pick_providers('cpu', avail)" in ast.unparse(case_main), "case failure does not precede accelerator construction")
    locus = benchmark + "/" + runners[0]
    _require(f'File "{locus}", line {sessions[0].lineno}, in main' in output
             and 'sess_p1, info_p1 = _create_session(' in output and "NoSuchFile:" in output
             and f"[warn] case failed: {cases[0]} (rc=1)" in output
             and all(f"[warn] missing runner: {benchmark}/{case}/run_split_onnxruntime.py" in output for case in cases[1:]),
             "observed case exception does not match the pre-runtime source locus")
    semantic_files = list((directory / "download/outputs/native_full_outputs").rglob("native_full_semantic_dump.json"))
    _require(len(semantic_files) == 1, "original semantic child evidence is not unique")
    semantic = read(semantic_files[0]); semantic_safety = device_completion_safety(semantic)
    preparation = row.get("preparation_evidence") or {}
    semantic_step = preparation.get("step") or {}
    semantic_stdout = json.loads(semantic_step.get("stdout") or "null")
    _require(semantic.get("ok") is True and semantic.get("backend") == "native_full_deepx"
             and semantic.get("model") == identity["model"] and semantic.get("setup_id") == identity["setup_id"]
             and semantic.get("image_sha256") == prepared["source_contract"].get("input_image_sha256")
             and semantic.get("serial_reference_calls_started") == semantic.get("serial_reference_calls_completed") == 1
             and semantic_safety.get("device_completion_proven") is True
             and device_completion_safety(preparation).get("device_completion_proven") is True
             and preparation.get("deepx_execution_lifecycle") == semantic.get("deepx_execution_lifecycle")
             and semantic_step.get("rc") == semantic_step.get("returncode") == 0 and not semantic_step.get("timed_out")
             and isinstance(semantic_stdout, Mapping) and all(semantic_stdout.get(key) == value for key, value in semantic.items()),
             "independent semantic child SDK drain or identity is not proven")
    for role, path in (("dxnn", "deepx/deepx_m1/full/model.dxnn"), ("source_onnx", "models/" + identity["model"] + ".onnx")):
        expected_sha = ((prepared["source_contract"].get("artifacts") or {}).get(role) or {}).get("sha256")
        if role == "source_onnx": expected_sha = expected_sha or prepared["source_contract"].get("source_model_sha256")
        _require(expected_sha and copied_files.get(path, {}).get("sha256") == expected_sha, "original semantic/source artifact staging differs: " + role)
    return {"kind": "full_ort_child_failed_before_accelerator_entry", "source_archive": str(archive),
            "supported_route": "original normal no-plan ORT CPU bootstrap; recognized pre-failure helper code, not arbitrary future code",
            "source_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(), "software_bundle_sha256": hashlib.sha256(tar_bytes).hexdigest(),
            "suite_sha256": hashlib.sha256(suitesource).hexdigest(), "case_sha256": hashlib.sha256(case_source).hexdigest(),
            "failure_line": sessions[0].lineno, "first_accelerator_constructor_line": min(node.lineno for node in constructors),
            "observed_case_count": len(cases), "semantic_completion_evidence": semantic_safety,
            "measured_child_accelerator_entry_proven_absent_by_source_and_trace": True,
            "measured_sdk_call_count": None, "raw_counters_rewritten": False, "old_attempt_validated": False}


def _native_numerical_case(run_dir, prepared_path, setup_id, source_archive=None):
    """Join an ordinary failed numerical dispatch without changing its result."""
    from ..workflow import hardware_acceptance_numerical as numerical
    run = Path(run_dir).expanduser()
    _require(run.is_absolute() and run.is_dir() and not run.is_symlink()
             and all(not path.is_symlink() for path in run.parents), "unsafe failed workflow path")
    run = run.resolve()
    snapshots = {}
    def read(path):
        path = Path(path)
        _require(path.is_absolute() and path.is_file() and not path.is_symlink()
                 and all(not parent.is_symlink() for parent in path.parents), "unsafe numerical evidence: " + str(path))
        data = path.read_bytes()
        _require(path not in snapshots or snapshots[path] == data, "numerical evidence changed while joining: " + str(path))
        snapshots[path] = data
        value = json.loads(snapshots[path])
        _require(isinstance(value, dict), "numerical evidence must be an object")
        return value
    def binding(value):
        _require(isinstance(value, Mapping), "numerical evidence binding missing")
        path = Path(str(value.get("path") or ""))
        result = read(path)
        _require(hashlib.sha256(snapshots[path]).hexdigest() == value.get("sha256"), "numerical evidence SHA differs: " + str(path))
        return path, result
    prepared_path = Path(prepared_path).expanduser()
    prepared = read(prepared_path)
    identity = prepared.get("identity") or {}
    _require(isinstance(identity, Mapping) and set(identity) == {"model", "setup_id", "case", "backend"}
             and all(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", value) for value in identity.values())
             and identity["setup_id"] == setup_id
             and ((identity["backend"] == "deepx_to_trt" and re.fullmatch(r"b[0-9]+", identity["case"]))
                  or (identity["backend"] == "native_full_deepx" and identity["case"] == "full")), "unsupported Native numerical identity")
    base = run / "native" / "__".join(identity[key] for key in ("model", "setup_id", "case"))
    directory = base / "numerical"
    _require(prepared_path == base / "PREPARED.json" and prepared.get("output") == str(base)
             and prepared.get("schema") == "onnx-splitpoint/hardware-acceptance-native-plan/v1"
             and prepared.get("deepx_runtime_mode") == "A4", "prepared job is not the canonical Native numerical cell")
    result = read(base / "NUMERICAL_RESULT.json")
    _require(result.get("schema") == numerical.SCHEMA and result.get("identity") == identity
             and result.get("status") == "failed" and result.get("attempt_state") == "terminal"
             and result.get("fresh_hardware_executed") is True
             and all(type(result.get(key)) is int and result[key] == 0
                     for key in ("new_capture_count", "new_performance_processes"))
             and result.get("phase_matrix") == prepared.get("phase_matrix")
             and result.get("source_contract_sha256") == (prepared.get("source_contract") or {}).get("contract_sha256"),
             "numerical cell is not its original failed terminal attempt")
    for name in ("source_plan", "source_raw_report", "source_validation"):
        path = Path(str(prepared.get(name) or ""))
        read(path)
        _require(hashlib.sha256(snapshots[path]).hexdigest() == prepared.get(name + "_sha256"), "original numerical source differs: " + name)
    command, dispatch = read(directory / "COMMAND.json"), read(directory / "DISPATCH.json")
    expected = numerical._numerical_item(prepared)
    images = read(directory / "image_bundle/images.json").get("images")
    _require(isinstance(images, list) and 1 <= len(images) <= 32
             and all(isinstance(value, str) and value.startswith(expected["numerical_benchmark_set"] + "/resources/") for value in images)
             and len(set(images)) == len(images),
             "original staged numerical images missing")
    if identity["backend"] == "native_full_deepx":
        numerical._set_option(expected["argv"], "--image-map", json.dumps({identity["model"]: {"full": images[0]}}))
    else:
        numerical._set_option(expected["argv"], "--image", images[0])
    numerical._set_option(expected["argv"], "--deepx-numerical-images", expected["numerical_benchmark_set"] + "/resources/numerical/images.json")
    _require(command.get("argv") == expected["argv"] == dispatch.get("argv")
             and command.get("env") == expected["env"]
             and (command.get("subprocess_only_env") is True or (identity["backend"] == "native_full_deepx"
                  and "subprocess_only_env" not in command and "subprocess_only_env" not in expected))
             and command.get("remote_output_root") == expected["remote_output_root"]
             and command.get("result_json") == expected["result_json"]
             and type(dispatch.get("returncode")) is int
             and (dispatch["returncode"] in {0, 4} or (dispatch["returncode"] == 3 and identity["backend"] == "native_full_deepx" and source_archive)),
             "original closed numerical command or dispatch differs")
    argv = command["argv"]
    _require(command["env"].get("ONNX_SPLITPOINT_NATIVE_TRT_BUILD") == "0"
             and not any(value.startswith(("--build", "--force", "--energy-")) for value in argv), "numerical recovery cannot admit build or energy work")
    relative = Path(expected["result_json"]).relative_to(expected["remote_output_root"])
    raw_path, raw = binding(result.get("worker_report"))
    _require(raw_path == directory / "download" / Path(expected["remote_output_root"]).name / relative,
             "worker report is outside its original numerical download")
    safety = numerical.device_completion_safety(raw)
    lifecycle = safety.get("execution_lifecycle") or {}
    source_completion = None
    if identity["backend"] == "native_full_deepx" and dispatch["returncode"] == 3 and source_archive:
        source_completion = _full_ort_pre_runtime_proof(raw, prepared, directory, source_archive, read, snapshots)
    _require(source_completion is not None or (safety.get("device_completion_proven") is True and lifecycle.get("lifecycle_present") is True
             and (safety.get("pre_runtime_proven") is True or lifecycle.get("device_completion_proven") is True
                  or safety.get("independent_completion_proven") is True)),
             "Native SDK completion is independently unsafe or unproven")
    deployment = read(base / "RUNTIME_DEPLOYMENT.json")
    runner = "native_full_baseline_eval_runner.py" if identity["backend"] == "native_full_deepx" else "native_deepx_trt_e2e_from_benchmarkset.py"
    verified = [step for step in deployment.get("steps", []) if step.get("name") == "verify_remote_script_" + runner.removesuffix(".py")]
    _require(deployment.get("status") == "verified" and len(verified) == 1 and verified[0].get("rc") == 0
             and verified[0].get("sha256_ok") is True and re.fullmatch(r"[a-f0-9]{64}", str(verified[0].get("remote_sha256") or ""))
             and verified[0].get("expected_sha256") == verified[0]["remote_sha256"], "original deployed Native worker not verified")
    worker_sha = verified[0]["remote_sha256"]
    _require(prepared["remote_tool_dir"].rstrip("/") + "/scripts/" + runner in argv[:3], "original Native worker path differs")
    if identity["backend"] == "deepx_to_trt":
        _require(raw.get("runner_sha256") == worker_sha and raw.get("case") == identity["case"]
                 and raw.get("deepx_runtime_mode") == "A4", "worker identity or deployed bytes differ")
        if not safety.get("pre_runtime_proven"):
            _require(argv.count("--expected-runner-sha256") == 1
                     and argv[argv.index("--expected-runner-sha256") + 1] == worker_sha,
                     "completed worker differs from its original dispatch")
        row, contract = raw, raw.get("native_command_contract")
        if "--source-contract-sha256" in argv:
            _require(argv.count("--source-contract-sha256") == 1
                     and raw.get("source_contract_sha256") == argv[argv.index("--source-contract-sha256") + 1],
                     "raw worker source differs from original dispatch")
    else:
        rows = raw.get("rows") or []
        _require(len(rows) == 1 and isinstance(rows[0], Mapping), "Full numerical recovery requires one original worker row")
        row, contract = rows[0], rows[0].get("full_command_contract")
        _require(isinstance(contract, Mapping), "Full original worker contract missing")
    if contract is not None:
        _require(isinstance(contract, Mapping), "original worker command contract malformed")
        unsealed = dict(contract)
        declared = unsealed.pop("contract_sha256", None)
        _require(declared == hashlib.sha256(json.dumps(unsealed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
                 and all(contract.get(key) == value for key, value in identity.items())
                 and contract.get("runner_sha256") == worker_sha, "original worker command contract identity or source differs")
        source = prepared["source_contract"]
        for role in (("dxnn", "source_onnx") if identity["backend"] == "native_full_deepx" else ("dxnn", "engine")):
            previous, current = (source.get("artifacts") or {}).get(role) or {}, (contract.get("artifacts") or {}).get(role) or {}
            _require(source_completion is not None or (previous.get("sha256") and previous["sha256"] == current.get("sha256")), "original worker artifact differs: " + role)
        _require(source_completion is not None or contract.get("input_image_sha256") == source.get("input_image_sha256"), "original worker timing image differs")
    closure = result.get("owned_worker_closure") or {}
    diagnostic_path, diagnostic = binding(closure.get("diagnostic"))
    exact_command = "cd -- " + shlex.quote(prepared["remote_tool_dir"]) + " && " + shlex.join(argv)
    _require(diagnostic_path.parent == base / "diagnostics" and diagnostic_path.name.startswith("ssh_")
             and closure.get("proven") is True and diagnostic.get("command") == exact_command
             and type(diagnostic.get("returncode")) is int and diagnostic["returncode"] == dispatch["returncode"]
             and diagnostic.get("cleanup_proven") is True and diagnostic.get("target_id") == setup_id,
             "original owned worker cleanup does not match the exact dispatch")
    manifest, control = read(run / "run_manifest.json"), read(run / "jobs/workflow_control.json")
    artifact_closure = read(run / "reports/artifact_index_closure.json")
    session = control.get("session_id")
    _require(isinstance(session, str) and re.fullmatch(r"[a-f0-9]{32}", session)
             and manifest.get("run_id") == control.get("run_id") == artifact_closure.get("run_id") == run.name
             and manifest.get("run_dir") == str(run) and manifest.get("current_session_id") == artifact_closure.get("session_id") == session
             and control.get("state") == "finished" and manifest.get("status") in {"partial", "failed", "cancelled"}
             and artifact_closure.get("status") == "pass" and artifact_closure.get("workflow_status") == manifest["status"],
             "original numerical workflow is not durably terminal")
    index = read(run / "artifact_index.json")
    indexed_prepared, indexed_result, indexed_paths = [], [], set()
    _require(index.get("run_id") == run.name and isinstance(index.get("terminal_closure"), Mapping)
             and index["terminal_closure"].get("status") == "pass"
             and (index["terminal_closure"]).get("session_id") == session
             and isinstance(index.get("artifacts"), list), "normal terminal artifact index missing")
    for record in index["artifacts"]:
        _require(isinstance(record, Mapping), "normal artifact index record malformed")
        relative = Path(str(record.get("path") or ""))
        _require(not relative.is_absolute() and ".." not in relative.parts and str(relative) not in indexed_paths,
                 "normal artifact index path is unsafe or repeated")
        indexed_paths.add(str(relative))
        path = run / relative
        _require(path.is_file() and not path.is_symlink() and all(not parent.is_symlink() for parent in path.parents),
                 "normal indexed evidence unavailable")
        data = path.read_bytes()
        _require(path not in snapshots or snapshots[path] == data, "indexed numerical evidence changed while joining: " + str(relative))
        _require(record.get("sha256") == "sha256:" + hashlib.sha256(data).hexdigest()
                 and record.get("size_bytes") == len(data), "normal terminal artifact hash differs: " + str(relative))
        snapshots[path] = data
        if relative.parts[0] == "reports" and relative.suffix == ".json":
            value = json.loads(data)
            if isinstance(value, Mapping) and value.get("identity") == identity:
                if value.get("schema") == prepared["schema"]:
                    indexed_prepared.append(value)
                if value.get("schema") == numerical.SCHEMA:
                    indexed_result.append(value)
    _require(len(indexed_prepared) == len(indexed_result) == 1 and indexed_prepared[0] == prepared
             and all(indexed_result[0].get(key) == value for key, value in result.items()),
             "normal indexed prepared/result projection differs from original numerical evidence")
    _require(source_completion is None or "evaluation_workflow.log" in indexed_paths, "source proof staging trace is not normally indexed")
    journal_root = run / "jobs/remote_process_leases"
    config = read(journal_root / session / "controller.resources.json")
    _require(config.get("run_id") == run.name and config.get("run_dir") == str(run) and config.get("session_id") == session
             and type(config.get("parent_pid")) is int and config["parent_pid"] > 0
             and str(config.get("parent_start_ticks") or "").isdigit(), "original numerical controller identity differs")
    _require(not list(journal_root.rglob("*.remote-lease.json")), "unresolved remote lease descriptor")
    requests = list(journal_root.rglob("*.resource-request.json"))
    _require(bool(requests), "original numerical resource journal missing")
    for path in requests:
        request = read(path)
        reply = read(path.with_name(path.name.replace(".resource-request.json", ".resource-reply.json")))
        _require(request.get("purpose") in {"transfer", "prepare"} and request.get("phase") == "FINALIZING"
                 and request.get("process_started") is False and reply.get("state") == "RELEASED"
                 and request.get("run_id") == run.name and request.get("setup_id") == setup_id
                 and _bound(reply, request, ("run_id", "session_id", "operation_id", "token")),
                 "numerical journal has capture, STOP, active or unresolved work")
    return {"run": run, "directory": directory, "prepared": prepared, "identity": identity,
            "session": session, "config": config, "snapshots": snapshots, "safety": safety,
            "source_completion_proof": source_completion,
            "journal_inventory": {str(path.relative_to(journal_root)) for path in journal_root.rglob("*") if path.is_file()},
            "worker_sha256": worker_sha, "raw_sha256": hashlib.sha256(snapshots[raw_path]).hexdigest()}


def recover_native_numerical_resource(*, run_dir, prepared_path, setup_id, operator,
                                      evidence_output, registry_path=None, source_archive=None):
    """Recover a proven stopped numerical worker; never validate its result."""
    _require(isinstance(operator, str) and bool(operator.strip()), "operator is required")
    case = _native_numerical_case(run_dir, prepared_path, setup_id, source_archive=source_archive)
    run, config, snapshots = case["run"], case["config"], case["snapshots"]
    output = Path(evidence_output).expanduser()
    _require(output.is_absolute() and not output.is_symlink() and all(not path.is_symlink() for path in output.parents)
             and not output.resolve().is_relative_to(run), "recovery evidence must be outside the failed run")
    from ..energy.config import default_registry_path, _read_hardware_registry_unlocked
    from ..platform_power import host_config_from_setup
    from ..workflow.execution_binding import physical_dut_key
    registry = Path(registry_path or default_registry_path()).expanduser()
    _require(registry.is_file() and not registry.is_symlink(), "normal hardware registry unavailable")
    before_registry = registry.read_bytes()
    payload, _ = _read_hardware_registry_unlocked(registry)
    setups = [value for value in payload.get("hardware_setups", []) if value.get("id") == setup_id]
    _require(len(setups) == 1, "registered setup missing or ambiguous")
    setup, prepared = setups[0], case["prepared"]
    host = host_config_from_setup(setup)
    fields = setup.get("host") if isinstance(setup.get("host"), Mapping) else {}
    resource = physical_dut_key({"remote": {"host": host.host,
        "physical_host_id": setup.get("physical_host_id") or fields.get("physical_host_id")}})
    _require(resource == physical_dut_key(prepared["target"])
             and resource in config.get("setups", {}).get(setup_id, []), "registered physical DUT differs from numerical job")
    saved_ssh = (prepared.get("target_binding") or {}).get("ssh")
    _require(saved_ssh == {"host": host.host, "port": int(host.port), "user": host.user},
             "registered SSH target differs from original numerical job")
    owner = {"run_id": run.name, "session_id": case["session"], "purpose": "explicit_native_numerical_recovery"}
    run_lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    lease = EvaluationRunLock.for_resource(resource, owner=owner)
    run_lock.acquire_for_recovery()
    try:
        lease.acquire_for_recovery()
        try:
            previous_run = run_lock.previous_owner
            _require(not os.path.lexists(run_lock.quarantine_path) and previous_run.get("run_id") == run.name
                     and previous_run.get("session_id") == case["session"] and previous_run.get("pid") == config["parent_pid"]
                     and previous_run.get("state") == "released", "original numerical run owner is not terminal")
            if output.is_file() and not os.path.lexists(lease.quarantine_path):
                prior = RemoteProcessLeaseJournal._read_json_exact(output)
                recovery = lease.previous_owner.get("resource_recovery") or {}
                _require(prior.get("status") == "released" and prior.get("kind") == "native_numerical"
                         and prior.get("run_dir") == str(run) and prior.get("identity") == case["identity"]
                         and prior.get("raw_sha256") == case["raw_sha256"] and prior.get("operator") == operator
                         and prior.get("resource") == resource and recovery.get("evidence_path") == str(output)
                         and recovery.get("run_id") == run.name and lease.previous_owner.get("state") == "released",
                         "missing fence or foreign numerical recovery receipt")
                return {"status": "already_released", "resource": resource, "evidence_path": str(output), "measurements_started": 0}
            _require(not output.exists() and lease.quarantine_path.is_file() and not lease.quarantine_path.is_symlink(), "missing fence or recovery output already exists")
            original_owner = lease.previous_owner
            fence = lease.quarantine_path.read_bytes()
            quarantine = json.loads(fence)
            _require(quarantine.get("state") == "quarantined" and quarantine.get("schema") == "onnx-splitpoint/evaluation-run-lock-quarantine"
                     and quarantine.get("run_dir") == str(lease.run_dir) and quarantine.get("run_id") == lease.run_dir.name
                     and quarantine.get("owner") == {"reason": _NATIVE_NUMERICAL_REASON, "evidence": str(case["directory"])}
                     and original_owner.get("physical_resource") == resource and original_owner.get("run_id") == run.name
                     and original_owner.get("purpose") == "hardware_acceptance_native_numerical"
                     and original_owner.get("pid") == config["parent_pid"] and original_owner.get("hostname") == socket.gethostname()
                     and original_owner.get("state") == "running", "foreign numerical fence or owner")
            observed = observe_resource_owners(setup, local_owners=[{"pid": config["parent_pid"],
                "start_time_ticks": config["parent_start_ticks"]}], diagnostics_dir=output.parent / "diagnostics")
            _require(observed.get("ok") is True, "active or unobservable numerical/local/remote owner")
            _require(registry.read_bytes() == before_registry
                     and all(not path.is_symlink() and path.read_bytes() == data for path, data in snapshots.items())
                     and not list((run / "jobs/remote_process_leases").rglob("*.remote-lease.json"))
                     and {str(path.relative_to(run / "jobs/remote_process_leases"))
                          for path in (run / "jobs/remote_process_leases").rglob("*") if path.is_file()} == case["journal_inventory"]
                     and lease._read_owner() == original_owner and lease.quarantine_path.read_bytes() == fence,
                     "numerical evidence, registry, lease or fence changed during observation")
            receipt = {"schema": "onnx-splitpoint/physical-resource-recovery", "status": "verified_native_numerical",
                "kind": "native_numerical", "run_id": run.name, "run_dir": str(run), "session_id": case["session"],
                "identity": case["identity"], "resource": resource, "setup_id": setup_id, "operator": operator,
                "evidence_path": str(output), "original_owner": original_owner, "original_quarantine": quarantine,
                "original_quarantine_bytes": fence.decode("utf-8"), "ownership_observation": observed,
                "original_evidence": [{"path": str(path), "sha256": hashlib.sha256(data).hexdigest()} for path, data in snapshots.items()],
                "device_completion_evidence": case["safety"], "worker_sha256": case["worker_sha256"], "raw_sha256": case["raw_sha256"],
                "independent_source_completion_proof": case["source_completion_proof"],
                "raw_counters_rewritten": False, "old_attempt_validated": False, "measurements_started": 0,
                "verified_at": datetime.now(timezone.utc).isoformat()}
            output.parent.mkdir(parents=True, exist_ok=True)
            _write(output, receipt)
            receipt.update(status="released", released_at=datetime.now(timezone.utc).isoformat())
            try:
                lease.resolve_quarantine(expected_owner=original_owner, expected_quarantine=fence,
                    recovery={"status": "verified_native_numerical", "resource": resource, "run_id": run.name,
                              "evidence_path": str(output)}, commit_evidence=lambda: _write(output, receipt))
            except BaseException:
                receipt.update(status="blocked")
                try:
                    _write(output, receipt)
                except BaseException:
                    pass
                raise
            return {"status": "released", "resource": resource, "evidence_path": str(output), "measurements_started": 0}
        finally:
            lease.release(write_released_state=False)
    finally:
        run_lock.release(write_released_state=False)


def _rest_capture_journal(run, session_id, attempt_dir):
    """Read the existing explicit YOLO rest attempt, without relocating its journal."""
    attempt = Path(attempt_dir).expanduser()
    _require(attempt.is_absolute() and attempt.is_dir() and not attempt.is_symlink()
             and attempt.name == session_id, "invalid rest attempt directory")
    attempt = attempt.resolve()
    read = RemoteProcessLeaseJournal._read_json_exact
    evidence = {}
    for name in ("production-start.json", "terminal-status.json", "cleanup-status.json"):
        path = attempt / name
        _require(path.is_file() and not path.is_symlink(), "missing exact rest evidence: " + name)
        evidence[path] = read(path)
    start, terminal, cleanup = (evidence[attempt / name] for name in (
        "production-start.json", "terminal-status.json", "cleanup-status.json"))
    plan_path = Path(str(start.get("plan") or ""))
    _require(plan_path.is_absolute() and plan_path.is_file() and not plan_path.is_symlink(),
             "rest source plan unavailable")
    plan = read(plan_path)
    _require(Path(str(plan.get("augmentation_run") or "")).resolve() == run
             and Path(str(plan.get("output_root") or "")).resolve() / "attempts" / session_id == attempt,
             "rest source plan/run/attempt binding mismatch")
    _require(type(start.get("pid")) is int and start["pid"] > 0
             and start.get("automatic_restart") is False
             and terminal.get("status") in {"stopped", "completed"}
             and type(terminal.get("exitcode")) is int
             and terminal.get("automatic_restart") is False
             and _time(start.get("started_at")) <= _time(terminal.get("finished_at")),
             "rest parent is not exactly terminal")
    exit_path = attempt / "exitcode.txt"
    _require(exit_path.is_file() and not exit_path.is_symlink()
             and exit_path.read_text().strip() == str(terminal["exitcode"]),
             "rest exit status is not durable")
    _require(type(cleanup.get("active_owned_remote_operations")) is int
             and cleanup["active_owned_remote_operations"] == 0
             and type(cleanup.get("owned_local_processes")) is int
             and cleanup["owned_local_processes"] == 0
             and cleanup.get("foreign_processes_stopped") is False
             and _time(start.get("started_at")) <= _time(cleanup.get("at"))
             <= _time(terminal.get("finished_at")),
             "rest owned process cleanup incomplete")
    directory = attempt / "remote-process-leases"
    _require(directory.is_dir() and not directory.is_symlink()
             and directory.resolve().parent == attempt, "invalid existing rest journal")
    evidence[plan_path] = plan
    evidence[exit_path] = terminal["exitcode"]
    return directory, start, evidence


def _rest_evidence_unchanged(evidence):
    read = RemoteProcessLeaseJournal._read_json_exact
    return all(not path.is_symlink() and path.is_file() and (
        path.read_text().strip() == str(value) if path.name == "exitcode.txt" else read(path) == value
    ) for path, value in evidence.items())


def _bounded_energy_journal(run, session_id, attempt_dir):
    """Bind an external bounded energy parent's existing terminal evidence.

    The first adapter version stored its journal directly in the run. Resumes
    keep separate attempt directories. Neither layout needs invented workflow
    manifests or relocation of historical ownership records.
    """
    attempt = Path(attempt_dir).expanduser()
    _require(attempt.is_absolute() and attempt.is_dir() and not attempt.is_symlink()
             and attempt.resolve().is_relative_to(run), "invalid bounded energy attempt")
    attempt = attempt.resolve()
    read = RemoteProcessLeaseJournal._read_json_exact
    evidence = {}
    for name in ("production-start.json", "production-result.json", "cleanup-status.json"):
        path = attempt / name
        _require(path.is_file() and not path.is_symlink(), "missing bounded energy evidence: " + name)
        evidence[path] = read(path)
    start, result, cleanup = (evidence[attempt / name] for name in (
        "production-start.json", "production-result.json", "cleanup-status.json"))
    generic = not (run / "energy_parent_plan.json").exists()
    plan_path = run / ("parent_plan.json" if generic else "energy_parent_plan.json")
    prepared_path = run / "PREPARED.json"
    _require(all(p.is_file() and not p.is_symlink() for p in (plan_path, prepared_path)),
             "bounded energy plan unavailable")
    plan, prepared = read(plan_path), read(prepared_path)
    import hashlib
    _require(plan.get("schema") == ("revision3/generic-energy-parent/v1" if generic else
                                   "yolov7-full-fix/bounded-energy-parent-v1")
             and plan.get("run_dir") == str(run) and prepared.get("ok") is True
             and prepared.get("parent_plan_sha256" if generic else "plan_sha256")
             == hashlib.sha256(plan_path.read_bytes()).hexdigest(),
             "bounded energy prepared plan binding mismatch")
    if generic:
        resumed = attempt != run
        _require(not resumed or (attempt.parent == run / "resume_attempts"
                 and re.fullmatch(r"[0-9a-f]{32}", attempt.name)), "invalid Generic energy attempt layout")
        _require(start.get("attempt_dir") == str(attempt)
                 and start.get("selected_identity") == plan.get("selected_identity")
                 and plan.get("point_count") == 1 and plan.get("capture_count") == 3
                 and plan.get("duration_s") == 60., "Generic energy parent scope mismatch")
        expected_argv = [plan.get("python_executable"), "-B", (plan.get("adapter") or {}).get("path"),
                         "worker", "--run-dir", str(run)]
        if resumed:
            expected_argv += ["--attempt-dir", str(attempt)]
        _require(all(isinstance(arg, str) and arg for arg in expected_argv)
                 and start.get("argv") == expected_argv, "Generic energy worker argv mismatch")
        normal_binding = plan.get("normal_energy_plan") or {}
        normal_path = Path(str(normal_binding.get("path") or ""))
        _require(normal_path.is_absolute() and normal_path.is_file() and not normal_path.is_symlink()
                 and normal_path.resolve().is_relative_to(run)
                 and hashlib.sha256(normal_path.read_bytes()).hexdigest() == normal_binding.get("sha256")
                 and start.get("normal_plan_sha256") == normal_binding.get("sha256"),
                 "Generic original energy plan binding mismatch")
        normal = read(normal_path)
        evidence[normal_path] = normal
        if resumed:
            worker_path = attempt / "worker-plan.json"
            _require(worker_path.is_file() and not worker_path.is_symlink(), "Generic worker plan missing")
            worker_binding = read(worker_path)
            worker_plan = attempt / "measurement/generic_energy_plan.json"
            _require(worker_binding.get("path") == str(worker_plan) and worker_plan.is_file()
                     and not worker_plan.is_symlink()
                     and hashlib.sha256(worker_plan.read_bytes()).hexdigest() == worker_binding.get("sha256"),
                     "Generic resumed energy plan binding mismatch")
            original = normal
            normal = read(worker_plan)
            _require(all(normal.get(key) == original.get(key) for key in (
                "identity", "campaign_row_id", "campaign_source_id", "campaign_budget_file",
                "campaign_limits", "performance_result", "performance_result_sha256",
                "require_environment_observation", "environment_run_id")), "Generic resumed campaign scope changed")
            evidence.update({worker_path: worker_binding, worker_plan: normal})
        identity = plan.get("selected_identity") or {}
        import json
        expected_row = json.dumps(["generic_completed:" + str(identity.get("comparison_backend") or ""),
            identity.get("model_id"), identity.get("case_id"), identity.get("setup_id")], separators=(",", ":"))
        _require(normal.get("schema") == "onnx-splitpoint/generic-completed-task-energy-plan/v1"
                 and normal.get("identity") == identity and normal.get("campaign_row_id") == expected_row
                 and normal.get("campaign_budget_file") == plan.get("campaign_budget_file")
                 and normal.get("campaign_source_id") == str((plan.get("target", {}).get("energy") or {})
                     .get("urecs_address") or "").strip().lower()
                 and normal.get("require_environment_observation") is True,
                 "Generic energy campaign identity mismatch")
    _require(start.get("run_id") == run.name and start.get("session_id") == session_id
             and type(start.get("parent_pid")) is int and start["parent_pid"] > 0
             and type(result.get("returncode")) is int
             and _time(start.get("started_at")) <= _time(result.get("finished_at"))
             <= _time(cleanup.get("at")), "bounded energy parent is not exactly terminal")
    _require(type(cleanup.get("owned_remote_active")) is int and cleanup["owned_remote_active"] == 0
             and type(cleanup.get("owned_local_active")) is int and cleanup["owned_local_active"] == 0
             and cleanup.get("foreign_processes_stopped") is False,
             "bounded energy owned process cleanup incomplete")
    directory = attempt / "remote-process-leases"
    _require(directory.is_dir() and not directory.is_symlink()
             and directory.resolve().parent == attempt, "invalid bounded energy journal")
    evidence.update({plan_path: plan, prepared_path: prepared})
    return directory, {**start, "pid": start["parent_pid"]}, evidence



def _bounded_generic_capture(run, parent_attempt, evidence):
    """Use the bound normal plan's actual measurement root, including refreshes."""
    read = RemoteProcessLeaseJournal._read_json_exact
    parent = evidence.get(run / "parent_plan.json")
    _require(parent is not None, "exact Generic prepared parent required")
    normal_path = (Path(read(parent_attempt / "worker-plan.json")["path"])
                   if parent_attempt != run else Path(parent["normal_energy_plan"]["path"]))
    _require(normal_path in evidence and normal_path.name == "generic_energy_plan.json"
             and normal_path.parent.is_relative_to(run), "bound Generic normal plan missing")
    return normal_path, evidence[normal_path]


def _interrupted_generic_capture(run, parent_attempt, request, reply, evidence):
    """Bind an owner-lost physical capture; its old source close remains false."""
    import hashlib, json
    from ..energy.task_budget import source_completion
    read = RemoteProcessLeaseJournal._read_json_exact
    _require(request.get("phase") == "ACQUIRING" and request.get("sequence") == 2
             and request.get("process_started") is None
             and reply.get("state") == "STOP" and reply.get("reason") == "controller_resource_owner_lost",
             "not the exact interrupted Generic acquire stage")
    normal_path, normal = _bounded_generic_capture(run, parent_attempt, evidence)
    capture = Path(str(request.get("attempt_dir") or ""))
    _require(capture.is_absolute() and capture.is_dir() and not capture.is_symlink()
             and capture.resolve() == capture and capture.name == "run_000"
             and capture.parent.parent == normal_path.parent,
             "interrupted capture differs from bound normal measurement root")
    calls = [call for call in normal.get("calls", [])
             if capture.parent.name == "capture_" + str(call.get("repetition_index"))]
    _require(len(calls) == 1, "interrupted capture has no unique original normal call")
    call = calls[0]; started_path = normal_path.parent / (capture.parent.name + "_started.json")
    summary_path = capture / "energy_summary.json"; stdout_path = capture / "collector_stdout.log"
    for path in (started_path, summary_path, stdout_path):
        _require(path.is_file() and not path.is_symlink(), "original interrupted collector evidence missing")
    started, summary = read(started_path), read(summary_path)
    _require(started.get("argv") == call.get("argv")
             and started.get("command_sha256") == call.get("command_sha256")
             and started.get("plan_sha256") == hashlib.sha256(normal_path.read_bytes()).hexdigest()
             and started.get("aggregate_absent_before_execution") is True
             and type(started.get("started_ns")) is int
             and summary.get("collector_started") is True
             and summary.get("collector_pid") == request.get("collector_pid")
             and summary.get("status") == "collector_running",
             "interrupted physical collector/start binding differs")
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    completion = source_completion(capture)
    _require(not completion["verified"] and len(completion["go"]) == 1
             and len(completion["source_close"]) == 1 and completion["source_close"][0][2] == "false"
             and "Fast Firmware first packet monotonic_ns=" in text
             and "Running command: " + str(capture / "energy_command.sh") in text
             and "Writer Flushed" in text,
             "physical start/workload/local collector shutdown is not durable")
    paths = {**evidence, started_path: started, summary_path: summary}
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (*paths, stdout_path)}
    return {"collector_started": True, "workload_started": True,
            "source_completion_verified": False, "source_completion": json.loads(json.dumps(completion)),
            "capture_directory": str(capture), "evidence_sha256": hashes}


def _interrupted_native_capture(run, parent_attempt, request, reply, evidence):
    """Bind the Native owner's cancelled acquire without inventing old cleanup."""
    import hashlib, json
    from ..energy.recovery_accounting import verify_original_chain_scope
    from ..energy.task_budget import source_completion
    read = RemoteProcessLeaseJournal._read_json_exact
    parent = evidence.get(run / "energy_parent_plan.json")
    _require(parent is not None and request.get("phase") == "ACQUIRING"
             and request.get("sequence") == 2 and request.get("process_started") is None
             and reply.get("state") == "STOP"
             and reply.get("reason") == "controller_resource_owner_lost",
             "not the exact interrupted Native acquire stage")
    start = evidence[parent_attempt / "production-start.json"]
    argv = start.get("argv")
    _require(isinstance(argv, list) and all(isinstance(arg, str) for arg in argv),
             "Native original worker argv missing")

    def option(name):
        indices = [i for i, arg in enumerate(argv) if arg == name or arg.startswith(name + "=")]
        _require(len(indices) == 1, "Native original option must be unique: " + name)
        index = indices[0]
        if argv[index].startswith(name + "="):
            return argv[index].split("=", 1)[1]
        _require(index + 1 < len(argv) and not argv[index + 1].startswith("--"),
                 "Native original option missing: " + name)
        return argv[index + 1]

    expected_output = run / "measurement"
    if parent_attempt != run:
        _require(parent_attempt.parent == run / "resume_attempts"
                 and re.fullmatch(r"[0-9a-f]{32}", parent_attempt.name),
                 "invalid interrupted Native parent attempt layout")
        expected_output = expected_output / "resume_attempts" / parent_attempt.name
    budget_path = Path(str(parent.get("campaign_budget_file") or ""))
    _require(option("--out-dir") == str(expected_output)
             and option("--summary") == parent.get("summary")
             and option("--campaign-budget-file") == str(budget_path)
             and budget_path.is_absolute() and budget_path.is_file() and not budget_path.is_symlink(),
             "interrupted Native worker differs from its prepared parent")
    capture = Path(str(request.get("attempt_dir") or ""))
    _require(capture.is_absolute() and capture.is_dir() and not capture.is_symlink()
             and capture.resolve() == capture and re.fullmatch(r"run_[0-9]{3}", capture.name)
             and capture.is_relative_to(expected_output / "measurements"),
             "interrupted Native capture differs from its normal measurement root")
    budget = read(budget_path)
    matches = [(key, task, chain) for key, task in budget.get("tasks", {}).items()
               for chain in task.get("chains", []) if chain.get("run_directory") == str(capture)]
    _require(len(matches) == 1, "interrupted Native capture has no unique original chain")
    key, task, chain = matches[0]
    identity = json.loads(key)
    _require(identity in parent.get("capture_identities", [])
             and len(identity) == 4 and identity[3] == request.get("setup_id")
             and _bound(chain, request, ("owner_pid", "owner_start_ticks"))
             and chain.get("collector_started") is True and chain.get("valid") is not True
             and chain.get("source_completion_verified") is not True
             and task.get("source_id") == str(((parent.get("targets") or {}).get(identity[3], {})
                                                .get("energy") or {}).get("urecs_address") or "").strip().lower(),
             "interrupted Native task or ownership differs from its prepared parent")
    scope_paths = verify_original_chain_scope(run=run, capture=capture, campaign_row=key, read=read)
    summary_path, stdout_path = capture / "energy_summary.json", capture / "collector_stdout.log"
    paths = [*scope_paths, summary_path, stdout_path, capture / "collector_stderr.log"]
    _require(all(path.is_file() and not path.is_symlink() for path in paths),
             "original interrupted Native collector evidence missing")
    summary = read(summary_path)
    collector = summary.get("collector_cmd")
    _require(summary.get("collector_started") is True
             and summary.get("collector_pid") == request.get("collector_pid")
             and summary.get("status") == "collector_running"
             and isinstance(collector, list)
             and collector.count("-c=" + str(capture / "energy_command.sh")) == 1
             and collector.count("--address=" + task["source_id"]) == 1,
             "interrupted Native physical collector/start binding differs")
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    completion = source_completion(capture)
    _require(not completion["verified"] and len(completion["go"]) == 1
             and len(completion["source_close"]) == 1 and completion["source_close"][0][2] == "false"
             and completion["go"][0][1] == completion["source_close"][0][1]
             and "Fast Firmware first packet monotonic_ns=" in text
             and "Running command: " + str(capture / "energy_command.sh") in text
             and "Writer Flushed" in text,
             "Native physical start/workload/local collector shutdown is not durable")
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (*evidence, *paths)}
    return {"collector_started": True, "workload_started": True,
            "source_completion_verified": False, "source_completion": json.loads(json.dumps(completion)),
            "capture_directory": str(capture), "measurement_root": str(expected_output / "measurements"),
            "campaign_row": key, "evidence_sha256": hashes}


def _interrupted_bounded_capture(run, parent_attempt, request, reply, evidence):
    proof = (_interrupted_generic_capture if run / "parent_plan.json" in evidence
             else _interrupted_native_capture)
    return proof(run, parent_attempt, request, reply, evidence)


def _partial_workflow_capture_evidence(run, session_id, request, reply, config):
    """Bind a terminal ordinary acceptance run to its actual failed capture.

    ``partial`` is a finished workflow outcome, but it is also too broad to
    admit without its exact session, final result and unchanged source STOP.
    All physical-owner and cleanup checks still apply in the caller.
    """
    read = RemoteProcessLeaseJournal._read_json_exact
    paths = [run / "run_manifest.json", run / "jobs/workflow_control.json",
             run / "reports/hardware_acceptance_results.json", run / "energy/energy_task_budget.json",
             Path(request["attempt_dir"]) / "energy_summary.json"]
    _require(all(path.is_file() and not path.is_symlink() for path in paths),
             "partial workflow terminal capture evidence missing")
    evidence = {path: read(path) for path in paths}
    manifest, control, result, budget, summary = (evidence[path] for path in paths)
    sessions = [row for row in manifest.get("execution_sessions", [])
                if isinstance(row, Mapping) and row.get("session_id") == session_id]
    _require(manifest.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
             and manifest.get("run_id") == run.name and manifest.get("run_dir") == str(run)
             and manifest.get("status") == "partial" and manifest.get("current_session_id") == session_id
             and manifest.get("workflow_stages") == ["hardware_acceptance"]
             and len(sessions) == 1 and sessions[0].get("status") == "partial"
             and _time(sessions[0].get("started_at")) <= _time(sessions[0].get("finished_at"))
             <= datetime.now(timezone.utc), "partial workflow session is not exactly terminal")
    _require(control.get("schema") == "onnx-splitpoint/workflow-control"
             and control.get("run_id") == run.name and control.get("session_id") == session_id
             and control.get("state") == "finished"
             and _time(sessions[0]["finished_at"]) <= _time(control.get("updated_at"))
             <= datetime.now(timezone.utc), "partial workflow control is not finished")
    admission = manifest.get("hardware_acceptance") or {}
    _require(result.get("schema") == admission.get("schema") == "onnx-splitpoint/hardware-acceptance/v1"
             and result.get("run_id") == run.name and result.get("status") == "partial"
             and result.get("complete") is False and result.get("source_campaign_resumed") is False
             and result.get("new_dataset_quality_started") is False
             and type(result.get("matrix_cells")) is int and result["matrix_cells"] > 0
             and result["matrix_cells"] == admission.get("cell_count")
             and type(result.get("runtime_paths")) is int and result["runtime_paths"] > 0
             and result["runtime_paths"] == admission.get("runtime_path_count")
             and isinstance(result.get("rows"), list) and len(result["rows"]) == result["runtime_paths"],
             "partial workflow final acceptance result is incomplete or foreign")
    key = summary.get("campaign_row")
    import json
    try:
        identity = json.loads(key)
    except (TypeError, ValueError):
        identity = None
    _require(isinstance(identity, list) and len(identity) == 4 and identity[3] == request["setup_id"],
             "partial workflow capture identity unavailable")
    variant = "generic_completed" if str(identity[0]).startswith("generic_completed:") else "native"
    rows = [row for row in result["rows"] if isinstance(row, Mapping)
            and (row.get("model_id"), row.get("case_id"), row.get("setup_id"), row.get("runtime_variant"))
            == (identity[1], identity[2], identity[3], variant)]
    _require(len(rows) == 1 and rows[0].get("status") == "failed"
             and rows[0].get("fresh_hardware_executed") is True
             and type(rows[0].get("new_capture_count")) is int and rows[0]["new_capture_count"] > 0
             and _time(rows[0].get("finished_at")) <= _time(sessions[0]["finished_at"]),
             "partial workflow failed result does not bind the actual capture")
    task = (budget.get("tasks") or {}).get(key) or {}
    chains = [chain for chain in task.get("chains", [])
              if chain.get("run_directory") == request["attempt_dir"]]
    sources = [resource.removeprefix("source:") for resource in config.get("setups", {}).get(request["setup_id"], [])
               if resource.startswith("source:")]
    _require(len(sources) == 1 and task.get("source_id") == sources[0]
             and task.get("stop_reason") == reply["reason"]
             and (budget.get("sources", {}).get(sources[0]) or {}).get("stop_reason") == reply["reason"]
             and len(chains) == 1 and chains[0].get("collector_started") is True
             and chains[0].get("finished") is True and chains[0].get("valid") is False
             and chains[0].get("source_completion_verified") is False
             and _bound(chains[0], request, ("owner_pid", "owner_start_ticks"))
             and summary.get("campaign_run_id") == run.name and summary.get("collector_started") is True
             and summary.get("collector_pid") == request["collector_pid"]
             and summary.get("status") == "collector_failed"
             and summary.get("final_energy_gate_reasons") == ["capture_release_failed"]
             and (summary.get("capture_release_error") or {}).get("detail") == reply["reason"],
             "partial workflow original invalid source chain differs")
    from ..energy.task_budget import source_completion
    completion = source_completion(Path(request["attempt_dir"]))
    _require(completion["verified"] is False
             and json.dumps(chains[0].get("source_completion"), sort_keys=True) == json.dumps(completion, sort_keys=True),
             "partial workflow original source completion changed")
    return evidence


def recover_capture_resources(*, run_dir, session_id, operation_id, setup_id,
                              operator_evidence=None, retry_authorization=None, registry_path=None,
                              rest_attempt_dir=None, bounded_energy_attempt_dir=None):
    """Recover exactly one four-resource capture STOP, under all four flocks.

    Exactly one explicit release ground is required. Evidence and original
    fences are retained in the existing resource reply.
    A repeated identical call is a no-op; a new or foreign quarantine blocks.
    Normal acquire never calls this function and still rejects quarantines.
    """
    _require((operator_evidence is None) != (retry_authorization is None),
             "provide exactly one operator action or new-attempt authorization")
    supplied = Path(run_dir).expanduser()
    _require(not supplied.is_symlink() and supplied.is_dir(), "invalid run directory")
    run = supplied.resolve()
    for value in (session_id, operation_id):
        _require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value)
                 and value not in {".", ".."}, "invalid session/operation")
    rest_evidence = {}
    _require(rest_attempt_dir is None or bounded_energy_attempt_dir is None,
             "only one external parent layout is allowed")
    if bounded_energy_attempt_dir is not None:
        directory, rest_start, rest_evidence = _bounded_energy_journal(
            run, session_id, bounded_energy_attempt_dir)
    elif rest_attempt_dir is None:
        directory = run / "jobs" / "remote_process_leases" / session_id
        _require(directory.is_dir() and not directory.is_symlink()
                 and directory.resolve().is_relative_to(run), "invalid existing session directory")
    else:
        directory, rest_start, rest_evidence = _rest_capture_journal(run, session_id, rest_attempt_dir)
    read = RemoteProcessLeaseJournal._read_json_exact
    config_path = directory / "controller.resources.json"
    request_path = directory / (operation_id + ".resource-request.json")
    reply_path = directory / (operation_id + ".resource-reply.json")
    config, request, reply = read(config_path), read(request_path), read(reply_path)
    identity = {"run_id": run.name, "session_id": session_id, "operation_id": operation_id}
    _require(_bound(config, identity, ("run_id", "session_id"))
             and config.get("run_dir") == str(run), "controller session binding mismatch")
    if rest_evidence:
        _require(config.get("parent_pid") == rest_start["pid"], "rest parent PID binding mismatch")
    interrupted = bool(bounded_energy_attempt_dir is not None
                       and request.get("phase") == "ACQUIRING"
                       and reply.get("reason") == "controller_resource_owner_lost")
    interrupted_proof = (_interrupted_bounded_capture(run, Path(bounded_energy_attempt_dir).resolve(),
                         request, reply, rest_evidence) if interrupted else None)
    _require(_bound(request, identity, identity) and request.get("setup_id") == setup_id
             and request.get("purpose", "capture") == "capture"
             and (interrupted or (request.get("phase") == "FINALIZING" and request.get("process_started") is True))
             and re.fullmatch(r"[0-9a-f]{32}", str(request.get("token") or "")),
             "capture operation binding mismatch")
    _require(_bound(reply, request, (*identity, "token")) and reply.get("state") == "STOP"
             and (interrupted or reply.get("reason") == "campaign_source_completion_unresolved"),
             "only the exact source-completion STOP is recoverable")
    attempt = Path(str(request.get("attempt_dir") or ""))
    _require(attempt.is_absolute() and not attempt.is_symlink()
             and attempt.resolve().is_relative_to(run), "capture attempt outside run")
    generic_plan = rest_evidence.get(run / "parent_plan.json")
    if generic_plan is not None:
        normal_path, _ = _bounded_generic_capture(run, Path(bounded_energy_attempt_dir).resolve(), rest_evidence)
        _require(attempt.resolve().is_relative_to(normal_path.parent)
                 and request.get("setup_id") == generic_plan["selected_identity"]["setup_id"],
                 "Generic capture differs from its exact parent attempt")
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
    workflow_evidence = {}
    terminal_workflow = None
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
        _require(_rest_evidence_unchanged(rest_evidence), "rest terminal evidence changed")
        previous_recovery = reply.get("recovery") or {}
        _require(not previous_recovery or (
            _bound(previous_recovery, identity, identity)
            and previous_recovery.get("resources") == resources), "foreign recovery metadata")
        if interrupted and previous_recovery:
            _require(previous_recovery.get("interrupted_capture") == interrupted_proof,
                     "recorded interrupted evidence changed")
        if bounded_energy_attempt_dir is None and not rest_evidence:
            manifest = read(run / "run_manifest.json")
            if manifest.get("status") == "partial" or previous_recovery.get("terminal_workflow"):
                workflow_evidence = _partial_workflow_capture_evidence(run, session_id, request, reply, config)
                terminal_workflow = {"run_id": run.name, "session_id": session_id,
                    "status": "partial", "state": "finished",
                    "campaign_row": workflow_evidence[attempt / "energy_summary.json"]["campaign_row"]}
                _require(not previous_recovery or previous_recovery.get("terminal_workflow") == terminal_workflow,
                         "recorded partial workflow terminal evidence changed")
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
                     and payload.get("reason") == reply["reason"]
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
        if retry_authorization is not None:
            evidence_key, other_key = "retry_authorization", "operator_evidence"
            evidence = _retry_authorization(retry_authorization, source, stopped_at)
        else:
            evidence_key, other_key = "operator_evidence", "retry_authorization"
            evidence = _operator_evidence(operator_evidence, source, stopped_at)
        _require(not previous_recovery or (previous_recovery.get(evidence_key) == evidence
                                          and other_key not in previous_recovery),
                 "repeat must refer to the same recorded release evidence")
        if previous_recovery.get("status") == "released" and all(
                not lease.quarantine_path.exists() and lease.previous_owner.get("state") == "released"
                for _, lease in leases):
            if workflow_evidence:
                _require(_partial_workflow_capture_evidence(run, session_id, request, reply, config) == workflow_evidence,
                         "partial workflow source completion changed during replay")
            return {"status": "already_released", **identity, "resources": resources,
                    "measurements_started": 0}

        if not interrupted:
            cleanup = read(attempt / "collector_stdout.log.cleanup.json")
            _require(cleanup.get("owned_tree_quiescent") is True and cleanup.get("process_started") is True
                     and _bound(cleanup, request, ("collector_pid", "collector_start_ticks")),
                     "exact durable collector cleanup is missing")
        # Owner-lost cancellation predates the old sidecar writer. The NEW
        # receipt binds terminal parent own0, original flushed collector output
        # and the fresh owner observation below; no old cleanup proof is added.
        if bounded_energy_attempt_dir is None:
            manifest = read(run / "run_manifest.json")
            _require(manifest.get("run_id") == run.name and (rest_evidence or workflow_evidence or manifest.get("status") in
                     {"failed", "cancelled", "completed", "success", "done"}), "parent workflow is not terminal")
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
                 and read(reply_path) == reply and read(config_path) == config
                 and _rest_evidence_unchanged(rest_evidence) and _rest_evidence_unchanged(workflow_evidence),
                 "registry or ownership changed during observation")
        if interrupted:
            _require(_interrupted_bounded_capture(run, Path(bounded_energy_attempt_dir).resolve(),
                     request, reply, rest_evidence) == interrupted_proof, "interrupted original capture changed")
        if workflow_evidence:
            _require(_partial_workflow_capture_evidence(run, session_id, request, reply, config) == workflow_evidence,
                     "partial workflow source completion changed during observation")
        recovery = {**identity, "resources": resources, evidence_key: evidence,
                    "original_resources": originals, "ownership_observation": observed,
                    "status": "releasing", "started_at": datetime.now(timezone.utc).isoformat(),
                    "measurements_started": 0, "old_attempt_validated": False}
        if terminal_workflow:
            recovery["terminal_workflow"] = terminal_workflow
        if interrupted:
            recovery["interrupted_capture"] = interrupted_proof
        # Commit original evidence and the explicit release ground to the STOP reply;
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
