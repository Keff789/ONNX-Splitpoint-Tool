"""Explicit recovery of one stopped capture's existing physical resource locks.

No measurement or device command is sent. An actual operator intervention and
authorization of one new workflow despite an unconfirmed source end are distinct
release grounds. Neither changes the old STOP, result or energy budgets.
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
            _require(manifest.get("run_id") == run.name and (rest_evidence or manifest.get("status") in
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
                 and _rest_evidence_unchanged(rest_evidence),
                 "registry or ownership changed during observation")
        if interrupted:
            _require(_interrupted_bounded_capture(run, Path(bounded_energy_attempt_dir).resolve(),
                     request, reply, rest_evidence) == interrupted_proof, "interrupted original capture changed")
        recovery = {**identity, "resources": resources, evidence_key: evidence,
                    "original_resources": originals, "ownership_observation": observed,
                    "status": "releasing", "started_at": datetime.now(timezone.utc).isoformat(),
                    "measurements_started": 0, "old_attempt_validated": False}
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
