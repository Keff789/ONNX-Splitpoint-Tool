"""Explicit recovery of a finite Native energy action, without legacy phases.

The physical recovery CLI starts no measurement. A later, live parent must use
the normal run-lock callback below and the original campaign budget. Neither
step validates the failed capture or rewrites its STOP, counters or limits.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re


def _read(path):
    from ..remote.process_lease import RemoteProcessLeaseJournal
    return RemoteProcessLeaseJournal._read_json_exact(Path(path))


def _require(condition, reason):
    if not condition:
        raise ValueError("finite Native energy recovery: " + reason)


def _option(argv, flag):
    matches = [i for i, value in enumerate(argv) if value == flag]
    _require(len(matches) == 1 and matches[0] + 1 < len(argv), "one exact " + flag + " required")
    value = argv[matches[0] + 1]
    _require(isinstance(value, str) and value and not value.startswith("--"), "missing " + flag)
    return value


def _descriptor(value):
    _require(isinstance(value, dict), "file descriptor missing")
    path = Path(str(value.get("path") or ""))
    _require(path.is_absolute() and path.resolve() == path and path.is_file()
             and not path.is_symlink(), "regular absolute evidence file required")
    _require(path.stat().st_size == value.get("size_bytes")
             and hashlib.sha256(path.read_bytes()).hexdigest() == value.get("sha256"),
             "bound evidence changed: " + str(path))
    return path


def finite_native_energy_journal(run, session_id, attempt):
    """Read the actual ACTION/start/result/uppercase-cleanup layout unchanged."""
    from ..remote.resource_recovery import _time
    run, attempt = Path(run).resolve(), Path(attempt).resolve()
    _require(attempt == run or (attempt.parent == run / "recovery_attempts"
             and re.fullmatch(r"[0-9a-f]{32}", attempt.name)), "foreign attempt layout")
    paths = {name: attempt / name for name in (
        "ACTION.json", "production-start.json", "production-result.json", "CLEANUP_STATUS.json")}
    evidence = {path: _read(path) for path in paths.values()}
    action, start, result, cleanup = (evidence[paths[name]] for name in paths)
    argv = action.get("argv")
    _require(action.get("schema") == "six-pack-normal-native-energy-action-v1"
             and action.get("engine_build_allowed") is False
             and action.get("automatic_controller_restart") is False
             and type(action.get("runs")) is int and action["runs"] == 3
             and type(action.get("duration_s")) in (int, float) and action["duration_s"] == 60
             and action.get("output_root") == str(run)
             and (attempt == run or action.get("attempt_dir") == str(attempt))
             and isinstance(argv, list) and all(isinstance(value, str) and value for value in argv)
             and len(argv) > 3 and argv[1] == "-B", "action identity/scope differs")
    export = Path(str(action.get("source_export") or ""))
    freeze_path = _descriptor(action.get("source_freeze")); freeze = _read(freeze_path)
    _require(export.is_absolute() and export.resolve() == export
             and freeze.get("source_export_root") == str(export)
             and freeze.get("pushed") is True
             and freeze.get("remote_branch_head") == freeze.get("commit")
             and argv[2] == str(export / "scripts/run_native_producer_energy_from_summary.py"),
             "published controller source binding differs")
    evidence[freeze_path] = freeze
    for item in action.get("input_descriptors", []):
        _descriptor(item)
    offline_path = _descriptor(action.get("normal_offline_plan")); offline = _read(offline_path)
    evidence[offline_path] = offline
    rows = offline.get("rows")
    _require(isinstance(rows, list) and len(rows) == 1 and not offline.get("excluded_rows"),
             "finite action must bind one normal admitted row")
    row = rows[0]; target = action.get("target") or {}; setup = target.get("id")
    identity = [row.get(key) for key in ("backend", "model", "case", "setup_id")]
    _require(all(isinstance(value, str) and value for value in identity)
             and identity[3] == setup
             and action.get("normal_quality_admission") == row.get("energy_quality_admission")
             and row.get("energy_quality_admission", {}).get("admission_scope") == "native_dataset_energy"
             and row["energy_quality_admission"].get("diagnostic_only") is False,
             "normal standalone identity/admission differs")
    for flag, expected in (("--duration-s", "60"), ("--runs", "3"), ("--frames", "0"),
                           ("--warmup", "0"), ("--environment-run-id", run.name)):
        _require(_option(argv, flag) == expected, "finite scientific scope differs: " + flag)
    _require(all(argv.count(flag) == 1 for flag in (
        "--require-environment-observation", "--require-runtime-work-units",
        "--require-command-window-alignment")), "normal command-window guards missing")
    budget = Path(_option(argv, "--campaign-budget-file"))
    output = Path(_option(argv, "--out-dir"))
    _require(budget == run / "energy_task_budget.json"
             and output.is_absolute() and output.resolve() == output and output.is_relative_to(attempt)
             and _option(argv, "--campaign-max-retries") == "2"
             and _option(argv, "--campaign-max-transport-failures") == "20",
             "original campaign budget/output binding differs")
    _require(start.get("session_id") == session_id and start.get("argv") == argv
             and start.get("actual_source_commit") == freeze.get("commit")
             and start.get("engine_build_allowed") is False
             and start.get("common_resources") == action.get("resource_keys")
             and type(start.get("parent_pid")) is int and start["parent_pid"] > 0
             and type(result.get("returncode")) is int
             and _time(start.get("at")) <= _time(result.get("finished_at"))
             <= _time(cleanup.get("finished_at"))
             and cleanup.get("session_id") == session_id
             and type(cleanup.get("active_owned_remote_operations")) is int
             and cleanup["active_owned_remote_operations"] == 0
             and type(cleanup.get("owned_local_active")) is int and cleanup["owned_local_active"] == 0
             and cleanup.get("foreign_processes_stopped") is False,
             "exact terminal parent/owned cleanup missing")
    directory = attempt / "remote-process-leases"
    _require(directory.is_dir() and not directory.is_symlink(), "original journal missing")
    return directory, {**start, "pid": start["parent_pid"]}, evidence


def finite_native_predecessor(run, attempt):
    """Bind one original failed capture to its actual normal plan and budget."""
    from ..native_command_contract import canonical_json_sha256
    from ..energy.task_budget import source_completion
    run, attempt = Path(run).resolve(), Path(attempt).resolve()
    start = _read(attempt / "production-start.json")
    directory, _, evidence = finite_native_energy_journal(run, start["session_id"], attempt)
    replies = [_read(path) for path in directory.glob("*.resource-reply.json")]
    stopped = [reply for reply in replies if reply.get("state") != "RELEASED"]
    _require(len(stopped) == 1, "one exact terminal source STOP required")
    reply = stopped[0]; request = _read(directory / (reply["operation_id"] + ".resource-request.json"))
    cfg = _read(directory / "controller.resources.json")
    action = evidence[attempt / "ACTION.json"]; argv = action["argv"]
    offline = evidence[Path(action["normal_offline_plan"]["path"])]["rows"][0]
    identity = [offline[key] for key in ("backend", "model", "case", "setup_id")]
    _require(all(reply.get(key) == request.get(key) for key in ("run_id", "session_id", "operation_id", "token"))
             and request.get("run_id") == run.name and request.get("session_id") == start["session_id"]
             and request.get("setup_id") == identity[3] and request.get("purpose") == "capture"
             and request.get("phase") == "FINALIZING" and request.get("process_started") is True
             and reply.get("state") == "STOP" and reply.get("reason") == "campaign_source_completion_unresolved"
             and cfg.get("run_id") == run.name and cfg.get("run_dir") == str(run)
             and cfg.get("session_id") == start["session_id"] and cfg.get("parent_pid") == start["parent_pid"],
             "source STOP/controller identity differs")
    capture = Path(str(request.get("attempt_dir") or "")); output = Path(_option(argv, "--out-dir"))
    plan_path = output / "plan/native_producer_energy_plan.json"; plan = _read(plan_path)
    rows = plan.get("rows")
    _require(isinstance(rows, list) and len(rows) == 1
             and [rows[0].get(key) for key in ("backend", "model", "case", "setup_id")] == identity,
             "actual core plan differs from finite action identity")
    row = rows[0]; base = Path(str(row.get("measurement_output_base_dir") or ""))
    _require(capture.is_absolute() and capture.resolve() == capture and capture.is_dir()
             and not capture.is_symlink() and base.is_relative_to(output)
             and capture.is_relative_to(base), "capture outside its actual normal measurement")
    contract_path = Path(row["command_contract_file"]); contract = _read(contract_path)
    digest = row.get("successful_command_contract_sha256")
    _require(hashlib.sha256(contract_path.read_bytes()).hexdigest() == row.get("command_contract_file_sha256")
             and contract.get("contract_sha256") == digest
             and canonical_json_sha256({k: v for k, v in contract.items() if k != "contract_sha256"}) == digest
             and digest == offline.get("successful_command_contract_sha256"), "exact Native command changed")
    preflight_path = capture / "preflight/preflight_evidence.json"; preflight = _read(preflight_path)
    attestation_path = capture / "preflight/preflight_attestation.json"; attestation = _read(attestation_path)
    _require(preflight.get("ok") is True and preflight.get("rc") == 0
             and preflight.get("expected_command_contract_sha256") == digest
             and attestation.get("ok") is True and attestation.get("command_contract_sha256") == digest
             and bool(preflight.get("nonce")) and preflight["nonce"] == attestation.get("nonce")
             and hashlib.sha256(attestation_path.read_bytes()).hexdigest() == preflight.get("attestation_file_sha256"),
             "actual preflight/Native identity differs")
    budget_path = run / "energy_task_budget.json"; budget = _read(budget_path)
    row_id = json.dumps(identity, separators=(",", ":")); task = budget.get("tasks", {}).get(row_id, {})
    chains = [chain for chain in task.get("chains", []) if chain.get("run_directory") == str(capture)]
    source_id = str(action["target"]["energy"]["urecs_address"]).strip().lower()
    _require(budget.get("campaign") is True and task.get("source_id") == source_id and len(chains) == 1
             and chains[0].get("finished") is True and chains[0].get("valid") is False
             and chains[0].get("collector_started") is True and chains[0].get("source_completion_verified") is False
             and all(chains[0].get(key) == request.get(key) for key in ("owner_pid", "owner_start_ticks"))
             and task.get("limits") == {"max_chains": 9, "max_transport_failures": 20, "max_retries": 2}
             and budget.get("max_transport_failures") == 20
             and not source_completion(capture)["verified"], "consumed invalid original budget chain differs")
    evidence.update({plan_path: plan, contract_path: contract, preflight_path: preflight,
                     attestation_path: attestation})
    return directory, request, reply, cfg, evidence


def recover_finite_run_fence(run, output, session, lock, fence, failed_attempt):
    """Run-lock acquire callback for the LIVE authorized replacement parent.

    Call only after the explicit physical recovery CLI. This returns permission
    to the normal lock implementation; it never deletes a fence itself.
    """
    from ..energy.task_budget import install_authorized_source_reentry
    from ..process_control import _proc_start_time
    from ..remote.resource_recovery import _rest_evidence_unchanged, _time
    from .energy_resume import released_recovery_owner, write
    from .run_control import EvaluationRunLock
    run, output, failed_attempt = map(lambda p: Path(p).resolve(), (run, output, failed_attempt))
    _require(output.parent == run / "recovery_attempts" and re.fullmatch(r"[0-9a-f]{32}", output.name)
             and not (output / "production-start.json").exists(), "fresh bounded replacement attempt required")
    directory, request, reply, cfg, evidence = finite_native_predecessor(run, failed_attempt)
    cleanup = evidence[failed_attempt / "CLEANUP_STATUS.json"]; owner = lock.previous_owner
    _require(lock.run_dir == run and fence.get("schema") == "onnx-splitpoint/evaluation-run-lock-quarantine"
             and fence.get("run_dir") == str(run) and fence.get("run_id") == run.name
             and fence.get("owner") == {"reason": "six_pack_native_energy_cleanup_unproven", "pending": 0,
                                       "errors": cleanup["errors"]}
             and owner.get("purpose") == "six_pack_native_energy" and owner.get("run_id") == run.name
             and owner.get("run_dir") == str(run)
             and owner.get("pid") == cfg["parent_pid"]
             and (owner.get("session_id") == request["session_id"]
                  or ("session_id" not in owner and failed_attempt == run))
             and _time(owner.get("acquired_at")) <= _time(evidence[failed_attempt / "production-start.json"]["at"])
             and cleanup.get("cleanup_proven") is False,
             "run fence is not the exact terminal finite parent")
    recovery = reply.get("recovery") or {}; resources = recovery.get("resources") or []
    identity = {key: request[key] for key in ("run_id", "session_id", "operation_id")}
    _require(recovery.get("status") == "released" and len(resources) == len(set(resources)) == 4
             and all(recovery.get(key) == value for key, value in identity.items())
             and not list(directory.glob("*.remote-lease.json")), "physical source recovery incomplete")
    # Bind the existing one-workflow authorization to this exact replacement,
    # while allowing its controller release and output paths to advance.
    import yaml
    authorization = recovery.get("retry_authorization") or {}
    profile_path = Path(str(authorization.get("new_workflow_profile") or ""))
    profile = yaml.safe_load(profile_path.read_text())
    old_action = evidence[failed_attempt / "ACTION.json"]
    new_action_path = output / "ACTION.json"; new_action = _read(new_action_path)
    old_argv, new_argv = old_action["argv"], new_action.get("argv")
    _require(isinstance(profile, dict) and profile.get("schema") == "six-pack-native-energy-retry-v1"
             and profile.get("run_dir") == str(run) and profile.get("failed_attempt") == str(failed_attempt)
             and profile.get("historical_operation") == request["operation_id"]
             and profile.get("setup_id") == request["setup_id"]
             and profile.get("source") == str(old_action["target"]["energy"]["urecs_address"]).strip().lower()
             and profile.get("same_original_budget") == str(run / "energy_task_budget.json")
             and profile.get("unchanged_limits") == {"max_chains": 9, "max_transport_failures": 20, "max_retries": 2}
             and profile.get("source_completion_remains_unproven") is True
             and profile.get("action_file") == str(new_action_path), "authorized replacement profile differs")
    _require(new_action.get("schema") == old_action["schema"]
             and new_action.get("output_root") == str(run) and new_action.get("attempt_dir") == str(output)
             and new_action.get("failed_attempt") == str(failed_attempt)
             and all(new_action.get(key) == old_action.get(key) for key in (
                 "engine_build_allowed", "automatic_controller_restart", "runs", "duration_s", "target",
                 "normal_quality_admission", "native_remote_source_epoch", "resource_keys"))
             and isinstance(new_argv, list) and len(new_argv) == len(old_argv),
             "replacement action scope differs")
    freeze = _read(_descriptor(new_action.get("source_freeze")))
    _require(freeze.get("pushed") is True and freeze.get("commit") == freeze.get("remote_branch_head")
             and freeze.get("source_export_root") == new_action.get("source_export")
             and new_argv[2] == str(Path(new_action["source_export"]) / "scripts/run_native_producer_energy_from_summary.py"),
             "replacement published source differs")
    expected = list(old_argv); expected[2] = new_argv[2]
    old_output = _option(old_argv, "--out-dir")
    expected[expected.index("--out-dir") + 1] = str(output / Path(old_output).relative_to(failed_attempt))
    if "--completed-row-validator" in expected:
        index = expected.index("--completed-row-validator") + 1
        expected[index] = expected[index].replace(str(failed_attempt) + "/", str(output) + "/")
    _require(new_argv == expected, "replacement scientific/runtime argv changed")
    profile_bytes = profile_path.read_bytes(); new_action_bytes = new_action_path.read_bytes()
    for pid, ticks in ((cfg["parent_pid"], cfg["parent_start_ticks"]),
                       (request["owner_pid"], request["owner_start_ticks"]),
                       (request["collector_pid"], request["collector_start_ticks"])):
        _require(type(pid) is int and pid > 0 and str(ticks).isdigit()
                 and _proc_start_time(pid) != ticks, "original owner still active/unbound")
    with ExitStack() as held:
        for resource in resources:
            lease = EvaluationRunLock.for_resource(resource, owner={})
            lease.acquire_for_recovery(); held.callback(lease.release, write_released_state=False)
            released_recovery_owner(lease, resource, identity, resources, recovery)
        _require(_rest_evidence_unchanged(evidence), "original terminal evidence changed")
        _require(profile_path.read_bytes() == profile_bytes and new_action_path.read_bytes() == new_action_bytes,
                 "replacement authority changed under lock")
        grant = install_authorized_source_reentry(run / "energy_task_budget.json",
            capture_request=request, recovery_reply=reply,
            new_parent={"run_id": run.name, "session_id": session, "pid": os.getpid(),
                        "start_ticks": _proc_start_time(os.getpid())})
        write(output / "source-budget-handoff.json", grant)
        write(output / "runlock-recovery.json", {"original_fence": fence, "original_owner": owner,
            "physical_recovery_reply": reply, "old_attempt_validated": False,
            "measurements_started": 0})
    return True


def main(argv=None):
    """Explicit physical recovery only; no new parent or hardware is started."""
    from ..remote.resource_recovery import recover_capture_resources
    from .run_control import EvaluationRunLock
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--failed-attempt", required=True)
    parser.add_argument("--authorization", required=True,
                        help="Actual normal post-STOP one-attempt retry authorization JSON")
    parser.add_argument("--registry", default=None)
    args = parser.parse_args(argv)
    run, attempt = Path(args.run_dir).resolve(), Path(args.failed_attempt).resolve()
    lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    lock.acquire_for_recovery()
    try:
        _, request, _, _, _ = finite_native_predecessor(run, attempt)
        result = recover_capture_resources(run_dir=run, session_id=request["session_id"],
            operation_id=request["operation_id"], setup_id=request["setup_id"], registry_path=args.registry,
            retry_authorization=_read(args.authorization), bounded_energy_attempt_dir=attempt)
        print(json.dumps(result, indent=2))
        return 0
    finally:
        lock.release(write_released_state=False)


if __name__ == "__main__":
    raise SystemExit(main())
