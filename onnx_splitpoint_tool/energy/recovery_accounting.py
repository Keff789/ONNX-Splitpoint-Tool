"""Reconcile a terminal capture whose resource STOP interrupted accounting.

Call while holding the run recovery lock. This only books negative evidence;
it never releases resource fences, authorizes reentry, or edits capture files.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .task_budget import _finished_chain, _reentry_require, source_completion


def reconcile_stopped_capture(budget_path, *, parent_attempt, operation_id, run_dir=None):
    from ..process_control import _proc_start_time
    from ..remote.resource_recovery import (_bounded_energy_journal, _interrupted_bounded_capture,
                                             _bounded_generic_capture)
    from ..remote.process_lease import RemoteProcessLeaseJournal
    from .collector import _read_workload_timing
    read = RemoteProcessLeaseJournal._read_json_exact
    path = Path(budget_path)
    _reentry_require(path.is_absolute() and path.is_file() and not path.is_symlink(), "regular budget required")
    run = Path(run_dir).resolve() if run_dir is not None else path.parent.resolve()
    attempt = Path(parent_attempt)
    start = read(attempt / "production-start.json")
    journal, _, terminal = _bounded_energy_journal(run, start["session_id"], attempt)
    _reentry_require(isinstance(operation_id, str) and operation_id.startswith("capture-")
                     and Path(operation_id).name == operation_id, "invalid capture operation")
    if run_dir is not None:
        plan = read(run / ("parent_plan.json" if run / "parent_plan.json" in terminal else "energy_parent_plan.json"))
        _reentry_require(plan.get("campaign_budget_file") == str(path.resolve()),
                         "shared budget is not bound by the prepared parent plan")
    request_path = journal / (operation_id + ".resource-request.json")
    reply_path = journal / (operation_id + ".resource-reply.json")
    cfg_path = journal / "controller.resources.json"
    request, reply, cfg = map(read, (request_path, reply_path, cfg_path))
    interrupted = bool(request.get("phase") == "ACQUIRING"
                       and reply.get("reason") == "controller_resource_owner_lost")
    physical = (_interrupted_bounded_capture(run, attempt, request, reply, terminal) if interrupted else None)
    _reentry_require(request.get("run_id") == run.name and request.get("session_id") == start["session_id"]
                     and request.get("operation_id") == operation_id
                     and all(reply.get(k) == request.get(k) for k in ("run_id", "session_id", "operation_id", "token"))
                     and request.get("purpose") == "capture"
                     and (interrupted or (request.get("phase") == "FINALIZING" and request.get("process_started") is True))
                     and reply.get("state") == "STOP"
                     and (interrupted or reply.get("reason") == "campaign_source_completion_unresolved")
                     and cfg.get("run_id") == run.name and cfg.get("run_dir") == str(run)
                     and cfg.get("session_id") == start["session_id"] and cfg.get("parent_pid") == start["parent_pid"],
                     "exact terminal source STOP required")
    for pid, ticks in ((cfg.get("parent_pid"), cfg.get("parent_start_ticks")),
                       (request.get("owner_pid"), request.get("owner_start_ticks")),
                       (request.get("collector_pid"), request.get("collector_start_ticks"))):
        _reentry_require(type(pid) is int and pid > 0 and str(ticks or "").isdigit()
                         and _proc_start_time(pid) != ticks, "capture owner still active or unbound")
    capture = Path(str(request.get("attempt_dir") or ""))
    capture_root = (Path(physical["measurement_root"]) if interrupted
                    and run / "energy_parent_plan.json" in terminal else attempt.resolve())
    _reentry_require(capture.is_absolute() and capture.is_dir() and not capture.is_symlink()
                     and capture.resolve().is_relative_to(capture_root), "capture outside bound parent measurement")
    paths = [*terminal, request_path, cfg_path, capture / "energy_summary.json",
             capture / "collector_stdout.log", capture / "collector_stderr.log"]
    timing = capture / "workload_timing.txt"
    if timing.exists():
        paths.append(timing)
    _reentry_require(all(p.is_file() and not p.is_symlink() for p in paths), "regular original evidence required")
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    # Recovery appends to the reply; its immutable STOP fields remain bound.
    stop = {k: v for k, v in reply.items() if k != "recovery"}
    summary = read(capture / "energy_summary.json")
    completion = source_completion(capture)
    _reentry_require(summary.get("collector_pid") == request["collector_pid"]
                     and summary.get("collector_started") is True
                     and summary.get("status") == "collector_running"
                     and not completion["verified"] and len(completion["source_close"]) == 1
                     and completion["source_close"][0][2] == "false", "not an interrupted negative capture")
    sources = [r.removeprefix("source:") for r in cfg["setups"][request["setup_id"]] if r.startswith("source:")]
    _reentry_require(len(sources) == 1, "unique physical source required")
    source_id = sources[0]
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        root = json.load(handle)
        matches = [(row, task, chain) for row, task in root["tasks"].items()
                   if task["source_id"] == source_id for chain in task["chains"]
                   if chain["run_directory"] == str(capture)]
        _reentry_require(len(matches) == 1, "unique original chain required")
        row, task, chain = matches[0]
        _reentry_require(json.loads(row)[3] == request["setup_id"]
                         and all(chain.get(k) == request[k] for k in ("owner_pid", "owner_start_ticks"))
                         and chain.get("collector_started") is True, "chain ownership mismatch")
        if run / "parent_plan.json" in terminal:
            scope_paths = verify_generic_chain_scope(run=run, attempt=attempt, capture=capture,
                campaign_row=row, terminal=terminal, read=read)
            hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scope_paths})
        elif run_dir is not None:
            scope_paths = verify_original_chain_scope(run=run, capture=capture, campaign_row=row, read=read)
            hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scope_paths})
        key = request["session_id"] + "/" + operation_id
        receipts = root.setdefault("failure_reconciliations", {})
        if key in receipts:
            receipt = receipts[key]
            _reentry_require(receipt["evidence_sha256"] == hashes and receipt["original_stop"] == stop
                             and receipt["booked_chain"] == chain, "reconciliation evidence changed")
            return json.loads(json.dumps(receipt))
        _reentry_require(chain.get("finished") is False and chain.get("valid") is not True,
                         "only an unbooked failure may be reconciled")
        entry = {**summary, "status": "collector_failed", "energy_efficiency_claim_eligible": False,
                 "final_energy_gate_status": "fail", "final_energy_gate_reasons": [reply["reason"]],
                 "workload_timing": _read_workload_timing(timing)}
        booked = _finished_chain(chain, entry, capture, campaign=True)
        if interrupted:
            booked["workload_started"] = physical["workload_started"]
            booked["transport_reasons"] = sorted(set(booked["transport_reasons"] + [reply["reason"]]))
        _reentry_require(booked["valid"] is False and booked["transport_reasons"]
                         and booked["source_completion_verified"] is False, "negative transport proof missing")
        receipt = {"schema": "onnx-splitpoint/stopped-capture-accounting-v1", "original_chain": copy.deepcopy(chain),
                   "booked_chain": copy.deepcopy(booked), "original_stop": stop, "evidence_sha256": hashes}
        if interrupted:receipt["interrupted_capture"] = physical
        source = root["sources"][source_id]
        continuous = source.get("continuous_admission")
        grant = source.get("authorized_reentry")
        if continuous is not None:
            expected = "campaign_source_interrupted_chain_unresolved" if interrupted else reply["reason"]
            _reentry_require(source["stop_reason"] in {continuous["original_source_stop_reason"], expected}
                             and continuous["source_stop_reason"] in {"", reply["reason"], expected}
                             and all(key == row and value in {reply["reason"], expected}
                                     for key,value in continuous["task_stop_reasons"].items()),
                             "different continuous source/task STOP cannot be reconciled")
            continuous["source_stop_reason"] = expected
            continuous["task_stop_reasons"][row] = expected
        elif grant is not None:
            _reentry_require(_proc_start_time(grant["new_parent"]["pid"]) != grant["new_parent"]["start_ticks"],
                             "authorized parent still active")
            if grant["chain"] == chain:
                receipt["previous_grant_chain"] = copy.deepcopy(grant["chain"])
                grant["chain"] = copy.deepcopy(booked)
            if not grant["source_stop_reason"]:
                grant["source_stop_reason"] = reply["reason"]
        elif source.get("unstarted_capture_reconciliation"):
            if not source.get("post_unstarted_capture_stop_reason"):
                source["post_unstarted_capture_stop_reason"] = reply["reason"]
        elif not source["stop_reason"]:
            source["stop_reason"] = reply["reason"]
        chain.update(booked)
        source["transport_failures"] = sum(bool(c.get("transport_reasons")) for t in root["tasks"].values()
                                           if t["source_id"] == source_id for c in t["chains"])
        receipts[key] = receipt
        _reentry_require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items()),
                         "original evidence changed during reconciliation")
        handle.seek(0)
        json.dump(root, handle, indent=2)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        return json.loads(json.dumps(receipt))



def verify_generic_chain_scope(*, run, attempt, capture, campaign_row, terminal, read):
    from ..remote.resource_recovery import _bounded_generic_capture
    normal_path, normal = _bounded_generic_capture(run, attempt, terminal)
    identity = normal["identity"]
    _reentry_require(campaign_row == normal["campaign_row_id"] == json.dumps([
        "generic_completed:" + identity["comparison_backend"], identity["model_id"],
        identity["case_id"], identity["setup_id"]], separators=(",", ":"))
        and capture.parent.parent == normal_path.parent,
        "Generic original chain scope differs from its bound normal plan")
    config_path = Path(normal["config_path"])
    _reentry_require(config_path.is_file() and not config_path.is_symlink()
                     and hashlib.sha256(config_path.read_bytes()).hexdigest() == normal["config_sha256"]
                     and read(config_path) == normal["config"], "Generic normal configuration changed")
    calls = [call for call in normal["calls"] if capture.parent.name == "capture_" + str(call["repetition_index"])]
    _reentry_require(len(calls) == 1, "Generic original capture call missing or ambiguous")
    argv = calls[0]["argv"]; flag = "--preflight-expected-command-contract-sha256"
    _reentry_require(argv.count(flag) == 1, "Generic original preflight digest missing")
    digest = argv[argv.index(flag) + 1]
    evidence_path = capture / "preflight/preflight_evidence.json"
    attestation_path = capture / "preflight/preflight_attestation.json"
    evidence, attestation = read(evidence_path), read(attestation_path)
    _reentry_require(evidence.get("ok") is True and evidence.get("rc") == 0
                     and evidence.get("expected_command_contract_sha256") == digest
                     and attestation.get("ok") is True and attestation.get("command_contract_sha256") == digest
                     and evidence.get("nonce") == attestation.get("nonce") and bool(evidence.get("nonce"))
                     and hashlib.sha256(attestation_path.read_bytes()).hexdigest() == evidence.get("attestation_file_sha256"),
                     "Generic actual preflight differs from its original bound task")
    return [normal_path, config_path, evidence_path, attestation_path]

def verify_original_chain_scope(*, run, capture, campaign_row, read):
    """Bind all four task-key fields to prepared input and actual preflight."""
    from ..native_command_contract import canonical_json_sha256
    identity = json.loads(campaign_row)
    _reentry_require(isinstance(identity, list) and len(identity) == 4
                     and all(isinstance(value, str) and value for value in identity),
                     "exact four-part original campaign identity required")
    parent = read(run / "energy_parent_plan.json")
    summary_path = Path(str(parent.get("summary") or ""))
    prepared_path = run / "plancheck" / "plan" / "native_producer_energy_plan.json"
    _reentry_require(summary_path.is_absolute() and summary_path.is_file()
                     and not summary_path.is_symlink() and prepared_path.is_file()
                     and not prepared_path.is_symlink()
                     and hashlib.sha256(summary_path.read_bytes()).hexdigest() == parent.get("summary_sha256"),
                     "original prepared summary binding missing or changed")
    fields = ("backend", "model", "case", "setup_id")
    prepared = [row for row in read(prepared_path).get("rows", [])
                if [row.get(key) for key in fields] == identity]
    originals = [row for row in read(summary_path).get("rows", [])
                 if [row.get(key) for key in fields] == identity]
    _reentry_require(len(prepared) == 1 and len(originals) == 1,
                     "campaign identity is not one exact original prepared task")
    row, original = prepared[0], originals[0]
    contract_path = Path(str(row.get("command_contract_file") or ""))
    workload_path = Path(str(row.get("command_file") or ""))
    _reentry_require(all(path.is_absolute() and path.is_file() and not path.is_symlink()
                         for path in (contract_path, workload_path)), "regular prepared command files required")
    contract = read(contract_path)
    digest = row.get("successful_command_contract_sha256")
    original_digest = original.get("full_command_contract_sha256") or original.get("native_command_contract_sha256")
    _reentry_require(isinstance(digest, str) and len(digest) == 64
                     and digest == original_digest == contract.get("contract_sha256")
                     and canonical_json_sha256({k: v for k, v in contract.items() if k != "contract_sha256"}) == digest
                     and hashlib.sha256(contract_path.read_bytes()).hexdigest() == row.get("command_contract_file_sha256")
                     and [contract.get(key) for key in fields] == identity,
                     "original task command contract identity or SHA differs")
    evidence_path = capture / "preflight" / "preflight_evidence.json"
    attestation_path = capture / "preflight" / "preflight_attestation.json"
    evidence, attestation = read(evidence_path), read(attestation_path)
    _reentry_require(evidence.get("ok") is True and evidence.get("rc") == 0
                     and evidence.get("expected_command_contract_sha256") == digest
                     and attestation.get("ok") is True and attestation.get("command_contract_sha256") == digest
                     and evidence.get("nonce") == attestation.get("nonce") and bool(evidence.get("nonce"))
                     and hashlib.sha256(attestation_path.read_bytes()).hexdigest() == evidence.get("attestation_file_sha256")
                     and hashlib.sha256(workload_path.read_text().strip().encode()).hexdigest() == evidence.get("workload_template_sha256"),
                     "actual preflight does not bind this original prepared task")
    return [summary_path, prepared_path, contract_path, workload_path, evidence_path, attestation_path]


def _authorized_derived_stop(root, source_id, row, chain, request, cfg, read, authorization_file):
    """Validate one explicit accounting authorization, never infer a release."""
    auth_path = Path(authorization_file)
    _reentry_require(auth_path.is_absolute() and auth_path.is_file() and not auth_path.is_symlink(),
                     "regular explicit zero-start authorization required")
    auth = read(auth_path)
    source = root["sources"][source_id]
    previous = root.get("unstarted_capture_reconciliations", {}).get(
        request["session_id"] + "/" + request["operation_id"])
    original_chain = previous["original_chain"] if previous else chain
    _reentry_require(auth.get("authorized") is True and auth.get("operation_id") == request["operation_id"]
                     and auth.get("source_id") == source_id and auth.get("setup_id") == request["setup_id"]
                     and auth.get("campaign_row") == row and auth.get("original_chain") == original_chain
                     and source.get("stop_reason") == "campaign_source_interrupted_chain_unresolved"
                     and not source.get("authorized_reentry")
                     and not source.get("post_unstarted_capture_stop_reason"),
                     "explicit derived zero-start scope or new source STOP differs")
    unresolved = [c for t in root["tasks"].values() if t["source_id"] == source_id for c in t["chains"]
                  if not c.get("finished") or (c.get("collector_started") and c.get("source_completion_verified") is not True)]
    _reentry_require(unresolved == ([chain] if previous is None else []),
                     "another unresolved physical source chain exists")
    allowed = auth.get("allowed_parents")
    _reentry_require(isinstance(allowed, dict) and bool(allowed), "explicit batch parents required")
    all_rows = []
    for directory, rows in allowed.items():
        parent = Path(directory)
        plan = read(parent / "energy_parent_plan.json")
        _reentry_require(parent.is_absolute() and not parent.is_symlink() and plan.get("run_dir") == str(parent)
                         and plan.get("phase") == "final" and not plan.get("allow_diagnostic_capture")
                         and plan.get("require_environment_observation") is True
                         and not plan.get("campaign_cohort")
                         and sorted(rows) == sorted(json.dumps(i, separators=(",", ":")) for i in plan["capture_identities"])
                         and list(plan["targets"]) == [request["setup_id"]]
                         and plan["targets"][request["setup_id"]]["energy"]["urecs_address"] == source_id,
                         "authorized prepared batch parent, policy or source changed")
        all_rows.extend(rows)
    _reentry_require(len(all_rows) == len(set(all_rows)) and row not in all_rows,
                     "batch scope repeats or includes historical Full capture")
    propagated = auth.get("propagated_task_stops", {})
    for key, reason in propagated.items():
        task = root["tasks"].get(key)
        _reentry_require(key in all_rows and reason == source["stop_reason"] and task is not None
                         and task["source_id"] == source_id and task["stop_reason"] == reason
                         and (previous is not None or task["chains"] == []),
                         "propagated task STOP is not this chainless refusal")
    limits = root["tasks"][row]["limits"]
    _reentry_require(limits == {"max_chains": 9, "max_retries": 2, "max_transport_failures": 20}
                     and root["max_transport_failures"] == 20,
                     "original physical budget differs")
    for key, task in root["tasks"].items():
        if task["source_id"] == source_id:
            _reentry_require(task["limits"] == limits and
                             (not task["stop_reason"] or key == row or key in propagated),
                             "another task STOP or physical budget remains binding")
    projection = {"original_stop_reason": source["stop_reason"], "original_limits": copy.deepcopy(limits),
                  "max_transport_failures": root["max_transport_failures"],
                  "allowed_parents": allowed, "propagated_task_stops": propagated,
                  "authorization_file": str(auth_path),
                  "authorization_sha256": hashlib.sha256(auth_path.read_bytes()).hexdigest()}
    return projection


def reconcile_unstarted_capture(budget_path, *, run_dir, parent_attempt, operation_id, apply=False,
                               authorization_file=None):
    """Book an exact terminal acquire failure with no collector or workload.

    This is negative accounting only: no resource fence or historical STOP is
    cleared, and the reserved chain continues to count against every limit.
    The default validates and returns the proposed receipt without writing.
    """
    from ..process_control import _proc_start_time
    from ..remote.resource_recovery import _bounded_energy_journal
    from ..remote.process_lease import RemoteProcessLeaseJournal
    read = RemoteProcessLeaseJournal._read_json_exact
    path, run, attempt = Path(budget_path), Path(run_dir), Path(parent_attempt)
    _reentry_require(path.is_absolute() and path.is_file() and not path.is_symlink()
                     and run.is_absolute() and run.is_dir() and not run.is_symlink(),
                     "regular explicit budget and run required")
    run = run.resolve()
    start = read(attempt / "production-start.json")
    journal, _, terminal = _bounded_energy_journal(run, start["session_id"], attempt)
    plan = read(run / "energy_parent_plan.json")
    _reentry_require(plan.get("campaign_budget_file") == str(path.resolve()),
                     "shared budget is not bound by the prepared parent plan")
    _reentry_require(isinstance(operation_id, str) and operation_id.startswith("capture-")
                     and Path(operation_id).name == operation_id, "invalid capture operation")
    request_path = journal / (operation_id + ".resource-request.json")
    reply_path = journal / (operation_id + ".resource-reply.json")
    cfg_path = journal / "controller.resources.json"
    request, reply, cfg = map(read, (request_path, reply_path, cfg_path))
    _reentry_require(request.get("run_id") == run.name and request.get("session_id") == start["session_id"]
                     and request.get("operation_id") == operation_id
                     and all(reply.get(k) == request.get(k) for k in ("run_id", "session_id", "operation_id", "token"))
                     and request.get("purpose") == "capture" and request.get("phase") == "UNUSED"
                     and request.get("sequence") == 2 and request.get("continuation") is None
                     and not any(k in request for k in ("collector_pid", "collector_start_ticks", "process_started"))
                     and reply.get("state") == "STOP" and reply.get("reason") == "campaign_source_completion_unresolved"
                     and cfg.get("run_id") == run.name and cfg.get("run_dir") == str(run)
                     and cfg.get("session_id") == start["session_id"] and cfg.get("parent_pid") == start["parent_pid"],
                     "exact unused acquire failure required")
    for pid, ticks in ((cfg.get("parent_pid"), cfg.get("parent_start_ticks")),
                       (request.get("owner_pid"), request.get("owner_start_ticks"))):
        _reentry_require(type(pid) is int and pid > 0 and str(ticks or "").isdigit()
                         and _proc_start_time(pid) != ticks, "capture owner still active or unbound")
    capture = Path(str(request.get("attempt_dir") or ""))
    _reentry_require(capture.is_absolute() and capture.is_dir() and not capture.is_symlink()
                     and capture.resolve().is_relative_to(attempt.resolve()), "capture outside parent attempt")
    summary = read(capture / "energy_summary.json")
    _reentry_require(summary.get("status") == "collector_starting"
                     and summary.get("collector_started") is False
                     and summary.get("collector_started_at_unix_ns") == 0
                     and summary.get("collector_pid") is None
                     and not summary.get("workload_started") and not summary.get("workload_timing"),
                     "capture summary does not prove zero start")
    storage = capture / "collector_storage"
    _reentry_require(storage.is_dir() and not storage.is_symlink() and not list(storage.iterdir()),
                     "collector storage is not empty")
    allowed = {"energy_summary.json", "workload_command.sh", "collector_command.txt", "energy_command.sh",
               "preflight", "collector_storage"}
    _reentry_require(all(p.name in allowed and not p.is_symlink() for p in capture.iterdir()),
                     "unexpected collector, GO, workload or timing evidence")
    paths = [*terminal, request_path, reply_path, cfg_path,
             *[p for p in capture.rglob("*") if p.is_file()]]
    _reentry_require(all(p.is_file() and not p.is_symlink() for p in paths)
                     and not any(p.is_symlink() for p in capture.rglob("*")), "regular original evidence required")
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    sources = [r.removeprefix("source:") for r in cfg["setups"][request["setup_id"]] if r.startswith("source:")]
    _reentry_require(len(sources) == 1, "unique physical source required")
    source_id = sources[0]
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        root = json.load(handle)
        source = root["sources"][source_id]
        _reentry_require((not source.get("stop_reason") or authorization_file is not None)
                         and not source.get("authorized_reentry"),
                         "actual source STOP or physical recovery remains binding")
        matches = [(row, task, chain) for row, task in root["tasks"].items()
                   if task["source_id"] == source_id for chain in task["chains"]
                   if chain["run_directory"] == str(capture)]
        _reentry_require(len(matches) == 1, "unique original chain required")
        row, task, chain = matches[0]
        _reentry_require(json.loads(row)[3] == request["setup_id"]
                         and all(chain.get(k) == request[k] for k in ("owner_pid", "owner_start_ticks"))
                         and chain.get("collector_started") is False
                         and not chain.get("workload_started")
                         and task["stop_reason"] == "task_execution_interrupted", "chain ownership or original STOP mismatch")
        scope_paths = verify_original_chain_scope(run=run, capture=capture, campaign_row=row, read=read)
        hashes.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in scope_paths})
        projection = None
        if source.get("stop_reason"):
            projection = _authorized_derived_stop(root, source_id, row, chain, request, cfg, read, authorization_file)
            # The corresponding real DUT operation must also prove no dispatch.
            dut = [(p, read(p)) for p in journal.glob("*.resource-request.json")
                   if read(p).get("attempt_dir") == str(capture) and read(p).get("purpose") == "dut_job"]
            _reentry_require(len(dut) == 1, "unique original DUT operation required")
            dut_path, dut_request = dut[0]
            dut_reply_path = dut_path.with_name(dut_path.name.replace("request", "reply"))
            dut_reply = read(dut_reply_path)
            _reentry_require(dut_request.get("phase") == "FINALIZING" and dut_request.get("process_started") is False
                             and all(dut_request.get(k) == request.get(k) for k in
                                     ("run_id", "session_id", "setup_id", "owner_pid", "owner_start_ticks"))
                             and dut_reply.get("state") == "RELEASED"
                             and all(dut_reply.get(k) == dut_request.get(k) for k in
                                     ("run_id", "session_id", "operation_id", "token")),
                             "original DUT dispatch or resource release unproven")
            for p in (dut_path, dut_reply_path, Path(projection["authorization_file"])):
                hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        key = request["session_id"] + "/" + operation_id
        receipts = root.setdefault("unstarted_capture_reconciliations", {})
        if key in receipts:
            receipt = receipts[key]
            _reentry_require(receipt["evidence_sha256"] == hashes and receipt["booked_chain"] == chain
                             and receipt["campaign_row"] == row and task.get("unstarted_capture_reconciliation") == key
                             and receipt.get("derived_source_stop") == projection,
                             "reconciliation evidence changed")
            return json.loads(json.dumps(receipt))
        _reentry_require(set(chain) == {"owner_pid", "owner_start_ticks", "logical_repeat", "run_directory",
                                       "preflight_requested", "collector_started", "finished"}
                         and chain.get("preflight_requested") is True
                         and chain.get("finished") is False
                         and not task.get("unstarted_capture_reconciliation"),
                         "only an exact untouched original reserved zero-start chain may be reconciled")
        # Keep every reserve/preflight count. There was no physical attempt to
        # classify as transport failure or to grant a physical source retry.
        booked = {**chain, "finished": True, "status": "capture_acquire_failed_before_start",
                  "valid": False, "workload_started": False, "transport_reasons": [],
                  "source_completion_verified": None, "source_completion": source_completion(capture)}
        receipt = {"schema": "onnx-splitpoint/unstarted-capture-accounting-v1",
                   "campaign_row": row, "source_id": source_id, "operation_id": operation_id,
                   "original_task_stop_reason": task["stop_reason"], "original_chain": copy.deepcopy(chain),
                   "booked_chain": copy.deepcopy(booked), "original_source": copy.deepcopy(source),
                   "original_reply": copy.deepcopy(reply), "evidence_sha256": hashes,
                   "physical_source_retry_authorized": False}
        if projection is not None:
            receipt["derived_source_stop"] = projection
        _reentry_require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
                         and not list(storage.iterdir()), "original evidence changed during reconciliation")
        if apply:
            chain.update(booked)
            receipts[key] = receipt
            task["unstarted_capture_reconciliation"] = key
            task["post_unstarted_capture_stop_reason"] = ""
            if projection is not None:
                source["unstarted_capture_reconciliation"] = key
                source["post_unstarted_capture_stop_reason"] = ""
                for propagated_row in projection["propagated_task_stops"]:
                    root["tasks"][propagated_row]["post_unstarted_capture_stop_reason"] = ""
            handle.seek(0)
            json.dump(root, handle, indent=2)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        return json.loads(json.dumps(receipt))
