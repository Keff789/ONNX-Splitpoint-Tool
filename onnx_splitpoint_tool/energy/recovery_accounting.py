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


def reconcile_stopped_capture(budget_path, *, parent_attempt, operation_id):
    from ..process_control import _proc_start_time
    from ..remote.resource_recovery import _bounded_energy_journal
    from ..remote.process_lease import RemoteProcessLeaseJournal
    from .collector import _read_workload_timing
    read = RemoteProcessLeaseJournal._read_json_exact
    path = Path(budget_path)
    _reentry_require(path.is_absolute() and path.is_file() and not path.is_symlink(), "regular budget required")
    run = path.parent.resolve()
    attempt = Path(parent_attempt)
    start = read(attempt / "production-start.json")
    journal, _, terminal = _bounded_energy_journal(run, start["session_id"], attempt)
    _reentry_require(isinstance(operation_id, str) and operation_id.startswith("capture-")
                     and Path(operation_id).name == operation_id, "invalid capture operation")
    request_path = journal / (operation_id + ".resource-request.json")
    reply_path = journal / (operation_id + ".resource-reply.json")
    cfg_path = journal / "controller.resources.json"
    request, reply, cfg = map(read, (request_path, reply_path, cfg_path))
    _reentry_require(request.get("run_id") == run.name and request.get("session_id") == start["session_id"]
                     and request.get("operation_id") == operation_id
                     and all(reply.get(k) == request.get(k) for k in ("run_id", "session_id", "operation_id", "token"))
                     and request.get("purpose") == "capture" and request.get("phase") == "FINALIZING"
                     and request.get("process_started") is True
                     and reply.get("state") == "STOP" and reply.get("reason") == "campaign_source_completion_unresolved"
                     and cfg.get("run_id") == run.name and cfg.get("run_dir") == str(run)
                     and cfg.get("session_id") == start["session_id"] and cfg.get("parent_pid") == start["parent_pid"],
                     "exact terminal source STOP required")
    for pid, ticks in ((cfg.get("parent_pid"), cfg.get("parent_start_ticks")),
                       (request.get("owner_pid"), request.get("owner_start_ticks")),
                       (request.get("collector_pid"), request.get("collector_start_ticks"))):
        _reentry_require(type(pid) is int and pid > 0 and str(ticks or "").isdigit()
                         and _proc_start_time(pid) != ticks, "capture owner still active or unbound")
    capture = Path(str(request.get("attempt_dir") or ""))
    _reentry_require(capture.is_absolute() and capture.is_dir() and not capture.is_symlink()
                     and capture.resolve().is_relative_to(attempt.resolve()), "capture outside parent attempt")
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
        _reentry_require(booked["valid"] is False and booked["transport_reasons"]
                         and booked["source_completion_verified"] is False, "negative transport proof missing")
        receipt = {"schema": "onnx-splitpoint/stopped-capture-accounting-v1", "original_chain": copy.deepcopy(chain),
                   "booked_chain": copy.deepcopy(booked), "original_stop": stop, "evidence_sha256": hashes}
        source = root["sources"][source_id]
        grant = source.get("authorized_reentry")
        if grant is not None:
            _reentry_require(_proc_start_time(grant["new_parent"]["pid"]) != grant["new_parent"]["start_ticks"],
                             "authorized parent still active")
            if grant["chain"] == chain:
                receipt["previous_grant_chain"] = copy.deepcopy(grant["chain"])
                grant["chain"] = copy.deepcopy(booked)
            if not grant["source_stop_reason"]:
                grant["source_stop_reason"] = reply["reason"]
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
