"""Opt-in start budget for one energy task, persisted across CLI entries.

The checkpoint is reserved before preflight; summaries are evidence, never the
trigger for another acquisition. Existing platform locks and process leases
remain the caller's responsibility and are not replaced here.
"""
from __future__ import annotations

import json
import re
from functools import wraps
from contextlib import contextmanager
import threading
import os
import fcntl
import copy
from pathlib import Path

DEFAULT_CAMPAIGN_BUDGET = {"enabled": True, "max_retries": 1, "max_transport_failures": 2}


def positive_transport_failure_limit(value):
    """One contract for profile, CLI forwarding and bounded acquisition."""
    if type(value) is not int or value < 1:
        raise ValueError("max_transport_failures must be a positive integer")
    return value


def source_completion(run_dir):
    """The R5 lifecycle contract, evaluated against this physical attempt."""
    path = Path(run_dir) / "collector_stdout.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    go = re.findall(r'Fast Firmware GO sent monotonic_ns=(\d+) local=(\S+) requested_device_us=(\d+)', text)
    end = re.findall(r'Fast Firmware protocol end verified monotonic_ns=(\d+) local=(\S+) elapsed_us=(\d+)', text)
    close = re.findall(r'Fast Firmware source closed monotonic_ns=(\d+) local=(\S+) verified=(\w+) reason=(.*)', text)
    ok = (len(go) == len(end) == len(close) == 1
          and go[0][1] == end[0][1] == close[0][1]
          and close[0][2:] == ('true', 'Ok(())')
          and int(go[0][0]) < int(end[0][0]) <= int(close[0][0])
          and int(end[0][2]) >= int(go[0][2]))
    return {"verified": ok, "go": go, "protocol_end": end, "source_close": close}


def campaign_budget_policy(native_config):
    """Resolve the opt-in profile policy without changing measurement recipes."""
    policy = (native_config.get("energy") or {}).get("task_budget") or {}
    if policy and type(policy.get("enabled")) is not bool:
        raise ValueError("native energy task_budget.enabled must be boolean")
    if not policy.get("enabled"):
        return {}
    retries = policy.get("max_retries")
    if type(retries) is not int or retries < 0:
        raise ValueError("native energy task_budget requires nonnegative integer max_retries")
    failures = positive_transport_failure_limit(policy.get("max_transport_failures"))
    return {"enabled": True, "max_retries": retries, "max_transport_failures": failures,
            "max_chains_per_row": "repeats * (1 + max_retries)",
            "stop_on_unresolved_source": True, "includes_preflights": True}


def campaign_budget_profile_args(native_config, run_dir):
    policy = campaign_budget_policy(native_config)
    if not policy:
        return []
    return ["--campaign-budget-file", str(Path(run_dir).resolve() / "energy_task_budget.json"),
            "--campaign-max-retries", str(policy["max_retries"]),
            "--campaign-max-transport-failures", str(policy["max_transport_failures"])]


def add_campaign_budget_arguments(parser, *, measurement=False):
    parser.add_argument("--require-environment-observation", action="store_true")
    parser.add_argument("--environment-run-id", default=None)
    parser.add_argument("--campaign-budget-file", default=None)
    parser.add_argument("--campaign-cohort", default=None,
                        help="Already authorized final cohort; never creates an authorization or new physical budget.")
    parser.add_argument("--campaign-max-retries", type=int, default=None)
    parser.add_argument("--campaign-max-transport-failures",
                        type=lambda text: positive_transport_failure_limit(int(text)), default=None)
    if measurement:
        parser.add_argument("--campaign-row-id", default=None)
        parser.add_argument("--campaign-repeats", type=int, default=None)
        parser.add_argument("--task-logical-repeat", default=None)
        parser.add_argument("--preflight-prepare-command", default=None,
                            help="Additional command before attested preflight, inside the same reserved chain.")


def campaign_budget_forward_args(args):
    values = [getattr(args, name, None) for name in (
        "campaign_budget_file", "campaign_max_retries", "campaign_max_transport_failures")]
    if all(value is None for value in values):
        if getattr(args,"campaign_cohort",None): raise ValueError("cohort requires campaign budget")
        return []
    path, retries, failures = values
    if not path or type(retries) is not int or retries < 0:
        raise ValueError("incomplete campaign budget arguments")
    failures = positive_transport_failure_limit(failures)
    forwarded = ["--campaign-budget-file", str(Path(path).expanduser().resolve()),
                 "--campaign-max-retries", str(retries), "--campaign-max-transport-failures", str(failures)]
    if getattr(args, "campaign_cohort", None):
        forwarded += ["--campaign-cohort", args.campaign_cohort]
    if getattr(args, "require_environment_observation", False):
        if not getattr(args,"environment_run_id",None): raise ValueError("required environment parent identity missing")
        forwarded += ["--require-environment-observation","--environment-run-id",args.environment_run_id]
    return forwarded


def campaign_budget_measurement_kwargs(args):
    return {name: getattr(args, name, None) for name in (
        "campaign_budget_file", "campaign_cohort", "campaign_row_id", "campaign_repeats",
        "campaign_max_retries", "campaign_max_transport_failures", "preflight_prepare_command", "require_environment_observation", "environment_run_id")}


def _budget_transaction(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.transaction():
            return method(self, *args, **kwargs)
    return call


def final_cohort_authorization(root, cohort_id):
    grants = [root.get("authorized_final_cohort"), *root.get("authorized_final_cohorts", [])]
    matched = [g for g in grants if g and g.get("cohort_id") == cohort_id]
    _reentry_require(len(matched) <= 1, "duplicate final cohort authorization")
    return matched[0] if matched else {}


def authorize_final_cohort(budget_path, *, cohort_id, parent_run_dir, row_ids, source_id, authorization):
    """Install one explicitly requested final selection in the existing budget.

    Task identities, chains, limits and source state are never rewritten. A
    second generation is rejected even after all its repeats have completed.
    """
    parent = Path(parent_run_dir).resolve()
    rows = sorted(row_ids)
    if (not cohort_id or parent.name != cohort_id or len(rows) != len(set(rows))
            or not rows or authorization.get("purpose") != "final"
            or authorization.get("repeats") != 3
            or sorted(authorization.get("row_ids", [])) != rows
            or authorization.get("generation") != cohort_id
            or not authorization.get("explicit_user_request")):
        raise ValueError("final cohort requires explicit exact-scope generation authorization")
    grant = {"cohort_id":cohort_id,"parent_run_dir":str(parent),"source_id":source_id,
             "row_ids":rows,"repeats":3,"purpose":"final","authorization":authorization,
             "environment_observation_required":True}
    with Path(budget_path).open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        root = json.load(handle)
        previous = final_cohort_authorization(root, cohort_id)
        if previous:
            if previous != grant:raise ValueError("different authorization for existing final cohort")
            return copy.deepcopy(previous)
        continuous = root["sources"][source_id].get("continuous_admission")
        if root.get("authorized_final_cohort") and continuous is None:
            raise ValueError("a different final cohort is already authorized; no second grant")
        if not root.get("campaign") or any(campaign_budget_stop_reason(root, r, source_id=source_id, parent_run_dir=parent) for r in rows):
            raise ValueError("final cohort cannot lift a campaign/source STOP")
        for row_id in rows:
            task = root["tasks"].setdefault(row_id, {
                "source_id": source_id, "limits": {"max_chains": 9, "max_retries": 2,
                "max_transport_failures": root["max_transport_failures"]}, "chains": [], "stop_reason": ""})
            if (task["source_id"] != source_id
                    or len(task["chains"]) >= task["limits"]["max_chains"]):
                raise ValueError("final cohort task source/STOP/physical budget is inadmissible")
            if any(c.get("cohort_id") for c in task["chains"]) and continuous is None:
                raise ValueError("existing final selection must not receive a second authorization")
        if any(not c.get("finished") or (c.get("collector_started") and c.get("source_completion_verified") is not True)
               for task in root["tasks"].values() if task["source_id"] == source_id for c in task["chains"]
               if not any(c == covered["chain"] for covered in _reentry_history(continuous))):
            raise ValueError("final cohort requires resolved original source chains")
        if root.get("authorized_final_cohort") is None:root["authorized_final_cohort"] = grant
        else:root.setdefault("authorized_final_cohorts", []).append(grant)
        handle.seek(0); json.dump(root, handle, indent=2); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        return copy.deepcopy(grant)


def _chain_owner_alive(chain):
    from ..process_control import _proc_start_time
    pid, ticks = chain.get("owner_pid"), chain.get("owner_start_ticks")
    return bool(pid and ticks and _proc_start_time(int(pid)) == ticks)


def _reentry_require(condition, detail):
    if not condition:
        raise ValueError("authorized energy source reentry: " + detail)


def _budget_parent(environment=None):
    """Read the normal broker identity; this does not create a lease journal."""
    from ..remote.process_lease import RemoteProcessLeaseJournal
    from ..process_control import _proc_start_time
    env = {**os.environ, **(environment or {})}
    run = env.get("ONNX_SPLITPOINT_REMOTE_LEASE_RUN_ID")
    session = env.get("ONNX_SPLITPOINT_REMOTE_LEASE_SESSION_ID")
    directory = Path(env.get("ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR") or "")
    _reentry_require(env.get("ONNX_SPLITPOINT_CONTROLLER_RESOURCES") == "required"
                     and run and session and directory.is_absolute(), "broker context missing")
    cfg = RemoteProcessLeaseJournal._read_json_exact(directory / "controller.resources.json")
    pid, ticks = cfg.get("parent_pid"), cfg.get("parent_start_ticks")
    _reentry_require(cfg.get("run_id") == run and cfg.get("session_id") == session
                     and Path(str(cfg.get("run_dir") or "")).name == run
                     and type(pid) is int and pid > 0 and ticks
                     and _proc_start_time(pid) == ticks, "broker parent binding invalid")
    return {"run_id": run, "session_id": session, "pid": pid, "start_ticks": ticks}


def _source_reentry(root, source_id, environment=None):
    source = root["sources"][source_id]
    continuous = source.get("continuous_admission")
    if continuous is not None:
        import hashlib
        authorization = Path(continuous["authorization_file"])
        _reentry_require(authorization.is_file() and not authorization.is_symlink()
                         and hashlib.sha256(authorization.read_bytes()).hexdigest() == continuous["authorization_sha256"],
                         "continuous campaign authorization changed")
        _reentry_require(source["stop_reason"] == continuous["original_source_stop_reason"]
                         and root["max_transport_failures"] == continuous["max_transport_failures"],
                         "continuous admission changed original source policy")
        for covered in _reentry_history(continuous):
            task = root["tasks"][covered["campaign_row"]]
            _reentry_require(task["source_id"] == source_id and covered["chain"] in task["chains"],
                             "continuous admission changed historical chain")
        _validate_sample_loss_authority(root, source_id, continuous)
        for row, change in continuous.get("limit_changes", {}).items():
            if change.get("kind") == "sample_loss_series":
                _validate_sample_loss_change(root, source_id, continuous, row, change)
                continue
            task = root["tasks"][row]
            diagnosis = Path(change["diagnosis_file"])
            _reentry_require(row in continuous["row_ids"] and task["source_id"] == source_id
                             and task["limits"] == change["before"] and change["before"]["max_retries"] == 2
                             and change["after"] == {**change["before"], "max_retries": 3}
                             and change["logical_repeat"] in {"repeat:0", "repeat:1", "repeat:2"}
                             and row in final_cohort_authorization(root, change["cohort_id"])["row_ids"]
                             and task["chains"][:len(change["original_chains"])] == change["original_chains"]
                             and diagnosis.is_file() and not diagnosis.is_symlink()
                             and hashlib.sha256(diagnosis.read_bytes()).hexdigest() == change["diagnosis_sha256"],
                             "administrative retry authorization or original evidence changed")
        # This is permission, not a resource lease or proof of the old source
        # close. Every executing caller still requires its ordinary broker.
        if (environment or os.environ).get("ONNX_SPLITPOINT_CONTROLLER_RESOURCES") == "required":
            _budget_parent(environment)
        return continuous
    grant = source.get("authorized_reentry")
    if grant is None:
        return None
    _reentry_require(_budget_parent(environment) == grant["new_parent"], "different parent session")
    task = root["tasks"][grant["campaign_row"]]
    matches = [c for c in task["chains"] if c.get("run_directory") == grant["chain"]["run_directory"]]
    _reentry_require(matches == [grant["chain"]] and task["source_id"] == source_id
                     and task["limits"] == grant["limits"]
                     and task["stop_reason"] == grant["original_task_stop_reason"]
                     and source["stop_reason"] == grant["original_source_stop_reason"]
                     and root["max_transport_failures"] == grant["max_transport_failures"],
                     "historical capture or budget contract changed")
    for row, change in grant.get("limit_changes", {}).items():
        current = root["tasks"].get(row)
        _reentry_require((current is None and change["before"] is None)
                         or (current is not None and current["source_id"] == source_id
                             and current["limits"] == change["after"]
                             and current["stop_reason"] == change["original_stop_reason"]),
                         "authorized row limits or original stop changed")
    for covered in _reentry_history(grant):
        previous_task = root["tasks"][covered["campaign_row"]]
        _reentry_require(previous_task["source_id"] == source_id
                         and covered["chain"] in previous_task["chains"],
                         "previous recovered chain changed")
    return grant


def _validate_sample_loss_authority(root, source_id, grant):
    policy = grant.get("sample_loss_retries")
    if policy is None:
        return
    import hashlib
    path = Path(policy["authorization_file"])
    history = policy["series_history"]
    _reentry_require(path.is_file() and not path.is_symlink()
                     and hashlib.sha256(path.read_bytes()).hexdigest() == policy["authorization_sha256"]
                     and policy["series_starts"] == 3
                     and policy["row_ids"] == sorted(set(policy["row_ids"]))
                     and set(policy["row_ids"]).issubset(grant["row_ids"])
                     and policy["original_source_failure_limit"] == root["max_transport_failures"]
                     and policy["effective_source_failure_limit"] == root["max_transport_failures"] + 3 * len(history),
                     "sample loss retry authority or finite source bounds changed")
    for index, item in enumerate(history):
        _reentry_require(item["campaign_row"] in policy["row_ids"]
                         and item["series_starts"] == 3
                         and item["source_before"] == root["max_transport_failures"] + 3 * index
                         and item["source_after"] == item["source_before"] + 3,
                         "sample loss source series changed")


def _validate_sample_loss_change(root, source_id, grant, row, change):
    task = root["tasks"][row]
    _reentry_require(row in grant.get("sample_loss_retries", {}).get("row_ids", [])
                     and task["source_id"] == source_id and task["limits"] == change["before"]
                     and task["chains"][:len(change["original_chains"])] == change["original_chains"]
                     and change["logical_repeat"] in {"repeat:0", "repeat:1", "repeat:2"},
                     "sample loss retry original task or slot changed")
    previous = change.get("previous_change")
    before = previous["after"] if previous else task["limits"]
    _reentry_require(set(change["after"]) == set(before)
                     and all(type(value) is int and value == before[key] + 3
                             for key, value in change["after"].items())
                     and change["cohort_id"] == (change["last_chain"].get("cohort_id") or "")
                     and change["last_chain"] in task["chains"]
                     and change["last_chain"].get("finished") is True
                     and change["last_chain"].get("valid") is False
                     and change["last_chain"].get("source_completion_verified") is True
                     and change["last_chain"].get("transport_reasons") == ["marker_dropped_samples_nonzero"],
                     "sample loss retry finite series or original failure changed")
    if change["cohort_id"]:
        _reentry_require(row in final_cohort_authorization(root, change["cohort_id"])["row_ids"],
                         "sample loss retry outside final cohort")
    if previous and previous.get("kind") == "sample_loss_series":
        _validate_sample_loss_change(root, source_id, grant, row, previous)
    elif previous:
        # The earlier one-slot diagnosis grant remains exactly bound.
        import hashlib
        diagnosis = Path(previous["diagnosis_file"])
        _reentry_require(previous["before"] == task["limits"]
                         and previous["after"] == {**task["limits"], "max_retries": 3}
                         and task["limits"]["max_retries"] == 2
                         and task["chains"][:len(previous["original_chains"])] == previous["original_chains"]
                         and diagnosis.is_file() and not diagnosis.is_symlink()
                         and hashlib.sha256(diagnosis.read_bytes()).hexdigest() == previous["diagnosis_sha256"],
                         "historical administrative retry grant changed")


def authorize_campaign_sample_loss_retries(path, source_id, authorization_file, row_ids):
    """Register the unchanged user supplement; do not release a physical STOP."""
    import hashlib
    path = Path(path).resolve(strict=True)
    authorization = Path(authorization_file).resolve(strict=True)
    rows = sorted(set(row_ids))
    text = authorization.read_text(encoding="utf-8")
    _reentry_require(not authorization.is_symlink()
                     and authorization.name == "AUFTRAG_RETRIES_20261011.txt"
                     and path == authorization.parents[2] / "energy/energy_task_budget.json"
                     and all(phrase in text for phrase in (
                         "PAKET-/SAMPLEVERLUST DURCH REPEATS BEHANDELN",
                         "Die bestehende Gesamtfreigabe bleibt wirksam",
                         "Nur den noch fehlenden logischen Repeat-Slot",
                         "Zähler zurücksetzen"))
                     and rows and all(isinstance(row, str) and len(json.loads(row)) == 4 for row in rows),
                     "exact sample loss retry supplement or row scope required")
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        root = json.load(handle)
        grant = _source_reentry(root, source_id, {})
        _reentry_require(grant and grant.get("continuous_campaign") is True
                         and set(rows).issubset(grant["row_ids"]),
                         "sample loss retries require existing continuous row authority")
        descriptor = {"authorization_file": str(authorization),
                      "authorization_sha256": hashlib.sha256(authorization.read_bytes()).hexdigest(),
                      "row_ids": rows, "series_starts": 3,
                      "original_source_failure_limit": root["max_transport_failures"]}
        existing = grant.get("sample_loss_retries")
        if existing is not None:
            _reentry_require(all(existing.get(key) == value for key, value in descriptor.items()),
                             "sample loss supplement or scope changed")
            return copy.deepcopy(grant)
        previous = copy.deepcopy(grant)
        grant["sample_loss_retries"] = {**descriptor,
            "effective_source_failure_limit": root["max_transport_failures"], "series_history": []}
        grant["previous_grants"] = [previous]
        handle.seek(0); json.dump(root, handle, indent=2); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        return copy.deepcopy(grant)


def _pure_sample_loss_failure(chain, row, cohort_id):
    """Classify retry permission without changing the failed scientific report."""
    directory = Path(chain["run_directory"])
    entry = json.loads((directory / "energy_summary.json").read_text(encoding="utf-8"))
    request = entry.get("command_window_request") or {}
    timing = entry.get("workload_timing") or {}
    calibration = entry.get("energy_calibration_verification") or {}
    if (entry.get("cancelled") or entry.get("collector_rc") != 0
            or entry.get("workload_command_rc") != 0 or timing.get("rc") != 0 or timing.get("status") != "ok"
            or entry.get("preflight_status") != "verified"
            or entry.get("runtime_work_unit_gate_status") != "pass"
            or entry.get("full_system_scope_calibration_status") not in {"pass", "not_full_system"}
            or entry.get("diagnostic_only") is True or entry.get("global_infrastructure_failure")
            or entry.get("postprocess_status") != "ok"
            or calibration.get("runtime_fail_closed_required") and calibration.get("verified") is not True
            or set(request.get("errors") or []) != {"marker_dropped_samples_nonzero", "marker_not_valid_for_final_energy"}
            or set(entry.get("final_energy_gate_reasons") or []) - {
                "explicit_trace_command_binding_missing_or_invalid", "raw_input_energy_provenance_unverified"}):
        return False
    from .collector import _energy_repeat_retry_reasons
    reasons = _energy_repeat_retry_reasons(entry, directory)
    if "marker_dropped_samples_nonzero" not in reasons or any(
            reason not in {"marker_dropped_samples_nonzero", "command_window_marker_contract_invalid"}
            and not reason.startswith("marker_binding_invalid:command_window_marker_contract_invalid")
            for reason in reasons):
        return False
    if cohort_id or entry.get("environment_observation_required"):
        from .environment_observer import validate_environment
        observed = entry.get("environment_observation") or {}
        if observed.get("ok") is not True:
            # A rejected marker prevents the original environment reader from
            # seeing its timestamps. Check the original observer samples only
            # for retry permission; retain the original INVALID report.
            if observed.get("reason") != "required_capture_environment_missing":
                return False
            marker_path = directory / "collector_storage/command_window_markers.json"
            if request.get("marker_path") != str(marker_path):
                return False
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            command = marker.get("command") or {}
            if (marker.get("run_id") != entry.get("collector_run_id")
                    or marker.get("window_id") != entry.get("collector_window_id")
                    or command.get("process_rc") != 0
                    or command.get("argv") != [str(directory / "energy_command.sh")]
                    or not isinstance(command.get("start_realtime_ns"), int)
                    or not isinstance(command.get("end_realtime_ns"), int)
                    or command["end_realtime_ns"] <= command["start_realtime_ns"]):
                return False
            probe = copy.deepcopy(entry)
            probe["campaign_row"] = row
            if cohort_id: probe["campaign_cohort"] = cohort_id
            probe["command_window_trace_binding"] = {
                **(entry.get("command_window_trace_binding") or {}),
                "command_start_realtime_ns": command["start_realtime_ns"],
                "command_end_realtime_ns": command["end_realtime_ns"]}
            if not validate_environment(directory, probe, freeze=False).get("ok"):
                return False
        elif not validate_environment(directory, entry, freeze=False).get("ok"):
            return False
    return True


@contextmanager
def _released_sample_loss_resources(chain, source_id, environment):
    """Inspect the last normal release and four existing resource locks."""
    from contextlib import ExitStack
    from ..remote.process_lease import RemoteProcessLeaseJournal
    from ..workflow.run_control import EvaluationRunLock
    from ..process_control import _proc_start_time
    read = RemoteProcessLeaseJournal._read_json_exact
    parent = _budget_parent(environment)
    env = {**os.environ, **(environment or {})}
    journal = Path(env["ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR"])
    current = read(journal / "controller.resources.json")
    root = Path(current["run_dir"])
    run = Path(chain["run_directory"])
    _reentry_require(run.is_absolute() and run.is_relative_to(root), "retry outside current normal parent")
    directories = {journal, root / "remote-process-leases"}
    relative = run.relative_to(root).parts
    if len(relative) > 2 and relative[0:2] == ("measurement", "resume_attempts"):
        directories.add(root / "resume_attempts" / relative[2] / "remote-process-leases")
    if len(relative) > 1 and relative[0] in {"attempts", "resume_attempts"}:
        directories.add(root / relative[0] / relative[1] / "remote-process-leases")
    matches = []
    for directory in directories:
        for request_path in directory.glob("*.resource-request.json"):
            request = read(request_path)
            if request.get("purpose") == "capture" and request.get("attempt_dir") == str(run):
                matches.append((directory, request_path, request))
    _reentry_require(len(matches) == 1, "exact last capture request required")
    directory, request_path, request = matches[0]
    cfg = read(directory / "controller.resources.json")
    reply = read(directory / (request_path.name.removesuffix(".resource-request.json") + ".resource-reply.json"))
    cleanup = read(run / "collector_stdout.log.cleanup.json")
    resources = sorted(r for r in cfg["setups"][request["setup_id"]]
                       if r.startswith(("source:", "dut:")) or r in {"controller:capture", "controller:nic"})
    current_resources = sorted(r for r in current["setups"][request["setup_id"]]
                              if r.startswith(("source:", "dut:")) or r in {"controller:capture", "controller:nic"})
    _reentry_require(resources == current_resources and len(resources) == 4
                     and "source:" + source_id in resources and sum(r.startswith("dut:") for r in resources) == 1
                     and cfg["run_dir"] == str(root) and cfg["run_id"] == parent["run_id"]
                     and request.get("run_id") == cfg["run_id"] and request.get("session_id") == cfg["session_id"]
                     and current.get("run_id") == parent["run_id"] and current.get("session_id") == parent["session_id"]
                     and current.get("parent_pid") == parent["pid"] and current.get("parent_start_ticks") == parent["start_ticks"]
                     and request.get("phase") == "FINALIZING" and request.get("process_started") is True
                     and request.get("owner_pid") == chain["owner_pid"]
                     and request.get("owner_start_ticks") == chain["owner_start_ticks"]
                     and reply.get("state") == "RELEASED" and not reply.get("reason")
                     and all(reply.get(key) == request.get(key) for key in ("run_id", "session_id", "operation_id", "token"))
                     and cleanup.get("owned_tree_quiescent") is True and cleanup.get("process_started") is True
                     and cleanup.get("collector_pid") == request.get("collector_pid")
                     and cleanup.get("collector_start_ticks") == request.get("collector_start_ticks")
                     and _proc_start_time(request["collector_pid"]) != request["collector_start_ticks"],
                     "last collector or normal resource release unresolved")
    with ExitStack() as held:
        for resource in resources:
            lease = EvaluationRunLock.for_resource(resource, owner={})
            lease.acquire_for_recovery(); held.callback(lease.release, write_released_state=False)
            _reentry_require(not lease.quarantine_path.exists() and lease.previous_owner.get("state") == "released",
                             "retry resource active or quarantined")
        yield {"request_path": str(request_path), "request": request, "reply": reply,
               "resources": resources, "parent": parent}


def _reentry_history(grant):
    if grant is None:
        return
    yield grant
    for previous in grant.get("previous_grants", []):
        yield from _reentry_history(previous)


def _historical_task_stop(task, row_id, grant):
    if grant.get("continuous_campaign"):
        _reentry_require(row_id in grant["row_ids"], "row outside authorized campaign")
        _reentry_require(task.get("stop_reason", "") == grant["original_task_stops"].get(row_id, ""),
                         "original task STOP changed after campaign admission")
        return ""
    if row_id == grant["campaign_row"] or row_id in grant.get("limit_changes", {}):
        return ""
    return task["stop_reason"]


def _unstarted_capture_task_stop(root, row_id):
    """Project a proved zero-start closure while retaining its historical STOP."""
    task = root["tasks"][row_id]
    key = task.get("unstarted_capture_reconciliation")
    if not key:
        return str(task.get("stop_reason") or "")
    receipt = root.get("unstarted_capture_reconciliations", {}).get(key)
    _reentry_require(isinstance(receipt, dict)
                     and receipt.get("schema") == "onnx-splitpoint/unstarted-capture-accounting-v1"
                     and receipt.get("campaign_row") == row_id and receipt.get("source_id") == task["source_id"]
                     and task["stop_reason"] == receipt.get("original_task_stop_reason") == "task_execution_interrupted"
                     and receipt.get("physical_source_retry_authorized") is False,
                     "zero-start accounting identity changed")
    booked = receipt.get("booked_chain", {})
    matches = [c for c in task["chains"] if c.get("run_directory") == booked.get("run_directory")]
    _reentry_require(matches == [booked] and booked.get("finished") is True
                     and booked.get("collector_started") is False and booked.get("workload_started") is False
                     and booked.get("valid") is False and not booked.get("transport_reasons")
                     and booked.get("source_completion_verified") is None
                     and isinstance(task.get("post_unstarted_capture_stop_reason"), str),
                     "zero-start negative chain changed")
    return task["post_unstarted_capture_stop_reason"]


def _validate_unstarted_capture_evidence(root, source_id):
    """Validate the existing zero-start reconciliation and its original bindings."""
    source = root["sources"][source_id]
    key = source.get("unstarted_capture_reconciliation")
    if not key:
        return None
    receipt = root.get("unstarted_capture_reconciliations", {}).get(key)
    _reentry_require(isinstance(receipt, dict), "zero-start source receipt missing")
    projection = receipt.get("derived_source_stop")
    _reentry_require(isinstance(projection, dict) and receipt.get("source_id") == source_id
                     and receipt.get("physical_source_retry_authorized") is False
                     and not source.get("authorized_reentry")
                     and source.get("stop_reason") == projection.get("original_stop_reason")
                         == "campaign_source_interrupted_chain_unresolved"
                     and isinstance(source.get("post_unstarted_capture_stop_reason"), str),
                     "derived zero-start source STOP changed")
    original = root["tasks"][receipt["campaign_row"]]
    _unstarted_capture_task_stop(root, receipt["campaign_row"])
    _reentry_require(original["chains"] == [receipt["booked_chain"]]
                     and original["limits"] == projection["original_limits"]
                     and root["max_transport_failures"] == projection["max_transport_failures"],
                     "original zero-start reservation or limits changed")
    from ..remote.process_lease import RemoteProcessLeaseJournal
    import hashlib
    auth_path = Path(projection["authorization_file"])
    _reentry_require(auth_path.is_file() and not auth_path.is_symlink()
                     and hashlib.sha256(auth_path.read_bytes()).hexdigest() == projection["authorization_sha256"],
                     "one-time zero-start authorization changed")
    auth = RemoteProcessLeaseJournal._read_json_exact(auth_path)
    _reentry_require(auth.get("authorized") is True and auth.get("operation_id") == receipt["operation_id"]
                     and auth.get("source_id") == source_id and auth.get("campaign_row") == receipt["campaign_row"],
                     "zero-start authorization scope changed")
    _reentry_require(auth.get("allowed_parents") == projection["allowed_parents"], "authorized batch scope changed")
    for evidence_file, digest in receipt["evidence_sha256"].items():
        path = Path(evidence_file)
        _reentry_require(path.is_file() and not path.is_symlink()
                         and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                         "bound zero-start original evidence changed")
    storage = Path(receipt["original_chain"]["run_directory"]) / "collector_storage"
    _reentry_require(storage.is_dir() and not storage.is_symlink() and not list(storage.iterdir()),
                     "old zero-start storage changed")
    return receipt


def _derived_unstarted_capture(root, source_id, row_id, environment=None, parent_run_dir=None):
    """The one expressly scoped accounting release; never a physical reentry."""
    receipt = _validate_unstarted_capture_evidence(root, source_id)
    if receipt is None:
        return None
    projection = receipt["derived_source_stop"]
    from ..remote.process_lease import RemoteProcessLeaseJournal
    if not any(row_id in rows for rows in projection["allowed_parents"].values()):
        return None
    if parent_run_dir is None:
        parent = _budget_parent(environment)
        env = {**os.environ, **(environment or {})}
        cfg = RemoteProcessLeaseJournal._read_json_exact(
            Path(env["ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR"]) / "controller.resources.json")
        parent_run_dir = cfg["run_dir"]
        _reentry_require(Path(parent_run_dir).name == parent["run_id"], "zero-start parent changed")
    rows = projection["allowed_parents"].get(str(Path(parent_run_dir).resolve()), [])
    if row_id not in rows:
        return None
    for key, source_task in root["tasks"].items():
        if source_task["source_id"] != source_id:
            continue
        for chain in source_task["chains"]:
            if chain == receipt["booked_chain"] or (chain.get("finished") is True
                    and (chain.get("collector_started") is False or chain.get("source_completion_verified") is True)):
                continue
            _reentry_require(not chain.get("finished") and _chain_owner_alive(chain)
                             and key in rows and Path(chain["run_directory"]).is_relative_to(Path(parent_run_dir)),
                             "another unresolved source chain blocks batch admission")
    task = root["tasks"].get(row_id)
    if task:
        _reentry_require(task["source_id"] == source_id and task["limits"] == projection["original_limits"],
                         "authorized batch source or limits changed")
        for chain in task["chains"]:
            _reentry_require(Path(chain["run_directory"]).is_relative_to(Path(parent_run_dir)),
                             "batch task has a foreign physical chain")
    return receipt


def _campaign_source_stop(root, source_id, row_id, environment=None, parent_run_dir=None):
    source = root["sources"][source_id]
    if source.get("post_unstarted_capture_stop_reason"):
        return str(source["post_unstarted_capture_stop_reason"])
    receipt = _derived_unstarted_capture(root, source_id, row_id, environment, parent_run_dir)
    return (source["post_unstarted_capture_stop_reason"] if receipt is not None
            else str(source.get("stop_reason") or ""))


def _campaign_task_stop(root, row_id, source_id, environment=None, parent_run_dir=None):
    task = root["tasks"].get(row_id)
    if task is None:
        return ""
    if root["sources"][source_id].get("post_unstarted_capture_stop_reason"):
        return str(task.get("post_unstarted_capture_stop_reason") or task.get("stop_reason") or "")
    receipt = _derived_unstarted_capture(root, source_id, row_id, environment, parent_run_dir)
    if receipt and row_id in receipt["derived_source_stop"]["propagated_task_stops"]:
        _reentry_require(task["stop_reason"] == receipt["derived_source_stop"]["propagated_task_stops"][row_id]
                         and isinstance(task.get("post_unstarted_capture_stop_reason"), str),
                         "propagated zero-start task STOP changed")
        return task["post_unstarted_capture_stop_reason"]
    return _unstarted_capture_task_stop(root, row_id)


def campaign_budget_stop_reason(root, row_id, environment=None, *, source_id=None, parent_run_dir=None):
    """Project the effective refusal without rewriting a historical STOP."""
    task = root["tasks"].get(row_id)
    _reentry_require(task is None or source_id is None or task["source_id"] == source_id,
                     "campaign permission source differs from original task")
    source_id = task["source_id"] if task else source_id
    _reentry_require(source_id in root["sources"], "campaign admission source missing")
    grant = _source_reentry(root, source_id, environment)
    if grant is not None and grant.get("continuous_campaign"):
        _reentry_require(row_id in grant["row_ids"], "row outside authorized campaign")
    if grant is None:
        return str(_campaign_source_stop(root, source_id, row_id, environment, parent_run_dir)
                   or _campaign_task_stop(root, row_id, source_id, environment, parent_run_dir))
    return str(grant["source_stop_reason"] or grant["task_stop_reasons"].get(row_id)
               or (_historical_task_stop(task, row_id, grant) if task is not None else ""))


def _authorize_closed_retry_continuation(path, source_id, authorization, rows, requested, diagnosis_file, previous_attempt_dir):
    """One diagnosed retry slot; preserve all physical history and base limits."""
    import hashlib
    from contextlib import ExitStack
    from datetime import datetime, timezone
    from ..remote.resource_recovery import _bounded_energy_journal, _rest_evidence_unchanged
    from ..remote.process_lease import RemoteProcessLeaseJournal
    from ..workflow.run_control import EvaluationRunLock
    from ..process_control import _proc_start_time
    read = RemoteProcessLeaseJournal._read_json_exact
    attempt = Path(previous_attempt_dir).resolve(strict=True)
    diagnosis_path = Path(diagnosis_file).resolve(strict=True)
    diagnosis = read(diagnosis_path)
    run = Path(diagnosis["parent_run_dir"]).resolve(strict=True)
    parent_plan = read(run / "energy_parent_plan.json")
    _reentry_require(parent_plan.get("campaign_budget_file") == str(path)
                     and parent_plan.get("campaign_cohort") == run.name
                     and json.loads(diagnosis["campaign_row"]) in parent_plan.get("capture_identities", []),
                     "retry differs from original prepared budget/cohort/capture scope")
    start = read(attempt / "production-start.json")
    directory, _, terminal = _bounded_energy_journal(run, start["session_id"], attempt)
    config = read(directory / "controller.resources.json")
    cleanup = read(attempt / "cleanup-status.json")
    _reentry_require(diagnosis["previous_attempt_dir"] == str(attempt)
                     and diagnosis["cohort_id"] == run.name
                     and start["run_id"] == run.name and config["run_dir"] == str(run)
                     and config["parent_pid"] == start["parent_pid"]
                     and _proc_start_time(config["parent_pid"]) != config["parent_start_ticks"]
                     and cleanup.get("cleanup_proven") is True and cleanup.get("errors") == []
                     and cleanup.get("owned_remote_active") == cleanup.get("owned_local_active") == 0
                     and not list(directory.glob("*.remote-lease.json")), "retry predecessor not clean and quiescent")
    resources = sorted({r for value in config["setups"].values() for r in value
                        if r.startswith(("source:", "dut:")) or r in {"controller:capture", "controller:nic"}})
    _reentry_require(len(resources) == 4 and "source:" + source_id in resources
                     and sum(r.startswith("dut:") for r in resources) == 1,
                     "retry requires its exact four original resources")
    for request_path in directory.glob("*.resource-request.json"):
        request = read(request_path)
        reply_path = directory / (request_path.name.removesuffix(".resource-request.json") + ".resource-reply.json")
        reply = read(reply_path)
        _reentry_require(reply.get("state") == "RELEASED"
                         and request.get("run_id") == run.name and request.get("session_id") == start["session_id"]
                         and all(reply.get(k) == request.get(k) for k in ("run_id", "session_id", "operation_id", "token")),
                         "retry predecessor operation not released")
        terminal.update({request_path: request, reply_path: reply})
    terminal[directory / "controller.resources.json"] = config
    diagnosis_sha = hashlib.sha256(diagnosis_path.read_bytes()).hexdigest()
    with ExitStack() as held:
        for resource in resources:
            lease = EvaluationRunLock.for_resource(resource, owner={})
            lease.acquire_for_recovery(); held.callback(lease.release, write_released_state=False)
            _reentry_require(not lease.quarantine_path.exists() and lease.previous_owner.get("state") == "released",
                             "retry resource active or quarantined")
        with path.open("r+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX); root = json.load(handle)
            grant = _source_reentry(root, source_id, {})
            _reentry_require(grant is not None and grant.get("continuous_campaign") is True
                             and grant["authorization_file"] == str(authorization) and grant["row_ids"] == rows
                             and not grant["source_stop_reason"], "exact current continuous authority required")
            if grant.get("basis", {}).get("diagnosis_sha256") == diagnosis_sha:
                _reentry_require(not any(grant["task_stop_reasons"].values()), "used retry authorization cannot clear a later STOP")
                return copy.deepcopy(grant)
            row = diagnosis["campaign_row"]; logical = diagnosis["logical_repeat"]
            _reentry_require(set(requested) == {row} and row in rows and logical in {"repeat:0", "repeat:1", "repeat:2"}
                             and diagnosis.get("stop_reason") == "task_logical_retry_limit"
                             and diagnosis.get("finding") == "received_counter_gaps_not_proven_reader_bug"
                             and grant["task_stop_reasons"] == {row: "task_logical_retry_limit"}
                             and row not in grant["limit_changes"], "exact diagnosed administrative retry STOP required")
            task = root["tasks"][row]; limits = task["limits"]
            cohort = final_cohort_authorization(root, run.name)
            _reentry_require(cohort["parent_run_dir"] == str(run) and cohort["source_id"] == source_id
                             and row in cohort["row_ids"] and task["source_id"] == source_id,
                             "retry outside original final parent/cohort/source")
            after = requested[row]
            _reentry_require(type(after.get("max_retries")) is int and limits["max_retries"] == 2
                             and after == {**limits, "max_retries": 3}
                             and root["sources"][source_id]["transport_failures"] < root["max_transport_failures"]
                             and len(task["chains"]) < limits["max_chains"], "only one bounded retry raise permitted")
            invalid = [c for c in task["chains"] if c.get("cohort_id") == run.name and c["logical_repeat"] == logical]
            _reentry_require(len(invalid) == 3 and all(c.get("valid") is False
                             and c.get("transport_reasons") == ["marker_dropped_samples_nonzero"] for c in invalid)
                             and diagnosis["invalid_run_directories"] == [c["run_directory"] for c in invalid],
                             "diagnosis differs from original three failed attempts")
            for other in root["tasks"].values():
                if other["source_id"] != source_id: continue
                for chain in other["chains"]:
                    _reentry_require(chain.get("finished") is True and not _chain_owner_alive(chain)
                                     and (not chain.get("collector_started") or chain.get("source_completion_verified") is True),
                                     "retry source has active or unresolved original acquisition")
                    if chain.get("cohort_id") == run.name and chain.get("collector_started"):
                        _reentry_require(source_completion(chain["run_directory"])["verified"], "original source close not verified")
            _reentry_require(_rest_evidence_unchanged(terminal)
                             and hashlib.sha256(diagnosis_path.read_bytes()).hexdigest() == diagnosis_sha,
                             "retry terminal or diagnosis evidence changed")
            updated = copy.deepcopy(grant)
            updated.update(campaign_row=row, chain=copy.deepcopy(invalid[-1]), previous_grants=[copy.deepcopy(grant)],
                           authorized_at=datetime.now(timezone.utc).isoformat(), task_stop_reasons={},
                           basis={"kind": "diagnosed_closed_administrative_retry", "diagnosis_file": str(diagnosis_path),
                                  "diagnosis_sha256": diagnosis_sha, "previous_attempt_dir": str(attempt),
                                  "original_task_stop_reason": "task_logical_retry_limit"})
            updated["limit_changes"][row] = {"before": copy.deepcopy(limits), "after": copy.deepcopy(after),
                                          "cohort_id": run.name, "logical_repeat": logical,
                                          "diagnosis_file": str(diagnosis_path), "diagnosis_sha256": diagnosis_sha,
                                          "original_chains": copy.deepcopy(task["chains"])}
            root["sources"][source_id]["continuous_admission"] = updated
            handle.seek(0); json.dump(root, handle, indent=2); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
            return copy.deepcopy(updated)


def authorize_campaign_source_continuation(budget_path, *, source_id, authorization_file, row_ids,
                                           authorized_limits=None, diagnosis_file=None, previous_attempt_dir=None):
    """Extend the existing reconciled source permission for this campaign only.

    Original STOPs, grants, chains, counters and limits remain untouched. This
    cannot release physical locks, validate a failed acquisition or clear a
    new source failure; live admission remains the enclosing parent's job.
    """
    import hashlib
    from datetime import datetime, timezone
    path = Path(budget_path).resolve(strict=True)
    authorization = Path(authorization_file).resolve(strict=True)
    rows = sorted(set(row_ids))
    _reentry_require(path == authorization.parents[2] / "energy/energy_task_budget.json"
                     and authorization.name == "AUFTRAG_AKTUELL.txt" and rows,
                     "continuation authorization must belong to this report campaign")
    text = authorization.read_text(encoding="utf-8")
    _reentry_require("REVISION 3 TATSÄCHLICH FERTIGSTELLEN" in text
                     and "durchgehend modellbegleiteter Abschluss" in text,
                     "continuous user authorization missing")
    if authorized_limits is not None:
        _reentry_require(diagnosis_file is not None and previous_attempt_dir is not None,
                         "administrative retry requires diagnosis and exact clean predecessor")
        return _authorize_closed_retry_continuation(path, source_id, authorization, rows,
                                                    authorized_limits, diagnosis_file, previous_attempt_dir)
    _reentry_require(diagnosis_file is None and previous_attempt_dir is None, "unexpected retry evidence without limits")
    for row_id in rows:
        identity = json.loads(row_id)
        _reentry_require(isinstance(identity, list) and len(identity) == 4,
                         "exact existing campaign rows required")
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX); root = json.load(handle)
        _reentry_require(root.get("campaign") is True and source_id in root["sources"], "source outside campaign")
        import csv
        report = authorization.parents[2]
        with (report / "REMEASUREMENT_PLAN.csv").open(newline="") as stream:
            required = list(csv.DictReader(stream))
        allowed = {json.dumps([r["backend"], r["model"], r["case"], r["setup"]], separators=(",", ":"))
                   for r in required if r["regime"] == "native" and r["energy_decision"] == "remeasure_required"}
        generic = json.loads((report / "current_results/generic_current.json").read_text())["rows"]
        required_generic = {(r["model"], r["case"], r["setup"]) for r in required
                            if r["regime"] == "generic_completed" and r["energy_decision"] == "remeasure_required"}
        allowed.update(json.dumps(["generic_completed:" + r["comparison_backend"], r["model_id"], r["case_id"], r["setup_id"]], separators=(",", ":"))
                       for r in generic if (r["model_id"], r["case_id"], r["setup_id"]) in required_generic)
        setups = {json.loads(r)[3] for r, t in root["tasks"].items() if t["source_id"] == source_id}
        _reentry_require(set(rows).issubset(allowed) and all(json.loads(r)[3] in setups for r in rows),
                         "continuation rows outside existing source/Delta scope")
        source = root["sources"][source_id]
        old = source.get("continuous_admission")
        if old is not None:
            _reentry_require(old["row_ids"] == rows and not old["source_stop_reason"]
                             and not any(old["task_stop_reasons"].values()), "new STOP needs normal recovery/diagnosis")
            _source_reentry(root, source_id, {})
            return copy.deepcopy(old)
        previous = source.get("authorized_reentry")
        if previous is not None:
            _reentry_require(previous["recovery_reply"]["recovery"]["status"] == "released"
                             and not previous["source_stop_reason"], "previous physical recovery not released")
            row, chain = previous["campaign_row"], previous["chain"]
            prior = [copy.deepcopy(previous)]
            basis = {"kind": "released_normal_recovery", "operation_id": previous["capture_request"]["operation_id"],
                     "historical_source_completion_verified": chain.get("source_completion_verified") is True}
        elif source.get("unstarted_capture_reconciliation"):
            key = source.get("unstarted_capture_reconciliation")
            receipt = _validate_unstarted_capture_evidence(root, source_id)
            _reentry_require(isinstance(receipt, dict) and receipt.get("source_id") == source_id,
                             "existing zero-start reconciliation required")
            row, chain = receipt["campaign_row"], receipt["booked_chain"]
            _reentry_require(chain.get("collector_started") is False and chain.get("finished") is True
                             and source.get("post_unstarted_capture_stop_reason") == "",
                             "zero-start source not reconciled")
            prior = []
            basis = {"kind": "existing_zero_start_reconciliation", "receipt": key, "operation_id": receipt["operation_id"],
                     "historical_source_completion_verified": False, "physical_start": False}
        else:
            _reentry_require(not source.get("stop_reason"), "unrecovered source STOP")
            completed = [(r, c) for r, t in root["tasks"].items() if t["source_id"] == source_id
                         for c in t["chains"] if c.get("finished") and c.get("valid") and c.get("source_completion_verified")]
            _reentry_require(bool(completed), "normal completed source basis missing")
            row, chain = completed[0]; prior = []
            basis = {"kind": "existing_normally_closed_source", "historical_source_completion_verified": True}
        for task_row, task in root["tasks"].items():
            if task["source_id"] != source_id:continue
            for entry in task["chains"]:
                _reentry_require(not _chain_owner_alive(entry) or entry.get("finished") is True,
                                 "active unfinished source chain")
                if entry.get("finished") is not True or (entry.get("collector_started") and entry.get("source_completion_verified") is not True):
                    _reentry_require(any(entry == g["chain"] for g in prior) or entry == chain,
                                     "additional unresolved source chain needs normal recovery")
        grant = {"continuous_campaign": True, "authorization_file": str(authorization),
                 "authorization_sha256": hashlib.sha256(authorization.read_bytes()).hexdigest(),
                 "authorized_at": datetime.now(timezone.utc).isoformat(), "row_ids": rows,
                 "campaign_row": row, "chain": copy.deepcopy(chain), "previous_grants": prior,
                 "original_source_stop_reason": source["stop_reason"],
                 "original_task_stops": {r: root["tasks"].get(r, {}).get("stop_reason", "") for r in rows},
                 "source_stop_reason": "", "task_stop_reasons": {}, "limit_changes": {},
                 "max_transport_failures": root["max_transport_failures"], "basis": basis,
                 "normal_live_ownership_resource_source_admission_required": True}
        source["continuous_admission"] = grant
        handle.seek(0);json.dump(root, handle, indent=2);handle.truncate();handle.flush();os.fsync(handle.fileno())
        return copy.deepcopy(grant)


def authorized_campaign_row_policy(budget_path, row_id, environment=None):
    """Read the explicit row override; preserved other sources need no new grant."""
    path = Path(budget_path)
    if not path.exists():
        return None
    _reentry_require(path.is_file() and not path.is_symlink(), "regular budget required")
    with path.open(encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_SH)
        root = json.load(handle)
    sources = [source_id for source_id, source in root.get("sources", {}).items()
               if row_id in (source.get("authorized_reentry") or {}).get("limit_changes", {})]
    if not sources:
        return None
    _reentry_require(len(sources) == 1, "ambiguous authorized row")
    grant = _source_reentry(root, sources[0], environment)
    change = grant["limit_changes"][row_id]
    return {"before": copy.deepcopy(change["before"]), "after": copy.deepcopy(change["after"]),
            "reconnect_backoff_s": 10.0}


def _h8_limit_changes(root, source_id, capture_row, requested, request, recovery):
    """The expressly authorized three H8 rows, without changing any counters."""
    expected = {json.dumps(["hailo8_to_trt", model, case, "orin_nx_hailo8_01"], separators=(",", ":"))
                for model, case in (("yolo26s", "b038"), ("yolov7_paper", "b044"), ("yolov7_paper", "b066"))}
    before = {"max_retries": 2, "max_chains": 9, "max_transport_failures": 20}
    after = {"max_retries": 5, "max_chains": 18, "max_transport_failures": 20}
    # The source address is deployment data. Bind it to the released original
    # capture and its four original resource owners, never to a source-code IP.
    original = recovery.get("original_resources") or {}
    identity = ("run_id", "session_id", "operation_id")
    _reentry_require(request.get("setup_id") == "orin_nx_hailo8_01"
                     and "source:" + source_id in original
                     and all((item.get("quarantine", {}).get("owner", {}).get("resource_request") == request
                              and item.get("lock_owner", {}).get("physical_resource") == resource
                              and all(item.get("lock_owner", {}).get(key) == request.get(key)
                                      for key in identity))
                             for resource, item in original.items()),
                     "H8 original capture resources differ")
    _reentry_require(set(requested) == expected
                     and json.loads(capture_row) == ["hailo8_to_trt", "yolo26s", "b038", "orin_nx_hailo8_01"]
                     and root["max_transport_failures"] == 20
                     and all(value == after for value in requested.values()), "H8 row limit scope differs")
    changes = {}
    for row in sorted(expected):
        task = root["tasks"].get(row)
        if task is not None:
            _reentry_require(task["source_id"] == source_id and task["limits"] == before
                             and (row == capture_row or
                                  (not task["chains"] and task["stop_reason"] in
                                   {"", "campaign_source_interrupted_chain_unresolved"})),
                             "H8 limit change has unrelated prior work or stop")
        else:
            _reentry_require(json.loads(row)[2] == "b066", "expected historical H8 row missing")
        changes[row] = {"before": copy.deepcopy(task["limits"]) if task is not None else None,
                        "after": after.copy(), "original_stop_reason": task["stop_reason"] if task is not None else ""}
    return changes


def install_authorized_source_reentry(budget_path, *, capture_request, recovery_reply, new_parent, authorized_limits=None):
    """Bind one released, explicitly authorized capture to one live parent.

    The caller holds the existing run fence and all four recovery resource
    locks. Only the existing budget inode is locked here. No historical chain,
    stop reason or counter is changed, and this creates no measurement. The
    optional explicit H8 limit increase retains its previous values in the grant.
    """
    from ..process_control import _proc_start_time
    from ..remote.resource_recovery import _retry_authorization
    request, reply = capture_request, recovery_reply
    recovery = reply.get("recovery") or {}
    identity = ("run_id", "session_id", "operation_id")
    _reentry_require(all(request.get(k) and reply.get(k) == request[k]
                         and recovery.get(k) == request[k] for k in identity)
                     and reply.get("token") == request.get("token")
                     and re.fullmatch(r"[0-9a-f]{32}", str(request.get("token") or ""))
                     and request.get("purpose", "capture") == "capture"
                     and request.get("phase") == "FINALIZING" and request.get("process_started") is True
                     and reply.get("state") == "STOP"
                     and reply.get("reason") == "campaign_source_completion_unresolved"
                     and recovery.get("status") == "released"
                     and recovery.get("old_attempt_validated") is False
                     and recovery.get("measurements_started") == 0
                     and (recovery.get("ownership_observation") or {}).get("ok") is True,
                     "exact released capture evidence missing")
    resources = recovery.get("resources") or []
    _reentry_require(len(resources) == len(set(resources)) == 4
                     and {"controller:capture", "controller:nic"}.issubset(resources)
                     and sum(r.startswith("source:") for r in resources) == 1
                     and sum(r.startswith("dut:") for r in resources) == 1,
                     "four released capture resources required")
    source = next(r for r in resources if r.startswith("source:"))
    original_resources = recovery.get("original_resources") or {}
    _reentry_require(set(original_resources) == set(resources), "original resource evidence incomplete")
    stopped = max(item["quarantine"]["quarantined_at"] for item in original_resources.values())
    authorization = _retry_authorization(recovery.get("retry_authorization"), source, stopped)
    _reentry_require("operator_evidence" not in recovery and recovery.get("released_at")
                     and set(new_parent) == {"run_id", "session_id", "pid", "start_ticks"}
                     and new_parent["run_id"] == request["run_id"]
                     and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", str(new_parent["session_id"]))
                     and new_parent["session_id"] != request["session_id"]
                     and new_parent["pid"] == os.getpid()
                     and new_parent["start_ticks"] == _proc_start_time(os.getpid()),
                     "one current new parent required")
    path = Path(budget_path)
    _reentry_require(path.is_file() and not path.is_symlink(), "existing regular budget required")
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        root = json.load(handle)
        source_id = source.removeprefix("source:")
        _reentry_require(root.get("campaign") is True and source_id in root["sources"], "campaign source missing")
        state = root["sources"][source_id]
        continuous = state.get("continuous_admission")
        if continuous is not None:
            _source_reentry(root, source_id, {})
            _reentry_require(authorized_limits is None, "continuous recovery cannot change limits")
            if continuous.get("capture_request") == request:
                _reentry_require(continuous.get("recovery_reply") == reply
                                 and not continuous["source_stop_reason"]
                                 and not any(continuous["task_stop_reasons"].values()),
                                 "later STOP requires its own exact released recovery")
                return copy.deepcopy(continuous)
            matches = [(key, task, chain) for key, task in root["tasks"].items()
                       if task["source_id"] == source_id for chain in task["chains"]
                       if chain.get("run_directory") == request.get("attempt_dir")]
            _reentry_require(len(matches) == 1, "current continuous capture chain missing or ambiguous")
            row, task, chain = matches[0]
            _reentry_require(row in continuous["row_ids"]
                             and json.loads(row)[3] == request["setup_id"]
                             and all(chain.get(k) == request.get(k) for k in ("owner_pid", "owner_start_ticks"))
                             and not _chain_owner_alive(chain)
                             and chain.get("finished") is True and chain.get("valid") is False
                             and chain.get("collector_started") is True and chain.get("workload_started") is True
                             and chain.get("source_completion_verified") is False and bool(chain.get("transport_reasons"))
                             and continuous["source_stop_reason"] == reply["reason"]
                             and continuous["task_stop_reasons"] == {row: reply["reason"]},
                             "released recovery does not cover the exact current continuous STOP")
            import yaml
            profile = yaml.safe_load(Path(authorization["new_workflow_profile"]).read_text())
            parent_root = path.resolve().parent / request["run_id"]
            if json.loads(row)[0].startswith("generic_completed:"):
                parent_root = path.resolve().parent / "generic_completed" / request["run_id"]
                capture = Path(request["attempt_dir"])
                _reentry_require(capture.is_absolute() and not capture.is_symlink()
                                 and capture.resolve().is_relative_to(parent_root),
                                 "Generic recovery capture outside its exact parent")
                parts = capture.relative_to(parent_root).parts
                if parts and parts[0] == "measurement":
                    parent_attempt = parent_root
                elif len(parts) >= 4 and parts[0] == "resume_attempts" and parts[2] == "measurement" \
                        and re.fullmatch(r"[0-9a-f]{32}", parts[1]):
                    parent_attempt = parent_root / "resume_attempts" / parts[1]
                else:
                    raise ValueError("Generic recovery capture attempt layout mismatch")
                from ..remote.resource_recovery import _bounded_energy_journal
                from ..remote.process_lease import RemoteProcessLeaseJournal
                journal, _, _ = _bounded_energy_journal(parent_root, request["session_id"], parent_attempt)
                read_exact = RemoteProcessLeaseJournal._read_json_exact
                _reentry_require(read_exact(journal / (request["operation_id"] + ".resource-request.json")) == request
                                 and read_exact(journal / (request["operation_id"] + ".resource-reply.json")) == reply,
                                 "Generic released journal differs from its exact budget handoff")
            _reentry_require(profile.get("run_dir") == str(parent_root)
                             and profile.get("historical_operation") == request["operation_id"]
                             and profile.get("source") == source_id and profile.get("setup_id") == request["setup_id"]
                             and profile.get("same_original_budget") == str(path.resolve())
                             and profile.get("unchanged_limits") == task["limits"]
                             and profile.get("source_completion_remains_unproven") is True,
                             "continuous recovery profile differs from this stopped parent/budget")
            unresolved = [c for t in root["tasks"].values() if t["source_id"] == source_id for c in t["chains"]
                          if (not c.get("finished") or (c.get("collector_started") and c.get("source_completion_verified") is not True))
                          and not any(c == old["chain"] for old in _reentry_history(continuous))]
            _reentry_require(unresolved == [chain], "another unresolved continuous source chain exists")
            _reentry_require(state.get("transport_failures", 0) < root["max_transport_failures"]
                             and len(task["chains"]) < task["limits"]["max_chains"]
                             and task["limits"]["max_transport_failures"] == root["max_transport_failures"],
                             "existing continuous budget exhausted or inconsistent")
            grant = copy.deepcopy(continuous)
            grant.update(campaign_row=row, chain=copy.deepcopy(chain), previous_grants=[copy.deepcopy(continuous)],
                         capture_request=copy.deepcopy(request), recovery_reply=copy.deepcopy(reply),
                         retry_authorization=authorization, new_parent=new_parent,
                         source_stop_reason="", task_stop_reasons={},
                         basis={"kind": "released_normal_recovery", "operation_id": request["operation_id"],
                                "historical_source_completion_verified": False})
            state["continuous_admission"] = copy.deepcopy(grant)
            handle.seek(0); json.dump(root, handle, indent=2); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
            return copy.deepcopy(grant)
        previous = state.get("authorized_reentry")
        if previous is not None and previous["capture_request"] == request:
            _reentry_require(previous["new_parent"] == new_parent
                             and previous["capture_request"] == request
                             and previous["recovery_reply"] == reply
                             and ({key: value["after"] for key, value in previous.get("limit_changes", {}).items()} or None)
                                 == authorized_limits, "authorization already consumed")
            return copy.deepcopy(previous)
        if previous is not None:
            old_parent = previous["new_parent"]
            _reentry_require(request["session_id"] == old_parent["session_id"]
                             and old_parent["run_id"] == request["run_id"]
                             and _proc_start_time(old_parent["pid"]) != old_parent["start_ticks"]
                             and previous["source_stop_reason"] == "campaign_source_completion_unresolved"
                             and not previous.get("limit_changes") and authorized_limits is None
                             and all(item["lock_owner"]["pid"] == old_parent["pid"]
                                     for item in original_resources.values()),
                             "new recovery must bind the failed authorized parent")
            for covered in _reentry_history(previous):
                old_task = root["tasks"][covered["campaign_row"]]
                _reentry_require(old_task["source_id"] == source_id and covered["chain"] in old_task["chains"]
                                 and old_task["limits"] == covered["limits"]
                                 and old_task["stop_reason"] == covered["original_task_stop_reason"]
                                 and state["stop_reason"] == covered["original_source_stop_reason"],
                                 "previous recovery history changed")
        matches = [(key, task, chain) for key, task in root["tasks"].items()
                   if task["source_id"] == source_id for chain in task["chains"]
                   if chain.get("run_directory") == request.get("attempt_dir")]
        _reentry_require(len(matches) == 1, "capture budget chain missing or ambiguous")
        row, task, chain = matches[0]
        row_identity = json.loads(row)
        _reentry_require(isinstance(row_identity, list) and len(row_identity) == 4
                         and row_identity[3] == request.get("setup_id")
                         and Path(chain["run_directory"]).is_absolute()
                         and all(chain.get(k) == request.get(k) for k in ("owner_pid", "owner_start_ticks"))
                         and not _chain_owner_alive(chain)
                         and (chain.get("finished") is False or
                              (chain.get("finished") is True and chain.get("valid") is False
                               and chain.get("source_completion_verified") is False
                               and bool(chain.get("transport_reasons"))))
                         and chain.get("collector_started") is True and chain.get("valid") is not True
                         and task["stop_reason"] in {"task_execution_interrupted", "task_interrupted_chain_unresolved",
                                                    "campaign_source_completion_unresolved"}
                         and state["stop_reason"] in {"", "campaign_source_interrupted_chain_unresolved",
                                                      "campaign_source_completion_unresolved"},
                         "only the exact historical interrupted capture is admissible")
        unresolved = [c for t in root["tasks"].values() if t["source_id"] == source_id for c in t["chains"]
                      if (not c.get("finished") or (c.get("collector_started") and c.get("source_completion_verified") is not True))
                      and not any(c == old["chain"] for old in _reentry_history(previous))]
        _reentry_require(unresolved == [chain], "another unresolved source chain exists")
        _reentry_require(state.get("transport_failures", 0) < root["max_transport_failures"]
                         and len(task["chains"]) < task["limits"]["max_chains"]
                         and task["limits"]["max_transport_failures"] == root["max_transport_failures"],
                         "existing budget exhausted or inconsistent")
        grant = {"new_parent": new_parent, "capture_request": request, "recovery_reply": reply,
                 "retry_authorization": authorization, "campaign_row": row, "chain": chain,
                 "limits": task["limits"], "max_transport_failures": root["max_transport_failures"],
                 "original_task_stop_reason": task["stop_reason"], "original_source_stop_reason": state["stop_reason"],
                 "task_stop_reasons": {}, "source_stop_reason": ""}
        if previous is not None:
            grant["previous_grants"] = [copy.deepcopy(previous)]
        if authorized_limits is not None:
            changes = _h8_limit_changes(root, source_id, row, authorized_limits, request, recovery)
            for changed_row, change in changes.items():
                if changed_row in root["tasks"]:
                    root["tasks"][changed_row]["limits"] = change["after"].copy()
            grant["limit_changes"] = changes
            grant["limits"] = task["limits"].copy()
        state["authorized_reentry"] = copy.deepcopy(grant)
        handle.seek(0)
        json.dump(root, handle, indent=2)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        return copy.deepcopy(grant)


def continue_authorized_source_reentry(budget_path, *, previous_attempt_dir, new_parent):
    """Continue a cleanly closed bounded parent without spending a new grant.

    Caller holds the run lock and all four physical locks. A new source STOP,
    unfinished acquisition, active predecessor or changed budget remains fatal.
    Only the parent binding advances; every chain, limit and stop is retained.
    """
    from ..process_control import _proc_start_time
    from ..remote.resource_recovery import _bounded_energy_journal, _rest_evidence_unchanged
    from ..remote.process_lease import RemoteProcessLeaseJournal
    read = RemoteProcessLeaseJournal._read_json_exact
    path = Path(budget_path).resolve(strict=True)
    run = path.parent
    attempt = Path(previous_attempt_dir).resolve(strict=True)
    start = read(attempt / "production-start.json")
    directory, _, evidence = _bounded_energy_journal(run, start.get("session_id"), attempt)
    config = read(directory / "controller.resources.json")
    cleanup = read(attempt / "cleanup-status.json")
    previous = {"run_id": start["run_id"], "session_id": start["session_id"],
                "pid": config.get("parent_pid"), "start_ticks": config.get("parent_start_ticks")}
    _reentry_require(config.get("run_id") == run.name and config.get("run_dir") == str(run)
                     and config.get("session_id") == start["session_id"]
                     and start["parent_pid"] == previous["pid"]
                     and str(previous["start_ticks"] or "").isdigit()
                     and _proc_start_time(previous["pid"]) != previous["start_ticks"]
                     and cleanup.get("cleanup_proven") is True and cleanup.get("errors") == []
                     and not list(directory.glob("*.remote-lease.json")),
                     "previous bounded parent is not clean and quiescent")
    _reentry_require(set(new_parent) == {"run_id", "session_id", "pid", "start_ticks"}
                     and new_parent["run_id"] == run.name
                     and re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", str(new_parent["session_id"]))
                     and new_parent["session_id"] != previous["session_id"]
                     and new_parent["pid"] == os.getpid()
                     and new_parent["start_ticks"] == _proc_start_time(os.getpid()),
                     "one current continuation parent required")
    requests = list(directory.glob("*.resource-request.json"))
    for request_path in requests:
        request = read(request_path)
        reply_path = directory / (request_path.name.removesuffix(".resource-request.json") + ".resource-reply.json")
        reply = read(reply_path)
        _reentry_require(request.get("run_id") == run.name
                         and request.get("session_id") == previous["session_id"]
                         and reply.get("state") == "RELEASED"
                         and all(reply.get(k) == request.get(k) for k in
                                 ("run_id", "session_id", "operation_id", "token")),
                         "previous bounded operation is not released")
        evidence.update({request_path: request, reply_path: reply})
    evidence[directory / "controller.resources.json"] = config
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        root = json.load(handle)
        matches = [(source_id, state, state.get("authorized_reentry"))
                   for source_id, state in root.get("sources", {}).items()
                   if (state.get("authorized_reentry") or {}).get("new_parent") == previous]
        _reentry_require(len(matches) == 1, "exact previous continuation grant missing")
        source_id, state, grant = matches[0]
        if not requests:
            # A clean parent may fail before creating its first operation.
            # Require its original handoff receipt to prove that even the
            # effective grant has not changed since that zero-dispatch start.
            receipt_path = attempt / "source-budget-handoff.json"
            _reentry_require(receipt_path.is_file() and not receipt_path.is_symlink()
                             and not list(directory.glob("*.resource-reply.json")),
                             "zero-operation parent has no exact handoff receipt")
            receipt = read(receipt_path)
            _reentry_require(receipt == grant, "zero-operation parent grant changed")
            evidence[receipt_path] = receipt
        task = root["tasks"][grant["campaign_row"]]
        _reentry_require(not grant["source_stop_reason"] and not any(grant["task_stop_reasons"].values())
                         and task["limits"] == grant["limits"]
                         and root["max_transport_failures"] == grant["max_transport_failures"]
                         and task["stop_reason"] == grant["original_task_stop_reason"]
                         and state["stop_reason"] == grant["original_source_stop_reason"]
                         and [c for c in task["chains"] if c.get("run_directory") == grant["chain"]["run_directory"]]
                             == [grant["chain"]], "continuation budget or stop changed")
        for other in root["tasks"].values():
            if other["source_id"] != source_id:
                continue
            for chain in other["chains"]:
                if any(chain == old["chain"] for old in _reentry_history(grant)):
                    continue
                _reentry_require(chain.get("finished") is True
                                 and (not chain.get("collector_started") or
                                      (chain.get("source_completion_verified") is True
                                       and source_completion(chain["run_directory"])["verified"])),
                                 "another source acquisition is unresolved")
        _reentry_require(_rest_evidence_unchanged(evidence), "clean continuation evidence changed")
        grant.setdefault("parent_history", []).append({"parent": previous,
            "attempt_dir": str(attempt), "terminal_evidence": {str(p): value for p, value in evidence.items()}})
        grant["new_parent"] = copy.deepcopy(new_parent)
        handle.seek(0)
        json.dump(root, handle, indent=2)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        return copy.deepcopy(grant)


def _finished_chain(chain, entry, run_dir, *, campaign):
    """One classification for live completion and evidence-bound reconciliation."""
    from .collector import (_energy_repeat_retry_reasons,
                            _energy_repeat_failure_text,
                            _energy_repeat_aggregation_eligible)
    result = copy.deepcopy(chain)
    reasons = _energy_repeat_retry_reasons(entry, run_dir)
    transport = [r for r in reasons if r in {
        "first_sample_barrier_invalid", "marker_dropped_samples_nonzero",
        "marker_trace_does_not_cover_window",
    }]
    text = _energy_repeat_failure_text(entry, run_dir).lower()
    if any(t in text for t in ("socket error", "connection reset",
                               "sending half is closed", "urecs_transport_unavailable")):
        transport.append("transport_log_error")
    timing = entry.get("workload_timing") or {}
    result.update(finished=True, status=entry.get("status"),
                 collector_started=chain["collector_started"] or entry.get("collector_started") is True,
                 workload_started=bool(timing.get("start_ns")),
                 valid=_energy_repeat_aggregation_eligible(entry, require_ab=False),
                 transport_reasons=transport)
    if campaign:
        completion = source_completion(run_dir)
        result["source_completion"] = completion
        result["source_completion_verified"] = completion["verified"] if result["collector_started"] else None
        if result["collector_started"] and not completion["verified"]:
            result["valid"] = False
    return result


class EnergyTaskBudget:
    def __init__(self, handle, limits, cancel_event=None, *, campaign_row=None, source_id=None, environment=None, cohort_id=None):
        positive_transport_failure_limit(limits.get("max_transport_failures"))
        self.handle = handle
        self.checkpoint_path = Path(handle.name) if isinstance(handle.name, str) else None
        self.cancel_event = cancel_event
        self.campaign_row = campaign_row
        self.source_id = source_id
        self.environment = environment
        self.cohort_id = cohort_id or ""
        self._reentry = None
        self._mutex = threading.RLock()
        self._transaction_depth = 0
        self._initialized = False
        self.busy_reason = ""
        with self.transaction():
            self._initialize(limits, campaign_row=campaign_row, source_id=source_id)
            self.save()
        self._initialized = True

    @contextmanager
    def transaction(self):
        # The existing checkpoint inode is locked only for local JSON updates.
        # Waiting for a Future, a remote process or a collector never owns it.
        with self._mutex:
            outer = self._transaction_depth == 0
            if outer:
                fcntl.flock(self.handle, fcntl.LOCK_EX)
                if self._initialized:
                    self.handle.seek(0)
                    self.root = json.load(self.handle)
                    self.data = self.root["tasks"][self.campaign_row] if self.campaign_row is not None else self.root
                    self.source = self.root["sources"][self.source_id] if self.campaign_row is not None else None
                    self._reentry = _source_reentry(self.root, self.source_id, self.environment) if self.source is not None else None
                    self._check_cohort()
            self._transaction_depth += 1
            try:
                yield
            finally:
                self._transaction_depth -= 1
                if outer:
                    fcntl.flock(self.handle, fcntl.LOCK_UN)

    def _initialize(self, limits, *, campaign_row, source_id):
        self.handle.seek(0)
        content = self.handle.read()
        self.root = json.loads(content) if content else None
        if self.cohort_id:
            self._check_cohort()
        self.source = None
        if campaign_row is not None:
            if not source_id or not campaign_row:
                raise ValueError("campaign budget requires physical source and stable row identity")
            if self.root is None:
                self.root = {"campaign": True, "tasks": {}, "sources": {},
                             "max_transport_failures": limits["max_transport_failures"]}
            if (self.root.get("campaign") is not True
                    or self.root["max_transport_failures"] != limits["max_transport_failures"]):
                raise ValueError("campaign budget contract differs from existing checkpoint")
            self.source = self.root["sources"].setdefault(source_id, {"stop_reason": ""})
            self.data = self.root["tasks"].setdefault(campaign_row, {
                "limits": limits, "chains": [], "stop_reason": "", "source_id": source_id})
            if self.data["source_id"] != source_id:
                raise ValueError("campaign row physical source changed")
            self._reentry = _source_reentry(self.root, source_id, self.environment)
            if self._reentry is not None and self._reentry.get("continuous_campaign"):
                _reentry_require(campaign_row in self._reentry["row_ids"], "row outside authorized campaign")
                _budget_parent(self.environment)
            for task in self.root["tasks"].values():
                if task["source_id"] == source_id and any(
                        (not c.get("finished") and not _chain_owner_alive(c)) or
                        (c.get("finished") and c.get("collector_started") and c.get("source_completion_verified") is not True)
                        for c in task["chains"] if not self._covered_chain(c)):
                    # Re-entry must not replace the original finished attempt's
                    # failure with the fallback for an interrupted checkpoint.
                    self._stop_source("campaign_source_interrupted_chain_unresolved")
            self._check_source()
        else:
            self.data = self.root if self.root is not None else {
            "limits": limits, "chains": [], "stop_reason": "",
            }
            self.root = self.data
        if self.data["limits"] != limits:
            raise ValueError("energy task limits differ from existing checkpoint")
        self._check_cohort()
        if any(not chain.get("finished") and not _chain_owner_alive(chain)
               for chain in self.data["chains"] if not self._covered_chain(chain)):
            self.stop("task_interrupted_chain_unresolved")

    def _covered_chain(self, chain):
        return any(chain == old["chain"] for old in _reentry_history(self._reentry))

    def _check_cohort(self):
        if not self.cohort_id:
            return
        grant = final_cohort_authorization(self.root, self.cohort_id)
        parent = _budget_parent(self.environment)
        env = {**os.environ, **(self.environment or {})}
        directory = Path(env.get("ONNX_SPLITPOINT_REMOTE_LEASE_JOURNAL_DIR") or "")
        from ..remote.process_lease import RemoteProcessLeaseJournal
        cfg = RemoteProcessLeaseJournal._read_json_exact(directory / "controller.resources.json")
        if (grant.get("cohort_id") != self.cohort_id or grant.get("purpose") != "final"
                or self.campaign_row not in grant.get("row_ids", []) or grant.get("source_id") != self.source_id
                or parent["run_id"] != self.cohort_id or cfg.get("run_dir") != grant.get("parent_run_dir")):
            raise ValueError("unapproved final cohort, task or parent generation")

    def matching_chains(self, logical_repeat):
        return [c for c in self.data["chains"] if c.get("logical_repeat") == logical_repeat
                and (c.get("cohort_id") or "") == self.cohort_id]

    def _stop_reason(self):
        if self._reentry is None:
            return (_campaign_task_stop(self.root, self.campaign_row, self.source_id, self.environment)
                    if self.campaign_row is not None else self.data["stop_reason"])
        return (self._reentry["task_stop_reasons"].get(self.campaign_row, "")
                or _historical_task_stop(self.data, self.campaign_row, self._reentry))

    def _effective_limits(self, logical_repeat):
        limits = dict(self.data["limits"])
        change = (self._reentry or {}).get("limit_changes", {}).get(self.campaign_row)
        first = True
        while change is not None:
            if change.get("cohort_id") == self.cohort_id:
                if first:
                    limits.update({key: change["after"][key] for key in ("max_chains", "max_transport_failures")})
                    first = False
                if change.get("logical_repeat") == logical_repeat:
                    limits["max_retries"] = change["after"]["max_retries"]
                    break
            change = change.get("previous_change")
        return limits

    def _source_failure_limit(self):
        return ((self._reentry or {}).get("sample_loss_retries") or {}).get(
            "effective_source_failure_limit", self.root.get("max_transport_failures"))

    @_budget_transaction
    def authorized_retry_limit(self):
        change = (self._reentry or {}).get("limit_changes", {}).get(self.campaign_row)
        return change["after"]["max_retries"] if change and change.get("cohort_id") == self.cohort_id else None

    @_budget_transaction
    def extend_sample_loss_retries(self, logical_repeat):
        """Permit a pure-loss retry, recording +3 only when a bound is spent."""
        policy = (self._reentry or {}).get("sample_loss_retries")
        if (policy is None or self.campaign_row not in policy["row_ids"]
                or logical_repeat not in {"repeat:0", "repeat:1", "repeat:2"}):
            return False
        attempts = self.matching_chains(logical_repeat)
        if not attempts or any(c.get("valid") is True or c.get("finished") is not True for c in attempts):
            return False
        last = attempts[-1]
        if (last.get("collector_started") is not True or last.get("valid") is not False
                or last.get("source_completion_verified") is not True
                or last.get("transport_reasons") != ["marker_dropped_samples_nonzero"]):
            return False
        administrative = {"", "task_logical_retry_limit", "task_chain_limit", "task_transport_failure_limit",
                          "campaign_source_transport_failure_limit"}
        if self._stop_reason() not in administrative or self._reentry["source_stop_reason"] not in {"", "campaign_source_transport_failure_limit"}:
            return False
        from ..process_control import current_process_registry
        registry = current_process_registry()
        if ((self.cancel_event is not None and self.cancel_event.is_set()) or registry is not None and registry.cancelled):
            return False
        for task in self.root["tasks"].values():
            if task["source_id"] != self.source_id:
                continue
            for chain in task["chains"]:
                if self._covered_chain(chain):
                    continue
                if not chain.get("finished") or (chain.get("collector_started") and chain.get("source_completion_verified") is not True):
                    return False
        try:
            if not source_completion(last["run_directory"])["verified"] or not _pure_sample_loss_failure(last, self.campaign_row, self.cohort_id):
                return False
            self._check_cohort()
            with _released_sample_loss_resources(last, self.source_id, self.environment) as release:
                limits = self._effective_limits(logical_repeat)
                counts = self.counts()
                source_count = sum(bool(c.get("transport_reasons")) for task in self.root["tasks"].values()
                                   if task["source_id"] == self.source_id for c in task["chains"])
                if (len(attempts) < 1 + limits["max_retries"]
                        and counts["begun_chains"] < limits["max_chains"]
                        and counts["transport_failures"] < limits["max_transport_failures"]
                        and source_count < self._source_failure_limit() and not self._stop_reason()
                        and not self._reentry["source_stop_reason"]):
                    return True
                previous = copy.deepcopy(self._reentry.get("limit_changes", {}).get(self.campaign_row))
                before = previous["after"] if previous else dict(self.data["limits"])
                after = {key: value + 3 for key, value in before.items()}
                change = {"kind": "sample_loss_series", "before": copy.deepcopy(self.data["limits"]),
                          "after": after, "previous_change": previous, "cohort_id": self.cohort_id,
                          "logical_repeat": logical_repeat, "original_chains": copy.deepcopy(self.data["chains"]),
                          "last_chain": copy.deepcopy(last), "released_capture": release,
                          "previous_task_stop": self._stop_reason(), "previous_source_stop": self._reentry["source_stop_reason"]}
                source_before = self._source_failure_limit()
                policy["series_history"].append({"campaign_row": self.campaign_row, "cohort_id": self.cohort_id,
                    "logical_repeat": logical_repeat, "series_starts": 3, "source_before": source_before,
                    "source_after": source_before + 3, "source_transport_failures": source_count,
                    "task_counts": counts, "last_run_directory": last["run_directory"], "parent": release["parent"]})
                policy["effective_source_failure_limit"] = source_before + 3
                self._reentry.setdefault("limit_changes", {})[self.campaign_row] = change
                self._reentry["task_stop_reasons"].pop(self.campaign_row, None)
                if self._reentry["source_stop_reason"] == "campaign_source_transport_failure_limit":
                    self._reentry["source_stop_reason"] = ""
                self.save()
                return True
        except (OSError, ValueError, KeyError, TypeError, RuntimeError):
            return False

    def _stop_source(self, reason):
        state = self._reentry if self._reentry is not None else self.source
        key = "source_stop_reason" if self._reentry is not None else "stop_reason"
        if self._reentry is None and self.source.get("unstarted_capture_reconciliation"):
            key = "post_unstarted_capture_stop_reason"
        if not state[key]:
            state[key] = reason

    def save(self):
        # Inode stays fixed across short transactions. A torn checkpoint
        # fails JSON decoding on re-entry; it must never reset a used budget.
        self.handle.seek(0)
        json.dump(self.root, self.handle, indent=2)
        self.handle.truncate()
        self.handle.flush()
        import os
        os.fsync(self.handle.fileno())

    @_budget_transaction
    def stop(self, reason):
        if not self._stop_reason():
            if self._reentry is not None:
                self._reentry["task_stop_reasons"][self.campaign_row] = reason
            elif self.campaign_row is not None and "post_unstarted_capture_stop_reason" in self.data:
                self.data["post_unstarted_capture_stop_reason"] = reason
            else:
                self.data["stop_reason"] = reason
            self.save()

    @_budget_transaction
    def is_set(self):
        self._check_source()
        from ..process_control import current_process_registry
        registry = current_process_registry()
        if ((self.cancel_event is not None and self.cancel_event.is_set())
                or (registry is not None and registry.cancelled)):
            self.stop("task_cancelled")
        return bool(self._stop_reason())

    def _check_source(self):
        reason = (self._reentry["source_stop_reason"] if self._reentry is not None
                  else _campaign_source_stop(self.root, self.source_id, self.campaign_row, self.environment)
                       if self.source is not None else "")
        if reason:
            self.stop(reason)

    @_budget_transaction
    def counts(self):
        chains = self.data["chains"]
        return {
            "begun_chains": len(chains),
            "preflight_chains": sum(c["preflight_requested"] for c in chains),
            "collector_starts": sum(c["collector_started"] for c in chains),
            "workload_starts": sum(c.get("workload_started", False) for c in chains),
            "valid_logical_repeats": len({(c.get("cohort_id") or "",c["logical_repeat"]) for c in chains if c.get("valid")}),
            "selected_cohort_valid_repeats": sum(c.get("valid") is True and (c.get("cohort_id") or "") == self.cohort_id for c in chains),
            "transport_failures": sum(bool(c.get("transport_reasons")) for c in chains),
        }

    @_budget_transaction
    def reusable_repeat(self, logical_repeat, *, command, command_contract_sha256, duration_s, require_ab):
        """Read an already accepted campaign repeat; never reserve it again."""
        if self.campaign_row is None:
            raise ValueError("repeat reuse requires the existing campaign budget")
        candidates = [c for c in self.matching_chains(logical_repeat)
                      if c.get("finished") is True and c.get("valid") is True]
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError("campaign repeat has ambiguous accepted attempts")
        chain = candidates[0]
        entry = read_verified_campaign_repeat(
            chain, command=command, command_contract_sha256=command_contract_sha256,
            duration_s=duration_s, require_ab=require_ab,
        )
        if self.checkpoint_path is None:
            raise ValueError("campaign checkpoint path missing for repeat reuse")
        reference = {"budget_file": str(self.checkpoint_path.resolve()),
                       "campaign_row": self.campaign_row, "logical_repeat": logical_repeat,
                       "run_directory": str(Path(chain["run_directory"]).resolve())}
        if self.cohort_id: reference["cohort_id"] = self.cohort_id
        return entry, reference

    @_budget_transaction
    def allowed(self, logical_repeat, cancel_event=None):
        self._check_cohort()
        if self.cohort_id and logical_repeat not in {"repeat:0","repeat:1","repeat:2"}:
            raise ValueError("final cohort repeat outside authorized three slots")
        if any(not c.get("finished") and not self._covered_chain(c) for c in self.matching_chains(logical_repeat)):
            self.busy_reason = "energy_repeat_busy:" + logical_repeat
            return False
        if self.is_set() or (cancel_event is not None and cancel_event.is_set()):
            self.stop("task_cancelled")
            return False
        # A live owner on this source is a concrete conflict, not evidence of
        # source failure. Dead owners are handled by initialization/re-entry.
        from ..process_control import _proc_start_time
        tasks = self.root["tasks"].values() if self.source is not None else [self.data]
        for task in tasks:
            if self.source is not None and task["source_id"] != self.source_id:
                continue
            for chain in task["chains"]:
                if chain.get("finished") or self._covered_chain(chain):
                    continue
                own = task is self.data and chain.get("owner_pid") == os.getpid() and chain.get("owner_start_ticks") == _proc_start_time(os.getpid())
                if not own:
                    self.busy_reason = "energy_source_busy:live_chain:" + str(chain["run_directory"])
                    if not _chain_owner_alive(chain):
                        if self.source is not None:
                            self._stop_source("campaign_source_interrupted_chain_unresolved")
                        self.stop("campaign_source_interrupted_chain_unresolved" if self.source is not None else "task_interrupted_chain_unresolved")
                    return False
        if self.source is not None and any(
                c.get("finished") is True and c.get("valid") is True for c in self.matching_chains(logical_repeat)):
            # Campaign row/repeat identities survive direct CLI re-entry. A
            # fresh output directory must not reacquire an accepted repeat.
            # Keep the original result and source usable; no recovery or replay
            # of old result files is attempted by this admission check.
            self.busy_reason = "energy_repeat_already_valid:" + str(logical_repeat)
            return False
        limits = self._effective_limits(logical_repeat)
        counts = self.counts()
        if counts["transport_failures"] >= limits["max_transport_failures"]:
            self.stop("task_transport_failure_limit")
        elif counts["begun_chains"] >= limits["max_chains"]:
            self.stop("task_chain_limit")
        elif sum(self.source is not None or c["collector_started"]
                 for c in self.matching_chains(logical_repeat)) >= 1 + limits["max_retries"]:
            self.stop("task_logical_retry_limit")
        return not self.is_set()

    @_budget_transaction
    def reserve(self, logical_repeat, run_dir, preflight_requested, cancel_event=None):
        if not self.allowed(logical_repeat, cancel_event):
            return None
        # Never overwrite a previous attempt when the caller re-enters with the
        # same output directory. Use a fresh output and the same task checkpoint.
        if (run_dir / "energy_summary.json").exists():
            self.stop("task_attempt_output_exists")
            return None
        from ..process_control import _proc_start_time
        chain = {"owner_pid": os.getpid(), "owner_start_ticks": _proc_start_time(os.getpid()),
                 "logical_repeat": logical_repeat, "run_directory": str(run_dir),
                 "preflight_requested": bool(preflight_requested),
                 "collector_started": False, "finished": False}
        if self.cohort_id: chain["cohort_id"] = self.cohort_id
        self.data["chains"].append(chain)
        self.save()
        return dict(chain)

    def _current_chain(self, identity):
        matches = [chain for chain in self.data["chains"] if chain["run_directory"] == identity["run_directory"]]
        if len(matches) != 1 or any(matches[0].get(key) != identity.get(key) for key in ("owner_pid", "owner_start_ticks")):
            raise ValueError("energy chain ownership changed")
        return matches[0]

    @_budget_transaction
    def collector_started(self, chain):
        identity = chain
        chain = self._current_chain(chain)
        chain["collector_started"] = True
        identity["collector_started"] = True
        self.save()

    @_budget_transaction
    def finish(self, chain, entry, run_dir):
        chain = self._current_chain(chain)
        if chain["finished"]:
            return
        if (self.cohort_id or getattr(self, "require_environment_observation", False)) and chain.get("collector_started"):
            from .environment_observer import validate_environment
            if self.cohort_id: entry["campaign_cohort"] = self.cohort_id
            else:
                entry["environment_observation_required"] = True
                entry["campaign_run_id"] = _budget_parent(self.environment)["run_id"]
            entry["campaign_row"] = self.campaign_row
            environment = validate_environment(run_dir, entry, freeze=True)
            entry["environment_observation"] = environment
            from .collector import _write_json
            if not environment["ok"]:
                entry.update(ok=False, energy_efficiency_claim_eligible=False,
                             scientific_primary_claim_eligible=False,
                             environment_observation_error=environment["reason"])
            _write_json(Path(run_dir) / "energy_summary.json", entry)
        chain.update(_finished_chain(chain, entry, run_dir, campaign=self.source is not None))
        if self.source is None:
            self.save()
        if self.source is not None:
            # A new process/output directory cannot erase an unresolved source.
            # Both messages must come from this attempt's original collector log.
            completion = source_completion(run_dir)
            verified = completion["verified"]
            chain["source_completion"] = completion
            chain["source_completion_verified"] = verified if chain["collector_started"] else None
            if chain["collector_started"] and not verified:
                chain["valid"] = False
                self._stop_source("campaign_source_completion_unresolved")
            source_failures = sum(bool(c.get("transport_reasons"))
                                  for task in self.root["tasks"].values()
                                  if task["source_id"] == self.data["source_id"]
                                  for c in task["chains"])
            self.source["transport_failures"] = source_failures
            if source_failures >= self._source_failure_limit():
                self._stop_source("campaign_source_transport_failure_limit")
            self._check_source()
            self.save()
        if self.counts()["transport_failures"] >= self._effective_limits(chain["logical_repeat"])["max_transport_failures"]:
            self.stop("task_transport_failure_limit")

    @_budget_transaction
    def report(self, result, out_dir):
        from .collector import _write_json
        self.is_set()
        if not result.get("ok"):
            chains = self.data["chains"]
            for cohort, logical in {(c.get("cohort_id") or "",c["logical_repeat"]) for c in chains}:
                attempts = [c for c in chains if (c.get("cohort_id") or "") == cohort and c["logical_repeat"] == logical and c["collector_started"]]
                if (len(attempts) >= 1 + (self._effective_limits(logical) if cohort == self.cohort_id else self.data["limits"])["max_retries"]
                        and all(c.get("finished") for c in attempts)
                        and not any(c.get("valid") for c in attempts)):
                    self.stop("task_logical_retry_limit")
        reason = self._stop_reason()
        if self.busy_reason and not reason:
            result.update(ok=False, status="not_dispatched", reason=self.busy_reason, physical_dispatch_started=False)
        result["task_budget"] = {**self.data, "counts": self.counts()}
        if (self._reentry or {}).get("sample_loss_retries"):
            result["task_budget"]["effective_limits"] = {logical: self._effective_limits(logical) for logical in ("repeat:0", "repeat:1", "repeat:2")}
        if self.source is not None:
            result["task_budget"]["campaign_source"] = dict(self.source)
        if self._reentry is not None or self.data.get("unstarted_capture_reconciliation"):
            result["task_budget"]["effective_stop_reason"] = reason
        if reason:
            result.update(ok=False, status="incomplete_task_budget", terminal=True,
                          error=reason, repeat_contract_complete=False,
                          energy_efficiency_claim_eligible=False,
                          scientific_primary_claim_eligible=False)
            if self.source is not None:
                if reason != "task_cancelled":
                    result["cancelled"] = False
                result.update(status="BLOCKED", execution_status="NOT_RUN" if not self.counts()["collector_starts"] else "INCOMPLETE")
        else:
            result["terminal"] = True
        _write_json(Path(out_dir) / "energy_aggregate.json", result)
        _write_json(Path(out_dir) / "energy_summary.json", result)
        _write_json(Path(out_dir) / "energy_current_status.json", {
            "status": result.get("status"), "terminal": True,
            "reason": reason, "task_budget_counts": self.counts(),
        })
        return result


def read_verified_campaign_repeat(chain, *, command, command_contract_sha256, duration_s, require_ab):
    """Verify the original accepted attempt with its existing local evidence."""
    from .collector import (_energy_repeat_aggregation_eligible, _sha256_path,
                            _sha256_text, _validate_energy_preflight_attestation)
    root = Path(str(chain.get("run_directory") or ""))
    if (not root.is_absolute() or root.resolve() != root or chain.get("finished") is not True
        or chain.get("valid") is not True or chain.get("source_completion_verified") is not True
        or not source_completion(root).get("verified")):
        raise ValueError("campaign repeat source completion invalid")
    entry = json.loads((root / "energy_summary.json").read_text(encoding="utf-8"))
    if chain.get("cohort_id") or entry.get("environment_observation_required"):
        from .environment_observer import validate_environment
        environment = entry.get("environment_observation") or {}
        if (entry.get("campaign_cohort") != chain.get("cohort_id") or entry.get("diagnostic_only") is True
                or not validate_environment(root,entry)["ok"]
                or _sha256_path(Path(environment.get("evidence_path") or "")) != environment.get("evidence_sha256")):
            raise ValueError("final cohort repeat environment/purpose binding invalid")
    if not _energy_repeat_aggregation_eligible(entry, require_ab=require_ab):
        raise ValueError("campaign repeat no longer valid")
    if float(entry.get("energy_configured_workload_duration_s") or 0) != float(duration_s):
        raise ValueError("campaign repeat duration drift")
    preflight = entry.get("preflight_evidence") or {}
    if (preflight.get("ok") is not True or preflight.get("workload_template_sha256") != _sha256_text(command)
        or preflight.get("expected_command_contract_sha256") != command_contract_sha256):
        raise ValueError("campaign repeat command contract drift")
    attestation_path = root / "preflight" / "preflight_attestation.json"
    if _sha256_path(attestation_path) != preflight.get("attestation_file_sha256"):
        raise ValueError("campaign repeat attestation changed")
    validation = _validate_energy_preflight_attestation(
        json.loads(attestation_path.read_text(encoding="utf-8")), nonce=preflight.get("nonce"),
        started_at_unix_ns=preflight.get("started_at_unix_ns"),
        ended_at_unix_ns=preflight.get("ended_at_unix_ns"),
        max_age_s=(preflight.get("validation") or {}).get("max_age_s"),
        expected_command_contract_sha256=command_contract_sha256,
    )
    if validation.get("ok") is not True:
        raise ValueError("campaign repeat original preflight invalid")
    return entry


def bounded_energy_task(function):
    @wraps(function)
    def run(command, out_dir, **kwargs):
        campaign = kwargs.pop("campaign_budget_file", None)
        cohort_id = kwargs.pop("campaign_cohort", None)
        require_environment = bool(kwargs.pop("require_environment_observation", False) or cohort_id)
        environment_run_id = kwargs.pop("environment_run_id",None)
        if require_environment and (not campaign or kwargs.get("diagnostic_only") or not kwargs.get("require_runtime_work_units") or not kwargs.get("require_command_window_alignment")):
            raise ValueError("required environment requires non-diagnostic campaign command-window task measurement")
        campaign_row = kwargs.pop("campaign_row_id", None)
        campaign_repeats = kwargs.pop("campaign_repeats", None)
        campaign_retries = kwargs.pop("campaign_max_retries", None)
        campaign_failures = kwargs.pop("campaign_max_transport_failures", None)
        checkpoint = kwargs.pop("task_budget_file", None)
        max_chains = kwargs.pop("task_max_chains", None)
        max_failures = kwargs.pop("task_max_transport_failures", None)
        if campaign:
            if cohort_id and (kwargs.get("diagnostic_only") or not kwargs.get("require_runtime_work_units")
                              or not kwargs.get("require_command_window_alignment")):
                raise ValueError("final cohort requires non-diagnostic exact command-window task measurement")
            if checkpoint or max_chains is not None or max_failures is not None:
                raise ValueError("campaign and diagnostic task budgets cannot be combined")
            if (type(campaign_repeats) is not int or campaign_repeats < 1
                    or type(campaign_retries) is not int or campaign_retries < 0 or not campaign_row):
                raise ValueError("campaign requires positive repeats, nonnegative retries and row identity")
            checkpoint = campaign
            max_chains = campaign_repeats * (1 + campaign_retries)
            max_failures = positive_transport_failure_limit(campaign_failures)
            # The explicit campaign policy also drives the existing outer
            # invalid-repeat loop when the caller supplies no dedicated override.
            # Recursive single-attempt calls pass zero and share this budget.
            if kwargs.get("invalid_repeat_max_retries") is None:
                kwargs["invalid_repeat_max_retries"] = campaign_retries
        elif any(v is not None for v in (cohort_id, campaign_row, campaign_repeats, campaign_retries, campaign_failures)):
            raise ValueError("campaign options require a campaign checkpoint")
        shared = kwargs.get("_task_budget")
        if shared is not None:
            return shared.report(function(command, out_dir, **kwargs), out_dir)
        if checkpoint is None and max_chains is None and max_failures is None:
            return function(command, out_dir, **kwargs)
        retries = kwargs.get("invalid_repeat_max_retries")
        if retries is None:
            retries = getattr(kwargs.get("defaults"), "invalid_repeat_max_retries", 1)
        if campaign and retries > campaign_retries:
            raise ValueError("collector retries exceed campaign retry contract")
        max_failures = positive_transport_failure_limit(max_failures)
        if (not checkpoint or type(max_chains) is not int or max_chains < 1
                or type(retries) is not int or retries < 0):
            raise ValueError("bounded energy task requires checkpoint, positive limits and nonnegative retries")
        import fcntl
        path = Path(checkpoint).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        import os
        with os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o600), "r+", encoding="utf-8") as handle:
            budget = EnergyTaskBudget(handle, {"max_chains": max_chains,
                "max_transport_failures": max_failures, "max_retries": campaign_retries if campaign else retries},
                kwargs.get("cancel_event"), campaign_row=campaign_row if campaign else None,
                source_id=str(getattr(kwargs.get("setup"), "urecs_address", "")).strip().lower() if campaign else None,
                environment=kwargs.get("subprocess_env"), cohort_id=cohort_id)
            budget.checkpoint_path = path
            kwargs.update(_task_budget=budget, cancel_event=budget)
            budget.require_environment_observation = require_environment
            if environment_run_id and _budget_parent(kwargs.get('subprocess_env'))['run_id'] != environment_run_id:
                raise ValueError("environment parent identity differs from actual owner")
            try:
                if budget.is_set():
                    result = {"ok": False, "status": "incomplete", "runs": []}
                else:
                    if require_environment:
                        from .environment_observer import EnergyEnvironmentObserver
                        with EnergyEnvironmentObserver(kwargs['setup'],out_dir,kwargs.get('subprocess_env'),cohort_id or _budget_parent(kwargs.get('subprocess_env'))['run_id']):
                            result = function(command, out_dir, **kwargs)
                    else:
                        result = function(command, out_dir, **kwargs)
            except BaseException as exc:
                budget.stop("task_execution_interrupted")
                try:
                    budget.report({"ok": False, "runs": [], "exception": type(exc).__name__,
                                   "exception_detail": str(exc)}, out_dir)
                except Exception:
                    pass  # Preserve the original execution exception.
                raise
            return budget.report(result, out_dir)
    return run


def reserve_controlled_test_start(kind, identity, *, journal_path=None):
    """Opt-in acceptance counters: reserve before dispatch, never reset on resume.

    This is a count bound in the explicitly supplied test journal; it adds no
    measurement duration limit and is absent from ordinary user workflows.
    """
    path = journal_path or os.environ.get("ONNX_SPLITPOINT_TEST_START_JOURNAL")
    if not path:
        return None
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        value = json.load(handle)
        if value.get("schema") != "onnx-splitpoint/controlled-test-starts":
            raise ValueError("controlled test start journal schema mismatch")
        limit = value["limits"].get(kind)
        if type(limit) is not int or limit < 1:
            raise ValueError("controlled test start limit missing")
        entries = value["entries"]
        if any(row["kind"] == kind and row["identity"] == str(identity) for row in entries):
            raise RuntimeError("controlled test attempt already reserved")
        if sum(row["kind"] == kind for row in entries) >= limit:
            raise RuntimeError("controlled test physical start budget exhausted:" + kind)
        import time
        row = {"kind": kind, "identity": str(identity), "reserved_at_unix": time.time(), "owner_pid": os.getpid()}
        entries.append(row)
        handle.seek(0)
        json.dump(value, handle, indent=2)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
        return row
