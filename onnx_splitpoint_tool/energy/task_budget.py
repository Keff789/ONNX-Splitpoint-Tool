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
    parser.add_argument("--campaign-budget-file", default=None)
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
        return []
    path, retries, failures = values
    if not path or type(retries) is not int or retries < 0:
        raise ValueError("incomplete campaign budget arguments")
    failures = positive_transport_failure_limit(failures)
    return ["--campaign-budget-file", str(Path(path).expanduser().resolve()),
            "--campaign-max-retries", str(retries), "--campaign-max-transport-failures", str(failures)]


def campaign_budget_measurement_kwargs(args):
    return {name: getattr(args, name, None) for name in (
        "campaign_budget_file", "campaign_row_id", "campaign_repeats",
        "campaign_max_retries", "campaign_max_transport_failures", "preflight_prepare_command")}


def _budget_transaction(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.transaction():
            return method(self, *args, **kwargs)
    return call


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
    return grant


def _historical_task_stop(task, row_id, grant):
    if row_id == grant["campaign_row"] or row_id in grant.get("limit_changes", {}):
        return ""
    return task["stop_reason"]


def campaign_budget_stop_reason(root, row_id, environment=None):
    """Project the effective refusal without rewriting a historical STOP."""
    task = root["tasks"][row_id]
    source = root["sources"][task["source_id"]]
    grant = _source_reentry(root, task["source_id"], environment)
    if grant is None:
        return str(source.get("stop_reason") or task.get("stop_reason") or "")
    return str(grant["source_stop_reason"] or grant["task_stop_reasons"].get(row_id)
               or _historical_task_stop(task, row_id, grant))


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
        previous = state.get("authorized_reentry")
        if previous is not None:
            _reentry_require(previous["new_parent"] == new_parent
                             and previous["capture_request"] == request
                             and previous["recovery_reply"] == reply
                             and ({key: value["after"] for key, value in previous.get("limit_changes", {}).items()} or None)
                                 == authorized_limits, "authorization already consumed")
            return copy.deepcopy(previous)
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
                         and not _chain_owner_alive(chain) and chain.get("finished") is False
                         and chain.get("collector_started") is True and chain.get("valid") is not True
                         and task["stop_reason"] in {"task_execution_interrupted", "task_interrupted_chain_unresolved"}
                         and state["stop_reason"] in {"", "campaign_source_interrupted_chain_unresolved",
                                                      "campaign_source_completion_unresolved"},
                         "only the exact historical interrupted capture is admissible")
        unresolved = [c for t in root["tasks"].values() if t["source_id"] == source_id for c in t["chains"]
                      if not c.get("finished") or (c.get("collector_started") and c.get("source_completion_verified") is not True)]
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


class EnergyTaskBudget:
    def __init__(self, handle, limits, cancel_event=None, *, campaign_row=None, source_id=None, environment=None):
        positive_transport_failure_limit(limits.get("max_transport_failures"))
        self.handle = handle
        self.checkpoint_path = Path(handle.name) if isinstance(handle.name, str) else None
        self.cancel_event = cancel_event
        self.campaign_row = campaign_row
        self.source_id = source_id
        self.environment = environment
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
        if any(not chain.get("finished") and not _chain_owner_alive(chain)
               for chain in self.data["chains"] if not self._covered_chain(chain)):
            self.stop("task_interrupted_chain_unresolved")

    def _covered_chain(self, chain):
        return self._reentry is not None and chain == self._reentry["chain"]

    def _stop_reason(self):
        if self._reentry is None:
            return self.data["stop_reason"]
        return (self._reentry["task_stop_reasons"].get(self.campaign_row, "")
                or _historical_task_stop(self.data, self.campaign_row, self._reentry))

    @_budget_transaction
    def authorized_retry_limit(self):
        change = (self._reentry or {}).get("limit_changes", {}).get(self.campaign_row)
        return change["after"]["max_retries"] if change is not None else None

    def _stop_source(self, reason):
        state = self._reentry if self._reentry is not None else self.source
        key = "source_stop_reason" if self._reentry is not None else "stop_reason"
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
                  else self.source["stop_reason"] if self.source is not None else "")
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
            "valid_logical_repeats": len({c["logical_repeat"] for c in chains if c.get("valid")}),
            "transport_failures": sum(bool(c.get("transport_reasons")) for c in chains),
        }

    @_budget_transaction
    def reusable_repeat(self, logical_repeat, *, command, command_contract_sha256, duration_s, require_ab):
        """Read an already accepted campaign repeat; never reserve it again."""
        if self.campaign_row is None:
            raise ValueError("repeat reuse requires the existing campaign budget")
        candidates = [c for c in self.data["chains"] if c.get("logical_repeat") == logical_repeat
                      and c.get("finished") is True and c.get("valid") is True]
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
        return entry, {"budget_file": str(self.checkpoint_path.resolve()),
                       "campaign_row": self.campaign_row, "logical_repeat": logical_repeat,
                       "run_directory": str(Path(chain["run_directory"]).resolve())}

    @_budget_transaction
    def allowed(self, logical_repeat, cancel_event=None):
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
                c["logical_repeat"] == logical_repeat and c.get("finished") is True
                and c.get("valid") is True for c in self.data["chains"]):
            # Campaign row/repeat identities survive direct CLI re-entry. A
            # fresh output directory must not reacquire an accepted repeat.
            # Keep the original result and source usable; no recovery or replay
            # of old result files is attempted by this admission check.
            self.busy_reason = "energy_repeat_already_valid:" + str(logical_repeat)
            return False
        limits = self.data["limits"]
        counts = self.counts()
        if counts["transport_failures"] >= limits["max_transport_failures"]:
            self.stop("task_transport_failure_limit")
        elif counts["begun_chains"] >= limits["max_chains"]:
            self.stop("task_chain_limit")
        elif sum(c["logical_repeat"] == logical_repeat and (self.source is not None or c["collector_started"])
                 for c in self.data["chains"]) >= 1 + limits["max_retries"]:
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
        from .collector import (_energy_repeat_retry_reasons,
                                _energy_repeat_failure_text,
                                _energy_repeat_aggregation_eligible)
        if chain["finished"]:
            return
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
        chain.update(finished=True, status=entry.get("status"),
                     collector_started=chain["collector_started"] or entry.get("collector_started") is True,
                     workload_started=bool(timing.get("start_ns")),
                     valid=_energy_repeat_aggregation_eligible(entry, require_ab=False),
                     transport_reasons=transport)
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
            if source_failures >= self.root["max_transport_failures"]:
                self._stop_source("campaign_source_transport_failure_limit")
            self._check_source()
            self.save()
        if self.counts()["transport_failures"] >= self.data["limits"]["max_transport_failures"]:
            self.stop("task_transport_failure_limit")

    @_budget_transaction
    def report(self, result, out_dir):
        from .collector import _write_json
        self.is_set()
        if not result.get("ok"):
            chains = self.data["chains"]
            for logical in {c["logical_repeat"] for c in chains}:
                attempts = [c for c in chains if c["logical_repeat"] == logical and c["collector_started"]]
                if (len(attempts) >= 1 + self.data["limits"]["max_retries"]
                        and all(c.get("finished") for c in attempts)
                        and not any(c.get("valid") for c in attempts)):
                    self.stop("task_logical_retry_limit")
        reason = self._stop_reason()
        if self.busy_reason and not reason:
            result.update(ok=False, status="not_dispatched", reason=self.busy_reason, physical_dispatch_started=False)
        result["task_budget"] = {**self.data, "counts": self.counts()}
        if self.source is not None:
            result["task_budget"]["campaign_source"] = dict(self.source)
        if self._reentry is not None:
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
        campaign_row = kwargs.pop("campaign_row_id", None)
        campaign_repeats = kwargs.pop("campaign_repeats", None)
        campaign_retries = kwargs.pop("campaign_max_retries", None)
        campaign_failures = kwargs.pop("campaign_max_transport_failures", None)
        checkpoint = kwargs.pop("task_budget_file", None)
        max_chains = kwargs.pop("task_max_chains", None)
        max_failures = kwargs.pop("task_max_transport_failures", None)
        if campaign:
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
        elif any(v is not None for v in (campaign_row, campaign_repeats, campaign_retries, campaign_failures)):
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
                environment=kwargs.get("subprocess_env"))
            budget.checkpoint_path = path
            kwargs.update(_task_budget=budget, cancel_event=budget)
            try:
                if budget.is_set():
                    result = {"ok": False, "status": "incomplete", "runs": []}
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
