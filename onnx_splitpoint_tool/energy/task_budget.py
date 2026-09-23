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
from pathlib import Path

DEFAULT_CAMPAIGN_BUDGET = {"enabled": True, "max_retries": 1, "max_transport_failures": 2}


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
    if (type(retries) is not int or retries < 0
            or type(policy.get("max_transport_failures")) is not int
            or policy["max_transport_failures"] not in (1, 2)):
        raise ValueError("native energy task_budget requires nonnegative max_retries and max_transport_failures in {1,2}")
    return {"enabled": True, "max_retries": retries, "max_transport_failures": policy["max_transport_failures"],
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
    parser.add_argument("--campaign-max-transport-failures", type=int, default=None)
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
    if (not path or type(retries) is not int or retries < 0
            or type(failures) is not int or failures not in (1, 2, 3)):
        raise ValueError("incomplete campaign budget arguments")
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


class EnergyTaskBudget:
    def __init__(self, handle, limits, cancel_event=None, *, campaign_row=None, source_id=None):
        self.handle = handle
        self.cancel_event = cancel_event
        self.campaign_row = campaign_row
        self.source_id = source_id
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
            for task in self.root["tasks"].values():
                if task["source_id"] == source_id and any(
                        (not c.get("finished") and not _chain_owner_alive(c)) or
                        (c.get("finished") and c.get("collector_started") and c.get("source_completion_verified") is not True)
                        for c in task["chains"]):
                    # Re-entry must not replace the original finished attempt's
                    # failure with the fallback for an interrupted checkpoint.
                    if not self.source["stop_reason"]:
                        self.source["stop_reason"] = "campaign_source_interrupted_chain_unresolved"
            self._check_source()
        else:
            self.data = self.root if self.root is not None else {
            "limits": limits, "chains": [], "stop_reason": "",
            }
            self.root = self.data
        if self.data["limits"] != limits:
            raise ValueError("energy task limits differ from existing checkpoint")
        if any(not chain.get("finished") and not _chain_owner_alive(chain) for chain in self.data["chains"]):
            self.stop("task_interrupted_chain_unresolved")

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
        if not self.data["stop_reason"]:
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
        return bool(self.data["stop_reason"])

    def _check_source(self):
        if self.source is not None and self.source["stop_reason"]:
            self.stop(self.source["stop_reason"])

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
                if chain.get("finished"):
                    continue
                own = task is self.data and chain.get("owner_pid") == os.getpid() and chain.get("owner_start_ticks") == _proc_start_time(os.getpid())
                if not own:
                    self.busy_reason = "energy_source_busy:live_chain:" + str(chain["run_directory"])
                    if not _chain_owner_alive(chain):
                        if self.source is not None:
                            self.source["stop_reason"] = "campaign_source_interrupted_chain_unresolved"
                        self.stop("campaign_source_interrupted_chain_unresolved" if self.source is not None else "task_interrupted_chain_unresolved")
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
                self.source["stop_reason"] = "campaign_source_completion_unresolved"
            source_failures = sum(bool(c.get("transport_reasons"))
                                  for task in self.root["tasks"].values()
                                  if task["source_id"] == self.data["source_id"]
                                  for c in task["chains"])
            self.source["transport_failures"] = source_failures
            if source_failures >= self.root["max_transport_failures"]:
                self.source["stop_reason"] = self.source["stop_reason"] or "campaign_source_transport_failure_limit"
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
        if self.busy_reason and not self.data["stop_reason"]:
            result.update(ok=False, status="not_dispatched", reason=self.busy_reason, physical_dispatch_started=False)
        result["task_budget"] = {**self.data, "counts": self.counts()}
        if self.source is not None:
            result["task_budget"]["campaign_source"] = dict(self.source)
        if self.data["stop_reason"]:
            result.update(ok=False, status="incomplete_task_budget", terminal=True,
                          error=self.data["stop_reason"], repeat_contract_complete=False,
                          energy_efficiency_claim_eligible=False,
                          scientific_primary_claim_eligible=False)
            if self.source is not None:
                if self.data["stop_reason"] != "task_cancelled":
                    result["cancelled"] = False
                result.update(status="BLOCKED", execution_status="NOT_RUN" if not self.counts()["collector_starts"] else "INCOMPLETE")
        else:
            result["terminal"] = True
        _write_json(Path(out_dir) / "energy_aggregate.json", result)
        _write_json(Path(out_dir) / "energy_summary.json", result)
        _write_json(Path(out_dir) / "energy_current_status.json", {
            "status": result.get("status"), "terminal": True,
            "reason": self.data["stop_reason"], "task_budget_counts": self.counts(),
        })
        return result


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
                    or type(campaign_retries) is not int or campaign_retries < 0
                    or type(campaign_failures) is not int or campaign_failures not in (1, 2, 3) or not campaign_row):
                raise ValueError("campaign requires repeats, retries, row identity and one to three source transport failures")
            checkpoint = campaign
            max_chains = campaign_repeats * (1 + campaign_retries)
            max_failures = campaign_failures
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
        if (not checkpoint or any(type(v) is not int or v < 1 for v in (max_chains, max_failures))
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
                source_id=str(getattr(kwargs.get("setup"), "urecs_address", "")).strip().lower() if campaign else None)
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
