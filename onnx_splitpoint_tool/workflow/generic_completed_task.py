"""Selective Generic completed-task measurements outside a closed EvaluationRun.

This adapter owns transport and scheduling only. Runtime execution remains in
the generated Generic runner; physical ownership and SSH cleanup use the same
locks and process leases as the normal evaluation workflow.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import inspect
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shlex
import shutil
import signal
import socket
import stat
import tarfile
import threading
import time
import traceback
import uuid

from ..benchmark.schema import write_json_atomic
from ..remote.process_lease import RemoteProcessLeaseRegistry, RemoteProcessLeaseScope
from ..remote.resource_recovery_observation import _observe_processes
from ..remote.ssh_transport import HostConfig, SSHTransport
from ..runners.task_completion import normalize_stage1_execution
from .execution_binding import physical_dut_key
from .run_control import EvaluationRunLock


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(path), value)


def case_name(row: dict) -> str:
    identity = row.get("identity", row)
    values = [identity[k] for k in ("model_id", "case_id", "setup_id", "precision")]
    for value in values:
        if not isinstance(value, str) or not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in value):
            raise ValueError("unsafe completion case identity")
    return "__".join(values)


def outside_primary(path: Path, primary: Path) -> Path:
    path = Path(path).expanduser().resolve()
    primary = Path(primary).expanduser().resolve()
    if path == primary or primary in path.parents:
        raise ValueError("completion output must be outside the closed primary run")
    return path


def bound_targets(primary: Path, rows: list[dict], *, producer_run_binding=None) -> dict[str, dict]:
    if producer_run_binding is not None:
        from ..generic_producer_binding import validate_producer_run_binding
        source = validate_producer_run_binding(primary, producer_run_binding, rows)
        targets = {r["id"]: r for r in source["hardware_targets"]}
    else:
        targets = {r["id"]: r for r in read_json(primary / "run_manifest.json")["hardware_targets"]}
    wanted = {(r.get("identity") or r)["setup_id"] for r in rows}
    result = {}
    for setup_id in sorted(wanted):
        target = targets[setup_id]
        remote = target.get("remote") or target.get("runtime") or {}
        if not remote.get("host") or not remote.get("user"):
            raise ValueError("original target binding missing: " + setup_id)
        result[setup_id] = {**target, "remote": remote}
    keys = [physical_dut_key(r) for r in result.values()]
    if len(keys) != len(set(keys)):
        raise ValueError("selected setup aliases share a physical device")
    return result


def transport_for(target: dict, diagnostics: Path, *, scope=None, registry=None, cancel=None, log=print):
    remote = target["remote"]
    host = HostConfig.from_dict({**remote, "id": target["id"], "label": target.get("label", target["id"])})
    transport = SSHTransport(host, log=log, cancel_event=cancel,
                             remote_lease_scope=scope, remote_lease_registry=registry)
    transport.diagnostics_dir = diagnostics
    return transport


def _scoped_controller_owners(targets: dict[str, dict], observed: dict) -> dict:
    """Attribute live controller work through its held normal EvaluationRun.

    This is ordinary FPS admission, not recovery or proof of cleanup. Nothing
    is exempted by PID or executable name. An unknown owner remains blocking.
    The existing raw probe remains responsible for namespace/lease visibility.
    """
    from .run_control import build_resume_contract
    from .start_snapshot import build_profile_start_snapshot
    import yaml

    def resource_keys(target):
        remote = target.get("remote") or target.get("runtime") or target
        require(bool(remote.get("host")) and bool(remote.get("user")), "unbound_physical_host")
        # Retain both the explicit chassis identity and resolved transport host:
        # different aliases/registry labels cannot make one SSH host independent.
        return {physical_dut_key(target), physical_dut_key({"remote": {"host": remote["host"]}})}

    snapshots, identities, authorities = {}, {}, {}
    errors = []

    def require(condition, reason):
        if not condition:
            raise ValueError(reason)

    selected = set().union(*(resource_keys(target) for target in targets.values()))

    def read(path, *, yaml_file=False):
        path = Path(path)
        require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents))
                and stat.S_ISREG(path.stat().st_mode), "unsafe_owner_control_file")
        data = path.read_bytes()
        snapshots[path] = data
        value = yaml.safe_load(data) if yaml_file else json.loads(data)
        require(isinstance(value, dict), "invalid_owner_control_file")
        return value

    def process(pid):
        require(type(pid) is int and pid > 0, "invalid_observed_pid")
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        row = {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
               "start_ticks": fields[19], "state": fields[0]}
        require(row["state"] != "Z" and row["start_ticks"].isdecimal(), "owner_identity_changed")
        previous = identities.setdefault(pid, row)
        changed = {key: {"expected": previous[key], "actual": row[key]}
                   for key in ("pid", "ppid", "pgid", "start_ticks") if previous[key] != row[key]}
        require(not changed, "owner_identity_changed: " + json.dumps({"pid": pid, "changed": changed}, sort_keys=True))
        return row

    def held_locks(owner):
        result = []
        for fd in Path(f"/proc/{owner['pid']}/fd").iterdir():
            try:
                path = Path(os.readlink(fd))
            except FileNotFoundError:
                continue
            if path.parent.name != ".onnx_splitpoint_run_locks" or path.suffix != ".lock":
                continue
            metadata = path.stat()
            for line in Path(f"/proc/{owner['pid']}/fdinfo/{fd.name}").read_text().splitlines():
                parts = line.split()
                if len(parts) != 9 or parts[:1] != ["lock:"] or parts[2:5] != ["FLOCK", "ADVISORY", "WRITE"]:
                    continue
                device = parts[6].split(":")
                if (int(parts[5]) == owner["pid"] and len(device) == 3
                        and (int(device[0], 16), int(device[1], 16), int(device[2]))
                        == (os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino)):
                    result.append((path, read(path)))
        return result

    def authority(pid):
        visited = set()
        while pid > 1 and pid not in visited and len(visited) < 64:
            visited.add(pid)
            owner = process(pid)
            if pid in authorities:
                return authorities[pid]
            locks = held_locks(owner)
            for path, lock in locks:
                if lock.get("physical_resource"):
                    continue
                run = Path(str(lock.get("run_dir") or ""))
                session_id = str(lock.get("session_id") or "")
                require(lock.get("schema") == "onnx-splitpoint/evaluation-run-lock"
                        and lock.get("state") == "running" and lock.get("pid") == pid
                        and lock.get("ppid") == owner["ppid"] and lock.get("pgid") == owner["pgid"]
                        and lock.get("hostname") == socket.gethostname()
                        and re.fullmatch(r"[0-9a-f]{32}", session_id)
                        and run.is_absolute() and lock.get("run_id") == run.name
                        and EvaluationRunLock(out_root=run.parent, run_dir=run, owner={}).lock_path == path,
                        "unbound_normal_run_lock")
                boot = next(line.split()[1] for line in Path("/proc/stat").read_text().splitlines()
                            if line.startswith("btime "))
                birth = int(boot) + int(owner["start_ticks"]) / os.sysconf("SC_CLK_TCK")
                acquired = datetime.fromisoformat(str(lock.get("acquired_at") or "").replace("Z", "+00:00"))
                require(acquired.tzinfo is not None and birth <= acquired.timestamp() <= time.time(),
                        "run_lock_predates_process_identity")
                manifest = read(run / "run_manifest.json")
                control = read(run / "jobs/workflow_control.json")
                profile = read(run / "profile.yaml", yaml_file=True)
                plan = read(run / "effective_execution_plan.json")
                archived = read(run / "profile_start_snapshot.json")
                source = read(run / "profile_source.yaml", yaml_file=True)
                rebuilt = build_profile_start_snapshot(
                    **{key: str(archived.get(key) or "") for key in
                       ("profile_request", "profile_id", "profile_path", "profile_source")},
                    source_profile=source, resolved_profile=profile,
                    runtime_bindings=archived.get("runtime_bindings") or {},
                    schema_version=int(archived.get("schema_version") or 1))
                contract = build_resume_contract(profile_payload=profile,
                    effective_execution_plan=plan, options=manifest.get("options") or {})
                sessions = manifest.get("execution_sessions") or []
                require(manifest.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
                        and manifest.get("run_id") == run.name and manifest.get("run_dir") == str(run)
                        and manifest.get("current_session_id") == session_id
                        and sessions and sessions[-1].get("session_id") == session_id
                        and contract == manifest.get("resume_contract")
                        and all(lock.get(key) == contract[key] == sessions[-1].get(key)
                                for key in ("profile_sha256", "plan_sha256"))
                        and control.get("schema") == "onnx-splitpoint/workflow-control"
                        and control.get("run_id") == run.name and control.get("session_id") == session_id
                        and archived.get("snapshot_sha256") == rebuilt["snapshot_sha256"]
                        == (manifest.get("profile_start_snapshot") or {}).get("snapshot_sha256")
                        and (archived.get("runtime_bindings") or {}).get("hardware_targets")
                        == manifest.get("hardware_targets"),
                        "normal_run_session_contract_mismatch")
                runs = [row for row in profile.get("run_profiles", []) if row.get("enabled", True)]
                model_ids = {row.get("id") for row in (profile.get("model_suite") or {}).get("primary", [])}
                run_ids = {row.get("id") for row in runs}
                planned_groups = plan.get("setup_groups") or {}
                planned_runs = [run for group in planned_groups.values() for run in group]
                require((profile.get("energy") or {}).get("enabled") is False
                        and (profile.get("native_producers") or {}).get("enabled") is False
                        and runs and all(row.get("quality_source_only") is True for row in runs),
                        "owner_is_not_source_only_quality_workflow")
                require(model_ids and set(plan.get("models") or []) == model_ids
                        and set(plan.get("logical_run_profiles") or []) == run_ids
                        and len(planned_runs) == len(run_ids) and set(planned_runs) == run_ids,
                        "effective_plan_model_or_setup_runs_differ")
                targets_by_id = {row["id"]: row for row in manifest.get("hardware_targets", [])}
                setup_ids = (profile.get("hardware") or {}).get("selected_setups") or []
                require(setup_ids and set(setup_ids) == {row.get("hardware_setup_id") for row in runs}
                        and set(setup_ids) <= set(targets_by_id), "normal_run_resource_scope_unbound")
                resources = set().union(*(resource_keys(targets_by_id[setup]) for setup in setup_ids))
                physical = [value for physical_path, value in locks if value.get("physical_resource")
                            and physical_path == EvaluationRunLock.for_resource(
                                value["physical_resource"], owner={}).lock_path]
                require(len(physical) == sum(bool(value.get("physical_resource")) for _, value in locks),
                        "physical_lock_path_binding_mismatch")
                require(all(value.get("schema") == "onnx-splitpoint/evaluation-run-lock"
                            and value.get("state") == "running" and value.get("pid") == pid
                            and value.get("run_id") == run.name and value.get("purpose") == "generic"
                            and value.get("physical_resource") in resources for value in physical),
                        "normal_run_physical_owner_mismatch")
                require(not any(value["physical_resource"] in selected for value in physical),
                        "selected_physical_dut_is_active")
                central_only = False
                completed_setups = []
                if resources & selected:
                    terminal = {"ok", "completed", "partial", "failed", "skipped"}
                    models = manifest.get("models")
                    root_stages = manifest.get("root_stages")
                    smoke = profile.get("hardware_smoke")
                    require(isinstance(models, dict) and isinstance(root_stages, dict)
                            and isinstance(smoke, dict), "normal_stage_status_scope_unbound")
                    # workflow_stages is the normal runner's ordered stage-name
                    # list. Actual workflow status is recorded in root_stages.
                    quality_stage = root_stages.get("evaluate_quality", {})
                    require(isinstance(quality_stage, dict), "normal_quality_stage_status_unbound")
                    for model in models.values():
                        require(isinstance(model, dict) and isinstance(model.get("stages"), dict)
                                and isinstance(model["stages"].get("run_benchmarks", {}), dict),
                                "normal_model_stage_status_unbound")
                    central_only = bool(model_ids and model_ids <= set(models) and all(
                        ((model.get("stages") or {}).get("run_benchmarks") or {}).get("status") in terminal
                        for model in models.values()) and (
                        quality_stage.get("status") in terminal | {"running"}
                        or manifest.get("status") in terminal)
                        and smoke.get("mode") == "summary_only"
                        and smoke.get("require_remote_for_hailo") is False
                        and not physical)
                    journal = run / "jobs/remote_process_leases"
                    if central_only:
                        require(journal.is_dir() and not list(journal.rglob("*.remote-lease.json")),
                                "central_only_remote_operations_unresolved")
                    else:
                        # Per-setup dispatch can finish while another setup is
                        # still capturing. Only normal, complete dispatch and
                        # transport receipts can close the selected setup; an
                        # inner case event or a missing journal is insufficient.
                        require(smoke.get("mode") == "summary_only"
                                and smoke.get("require_remote_for_hailo") is False,
                                "selected_dut_workflow_has_not_entered_central_only_phase")
                        from .execution_binding import _remote_transport_run_id, _profile_remote_with_override

                        def timestamp(value):
                            stamp = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
                            require(stamp.tzinfo is not None, "selected_setup_terminal_time_unbound")
                            return stamp.timestamp()

                        completed_setups = [setup for setup in setup_ids
                            if resource_keys(targets_by_id[setup]) & selected]
                        completed_at = {}
                        for setup in completed_setups:
                            expected_runs = {row["id"] for row in runs
                                             if row.get("hardware_setup_id") == setup}
                            require(expected_runs, "selected_setup_planned_runs_missing")
                            target = targets_by_id[setup]
                            remote = target.get("remote") or target.get("runtime")
                            last_terminal = 0
                            for model_id in sorted(model_ids):
                                require(isinstance(model_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", model_id)
                                        and re.fullmatch(r"[A-Za-z0-9_.-]+", setup),
                                        "selected_setup_identity_unsafe")
                                directory = run / "models" / model_id / "benchmark_results"
                                dispatch_path = directory / ("remote_benchmark_dispatch_" + setup + ".json")
                                status_path = directory / ("remote_benchmark_status_" + setup + ".json")
                                require(dispatch_path.is_file() and status_path.is_file(),
                                        "selected_dut_workflow_has_not_entered_central_only_phase")
                                dispatch = read(dispatch_path)
                                status = read(status_path)
                                gates = dispatch.get("execution_gates")
                                require(isinstance(gates, dict), "selected_setup_dispatch_gates_unbound")
                                planned = gates.get("hardware_run_ids")
                                require(isinstance(planned, list) and len(planned) == len(expected_runs)
                                        and set(planned) == expected_runs
                                        and gates.get("hardware_run_id") in expected_runs
                                        and gates.get("hardware_target") == target,
                                        "selected_setup_planned_runs_differ")
                                expected_host = {"user": remote["user"], "host": remote["host"],
                                                 "port": int(remote.get("port", 22))}
                                host = dispatch.get("host")
                                require(isinstance(host, dict) and all(host.get(k) == v for k, v in expected_host.items())
                                        and dispatch.get("schema") == "onnx-splitpoint/remote-benchmark-dispatch"
                                        and dispatch.get("schema_version") == 1
                                        and dispatch.get("status") == "dispatching"
                                        and dispatch.get("hardware_target_id") == setup
                                        and dispatch.get("model_id") == model_id
                                        and status.get("schema") == "onnx-splitpoint/remote-benchmark-status"
                                        and status.get("schema_version") == 1
                                        and status.get("hardware_target_id") == setup
                                        and status.get("model_id") == model_id
                                        and status.get("status") in {"ok", "partial", "failed", "cancelled"}
                                        and status.get("host") == host and status.get("execution_gates") == gates
                                        and status.get("suite_dir") == dispatch.get("suite_dir"),
                                        "selected_setup_terminal_dispatch_binding_differs")
                                arguments = dispatch.get("args")
                                require(isinstance(arguments, dict)
                                        and arguments.get("quality_evidence_eval_id") == run.name
                                        and arguments.get("quality_evidence_model_id") == model_id
                                        and arguments.get("quality_evidence_setup_id") == setup
                                        and arguments.get("energy_enabled") is False,
                                        "selected_setup_quality_dispatch_scope_unbound")
                                transport_id = _remote_transport_run_id(run_root=run, model_id=model_id,
                                    target_id=setup, gates=gates, workflow_session_id=session_id)
                                require(dispatch.get("run_id") == transport_id,
                                        "selected_setup_transport_run_unbound")
                                local = Path(str(status.get("local_run_dir") or ""))
                                meta = read(local / "run_meta.json")
                                transport_status = read(local / "run_status.json")
                                requested = meta.get("resume") or {}
                                suite = Path(str(dispatch.get("suite_dir") or ""))
                                benchmark = Path(str(dispatch.get("benchmark_set_json") or ""))
                                require(not suite.is_absolute() and not benchmark.is_absolute()
                                        and ".." not in suite.parts and ".." not in benchmark.parts
                                        and bool(suite.parts) and benchmark.parent == suite,
                                        "selected_setup_suite_path_unbound")
                                working = Path(str(dispatch.get("local_working_dir") or "")).expanduser()
                                expected_working = Path(str((manifest.get("options") or {}).get("remote_working_dir")
                                    or _profile_remote_with_override(profile, remote).get("local_working_dir")
                                    or directory / "remote_runs")).expanduser().resolve()
                                require(working.is_absolute() and working == expected_working
                                        and meta.get("schema_version") == 1
                                        and meta.get("run_id") == local.name
                                        and requested.get("requested_run_id") == transport_id
                                        and requested.get("active_run_id") == local.name
                                        and meta.get("repeat_idx") == "1"
                                        and local == working / "Results" / suite.name / "1" / local.name
                                        and meta.get("host") == host
                                        and meta.get("suite_dir") == str(run / suite)
                                        and meta.get("benchmark_set_json") == str(run / benchmark)
                                        and isinstance(meta.get("args"), dict)
                                        and meta["args"].get("energy_enabled") is False
                                        and meta["args"].get("add_args") == arguments.get("add_args")
                                        and transport_status.get("schema_version") == 1
                                        and transport_status.get("status") in {"ok", "partial", "failed", "cancelled"}
                                        and type(transport_status.get("remote_rc")) is int
                                        and transport_status.get("started_at") == meta.get("started_at"),
                                        "selected_setup_transport_terminal_unbound")
                                start = timestamp(meta.get("started_at"))
                                end = timestamp(transport_status.get("ended_at"))
                                published = timestamp(status.get("created_at"))
                                require(acquired.timestamp() <= timestamp(dispatch.get("created_at")) <= start <= end <= published <= time.time(),
                                        "selected_setup_terminal_predates_current_owner")
                                last_terminal = max(last_terminal, end)
                            completed_at[setup] = last_terminal
                            # The normal physical resource gate still acquires
                            # at dispatch. Here require its genuine release and
                            # reject every current kernel holder without taking
                            # an observer lock or rewriting owner metadata.
                            for resource in resource_keys(target):
                                physical_lock = EvaluationRunLock.for_resource(resource, owner={})
                                require(not os.path.lexists(physical_lock.quarantine_path),
                                        "selected_setup_physical_quarantine_unresolved")
                                released = read(physical_lock.lock_path)
                                require(released.get("schema") == "onnx-splitpoint/evaluation-run-lock"
                                        and released.get("schema_version") == 1 and released.get("state") == "released"
                                        and released.get("physical_resource") == resource
                                        and released.get("run_id") == run.name and released.get("pid") == pid
                                        and released.get("purpose") == "generic"
                                        and released.get("hostname") == socket.gethostname()
                                        and released.get("ppid") == owner["ppid"]
                                        and released.get("pgid") == owner["pgid"]
                                        and ("session_id" not in released or released["session_id"] == session_id)
                                        and acquired.timestamp() <= timestamp(released.get("acquired_at")) <= last_terminal
                                        <= timestamp(released.get("released_at")) <= time.time(),
                                        "selected_setup_physical_release_unproven")
                                info = physical_lock.lock_path.stat()
                                key = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
                                for line in Path("/proc/locks").read_text().splitlines():
                                    fields = line.split()
                                    if "->" in fields: continue
                                    if len(fields) >= 6 and fields[1:3] == ["FLOCK", "ADVISORY"]:
                                        device = fields[5].split(":")
                                        require(len(device) != 3 or (int(device[0], 16), int(device[1], 16), int(device[2])) != key,
                                                "selected_setup_physical_lock_still_held")
                        require(journal.is_dir() and not journal.is_symlink(), "selected_setup_remote_journal_unbound")
                        require(all(directory.is_dir() and not directory.is_symlink()
                                    and directory.name in {row["session_id"] for row in sessions}
                                    for directory in journal.iterdir()), "selected_setup_remote_journal_scope_unknown")
                        pending = list(journal.rglob("*.remote-lease.json"))
                        scope = RemoteProcessLeaseScope(run.name, session_id)
                        for entry in pending:
                            operation = read(entry)
                            prefix = operation.get("control_argv_prefix")
                            require(entry.parent == journal / session_id
                                    and operation.get("schema") == "onnx-splitpoint/remote-process-lease-journal"
                                    and operation.get("schema_version") == 1
                                    and operation.get("run_sha256") == scope.run_sha256
                                    and operation.get("session_sha256") == scope.session_sha256
                                    and operation.get("operation_id") == entry.name[:-len(".remote-lease.json")]
                                    and isinstance(operation.get("token"), str) and bool(operation["token"])
                                    and isinstance(prefix, list) and all(isinstance(arg, str) for arg in prefix)
                                    and prefix.count("-p") == 1 and prefix.index("-p") + 1 < len(prefix),
                                    "selected_setup_remote_journal_scope_unknown")
                            matched = [other for other in setup_ids if
                                {"user_host": (targets_by_id[other].get("remote") or targets_by_id[other].get("runtime"))["user"]
                                    + "@" + (targets_by_id[other].get("remote") or targets_by_id[other].get("runtime"))["host"],
                                 "port": str(int((targets_by_id[other].get("remote") or targets_by_id[other].get("runtime")).get("port", 22)))}
                                == {"user_host": prefix[-1], "port": prefix[prefix.index("-p") + 1]}]
                            require(matched and all(not (resource_keys(targets_by_id[other]) & selected) for other in matched),
                                    "selected_setup_remote_operations_unresolved")
                        require(set(journal.rglob("*.remote-lease.json")) == set(pending),
                                "owner_source_changed_during_observation")
                value = {"pid": pid, "start_ticks": owner["start_ticks"], "run_id": run.name,
                         "run_dir": str(run), "session_id": session_id,
                         "resources": sorted(resources), "central_only": central_only,
                         "completed_selected_setups": sorted(completed_setups),
                         "ssh_targets": [{"user_host": (targets_by_id[setup].get("remote")
                             or targets_by_id[setup].get("runtime"))["user"] + "@" +
                             (targets_by_id[setup].get("remote") or targets_by_id[setup].get("runtime"))["host"],
                             "port": str(int((targets_by_id[setup].get("remote")
                                 or targets_by_id[setup].get("runtime")).get("port", 22)))} for setup in setup_ids]}
                authorities[pid] = value
                return value
            pid = owner["ppid"]
        raise ValueError("observed_process_has_no_live_normal_run_owner")

    try:
        require(isinstance(observed.get("processes"), list)
                and isinstance(observed.get("live_product_leases"), list)
                and observed.get("read_only") is True
                and observed.get("recorded_owners") == [] and observed.get("visibility_errors") == [],
                "controller_visibility_or_owner_scope_unknown")
        require((observed.get("ok") is True and observed.get("status") == "no_known_competing_work_seen"
                 and not observed["processes"] and not observed["live_product_leases"])
                or (observed.get("ok") is False and observed.get("status") == "busy"
                    and bool(observed["processes"] or observed["live_product_leases"])),
                "controller_observation_status_unknown")
        for item in observed["processes"] + observed["live_product_leases"]:
            current = process(item.get("pid"))
            require(current["start_ticks"] == str(item.get("start_time_ticks")), "observed_process_identity_changed")
            bound = authority(current["pid"])
            if "path" in item:
                lease = read(Path(item["path"]))
                scope = RemoteProcessLeaseScope(bound["run_id"], bound["session_id"])
                require(lease.get("schema") == "onnx-splitpoint/remote-process-lease"
                        and lease.get("pid") == current["pid"] and lease.get("pgid") == current["pgid"]
                        and str(lease.get("start_time_ticks")) == current["start_ticks"]
                        and lease.get("run_sha256") == scope.run_sha256
                        and lease.get("session_sha256") == scope.session_sha256
                        and not bound["central_only"], "local_lease_scope_unbound")
                operation = lease.get("operation_id")
                require(isinstance(operation, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", operation),
                        "local_lease_operation_unbound")
                journal = read(Path(bound["run_dir"]) / "jobs/remote_process_leases"
                               / bound["session_id"] / (operation + ".remote-lease.json"))
                prefix = journal.get("control_argv_prefix") or []
                require(journal.get("schema") == "onnx-splitpoint/remote-process-lease-journal"
                        and all(journal.get(key) == lease.get(key) for key in
                                ("operation_id", "token", "run_sha256", "session_sha256"))
                        and isinstance(prefix, list) and all(isinstance(arg, str) for arg in prefix)
                        and prefix and prefix.count("-p") == 1 and prefix.index("-p") + 1 < len(prefix)
                        and {"user_host": prefix[-1], "port": prefix[prefix.index("-p") + 1]}
                        in bound["ssh_targets"], "local_lease_run_session_or_target_unbound")
        require(all(path.read_bytes() == data and not path.is_symlink() for path, data in snapshots.items()),
                "owner_source_changed_during_observation")
        for pid in list(identities):
            process(pid)
    except (OSError, ValueError, TypeError, KeyError, IndexError, StopIteration, AttributeError) as exc:
        errors.append(str(exc) or type(exc).__name__)
    return {"ok": not errors, "read_only": True,
            "status": "resource_scope_unresolved" if errors else "no_conflict_with_selected_resources",
            "selected_resources": sorted(selected), "raw_observation": observed,
            "normal_quality_owners": list(authorities.values()), "scope_errors": errors,
            "cleanup_proven": False, "exclusive_start_permission": False}


def observe_owners(targets: dict[str, dict], output: Path, resource_scoped: bool = False) -> dict:
    """Read-only existing owner probe, without recovery or stopping a process."""
    controller = _observe_processes([])
    if resource_scoped:
        controller = _scoped_controller_owners(targets, controller)
    result = {"observed_at": now(), "controller": controller, "setups": {}, "ok": False}
    if not controller.get("ok"):
        write_json(output / "owners.json", result)
        raise RuntimeError("controller has competing work or incomplete visibility")
    code = inspect.getsource(_observe_processes)
    command = ("python3 -B - <<'GENERIC_COMPLETION_OWNER_PROBE'\n" + code
               + "\nimport json\nprint('OWNERS=' + json.dumps(_observe_processes([])))\n"
               + "GENERIC_COMPLETION_OWNER_PROBE")
    for setup_id, target in targets.items():
        transport = transport_for(target, output / setup_id)
        rc, stdout = transport.run_read_only(command, timeout=40,
                                             env={"PYTHONDONTWRITEBYTECODE": "1"})
        records = [line[7:] for line in stdout.splitlines() if line.startswith("OWNERS=")]
        observation = json.loads(records[0]) if rc == 0 and len(records) == 1 else {
            "ok": False, "error": "owner observation failed", "rc": rc, "output": stdout[-4000:]}
        if resource_scoped and (not isinstance(observation, dict)
                or observation.get("read_only") is not True
                or observation.get("status") != "no_known_competing_work_seen"
                or any(observation.get(key) != [] for key in
                       ("processes", "live_product_leases", "recorded_owners", "visibility_errors"))):
            observation = {"ok": False, "error": "device owner visibility or ownership is unresolved",
                           "raw_observation": observation}
        result["setups"][setup_id] = observation
        write_json(output / "owners.json", result)
        if not observation.get("ok"):
            raise RuntimeError("device has competing work or incomplete visibility: " + setup_id)
    result["ok"] = True
    write_json(output / "owners.json", result)
    return result


def capture_configuration(transport: SSHTransport, target: dict, *, primary: Path, output: Path, phase: str):
    """Run the existing non-mutating host telemetry collector over normal SSH."""
    from .. import resources_utils
    code = resources_utils.read_text("resources", "remote_scripts", "native_host_telemetry.py")
    namespace = "__generic_configuration_probe__"
    args = {"phase": phase, "backend": str(target.get("accelerator") or ""),
            "setup_id": target["id"], "run_id": primary.name, "capture_group": "generic-completed-task"}
    script = ("import json\nns={'__name__':" + repr(namespace) + "}\nexec(" + repr(code)
              + ", ns)\nprint('CONFIGURATION=' + json.dumps(ns['capture'](**" + repr(args) + ")))\n")
    rc, stdout = transport.run_read_only("python3 -B - <<'GENERIC_CONFIGURATION'\n" + script
                                         + "GENERIC_CONFIGURATION", timeout=60)
    matches = [line[len("CONFIGURATION="):] for line in stdout.splitlines() if line.startswith("CONFIGURATION=")]
    if rc != 0 or len(matches) != 1:
        raise RuntimeError("read-only configuration snapshot failed: " + target["id"])
    payload = json.loads(matches[0])
    write_json(output, payload)
    return payload


def verify_runtime_environment(transport: SSHTransport, row: dict, output: Path):
    if row.get("functional_reference_source") is not None:
        from ..generic_completion_plan import apply_native_functional_reference
        row = apply_native_functional_reference(row)
    python = row["remote_python_executable"]
    expected = (row["generic_runtime_environment"]["python_sha256"]
                if row.get("admission_mode") == "generic_only" else
                row["native_command_contract"]["artifacts"]["python_executable"]["sha256"])
    modules = ["numpy", "onnx", "onnxruntime", "tensorrt", "hailo_platform", "dx_engine"]
    script = ("import sys,json,os,hashlib,importlib.util,importlib.metadata,site\nfrom pathlib import Path\n"
              "for p in os.environ.get('SPLITPOINT_EXTRA_SITES','').split(os.pathsep):\n"
              " if p and os.path.isdir(p):site.addsitedir(p)\n"
              "exe=Path(sys.executable).resolve()\n"
              "result={'executable':sys.executable,'resolved_executable':str(exe),"
              "'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'version':sys.version,"
              "'modules':{},'thread_environment':{k:os.environ.get(k) for k in "
              "['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES']}}\n"
              "for name in " + repr(modules) + ":\n"
              " spec=importlib.util.find_spec(name)\n result['modules'][name]={'available':spec is not None,'origin':spec.origin if spec else None}\n"
              " try:result['modules'][name]['version']=importlib.metadata.version(name)\n"
              " except importlib.metadata.PackageNotFoundError:pass\n"
              "print('RUNTIME_ENVIRONMENT='+json.dumps(result))\n")
    rc, stdout = transport.run_read_only(shlex.quote(python) + " -B - <<'GENERIC_RUNTIME_ENVIRONMENT'\n"
                                         + script + "GENERIC_RUNTIME_ENVIRONMENT", timeout=40,
                                         env=row.get("remote_environment", {}))
    records = [line.split("=", 1)[1] for line in stdout.splitlines() if line.startswith("RUNTIME_ENVIRONMENT=")]
    if rc or len(records) != 1:
        raise RuntimeError("bound interpreter dependency inspection failed: " + stdout[-2000:])
    result = json.loads(records[0])
    required = ["numpy", "onnx", "onnxruntime", "tensorrt",
                "dx_engine" if row["setup_id"] == "orin_nx_deepx_m1_01" else "hailo_platform"]
    result["required_modules"] = required
    result["matches_original_interpreter"] = result["executable_sha256"] == expected
    result["missing_modules"] = [name for name in required if not result["modules"][name]["available"]]
    write_json(output, result)
    if not result["matches_original_interpreter"] or result["missing_modules"]:
        raise RuntimeError("bound runtime environment unavailable or changed: " + row["setup_id"])
    return result


def checked_remote(command: str, transport: SSHTransport, *, timeout=120):
    rc, output = transport.run(command, timeout=timeout,
                               env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"})
    if rc:
        raise RuntimeError(f"remote command failed ({rc}): {output[-6000:]}")
    return output


def prepare_runner_bundle(destination: Path) -> Path:
    """Use the product writer in a new session directory, never refresh a run."""
    from ..split_export_runners import write_runner_skeleton_onnxruntime
    from ..gui.controller import _copy_runner_lib
    destination.mkdir(parents=True, exist_ok=False)
    _copy_runner_lib(destination)
    write_runner_skeleton_onnxruntime(str(destination / "runner"))
    bundle = destination.with_suffix(".tar.gz")
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(destination / "splitpoint_runners", arcname="splitpoint_runners",
                    filter=lambda info: None if "__pycache__" in info.name or info.name.endswith(".pyc") else info)
        archive.add(destination / "runner" / "run_split_onnxruntime.py", arcname="runner/run_split_onnxruntime.py")
    return bundle


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def upload_new_file(transport: SSHTransport, local: Path, remote: str, *, expected_sha=""):
    """Copy into this attempt only; existing unequal destination bytes block."""
    actual = file_sha256(local)
    if expected_sha and actual != expected_sha:
        raise ValueError("staging source differs from its original binding: " + str(local))
    probe = ("from pathlib import Path\nimport hashlib,json\np=Path(" + repr(remote) + ")\n"
             "h=hashlib.sha256()\n"
             "if p.is_file():\n"
             " with p.open('rb') as f:\n"
             "  for b in iter(lambda:f.read(8388608),b''):h.update(b)\n"
             "print(json.dumps({'exists':p.exists(),'sha256':h.hexdigest() if p.is_file() else ''}))\n")
    command = "python3 -B - <<'COMPLETION_FILE_PROBE'\n" + probe + "COMPLETION_FILE_PROBE"
    rc, stdout = transport.run_read_only(command, timeout=120)
    if rc:
        raise RuntimeError("staging destination probe failed: " + stdout[-2000:])
    observation = json.loads(stdout)
    if observation["exists"]:
        if observation["sha256"] != actual:
            raise ValueError("existing staging destination differs: " + remote)
        return {"path": remote, "sha256": actual, "status": "reused_exact_bytes"}
    parent = str(PurePosixPath(remote).parent)
    temporary = remote + ".upload-" + uuid.uuid4().hex
    checked_remote("mkdir -p -- " + shlex.quote(parent), transport)
    rc, stdout = transport.scp_upload(str(local), temporary, timeout=600)
    if rc:
        raise RuntimeError("staging transfer failed: " + stdout[-2000:])
    commit = ("from pathlib import Path\nimport hashlib,os\np=Path(" + repr(temporary) + ")\n"
              "h=hashlib.sha256()\nwith p.open('rb') as f:\n"
              " for b in iter(lambda:f.read(8388608),b''):h.update(b)\n"
              "assert h.hexdigest()==" + repr(actual) + ", 'transferred artifact hash mismatch'\n"
              "os.link(p," + repr(remote) + ")\np.unlink()\n")
    checked_remote("python3 -B - <<'COMPLETION_FILE_COMMIT'\n" + commit + "COMPLETION_FILE_COMMIT", transport)
    return {"path": remote, "sha256": actual, "status": "uploaded_exact_bytes"}


def _acceptance_repetitions(scope: dict) -> int:
    """Reduce controller dispatch only for an explicit functional acceptance."""
    from ..generic_energy_reporting import _acceptance_repetitions as repetitions
    return repetitions(scope)


def _is_hailo10_stage1(config: dict) -> bool:
    original = config.get("original_run_cfg") or {}
    provider = str(original.get("stage1_provider") or original.get("provider") or "").strip().lower()
    return provider in {"hailo10", "hailo10h"}


def _is_deepx_stage1(config: dict) -> bool:
    original = config.get("original_run_cfg") or {}
    provider = str(original.get("stage1_provider") or original.get("provider") or "").strip().lower()
    return provider == "deepx_m1"


def _requested_stage1_policies(options: dict) -> dict:
    return {key: normalize_stage1_execution({"policy": policy, "inflight": options[key]})
            for key, policy in (("hailo_async_inflight", "hailo_async_slots_v1"),
                                ("deepx_async_inflight", "deepx_async_slots_v1"))
            if options.get(key) is not None}


def _stage1_request_for_row(requested: dict, config: dict) -> dict | None:
    if _is_hailo10_stage1(config):
        return requested.get("hailo_async_inflight")
    if _is_deepx_stage1(config):
        return requested.get("deepx_async_inflight")
    return None


def _validated_stage1_execution(config: dict) -> dict:
    execution = normalize_stage1_execution(config.get("stage1_execution"))
    if execution["policy"] == "hailo_async_slots_v1" and not _is_hailo10_stage1(config):
        raise ValueError("Generic async stage1 execution requires the bound Hailo10H provider")
    if execution["policy"] == "deepx_async_slots_v1" and not _is_deepx_stage1(config):
        raise ValueError("Generic async stage1 execution requires the bound DeepX M1 provider")
    return execution


def validate_plan(plan: dict, scope_csv: Path, primary: Path) -> list[dict]:
    import csv
    from ..generic_completion_plan import completion_identity
    if plan.get("scope_mode") == "generic_only_completed_task":
        from ..generic_only_plan import validate_plan as validate_generic_only_plan
        return validate_generic_only_plan(plan, scope_csv, primary)
    if plan.get("admission_mode") == "generic_only":
        raise ValueError("Generic-only admission requires its explicitly separated plan mode")
    with Path(scope_csv).open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    rows = plan.get("rows")
    selective = plan.get("scope_mode") == "audited_selective_remeasurement"
    expected_count = plan.get("authorized_row_count") if selective else 192
    if type(expected_count) is not int or not 1 <= expected_count <= 246:
        raise ValueError("selective completion requires an explicit bounded authorized row count")
    if not isinstance(rows, list) or len(rows) != expected_count or len(source) != expected_count:
        raise ValueError(f"authorized Generic completion scope requires exactly {expected_count} rows")
    if selective:
        if file_sha256(scope_csv) != plan.get("scope_source_sha256"):
            raise ValueError("selective completion scope CSV hash mismatch")
        provenance = plan.get("source_execution_plans")
        if not isinstance(provenance, list) or not provenance:
            raise ValueError("selective completion needs preserved source plan provenance")
        permitted = set()
        historical_roots = {}
        for descriptor in provenance:
            path = Path(descriptor["path"])
            if file_sha256(path) != descriptor.get("sha256"):
                raise ValueError("selective completion source plan hash mismatch")
            historical = read_json(path)
            permitted.update(completion_identity(r.get("identity", r)) for r in historical["rows"])
            historical_roots.update({completion_identity(r.get("identity", r)): str(r.get("original_run") or historical.get("original_run") or "")
                                     for r in historical["rows"]})
        if not {completion_identity(r.get("identity", r)) for r in rows} <= permitted:
            raise ValueError("selective completion contains an identity absent from audited source plans")
        run_bindings = {}
        for binding in plan.get("source_run_bindings", []):
            original = str(binding["historical_root"])
            archive = Path(binding["archive_root"]).resolve()
            if not original or original in run_bindings:
                raise ValueError("selective completion source run mapping duplicated or missing")
            if file_sha256(archive / "run_manifest.json") != binding.get("run_manifest_sha256"):
                raise ValueError("selective completion archive run manifest hash mismatch")
            run_bindings[original] = archive
        for row in rows:
            identity = completion_identity(row.get("identity", row))
            historical_root = historical_roots[identity]
            if run_bindings:
                if (row.get("historical_original_run") != historical_root
                        or run_bindings.get(historical_root) != Path(row["original_run"]).resolve()):
                    raise ValueError("selective completion row differs from explicit historical/archive run mapping")
            elif Path(row["original_run"]).resolve() != primary.resolve():
                raise ValueError("selective completion mixed original runs require explicit archive mappings")
    def key(row):
        identity = row.get("identity", row)
        return completion_identity(identity)
    expected, actual = [key(r) for r in source], [key(r) for r in rows]
    if len(set(expected)) != expected_count or len(set(actual)) != expected_count or set(expected) != set(actual):
        raise ValueError(f"completion plan differs from the authorized {expected_count} identities")
    requested_repetitions = _acceptance_repetitions(plan)
    stage1_requests = _requested_stage1_policies(plan)
    for row in rows:
        if _acceptance_repetitions(row) != requested_repetitions:
            raise ValueError('Generic acceptance row differs from its controller recipe')
        config = row.get("runtime_config", row)
        execution = _validated_stage1_execution(config)
        if stage1_requests:
            expected = _stage1_request_for_row(stage1_requests, config) or normalize_stage1_execution()
            if execution != expected:
                raise ValueError("Generic stage1 execution differs from the prepared policy")
        if key(config) != key(row) or (row.get("identity") and key(row) != completion_identity(row)):
            raise ValueError("completion runtime identity differs from its authorized row")
        if any(config.get(k) != v for k, v in (("repetitions", 3), ("warmup_count", 100), ("task_count", 1000))):
            raise ValueError("completion plan work budget changed")
        expected_original = Path(row["original_run"]).resolve() if selective else primary.resolve()
        if Path(config.get("original_run", "")).resolve() != expected_original:
            raise ValueError("completion plan original run binding changed")
        relative = PurePosixPath(row["case_output_relative"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) != "/".join(
                str(row[k]) for k in ("setup_id", "model_id", "case_id")):
            raise ValueError("completion case output path differs from its identity")
        if plan.get("output_root"):
            root = outside_primary(Path(plan["output_root"]), primary)
            expected_result = (root / "results" / str(relative) / "generic_completion_results.json").resolve()
            if Path(row["result_path"]).resolve() != expected_result or root not in expected_result.parents:
                raise ValueError("completion result path escapes its external case output")
        if "runtime_config" in row:
            if config.get("execution_id") != plan.get("execution_id"):
                raise ValueError("completion execution identity changed")
            remote_root = PurePosixPath(row["remote_root"])
            remote_paths = [item["remote"] for item in row["staging_files"]]
            remote_paths.extend(p for p in (config.get("case_dir"), config.get("original_run_cfg", {}).get(
                "native_split_quality_binding")) if p)
            if not remote_root.is_absolute() or any(".." in PurePosixPath(p).parts or
                    remote_root not in PurePosixPath(p).parents for p in remote_paths):
                raise ValueError("completion staging path escapes its session")
    return rows


def _bind_prepared_numeric_domain(row: dict, command: dict, prepared: dict, runtime_input_format: str) -> None:
    """Describe an exact historical RGB feed without changing its bytes."""
    binding = row['input_binding']
    contract = binding['prepared_input_contract']
    declarations = [prepared, contract, contract.get('source_input_contract') or {}, *contract.get('entries', [])]
    for key in ('normalization', 'source_domain'):
        values = {item[key] for item in declarations if item.get(key)}
        if len(values) > 1:
            raise ValueError('Generic prepared input numeric contract conflict: ' + key)
    if any(item.get(key) for item in declarations for key in ('normalization', 'source_domain')):
        return
    entries = binding.get('entries') or []
    artifact = (command.get('artifacts') or {}).get('prepared_input') or {}
    shape = prepared.get('shape') or []
    if (row.get('task') != 'detection' or command.get('backend') != 'hailo8_to_trt'
            or runtime_input_format != 'uint8' or len(entries) != 1
            or contract != command.get('prepared_input_contract')
            or contract.get('format') != 'raw_runtime_tensor'
            or contract.get('preprocess') != 'opencv_letterbox_rgb_uint8'
            or contract.get('layout') != 'HWC' or contract.get('dtype') != 'uint8'
            or len(shape) != 3 or shape[-1] != 3 or contract.get('shape') != shape
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or prepared.get('dtype') != 'uint8' or prepared.get('format') != 'raw'
            or prepared.get('path') != artifact.get('path')
            or prepared.get('sha256') != artifact.get('sha256')
            or not re.fullmatch('[a-f0-9]{64}', str(prepared.get('sha256') or ''))
            or contract.get('shared_rgb_source_sha256') != prepared['sha256']
            or not contract.get('shared_rgb_source')
            or contract.get('source_image_sha256') != command.get('input_image_sha256')
            or binding.get('source_image_sha256') != command.get('input_image_sha256')
            or command.get('contract_sha256') != row.get('native_command_contract_sha256')):
        raise ValueError('Generic prepared input lacks an explicitly bound numeric domain/normalization')
    local = Path(entries[0].get('local_path') or '')
    if (not local.is_file() or local.stat().st_size != shape[0] * shape[1] * shape[2]
            or file_sha256(local) != prepared['sha256']):
        raise ValueError('Generic legacy RGB prepared input bytes differ from the bound source')
    prepared['source_domain'] = 'uint8_0_255'
    prepared['source_domain_binding'] = {
        'schema': 'onnx-splitpoint/generic-prepared-input-domain-binding/v1',
        'source_domain': 'uint8_0_255', 'numeric_transform_applied': False,
        'derivation': 'verified_hailo8_opencv_letterbox_rgb_uint8_shared_source_exact_bytes',
        'native_source': {'path': row['native_source'], 'sha256': file_sha256(Path(row['native_source']))},
        'native_command_contract_sha256': command['contract_sha256'],
        'native_quality_binding_sha256': row['native_quality_binding_sha256'],
        'source_contract_field': 'native_command_contract.prepared_input_contract',
        'preprocess': contract['preprocess'], 'prepared_input_sha256': prepared['sha256'],
        'shared_rgb_source': {'path': contract['shared_rgb_source'], 'sha256': contract['shared_rgb_source_sha256']}}


def runtime_configuration(row: dict, remote_root: str, execution_id: str,
                          *, stage1_execution: dict | None = None) -> tuple[dict, list[dict]]:
    """Project original bound files into a new attempt, with unchanged content."""
    if row.get("admission_mode") == "generic_only":
        from ..generic_only_plan import runtime_configuration as generic_only_configuration
        return generic_only_configuration(row, remote_root, execution_id, stage1_execution=stage1_execution)
    if row.get("functional_reference_source") is not None:
        from ..generic_completion_plan import apply_native_functional_reference
        row = apply_native_functional_reference(row)
    import posixpath
    execution = _validated_stage1_execution({"original_run_cfg": row["original_run_cfg"],
                                             "stage1_execution": stage1_execution})
    local_case = Path(row["original_local_case_dir"])
    case_dir = f"{remote_root}/cases/{row['model_id']}/{row['case_id']}"
    config_dir = f"{remote_root}/configs/{row['model_id']}/{row['case_id']}"
    manifest_path = local_case / "split_manifest.json"
    manifest = read_json(manifest_path)
    staged = [{"local": str(manifest_path), "remote": case_dir + "/split_manifest.json",
               "sha256": file_sha256(manifest_path), "role": "original_manifest"}]
    for role in ("part1", "part2"):
        relative = str(manifest.get(role) or (manifest.get("models", {}).get(role) or {}).get("path") or "")
        if not relative or PurePosixPath(relative).is_absolute():
            raise ValueError("unsupported original model path: " + role)
        remote = posixpath.normpath(case_dir + "/" + relative)
        if not remote.startswith(remote_root + "/"):
            raise ValueError("original model path escapes completion workspace")
        local = (local_case / relative).resolve()
        if not local.is_file():
            raise FileNotFoundError("original Generic model metadata missing: " + str(local))
        sha = file_sha256(local)
        expected = row["expected_runtime_artifacts"].get("source_part2_onnx", {}) if role == "part2" else {}
        if expected.get("sha256") and sha != expected["sha256"]:
            raise ValueError("original Generic Part2 source binding differs")
        staged.append({"local": str(local), "remote": remote, "sha256": sha,
                       "role": role + "_onnx_metadata",
                       "existing_remote": expected.get("path")})
    original_native = read_json(Path(row["native_source"]))
    from ..native_command_contract import verify_native_command_contract
    command, command_status = verify_native_command_contract(original_native["native_command_contract"])
    if command is None or command["contract_sha256"] != row["native_command_contract_sha256"]:
        raise ValueError("original Native command changed since planning")
    binding = command["native_split_quality_binding"]
    if binding["binding_sha256"] != row["native_quality_binding_sha256"]:
        raise ValueError("original Generic/Native split artifact binding differs")
    prepared = row["input_binding"]["entries"][0]
    input_spec = {"path": prepared["remote_path"], "sha256": prepared["sha256"],
                  "shape": prepared["shape"], "dtype": prepared["dtype"],
                  "input_name": (manifest.get("orig_inputs") or [prepared["name"]])[0],
                  "format": "raw" if str(prepared["remote_path"]).endswith(".bin") else "npy"}
    reference_path = Path(row["functional_reference"]["output_manifest"])
    expected_manifest = command["artifacts"]["semantic_output_manifest"]
    if file_sha256(reference_path) != expected_manifest["sha256"]:
        raise ValueError("original functional output manifest changed")
    reference = read_json(reference_path)
    # H8's original C++ output writer records the input path and sequence; its
    # manifest SHA and input-image SHA are jointly sealed by the original
    # Native command. Other writers also include the image SHA in the manifest.
    reference_image = reference.get("input_image") or (reference.get("provenance") or {}).get("image")
    reference_image_sha = reference.get("input_image_sha256") or (reference.get("provenance") or {}).get("image_sha256")
    if (reference_image != command["input_image"] or
            command["input_image_sha256"] != row["input_binding"]["source_image_sha256"] or
            (reference_image_sha and reference_image_sha != command["input_image_sha256"])):
        raise ValueError("functional output reference uses a different original input")
    outputs = {}
    for descriptor in reference["outputs"]:
        local = (reference_path.parent / descriptor["file"]).resolve()
        if local.parent != reference_path.parent.resolve():
            raise ValueError("functional output tensor escapes original output directory")
        if file_sha256(local) != descriptor["sha256"]:
            raise ValueError("functional output tensor changed")
        remote = config_dir + "/reference/" + local.name
        staged.append({"local": str(local), "remote": remote, "sha256": descriptor["sha256"], "role": "functional_reference"})
        outputs[descriptor["name"]] = {"path": remote, "sha256": descriptor["sha256"],
            "shape": descriptor["shape"], "dtype": descriptor["dtype"], "format": "raw"}
    original_cfg = dict(row["original_run_cfg"])
    original_cfg["manifest"] = case_dir + "/split_manifest.json"
    original_cfg["native_split_quality_binding"] = config_dir + "/native_split_quality_binding.json"
    policy = json.loads(row["original_run_cfg"]["quality_gate_json"])
    limits = policy["native_contract"]["detection_self_reference"]
    runtime_options = command["runtime_options"]
    if command["backend"] == "hailo8_to_trt":
        runtime_input_format = runtime_options.get("hailo_format")
    elif command["backend"] == "hailo10h_to_trt":
        if type(runtime_options.get("quantized_inputs")) is not bool:
            raise ValueError("original Hailo10 input quantization binding missing")
        runtime_input_format = "uint8" if runtime_options["quantized_inputs"] else "float32"
    else:
        runtime_input_format = input_spec["dtype"]
    if runtime_input_format not in {"uint8", "float32"}:
        raise ValueError("original accelerator runtime input format missing")
    _bind_prepared_numeric_domain(row, command, input_spec, runtime_input_format)
    completion = row["completion_contract"] if row["task"] == "detection" else {
        "task": "classification", "operation": "top1_top5", "label_offset": 0}
    config = {"identity": row["identity"], "execution_id": execution_id, "original_run": row["original_run"],
              "task": row["task"], "case_dir": case_dir, "original_run_cfg": original_cfg,
              "repetitions": 3, "warmup_count": 100, "task_count": 1000,
              "queue_depth": int(original_cfg["throughput_queue_depth"]),
              "timeout_s": 3600, "measurement_boundary": "prepared_input_before_p1_to_last_completed_task",
              "input_binding": {**row["input_binding"], "prepared_input": input_spec,
                  "runtime_input_format": runtime_input_format,
                  "runtime_input_format_source": row["native_source"] + "#native_command_contract.runtime_options"},
              "completion_contract": completion,
              "expected_runtime_artifacts": {k: {"path": row["expected_runtime_artifacts"][k]["path"],
                   "sha256": row["expected_runtime_artifacts"][k]["sha256"]} for k in ("part1_runtime", "engine")},
              "functional_reference": {"input_sha256": input_spec["sha256"], "outputs": outputs,
                  "source_manifest": str(reference_path), "source_manifest_sha256": expected_manifest["sha256"],
                  "tolerances": {"match_iou_threshold": limits["iou_threshold"],
                                 "min_match_ratio": limits["min_reference_match_ratio"],
                                 "min_mean_iou": limits["min_mean_matched_iou"]}}}
    config["measurement_contract"] = row["measurement_boundary"]
    # Absence keeps historical synchronous configs and their existing energy
    # operation hashes unchanged. New scheduling experiments opt in explicitly.
    if stage1_execution is not None:
        config["stage1_execution"] = execution
    return config, [*staged, {"role": "original_split_binding", "payload": binding,
                             "remote": original_cfg["native_split_quality_binding"]}]


def stage_case(row: dict, config: dict, files: list[dict], transport: SSHTransport, local: Path) -> str:
    local.mkdir(parents=True, exist_ok=True)
    receipts = []
    for item in files:
        if "payload" in item:
            path = local / "native_split_quality_binding.json"
            write_json(path, item["payload"])
        else:
            path = Path(item["local"])
        existing = item.get("existing_remote")
        if existing:
            # Source Part2 is already attested in the immutable Native cache.
            # A session-local symlink avoids another model transfer.
            script = ("from pathlib import Path\nimport hashlib,os\nsrc=Path(" + repr(existing) + ")\n"
                      "dst=Path(" + repr(item["remote"]) + ")\nh=hashlib.sha256()\nwith src.open('rb') as f:\n"
                      " for b in iter(lambda:f.read(8388608),b''):h.update(b)\n"
                      "assert h.hexdigest()==" + repr(item["sha256"]) + ", 'existing source Part2 differs'\n"
                      "dst.parent.mkdir(parents=True,exist_ok=True)\n"
                      "if dst.is_symlink():\n assert dst.resolve()==src.resolve(), 'session source alias differs'\n"
                      "else:\n assert not dst.exists(), 'session model path already occupied'\n dst.symlink_to(src)\n")
            checked_remote("python3 -B - <<'COMPLETION_SOURCE_ALIAS'\n" + script + "COMPLETION_SOURCE_ALIAS", transport)
            receipts.append({"role": item["role"], "path": item["remote"], "source": existing, "sha256": item["sha256"], "status": "session_alias_to_exact_source"})
        else:
            receipts.append({"role": item["role"], **upload_new_file(transport, path, item["remote"], expected_sha=item.get("sha256", ""))})
    config_path = local / "generic_completion_config.json"
    write_json(config_path, config)
    config_directory = (config["config_directory"] if config.get("admission_mode") == "generic_only" else
                        str(PurePosixPath(config["original_run_cfg"]["native_split_quality_binding"]).parent))
    remote_config = str(PurePosixPath(config_directory) / config_path.name)
    receipts.append(upload_new_file(transport, config_path, remote_config))
    write_json(local / "staging_receipts.json", {"files": receipts, "completed_at": now()})
    return remote_config


def prepare_execution(scope_plan: Path, output: Path, *, hailo_async_inflight: int | None = None,
                      deepx_async_inflight: int | None = None) -> Path:
    from ..generic_only_plan import validation_scope
    with validation_scope():
        return _prepare_execution(scope_plan, output, hailo_async_inflight=hailo_async_inflight,
                                  deepx_async_inflight=deepx_async_inflight)


def _prepare_execution(scope_plan: Path, output: Path, *, hailo_async_inflight: int | None = None,
                       deepx_async_inflight: int | None = None) -> Path:
    plan = read_json(scope_plan)
    primary = Path(plan["original_run"]).resolve()
    output = outside_primary(output, primary)
    rows = validate_plan(plan, Path(plan["scope_source"]), primary)
    stage1_requests = _requested_stage1_policies({"hailo_async_inflight": hailo_async_inflight,
                                                 "deepx_async_inflight": deepx_async_inflight})
    for key, eligible, label in (("hailo_async_inflight", _is_hailo10_stage1, "Hailo10H"),
                                ("deepx_async_inflight", _is_deepx_stage1, "DeepX M1")):
        if key in stage1_requests and not any(eligible(row) for row in rows):
            raise ValueError(f"--{key.replace('_', '-')} requires at least one bound {label} row")
    execution_path = output / "plan" / "execution-plan.json"
    if execution_path.exists():
        raise FileExistsError("execution plan already exists; use its bound entry: " + str(execution_path))
    execution_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
    targets = bound_targets(primary, rows, **({"producer_run_binding": plan["producer_run_binding"]}
        if plan.get("producer_run_binding") is not None else {}))
    for row in rows:
        target = targets[row["setup_id"]]
        runtime = original_generic_environment(row)
        row.update(remote_python_executable=runtime["python"], remote_environment=runtime["environment"],
                   generic_environment_source=runtime["source"], generic_environment_original_line=runtime["original_line"])
        remote_root = f"/home/{target['remote']['user']}/splitpoint_runs/generic_completed_task/{execution_id}"
        execution = _stage1_request_for_row(stage1_requests, row)
        if execution is not None:
            config, files = runtime_configuration(row, remote_root, execution_id,
                                                  stage1_execution=execution)
        else:
            config, files = runtime_configuration(row, remote_root, execution_id)
        row.update(runtime_config=config, staging_files=files, remote_root=remote_root,
                   result_path=str(output / "results" / row["case_output_relative"] / "generic_completion_results.json"))
    plan.update(execution_id=execution_id, prepared_at=now(), output_root=str(output),
                scope_plan_source=str(scope_plan.resolve()), hardware_targets=targets)
    if hailo_async_inflight is not None:
        plan["hailo_async_inflight"] = hailo_async_inflight
    if deepx_async_inflight is not None:
        plan["deepx_async_inflight"] = deepx_async_inflight
    validate_plan(plan, Path(plan["scope_source"]), primary)
    write_json(execution_path, plan)
    _write_completion_reports(plan, output / "results", output / "reports")
    return execution_path


def original_generic_environment(row: dict) -> dict:
    """Replay the original normal dispatch's interpreter and appended SDK sites.

    H8 uses system GPU Python plus its existing Hailo venv's appended sites.
    Merely using the Native C++ controller interpreter loses that SDK binding.
    """
    if row.get("admission_mode") == "generic_only":
        from ..generic_only_plan import environment
        return environment(row)
    source = Path(row["generic_raw_source"]).resolve()
    original_results = (Path(row["original_run"]) / "models" / row["model_id"] / "benchmark_results").resolve()
    if original_results not in source.parents:
        raise ValueError("Generic runtime source is outside its original model results")
    # Successful supplemental rows retain their own dispatch alongside the
    # bound raw result. The canonical stdout can belong to an earlier failure.
    path = source.parent / ("remote_benchmark_stdout_" + row["setup_id"] + ".txt")
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "[remote] python:" not in line:
            continue
        python = re.search(r"\bRUN_PY=([^\s|]+)", line)
        extra = re.search(r"\bEXTRA_SITES=([^\s]*)", line)
        if python and extra:
            matches.append((python.group(1), extra.group(1), line))
    if not matches or len({(p, sites) for p, sites, _ in matches}) != 1:
        raise ValueError("original Generic runtime environment is not uniquely bound: " + str(path))
    python, extra, line = matches[0]
    if not PurePosixPath(python).is_absolute() or any(not PurePosixPath(p).is_absolute() for p in extra.split(":") if p):
        raise ValueError("original Generic runtime environment path is not absolute")
    return {"python": python, "environment": {"SPLITPOINT_EXTRA_SITES": extra},
            "source": str(path), "original_line": line}


def configuration_comparison(row: dict, current: dict, current_path: Path) -> dict:
    """Compare available original snapshots; preserve unknown historical fields."""
    if row.get("admission_mode") == "generic_only":
        from ..generic_only_plan import configuration_comparison as compare_generic_only
        return compare_generic_only(row, current, current_path)
    from ..resources.remote_scripts.native_host_telemetry import _identity
    primary = Path(row["original_run"])
    captures = read_json(primary / "reports" / "native_host_telemetry_summary.json")["captures"]
    relevant = [c for c in captures if (c.get("identity") or {}).get("setup_id") == row["setup_id"]]
    # The result path identifies the exact production attempt. Original cases
    # outside attempts use the canonical backend directory's captures.
    native_source = Path(row["native_source"])
    candidates = [c for c in relevant if Path(c["path"]).parent.parent in native_source.parents]
    if not candidates:
        command = row["native_command_contract"]
        backend = {"deepx_to_trt": "deepx", "hailo10h_to_trt": "hailo10h", "hailo8_to_trt": "hailo8"}[command["backend"]]
        source_root = primary / "native_producers" / backend
        if "attempts" in native_source.parts:
            i = native_source.parts.index("attempts")
            source_root = Path(*native_source.parts[:i + 3])
        candidates = [c for c in relevant if Path(c["path"]).parent.parent == source_root]
    current_identity = _identity(current)
    fields = ("cpu_governors", "cpu_min_max", "device_governors", "device_min_max")
    compared, differences, unknown = {}, [], []
    for key in fields:
        originals = [(c["path"], (c.get("identity") or {}).get(key)) for c in candidates]
        originals = [(path, value) for path, value in originals if value]
        current_value = json.loads(json.dumps(current_identity.get(key)))
        if not originals or not current_value:
            unknown.append(key)
            continue
        matches = all(value == current_value for _, value in originals)
        compared[key] = {"matches": matches, "current": current_value, "historical": originals}
        if not matches:
            differences.append(key)
    # nvpmodel query text contains current state as well as the selected mode.
    def mode(payload):
        return str(((payload.get("nvidia_commands") or {}).get("nvpmodel_mode") or {}).get("stdout") or "").strip()
    old_modes = [(c["path"], mode(read_json(Path(c["path"])))) for c in candidates]
    old_modes = [(p, value) for p, value in old_modes if value]
    if old_modes and mode(current):
        matched = all(value == mode(current) for _, value in old_modes)
        compared["nvpmodel_mode"] = {"matches": matched, "current": mode(current), "historical": old_modes}
        if not matched:
            differences.append("nvpmodel_mode")
    else:
        unknown.append("nvpmodel_mode")
    unknown += ["historical_thread_environment_not_separately_recorded", "thermal_state_is_observed_not_equalized"]
    return {"status": "conflict" if differences else "observed_no_known_configuration_conflict",
            "current_snapshot": str(current_path), "historical_snapshots": [c["path"] for c in candidates],
            "checks": compared, "differences": differences, "unknown_or_not_equalized": unknown,
            "note": "No tuning. Runtime/engine identity is independently checked by original command and artifact receipts."}


def download_case(transport: SSHTransport, remote_output: str, local_output: Path, attempt: int,
                  *, result_filename: str = "generic_completion_results.json") -> dict:
    """Keep all attempt downloads, including evidence after a failed process."""
    from ..generic_completion_reporting import _read_result
    if result_filename not in {"generic_completion_results.json", "generic_energy_results.json"}:
        raise ValueError("unsupported Generic result publication")
    destination = local_output / "downloads" / f"attempt_{attempt:03d}"
    destination.mkdir(parents=True, exist_ok=False)
    rc, stdout = transport.scp_download(remote_output, str(destination), recursive=True, timeout=600)
    record = {"at": now(), "rc": rc, "output": stdout[-4000:], "destination": str(destination)}
    if rc:
        return record
    source_dir = destination / PurePosixPath(remote_output).name
    source = source_dir / result_filename
    if source.is_file():
        payload = _read_result(source)
        # Keep the declared, immutable functional samples next to the imported
        # result so the normal parser verifies actual bytes on every reentry.
        for repetition in payload.get("repetitions", []):
            descriptors = list((repetition.get("raw_output_samples") or {}).values())
            if repetition.get("completed_result_artifact"):
                descriptors.append(repetition["completed_result_artifact"])
            if (repetition.get("mixed_input_control") or {}).get("report"):
                descriptors.append(repetition["mixed_input_control"]["report"])
            serial = repetition.get("serial_reference") or {}
            descriptors.extend((serial.get("raw_output_samples") or {}).values())
            if serial.get("completed_result_artifact"):
                descriptors.append(serial["completed_result_artifact"])
            for descriptor in descriptors:
                relative = Path(str(descriptor.get("path") or ""))
                if not descriptor.get("path") or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("downloaded functional output sample path invalid")
                sample = source_dir / relative
                if not sample.resolve().is_relative_to(source_dir.resolve()):
                    raise ValueError("downloaded functional output sample escapes attempt")
                if not sample.is_file():
                    # Preserve the result to let the ordinary reporter record
                    # the exact missing-output error instead of inventing reuse.
                    continue
                if file_sha256(sample) != descriptor.get("sha256"):
                    raise ValueError("downloaded functional output sample changed")
                target = local_output / relative
                if target.exists():
                    if file_sha256(target) != descriptor["sha256"]:
                        raise ValueError("existing collected functional output sample differs")
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(sample, target)
        # Energy completion markers bind the exact published bytes. Preserve
        # those bytes through transport; a JSON reserialization changes SHA.
        if result_filename == "generic_energy_results.json":
            target = local_output / result_filename
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError("existing Generic energy publication differs")
            if not target.exists():
                shutil.copy2(source, target)
        else:
            write_json(local_output / result_filename, payload)
        record["result_collected"] = True
        record["remote_output_mirror"] = str(source_dir)
    else:
        record["result_collected"] = False
    return record


def run_case(row: dict, transport: SSHTransport, output: Path, index: int, *, total: int = 192,
             runner_root: str | None = None, cell_context: dict | None = None,
             source_guard=None) -> None:
    from ..generic_completion_reporting import load_valid_repetitions, validate_completion_case
    requested_repetitions = _acceptance_repetitions(row)
    requested_indices = set(range(1, requested_repetitions + 1))
    config = row["runtime_config"]
    if source_guard is not None:
        source_guard()
    if cell_context is not None:
        from .generic_only_cell_policy import require_same_case, require_clean_ownership
        require_same_case(row, cell_context["initial_row"])
        require_clean_ownership(cell_context["registry"], cell_context["journal"], cell_context["scope"])
    local_output = Path(row["result_path"]).parent
    local_output.mkdir(parents=True, exist_ok=True)
    valid = load_valid_repetitions(row["result_path"], row)
    if row.get("admission_mode") == "generic_only" and valid:
        from ..generic_only_reporting import validate_completion_attestations
        validate_completion_attestations({"repetitions": valid}, row, expected_repetitions=len(valid))
    if cell_context is not None:
        from .generic_only_cell_policy import read_dispatch_attempts, validate_dispatch_history
        attempts = read_dispatch_attempts(row["result_path"])
        validate_dispatch_history(row, valid, attempts)
    completed = {r["repetition_index"] for r in valid}
    if completed == requested_indices:
        print(f"[completion] {index}/{total} {case_name(row)} reuse={requested_repetitions}/{requested_repetitions}", flush=True)
        return
    if not completed <= requested_indices:
        raise ValueError('Generic acceptance result contains unrequested repetitions')
    remote_output = row["remote_root"] + "/results/" + row["case_output_relative"]
    runtime_root = runner_root or row["remote_root"]
    try:
        remote_config = stage_case(row, config, row["staging_files"], transport,
                                  output / "staging" / row["case_output_relative"])
        checked_remote("mkdir -p -- " + shlex.quote(remote_output), transport)
        # Preserve exact parsed work when a prior remote workspace was lost.
        if completed:
            for repetition in valid:
                descriptors = [*(repetition.get("raw_output_samples") or {}).values(),
                               repetition["completed_result_artifact"]]
                if (repetition.get("mixed_input_control") or {}).get("report"):
                    descriptors.append(repetition["mixed_input_control"]["report"])
                serial = repetition.get("serial_reference") or {}
                descriptors.extend((serial.get("raw_output_samples") or {}).values())
                if serial.get("completed_result_artifact"):
                    descriptors.append(serial["completed_result_artifact"])
                for descriptor in descriptors:
                    relative = descriptor["path"]  # checked by load_valid_repetitions
                    upload_new_file(transport, local_output / relative, remote_output + "/" + relative,
                                    expected_sha=descriptor["sha256"])
            upload_new_file(transport, Path(row["result_path"]), remote_output + "/generic_completion_results.json")
    except Exception as staging_error:
        if cell_context is not None:
            from .generic_only_cell_policy import InvalidResumeBinding
            raise InvalidResumeBinding("Normal staging/source admission failed before measurement reservation") from staging_error
        raise
    attempts_path = local_output / "dispatch_attempts.json"
    if cell_context is None:
        attempts = read_json(attempts_path).get("attempts", []) if attempts_path.exists() else []
    for repetition in sorted(requested_indices):
        if repetition in completed:
            continue
        if source_guard is not None:
            source_guard()
        argv = [row["remote_python_executable"], "-B", "-u",
                runtime_root + "/runner/run_split_onnxruntime.py",
                "--generic-completion-config", remote_config,
                "--generic-completion-repetition", str(repetition), "--out-dir", remote_output]
        command = "cd -- " + shlex.quote(runtime_root) + " && " + shlex.join(argv)
        attempt = {"repetition_index": repetition, "command": command, "started_at": now(), "status": "running"}
        if cell_context is not None:
            from .generic_only_cell_policy import admit_missing_repetition
            attempt["bounded_cell_admission"] = admit_missing_repetition(row,
                cell_context["initial_row"], repetition, valid, attempts,
                control=cell_context["control"], initial_control=cell_context["initial_control"],
                registry=cell_context["registry"], journal=cell_context["journal"],
                expected_scope=cell_context["scope"])
        attempts.append(attempt)
        write_json(attempts_path, {"attempts": attempts})
        print(f"[completion] {index}/{total} {case_name(row)} repetition={repetition}/{requested_repetitions} warmup=100 tasks=1000 START", flush=True)
        logfile = local_output / f"dispatch_{len(attempts):03d}.log"
        try:
            with logfile.open("x", encoding="utf-8") as stream:
                def on_line(line):
                    stream.write(line + "\n"); stream.flush()
                    print(f"[{index}/{total} rep{repetition}/{requested_repetitions}] {line}", flush=True)
                row["dispatch_attempted"] = True
                if cell_context is not None:
                    attempt["transport_call_entered"] = True
                rc = transport.run_streaming(command, on_line=on_line, timeout=7200,
                    env={**row.get("remote_environment", {}), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                         "ONNX_SPLITPOINT_ARTIFACT_POLICY": "cache_verify_only"})
            attempt.update(returncode=rc, status="exited")
        except BaseException as dispatch_error:
            if cell_context is not None:
                attempt.update(status="dispatch_error", error=f"{type(dispatch_error).__name__}: {dispatch_error}")
            raise
        finally:
            attempt["finished_at"] = now()
            attempt["download"] = download_case(transport, remote_output, local_output, len(attempts))
            write_json(attempts_path, {"attempts": attempts})
        valid = load_valid_repetitions(row["result_path"], row)
        if row.get("admission_mode") == "generic_only" and valid:
            from ..generic_only_reporting import validate_completion_attestations
            validate_completion_attestations({"repetitions": valid}, row, expected_repetitions=len(valid))
        completed = {r["repetition_index"] for r in valid}
        if attempt.get("returncode") != 0 or repetition not in completed:
            if cell_context is not None:
                from .generic_only_cell_policy import require_no_recorded_integrity_failure
                require_no_recorded_integrity_failure(logfile)
            raise RuntimeError(f"Generic case stopped: {case_name(row)} repetition={repetition}, rc={attempt.get('returncode')}")
        print(f"[completion] {index}/{total} {case_name(row)} repetition={repetition}/{requested_repetitions} VALID", flush=True)
    final = validate_completion_case(read_json(Path(row["result_path"])), row, result_root=local_output)
    if requested_repetitions == 1:
        if final['valid_repetition_count'] != 1 or final['pending_repetition_indices'] != [2, 3]:
            raise RuntimeError('Generic functional acceptance lacks one complete validated process')
    elif final["status"] != "completed":
        raise RuntimeError("Generic case lacks three valid independent repetitions")


def terminal_bundle(output: Path) -> None:
    """Small diagnostics only; never include models, datasets or raw tensors."""
    destination = output / "completion-diagnostics.tar.gz"
    roots = [output / "reports", output / "plan", output / "device-preflight"]
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file() and
                         p.suffix in {".json", ".csv", ".txt", ".log", ".md"} and p.stat().st_size <= 5 * 1024**2)
    files.extend(p for p in output.iterdir() if p.is_file() and p.suffix in {".json", ".log", ".txt", ".md"}
                 and p.stat().st_size <= 5 * 1024**2)
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=str(path.relative_to(output)), recursive=False)


def write_status_report(plan: dict, summary: dict, output: Path, *, status: str, error: str = ""):
    if plan.get("scope_mode") == "generic_only_completed_task":
        from ..generic_only_reporting import write_status_report as write_generic_only_status
        return write_generic_only_status(plan, summary, output, status=status, error=error)
    remaining = summary["planned_cases"] - summary["status_counts"]["completed"]
    text = ["THESIS20 Generic Completed Task", f"Status: {status}",
            f"Execution: {plan['execution_id']}", f"Updated: {now()}",
            f"Original run (read only): {plan['original_run']}",
            f"Case balance: {json.dumps(summary['status_counts'], sort_keys=True)}",
            f"Valid repetitions: {summary['valid_repetitions']}/576; remaining cases: {remaining}",
            f"Actual measured completed tasks: {summary['completed_task_count']}/576000",
            f"Technical/quality/claim pairs: {summary['technical_pairs']}/{summary['quality_pairs']}/{summary['claim_pairs']}",
            "Each valid repetition contains 100 completed warmups and 1000 completed measured tasks.",
            "A single bound prepared input is repeated; this is not a 1000-image quality dataset.",
            "Original Generic, Quality, Native performance and energy results are preserved.",
            "No Quality campaign, Native performance, energy capture or model compilation is part of this entry.",
            "Development/screening evidence remains development/screening; no holdout claim is inferred.",
            "Completed-task FPS is count/makespan. Per-request latency is not derived from its reciprocal.",
            "Tables: reports/generic_completion_observations.csv, generic_completion_cross_runner_pairs.csv, generic_completion_cross_runner_groups.csv",
            "Raw repetitions: results/<setup>/<model>/<case>/; every dispatch and failed attempt is retained.",
            "Exact commands: per-case dispatch_attempts.json and dispatch_*.log.",
            "Plan/boundaries: plan/execution-plan.json; small diagnostic archive: completion-diagnostics.tar.gz."]
    if error:
        text.append("Stopped without automatic repair/restart: " + error)
    (output / "STATUS.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    observations = read_json(output / "reports" / "generic_completion_observations.json")["rows"]
    import csv
    fields = ("model_id", "case_id", "direction", "setup_id", "precision", "status",
              "valid_repetition_count", "completed_task_count", "completed_task_fps", "errors", "result_path")
    with (output / "reports" / "case-balance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict))
                         else row.get(key) for key in fields} for row in observations)


def _write_completion_reports(plan, results, reports):
    if plan.get("scope_mode") == "generic_only_completed_task":
        from ..generic_only_reporting import write_completion_reports
    else:
        from ..generic_completion_reporting import write_completion_reports
    return write_completion_reports(plan, results, reports)


def execute_plan(plan_path: Path, *, resume: bool = False,
                 _cancel_event: threading.Event | None = None, _source_guard=None) -> int:
    plan = read_json(plan_path)
    from ..generic_only_plan import MODE, validation_scope
    with validation_scope() if plan.get("scope_mode") == MODE else nullcontext():
        return _execute_plan(plan_path, plan, resume=resume,
                             _cancel_event=_cancel_event, _source_guard=_source_guard)


def _execute_plan(plan_path: Path, plan: dict, *, resume: bool = False,
                  _cancel_event: threading.Event | None = None, _source_guard=None) -> int:
    from ..generic_completion_reporting import load_valid_repetitions, completion_pair, validate_completion_case
    primary = Path(plan["original_run"]).resolve()
    output = outside_primary(Path(plan["output_root"]), primary)
    rows = validate_plan(plan, Path(plan["scope_source"]), primary)
    bundle = output / "runner-bundle.tar.gz"
    if not bundle.is_file():
        raise FileNotFoundError("prepare the final tested session runner bundle before production")
    targets = bound_targets(primary, rows, **({"producer_run_binding": plan["producer_run_binding"]}
        if plan.get("producer_run_binding") is not None else {}))
    starts = plan["start_case_keys"]
    selected_groups = {(r["setup_id"], r["task"]) for r in rows}
    generic_only = plan.get("scope_mode") == "generic_only_completed_task"
    independent = plan.get("cell_execution_control") is not None
    if independent:
        from .generic_only_cell_policy import require_bound_control
        require_bound_control(plan["cell_execution_control"], plan["cell_execution_control"])
        if not generic_only:
            raise ValueError("Independent cell continuation requires Generic-only admission")
    elif resume:
        raise ValueError("--resume requires an explicitly prepared independent-cell plan")
    required_groups = (selected_groups if plan.get("scope_mode") in {"audited_selective_remeasurement", "generic_only_completed_task"}
                       else {(setup, task) for setup in targets for task in ("classification", "detection")})
    if len(starts) != len(required_groups) or len(set(starts)) != len(starts) or not set(starts) <= {r["case_key"] for r in rows}:
        raise ValueError("authorized start cases missing for selected setup/task groups")
    start_rows = [next(r for r in rows if r["case_key"] == key) for key in starts]
    if {(r["setup_id"], r["task"]) for r in start_rows} != required_groups:
        raise ValueError("start cases must cover each selected setup/task group")
    ordered = start_rows + [r for r in rows if r["case_key"] not in starts]
    session = plan["execution_id"]
    cancel = _cancel_event if _cancel_event is not None else threading.Event()
    if threading.current_thread() is not threading.main_thread() and _cancel_event is None:
        raise ValueError("Threaded Generic dispatch requires the owning controller cancellation event")
    scope = RemoteProcessLeaseScope(primary.name, session)
    registry = RemoteProcessLeaseRegistry()
    branch_ownership = {}
    transports = {}
    status, error, current = "blocked", "", None
    rc = 1
    old_handlers = {}
    admitted = False
    controller_marker = None
    initial_plan = None
    blocked_setups = set()
    lock_failures = {}
    stack = ExitStack()
    def cancelled(signum, frame):
        cancel.set()
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, cancelled)
    try:
        stack.enter_context(EvaluationRunLock(out_root=output.parent, run_dir=output,
                            owner={"run_id": primary.name, "session_id": session, "purpose": "generic_completed_task_192"}))
        physical_locks = {}
        for setup, target in targets.items():
            try:
                lock = EvaluationRunLock.for_resource(physical_dut_key(target),
                    owner={"run_id": primary.name, "session_id": session, "purpose": "generic_completed_task_192"})
                stack.enter_context(lock)
                physical_locks[setup] = lock
            except Exception as lock_error:
                if not independent:
                    raise
                blocked_setups.add(setup)
                lock_failures[setup] = f"{type(lock_error).__name__}: {lock_error}"
        def fence_unresolved_cleanup():
            if independent:
                for setup, owner in branch_ownership.items():
                    pending = owner["registry"].active_count()
                    if pending or owner["registry"].cancelled or owner["journal"].pending_count():
                        physical_locks[setup].commit_quarantine_fence({
                            "reason": "generic_completion_remote_cleanup_unproven",
                            "pending_operations": pending, "journal": str(owner["journal"].directory)})
                return
            pending = registry.active_count()
            if pending:
                for lock in physical_locks.values():
                    lock.commit_quarantine_fence({"reason": "generic_completion_remote_cleanup_unproven",
                        "pending_operations": pending, "journal": str(output / "remote-process-leases" / session)})
        stack.callback(fence_unresolved_cleanup)
        start_record = output / "production-start.json"
        if independent:
            from .generic_only_cell_policy import admit_controller_attempt, require_clean_ownership
            initial_plan, controller_marker = admit_controller_attempt(plan, plan_path, output, bundle,
                resume=resume, started_at=now())
        else:
            # The default path retains its existing once-only entry.
            with start_record.open("x", encoding="utf-8") as stream:
                json.dump({"execution_id": session, "pid": os.getpid(), "ppid": os.getppid(),
                           "started_at": now(), "plan": str(plan_path), "runner_bundle": str(bundle),
                           "runner_bundle_sha256": file_sha256(bundle)}, stream, indent=2)
        admitted = True
        if not independent:
            journal = registry.configure_journal(scope=scope, journal_dir=output / "remote-process-leases" / session)
        preflight = (output / "production-preflight" / controller_marker.stem
                     if independent else output / "production-preflight")
        if not independent:
            observe_owners(targets, preflight, generic_only)
        for setup, target in targets.items():
            if independent and setup in lock_failures:
                write_json(output / "branch-stops" / controller_marker.stem / (setup + ".json"),
                    {"at": now(), "setup_id": setup, "phase": "normal_physical_resource_lock",
                     "error": lock_failures[setup], "measurement_execution_id": session, "automatic_retry": False})
                for row in rows:
                    if row["setup_id"] == setup:
                        row.update(execution_status="blocked", execution_error=lock_failures[setup])
                continue
            try:
                branch_scope, branch_registry = scope, registry
                if independent:
                    branch_scope = RemoteProcessLeaseScope(primary.name, session + ":" + setup)
                    branch_registry = RemoteProcessLeaseRegistry()
                    branch_journal = branch_registry.configure_journal(scope=branch_scope,
                        journal_dir=output / "remote-process-leases" / session / setup)
                    branch_ownership[setup] = {"registry": branch_registry, "journal": branch_journal,
                        "scope": branch_scope}
                    require_clean_ownership(branch_registry, branch_journal, branch_scope)
                    observe_owners({setup: target}, preflight / "branches" / setup, True)
                transport = transport_for(target, output / "diagnostics" / setup,
                    scope=branch_scope, registry=branch_registry, cancel=cancel)
                transports[setup] = transport
                config_path = preflight / setup / "configuration.json"
                snapshot = capture_configuration(transport, target, primary=primary, output=config_path, phase="pre")
                representative = next(r for r in rows if r["setup_id"] == setup)
                verify_runtime_environment(transport, representative,
                    preflight / setup / "runtime-environment.json")
                for row in rows:
                    if row["setup_id"] == setup:
                        row["runtime_environment_comparison"] = configuration_comparison(row, snapshot, config_path)
                remote_root = next(r["remote_root"] for r in rows if r["setup_id"] == setup)
                upload_new_file(transport, bundle, remote_root + "/runner-bundle.tar.gz")
                checked_remote("tar -xzf " + shlex.quote(remote_root + "/runner-bundle.tar.gz")
                               + " -C " + shlex.quote(remote_root), transport)
            except Exception as branch_error:
                if not independent or cancel.is_set():
                    raise
                blocked_setups.add(setup)
                message = f"{type(branch_error).__name__}: {branch_error}"
                write_json(output / "branch-stops" / controller_marker.stem / (setup + ".json"),
                    {"at": now(), "setup_id": setup, "phase": "normal_preflight", "error": message,
                     "measurement_execution_id": session, "automatic_retry": False})
                for row in rows:
                    if row["setup_id"] == setup:
                        row.update(execution_status="blocked", execution_error=message)
        write_json(plan_path, plan)
        status = "start_cases_running"
        for index, row in enumerate(ordered, 1):
            current = row
            if independent and row["setup_id"] in blocked_setups:
                row.update(execution_status="blocked", execution_error="Affected backend branch blocked by earlier source/ownership failure")
                write_json(plan_path, plan)
                continue
            if cancel.is_set():
                raise RuntimeError("completion cancelled before next case")
            entered = time.monotonic()
            context = None
            if independent:
                from .generic_only_cell_policy import (
                    CellAttemptLimitReached, CellCompletionRejected, InvalidResumeBinding, UnresolvedOwnedProcesses,
                )
                owner = branch_ownership[row["setup_id"]]
                context = {"initial_row": next(item for item in initial_plan["rows"] if item["case_key"] == row["case_key"]),
                    "control": plan["cell_execution_control"], "initial_control": initial_plan["cell_execution_control"],
                    "registry": owner["registry"], "journal": owner["journal"], "scope": owner["scope"]}
                cell_error = None
                # At most six durable reservations exist: two for each original
                # index. run_case itself skips every normally validated index.
                for _ in range(7):
                    try:
                        run_case(row, transports[row["setup_id"]], output, index, total=len(rows),
                            cell_context=context, source_guard=_source_guard)
                        cell_error = None
                        break
                    except (CellAttemptLimitReached, CellCompletionRejected, InvalidResumeBinding, UnresolvedOwnedProcesses) as exc:
                        cell_error = exc
                        if exc.failure_scope != "cell":
                            blocked_setups.add(row["setup_id"])
                        break
                    except RuntimeError as exc:
                        cell_error = exc
                        if cancel.is_set():
                            raise RuntimeError("completion cancelled before bounded retry") from exc
                        try:
                            require_clean_ownership(owner["registry"], owner["journal"], owner["scope"])
                        except UnresolvedOwnedProcesses as unsafe:
                            cell_error = unsafe
                            blocked_setups.add(row["setup_id"])
                            break
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        cell_error = exc
                        blocked_setups.add(row["setup_id"])
                        break
                if cell_error is not None:
                    row.update(execution_status="failed" if row.get("dispatch_attempted") else "blocked",
                        execution_error=f"{type(cell_error).__name__}: {cell_error}")
                    write_json(plan_path, plan)
                    summary = _write_completion_reports(plan, output / "results", output / "reports")
                    write_json(output / "cell-stops" / controller_marker.stem / (case_name(row) + ".json"), {
                        "at": now(), "case_key": row["case_key"], "error": row["execution_error"],
                        "failure_scope": getattr(cell_error, "failure_scope", "backend" if row["setup_id"] in blocked_setups else "cell"),
                        "valid_repetition_indices": [item["repetition_index"] for item in load_valid_repetitions(row["result_path"], row)],
                        "automatic_unbounded_retry": False})
                    continue
            else:
                run_case(row, transports[row["setup_id"]], output, index, total=len(rows), source_guard=_source_guard)
            if _source_guard is not None:
                _source_guard()
            row["completion_observed_at"] = now()
            row["dispatch_wall_seconds"] = time.monotonic() - entered
            if independent:
                row["execution_status"] = "completed"
                row.pop("execution_error", None)
            observed = validate_completion_case(read_json(Path(row["result_path"])), row,
                result_root=Path(row["result_path"]).parent)
            if generic_only:
                from ..generic_only_reporting import generic_only_admission
                admission = (generic_only_admission(observed, row, cell_control=plan["cell_execution_control"])
                             if independent else generic_only_admission(observed, row))
                if not admission["eligible_for_generic_only_execution"]:
                    if independent:
                        row.update(execution_status="failed", execution_error="case has no valid Generic-only completion admission")
                        write_json(output / "cell-stops" / controller_marker.stem / (case_name(row) + ".admission.json"), admission)
                        write_json(plan_path, plan)
                        _write_completion_reports(plan, output / "results", output / "reports")
                        continue
                    else:
                        write_json(output / "generic-only-admission-failure.json", admission)
                        raise RuntimeError("case has no valid Generic-only completion admission: " + row["case_key"])
            else:
                pair = completion_pair(observed, row)
                if row["case_key"] in starts and not pair["eligible_for_technical_transfer"]:
                    write_json(output / "start-case-pairing-failure.json", pair)
                    raise RuntimeError("start case has no valid completed-task pair: " + row["case_key"])
            write_json(plan_path, plan)
            summary = _write_completion_reports(plan, output / "results", output / "reports")
            if index == len(starts) and (not independent or all(
                    item.get("execution_status") == "completed" for item in start_rows)):
                write_json(output / "start-cases-accepted.json", {"accepted_at": now(),
                    "case_keys": starts, "valid_repetitions": 3 * len(starts), "remaining_cases": len(rows) - len(starts),
                    "original_start_results_reused": True, "summary": summary})
                print(f"[completion] START CASES ACCEPTED {len(starts)}/{len(starts)}; continuing remaining {len(rows)-len(starts)}", flush=True)
                status = "remainder_running"
            durations = {(r["model_id"], r["setup_id"]): r["dispatch_wall_seconds"] for r in rows if r.get("dispatch_wall_seconds")}
            unresolved_groups = []
            estimate = 0.0
            for pending in ordered[index:]:
                measured = durations.get((pending["model_id"], pending["setup_id"]))
                if measured is None:
                    unresolved_groups.append((pending["model_id"], pending["setup_id"]))
                else:
                    estimate += measured
            write_json(output / "progress.json", {"updated_at": now(), "status": status,
                "last_completed_case": row["case_key"], "case_index": index, "total": len(rows),
                "remaining": len(rows)-index, "measured_group_remaining_seconds": estimate,
                "groups_without_observed_duration": sorted(set(unresolved_groups)), "summary": summary})
            write_status_report(plan, summary, output, status=status)
        summary = _write_completion_reports(plan, output / "results", output / "reports") if independent else None
        status, rc = (("complete", 0) if not independent or summary["all_cases_measured"]
                      else ("complete_with_cell_failures", 1))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if not admitted:
            return rc
        status = "cancelled" if cancel.is_set() else "stopped"
        if current is not None:
            current["execution_status"] = "failed" if current.get("dispatch_attempted") else "blocked"
            current["execution_error"] = error
            write_json(plan_path, plan)
        stop = {"at": now(), "error": error,
            "case_key": current.get("case_key") if current else None, "automatic_restart": False}
        write_json(output / "technical-stop.json", stop)
        if controller_marker is not None:
            write_json(controller_marker.with_suffix(".technical-stop.json"), stop)
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        try:
            if admitted:
                # SSHTransport already performs exact token-bound cleanup on failures.
                # Any unresolved lease is retained and fenced, never globally killed.
                cleanup = {"at": now(), "active_owned_remote_operations": registry.active_count(),
                    "registry_cancelled_or_poisoned": registry.cancelled, "foreign_processes_stopped": False}
                if independent:
                    branches = {}
                    for setup, owner in branch_ownership.items():
                        directory = owner["journal"].directory
                        branches[setup] = {"active_owned_remote_operations": owner["registry"].active_count(),
                            "registry_cancelled_or_poisoned": owner["registry"].cancelled,
                            "journal": str(directory), "journal_pending_count": owner["journal"].pending_count(),
                            "journal_cancelled": owner["journal"].is_cancelled(),
                            "journal_empty": directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir()),
                            "ownership_scope": {"run_sha256": owner["scope"].run_sha256,
                                "session_sha256": owner["scope"].session_sha256}}
                    cleanup.update(per_backend_journals=branches,
                        active_owned_remote_operations=sum(item["active_owned_remote_operations"] for item in branches.values()),
                        registry_cancelled_or_poisoned=any(item["registry_cancelled_or_poisoned"] for item in branches.values()))
                write_json(output / "cleanup-status.json", cleanup)
                summary = _write_completion_reports(plan, output / "results", output / "reports")
                write_status_report(plan, summary, output, status=status, error=error)
                write_json(output / "terminal-status.json", {"finished_at": now(), "status": status, "exitcode": rc,
                           "summary": summary, "error": error, "automatic_restart": False})
                if controller_marker is not None:
                    write_json(controller_marker.with_suffix(".terminal.json"), {
                        "finished_at": now(), "status": status, "exitcode": rc, "summary": summary,
                        "cleanup": read_json(output / "cleanup-status.json"), "error": error})
                terminal_bundle(output)
        finally:
            stack.close()
    return rc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", type=Path, metavar="SCOPE_PLAN_JSON")
    actions.add_argument("--execute", type=Path, metavar="EXECUTION_PLAN_JSON")
    actions.add_argument("--execute-batch", type=Path, metavar="GENERIC_ONLY_PARALLEL_BATCH_JSON")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true", help="explicit bounded continuation of the same independent-cell execution")
    parser.add_argument("--hailo-async-inflight", type=int, choices=range(1, 33), metavar="N",
                        help="prepare Hailo10H rows with N asynchronous stage1 slots (default: historical synchronous stage1)")
    parser.add_argument("--deepx-async-inflight", type=int, choices=range(1, 33), metavar="N",
                        help="prepare DeepX M1 rows with N asynchronous stage1 slots (default: historical synchronous stage1)")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.resume:
            parser.error("--resume is execute-only")
        if args.output_root is None:
            parser.error("--prepare requires --output-root outside the primary run")
        kwargs = {key: getattr(args, key) for key in ("hailo_async_inflight", "deepx_async_inflight")
                  if getattr(args, key) is not None}
        print(prepare_execution(args.prepare, args.output_root, **kwargs))
        return 0
    if args.hailo_async_inflight is not None or args.deepx_async_inflight is not None:
        parser.error("async inflight options are prepare-only; --execute uses its bound plan")
    if args.execute_batch:
        from .generic_only_parallel import execute_batch
        return execute_batch(args.execute_batch, resume=args.resume)
    return execute_plan(args.execute, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())


def bind_runtime_source_closure(row: dict, bundle_root: Path, *, runner_root: str | None = None) -> dict:
    """Bind the normal generated suite before either performance or energy runs."""
    from ..runners.generic_energy import file_hash
    root = Path(bundle_root).resolve()
    remote = runner_root or row['remote_root']
    runner = root / 'runner' / 'run_split_onnxruntime.py'
    if not runner.is_file() or not (root / 'splitpoint_runners' / 'generic_energy.py').is_file():
        raise ValueError('Generic energy requires the current generated suite closure')
    paths = [runner, *sorted((root / 'splitpoint_runners').rglob('*.py'))]
    sources = sorted([{'path': remote + '/' + str(p.relative_to(root)), 'sha256': file_hash(p)} for p in paths], key=lambda x: x['path'])
    # Bind the already prepared model metadata and sealed split binding too:
    # those files select tensor mappings even though they are not Python code.
    from ..benchmark.schema import dumps_json_safe
    for item in row.get('staging_files') or []:
        digest = (hashlib.sha256(dumps_json_safe(dict(item['payload']), indent=2).encode('utf-8')).hexdigest()
                  if 'payload' in item else item['sha256'])
        sources.append({'path': item['remote'], 'sha256': digest})
    if row.get('admission_mode') == 'generic_only':
        for artifact in row['runtime_config']['generic_artifact_binding']['artifacts'].values():
            descriptor = {key: artifact[key] for key in ('path', 'sha256')}
            existing = [source for source in sources if source['path'] == descriptor['path']]
            if existing and any(source != descriptor for source in existing):
                raise ValueError('Generic-only artifact source binding conflicts')
            if not existing:
                sources.append(descriptor)
    if row.get('runtime_config', {}).get('mixed_input_control'):
        # The first functional input is the existing bound throughput input.
        # Observe its bytes without staging into or rewriting its old archive.
        prepared = row['runtime_config']['input_binding']['prepared_input']
        descriptor = {'path': prepared['path'], 'sha256': prepared['sha256']}
        existing = [item for item in sources if item['path'] == descriptor['path']]
        if existing and any(item != descriptor for item in existing):
            raise ValueError('Mixed control prepared input source binding conflicts')
        if not existing:
            sources.append(descriptor)
    sources.sort(key=lambda x: x['path'])
    if row.get('runtime_config', {}).get('runtime_source_files') not in (None, sources):
        raise ValueError('Existing Generic runtime source closure differs')
    row['runtime_config']['runtime_source_files'] = sources
    from ..runners.generic_energy import execution_environment
    row['runtime_config']['execution_environment'] = execution_environment(row['remote_python_executable'], row.get('remote_environment') or {})
    return row


def _restore_energy_execution_environment(command: str, execution_identity: dict) -> str:
    """Restore the measured environment before executing the unchanged payload."""
    argv = shlex.split(command)
    if not any(Path(part).name == 'ssh' for part in argv[:-1]):
        return command
    environment = execution_identity.get('environment')
    if not argv or not isinstance(environment, dict) or not environment:
        raise ValueError('Measured Generic execution environment missing')
    unset = []
    values = []
    for name, value in sorted(environment.items()):
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
            raise ValueError('Measured Generic environment name invalid')
        if value is None:
            unset += ['-u', name]
        elif isinstance(value, str):
            values.append(name + '=' + value)
        else:
            raise ValueError('Measured Generic environment value invalid')
    argv[-1] = shlex.join(['env', *unset, *values, 'bash', '-c', argv[-1]])
    return shlex.join(argv)


def _prepare_generic_only_energy_sources(row, config, performance_result, output, remote_base,
                                         original_bundle, admission):
    """Clone the existing code seal; change only declared untimed energy guards."""
    import hashlib
    import io
    import tarfile
    import shutil
    from ..generic_energy_reporting import (ENERGY_VALIDATOR_FUNCTIONS,
        ENERGY_OLD_COUNT_CALL, ENERGY_NEW_COUNT_CALL)
    from ..runners.generic_energy import file_hash
    if (admission or {}).get('admission_mode') != 'generic_only':
        raise ValueError('Generic-only energy requires its completed normal source admission')
    original_bundle = Path(original_bundle).resolve(strict=True)
    old_sources = list(config['runtime_source_files'])
    runners = [item for item in old_sources if item['path'].endswith('/runner/run_split_onnxruntime.py')]
    if len(runners) != 1:
        raise ValueError('Unique original FPS runner required')
    old_root = str(PurePosixPath(runners[0]['path']).parent.parent)
    new_root = remote_base + '/runtime'
    source_map = {item['path']: item['sha256'] for item in old_sources}
    names = sorted(path[len(old_root) + 1:] for path in source_map
                   if path.startswith(old_root + '/runner/') or path.startswith(old_root + '/splitpoint_runners/'))
    bundle = output / 'energy-runtime-bundle.tar.gz'
    additions = []
    with tarfile.open(original_bundle) as before, tarfile.open(bundle, 'w:gz') as after:
        for name in names:
            member = before.getmember(name)
            if not member.isfile() or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Original energy source bundle member differs')
            raw = before.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != source_map[old_root + '/' + name]:
                raise ValueError('Original bundle is not the actual FPS source closure')
            if name in ENERGY_VALIDATOR_FUNCTIONS:
                raw = (Path(__file__).resolve().parents[1] / 'runners' / Path(name).name).read_bytes()
            elif name == 'runner/run_split_onnxruntime.py':
                text = raw.decode()
                if text.count(ENERGY_OLD_COUNT_CALL) != 1:
                    raise ValueError('Original runner lacks the exact audited postflight call')
                raw = text.replace(ENERGY_OLD_COUNT_CALL, ENERGY_NEW_COUNT_CALL, 1).encode()
            info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o644
            after.addfile(info, io.BytesIO(raw))
            additions.append({'path': new_root + '/' + name, 'sha256': hashlib.sha256(raw).hexdigest()})
    evidence = []
    for role, source in [('performance_result', performance_result),
                         ('normal_admission_report', admission['normal_generic_only_report']['path'])]:
        source = Path(source)
        target = output / (role + '.json'); shutil.copyfile(source, target)
        descriptor = {'path': str(target), 'sha256': file_hash(target)}
        if role == 'normal_admission_report' and descriptor['sha256'] != admission['normal_generic_only_report']['sha256']:
            raise ValueError('Generic-only normal admission changed during preparation')
        remote = remote_base + '/evidence/' + target.name
        evidence.append({'role': role, 'local': descriptor, 'runtime_path': remote})
        additions.append({'path': remote, 'sha256': descriptor['sha256']})
    binding = {'schema': 'onnx-splitpoint/generic-only-energy-verifier-binding/v1',
        'original_bundle': {'path': str(original_bundle), 'sha256': file_hash(original_bundle)},
        'energy_bundle': {'path': str(bundle), 'sha256': file_hash(bundle)},
        'original_runtime_root': old_root, 'energy_runtime_root': new_root,
        'original_source_files': old_sources, 'changed_functions': ENERGY_VALIDATOR_FUNCTIONS,
        'runner_delta': 'exact_postflight_actual_completed_count_call', 'evidence': evidence}
    reused = {'schema': 'onnx-splitpoint/generic-only-energy-control-reuse/v1',
        'original_mixed_input_control': config['mixed_input_control'],
        'normal_admission': admission['normal_generic_only_admission']}
    for item in evidence:
        reused[item['role']] = {'path': item['runtime_path'], 'sha256': item['local']['sha256']}
    config['mixed_input_control'] = False
    config['runtime_source_files'] = sorted([*old_sources, *additions], key=lambda item: item['path'])
    return binding, reused


def prepare_energy_case(row: dict, *, performance_result: Path, output: Path,
                        collector_python: str, collector_script: Path,
                        hardware_setups_file: Path, remote_command_prefix: list[str],
                        campaign_budget_file: Path,
                        duration_s: float = 60.0,
                        require_environment_observation: bool = False,
                        environment_run_id: str | None = None,
                        original_runtime_bundle: Path | None = None,
                        energy_performance_admission: dict | None = None) -> dict:
    """Prepare ordinary collector commands; perform no device or sensor action.

    The caller supplies the existing resource-owned remote command prefix.
    This function neither adds an SSH transport nor schedules a capture.
    """
    import copy
    requested_repetitions = _acceptance_repetitions(row)
    if require_environment_observation and not environment_run_id:
        raise ValueError('Required Generic environment needs its parent run identity')
    campaign_budget_file = Path(campaign_budget_file).resolve(strict=True)
    if not campaign_budget_file.is_file():
        raise ValueError('Existing shared campaign budget required')
    from ..runners.generic_energy import (build_contract, file_hash, canonical_hash,
        NONCE_TOKEN, REPEAT_TOKEN, ATTESTATION_TOKEN)
    from ..runners.task_completion import valid_split_completion_repetitions
    config = copy.deepcopy(row['runtime_config'])
    identity = config['identity']
    campaign_row_id = json.dumps(['generic_completed:' + identity['comparison_backend'],
        identity['model_id'], identity['case_id'], identity['setup_id']], separators=(',', ':'))
    from ..energy.config import energy_setup_from_registry, load_hardware_registry
    setup = energy_setup_from_registry(load_hardware_registry(hardware_setups_file), identity['setup_id'], registry_path=hardware_setups_file)
    source_id = str(setup.urecs_address).strip().lower()
    if not source_id:
        raise ValueError('Generic energy requires the declared physical campaign source')
    performance = read_json(performance_result)
    repeats = valid_split_completion_repetitions(performance, config, result_root=performance_result.parent)
    if (len(repeats) != requested_repetitions or [r['repetition_index'] for r in repeats] != list(range(1, requested_repetitions + 1))
            or performance.get('process_repetition_policy') != 'separate_processes'):
        raise ValueError('Generic energy requires the requested actual current performance processes')
    sources = config.get('runtime_source_files') or []
    wanted = {r['path']: r['sha256'] for r in sources}
    modes = {r.get('completion_runtime_mode', 'classification_top1_top5') for r in repeats}
    if not wanted or len(modes) != 1 or any(r.get('runtime_source_binding') != {'before': wanted, 'after': wanted, 'unchanged': True} for r in repeats):
        raise ValueError('Generic energy performance source/mode binding missing or changed')
    execution_identity = repeats[0].get('execution_identity')
    from ..runners.generic_energy import execution_environment
    declared = execution_environment(row['remote_python_executable'], row.get('remote_environment') or {})
    if (not execution_identity or any(r.get('execution_identity') != execution_identity for r in repeats)
            or config.get('execution_environment') != declared):
        raise ValueError('Generic energy performance Python/environment binding missing or changed')
    runner_paths = [r['path'] for r in sources if r['path'].endswith('/runner/run_split_onnxruntime.py')]
    if len(runner_paths) != 1:
        raise ValueError('Unique bound Generic runner required')
    output = outside_primary(output, Path(row['original_run']))
    output.mkdir(parents=True, exist_ok=False)
    capture_id = uuid.uuid4().hex
    remote_root = str(PurePosixPath(runner_paths[0]).parent.parent)
    remote_base = remote_root + '/energy/' + case_name(row) + '/' + capture_id
    binding = reused_control = None
    active_runner = runner_paths[0]
    if config.get('admission_mode') == 'generic_only':
        if original_runtime_bundle is None:
            raise ValueError('Generic-only energy requires its original sealed runtime bundle')
        binding, reused_control = _prepare_generic_only_energy_sources(row, config, performance_result,
            output, remote_base, original_runtime_bundle, energy_performance_admission)
        remote_root = binding['energy_runtime_root']
        active_runner = remote_root + '/runner/run_split_onnxruntime.py'
        sources = config['runtime_source_files']
    contract = build_contract(config, source_files=sources, runner_path=active_runner,
                              completion_runtime_mode=next(iter(modes)), duration_s=duration_s,
                              execution_identity=execution_identity)
    config['energy_capture'] = {'contract': contract, 'performance_result_sha256': file_hash(performance_result)}
    if binding:
        config['energy_capture'].update(verifier_binding=binding, generic_only_control_reuse=reused_control)
        from ..generic_energy_reporting import verified_energy_performance_sources
        verified_energy_performance_sources(config, row['runtime_config'])
    config_path = output / 'generic_energy_config.json'; write_json(config_path, config)
    remote_config = remote_base + '/generic_energy_config.json'
    runtime_attestation = remote_base + '/preflight_' + NONCE_TOKEN + '.json'
    calls = []
    for repetition in range(1, requested_repetitions + 1):
        remote_output = remote_base + f'/repetition_{repetition}/' + NONCE_TOKEN
        workload = [row['remote_python_executable'], '-B', '-u', active_runner,
            '--generic-completion-config', remote_config, '--generic-completion-repetition', str(repetition),
            '--generic-completion-energy', '--energy-preflight-nonce', NONCE_TOKEN,
            '--energy-preflight-attestation', ATTESTATION_TOKEN,
            '--energy-command-contract-sha256', contract['command_contract_sha256'], '--out-dir', remote_output]
        preflight = [row['remote_python_executable'], '-B', '-m', 'splitpoint_runners.generic_energy',
            '--config', remote_config, '--nonce', NONCE_TOKEN, '--attestation-path', ATTESTATION_TOKEN,
            '--expected-command-contract-sha256', contract['command_contract_sha256']]
        def command(argv):
            environment = dict(row.get('remote_environment') or {}, PYTHONDONTWRITEBYTECODE='1', ONNX_SPLITPOINT_ARTIFACT_POLICY='cache_verify_only')
            shell = 'cd -- ' + shlex.quote(remote_root) + ' && ' + shlex.join(['env', *[k+'='+v for k,v in sorted(environment.items())], *argv])
            if remote_command_prefix:
                return _restore_energy_execution_environment(
                    shlex.join([*remote_command_prefix, shell]), execution_identity)
            return shell
        command_file = output / f'workload_{repetition}.txt'; command_file.write_text(command(workload)+'\n')
        preflight_file = output / f'preflight_{repetition}.txt'; preflight_file.write_text(command(preflight)+'\n')
        argv = [collector_python, '-B', str(collector_script), 'measure', '--setup-id', row['setup_id'],
            '--hardware-setups-file', str(hardware_setups_file), '--out', str(output / f'capture_{repetition}'),
            '--command-file', str(command_file), '--preflight-command-file', str(preflight_file),
            '--preflight-runtime-attestation-path', runtime_attestation,
            '--preflight-expected-command-contract-sha256', contract['command_contract_sha256'],
            '--duration', str(duration_s), '--runs', '1', '--exact-run-count', '--invalid-repeat-max-retries', '0',
            '--campaign-budget-file', str(campaign_budget_file), '--campaign-row-id', campaign_row_id,
            '--campaign-repeats', str(requested_repetitions), '--campaign-max-retries', '2', '--campaign-max-transport-failures', '20',
            '--task-logical-repeat', f'repeat:{repetition - 1}',
            '--run-id', config['execution_id'] + f'-generic-energy-{capture_id}-{repetition}',
            '--require-runtime-work-units', '--require-command-window-alignment']
        if require_environment_observation:
            argv += ['--require-environment-observation', '--environment-run-id', environment_run_id]
        calls.append({'repetition_index': repetition, 'argv': argv, 'command_sha256': canonical_hash(argv),
            'workload_command_file': str(command_file), 'workload_command_sha256': file_hash(command_file),
            'preflight_command_file': str(preflight_file), 'preflight_command_sha256': file_hash(preflight_file),
            'remote_result_directory_template': remote_output})
    plan = {'schema': 'onnx-splitpoint/generic-completed-task-energy-plan/v1', 'identity': row['identity'],
        'acceptance_mode': row.get('acceptance_mode', 'scientific'), 'requested_repetitions': requested_repetitions,
        'config': config, 'config_path': str(config_path), 'config_sha256': file_hash(config_path),
        'remote_config_path': remote_config, 'calls': calls, 'automatic_retry': False,
        'execution_environment_restored': bool(remote_command_prefix),
        'campaign_budget_file': str(campaign_budget_file), 'campaign_row_id': campaign_row_id,
        'campaign_source_id': source_id,
        'require_environment_observation': require_environment_observation,
        'environment_run_id': environment_run_id,
        'campaign_limits': {'max_chains': 3 * requested_repetitions, 'max_retries': 2, 'max_transport_failures': 20},
        'performance_result': str(performance_result), 'performance_result_sha256': file_hash(performance_result),
        'collector_script': str(collector_script), 'collector_script_sha256': file_hash(collector_script),
        'hardware_setups_file': str(hardware_setups_file), 'hardware_setups_file_sha256': file_hash(hardware_setups_file),
        'resource_ownership': 'normal_capture_controller_and_DUT_lease_required', 'hardware_executed': False}
    if binding:
        plan['energy_runtime_bundle'] = {'path': binding['energy_bundle']['path'],
            'sha256': binding['energy_bundle']['sha256'], 'remote_path': remote_base + '/energy-runtime-bundle.tar.gz',
            'runtime_root': binding['energy_runtime_root']}
        plan['energy_evidence_sources'] = binding['evidence']
    write_json(output / 'generic_energy_plan.json', plan)
    return plan


def check_energy_campaign(plan: dict, *, logical_repeat: str | None = None,
                          allow_completed_repeats: bool = False,
                          parent_run_dir: Path | None = None) -> dict:
    """Read the shared normal budget before transport; never reset its history."""
    import copy
    import fcntl
    from ..energy.config import energy_setup_from_registry, load_hardware_registry
    from ..energy.task_budget import campaign_budget_stop_reason, EnergyTaskBudget, _source_reentry
    identity = plan['identity']
    requested_repetitions = _acceptance_repetitions(plan)
    expected_key = json.dumps(['generic_completed:' + identity['comparison_backend'],
        identity['model_id'], identity['case_id'], identity['setup_id']], separators=(',', ':'))
    limits = {'max_chains': 3 * requested_repetitions, 'max_retries': 2, 'max_transport_failures': 20}
    if plan.get('campaign_row_id') != expected_key or plan.get('campaign_limits') != limits:
        raise ValueError('Generic energy shared campaign identity/limits changed')
    path = Path(plan['campaign_budget_file'])
    if not path.is_file() or path.is_symlink():
        raise ValueError('Existing regular shared campaign budget required')
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        budget = json.load(stream)
    if budget.get('campaign') is not True or budget.get('max_transport_failures') != 20:
        raise ValueError('Generic energy shared campaign policy differs')
    registry_path = Path(plan['hardware_setups_file'])
    setup = energy_setup_from_registry(load_hardware_registry(registry_path), identity['setup_id'], registry_path=registry_path)
    source_id = str(setup.urecs_address).strip().lower()
    if not source_id or source_id != plan.get('campaign_source_id') or source_id not in budget['sources']:
        raise ValueError('Generic energy requires the existing campaign physical source')
    task = budget['tasks'].get(expected_key)
    if task is not None and (task['source_id'] != source_id or task['limits'] != limits):
        raise ValueError('Generic energy existing case budget differs')
    # Add only to this in-memory projection so the existing normal STOP resolver
    # can also evaluate a Generic case that has never acquired a chain.
    budget = copy.deepcopy(budget)
    task = budget['tasks'].setdefault(expected_key, {'limits': limits, 'chains': [], 'stop_reason': '', 'source_id': source_id})
    reason = campaign_budget_stop_reason(budget, expected_key,
        parent_run_dir=parent_run_dir or plan.get('parent_run_dir'))
    if reason:
        raise RuntimeError('Generic energy shared campaign STOP: ' + reason)
    projection = object.__new__(EnergyTaskBudget)
    projection.data, projection.campaign_row, projection.cohort_id = task, expected_key, ''
    projection.root, projection.environment = budget, {}
    projection._reentry = _source_reentry(budget, source_id, {})
    effective = projection._effective_limits(logical_repeat or 'repeat:0')
    complete = {c.get('logical_repeat') for c in task['chains']
                if c.get('finished') is True and c.get('valid') is True
                and c.get('source_completion_verified') is True}
    reuse_only = allow_completed_repeats and (logical_repeat in complete if logical_repeat is not None
                                             else complete >= {f'repeat:{i}' for i in range(requested_repetitions)})
    if not reuse_only and len(task['chains']) >= effective['max_chains']:
        raise RuntimeError('Generic energy shared case chain limit reached')
    if logical_repeat is not None:
        if logical_repeat not in {f'repeat:{i}' for i in range(requested_repetitions)}:
            raise ValueError('Generic energy logical repeat differs')
        chains = [c for c in task['chains'] if c['logical_repeat'] == logical_repeat]
        if not reuse_only and (any(c.get('finished') and c.get('valid') for c in chains) or len(chains) >= 1 + effective['max_retries']):
            raise RuntimeError('Generic energy logical repeat already valid or retry limit reached')
    return {'campaign_row_id': expected_key, 'source_id': source_id, 'begun_chains': len(task['chains'])}


def stage_energy_case(plan: dict, transport: SSHTransport) -> dict:
    """Stage only the fresh energy config through the existing checked transport."""
    from ..runners.generic_energy import file_hash
    config_path = Path(plan['config_path'])
    if file_hash(config_path) != plan['config_sha256']:
        raise ValueError('Generic energy staged configuration changed')
    bundle = plan.get('energy_runtime_bundle')
    if bundle:
        from ..generic_energy_reporting import verified_energy_performance_sources
        binding = plan['config']['energy_capture']['verifier_binding']
        original = dict(plan['config'], runtime_source_files=binding['original_source_files'])
        verified_energy_performance_sources(plan['config'], original)
        if file_hash(bundle['path']) != bundle['sha256']:
            raise ValueError('Generic energy runtime bundle changed')
        upload_new_file(transport, Path(bundle['path']), bundle['remote_path'], expected_sha=bundle['sha256'])
        checked_remote('mkdir -p -- ' + shlex.quote(bundle['runtime_root']) + ' && tar -xzf '
                       + shlex.quote(bundle['remote_path']) + ' -C ' + shlex.quote(bundle['runtime_root']), transport)
        for item in plan['energy_evidence_sources']:
            upload_new_file(transport, Path(item['local']['path']), item['runtime_path'], expected_sha=item['local']['sha256'])
    return upload_new_file(transport, config_path, plan['remote_config_path'], expected_sha=plan['config_sha256'])


def _verified_energy_resume_command_context(call: dict, base: dict, config: dict) -> bool:
    """Accept only the exact measured-environment wrapper around original bytes."""
    for item in (base,call):
        argv=item['argv']
        for kind,flag in (('workload_command','--command-file'),
                          ('preflight_command','--preflight-command-file')):
            if (argv.count(flag)!=1 or argv.index(flag)+1>=len(argv)
                    or argv[argv.index(flag)+1]!=item[kind+'_file']):
                raise ValueError('Generic actual command file differs from bound metadata')
    changed = ('workload_command_file', 'workload_command_sha256',
               'preflight_command_file', 'preflight_command_sha256', 'original_command_files')
    ignored = ('argv', 'command_sha256')
    if {k:v for k,v in call.items() if k not in ignored} == {
            k:v for k,v in base.items() if k not in ignored}:
        return False
    if {k:v for k,v in call.items() if k not in (*ignored,*changed)} != {
            k:v for k,v in base.items() if k not in (*ignored,*changed)}:
        raise ValueError('Generic resume changed prepared command scope')
    identity = config['energy_capture']['contract']['execution_identity']
    expected_original = {}
    from ..runners.generic_energy import file_hash
    for kind in ('workload_command', 'preflight_command'):
        original = Path(base[kind+'_file'])
        expected_original[kind] = {'path':str(original), 'sha256':base[kind+'_sha256']}
        actual = Path(call[kind+'_file'])
        if (file_hash(original) != base[kind+'_sha256']
                or file_hash(actual) != call[kind+'_sha256']
                or actual.read_text() != _restore_energy_execution_environment(
                    original.read_text().strip(), identity)+'\n'):
            raise ValueError('Generic resume execution environment repair differs')
    if call.get('original_command_files') != expected_original:
        raise ValueError('Generic resume original command bindings differ')
    return True


def prepare_energy_resume(plan_path: Path, case: dict, *, output: Path,
                          previous_plans: list[Path], transport: SSHTransport | None = None,
                          parent_run_dir: Path | None = None) -> dict:
    """Keep original inputs and first normally verified sources; relocate missing captures."""
    import copy
    from ..runners.generic_energy import canonical_hash, file_hash
    from ..generic_energy_reporting import import_generic_energy_capture
    original = read_json(plan_path); selected = {}; selected_calls = {}; rejected = []
    requested_repetitions = _acceptance_repetitions(original)
    unchanged = ('identity', 'acceptance_mode', 'requested_repetitions', 'config', 'config_path', 'config_sha256', 'remote_config_path',
        'performance_result', 'performance_result_sha256', 'collector_script', 'collector_script_sha256',
        'hardware_setups_file', 'hardware_setups_file_sha256', 'campaign_budget_file', 'campaign_row_id',
        'campaign_source_id', 'campaign_limits', 'require_environment_observation', 'environment_run_id')
    output = Path(output)
    if output.exists():raise FileExistsError('Generic resume measurement output must be fresh')
    for previous_path in previous_plans:
        previous = read_json(previous_path); root = Path(previous_path).parent
        if any(previous.get(k) != original.get(k) for k in unchanged):
            raise ValueError('Generic resume changed original scientific inputs or campaign')
        calls = {c['repetition_index']: c for c in previous['calls']}
        for call, base in zip(previous['calls'], original['calls']):
            restored = _verified_energy_resume_command_context(call,base,original['config'])
            argv = list(call['argv']);expected = list(base['argv'])
            for flag in (('--out','--run-id','--command-file','--preflight-command-file')
                         if restored else ('--out','--run-id')):
                if argv.count(flag) != 1 or expected.count(flag) != 1:
                    raise ValueError('Generic resume capture output/run identity missing or ambiguous')
                argv[argv.index(flag)+1] = expected[expected.index(flag)+1]
            if argv != expected or canonical_hash(call['argv']) != call['command_sha256']:
                raise ValueError('Generic resume changed actual collector policy')
        manifest_path = root / 'capture_sources.json'
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        if manifest and manifest.get('plan_sha256') != file_hash(previous_path):
            raise ValueError('Generic resume prior capture manifest plan binding differs')
        descriptors = {d['repetition_index']: d for d in manifest.get('sources', [])}
        for index in range(1, requested_repetitions + 1):
            if index in selected:continue
            call = calls[index]; descriptor = descriptors.get(index)
            if descriptor is None:
                receipt_path = root / f'capture_{index}_process_receipt.json'
                if not receipt_path.is_file():continue
                receipt = read_json(receipt_path)
                if receipt.get('returncode') != 0:continue
                started = read_json(root / f'capture_{index}_started.json')
                if (started.get('argv') != call['argv'] or
                        started.get('command_sha256') != call['command_sha256'] or
                        started.get('plan_sha256') != file_hash(previous_path) or
                        started.get('aggregate_absent_before_execution') is not True or
                        any(receipt.get(k) != v for k, v in started.items()) or
                        not isinstance(started.get('started_ns'), int) or
                        not isinstance(receipt.get('finished_ns'), int) or
                        receipt['finished_ns'] <= started['started_ns']):
                    raise ValueError('Generic resume prior capture start/process binding differs')
                argv = call['argv']; capture = Path(argv[argv.index('--out') + 1]); raw = capture / 'energy_aggregate.json'
                aggregate = read_json(raw)
                records = aggregate.get('runs') or []
                if len(records) != 1:raise ValueError('Generic resume prior capture record count differs')
                nonce = records[0].get('preflight_evidence', {}).get('nonce')
                if not isinstance(nonce, str) or re.fullmatch(r'[0-9a-f]{64}', nonce) is None:
                    raise ValueError('Generic resume prior capture nonce differs')
                remote = call['remote_result_directory_template'].replace('__ONNX_SPLITPOINT_PREFLIGHT_NONCE__', nonce)
                local = root / f'runtime_result_{index}'
                if (local / 'generic_energy_results.json').is_file():
                    downloaded = read_json(root / f'capture_{index}_download.json')
                    if downloaded.get('result_collected') is not True:
                        raise ValueError('Generic resume prior runtime download was not completed')
                if not (local / 'generic_energy_results.json').is_file():
                    if transport is None:
                        raise RuntimeError('Generic resume requires owned runtime publication recovery')
                    local = output / f'recovered_runtime_{index}_{file_hash(previous_path)[:12]}'
                    local.mkdir(parents=True, exist_ok=False)
                    downloaded = download_case(transport, remote, local, 1, result_filename='generic_energy_results.json')
                    write_json(local / 'DOWNLOAD.json', downloaded)
                    if downloaded.get('result_collected') is not True:
                        raise RuntimeError('Generic resume runtime publication recovery failed')
                descriptor = {'repetition_index': index, 'command': argv, 'command_sha256': call['command_sha256'],
                    'output_dir': str(capture), 'raw_aggregate_path': str(raw), 'raw_aggregate_sha256': file_hash(raw),
                    'process_receipt': receipt, 'runtime_result_path': str(local / 'generic_energy_results.json'),
                    'runtime_remote_result_path': remote + '/generic_energy_results.json'}
            if descriptor['command'] != call['argv'] or canonical_hash(call['argv']) != call['command_sha256']:
                raise ValueError('Generic resume prior descriptor differs from actual prepared command')
            try:
                import_generic_energy_capture(descriptor, original['config'],
                    expected_setup_id=case['setup_id'], expected_campaign=original)
            except (ValueError, RuntimeError) as exc:
                rejected.append({'plan':str(previous_path),'repetition_index':index,'error':str(exc)})
                continue
            selected[index] = copy.deepcopy(descriptor)
            selected_calls[index] = copy.deepcopy(call)
    # A recovered runtime directory may already have created output; all capture
    # paths below remain fresh and source files retain their original bindings.
    output.mkdir(parents=True, exist_ok=True)
    resumed = copy.deepcopy(original)
    resumed['parent_run_dir'] = str(parent_run_dir) if parent_run_dir is not None else original.get('parent_run_dir')
    resumed['resume_previous_plans'] = [{'path':str(p),'sha256':file_hash(p)} for p in previous_plans]
    resumed['resume_reused_sources'] = [selected[i] for i in sorted(selected)]
    resumed['resume_rejected_sources'] = rejected
    for position,call in enumerate(resumed['calls']):
        index = call['repetition_index']
        if index in selected:
            resumed['calls'][position] = selected_calls[index]
        else:
            argv = call['argv']
            if not original.get('execution_environment_restored') and not call.get('original_command_files'):
                originals = {}
                for kind,flag in (('workload_command','--command-file'),
                                  ('preflight_command','--preflight-command-file')):
                    original_file = Path(call[kind+'_file'])
                    if file_hash(original_file) != call[kind+'_sha256']:
                        raise ValueError('Generic original command file changed')
                    originals[kind] = {'path':str(original_file),'sha256':call[kind+'_sha256']}
                    repaired = output/f'{kind}_{index}.txt'
                    repaired.write_text(_restore_energy_execution_environment(
                        original_file.read_text().strip(),original['config']['energy_capture']['contract']['execution_identity'])+'\n')
                    call[kind+'_file'] = str(repaired);call[kind+'_sha256'] = file_hash(repaired)
                    argv[argv.index(flag)+1] = str(repaired)
                call['original_command_files'] = originals
            argv[argv.index('--out') + 1] = str(output / f'capture_{index}')
            argv[argv.index('--run-id') + 1] += '-resume-' + output.parent.name
            call['command_sha256'] = canonical_hash(argv)
    write_json(output / 'generic_energy_plan.json', resumed)
    return resumed


def run_energy_case(plan_path: Path, case: dict, transport: SSHTransport, *, performance_admission=None,
                    process_registry=None, cancel_event=None, before_capture=None) -> dict:
    """Run prepared ordinary collector commands under the caller's ownership.

    This is the same transport/process/collector stack as normal measurements.
    A durable attempted capture is never implicitly repeated by reentry.
    """
    from ..native_progress import run_streaming
    from ..process_control import ProcessTreeRegistry
    from ..remote.process_lease import RemoteProcessLeaseJournal
    from ..runners.generic_energy import canonical_hash, file_hash, validate_config
    from ..generic_energy_reporting import import_generic_energy_capture
    from ..generic_completion_reporting import write_generic_energy_report
    RemoteProcessLeaseJournal.from_environment(required=True)
    if os.environ.get('ONNX_SPLITPOINT_CONTROLLER_RESOURCES') != 'required':
        raise RuntimeError('Generic energy requires the normal resource-owned capture parent')
    plan = read_json(plan_path); root = plan_path.parent
    requested_repetitions = _acceptance_repetitions(plan)
    config = plan['config']; validate_config(config)
    for call in plan['calls']:
        _verified_energy_resume_command_context(call,call,config)
    if case['identity'] != plan['identity']:
        raise ValueError('Generic energy case differs from prepared identity')
    for path_key in ('config', 'performance_result', 'collector_script', 'hardware_setups_file'):
        path = plan[path_key + '_path'] if path_key == 'config' else plan[path_key]
        if file_hash(path) != plan[path_key + '_sha256']:
            raise ValueError('Generic energy prepared source changed: ' + path_key)
    if sorted(c['repetition_index'] for c in plan['calls']) != list(range(1, requested_repetitions + 1)):
        raise ValueError('Generic energy prepared collector calls differ from acceptance recipe')
    if (root / 'capture_sources.json').exists() or any((root / f'capture_{i}_started.json').exists() for i in range(1, requested_repetitions + 1)):
        raise FileExistsError('Generic energy already attempted; preserve evidence and require explicit reviewed recovery')
    descriptors = list(plan.get('resume_reused_sources', []))
    from ..generic_energy_reporting import import_generic_energy_capture
    for descriptor in descriptors:
        call = plan['calls'][descriptor['repetition_index'] - 1]
        if descriptor['command'] != call['argv']:
            raise ValueError('Generic resumed source differs from actual retained command')
        import_generic_energy_capture(descriptor, config, expected_setup_id=case['setup_id'], expected_campaign=plan)
    reused = {d['repetition_index'] for d in descriptors}
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError('Generic energy cancelled before capture dispatch')
    check_energy_campaign(plan, allow_completed_repeats=bool(descriptors))
    if len(reused) < requested_repetitions:stage_energy_case(plan, transport)
    registry = process_registry if process_registry is not None else ProcessTreeRegistry()
    sample_loss_retry = None
    try:
        for call in plan['calls']:
            index = call['repetition_index']; argv = call['argv']
            if index in reused:continue
            if (cancel_event is not None and cancel_event.is_set()) or registry.cancelled:
                raise RuntimeError('Generic energy cancelled before next capture')
            if canonical_hash(argv) != call['command_sha256']:
                raise ValueError('Generic energy command changed')
            expected_budget = {'--campaign-budget-file': plan['campaign_budget_file'],
                '--campaign-row-id': plan['campaign_row_id'], '--campaign-repeats': str(requested_repetitions),
                '--campaign-max-retries': '2', '--campaign-max-transport-failures': '20',
                '--task-logical-repeat': f'repeat:{index - 1}', '--invalid-repeat-max-retries': '0'}
            if any(argv.count(k) != 1 or argv[argv.index(k)+1] != v for k, v in expected_budget.items()) or '--task-budget-file' in argv:
                raise ValueError('Generic energy shared campaign command changed')
            check_energy_campaign(plan, logical_repeat=f'repeat:{index - 1}')
            for kind in ('workload_command', 'preflight_command'):
                if file_hash(call[kind + '_file']) != call[kind + '_sha256']:
                    raise ValueError('Generic energy command file changed')
            capture_output = Path(argv[argv.index('--out') + 1])
            if capture_output.exists():
                raise FileExistsError('Generic energy collector output must be fresh')
            if before_capture is not None:
                before_capture(plan, call)
            receipt = {'argv': argv, 'command_sha256': call['command_sha256'], 'started_ns': time.time_ns(),
                       'aggregate_absent_before_execution': True, 'plan_sha256': file_hash(plan_path)}
            write_json(root / f'capture_{index}_started.json', receipt)
            with (root / f'capture_{index}.log').open('x') as stream:
                def line(text):
                    stream.write(text+'\n'); stream.flush(); print(text, flush=True)
                try:
                    result = run_streaming(argv, cwd=Path(plan['collector_script']).parent.parent,
                        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=900,
                        label=f'generic-completed-energy-{index}', line_callback=line, process_registry=registry,
                        cancel_event=cancel_event, should_cancel=lambda: registry.cancelled)
                    receipt['returncode'] = result.returncode
                finally:
                    receipt['finished_ns'] = time.time_ns()
                    write_json(root / f'capture_{index}_process_receipt.json', receipt)
            if receipt.get('returncode') != 0:
                # Exact single captures remain unchanged. Only the normal
                # budget can permit a clean, subsequent Parent resume.
                if registry.remaining_owned_count():registry.terminate_all(grace_s=3)
                if registry.remaining_owned_count() == 0:
                    from ..energy.task_budget import EnergyTaskBudget, _budget_parent
                    with Path(plan['campaign_budget_file']).open('r+') as handle:
                        budget = EnergyTaskBudget(handle, plan['campaign_limits'],
                            campaign_row=plan['campaign_row_id'], source_id=plan['campaign_source_id'],
                            environment=os.environ)
                        logical = f'repeat:{index - 1}'
                        if budget.extend_sample_loss_retries(logical):
                            policy = budget._reentry['sample_loss_retries']
                            sample_loss_retry = {'kind':'pure_marker_dropped_samples_nonzero',
                                'plan':{'path':str(plan_path.resolve()),'sha256':file_hash(plan_path)},
                                'campaign_budget_file':plan['campaign_budget_file'],
                                'campaign_row_id':plan['campaign_row_id'],'source_id':plan['campaign_source_id'],
                                'logical_repeat':logical,'chain':dict(budget.matching_chains(logical)[-1]),
                                'parent':_budget_parent(os.environ),
                                'authorization':{k:policy[k] for k in ('authorization_file','authorization_sha256')}}
                raise RuntimeError('Generic energy collector failed; no automatic retry')
            raw_path = capture_output / 'energy_aggregate.json'; raw = read_json(raw_path)
            if len(raw.get('runs') or []) != 1:
                raise RuntimeError('Generic energy single capture did not publish exactly one record')
            nonce = raw['runs'][0].get('preflight_evidence', {}).get('nonce')
            if not isinstance(nonce, str) or re.fullmatch(r'[0-9a-f]{64}', nonce) is None:
                raise ValueError('Generic energy collector nonce invalid')
            remote_dir = call['remote_result_directory_template'].replace('__ONNX_SPLITPOINT_PREFLIGHT_NONCE__', nonce)
            local_dir = root / f'runtime_result_{index}'; local_dir.mkdir()
            downloaded = download_case(transport, remote_dir, local_dir, 1, result_filename='generic_energy_results.json')
            write_json(root / f'capture_{index}_download.json', downloaded)
            if downloaded.get('result_collected') is not True:
                raise RuntimeError('Generic energy runtime publication download failed')
            descriptor = {'repetition_index': index, 'command': argv, 'command_sha256': call['command_sha256'],
                'output_dir': str(capture_output), 'raw_aggregate_path': str(raw_path), 'raw_aggregate_sha256': file_hash(raw_path),
                'process_receipt': receipt, 'runtime_result_path': str(local_dir / 'generic_energy_results.json'),
                'runtime_remote_result_path': remote_dir + '/generic_energy_results.json'}
            descriptors.append(descriptor)
            write_json(root / 'capture_sources.json', {'schema': 'onnx-splitpoint/generic-energy-capture-sources/v1',
                'plan_sha256': file_hash(plan_path), 'sources': sorted(descriptors, key=lambda d:d['repetition_index'])})
            # Preserve the exact source before a consumer can reject its report.
            # Explicit resume revalidates these sources through the normal reader;
            # this manifest records evidence and grants no scientific validity.
            imported = import_generic_energy_capture(descriptor, config, expected_setup_id=case['setup_id'], expected_campaign=plan)
            write_json(root / f'capture_{index}_normal_import.json', imported)
        manifest = root / 'capture_sources.json'
        write_json(manifest, {'schema': 'onnx-splitpoint/generic-energy-capture-sources/v1',
            'plan_sha256': file_hash(plan_path), 'sources': sorted(descriptors, key=lambda d:d['repetition_index'])})
        return write_generic_energy_report(plan_path, manifest, case, root / 'reports',
            performance_admission=performance_admission)
    finally:
        if registry.remaining_owned_count():
            registry.terminate_all(grace_s=3)
        cleanup = {'owned_local_remaining':registry.remaining_owned_count(),
            'remote_cleanup_owner':'normal enclosing energy parent'}
        if sample_loss_retry is not None:cleanup['sample_loss_retry'] = sample_loss_retry
        write_json(root / 'controller_cleanup.json', cleanup)
