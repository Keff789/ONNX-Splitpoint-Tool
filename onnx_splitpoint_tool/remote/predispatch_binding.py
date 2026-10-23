"""Direct, read-only binding of an existing live normal Generic controller.

This records a new observation. It never reconstructs historical launcher or
logger records and does not establish termination, cleanup or resource release.
"""
from __future__ import annotations

import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import time
from typing import Any, Mapping

from ..workflow.run_control import EvaluationRunLock, WorkflowRunCleanupQuarantineError


SCHEMA = "onnx-splitpoint/generic-predispatch-live-binding-v1"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise WorkflowRunCleanupQuarantineError("generic_predispatch_live_binding: " + reason)


def _safe(path, *, directory=False) -> Path:
    _require(isinstance(path, (str, os.PathLike)), "control path is missing")
    path = Path(path).expanduser()
    _require(path.is_absolute() and ".." not in path.parts
             and not any(p.is_symlink() for p in (path, *path.parents)),
             "unsafe control path: " + str(path))
    _require(path.is_dir() if directory else path.is_file(), "missing control path: " + str(path))
    _require(directory or stat.S_ISREG(path.stat().st_mode), "nonregular control file: " + str(path))
    return path


def _descriptor(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _object(raw: bytes) -> dict[str, Any]:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "duplicate control key")
            value[key] = item
        return value
    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise WorkflowRunCleanupQuarantineError("generic_predispatch_live_binding: invalid JSON") from exc
    _require(isinstance(value, dict), "control JSON is not an object")
    return value


def _read(path, snapshots, descriptor=None, *, yaml_file=False, binary=False):
    path = _safe(path)
    raw = path.read_bytes()
    _require(path not in snapshots or snapshots[path] == raw, "control source changed: " + str(path))
    if descriptor is not None:
        _require(isinstance(descriptor, Mapping) and descriptor == _descriptor(path, raw),
                 "control descriptor changed: " + str(path))
    snapshots[path] = raw
    if binary:
        return raw
    if yaml_file:
        import yaml
        value = yaml.safe_load(raw)
        _require(isinstance(value, dict), "profile/registry is not a mapping")
        return value
    return _object(raw)


def _flag(argv, name):
    _require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv),
             "exact original CLI flag missing: " + name)
    return argv[argv.index(name) + 1]


def _identity(pid):
    raw = (Path("/proc") / str(pid) / "stat").read_text()
    _require(raw.startswith(str(pid) + " (") and ") " in raw, "invalid controller stat")
    fields = raw.rsplit(") ", 1)[1].split()
    _require(fields[0] != "Z" and fields[19].isdecimal(), "controller is not live")
    return {"pid": pid, "ppid": int(fields[1]), "pgid": int(fields[2]),
            "session": int(fields[3]), "start_ticks": fields[19]}


def _sample(pid, ticks, argv, cwd, lock_path, owner_bytes):
    _require(int(Path("/proc/self/stat").read_text().split("(", 1)[0]) == os.getpid(),
             "local proc PID namespace is unproven")
    process = _identity(pid)
    _require(process["start_ticks"] == str(ticks), "controller PID was reused")
    root = Path("/proc") / str(pid)
    args = (root / "cmdline").read_bytes().split(b"\0")
    if args and args[-1] == b"":
        args.pop()
    _require([arg.decode("utf-8") for arg in args] == argv
             and os.readlink(root / "cwd") == str(cwd), "live argv/cwd differs from immutable Action")
    owner = _object(owner_bytes)
    _require(owner.get("pid") == pid and owner.get("pgid") == process["pgid"],
             "normal RunLock does not bind the live controller")
    reparented = owner.get("ppid") != process["ppid"]
    if reparented:
        parent = owner.get("ppid")
        _require(process["ppid"] == 1 and type(parent) is int and parent > 1
                 and parent == process["pgid"], "normal RunLock parent changed without proof")
        try:
            (Path("/proc") / str(parent)).stat()
        except FileNotFoundError:
            pass
        else:
            _require(False, "recorded original parent is still present")
    metadata = _safe(lock_path).stat()
    key = (os.major(metadata.st_dev), os.minor(metadata.st_dev), metadata.st_ino)
    held = []
    for fd in (root / "fd").iterdir():
        try:
            if os.readlink(fd) != str(lock_path):
                continue
            opened = fd.stat()
            _require((opened.st_dev, opened.st_ino) == (metadata.st_dev, metadata.st_ino),
                     "held canonical RunLock inode differs")
            for line in (root / "fdinfo" / fd.name).read_text().splitlines():
                parts = line.split()
                if (len(parts) == 9 and parts[:1] == ["lock:"]
                        and parts[2:5] == ["FLOCK", "ADVISORY", "WRITE"]
                        and int(parts[5]) == pid):
                    device = parts[6].split(":")
                    if (len(device) == 3 and
                            (int(device[0], 16), int(device[1], 16), int(device[2])) == key):
                        held.append({"fd": int(fd.name), "device": list(key[:2]), "inode": key[2]})
        except FileNotFoundError:
            continue
    _require(len(held) == 1 and _identity(pid) == process
             and _safe(lock_path).read_bytes() == owner_bytes,
             "exact normal RunLock WRITE flock or stable owner is unproven")
    return {"process": process, "argv": list(argv), "cwd": str(cwd),
            "lock_path": str(lock_path), "held_run_lock": held[0],
            "owner_sha256": hashlib.sha256(owner_bytes).hexdigest(),
            "reparented_to_init": reparented, "recorded_ppid": owner.get("ppid"),
            "recorded_parent_absent": reparented}


def _context(*, run, session_id, setup_id, action_path, started_path, registry_path,
             snapshots, historical_sources=None):
    from ..workflow.run_control import build_resume_contract
    from ..workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
    from ..energy.config import default_registry_path
    from ..platform_power import host_config_from_setup
    from ..workflow.execution_binding import physical_dut_key
    from .resource_recovery import _generic_predispatch_no_work

    def read(path, *, yaml_file=False, binary=False):
        path = _safe(path)
        if historical_sources is not None and path in historical_sources:
            raw = historical_sources[path]
            _require(path not in snapshots or snapshots[path] == raw,
                     "committed historical source differs")
            snapshots[path] = raw
            return raw if binary else _object(raw)
        return _read(path, snapshots, yaml_file=yaml_file, binary=binary)

    action_path = _safe(action_path)
    action = read(action_path)
    argv = action.get("argv")
    typed_scope = action.get("scope") if isinstance(action.get("scope"), Mapping) else {}
    fps_requested = (action["FPS_measurement_requested"] if "FPS_measurement_requested" in action
                     else typed_scope.get("FPS"))
    _require(isinstance(argv, list) and bool(argv) and all(isinstance(a, str) for a in argv)
             and argv.count("--profile-driven") == 1
             and action.get("build_allowed") is False
             and fps_requested is False
             and ("FPS" not in typed_scope or typed_scope["FPS"] is False),
             "Action is not an explicit nonbuilding Generic Quality invocation")
    cwd = _safe(action.get("cwd"), directory=True)
    profile_path = _safe(_flag(argv, "--profile"))
    _require(Path(_flag(argv, "--out")).expanduser() == run.parent,
             "Action output root differs from the actual Run")
    model_id = _flag(argv, "--only-model")
    _require(re.fullmatch(r"[A-Za-z0-9_.-]+", model_id) and model_id not in {".", ".."},
             "exact model selection is invalid")
    scripts = [Path(arg) if Path(arg).is_absolute() else cwd / arg for arg in argv
               if Path(arg).name == "run_evaluation_workflow.py"]
    _require(len(scripts) == 1 and scripts[0].is_relative_to(cwd), "normal CLI source is outside Action cwd")
    script = _safe(scripts[0])
    read(script, binary=True)
    source_profile = read(profile_path, yaml_file=True)
    declared_profile = action.get("profile")
    if isinstance(declared_profile, Mapping):
        _require(declared_profile == _descriptor(profile_path, snapshots[profile_path]),
                 "Action original Profile descriptor differs")
    elif isinstance(declared_profile, str):
        _require(Path(declared_profile).expanduser() == profile_path, "Action original Profile path differs")
    manifest_path = run / "run_manifest.json"
    manifest = read(manifest_path)
    resolved = read(run / "profile.yaml", yaml_file=True)
    archived_source = read(run / "profile_source.yaml", yaml_file=True)
    _require(archived_source == source_profile, "archived OriginalProfile differs from Action Profile")
    plan = read(run / "effective_execution_plan.json")
    archived = read(run / "profile_start_snapshot.json")
    control = read(run / "jobs/workflow_control.json")
    lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    owner_bytes = read(lock.lock_path, binary=True)
    owner = _object(owner_bytes)
    _require(owner.get("schema") == "onnx-splitpoint/evaluation-run-lock"
             and type(owner.get("schema_version")) is int and owner["schema_version"] == 1
             and owner.get("state") == "running" and owner.get("run_id") == run.name
             and owner.get("run_dir") == str(run) and owner.get("session_id") == session_id
             and owner.get("physical_resource") is None
             and owner.get("hostname") == socket.gethostname()
             and type(owner.get("pid")) is int and owner["pid"] > 0
             and not any("recovery" in str(key) for key in owner), "actual original RunLock identity differs")
    _require(manifest.get("schema") == "onnx-splitpoint/evaluation-run-manifest"
             and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
             and manifest.get("run_id") == run.name and manifest.get("run_dir") == str(run)
             and manifest.get("status") == "running" and manifest.get("current_session_id") == session_id
             and isinstance(manifest.get("created_at"), str) and bool(manifest["created_at"])
             and control.get("schema") == "onnx-splitpoint/workflow-control"
             and control.get("run_id") == run.name and control.get("session_id") == session_id,
             "normal Creation/current-session/control binding differs")
    sessions = manifest.get("execution_sessions")
    terminal = {"ok", "partial", "failed", "cancelled", "completed", "skipped"}
    _require(isinstance(sessions, list) and bool(sessions)
             and all(isinstance(row, Mapping) and isinstance(row.get("session_id"), str)
                     and re.fullmatch(r"[0-9a-f]{32}", row["session_id"]) for row in sessions)
             and len({row["session_id"] for row in sessions}) == len(sessions)
             and all(row.get("status") in terminal for row in sessions[:-1])
             and sessions[-1].get("session_id") == session_id and sessions[-1].get("status") == "running"
             and sessions[-1].get("start_origin", {}).get("pid") == owner["pid"],
             "original normal execution session does not bind the actual controller")
    session = sessions[-1]
    options = manifest.get("options")
    _require(isinstance(options, Mapping), "normal archived options missing")
    contract = build_resume_contract(profile_payload=resolved, effective_execution_plan=plan, options=options)
    _require(contract == manifest.get("resume_contract")
             and all(session.get(key) == contract[key] for key in
                     ("profile_sha256", "plan_sha256", "resume_contract_sha256"))
             and all(owner.get(key) == contract[key] for key in ("profile_sha256", "plan_sha256")),
             "normal Resume/source contract differs")
    _require(type(archived.get("schema_version")) is int and archived["schema_version"] in {1, 2},
             "normal start snapshot version is unsupported")
    rebuilt = build_profile_start_snapshot(
        **{key: str(archived.get(key) or "") for key in
           ("profile_request", "profile_id", "profile_path", "profile_source")},
        source_profile=archived_source, resolved_profile=resolved,
        runtime_bindings=archived.get("runtime_bindings") or {}, schema_version=archived["schema_version"])
    _require(archived == public_start_snapshot_metadata(rebuilt) == manifest.get("profile_start_snapshot")
             and archived.get("profile_path") == str(profile_path), "normal Creation start snapshot differs")
    cases = action["case_ids"] if "case_ids" in action else typed_scope.get("case_ids")
    _require(isinstance(cases, list) and bool(cases)
             and all(isinstance(case, str) and re.fullmatch(r"b[0-9]+", case) for case in cases)
             and len(set(cases)) == len(cases)
             and ("case_ids" not in typed_scope or typed_scope["case_ids"] == cases),
             "Action split-case selection is invalid or conflicting")
    for profile in (archived_source, resolved):
        runs = [row for row in profile.get("run_profiles", [])
                if isinstance(row, Mapping) and row.get("enabled", True)]
        _require((profile.get("hardware") or {}).get("selected_setups") == [setup_id]
                 and (profile.get("energy") or {}).get("enabled") is False
                 and (profile.get("native_producers") or {}).get("enabled") is False
                 and len(runs) == 1 and runs[0].get("quality_source_only") is True
                 and runs[0].get("hardware_setup_id") == setup_id and runs[0].get("case_ids") == cases,
                 "archived typed Profile is not the exact source-only single-setup Quality scope")
    _require(options.get("energy_enabled") is False and options.get("native_producer_enabled") is False
             and options.get("only_model") == model_id, "actual normal options differ from Action scope")
    current_build = manifest.get("current_tool_build")
    creation_build = manifest.get("tool_build") or (manifest.get("environment") or {}).get("tool_build")
    for build in (creation_build, current_build, session.get("tool_build")):
        _require(isinstance(build, Mapping) and build.get("schema") == "onnx-splitpoint/package-build-identity"
                 and type(build.get("schema_version")) is int and build["schema_version"] == 1
                 and isinstance(build.get("package_root"), str)
                 and all(isinstance(build.get(key), str)
                         and re.fullmatch(r"[0-9a-f]{64}", build[key].removeprefix("sha256:"))
                         for key in ("package_content_sha256", "critical_code_digest_sha256")),
                 "archived PackageSource identity is incomplete")
    _require(current_build == session["tool_build"]
             and (session.get("environment") or {}).get("tool_build") == current_build
             and (manifest.get("current_environment") or {}).get("tool_build") == current_build
             and Path(current_build["package_root"]) == cwd / "onnx_splitpoint_tool",
             "current/session PackageSource does not bind Action cwd")
    registry_file = _safe(registry_path or default_registry_path())
    registry = read(registry_file, yaml_file=True)
    setups = [row for row in registry.get("hardware_setups", [])
              if isinstance(row, Mapping) and row.get("id") == setup_id]
    _require(len(setups) == 1, "actual registered setup is missing or ambiguous")
    setup = setups[0]
    host = host_config_from_setup(setup)
    fields = setup.get("host") if isinstance(setup.get("host"), Mapping) else {}
    resource = physical_dut_key({"remote": {"host": host.host,
        "physical_host_id": setup.get("physical_host_id") or fields.get("physical_host_id")}})
    started = None
    if started_path is not None:
        started_path = _safe(started_path)
        started = read(started_path)
        _require(started.get("action_spec") == str(action_path) and started.get("argv") == argv,
                 "existing started record differs from Action")
    inventory_sources = {}
    inventory = [[name, list(entries)] for name, entries in
                 _generic_predispatch_no_work(run, model_id, manifest, inventory_sources)]
    for path, raw in inventory_sources.items():
        _require(path not in snapshots or snapshots[path] == raw,
                 "pre-dispatch journal changed from its frozen source")
        snapshots[path] = raw
    return {"action": action, "action_path": str(action_path), "cwd": str(cwd),
        "source_profile": archived_source, "resolved_profile": resolved, "manifest": manifest,
        "session": session, "run_owner": owner, "owner_bytes": owner_bytes, "run_lock_path": str(lock.lock_path),
        "scope": {"model_id": model_id, "case_ids": list(cases), "Native": False,
                  "Full": False, "energy": False, "FPS": False},
        "target": {"setup_id": setup_id, "host": host.host, "user": host.user,
                   "port": int(host.port), "physical_resource": resource},
        "registry": registry, "registry_path": str(registry_file), "resume_contract": contract,
        "predispatch_journal_inventory": inventory,
        "started": started, "started_path": str(started_path) if started_path is not None else None,
        "package_source": {"cli_script": _descriptor(script, snapshots[script]),
            "creation_tool_build": copy.deepcopy(creation_build), "current_tool_build": copy.deepcopy(current_build),
            "session_tool_build": copy.deepcopy(session["tool_build"]), "historical_source_closure_rebuilt": False}}


def _unchanged(snapshots):
    _require(all(_safe(path).read_bytes() == raw for path, raw in snapshots.items()),
             "frozen source bytes changed during binding")


def observe_generic_predispatch_run_binding(*, run_dir, session_id, setup_id, action_path,
        evidence_output, expected_pid, expected_start_ticks, started_path=None, registry_path=None):
    """Capture two stable live samples and write only a separate new report."""
    run = _safe(run_dir, directory=True)
    _require(isinstance(session_id, str) and re.fullmatch(r"[0-9a-f]{32}", session_id)
             and isinstance(setup_id, str) and bool(setup_id.strip())
             and type(expected_pid) is int and expected_pid > 1
             and not isinstance(expected_start_ticks, bool)
             and str(expected_start_ticks).isdecimal() and int(expected_start_ticks) > 0,
             "exact original run/session/setup/PID/start identity is required")
    snapshots = {}
    action = _read(action_path, snapshots)
    cwd = _safe(action.get("cwd"), directory=True)
    lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    owner_bytes = _read(lock.lock_path, snapshots, binary=True)
    first = _sample(expected_pid, expected_start_ticks, action.get("argv"), cwd, lock.lock_path, owner_bytes)
    context = _context(run=run, session_id=session_id, setup_id=setup_id, action_path=action_path,
        started_path=started_path, registry_path=registry_path, snapshots=snapshots)
    second = _sample(expected_pid, expected_start_ticks, context["action"]["argv"], cwd,
                     lock.lock_path, owner_bytes)
    _require(first == second and context["run_owner"]["pid"] == expected_pid,
             "the two live controller/held-lock samples are not stable")
    _unchanged(snapshots)
    output = Path(evidence_output).expanduser()
    _require(output.is_absolute() and ".." not in output.parts
             and not any(p.is_symlink() for p in (output, *output.parents))
             and not output.is_relative_to(run) and not output.is_relative_to(cwd)
             and output not in snapshots and not os.path.lexists(output),
             "observation output must be new and separate from runtime/source files")
    report = {"schema": SCHEMA, "schema_version": 1, "report_kind": "new_live_observation",
        "observed_at": datetime.now(timezone.utc).isoformat(), "run_id": run.name, "run_dir": str(run),
        "session_id": session_id, "setup_id": setup_id, "action_path": context["action_path"],
        "started_path": context["started_path"], "started_record_status": "observed_existing" if started_path is not None else "not_supplied",
        "registry_path": context["registry_path"], "process": first["process"], "samples": [first, second],
        "scope": context["scope"], "target": context["target"], "package_source": context["package_source"],
        "predispatch_journal_inventory": context["predispatch_journal_inventory"],
        "original_run_owner": context["run_owner"],
        "original_run_owner_base64": base64.b64encode(owner_bytes).decode("ascii"),
        "original_run_owner_sha256": hashlib.sha256(owner_bytes).hexdigest(),
        "sources": [_descriptor(path, raw) for path, raw in snapshots.items()],
        "read_only_runtime": True, "historical_launcher_reconstructed": False,
        "historical_logger_known": False, "cleanup_proven": False,
        "controller_exit_proven": False, "resource_release_proven": False,
        "exclusive_start_permission": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    _safe(output.parent, directory=True)
    raw = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{time.time_ns()}.binding.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw); handle.flush(); os.fsync(handle.fileno())
        _unchanged(snapshots)
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def load_generic_predispatch_live_binding(*, run_dir, session_id, setup_id, evidence,
        snapshots, run_lock, registry_path=None):
    """Verify immutable live binding and current held original owner, without /proc."""
    from ..workflow.run_control import (_generic_orphan_lock_bytes,
        validate_generic_predispatch_run_only_recovery_owner)

    run = _safe(run_dir, directory=True)
    descriptor = evidence.get("live_binding") if isinstance(evidence, Mapping) else None
    _require(isinstance(descriptor, Mapping), "exact live_binding descriptor is required")
    report = _read(descriptor.get("path"), snapshots, descriptor)
    _require(report.get("schema") == SCHEMA and type(report.get("schema_version")) is int
             and report["schema_version"] == 1 and report.get("report_kind") == "new_live_observation"
             and report.get("run_id") == run.name and report.get("run_dir") == str(run)
             and report.get("session_id") == session_id and report.get("setup_id") == setup_id
             and report.get("read_only_runtime") is True
             and report.get("started_record_status") == (
                 "not_supplied" if report.get("started_path") is None else "observed_existing")
             and all(report.get(key) is False for key in ("historical_launcher_reconstructed",
                "historical_logger_known", "cleanup_proven", "controller_exit_proven",
                "resource_release_proven", "exclusive_start_permission")), "live report scope or evidence kind differs")
    sources = report.get("sources")
    _require(isinstance(sources, list) and bool(sources), "immutable live source inventory missing")
    owner_bytes = base64.b64decode(report.get("original_run_owner_base64", ""), validate=True)
    current_owner_bytes = _generic_orphan_lock_bytes(run_lock)
    historical_sources = {}
    committed = None
    if current_owner_bytes != owner_bytes:
        committed = validate_generic_predispatch_run_only_recovery_owner(run_lock)
        _require(committed.get("run_id") == run.name and committed.get("run_dir") == str(run)
                 and committed.get("session_id") == session_id and committed.get("setup_id") == setup_id,
                 "committed receipt belongs to a foreign live binding")
        group = committed.get("group")
        _require(isinstance(group, list) and len(group) == 1
                 and group[0].get("lock_path") == str(run_lock.lock_path)
                 and base64.b64decode(group[0].get("original_owner_base64", ""), validate=True) == owner_bytes,
                 "committed receipt does not bind the observed original owner")
        historical_sources = {Path(run_lock.lock_path): owner_bytes,
            run / "run_manifest.json": base64.b64decode(
                committed.get("manifest_before_base64", ""), validate=True)}
    bound_sources = {}
    seen = set()
    for source in sources:
        _require(isinstance(source, Mapping) and isinstance(source.get("path"), str)
                 and bool(source["path"]) and source["path"] not in seen,
                 "invalid or duplicate immutable live source descriptor")
        seen.add(source["path"])
        path = _safe(source["path"])
        if path in historical_sources:
            raw = historical_sources[path]
            _require(source == _descriptor(path, raw), "committed original source descriptor differs")
            bound_sources[path] = raw
        else:
            _read(path, bound_sources, source, binary=True)
    _require(registry_path is None or str(Path(registry_path).expanduser()) == report.get("registry_path"),
             "requested normal registry differs from live observation")
    context = _context(run=run, session_id=session_id, setup_id=setup_id,
        action_path=report.get("action_path"), started_path=report.get("started_path"),
        registry_path=report.get("registry_path"), snapshots=bound_sources,
        historical_sources=historical_sources)
    _require({str(path) for path in bound_sources} == seen,
             "live report omitted a required immutable source descriptor")
    samples = report.get("samples")
    _require(isinstance(samples, list) and len(samples) == 2 and samples[0] == samples[1]
             and isinstance(samples[0], Mapping) and samples[0].get("process") == report.get("process")
             and samples[0].get("argv") == context["action"]["argv"]
             and samples[0].get("cwd") == context["cwd"]
             and samples[0].get("lock_path") == str(run_lock.lock_path)
             and samples[0].get("owner_sha256") == hashlib.sha256(owner_bytes).hexdigest()
             and context["owner_bytes"] == owner_bytes
             and (committed is not None or run_lock.previous_owner == context["run_owner"])
             and context["run_owner"] == report.get("original_run_owner")
             and report.get("original_run_owner_sha256") == hashlib.sha256(owner_bytes).hexdigest()
             and report.get("scope") == context["scope"] and report.get("target") == context["target"]
             and report.get("predispatch_journal_inventory") == context["predispatch_journal_inventory"]
             and report.get("package_source") == context["package_source"],
             "original live samples, held owner or immutable source bindings differ")
    process = report.get("process")
    _require(isinstance(process, Mapping) and process.get("pid") == context["run_owner"]["pid"]
             and str(process.get("start_ticks") or "").isdecimal()
             and int(process["start_ticks"]) > 0
             and process.get("pgid") == context["run_owner"].get("pgid")
             and all(type(process.get(key)) is int and process[key] > 0 for key in
                     ("pid", "ppid", "pgid", "session")), "observed live process identity is incomplete")
    held = samples[0].get("held_run_lock")
    metadata = os.fstat(run_lock._fh.fileno())
    _require(isinstance(held, Mapping) and held.get("device") == [os.major(metadata.st_dev), os.minor(metadata.st_dev)]
             and held.get("inode") == metadata.st_ino
             and samples[0].get("recorded_ppid") == context["run_owner"].get("ppid")
             and ((process["ppid"] == context["run_owner"].get("ppid")
                   and samples[0].get("reparented_to_init") is False
                   and samples[0].get("recorded_parent_absent") is False)
                  or (process["ppid"] == 1 and context["run_owner"].get("ppid") == process["pgid"]
                      and samples[0].get("reparented_to_init") is True
                      and samples[0].get("recorded_parent_absent") is True)),
             "observed held inode or original/reparented process association differs")
    _unchanged({path: raw for path, raw in bound_sources.items() if path not in historical_sources})
    if committed is not None:
        _require(validate_generic_predispatch_run_only_recovery_owner(run_lock) == committed,
                 "committed replay owner/lifecycle changed")
    else:
        _require(_generic_orphan_lock_bytes(run_lock) == owner_bytes,
                 "original held RunLock changed while loading")
    for path, raw in bound_sources.items():
        current = _safe(path).read_bytes() if path in historical_sources else raw
        _require(path not in snapshots or snapshots[path] == current,
                 "recovery source snapshot differs from current control bytes")
        snapshots[path] = current
    _unchanged(snapshots)
    return {**context, "report": report, "process": dict(process),
            "source_snapshots": bound_sources, "committed_receipt": committed}
