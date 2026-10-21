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
