"""Direct binding of controlled live normal CLI owners, without fake history.

No product workflow, hardware action or process signal is executed. A bounded
fixture script holds the real normal RunLock and exits on one stdin byte.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool.remote import predispatch_binding, resource_recovery
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.workflow.artifacts import ensure_run_layout, sha256_json
from onnx_splitpoint_tool.workflow.contracts import StageResult
from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata


SETUP = "fixture-deepx"
SESSION = "a" * 32


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _descriptor(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


@pytest.fixture
def direct_case(tmp_path, monkeypatch, request):
    mode = getattr(request, "param", "actual8")
    cwd = tmp_path / "source"
    cwd.mkdir()
    (cwd / "onnx_splitpoint_tool").mkdir()
    run = tmp_path / "runs" / ("direct-quality-" + mode)
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({"schema_version": 1, "hardware_setups": [{
        "id": SETUP, "host": {"address": "192.0.2.102", "user": "fixture", "port": 22}}]}))
    profile_path = tmp_path / "profile.yaml"
    profile = {"name": "direct fixture " + mode,
        "model_suite": {"primary": [{"id": "fixture", "task": "detection", "enabled": True}]},
        "hardware": {"selected_setups": [SETUP], "setups_file": str(registry_path)},
        "energy": {"enabled": False}, "native_producers": {"enabled": False},
        "run_profiles": [{"id": "deepx_to_trt", "enabled": True, "hardware_setup_id": SETUP,
            "quality_source_only": True, "variants": ["composed"], "case_ids": ["b003"]}]}
    profile_path.write_text(yaml.safe_dump(profile))
    targets = [{"setup_id": SETUP, "host": "192.0.2.102"}]
    snapshot_full = build_profile_start_snapshot(profile_request=str(profile_path),
        source_profile=profile, resolved_profile=profile, profile_id="direct-fixture",
        profile_path=str(profile_path), profile_source="file", runtime_bindings={"hardware_targets": targets})
    snapshot = public_start_snapshot_metadata(snapshot_full)
    plan = snapshot_full["effective_execution_plan"]
    options = {"only_model": "fixture", "energy_enabled": False, "native_producer_enabled": False,
        "stop_after": "generate_benchmark_set" if mode == "pregeneration" else None}
    contract = run_control.build_resume_contract(profile_payload=profile,
        effective_execution_plan=plan, options=options)
    locks_root = tmp_path / "locks/platform.lock"
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: locks_root)
    # A separate normal predecessor really acquires and releases its temporary
    # Run/resource locks before this original controller starts.
    prior_run = tmp_path / "prior-normal-quality"
    prior_session = "b" * 32
    prior_owner = {"run_id": prior_run.name, "session_id": prior_session,
        "profile_sha256": contract["profile_sha256"], "plan_sha256": contract["plan_sha256"]}
    prior_lock = run_control.EvaluationRunLock(out_root=prior_run.parent, run_dir=prior_run, owner=prior_owner)
    prior_lock.acquire()
    resource = "dut:192.0.2.102"
    physical = run_control.EvaluationRunLock.for_resource(resource, owner={**prior_owner, "purpose": "generic"})
    physical.acquire()
    physical.release()
    prior_lock.release()
    prior_manifest = prior_run / "run_manifest.json"
    _write(prior_manifest, {"schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": prior_run.name, "run_dir": str(prior_run), "status": "completed", "technical_status": "completed",
        "current_session_id": prior_session, "resume_contract": contract,
        "execution_sessions": [{"session_id": prior_session, "status": "completed",
            "start_origin": {"pid": os.getpid()}, **contract}]})
    script = cwd / "run_evaluation_workflow.py"
    script.write_text(r'''
import json,os,pathlib,sys,yaml
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.workflow.artifacts import sha256_json
c=json.loads(pathlib.Path(os.environ["DIRECT_BINDING_FIXTURE_CONFIG"]).read_text())
run=pathlib.Path(c["run"]);run.mkdir(parents=True)
run_control.platform_workflow_interlock_path=lambda:pathlib.Path(c["platform_lock"])
def write(name,value):
 p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
for name in ("profile.yaml","profile_source.yaml"):(run/name).write_text(yaml.safe_dump(c["profile"]))
write("effective_execution_plan.json",c["plan"])
write("profile_start_snapshot.json",c["snapshot"])
build={"schema":"onnx-splitpoint/package-build-identity","schema_version":1,"build_contract_version":2,
 "package_root":str(pathlib.Path.cwd()/"onnx_splitpoint_tool"),
 "package_content_sha256":sha256_json({"fixture_package":"1"*64}),
 "critical_code_digest_sha256":sha256_json({"fixture_critical":"2"*64})}
origin={"kind":"cli","entrypoint":"onnx_splitpoint_tool.workflow.run_evaluation",
 "profile_driven":True,"pid":os.getpid()}
session={"session_id":c["session"],"status":"running",**c["contract"],
 "start_origin":origin,"tool_build":build,"environment":{"tool_build":build}}
write("run_manifest.json",{"schema":"onnx-splitpoint/evaluation-run-manifest","schema_version":1,
 "run_id":run.name,"run_dir":str(run),"status":"running","technical_status":"running",
 "created_at":"2026-10-20T00:00:00+00:00","quality_decision":"not_evaluated",
 "current_session_id":c["session"],"execution_sessions":[session],"options":c["options"],
 "profile_path":c["profile_path"],"resume_contract":c["contract"],"profile_start_snapshot":c["snapshot"],
 "hardware_targets":c["targets"],"environment":{"tool_build":build},"current_tool_build":build,
 "current_environment":{"tool_build":build},"models":{"fixture":{"stages":{
 c["stage"]:{"state":"running","complete":False}}}}})
write("jobs/workflow_control.json",{"schema":"onnx-splitpoint/workflow-control","schema_version":1,
 "run_id":run.name,"session_id":c["session"],"state":"attached","cancel_requested":False})
(run/"models/fixture").mkdir(parents=True)
lock=run_control.EvaluationRunLock(out_root=run.parent,run_dir=run,
 owner={"session_id":c["session"],"profile_sha256":c["contract"]["profile_sha256"],
 "plan_sha256":c["contract"]["plan_sha256"]})
lock.acquire()
fields=pathlib.Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()
print(json.dumps({"pid":os.getpid(),"start_ticks":fields[19]}),flush=True)
sys.stdin.buffer.read(1)
lock.release(write_released_state=False)
''')
    argv = [sys.executable, "-B", str(script), "--profile", str(profile_path), "--out", str(run.parent),
        "--profile-driven", "--only-model", "fixture", "--json"]
    if mode == "pregeneration":
        argv += ["--stop-after", "generate_benchmark_set"]
    action_path = tmp_path / "action.json"
    action = {"argv": argv, "cwd": str(cwd), "profile": str(profile_path), "case_ids": ["b003"],
        "build_allowed": False, "FPS_measurement_requested": False}
    if mode == "actual8":
        action.update(scope="Generic split-only Quality b003; no Native, Energy or FPS", target="fixture-deepx normal Quality")
    else:
        action.pop("profile")  # This normal launcher binds Profile through exact --profile argv.
        action.update(scope={"model_id": "fixture", "case_ids": ["b003"], "Native": False,
            "energy": False, "FPS": False}, target={"future_setup_id": SETUP})
    _write(action_path, action)
    config = tmp_path / "fixture-config.json"
    _write(config, {"run": str(run), "platform_lock": str(locks_root), "profile": profile,
        "profile_path": str(profile_path), "snapshot": snapshot, "plan": plan, "targets": targets,
        "session": SESSION, "contract": contract, "options": options,
        "stage": "generate_benchmark_set" if mode == "pregeneration" else "analyze_model"})
    env = {**os.environ, "DIRECT_BINDING_FIXTURE_CONFIG": str(config),
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    line = child.stdout.readline()
    assert line, child.stderr.read()
    identity = json.loads(line)

    def finish():
        if child.poll() is None:
            child.stdin.write("x")
            child.stdin.flush()
        child.wait(timeout=10)
        assert child.returncode == 0, child.stderr.read()

    run_lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run, owner={})
    case = SimpleNamespace(run=run, session=SESSION, setup=SETUP, mode=mode, child=child,
        identity=identity, finish=finish, action_path=action_path, profile_path=profile_path,
        registry_path=registry_path, binding_path=tmp_path / "new-evidence/live-binding.json",
        run_lock=run_lock, physical=physical, physical_before=physical.lock_path.read_bytes(),
        predecessor={"owner": _descriptor(physical.lock_path), "run_manifest": _descriptor(prior_manifest),
            "run_lock": _descriptor(prior_lock.lock_path)})
    case.observe_kwargs = dict(run_dir=run, session_id=SESSION, setup_id=SETUP,
        action_path=action_path, evidence_output=case.binding_path,
        expected_pid=identity["pid"], expected_start_ticks=identity["start_ticks"], registry_path=registry_path)
    case.originals = {path: path.read_bytes() for path in [
        *(path for path in run.rglob("*") if path.is_file()), action_path, profile_path,
        registry_path, script, run_lock.lock_path, physical.lock_path, prior_manifest, prior_lock.lock_path]}
    try:
        yield case
    finally:
        finish()


def _observe(case):
    return predispatch_binding.observe_generic_predispatch_run_binding(**case.observe_kwargs)


def test_live_binding_accepts_normal_writer_prefixed_package_source(direct_case):
    case = direct_case
    manifest_path = case.run / "run_manifest.json"
    before = manifest_path.read_bytes()
    build = json.loads(before)["current_tool_build"]
    assert build["build_contract_version"] == 2
    assert build["package_content_sha256"] == sha256_json({"fixture_package": "1" * 64})
    assert build["critical_code_digest_sha256"] == sha256_json({"fixture_critical": "2" * 64})
    assert all(build[key].startswith("sha256:") for key in
               ("package_content_sha256", "critical_code_digest_sha256"))
    assert _observe(case)["schema"] == predispatch_binding.SCHEMA
    assert manifest_path.read_bytes() == before


@pytest.mark.parametrize("direct_case", ["actual8", "pregeneration"], indirect=True)
def test_direct_live_cli_binding_uses_real_stable_owner_without_historical_logger(direct_case):
    case = direct_case
    report = _observe(case)
    assert report["schema"] == "onnx-splitpoint/generic-predispatch-live-binding-v1"
    assert report["samples"][0] == report["samples"][1]
    assert report["process"]["pid"] == case.identity["pid"]
    assert report["process"]["start_ticks"] == case.identity["start_ticks"]
    assert report["started_path"] is None and report["started_record_status"] == "not_supplied"
    assert report["historical_launcher_reconstructed"] is report["historical_logger_known"] is False
    assert report["cleanup_proven"] is report["controller_exit_proven"] is False
    assert all(path.read_bytes() == raw for path, raw in case.originals.items())
    assert not list(case.run.rglob("*benchmark_plan*"))
    case.finish()

    def no_fresh_process_sample(*_args, **_kwargs):
        pytest.fail("immutable binding load must not reconstruct fresh historical PID observations")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(predispatch_binding, "_sample", no_fresh_process_sample)
        patch.setattr(predispatch_binding, "_identity", no_fresh_process_sample)
        case.run_lock.acquire_for_recovery()
        try:
            context = predispatch_binding.load_generic_predispatch_live_binding(
                run_dir=case.run, session_id=case.session, setup_id=case.setup,
                evidence={"live_binding": _descriptor(case.binding_path)}, snapshots={},
                run_lock=case.run_lock, registry_path=case.registry_path)
        finally:
            case.run_lock.release(write_released_state=False)
    assert context["process"] == report["process"]
    assert context["source_profile"] == yaml.safe_load(case.profile_path.read_text())
    assert all(path.read_bytes() == raw for path, raw in case.originals.items())


def test_live_observation_requires_two_equal_real_owner_samples(direct_case, monkeypatch):
    case = direct_case
    sample = predispatch_binding._sample
    calls = []

    def unstable(*args, **kwargs):
        observed = sample(*args, **kwargs)
        calls.append(True)
        if len(calls) == 2:
            observed = copy.deepcopy(observed)
            observed["owner_sha256"] = "f" * 64
        return observed

    monkeypatch.setattr(predispatch_binding, "_sample", unstable)
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
        _observe(case)
    assert len(calls) == 2
    assert not case.binding_path.exists()
    assert all(path.read_bytes() == raw for path, raw in case.originals.items())


@pytest.mark.parametrize("fault", ["sample", "typed_scope", "source_snapshot"])
def test_immutable_live_loader_rejects_tampered_binding_or_original_source(direct_case, fault):
    case = direct_case
    report = _observe(case)
    case.finish()
    if fault == "sample":
        report["samples"][1]["cwd"] += "-foreign"
        _write(case.binding_path, report)
    elif fault == "typed_scope":
        report["scope"]["case_ids"] = ["b999"]
        _write(case.binding_path, report)
    else:
        path = case.run / "profile_start_snapshot.json"
        value = json.loads(path.read_text())
        value["requested_selection"]["profile_name"] = "changed-after-observation"
        _write(path, value)
    before = {path: path.read_bytes() for path in (case.run / "run_manifest.json", case.run_lock.lock_path, case.physical.lock_path)}
    case.run_lock.acquire_for_recovery()
    try:
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            predispatch_binding.load_generic_predispatch_live_binding(
                run_dir=case.run, session_id=case.session, setup_id=case.setup,
                evidence={"live_binding": _descriptor(case.binding_path)}, snapshots={},
                run_lock=case.run_lock, registry_path=case.registry_path)
    finally:
        case.run_lock.release(write_released_state=False)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def _direct_recovery_input(case, monkeypatch, *, finish=True):
    from onnx_splitpoint_tool.remote import resource_recovery_observation
    _observe(case)
    if finish:
        case.finish()
    end_path = case.binding_path.parent / "lifecycle-end.json"
    identity = {"run_id": case.run.name, "run_dir": str(case.run),
        "session_id": case.session, "setup_id": case.setup}
    _write(end_path, {"schema": "onnx-splitpoint/generic-predispatch-lifecycle-end-v1",
        "schema_version": 1, **identity, "observed_at": datetime.now(timezone.utc).isoformat(),
        "live_binding": _descriptor(case.binding_path),
        "original_process_identity": copy.deepcopy(case.identity),
        "process_tree_report": {"root_pid": case.identity["pid"], "captured_process_count": 1,
            "captured_process_group_count": 1, "remaining_process_count": 0,
            "nested_session_cancellation_safe": True}})
    input_path = case.binding_path.parent / "recovery-input.json"
    _write(input_path, {"schema": "onnx-splitpoint/generic-predispatch-run-only-recovery-input",
        "schema_version": 1, **identity, "live_binding": _descriptor(case.binding_path),
        "lifecycle_end": _descriptor(end_path), "registry": _descriptor(case.registry_path),
        "released_physical_predecessor": case.predecessor})
    auth_path = case.binding_path.parent / "authorization.json"
    _write(auth_path, {"schema": "onnx-splitpoint/generic-predispatch-run-only-recovery-authorization",
        "schema_version": 1, **identity, "authorized": True, "action": "recover_generic_predispatch_run_only",
        "operator": "fixture-operator", "source": "explicit temporary fixture authorization",
        "authorized_at": datetime.now(timezone.utc).isoformat(), "evidence": _descriptor(input_path)})
    output = case.binding_path.parent / "receipt.json"

    def controlled_process_observation(owners, **_kwargs):
        recorded = []
        for owner in owners:
            path = Path("/proc") / str(owner["pid"]) / "stat"
            if not path.exists():
                actual, state = None, "absent"
            else:
                fields = path.read_text().rsplit(") ", 1)[1].split()
                actual = fields[19]
                state = "pid_reused" if actual != str(owner["start_time_ticks"]) else "zombie" if fields[0] == "Z" else "active"
            recorded.append({**owner, "observed_start_time_ticks": actual, "state": state})
        busy = any(row["state"] == "active" for row in recorded)
        return {"ok": not busy, "read_only": True, "status": "busy" if busy else "no_known_competing_work_seen",
            "recorded_owners": recorded, "processes": [], "live_product_leases": [], "visibility_errors": [],
            "cleanup_proven": False, "exclusive_start_permission": False}

    class Transport:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_read_only(self, *_args, **_kwargs):
            remote = {"ok": True, "read_only": True, "status": "no_known_competing_work_seen",
                "recorded_owners": [], "processes": [], "live_product_leases": [], "visibility_errors": []}
            return 0, "RESOURCE_OWNERS=" + json.dumps(remote)

    monkeypatch.setattr(resource_recovery_observation, "_observe_processes", controlled_process_observation)
    monkeypatch.setattr(resource_recovery_observation, "SSHTransport", Transport)
    case.end_path, case.input_path, case.auth_path, case.receipt_path = end_path, input_path, auth_path, output
    case.recover_kwargs = dict(run_dir=case.run, session_id=case.session, setup_id=case.setup,
        evidence_path=input_path, authorization_path=auth_path, evidence_output=output,
        registry_path=case.registry_path)
    return case.recover_kwargs


def test_normal_empty_startup_report_tree_allows_direct_binding_and_recovery(direct_case, monkeypatch):
    case = direct_case
    ensure_run_layout(case.run)
    assert {path.name for path in (case.run / "reports").iterdir()} == {"tables", "figures"}
    kwargs = _direct_recovery_input(case, monkeypatch)
    assert resource_recovery.recover_generic_predispatch_run_only(**kwargs)["status"] == "committed"
    assert all(not list((case.run / "reports" / name).iterdir()) for name in ("tables", "figures"))
    assert case.physical.lock_path.read_bytes() == case.physical_before


def test_startup_report_directory_with_an_output_file_blocks_direct_binding(direct_case):
    case = direct_case
    ensure_run_layout(case.run)
    output = case.run / "reports" / "tables" / "results.csv"
    output.write_text("case_id,latency_ms\nb003,1.0\n")
    before = {path: path.read_bytes() for path in
              (case.run / "run_manifest.json", case.run_lock.lock_path, case.physical.lock_path, output)}
    with pytest.raises(run_control.WorkflowRunCleanupQuarantineError, match="report output or unknown entry"):
        _observe(case)
    assert not case.binding_path.exists()
    assert all(path.read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize("fault", [None, "hardware_action", "foreign_model", "unstarted_stage"])
def test_normal_read_only_cache_preflight_is_preparation_bound_to_original_source(direct_case, monkeypatch, fault):
    case = direct_case
    ensure_run_layout(case.run)
    manifest_path = case.run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    # Actual _run_stage_steps publishes this file before entering fn().
    # Its root_stages manifest row is not recorded until fn() returns.
    stage = StageResult(stage="artifact_cache_preflight", model_id=None, status="planned",
        state="running", complete=False, started_at=manifest["created_at"], finished_at="").to_dict()
    if fault == "unstarted_stage":
        stage["started_at"] = ""
    _write(case.run / "stages/artifact_cache_preflight/stage_result.json", stage)
    _write(manifest_path, manifest)
    probe = {"schema": "onnx-splitpoint/remote-trt-cache-preflight", "schema_version": 1,
        "model_id": "foreign" if fault == "foreign_model" else "fixture", "setup_id": case.setup,
        "suite_dir": str(case.run / "models/fixture/benchmark_set/legacy_suite"),
        "read_only": True, "hardware_action_performed": fault == "hardware_action",
        "status": "ok", "requirements": [], "observations": []}
    path = case.run / "reports/remote_trt_cache_preflight.json"
    _write(path, {"schema": "onnx-splitpoint/remote-trt-cache-preflight-set", "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(), "read_only": True,
        "hardware_action_performed": False, "probe_count": 1, "probes": [probe]})
    before = path.read_bytes()
    if fault is not None:
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            _observe(case)
        assert not case.binding_path.exists()
    else:
        kwargs = _direct_recovery_input(case, monkeypatch)
        assert resource_recovery.recover_generic_predispatch_run_only(**kwargs)["status"] == "committed"
        assert case.physical.lock_path.read_bytes() == case.physical_before
    assert path.read_bytes() == before


@pytest.mark.parametrize("fault", [None, "case_conflict", "fps_conflict"])
def test_normal_typed_scope_case_selection_requires_exact_profile_and_unambiguous_action(direct_case, monkeypatch, fault):
    case = direct_case
    action = json.loads(case.action_path.read_text())
    action["scope"] = {"model_id": "fixture", "case_ids": ["b003"], "FPS": False}
    action.pop("FPS_measurement_requested")
    if fault == "fps_conflict":
        action["FPS_measurement_requested"] = True
    if fault == "case_conflict":
        action["case_ids"] = ["b999"]
    else:
        action.pop("case_ids")
    _write(case.action_path, action)
    if fault is not None:
        with pytest.raises(run_control.WorkflowRunCleanupQuarantineError):
            _observe(case)
        assert not case.binding_path.exists()
    else:
        kwargs = _direct_recovery_input(case, monkeypatch)
        assert resource_recovery.recover_generic_predispatch_run_only(**kwargs)["status"] == "committed"


def _refresh_direct_input(case):
    value = json.loads(case.input_path.read_text())
    value["lifecycle_end"] = _descriptor(case.end_path)
    _write(case.input_path, value)
    auth = json.loads(case.auth_path.read_text())
    auth["evidence"] = _descriptor(case.input_path)
    _write(case.auth_path, auth)


@pytest.mark.parametrize("direct_case", ["actual8", "pregeneration"], indirect=True)
def test_direct_binding_and_genuine_fixture_end_close_only_original_management_lifecycle(direct_case, monkeypatch):
    case = direct_case
    kwargs = _direct_recovery_input(case, monkeypatch)
    before = json.loads((case.run / "run_manifest.json").read_text())
    assert resource_recovery.recover_generic_predispatch_run_only(**kwargs)["status"] == "committed"
    after = json.loads((case.run / "run_manifest.json").read_text())
    assert after["status"] == after["technical_status"] == "cancelled"
    assert after["models"] == before["models"]
    assert after["created_at"] == before["created_at"]
    assert after["quality_decision"] == before["quality_decision"]
    assert after["execution_sessions"][-1]["start_origin"] == before["execution_sessions"][-1]["start_origin"]
    mutable = {case.run / "run_manifest.json", case.run_lock.lock_path}
    assert all(path.read_bytes() == raw for path, raw in case.originals.items() if path not in mutable)
    report = json.loads(case.binding_path.read_text())
    assert report["historical_launcher_reconstructed"] is report["historical_logger_known"] is False
    assert not (case.run / "initial_lineage.json").exists()
    replay_paths = [case.receipt_path, case.run / "run_manifest.json", case.run_lock.lock_path,
        case.physical.lock_path, case.binding_path, case.end_path, case.input_path, case.auth_path]
    replay_before = {path: path.read_bytes() for path in replay_paths}

    def no_reobservation(*_args, **_kwargs):
        pytest.fail("exact committed direct recovery replay must use the original immutable observation")

    monkeypatch.setattr(predispatch_binding, "_sample", no_reobservation)
    monkeypatch.setattr(predispatch_binding, "_identity", no_reobservation)
    monkeypatch.setattr(resource_recovery, "observe_resource_owners", no_reobservation)
    assert resource_recovery.recover_generic_predispatch_run_only(**kwargs)["status"] == "already_committed"
    assert all(path.read_bytes() == raw for path, raw in replay_before.items())


@pytest.mark.parametrize("fault", ["still_active", "end_schema", "wrong_original_pid", "remaining_process", "wrong_binding"])
def test_direct_recovery_never_substitutes_a_purported_end_for_exact_fresh_quiescence(direct_case, monkeypatch, fault):
    case = direct_case
    kwargs = _direct_recovery_input(case, monkeypatch, finish=fault != "still_active")
    if fault != "still_active":
        end = json.loads(case.end_path.read_text())
        if fault == "end_schema":
            end["schema"] = "operator-cleanup-claim"
        elif fault == "wrong_original_pid":
            end["original_process_identity"]["pid"] += 1
        elif fault == "remaining_process":
            end["process_tree_report"]["remaining_process_count"] = 1
        else:
            end["live_binding"]["sha256"] = "f" * 64
        _write(case.end_path, end)
        _refresh_direct_input(case)
    before = {path: path.read_bytes() for path in (case.run / "run_manifest.json", case.run_lock.lock_path, case.physical.lock_path)}
    with pytest.raises((run_control.WorkflowRunCleanupQuarantineError, run_control.WorkflowRunLockedError)):
        resource_recovery.recover_generic_predispatch_run_only(**kwargs)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not case.receipt_path.exists()
