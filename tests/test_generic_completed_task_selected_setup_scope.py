"""FPS resource admission with real local run ownership and no hardware."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.workflow import generic_completed_task as entry, run_control


def target(host="192.0.2.8", *, alias="", setup_id="fixture"):
    return {"id": setup_id, "physical_host_id": alias,
            "remote": {"host": host, "user": "fixture", "port": 22}}


def idle():
    return {"ok": True, "read_only": True, "status": "no_known_competing_work_seen",
            "processes": [], "recorded_owners": [], "live_product_leases": [], "visibility_errors": []}


@pytest.fixture
def setup_workflow(tmp_path):
    """A live subprocess holds the actual product Run flock until stdin closes."""
    children = []

    def start(*, host="192.0.2.10", central=False, physical=False, alias="",
              targets=None, finalizing=False, smoke_mode="summary_only",
              require_remote_for_hailo=False, completed_setups=(), journal_setups=(), physical_setup=None, model_ids=("m",), completed_models=None):
        run = tmp_path / ("run-" + str(len(children)))
        config = {"run": str(run), "targets": targets or [target(host, alias=alias)],
                  "central": central, "physical": physical, "finalizing": finalizing,
                  "smoke_mode": smoke_mode,
                  "require_remote_for_hailo": require_remote_for_hailo, "completed_setups":list(completed_setups),
                  "journal_setups":list(journal_setups), "physical_setup":physical_setup, "model_ids":list(model_ids),
                  "completed_models":list(model_ids) if completed_models is None else list(completed_models)}
        source = r'''
import json,sys,pathlib,os,yaml
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.workflow.runner import ALL_STAGES
from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot,public_start_snapshot_metadata
c=json.loads(sys.argv[1]);run=pathlib.Path(c["run"]);run.mkdir()
run_control.platform_workflow_interlock_path=lambda: run.parent/"locks/platform.lock"
setups=[row["id"] for row in c["targets"]]
runs=[{"id":setup+"-quality","hardware_setup_id":setup,"quality_source_only":True} for setup in setups]
profile={"name":"fixture", "model_suite":{"primary":[{"id":m} for m in c["model_ids"]]},
 "energy":{"enabled":False},"native_producers":{"enabled":False},
 "hardware_smoke":{"mode":c["smoke_mode"],"require_remote_for_hailo":c["require_remote_for_hailo"]},
 "hardware":{"selected_setups":setups},"run_profiles":runs,
 "quality_gate":{"statistics":{"workers":6,"max_active_requests":2,"execution_location":"central_management"}}}
plan={"models":c["model_ids"],"logical_run_profiles":[row["id"] for row in runs],
 "setup_groups":{setup:[setup+"-quality"] for setup in setups}}
options={};contract=run_control.build_resume_contract(profile_payload=profile,effective_execution_plan=plan,options=options)
session="a"*32
snap=build_profile_start_snapshot(profile_request="fixture",source_profile=profile,resolved_profile=profile,
 profile_id="fixture",profile_path=str(run/"profile.yaml"),profile_source="file",runtime_bindings={"hardware_targets":c["targets"]})
def write(name,value):
 p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
for name in ("profile.yaml","profile_source.yaml"):(run/name).write_text(yaml.safe_dump(profile))
write("effective_execution_plan.json",plan)
write("profile_start_snapshot.json",public_start_snapshot_metadata(snap))
write("run_manifest.json",{"schema":"onnx-splitpoint/evaluation-run-manifest","run_id":run.name,"run_dir":str(run),
 "current_session_id":session,"options":options,"resume_contract":contract,"profile_start_snapshot":public_start_snapshot_metadata(snap),
 "hardware_targets":c["targets"],"status":"partial" if c["finalizing"] else "running",
 "execution_sessions":[{"session_id":session,"status":"partial" if c["finalizing"] else "running",**contract}],
 "models":{m:{"model_id":m,"stages":{
   "run_benchmarks":{"stage":"run_benchmarks","model_id":m,"status":"ok" if c["central"] else "running"},
   "hardware_smoke":{"stage":"hardware_smoke","model_id":m,"status":"skipped" if c["finalizing"] else "pending"}}} for m in c["model_ids"]},
 "workflow_stages":list(ALL_STAGES),
 "root_stages":{"evaluate_quality":{"stage":"evaluate_quality","model_id":None,
   "status":"skipped" if c["finalizing"] else "running" if c["central"] else "pending"}}})
write("jobs/workflow_control.json",{"schema":"onnx-splitpoint/workflow-control","run_id":run.name,"session_id":session})
(run/"jobs/remote_process_leases"/session).mkdir(parents=True)
lock=run_control.EvaluationRunLock(out_root=run.parent,run_dir=run,
 owner={"session_id":session,"profile_sha256":contract["profile_sha256"],"plan_sha256":contract["plan_sha256"]})
lock.acquire()
from datetime import datetime,timezone
from onnx_splitpoint_tool.workflow.execution_binding import physical_dut_key
from onnx_splitpoint_tool.benchmark.remote_run import init_local_run_artifacts,_finalize_run_status
from onnx_splitpoint_tool.remote.process_lease import RemoteProcessLeaseScope,RemoteProcessLeaseJournal
stamp=lambda:datetime.now(timezone.utc).isoformat()
for t in c["targets"]:
 setup=t["id"]
 if setup in c["completed_setups"]:
  for model in c["completed_models"]:
   physical=run_control.EvaluationRunLock.for_resource(physical_dut_key(t),owner={"run_id":run.name,"purpose":"generic"});physical.acquire()
   suite=pathlib.Path("models")/model/"suite";benchmark=suite/"benchmark_set.json"
   host={"id":setup,"label":setup,**t["remote"]}
   transport=run.name+"_"+model+"_"+setup
   args={"quality_evidence_eval_id":run.name,"quality_evidence_model_id":model,"quality_evidence_setup_id":setup,"energy_enabled":False,"add_args":"--run-id "+setup+"-quality"}
   gates={"hardware_target":t,"hardware_run_id":setup+"-quality","hardware_run_ids":[setup+"-quality"]}
   directory=run/"models"/model/"benchmark_results"
   created=stamp();started=stamp()
   local=directory/"remote_runs/Results/suite/1"/transport
   meta={"schema_version":1,"run_id":transport,"repeat_idx":"1","started_at":started,"created_at":started,"suite_dir":str(run/suite),"benchmark_set_json":str(run/benchmark),"host":host,"args":args,"resume":{"requested_run_id":transport,"active_run_id":transport}}
   init_local_run_artifacts(local,meta)
   _finalize_run_status(local,status="ok",started_at=started,ended_at=stamp(),remote_rc=0)
   physical.release()
   write(str(directory.relative_to(run)/("remote_benchmark_dispatch_"+setup+".json")),{"schema":"onnx-splitpoint/remote-benchmark-dispatch","schema_version":1,"created_at":created,"model_id":model,"hardware_target_id":setup,"status":"dispatching","host":host,"execution_gates":gates,"args":args,"run_id":transport,"suite_dir":str(suite),"benchmark_set_json":str(benchmark),"local_working_dir":str(directory/"remote_runs")})
   write(str(directory.relative_to(run)/("remote_benchmark_status_"+setup+".json")),{"schema":"onnx-splitpoint/remote-benchmark-status","schema_version":1,"created_at":stamp(),"model_id":model,"hardware_target_id":setup,"status":"ok","suite_dir":str(suite),"host":host,"execution_gates":gates,"local_run_dir":str(local)})
 if setup in c["journal_setups"]:
  scope=RemoteProcessLeaseScope(run.name,session)
  journal=RemoteProcessLeaseJournal(scope=scope,directory=run/"jobs/remote_process_leases"/session)
  journal.record_operation(scope.operation(label="fixture",control_argv_prefix=["ssh","-p",str(t["remote"].get("port",22)),t["remote"]["user"]+"@"+t["remote"]["host"]]))
other=None
if c["physical"]:
 t=next((t for t in c["targets"] if t["id"]==c["physical_setup"]),c["targets"][0])
 other=run_control.EvaluationRunLock.for_resource(physical_dut_key(t),owner={"run_id":run.name,"purpose":"generic"});other.acquire()
fields=pathlib.Path("/proc/self/stat").read_text().rsplit(") ",1)[1].split()
print(json.dumps({"pid":os.getpid(),"start_time_ticks":fields[19]}),flush=True)
sys.stdin.buffer.read(1)
if other:other.release()
lock.release()
'''
        child = subprocess.Popen([sys.executable, "-B", "-c", source, json.dumps(config)],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
        children.append(child)
        line = child.stdout.readline()
        assert line, child.stderr.read().decode()
        identity = json.loads(line)
        observed = {**idle(), "ok": False, "status": "busy",
                    "processes": [{**identity, "matched_programs": ["arbitrary-name-is-not-an-exemption"]}]}
        return run, observed

    yield start
    for child in children:
        if child.poll() is None:
            child.stdin.write(b"x")
            child.stdin.flush()
        child.wait(timeout=5)



def case_files(run):
    directory = run / "models/m/benchmark_results"
    status = directory / "remote_benchmark_status_dx.json"
    dispatch = directory / "remote_benchmark_dispatch_dx.json"
    local = directory / "remote_runs/Results/suite/1" / (run.name + "_m_dx")
    return status, dispatch, local


def select_dx(setup_workflow, **kwargs):
    return setup_workflow(targets=[target("192.0.2.8", setup_id="dx"),
                                   target("192.0.2.10", setup_id="h8")], **kwargs)


def result(observed):
    return entry._scoped_controller_owners({"fps": target()}, observed)


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    path.write_text(json.dumps(value))


def test_selected_setup_complete_allows_other_setup_live_lock_and_journal(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"],
        journal_setups=["h8"], physical=True, physical_setup="h8")
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    actual = result(observed)
    assert actual["ok"], actual
    owner = actual["normal_quality_owners"][0]
    assert owner["completed_selected_setups"] == ["dx"]
    assert owner["central_only"] is False
    assert owner["resources"] == ["dut:192.0.2.10", "dut:192.0.2.8"]
    assert actual["cleanup_proven"] is actual["exclusive_start_permission"] is False


def measurement_lock(run, monkeypatch, *, resource="dut:192.0.2.8"):
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    return run_control.EvaluationRunLock.for_resource(resource,
        owner={"run_id": "fps-source", "session_id": "fps-execution",
               "purpose": "generic_completed_task_192"})


def handoff_result(observed, lock, *, identity=("fps-source", "fps-execution")):
    return entry._scoped_controller_owners({"fps": target()}, observed,
        current_physical_locks={"fps": lock}, measurement_identity=identity)


def test_direct_released_setup_handoff_under_actual_new_flock(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"], journal_setups=["h8"])
    with measurement_lock(run, monkeypatch) as lock:
        assert lock.previous_owner["run_id"] == run.name and lock.previous_owner["state"] == "released"
        assert json.loads(lock.lock_path.read_text())["state"] == "running"
        assert not result(observed)["ok"]  # The old strict observer remains strict.
        actual = handoff_result(observed, lock)
        assert actual["ok"], actual
        proof = actual["normal_quality_owners"][0]["direct_physical_handoffs"]["dut:192.0.2.8"]
        assert proof["previous_owner"] == lock.previous_owner
        assert proof["current_owner"] == lock.owner
        assert proof["inode"][2] == os.fstat(lock._fh.fileno()).st_ino


@pytest.mark.parametrize("fault", ["not_acquired", "closed", "unlocked", "wrong_path", "replaced_inode",
    "wrong_resource", "wrong_run", "wrong_session", "wrong_pid", "quarantine"])
def test_invalid_current_handoff_lock_denies(setup_workflow, monkeypatch, fault):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"])
    resource = "dut:192.0.2.99" if fault == "wrong_resource" else "dut:192.0.2.8"
    lock = measurement_lock(run, monkeypatch, resource=resource)
    if fault == "not_acquired":
        assert not handoff_result(observed, lock)["ok"]
        return
    with lock:
        identity = ("wrong", "fps-execution") if fault == "wrong_run" else ("fps-source", "wrong") if fault == "wrong_session" else ("fps-source", "fps-execution")
        path, renamed = lock.lock_path, None
        try:
            if fault == "closed": lock.release()
            elif fault == "unlocked":
                import fcntl
                fcntl.flock(lock._fh.fileno(), fcntl.LOCK_UN)
            elif fault == "wrong_path": lock.lock_path = path.with_name("wrong.lock")
            elif fault == "replaced_inode":
                renamed = path.with_suffix(".held")
                path.rename(renamed)
                path.write_text(json.dumps(lock.owner))
            elif fault == "wrong_pid":
                lock.owner["pid"] += 1
                path.write_text(json.dumps(lock.owner))
            elif fault == "quarantine": lock.quarantine_path.write_text("{}")
            actual = handoff_result(observed, lock, identity=identity)
            assert not actual["ok"], (fault, actual)
        finally:
            lock.lock_path = path
            if renamed is not None:
                path.unlink()
                renamed.rename(path)


@pytest.mark.parametrize("field,value", [("state", "running"), ("state", "quarantined"),
    ("run_id", "other"), ("pid", 1), ("ppid", 1), ("pgid", 1), ("hostname", "other"),
    ("session_id", "wrong"), ("physical_resource", "dut:192.0.2.99"),
    ("released_at", "2000-01-01T00:00:00+00:00"), ("generic_orphan_recovery", {})])
def test_wrong_direct_predecessor_denies(setup_workflow, monkeypatch, field, value):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"])
    with measurement_lock(run, monkeypatch) as lock:
        lock.previous_owner[field] = value
        actual = handoff_result(observed, lock)
        assert not actual["ok"], (field, actual)


def test_intervening_owner_cannot_supply_old_source_release(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"])
    with measurement_lock(run, monkeypatch):
        pass
    with measurement_lock(run, monkeypatch) as later:
        assert later.previous_owner["run_id"] == "fps-source"
        assert not handoff_result(observed, later)["ok"]


@pytest.mark.parametrize("fault", ["selected_journal", "visibility", "unknown_process", "active_dut"])
def test_current_handoff_does_not_exempt_other_blockers(setup_workflow, monkeypatch, fault):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"],
        journal_setups=["dx"] if fault == "selected_journal" else [])
    if fault == "active_dut":
        other, active = select_dx(setup_workflow, physical=True)
        observed["processes"] += active["processes"]
        # The actual new lock must fail while this other owner holds the DUT.
        with pytest.raises(run_control.WorkflowRunLockedError):
            measurement_lock(run, monkeypatch).acquire()
        return
    if fault == "visibility": observed["visibility_errors"] = ["incomplete"]
    if fault == "unknown_process":
        observed["processes"].append({"pid": 999999999, "start_time_ticks": "1"})
    with measurement_lock(run, monkeypatch) as lock:
        assert not handoff_result(observed, lock)["ok"]


@pytest.mark.parametrize("fault", ["status_absent", "status_pending", "status_wrong_setup", "status_wrong_host",
    "status_wrong_model", "dispatch_wrong_run", "dispatch_missing_run", "dispatch_extra_run", "dispatch_wrong_session_time",
    "meta_absent", "meta_wrong_host", "meta_wrong_run", "meta_wrong_args", "meta_wrong_working_root",
    "transport_pending", "transport_no_end", "transport_wrong_start", "physical_running", "physical_missing",
    "physical_quarantine", "selected_journal", "unknown_journal", "journal_wrong_session", "journal_wrong_token_shape"])
def test_selected_incomplete_or_unknown_closure_denies(setup_workflow, monkeypatch, fault):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"], journal_setups=["h8"])
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    status, dispatch, local = case_files(run)
    lock = run_control.EvaluationRunLock.for_resource("dut:192.0.2.8", owner={})
    journals = list((run/"jobs/remote_process_leases").rglob("*.remote-lease.json"))
    if fault == "status_absent": status.unlink()
    elif fault == "status_pending": mutate(status, lambda v:v.update(status="running"))
    elif fault == "status_wrong_setup": mutate(status, lambda v:v.update(hardware_target_id="h8"))
    elif fault == "status_wrong_host": mutate(status, lambda v:v["host"].update(host="192.0.2.99"))
    elif fault == "status_wrong_model": mutate(status, lambda v:v.update(model_id="other"))
    elif fault == "dispatch_wrong_run": mutate(dispatch, lambda v:v.update(run_id="other"))
    elif fault == "dispatch_missing_run": mutate(dispatch, lambda v:v["execution_gates"].update(hardware_run_ids=[]))
    elif fault == "dispatch_extra_run": mutate(dispatch, lambda v:v["execution_gates"].update(hardware_run_ids=["dx-quality","other"]))
    elif fault == "dispatch_wrong_session_time": mutate(dispatch, lambda v:v.update(created_at="2000-01-01T00:00:00+00:00"))
    elif fault == "meta_absent": (local/"run_meta.json").unlink()
    elif fault == "meta_wrong_host": mutate(local/"run_meta.json", lambda v:v["host"].update(host="192.0.2.99"))
    elif fault == "meta_wrong_run": mutate(local/"run_meta.json", lambda v:v["resume"].update(requested_run_id="other"))
    elif fault == "meta_wrong_args": mutate(local/"run_meta.json", lambda v:v["args"].update(add_args="--run-id h8-quality"))
    elif fault == "meta_wrong_working_root": mutate(dispatch, lambda v:v.update(local_working_dir=str(run.parent)))
    elif fault == "transport_pending": mutate(local/"run_status.json", lambda v:v.update(status="running"))
    elif fault == "transport_no_end": mutate(local/"run_status.json", lambda v:v.update(ended_at=None))
    elif fault == "transport_wrong_start": mutate(local/"run_status.json", lambda v:v.update(started_at="2000-01-01T00:00:00+00:00"))
    elif fault == "physical_running": mutate(lock.lock_path, lambda v:v.update(state="running"))
    elif fault == "physical_missing": lock.lock_path.unlink()
    elif fault == "physical_quarantine": lock.quarantine_path.write_text("{}")
    elif fault == "selected_journal": mutate(journals[0], lambda v:v["control_argv_prefix"].__setitem__(-1,"fixture@192.0.2.8"))
    elif fault == "unknown_journal": mutate(journals[0], lambda v:v["control_argv_prefix"].__setitem__(-1,"fixture@192.0.2.99"))
    elif fault == "journal_wrong_session": mutate(journals[0], lambda v:v.update(session_sha256="f"*64))
    elif fault == "journal_wrong_token_shape": mutate(journals[0], lambda v:v.update(token=None))
    actual = result(observed)
    assert not actual["ok"], (fault, actual)
    assert actual["scope_errors"]


def test_current_capture_dispatch_without_terminal_and_running_physical_denies(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, journal_setups=["dx", "h8"], physical=True, physical_setup="dx")
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    actual = result(observed)
    assert not actual["ok"]
    assert "selected_physical_dut_is_active" in actual["scope_errors"]


def test_all_planned_models_need_the_selected_setup_terminal(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"],
                              model_ids=["m", "second"], completed_models=["m"])
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    actual = result(observed)
    assert not actual["ok"], actual
    assert "selected_dut_workflow_has_not_entered_central_only_phase" in actual["scope_errors"]


def test_readonly_owner_scope_never_acquires_resource_or_changes_metadata(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"], journal_setups=["h8"])
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    snapshots = {p:p.read_bytes() for p in run.rglob("*.json")}
    locks = {p:p.read_bytes() for p in (run.parent/"locks").rglob("*.lock")}
    monkeypatch.setattr(run_control.EvaluationRunLock, "acquire", lambda *_a,**_k:pytest.fail("no observer acquisition"))
    assert result(observed)["ok"]
    assert all(p.read_bytes()==b for p,b in {**snapshots,**locks}.items())


def test_every_planned_model_terminal_can_close_selected_setup(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"],
                              model_ids=["m", "second"], journal_setups=["h8"])
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    actual = result(observed)
    assert actual["ok"], actual


def test_other_setup_live_local_lease_stays_bound_to_its_target(setup_workflow, monkeypatch):
    run, observed = select_dx(setup_workflow, completed_setups=["dx"], journal_setups=["h8"])
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    identity = observed["processes"][0]
    fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
    journal_path = next((run/"jobs/remote_process_leases").rglob("*.remote-lease.json"))
    journal = json.loads(journal_path.read_text())
    path = run/"fixture-live.lease.json"
    path.write_text(json.dumps({"schema":"onnx-splitpoint/remote-process-lease",
        "pid":identity["pid"],"pgid":int(fields[2]),"start_time_ticks":identity["start_time_ticks"],
        **{key:journal[key] for key in ("operation_id","token","run_sha256","session_sha256")}}))
    observed["live_product_leases"] = [{**identity,"path":str(path)}]
    actual = result(observed)
    assert actual["ok"], actual
    mutate(journal_path, lambda value:value["control_argv_prefix"].__setitem__(-1,"fixture@192.0.2.8"))
    assert not result(observed)["ok"]
