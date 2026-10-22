"""FPS resource admission with real local run ownership and no hardware."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from onnx_splitpoint_tool.workflow import generic_completed_task as entry, run_control


def target(host="192.0.2.8", *, alias=""):
    return {"id": "fixture", "physical_host_id": alias,
            "remote": {"host": host, "user": "fixture", "port": 22}}


def idle():
    return {"ok": True, "read_only": True, "status": "no_known_competing_work_seen",
            "processes": [], "recorded_owners": [], "live_product_leases": [], "visibility_errors": []}


@pytest.fixture
def workflow(tmp_path):
    """A live subprocess holds the actual product Run flock until stdin closes."""
    children = []

    def start(*, host="192.0.2.10", central=False, physical=False, alias=""):
        run = tmp_path / ("run-" + str(len(children)))
        config = {"run": str(run), "target": target(host, alias=alias),
                  "central": central, "physical": physical}
        source = r'''
import json,sys,pathlib,os,yaml
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot,public_start_snapshot_metadata
c=json.loads(sys.argv[1]);run=pathlib.Path(c["run"]);run.mkdir()
run_control.platform_workflow_interlock_path=lambda: run.parent/"locks/platform.lock"
profile={"name":"fixture", "model_suite":{"primary":[{"id":"m"}]},
 "energy":{"enabled":False},"native_producers":{"enabled":False},"hardware_smoke":{"enabled":False},
 "hardware":{"selected_setups":["fixture"]},
 "run_profiles":[{"id":"fixture-quality","hardware_setup_id":"fixture","quality_source_only":True}],
 "quality_gate":{"statistics":{"workers":6,"max_active_requests":2,"execution_location":"central_management"}}}
plan={"models":["m"],"logical_run_profiles":["fixture-quality"],"setup_groups":{"fixture":["fixture-quality"]}}
options={};contract=run_control.build_resume_contract(profile_payload=profile,effective_execution_plan=plan,options=options)
session="a"*32
snap=build_profile_start_snapshot(profile_request="fixture",source_profile=profile,resolved_profile=profile,
 profile_id="fixture",profile_path=str(run/"profile.yaml"),profile_source="file",runtime_bindings={"hardware_targets":[c["target"]]})
def write(name,value):
 p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
for name in ("profile.yaml","profile_source.yaml"):(run/name).write_text(yaml.safe_dump(profile))
write("effective_execution_plan.json",plan)
write("profile_start_snapshot.json",public_start_snapshot_metadata(snap))
write("run_manifest.json",{"schema":"onnx-splitpoint/evaluation-run-manifest","run_id":run.name,"run_dir":str(run),
 "current_session_id":session,"options":options,"resume_contract":contract,"profile_start_snapshot":public_start_snapshot_metadata(snap),
 "hardware_targets":[c["target"]],"status":"running",
 "execution_sessions":[{"session_id":session,**contract}],
 "models":{"m":{"stages":{"run_benchmarks":{"status":"completed" if c["central"] else "running"}}}},
 "workflow_stages":{"evaluate_quality":{"status":"running" if c["central"] else "pending"}}})
write("jobs/workflow_control.json",{"schema":"onnx-splitpoint/workflow-control","run_id":run.name,"session_id":session})
(run/"jobs/remote_process_leases"/session).mkdir(parents=True)
lock=run_control.EvaluationRunLock(out_root=run.parent,run_dir=run,
 owner={"session_id":session,"profile_sha256":contract["profile_sha256"],"plan_sha256":contract["plan_sha256"]})
lock.acquire()
other=None
if c["physical"]:
 from onnx_splitpoint_tool.workflow.execution_binding import physical_dut_key
 other=run_control.EvaluationRunLock.for_resource(physical_dut_key(c["target"]),owner={"run_id":run.name,"purpose":"generic"});other.acquire()
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


def test_other_dut_workflow_is_proven_from_held_run_not_program_name(workflow):
    _, observed = workflow()
    result = entry._scoped_controller_owners({"fps": target()}, observed)
    assert result["ok"], result
    assert result["normal_quality_owners"][0]["resources"] == ["dut:192.0.2.10"]
    assert result["cleanup_proven"] is result["exclusive_start_permission"] is False


def test_same_dut_workflow_blocks_even_with_different_registry_alias(workflow):
    _, observed = workflow(host="192.0.2.8", alias="quality-alias")
    result = entry._scoped_controller_owners({"fps": target(alias="fps-alias")}, observed)
    assert not result["ok"]
    assert "selected_dut_workflow_has_not_entered_central_only_phase" in result["scope_errors"]


def test_central_only_statistics_can_coexist_without_new_pool(workflow, monkeypatch):
    _, observed = workflow(host="192.0.2.8", central=True)
    result = entry._scoped_controller_owners({"fps": target()}, observed)
    assert result["ok"], result
    assert result["normal_quality_owners"][0]["central_only"] is True


def test_held_same_dut_blocks_even_if_stage_claims_central_only(workflow, tmp_path, monkeypatch):
    run, observed = workflow(host="192.0.2.8", central=True, physical=True)
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path", lambda: run.parent/"locks/platform.lock")
    result = entry._scoped_controller_owners({"fps": target()}, observed)
    assert not result["ok"]
    assert "selected_physical_dut_is_active" in result["scope_errors"]


def test_central_only_requires_empty_actual_remote_lease_journal(workflow):
    run, observed = workflow(host="192.0.2.8", central=True)
    (run/"jobs/remote_process_leases"/("a"*32)/"unfinished.remote-lease.json").write_text("{}")
    result = entry._scoped_controller_owners({"fps": target()}, observed)
    assert not result["ok"]


@pytest.mark.parametrize("fault", ["pid_start", "session", "target_binding", "plan_runs"])
def test_changed_identity_or_archived_bindings_block(workflow, fault):
    run, observed = workflow()
    if fault == "pid_start":
        observed["processes"][0]["start_time_ticks"] = "1"
    else:
        path = run / ("effective_execution_plan.json" if fault == "plan_runs" else "run_manifest.json")
        value = json.loads(path.read_text())
        if fault == "session": value["current_session_id"] = "b"*32
        elif fault == "target_binding": value["hardware_targets"] = [target()]
        else: value["logical_run_profiles"] = ["foreign-run"]
        path.write_text(json.dumps(value))
    result = entry._scoped_controller_owners({"fps": target()}, observed)
    assert not result["ok"]


@pytest.mark.parametrize("field", ["read_only", "processes", "visibility_errors", "recorded_owners", "live_product_leases"])
def test_incomplete_raw_visibility_never_counts_as_idle(field):
    observed = idle()
    observed.pop(field)
    assert not entry._scoped_controller_owners({"fps": target()}, observed)["ok"]


def test_indeterminate_raw_status_does_not_count_as_idle():
    observed = {**idle(), "ok": False, "status": "unknown"}
    assert not entry._scoped_controller_owners({"fps": target()}, observed)["ok"]


def test_unknown_lease_scope_blocks_even_for_independent_dut(workflow, tmp_path):
    _, observed = workflow()
    owner = observed["processes"][0]
    fields = Path(f"/proc/{owner['pid']}/stat").read_text().rsplit(") ", 1)[1].split()
    path = tmp_path/"foreign.lease.json"
    path.write_text(json.dumps({"schema": "onnx-splitpoint/remote-process-lease", "pid": owner["pid"],
        "pgid": int(fields[2]), "start_time_ticks": owner["start_time_ticks"],
        "run_sha256": "f"*64, "session_sha256": "e"*64}))
    observed["live_product_leases"] = [{**owner, "path": str(path)}]
    assert not entry._scoped_controller_owners({"fps": target()}, observed)["ok"]


def test_unknown_process_with_no_evaluation_run_owner_blocks(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import sys;sys.stdin.buffer.read(1)"], stdin=subprocess.PIPE)
    try:
        fields = Path(f"/proc/{child.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        observed = {**idle(), "ok": False, "status": "busy",
                    "processes": [{"pid": child.pid, "start_time_ticks": fields[19]}]}
        result = entry._scoped_controller_owners({"fps": target()}, observed)
        assert not result["ok"]
        assert "observed_process_has_no_live_normal_run_owner" in result["scope_errors"]
    finally:
        child.stdin.write(b"x"); child.stdin.flush(); child.wait(timeout=5)


def test_default_native_energy_probe_retains_strict_global_gate(workflow, tmp_path, monkeypatch):
    _, observed = workflow()
    monkeypatch.setattr(entry, "_observe_processes", lambda _: observed)
    monkeypatch.setattr(entry, "transport_for", lambda *_: pytest.fail("strict controller gate must block before SSH"))
    with pytest.raises(RuntimeError, match="controller"):
        entry.observe_owners({"fps": target()}, tmp_path/"strict")


def test_resource_mode_keeps_remote_same_dut_probe_strict(workflow, tmp_path, monkeypatch):
    _, observed = workflow()
    monkeypatch.setattr(entry, "_observe_processes", lambda _: observed)
    remote = {**idle(), "ok": False, "processes": [{"pid": 99}]}
    class Transport:
        def run_read_only(self, *args, **kwargs):
            return 0, "OWNERS="+json.dumps(remote)
    monkeypatch.setattr(entry, "transport_for", lambda *_: Transport())
    with pytest.raises(RuntimeError, match="device"):
        entry.observe_owners({"fps": target()}, tmp_path/"scoped", resource_scoped=True)
