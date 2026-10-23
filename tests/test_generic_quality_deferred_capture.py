"""Explicit capture deferral keeps normal source/lifecycle/transport contracts."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from onnx_splitpoint_tool import generic_producer_binding, quality_service
from onnx_splitpoint_tool.benchmark.evaluation_profiles import validate_evaluation_profile_payload
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.process_control import bind_workflow_resource_options, controller_local_activity
from onnx_splitpoint_tool.runners.generic_quality_accounting import verify_capture
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.execution_binding import _copy_remote_result_files
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from onnx_splitpoint_tool.workflow.start_snapshot import (
    StartSnapshotConsistencyError, build_profile_start_snapshot, profile_selection_view,
)
from tests.test_ap08_stage_lifecycle import MODELS, _runner
from tests.test_generic_producer_creation_binding import _normal_creation
from tests.test_generic_existing_cpu_reference import sample
from tests.test_generic_quality_accounting import real_capture
from tests.test_generic_quality_profile_transport import profile
from tests.test_generic_setup_quality_callback import SETUPS, _start_parent, controlled


def capture_profile():
    value = profile()
    value["workflow_execution"] = {"defer_central_quality": True}
    value["quality_gate"].setdefault("statistics", {})["execution_location"] = "central_management"
    value["native_producers"]["enabled"] = False
    value["native_producers"]["energy"]["enabled"] = False
    value["energy"]["enabled"] = False
    value["workflow"]["skip_runtime_benchmarks"] = False
    value.setdefault("hardware", {})["selected_setups"] = [value["run_profiles"][0]["hardware_setup_id"]]
    value["model_suite"] = {"primary": [{"id": "yolo26s", "task": "detection"}]}
    return value


def snapshot(value, resolved=None):
    return build_profile_start_snapshot(profile_request="deferred-capture", source_profile=value,
        resolved_profile=value if resolved is None else resolved, profile_id="deferred-capture",
        profile_path="/controlled/deferred.json", profile_source="file")


def test_schema_freezes_explicit_flag_and_absent_legacy_shape():
    source = capture_profile()
    validate_evaluation_profile_payload(source, source="deferred capture test")
    frozen = snapshot(source)
    assert frozen["source_profile"]["workflow_execution"]["defer_central_quality"] is True
    assert frozen["resolved_profile"]["workflow_execution"]["defer_central_quality"] is True
    legacy = deepcopy(source)
    legacy.pop("workflow_execution")
    expected = {"native_release_mode": "global_barrier", "setup_queue_mode": "model_barrier"}
    assert profile_selection_view(legacy)["workflow_execution"] == expected
    assert build_effective_execution_plan(legacy)["workflow_execution"] == expected
    assert frozen["effective_execution_plan"]["workflow_execution"] == expected
    assert snapshot(legacy)["resolved_execution_sha256"] != frozen["resolved_execution_sha256"]
    assert snapshot(legacy)["source_profile_sha256"] != frozen["source_profile_sha256"]
    changed = deepcopy(source)
    changed["workflow_execution"]["defer_central_quality"] = False
    with pytest.raises(StartSnapshotConsistencyError, match="workflow_execution"):
        snapshot(source, changed)


@pytest.mark.parametrize("raw", ["true", 1, None])
def test_schema_and_runner_reject_non_boolean_flag(tmp_path, raw):
    value = capture_profile()
    value["workflow_execution"]["defer_central_quality"] = raw
    with pytest.raises(ValueError):
        validate_evaluation_profile_payload(value, source="bad capture flag")
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = value
    with pytest.raises(ValueError, match="must be boolean"):
        runner._validate_requested_start_contract()


@pytest.mark.parametrize("fault", ["native", "energy", "full", "no_source", "local", "reference",
                                  "setup", "source_hash", "cases"])
def test_invalid_scope_blocks_before_any_run_is_opened(tmp_path, monkeypatch, fault):
    value = capture_profile()
    run = value["run_profiles"][0]
    if fault == "native": value["native_producers"]["enabled"] = True
    elif fault == "energy": value["energy"]["enabled"] = True
    elif fault == "full": run["variants"] = ["full", "composed"]
    elif fault == "no_source": run.pop("generic_quality_completion_configs")
    elif fault == "local": value["quality_gate"]["statistics"]["execution_location"] = "local"
    elif fault == "reference": value["quality_gate"]["management_reference"].pop("existing_reference")
    elif fault == "setup": value["hardware"]["selected_setups"] = ["another-setup"]
    elif fault == "source_hash": run["generic_quality_completion_config_sha256"]["b039"] = "b" * 64
    else: run["case_ids"] = ["b040"]
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    monkeypatch.setattr(runner, "_load_profile", lambda: setattr(runner, "profile_payload", value))
    monkeypatch.setattr(runner, "_open_run_dir", lambda: pytest.fail("invalid scope opened a run"))
    with pytest.raises(ValueError):
        runner.run()


def test_gate_and_normal_stage_defer_without_constructing_or_submitting(tmp_path, monkeypatch):
    runner, _ = _runner(tmp_path)
    runner.profile_payload = capture_profile()
    def forbidden(*args, **kwargs):
        pytest.fail("deferred capture constructed or submitted central CPU work")
    monkeypatch.setattr(quality_service, "ManagementQualityService", forbidden)
    monkeypatch.setattr(quality_service, "ProcessPoolExecutor", forbidden)
    gate = runner._benchmark_controller_gate()
    assert gate is runner._benchmark_controller_gate()
    with bind_workflow_resource_options(controller_gate=gate):
        with controller_local_activity("transfer"):
            assert gate.snapshot()["cpu_active"] == 1
    assert gate.snapshot()["cpu_active"] == 0
    assert runner._queue_central_quality_requests("yolo26s") == 0
    with pytest.raises(RuntimeError, match="forbidden"):
        runner._ensure_central_quality_service()
    stage = runner._run_stage(None, "evaluate_quality", runner._stage_evaluate_quality)
    assert stage.status == "partial" and stage.complete is True
    summary = json.loads((runner.run_dir / "quality_management/central_quality_summary.json").read_text())
    assert summary["status"] == "deferred" and summary["quality_required"] is True
    assert summary["technical_status"] == summary["quality_decision"] == summary["scientific_status"] == "not_evaluated"
    assert summary["scientific_pass"] is summary["claim_eligible"] is False
    assert summary["workers_started"] == summary["completed_count"] == 0
    assert runner._central_quality_service is runner._central_quality_coord_executor is None
    assert not runner._central_quality_futures
    assert not list(runner.run_dir.rglob("*_central_result.json"))
    assert runner._derive_final_status()[0] == "partial"


def test_normal_setup_callbacks_and_model_finalizers_submit_nothing_when_deferred(controlled):
    runner, _ = controlled
    runner.profile_payload["workflow_execution"]["defer_central_quality"] = True
    runner.profile_payload["quality_gate"]["management_reference"] = {
        "existing_reference": {"reference_role": "canonical_cpu_ort"}}
    for run in runner.profile_payload["run_profiles"]:
        if not run.get("enabled", True):
            continue
        run.update(deepcopy(capture_profile()["run_profiles"][0]))
        run["id"] = run["hardware_setup_id"] = SETUPS[0] if run is runner.profile_payload["run_profiles"][0] else SETUPS[1]
    runner.release_fast.set()
    runner.release_slow.set()
    with ThreadPoolExecutor(max_workers=1) as controller:
        assert _start_parent(runner, controller).result(timeout=12) == set(MODELS)
    assert runner.callback_threads and runner.queue_calls
    assert not runner.service_instances and not runner.evaluated
    assert not runner._central_quality_futures


def test_absent_flag_keeps_one_normal_service_and_gate(tmp_path, monkeypatch):
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_dir = tmp_path
    runner.profile_payload = capture_profile()
    runner.profile_payload.pop("workflow_execution")
    instances = []
    class Service:
        def __init__(self, cache, **kwargs):
            self.pause_gate = kwargs["pause_gate"]
            instances.append(self)
    monkeypatch.setattr(quality_service, "ManagementQualityService", Service)
    assert runner._benchmark_controller_gate() is runner._ensure_central_quality_service().pause_gate
    assert len(instances) == 1
    runner._central_quality_coord_executor.shutdown()


def test_normal_existing_n5000_reference_is_bound_without_any_cpu_service(sample, monkeypatch):
    from onnx_splitpoint_tool.workflow.generic_quality_sidecars import bind_existing_generic_cpu_reference
    runner, suite, plan, policy, original, old = sample
    saved = {path: path.read_bytes() for path in old.iterdir()}
    runner.profile_payload = {**capture_profile(), "quality_gate": policy}
    monkeypatch.setattr(quality_service, "ManagementQualityService",
        lambda *args, **kwargs: pytest.fail("reference reuse opened a CPU service"))
    bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
        model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)
    (suite / "benchmark_plan.json").write_text(json.dumps(plan))
    runner._schedule_management_cpu_reference("yolo11l", suite)
    records, status, encoded = runner._management_reference_records("yolo11l")
    assert len(records) == 5000 and status["reference_sha256"] == original["sha256"]
    assert runner._central_quality_service is runner._management_reference_executor is None
    assert all(path.read_bytes() == content for path, content in saved.items())


def test_normal_collection_preserves_portable_capture_bytes(real_capture, tmp_path, monkeypatch):
    fixture, request_path, _, _ = real_capture
    request = json.loads(request_path.read_text())
    original_candidate = request_path.parent / request["candidate"]["path"]
    downloaded = tmp_path / "download/results/b039/results_hailo10_to_tensorrt/task_quality_inputs"
    downloaded.mkdir(parents=True)
    candidate = downloaded / "composed_candidate.json"
    candidate.write_bytes(original_candidate.read_bytes())
    request["candidate"]["path"] = candidate.name
    request_file = downloaded / "composed_request.json"
    request_file.write_text(json.dumps(request))
    before = {path.name: path.read_bytes() for path in (candidate, request_file)}
    setup = fixture["config"]["identity"]["setup_id"]
    copied = _copy_remote_result_files(tmp_path / "download", tmp_path / "normal/benchmark_results", flat_prefix=setup)
    local_request = next(Path(row["destination"]) for row in copied
        if row["kind"] == "central_quality_input" and row["destination"].endswith("composed_request.json"))
    proof = verify_capture(local_request, fixture["config"], config_path=fixture["path"],
        eval_run_id="synthetic", source_run_id="hailo10_to_tensorrt")
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.profile_payload = capture_profile()
    monkeypatch.setattr(quality_service, "ManagementQualityService", lambda *a, **k: pytest.fail("pool"))
    assert runner._queue_central_quality_requests("yolo26s", results_root=local_request.parent) == 0
    assert proof["central_quality_decision"] == "pending"
    assert local_request.read_bytes() == before["composed_request.json"]
    assert (local_request.parent / candidate.name).read_bytes() == before[candidate.name]


def test_deferred_creation_remains_regular_producer_binding(tmp_path):
    value = capture_profile()
    setup = value["hardware"]["selected_setups"][0]
    created = _normal_creation(tmp_path / "producer", setup=setup, profile_extra=value)
    binding = generic_producer_binding.build_producer_run_binding(created.primary, session_id=created.session)
    row = {"original_run": str(created.primary), "producer_run_binding": binding,
           "identity": {"model_id": "yolo26s", "case_id": "b039", "setup_id": setup}}
    assert generic_producer_binding.validate_producer_run_binding(created.primary, binding, [row])["session_id"] == created.session
    plan = json.loads((created.primary / "effective_execution_plan.json").read_text())
    assert "defer_central_quality" not in plan["workflow_execution"]


def test_f18_readers_rebuild_and_attribute_a_live_deferred_producer(tmp_path):
    """Run actual pinned F18 code in another interpreter, with a real Run flock."""
    repo = Path(__file__).resolve().parents[1]
    archive = subprocess.check_output(["git", "archive", "5e66064", "onnx_splitpoint_tool"], cwd=repo)
    baseline = tmp_path / "f18"
    baseline.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(baseline, filter="data")
    value = capture_profile()
    setup = value["hardware"]["selected_setups"][0]
    created = _normal_creation(tmp_path / "live-producer", setup=setup, profile_extra=value)
    binding = generic_producer_binding.build_producer_run_binding(created.primary, session_id=created.session)
    owner_source = r'''
import json, pathlib, sys, os
from onnx_splitpoint_tool.workflow.run_control import EvaluationRunLock
run = pathlib.Path(sys.argv[1]); manifest = json.loads((run / "run_manifest.json").read_text())
contract = manifest["resume_contract"]
lock = EvaluationRunLock(out_root=run.parent, run_dir=run, owner={
    "session_id": manifest["current_session_id"], "profile_sha256": contract["profile_sha256"],
    "plan_sha256": contract["plan_sha256"]})
lock.acquire()
fields = pathlib.Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()
print(json.dumps({"pid": os.getpid(), "start_time_ticks": fields[19]}), flush=True)
sys.stdin.buffer.read(1)
lock.release()
'''
    child = subprocess.Popen([sys.executable, "-B", "-c", owner_source, str(created.primary)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(repo)})
    try:
        line = child.stdout.readline()
        assert line, child.stderr.read().decode()
        observed = {"ok": False, "read_only": True, "status": "busy",
            "processes": [json.loads(line)], "recorded_owners": [], "live_product_leases": [], "visibility_errors": []}
        reader_source = r'''
import json, pathlib, sys, yaml
sys.path.insert(0, sys.argv[1])
from onnx_splitpoint_tool.workflow.start_snapshot import build_profile_start_snapshot, public_start_snapshot_metadata
from onnx_splitpoint_tool.workflow.run_control import build_resume_contract
from onnx_splitpoint_tool.generic_producer_binding import validate_producer_run_binding
from onnx_splitpoint_tool.workflow.generic_completed_task import _scoped_controller_owners
run = pathlib.Path(sys.argv[2]); binding = json.loads(sys.argv[3]); observed = json.loads(sys.argv[4])
read = lambda name: json.loads((run / name).read_text())
profile = yaml.safe_load((run / "profile.yaml").read_text())
source = yaml.safe_load((run / "profile_source.yaml").read_text())
saved = read("profile_start_snapshot.json"); manifest = read("run_manifest.json")
rebuilt = build_profile_start_snapshot(**{key: saved[key] for key in
    ("profile_request", "profile_id", "profile_path", "profile_source")},
    source_profile=source, resolved_profile=profile, runtime_bindings=saved["runtime_bindings"],
    schema_version=saved["schema_version"])
assert public_start_snapshot_metadata(rebuilt) == saved
assert build_resume_contract(profile_payload=profile, effective_execution_plan=read("effective_execution_plan.json"),
    options=manifest["options"]) == manifest["resume_contract"]
assert validate_producer_run_binding(run, binding)["session_id"] == manifest["current_session_id"]
fps = {"fps": {"id": "fps", "remote": {"host": "192.0.2.99", "user": "fixture", "port": 22}}}
result = _scoped_controller_owners(fps, observed)
assert result["ok"], result
assert result["normal_quality_owners"][0]["run_id"] == run.name
assert result["cleanup_proven"] is result["exclusive_start_permission"] is False
# The same unchanged F18 reader must still reject unclosed work on its own DUT.
assert not _scoped_controller_owners({"fps": manifest["hardware_targets"][0]}, observed)["ok"]
print(json.dumps({"snapshot_equal": True, "resume_contract_equal": True,
    "producer_binding_valid": True, "live_other_dut_owner_valid": True}))
'''
        result = subprocess.run([sys.executable, "-I", "-B", "-c", reader_source,
            str(baseline), str(created.primary), json.dumps(binding), json.dumps(observed)],
            cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)["live_other_dut_owner_valid"] is True
    finally:
        if child.poll() is None:
            child.stdin.write(b"x")
            child.stdin.flush()
        child.wait(timeout=10)
