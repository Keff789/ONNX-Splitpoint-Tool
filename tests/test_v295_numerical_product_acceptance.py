"""Normal acceptance coordinator's bounded A4 numerical admission phase.

Only hardware transport/dispatch seams are replaced.  The saved profile,
coordinator, proof binding, job results and phase distinction remain real.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from onnx_splitpoint_tool.workflow import hardware_acceptance as coordinator
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.native_command_contract import seal_native_command_contract


SETUP = "orin_nx_deepx_m1_01"
IDENTITY = {"model": "resnet50", "setup_id": SETUP, "case": "b045", "backend": "deepx_to_trt"}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return coordinator.file_binding(path)


def split_contract(*, warmup=100, frames=1000):
    python = "/home/nx/venvs/deepx-runtime/bin/python"
    artifacts = {"python_executable": {"path": python, "sha256": "a" * 64},
                 "dxnn": {"path": "/home/nx/original/model.dxnn", "sha256": "b" * 64},
                 "engine": {"path": "/home/nx/original/suffix.engine", "sha256": "d" * 64}}
    return seal_native_command_contract({**IDENTITY, "complete": True, "task": "classification",
        "runner": "/home/nx/ONNX-Splitpoint-Tool/scripts/native_deepx_trt_e2e_from_benchmarkset.py",
        "runner_sha256": "e" * 64, "input_image": "/home/nx/image.jpg", "input_image_sha256": "f" * 64,
        "python_executable": python, "benchmark_set": "/home/nx/original/benchmark_set", "artifacts": artifacts,
        "interpreter_identity": {"executable": python, "executable_sha256": "a" * 64},
        "runtime_options": {"warmup": warmup, "frames": frames, "queue_depth": 3,
            "deepx_runtime_mode": "A4", "deepx_max_inflight": 4},
        "boundary_contract": {"boundary_layout_effective": "as_input"}})


def valid_split_numerical_raw():
    return {"ok": True, "case": "b045", "deepx_runtime_mode": "A4", "deepx_max_inflight": 4,
        "native_command_contract": split_contract(warmup=0, frames=64), "completed_frames": 64,
        "completed_work_units": 64, "warmup": 0, "postprocess_completion_verified": True,
        "deepx_final_drain_complete": True,
        "deepx_execution_lifecycle": {"phase": "drained", "completion": "drained",
            "runtime_construction_started": True, "runtime_construction_completed": True, "inference_started": True,
            "runtime_constructors_entered": 2, "runtime_constructors_returned": 2,
            "runtime_calls_entered": 257, "runtime_calls_completed": 257},
        "deepx_runtime_observation": {"submitted": 64, "submit_attempts": 64, "wait_completed": 64,
            "pending": 0, "unreleased": 0, "unsafe_runtime_state": False},
        "deepx_numerical_preflight": {
            "status": "passed", "mode": "A4", "image_count": 32, "serial_reference_requests": 32,
            "mixed_completions": 32, "drained": True, "actual_pending_max": 4, "queue_depth": 3,
            "float_rtol": 1e-4, "float_atol": 1e-5, "integer_exact": True, "nonfinite_rejected": True,
            "comparison_outside_performance_timing": True,
            "requests": [{"request_id": index, "image": f"/home/nx/image{index:02d}.jpg", "completed_task_equal": True}
                         for index in range(32)]}}


@pytest.fixture
def acceptance_request(tmp_path):
    source = tmp_path / "historical"
    source.mkdir()
    plan = save(source / "plan.json", IDENTITY)
    images = save(tmp_path / "phase/numerical_images.json", {
        "images": [f"/home/nx/original/images/image{index:02d}.jpg" for index in range(32)]})
    profile = {
        "name": "bounded_a4_numerical", "model_suite": {"primary": [{"id": "resnet50"}]},
        "hardware": {"selected_setups": [SETUP]},
        "native_producers": {"frames": 1000, "warmup": 100, "repetitions": 3, "deepx_runtime_mode": "A4",
            "energy": {"duration_s": 60, "physical_scope": "FS", "window_label": "command",
                "task_budget": {"enabled": True, "max_retries": 2, "max_transport_failures": 20}}},
        "hardware_acceptance": {"enabled": True, "phase": "numerical", "mode": "scientific",
            "source_report_root": str(source), "source_primary_run": str(source), "cells": [{
                "model_id": "resnet50", "setup_id": SETUP, "case_id": "b045", "native_backend": "deepx_to_trt",
                "runtime_variants": ["native"], "native_source_plan": plan, "numerical_images": images}]}}
    options = WorkflowOptions("profile.yaml", str(tmp_path / "fresh"), execution_mode="generate_and_run")
    return profile, options


def numerical_result(root):
    evidence = root / "reports/worker.json"
    save(evidence, {"ok": True})
    return {"status": "numerical_passed", "image_count": 32, "serial_reference_count": 32,
            "mixed_completed_count": 32, "smoke_completed_count": 64, "deepx_runtime_mode": "A4",
            "device_completion_proven": True, "fresh_hardware_executed": True,
            "new_capture_count": 0, "new_performance_processes": 0,
            "scientific_claim_eligible": False, "scientific_cohort_complete": False,
            "scientific_statistics_published": False,
            "evidence_paths": [str(evidence)]}


def runner_fixture(profile, options, root, events):
    root.mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    return SimpleNamespace(
        profile_payload=profile, options=options, run_dir=root, run_id=root.name,
        manifest_path=root / "run_manifest.json", artifact_index_path=root / "artifact_index.json",
        manifest={}, stage_results=[], report_paths=[], outputs={}, warnings=[],
        _cancel_event=threading.Event(), _process_registry=object(),
        _remote_process_registry=SimpleNamespace(cancelled=False),
        log=lambda *args: None, progress=lambda *args: None, job_event=events.append,
        _save_manifest=lambda status: None, _write_workflow_control=lambda: None,
        _register_artifacts=lambda *args, **kwargs: None)


def coordinator_seams(monkeypatch, prepared):
    calls = []
    class Broker:
        def tick(self):
            calls.append("broker_tick")
        def close(self):
            calls.append("broker_close")
    def resource_context(runner, config, context):
        calls.append("broker_context")
        context.update(environment={}, source_ids={SETUP: "192.0.2.10"})
        return Broker()
    def adapter(variant):
        assert variant == "native"
        def prepare(cell, context):
            calls.append(("prepare", context["acceptance_phase"]))
            return copy.deepcopy(prepared)
        def execute(*args):
            calls.append("regular_execute")
            pytest.fail("numerical phase must not run regular performance/energy")
        return prepare, execute
    monkeypatch.setattr(coordinator, "_resource_context", resource_context)
    monkeypatch.setattr(coordinator, "_adapter", adapter)
    return calls


def test_explicit_numerical_phase_preserves_saved_regular_profile_recipe(acceptance_request):
    profile, options = acceptance_request
    original = copy.deepcopy(profile)
    config = coordinator.validate_request(profile, options)
    assert config["phase"] == "numerical" and config["mode"] == "scientific"
    assert config["regular_budget"]["frames"] == 1000
    assert config["regular_budget"]["warmup"] == 100
    assert config["regular_budget"]["repetitions"] == 3
    assert config["regular_budget"]["duration_s"] == 60
    assert profile == original


@pytest.mark.parametrize("change", ["unknown_phase", "S1", "unspecified", "generic", "hailo", "functional"])
def test_numerical_phase_cannot_change_generic_hailo_or_historical_s1(acceptance_request, change):
    profile, options = acceptance_request
    config = profile["hardware_acceptance"]
    if change == "unknown_phase":
        config["phase"] = "all_models"
    elif change == "S1":
        profile["native_producers"]["deepx_runtime_mode"] = "S1"
    elif change == "unspecified":
        profile["native_producers"].pop("deepx_runtime_mode")
    elif change == "generic":
        config["cells"][0]["runtime_variants"] = ["generic_completed"]
    elif change == "hailo":
        config["cells"][0]["native_backend"] = "hailo8_to_trt"
    else:
        config["mode"] = "functional"
        profile["native_producers"]["repetitions"] = 1
    with pytest.raises(ValueError, match="phase|Native DeepX A4"):
        coordinator.validate_request(profile, options)


def test_missing_phase_keeps_existing_complete_acceptance(acceptance_request):
    profile, options = acceptance_request
    profile["hardware_acceptance"].pop("phase")
    assert coordinator.validate_request(profile, options)["phase"] == "complete"


def test_saved_profile_start_snapshot_preserves_explicit_numerical_phase(acceptance_request, tmp_path):
    import yaml
    from onnx_splitpoint_tool.run_modes import default_run_modes_config
    from onnx_splitpoint_tool.workflow.profile_options import load_runtime_profile_snapshot, workflow_options_from_profile_snapshot
    minimal, _ = acceptance_request
    repo = Path(__file__).resolve().parents[1]
    profile = yaml.safe_load((repo / "onnx_splitpoint_tool/resources/evaluation_profiles/smoke_regression_v1.yaml").read_text())
    profile["hardware_acceptance"] = minimal["hardware_acceptance"]
    profile["hardware"] = minimal["hardware"]
    profile["model_suite"]["primary"] = [row for row in profile["model_suite"]["primary"] if row["id"] == "resnet50"]
    profile["model_suite"]["reserve"] = []
    profile.setdefault("native_producers", {}).update(minimal["native_producers"], enabled=True, backends=["deepx"])
    mode = default_run_modes_config()["modes"]["standard"]
    profile["execution_preset"] = {"id": "standard", "follow_tool_config": False, "snapshot": mode,
        "overrides": {"native_enabled": True, "energy_enabled": True,
            "native_performance": {"frames": 1000, "warmup": 100, "repetitions": 3}}}
    profile["workflow"] = {"execution_mode": "generate_and_run", "skip_runtime_benchmarks": False}
    path = tmp_path / "numerical_profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    resolved, snapshot = load_runtime_profile_snapshot(str(path))
    options = workflow_options_from_profile_snapshot(profile_request=str(path), out_root=str(tmp_path / "outputs"), start_snapshot=snapshot)
    assert resolved["hardware_acceptance"]["phase"] == "numerical"
    assert resolved["hardware_acceptance"]["cells"][0]["numerical_images"] == minimal["hardware_acceptance"]["cells"][0]["numerical_images"]
    assert options.native_producer_deepx_runtime_mode == "A4"
    assert coordinator.validate_request(resolved, options)["phase"] == "numerical"


@pytest.mark.parametrize("field,value", [
    ("image_count", 0), ("image_count", 33), ("serial_reference_count", 31),
    ("mixed_completed_count", 31), ("smoke_completed_count", 63),
    ("deepx_runtime_mode", "S1"), ("device_completion_proven", False),
    ("fresh_hardware_executed", False), ("new_capture_count", 1), ("new_performance_processes", 1),
    ("new_capture_count", False), ("new_performance_processes", False),
    ("scientific_claim_eligible", True), ("scientific_cohort_complete", True), ("scientific_statistics_published", True),
])
def test_numerical_pass_requires_actual_bounded_completion_and_zero_energy(tmp_path, field, value):
    result = numerical_result(tmp_path)
    result[field] = value
    with pytest.raises(ValueError, match="numerical PASS"):
        coordinator._accepted_numerical_result(result, {"run_dir": str(tmp_path)})


def test_numerical_pass_requires_fresh_bound_evidence_and_preserves_failures(tmp_path):
    current = tmp_path / "current"
    current.mkdir()
    result = numerical_result(current)
    assert coordinator._accepted_numerical_result(result, {"run_dir": str(current)}) == result
    failed = {**result, "status": "failed", "device_completion_proven": False}
    assert coordinator._accepted_numerical_result(failed, {"run_dir": str(current)}) == failed
    result["evidence_paths"] = [str(tmp_path / "outside.json")]
    save(Path(result["evidence_paths"][0]), {})
    with pytest.raises(ValueError, match="new normal EvaluationRun"):
        coordinator._accepted_numerical_result(result, {"run_dir": str(current)})


def test_numerical_success_uses_normal_coordinator_and_broker_without_scientific_or_energy_pass(acceptance_request, tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
    profile, options = acceptance_request
    root = tmp_path / "phase/hardware_runs/resnet50__orin_nx_deepx_m1_01/numerical"
    events = []
    runner = runner_fixture(profile, options, root, events)
    prepared = {"identity": IDENTITY, "output": str(root / "native/cell"), "deepx_runtime_mode": "A4"}
    calls = coordinator_seams(monkeypatch, prepared)
    def run(prepared, context, images):
        calls.append(("numerical_execute", context["acceptance_phase"]))
        assert images == profile["hardware_acceptance"]["cells"][0]["numerical_images"]
        assert context["expected_frames"] == 1000 and context["expected_warmup"] == 100
        return numerical_result(root)
    monkeypatch.setattr(numerical, "run_numerical_native", run)
    result = coordinator.run_acceptance(runner)
    report = json.loads((root / "reports/hardware_acceptance_results.json").read_text())
    assert result.status == "ok" and result.ok
    assert report["acceptance_phase"] == "numerical"
    assert report["numerical_pass_paths"] == 1 and report["fresh_pass_paths"] == 0
    assert report["hardware_acceptance_complete"] is False and report["new_capture_count"] == 0
    assert report["rows"][0]["status"] == "numerical_passed"
    assert report["source_campaign_resumed"] is False and report["new_dataset_quality_started"] is False
    assert ("prepare", "numerical") in calls and ("numerical_execute", "numerical") in calls
    assert "regular_execute" not in calls
    assert calls.index("broker_context") < calls.index(("prepare", "numerical"))
    assert calls[-1] == "broker_close"
    assert len(events) == 2 and all(row["job_type"] == "HardwareAcceptanceJob" for row in events)


@pytest.mark.parametrize("failure", ["failed", "unknown_drain"])
def test_numerical_failure_never_projects_as_completed_hardware_acceptance(acceptance_request, tmp_path, monkeypatch, failure):
    from onnx_splitpoint_tool.workflow import hardware_acceptance_numerical as numerical
    profile, options = acceptance_request
    root = tmp_path / "phase/hardware_runs/resnet50__orin_nx_deepx_m1_01/numerical"
    runner = runner_fixture(profile, options, root, [])
    prepared = {"identity": IDENTITY, "output": str(root / "native/cell"), "deepx_runtime_mode": "A4"}
    calls = coordinator_seams(monkeypatch, prepared)
    def run(*args):
        result = numerical_result(root)
        if failure == "failed":
            result.update(status="failed", reason="actual mixed parity failed")
        else:
            result["device_completion_proven"] = False
        return result
    monkeypatch.setattr(numerical, "run_numerical_native", run)
    result = coordinator.run_acceptance(runner)
    report = json.loads((root / "reports/hardware_acceptance_results.json").read_text())
    assert result.status == "partial" and not result.ok
    assert report["numerical_pass_paths"] == report["fresh_pass_paths"] == 0
    assert report["hardware_acceptance_complete"] is False and report["new_capture_count"] == 0
    assert report["rows"][0]["status"] == "failed"
    assert "regular_execute" not in calls and calls[-1] == "broker_close"


@pytest.fixture
def proof(tmp_path):
    phase = tmp_path / "phase"
    cell_root = phase / "hardware_runs/resnet50__orin_nx_deepx_m1_01"
    origin, current = cell_root / "numerical/native/cell", cell_root / "complete"
    matrix = save(phase / "MATRIX.json", {"cells": [IDENTITY]})
    images = save(phase / "images.json", {"images": [f"/home/nx/image{index:02d}.jpg" for index in range(32)]})
    contract = split_contract()
    worker = save(origin / "worker.json", valid_split_numerical_raw())
    payload = {"schema": "onnx-splitpoint/hardware-acceptance-native-numerical/v1", "status": "numerical_passed",
               "identity": IDENTITY, "phase_matrix": matrix, "source_contract_sha256": contract["contract_sha256"],
               "deepx_runtime_mode": "A4", "device_completion_proven": True, "numerical_images": images,
               "image_count": 32, "serial_reference_count": 32, "mixed_completed_count": 32, "smoke_completed_count": 64,
               "worker_report": worker}
    path = origin / "RESULT.json"
    reference = save(path, payload)
    cell = {"model_id": "resnet50", "setup_id": SETUP, "case_id": "b045", "numerical_images": images,
            "numerical_reference": reference}
    prepared = {"identity": IDENTITY, "source_contract": contract,
                "deepx_runtime_mode": "A4", "phase_matrix": matrix, "output": str(current / "native/cell")}
    return cell, prepared, {"run_dir": str(current)}, payload, path


def test_complete_phase_revalidates_bound_numerical_reference(proof):
    cell, prepared, context, _, _ = proof
    result = coordinator._verify_numerical_reference(cell, prepared, context)
    assert result["status"] == "numerical_passed" and result["identity"] == IDENTITY


@pytest.mark.parametrize("change", ["failed_status", "failed_parity", "unsafe_sdk", "bad_artifact"])
def test_complete_phase_rechecks_actual_worker_instead_of_trusting_pass_summary(proof, change):
    cell, prepared, context, payload, path = proof
    worker_path = Path(payload["worker_report"]["path"])
    raw = json.loads(worker_path.read_text())
    if change == "failed_status":
        raw["ok"] = False
    elif change == "failed_parity":
        raw["deepx_numerical_preflight"]["status"] = "failed"
    elif change == "unsafe_sdk":
        raw["unsafe_runtime_state"] = True
    else:
        contract = raw["native_command_contract"]
        contract["artifacts"]["dxnn"]["sha256"] = "1" * 64
        raw["native_command_contract"] = seal_native_command_contract(contract)
    payload["worker_report"] = save(worker_path, raw)
    cell["numerical_reference"] = save(path, payload)
    with pytest.raises(ValueError):
        coordinator._verify_numerical_reference(cell, prepared, context)


@pytest.mark.parametrize("change", ["missing", "nohash", "hashdrift", "mode", "identity", "matrix", "artifact", "drain",
                                    "images", "serial", "mixed", "serial_bool", "mixed_bool", "smoke", "workerhash", "workeroutside", "currentrun"])
def test_complete_phase_rejects_missing_stale_or_mismatched_numerical_proof(proof, tmp_path, change):
    cell, prepared, context, payload, path = proof
    if change == "missing":
        cell.pop("numerical_reference")
    elif change == "nohash":
        cell["numerical_reference"].pop("sha256")
    elif change == "hashdrift":
        path.write_text("{}")
    elif change == "currentrun":
        context["run_dir"] = str(path.parent)
    else:
        if change == "mode":
            payload["deepx_runtime_mode"] = "S1"
        elif change == "identity":
            payload["identity"] = {**IDENTITY, "case": "b046"}
        elif change == "matrix":
            payload["phase_matrix"] = {**payload["phase_matrix"], "sha256": "a" * 64}
        elif change == "artifact":
            payload["source_contract_sha256"] = "a" * 64
        elif change == "drain":
            payload["device_completion_proven"] = False
        elif change == "images":
            payload["numerical_images"] = {**payload["numerical_images"], "sha256": "a" * 64}
        elif change == "serial":
            payload["serial_reference_count"] = 0
        elif change == "mixed":
            payload["mixed_completed_count"] = 0
        elif change == "serial_bool":
            payload["serial_reference_count"] = True
        elif change == "mixed_bool":
            payload["mixed_completed_count"] = True
        elif change == "smoke":
            payload["smoke_completed_count"] = 63
        elif change == "workerhash":
            Path(payload["worker_report"]["path"]).write_text("{}")
        else:
            payload["worker_report"] = save(tmp_path / "outside/worker.json", {"ok": True})
        cell["numerical_reference"] = save(path, payload)
    with pytest.raises((ValueError, FileNotFoundError)):
        coordinator._verify_numerical_reference(cell, prepared, context)


def test_complete_phase_without_numerical_proof_cannot_reach_performance_or_energy(acceptance_request, tmp_path, monkeypatch):
    profile, options = acceptance_request
    profile["hardware_acceptance"]["phase"] = "complete"
    root = tmp_path / "phase/hardware_runs/resnet50__orin_nx_deepx_m1_01/complete"
    runner = runner_fixture(profile, options, root, [])
    matrix = save(tmp_path / "phase/MATRIX.json", {"cells": [IDENTITY]})
    prepared = {"identity": IDENTITY, "output": str(root / "native/cell"), "deepx_runtime_mode": "A4",
                "phase_matrix": matrix, "source_contract": {"contract_sha256": "c" * 64}}
    calls = coordinator_seams(monkeypatch, prepared)
    result = coordinator.run_acceptance(runner)
    report = json.loads((root / "reports/hardware_acceptance_results.json").read_text())
    assert result.status == "partial" and not result.ok
    assert report["new_capture_count"] == report["fresh_pass_paths"] == 0
    assert "numerical PASS reference" in report["rows"][0]["reason"]
    assert "regular_execute" not in calls


def test_complete_phase_dispatches_only_after_actual_bound_numerical_proof(acceptance_request, proof, monkeypatch):
    profile, options = acceptance_request
    cell, prepared, context, _, _ = proof
    profile["hardware_acceptance"]["phase"] = "complete"
    profile["hardware_acceptance"]["cells"][0].update(cell)
    root = Path(context["run_dir"])
    runner = runner_fixture(profile, options, root, [])
    calls = coordinator_seams(monkeypatch, prepared)
    def adapter(variant):
        assert variant == "native"
        def prepare(cell, context):
            calls.append("prepare")
            return copy.deepcopy(prepared)
        def execute(prepared, context):
            admission = Path(prepared["output"]) / "NUMERICAL_ADMISSION.json"
            assert admission.is_file(), "regular execution started without the normal numeric admission receipt"
            receipt = json.loads(admission.read_text())
            assert receipt["status"] == "numerical_passed" and receipt["identity"] == IDENTITY
            calls.append("regular_execute")
            evidence = root / "reports/physical_cohort_fixture.json"
            save(evidence, {"transport_is_mocked": True})
            return {"status": "completed", "fresh_hardware_executed": True, "new_capture_count": 3,
                    "valid_repeats": {"performance": 3, "energy": 3}, "evidence_paths": [str(evidence)]}
        return prepare, execute
    monkeypatch.setattr(coordinator, "_adapter", adapter)
    result = coordinator.run_acceptance(runner)
    report = json.loads((root / "reports/hardware_acceptance_results.json").read_text())
    assert result.ok and report["hardware_acceptance_complete"] is True
    assert report["fresh_pass_paths"] == 1 and report["numerical_pass_paths"] == 0
    assert calls.index("prepare") < calls.index("regular_execute") < calls.index("broker_close")
