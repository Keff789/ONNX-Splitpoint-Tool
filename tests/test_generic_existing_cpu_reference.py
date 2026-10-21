"""Explicit Generic source-only CPU reuse; no inference or compiler execution."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.preprocessing_contract import canonical_image_preprocessing_contract
from onnx_splitpoint_tool.quality_cache import image_ids_fingerprint, json_fingerprint
from onnx_splitpoint_tool.workflow.contracts import WorkflowOptions
from onnx_splitpoint_tool.workflow.generic_quality_sidecars import bind_existing_generic_cpu_reference
from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner
from tests.test_v269d_trt_central_quality_producer import _detection_quality_contract


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True).encode()
    path.write_bytes(encoded)
    return {"path": str(path), "sha256": hashlib.sha256(encoded).hexdigest(), "size_bytes": len(encoded)}


@pytest.fixture
def sample(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("explicit Generic reference reuse must never queue CPU inference")
    monkeypatch.setattr("onnx_splitpoint_tool.management_reference.generate_management_cpu_reference", forbidden)
    old, active = tmp_path / "closed", tmp_path / "active"
    suite = active / "models/yolo11l/benchmark_set/legacy_suite"
    validation = _write(suite / "validation/manifest.json", {"frozen_validation": True})
    rows = [{"image_id": f"{i:05}.jpg", "reference": [], "ground_truth": []} for i in range(5000)]
    contract = _detection_quality_contract()
    contract["dataset"].update(image_count=5000, manifest_sha256=validation["sha256"],
        image_ids_sha256=image_ids_fingerprint([r["image_id"] for r in rows]),
        ground_truth_sha256=json_fingerprint([{k: r[k] for k in ("image_id", "ground_truth")} for r in rows]))
    prep = canonical_image_preprocessing_contract("detection", [640, 640])
    contract["preprocessing"] = {**contract["preprocessing"], "identity": prep, "sha256": json_fingerprint(prep)}
    contract.pop("quality_contract_sha256")
    contract["quality_contract_sha256"] = json_fingerprint(contract)
    payload = {"schema": "onnx-splitpoint/task-quality-reference-input", "schema_version": 1,
        "task": "detection", "pairing_key": "image_id", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "provenance_required": True, "quality_contract": contract,
        "quality_contract_sha256": contract["quality_contract_sha256"], "records": rows}
    original = _write(old / "canonical_cpu_reference.json", payload)
    status = {"schema": "onnx-splitpoint/management-cpu-reference-job", "schema_version": 1,
        "status": "completed", "model_id": "yolo11l", "provider": "onnxruntime_cpu",
        "record_count": 5000, "semantic_reference_only": True, "execution_location": "central_management",
        "reference_sha256": original["sha256"], "reference_size_bytes": original["size_bytes"],
        "source_contract_sha256": "a" * 64, "quality_contract": contract,
        "include_in_latency_fps_energy": False, "include_in_ranking": False, "include_in_pareto": False}
    status_descriptor = _write(old / "status.json", status)
    policy = {"statistics": {"execution_location": "central_management", "bootstrap_repetitions": 1000,
                             "seed": 20260710}}
    policy_descriptor = _write(old / "policy.json", policy)
    descriptor = {"reference_role": "canonical_cpu_ort", "semantic_reference_only": True,
        "source_evaluation_run_id": "original", "reference_path": original["path"],
        "reference_sha256": original["sha256"], "reference_size_bytes": original["size_bytes"],
        "source_status": status_descriptor, "policy_source": policy_descriptor,
        "quality_contract": contract, "quality_contract_sha256": contract["quality_contract_sha256"]}
    policy["management_reference"] = {"existing_reference": descriptor}
    _write(active / "models/yolo11l/model_manifest.json", {"model_sha256": contract["model"]["sha256"]})
    config = {"expected_image_ids": [r["image_id"] for r in rows],
        "preprocess": {"mode": "letterbox", "rgb": True, "pad_value": 114},
        "completion_contract": {"processor_contract": {"input_hw": [640, 640]}}}
    sidecar = _write(suite / "quality_runtime/b039.json", config)
    plan = {"model_id": "yolo11l", "runs": [{"id": "hailo10_to_tensorrt", "type": "matrix",
        "variants": ["composed"], "quality_source_only": True, "image_scale": "norm",
        "validation_images": "validation", "generic_quality_completion_configs": {"b039": "quality_runtime/b039.json"},
        "generic_quality_completion_config_sha256": {"b039": sidecar["sha256"]}}]}
    runner = EvaluationWorkflowRunner(WorkflowOptions(profile="", out=str(tmp_path)))
    runner.run_dir, runner.run_id, runner.profile_payload = active, active.name, {"quality_gate": policy}
    runner._emit_log = lambda *_: None
    return runner, suite, plan, policy, original, old


def test_explicit_reuse_projects_and_uses_real_normal_reference_reader(sample):
    runner, suite, plan, policy, original, old = sample
    before = {p: p.read_bytes() for p in old.iterdir()}
    bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
        model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)
    _write(suite / "benchmark_plan.json", plan)
    runner._schedule_management_cpu_reference("yolo11l", suite)
    records, status, encoded = runner._management_reference_records("yolo11l")
    assert len(records) == 5000
    assert hashlib.sha256(encoded).hexdigest() == original["sha256"]
    assert status["explicit_existing_reference_reuse"] is True
    assert status["status"] == "cache_hit"
    assert Path(status["reference_path"]).is_relative_to(runner.run_dir)
    assert runner._management_reference_executor is None
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("damage", ["missing_reference", "null_reference", "empty_map", "reference_bytes", "model", "dataset",
                                    "bootstrap", "run_preprocess", "config_preprocess", "image_ids"])
def test_explicit_reuse_rejects_drift_before_any_cpu_queue(sample, damage):
    runner, suite, plan, policy, original, old = sample
    if damage == "missing_reference":
        policy["management_reference"].pop("existing_reference")
    elif damage == "null_reference":
        policy["management_reference"]["existing_reference"] = None
    elif damage == "empty_map":
        plan["runs"][0]["generic_quality_completion_configs"] = {}
    elif damage == "reference_bytes":
        Path(original["path"]).write_bytes(b"changed reference")
    elif damage == "model":
        _write(runner.run_dir / "models/yolo11l/model_manifest.json", {"model_sha256": "f" * 64})
    elif damage == "dataset":
        _write(suite / "validation/manifest.json", {"changed": True})
    elif damage == "bootstrap":
        policy["statistics"]["bootstrap_repetitions"] = 5000
    elif damage == "run_preprocess":
        plan["runs"][0]["image_scale"] = "raw"
    else:
        config = json.loads((suite / "quality_runtime/b039.json").read_text())
        if damage == "config_preprocess":
            config["preprocess"]["pad_value"] = 0
        else:
            config["expected_image_ids"] = config["expected_image_ids"][:-1]
        sidecar = _write(suite / "quality_runtime/b039.json", config)
        plan["runs"][0]["generic_quality_completion_config_sha256"]["b039"] = sidecar["sha256"]
    _write(suite / "benchmark_plan.json", plan)
    with pytest.raises((ValueError, RuntimeError)):
        runner._schedule_management_cpu_reference("yolo11l", suite)
    assert runner._management_reference_futures == {}
    assert runner._management_reference_executor is None


def test_ordinary_profile_has_no_explicit_reuse_projection(tmp_path):
    assert bind_existing_generic_cpu_reference(suite_dir=tmp_path, run_root=tmp_path,
        model_id="model", benchmark_plan={"runs": []}, quality_gate_policy={}) is None


@pytest.mark.parametrize("source_drift", [False, True])
def test_explicit_cpu_reference_normal_resume_revalidates_without_generator(sample, source_drift):
    runner, suite, plan, policy, original, old = sample
    bind_existing_generic_cpu_reference(suite_dir=suite, run_root=runner.run_dir,
        model_id="yolo11l", benchmark_plan=plan, quality_gate_policy=policy)
    _write(suite / "benchmark_plan.json", plan)
    if source_drift:
        Path(original["path"]).write_bytes(b"changed original reference")
        with pytest.raises((ValueError, RuntimeError)):
            runner._restore_management_cpu_reference("yolo11l")
        assert runner._management_reference_futures == {}
    else:
        runner._restore_management_cpu_reference("yolo11l")
        records, status, encoded = runner._management_reference_records("yolo11l")
        assert len(records) == 5000
        assert hashlib.sha256(encoded).hexdigest() == original["sha256"]
        assert status["explicit_existing_reference_reuse"] is True
    assert runner._management_reference_executor is None
