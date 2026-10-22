"""Cache-only first publication of preserved Generic evidence; no devices.

The synthetic original lifecycle is closed through the real recovery-group
commit. Candidate export, completion proof, descriptor reads, pairing and the
completed-cache envelope use product code. No evaluator or worker starts.
"""
from __future__ import annotations

from copy import deepcopy
import concurrent.futures
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np
import yaml

from onnx_splitpoint_tool import quality_service
from onnx_splitpoint_tool import runners
from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY, assess_accuracy
from onnx_splitpoint_tool.quality_cache import (
    PersistentQualityCache,
    image_ids_fingerprint,
    json_fingerprint,
)
from onnx_splitpoint_tool.runners import generic_quality_completion as completion
from onnx_splitpoint_tool.workflow import run_control
from onnx_splitpoint_tool.workflow import generic_quality_sidecars
from tests.test_generic_quality_completion import capture, descriptor, make_config
from tests.test_v269d_trt_central_quality_producer import _detection_quality_contract
from tests.test_v27911_composed_provenance_binding import _functions


def _document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return descriptor(path)


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def _recovery(run, session, receipt_path, profile_payload):
    """Commit temporary fixture locks using the existing recovery protocol."""
    manifest_path = run / "run_manifest.json"
    from onnx_splitpoint_tool.execution_plan import _setup_class
    logical_id = profile_payload["run_profiles"][0]["id"]
    plan = {"models": ["yolo26s"], "logical_run_profiles": [logical_id],
            "setup_groups": {_setup_class(logical_id): [logical_id]}}
    _document(run / "effective_execution_plan.json", plan)
    before = {
        "schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": run.name, "run_dir": str(run), "current_session_id": session,
        "status": "running", "technical_status": "running",
        "quality_decision": "not_evaluated",
        "execution_sessions": [{"session_id": session, "status": "running"}],
        "models": {"yolo26s": {"stages": {"quality": {"state": "running"}}}},
        "options": {},
        "resume_contract": run_control.build_resume_contract(
            profile_payload=profile_payload,
            effective_execution_plan=plan,
            options={},
        ),
    }
    _document(manifest_path, before)
    owner = {"run_id": run.name, "session_id": session, "pid": 1000000000,
             "state": "running", "purpose": "generic"}
    lock = run_control.EvaluationRunLock(out_root=run.parent, run_dir=run, owner=owner)
    physical = run_control.EvaluationRunLock.for_resource("dut:192.0.2.99", owner=owner)
    for item in (lock, physical):
        _document(item.lock_path, item.owner)
        item.acquire_for_recovery()
    after = deepcopy(before)
    after.update(status="cancelled", technical_status="cancelled")
    after["execution_sessions"][0]["status"] = "cancelled"
    try:
        run_control.commit_generic_orphan_recovery_group(
            run_lock=lock, physical_locks={"dut:192.0.2.99": physical},
            receipt_path=receipt_path,
            receipt_data={"schema": "onnx-splitpoint/generic-orphan-recovery", "schema_version": 1,
                "recovery_id": "synthetic-committed-recovery", "run_id": run.name,
                "run_dir": str(run), "session_id": session},
            manifest_path=manifest_path, manifest_before=manifest_path.read_bytes(),
            manifest_after=after,
        )
    finally:
        physical.release(write_released_state=False)
        lock.release(write_released_state=False)
    return lock


def _completed_result(key, payload, quality_status):
    """Preserved synthetic v3 completed result, without evaluating anything."""
    reference, candidate = 1.0, (2.0 / 3.0 if quality_status == "accuracy_loss" else 1.0)
    delta = candidate - reference
    loss = (reference - candidate) / reference
    assessment = assess_accuracy(reference, candidate, [loss, loss], policy=DEFAULT_REPORTING_POLICY)
    primary = {
        "metric": payload["metric_gate_config"]["primary_metric"],
        "legacy_decision": "fail" if quality_status == "accuracy_loss" else "pass",
        "status": "completed", "reference": reference, "candidate": candidate, "delta": delta,
        "ci_low": delta, "ci_high": delta, "ci_computed": True,
        "bootstrap_repetitions": 7, "bootstrap_repetitions_requested": 7,
        "decision_basis": "paired_bootstrap_lower_bound", "gate_bound_value": delta,
        "uncertainty_status": "computed_bootstrap", "bootstrap_skipped_reason": "",
        "prediction_identity_verified": False,
    }
    names = quality_service.configured_guardrail_names(payload["metric_gate_config"])
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result", "schema_version": 1,
        "quality_result_contract_version": 3, "guardrail_contract_complete": True,
        "configured_guardrails": list(names),
        "guardrails": {name: {**deepcopy(primary), "metric": name} for name in names},
        "primary": primary,
        "decision": assessment["accuracy_class"], "status": "completed", "technical_status": "completed",
        "accuracy_assessment": assessment, "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY),
        "secondary_accuracy_assessments": {name: deepcopy(assessment) for name in names},
        "legacy_decision": primary["legacy_decision"], "observed_image_ids": payload["image_ids"],
        "cpu_reference_semantic_only": True,
        "evaluation_fingerprint": key, "request_id": payload["request_id"],
        "reference_identity": payload["reference_identity"],
        "candidate_execution_completion_contract_sha256": payload["candidate_execution_completion_contract_sha256"],
        "reference_predictions_sha256": payload["reference_predictions_sha256"],
        "candidate_predictions_sha256": payload["candidate_predictions_sha256"],
        "annotations_sha256": payload["annotations_sha256"],
        "algorithm_version": payload["algorithm_version"], "seed_schema": payload["seed_schema"],
        "n": len(payload["image_ids"]), "evaluated_images": len(payload["image_ids"]),
    }


@pytest.fixture
def preserved_case(tmp_path, monkeypatch, request):
    quality_status = getattr(request, "param", "reference_close")
    monkeypatch.setattr(run_control, "platform_workflow_interlock_path",
                        lambda: tmp_path / "locks/platform.lock")
    run = tmp_path / "runs/original-generic-quality"
    fixture_root = run / "synthetic_inputs"
    fixture_root.mkdir(parents=True)
    fixture = make_config(fixture_root)
    representative_image = fixture["images"][0]
    prepared_path = fixture_root / "representative.npy"
    prepared = fixture["inputs"]["images"]
    np.save(prepared_path, prepared)
    fixture["config"]["representative_input"] = {
        "image_id": representative_image.name,
        "source_image_sha256": completion._sha(representative_image), "original_wh": [80, 60],
        "prepared_input": {**descriptor(prepared_path), "shape": list(prepared.shape),
            "dtype": str(prepared.dtype), "format": "npy"}, "runtime_input_name": "images",
    }
    fixture["path"].write_text(json.dumps(fixture["config"]))
    runtime, detections = capture(fixture)
    evidence = runtime.evidence(detections)
    full = fixture_root / "full-model.onnx"
    full.write_bytes(b"synthetic original full model authority")
    quality = _detection_quality_contract()
    quality["model"]["sha256"] = completion._sha(full)
    ids = sorted(detections)
    ground_truth = {image_id: deepcopy(detections[image_id]) for image_id in ids}
    if quality_status == "accuracy_loss":
        for values in ground_truth.values():
            values.append({"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0,
                "score": 1.0, "class_id": 79, "class_name": "synthetic_class_79"})
    quality["dataset"].update(
        image_count=len(ids), image_ids_sha256=image_ids_fingerprint(ids),
        ground_truth_sha256=json_fingerprint([
            {"image_id": image_id, "ground_truth": ground_truth[image_id]} for image_id in ids]),
    )
    quality.pop("quality_contract_sha256")
    quality["quality_contract_sha256"] = json_fingerprint(quality)
    ns = _functions([
        "_task_quality_execution_location", "_quality_json_safe", "_write_stable_quality_json",
        "_quality_file_sha256", "_quality_contract_sha256", "_quality_artifact_binding",
        "_build_generic_composed_quality_producer_binding", "_export_central_quality_inputs",
        "_load_task_quality_policy",
    ])
    raw_policy = {
        "reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY),
        "detection": {"primary_metric": "coco_ap_50_95", "guardrails": {}},
        "statistics": {"method": "paired_bootstrap", "execution_location": "central_management",
            "bootstrap_repetitions": 7, "seed": 7, "confidence_level": 0.95,
            "decision": "lower_one_sided_bound"},
    }
    policy = ns["_load_task_quality_policy"](raw_policy)
    profile = run / "profile.yaml"
    _document(run / "models/yolo26s/model_manifest.json", {
        "model_id": "yolo26s", "task": "detection", "model_sha256": completion._sha(full),
    })
    ns["_verified_suite_vendored_endpoint_attestor_sha256"] = lambda: quality["quality_record_endpoint"]["identity"]["vendored_endpoint_attestor_sha256"]
    monkeypatch.setitem(sys.modules, "splitpoint_runners", runners)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_quality_completion", completion)
    context = dict(fixture["context"], eval_run_id=run.name, source_run_id="hailo10_to_tensorrt",
        stage1_backend="hailo10h", stage2_backend="tensorrt", source_model_path=str(full),
        implementation_runner_path=evidence["implementation_runner"]["path"])
    for row in context["runtime_artifacts"]:
        row["backend"] = "hailo10h" if row["role"] == "stage1_runtime" else "native_tensorrt"
    output = run / "models/yolo26s/benchmark_results/quality_inputs/test_only_h10/b039/results_hailo10h_to_trt"
    exported = ns["_export_central_quality_inputs"](
        out_dir=output, task="detection", variant="composed", policy=policy,
        gt_by_image=ground_truth, candidate_by_image=detections,
        quality_contract=quality, endpoint_contract=fixture["config"]["source_endpoint_contract"],
        runtime_precision_identity=fixture["config"]["identity"]["precision"],
        composed_producer_context=context, generic_completed_quality_evidence=evidence,
    )
    request_path = Path(exported["request"]["path"])
    request = json.loads(request_path.read_text())
    source_contract = hashlib.sha256(b"synthetic original suite source contract").hexdigest()
    reference_path = run / "quality_management/references/yolo26s/by_source_contract" / source_contract / "canonical_cpu_reference.json"
    reference = _document(reference_path, {
        "schema": "onnx-splitpoint/task-quality-reference-input", "schema_version": 1,
        "task": "detection", "pairing_key": "image_id", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "provenance_required": True,
        "quality_contract": quality, "quality_contract_sha256": quality["quality_contract_sha256"],
        "records": [{"image_id": image_id, "ground_truth": ground_truth[image_id], "reference": ground_truth[image_id]}
                    for image_id in ids],
    })
    status_path = tmp_path / "external/management_cpu_reference_status.json"
    reference_status = {
        "schema": "onnx-splitpoint/management-cpu-reference-job", "schema_version": 1,
        "status": "completed", "model_id": "yolo26s", "task": "detection",
        "provider": "onnxruntime_cpu", "execution_location": "central_management",
        "semantic_reference_only": True, "include_in_latency_fps_energy": False,
        "include_in_ranking": False, "include_in_pareto": False,
        "reference_path": str(reference_path), "reference_sha256": reference["sha256"],
        "reference_size_bytes": reference["size_bytes"], "reference_storage": "immutable_source_contract",
        "reference_immutable": True, "source_contract_sha256": source_contract,
        "record_count": len(ids), "image_ids_sha256": image_ids_fingerprint(ids),
        "quality_contract": quality, "quality_contract_sha256": quality["quality_contract_sha256"],
    }
    source_status = _document(status_path, reference_status)
    reference["source_contract_sha256"] = source_contract
    policy_source = _document(tmp_path / "external/policy.json", {"quality_gate": raw_policy})
    saved_policy = deepcopy(policy)
    saved_policy["management_reference"] = {"existing_reference": {
        "reference_path": str(reference_path), "reference_sha256": reference["sha256"],
        "reference_size_bytes": reference["size_bytes"], "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "source_evaluation_run_id": "synthetic-reference-source",
        "source_status": source_status, "policy_source": policy_source,
        "quality_contract": quality, "quality_contract_sha256": quality["quality_contract_sha256"],
    }}
    profile_payload = {"name": "synthetic recovered publication", "quality_gate": saved_policy,
        "validation_execution": {"max_items": {"detection": len(ids)}},
        "hardware": {"selected_setups": ["test_only_h10"]},
        "model_suite": {"primary": [{"id": "yolo26s"}]},
        "run_profiles": [{"id": "hailo10_to_tensorrt", "hardware_setup_id": "test_only_h10",
            "quality_source_only": True, "variants": ["composed"], "case_ids": ["b039"],
            "generic_quality_completion_configs": {"b039": "quality_runtime/b039.json"},
            "generic_quality_completion_config_sha256": {"b039": completion._sha(fixture["path"])},
            "generic_quality_completion_sources": {"b039": descriptor(fixture["path"])}}],
    }
    profile.write_text(yaml.safe_dump(profile_payload))
    suite = run / "models/yolo26s/benchmark_set/legacy_suite"
    (suite / "quality_runtime").mkdir(parents=True)
    (suite / "quality_runtime/b039.json").write_bytes(fixture["path"].read_bytes())
    _document(suite / "benchmark_plan.json", {"model_id": "yolo26s",
        "runs": deepcopy(profile_payload["run_profiles"]),
        "management_cpu_reference": {**saved_policy["management_reference"]["existing_reference"],
            "reference_path": str(reference_path)},
    })
    loaded = quality_service.quality_request_from_manifest(request_path, reference_artifact=reference)
    key, payload = quality_service.prepare_evaluation(loaded)
    assert (payload["reference_predictions_sha256"] == payload["candidate_predictions_sha256"]) == (quality_status == "reference_close")
    cache_root = tmp_path / "completed-cache"
    cache = PersistentQualityCache(cache_root)
    cached = _completed_result(key, payload, quality_status)
    cache_path = cache.put(key, cached)
    session = "a" * 32
    receipt_path = tmp_path / "recovery/receipt.json"
    lock = _recovery(run, session, receipt_path, profile_payload)
    authorization = _document(tmp_path / "authorization.json", {
        "schema": "onnx-splitpoint/recovered-quality-publication-authorization", "schema_version": 1,
        "approved": True, "action": "publish_preserved_completed_quality",
        "run_id": run.name, "session_id": session,
        "request_sha256s": [completion._sha(request_path)],
        "profile_sha256": completion._sha(profile), "reference_sha256": reference["sha256"],
        "recovery_receipt_sha256": completion._sha(receipt_path),
        "authorized_at": "2026-10-22T12:00:00Z", "provenance": "synthetic explicit test authorization",
    })
    fps = run / "models/yolo26s/benchmark_results/preserved_fps.json"
    _document(fps, {"status": "completed", "FPS": 123.0, "original_measurement": True})
    # Existing-reference validation has its own 5,000-image suite. This seam
    # keeps this focused fixture at two images; the shared sealed status
    # reader, artifact loader, model authority and request identity stay real.
    from onnx_splitpoint_tool.management_reference import _successful_reference_status
    adopted_reference_status = _successful_reference_status(
        status="cache_hit", model_id="yolo26s", workers=1,
        source_contract_sha256="sha256:" + source_contract,
        reference_path=reference_path, reference_sha256="sha256:" + reference["sha256"],
        reference_metadata={key: reference_status[key] for key in
            ("task", "record_count", "image_ids_sha256", "quality_contract", "quality_contract_sha256")},
        extra={"explicit_existing_reference_reuse": True})
    monkeypatch.setattr(generic_quality_sidecars, "bind_existing_generic_cpu_reference",
        lambda **kwargs: deepcopy(adopted_reference_status))
    return SimpleNamespace(
        run=run, fixture=fixture, request=request, request_path=request_path,
        reference=reference, reference_status=reference_status, source_contract=source_contract,
        adopted_reference_status=adopted_reference_status,
        quality_status=quality_status,
        profile=profile, receipt_path=receipt_path, lock=lock, cache=cache, cache_path=cache_path,
        cached=cached, key=key, payload=payload, authorization=authorization,
        kwargs={"original_run_dir": run, "recovery_receipt": descriptor(receipt_path),
            "request_descriptors": [descriptor(request_path)], "completed_cache_root": cache_root,
            "canonical_reference_descriptor": reference, "profile_descriptor": descriptor(profile),
            "authorization_descriptor": authorization,
            "out_dir": run / "quality_management/recovered_publications/test-publication"},
    )


def _no_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preserved completion admission must not start service/evaluation/workers/bootstrap")

    monkeypatch.setattr(quality_service.ManagementQualityService, "__init__", forbidden)
    monkeypatch.setattr(quality_service.ManagementQualityService, "evaluate", forbidden)
    monkeypatch.setattr(quality_service, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(multiprocessing, "Pool", forbidden)
    monkeypatch.setattr(quality_service, "_evaluate_payload", forbidden)
    from onnx_splitpoint_tool import quality_statistics_blocks
    monkeypatch.setattr(quality_statistics_blocks, "prepare_plan", forbidden)


def _admit(case):
    from onnx_splitpoint_tool.recovered_quality_admission import admit_preserved_completed_quality
    return admit_preserved_completed_quality(**case.kwargs)


def _original_unchanged(case, before):
    current = _snapshot(case.run)
    assert {name: current[name] for name in before} == before
    publication = case.kwargs["out_dir"]
    assert all((case.run / name).is_relative_to(publication)
               or name == "quality_management/central_quality_summary.json"
               for name in current.keys() - before.keys())


def test_first_publication_uses_actual_preserved_completed_cache_without_runtime(preserved_case, monkeypatch):
    case = preserved_case
    _no_runtime(monkeypatch)
    before = _snapshot(case.run)
    cache_before = _snapshot(case.cache.root)
    receipt_before = case.receipt_path.read_bytes()
    lock_before = case.lock.lock_path.read_bytes()
    assert not (case.run / "quality_management/central_quality_summary.json").exists()
    published = _admit(case)
    assert {"summary", "results", "publication_receipt"}.issubset(published)
    for bound in [published["summary"], *published["results"], published["publication_receipt"]]:
        assert descriptor(bound["path"]) == bound
        assert Path(bound["path"]).is_relative_to(case.kwargs["out_dir"])
    summary = json.loads(Path(published["summary"]["path"]).read_text())
    assert summary["schema"] == "onnx-splitpoint/central-quality-summary"
    assert summary["publication_scope"] == "recovered_completed_quality"
    assert summary["original_run_status"] == "cancelled"
    assert len(published["results"]) == 1
    result = json.loads(Path(published["results"][0]["path"]).read_text())
    assert result["technical_status"] == "completed"
    assert result["cache_hit"] is True
    assert result["source_request_sha256"].removeprefix("sha256:") == completion._sha(case.request_path)
    assert result["eval_run_id"] == case.request["producer_identity"]["eval_run_id"] == case.run.name
    assert result["producer_identity"] == case.request["producer_identity"]
    assert result["evaluation_fingerprint"] == case.key
    assert result["producer_binding_eligible"] is True
    assert result["primary"] == case.cached["primary"]
    assert result["guardrails"] == case.cached["guardrails"]
    assert result["performance_claims_emitted"] is False
    publication = json.loads(Path(published["publication_receipt"]["path"]).read_text())
    assert publication["state"] == "committed"
    assert publication["statistics_recomputed"] is False
    assert publication["hardware_dispatched"] is False
    assert publication["cache_sources"] == [descriptor(case.cache_path)]
    _original_unchanged(case, before)
    assert _snapshot(case.cache.root) == cache_before
    assert case.receipt_path.read_bytes() == receipt_before
    assert case.lock.lock_path.read_bytes() == lock_before
    canonical = case.run / "quality_management/central_quality_summary.json"
    if canonical.exists():
        assert json.loads(canonical.read_text()) == summary
        assert summary["original_evaluation_run_completed"] is False


@pytest.mark.parametrize("preserved_case,eligible", [
    ("reference_close", True), ("accuracy_loss", False),
], indirect=["preserved_case"])
def test_new_publication_is_accepted_by_normal_generic_quality_reader(preserved_case, monkeypatch, eligible):
    from onnx_splitpoint_tool.generic_only_plan import validate_quality_reference

    case = preserved_case
    _no_runtime(monkeypatch)
    before = _snapshot(case.run)
    published = _admit(case)
    result = json.loads(Path(published["results"][0]["path"]).read_text())
    copied_request = descriptor(case.run / result["source_request"])
    assert copied_request["sha256"] == completion._sha(case.request_path)
    config = case.fixture["config"]
    representative = config["representative_input"]
    row = {
        "task": "detection",
        "identity": {**config["identity"],
            "output_endpoint_id": config["completion_contract"]["source_endpoint"]["output_endpoint_id"]},
        "quality_reference": {"request": copied_request, "result": published["results"][0],
            "cpu_reference": case.reference, "policy_source": descriptor(case.profile),
            "policy_sha256": case.request["policy_sha256"],
            "dataset_manifest_sha256": case.request["quality_contract"]["dataset"]["manifest_sha256"]},
        "completion_contract": config["completion_contract"],
        "generic_artifact_binding": config["generic_artifact_binding"],
        "boundary_observation": config.get("boundary_observation", {}),
        "input_binding": {"prepared_input": representative["prepared_input"],
            "source_image_sha256": representative["source_image_sha256"]},
    }
    admitted = validate_quality_reference(row)
    assert admitted["status"] == case.quality_status
    assert admitted["quality_eligible"] is eligible
    assert admitted["request_sha256"] == copied_request["sha256"]
    assert admitted["result_sha256"] == published["results"][0]["sha256"]
    assert admitted["completed_quality_binding"]["all_actual_completed_records_verified"] is True
    _original_unchanged(case, before)
    summary = json.loads(Path(published["summary"]["path"]).read_text())
    assert summary["schema"] == "onnx-splitpoint/central-quality-summary"
    assert summary["publication_scope"] == "recovered_completed_quality"
    assert summary["original_run_status"] == "cancelled"
    assert json.loads((case.run / "run_manifest.json").read_text())["status"] == "cancelled"


@pytest.mark.parametrize("fault", [
    "request_descriptor", "reference_descriptor", "profile_descriptor", "cache_miss",
    "cache_checksum", "cache_source_request", "cache_eval_run", "source_model",
])
def test_mismatched_or_missing_evidence_fails_closed(preserved_case, monkeypatch, fault):
    case = preserved_case
    _no_runtime(monkeypatch)
    if fault == "request_descriptor":
        case.kwargs["request_descriptors"][0]["sha256"] = "f" * 64
    elif fault == "reference_descriptor":
        case.kwargs["canonical_reference_descriptor"]["sha256"] = "f" * 64
    elif fault == "profile_descriptor":
        case.kwargs["profile_descriptor"]["sha256"] = "f" * 64
    elif fault == "cache_miss":
        # Only the test-owned synthetic completed entry is withdrawn.
        case.cache_path.unlink()
    elif fault == "cache_checksum":
        envelope = json.loads(case.cache_path.read_text())
        envelope["result_sha256"] = "f" * 64
        case.cache_path.write_text(json.dumps(envelope))
    elif fault in {"cache_source_request", "cache_eval_run"}:
        value = deepcopy(case.cached)
        value["source_request_sha256" if fault == "cache_source_request" else "eval_run_id"] = "f" * 64
        case.cache.put(case.key, value)
    else:
        _document(case.run / "models/yolo26s/model_manifest.json", {
            "model_id": "yolo26s", "task": "detection", "model_sha256": "f" * 64,
        })
    before = _snapshot(case.run)
    cache_before = _snapshot(case.cache.root)
    lock_before = case.lock.lock_path.read_bytes()
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
        _admit(case)
    _original_unchanged(case, before)
    assert _snapshot(case.cache.root) == cache_before
    assert case.lock.lock_path.read_bytes() == lock_before
    assert not list(case.kwargs["out_dir"].rglob("central_quality_summary.json"))
    assert not (case.run / "quality_management/central_quality_summary.json").exists()


def test_active_original_run_lock_blocks_publication(preserved_case, monkeypatch):
    case = preserved_case
    _no_runtime(monkeypatch)
    case.lock.acquire_for_recovery()
    before = _snapshot(case.run)
    try:
        with pytest.raises(run_control.WorkflowRunLockedError):
            _admit(case)
    finally:
        case.lock.release(write_released_state=False)
    _original_unchanged(case, before)


def test_recovery_receipt_from_another_run_is_rejected(preserved_case, monkeypatch):
    case = preserved_case
    _no_runtime(monkeypatch)
    foreign = case.run.parent / "foreign-original"
    foreign.mkdir()
    foreign_receipt = case.receipt_path.parent / "foreign-receipt.json"
    foreign_lock = _recovery(foreign, "b" * 32, foreign_receipt, yaml.safe_load(case.profile.read_text()))
    case.kwargs["recovery_receipt"] = descriptor(foreign_receipt)
    authorization_path = Path(case.authorization["path"])
    authorization = json.loads(authorization_path.read_text())
    authorization["recovery_receipt_sha256"] = completion._sha(foreign_receipt)
    case.kwargs["authorization_descriptor"] = _document(authorization_path, authorization)
    before = _snapshot(case.run)
    foreign_before = _snapshot(foreign)
    foreign_lock_before = foreign_lock.lock_path.read_bytes()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _admit(case)
    _original_unchanged(case, before)
    assert _snapshot(foreign) == foreign_before
    assert foreign_lock.lock_path.read_bytes() == foreign_lock_before


@pytest.mark.parametrize("prefix", ["", "sha256:"])
def test_normal_adoption_status_digest_is_joined_after_strict_normal_reader(preserved_case, monkeypatch, prefix):
    case = preserved_case
    _no_runtime(monkeypatch)
    assert case.adopted_reference_status["reference_sha256"] == "sha256:" + case.reference["sha256"]
    status = {**case.adopted_reference_status, "reference_sha256": prefix + case.reference["sha256"]}
    monkeypatch.setattr(generic_quality_sidecars, "bind_existing_generic_cpu_reference", lambda **kwargs: status)
    published = _admit(case)
    result = json.loads(Path(published["results"][0]["path"]).read_text())
    assert result["management_cpu_reference"]["reference_sha256"] == case.reference["sha256"]
    assert result["management_cpu_reference"]["reference_path"] == case.reference["path"]


@pytest.mark.parametrize("fault", ["digest", "path"])
def test_adopted_reference_mismatch_still_rejected_before_publication(preserved_case, monkeypatch, fault):
    case = preserved_case
    _no_runtime(monkeypatch)
    status = deepcopy(case.adopted_reference_status)
    if fault == "digest":
        status["reference_sha256"] = "sha256:" + "f" * 64
    else:
        status["reference_path"] = str(case.run / "foreign/canonical_cpu_reference.json")
    monkeypatch.setattr(generic_quality_sidecars, "bind_existing_generic_cpu_reference", lambda **kwargs: status)
    before = _snapshot(case.run)
    with pytest.raises((ValueError, RuntimeError)):
        _admit(case)
    _original_unchanged(case, before)
    assert not case.kwargs["out_dir"].exists()
    assert not (case.run / "quality_management/central_quality_summary.json").exists()


@pytest.mark.parametrize("group", ["deepx_setup", "orin_nx_deepx_m1_01"])
def test_original_normal_logical_setup_group_and_exact_physical_group_are_supported(group):
    from onnx_splitpoint_tool.recovered_quality_admission import _validate_original_setup_plan_binding
    # The actual interrupted original plan uses the logical key deepx_setup,
    # while its source-only profile binds the physical id independently.
    logical = {"id": "deepx_m1_to_tensorrt", "hardware_setup_id": "orin_nx_deepx_m1_01"}
    plan = {"logical_run_profiles": [logical["id"]], "setup_groups": {group: [logical["id"]]}}
    _validate_original_setup_plan_binding(logical, plan, logical["hardware_setup_id"])


@pytest.mark.parametrize("declared,groups", [
    (["deepx_m1_to_tensorrt"], {}),
    (["deepx_m1_to_tensorrt"], {"hailo8_setup": ["deepx_m1_to_tensorrt"]}),
    (["deepx_m1_to_tensorrt"], {"foreign_physical_setup": ["deepx_m1_to_tensorrt"]}),
    (["deepx_m1_to_tensorrt"], {"deepx_setup": ["deepx_m1_to_tensorrt", "deepx_m1_to_tensorrt"]}),
    (["deepx_m1_to_tensorrt"], {"deepx_setup": ["deepx_m1_to_tensorrt"],
                                  "orin_nx_deepx_m1_01": ["deepx_m1_to_tensorrt"]}),
    (["deepx_m1_to_tensorrt"], {"deepx_setup": ["deepx_m1_to_tensorrt", "foreign_run"]}),
    (["deepx_m1_to_tensorrt", "missing_run"], {"deepx_setup": ["deepx_m1_to_tensorrt"]}),
    (["deepx_m1_to_tensorrt", "deepx_m1_to_tensorrt"], {"deepx_setup": ["deepx_m1_to_tensorrt"]}),
    (["foreign_run"], {"deepx_setup": ["foreign_run"]}),
    (["deepx_m1_to_tensorrt"], {"deepx_setup": "deepx_m1_to_tensorrt"}),
    (["deepx_m1_to_tensorrt"], {"deepx_setup": []}),
    (None, {"deepx_setup": ["deepx_m1_to_tensorrt"]}),
])
def test_original_setup_plan_join_rejects_foreign_missing_ambiguous_or_malformed_mapping(declared, groups):
    from onnx_splitpoint_tool.recovered_quality_admission import _validate_original_setup_plan_binding
    with pytest.raises(ValueError, match="original setup/plan binding mismatch"):
        _validate_original_setup_plan_binding({"id": "deepx_m1_to_tensorrt"},
            {"logical_run_profiles": declared, "setup_groups": groups}, "orin_nx_deepx_m1_01")
