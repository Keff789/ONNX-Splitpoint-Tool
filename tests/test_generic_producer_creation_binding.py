"""Neutral Generic producer creation authority; all files are temporary fixtures.

The normal snapshot factory, immutable-source reader and ordinary two-image
Quality export/loader stay real. No service, worker or hardware execution starts.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

from onnx_splitpoint_tool import generic_only_plan as admission
from onnx_splitpoint_tool import generic_producer_binding as producer
from onnx_splitpoint_tool.workflow import generic_completed_task as completed
from onnx_splitpoint_tool.workflow.run_control import build_resume_contract
from onnx_splitpoint_tool.workflow.start_snapshot import (
    build_profile_start_snapshot,
    public_start_snapshot_metadata,
)


def _document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return _descriptor(path)


def _descriptor(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def _change(path, mutation):
    value = json.loads(path.read_text())
    mutation(value)
    _document(path, value)


def _normal_creation(primary, *, setup="test_only_h10", profile_extra=None):
    """Write the four ordinary creation files using the production factory."""
    primary.mkdir(parents=True, exist_ok=True)
    session = "a" * 32
    targets = [{"id": setup, "label": "temporary fixture H10",
                "remote": {"host": "192.0.2.10", "user": "fixture", "port": 22}}]
    profile = {"name": "normal Generic source-only fixture",
        "model_suite": {"primary": [{"id": "yolo26s", "task": "detection", "enabled": True}]},
        "run_profiles": [{"id": "hailo10h_to_trt", "hardware_setup_id": setup,
            "quality_source_only": True, "variants": ["composed"], "case_ids": ["b039"]}],
        "hardware": {"selected_setups": [setup]},
        "energy": {"enabled": False},
        "native_producers": {"enabled": False, "energy": {"enabled": False}}}
    profile.update(deepcopy(profile_extra or {}))
    snapshot = build_profile_start_snapshot(profile_request="normal-generic-fixture",
        source_profile=profile, resolved_profile=profile, profile_id="normal-generic-fixture",
        profile_path=str(primary / "profile_source.yaml"), profile_source="file",
        runtime_bindings={"hardware_targets": targets})
    metadata = public_start_snapshot_metadata(snapshot)
    for name in ("profile_source.yaml", "profile.yaml"):
        (primary / name).write_text(yaml.safe_dump(profile, sort_keys=True))
    _document(primary / "profile_start_snapshot.json", metadata)
    plan = snapshot["effective_execution_plan"]
    _document(primary / "effective_execution_plan.json", plan)
    contract = build_resume_contract(profile_payload=profile,
        effective_execution_plan=plan, options={})
    manifest = {"schema": "onnx-splitpoint/evaluation-run-manifest", "schema_version": 1,
        "run_id": primary.name, "run_dir": str(primary), "current_session_id": session,
        "status": "running", "technical_status": "running", "updated_at": "before",
        "options": {}, "resume_contract": contract, "hardware_targets": targets,
        "profile_start_snapshot": metadata,
        "execution_sessions": [{"session_id": session, "status": "running",
            "profile_sha256": contract["profile_sha256"], "plan_sha256": contract["plan_sha256"]}],
        "models": {"yolo26s": {"stages": {"quality": {"state": "running"}}}}}
    _document(primary / "run_manifest.json", manifest)
    _document(primary / "jobs/workflow_control.json", {
        "schema": "onnx-splitpoint/workflow-control", "schema_version": 1,
        "run_id": primary.name, "session_id": session, "state": "running",
        "cancel_requested": False, "updated_at": "before"})
    return SimpleNamespace(primary=primary, session=session, targets=targets,
        profile=profile, snapshot=metadata, manifest=manifest, contract=contract)


@pytest.fixture
def creation(tmp_path):
    fixture = _normal_creation(tmp_path / "producer")
    fixture.binding = producer.build_producer_run_binding(fixture.primary, session_id=fixture.session)
    return fixture


def test_build_reads_genuine_normal_creation_files_without_a_manifest_digest(creation):
    binding = creation.binding
    assert binding["mode"] == "creation_snapshot_v1"
    assert binding["run_id"] == creation.primary.name
    assert binding["run_dir"] == str(creation.primary)
    assert binding["session_id"] == creation.session
    for role, name in (("creation_snapshot", "profile_start_snapshot.json"),
                       ("source_profile", "profile_source.yaml"), ("profile", "profile.yaml"),
                       ("execution_plan", "effective_execution_plan.json")):
        assert binding[role]["path"] == str(creation.primary / name)
        assert binding[role]["sha256"] == _descriptor(creation.primary / name)["sha256"]
    assert "producer_run_manifest" not in binding
    assert producer.validate_producer_run_binding(creation.primary, binding) == {
        "run_id": creation.primary.name, "session_id": creation.session,
        "hardware_targets": creation.targets}


def _advance_lifecycle(creation):
    def advance(manifest):
        manifest.update(status="partial", technical_status="completed", updated_at="later")
        manifest["execution_sessions"][-1].update(status="partial", completed_at="later")
        manifest["models"]["yolo26s"]["stages"]["quality"].update(state="completed", updated_at="later")
    _change(creation.primary / "run_manifest.json", advance)
    _change(creation.primary / "jobs/workflow_control.json",
        lambda value: value.update(state="completed", reason="normal progress", updated_at="later"))


def test_lifecycle_changes_are_allowed_even_after_the_final_consumer(creation):
    with admission.validation_scope():
        first = producer.validate_producer_run_binding(creation.primary, creation.binding)
        _advance_lifecycle(creation)
        assert producer.validate_producer_run_binding(creation.primary, creation.binding) == first
        _advance_lifecycle(creation)
    assert producer.build_producer_run_binding(creation.primary, session_id=creation.session) == creation.binding


_CRITICAL_MANIFEST_MUTATIONS = [
    pytest.param(lambda value: value.update(schema="unrelated"), id="schema"),
    pytest.param(lambda value: value.update(schema_version=999), id="schema-version"),
    pytest.param(lambda value: value.update(run_id="another-run"), id="run-id"),
    pytest.param(lambda value: value.update(run_dir="/another/run"), id="run-directory"),
    pytest.param(lambda value: value.update(current_session_id="b" * 32), id="current-session"),
    pytest.param(lambda value: value["execution_sessions"][-1].update(session_id="b" * 32), id="latest-session"),
    pytest.param(lambda value: value["execution_sessions"][-1].update(profile_sha256="f" * 64), id="session-profile"),
    pytest.param(lambda value: value["execution_sessions"][-1].update(plan_sha256="f" * 64), id="session-plan"),
    pytest.param(lambda value: value["hardware_targets"][0]["remote"].update(host="192.0.2.11"), id="host"),
    pytest.param(lambda value: value["hardware_targets"][0]["remote"].update(user="different"), id="user"),
    pytest.param(lambda value: value["profile_start_snapshot"].update(snapshot_sha256="sha256:" + "f" * 64), id="snapshot"),
    pytest.param(lambda value: value["resume_contract"].update(profile_sha256="f" * 64), id="resume-contract"),
    pytest.param(lambda value: value["options"].update(changed_execution_option=True), id="options"),
]


@pytest.mark.parametrize("mutation", _CRITICAL_MANIFEST_MUTATIONS)
@pytest.mark.parametrize("after_last_consumer", [False, True])
def test_critical_manifest_changes_never_escape_validation(creation, mutation, after_last_consumer):
    with pytest.raises(ValueError):
        with admission.validation_scope():
            producer.validate_producer_run_binding(creation.primary, creation.binding)
            _change(creation.primary / "run_manifest.json", mutation)
            if not after_last_consumer:
                producer.validate_producer_run_binding(creation.primary, creation.binding)


@pytest.mark.parametrize("field,value", [("schema", "unrelated"), ("schema_version", 999),
    ("run_id", "another-run"), ("session_id", "b" * 32)])
def test_control_identity_changes_after_the_last_consumer_fail(creation, field, value):
    with pytest.raises(ValueError):
        with admission.validation_scope():
            producer.validate_producer_run_binding(creation.primary, creation.binding)
            _change(creation.primary / "jobs/workflow_control.json", lambda item: item.update({field: value}))


@pytest.mark.parametrize("name", ["profile_start_snapshot.json", "profile_source.yaml", "profile.yaml",
                                  "effective_execution_plan.json"])
def test_missing_actual_creation_file_is_not_replaced_from_mutable_manifest(creation, name):
    (creation.primary / name).unlink()
    with pytest.raises((ValueError, OSError)):
        producer.build_producer_run_binding(creation.primary, session_id=creation.session)


@pytest.mark.parametrize("role", ["creation_snapshot", "source_profile", "profile", "execution_plan"])
def test_final_full_hash_detects_immutable_drift_even_with_hidden_version(creation, monkeypatch, role):
    path = Path(creation.binding[role]["path"])
    original_version = admission._file_version
    initial = original_version(path)
    monkeypatch.setattr(admission, "_file_version", lambda value:
        initial if Path(value).resolve() == path.resolve() else original_version(value))
    with pytest.raises(ValueError):
        with admission.validation_scope():
            producer.validate_producer_run_binding(creation.primary, creation.binding)
            path.write_bytes(path.read_bytes() + b" ")


def test_wrong_explicit_session_and_sibling_creation_authority_fail(creation, tmp_path):
    with pytest.raises(ValueError):
        producer.build_producer_run_binding(creation.primary, session_id="b" * 32)
    sibling = _normal_creation(tmp_path / "sibling")
    wrong = deepcopy(creation.binding)
    wrong["creation_snapshot"] = _descriptor(sibling.primary / "profile_start_snapshot.json")
    with pytest.raises(ValueError):
        producer.validate_producer_run_binding(creation.primary, wrong)


def test_snapshot_metadata_must_rebuild_from_both_normal_profiles(creation):
    path = creation.primary / "profile_start_snapshot.json"
    _change(path, lambda value: value.update(selection_fingerprint="sha256:" + "f" * 64))
    changed = json.loads(path.read_text())
    _change(creation.primary / "run_manifest.json", lambda value: value.update(profile_start_snapshot=changed))
    with pytest.raises(ValueError):
        producer.build_producer_run_binding(creation.primary, session_id=creation.session)


def test_group_identity_is_stable_across_lifecycle_and_json_order(creation):
    expected = producer.producer_group_identity(creation.binding)
    assert expected == producer.producer_group_identity(dict(reversed(list(creation.binding.items()))))
    _advance_lifecycle(creation)
    current = producer.build_producer_run_binding(creation.primary, session_id=creation.session)
    assert expected == producer.producer_group_identity(current)
    other = deepcopy(current)
    other["session_id"] = "b" * 32
    assert producer.producer_group_identity(other) != expected


def _creation_plan(tmp_path, monkeypatch):
    from test_generic_only_plan import scope_fixture
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    fixture = _normal_creation(primary, setup=row["setup_id"])
    fixture.binding = producer.build_producer_run_binding(primary, session_id=fixture.session)
    plan.pop("producer_run_manifest")
    plan["producer_run_binding"] = deepcopy(fixture.binding)
    row["producer_run_binding"] = deepcopy(fixture.binding)
    return plan, scope, fixture, row


def test_explicit_new_plan_and_rows_use_creation_authority(tmp_path, monkeypatch):
    plan, scope, creation, row = _creation_plan(tmp_path, monkeypatch)
    with admission.validation_scope():
        assert completed.validate_plan(plan, scope, creation.primary) == [row]
        targets = completed.bound_targets(creation.primary, [row], producer_run_binding=creation.binding)
        assert targets[row["setup_id"]]["remote"]["host"] == "192.0.2.10"
        _advance_lifecycle(creation)


@pytest.mark.parametrize("fault", ["missing-row", "row-session", "legacy-also", "row-only"])
def test_new_plan_binding_cannot_be_omitted_or_mixed_with_legacy(tmp_path, monkeypatch, fault):
    plan, scope, creation, row = _creation_plan(tmp_path, monkeypatch)
    if fault == "missing-row":
        row.pop("producer_run_binding")
    elif fault == "row-session":
        row["producer_run_binding"]["session_id"] = "b" * 32
    elif fault == "legacy-also":
        plan["producer_run_manifest"] = _descriptor(creation.primary / "run_manifest.json")
    else:
        plan.pop("producer_run_binding")
        plan["producer_run_manifest"] = _descriptor(creation.primary / "run_manifest.json")
    with pytest.raises(ValueError):
        completed.validate_plan(plan, scope, creation.primary)


def test_legacy_plan_keeps_its_whole_manifest_authority(tmp_path, monkeypatch):
    from test_generic_only_plan import scope_fixture
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    assert "producer_run_binding" not in plan and "producer_run_binding" not in row
    assert completed.validate_plan(plan, scope, primary) == [row]
    _change(primary / "run_manifest.json", lambda value: value.update(status="progressed"))
    with pytest.raises(ValueError, match="changed"):
        completed.validate_plan(plan, scope, primary)


def _case_metadata(creation):
    identity = {"model_id": "yolo26s", "case_id": "b039", "setup_id": creation.targets[0]["id"],
                "eval_run_id": creation.primary.name, "source_run_id": "hailo10h_to_trt"}
    directory = creation.primary / "models/yolo26s/benchmark_results/quality_inputs" / identity["setup_id"] / "b039/results_hailo10h_to_trt/task_quality_inputs"
    request_path = directory / "detection_composed_request.json"
    request = {**identity, "schema": "onnx-splitpoint/central-quality-evaluation-request", "schema_version": 1,
        "task": "detection", "variant": "composed", "execution_role": "generic_composed_quality_candidate",
        "producer_identity": deepcopy(identity)}
    request_descriptor = _document(request_path, request)
    result = {**identity, "schema": "onnx-splitpoint/management-paired-quality-result", "schema_version": 1,
        "task": "detection", "variant": "composed", "execution_role": "generic_composed_quality_candidate",
        "status": "completed", "technical_status": "completed", "collection_eval_run_id": creation.primary.name,
        "producer_identity": deepcopy(identity), "source_request": request_path.relative_to(creation.primary).as_posix(),
        "source_request_sha256": request_descriptor["sha256"]}
    result_path = request_path.with_name("detection_composed_central_result.json")
    _document(result_path, result)
    return request_path, result_path, result


def test_case_loader_binds_exact_original_normal_result_sibling(creation):
    _, path, expected = _case_metadata(creation)
    value, descriptor = producer.load_completed_case_result(creation.primary, path,
        producer_run_binding=creation.binding)
    assert value == expected
    assert descriptor["path"] == str(path)
    assert descriptor["sha256"] == _descriptor(path)["sha256"]


@pytest.mark.parametrize("field,value", [("eval_run_id", "other"), ("collection_eval_run_id", "other"),
    ("status", "running"), ("technical_status", "failed"), ("schema", "unrelated"),
    ("model_id", "different-model"), ("case_id", "b099"), ("setup_id", "unselected"),
    ("source_request_sha256", "f" * 64), ("variant", "full"), ("task", "classification")])
def test_case_loader_rejects_wrong_original_identity_or_incomplete_result(creation, field, value):
    _, path, _ = _case_metadata(creation)
    _change(path, lambda item: item.update({field: value}))
    with pytest.raises(ValueError):
        producer.load_completed_case_result(creation.primary, path, producer_run_binding=creation.binding)


@pytest.mark.parametrize("fault", ["wrong-sibling", "absolute-request", "escaping-request", "producer-run",
                                   "request-run", "request-producer-run"])
def test_case_loader_rejects_substituted_paths_and_request_run_joins(creation, fault):
    request_path, path, _ = _case_metadata(creation)
    if fault == "wrong-sibling":
        other = path.with_name("unrelated_central_result.json")
        other.write_bytes(path.read_bytes())
        path = other
    elif fault in {"absolute-request", "escaping-request"}:
        value = str(request_path) if fault == "absolute-request" else "../escape_request.json"
        _change(path, lambda item: item.update(source_request=value))
    elif fault == "producer-run":
        _change(path, lambda item: item["producer_identity"].update(eval_run_id="other"))
    else:
        if fault == "request-run":
            _change(request_path, lambda item: item.update(eval_run_id="other"))
        else:
            _change(request_path, lambda item: item["producer_identity"].update(eval_run_id="other"))
        _change(path, lambda item: item.update(source_request_sha256=_descriptor(request_path)["sha256"]))
    with pytest.raises(ValueError):
        producer.load_completed_case_result(creation.primary, path, producer_run_binding=creation.binding)


def _normal_quality_case(tmp_path, monkeypatch, status="accuracy_loss"):
    """Real normal candidate export/completion metadata; preserved numeric result."""
    import numpy as np
    from onnx_splitpoint_tool import quality_service, runners
    from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
    from onnx_splitpoint_tool.quality_cache import image_ids_fingerprint, json_fingerprint
    from onnx_splitpoint_tool.quality_result_enrichment import enrich_completed_quality_result
    from onnx_splitpoint_tool.runners import generic_quality_completion as completion
    from onnx_splitpoint_tool.workflow.runner import EvaluationWorkflowRunner as Runner
    from test_generic_quality_completion import capture, make_config
    from test_recovered_quality_admission import _completed_result
    from test_v269d_trt_central_quality_producer import _detection_quality_contract
    from test_v27911_composed_provenance_binding import _functions

    primary = tmp_path / "normal-producer"
    inputs = primary / "synthetic_inputs"
    inputs.mkdir(parents=True)
    fixture = make_config(inputs)
    values = fixture["inputs"]["images"]
    timing = inputs / "timing.npy"
    np.save(timing, values, allow_pickle=False)
    prepared = {**_descriptor(timing), "shape": list(values.shape), "dtype": str(values.dtype), "format": "npy"}
    image = fixture["images"][0]
    representative = {"image_id": image.name, "source_image_sha256": completion._sha(image),
        "original_wh": [80, 60], "prepared_input": prepared, "runtime_input_name": "images"}
    fixture["config"]["representative_input"] = representative
    fixture["path"].write_text(json.dumps(fixture["config"]))
    runtime, detections = capture(fixture)
    evidence = runtime.evidence(detections)
    full = inputs / "full-model.onnx"
    full.write_bytes(b"synthetic original full model authority")
    quality = _detection_quality_contract()
    quality["model"]["sha256"] = completion._sha(full)
    ids = sorted(detections)
    ground_truth = {image_id: deepcopy(detections[image_id]) for image_id in ids}
    if status == "accuracy_loss":
        for records in ground_truth.values():
            records.append({"x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0,
                "score": 1.0, "class_id": 79, "class_name": "synthetic_class_79"})
    quality["dataset"].update(image_count=len(ids), image_ids_sha256=image_ids_fingerprint(ids),
        ground_truth_sha256=json_fingerprint([
            {"image_id": image_id, "ground_truth": ground_truth[image_id]} for image_id in ids]))
    quality.pop("quality_contract_sha256")
    quality["quality_contract_sha256"] = json_fingerprint(quality)
    normal = _functions([
        "_task_quality_execution_location", "_quality_json_safe", "_write_stable_quality_json",
        "_quality_file_sha256", "_quality_contract_sha256", "_quality_artifact_binding",
        "_build_generic_composed_quality_producer_binding", "_export_central_quality_inputs",
        "_load_task_quality_policy"])
    raw_policy = {"reporting_policy": deepcopy(DEFAULT_REPORTING_POLICY),
        "detection": {"primary_metric": "coco_ap_50_95", "guardrails": {}},
        "statistics": {"method": "paired_bootstrap", "execution_location": "central_management",
            "bootstrap_repetitions": 7, "seed": 7, "confidence_level": .95,
            "decision": "lower_one_sided_bound"}}
    policy = normal["_load_task_quality_policy"](raw_policy)
    normal["_verified_suite_vendored_endpoint_attestor_sha256"] = lambda: quality[
        "quality_record_endpoint"]["identity"]["vendored_endpoint_attestor_sha256"]
    monkeypatch.setitem(sys.modules, "splitpoint_runners", runners)
    monkeypatch.setitem(sys.modules, "splitpoint_runners.generic_quality_completion", completion)
    context = dict(fixture["context"], eval_run_id=primary.name, source_run_id="hailo10h_to_trt",
        stage1_backend="hailo10h", stage2_backend="tensorrt", source_model_path=str(full),
        implementation_runner_path=evidence["implementation_runner"]["path"])
    for artifact in context["runtime_artifacts"]:
        artifact["backend"] = "hailo10h" if artifact["role"] == "stage1_runtime" else "native_tensorrt"
    output = primary / "models/yolo26s/benchmark_results/quality_inputs/test_only_h10/b039/results_hailo10h_to_trt"
    exported = normal["_export_central_quality_inputs"](out_dir=output, task="detection", variant="composed",
        policy=policy, gt_by_image=ground_truth, candidate_by_image=detections,
        quality_contract=quality, endpoint_contract=fixture["config"]["source_endpoint_contract"],
        runtime_precision_identity=fixture["config"]["identity"]["precision"],
        composed_producer_context=context, generic_completed_quality_evidence=evidence)
    request_path = Path(exported["request"]["path"])
    request = json.loads(request_path.read_text())
    reference = _document(primary / "quality_management/references/yolo26s/canonical_cpu_reference.json", {
        "schema": "onnx-splitpoint/task-quality-reference-input", "schema_version": 1,
        "task": "detection", "pairing_key": "image_id", "reference_role": "canonical_cpu_ort",
        "semantic_reference_only": True, "provenance_required": True, "quality_contract": quality,
        "quality_contract_sha256": quality["quality_contract_sha256"],
        "records": [{"image_id": image_id, "ground_truth": ground_truth[image_id],
                     "reference": ground_truth[image_id]} for image_id in ids]})
    reference_status = {"status": "completed", "provider": "onnxruntime_cpu",
        "semantic_reference_only": True, "reference_sha256": reference["sha256"],
        "reference_path": reference["path"], "reference_size_bytes": reference["size_bytes"]}
    loaded = quality_service.quality_request_from_manifest(request_path, reference_artifact=reference)
    key, payload = quality_service.prepare_evaluation(loaded)
    identity = Runner._quality_request_identity(request_path, model_id="yolo26s", manifest=request)
    assert identity["identity_valid"] is True and identity["producer_identity_validated"] is True
    result = enrich_completed_quality_result(_completed_result(key, payload, status),
        request_identity=identity, manifest=request, model_id="yolo26s", request_path=request_path,
        run_dir=primary, collection_eval_run_id=primary.name, case_id="b039", run_id="hailo10h_to_trt",
        reference_status=reference_status, producer_flat_field_names=Runner._trt_quality_flat_field_names())
    result_path = request_path.with_name(request_path.stem.replace("_request", "") + "_central_result.json")
    _document(result_path, result)
    creation = _normal_creation(primary, profile_extra={"quality_gate": policy,
        "validation_execution": {"max_items": {"detection": len(ids)}}})
    creation.binding = producer.build_producer_run_binding(primary, session_id=creation.session)
    row_identity = {**fixture["config"]["identity"],
        "output_endpoint_id": fixture["config"]["completion_contract"]["source_endpoint"]["output_endpoint_id"]}
    row = {"task": "detection", "identity": row_identity, "original_run": str(primary),
        "producer_run_binding": deepcopy(creation.binding), "quality_status": status,
        "completion_contract": fixture["config"]["completion_contract"],
        "input_binding": {"prepared_input": prepared, "source_image_sha256": representative["source_image_sha256"]},
        "generic_artifact_binding": {"artifacts": fixture["config"]["expected_runtime_artifacts"]},
        "quality_reference": {"request": _descriptor(request_path), "result": _descriptor(result_path),
            "cpu_reference": reference, "policy_source": _descriptor(primary / "profile.yaml"),
            "policy_sha256": result["policy_sha256"],
            "dataset_manifest_sha256": result["validation_dataset_manifest_sha256"]}}
    return SimpleNamespace(creation=creation, row=row, request=request, request_path=request_path,
        result=result, result_path=result_path, fixture=fixture)


@pytest.mark.parametrize("status", ["reference_close", "accuracy_loss"])
def test_actual_normal_case_quality_remains_scientifically_valid_without_manifest_freeze(tmp_path, monkeypatch, status):
    case = _normal_quality_case(tmp_path, monkeypatch, status)
    with admission.validation_scope():
        value, descriptor = producer.load_completed_case_result(case.creation.primary, case.result_path,
            producer_run_binding=case.creation.binding)
        assert descriptor == case.row["quality_reference"]["result"]
        assert value["accuracy_assessment"]["accuracy_class"] == status
        proof = admission.validate_quality_reference(case.row)
        assert proof["status"] == status
        assert proof["quality_eligible"] is (status == "reference_close")
        assert proof["completed_quality_binding"]["image_count"] == 2
        assert proof["completed_quality_binding"]["all_actual_completed_records_verified"] is True
        _advance_lifecycle(case.creation)
    assert admission._VALIDATION_SCOPE.get() is None


def test_creation_binding_never_bypasses_real_normal_completion_validation(tmp_path, monkeypatch):
    case = _normal_quality_case(tmp_path, monkeypatch)
    _change(case.result_path, lambda value: value.update(producer_identity_sha256="f" * 64))
    case.row["quality_reference"]["result"] = _descriptor(case.result_path)
    with pytest.raises(ValueError, match="producer|normal result"):
        admission.validate_quality_reference(case.row)
