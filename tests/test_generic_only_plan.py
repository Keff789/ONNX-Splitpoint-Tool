"""Separated Generic-only admission and unchanged normal transport boundaries."""
import copy
import csv
import json
import os
from types import SimpleNamespace

import pytest

from onnx_splitpoint_tool import generic_only_plan as admission
from onnx_splitpoint_tool import generic_only_reporting as reporting
from onnx_splitpoint_tool.workflow import generic_completed_task as completed


def document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"path": str(path), "sha256": admission.file_sha256(path)}


def scope_fixture(tmp_path, monkeypatch, backend="hailo10h"):
    primary = tmp_path / "producer"
    setup = {"hailo8": "orin_nx_hailo8_01", "hailo10h": "orin_nx_hailo10_01", "deepx": "orin_nx_deepx_m1_01"}[backend]
    direction = {"hailo8": "hailo8_to_trt", "hailo10h": "hailo10h_to_trt", "deepx": "deepx_to_trt"}[backend]
    identity = dict(model_id="yolo26s", case_id="b039", setup_id=setup,
        direction=direction, comparison_backend=backend, precision="float32_layout_fp16",
        output_endpoint_id="detection:decoded_nms:" + "f" * 64)
    manifest = document(primary / "run_manifest.json", {"hardware_targets": [{"id": setup}]})
    selection = document(primary / "final_candidate_plan.json", {
        "model_id": "yolo26s", "selected_candidates": [{"case_id": "b039"}]})
    local = primary / "cases/b039"
    split = document(local / "split_manifest.json", {"part1": "p1.onnx", "part2": "p2.onnx"})
    artifacts = {}
    for role in admission.ARTIFACT_ROLES:
        p = local / ("p1.onnx" if role == "source_part1_onnx" else
                     "p2.onnx" if role == "source_part2_onnx" else role)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((role + " real fixture bytes").encode())
        artifacts[role] = {"path": str(p), "local_path": str(p), "sha256": admission.file_sha256(p)}
    env = document(primary / "runtime-environment.json", {
        "executable": "/venv/bin/python3", "executable_sha256": "a" * 64})
    quality = {"request": {"path": "/fixture/request.json", "sha256": "b" * 64}}
    boundary = {"schema_version": 1, "minimum_boundary_tensors": 2,
        "part1_outputs": {n: {"shape": [1, 2], "dtype": "float32"} for n in ("left", "right")},
        "part2_inputs": {n: {"shape": [1, 2], "dtype": "float32"} for n in ("x", "y")},
        "part2_to_part1": {"x": "left", "y": "right"}, "passthrough_inputs": {},
        "transfer_contract": {"boundary_mode": "canonical_float32"}}
    row = {**identity, "identity": identity, "admission_mode": "generic_only", "task": "detection",
        "case_key": "|".join(admission.completion_identity(identity)), "original_run": str(primary),
        "original_local_case_dir": str(local), "selection_source": selection, "split_manifest": split,
        "case_output_relative": f"{setup}/yolo26s/b039", "quality_status": "accuracy_loss",
        "quality_reference": quality, "boundary_observation": boundary,
        "original_run_cfg": {"stage1_provider": {"hailo8": "hailo8", "hailo10h": "hailo10h", "deepx": "deepx_m1"}[backend],
                             "stage2_provider": "tensorrt", "throughput_queue_depth": 2},
        "generic_runtime_environment": {"python": "/venv/bin/python3", "python_sha256": "a" * 64,
            "environment": {}, "source": env},
        "generic_artifact_binding": {"schema": "onnx-splitpoint/generic-only-artifact-binding/v1",
            "runtime_precision": identity["precision"],
            "artifacts": artifacts, "producer_source": quality["request"],
            "quality_reference": quality, "boundary_contract": boundary},
        "input_binding": {"prepared_input": {"path": "/fixture/prepared.npy", "sha256": "c" * 64}},
        "functional_reference": {"kind": "serial_compiled_subnets", "input_sha256": "c" * 64,
            "outputs": {"detections": {"shape": [1, 300, 6], "dtype": "float32"}}},
        "completion_contract": {"fixture": "the runtime/normal builder validates the real execution contract"},
        "mixed_input_control": {"path": "/fixture/mixed.json", "sha256": "d" * 64},
        "stage1_execution": {"policy": {"hailo8": "synchronous", "hailo10h": "hailo_async_slots_v1", "deepx": "deepx_async_slots_v1"}[backend],
                             "inflight": {"hailo8": 1, "hailo10h": 8, "deepx": 4}[backend]}}
    row["expected_runtime_artifacts"] = {role: {k: artifacts[role][k] for k in ("path", "sha256")}
                                        for role in ("part1_runtime", "engine")}
    scope = tmp_path / "scope.csv"
    with scope.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(identity)); writer.writeheader(); writer.writerow(identity)
    plan = {"scope_mode": admission.MODE, "admission_mode": "generic_only", "authorized_row_count": 1, "row_count": 1,
        "build_allowed": False, "fallback_allowed": False, "automatic_retry": False,
        "scope_source_sha256": admission.file_sha256(scope), "producer_run_manifest": manifest, "rows": [row]}
    monkeypatch.setattr(admission, "validate_quality_reference", lambda value: {"status": "accuracy_loss"})
    monkeypatch.setattr(reporting, "validate_quality_reference", lambda value: {"status": "accuracy_loss"})
    return plan, scope, primary, row


@pytest.mark.parametrize("backend", ["hailo8", "hailo10h", "deepx"])
def test_scope_admits_quality_fail_without_any_native_row(tmp_path, monkeypatch, backend):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch, backend)
    assert completed.validate_plan(plan, scope, primary) == [row]
    config, staged = completed.runtime_configuration(row, "/session/new", "fresh")
    assert config["measurement_mode"] == admission.MODE
    assert config["stage1_execution"]["inflight"] == {"hailo8": 1, "hailo10h": 8, "deepx": 4}[backend]
    assert config["original_run_cfg"]["native_split_quality_binding"] == ""
    assert config["boundary_observation"] == row["boundary_observation"]
    assert config["functional_reference"]["outputs"] == row["functional_reference"]["outputs"]
    assert all("payload" not in item for item in staged)
    assert set(row).isdisjoint(admission.FORBIDDEN_NATIVE)


@pytest.mark.parametrize("mutation,error", [
    (lambda p, r: p.update(admission_mode="paired"), "explicit"),
    (lambda p, r: r.update(native_source="/fake/native.json"), "Native"),
    (lambda p, r: r.update(energy_capture={"enabled": True}), "energy"),
    (lambda p, r: r.update(case_output_relative="../escape"), "output path"),
    (lambda p, r: r.pop("boundary_observation"), "boundary observation"),
    (lambda p, r: r["identity"].update(case_id="b099"), "identities"),
    (lambda p, r: r["generic_runtime_environment"].update(python_sha256="e" * 64), "interpreter"),
    (lambda p, r: r.update(quality_status="reference_close"), "quality decision"),
    (lambda p, r: r["generic_artifact_binding"]["artifacts"].pop("engine_build_receipt"), "receipt set"),
    (lambda p, r: p.update(build_allowed=True), "forbids builds"),
    (lambda p, r: p.update(row_count=2), "explicitly authorize"),
    (lambda p, r: p.pop("row_count"), "explicitly authorize"),
])
def test_scope_rejects_missing_or_changed_authority(tmp_path, monkeypatch, mutation, error):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    mutation(plan, row)
    with pytest.raises((ValueError, KeyError), match=error):
        completed.validate_plan(plan, scope, primary)


def test_artifact_and_selection_bytes_are_authoritative(tmp_path, monkeypatch):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    (primary / "final_candidate_plan.json").write_text('{"selected_candidates": []}')
    with pytest.raises(ValueError, match="changed"):
        completed.validate_plan(plan, scope, primary)


def test_preserved_local_compiled_part_is_staged_without_fabricating_remote_reuse(tmp_path, monkeypatch):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    artifact = row["generic_artifact_binding"]["artifacts"]["part1_runtime"]
    original = copy.deepcopy(artifact)
    artifact["stage_local"] = True
    config, staged = admission.runtime_configuration(row, "/session/new", "fresh")
    projected = config["generic_artifact_binding"]["artifacts"]["part1_runtime"]
    assert projected["sha256"] == original["sha256"]
    assert projected["path"].startswith(config["case_dir"] + "/artifacts/")
    assert row["generic_artifact_binding"]["artifacts"]["part1_runtime"]["path"] == original["path"]
    assert any(item["role"] == "generic_part1_runtime" and item["local"] == original["local_path"]
               and item["remote"] == projected["path"] and not item.get("existing_remote") for item in staged)
    assert config["expected_runtime_artifacts"]["part1_runtime"]["path"] == projected["path"]


@pytest.mark.parametrize("role", ["engine", "build_part2_onnx"])
def test_immutable_tensorrt_receipt_paths_cannot_be_relocated(tmp_path, monkeypatch, role):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    row["generic_artifact_binding"]["artifacts"][role]["stage_local"] = True
    with pytest.raises(ValueError, match="verified receipt path"):
        admission.runtime_configuration(row, "/session/new", "fresh")


@pytest.mark.parametrize("fault,error", [
    ("policy", "fixed backend"), ("budget", "work budget"), ("artifact", "source/quality"),
    ("input", "operation differs"), ("completion", "operation differs"),
    ("reference", "operation differs"), ("mixed", "operation differs"),
    ("boundary", "operation differs"), ("original", "operation differs"),
    ("directory", "operation differs"), ("staging", "staging differs"),
    ("python", "dispatch interpreter"), ("environment", "dispatch interpreter"),
])
def test_prepared_generic_only_config_rejects_drift(tmp_path, monkeypatch, fault, error):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    config, staged = admission.runtime_configuration(row, "/session/new", "fresh")
    output = tmp_path / "measurement"
    plan.update(execution_id="fresh", output_root=str(output))
    row.update(runtime_config=config, staging_files=staged, remote_root="/session/new",
        remote_python_executable=row["generic_runtime_environment"]["python"],
        remote_environment=copy.deepcopy(row["generic_runtime_environment"]["environment"]),
        result_path=str(output / "results" / row["case_output_relative"] / "generic_completion_results.json"))
    assert completed.validate_plan(plan, scope, primary) == [row]
    if fault == "policy":
        config["stage1_execution"]["inflight"] = 1
    elif fault == "budget":
        config["task_count"] = 10
    elif fault == "artifact":
        config["generic_artifact_binding"]["artifacts"]["source_part2_onnx"]["sha256"] = "f" * 64
    elif fault == "input":
        config["input_binding"]["prepared_input"]["path"] = "/different/input.npy"
    elif fault == "completion":
        config["completion_contract"] = {"changed": True}
    elif fault == "reference":
        config["functional_reference"]["outputs"]["detections"]["dtype"] = "float16"
    elif fault == "mixed":
        config["mixed_input_control"]["sha256"] = "e" * 64
    elif fault == "boundary":
        config["boundary_observation"]["transfer_contract"] = {"changed": True}
    elif fault == "original":
        config["original_run_cfg"]["throughput_queue_depth"] = 99
    elif fault == "directory":
        config["config_directory"] = "/other/session"
    elif fault == "python":
        row["remote_python_executable"] = "/different/python"
    elif fault == "environment":
        row["remote_environment"]["OMP_NUM_THREADS"] = "99"
    else:
        row["staging_files"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match=error):
        completed.validate_plan(plan, scope, primary)


def test_manifest_binding_cannot_substitute_a_sibling_target_authority(tmp_path, monkeypatch):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    manifest = json.loads((primary / "run_manifest.json").read_text())
    plan["producer_run_manifest"] = document(primary / "different_manifest.json", manifest)
    with pytest.raises(ValueError, match="producer run manifest"):
        completed.validate_plan(plan, scope, primary)


def quality_fixture(tmp_path, monkeypatch, status="accuracy_loss"):
    import numpy as np
    from onnx_splitpoint_tool import quality_service, native_detection_postprocess
    identity = {"model_id": "yolo26s", "case_id": "b039", "setup_id": "h10",
                "precision": "float32_layout_fp16", "output_endpoint_id": "endpoint"}
    arts = {role: {"path": "/remote/" + role, "sha256": str(i) * 64}
            for i, role in enumerate(admission.ARTIFACT_ROLES, 1)}
    cpu = document(tmp_path / "cpu.json", {"fixture": "canonical CPU records are checked by ordinary loader"})
    policy = {"detection": {"non_inferiority_margin": 0.01}, "reporting_policy": {"relative_loss_threshold": 0.05},
        "statistics": {"bootstrap_repetitions": 1000, "method": "paired_bootstrap", "confidence_level": .95,
                       "seed": 20260710, "decision": "lower_one_sided_bound"}}
    gate = {**policy["detection"], "reporting_policy": policy["reporting_policy"]}
    candidate = document(tmp_path / "candidate.json", {"records": []})
    req = {"model_id": "yolo26s", "case_id": "b039", "setup_id": "h10", "metric_gate_config": gate,
        "candidate": {"path": "candidate.json", "sha256": candidate["sha256"]},
        "statistics": policy["statistics"], **{key: "a" * 64 for key in (
        "policy_sha256", "quality_contract_sha256", "endpoint_contract_hash",
        "candidate_execution_completion_contract_sha256", "producer_identity_sha256")}}
    request = document(tmp_path / "request.json", req)
    producer = {"runtime_artifacts": [
        {"role": "stage1_runtime", "sha256": arts["part1_runtime"]["sha256"]},
        {"role": "stage2_runtime", "sha256": arts["engine"]["sha256"]}],
        "model": {"part1_onnx_sha256": arts["source_part1_onnx"]["sha256"],
                  "part2_onnx_sha256": arts["source_part2_onnx"]["sha256"]},
        "runtime_precision_identity": identity["precision"]}
    result = {**req, "schema": "onnx-splitpoint/management-paired-quality-result", "status": "completed",
        "technical_status": "completed", "task": "detection", "variant": "composed",
        "cpu_reference_semantic_only": True, "artifact_provenance_binding_status": "verified",
        "execution_role": "generic_composed_quality_candidate", "source_request_sha256": request["sha256"],
        "management_cpu_reference": {"status": "completed", "provider": "onnxruntime_cpu",
            "reference_sha256": cpu["sha256"], "semantic_reference_only": True},
        "producer_identity": producer, "validation_dataset_manifest_sha256": "d" * 64,
        "n": 2, "evaluated_images": 2, "primary": {"bootstrap_repetitions_requested": 1000},
        "accuracy_assessment": {"accuracy_class": status}}
    reference = {"request": request, "result": document(tmp_path / "result.json", result), "cpu_reference": cpu,
        "policy_source": document(tmp_path / "profile.json", {"quality_gate": policy,
            "validation_execution": {"max_items": {"detection": 2}}}),
        "policy_sha256": "a" * 64, "dataset_manifest_sha256": "d" * 64}
    array = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    prepared_path = tmp_path / "prepared.npy"
    np.save(prepared_path, array, allow_pickle=False)
    prepared = {"path": str(prepared_path), "sha256": admission.file_sha256(prepared_path),
                "shape": list(array.shape), "dtype": str(array.dtype)}
    row = {"identity": identity, "quality_reference": reference,
           "input_binding": {"prepared_input": prepared, "source_image_sha256": "e" * 64},
           "completion_contract": {"processor_contract": {"original_wh": [80, 60]}},
           "generic_artifact_binding": {"artifacts": arts}}
    monkeypatch.setattr(quality_service, "_validate_generic_composed_candidate_execution_contract",
        lambda value, **kwargs: (value, "a" * 64))
    monkeypatch.setattr(native_detection_postprocess, "verify_detection_completion_execution_contract",
        lambda value: {"source_endpoint": {"endpoint_contract_hash": "a" * 64, "output_endpoint_id": "endpoint"}})
    monkeypatch.setattr(quality_service, "quality_request_from_manifest", lambda *args, **kwargs:
        SimpleNamespace(reference_records=[{}, {}], repetitions=1000))
    representative = {"prepared_input": copy.deepcopy(prepared), "source_image_sha256": "e" * 64,
        "original_wh": [80, 60], "image_id": "test-only.png"}
    monkeypatch.setattr(admission, "_completed_quality_binding", lambda *args, **kwargs:
        {"status": "verified_real_exporter_in_separate_tests", "representative_input": representative,
         "representative_input_comparison": {"prepared_tensor": {"shape": list(array.shape),
            "dtype": str(array.dtype), "tensor_sha256": admission.hashlib.sha256(array.tobytes()).hexdigest()}}})
    return row, result


@pytest.mark.parametrize("status", ["reference_close", "accuracy_loss", "inconclusive"])
def test_quality_admits_real_completed_decision_without_upgrading_fail(tmp_path, monkeypatch, status):
    row, _ = quality_fixture(tmp_path, monkeypatch, status)
    proof = admission.validate_quality_reference(row)
    assert proof["status"] == status
    assert proof["quality_eligible"] == (status == "reference_close")


@pytest.mark.parametrize("mutation,error", [
    (lambda r: r.update(status="failed"), "completed normal"),
    (lambda r: r.update(source_request_sha256="b" * 64), "request/result"),
    (lambda r: r["management_cpu_reference"].update(provider="cuda"), "CPU reference"),
    (lambda r: r["producer_identity"].update(runtime_precision_identity="uint8_dequant_fp16"), "precision"),
    (lambda r: r["producer_identity"]["runtime_artifacts"][0].update(sha256="f" * 64), "compiled artifact"),
    (lambda r: r.update(validation_dataset_manifest_sha256="f" * 64), "policy/dataset"),
    (lambda r: r["metric_gate_config"].update(non_inferiority_margin=1), "threshold"),
])
def test_quality_rejects_actual_source_binding_conflicts(tmp_path, monkeypatch, mutation, error):
    row, result = quality_fixture(tmp_path, monkeypatch)
    # Change the actual bound report and its declared bytes together. Admission
    # must still reject the semantic conflict, not merely an obsolete SHA.
    mutation(result)
    row["quality_reference"]["result"] = document(tmp_path / "result.json", result)
    with pytest.raises(ValueError, match=error):
        admission.validate_quality_reference(row)


def test_quality_can_select_only_one_original_completed_summary_record(tmp_path):
    source = document(tmp_path / "summary.json", {"results": [
        {"case_id": "b001", "status": "failed"}, {"case_id": "b002", "status": "completed"}]})
    source["selector"] = {"case_id": "b002"}
    assert admission._json(source)["status"] == "completed"
    source["selector"] = {"case_id": "b099"}
    with pytest.raises(ValueError, match="absent or ambiguous"):
        admission._json(source)


def test_validation_scope_hashes_once_then_fully_rechecks_and_does_not_escape(tmp_path, monkeypatch):
    source = document(tmp_path / "source.json", {"value": 1})
    original = admission._file_sha256
    calls = []
    def counted(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(admission, "_file_sha256", counted)
    with admission.validation_scope():
        for _ in range(5):
            admission.read_bound(source)
        with admission.validation_scope():
            admission.read_bound(source)
        assert len(calls) == 1
    assert len(calls) == 2
    admission.read_bound(source)
    assert len(calls) == 3


@pytest.mark.parametrize("fault", ["content", "same_size_mtime", "replace_inode", "expected_digest"])
def test_validation_scope_never_reuses_changed_authority(tmp_path, fault):
    source = document(tmp_path / "source.json", {"value": 1})
    path = tmp_path / "source.json"
    before = path.stat()
    with pytest.raises(ValueError, match="changed"):
        with admission.validation_scope():
            admission.read_bound(source)
            if fault == "expected_digest":
                source["sha256"] = "f" * 64
            elif fault == "replace_inode":
                replacement = tmp_path / "replacement.json"
                replacement.write_bytes(path.read_bytes())
                os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
                replacement.replace(path)
            else:
                path.write_bytes(path.read_bytes().replace(b"1", b"2"))
                if fault == "same_size_mtime":
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            admission.read_bound(source)


def test_validation_scope_checks_changes_even_after_last_consumer(tmp_path):
    source = document(tmp_path / "source.json", {"value": 1})
    with pytest.raises(ValueError, match="changed"):
        with admission.validation_scope():
            admission.read_bound(source)
            (tmp_path / "source.json").write_text('{"value": 2}')


def test_validation_scope_final_hash_is_not_replaced_by_metadata_check(tmp_path, monkeypatch):
    source = document(tmp_path / "source.json", {"value": 1})
    version = admission._file_version(tmp_path / "source.json")
    monkeypatch.setattr(admission, "_file_version", lambda _: version)
    with pytest.raises(ValueError, match="changed"):
        with admission.validation_scope():
            admission.read_bound(source)
            (tmp_path / "source.json").write_text('{"value": 2}')


@pytest.mark.parametrize("change_after_last_read", [False, True])
def test_normal_preparation_uses_one_scope_and_cannot_return_changed_source(
        tmp_path, monkeypatch, change_after_last_read):
    source = document(tmp_path / "source.json", {"value": 1})
    calls = []
    original = admission._file_sha256
    def counted(path):
        calls.append(str(path))
        return original(path)
    def preparation(*args, **kwargs):
        for _ in range(4):
            admission.read_bound(source)
        if change_after_last_read:
            (tmp_path / "source.json").write_text('{"value": 2}')
        return tmp_path / "execution-plan.json"
    monkeypatch.setattr(admission, "_file_sha256", counted)
    monkeypatch.setattr(completed, "_prepare_execution", preparation)
    if change_after_last_read:
        with pytest.raises(ValueError, match="changed"):
            completed.prepare_execution(tmp_path / "scope.json", tmp_path / "output")
    else:
        assert completed.prepare_execution(tmp_path / "scope.json", tmp_path / "output") == tmp_path / "execution-plan.json"
        assert len(calls) == 2


def test_json_version_detects_replacement_with_preserved_size_and_mtime(tmp_path):
    source = document(tmp_path / "source.json", {"value": 1})
    assert admission._json(source) == {"value": 1}
    path = tmp_path / "source.json"
    before = path.stat()
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(path.read_bytes().replace(b"1", b"2"))
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    with pytest.raises(ValueError, match="changed"):
        admission._json(source)


def test_quality_scope_reuses_small_proof_and_preserves_fail(tmp_path, monkeypatch):
    row, _ = quality_fixture(tmp_path, monkeypatch, "accuracy_loss")
    original = admission._validate_quality_reference
    calls = []
    def counted(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(admission, "_validate_quality_reference", counted)
    with admission.validation_scope():
        first = admission.validate_quality_reference(row)
        first["status"] = "reference_close"
        second = admission.validate_quality_reference(row)
        assert second["status"] == "accuracy_loss"
        assert second["quality_eligible"] is False
        assert len(calls) == 1
    admission.validate_quality_reference(row)
    assert len(calls) == 2


def test_quality_scope_does_not_reuse_changed_scientific_row(tmp_path, monkeypatch):
    row, _ = quality_fixture(tmp_path, monkeypatch)
    with admission.validation_scope():
        admission.validate_quality_reference(row)
        row["identity"]["precision"] = "other_precision"
        with pytest.raises(ValueError, match="precision"):
            admission.validate_quality_reference(row)


def test_quality_scope_tracks_candidate_even_when_normal_loader_is_separately_tested(tmp_path, monkeypatch):
    row, _ = quality_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="changed"):
        with admission.validation_scope():
            admission.validate_quality_reference(row)
            (tmp_path / "candidate.json").write_text('{"records": [1]}')
            admission.validate_quality_reference(row)


@pytest.mark.parametrize("change_after_terminal", [False, True])
def test_public_generic_execution_reuses_quality_without_binding_mutable_outputs(
        tmp_path, monkeypatch, change_after_terminal):
    row, _ = quality_fixture(tmp_path, monkeypatch, "accuracy_loss")
    plan_path = tmp_path / "execution-plan.json"
    document(plan_path, {"scope_mode": admission.MODE})
    original = admission._validate_quality_reference
    validations, guards = [], []
    def validate(value):
        validations.append(value)
        return original(value)
    source_guard = lambda: guards.append("checked")
    def execute(path, plan, **kwargs):
        assert kwargs["_source_guard"] is source_guard
        for phase in ("start", "report", "terminal"):
            kwargs["_source_guard"]()
            proof = admission.validate_quality_reference(row)
            assert proof["status"] == "accuracy_loss"
            # Normal execution changes these files; they are not authorities
            # read through admission.read_bound and must not enter its cache.
            plan["phase"] = phase
            completed.write_json(path, plan)
            completed.write_json(tmp_path / "dispatch_attempts.json", {"phase": phase})
        completed.write_json(tmp_path / "terminal-status.json", {"exitcode": 0})
        if change_after_terminal:
            (tmp_path / "candidate.json").write_text('{"records": [1]}')
        return 0
    monkeypatch.setattr(admission, "_validate_quality_reference", validate)
    monkeypatch.setattr(completed, "_execute_plan", execute)
    if change_after_terminal:
        with pytest.raises(ValueError, match="changed"):
            completed.execute_plan(plan_path, _source_guard=source_guard)
        # Preserve raw observations; the public call still must not succeed.
        assert json.loads((tmp_path / "terminal-status.json").read_text())["exitcode"] == 0
    else:
        assert completed.execute_plan(plan_path, _source_guard=source_guard) == 0
    assert len(validations) == 1 and len(guards) == 3
    assert admission._VALIDATION_SCOPE.get() is None


def test_public_native_execution_retains_default_path_without_generic_scope(tmp_path, monkeypatch):
    path = tmp_path / "native-plan.json"
    document(path, {"scope_mode": "audited_selective_remeasurement"})
    def execute(plan_path, plan, **kwargs):
        assert plan_path == path and plan["scope_mode"] == "audited_selective_remeasurement"
        assert admission._VALIDATION_SCOPE.get() is None
        assert kwargs["resume"] is True
        return 7
    monkeypatch.setattr(admission, "validation_scope", lambda: pytest.fail("Native must retain its default path"))
    monkeypatch.setattr(completed, "_execute_plan", execute)
    assert completed.execute_plan(path, resume=True) == 7


def test_raw_quality_without_verified_complete_task_export_cannot_admit(tmp_path, monkeypatch):
    row, _ = quality_fixture(tmp_path, monkeypatch)
    def missing(*args, **kwargs):
        raise ValueError("actual completed-quality execution binding missing")
    monkeypatch.setattr(admission, "_completed_quality_binding", missing)
    with pytest.raises(ValueError, match="completed-quality execution binding missing"):
        admission.validate_quality_reference(row)


def test_quality_cannot_shorten_the_frozen_dataset_even_if_result_and_reference_agree(tmp_path, monkeypatch):
    from onnx_splitpoint_tool import quality_service
    row, result = quality_fixture(tmp_path, monkeypatch)
    result.update(n=1, evaluated_images=1)
    row["quality_reference"]["result"] = document(tmp_path / "result.json", result)
    monkeypatch.setattr(quality_service, "quality_request_from_manifest", lambda *args, **kwargs:
        SimpleNamespace(reference_records=[{}], repetitions=1000))
    with pytest.raises(ValueError, match="image/statistics/threshold"):
        admission.validate_quality_reference(row)


@pytest.mark.parametrize("fault", ["image", "geometry", "dtype", "payload"])
def test_quality_representative_must_match_actual_timed_input(tmp_path, monkeypatch, fault):
    row, _ = quality_fixture(tmp_path, monkeypatch)
    proof = admission._completed_quality_binding()
    if fault == "image":
        proof["representative_input"]["source_image_sha256"] = "f" * 64
    elif fault == "geometry":
        proof["representative_input"]["original_wh"] = [60, 80]
    elif fault == "dtype":
        proof["representative_input"]["prepared_input"]["dtype"] = "float32"
    else:
        proof["representative_input_comparison"]["prepared_tensor"]["tensor_sha256"] = "f" * 64
    monkeypatch.setattr(admission, "_completed_quality_binding", lambda *args, **kwargs: proof)
    with pytest.raises(ValueError, match="Quality representative"):
        admission.validate_quality_reference(row)


def test_actual_completed_quality_export_joins_exact_timing_prepared_array(tmp_path):
    """Actual normal per-image factories/export validator; no device/results claim."""
    import numpy as np
    from test_generic_quality_completion import make_config, envelope, descriptor
    fixture = make_config(tmp_path)
    image = fixture["images"][0]
    prepared_path = tmp_path / "timing.npy"
    values = fixture["inputs"]["images"]
    np.save(prepared_path, values, allow_pickle=False)
    prepared = {**descriptor(prepared_path), "shape": list(values.shape), "dtype": str(values.dtype), "format": "npy"}
    representative = {"image_id": image.name, "source_image_sha256": admission.file_sha256(image),
        "original_wh": [80, 60], "prepared_input": prepared, "runtime_input_name": "images"}
    fixture["config"]["representative_input"] = representative
    fixture["path"].write_text(json.dumps(fixture["config"]))
    request, result, request_path, candidate_path = envelope(fixture, tmp_path)
    proof = admission._completed_quality_binding(request, result, fixture["config"]["completion_contract"],
        fixture["config"]["expected_runtime_artifacts"], request_path=request_path, reference_descriptor={})
    row = {"input_binding": {"prepared_input": prepared, "source_image_sha256": representative["source_image_sha256"]},
        "completion_contract": fixture["config"]["completion_contract"]}
    admission.validate_quality_input_binding(row, proof)
    assert proof["image_count"] == 2 and proof["all_actual_completed_records_verified"] is True
    assert "per_image" not in proof and len(json.dumps(proof)) < 65536
    row["input_binding"]["source_image_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="Quality representative"):
        admission.validate_quality_input_binding(row, proof)


def test_generic_only_report_preserves_unstarted_and_quality_fail_without_pairs(tmp_path, monkeypatch):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    config, _ = admission.runtime_configuration(row, "/remote/session", "session")
    row["runtime_config"] = config
    output = tmp_path / "measurement"
    row["result_path"] = str(output / "results" / row["case_output_relative"] / "generic_completion_results.json")
    summary = reporting.write_completion_reports(plan, output / "results", output / "reports")
    assert summary["status_counts"]["unstarted"] == 1 and summary["valid_repetitions"] == 0
    assert summary["quality_status_counts"] == {"accuracy_loss": 1}
    assert summary["native_performance_claimed"] is False
    assert not list((output / "reports").glob("*cross_runner*"))
    assert not any("native_fps" in p.read_text() for p in (output / "reports").iterdir())


@pytest.mark.parametrize("fault", ["quality_status", "quality_reference"])
def test_generic_only_report_cannot_upgrade_row_quality_or_substitute_config(tmp_path, monkeypatch, fault):
    plan, scope, primary, row = scope_fixture(tmp_path, monkeypatch)
    config, _ = admission.runtime_configuration(row, "/remote/session", "session")
    row["runtime_config"] = config
    output = tmp_path / "measurement"
    row["result_path"] = str(output / "results" / row["case_output_relative"] / "generic_completion_results.json")
    if fault == "quality_status":
        row["quality_status"] = "reference_close"
    else:
        config["quality_reference"]["request"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="actual CPU evidence/config"):
        reporting.write_completion_reports(plan, output / "results", output / "reports")


def test_generic_only_admission_requires_observed_settings_but_not_quality_pass(monkeypatch):
    monkeypatch.setattr(reporting, "validate_completion_attestations", lambda *args: [{"verified": True}] * 3)
    row = {"case_key": "case", "measurement_mode": admission.MODE, "admission_mode": "generic_only",
        "quality_status": "accuracy_loss", "runtime_environment_comparison": {
            "status": "observed_no_known_configuration_conflict"}}
    observation = {"status": "completed", "valid_repetition_count": 3}
    assert reporting.generic_only_admission(observation, row)["eligible_for_generic_only_execution"] is True
    row["runtime_environment_comparison"]["status"] = "conflict"
    assert reporting.generic_only_admission(observation, row)["eligible_for_generic_only_execution"] is False


def test_paired_plan_cannot_silently_enable_generic_only(tmp_path):
    with pytest.raises(ValueError, match="explicitly separated"):
        completed.validate_plan({"admission_mode": "generic_only"}, tmp_path / "missing", tmp_path)


def test_normal_download_preserves_real_serial_reference_artifacts(tmp_path):
    import numpy as np
    from tests.test_generic_completion_dispatch import LocalTransport
    remote = tmp_path / "remote/output"
    remote.mkdir(parents=True)
    sample = remote / "serial_reference/repetition_1/output_0.npy"
    sample.parent.mkdir(parents=True)
    np.save(sample, np.asarray([1, 2], dtype=np.float32), allow_pickle=False)
    tensor = {"path": str(sample.relative_to(remote)), "sha256": admission.file_sha256(sample)}
    result = document(sample.parent / "completed_task.json", {"detection_count": 1})
    result["path"] = str((sample.parent / "completed_task.json").relative_to(remote))
    document(remote / "generic_completion_results.json", {"repetitions": [{"serial_reference": {
        "raw_output_samples": {"output": tensor}, "completed_result_artifact": result}}]})
    local = tmp_path / "local"
    assert completed.download_case(LocalTransport(), str(remote), local, 1)["result_collected"] is True
    assert admission.file_sha256(local / tensor["path"]) == tensor["sha256"]
    assert admission.file_sha256(local / result["path"]) == result["sha256"]
    (local / tensor["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="existing collected"):
        completed.download_case(LocalTransport(), str(remote), local, 2)


@pytest.mark.parametrize("mode", ["strict", "fast_oracle_outside_timing"])
def test_real_detection_completion_attestations_and_sentinel_are_verified(tmp_path, mode):
    import numpy as np
    from onnx_splitpoint_tool import native_detection_postprocess as detection
    from tests.test_v272_detection_completion_runtime import _decoded_outputs, _decoded_source
    outputs = _decoded_outputs()
    contract = detection.build_detection_completion_execution_contract(model_id="yolo26s", outputs=outputs,
        input_hw=[64, 64], original_wh=[80, 60], source_endpoint_contract=_decoded_source(outputs),
        preprocess={"mode": "letterbox", "rgb": True, "pad_value": 114})
    repetitions = []
    source_map = None
    for index in (1, 2, 3):
        processor = detection.create_detection_completion_runtime(contract, runtime_mode=mode,
            source_paths=(__file__,), sentinel_context={"prepared_input_sha256": "a" * 64})
        for _ in range(1100):
            processor.process(outputs)
        proof = processor.attestation(completed_work_units=processor.completed_count)
        source = detection.detection_completion_runtime_source_check(processor)
        source_map = source["before"]
        samples = {}
        for ordinal, (name, array) in enumerate(outputs.items()):
            path = tmp_path / f"sample_{index}_{ordinal}.npy"
            np.save(path, array, allow_pickle=False)
            samples[name] = {"path": path.name, "sha256": admission.file_sha256(path)}
        repetitions.append({"repetition_index": index, "warmup_completed_count": 100,
            "execution_id": "fresh", "boot_id": "cpu-fixture-boot", "pid": 100 + index,
            "process_start_ticks": index, "runtime_instance_id": f"fresh:{index}",
            "completed_task_count": 1000, "completion_source_check": source,
            "completion_runtime_mode": mode, "completion_execution_attestation": proof,
            "completion_attestation_scope": "last_completed_task_postflight_including_warmup_count",
            "last_completed_result": processor.last_result, "raw_output_samples": samples,
            "stage1_execution": {"policy": "hailo_async_slots_v1", "inflight": 8},
            "stage1_peak_uncollected_jobs": 8})
    config = {"execution_id": "fresh", "completion_contract": contract,
              "stage1_execution": {"policy": "hailo_async_slots_v1", "inflight": 8},
              "runtime_source_files": [{"path": path, "sha256": digest} for path, digest in source_map.items()]}
    row = {"runtime_config": config, "result_path": str(tmp_path / "generic_completion_results.json")}
    observation = {"repetitions": repetitions}
    assert len(reporting.validate_completion_attestations(observation, row)) == 3
    changed = copy.deepcopy(observation)
    changed["repetitions"][1]["last_completed_result"]["detection_count"] += 1
    with pytest.raises(ValueError, match="sentinel"):
        reporting.validate_completion_attestations(changed, row)
    changed = copy.deepcopy(observation)
    changed["repetitions"][1]["completion_execution_attestation"]["completed_work_units"] = 1000
    with pytest.raises(ValueError, match="task count"):
        reporting.validate_completion_attestations(changed, row)
    changed = copy.deepcopy(observation)
    changed["repetitions"][0]["stage1_peak_uncollected_jobs"] = 1
    with pytest.raises(ValueError, match="host peak"):
        reporting.validate_completion_attestations(changed, row)
    changed = copy.deepcopy(observation)
    changed["repetitions"][1].update({key: changed["repetitions"][0][key]
        for key in ("boot_id", "pid", "process_start_ticks")})
    with pytest.raises(ValueError, match="share a process"):
        reporting.validate_completion_attestations(changed, row)
    changed = copy.deepcopy(observation)
    changed["repetitions"][0]["execution_id"] = "foreign"
    with pytest.raises(ValueError, match="execution identity"):
        reporting.validate_completion_attestations(changed, row)
    changed = copy.deepcopy(observation)
    changed["repetitions"][0].pop("process_start_ticks")
    with pytest.raises(ValueError, match="process/execution identity"):
        reporting.validate_completion_attestations(changed, row)


@pytest.mark.parametrize("malformed", [{}, ""])
def test_malformed_explicit_functional_reference_cannot_skip_preparation_admission(tmp_path, malformed):
    from onnx_splitpoint_tool.generic_completion_plan import apply_native_functional_reference
    row = {"functional_reference_source": malformed}
    with pytest.raises((ValueError, KeyError, TypeError)):
        completed.runtime_configuration(row, "/session", "execution")
    with pytest.raises((ValueError, KeyError, TypeError)):
        completed.verify_runtime_environment(None, row, tmp_path / "unused.json")


def test_quality_scope_retains_first_64_small_proofs_but_not_65(tmp_path, monkeypatch):
    source = document(tmp_path / "proof-authority.json", {"value": 1})
    calls = {}
    def validate(row):
        case = row["identity"]["case_id"]
        calls[case] = calls.get(case, 0) + 1
        admission.read_bound(source)
        return {"case_id": case, "status": "accuracy_loss", "quality_eligible": False}
    monkeypatch.setattr(admission, "_validate_quality_reference", validate)
    rows = [{"task": "detection", "identity": {"case_id": f"b{index:03d}"}}
            for index in range(65)]
    with admission.validation_scope() as scope:
        for row in rows:
            admission.validate_quality_reference(row)
        assert len(scope.proofs) == 64
        for row in rows:
            proof = admission.validate_quality_reference(row)
            assert proof["status"] == "accuracy_loss" and proof["quality_eligible"] is False
        assert all(calls[row["identity"]["case_id"]] == 1 for row in rows[:64])
        assert calls[rows[64]["identity"]["case_id"]] == 2
        assert len(scope.proofs) == 64


@pytest.mark.parametrize("hide_version_change", [False, True])
def test_quality_scope_full_budget_still_rehashes_authority_at_end(
        tmp_path, monkeypatch, hide_version_change):
    source = document(tmp_path / "proof-authority.json", {"value": 1})
    path = tmp_path / "proof-authority.json"
    if hide_version_change:
        version = admission._file_version(path)
        monkeypatch.setattr(admission, "_file_version", lambda _: version)
    def validate(row):
        admission.read_bound(source)
        return {"case_id": row["identity"]["case_id"], "status": "reference_close"}
    monkeypatch.setattr(admission, "_validate_quality_reference", validate)
    with pytest.raises(ValueError, match="changed"):
        with admission.validation_scope() as scope:
            for index in range(65):
                admission.validate_quality_reference({
                    "task": "detection", "identity": {"case_id": f"b{index:03d}"}})
            assert len(scope.proofs) == 64
            path.write_bytes(path.read_bytes().replace(b"1", b"2"))
