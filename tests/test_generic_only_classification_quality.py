"""Normal writer/portable reader fixtures; synthetic metadata, no inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import generic_only_classification_quality as admission
from onnx_splitpoint_tool.accuracy_reporting import DEFAULT_REPORTING_POLICY
from onnx_splitpoint_tool.quality_cache import json_fingerprint
from onnx_splitpoint_tool.quality_service import quality_request_from_manifest, prepare_evaluation
from test_v27911_composed_provenance_binding import _functions, RUNNER


def descriptor(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return descriptor(path)


def quality_fixture(tmp_path, monkeypatch, backend="hailo8", status="reference_close"):
    """Actual normal producer/export/strict reader; no mocked validator."""
    ns = _functions([
        "_quality_json_safe", "_quality_file_sha256", "_quality_contract_sha256", "_quality_artifact_binding",
        "_build_generic_composed_quality_producer_binding", "_build_classification_quality_contract",
        "_portable_dataset_manifest_sha256", "_export_central_quality_inputs", "_write_stable_quality_json",
        "_task_quality_execution_location", "_load_task_quality_policy",
    ])
    runner_sha = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    ns["_verified_suite_vendored_endpoint_attestor_sha256"] = lambda: runner_sha
    files = {}
    for role in ("source_model", "part1_runtime", "engine", "source_part1_onnx", "source_part2_onnx",
                 "build_part2_onnx", "engine_build_receipt", "boundary_metadata"):
        p = tmp_path / role; p.write_bytes(("synthetic metadata: " + role).encode()); files[role] = descriptor(p)
    rows = [{"image": f"synthetic_{i:04}.jpg", "label_id": 1,
             "gt": {"top1": 1, "top5": [1, 2, 3, 4, 5], "top1_hit": True, "top5_hit": True},
             "gt_reference": {"top1": 1, "top5": [1, 2, 3, 4, 5], "top1_hit": True, "top5_hit": True}}
            for i in range(5000)]
    manifest = tmp_path / "manifest.json"; manifest.write_text(json.dumps({"synthetic": True}))
    contract = ns["_build_classification_quality_contract"](model_path=Path(files["source_model"]["path"]),
        validation_source=manifest, rows=rows, image_scale="imagenet", letterbox=False,
        input_hw=(224, 224), input_dtype="float32", runner_path=RUNNER, endpoint_attestor_sha256=runner_sha)
    preprocessing = contract["preprocessing"]["identity"]
    receipt = {"source_onnx_sha256": files["source_part1_onnx"]["sha256"]}
    if backend == "deepx":
        receipt.update(dxnn_sha256=files["part1_runtime"]["sha256"], dxnn_size_bytes=files["part1_runtime"]["size_bytes"],
            classification_preprocessing="imagenet_mean_std", preprocessing_contract={"mode": "imagenet_mean_std",
                "dxcom_loader": {"scale_divisor": 255.0},
                "build_onnx_adapter": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}})
    else:
        receipt.update(hef_sha256=files["part1_runtime"]["sha256"], hef_size_bytes=files["part1_runtime"]["size_bytes"],
                       hw_arch=backend, preprocessing_contract=preprocessing)
    files["part1_build_receipt"] = document(tmp_path / "part1_build_receipt", receipt)
    policy = ns["_load_task_quality_policy"]({"statistics": {"execution_location": "central_management",
        "method": "paired_bootstrap", "bootstrap_repetitions": 1000, "seed": 20260710,
        "confidence_level": 0.95, "decision": "lower_one_sided_bound"}, "reporting_policy": DEFAULT_REPORTING_POLICY})
    monkeypatch.setenv("ONNX_SPLITPOINT_CPU_REFERENCE_ONLY", "1")
    ns["_export_central_quality_inputs"](out_dir=tmp_path / "cpu", task="classification", variant="full",
        policy=policy, classification_rows=rows, quality_contract=contract)
    monkeypatch.delenv("ONNX_SPLITPOINT_CPU_REFERENCE_ONLY")
    if status == "accuracy_loss":
        for item in rows:
            item["gt"] = {"top1":2,"top5":[2,3,4,5,6],"top1_hit":False,"top5_hit":False}
    cpu_path = tmp_path / "cpu/task_quality_inputs/canonical_classification_reference.json"
    cpu = descriptor(cpu_path)
    setup = f"test_{backend}"; run = {"hailo8": "hailo8_to_trt", "hailo10h": "hailo10h_to_trt", "deepx": "deepx_to_trt"}[backend]
    endpoint = {"task": "classification", "stage": "classification_logits", "contract_family": "classification_logits",
        "endpoint_contract_complete": True, "endpoint_contract_hash": "2" * 64,
        "full_model_sha256": files["source_model"]["sha256"], "terminal_model_sha256": files["source_part2_onnx"]["sha256"],
        "tensor_signature": {"tensor_count": 1, "tensors": [{"name": "logits", "shape": [1, 1000], "dtype": "float32", "index": 0, "rank": 2}]}}
    context = {"eval_run_id": "synthetic_reuse_fixture", "model_id": "resnet50", "setup_id": setup,
        "source_run_id": run, "case_id": "b028", "stage1_backend": "deepx_m1" if backend == "deepx" else backend,
        "stage2_backend": "tensorrt", "source_model_path": files["source_model"]["path"],
        "part1_model_path": files["source_part1_onnx"]["path"], "part2_model_path": files["source_part2_onnx"]["path"],
        "implementation_runner_path": str(RUNNER), "runtime_artifacts": [
            {"role": "stage1_runtime", "backend": "deepx_m1" if backend == "deepx" else backend, "path": files["part1_runtime"]["path"]},
            {"role": "stage2_runtime", "backend": "native_tensorrt", "path": files["engine"]["path"]}]}
    exported = ns["_export_central_quality_inputs"](out_dir=tmp_path / "capture", task="classification", variant="composed",
        policy=policy, classification_rows=rows, quality_contract=contract, endpoint_contract=endpoint,
        runtime_precision_identity="float32_layout_fp16", composed_producer_context=context)
    request_path = Path(exported["request"]["path"]); request = json.loads(request_path.read_text())
    loaded = quality_request_from_manifest(request_path, reference_artifact=cpu)
    _, paired = prepare_evaluation(loaded)
    from onnx_splitpoint_tool.quality_metrics import classification_quality_evaluator
    from onnx_splitpoint_tool.accuracy_reporting import assess_accuracy
    point = classification_quality_evaluator(paired["reference_records"],paired["candidate_records"],
        paired["annotations"],paired).evaluate(np.ones(5000,dtype=np.int64))
    assessment = assess_accuracy(point["primary"]["reference"],point["primary"]["candidate"],
        [1.,1.] if status == "accuracy_loss" else [0.,0.],policy=policy["reporting_policy"])
    # Explicit unit-test completed-result metadata. Genuine historical result
    # replay is recorded separately; this fixture runs no scientific worker.
    result = {**request, "schema": "onnx-splitpoint/management-paired-quality-result", "status": "completed",
        "technical_status": "completed", "cpu_reference_semantic_only": True, "artifact_provenance_binding_status": "verified",
        "source_request_sha256": descriptor(request_path)["sha256"],
        "management_cpu_reference": {"provider": "onnxruntime_cpu", "status": "completed", "reference_sha256": cpu["sha256"],
            "reference_size_bytes": cpu["size_bytes"], "semantic_reference_only": True},
        "validation_dataset_manifest_sha256": contract["dataset"]["manifest_sha256"], "n": 5000, "evaluated_images": 5000,
        "primary": {**point["primary"],"bootstrap_repetitions_requested": 1000},
        "guardrails":point["guardrails"], "accuracy_assessment": assessment,"reporting_policy":policy["reporting_policy"],
        **{k: paired[k] for k in ("reference_predictions_sha256", "candidate_predictions_sha256", "annotations_sha256", "seed_schema")}}
    reference = {"request": descriptor(request_path), "result": document(tmp_path / "result.json", result), "cpu_reference": cpu,
        "implementation_runner": descriptor(RUNNER),
        "policy_source": document(tmp_path / "profile.json", {"quality_gate": policy,
            "validation_execution": {"max_items": {"classification": 5000}}}),
        "policy_sha256": request["policy_sha256"], "dataset_manifest_sha256": contract["dataset"]["manifest_sha256"]}
    prepared_path = tmp_path / "prepared.npy"; np.save(prepared_path, np.full((224, 224, 3), 128, dtype=np.uint8))
    prepared = {**descriptor(prepared_path), "shape": [224, 224, 3], "dtype": "uint8", "layout": "HWC"}
    normalization = ("hef_quant_info_from_imagenet_float32" if backend == "hailo10h" else
                     "embedded_dxcom_preprocessing" if backend == "deepx" else "imagenet_mean_std")
    image_sha = hashlib.sha256(b"synthetic source image").hexdigest()
    identity = {"model_id": "resnet50", "case_id": "b028", "setup_id": setup, "direction": run,
        "comparison_backend": backend, "precision": "float32_layout_fp16",
        "output_endpoint_id": "classification:classification_logits:" + "2" * 64}
    return {"task": "classification", "identity": identity, "completion_contract": deepcopy(admission.CLASSIFICATION_CONTRACT),
        "source_model": files.pop("source_model"), "quality_reference": reference,
        "generic_artifact_binding": {"artifacts": files},
        "functional_reference": {"outputs": {"logits": {"shape": [1, 1000], "dtype": "float32"}}},
        "input_binding": {"prepared_input": prepared, "source_image_sha256": image_sha,
            "runtime_input_format": "float32" if backend == "hailo8" else "uint8",
            "prepared_input_contract": {"task": "classification", "layout": "HWC", "normalization": normalization,
                "source_input_contract": {"classification_preprocessing": "imagenet_mean_std",
                    "embedded_numeric_path": "dxcom_div255_then_build_onnx_imagenet_mean_std"} if backend == "deepx" else {},
                "source_image_sha256": image_sha, "timing_input_binding": {"preprocessing_contract": preprocessing}}}}


@pytest.mark.parametrize("backend", ["hailo8", "hailo10h", "deepx"])
def test_normal_classifier_writer_and_reader_reuse_all_backends(tmp_path, monkeypatch, backend):
    row = quality_fixture(tmp_path, monkeypatch, backend)
    proof = admission.validate_classification_quality_reference(row)
    assert proof["quality_eligible"] is True
    assert proof["historical_native_performance_imported"] is False
    assert proof["quality_input_adapter"]["raw_prepared_bytes_equal_sdk_input_claimed"] is False


@pytest.mark.parametrize("status", ["accuracy_loss"])
def test_historical_quality_loss_stays_measurable_but_not_eligible(tmp_path, monkeypatch, status):
    row = quality_fixture(tmp_path, monkeypatch, status=status)
    proof = admission.validate_classification_quality_reference(row)
    assert proof["status"] == status and proof["quality_eligible"] is False


def test_existing_raw_h8_prepared_rgb_uses_normal_bound_array_loader(tmp_path, monkeypatch):
    row = quality_fixture(tmp_path, monkeypatch)
    previous = row["input_binding"]["prepared_input"]
    path = tmp_path / "prepared.bin"
    np.load(previous["path"], allow_pickle=False).tofile(path)
    row["input_binding"]["prepared_input"] = {**previous, **descriptor(path), "format": "raw"}
    assert admission.validate_classification_quality_reference(row)["quality_eligible"] is True


def test_actual_normal_quality_and_timed_ranking_match_ties_and_float32_conversion():
    from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
    ns = _functions(["_pick_classification_logits", "_classification_logits_vector", "_classification_topk", "_softmax1d", "_classification_gt_metrics"])
    values = np.zeros((1, 1000), dtype=np.float64)
    values[0, :8] = [1.0, 1.0, -2.0, 1.0 + 1e-12, 1000.0, -1000.0, 0.9, 0.8]
    expected = ns["_classification_topk"](ns["_classification_logits_vector"](["logits"], [values]), k=5)
    processor = ClassificationCompletion(); actual = processor.process({"logits": values})
    assert actual["top1"].tolist() == [expected[0]]
    assert actual["top5"].tolist() == [expected[1]]


def test_altered_archived_ranking_implementation_is_not_reusable(tmp_path, monkeypatch):
    row = quality_fixture(tmp_path, monkeypatch)
    archived = tmp_path / "archived_runner.py"
    archived.write_text(RUNNER.read_text().replace('order = np.argsort(-vec, kind="stable")', 'order = np.argsort(vec, kind="stable")'))
    row["quality_reference"]["implementation_runner"] = descriptor(archived)
    with pytest.raises(ValueError, match="implementation differs"):
        admission.validate_classification_quality_reference(row)


@pytest.mark.parametrize("fault", ["precision", "endpoint", "compiled", "source", "topk_offset", "preprocess", "normalization", "shape", "prepared_bytes"])
def test_current_target_drift_is_not_historical_reuse(tmp_path, monkeypatch, fault):
    row = quality_fixture(tmp_path, monkeypatch)
    if fault == "precision": row["identity"]["precision"] = "uint8_dequant_fp16"
    elif fault == "endpoint": row["identity"]["output_endpoint_id"] += "changed"
    elif fault == "compiled": row["generic_artifact_binding"]["artifacts"]["engine"]["sha256"] = "f" * 64
    elif fault == "source": row["source_model"]["sha256"] = "f" * 64
    elif fault == "topk_offset": row["completion_contract"]["label_offset"] = 1
    elif fault == "preprocess": row["input_binding"]["prepared_input_contract"]["timing_input_binding"]["preprocessing_contract"]["letterbox"] = True
    elif fault == "normalization": row["input_binding"]["prepared_input_contract"]["normalization"] = "raw_uint8"
    elif fault == "shape": row["functional_reference"]["outputs"]["logits"]["shape"] = [1, 999]
    else: Path(row["input_binding"]["prepared_input"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError): admission.validate_classification_quality_reference(row)


@pytest.mark.parametrize("fault", ["candidate_predictions_sha256", "seed_schema", "n", "cpu", "role", "result_policy", "profile_policy", "accuracy_class", "primary_hits", "top5_hits", "reporting_policy"])
def test_rehashed_result_or_policy_substitution_rejects(tmp_path, monkeypatch, fault):
    row = quality_fixture(tmp_path, monkeypatch)
    reference = row["quality_reference"]; p = Path(reference["result"]["path"]); result = json.loads(p.read_text())
    if fault == "cpu": result["management_cpu_reference"]["reference_sha256"] = "f" * 64
    elif fault == "role": result["performance_claims_emitted"] = True
    elif fault == "result_policy": result["policy_sha256"] = "f" * 64
    elif fault == "profile_policy":
        profile_path = Path(reference["policy_source"]["path"]); profile = json.loads(profile_path.read_text())
        profile["quality_gate"]["statistics"]["seed"] += 1
        reference["policy_source"] = document(profile_path, profile)
    elif fault == "n": result["n"] = 4999
    elif fault == "accuracy_class": result["accuracy_assessment"]["accuracy_class"] = "accuracy_loss"
    elif fault == "primary_hits": result["primary"]["candidate_hits"] = 4999
    elif fault == "top5_hits": result["guardrails"]["top5_accuracy"]["candidate_hits"] = 4999
    elif fault == "reporting_policy": result["reporting_policy"] = {}
    elif fault == "seed_schema": result["seed_schema"]["seed"] += 1
    else: result[fault] = "f" * 64
    reference["result"] = document(p, result)
    with pytest.raises(ValueError): admission.validate_classification_quality_reference(row)
