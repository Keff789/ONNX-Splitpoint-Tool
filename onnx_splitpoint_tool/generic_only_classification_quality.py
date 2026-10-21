"""Reuse authentic classifier Quality without creating Native performance.

The normal portable reader remains authoritative for labels, CPU predictions,
producer contracts and Image-ID pairing. This admission only joins that saved
evidence to the selected compiled parts and the ordinary timed input adapter;
it performs no inference, dataset evaluation or bootstrap.
"""
from __future__ import annotations

from collections.abc import Mapping
import ast
import json
import math
from pathlib import Path

from .quality_cache import json_fingerprint


CLASSIFICATION_CONTRACT = {"task": "classification", "operation": "top1_top5", "label_offset": 0}


def _implementation_relation(reference, quality):
    """Verify existing producer bytes and unchanged normal ranking functions."""
    from .generic_only_plan import read_bound, file_sha256
    source = reference.get("implementation_runner")
    archived = read_bound(source)
    if source["sha256"] != quality["postprocessor"]["identity"].get("implementation_runner_sha256"):
        raise ValueError("Classifier archived implementation differs from its genuine Quality producer")
    current = Path(__file__).parent / "resources/templates/run_split_onnxruntime.py.txt"
    names = ("_pick_classification_logits", "_classification_logits_vector", "_classification_topk",
             "_classification_gt_metrics", "_softmax1d")
    def functions(path):
        return {item.name: ast.dump(item, include_attributes=False) for item in ast.parse(path.read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name in names}
    old, new = functions(archived), functions(current)
    if set(old) != set(names) or old != new:
        raise ValueError("Classifier historical/current normal ranking implementation differs")
    return {"historical_runner_sha256": source["sha256"], "current_runner_sha256": file_sha256(current),
            "unchanged_quality_functions": list(names), "timed_operation": "stable_float32_top1_top5_label_offset_zero"}


def _token(value):
    return str(value or "").removeprefix("sha256:")


def _same_artifact(actual, selected, role):
    if (not isinstance(actual, Mapping) or not isinstance(selected, Mapping)
            or len(_token(actual.get("sha256"))) != 64
            or actual.get("sha256") != selected.get("sha256")
            or type(actual.get("size_bytes")) is not int or actual["size_bytes"] < 1
            or actual.get("size_bytes") != selected.get("size_bytes")):
        raise ValueError("Classifier Quality compiled/source artifact differs: " + role)


def _preprocessing_input(row, quality_contract, receipt):
    from .generic_only_plan import read_bound, file_sha256
    from .runners.task_completion import load_completion_bound_array

    expected = {
        "schema": "onnx-splitpoint/image-preprocessing-contract", "schema_version": 2,
        "contract_scope": "prepared_rgb_uint8_semantics", "task": "classification",
        "preprocess_mode": "resize", "spatial_transform": "direct_resize",
        "target_hw": [224, 224], "color_space": "RGB", "input_domain": "uint8_0_255",
        "resize_interpolation": "bilinear", "resize_rounding": "python_round_ties_to_even",
        "placement": "not_applicable", "pad_value": 0, "letterbox_pad_value": 0,
        "image_scale": "imagenet", "letterbox": False,
    }
    preprocessing = quality_contract.get("preprocessing") or {}
    if (preprocessing.get("identity") != expected
            or preprocessing.get("sha256") != json_fingerprint(expected)):
        raise ValueError("Classifier Quality canonical RGB preprocessing differs")
    binding = row.get("input_binding") or {}
    contract = binding.get("prepared_input_contract") or {}
    source_input = contract.get("source_input_contract") or {}
    observed = ((contract.get("timing_input_binding") or {}).get("preprocessing_contract")
                or binding.get("preprocessing_contract"))
    if (observed != expected or contract.get("task") != "classification"
            or contract.get("source_image_sha256") != binding.get("source_image_sha256")
            or len(_token(binding.get("source_image_sha256"))) != 64):
        raise ValueError("Classifier timing input lacks its bound canonical preprocessing/source image")
    prepared = binding.get("prepared_input") or {}
    array = load_completion_bound_array({**prepared, "path": str(read_bound(prepared))},
                                        file_sha256=file_sha256)
    layout = prepared.get("layout") or contract.get("layout") or source_input.get("layout")
    if (layout not in {"HWC", "CHW", "NHWC", "NCHW"}
            or list(array.shape) != prepared.get("shape") or str(array.dtype) != prepared.get("dtype")):
        raise ValueError("Classifier timing prepared tensor shape/dtype/layout differs")
    expected_shape = {"HWC": [224, 224, 3], "CHW": [3, 224, 224],
                      "NHWC": [1, 224, 224, 3], "NCHW": [1, 3, 224, 224]}[layout]
    if prepared["shape"] != expected_shape:
        raise ValueError("Classifier timing prepared tensor does not match the Quality image shape")
    backend = row["identity"]["comparison_backend"]
    normalization = prepared.get("normalization") or contract.get("normalization") or source_input.get("normalization")
    runtime_format = binding.get("runtime_input_format")
    encoding = preprocessing.get("runtime_input_encoding") or {}
    if encoding.get("image_scale") != "imagenet" or encoding.get("input_dtype") != "float32":
        raise ValueError("Classifier Quality ONNX numeric domain differs")
    if backend == "hailo8":
        if (str(array.dtype) != "uint8" or normalization != "imagenet_mean_std"
                or runtime_format not in {"float32", "dequant_float32"}
                or receipt.get("preprocessing_contract") != expected
                or receipt.get("hw_arch") != "hailo8"):
            raise ValueError("H8 classifier lacks the ordinary RGB-to-ImageNet FLOAT32 adapter relation")
        relation = "prepared_rgb_uint8_to_float32_div255_then_imagenet_mean_std_then_existing_hailo_layout"
    elif backend in {"hailo10", "hailo10h"}:
        expected_normalization = ("hef_quant_info_from_imagenet_float32" if str(array.dtype) == "uint8"
                                  else "imagenet_float32")
        expected_formats = {"uint8", "raw_uint8"} if str(array.dtype) == "uint8" else {"float32", "dequant_float32"}
        if (normalization != expected_normalization or runtime_format not in expected_formats
                or str(array.dtype) not in {"uint8", "float32"}
                or receipt.get("preprocessing_contract") != expected
                or receipt.get("hw_arch") not in {"hailo10", "hailo10h"}):
            raise ValueError("H10 classifier prepared domain/HEF relation differs")
        relation = expected_normalization + "_bound_to_exact_same_hef"
    elif backend in {"deepx", "deepx_m1"}:
        deepx = receipt.get("preprocessing_contract") or {}
        adapter = deepx.get("build_onnx_adapter") or {}
        loader = deepx.get("dxcom_loader") or {}
        if (str(array.dtype) != "uint8" or layout != "HWC"
                or normalization != "embedded_dxcom_preprocessing"
                or runtime_format not in {"uint8", "raw_uint8"}
                or source_input.get("classification_preprocessing") != "imagenet_mean_std"
                or source_input.get("embedded_numeric_path") != "dxcom_div255_then_build_onnx_imagenet_mean_std"
                or receipt.get("classification_preprocessing") != "imagenet_mean_std"
                or deepx.get("mode") != "imagenet_mean_std"
                or loader.get("scale_divisor") != 255.0
                or adapter.get("mean") != [0.485, 0.456, 0.406]
                or adapter.get("std") != [0.229, 0.224, 0.225]):
            raise ValueError("DeepX classifier RGB/embedded ImageNet adapter relation differs")
        relation = "prepared_rgb_uint8_to_dxcom_div255_then_bound_build_onnx_imagenet_mean_std"
    else:
        raise ValueError("Classifier Quality backend is unsupported")
    return {"canonical_preprocessing_sha256": preprocessing["sha256"],
            "prepared_input_sha256": prepared["sha256"], "runtime_adapter_relation": relation,
            "raw_prepared_bytes_equal_sdk_input_claimed": False}


def validate_classification_quality_reference(row):
    """Verify saved N5000/B1000 Quality against one exact current classifier."""
    from .generic_only_plan import _json, read_bound
    from .accuracy_gates import AccuracyGatePolicy
    from .accuracy_reporting import assess_accuracy
    from .quality_metrics import classification_quality_evaluator
    from .quality_service import quality_request_from_manifest, prepare_evaluation, _validate_classification_quality_contract
    from .quality_service import _validate_generic_composed_candidate_execution_contract
    from .native_split_quality import validate_native_split_quality_binding, _backend
    import yaml

    if row.get("task") != "classification" or row.get("completion_contract") != CLASSIFICATION_CONTRACT:
        raise ValueError("Classifier Quality requires the unchanged Top1/Top5 zero-offset completion")
    identity = row["identity"]
    reference = row.get("quality_reference") or {}
    request_path = read_bound(reference.get("request"))
    result = _json(reference.get("result"))
    cpu_path = read_bound(reference.get("cpu_reference"))
    policy_document = yaml.safe_load(read_bound(reference.get("policy_source")).read_text())
    policy = policy_document.get("quality_gate", policy_document)
    request = json.loads(request_path.read_text())
    if (result.get("schema") != "onnx-splitpoint/management-paired-quality-result"
            or result.get("status") != "completed" or result.get("technical_status") != "completed"
            or result.get("task") != "classification" or result.get("variant") != "composed"
            or result.get("cpu_reference_semantic_only") is not True
            or result.get("artifact_provenance_binding_status") != "verified"):
        raise ValueError("Classifier Quality must be a completed normal CPU comparison")
    for key in ("model_id", "case_id"):
        if result.get(key) != identity[key] or request.get(key) != identity[key]:
            raise ValueError("Classifier Quality belongs to another model/case")
    for value in (request, result):
        setup = value.get("setup_id") or value.get("source_setup_id")
        if setup != identity["setup_id"] or value.get("runtime_precision_identity") != identity["precision"]:
            raise ValueError("Classifier Quality setup/precision differs")
        if value.get("setup_id") and value.get("source_setup_id") and value["setup_id"] != value["source_setup_id"]:
            raise ValueError("Classifier Quality setup aliases contradict")
        if (_backend(value.get("source_run_id")) != _backend(identity["direction"])
                or value.get("performance_claims_emitted") is not False):
            raise ValueError("Classifier Quality source run/performance role differs")
    if _token(result.get("source_request_sha256")) != reference["request"]["sha256"]:
        raise ValueError("Classifier Quality request/result source binding differs")
    cpu = result.get("management_cpu_reference") or {}
    if (cpu.get("provider") != "onnxruntime_cpu" or cpu.get("status") != "completed"
            or cpu.get("reference_sha256") != reference["cpu_reference"]["sha256"]
            or cpu.get("reference_size_bytes") != cpu_path.stat().st_size
            or cpu.get("semantic_reference_only") is not True):
        raise ValueError("Classifier Quality CPU reference differs")
    loaded = quality_request_from_manifest(request_path, verify_artifacts=True,
        reference_artifact={"path": str(cpu_path), "sha256": reference["cpu_reference"]["sha256"],
                            "size_bytes": cpu_path.stat().st_size})
    _, prepared_evaluation = prepare_evaluation(loaded)
    if (any(result.get(key) != prepared_evaluation[key] for key in (
            "reference_predictions_sha256", "candidate_predictions_sha256", "annotations_sha256", "seed_schema"))
            or type(result.get("evaluated_images")) is not int or result["evaluated_images"] != 5000
            or type((result.get("primary") or {}).get("bootstrap_repetitions_requested")) is not int
            or result["primary"]["bootstrap_repetitions_requested"] != 1000):
        raise ValueError("Classifier saved result differs from the actual paired prediction/statistics bytes")
    quality, _ = _validate_classification_quality_contract(request.get("quality_contract"), role="Classifier source")
    source_model = row.get("source_model") or (row.get("generic_artifact_binding") or {}).get("source_model")
    read_bound(source_model)
    if quality["model"]["sha256"] != source_model["sha256"]:
        raise ValueError("Classifier Quality full source ONNX differs")
    artifacts = row["generic_artifact_binding"]["artifacts"]
    receipt = json.loads(read_bound(artifacts["part1_build_receipt"]).read_text())
    if receipt.get("source_onnx_sha256") != artifacts["source_part1_onnx"]["sha256"]:
        raise ValueError("Classifier P1 receipt/source ONNX ancestry differs")
    for role in ("source_part1_onnx", "source_part2_onnx"):
        read_bound(artifacts[role])
    backend = identity["comparison_backend"]
    binary_hash_key, binary_size_key = (("dxnn_sha256", "dxnn_size_bytes")
        if backend in {"deepx", "deepx_m1"} else ("hef_sha256", "hef_size_bytes"))
    _same_artifact({"sha256": receipt.get(binary_hash_key), "size_bytes": receipt.get(binary_size_key)},
                   artifacts["part1_runtime"], "part1_build_receipt")
    producer = request.get("producer_identity")
    if producer:
        producer, producer_sha = _validate_generic_composed_candidate_execution_contract(
            producer, role="Classifier source", task="classification")
        if (result.get("producer_identity") != producer or result.get("producer_identity_sha256") != producer_sha
                or result.get("execution_role") != "generic_composed_quality_candidate"):
            raise ValueError("Classifier saved Generic producer differs")
        physical = {item["role"]: item for item in producer["runtime_artifacts"]}
        for source_role, role in (("stage1_runtime", "part1_runtime"), ("stage2_runtime", "engine")):
            _same_artifact(physical.get(source_role), artifacts[role], role)
        for source_role, role in (("part1_onnx_sha256", "source_part1_onnx"), ("part2_onnx_sha256", "source_part2_onnx")):
            if producer["model"][source_role] != artifacts[role]["sha256"]:
                raise ValueError("Classifier Quality split ONNX differs: " + role)
        historical_kind = "normal_generic_composed_quality"
        historical_binding_sha = None
    else:
        binding, status = validate_native_split_quality_binding(request.get("native_split_quality_binding"),
            expected_identity={"model": identity["model_id"], "case": identity["case_id"],
                "setup_id": identity["setup_id"], "backend": request.get("source_run_id"),
                "task": "classification", "precision": identity["precision"]})
        if (binding is None or result.get("native_split_quality_binding") != binding
                or result.get("native_split_quality_binding_sha256") != binding["binding_sha256"]):
            raise ValueError("Classifier historical split Quality proof differs: " + status)
        for role in ("part1_runtime", "engine", "source_part2_onnx", "build_part2_onnx", "engine_build_receipt", "boundary_metadata"):
            _same_artifact(binding["artifacts"].get(role), artifacts[role], role)
        producer_sha = None  # This genuine old source has no Generic producer identity.
        historical_binding_sha = binding["binding_sha256"]
        historical_kind = "normal_historical_split_quality_without_native_performance"
    endpoint = request.get("endpoint_contract") or {}
    endpoint_sha = endpoint.get("endpoint_contract_hash")
    if (endpoint.get("task") != "classification" or endpoint.get("stage") not in {"classification_logits", "classification_probabilities"}
            or endpoint.get("endpoint_contract_complete") is not True
            or identity["output_endpoint_id"] != f"classification:{endpoint['stage']}:{endpoint_sha}"
            or result.get("endpoint_contract_hash") != endpoint_sha
            or endpoint.get("full_model_sha256") != source_model["sha256"]
            or endpoint.get("terminal_model_sha256") not in {artifacts["source_part2_onnx"]["sha256"], artifacts["build_part2_onnx"]["sha256"]}):
        raise ValueError("Classifier Quality physical endpoint differs")
    observed = (endpoint.get("tensor_signature") or {}).get("tensors") or []
    outputs = (row.get("functional_reference") or {}).get("outputs") or {}
    if (len(observed) != 1 or {item.get("name"): {"shape": item.get("shape"), "dtype": item.get("dtype")}
            for item in observed} != outputs):
        raise ValueError("Classifier physical output names/shapes/dtypes differ")
    post = quality["postprocessor"]["identity"]
    if (post.get("topk") != 5 or post.get("sort_order") != "score_descending"
            or post.get("canonical_record_endpoint") != "classification_topk_hits"):
        raise ValueError("Classifier Quality Top1/Top5 semantics differ")
    implementation = _implementation_relation(reference, quality)
    policy_sha = AccuracyGatePolicy.from_mapping(policy).sha256()
    metric = {**policy.get("classification", {}), "reporting_policy": policy.get("reporting_policy")}
    if (any(_token(value.get("policy_sha256")) != policy_sha for value in (request, result, reference))
            or request.get("metric_gate_config") != metric or result.get("metric_gate_config") != metric
            or any(request.get("statistics", {}).get(key) != policy.get("statistics", {}).get(key)
                   for key in ("method", "confidence_level", "bootstrap_repetitions", "seed", "decision"))
            or loaded.repetitions != 1000 or policy.get("statistics", {}).get("bootstrap_repetitions") != 1000
            or len(loaded.reference_records) != 5000 or len(loaded.candidate_records) != 5000
            or type(result.get("n")) is not int or result["n"] != 5000
            or ((policy_document.get("validation_execution") or {}).get("max_items") or {}).get("classification") != 5000
            or reference.get("dataset_manifest_sha256") != quality["dataset"]["manifest_sha256"]
            or result.get("validation_dataset_manifest_sha256") != quality["dataset"]["manifest_sha256"]):
        raise ValueError("Classifier Quality N5000/B1000 policy/dataset contract differs")
    if any(result.get(key) != request.get(key) for key in (
            "quality_contract_sha256", "endpoint_contract_hash", "candidate_execution_completion_contract_sha256")):
        raise ValueError("Classifier Quality result/producer endpoint contract differs")
    adapter = _preprocessing_input(row, quality, receipt)
    # One deterministic count over the already verified paired records. Reuse
    # the saved bootstrap interval, never run a new bootstrap or inference.
    import numpy as np
    point = classification_quality_evaluator(prepared_evaluation["reference_records"],
        prepared_evaluation["candidate_records"], prepared_evaluation["annotations"],
        prepared_evaluation).evaluate(np.ones(5000, dtype=np.int64))
    for expected, actual in ((point["primary"], result.get("primary") or {}),
            (point["guardrails"]["top5_accuracy"], (result.get("guardrails") or {}).get("top5_accuracy") or {})):
        if (any(actual.get(key) != expected.get(key) for key in
                ("metric", "reference_hits", "candidate_hits", "sample_count", "reference", "candidate", "margin"))
                or isinstance(actual.get("delta"), bool)
                or not isinstance(actual.get("delta"), (int, float))
                or not math.isclose(actual["delta"], expected["delta"], rel_tol=0, abs_tol=1e-14)):
            raise ValueError("Classifier saved Top1/Top5 metrics contradict the actual paired records")
    assessment = result.get("accuracy_assessment") or {}
    if (result.get("reporting_policy") != policy.get("reporting_policy")
            or assess_accuracy(point["primary"]["reference"], point["primary"]["candidate"],
                assessment.get("relative_loss_ci"), policy=policy.get("reporting_policy")) != assessment):
        raise ValueError("Classifier saved accuracy decision contradicts the normal reporting policy")
    status = (result.get("accuracy_assessment") or {}).get("accuracy_class")
    if status not in {"reference_close", "accuracy_loss", "inconclusive"}:
        raise ValueError("Classifier Quality classification is absent or technically failed")
    return {"status": status, "quality_eligible": status == "reference_close",
        "policy_sha256": policy_sha, "dataset_manifest_sha256": reference["dataset_manifest_sha256"],
        "producer_identity_sha256": producer_sha, "historical_source_kind": historical_kind,
        "historical_quality_binding_sha256": historical_binding_sha,
        "quality_input_adapter": adapter, "request_sha256": reference["request"]["sha256"],
        "implementation_relation": implementation,
        "result_sha256": reference["result"]["sha256"], "historical_native_performance_imported": False}
