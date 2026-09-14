from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from onnx_splitpoint_tool.quality_cache import canonical_json, json_fingerprint
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"quality_first_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_quality_fixture_module():
    path = ROOT / "tests" / "test_v269d_trt_central_quality_producer.py"
    spec = importlib.util.spec_from_file_location("quality_first_source_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seal_producer(producer: dict) -> dict:
    producer = copy.deepcopy(producer)
    producer.pop("producer_identity_sha256", None)
    producer["producer_identity_sha256"] = json_fingerprint(producer)
    return producer


def _quality_first_fixture() -> tuple[dict, dict, dict, AccuracyGatePolicy]:
    source = _load_quality_fixture_module()
    policy = AccuracyGatePolicy()
    producer = copy.deepcopy(source._producer())
    producer["source_run_id"] = "native_full_tensorrt"
    producer["policy_sha256"] = policy.sha256()
    producer["engine_build_receipt_file_sha256"] = producer[
        "engine_build_receipt"
    ]["sha256"]
    producer = _seal_producer(producer)

    build = producer["build_onnx"]
    engine = producer["engine"]
    trtexec = producer["trtexec"]
    receipt_binding = producer["engine_build_receipt"]
    artifacts = {
        "command_python_executable": {
            "path": "/evidence/python3", "invocation_path": "/evidence/python3",
            "sha256": "0" * 64, "interpreter_identity": {},
        },
        "trtexec": copy.deepcopy(trtexec),
        "source_onnx": copy.deepcopy(build),
        "engine": {
            **copy.deepcopy(engine),
            "compiled_from_source_onnx_sha256": producer["source_onnx"]["sha256"],
        },
        "input_manifest": {
            "path": "/evidence/input.json", "sha256": "e" * 64,
        },
        "runtime_input_tensor": {
            "path": "/evidence/input.bin", "sha256": "f" * 64,
        },
        "engine_build_receipt": {
            "path": receipt_binding["path"],
            "sha256": producer["engine_build_receipt_file_sha256"],
            "size_bytes": receipt_binding["size_bytes"],
            "file_size_bytes": receipt_binding["size_bytes"],
        },
    }
    contract = {
        "schema": "onnx-splitpoint/native-full-command-contract",
        "schema_version": 1, "complete": True,
        "backend": "native_full_tensorrt", "backend_arg": "tensorrt",
        "model": producer["model_id"], "case": "full",
        "setup_id": producer["setup_id"], "comparison_backend": "hailo8",
        "comparison_precision": "fp16", "legacy_comparison_precision": "fp16",
        "execution_precision": "fp16", "full_runtime_precision": "fp16",
        "python_executable": "/evidence/python3",
        "runner": "scripts/native_full_baseline_eval_runner.py",
        "runner_sha256": "d" * 64, "root": "/evidence",
        "benchmark_set": "/evidence/benchmark_set", "input_case": "b052",
        "input_image": "/evidence/a.jpg", "input_image_sha256": "e" * 64,
        "source_model_sha256": producer["source_onnx"]["sha256"],
        "trt_engine_build_receipt": copy.deepcopy(receipt_binding["receipt"]),
        "trt_engine_build_receipt_status": "engine_build_receipt_verified",
        "engine_build_receipt_path": receipt_binding["path"],
        "engine_build_receipt_sha256": receipt_binding["sha256"],
        "engine_build_receipt_file_sha256": producer[
            "engine_build_receipt_file_sha256"
        ],
        "trt_engine_build_receipt_sha256": receipt_binding[
            "receipt"
        ]["receipt_sha256"],
        "engine_build_receipt_size_bytes": receipt_binding["size_bytes"],
        "engine_build_receipt_file_size_bytes": receipt_binding["size_bytes"],
        "quality_first_producer_identity": copy.deepcopy(producer),
        "quality_first_producer_identity_sha256": producer[
            "producer_identity_sha256"
        ],
        "model_binding": {
            "source_artifact": "source_onnx",
            "source_onnx_sha256": build["sha256"],
            "compiled_artifact": "engine",
            "compiled_artifact_sha256": engine["sha256"],
            "status": "verified_engine_build_receipt_bound",
        },
        "runtime_options": {
            "frames": 1000, "warmup": 100, "inflight": 4,
            "trt_precision": "fp16", "workspace_mb": 4096,
            "engine_build_python": "/evidence/python3", "no_shapes": False,
            "dump_outputs": True, "diagnostic_deepx_input_probes": False,
            "image_map": {},
        },
        "energy_workload": {
            "available": True, "kind": "tensorrt_full_hotloop",
            "trtexec_artifact": "trtexec", "engine_artifact": "engine",
            "source_model_artifact": "source_onnx",
            "engine_build_receipt_artifact": "engine_build_receipt",
            "engine_build_receipt_status": "engine_build_receipt_verified",
            "input_manifest_artifact": "input_manifest",
            "runtime_input_artifact": "runtime_input_tensor",
            "runtime_input_name": "images",
            "input_mode": "exact_semantic_dump_runtime_tensor",
        },
        "artifacts": artifacts,
    }
    contract["contract_sha256"] = hashlib.sha256(canonical_json(contract).encode()).hexdigest()

    endpoint_sha = producer["endpoint_contract_hash"]
    endpoint_stage = producer["endpoint"]["identity"]["stage"]
    attestation = {
        "attested": True, "status": "passed", "task": producer["task"],
        "stage": endpoint_stage, "endpoint": endpoint_stage,
        "endpoint_contract_hash": endpoint_sha,
    }
    row = {
        "backend": "native_full_tensorrt", "model": producer["model_id"],
        "case": "full", "setup_id": producer["setup_id"],
        "comparison_backend": "hailo8", "task": producer["task"],
        "execution_mode": "native_full_baseline", "execution_precision": "fp16",
        "full_runtime_precision": "fp16", "runtime_precision_identity": "fp16",
        "endpoint_contract_complete": True, "endpoint_contract_hash": endpoint_sha,
        "stage": endpoint_stage, "contract_family": endpoint_stage,
        "output_endpoint_attestation": attestation,
        "full_command_contract": contract,
        "full_command_contract_sha256": contract["contract_sha256"],
        "quality_first_producer_identity": copy.deepcopy(producer),
        "quality_first_producer_identity_sha256": producer["producer_identity_sha256"],
        "quality_first_semantic_dump_binding_valid": True,
    }

    quality = producer["quality_contract"]
    dataset = producer["dataset"]
    request_sha = "1" * 64
    identity = {
        "schema_version": 4, "identity_valid": True,
        "producer_identity_validated": True, "producer_binding_eligible": True,
        "producer_identity": copy.deepcopy(producer),
        "producer_identity_sha256": producer["producer_identity_sha256"],
        "model_id": producer["model_id"], "task": producer["task"],
        "case_id": "full", "source_run_id": "native_full_tensorrt",
        "setup_id": producer["setup_id"], "variant": "full",
        "source_request_sha256": request_sha,
        "endpoint_contract_hash": endpoint_sha,
        "runtime_precision_identity": "fp16",
    }
    result = {
        **identity,
        "request_identity": copy.deepcopy(identity),
        "source_setup_id": producer["setup_id"],
        "status": "completed", "technical_status": "completed",
        "decision": "pass", "producer_binding_eligible": True,
        "source_model_sha256": producer["source_onnx"]["sha256"],
        "model_sha256": producer["source_onnx"]["sha256"],
        "validation_dataset_sha256": dataset["manifest_sha256"],
        "validation_dataset_image_ids_sha256": dataset["image_ids_sha256"],
        "validation_dataset_ground_truth_sha256": dataset["ground_truth_sha256"],
        "policy_sha256": policy.sha256(),
        "task_quality_policy_sha256": policy.sha256(),
        "quality_contract_sha256": producer["quality_contract_sha256"],
        "preprocessing_contract_sha256": producer["preprocessing_contract_sha256"],
        "decoder_contract_sha256": producer["decoder_contract_sha256"],
        "nms_contract_sha256": producer["nms_contract_sha256"],
        "quality_record_endpoint_contract_sha256": producer[
            "quality_record_endpoint_contract_sha256"
        ],
        "quality_contract": copy.deepcopy(quality),
        "source_request_sha256": request_sha,
        "primary": {"metric": "top1_accuracy", "delta": 0.0, "ci_low": 0.0, "margin": 0.01},
    }
    for field in (
        "source_model_sha256", "model_sha256", "validation_dataset_sha256",
        "validation_dataset_image_ids_sha256",
        "validation_dataset_ground_truth_sha256", "policy_sha256",
        "task_quality_policy_sha256", "quality_contract_sha256",
        "preprocessing_contract_sha256", "decoder_contract_sha256",
        "nms_contract_sha256", "quality_record_endpoint_contract_sha256",
    ):
        identity[field] = result[field]
    inner_receipt = copy.deepcopy(receipt_binding["receipt"])
    inner_receipt.pop("receipt_sha256")
    flat = {
        "producer_identity_sha256": producer["producer_identity_sha256"],
        "eval_run_id": producer["eval_run_id"],
        "model_id": producer["model_id"],
        "setup_id": producer["setup_id"],
        "source_run_id": producer["source_run_id"],
        "originating_plan_run_id": producer.get("originating_plan_run_id", ""),
        "case_id": producer["case_id"],
        "execution_role": producer["execution_role"],
        "backend": producer["backend"],
        "variant": producer["variant"], "task": producer["task"],
        "performance_claims_emitted": False,
        "source_onnx_path": producer["source_onnx"]["path"],
        "source_onnx_sha256": producer["source_onnx"]["sha256"],
        "source_onnx_size_bytes": producer["source_onnx"]["size_bytes"],
        "source_model_sha256": producer["source_onnx"]["sha256"],
        "source_model_size_bytes": producer["source_onnx"]["size_bytes"],
        "build_onnx_path": producer["build_onnx"]["path"],
        "build_onnx_sha256": producer["build_onnx"]["sha256"],
        "build_onnx_size_bytes": producer["build_onnx"]["size_bytes"],
        "engine_path": producer["engine"]["path"],
        "engine_sha256": producer["engine"]["sha256"],
        "engine_size_bytes": producer["engine"]["size_bytes"],
        "runtime_artifact_sha256": producer["engine"]["sha256"],
        "runtime_artifact_size_bytes": producer["engine"]["size_bytes"],
        "trtexec_path": producer["trtexec"]["path"],
        "trtexec_sha256": producer["trtexec"]["sha256"],
        "trtexec_size_bytes": producer["trtexec"]["size_bytes"],
        "engine_build_receipt_path": receipt_binding["path"],
        "engine_build_receipt_sha256": receipt_binding["sha256"],
        "engine_build_receipt_file_sha256": producer[
            "engine_build_receipt_file_sha256"
        ],
        "engine_build_receipt_size_bytes": receipt_binding["size_bytes"],
        "trt_engine_build_receipt_sha256": receipt_binding["receipt"]["receipt_sha256"],
        "trt_engine_build_receipt_size_bytes": len(canonical_json(inner_receipt).encode()),
    }
    result.update(copy.deepcopy(flat))
    identity.update(copy.deepcopy(flat))
    # Central Quality mirrors the logical backend while retaining the signed
    # execution backend separately.  This is the production shape emitted by
    # the setup-local Full-TensorRT quality path.
    result["backend"] = "tensorrt"
    identity["backend"] = "tensorrt"
    identity["producer_backend"] = producer["backend"]
    result["request_identity"] = copy.deepcopy(identity)
    return row, result, producer, policy


def test_native_validator_requires_exact_central_quality_first_producer() -> None:
    module = _load_script("native_producer_validate_visualize.py")
    row, result, producer, policy = _quality_first_fixture()
    module._bind_central_quality_evidence(row, [result], policy)
    assert row["central_quality_evidence_verified"] is True
    assert row["quality_first_binding_status"] == "central_native_exact_identity_match"
    assert row["quality_first_producer_identity"] == producer

    mirrored = copy.deepcopy(row)
    module._bind_central_quality_evidence(
        mirrored, [result, copy.deepcopy(result)], policy,
    )
    assert mirrored["central_quality_evidence_verified"] is True
    assert mirrored["central_quality_binding_raw_candidate_count"] == 2
    assert mirrored["central_quality_binding_candidate_count"] == 1

    result_drift = copy.deepcopy(result)
    result_drift["primary"]["delta"] = 0.001
    drifted = copy.deepcopy(row)
    module._bind_central_quality_evidence(drifted, [result, result_drift], policy)
    assert drifted["central_quality_evidence_verified"] is False
    assert drifted["central_quality_binding_status"] == "ambiguous"

    conflict = copy.deepcopy(result)
    conflict["request_identity"]["producer_identity"]["setup_id"] = "other"
    rejected = copy.deepcopy(row)
    module._bind_central_quality_evidence(rejected, [conflict], policy)
    assert rejected["central_quality_evidence_verified"] is False

    flat_file_conflict = copy.deepcopy(result)
    flat_file_conflict["engine_build_receipt_file_sha256"] = "f" * 64
    rejected = copy.deepcopy(row)
    module._bind_central_quality_evidence(rejected, [flat_file_conflict], policy)
    assert rejected["central_quality_evidence_verified"] is False

    for wrong_backend in (
        "cpu", "native_full_tensorrt", "ort_tensorrt", "trt",
    ):
        wrong_logical_backend = copy.deepcopy(result)
        wrong_logical_backend["backend"] = wrong_backend
        wrong_logical_backend["request_identity"]["backend"] = wrong_backend
        rejected = copy.deepcopy(row)
        module._bind_central_quality_evidence(
            rejected, [wrong_logical_backend], policy,
        )
        assert rejected["central_quality_evidence_verified"] is False

    optional_backend_absent = copy.deepcopy(result)
    optional_backend_absent["request_identity"].pop("producer_backend")
    accepted = copy.deepcopy(row)
    module._bind_central_quality_evidence(
        accepted, [optional_backend_absent], policy,
    )
    assert accepted["central_quality_evidence_verified"] is True

    for container_name, invalid_backend in (
        ("result", "cpu"),
        ("request_identity", "cpu"),
        ("request_identity", None),
        ("request_identity", ""),
    ):
        wrong_signed_backend = copy.deepcopy(result)
        container = (
            wrong_signed_backend
            if container_name == "result"
            else wrong_signed_backend["request_identity"]
        )
        container["producer_backend"] = invalid_backend
        rejected = copy.deepcopy(row)
        module._bind_central_quality_evidence(
            rejected, [wrong_signed_backend], policy,
        )
        assert rejected["central_quality_evidence_verified"] is False

    missing = copy.deepcopy(row)
    missing["full_command_contract"].pop("quality_first_producer_identity")
    body = copy.deepcopy(missing["full_command_contract"])
    body.pop("contract_sha256", None)
    missing["full_command_contract"]["contract_sha256"] = hashlib.sha256(
        canonical_json(body).encode()
    ).hexdigest()
    missing["full_command_contract_sha256"] = missing["full_command_contract"]["contract_sha256"]
    module._bind_central_quality_evidence(missing, [result], policy)
    assert missing["central_quality_evidence_verified"] is False
    assert missing["central_quality_binding_status"] == "native_quality_first_binding_invalid"

    semantic_unbound = copy.deepcopy(row)
    semantic_unbound["quality_first_semantic_dump_binding_valid"] = False
    module._bind_central_quality_evidence(semantic_unbound, [result], policy)
    assert semantic_unbound["central_quality_evidence_verified"] is False

    receipt_domain_conflict = copy.deepcopy(row)
    receipt_domain_conflict["full_command_contract"][
        "engine_build_receipt_file_sha256"
    ] = "f" * 64
    body = copy.deepcopy(receipt_domain_conflict["full_command_contract"])
    body.pop("contract_sha256", None)
    receipt_domain_conflict["full_command_contract"]["contract_sha256"] = hashlib.sha256(
        canonical_json(body).encode()
    ).hexdigest()
    receipt_domain_conflict["full_command_contract_sha256"] = (
        receipt_domain_conflict["full_command_contract"]["contract_sha256"]
    )
    module._bind_central_quality_evidence(
        receipt_domain_conflict, [result], policy,
    )
    assert receipt_domain_conflict["central_quality_evidence_verified"] is False


def test_final_report_exact_join_and_repetition_drift_fail_closed() -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    final = _load_script("native_producer_final_report.py")
    row, result, _producer, policy = _quality_first_fixture()
    validator._bind_central_quality_evidence(row, [result], policy)
    row.update({
        "ok": True, "claim_ok": True, "status": "claim_ok",
        "gate_status": "eligible", "contract_consistent": True,
        "task_valid": True, "accuracy_gate_pass": True,
        "eligible_for_ranking": True, "accuracy_gate_policy_match": True,
        "accuracy_gate_policy_sha256": policy.sha256(),
    })
    performance = copy.deepcopy(row)
    attached, metadata = final._attach_quality_evidence([performance], {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 7, "status": "complete", "row_count": 1,
        "rows": [copy.deepcopy(row)],
    }, quality_summary_status="loaded")
    assert attached[0]["quality_evidence_verified"] is True
    assert attached[0]["quality_first_producer_identity_match"] is True
    assert metadata["verified_performance_row_count"] == 1

    base = {
        "backend": "native_full_tensorrt", "producer_impl": "trt",
        "model": "resnet50", "case": "full", "setup_id": "setup-a",
        "comparison_backend": "hailo8", "execution_mode": "native_full_baseline",
        "execution_precision": "fp16", "task": "classification",
        "ok": True, "fps_makespan": 100.0,
        "quality_first_producer_identity_sha256": "a" * 64,
        "repetition_count_requested": 1, "repetition_count_attempted": 1,
        "repetition_records": [],
    }
    drift_rows = [
        {**base, "full_command_contract_sha256": "b" * 64, "report": "/one.json"},
        {**base, "full_command_contract_sha256": "c" * 64, "report": "/two.json"},
    ]
    aggregated = final._aggregate_repetitions(drift_rows)
    assert len(aggregated) == 1
    assert aggregated[0]["ok"] is False
    assert aggregated[0]["repetition_claim_identity_drift"] is True
    assert aggregated[0]["failure_reason"].startswith(
        "native_repetition_claim_identity_drift"
    )


def test_quality_only_payload_never_becomes_native_performance_row(tmp_path: Path) -> None:
    final = _load_script("native_producer_final_report.py")
    analysis = tmp_path / "analysis_tables"
    analysis.mkdir()
    (analysis / "native_full_baseline_eval.json").write_text(json.dumps({
        "rows": [{
            "backend": "native_full_tensorrt", "model": "resnet50",
            "case": "full", "quality_evidence_only": True,
            "execution_role": "full_quality_only",
            "performance_claims_emitted": False, "ok": True,
            "fps_makespan": 999999.0,
        }],
    }), encoding="utf-8")
    assert final._rows_from_native_full(tmp_path) == []
