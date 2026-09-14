from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"v269d_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_row() -> dict[str, object]:
    return {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "setup_id": "orin_nx_hailo8_01",
        "task": "classification",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "sha256:" + "1" * 64,
        "runtime_precision_identity": "float32_layout_fp16",
    }


def _central_result(policy: AccuracyGatePolicy) -> dict[str, object]:
    request_sha = "2" * 64
    model_sha = "3" * 64
    dataset_sha = "4" * 64
    image_ids_sha = "5" * 64
    ground_truth_sha = "6" * 64
    quality_sha = "7" * 64
    preprocessing_sha = "8" * 64
    policy_sha = policy.sha256()
    identity = {
        "schema_version": 4,
        "identity_valid": True,
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "source_run_id": "hailo8_to_trt",
        "setup_id": "orin_nx_hailo8_01",
        "variant": "composed",
        "source_request_sha256": request_sha,
        "model_sha256": model_sha,
        "validation_dataset_sha256": dataset_sha,
        "validation_dataset_image_ids_sha256": image_ids_sha,
        "validation_dataset_ground_truth_sha256": ground_truth_sha,
        "policy_sha256": policy_sha,
        "endpoint_contract_hash": "1" * 64,
        "runtime_precision_identity": "float32_layout_fp16",
        "quality_contract_sha256": quality_sha,
        "preprocessing_contract_sha256": preprocessing_sha,
    }
    return {
        "model_id": "resnet50",
        "task": "classification",
        "case_id": "b052",
        "source_run_id": "hailo8_to_trt",
        "source_setup_id": "orin_nx_hailo8_01",
        "variant": "composed",
        "status": "completed",
        "technical_status": "completed",
        "decision": "pass",
        "policy_sha256": "sha256:" + policy_sha,
        # This is the production serialization emitted by workflow.sha256_file.
        "source_request_sha256": "sha256:" + request_sha,
        "model_sha256": "sha256:" + model_sha,
        "validation_dataset_sha256": "sha256:" + dataset_sha,
        "validation_dataset_image_ids_sha256": "sha256:" + image_ids_sha,
        "validation_dataset_ground_truth_sha256": "sha256:" + ground_truth_sha,
        "endpoint_contract_hash": "sha256:" + "1" * 64,
        "runtime_precision_identity": "float32_layout_fp16",
        "quality_contract_sha256": "sha256:" + quality_sha,
        "preprocessing_contract_sha256": "sha256:" + preprocessing_sha,
        "request_identity": identity,
        "primary": {
            "metric": "top1_accuracy",
            "delta": 0.0,
            "ci_low": 0.0,
            "margin": 0.01,
        },
    }


def test_production_prefixed_sha_binding_is_canonical_and_exact() -> None:
    module = _load_script("native_producer_validate_visualize.py")
    policy = AccuracyGatePolicy()
    row = _native_row()

    module._bind_central_quality_evidence(row, [_central_result(policy)], policy)

    assert row["central_quality_evidence_verified"] is True
    assert row["precision_quality_verified"] is True
    assert row["central_quality_binding_status"] == "exact_identity_match"
    assert row["endpoint_contract_hash"] == "1" * 64
    assert row["source_request_sha256"] == "2" * 64
    assert row["model_sha256"] == "3" * 64
    assert row["validation_dataset_sha256"] == "4" * 64
    assert row["validation_dataset_image_ids_sha256"] == "5" * 64
    assert row["validation_dataset_ground_truth_sha256"] == "6" * 64
    assert row["quality_contract_sha256"] == "7" * 64
    assert row["preprocessing_contract_sha256"] == "8" * 64
    assert row["task_quality_policy_sha256"] == policy.sha256()
    assert row["runtime_quality_gate_policy_sha256"] == policy.sha256()

    assert module._normalize_sha256("sha256:" + "a" * 64) == "a" * 64
    assert module._normalize_sha256("sha256:sha256:" + "a" * 64) == ""
    assert module._normalize_sha256("sha512:" + "a" * 64) == ""


def test_explicit_quality_first_split_row_cannot_fall_back_to_legacy_join() -> None:
    module = _load_script("native_producer_validate_visualize.py")
    policy = AccuracyGatePolicy()
    row = _native_row()
    row["native_split_quality_binding_required"] = True

    module._bind_central_quality_evidence(row, [_central_result(policy)], policy)

    assert row["central_quality_evidence_verified"] is False
    assert row["precision_quality_verified"] is False
    assert row["central_quality_binding_status"] == (
        "native_split_quality_binding_invalid_or_missing"
    )
    assert "native_split_quality_selected_binding_missing" in row[
        "quality_first_binding_errors"
    ]


def test_missing_malformed_or_conflicting_provenance_fails_closed() -> None:
    module = _load_script("native_producer_validate_visualize.py")
    policy = AccuracyGatePolicy()

    missing_model = _central_result(policy)
    missing_model.pop("model_sha256")
    missing_model["request_identity"].pop("model_sha256")
    row = _native_row()
    module._bind_central_quality_evidence(row, [missing_model], policy)
    assert row["central_quality_evidence_verified"] is False
    assert row["precision_quality_verified"] is False

    conflicting_dataset = _central_result(policy)
    conflicting_dataset["request_identity"]["validation_dataset_sha256"] = "f" * 64
    row = _native_row()
    module._bind_central_quality_evidence(row, [conflicting_dataset], policy)
    assert row["central_quality_evidence_verified"] is False
    assert row["central_quality_binding_status"] == "no_exact_identity_match"

    malformed_request = _central_result(policy)
    malformed_request["source_request_sha256"] = "sha256:sha256:" + "2" * 64
    row = _native_row()
    module._bind_central_quality_evidence(row, [malformed_request], policy)
    assert row["central_quality_evidence_verified"] is False

    conflicting_native = _native_row()
    conflicting_native["model_sha256"] = "e" * 64
    module._bind_central_quality_evidence(
        conflicting_native, [_central_result(policy)], policy,
    )
    assert conflicting_native["central_quality_evidence_verified"] is False


def test_energy_prefers_vendor_artifact_runtime_identity_and_carries_provenance() -> None:
    module = _load_script("native_producer_energy_plan.py")
    runtime_identity = "deepx_dxnn_sha256:" + "d" * 64
    validation = {
        "runtime_precision_identity": runtime_identity,
        "full_runtime_precision": "fp16",
        "source_request_sha256": "sha256:" + "1" * 64,
        "model_sha256": "sha256:" + "2" * 64,
        "validation_dataset_sha256": "sha256:" + "3" * 64,
        "validation_dataset_image_ids_sha256": "sha256:" + "4" * 64,
        "validation_dataset_ground_truth_sha256": "sha256:" + "5" * 64,
        "accuracy_gate_policy_sha256": "sha256:" + "6" * 64,
        "task_quality_policy_sha256": "sha256:" + "6" * 64,
        "runtime_quality_gate_policy_sha256": "6" * 64,
    }
    assert module._full_runtime_precision(
        {"execution_precision": "fp16", "precision": "legacy"}, validation,
    ) == runtime_identity

    provenance, conflicts = module._validated_quality_provenance(validation)
    assert conflicts == []
    assert provenance == {
        "source_request_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "validation_dataset_sha256": "3" * 64,
        "validation_dataset_image_ids_sha256": "4" * 64,
        "validation_dataset_ground_truth_sha256": "5" * 64,
        "accuracy_gate_policy_sha256": "6" * 64,
        "task_quality_policy_sha256": "6" * 64,
        "runtime_quality_gate_policy_sha256": "6" * 64,
    }

    conflicting = copy.deepcopy(validation)
    conflicting["runtime_quality_gate_policy_sha256"] = "7" * 64
    _provenance, conflicts = module._validated_quality_provenance(conflicting)
    assert conflicts == [
        "accuracy_gate_policy_sha256", "task_quality_policy_sha256",
        "runtime_quality_gate_policy_sha256",
    ]


def _final_report_rows() -> tuple[dict[str, object], dict[str, object]]:
    endpoint = "1" * 64
    source_model_sha = "3" * 64
    engine_sha = "c" * 64
    trtexec_sha = "d" * 64
    source_path = "/evidence/yolo26s.onnx"
    engine_path = "/evidence/yolo26s.engine"
    trtexec_path = "/evidence/trtexec"
    attestation = {
        "attested": True, "status": "passed",
        "stage": "decoded_nms", "endpoint": "decoded_nms",
        "endpoint_contract_hash": "sha256:" + endpoint,
    }
    shared = {
        "backend": "native_full_tensorrt", "model": "yolo26s",
        "case": "full", "setup_id": "h8", "comparison_backend": "hailo8",
        "task": "detection", "stage": "decoded_nms",
        "execution_mode": "native_full_baseline", "execution_precision": "fp16",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "sha256:" + endpoint,
        "output_endpoint_attestation": attestation,
    }
    receipt: dict[str, object] = {
        "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
        "schema_version": 1, "build_returncode": 0, "dry_run": False,
        "command": [
            trtexec_path, f"--onnx={source_path}",
            f"--saveEngine={engine_path}", "--fp16",
        ],
        "source_onnx": source_path,
        "source_onnx_sha256": source_model_sha,
        "engine": engine_path, "engine_sha256": engine_sha,
        "trtexec": trtexec_path, "trtexec_sha256": trtexec_sha,
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    command: dict[str, object] = {
        "schema": "onnx-splitpoint/native-full-command-contract",
        "schema_version": 1, "complete": True,
        "backend": shared["backend"], "model": shared["model"],
        "case": shared["case"], "setup_id": shared["setup_id"],
        "comparison_backend": shared["comparison_backend"],
        "python_executable": "/evidence/python3",
        "runner": "scripts/native_full_baseline_eval_runner.py",
        "root": "/evidence", "benchmark_set": "/evidence/benchmark_set",
        "backend_arg": "tensorrt",
        "runner_sha256": "d" * 64, "input_image_sha256": "e" * 64,
        "source_model_sha256": source_model_sha,
        "trt_engine_build_receipt": receipt,
        "trt_engine_build_receipt_status": "engine_build_receipt_verified",
        "model_binding": {
            "source_artifact": "source_onnx",
            "source_onnx_sha256": source_model_sha,
            "compiled_artifact": "engine",
            "compiled_artifact_sha256": engine_sha,
            "status": "verified_engine_build_receipt_bound",
        },
        "artifacts": {
            "command_python_executable": {
                "path": "/evidence/python3", "invocation_path": "/evidence/python3",
                "sha256": "0" * 64, "interpreter_identity": {},
            },
            "trtexec": {"path": trtexec_path, "sha256": trtexec_sha},
            "source_onnx": {"path": source_path, "sha256": source_model_sha},
            "engine": {
                "path": engine_path, "sha256": engine_sha,
                "compiled_from_source_onnx_sha256": source_model_sha,
            },
            "input_manifest": {"path": "/evidence/input.json", "sha256": "e" * 64},
            "runtime_input_tensor": {"path": "/evidence/input.bin", "sha256": "f" * 64},
            "engine_build_receipt": {
                "path": "/evidence/engine_build_receipt.json", "sha256": "9" * 64,
            },
        },
        "runtime_options": {
            "frames": 1000, "warmup": 100, "inflight": 4,
            "trt_precision": "fp16", "workspace_mb": 1024,
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
    }
    command["contract_sha256"] = hashlib.sha256(json.dumps(
        command, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    shared["full_command_contract_sha256"] = command["contract_sha256"]
    provenance = {
        "source_request_sha256": "sha256:" + "2" * 64,
        "model_sha256": "sha256:" + source_model_sha,
        "validation_dataset_sha256": "sha256:" + "4" * 64,
        "validation_dataset_image_ids_sha256": "sha256:" + "5" * 64,
        "validation_dataset_ground_truth_sha256": "sha256:" + "6" * 64,
        "accuracy_gate_policy_sha256": "sha256:" + "7" * 64,
        "task_quality_policy_sha256": "7" * 64,
        "runtime_quality_gate_policy_sha256": "sha256:" + "7" * 64,
        "quality_contract_sha256": "8" * 64,
        "preprocessing_contract_sha256": "9" * 64,
        "quality_record_endpoint_contract_sha256": "e" * 64,
        "decoder_contract_sha256": "a" * 64,
        "nms_contract_sha256": "b" * 64,
    }
    performance = {**shared, **provenance, "full_command_contract": command}
    quality = {
        **shared,
        **provenance,
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "ok": True, "claim_ok": True, "status": "claim_ok",
        "gate_status": "eligible", "contract_consistent": True,
        "task_valid": True, "accuracy_gate_pass": True,
        "eligible_for_ranking": True, "accuracy_gate_policy_match": True,
    }
    return performance, quality


def test_final_report_requires_complete_conflict_free_central_provenance() -> None:
    module = _load_script("native_producer_final_report.py")
    performance, quality = _final_report_rows()

    def attach(
        candidate: dict[str, object],
        performance_candidate: dict[str, object] | None = None,
    ) -> dict[str, object]:
        rows, _ = module._attach_quality_evidence([
            dict(performance_candidate if performance_candidate is not None else performance)
        ], {
            "schema": "onnx-splitpoint/native-producer-validation-summary",
            "schema_version": 6, "status": "complete", "row_count": 1,
            "rows": [candidate],
        })
        return rows[0]

    admitted = attach(dict(quality))
    assert admitted["quality_evidence_verified"] is True
    assert admitted["precision_quality_binding_verified"] is True
    assert admitted["task_quality_observation_valid"] is True
    assert admitted["quality_claim_result_verified"] is True
    assert admitted["source_request_sha256"] == "2" * 64
    assert admitted["validation_dataset_image_ids_sha256"] == "5" * 64
    assert admitted["endpoint_contract_hash"] == "1" * 64

    performance_without_runtime_policy = dict(performance)
    performance_without_runtime_policy.pop(
        "runtime_quality_gate_policy_sha256"
    )
    quality_without_runtime_policy = dict(quality)
    quality_without_runtime_policy.pop(
        "runtime_quality_gate_policy_sha256"
    )
    diagnostic_missing = attach(
        quality_without_runtime_policy,
        performance_without_runtime_policy,
    )
    assert diagnostic_missing["quality_evidence_verified"] is True
    assert diagnostic_missing["precision_quality_binding_verified"] is True

    diagnostic_conflict = attach(
        dict(quality, runtime_quality_gate_policy_sha256="8" * 64),
        dict(performance, runtime_quality_gate_policy_sha256="9" * 64),
    )
    assert diagnostic_conflict["quality_evidence_verified"] is True
    assert diagnostic_conflict["precision_quality_binding_verified"] is True
    for task in ("classification", "detection"):
        assert "runtime_quality_gate_policy_sha256" not in (
            module._required_quality_bindings(task)
        )

    negative_observation = dict(
        quality,
        ok=False,
        claim_ok=False,
        status="accuracy_gate_failed",
        gate_status="excluded",
        task_valid=False,
        accuracy_gate_pass=False,
        eligible_for_ranking=False,
    )
    observed = attach(negative_observation)
    # Exact evidence binding and the measured negative result are independent
    # axes.  A failed accuracy gate must not erase that the observation was
    # bound to the exact precision/runtime contract.
    assert observed["precision_quality_binding_verified"] is True
    assert observed["task_quality_observation_valid"] is True
    assert observed["quality_claim_result_verified"] is False
    assert observed["quality_accuracy_gate_pass"] is False

    raw_performance = {
        key: value for key, value in performance.items()
        if key not in {
            alias
            for canonical, aliases in module._QUALITY_BINDING_ALIASES.items()
            if canonical != "command_contract_sha256"
            for alias in aliases
        }
    }
    imported = attach(dict(quality), raw_performance)
    assert imported["quality_evidence_verified"] is False
    assert (
        "command_contract_sha256:tensorrt_performance_model_sha256_missing"
        in imported["quality_binding_errors"]["performance"]
    )

    no_command_performance = dict(performance)
    no_command_performance.pop("full_command_contract_sha256")
    rejected = attach(dict(quality), no_command_performance)
    assert rejected["quality_evidence_verified"] is False
    assert rejected["quality_match_status"] == "exact_identity_provenance_incomplete"
    assert "command_contract_sha256" in rejected["quality_binding_missing"]["performance"]

    wrong_command_performance = dict(performance)
    wrong_command_performance["full_command_contract_sha256"] = "d" * 64
    rejected = attach(dict(quality), wrong_command_performance)
    assert rejected["quality_evidence_verified"] is False
    assert rejected["quality_match_status"] == "exact_identity_provenance_malformed_or_conflicting"
    assert "command_contract_sha256:nested_top_level_mismatch" in (
        rejected["quality_binding_errors"]["performance"]
    )

    missing = dict(quality)
    missing.pop("validation_dataset_ground_truth_sha256")
    rejected = attach(missing)
    assert rejected["quality_evidence_verified"] is False
    assert rejected["quality_match_status"] == "exact_identity_provenance_incomplete"

    missing_task_policy = dict(quality)
    missing_task_policy.pop("task_quality_policy_sha256")
    rejected = attach(missing_task_policy)
    assert rejected["quality_evidence_verified"] is False
    assert "task_quality_policy_sha256" in rejected[
        "quality_binding_missing"
    ]["quality"]

    for task in ("classification", "detection"):
        assert "quality_record_endpoint_contract_sha256" in (
            module._required_quality_bindings(task)
        )

    missing_endpoint_quality = dict(quality)
    missing_endpoint_quality.pop(
        "quality_record_endpoint_contract_sha256"
    )
    rejected = attach(missing_endpoint_quality)
    assert rejected["quality_evidence_verified"] is False
    assert rejected["quality_match_status"] == (
        "exact_identity_provenance_incomplete"
    )
    assert "quality_record_endpoint_contract_sha256" in rejected[
        "quality_binding_missing"
    ]["quality"]

    conflicting = dict(quality, source_onnx_sha256="c" * 64)
    rejected = attach(conflicting)
    assert rejected["quality_evidence_verified"] is False
    assert "model_sha256:conflict" in rejected["quality_binding_errors"]["quality"]

    malformed = dict(quality)
    malformed["source_request_sha256"] = "sha256:sha256:" + "2" * 64
    rejected = attach(malformed)
    assert rejected["quality_evidence_verified"] is False
    assert "source_request_sha256:malformed" in rejected["quality_binding_errors"]["quality"]

    unverified = dict(quality, central_quality_evidence_verified=False)
    assert attach(unverified)["quality_evidence_verified"] is False


def test_final_report_rejects_self_asserted_nested_contract_and_endpoint_conflicts() -> None:
    module = _load_script("native_producer_final_report.py")

    def attach(performance: dict[str, object], quality: dict[str, object]) -> dict[str, object]:
        attached, _ = module._attach_quality_evidence([performance], {
            "schema": "onnx-splitpoint/native-producer-validation-summary",
            "schema_version": 6, "status": "complete", "row_count": 1,
            "rows": [quality],
        })
        return attached[0]

    for mutation in ("incomplete", "arbitrary"):
        performance, quality = _final_report_rows()
        nested = copy.deepcopy(performance["full_command_contract"])
        nested.pop("contract_sha256")
        if mutation == "incomplete":
            nested["complete"] = False
        else:
            nested = {
                "schema": "onnx-splitpoint/native-full-command-contract",
                "schema_version": 1, "complete": True,
            }
        nested["contract_sha256"] = hashlib.sha256(json.dumps(
            nested, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        performance["full_command_contract"] = nested
        performance["full_command_contract_sha256"] = nested["contract_sha256"]
        quality["full_command_contract_sha256"] = nested["contract_sha256"]
        rejected = attach(performance, quality)
        assert rejected["quality_evidence_verified"] is False
        assert any(
            "full_command_contract" in reason
            for reason in rejected["quality_binding_errors"]["performance"]
        )

    performance, quality = _final_report_rows()
    performance["output_endpoint_attestation"] = {
        **performance["output_endpoint_attestation"],
        "stage": "raw_head", "endpoint": "raw_head",
    }
    assert attach(performance, quality)["quality_evidence_verified"] is False
    assert module._explicit_output_endpoint(performance) == ""

    performance, quality = _final_report_rows()
    performance["output_endpoint_id"] = (
        "detection:raw_head:" + "f" * 64
    )
    assert attach(performance, quality)["quality_evidence_verified"] is False
    assert module._explicit_output_endpoint(performance) == ""

    performance, quality = _final_report_rows()
    performance["execution_precision"] = "banana"
    quality["execution_precision"] = "banana"
    rejected = attach(performance, quality)
    assert rejected["quality_evidence_verified"] is False
    assert rejected["quality_identity_complete"] is False


def test_final_report_rejects_resealed_unbacked_or_tampered_trt_binding() -> None:
    module = _load_script("native_producer_final_report.py")

    def reseal(contract: dict[str, object]) -> dict[str, object]:
        contract = copy.deepcopy(contract)
        contract.pop("contract_sha256", None)
        contract["contract_sha256"] = hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        return contract

    def attach(performance: dict[str, object], quality: dict[str, object]) -> dict[str, object]:
        attached, _ = module._attach_quality_evidence([performance], {
            "schema": "onnx-splitpoint/native-producer-validation-summary",
            "schema_version": 6, "status": "complete", "row_count": 1,
            "rows": [quality],
        })
        return attached[0]

    for mutation in ("missing_receipt", "tampered_receipt", "false_model_binding"):
        performance, quality = _final_report_rows()
        contract = copy.deepcopy(performance["full_command_contract"])
        if mutation == "missing_receipt":
            contract.pop("trt_engine_build_receipt")
        elif mutation == "tampered_receipt":
            receipt = copy.deepcopy(contract["trt_engine_build_receipt"])
            receipt["source_onnx_sha256"] = "f" * 64
            # Re-sealing the outer contract must not make a stale inner build
            # receipt authoritative.
            contract["trt_engine_build_receipt"] = receipt
        else:
            contract["artifacts"]["engine"][
                "compiled_from_source_onnx_sha256"
            ] = "f" * 64
            # This is a fully populated, canonically re-sealed command body;
            # only the backend-specific ONNX->engine check can reject it.
        contract = reseal(contract)
        performance["full_command_contract"] = contract
        performance["full_command_contract_sha256"] = contract["contract_sha256"]
        quality["full_command_contract_sha256"] = contract["contract_sha256"]

        rejected = attach(performance, quality)
        assert rejected["quality_evidence_verified"] is False, mutation
        assert any(
            "strict_validation_failed" in reason
            for reason in rejected["quality_binding_errors"]["performance"]
        ), mutation


def test_final_report_rejects_resealed_unbacked_split_contract() -> None:
    module = _load_script("native_producer_final_report.py")
    contract: dict[str, object] = {
        "schema": "onnx-splitpoint/native-command-contract",
        "schema_version": 1, "complete": True,
        "backend": "hailo8_to_trt", "model": "resnet50", "case": "b052",
        "precision": "uint8_dequant_fp16", "setup_id": "h8",
        "comparison_backend": "hailo8", "runner_sha256": "1" * 64,
        "input_image_sha256": "2" * 64,
        # Plausible-looking but incomplete artifact evidence.  Re-sealing this
        # body used to be sufficient for the final quality join.
        "artifacts": {"engine": {"path": "/fake.engine", "sha256": "3" * 64}},
        "runtime_options": {"warmup": 10, "queue_depth": 2},
        "boundary_contract": {"boundary_layout_effective": "NCHW"},
        "prepared_input_contract": {},
    }
    contract["contract_sha256"] = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    row = {
        "backend": "hailo8_to_trt", "model": "resnet50", "case": "b052",
        "precision": "uint8_dequant_fp16", "setup_id": "h8",
        "comparison_backend": "hailo8", "native_command_contract": contract,
        "native_command_contract_sha256": contract["contract_sha256"],
    }

    digest, errors = module._verified_performance_command_contract(row)

    assert digest == ""
    assert any("strict_validation_failed" in reason for reason in errors)
    assert any("required_artifact_missing" in reason for reason in errors)


def test_runtime_precision_identity_uses_closed_backend_aware_grammar() -> None:
    module = _load_script("native_producer_final_report.py")
    assert module._runtime_precision_identity({
        "backend": "native_full_tensorrt", "case": "full",
        "execution_mode": "native_full_baseline", "execution_precision": "banana",
    }) == ""
    assert module._runtime_precision_identity({
        "backend": "native_full_deepx", "case": "full",
        "runtime_precision_identity": "deepx_dxnn_sha256:" + "d" * 64,
    }) == "deepx_dxnn_sha256:" + "d" * 64
    assert module._runtime_precision_identity({
        "backend": "native_full_hailo8", "case": "full",
        "runtime_precision_identity": "hailo_hef_sha256:" + "e" * 64,
    }) == "hailo_hef_sha256:" + "e" * 64
    assert module._runtime_precision_identity({
        "backend": "native_full_tensorrt", "case": "full",
        "runtime_precision_identity": "deepx_dxnn_sha256:" + "d" * 64,
    }) == ""
