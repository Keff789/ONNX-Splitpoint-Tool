from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_HASH = "a" * 64


def _attestation():
    return {
        "attested": True,
        "status": "passed",
        "stage": "decoded_nms",
        "endpoint": "decoded_nms",
        "endpoint_contract_hash": ENDPOINT_HASH,
    }


def _summary(rows):
    return {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 5,
        "status": "complete",
        "row_count": len(rows),
        "rows": rows,
    }


def _load():
    path = ROOT / "scripts" / "native_producer_final_report.py"
    spec = importlib.util.spec_from_file_location("v268_quality_final_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _performance_row(**updates):
    row = {
        "backend": "native_full_tensorrt",
        "model": "yolo",
        "case": "full",
        "precision": "fp16",
        "setup_id": "jetson-1",
        "comparison_backend": "hailo8",
        "task": "detection",
        "stage": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": ENDPOINT_HASH,
        "output_endpoint_attestation": _attestation(),
        "contract_family": "decoded_nms",
        "output_format": "bn6_detections",
        "execution_mode": "native_full_baseline",
        "execution_precision": "fp16",
        "ok": True,
        "claim_ok": True,
        "semantic_ok": True,
        "contract_consistent": True,
        "repetition_count_valid": 3,
        "repetition_status": "complete",
        "source_request_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "validation_dataset_sha256": "3" * 64,
        "validation_dataset_image_ids_sha256": "4" * 64,
        "validation_dataset_ground_truth_sha256": "5" * 64,
        "accuracy_gate_policy_sha256": "6" * 64,
        "task_quality_policy_sha256": "6" * 64,
        "runtime_quality_gate_policy_sha256": "6" * 64,
        "quality_contract_sha256": "7" * 64,
        "preprocessing_contract_sha256": "8" * 64,
        "quality_record_endpoint_contract_sha256": "d" * 64,
        "decoder_contract_sha256": "9" * 64,
        "nms_contract_sha256": "b" * 64,
    }
    row.update(updates)
    if "full_command_contract" not in row:
        source_model_sha = str(row["model_sha256"])
        source_path = "/evidence/yolo.onnx"
        engine_path = "/evidence/yolo.engine"
        trtexec_path = "/evidence/trtexec"
        engine_sha = "e" * 64
        trtexec_sha = "f" * 64
        receipt = {
            "schema": "onnx-splitpoint/tensorrt-engine-build-receipt",
            "schema_version": 1,
            "build_returncode": 0,
            "dry_run": False,
            "command": [
                trtexec_path,
                f"--onnx={source_path}",
                f"--saveEngine={engine_path}",
                "--fp16",
            ],
            "source_onnx": source_path,
            "source_onnx_sha256": source_model_sha,
            "engine": engine_path,
            "engine_sha256": engine_sha,
            "trtexec": trtexec_path,
            "trtexec_sha256": trtexec_sha,
        }
        receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        contract = {
            "schema": "onnx-splitpoint/native-full-command-contract",
            "schema_version": 1, "complete": True,
            "backend": row["backend"], "model": row["model"],
            "case": row["case"], "setup_id": row["setup_id"],
            "comparison_backend": row["comparison_backend"],
            "python_executable": "/evidence/python3",
            "runner": "scripts/native_full_baseline_eval_runner.py",
            "root": "/evidence",
            "benchmark_set": "/evidence/benchmark_set",
            "backend_arg": "tensorrt",
            "runner_sha256": "c" * 64, "input_image_sha256": "d" * 64,
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
                    "path": "/evidence/python3",
                    "invocation_path": "/evidence/python3",
                    "sha256": "0" * 64,
                    "interpreter_identity": {},
                },
                "trtexec": {"path": trtexec_path, "sha256": trtexec_sha},
                "source_onnx": {"path": source_path, "sha256": source_model_sha},
                "engine": {
                    "path": engine_path,
                    "sha256": engine_sha,
                    "compiled_from_source_onnx_sha256": source_model_sha,
                },
                "input_manifest": {
                    "path": "/evidence/input.json", "sha256": "1" * 64,
                },
                "runtime_input_tensor": {
                    "path": "/evidence/input.bin", "sha256": "d" * 64,
                },
                "engine_build_receipt": {
                    "path": "/evidence/engine_build_receipt.json",
                    "sha256": "9" * 64,
                },
            },
            "runtime_options": {
                "frames": 1000,
                "warmup": 100,
                "inflight": 4,
                "trt_precision": "fp16",
                "workspace_mb": 1024,
                "engine_build_python": "/evidence/python3",
                "no_shapes": False,
                "dump_outputs": True,
                "diagnostic_deepx_input_probes": False,
                "image_map": {},
            },
            "energy_workload": {
                "available": True,
                "kind": "tensorrt_full_hotloop",
                "trtexec_artifact": "trtexec",
                "engine_artifact": "engine",
                "source_model_artifact": "source_onnx",
                "engine_build_receipt_artifact": "engine_build_receipt",
                "engine_build_receipt_status": "engine_build_receipt_verified",
                "input_manifest_artifact": "input_manifest",
                "runtime_input_artifact": "runtime_input_tensor",
                "runtime_input_name": "images",
                "input_mode": "exact_semantic_dump_runtime_tensor",
            },
        }
        contract["contract_sha256"] = hashlib.sha256(json.dumps(
            contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        row["full_command_contract"] = contract
        row["full_command_contract_sha256"] = contract["contract_sha256"]
    return row


def _quality_row(mod, row, **updates):
    quality = {key: row.get(key, "") for key in mod._QUALITY_IDENTITY_FIELDS}
    quality.update({
        "full_command_contract_sha256": row["full_command_contract_sha256"],
        "source_request_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "validation_dataset_sha256": "3" * 64,
        "validation_dataset_image_ids_sha256": "4" * 64,
        "validation_dataset_ground_truth_sha256": "5" * 64,
        "accuracy_gate_policy_sha256": "6" * 64,
        "task_quality_policy_sha256": "6" * 64,
        "runtime_quality_gate_policy_sha256": "6" * 64,
        "quality_contract_sha256": "7" * 64,
        "preprocessing_contract_sha256": "8" * 64,
        "quality_record_endpoint_contract_sha256": "d" * 64,
        "decoder_contract_sha256": "9" * 64,
        "nms_contract_sha256": "b" * 64,
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "ok": True,
        "claim_ok": True,
        "status": "claim_ok",
        "gate_status": "eligible",
        "contract_consistent": True,
        "task_valid": True,
        "accuracy_gate_pass": True,
        "eligible_for_ranking": True,
        "accuracy_gate_policy_match": True,
    })
    quality.update(updates)
    return quality


def test_quality_gate_does_not_trust_runtime_claim_flags_without_summary() -> None:
    mod = _load()
    row = _performance_row()
    rows, metadata = mod._attach_quality_evidence([row], None)
    gated = mod._apply_comparison_claim_gates(rows)
    assert metadata["status"] == "not_provided"
    assert gated[0]["precision_quality_verified"] is False
    assert "precision_variant_quality_not_verified" in gated[0]["performance_claim_exclusion_reasons"]


def test_quality_join_requires_one_exact_identity_and_strict_boolean_gates() -> None:
    mod = _load()
    row = _performance_row()
    good = _quality_row(mod, row)
    rows, metadata = mod._attach_quality_evidence([row], _summary([good]), quality_summary_status="loaded")
    assert rows[0]["quality_match_status"] == "exact_identity_match"
    assert rows[0]["quality_evidence_verified"] is True
    assert metadata["verified_performance_row_count"] == 1

    missing_endpoint = _quality_row(mod, row)
    missing_endpoint.pop("quality_record_endpoint_contract_sha256")
    rows, _ = mod._attach_quality_evidence(
        [_performance_row()], _summary([missing_endpoint]),
    )
    assert rows[0]["quality_match_status"] == (
        "exact_identity_provenance_incomplete"
    )
    assert rows[0]["quality_evidence_verified"] is False

    drifted_endpoint = _quality_row(
        mod, row, quality_record_endpoint_contract_sha256="e" * 64,
    )
    rows, _ = mod._attach_quality_evidence(
        [_performance_row()], _summary([drifted_endpoint]),
    )
    assert rows[0]["quality_match_status"] == (
        "exact_identity_provenance_mismatch"
    )
    assert rows[0]["quality_evidence_verified"] is False

    mismatched = _quality_row(mod, row, setup_id="jetson-2")
    rows, _ = mod._attach_quality_evidence([_performance_row()], _summary([mismatched]))
    assert rows[0]["quality_match_status"] == "no_exact_identity_match"
    assert rows[0]["quality_evidence_verified"] is False

    string_true = _quality_row(mod, row, task_valid="True")
    rows, _ = mod._attach_quality_evidence([_performance_row()], _summary([string_true]))
    assert rows[0]["precision_quality_binding_verified"] is True
    assert rows[0]["task_quality_observation_valid"] is False
    assert rows[0]["quality_claim_result_verified"] is False
    assert rows[0]["quality_evidence_verified"] is False

    rows, metadata = mod._attach_quality_evidence([_performance_row()], _summary([good, dict(good)]))
    assert rows[0]["quality_match_status"] == "ambiguous_exact_identity_match"
    assert rows[0]["quality_evidence_verified"] is False
    assert metadata["duplicate_identity_count"] == 1


def test_strict_bool_rejects_serialized_and_numeric_aliases() -> None:
    mod = _load()
    assert mod._strict_bool(True) is True
    assert mod._strict_bool(False) is False
    for malformed in ("True", "true", "1", "False", "0", 1, 0, 1.0, 0.0):
        assert mod._strict_bool(malformed) is None


def test_split_output_endpoint_is_loaded_only_from_explicit_manifest(tmp_path: Path) -> None:
    mod = _load()
    result = tmp_path / "native_fifo_results.json"
    result.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "native_fifo_outputs" / "native_fifo_outputs_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({
        "task": "detection",
        "stage": "decoded_nms",
        "output_format": "bn6_detections",
        "contract_family": "decoded_nms",
        "contract_source": "frozen_host_tail",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": ENDPOINT_HASH,
        "output_endpoint_attestation": _attestation(),
    }), encoding="utf-8")
    fields = mod._split_output_contract(
        result,
        {"output_manifest": "/remote/path/does/not/exist.json"},
        fallback_relpaths=("native_fifo_outputs/native_fifo_outputs_manifest.json",),
    )
    assert fields["output_contract_manifest_status"] == "explicit_complete"
    assert fields["task"] == "detection"
    assert fields["output_format"] == "bn6_detections"
    assert fields["contract_family"] == "decoded_nms"

    manifest.write_text(json.dumps({"task": "detection"}), encoding="utf-8")
    incomplete = mod._split_output_contract(
        result,
        fallback_relpaths=("native_fifo_outputs/native_fifo_outputs_manifest.json",),
    )
    assert incomplete["output_contract_manifest_status"] == "explicit_incomplete"
    assert mod._explicit_output_endpoint(incomplete) == ""


def test_unknown_tensor_endpoint_is_not_explicit() -> None:
    mod = _load()
    assert mod._explicit_output_endpoint({
        "task": "detection", "contract_family": "unknown", "output_format": "tensor_outputs",
    }) == ""


def test_full_latency_repetition_samples_are_derived_from_raw_records() -> None:
    mod = _load()
    fields = mod._repeat_fields({
        "repetition_records": [
            {"ok": True, "fps_makespan": 10.0, "latency_mean_ms": 9.0},
            {"ok": True, "fps_makespan": 11.0, "latency_median_ms": 8.0},
            {"ok": False, "fps_makespan": 99.0, "latency_mean_ms": 1.0},
        ],
    })
    assert fields["latency_mean_repetition_samples_ms"] == [9.0, 8.0]
