from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import sys
from pathlib import Path

from onnx_splitpoint_tool.workflow.cross_runner_reporting import compute_cross_runner_report


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT_HASH = "a" * 64


def _command_contract() -> dict:
    source_model_sha = "2" * 64
    source_path = "/evidence/m.onnx"
    engine_path = "/evidence/m.engine"
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
        "backend": "native_full_tensorrt", "model": "m", "case": "full",
        "setup_id": "setup-a", "comparison_backend": "hailo8",
        "python_executable": "/evidence/python3",
        "runner": "scripts/native_full_baseline_eval_runner.py",
        "root": "/evidence",
        "benchmark_set": "/evidence/benchmark_set",
        "backend_arg": "tensorrt",
        "runner_sha256": "b" * 64, "input_image_sha256": "c" * 64,
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
                "path": "/evidence/input.bin", "sha256": "c" * 64,
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
    return contract


COMMAND_HASH = str(_command_contract()["contract_sha256"])


def _load_final_report():
    path = ROOT / "scripts" / "native_producer_final_report.py"
    spec = importlib.util.spec_from_file_location("v269_claim_final_report", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _endpoint() -> dict:
    return {
        "task": "classification",
        "stage": "classification_logits",
        "contract_family": "classification_logits",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": ENDPOINT_HASH,
        "output_endpoint_attestation": {
            "attested": True,
            "status": "passed",
            "stage": "classification_logits",
            "endpoint": "classification_logits",
            "endpoint_contract_hash": ENDPOINT_HASH,
        },
    }


def _repeat_row(**updates) -> dict:
    records = [
        {
            "repetition_index": index,
            "runtime_instance_id": f"runtime-{index}",
            "ok": True,
            "status": "ok",
            "fps_makespan": fps,
            "completed_work_units": 100,
            "workload_contract_sha256": COMMAND_HASH,
        }
        for index, fps in enumerate((10.0, 20.0, 30.0), start=1)
    ]
    row = {
        "backend": "native_full_tensorrt",
        "producer_impl": "trt",
        "model": "m",
        "case": "full",
        "execution_mode": "native_full_baseline",
        "setup_id": "setup-a",
        "comparison_backend": "hailo8",
        "execution_precision": "fp16",
        "ok": True,
        "full_command_contract": _command_contract(),
        "full_command_contract_sha256": COMMAND_HASH,
        **_endpoint(),
        "repetition_count_requested": 3,
        "repetition_count_attempted": 3,
        "repetition_count_valid": 3,
        "repetition_status": "complete",
        "repetition_aggregation": "median_with_deterministic_percentile_bootstrap_ci95",
        "repetition_runtime_scope": "fresh_runtime_per_repetition",
        "repetition_independence_verified": True,
        "repetition_records": records,
        "fps_repetition_samples": [10.0, 20.0, 30.0],
        "fps_median": 20.0,
        "fps_ci95_low": 10.0,
        "fps_ci95_high": 30.0,
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
    }
    row.update(updates)
    return row


def test_repeat_claim_recomputes_raw_median_and_ci() -> None:
    module = _load_final_report()
    row = _repeat_row()
    passed, reasons = module._validate_repeat_claim_evidence(row)
    assert passed is True
    assert reasons == []
    assert row["fps_median"] == 20.0
    assert row["fps_ci95_low"] is not None
    assert row["fps_ci95_high"] is not None
    assert row["repetition_statistics_source"] == "recomputed_from_unique_raw_repeat_records"


def test_repeat_claim_rejects_best_of_duplicate_nan_missing_ci_and_shared_runtime() -> None:
    module = _load_final_report()

    best = _repeat_row(repetition_aggregation="best_of_three")
    assert module._validate_repeat_claim_evidence(best)[0] is False

    duplicate = _repeat_row()
    duplicate["repetition_records"][1]["repetition_index"] = 1
    assert "repetition_id_duplicate" in module._validate_repeat_claim_evidence(duplicate)[1]

    nonfinite = _repeat_row()
    nonfinite["repetition_records"][1]["fps_makespan"] = math.nan
    assert "repetition_record_fps_nonfinite_or_nonpositive" in module._validate_repeat_claim_evidence(nonfinite)[1]

    missing_ci = _repeat_row(fps_ci95_low=None)
    assert "repetition_reported_median_or_ci_missing" in module._validate_repeat_claim_evidence(missing_ci)[1]

    serialized_false = _repeat_row(repetition_independence_verified="false")
    assert "repetition_independence_not_verified" in module._validate_repeat_claim_evidence(serialized_false)[1]

    shared = _repeat_row(repetition_runtime_scope="shared_initialized_runtime")
    assert "repetition_runtime_scope_not_independent" in module._validate_repeat_claim_evidence(shared)[1]

    unequal_work = _repeat_row()
    unequal_work["repetition_records"][1]["completed_work_units"] = 99
    assert "repetition_completed_work_units_mismatch" in module._validate_repeat_claim_evidence(unequal_work)[1]

    unequal_contract = _repeat_row()
    unequal_contract["repetition_records"][1]["workload_contract_sha256"] = "e" * 64
    assert "repetition_workload_contract_mismatch" in module._validate_repeat_claim_evidence(unequal_contract)[1]


def _quality_payload(row: dict, module, **row_updates) -> dict:
    quality_row = {key: row.get(key, "") for key in module._QUALITY_IDENTITY_FIELDS}
    quality_row.update({
        "full_command_contract_sha256": COMMAND_HASH,
        "input_image_sha256": "c" * 64,
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
    quality_row.update(row_updates)
    return {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 5,
        "status": "complete",
        "row_count": 1,
        "rows": [quality_row],
    }


def test_quality_join_binds_runtime_precision_schema_endpoint_and_hashes() -> None:
    module = _load_final_report()
    row = {
        **_repeat_row(),
        "input_image_sha256": "c" * 64,
    }
    attached, _ = module._attach_quality_evidence(
        [dict(row)], _quality_payload(row, module), quality_summary_status="loaded",
    )
    assert attached[0]["quality_evidence_verified"] is True

    missing_endpoint = _quality_payload(row, module)
    missing_endpoint["rows"][0].pop(
        "quality_record_endpoint_contract_sha256"
    )
    attached, _ = module._attach_quality_evidence(
        [dict(row)], missing_endpoint, quality_summary_status="loaded",
    )
    assert attached[0]["quality_match_status"] == (
        "exact_identity_provenance_incomplete"
    )
    assert attached[0]["quality_evidence_verified"] is False

    drifted_endpoint = _quality_payload(
        row, module,
        quality_record_endpoint_contract_sha256="e" * 64,
    )
    attached, _ = module._attach_quality_evidence(
        [dict(row)], drifted_endpoint, quality_summary_status="loaded",
    )
    assert attached[0]["quality_match_status"] == (
        "exact_identity_provenance_mismatch"
    )
    assert attached[0]["quality_evidence_verified"] is False

    stale_schema = _quality_payload(row, module)
    stale_schema["schema_version"] = 4
    attached, metadata = module._attach_quality_evidence(
        [dict(row)], stale_schema, quality_summary_status="loaded",
    )
    assert metadata["schema_valid"] is False
    assert attached[0]["quality_evidence_verified"] is False

    wrong_precision = _quality_payload(row, module, execution_precision="int8")
    attached, _ = module._attach_quality_evidence([dict(row)], wrong_precision, quality_summary_status="loaded")
    assert attached[0]["quality_match_status"] == "no_exact_identity_match"

    wrong_hash = _quality_payload(row, module, full_command_contract_sha256="d" * 64)
    attached, _ = module._attach_quality_evidence([dict(row)], wrong_hash, quality_summary_status="loaded")
    assert attached[0]["quality_match_status"] == "exact_identity_provenance_mismatch"
    assert attached[0]["quality_evidence_verified"] is False


def _write_cross_inputs(run_dir: Path, *, native_updates: dict | None = None) -> dict:
    reports = run_dir / "reports"
    (reports / "native_validation").mkdir(parents=True, exist_ok=True)
    common = {
        "model": "m",
        "case": "b001",
        "precision": "fp16",
        "setup_id": "setup-a",
        "comparison_backend": "deepx",
        **_endpoint(),
    }
    native = {
        **common,
        "backend": "deepx_to_trt",
        "ok": True,
        "fps_makespan": 100.0,
        "performance_claim_eligible": True,
        "output_endpoint_match": True,
        "precision_quality_verified": True,
        "quality_evidence_verified": True,
        "repeat_claim_gate_pass": True,
        "comparison_stratum_explicit": True,
    }
    native.update(native_updates or {})
    validation = {
        **common,
        "backend": "deepx_to_trt",
        "contract_consistent": True,
        "semantic_ok": True,
        "claim_ok": True,
        "task_valid": True,
        "accuracy_gate_pass": True,
        "eligible_for_ranking": True,
        "status": "claim_ok",
        "gate_status": "eligible",
    }
    (reports / "native_producer_combined_summary.json").write_text(
        json.dumps({"rows": [native]}), encoding="utf-8",
    )
    (reports / "native_validation" / "native_producer_validation_summary.json").write_text(
        json.dumps({"rows": [validation]}), encoding="utf-8",
    )
    return {
        **common,
        "model_id": "m",
        "case_id": "b001",
        "direction": "deepx_to_trt",
        "backend": "deepx_m1_to_tensorrt",
        "runner_regime": "generic",
        "variant": "split",
        "cycle_ms": 12.0,
        "contract_consistent": True,
        "task_quality_status": "pass",
        "eligible_for_ranking": True,
    }


def test_cross_runner_requires_final_claim_gates_and_never_selects_fastest_duplicate(tmp_path: Path) -> None:
    generic = _write_cross_inputs(tmp_path, native_updates={"performance_claim_eligible": False})
    report = compute_cross_runner_report(tmp_path, [generic], minimum_candidates=1)
    assert report["pair_count"] == 1
    assert report["eligible_pair_count"] == 0
    assert "native_performance_claim_eligible" in report["pairs"][0]["transfer_exclusion_reasons"]

    generic = _write_cross_inputs(tmp_path)
    faster_duplicate = {**generic, "cycle_ms": 1.0}
    report = compute_cross_runner_report(tmp_path, [generic, faster_duplicate], minimum_candidates=1)
    assert report["pair_count"] == 0
    assert any(
        item["reason"] == "ambiguous_or_missing_exact_identity_join"
        and item["generic_match_count"] == 2
        for item in report["identity_exclusions"]
    )

    wrong_setup = {**generic, "setup_id": "setup-b"}
    report = compute_cross_runner_report(tmp_path, [wrong_setup], minimum_candidates=1)
    assert report["eligible_pair_count"] == 0
    assert any(item["reason"] == "missing_exact_counterpart" for item in report["identity_exclusions"])
