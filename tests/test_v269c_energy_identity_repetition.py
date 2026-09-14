from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from onnx_splitpoint_tool.native_energy_reporting import (
    build_native_energy_pairs,
    collect_native_energy,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"v269c_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _energy_payload(energy: float) -> str:
    return json.dumps({
        "energy_total_j": energy * 100.0,
        "energy_per_work_unit_j": energy,
        "active_duration_s": 60.0,
        "avg_power_w": energy * 100.0 / 60.0,
        "energy_work_units_used": 100,
        "energy_work_units_source": "runtime_completed_work_units",
        "runtime_completed_work_unit_run_count": 1,
        "valid_postprocessed_runs": 1,
        "run_count": 1,
        "postprocess_status": "ok",
        "final_energy_gate_status": "pass",
        "energy_efficiency_claim_eligible": True,
        "energy_physical_scope": "MB",
        "energy_window_effective": "command_window",
        "energy_primary_metric": "calibrated_input_energy_unsubtracted",
        "energy_calibrated_input_unsubtracted": True,
        "energy_raw_primary": True,
    })


def _plan(backend: str, *, energy: float) -> dict[str, object]:
    full = backend.startswith("native_full_")
    return {
        "backend": backend,
        "model": "resnet50",
        "case": "full" if full else "b052",
        "precision": "legacy" if full else "fp16",
        "split_boundary_precision": "" if full else "fp16",
        "setup_id": "deepx-host",
        "comparison_backend": "deepx",
        "direction": "deepx" if full else "deepx_to_trt",
        "task": "classification",
        "duration_s": 60.0,
        "contract_consistent": True,
        # Deliberately empty: these fields must come from the exact Native
        # validation/central-quality bridge, not from a relaxed claim gate.
        "contract_hash": "",
        "preprocessing_hash": "",
        "execution_precision": "",
        "prepared_feed_task": "classification",
        "prepared_feed_preprocess_mode": "resize",
        "prepared_feed_letterbox_pad_value": 0,
        "prepared_feed_source_image_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "validation_input_or_image_sha256": "1" * 64,
        "_energy": energy,
    }


def _validation(plan: dict[str, object]) -> dict[str, object]:
    return {
        "backend": plan["backend"],
        "model": plan["model"],
        "case": plan["case"],
        "precision": plan["precision"],
        "setup_id": plan["setup_id"],
        "comparison_backend": plan["comparison_backend"],
        "task": "classification",
        "runtime_precision_identity": "fp16",
        "execution_precision": "fp16",
        "full_runtime_precision": "fp16" if str(plan["backend"]).startswith("native_full_") else "",
        "endpoint_contract_hash": "3" * 64,
        "endpoint_contract_complete": True,
        "stage": "classification_logits",
        "contract_family": "classification_logits",
        "quality_contract_sha256": "4" * 64,
        "preprocessing_contract_sha256": "5" * 64,
        "source_request_sha256": "6" * 64,
        "model_sha256": "2" * 64,
        "validation_dataset_sha256": "7" * 64,
        "validation_dataset_image_ids_sha256": "8" * 64,
        "validation_dataset_ground_truth_sha256": "9" * 64,
        "accuracy_gate_policy_sha256": "a" * 64,
        "task_quality_policy_sha256": "a" * 64,
        "runtime_quality_gate_policy_sha256": "a" * 64,
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_claim_result_verified": True,
        "top1_match": True,
        "semantic_ok": True,
        "claim_ok": True,
        "contract_consistent": True,
        "gate_status": "pass",
    }


def test_energy_import_joins_exact_native_identity_and_makes_valid_pair_comparable(
    tmp_path: Path,
) -> None:
    split = _plan("deepx_to_trt", energy=0.10)
    full = _plan("native_full_deepx", energy=0.12)
    validation_rows = [_validation(split), _validation(full)]
    # Central-quality contracts describe the producer implementation.  A
    # composed DeepX->TRT producer and a native DeepX Full producer therefore
    # legitimately have different quality/preprocessing contract hashes.
    validation_rows[1]["quality_contract_sha256"] = "6" * 64
    validation_rows[1]["preprocessing_contract_sha256"] = "7" * 64
    validation_path = (
        tmp_path / "reports" / "native_validation"
        / "native_producer_validation_summary.json"
    )
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        json.dumps({"rows": validation_rows}), encoding="utf-8",
    )
    results_path = (
        tmp_path / "reports" / "native_energy_measurements"
        / "native_producer_energy_results.json"
    )
    results_path.parent.mkdir(parents=True)
    results_path.write_text(json.dumps({
        "rows": [
            {
                "row": {key: value for key, value in plan.items() if key != "_energy"},
                "ok": True,
                "run": {"rc": 0, "stdout_tail": _energy_payload(float(plan["_energy"]))},
            }
            for plan in (split, full)
        ],
    }), encoding="utf-8")

    rows = collect_native_energy(tmp_path)
    assert len(rows) == 2
    assert all(row["native_validation_join_status"] == "exact_unique" for row in rows)
    assert {row["quality_contract_sha256"] for row in rows} == {"4" * 64, "6" * 64}
    assert {row["preprocessing_contract_sha256"] for row in rows} == {"5" * 64, "7" * 64}
    assert all(row["output_endpoint_id"].endswith("3" * 64) for row in rows)
    assert all(row["runtime_precision_status"] == "verified_explicit" for row in rows)
    assert all(row["quality_verified"] is True for row in rows)
    assert all(row["quality_provenance_complete"] is True for row in rows)
    assert {row["source_request_sha256"] for row in rows} == {"6" * 64}
    assert {row["validation_dataset_sha256"] for row in rows} == {"7" * 64}
    assert {row["validation_dataset_image_ids_sha256"] for row in rows} == {"8" * 64}
    assert {row["validation_dataset_ground_truth_sha256"] for row in rows} == {"9" * 64}
    assert {row["task_quality_policy_sha256"] for row in rows} == {"a" * 64}
    assert {row["accuracy_gate_policy_sha256"] for row in rows} == {"a" * 64}
    assert all(row["semantic_claim_ok"] is True for row in rows)
    assert all(row["claim_eligible"] is True for row in rows)
    # The integer zero is a real resize pad value and must survive ingestion.
    assert all(row["prepared_feed_letterbox_pad_value"] == "0" for row in rows)

    pair = build_native_energy_pairs(rows)[0]
    assert pair["comparable"] is True
    assert pair["split_output_endpoint_id"] == pair["baseline_output_endpoint_id"]
    assert pair["energy_ratio"] == 0.10 / 0.12
    assert pair["validation_dataset_sha256"] == "7" * 64

    producer_specific_request = dict(rows[1], source_request_sha256="b" * 64)
    assert build_native_energy_pairs([rows[0], producer_specific_request])[0]["comparable"] is True

    missing_request = dict(rows[1], source_request_sha256="")
    rejected = build_native_energy_pairs([rows[0], missing_request])[0]
    assert rejected["comparable"] is False
    assert "baseline_source_request_sha256_missing_or_invalid" in rejected["comparison_reasons"]

    mismatched_full = dict(rows[1], validation_dataset_sha256="f" * 64)
    rejected = build_native_energy_pairs([rows[0], mismatched_full])[0]
    assert rejected["comparable"] is False
    assert "validation_dataset_hash_mismatch_or_missing" in rejected["comparison_reasons"]

    mismatched_policy = dict(rows[1], accuracy_gate_policy_sha256="f" * 64)
    rejected = build_native_energy_pairs([rows[0], mismatched_policy])[0]
    assert rejected["comparable"] is False
    assert "accuracy_gate_policy_hash_mismatch_or_missing" in rejected["comparison_reasons"]


def test_energy_import_keeps_pipeline_and_quality_hash_namespaces_separate(
    tmp_path: Path,
) -> None:
    plan = _plan("deepx_to_trt", energy=0.10)
    plan["pipeline_contract_sha256"] = "a" * 64
    validation = _validation(plan)
    validation["quality_contract_sha256"] = "b" * 64
    validation_path = (
        tmp_path / "reports" / "native_validation"
        / "native_producer_validation_summary.json"
    )
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(json.dumps({"rows": [validation]}), encoding="utf-8")
    results_path = (
        tmp_path / "reports" / "native_energy_measurements"
        / "native_producer_energy_results.json"
    )
    results_path.parent.mkdir(parents=True)
    results_path.write_text(json.dumps({"rows": [{
        "row": {key: value for key, value in plan.items() if key != "_energy"},
        "ok": True,
        "run": {"rc": 0, "stdout_tail": _energy_payload(0.10)},
    }]}), encoding="utf-8")

    row = collect_native_energy(tmp_path)[0]
    assert row["pipeline_contract_sha256"] == "a" * 64
    assert row["quality_contract_sha256"] == "b" * 64
    assert row["contract_hash"] == "a" * 64
    assert row["native_identity_evidence_conflicts"] == []
    assert row["claim_eligible"] is True


def test_recursive_hailo8_mirror_is_not_counted_as_second_repetition() -> None:
    module = _load_script("native_producer_final_report.py")
    record = {
        "repetition_index": 1,
        "repetition_id": "hailo8:one-runtime",
        "runtime_instance_id": "fresh_process:one-runtime",
        "workload_contract_sha256": "c" * 64,
        "repetition_runtime_scope": "fresh_process_per_repetition",
        "ok": True,
        "status": "ok",
        "fps_makespan": 123.0,
        "latency_mean_ms": 1000.0 / 123.0,
    }
    common = {
        "backend": "hailo8_to_trt",
        "producer_impl": "hailo8_fifo_trt_part2",
        "model": "resnet50",
        "case": "b052",
        "precision": "fp16",
        "setup_id": "hailo8-host",
        "task": "classification",
        "ok": True,
        "report": "/same/native_fifo_results.json",
        "fps_makespan": 123.0,
        "fps_repetition_samples": [123.0],
        "repetition_count_requested": 1,
        "repetition_count_attempted": 1,
        "repetition_count_valid": 1,
        "repetition_records": [record],
    }
    rows = [
        {**common, "analysis_summary": "/parent/analysis.json", "source_root": "/parent"},
        {**common, "analysis_summary": "/child/analysis.json", "source_root": "/parent/child"},
    ]

    result = module._aggregate_repetitions(rows)[0]
    assert result["repetition_count_requested"] == 1
    assert result["repetition_count_attempted"] == 1
    assert result["repetition_count_valid"] == 1
    assert result["repetition_status"] == "complete"
    assert result["fps_repetition_samples"] == [123.0]
    assert len(result["repetition_records"]) == 1
    assert result["source_reports"] == ["/same/native_fifo_results.json"]


def test_repetition_count_never_claims_more_valid_samples_than_vector() -> None:
    module = _load_script("native_producer_final_report.py")
    row = {
        "backend": "hailo8_to_trt",
        "producer_impl": "hailo8_fifo_trt_part2",
        "model": "resnet50", "case": "b052", "precision": "fp16",
        "setup_id": "hailo8-host", "task": "classification", "ok": True,
        "report": "/one/report.json", "fps_makespan": 100.0,
        "fps_repetition_samples": [100.0, 100.0],
        "repetition_count_requested": 2,
        "repetition_count_attempted": 2,
        "repetition_count_valid": 2,
        "repetition_records": [{
            "repetition_id": "hailo8:only-one",
            "runtime_instance_id": "fresh_process:only-one",
            "ok": True, "status": "ok", "fps_makespan": 100.0,
        }],
    }
    result = module._aggregate_repetitions([row])[0]
    assert result["repetition_count_requested"] == 2
    assert result["repetition_count_attempted"] == 2
    assert result["repetition_count_valid"] == 1
    assert result["repetition_status"] == "partial"
    assert result["fps_repetition_samples"] == [100.0]


def test_energy_plan_accepts_only_complete_exact_native_quality_bridge() -> None:
    module = _load_script("native_producer_energy_plan.py")
    validation = _validation(_plan("deepx_to_trt", energy=0.10))
    assert module._native_quality_bridge_verified(validation, "classification") is True
    endpoint = module._native_output_endpoint_id(validation, "classification")
    assert endpoint == f"classification:classification_logits:{'3' * 64}"
    validation["runtime_quality_gate_policy_sha256"] = ""
    assert module._native_quality_bridge_verified(validation, "classification") is False
    validation["runtime_quality_gate_policy_sha256"] = "a" * 64
    validation["runtime_precision_identity"] = ""
    validation["execution_precision"] = ""
    assert module._native_quality_bridge_verified(validation, "classification") is False


def test_final_report_accepts_validation_summary_v6_and_hailo_artifact_precision() -> None:
    module = _load_script("native_producer_final_report.py")
    rows, metadata = module._attach_quality_evidence([], {
        "schema": "onnx-splitpoint/native-producer-validation-summary",
        "schema_version": 6,
        "status": "complete",
        "row_count": 0,
        "rows": [],
    })
    assert rows == []
    assert metadata["schema_valid"] is True
    assert metadata["schema_version"] == 6

    hef_sha = "a" * 64
    assert module._runtime_precision_identity({
        "backend": "native_full_hailo8",
        "case": "full",
        "full_command_contract": {
            "artifacts": {"hef": {"sha256": hef_sha}},
        },
    }) == f"hailo_hef_sha256:{hef_sha}"
