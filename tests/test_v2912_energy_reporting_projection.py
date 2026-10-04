from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.native_energy_reporting import (
    build_native_energy_pairs, collect_native_energy, scientific_energy_rows,
    summarize_native_energy_reporting,
)
from onnx_splitpoint_tool.workflow import scientific_reporting as reporting


def _row(full=False):
    row = {
        "model": "detector", "backend": "native_full_deepx" if full else "deepx_to_trt",
        "case": "full" if full else "b001", "setup_id": "setup",
        "execution_mode": "native_full_baseline" if full else "native_split",
        "task": "detection", "task_source": "declared", "direction": "deepx_to_trt",
        "precision": "fp16", "prepared_feed_task": "detection",
        "prepared_feed_preprocess_mode": "letterbox", "prepared_feed_letterbox_pad_value": 0,
        "energy_scope": "full_system", "energy_window_effective": "command_window",
        "target_duration_s": 60.0, "active_duration_s": 60.0,
        "energy_per_work_j": 2.0 if full else 1.0, "average_power_w": 10.0,
        "energy_primary_metric": "raw_input_energy", "energy_raw_primary": True,
        "energy_calibrated_input_unsubtracted": True, "ok": True,
        "runtime_work_units_exact": True, "energy_calibration_verified": True,
        "endpoint_contract_complete": True, "output_endpoint_match": True,
        "completion_pairing_eligible": True, "semantic_claim_ok": True,
        "screening_only": True, "diagnostic_only": True, "screening_comparable": False,
        "claim_eligible": False, "claim_comparable": False, "energy_repeat_valid_n": 3,
        "evaluation_role": "unknown", "evaluation_role_raw": "",
        "evaluation_role_source": "original.json#/row/evaluation_role",
    }
    for field in (
        "model_sha256", "validation_dataset_sha256", "validation_dataset_image_ids_sha256",
        "validation_dataset_ground_truth_sha256", "validation_input_or_image_sha256",
        "prepared_feed_source_image_sha256", "preprocessing_contract_sha256",
        "decoder_contract_sha256", "nms_contract_sha256", "energy_calibration_sha256",
    ):
        row[field] = "a" * 64
    row["comparison_output_endpoint_id"] = "detection:decoded_nms:comparison:" + "a" * 64
    row["comparison_endpoint_contract_hash"] = "a" * 64
    row["comparison_endpoint_stage"] = "decoded_nms"
    row["output_endpoint_id"] = row["comparison_output_endpoint_id"]
    row["endpoint_contract_hash"] = "a" * 64
    return row


def test_screening_flags_do_not_hide_exact_scope_descriptive_ratio():
    rows = [_row(), _row(True)]
    original = copy.deepcopy(rows)
    pair = build_native_energy_pairs(rows)[0]
    assert pair["descriptive_comparable"] is True
    assert pair["descriptive_energy_ratio"] == 0.5
    assert pair["descriptive_energy_saving_fraction"] == 0.5
    assert pair["semantic_comparable"] is True
    assert pair["claim_comparable"] is False
    assert pair["screening_comparable"] is False
    assert pair["energy_ratio"] is None
    assert rows == original
    summary = summarize_native_energy_reporting(rows, [pair])
    assert summary["energy_recorded_row_count"] == 2
    assert summary["energy_valid_replicate_count"] == 6
    assert summary["energy_descriptive_pair_count"] == 1
    assert summary["energy_semantically_matching_pair_count"] == 1
    assert summary["energy_scientifically_released_pair_count"] == 0


@pytest.mark.parametrize("field,value", [
    ("setup_id", "other"), ("model_sha256", "b" * 64),
    ("validation_input_or_image_sha256", "b" * 64),
    ("validation_dataset_sha256", "b" * 64),
    ("prepared_feed_letterbox_pad_value", 114),
    ("preprocessing_contract_sha256", "b" * 64),
    ("comparison_endpoint_contract_hash", "b" * 64),
    ("energy_scope", "module"), ("energy_window_effective", "idle"),
    ("energy_calibration_verified", False), ("runtime_work_units_exact", False),
    ("target_duration_s", 30), ("active_duration_s", 45),
    ("energy_per_work_j", float("nan")),
])
def test_descriptive_ratios_reject_scope_or_numeric_drift(field, value):
    split, full = _row(), _row(True)
    full[field] = value
    pair = build_native_energy_pairs([split, full])[0]
    assert pair["descriptive_comparable"] is False
    assert pair["descriptive_energy_ratio"] is None


def test_different_decoder_contract_is_not_declared_semantically_equal():
    split, full = _row(), _row(True)
    full["decoder_contract_sha256"] = "b" * 64
    pair = build_native_energy_pairs([split, full])[0]
    assert pair["descriptive_comparable"] is True
    assert pair["semantic_comparable"] is False
    assert "decoder_contract_sha256_identity_differs_or_missing" in pair["semantic_comparison_reasons"]
    assert pair["baseline_decoder_contract_sha256"] == "b" * 64
    assert pair["claim_comparable"] is False


def test_original_missing_role_and_flags_are_preserved(tmp_path):
    dest = tmp_path / "reports/native_energy_measurements/native_producer_energy_results.json"
    dest.parent.mkdir(parents=True)
    dest.write_text(json.dumps({"screening_only": True, "rows": [{
        "row": {"backend": "native_full_deepx", "model": "m", "evaluation_role": "",
                "screening_only": True, "diagnostic_only": True, "claim_eligible": False},
        "run": {"rc": 0}, "ok": True,
    }]}))
    row = collect_native_energy(tmp_path)[0]
    assert row["evaluation_role"] == "unknown"
    assert row["evaluation_role_raw"] == ""
    assert row["screening_only"] is True
    assert row["energy_role_declarations"]["plan"]["claim_eligible"] is False
    assert row["evaluation_role_source"].endswith("#/rows/0/row/evaluation_role")


def test_raw_fs_primary_is_not_replaced_by_trt_idle_estimate():
    row = {**_row(True), "backend": "native_full_tensorrt",
        "host_normalization_role": "tensorrt_full",
        "host_normalization_source_run_id": "native_full_tensorrt",
        "host_normalization_target_variant": "full", "host_normalization_identity_verified": True,
        "accelerator_idle_correction_requested": True, "accelerator_idle_correction_applied": True,
        "accelerator_idle_correction_statuses": ["applied"], "accelerator_idle_w_applied": 4.0,
        "accelerator_idle_calibration_verified": True, "accelerator_idle_calibration_status": "verified",
        "accelerator_idle_calibration_binding_sha256": "a" * 64,
        "energy_efficiency_claim_eligible": True, "energy_total_j": 120.0,
        "host_normalized_energy_est_j": 80.0, "energy_per_work_j": 0.12,
        "host_normalized_energy_per_work_est_j": 0.08, "average_power_w": 12.0,
        "host_normalized_average_power_est_w": 8.0, "active_duration_s": 10.0, "work_units": 1000,
        "energy_per_work_j_sample_stddev": 0.02, "energy_per_work_j_ci_low": 0.10,
        "energy_per_work_j_ci_high": 0.14,
    }
    projected = scientific_energy_rows(observations=[row])[0]
    assert projected["energy_total_j"] == 120.0
    assert projected["energy_per_work_j"] == 0.12
    assert projected["average_power_w"] == 12.0
    assert projected["energy_per_work_ci_low_j"] == 0.10
    assert projected["host_normalized_energy_per_work_est_j"] == 0.08
    assert projected["energy_eligible"] is False


def test_energy_export_contains_measurements_not_generic_eligibility(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "_make_figures", lambda *a, **kw: [])
    observations = [_row(), _row(True)]
    energy = scientific_energy_rows(observations=observations)
    generic = {"row_role": "performance_observation", "model_id": "generic",
               "energy_eligible": True, "throughput_fps": 100, "runtime_executable": True}
    reporting._write_reports(tmp_path / "reports", {
        "rows": [generic, *energy], "native_energy_observations": observations,
        "native_energy_pair_comparison": build_native_energy_pairs(observations), "summary": {},
    })
    def read(name):
        with (tmp_path / "reports" / name).open() as f:
            return list(csv.DictReader(f))
    exported = read("energy_results.csv")
    assert len(exported) == 2
    assert all(float(row["energy_per_work_j"]) > 0 for row in exported)
    assert read("claim_eligible_energy.csv") == []
    summary = json.loads((tmp_path / "reports/scientific_report.json").read_text())["summary"]
    assert summary["energy_eligible_count"] == 0
    assert summary["energy_recorded_row_count"] == 2
    assert summary["energy_descriptive_pair_count"] == 1
