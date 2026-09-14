from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.native_energy_reporting import (
    build_native_energy_ab_aggregates,
    collect_native_energy,
)
from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
from onnx_splitpoint_tool.workflow.analysis_pack import create_analysis_pack
from onnx_splitpoint_tool.workflow.cross_runner_reporting import compute_cross_runner_report


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def test_native_matrix_exports_all_rows_as_screening_and_keeps_measured_fps(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    rows = []
    expected = []
    for index in range(18):
        model = "yolo26s" if index % 2 else "resnet50"
        backend = "hailo8_to_trt" if index == 1 else f"backend_{index}"
        case = "b038" if model == "yolo26s" else "b052"
        fps = 198.801799 if index == 1 else float(index + 1)
        rows.append({
            "model": model, "backend": backend, "case": case, "precision": "p",
            "execution_mode": "native_split", "runtime_status": "ok", "fps": fps,
            "semantic_status": "claim_ok", "semantic_ok": True, "claim_ok": True,
            "contract_consistent": True,
        })
        expected.append({"model": model, "backend": backend, "case": case})
    _write(reports / "native_stage_concise_summary.json", {"rows": rows})
    _write(reports / "native_expected_matrix.json", {
        "expected_row_count": 18, "present_expected_row_count": 18,
        "missing_expected_row_count": 0, "present_expected_rows": expected,
    })
    _write(reports / "native_producer_combined_summary.json", {"rows": [{
        "model": "yolo26s", "backend": "hailo8_to_trt", "case": "b038", "precision": "p",
        "ok": True, "fps_makespan": 198.801799, "paper_fps": 212.712848,
        "p1_thread_ms": 2.196514, "p2_thread_ms": 4.701173,
        "native_command_contract": {"setup_id": "h8_setup"},
    }]})

    matrix = collect_native_performance_matrix(tmp_path)
    assert matrix["matrix_complete"] is True
    assert matrix["observation_count"] == 18
    h8 = next(row for row in matrix["observations"] if row["backend"] == "hailo8_to_trt")
    assert h8["native_measured_throughput_fps"] == 198.801799
    assert h8["native_theoretical_cycle_rate_fps"] == 212.712848
    assert h8["claim_eligible"] is False
    assert h8["setup_id"] == "h8_setup"


def test_cross_runner_labels_measured_fps_not_inverse_stage_cycle(tmp_path: Path) -> None:
    endpoint_hash = "a" * 64
    endpoint = {
        "task": "detection", "stage": "decoded_nms",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": endpoint_hash,
        "output_endpoint_attestation": {
            "attested": True, "status": "passed",
            "endpoint_contract_hash": endpoint_hash,
        },
    }
    _write(tmp_path / "reports" / "native_producer_combined_summary.json", {"rows": [{
        "model": "yolo26s", "backend": "hailo8_to_trt", "case": "b038",
        "precision": "uint8_dequant_fp16", "ok": True, "status": "ok",
        "setup_id": "h8_setup", "comparison_backend": "hailo8",
        "fps_makespan": 198.801799, "paper_fps": 212.712848,
        "p1_thread_ms": 2.196514, "p2_thread_ms": 4.701173,
        **endpoint,
    }]})
    _write(tmp_path / "reports" / "native_validation" / "native_producer_validation_summary.json", {"rows": [{
        "model": "yolo26s", "backend": "hailo8_to_trt", "case": "b038",
        "precision": "uint8_dequant_fp16", "setup_id": "h8_setup",
        "comparison_backend": "hailo8", **endpoint,
    }]})
    generic = [{
        "model_id": "yolo26s", "case_id": "b038", "direction": "hailo8_to_trt",
        "backend": "hailo8_to_tensorrt", "runner_regime": "generic",
        "precision": "uint8_dequant_fp16", "setup_id": "h8_setup",
        "comparison_backend": "hailo8", **endpoint,
        "cycle_ms": 17.95, "contract_consistent": False,
        "task_quality_status": "inconclusive",
    }]
    result = compute_cross_runner_report(tmp_path, generic)
    pair = result["pairs"][0]
    assert pair["native_throughput_fps"] == 198.801799
    assert pair["native_measured_throughput_fps"] == 198.801799
    assert pair["native_theoretical_cycle_rate_fps"] == 212.712848
    assert pair["native_throughput_fps"] != 1000.0 / 2.196514


def test_embedded_energy_aggregate_exports_repeat_n_mean_ci_and_ab_status(tmp_path: Path) -> None:
    aggregate = {
        "status": "ok", "run_count": 3, "valid_postprocessed_runs": 3,
        "raw_postprocessed_run_count": 3, "confidence_level": 0.95,
        "scientific_primary_method_frozen": True,
        "scientific_primary_method": "chapter4_baseline",
        "scientific_primary_energy_status": "available",
        "scientific_primary_valid_run_count": 3,
        "scientific_primary_claim_eligible": True,
        "scientific_primary_energy_total_j": 10.0,
        "scientific_primary_energy_per_work_unit_j": 0.1,
        "scientific_primary_active_duration_s": 2.0,
        "scientific_primary_avg_power_w": 5.0,
        "scientific_primary_energy_statistics": {
            "energy_j": {"n": 3, "mean": 10.0, "sample_stddev": 0.2, "ci_low": 9.5, "ci_high": 10.5},
            "energy_per_work_unit_j": {"n": 3, "mean": 0.1, "sample_stddev": 0.002, "ci_low": 0.095, "ci_high": 0.105},
            "avg_power_w": {"n": 3, "mean": 5.0, "ci_low": 4.8, "ci_high": 5.2},
        },
        "candidate_v263_shadow_energy_total_j": 9.8,
        "candidate_v263_shadow_energy_per_work_unit_j": 0.098,
        "candidate_v263_shadow_statistics": {
            "energy_j": {"n": 3, "mean": 9.8, "ci_low": 9.4, "ci_high": 10.2},
            "energy_per_work_unit_j": {"n": 3, "mean": 0.098, "ci_low": 0.094, "ci_high": 0.102},
        },
        "legacy_window_comparison_requested": True,
        "legacy_window_comparison_attempted_runs": 3,
        "legacy_window_comparison_successful_runs": 3,
        "window_method_comparison_statistics": {
            "difference_direction": "legacy_minus_command_window",
            "energy_j": {"relative_percent": {"n": 3, "mean": 1.1, "ci_low": 0.8, "ci_high": 1.4}},
            "duration_s": {"relative_percent": {"n": 3, "mean": 1.3, "ci_low": 1.0, "ci_high": 1.6}},
            "average_power_w": {"relative_percent": {"n": 3, "mean": -0.2, "ci_low": -0.3, "ci_high": -0.1}},
        },
        "avg_energy_work_units_used": 100,
        "energy_work_units_source": "runtime_completed_work_units",
        "runtime_completed_work_unit_run_count": 3,
        "energy_efficiency_claim_eligible": True,
        "final_energy_gate_status": "pass", "postprocess_status": "ok",
        "energy_window_effective_values": ["command_window"],
        "energy_window_requested": "command", "energy_physical_scope": "MB",
        "energy_primary_metric": "calibrated_input_energy_unsubtracted",
        "energy_calibrated_input_unsubtracted": True, "energy_raw_primary": True,
    }
    result = {"rows": [{
        "ok": True,
        "row": {
            "backend": "deepx_to_trt", "model": "resnet50", "case": "b052",
            "precision": "fp16", "setup_id": "deepx", "claim_ok": True,
            "contract_consistent": True, "task": "classification",
            "prepared_feed_task": "classification", "prepared_feed_preprocess_mode": "resize",
            "prepared_feed_letterbox_pad_value": 0, "prepared_feed_source_image_sha256": "a" * 64,
        },
        "run": {"rc": 0, "energy_aggregate": aggregate, "energy_aggregate_embedded": True},
    }]}
    _write(tmp_path / "reports" / "native_energy_measurements" / "native_producer_energy_results.json", result)
    row = collect_native_energy(tmp_path)[0]
    assert row["energy_repeat_n"] == 3
    assert row["energy_repeat_status"] == "complete"
    assert row["energy_total_j_mean"] == 10.0
    assert row["energy_total_j_ci_low"] == 9.5
    assert row["energy_ab_status"] == "complete"
    assert row["energy_ab_valid_n"] == 3
    assert row["ab_energy_relative_percent_mean"] == 1.1
    exported = build_native_energy_ab_aggregates([row])[0]
    assert exported["energy_repeat_valid_n"] == 3
    assert exported["ab_energy_relative_percent_ci_high"] == 1.4


def test_execution_plan_distinguishes_logical_runs_from_expected_rows() -> None:
    profile = {
        "model_suite": {"primary": [
            {"id": "resnet50", "task": "classification", "enabled": True},
            {"id": "yolo26s", "task": "detection", "enabled": True},
        ]},
        "run_profiles": [{"id": run_id, "enabled": True} for run_id in (
            "ort_tensorrt", "hailo8", "hailo8_to_trt", "hailo10",
            "hailo10_to_tensorrt", "deepx_m1_full", "deepx_m1_to_tensorrt",
        )],
        "selection_policy": {"max_accepted_cases_per_model": 1},
        "quality_gate": {"statistics": {"execution_location": "central_management", "workers": 4}},
        "execution_preset": {"id": "smoke", "snapshot": {"quality": {}, "runtime": {}, "defaults": {}}},
    }
    plan = build_effective_execution_plan(profile)
    assert plan["generic_logical_runs_total"] == 14
    assert plan["expected_generic_result_rows_total"] == 16
    assert plan["generic_rows_total"] == 16


def test_analysis_pack_keeps_compact_root_evidence(tmp_path: Path) -> None:
    run = tmp_path / "run"
    scientific = run / "reports" / "scientific"
    _write(scientific / "scientific_report.json", {"schema": "onnx-splitpoint/scientific-report"})
    (scientific / "scientific_report.md").write_text("# report\n", encoding="utf-8")
    for name in ("row_eligibility.csv", "task_quality.csv", "performance_results.csv", "energy_results.csv"):
        (scientific / name).write_text("model_id\n", encoding="utf-8")
    _write(scientific / "native_performance_matrix.json", {"observations": []})
    _write(scientific / "native_energy_ab_aggregates.json", [])
    _write(run / "stages" / "run_native_producers" / "stage_result.json", {"status": "ok"})
    _write(run / "quality_management" / "central_quality_summary.json", {"status": "ok"})
    _write(run / "reports" / "native_expected_matrix.json", {"expected_row_count": 18})
    _write(run / "run_manifest.json", {"run_id": "run"})
    (run / "profile.yaml").write_text(yaml.safe_dump({"model_suite": {"primary": []}}), encoding="utf-8")
    out = tmp_path / "analysis.zip"
    create_analysis_pack(run, out, materialize_missing_report=True)
    with zipfile.ZipFile(out) as archive:
        names = set(archive.namelist())
    assert "99_provenance/stages/run_native_producers/stage_result.json" in names
    assert "99_provenance/quality_management/central_quality_summary.json" in names
    assert "99_provenance/reports/native_expected_matrix.json" in names
    assert "01_results/native_performance_matrix.json" in names
    assert "01_results/native_energy_ab_aggregates.json" in names
