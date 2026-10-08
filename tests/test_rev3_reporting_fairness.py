"""Revision 3 comparison gates keep observations separate from estimates."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.cross_runner_reporting import (
    _cycle_ms, _measured_generic_fps, _quality_decision, _timing_input_match,
    compute_cross_runner_report,
)
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row
from onnx_splitpoint_tool.workflow.scientific_reporting import _scientific_row
from test_v276_cross_runner_reporting import _build_bounded_24_fixture


def _binding():
    return {"input_image_sha256": "a" * 64, "canonical_rgb_sha256": "b" * 64,
            "image_sequence": ["image.jpg"], "preprocessing_contract_sha256": "c" * 64}


def test_measured_count_time_wins_over_faster_cycle_estimate():
    row = {"throughput_fps_makespan": 40.0, "pipeline_cycle_selected_ms": 2.0,
           "throughput_mode": "measured_streaming"}
    assert _measured_generic_fps(row) == (40.0, "throughput_fps_makespan")
    assert _cycle_ms(row) == 25.0
    out = _scientific_row(row)
    assert out["throughput_fps"] == 40.0
    assert out["cycle_ms"] == 25.0
    assert out["diagnostic_cycle_ms"] == 2.0


def test_estimated_rate_is_retained_only_as_diagnostic_ordering():
    row = {"throughput_fps_makespan": 500.0, "pipeline_cycle_selected_ms": 2.0,
           "throughput_mode": "estimated_interleaved"}
    assert _measured_generic_fps(row)[0] is None
    assert _cycle_ms(row) == 2.0
    assert _scientific_row(row)["measured_throughput_fps"] is None


@pytest.mark.parametrize("key,value,reason", [
    ("input_image_sha256", "d" * 64, "source_image_mismatch"),
    ("canonical_rgb_sha256", "d" * 64, "canonical_rgb_mismatch"),
    ("image_sequence", ["other.jpg"], "image_sequence_mismatch"),
])
def test_timing_mismatch_rejected(key, value, reason):
    a, b = _binding(), _binding()
    b[key] = value
    assert _timing_input_match({"timing_input_binding": a}, {"timing_input_binding": b}) == (False, reason)


def test_backend_dtype_and_parallelism_can_differ_with_equal_canonical_raster():
    a, b = _binding(), _binding()
    a["backend_prepared_inputs"] = {"input": {"dtype": "float16", "sha256": "1" * 64}}
    b["backend_prepared_inputs"] = {"input": {"dtype": "uint8", "sha256": "2" * 64}}
    assert _timing_input_match({"timing_input_binding": a, "inflight": 1},
                               {"timing_input_binding": b, "inflight": 8}) == (True, "canonical_rgb_match")


def test_historical_exact_prepared_input_requires_bytes_shape_and_dtype():
    a = {"source_image_sha256": "a" * 64, "entries": [
        {"name": "input", "sha256": "b" * 64, "shape": [1, 3, 5, 7], "dtype": "float32"}]}
    assert _timing_input_match({"input_binding": a}, {"input_binding": a}) == (True, "historical_exact_prepared_input_match")
    b = deepcopy(a)
    b["entries"][0]["shape"] = [1, 3, 7, 5]
    assert _timing_input_match({"input_binding": a}, {"input_binding": b}) == (False, "prepared_input_mismatch")
    assert not _timing_input_match({"input_image_sha256": "a" * 64}, {"input_image_sha256": "a" * 64})[0]


def test_quality_gate_boolean_cannot_hide_loss():
    assert _quality_decision({"accuracy_gate_pass": True}) == "unavailable"
    assert _quality_decision({"accuracy_gate_pass": True, "task_quality_status": "accuracy_loss"}) == "accuracy_loss"


def test_normal_and_scientific_exports_keep_actual_input_and_scheduling(tmp_path):
    raw = {"case_id": "b001", "primary_variant": "composed", "stage1_provider": "hailo8",
           "stage2_provider": "tensorrt", "composed_mean_ms": 5.0,
           "timing_input_binding": _binding(), "fps_makespan": 40.0,
           "streaming_impl": "generic", "inflight": 4, "input_copy_mode": "owned",
           "generic_runtime_source_binding": {"source_sha256": "d" * 64}}
    row = normalize_benchmark_row(raw, model_id="resnet50", source_path=tmp_path / "raw.json", tag="hailo8_to_trt")
    result = _scientific_row(row)
    for k in ["timing_input_binding", "streaming_impl", "inflight", "input_copy_mode", "generic_runtime_source_binding"]:
        assert result[k] == raw[k]
    assert result["measured_throughput_fps"] == 40.0


@pytest.mark.parametrize("mutation,reason", [
    ("input", "timing_input_matches"),
    ("raw", "generic_completed_task_measured"),
    ("estimated", "generic_measured_makespan_present"),
    ("window", "measurement_windows_comparable"),
])
def test_real_comparison_gate_keeps_invalid_pairs_visible(tmp_path, mutation, reason):
    g, n = tmp_path / "generic", tmp_path / "native"
    rows = _build_bounded_24_fixture(g, n)
    if mutation == "input":
        rows[0]["timing_input_binding"]["canonical_rgb_sha256"] = "9" * 64
    elif mutation == "raw":
        rows[0]["measurement_endpoint"] = "classification_logits"
    elif mutation == "window":
        rows[0]["measurement_boundary"] = "outputs_ready_to_postprocess_done"
    else:
        rows[0]["throughput_mode"] = "estimated_interleaved"
    report = compute_cross_runner_report(g, rows, native_run_dir=n, minimum_candidates=3)
    assert report["pair_count"] == 24
    assert report["technical_pair_count"] == 23
    pair = next(x for x in report["pairs"] if x["model_id"] == rows[0]["model_id"] and x["case_id"] == rows[0]["case_id"] and x["setup_id"] == rows[0]["setup_id"])
    assert pair["technical_transfer_eligibility_checks"][reason] is False
    assert pair["eligible_for_technical_transfer"] is False


def test_native_projection_uses_admitted_raw_input_binding(tmp_path):
    from onnx_splitpoint_tool.native_rate_endpoints import report_rate_fields
    raw = {"task": "classification", "measurement_boundary": "first_task_start_to_last_task_completion",
           "measurement_endpoint": "completed_task", "completed_work_units": 1000,
           "makespan_ms": 10000.0, "fps_makespan": 100.0,
           "timing_input_binding": _binding(), "inflight": 4, "runtime_preprocessing_sha256": "e" * 64}
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw))
    projected = report_rate_fields({"report": str(path)}, tmp_path)
    assert projected["timing_input_binding"] == raw["timing_input_binding"]
    assert projected["runtime_preprocessing_sha256"] == "e" * 64
    assert projected["inflight"] == 4
    assert projected["completed_task_fps"] == 100.0


def test_native_projection_rejects_last_repeat_input_as_mixed_cohort_proof():
    from onnx_splitpoint_tool.native_rate_endpoints import _execution_metadata_fields
    a, b = _binding(), _binding()
    b["canonical_rgb_sha256"] = "f" * 64
    projected = _execution_metadata_fields({"timing_input_binding": b, "repetition_records": [
        {"timing_input_binding": a}, {"timing_input_binding": b}]})
    assert projected["execution_metadata_conflicts"] == ["timing_input_binding"]
    assert _timing_input_match({"timing_input_binding": a}, projected) == (False, "mixed_repetition_execution_metadata")
