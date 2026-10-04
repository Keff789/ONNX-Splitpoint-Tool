"""Selective Generic completion: actual timer -> writer -> parser -> comparison."""
import copy
import hashlib
import json

import numpy as np
import pytest

from onnx_splitpoint_tool.generic_completion_reporting import (
    BOUNDARY_CHECKS, IDENTITY_FIELDS, case_result_path, completion_groups,
    completion_pair, load_valid_repetitions, validate_completion_case,
    write_completion_reports,
)
from onnx_splitpoint_tool.runners.harness.classification import ClassificationCompletion
from onnx_splitpoint_tool.runners.task_completion import measure_split_completion
from onnx_splitpoint_tool.workflow.checkpoints import atomic_write_json


def case(boundary="b001", setup="setup1", endpoint="classification:bound_endpoint"):
    identity = dict(zip(IDENTITY_FIELDS, (
        "resnet50", boundary, "hailo8_to_tensorrt", "float32_layout_fp16",
        setup, "hailo8", endpoint,
    )))
    checks = {"generic_completed_task_measured": False, "native_runtime_ok": True,
              "native_completed_task_measured": True, "exact_identity_complete": True}
    return {
        "identity": identity, "case_key": "|".join(identity.values()),
        "input_binding": {"prepared_input": {"sha256": "existing-input-id"}},
        "completion_contract": {"task": "classification", "operation": "top1_top5", "label_offset": 0},
        "measurement_boundary": "prepared_input_before_p1_to_last_completed_task",
        "expected_runtime_artifacts": {"p1": {"path": "/fixture/p1.hef", "sha256": "bound-p1"}},
        "functional_reference": {"source": "/fixture/native/raw", "input_sha256": "existing-input-id",
                                 "outputs": {"logits": {"shape": [1, 10], "dtype": "float32"}}},
        "task": "classification", "original_run": "/fixture/original",
        "generic_raw_source": "/fixture/generic/raw", "native_source": "/fixture/native/raw",
        "evaluation_role": "development", "evaluation_role_source": "profile:model_suite.primary",
        "measurement_boundary_comparison": {"checks": dict.fromkeys(BOUNDARY_CHECKS, True),
            "sources": ["/fixture/native/command_contract.json", "/fixture/generic/config.json"]},
        "native_measurement_boundary": {"observed_completed_counts": [1000, 1000, 1000],
                                        "observed_measurement_times_s": [20, 20, 20]},
        "historical_pair": {**identity, "generic_cycle_ms": 10, "generic_throughput_fps": 100,
            "native_measured_throughput_fps": 50, "native_measured_fps_source": "completed_task_count_time",
            "generic_task_quality_status": "reference_close", "native_task_quality_status": "reference_close",
            "technical_transfer_eligibility_checks": checks,
            "quality_transfer_eligibility_checks": {**checks, "validation_accuracy_gate_pass": True},
            "claim_transfer_eligibility_checks": {**checks, "generic_quality_pass": False}},
    }


def repetition(config, index=1, seconds=10.0):
    return {
        "producer": "generic_split", "status": "ok", "repetition_index": index, "execution_id": "new-session",
        "runtime_instance_id": f"new-session:process-{index}", "pid": 1000 + index,
        "measurement_boundary": config["measurement_boundary"], "clock": "perf_counter_ns",
        "worker_count": 2, "queue_depth": 2, "handoff_payload_copy": True,
        "warmup_end_ns": 1, "measurement_start_ns": 2,
        "measurement_end_ns": 2 + int(seconds * 1e9), "measured_makespan_seconds": seconds,
        "completed_task_fps": 1000 / seconds,
        "requested_warmup_count": 100, "warmup_completed_count": 100,
        "requested_task_count": 1000, "completed_task_count": 1000, "processor_count_delta": 1100,
        "fill_and_drain_included": True, "preprocessing_timed": False, "handoff_timed": True,
        "postprocess_timed": True, "output_materialization_timed": True,
        "runtime_artifacts": config["expected_runtime_artifacts"],
        "functional_control": {"status": "passed", "reference_binding": config["functional_reference"],
                               "checked_outside_timer": True, "metrics": {"passed": True}},
        "output_sample_relation": "last_measured_completed_task", "output_samples_written_outside_timer": True,
        "raw_output_samples": {"logits": {"path": f"repeat_{index}/output.npy", "sha256": "fixture", "shape": [1, 10], "dtype": "float32"}},
        "last_completed_result": {"top1": [9], "top5": [[9, 8, 7, 6, 5]]},
        "completed_result_artifact": {"path": f"repeat_{index}/completed_task.json", "sha256": "fixture"},
    }


def result(config, count=3):
    return {**{key: copy.deepcopy(config[key]) for key in (
        "identity", "input_binding", "completion_contract", "measurement_boundary",
        "expected_runtime_artifacts", "functional_reference", "task", "original_run")},
        "schema_version": 1, "measurement_mode": "generic_completed_task", "execution_id": "new-session",
        "status": "ok" if count == 3 else "started",
        "repetitions": [repetition(config, i, seconds=10 * i) for i in range(1, count + 1)]}


def write_result(path, payload):
    """Save the measured output probe alongside the authoritative result JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for rep in payload["repetitions"]:
        if rep.get("status") != "ok":
            continue
        descriptor = rep["raw_output_samples"]["logits"]
        sample = path.parent / descriptor["path"]
        sample.parent.mkdir(parents=True, exist_ok=True)
        np.save(sample, np.arange(10, dtype=np.float32)[None, :], allow_pickle=False)
        descriptor["sha256"] = hashlib.sha256(sample.read_bytes()).hexdigest()
        completed = path.parent / rep["completed_result_artifact"]["path"]
        atomic_write_json(completed, rep["last_completed_result"])
        rep["completed_result_artifact"]["sha256"] = hashlib.sha256(completed.read_bytes()).hexdigest()
    atomic_write_json(path, payload)


def test_actual_timer_writer_parser_reporter_and_second_import(tmp_path):
    config = case()
    payload = result(config, 0)
    calls = []
    for index in range(1, 4):
        processor = ClassificationCompletion()
        measured = measure_split_completion(
            lambda: {"features": np.arange(10, dtype=np.float32)[None, :]},
            lambda output: output,
            lambda inputs: {"logits": inputs["features"]}, processor,
        )
        calls.append(processor.completed_count)
        rep = repetition(config, index)
        rep.update(measured)
        payload["repetitions"].append(rep)
        write_result(case_result_path(config, tmp_path / "results"), payload)
        assert len(load_valid_repetitions(case_result_path(config, tmp_path / "results"), config)) == index
    summary = write_completion_reports({"rows": [config]}, tmp_path / "results", tmp_path / "report")
    assert calls == [1100, 1100, 1100]
    assert summary["status_counts"]["completed"] == 1
    assert summary["valid_repetitions"] == 3 and summary["completed_task_count"] == 3000
    assert summary["technical_pairs"] == 1 and summary["claim_pairs"] == 0
    before = (tmp_path / "report/generic_completion_cross_runner.json").read_bytes()
    assert write_completion_reports({"rows": [config]}, tmp_path / "results", tmp_path / "report") == summary
    assert (tmp_path / "report/generic_completion_cross_runner.json").read_bytes() == before


def test_all_repetitions_used_median_and_sample_spread():
    config = case()
    observation = validate_completion_case(result(config), config)
    assert observation["completed_task_fps_median"] == 50
    assert observation["completed_task_fps_min"] == pytest.approx(100 / 3)
    assert observation["completed_task_fps_max"] == 100
    assert observation["measured_makespan_seconds_sum"] == 60
    assert observation["completed_task_fps_stddev"] > 30


@pytest.mark.parametrize("key", ["identity", "input_binding", "completion_contract", "measurement_boundary", "expected_runtime_artifacts", "functional_reference"])
def test_binding_changes_stop_import_and_resume(tmp_path, key):
    config = case(); payload = result(config)
    payload[key] = {"foreign": True}
    path = tmp_path / "result.json"; write_result(path, payload)
    with pytest.raises(ValueError, match="mismatch"):
        load_valid_repetitions(path, config)
    assert validate_completion_case(payload, config)["status"] == "blocked"


@pytest.mark.parametrize("field,value", [
    ("completed_task_count", 999), ("warmup_completed_count", 99), ("completed_task_fps", 99999),
    ("measurement_end_ns", 1), ("fill_and_drain_included", False), ("postprocess_timed", False),
    ("output_materialization_timed", False), ("preprocessing_timed", True),
    ("runtime_artifacts", {}), ("functional_control", {"status": "passed"}),
])
def test_invalid_repeat_does_not_make_complete(field, value):
    config = case(); payload = result(config)
    payload["repetitions"][0][field] = value
    assert validate_completion_case(payload, config)["status"] == "blocked"


def test_controlled_interruption_retains_completed_repetitions_and_no_false_success(tmp_path):
    config = case(); payload = result(config, 1)
    payload["repetitions"].append({"repetition_index": 2, "execution_id": "old-session", "status": "started"})
    path = tmp_path / "partial.json"; write_result(path, payload)
    assert [row["repetition_index"] for row in load_valid_repetitions(path, config)] == [1]
    projected = validate_completion_case(payload, config)
    assert projected["status"] == "partial" and projected["pending_repetition_indices"] == [2, 3]
    assert validate_completion_case(result(config, 0), config)["status"] == "unstarted"
    assert validate_completion_case(None, config)["status"] == "unstarted"


def test_exact_repeat_duplicate_deduplicates_but_conflicting_valid_repeat_blocks():
    config = case(); payload = result(config)
    payload["repetitions"].append(copy.deepcopy(payload["repetitions"][0]))
    assert validate_completion_case(payload, config)["valid_repetition_count"] == 3
    payload["repetitions"][-1]["execution_id"] = "another-valid-measurement"
    assert validate_completion_case(payload, config)["status"] == "blocked"


@pytest.mark.parametrize("check", BOUNDARY_CHECKS)
def test_ratio_needs_both_actual_boundaries(check):
    config = case(); observation = validate_completion_case(result(config), config)
    config["measurement_boundary_comparison"]["checks"][check] = False
    pair = completion_pair(observation, config)
    assert pair["native_over_generic_completed_task_ratio"] is None
    assert not pair["eligible_for_technical_transfer"]
    assert pair["historical_generic_raw_fps"] == 100
    assert pair["native_completed_task_fps"] == 50


def test_measured_native_only_and_known_hardware_conflicts():
    config = case(); observation = validate_completion_case(result(config), config)
    config["historical_pair"]["native_measured_fps_source"] = "paper_fps"
    assert completion_pair(observation, config)["native_over_generic_completed_task_ratio"] is None
    config["historical_pair"]["native_measured_fps_source"] = "completed_task_count_time"
    config["runtime_environment_comparison"] = {"status": "conflict", "reason": "changed_power_mode"}
    assert completion_pair(observation, config)["native_over_generic_completed_task_ratio"] is None


def test_negative_quality_is_measured_without_claim_or_role_upgrade():
    config = case()
    config["historical_pair"]["generic_task_quality_status"] = "accuracy_loss"
    config["historical_pair"]["quality_transfer_eligibility_checks"]["validation_accuracy_gate_pass"] = False
    pair = completion_pair(validate_completion_case(result(config), config), config)
    assert pair["native_over_generic_completed_task_ratio"] == 1
    assert pair["eligible_for_technical_transfer"]
    assert not pair["eligible_for_quality_transfer"] and not pair["eligible_for_claim_transfer"]
    assert pair["evaluation_role"] == "development" and pair["generic_task_quality_status"] == "accuracy_loss"


def test_rank_groups_do_not_pool_repeats_setups_or_endpoints_and_keep_ties():
    pairs = []
    for setup, boundaries in (("setup1", ("b001", "b002", "b003")), ("setup2", ("b001",))):
        for index, boundary in enumerate(boundaries, 1):
            config = case(boundary, setup)
            payload = result(config)
            # Same rates across candidates produce undefined correlation.
            pairs.append(completion_pair(validate_completion_case(payload, config), config))
    groups = completion_groups(pairs)
    assert len(groups) == 2
    assert groups[0]["technical_candidate_count"] == 3
    assert groups[0]["technical_status"] == "constant_ranks"
    assert groups[0]["technical_spearman_rho"] is None
    assert groups[1]["technical_candidate_count"] == 1
    assert groups[1]["technical_status"] == "insufficient_candidates"


def test_full_accounting_without_glob_import_or_duplicate_identity(tmp_path):
    configs = [case(f"b{i:03}") for i in range(1, 5)]
    for config, count in zip(configs[:3], (3, 1, 0)):
        payload = result(config, count)
        if count == 0:
            payload["status"] = "failed"
        write_result(case_result_path(config, tmp_path), payload)
    # An unrelated successful file cannot increase the result count.
    write_result(tmp_path / "foreign/generic_completion_results.json", result(case("foreign")))
    summary = write_completion_reports({"rows": configs}, tmp_path, tmp_path / "report")
    assert summary["status_counts"] == {"completed": 1, "partial": 1, "failed": 1, "blocked": 0, "cancelled": 0, "unstarted": 1}
    assert summary["valid_repetitions"] == 4
    with pytest.raises(ValueError, match="duplicate"):
        write_completion_reports({"rows": [configs[0], configs[0]]}, tmp_path, tmp_path / "report")


def test_native_reported_median_must_bind_original_completion_counts_and_spans():
    config = case()
    config["historical_pair"]["native_measured_throughput_fps"] = 999
    pair = completion_pair(validate_completion_case(result(config), config), config)
    assert pair["native_over_generic_completed_task_ratio"] is None
    assert "native_original_count_time_matches" in pair["technical_exclusion_reasons"]


def test_historical_backend_alias_and_boundary_source_mapping_join_normally():
    config = case()
    config["historical_pair"]["direction"] = "hailo8_to_trt"
    config["measurement_boundary_comparison"]["sources"] = {
        "native_source": "/fixture/native", "generic_source": "/fixture/generic"}
    assert completion_pair(validate_completion_case(result(config), config), config)["native_over_generic_completed_task_ratio"] == 1


def test_native_cpp_six_decimal_rate_keeps_existing_count_time_tolerance():
    config = case()
    config["native_measurement_boundary"]["observed_measurement_times_s"] = [7.262651378, 7.254796292, 7.265980638]
    config["historical_pair"]["native_measured_throughput_fps"] = 137.690762
    pair = completion_pair(validate_completion_case(result(config), config), config)
    assert pair["technical_transfer_eligibility_checks"]["native_original_count_time_matches"]
    assert pair["native_over_generic_completed_task_ratio"] == pytest.approx(137.690762 / 50)


def test_unknown_historical_configuration_is_explicit_diagnostic_limit():
    config = case()
    pair = completion_pair(validate_completion_case(result(config), config), config)
    assert pair["eligible_for_technical_transfer"]
    assert not pair["runtime_conditions_comparable"]
    assert not pair["claim_transfer_eligibility_checks"]["runtime_environment_comparability_verified"]
    assert "diagnostic" in pair["throughput_factor_interpretation"]


@pytest.mark.parametrize("change", ["missing", "changed", "escaped", "wrong_completed_result"])
def test_resume_checks_actual_persisted_output_samples(tmp_path, change):
    config = case(); payload = result(config)
    path = tmp_path / "result.json"; write_result(path, payload)
    sample = tmp_path / payload["repetitions"][0]["raw_output_samples"]["logits"]["path"]
    if change == "missing":
        sample.unlink()
    elif change == "changed":
        sample.write_bytes(b"changed output")
    elif change == "escaped":
        payload["repetitions"][0]["raw_output_samples"]["logits"]["path"] = "../outside.npy"
        atomic_write_json(path, payload)
    else:
        completed = tmp_path / payload["repetitions"][0]["completed_result_artifact"]["path"]
        atomic_write_json(completed, {"top1": [0]})
        payload["repetitions"][0]["completed_result_artifact"]["sha256"] = hashlib.sha256(completed.read_bytes()).hexdigest()
        atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="sample"):
        load_valid_repetitions(path, config)


def test_repetitions_are_independent_runtime_instances():
    config = case(); payload = result(config)
    payload["repetitions"][1]["runtime_instance_id"] = payload["repetitions"][0]["runtime_instance_id"]
    assert validate_completion_case(payload, config)["status"] == "blocked"


def test_preflight_failure_is_not_unstarted_and_keeps_valid_partial_work(tmp_path):
    blocked = case("b001"); blocked.update(execution_status="blocked", execution_error="missing bound artifact")
    partial = case("b002"); partial.update(execution_status="failed", execution_error="controlled interrupted second process")
    write_result(case_result_path(partial, tmp_path), result(partial, 1))
    summary = write_completion_reports({"rows": [blocked, partial]}, tmp_path, tmp_path / "report")
    assert summary["status_counts"]["blocked"] == 1
    assert summary["status_counts"]["partial"] == 1
    assert summary["status_counts"]["unstarted"] == 0
    assert summary["valid_repetitions"] == 1
    rows = json.loads((tmp_path / "report/generic_completion_observations.json").read_text())["rows"]
    assert rows[0]["errors"] == ["missing bound artifact"]
    assert rows[1]["errors"] == ["controlled interrupted second process"]
