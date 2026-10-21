"""Read-only actual-cohort replay; mutation cases operate only on local copies."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from onnx_splitpoint_tool import native_rate_endpoints as rates
from onnx_splitpoint_tool.native_performance_reporting import collect_native_performance_matrix
from onnx_splitpoint_tool.resources.remote_scripts import native_producer_final_report as reporter
from onnx_splitpoint_tool.workflow import hardware_acceptance_native as native


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture(scope="module")
def actual_points():
    value = os.environ.get("ONNX_SPLITPOINT_ACTUAL_SPLIT_RATE_POINTS")
    if not value:
        pytest.skip("Explicit completed classification/BN6 cohort paths are required")
    points = list(map(Path, json.loads(value)))
    assert len(points) == 2 and all((point / "performance_import.json").is_file() for point in points)
    return points


def summary(point):
    return read(point / "reports/native_producer_combined_summary.json")["rows"][0]


def rebind_summary(point, value):
    path = point / "reports/native_producer_combined_summary.json"
    write(path, {"rows": [value]})
    receipt = read(point / "performance_import.json")
    receipt["summary"]["sha256"] = native._sha(path)
    write(point / "performance_import.json", receipt)


def copy_point(source, destination):
    destination.mkdir()
    for name in ("PREPARED.json", "DISPATCH_ATTEMPTS.json", "performance_import.json"):
        shutil.copyfile(source / name, destination / name)
    for index in (1, 2, 3):
        directory = f"rep{index:02d}"
        for name in ("VALIDATED.json", "SDK_COMPLETION.json"):
            path = destination / directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / directory / name, path)
        shutil.copytree(source / directory / "download", destination / directory / "download")
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            return value.replace(str(source), str(destination))
        return value
    for path in destination.glob("*.json"):
        write(path, relocate(read(path)))
    for path in destination.glob("rep*/VALIDATED.json"):
        write(path, relocate(read(path)))
    rebind_summary(destination, relocate(summary(source)))
    return destination


@pytest.mark.parametrize("index", [0, 1])
def test_actual_original_cohort_projection_retains_slots_statistics_and_limitations(actual_points, index):
    point = actual_points[index]
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (point / "PREPARED.json", point / "DISPATCH_ATTEMPTS.json",
                           point / "performance_import.json", point / "reports/native_producer_combined_summary.json")}
    row = summary(point)
    original = copy.deepcopy(row)
    fields = rates.report_rate_fields(row, point)
    slots = native._same_run_performance_slots(read(point / "PREPARED.json"))["slots"]
    raw_rates = [rates.project_rate_endpoint(read(Path(slots[n]["report"])), "completed_task") for n in (1, 2, 3)]
    assert fields["completed_task_rate"]["status"] == "available"
    assert fields["completed_task_rate"]["repetition_ids"] == [slots[n]["runtime_instance_id"] for n in (1, 2, 3)]
    assert fields["completed_task_measurement_times_s"] == [item["measurement_times_s"][0] for item in raw_rates]
    assert fields["fps_repetition_samples"] == [slots[n]["fps"] for n in (1, 2, 3)]
    assert fields["completed_task_work_unit_counts"] == [1000, 1000, 1000]
    assert fields["fps_median"] == original["fps_median_reported"]
    assert fields["fps_ci95_low"] == original["fps_ci95_low_reported"]
    assert fields["fps_ci95_high"] == original["fps_ci95_high_reported"]
    old_fields = rates.rate_endpoint_fields(original)
    for key in old_fields:
        if key.startswith(("request_latency_", "p2_", "host_output_")):
            assert fields[key] == old_fields[key]
    observation = collect_native_performance_matrix(point)["observations"][0]
    assert observation["claim_eligible"] is False and observation["performance_contract_eligible"] is False
    assert set(original.get("performance_claim_exclusion_reasons") or []) <= set(observation["performance_claim_exclusion_reasons"])
    assert observation["repetition_status"] == original["repetition_status"]
    assert row == original
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in before.items())
    if index == 1:
        assert len({read(Path(slots[n]["report"]))["completed_task_endpoint_contract_hash"] for n in (1, 2, 3)}) == 3
        assert all(proof["completion_contract_provenance_normalized"] is True for proof in fields["rate_endpoint_operating_point_evidence"])


@pytest.mark.parametrize("mutation", ["raw_hash", "import_hash", "prepared_argv", "duplicate_uuid", "mixed_task",
    "record_time", "record_count", "missing_record", "summary_requested", "summary_attempted", "summary_valid", "reported_statistic", "foreign_source",
    "missing_import", "unclosed_dispatch", "raw_task_complete", "summary_valid_low", "summary_mode", "summary_inflight",
    "summary_boundary", "summary_stage", "summary_postprocess", "summary_task_complete", "record_task_complete_false"])
def test_replay_rejects_mutated_local_authority(actual_points, tmp_path, mutation):
    point = copy_point(actual_points[0], tmp_path / "point")
    row = summary(point)
    assert rates.report_rate_fields(row, point)["completed_task_rate"]["status"] == "available"
    raw_path = Path(row["repetition_records"][0]["operating_point_source_report"])
    if mutation == "raw_hash":
        raw_path.write_bytes(raw_path.read_bytes() + b" ")
    elif mutation == "missing_import":
        (point / "performance_import.json").unlink()
    elif mutation == "unclosed_dispatch":
        dispatch = read(point / "DISPATCH_ATTEMPTS.json")
        dispatch["attempts"][0].pop("completed_at")
        write(point / "DISPATCH_ATTEMPTS.json", dispatch)
    elif mutation == "raw_task_complete":
        raw = read(raw_path); raw["task_complete"] = False; write(raw_path, raw)
        checkpoint = point / "rep01/VALIDATED.json"; validated = read(checkpoint)
        validated["report_sha256"] = native._sha(raw_path); write(checkpoint, validated)
        receipt = read(point / "performance_import.json")
        receipt["source_reports"][0]["sha256"] = native._sha(raw_path)
        write(point / "performance_import.json", receipt)
    elif mutation == "import_hash":
        receipt = read(point / "performance_import.json"); receipt["summary"]["sha256"] = "0" * 64
        write(point / "performance_import.json", receipt)
    elif mutation == "prepared_argv":
        prepared = read(point / "PREPARED.json"); argv = prepared["runs"][0]["argv"]
        argv[argv.index("--warmup") + 1] = "99"; write(point / "PREPARED.json", prepared)
    else:
        record = row["repetition_records"][0]
        if mutation == "duplicate_uuid": record["runtime_instance_id"] = row["repetition_records"][1]["runtime_instance_id"]
        elif mutation == "mixed_task": record["task"] = "detection"
        elif mutation == "record_time": record["makespan_ms"] *= 2
        elif mutation == "record_count": record["completed_work_units"] = 999
        elif mutation == "record_task_complete_false": record["task_complete"] = False
        elif mutation == "missing_record": row["repetition_records"].pop()
        elif mutation == "summary_valid_low": row["repetition_count_valid"] = 1
        elif mutation == "summary_mode": row["deepx_runtime_mode"] = "S1"
        elif mutation == "summary_inflight": row["deepx_max_inflight"] = 1
        elif mutation == "summary_boundary": row["measurement_boundary"] = "foreign_boundary"
        elif mutation == "summary_stage": row["completed_task_stage"] = "foreign_stage"
        elif mutation == "summary_postprocess": row["postprocess_completion_verified"] = False
        elif mutation == "summary_task_complete": row["task_complete"] = False
        elif mutation.startswith("summary_"): row["repetition_count_" + mutation.removeprefix("summary_")] = 4
        elif mutation == "reported_statistic": row["fps_median_reported"] *= 2
        elif mutation == "foreign_source": record["operating_point_source_report"] = str(actual_points[0] / "foreign.json")
        rebind_summary(point, row)
    assert rates.report_rate_fields(row, point)["completed_task_rate"]["status"] == "unavailable"


@pytest.mark.parametrize("mutation", ["declaration_hash", "declaration_symlink", "declaration_foreign_suffix"])
def test_bn6_declaration_fallback_remains_exact(actual_points, tmp_path, mutation):
    point = copy_point(actual_points[1], tmp_path / "point")
    row = summary(point)
    assert rates.report_rate_fields(row, point)["completed_task_rate"]["status"] == "available"
    raw_path = Path(row["repetition_records"][0]["operating_point_source_report"])
    declaration = raw_path.parent / "benchmark_set/output_contracts.json"
    if mutation == "declaration_hash":
        declaration.write_bytes(declaration.read_bytes() + b" ")
    elif mutation == "declaration_symlink":
        foreign = tmp_path / "identical.json"; foreign.write_bytes(declaration.read_bytes())
        declaration.unlink(); declaration.symlink_to(foreign)
    else:
        raw = read(raw_path)
        assert reporter._normal_split_collected_declaration(raw_path, raw["native_command_contract"], "/foreign/output_contracts.json") is None
        return
    assert rates.report_rate_fields(row, point)["completed_task_rate"]["status"] == "unavailable"


def test_split_replay_does_not_apply_to_generic_hailo_or_s1():
    for backend, mode in (("hailo8_to_trt", "A4"), ("hailo10h_to_trt", "A4"), ("deepx_to_trt", "S1"), ("generic", "A4")):
        assert rates._normal_split_a4_cohort_rate_fields({"backend": backend, "deepx_runtime_mode": mode, "repetition_count_valid": 3}, None) is None


@pytest.mark.parametrize("records", [None, [], [None, None, None], [{"operating_point_source_report": "relative/rep01.json"}] * 3])
def test_split_observation_without_normal_authority_is_unavailable(tmp_path, records):
    row = {"backend": "deepx_to_trt", "deepx_runtime_mode": "A4", "repetition_count_valid": 3,
           "repetition_records": records, "fps_median": 12345.0}
    fields = rates.report_rate_fields(row, tmp_path)
    assert fields["completed_task_rate"]["status"] == "unavailable"
    assert fields["completed_task_measurement_times_s"] == []


def test_remote_reporter_mirror_stays_byte_identical():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts/native_producer_final_report.py").read_bytes() == (root / "onnx_splitpoint_tool/resources/remote_scripts/native_producer_final_report.py").read_bytes()
