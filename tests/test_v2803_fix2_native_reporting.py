"""Replay the observed Hailo8 endpoint and 17 Split counter failures locally.

Fixtures contain unchanged collected manifests/results and a documented
field-only repetition projection. No model, runtime or tensor acceptance is
performed here; missing payloads must remain unverified.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from onnx_splitpoint_tool.workflow.runner import _native_concise_summary_v60w

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/v2803_fix2_native_reporting"
FALLBACKS = (
    "native_fifo_outputs/native_fifo_output_manifest.json",
    "native_fifo_outputs/native_fifo_outputs_manifest.json",
    "native_outputs/native_outputs_manifest.json",
)
CASES = [("mobilenet_v3_large", "b135"), ("regnet_x_1_6gf", "b132"), ("resnet50", "b119")]


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("fix2_native_report", ROOT / "scripts/native_producer_final_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _job(tmp_path, model="mobilenet_v3_large", case="b135"):
    relative = Path(f"native_producers/hailo8/{model}/benchmark_set/native_pipeline/{case}/hailo_to_trt/float32_layout_fp16")
    path = tmp_path / relative
    shutil.copytree(FIXTURE / "original" / relative, path)
    result_path = path / "native_fifo_results.json"
    return result_path, json.loads(result_path.read_text())


def test_fixture_bytes_and_remote_script_mirror():
    provenance = json.loads((FIXTURE / "PROVENANCE.json").read_text())
    for entry in provenance["original_members"] + provenance["derived_members"]:
        data = (FIXTURE / entry["path"]).read_bytes()
        assert len(data) == entry["size_bytes"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert (ROOT / "scripts/native_producer_final_report.py").read_bytes() == (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts/native_producer_final_report.py"
    ).read_bytes()


@pytest.mark.parametrize("model,case", CASES)
def test_actual_classification_manifests_restore_endpoint_without_inventing_quality(tmp_path, report, model, case):
    result_path, result = _job(tmp_path, model, case)
    before = {path: path.read_bytes() for path in result_path.parent.rglob("*.json")}
    projected = report._split_output_contract(result_path, result, fallback_relpaths=FALLBACKS)
    assert projected["manifest_expected_role"] == "model_outputs"
    assert projected["manifest_resolution_status"] == "manifest_verified"
    assert projected["output_contract_manifest_status"] == "explicit_complete"
    assert projected["endpoint_contract_complete"] is True
    assert projected["endpoint_contract_hash"] == result["endpoint_contract_hash"]
    endpoint = f"classification:classification_logits:{result['endpoint_contract_hash']}"
    assert report._explicit_output_endpoint(projected) == endpoint
    assert report._comparison_output_endpoint(projected) == endpoint
    boundary, resolution = report._resolve_split_semantic_manifest(
        result_path, [result], kind="boundary",
        fallback_relpaths=("native_fifo_boundary/native_fifo_boundary_manifest.json",),
    )
    assert boundary.is_file()
    assert resolution["manifest_resolution_status"] == "manifest_verified"
    # This compact fixture deliberately carries no raw tensor payloads.
    assert projected["native_split_semantic_binding_valid"] is False
    assert projected.get("quality_claim_result_verified") is not True
    assert all(path.read_bytes() == data for path, data in before.items())


def test_actual_three_jobs_reach_report_endpoint_columns_without_quality_acceptance(tmp_path, report):
    for model, case in CASES:
        _job(tmp_path, model, case)
    rows = report._rows_from_native_fifo_roots([tmp_path / "native_producers/hailo8"])
    assert len(rows) == 3
    projected = report._apply_comparison_claim_gates(rows)
    for row in projected:
        expected = f"classification:classification_logits:{row['endpoint_contract_hash']}"
        assert row["output_endpoint_id"] == expected
        assert row["physical_output_endpoint_id"] == expected
        assert row["comparison_output_endpoint_id"] == expected
        assert row["native_split_semantic_binding_valid"] is False
        assert row["performance_claim_eligible"] is False


@pytest.mark.parametrize("mutation,expected", [
    ("detection", "manifest_endpoint_role_conflict"),
    ("raw_role", "manifest_endpoint_role_conflict"),
    ("completed_role", "manifest_endpoint_role_conflict"),
    ("wrong_case", "manifest_job_identity_conflict:case"),
    ("wrong_setup", "manifest_job_identity_conflict:setup_id"),
    ("wrong_precision", "manifest_job_identity_conflict:precision"),
    ("command_changed", "manifest_command_contract_invalid"),
    ("command_missing", "manifest_endpoint_role_conflict"),
    ("manifest_changed", "semantic_output_manifest_sha256_mismatch"),
    ("manifest_missing", "missing"),
    ("cross_role_path", "manifest_endpoint_role_conflict"),
    ("traversal", "manifest_path_traversal"),
])
def test_classification_admission_keeps_job_hash_and_role_guards(tmp_path, report, mutation, expected):
    result_path, result = _job(tmp_path)
    manifest = result_path.parent / "native_fifo_outputs/native_fifo_outputs_manifest.json"
    if mutation == "detection":
        result["task"] = "detection"
    elif mutation == "raw_role":
        result["measurement_endpoint"] = "raw_model_outputs"
    elif mutation == "completed_role":
        result["measurement_endpoint"] = "completed_task"
    elif mutation == "wrong_case":
        result["case"] = "b999"
    elif mutation == "wrong_setup":
        result["setup_id"] = "different_setup"
    elif mutation == "wrong_precision":
        result["precision"] = "uint8_dequant_fp16"
    elif mutation == "command_changed":
        result["native_command_contract"]["model"] = "different_model"
    elif mutation == "command_missing":
        result.pop("native_command_contract")
    elif mutation == "manifest_changed":
        manifest.write_bytes(manifest.read_bytes() + b"\n")
    elif mutation == "manifest_missing":
        manifest.unlink()
    elif mutation == "cross_role_path":
        wrong = result_path.parent / "endpoints/completed_task/native_fifo_outputs/native_fifo_outputs_manifest.json"
        wrong.parent.mkdir(parents=True)
        shutil.copy2(manifest, wrong)
        result["native_output_manifest"] = str(wrong)
    elif mutation == "traversal":
        result["native_output_manifest"] = "../unrelated/native_fifo_outputs_manifest.json"
    path, status = report._resolve_split_semantic_manifest(result_path, [result], kind="output", fallback_relpaths=FALLBACKS)
    assert status["manifest_resolution_status"] == expected
    projected = report._split_output_contract(result_path, result, fallback_relpaths=FALLBACKS)
    assert projected["endpoint_contract_complete"] is False
    assert report._explicit_output_endpoint(projected) == ""


def _concise(tmp_path, rows):
    reports = tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "native_producer_summary.json").write_text(json.dumps({"rows": rows}))
    return _native_concise_summary_v60w(reports)[1]


def test_actual_63_rows_count_all_177_successful_repeats_and_four_nonstarts(tmp_path):
    summary = json.loads((FIXTURE / "repetition_projection.json").read_text())
    original = copy.deepcopy(summary)
    rows = _concise(tmp_path, summary["rows"])
    assert len(rows) == 63
    successful = [row for row in rows if row["runtime_status"] == "ok"]
    assert len(successful) == 59
    assert sum(row["runtime_repetition_count_successful"] for row in rows) == 177
    assert all(row["runtime_repetition_count_successful"] == 3 for row in successful)
    split = [row for row in successful if not row["backend"].startswith("native_full_")]
    assert len(split) == 17
    assert all(row["runtime_repetition_count_successful"] == 3 for row in split)
    nonstarts = [row for row in rows if row["runtime_status"] != "ok"]
    assert len(nonstarts) == 4
    assert all(row["runtime_repetition_count_successful"] == row["repetition_count_attempted"] == 0 for row in nonstarts)
    assert summary == original


@pytest.mark.parametrize("change", [
    {"runtime_success": False}, {"ok": False}, {"result_ok": False},
    {"runtime_success": "true"}, {"runtime_success": 1}, {"ok": "true"},
    {"status": "failed"}, {"status": "incomplete"}, {"status": "blocked"},
    {"timed_out": True}, {"prerequisite_status": "blocked"}, {"returncode": 1},
    {"completed_work_units": 0}, {"completed_frames": 999},
    {"completed_work_units": "1000"}, {"requested_frames": 1001},
    {"runtime_success": True, "completed_work_units": 0},
    {"runtime_success": True, "status": "unknown"},
    {"runtime_success": True, "status": "blocked"},
])
def test_counter_never_promotes_explicit_failure_or_incomplete_repeat(tmp_path, change):
    summary = json.loads((FIXTURE / "repetition_projection.json").read_text())
    row = next(row for row in summary["rows"] if row.get("ok") is True and row["backend"] == "hailo8_to_trt")
    row["repetition_records"][0].update(change)
    assert _concise(tmp_path, [row])[0]["runtime_repetition_count_successful"] == 2


@pytest.mark.parametrize("explicit", [None, True])
def test_counter_requires_observed_completion_not_summary_counts(tmp_path, explicit):
    summary = json.loads((FIXTURE / "repetition_projection.json").read_text())
    row = next(row for row in summary["rows"] if row.get("ok") is True)
    first = row["repetition_records"][0]
    first["runtime_success"] = explicit
    for field in ("completed_work_units", "completed_frames", "frames_completed"):
        first.pop(field, None)
    assert row["repetition_count_valid"] == 3
    assert _concise(tmp_path, [row])[0]["runtime_repetition_count_successful"] == 2


def test_blocked_row_cannot_gain_repeat_count_from_stray_records(tmp_path):
    summary = json.loads((FIXTURE / "repetition_projection.json").read_text())
    row = next(row for row in summary["rows"] if row.get("ok") is True)
    row.update(ok=False, runtime_success=False, status="blocked", prerequisite_status="blocked", repetition_count_attempted=0)
    assert _concise(tmp_path, [row])[0]["runtime_repetition_count_successful"] == 0
