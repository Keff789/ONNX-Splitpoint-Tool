"""Replay preserved completed pairs; never start a runner or alter source data.

GENERIC_COMPLETION_PAIR_PLAN points to an existing execution-plan.json. The
negative cases change only an external copy of the original Native JSON and
substitute its read at the normal consumer boundary; all verifiers remain real.
"""
import copy
import json
import os
from pathlib import Path

import pytest

from onnx_splitpoint_tool import generic_completion_reporting as reporting


@pytest.fixture(scope="module")
def measured_case():
    source = os.environ.get("GENERIC_COMPLETION_PAIR_PLAN")
    if not source:
        pytest.skip("requires preserved measured Generic/Native pair")
    plan = json.loads(Path(source).read_text())
    completed = []
    for case in plan["rows"]:
        path = Path(case["result_path"])
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        observed = reporting.validate_completion_case(result, case, result_root=path.parent)
        if observed["status"] == "completed":
            completed.append((case, observed))
    assert completed, "fixture must contain an actual completed pair"
    case, observed = completed[0]
    path = Path(case["native_source"])
    if not path.is_absolute():
        path = Path(case["original_run"]) / path
    native_bytes = path.read_bytes()
    native = json.loads(native_bytes)
    assert native.get("backend") in (None, "")
    assert native["native_command_contract"]["backend"] == case["runtime_config"]["identity"]["direction"]
    return case, observed, path, native_bytes


def test_original_missing_raw_backend_uses_verified_command_without_mutation(measured_case):
    case, observed, source, native_bytes = measured_case
    original_case = copy.deepcopy(case)
    generic_path = Path(case["result_path"])
    generic_bytes = generic_path.read_bytes()
    request_path = Path(case["generic_original_quality_request"])
    if not request_path.is_absolute():
        request_path = Path(case["original_run"]) / request_path
    request_bytes = request_path.read_bytes()
    pair = reporting.completion_pair(observed, case)
    assert observed["valid_repetition_count"] == 3
    assert pair["native_generic_completed_endpoint_evidence"]["status"] == "verified", pair
    assert pair["eligible_for_technical_transfer"] is True
    assert pair["native_over_generic_completed_task_ratio"] > 0
    assert pair["quality_transfer_eligibility_checks"]["native_output_endpoint_match"] is False
    assert pair["claim_transfer_eligibility_checks"]["native_output_endpoint_match"] is False
    assert pair["eligible_for_quality_transfer"] is False
    assert pair["eligible_for_claim_transfer"] is False
    assert case == original_case
    assert source.read_bytes() == native_bytes
    assert generic_path.read_bytes() == generic_bytes
    assert request_path.read_bytes() == request_bytes


@pytest.mark.parametrize("field,value", [
    ("backend", "deepx_to_trt"),
    ("backend", " "),
    ("setup_id", "foreign_setup"),
    ("model", "foreign_model"),
    ("comparison_backend", "deepx"),
    ("precision", "uint8_cast_fp16"),
    ("runtime_precision_identity", "uint8_cast_fp16"),
    ("execution_precision", "uint8_cast_fp16"),
    ("source_request_sha256", "0" * 64),
    ("native_split_quality_source_request_sha256", "0" * 64),
    ("native_split_quality_source_run_id", "deepx_to_trt"),
    ("source_run_id", "deepx_to_trt"),
    ("model_id", "foreign_model"),
    ("case_id", "b999"),
    ("boundary", "b999"),
    ("measurement_setup_id", "foreign_setup"),
    ("setup", "foreign_setup"),
    ("producer_backend", "deepx_to_trt"),
    ("contract", "uint8_cast_fp16"),
])
def test_contradictory_raw_assertion_remains_blocked(measured_case, tmp_path, monkeypatch, field, value):
    case, observed, source, native_bytes = measured_case
    changed = json.loads(native_bytes)
    # Keep the primary assertions correct while a second explicit alias lies.
    for primary in ("backend", "model", "case", "setup_id", "precision"):
        changed.setdefault(primary, changed["native_command_contract"][primary])
    changed[field] = value
    copied_source = tmp_path / "contradictory-native-result.json"
    copied_source.write_text(json.dumps(changed))
    real_read = reporting._read_result

    def read_external_copy(path):
        return real_read(copied_source if Path(path) == source else path)

    monkeypatch.setattr(reporting, "_read_result", read_external_copy)
    pair = reporting.completion_pair(observed, case)
    evidence = pair["native_generic_completed_endpoint_evidence"]
    assert evidence["status"] == "unverified", (field, pair)
    assert evidence.get("reason")
    assert pair["eligible_for_technical_transfer"] is False
    assert pair["native_over_generic_completed_task_ratio"] is None
    assert pair["eligible_for_quality_transfer"] is False
    assert pair["eligible_for_claim_transfer"] is False
    assert source.read_bytes() == native_bytes


@pytest.mark.parametrize("mutation", [
    "unknown_mode", "wrong_relation", "fast_count", "fast_content",
    "fast_endpoint", "dump_repetition", "persisted_output_binding",
])
def test_real_fast_oracle_or_output_binding_damage_is_not_a_pair(
    measured_case, tmp_path, monkeypatch, mutation,
):
    case, observed, source, native_bytes = measured_case
    changed = json.loads(native_bytes)
    attestation = changed["completed_task_endpoint_attestation"]
    if mutation == "unknown_mode":
        changed["completed_task_completion_mode"] = "unknown_completion_mode"
    elif mutation == "wrong_relation":
        attestation["observation_relation"] = "same_hotloop_sentinel"
    elif mutation == "fast_count":
        attestation["completed_work_units"] = 1001
    elif mutation == "fast_content":
        attestation["fast_content_sha256"] = "0" * 64
    elif mutation == "fast_endpoint":
        attestation["comparison_endpoint_contract"]["output_endpoint_id"] = "foreign_endpoint"
    elif mutation == "dump_repetition":
        changed["semantic_evidence_repetition_index"] = 2
    elif mutation == "persisted_output_binding":
        changed["completed_task_result_artifact_file_sha256"] = "0" * 64
    copied_source = tmp_path / "contradictory-native-result.json"
    copied_source.write_text(json.dumps(changed))
    real_read = reporting._read_result
    monkeypatch.setattr(reporting, "_read_result", lambda path: real_read(
        copied_source if Path(path) == source else path))
    pair = reporting.completion_pair(observed, case)
    assert pair["native_generic_completed_endpoint_evidence"]["status"] == "unverified", pair
    assert pair["eligible_for_technical_transfer"] is False
    assert pair["native_over_generic_completed_task_ratio"] is None
    assert pair["eligible_for_quality_transfer"] is False
    assert pair["eligible_for_claim_transfer"] is False
    assert source.read_bytes() == native_bytes
