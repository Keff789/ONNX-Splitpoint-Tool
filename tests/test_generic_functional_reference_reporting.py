"""Functional Native controls stay separate from historical Native science.

These are admission/projection regressions.  The source verifier is replaced
only in the projection fixture; no synthetic output is a hardware observation.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from onnx_splitpoint_tool import generic_completion_reporting as reporting
from tests.test_generic_completion_reporting import case, result, write_result
from tests.test_v272_detection_completion_runtime import _decoded_outputs, _runtime


GENERIC_ONLY_MARKERS = [
    {"scope_mode": "generic_only_completed_task"},
    {"measurement_mode": "generic_only_completed_task"},
    {"admission_mode": "generic_only"},
    {"generic_artifact_binding": {}},
    {"functional_reference": {"kind": "serial_compiled_subnets"}},
]


@pytest.mark.parametrize("marker", GENERIC_ONLY_MARKERS)
@pytest.mark.parametrize("location", ["case", "runtime_config", "observation"])
def test_generic_only_never_becomes_a_paired_row(marker, location):
    bound_case = case()
    observation = reporting.validate_completion_case(result(bound_case), bound_case)
    if location == "case":
        bound_case.update(marker)
    elif location == "runtime_config":
        bound_case["runtime_config"] = {**copy.deepcopy(bound_case), **marker}
    else:
        observation.update(marker)
    with pytest.raises(ValueError, match="generic_only_cannot_enter_paired"):
        reporting.completion_pair(observation, bound_case)


@pytest.mark.parametrize("marker", GENERIC_ONLY_MARKERS)
@pytest.mark.parametrize("location", ["plan", "case", "runtime_config", "result"])
def test_paired_writer_rejects_generic_only_before_creating_reports(tmp_path, marker, location):
    bound_case = case()
    path = tmp_path / "results.json"
    bound_case["result_path"] = str(path)
    plan = {"rows": [bound_case]}
    if location == "plan":
        plan.update(marker)
    elif location == "case":
        bound_case.update(marker)
    elif location == "runtime_config":
        bound_case["runtime_config"] = {**copy.deepcopy(bound_case), **marker}
    else:
        payload = result(bound_case)
        write_result(path, payload)
        path.write_text(json.dumps({**payload, **marker}))
    output = tmp_path / "reports"
    with pytest.raises(ValueError, match="generic_only_cannot_enter_paired"):
        reporting.write_completion_reports(plan, tmp_path / "results", output)
    assert not output.exists()


def test_unreadable_result_remains_a_blocked_paired_case(tmp_path):
    bound_case = case()
    path = tmp_path / "results.json"
    path.write_text("{invalid")
    bound_case["result_path"] = str(path)
    summary = reporting.write_completion_reports({"rows": [bound_case]}, tmp_path, tmp_path / "reports")
    assert summary["status_counts"]["blocked"] == 1
    assert summary["technical_pairs"] == 0


@pytest.mark.parametrize("old_endpoint_check", [False, True])
def test_explicit_functional_source_always_requires_independent_proof(monkeypatch, old_endpoint_check):
    bound_case = case()
    for tier in ("technical", "quality", "claim"):
        bound_case["historical_pair"][tier + "_transfer_eligibility_checks"]["native_output_endpoint_match"] = old_endpoint_check
    # A status flag in the descriptor is not an admission proof.
    bound_case["functional_reference_source"] = {"path": "/missing", "sha256": "0" * 64, "status": "verified"}
    calls = []
    def unverified(observation, supplied_case):
        calls.append(supplied_case)
        return {"status": "unverified", "reason": "source bytes do not verify"}
    monkeypatch.setattr(reporting, "_completed_endpoint_pair_evidence", unverified)
    observation = reporting.validate_completion_case(result(bound_case), bound_case)
    pair = reporting.completion_pair(observation, bound_case)
    assert len(calls) == 1
    assert pair["native_over_generic_completed_task_ratio"] is None
    assert not pair["eligible_for_technical_transfer"]
    assert "native_functional_reference_verified" in pair["technical_exclusion_reasons"]


def test_functional_control_preserves_historical_rates_quality_and_contract(monkeypatch):
    bound_case = case()
    for tier in ("technical", "quality", "claim"):
        bound_case["historical_pair"][tier + "_transfer_eligibility_checks"]["native_output_endpoint_match"] = False
    bound_case["functional_reference_source"] = {"path": "/fresh/functional.json", "sha256": "a" * 64}
    original_case = copy.deepcopy(bound_case)
    monkeypatch.setattr(reporting, "_completed_endpoint_pair_evidence", lambda *_: {
        "status": "verified", "native_performance_contract_relation": "different_implementation_same_comparison_endpoint",
    })
    observation = reporting.validate_completion_case(result(bound_case), bound_case)
    pair = reporting.completion_pair(observation, bound_case)
    assert pair["eligible_for_technical_transfer"]
    assert pair["native_completed_task_fps"] == 50
    assert pair["native_over_generic_completed_task_ratio"] == 1
    assert pair["native_scientific_performance_source"] == bound_case["native_source"]
    assert pair["native_functional_reference_source"] == bound_case["functional_reference_source"]
    assert pair["native_functional_reference_role"] == "functional_control_only"
    assert pair["quality_transfer_eligibility_checks"]["native_output_endpoint_match"] is False
    assert pair["claim_transfer_eligibility_checks"]["native_output_endpoint_match"] is False
    assert not pair["eligible_for_quality_transfer"] and not pair["eligible_for_claim_transfer"]
    assert pair["native_task_quality_status"] == bound_case["historical_pair"]["native_task_quality_status"]
    assert bound_case == original_case


@pytest.mark.parametrize("fault", ["historical_gate", "rate_source", "count_time", "boundary", "identity", "environment", "partial"])
def test_functional_proof_cannot_release_an_existing_paired_guard(monkeypatch, fault):
    bound_case = case()
    bound_case["functional_reference_source"] = {"path": "/functional.json", "sha256": "a" * 64}
    observation = reporting.validate_completion_case(result(bound_case), bound_case)
    monkeypatch.setattr(reporting, "_completed_endpoint_pair_evidence", lambda *_: {"status": "verified"})
    if fault == "historical_gate": bound_case["historical_pair"]["technical_transfer_eligibility_checks"]["native_runtime_ok"] = False
    elif fault == "rate_source": bound_case["historical_pair"]["native_measured_fps_source"] = "functional_control"
    elif fault == "count_time": bound_case["native_measurement_boundary"]["observed_completed_counts"] = [1]
    elif fault == "boundary": bound_case["measurement_boundary_comparison"]["checks"]["input_sequence_matches"] = False
    elif fault == "identity": bound_case["historical_pair"]["case_id"] = "b999"
    elif fault == "environment": bound_case["runtime_environment_comparison"] = {"status": "conflict"}
    else: observation["status"] = "partial"
    pair = reporting.completion_pair(observation, bound_case)
    assert pair["native_generic_completed_endpoint_evidence"]["status"] == "verified"
    assert not pair["eligible_for_technical_transfer"]
    assert pair["native_over_generic_completed_task_ratio"] is None


@pytest.fixture
def functional_projection(tmp_path, monkeypatch):
    from onnx_splitpoint_tool.workflow import native_functional_reference

    tensors = _decoded_outputs()
    contract = _runtime(tensors).execution_contract
    bound_case = case(setup="orin_nx_hailo10_01", boundary="b038",
                      endpoint=contract["source_endpoint"]["output_endpoint_id"])
    bound_case["identity"].update(model_id="yolo26s", direction="hailo10h_to_trt", comparison_backend="hailo10h")
    bound_case["historical_pair"].update(bound_case["identity"])
    bound_case["task"] = "detection"
    bound_case["original_run"] = str(tmp_path / "historical-run")
    # H10's historical science source can be outside the original run.  The
    # separate real source verifier, rather than an invented inside-run flag,
    # owns that historical origin and the new functional source's provenance.
    bound_case["native_source"] = str(tmp_path / "external-historical-native.json")
    bound_case["functional_reference_source"] = {"path": str(tmp_path / "functional.json"), "sha256": "a" * 64}
    boundary = {"completion_output_endpoint_id": contract["comparison_endpoint_contract"]["output_endpoint_id"]}
    bound_case["native_measurement_boundary"].update(boundary)
    bound_case["measurement_boundary"] = boundary
    entry = {"name": "images", "remote_path": "/original/prepared.bin", "sha256": "b" * 64,
             "shape": [64, 64, 3], "dtype": "uint8"}
    bound_case["input_binding"] = {"entries": [entry], "source": bound_case["native_source"],
                                   "prepared_input_contract": {"format": "raw_runtime_tensor", "layout": "HWC"}}
    bound_case["expected_runtime_artifacts"] = {
        "part1_runtime": {"path": "/compiled/part1.hef", "sha256": "c" * 64},
        "engine": {"path": "/compiled/part2.engine", "sha256": "d" * 64},
    }
    limits = {"iou_threshold": .5, "min_reference_match_ratio": .95, "min_mean_matched_iou": .98}
    bound_case["original_run_cfg"] = {"quality_gate_json": json.dumps({
        "native_contract": {"detection_self_reference": limits}})}
    reference_outputs = []
    for name, tensor in tensors.items():
        path = tmp_path / (name + ".bin")
        tensor.tofile(path)
        reference_outputs.append({"name": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "shape": list(tensor.shape), "dtype": str(tensor.dtype), "file": path.name})
    reference = {"output_manifest": str(tmp_path / "outputs.json"), "output_manifest_sha256": "e" * 64,
                 "outputs": reference_outputs}
    runtime_case = {**copy.deepcopy(bound_case), "completion_contract": contract, "functional_reference": reference}
    runtime_case["native_source"] = str(tmp_path / "new-functional-native.json")
    runtime_case["native_command_contract"] = {"backend": "hailo10h_to_trt", "runtime_options": {"quantized_inputs": True}}
    overlay = {"status": "verified", "runtime_case": runtime_case, "completion_contract": contract,
               "functional_reference": reference, "evidence": {"source_role": "functional_only"},
               "version_relation": {"historical_native_full_contract_equal": False,
                                    "comparison_endpoint_equal": True}}
    config = {
        "identity": copy.deepcopy(bound_case["identity"]), "task": "detection",
        "original_run": bound_case["original_run"], "completion_contract": copy.deepcopy(contract),
        "original_run_cfg": {**copy.deepcopy(bound_case["original_run_cfg"]), "streaming_impl": "generic"},
        "measurement_boundary": "prepared_input_before_p1_to_last_completed_task",
        "measurement_contract": boundary, "expected_runtime_artifacts": copy.deepcopy(bound_case["expected_runtime_artifacts"]),
        "input_binding": {**copy.deepcopy(bound_case["input_binding"]), "prepared_input": {
            "path": entry["remote_path"], "sha256": entry["sha256"], "shape": entry["shape"], "dtype": entry["dtype"], "format": "raw"},
            "runtime_input_format": "uint8", "runtime_input_format_source": runtime_case["native_source"] + "#native_command_contract.runtime_options"},
        "functional_reference": {"input_sha256": entry["sha256"], "source_manifest": reference["output_manifest"],
            "source_manifest_sha256": reference["output_manifest_sha256"],
            "outputs": {descriptor["name"]: {key: descriptor[key] for key in ("sha256", "shape", "dtype")}
                        | {"path": "/staged/reference/" + descriptor["file"], "format": "raw"}
                        for descriptor in reference_outputs},
            "tolerances": {"match_iou_threshold": .5, "min_match_ratio": .95, "min_mean_iou": .98}},
    }
    monkeypatch.setattr(native_functional_reference, "verify_native_functional_reference", lambda _: copy.deepcopy(overlay))
    return bound_case, config, overlay


def _persist_generic_detection_case(tmp_path, bound_case, config):
    """Write portable validator fixtures, never an accelerator measurement."""
    tensors = _decoded_outputs()
    completed = _runtime(tensors).process(tensors)
    payload = result(config)
    for repetition in payload["repetitions"]:
        index = repetition["repetition_index"]
        samples = {}
        for name, tensor in tensors.items():
            relative = f"repeat_{index}/{name}.npy"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, tensor, allow_pickle=False)
            samples[name] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        repetition["raw_output_samples"] = samples
        repetition["last_completed_result"] = copy.deepcopy(completed)
        relative = f"repeat_{index}/completed_task.json"
        path = tmp_path / relative
        path.write_text(json.dumps(completed))
        repetition["completed_result_artifact"] = {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = tmp_path / "generic_completion_results.json"
    path.write_text(json.dumps(payload))
    bound_case["runtime_config"] = config
    bound_case["result_path"] = str(path)
    observed = reporting.validate_completion_case(payload, bound_case, result_root=tmp_path)
    observed["result_path"] = str(path)
    assert observed["status"] == "completed", observed
    return observed


def test_actual_generic_projection_uses_new_functional_contract_before_historical_inside_check(
        tmp_path, functional_projection):
    bound_case, config, _ = functional_projection
    observed = _persist_generic_detection_case(tmp_path, bound_case, config)
    before = copy.deepcopy(bound_case)
    evidence = reporting._completed_endpoint_pair_evidence(observed, bound_case)
    assert evidence["status"] == "verified", evidence
    assert evidence["native_scientific_performance_source"] == bound_case["native_source"]
    assert evidence["generic_valid_repetitions"] == 3
    assert bound_case == before


@pytest.mark.parametrize("fault", ["rate", "count", "repetition", "output_file"])
def test_functional_source_does_not_release_actual_generic_repetition_proof(tmp_path, functional_projection, fault):
    bound_case, config, _ = functional_projection
    observed = _persist_generic_detection_case(tmp_path, bound_case, config)
    if fault == "rate": observed["completed_task_fps"] += 1
    elif fault == "count": observed["completed_task_count"] -= 1
    elif fault == "repetition": observed["repetitions"] = observed["repetitions"][:2]
    else:
        path = tmp_path / observed["repetitions"][0]["raw_output_samples"]["detections"]["path"]
        path.write_bytes(b"modified output")
    evidence = reporting._completed_endpoint_pair_evidence(observed, bound_case)
    assert evidence["status"] == "unverified"
    assert "projection_mismatch" in evidence["reason"]


def test_verified_functional_reference_projection_names_historical_science_separately(functional_projection):
    bound_case, config, overlay = functional_projection
    before = copy.deepcopy(bound_case)
    evidence = reporting._functional_reference_pair_evidence(bound_case, config, Path("/generic.json"), {})
    assert evidence["status"] == "verified"
    assert evidence["native_scientific_performance_source"] == bound_case["native_source"]
    assert evidence["native_functional_reference_source"] == bound_case["functional_reference_source"]
    assert evidence["native_performance_contract_relation"] == overlay["version_relation"]
    assert evidence["generic_valid_repetitions"] == 3
    assert bound_case == before


@pytest.mark.parametrize("fault", [
    "completion_contract", "completion_endpoint", "measurement_contract", "source", "input_binding",
    "prepared_input", "prepared_format", "prepared_layout", "runtime_input_format", "artifact", "manifest", "manifest_hash", "output_hash", "output_name",
    "output_dtype", "output_shape", "output_format", "tolerance",
])
def test_generic_projection_must_use_the_verified_functional_source(functional_projection, fault):
    bound_case, config, _ = functional_projection
    outputs = config["functional_reference"]["outputs"]
    descriptor = next(iter(outputs.values()))
    if fault == "completion_contract": config["completion_contract"] = bound_case["completion_contract"]
    elif fault == "completion_endpoint": bound_case["native_measurement_boundary"]["completion_output_endpoint_id"] = "foreign"
    elif fault == "measurement_contract": config["measurement_contract"] = {**config["measurement_contract"], "extra": True}
    elif fault == "source": config["functional_reference_source"] = {"path": "/foreign", "sha256": "f" * 64}
    elif fault == "input_binding": config["input_binding"]["source"] = "/foreign"
    elif fault == "prepared_input": config["input_binding"]["prepared_input"]["sha256"] = "f" * 64
    elif fault == "prepared_format": config["input_binding"]["prepared_input"]["format"] = "npy"
    elif fault == "prepared_layout": config["input_binding"]["prepared_input"]["layout"] = "CHW"
    elif fault == "runtime_input_format": config["input_binding"]["runtime_input_format"] = "float32"
    elif fault == "artifact": config["expected_runtime_artifacts"]["engine"]["sha256"] = "f" * 64
    elif fault == "manifest": config["functional_reference"]["source_manifest"] = "/foreign"
    elif fault == "manifest_hash": config["functional_reference"]["source_manifest_sha256"] = "f" * 64
    elif fault == "output_hash": descriptor["sha256"] = "f" * 64
    elif fault == "output_name": outputs["foreign"] = outputs.pop(next(iter(outputs)))
    elif fault == "output_dtype": descriptor["dtype"] = "float64"
    elif fault == "output_shape": descriptor["shape"] = [1, 6]
    elif fault == "output_format": descriptor["format"] = "npy"
    else: config["functional_reference"]["tolerances"]["min_match_ratio"] = 0
    with pytest.raises(ValueError, match="native_generic_functional_"):
        reporting._functional_reference_pair_evidence(bound_case, config, Path("/generic.json"), {})
